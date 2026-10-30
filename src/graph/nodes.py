"""Node implementations for the coach graph.

Every node has the signature ``node(deps, state) -> dict`` and returns only the
state keys it changes. ``deps`` is bound at build time with ``functools.partial``
so the state stays JSON-serializable for the checkpointer.

The bodies intentionally mirror ``Orchestrator.generate_plan`` step-for-step so
the two backends are directly comparable; terminal-state logic is delegated to
``src.core.plan_finalization`` so both share one deterministic implementation.
"""

import logging

from langgraph.config import get_stream_writer
from langgraph.types import interrupt

from src.core.goal_contract import GoalConsistencyError, plan_goal_matches
from src.core.plan_finalization import (
    REVIEW_PROHIBITED_ACTIONS,
    build_review_pending_payload,
    check_with_goal_issue,
    create_review_artifact,
    emergency_signals_for,
    emergency_signals_in,
    emergency_stop_payload,
    finalize_result,
    generation_failed_payload,
    long_term_user_key,
    normalize_plan,
    persist_if_safe,
    prune_conflicts_with_recheck,
    safe_cached_result,
    safety_blocked_payload,
    summarize_plan_for_context,
)
from src.core.plan_explanation import build_explanation
from src.hitl.review_resolution import APPROVED, REJECTED, VALID_DECISIONS
from src.llm.prompts.advice import build_advice_messages

logger = logging.getLogger(__name__)


def _emit(event: str, data) -> None:
    """Forward a progress event to the stream writer.

    Under ``graph.stream(..., stream_mode=["updates","custom"])`` this reaches the
    client; under a plain ``invoke`` LangGraph supplies a no-op writer, so the same
    node code works for both the sync and the SSE endpoints.
    """
    try:
        get_stream_writer()({"event": event, "data": data})
    except Exception:  # streaming is best-effort; never break the pipeline for it
        logger.debug("stream writer unavailable; dropping event %s", event)


# ---------------------------------------------------------------------------
# Entry / cache
# ---------------------------------------------------------------------------

def ingest_node(deps, state: dict) -> dict:
    """Validate the goal, default the query, and hydrate conversation context."""
    from src.core.goal_contract import validate_requested_goal
    from src.core.training_history import build_training_context

    profile = state.get("profile") or {}
    session_id = state.get("session_id")
    query = state.get("query") or f"为{profile.get('goal', '')}目标生成训练计划"
    expected_goal = validate_requested_goal(profile.get("goal"))

    # 急症信号在这里一次定下来，供路由、交付与持久化三处共用。
    # 放在 ingest（而不是各消费点各判一次）是因为它没有依赖、不需要 LLM，
    # 而下游每一处漏判的代价都不一样大：路由漏了白等一分钟，交付漏了直接
    # 发出训练计划，持久化漏了会把计划缓存起来供下次命中。
    injuries = profile.get("injuries") or []
    if isinstance(injuries, str):
        injuries = [injuries]
    emergency_signals = (emergency_signals_in(query)
                         or emergency_signals_in(" ".join(map(str, injuries))))

    _emit("stage", "[分析] 正在分析你的情况...")

    conv_context = ""
    plan_context = ""
    if session_id:
        user_turn_preview = query[:200] if query else "生成训练计划"
        deps.conversation.add_turn(session_id, "user", user_turn_preview)
        conv_context = deps.conversation.build_context_for_prompt(session_id, query or "")
        plan_context = deps.conversation.get_plan_state(session_id) or ""

    # 训练历史与手写编排器共用 build_training_context()，两后端因此不会漂移
    training_context = build_training_context(state.get("athlete_key"))

    # 跨会话画像记忆。此前这一侧只写不读（persist_if_safe 会写，却没人读回来），
    # 于是状态图缺少命令式那一侧的"这个用户之前来过"上下文。
    # 身份键与写入侧共用 long_term_user_key()，否则读了也读不到自己写的那份。
    long_term_context = ""
    try:
        long_term_key = long_term_user_key(profile, state.get("athlete_key"))
        long_term_context = deps.long_term.build_context_for_prompt(long_term_key) or ""
    except Exception as exc:  # noqa: BLE001 - 记忆是增强，读不到不该阻断生成
        logger.debug("[graph.ingest] 长期记忆读取跳过：%s", exc)

    return {
        "query": query,
        "expected_goal": expected_goal,
        "conv_context": conv_context,
        "plan_context": plan_context,
        "training_context": training_context,
        "long_term_context": long_term_context,
        "emergency_signals": emergency_signals,
        "rewrite_count": 0,
        "checks": [],
        "provider_degraded": False,
        "cache_hit": None,
    }


def check_cache_node(deps, state: dict) -> dict:
    """Return a safe cached plan when there is no active session."""
    if state.get("session_id"):
        return {"cache_hit": None}
    cached = deps.cache.get(state.get("profile"), state.get("query"))
    return {"cache_hit": safe_cached_result(cached, expected_goal=state.get("expected_goal"),
                                            profile=state.get("profile") or {})}


def deliver_cached_node(deps, state: dict) -> dict:
    """Deliver a cache hit through the same terminal payload shape.

    缓存对象本身不带 ``delivery_status``（它存的是计划内容，不是交付决定）。
    直接原样返回的话，前端拿到的是没有该字段的载荷，而它的写法是
    ``if (data?.delivery_status !== 'safe_delivered') return;``——
    缓存命中的计划有内容，用户却看到一片空白。交付状态在交付这一步补，
    不要写进缓存，否则缓存里会存着一份"当时的交付决定"。
    """
    cached = dict(state.get("cache_hit") or {})
    cached["delivery_status"] = "safe_delivered"
    return {"final_payload": cached}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def plan_node(deps, state: dict) -> dict:
    _emit("stage", "[分析] 正在分析你的情况...")
    _emit_advice(deps, state)

    _emit("stage", "[规划] Planner 正在拆解任务...")
    plan = deps.planner.plan(
        state.get("query", ""), state.get("profile") or {},
        conv_context=state.get("conv_context", ""),
        plan_context=state.get("plan_context", ""),
        training_context=state.get("training_context", ""),
        long_term_context=state.get("long_term_context", ""),
    )
    return {"plan": plan}


def _emit_advice(deps, state: dict) -> None:
    """生成并流式输出教练开场白，然后接上规划阶段。

    放在 ``plan`` 节点开头而不是 ``ingest``：``ingest`` 之后紧跟缓存检查，
    把开场白放在那里会让**每一次缓存命中都白付一次模型调用**。放在这里，
    只有真正要生成计划的请求才付这笔钱——与命令式编排的次序一致。

    失败一律吞掉：开场白是暖场，不是交付内容。模型不可用、替身没提供 llm、
    流式返回结构异常，都不该把一个本来能出计划的请求带崩。
    """
    llm = getattr(getattr(deps, "writer", None), "llm", None)
    if llm is None or not hasattr(llm, "chat_stream"):
        return
    profile = state.get("profile") or {}
    try:
        stream = llm.chat_stream(
            build_advice_messages(profile, state.get("query", ""),
                                  long_term_context=state.get("long_term_context", ""),
                                  plan_context=state.get("plan_context", "")),
            temperature=0.5,
        )
        text = ""
        for chunk in stream:
            text += chunk
            _emit("advice_chunk", chunk)
        _emit("advice_done", text)
    except Exception as exc:  # noqa: BLE001 - 暖场失败不阻断主流程
        logger.debug("[graph.plan] 开场白生成跳过：%s", exc)


def retrieve_node(deps, state: dict) -> dict:
    _emit("stage", "[检索] Retriever 正在检索动作库...")
    retrieved = deps.retriever.retrieve(state.get("plan") or {})
    return {"retrieved": retrieved}


def write_node(deps, state: dict) -> dict:
    _emit("stage", "[生成] Writer 正在生成训练计划...")
    profile = state.get("profile") or {}
    plan_config = (state.get("plan") or {}).get("skill_config", {})
    provider_degraded = state.get("provider_degraded", False)
    result = {}
    raw_text = ""
    for event, data in deps.writer.write_plan_stream(
        state.get("retrieved") or {}, profile, plan_config,
        plan_context=state.get("plan_context", ""), user_query=state.get("query", ""),
        training_context=state.get("training_context", ""),
        long_term_context=state.get("long_term_context", ""),
    ):
        if event == "chunk":
            raw_text += data if isinstance(data, str) else ""
            _emit("writer_chunk", data)
        elif event == "done":
            result = data
            provider_degraded = provider_degraded or bool(data.get("_degraded"))
    # 初稿全文单独发一次：前端据此渲染"初稿已生成，正在安全检查"的草稿预览。
    # 若只发 writer_chunk，前端得自己把碎片拼起来并猜测"写完了没有"，
    # 而"写完了"这件事只有这里知道。
    _emit("writer_done_raw", raw_text)
    result = normalize_plan(result, profile=profile, plan_config=plan_config)
    return {"result": result, "provider_degraded": provider_degraded}


def check_node(deps, state: dict) -> dict:
    result = state.get("result") or {}
    profile = state.get("profile") or {}
    expected_goal = state.get("expected_goal")
    merged = check_with_goal_issue(
        result, deps.fact_checker.check(result, profile), expected_goal)
    provider_degraded = state.get("provider_degraded", False) or bool(merged.get("_degraded"))
    return {"latest_check": merged, "checks": [merged], "provider_degraded": provider_degraded}


def rewrite_node(deps, state: dict) -> dict:
    result = state.get("result") or {}
    profile = state.get("profile") or {}
    plan_config = (state.get("plan") or {}).get("skill_config", {})
    issues = (state.get("latest_check") or {}).get("issues", [])
    rewrite_count = state.get("rewrite_count", 0)
    provider_degraded = state.get("provider_degraded", False)

    _emit("stage", f"[修正] 安全检查发现 {len(issues)} 个问题，第 {rewrite_count + 1} 次重写...")

    new_result = deps.writer.rewrite_plan(result, issues, state.get("retrieved") or {}, profile)
    provider_degraded = provider_degraded or bool(new_result.get("_degraded"))
    new_result = normalize_plan(new_result, profile=profile, plan_config=plan_config)
    return {"result": new_result, "rewrite_count": rewrite_count + 1,
            "provider_degraded": provider_degraded}


def _known_library_names(deps) -> set | None:
    """动作库名单，供 finalize_result 校验库外动作名。

    取不到时返回 None——``collect_unknown_exercises`` 对 None 返回空列表即跳过
    该校验。动作库不可用不应让整条生成流程失败（校验是增强而非必需），
    测试用的 SimpleNamespace 伪造 deps 也不带这个能力。
    """
    getter = getattr(deps, "library_exercise_names_fn", None)
    if not callable(getter):
        return None
    try:
        return getter()
    except Exception:  # noqa: BLE001 - 增强校验失败不应中断生成
        return None


def finalize_node(deps, state: dict) -> dict:
    checks = list(state.get("checks") or [])
    result = dict(state.get("result") or {})
    profile = state.get("profile") or {}

    # 重写预算用尽仍在冲突 → 直接摘掉这些动作，而不是整份拒绝。与命令式
    # 编排同一口径（见 Orchestrator.generate_plan）。两条后端各写一份的话，
    # 前端走哪条就会拿到不同结果。
    if checks and getattr(deps, "fact_checker", None) is not None:
        result, last_check, pruned = prune_conflicts_with_recheck(
            result, checks[-1],
            lambda p: deps.fact_checker.check(p, profile))
        if pruned:
            checks.append(last_check)
            result["pruned_exercises"] = pruned
            result.setdefault("warnings", []).extend(
                f"「{name}」与你的伤病冲突，已从计划中移除" for name in pruned)
        if pruned and not [d for d in (result.get("days") or [])
                           if (d or {}).get("exercises")]:
            result["is_safe"] = False
            result["requires_human_review"] = True

    result = finalize_result(
        result, checks,
        state.get("rewrite_count", 0),
        provider_degraded=state.get("provider_degraded", False),
        expected_goal=state.get("expected_goal"),
        known_exercise_names=_known_library_names(deps))
    if not plan_goal_matches(result, state.get("expected_goal")):
        raise GoalConsistencyError("训练计划目标与用户目标不一致")
    # 与手写编排器共用同一个构建函数，两后端的解释块因此不会漂移
    result["explain"] = build_explanation(
        state.get("plan") or {}, state.get("retrieved") or {}, result,
        profile=state.get("profile") or {})
    return {"result": result}


# ---------------------------------------------------------------------------
# Human-in-the-loop
# ---------------------------------------------------------------------------

def open_review_node(deps, state: dict) -> dict:
    """Create the review artifact and register the thread. Runs to completion.

    Side effects live here (not in ``review_gate``) because LangGraph re-runs the
    interrupted node from the top on resume; doing the create there would duplicate
    artifacts. This node commits before the interrupt, so it runs exactly once.
    """
    result = state.get("result") or {}
    # owner_key 记下工件归属，审核接口据此拒绝他人查看/解除。
    artifact = create_review_artifact(
        deps.review_store, state.get("profile") or {}, state.get("query", ""), result,
        owner_key=state.get("athlete_key") or "")
    thread_id = state.get("thread_id")
    if thread_id:
        deps.thread_index.register(artifact.review_id, thread_id)
    payload = build_review_pending_payload(result, artifact, thread_id=thread_id)
    return {"review_id": artifact.review_id, "review_payload": payload}


def review_gate_node(deps, state: dict) -> dict:
    """Pause for human review; on resume record the decision. Idempotent."""
    resume_value = interrupt(state.get("review_payload"))
    decision = resume_value.get("decision") if isinstance(resume_value, dict) else resume_value
    if decision not in VALID_DECISIONS:
        decision = REJECTED
    return {"review_decision": decision}


def deliver_node(deps, state: dict) -> dict:
    """Assemble the terminal payload (and persist only a fully safe plan)."""
    result = state.get("result") or {}
    profile = state.get("profile") or {}
    query = state.get("query", "")
    session_id = state.get("session_id")
    review_decision = state.get("review_decision")

    # 急症信号优先于本节点其余所有分支——包括人工审核放行。
    #
    # 原先这里只认「生成的计划违不违规」：planner 检测到急症后会把 skill 覆盖成
    # exercise_analysis，可那只是换了拆解方向，下游照常生成并交付一份完整的
    # 训练计划。实测「训练时突然胸闷、头晕、眼前发黑，还能继续练吗」拿到了
    # 4 天增肌计划，而同一句话在问答入口会被明确挡下——同一条安全口径在两条
    # 路径上一处生效一处失效。症状可能指向心血管事件，此刻任何训练动作建议
    # 都是有害的，因此这里直接给出不含训练内容的终态载荷。
    #
    # 判定本身在 ingest 阶段做过（路由据此已经跳过了检索与生成），这里再验
    # 一次是**有意的重复**：这是数据流出的最后一道闸门，代价只是一次子串匹配，
    # 而漏判的代价是发出训练计划。state 里没有该键时（例如测试直接调本节点）
    # 就地补算。
    if "emergency_signals" in state:
        signals = state.get("emergency_signals") or []
    else:
        signals = emergency_signals_for(profile, query)
    if signals:
        return {"final_payload": emergency_stop_payload(signals)}

    if review_decision == APPROVED:
        # Human approval delivers the plan but does not pollute the automated
        # semantic cache / long-term memory; only the conversation is updated.
        final = dict(result)
        final["delivery_status"] = "review_approved"
        if session_id:
            summary = summarize_plan_for_context(result)
            deps.conversation.set_plan_state(session_id, summary)
            deps.conversation.add_turn(session_id, "assistant", summary[:500])
        return {"final_payload": final}

    if review_decision == REJECTED:
        return {"final_payload": {
            "delivery_status": "review_rejected",
            "requires_review": True,
            "review": {
                "review_id": state.get("review_id"),
                "prohibited_actions": REVIEW_PROHIBITED_ACTIONS,
                "next_step": "审核未通过，请勿执行该计划。",
            },
        }}

    # 安全类拒绝排在交付判定之前：它要给出的是"哪些动作不能做、该找谁"，
    # 而不是一句"这次没生成好"。它也取代了过去"送去人工审核"的位置——
    # 本项目没有审核人，扣下等人放行只会让用户永远拿不到计划。
    if result.get("_safety_block"):
        return {"final_payload": safety_blocked_payload(result)}

    # 交付判据用 _delivery_allowed，而不是 _persistence_allowed，也不是只看
    # requires_review：
    #   * _persistence_allowed 还要求问题清单干净（缓存的要求）。用它判断交付，
    #     带建议性警告的计划会拿不到 delivery_status，前端直接不渲染。
    #   * requires_review 只覆盖安全类问题。结构校验没过、置信度无效、目标不一致、
    #     生成降级这四项它都不反映，于是会出现"内部判定不允许交付、接口却说
    #     safe_delivered"的口径分裂。
    # _delivery_allowed 恰好是这两者之间那一档：不含 issues_empty，含其余全部。
    if result.get("_delivery_allowed") is True and result.get("requires_review") is not True:
        # athlete_key 必须传：它是长期记忆的身份键。漏传时 persist_if_safe 退回
        # make_user_key(profile) 指纹，而 ingest_node 是拿 athlete_key 去读的——
        # **写和读落在两个键上，用户永远读不回自己刚存的习惯与旧伤**。
        # 命令式编排一直传着这个参数，所以这不是"两边都错"，是两条后端分叉：
        # 只有状态图这一侧（也就是前端实际走的那条）的记忆是死的。
        persist_if_safe(deps.cache, deps.conversation, deps.long_term,
                        profile, query, result, session_id,
                        athlete_key=state.get("athlete_key"))
        final = dict(result)
        final["delivery_status"] = "safe_delivered"
        return {"final_payload": final}

    if result.get("requires_review") is True:
        # 兜底：带着"需要审核"的标记却没被送审。正常拓扑走不到这里
        # （requires_review 为真时路由会先去 open_review），但真出现时不能
        # 返回一个没有 delivery_status 的载荷——那等于让计划静默消失。
        final = dict(result)
        final["delivery_status"] = "review_pending"
        final.setdefault("requires_review", True)
        return {"final_payload": final}

    # 剩下的只有一种：交付闸门拒绝，但拒绝理由不是安全。结构、置信度、目标、
    # 降级这四项送去人工审核没有意义——审核员无从补齐缺失的字段。给出可重试的
    # 失败态，并保留已产出的 warnings（器械缺件这类提示对"该改什么"有用）。
    return {"final_payload": generation_failed_payload(result)}
