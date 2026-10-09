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
    finalize_result,
    long_term_user_key,
    normalize_plan,
    persist_if_safe,
    safe_cached_result,
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
    return {"cache_hit": safe_cached_result(cached, expected_goal=state.get("expected_goal"))}


def deliver_cached_node(deps, state: dict) -> dict:
    return {"final_payload": state.get("cache_hit")}


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
    result = finalize_result(
        state.get("result") or {}, state.get("checks") or [],
        state.get("rewrite_count", 0),
        provider_degraded=state.get("provider_degraded", False),
        expected_goal=state.get("expected_goal"),
        known_exercise_names=_known_library_names(deps))
    if not plan_goal_matches(result, state.get("expected_goal")):
        raise GoalConsistencyError("训练计划目标与用户目标不一致")
    # 与手写编排器共用同一个构建函数，两后端的解释块因此不会漂移
    result["explain"] = build_explanation(
        state.get("plan") or {}, state.get("retrieved") or {}, result)
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

    # 交付判据用 requires_review（与命令式编排同一口径），而不是 _persistence_allowed。
    # 后者还额外交代了"问题清单必须干净"——那是缓存的要求。用错判据的后果不是
    # 少缓一次，而是**交付带建议性警告的计划拿不到 delivery_status**，
    # 前端因 `!== 'safe_delivered'` 直接不渲染：计划有内容，用户却什么都看不到。
    # 这也与最终归一化里新增的 _delivery_allowed 保持同一语义。
    if result.get("requires_review") is not True:
        persist_if_safe(deps.cache, deps.conversation, deps.long_term,
                        profile, query, result, session_id)
        final = dict(result)
        final["delivery_status"] = "safe_delivered"
        return {"final_payload": final}

    # 兜底：既没被送去审核、又带着"需要审核"的标记。正常拓扑走不到这里
    # （requires_review 为真时路由会先去 open_review），但真出现时不能
    # 返回一个没有 delivery_status 的载荷——那等于让计划静默消失。
    final = dict(result)
    final["delivery_status"] = "review_pending"
    final.setdefault("requires_review", True)
    return {"final_payload": final}
