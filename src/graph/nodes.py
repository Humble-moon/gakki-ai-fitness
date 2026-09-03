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
    normalize_plan,
    persist_if_safe,
    safe_cached_result,
    summarize_plan_for_context,
)
from src.hitl.review_resolution import APPROVED, REJECTED, VALID_DECISIONS

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

    return {
        "query": query,
        "expected_goal": expected_goal,
        "conv_context": conv_context,
        "plan_context": plan_context,
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
    _emit("stage", "[规划] Planner 正在拆解任务...")
    plan = deps.planner.plan(
        state.get("query", ""), state.get("profile") or {},
        conv_context=state.get("conv_context", ""),
        plan_context=state.get("plan_context", ""),
    )
    return {"plan": plan}


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
    for event, data in deps.writer.write_plan_stream(
        state.get("retrieved") or {}, profile, plan_config,
        plan_context=state.get("plan_context", ""), user_query=state.get("query", ""),
    ):
        if event == "chunk":
            _emit("writer_chunk", data)
        elif event == "done":
            result = data
            provider_degraded = provider_degraded or bool(data.get("_degraded"))
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


def finalize_node(deps, state: dict) -> dict:
    result = finalize_result(
        state.get("result") or {}, state.get("checks") or [],
        state.get("rewrite_count", 0),
        provider_degraded=state.get("provider_degraded", False),
        expected_goal=state.get("expected_goal"))
    if not plan_goal_matches(result, state.get("expected_goal")):
        raise GoalConsistencyError("训练计划目标与用户目标不一致")
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
    artifact = create_review_artifact(
        deps.review_store, state.get("profile") or {}, state.get("query", ""), result)
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

    if result.get("_persistence_allowed") is True:
        persist_if_safe(deps.cache, deps.conversation, deps.long_term,
                        profile, query, result, session_id)
        final = dict(result)
        final["delivery_status"] = "safe_delivered"
        return {"final_payload": final}

    # Not persistable and not escalated to review (e.g. degraded provider):
    # return the result as-is, mirroring the legacy orchestrator.
    return {"final_payload": dict(result)}
