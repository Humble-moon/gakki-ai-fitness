"""Conditional-edge routing functions for the coach graph.

These three pure functions encode every branch of the pipeline. They read the
state and return the name of the next node; the builder wires the returned names
to real nodes via ``add_conditional_edges``.
"""

from src.graph.state import MAX_REWRITES


def route_after_cache(state: dict) -> str:
    """Short-circuit to delivery on a safe cache hit, otherwise start planning."""
    return "deliver_cached" if state.get("cache_hit") else "plan"


def route_after_check(state: dict) -> str:
    """Drive the rewrite loop; equivalent to the legacy while-condition.

    Legacy (current): ``while not check.is_safe and rewrite_count < 3``.
    An earlier version also looped on non-empty ``issues``; that was dropped
    because the LLM checker essentially never returns an empty issue list
    (advisory notes only), so the loop ran to the budget every time while
    changing nothing about delivery — see the comment in
    ``Orchestrator.generate_plan_stream``. Keep this condition in sync with the
    legacy orchestrator: the two backends are meant to behave equivalently so
    the paper's line-count comparison stays fair.
    """
    check = state.get("latest_check") or {}
    if check.get("is_safe", True) or state.get("rewrite_count", 0) >= MAX_REWRITES:
        return "finalize"
    return "rewrite"


def route_after_finalize(state: dict) -> str:
    """Escalate to human review when the final gate is not fully safe."""
    result = state.get("result") or {}
    if result.get("requires_review") and result.get("_persistence_allowed") is not True:
        return "open_review"
    return "deliver"
