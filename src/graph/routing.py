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
    """Drive the rewrite loop; verbatim equivalent to the legacy while-condition.

    Legacy: ``while (not check.is_safe or check.issues) and rewrite_count < 3``.
    Here we finalize when the plan is clean OR the rewrite budget is exhausted;
    otherwise we loop back through ``rewrite``.
    """
    check = state.get("latest_check") or {}
    clean = check.get("is_safe", True) and not check.get("issues")
    if clean or state.get("rewrite_count", 0) >= MAX_REWRITES:
        return "finalize"
    return "rewrite"


def route_after_finalize(state: dict) -> str:
    """Escalate to human review when the final gate is not fully safe."""
    result = state.get("result") or {}
    if result.get("requires_review") and result.get("_persistence_allowed") is not True:
        return "open_review"
    return "deliver"
