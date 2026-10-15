"""Conditional-edge routing functions for the coach graph.

These three pure functions encode every branch of the pipeline. They read the
state and return the name of the next node; the builder wires the returned names
to real nodes via ``add_conditional_edges``.
"""

from src.graph.state import MAX_REWRITES


def route_after_cache(state: dict) -> str:
    """Short-circuit to delivery on a safe cache hit, otherwise start planning.

    排在缓存之前的是急症：命中急症信号时既不能交付缓存里的计划（那同样是
    训练内容），也没有必要生成新的——直接进交付节点给出处置。这条分流是
    **响应时间**上的关键一步：没有它，用户带着胸闷头晕填完表单，要等一次
    完整的检索与生成（实测 60~90 秒）才能看到「请先就医」。
    """
    if state.get("emergency_signals"):
        return "deliver"
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
    """Escalate to human review when the final gate is not fully safe.

    判据只看 ``requires_review``。原先附带 ``_persistence_allowed is not True``，
    但那个标志是缓存闸门、额外要求问题清单干净，与"该不该送审"无关；
    两个条件在最终归一化里本就互斥（可缓存必然蕴含不需要审核），
    附带条件因此从未生效——留着只会让人误以为送审还与缓存有关。
    """
    result = state.get("result") or {}
    if result.get("requires_review"):
        return "open_review"
    return "deliver"
