"""Conditional-edge routing functions for the coach graph.

These three pure functions encode every branch of the pipeline. They read the
state and return the name of the next node; the builder wires the returned names
to real nodes via ``add_conditional_edges``.
"""

from src.config import MANUAL_REVIEW_ENABLED
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
    """Send the plan to delivery; human review is opt-in only.

    默认一律走交付节点。最终归一化已经把"能不能交付"和"为什么不能"都算好
    了，交付节点据此产出 safe_delivered / safety_blocked / generation_failed
    三种终态。**"等待人工审核"不再是其中之一**——本项目没有审核人这个角色，
    扣下等人放行等于让用户永远拿不到计划。

    ``COACH_REVIEW_MODE=manual`` 时恢复旧的送审分支，用于开发期核对规则
    引擎有没有误报。
    """
    result = state.get("result") or {}
    if MANUAL_REVIEW_ENABLED and result.get("requires_review"):
        return "open_review"
    return "deliver"
