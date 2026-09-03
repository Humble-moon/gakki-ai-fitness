"""Assemble the coach StateGraph.

Topology (the thesis's core diagram — export with ``graph.get_graph().draw_mermaid()``):

    START → ingest → check_cache
    check_cache --hit--> deliver_cached → END
    check_cache --miss--> plan → retrieve → write → check
    check --clean--> finalize
    check --issues & budget left--> rewrite → check        (rewrite loop)
    check --issues & budget spent--> finalize
    finalize --safe--> deliver → END
    finalize --needs review--> open_review → review_gate(interrupt)
    review_gate --resume--> deliver → END
"""

from functools import partial

from langgraph.graph import END, StateGraph

from src.graph import nodes, routing
from src.graph.state import CoachState

_NODE_FUNCS = [
    ("ingest", nodes.ingest_node),
    ("check_cache", nodes.check_cache_node),
    ("deliver_cached", nodes.deliver_cached_node),
    ("plan", nodes.plan_node),
    ("retrieve", nodes.retrieve_node),
    ("write", nodes.write_node),
    ("check", nodes.check_node),
    ("rewrite", nodes.rewrite_node),
    ("finalize", nodes.finalize_node),
    ("open_review", nodes.open_review_node),
    ("review_gate", nodes.review_gate_node),
    ("deliver", nodes.deliver_node),
]


def build_coach_graph(deps, checkpointer=None):
    """Compile the coach pipeline into a runnable graph.

    ``deps`` is bound to every node via ``functools.partial`` so the state schema
    stays free of live objects. ``checkpointer`` enables persistence / interrupt-resume;
    pass ``None`` only for stateless single-shot runs.
    """
    g = StateGraph(CoachState)
    for name, fn in _NODE_FUNCS:
        g.add_node(name, partial(fn, deps))

    g.set_entry_point("ingest")
    g.add_edge("ingest", "check_cache")
    g.add_conditional_edges(
        "check_cache", routing.route_after_cache,
        {"deliver_cached": "deliver_cached", "plan": "plan"})
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "write")
    g.add_edge("write", "check")
    g.add_conditional_edges(
        "check", routing.route_after_check,
        {"finalize": "finalize", "rewrite": "rewrite"})
    g.add_edge("rewrite", "check")  # rewrite loop back-edge
    g.add_conditional_edges(
        "finalize", routing.route_after_finalize,
        {"open_review": "open_review", "deliver": "deliver"})
    g.add_edge("open_review", "review_gate")
    g.add_edge("review_gate", "deliver")  # after resume, always deliver
    g.add_edge("deliver", END)
    g.add_edge("deliver_cached", END)

    return g.compile(checkpointer=checkpointer)
