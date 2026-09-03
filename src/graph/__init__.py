"""LangGraph-based orchestration backend for the AI fitness coach.

This package re-implements the ``Orchestrator`` plan-generation pipeline as an
explicit LangGraph ``StateGraph`` so the workflow's states, branches, rewrite loop,
and human-review pause/resume are first-class, inspectable, and checkpointable. It
runs alongside the legacy orchestrator (kept as a comparison baseline) and reuses
the same agents, cache, memory, review store, and the shared
``src.core.plan_finalization`` terminal-state layer.
"""

from src.graph.builder import build_coach_graph
from src.graph.deps import CoachGraphDeps, deps_from_orchestrator
from src.graph.events import build_inputs, graph_stream_events
from src.graph.runtime import GraphRuntime, build_runtime
from src.graph.state import MAX_REWRITES, CoachState

__all__ = [
    "CoachState",
    "MAX_REWRITES",
    "CoachGraphDeps",
    "deps_from_orchestrator",
    "build_coach_graph",
    "build_runtime",
    "GraphRuntime",
    "build_inputs",
    "graph_stream_events",
]
