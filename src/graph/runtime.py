"""Runtime assembly for the coach graph: dependencies, checkpointer, thread ids.

``build_runtime`` is the single entry point used by the FastAPI layer. It wires the
graph's dependencies from the existing Orchestrator (reusing the same agents and
stores) and attaches a checkpointer so that interrupt/resume and cross-restart
recovery work.
"""

import os
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver

from src.graph.builder import build_coach_graph
from src.graph.deps import CoachGraphDeps, deps_from_orchestrator
from src.hitl.review_resolution import InMemoryReviewResolutionStore, ReviewThreadIndex

# Worst case is ~16 supersteps (3 rewrite rounds); leave comfortable headroom.
RECURSION_LIMIT = 60
DEFAULT_DB_PATH = "data/graph_checkpoints.db"


def _make_checkpointer(db_path: str = DEFAULT_DB_PATH):
    """Pick a checkpointer: MemorySaver for tests, SqliteSaver for durable demos."""
    if os.environ.get("COACH_CHECKPOINTER") == "memory":
        return MemorySaver()
    from langgraph.checkpoint.sqlite import SqliteSaver
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()  # idempotent table creation
    return saver


@dataclass
class GraphRuntime:
    graph: object
    deps: CoachGraphDeps
    resolutions: InMemoryReviewResolutionStore
    thread_index: ReviewThreadIndex
    checkpointer: object

    def new_thread_id(self) -> str:
        return str(uuid.uuid4())

    def config_for(self, thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id},
                "recursion_limit": RECURSION_LIMIT}


def build_runtime(orchestrator, *, checkpointer=None,
                  db_path: str = DEFAULT_DB_PATH) -> GraphRuntime:
    """Assemble a ready-to-invoke graph runtime around an existing Orchestrator."""
    resolutions = InMemoryReviewResolutionStore()
    thread_index = ReviewThreadIndex()
    deps = deps_from_orchestrator(orchestrator, resolutions, thread_index)
    if checkpointer is None:
        checkpointer = _make_checkpointer(db_path)
    graph = build_coach_graph(deps, checkpointer)
    return GraphRuntime(graph=graph, deps=deps, resolutions=resolutions,
                        thread_index=thread_index, checkpointer=checkpointer)
