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
DEFAULT_HITL_DB_PATH = "data/hitl_reviews.db"


def _make_checkpointer(db_path: str = DEFAULT_DB_PATH):
    """Pick a checkpointer: MemorySaver for tests, SqliteSaver for durable demos,
    PostgresSaver for production-grade concurrent deployments.

    Backends:
    * ``COACH_CHECKPOINTER=memory`` — in-memory, offline tests (no persistence)
    * ``GRAPH_CHECKPOINT_BACKEND=postgres`` — PostgreSQL with a connection pool
      (concurrency-safe; SQLite's single connection is the demo-tier limit)
    * default — SqliteSaver at ``db_path``
    """
    if os.environ.get("COACH_CHECKPOINTER") == "memory":
        return MemorySaver()
    if os.environ.get("GRAPH_CHECKPOINT_BACKEND", "sqlite").lower() == "postgres":
        return _make_postgres_checkpointer()
    from langgraph.checkpoint.sqlite import SqliteSaver
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()  # idempotent table creation
    return saver


def _make_postgres_checkpointer():
    """PostgreSQL-backed checkpointer for concurrent production use.

    Uses a psycopg connection pool (SqliteSaver's single connection is the
    bottleneck under concurrent requests). ``prepare_threshold=0`` keeps the
    pool compatible with connection multiplexers like pgbouncer.
    """
    from psycopg_pool import ConnectionPool
    from langgraph.checkpoint.postgres import PostgresSaver
    from src.config import DATABASE_URL
    pool = ConnectionPool(
        conninfo=DATABASE_URL,
        min_size=1,
        max_size=10,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    )
    saver = PostgresSaver(pool)
    saver.setup()  # idempotent table creation
    return saver


@dataclass
class GraphRuntime:
    graph: object
    deps: CoachGraphDeps
    resolutions: object  # InMemoryReviewResolutionStore | SqliteReviewResolutionStore
    thread_index: object  # ReviewThreadIndex | SqliteReviewThreadIndex
    checkpointer: object

    def new_thread_id(self) -> str:
        return str(uuid.uuid4())

    def config_for(self, thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id},
                "recursion_limit": RECURSION_LIMIT}


def _make_hitl_stores(orchestrator, hitl_db_path: str | None):
    """Pick the review-loop stores: SQLite by default, memory for tests.

    ``HITL_STORE_BACKEND=memory`` restores the legacy in-process stores (offline
    tests, throwaway demos). With the default SQLite backend the artifact store
    is also swapped into the Orchestrator, so the legacy and graph backends
    share one durable store instead of two memory ones.
    """
    backend = os.environ.get("HITL_STORE_BACKEND", "sqlite").lower()
    if backend == "memory":
        return InMemoryReviewResolutionStore(), ReviewThreadIndex()
    from src.hitl.review_storage import (SqliteReviewArtifactStore,
                                         SqliteReviewResolutionStore,
                                         SqliteReviewThreadIndex)
    path = hitl_db_path or os.environ.get("HITL_DB_PATH", DEFAULT_HITL_DB_PATH)
    orchestrator.review_store = SqliteReviewArtifactStore(path)
    return SqliteReviewResolutionStore(path), SqliteReviewThreadIndex(path)


def build_runtime(orchestrator, *, checkpointer=None,
                  db_path: str = DEFAULT_DB_PATH,
                  hitl_db_path: str | None = None) -> GraphRuntime:
    """Assemble a ready-to-invoke graph runtime around an existing Orchestrator."""
    resolutions, thread_index = _make_hitl_stores(orchestrator, hitl_db_path)
    deps = deps_from_orchestrator(orchestrator, resolutions, thread_index)
    if checkpointer is None:
        checkpointer = _make_checkpointer(db_path)
    graph = build_coach_graph(deps, checkpointer)
    return GraphRuntime(graph=graph, deps=deps, resolutions=resolutions,
                        thread_index=thread_index, checkpointer=checkpointer)
