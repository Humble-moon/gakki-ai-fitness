"""LangGraph state schema for the coach plan-generation pipeline.

The whole pipeline is modelled as a single ``CoachState`` TypedDict that flows
through every node. Only ``checks`` needs a reducer (each rewrite round appends
a new FactChecker result instead of overwriting); every other field is written
by exactly one node in sequence, so last-write-wins is correct.

IMPORTANT: every key a node returns MUST be declared here. LangGraph silently
drops updates for keys that are not part of the schema, and the checkpointer can
only serialize JSON-compatible primitives — never put live objects (agents,
pydantic models, stores) into the state; those belong in :mod:`src.graph.deps`.
"""

import operator
from typing import Annotated, TypedDict

# Maximum number of FactChecker-driven rewrites before the plan is escalated.
# Mirrors the legacy ``while ... and rewrite_count < 3`` loop in the Orchestrator.
MAX_REWRITES = 3


class CoachState(TypedDict, total=False):
    # ---- Request inputs (written by the ``ingest`` node) ----
    query: str                    # user query, already defaulted to "为{goal}目标生成训练计划"
    profile: dict                 # profile.model_dump(); plain dict so checkpoints serialize
    expected_goal: str            # canonical goal from validate_requested_goal
    session_id: str | None        # enables multi-turn context when present
    thread_id: str                # LangGraph thread id, echoed into review payloads

    # ---- Context (populated by ``ingest`` when a session is active) ----
    conv_context: str             # sliding-window + summarized conversation context
    plan_context: str             # summary of a previously delivered plan

    # ---- Pipeline artifacts ----
    cache_hit: dict | None        # safe cached plan, or None on miss
    plan: dict                    # Planner output (skill, subtasks, skill_config)
    retrieved: dict               # Retriever output (exercises, ...)
    result: dict                  # current plan draft; normalized after every write/rewrite

    # ---- Checking & rewrite-loop control ----
    checks: Annotated[list[dict], operator.add]  # append-only history of FactChecker results
    latest_check: dict            # the most recent merged FactChecker result
    rewrite_count: int            # incremented by the ``rewrite`` node each round
    provider_degraded: bool       # True if any LLM call fell back / degraded

    # ---- Human-in-the-loop ----
    review_payload: dict          # built by ``open_review``; passed to interrupt()
    review_id: str                # id of the created ReviewArtifact
    review_decision: str          # "approved" | "rejected" after resume

    # ---- Terminal ----
    final_payload: dict           # the payload delivered to the caller
