"""Offline flow tests for the LangGraph coach pipeline.

All dependencies are faked with ``SimpleNamespace`` / lambdas following the
project's existing mock idioms; no network, no live LLM. The graph uses an
in-memory checkpointer so interrupt/resume can be exercised within a test.
"""

import copy
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import MemorySaver

from src.core.goal_contract import GoalConsistencyError
from src.graph.builder import build_coach_graph
from src.graph.deps import CoachGraphDeps
from src.graph.events import build_inputs
from src.graph.runtime import GraphRuntime
from src.hitl.review_resolution import InMemoryReviewResolutionStore, ReviewThreadIndex
from src.hitl.review_store import InMemoryReviewArtifactStore
from src.llm.provider import LLMUnavailableError
from src.models.schemas import UserProfileInput


def valid_plan(goal="增肌"):
    return {
        "plan_id": "plan-1", "user_id": 1, "goal": goal, "weeks": 4,
        "sessions_per_week": 1,
        "days": [{"day": 1, "focus": "全身", "exercises": []}],
    }


def safe_check(**overrides):
    check = {"is_safe": True, "issues": [], "requires_human_review": False, "confidence": .9}
    check.update(overrides)
    return check


def unsafe_check(**overrides):
    check = {"is_safe": False, "issues": [{"issue": "强度过高", "severity": "warning"}],
             "requires_human_review": False, "confidence": .8}
    check.update(overrides)
    return check


class StoreSpy:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
        return record

    def assert_empty(self):
        assert self.calls == []

    def names(self):
        return [name for name, _, _ in self.calls]


def make_profile(goal="增肌", injuries=()):
    return UserProfileInput(height=180, weight=80, training_years=1.5, goal=goal,
                            available_equipment=["哑铃", "杠铃"], days_per_week=4,
                            injuries=list(injuries))


def make_runtime(*, plan=None, checks=None, rewrite_out=None, cache_return=None,
                 planner_raises=False, retrieved=None):
    """Build a GraphRuntime with fully faked dependencies."""
    plan = plan if plan is not None else valid_plan()
    checks = list(checks) if checks else [safe_check()]
    rewrite_out = rewrite_out if rewrite_out is not None else plan
    retrieved = retrieved if retrieved is not None else {"exercises": []}
    planner_out = {"skill": "general", "subtasks": [], "skill_config": {}}

    checks_queue = list(checks) if checks else [safe_check()]
    fallback_check = checks_queue[-1]
    rewrite_calls = {"count": 0}

    def check_fn(p, prof):
        # Pop provided checks in order; once exhausted, reuse the last one so a
        # mis-sized list never raises StopIteration inside the graph.
        return checks_queue.pop(0) if checks_queue else fallback_check

    if planner_raises:
        def plan_fn(*args, **kwargs):
            raise LLMUnavailableError("all models down",
                                      attempted_models=["m"], errors=["down"])
    else:
        def plan_fn(user_input, prof, conv_context="", plan_context=""):
            return planner_out

    planner = SimpleNamespace(plan=plan_fn)
    retriever = SimpleNamespace(retrieve=lambda p, route=None: retrieved)

    def write_plan_stream(retr, prof, plan_config, plan_context="", user_query=""):
        yield ("chunk", "生成中")
        yield ("done", copy.deepcopy(plan))

    def rewrite_plan(original, issues, retr, prof):
        rewrite_calls["count"] += 1
        return copy.deepcopy(rewrite_out)

    writer = SimpleNamespace(write_plan_stream=write_plan_stream, rewrite_plan=rewrite_plan)
    fact_checker = SimpleNamespace(check=check_fn)

    cache = StoreSpy()
    if cache_return is not None:
        cache.get = lambda prof, q: cache_return

    conversation = SimpleNamespace(
        add_turn=lambda *a, **k: None,
        build_context_for_prompt=lambda *a, **k: "",
        get_plan_state=lambda *a, **k: "",
        set_plan_state=lambda *a, **k: None,
    )
    long_term = StoreSpy()
    review_store = InMemoryReviewArtifactStore()
    resolutions = InMemoryReviewResolutionStore()
    thread_index = ReviewThreadIndex()

    deps = CoachGraphDeps(planner=planner, retriever=retriever, writer=writer,
                          fact_checker=fact_checker, cache=cache, conversation=conversation,
                          long_term=long_term, review_store=review_store,
                          resolutions=resolutions, thread_index=thread_index)
    graph = build_coach_graph(deps, MemorySaver())
    runtime = GraphRuntime(graph=graph, deps=deps, resolutions=resolutions,
                           thread_index=thread_index, checkpointer=None)
    runtime._rewrite_calls = rewrite_calls
    return runtime


def invoke_plan(runtime, profile, query="帮我设计计划"):
    thread_id = runtime.new_thread_id()
    config = runtime.config_for(thread_id)
    inputs = build_inputs(profile, query=query, thread_id=thread_id)
    return runtime.graph.invoke(inputs, config), thread_id, config


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_happy_path_delivers_safe_and_persists():
    runtime = make_runtime()
    state, thread_id, config = invoke_plan(runtime, make_profile())

    final = state["final_payload"]
    assert final["delivery_status"] == "safe_delivered"
    assert final["days"] == valid_plan()["days"]
    assert state["rewrite_count"] == 0
    # persisted to semantic cache + long-term memory
    assert "set" in runtime.deps.cache.names()
    assert runtime.deps.long_term.names().count("save_preference") == 3


def test_cache_hit_short_circuits_pipeline():
    cached = valid_plan() | {"_persistence_allowed": True, "requires_review": False, "warnings": []}
    runtime = make_runtime(cache_return=cached, planner_raises=True)  # planner must NOT run
    state, _, _ = invoke_plan(runtime, make_profile())

    assert state["final_payload"] == cached
    assert state["final_payload"]["plan_id"] == "plan-1"


# ---------------------------------------------------------------------------
# Rewrite loop
# ---------------------------------------------------------------------------

def test_single_rewrite_then_safe_delivery():
    runtime = make_runtime(checks=[unsafe_check(), safe_check()])
    state, _, _ = invoke_plan(runtime, make_profile())

    assert state["rewrite_count"] == 1
    assert runtime._rewrite_calls["count"] == 1
    assert state["final_payload"]["delivery_status"] == "safe_delivered"
    # checks accumulated via the reducer: initial + post-rewrite
    assert len(state["checks"]) == 2


def test_rewrite_budget_exhausted_escalates_to_review():
    runtime = make_runtime(plan=valid_plan(),
                           checks=[unsafe_check(), unsafe_check(), unsafe_check(), unsafe_check()])
    state, thread_id, config = invoke_plan(runtime, make_profile(injuries=["膝盖疼"]))

    assert state["rewrite_count"] == 3
    assert runtime._rewrite_calls["count"] == 3
    # interrupted at review_gate with a review payload
    assert "__interrupt__" in state
    payload = state["review_payload"]
    assert payload["delivery_status"] == "review_pending"
    assert payload["review"]["review_id"] == state["review_id"]
    assert payload["thread_id"] == thread_id
    # artifact stored + thread registered for later resume
    assert runtime.deps.review_store.get(state["review_id"]) is not None
    assert runtime.deps.thread_index.thread_for(state["review_id"]) == thread_id


# ---------------------------------------------------------------------------
# Resume (human-in-the-loop)
# ---------------------------------------------------------------------------

def _needs_review_check():
    return safe_check(requires_human_review=True,
                      review_reason="伤病冲突需要人工确认",
                      review_severity="warning")


def test_resume_approved_delivers_without_polluting_cache():
    from langgraph.types import Command
    runtime = make_runtime(checks=[_needs_review_check()])
    state, thread_id, config = invoke_plan(runtime, make_profile(injuries=["膝盖疼"]))
    assert "__interrupt__" in state

    cache_calls_before = len(runtime.deps.cache.calls)
    resumed = runtime.graph.invoke(Command(resume={"decision": "approved"}), config)
    assert resumed["review_decision"] == "approved"
    assert resumed["final_payload"]["delivery_status"] == "review_approved"
    # human approval must not write the semantic cache
    assert len(runtime.deps.cache.calls) == cache_calls_before


def test_resume_rejected_withholds_plan():
    from langgraph.types import Command
    runtime = make_runtime(checks=[_needs_review_check()])
    state, thread_id, config = invoke_plan(runtime, make_profile(injuries=["膝盖疼"]))
    assert "__interrupt__" in state

    resumed = runtime.graph.invoke(Command(resume={"decision": "rejected"}), config)
    assert resumed["review_decision"] == "rejected"
    final = resumed["final_payload"]
    assert final["delivery_status"] == "review_rejected"
    assert "days" not in final
    assert final["review"]["review_id"] == state["review_id"]


def test_resume_with_invalid_decision_fails_closed_to_rejected():
    from langgraph.types import Command
    runtime = make_runtime(checks=[_needs_review_check()])
    state, thread_id, config = invoke_plan(runtime, make_profile(injuries=["膝盖疼"]))
    resumed = runtime.graph.invoke(Command(resume={"decision": "banana"}), config)
    assert resumed["review_decision"] == "rejected"
    assert resumed["final_payload"]["delivery_status"] == "review_rejected"


# ---------------------------------------------------------------------------
# Failure semantics
# ---------------------------------------------------------------------------

def test_goal_mismatch_raises_goal_consistency_error():
    runtime = make_runtime(plan=valid_plan(goal="增肌"))
    with pytest.raises(GoalConsistencyError) as exc:
        invoke_plan(runtime, make_profile(goal="减脂"))
    assert exc.value.code == "GOAL_CONSISTENCY_FAILED"


def test_invalid_goal_fails_at_ingest():
    runtime = make_runtime()
    thread_id = runtime.new_thread_id()
    config = runtime.config_for(thread_id)
    inputs = {"profile": {"goal": "无效目标", "height": 180, "weight": 80},
              "query": "", "session_id": None, "thread_id": thread_id}
    with pytest.raises(GoalConsistencyError):
        runtime.graph.invoke(inputs, config)


def test_planner_unavailable_propagates():
    runtime = make_runtime(planner_raises=True)
    with pytest.raises(LLMUnavailableError):
        invoke_plan(runtime, make_profile())
