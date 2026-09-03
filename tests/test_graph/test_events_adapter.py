"""Tests for the LangGraph → (event, data) SSE adapter."""

import pytest

from src.graph.events import build_inputs, graph_stream_events
from tests.test_graph.test_graph_flows import (
    _needs_review_check,
    make_profile,
    make_runtime,
    valid_plan,
)


def run_events(runtime, profile=None, query="测试"):
    profile = profile or make_profile()
    thread_id = runtime.new_thread_id()
    config = runtime.config_for(thread_id)
    inputs = build_inputs(profile, query=query, thread_id=thread_id)
    return list(graph_stream_events(runtime, inputs, config))


def test_happy_path_event_sequence_ends_with_done():
    runtime = make_runtime()
    events = run_events(runtime)

    names = [event for event, _ in events]
    assert names[-1] == "done"
    # progressive events reuse the legacy protocol names, in order
    for expected in ["planner_done", "retriever_done", "writer_chunk", "factcheck_done"]:
        assert expected in names
    assert names.index("planner_done") < names.index("retriever_done")
    assert names.index("retriever_done") < names.index("factcheck_done")
    # terminal payload is the delivered plan
    final_event, final_payload = events[-1]
    assert final_payload["delivery_status"] == "safe_delivered"


def test_interrupt_surfaces_review_payload_as_done():
    runtime = make_runtime(checks=[_needs_review_check()])
    events = run_events(runtime)
    final_event, final_payload = events[-1]
    assert final_event == "done"
    assert final_payload["delivery_status"] == "review_pending"
    assert final_payload["review"]["review_id"]


def test_goal_mismatch_emits_single_error_terminal():
    runtime = make_runtime(plan=valid_plan(goal="增肌"))
    events = run_events(runtime, profile=make_profile(goal="减脂"))
    terminals = [(e, d) for e, d in events if e in {"done", "error"}]
    assert len(terminals) == 1
    event, payload = terminals[0]
    assert event == "error"
    assert payload["code"] == "GOAL_CONSISTENCY_FAILED"


def test_llm_unavailable_emits_dependency_error():
    runtime = make_runtime(planner_raises=True)
    events = run_events(runtime)
    terminals = [(e, d) for e, d in events if e in {"done", "error"}]
    assert terminals == [("error", {"code": "DEPENDENCY_UNAVAILABLE",
                                    "message": "演示依赖暂时不可用"})]


def test_cache_hit_emits_cache_hit_then_done():
    cached = valid_plan() | {"_persistence_allowed": True,
                             "requires_review": False, "warnings": []}
    runtime = make_runtime(cache_return=cached)
    events = run_events(runtime)
    names = [event for event, _ in events]
    assert "cache_hit" in names
    assert names[-1] == "done"
    assert events[-1][1] == cached
