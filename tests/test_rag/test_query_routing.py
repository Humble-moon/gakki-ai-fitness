from pathlib import Path

import pytest

from src.rag.query_routing import QueryRoute, classify_query


@pytest.mark.parametrize(
    "query",
    [
        "请教我杠铃深蹲的动作要领",
        "哑铃卧推怎么做？",
    ],
)
def test_real_seed_exercise_routes_to_exact_action(query):
    assert classify_query(query) is QueryRoute.EXACT_ACTION


def test_injury_sensitive_takes_priority_over_action_and_knowledge():
    assert classify_query("杠铃深蹲时膝盖疼，应该怎么调整？") is QueryRoute.INJURY_SENSITIVE


def test_multi_hop_relationship_routes_to_graph():
    assert classify_query("哪些动作训练股四头肌，并且适合初级难度？") is QueryRoute.GRAPH


def test_general_training_question_routes_to_knowledge():
    assert classify_query("如何安排一周三次的力量训练？") is QueryRoute.KNOWLEDGE


@pytest.mark.parametrize("query", ["", "   ", "x" * 1001, None])
def test_unknown_empty_or_oversized_input_falls_back(query):
    assert classify_query(query) is QueryRoute.FALLBACK


def test_router_does_not_access_external_services(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("external service must not be called")

    monkeypatch.setattr("socket.socket", fail)
    monkeypatch.setattr("urllib.request.urlopen", fail)
    assert classify_query("杠铃深蹲怎么做") is QueryRoute.EXACT_ACTION


def test_seed_file_is_local_fixture_only():
    assert Path("data/seed_exercises.json").is_file()
