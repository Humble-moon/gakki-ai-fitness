from types import SimpleNamespace

import pytest

from src.agents.retriever import RetrieverAgent
from src.mcp.exercise_server import McpToolError


EMPTY_RESULT = {"exercises": [], "knowledge": []}


def make_agent(rag_search, tool_call):
    agent = RetrieverAgent.__new__(RetrieverAgent)
    agent.agentic_rag = SimpleNamespace(search=rag_search)
    agent.tools = SimpleNamespace(call=tool_call)
    return agent


@pytest.mark.parametrize(
    "plan",
    [
        None,
        [],
        "plan",
        {},
        {"subtasks": None},
        {"subtasks": "胸部训练"},
        {"subtasks": [None]},
        {"subtasks": [""]},
    ],
)
def test_malformed_or_empty_plan_returns_empty_without_retrieval(plan):
    calls = []
    agent = make_agent(
        lambda *args, **kwargs: calls.append("rag") or [],
        lambda *args, **kwargs: calls.append("mcp") or [],
    )

    assert agent.retrieve(plan) == EMPTY_RESULT
    assert calls == []


@pytest.mark.parametrize(
    "skill_config",
    [None, [], "bad", 1, {"retrieval_filters": None}, {"retrieval_filters": []}],
)
def test_malformed_skill_config_uses_empty_filters(skill_config):
    calls = []
    agent = make_agent(
        lambda query, **kwargs: calls.append((query, kwargs)) or [],
        lambda *args, **kwargs: [],
    )

    assert agent.retrieve({"subtasks": ["腿部训练"], "skill_config": skill_config}) == EMPTY_RESULT
    assert calls == [("腿部训练", {"filters": {}})]


def test_mcp_failure_is_best_effort_and_preserves_rag_results():
    rag_row = {"name": "深蹲", "source": "vector"}

    def failing_tool(*args, **kwargs):
        raise McpToolError(code=-32603, message="Internal error", tool_name="search_by_muscle")

    agent = make_agent(lambda *args, **kwargs: [rag_row], failing_tool)

    assert agent.retrieve({"subtasks": ["腿部训练"]}) == {
        "exercises": [rag_row],
        "knowledge": [],
    }


def test_malformed_mcp_rows_are_skipped_and_optional_fields_are_safe():
    agent = make_agent(
        lambda *args, **kwargs: [],
        lambda *args, **kwargs: [
            None,
            "bad",
            {},
            {"name": ""},
            {"name": "臀桥"},
        ],
    )

    result = agent.retrieve({"subtasks": ["腿部训练"]})

    assert result == {
        "exercises": [
            {
                "name": "臀桥",
                "source": "mcp",
                "muscles": [],
                "equipment": None,
                "difficulty": None,
                "type": None,
            }
        ],
        "knowledge": [],
    }


def test_rag_and_mcp_results_are_stably_deduplicated_by_name():
    rag_row = {"name": "深蹲", "source": "vector"}
    mcp_rows = [
        {
            "name": "深蹲",
            "target_muscles": ["腿"],
            "equipment": "杠铃",
            "difficulty": "中级",
            "type": "力量",
        },
        {
            "name": "臀桥",
            "target_muscles": ["臀"],
            "equipment": "自重",
            "difficulty": "初级",
            "type": "力量",
        },
    ]
    agent = make_agent(lambda *args, **kwargs: [rag_row], lambda *args, **kwargs: mcp_rows)

    result = agent.retrieve({"subtasks": ["腿部训练"]})

    assert [row["name"] for row in result["exercises"]] == ["深蹲", "臀桥"]
    assert result["exercises"][0] is rag_row


def test_malformed_rag_rows_are_skipped_at_retriever_boundary():
    valid = {"name": "硬拉", "source": "keyword", "extra": 1}
    agent = make_agent(
        lambda *args, **kwargs: [None, "bad", {}, {"name": " "}, valid],
        lambda *args, **kwargs: [],
    )

    assert agent.retrieve({"subtasks": ["拉类训练"]})["exercises"] == [valid]


@pytest.mark.parametrize("route", ["graph", "injury_sensitive"])
def test_explicit_safety_sensitive_route_never_supplements_with_mcp(route):
    tool_calls = []
    agent = make_agent(
        lambda *args, **kwargs: [],
        lambda *args, **kwargs: tool_calls.append(args) or [{"name": "普通动作"}],
    )

    assert agent.retrieve({"subtasks": ["肩膀疼"]}, route=route) == EMPTY_RESULT
    assert tool_calls == []


def test_rag_failure_remains_a_hard_failure():
    agent = make_agent(
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("rag down")),
        lambda *args, **kwargs: [],
    )

    with pytest.raises(RuntimeError, match="rag down"):
        agent.retrieve({"subtasks": ["腿部训练"]})
