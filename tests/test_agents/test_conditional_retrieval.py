from types import SimpleNamespace

from src.agents.retriever import RetrieverAgent
from src.rag.agentic_rag import AgenticRAG


def test_retriever_forwards_explicit_route_and_filters(monkeypatch):
    agent = RetrieverAgent.__new__(RetrieverAgent)
    calls = []
    agent.agentic_rag = SimpleNamespace(
        search=lambda query, **kwargs: calls.append((query, kwargs)) or [{"name": "row"}]
    )
    agent.tools = SimpleNamespace(call=lambda *args: [])

    result = agent.retrieve(
        {"subtasks": ["胸部"], "skill_config": {"retrieval_filters": {"equipment": "哑铃"}}},
        route="graph",
    )

    assert result["exercises"] == [{"name": "row"}]
    assert calls == [("胸部", {"filters": {"equipment": "哑铃"}, "route": "graph"})]


def test_retriever_old_call_remains_valid(monkeypatch):
    agent = RetrieverAgent.__new__(RetrieverAgent)
    calls = []
    agent.agentic_rag = SimpleNamespace(
        search=lambda query, **kwargs: calls.append(kwargs) or []
    )
    agent.tools = SimpleNamespace(call=lambda *args: [])

    assert agent.retrieve({"subtasks": []}) == {"exercises": [], "knowledge": []}
    assert calls == []


def test_agentic_route_none_uses_legacy_path_and_preserves_filters():
    rag = AgenticRAG.__new__(AgenticRAG)
    calls = []
    rag._legacy_search = lambda query, **kwargs: calls.append((query, kwargs)) or [{"name": "x"}]

    assert rag.search("练胸", filters={"equipment": "哑铃"}) == [{"name": "x"}]
    assert calls == [("练胸", {"filters": {"equipment": "哑铃"}, "max_retries": None})]


def test_explicit_graph_route_returns_graph_output_without_downgrade():
    rag = AgenticRAG.__new__(AgenticRAG)
    rag._graph = SimpleNamespace(search=lambda query, **kwargs: [{"name": "graph-hit"}])

    assert rag.search("查关系", filters={"difficulty": "初级"}, route="graph") == [{"name": "graph-hit"}]


def test_graph_route_failure_is_explicit_empty_not_legacy_results():
    rag = AgenticRAG.__new__(AgenticRAG)
    rag._graph = SimpleNamespace(search=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("down")))
    rag._legacy_search = lambda *args, **kwargs: [{"name": "unsafe-downgrade"}]

    assert rag.search("肩膀疼", route="injury_sensitive") == []


def test_graph_backend_without_generic_search_returns_empty():
    rag = AgenticRAG.__new__(AgenticRAG)
    rag._graph = SimpleNamespace()
    rag._legacy_search = lambda *args, **kwargs: [{"name": "ordinary"}]

    assert rag.search("关系查询", route="graph") == []
