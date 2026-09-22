"""问答工具门面的测试。

重点覆盖**降级路径**：图谱不可用、参数缺失、底层检索抛异常。
这些是模型最容易撞上的情况，也必须给出模型看得懂的反馈，
而不是把异常直接抛穿整个循环。
"""

from __future__ import annotations

import pytest

from src.harness.qa_tools import QAToolKit


class FakeKnowledge:
    def __init__(self, chunks=None, raises=None):
        self._chunks = chunks if chunks is not None else []
        self._raises = raises
        self.queries: list[str] = []

    def search_with_fallback(self, query):
        self.queries.append(query)
        if self._raises:
            raise self._raises
        return self._chunks


class FakeRetriever:
    def __init__(self, exercises=None):
        self._exercises = exercises if exercises is not None else []
        self.plans: list[dict] = []

    def retrieve(self, plan):
        self.plans.append(plan)
        return {"exercises": self._exercises, "knowledge": []}


class FakeRegistry:
    def __init__(self, by_muscle=None, pain=None, raises=None):
        self._by_muscle = by_muscle or []
        self._pain = pain or {}
        self._raises = raises
        self.calls: list[tuple] = []

    def call(self, name, params):
        self.calls.append((name, params))
        if self._raises:
            raise self._raises
        if name == "search_by_muscle":
            return self._by_muscle
        if name == "graph_reason_pain":
            return self._pain
        raise ValueError(f"unknown tool {name}")


def _kit(**kw) -> QAToolKit:
    return QAToolKit(
        knowledge=kw.get("knowledge", FakeKnowledge()),
        retriever=kw.get("retriever", FakeRetriever()),
        tool_registry=kw.get("registry", FakeRegistry()),
    )


class TestToolSchema:
    def test_exposes_the_expected_tools(self):
        names = {t["name"] for t in _kit().list_tools()}
        assert names == {"search_knowledge", "search_exercises", "reason_injury"}

    def test_every_tool_declares_a_schema_and_description(self):
        for tool in _kit().list_tools():
            assert tool.get("description"), f"{tool['name']} 缺 description"
            assert tool.get("inputSchema", {}).get("type") == "object"
            assert "query" in tool["inputSchema"].get("properties", {}) or \
                   "exercise" in tool["inputSchema"].get("properties", {})

    def test_graph_tool_hidden_when_registry_absent(self):
        """图谱不可用时不该列出该工具——列出来只会让模型白试一次。"""
        names = {t["name"] for t in _kit(registry=None).list_tools()}
        assert "reason_injury" not in names
        assert names == {"search_knowledge", "search_exercises"}


class TestSearchKnowledge:
    def test_returns_structured_items(self):
        kit = _kit(knowledge=FakeKnowledge([
            {"text": "深蹲要点" * 20, "source": "doc1", "rerank_score": 8},
        ]))
        out = kit.call("search_knowledge", {"query": "深蹲"})
        assert out["count"] == 1
        assert out["items"][0]["score"] == 8
        assert len(out["items"][0]["text"]) <= 160  # 已截断

    def test_caps_item_count(self):
        kit = _kit(knowledge=FakeKnowledge([{"text": f"t{i}"} for i in range(50)]))
        out = kit.call("search_knowledge", {"query": "x"})
        assert out["count"] == 50          # 原始条数如实报告
        assert len(out["items"]) == 5      # 回喂模型的数量被收敛

    def test_empty_query_is_rejected_not_crashed(self):
        out = _kit().call("search_knowledge", {})
        assert out["error"] == "empty_query"

    def test_whitespace_query_is_rejected(self):
        assert _kit().call("search_knowledge", {"query": "   "})["error"] == "empty_query"

    def test_underlying_failure_propagates_as_exception(self):
        """检索彻底失败时抛异常，交由循环捕获并回喂——不在这里伪装成功。"""
        kit = _kit(knowledge=FakeKnowledge(raises=RuntimeError("向量库挂了")))
        with pytest.raises(RuntimeError):
            kit.call("search_knowledge", {"query": "深蹲"})


class TestSearchExercises:
    def test_semantic_mode_uses_retriever(self):
        retriever = FakeRetriever([{"name": "深蹲", "target_muscles": ["股四头肌"]}])
        out = _kit(retriever=retriever).call("search_exercises", {"query": "练腿"})
        assert out["mode"] == "semantic"
        assert out["items"][0]["name"] == "深蹲"
        # 最小 plan 应被正确构造
        assert retriever.plans[0]["subtasks"] == ["练腿"]

    def test_by_muscle_mode_uses_registry(self):
        registry = FakeRegistry(by_muscle=[{"name": "卧推", "equipment": "杠铃"}])
        out = _kit(registry=registry).call(
            "search_exercises", {"query": "胸大肌", "mode": "by_muscle"}
        )
        assert out["mode"] == "by_muscle"
        assert registry.calls == [("search_by_muscle", {"muscle": "胸大肌"})]

    def test_by_muscle_falls_back_to_semantic_without_registry(self):
        """没有注册表时不能抛异常，应退回语义检索。"""
        out = _kit(registry=None).call(
            "search_exercises", {"query": "胸大肌", "mode": "by_muscle"}
        )
        assert out["mode"] == "semantic"

    def test_empty_query_rejected(self):
        assert _kit().call("search_exercises", {})["error"] == "empty_query"


class TestReasonInjury:
    def test_passes_arguments_through(self):
        registry = FakeRegistry(pain={"causes": ["髌骨软化"]})
        out = _kit(registry=registry).call(
            "reason_injury", {"exercise": "深蹲", "symptom": "膝盖疼"}
        )
        assert registry.calls == [
            ("graph_reason_pain", {"exercise": "深蹲", "symptom": "膝盖疼"})
        ]
        assert out["result"]["causes"] == ["髌骨软化"]

    def test_missing_registry_returns_actionable_error(self):
        """错误信息要能指导模型下一步，而不是只说"失败"。"""
        out = _kit(registry=None).call(
            "reason_injury", {"exercise": "深蹲", "symptom": "膝盖疼"}
        )
        assert out["error"] == "graph_unavailable"
        assert "search_knowledge" in out["message"]  # 指向替代方案

    def test_missing_arguments_reported(self):
        out = _kit().call("reason_injury", {"exercise": "深蹲"})
        assert out["error"] == "missing_argument"

    def test_blank_arguments_reported(self):
        out = _kit().call("reason_injury", {"exercise": "  ", "symptom": "疼"})
        assert out["error"] == "missing_argument"


class TestUnknownTool:
    def test_unknown_tool_raises(self):
        with pytest.raises(ValueError, match="未知工具"):
            _kit().call("no_such_tool", {})


class TestRealIntegrations:
    """用真实 ToolRegistry 验证接口对得上。"""

    def test_kit_satisfies_loop_protocol_with_real_registry(self):
        from src.mcp.tool_registry import ToolRegistry

        kit = QAToolKit(
            knowledge=FakeKnowledge(),
            retriever=FakeRetriever(),
            tool_registry=ToolRegistry(),
        )
        schemas = kit.list_tools()
        assert len(schemas) == 3
        # 协议要求：list_tools 返回的 name 必须都能被 call 认出
        for schema in schemas:
            assert schema["name"] in {"search_knowledge", "search_exercises", "reason_injury"}

    def test_real_by_muscle_query_works_offline(self):
        from src.mcp.tool_registry import ToolRegistry

        kit = QAToolKit(FakeKnowledge(), FakeRetriever(), ToolRegistry())
        out = kit.call("search_exercises", {"query": "胸大肌", "mode": "by_muscle"})
        assert out["count"] >= 1
        assert out["items"][0]["name"]
