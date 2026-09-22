"""ReAct 自主循环的语义测试。

全部使用脚本化的假模型与假工具，不触网、不依赖 LLM 凭据——循环的正确性
（预算刹车、错误回喂、终止条件）与"用哪个模型"无关，正是把它设计成
依赖注入的目的。
"""

from __future__ import annotations

import pytest

from src.harness.loop import (
    AGENT_LOOP_ENV,
    LoopResult,
    ModelTurn,
    StepRecord,
    ToolCall,
    agent_loop_enabled,
    run_agent_loop,
)


# ---------------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------------


class ScriptedModel:
    """按剧本依次返回预设动作，剧本用完则重复最后一幕。"""

    def __init__(self, turns: list[ModelTurn]):
        self._turns = turns
        self.calls: list[list] = []  # 记录每次收到的 messages 快照长度

    def complete(self, messages: list, tools: list) -> ModelTurn:
        self.calls.append(list(messages))
        idx = min(len(self.calls) - 1, len(self._turns) - 1)
        return self._turns[idx]


class FakeTools:
    """带可编程行为的假工具集。"""

    def __init__(self, results: dict | None = None, names: list[str] | None = None):
        self._results = results or {}
        self._names = names if names is not None else ["lookup", "graph_query"]
        self.invocations: list[tuple[str, dict]] = []

    def list_tools(self) -> list[dict]:
        return [{"name": n, "parameters": {}} for n in self._names]

    def call(self, name: str, args: dict):
        self.invocations.append((name, args))
        outcome = self._results.get(name, {"ok": True})
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _turn_with_tool(name: str, args: dict | None = None, tokens: int = 10) -> ModelTurn:
    return ModelTurn(
        content=None,
        tool_calls=[ToolCall(name=name, args=args or {}, call_id=f"c_{name}")],
        tokens=tokens,
    )


def _final_turn(text: str, tokens: int = 10) -> ModelTurn:
    return ModelTurn(content=text, tool_calls=[], tokens=tokens)


# ---------------------------------------------------------------------------
# 正常终止
# ---------------------------------------------------------------------------


class TestCompletion:
    def test_returns_answer_when_model_calls_no_tools(self):
        model = ScriptedModel([_final_turn("这是最终答案")])
        result = run_agent_loop(model, FakeTools(), [])

        assert isinstance(result, LoopResult)
        assert result.completed
        assert result.exited_reason == "completed"
        assert result.content == "这是最终答案"
        assert result.steps == 1
        assert result.transcript == []

    def test_runs_tool_then_returns_answer(self):
        tools = FakeTools(results={"lookup": {"rows": 3}})
        model = ScriptedModel([_turn_with_tool("lookup", {"q": "深蹲"}), _final_turn("完成")])
        messages: list = []

        result = run_agent_loop(model, tools, messages)

        assert result.completed
        assert result.content == "完成"
        assert result.steps == 2
        assert tools.invocations == [("lookup", {"q": "深蹲"})]
        assert len(result.transcript) == 1
        assert result.transcript[0].ok is True
        assert result.transcript[0].tool == "lookup"

    def test_tool_result_is_appended_to_messages_for_the_model(self):
        """工具结果必须回喂模型，否则下一轮它看不到自己刚查到什么。"""
        tools = FakeTools(results={"lookup": {"rows": 3}})
        model = ScriptedModel([_turn_with_tool("lookup"), _final_turn("完成")])
        messages: list = []

        run_agent_loop(model, tools, messages)

        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["name"] == "lookup"
        assert "rows" in tool_msgs[0]["content"]

    def test_multiple_tool_calls_in_one_turn_are_all_executed(self):
        tools = FakeTools(results={"lookup": 1, "graph_query": 2})
        turn = ModelTurn(
            content=None,
            tool_calls=[
                ToolCall(name="lookup", args={"a": 1}, call_id="c1"),
                ToolCall(name="graph_query", args={"b": 2}, call_id="c2"),
            ],
            tokens=5,
        )
        model = ScriptedModel([turn, _final_turn("完成")])

        result = run_agent_loop(model, tools, [])

        assert [name for name, _ in tools.invocations] == ["lookup", "graph_query"]
        assert len(result.transcript) == 2


# ---------------------------------------------------------------------------
# 失败恢复：错误回喂而不是中断
# ---------------------------------------------------------------------------


class TestToolErrorFeedback:
    def test_tool_exception_does_not_abort_the_loop(self):
        """工具抛异常时循环应继续，让模型看到错误并换策略。"""
        tools = FakeTools(results={"lookup": RuntimeError("数据库连接超时")})
        model = ScriptedModel([_turn_with_tool("lookup"), _final_turn("改用其他方式")])

        result = run_agent_loop(model, tools, [])

        assert result.completed
        assert result.content == "改用其他方式"
        assert result.tool_errors == 1

    def test_error_payload_is_fed_back_to_the_model(self):
        tools = FakeTools(results={"lookup": RuntimeError("数据库连接超时")})
        model = ScriptedModel([_turn_with_tool("lookup"), _final_turn("ok")])
        messages: list = []

        run_agent_loop(model, tools, messages)

        tool_msg = next(m for m in messages if m.get("role") == "tool")
        assert "RuntimeError" in tool_msg["content"]
        assert "数据库连接超时" in tool_msg["content"]

    def test_unknown_tool_is_reported_with_available_names(self):
        """模型幻觉出不存在的工具是常见失败模式，提示应有助于自我纠正。"""
        tools = FakeTools(names=["lookup", "graph_query"])
        model = ScriptedModel([_turn_with_tool("nonexistent"), _final_turn("好的")])
        messages: list = []

        result = run_agent_loop(model, tools, messages)

        assert result.tool_errors == 1
        assert result.transcript[0].error_type == "UnknownTool"
        tool_msg = next(m for m in messages if m.get("role") == "tool")
        assert "unknown_tool" in tool_msg["content"]
        assert "lookup" in tool_msg["content"]
        # 未知工具不应被真的调用
        assert tools.invocations == []

    def test_transcript_records_error_type(self):
        tools = FakeTools(results={"lookup": ValueError("bad arg")})
        model = ScriptedModel([_turn_with_tool("lookup"), _final_turn("done")])

        result = run_agent_loop(model, tools, [])

        assert result.transcript[0].ok is False
        assert result.transcript[0].error_type == "ValueError"


# ---------------------------------------------------------------------------
# 预算刹车
# ---------------------------------------------------------------------------


class TestBudgets:
    def test_max_steps_stops_a_never_terminating_loop(self):
        """模型永远请求工具时必须被步数上限截断，而不是无限循环。"""
        tools = FakeTools()
        # 剧本只有工具调用，永远不给最终答案
        model = ScriptedModel([_turn_with_tool("lookup")])

        result = run_agent_loop(model, tools, [], max_steps=4)

        assert result.exited_reason == "max_steps"
        assert result.completed is False
        assert result.steps == 4
        assert len(tools.invocations) == 4

    def test_max_steps_exhaustion_returns_empty_content(self):
        """步数耗尽时不能把半成品当答案返回——调用方必须能分辨。"""
        model = ScriptedModel([_turn_with_tool("lookup")])

        result = run_agent_loop(model, FakeTools(), [], max_steps=3)

        assert result.content == ""
        assert not result.completed

    def test_token_budget_stops_the_loop(self):
        model = ScriptedModel([_turn_with_tool("lookup", tokens=10_000)])

        result = run_agent_loop(
            model, FakeTools(), [], max_steps=100, token_budget=5_000
        )

        assert result.exited_reason == "token_budget"
        assert result.tokens_used > 5_000
        assert result.content == ""

    def test_token_usage_accumulates_across_turns(self):
        tools = FakeTools()
        model = ScriptedModel([
            _turn_with_tool("lookup", tokens=300),
            _turn_with_tool("lookup", tokens=400),
            _final_turn("完成", tokens=100),
        ])

        result = run_agent_loop(model, tools, [], token_budget=100_000)

        assert result.tokens_used == 800
        assert result.completed

    def test_budget_check_happens_before_appending_final_answer(self):
        """超支应在模型回复后立刻判定，而不是先返回一个超预算的答案。"""
        model = ScriptedModel([_final_turn("答案", tokens=999_999)])

        result = run_agent_loop(model, FakeTools(), [], token_budget=1_000)

        assert result.exited_reason == "token_budget"
        assert result.content == ""


# ---------------------------------------------------------------------------
# 可观测性与开关
# ---------------------------------------------------------------------------


class TestObservability:
    def test_on_step_callback_fires_per_tool_call(self):
        seen: list[StepRecord] = []
        tools = FakeTools(results={"lookup": 1, "graph_query": 2})
        model = ScriptedModel([
            _turn_with_tool("lookup"),
            _turn_with_tool("graph_query"),
            _final_turn("完成"),
        ])

        run_agent_loop(model, tools, [], on_step=seen.append)

        assert [r.tool for r in seen] == ["lookup", "graph_query"]
        assert all(isinstance(r, StepRecord) for r in seen)

    def test_on_step_fires_even_when_the_tool_failed(self):
        """失败步骤同样要落 checkpoint，否则续跑会丢失失败历史。"""
        seen: list[StepRecord] = []
        tools = FakeTools(results={"lookup": RuntimeError("boom")})
        model = ScriptedModel([_turn_with_tool("lookup"), _final_turn("done")])

        run_agent_loop(model, tools, [], on_step=seen.append)

        assert len(seen) == 1
        assert seen[0].ok is False

    def test_long_tool_result_is_truncated_before_feeding_back(self):
        tools = FakeTools(results={"lookup": "x" * 5000})
        model = ScriptedModel([_turn_with_tool("lookup"), _final_turn("done")])
        messages: list = []

        run_agent_loop(model, tools, messages)

        tool_msg = next(m for m in messages if m.get("role") == "tool")
        assert len(tool_msg["content"]) <= 500


class TestAgentLoopFlag:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv(AGENT_LOOP_ENV, raising=False)
        assert agent_loop_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "Yes"])
    def test_enabled_by_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv(AGENT_LOOP_ENV, value)
        assert agent_loop_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "", "  "])
    def test_disabled_by_falsy_values(self, monkeypatch, value):
        monkeypatch.setenv(AGENT_LOOP_ENV, value)
        assert agent_loop_enabled() is False
