"""提示词工具调用适配器的测试。

重点覆盖**解析的失败分支**：模型不听话是常态（返回自然语言、漏字段、
格式非法），每一种失败都必须落到一个明确且安全的行为上，而不是抛异常
或伪装成成功。
"""

from __future__ import annotations

import json

import pytest

from src.harness.loop import ToolCall, run_agent_loop
from src.harness.tool_calling import (
    PromptToolCallingModel,
    _strip_code_fence,
    render_tool_schemas,
)


class FakeLLM:
    """记录收到的消息并返回预设文本的假 LLM。"""

    def __init__(self, contents: list[str], tokens: int = 7):
        self._contents = contents
        self._tokens = tokens
        self.seen: list[list] = []

    def chat(self, messages, temperature=0.3, model=None):
        self.seen.append([dict(m) for m in messages])

        class _Resp:
            pass

        r = _Resp()
        idx = min(len(self.seen) - 1, len(self._contents) - 1)
        r.content = self._contents[idx]
        r.tokens = self._tokens
        return r


class FakeTools:
    def __init__(self, results=None):
        self._results = results or {}
        self.invocations: list[tuple[str, dict]] = []

    def list_tools(self):
        return [
            {"name": "lookup", "description": "查动作", "parameters": {"type": "object"}},
            {"name": "graph_query", "description": "查图谱", "parameters": {"type": "object"}},
        ]

    def call(self, name, args):
        self.invocations.append((name, args))
        outcome = self._results.get(name, {"ok": True})
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_parses_tool_calls(self):
        turn = PromptToolCallingModel.parse_response(
            '{"tool_calls": [{"name": "lookup", "args": {"q": "深蹲"}}]}'
        )
        assert turn.content is None
        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0].name == "lookup"
        assert turn.tool_calls[0].args == {"q": "深蹲"}

    def test_parses_answer(self):
        turn = PromptToolCallingModel.parse_response('{"answer": "这是答案"}')
        assert turn.content == "这是答案"
        assert turn.tool_calls == []

    def test_parses_fenced_json(self):
        raw = '```json\n{"answer": "围栏里的答案"}\n```'
        assert PromptToolCallingModel.parse_response(raw).content == "围栏里的答案"

    def test_parses_bare_fence_without_language_tag(self):
        raw = '```\n{"answer": "无标签围栏"}\n```'
        assert PromptToolCallingModel.parse_response(raw).content == "无标签围栏"

    def test_plain_text_becomes_final_answer(self):
        """模型没按协议返回时按答案处理——重试只会在同一个坑里空转。"""
        turn = PromptToolCallingModel.parse_response("直接说了一句人话")
        assert turn.content == "直接说了一句人话"
        assert turn.tool_calls == []

    def test_json_without_expected_fields_becomes_answer(self):
        turn = PromptToolCallingModel.parse_response('{"foo": "bar"}')
        assert turn.content == '{"foo": "bar"}'
        assert turn.tool_calls == []

    def test_empty_content_yields_empty_turn(self):
        turn = PromptToolCallingModel.parse_response("")
        assert turn.content == ""
        assert turn.tool_calls == []

    def test_tokens_are_propagated(self):
        turn = PromptToolCallingModel.parse_response('{"answer": "x"}', tokens=42)
        assert turn.tokens == 42

    def test_non_dict_json_becomes_answer(self):
        turn = PromptToolCallingModel.parse_response("[1, 2, 3]")
        assert turn.content == "[1, 2, 3]"
        assert turn.tool_calls == []


class TestMalformedToolCalls:
    """格式非法的 tool_calls 不能导致"空成功"。"""

    @pytest.mark.parametrize(
        "payload",
        [
            '{"tool_calls": ["not-a-dict"]}',
            '{"tool_calls": [{"args": {}}]}',
            '{"tool_calls": [{"name": ""}]}',
            '{"tool_calls": [{"name": 123}]}',
            '{"tool_calls": [null]}',
        ],
    )
    def test_all_invalid_calls_fall_back_to_answer_not_empty_turn(self, payload):
        """若返回空 tool_calls，循环会误判为"模型给出最终答案"并返回空串。

        这是刻意防守的回归点：一次格式错误绝不能伪装成一次成功。
        """
        turn = PromptToolCallingModel.parse_response(payload)
        assert turn.tool_calls == []
        assert turn.content == payload  # 回落成答案，而非 None/空

    def test_valid_calls_survive_alongside_invalid_ones(self):
        payload = json.dumps({
            "tool_calls": [
                {"name": "lookup", "args": {"q": "x"}},
                "garbage",
                {"name": "graph_query"},
            ]
        })
        turn = PromptToolCallingModel.parse_response(payload)
        assert [c.name for c in turn.tool_calls] == ["lookup", "graph_query"]

    def test_null_args_coerced_to_empty_dict(self):
        payload = '{"tool_calls": [{"name": "lookup", "args": null}]}'
        turn = PromptToolCallingModel.parse_response(payload)
        assert turn.tool_calls[0].args == {}

    def test_missing_id_gets_synthesised(self):
        payload = '{"tool_calls": [{"name": "lookup"}]}'
        turn = PromptToolCallingModel.parse_response(payload)
        assert turn.tool_calls[0].call_id


# ---------------------------------------------------------------------------
# 协议注入
# ---------------------------------------------------------------------------


class TestProtocolInjection:
    def test_protocol_is_prepended_as_system_message(self):
        msgs = [{"role": "user", "content": "你好"}]
        out = PromptToolCallingModel._with_protocol(msgs, FakeTools().list_tools())
        assert out[0]["role"] == "system"
        assert "tool_calls" in out[0]["content"]

    def test_existing_system_message_is_preserved(self):
        """项目有伤病安全约束等 system 消息，不能被协议顶掉。"""
        msgs = [
            {"role": "system", "content": "伤病安全约束：禁止高风险动作"},
            {"role": "user", "content": "生成计划"},
        ]
        out = PromptToolCallingModel._with_protocol(msgs, FakeTools().list_tools())
        contents = " ".join(m["content"] for m in out)
        assert "伤病安全约束" in contents
        assert out[0]["role"] == "system"

    def test_original_messages_are_not_mutated(self):
        msgs = [{"role": "user", "content": "你好"}]
        before = [dict(m) for m in msgs]
        PromptToolCallingModel._with_protocol(msgs, FakeTools().list_tools())
        assert msgs == before

    def test_tool_schemas_are_rendered(self):
        out = PromptToolCallingModel._with_protocol([], FakeTools().list_tools())
        assert "lookup" in out[0]["content"]
        assert "graph_query" in out[0]["content"]

    def test_no_tools_renders_placeholder(self):
        assert "没有可用工具" in render_tool_schemas([])

    def test_nameless_tools_are_skipped(self):
        rendered = render_tool_schemas([{"description": "无名字"}, {"name": "ok"}])
        assert "ok" in rendered
        assert "无名字" not in rendered


# ---------------------------------------------------------------------------
# 与循环联调
# ---------------------------------------------------------------------------


class TestLoopIntegration:
    def test_adapter_drives_the_loop_to_completion(self):
        llm = FakeLLM([
            '{"tool_calls": [{"name": "lookup", "args": {"q": "深蹲"}}]}',
            '{"answer": "最终计划"}',
        ])
        tools = FakeTools(results={"lookup": {"rows": 2}})
        model = PromptToolCallingModel(llm)

        result = run_agent_loop(model, tools, [{"role": "user", "content": "帮我"}])

        assert result.completed
        assert result.content == "最终计划"
        assert result.steps == 2
        assert tools.invocations == [("lookup", {"q": "深蹲"})]

    def test_protocol_resent_every_turn(self):
        """协议必须每轮重新注入，否则第二轮的模型看不到可用工具。"""
        llm = FakeLLM([
            '{"tool_calls": [{"name": "lookup", "args": {}}]}',
            '{"answer": "完成"}',
        ])
        model = PromptToolCallingModel(llm)

        run_agent_loop(model, FakeTools(), [{"role": "user", "content": "帮我"}])

        assert len(llm.seen) == 2
        for messages in llm.seen:
            assert messages[0]["role"] == "system"
            assert "lookup" in messages[0]["content"]

    def test_token_usage_flows_into_budget_accounting(self):
        llm = FakeLLM(['{"answer": "x"}'], tokens=250)
        model = PromptToolCallingModel(llm)

        result = run_agent_loop(model, FakeTools(), [], token_budget=10_000)

        assert result.tokens_used == 250

    def test_tool_error_reaches_the_model_on_the_next_turn(self):
        llm = FakeLLM([
            '{"tool_calls": [{"name": "lookup", "args": {}}]}',
            '{"answer": "改用图谱了"}',
        ])
        tools = FakeTools(results={"lookup": RuntimeError("库挂了")})
        model = PromptToolCallingModel(llm)

        result = run_agent_loop(model, tools, [{"role": "user", "content": "帮我"}])

        assert result.completed
        assert result.tool_errors == 1
        # 第二轮的消息里必须带上工具错误
        second_turn_msgs = llm.seen[1]
        tool_msgs = [m for m in second_turn_msgs if m.get("role") == "tool"]
        assert tool_msgs and "库挂了" in tool_msgs[0]["content"]


class TestRealToolRegistryIntegration:
    """用**真实的** ToolRegistry 跑一遍，证明这个接缝不是纸上的。

    前面所有测试都用假工具，能证明循环逻辑正确，但证明不了
    ``ToolRegistry`` 的接口真的对得上。这里把两者接起来：如果哪天有人改了
    ``call()`` 或 ``list_tools()`` 的签名，这些测试会立刻失败。
    """

    @pytest.fixture(scope="class")
    def registry(self):
        from src.mcp.tool_registry import ToolRegistry

        return ToolRegistry()

    def test_registry_satisfies_the_loop_tool_protocol(self, registry):
        schemas = registry.list_tools()
        assert schemas, "ToolRegistry 应至少注册一个工具"
        assert all(isinstance(s, dict) and s.get("name") for s in schemas)

    def test_loop_drives_a_real_tool_call(self, registry):
        """模型请求 search_by_muscle，循环真的去调它并拿到动作数据。"""
        target = "search_by_muscle"
        llm = FakeLLM([
            json.dumps({"tool_calls": [{"name": target, "args": {"muscle": "胸大肌"}}]}),
            json.dumps({"answer": "已找到胸部动作"}),
        ])

        result = run_agent_loop(
            PromptToolCallingModel(llm), registry, [{"role": "user", "content": "练胸"}]
        )

        assert result.completed
        assert result.content == "已找到胸部动作"
        assert result.tool_errors == 0
        assert result.transcript[0].tool == target
        assert result.transcript[0].ok is True

    def test_real_tool_error_is_fed_back_rather_than_raised(self, registry):
        """用一个不存在的工具名，验证真实注册表的错误路径也走"回喂"。"""
        llm = FakeLLM([
            json.dumps({"tool_calls": [{"name": "no_such_tool", "args": {}}]}),
            json.dumps({"answer": "换个工具"}),
        ])

        result = run_agent_loop(
            PromptToolCallingModel(llm), registry, [{"role": "user", "content": "x"}]
        )

        assert result.completed
        assert result.tool_errors == 1
        tool_msgs = [m for m in llm.seen[-1] if m.get("role") == "tool"]
        assert tool_msgs, "错误必须作为工具消息回喂模型"


class TestStripFence:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("```json\n{}\n```", "{}"),
            ("```\n{}\n```", "{}"),
            ("{}", "{}"),
            ("```json\n{\"a\":1}\n```\n", '{"a":1}'),
        ],
    )
    def test_strips(self, raw, expected):
        assert _strip_code_fence(raw) == expected
