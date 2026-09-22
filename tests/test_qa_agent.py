"""问答自主循环路径的测试。

全部离线：用假 LLM 与假工具驱动，不触网。循环的正确性与"用哪个模型"
无关——这正是把它设计成依赖注入的目的。
"""

from __future__ import annotations

import json

import pytest

from src.core.qa_agent import _BASE_SYSTEM, answer_with_agent_loop, build_qa_messages
from src.core.qa_safety import SAFETY_SYSTEM_MESSAGE

PROFILE = {"height": 175, "weight": 70, "training_years": 2, "injuries": []}


class FakeLLM:
    """按剧本依次返回内容的假 LLM。"""

    def __init__(self, contents: list[str], tokens: int = 12):
        self._contents = contents
        self._tokens = tokens
        self.seen: list[list] = []

    def chat(self, messages, temperature=0.3, model=None):
        self.seen.append([dict(m) for m in messages])

        class _R:
            pass

        r = _R()
        idx = min(len(self.seen) - 1, len(self._contents) - 1)
        r.content = self._contents[idx]
        r.tokens = self._tokens
        return r


class FakeTools:
    def __init__(self):
        self.invocations: list[tuple] = []

    def list_tools(self):
        return [
            {"name": "search_knowledge", "description": "查知识库"},
            {"name": "search_exercises", "description": "查动作库"},
        ]

    def call(self, name, args):
        self.invocations.append((name, args))
        return {"count": 1, "items": [{"text": "结果"}]}


def _tool_call(name: str, args: dict | None = None) -> str:
    return json.dumps({"tool_calls": [{"name": name, "args": args or {}}]})


def _answer(text: str) -> str:
    return json.dumps({"answer": text})


# ---------------------------------------------------------------------------
# 消息组装
# ---------------------------------------------------------------------------


class TestBuildQaMessages:
    def test_includes_profile_and_question(self):
        msgs = build_qa_messages("增肌怎么吃", PROFILE)
        user = [m for m in msgs if m["role"] == "user"][0]
        assert "175cm" in user["content"]
        assert "增肌怎么吃" in user["content"]

    def test_non_safety_question_has_no_safety_system_message(self):
        """普通问题不该背上安全话术——那会污染回答风格。"""
        msgs = build_qa_messages("增肌怎么吃", PROFILE)
        contents = " ".join(m["content"] for m in msgs if m["role"] == "system")
        assert SAFETY_SYSTEM_MESSAGE not in contents

    def test_pain_question_injects_hard_constraint(self):
        msgs = build_qa_messages("深蹲膝盖疼怎么办", PROFILE)
        systems = [m["content"] for m in msgs if m["role"] == "system"]
        assert any(SAFETY_SYSTEM_MESSAGE in s for s in systems)

    def test_injury_history_alone_triggers_safety(self):
        """没有疼痛关键词，但有伤病史，同样要走安全话术。"""
        msgs = build_qa_messages(
            "今天练什么", {**PROFILE, "injuries": ["腰椎间盘突出"]}
        )
        systems = [m["content"] for m in msgs if m["role"] == "system"]
        assert any(SAFETY_SYSTEM_MESSAGE in s for s in systems)

    def test_safety_system_message_comes_before_user(self):
        """硬约束必须在 user 消息之前——否则更容易被覆盖。"""
        msgs = build_qa_messages("膝盖疼", PROFILE)
        roles = [m["role"] for m in msgs]
        assert roles.index("system") < roles.index("user")

    def test_base_persona_always_present(self):
        msgs = build_qa_messages("增肌", PROFILE)
        assert any(_BASE_SYSTEM in m["content"] for m in msgs)

    def test_evasion_still_triggers_safety(self):
        """绕过写法经过归一化后仍应触发安全约束。"""
        msgs = build_qa_messages("膝​盖 疼", PROFILE)
        systems = [m["content"] for m in msgs if m["role"] == "system"]
        assert any(SAFETY_SYSTEM_MESSAGE in s for s in systems)


# ---------------------------------------------------------------------------
# 循环行为
# ---------------------------------------------------------------------------


class TestAnswerWithAgentLoop:
    def test_returns_answer_and_result_on_completion(self):
        llm = FakeLLM([_answer("练胸建议这样安排")])
        answer, result = answer_with_agent_loop(llm, FakeTools(), "怎么练胸", PROFILE)
        assert answer == "练胸建议这样安排"
        assert result.completed
        assert result.steps == 1

    def test_model_can_query_then_answer(self):
        tools = FakeTools()
        llm = FakeLLM([
            _tool_call("search_knowledge", {"query": "增肌饮食"}),
            _answer("多吃蛋白质"),
        ])
        answer, result = answer_with_agent_loop(llm, tools, "增肌怎么吃", PROFILE)

        assert result.completed
        assert answer == "多吃蛋白质"
        assert tools.invocations == [("search_knowledge", {"query": "增肌饮食"})]
        assert result.steps == 2

    def test_budget_exhaustion_returns_empty_answer(self):
        """触顶时必须返回空串 + completed=False，让调用方能回退。"""
        tools = FakeTools()
        llm = FakeLLM([_tool_call("search_knowledge")])  # 永远查，不给答案
        answer, result = answer_with_agent_loop(
            llm, tools, "问题", PROFILE, max_steps=3
        )

        assert answer == ""
        assert result.completed is False
        assert result.exited_reason == "max_steps"
        assert len(tools.invocations) == 3

    def test_token_budget_respected(self):
        llm = FakeLLM([_tool_call("search_knowledge")], tokens=50_000)
        answer, result = answer_with_agent_loop(
            llm, FakeTools(), "问题", PROFILE, token_budget=1_000
        )
        assert result.exited_reason == "token_budget"
        assert answer == ""

    def test_model_calling_no_tools_answers_immediately(self):
        """模型不问就答也应当是合法路径。"""
        llm = FakeLLM([_answer("直接回答")])
        answer, result = answer_with_agent_loop(llm, FakeTools(), "问题", PROFILE)
        assert answer == "直接回答"
        assert result.steps == 1

    def test_on_step_callback_is_forwarded(self):
        seen = []
        llm = FakeLLM([
            _tool_call("search_knowledge", {"query": "a"}),
            _answer("好了"),
        ])
        answer_with_agent_loop(
            llm, FakeTools(), "问题", PROFILE, on_step=seen.append
        )
        assert len(seen) == 1
        assert seen[0].tool == "search_knowledge"

    def test_tool_errors_do_not_abort(self):
        class Boom(FakeTools):
            def call(self, name, args):
                raise RuntimeError("检索服务挂了")

        llm = FakeLLM([
            _tool_call("search_knowledge"),
            _answer("换个方式回答"),
        ])
        answer, result = answer_with_agent_loop(llm, Boom(), "问题", PROFILE)

        assert result.completed
        assert answer == "换个方式回答"
        assert result.tool_errors == 1

    def test_plain_text_response_is_treated_as_answer(self):
        """模型不按协议返回时按答案处理，不该空转到预算耗尽。"""
        llm = FakeLLM(["我直接说了一句人话"])
        answer, result = answer_with_agent_loop(llm, FakeTools(), "问题", PROFILE)
        assert result.completed
        assert answer == "我直接说了一句人话"


class TestSafetyIsPreservedInLoop:
    """安全约束必须真的到达模型——这是本路径最不能出错的地方。"""

    def test_hard_constraint_reaches_the_model_on_first_turn(self):
        llm = FakeLLM([_answer("好的")])
        answer_with_agent_loop(llm, FakeTools(), "膝盖疼怎么办", PROFILE)
        first = " ".join(m["content"] for m in llm.seen[0])
        assert SAFETY_SYSTEM_MESSAGE in first

    def test_hard_constraint_resent_on_every_turn(self):
        """多轮后约束不能丢——PromptToolCallingModel 每轮重新注入协议，
        但安全消息来自 messages 本身，须确保它一直留在列表里。"""
        llm = FakeLLM([
            _tool_call("search_knowledge"),
            _answer("好了"),
        ])
        answer_with_agent_loop(llm, FakeTools(), "膝盖疼", PROFILE)

        assert len(llm.seen) == 2
        for turn in llm.seen:
            joined = " ".join(m["content"] for m in turn)
            assert SAFETY_SYSTEM_MESSAGE in joined

    def test_non_safety_question_never_gets_safety_text(self):
        llm = FakeLLM([_answer("好的")])
        answer_with_agent_loop(llm, FakeTools(), "增肌怎么吃", PROFILE)
        joined = " ".join(m["content"] for m in llm.seen[0])
        assert SAFETY_SYSTEM_MESSAGE not in joined


class TestToolProtocolInstructions:
    def test_tool_schemas_are_visible_to_the_model(self):
        llm = FakeLLM([_answer("x")])
        answer_with_agent_loop(llm, FakeTools(), "问题", PROFILE)
        first = " ".join(m["content"] for m in llm.seen[0])
        assert "search_knowledge" in first
        assert "search_exercises" in first
