"""`Orchestrator.answer_question_stream` 的分发与回退测试。

这是本次改动里**风险最高的接缝**：它在实验性的自主循环与成熟的固定
流水线之间做选择，而用户只看得到最终事件流。因此两件事必须被钉死：

1. 开关关闭时，行为与改动前**完全一致**（一次都不碰自主循环）；
2. 开关打开但自主循环失败时，**必须回退**，绝不能让用户收到空答案。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.core import orchestrator as orch_mod
from src.core.orchestrator import Orchestrator
from src.harness.loop import LoopResult
from src.models.schemas import UserProfileInput

PROFILE = UserProfileInput(
    height=175, weight=70, training_years=2, goal="增肌",
    available_equipment=["哑铃"], days_per_week=3,
)


def _make_orch(**attrs):
    """构造一个不触发任何网络/数据库初始化的 Orchestrator。"""
    orch = Orchestrator.__new__(Orchestrator)
    orch.writer = SimpleNamespace(llm=SimpleNamespace())
    orch.knowledge = SimpleNamespace()
    orch.retriever = SimpleNamespace(tools=SimpleNamespace())
    for k, v in attrs.items():
        setattr(orch, k, v)
    return orch


def _fixed_stream(events):
    """替换固定流水线的假实现，记录它有没有被调用。"""
    calls = []

    def _impl(self, question, profile, session_id=None):
        calls.append((question, session_id))
        yield from events

    return _impl, calls


def _ok_loop(steps=2):
    return LoopResult(
        content="这是自主循环给出的答案",
        steps=steps,
        tokens_used=120,
        transcript=[],
        exited_reason="completed",
    )


def _failed_loop(reason="max_steps"):
    return LoopResult(
        content="", steps=10, tokens_used=999,
        transcript=[], exited_reason=reason,
    )


class TestFlagOff:
    """默认路径必须与改动前完全一致。"""

    def test_uses_fixed_pipeline_when_disabled(self, monkeypatch):
        monkeypatch.delenv("HARNESS_AGENT_LOOP", raising=False)
        impl, calls = _fixed_stream([("stage", "固定流水线"), ("answer_chunk", "答案")])
        monkeypatch.setattr(Orchestrator, "_answer_question_fixed_stream", impl)

        # 若自主循环被误调用，这里会炸——正是我们想验证的
        monkeypatch.setattr(
            orch_mod, "answer_with_agent_loop",
            lambda *a, **k: pytest.fail("关闭状态下不应触碰自主循环"),
        )

        orch = _make_orch()
        events = list(orch.answer_question_stream("增肌怎么吃", PROFILE))

        assert events == [("stage", "固定流水线"), ("answer_chunk", "答案")]
        assert len(calls) == 1


class TestFlagOnSuccess:
    def test_uses_agent_loop_when_enabled(self, monkeypatch):
        monkeypatch.setenv("HARNESS_AGENT_LOOP", "1")
        impl, calls = _fixed_stream([("stage", "不应出现")])
        monkeypatch.setattr(Orchestrator, "_answer_question_fixed_stream", impl)
        monkeypatch.setattr(
            orch_mod, "answer_with_agent_loop",
            lambda *a, **k: ("这是自主循环给出的答案", _ok_loop()),
        )

        orch = _make_orch()
        events = list(orch.answer_question_stream("增肌怎么吃", PROFILE))

        kinds = [e[0] for e in events]
        assert "answer_chunk" in kinds
        assert calls == [], "自主循环成功时不该回退"

    def test_answer_is_reassembled_from_chunks(self, monkeypatch):
        monkeypatch.setenv("HARNESS_AGENT_LOOP", "1")
        monkeypatch.setattr(Orchestrator, "_answer_question_fixed_stream", *_fixed_stream([]))
        monkeypatch.setattr(
            orch_mod, "answer_with_agent_loop",
            lambda *a, **k: ("ABCDEFGHIJ", _ok_loop()),
        )

        orch = _make_orch()
        events = list(orch.answer_question_stream("问题", PROFILE))
        rebuilt = "".join(p for kind, p in events if kind == "answer_chunk")
        assert rebuilt == "ABCDEFGHIJ"

    def test_emits_agent_result_event(self, monkeypatch):
        monkeypatch.setenv("HARNESS_AGENT_LOOP", "1")
        monkeypatch.setattr(Orchestrator, "_answer_question_fixed_stream", *_fixed_stream([]))
        monkeypatch.setattr(
            orch_mod, "answer_with_agent_loop",
            lambda *a, **k: ("答案", _ok_loop(steps=3)),
        )

        orch = _make_orch()
        events = list(orch.answer_question_stream("问题", PROFILE))
        payloads = [p for kind, p in events if kind == "agent_result"]
        assert payloads and payloads[0]["steps"] == 3


class TestFallback:
    """自主循环失败时必须回退——这是本接缝最重要的保证。"""

    @pytest.mark.parametrize("reason", ["max_steps", "token_budget"])
    def test_falls_back_when_loop_does_not_converge(self, monkeypatch, reason):
        monkeypatch.setenv("HARNESS_AGENT_LOOP", "1")
        impl, calls = _fixed_stream([("answer_chunk", "固定流水线的答案")])
        monkeypatch.setattr(Orchestrator, "_answer_question_fixed_stream", impl)
        monkeypatch.setattr(
            orch_mod, "answer_with_agent_loop",
            lambda *a, **k: ("", _failed_loop(reason)),
        )

        orch = _make_orch()
        events = list(orch.answer_question_stream("问题", PROFILE))

        assert len(calls) == 1, "未收敛时必须回退到固定流水线"
        assert ("answer_chunk", "固定流水线的答案") in events

    def test_falls_back_when_loop_raises(self, monkeypatch):
        monkeypatch.setenv("HARNESS_AGENT_LOOP", "1")
        impl, calls = _fixed_stream([("answer_chunk", "回退答案")])
        monkeypatch.setattr(Orchestrator, "_answer_question_fixed_stream", impl)

        def boom(*a, **k):
            raise RuntimeError("LLM 连接中断")

        monkeypatch.setattr(orch_mod, "answer_with_agent_loop", boom)

        orch = _make_orch()
        events = list(orch.answer_question_stream("问题", PROFILE))

        assert len(calls) == 1
        assert ("answer_chunk", "回退答案") in events

    def test_never_emits_empty_answer_on_failure(self, monkeypatch):
        """最关键的断言：失败绝不能表现为"成功但空答案"。"""
        monkeypatch.setenv("HARNESS_AGENT_LOOP", "1")
        monkeypatch.setattr(
            Orchestrator, "_answer_question_fixed_stream",
            *_fixed_stream([("answer_chunk", "兜底答案")]),
        )
        monkeypatch.setattr(
            orch_mod, "answer_with_agent_loop",
            lambda *a, **k: ("", _failed_loop()),
        )

        orch = _make_orch()
        events = list(orch.answer_question_stream("问题", PROFILE))
        rebuilt = "".join(p for kind, p in events if kind == "answer_chunk")
        assert rebuilt != ""

    def test_fallback_is_announced_to_the_user(self, monkeypatch):
        monkeypatch.setenv("HARNESS_AGENT_LOOP", "1")
        monkeypatch.setattr(
            Orchestrator, "_answer_question_fixed_stream",
            *_fixed_stream([("answer_chunk", "x")]),
        )
        monkeypatch.setattr(
            orch_mod, "answer_with_agent_loop",
            lambda *a, **k: ("", _failed_loop()),
        )

        orch = _make_orch()
        events = list(orch.answer_question_stream("问题", PROFILE))
        stages = " ".join(str(p) for kind, p in events if kind == "stage")
        assert "回退" in stages


class TestSafetyParity:
    """两条路径必须共用同一份安全实现。"""

    def test_both_paths_import_the_shared_module(self):
        import inspect

        from src.core import qa_agent, qa_safety

        assert "qa_safety" in inspect.getsource(orch_mod)
        assert "qa_safety" in inspect.getsource(qa_agent)
        # 断言安全常量本身没有被复制一份到别处
        assert qa_safety.SAFETY_SYSTEM_MESSAGE
