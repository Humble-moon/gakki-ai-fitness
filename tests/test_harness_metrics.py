"""harness 过程指标的测试。"""

from __future__ import annotations

import pytest

from eval.metrics.harness_metrics import (
    evaluate_checkpoint_resume,
    evaluate_harness_loop,
    format_harness_report,
)
from src.harness.loop import LoopResult, StepRecord


def _loop_result(
    *,
    reason: str = "completed",
    content: str = "答案",
    steps: int = 1,
    tokens: int = 10,
    steps_records: list | None = None,
    tool_errors: int = 0,
) -> LoopResult:
    return LoopResult(
        content=content,
        steps=steps,
        tokens_used=tokens,
        transcript=steps_records or [],
        exited_reason=reason,
        tool_errors=tool_errors,
    )


def _record(tool: str = "lookup", ok: bool = True, error: str | None = None) -> StepRecord:
    return StepRecord(
        step=1, tool=tool, args={}, ok=ok, result_preview="", error_type=error
    )


class TestEmptyInput:
    def test_no_results_returns_error(self):
        assert "error" in evaluate_harness_loop([])

    def test_no_resume_attempts_returns_error(self):
        assert "error" in evaluate_checkpoint_resume([])


class TestSuccessRate:
    def test_all_completed(self):
        m = evaluate_harness_loop([_loop_result(), _loop_result()])
        assert m["task_success_rate"] == 1.0
        assert m["budget_exceeded_rate"] == 0.0

    def test_budget_exhaustion_is_not_success(self):
        """撞上预算上限的运行绝不能计入成功——它没有产出答案。"""
        m = evaluate_harness_loop([
            _loop_result(reason="max_steps", content=""),
            _loop_result(reason="token_budget", content=""),
        ])
        assert m["task_success_rate"] == 0.0
        assert m["budget_exceeded_rate"] == 1.0

    def test_mixed_outcomes(self):
        m = evaluate_harness_loop([
            _loop_result(reason="completed"),
            _loop_result(reason="completed"),
            _loop_result(reason="max_steps", content=""),
            _loop_result(reason="token_budget", content=""),
        ])
        assert m["task_success_rate"] == 0.5
        assert m["budget_exceeded_rate"] == 0.5


class TestToolCallMetrics:
    def test_efficiency_counts_failed_calls_against_it(self):
        m = evaluate_harness_loop([
            _loop_result(steps_records=[_record(ok=True), _record(ok=False, error="E")])
        ])
        assert m["total_tool_calls"] == 2
        assert m["tool_call_efficiency"] == 0.5

    def test_efficiency_is_none_when_no_tools_called(self):
        """没调用过工具时效率无定义，返回 None 而不是编一个 1.0。"""
        m = evaluate_harness_loop([_loop_result()])
        assert m["tool_call_efficiency"] is None

    def test_error_types_are_tallied(self):
        m = evaluate_harness_loop([
            _loop_result(steps_records=[
                _record(ok=False, error="UnknownTool"),
                _record(ok=False, error="UnknownTool"),
                _record(ok=False, error="McpToolError"),
            ])
        ])
        assert m["error_types"] == {"UnknownTool": 2, "McpToolError": 1}


class TestRecoveryRate:
    def test_recovery_rate_counts_only_runs_that_errored(self):
        m = evaluate_harness_loop([
            _loop_result(tool_errors=1, reason="completed"),   # 报错但完成 -> 恢复
            _loop_result(tool_errors=2, reason="max_steps", content=""),  # 报错且失败
            _loop_result(tool_errors=0),                        # 没报错，不进分母
        ])
        assert m["runs_with_tool_errors"] == 2
        assert m["recovery_rate"] == 0.5

    def test_recovery_rate_is_none_when_nothing_errored(self):
        m = evaluate_harness_loop([_loop_result(), _loop_result()])
        assert m["recovery_rate"] is None

    def test_full_recovery(self):
        m = evaluate_harness_loop([_loop_result(tool_errors=3, reason="completed")])
        assert m["recovery_rate"] == 1.0


class TestAverages:
    def test_averages(self):
        m = evaluate_harness_loop([
            _loop_result(steps=2, tokens=100),
            _loop_result(steps=4, tokens=300),
        ])
        assert m["avg_steps"] == 3.0
        assert m["avg_tokens"] == 200.0

    def test_completion_reasons_tallied(self):
        m = evaluate_harness_loop([
            _loop_result(reason="completed"),
            _loop_result(reason="completed"),
            _loop_result(reason="max_steps", content=""),
        ])
        assert m["completion_reasons"] == {"completed": 2, "max_steps": 1}


class TestDictCompatibility:
    def test_accepts_serialised_dicts(self):
        """评测产物常以 JSON 落盘，指标函数必须能吃回去。"""
        m = evaluate_harness_loop([
            {
                "exited_reason": "completed",
                "steps": 3,
                "tokens_used": 50,
                "tool_errors": 0,
                "transcript": [{"tool": "a", "ok": True}],
            }
        ])
        assert m["task_success_rate"] == 1.0
        assert m["total_tool_calls"] == 1
        assert m["avg_steps"] == 3.0

    def test_mixed_objects_and_dicts(self):
        m = evaluate_harness_loop([
            _loop_result(reason="completed"),
            {"exited_reason": "max_steps", "steps": 10, "transcript": []},
        ])
        assert m["total_runs"] == 2
        assert m["task_success_rate"] == 0.5


class TestCheckpointResume:
    def test_metrics(self):
        m = evaluate_checkpoint_resume([
            {"resumed": True, "lost_state": False, "duplicate_side_effects": False},
            {"resumed": True, "lost_state": True, "duplicate_side_effects": False},
            {"resumed": False, "lost_state": True, "duplicate_side_effects": True},
        ])
        assert m["total_attempts"] == 3
        assert m["resume_success_rate"] == pytest.approx(2 / 3, abs=1e-4)
        assert m["state_loss_rate"] == pytest.approx(2 / 3, abs=1e-4)
        assert m["duplicate_effect_rate"] == pytest.approx(1 / 3, abs=1e-4)

    def test_ignores_non_dict_entries(self):
        m = evaluate_checkpoint_resume([{"resumed": True}, "garbage", None])
        assert m["total_attempts"] == 1


class TestFormatReport:
    def test_renders_markdown_table(self):
        text = format_harness_report(
            evaluate_harness_loop([_loop_result(steps_records=[_record()])])
        )
        assert "Harness 过程指标" in text
        assert "任务成功率" in text
        assert "100.0%" in text

    def test_handles_missing_metrics(self):
        text = format_harness_report({"error": "No loop results"})
        assert "不可用" in text

    def test_renders_none_efficiency_readably(self):
        """None 不能渲染成 'None%' 或崩溃。"""
        text = format_harness_report(evaluate_harness_loop([_loop_result()]))
        assert "n/a" in text

    def test_appends_resume_section_when_provided(self):
        text = format_harness_report(
            evaluate_harness_loop([_loop_result()]),
            evaluate_checkpoint_resume([{"resumed": True}]),
        )
        assert "Checkpoint 续跑" in text
