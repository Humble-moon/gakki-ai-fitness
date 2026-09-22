"""Agent 自主循环的**过程**指标。

与 ``agent_metrics.py`` 的区别在于评的是什么：

* ``agent_metrics`` 评**结果**——计划对不对、安全检查准不准；
* 本模块评**过程**——花了多少步、工具调用有没有浪费、工具报错后能不能
  自己恢复、有没有撞上预算上限。

为什么要单独评过程：一个 agent 可能答对了，但用了 9 步、中途报错 4 次、
差点触顶。这种"险胜"在生产环境不可靠，而单看结果指标完全发现不了。
反过来，一个 agent 可能答案略有瑕疵但路径干净利落——过程指标能区分
这两种情况。

数据来源是 :class:`src.harness.loop.LoopResult`（或其序列化后的 dict）。
"""

from __future__ import annotations

from typing import Any, Iterable


def _as_dict(result: Any) -> dict:
    """兼容 LoopResult 对象与其序列化后的 dict。"""
    if isinstance(result, dict):
        return result
    transcript = getattr(result, "transcript", []) or []
    return {
        "content": getattr(result, "content", ""),
        "steps": getattr(result, "steps", 0),
        "tokens_used": getattr(result, "tokens_used", 0),
        "exited_reason": getattr(result, "exited_reason", "unknown"),
        "tool_errors": getattr(result, "tool_errors", 0),
        "transcript": [
            {
                "tool": getattr(r, "tool", ""),
                "ok": getattr(r, "ok", True),
                "error_type": getattr(r, "error_type", None),
            }
            for r in transcript
        ],
    }


def evaluate_harness_loop(results: Iterable[Any]) -> dict:
    """汇总一批自主循环运行的过程指标。

    Args:
        results: ``LoopResult`` 或等价 dict 的可迭代对象。

    Returns:
        指标 dict；无有效样本时返回 ``{"error": ...}``，与
        ``agent_metrics.py`` 的约定一致。

    指标含义：
        * ``task_success_rate``   —— 正常给出答案的比例。撞预算上限的不算成功。
        * ``budget_exceeded_rate``—— 步数或 token 触顶的比例。这个数字上升
          说明预算太紧，或任务难度超出当前 harness 的能力。
        * ``recovery_rate``       —— 出现工具报错后**仍然完成**的比例。
          衡量"错误回喂模型"这个机制到底有没有用。
        * ``tool_call_efficiency``—— 有效工具调用 / 全部调用。低于 1.0 说明
          模型在做无效调用（重复查同一个、参数写错）。
        * ``avg_steps``           —— 平均步数，用于观察任务复杂度与预算余量。
    """
    runs = [_as_dict(r) for r in results]
    if not runs:
        return {"error": "No loop results"}

    total = len(runs)
    completed = [r for r in runs if r.get("exited_reason") == "completed"]
    budget_hit = [
        r for r in runs if r.get("exited_reason") in {"max_steps", "token_budget"}
    ]

    all_steps = [s for r in runs for s in (r.get("transcript") or [])]
    ok_steps = [s for s in all_steps if s.get("ok")]
    errored_runs = [r for r in runs if (r.get("tool_errors") or 0) > 0]
    recovered_runs = [
        r for r in errored_runs if r.get("exited_reason") == "completed"
    ]

    return {
        "total_runs": total,
        "task_success_rate": round(len(completed) / total, 4),
        "budget_exceeded_rate": round(len(budget_hit) / total, 4),
        "avg_steps": round(sum(r.get("steps") or 0 for r in runs) / total, 2),
        "avg_tokens": round(sum(r.get("tokens_used") or 0 for r in runs) / total, 1),
        "total_tool_calls": len(all_steps),
        "tool_call_efficiency": (
            round(len(ok_steps) / len(all_steps), 4) if all_steps else None
        ),
        "runs_with_tool_errors": len(errored_runs),
        "recovery_rate": (
            round(len(recovered_runs) / len(errored_runs), 4)
            if errored_runs
            else None
        ),
        "completion_reasons": _count_by(runs, "exited_reason"),
        "error_types": _count_by([s for s in all_steps if not s.get("ok")], "error_type"),
    }


def _count_by(items: list[dict], key: str) -> dict:
    counts: dict[str, int] = {}
    for item in items:
        value = item.get(key)
        if value is None:
            continue
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def evaluate_checkpoint_resume(attempts: Iterable[dict]) -> dict:
    """汇总 checkpoint 续跑的成败。

    Args:
        attempts: 每项形如 ``{"resumed": bool, "lost_state": bool,
            "duplicate_side_effects": bool}``。``lost_state`` 指续跑后
            丢失了中断前的进度；``duplicate_side_effects`` 指同一份产物
            被重复投递——两者都是续跑正确性的关键失败模式。

    Returns:
        指标 dict；无样本时返回 ``{"error": ...}``。
    """
    rows = [a for a in attempts if isinstance(a, dict)]
    if not rows:
        return {"error": "No resume attempts"}

    total = len(rows)
    resumed = [r for r in rows if r.get("resumed")]
    lost = [r for r in rows if r.get("lost_state")]
    duplicated = [r for r in rows if r.get("duplicate_side_effects")]

    return {
        "total_attempts": total,
        "resume_success_rate": round(len(resumed) / total, 4),
        "state_loss_rate": round(len(lost) / total, 4),
        "duplicate_effect_rate": round(len(duplicated) / total, 4),
    }


def format_harness_report(metrics: dict, resume_metrics: dict | None = None) -> str:
    """把指标渲染成可直接贴进评测报告的 markdown 片段。"""
    lines = ["## Harness 过程指标", ""]

    if "error" in metrics:
        lines.append(f"循环指标不可用：{metrics['error']}")
    else:
        lines += [
            "| 指标 | 值 | 说明 |",
            "|---|---|---|",
            f"| 运行次数 | {metrics['total_runs']} | |",
            f"| 任务成功率 | {metrics['task_success_rate']:.1%} | 正常给出答案的比例 |",
            f"| 预算触顶率 | {metrics['budget_exceeded_rate']:.1%} | 越低越好；上升说明预算偏紧 |",
            f"| 平均步数 | {metrics['avg_steps']} | |",
            f"| 工具调用有效率 | {_fmt_opt(metrics.get('tool_call_efficiency'))} | 1.0 表示无无效调用 |",
            f"| 错误恢复率 | {_fmt_opt(metrics.get('recovery_rate'))} | 报错后仍完成的比例 |",
            "",
            f"终止原因分布：`{metrics.get('completion_reasons', {})}`",
        ]

    if resume_metrics and "error" not in resume_metrics:
        lines += [
            "",
            "### Checkpoint 续跑",
            "",
            f"- 续跑成功率：{resume_metrics['resume_success_rate']:.1%}",
            f"- 状态丢失率：{resume_metrics['state_loss_rate']:.1%}",
            f"- 重复副作用率：{resume_metrics['duplicate_effect_rate']:.1%}",
        ]

    return "\n".join(lines)


def _fmt_opt(value) -> str:
    return "n/a（无数据）" if value is None else f"{value:.1%}"
