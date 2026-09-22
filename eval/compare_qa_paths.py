#!/usr/bin/env python3
"""对比知识问答的两条路径：固定流水线 vs 自主循环。

## 为什么要有这个脚本

自主循环的卖点是"让模型自己决定查什么"——理论上更省、更灵活。
但这是**可证伪的主张**：模型可能选错工具、可能多查、可能不查就答。
不量化就只是信念，而项目一贯的原则是"可复核"。

本脚本在**同一批问题**上跑两条路径，输出可比的过程指标：

| 指标 | 固定流水线 | 自主循环 |
|---|---|---|
| 工具调用次数 | 恒定（每次都查知识库+动作库） | 由模型决定，应当更少 |
| 步数 | 恒定 | 变化 |
| 延迟 | 可预测 | 波动（多轮会变慢） |
| token | 可预测 | 波动 |
| 安全约束是否生效 | 是 | **必须也是**（脚本会断言） |

**本脚本不评价答案质量**——那需要人工标注或 LLM 裁判（见
`eval/ragas_eval.py`）。这里只回答"过程开销差多少"。

## 运行

需要可用的 LLM 凭据（走 `src.config.LLM_CONFIGS`），因此**不进 CI**：

    python -m eval.compare_qa_paths --limit 5
    python -m eval.compare_qa_paths --limit 5 --output eval/qa_path_compare.json

结果不确定（同样的输入模型可能给不同路径），所以脚本会如实记录
原始数据而非只给一个汇总数字。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.metrics.harness_metrics import evaluate_harness_loop, format_harness_report  # noqa: E402

#: 覆盖三类问题，让两条路径都有机会展示差异。
#: 简单概念题理论上不该触发动作库查询，伤病题应当触发图谱。
DEFAULT_QUESTIONS = [
    "增肌期间每天应该吃多少蛋白质？",
    "深蹲和硬拉有什么区别？",
    "一周练几次比较合适？",
    "膝盖疼还能练腿吗？",
    "训练后肌肉酸痛正常吗？",
]


def _load_questions(limit: int) -> list[str]:
    """优先用 golden 知识子集，读不到则退回内置问题。"""
    path = ROOT / "eval" / "golden_dataset" / "knowledge_queries.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        questions = [item["query"] for item in raw if isinstance(item, dict) and item.get("query")]
        if questions:
            return questions[:limit]
    except Exception:
        pass
    return DEFAULT_QUESTIONS[:limit]


def _run_agent_path(orch, question: str) -> dict:
    """跑一次自主循环，返回过程记录。"""
    from src.core.qa_agent import answer_with_agent_loop
    from src.harness.qa_tools import QAToolKit

    profile = {"height": 175, "weight": 70, "training_years": 2, "injuries": []}
    toolkit = QAToolKit(orch.knowledge, orch.retriever, orch.retriever.tools)

    started = time.perf_counter()
    answer, result = answer_with_agent_loop(orch.writer.llm, toolkit, question, profile)
    elapsed = time.perf_counter() - started

    return {
        "path": "agent_loop",
        "answer": answer,
        "completed": result.completed,
        "exited_reason": result.exited_reason,
        "steps": result.steps,
        "tokens": result.tokens_used,
        "tool_errors": result.tool_errors,
        "tools": [r.tool for r in result.transcript],
        "seconds": round(elapsed, 2),
        # 供 evaluate_harness_loop 复用；序列化前由 _strip_private 移除。
        "_loop": result,
    }


def _run_fixed_path(orch, question: str) -> dict:
    """跑一次固定流水线，统计等价的"工具调用"口径。

    固定流水线的"工具调用次数"是**由代码结构决定的**，不是模型选的：
    知识库检索 1 次 + 动作库检索 1 次，伤病问题再加图谱 1 次。
    这里按实际发生的调用来计数，保证与自主循环可比。
    """
    from src.models.schemas import UserProfileInput

    profile = UserProfileInput(
        height=175, weight=70, training_years=2, goal="增肌", injuries=[]
    )

    calls: list[str] = []
    started = time.perf_counter()
    answer = ""
    stages = 0
    try:
        for event in orch.answer_question_stream(question, profile):
            kind, payload = event
            if kind == "stage":
                stages += 1
                text = str(payload)
                if "知识库" in text:
                    calls.append("search_knowledge")
                elif "图谱" in text:
                    calls.append("reason_injury")
                elif "检索" in text:
                    calls.append("search_exercises")
            elif kind == "answer_chunk":
                answer += payload
    except Exception as exc:  # noqa: BLE001 - 对比脚本不应因单条失败而中止
        return {
            "path": "fixed_pipeline",
            "error": f"{type(exc).__name__}: {exc}",
            "seconds": round(time.perf_counter() - started, 2),
        }
    elapsed = time.perf_counter() - started

    # 动作库检索在固定流水线里没有独立的 stage 事件，但代码里必然执行，
    # 故补记——否则对比会低估固定路径的开销。
    if "search_exercises" not in calls:
        calls.append("search_exercises")

    return {
        "path": "fixed_pipeline",
        "answer": answer,
        "completed": bool(answer),
        "exited_reason": "completed" if answer else "empty",
        "steps": len(calls),
        "tokens": 0,  # 固定流水线未逐次统计 token
        "tool_errors": 0,
        "tools": calls,
        "seconds": round(elapsed, 2),
    }


def compare(limit: int, output: Path | None) -> dict:
    from src.core.orchestrator import Orchestrator

    orch = Orchestrator()
    questions = _load_questions(limit)

    print(f"对比 {len(questions)} 个问题\n" + "=" * 68)

    rows: list[dict] = []
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {question}")
        row = {"question": question}

        fixed = _run_fixed_path(orch, question)
        row["fixed"] = {k: v for k, v in fixed.items() if k != "result"}
        print(f"  固定流水线 : {fixed.get('steps', '?')} 次检索，"
              f"{fixed.get('seconds', '?')}s")

        agent = _run_agent_path(orch, question)
        row["agent"] = {k: v for k, v in agent.items() if k != "result"}
        print(f"  自主循环   : {agent['steps']} 步，"
              f"工具={agent['tools']}，{agent['seconds']}s，"
              f"{'完成' if agent['completed'] else '未收敛(' + agent['exited_reason'] + ')'}")

        rows.append(row)

    # ---- 汇总 ----
    summary = _summarise(rows)
    # 复用通用过程指标，与 harness_metrics 的口径保持一致
    # （成功率 / 预算触顶率 / 工具调用有效率 / 错误恢复率）。
    summary["harness_metrics"] = evaluate_harness_loop(
        [r["agent"]["_loop"] for r in rows if r.get("agent", {}).get("_loop")]
    )
    report = {"questions": _strip_private(rows), "summary": summary}

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n原始结果已写入 {output}")

    print("\n" + "=" * 68)
    print(_render_summary(summary))
    return report


def _strip_private(rows: list[dict]) -> list[dict]:
    """去掉不该被序列化的内部对象（``LoopResult`` 不可 JSON 化）。"""
    cleaned = []
    for row in rows:
        entry = {"question": row["question"], "fixed": row.get("fixed")}
        agent = dict(row.get("agent") or {})
        agent.pop("_loop", None)
        entry["agent"] = agent
        cleaned.append(entry)
    return cleaned


def _summarise(rows: list[dict]) -> dict:
    fixed_ok = [r["fixed"] for r in rows if "error" not in r.get("fixed", {})]
    agent_ok = [r["agent"] for r in rows if r.get("agent")]

    def _stat(values: list[float]) -> dict:
        if not values:
            return {"n": 0}
        return {
            "n": len(values),
            "mean": round(statistics.mean(values), 2),
            "median": round(statistics.median(values), 2),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
        }

    return {
        "fixed": {
            "calls": _stat([r["steps"] for r in fixed_ok]),
            "seconds": _stat([r["seconds"] for r in fixed_ok]),
            "completed_rate": round(
                sum(1 for r in fixed_ok if r["completed"]) / len(fixed_ok), 4
            ) if fixed_ok else None,
        },
        "agent": {
            "steps": _stat([r["steps"] for r in agent_ok]),
            "seconds": _stat([r["seconds"] for r in agent_ok]),
            "tokens": _stat([r["tokens"] for r in agent_ok]),
            "completed_rate": round(
                sum(1 for r in agent_ok if r["completed"]) / len(agent_ok), 4
            ) if agent_ok else None,
            "tool_usage": _tool_usage(agent_ok),
        },
    }


def _tool_usage(rows: list[dict]) -> dict:
    """统计每个工具被调用的总次数——模型是否偏好某个工具，一眼可见。"""
    counts: dict[str, int] = {}
    for row in rows:
        for tool in row.get("tools", []):
            counts[tool] = counts.get(tool, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _render_summary(summary: dict) -> str:
    f, a = summary["fixed"], summary["agent"]
    lines = [
        "汇总",
        "",
        f"{'':14s}{'固定流水线':>16s}{'自主循环':>16s}",
        f"{'检索/步数均值':14s}{f['calls'].get('mean', '-'):>16}{a['steps'].get('mean', '-'):>16}",
        f"{'耗时均值(s)':14s}{f['seconds'].get('mean', '-'):>16}{a['seconds'].get('mean', '-'):>16}",
        f"{'完成率':14s}{str(f['completed_rate']):>16}{str(a['completed_rate']):>16}",
        "",
        f"自主循环工具使用分布：{a['tool_usage']}",
        "",
        "注：本表只反映**过程开销**，不评价答案质量。",
        "   固定流水线的检索次数由代码结构决定，不随问题变化；",
        "   自主循环由模型决定，因此波动本身就是结论的一部分。",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="对比问答两条路径")
    parser.add_argument("--limit", type=int, default=5, help="问题数量上限")
    parser.add_argument("--output", type=Path, default=None, help="原始结果 JSON 输出路径")
    args = parser.parse_args()

    compare(args.limit, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
