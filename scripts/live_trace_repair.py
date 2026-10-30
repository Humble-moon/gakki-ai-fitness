"""带逐轮追踪的真实生成：看清重写循环到底做了什么。

live_check_safety_repair.py 只给终态。这里把每一轮检查的冲突动作、
重写后的动作全打出来，用来回答一个具体问题：
**自动替换为什么没救回来？** 是模型没照做，还是可选动作被禁忌表压得太窄。

用法：python scripts/live_trace_repair.py ["伤病描述"]
"""

from __future__ import annotations

import json
import sys
import time

from src.agents.fact_checker import FactCheckerAgent
from src.core.orchestrator import Orchestrator
from src.models.schemas import UserProfileInput

INJURY = sys.argv[1] if len(sys.argv) > 1 else "腰椎间盘突出"


def names_of(plan: dict) -> list[str]:
    return [ex.get("name", ex.get("exercise", ""))
            for day in (plan.get("days") or [])
            for ex in day.get("exercises", [])]


def main() -> int:
    original = FactCheckerAgent.check
    round_no = {"n": 0}

    def traced(self, plan, profile):
        round_no["n"] += 1
        result = original(self, plan, profile)
        print(f"\n--- 第 {round_no['n']} 次检查 ---", flush=True)
        print("  计划动作:", "、".join(names_of(plan)) or "(空)")
        print("  冲突动作:", result.get("conflict_exercises") or "无")
        print("  is_safe :", result.get("is_safe"))
        print("  规则明细:", (result.get("review_suggestions") or [])[:2])
        return result

    FactCheckerAgent.check = traced  # type: ignore[method-assign]

    profile = UserProfileInput(
        height=178, weight=75, training_years=1.0, goal="增肌",
        available_equipment=["哑铃", "杠铃"], days_per_week=3,
        injuries=[INJURY])

    started = time.time()
    print(f"伤病: {INJURY}  开始 …", flush=True)
    result = Orchestrator().generate_plan(
        profile, "帮我设计一份增肌计划，每次控制在 60 分钟内",
        athlete_key="live-trace")

    print(f"\n{'=' * 60}")
    print("耗时:", round(time.time() - started, 1), "s")
    print("终态:", result.get("delivery_status"))
    print("rewrite_count:", result.get("rewrite_count"))
    print("交付动作:", "、".join(names_of(result)) or "(无)")
    if result.get("safety"):
        print("拒绝文案:", result["safety"].get("message"))
    print(json.dumps({k: result.get(k) for k in
                      ("rewrite_count", "_delivery_allowed", "_delivery_block_reasons")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
