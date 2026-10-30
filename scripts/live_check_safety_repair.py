"""用真实模型验证「冲突自动替换 + 诚实拒绝」这条链路。

离线测试全用假 LLM，能证明数据流是对的，但证明不了模型真的会照着
"替换这个动作"的指令改。这个脚本跑真实生成，把两种情况都走一遍：

  A. 健康用户           → 期望 safe_delivered
  B. 腰椎间盘突出用户   → 期望自动替换后交付；替换不掉则 safety_blocked

跑一次约 60~120 秒、产生真实费用，所以不进默认测试套件。

用法：
    python scripts/live_check_safety_repair.py [A|B|both]
"""

from __future__ import annotations

import json
import sys
import time

from src.core.orchestrator import Orchestrator
from src.models.schemas import UserProfileInput

HEALTHY = dict(height=178, weight=75, training_years=1.0, goal="增肌",
               available_equipment=["哑铃", "杠铃"],
               days_per_week=3, injuries=[])

LUMBAR = dict(height=178, weight=75, training_years=1.0, goal="增肌",
              available_equipment=["哑铃", "杠铃"],
              days_per_week=3, injuries=["腰椎间盘突出"])

QUERY = "帮我设计一份增肌计划，每次控制在 60 分钟内"


def run(label: str, profile_kwargs: dict) -> dict:
    print(f"\n{'=' * 68}\n[{label}] 开始真实生成 …\n{'=' * 68}", flush=True)
    started = time.time()
    orch = Orchestrator()
    profile = UserProfileInput(**profile_kwargs)
    result = orch.generate_plan(profile, QUERY, athlete_key=f"live-check-{label}")

    status = result.get("delivery_status")
    names = [ex.get("name", ex.get("exercise", ""))
             for day in (result.get("days") or [])
             for ex in day.get("exercises", [])]
    elapsed = round(time.time() - started, 1)

    print(f"\n  耗时        : {elapsed}s")
    print(f"  交付状态    : {status}")
    print(f"  rewrite 轮数: {result.get('rewrite_count')}")
    print(f"  days 条目   : {len(result.get('days') or [])}")
    print(f"  动作        : {'、'.join(names[:12]) or '(无)'}")

    if status == "safety_blocked":
        print(f"  拒绝原因    : {result.get('safety', {}).get('message', '')[:160]}")
        print(f"  下一步      : {result.get('safety', {}).get('next_step', '')[:160]}")
    if status == "generation_failed":
        print(f"  失败原因    : {result.get('failure', {}).get('reasons')}")
    warns = result.get("warnings") or []
    if warns:
        print(f"  警告({len(warns)})    : {warns[0][:120]}")

    return {"label": label, "status": status, "elapsed": elapsed,
            "rewrite_count": result.get("rewrite_count"),
            "exercises": names,
            "cautions": result.get("cautions") or [],
            "pending": result.get("review") or None}


def main() -> int:
    which = (sys.argv[1] if len(sys.argv) > 1 else "both").upper()
    runs = []
    if which in ("A", "BOTH"):
        runs.append(("A-健康用户", HEALTHY))
    if which in ("B", "BOTH"):
        runs.append(("B-腰突用户", LUMBAR))

    results = [run(label, kw) for label, kw in runs]

    print(f"\n{'=' * 68}\n汇总\n{'=' * 68}")
    for r in results:
        print(f"  {r['label']}: {r['status']}  ({r['elapsed']}s, "
              f"rewrite={r['rewrite_count']})")
    print("\n" + json.dumps(results, ensure_ascii=False, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
