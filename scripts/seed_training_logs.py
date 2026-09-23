"""
生成演示用的训练日志种子数据。

用途：新库是空的，训练分析看板会是一片空白，面试演示时无从展示。本脚本
      造出一段「像真的」训练历史——重量渐进、每四周减载、中途有低谷、
      末尾留一个平台期，让调整建议面板必然有内容可讲。

设计要点：
    1. 走 TrainingLogStore.save_log()，与线上写入路径完全一致，不另写 INSERT。
    2. 用固定随机种子，重复运行结果一致（演示可复现）。
    3. 动作名取自 data/seed_exercises.json，保证与动作库对齐，聚合查询才对得上。

使用：
    python scripts/seed_training_logs.py --reset          # 清空后重灌
    python scripts/seed_training_logs.py --dry-run        # 只看生成计划不写库

灌完后在「训练日志」tab 点「载入演示数据」即可看到图表。
"""

import argparse
import json
import logging
import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("seed_training_logs")

# 默认绑定的运动员标识（前端「载入演示数据」按钮写的就是这个值）
DEFAULT_ATHLETE_KEY = "demo-athlete"

# 训练分化：每个训练日练哪些动作，以及基准重量与动作类型。
# 动作名必须与 data/seed_exercises.json 对齐——不一致会导致按动作名聚合时对不上。
SPLITS = [
    ("胸+三头", [("杠铃卧推", 60.0, "复合"), ("哑铃飞鸟", 14.0, "孤立"), ("绳索下压", 25.0, "孤立")]),
    ("背+二头", [("高位下拉", 55.0, "复合"), ("杠铃划船", 50.0, "复合"), ("哑铃弯举", 12.0, "孤立")]),
    ("腿", [("杠铃深蹲", 70.0, "复合"), ("腿举", 110.0, "复合"), ("哑铃罗马尼亚硬拉", 40.0, "复合")]),
    ("肩+核心", [("哑铃推举", 22.0, "复合"), ("哑铃侧平举", 8.0, "孤立"), ("绳索面拉", 20.0, "孤立")]),
]

# 每周的完成率基线。第 5-7 周安排一次低谷（模拟出差/生病），
# 这样图表上会出现明显可见的塌陷，也是自适应建议的触发点。
COMPLETION_BY_WEEK = [0.95, 0.93, 0.96, 0.90, 0.72, 0.62, 0.78, 0.90, 0.94, 0.96, 0.93, 0.95]

# 训练日固定排在周一/周二/周四/周五
TRAINING_WEEKDAYS = [0, 1, 3, 4]

# 平台期：最后几周让这个动作停止进步且疲劳度偏高，确保触发减载建议
PLATEAU_EXERCISE = "杠铃卧推"
PLATEAU_WEEKS = 3


def build_sessions(weeks: int, athlete_key: str, rng: random.Random) -> list[dict]:
    """生成 weeks 周的训练日志。

    数据模型分三层叠加，让曲线看起来像真实训练：
        1. 重量按周渐进（复合 +2%/周，孤立 +1%/周），每第 4 周减载 10%
        2. 次数随重量反向漂移——重量涨了次数自然回落，避免出现"又重又多次"的假数据
        3. 完成率与 RPE 带噪声波动，未完成的那次组数次数同步缩水
    """
    today = datetime.now().date()
    # 从 weeks 周前的那个周一开始排
    start = today - timedelta(weeks=weeks)
    start -= timedelta(days=start.weekday())

    sessions: list[dict] = []
    for week_index in range(weeks):
        completion_base = COMPLETION_BY_WEEK[week_index % len(COMPLETION_BY_WEEK)]
        is_deload_week = (week_index % 4 == 3)
        is_plateau = week_index >= weeks - PLATEAU_WEEKS

        for slot, weekday in enumerate(TRAINING_WEEKDAYS):
            day = start + timedelta(weeks=week_index, days=weekday)
            if day > today:
                continue

            focus, movements = SPLITS[slot % len(SPLITS)]
            completed_session = rng.random() < completion_base

            entries = []
            for name, base_weight, kind in movements:
                weight = _weight_for(base_weight, kind, week_index, is_deload_week)
                # 平台期：目标动作爬到峰值后逐周回退，制造「力量下滑 + 高疲劳」信号。
                # 注意基准要定格在进入平台期时的水平，否则基础递增会盖过回退。
                if is_plateau and name == PLATEAU_EXERCISE:
                    step = week_index - (weeks - PLATEAU_WEEKS)   # 0, 1, 2 ...
                    peak = base_weight * (1 + 0.02) ** (weeks - PLATEAU_WEEKS)
                    weight = round(peak * (1 - 0.04 * step), 1)

                reps = _reps_for(base_weight, weight)
                rpe = _rpe_for(base_weight, weight, is_deload_week, is_plateau,
                               name, rng)

                if completed_session:
                    sets = 4
                    done = True
                else:
                    # 没练完：组数砍半、次数打折，这样容量与完成率同步塌陷
                    sets = rng.choice([1, 2])
                    reps = max(4, reps - rng.randint(1, 3))
                    done = False

                entries.append({
                    "exercise_name": name, "sets": sets, "reps": reps,
                    "weight": weight, "rpe": rpe, "completed": done,
                })

            sessions.append({
                "log_date": day.isoformat(),
                "day_index": slot + 1,
                "focus": focus,
                "session_rpe": round(min(10.0, max(5.0, 6.5 + week_index * 0.12)), 1),
                "duration_min": rng.choice([55, 60, 65, 70]),
                "body_weight": round(80.0 + week_index * 0.15, 1),
                "notes": "",
                "entries": entries,
            })

    return sessions


def _weight_for(base_weight: float, kind: str, week_index: int,
                is_deload_week: bool) -> float:
    """按周渐进叠加周期性减载。复合动作涨得快、孤立动作涨得慢。"""
    weekly_gain = 0.02 if kind == "复合" else 0.01
    weight = base_weight * (1 + weekly_gain) ** week_index
    if is_deload_week:
        weight *= 0.90
    return round(weight, 1)


def _reps_for(base_weight: float, weight: float) -> int:
    """次数随重量反向漂移：重量涨上去，次数自然回落（双重渐进）。"""
    if base_weight <= 0:
        return 10
    ratio = weight / base_weight
    return max(6, min(12, int(round(12 - (ratio - 1) * 6))))


def _rpe_for(base_weight: float, weight: float, is_deload_week: bool,
             is_plateau: bool, name: str, rng: random.Random) -> float:
    """RPE 与相对负荷正相关，减载周偏低，平台期的目标动作偏高。"""
    ratio = weight / base_weight if base_weight else 1.0
    rpe = 6.0 + (ratio - 1) * 4 + rng.uniform(-0.5, 0.5)
    if is_deload_week:
        rpe -= 1.2
    if is_plateau and name == PLATEAU_EXERCISE:
        rpe = max(rpe, 8.5)
    return round(max(5.0, min(10.0, rpe)), 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="生成演示用训练日志")
    parser.add_argument("--athlete-key", default=DEFAULT_ATHLETE_KEY,
                        help=f"运动员标识（默认 {DEFAULT_ATHLETE_KEY}）")
    parser.add_argument("--weeks", type=int, default=12, help="生成多少周（默认 12）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    parser.add_argument("--reset", action="store_true", help="先清空该运动员的既有日志")
    parser.add_argument("--dry-run", action="store_true", help="只生成不写库")
    args = parser.parse_args()

    if args.weeks < 4:
        logger.error("至少需要 4 周数据，否则规则引擎无法判断趋势。")
        return 1

    rng = random.Random(args.seed)
    sessions = build_sessions(args.weeks, args.athlete_key, rng)

    total_entries = sum(len(s["entries"]) for s in sessions)
    logger.info("生成 %d 次训练、%d 条动作明细，时间范围 %s ~ %s",
                len(sessions), total_entries,
                sessions[0]["log_date"] if sessions else "-",
                sessions[-1]["log_date"] if sessions else "-")

    if args.dry_run:
        for s in sessions[:5]:
            logger.info("  %s %s: %s", s["log_date"], s["focus"],
                        "、".join(e["exercise_name"] for e in s["entries"]))
        logger.info("  ... 共 %d 次（--dry-run 未写库）", len(sessions))
        return 0

    from src.storage.training_log_store import TrainingLogStore

    store = TrainingLogStore()
    store.ensure_table()
    if not store.enabled:
        logger.error("训练日志存储不可用，请确认 PostgreSQL 已启动。")
        return 1

    if args.reset:
        removed = store.delete_logs(args.athlete_key)
        logger.info("已清空 %s 的 %d 条既有日志", args.athlete_key, removed)

    written = 0
    for session in sessions:
        log_id = store.save_log(
            athlete_key=args.athlete_key,
            entries=session["entries"],
            log_date=session["log_date"],
            day_index=session["day_index"],
            focus=session["focus"],
            session_rpe=session["session_rpe"],
            duration_min=session["duration_min"],
            body_weight=session["body_weight"],
            notes=session["notes"],
        )
        if log_id:
            written += 1

    logger.info("已写入 %d 条训练日志（运动员：%s）", written, args.athlete_key)

    _print_summary(args.athlete_key, store)
    logger.info("")
    logger.info("下一步：浏览器打开 http://localhost:8503 → 「训练日志」tab → 点「载入演示数据」")
    return 0


def _print_summary(athlete_key: str, store) -> None:
    """打印统计摘要，让脚本输出本身就能当演示素材。"""
    from src.core.training_analytics import compute_stats, rule_engine

    logs = store.get_logs(athlete_key, weeks=52)
    stats = compute_stats(logs)
    totals = stats["totals"]
    logger.info("")
    logger.info("统计摘要：%d 次训练，总容量 %s kg，平均完成率 %.0f%%，平均 RPE %s",
                totals["sessions"], f"{totals['volume']:,.0f}",
                totals["avg_completion"] * 100, totals["avg_rpe"])

    items = rule_engine(stats)
    if items:
        logger.info("规则引擎命中 %d 条建议（演示时点「生成调整建议」即可看到）：", len(items))
        for item in items:
            logger.info("  [%s] %s", item["type"], item["exercise_name"])
    else:
        logger.info("规则引擎未命中建议——可加大 --weeks 或调整种子。")


if __name__ == "__main__":
    raise SystemExit(main())
