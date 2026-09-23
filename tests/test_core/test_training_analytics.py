"""训练统计聚合与规则引擎的纯函数测试（无 IO、无 LLM、无数据库）。"""

import pytest

from src.core.training_analytics import (
    compute_stats,
    estimate_1rm,
    round_to_step,
    rule_engine,
    volume_of,
    week_start_of,
)


# ----------------------------------------------------------------------
# 构造工具
# ----------------------------------------------------------------------

def _entry(name, sets, reps, weight, rpe=None, completed=True):
    return {"exercise_name": name, "sets": sets, "reps": reps,
            "weight": weight, "rpe": rpe, "completed": completed}


def _log(date_str, entries, session_rpe=None):
    return {"log_date": date_str, "session_rpe": session_rpe, "entries": entries}


# ----------------------------------------------------------------------
# 基础计算
# ----------------------------------------------------------------------

def test_estimate_1rm_uses_epley_formula():
    # 80kg × 8 次 → 80 * (1 + 8/30) = 101.33
    assert estimate_1rm(80.0, 8) == 101.33


@pytest.mark.parametrize("weight,reps", [(0, 8), (80, 0), (-5, 8), (80, -1)])
def test_estimate_1rm_returns_zero_for_meaningless_input(weight, reps):
    """自重动作（weight=0）与缺失次数都应返回 0，而不是抛异常。"""
    assert estimate_1rm(weight, reps) == 0.0


def test_estimate_1rm_rises_with_reps_at_same_weight():
    """同重量下次数越多代表力量越强，估算 1RM 必须单调递增。"""
    assert estimate_1rm(60.0, 12) > estimate_1rm(60.0, 8)


def test_volume_of_multiplies_sets_reps_weight():
    assert volume_of(4, 8, 60.0) == 1920.0


def test_volume_of_returns_zero_for_bodyweight():
    assert volume_of(4, 8, 0.0) == 0.0


def test_week_start_of_monday_is_itself():
    from datetime import date
    assert week_start_of(date(2026, 9, 21)) == date(2026, 9, 21)  # 周一


def test_week_start_of_sunday_rolls_back_to_monday():
    from datetime import date
    assert week_start_of(date(2026, 9, 27)) == date(2026, 9, 21)  # 周日 → 本周一


@pytest.mark.parametrize("raw,expected", [(61.0, 60.0), (62.0, 62.5), (100.0, 100.0)])
def test_round_to_step_snaps_to_weight_plates(raw, expected):
    assert round_to_step(raw) == expected


# ----------------------------------------------------------------------
# compute_stats
# ----------------------------------------------------------------------

def test_compute_stats_returns_complete_shell_for_empty_input():
    """空输入返回结构完整的空壳而非 None，前端渲染逻辑不必到处判空。"""
    stats = compute_stats([])
    assert stats["totals"]["sessions"] == 0
    assert stats["totals"]["volume"] == 0.0
    assert stats["weekly"] == []
    assert stats["exercises"] == []


def test_compute_stats_aggregates_totals():
    logs = [
        _log("2026-09-21", [_entry("卧推", 4, 8, 60.0, rpe=7.0)]),
        _log("2026-09-22", [_entry("深蹲", 4, 8, 80.0, rpe=8.0)]),
    ]
    stats = compute_stats(logs)
    assert stats["totals"]["sessions"] == 2
    # 4*8*60 + 4*8*80 = 1920 + 2560 = 4480
    assert stats["totals"]["volume"] == 4480.0
    assert stats["totals"]["avg_completion"] == 1.0
    assert stats["totals"]["avg_rpe"] == 7.5


def test_compute_stats_groups_same_week_into_one_bucket():
    logs = [
        _log("2026-09-21", [_entry("卧推", 4, 8, 60.0)]),   # 周一
        _log("2026-09-24", [_entry("卧推", 4, 8, 60.0)]),   # 同周周四
        _log("2026-09-28", [_entry("卧推", 4, 8, 60.0)]),   # 下周一
    ]
    stats = compute_stats(logs)
    assert len(stats["weekly"]) == 2
    assert stats["weekly"][0]["week_start"] == "2026-09-21"
    assert stats["weekly"][0]["sessions"] == 2
    assert stats["weekly"][1]["sessions"] == 1


def test_compute_stats_orders_weekly_buckets_chronologically():
    logs = [
        _log("2026-09-28", [_entry("卧推", 4, 8, 60.0)]),
        _log("2026-09-14", [_entry("卧推", 4, 8, 60.0)]),
        _log("2026-09-21", [_entry("卧推", 4, 8, 60.0)]),
    ]
    weeks = [w["week_start"] for w in compute_stats(logs)["weekly"]]
    assert weeks == sorted(weeks)


def test_compute_stats_builds_exercise_series_in_chronological_order():
    logs = [
        _log("2026-09-28", [_entry("深蹲", 4, 8, 85.0)]),
        _log("2026-09-14", [_entry("深蹲", 4, 8, 75.0)]),
        _log("2026-09-21", [_entry("深蹲", 4, 8, 80.0)]),
    ]
    ex = compute_stats(logs)["exercises"][0]
    assert ex["name"] == "深蹲"
    assert [s["weight"] for s in ex["series"]] == [75.0, 80.0, 85.0]
    assert ex["first_weight"] == 75.0
    assert ex["last_weight"] == 85.0


def test_compute_stats_computes_positive_trend():
    logs = [
        _log("2026-09-14", [_entry("深蹲", 4, 8, 80.0)]),
        _log("2026-09-21", [_entry("深蹲", 4, 8, 88.0)]),
    ]
    ex = compute_stats(logs)["exercises"][0]
    assert ex["trend_pct"] > 0


def test_compute_stats_trend_is_none_for_bodyweight_exercise():
    """自重动作 est_1rm 恒为 0，比较无意义，趋势必须是 None 而不是 0。"""
    logs = [
        _log("2026-09-14", [_entry("引体向上", 4, 8, 0)]),
        _log("2026-09-21", [_entry("引体向上", 4, 9, 0)]),
    ]
    assert compute_stats(logs)["exercises"][0]["trend_pct"] is None


def test_compute_stats_ignores_entries_without_exercise_name():
    logs = [_log("2026-09-21", [_entry("", 4, 8, 60.0), _entry("卧推", 4, 8, 60.0)])]
    stats = compute_stats(logs)
    assert len(stats["exercises"]) == 1


def test_compute_stats_handles_unparseable_date_without_crashing():
    stats = compute_stats([_log("not-a-date", [_entry("卧推", 4, 8, 60.0)])])
    assert stats["weekly"] == []
    assert stats["totals"]["sessions"] == 1


# ----------------------------------------------------------------------
# rule_engine — 四条规则
# ----------------------------------------------------------------------

def _stats_for(exercise_name, series, rpes, completions):
    """构造单个动作的 stats，供规则测试使用。"""
    logs = [
        _log(day, [_entry(exercise_name, 4, reps, weight, rpe=rpe, completed=done)])
        for (day, weight, reps), rpe, done in zip(series, rpes, completions)
    ]
    return compute_stats(logs)


def test_rule_engine_recommends_deload_on_strength_drop_with_high_rpe():
    stats = _stats_for(
        "深蹲",
        [("2026-09-07", 80.0, 8), ("2026-09-14", 80.0, 8), ("2026-09-21", 74.0, 8)],
        [8.5, 8.6, 8.7],
        [True, True, True],
    )
    items = rule_engine(stats)
    assert len(items) == 1
    assert items[0]["type"] == "deload"
    assert items[0]["suggested"]["weight"] < 74.0


def test_rule_engine_does_not_deload_when_rpe_is_low():
    """力量下滑但 RPE 不高，说明不是疲劳导致的，不该减载。"""
    stats = _stats_for(
        "深蹲",
        [("2026-09-07", 80.0, 8), ("2026-09-14", 80.0, 8), ("2026-09-21", 74.0, 8)],
        [6.0, 6.2, 6.1],
        [True, True, True],
    )
    assert all(i["type"] != "deload" for i in rule_engine(stats))


def test_deload_judges_recent_fatigue_not_lifetime_average():
    """全期 RPE 被前期低负荷拉低时，仍要依据近期 RPE 判出减载。

    这是一个真实踩过的坑：12 周数据里前 9 周 RPE 都在 6 上下，只有最后三次
    升到 8.5+，若用全期平均（约 6.6）判断就会漏掉这个最该减载的动作。
    """
    stats = _stats_for(
        "卧推",
        [("2026-08-01", 60.0, 10), ("2026-08-08", 62.0, 10), ("2026-08-15", 64.0, 9),
         ("2026-08-22", 66.0, 9), ("2026-08-29", 68.0, 8), ("2026-09-05", 70.0, 8),
         ("2026-09-12", 68.0, 8), ("2026-09-19", 65.0, 8)],
        [6.0, 6.0, 6.0, 6.0, 6.0, 8.6, 8.8, 9.0],
        [True] * 8,
    )
    ex = stats["exercises"][0]
    assert ex["avg_rpe"] < 8.0        # 全期平均被前期低负荷拉低
    assert ex["recent_rpe"] >= 8.0    # 但近期确实很吃力
    assert any(i["type"] == "deload" for i in rule_engine(stats))


def test_rule_engine_recommends_progress_when_room_remains():
    stats = _stats_for(
        "卧推",
        [("2026-09-14", 60.0, 8), ("2026-09-21", 62.0, 8)],
        [6.0, 6.0],
        [True, True],
    )
    items = rule_engine(stats)
    assert len(items) == 1
    assert items[0]["type"] == "progress"
    assert items[0]["suggested"]["weight"] > 62.0


def test_rule_engine_does_not_progress_without_enough_data():
    """只练过一次算不出趋势，不该建议加重。"""
    stats = _stats_for("卧推", [("2026-09-21", 60.0, 8)], [6.0], [True])
    assert all(i["type"] != "progress" for i in rule_engine(stats))


def test_rule_engine_does_not_progress_when_trend_is_negative():
    stats = _stats_for(
        "卧推",
        [("2026-09-14", 70.0, 8), ("2026-09-21", 60.0, 8)],
        [6.0, 6.0],
        [True, True],
    )
    assert all(i["type"] != "progress" for i in rule_engine(stats))


def test_rule_engine_recommends_volume_cut_on_low_completion():
    logs = [_log("2026-09-21", [
        _entry("划船", 4, 8, 50.0, rpe=7.0, completed=True),
        _entry("划船", 4, 8, 50.0, rpe=7.0, completed=False),
        _entry("划船", 4, 8, 50.0, rpe=7.0, completed=False),
    ])]
    items = rule_engine(compute_stats(logs))
    assert len(items) == 1
    assert items[0]["type"] == "volume_cut"
    assert items[0]["suggested"]["sets_delta"] == -1


def test_rule_engine_recommends_swap_for_rarely_performed_exercise():
    stats = _stats_for("硬拉", [("2026-09-21", 100.0, 5)], [7.0], [True])
    items = rule_engine(stats)
    assert len(items) == 1
    assert items[0]["type"] == "swap"


def test_rule_engine_emits_at_most_one_item_per_exercise():
    """四条规则互斥：同一动作不能在建议面板里出现两条互相打架的建议。"""
    logs = [_log("2026-09-21", [
        _entry("卧推", 4, 8, 60.0, rpe=9.0, completed=False),
        _entry("深蹲", 4, 8, 80.0, rpe=6.0, completed=True),
        _entry("硬拉", 4, 5, 100.0, rpe=7.0, completed=True),
    ])]
    items = rule_engine(compute_stats(logs))
    names = [i["exercise_name"] for i in items]
    assert len(names) == len(set(names))


def test_rule_engine_returns_empty_for_empty_stats():
    assert rule_engine(compute_stats([])) == []


def test_rule_item_ids_are_stable_across_calls():
    """item_id 必须稳定，apply 时靠它校验用户勾选的条目是否仍有效。"""
    stats = _stats_for(
        "卧推",
        [("2026-09-14", 60.0, 8), ("2026-09-21", 62.0, 8)],
        [6.0, 6.0],
        [True, True],
    )
    assert [i["item_id"] for i in rule_engine(stats)] == \
           [i["item_id"] for i in rule_engine(stats)]


def test_rule_items_carry_reason_and_confidence():
    stats = _stats_for(
        "卧推",
        [("2026-09-14", 60.0, 8), ("2026-09-21", 62.0, 8)],
        [6.0, 6.0],
        [True, True],
    )
    item = rule_engine(stats)[0]
    assert item["reason"]            # LLM 不可用时也要有确定性文案
    assert 0.0 < item["confidence"] <= 0.9


def test_progress_delta_never_exceeds_declared_cap_for_heavy_loads():
    """重载动作按比例递增时，增幅不应超过声明的上限。"""
    stats = _stats_for(
        "深蹲",
        [("2026-09-14", 100.0, 5), ("2026-09-21", 102.0, 5)],
        [6.0, 6.0],
        [True, True],
    )
    item = rule_engine(stats)[0]
    assert item["type"] == "progress"
    assert item["delta_pct"] <= 5.0
