"""训练统计聚合与规则引擎的纯函数测试（无 IO、无 LLM、无数据库）。"""

from datetime import date

import pytest

from src.core.training_analytics import (
    compute_muscle_volume,
    compute_records,
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
    # 参照日传最后一个有数据的周，否则周序列会一路补到"本周"，
    # 断言的长度就取决于跑测试的那一天
    stats = compute_stats(logs, today=date(2026, 9, 28))
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
    weeks = [w["week_start"] for w in compute_stats(logs, today=date(2026, 9, 28))["weekly"]]
    assert weeks == sorted(weeks)


# ----------------------------------------------------------------------
# 周序列的空缺与未结束周
#
# 折线图按数组下标连线，所以"周序列里有什么"直接决定图长什么样：
# 缺周被跳过 → 连续几周没练被画成平滑下滑；本周被当成整周 → 永远显示暴跌。
# ----------------------------------------------------------------------

def test_compute_stats_fills_missing_weeks_with_null_and_flag():
    """中间没记录的周要补成占位项，且数值为 None —— 0 会被读成"练了但容量为零"。"""
    logs = [
        _log("2026-09-07", [_entry("卧推", 4, 8, 60.0)]),
        # 09-14、09-21 两周完全没练
        _log("2026-09-28", [_entry("卧推", 4, 8, 60.0)]),
    ]
    weeks = compute_stats(logs, today=date(2026, 9, 28))["weekly"]
    assert [w["week_start"] for w in weeks] == [
        "2026-09-07", "2026-09-14", "2026-09-21", "2026-09-28",
    ]
    gap = weeks[1]
    assert gap["has_data"] is False
    assert gap["volume"] is None
    assert gap["completion"] is None
    assert gap["avg_rpe"] is None
    assert gap["sessions"] == 0
    assert weeks[0]["has_data"] is True


def test_compute_stats_extends_series_to_current_week():
    """尾部也要补到本周：只补中间不补尾巴，恰好把"最近三周没练"从图上删掉了。"""
    logs = [_log("2026-09-07", [_entry("卧推", 4, 8, 60.0)])]
    weeks = compute_stats(logs, today=date(2026, 10, 5))["weekly"]
    assert [w["week_start"] for w in weeks] == [
        "2026-09-07", "2026-09-14", "2026-09-21", "2026-09-28", "2026-10-05",
    ]
    assert all(w["has_data"] is False for w in weeks[1:])


def test_compute_stats_marks_only_current_week_as_partial():
    """只有包含参照日的那一周算"未结束"，已过去的周都是整周。"""
    logs = [
        _log("2026-09-28", [_entry("卧推", 4, 8, 60.0)]),
        _log("2026-10-05", [_entry("卧推", 4, 8, 60.0)]),
    ]
    weeks = compute_stats(logs, today=date(2026, 10, 7))["weekly"]   # 10-05 那周的周三
    assert [w["is_partial"] for w in weeks] == [False, True]


def test_compute_stats_marks_current_week_partial_even_without_records():
    """本周还没练也要标未结束：留空的当前周同样是"还没练完"，不是"零"。"""
    logs = [_log("2026-09-28", [_entry("卧推", 4, 8, 60.0)])]
    weeks = compute_stats(logs, today=date(2026, 10, 7))["weekly"]
    assert weeks[-1]["week_start"] == "2026-10-05"
    assert weeks[-1]["has_data"] is False
    assert weeks[-1]["is_partial"] is True


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


# ----------------------------------------------------------------------
# 肌群分布
# ----------------------------------------------------------------------

_MUSCLES = {
    "杠铃卧推": ["胸大肌", "三角肌前束", "肱三头肌"],
    "哑铃侧平举": ["三角肌中束"],
    "平板支撑": ["腹直肌"],          # 自重动作
    "深蹲": ["股四头肌", "臀大肌"],
}


def test_muscle_volume_weights_primary_higher_than_secondary():
    """主肌群 1.0、协同肌群 0.5：卧推 4 组 = 4 组胸 + 2 组肩 + 2 组三头。"""
    logs = [_log("2026-09-21", [_entry("杠铃卧推", 4, 8, 60.0)])]
    out = compute_muscle_volume(logs, _MUSCLES)
    by_group = {g["group"]: g["sets"] for g in out["groups"]}
    assert by_group["胸"] == 4.0
    assert by_group["肩"] == 2.0
    assert by_group["手臂"] == 2.0


def test_muscle_volume_counts_bodyweight_sets():
    """自重动作没有容量（kg=0），但组数必须算进去。

    只看容量的话，平板支撑会被算成"核心没练过"——而这张图存在的意义
    就是回答"我漏练了哪儿"，这里出错代价最直接。
    """
    logs = [_log("2026-09-21", [_entry("平板支撑", 3, 60, 0.0)])]
    out = compute_muscle_volume(logs, _MUSCLES)
    by_group = {g["group"]: g for g in out["groups"]}
    assert by_group["核心"]["sets"] == 3.0
    assert by_group["核心"]["volume"] == 0.0     # 容量确实是 0，不编造


def test_muscle_volume_share_sums_to_one():
    """各肌群占比合计恰好 100%：share 的分母是折算后总和，不是原始组数。"""
    logs = [
        _log("2026-09-21", [_entry("杠铃卧推", 4, 8, 60.0)]),
        _log("2026-09-22", [_entry("深蹲", 4, 8, 80.0)]),
        _log("2026-09-23", [_entry("哑铃侧平举", 3, 12, 8.0)]),
    ]
    out = compute_muscle_volume(logs, _MUSCLES)
    assert out["total_sets"] == 11.0            # 原始组数，不折算
    # 容差 1e-3：每组占比各自四舍五入到 4 位小数，累加会有约 1e-4 的偏差
    assert abs(sum(g["share"] for g in out["groups"]) - 1.0) < 1e-3


def test_muscle_volume_reports_unmatched_exercises_instead_of_dropping_them():
    """动作库查不到的动作要显式报出来，不能悄悄消失。"""
    logs = [_log("2026-09-21", [
        _entry("杠铃卧推", 4, 8, 60.0),
        _entry("某个库里没有的动作", 3, 10, 20.0),
    ])]
    out = compute_muscle_volume(logs, _MUSCLES)
    assert out["unmatched_exercises"] == ["某个库里没有的动作"]
    assert out["total_sets"] == 7.0
    assert out["coverage"] == pytest.approx(4 / 7, abs=1e-4)


def test_muscle_volume_coverage_is_one_when_everything_matches():
    logs = [_log("2026-09-21", [_entry("杠铃卧推", 4, 8, 60.0)])]
    assert compute_muscle_volume(logs, _MUSCLES)["coverage"] == 1.0


def test_muscle_volume_skips_incomplete_sets():
    """明确标为没做完的组不计入训练量。"""
    logs = [_log("2026-09-21", [_entry("杠铃卧推", 4, 8, 60.0, completed=False)])]
    out = compute_muscle_volume(logs, _MUSCLES)
    assert out["groups"] == []
    assert out["total_sets"] == 0.0


def test_muscle_volume_without_mapping_marks_everything_unmatched():
    """动作库不可用时，动作全进未匹配，而不是画出"什么都没练"的空图。"""
    logs = [_log("2026-09-21", [_entry("杠铃卧推", 4, 8, 60.0)])]
    out = compute_muscle_volume(logs, None)
    assert out["groups"] == []
    assert out["unmatched_exercises"] == ["杠铃卧推"]


def test_muscle_volume_sorted_by_sets_descending():
    logs = [
        _log("2026-09-21", [_entry("杠铃卧推", 2, 8, 60.0)]),
        _log("2026-09-22", [_entry("深蹲", 6, 8, 80.0)]),
    ]
    sets = [g["sets"] for g in compute_muscle_volume(logs, _MUSCLES)["groups"]]
    assert sets == sorted(sets, reverse=True)


# ----------------------------------------------------------------------
# 个人纪录
# ----------------------------------------------------------------------

def test_records_track_best_1rm_weight_and_volume_separately():
    """三项纪录分开记：重量涨了次数掉、次数涨了组数掉，各自都能被看见。"""
    logs = [
        _log("2026-09-14", [_entry("杠铃卧推", 3, 10, 60.0)]),   # 容量最大
        _log("2026-09-21", [_entry("杠铃卧推", 5, 5, 70.0)]),    # 最重
    ]
    rec = compute_records(logs)[0]
    # 60*(1+10/30)=80 > 70*(1+5/30)=81.67 → 后者 1RM 更高
    assert rec["best_1rm"] == 81.67
    assert rec["best_1rm_date"] == "2026-09-21"
    assert rec["best_weight"] == 70.0
    assert rec["best_volume"] == 3 * 10 * 60.0        # 1800 > 5*5*70=1750
    assert rec["best_volume_date"] == "2026-09-14"


def test_records_flag_recent_pr_only_when_latest_session_set_a_record():
    fresh = [_log("2026-09-14", [_entry("杠铃卧推", 3, 8, 60.0)]),
             _log("2026-09-21", [_entry("杠铃卧推", 3, 8, 70.0)])]
    assert compute_records(fresh)[0]["is_recent_pr"] is True

    stale = [_log("2026-09-14", [_entry("杠铃卧推", 3, 8, 70.0)]),
             _log("2026-09-21", [_entry("杠铃卧推", 3, 8, 60.0)])]
    assert compute_records(stale)[0]["is_recent_pr"] is False


def test_records_exclude_bodyweight_exercises():
    """自重动作 est_1rm 恒为 0，不该以"最佳 0kg"的形式混进纪录榜。"""
    logs = [_log("2026-09-21", [_entry("平板支撑", 3, 60, 0.0)])]
    assert compute_records(logs) == []


def test_records_skip_incomplete_entries():
    logs = [_log("2026-09-21", [_entry("杠铃卧推", 3, 8, 60.0, completed=False)])]
    assert compute_records(logs) == []


def test_records_recent_pr_ranks_first():
    logs = [
        # 深蹲更重，但最后一次没有刷新任何纪录
        _log("2026-09-07", [_entry("深蹲", 3, 5, 100.0)]),
        _log("2026-09-14", [_entry("深蹲", 3, 5, 90.0)]),
        # 卧推较轻，但最后一次刷了纪录
        _log("2026-09-07", [_entry("卧推", 3, 5, 40.0)]),
        _log("2026-09-14", [_entry("卧推", 3, 5, 45.0)]),
    ]
    records = compute_records(logs)
    assert records[0]["name"] == "卧推"
    assert records[0]["is_recent_pr"] is True
    assert records[1]["is_recent_pr"] is False


def test_records_do_not_flag_first_ever_session_as_pr():
    """只练过一次的动作不标 PR：否则新用户每加一个动作就点亮一次。"""
    logs = [_log("2026-09-14", [_entry("卧推", 3, 5, 60.0)])]
    rec = compute_records(logs)[0]
    assert rec["best_1rm"] > 0          # 纪录本身照样记下来
    assert rec["is_recent_pr"] is False


def test_records_do_not_flag_first_ever_session_as_pr():
    """只练过一次的动作不标 PR：否则新用户每加一个动作就点亮一次。"""
    logs = [_log("2026-09-14", [_entry("卧推", 3, 5, 60.0)])]
    rec = compute_records(logs)[0]
    assert rec["best_1rm"] > 0          # 纪录本身照样记下来
    assert rec["is_recent_pr"] is False


def test_muscle_volume_lists_untrained_groups_as_zero():
    """没练到的肌群也要出现（0 组长度），否则"没练"和"没统计到"分不出来。

    这张图存在的意义就是回答"我漏练了哪儿"——把零组那一格整个省掉，
    读的人只能看到一个更短的列表，看不出少了什么。
    """
    logs = [_log("2026-09-21", [_entry("杠铃卧推", 4, 8, 60.0)])]
    out = compute_muscle_volume(logs, _MUSCLES)
    by_group = {g["group"]: g["sets"] for g in out["groups"]}
    assert by_group["胸"] == 4.0
    assert by_group["腿"] == 0.0          # 没练，但要露出来
    assert by_group["核心"] == 0.0
    # 有数据的排在前面
    assert out["groups"][0]["group"] == "胸"
    assert out["groups"][-1]["sets"] == 0.0


def test_muscle_volume_omits_other_group_when_empty():
    """"其他"不是肌群，没数据时列出来只是占位。"""
    logs = [_log("2026-09-21", [_entry("杠铃卧推", 4, 8, 60.0)])]
    groups = [g["group"] for g in compute_muscle_volume(logs, _MUSCLES)["groups"]]
    assert "其他" not in groups


def test_muscle_volume_returns_empty_when_nothing_could_be_attributed():
    """一组都没归出去时返回空，而不是 7 根零长度的条——那会被读成"全都练了 0 组"。"""
    logs = [_log("2026-09-21", [_entry("库里没有的动作", 4, 8, 60.0)])]
    out = compute_muscle_volume(logs, _MUSCLES)
    assert out["groups"] == []
    assert out["total_sets"] == 4.0
    assert out["coverage"] == 0.0
