"""摄入统计纯函数契约。

锁四件事：
* 空日 has_data=False、数值 None 而不是 0——折线不能跨过没记录的缺口；
* 日类型用实际训练日期判定，没有训练史时「未分类」+每日均值目标；
* 达标带 ±10%（不是配餐容差 ±50kcal）、蛋白命中率 90%；
* 汇总只按有记录的天数平均，空日不摊薄；当天标 is_partial。
"""
from __future__ import annotations

from datetime import date, timedelta

from src.core.diet_analytics import compute_diet_stats

TODAY = date(2026, 11, 19)


def _log(day: date, kcal: float, protein: float = 0.0, custom: float = 0.0) -> dict:
    return {"log_date": day.isoformat(), "kcal": kcal,
            "protein_g": protein, "custom_kcal": custom}


def _targets(train=2400, rest=2000, protein=140, daily=2200) -> dict:
    return {"training_day_kcal": train, "rest_day_kcal": rest,
            "protein_g": protein, "daily_kcal": daily}


class TestDaySeries:
    def test_window_is_continuous_with_gaps_marked(self):
        logs = [_log(TODAY - timedelta(days=2), 2100)]
        out = compute_diet_stats(logs, today=TODAY, window_days=5)
        assert len(out["days"]) == 5
        assert [d["date"] for d in out["days"]] == [
            (TODAY - timedelta(days=4 - i)).isoformat() for i in range(5)]
        empty = [d for d in out["days"] if not d["has_data"]]
        assert len(empty) == 4
        for d in empty:
            assert d["kcal"] is None and d["protein_g"] is None
            assert d["kcal_within_band"] is None and d["protein_hit"] is None

    def test_multiple_meals_same_day_sum(self):
        logs = [_log(TODAY, 600, 30), _log(TODAY, 800, 50), _log(TODAY, 700, 40)]
        day = compute_diet_stats(logs, today=TODAY, window_days=1)["days"][0]
        assert day["kcal"] == 2100 and day["protein_g"] == 120.0

    def test_today_is_partial(self):
        out = compute_diet_stats([_log(TODAY, 500)], today=TODAY, window_days=3)
        assert out["days"][-1]["is_partial"] is True
        assert all(not d["is_partial"] for d in out["days"][:-1])

    def test_logs_outside_window_ignored(self):
        logs = [_log(TODAY - timedelta(days=30), 9999)]
        out = compute_diet_stats(logs, today=TODAY, window_days=7)
        assert out["summary"]["days_logged"] == 0


class TestDayTypeClassification:
    def test_training_dates_decide_day_type(self):
        trained = TODAY - timedelta(days=1)
        logs = [_log(trained, 2350, 140), _log(TODAY, 2050, 140)]
        out = compute_diet_stats(logs, targets=_targets(),
                                 training_dates={trained.isoformat()},
                                 today=TODAY, window_days=2)
        by_date = {d["date"]: d for d in out["days"]}
        assert by_date[trained.isoformat()]["day_type"] == "训练日"
        assert by_date[trained.isoformat()]["target_kcal"] == 2400
        assert by_date[TODAY.isoformat()]["day_type"] == "休息日"
        assert by_date[TODAY.isoformat()]["target_kcal"] == 2000

    def test_training_dates_accepts_date_objects(self):
        trained = TODAY - timedelta(days=1)
        out = compute_diet_stats([], targets=_targets(), training_dates={trained},
                                 today=TODAY, window_days=2)
        assert out["days"][0]["day_type"] == "训练日"

    def test_no_training_history_is_unclassified_with_daily_target(self):
        out = compute_diet_stats([_log(TODAY, 2200)], targets=_targets(),
                                 today=TODAY, window_days=1)
        day = out["days"][0]
        assert day["day_type"] == "未分类"
        assert day["target_kcal"] == 2200
        assert out["summary"]["classified"] is False

    def test_empty_training_set_still_classifies_rest_days(self):
        """有训练史但窗口内没练过 ≠ 不知道：全是休息日。"""
        out = compute_diet_stats([_log(TODAY, 2000)], targets=_targets(),
                                 training_dates=set(), today=TODAY, window_days=1)
        assert out["days"][0]["day_type"] == "休息日"
        assert out["summary"]["classified"] is True


class TestBands:
    def test_within_ten_percent_band(self):
        logs = [_log(TODAY, 2200)]  # 目标 2200（未分类走 daily）
        out = compute_diet_stats(logs, targets=_targets(daily=2200),
                                 today=TODAY, window_days=1)
        assert out["days"][0]["kcal_within_band"] is True

    def test_band_boundary_inclusive(self):
        logs = [_log(TODAY, 2420)]  # 2200 + 10%
        out = compute_diet_stats(logs, targets=_targets(daily=2200),
                                 today=TODAY, window_days=1)
        assert out["days"][0]["kcal_within_band"] is True

    def test_outside_band(self):
        logs = [_log(TODAY, 1500)]
        out = compute_diet_stats(logs, targets=_targets(daily=2200),
                                 today=TODAY, window_days=1)
        assert out["days"][0]["kcal_within_band"] is False

    def test_small_target_band_floor_50kcal(self):
        """低热量目标（如 400kcal 加餐日）±10%=40 太苛刻，下限 50。"""
        logs = [_log(TODAY, 445)]
        out = compute_diet_stats(logs, targets=_targets(daily=400),
                                 today=TODAY, window_days=1)
        assert out["days"][0]["kcal_within_band"] is True

    def test_protein_hit_at_ninety_percent(self):
        logs = [_log(TODAY, 2200, protein=126)]  # 140 × 0.9
        out = compute_diet_stats(logs, targets=_targets(),
                                 today=TODAY, window_days=1)
        assert out["days"][0]["protein_hit"] is True

    def test_protein_miss_below_ninety(self):
        logs = [_log(TODAY, 2200, protein=100)]
        out = compute_diet_stats(logs, targets=_targets(),
                                 today=TODAY, window_days=1)
        assert out["days"][0]["protein_hit"] is False

    def test_no_targets_means_no_verdict(self):
        out = compute_diet_stats([_log(TODAY, 2200, 100)], today=TODAY, window_days=1)
        day = out["days"][0]
        assert day["target_kcal"] is None
        assert day["kcal_within_band"] is None and day["protein_hit"] is None
        assert out["summary"]["adherence_rate"] is None


class TestSummary:
    def test_averages_only_over_logged_days(self):
        logs = [_log(TODAY, 2000, 100), _log(TODAY - timedelta(days=1), 2400, 140)]
        out = compute_diet_stats(logs, targets=_targets(), training_dates=set(),
                                 today=TODAY, window_days=7)
        s = out["summary"]
        assert s["days_logged"] == 2
        assert s["total_kcal"] == 4400
        assert s["avg_kcal"] == 2200
        assert s["avg_protein_g"] == 120.0

    def test_adherence_and_protein_rates(self):
        d1, d2 = TODAY - timedelta(days=1), TODAY - timedelta(days=2)
        logs = [_log(d1, 2000, 140), _log(d2, 1200, 80)]  # 休息日目标 2000
        out = compute_diet_stats(logs, targets=_targets(), training_dates=set(),
                                 today=TODAY, window_days=3)
        s = out["summary"]
        assert s["adherence_rate"] == 0.5
        assert s["protein_hit_rate"] == 0.5

    def test_custom_share(self):
        logs = [_log(TODAY, 1000, custom=400), _log(TODAY, 1000, custom=0)]
        out = compute_diet_stats(logs, today=TODAY, window_days=1)
        assert out["summary"]["custom_share"] == 0.2

    def test_empty_logs_summary_nulls_not_zeros(self):
        out = compute_diet_stats([], today=TODAY, window_days=14)
        s = out["summary"]
        assert s["days_logged"] == 0 and s["total_kcal"] == 0
        assert s["avg_kcal"] is None and s["adherence_rate"] is None
        assert s["custom_share"] is None

    def test_window_clamped(self):
        out = compute_diet_stats([], today=TODAY, window_days=999)
        assert out["summary"]["window_days"] == 90
        out = compute_diet_stats([], today=TODAY, window_days=0)
        assert out["summary"]["window_days"] == 1