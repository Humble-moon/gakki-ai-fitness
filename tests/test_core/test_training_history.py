"""训练历史上下文的测试。用假存储替身，不需要 PostgreSQL 或 Redis。"""

import json
from datetime import datetime, timedelta, timezone

from src.core.training_history import (
    ADJUSTMENT_KEY,
    build_training_context,
    load_adjustments,
    save_adjustments,
)


# ----------------------------------------------------------------------
# 替身
# ----------------------------------------------------------------------

class FakeStore:
    def __init__(self, logs=None, enabled=True, raises=False):
        self._logs = logs or []
        self.enabled = enabled
        self._raises = raises
        self.calls = []

    def get_logs(self, athlete_key, weeks=12, **kwargs):
        self.calls.append((athlete_key, weeks))
        if self._raises:
            raise RuntimeError("boom")
        return self._logs


class FakeRedis:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)


class FakeLongTerm:
    def __init__(self):
        self.prefix = "memory:user:"
        self.redis = FakeRedis()

    def save_preference(self, user_id, key, value):
        self.redis.data[f"{self.prefix}{user_id}:pref:{key}"] = json.dumps({
            "v": value,
            "ts": datetime.now(timezone.utc).isoformat(),
        })


def _log(date_str, name, weight, reps=8, rpe=7.0, completed=True):
    return {"log_date": date_str, "session_rpe": rpe, "entries": [
        {"exercise_name": name, "sets": 4, "reps": reps, "weight": weight,
         "rpe": rpe, "completed": completed}
    ]}


# ----------------------------------------------------------------------
# build_training_context
# ----------------------------------------------------------------------

def test_returns_empty_string_without_athlete_key():
    assert build_training_context(None, store=FakeStore()) == ""
    assert build_training_context("", store=FakeStore()) == ""


def test_returns_empty_string_when_no_logs():
    assert build_training_context("k", store=FakeStore([])) == ""


def test_returns_empty_string_when_store_disabled():
    """存储降级时按「没有历史」处理，而不是抛异常阻断生成。"""
    store = FakeStore([_log("2026-09-21", "深蹲", 80.0)], enabled=False)
    assert build_training_context("k", store=store) == ""


def test_returns_empty_string_and_swallows_store_errors():
    store = FakeStore(raises=True)
    assert build_training_context("k", store=store) == ""


def test_includes_session_count_and_completion():
    logs = [_log("2026-09-14", "深蹲", 80.0), _log("2026-09-21", "深蹲", 85.0)]
    ctx = build_training_context("k", store=FakeStore(logs), long_term=FakeLongTerm())
    assert "2 次训练" in ctx
    assert "完成率" in ctx


def test_includes_exercise_progress_direction():
    logs = [_log("2026-09-14", "深蹲", 80.0), _log("2026-09-21", "深蹲", 88.0)]
    ctx = build_training_context("k", store=FakeStore(logs), long_term=FakeLongTerm())
    assert "深蹲" in ctx
    assert "80kg → 88kg" in ctx


def test_reports_decline_when_weight_flat_but_reps_drop():
    """重量没变但次数从 10 掉到 8，同样是退步，不能描述成「维持在」。"""
    logs = [
        _log("2026-09-14", "卧推", 50.0, reps=10),
        _log("2026-09-21", "卧推", 50.0, reps=8),
    ]
    ctx = build_training_context("k", store=FakeStore(logs), long_term=FakeLongTerm())
    assert "下滑" in ctx
    assert "维持在" not in ctx


def test_reports_flat_when_only_one_weighted_session():
    """只有一次加权记录时算不出趋势，如实说「维持」而不是编造涨跌。"""
    logs = [_log("2026-09-21", "卧推", 50.0)]
    ctx = build_training_context("k", store=FakeStore(logs), long_term=FakeLongTerm())
    assert "维持在 50kg" in ctx


def test_omits_adjustments_section_when_none_confirmed():
    logs = [_log("2026-09-21", "深蹲", 80.0)]
    ctx = build_training_context("k", store=FakeStore(logs), long_term=FakeLongTerm())
    assert "已确认的调整" not in ctx


def test_includes_confirmed_adjustments_section():
    long_term = FakeLongTerm()
    save_adjustments("k", [{
        "exercise_name": "深蹲", "type": "progress",
        "suggested": {"weight": 82.5},
    }], long_term=long_term)

    logs = [_log("2026-09-21", "深蹲", 80.0)]
    ctx = build_training_context("k", store=FakeStore(logs), long_term=long_term)
    assert "已确认的调整" in ctx
    assert "82.5kg" in ctx


def test_caps_number_of_exercises_in_context():
    """上下文不是越长越好，超出上限的动作要被截断。"""
    logs = [
        _log("2026-09-21", f"动作{i}", 50.0 + i)
        for i in range(12)
    ]
    ctx = build_training_context("k", store=FakeStore(logs), long_term=FakeLongTerm())
    assert ctx.count("- 动作") <= 6


# ----------------------------------------------------------------------
# save / load adjustments
# ----------------------------------------------------------------------

def test_save_adjustments_roundtrip():
    long_term = FakeLongTerm()
    items = [{"exercise_name": "卧推", "type": "progress", "suggested": {"weight": 62.5}}]
    assert save_adjustments("k", items, long_term=long_term) is True
    assert load_adjustments("k", long_term=long_term) == items


def test_save_adjustments_rejects_empty_input():
    long_term = FakeLongTerm()
    assert save_adjustments(None, [{"a": 1}], long_term=long_term) is False
    assert save_adjustments("k", [], long_term=long_term) is False


def test_load_adjustments_returns_empty_when_absent():
    assert load_adjustments("nobody", long_term=FakeLongTerm()) == []


def test_load_adjustments_returns_empty_for_blank_key():
    assert load_adjustments(None, long_term=FakeLongTerm()) == []


def test_load_adjustments_ignores_expired_entries():
    """超过有效期的调整不再注入——用户水平可能已经变化。"""
    long_term = FakeLongTerm()
    old = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
    long_term.redis.data[f"{long_term.prefix}k:pref:{ADJUSTMENT_KEY}"] = json.dumps({
        "v": [{"exercise_name": "深蹲", "suggested": {"weight": 100}}],
        "ts": old,
    })
    assert load_adjustments("k", long_term=long_term) == []


def test_load_adjustments_keeps_fresh_entries():
    long_term = FakeLongTerm()
    save_adjustments("k", [{"exercise_name": "深蹲"}], long_term=long_term)
    assert len(load_adjustments("k", long_term=long_term)) == 1


def test_load_adjustments_survives_corrupt_payload():
    long_term = FakeLongTerm()
    long_term.redis.data[f"{long_term.prefix}k:pref:{ADJUSTMENT_KEY}"] = "not-json"
    assert load_adjustments("k", long_term=long_term) == []


def test_load_adjustments_returns_empty_for_non_list_value():
    long_term = FakeLongTerm()
    long_term.redis.data[f"{long_term.prefix}k:pref:{ADJUSTMENT_KEY}"] = json.dumps(
        {"v": {"unexpected": "shape"}, "ts": datetime.now(timezone.utc).isoformat()}
    )
    assert load_adjustments("k", long_term=long_term) == []
