"""饮食存储层测试：离线 SQL 契约 + 纯函数钳制 + 降级语义。

真实 upsert/rekey 的行级行为由 integration 标记用例连真库验证
（本地 docker PG 在跑）；默认套件离线，靠 SQL 形状与参数断言锁住
归属 WHERE、先数后删、FOR UPDATE、ON CONFLICT 约束名这些关键不变量。
"""

import json
from contextlib import contextmanager

import pytest

from src.storage import diet_log_store as dls
from src.storage import diet_plan_store as dps


class _Row(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


class _FakeTx:
    def __init__(self, db):
        self.db = db

    def execute(self, sql, params=None):
        self.db.statements.append((sql, params or {}))
        return None

    def fetch_one(self, sql, params=None):
        self.db.statements.append((sql, params or {}))
        return self.db.next_row

    def fetch_all(self, sql, params=None):
        self.db.statements.append((sql, params or {}))
        return self.db.next_rows


class _FakeDB:
    def __init__(self, next_row=None, next_rows=None):
        self.statements = []
        self.next_row = next_row
        self.next_rows = next_rows or []

    @contextmanager
    def transaction(self):
        yield _FakeTx(self)

    def fetch_one(self, sql, params=None):
        self.statements.append((sql, params or {}))
        return self.next_row

    def fetch_all(self, sql, params=None):
        self.statements.append((sql, params or {}))
        return self.next_rows


def _plan_store(next_row=None):
    store = dps.DietPlanStore.__new__(dps.DietPlanStore)
    store.db = _FakeDB(next_row=next_row)
    store.enabled = True
    store._table_ready = True
    return store


def _log_store(next_row=None):
    store = dls.DietLogStore.__new__(dls.DietLogStore)
    store.db = _FakeDB(next_row=next_row)
    store.enabled = True
    store._table_ready = True
    return store


class TestDietPlanStoreSQL:
    def test_save_strips_volatile_and_carries_owner(self):
        store = _plan_store(next_row=_Row(id=7))
        plan = {"plan_name": "x", "goal": "增肌", "days": [{"day_type": "训练日"}],
                "confidence": 0.9, "_delivery_allowed": True, "_safety_block": None}
        pid = store.save("g1_abc", plan)
        assert pid == 7
        sql, params = store.db.statements[-1]
        assert "INSERT INTO diet_plans" in sql and params["ok"] == "g1_abc"
        data = json.loads(params["data"])
        assert "_delivery_allowed" not in data and "_safety_block" not in data
        assert data["plan_name"] == "x"

    def test_save_without_days_writes_nothing(self):
        store = _plan_store()
        assert store.save("g1_abc", {"plan_name": "x"}) is None
        assert store.db.statements == []

    def test_get_plan_owner_in_where(self):
        store = _plan_store(next_row=_Row(id=7, goal="增肌", plan_data={"days": []},
                                          confidence=0.9, created_at=None))
        store.get_plan("g1_abc", 7)
        sql, params = store.db.statements[-1]
        assert "owner_key = :ok" in sql and params["ok"] == "g1_abc" and params["pid"] == 7

    def test_rename_uses_for_update(self):
        store = _plan_store(next_row=_Row(plan_data={"plan_name": "old"}))
        assert store.rename_plan("g1_abc", 7, "new") is True
        select_sql, _ = store.db.statements[0]
        update_sql, update_params = store.db.statements[1]
        assert "FOR UPDATE" in select_sql
        assert json.loads(update_params["data"])["plan_name"] == "new"

    def test_delete_counts_before_deleting(self):
        store = _plan_store(next_row=_Row(c=1))
        assert store.delete_plan("g1_abc", 7) is True
        assert "COUNT(*)" in store.db.statements[0][0]
        assert "DELETE FROM diet_plans" in store.db.statements[1][0]

    def test_delete_absent_returns_false(self):
        store = _plan_store(next_row=_Row(c=0))
        assert store.delete_plan("g1_abc", 7) is False
        assert len(store.db.statements) == 1  # 没有发出 DELETE

    def test_rekey_updates_owner(self):
        store = _plan_store(next_row=_Row(c=3))
        assert store.rekey("g1_old", "u:1") == 3
        sql, params = store.db.statements[-1]
        assert "UPDATE diet_plans SET owner_key" in sql
        assert params == {"new": "u:1", "old": "g1_old"}


class TestDietLogClean:
    def test_catalog_food_recomputed_not_trusted(self):
        store = _log_store()
        cleaned, reason = store._clean_foods([{"name": "米饭（蒸）", "grams": 150, "kcal": 99999}])
        assert reason is None
        assert cleaned[0]["kcal"] == 174 and cleaned[0]["source"] == "catalog"

    def test_custom_food_clamped_and_labelled(self):
        store = _log_store()
        cleaned, reason = store._clean_foods([{"name": "外卖炒饭", "grams": 300,
                                               "kcal": 99999, "protein_g": 20,
                                               "source": "custom"}])
        assert reason is None
        assert cleaned[0]["kcal"] == dls.MAX_CUSTOM_FOOD_KCAL
        assert cleaned[0]["source"] == "custom"

    def test_meal_total_cap_rejects(self):
        store = _log_store()
        foods = [{"name": "外卖大餐", "grams": 500, "kcal": 2000, "source": "custom"},
                 {"name": "外卖加餐", "grams": 500, "kcal": 2000, "source": "custom"}]
        cleaned, reason = store._clean_foods(foods)
        assert cleaned is None
        assert reason["code"] == "MEAL_KCAL_CAP" and reason["total"] == 4000

    def test_unknown_catalog_food_rejected_with_row_and_name(self):
        store = _log_store()
        cleaned, reason = store._clean_foods([
            {"name": "米饭（蒸）", "grams": 100}, {"name": "炸鸡架", "grams": 100}])
        assert cleaned is None
        assert reason == {"code": "UNKNOWN_FOOD", "index": 1, "name": "炸鸡架"}

    @pytest.mark.parametrize("foods", [
        [], [{"name": "米饭（蒸）", "grams": 0}],
        [{"name": "米饭（蒸）", "grams": -5}],
        [{"name": "", "grams": 100}],
        [{"name": "米饭（蒸）", "grams": 5000}],
        "not a list",
    ])
    def test_invalid_entries_rejected(self, foods):
        cleaned, reason = _log_store()._clean_foods(foods)
        assert cleaned is None and reason is not None

    def test_explain_rejection_is_pure(self):
        store = _log_store()
        reason = store.explain_rejection([{"name": "炸鸡架", "grams": 100}])
        assert reason["code"] == "UNKNOWN_FOOD"
        assert store.explain_rejection([{"name": "炸鸡架", "grams": 100}]) == reason


class TestDietLogStoreSQL:
    def test_upsert_uses_named_unique_constraint(self):
        store = _log_store(next_row=_Row(id=3))
        lid = store.upsert_meal("g1_abc", "2026-11-19", "早餐",
                                [{"name": "燕麦粥（煮）", "grams": 250}])
        assert lid == 3
        sql, params = store.db.statements[-1]
        assert "ON CONFLICT ON CONSTRAINT uq_diet_logs_owner_date_meal" in sql
        assert "DO UPDATE SET foods" in sql
        assert params["m"] == "早餐" and params["ok"] == "g1_abc"

    def test_upsert_rejects_bad_meal_type(self):
        store = _log_store()
        assert store.upsert_meal("g1_abc", "2026-11-19", "夜宵",
                                 [{"name": "燕麦粥（煮）", "grams": 250}]) is None
        assert store.db.statements == []

    def test_get_logs_since_param(self):
        from datetime import date
        store = _log_store()
        store.get_logs("g1_abc", since=date(2026, 11, 1))
        sql, params = store.db.statements[-1]
        assert "log_date >= :since" in sql and params["since"] == date(2026, 11, 1)

    def test_rekey_guards_unique_collision(self):
        store = _log_store(next_row=_Row(c=2))
        assert store.rekey("g1_old", "u:1") == 2
        sql, _ = store.db.statements[-1]
        assert "NOT EXISTS" in sql and "d2.meal_type = diet_logs.meal_type" in sql


class TestDegradation:
    def test_plan_store_degrades_when_table_creation_fails(self, monkeypatch):
        store = dps.DietPlanStore.__new__(dps.DietPlanStore)
        store.enabled = True
        store._table_ready = False

        class _BrokenDB:
            engine = None

        store.db = _BrokenDB()
        monkeypatch.setattr(dps, "PGClient", lambda: _BrokenDB())
        store.ensure_table()
        assert store.enabled is False
        assert store.save("g1_abc", {"days": [1]}) is None
        assert store.list_plans("g1_abc") == []
        assert store.delete_all("g1_abc") == 0

    def test_log_store_degrades_when_table_creation_fails(self):
        store = dls.DietLogStore.__new__(dls.DietLogStore)
        store.enabled = True
        store._table_ready = False

        class _BrokenDB:
            engine = None

        store.db = _BrokenDB()
        store.ensure_table()
        assert store.enabled is False
        assert store.upsert_meal("g1_abc", "2026-11-19", "早餐",
                                 [{"name": "燕麦粥（煮）", "grams": 250}]) is None
        assert store.get_logs("g1_abc") == []