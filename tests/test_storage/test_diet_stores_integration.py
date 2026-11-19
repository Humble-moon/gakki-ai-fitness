"""饮食存储层连真库验证（integration，默认跳过）。

离线 SQL 契约锁形状，这里锁行级行为：upsert 覆盖语义、UNIQUE 约束真实存在、
rekey 的碰撞守卫、归属隔离。用一次性 owner key，结束 delete_all 清场，
不碰演示数据。
"""

import uuid
from datetime import date

import pytest

from src.storage.diet_log_store import DietLogStore
from src.storage.diet_plan_store import DietPlanStore

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _migrated_schema():
    """先跑迁移再碰 store：与服务器启动同序（AUTO_MIGRATE 先于 ensure_table）。

    若让 ensure_table 的 create_all 先建表，启动期 alembic 的 create_table
    会撞「表已存在」而拒启动——顺序本身就是被测契约的一部分。
    """
    from src.db_migrate import upgrade_to_head
    upgrade_to_head()
    yield


@pytest.fixture()
def owner():
    key = f"test-diet-{uuid.uuid4().hex[:12]}"
    yield key
    DietLogStore().delete_all(key)
    DietPlanStore().delete_all(key)


class TestDietLogUpsert:
    def test_same_meal_twice_overwrites_not_duplicates(self, owner):
        store = DietLogStore()
        first = store.upsert_meal(owner, date(2026, 11, 19), "早餐",
                                  [{"name": "燕麦粥（煮）", "grams": 250}])
        second = store.upsert_meal(owner, date(2026, 11, 19), "早餐",
                                   [{"name": "米饭（蒸）", "grams": 150}])
        assert first is not None and second is not None
        logs = store.get_logs(owner)
        assert len(logs) == 1
        assert logs[0]["foods"][0]["name"] == "米饭（蒸）"
        assert logs[0]["kcal"] == 174  # 服务端重算值

    def test_different_meals_coexist(self, owner):
        store = DietLogStore()
        store.upsert_meal(owner, date(2026, 11, 19), "早餐",
                          [{"name": "燕麦粥（煮）", "grams": 250}])
        store.upsert_meal(owner, date(2026, 11, 19), "午餐",
                          [{"name": "米饭（蒸）", "grams": 150}])
        assert len(store.get_logs(owner)) == 2

    def test_custom_food_stored_with_label(self, owner):
        store = DietLogStore()
        store.upsert_meal(owner, date(2026, 11, 19), "午餐",
                          [{"name": "食堂红烧肉", "grams": 150, "kcal": 450,
                            "protein_g": 15, "source": "custom"}])
        log = store.get_logs(owner)[0]
        assert log["foods"][0]["source"] == "custom"
        assert log["custom_kcal"] == 450

    def test_owner_isolation(self, owner):
        store = DietLogStore()
        store.upsert_meal(owner, date(2026, 11, 19), "早餐",
                          [{"name": "燕麦粥（煮）", "grams": 250}])
        assert store.get_logs(owner + "-other") == []
        assert store.delete_log(owner + "-other", 1) is False

    def test_rekey_moves_logs_and_guards_collision(self, owner):
        store = DietLogStore()
        new_key = owner + "-claimed"
        store.upsert_meal(owner, date(2026, 11, 19), "早餐",
                          [{"name": "燕麦粥（煮）", "grams": 250}])
        store.upsert_meal(new_key, date(2026, 11, 19), "早餐",
                          [{"name": "米饭（蒸）", "grams": 100}])
        moved = store.rekey(owner, new_key)
        assert moved == 0  # 新身份已有同餐 → 守卫跳过，不覆盖
        logs = store.get_logs(new_key)
        assert len(logs) == 1 and logs[0]["foods"][0]["name"] == "米饭（蒸）"
        assert store.get_logs(owner) != []  # 旧的那份仍在（未被吞掉）
        store.delete_all(new_key)

    def test_rekey_moves_when_no_collision(self, owner):
        store = DietLogStore()
        new_key = owner + "-claimed2"
        store.upsert_meal(owner, date(2026, 11, 18), "晚餐",
                          [{"name": "鸡胸肉（煮）", "grams": 150}])
        assert store.rekey(owner, new_key) == 1
        assert len(store.get_logs(new_key)) == 1
        assert store.get_logs(owner) == []
        store.delete_all(new_key)


class TestDietPlanStore:
    def test_save_list_get_rename_delete_roundtrip(self, owner):
        store = DietPlanStore()
        plan = {"plan_name": "集成测试饮食", "goal": "增肌", "confidence": 0.9,
                "days": [{"day_type": "训练日", "meals": []}],
                "_delivery_allowed": True}
        pid = store.save(owner, plan)
        assert pid is not None
        listed = store.list_plans(owner)
        assert listed[0]["title"] == "集成测试饮食"
        fetched = store.get_plan(owner, pid)
        assert "_delivery_allowed" not in fetched
        assert store.rename_plan(owner, pid, "改名后") is True
        assert store.get_plan(owner, pid)["plan_name"] == "改名后"
        assert store.get_plan(owner + "-other", pid) is None
        assert store.delete_plan(owner, pid) is True
        assert store.delete_plan(owner, pid) is False

    def test_rekey_moves_plans(self, owner):
        store = DietPlanStore()
        new_key = owner + "-claimed"
        pid = store.save(owner, {"plan_name": "x", "goal": "增肌",
                                 "days": [{"day_type": "训练日"}]})
        assert store.rekey(owner, new_key) == 1
        assert store.get_plan(new_key, pid) is not None
        assert store.get_plan(owner, pid) is None
        store.delete_all(new_key)