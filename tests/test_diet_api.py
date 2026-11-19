"""饮食 API 路由契约（存储替身，全离线）。

锁的边界与训练侧 plans CRUD 同源：
* 生成端点把请求映射成 profile/prefs 并透传 meal_count/query，终态在载荷里
  （HTTP 恒 200，拒绝不是 500）；
* 保存闸门：只收 safe_delivered 且非空的计划；
* 越权/不存在一律 404 防枚举；存储不可用 503；
* 记录端点：非法食物 400 而不是静默丢弃；日期/餐次在路由层校验；
* 统计端点：日类型来自训练日志（按周取数再过滤），无档案时不给目标。
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import server  # noqa: E402

KEY = "deebb161-7d2f-4155-8eb6-35dfe76e7918"
TODAY = date.today()


class _FakeDietPlanStore:
    enabled = True

    def __init__(self):
        self.saved: list[tuple] = []
        self.plans: dict[int, dict] = {}
        self.renamed: list[tuple] = []
        self.deleted: list[int] = []

    def ensure_table(self):
        pass

    def save(self, owner, plan):
        pid = len(self.saved) + 1
        self.saved.append((owner, plan))
        self.plans[pid] = dict(plan, plan_id=pid)
        return pid

    def list_plans(self, owner, limit=50):
        return [{"plan_id": pid, "title": p.get("plan_name", "")}
                for pid, p in self.plans.items()]

    def get_plan(self, owner, plan_id):
        return self.plans.get(plan_id)

    def rename_plan(self, owner, plan_id, title):
        self.renamed.append((owner, plan_id, title))
        return plan_id in self.plans

    def delete_plan(self, owner, plan_id):
        self.deleted.append(plan_id)
        return self.plans.pop(plan_id, None) is not None

    def delete_all(self, owner):
        n = len(self.plans)
        self.plans.clear()
        return n


class _FakeDietLogStore:
    enabled = True

    def __init__(self):
        self.upserts: list[dict] = []
        self.logs: list[dict] = []
        self.reject = False

    def ensure_table(self):
        pass

    def upsert_meal(self, owner, log_date, meal_type, foods, *, plan_id=None, notes=""):
        if self.reject:
            return None
        self.upserts.append({"owner": owner, "log_date": log_date,
                             "meal_type": meal_type, "foods": foods,
                             "plan_id": plan_id, "notes": notes})
        return len(self.upserts)

    def get_logs(self, owner, since=None, limit=200):
        return [log for log in self.logs
                if since is None or log["log_date"] >= since.isoformat()]

    def delete_log(self, owner, log_id):
        before = len(self.logs)
        self.logs = [log for log in self.logs if log["id"] != log_id]
        return len(self.logs) < before

    def delete_all(self, owner):
        n = len(self.logs)
        self.logs.clear()
        return n


class _FakeTrainingStore:
    enabled = True

    def __init__(self, dates=()):
        self.dates = list(dates)
        self.weeks_seen: list[int] = []

    def ensure_table(self):
        pass

    def get_logs(self, owner, weeks=12, limit=100, exercise=None):
        self.weeks_seen.append(weeks)
        return [{"log_date": d} for d in self.dates]


@pytest.fixture()
def rig(monkeypatch):
    plans, logs = _FakeDietPlanStore(), _FakeDietLogStore()
    training = _FakeTrainingStore()
    monkeypatch.setattr(server, "_get_diet_plan_store", lambda: plans)
    monkeypatch.setattr(server, "_get_diet_log_store", lambda: logs)
    monkeypatch.setattr(server, "_get_training_log_store", lambda: training)
    client = TestClient(server.app, raise_server_exceptions=False)
    return client, plans, logs, training


def _delivered_plan() -> dict:
    return {"delivery_status": "safe_delivered", "plan_name": "测试饮食",
            "goal": "增肌", "days": [{"day_type": "训练日", "meals": []}]}


class TestGenerate:
    def test_maps_request_and_returns_payload(self, rig, monkeypatch):
        client, *_ = rig
        seen: dict = {}

        def fake_build(profile, prefs, **kwargs):
            seen.update(profile=profile, prefs=prefs, **kwargs)
            return _delivered_plan()

        import src.core.diet_service as svc
        monkeypatch.setattr(svc, "build_diet_plan", fake_build)
        r = client.post("/api/diet-plans/generate", json={
            "height": 175, "weight": 70, "age": 25, "sex": "男", "goal": "增肌",
            "days_per_week": 4, "allergies": ["花生"], "diseases": ["糖尿病"],
            "avoid_foods": ["香菜"], "notes": "不吃辣", "meal_count": 3,
            "query": "帮我配一周的饭", "athlete_key": KEY})
        assert r.status_code == 200, r.text
        assert r.json()["delivery_status"] == "safe_delivered"
        assert seen["profile"]["weight"] == 70
        assert seen["prefs"]["allergies"] == ["花生"]
        assert seen["meal_count"] == 3
        assert seen["query"] == "帮我配一周的饭"
        assert seen["owner_key"] == KEY

    def test_rejection_payload_is_still_http_200(self, rig, monkeypatch):
        """拒绝是业务终态不是服务器错误——前端按 delivery_status 分流。"""
        client, *_ = rig
        import src.core.diet_service as svc
        monkeypatch.setattr(svc, "build_diet_plan",
                            lambda *a, **k: {"delivery_status": "diet_crisis_stopped",
                                             "days": []})
        r = client.post("/api/diet-plans/generate",
                        json={"athlete_key": KEY, "query": "教我催吐"})
        assert r.status_code == 200
        assert r.json()["delivery_status"] == "diet_crisis_stopped"

    def test_requires_identity(self, rig):
        client, *_ = rig
        r = client.post("/api/diet-plans/generate", json={})
        assert r.status_code == 400

    @pytest.mark.parametrize("body,why", [
        ({"weight": 20}, "体重低于下限"),
        ({"meal_count": 5}, "餐次超上限"),
        ({"days_per_week": 9}, "训练天数超上限"),
        ({"query": "字" * 2001}, "query 超长"),
    ])
    def test_field_validation(self, rig, body, why):
        client, *_ = rig
        r = client.post("/api/diet-plans/generate",
                        json=dict(body, athlete_key=KEY))
        assert r.status_code == 422, why

    def test_inf_weight_rejected_by_schema(self):
        """inf/nan 不是合法体重——allow_inf_nan=False 在模型层拒掉。

        不走路由发 Infinity 字面量：FastAPI 把含 inf 的校验错误序列化回
        422 时会自炸成 500（既有端点同样如此，属框架通病）；这里钉的是
        请求模型本身的约束，与训练侧 schemas 同一防线。
        """
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            server.DietGenerateRequest.model_validate(
                {"weight": float("inf"), "athlete_key": KEY})
        with pytest.raises(ValidationError):
            server.DietGenerateRequest.model_validate(
                {"weight": float("nan"), "athlete_key": KEY})


class TestPlanCRUD:
    def test_save_gate_rejects_undelivered(self, rig):
        client, plans, *_ = rig
        r = client.post("/api/diet-plans", json={
            "plan": {"delivery_status": "safety_blocked", "days": [{"x": 1}]},
            "athlete_key": KEY})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "PLAN_NOT_DELIVERED"
        assert plans.saved == []

    def test_save_gate_rejects_empty_days(self, rig):
        client, plans, *_ = rig
        r = client.post("/api/diet-plans", json={
            "plan": {"delivery_status": "safe_delivered", "days": []},
            "athlete_key": KEY})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "PLAN_EMPTY"
        assert plans.saved == []

    def test_save_list_get_roundtrip(self, rig):
        client, *_ = rig
        r = client.post("/api/diet-plans",
                        json={"plan": _delivered_plan(), "athlete_key": KEY})
        assert r.status_code == 200 and r.json()["saved"] is True
        pid = r.json()["plan_id"]
        listed = client.get(f"/api/diet-plans?athlete_key={KEY}").json()["plans"]
        assert listed[0]["plan_id"] == pid
        got = client.get(f"/api/diet-plans/{pid}?athlete_key={KEY}").json()
        assert got["plan_name"] == "测试饮食"

    def test_get_missing_is_404(self, rig):
        client, *_ = rig
        r = client.get(f"/api/diet-plans/999?athlete_key={KEY}")
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "PLAN_NOT_FOUND"

    def test_rename_trims_and_404_when_missing(self, rig):
        client, plans, *_ = rig
        pid = client.post("/api/diet-plans",
                          json={"plan": _delivered_plan(), "athlete_key": KEY}
                          ).json()["plan_id"]
        r = client.patch(f"/api/diet-plans/{pid}",
                         json={"title": "  新名字  ", "athlete_key": KEY})
        assert r.status_code == 200
        assert plans.renamed[0][2] == "新名字"
        r = client.patch("/api/diet-plans/999",
                         json={"title": "x", "athlete_key": KEY})
        assert r.status_code == 404

    def test_rename_blank_rejected_by_schema(self, rig):
        client, plans, *_ = rig
        r = client.patch("/api/diet-plans/1",
                         json={"title": "", "athlete_key": KEY})
        assert r.status_code == 422
        assert plans.renamed == []

    def test_delete_then_404(self, rig):
        client, *_ = rig
        pid = client.post("/api/diet-plans",
                          json={"plan": _delivered_plan(), "athlete_key": KEY}
                          ).json()["plan_id"]
        assert client.delete(f"/api/diet-plans/{pid}?athlete_key={KEY}"
                             ).status_code == 200
        assert client.delete(f"/api/diet-plans/{pid}?athlete_key={KEY}"
                             ).status_code == 404

    def test_storage_unavailable_is_503(self, rig, monkeypatch):
        client, plans, *_ = rig
        plans.enabled = False
        r = client.get(f"/api/diet-plans?athlete_key={KEY}")
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "STORAGE_UNAVAILABLE"


class TestDietLogs:
    def test_upsert_passes_through(self, rig):
        client, _, logs, _ = rig
        foods = [{"name": "燕麦粥（煮）", "grams": 250}]
        r = client.post("/api/diet-logs", json={
            "log_date": TODAY.isoformat(), "meal_type": "早餐",
            "foods": foods, "notes": "按计划吃的", "athlete_key": KEY})
        assert r.status_code == 200 and r.json()["saved"] is True
        assert logs.upserts[0]["meal_type"] == "早餐"
        assert logs.upserts[0]["foods"] == foods

    def test_invalid_foods_is_400_not_silent_drop(self, rig):
        client, _, logs, _ = rig
        logs.reject = True
        r = client.post("/api/diet-logs", json={
            "log_date": TODAY.isoformat(), "meal_type": "午餐",
            "foods": [{"name": "不存在的食物", "grams": 100}], "athlete_key": KEY})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "INVALID_DIET_LOG"

    @pytest.mark.parametrize("body,why", [
        ({"log_date": "2026/11/19"}, "日期格式"),
        ({"meal_type": "夜宵"}, "餐次不在枚举"),
        ({"foods": []}, "空食物列表"),
        ({"foods": [{"name": "x", "grams": 1}] * 13}, "食物超 12 条"),
    ])
    def test_log_validation(self, rig, body, why):
        client, *_ = rig
        base = {"log_date": TODAY.isoformat(), "meal_type": "早餐",
                "foods": [{"name": "米饭（蒸）", "grams": 100}], "athlete_key": KEY}
        r = client.post("/api/diet-logs", json=dict(base, **body))
        assert r.status_code == 422, why

    def test_list_and_delete(self, rig):
        client, _, logs, _ = rig
        logs.logs = [{"id": 1, "log_date": TODAY.isoformat(), "meal_type": "早餐",
                      "kcal": 300, "protein_g": 10, "custom_kcal": 0, "foods": []}]
        listed = client.get(f"/api/diet-logs?athlete_key={KEY}").json()["logs"]
        assert len(listed) == 1
        assert client.delete(f"/api/diet-logs/1?athlete_key={KEY}").status_code == 200
        assert client.delete(f"/api/diet-logs/1?athlete_key={KEY}").status_code == 404

    def test_storage_unavailable_is_503(self, rig):
        client, _, logs, _ = rig
        logs.enabled = False
        r = client.post("/api/diet-logs", json={
            "log_date": TODAY.isoformat(), "meal_type": "早餐",
            "foods": [{"name": "米饭（蒸）", "grams": 100}], "athlete_key": KEY})
        assert r.status_code == 503


class TestDietStats:
    def test_without_profile_no_targets(self, rig):
        client, _, logs, _ = rig
        logs.logs = [{"log_date": TODAY.isoformat(), "kcal": 2100,
                      "protein_g": 120, "custom_kcal": 0}]
        body = client.get(f"/api/diet-stats?athlete_key={KEY}&days=7").json()
        assert body["summary"]["days_logged"] == 1
        assert body["days"][-1]["target_kcal"] is None
        # 训练替身可用但零记录＝「没练过」，不是「不知道」：全休息日。
        assert body["summary"]["classified"] is True
        assert body["days"][-1]["day_type"] == "休息日"

    def test_training_dates_classify_days(self, rig):
        client, _, logs, training = rig
        yesterday = (TODAY - timedelta(days=1)).isoformat()
        training.dates = [yesterday]
        logs.logs = [{"log_date": yesterday, "kcal": 2350,
                      "protein_g": 140, "custom_kcal": 0}]
        body = client.get(
            f"/api/diet-stats?athlete_key={KEY}&days=7&weight=70&height=175"
            f"&age=25&sex=男&goal=增肌&days_per_week=4").json()
        by_date = {d["date"]: d for d in body["days"]}
        assert by_date[yesterday]["day_type"] == "训练日"
        assert by_date[yesterday]["target_kcal"] is not None
        assert by_date[TODAY.isoformat()]["day_type"] == "休息日"
        assert training.weeks_seen, "训练史按周取数"

    def test_training_store_failure_degrades_not_blocks(self, rig, monkeypatch):
        client, _, logs, training = rig
        logs.logs = [{"log_date": TODAY.isoformat(), "kcal": 2100,
                      "protein_g": 120, "custom_kcal": 0}]

        def boom(*a, **k):
            raise RuntimeError("pg down")

        monkeypatch.setattr(training, "get_logs", boom)
        body = client.get(f"/api/diet-stats?athlete_key={KEY}&days=7").json()
        assert body["summary"]["classified"] is False
        assert body["summary"]["days_logged"] == 1

    def test_window_clamped_to_90(self, rig):
        client, *_ = rig
        body = client.get(f"/api/diet-stats?athlete_key={KEY}&days=999").json()
        assert body["summary"]["window_days"] == 90

    def test_storage_unavailable_is_503(self, rig):
        client, _, logs, _ = rig
        logs.enabled = False
        r = client.get(f"/api/diet-stats?athlete_key={KEY}")
        assert r.status_code == 503