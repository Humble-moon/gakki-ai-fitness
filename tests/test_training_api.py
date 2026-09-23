"""训练闭环端点的 API 测试。用存储替身，不需要 PostgreSQL 或 Redis。"""

import httpx
import pytest


class FakeTrainingLogStore:
    """TrainingLogStore 的替身：内存读写，可模拟降级。"""

    def __init__(self, logs=None, enabled=True):
        self.enabled = enabled
        self._logs = logs or []
        self.saved_calls = []

    def ensure_table(self):
        pass

    def save_log(self, **kwargs):
        if not self.enabled:
            return None
        self.saved_calls.append(kwargs)
        return len(self.saved_calls)

    def get_logs(self, athlete_key, **kwargs):
        return self._logs


def _log(date_str, name, weight, reps=8, rpe=6.0, completed=True):
    return {"log_date": date_str, "session_rpe": rpe, "entries": [
        {"exercise_name": name, "sets": 4, "reps": reps, "weight": weight,
         "rpe": rpe, "completed": completed}
    ]}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def store():
    return FakeTrainingLogStore()


@pytest.fixture
async def client(monkeypatch, store):
    from app import server
    from src.core import training_advice

    monkeypatch.setattr(server, "_training_log_store", store)
    training_advice.advice_store.clear()

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ----------------------------------------------------------------------
# 写日志
# ----------------------------------------------------------------------

@pytest.mark.anyio
async def test_create_log_returns_id(client, store):
    resp = await client.post("/api/training-logs", json={
        "athlete_key": "k1",
        "log_date": "2026-09-23",
        "focus": "胸",
        "entries": [{"exercise_name": "卧推", "sets": 4, "reps": 10, "weight": 60.0}],
    })
    assert resp.status_code == 200
    assert resp.json()["log_id"] == 1
    assert resp.json()["entries_saved"] == 1
    assert store.saved_calls[0]["athlete_key"] == "k1"


@pytest.mark.anyio
async def test_create_log_requires_athlete_key(client):
    resp = await client.post("/api/training-logs", json={"entries": []})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "MISSING_ATHLETE_KEY"


@pytest.mark.anyio
async def test_create_log_returns_503_when_storage_disabled(monkeypatch, store):
    store.enabled = False
    from app import server

    monkeypatch.setattr(server, "_training_log_store", store)
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post("/api/training-logs", json={
            "athlete_key": "k1", "entries": []})
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "STORAGE_UNAVAILABLE"


@pytest.mark.anyio
async def test_create_log_counts_only_named_entries(client):
    resp = await client.post("/api/training-logs", json={
        "athlete_key": "k1",
        "entries": [
            {"exercise_name": "卧推", "sets": 4, "reps": 10, "weight": 60.0},
            {"exercise_name": "", "sets": 0, "reps": 0, "weight": 0},
        ],
    })
    assert resp.json()["entries_saved"] == 1


# ----------------------------------------------------------------------
# 查日志 / 统计
# ----------------------------------------------------------------------

@pytest.mark.anyio
async def test_list_logs_requires_athlete_key(client):
    resp = await client.get("/api/training-logs")
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_list_logs_returns_entries(client, store):
    store._logs = [_log("2026-09-23", "卧推", 60.0)]
    resp = await client.get("/api/training-logs", params={"athlete_key": "k1"})
    assert resp.status_code == 200
    assert resp.json()["count"] == 1


@pytest.mark.anyio
async def test_stats_returns_complete_shape_for_empty_history(client):
    resp = await client.get("/api/training-stats", params={"athlete_key": "k1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["totals"]["sessions"] == 0
    assert body["weekly"] == []
    assert body["exercises"] == []


@pytest.mark.anyio
async def test_stats_aggregates_history(client, store):
    store._logs = [_log("2026-09-16", "深蹲", 80.0), _log("2026-09-23", "深蹲", 85.0)]
    resp = await client.get("/api/training-stats", params={"athlete_key": "k1"})
    body = resp.json()
    assert body["totals"]["sessions"] == 2
    assert body["exercises"][0]["name"] == "深蹲"
    assert body["exercises"][0]["trend_pct"] > 0


@pytest.mark.anyio
async def test_stats_requires_athlete_key(client):
    resp = await client.get("/api/training-stats")
    assert resp.status_code == 400


# ----------------------------------------------------------------------
# 调整建议
# ----------------------------------------------------------------------

@pytest.mark.anyio
async def test_advice_returns_pending_artifact(client, store):
    store._logs = [
        _log("2026-09-16", "深蹲", 80.0, rpe=6.0),
        _log("2026-09-23", "深蹲", 82.5, rpe=6.0),
    ]
    resp = await client.post("/api/training-advice", json={"athlete_key": "k1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "advice_pending"
    assert body["advice_id"].startswith("advice_")
    assert len(body["items"]) == 1
    assert body["items"][0]["type"] == "progress"


@pytest.mark.anyio
async def test_advice_reports_empty_history_gracefully(client):
    resp = await client.post("/api/training-advice", json={"athlete_key": "k1"})
    assert resp.status_code == 200
    assert resp.json()["items"] == []


@pytest.mark.anyio
async def test_advice_requires_athlete_key(client):
    resp = await client.post("/api/training-advice", json={})
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_apply_persists_only_selected_items(client, store, monkeypatch):
    """只把用户勾选的条目写进长期记忆，未勾选的不写入。

    patch 打在 training_history 上而不是 training_advice 上：resolve_advice
    是在函数体内 import 的，patch 调用方的模块属性不生效，会漏到真实 Redis。
    """
    store._logs = [
        _log("2026-09-16", "深蹲", 80.0, rpe=6.0),
        _log("2026-09-23", "深蹲", 82.5, rpe=6.0),
        _log("2026-09-23", "划船", 50.0, rpe=6.0),   # 只练一次 → swap 建议
    ]
    created = (await client.post("/api/training-advice",
                                 json={"athlete_key": "k1"})).json()
    assert len(created["items"]) == 2

    target = created["items"][0]["item_id"]
    captured = {}

    def fake_save(athlete_key, items, *, long_term=None):
        captured["athlete_key"] = athlete_key
        captured["items"] = items
        return True

    monkeypatch.setattr("src.core.training_history.save_adjustments", fake_save)

    resp = await client.post("/api/training-advice/apply", json={
        "advice_id": created["advice_id"], "accepted_item_ids": [target]})
    assert resp.status_code == 200
    assert resp.json()["status"] == "advice_applied"
    # 只写入勾选的那一条，未勾选的 swap 不应被持久化
    assert captured["athlete_key"] == "k1"
    assert [i["item_id"] for i in captured["items"]] == [target]


@pytest.mark.anyio
async def test_apply_with_unknown_id_returns_409(client):
    resp = await client.post("/api/training-advice/apply", json={
        "advice_id": "advice_does_not_exist", "accepted_item_ids": []})
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "ADVICE_EXPIRED"


@pytest.mark.anyio
async def test_apply_with_empty_selection_marks_dismissed(client, store):
    store._logs = [
        _log("2026-09-16", "深蹲", 80.0, rpe=6.0),
        _log("2026-09-23", "深蹲", 82.5, rpe=6.0),
    ]
    created = (await client.post("/api/training-advice",
                                 json={"athlete_key": "k1"})).json()
    resp = await client.post("/api/training-advice/apply", json={
        "advice_id": created["advice_id"], "accepted_item_ids": []})
    assert resp.status_code == 200
    assert resp.json()["status"] == "advice_dismissed"
    assert resp.json()["applied"] == []


@pytest.mark.anyio
async def test_apply_rejects_forged_item_ids(client, store):
    """前端传来不在该工件里的 item_id 时，不能被写入。"""
    store._logs = [
        _log("2026-09-16", "深蹲", 80.0, rpe=6.0),
        _log("2026-09-23", "深蹲", 82.5, rpe=6.0),
    ]
    created = (await client.post("/api/training-advice",
                                 json={"athlete_key": "k1"})).json()
    resp = await client.post("/api/training-advice/apply", json={
        "advice_id": created["advice_id"],
        "accepted_item_ids": ["伪造的动作:progress"]})
    assert resp.status_code == 200
    assert resp.json()["applied"] == []
    assert resp.json()["status"] == "advice_dismissed"
