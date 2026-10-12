"""AI 记忆面板端点的 API 测试。用长期记忆替身，不需要 Redis。"""

import httpx
import pytest


class FakeLongTerm:
    """LongTermMemory 的替身：内存读写，记录删除动作。"""

    def __init__(self, preferences=None, feedback=None, injuries=None):
        self._prefs = dict(preferences or {})
        self._feedback = list(feedback or [])
        self._injuries = list(injuries or [])
        self.deleted_keys = []

    def get_preferences_detailed(self, user_id):
        return self._prefs

    def get_feedback(self, user_id):
        return self._feedback

    def get_injury_history(self, user_id):
        return self._injuries

    def delete_preference(self, user_id, key):
        if key not in self._prefs:
            return False
        del self._prefs[key]
        self.deleted_keys.append(key)
        return True

    def purge(self, user_id):
        removed = len(self._prefs) + len(self._feedback)
        self._prefs.clear()
        self._feedback.clear()
        return removed


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def memory(monkeypatch):
    from app import server

    fake = FakeLongTerm(
        preferences={
            "goal": {"value": "增肌", "updated_at": "2026-10-01T00:00:00+00:00"},
            "equipment": {"value": ["哑铃", "杠铃"], "updated_at": None},
            "training_adjustments": {"value": [{"exercise_name": "深蹲"}],
                                     "updated_at": "2026-10-09T00:00:00+00:00"},
        },
        feedback=[{"plan_id": "plan_1", "rating": 5, "comment": "强度合适",
                   "timestamp": "2026-10-05T10:00:00"}],
        injuries=["肩袖损伤"],
    )
    monkeypatch.setattr(server.orch, "long_term", fake)
    return fake


@pytest.fixture
async def client(memory):
    from app import server

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ----------------------------------------------------------------------
# 读取
# ----------------------------------------------------------------------

@pytest.mark.anyio
async def test_get_memory_returns_grouped_records(client):
    resp = await client.get("/api/memory", params={"athlete_key": "k1"})
    body = resp.json()

    assert resp.status_code == 200
    keys = [p["key"] for p in body["preferences"]]
    assert set(keys) == {"goal", "equipment", "training_adjustments"}
    assert body["injuries"] == ["肩袖损伤"]
    assert body["counts"] == {"preferences": 3, "feedback": 1, "injuries": 1}


@pytest.mark.anyio
async def test_get_memory_labels_source_by_origin(client):
    """用户填写的与 AI 推断的必须能区分——后者更该被质疑。"""
    body = (await client.get("/api/memory", params={"athlete_key": "k1"})).json()
    by_key = {p["key"]: p for p in body["preferences"]}

    assert by_key["goal"]["source_label"] == "你在训练档案里填写的"
    assert by_key["training_adjustments"]["source_label"] == "AI 建议，你确认采纳"


@pytest.mark.anyio
async def test_get_memory_marks_stale_entries(client):
    """写入时间久远的条目要能被标出来，让用户知道它可能已不作数。"""
    from app import server

    server.orch.long_term._prefs["goal"]["updated_at"] = "2020-01-01T00:00:00+00:00"
    body = (await client.get("/api/memory", params={"athlete_key": "k1"})).json()
    goal = next(p for p in body["preferences"] if p["key"] == "goal")

    assert goal["stale"] is True
    assert goal["age_days"] > 90
    # 没有时间戳的旧格式记录不该被误判为过时
    equipment = next(p for p in body["preferences"] if p["key"] == "equipment")
    assert equipment["stale"] is False


@pytest.mark.anyio
async def test_get_memory_requires_athlete_key(client):
    resp = await client.get("/api/memory")
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "MISSING_ATHLETE_KEY"


# ----------------------------------------------------------------------
# 删除单条
# ----------------------------------------------------------------------

@pytest.mark.anyio
async def test_delete_single_preference(client, memory):
    resp = await client.delete("/api/memory/goal", params={"athlete_key": "k1"})

    assert resp.status_code == 200
    assert resp.json()["deleted"] is True
    assert memory.deleted_keys == ["goal"]

    body = (await client.get("/api/memory", params={"athlete_key": "k1"})).json()
    assert "goal" not in [p["key"] for p in body["preferences"]]


@pytest.mark.anyio
async def test_delete_unknown_preference_returns_404(client):
    resp = await client.delete("/api/memory/does_not_exist",
                               params={"athlete_key": "k1"})
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "MEMORY_NOT_FOUND"


@pytest.mark.anyio
async def test_delete_preference_requires_athlete_key(client):
    resp = await client.delete("/api/memory/goal")
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "MISSING_ATHLETE_KEY"


# ----------------------------------------------------------------------
# 清空全部
# ----------------------------------------------------------------------

@pytest.mark.anyio
async def test_clear_memory_requires_explicit_confirmation(client, memory):
    """一个拼错的请求不该把用户攒了几个月的画像抹掉。"""
    resp = await client.delete("/api/memory", params={"athlete_key": "k1"})

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "CONFIRMATION_REQUIRED"
    assert memory._prefs, "未确认时不应删除任何东西"


@pytest.mark.anyio
async def test_clear_memory_removes_everything(client, memory):
    resp = await client.delete("/api/memory",
                               params={"athlete_key": "k1", "confirm": "DELETE"})

    assert resp.status_code == 200
    assert resp.json()["deleted_keys"] == 4  # 3 条偏好 + 1 条反馈

    body = (await client.get("/api/memory", params={"athlete_key": "k1"})).json()
    assert body["counts"]["preferences"] == 0
    assert body["counts"]["feedback"] == 0
