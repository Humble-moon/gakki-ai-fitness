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


class FakeLLM:
    """LLMProvider 替身：按配置返回 JSON 或抛错，并记录调用。"""

    def __init__(self, payload=None, error=None):
        self._payload = payload
        self._error = error
        self.calls = []

    def chat_with_json_mode(self, messages, **kwargs):
        self.calls.append(messages)
        if self._error is not None:
            raise self._error
        return self._payload


def _patch_llm(monkeypatch, fake):
    """把 build_advice 内部现构造的 LLMProvider 换成替身。"""
    monkeypatch.setattr("src.llm.provider.LLMProvider", lambda: fake)


@pytest.fixture(autouse=True)
def offline_llm(monkeypatch):
    """默认让模型不可用，周报走规则降级路径。

    离线套件不发真实请求（见 conftest 的说明）。若不打这个桩，每次
    build_advice 都会真的打到 provider——测试变慢、依赖余额、还会把
    用户的 API 额度消耗在断言不了的事情上。需要验证 LLM 路径的用例
    用 _patch_llm 覆盖本替身即可。
    """
    from src.llm.provider import LLMUnavailableError

    class _OfflineLLM:
        def chat_with_json_mode(self, messages, **kwargs):
            raise LLMUnavailableError("离线测试：不发起真实请求",
                                      attempted_models=["offline"],
                                      errors=["offline"])

    monkeypatch.setattr("src.llm.provider.LLMProvider", lambda: _OfflineLLM())


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


# ----------------------------------------------------------------------
# 周报：LLM 增强 + 降级
# ----------------------------------------------------------------------

@pytest.mark.anyio
async def test_advice_carries_llm_report_when_model_available(client, store, monkeypatch):
    store._logs = [
        _log("2026-09-16", "深蹲", 80.0, rpe=6.0),
        _log("2026-09-23", "深蹲", 82.5, rpe=6.0),
    ]
    fake = FakeLLM(payload={
        "headline": "本周完成 2 次训练，深蹲在稳步加重",
        "highlights": [{"text": "深蹲进步明显", "evidence": "重量 80.0 → 82.5 kg"}],
        "concerns": [],
        "next_week": ["深蹲再加 2.5kg"],
    })
    _patch_llm(monkeypatch, fake)

    body = (await client.post("/api/training-advice",
                              json={"athlete_key": "k1"})).json()

    assert body["summary_source"] == "llm"
    assert body["summary_detail"]["headline"] == "本周完成 2 次训练，深蹲在稳步加重"
    assert body["summary"] == body["summary_detail"]["headline"]
    # 规则条目不受 LLM 影响，依然完整
    assert body["items"]
    # 数字必须来自注入的事实清单，而不是模型自己算的
    assert fake.calls, "应当确实调用过模型"
    sent = fake.calls[0][-1]["content"]
    assert "80.0" in sent and "82.5" in sent


@pytest.mark.anyio
async def test_advice_falls_back_to_rule_summary_when_llm_unavailable(client, store, monkeypatch):
    """模型全线不可用时，规则摘要与条目必须照常返回——降级不是错误。"""
    from src.llm.provider import LLMUnavailableError

    store._logs = [
        _log("2026-09-16", "深蹲", 80.0, rpe=6.0),
        _log("2026-09-23", "深蹲", 82.5, rpe=6.0),
    ]
    _patch_llm(monkeypatch, FakeLLM(error=LLMUnavailableError(
        "全部模型失败", attempted_models=["deepseek-chat"], errors=["boom"])))

    resp = await client.post("/api/training-advice", json={"athlete_key": "k1"})
    body = resp.json()

    assert resp.status_code == 200
    assert body["summary_source"] == "rule"
    assert body["summary_detail"] is None
    assert body["summary"], "降级时仍要有可读摘要"
    assert body["items"], "规则条目不该受模型可用性影响"


@pytest.mark.anyio
async def test_advice_falls_back_when_report_structure_is_unusable(client, store, monkeypatch):
    """模型有返回但结构不可用（例如 JSON 解析失败）时同样降级，不放半成品出去。"""
    store._logs = [_log("2026-09-16", "深蹲", 80.0, rpe=6.0)]
    _patch_llm(monkeypatch, FakeLLM(payload={"raw": "抱歉，我无法完成这个请求。"}))

    body = (await client.post("/api/training-advice",
                              json={"athlete_key": "k1"})).json()

    assert body["summary_source"] == "rule"
    assert body["summary_detail"] is None
    assert body["summary"]
