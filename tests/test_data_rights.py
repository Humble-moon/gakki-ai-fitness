"""个人数据导出与删除。

健康数据属于个人信息，用户必须能取回、也能删掉自己产生的全部内容。
这两个端点覆盖三处存储：训练日志（PostgreSQL）、长期记忆（Redis）、
会话（Redis）。缓存不在范围内——它是"同一份身体数据 + 同一目标命中同一
计划"的去重产物，不绑定个人身份。

删除是不可逆操作，因此本文件重点钉三件事：
  1. 没有显式确认时**绝不能删**
  2. 删除范围要覆盖全部存储，漏一处就是"删不干净"
  3. 部分失败必须如实上报，不得谎报"已全部删除"
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class _FakeLongTerm:
    def __init__(self):
        self.prefs: dict = {}
        self.purged: list = []
        self.raise_on_purge = False

    def get_preferences(self, user_id):
        return self.prefs.get(str(user_id), {})

    def purge(self, user_id):
        if self.raise_on_purge:
            raise RuntimeError("redis down")
        self.purged.append(user_id)
        return 3


class _FakeConversation:
    def __init__(self):
        self.purged: list = []
        self.raise_on_purge = False

    def get_context(self, session_id):
        return "用户: 增肌怎么吃\n助手: 多吃蛋白质"

    def get_plan_state(self, session_id):
        return "第1天(胸): 卧推"

    def purge(self, session_id):
        if self.raise_on_purge:
            raise RuntimeError("redis down")
        self.purged.append(session_id)
        return 2


class _FakeLogStore:
    def __init__(self):
        self.deleted: list = []
        self.raise_on_delete = False

    def get_logs(self, athlete_key, weeks=12, limit=100):
        return [{"id": 1, "log_date": "2026-09-20", "entries": []}]

    def delete_logs(self, athlete_key):
        if self.raise_on_delete:
            raise RuntimeError("pg down")
        self.deleted.append(athlete_key)
        return 5


@pytest.fixture
def rig(monkeypatch):
    from app import server

    long_term, conversation, store = _FakeLongTerm(), _FakeConversation(), _FakeLogStore()
    long_term.prefs["athlete-abc"] = {"goal": "增肌"}
    monkeypatch.setattr(server.orch, "long_term", long_term, raising=False)
    monkeypatch.setattr(server.orch, "conversation", conversation, raising=False)
    monkeypatch.setattr(server, "_get_training_log_store", lambda: store)

    client = TestClient(server.app)
    return client, long_term, conversation, store


class TestExport:
    def test_requires_athlete_key(self, rig):
        client, *_ = rig
        resp = client.get("/api/user-data/export")
        assert resp.status_code == 400
        assert resp.json()["error"] == "athlete_key_required"

    def test_returns_all_three_sources(self, rig):
        client, *_ = rig
        resp = client.get(
            "/api/user-data/export?athlete_key=athlete-abc&session_id=s-1"
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["athlete_key"] == "athlete-abc"
        assert len(body["training_logs"]) == 1
        assert body["long_term_memory"] == {"goal": "增肌"}
        assert body["conversation"]["session_id"] == "s-1"
        assert "exported_at" in body

    def test_session_optional(self, rig):
        client, *_ = rig
        body = client.get("/api/user-data/export?athlete_key=athlete-abc").json()
        assert body["conversation"] is None

    def test_one_source_failure_does_not_kill_the_export(self, rig, monkeypatch):
        """单个存储导出失败时，其余部分仍要交给用户——半份数据好过没有。"""
        client, _, _, store = rig
        monkeypatch.setattr(store, "get_logs", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("pg down")))
        body = client.get("/api/user-data/export?athlete_key=athlete-abc").json()
        assert body["training_logs"] == []
        assert "training_logs_error" in body
        # 其他源不受影响
        assert body["long_term_memory"] == {"goal": "增肌"}


class TestDeleteSafety:
    def test_requires_athlete_key(self, rig):
        client, *_ = rig
        resp = client.delete("/api/user-data?confirm=DELETE")
        assert resp.status_code == 400

    def test_refuses_without_explicit_confirmation(self, rig):
        """不可逆操作：没有 confirm=DELETE 时一个字节都不能删。"""
        client, long_term, conversation, store = rig
        resp = client.delete("/api/user-data?athlete_key=athlete-abc")
        assert resp.status_code == 400
        assert resp.json()["error"] == "confirmation_required"
        assert store.deleted == [] and long_term.purged == [] and conversation.purged == []

    def test_rejects_wrong_confirmation_value(self, rig):
        client, long_term, _, store = rig
        resp = client.delete("/api/user-data?athlete_key=athlete-abc&confirm=yes")
        assert resp.status_code == 400
        assert store.deleted == [] and long_term.purged == []


class TestDeleteBehaviour:
    def test_deletes_all_sources(self, rig):
        client, long_term, conversation, store = rig
        resp = client.delete(
            "/api/user-data?athlete_key=athlete-abc&confirm=DELETE&session_id=s-1"
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["deleted"] == {
            "training_logs": 5, "long_term_keys": 3, "conversation_keys": 2,
        }
        assert store.deleted == ["athlete-abc"]
        assert long_term.purged == ["athlete-abc"]
        assert conversation.purged == ["s-1"]

    def test_session_optional(self, rig):
        client, _, conversation, _ = rig
        body = client.delete(
            "/api/user-data?athlete_key=athlete-abc&confirm=DELETE"
        ).json()
        assert "conversation_keys" not in body["deleted"]
        assert conversation.purged == []

    def test_partial_failure_is_reported_not_hidden(self, rig):
        """一个源删失败时必须如实上报——谎报"已全部删除"会让用户以为
        数据没了，实际还留在库里。"""
        client, long_term, _, _ = rig
        long_term.raise_on_purge = True
        resp = client.delete("/api/user-data?athlete_key=athlete-abc&confirm=DELETE")
        assert resp.status_code == 500
        body = resp.json()
        assert body["ok"] is False
        assert "long_term" in body["errors"]
        # 失败源不进 deleted，成功的源如实记录
        assert "long_term_keys" not in body["deleted"]
        assert body["deleted"]["training_logs"] == 5

    def test_log_store_failure_is_reported(self, rig):
        client, _, _, store = rig
        store.raise_on_delete = True
        body = client.delete(
            "/api/user-data?athlete_key=athlete-abc&confirm=DELETE"
        ).json()
        assert body["ok"] is False
        assert "training_logs" in body["errors"]


class TestPurgeUsesPrefixScan:
    """purge 用 SCAN 前缀匹配，避免枚举已知字段——漏一个字段就是删不干净。"""

    def test_long_term_purge_removes_all_keys_under_prefix(self):
        from src.memory.long_term import LongTermMemory

        class _ScanRedis:
            def __init__(self):
                self.conn = self
                self.deleted: list = []
                self.keys = ["memory:user:u1:pref:goal", "memory:user:u1:feedback:p1"]

            def scan_iter(self, match=None, count=None):
                return [k.encode() for k in self.keys]

            def delete(self, key):
                self.deleted.append(key)

        lt = LongTermMemory.__new__(LongTermMemory)
        lt.redis = _ScanRedis()
        lt.prefix = "memory:user:"
        assert lt.purge("u1") == 2
        assert len(lt.redis.deleted) == 2

    def test_conversation_purge_removes_all_keys_under_prefix(self):
        from src.memory.conversation import ConversationManager

        class _ScanRedis:
            def __init__(self):
                self.conn = self
                self.deleted: list = []
                self.keys = ["conv:s1:turns", "conv:s1:plan", "conv:s1:summary"]

            def scan_iter(self, match=None, count=None):
                return [k.encode() for k in self.keys]

            def delete(self, key):
                self.deleted.append(key)

        cm = ConversationManager.__new__(ConversationManager)
        cm.redis = _ScanRedis()
        assert cm.purge("s1") == 3
        assert len(cm.redis.deleted) == 3
