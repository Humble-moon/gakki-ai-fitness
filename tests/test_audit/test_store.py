"""审计表：写入、过滤、截断与失败不阻塞。"""

import json

from src.audit.store import AuditStore
from src.observability.logging import request_id_var


def test_record_and_read_back(tmp_path):
    store = AuditStore(str(tmp_path / "audit.db"))
    event_id = store.record(
        "review.resolved", actor="u:1", target_type="review",
        target_id="r-1", detail={"decision": "approve"})

    events = store.list_events()
    assert len(events) == 1
    assert events[0]["event_id"] == event_id
    assert events[0]["actor"] == "u:1"
    assert events[0]["action"] == "review.resolved"
    assert events[0]["target_id"] == "r-1"
    assert events[0]["detail"] == {"decision": "approve"}
    assert events[0]["outcome"] == "success"


def test_events_are_returned_newest_first(tmp_path):
    store = AuditStore(str(tmp_path / "audit.db"))
    store.record("a", actor="x")
    store.record("b", actor="x")
    store.record("c", actor="x")

    assert [e["action"] for e in store.list_events()] == ["c", "b", "a"]


def test_filter_by_action_and_actor(tmp_path):
    store = AuditStore(str(tmp_path / "audit.db"))
    store.record("auth.login", actor="u:1")
    store.record("auth.login", actor="u:2")
    store.record("data.exported", actor="u:1")

    assert [e["actor"] for e in store.list_events(action="auth.login")] == ["u:2", "u:1"]
    assert [e["action"] for e in store.list_events(actor="u:1")] == ["data.exported", "auth.login"]
    assert store.count(action="auth.login") == 2
    assert store.count() == 3


def test_limit_is_applied(tmp_path):
    store = AuditStore(str(tmp_path / "audit.db"))
    for index in range(5):
        store.record("e", target_id=str(index))
    assert len(store.list_events(limit=2)) == 2


def test_detail_is_truncated_into_valid_json(tmp_path):
    store = AuditStore(str(tmp_path / "audit.db"))
    store.record("e", detail={"blob": "x" * 10000})

    event = store.list_events()[0]
    assert event["detail"]["_truncated"] is True
    assert event["detail"]["preview"].startswith("{")
    assert len(json.dumps(event["detail"], ensure_ascii=False)) <= 4100


def test_request_id_is_captured_from_context(tmp_path):
    store = AuditStore(str(tmp_path / "audit.db"))
    token = request_id_var.set("req-abc")
    try:
        store.record("e")
    finally:
        request_id_var.reset(token)

    assert store.list_events()[0]["request_id"] == "req-abc"


def test_write_failure_does_not_raise(tmp_path, monkeypatch):
    import sqlite3

    store = AuditStore(str(tmp_path / "audit.db"))

    def locked():
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(store, "_connect", locked)
    # 审计是旁路：写入失败必须被吞掉，不能把主请求带崩。
    assert store.record("e") is not None
