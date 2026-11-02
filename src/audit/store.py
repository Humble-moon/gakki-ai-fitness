"""审计事件表：谁在什么时候对什么做了什么。

``review_resolutions`` 里已经有审核决定，但那是业务状态而不是审计记录：
它只覆盖「审核被解除」这一种事件，没有请求关联 ID、没有失败结果，
也不记录导出和删除。这里用一张只追加的表把这条线补齐。

三条约束：
* **只追加**：不提供 update/delete。审计表一旦允许改写，就不能再用来
  回答「当时到底发生了什么」。
* **不阻塞业务**：写审计失败只记日志，不把用户请求带崩。
* **不存原文**：``detail`` 只放标识与结论，用户画像和问题原文不进来。
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/audit.db"


class AuditStore:
    """SQLite 追加表。与 review 存储分文件，互不影响各自的迁移与清理。"""

    def __init__(self, db_path: str = "") -> None:
        self._db_path = db_path or os.getenv("AUDIT_DB_PATH", DEFAULT_DB_PATH)
        self._lock = threading.Lock()
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._connect() as conn:
            self._ensure_schema(conn)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                occurred_at TEXT NOT NULL,
                actor TEXT NOT NULL,
                action TEXT NOT NULL,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                outcome TEXT NOT NULL,
                request_id TEXT NOT NULL,
                detail TEXT NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_audit_events_occurred_at "
            "ON audit_events(occurred_at DESC)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_audit_events_actor ON audit_events(actor)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_audit_events_action ON audit_events(action)")

    def record(self, action: str, actor: str = "", target_type: str = "",
               target_id: str = "", outcome: str = "success",
               detail: dict | None = None) -> str:
        """写一条事件，返回 ``event_id``。

        失败时只记日志：审计是旁路，不能因为磁盘满或锁等待把主流程拖垮。
        ``detail`` 截断到 4000 字符，避免把大对象写进来撑爆表。
        """
        event_id = uuid4().hex
        try:
            from src.observability.logging import request_id_var
            request_id = request_id_var.get()
        except Exception:  # noqa: BLE001 - 日志模块不可用不应影响审计
            request_id = ""

        payload = json.dumps(detail or {}, ensure_ascii=False, default=str)
        if len(payload) > 4000:
            # 直接截断会把 JSON 切成半截，读取侧只能得到空对象；
            # 包一层 preview 至少让"发生过截断"这件事本身可追溯。
            payload = json.dumps(
                {"_truncated": True, "preview": payload[:3800]},
                ensure_ascii=False)
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    """INSERT INTO audit_events
                       (event_id, occurred_at, actor, action, target_type,
                        target_id, outcome, request_id, detail)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (event_id, datetime.now(timezone.utc).isoformat(),
                     actor or "anonymous", action, target_type, target_id,
                     outcome, request_id, payload),
                )
        except Exception:  # noqa: BLE001
            logger.exception("[audit] failed to record action=%s", action)
        return event_id

    def list_events(self, limit: int = 100, action: str | None = None,
                    actor: str | None = None) -> list[dict]:
        clauses, params = [], []
        if action:
            clauses.append("action = ?")
            params.append(action)
        if actor:
            clauses.append("actor = ?")
            params.append(actor)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(max(1, min(int(limit), 1000)))
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM audit_events{where} ORDER BY id DESC LIMIT ?",
                params,
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def count(self, action: str | None = None) -> int:
        with self._lock, self._connect() as conn:
            if action:
                row = conn.execute(
                    "SELECT COUNT(*) FROM audit_events WHERE action = ?",
                    (action,)).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM audit_events").fetchone()
        return int(row[0]) if row else 0

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        try:
            detail = json.loads(row["detail"])
        except (TypeError, ValueError):
            detail = {}
        return {
            "event_id": row["event_id"],
            "occurred_at": row["occurred_at"],
            "actor": row["actor"],
            "action": row["action"],
            "target_type": row["target_type"],
            "target_id": row["target_id"],
            "outcome": row["outcome"],
            "request_id": row["request_id"],
            "detail": detail,
        }


_default_store: AuditStore | None = None


def get_audit_store() -> AuditStore:
    """进程内单例。首次调用时才建库，导入模块不会创建文件。"""
    global _default_store
    if _default_store is None:
        _default_store = AuditStore()
    return _default_store


def record_event(action: str, **kwargs) -> str:
    return get_audit_store().record(action, **kwargs)


__all__ = ["AuditStore", "get_audit_store", "record_event"]
