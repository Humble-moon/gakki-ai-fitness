"""饮食计划的存储层 — diet_plans 表的读写。

克隆 ``plan_store.PlanStore`` 范式：幂等按需建表（纯 API 启动不走 init_db）、
建表失败整体降级 ``enabled=False``（API 转 503，不拖垮生成链路）、归属校验
写进 SQL WHERE、入库剥 volatile 键、rename 用 FOR UPDATE、delete 先数后删。

与训练侧的唯一差异：没有旧库就地迁移分支——diet_plans 是新增表，不存在
「按旧定义建过表」的历史包袱。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from src.storage.pg import PGClient

logger = logging.getLogger(__name__)

MAX_OWNER_KEY_LEN = 64
MAX_PLANS_PER_QUERY = 100

#: 与训练侧同口径：本次运行的中间产物不进库。
_VOLATILE_KEYS = (
    "_degraded", "_persistence_allowed", "_delivery_allowed",
    "active_issues", "resolved_issues", "review_suggestions",
    "review_reason", "review_severity", "refresh_payload",
    "_safety_block", "_delivery_block_reasons",
)


class DietPlanStore:
    """diet_plans 表的读写。降级语义与 PlanStore 一致。"""

    def __init__(self):
        self.db = PGClient()
        self.enabled = True
        self._table_ready = False

    def ensure_table(self) -> None:
        if self._table_ready:
            return
        try:
            from src.models.db_models import Base, DietPlan
            Base.metadata.create_all(self.db.engine, tables=[DietPlan.__table__],
                                     checkfirst=True)
            self._table_ready = True
        except Exception as exc:  # noqa: BLE001 - 存储不可用不应阻断主链路
            logger.warning("饮食计划表创建失败，饮食计划保存功能降级禁用: %s", exc)
            self.enabled = False

    def _ready(self) -> bool:
        if not self._table_ready and self.enabled:
            self.ensure_table()
        return self._table_ready and self.enabled

    def save(self, owner_key: str, plan: dict) -> int | None:
        """保存一份饮食计划快照。返回 id；无 days 或存储不可用返回 None。"""
        if not self._ready():
            return None
        if not isinstance(plan, dict) or not plan.get("days"):
            return None
        clean = {k: v for k, v in plan.items() if k not in _VOLATILE_KEYS}
        owner = (owner_key or "")[:MAX_OWNER_KEY_LEN]
        if not owner:
            return None
        with self.db.transaction() as tx:
            row = tx.fetch_one(
                """INSERT INTO diet_plans
                   (owner_key, goal, plan_data, confidence, created_at)
                   VALUES (:ok, :goal, :data, :conf, :ts)
                   RETURNING id""",
                {"ok": owner, "goal": (clean.get("goal") or "")[:20],
                 "data": json.dumps(clean, ensure_ascii=False),
                 "conf": float(clean.get("confidence") or 0.0),
                 "ts": datetime.utcnow()},
            )
            return int(row.id) if row else None

    def list_plans(self, owner_key: str, limit: int = 20) -> list[dict]:
        if not self._ready():
            return []
        rows = self.db.fetch_all(
            """SELECT id, goal, plan_data, confidence, created_at
               FROM diet_plans WHERE owner_key = :ok
               ORDER BY created_at DESC LIMIT :lim""",
            {"ok": (owner_key or "")[:MAX_OWNER_KEY_LEN],
             "lim": max(1, min(int(limit), MAX_PLANS_PER_QUERY))},
        )
        out = []
        for r in rows:
            data = r.plan_data if isinstance(r.plan_data, dict) else json.loads(r.plan_data or "{}")
            days = data.get("days") or []
            out.append({
                "plan_id": r.id,
                "title": data.get("plan_name") or f"{r.goal or ''}饮食计划".strip(),
                "goal": r.goal,
                "day_count": len(days),
                "confidence": r.confidence,
                "created_at": r.created_at.isoformat() if r.created_at else "",
            })
        return out

    def get_plan(self, owner_key: str, plan_id: int) -> dict | None:
        """归属校验在 WHERE 里：判错也只是拿不到，不会先把别人的计划读进内存。"""
        if not self._ready():
            return None
        row = self.db.fetch_one(
            """SELECT id, goal, plan_data, confidence, created_at
               FROM diet_plans WHERE id = :pid AND owner_key = :ok""",
            {"pid": int(plan_id), "ok": (owner_key or "")[:MAX_OWNER_KEY_LEN]},
        )
        if row is None:
            return None
        data = row.plan_data if isinstance(row.plan_data, dict) else json.loads(row.plan_data or "{}")
        data["plan_id"] = row.id
        data["saved_at"] = row.created_at.isoformat() if row.created_at else ""
        return data

    def rename_plan(self, owner_key: str, plan_id: int, title: str) -> bool:
        if not self._ready():
            return False
        owner = (owner_key or "")[:MAX_OWNER_KEY_LEN]
        with self.db.transaction() as tx:
            row = tx.fetch_one(
                """SELECT plan_data FROM diet_plans
                   WHERE id = :pid AND owner_key = :ok FOR UPDATE""",
                {"pid": int(plan_id), "ok": owner},
            )
            if row is None:
                return False
            data = row.plan_data if isinstance(row.plan_data, dict) else json.loads(row.plan_data or "{}")
            if not isinstance(data, dict):
                data = {}
            data["plan_name"] = title
            tx.execute(
                """UPDATE diet_plans SET plan_data = :data
                   WHERE id = :pid AND owner_key = :ok""",
                {"data": json.dumps(data, ensure_ascii=False),
                 "pid": int(plan_id), "ok": owner},
            )
            return True

    def delete_plan(self, owner_key: str, plan_id: int) -> bool:
        if not self._ready():
            return False
        owner = (owner_key or "")[:MAX_OWNER_KEY_LEN]
        with self.db.transaction() as tx:
            row = tx.fetch_one(
                """SELECT COUNT(*) AS c FROM diet_plans
                   WHERE id = :pid AND owner_key = :ok""",
                {"pid": int(plan_id), "ok": owner},
            )
            if not row or int(row.c) == 0:
                return False
            tx.execute("DELETE FROM diet_plans WHERE id = :pid AND owner_key = :ok",
                       {"pid": int(plan_id), "ok": owner})
            return True

    def delete_all(self, owner_key: str) -> int:
        """「删除我的数据」用。先数后删（DELETE 影响行数在 text() 里拿不到）。"""
        if not self._ready():
            return 0
        owner = (owner_key or "")[:MAX_OWNER_KEY_LEN]
        if not owner:
            return 0
        with self.db.transaction() as tx:
            row = tx.fetch_one("SELECT COUNT(*) AS c FROM diet_plans WHERE owner_key = :ok",
                               {"ok": owner})
            count = int(row.c) if row else 0
            if count == 0:
                return 0
            tx.execute("DELETE FROM diet_plans WHERE owner_key = :ok", {"ok": owner})
            return count

    def rekey(self, old_key: str, new_key: str) -> int:
        """游客注册认领：旧身份名下计划改挂新身份。无 UNIQUE 约束，直接改。"""
        if not self._ready():
            return 0
        old_key = (old_key or "")[:MAX_OWNER_KEY_LEN]
        new_key = (new_key or "")[:MAX_OWNER_KEY_LEN]
        if not old_key or not new_key or old_key == new_key:
            return 0
        with self.db.transaction() as tx:
            row = tx.fetch_one("SELECT COUNT(*) AS c FROM diet_plans WHERE owner_key = :ok",
                               {"ok": old_key})
            count = int(row.c) if row else 0
            if count:
                tx.execute("UPDATE diet_plans SET owner_key = :new WHERE owner_key = :old",
                           {"new": new_key, "old": old_key})
            return count