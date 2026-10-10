"""训练计划的存储层 — training_plans 表的读写。

与 TrainingLogStore 同一套范式：这张表不在 ``init_db()`` 的调用路径上
（纯 API 启动不会调它），因此必须自己幂等按需建表，建表失败时整体降级。

为什么值得存到服务端：计划此前只写在浏览器 localStorage 里（上限 10 条、
清缓存即丢）。对一个刚加上账号体系的应用来说，"登录后能找回自己的计划"
与"登录后能找回自己的记忆"是同一件事的两面——只做后者会让前者显得突兀。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from src.storage.pg import PGClient

logger = logging.getLogger(__name__)

# 与 training_logs.athlete_key 同一上限
MAX_OWNER_KEY_LEN = 64

# 单次列表返回条数上限
MAX_PLANS_PER_QUERY = 100

#: 计划正文里不该进库的字段：它们是本次运行的中间产物，不是计划的定义。
#: 存下来会让历史记录里混入过期的解释块与内部标志。
_VOLATILE_KEYS = (
    "_degraded", "_persistence_allowed", "_delivery_allowed",
    "active_issues", "resolved_issues", "review_suggestions",
    "review_reason", "review_severity", "refresh_payload",
)


class PlanStore:
    """training_plans 表的读写。

    降级语义与 TrainingLogStore 一致：存储不可用时 ``enabled=False``，
    所有方法返回空值，由 API 层转 503——不把计划生成链路一起带崩。
    """

    def __init__(self):
        self.db = PGClient()
        self.enabled = True
        self._table_ready = False

    # ------------------------------------------------------------------
    # 建表
    # ------------------------------------------------------------------
    def ensure_table(self) -> None:
        """按需建表。幂等；失败只记日志并降级。

        这条路径要同时服务两种库：全新建的库由 ORM 直接生成正确结构；
        早就按旧定义建过表的库（user_id INTEGER NOT NULL、没有 owner_key）
        需要就地改造。旧表从未写入过任何一行——它定义了却没人用，
        所以补列与解除 NOT NULL 都没有数据风险。
        """
        if self._table_ready:
            return
        try:
            from src.models.db_models import Base, TrainingPlan
            Base.metadata.create_all(self.db.engine,
                                     tables=[TrainingPlan.__table__],
                                     checkfirst=True)
            self._migrate_legacy_table()
            self._table_ready = True
        except Exception as exc:  # noqa: BLE001 - 存储不可用不应阻断主链路
            logger.warning("训练计划表创建失败，计划保存功能降级禁用: %s", exc)
            self.enabled = False

    def _migrate_legacy_table(self) -> None:
        """把旧定义的 training_plans 补齐成当前结构（仅对已存在的旧库生效）。"""
        with self.db.transaction() as tx:
            cols = {r.column_name for r in tx.fetch_all(
                """SELECT column_name FROM information_schema.columns
                   WHERE table_name = 'training_plans'""")}
            if "owner_key" not in cols:
                tx.execute("ALTER TABLE training_plans ADD COLUMN owner_key VARCHAR(64)")
                tx.execute("CREATE INDEX IF NOT EXISTS ix_training_plans_owner_key "
                           "ON training_plans (owner_key)")
            # 旧定义的 user_id 是 NOT NULL，而我们现在不写它
            tx.execute("ALTER TABLE training_plans ALTER COLUMN user_id DROP NOT NULL")
            # 旧库可能已经建过表但没有任何索引
            tx.execute("CREATE INDEX IF NOT EXISTS ix_training_plans_owner_key "
                       "ON training_plans (owner_key)")

    def _ready(self) -> bool:
        if not self._table_ready and self.enabled:
            self.ensure_table()
        return self._table_ready and self.enabled

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def save(self, owner_key: str, plan: dict, *, title: str = "") -> int | None:
        """保存一份计划快照。返回 plan_id，存储不可用时返回 None。

        ``title`` 只用于列表展示，不单独建列——它可以从 ``plan_data`` 派生，
        多存一列就多一个会与正文不一致的地方。
        """
        if not self._ready():
            return None
        if not isinstance(plan, dict) or not plan.get("days"):
            # 没有训练日的"计划"没有保存价值，存下来只会污染历史列表
            return None

        clean = {k: v for k, v in plan.items() if k not in _VOLATILE_KEYS}
        owner = (owner_key or "")[:MAX_OWNER_KEY_LEN]
        if not owner:
            return None

        with self.db.transaction() as tx:
            row = tx.fetch_one(
                """INSERT INTO training_plans
                   (owner_key, goal, plan_data, confidence, created_at)
                   VALUES (:ok, :goal, :data, :conf, :ts)
                   RETURNING id""",
                {"ok": owner, "goal": (clean.get("goal") or "")[:20],
                 "data": json.dumps(clean, ensure_ascii=False),
                 "conf": float(clean.get("confidence") or 0.0),
                 "ts": datetime.utcnow()},
            )
            return int(row.id) if row else None

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------
    def list_plans(self, owner_key: str, limit: int = 20) -> list[dict]:
        """按身份列出计划（不含完整正文，避免列表接口返回过大数据）。"""
        if not self._ready():
            return []
        rows = self.db.fetch_all(
            """SELECT id, goal, plan_data, confidence, created_at
               FROM training_plans WHERE owner_key = :ok
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
                "title": data.get("plan_name") or f"{r.goal or ''}计划".strip(),
                "goal": r.goal,
                "day_count": len(days),
                "confidence": r.confidence,
                "created_at": r.created_at.isoformat() if r.created_at else "",
            })
        return out

    def get_plan(self, owner_key: str, plan_id: int) -> dict | None:
        """取单份计划的完整正文。归属不匹配时返回 None。

        归属校验放在 SQL 的 WHERE 里而不是取出来再判断：那样即使判错也只是
        拿不到数据，不会先把别人的计划读进内存。
        """
        if not self._ready():
            return None
        row = self.db.fetch_one(
            """SELECT id, goal, plan_data, confidence, created_at
               FROM training_plans WHERE id = :pid AND owner_key = :ok""",
            {"pid": int(plan_id), "ok": (owner_key or "")[:MAX_OWNER_KEY_LEN]},
        )
        if row is None:
            return None
        data = row.plan_data if isinstance(row.plan_data, dict) else json.loads(row.plan_data or "{}")
        data["plan_id"] = row.id
        data["saved_at"] = row.created_at.isoformat() if row.created_at else ""
        return data

    def rename_plan(self, owner_key: str, plan_id: int, title: str) -> bool:
        """给已保存的计划改名字。返回是否确实改了一行。

        名字存在 ``plan_data.plan_name`` 里，不单独建列——列表展示名正是从
        这个字段派生的（见 ``list_plans``）。改这里，列表与详情会同时变，
        不会出现"列表显示新名、点进去还是旧名"的两处不一致。

        调用方需保证 ``title`` 已校验（非空、未超长）；本方法只负责写。
        返回 False 只有一种含义：该计划不存在，或不属于这个身份。
        """
        if not self._ready():
            return False
        owner = (owner_key or "")[:MAX_OWNER_KEY_LEN]
        with self.db.transaction() as tx:
            # FOR UPDATE：读-改-写之间若有人并发改同一行，后到的会覆盖先到的，
            # 用户刚起的名字会凭空变回去。个人计划并发极低，但加锁成本是零。
            row = tx.fetch_one(
                """SELECT plan_data FROM training_plans
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
                """UPDATE training_plans SET plan_data = :data
                   WHERE id = :pid AND owner_key = :ok""",
                {"data": json.dumps(data, ensure_ascii=False),
                 "pid": int(plan_id), "ok": owner},
            )
            return True

    def delete_plan(self, owner_key: str, plan_id: int) -> bool:
        """删除一份计划。返回是否确实删掉了一行。

        先查后删（同一事务内）：DELETE 的影响行数在 ``text()`` 查询里拿不到，
        删完再数只会得到 0，于是"删成功"和"本来就不存在"无法区分——
        接口会把后者也报成成功。
        """
        if not self._ready():
            return False
        with self.db.transaction() as tx:
            row = tx.fetch_one(
                """SELECT COUNT(*) AS c FROM training_plans
                   WHERE id = :pid AND owner_key = :ok""",
                {"pid": int(plan_id), "ok": (owner_key or "")[:MAX_OWNER_KEY_LEN]},
            )
            if not row or int(row.c) == 0:
                return False
            tx.execute("DELETE FROM training_plans WHERE id = :pid AND owner_key = :ok",
                       {"pid": int(plan_id), "ok": (owner_key or "")[:MAX_OWNER_KEY_LEN]})
            return True
