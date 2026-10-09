"""用户账号存储层 — users 表的读写。

与 TrainingLogStore 同一套范式，理由也一样：这张表不在 ``init_db()`` 的调用
路径上（``init_db`` 只在 seed() 与知识库摄入时触发，纯 API 启动不会调），
所以必须自己幂等按需建表，并在建表失败时整体降级而不是把服务带崩。

降级语义：PostgreSQL 不可用时 ``enabled=False``，所有方法返回 None。
调用方（认证接口）转成 503 并说明原因——比抛异常好，因为数据库没起来
不该让整个应用无法启动，游客模式仍然可用。
"""

from __future__ import annotations

import logging
from datetime import datetime

from src.storage.pg import PGClient

logger = logging.getLogger(__name__)

# 用户名长度上限，与 User.username 的 String(64) 对齐
MAX_USERNAME_LEN = 64


class UserStore:
    """users 表的读写。

    只做存取，不含任何密码学逻辑——哈希与校验在 src/security/auth.py，
    这样存储层可以被 ORM/数据库的测试独立覆盖，不必构造真实密码。
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

        用 ORM 生成 DDL 而不是手写 CREATE TABLE：users 表本身就在
        ``db_models.User`` 这个单点真相里，手写 DDL 会重演
        knowledge_chunks.parent_id 那种「ORM 改了、建表语句忘了改」的漂移。
        """
        if self._table_ready:
            return
        try:
            from src.models.db_models import Base, User
            Base.metadata.create_all(
                self.db.engine, tables=[User.__table__], checkfirst=True)
            self._table_ready = True
        except Exception as exc:  # noqa: BLE001 - 存储不可用不应阻断启动
            logger.warning("用户表创建失败，账号功能降级禁用: %s", exc)
            self.enabled = False

    def _ready(self) -> bool:
        if not self._table_ready and self.enabled:
            self.ensure_table()
        return self._table_ready and self.enabled

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def create(self, username: str, password_hash: str) -> dict | None:
        """创建账号。用户名已存在时返回 None。

        用 ``ON CONFLICT DO NOTHING`` + RETURNING 而不是先查后插：
        先查后插在并发注册同一用户名时会有竞态，两个请求都能通过检查，
        然后其中一个以唯一约束报错。交给数据库裁决是原子的。

        必须走 transaction()：PGClient 的 fetch_one/fetch_all 只读不提交，
        写操作交给它们会静默回滚——语句"成功"返回了 RETURNING 的行，
        但数据没落库，唯一约束也就永远不会触发。
        """
        if not self._ready():
            return None
        username = normalize_username(username)
        with self.db.transaction() as tx:
            row = tx.fetch_one(
                """INSERT INTO users (username, password_hash, is_active, created_at, updated_at)
                   VALUES (:un, :ph, TRUE, :ts, :ts)
                   ON CONFLICT (username) DO NOTHING
                   RETURNING id, username, is_active""",
                {"un": username, "ph": password_hash, "ts": datetime.utcnow()},
            )
            return dict(row._mapping) if row else None

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------
    def find_by_username(self, username: str) -> dict | None:
        """按用户名取账号（含 password_hash，仅供认证校验使用）。"""
        if not self._ready():
            return None
        row = self.db.fetch_one(
            """SELECT id, username, password_hash, is_active
               FROM users WHERE username = :un""",
            {"un": normalize_username(username)},
        )
        return dict(row._mapping) if row else None

    def find_by_id(self, user_id: int) -> dict | None:
        """按 id 取账号（不含 password_hash——多数调用点不需要它）。"""
        if not self._ready():
            return None
        row = self.db.fetch_one(
            """SELECT id, username, is_active
               FROM users WHERE id = :uid""",
            {"uid": int(user_id)},
        )
        return dict(row._mapping) if row else None


def normalize_username(username: str) -> str:
    """用户名归一化：去首尾空白并截断到上限。

    不做大小写折叠——中文用户名没有大小写概念，而对英文用户名做折叠会
    让 ``Alice`` 与 ``alice`` 静默变成同一个账号，用户会以为"账号被盗了"。
    唯一性由数据库的 unique 约束按原样判定。
    """
    if not isinstance(username, str):
        return ""
    return username.strip()[:MAX_USERNAME_LEN]
