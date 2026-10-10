"""会话存储与身份解析。

这一层只做两件事，都不依赖 FastAPI，因此可以被独立测试：
1. 会话令牌 ↔ user_id 的存取（Redis）；
2. 把「登录用户」或「游客」解析成业务层使用的身份键 ``owner_key``。

【为什么业务身份统一叫 owner_key】
    训练日志、长期记忆、数据导出/删除原本各自读前端传来的 ``athlete_key``。
    引入账号后，这些地方需要的是「这条数据属于谁」——登录用户属于账号，
    游客属于本机 UUID。把两者的归一收进一个函数，是为了让归属判断只有
    一处实现；散在各路由里判断，早晚会有一处漏掉，那就是越权。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from src.security.auth import (
    SESSION_TTL_SECONDS,
    new_session_token,
    session_key,
)
from src.storage.redis_client import RedisClient

logger = logging.getLogger(__name__)

# 登录用户身份键的前缀。用前缀而不是裸 id，是为了让「账号 u:3」与
# 「游客的随机 UUID」在同一命名空间里不会撞车——否则一个游客把自己的
# athlete_key 设成 "3" 就落到了账号 3 的数据上。
USER_KEY_PREFIX = "u:"

from src.storage.training_log_store import MAX_ATHLETE_KEY_LEN as _MAX_KEY_LEN

# 游客身份键的合法字符。历史数据里有 UUID（十六进制 + 连字符），
# 演示数据是 "demo-athlete"，都用这个集合覆盖。长度上限直接取自
# 存储层的常量，而不是在这里再写一个 64——否则改了一处忘了另一处，
# 就会出现"校验放行、入库被截断"或反过来。
_GUEST_KEY_RE = re.compile(rf"^[A-Za-z0-9._-]{{1,{_MAX_KEY_LEN}}}$")


def is_safe_guest_key(key: str | None) -> bool:
    """游客身份键是否可安全进入存储路径。

    不要绕过这个函数直接把客户端传来的 key 拼进 Redis 键名或 SCAN 的
    match 模式：长期记忆的认领（``LongTermMemory.rekey``）正是按
    ``{prefix}{key}:*`` 做前缀扫描的，一旦放行 ``*``、``?``、``[``
    这类 glob 字符，一次请求就能匹配并搬走/删除**所有**用户的记忆。
    校验规则与 ``resolve_owner_key`` 保持同一份正则，避免两处漂移。
    """
    if not key:
        return False
    stripped = key.strip()
    return bool(stripped) and bool(_GUEST_KEY_RE.match(stripped))


class GuestKeyError(ValueError):
    """游客身份键不合法。"""


@dataclass(frozen=True)
class Principal:
    """当前请求的身份。``user_id`` 为 None 表示游客。"""

    user_id: int | None = None
    username: str | None = None

    @property
    def is_authenticated(self) -> bool:
        return self.user_id is not None


ANONYMOUS = Principal()


def user_owner_key(user_id: int) -> str:
    """账号对应的业务身份键。"""
    return f"{USER_KEY_PREFIX}{int(user_id)}"


def is_account_key(key: str | None) -> bool:
    """判断一个身份键是不是账号键（而非游客键）。"""
    return bool(key) and key.startswith(USER_KEY_PREFIX)


def resolve_owner_key(principal: Principal, client_key: str | None) -> str:
    """把请求解析成业务数据归属的 ``owner_key``。

    登录用户：一律用 ``u:{id}``，**忽略客户端传来的 key**。这一步是防越权的
    关键——否则任何登录用户只要把 athlete_key 改成别人的，就能读写他人数据；
    而 athlete_key 本来就在前端 localStorage 里，攻击者自己就能编。

    游客：沿用客户端提供的 UUID（本机数据只在本机可见，这是既有的可见性
    边界，不是新增的弱点）。但必须校验格式，因为游客 key 会直接进入
    Redis 键名与 SQL 参数——不做校验就等于把任意字符串放进存储路径。
    """
    if principal.user_id is not None:
        return user_owner_key(principal.user_id)

    key = (client_key or "").strip()
    if not key or len(key) > _MAX_KEY_LEN or not _GUEST_KEY_RE.match(key):
        raise GuestKeyError(
            "游客身份标识缺失或格式不合法；请携带有效的 athlete_key，或先登录。")
    return key


class SessionStore:
    """会话令牌的存取。

    只存令牌的 sha256 摘要（见 src/security/auth.py 的说明），
    因此即使 Redis 内容被读走，也无法反推出可用的令牌。
    """

    def __init__(self, redis: RedisClient | None = None):
        self.redis = redis or RedisClient()

    def create(self, user_id: int, username: str = "") -> str:
        """签发会话，返回明文令牌（仅此一次返回给客户端）。"""
        token = new_session_token()
        self.redis.set(
            session_key(token),
            json.dumps({"user_id": int(user_id), "username": username}),
            ex=SESSION_TTL_SECONDS,
        )
        return token

    def resolve(self, token: str | None) -> Principal:
        """把令牌解析为身份；无效或过期返回匿名。"""
        if not token:
            return ANONYMOUS
        try:
            raw = self.redis.get(session_key(token))
        except Exception as exc:  # noqa: BLE001 - Redis 抖动不应把请求变成 500
            logger.warning("[session] 会话读取失败，按未登录处理: %s", exc)
            return ANONYMOUS
        if not raw:
            return ANONYMOUS
        try:
            data = json.loads(raw)
            return Principal(user_id=int(data["user_id"]),
                             username=data.get("username") or None)
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("[session] 会话载荷损坏，按未登录处理: %s", exc)
            return ANONYMOUS

    def revoke(self, token: str | None) -> None:
        """登出。删除失败只记日志——客户端仍会丢弃 Cookie，效果等价。"""
        if not token:
            return
        try:
            self.redis.delete(session_key(token))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[session] 会话删除失败: %s", exc)
