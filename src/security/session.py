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
import secrets
from dataclasses import dataclass

from src.config import ALLOW_LEGACY_GUEST_KEY, GUEST_TOKEN_TTL_SECONDS
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


GUEST_TOKEN_PREFIX = "g1_"
GUEST_TOKEN_KEY_PREFIX = "guest:token:"

_guest_store: "GuestIdentityStore | None" = None


def new_guest_token() -> str:
    """生成不可猜测的游客令牌。

    前缀不只是命名空间：``resolve_owner_key`` 靠它区分"服务端签发的令牌"
    与"旧版客户端自己编的 key"，两者必须走完全不同的信任路径。
    """
    return GUEST_TOKEN_PREFIX + secrets.token_hex(16)


class GuestIdentityStore:
    """游客令牌 ↔ 业务归属键的映射，落在 Redis。

    令牌本身是随机值、不含任何业务信息；归属键单独存，因此同一个
    owner_key 可以有多枚令牌（换设备、重新签发）而不影响数据归属。
    """

    def __init__(self, redis: RedisClient | None = None):
        self.redis = redis or RedisClient()

    def issue(self, legacy_key: str | None = None) -> tuple[str, str]:
        """签发 ``(token, owner_key)``。

        传入合法的 legacy_key 时沿用它作为 owner_key——老用户的数据按
        原来的 UUID 存放，换令牌不能换归属键，否则历史数据就找不到了。
        传入账号键或非法值时一律新建，避免把账号数据挂到游客名下。
        """
        owner_key = (legacy_key or "").strip()
        if (not owner_key or not is_safe_guest_key(owner_key)
                or is_account_key(owner_key)):
            owner_key = "guest-" + secrets.token_hex(8)
        token = new_guest_token()
        self.redis.set(f"{GUEST_TOKEN_KEY_PREFIX}{token}", owner_key,
                       ex=GUEST_TOKEN_TTL_SECONDS)
        return token, owner_key

    def resolve(self, token: str) -> str | None:
        """令牌 → owner_key；无效返回 None，Redis 故障抛 GuestKeyError。

        故障时不放行：身份解析是安全边界，拒绝请求比猜一个 key 更安全。
        """
        if not token.startswith(GUEST_TOKEN_PREFIX):
            return None
        try:
            value = self.redis.get(f"{GUEST_TOKEN_KEY_PREFIX}{token}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[guest] 令牌解析失败: %s", exc)
            raise GuestKeyError("游客身份服务暂时不可用，请稍后重试。") from exc
        return value or None


def get_guest_identity_store() -> GuestIdentityStore:
    """进程级单例。Redis 客户端本身是惰性连接，构造代价可以忽略。"""
    global _guest_store
    if _guest_store is None:
        _guest_store = GuestIdentityStore()
    return _guest_store


def reset_guest_identity_store() -> None:
    """测试用：丢弃单例，让下一次调用重新读取 Redis 配置。"""
    global _guest_store
    _guest_store = None


def resolve_owner_key(principal: Principal, client_key: str | None) -> str:
    """把请求解析成业务数据归属的 ``owner_key``。

    登录用户：一律用 ``u:{id}``，**忽略客户端传来的 key**。否则任何登录
    用户只要改一下请求里的 key，就能读写他人数据。

    游客：只接受服务端签发的令牌（``g1_`` 前缀），令牌背后映射到一个
    owner_key。旧版客户端自己编的 UUID 默认拒绝——那不是所有权证明，
    只是格式合法的字符串；知道别人的 key 就能读写对方数据。
    """
    if principal.user_id is not None:
        return user_owner_key(principal.user_id)

    key = (client_key or "").strip()
    if not key or len(key) > _MAX_KEY_LEN:
        raise GuestKeyError(
            "游客身份标识缺失或格式不合法；请先调用 /api/auth/guest 获取令牌。")

    if key.startswith(GUEST_TOKEN_PREFIX):
        owner = get_guest_identity_store().resolve(key)
        if owner is None:
            raise GuestKeyError("游客身份令牌无效或已过期；请重新获取身份。")
        return owner

    # 旧版裸 key：默认关闭，只在一次性迁移窗口内临时放行。
    if not ALLOW_LEGACY_GUEST_KEY:
        raise GuestKeyError(
            "缺少服务端签发的游客身份令牌；请先调用 /api/auth/guest 获取。")
    if not _GUEST_KEY_RE.match(key):
        raise GuestKeyError("游客身份标识格式不合法。")
    return key


def scoped_session_id(owner_key: str, raw_session_id: str | None) -> str | None:
    """把客户端会话 ID 收进 owner_key 命名空间。

    会话记忆与上传文档此前只按客户端传来的 session_id 分区——任何知道
    这个值的人都能读到别人的对话和文档。加上 owner 前缀后，跨身份访问
    需要同时拿到两段信息，而不是只猜一个 ID。函数是幂等的：已是本身份
    命名空间的 ID 原样返回，避免前端回传时叠加前缀。
    """
    if not raw_session_id:
        return raw_session_id
    prefix = f"{owner_key}:"
    if raw_session_id.startswith(prefix):
        return raw_session_id
    return prefix + raw_session_id


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
