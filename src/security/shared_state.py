"""跨进程共享的限流与熔断状态。

单副本时进程内字典够用；多副本部署后，同一客户端会在每个副本各拿一份
配额，限流总上限被乘以副本数；熔断也一样——A 副本已经打开熔断，B 副本
仍会继续往故障上游打。

Redis 不可用时回退进程内。理由有两条：本地开发不该为了跑起来先起 Redis；
Redis 抖动时「限流退化为每实例限流」也比「所有请求 5xx」更能接受。
失败后会进入一个短暂的冷却期，避免每个请求都为连不上的 Redis 等一次超时。
"""

from __future__ import annotations

import logging
import sys
import time
from threading import Lock
from uuid import uuid4

from src.config import REDIS_HOST, REDIS_PORT, SHARED_STATE_BACKEND

logger = logging.getLogger(__name__)

#: Redis 故障后的静默窗口（秒）。期间不再尝试连接，直接走本地实现。
_DOWN_COOLDOWN = 5.0
#: 连接与读写超时。限流处在请求关键路径上，等太久比不限流更糟。
_SOCKET_TIMEOUT = 0.5

_RATE_LIMIT_LUA = """
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
local window = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local cutoff = now - window
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', cutoff)
local count = redis.call('ZCARD', KEYS[1])
if count >= limit then
  local oldest = redis.call('ZRANGE', KEYS[1], 0, 0, 'WITHSCORES')
  local retry = 1
  if oldest[2] then
    retry = math.max(1, math.ceil(tonumber(oldest[2]) + window - now))
  end
  return {0, retry}
end
redis.call('ZADD', KEYS[1], now, ARGV[3])
redis.call('PEXPIRE', KEYS[1], math.ceil(window * 1000))
return {1, 0}
"""


def _testing() -> bool:
    """测试进程强制走进程内实现。

    Redis 状态不随用例重启而清空，一旦测试机恰好跑着 Redis，用例之间就会
    互相看到对方留下的窗口与熔断标记。真实 Redis 的行为由标记为
    integration 的用例和 scripts/verify_shared_state.py 覆盖。
    """
    return "pytest" in sys.modules


def _make_connection():
    import redis

    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True,
        socket_connect_timeout=_SOCKET_TIMEOUT,
        socket_timeout=_SOCKET_TIMEOUT,
    )


class RedisRateLimiter:
    """基于 ZSET 的滑动窗口。

    时间取 Redis 服务器时间而不是各副本的本地时间，避免副本间时钟偏差
    把窗口切歪。
    """

    def __init__(self, conn=None, prefix: str = "ratelimit") -> None:
        self._conn = conn
        self._prefix = prefix
        self._down_until = 0.0
        self._lock = Lock()

    def set_connection(self, conn) -> None:
        """注入连接。测试与验证脚本用它替换默认连接。"""
        self._conn = conn

    def _connection(self):
        if self._conn is None:
            self._conn = _make_connection()
        return self._conn

    def allow(self, client: str, limit: int, window: float) -> tuple[bool, int] | None:
        """返回 ``(是否放行, 重试秒数)``；Redis 不可用时返回 ``None``。"""
        with self._lock:
            if time.monotonic() < self._down_until:
                return None
            try:
                result = self._connection().eval(
                    _RATE_LIMIT_LUA,
                    1,
                    f"{self._prefix}:{client}",
                    window,
                    limit,
                    f"{client}:{uuid4().hex}",
                )
                return bool(int(result[0])), int(result[1])
            except Exception as exc:  # noqa: BLE001 - 回退是预期路径
                self._down_until = time.monotonic() + _DOWN_COOLDOWN
                logger.warning("[shared-state] Redis 限流不可用，回退进程内：%s", exc)
                return None


class RedisCircuitState:
    """跨副本熔断状态。

    失败计数与打开时间戳分开存：计数用短 TTL 自然衰减，打开标记的 TTL
    等于冷却期，到期自动消失即半开。
    """

    def __init__(self, conn=None, prefix: str = "circuit") -> None:
        self._conn = conn
        self._prefix = prefix
        self._down_until = 0.0
        self._lock = Lock()

    def set_connection(self, conn) -> None:
        self._conn = conn

    def _connection(self):
        if self._conn is None:
            self._conn = _make_connection()
        return self._conn

    def _keys(self, alias: str) -> tuple[str, str]:
        return (f"{self._prefix}:{alias}:failures", f"{self._prefix}:{alias}:open_until")

    def _unavailable(self) -> bool:
        return time.monotonic() < self._down_until

    def _mark_down(self, exc: Exception) -> None:
        self._down_until = time.monotonic() + _DOWN_COOLDOWN
        logger.warning("[shared-state] Redis 熔断不可用，回退进程内：%s", exc)

    def record_failure(self, alias: str, threshold: int, cooldown: float) -> bool:
        """记录失败。返回 False 表示 Redis 不可用、调用方应回退。"""
        if self._unavailable():
            return False
        failures_key, open_key = self._keys(alias)
        try:
            conn = self._connection()
            count = int(conn.incr(failures_key))
            conn.expire(failures_key, max(1, int(cooldown * 2)))
            if count >= threshold:
                conn.set(open_key, time.time() + cooldown, ex=max(1, int(cooldown)))
                conn.delete(failures_key)
                logger.warning(
                    "[CircuitBreaker] OPEN for %s — %s failures within one replica set",
                    alias, threshold)
            return True
        except Exception as exc:  # noqa: BLE001
            self._mark_down(exc)
            return False

    def record_success(self, alias: str) -> bool:
        if self._unavailable():
            return False
        failures_key, open_key = self._keys(alias)
        try:
            conn = self._connection()
            existed = conn.exists(open_key) or conn.exists(failures_key)
            conn.delete(open_key, failures_key)
            if existed:
                logger.info("[CircuitBreaker] CLOSED for %s — success after failures", alias)
            return True
        except Exception as exc:  # noqa: BLE001
            self._mark_down(exc)
            return False

    def is_open(self, alias: str) -> bool | None:
        """返回 True/False；Redis 不可用时返回 None。"""
        if self._unavailable():
            return None
        _, open_key = self._keys(alias)
        try:
            raw = self._connection().get(open_key)
            if raw is None:
                return False
            return time.time() < float(raw)
        except Exception as exc:  # noqa: BLE001
            self._mark_down(exc)
            return None


def build_rate_limiter() -> RedisRateLimiter | None:
    """按配置构造共享限流器；memory 或测试环境返回 None。"""
    if _testing() or SHARED_STATE_BACKEND == "memory":
        return None
    if SHARED_STATE_BACKEND not in {"auto", "redis"}:
        logger.warning(
            "[shared-state] 未知 SHARED_STATE_BACKEND=%s，按 auto 处理",
            SHARED_STATE_BACKEND)
    limiter = RedisRateLimiter()
    if SHARED_STATE_BACKEND == "redis":
        # 显式选择 redis 时，连不上就立刻暴露问题，而不是静默降级。
        try:
            limiter._connection().ping()  # noqa: SLF001 - 启动期自检
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"SHARED_STATE_BACKEND=redis，但连不上 {REDIS_HOST}:{REDIS_PORT}") from exc
    return limiter


def build_circuit_backend() -> RedisCircuitState | None:
    if _testing() or SHARED_STATE_BACKEND == "memory":
        return None
    backend = RedisCircuitState()
    if SHARED_STATE_BACKEND == "redis":
        try:
            backend._connection().ping()  # noqa: SLF001
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"SHARED_STATE_BACKEND=redis，但连不上 {REDIS_HOST}:{REDIS_PORT}") from exc
    return backend


__all__ = [
    "RedisRateLimiter", "RedisCircuitState",
    "build_rate_limiter", "build_circuit_backend",
]
