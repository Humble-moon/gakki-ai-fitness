"""共享限流与熔断：Redis 故障时回退，不把请求带崩。"""

import time

import pytest

from src.security.api_guard import RateLimitMiddleware
from src.security.shared_state import RedisCircuitState, RedisRateLimiter


class _BrokenConnection:
    def eval(self, *args, **kwargs):
        raise RuntimeError("redis down")

    def ping(self):
        raise RuntimeError("redis down")

    def incr(self, *args, **kwargs):
        raise RuntimeError("redis down")

    def expire(self, *args, **kwargs):
        raise RuntimeError("redis down")

    def set(self, *args, **kwargs):
        raise RuntimeError("redis down")

    def get(self, *args, **kwargs):
        raise RuntimeError("redis down")

    def delete(self, *args, **kwargs):
        raise RuntimeError("redis down")

    def exists(self, *args, **kwargs):
        raise RuntimeError("redis down")


def test_rate_limiter_falls_back_when_redis_is_down():
    limiter = RedisRateLimiter(conn=_BrokenConnection())
    assert limiter.allow("1.2.3.4", limit=10, window=60) is None


def test_rate_limiter_stops_retrying_during_cooldown(monkeypatch):
    calls = {"count": 0}

    class CountingBroken(_BrokenConnection):
        def eval(self, *args, **kwargs):
            calls["count"] += 1
            raise RuntimeError("redis down")

    limiter = RedisRateLimiter(conn=CountingBroken())
    limiter.allow("1.2.3.4", limit=10, window=60)
    limiter.allow("1.2.3.4", limit=10, window=60)
    limiter.allow("1.2.3.4", limit=10, window=60)
    # 第一次失败后进入冷却期，不再为每个请求等一次超时
    assert calls["count"] == 1


def test_circuit_backend_returns_none_when_redis_is_down():
    backend = RedisCircuitState(conn=_BrokenConnection())
    assert backend.is_open("default") is None
    assert backend.record_failure("default", threshold=3, cooldown=30) is False
    assert backend.record_success("default") is False


def test_circuit_breaker_uses_remote_state_when_available():
    from src.llm.provider import CircuitBreaker

    class RemoteOpen:
        def is_open(self, alias):
            return True

        def record_failure(self, alias, threshold, cooldown):
            return True

        def record_success(self, alias):
            return True

    breaker = CircuitBreaker(remote=RemoteOpen())
    assert breaker.is_open("default") is True


def test_circuit_breaker_falls_back_to_local_when_remote_unknown():
    from src.llm.provider import CircuitBreaker

    class RemoteDown:
        def is_open(self, alias):
            return None

        def record_failure(self, alias, threshold, cooldown):
            return False

        def record_success(self, alias):
            return False

    breaker = CircuitBreaker(failure_threshold=2, cooldown_seconds=30,
                             remote=RemoteDown())
    breaker.record_failure("default")
    breaker.record_failure("default")
    assert breaker.is_open("default") is True


def test_middleware_keeps_local_window_when_shared_limiter_absent():
    middleware = RateLimitMiddleware(app=None, requests_per_minute=2,
                                     shared_limiter=None)
    assert middleware._shared is None
    assert middleware._allow("1.1.1.1") == (True, 0)
    assert middleware._allow("1.1.1.1") == (True, 0)
    allowed, retry_after = middleware._allow("1.1.1.1")
    assert allowed is False
    assert retry_after >= 1


def test_middleware_prefers_shared_limiter_result():
    class AlwaysDeny:
        def allow(self, client, limit, window):
            return False, 42

    middleware = RateLimitMiddleware(app=None, requests_per_minute=60,
                                     shared_limiter=AlwaysDeny())
    assert middleware._allow("1.1.1.1") == (False, 42)


@pytest.mark.integration
def test_redis_limit_is_shared_across_instances():
    """两个实例共用同一份配额——这正是迁 Redis 要解决的问题。

    只在显式开启 integration 且本机 Redis 可用时运行。
    """
    import redis

    from src.config import REDIS_HOST, REDIS_PORT

    try:
        conn = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True,
                           socket_connect_timeout=0.5, socket_timeout=0.5)
        conn.ping()
    except Exception:  # noqa: BLE001
        pytest.skip("Redis 不可用")

    prefix = f"ratelimit:test-{time.time_ns()}"
    # 两个实例操作同一组键，等价于两个副本共享同一份配额。
    first = RedisRateLimiter(conn=conn, prefix=prefix)
    second = RedisRateLimiter(conn=conn, prefix=prefix)
    try:
        results = [
            first.allow("client", 3, 60) if index % 2 == 0
            else second.allow("client", 3, 60)
            for index in range(6)
        ]
        assert sum(1 for ok, _ in results if ok) == 3
    finally:
        conn.delete(f"{prefix}:client")
