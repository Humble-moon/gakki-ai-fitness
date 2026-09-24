"""当日 LLM 成本上限：记账、拦截、降级。

补这组用例的背景：项目原本只有 `cost_tracker`——它做的是**观测**（内存累加
+ 打日志），没有任何上限。本地 demo 无所谓，对外开放后一次脚本刷量或用户
反复点"重新生成"，成本就会无界增长，而这笔钱是真要付的。

三个必须守住的点：
  1. 计数跨进程存活（存 Redis）——否则重启即重置预算，限制形同虚设
  2. 超限时拒绝会花钱的端点，但**不误伤**只写本地存储的端点
  3. Redis 不可用时降级放行——这是成本保护不是安全边界，
     为此让全站不可用的代价更大（限流仍在内层兜底）
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.security.api_guard import CostLimitMiddleware
from src.security.cost_guard import CostGuard, _today_key


class _FakeRedis:
    """最小 get/set 实现，含过期参数（守卫会传 ex=）。"""

    def __init__(self):
        self.store: dict[str, str] = {}
        self.raise_on_call = False

    def get(self, key):
        if self.raise_on_call:
            raise ConnectionError("redis down")
        return self.store.get(key)

    def set(self, key, value, ex=None):
        if self.raise_on_call:
            raise ConnectionError("redis down")
        self.store[key] = value


class TestAccounting:
    def test_accumulates_within_the_day(self):
        redis = _FakeRedis()
        guard = CostGuard(redis_client=redis, daily_limit=10.0)
        guard.record(0.32)
        guard.record(0.18)
        assert abs(guard.spent_today() - 0.5) < 1e-9

    def test_survives_process_restart(self):
        """新实例读同一个 Redis，当日累计必须延续——这是存 Redis 的理由。"""
        redis = _FakeRedis()
        CostGuard(redis_client=redis, daily_limit=10.0).record(3.5)
        assert abs(CostGuard(redis_client=redis, daily_limit=10.0).spent_today() - 3.5) < 1e-9

    def test_key_is_per_calendar_day(self):
        from datetime import datetime, timezone

        # 东八区 23:59 与次日 00:01 必须落在不同的键上
        late = datetime(2026, 9, 24, 23, 59, tzinfo=timezone.utc)
        assert _today_key(late).endswith("2026-09-25")  # UTC 23:59 = 东八区次日 07:59
        assert _today_key(late) != _today_key(datetime(2026, 9, 24, 1, 0, tzinfo=timezone.utc))

    def test_ignores_non_positive_cost(self):
        redis = _FakeRedis()
        guard = CostGuard(redis_client=redis, daily_limit=10.0)
        guard.record(0)
        guard.record(-1)
        assert guard.spent_today() == 0.0


class TestLimitCheck:
    def test_allows_below_limit(self):
        guard = CostGuard(redis_client=_FakeRedis(), daily_limit=10.0)
        guard.record(9.99)
        allowed, spent = guard.check()
        assert allowed is True and abs(spent - 9.99) < 1e-9

    def test_blocks_at_or_above_limit(self):
        guard = CostGuard(redis_client=_FakeRedis(), daily_limit=10.0)
        guard.record(10.0)
        allowed, _ = guard.check()
        assert allowed is False

    def test_zero_limit_means_unlimited(self):
        guard = CostGuard(redis_client=_FakeRedis(), daily_limit=0)
        guard.record(999)
        allowed, _ = guard.check()
        assert allowed is True

    def test_no_redis_means_unlimited(self):
        guard = CostGuard(redis_client=None, daily_limit=1.0)
        allowed, _ = guard.check()
        assert allowed is True


class TestDegradation:
    """Redis 故障时降级放行——但要留痕，否则"保护失效"会无人知晓。"""

    def test_read_failure_does_not_raise(self):
        redis = _FakeRedis()
        redis.raise_on_call = True
        guard = CostGuard(redis_client=redis, daily_limit=1.0)
        assert guard.spent_today() == 0.0
        allowed, _ = guard.check()
        assert allowed is True

    def test_write_failure_does_not_raise(self):
        redis = _FakeRedis()
        redis.raise_on_call = True
        guard = CostGuard(redis_client=redis, daily_limit=1.0)
        guard.record(5.0)  # 不得抛异常


class _Recorder:
    def __init__(self):
        self.costs: list[float] = []

    def __call__(self, cost):
        self.costs.append(cost)


class TestCostTrackerSink:
    """cost_tracker 通过 sink 广播成本，LLMProvider 无需知道预算的存在。"""

    def test_sink_receives_each_cost(self):
        from src.llm.cost_tracker import CostTracker

        tracker = CostTracker()
        rec = _Recorder()
        tracker.add_sink(rec)
        tracker.record("deepseek-chat", tokens=1000)
        tracker.record("deepseek-chat", tokens=2000)
        assert len(rec.costs) == 2
        assert all(c > 0 for c in rec.costs)

    def test_sink_failure_does_not_break_recording(self):
        """记账失败最多让预算不准；为此中断一次正常调用是更糟的取舍。"""
        from src.llm.cost_tracker import CostTracker

        def boom(cost):
            raise RuntimeError("sink exploded")

        tracker = CostTracker()
        tracker.add_sink(boom)
        tracker.record("deepseek-chat", tokens=1000)  # 不得抛异常

    def test_guard_as_sink_end_to_end(self):
        from src.llm.cost_tracker import CostTracker

        guard = CostGuard(redis_client=_FakeRedis(), daily_limit=1.0)
        tracker = CostTracker()
        tracker.add_sink(guard.record)
        tracker.record("deepseek-reasoner", tokens=50_000)
        assert guard.spent_today() > 0


def _build_app(guard):
    app = FastAPI()

    @app.get("/health/live")
    def live():
        return {"status": "ok"}

    @app.post("/api/generate-plan")
    def plan():
        return {"ok": True}

    @app.post("/api/training-logs")
    def logs():
        return {"ok": True}

    @app.get("/api/training-stats")
    def stats():
        return {"ok": True}

    app.add_middleware(CostLimitMiddleware, guard=guard)
    return app


class TestMiddleware:
    def test_allows_when_under_limit(self):
        guard = CostGuard(redis_client=_FakeRedis(), daily_limit=10.0)
        client = TestClient(_build_app(guard))
        assert client.post("/api/generate-plan").status_code == 200

    def test_blocks_generating_endpoint_when_exhausted(self):
        guard = CostGuard(redis_client=_FakeRedis(), daily_limit=1.0)
        guard.record(1.5)
        client = TestClient(_build_app(guard))
        resp = client.post("/api/generate-plan")
        assert resp.status_code == 429
        body = resp.json()
        assert body["error"] == "daily_cost_limit_reached"
        assert "spent_today" in body and "daily_limit" in body

    def test_does_not_block_free_endpoints(self):
        """记训练日志不调用 LLM，成本超限不该把它一起挡掉。"""
        guard = CostGuard(redis_client=_FakeRedis(), daily_limit=1.0)
        guard.record(99.0)
        client = TestClient(_build_app(guard))
        assert client.post("/api/training-logs").status_code == 200

    def test_does_not_block_reads(self):
        """GET 是查询，不产生开销。"""
        guard = CostGuard(redis_client=_FakeRedis(), daily_limit=1.0)
        guard.record(99.0)
        client = TestClient(_build_app(guard))
        assert client.get("/api/training-stats").status_code == 200

    def test_health_probe_unaffected(self):
        guard = CostGuard(redis_client=_FakeRedis(), daily_limit=1.0)
        guard.record(99.0)
        client = TestClient(_build_app(guard))
        assert client.get("/health/live").status_code == 200

    def test_redis_outage_fails_open(self):
        """降级放行：成本保护不该让全站不可用。"""
        redis = _FakeRedis()
        redis.raise_on_call = True
        guard = CostGuard(redis_client=redis, daily_limit=0.0001)
        client = TestClient(_build_app(guard))
        assert client.post("/api/generate-plan").status_code == 200
