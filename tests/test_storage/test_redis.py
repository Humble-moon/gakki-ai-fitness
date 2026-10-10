import pytest
import json
import numpy as np
from src.storage.redis_client import RedisClient

pytestmark = pytest.mark.integration

@pytest.fixture
def redis():
    """只清理本测试自己写过的键。

    这里曾经调用 client.flushdb()——那会清空整个 Redis，把当天的 LLM
    成本计数、语义缓存和登录会话一起抹掉（开发机上跑一次 integration
    就等于重置一次生产数据）。RedisClient.flushdb 后来加了 confirmed=True
    的护栏，正好把这个隐患暴露出来。改成定向删除后，测试不再有副作用。
    """
    client = RedisClient()
    yield client
    for key in ("test_key", "cache:plan:1", "temp", "vec:test"):
        try:
            client.delete(key)
        except Exception:  # noqa: BLE001 — 清理失败不该让测试本身报错
            pass
    client.close()

class TestRedisClient:
    def test_set_and_get(self, redis):
        redis.set("test_key", "hello")
        assert redis.get("test_key") == "hello"

    def test_cache_json(self, redis):
        data = {"plan": "增肌计划", "exercises": ["卧推", "深蹲"]}
        redis.set("cache:plan:1", json.dumps(data))
        result = json.loads(redis.get("cache:plan:1"))
        assert result["plan"] == "增肌计划"
        assert len(result["exercises"]) == 2

    def test_delete(self, redis):
        redis.set("temp", "val")
        redis.delete("temp")
        assert redis.get("temp") is None

    def test_vector_bytes(self, redis):
        vec = np.random.rand(512).astype(np.float32).tobytes()
        redis.set_bytes("vec:test", vec)
        stored = redis.get_bytes("vec:test")
        assert stored is not None
