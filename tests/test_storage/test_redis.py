import pytest
import json
import numpy as np
from src.storage.redis_client import RedisClient

@pytest.fixture
def redis():
    client = RedisClient()
    yield client
    client.flushdb()
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
