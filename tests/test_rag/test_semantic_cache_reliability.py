import json

import pytest

from src.rag.semantic_cache import SemanticCache, _MAX_SCAN


class FakeRedisConnection:
    def __init__(self):
        self.entries = {}

    def scan_iter(self, match=None, count=50):
        prefix = match[:-1] if match and match.endswith("*") else match
        for key in list(self.entries):
            if prefix is None or key.startswith(prefix):
                yield key


class FakeRedis:
    def __init__(self):
        self.conn = FakeRedisConnection()

    def get(self, key):
        return self.conn.entries.get(key)

    def set(self, key, value, ex=None):
        self.conn.entries[key] = value


class FakeEmbedding:
    def embed(self, query):
        return [1.0, 0.0] if query else [0.0, 1.0]


@pytest.fixture
def cache():
    instance = SemanticCache.__new__(SemanticCache)
    instance.redis = FakeRedis()
    instance.emb = FakeEmbedding()
    instance._hits_exact = instance._hits_semantic = 0
    instance._misses = instance._sets = 0
    instance._ann = None  # linear-scan backend for these reliability tests
    return instance


def make_entry(cache, profile, query, result, *, version=2, fingerprint=None, embedding=None):
    return json.dumps({
        "schema_version": version,
        "profile_fingerprint": fingerprint or cache._profile_fingerprint(profile),
        "query": query,
        "_embedding": embedding or [1.0, 0.0],
        "result": result,
    })


def test_key_contains_version_and_profile_fingerprint(cache):
    key = cache._make_key({"goal": "增肌"}, "q")
    assert key.startswith("cache:fitness:v2:")
    assert key.count(":") == 4
    _, fingerprint, query_hash = key.rsplit(":", 2)
    assert len(fingerprint) == 16
    assert len(query_hash) == 64
    assert key != cache._make_key({"goal": "减脂"}, "q")


def test_set_entry_contains_version_fingerprint_query_embedding_and_result(cache):
    profile = {"goal": "增肌"}
    cache.set(profile, "q", {"answer": "safe"})
    entry = json.loads(cache.redis.get(cache._make_key(profile, "q")))
    assert entry["schema_version"] == 2
    assert entry["profile_fingerprint"] == cache._profile_fingerprint(profile)
    assert entry["query"] == "q"
    assert entry["embedding"] if "embedding" in entry else entry["_embedding"]
    assert entry["result"] == {"answer": "safe"}


def test_semantic_lookup_never_crosses_profile(cache):
    other = {"goal": "减脂"}
    wrong_key = cache._make_key(other, "q")
    cache.redis.conn.entries[wrong_key] = make_entry(cache, other, "q", {"answer": "wrong"})
    assert cache.get({"goal": "增肌"}, "similar") is None


def test_old_version_bad_json_and_invalid_result_are_ignored(cache):
    profile = {"goal": "增肌"}
    key = cache._make_key(profile, "q")
    cache.redis.conn.entries[key] = make_entry(cache, profile, "q", {"answer": "old"}, version=1)
    assert cache.get(profile, "q") is None
    cache.redis.conn.entries[key] = "{bad json"
    assert cache.get(profile, "q") is None
    cache.redis.conn.entries[key] = make_entry(cache, profile, "q", {})
    assert cache.get(profile, "q") is None


def test_redis_get_failure_is_cache_miss(cache, monkeypatch):
    monkeypatch.setattr(cache.redis, "get", lambda key: (_ for _ in ()).throw(ConnectionError()))
    assert cache.get({}, "q") is None


def test_redis_set_failure_does_not_break_request(cache, monkeypatch):
    monkeypatch.setattr(cache.redis, "set", lambda *args, **kwargs: (_ for _ in ()).throw(ConnectionError()))
    cache.set({}, "q", {"answer": "safe"})


def test_embedding_failure_is_cache_miss_and_set_noop(cache, monkeypatch):
    monkeypatch.setattr(cache.emb, "embed", lambda query: (_ for _ in ()).throw(RuntimeError()))
    assert cache.get({}, "q") is None
    cache.set({}, "q", {"answer": "safe"})


def test_semantic_hit_is_limited_to_current_namespace(cache):
    profile = {"goal": "增肌"}
    key = cache._make_key(profile, "stored")
    cache.redis.conn.entries[key] = make_entry(cache, profile, "stored", {"answer": "right"})
    assert cache.get(profile, "different") == {"answer": "right"}


def test_malformed_keys_consume_semantic_scan_budget(cache):
    profile = {"goal": "增肌"}
    prefix = cache._make_key(profile, "query").rsplit(":", 1)[0]
    for index in range(_MAX_SCAN + 25):
        cache.redis.conn.entries[f"{prefix}:{index:064x}"] = "{bad json"

    calls = 0
    original_get = cache.redis.get

    def counting_get(key):
        nonlocal calls
        calls += 1
        return original_get(key)

    cache.redis.get = counting_get
    assert cache._semantic_get(profile, "different") is None
    assert calls == _MAX_SCAN


def test_failing_keys_consume_semantic_scan_budget(cache):
    profile = {"goal": "增肌"}
    prefix = cache._make_key(profile, "query").rsplit(":", 1)[0]
    for index in range(_MAX_SCAN + 25):
        cache.redis.conn.entries[f"{prefix}:{index:064x}"] = "unused"

    calls = 0

    def failing_get(key):
        nonlocal calls
        calls += 1
        raise ConnectionError("offline")

    cache.redis.get = failing_get
    assert cache._semantic_get(profile, "different") is None
    assert calls == _MAX_SCAN
