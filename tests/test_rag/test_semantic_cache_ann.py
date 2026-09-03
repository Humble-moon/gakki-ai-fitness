"""Semantic cache ANN backend: hit/miss semantics and linear-scan fallback."""

import json

from src.rag.semantic_cache import SemanticCache


class FakeRedisConnection:
    def __init__(self):
        self.entries = {}
        self.scan_calls = 0

    def scan_iter(self, match=None, count=50):
        self.scan_calls += 1
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
        return [1.0, 0.0]


class FakeAnn:
    def __init__(self, keys=None, fail=False):
        self.keys = keys or []
        self.fail = fail
        self.upserts = []

    def candidates(self, fingerprint, embedding, limit=5):
        if self.fail:
            raise RuntimeError("ann boom")
        return self.keys[:limit]

    def upsert(self, cache_key, fingerprint, embedding):
        self.upserts.append(cache_key)


def _build(ann):
    instance = SemanticCache.__new__(SemanticCache)
    instance.redis = FakeRedis()
    instance.emb = FakeEmbedding()
    instance._hits_exact = instance._hits_semantic = 0
    instance._misses = instance._sets = 0
    instance._ann = ann
    return instance


def _put_entry(cache, profile, query, result, embedding):
    key = cache._make_key(profile, query)
    cache.redis.set(key, json.dumps({
        "schema_version": 2,
        "profile_fingerprint": cache._profile_fingerprint(profile),
        "query": query,
        "embedding": embedding,
        "result": result,
    }))
    return key


PROFILE = {"height": 180}


def test_ann_hit_returns_cached_result_without_scan():
    cache = _build(FakeAnn())
    key = _put_entry(cache, PROFILE, "练胸", {"plan": "ok"}, [1.0, 0.0])
    cache._ann.keys = [key]

    assert cache.get(PROFILE, "练胸计划") == {"plan": "ok"}
    assert cache.redis.conn.scan_calls == 0  # ANN replaced the scan entirely


def test_ann_authoritative_miss_skips_linear_scan():
    cache = _build(FakeAnn(keys=[]))

    assert cache.get(PROFILE, "练胸") is None
    assert cache.redis.conn.scan_calls == 0


def test_ann_failure_falls_back_to_linear_scan():
    cache = _build(FakeAnn(fail=True))
    key = _put_entry(cache, PROFILE, "练胸", {"plan": "ok"}, [1.0, 0.0])

    # fallback scan finds the entry through the Redis namespace
    assert cache.get(PROFILE, "练胸计划") == {"plan": "ok"}
    assert cache.redis.conn.scan_calls == 1
    assert key  # silence lint: key used via redis store


def test_ann_expired_index_row_is_skipped():
    cache = _build(FakeAnn())
    cache._ann.keys = ["cache:fitness:v2:gone"]  # Redis entry expired

    assert cache.get(PROFILE, "练胸") is None


def test_set_writes_through_to_ann_index():
    cache = _build(FakeAnn())
    cache.set(PROFILE, "练背", {"plan": "back"})

    assert cache._ann.upserts == [cache._make_key(PROFILE, "练背")]
