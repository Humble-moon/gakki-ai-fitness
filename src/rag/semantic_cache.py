import json
import hashlib
from src.rag.embedding import EmbeddingService
from src.storage.redis_client import RedisClient
from src.config import CACHE_SIMILARITY_THRESHOLD


class SemanticCache:
    def __init__(self):
        self.redis = RedisClient()
        self.emb = EmbeddingService()

    def _make_key(self, profile: dict, query: str) -> str:
        raw = json.dumps(profile, sort_keys=True) + query
        return f"cache:fitness:{hashlib.md5(raw.encode()).hexdigest()}"

    def get(self, profile: dict, query: str) -> dict | None:
        cache_key = self._make_key(profile, query)
        data = self.redis.get(cache_key)
        if data:
            entry = json.loads(data)
            # Unwrap the stored entry (same structure as set() stores)
            if isinstance(entry, dict) and "result" in entry:
                return entry["result"]
            return entry
        # Try similar cache
        query_vec = self.emb.embed(query)
        keys = self.redis.conn.keys("cache:fitness:*")
        for k in keys:
            cached = self.redis.get(k.decode())
            if cached:
                entry = json.loads(cached)
                stored_vec = entry.get("_embedding")
                if stored_vec and self.emb.similarity(query_vec, stored_vec) >= CACHE_SIMILARITY_THRESHOLD:
                    return entry.get("result")
        return None

    def set(self, profile: dict, query: str, result: dict):
        cache_key = self._make_key(profile, query)
        query_vec = self.emb.embed(query)
        entry = {"_embedding": query_vec, "result": result}
        self.redis.set(cache_key, json.dumps(entry, ensure_ascii=False), ex=3600)
