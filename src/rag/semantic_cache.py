"""Versioned, profile-isolated semantic cache with best-effort failures."""

import hashlib
import json
import logging
import time

import numpy as np

from src.config import CACHE_SIMILARITY_THRESHOLD
from src.rag.embedding import EmbeddingService
from src.storage.redis_client import RedisClient

logger = logging.getLogger(__name__)

_CACHE_PREFIX = "cache:fitness:v2:"
_SCHEMA_VERSION = 2
_MAX_SCAN = 200
_SCAN_WARN_MS = 100
_FINGERPRINT_LENGTH = 16
_QUERY_HASH_LENGTH = 64


class SemanticCache:
    """Two-level semantic cache: exact lookup followed by vector similarity."""

    def __init__(self):
        self.redis = RedisClient()
        self.emb = EmbeddingService()
        self._hits_exact = 0
        self._hits_semantic = 0
        self._misses = 0
        self._sets = 0

    def _profile_fingerprint(self, profile: dict) -> str:
        """Return a stable, short fingerprint for the complete user profile."""
        canonical = json.dumps(
            profile, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:_FINGERPRINT_LENGTH]

    def _make_key(self, profile: dict, query: str) -> str:
        fingerprint = self._profile_fingerprint(profile)
        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:_QUERY_HASH_LENGTH]
        return f"{_CACHE_PREFIX}{fingerprint}:{query_hash}"

    @staticmethod
    def _decode_entry(data) -> dict | None:
        if not data:
            return None
        try:
            entry = json.loads(data)
        except (TypeError, ValueError, UnicodeDecodeError):
            return None
        if not isinstance(entry, dict):
            return None
        if entry.get("schema_version") != _SCHEMA_VERSION:
            return None
        if not isinstance(entry.get("profile_fingerprint"), str):
            return None
        result = entry.get("result")
        if not isinstance(result, dict) or not result:
            return None
        return entry

    def _exact_get(self, profile: dict, query: str) -> dict | None:
        try:
            data = self.redis.get(self._make_key(profile, query))
        except Exception as exc:
            logger.warning("Semantic cache exact get failed: %s", type(exc).__name__)
            return None
        entry = self._decode_entry(data)
        if entry is None or entry["profile_fingerprint"] != self._profile_fingerprint(profile):
            return None
        if entry.get("query") != query:
            return None
        return entry["result"]

    def _semantic_get(self, profile: dict, query: str) -> dict | None:
        started = time.monotonic()
        fingerprint = self._profile_fingerprint(profile)
        try:
            query_vec = np.asarray(self.emb.embed(query), dtype=float)
            if query_vec.ndim != 1 or not np.all(np.isfinite(query_vec)):
                return None
        except Exception as exc:
            logger.warning("Semantic cache embedding failed: %s", type(exc).__name__)
            return None

        best_sim = 0.0
        best_result = None
        scanned = 0
        namespace = f"{_CACHE_PREFIX}{fingerprint}:*"
        try:
            keys = self.redis.conn.scan_iter(match=namespace, count=50)
            for key_bytes in keys:
                if scanned >= _MAX_SCAN:
                    break
                key = key_bytes.decode("utf-8") if isinstance(key_bytes, bytes) else key_bytes
                try:
                    data = self.redis.get(key)
                except Exception as exc:
                    logger.warning("Semantic cache scan get failed: %s", type(exc).__name__)
                    continue
                entry = self._decode_entry(data)
                if entry is None or entry.get("profile_fingerprint") != fingerprint:
                    continue
                embedding = entry.get("_embedding", entry.get("embedding"))
                try:
                    candidate = np.asarray(embedding, dtype=float)
                    if candidate.ndim != 1 or candidate.shape != query_vec.shape:
                        continue
                    if not np.all(np.isfinite(candidate)):
                        continue
                    denominator = np.linalg.norm(query_vec) * np.linalg.norm(candidate)
                    if denominator == 0:
                        continue
                    sim = float(np.dot(query_vec, candidate) / denominator)
                except (TypeError, ValueError, FloatingPointError):
                    continue
                scanned += 1
                if sim > best_sim:
                    best_sim = sim
                    best_result = entry["result"]
        except Exception as exc:
            logger.warning("Semantic cache scan failed: %s", type(exc).__name__)
            return None
        elapsed_ms = (time.monotonic() - started) * 1000
        if elapsed_ms > _SCAN_WARN_MS:
            logger.warning("Semantic cache scan took %.1fms", elapsed_ms)
        if best_result is not None and best_sim >= CACHE_SIMILARITY_THRESHOLD:
            return best_result
        return None

    def get(self, profile: dict, query: str) -> dict | None:
        try:
            result = self._exact_get(profile, query)
            if result is not None:
                self._hits_exact += 1
                return result
            result = self._semantic_get(profile, query)
        except Exception as exc:
            logger.warning("Semantic cache lookup failed: %s", type(exc).__name__)
            result = None
        if result is not None:
            self._hits_semantic += 1
        else:
            self._misses += 1
        return result

    def set(self, profile: dict, query: str, result: dict):
        """Best-effort write; cache failures never affect the request."""
        if not isinstance(result, dict) or not result:
            return
        try:
            fingerprint = self._profile_fingerprint(profile)
            query_vec = np.asarray(self.emb.embed(query), dtype=float)
            if query_vec.ndim != 1 or not np.all(np.isfinite(query_vec)):
                return
            entry = {
                "schema_version": _SCHEMA_VERSION,
                "profile_fingerprint": fingerprint,
                "query": query,
                "embedding": query_vec.tolist(),
                "result": result,
            }
            self.redis.set(
                self._make_key(profile, query),
                json.dumps(entry, ensure_ascii=False, separators=(",", ":")),
                ex=3600,
            )
            self._sets += 1
        except Exception as exc:
            logger.warning("Semantic cache set failed: %s", type(exc).__name__)

    def get_stats(self) -> dict:
        total = self._hits_exact + self._hits_semantic + self._misses
        hits = self._hits_exact + self._hits_semantic
        return {"cache": {
            "hits_exact": self._hits_exact,
            "hits_semantic": self._hits_semantic,
            "hits_total": hits,
            "misses": self._misses,
            "sets": self._sets,
            "total_lookups": total,
            "hit_rate_exact": round(self._hits_exact / total, 4) if total else 0,
            "hit_rate_semantic": round(self._hits_semantic / total, 4) if total else 0,
            "hit_rate_total": round(hits / total, 4) if total else 0,
        }}
