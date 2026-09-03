"""Versioned, profile-isolated semantic cache with best-effort failures."""

import hashlib
import json
import logging
import time

import numpy as np

from src.config import CACHE_SCAN_BACKEND, CACHE_SIMILARITY_THRESHOLD, EMBEDDING_DIM
from src.rag.embedding import EmbeddingService
from src.storage.redis_client import RedisClient

logger = logging.getLogger(__name__)

_CACHE_PREFIX = "cache:fitness:v2:"
_SCHEMA_VERSION = 2
_MAX_SCAN = 200
_SCAN_WARN_MS = 100
_FINGERPRINT_LENGTH = 16
_QUERY_HASH_LENGTH = 64


class CacheAnnIndex:
    """pgvector ANN index over cache keys — the scalable alternative to the
    Redis keyspace linear scan.

    The cache entries themselves stay in Redis (with TTL); this table only
    stores (cache_key, profile_fingerprint, embedding) so similarity search
    becomes a HNSW query instead of an O(N) client-side loop. Rows whose
    Redis entry has expired are pruned opportunistically on write and simply
    resolve to a miss on read.
    """

    TABLE = "semantic_cache_index"

    def __init__(self):
        from src.storage.pg import PGClient
        self.pg = PGClient()
        self.dim = EMBEDDING_DIM
        self.pg.execute(
            f"CREATE TABLE IF NOT EXISTS {self.TABLE} ("
            "cache_key TEXT PRIMARY KEY, "
            "profile_fingerprint TEXT NOT NULL, "
            f"embedding vector({self.dim}), "
            "created_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
        self.pg.execute(
            f"CREATE INDEX IF NOT EXISTS {self.TABLE}_ann "
            f"ON {self.TABLE} USING hnsw (embedding vector_cosine_ops)"
        )

    def upsert(self, cache_key: str, fingerprint: str, embedding: list) -> None:
        vec_str = f"[{','.join(str(v) for v in embedding)}]"
        self.pg.execute(
            f"INSERT INTO {self.TABLE} (cache_key, profile_fingerprint, embedding) "
            f"VALUES (:key, :fp, '{vec_str}'::vector) "
            "ON CONFLICT (cache_key) DO UPDATE SET "
            f"embedding = EXCLUDED.embedding, created_at = now()",
            {"key": cache_key, "fp": fingerprint},
        )
        # Opportunistic prune: Redis entries expire after 1h; drop index rows
        # older than 2h so the ANN index does not grow without bound.
        self.pg.execute(
            f"DELETE FROM {self.TABLE} WHERE created_at < now() - interval '2 hours'"
        )

    def candidates(self, fingerprint: str, embedding: list, limit: int = 5) -> list:
        """Cache keys whose embeddings are nearest to the query, within one
        profile namespace. Returns [] when the namespace is empty."""
        vec_str = f"[{','.join(str(v) for v in embedding)}]"
        rows = self.pg.fetch_all(
            f"SELECT cache_key FROM {self.TABLE} "
            f"WHERE profile_fingerprint = :fp AND embedding IS NOT NULL "
            f"ORDER BY embedding <=> '{vec_str}'::vector LIMIT :limit",
            {"fp": fingerprint, "limit": limit},
        )
        return [r[0] for r in rows]


class SemanticCache:
    """Two-level semantic cache: exact lookup followed by vector similarity.

    The similarity scan has two backends (``CACHE_SCAN_BACKEND``):
    * ``linear`` (default) — scan up to 200 Redis keys of the profile
      namespace and compute similarities client-side;
    * ``ann`` — pgvector HNSW index (``CacheAnnIndex``), which scales past
      the linear-scan ceiling. Any ANN failure degrades to the linear scan,
      keeping cache best-effort semantics.
    """

    def __init__(self, scan_backend: str | None = None):
        self.redis = RedisClient()
        self.emb = EmbeddingService()
        self._hits_exact = 0
        self._hits_semantic = 0
        self._misses = 0
        self._sets = 0
        self._ann = None
        backend = (scan_backend or CACHE_SCAN_BACKEND or "linear").lower()
        if backend == "ann":
            try:
                self._ann = CacheAnnIndex()
            except Exception as exc:
                logger.warning(
                    "Semantic cache ANN index unavailable (%s), "
                    "falling back to linear scan", type(exc).__name__)

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

    def _ann_lookup(self, profile: dict, fingerprint: str, query_vec) -> tuple:
        """ANN-backed similarity lookup. Returns (best_result, best_sim).

        The HNSW query is exhaustive over the profile namespace, so a miss
        here is authoritative — callers do not need a linear rescan unless
        this method raises.
        """
        cache_keys = self._ann.candidates(fingerprint, query_vec.tolist(), limit=5)
        best_sim = 0.0
        best_result = None
        for cache_key in cache_keys:
            entry = self._decode_entry(self.redis.get(cache_key))
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
            if sim > best_sim:
                best_sim = sim
                best_result = entry["result"]
        return best_result, best_sim

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

        # ANN backend: HNSW over the whole profile namespace replaces the scan.
        if self._ann is not None:
            try:
                best_result, best_sim = self._ann_lookup(profile, fingerprint, query_vec)
                elapsed_ms = (time.monotonic() - started) * 1000
                if elapsed_ms > _SCAN_WARN_MS:
                    logger.warning("Semantic cache ANN lookup took %.1fms", elapsed_ms)
                if best_result is not None and best_sim >= CACHE_SIMILARITY_THRESHOLD:
                    return best_result
                return None
            except Exception as exc:
                logger.warning(
                    "Semantic cache ANN lookup failed (%s), falling back to scan",
                    type(exc).__name__)

        best_sim = 0.0
        best_result = None
        scanned = 0
        namespace = f"{_CACHE_PREFIX}{fingerprint}:*"
        try:
            keys = self.redis.conn.scan_iter(match=namespace, count=50)
            for key_bytes in keys:
                if scanned >= _MAX_SCAN:
                    break
                scanned += 1
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
            if self._ann is not None:
                try:
                    self._ann.upsert(self._make_key(profile, query),
                                     fingerprint, query_vec.tolist())
                except Exception as exc:
                    logger.warning("Semantic cache ANN index write failed: %s",
                                   type(exc).__name__)
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
