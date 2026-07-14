"""
=============================================================================
semantic_cache.py — 语义缓存层（第 5 层）
=============================================================================
【项目角色】
    这是 RAG 五层检索体系中的第 5 层 — 语义缓存加速层。
    将用户的 (用户画像 + 查询) → LLM 生成的回答缓存到 Redis 中，
    对于相同或相似的查询直接返回缓存结果，避免重复调用 LLM，大幅降低延迟和成本。

【五层检索关系】
    第 0 层:  EmbeddingService（embedding.py）—— 文本 → 向量
    第 1 层:  VectorSearch   —— 向量检索
    第 1 层:  KeywordSearch  —— 关键词检索
    第 2 层:  KnowledgeSearch —— RRF 融合 + 重排序
    第 3 层:  AgenticRAG     —— 迭代评估 + 查询改写
    第 4 层:  GraphRAG       —— 知识图谱推理
    第 5 层:  SemanticCache（本文件）—— 语义缓存加速                   ← 你在这里

【被谁调用】应用层 API 路由（在调用 AgenticRAG 之前先检查缓存）
【调用谁】  EmbeddingService.embed()（向量相似度匹配）、
           RedisClient（缓存读写 + 键扫描）

【缓存策略 — 两级命中】
    第一级（精确匹配）：基于 (用户画像 JSON + 查询文本) 的 MD5 哈希
      → 相同输入直接命中，延迟 < 1ms
    第二级（语义匹配）：精确匹配未命中时，用查询向量在 Redis 中做余弦相似度检索
      → "怎么练胸肌" 和 "胸部训练方法" 相似度 >= 0.92 → 命中
      → 延迟 ~50ms（一次 embedding + Redis 扫描 + N 次余弦计算）

    - 过期时间：3600 秒（1 小时），健身知识时效性低
    - 存储内容：_embedding（查询向量，用于语义匹配）+ result（LLM 回答）
    - 命名空间：cache:fitness:{hash} 前缀隔离

【为什么需要语义匹配】
    LLM 调用是系统中最昂贵的操作（延迟 2~10 秒 + API 费用）。
    用户在短时间内可能反复问语义相似的问题：
    - "我想练胸肌有什么动作？"（第 1 次 → 调用 LLM）
    - "胸部训练方法"（第 2 次 → 语义匹配命中，< 50ms！）
    相比精确匹配，语义匹配把同一意图的不同问法也覆盖了，命中率大幅提升。
=============================================================================
"""

import json
import hashlib
import logging

import numpy as np

from src.rag.embedding import EmbeddingService
from src.storage.redis_client import RedisClient
from src.config import CACHE_SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

# Redis 缓存键前缀，用于命名空间隔离和扫描
_CACHE_PREFIX = "cache:fitness:"
# 语义匹配时最多扫描的条目数，防止 Redis 扫描耗时过长
_MAX_SCAN = 200


class SemanticCache:
    """语义缓存 — 两级命中：精确匹配 + 向量相似度兜底。

    【使用流程】
        1. 用户提问 → SemanticCache.get(profile, query)
        2. 精确匹配命中 → 直接返回（< 1ms）
        3. 精确匹配未命中 → 语义匹配：embed(query) + Redis 扫描 + 余弦相似度
           → 找到相似度 >= 0.92 的历史查询 → 返回缓存结果（~50ms）
        4. 两级都未命中 → 走 RAG 流程 → SemanticCache.set() 写入缓存
           → 下次相同或相似查询都能命中

    【为什么缓存键包含 profile（用户画像）】
        同一个查询"推荐训练动作"，不同用户画像会得到完全不同的回答：
        - 新手 + 减脂目标 → 推荐低强度全身动作
        - 老手 + 增肌目标 → 推荐大重量分化训练
        精确匹配用 profile+query 的联合哈希保证按画像隔离。
    """

    def __init__(self):
        self.redis = RedisClient()
        self.emb = EmbeddingService()

    # ------------------------------------------------------------------
    # 缓存键
    # ------------------------------------------------------------------

    def _make_key(self, profile: dict, query: str) -> str:
        raw = json.dumps(profile, sort_keys=True) + query
        return f"{_CACHE_PREFIX}{hashlib.md5(raw.encode()).hexdigest()}"

    # ------------------------------------------------------------------
    # 第一级：精确匹配
    # ------------------------------------------------------------------

    def _exact_get(self, profile: dict, query: str) -> dict | None:
        """MD5 精确匹配，延迟 < 1ms。"""
        data = self.redis.get(self._make_key(profile, query))
        if not data:
            return None
        entry = json.loads(data)
        if isinstance(entry, dict) and "result" in entry:
            return entry["result"]
        return entry  # 旧格式兼容

    # ------------------------------------------------------------------
    # 第二级：语义相似度匹配
    # ------------------------------------------------------------------

    def _semantic_get(self, query: str) -> dict | None:
        """精确匹配未命中时，在 Redis 缓存中搜索向量相似的历史查询。

        流程：
          1. embed 当前 query
          2. scan_iter 遍历 Redis 中 cache:fitness:* 键
          3. 逐个加载 _embedding，计算余弦相似度
          4. 找到相似度 >= CACHE_SIMILARITY_THRESHOLD 的条目 → 返回其 result

        复杂度：O(N × D)，N = min(缓存条目数, 200)，D = 1024
        延迟：~50ms（embed API ~30ms + 扫描计算 ~20ms），远低于 LLM 的 2-10s
        """
        query_vec = self.emb.embed(query)

        best_sim = 0.0
        best_result = None
        scanned = 0

        # scan_iter 分批迭代，不阻塞 Redis
        for key_bytes in self.redis.conn.scan_iter(match=f"{_CACHE_PREFIX}*", count=50):
            if scanned >= _MAX_SCAN:
                break

            key = key_bytes.decode("utf-8") if isinstance(key_bytes, bytes) else key_bytes
            data = self.redis.get(key)
            if not data:
                continue

            try:
                entry = json.loads(data)
            except json.JSONDecodeError:
                continue

            scanned += 1

            # 只处理带 _embedding 的条目
            if not isinstance(entry, dict) or "_embedding" not in entry:
                continue

            sim = float(np.dot(query_vec, entry["_embedding"]))
            if sim > best_sim:
                result = entry.get("result")
                if result:
                    best_sim = sim
                    best_result = result

        if best_result and best_sim >= CACHE_SIMILARITY_THRESHOLD:
            logger.info(
                f"Semantic cache hit: sim={best_sim:.3f} "
                f"(thresh={CACHE_SIMILARITY_THRESHOLD}), scanned={scanned}"
            )
            return best_result

        if scanned > 0:
            logger.debug(
                f"Semantic cache miss: best_sim={best_sim:.3f} "
                f"(thresh={CACHE_SIMILARITY_THRESHOLD}), scanned={scanned}"
            )
        return None

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def get(self, profile: dict, query: str) -> dict | None:
        """两级缓存查询：精确匹配 → 语义匹配兜底。

        返回 dict | None，命中时返回 LLM 回答，未命中返回 None。
        """
        # 第一级：精确匹配（< 1ms）
        result = self._exact_get(profile, query)
        if result:
            logger.debug("Exact cache hit")
            return result

        # 第二级：语义相似度匹配（~50ms）
        return self._semantic_get(query)

    def set(self, profile: dict, query: str, result: dict):
        """写入缓存，同时存储 query embedding 供语义匹配使用。

        存储格式：{"_embedding": [1024 floats], "result": {...LLM 回答...}}
        过期时间：3600 秒（1 小时）
        """
        cache_key = self._make_key(profile, query)
        query_vec = self.emb.embed(query)
        entry = {"_embedding": query_vec, "result": result}
        self.redis.set(cache_key, json.dumps(entry, ensure_ascii=False), ex=3600)
