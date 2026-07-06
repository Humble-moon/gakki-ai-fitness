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
【调用谁】  EmbeddingService.embed()（暂未使用，预留用于语义相似度匹配）、
           RedisClient（缓存读写）

【缓存策略】
    - 键生成：基于 (用户画像 JSON + 查询文本) 的 MD5 哈希
      → 只有画像和查询完全相同时才命中（精确匹配，非语义匹配）
    - 过期时间：3600 秒（1 小时）
      → 健身知识不像新闻，1 小时内不会变，过期后重新生成即可
    - 存储内容：_embedding（查询向量，预留）+ result（LLM 回答）
    - 命名空间：cache:fitness:{hash} 前缀隔离，防止与其他缓存键冲突

【为什么需要 SemanticCache】
    LLM 调用是系统中最昂贵的操作（延迟 2~10 秒 + API 费用）。
    用户在短时间内可能反复问相同或类似的问题：
    - "我想练胸肌有什么动作？"（第 1 次 → 调用 LLM）
    - "我想练胸肌"（第 2 次 → 从缓存返回！）
    缓存命中后延迟从秒级降到毫秒级，且零 API 费用。

【当前实现 vs 未来演进】
    当前：精确匹配缓存（profile + query 完全相同时命中）
    未来：语义匹配缓存（利用存储的 _embedding 做向量相似度，相似查询也能命中）
    配置中的 CACHE_SIMILARITY_THRESHOLD 为语义匹配预留。
=============================================================================
"""

import json
import hashlib
from src.rag.embedding import EmbeddingService
from src.storage.redis_client import RedisClient
from src.config import CACHE_SIMILARITY_THRESHOLD


class SemanticCache:
    """语义缓存类（当前为精确匹配缓存 + 预留语义匹配能力）。

    【职责】
        缓存 (用户画像, 查询) → LLM 生成的健身回答，避免重复调用 LLM。

    【使用流程】
        1. 用户提问 → 先查 SemanticCache.get(profile, query)
        2. 命中缓存 → 直接返回结果（延迟 < 10ms，零 LLM 费用）
        3. 未命中 → 走完整 RAG 流程 → 调用 SemanticCache.set(profile, query, result)
           → 下次相同查询直接命中

    【为什么缓存键包含 profile（用户画像）】
        同一个查询"推荐训练动作"，不同用户画像会得到完全不同的回答：
        - 新手 + 减脂目标 → 推荐低强度全身动作
        - 老手 + 增肌目标 → 推荐大重量分化训练
        如果只用 query 做键，会产生错误的缓存命中。
    """

    def __init__(self):
        self.redis = RedisClient()   # Redis 客户端，用于缓存读写
        self.emb = EmbeddingService()  # 向量服务（当前用于存储查询向量，未来用于语义匹配）

    def _make_key(self, profile: dict, query: str) -> str:
        """生成缓存键。

        输入：
            profile: dict — 用户画像（如 {"level": "中级", "goal": "增肌"}）
            query:   str  — 用户查询文本

        输出：
            str — Redis 缓存键，格式为 "cache:fitness:{32位MD5哈希}"

        生成逻辑：
            1. 将 profile 字典 JSON 序列化（sort_keys=True 保证键顺序一致）
            2. 拼接 query 文本
            3. 对整个字符串做 MD5 哈希
            4. 添加 "cache:fitness:" 前缀做命名空间隔离

        为什么用 MD5 而非直接拼接字符串做键：
            - profile 可能是很长的 JSON，直接做键会超出 Redis 建议的键长度
            - MD5 产生固定 32 字符的十六进制串，紧凑且均匀分布
            - sort_keys=True 确保 {"a":1,"b":2} 和 {"b":2,"a":1} 产生相同的键
        """
        raw = json.dumps(profile, sort_keys=True) + query
        return f"cache:fitness:{hashlib.md5(raw.encode()).hexdigest()}"

    def get(self, profile: dict, query: str) -> dict | None:
        """从缓存中获取结果。

        输入：
            profile: dict — 用户画像
            query:   str  — 用户查询

        输出：
            dict | None — 缓存命中时返回 LLM 的回答字典，
                          缓存未命中或过期时返回 None

        逻辑：
            1. 根据 profile+query 生成缓存键
            2. 从 Redis 读取该键的值
            3. 如果值是 {"result": ..., "_embedding": ...} 格式，提取 result 字段
            4. 如果值是旧格式（直接的 dict），直接返回
               （向后兼容旧版缓存数据）
        """
        cache_key = self._make_key(profile, query)
        data = self.redis.get(cache_key)
        if data:
            entry = json.loads(data)
            # 新格式：{"_embedding": [...], "result": {...}}
            if isinstance(entry, dict) and "result" in entry:
                return entry["result"]
            # 旧格式兼容：直接存储的 dict（无 _embedding 包装）
            return entry
        return None

    def set(self, profile: dict, query: str, result: dict):
        """将结果写入缓存。

        输入：
            profile: dict — 用户画像
            query:   str  — 用户查询
            result:  dict — LLM 生成的回答（包含动作推荐、解释等）

        输出：
            无（副作用：写入 Redis）

        逻辑：
            1. 生成缓存键
            2. 将查询文本向量化（_embedding 字段，为未来的语义匹配预留）
            3. 包装为 {"_embedding": [...], "result": {...}} 格式
            4. 写入 Redis，设置 3600 秒（1 小时）过期时间

        为什么存储 _embedding：
            当前只做精确匹配，但向量已存储在缓存中。
            未来可以实现"搜索相似缓存"：当精确匹配未命中时，
            用 CACHE_SIMILARITY_THRESHOLD 比较新查询和历史查询的向量相似度，
            相似度超过阈值则复用缓存结果。这在健身场景中很有价值——
            "怎么练胸肌"和"胸部训练方法"应该命中同一个缓存。
        """
        cache_key = self._make_key(profile, query)
        # 将查询向量化并存入缓存（预留语义匹配能力）
        query_vec = self.emb.embed(query)
        entry = {"_embedding": query_vec, "result": result}
        # ex=3600 表示 1 小时后自动过期（Redis TTL）
        # 健身知识不像新闻/股票那样实时变化，1 小时是合理的时间窗口
        self.redis.set(cache_key, json.dumps(entry, ensure_ascii=False), ex=3600)
