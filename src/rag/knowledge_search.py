"""
=============================================================================
knowledge_search.py — 知识库混合检索 + RRF 融合 + LLM 重排序（第 2 层）
=============================================================================
【项目角色】
    这是 RAG 五层检索体系中的第 2 层 — 知识库混合检索层。
    针对 knowledge_chunks 表（外部知识文档分块），执行"双路检索 → RRF 融合 → LLM 重排序"
    三阶段流水线，是项目中检索质量最高的单次检索模块。

【五层检索关系】
    第 0 层:  EmbeddingService（embedding.py）—— 文本 → 向量
    第 1 层:  VectorSearch   —— 对 exercises 表的向量检索
    第 1 层:  KeywordSearch  —— 对 exercises 表的关键词检索
    第 2 层:  KnowledgeSearch（本文件）—— 对 knowledge_chunks 的混合检索 + RRF + 重排序  ← 你在这里
    第 3 层:  AgenticRAG     —— 迭代评估 + 查询改写
    第 4 层:  GraphRAG       —— 知识图谱推理
    第 5 层:  SemanticCache  —— 语义缓存加速

    注意区别：
    - VectorSearch/KeywordSearch 检索的是 exercises 表（内置健身动作库）
    - KnowledgeSearch 检索的是 knowledge_chunks 表（外部 Markdown 知识文档）
      → 这是两个独立的知识来源，服务于不同的问答场景

【被谁调用】AgenticRAG（上层）、API 路由（直接查询知识库）
【调用谁】  EmbeddingService.embed()、LLMProvider.chat_with_json_mode()、PGClient

【架构流水线】
    用户查询
      │
      ├── 向量检索（PG cosine）──→ 20 条候选
      │
      ├── 关键词检索（PG trigram）──→ 20 条候选
      │
      └── RRF 融合（倒数排名融合）
            │
            └── Top-20 候选（去重 + 加权排序）
                  │
                  └── LLM 重排序（最多取 15 条给 LLM 打分）
                        │
                        └── 最终 Top-5 结果

【为什么需要 RRF（Reciprocal Rank Fusion）】
    向量检索和关键词检索的分数分布完全不同：
    - 向量检索：余弦相似度 0.7~0.99（集中在高分段）
    - 关键词检索：Trigram 相似度 0.0~1.0（分布均匀）
    直接用分数相加会导致向量检索的结果总是排前面，关键词检索结果被淹没。
    RRF 只关心排名不关心原始分数，天然消除了不同分布带来的偏差。

【为什么需要 LLM 重排序】
    RRF 融合后的 Top-20 仍可能包含"看似相关但实际不相关"的片段。
    例如查询"减脂饮食计划"，RRF 可能把一篇讲"增肌饮食"的文章排到前面，
    因为向量空间里"饮食"这个词把两者拉近了。
    LLM 重排序通过语义理解辨别真正的相关性，过滤掉这类误报。
=============================================================================
"""

import logging
from typing import List, Dict, Optional

from src.storage.pg import PGClient
from src.rag.embedding import EmbeddingService
from src.llm.provider import LLMProvider

logger = logging.getLogger(__name__)

# ---- RRF 融合常量 ----
# RRF_K: RRF 公式中的平滑常数 k
# 取值 60 是文献中的标准值（Cormack et al., 2009, "Reciprocal Rank Fusion"）
# 作用：防止排名靠前的文档权重过大，让公式更平滑
# 公式：RRF_score(d) = sum over lists of 1/(k + rank_i(d))
# 例如 rank=1 时得分为 1/(60+1)=0.0164，rank=100 时得分为 1/(60+100)=0.0063
# 两者差距约 2.6 倍而非 100 倍，起到了平滑作用
RRF_K = 60

# 重排序前的候选数量（双路各取 20 条，RRF 融合后约 20~40 条去重结果）
PRE_RANK_TOP = 20

# 重排序后的最终返回数量
FINAL_TOP = 5


class KnowledgeSearch:
    """知识库混合检索类（向量 + 关键词 → RRF → 重排序）。

    【职责】
        对 knowledge_chunks 表执行完整的三阶段检索流水线：
        1. 双路并行检索（向量语义 + 关键词字面）
        2. RRF 倒数排名融合（去重 + 排序）
        3. LLM 语义重排序（精排 Top-5）

    【使用场景】
        - 用户问"如何制定减脂饮食计划？"→ 从知识库中检索相关文档片段
        - AgenticRAG 迭代评估时作为备选检索手段
        - API 直接调用 search_with_fallback() 做带兜底的稳健检索
    """

    def __init__(self):
        self.pg = PGClient()           # PostgreSQL 客户端
        self.emb = EmbeddingService()  # 文本向量化服务
        self.llm = LLMProvider()       # LLM 调用服务（用于重排序）

    # =====================================================================
    # 阶段 1a: 向量检索
    # =====================================================================

    def vector_search(self, query: str, top_k: int = PRE_RANK_TOP) -> List[dict]:
        """在 knowledge_chunks 表上执行向量余弦相似度检索。

        输入：
            query: str — 用户查询文本
            top_k: int — 返回数量，默认 20（PRE_RANK_TOP）

        输出：
            List[dict] — 每个字典包含 chunk_id, title, content, source_file,
                         chunk_index, score（余弦相似度）, source="vector"

        说明：
            与 VectorSearch 类的区别：
            - VectorSearch 检索的是 exercises 表（健身动作）
            - 本方法检索的是 knowledge_chunks 表（外部知识文档分块）
            两者使用相同的向量检索逻辑，但面向不同的数据表和数据源。
        """
        # 查询文本 → 向量
        vec = self.emb.embed(query)
        vec_str = f"[{','.join(str(v) for v in vec)}]"

        # pgvector 余弦距离检索
        # <=> 运算符: cosine_distance = 1 - cosine_similarity
        # 1 - distance 将距离转回相似度分数
        sql = """
            SELECT chunk_id, title, content, source_file, chunk_index,
                   1 - (embedding <=> CAST(:vec AS vector)) AS similarity
            FROM knowledge_chunks
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> CAST(:vec AS vector)
            LIMIT :limit
        """
        rows = self.pg.fetch_all(sql, {"vec": vec_str, "limit": top_k})
        return [
            {"chunk_id": r[0], "title": r[1], "content": r[2],
             "source_file": r[3], "chunk_index": r[4],
             "score": float(r[5]), "source": "vector"}
            for r in rows
        ]

    # =====================================================================
    # 阶段 1b: 关键词检索
    # =====================================================================

    def keyword_search(self, query: str, top_k: int = PRE_RANK_TOP) -> List[dict]:
        """在 knowledge_chunks 表上执行 Trigram + ILIKE 关键词检索。

        输入：
            query: str — 用户查询文本
            top_k: int — 返回数量，默认 20

        输出：
            List[dict] — 匹配的知识片段，source="keyword"

        说明：
            同时搜索 content 和 title 两个字段，比 KeywordSearch 类（只搜 name 字段）
            覆盖范围更广。
        """
        sql = """
            SELECT chunk_id, title, content, source_file, chunk_index,
                   similarity(content, :query) AS sim
            FROM knowledge_chunks
            WHERE content % :query OR content ILIKE '%' || :query || '%'
               OR title ILIKE '%' || :query || '%'
            ORDER BY sim DESC
            LIMIT :limit
        """
        rows = self.pg.fetch_all(sql, {"query": query, "limit": top_k})
        return [
            {"chunk_id": r[0], "title": r[1], "content": r[2],
             "source_file": r[3], "chunk_index": r[4],
             "score": float(r[5]) if r[5] else 0.0, "source": "keyword"}
            for r in rows
        ]

    # =====================================================================
    # 阶段 2: RRF 倒数排名融合
    # =====================================================================

    def rrf_fusion(
        self,
        vector_results: List[dict],
        keyword_results: List[dict],
        k: int = RRF_K,
    ) -> List[dict]:
        """倒数排名融合（RRF）：无需分数校准即可合并两个排序列表。

        输入：
            vector_results:  List[dict] — 向量检索结果列表（已按相似度排序）
            keyword_results: List[dict] — 关键词检索结果列表（已按相似度排序）
            k:              int        — RRF 平滑常数，默认 60

        输出：
            List[dict] — 按 RRF 分数降序排列的融合结果，每个 dict 新增 rrf_score 字段

        RRF 公式:
            RRF_score(d) = sum(1 / (k + rank_i(d)))，i 遍历 {vector, keyword}

        为什么 RRF 优于简单去重：
            1. 在两个列表中排名都靠前的文档会得到加权提升
               （例如某文档在向量排第 2、关键词排第 3 → RRF 分高于两个单独排第 5 的文档）
            2. 无需对不同分数分布进行归一化
               （向量分 0.9 ≠ 关键词分 0.9，RRF 直接忽略原始分数只关心排名）
            3. 对任一列表中的离群值具有鲁棒性
               （即使向量检索的第一名是噪声，RRF 也会因为有平滑常数而不过度惩罚）
        """
        scores = {}  # chunk_id → 累计 RRF 分数
        docs = {}    # chunk_id → 原始文档信息（保留完整的元数据）

        # 遍历向量检索结果，累加 RRF 分数
        # enumerate(..., start=1) 使排名从 1 开始（不是 0）
        for rank, doc in enumerate(vector_results, start=1):
            cid = doc["chunk_id"]
            # RRF 核心公式：1 / (k + rank)
            # scores.get(cid, 0) 如果该文档也出现在关键词结果中，则在已有分数上累加
            scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank)
            docs[cid] = doc  # 保留文档引用（同一个 chunk_id 只保留一份）

        # 遍历关键词检索结果，累加 RRF 分数
        for rank, doc in enumerate(keyword_results, start=1):
            cid = doc["chunk_id"]
            scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank)
            docs[cid] = doc

        # 按 RRF 分数降序排列（分数越高 = 综合排名越靠前）
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        fused = []
        for cid, rrf_score in ranked:
            doc = docs[cid].copy()  # 复制避免修改原始数据
            doc["rrf_score"] = round(rrf_score, 6)
            fused.append(doc)

        return fused

    # =====================================================================
    # 阶段 3: LLM 重排序
    # =====================================================================

    def re_rank(self, query: str, candidates: List[dict], top_k: int = FINAL_TOP) -> List[dict]:
        """基于 LLM 对 Top 候选进行语义重排序。

        输入：
            query:      str       — 用户原始查询
            candidates: List[dict]— RRF 融合后的候选列表（约 20~40 条）
            top_k:      int       — 最终返回数量，默认 5

        输出：
            List[dict] — 按 LLM 打分降序的最终结果，新增 rerank_score 和 rerank_reason 字段

        为什么需要 LLM 重排序：
            RRF 只是基于排名的统计融合，无法理解真正的语义相关性。
            例如查询"减脂饮食"，RRF 可能把"增肌期饮食策略"排到前面，
            因为两个文档都大量使用了"蛋白质"、"热量"等共享词汇。
            LLM 能理解"减脂"和"增肌"是不同目标，从而给前者打低分。

        为什么最多只取 15 条给 LLM：
            LLM 的上下文窗口有限，且候选越多耗时越长、成本越高。
            15 条是成本与效果的经验平衡点。
        """
        # 候选数量不超过 top_k 时，无需重排序，直接返回
        if len(candidates) <= top_k:
            return candidates

        # 构建给 LLM 的重排序提示词
        # 每条候选截取前 300 字符作为摘要（太多内容 LLM 处理慢，太少信息不足）
        items_text = []
        for i, doc in enumerate(candidates[:15]):  # 最多取 15 条候选给 LLM
            snippet = doc["content"][:300].replace("\n", " ")
            items_text.append(f"[{i}] 【{doc['title']}】{snippet}")

        prompt = f"""你是一位健身知识检索专家。用户提出了以下问题：

用户问题：{query}

以下是检索到的候选文档片段。请判断每条与用户问题的相关程度，给出 0-10 的分数（10=完全相关，0=完全无关）。

候选片段：
{chr(10).join(items_text)}

请以 JSON 格式返回评分结果：
{{"scores": [{{"index": 0, "score": 8, "reason": "直接回答了..."}}, ...]}}

只返回分数最高、最相关的 5 条，按分数降序排列。"""

        try:
            # 调用 LLM，使用 JSON 模式确保返回格式可控
            result = self.llm.chat_with_json_mode(
                [{"role": "user", "content": prompt}]
            )
            scores_list = result.get("scores", [])

            # 将 LLM 返回的评分映射回原始候选文档
            scored = []
            for item in scores_list:
                idx = item.get("index", -1)
                # 安全检查：确保索引在有效范围内
                if 0 <= idx < len(candidates):
                    doc = candidates[idx].copy()
                    doc["rerank_score"] = item.get("score", 0)
                    doc["rerank_reason"] = item.get("reason", "")
                    scored.append(doc)

            # 按 LLM 评分降序排列
            scored.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            return scored[:top_k]

        except Exception as e:
            # LLM 调用失败时的降级策略：返回 RRF 融合的前 top_k 条
            # 宁可返回未经精排的结果，也好过返回空列表
            logger.warning(f"Re-rank failed, returning top candidates: {e}")
            return candidates[:top_k]

    # =====================================================================
    # 完整检索流水线
    # =====================================================================

    def search(
        self,
        query: str,
        top_k: int = FINAL_TOP,
        enable_rerank: bool = True,
    ) -> List[dict]:
        """完整的三阶段检索流水线入口。

        输入：
            query:         str  — 用户查询文本
            top_k:         int  — 最终返回数量，默认 5
            enable_rerank: bool — 是否启用 LLM 重排序，默认 True

        输出：
            List[dict] — 最终检索结果，包含 chunk_id, title, content, source_file,
                         chunk_index, score, source, rrf_score, rerank_score/rerank_reason

        流水线步骤：
            阶段 1: 双路并行检索 → 向量 20 条 + 关键词 20 条
            阶段 2: RRF 倒数排名融合 → 去重 + 综合排序
            阶段 3: LLM 重排序（可选）→ 精排 Top-5
        """
        # ---- 阶段 1：双路检索 ----
        # 向量和关键词检索各自独立，可以并行（当前是串行）
        # TODO: 可使用 asyncio.gather 并发执行以加速
        vector_results = self.vector_search(query)
        keyword_results = self.keyword_search(query)

        # ---- 阶段 2：RRF 融合 ----
        # 将两路结果按排名融合，消除不同分数分布的偏差
        fused = self.rrf_fusion(vector_results, keyword_results)

        # ---- 阶段 3：重排序 ----
        # 注意：只有融合结果多于 top_k 时才需要重排序
        # 如果融合后只剩 ≤ top_k 条，重排序没有意义（LLM 只是重新排列这些条目）
        if enable_rerank and len(fused) > top_k:
            results = self.re_rank(query, fused, top_k)
        else:
            results = fused[:top_k]

        # 空结果日志：帮助排查知识库覆盖不足的问题
        if not results:
            logger.info(f"No results found for query: {query[:50]}...")
            return []

        return results

    # =====================================================================
    # 带兜底的稳健检索
    # =====================================================================

    def search_with_fallback(self, query: str) -> List[dict]:
        """带兜底策略的检索：混合检索失败时回退到纯关键词检索。

        输入：
            query: str — 用户查询文本

        输出：
            List[dict] — 检索结果（保证非空或至少尝试过兜底策略）

        为什么需要兜底：
            向量检索依赖知识库中已向量化的片段。如果 ingestion 流程未完成
            或某文档的 embedding 写入失败，向量检索可能返回空结果。
            但关键词检索不需要 embedding，可以作为可靠的兜底方案。

        策略：
            1. 先尝试完整的混合检索流水线
            2. 如果混合检索返回空，回退到纯关键词检索
            3. 纯关键词检索仍然空 → 返回空列表（暴露问题，让上游处理）
        """
        results = self.search(query)

        if not results:
            logger.info("Full search returned empty, falling back to keyword-only...")
            results = self.keyword_search(query, top_k=FINAL_TOP)

        return results
