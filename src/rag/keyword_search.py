"""
=============================================================================
keyword_search.py — 关键词精确检索（第 1 层）
=============================================================================
【项目角色】
    这是 RAG 五层检索体系中的第 1 层 - 关键词检索层。
    通过 PostgreSQL 的 Trigram 模糊匹配（pg_trgm 扩展）和 ILIKE 子串匹配，
    对 exercises 表的 name 字段做精确/模糊关键词搜索。

【五层检索关系】
    第 0 层:  EmbeddingService（embedding.py）—— 文本 → 向量
    第 1 层:  KeywordSearch（本文件）        —— 关键词模糊匹配（字面匹配）
    第 1 层:  VectorSearch                   —— 向量语义匹配（语义匹配）
    第 2 层:  KnowledgeSearch                 —— RRF 融合 + LLM 重排序
    第 3 层:  AgenticRAG                      —— 迭代评估 + 查询改写
    第 4 层:  GraphRAG                        —— 知识图谱推理
    第 5 层:  SemanticCache                   —— 语义缓存加速

    本层与 VectorSearch 同级，被 AgenticRAG 同时调用，两者结果合并去重。

【被谁调用】AgenticRAG.search()、其他需要快速关键词匹配的上层模块
【调用谁】  PGClient.fetch_all()（执行 PostgreSQL 查询）

【为什么需要关键词检索】
    向量检索擅长语义匹配，但有时候用户的查询就是精确的动做名字
    （如"卧推"、"深蹲"），此时关键词匹配的精度和速度都优于向量检索。
    两者互补：关键词负责精确命中，向量负责语义泛化。

【Trigram 原理（pg_trgm 扩展）】
    pg_trgm 的 similarity() 函数基于字符串的 trigram 重合度计算相似性。
    例如 "bench press" 的 trigram 集合 = {"b", "be", "ben", "en", "enc", ...}
    "bench" 的 trigram 集合与之高度重合，因此 similarity 分数高。
    运算符 % 等价于 similarity(a, b) > pg_trgm.similarity_threshold（默认 0.3）。
=============================================================================
"""

from src.storage.pg import PGClient


class KeywordSearch:
    """PostgreSQL 关键词检索类。

    【职责】
        利用 PostgreSQL 的 pg_trgm 扩展（Trigram 模糊匹配）和 ILIKE 子串匹配，
        对 exercises 表的 name 字段做模糊关键词搜索。

    【使用流程】
        - AgenticRAG 每轮迭代调用 search() 获取 5 条关键词匹配结果
        - 与 VectorSearch 的结果合并后去重，进入 LLM 评估环节

    【与 VectorSearch 的互补关系】
        - VectorSearch: 搜"练胸" → 能召回"卧推"（语义相关但字面不同）
        - KeywordSearch: 搜"卧推" → 精确命中"哑铃卧推"、"杠铃卧推"（字面匹配）
        两者结果合并后覆盖面更广，再通过 RRF 融合去重排序。
    """

    def __init__(self):
        self.pg = PGClient()  # PostgreSQL 客户端，需要 pg_trgm 扩展已启用

    def search(self, query: str, top_k: int = 10) -> list:
        """关键词模糊搜索主方法。

        输入：
            query: str  — 用户查询文本（如"卧推"、"深蹲"）

        输出：
            list[dict] — 匹配的动作列表，包含名称、类型、难度、器材、目标肌群、
                         描述、常见错误、相似度分数、来源标记 "source":"keyword"

        核心逻辑步骤：
            1. 使用 pg_trgm 的 % 运算符做 Trigram 模糊匹配
               （如搜"握推"也能匹配到"卧推"——容忍拼写差异或读音相近的输入）
            2. 使用 ILIKE 做大小写不敏感的子串匹配作为补充
               （如搜"蹲"能匹配到"深蹲"、"箭步蹲"等所有包含该字的动作）
            3. 按 pg_trgm.similarity() 分数降序排列
        """
        # 构建 PostgreSQL 全文检索 SQL
        # WHERE 子句使用两个条件做 OR 组合：
        #   条件 1: name % :query
        #     → pg_trgm 的 Trigram 相似度运算符
        #     → 当 similarity(name, query) > 阈值（默认 0.3）时返回 True
        #     → 能处理拼写错误（"握推" ↔ "卧推"）
        #   条件 2: name ILIKE '%' || :query || '%'
        #     → 大小写不敏感的 LIKE 子串匹配
        #     → 作为 Trigram 匹配的补充，覆盖短查询词（2 字以下 Trigram 效果差）
        # ORDER BY similarity(name, :query) DESC
        #     → 按 Trigram 相似度降序，最匹配的排前面
        sql = """
            SELECT name, exercise_type, difficulty, equipment,
                   target_muscles, description, common_errors,
                   similarity(name, :query) AS sim
            FROM exercises
            WHERE name % :query OR name ILIKE '%' || :query || '%'
            ORDER BY sim DESC
            LIMIT :limit
        """
        rows = self.pg.fetch_all(sql, {"query": query, "limit": top_k})

        # 转为标准字典格式
        # 标记 source="keyword" 用于上层去重和来源展示
        # similarity 可能为 None（当 ILIKE 命中但 Trigram 分数不达标时），此时设为 0.0
        return [
            {"name": r[0], "type": r[1], "difficulty": r[2], "equipment": r[3],
             "target_muscles": r[4], "description": r[5], "common_errors": r[6],
             "similarity": float(r[7]) if r[7] else 0.0, "source": "keyword"}
            for r in rows
        ]
