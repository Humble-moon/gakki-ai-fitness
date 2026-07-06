"""
=============================================================================
vector_search.py — 向量语义检索（第 1 层）
=============================================================================
【项目角色】
    这是 RAG 五层检索体系中的第 1 层 - 向量检索层。
    通过将用户查询转为向量，在 PostgreSQL pgvector 扩展中对 exercises 表
    做余弦距离最近邻搜索，找到语义相关但措辞可能不同的动作。

【五层检索关系】
    第 0 层:  EmbeddingService（embedding.py）—— 文本 → 向量
    第 1 层:  VectorSearch（本文件）      —— 向量相似度检索（语义匹配）
    第 2 层:  KnowledgeSearch             —— RRF 融合 + LLM 重排序
    第 3 层:  AgenticRAG                  —— 迭代评估 + 查询改写
    第 4 层:  GraphRAG                    —— 知识图谱推理
    第 5 层:  SemanticCache               —— 语义缓存加速

    本层被 AgenticRAG 直接调用（self.vector.search(...)）。
    也可被更上层的 orchestration 代码直接使用做快速语义召回。

【被谁调用】AgenticRAG.search()、其他需要语义检索的上层模块
【调用谁】  EmbeddingService.embed()（文本向量化）、PGClient.fetch_all()（PG 向量检索）

【检索原理】
    pgvector 的 <=> 运算符计算余弦距离（cosine distance = 1 - cosine similarity）。
    ORDER BY embedding <=> 按距离升序排列 = 按相似度降序排列。
    使用 1 - distance 将距离转回相似度分数（0~1，1 表示完全匹配）。
=============================================================================
"""

from src.storage.pg import PGClient
from src.rag.embedding import EmbeddingService


class VectorSearch:
    """向量语义检索类。

    【职责】
        对 PostgreSQL 中的 exercises 表做基于向量余弦相似度的语义检索，
        找到与用户查询语义最接近的健身动作。

    【使用流程】
        AgenticRAG 迭代检索时，每轮都会调用 vector.search() 获取 5 条候选，
        与 keyword.search() 结果合并后做去重和评估。

    【为什么需要向量检索】
        关键词匹配只能找到字面相同的词，比如用户搜"练背"但数据库里写的是"引体向上"，
        关键词搜不到。向量检索通过语义向量将"练背"和"引体向上"映射到相邻空间，
        从而召回措辞不同但语义相关的动作。
    """

    def __init__(self):
        self.pg = PGClient()           # PostgreSQL 客户端，用于执行向量查询
        self.emb = EmbeddingService()  # 文本向量化服务，将查询转为向量

    def search(self, query: str, top_k: int = 10, filters: dict = None) -> list:
        """向量语义检索主方法。

        输入：
            query:   str   — 用户查询文本（如"如何练胸肌"）
            top_k:   int   — 返回结果数量，默认 10
            filters: dict  — 可选过滤条件，如 {"equipment": "哑铃"} 限定器材

        输出：
            list[dict] — 每个字典包含动作的名称、难度、类型、器材、目标肌群、
                         描述、常见错误、相似度分数、来源标记 "source":"vector"

        核心逻辑步骤：
            1. 将查询文本通过 EmbeddingService 转为向量
            2. 将向量格式化为 pgvector 兼容的字符串格式 "[0.1, 0.2, ...]"
            3. 构建 SQL：使用 pgvector 的 <=> 运算符计算余弦距离
            4. 支持按 equipment 字段过滤（如只想看"哑铃"类动作）
            5. 按距离升序排列（距离越小 = 越相似），返回 top_k 条
        """
        # 步骤 1：文本 → 向量（归一化后的单位向量）
        vec = self.emb.embed(query)
        # 步骤 2：将 Python list 转为 PostgreSQL vector 类型可接受的字符串
        # 例：[0.1, -0.05, 0.3] → "[0.1,-0.05,0.3]"
        vec_str = f"[{','.join(str(v) for v in vec)}]"

        # 步骤 3：构建参数化查询的过滤条件
        # 使用参数化查询防止 SQL 注入：用户输入作为参数绑定而非拼入 SQL 字符串
        params = {"vec_str": vec_str, "top_k": top_k}
        filter_clause = ""
        if filters:
            if "equipment" in filters:
                filter_clause = "AND equipment = :equipment"
                params["equipment"] = filters["equipment"]

        # 步骤 4：构建并执行向量查询 SQL
        # <=> 是 pgvector 扩展提供的余弦距离运算符
        # embedding <=> vector 返回 0~2 的距离值（0 = 完全相同方向，2 = 完全相反）
        # 1 - distance 将距离转换为相似度（1 = 完全相同，-1 = 完全相反）
        # 这种转换让分数越接近 1 表示越相关，符合直觉
        sql = f"""
            SELECT name, name_en, exercise_type, difficulty, equipment,
                   target_muscles, description, common_errors,
                   1 - (embedding <=> :vec_str::vector) AS similarity
            FROM exercises
            WHERE embedding IS NOT NULL {filter_clause}
            ORDER BY embedding <=> :vec_str::vector
            LIMIT :top_k
        """
        rows = self.pg.fetch_all(sql, params)

        # 步骤 5：将数据库行转为标准字典格式
        # 标记 source="vector" 用于：
        #   1) 上层 AgenticRAG 的去重逻辑
        #   2) 在 UI 中展示检索来源（向量匹配 vs 关键词匹配）
        return [
            {"name": r[0], "name_en": r[1], "type": r[2], "difficulty": r[3],
             "equipment": r[4], "target_muscles": r[5], "description": r[6],
             "common_errors": r[7], "similarity": float(r[8]),
             "source": "vector"}
            for r in rows
        ]
