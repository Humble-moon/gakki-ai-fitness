"""
=============================================================================
agentic_rag.py — Agentic RAG 迭代检索（第 3 层）
=============================================================================
【项目角色】
    这是 RAG 五层检索体系中的第 3 层 — Agentic 智能检索层。
    核心创新在于：每次检索后让 LLM 评估结果质量，质量不达标则自动改写查询重新检索，
    形成"检索 → 评估 → 改写 → 再检索"的闭环，直到结果满意或达到最大重试次数。

    这是项目中位于最上层的检索编排器（orchestrator），
    协调 VectorSearch、KeywordSearch 和 LLM 三者协作完成智能检索。

【五层检索关系】
    第 0 层:  EmbeddingService（embedding.py）—— 文本 → 向量
    第 1 层:  VectorSearch   —— 对 exercises 表的向量检索        ← 被本层调用
    第 1 层:  KeywordSearch  —— 对 exercises 表的关键词检索      ← 被本层调用
    第 2 层:  KnowledgeSearch —— 对 knowledge_chunks 的混合检索
    第 3 层:  AgenticRAG（本文件）—— 迭代评估 + 查询改写           ← 你在这里
    第 4 层:  GraphRAG       —— 知识图谱推理
    第 5 层:  SemanticCache  —— 语义缓存加速

【被谁调用】应用层 API 路由（如 FastAPI endpoint /api/search）
【调用谁】  VectorSearch.search()、KeywordSearch.search()、LLMProvider、build_retriever_eval_messages

【核心创新 — Agentic 循环】
    传统 RAG = 查询 → 检索 → 返回结果（一次性，质量不可控）
    Agentic RAG = 查询 → 检索 → LLM 评估质量 → 不达标？改写查询 → 重新检索 → ...

    这个循环模拟了人类搜索时的行为：
    1. 先搜一次看结果
    2. 如果结果不满意，换个说法再搜
    3. 直到找到满意的结果或放弃

    在项目中，这解决了健身领域特有的问题：
    - 用户表达不精确："我想瘦肚子" → LLM 改写为"腹部减脂训练动作"
    - 术语不统一："练背" → LLM 改写为"背部肌群训练 引体向上 划船"
    - 结果太泛："怎么健身" → LLM 先评估质量低，再改写为更具体的查询

【示例流程】
    用户查询: "我肩膀疼，还能练胸吗"
      ↓ 第 1 轮
      检索 → 得到动作列表（可能主要是普通胸肌训练）
      LLM 评估 → quality_score=0.3（不够好，因为没考虑伤病限制）
      LLM 改写 → "低肩部压力胸肌训练 肩关节友好动作"
      ↓ 第 2 轮
      检索 → 得到更精准的动作（如器械飞鸟、绳索夹胸等肩关节友好的动作）
      LLM 评估 → quality_score=0.85（达标！）
      ↓
      返回所有轮次的去重合并结果
=============================================================================
"""

from src.rag.vector_search import VectorSearch
from src.rag.keyword_search import KeywordSearch
from src.llm.provider import LLMProvider
from src.llm.prompts.retriever import build_retriever_eval_messages
from src.config import AGENTIC_RAG_MAX_RETRIES, REWRITE_MODEL
from src.graphrag.search import GraphSearch


class AgenticRAG:
    """Agentic RAG 迭代检索编排器。

    【职责】
        编排 VectorSearch + KeywordSearch + LLM 三者协作，
        通过"检索-评估-改写"循环实现自适应检索。

    【使用流程】
        被应用层 API 直接调用：
        1. 用户提交查询 → AgenticRAG.search(query)
        2. 内部执行多轮迭代直到结果达标或达到最大重试次数
        3. 返回所有轮次去重后的合并结果

    【为什么需要 Agentic 循环】
        单次检索的质量高度依赖查询措辞。用户可能：
        - 使用口语化表达（"我想减肚子" vs 数据库中的"腹部减脂训练"）
        - 表达过于宽泛（"怎么健身" 返回结果太泛）
        - 遗漏关键约束（"肩膀疼能练胸吗" 需要额外添加"肩关节友好"条件）
        LLM 能识别这些问题并改写查询，比让用户自己反复尝试更高效。
    """

    def __init__(self):
        self.vector = VectorSearch()
        self.keyword = KeywordSearch()
        self.llm = LLMProvider()
        self._graph = None

    @property
    def graph(self):
        if self._graph is None:
            self._graph = GraphSearch()
        return self._graph

    def search(self, query: str, filters: dict = None, max_retries: int = None, route=None) -> list:
        """Search; route=None preserves the historical Agentic RAG path."""
        if route is not None:
            route_name = getattr(route, "name", route)
            route_name = str(route_name).split(".")[-1].lower()
            if route_name in {"graph", "injury_sensitive"}:
                try:
                    search = getattr(self.graph, "search", None)
                    if callable(search):
                        result = search(query, filters=filters)
                        return result if isinstance(result, list) else []
                except Exception:
                    return []
                return []
            if route_name == "knowledge":
                return []
        return self._legacy_search(query, filters=filters, max_retries=max_retries)

    def _legacy_search(self, query: str, filters: dict = None, max_retries: int = None) -> list:
        max_retries = max_retries or AGENTIC_RAG_MAX_RETRIES
        current_query = query
        all_results = []
        for attempt in range(max_retries):
            vec_results = self.vector.search(current_query, top_k=5, filters=filters)
            kw_results = self.keyword.search(current_query, top_k=5)
            combined = self._deduplicate(vec_results + kw_results)
            all_results.extend(combined)
            if attempt < max_retries - 1:
                eval_msgs = build_retriever_eval_messages(query, combined[:10])
                eval_result = self.llm.chat_with_json_mode(eval_msgs, model=REWRITE_MODEL)
                if eval_result.get("quality_score", 0) >= 0.7:
                    break
                current_query = eval_result.get("rewritten_query", current_query)
        return self._deduplicate(all_results)

    def _deduplicate(self, results: list) -> list:
        """按 'name' 字段去重（保留首次出现的条目）。

        输入：
            results: list[dict] — 可能包含重复动作的列表

        输出：
            list[dict] — 去重后的列表，保持原始顺序

        为什么按 name 去重而非 chunk_id：
            exercises 表的唯一标识是 name（动作名称），同一个动作无论
            是从向量检索还是关键词检索来的，都视为同一条结果。
            保留首次出现的条目意味着：如果向量检索先找到"卧推"，
            关键词检索后找到的"卧推"会被丢弃。
            这样可以保证 source 标记反映的是首次命中的检索路径。
        """
        seen = set()
        unique = []
        for r in results:
            if r["name"] not in seen:
                seen.add(r["name"])
                unique.append(r)
        return unique
