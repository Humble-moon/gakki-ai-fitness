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
        # 实例化下层检索器（第 1 层的两个独立检索服务）
        self.vector = VectorSearch()   # 向量语义检索 - 找语义相关的动作
        self.keyword = KeywordSearch()  # 关键词精确检索 - 找名称匹配的动作
        self.llm = LLMProvider()       # LLM 服务 - 用于评估检索质量 + 改写查询

    def search(self, query: str, filters: dict = None, max_retries: int = None) -> list:
        """Agentic RAG 的核心检索方法：迭代检索直到结果达标。

        输入：
            query:       str   — 用户的原始查询文本
            filters:     dict  — 可选过滤条件，如 {"equipment": "哑铃"}，仅传给向量检索
            max_retries: int   — 最大重试次数，默认从配置文件读取 AGENTIC_RAG_MAX_RETRIES

        输出：
            list[dict] — 所有轮次合并去重后的动作列表，每项包含 name, type, difficulty,
                         equipment, target_muscles, description, common_errors,
                         similarity, source（"vector" 或 "keyword"）

        核心逻辑 — Agentic 循环：
            for 每轮迭代:
                1. 并行执行向量检索 + 关键词检索（各取 5 条）
                2. 合并去重后加入累计结果集
                3. 如果不是最后一轮，让 LLM 评估结果质量：
                   a. 如果 quality_score >= 0.7 → 质量达标，提前终止
                   b. 如果 quality_score < 0.7 → 使用 LLM 改写后的查询进入下一轮
                4. 最后一轮无论质量如何都直接结束
        """
        # 确定最大重试次数：优先用传入参数，否则用全局配置
        max_retries = max_retries or AGENTIC_RAG_MAX_RETRIES

        # current_query 是每轮实际使用的查询文本
        # 第 1 轮使用用户原始查询，后续轮次可能被 LLM 改写
        current_query = query

        # all_results 累积所有轮次的检索结果
        # 为什么累积而非替换？不同改写角度可能召回不同但都有价值的动作
        all_results = []

        for attempt in range(max_retries):
            # ---- 步骤 1：并行双路检索 ----
            # 向量检索：找语义相关的动作（如"练胸"找到"卧推"）
            vec_results = self.vector.search(current_query, top_k=5, filters=filters)
            # 关键词检索：找名称匹配的动作（如"卧推"精确命中"哑铃卧推"）
            kw_results = self.keyword.search(current_query, top_k=5)
            # 合并 + 去重（同一个动作可能同时被两路检索到）
            combined = self._deduplicate(vec_results + kw_results)
            all_results.extend(combined)

            # ---- 步骤 2：LLM 质量评估（最后一轮跳过） ----
            # 最后一轮不需要评估，因为已经没有机会再改写了
            if attempt < max_retries - 1:
                # 构建评估消息：告诉 LLM 原始查询是什么、检索到了什么
                # 只传前 10 条避免超出 LLM 上下文窗口
                eval_msgs = build_retriever_eval_messages(query, combined[:10])
                # 调用小模型做检索评估 + 查询改写（REWRITE_MODEL，独立配置，默认 deepseek-chat）
                eval_result = self.llm.chat_with_json_mode(eval_msgs, model=REWRITE_MODEL)

                # quality_score: LLM 对当前检索结果的评分（0~1）
                # 0.7 是经验阈值：健身领域的查询通常需要较高精度
                # 低于 0.7 说明结果不够精确，需要改写查询重新搜索
                score = eval_result.get("quality_score", 0)
                if score >= 0.7:
                    # 质量达标！提前结束循环，不必耗尽所有重试次数
                    break

                # 质量不达标：使用 LLM 改写后的查询进入下一轮
                # rewritten_query 是 LLM 认为能产生更好结果的新查询
                # 例如 "瘦肚子" → "腹部脂肪燃烧 HIIT 训练动作"
                # fallback 到原查询以防 LLM 没有返回改写
                current_query = eval_result.get("rewritten_query", current_query)

        # 最终去重：多轮累积的结果可能有重复，再次去重确保返回干净的列表
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
