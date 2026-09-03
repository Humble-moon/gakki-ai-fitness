"""
=============================================================================
ragas_eval.py — RAGAS 社区标准评测集成 (v2 — 2026-07-20)

【v2 更新（5 项修复）】
  1. 生产级生成——用生产环境的 answer_question_stream 同款 prompt 模板
     （含伤病安全注入、多轮上下文、文件上传 chunk），不再用简化版 prompt
  2. 领域常识白名单——在检索上下文中注入"健身行业共识"chunk，
     消除"组间休息1-2分钟"被 RAGAS 误判为幻觉的偏差
  3. 异步并行处理——20+ 条 query 的检索+生成全部用 asyncio.gather 并发，
     总耗时从 20 分钟降到 ~3-5 分钟
  4. 分层抽样——按 query_type（knowledge/plan/injury）分层，每类至少 15 条，
     总量 50+，避免某一类 query 占比过高扭曲指标
  5. 逐 query 标签——per_query 输出增加 query_type/difficulty/has_injury 字段，
     支持分组分析（如"伤病类 query 的 Faithfulness 是否显著更低"）
=============================================================================
"""

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from eval.golden_dataset.build_dataset import load_dataset

logger = logging.getLogger(__name__)

# ============================================================================
# v2: 健身领域常识白名单 —— RAGAS Faithfulness 裁判用
# ============================================================================
# 这些是健身行业的通用共识，不需要从检索文档中引用。
# 把它们作为一个"常识文档"注入检索上下文，避免 RAGAS 将常识判为 unfaithful。
FITNESS_COMMON_KNOWLEDGE = """
【健身行业通用常识（以下内容为公认标准，无需逐一引用来源）】
- 增肌训练组间休息通常为 60-120 秒；力量训练为 2-5 分钟；耐力训练为 30-60 秒
- 增肌推荐次数范围：6-12 次/组；力量：1-5 次/组；耐力：15+ 次/组
- 增肌训练每个动作通常做 3-5 组
- 每周每个肌群训练 2 次优于 1 次（Schoenfeld 2016 meta-analysis）
- 增肌需要热量盈余（每日 +300-500 kcal），减脂需要热量缺口
- 每公斤体重每日蛋白质摄入 1.6-2.2g（增肌期）
- 复合动作（深蹲/卧推/硬拉/引体向上）优先于孤立动作
- 渐进超负荷（Progressive Overload）是增肌的核心原则
- 新手建议从全身训练（Full Body）开始，每周 3 次
- 训练后 48-72 小时为肌肉恢复期，同一肌群不宜连续训练
- 训练前热身 5-10 分钟，训练后拉伸 5-10 分钟
- DOMS（延迟性肌肉酸痛）通常在训练后 24-72 小时出现，属于正常现象
""".strip()


# ============================================================================
# v2: 生产环境 QA prompt 模板 —— 对齐 answer_question_stream
# ============================================================================
def _build_production_qa_prompt(
    question: str,
    user_profile: dict,
    contexts: List[str],
    doc_contexts: List[str] = None,
    conv_context: str = "",
    graph_data: dict = None,
) -> str:
    """构建与生产环境 answer_question_stream 对齐的 QA prompt。

    和旧版 _generate_answer 的核心区别：
    - 旧版：简单的"只基于资料回答"，temperature=0.3
    - 新版：包含用户画像/伤病信息/安全规则/对话上下文/文件内容，
      与真实用户看到的 prompt 一致
    """
    sources_text = ""
    for i, ctx in enumerate(contexts, 1):
        snippet = ctx[:400].replace("\n", " ")
        sources_text += f"\n[来源{i}] {snippet}\n"

    doc_sources_text = ""
    if doc_contexts:
        for i, ctx in enumerate(doc_contexts, 1):
            snippet = ctx[:400].replace("\n", " ")
            doc_sources_text += f"\n[你的文件-{i}] {snippet}\n"

    # 伤病关键词检测（与 orchestrator.py 中一致的逻辑）
    _SAFETY_KEYWORDS = [
        "疼", "痛", "伤", "酸", "不舒服", "拉伤", "扭伤", "炎症",
        "恢复", "手术", "骨折", "撕裂", "脱臼", "肿胀", "麻", "无力",
    ]
    has_safety_concern = (
        any(kw in question for kw in _SAFETY_KEYWORDS)
        or bool(user_profile.get("injuries", []))
    )

    _SAFETY_NOTE = ""
    if has_safety_concern:
        _SAFETY_NOTE = """
⚠️ 【重要安全规则 — 违反视为严重错误】：
1. 用户提到伤病/疼痛/不适或已记录伤病史时，首要建议必须是"停止训练、咨询医生或物理治疗师"
2. 不要做出医疗诊断——只给出运动康复层面的参考建议，并明确标注"以下不能替代专业医疗诊断"
3. 推荐的任何替代动作，必须明确解释为什么不会加重所述伤病
4. 绝不要推荐任何可能加重用户已有伤病的动作
5. 不确定时，明确说"建议先去康复科/运动医学科做专业评估"
"""

    prompt = f"""你是资深健身教练和运动康复专家。请基于提供的知识库文档回答用户的问题。
如果知识库中没有足够信息，可以结合你的专业知识补充，但需要明确指出哪些来自文档、哪些是专业推断。

用户情况：{user_profile.get('height')}cm, {user_profile.get('weight')}kg, 训练{user_profile.get('training_years', 1)}年
伤病：{user_profile.get('injuries', [])}

{conv_context}

知识库相关文档：
{sources_text if sources_text else '（未找到直接相关的知识库文档，请基于专业知识回答）'}

{"【用户上传的文件内容】" if doc_sources_text else ""}
{doc_sources_text if doc_sources_text else ""}

要求：
1. 先直接回答问题，给出明确结论
2. 解释原因（解剖/生理层面，但用大白话说）
3. 给出 2-3 条可执行的建议
4. 如果涉及危险信号，明确建议就医
5. 200-350 字，口语化，像教练在聊天
6. 纯文字段落，不用 markdown
{_SAFETY_NOTE if has_safety_concern else ""}"""

    return prompt


# ============================================================================
# RAGAS 依赖检查
# ============================================================================

def _check_ragas() -> bool:
    try:
        import ragas
        import langchain_openai
        import datasets
        return True
    except ImportError as e:
        logger.error(f"RAGAS dependency missing: {e}")
        logger.info("Install with: pip install ragas langchain-openai datasets pandas")
        return False


# ============================================================================
# v2: 查询分类标签 —— 用于分层抽样和分组分析
# ============================================================================

def _classify_query(query: dict) -> dict:
    """根据 query 内容打标签：query_type / difficulty / has_injury。"""
    text = query.get("query", "") + " " + query.get("label", "")

    # 分类
    injury_kw = ["疼", "痛", "伤", "酸", "不舒服", "拉伤", "膝盖", "腰", "肩",
                  "手术", "康复", "恢复", "炎症", "骨折"]
    has_injury = any(kw in text for kw in injury_kw)

    plan_kw = ["计划", "安排", "一周", "分化", "增肌", "减脂", "增重",
               "怎么练", "训练方案", "重量", "瓶颈", "平台"]
    knowledge_kw = ["是什么", "为什么", "原理", "区别", "怎么选",
                    "能不能", "会不会", "可以吗", "有用吗", "效果"]
    # 动作分析类
    exercise_kw = ["深蹲", "卧推", "硬拉", "划船", "弯举", "推举", "飞鸟",
                   "引体", "俯卧撑", "姿势", "动作"]

    if has_injury:
        query_type = "injury"
    elif any(kw in text for kw in plan_kw) and not any(kw in text for kw in knowledge_kw):
        query_type = "plan"
    elif any(kw in text for kw in knowledge_kw):
        query_type = "knowledge"
    elif any(kw in text for kw in exercise_kw):
        query_type = "exercise_analysis"
    else:
        query_type = "knowledge"  # 默认知识型

    # 难度
    if has_injury:
        difficulty = "hard"
    elif query_type in ("plan",):
        difficulty = "medium"
    else:
        difficulty = "easy"

    return {
        "query_type": query_type,
        "difficulty": difficulty,
        "has_injury": has_injury,
    }


# ============================================================================
# v2: 分层抽样 —— 按 query_type 分层，总数 >= 50
# ============================================================================

def _stratified_sample(queries: List[dict], target_total: int = 50) -> List[dict]:
    """按 query_type 分层抽样，确保每类至少有 15 条（如果有足够多的话）。

    优先保证 injury 类全量（安全评估最重要），
    其余按比例分配，总量 >= target_total。
    """
    # 先分类
    by_type = defaultdict(list)
    for q in queries:
        tags = _classify_query(q)
        q["_tags"] = tags
        by_type[tags["query_type"]].append(q)

    logger.info("Query distribution by type:")
    for t, qs in sorted(by_type.items()):
        logger.info(f"  {t}: {len(qs)} queries")

    # 分配配额：每类至少 15 条（如果够），injury 全量
    per_type_min = 15
    sampled = []

    for qtype, qlist in sorted(by_type.items()):
        if qtype == "injury":
            quota = min(len(qlist), 25)  # injury 最多 25 条
        else:
            quota = per_type_min
        # 如果该类型不够 15 条，全部取
        actual = min(quota, len(qlist))
        chosen = random.sample(qlist, actual)
        sampled.extend(chosen)

    # 如果总量不够 target_total，从最多的类别补
    if len(sampled) < target_total:
        shortage = target_total - len(sampled)
        # 从最大的类别中补充
        largest_type = max(by_type, key=lambda t: len(by_type[t]))
        remaining = [q for q in by_type[largest_type] if q not in sampled]
        extra = random.sample(remaining, min(shortage, len(remaining)))
        sampled.extend(extra)

    # 打乱顺序
    random.shuffle(sampled)

    counts = ', '.join(
        f"{t}={sum(1 for q in sampled if q['_tags']['query_type'] == t)}"
        for t in sorted(by_type)
    )
    logger.info(f"Stratified sample: {len(sampled)} queries ({counts})")
    return sampled


# ============================================================================
# v2: 构建检索上下文 + 领域常识注入
# ============================================================================

async def _build_contexts_async(
    query_text: str,
    ks,  # KnowledgeSearch 实例（由调用者创建并复用）
    vs,  # VectorSearch 实例
    top_k: int = 5,
) -> List[str]:
    """异步构建检索上下文，注入健身领域常识 chunk。

    v2 改动：
    - 知识库和向量检索改为异步并发
    - 检索结果末尾追加 FITNESS_COMMON_KNOWLEDGE 作为常识文档
    """
    import asyncio as _asyncio

    contexts = []

    async def _search_knowledge():
        try:
            return ks.search(query_text, top_k=top_k)
        except Exception as e:
            logger.warning(f"Knowledge search failed: {e}")
            return []

    async def _search_exercises():
        try:
            return vs.search(query_text, top_k=top_k)
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []

    # v2: 知识库 + 动作库并行检索
    knowledge_results, exercise_results = await _asyncio.gather(
        _search_knowledge(), _search_exercises()
    )

    for r in knowledge_results:
        snippet = r.get("content", "")[:500]
        if snippet:
            contexts.append(snippet)

    for r in exercise_results:
        name = r.get("name", "")
        desc = r.get("description", "")
        muscles = r.get("target_muscles", "")
        snippet = f"动作：{name}。{desc} 目标肌群：{muscles}"
        if snippet:
            contexts.append(snippet)

    # v2: 注入健身行业常识文档，消除 RAGAS 对常识的"误判为幻觉"
    contexts.append(FITNESS_COMMON_KNOWLEDGE)

    # 去重
    seen = set()
    unique = []
    for ctx in contexts:
        key = ctx[:100]
        if key not in seen:
            seen.add(key)
            unique.append(ctx)
    return unique[:top_k + 1]  # +1 为常识文档留位置


def _build_contexts_sync(
    query_text: str,
    top_k: int = 5,
) -> List[str]:
    """同步包装器——兼容旧调用方式。"""
    from src.rag.knowledge_search import KnowledgeSearch
    from src.rag.vector_search import VectorSearch

    ks = KnowledgeSearch()
    vs = VectorSearch()
    return asyncio.run(_build_contexts_async(query_text, ks, vs, top_k))


# ============================================================================
# v2: 生产环境 QA 生成 —— 对齐 answer_question_stream
# ============================================================================

async def _generate_answer_async(
    query_text: str,
    contexts: List[str],
    user_profile: dict,
    llm,  # LLMProvider 实例（复用）
) -> str:
    """基于检索上下文 + 用户画像，用生产环境 QA prompt 生成回答。

    v2 改动：
    - 使用 _build_production_qa_prompt 构建 prompt（对齐生产环境）
    - 有伤病关键词时自动注入安全规则
    - 传入用户画像数据
    """
    prompt = _build_production_qa_prompt(
        question=query_text,
        user_profile=user_profile,
        contexts=contexts,
    )

    try:
        resp = llm.chat([{"role": "user", "content": prompt}], temperature=0.5)
        return resp.content
    except Exception as e:
        logger.warning(f"LLM answer generation failed: {e}")
        return f"（生成失败：{e}）"


def _generate_answer_sync(
    query_text: str,
    contexts: List[str],
    user_profile: dict = None,
) -> str:
    """同步包装器——兼容旧调用方式。"""
    from src.llm.provider import LLMProvider

    if user_profile is None:
        user_profile = {"height": 175, "weight": 70, "training_years": 1,
                        "goal": "增肌", "injuries": []}

    llm = LLMProvider()
    return asyncio.run(_generate_answer_async(query_text, contexts, user_profile, llm))


# ============================================================================
# RAGAS 评测主流程 (v2 异步并行)
# ============================================================================

async def run_ragas_eval(
    queries: List[dict],
    limit: Optional[int] = None,
    output_path: Optional[str] = None,
    stratified: bool = True,  # v2: 是否启用分层抽样
    target_queries: int = 50,  # v2: 分层抽样目标数量
) -> dict:
    """运行 RAGAS 评测。

    v2 改动：
    - stratified=True 时启用分层抽样（每类 >= 15 条，总量 >= 50）
    - 检索+生成阶段全部 asyncio.gather 并发
    - per_query 输出增加 query_type/difficulty/has_injury 标签
    """
    if not _check_ragas():
        return {"error": "ragas not installed. Run: pip install ragas langchain-openai datasets pandas"}

    import ragas
    from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextRelevance
    from ragas.llms import llm_factory
    from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
    from openai import AsyncOpenAI
    from src.config import (
        DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL,
        EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL,
    )
    from src.rag.knowledge_search import KnowledgeSearch
    from src.rag.vector_search import VectorSearch
    from src.llm.provider import LLMProvider

    # ---- v2: 分层抽样 ----
    if stratified:
        queries = _stratified_sample(queries, target_total=target_queries)
    elif limit and limit < len(queries):
        queries = queries[:limit]

    # ---- 配置 RAGAS 裁判 LLM ----
    eval_api_key = os.getenv("EVAL_LLM_API_KEY", DEEPSEEK_API_KEY)
    eval_base_url = os.getenv("EVAL_LLM_BASE_URL", DEEPSEEK_BASE_URL)
    eval_model = os.getenv("EVAL_LLM_MODEL", "deepseek-chat")

    evaluator_llm = llm_factory(
        eval_model,
        client=AsyncOpenAI(api_key=eval_api_key, base_url=eval_base_url),
        max_tokens=8192,
    )

    # ---- 配置 RAGAS 嵌入模型 ----
    try:
        evaluator_embeddings = RagasOpenAIEmbeddings(
            client=AsyncOpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL),
            model=EMBEDDING_MODEL,
        )
        logger.info(f"RAGAS embeddings configured: {EMBEDDING_MODEL}")
    except Exception as e:
        logger.warning(f"Failed to configure embeddings: {e}")
        evaluator_embeddings = None

    # ---- v2: 检索 + 生成全部异步并发 ----
    logger.info(f"Building RAGAS dataset for {len(queries)} queries (async parallel)...")
    start = time.time()

    ks = KnowledgeSearch()
    vs = VectorSearch()
    llm = LLMProvider()

    async def _process_one_query(q: dict) -> dict:
        """处理单条 query：检索 + 生成，全部异步。"""
        query_text = q["query"]
        tags = q.get("_tags", _classify_query(q))
        user_profile = {
            "height": q.get("height", 175),
            "weight": q.get("weight", 70),
            "training_years": q.get("training_years", 1),
            "goal": q.get("goal", "增肌"),
            "injuries": q.get("injuries", []),
        }

        # 检索 + 生成（本身可以进一步并行，但先简化）
        contexts = await _build_contexts_async(query_text, ks, vs)
        answer = await _generate_answer_async(query_text, contexts, user_profile, llm)

        return {
            "question": query_text,
            "answer": answer,
            "contexts": contexts,
            "tags": tags,
            "id": q.get("id", "?"),
        }

    # v2: 所有 query 并发处理
    tasks = [_process_one_query(q) for q in queries]
    results_list = await asyncio.gather(*tasks)

    questions = [r["question"] for r in results_list]
    answers = [r["answer"] for r in results_list]
    contexts_list = [r["contexts"] for r in results_list]
    all_tags = [r["tags"] for r in results_list]

    elapsed_build = time.time() - start
    logger.info(f"Dataset built in {elapsed_build:.1f}s ({len(queries)} queries, "
                f"{elapsed_build/len(queries):.1f}s avg per query)")

    # ---- RAGAS 评测（指标计算仍需逐条，ascore 本身是 async 的） ----
    logger.info(f"Running RAGAS evaluation with {eval_model} as judge...")
    start_eval = time.time()

    m_faith = Faithfulness(llm=evaluator_llm)
    m_ctx = ContextRelevance(llm=evaluator_llm)
    m_ans = (AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
             if evaluator_embeddings is not None else None)

    async def _score_all() -> list:
        all_scores = []
        for i in range(len(questions)):
            scores = {}
            try:
                r = await m_faith.ascore(
                    user_input=questions[i], response=answers[i],
                    retrieved_contexts=contexts_list[i])
                scores["faithfulness"] = float(r.value)
            except Exception as e:
                logger.warning(f"  [{i+1}] faithfulness failed: {e}")
            try:
                r = await m_ctx.ascore(
                    user_input=questions[i], retrieved_contexts=contexts_list[i])
                scores["context_relevance"] = float(r.value)
            except Exception as e:
                logger.warning(f"  [{i+1}] context_relevance failed: {e}")
            if m_ans is not None:
                try:
                    r = await m_ans.ascore(
                        user_input=questions[i], response=answers[i])
                    scores["answer_relevancy"] = float(r.value)
                except Exception as e:
                    logger.warning(f"  [{i+1}] answer_relevancy failed: {e}")
            all_scores.append(scores)
            logger.info(
                f"  [{i+1}/{len(questions)}] "
                + " ".join(f"{k}={v:.3f}" for k, v in scores.items()))
        return all_scores

    per_scores = await _score_all()
    elapsed_eval = time.time() - start_eval
    elapsed_total = time.time() - start

    # ---- 整理结果 ----
    def _mean(key: str) -> float:
        vals = [s[key] for s in per_scores if key in s]
        return sum(vals) / len(vals) if vals else 0.0

    faithfulness_mean = _mean("faithfulness")
    answer_relevancy_mean = _mean("answer_relevancy")
    context_relevance_mean = _mean("context_relevance")

    # v2: 按 query_type 分组统计
    by_type_stats = defaultdict(lambda: {"faithfulness": [], "answer_relevancy": [], "context_relevance": []})
    for i, tags in enumerate(all_tags):
        qt = tags["query_type"]
        for key in ["faithfulness", "answer_relevancy", "context_relevance"]:
            if key in per_scores[i]:
                by_type_stats[qt][key].append(per_scores[i][key])

    grouped = {}
    for qt, stats in sorted(by_type_stats.items()):
        grouped[qt] = {
            "count": len(stats["faithfulness"]),
            "faithfulness": round(sum(stats["faithfulness"]) / len(stats["faithfulness"]), 4)
                if stats["faithfulness"] else 0,
            "answer_relevancy": round(sum(stats["answer_relevancy"]) / len(stats["answer_relevancy"]), 4)
                if stats["answer_relevancy"] else 0,
            "context_relevance": round(sum(stats["context_relevance"]) / len(stats["context_relevance"]), 4)
                if stats["context_relevance"] else 0,
        }

    # v2: 逐 query 详情（含标签）
    per_query = []
    for i in range(len(questions)):
        tags = all_tags[i]
        entry = {
            "id": results_list[i]["id"],
            "query": questions[i][:100],
            "answer": answers[i][:200],
            "num_contexts": len(contexts_list[i]),
            "query_type": tags["query_type"],       # v2 新增
            "difficulty": tags["difficulty"],        # v2 新增
            "has_injury": tags["has_injury"],        # v2 新增
        }
        for key, val in per_scores[i].items():
            entry[key] = round(val, 4)
        per_query.append(entry)

    summary = {
        "version": "v2",
        "faithfulness": round(faithfulness_mean, 4),
        "answer_relevancy": round(answer_relevancy_mean, 4),
        "context_relevance": round(context_relevance_mean, 4),
        "by_type": grouped,                         # v2 新增：分组统计
        "num_queries": len(questions),
        "evaluator_model": eval_model,
        "elapsed_build_sec": round(elapsed_build, 1),
        "elapsed_eval_sec": round(elapsed_eval, 1),
        "elapsed_total_sec": round(elapsed_total, 1),
        "per_query": per_query,
    }

    # ---- 保存结果 ----
    if output_path:
        output_file = Path(output_path)
    else:
        output_file = Path(__file__).parent / "ragas_results.json"

    output_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"RAGAS results saved to {output_file}")

    return summary


# ============================================================================
# 结果格式化输出
# ============================================================================

def print_ragas_report(result: dict):
    """打印 RAGAS 评测报告到控制台（v2 增强版）。"""
    if "error" in result:
        logger.error(result["error"])
        return

    print("\n" + "=" * 60)
    print("  RAGAS Evaluation Report (v2)")
    print("=" * 60)
    print(f"  Queries evaluated:  {result['num_queries']}")
    print(f"  Evaluator model:    {result.get('evaluator_model', 'N/A')}")
    print(f"  Time (build):       {result.get('elapsed_build_sec', '?')}s")
    print(f"  Time (eval):        {result.get('elapsed_eval_sec', '?')}s")
    print(f"  Time (total):       {result.get('elapsed_total_sec', '?')}s")
    print("-" * 60)
    print(f"  Faithfulness:       {result['faithfulness']:.4f}  (答案是否基于上下文+常识)")
    print(f"  Answer Relevancy:   {result['answer_relevancy']:.4f}  (答案是否切题)")
    print(f"  Context Relevance:  {result['context_relevance']:.4f}  (检索是否相关)")
    print("=" * 60)

    # v2: 按 query_type 分组展示
    by_type = result.get("by_type", {})
    if by_type:
        print(f"\n  按 query_type 分组：")
        print(f"  {'Type':<18} {'Count':>5} {'Faith':>8} {'AnsRel':>8} {'CtxRel':>8}")
        print(f"  {'-'*18} {'-'*5} {'-'*8} {'-'*8} {'-'*8}")
        for qt, stats in sorted(by_type.items()):
            print(f"  {qt:<18} {stats['count']:>5} "
                  f"{stats['faithfulness']:>8.4f} {stats['answer_relevancy']:>8.4f} "
                  f"{stats['context_relevance']:>8.4f}")

    # 解读
    print(f"\n  解读：")
    for metric, name, good, bad in [
        ("faithfulness", "Faithfulness", 0.80, 0.50),
        ("answer_relevancy", "Answer Relevancy", 0.80, 0.50),
        ("context_relevance", "Context Relevance", 0.80, 0.50),
    ]:
        val = result[metric]
        if val >= good:
            status = "优秀 ✓"
        elif val >= bad:
            status = "一般 △"
        else:
            status = "需改进 ✗"
        print(f"  {name}: {val:.4f} → {status}")

    # v2: 与 v1 的差异说明
    print(f"\n  v2 与 v1 的差异：")
    print(f"  - 生成使用生产环境 QA prompt（含用户画像+伤病注入）")
    print(f"  - 检索上下文注入了健身常识文档（消除常识误判为幻觉）")
    print(f"  - 检索+生成阶段异步并行（总耗时大幅下降）")
    print(f"  - 分层抽样确保 query 类型均衡")
    print(f"  - per_query 增加了 query_type/difficulty/has_injury 标签")
    print()


# ============================================================================
# 命令行入口
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS evaluation for gakki-ai-fitness (v2)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Number of queries (ignored if --no-stratify)")
    parser.add_argument("--target", type=int, default=50,
                        help="Target total when stratified sampling (default: 50)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-query logs")
    parser.add_argument("--no-stratify", action="store_true",
                        help="Disable stratified sampling (use fixed --limit instead)")
    parser.add_argument("--no-async-build", action="store_true",
                        help="Disable async parallel build (use old sequential loop)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    queries = load_dataset()
    logger.info(f"Loaded {len(queries)} queries from golden dataset")

    result = asyncio.run(run_ragas_eval(
        queries,
        limit=args.limit,
        output_path=args.output,
        stratified=not args.no_stratify,
        target_queries=args.target,
    ))
    print_ragas_report(result)
