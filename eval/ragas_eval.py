"""
=============================================================================
ragas_eval.py — RAGAS 社区标准评测集成
=============================================================================
【项目角色】
    这是 RAG 评测体系的第三维度，引入 RAGAS（RAG Assessment）框架，
    从"生成质量"角度评估 RAG pipeline，而非仅看检索指标。

【评测维度】
    RAGAS 三个核心指标：
    1. Faithfulness（忠实度）     — 答案是否基于检索到的文档？有没有编造？
    2. Answer Relevancy（答案相关性）— 答案是否紧扣用户问题？有没有跑题？
    3. Context Relevancy（上下文相关性）— 检索到的文档和问题相关吗？

【与现有评测的关系】
    现有评测                      RAGAS 补充
    ─────────                    ──────────
    P@K / R@K / MRR / NDCG        Context Relevancy（更细粒度的检索质量）
    LLM-as-Judge 端到端 4.90/5    Faithfulness + Answer Relevancy（拆解开）
    FactChecker FNR=0.0%          Faithfulness（答案是否忠于上下文）
    HN-Recall@5                   三者互补，各覆盖不同盲区

    核心区别：Golden Dataset 依赖人工标注，RAGAS 用 LLM 自动打分，
    适合快速迭代验证、CI/CD 集成。

【用法】
    python eval/ragas_eval.py                    # 用默认 20 条跑全量
    python eval/ragas_eval.py --limit 10         # 只测 10 条
    python eval/ragas_eval.py --output report.md # 指定输出路径

    也可通过 run_eval.py 的 --ragas 标记运行：
    python eval/run_eval.py --ragas
=============================================================================
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from eval.golden_dataset.build_dataset import load_dataset

logger = logging.getLogger(__name__)

# ============================================================================
# RAGAS 依赖检查
# ============================================================================

def _check_ragas() -> bool:
    """检查 ragas 及相关依赖是否可用。"""
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
# 构建 RAG 上下文 + 答案
# ============================================================================

def _build_contexts_for_query(query: str, top_k: int = 5) -> List[str]:
    """为单个查询构建检索上下文（知识库 + 动作库）。

    RAGAS 的 Context Relevancy 需要传入 contexts（字符串列表），
    这里用 KnowledgeSearch + VectorSearch 双路检索后合并。
    """
    from src.rag.knowledge_search import KnowledgeSearch
    from src.rag.vector_search import VectorSearch

    ks = KnowledgeSearch()
    vs = VectorSearch()

    contexts = []

    # 知识库检索（含 RRF + LLM 重排序，质量最高）
    try:
        knowledge_results = ks.search(query, top_k=top_k)
        for r in knowledge_results:
            snippet = r.get("content", "")[:500]  # 截断长文本，RAGAS 不依赖全文
            if snippet:
                contexts.append(snippet)
    except Exception as e:
        logger.warning(f"Knowledge search failed for query: {e}")

    # 动作库向量检索
    try:
        exercise_results = vs.search(query, top_k=top_k)
        for r in exercise_results:
            name = r.get("name", "")
            desc = r.get("description", "")
            muscles = r.get("target_muscles", "")
            snippet = f"动作：{name}。{desc} 目标肌群：{muscles}"
            if snippet:
                contexts.append(snippet)
    except Exception as e:
        logger.warning(f"Vector search failed for query: {e}")

    # 去重（基于前 100 字符的模糊去重）
    seen = set()
    unique = []
    for ctx in contexts:
        key = ctx[:100]
        if key not in seen:
            seen.add(key)
            unique.append(ctx)
    return unique[:top_k]


def _generate_answer(query: str, contexts: List[str]) -> str:
    """基于检索到的上下文，用 LLM 生成回答。

    注意：这里不走完整的 Orchestrator pipeline（Planner→Writer→FactChecker），
    而是用轻量级的 prompt 直接生成。原因是：
    1. RAGAS 主要评估 RAG 部分（检索→生成），不评估任务规划和安全审查
    2. 轻量级调用更快（20 条 × 完整 pipeline = 太多 LLM 调用）
    3. WriterAgent 生成训练计划 JSON，而 RAGAS 期望自然语言回答
    """
    from src.llm.provider import LLMProvider

    llm = LLMProvider()

    context_text = "\n\n---\n\n".join(contexts) if contexts else "（无检索结果）"
    prompt = f"""你是一个健身教练助手。基于以下检索到的健身知识，回答用户的问题。
只使用下面提供的信息，不要编造任何不在资料中的内容。

【检索到的资料】
{context_text}

【用户问题】
{query}

请用中文简洁回答（100-300字），只基于上面提供的资料。"""

    try:
        resp = llm.chat([{"role": "user", "content": prompt}])
        return resp.content
    except Exception as e:
        logger.warning(f"LLM answer generation failed: {e}")
        return f"（生成失败：{e}）"


# ============================================================================
# RAGAS 评测主流程
# ============================================================================

def run_ragas_eval(
    queries: List[dict],
    limit: Optional[int] = None,
    output_path: Optional[str] = None,
) -> dict:
    """运行 RAGAS 评测，返回 Faithfulness / Answer Relevancy / Context Relevancy。

    输入：
        queries:     list[dict] — 黄金数据集查询
        limit:       int|None  — 限制评测数量（None=全部，建议 20 以内）
        output_path: str|None  — JSON 结果输出路径

    输出：
        dict — {"faithfulness": float, "answer_relevancy": float,
                "context_relevancy": float, "per_query": [...], "elapsed_sec": float}

    流程：
        1. 抽样 N 条查询
        2. 对每条：检索 contexts → LLM 生成 answer
        3. 构造 RAGAS EvaluationDataset
        4. 用 DeepSeek 当裁判 LLM 计算三个指标
        5. 保存结果
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

    # ---- 评估模式与生产模式用不同的 LLM ----
    # 如果 .env 中设置了 EVAL_LLM 系列变量则用评估 LLM，否则 fallback 到生产 LLM
    eval_api_key = os.getenv("EVAL_LLM_API_KEY", DEEPSEEK_API_KEY)
    eval_base_url = os.getenv("EVAL_LLM_BASE_URL", DEEPSEEK_BASE_URL)
    eval_model = os.getenv("EVAL_LLM_MODEL", "deepseek-chat")

    if limit and limit < len(queries):
        queries = queries[:limit]

    # ---- 配置 RAGAS 裁判 LLM（OpenAI 兼容，ragas 0.4.x 要求 llm_factory/InstructorLLM）----
    # collections 指标的 ascore() 走异步路径，必须传 AsyncOpenAI 客户端
    evaluator_llm = llm_factory(
        eval_model,
        client=AsyncOpenAI(api_key=eval_api_key, base_url=eval_base_url),
        max_tokens=8192,  # Faithfulness 语句分解+逐条判定的输出较长，默认上限会截断
    )

    # ---- 配置 RAGAS 嵌入模型（DashScope OpenAI 兼容 API，与生产链路同款）----
    # AnswerRelevancy 需要 embeddings 来比较生成问题与原问题的语义相似度
    try:
        evaluator_embeddings = RagasOpenAIEmbeddings(
            client=AsyncOpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL),
            model=EMBEDDING_MODEL,
        )
        logger.info(f"RAGAS embeddings configured: {EMBEDDING_MODEL}")
    except Exception as e:
        logger.warning(f"Failed to configure embeddings {EMBEDDING_MODEL}: {e}")
        logger.info("AnswerRelevancy will be skipped (requires embeddings)")
        evaluator_embeddings = None

    # ---- 步骤 1：为每条查询构建 (question, answer, contexts) 三元组 ----
    logger.info(f"Building RAGAS dataset for {len(queries)} queries...")
    questions = []
    answers = []
    contexts_list = []

    for i, q in enumerate(queries):
        query_text = q["query"]
        logger.info(f"  [{i+1}/{len(queries)}] Processing: {query_text[:60]}...")

        # 检索上下文
        contexts = _build_contexts_for_query(query_text)
        # 基于上下文生成回答
        answer = _generate_answer(query_text, contexts)

        questions.append(query_text)
        answers.append(answer)
        contexts_list.append(contexts)

    # ---- 步骤 2/3：运行 RAGAS 评测（ragas 0.4.x collections API：逐样本 score）----
    # 0.4.x 的 collections 指标不再兼容旧 evaluate() 入口，
    # 每个指标直接提供 score(user_input=..., ...) -> MetricResult，天然支持逐 query 明细
    logger.info(f"Running RAGAS evaluation with {eval_model} as judge...")
    start = time.time()

    m_faith = Faithfulness(llm=evaluator_llm)        # 答案是否忠于上下文
    m_ctx = ContextRelevance(llm=evaluator_llm)      # 检索的上下文是否相关
    m_ans = (AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
             if evaluator_embeddings is not None else None)  # 需要 embeddings

    async def _score_all() -> list:
        all_scores = []
        for i, (q_text, ans, ctxs) in enumerate(zip(questions, answers, contexts_list), 1):
            scores = {}
            try:
                r = await m_faith.ascore(
                    user_input=q_text, response=ans, retrieved_contexts=ctxs)
                scores["faithfulness"] = float(r.value)
            except Exception as e:
                logger.warning(f"  [{i}] faithfulness failed: {e}")
            try:
                r = await m_ctx.ascore(user_input=q_text, retrieved_contexts=ctxs)
                scores["context_relevance"] = float(r.value)
            except Exception as e:
                logger.warning(f"  [{i}] context_relevance failed: {e}")
            if m_ans is not None:
                try:
                    r = await m_ans.ascore(user_input=q_text, response=ans)
                    scores["answer_relevancy"] = float(r.value)
                except Exception as e:
                    logger.warning(f"  [{i}] answer_relevancy failed: {e}")
            all_scores.append(scores)
            logger.info(f"  [{i}/{len(questions)}] " +
                        " ".join(f"{k}={v:.3f}" for k, v in scores.items()))
        return all_scores

    import asyncio
    per_scores = asyncio.run(_score_all())
    elapsed = time.time() - start

    # ---- 步骤 4：整理结果 ----
    def _mean(key: str) -> float:
        vals = [s[key] for s in per_scores if key in s]
        return sum(vals) / len(vals) if vals else 0.0

    faithfulness_mean = _mean("faithfulness")
    answer_relevancy_mean = _mean("answer_relevancy")
    context_relevance_mean = _mean("context_relevance")

    # 逐 query 详情
    per_query = []
    for i, q in enumerate(queries):
        entry = {
            "id": q["id"],
            "query": questions[i][:100],
            "answer": answers[i][:200],
            "num_contexts": len(contexts_list[i]),
        }
        for key, val in per_scores[i].items():
            entry[key] = round(val, 4)
        per_query.append(entry)

    summary = {
        "faithfulness": round(faithfulness_mean, 4),
        "answer_relevancy": round(answer_relevancy_mean, 4),
        "context_relevance": round(context_relevance_mean, 4),
        "num_queries": len(queries),
        "evaluator_model": eval_model,
        "elapsed_sec": round(elapsed, 1),
        "per_query": per_query,
    }

    # ---- 步骤 5：保存结果 ----
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
    """打印 RAGAS 评测报告到控制台。"""
    if "error" in result:
        logger.error(result["error"])
        return

    print("\n" + "=" * 60)
    print("  RAGAS Evaluation Report")
    print("=" * 60)
    print(f"  Queries evaluated:  {result['num_queries']}")
    print(f"  Evaluator model:    {result.get('evaluator_model', 'N/A')}")
    print(f"  Time elapsed:       {result['elapsed_sec']}s")
    print("-" * 60)
    print(f"  Faithfulness:       {result['faithfulness']:.4f}  (答案是否基于上下文，0=编造 1=忠实)")
    print(f"  Answer Relevancy:   {result['answer_relevancy']:.4f}  (答案是否切题，0=跑题 1=精准)")
    print(f"  Context Relevance:  {result['context_relevance']:.4f}  (检索是否相关，0=无关 1=全部相关)")
    print("=" * 60)

    # 解读
    print("\n  解读：")
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

    print(f"\n  与现有评测的关系：")
    print(f"  - Golden Dataset P@5/R@5 测的是'检索对不对'")
    print(f"  - Context Relevance 用 LLM 自动测同一件事（无需人工标注）")
    print(f"  - Faithfulness 测的是'答案有没有编造'（FactChecker 关注的是安全，不是编造）")
    print(f"  - Answer Relevancy 测的是'答案有没有跑题'（LLM-as-Judge 4.90 的综合拆解）")
    print()


# ============================================================================
# 命令行入口
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS evaluation for gakki-ai-fitness")
    parser.add_argument("--limit", type=int, default=20, help="Number of queries (default: 20)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-query logs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    queries = load_dataset()
    logger.info(f"Loaded {len(queries)} queries, using {args.limit} for RAGAS eval")

    result = run_ragas_eval(queries, limit=args.limit, output_path=args.output)
    print_ragas_report(result)
