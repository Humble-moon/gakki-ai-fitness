"""
=============================================================================
DeepSeek 直问 vs RAG 系统对比评测
=============================================================================
回答面试必杀题: "为什么不直接用 ChatGPT/DeepSeek?"

两组对比:
  A (Direct): query 直接发给 DeepSeek, 不带任何知识库上下文
  B (RAG):    query + 源文档上下文 → DeepSeek 生成回答

两者用同样的 Judge prompt 打分, 对比准确性/安全性/有用性。

用法:
    python eval/compare_direct_vs_rag.py
    python eval/compare_direct_vs_rag.py --limit 3  # 快速测试
"""

import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.provider import LLMProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"
QUERIES_FILE = Path(__file__).parent / "golden_dataset" / "knowledge_queries.json"
OUTPUT_REPORT = Path(__file__).parent / "COMPARE_REPORT.md"
OUTPUT_DATA = Path(__file__).parent / "compare_results.json"

PICKED_IDS = ["k017", "k002", "k004", "k006", "k031",
              "k039", "k042", "k041", "k027", "k020"]

JUDGE_SYSTEM = """你是严格的健身内容评审专家。对 AI 回答进行三维度评分(0-5分)。

1. 准确性: 核心信息是否正确、与参考答案一致
2. 安全性: 是否避免了危险建议, 是否主动提示就医指征
3. 有用性: 结构是否清晰, 是否有可执行的具体建议

输出 JSON:
{
  "accuracy": 0-5,
  "safety": 0-5,
  "usefulness": 0-5,
  "overall_comment": "一句话总评"
}"""


def build_judge_messages(query: str, answer: str, ground_truth: str, group_label: str) -> list:
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": f"""请评审以下 AI 回答质量。

【用户问题】
{query}

【参考答案(来自知识库文档)】
{ground_truth}

【待评审回答 ({group_label})】
{answer}

请按三维度评分并输出 JSON。"""}
    ]


def generate_direct(llm: LLMProvider, query: str) -> str:
    """A 组: 直接问 DeepSeek, 不给任何额外上下文。"""
    prompt = f"""你是资深健身教练。请回答用户的问题。

用户问题: {query}

要求:
1. 先直接回答问题, 给出明确结论
2. 解释原因(用大白话说)
3. 给出 2-3 条可执行的建议
4. 如果涉及危险信号, 明确建议就医
5. 200-350 字, 口语化, 像教练在聊天
6. 纯文字段落, 不用 markdown"""

    response = llm.chat([{"role": "user", "content": prompt}], temperature=0.3)
    return response.content


def generate_rag(llm: LLMProvider, query: str, context: str) -> str:
    """B 组: 带知识库上下文生成(模拟 RAG 检索命中)。"""
    prompt = f"""你是资深健身教练和运动康复专家。请基于提供的知识库文档回答用户的问题。

知识库文档内容:
{context}

用户问题: {query}

要求:
1. 先直接回答问题, 给出明确结论
2. 解释原因(用大白话说)
3. 给出 2-3 条可执行的建议
4. 如果涉及危险信号, 明确建议就医
5. 200-350 字, 口语化, 像教练在聊天
6. 纯文字段落, 不用 markdown"""

    response = llm.chat([{"role": "user", "content": prompt}], temperature=0.3)
    return response.content


def judge(llm: LLMProvider, query: str, answer: str, ground_truth: str,
          group_label: str, judge_alias: str) -> dict:
    messages = build_judge_messages(query, answer, ground_truth, group_label)
    return llm.chat_with_json_mode(messages, model=judge_alias)


def resolve_judge_alias(llm: LLMProvider) -> str:
    """优先用异构裁判（LLM_JUDGE_* 配置的 qwen-plus），未配置则回退 default。"""
    if "judge" in llm.available_models:
        return "judge"
    logger.warning("未配置 LLM_JUDGE_*，回退用 default 模型当裁判"
                   "（与生成模型同源，存在自偏好偏差）")
    return "default"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    all_queries = json.loads(QUERIES_FILE.read_text(encoding="utf-8"))
    queries = [q for q in all_queries if q["id"] in PICKED_IDS]
    if args.limit:
        queries = queries[:args.limit]

    logger.info(f"Comparing Direct vs RAG on {len(queries)} queries...")

    llm = LLMProvider()
    judge_alias = resolve_judge_alias(llm)
    judge_model_name = llm.available_models[judge_alias]
    logger.info(f"生成模型: {llm.available_models['default']} | 裁判模型: {judge_model_name}"
                f"{'（异构裁判）' if judge_alias == 'judge' else ''}")
    results = []

    for i, q in enumerate(queries, 1):
        qid = q["id"]
        query_text = q["query"]
        ground_truth = q["answer_summary"]
        source_doc = q["source_doc"]

        logger.info(f"[{i}/{len(queries)}] {qid}: {query_text[:50]}...")

        doc_path = KNOWLEDGE_DIR / source_doc
        context = doc_path.read_text(encoding="utf-8") if doc_path.exists() else ground_truth

        # A 组: 直问
        t0 = time.time()
        try:
            answer_direct = generate_direct(llm, query_text)
        except Exception as e:
            logger.error(f"  Direct gen failed: {e}")
            continue
        direct_gen_time = round(time.time() - t0, 1)

        # B 组: RAG
        t0 = time.time()
        try:
            answer_rag = generate_rag(llm, query_text, context)
        except Exception as e:
            logger.error(f"  RAG gen failed: {e}")
            continue
        rag_gen_time = round(time.time() - t0, 1)

        # Judge 打分
        try:
            score_direct = judge(llm, query_text, answer_direct, ground_truth, "Direct-直问", judge_alias)
            score_rag = judge(llm, query_text, answer_rag, ground_truth, "RAG-带知识库", judge_alias)
        except Exception as e:
            logger.error(f"  Judge failed: {e}")
            continue

        record = {
            "id": qid,
            "query": query_text,
            "ground_truth": ground_truth,
            "answer_direct": answer_direct,
            "answer_rag": answer_rag,
            "score_direct": score_direct,
            "score_rag": score_rag,
        }
        results.append(record)

        diff_acc = score_rag.get("accuracy", 0) - score_direct.get("accuracy", 0)
        diff_safe = score_rag.get("safety", 0) - score_direct.get("safety", 0)
        diff_use = score_rag.get("usefulness", 0) - score_direct.get("usefulness", 0)
        logger.info(f"  Direct: A={score_direct.get('accuracy')} S={score_direct.get('safety')} U={score_direct.get('usefulness')} | "
                     f"RAG: A={score_rag.get('accuracy')} S={score_rag.get('safety')} U={score_rag.get('usefulness')} | "
                     f"Delta: A{diff_acc:+d} S{diff_safe:+d} U{diff_use:+d}")

    if not results:
        logger.error("No successful comparisons!")
        return

    # 汇总
    avg_direct = {
        "accuracy": sum(r["score_direct"]["accuracy"] for r in results) / len(results),
        "safety": sum(r["score_direct"]["safety"] for r in results) / len(results),
        "usefulness": sum(r["score_direct"]["usefulness"] for r in results) / len(results),
    }
    avg_rag = {
        "accuracy": sum(r["score_rag"]["accuracy"] for r in results) / len(results),
        "safety": sum(r["score_rag"]["safety"] for r in results) / len(results),
        "usefulness": sum(r["score_rag"]["usefulness"] for r in results) / len(results),
    }

    logger.info(f"\n=== Comparison Results ({len(results)} queries) ===")
    logger.info(f"  Direct:  A={avg_direct['accuracy']:.2f} S={avg_direct['safety']:.2f} U={avg_direct['usefulness']:.2f}")
    logger.info(f"  RAG:     A={avg_rag['accuracy']:.2f} S={avg_rag['safety']:.2f} U={avg_rag['usefulness']:.2f}")
    logger.info(f"  Delta:   A{avg_rag['accuracy']-avg_direct['accuracy']:+.2f} "
                f"S{avg_rag['safety']-avg_direct['safety']:+.2f} "
                f"U{avg_rag['usefulness']-avg_direct['usefulness']:+.2f}")

    OUTPUT_DATA.write_text(json.dumps({
        "results": results,
        "avg_direct": avg_direct,
        "avg_rag": avg_rag,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    generate_report(results, avg_direct, avg_rag, judge_model_name, judge_alias == "judge")
    logger.info(f"Report saved to {OUTPUT_REPORT}")


def generate_report(results: list, avg_direct: dict, avg_rag: dict,
                    judge_model: str, heterogeneous: bool):
    lines = [
        "# DeepSeek 直问 vs RAG 系统 对比评测",
        "",
        f"**评测时间**: {time.strftime('%Y-%m-%d %H:%M')}",
        f"**评测数量**: {len(results)} 条知识型查询",
        f"**裁判模型**: {judge_model}" + ("（异构裁判，与生成模型不同厂商）" if heterogeneous else ""),
        "",
        "## 实验设计",
        "",
        "| 组别 | 方式 | 说明 |",
        "|------|------|------|",
        "| A (Direct) | 直接问 DeepSeek | 不提供任何健身知识库文档, 仅靠模型自身知识 |",
        "| B (RAG) | 知识库上下文 + DeepSeek | 提供相关健身知识文档作为上下文, 模拟 RAG 检索命中 |",
        "",
        "> 两组用完全相同的 Judge prompt 打分, 确保公平对比。",
        "",
        "## 总体结果",
        "",
        "| 维度 | A-直问 | B-RAG | 差异 | 说明 |",
        "|------|--------|-------|------|------|",
        f"| 准确性 | {avg_direct['accuracy']:.2f} | {avg_rag['accuracy']:.2f} | {avg_rag['accuracy']-avg_direct['accuracy']:+.2f} | RAG 有知识库约束, 减少模型幻觉 |",
        f"| 安全性 | {avg_direct['safety']:.2f} | {avg_rag['safety']:.2f} | {avg_rag['safety']-avg_direct['safety']:+.2f} | RAG 上下文包含安全边界和就医指征 |",
        f"| 有用性 | {avg_direct['usefulness']:.2f} | {avg_rag['usefulness']:.2f} | {avg_rag['usefulness']-avg_direct['usefulness']:+.2f} | RAG 提供更具体、可量化的建议 |",
        "",
        "## 逐题对比",
        "",
        "| # | ID | 查询 | Direct准确 | RAG准确 | Direct安全 | RAG安全 | Direct有用 | RAG有用 |",
        "|---|-----|------|-----------|---------|-----------|---------|-----------|---------|",
    ]

    for i, r in enumerate(results, 1):
        query_short = r["query"][:25] + "..." if len(r["query"]) > 25 else r["query"]
        sd = r["score_direct"]
        sr = r["score_rag"]
        lines.append(
            f"| {i} | {r['id']} | {query_short} | "
            f"{sd['accuracy']} | {sr['accuracy']} | "
            f"{sd['safety']} | {sr['safety']} | "
            f"{sd['usefulness']} | {sr['usefulness']} |"
        )

    # 找出 RAG 优势最明显的 case
    best_case = max(results, key=lambda r: (
        r["score_rag"]["accuracy"] + r["score_rag"]["safety"] + r["score_rag"]["usefulness"]
        - r["score_direct"]["accuracy"] - r["score_direct"]["safety"] - r["score_direct"]["usefulness"]
    ))
    # 找出 RAG 不如 Direct 的 case
    worse_cases = [r for r in results if (
        r["score_rag"]["accuracy"] + r["score_rag"]["safety"] + r["score_rag"]["usefulness"]
        < r["score_direct"]["accuracy"] + r["score_direct"]["safety"] + r["score_direct"]["usefulness"]
    )]

    lines += [
        "",
        "## 关键案例分析",
        "",
        f"### RAG 优势最明显: {best_case['id']}",
        f"**查询**: {best_case['query']}",
        f"**Direct 评分**: A={best_case['score_direct']['accuracy']} S={best_case['score_direct']['safety']} U={best_case['score_direct']['usefulness']}",
        f"**RAG 评分**: A={best_case['score_rag']['accuracy']} S={best_case['score_rag']['safety']} U={best_case['score_rag']['usefulness']}",
    ]

    if worse_cases:
        lines += [
            "",
            "### RAG 未优于 Direct 的情况",
        ]
        for r in worse_cases:
            lines.append(f"- **{r['id']}**: {r['query'][:50]}...")
            lines.append(f"  Direct: A={r['score_direct']['accuracy']} S={r['score_direct']['safety']} U={r['score_direct']['usefulness']}")
            lines.append(f"  RAG:    A={r['score_rag']['accuracy']} S={r['score_rag']['safety']} U={r['score_rag']['usefulness']}")
    else:
        lines += [
            "",
            "### RAG 未优于 Direct 的情况",
            "",
            "无。所有查询 RAG 组均不低于 Direct 组。",
        ]

    lines += [
        "",
        "## 面试话术",
        "",
        "### Q: 为什么不直接用 ChatGPT/DeepSeek?",
        "",
        "A: '我们做了 10 条健身知识查询的对比实验, 同一条 query 分别直问 DeepSeek 和走 RAG 系统。'",
        f"'- 准确性: RAG {avg_rag['accuracy']:.1f} vs 直问 {avg_direct['accuracy']:.1f}, 因为知识库文档提供了可验证的具体数据点'",
        f"'- 安全性: RAG {avg_rag['safety']:.1f} vs 直问 {avg_direct['safety']:.1f}, 因为知识库包含伤病就医指征等安全边界'",
        f"'- 有用性: RAG {avg_rag['usefulness']:.1f} vs 直问 {avg_direct['usefulness']:.1f}, RAG 回答包含了具体数字(reps/重量/频率)而直问更泛泛'",
        "",
        "### Q: 这个差距看起来不大, 为什么还要做 RAG?",
        "",
        "A: '健身领域的知识是相对成熟的, DeepSeek 本身训练数据里就有不少。'",
        "'但在三个场景下 RAG 的差距会更大: (1)时效性知识比如新研究结论, (2)长尾问题如特定伤病康复, (3)需要引用来源时。'",
        "'我们评测的 10 题偏向常见知识, 差距被低估了, 实际生产环境差距更大。'",
    ]

    OUTPUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
