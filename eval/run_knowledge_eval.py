"""
=============================================================================
run_knowledge_eval.py —— 知识库检索消融评测（基于 LLM 生成的 54 道题）
=============================================================================

评测设计：
  A (Baseline)  = 纯向量检索 knowledge_chunks（PG cosine similarity）
  B (+Keyword) = 向量 + 关键词 trigram 双路 → RRF 融合
  C (Full)      = B + LLM 重排序（完整 KnowledgeSearch 流水线）

消融逻辑：
  A→B 的增益量化了关键词检索的贡献（解决"同义词/近义词"语义搜索盲区）
  B→C 的增益量化了 LLM 重排序的贡献（过滤"看起来相关但不相关"的片段）

用法：
  python eval/run_knowledge_eval.py --all
  python eval/run_knowledge_eval.py --ablation A  # 单组测试
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.storage.pg import PGClient
from src.rag.embedding import EmbeddingService
from src.rag.knowledge_search import KnowledgeSearch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

QUERIES_FILE = Path(__file__).parent / "golden_dataset" / "knowledge_queries.json"
RESULTS_FILE = Path(__file__).parent / "knowledge_results.json"
REPORT_FILE = Path(__file__).parent / "KNOWLEDGE_EVAL_REPORT.md"
KS = [1, 3, 5, 10]


def load_queries():
    return json.loads(QUERIES_FILE.read_text(encoding="utf-8"))


# =========================================================================
# 消融组 A：纯向量检索 knowledge_chunks
# =========================================================================
def _make_vector_only_knowledge():
    pg = PGClient()
    emb = EmbeddingService()

    def run(query):
        vec = emb.embed(query)
        vec_str = f"[{','.join(str(v) for v in vec)}]"
        sql = f"""
            SELECT title, content,
                   1 - (embedding <=> '{vec_str}'::vector) AS similarity
            FROM knowledge_chunks
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> '{vec_str}'::vector
            LIMIT 10
        """
        rows = pg.fetch_all(sql, {})
        return [{"name": r[0], "content": r[1], "score": float(r[2])} for r in rows]
    return run


# =========================================================================
# 消融组 B：向量 + 关键词 trigram → RRF 融合（去掉 LLM 重排序）
# =========================================================================
def _make_rrf_knowledge():
    pg = PGClient()
    emb = EmbeddingService()

    def run(query):
        # 1. 向量检索 Top-20
        vec = emb.embed(query)
        vec_str = f"[{','.join(str(v) for v in vec)}]"
        vec_sql = f"""
            SELECT title, content,
                   1 - (embedding <=> '{vec_str}'::vector) AS score
            FROM knowledge_chunks
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> '{vec_str}'::vector
            LIMIT 20
        """
        vec_rows = pg.fetch_all(vec_sql, {})

        # 2. 关键词检索 Top-20（pg_trgm）
        kw_sql = """
            SELECT title, content,
                   similarity(title, :query) AS score
            FROM knowledge_chunks
            WHERE title %% :query OR title ILIKE '%' || :query || '%'
            ORDER BY score DESC
            LIMIT 20
        """
        try:
            kw_rows = pg.fetch_all(kw_sql, {"query": query})
        except Exception:
            kw_rows = []

        # 3. RRF 融合
        rrf_scores = {}
        for rank, row in enumerate(vec_rows, 1):
            title = row[0]
            rrf_scores[title] = rrf_scores.get(title, 0) + 1.0 / (60 + rank)
        for rank, row in enumerate(kw_rows, 1):
            title = row[0]
            rrf_scores[title] = rrf_scores.get(title, 0) + 1.0 / (60 + rank)

        # 4. 按 RRF 分数排序取 Top-10
        seen = set()
        merged = []
        for row in vec_rows + kw_rows:
            title = row[0]
            if title not in seen:
                seen.add(title)
                merged.append({
                    "name": row[0],
                    "content": row[1],
                    "score": rrf_scores.get(title, 0),
                })
        merged.sort(key=lambda x: x["score"], reverse=True)
        return merged[:10]
    return run


# =========================================================================
# 消融组 C：完整 KnowledgeSearch（向量 + 关键词 + LLM 重排序）
# =========================================================================
def _make_full_knowledge():
    ks = KnowledgeSearch()

    def run(query):
        try:
            results = ks.search(query, top_k=10)
        except Exception:
            results = []
        return [{"name": r.get("title", ""), "content": r.get("content", ""),
                 "score": r.get("rerank_score", r.get("rrf_score", 0))}
                for r in results]
    return run


# =========================================================================
# 评测入口
# =========================================================================
def run_ablation(queries, group="all"):
    runners = {}
    if group in ("A", "all"):
        runners["A-VectorOnly"] = _make_vector_only_knowledge()
    if group in ("B", "all"):
        runners["B-RRF"] = _make_rrf_knowledge()
    if group in ("C", "all"):
        runners["C-Full"] = _make_full_knowledge()

    results = {}
    for name, runner in runners.items():
        logger.info(f"Running: {name} ({len(queries)} queries)...")
        start = time.time()

        # 逐 query 评测
        per_query = []
        for q in queries:
            try:
                retrieved = runner(q["query"])
                retrieved_names = [r["name"] for r in retrieved]
                relevant = set(q.get("relevant_doc_ids", []))
                # 用 source_title 也做匹配（有些题目标的是完整标题）
                if q.get("source_title"):
                    relevant.add(q["source_title"])

                # 计算逐 query 指标
                pq = {
                    "id": q["id"],
                    "query": q["query"],
                    "retrieved": retrieved_names,
                    "relevant": list(relevant),
                }
                for k in KS:
                    top_k_names = retrieved_names[:k]
                    hits = sum(1 for n in top_k_names for rel in relevant
                               if rel in n or n in rel)  # 模糊匹配
                    pq[f"precision@{k}"] = min(hits, k) / k if k > 0 else 0
                    pq[f"recall@{k}"] = hits / len(relevant) if relevant else 0
                    # NDCG simplified: relevance = 1 if hit
                    dcg = sum(1.0 / (i + 2) for i, n in enumerate(top_k_names)
                              if any(rel in n or n in rel for rel in relevant))
                    idcg = sum(1.0 / (i + 2) for i in range(min(len(relevant), k)))
                    pq[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0

                # MRR
                mrr = 0
                for i, n in enumerate(retrieved_names):
                    if any(rel in n or n in rel for rel in relevant):
                        mrr = 1.0 / (i + 1)
                        break
                pq["mrr"] = mrr
                per_query.append(pq)

            except Exception as e:
                per_query.append({"id": q["id"], "error": str(e)})

        elapsed = time.time() - start

        # 汇总平均
        valid = [pq for pq in per_query if "error" not in pq]
        averages = {}
        if valid:
            for metric in [f"precision@{k}" for k in KS] + [f"recall@{k}" for k in KS] + \
                          [f"ndcg@{k}" for k in KS] + ["mrr"]:
                averages[metric] = sum(pq[metric] for pq in valid) / len(valid)

        results[name] = {
            "num_queries": len(queries),
            "num_successful": len(valid),
            "averages": averages,
            "per_query": per_query,
            "elapsed_sec": round(elapsed, 1),
        }

        if averages:
            logger.info(
                f"  {name}: P@5={averages.get('precision@5', 0):.3f}, "
                f"R@5={averages.get('recall@5', 0):.3f}, "
                f"MRR={averages.get('mrr', 0):.3f}, "
                f"NDCG@5={averages.get('ndcg@5', 0):.3f}"
            )

    return results


def generate_report(ablation_results, num_queries, output_path):
    lines = []
    lines.append("# 知识库检索 — 消融评测报告")
    lines.append(f"\n**生成时间**: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**评测数据集**: {num_queries} 条 LLM 生成的知识型问题")
    lines.append(f"**评测 K 值**: {', '.join(map(str, KS))}")
    lines.append(f"**题目来源**: 基于 18 篇知识文档由 LLM 自动生成，人工审核验证\n")
    lines.append("---\n")
    lines.append("## 1. 消融实验设计\n")
    lines.append("| 组 | 配置 | 描述 |")
    lines.append("|----|------|------|")
    lines.append("| A (Baseline) | 纯向量检索 | PG vector cosine similarity on knowledge_chunks |")
    lines.append("| B | +关键词 RRF | 向量 + pg_trgm 关键词双路 → RRF 融合 |")
    lines.append("| C (Full) | +LLM 重排序 | B + LLM 对 Top-20 精排打分取 Top-5 |\n")
    lines.append("**消融逻辑**:")
    lines.append("- A→B 量化关键词检索对语义搜索盲区的补偿")
    lines.append("- B→C 量化 LLM 重排序对假阳性片段的过滤能力\n")

    lines.append("---\n")
    lines.append("## 2. 综合对比（K=5）\n")
    lines.append("| 指标 | A-纯向量 | B-RRF | C-Full | 提升(A→C) |")
    lines.append("|------|----------|-------|--------|------------|")

    for metric, label in [("precision@5", "Precision@5"), ("recall@5", "Recall@5"),
                          ("mrr", "MRR"), ("ndcg@5", "NDCG@5")]:
        vals = []
        for g in ["A-VectorOnly", "B-RRF", "C-Full"]:
            gd = ablation_results.get(g, {}).get("averages", {})
            vals.append(f"{gd.get(metric, 0):.4f}")
        a_val = float(vals[0])
        c_val = float(vals[2])
        pct = f"+{(c_val - a_val) / a_val * 100:.0f}%" if a_val > 0 else "N/A"
        lines.append(f"| {label} | {vals[0]} | {vals[1]} | {vals[2]} | {pct} |")

    lines.append("\n---\n")
    lines.append("## 3. 各 K 值详细数据\n")

    for g_name in ["A-VectorOnly", "B-RRF", "C-Full"]:
        gd = ablation_results.get(g_name, {})
        lines.append(f"### {g_name}\n")
        lines.append("| K | Precision | Recall | NDCG |")
        lines.append("|---|-----------|--------|------|")
        for k in KS:
            p = gd.get("averages", {}).get(f"precision@{k}", 0)
            r = gd.get("averages", {}).get(f"recall@{k}", 0)
            n = gd.get("averages", {}).get(f"ndcg@{k}", 0)
            lines.append(f"| {k} | {p:.4f} | {r:.4f} | {n:.4f} |")
        mrr = gd.get("averages", {}).get("mrr", 0)
        lines.append(f"| MRR | — | — | {mrr:.4f} |")
        lines.append("")

    lines.append("---\n")
    lines.append("*报告由 eval/run_knowledge_eval.py 自动生成*")
    lines.append(f"*题目由 scripts/build_knowledge_eval.py 基于 18 篇知识文档通过 LLM 自动生成*")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Knowledge Retrieval Evaluation")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--ablation", nargs="?", const="all",
                        choices=["A", "B", "C", "all"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.all:
        args.ablation = "all"

    queries = load_queries()
    if args.limit:
        queries = queries[:args.limit]
    logger.info(f"Loaded {len(queries)} knowledge queries")

    results = run_ablation(queries, args.ablation or "all")

    if results:
        RESULTS_FILE.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Results saved to {RESULTS_FILE}")

    generate_report(results, len(queries), str(REPORT_FILE))
    logger.info("Knowledge evaluation complete!")


if __name__ == "__main__":
    main()
