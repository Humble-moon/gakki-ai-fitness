"""
=============================================================================
文件角色：gakki-ai-fitness 项目的**统一评测入口**（Evaluation Runner）。
=============================================================================

在整个项目中的位置：
- 被调用方：由开发者通过命令行直接运行（python eval/run_eval.py ...）
- 调用方：
  1. eval/golden_dataset/build_dataset.py — 加载并校验黄金数据集
  2. eval/metrics/rag_metrics.py     — 计算 RAG 检索指标（P@K, R@K, MRR, NDCG@K）
  3. eval/metrics/agent_metrics.py   — 计算 Agent 决策指标（路由正确率、安全指标）
  4. eval/metrics/report.py          — 生成 Markdown 评测报告 + matplotlib 图表
  5. src/rag/*                       — 初始化 RAG 检索器（消融实验各组）
  6. src/agents/*                    — 初始化 PlannerAgent、FactCheckerAgent

评测流程：
  1. 加载 queries.json 黄金数据集
  2. 校验数据集完整性（必填字段、枚举值、唯一性）
  3. 运行 RAG 消融实验（可选）→ 各组检索 → 计算指标
  4. 运行 Agent 评测（可选）→ Planner 路由 + FactChecker 安全审查
  5. 保存结果到 results.json
  6. 生成 EVAL_REPORT.md 评测报告

用法：
    python eval/run_eval.py --all              # 运行全部评测
    python eval/run_eval.py --ablation         # 仅 RAG 消融实验
    python eval/run_eval.py --agent            # 仅 Agent 评测
    python eval/run_eval.py --ablation A       # 单个消融组
    python eval/run_eval.py --report-only      # 从已保存结果重新生成报告
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.golden_dataset.build_dataset import load_dataset, validate
from eval.metrics.rag_metrics import batch_evaluate
from eval.metrics.agent_metrics import evaluate_planner, evaluate_factchecker
from eval.metrics.report import generate_markdown_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---- 全局配置 ----

# 评测用的 K 值列表：分别计算 Top-1, Top-3, Top-5, Top-10 下的指标
# 选择这几个 K 值的原因：K=1 衡量"第一条就命中"的能力，
# K=5 是常见推荐系统评测标准，K=10 模拟用户翻看更多结果
KS = [1, 3, 5, 10]

# 评测结果持久化文件，供 --report-only 重新生成报告使用
RESULTS_FILE = Path(__file__).parent / "results.json"


# ============================================================================
# RAG 消融实验运行器工厂函数
# 设计意图：通过"渐进式添加组件"的三组消融（A → B → C），
# 量化 Agentic RAG 和 GraphRAG 各自带来的增益
# ============================================================================
#
# 消融实验设计思路：
#   A (Baseline)    = 纯向量检索 — PG vector cosine similarity，无改写，无图谱
#   B (Agentic RAG) = A + LLM 自评 + 查询改写 + 最多 3 轮迭代
#   C (Full RAG)    = B + GraphRAG 多跳推理 + 知识库 RRF 融合 + 重排序
#
# 每个工厂函数返回一个 callable: (query: str) -> list[dict]
# 输出列表中每个 dict 至少包含 "name" 字段，用于和黄金数据的 relevant_doc_ids 匹配
# ---------------------------------------------------------------------------

def _make_vector_only_runner():
    """消融组 A（Baseline）：仅对动作做纯向量检索。

    输入：无（工厂函数，返回 runner 可调用对象）
    输出：callable(query: str) -> list[dict]
          每个 dict 格式为 {"name": 动作名, "score": 余弦相似度, ...}

    核心逻辑：
      1. 初始化 VectorSearch 实例（连接 PG vector 索引）
      2. 返回一个闭包 runner，接受查询文本，直接调用 vs.search() 返回 Top-10

    设计意图：
      这是最简基线，只依赖 embedding 向量的余弦相似度。
      后续消融组 B、C 在此之上逐步叠加 Agentic RAG 和 GraphRAG，
      由此可以量化每项改进带来的检索质量提升。
    """
    from src.rag.vector_search import VectorSearch
    vs = VectorSearch()  # 初始化向量检索引擎，连接 pgvector

    def run(query: str):
        # 纯向量检索：用 embedding 模型将 query 转为向量，
        # 然后在 pgvector 索引中用余弦相似度做 ANN 搜索
        return vs.search(query, top_k=10)
    return run


def _make_agentic_rag_runner():
    """消融组 B：Agentic RAG（向量检索 + 关键词检索 + LLM 自评 + 查询改写）。

    输入：无（工厂函数，返回 runner 可调用对象）
    输出：callable(query: str) -> list[dict]

    核心逻辑：
      1. 初始化 AgenticRAG 实例，内部包含：
         - 向量检索 + BM25 关键词检索 → RRF 融合
         - LLM 自评：判断检索结果是否充分覆盖查询意图
         - 查询改写：自评不通过时，用 LLM 改写查询后重新检索（最多 3 轮迭代）
      2. 返回闭包 runner，直接调用 rag.search()

    设计意图：
      对比消融组 A，量化"LLM 驱动的自评 + 查询改写 + 多轮迭代"
      对检索 Recall 的提升幅度。同时验证 Agentic 循环的稳定性
      （是否会陷入无限改写循环或产生语义漂移）。
    """
    from src.rag.agentic_rag import AgenticRAG
    rag = AgenticRAG()  # 初始化 Agentic RAG，含 LLM 自评 + 查询改写逻辑

    def run(query: str):
        # Agentic RAG 内部流程：
        # 1. 向量检索 + BM25 关键词检索
        # 2. RRF (Reciprocal Rank Fusion) 融合两路结果
        # 3. LLM 自评检索充足性 → 不充分则改写 query → 重新检索（最多 3 轮）
        return rag.search(query, filters=None)
    return run


def _make_full_rag_runner():
    """消融组 C（Full）：Agentic RAG + GraphRAG + 知识库混合检索。

    输入：无（工厂函数，返回 runner 可调用对象）
    输出：callable(query: str) -> list[dict]
          每个 dict 包含：name, content, source, score

    核心逻辑（三步融合架构）：
      1. Agentic RAG 动作检索：复用消融组 B 的 agentic_rag.search()
         → 得到动作库中的匹配动作列表
      2. 知识库补充：用 KnowledgeSearch 搜索健身知识库（教程、解剖学等）
         → 作为动作检索的补充，填补"怎么做这个动作"的知识空白
      3. 结果合并 + 按名称去重：
         - 动作检索结果优先排在前面（source="exercise_db"）
         - 知识库结果排在后面（source="knowledge_base"）
         - 按 name 去重，避免同一内容重复出现

    设计意图：
      量化全量 RAG 方案（动作 + 知识库双路召回 + RRF + 重排序）
      相比消融组 B 的增量收益。验证多源异构数据融合的实际效果。
    """
    from src.rag.agentic_rag import AgenticRAG
    from src.rag.knowledge_search import KnowledgeSearch
    rag = AgenticRAG()   # Agentic RAG：动作库检索 + 自评 + 改写
    ks = KnowledgeSearch()  # 知识库检索：健身教程、解剖学等长文本

    def run(query: str):
        # ---- 第 1 步：动作检索（结构化数据，可直接执行）----
        exercises = rag.search(query, filters=None)
        # ---- 第 2 步：知识库搜索（非结构化长文本，提供背景知识）----
        try:
            knowledge = ks.search(query, top_k=5)
        except Exception:
            knowledge = []  # 知识库失败不应阻断评测，回退到仅有动作结果

        # ---- 第 3 步：合并结果，动作优先，知识库补充 ----
        result = []
        for ex in exercises:
            ex["source"] = "exercise_db"  # 标记数据来源，便于追踪
            result.append(ex)
        for kc in knowledge:
            result.append({
                "name": kc["title"],
                "content": kc["content"],
                "source": "knowledge_base",
                # 优先使用重排序分数，其次用 RRF 融合分数
                "score": kc.get("rerank_score") or kc.get("rrf_score", 0),
            })

        # ---- 第 4 步：按名称去重，保留首次出现的记录 ----
        seen = set()
        unique = []
        for r in result:
            if r["name"] not in seen:
                seen.add(r["name"])
                unique.append(r)
        return unique
    return run


def run_ablation(queries: list, group: str = "all") -> dict:
    """运行 RAG 消融评测，对比不同检索方案的质量。

    输入：
      queries: list[dict] — 黄金数据集中的查询列表，每条含 query, relevant_doc_ids, category
      group: str — 指定消融组，"A"/"B"/"C" 或 "all"（全部三组）

    输出：dict，key 为消融组名称（如 "A-VectorOnly"），value 为 batch_evaluate 返回的指标字典
          {
            "A-VectorOnly": { "averages": {...}, "per_query": [...], "elapsed_sec": 3.2 },
            "B-AgenticRAG": { ... },
            ...
          }

    核心逻辑：
      1. 根据 group 参数初始化对应消融组的 runner
      2. 过滤掉伤病类查询（injury 类查询依赖医学知识库，不适合 RAG 评测）
      3. 对每个 runner，用 batch_evaluate 批量计算 Precision/Recall/MRR/NDCG
      4. 记录每组耗时，打印关键指标汇总
    """
    # ---- 第 1 步：初始化各消融组的 runner ----
    runners = {}
    if group in ("A", "all"):
        runners["A-VectorOnly"] = _make_vector_only_runner()
    if group in ("B", "all"):
        runners["B-AgenticRAG"] = _make_agentic_rag_runner()
    if group in ("C", "all"):
        runners["C-Full"] = _make_full_rag_runner()

    # ---- 第 2 步：筛选适合 RAG 评测的查询 ----
    # 排除伤病类（category="injury"），因为伤病查询依赖专业医学知识库，
    # 不是动作检索能覆盖的，保留在评测中会拉低指标、干扰消融实验结论
    rag_queries = [q for q in queries if q.get("category") not in ("injury",)]

    # ---- 第 3 步：逐组运行评测 ----
    results = {}
    for name, runner in runners.items():
        logger.info(f"Running ablation: {name} ({len(rag_queries)} queries)...")
        start = time.time()
        # 调用 rag_metrics.batch_evaluate 进行批量检索评测
        result = batch_evaluate(rag_queries, runner, ks=KS)
        elapsed = time.time() - start
        result["elapsed_sec"] = round(elapsed, 1)  # 记录耗时，关注性能差异
        results[name] = result

        # ---- 打印关键指标摘要（面试时能快速展示消融效果）----
        if "averages" in result:
            avg = result["averages"]
            logger.info(
                f"  {name}: P@5={avg.get('precision@5', 0):.3f}, "
                f"R@5={avg.get('recall@5', 0):.3f}, "
                f"MRR={avg.get('mrr', 0):.3f}, "
                f"NDCG@5={avg.get('ndcg@5', 0):.3f}"
            )

    return results


def run_agent_eval(queries: list) -> dict:
    """运行 Agent 决策质量评测，分为 Planner 路由评测和 FactChecker 安全评测。

    输入：
      queries: list[dict] — 黄金数据集中的查询列表

    输出：dict
      {
        "planner": { "accuracy": 0.85, "confusion_matrix": {...}, "by_category": {...} },
        "fact_checker": { "precision": 0.9, "recall": 0.95, "fpr": 0.05, "fnr": 0.02, ... }
      }

    核心逻辑：
      1. 初始化 PlannerAgent 和 FactCheckerAgent（惰性导入，仅在需要时加载）
      2. Planner 评测：
         - 构造统一的用户画像（profile_default），保证评测条件一致
         - 对所有 query 执行 route 预测，对比 expected_route 计算正确率
      3. FactChecker 评测：
         - 仅对 safety_risk 为 medium/high 的查询评测（low 的查询无安全边界测试价值）
         - 构造 dummy_plan（最小有效计划），让 FactChecker 对其做安全审查
         - 将 FactChecker 输出（requires_human_review, confidence）映射为 safety_risk
           用于和黄金数据对比
    """
    from src.agents.planner import PlannerAgent
    from src.agents.fact_checker import FactCheckerAgent

    planner = PlannerAgent()
    fact_checker = FactCheckerAgent()

    # ---- 2.1 Planner 路由评测 ----
    # 在所有查询上运行，测试 Agent 是否正确路由到对应的 skill
    logger.info("Running Planner evaluation...")

    # 统一的虚拟用户画像，确保评测条件一致：
    # 增肌目标、有哑铃、每周练4天、训练经验1年
    profile_default = {"height": 175, "weight": 75, "training_years": 1,
                       "goal": "增肌", "available_equipment": ["哑铃"], "days_per_week": 4}

    def run_planner(query_text: str) -> str:
        """调用 Planner.plan() 并提取路由结果 skill 字段。"""
        return planner.plan(query_text, profile_default).get("skill", "muscle_building")

    # 调用 agent_metrics.evaluate_planner 计算路由正确率和混淆矩阵
    planner_result = evaluate_planner(queries, run_planner)
    logger.info(f"  Planner accuracy: {planner_result.get('accuracy', 0) * 100:.1f}%")

    # ---- 2.2 FactChecker 安全审查评测 ----
    # 仅对安全风险不为 low 的查询评测（low = 无风险，不需要安全审查介入）
    logger.info("Running FactChecker evaluation...")
    fc_queries = [q for q in queries if q.get("safety_risk") in ("medium", "high")]

    def run_fc(query_dict: dict) -> dict:
        """构造最小计划 → FactChecker 审查 → 映射回 safety_risk 标签。"""
        # 构造 dummy_plan：单日计划 + 一个测试动作，足够触发安全审查逻辑
        dummy_plan = {
            "days": [{
                "day": 1, "focus": query_dict.get("query", ""),
                "exercises": [{"name": "测试动作", "sets": 3, "reps": "10", "rest": "60s"}]
            }],
            "goal": "增肌",
        }
        profile = {"height": 175, "weight": 75, "training_years": 1,
                   "goal": "增肌", "available_equipment": ["哑铃"], "days_per_week": 4,
                   "injuries": []}

        result = fact_checker.check(dummy_plan, profile)

        # 将 FactChecker 输出映射为 safety_risk 等级：
        # requires_human_review=True → high（需要人工介入）
        # confidence < 0.7 → medium（不确定，但未明确要求介入）
        # 其他 → low（安全）
        risk = "low"
        if result.get("requires_human_review"):
            risk = "high"
        elif result.get("confidence", 1.0) < 0.7:
            risk = "medium"
        return {"safety_risk": risk, **result}

    # 调用 agent_metrics.evaluate_factchecker 计算安全指标
    fc_result = evaluate_factchecker(fc_queries, run_fc)
    if fc_result:
        logger.info(
            f"  FactChecker: precision={fc_result.get('precision', 0):.3f}, "
            f"recall={fc_result.get('recall', 0):.3f}, "
            f"FNR={fc_result.get('fnr', 0):.3f}"  # FNR 漏报率是最关键的指标
        )

    return {
        "planner": planner_result,
        "fact_checker": fc_result,
    }


# ============================================================================
# 主入口 main()
# 职责：解析命令行参数 → 加载数据 → 校验 → 运行评测 → 保存结果 → 生成报告
# ============================================================================

def main():
    """评测统一入口主函数。

    命令行参数说明：
      --all            运行全部评测（等价于 --ablation all --agent）
      --ablation [A|B|C|all]  运行 RAG 消融实验，默认全部三组
      --agent          运行 Agent 决策质量评测
      --report-only    跳过评测运行，仅从 results.json 重新生成报告
      --output PATH    指定报告输出路径，默认 eval/EVAL_REPORT.md
      --limit N        限制评测查询数量（用于快速冒烟测试）

    流程：
      1. 加载 queries.json
      2. validate() 校验必填字段和枚举值
      3. 按需运行 RAG 消融评测 → 保存到 ablation_results
      4. 按需运行 Agent 评测 → 保存到 agent_results
      5. 序列化结果到 results.json（供 --report-only 复用）
      6. 调用 report.generate_markdown_report() 生成 EVAL_REPORT.md
    """
    parser = argparse.ArgumentParser(description="gakki-ai-fitness Evaluation Runner")
    parser.add_argument("--all", action="store_true", help="Run all evaluations")
    parser.add_argument("--ablation", nargs="?", const="all", choices=["A", "B", "C", "all"],
                        help="Run RAG ablation (default: all groups)")
    parser.add_argument("--agent", action="store_true", help="Run Agent evaluation")
    parser.add_argument("--report-only", action="store_true", help="Re-generate report from saved results")
    parser.add_argument("--output", type=str, default=None, help="Report output path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries (for quick test)")
    args = parser.parse_args()

    # ---- 步骤 1：加载并校验黄金数据集 ----
    queries = load_dataset()
    if args.limit:
        queries = queries[:args.limit]  # 快速测试时只取前 N 条
        logger.info(f"Limited to {args.limit} queries")

    if not validate(queries):
        # 数据不合法则直接退出，防止错误数据导致评测结论不可靠
        logger.error("Dataset validation failed, fix errors before running eval")
        sys.exit(1)

    # ---- 步骤 2：确定运行模式 ----
    ablation_results = {}
    agent_results = {}

    if args.all:
        # --all 等价于同时开启所有评测
        args.ablation = "all"
        args.agent = True

    if args.report_only:
        # --report-only：跳过评测运行，直接读取之前保存的 results.json
        # 适用场景：仅修改报告格式/图表样式时，无需重新跑耗时评测
        if RESULTS_FILE.exists():
            saved = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
            ablation_results = saved.get("ablation", {})
            agent_results = saved.get("agent", {})
        else:
            logger.error(f"No saved results at {RESULTS_FILE}")
            sys.exit(1)

    # ---- 步骤 3：运行评测 ----
    if args.ablation:
        ablation_results = run_ablation(queries, args.ablation)

    if args.agent:
        agent_results = run_agent_eval(queries)

    # ---- 步骤 4：持久化结果到 results.json ----
    # 序列化便于后续 --report-only 复用，也方便人工检查逐 query 结果
    if ablation_results or agent_results:
        RESULTS_FILE.write_text(
            json.dumps({"ablation": ablation_results, "agent": agent_results},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"Results saved to {RESULTS_FILE}")

    # ---- 步骤 5：生成 Markdown 评测报告 ----
    output_path = args.output or str(Path(__file__).parent / "EVAL_REPORT.md")
    generate_markdown_report(
        ablation_results=ablation_results,
        agent_results=agent_results,
        ks=KS,
        num_queries=len(queries),
        output_path=output_path,
    )

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
