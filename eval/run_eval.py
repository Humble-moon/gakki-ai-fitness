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


def _extract_injuries_from_query(query: str) -> list:
    """从 query 文本提取伤病关键词，返回 injury 描述列表供 FactChecker 检查。

    之前 eval 用空 injuries=[] 导致所有伤病 query 的 FactChecker 漏报——
    FactChecker 只看 profile["injuries"]，不看 query 文本。
    """
    injury_map = {
        "腰": "腰痛/腰椎问题",
        "背": "背痛/脊椎问题",
        "膝": "膝关节损伤",
        "肩": "肩关节损伤",
        "肘": "肘关节损伤",
        "腕": "腕关节损伤",
        "踝": "踝关节损伤",
        "颈": "颈椎问题",
        "疼": "训练疼痛",
        "痛": "训练疼痛",
        "伤": "运动损伤史",
        "间盘": "腰椎间盘突出",
        "腰突": "腰椎间盘突出",
        "半月板": "半月板损伤",
        "髌骨": "髌骨软化",
        "脱臼": "关节脱臼史",
        "腱鞘炎": "腱鞘炎",
        "网球肘": "网球肘",
        "肩峰撞击": "肩峰撞击综合征",
        "跟腱炎": "跟腱炎",
        "TFCC": "TFCC损伤",
        "炎症": "慢性炎症",
        "手术": "术后恢复期",
        "术后": "术后恢复期",
        "重建": "术后重建期",
    }
    found = []
    for keyword, description in injury_map.items():
        if keyword in query and description not in found:
            found.append(description)
    return found


def _extract_exercise_for_eval(query: str) -> str:
    """从 query 中提取动作名，用于构造 eval dummy plan 中的 exercise name。

    FactChecker 会根据 exercise name + profile["injuries"] 判断伤病冲突。
    如果永远传"测试动作"，FactChecker 无法发现"腰痛还做硬拉"这种危险组合。
    """
    common_exercises = [
        "杠铃深蹲", "深蹲", "杠铃硬拉", "硬拉", "杠铃卧推", "卧推",
        "杠铃推举", "推举", "杠铃划船", "杠铃弯举", "哑铃弯举",
        "哑铃卧推", "哑铃推举", "哑铃侧平举", "侧平举",
        "坐姿绳索划船", "坐姿划船", "高位下拉", "引体向上",
        "罗马尼亚硬拉", "哑铃罗马尼亚硬拉", "箭步蹲", "哑铃负重箭步蹲",
        "保加利亚分腿蹲", "臀推", "杠铃臀推", "绳索面拉", "面拉",
        "窄距杠铃卧推", "窄距卧推", "双杠臂屈伸", "臂屈伸",
        "哑铃颈后臂屈伸", "颈后臂屈伸", "绳索下压",
        "哑铃飞鸟", "哑铃前平举", "前平举", "锤式弯举",
        "高脚杯深蹲", "史密斯机深蹲", "腿举", "站姿提踵", "提踵",
        "哑铃耸肩", "耸肩", "悬垂举腿", "举腿", "平板支撑",
        "俄罗斯转体", "绳索夹胸", "直臂下压", "哑铃俯身飞鸟",
        "上斜哑铃卧推", "上斜卧推",
    ]
    for ex in common_exercises:
        if ex in query:
            return ex

    # 身体部位 → 对应动作映射（当 query 中没有明确动作名时推测）
    body_part_to_exercise = {
        "小腿": "站姿提踵",
        "腹": "平板支撑",
        "核心": "平板支撑",
        "臀": "臀推",
        "大腿": "深蹲",
        "二头": "哑铃弯举",
        "三头": "绳索下压",
        "胸": "杠铃卧推",
        "背": "杠铃划船",
        "肩": "哑铃推举",
    }
    for body_part, exercise in body_part_to_exercise.items():
        if body_part in query:
            return exercise

    return "杠铃深蹲"  # 默认用深蹲（常见动作，比"测试动作"更真实）


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
        result["elapsed_sec"] = round(elapsed, 1)

        # ---- Hard Negative Recall：衡量"排除危险结果"的能力 ----
        # 对所有带 hard_negative_ids 的查询计算 HN-Recall
        hn_queries = [q for q in rag_queries if q.get("hard_negative_ids")]
        if hn_queries:
            hn_result = _compute_hard_negative_recall(hn_queries, runner, ks=KS)
            result["hard_negative_recall"] = hn_result

        results[name] = result

        # ---- 打印关键指标摘要 ----
        if "averages" in result:
            avg = result["averages"]
            hn = result.get("hard_negative_recall", {})
            logger.info(
                f"  {name}: P@5={avg.get('precision@5', 0):.3f}, "
                f"R@5={avg.get('recall@5', 0):.3f}, "
                f"MRR={avg.get('mrr', 0):.3f}, "
                f"NDCG@5={avg.get('ndcg@5', 0):.3f}, "
                f"HN-Recall@5={hn.get('recall@5', 0):.3f}"
            )

    return results


def run_capability_test(queries: list) -> dict:
    """能力专项评测：量化 Agentic RAG 和 GraphRAG 在不同查询类型上的增量。

    输入：
      queries: list[dict] — 黄金数据集，其中带 _capability 字段的子集参与评测

    输出：dict
      {
        "semantic": {  # 模糊语义查询 → 对比 A vs B
          "num_queries": int,
          "A-VectorOnly": { "precision@5": ..., "recall@5": ..., ... },
          "B-AgenticRAG": { "precision@5": ..., "recall@5": ..., ... },
          "rewrite_gain": { "recall@5": +0.xx, ... }  # B - A 的增量
        },
        "graph": {     # 多跳推理查询 → 对比 B vs C
          "num_queries": int,
          "B-AgenticRAG": { ... },
          "C-Full": { ... },
          "graph_gain": { "recall@5": +0.xx, ... },
          "hard_negative_recall": { ... }
        }
      }

    设计意图：
      之前的消融实验 A≈B≈C，因为 60 条查询大多是直接包含动作名的关键词型查询，
      纯向量检索已经足够好。这个测试专门构造了两类"纯向量搞不定"的查询：
      - semantic：用口语/模糊表述（"胳膊练粗""倒三角"），测查询改写的价值
      - graph：需要跨实体推理（伤病×动作×器械），测知识图谱多跳推理的价值
    """
    semantic_queries = [q for q in queries if q.get("_capability") == "semantic"]
    graph_queries = [q for q in queries if q.get("_capability") == "graph"]

    result = {}

    # =====================================================================
    # 子评测 1：语义模糊查询 → Agentic RAG 改写增益（A vs B）
    # =====================================================================
    if semantic_queries:
        logger.info(f"=== Capability: Semantic ({len(semantic_queries)} queries) ===")
        runner_a = _make_vector_only_runner()
        runner_b = _make_agentic_rag_runner()

        logger.info("  Running A-VectorOnly on semantic queries...")
        start = time.time()
        result_a = batch_evaluate(semantic_queries, runner_a, ks=KS)
        result_a["elapsed_sec"] = round(time.time() - start, 1)

        logger.info("  Running B-AgenticRAG on semantic queries...")
        start = time.time()
        result_b = batch_evaluate(semantic_queries, runner_b, ks=KS)
        result_b["elapsed_sec"] = round(time.time() - start, 1)

        # 计算改写增益：B 组各指标 - A 组各指标
        rewrite_gain = {}
        if "averages" in result_a and "averages" in result_b:
            for key in result_b["averages"]:
                rewrite_gain[key] = round(
                    result_b["averages"][key] - result_a["averages"].get(key, 0), 4
                )

        result["semantic"] = {
            "num_queries": len(semantic_queries),
            "A-VectorOnly": result_a,
            "B-AgenticRAG": result_b,
            "rewrite_gain": rewrite_gain,
        }

        if "averages" in result_a:
            logger.info(
                f"  A-VectorOnly: P@5={result_a['averages'].get('precision@5',0):.3f}, "
                f"R@5={result_a['averages'].get('recall@5',0):.3f}"
            )
        if "averages" in result_b:
            logger.info(
                f"  B-AgenticRAG: P@5={result_b['averages'].get('precision@5',0):.3f}, "
                f"R@5={result_b['averages'].get('recall@5',0):.3f}"
            )
        logger.info(f"  → Rewrite Gain: R@5={rewrite_gain.get('recall@5', 0):+.4f}")

    # =====================================================================
    # 子评测 2：多跳推理查询 → GraphRAG 推理增益（B vs C）
    # =====================================================================
    if graph_queries:
        logger.info(f"=== Capability: Graph ({len(graph_queries)} queries) ===")
        runner_b = _make_agentic_rag_runner()
        runner_c = _make_full_rag_runner()

        logger.info("  Running B-AgenticRAG on graph queries...")
        start = time.time()
        result_b = batch_evaluate(graph_queries, runner_b, ks=KS)
        result_b["elapsed_sec"] = round(time.time() - start, 1)

        logger.info("  Running C-Full on graph queries...")
        start = time.time()
        result_c = batch_evaluate(graph_queries, runner_c, ks=KS)
        result_c["elapsed_sec"] = round(time.time() - start, 1)

        # 计算图谱增益：C 组各指标 - B 组各指标
        graph_gain = {}
        if "averages" in result_b and "averages" in result_c:
            for key in result_c["averages"]:
                graph_gain[key] = round(
                    result_c["averages"][key] - result_b["averages"].get(key, 0), 4
                )

        # 计算 hard_negative_recall：hard_negative 被正确排除的比例
        # = 1 - (hard_negative 出现在 Top-K 中的数量 / 总 hard_negative 数量)
        hn_recall = _compute_hard_negative_recall(graph_queries, runner_c, ks=KS)

        result["graph"] = {
            "num_queries": len(graph_queries),
            "B-AgenticRAG": result_b,
            "C-Full": result_c,
            "graph_gain": graph_gain,
            "hard_negative_recall": hn_recall,
        }

        if "averages" in result_b:
            logger.info(
                f"  B-AgenticRAG: P@5={result_b['averages'].get('precision@5',0):.3f}, "
                f"R@5={result_b['averages'].get('recall@5',0):.3f}"
            )
        if "averages" in result_c:
            logger.info(
                f"  C-Full: P@5={result_c['averages'].get('precision@5',0):.3f}, "
                f"R@5={result_c['averages'].get('recall@5',0):.3f}"
            )
        logger.info(f"  → Graph Gain: R@5={graph_gain.get('recall@5', 0):+.4f}")
        logger.info(f"  → Hard Negative Recall: {hn_recall.get('recall@5', 0):.3f}")

    return result


def _compute_hard_negative_recall(queries: list, runner, ks: list) -> dict:
    """计算 hard_negative_recall：硬负例被正确排除的比例。

    定义：HN-Recall@K = 1 - (出现在 Top-K 中的 hard_negative 数 / 总 hard_negative 数)
    值越接近 1 越好——说明系统能正确排除"看起来相关但实际不该出现"的结果。

    这对 GraphRAG 评测特别重要：多跳推理不仅要检索到正确的，还要排除
    表面相关但实际危险的（如"膝关节损伤→排除深蹲类"）。
    """
    hn_found = {k: 0 for k in ks}
    hn_total = {k: 0 for k in ks}
    valid = 0

    for q in queries:
        hard_neg = q.get("hard_negative_ids", [])
        if not hard_neg:
            continue
        valid += 1
        try:
            results = runner(q["query"])
            retrieved_names = [r.get("name", r.get("exercise", "")) for r in results]
        except Exception:
            continue

        for k in ks:
            top_k = retrieved_names[:k]
            hn_total[k] += len(hard_neg)
            # Count how many hard negatives (or substring matches) appear in top-K
            for hn in hard_neg:
                for name in top_k:
                    if hn in name or name in hn:
                        hn_found[k] += 1
                        break

    result = {}
    for k in ks:
        if hn_total[k] > 0:
            result[f"recall@{k}"] = round(1.0 - hn_found[k] / hn_total[k], 4)
        else:
            result[f"recall@{k}"] = 1.0

    result["num_queries_with_hn"] = valid
    return result


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
        query_text = query_dict.get("query", "")

        # 从 query 提取伤病信息注入 profile —— 之前用空 injuries=[ ]，
        # FactChecker 看不到任何伤病信号，导致漏报 q042 这类"腰痛"查询
        injuries = _extract_injuries_from_query(query_text)

        # 从 query 提取真实动作名，而非总是用"测试动作"
        exercise_name = _extract_exercise_for_eval(query_text)

        dummy_plan = {
            "days": [{
                "day": 1, "focus": query_text,
                "exercises": [{"name": exercise_name, "sets": 3, "reps": "10", "rest": "60s"}]
            }],
            "goal": "增肌",
            "user_query": query_text,  # 注入原始 query 供 FactChecker prompt 检查
        }
        profile = {"height": 175, "weight": 75, "training_years": 1,
                   "goal": "增肌", "available_equipment": ["哑铃"], "days_per_week": 4,
                   "injuries": injuries}

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
    parser.add_argument("--capability", action="store_true",
                        help="Run capability test (semantic + graph query subsets)")
    parser.add_argument("--ragas", action="store_true",
                        help="Run RAGAS evaluation (faithfulness + answer_relevancy + context_relevancy)")
    parser.add_argument("--ragas-limit", type=int, default=20,
                        help="Max queries for RAGAS eval (default: 20)")
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
        args.capability = True
        args.ragas = True

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

    # ---- 步骤 3.5：能力专项评测 ----
    capability_results = {}
    if args.capability:
        capability_results = run_capability_test(queries)

    # ---- 步骤 3.6：RAGAS 生成质量评测 ----
    ragas_results = {}
    if args.ragas:
        from eval.ragas_eval import run_ragas_eval, print_ragas_report
        logger.info("Running RAGAS evaluation...")
        ragas_results = run_ragas_eval(queries, limit=args.ragas_limit)
        print_ragas_report(ragas_results)

    # ---- 步骤 4：持久化结果到 results.json ----
    # 序列化便于后续 --report-only 复用，也方便人工检查逐 query 结果
    if ablation_results or agent_results or capability_results or ragas_results:
        RESULTS_FILE.write_text(
            json.dumps({"ablation": ablation_results, "agent": agent_results,
                        "capability": capability_results, "ragas": ragas_results},
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
