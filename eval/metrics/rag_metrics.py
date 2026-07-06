"""
=============================================================================
文件角色：RAG 检索质量评测指标计算模块。
=============================================================================

在整个项目中的位置：
- 被调用方：eval/run_eval.py 的 run_ablation() → batch_evaluate()
- 调用方：无（纯函数模块，不依赖外部组件）

实现的评测指标（均为信息检索领域标准指标）：
  1. Precision@K  — Top-K 结果中相关文档的比例（衡量"搜得准不准"）
  2. Recall@K     — 所有相关文档中被检索到的比例（衡量"搜得全不全"）
  3. MRR          — 第一个相关文档排名的倒数（衡量"第一条就命中"的能力）
  4. DCG@K        — 折损累积增益，考虑排名位置和相关性等级
  5. NDCG@K       — 归一化 DCG，除以理想排序的 DCG（0~1 之间，越接近 1 越好）

关键设计决策：
  - 所有指标以文档 name 为匹配键（而非 ID），因为检索结果中 name 更通用
  - 支持分级相关度（relevance_scores），默认为二值相关度
  - 空相关集处理：Recall 和 NDCG 对空相关集返回 1.0（保守乐观处理）
"""

import math
from typing import List, Dict


def precision_at_k(
    retrieved_doc_names: List[str],
    relevant_doc_names: List[str],
    k: int,
) -> float:
    """Precision@K：Top-K 检索结果中相关文档的比例。

    数学含义：
      P@K = (前 K 个结果中相关文档数量) / K
      取值范围 [0, 1]，值越大说明检索结果越"精准"。

    输入：
      retrieved_doc_names: List[str] — 检索系统返回的文档名列表（按相似度降序排列）
      relevant_doc_names: List[str] — 标注为相关的文档名列表（黄金标准答案）
      k: int — 截断位置（只看前 K 个结果）

    输出：
      float — Precision@K 值，范围 [0.0, 1.0]

    核心逻辑：
      1. 取检索结果的前 K 个（top_k = retrieved[:k]）
      2. 将相关文档集合转为 set 以加速查找（O(1) 查找）
      3. 统计 top_k 中有多少个在相关文档集合中 → hits
      4. 返回 hits / k
    """
    if k <= 0:
        return 0.0  # 非法 K 值返回 0
    top_k = retrieved_doc_names[:k]  # 只取前 K 个
    if not top_k:
        return 0.0  # 检索结果为空
    relevant_set = set(relevant_doc_names)  # 转为 set，O(1) 查找
    hits = sum(1 for doc in top_k if doc in relevant_set)  # 数命中数
    return hits / k  # 比例 = 命中数 / K


def recall_at_k(
    retrieved_doc_names: List[str],
    relevant_doc_names: List[str],
    k: int,
) -> float:
    """Recall@K：所有相关文档中，在 Top-K 里被检索到的比例。

    数学含义：
      R@K = (前 K 个结果中相关文档数量) / (总相关文档数量)
      取值范围 [0, 1]，值越大说明检索结果越"全面"。

    输入：
      retrieved_doc_names: List[str] — 检索系统返回的文档名列表
      relevant_doc_names: List[str] — 标注为相关的文档名列表
      k: int — 截断位置

    输出：
      float — Recall@K 值，范围 [0.0, 1.0]

    核心逻辑：
      1. 取检索结果前 K 个，数命中数
      2. 分母是总相关文档数（而非 K），因此 Recall 衡量"找到的覆盖率"

    注意：
      空相关集返回 1.0（"全部 0 个相关文档都被找到了"），
      这是一种保守乐观处理，避免除零。
    """
    if not relevant_doc_names:
        return 1.0  # 空相关集视为全部找到（保守乐观处理，避免除零）
    if k <= 0:
        return 0.0
    top_k = retrieved_doc_names[:k]
    relevant_set = set(relevant_doc_names)  # set O(1) 查找
    hits = sum(1 for doc in top_k if doc in relevant_set)
    return hits / len(relevant_set)  # 比例 = 命中数 / 总相关文档数


def mrr(
    retrieved_doc_names: List[str],
    relevant_doc_names: List[str],
) -> float:
    """MRR（Mean Reciprocal Rank，平均倒数排名）：1 / 第一个相关文档的排名。

    数学含义：
      MRR = 1 / rank_of_first_relevant_doc
      如果第一个相关文档排在第 1 位 → MRR = 1.0
      如果第一个相关文档排在第 2 位 → MRR = 0.5
      如果第一个相关文档排在第 10 位 → MRR = 0.1
      如果一个相关文档都没找到 → MRR = 0.0

    适用场景：
      MRR 衡量系统能否"第一条结果就命中"，适用于推荐场景中
      用户只看前几条结果的情况。值越高说明相关结果越靠前。

    输入：
      retrieved_doc_names: List[str] — 检索结果（按序排列）
      relevant_doc_names: List[str] — 相关文档列表

    输出：
      float — MRR 值，范围 [0.0, 1.0]
    """
    relevant_set = set(relevant_doc_names)  # O(1) 查找
    for i, doc in enumerate(retrieved_doc_names, start=1):  # 从 rank=1 开始遍历
        if doc in relevant_set:
            return 1.0 / i  # 返回 1/排名，排得越前值越高
    return 0.0  # 一个相关文档都没找到


def dcg_at_k(
    retrieved_doc_names: List[str],
    relevance_scores: Dict[str, int],
    k: int,
) -> float:
    """DCG@K（Discounted Cumulative Gain，折损累积增益）。

    数学含义：
      DCG@K = SUM(i=1 to K) [ rel_i / log2(i+1) ]
      其中 rel_i 是排在第 i 位文档的相关度等级，log2(i+1) 是折损因子。
      排名越靠后的文档，其贡献被对数折损得越厉害 —
      这模拟了"用户更关注排名靠前的结果"的行为。

    输入：
      retrieved_doc_names: List[str] — 检索结果（按序排列）
      relevance_scores: Dict[str, int] — 文档名 → 相关度等级的映射
                              例如：{0: 无关, 1: 部分相关, 2: 高度相关}
                              未在字典中的文档默认相关度为 0（无关）
      k: int — 截断位置

    输出：
      float — DCG@K 值（非归一化，可能大于 1）

    与 Precision/Recall 的区别：
      DCG 考虑了排名位置和相关度等级，而 P@K / R@K 仅做二值判断。
      例如：排在位置 1 的高度相关文档比排在位置 5 的部分相关文档贡献更大，
      这在 DCG 中会被准确反映，但在 P@K 中两者权重相同。
    """
    dcg = 0.0
    for i, doc in enumerate(retrieved_doc_names[:k], start=1):  # i 从 1 开始（排名）
        rel = relevance_scores.get(doc, 0)  # 获取相关度等级，默认 0
        dcg += rel / math.log2(i + 1)  # rel 除以 log2(i+1) 实现折损
    return dcg


def ndcg_at_k(
    retrieved_doc_names: List[str],
    relevant_doc_names: List[str],
    k: int,
    relevance_scores: Dict[str, int] = None,
) -> float:
    """NDCG@K（Normalized DCG，归一化折损累积增益）。

    数学含义：
      NDCG@K = actual_DCG@K / ideal_DCG@K
      = (实际排序的 DCG) / (完美排序的 DCG)
      取值范围 [0, 1]，越接近 1 表示排序越接近完美。

    归一化的作用：
      DCG 的绝对值受相关度等级数量和查询难易程度影响，不同查询之间不可比。
      除以理想 DCG（所有相关文档按相关度降序排列、排在所有无关文档之前的 DCG）
      后，NDCG 被归一化到 [0, 1]，跨查询可比。

    输入：
      retrieved_doc_names: List[str] — 检索结果（按序排列）
      relevant_doc_names: List[str] — 相关文档列表
      k: int — 截断位置
      relevance_scores: Dict[str, int] — 文档→相关度等级，为 None 时使用二值相关度

    输出：
      float — NDCG@K 值，范围 [0.0, 1.0]

    核心逻辑：
      1. 如果未提供 relevance_scores，默认所有相关文档相关度为 1（二值）
      2. 计算实际 DCG（actual_dcg）
      3. 构造理想排序 — 所有相关文档按相关度降序排在最前面
      4. 计算理想 DCG（ideal_dcg）
      5. NDCG = actual_dcg / ideal_dcg
    """
    if not relevant_doc_names:
        return 1.0  # 空相关集的处理同 Recall

    # 默认：二值相关度 — 标注为相关的文档相关度=1，其他为 0
    if relevance_scores is None:
        relevance_scores = {doc: 1 for doc in relevant_doc_names}

    # 实际排序的 DCG
    actual_dcg = dcg_at_k(retrieved_doc_names, relevance_scores, k)

    # 理想排序的 DCG：所有相关文档按相关度降序排在最前
    # 这是"如果系统完美排序"能达到的最大 DCG
    ideal_order = sorted(
        relevant_doc_names,
        key=lambda d: relevance_scores.get(d, 0),  # 按相关度排序
        reverse=True,  # 降序：高相关度的排前面
    )
    ideal_dcg = dcg_at_k(ideal_order, relevance_scores, k)

    if ideal_dcg == 0:
        return 0.0  # 所有文档相关度都为 0，NDCG 无意义
    return actual_dcg / ideal_dcg  # 归一化：实际 DCG 占理想 DCG 的比例


def evaluate_retrieval(
    retrieved_doc_names: List[str],
    relevant_doc_names: List[str],
    ks: List[int] = None,
    relevance_scores: Dict[str, int] = None,
) -> dict:
    """对单条查询运行全部检索指标（单查询评测）。

    输入：
      retrieved_doc_names: List[str] — 检索结果文档名列表
      relevant_doc_names: List[str] — 黄金标准相关文档名列表
      ks: List[int] — 评测用的 K 值列表，默认 [1, 3, 5, 10]
      relevance_scores: Dict[str, int] — 文档→相关度等级，默认用二值相关度

    输出：
      dict — 包含各 K 值下所有指标的字典：
        {
          "precision@1": 0.8, "recall@1": 0.4, "ndcg@1": 0.6,
          "precision@3": 0.67, "recall@3": 0.8, "ndcg@3": 0.72,
          ...
          "mrr": 0.8
        }

    核心逻辑：
      遍历每个 K 值，分别计算 P@K, R@K, NDCG@K。
      MRR 与 K 无关，单独计算一次即可。
    """
    if ks is None:
        ks = [1, 3, 5, 10]  # 默认 K 值：衡量 Top-1, -3, -5, -10

    result = {}
    for k in ks:
        # 为每个 K 值计算三个指标
        result[f"precision@{k}"] = precision_at_k(retrieved_doc_names, relevant_doc_names, k)
        result[f"recall@{k}"] = recall_at_k(retrieved_doc_names, relevant_doc_names, k)
        result[f"ndcg@{k}"] = ndcg_at_k(retrieved_doc_names, relevant_doc_names, k, relevance_scores)

    # MRR 与 K 无关，全量列表上计算，只算一次
    result["mrr"] = mrr(retrieved_doc_names, relevant_doc_names)
    return result


def batch_evaluate(
    queries: List[dict],
    run_retrieval,
    ks: List[int] = None,
) -> dict:
    """对一批查询运行检索评测（批量评测入口）。

    这是 run_eval.py 直接调用的函数，负责协调"逐查询检索 → 逐查询计算指标 → 聚合平均"。

    输入：
      queries: List[dict] — 黄金查询列表，每条必须包含：
               - "id": 查询唯一标识
               - "query": 查询文本
               - "relevant_doc_ids": List[str] — 黄金标准的相关文档名
      run_retrieval: callable(query_text: str) -> List[dict]
                     检索执行函数，返回的结果列表中每个 dict 必须包含 "name" 或 "exercise" 键
      ks: List[int] — 评测用 K 值列表，默认 [1, 3, 5, 10]

    输出：
      dict:
        {
          "num_queries": int,           # 成功评测的查询数
          "averages": {                 # 所有查询的平均指标
            "precision@5": 0.75,
            "recall@5": 0.82,
            "mrr": 0.68,
            "ndcg@5": 0.73,
            ...
          },
          "per_query": [               # 逐查询详情（用于 Debug）
            {"id": "q01", "query": "...", "retrieved": [...], "relevant": [...], ...},
            ...
          ]
        }

    核心逻辑（三步流水线）：
      1. 逐查询执行检索 run_retrieval(query_text)
         → 从结果中提取文档名（兼容 "name" 和 "exercise" 字段名）
      2. 逐查询计算指标 evaluate_retrieval()
         → 得到每条查询的 P@K, R@K, MRR, NDCG@K
      3. 跨查询平均：每个指标对所有查询取算术平均
         → 得到宏观指标用于消融实验对比
    """
    if ks is None:
        ks = [1, 3, 5, 10]

    all_metrics = []  # 成功评测的查询指标列表
    per_query = []    # 逐查询详情（包含前 5 个检索结果，方便 Debug）

    for q in queries:
        query_text = q["query"]        # 查询文本
        relevant = q["relevant_doc_ids"]  # 黄金标准相关文档

        # ---- 步骤 1：执行检索 ----
        try:
            results = run_retrieval(query_text)
        except Exception as e:
            # 单个查询失败不应中断全量评测，记录错误后继续
            per_query.append({"id": q["id"], "error": str(e)})
            continue

        # ---- 步骤 2：提取文档名 ----
        # 兼容不同检索器返回的字段名（name 或 exercise）
        retrieved_names = [r.get("name", r.get("exercise", "")) for r in results]

        # ---- 步骤 3：计算指标 ----
        metrics = evaluate_retrieval(retrieved_names, relevant, ks)
        metrics["id"] = q["id"]
        all_metrics.append(metrics)

        # ---- 步骤 4：记录逐查询详情（截断前 5 个，避免报告过长）----
        per_query.append({
            "id": q["id"],
            "query": query_text[:80],       # 截断到 80 字符
            "retrieved": retrieved_names[:5], # 只保留前 5 个检索结果
            "relevant": relevant[:5],         # 只保留前 5 个相关文档
            **metrics,
        })

    # ---- 步骤 5：跨查询聚合平均指标 ----
    if not all_metrics:
        return {"error": "No successful queries", "per_query": per_query}

    # 构建所有需要平均的指标名列表
    key_names = [f"precision@{k}" for k in ks] + [f"recall@{k}" for k in ks] + [f"ndcg@{k}" for k in ks] + ["mrr"]
    avg_metrics = {}
    for key in key_names:
        # 算术平均：所有查询的指标值求和 / 查询数量
        avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

    return {
        "num_queries": len(all_metrics),
        "averages": avg_metrics,  # 平均指标 → 用于消融实验对比表
        "per_query": per_query,    # 逐查询详情 → 用于 Debug 和错误分析
    }
