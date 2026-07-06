"""
=============================================================================
文件角色：Agent 决策质量评测指标计算模块。
=============================================================================

在整个项目中的位置：
- 被调用方：eval/run_eval.py 的 run_agent_eval()
  → evaluate_planner()  / evaluate_factchecker()
- 调用方：无（纯函数模块）

评测维度：
  1. Planner 路由正确率 — Agent 是否把用户查询路由到了正确的 skill
     - route_accuracy()     整体正确率 + 各类别 Precision/Recall/F1
     - confusion_matrix()   混淆矩阵（行=真实路由, 列=预测路由）
  2. FactChecker 安全分类 — Agent 是否识别出了有安全风险的查询
     - safety_metrics()     将 "high" 视为正类，计算 Precision/Recall/FPR/FNR
     - evaluate_planner()   批量评测入口，含按 category 斜切分析
     - evaluate_factchecker()  批量评测入口，含 injury 专项子集分析

安全评测中的关键指标解释：
  - FPR（假阳性率/误报率）= 无害查询被误标为危险的占比
    → FPR 高会降低用户体验（频繁弹警告），但不会造成伤害
  - FNR（假阴性率/漏报率）= 危险查询未被标记的占比
    → FNR 是安全评测中**最重要**的指标，应为 0%
    → 宁可误报多一些（高 FPR），也不能漏掉一个危险（高 FNR）
"""

from typing import List, Dict, Tuple
from collections import Counter

# Planner 支持的全部路由标签，顺序固定以保持混淆矩阵排列一致
ROUTE_LABELS = ["muscle_building", "fat_loss", "exercise_analysis", "qa"]


def route_accuracy(
    y_true: List[str],
    y_pred: List[str],
) -> dict:
    """计算 Planner 路由预测的正确率及每个类别的详细指标。

    输入：
      y_true: List[str] — 真实路由标签（来自黄金数据的 expected_route）
      y_pred: List[str] — 预测的路由标签（来自 PlannerAgent.plan() 的 skill 字段）

    输出：
      dict:
        {
          "num_samples": int,       # 样本总数
          "correct": int,           # 正确预测数
          "accuracy": float,        # 整体正确率 = correct / num_samples
          "per_class": {            # 每个类别的分解指标
            "muscle_building": {
              "precision": 0.85,    # = TP / (TP + FP) — 预测为该类别的样本中真正属于该类别的比例
              "recall": 0.90,       # = TP / (TP + FN) — 真正属该类别的样本中被正确预测的比例
              "f1": 0.87,           # = 2 * P * R / (P + R) — Precision 和 Recall 的调和平均
              "support": 20,        # 该类别在真实数据中的样本数
              "correct": 18,        # 正确预测数
              "predicted_as": 22,   # 被预测为该类别的总数
            },
            ...
          }
        }

    核心逻辑：
      1. 长度校验 → 防御性编程
      2. 整体正确率：逐对比较 y_true[i] vs y_pred[i]
      3. 各类别指标：
         - 对每个类别 cls:
           TP = 真实是 cls 且预测是 cls
           FP = 真实不是 cls 但预测是 cls（错误路由到此类）
           FN = 真实是 cls 但预测不是 cls（漏掉了此类）
         - 计算 Precision, Recall, F1
         - support = 真实数据中该类的样本数（用于判断指标可信度）
    """
    if len(y_true) != len(y_pred):
        return {"error": f"Length mismatch: {len(y_true)} vs {len(y_pred)}"}

    n = len(y_true)
    # 整体正确率：预测与真实一致的样本占比
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / n if n > 0 else 0.0

    # ---- 各类别指标 ----
    # 使用 y_true 和 y_pred 的并集确保覆盖所有出现过的类别
    classes = sorted(set(y_true) | set(y_pred))
    per_class = {}
    for cls in classes:
        # TP: 真实类别 = cls 且预测类别 = cls（正确路由）
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        # FP: 真实类别 != cls 但预测类别 = cls（错误路由到此类，张冠李戴）
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        # FN: 真实类别 = cls 但预测类别 != cls（漏掉了此类，没有正确路由）
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)

        # Precision = TP / (TP + FP)：预测为此类的样本中有多少是对的
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        # Recall = TP / (TP + FN)：真正此类的样本中有多少被找到了
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # F1 = 2 * P * R / (P + R)：调和平均，兼顾精确率和召回率
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(1 for t in y_true if t == cls),  # 真实样本数（决定指标可信度）
            "correct": tp,                                     # 正确预测数
            "predicted_as": sum(1 for p in y_pred if p == cls),  # 预测为该类的总数
        }

    return {
        "num_samples": n,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "per_class": per_class,
    }


def confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
) -> dict:
    """构建 Planner 路由的混淆矩阵。

    混淆矩阵解读（假设 3 个类别 A, B, C）：
              预测 A  预测 B  预测 C
      真实 A  [ 15      2      0  ]  ← 真实是 A 的样本中，15 个正确，2 个误判为 B
      真实 B  [  1     12      1  ]  ← 对角线 = 正确预测数
      真实 C  [  0      0      8  ]  ← 非对角线 = 错误预测，分布反映混淆模式

    输入：
      y_true: List[str] — 真实路由标签
      y_pred: List[str] — 预测路由标签

    输出：
      dict:
        {
          "labels": ["muscle_building", "fat_loss", ...],  # 类别标签列表
          "matrix": [            # 矩阵[真实行][预测列]
            [15, 2, 0, 0],       # muscle_building 的预测分布
            [1, 12, 1, 0],       # fat_loss 的预测分布
            ...
          ]
        }

    核心逻辑：
      1. 确定使用的标签集：优先用 ROUTE_LABELS，如数据中标签不同则用实际出现的标签
      2. 构建标签→索引的映射
      3. 遍历每对 (true, pred)，累加 matrix[true_idx][pred_idx]
    """
    labels = ROUTE_LABELS
    # 如果实际数据中出现了 ROUTE_LABELS 之外的标签，使用实际标签
    active = sorted(set(y_true) | set(y_pred))
    if active != labels:
        labels = active  # 回退到实际出现的标签

    idx = {label: i for i, label in enumerate(labels)}  # 标签 → 行列索引
    # 初始化全零矩阵
    matrix = [[0] * len(labels) for _ in range(len(labels))]

    # 遍历填充：matrix[真实标签行][预测标签列] += 1
    for t, p in zip(y_true, y_pred):
        matrix[idx[t]][idx[p]] += 1

    return {"labels": labels, "matrix": matrix}


def safety_metrics(
    y_true_safety: List[str],
    y_pred_safety: List[str],
) -> dict:
    """计算安全分类指标（FactChecker 的核心评测函数）。

    将三分类（high/medium/low）简化为二分类问题：
      - 正类（Positive）= "high" — 需要人工介入的危险查询
      - 负类（Negative）= "medium" 或 "low" — 不需要人工介入的查询

    输入：
      y_true_safety: List[str] — 真实风险等级，取值为 "high"/"medium"/"low"
      y_pred_safety: List[str] — 预测的风险等级

    输出：
      dict:
        {
          "num_samples": int,
          "true_positive": int,   # TP: 实际危险且被正确标记
          "false_positive": int,  # FP: 实际不危险但被误标为危险（误报）
          "false_negative": int,  # FN: 实际危险但未被标记（漏报！最关键！）
          "true_negative": int,   # TN: 实际不危险且未被标记
          "precision": float,     # = TP/(TP+FP) — 标记为危险的样本中真正危险的比例
          "recall": float,        # = TP/(TP+FN) — 所有危险样本中被标记的比例
          "fpr": float,           # = FP/(FP+TN) — 假阳性率/误报率，无关样本被误标
          "fnr": float,           # = FN/(FN+TP) — 假阴性率/漏报率，危险样本被漏标
          "f1": float,            # Precision 和 Recall 的调和平均
        }

    关键指标解释：
      - FNR（漏报率）是**安全评测中最重要的指标**。
        漏报 = 危险内容被放行 = 用户可能受伤 → 必须为 0%。
      - FPR（误报率）高了会降低使用体验（频繁弹警告），但不会导致伤害。
        因此安全策略可以容忍较高的 FPR，但不能容忍 FNR > 0。

    核心逻辑：
      1. 将三分类转为二分类：high vs 其他
      2. 计算混淆矩阵的四格表：TP, FP, FN, TN
      3. 推导 Precision, Recall, FPR, FNR, F1
    """
    n = len(y_true_safety)
    if n == 0:
        return {"error": "No samples"}

    # ---- 二分类混淆矩阵四格表 ----
    # 正类 = "high"：需要人工介入（Human-In-The-Loop）
    tp = sum(1 for t, p in zip(y_true_safety, y_pred_safety) if t == "high" and p == "high")  # 正确报警
    fp = sum(1 for t, p in zip(y_true_safety, y_pred_safety) if t != "high" and p == "high")  # 误报
    fn = sum(1 for t, p in zip(y_true_safety, y_pred_safety) if t == "high" and p != "high")  # 漏报！
    tn = sum(1 for t, p in zip(y_true_safety, y_pred_safety) if t != "high" and p != "high")  # 正确不报警

    # ---- 推导指标 ----
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # 报警的命中率
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0     # 危险的召回率
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0        # 假阳性率（误报率）
                                                           # FPR = FP / (FP + TN)，即所有负样本中被误报的比例
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0        # 假阴性率（漏报率）！
                                                           # FNR = FN / (FN + TP)，即所有正样本中被漏掉的比例

    return {
        "num_samples": n,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0,
    }


def evaluate_planner(
    queries: List[dict],
    run_planner,
) -> dict:
    """对一批查询评测 Planner 路由效果（批量评测入口）。

    输入：
      queries: List[dict] — 黄金查询列表，每条含：
               - "id": 唯一标识
               - "query": 查询文本
               - "expected_route": 期望路由标签（muscle_building/fat_loss/exercise_analysis/qa）
               - "category": 场景分类（可选，用于按分类斜切分析）
      run_planner: callable(query_text: str) -> str
                   执行路由预测的函数，接受查询文本，返回路由标签字符串

    输出：
      dict — 整合 route_accuracy、confusion_matrix、by_category 的完整结果：
        {
          "num_samples": int, "correct": int, "accuracy": float,
          "per_class": { ... },              # 各类别 Precision/Recall/F1
          "confusion_matrix": {              # 混淆矩阵
            "labels": [...],
            "matrix": [[...], ...]
          },
          "by_category": {                   # 按业务场景分组的斜切分析
            "muscle_building": { "accuracy": 0.9, "per_class": {...} },
            "fat_loss": { ... },
            ...
          },
          "errors": [...]                    # 执行失败记录的列表
        }

    核心逻辑：
      1. 逐查询调用 run_planner，收集 y_true/y_pred
      2. 计算整体路由正确率和混淆矩阵
      3. by_category 斜切分析：按 category 分组计算子集指标
         → 发现"在增肌场景表现好在减脂场景表现差"等问题
    """
    y_true = []
    y_pred = []
    errors = []

    # ---- 步骤 1：逐查询执行 Planner 路由 ----
    for q in queries:
        expected = q["expected_route"]  # 黄金标准路由
        try:
            predicted = run_planner(q["query"])  # Planner 实际路由结果
        except Exception as e:
            errors.append({"id": q["id"], "error": str(e)})
            continue

        y_true.append(expected)
        y_pred.append(predicted)

    if not y_true:
        return {"error": "No successful planner calls", "errors": errors}

    # ---- 步骤 2：整体指标 ----
    route_result = route_accuracy(y_true, y_pred)     # 正确率 + 各类别指标
    cm = confusion_matrix(y_true, y_pred)              # 混淆矩阵

    # ---- 步骤 3：按业务场景分组的斜切分析 ----
    # 目的：发现模型在不同场景下的表现差异
    # 例如：muscle_building 场景准确 95%，但 injury 场景仅 60%
    by_category = {}
    for cat in set(q.get("category", "?") for q in queries):
        cat_queries = [q for q in queries if q.get("category") == cat]
        cat_true = []
        cat_pred = []
        for q in cat_queries:
            try:
                cat_true.append(q["expected_route"])
                cat_pred.append(run_planner(q["query"]))
            except Exception:
                pass
        if cat_true:
            by_category[cat] = route_accuracy(cat_true, cat_pred)

    return {
        **route_result,
        "confusion_matrix": cm,
        "by_category": by_category,  # 按场景分组的斜切分析
        "errors": errors,
    }


def evaluate_factchecker(
    queries: List[dict],
    run_factchecker,
) -> dict:
    """评测 FactChecker 安全分类效果（批量评测入口）。

    输入：
      queries: List[dict] — 黄金查询列表，每条含：
               - "id": 唯一标识
               - "safety_risk": 黄金标准风险等级（"high"/"medium"/"low"）
               - "category": 场景分类（用于 injury 专项子集分析）
      run_factchecker: callable(query_dict: dict) -> dict
                       执行安全审查的函数，输出必须包含 "safety_risk" 字段

    输出：
      dict — 整合 safety_metrics 和 injury 专项子集分析的结果：
        {
          "num_samples": int, "precision": float, "recall": float,
          "fpr": float, "fnr": float, "f1": float,
          "injury_subset": { ... } | None,  # 伤病查询专项指标（最关键子集）
          "errors": [...]
        }

    核心逻辑：
      1. 逐查询执行 FactChecker 安全审查，收集 y_true_safety / y_pred_safety
      2. 计算整体安全指标（FPR, FNR, Precision, Recall）
      3. injury 专项子集：单独分析伤病类查询的安全指标
         → 伤病查询在安全上是最关键的，应全部被标记为 high
         → 如果 injury 子集 FNR > 0，说明存在严重安全隐患
    """
    y_true_safety = []
    y_pred_safety = []
    errors = []

    # ---- 步骤 1：逐查询执行安全审查 ----
    for q in queries:
        expected = q.get("safety_risk", "low")  # 默认无风险
        try:
            result = run_factchecker(q)  # 调用 FactCheckerAgent.check()
            predicted = result.get("safety_risk", "low")  # 提取预测的风险等级
        except Exception as e:
            errors.append({"id": q["id"], "error": str(e)})
            continue

        y_true_safety.append(expected)
        y_pred_safety.append(predicted)

    if not y_true_safety:
        return {"error": "No successful factchecker calls", "errors": errors}

    # ---- 步骤 2：整体安全指标 ----
    metrics = safety_metrics(y_true_safety, y_pred_safety)

    # ---- 步骤 3：injury 专项子集分析 ----
    # 伤病类查询（category="injury"）是安全评测中最关键的子集
    # 设计意图：伤病查询涉及用户健康安全，应全部被标记为 high
    # 如果此子集 FNR > 0 → 存在严重安全隐患，不容忽视
    injury_queries = [q for q in queries if q.get("category") == "injury"]
    if injury_queries:
        injury_true = []
        injury_pred = []
        for q in injury_queries:
            try:
                injury_true.append(q.get("safety_risk", "high"))  # 伤病查询的 safety_risk 应为 high
                injury_pred.append(run_factchecker(q).get("safety_risk", "low"))
            except Exception:
                pass
        injury_metrics = safety_metrics(injury_true, injury_pred)
    else:
        injury_metrics = None

    return {
        **metrics,
        "injury_subset": injury_metrics,  # 伤病专项指标（面试时可单独强调"injury 子集 FNR=0%"）
        "errors": errors,
    }
