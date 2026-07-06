"""
=============================================================================
文件角色：**黄金评测数据集（Golden Dataset）的构建与校验工具**。
=============================================================================

在整个项目中的位置：
- 被调用方：
  1. 命令行直接运行（python build_dataset.py validate|stats）
  2. eval/run_eval.py 调用 load_dataset() 加载数据、validate() 校验数据
- 调用方：无外部依赖，仅读写 queries.json

数据集结构（queries.json 中每条记录包含）：
  - id                  唯一标识符（必填）
  - query               用户查询文本（必填）
  - category            场景分类：muscle_building/fat_loss/exercise_analysis/injury/mixed
  - relevant_doc_ids    相关文档 ID 列表（必填，人工标注的"理想检索结果"）
  - expected_route      期望的路由目标：muscle_building/fat_loss/exercise_analysis/qa
  - safety_risk         安全风险等级：low/medium/high
  - difficulty          难度等级（可选）
  - expected_hitl       是否预期需要人工介入（可选）

用法：
    python eval/golden_dataset/build_dataset.py validate     # 校验现有 queries.json
    python eval/golden_dataset/build_dataset.py stats         # 打印数据集统计信息
"""
import json
import sys
from pathlib import Path
from collections import Counter

# 黄金数据集文件路径，与当前脚本同目录
DATASET_PATH = Path(__file__).parent / "queries.json"


def load_dataset() -> list[dict]:
    """加载黄金数据集 JSON 文件。

    输入：无（从 DATASET_PATH 读取 queries.json）
    输出：list[dict] — 查询字典列表，每个 dict 包含一条评测查询的完整信息

    核心逻辑：
      直接读取 JSON 文件并解析为 Python list[dict]。
      不做任何校验（校验逻辑在 validate() 中），保持数据加载与校验分离。
    """
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def validate(dataset: list[dict] = None) -> bool:
    """校验数据集中每条查询的完整性和合法性。

    输入：
      dataset: list[dict] — 待校验的查询列表，为 None 时自动从 queries.json 加载

    输出：
      bool — True 表示全部校验通过，False 表示存在错误

    核心逻辑（逐条逐字段校验）：
      1. 必填字段检查：id, query, category, relevant_doc_ids, expected_route, safety_risk
      2. ID 唯一性检查：确保无重复 id（评测结果按 id 索引）
      3. 枚举值校验：category / route / safety_risk 必须在预定义合法值范围内
      4. relevant_doc_ids 非空校验：空的 relevant_doc_ids 无法计算 Recall 等指标
    """
    if dataset is None:
        dataset = load_dataset()

    # ---- 校验规则定义 ----
    required = ["id", "query", "category", "relevant_doc_ids", "expected_route", "safety_risk"]
    valid_categories = {"muscle_building", "fat_loss", "exercise_analysis", "injury", "mixed"}
    valid_routes = {"muscle_building", "fat_loss", "exercise_analysis", "qa"}
    valid_risks = {"low", "medium", "high"}

    errors = []
    ids = set()  # 用于检测重复 id

    for q in dataset:
        # ---- 规则 1：必填字段存在性 ----
        for field in required:
            if field not in q:
                errors.append(f"{q.get('id', '?')}: missing '{field}'")

        # ---- 规则 2：ID 唯一性 ----
        # 重复 id 会导致评测结果无法唯一关联到具体查询
        qid = q.get("id", "")
        if qid in ids:
            errors.append(f"{qid}: duplicate id")
        ids.add(qid)

        # ---- 规则 3：category 枚举值合法性 ----
        if q.get("category") not in valid_categories:
            errors.append(f"{qid}: invalid category '{q.get('category')}'")

        # ---- 规则 4：expected_route 枚举值合法性 ----
        # route 决定了 Planner 应该将查询路由到哪个 skill
        if q.get("expected_route") not in valid_routes:
            errors.append(f"{qid}: invalid route '{q.get('expected_route')}'")

        # ---- 规则 5：safety_risk 枚举值合法性 ----
        if q.get("safety_risk") not in valid_risks:
            errors.append(f"{qid}: invalid safety_risk '{q.get('safety_risk')}'")

        # ---- 规则 6：relevant_doc_ids 非空 ----
        # 空列表意味着没有"正确答案"，无法计算 Recall/Precision
        if not q.get("relevant_doc_ids"):
            errors.append(f"{qid}: relevant_doc_ids is empty")

    if errors:
        print(f"VALIDATION FAILED — {len(errors)} errors:")
        for e in errors:
            print(f"  - {e}")
        return False

    print(f"VALIDATION PASSED — {len(dataset)} queries, {len(ids)} unique IDs")
    return True


def stats(dataset: list[dict] = None):
    """打印数据集的统计信息，帮助评估数据分布是否合理。

    输入：
      dataset: list[dict] — 待统计的查询列表，为 None 时从 queries.json 自动加载
    输出：
      无返回值（直接打印到 stdout）

    核心逻辑（6 个维度的分布统计）：
      1. 按 category 分布 — 检查各场景样本是否均衡
      2. 按 safety_risk 分布 — 检查风险等级比例（low 应占多数，但不能全是 low）
      3. 按 difficulty 分布 — 检查难度梯度是否合理
      4. 按 expected_route 分布 — 路由目标分布
      5. HITL 预期占比 — 人工介入的预期比例
      6. 平均相关文档数 — 每个 query 平均标注了多少个 relevant doc
         （太少则 Recall 计算不稳定，太多则标注成本高）
    """
    if dataset is None:
        dataset = load_dataset()

    print(f"Total queries: {len(dataset)}")
    print()

    # ---- 维度 1：按场景分类分布 ----
    # 5 个 category 应尽量均衡，避免评测偏向某一场景
    print("By category:")
    cat_counts = Counter(q["category"] for q in dataset)
    for cat, count in cat_counts.most_common():
        print(f"  {cat}: {count}")
    print()

    # ---- 维度 2：按安全风险等级分布 ----
    # low 应占多数（正常查询），medium/high 是安全评测的关键样本
    print("By safety risk:")
    risk_counts = Counter(q["safety_risk"] for q in dataset)
    for risk, count in risk_counts.most_common():
        print(f"  {risk}: {count}")
    print()

    # ---- 维度 3：按难度分布 ----
    # 使用 .get("difficulty", "?") 处理可选字段缺失的情况
    print("By difficulty:")
    diff_counts = Counter(q.get("difficulty", "?") for q in dataset)
    for diff, count in diff_counts.most_common():
        print(f"  {diff}: {count}")
    print()

    # ---- 维度 4：按期望路由分布 ----
    # Planner 会将用户 query 路由到不同 skill，标注的 route 分布反映真实用户意图分布
    print("By expected route:")
    route_counts = Counter(q["expected_route"] for q in dataset)
    for route, count in route_counts.most_common():
        print(f"  {route}: {count}")
    print()

    # ---- 维度 5：HITL（Human-In-The-Loop）预期分布 ----
    # expected_hitl=True 的查询需要人工介入（如伤病风险场景）
    hitl_yes = sum(1 for q in dataset if q.get("expected_hitl"))
    print(f"HITL expected: {hitl_yes}/{len(dataset)}")

    # ---- 维度 6：平均相关文档标注数 ----
    # 太少 → Recall 方差大；太多 → 标注成本高。一般 3-8 为宜
    avg_rel = sum(len(q.get("relevant_doc_ids", [])) for q in dataset) / len(dataset)
    print(f"Avg relevant_doc_ids per query: {avg_rel:.1f}")


# ============================================================================
# CLI 入口：手动校验数据集或查看统计信息
# 使用方法：python build_dataset.py [validate|stats]
# ============================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python build_dataset.py [validate|stats]")
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "validate":
        validate()  # 校验 queries.json 数据完整性
    elif cmd == "stats":
        stats()     # 打印数据分布统计
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
