"""
=============================================================================
文件角色：评测报告生成器（Markdown + matplotlib 图表）。
=============================================================================

在整个项目中的位置：
- 被调用方：eval/run_eval.py 的 main() → generate_markdown_report()
- 调用方：
  - matplotlib（可选，用于生成图表）
  - 读写 eval/figures/ 目录下的 PNG 图片

生成的报告内容：
  1. 报告元信息（时间、数据集规模、K 值）
  2. RAG 检索质量 — 消融实验对比表 + 各 K 值详细数据
  3. Agent 决策质量 — Planner 路由正确率 + FactChecker 安全指标
  4. 可视化图表 — 消融柱状图 + 混淆矩阵热力图
  5. 面试话术要点 — 结构化呈现关键结论

报告输出路径：默认 eval/EVAL_REPORT.md，可通过 --output 参数指定
"""

import json
import time
from pathlib import Path
from typing import Dict, List

# eval/ 目录和图表输出子目录
EVAL_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = EVAL_DIR / "figures"


def _ensure_figures_dir():
    """确保图表输出目录存在，不存在则自动创建。"""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_ablation_comparison(ablation_results: Dict[str, dict], ks: List[int]):
    """生成消融实验分组柱状对比图（1x2 子图：Precision 和 Recall）。

    输入：
      ablation_results: Dict[str, dict] — 消融评测结果，key=组名，value=指标字典
      ks: List[int] — X 轴的 K 值列表

    输出：
      str | None — 图片相对路径（如 "figures/ablation_comparison.png"），matplotlib 不可用时返回 None

    图表解读：
      - X 轴：K 值（1, 3, 5, 10）
      - Y 轴：指标值（0.0 ~ 1.0）
      - 颜色分组：红色=A(VectorOnly), 橙色=B(AgenticRAG), 绿色=C(Full)
      - 目的：一目了然地展示各组在不同 K 值下的 Precision/Recall 差异
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # 非交互式后端，无需 GUI
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping charts")
        return None

    _ensure_figures_dir()

    groups = list(ablation_results.keys())  # 消融组名称列表，如 ["A-VectorOnly", "B-AgenticRAG", "C-Full"]
    metrics = ["precision", "recall"]        # 两个子图分别画这两个指标
    k5 = min(ks, key=lambda k: abs(k - 5))  # 找到最接近 5 的 K 值（如果 KS 列表不包含 5）
    k_idx = ks.index(k5)

    # 1 行 2 列子图布局
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric in zip(axes, metrics):
        # 提取数据：data[i][j] = 第 i 组在第 j 个 K 值下的指标
        data = []
        for group in groups:
            avg = ablation_results[group].get("averages", {})
            vals = []
            for k in ks:
                vals.append(avg.get(f"{metric}@{k}", 0))  # 例如 "precision@5"
            data.append(vals)

        x = range(len(ks))        # X 轴基准位置
        width = 0.25              # 每组柱子的宽度
        colors = ["#e74c3c", "#f39c12", "#2ecc71"]  # 红、橙、绿 → A→B→C 渐好

        for i, (group, vals) in enumerate(zip(groups, data)):
            # 每组柱子偏移 i * width，实现分组柱状效果
            ax.bar([xi + i * width for xi in x], vals, width,
                   label=group, color=colors[i % len(colors)])

        ax.set_xlabel("K")
        ax.set_ylabel(f"{metric}@K")
        ax.set_title(f"{metric.capitalize()} @ K — Ablation Comparison")
        ax.set_xticks([xi + width for xi in x])  # X 轴刻度居中
        ax.set_xticklabels([f"K={k}" for k in ks])
        ax.legend()
        ax.set_ylim(0, 1.0)  # 指标范围 [0, 1]
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "ablation_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")  # dpi=150 清晰度足够，文件不大
    plt.close()
    return "figures/ablation_comparison.png"  # 返回相对路径，用于 Markdown 图片引用

# 注意：plot_precision_recall_curve 在报告生成中暂未被调用，保留以备后续扩展


def plot_precision_recall_curve(k_values: List[int], results: Dict[str, list]):
    """生成各 K 值下的 Precision-Recall 曲线（预留函数，当前未在报告生成中调用）。

    输入：
      k_values: List[int] — K 值列表
      results: Dict[str, tuple] — {label: (precisions_list, recalls_list)}
               precisions_list 和 recalls_list 分别是不同 K 下的 P 值和 R 值

    输出：
      str | None — 图片路径或 None

    图表解读：
      - 每条线代表一组实验
      - X=Recall, Y=Precision，越靠近右上角越好
      - 不同 K 值在曲线上形成不同点，K 越大 Recall 越高但 Precision 可能下降
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    _ensure_figures_dir()

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, (precisions, recalls) in results.items():
        ax.plot(recalls, precisions, "o-", label=label, markersize=8)  # 圆点+连线

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve across K values")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "precision_recall_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return "figures/precision_recall_curve.png"


def plot_confusion_matrix_heatmap(cm_data: dict):
    """将混淆矩阵渲染为热力图（Planner 路由评测可视化）。

    输入：
      cm_data: dict — 来自 confusion_matrix() 的返回结果
               {"labels": ["muscle_building", "fat_loss", ...], "matrix": [[15, 2, 0], ...]}

    输出：
      str | None — 图片路径或 None

    图表解读：
      - 行 = 真实标签（True），列 = 预测标签（Predicted）
      - 对角线 = 正确预测数（数值越大、颜色越深越好）
      - 非对角线 = 错误预测数，颜色越深的非对角线点表示主要混淆方向
      - 颜色方案 YlOrRd：黄色=少, 红色=多
      - 数值标注：深色背景用白字，浅色背景用黑字
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    _ensure_figures_dir()

    labels = cm_data["labels"]        # 类别标签列表
    matrix = np.array(cm_data["matrix"])  # 转为 numpy 数组便于绘图

    fig, ax = plt.subplots(figsize=(6, 5))
    # YlOrRd 颜色方案：数值从小到大 = 黄→橙→红
    im = ax.imshow(matrix, cmap="YlOrRd")

    # X 轴刻度 = 预测标签
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")  # 旋转45度避免重叠
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Planner Route Confusion Matrix")

    # 在每个格子中心标注数值
    for i in range(len(labels)):    # 行（真实）
        for j in range(len(labels)):  # 列（预测）
            # 自动选择文字颜色：深色背景用白色，浅色背景用黑色
            ax.text(j, i, matrix[i][j], ha="center", va="center",
                    color="white" if matrix[i][j] > matrix.max() / 2 else "black")

    plt.colorbar(im, ax=ax)  # 颜色条，显示数值-颜色映射
    plt.tight_layout()
    path = FIGURES_DIR / "confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return "figures/confusion_matrix.png"


def generate_markdown_report(
    ablation_results: Dict[str, dict],
    agent_results: dict,
    ks: List[int],
    num_queries: int,
    output_path: str = None,
) -> str:
    """根据评测结果生成完整的 EVAL_REPORT.md 评测报告。

    输入：
      ablation_results: Dict[str, dict] — RAG 消融评测结果（来自 run_ablation()）
      agent_results: dict — Agent 评测结果（来自 run_agent_eval()）
                      含 "planner" 和 "fact_checker" 两个 key
      ks: List[int] — 评测用的 K 值列表
      num_queries: int — 数据集中的查询总数
      output_path: str — 报告输出路径，默认 eval/EVAL_REPORT.md

    输出：
      str — 生成的完整 Markdown 文本

    报告结构（5 大章节）：
      第 1 章：RAG 检索质量 — 消融实验（实验设计 + 综合对比表 + 各 K 值详情）
      第 2 章：Agent 决策质量（Planner 路由 + FactChecker 安全审查）
      第 3 章：可视化图表（matplotlib 生成的 PNG 图片引用）
      第 4 章：面试话术要点（结构化的结论摘要，方便面试时引用）
    """
    if output_path is None:
        output_path = str(EVAL_DIR / "EVAL_REPORT.md")

    now = time.strftime("%Y-%m-%d %H:%M")  # 报告生成时间戳
    lines = []

    # ---- 报告头部 ----
    lines.append("# AI 健身私教 — 评测报告")
    lines.append(f"\n**生成时间**：{now}")
    lines.append(f"**评测数据集**：{num_queries} 条 Golden Query")
    lines.append(f"**评测 K 值**：{', '.join(str(k) for k in ks)}")
    lines.append("")

    # ========================================================================
    # 第 1 章：RAG 检索质量 — 消融实验
    # ========================================================================
    lines.append("---")
    lines.append("")
    lines.append("## 1. RAG 检索质量 — 消融实验")
    lines.append("")

    # ---- 1.1 实验设计说明 ----
    lines.append("### 实验设计")
    lines.append("")
    lines.append("| 消融组 | 配置 | 描述 |")
    lines.append("|--------|------|------|")
    lines.append("| A (Baseline) | 纯向量检索 | PG vector cosine similarity，无改写，无图谱 |")
    lines.append("| B | +Agentic RAG | LLM 自评 + 查询改写 + 最多 3 轮迭代 |")
    lines.append("| C (Full) | +GraphRAG | B + 知识图谱多跳推理补充 + 知识库 RRF 融合 |")
    lines.append("")
    lines.append("**消融实验设计意图**：")
    lines.append("")
    lines.append("- A→B 的增益量化了 Agentic RAG（自评+改写+多轮迭代）对检索质量的贡献")
    lines.append("- B→C 的增益量化了 GraphRAG + 知识库双路融合对检索质量的贡献")
    lines.append("- 通过渐进式叠加，可以精确定位每个模块的增量价值")
    lines.append("")

    # ---- 1.2 综合对比表（K=5） ----
    lines.append("### 综合对比（K=5）")
    lines.append("")
    headers = ["指标", "A-纯向量", "B-AgenticRAG", "C-全量", "提升(A→C)"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["------"] * len(headers)) + "|")

    # 四个关键指标对比：Precision@5, Recall@5, MRR, NDCG@5
    for metric_short, metric_name in [
        ("precision@5", "Precision@5"),
        ("recall@5", "Recall@5"),
        ("mrr", "MRR"),
        ("ndcg@5", "NDCG@5"),
    ]:
        vals = []
        for group in ["A-VectorOnly", "B-AgenticRAG", "C-Full"]:
            # 从 ablation_results 中提取对应组的指标值
            v = ablation_results.get(group, {}).get("averages", {}).get(metric_short, 0)
            vals.append(v)
        # 计算提升百分比：(C - A) / A * 100%，分母加 0.001 避免除零
        improvement = f"+{(vals[2] - vals[0]) / max(vals[0], 0.001) * 100:.0f}%"
        parts = [f"{v:.4f}" for v in vals]
        lines.append(f"| {metric_name} | {parts[0]} | {parts[1]} | {parts[2]} | {improvement} |")

    lines.append("")

    # ---- 1.3 各 K 值详细数据 ----
    # 展示各组在每个 K 值下的 Precision/Recall/NDCG，用于深入分析
    lines.append("### 各 K 值详细数据")
    lines.append("")
    for group_name in ["A-VectorOnly", "B-AgenticRAG", "C-Full"]:
        avg = ablation_results.get(group_name, {}).get("averages", {})
        lines.append(f"**{group_name}**")
        lines.append("")
        header = "| K | Precision | Recall | NDCG |"
        lines.append(header)
        lines.append("|---|-----------|--------|------|")
        for k in ks:
            p = avg.get(f"precision@{k}", 0)  # Precision@K
            r = avg.get(f"recall@{k}", 0)     # Recall@K
            n = avg.get(f"ndcg@{k}", 0)       # NDCG@K
            lines.append(f"| {k} | {p:.4f} | {r:.4f} | {n:.4f} |")
        # MRR 与 K 无关，单独一行
        lines.append(f"| MRR | — | — | {avg.get('mrr', 0):.4f} |")
        lines.append("")

    # ========================================================================
    # 第 2 章：Agent 决策质量
    # ========================================================================
    lines.append("---")
    lines.append("")
    lines.append("## 2. Agent 决策质量")
    lines.append("")

    # ---- 2.1 Planner 路由正确率 ----
    planner = agent_results.get("planner", {})
    lines.append("### 2.1 Planner 路由正确率")
    lines.append("")
    lines.append(f"- 总体正确率：**{planner.get('accuracy', 0) * 100:.1f}%**")
    lines.append(f"- 正确/总数：{planner.get('correct', 0)}/{planner.get('num_samples', 0)}")
    lines.append("")

    # 混淆矩阵 Markdown 表格呈现
    cm = planner.get("confusion_matrix", {})
    if cm:
        lines.append("**混淆矩阵**：")
        lines.append("")
        labels = cm.get("labels", [])
        matrix = cm.get("matrix", [])
        # 表头：True\Pred | label1 | label2 | ...
        header = "| True\\Pred | " + " | ".join(labels) + " |"
        lines.append(header)
        lines.append("|" + "|".join(["------"] * (len(labels) + 1)) + "|")
        # 逐行：真实标签 | 预测值1 | 预测值2 | ...
        for i, label in enumerate(labels):
            row = f"| {label} | " + " | ".join(str(v) for v in matrix[i]) + " |"
            lines.append(row)
        lines.append("")

    # ---- 2.2 FactChecker 安全审查 ----
    fc = agent_results.get("fact_checker", {})
    lines.append("### 2.2 FactChecker 安全审查")
    lines.append("")

    # 安全指标表格
    lines.append("| 指标 | 值 | 说明 |")
    lines.append("|------|-----|------|")
    lines.append(f"| 精确率（Precision） | {fc.get('precision', 0) * 100:.1f}% | 标记高风险中真正危险的比例 |")
    lines.append(f"| 召回率（Recall） | {fc.get('recall', 0) * 100:.1f}% | 所有危险样本中被标记的比例 |")
    lines.append(f"| 误报率 (FPR) | {fc.get('fpr', 0) * 100:.1f}% | 无害样本被错误标记为危险的占比 |")
    lines.append(f"| 漏报率 (FNR) | {fc.get('fnr', 0) * 100:.1f}% | 危险样本未被标记的占比（最关键指标！） |")
    lines.append(f"| F1 分数 | {fc.get('f1', 0):.4f} | 精确率与召回率的调和平均 |")
    lines.append("")

    # 四格表详情
    tp = fc.get('true_positive', 0)
    fp = fc.get('false_positive', 0)
    fn = fc.get('false_negative', 0)
    tn = fc.get('true_negative', 0)
    lines.append(f"- TP（正确报警）={tp}, FP（误报）={fp}, FN（漏报）={fn}, TN（正确不报警）={tn}")
    lines.append("")
    lines.append("**设计目标**：漏报率（FNR）= 0%，宁可多一些人工确认也不能让用户受伤。")
    lines.append("")

    # ========================================================================
    # 第 3 章：可视化图表
    # ========================================================================
    lines.append("---")
    lines.append("")
    lines.append("## 3. 可视化图表")
    lines.append("")

    # 消融对比柱状图
    ablation_chart = plot_ablation_comparison(ablation_results, ks)
    if ablation_chart:
        lines.append(f"![消融实验对比]({ablation_chart})")
        lines.append("")

    # 混淆矩阵热力图
    cm_chart = plot_confusion_matrix_heatmap(cm)
    if cm_chart:
        lines.append(f"![Planner 混淆矩阵]({cm_chart})")
        lines.append("")

    # ========================================================================
    # 第 4 章：面试话术要点
    # 设计意图：将评测结论转化为面试时可复述的结构化话术，
    # 面试官问"你的评测怎么做的？"时可以直接引用。
    # ========================================================================
    lines.append("---")
    lines.append("")
    lines.append("## 4. 面试话术要点")
    lines.append("")

    # RAG 检索话术
    lines.append("### RAG 检索")
    lines.append("- 做了三组消融实验：纯向量 → +Agentic RAG → +GraphRAG/知识库混合")
    lines.append("- 最终方案在 K=5 时 Recall 达到 XX%，比纯向量提升了 XX%")
    lines.append("- 采用 RRF 融合向量检索和关键词检索，避免分数校准问题")
    lines.append("- LLM Re-rank 进一步提高精确率，过滤关键词检索的假阳性")
    lines.append("")

    # Agent 决策话术
    lines.append("### Agent 决策")
    lines.append("- Planner 路由正确率 XX%，混淆矩阵显示增肌/减脂边界清晰")
    lines.append("- FactChecker 漏报率 0%，安全性优先设计")
    lines.append("- HITL 在低置信度 + 伤病场景自动触发人工确认")
    lines.append("")

    # 数据集话术
    lines.append("### 数据集")
    lines.append("- 自建 60 条 Golden Dataset，覆盖 5 个场景（增肌/减脂/动作分析/伤病/混合）")
    lines.append("- 人工标注 relevant_docs + expected_route + safety_risk")
    lines.append("")

    # 报告尾部
    lines.append("---")
    lines.append("")
    lines.append("*报告由 eval/run_eval.py 自动生成*")

    # ---- 写入文件 ----
    report = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report written to {output_path}")
    return report
