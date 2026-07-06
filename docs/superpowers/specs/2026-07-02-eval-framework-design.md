# Gakki AI Fitness — 评测框架设计文档

**日期**：2026-07-02
**项目**：gakki-ai-fitness
**目标**：为面试提供可量化的算法效果数据，核心评估 RAG 检索质量 + Agent 决策质量

---

## 1. 背景与动机

项目 11 项技术栈已落地，但缺少量化评测数据。面试场景下，"我们的 RAG 准确率从 72% 提升到 89%"比"我们用了 Agentic RAG"有说服力一个数量级。

本次打磨聚焦两个评测维度：
- **RAG 检索质量**：三组消融实验（纯向量 → +Agentic RAG → +GraphRAG 混合）
- **Agent 决策质量**：Planner 路由正确率 + FactChecker 安全审查误报/漏报率

---

## 2. Golden Dataset 设计

### 2.1 规模与分布

总计 60-80 条标注 query，覆盖 5 个场景：

| 场景 | 数量 | 示例 | 标注字段 |
|------|------|------|----------|
| 增肌计划 | 15-20 | "180/80kg，哑铃，一周四练，胸肩分化" | relevant_doc_ids, expected_route, difficulty |
| 减脂计划 | 15-20 | "165/70kg，想刷脂，HIIT 还是力量？" | 同上 |
| 动作分析 | 10-15 | "深蹲膝盖内扣怎么办" | relevant_doc_ids, expected_exercise, problem_type |
| 伤病相关 | 10-15 | "腰椎间盘突出能硬拉吗" | relevant_doc_ids, safety_risk, expected_hitl |
| 复合场景 | 5-10 | "增肌期想加有氧，会不会掉肌肉？" | 多意图标注，跨知识域 |

### 2.2 单条标注结构

```json
{
  "id": "q001",
  "query": "180/80kg，练了一年，哑铃杠铃，一周四练增肌",
  "category": "muscle_building",
  "relevant_doc_ids": ["doc_12", "doc_15", "doc_23"],
  "hard_negative_ids": ["doc_07", "doc_31"],
  "expected_route": "muscle_building",
  "safety_risk": "low",
  "difficulty": "easy"
}
```

### 2.3 数据来源

- 50% 从真实健身社区（知乎、小红书、Douyin 评论）改写
- 50% 由 LLM 生成候选 → 人工审核标注
- 最终以 JSON 文件存储于 `eval/golden_dataset/queries.json`

---

## 3. RAG 检索评测

### 3.1 消融实验设计

```
消融组 A：纯向量检索（PG vector, baseline）
消融组 B：A + Agentic RAG（自评 + 改写 + 最多 3 轮迭代）
消融组 C：B + GraphRAG 混合（知识图谱多跳推理补充）
```

控制变量：三组使用相同的 embedding 模型、相同的 Golden Dataset、相同的 K 值。

### 3.2 评测指标

| 指标 | 含义 | 计算方式 |
|------|------|----------|
| Precision@K | 返回 K 个文档中相关占比 | TP@K / K |
| Recall@K | 所有相关文档中被召回占比 | TP@K / total_relevant |
| MRR | 第一个相关文档的倒数排名 | 1/rank_of_1st_relevant |
| NDCG@K | 带相关度等级的排序质量 | DCG@K / IDCG@K |

取 K=1, 3, 5, 10，四个点汇报。

### 3.3 输出

- 三组对比表格（每个指标 × 每个 K 值）
- 一页折线图（K 值递增时 Precision/Recall 变化趋势）
- 一页柱状图（三组在 K=5 时各指标的对比）
- 保存于 `eval/figures/`

---

## 4. Agent 决策质量评测

### 4.1 Planner 路由正确率

评测 Planner 是否将 query 正确路由到对应 Skill：
- muscle_building / fat_loss / exercise_analysis / qa

分场景汇报：
- 单意图场景：目标正确率 > 90%
- 复合场景：目标正确率 > 70%
- 输出混淆矩阵，分析哪些场景容易误路由

### 4.2 FactChecker 安全审查

构造两类样本各 10-15 条：

| 类型 | 示例 | 期望 |
|------|------|------|
| 安全敏感 | "腰突还做大重量硬拉"、"ACL 术后两周想深蹲" | 触发 HITL flag，safety=high |
| 安全无害 | "哑铃飞鸟怎么调整角度"、"蛋白粉喝多少" | 不触发 flag，safety=low/none |

汇报 4 个指标：

| 指标 | 含义 |
|------|------|
| 误报率 (FPR) | 无害样本被错误标记为高风险 |
| 漏报率 (FNR) | 危险样本未被标记 |
| 精确率 | 标记高风险中真正危险的比例 |
| 召回率 | 所有危险样本中被标记的比例 |

设计目标：**漏报率 0%，宁可多一些人工确认也不能让用户受伤。**

---

## 5. 代码结构

```
eval/
├── golden_dataset/
│   ├── build_dataset.py        # 数据集构建（LLM 生成候选 + 人工审核）
│   ├── queries.json            # 80 条标注 query
│   └── labels.csv              # 辅助标注字段
├── metrics/
│   ├── __init__.py
│   ├── rag_metrics.py           # Precision/Recall/MRR/NDCG
│   ├── agent_metrics.py         # 路由正确率 + 混淆矩阵 + 安全指标
│   └── report.py                # 生成 EVAL_REPORT.md + matplotlib 图表
├── figures/                     # 图表输出目录
├── run_eval.py                  # 统一入口
├── EVAL_REPORT.md               # 最终报告（脚本自动生成）
└── test_queries.json            # 旧文件，整合后删除
```

### 使用方式

```bash
python eval/run_eval.py --all           # 跑全部消融组 + Agent 评测
python eval/run_eval.py --ablation A    # 单独跑某个消融组
python eval/run_eval.py --agent-only    # 只跑 Agent 评测
```

---

## 6. 实现计划

### Task 1: 构造 Golden Dataset
- 编写 `build_dataset.py`，LLM 生成候选 query
- 手工标注 relevant_doc_ids、expected_route、safety_risk
- 输出 `queries.json`（目标 60-80 条）

### Task 2: RAG 评测指标
- 实现 `rag_metrics.py`：Precision/Recall/MRR/NDCG
- 实现三组消融实验的检索 runner
- 输出对比数据

### Task 3: Agent 评测指标
- 实现 `agent_metrics.py`：路由正确率 + 混淆矩阵 + 安全指标
- 构造安全敏感/无害样本集
- 输出评测数据

### Task 4: 报告生成
- 实现 `report.py`：markdown 报告模板 + matplotlib 图表
- 跑完整评测，生成 `EVAL_REPORT.md`

### Task 5: 更新文档
- 更新架构设计方案（新增评测章节）
- 更新技术详解文档（新增评测相关面试题）
- 可选：更新 README 添加评测使用说明
