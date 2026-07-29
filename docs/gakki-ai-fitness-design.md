# Gakki AI Fitness — 设计文档

**日期**：2026-07-01
**项目名称**：gakki-ai-fitness
**项目位置**：E:\gakki-ai-fitness
**来源**：gakki-ai-research 技术栈复用，业务重写

---

## 1. 项目概述

AI 健身私教，面向撸铁增肌和减脂塑形人群。用户输入身体数据、训练目标、可用器械，Multi-Agent 协作生成个性化训练计划。输入动作描述，AI 分析动作质量并给出改进方案。

### 用户场景

- "我 180cm/80kg，练了 1 年，想增肌，只有可调节哑铃，一周四练"→ 生成四周计划
- "今天练深蹲时感觉下背发紧，是不是姿势不对"→ 分析问题 → 给出改进建议
- "硬拉腰酸怎么回事"→ GraphRAG 知识图谱多跳推理 → 原因 + 方案

### 三个核心功能

| 功能 | 输入 | 输出 |
|------|------|------|
| 智能计划生成 | 身高体重、训练年限、目标、器械、每周天数 | 周计划（动作、组数、次数、休息） |
| 动作分析 | 动作名称 + 训练感受描述 | 问题诊断 + 改进方案 |
| 训练问答 | 自然语言问题 | 基于知识图谱的推理回答 |

---

## 2. 技术栈

| # | 技术 | 项目中的角色 |
|---|------|-------------|
| 1 | Multi-Agent | Planner → Retriever → Writer → FactChecker 四 Agent 协作 |
| 2 | Agentic RAG | 检索后自评 + 改写 + 最多 3 轮迭代 |
| 3 | GraphRAG | 动作-肌肉-器械-伤病知识图谱（Neo4j） |
| 4 | MCP 协议 | 健身标准动作库 MCP Server |
| 5 | Tool Calling | Agent 自主决策调内部检索或 MCP 工具 |
| 6 | HITL 人在回路 | 低置信度安全敏感建议暂停等人工确认 |
| 7 | Semantic Cache | FAISS + Redis，相似体型目标计划直接复用 |
| 8 | Structured Output | 训练计划用 JSON Schema 约束输出结构 |
| 9 | Skill 系统 | 增肌计划 / 减脂计划 / 动作分析 3 个 Skill |
| 10 | A2A 消息 | Writer ↔ FactChecker 轻量 Task/Artifact 通信 |
| 11 | 长期记忆 | 训练历史 + 偏好 + 伤病史 + HITL 反馈学习 |

### 基础设施（Docker Compose）

| 服务 | 镜像 | 端口 | 用途 |
|------|------|------|------|
| postgres | pgvector/pgvector:pg16 | 5432 | 向量检索 + 全文检索 + 用户数据 |
| neo4j | neo4j:5.20 | 7474/7687 | 知识图谱 |
| redis | redis/redis-stack:latest | 6380/8002 | Semantic Cache + 会话 |
| minio | minio/minio:latest | 9000/9001 | 导出文件存储 |

---

## 3. 系统架构

```
用户 (Streamlit Web)
         │
         ▼
┌─────────────────────────────────────────────┐
│               Orchestrator                   │
│         (状态机 + 流程编排)                    │
│                                              │
│   用户意图识别 → 路由到对应 Skill              │
│         │         │         │                │
│    增肌计划  减脂计划  动作分析  自由问答       │
└────────┬────────┬────────┬──────────────────┘
         │        │        │
         ▼        ▼        ▼
    Multi-Agent 流水线 (Agentic RAG + Tool Calling)
         │
    ┌────┼────┬────────┐
    ▼    ▼    ▼        ▼
  Planner Retriever Writer FactChecker
    │      │      │        │
    └──────┴──────┴────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│              服务层                           │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐        │
│  │Vector│ │Keywd │ │Graph │ │MCP   │        │
│  │Search│ │Search│ │Search│ │Server│        │
│  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘        │
│     │        │        │        │             │
│     ▼        ▼        ▼        ▼             │
│  ┌────────┐ ┌──────┐ ┌──────┐              │
│  │PostgreSQL│Neo4j │ │Redis │              │
│  │(pgvector│(Graph│ │(Cache│              │
│  │+tsvect) │ RAG) │ │+Stat)│              │
│  └────────┘ └──────┘ └──────┘              │
│     ┌──────────┐                            │
│     │  MinIO   │  (计划/数据文件)             │
│     └──────────┘                            │
└─────────────────────────────────────────────┘
```

---

## 4. Agent 设计

### Planner Agent
- **职责**：接收用户需求，分解为子任务，决定调用哪些 Agent
- **输入**：用户自然语言输入 + 用户画像（训练历史/偏好/伤病史）
- **输出**：任务分解计划（JSON），指定 Retriever 检索策略和 Writer 输出约束

### Retriever Agent
- **职责**：检索动作库和训练理论，自评质量，不够准就改写重试
- **输入**：Planner 的检索指令
- **输出**：检索到的动作列表 + 相关训练理论
- **检索源**：向量检索（pgvector）+ 关键词检索（tsvector）+ MCP 动作库
- **重试策略**：置信度 < 0.7 → 改写查询 → 最多 3 轮

### Writer Agent
- **职责**：根据检索结果生成结构化训练计划 / 动作分析报告
- **输入**：Retriever 输出 + 用户画像
- **输出**：JSON Schema 约束的训练计划（动作/组数/次数/重量范围/休息时间）
- **Skill 区分**：增肌计划模板 vs 减脂计划模板（不同的 Rep Range、休息时间等）

### FactChecker Agent
- **职责**：安全校验 + 事实核查
- **校验项**：
  - 动作是否适合该用户水平（初学者不推大重量复合动作）
  - 训练量是否合理（单次训练不超过 20 组）
  - 动作是否存在已知伤病风险
  - 器械约束是否满足（无杠铃时不能推荐杠铃动作）
- **置信度 < 0.7**：触发 HITL 暂停

---

## 5. 知识图谱设计（GraphRAG）

### 实体类型
- **Muscle**（肌肉）：名称、部位（上肢/下肢/核心）、功能
- **Exercise**（动作）：名称、类型（复合/孤立）、难度、所需器械
- **Equipment**（器械）：哑铃、杠铃、绳索、自重等
- **Injury**（伤病）：常见健身伤病及关联动作

### 关系类型
- `(Exercise)-[:TARGETS]->(Muscle)` — 动作练哪块肌肉
- `(Exercise)-[:REQUIRES]->(Equipment)` — 动作需要什么器械
- `(Exercise)-[:MAY_CAUSE]->(Injury)` — 动作可能导致的伤病
- `(Injury)-[:RECOVERED_BY]->(Exercise)` — 伤病康复推荐动作
- `(Exercise)-[:PROGRESSES_TO]->(Exercise)` — 动作进阶路线

### 示例查询
```
"深蹲膝盖疼"
→ MATCH (e:Exercise)-[:MAY_CAUSE]->(i:Injury {name: "膝盖疼痛"})
→ MATCH (i)-[:RECOVERED_BY]->(rehab:Exercise)
→ MATCH (e)-[:TARGETS]->(m:Muscle)
→ 返回：原因链 + 替代动作 + 康复建议
```

---

## 6. Skill 系统

| Skill | 触发条件 | 检索策略 | 计划模板 |
|-------|---------|---------|---------|
| 增肌计划 | 目标含"增肌" | 复合动作为主，Rep Range 6-12 | 四/五分化模板 |
| 减脂计划 | 目标含"减脂" | 复合动作 + HIIT，Rep Range 12-15 | 上下肢分化/全身模板 |
| 动作分析 | 输入含具体动作名 + 感受描述 | 检索该动作标准规范 + 常见错误 | 分析报告模板 |

---

## 7. 数据模型

### 用户画像（PostgreSQL）
```sql
user_profile (
  id, height, weight, training_years, goal (增肌/减脂),
  available_equipment (JSON), days_per_week, injuries (JSON),
  preferences (JSON), created_at, updated_at
)
```

### 训练计划（Structured Output）
```json
{
  "plan_id": "uuid",
  "user_id": "uuid",
  "goal": "增肌",
  "weeks": 4,
  "sessions_per_week": 4,
  "days": [
    {
      "day": 1, "focus": "胸+三头",
      "exercises": [
        {"name": "哑铃卧推", "sets": 4, "reps": "8-12", "rest": "90s", "notes": "..."}
      ]
    }
  ]
}
```

---

## 8. 启动命令

```bash
cd E:\gakki-ai-fitness
docker compose up -d                    # 启动 4 个服务
streamlit run app/streamlit_app.py      # Web 界面
python -m src.main --query "..."
```

---

## 9. 不做（v1 范围外）

- 视频/图片动作姿态分析（需 CV 模型，v2 考虑）
- 饮食热量计算和食谱生成（后续扩展）
- 实时运动追踪硬件接入
- 移动端 APP
