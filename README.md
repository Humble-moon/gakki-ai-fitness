# Gakki AI Fitness

AI 健身私教 —— Multi-Agent 协作生成个性化训练计划，GraphRAG 伤病推理，RAG 五层演进的知识问答系统。

> 当前版本定位为 **localhost 单用户演示**，不承诺公网多用户、完整人工审核闭环或生产服务等级。可复核数字、证据路径和未核验口径见 [项目事实基线](docs/project-fact-baseline.md)。

## 架构

```
用户 → FastAPI (SSE 流式)
         │
    Orchestrator (自研编排引擎)
         │
    ┌────┼────┬──────────┐
    ▼    ▼     ▼          ▼
 Planner  Retriever  Writer  FactChecker
(任务拆解) (多源检索) (计划生成) (安全审查+HITL)
    │    │  │
    │    │  └── GraphRAG (Neo4j 知识图谱 338 动作节点多跳推理)
    │    └───── Agentic RAG (自评 + 改写 + 3 轮迭代)
    └────────── Skill 系统 (v3 插件化目录 + Planner v4 安全闸门)
         │
    ┌────┼────┬──────────┐
    ▼    ▼     ▼          ▼
 PostgreSQL  Neo4j   Redis   MinIO
 (pgvector)         (缓存+记忆)
```

## 技术栈

| 层级 | 技术 |
|------|------|
| Agent 框架 | 自研 Orchestrator（Planner → Retriever → Writer → FactChecker） |
| 协议 | FastMCP 完整协议实现（Tools + Resources + JSON-RPC 错误码） |
| 模型 | deepseek-chat + deepseek-reasoner 双模型架构 + 熔断器 |
| RAG | 向量检索（HNSW）+ 关键词检索 → RRF 融合 → LLM Re-rank |
| 知识图谱 | Neo4j + Cypher（动作→肌肉→器械→伤病 四类实体） |
| 向量化 | DashScope text-embedding-v4（1024 维，API 调用） |
| 缓存 | Redis 语义缓存（二级命中：精确 + 余弦相似度扫描） |
| 安全 | FactChecker 双重校验 + HITL（关键词 + embedding 语义双路） |
| 记忆 | 短期滑动窗口 + 长期记忆（带时间戳） |
| 评测 | 206 条 Golden Dataset + 三组消融实验 + E2E/RAGAS + Serving 压测 |
| 前端 | FastAPI + SSE + 暗黑工业风 HTML/CSS/JS |
| 部署 | Docker Compose（PostgreSQL + Neo4j + Redis + MinIO） |

## 核心功能

- **智能计划生成** — 输入身高体重/目标/场景，AI 先给个性化分析，Multi-Agent 流水线生成周训练计划，FactChecker 安全审查 + 修正回路
- **动作分析** — 输入动作名 + 训练感受，检索标准规范，诊断问题，给出改进方案
- **知识问答** — 自然语言健身问题，162 篇知识文档（90 自写 + 32 PubMed 翻译 + 40 扩展专题；824 chunks 为 README 历史声明，当前未独立复核）混合检索，RRF 融合 + Re-rank 精排，带来源引用

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 DeepSeek API Key 和 DashScope API Key

# 3. 启动基础服务（PostgreSQL/Neo4j/Redis/MinIO）
docker compose up -d

# 4. 灌入种子数据（338 个动作 → PG + Neo4j）
python -m src.main --seed

# 5. 摄入知识库（162 篇文档；chunk 数量以实际摄入输出为准 → pgvector）
python -m src.rag.knowledge_ingestion --dir data/knowledge

# 6. 启动服务
python app/server.py
# → 浏览器打开 http://localhost:8503
```

## 扩展知识库（可选）

```bash
# 从 PubMed 爬取运动科学文献
python scripts/fetch_knowledge.py

# 翻译改写为中文健身科普文章
python scripts/translate_knowledge.py

# LLM 批量扩展动作库 / 知识库
python scripts/expand_exercises.py
python scripts/expand_knowledge.py

# 增量摄入（只处理变更文件）
python -m src.rag.knowledge_ingestion --dir data/knowledge --incremental
```

## 项目结构

```
├── app/                          # FastAPI 后端 + 前端
│   ├── server.py                 # API 入口（SSE 流式）
│   └── static/index.html         # IRONMIND 暗黑工业风 UI
├── src/
│   ├── agents/                   # 四 Agent（Planner/Retriever/Writer/FactChecker）
│   ├── core/                     # Orchestrator 编排引擎
│   ├── mcp/                      # FastMCP 完整协议实现
│   │   ├── exercise_server.py    # MCP 工具（接 PG 数据库）
│   │   └── tool_registry.py      # 工具注册门面
│   ├── rag/                      # RAG 五层检索体系
│   │   ├── agentic_rag.py        # 自评改写迭代检索
│   │   ├── knowledge_search.py   # RRF 融合 + LLM Re-rank
│   │   └── semantic_cache.py     # Redis 语义缓存（二级命中）
│   ├── graphrag/                 # Neo4j 知识图谱检索
│   ├── llm/                      # LLMProvider 多模型管理 + 熔断器
│   │   └── provider.py           # chat + chat_stream + JSON mode
│   ├── memory/                   # 多轮对话 + 长期记忆（带时间戳）
│   ├── storage/                  # PG/Neo4j/Redis/MinIO 客户端
│   ├── skills/                   # Skill 系统（SkillLoader 自动发现）
│   ├── a2a/                      # A2A 消息总线（Task/Artifact）
│   ├── hitl/                     # HITL 人在回路（关键词 + embedding 语义）
│   └── models/                   # Pydantic 数据模型
├── skills/                       # Skill 插件目录（SKILL.md + references + scripts）
│   ├── muscle_building/
│   ├── fat_loss/
│   └── exercise_analysis/
├── tests/                        # 测试用例（数量以 pytest 实际收集为准）
├── scripts/                      # 数据工具
│   ├── fetch_knowledge.py        # PubMed 爬取
│   ├── translate_knowledge.py    # LLM 翻译改写
│   ├── expand_exercises.py       # LLM 扩展动作库
│   ├── expand_knowledge.py       # LLM 扩展知识库
│   └── create_hnsw_indexes.sql   # HNSW 索引迁移
├── eval/                         # 评测框架（206 条 Golden Dataset + 消融/E2E/RAGAS/压测）
├── data/knowledge/               # 健身知识库（162 篇文档；chunk 数量以实际摄入输出为准）
├── run_mcp_server.py             # MCP 独立服务器（stdio/SSE/HTTP）
├── docker-compose.yml
└── requirements.txt
```

## 验证事实

```bash
python scripts/verify_project_facts.py --json
```

该命令只读统计当前仓库的动作、知识文档、评测样本和测试文件；输出中的 `warnings` 会保留无法独立复核的历史口径。

## 评测与验证入口

默认 `pytest` 通过 `pytest.ini` 排除 `integration/live`；这两类测试必须显式 opt-in，不代表默认离线测试覆盖真实外部服务。评测元数据与指标范围可运行：

```bash
make facts
make eval
make test
```

`make e2e` 仅运行当前仓库可用的离线/核心 E2E 检查；缺少真实依赖时必须明确失败或跳过，不伪造成功。评测数据集、历史状态、可比性和限制见 [评测索引](eval/README.md)。
