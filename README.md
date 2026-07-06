# Gakki AI Fitness

AI 健身私教 —— Multi-Agent 协作生成个性化训练计划，GraphRAG 伤病推理，RAG 五层演进的知识问答系统。

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
    │    │  └── GraphRAG (Neo4j 知识图谱 39 节点多跳推理)
    │    └───── Agentic RAG (自评 + 改写 + 3 轮迭代)
    └────────── Skill 系统 (关键词触发 + 策略模板)
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
| 模型 | deepseek-chat + deepseek-reasoner 双模型架构 |
| RAG | 向量检索 + 关键词检索 → RRF 融合 → LLM Re-rank |
| 知识图谱 | Neo4j + Cypher（动作→肌肉→器械→伤病 四类实体） |
| 向量化 | BGE-small-zh（512 维，本地 CPU 推理） |
| 缓存 | FAISS + Redis Semantic Cache |
| 安全 | FactChecker 双重校验 + HITL 人在回路 |
| 评测 | 60 条 Golden Dataset + 三组消融实验 |
| 前端 | FastAPI + SSE + 暗黑工业风 HTML/CSS/JS |
| 部署 | Docker Compose（PostgreSQL + Neo4j + Redis + MinIO） |

## 核心功能

- **智能计划生成** — 输入身高体重/目标/场景，AI 先给个性化分析，Multi-Agent 流水线生成周训练计划，FactChecker 安全审查
- **动作分析** — 输入动作名 + 训练感受，检索标准规范，诊断问题，给出改进方案
- **知识问答** — 自然语言健身问题，18 篇自写文档 + 62 篇 PubMed 文献混合检索，RRF 融合 + Re-rank 精排，带来源引用

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 DeepSeek API Key

# 3. 启动基础服务
docker compose up -d

# 4. 灌入种子数据（39 个标准动作）
python -m src.main --seed

# 5. 启动服务
python app/server.py
# → 浏览器打开 http://localhost:8503
```

## 扩展知识库（可选）

```bash
# 从 PubMed 爬取运动科学文献
python scripts/fetch_knowledge.py

# 翻译改写为中文健身科普文章
python scripts/translate_knowledge.py

# 摄入向量数据库
python -m src.rag.knowledge_ingestion --dir data/knowledge
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
│   │   ├── exercise_server.py    # @mcp.tool() + @mcp.resource()
│   │   └── tool_registry.py      # 7 Tools + 11 Resources 统一门面
│   ├── rag/                      # RAG 五层检索体系
│   │   ├── agentic_rag.py        # 自评改写迭代检索
│   │   ├── knowledge_search.py   # RRF 融合 + LLM Re-rank
│   │   └── semantic_cache.py     # FAISS + Redis 缓存
│   ├── graphrag/                 # Neo4j 知识图谱检索
│   ├── llm/                      # LLMProvider 多模型管理
│   │   └── provider.py           # chat + chat_stream + JSON mode
│   ├── memory/                   # 多轮对话 + 长期记忆
│   ├── storage/                  # PG/Neo4j/Redis/MinIO 客户端
│   ├── skills/                   # Skill 系统（增肌/减脂/动作分析）
│   ├── a2a/                      # A2A 消息总线（Task/Artifact）
│   ├── hitl/                     # HITL 人在回路
│   └── models/                   # Pydantic 数据模型
├── tests/                        # 46 个测试用例
├── scripts/                      # 知识库工具
│   ├── fetch_knowledge.py        # PubMed 爬取
│   └── translate_knowledge.py    # LLM 翻译改写
├── eval/                         # 评测框架（60 条 Golden Dataset）
├── data/knowledge/               # 健身知识库（80+ 篇文档）
├── run_mcp_server.py             # MCP 独立服务器（stdio/SSE/HTTP）
├── docker-compose.yml
└── requirements.txt
```
