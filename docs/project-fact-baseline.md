# Project Fact Baseline

> 统计日期：2026-08-19。本文是当前版本的事实基线；无法由仓库文件独立复核的内容不作为强事实。

## 统计方法

使用只读脚本复核项目规模：

```bash
python scripts/verify_project_facts.py --json
```

输出字段固定包含 `data_counts`、`eval_counts`、`test_counts`、`runtime_contract` 和 `warnings`。脚本从 JSON 文件、Markdown 文件和测试文件读取计数，不读取 `.env`，不打印密钥，也不修改数据。

## 可复核事实

| 事实 | 证据路径 | 状态 | 统计日期 | 对外话术 |
|---|---|---|---|---|
| 种子动作 338 条 | `data/seed_exercises.json` 的 JSON list 长度 | 已复核 | 2026-08-19 | 当前仓库包含 338 条种子动作记录 |
| 知识 Markdown 162 篇 | `data/knowledge/**/*.md` 文件数 | 已复核 | 2026-08-19 | 当前仓库包含 162 篇知识 Markdown |
| 主 Golden 206 条 | `eval/golden_dataset/queries.json` 的 JSON list 长度 | 已复核 | 2026-08-19 | 主 Golden 数据集有 206 条记录 |
| 知识子集 54 条 | `eval/golden_dataset/knowledge_queries.json` 的 JSON list 长度 | 已复核 | 2026-08-19 | 知识评测子集有 54 条记录 |
| 条件上下文生成集 10 条 | `eval/e2e_results.json` 的 JSON list 长度 | 已复核 | 2026-08-19 | 条件上下文生成评测包含 10 条结果 |
| RAGAS 子集 68 条 | `eval/ragas_results.json` 的 `num_queries` | 已复核 | 2026-08-19 | RAGAS 结果声明 68 条查询 |
| localhost 单用户演示 | `runtime_contract` 及 `app/server.py` 路由 | 已复核为产品边界 | 2026-08-19 | 仅定位为 localhost 单用户演示，不代表公网多用户生产系统 |

## 架构与能力边界

仓库代码和文档可见 FastAPI + SSE 入口、自研 Orchestrator，以及 Planner、Retriever、Writer、FactChecker 分层；还包含 GraphRAG、MCP、Redis 语义缓存、Provider 熔断器和 HITL 升级判定。HITL 审核闭环（工件创建 → 中断暂停 → resolve 接口恢复 → 解除记录）已全链路打通并具备重启存活性（见下节）；GraphRAG、MCP 和部分高级能力应以实际配置与依赖可用性为前提。

当前运行契约为 `localhost:8503`，SSE 入口包括 `/api/generate-plan`、`/api/analyze-exercise` 和 `/api/ask-question`。默认测试通过 `pytest` 排除 `integration` 与 `live` 标记；需要外部服务或真实模型的评测必须显式 opt-in。

## 2026-09-03 工程加固（可复核）

| 事实 | 证据路径 | 状态 |
|---|---|---|
| 人工审核存储 SQLite 持久化（工件/解除记录/线程索引重启不丢） | `src/hitl/review_storage.py`、`tests/test_hitl/test_review_storage.py` | 已实现+已测试 |
| HITL 闭环默认启用持久化（`HITL_STORE_BACKEND=sqlite`，memory 供离线测试） | `src/graph/runtime.py` `_make_hitl_stores` | 已实现 |
| HTTP 安全中间件：可选 API key 认证、每 IP 滑动窗口限流、CORS 白名单；未配置任何令牌时 `/admin/*` 失败关闭（403） | `src/security/api_guard.py`、`tests/test_security/test_api_guard.py` | 已实现+已测试 |
| 动作名数据驱动加载（338 条，最长优先），替换 QA 链路硬编码列表 | `src/rag/exercise_catalog.py`、`tests/test_rag/test_exercise_catalog.py` | 已实现+已测试 |
| MCP 旧版资源（library/muscles/standards）数据库优先，与工具层口径一致 | `src/mcp/exercise_server.py` v2.1 | 已实现 |
| 动作检索双路 RRF 融合（共享 `src/rag/fusion.py`；`EXERCISE_FUSION=concat` 回退消融对照） | `src/rag/agentic_rag.py`、`tests/test_rag/test_fusion.py` | 已实现+已测试 |
| 检查点后端可选 PostgresSaver（`GRAPH_CHECKPOINT_BACKEND=postgres`，连接池、并发安全） | `src/graph/runtime.py`、`tests/test_graph/test_runtime_checkpointer.py` | 已实现+已测试+本机 PG 冒烟 |
| 语义缓存可选 pgvector ANN 扫描（`CACHE_SCAN_BACKEND=ann`，失败回退线性扫描） | `src/rag/semantic_cache.py`、`tests/test_rag/test_semantic_cache_ann.py` | 已实现+已测试+本机 PG 冒烟 |
| 消融重跑修复：2026-08-30 重跑实际只执行了 A 组（B/C 缺失被报告渲染为 0.0）；2026-09-03 重跑 A/B/D 三组并合并保存（部分组重跑不覆盖历史分区） | `eval/run_eval.py`、`eval/results.json`、manifest `retrieval_ablation_rerun_2026-09-03` | 已修复+已登记 |

**消融重跑结论（2026-09-03，170 条主评测集）**：MRR A-纯向量 0.4110 / B-AgenticRAG 0.4186 / D-混合RRF 0.3975，P@5/R@5/NDCG@5 三组持平。查询集偏关键词型，纯向量已近最优；混合融合无增益，增益集中在 Agentic 改写环节（+2%）。此结论与 2026-07-17 历史消融一致，作为诚实阴性结果保留。

## 未核验与历史结果

README 曾声明“824 chunks”。该数字目前仅是 README 历史声明，当前未独立复核；本基线不把它列为已复核事实。历史评测报告和 JSON 结果是可追溯的历史产物，不自动等同于当前版本的生产准确率、医疗级安全、整体零漏报或生产 SLA。

“设计-only”能力、架构图和历史报告中的指标，只有在对应实现、配置、数据版本和运行命令均可复核时，才可升级为当前事实。不要修改原始面试资料来补足证据。

## 推荐边界话术

可以说：项目是一个运行在 localhost 的单用户 AI 健身应用展示版，当前仓库包含上述可复核的数据和评测样本，并提供 FastAPI/SSE、多 Agent 编排和检索相关实现。不能说：它已经是公网多用户系统、完整人工审核闭环、医疗级产品或具有生产 SLA。
