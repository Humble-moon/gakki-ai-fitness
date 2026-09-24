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

## 2026-09-22 执行脚手架（Harness）整理（可复核）

| 事实 | 证据路径 | 状态 |
|---|---|---|
| 执行预算集中到唯一事实源（重试 3 次 / 退避 2.0 / 重写 3 轮 / 递归 60 步 / 循环 10 步 / 50000 token） | `src/harness/config.py` | 已实现+已测试 |
| 配置与消费方（provider / graph.state / graph.runtime）取值一致性由测试钉住 | `tests/test_harness_contract.py` | 已实现+已测试 |
| ReAct 自主循环：预算硬上限、工具错误回喂模型、每步 checkpoint 回调 | `src/harness/loop.py`、`tests/test_harness_loop.py` | 已实现+已测试 |
| 提示词工具调用适配器（解析失败降级为最终答案而非空成功） | `src/harness/tool_calling.py`、`tests/test_harness_tool_calling.py` | 已实现+已测试 |
| 循环可直接驱动真实 `ToolRegistry`（9 个工具，离线可调用） | `tests/test_harness_tool_calling.py::TestRealToolRegistryIntegration` | 已实现+已测试 |
| 过程指标：成功率 / 预算触顶率 / 工具调用有效率 / 错误恢复率 / 续跑正确性 | `eval/metrics/harness_metrics.py`、`tests/test_harness_metrics.py` | 已实现+已测试 |
| 删除 `src/core/harness.py`（全仓库零引用，且其 `with_retry` 会无条件重试逻辑错误） | 该文件已删除；超时能力迁至 `src/harness/resilience.py` | 已清理 |
| CI：离线测试 + 事实核验 + 署名守卫（拒绝 AI 联合署名 trailer） | `.github/workflows/ci.yml` | 已实现+**已在 GitHub 实跑通过**（PR #1 连续 6 次，最近一次 Tests 1m26s / Attribution guard 4s，零 warning） |
| 裁剪依赖后安装耗时 1m58s → 30s（移除 torch 等约 2GB） | `requirements.txt`、`requirements-eval.txt` | 已实测（CI #5 步骤级计时） |
| 离线测试不再依赖 `.env`（新增占位凭据 conftest，只补缺失项） | `tests/conftest.py` | 已实现+双场景验证（有无 `.env` 均 534 passed） |

**能力边界（重要）**：`src/llm/provider.py` 不支持原生 function calling，自主循环依赖提示词协议驱动模型返回结构化 JSON，其可靠性**低于**原生 function calling；该路径默认关闭（`HARNESS_AGENT_LOOP`），也未在真实模型上端到端验证——循环与适配器的正确性由离线假模型测试保证，不等同于真实链路的成功率。

## 2026-09-22 知识问答自主循环路径（可复核）

| 事实 | 证据路径 | 状态 |
|---|---|---|
| 问答工具门面：知识库/动作库/图谱三类能力收敛为 3 个工具 | `src/harness/qa_tools.py`、`tests/test_qa_tools.py` | 已实现+已测试（含真实 ToolRegistry 联调） |
| 问答自主循环路径（提示词协议驱动，默认关闭） | `src/core/qa_agent.py`、`tests/test_qa_agent.py` | 已实现+已测试 |
| 统一入口按开关分发；**自主循环失败自动回退固定流水线**并告知用户 | `src/core/orchestrator.py::answer_question_stream`、`tests/test_qa_dispatch.py` | 已实现+已测试 |
| 安全检测与硬约束抽为**两条路径共用**（此前内联在固定流水线里，直测缺失） | `src/core/qa_safety.py`、`tests/test_qa_safety.py` | 已实现+已测试（含与原内联实现的逐字对照测试） |
| 两条路径过程开销对比脚本 | `eval/compare_qa_paths.py` | 已实现+**已实跑**（见下方边界） |
| 自主循环**已在真实模型（deepseek-chat）上端到端跑通**：2/2 完成，均 3 步，耗时 4.6/4.8s，模型自行选择 `search_knowledge` | `eval/qa_path_compare.json` | 已实测 |

**实跑发现的缺陷（已修）**：提示词协议下发出的 `role="tool"` 消息被 OpenAI 兼容服务端拒绝——`400 Messages with role 'tool' must be a response to a preceding message with 'tool_calls'`。原生 function calling 才有 `tool_calls` 配对，提示词协议没有。**假模型不校验消息结构，37 个离线用例全绿也没发现**，只有真实调用才暴露。修法：工具结果的消息格式改由适配器决定（`ToolCallingModel.format_tool_result` 钩子），提示词协议下改用 `role="user"` 并复述调用内容。回归保护见 `tests/test_harness_tool_calling.py::TestToolResultMessageShape`。

**对比结论的边界（重要）**：本次实跑时 PostgreSQL/Redis 未启动，**固定流水线 4/4 全部失败**（`psycopg2.OperationalError`），自主循环的工具调用也全部失败（4 次，`tool_call_efficiency=0.0`）——**因此本轮不构成有效的路径对比**，不得引用任何"自主循环比固定流水线快/慢"的结论。可确认的只有：①循环机制在真实模型上跑得通；②工具报错回喂模型后仍能产出连贯答案（`recovery_rate=1.0`），但该答案来自模型自身知识而非检索材料。完整对比需先 `docker compose up -d`。

**重构说明**：问答安全逻辑（关键词归一化 / 检测 / 安全提示词）原内联于 `answer_question_stream` 约 40 行中，现抽至 `src/core/qa_safety.py`。抽取后行为经逐字对照测试验证不变（`tests/test_qa_safety.py::TestExtractionParity`）。抽取动因：两条问答路径若各存一份安全实现会各自漂移，而用户看不出自己走的是哪条。

**历史修正**：`src/core/harness.py` 曾以文件头注释声称"被各 Agent 类通过 `@with_retry` 装饰其内部方法"，但全仓库零 import，实际重试逻辑在 `src/llm/provider.py` 另行实现。注释与事实不符的情况已随文件删除消除。

## 2026-09-24 安全门修复与上线准备（可复核）

本轮由一次**真实执行的对抗测试**驱动：不看代码声明，直接发真实请求看输出，打出并修复了若干缺陷。以下每项都可按证据路径复核。

### 安全门

| 事实 | 证据路径 | 状态 |
|---|---|---|
| **急症词表**独立于伤病词表（心血管/神经/呼吸/体征四类），命中即注入**禁止任何训练建议**的约束，且与普通安全话术**互斥**（伤病允许康复建议、急症禁止训练指导，叠加会自相矛盾） | `src/core/qa_safety.py::EMERGENCY_KEYWORDS`、`tests/test_safety_emergency.py` | 已实现+已测试+**端到端实测** |
| **Planner 安全闸门复用同一急症词表**（此前是第三张独立词表，同样缺急症词） | `src/agents/planner.py`、`tests/test_safety_emergency.py::TestPlannerSafetyGate` | 已实现+已测试（含"复用单一事实源"的接线断言） |
| **动作库校验**：计划中的动作名必须在 338 库内；比较前做名称归一化（去括号说明 + 剥一层姿势前缀），归后仍在库外则仅**提示**不阻断 | `src/core/plan_finalization.py::collect_unknown_exercises`、`tests/test_plan_quality_gates.py` | 已实现+已测试 |
| 规则引擎不再把训练日 `focus` 当伤病文本（此前 focus「胸肌与肩部」会让卧推被判与腰突冲突） | `src/hitl/review.py::_check_conflicts` | 已实现+已测试（含修复前会失败的反证） |
| 同「伤病+动作」的冲突合并为一条，触发词并列展示（`触发词: 腰/椎/间盘`）；端到端 16 条 → 4 条 | 同上 | 已实现+已测试+端到端实测 |
| **交付分流**：伤病冲突/语义匹配/danger/低置信度 → 扣下送审；**warning 级建议 → 照常交付** | `src/hitl/review.py::check`、`tests/test_advanced.py` | 已实现+已测试+端到端实测 |

**实测记录**：修复前「训练时突然胸闷、头晕，眼前发黑」答的是呼吸技巧、未建议就医；修复后答「请立即停止运动，尽快就医或拨打急救电话……我不能也不应该判断它严不严重」。交付分流实测：健康新手 `safe_delivered` + 9 条警告，腰突患者 `review_pending`（规则引擎检出冲突）。

**为什么 warning 级不阻断（重要）**：LLM 检查器几乎从不返回空 issue 列表，把 warning 当送审判据会让**所有计划都进审核队列**；而本项目 HITL 只实现了机制、未定义"审核人"角色，形成死锁——用户拿不到计划、也无人审核。**这不等于放宽安全**：伤病冲突、danger、低置信度三层拦截保持不变。

**为什么库外动作不阻断（重要）**：实测中被点名的多是命名变体（`保加利亚分腿蹲（扶支撑）` → 库里是 `保加利亚分腿蹲`）与真实但本库未收录的动作（`帕洛夫推`），而 LLM 生成的计划本就用不全库内动作名。一律拦截等于 100% 的计划交付不出去。真幻觉（如 `慢离心哑铃地板卧推`，由两个真名拼接）仍会被识别并提示。

### 性能与可运维性

| 事实 | 证据路径 | 状态 |
|---|---|---|
| 计划生成 **226s → 47s**：子任务并发（检索 22.8→3.9s，`executor.map` 保序）、重写改用 `REWRITE_MODEL`（单次 42~58s→5~8s）、重写循环收敛即停 | `src/agents/retriever.py`、`src/agents/writer.py`、`src/core/orchestrator.py`、`src/graph/routing.py` | 已实测（A/B 对照显示两模型重写输出逐条一致） |
| 循环条件与交付闸门对齐：由 `not is_safe or issues` 改为 `not is_safe`，因 `finalize_result` 决定交付的是 `is_safe` | 同上（两后端同步改） | 已实现+已测试 |
| **当日 LLM 成本上限**：Redis 计数（跨进程有效）、API 入口快速失败、排除法拦截（默认拦所有 POST，只放行不花钱端点）、Redis 故障降级放行 | `src/security/cost_guard.py`、`src/security/api_guard.py::CostLimitMiddleware`、`tests/test_security/test_cost_guard.py` | 已实现+已测试+端到端实测 |
| **个人数据导出与删除**：覆盖训练日志/长期记忆/会话；删除要求 `confirm=DELETE`，部分失败如实上报不谎报 | `app/server.py`、`src/memory/long_term.py::purge`、`tests/test_data_rights.py` | 已实现+已测试+端到端实测（删后 Redis 无残留、PG 主从表 0 条） |
| 长期记忆身份改用稳定的 `athlete_key`（原用身高体重哈希，**体重一变就换身份**，读不回偏好且无从删除） | `src/core/plan_finalization.py::long_term_user_key` | 已实现+已测试 |
| 计划草稿在安全检查完成前流式渲染，附「请勿照此训练」横幅 + 虚线视觉降级 + 隐藏操作入口 | `app/static/index.html::renderPlanDraft`、`tests/test_web_frontend_contract.py` | 已实现+已测试+headless 渲染验证 |

**实测成本参考**：单份计划约 **¥0.32**（36 次 LLM 调用，其中 reasoner 占约 91% 的开销）；单次问答约 ¥0.01。并发 5 路问答墙钟 12.2s（并行度 4.7x），语义缓存命中约 0.0s。

**动作库查询修复**：`target_muscles` 是 json 列却用 `ILIKE` 查询，PostgreSQL 无 `json ~~* unknown` 操作符 → 整条 SQL 报错并**静默降级到 3 条演示数据**（实测 `search_by_muscle("胸")` 只返回 1 条）。已改为 `jsonb_array_elements_text` 展开，修复后返回 20 条。该缺陷能长期存活，是因为 `tests/test_advanced.py` 与 `tests/test_mcp_v2.py` 的 autouse fixture 强制关闭数据库分支——整个 MCP 工具层测试都在 3 条演示数据上跑；现补 `tests/test_exercise_query_sql.py`（离线断言 SQL 形态 + 集成层连真实库）。

## 未核验与历史结果

**开发时间线（仓库不可独立复核，主动声明）**：本项目自 **2026-04** 起在本地开发，**2026-07-01 才初始化 Git 仓库**并开始产生提交历史。因此 `git log` 的最早提交（`4960e07`）晚于实际开工时间约三个月；且早期提交是在 07-01 上午批量落地的（最早三个提交相隔 3–4 分钟），不代表"当天才开始写第一行代码"。这一段属于开发者的一手陈述，**无法由仓库文件独立复核**，故按本文档标准不计入强事实，仅作背景说明——被问到时以此为准，不主张为可验证事实。

**知识块数量**：README 原声明“824 chunks”，该数字无独立复核依据，已于 2026-09-09 改为 **557**——取自 2026-08-30 扩展语料后的实际摄入输出，与评测 manifest 同源。需要说明的是：块数量存在 PostgreSQL 的 `knowledge_chunks` 表里，`scripts/verify_project_facts.py` 这类静态清单**核验不了它**（脚本能核验的是 `data/seed_exercises.json` 的 338 与 `data/knowledge` 的 162 篇，因为那些是磁盘上的文件）。脚本现在会主动报告“README 当前声称多少块、且该数字无法静态核验”，要重新测量就跑一次 `python -m src.rag.knowledge_ingestion`。历史评测报告和 JSON 结果是可追溯的历史产物，不自动等同于当前版本的生产准确率、医疗级安全、整体零漏报或生产 SLA。

**测试数量**：本文档与 README 都不写死测试用例数，一律以 `pytest` 实际收集为准。静态清单统计的是测试**文件数与函数数**（当前 51 文件 / 334 函数），**该口径只统计模块级 `def test_`，类内测试方法不计入**，因此显著低于实际规模——引用该数字时需说明这一点。参数化展开后的实际用例数更高（当前 `pytest -q` 为 534 passed）。论文截稿口径 295 用例（281 函数参数化展开）是历史事实，不随后续加固而改变。

“设计-only”能力、架构图和历史报告中的指标，只有在对应实现、配置、数据版本和运行命令均可复核时，才可升级为当前事实。不要修改原始面试资料来补足证据。

## 推荐边界话术

可以说：项目是一个运行在 localhost 的单用户 AI 健身应用展示版，当前仓库包含上述可复核的数据和评测样本，并提供 FastAPI/SSE、多 Agent 编排和检索相关实现。不能说：它已经是公网多用户系统、完整人工审核闭环、医疗级产品或具有生产 SLA。
