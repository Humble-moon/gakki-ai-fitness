# 面试展示版项目治理与优化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 gakki-ai-fitness 整理为事实可信、评测可解释、链路可演示、失败可复现、边界明确的 localhost 单用户面试展示版。

**Architecture:** 保留现有 FastAPI + 自研 Orchestrator + Planner/Retriever/Writer/FactChecker 架构，不迁移 LangGraph、不引入 Kafka、不新增登录体系。先建立机器可复核的事实和评测元数据，再提供 demo/full/test/e2e 运行入口，最后在 Retriever 前增加查询类型路由，让复杂检索按需触发并通过同一数据集的对照实验验收。

**Tech Stack:** Python 3、FastAPI、pytest、PostgreSQL/pgvector、Redis、Neo4j、MinIO、SSE、现有 LLM Provider、现有 Vanilla HTML/CSS/JS 前端、Makefile。

## Global Constraints

- 项目继续定位为 localhost 单用户演示，不实现登录注册、完整多用户权限、生产级 reviewer 闭环、Kafka、微服务拆分、LangGraph 迁移、医疗级认证或公网生产 SLA。
- 不删除或改写历史评测结果；历史结果只能增加 `historical`、`deprecated`、`not-comparable` 或 `design-only` 说明。
- 当前可直接核对的规模以 338 个种子动作、162 个知识 Markdown、206 条主 Golden、54 条知识子集、10 条条件上下文生成集和 68 条 RAGAS 子集为准；824 chunks 只有 README 声明时必须标注“待独立复核”。
- 不把条件上下文生成评测称为完整端到端准确率；不把 HITL 升级判定称为人工审核闭环；不使用“医疗级安全”“整体零漏报”“生产 SLA”等未经证实表述。
- 默认测试必须继续排除 `integration` 和 `live`；所有 opt-in 测试必须在命令输出和文档中显式标注。
- 每个修改任务都先补失败测试或验证脚本，再实现最小变更；每个任务独立运行相关测试并提交一次。
- 不写入、打印或提交任何真实 API key、token、cookie 或本地 `.env` 内容。

---

## 文件结构与职责

### 阶段一文件

- Create: `docs/project-fact-baseline.md` — 当前版本唯一事实基线、证据路径、统计命令、推荐对外话术。
- Create: `scripts/verify_project_facts.py` — 只读统计动作、知识文档、Golden 数据集和测试文件数量，并输出稳定 JSON。
- Modify: `README.md` — 删除/改写未核验的强事实，补充版本边界和事实基线入口。
- Create: `tests/test_project_fact_baseline.py` — 验证统计脚本结构、关键分层和 README 边界文案。

### 阶段二文件

- Create: `eval/README.md` — 评测数据集、实验类型、指标公式、版本记录和限制。
- Create: `eval/evaluation_manifest.json` — 现有结果文件的机器可读元数据索引。
- Create: `eval/scripts/validate_metrics.py` — 对结果 JSON 做样本数、指标范围和实验类型校验。
- Modify: `eval/EVAL_REPORT.md` — 标明 206 主集、历史状态和不可与其他报告直接比较的范围。
- Modify: `eval/KNOWLEDGE_EVAL_REPORT.md` — 标明 54 条知识子集，并解释多标签指标；禁止普通 Recall 出现未解释的大于 1 结果。
- Modify: `eval/E2E_EVAL_REPORT.md` — 将条件上下文生成评测重新命名和定位，不宣称全链路准确率。
- Modify: `eval/RAGAS` 相关说明或结果索引 — 标明 68 条子集而非 206 条全量。
- Create: `tests/test_eval_metadata.py` — 评测元数据和指标校验测试。

### 阶段三文件

- Create: `Makefile` — `make test`、`make e2e`、`make eval`、`make facts`、`make demo-help` 入口。
- Create: `scripts/run_demo.py` — 检查配置并启动 localhost demo；不绕过现有安全闸门。
- Create: `scripts/run_e2e.py` — 运行三条业务 SSE 链路的离线/opt-in 验证并输出 terminal 统计。
- Modify: `README.md` — 增加 demo/full/test/e2e 启动说明、失败恢复操作和依赖边界。
- Create: `tests/test_run_entries.py` — 验证 Makefile 和运行脚本的参数、默认模式及安全提示。
- Create: `tests/e2e/test_sse_business_flows.py` — 使用 fake provider/可替换依赖验证三条链路的 terminal 事件、失败和恢复。
- Optionally Create: `tests/e2e/test_browser_flows.py` — 仅在仓库已有 Playwright 依赖和可启动服务时增加真实浏览器回归；若环境不满足，保留明确 skip，不伪造通过结果。

### 阶段四文件

- Create: `src/rag/query_routing.py` — 定义 `QueryRoute` 枚举/数据类和 `classify_query(query, profile) -> QueryRoute`，只负责路由，不执行检索。
- Modify: `src/agents/retriever.py` — 根据 `QueryRoute` 选择最小必要检索路径，并保留统一结果结构、超时和失败降级语义。
- Modify: `src/rag/agentic_rag.py` — 将 Query Rewrite/Re-rank 作为条件触发路径，不改变已有默认安全行为。
- Create: `tests/test_rag/test_query_routing.py` — 精确动作、普通知识、多跳、伤病和低置信度路由测试。
- Create: `tests/test_agents/test_conditional_retrieval.py` — 验证每种路由只调用允许的检索器，并验证分支失败不会破坏整体结果。
- Create: `eval/scripts/compare_retrieval_routes.py` — 在同一数据集和指标定义下比较 baseline 与 conditional route，输出延迟、调用次数和质量指标。
- Create: `eval/results/README.md` — 说明结果目录按日期/代码版本保存，不覆盖历史结果。

---

## 阶段一：事实与版本口径治理

### Task 1: 建立机器可复核的事实统计脚本

**Files:**
- Create: `scripts/verify_project_facts.py`
- Test: `tests/test_project_fact_baseline.py`

**Interfaces:**
- Produces `python scripts/verify_project_facts.py --json`，输出包含 `data_counts`、`eval_counts`、`test_counts`、`runtime_contract` 和 `warnings`。
- `data_counts.seed_exercises` 必须为 JSON list 长度；`data_counts.knowledge_markdown` 必须为 `data/knowledge/**/*.md` 数量。
- `eval_counts` 至少包含主 Golden、知识子集、条件上下文生成集和 RAGAS 样本数；无法从文件可靠提取的值必须进入 `warnings`，不能硬编码为事实。

- [ ] **Step 1: 写失败测试**

```python
def test_fact_script_reports_current_dataset_counts():
    result = run_fact_script()
    assert result["data_counts"]["seed_exercises"] == 338
    assert result["data_counts"]["knowledge_markdown"] == 162
    assert result["eval_counts"]["golden_main"] == 206
    assert result["eval_counts"]["knowledge_subset"] == 54
    assert result["eval_counts"]["conditional_generation"] == 10
    assert result["eval_counts"]["ragas"] == 68
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_project_fact_baseline.py -q`

Expected: FAIL because `scripts/verify_project_facts.py` and `run_fact_script` do not exist.

- [ ] **Step 3: 实现只读统计**

实现 `main()` 和 `collect_facts(repo_root: Path) -> dict`：使用 `json.loads`、`Path.rglob` 和现有结果 JSON；对缺失文件返回结构化 warning，不创建或修改数据文件。

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_project_fact_baseline.py -q`

Expected: PASS；再运行 `python scripts/verify_project_facts.py --json`，确认 stdout 不包含 secret 或 `.env` 内容。

- [ ] **Step 5: 提交**

```bash
git add scripts/verify_project_facts.py tests/test_project_fact_baseline.py
git commit -m "test: add reproducible project fact inventory"
```

### Task 2: 编写唯一事实基线并修正 README 口径

**Files:**
- Create: `docs/project-fact-baseline.md`
- Modify: `README.md`
- Test: `tests/test_project_fact_baseline.py`

**Interfaces:**
- `docs/project-fact-baseline.md` 必须引用 Task 1 的命令和输出字段。
- README 必须链接到事实基线，并把 824 chunks 改为“README 历史声明，当前未独立复核”。

- [ ] **Step 1: 写失败契约测试**

```python
def test_readme_declares_localhost_single_user_boundary():
    readme = Path("README.md").read_text()
    assert "localhost 单用户" in readme
    assert "医疗级" not in readme
    assert "生产 SLA" not in readme


def test_fact_baseline_contains_evidence_and_status_columns():
    baseline = Path("docs/project-fact-baseline.md").read_text()
    for marker in ("事实", "证据路径", "状态", "统计日期", "对外话术"):
        assert marker in baseline
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_project_fact_baseline.py -q`

Expected: FAIL because the baseline document and the new README boundary text are absent.

- [ ] **Step 3: 写入基线和 README 修订**

基线至少列出：FastAPI/SSE endpoint、Planner/Retriever/Writer/FactChecker、自研 Orchestrator、GraphRAG、MCP、HITL 边界、Redis/cache、Provider/CircuitBreaker、338/162/206/54/10/68 数据规模、端口、测试命令、历史报告状态和设计-only 能力。

- [ ] **Step 4: 运行测试和静态检查**

Run: `pytest tests/test_project_fact_baseline.py -q && git diff --check`

Expected: PASS，无尾随空格；确认 README 不再把 824 chunks 作为当前强事实。

- [ ] **Step 5: 提交**

```bash
git add docs/project-fact-baseline.md README.md tests/test_project_fact_baseline.py
git commit -m "docs: establish project fact baseline"
```

---

## 阶段二：评测体系与统一验证入口

### Task 3: 建立评测元数据索引和指标校验器

**Files:**
- Create: `eval/README.md`
- Create: `eval/evaluation_manifest.json`
- Create: `eval/scripts/validate_metrics.py`
- Test: `tests/test_eval_metadata.py`

**Interfaces:**
- `python eval/scripts/validate_metrics.py --manifest eval/evaluation_manifest.json` 返回非零状态表示校验失败。
- manifest 每项至少包含 `id`、`kind`、`dataset_size`、`source_path`、`status`、`comparability`、`known_limits`。
- `kind` 取 `retrieval`、`generation_conditioned`、`safety`、`ragas_subset`、`load_local`。

- [ ] **Step 1: 写失败测试**

```python
def test_manifest_separates_main_and_knowledge_retrieval_sets():
    manifest = json.loads(Path("eval/evaluation_manifest.json").read_text())
    ids = {item["id"] for item in manifest["experiments"]}
    assert "retrieval_main_206" in ids
    assert "retrieval_knowledge_subset_54" in ids


def test_metric_validator_rejects_unbounded_recall():
    result = subprocess.run(
        [sys.executable, "eval/scripts/validate_metrics.py", "--manifest", "eval/evaluation_manifest.json"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_eval_metadata.py -q`

Expected: FAIL because the manifest and validator do not exist.

- [ ] **Step 3: 实现元数据和范围校验**

为历史结果添加 `historical` 或 `not-comparable`，不要覆盖结果 JSON。对普通 `precision`、`recall`、`ndcg`、`mrr` 校验 `0 <= value <= 1`；对多标签或特殊定义要求 manifest 明确 `metric_definition`，否则失败。

- [ ] **Step 4: 运行验证**

Run: `pytest tests/test_eval_metadata.py -q && python eval/scripts/validate_metrics.py --manifest eval/evaluation_manifest.json`

Expected: PASS；旧报告如果存在不可比或范围异常，只通过明确标注后校验，不删除原数据。

- [ ] **Step 5: 提交**

```bash
git add eval/README.md eval/evaluation_manifest.json eval/scripts/validate_metrics.py tests/test_eval_metadata.py
git commit -m "docs: index evaluation datasets and limits"
```

### Task 4: 修正评测报告命名、范围和测试统计口径

**Files:**
- Modify: `eval/EVAL_REPORT.md`
- Modify: `eval/KNOWLEDGE_EVAL_REPORT.md`
- Modify: `eval/E2E_EVAL_REPORT.md`
- Modify: RAGAS 结果说明所在文件
- Modify: `README.md`
- Test: `tests/test_eval_metadata.py`

**Interfaces:**
- 报告必须明确 `dataset_size`、评测类型、日期、模型、是否可与其他报告比较。
- 条件上下文生成报告必须出现“不是完整检索 E2E”的限制说明。
- README 不得声称“81 个测试用例全部通过”；必须说明默认 pytest marker 范围。

- [ ] **Step 1: 添加失败断言**

```python
def test_conditioned_generation_is_not_described_as_full_e2e():
    report = Path("eval/E2E_EVAL_REPORT.md").read_text()
    assert "条件上下文" in report
    assert "不是完整" in report or "不等同" in report


def test_readme_describes_default_pytest_scope():
    readme = Path("README.md").read_text()
    assert "not integration and not live" in readme or "integration/live" in readme
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_eval_metadata.py -q`

Expected: FAIL on at least one missing limitation marker.

- [ ] **Step 3: 只做说明性修订**

保留历史数值，补充实验日期、数据集、模型、样本量和不可比说明；将 10 条 E2E 改称条件上下文生成评测；将 68 条 RAGAS 标为子集；解释知识子集特殊 Recall 定义或从对外摘要中移除异常未解释数值。

- [ ] **Step 4: 运行测试**

Run: `pytest tests/test_eval_metadata.py -q && git diff --check`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add eval README.md tests/test_eval_metadata.py
git commit -m "docs: clarify evaluation scope and comparability"
```

### Task 5: 增加统一 Makefile 验证入口

**Files:**
- Create: `Makefile`
- Modify: `README.md`
- Test: `tests/test_run_entries.py`

**Interfaces:**
- `make facts` 调用事实统计；
- `make test` 调用默认离线测试；
- `make e2e` 调用核心 E2E，依赖缺失时明确失败或 skip，不伪造成功；
- `make eval` 调用评测 manifest 校验；
- `make demo-help` 输出 demo/full/test/e2e 边界。

- [ ] **Step 1: 写失败测试**

```python
def test_makefile_exposes_reproducible_targets():
    text = Path("Makefile").read_text()
    for target in ("facts:", "test:", "e2e:", "eval:", "demo-help:"):
        assert target in text
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_run_entries.py -q`

Expected: FAIL because `Makefile` does not exist.

- [ ] **Step 3: 创建最小入口**

Makefile 使用当前项目 `.venv/bin/python`（存在时）或 `python`，不自动安装依赖，不读取或打印 `.env`，并将 integration/live opt-in 明确写入帮助文本。

- [ ] **Step 4: 运行入口**

Run: `make facts && make eval && make test`

Expected: facts/eval PASS；默认测试保持现有 marker 语义；缺少真实依赖时不把 E2E 报成通过。

- [ ] **Step 5: 提交**

```bash
git add Makefile README.md tests/test_run_entries.py
git commit -m "chore: add reproducible verification commands"
```

---

## 阶段三：Demo 可复现化与业务 E2E

### Task 6: 增加运行模式解析和依赖边界

**Files:**
- Create: `scripts/run_demo.py`
- Modify: `README.md`
- Test: `tests/test_run_entries.py`

**Interfaces:**
- `python scripts/run_demo.py --mode demo --host 127.0.0.1 --port 8503`：仅启动面试展示模式并打印依赖边界。
- `--mode full`：要求真实依赖配置，不自动下载、安装或启动外部服务。
- `--check`：只检查配置和可达性，不启动服务。

- [ ] **Step 1: 写失败测试**

```python
def test_demo_mode_does_not_require_real_api_key():
    result = run_demo(["--mode", "demo", "--check"], env_without_keys())
    assert result.returncode == 0
    assert "demo" in result.stdout.lower()
    assert "localhost" in result.stdout.lower()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_run_entries.py::test_demo_mode_does_not_require_real_api_key -q`

Expected: FAIL because no mode-aware runner exists.

- [ ] **Step 3: 实现模式解析**

先复用现有应用入口和 fake provider/测试替换点；若当前应用没有安全的 fake provider 注入点，则 `--mode demo --check` 先提供无副作用配置检查，并显式说明完整业务演示仍需测试替身改造，不得伪造真实链路已成功。

- [ ] **Step 4: 运行验证**

Run: `pytest tests/test_run_entries.py -q && python scripts/run_demo.py --mode demo --check`

Expected: PASS，并显示端口、模式、外部依赖和单用户限制。

- [ ] **Step 5: 提交**

```bash
git add scripts/run_demo.py README.md tests/test_run_entries.py
 git commit -m "feat: add explicit demo and full run modes"
```

### Task 7: 增加三条 SSE 业务链路的离线 E2E

**Files:**
- Create: `scripts/run_e2e.py`
- Create: `tests/e2e/test_sse_business_flows.py`
- Modify: `Makefile`
- Test: `tests/e2e/test_sse_business_flows.py`

**Interfaces:**
- `python scripts/run_e2e.py --mode offline` 输出每条链路的 `status`、`terminal_event`、`event_counts` 和 `failure_reason`。
- 三条链路必须验证 `done`、`error`、`cancelled` 至少一种终止路径；失败时不能仅靠 EOF 判定成功。

- [ ] **Step 1: 写失败测试**

```python
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_three_business_streams_have_explicit_terminal_events(fake_app):
    for path, payload in BUSINESS_CASES:
        events = await collect_sse(fake_app, path, payload)
        assert events[-1].event in {"done", "error", "cancelled"}
        assert events[-1].event != "eof"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/e2e/test_sse_business_flows.py -q`

Expected: FAIL until the fake dependency fixture and runner are wired to the current application entry points.

- [ ] **Step 3: 接入 fake provider 和可替换存储**

只在测试 fixture 中替换 LLM/存储，不修改 production 默认安全策略；覆盖训练计划、动作分析、知识问答的成功、provider failure、取消和恢复场景；所有事件解析使用现有 SSE contract。

- [ ] **Step 4: 运行离线 E2E**

Run: `pytest -m e2e tests/e2e/test_sse_business_flows.py -q && python scripts/run_e2e.py --mode offline`

Expected: 每条链路输出明确 terminal；失败场景输出稳定 error/cancelled；无裸断成功。

- [ ] **Step 5: 提交**

```bash
git add scripts/run_e2e.py tests/e2e/test_sse_business_flows.py Makefile
 git commit -m "test: add offline SSE business flow checks"
```

### Task 8: 增加可选浏览器回归，不混淆静态契约测试

**Files:**
- Create: `tests/e2e/test_browser_flows.py`（仅已有 Playwright 依赖时）
- Modify: `README.md`
- Modify: `Makefile`

**Interfaces:**
- `make e2e-browser` 必须在服务未启动或 Playwright 未安装时给出明确 skip/failure reason。
- 浏览器测试至少覆盖：不出现内部 JSON、三条主要页面入口、移动端无横向溢出、失败后可重试。

- [ ] **Step 1: 先检查依赖和服务入口**

Run: `python -c "import importlib.util; print(bool(importlib.util.find_spec('playwright')))"`

Expected: 根据结果决定增加真实浏览器测试或只在 README 中记录当前未覆盖；不得为了通过测试伪造浏览器结果。

- [ ] **Step 2: 写最小浏览器契约**

```python
def test_plan_page_does_not_render_internal_writer_json(page):
    page.goto(BASE_URL)
    page.get_by_role("button", name="生成训练计划").click()
    expect(page.locator("body")).not_to_contain_text('"plan_id"')
```

- [ ] **Step 3: 运行并记录限制**

Run: `make e2e-browser`

Expected: 服务和依赖齐全时 PASS；否则输出 skip 原因，并在 README 区分“静态契约已覆盖”和“浏览器 E2E 未运行”。

- [ ] **Step 4: 提交**

```bash
git add tests/e2e/test_browser_flows.py README.md Makefile
 git commit -m "test: document optional browser regression coverage"
```

---

## 阶段四：条件化检索与可比实验

### Task 9: 定义 QueryRoute 和纯函数路由器

**Files:**
- Create: `src/rag/query_routing.py`
- Create: `tests/test_rag/test_query_routing.py`

**Interfaces:**

```python
class QueryRoute(str, Enum):
    EXACT_ACTION = "exact_action"
    KNOWLEDGE = "knowledge"
    HYBRID = "hybrid"
    GRAPH = "graph"
    INJURY_SENSITIVE = "injury_sensitive"
    FALLBACK = "fallback"

def classify_query(query: str, profile: dict | None = None) -> QueryRoute:
    ...
```

- [ ] **Step 1: 写失败测试**

```python
def test_classify_query_routes_exact_action():
    assert classify_query("杠铃深蹲怎么做") == QueryRoute.EXACT_ACTION


def test_classify_query_routes_injury_sensitive_query():
    assert classify_query("膝盖疼还能练深蹲吗") == QueryRoute.INJURY_SENSITIVE


def test_classify_query_routes_graph_question():
    assert classify_query("哪些动作同时训练胸和三头肌") == QueryRoute.GRAPH
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_rag/test_query_routing.py -q`

Expected: FAIL because `query_routing.py` does not exist.

- [ ] **Step 3: 实现最小确定性规则**

优先使用现有动作名称、伤病关键词和关系问题关键词；未知或低置信度返回 `FALLBACK`，不直接把低置信度分类当作安全结论；路由器不调用 LLM、数据库或网络。

- [ ] **Step 4: 运行测试和边界测试**

Run: `pytest tests/test_rag/test_query_routing.py -q`

Expected: PASS；额外验证空字符串、超长字符串、混合中英文和 profile 缺失不抛出未处理异常。

- [ ] **Step 5: 提交**

```bash
git add src/rag/query_routing.py tests/test_rag/test_query_routing.py
git commit -m "feat: add deterministic query route classification"
```

### Task 10: 接入 Retriever 的最小必要路径

**Files:**
- Modify: `src/agents/retriever.py`
- Modify: `src/rag/agentic_rag.py`
- Create: `tests/test_agents/test_conditional_retrieval.py`

**Interfaces:**
- `Retriever.retrieve(..., route: QueryRoute | None = None)` 保持现有调用兼容；route 缺失时使用现有安全默认路径。
- 每条 route 返回既有检索结果结构，不让上游感知底层检索器数量变化。
- 每个可选分支有独立 timeout、异常降级和请求取消传播。

- [ ] **Step 1: 写失败测试**

```python
@pytest.mark.asyncio
async def test_exact_action_route_skips_graph_and_rerank(retriever, mocks):
    result = await retriever.retrieve("杠铃深蹲", route=QueryRoute.EXACT_ACTION)
    mocks.graph.assert_not_called()
    mocks.rerank.assert_not_called()
    assert result.documents


@pytest.mark.asyncio
async def test_injury_route_keeps_safety_context(retriever):
    result = await retriever.retrieve("膝盖疼还能练深蹲吗", route=QueryRoute.INJURY_SENSITIVE)
    assert result.safety_sensitive is True
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_agents/test_conditional_retrieval.py -q`

Expected: FAIL because `route` is not accepted or route-specific calls are not implemented.

- [ ] **Step 3: 实现最小接入**

只在 Retriever 层选择已有能力：精确动作查询、向量/关键词、GraphRAG、AgenticRAG/Re-rank；保留现有错误事件和 FactChecker 输入，不在此任务中改写 Orchestrator。

- [ ] **Step 4: 运行回归**

Run: `pytest tests/test_agents/test_conditional_retrieval.py tests/test_agents/test_retriever.py -q`

Expected: 新测试和旧 Retriever 测试全部 PASS；确认默认 route 的行为没有变化。

- [ ] **Step 5: 提交**

```bash
git add src/agents/retriever.py src/rag/agentic_rag.py tests/test_agents/test_conditional_retrieval.py
git commit -m "feat: route retrieval by query type"
```

### Task 11: 运行可比检索实验并决定是否保留复杂路径

**Files:**
- Create: `eval/scripts/compare_retrieval_routes.py`
- Create: `eval/results/README.md`
- Modify: `eval/evaluation_manifest.json`
- Test: `tests/test_eval_metadata.py`

**Interfaces:**
- `python eval/scripts/compare_retrieval_routes.py --dataset eval/golden_dataset/queries.json --output eval/results/<run-id>.json`。
- 输出必须包含 `baseline`、`conditional`、`dataset_size`、`metrics`、`latency_ms`、`provider_calls`、`route_distribution`、`limitations`。
- 不允许覆盖历史结果；run id 必须包含日期或代码版本。

- [ ] **Step 1: 写失败校验**

```python
def test_route_comparison_result_contains_comparable_metadata(tmp_path):
    result = run_comparison_fixture(tmp_path)
    for key in ("baseline", "conditional", "dataset_size", "metrics", "latency_ms", "provider_calls"):
        assert key in result
    assert result["dataset_size"] > 0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_eval_metadata.py::test_route_comparison_result_contains_comparable_metadata -q`

Expected: FAIL because comparison script does not exist.

- [ ] **Step 3: 实现固定口径对比**

使用同一数据集、同一 K、同一指标实现和同一模型配置；记录每条 Query 的 route、耗时、调用次数、空结果和异常；对普通指标做 0–1 范围校验；不把历史 206/54 报告混入新结果。

- [ ] **Step 4: 运行实验并记录结论**

Run: `python eval/scripts/compare_retrieval_routes.py --dataset eval/golden_dataset/queries.json --output eval/results/conditional-route-baseline.json`

Expected: 输出可复核 JSON；如果指标无提升或延迟变差，报告必须保留更简单路径并记录“未证明收益”，不得强行宣传优化成功。

- [ ] **Step 5: 提交**

```bash
git add eval/scripts/compare_retrieval_routes.py eval/results/README.md eval/evaluation_manifest.json tests/test_eval_metadata.py
git commit -m "eval: compare conditional retrieval routes"
```

---

## 最终验证与交付检查

- [ ] 运行 `make facts`，确认事实基线数字和 warning 与文档一致。
- [ ] 运行 `make eval`，确认所有 manifest 项目有数据集、样本数、状态和限制。
- [ ] 运行 `make test`，记录默认 marker 范围和通过/跳过数量。
- [ ] 运行 `make e2e`，确认三条 SSE 链路的成功、失败和取消终止语义。
- [ ] 如环境支持，运行 `make e2e-browser`；否则在 README 记录未运行原因。
- [ ] 运行条件化检索对比实验，确认结果未混用历史实验。
- [ ] 运行 `git diff --check`。
- [ ] 检查 `git status`，不得提交 `.env`、密钥、缓存数据库或临时输出。
- [ ] 最终同步 README、事实基线和面试材料；面试材料只引用已证实或明确限定的数字。

## 预期提交顺序

1. `test: add reproducible project fact inventory`
2. `docs: establish project fact baseline`
3. `docs: index evaluation datasets and limits`
4. `docs: clarify evaluation scope and comparability`
5. `chore: add reproducible verification commands`
6. `feat: add explicit demo and full run modes`
7. `test: add offline SSE business flow checks`
8. `test: document optional browser regression coverage`
9. `feat: add deterministic query route classification`
10. `feat: route retrieval by query type`
11. `eval: compare conditional retrieval routes`

每个提交只包含对应任务的代码、测试和文档，避免把事实治理、运行模式和检索算法混成一个不可回滚的大提交。
