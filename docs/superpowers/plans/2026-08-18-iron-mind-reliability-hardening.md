# IRON MIND 可靠性与安全终态加固实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 localhost 单用户 Demo 边界内完成 Provider、Orchestrator、SemanticCache、DocumentStore 和测试链路的可靠性加固，使默认测试离线通过且不安全结果绝不持久化。

**Architecture:** 保留现有模块边界，以 `LLMUnavailableError` 统一表达全模型失败，以 Orchestrator 共享的 `_finalize_result()` 和 `_persist_if_safe()` 统一同步/流式安全终态。SemanticCache 采用版本化 profile fingerprint 并 best-effort；DocumentStore 通过 PGClient 事务完成文档和 chunks 的原子保存。

**Tech Stack:** Python 3.11+, FastAPI, pytest, OpenAI-compatible SDK, PostgreSQL/pgvector, Redis, NumPy。

## Global Constraints

- 默认 `pytest` 不访问 DeepSeek、DashScope 或其他外部 API。
- 真实 API 测试仅在 `RUN_LIVE_LLM_TESTS=1` 时运行，并只发送固定脱敏问题。
- 不把任何真实 API key 写入代码、测试、日志、文档或 Git。
- 只有 schema 有效、FactChecker 明确安全、无 issues、无需人工复核且 provider 未 degraded 的结果才能持久化。
- 缓存故障按 miss/best-effort 处理，不阻断主流程。
- 文档保存任一步骤失败都必须回滚文档和 chunks。
- 不进行公网多用户化或完整事件状态机重构。

---

### Task 1: Provider 全失败错误语义

**Files:**
- Modify: `src/llm/provider.py:127-370`
- Test: `tests/test_llm.py`
- Create: `tests/test_llm_failure_semantics.py`

**Interfaces:**
- Produces `LLMUnavailableError(RuntimeError)`，包含 `attempted_models: list[str]` 和 `errors: list[str]`。
- `LLMProvider.chat()` 全部模型失败时抛出 `LLMUnavailableError`；fallback 成功仍返回 `LLMResponse(degraded=True)`。
- `LLMProvider.chat_stream()` 建立连接阶段全部失败时抛出相同异常。

- [ ] **Step 1: 写失败测试**

在 `tests/test_llm_failure_semantics.py` 使用 fake clients 替换 provider 的 `_clients`，覆盖：

```python

def test_chat_raises_when_all_models_fail(provider, monkeypatch):
    monkeypatch.setattr(provider, "_call_api_with_retry", always_fail)
    with pytest.raises(LLMUnavailableError) as exc:
        provider.chat([{"role": "user", "content": "固定脱敏测试问题"}])
    assert exc.value.attempted_models
    assert exc.value.errors


def test_chat_marks_fallback_success_as_degraded(provider, monkeypatch):
    monkeypatch.setattr(provider, "_call_api_with_retry", fail_primary_then_success)
    response = provider.chat([{"role": "user", "content": "固定脱敏测试问题"}])
    assert response.content == "fake answer"
    assert response.degraded is True


def test_stream_raises_when_connection_cannot_start(provider, monkeypatch):
    monkeypatch.setattr(provider, "_open_stream", always_fail)
    with pytest.raises(LLMUnavailableError):
        list(provider.chat_stream([{"role": "user", "content": "固定脱敏测试问题"}]))
```

若现有实现没有 `_open_stream`，测试应替换实际的单模型流式调用辅助方法，而不是引入生产无关接口。

- [ ] **Step 2: 运行失败测试**

Run: `cd /Users/mt/Desktop/gakki-ai-fitness && pytest tests/test_llm_failure_semantics.py -q`

Expected: FAIL，因为当前全失败路径返回 `LLMResponse` 而不是抛出异常。

- [ ] **Step 3: 实现最小修改**

在 provider 顶部加入异常类；保留 retry、fallback 和 `degraded` 语义；将全失败的返回替换为：

```python
raise LLMUnavailableError(
    "所有配置的 LLM provider 均不可用",
    attempted_models=attempted,
    errors=errors,
)
```

流式连接阶段同样抛出，不生成“服务暂时不可用”的 token。日志只保留模型标识、异常类型和截断后的错误摘要。

- [ ] **Step 4: 运行通过测试**

Run: `pytest tests/test_llm.py tests/test_llm_failure_semantics.py -q`

Expected: PASS；若已有测试断言旧降级文本，更新为断言 `LLMUnavailableError`，不改变 fallback 成功测试。

- [ ] **Step 5: 提交**

```bash
git add src/llm/provider.py tests/test_llm.py tests/test_llm_failure_semantics.py
git commit -m "fix: fail closed when all llm providers are unavailable"
```

---

### Task 2: Orchestrator 统一安全闸门

**Files:**
- Modify: `src/core/orchestrator.py:71-356`
- Modify: `src/agents/output_validation.py`（若现有 schema helper 需要复用）
- Test: `tests/test_core/test_orchestrator.py`
- Create: `tests/test_core/test_orchestrator_safety_gate.py`

**Interfaces:**
- `_finalize_result(result: dict, checks: list[dict], rewrite_count: int, *, provider_degraded: bool = False) -> dict` 返回带 `warnings`、`requires_review`、`confidence`、`rewrite_count`、`_persistence_allowed` 的最终结果。
- `_persist_if_safe(profile: dict, query: str, result: dict, session_id: str | None = None) -> bool` 只在 `_persistence_allowed` 为真时写 cache/conversation/long-term。

- [ ] **Step 1: 写失败测试**

使用 stub `cache`, `conversation`, `long_term` 和 fake writer/fact checker，覆盖：

```python

def test_unsafe_sync_result_is_not_cached_or_memorized(orch, stores):
    result = orch._finalize_result({"days": []}, [{"is_safe": False, "issues": [{"issue": "危险动作"}]}], 3)
    assert result["_persistence_allowed"] is False
    assert orch._persist_if_safe({}, "q", result) is False
    stores.assert_no_calls()


def test_review_required_stream_result_is_not_persisted(orch, stores):
    result = orch._finalize_result({"days": []}, [{"is_safe": True, "issues": [], "requires_human_review": True}], 0)
    assert result["requires_review"] is True
    assert orch._persist_if_safe({}, "q", result, "session") is False


def test_safe_result_persists_once(orch, stores):
    result = orch._finalize_result({"days": []}, [{"is_safe": True, "issues": [], "requires_human_review": False, "confidence": .9}], 0)
    assert orch._persist_if_safe({}, "q", result) is True
    stores.assert_expected_safe_writes()


def test_degraded_provider_result_cannot_persist(orch, stores):
    result = orch._finalize_result({"days": []}, [{"is_safe": True, "issues": []}], 0, provider_degraded=True)
    assert result["_persistence_allowed"] is False
```

同时为同步 `generate_plan()` 和流式 `generate_plan_stream()` 注入 unsafe/review/degraded fake 结果，确认二者都经过同一 helper。

- [ ] **Step 2: 运行失败测试**

Run: `pytest tests/test_core/test_orchestrator_safety_gate.py -q`

Expected: FAIL，因为现有同步和流式路径直接 `cache.set()`、保存 conversation 和 long-term memory。

- [ ] **Step 3: 实现共享最终化与持久化**

在 Orchestrator 中集中计算最终状态，默认 fail-closed：缺失 `is_safe`、缺失 confidence、schema 校验失败或存在 issues 都不能持久化。替换同步路径第 150-168 行和流式路径第 325-356 行的直接写入。流式多轮路径在安全闸门通过前不调用 `set_plan_state()`、`add_turn()` 或 `save_preference()`。

Provider 返回 `degraded=True` 时将标记传给 `_finalize_result()`；`LLMUnavailableError` 直接让同步调用抛出、流式调用产生统一 error 事件，不写任何存储。

- [ ] **Step 4: 运行通过测试**

Run: `pytest tests/test_core/test_orchestrator.py tests/test_core/test_orchestrator_safety_gate.py -q`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/core/orchestrator.py src/agents/output_validation.py tests/test_core
git commit -m "fix: enforce orchestrator safety persistence gate"
```

---

### Task 3: SemanticCache 版本化与 best-effort

**Files:**
- Modify: `src/rag/semantic_cache.py:62-236`
- Test: `tests/test_advanced.py` 或现有 RAG 测试位置
- Create: `tests/test_rag/test_semantic_cache_reliability.py`

**Interfaces:**
- `_profile_fingerprint(profile: dict) -> str`
- `_make_key(profile: dict, query: str) -> str` 返回 `cache:fitness:v2:<fingerprint>:<hash>`。
- `get()` 在 Redis/embedding/JSON 异常时返回 `None`；`set()` 失败时不向业务抛出异常。

- [ ] **Step 1: 写失败测试**

覆盖 profile 隔离、版本拒绝、Redis get/set 异常、embedding 异常和损坏 JSON：

```python

def test_key_contains_version_and_profile_fingerprint(cache):
    assert cache._make_key({"goal": "增肌"}, "q").startswith("cache:fitness:v2:")
    assert cache._make_key({"goal": "增肌"}, "q") != cache._make_key({"goal": "减脂"}, "q")


def test_semantic_lookup_never_crosses_profile(cache, fake_redis):
    fake_redis.entries = [entry_for_profile("减脂", "q", {"answer": "wrong"})]
    assert cache.get({"goal": "增肌"}, "相似问题") is None


def test_redis_failure_is_cache_miss(cache, monkeypatch):
    monkeypatch.setattr(cache.redis, "get", lambda key: (_ for _ in ()).throw(ConnectionError()))
    assert cache.get({}, "q") is None


def test_set_failure_does_not_break_request(cache, monkeypatch):
    monkeypatch.setattr(cache.redis, "set", lambda *args, **kwargs: (_ for _ in ()).throw(ConnectionError()))
    cache.set({}, "q", {"answer": "safe"})
```

- [ ] **Step 2: 运行失败测试**

Run: `pytest tests/test_rag/test_semantic_cache_reliability.py -q`

Expected: FAIL，因为当前键没有显式版本，语义扫描不隔离 profile，异常会传播。

- [ ] **Step 3: 实现缓存升级**

将 profile 做 `json.dumps(..., sort_keys=True, separators=...)` 后 SHA-256 截断为 fingerprint；entry 增加 `schema_version`, `profile_fingerprint`, `query`。读取时拒绝版本不符和 fingerprint 不符。分别包裹 Redis 和 embedding 操作，精确命中失败后继续 miss；语义匹配 embedding 失败直接跳过；set 失败记录 warning 并返回。

- [ ] **Step 4: 运行通过测试**

Run: `pytest tests/test_rag/test_semantic_cache_reliability.py tests/test_core/test_orchestrator.py -q`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/rag/semantic_cache.py tests/test_rag/test_semantic_cache_reliability.py
git commit -m "fix: isolate and harden semantic cache entries"
```

---

### Task 4: DocumentStore 原子保存与回滚

**Files:**
- Modify: `src/storage/pg.py`
- Modify: `src/storage/document_store.py:38-81,146-168`
- Test: `tests/test_storage/test_pg.py`
- Create: `tests/test_storage/test_document_store_atomicity.py`

**Interfaces:**
- `PGClient.transaction()` 提供 context manager；上下文正常退出提交，异常退出回滚。
- `DocumentStore.save(...) -> int` 保持原签名；embedding、chunk insert 或 quota 清理失败时抛出原异常/明确 `DocumentStoreError`，不返回 id。

- [ ] **Step 1: 写失败测试**

使用 fake connection 记录 `begin/commit/rollback`，覆盖 embedding 失败、chunk insert 失败、旧文档清理失败和成功提交：

```python

def test_embedding_failure_rolls_back_document_and_chunks(store):
    store.embedder.embed = fail_on_second_chunk
    with pytest.raises(Exception):
        store.save("s", "x.txt", "txt", 10, "chunk one\n\nchunk two", 1, "x", True, "")
    assert store.db.transaction_state == "rolled_back"
    assert store.db.documents == []
    assert store.db.chunks == []


def test_success_commits_once(store):
    doc_id = store.save("s", "x.txt", "txt", 10, "safe text", 1, "x", True, "")
    assert doc_id == 1
    assert store.db.transaction_state == "committed"


def test_quota_cleanup_is_inside_same_transaction(store):
    store.seed_max_documents("s")
    store.save("s", "new.txt", "txt", 10, "safe text", 1, "new", True, "")
    assert store.db.transaction_begin_count == 1
```

- [ ] **Step 2: 运行失败测试**

Run: `pytest tests/test_storage/test_document_store_atomicity.py -q`

Expected: FAIL，因为现有 save 和 `_enforce_limit` 使用多个独立 execute，没有事务回滚。

- [ ] **Step 3: 增加事务并迁移保存流程**

在 `PGClient` 中提供事务上下文，复用现有连接池/连接获取方式；在 `DocumentStore.save()` 中把 `_enforce_limit`, INSERT 文档和所有 chunk 操作放在同一 context。embedding 失败不再 `continue`，而是抛出并触发回滚。`_enforce_limit` 接受事务连接/会话对象，避免打开独立连接。

- [ ] **Step 4: 运行通过测试**

Run: `pytest tests/test_storage/test_document_store_atomicity.py tests/test_storage/test_pg.py -q`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/storage/pg.py src/storage/document_store.py tests/test_storage
git commit -m "fix: make document ingestion atomic"
```

---

### Task 5: Live 测试显式 opt-in

**Files:**
- Modify: `pyproject.toml` 或 `pytest.ini`（按仓库现有配置选择）
- Create: `tests/live/test_live_llm.py`
- Modify: `.gitignore`（如需忽略本地 live 输出）

**Interfaces:**
- marker：`@pytest.mark.live`
- 默认未设置 `RUN_LIVE_LLM_TESTS=1` 时 skip。
- 运行命令：`RUN_LIVE_LLM_TESTS=1 pytest -m live -q`

- [ ] **Step 1: 写 live 测试**

```python
@pytest.mark.live
def test_configured_provider_answers_fixed_sanitized_prompt():
    if os.getenv("RUN_LIVE_LLM_TESTS") != "1":
        pytest.skip("live tests require RUN_LIVE_LLM_TESTS=1")
    response = LLMProvider().chat([
        {"role": "user", "content": "请用一句中文说明热身的目的，不要包含个人信息。"}
    ])
    assert response.content.strip()
    assert response.model != "none"
```

测试不得写 cache、conversation、long-term 或上传任何文档。

- [ ] **Step 2: 配置 marker 并运行默认测试**

Run: `pytest -q`

Expected: live 测试显示 skipped，其他测试不触网。

- [ ] **Step 3: 运行显式 live 测试（仅用户配置新 key 后）**

Run: `RUN_LIVE_LLM_TESTS=1 pytest -m live -q`

Expected: 仅在本地 `.env` 有效且网络可用时 PASS；失败时报告 provider 错误，不暴露 key。

- [ ] **Step 4: 提交**

```bash
git add tests/live pyproject.toml pytest.ini .gitignore
 git commit -m "test: add explicit opt-in live llm checks"
```

---

### Task 6: Health/ASGI 与全量验证

**Files:**
- Modify: `app/server.py`
- Modify: `src/health.py`
- Test: `tests/test_web_api_sse.py`
- Test: `tests/test_sse_streaming.py`
- Create: `tests/test_health_asgi.py`
- Modify: `README.md`

**Interfaces:**
- `/health/live` 始终返回本地进程存活状态。
- `/health/ready` 明确报告 provider/Redis/PostgreSQL 依赖状态，不把 degraded 当作正常 provider 成功。
- ASGI 测试使用 `httpx.ASGITransport`，不启动外部服务。

- [ ] **Step 1: 写 ASGI/health 失败测试**

```python
@pytest.mark.anyio
async def test_live_health_is_200(client):
    response = await client.get("/health/live")
    assert response.status_code == 200

@pytest.mark.anyio
async def test_ready_health_reports_dependency_failure(client, monkeypatch):
    monkeypatch.setattr("src.health.check_dependencies", lambda: {"provider": "down"})
    response = await client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
```

- [ ] **Step 2: 运行失败测试**

Run: `pytest tests/test_health_asgi.py tests/test_web_api_sse.py -q`

Expected: FAIL 或暴露当前 app wiring/依赖注入问题。

- [ ] **Step 3: 修复 app wiring 和真实 SSE 终态**

确认 `app/server.py` 只创建一次 FastAPI app 和路由；统一 SSE headers；provider 异常转为 error 事件，正常 generator EOF 不生成 done；取消请求生成 cancelled 事件。ASGI 测试 mock Orchestrator，不访问真实外部服务。

- [ ] **Step 4: 运行完整验证**

Run:

```bash
python -m compileall -q src app
git diff --check
pytest -q
npm test --if-present
npm run build --if-present
```

Expected: Python 编译、差异检查和默认 pytest 全部通过；不存在必须联网的默认测试。

- [ ] **Step 5: 提交**

```bash
git add app/server.py src/health.py tests README.md
git commit -m "test: verify health and asgi streaming reliability"
```

---

## 计划自检

- Provider 失败语义：Task 1。
- 同步/流式安全闸门：Task 2。
- SemanticCache 版本、fingerprint、best-effort：Task 3。
- DocumentStore 事务和回滚：Task 4。
- Live marker 与显式 opt-in：Task 5。
- Health、ASGI、SSE 和完整默认验证：Task 6。
- 所有任务均包含具体文件、接口、失败测试、运行命令和提交步骤；未使用 TBD/TODO/“适当处理”等占位表述。
