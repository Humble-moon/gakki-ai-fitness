# 本地单用户演示可靠性修复实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在本地单用户演示模式下，修复 IRON MIND 的 SSE 假成功、依赖异常无提示、上传请求竞态和 session 生命周期问题，并用自动化测试与浏览器验收确认行为。

**Architecture:** 保留 FastAPI、同步 orchestrator generator、StreamingResponse 和单文件原生前端。后端在统一 SSE envelope 层捕获 generator 异常并输出唯一的 `done`/`error`/`cancelled` 终止事件；前端把 plan、analysis、QA 三套重复 parser 合并为共享 `consumeSse()`，并用 request generation、session snapshot、AbortController 管理所有流和上传请求。本轮明确只支持 localhost 单用户演示，不实现认证、多用户 ownership、真实删除 API 或 ingestion 重构。

**Tech Stack:** FastAPI；Starlette `StreamingResponse`；原生 HTML/CSS/JavaScript；ReadableStream；TextDecoder；pytest；HTTPX/ASGI 测试（依赖可用时）；Playwright 或浏览器 CDP。

## Global Constraints

- 本轮仅保证 localhost 单用户演示模式；不把 `session_id` 当作多用户安全边界。
- 不实现完整认证、管理员授权、文档真实删除、长期记忆 retention、ingestion 状态机或前端模块拆分。
- 保留现有业务事件名称；新增统一终止事件 `done`、`error`、`cancelled`。
- 只有收到 `done` 才能将流标记为成功；EOF 无终止事件必须视为断流错误。
- 动态内容继续使用 `textContent`、文本节点和 `replaceChildren`，禁止新增不可信数据 `innerHTML` sink。
- 依赖不可用时必须展示可理解的错误，不泄露 traceback、密钥或完整 prompt。
- 不因普通 tab 切换强制取消用户操作；新 session、页面离开和替换上传文件时必须使旧请求失效。

---

## 文件结构与职责

- Modify: `app/server.py`：统一 SSE 编码、异常映射、响应 headers 和本地演示错误消息。
- Modify: `app/static/index.html`：共享 SSE parser、请求上下文、上传取消/超时/竞态保护、session/page 生命周期和 UI 状态。
- Modify: `tests/test_web_frontend_contract.py`：补充终止协议、上传生命周期和 session invalidation 静态契约。
- Modify: `tests/test_web_api_smoke.py`：补充 fake orchestrator 的 SSE envelope 测试；若当前文件不适合承载，则创建 `tests/test_sse_streaming.py`。
- Create: `tests/test_sse_streaming.py`（仅当 smoke 文件缺少合适边界）：后端 generator 正常、异常和终止事件测试。
- Create: `tests/fixtures/sse_stream_cases.py`（可选，仅在测试数据重复时）：前端 parser 使用的 chunk、UTF-8、EOF 和 malformed event fixtures。
- Modify: `README.md`：明确本地单用户模式、Redis 等依赖要求和已知边界，不声称支持公网多用户。

---

### Task 1: 建立后端 SSE 终止协议和依赖错误映射

**Files:**
- Modify: `app/server.py:137-159,165-284`
- Modify: `tests/test_web_api_smoke.py` 或 Create: `tests/test_sse_streaming.py`

**Interfaces:**
- Produces `encode_sse_event(event: str, data: object) -> str`。
- Produces `_stream_events(generator, request_kind: str = "request")`，保证最多一个 terminal event。
- Terminal events: `done` with `success: true`; `error` with `code/message/retryable/stage`; `cancelled` with message。

- [ ] **Step 1: Write failing backend stream tests**

```python
import json


def decode_events(chunks):
    events = []
    for chunk in chunks:
        for line in chunk.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


def test_stream_emits_done_after_normal_generator():
    chunks = list(_stream_events(iter([("stage", {"label": "开始"})]), "plan"))
    events = decode_events(chunks)
    assert events[-1] == {"event": "done", "data": {"success": True}}


def test_stream_emits_structured_error_after_generator_failure():
    def broken():
        yield "writer_chunk", {"text": "部分内容"}
        raise ConnectionError("redis unavailable")

    events = decode_events(list(_stream_events(broken(), "plan")))
    assert events[-1]["event"] == "error"
    assert events[-1]["data"]["code"] == "DEPENDENCY_UNAVAILABLE"
    assert "redis unavailable" not in events[-1]["data"]["message"]


def test_stream_does_not_duplicate_terminal_event():
    generator = iter([("done", {"success": True})])
    events = decode_events(list(_stream_events(generator, "qa")))
    assert [event["event"] for event in events].count("done") == 1
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
cd /Users/mt/Desktop/gakki-ai-fitness/.worktrees/iron-mind-visual-redesign
.venv/bin/python -m pytest tests/test_sse_streaming.py -q
```

Expected: FAIL because the helper does not yet append a terminal event and does not map generator exceptions.

- [ ] **Step 3: Implement safe event encoding and exception envelope**

Implement JSON encoding with `ensure_ascii=False`. Wrap generator iteration in `try/except`; preserve already-emitted business events, classify Redis/connection failures as `DEPENDENCY_UNAVAILABLE`, classify cancellation separately, and classify unknown failures as `STREAM_FAILED`. Never expose exception text to the browser. If the generator already yielded `done`, `error` or `cancelled`, do not append another terminal event.

Add SSE response headers: `Cache-Control: no-cache`, `Connection: keep-alive`, and `X-Accel-Buffering: no` where supported by the response path.

- [ ] **Step 4: Run focused tests and syntax checks**

```bash
.venv/bin/python -m pytest tests/test_sse_streaming.py -q
.venv/bin/python -m compileall -q app src
```

Expected: PASS and no compile errors.

- [ ] **Step 5: Commit the backend protocol**

```bash
git add app/server.py tests/test_sse_streaming.py tests/test_web_api_smoke.py
git commit -m "fix: make SSE stream termination explicit"
```

---

### Task 2: Extract the shared frontend SSE consumer

**Files:**
- Modify: `app/static/index.html` around existing `streamFrom`, analysis stream and QA stream readers
- Modify: `tests/test_web_frontend_contract.py`
- Create: `tests/test_frontend_sse_parser.py` only if parser logic is extracted into a separately testable fixture; otherwise use Node syntax/static contracts

**Interfaces:**
- Produces `async function consumeSse(response, options)`.
- `options = { signal, isCurrent, onEvent, onTerminal, timeoutMs }`.
- Throws `SseProtocolError` for non-SSE responses, malformed terminal state, or EOF without terminal event.
- `onEvent(event, data)` receives business events only while `isCurrent()` is true.

- [ ] **Step 1: Add failing source contracts**

```python
def test_frontend_has_one_shared_sse_consumer():
    assert "async function consumeSse(" in HTML
    assert "TextDecoder" in HTML
    assert "decoder.decode()" in HTML
    assert "event === 'error'" in HTML
    assert "event === 'cancelled'" in HTML
    assert "doneSeen" in HTML


def test_stream_requires_terminal_event():
    start = HTML.index("async function consumeSse(")
    body = HTML[start:HTML.index("// ──", start)]
    assert "EOF" in body or "stream" in body
    assert "doneSeen" in body
    assert "terminal" in body
```

- [ ] **Step 2: Run the focused contract test and verify failure**

```bash
.venv/bin/python -m pytest tests/test_web_frontend_contract.py -q
```

Expected: FAIL until the shared consumer and terminal checks are present.

- [ ] **Step 3: Implement `consumeSse()`**

Before `getReader()`, require `response.ok`, `response.body`, and a `Content-Type` containing `text/event-stream`. Use a single `TextDecoder`, flush it with `decoder.decode()` on EOF, process residual complete records, and parse each event in a guarded block. Track `doneSeen`, `errorSeen`, and `cancelledSeen`; reject EOF without a terminal event. Treat `error` and `cancelled` as terminal and invoke `onTerminal`. Ignore stale business events but still drain/close the current reader safely.

Use an inactivity timeout with `AbortController`; preserve caller cancellation as `AbortError`, and turn timeout into a retryable protocol error. Do not auto-retry POST streams in this task.

- [ ] **Step 4: Route plan, analysis and QA through the shared consumer**

Remove the three duplicated `TextDecoder`/`reader.read()` loops. Keep their existing event-specific rendering branches inside `onEvent`. Handle `error` by showing the structured safe message and preserving QA draft. Handle `cancelled` as a cancelled state, not a generic error.

- [ ] **Step 5: Run contract and JavaScript checks**

```bash
.venv/bin/python -m pytest tests/test_web_frontend_contract.py -q
node --check /tmp/iron-mind-inline.js
```

Extract the inline script to `/tmp/iron-mind-inline.js` for syntax checking if needed; do not store generated files in the repository.

- [ ] **Step 6: Commit the shared parser**

```bash
git add app/static/index.html tests/test_web_frontend_contract.py
 git commit -m "fix: centralize frontend SSE parsing"
```

---

### Task 3: 修复 request generation、QA 新 session 和页面生命周期

**Files:**
- Modify: `app/static/index.html` around `activeRequests`, `newQaSession`, `activateTab`, request functions and initialization
- Modify: `tests/test_web_frontend_contract.py`

**Interfaces:**
- Produces `createRequestContext(kind, { sessionId, fileKey })`。
- Produces `invalidateRequest(kind)` and `invalidateAllRequests()`。
- Every context contains `controller`, `generation`, `sessionId`, and optional `fileKey`。

- [ ] **Step 1: Add failing lifecycle contracts**

```python
def test_requests_have_generation_and_session_identity():
    assert "generation" in HTML
    assert "sessionId" in HTML
    assert "invalidateAllRequests" in HTML


def test_new_session_invalidates_before_abort_cleanup():
    body = HTML[HTML.index("function newQaSession"):HTML.index("// ── Progress Bar", HTML.index("function newQaSession"))]
    assert "activeRequests.qa = null" in body or "invalidateRequest('qa')" in body
    assert "abort()" in body


def test_pagehide_invalidates_requests():
    assert "pagehide" in HTML
    assert "invalidateAllRequests" in HTML
```

- [ ] **Step 2: Implement request contexts**

Maintain monotonically increasing generations per request kind. On starting a request, capture the current `globalSessionId` or `qaSessionId`. `isCurrent()` must compare object identity, generation and session ID. `finally` may clear a slot only if it still owns that slot.

- [ ] **Step 3: Fix `newQaSession()` synchronously**

Invalidate and clear the QA slot before aborting the old controller. Increment the session generation, update `globalSessionId`/`qaSessionId`, reset QA view and upload UI, and ensure stale cancellation/error callbacks cannot touch the new view.

- [ ] **Step 4: Add page lifecycle cleanup**

Register `pagehide` to invalidate and abort all active requests. Do not add automatic cancellation on ordinary tab switching; preserve in-flight work while the user moves between visible features.

- [ ] **Step 5: Run contracts and syntax checks**

```bash
.venv/bin/python -m pytest tests/test_web_frontend_contract.py -q
node --check /tmp/iron-mind-inline.js
```

- [ ] **Step 6: Commit lifecycle protection**

```bash
git add app/static/index.html tests/test_web_frontend_contract.py
git commit -m "fix: invalidate stale frontend requests by session"
```

---

### Task 4: 修复上传请求竞态、超时和错误提示

**Files:**
- Modify: `app/static/index.html` upload handler and upload UI helpers
- Modify: `tests/test_web_frontend_contract.py`

**Interfaces:**
- Upload uses the same request context shape as streams, with `kind = 'upload'` and `fileKey`.
- Produces `uploadFile(file)`, `retryUpload()`, `renderUploadError(message, retryable)`。

- [ ] **Step 1: Add failing upload contracts**

```python
def test_upload_has_request_identity_and_status_validation():
    upload = HTML[HTML.index("async function uploadFile"):HTML.index("// ──", HTML.index("async function uploadFile"))]
    assert "AbortController" in upload
    assert "response.ok" in upload or "resp.ok" in upload
    assert "sessionId" in upload
    assert "generation" in upload
    assert "retry" in upload.lower()


def test_new_session_clears_upload_request():
    body = HTML[HTML.index("function newQaSession"):HTML.index("// ── Progress Bar", HTML.index("function newQaSession"))]
    assert "upload" in body
```

- [ ] **Step 2: Implement upload context and cancellation**

Capture file identity (`name`, `size`, `lastModified`) and session ID before constructing `FormData`. Abort/invalidate the previous upload when a new file is selected, when the file is removed, and in `newQaSession()`. Render completion only when request context is current and the file/session still match.

- [ ] **Step 3: Add response validation and timeout**

Check `resp.ok` before parsing. Parse JSON only when content type indicates JSON; otherwise use a safe generic message. Map 413 to a size message, 5xx to a retryable service message, and timeout to a retryable timeout message. Keep the last file reference only for the current page session and expose a Retry button without pretending that removal physically deletes server data.

- [ ] **Step 4: Run upload contracts and syntax checks**

```bash
.venv/bin/python -m pytest tests/test_web_frontend_contract.py -q
node --check /tmp/iron-mind-inline.js
```

- [ ] **Step 5: Commit upload lifecycle fixes**

```bash
git add app/static/index.html tests/test_web_frontend_contract.py
git commit -m "fix: guard uploads against stale sessions"
```

---

### Task 5: 明确本地演示依赖错误和运行说明

**Files:**
- Modify: `app/server.py` only if Task 1 does not cover startup/dependency classification
- Modify: `README.md`
- Modify: `tests/test_web_api_smoke.py`

**Interfaces:**
- Local demo errors use safe Chinese messages such as `请先启动 Redis（localhost:6380）后再试`。
- README startup order accurately lists required services and states that public/multi-user deployment is unsupported in this scope。

- [ ] **Step 1: Add README contract checks**

```python
def test_readme_declares_local_single_user_boundary():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "单用户" in readme or "本地" in readme
    assert "Redis" in readme
    assert "公网" in readme or "多用户" in readme
```

- [ ] **Step 2: Update startup and failure guidance**

Document Python 3.11, dependency startup order, Redis port `6380`, browser URL, and the fact that upload removal is UI/session-context removal rather than server physical deletion. State clearly that this branch does not provide authentication or multi-user isolation.

- [ ] **Step 3: Run smoke and compile checks**

```bash
.venv/bin/python -m pytest tests/test_web_api_smoke.py tests/test_sse_streaming.py -q
.venv/bin/python -m compileall -q app src
```

- [ ] **Step 4: Commit documentation and dependency messaging**

```bash
git add README.md app/server.py tests/test_web_api_smoke.py
git commit -m "docs: clarify local demo dependencies and limits"
```

---

### Task 6: 运行时验证与浏览器验收

**Files:**
- Modify: `tests/test_sse_streaming.py` or `tests/test_web_api_smoke.py` only if verification exposes a regression
- No product source changes unless a failing acceptance case requires a targeted fix

- [ ] **Step 1: Run the focused and full available test suite**

```bash
cd /Users/mt/Desktop/gakki-ai-fitness/.worktrees/iron-mind-visual-redesign
.venv/bin/python -m pytest tests/test_web_frontend_contract.py tests/test_web_api_smoke.py tests/test_sse_streaming.py -q
.venv/bin/python -m compileall -q app src
```

Record skips separately from failures; do not claim full integration coverage if Redis/PostgreSQL/LLM are unavailable.

- [ ] **Step 2: Start the local server with the project Python 3.11 environment**

```bash
.venv/bin/python app/server.py > /tmp/iron-mind-reliability.log 2>&1 &
```

Verify `GET /` returns 200. Do not start Docker or external services without user authorization; the local dependency-unavailable path is itself an acceptance case.

- [ ] **Step 3: Browser-check normal page and responsive layout**

Use the existing browser/Playwright path at widths 375, 768, 1100 and 1440. Verify no horizontal overflow, profile fields remain usable, QA composer remains visible, and the new session button remains on the right side of the QA title.

- [ ] **Step 4: Browser-check dependency failure behavior**

With Redis unavailable, submit a plan or QA request. Expected: a visible Chinese dependency error or structured failure state; no indefinite loading; QA draft remains available; no raw traceback appears in the page.

- [ ] **Step 5: Browser-check request races**

Using a local delayed mock or controlled network throttling:

1. Start QA, click New Session, immediately submit a new question; the new question must not be blocked.
2. Start upload A, then upload B; A must not overwrite B’s visible state.
3. Start upload, click New Session; old upload completion must not repopulate the new session.
4. End a stream without `done`; UI must show interrupted/error, not success.

- [ ] **Step 6: Inspect logs and report exact verification status**

Read `/tmp/iron-mind-reliability.log`; ensure no unhandled exception is exposed to the browser. Report which tests passed, which were skipped because dependencies were unavailable, and which browser scenarios were verified.

- [ ] **Step 7: Commit only targeted verification fixes**

If all checks pass without additional code, do not create a no-op commit. If a targeted regression fix is needed:

```bash
git add <verified-files>
git commit -m "fix: address local demo reliability regression"
```

---

## Self-review checklist

- SSE success requires `done`; EOF without terminal event is never success。
- Backend generator errors are safe `error` events, while detailed exceptions stay server-side。
- Frontend plan/analysis/QA share one parser and one terminal-event policy。
- Uploads use generation, session snapshot, file identity, AbortController and timeout。
- New session invalidates old QA and upload requests synchronously。
- Pagehide cancels all active requests；ordinary tab switching does not unexpectedly cancel work。
- Redis failure produces actionable local-demo messaging。
- No task adds authentication, multi-user claims, deletion guarantees or ingestion redesign。
- Tests cover both static contracts and runtime SSE behavior where dependencies allow。
