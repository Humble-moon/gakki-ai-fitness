# IRON MIND 视觉与关键体验改造实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保留 FastAPI + SSE + 原生前端路线的前提下，将 IRON MIND 改造成“精炼工业运动＋温暖专业教练”风格，并修复核心流式交互、移动端、可访问性和动态内容渲染问题。

**Architecture:** 本计划先在现有 `app/static/index.html` 内建立稳定的页面状态和渲染边界，再以增量方式改造 CSS、HTML 和 JavaScript，不引入 React/Vue，也不改变现有业务 SSE 事件名称。每个任务都以浏览器或自动化检查验证，最后补充轻量前端回归测试和响应式验收。

**Tech Stack:** FastAPI；原生 HTML/CSS/JavaScript；Server-Sent Events；浏览器 CDP；Python `pytest`；现有 `.venv` Python 3.11。

## Global Constraints

- 保留现有 FastAPI + SSE 接口和原生前端技术路线，不引入新的前端框架。
- 视觉方向以 A「精炼工业运动」为主，融入 B「温暖专业教练」。
- 主背景使用深黑蓝，主行动强调使用橙色，完成/安全状态使用柔和森林绿。
- 正文建议保持 14–16px，辅助说明 12–14px，metadata 11–12px。
- 中文负责主要任务和行动；英文仅用于品牌和少量技术状态。
- 所有用户、LLM、知识库和动作库数据都视为不可信，禁止直接拼接为可执行 HTML。
- 计划进度、教练建议、流式预览和最终结果必须是相互独立的 DOM 区域。
- 不能用低对比度、持续 glow 或无限动画作为主要信息层级。
- 必须支持 `prefers-reduced-motion`、键盘焦点和合理的 `aria-live` 播报。
- 本次不包含文件真实删除 API、管理员认证、CORS 收紧、上传任务队列、完整前端模块拆分和认证体系。

---

## 文件结构与职责

本次保持单页面入口，但为后续拆分预留清晰边界：

- Modify: `app/static/index.html`
  - CSS tokens、布局、组件样式、响应式规则、语义化 HTML、页面状态、SSE 渲染和安全 DOM 构造。
- Create: `tests/test_web_frontend_contract.py`
  - 不启动真实 LLM 的静态前端契约检查，确保关键 DOM id、ARIA 属性、安全渲染 helper 和状态容器不会回退。
- Create: `tests/test_web_api_smoke.py`
  - 仅在依赖可用且通过 monkeypatch 隔离编排器后，验证 FastAPI 首页与请求验证路径。
- Create: `docs/superpowers/plans/2026-08-17-iron-mind-visual-redesign.md`
  - 本实施计划。

实现过程中若发现 `index.html` 中已有函数名与计划略有差异，以当前文件实际函数为准，但必须保持本计划定义的行为和接口边界。

---

### Task 1: 建立前端回归基线与状态容器

**Files:**
- Modify: `app/static/index.html:1300-1705`
- Create: `tests/test_web_frontend_contract.py`

**Interfaces:**
- Produces stable DOM containers: `#app-shell`, `#profile-summary`, `#profile-editor`, `#plan-status`, `#plan-advice`, `#plan-stream`, `#plan-result`, `#qa-output`, `#qa-chat`, `#qa-welcome`, `#qa-sources`。
- Produces state helpers: `setViewState(view, state)`、`resetQaView()`、`setLoading(button, label, busy)`。

- [ ] **Step 1: Write the failing contract checks**

```python
from pathlib import Path

HTML = Path("app/static/index.html").read_text(encoding="utf-8")


def test_stable_stream_and_qa_containers_exist():
    for element_id in (
        "plan-status", "plan-advice", "plan-stream", "plan-result",
        "qa-output", "qa-chat", "qa-welcome", "qa-sources",
    ):
        assert f'id="{element_id}"' in HTML


def test_motion_preference_and_live_regions_exist():
    assert "prefers-reduced-motion" in HTML
    assert 'aria-live="polite"' in HTML
    assert 'role="alert"' in HTML
```

- [ ] **Step 2: Run the failing checks**

Run: `cd /Users/mt/Desktop/gakki-ai-fitness && .venv/bin/python -m pytest tests/test_web_frontend_contract.py -q`

Expected: FAIL because the stable containers and accessibility contract do not yet exist.

- [ ] **Step 3: Replace destructive result-root markup with stable containers**

Keep one persistent `#qa-output` shell and move welcome/chat/sources inside it. Add stable plan layers instead of allowing event handlers to replace the entire result root:

```html
<section id="plan-workspace" aria-live="polite">
  <div id="plan-status" class="status-panel" role="status"></div>
  <div id="plan-advice" class="stream-panel"></div>
  <div id="plan-stream" class="stream-panel"></div>
  <div id="plan-result" class="result-panel"></div>
  <div id="plan-error" class="error-panel" role="alert" hidden></div>
</section>

<section id="qa-output">
  <div id="qa-welcome" class="qa-empty-state"></div>
  <div id="qa-chat" class="qa-chat" hidden></div>
  <div id="qa-sources" class="qa-sources" hidden></div>
  <div id="qa-error" role="alert" hidden></div>
</section>
```

- [ ] **Step 4: Add minimal state helpers and wire existing tab changes through them**

`setViewState` must only toggle classes/`hidden`; it must not replace `#app-shell` or `#qa-output`. `resetQaView()` must clear only `#qa-chat`, `#qa-sources`, `#qa-error`, reset counters and show `#qa-welcome`.

- [ ] **Step 5: Run the contract checks**

Run: `.venv/bin/python -m pytest tests/test_web_frontend_contract.py -q`

Expected: PASS.

- [ ] **Step 6: Commit the stable state shell**

```bash
git add app/static/index.html tests/test_web_frontend_contract.py
git commit -m "feat: add stable frontend state containers"
```

---

### Task 2: 重做视觉 tokens、布局和页面信息层级

**Files:**
- Modify: `app/static/index.html:1-1400`
- Modify: `app/static/index.html:1400-1710`

**Interfaces:**
- Consumes the stable containers from Task 1。
- Produces CSS tokens and layout classes: `.app-shell`、`.profile-panel`、`.profile-summary`、`.profile-editor`、`.workspace`、`.workspace-header`、`.feature-tabs`、`.feature-panel`。

- [ ] **Step 1: Add a visual contract test for required design tokens and responsive rules**

```python
def test_visual_system_uses_iron_mind_tokens():
    assert "--bg-deep:" in HTML
    assert "--accent-orange:" in HTML
    assert "--status-safe:" in HTML
    assert "@media (max-width: 767px)" in HTML
    assert "@media (prefers-reduced-motion: reduce)" in HTML
```

- [ ] **Step 2: Replace the token block with the approved palette**

Use named variables rather than scattered literal colors:

```css
:root {
  --bg-deep: #090d12;
  --bg-panel: #111820;
  --bg-elevated: #17212b;
  --border: #263340;
  --accent-orange: #f2764a;
  --accent-orange-soft: #d98a6b;
  --status-safe: #6fa88b;
  --text-primary: #e9e6df;
  --text-secondary: #9ba7b2;
  --text-muted: #68727b;
  --focus-ring: #ffc2a9;
}
```

- [ ] **Step 3: Rebuild the desktop grid without changing business fields**

Use a constrained workspace instead of leaving a large unused black area:

```css
.app-shell {
  min-height: 100vh;
  display: grid;
  grid-template-columns: minmax(280px, 340px) minmax(0, 1fr);
}
.workspace {
  width: min(100%, 1120px);
  margin: 0 auto;
  padding: 32px clamp(20px, 4vw, 64px) 56px;
}
```

Group the existing profile inputs into summary/editor sections and move the main task title and Tab into the workspace header.

- [ ] **Step 4: Add mobile information architecture**

At `max-width: 767px`, make the app one column, place feature tabs before detailed profile fields, collapse `.profile-editor` by default, and make the summary actionable with a real button. Do not hide important sources or status information; move them into compact cards.

- [ ] **Step 5: Add reduced-motion and focus-visible rules**

```css
:where(button, input, select, textarea, [role="tab"]):focus-visible {
  outline: 3px solid var(--focus-ring);
  outline-offset: 3px;
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: .01ms !important;
    animation-iteration-count: 1 !important;
    scroll-behavior: auto !important;
    transition-duration: .01ms !important;
  }
}
```

- [ ] **Step 6: Run the visual contract test and inspect screenshots**

Run: `.venv/bin/python -m pytest tests/test_web_frontend_contract.py -q`

Then open `http://127.0.0.1:8503`, capture 1440px and 375px screenshots, and verify the task header, profile summary, tabs and primary action are visible without excessive empty space.

- [ ] **Step 7: Commit the visual system**

```bash
git add app/static/index.html tests/test_web_frontend_contract.py
git commit -m "feat: refresh IRON MIND visual system and layout"
```

---

### Task 3: 语义化 Tab、训练档案和核心按钮

**Files:**
- Modify: `app/static/index.html:1400-1710`
- Modify: `app/static/index.html:1870-1888`

**Interfaces:**
- Produces `activateTab(tabId)` and `toggleProfileEditor()` with keyboard-safe behavior。
- Tab ids: `tab-plan`/`panel-plan`、`tab-analysis`/`panel-analysis`、`tab-qa`/`panel-qa`。

- [ ] **Step 1: Add failing accessibility assertions**

```python
def test_tabs_have_semantics_and_associations():
    assert 'role="tablist"' in HTML
    assert 'role="tab"' in HTML
    assert 'role="tabpanel"' in HTML
    assert 'aria-selected="true"' in HTML
    assert 'aria-controls=' in HTML


def test_profile_labels_are_associated():
    assert '<label for="height">' in HTML
    assert '<label for="weight">' in HTML
```

- [ ] **Step 2: Convert current tab spans/divs to buttons**

Use:

```html
<div class="feature-tabs" role="tablist" aria-label="主要功能">
  <button id="tab-plan" role="tab" aria-selected="true" aria-controls="panel-plan" tabindex="0">训练计划</button>
  <button id="tab-analysis" role="tab" aria-selected="false" aria-controls="panel-analysis" tabindex="-1">动作分析</button>
  <button id="tab-qa" role="tab" aria-selected="false" aria-controls="panel-qa" tabindex="-1">知识问答</button>
</div>
```

- [ ] **Step 3: Implement roving tabindex and arrow-key navigation**

`activateTab(tabId, {focus = false})` must set exactly one selected tab, exactly one `tabindex="0"`, and exactly one visible panel. A `keydown` handler maps `ArrowLeft`/`ArrowUp` to the previous tab and `ArrowRight`/`ArrowDown` to the next tab; Home/End go to the first/last tab.

- [ ] **Step 4: Convert profile labels, scenes, chips and upload affordance to native controls**

Use `label[for]` for every input. Render scenes and example questions as `button type="button"`; use a real `label for="file-input"` for upload. Preserve existing classes so styling remains controlled by CSS.

- [ ] **Step 5: Implement profile summary/editor toggle**

`toggleProfileEditor()` toggles `hidden` and `aria-expanded` on the editor button, then focuses the first input when opened. The summary text must derive from current field values, not duplicated hard-coded values.

- [ ] **Step 6: Run contract checks and manual keyboard verification**

Run: `.venv/bin/python -m pytest tests/test_web_frontend_contract.py -q`

Browser verification: focus the first Tab, use arrow keys and Enter, then open the profile editor using keyboard only. Expected: focus remains visible and one panel changes without page reload.

- [ ] **Step 7: Commit accessible controls**

```bash
git add app/static/index.html tests/test_web_frontend_contract.py
git commit -m "feat: make navigation and profile controls accessible"
```

---

### Task 4: 安全 DOM 渲染和统一请求状态

**Files:**
- Modify: `app/static/index.html:1670-2425`
- Modify: `tests/test_web_frontend_contract.py`

**Interfaces:**
- Produces `appendText(node, value)`、`setText(node, value)`、`renderSourceList(node, sources)`、`setRequestState(kind, state)`。
- `setRequestState` states: `idle | loading | streaming | success | error | cancelled`。

- [ ] **Step 1: Add failing source-level safety checks**

```python
def test_dynamic_content_does_not_use_raw_stream_inner_html():
    forbidden = (
        "adviceArea.innerHTML +=",
        "answerArea.innerHTML +=",
        "source.title",
        "result.notes",
    )
    assert not any(token in HTML for token in forbidden)
    assert "textContent" in HTML


def test_request_state_supports_error_and_cancel():
    assert "setRequestState" in HTML
    assert "AbortController" in HTML
    assert "role=\"alert\"" in HTML
```

- [ ] **Step 2: Implement text-only rendering helpers**

```js
function appendText(node, value) {
  if (!node || value == null) return;
  node.appendChild(document.createTextNode(String(value)));
}

function setText(node, value) {
  if (!node) return;
  node.textContent = value == null ? '' : String(value);
}
```

Use these for advice chunks, writer chunks, answer chunks, exercise names, notes and source titles. Static markup may still use `innerHTML` only when the template contains no external data.

- [ ] **Step 3: Add AbortController per active stream**

Maintain:

```js
const activeRequests = {
  plan: null,
  analysis: null,
  qa: null,
};
```

Before starting a request, abort an existing request of the same kind. On new QA session and Tab changes, abort the relevant stream. Always clear the controller in `finally`.

- [ ] **Step 4: Normalize response status before reading SSE**

Before `getReader()`:

```js
if (!response.ok || !response.body) {
  const detail = await response.text().catch(() => '');
  throw new Error(detail || `请求失败（${response.status}）`);
}
```

Map HTTP 422 to a validation message, 5xx to a retryable service message, and stream termination without `done` to an interrupted-stream error.

- [ ] **Step 5: Implement request-state button and live-region updates**

`setRequestState(kind, state)` updates button text, `disabled`, `aria-busy`, the corresponding status region and the cancel action. Do not rely on color alone; every state has visible text.

- [ ] **Step 6: Run contract checks and inject hostile text manually**

Run: `.venv/bin/python -m pytest tests/test_web_frontend_contract.py -q`

In the browser console or a temporary local mock, pass values containing `<img src=x onerror=alert(1)>` and `&lt;tag&gt;`. Expected: text displays literally and no element is inserted.

- [ ] **Step 7: Commit safe rendering and request lifecycle**

```bash
git add app/static/index.html tests/test_web_frontend_contract.py
git commit -m "fix: harden streaming rendering and request states"
```

---

### Task 5: 修复训练计划流式区域和结果卡片

**Files:**
- Modify: `app/static/index.html:1740-2140`
- Modify: `tests/test_web_frontend_contract.py`

**Interfaces:**
- Consumes `setRequestState` and stable plan containers from Tasks 1 and 4。
- Produces `updatePlanStage(eventData)`、`renderPlanResult(plan)`。

- [ ] **Step 1: Add event-to-layer contract checks**

```python
def test_plan_events_target_stable_layers():
    for event_name in ("stage", "advice_chunk", "writer_chunk", "factcheck_done", "done"):
        assert event_name in HTML
    assert "plan-status" in HTML
    assert "plan-advice" in HTML
    assert "plan-stream" in HTML
    assert "plan-result" in HTML
```

- [ ] **Step 2: Replace `resultArea.innerHTML = ''` in the stage handler**

The stage handler may clear only the current stream preview when a new generation starts; it must never clear `#plan-workspace`, `#plan-status`, or other layers. Set progress text and a data percentage on the progress bar instead.

- [ ] **Step 3: Implement `updatePlanStage`**

Map the known stage labels to five visible progress states: 分析、规划、检索、生成、安全检查。Unknown stage text remains visible but does not break progress. Update `role="status"` text and `aria-valuenow`.

- [ ] **Step 4: Implement safe `renderPlanResult`**

Construct day cards with `document.createElement`, set all dynamic fields with `textContent`, and render each exercise as:

```text
动作名称
3 组 · 8–12 次 · 休息 90 秒
注意事项
```

Use desktop grid columns only above 768px; use stacked mobile cards below that width.

- [ ] **Step 5: Verify plan state transitions with a local mock stream**

Use a browser-only mock that dispatches `stage → advice_chunk → writer_chunk → factcheck_done → done`. Expected: progress, advice, preview and final result remain visible simultaneously; no layer disappears.

- [ ] **Step 6: Commit plan stream repair**

```bash
git add app/static/index.html tests/test_web_frontend_contract.py
git commit -m "fix: preserve plan progress and render result layers"
```

---

### Task 6: 修复问答新会话、动作分析和来源卡片

**Files:**
- Modify: `app/static/index.html:2160-2425`
- Modify: `tests/test_web_frontend_contract.py`

**Interfaces:**
- Consumes `resetQaView`, `appendText`, `renderSourceList`, `setRequestState`。
- Produces stable QA behavior: `newQaSession()`、`appendQaTurn(role, text)`、`renderSourceList(node, sources)`。

- [ ] **Step 1: Add regression assertions for new-session behavior**

```python
def test_new_session_does_not_replace_qa_shell():
    start = HTML.index("function newQaSession")
    end = HTML.find("function ", start + 10)
    body = HTML[start:end if end != -1 else None]
    assert "resetQaView()" in body
    assert "qa-output.innerHTML" not in body
```

- [ ] **Step 2: Rewrite `newQaSession()` to reset state only**

Generate a new session id, call `resetQaView()`, update counters and focus the question textarea. Do not replace the contents of `#qa-output` with a different DOM shape.

- [ ] **Step 3: Render QA turns with DOM nodes**

`appendQaTurn(role, text)` creates a turn element, creates a text node for user/assistant text, appends it to `#qa-chat`, unhides the chat, and keeps the scroll position. Do not assign external text to `innerHTML`.

- [ ] **Step 4: Render source citations safely and responsively**

`renderSourceList` creates a `<details>` wrapper on narrow screens and a regular card group on desktop. Titles, filenames and scores use `textContent`; an empty source list renders “本次回答未找到直接相关来源”。

- [ ] **Step 5: Verify QA regression in the browser**

Click “新对话”, type a question, and submit. Expected: no null `#qa-chat` error, user turn appears, assistant stream area appears, and new sources render below the answer. Repeat after switching away from and back to the QA tab.

- [ ] **Step 6: Verify action-analysis states**

Submit the action-analysis form with the API unavailable. Expected: button enters loading, then shows a readable error with retry; no stale stream writes into a different tab.

- [ ] **Step 7: Commit QA and analysis fixes**

```bash
git add app/static/index.html tests/test_web_frontend_contract.py
git commit -m "fix: stabilize QA sessions and analysis rendering"
```

---

### Task 7: 移动端计划卡片、来源与动效验收

**Files:**
- Modify: `app/static/index.html:400-1400`
- Create: `tests/test_web_api_smoke.py`

**Interfaces:**
- Consumes all frontend behavior from Tasks 1–6。
- Produces verified responsive behavior at 375px, 768px and 1440px。

- [ ] **Step 1: Add FastAPI smoke-test scaffold without invoking real agents**

```python
import pytest
from fastapi.testclient import TestClient


@pytest.mark.skipif(
    __import__('importlib').util.find_spec('fastapi') is None,
    reason='FastAPI is not installed',
)
def test_homepage_serves_iron_mind(monkeypatch):
    from app.server import app
    response = TestClient(app).get('/')
    assert response.status_code == 200
    assert 'IRON MIND' in response.text
```

If importing `app.server` initializes unavailable external clients, move the import behind the test and monkeypatch only that initialization; do not add production behavior solely to satisfy the test.

- [ ] **Step 2: Replace narrow-screen four-column exercise rows**

At `max-width: 767px`, render each exercise as a card with name, a single readable metadata line and notes. Desktop keeps a four-column layout. Avoid inline grid definitions that conflict with CSS media rules.

- [ ] **Step 3: Keep compact status and citations visible below 1100px**

Do not use `display:none` for QA statistics, sources or uploaded-file state. Move them into a compact horizontal summary or collapsible panel near the relevant content.

- [ ] **Step 4: Verify reduced-motion and touch targets**

Use browser emulation or CSS evaluation for `prefers-reduced-motion: reduce`. Verify persistent decorative animations are disabled and interactive controls have at least 44px effective touch height.

- [ ] **Step 5: Run all available checks**

Run:

```bash
cd /Users/mt/Desktop/gakki-ai-fitness
.venv/bin/python -m pytest tests/test_web_frontend_contract.py tests/test_web_api_smoke.py -q
```

Expected: frontend contract passes; API smoke either passes or is explicitly skipped only when an optional external dependency is unavailable. Do not claim full API functionality without API keys and backing services.

- [ ] **Step 6: Perform browser acceptance at three viewports**

At 1440px verify desktop hierarchy, at 768px verify tablet compact panels, and at 375px verify collapsed profile, readable plan cards, visible tabs and usable buttons. Capture screenshots and inspect them visually; also check browser console for uncaught errors.

- [ ] **Step 7: Commit responsive polish and tests**

```bash
git add app/static/index.html tests/test_web_api_smoke.py
git commit -m "test: verify responsive IRON MIND experience"
```

---

### Task 8: 最终验证、差异审查与交付

**Files:**
- Modify: `docs/README.md` only if an existing documentation page links to the old UI contract; otherwise no documentation change。
- Verify: `app/static/index.html`、`tests/test_web_frontend_contract.py`、`tests/test_web_api_smoke.py`

- [ ] **Step 1: Run repository checks relevant to the changed surface**

```bash
cd /Users/mt/Desktop/gakki-ai-fitness
.venv/bin/python -m pytest tests/test_web_frontend_contract.py tests/test_web_api_smoke.py -q
.venv/bin/python -m compileall -q app src
 git diff --check HEAD~7..HEAD
```

- [ ] **Step 2: Review the diff for scope creep**

Confirm no changes were made to database schemas, authentication, admin authorization, Docker services, LLM prompts or SSE event names. Confirm no raw dynamic `innerHTML` remains in stream/source rendering paths.

- [ ] **Step 3: Run the app and verify the real page**

Start with:

```bash
cd /Users/mt/Desktop/gakki-ai-fitness
.venv/bin/python app/server.py
```

Open `http://127.0.0.1:8503`, verify initial page, tab navigation, profile editing, empty states, error states and mobile layout. External AI generation is only tested if `.env` keys and backing services are intentionally configured.

- [ ] **Step 4: Check git status and prepare delivery summary**

```bash
git status --short --branch
git log -8 --oneline
```

Report changed files, verification commands and any limitations caused by missing API keys, Docker services or unavailable external dependencies.

- [ ] **Step 5: Commit only if final verification is green**

```bash
git add app/static/index.html tests/test_web_frontend_contract.py tests/test_web_api_smoke.py
 git commit -m "feat: complete IRON MIND visual and UX refresh"
```

Do not create this final commit if any required check is failing; report the failure and keep the working tree state explicit.

---

## Plan Self-Review

### Spec coverage

- 页面结构与状态：Task 1、Task 2。
- A＋B 配色、字体、卡片、动画：Task 2。
- 语义化 Tab、档案面板、移动端折叠：Task 3、Task 7。
- 安全文本渲染、HTTP 错误、取消与并发：Task 4。
- 计划稳定进度和结果分层：Task 5。
- QA 新会话、动作分析、引用来源：Task 6。
- 响应式、reduced-motion、touch target：Task 7。
- 验收、范围审查和交付：Task 8。
- 明确不在范围内的后端专项：Global Constraints 与 Task 8 Step 2。

### Placeholder scan

计划中没有 `TBD`、`TODO`、`待定`、`待补` 或“写测试但未给出测试内容”的步骤。每个任务都包含文件、接口、可执行步骤、命令和预期结果。

### Type/name consistency

- Task 1 定义 `setViewState`、`resetQaView`、`setLoading`；Task 6 使用 `resetQaView`。
- Task 4 定义 `appendText`、`setText`、`renderSourceList`、`setRequestState`；Task 5/6 按同名使用。
- Task 5 定义 `updatePlanStage`、`renderPlanResult`；后续只依赖其行为，不引入第二套名称。
- Task 3 定义 `activateTab`、`toggleProfileEditor`；没有重复或冲突的替代名称。

### 必要性复核

本计划优先修复会中断用户主流程的状态和 DOM 问题，再做视觉层和响应式；没有把完整架构重构混入本次工作。契约测试和浏览器验收同时存在，是为了避免只通过静态检查或只截图而遗漏真实交互问题。
