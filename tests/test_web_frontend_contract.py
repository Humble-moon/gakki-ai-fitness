import re
from pathlib import Path


HTML = Path("app/static/index.html").read_text(encoding="utf-8")


def _fn_body(name):
    """取一个顶层函数的函数体。

    必须同时识别 ``function x()`` 与 ``async function x()``——
    只按 ``\nfunction `` 切会在遇到 async 函数时越界，把下一个函数
    也吞进来，于是断言读到的是别人的代码（这类切片错误会伪装成
    "契约通过"，方向相反时则会伪装成"实现有问题"）。
    """
    import re
    start = re.search(rf"^(?:async )?function {re.escape(name)}\(", HTML, re.M)
    assert start, f"找不到函数 {name}"
    nxt = re.search(r"^(?:async )?function ", HTML[start.end():], re.M)
    end = start.end() + nxt.start() if nxt else len(HTML)
    return HTML[start.start():end]


def test_stable_stream_and_qa_containers_exist():
    for element_id in (
        "app-shell", "profile-summary", "profile-editor", "plan-status",
        "plan-advice", "plan-stream", "plan-result", "qa-output", "qa-chat",
        "qa-welcome", "qa-sources",
    ):
        assert f'id="{element_id}"' in HTML


def test_motion_preference_and_live_regions_exist():
    assert "prefers-reduced-motion" in HTML
    assert 'aria-live="polite"' in HTML
    assert 'role="alert"' in HTML


def test_state_helpers_and_request_lifecycle_exist():
    for helper in ("setViewState", "resetQaView", "setLoading", "appendText", "setText", "renderSourceList", "setRequestState"):
        assert f"function {helper}(" in HTML
    assert "const activeRequests" in HTML
    assert "AbortController" in HTML
    assert "replaceChildren" in HTML
    assert "textContent" in HTML


def test_response_validation_precedes_reader():
    start = HTML.index("async function consumeSse")
    end = HTML.index("async function streamFrom", start)
    segment = HTML[start:end]
    assert ".ok" in segment and ".body" in segment and "getReader" in segment
    assert segment.index(".ok") < segment.index("getReader")
    assert "text/event-stream" in segment
    # 计划生成走状态图接口（见 test_plan_generation_uses_the_state_graph_endpoint），
    # 其余两条仍由命令式编排提供。
    for marker in ("/api/v2/plan/stream", "/api/analyze-exercise", "/api/ask-question"):
        assert marker in HTML


def test_dynamic_stream_text_does_not_use_raw_html():
    for token in ("sourceList.innerHTML", "srcDiv.innerHTML", "aiBubble.innerHTML", "stageEl.innerHTML"):
        assert token not in HTML


def test_qa_reset_does_not_replace_shell():
    start = HTML.index("function resetQaView")
    end = HTML.find("function ", start + 10)
    body = HTML[start:end if end != -1 else None]
    assert "qa-output.innerHTML" not in body
    assert "replaceChildren" in body


def test_new_session_does_not_replace_qa_shell():
    start = HTML.index("function newQaSession")
    end = HTML.find("function ", start + 10)
    body = HTML[start:end if end != -1 else None]
    assert "resetQaView()" in body
    assert "qa-output.innerHTML" not in body


def test_qa_turns_and_sources_use_dom_text_content():
    assert "function appendQaTurn(" in HTML
    start = HTML.index("function renderSourceList")
    end = HTML.index("/* ================================================================", start)
    body = HTML[start:end]
    assert "textContent" in body
    assert "<details>" not in body
    assert "source.title" not in body


def test_analysis_state_is_scoped_and_retryable():
    start = HTML.index("async function analyzeExercise")
    end = HTML.find("// ── File Upload", start)
    body = HTML[start:end]
    assert "setRequestState('analysis'" in body
    assert "retry" in body.lower()
    assert "output.innerHTML" not in body
    assert "activeRequests.analysis === controller" in body


def test_visual_system_uses_iron_mind_tokens_and_layout():
    for token in ("--bg-deep:", "--accent-orange:", "--status-safe:", "--focus-ring:"):
        assert token in HTML
    for selector in (".app-shell", ".profile-panel", ".profile-summary", ".profile-editor", ".workspace", ".workspace-header", ".feature-tabs", ".feature-panel"):
        assert selector in HTML
    assert "@media (max-width: 767px)" in HTML
    assert "@media (prefers-reduced-motion: reduce)" in HTML


def test_profile_fields_are_visible_by_default_and_mobile_toggle_starts_closed():
    assert 'id="profile-editor" class="field-group profile-editor">' in HTML
    assert 'id="height"' in HTML and 'id="weight"' in HTML and 'id="years"' in HTML
    assert 'id="injuries-input"' in HTML
    assert 'aria-expanded="false"' in HTML
    assert 'const open = button.getAttribute("aria-expanded") === "true";' in HTML


def test_qa_workspace_has_title_meta_and_composer_regions():
    for element_id in (
        "qa-header", "qa-meta", "qa-composer", "qa-keyboard-hint",
        "qa-new-session-btn", "qa-query", "btn-ask", "qa-output",
        "qa-welcome", "qa-chat", "qa-sources", "qa-error", "qa-coach-context",
        "qa-retry", "qa-upload-status",
    ):
        assert f'id="{element_id}"' in HTML
    assert 'aria-controls="qa-output"' not in HTML
    assert 'aria-live="polite"' in HTML


def test_qa_visual_contract_prioritizes_coach_guidance_and_safe_context():
    assert 'class="qa-hero-kicker"' in HTML
    assert 'class="qa-lead"' in HTML
    assert 'class="qa-topic-list"' in HTML
    assert 'class="qa-secondary-topics"' in HTML
    assert 'function renderQaCoachContext(' in HTML
    assert 'textContent' in HTML[HTML.index("function renderQaCoachContext"):HTML.index("function renderQaCoachContext") + 1200]
    assert 'class="qa-safety-note"' in HTML
    assert 'qa-answer-meta' in HTML


def test_qa_upload_is_keyboard_accessible_and_sets_safe_status():
    assert 'role="button"' in HTML[HTML.index('id="file-upload-area"') - 100:HTML.index('id="file-upload-area"') + 500]
    assert 'tabindex="0"' in HTML[HTML.index('id="file-upload-area"') - 100:HTML.index('id="file-upload-area"') + 500]
    assert 'onkeydown="' in HTML[HTML.index('id="file-upload-area"') - 100:HTML.index('id="file-upload-area"') + 500]
    assert 'id="qa-upload-status"' in HTML
    assert '无需上传也可以直接提问' in HTML
    assert '文件格式' in HTML


def test_qa_failure_preserves_draft_and_exposes_retry_without_race():
    qa = HTML[HTML.index("async function askQuestion"):HTML.index("</script>", HTML.index("async function askQuestion"))]
    assert "lastQaQuestion" in HTML
    assert "retryQaQuestion" in HTML
    assert "input.value = ''" in qa or "value = '';" in qa
    assert "if (e.name === 'AbortError')" in qa
    assert "qa-retry" in HTML
    assert "activeRequests.qa === controller" in qa
    assert "finally" in qa


def test_qa_chat_scroll_and_responsive_overflow_contract():
    assert ".qa-chat {" in HTML
    assert "overflow-y: auto" in HTML[HTML.index(".qa-chat {"):HTML.index(".qa-chat {") + 700]
    assert "overflow-x: hidden" in HTML
    assert "@media (max-width: 375px)" in HTML
    assert "@media (min-width: 768px)" in HTML


def test_qa_layout_is_single_column_and_mobile_safe():
    assert ".qa-workspace" in HTML
    assert ".qa-header" in HTML
    assert ".qa-meta" in HTML
    assert ".qa-composer" in HTML
    assert "grid-template-columns: 1fr 320px" not in HTML
    assert ".qa-side-panel { display: none; }" not in HTML
    assert "@media (max-width: 767px)" in HTML
    assert "#qa-new-session-btn" in HTML
    assert "min-height: 44px" in HTML




def test_qa_state_transitions_keep_composer_and_session_controls():
    reset = HTML[HTML.index("function resetQaView"):HTML.index("function setLoading", HTML.index("function resetQaView"))]
    new_session = HTML[HTML.index("function newQaSession"):HTML.index("// ── Progress Bar", HTML.index("function newQaSession"))]
    ask = HTML[HTML.index("async function askQuestion"):HTML.index("</script>", HTML.index("async function askQuestion"))]
    assert "qa-welcome" in reset and "qa-chat" in reset and "qa-sources" in reset
    assert "qa-new-session-btn" in new_session and "qa-output" not in new_session
    assert "qa-composer" not in ask or "qa-query" in ask
    assert "answer_chunk" in ask and "renderSourceList" in ask and "qa-stat-sources" in ask
    assert "activeRequests.qa === controller" in ask
    assert "if (welcome) welcome.hidden = true;" in ask
    assert "if (chat) chat.hidden = false;" in ask
    assert "badge?.removeAttribute('hidden')" in ask
    assert "newButton?.removeAttribute('hidden')" in ask
    assert "badge.style.display" not in new_session
    assert "newButton.style.display" not in new_session


def test_plan_events_target_stable_layers():
    for event_name in ("stage", "advice_chunk", "factcheck_done", "done"):
        assert event_name in HTML
    assert "writer_chunk" not in HTML
    for element_id in ("plan-status", "plan-advice", "plan-stream", "plan-result"):
        assert element_id in HTML
    assert "function updatePlanStage(" in HTML
    assert "function renderPlanResult(" in HTML


def test_plan_stage_does_not_clear_workspace_root():
    start = HTML.index("if (event === 'stage')")
    end = HTML.find("else if (event === 'advice_chunk')", start)
    body = HTML[start:end]
    assert "resultArea.innerHTML" not in body
    assert "updatePlanStage(data)" in body
    assert "plan-stream" in body


def test_review_pending_done_renders_only_safe_review_panel_and_skips_autosave():
    start = HTML.index("onTerminal: (event, data) =>")
    end = HTML.index("// ═══════════════ RENDER PLAN", start)
    terminal = HTML[start:end]
    assert "data?.delivery_status === 'review_pending'" in terminal
    assert "renderReviewPending(resultArea, data?.review)" in terminal
    assert "data?.delivery_status !== 'safe_delivered'" in terminal
    # 交付的守卫必须挡在自动收录之前：被扣下待审的计划不进「我的计划」。
    # 顺序是这条安全语义的全部依据——把断言拆开就能看出顺序错了。
    assert "autoSavePlan" in terminal
    assert terminal.index("data?.delivery_status !== 'safe_delivered'") \
        < terminal.index("autoSavePlan"), "守卫必须在自动收录之前"

    start = HTML.index("function renderReviewPending")
    end = HTML.index("function renderPlanResult", start)
    panel = HTML[start:end]
    assert ".innerHTML" not in panel
    assert "风险原因" in panel
    assert "下一步" in panel
    # 界面上不该出现这三样：审核编号（无意义的 UUID）、动作库未收录（我们的数据缺口，
    # 不是用户的风险）、禁止项（一串祈使句，且与「下一步」重复）。
    assert "审核编号：${" not in panel, "审核编号不该出现在界面上"
    assert "'动作库中未收录：'" not in panel, "动作库缺口不该占风险清单的一行"
    assert "'禁止项：'" not in panel, "禁止项应并入下一步，不再单列"


def test_plan_result_uses_safe_dynamic_dom_and_responsive_cards():
    """计划卡片必须用安全 DOM API 构造（防 XSS），响应式布局在位。

    范围覆盖 `renderPlanDays` 与 `renderPlanResult`：卡片构造逻辑在
    2026-09-24 被抽到 `renderPlanDays`，供「正式结果」与「流式草稿」共用。
    若只检查后者，卡片构造就脱离了这条安全契约的保护。
    """
    start = HTML.index("function renderPlanDays")
    end = HTML.find("function renderAnalysisResult", start)
    body = HTML[start:end]
    assert ".innerHTML" not in body
    assert "appendText" in body
    assert "el('article', 'plan-day-column'" in body
    assert "plan-ex-notes" in body
    assert "@media (max-width: 767px)" in HTML
    assert "grid-template-columns: repeat(2, minmax(0, 1fr))" in HTML


def test_plan_generation_clears_only_plan_layers():
    start = HTML.index("async function generatePlan")
    end = HTML.find("async function analyzeExercise", start)
    body = HTML[start:end]
    assert "plan-output').innerHTML" not in body
    assert "plan-advice" in body and "plan-stream" in body and "plan-result" in body
    assert "showProgress('plan-status')" in body


def test_saved_plan_renders_into_result_layer_with_safe_dom():
    """打开一份旧计划要渲染进 #plan-result，且用安全 DOM 构造。

    合并后「打开旧计划」只剩 openSavedPlan 一条路（原来的 restorePlan
    连同本机历史一起删掉了），所以这条契约挂在它上面。
    """
    body = _fn_body("openSavedPlan")
    assert "plan-result" in body
    assert "renderPlanResult(area, plan)" in body
    assert ".innerHTML" not in body


def test_no_result_root_destructive_clear():
    assert "resultArea.innerHTML = ''" not in HTML


def test_plan_desktop_cards_stretch_to_equal_row_height():
    grid = HTML[HTML.index(".plan-result-grid {"):HTML.index(".plan-day-column {", HTML.index(".plan-result-grid {"))]
    card = HTML[HTML.index(".plan-day-column {"):HTML.index(".plan-day-column::before", HTML.index(".plan-day-column {"))]
    body = HTML[HTML.index(".plan-day-body {"):HTML.index(".plan-ex-heading,", HTML.index(".plan-day-body {"))]
    assert "align-items: stretch" in grid
    assert "display: flex" in card and "flex-direction: column" in card
    assert "height: 100%" in card
    assert "flex: 1" in body
    assert "@media (max-width: 767px)" in HTML
    mobile = HTML[HTML.index("@media (max-width: 767px)"):HTML.index("@media (min-width: 768px)", HTML.index("@media (max-width: 767px)"))]
    assert ".plan-day-column { height: auto; }" in mobile


    shared_grid = "--plan-columns: minmax(0, 2.4fr) minmax(56px, .7fr) minmax(72px, .9fr) minmax(92px, 1.1fr);"
    assert shared_grid in HTML
    assert ".plan-ex-heading," in HTML
    assert ".plan-ex-row {" in HTML
    assert "grid-template-columns: var(--plan-columns);" in HTML
    assert "dataset.planColumn" in HTML


def test_plan_mobile_card_copy_is_readable():
    assert ".plan-ex-name { flex: 1 1 100%;" in HTML
    assert ".plan-ex-notes { flex: 1 1 100%;" in HTML
    assert "休息 ${exercise?.rest || '60 秒'}" in HTML
    assert "${exercise?.reps ?? '?'} 次" in HTML


def test_saved_plans_can_be_renamed_in_place():
    """已保存的计划能在列表里就地改名。

    锁三件事：入口在（每行一个「重命名」）、请求打对地址（PATCH /api/plans/{id}）、
    以及编辑是原地替换而不是原生弹窗——prompt() 会打断整页且样式无法统一。
    """
    start = HTML.index("function startRenamePlan")
    end = HTML.find("function savePlanToHistory", start)
    body = HTML[start:end]
    assert "saved-plan-rename-input" in body
    assert "replaceWith" in body, "应原地替换标题，而不是另开一处编辑"
    assert "Escape" in body and "Enter" in body
    assert "maxLength = 40" in body
    assert "prompt(" not in body, "不应使用原生 prompt 弹窗"

    assert "function renameSavedPlan" in HTML
    assert "method: 'PATCH'" in HTML
    assert "startRenamePlan(row, info, p)" in HTML
    assert "'重命名'" in HTML
    assert ".saved-plan-rename-input {" in HTML


def test_internal_notices_are_classified_but_never_rendered():
    """库外动作名只归类、不展示。

    这条提示反映的是词表覆盖缺口，不是用户写错了名字：动作库收了 8 个俯卧撑
    变式却没有通用的「标准俯卧撑」，精确匹配必然落空。用户看到「标准俯卧撑」
    被说成"不在动作库中，请确认动作名称是否正确"，既无从确认，也只会觉得
    这个产品不认识俯卧撑。
    """
    assert "internal: true" in HTML
    start = HTML.index("function isInternalNotice")
    end = HTML.find("function renderPlanNotice", start)
    body = HTML[start:end]
    # 判定必须复用 NOTICE_TOPICS，而不是在别处再抄一份名单
    assert "NOTICE_TOPICS.find" in body
    assert "topic.internal" in body
    # 分流同时给出 visible 与 internal 两侧
    assert "visible" in body and "internal" in body


def test_warning_count_only_counts_what_is_shown():
    """顶部条数必须只数给用户看的那部分。

    否则会出现"3 条提示"点开却是空的——内部条目被过滤掉了，计数却没跟着过滤。
    """
    start = HTML.index("function renderPlanResult")
    end = HTML.find("const notice = renderPlanNotice", start)
    body = HTML[start:end]
    assert "splitWarnings(plan.warnings).visible" in body
    assert "planWarnings.length" in body
    assert "plan.warnings.length" not in body, "计数不能再直接用原始列表长度"


def test_notice_panel_is_absent_when_nothing_is_showable():
    """没有可展示内容时连外框都不该出现，否则是个点开为空的面板。"""
    start = HTML.index("function renderPlanNotice")
    body = HTML[start:HTML.find("function renderPlanResult", start)]
    assert "if (!items.length) return null;" in body
    assert "const notice = renderPlanNotice(planWarnings);" in HTML
    assert "if (notice) container.appendChild(notice);" in HTML


def test_rename_failure_notice_stays_on_the_saved_tab():
    """重命名失败的提示要落在「我的计划」这一屏。

    用全局的 showStatus 会写进训练日志那一屏——用户正站在「我的计划」上，
    等于什么都没看见。
    """
    start = HTML.index("function showSavedPlanNotice")
    end = HTML.find("\n}", start)
    body = HTML[start:end]
    assert "saved-plans-head" in body
    assert "log-status" not in body
    assert ".saved-plan-notice {" in HTML




def test_plan_backend_json_stream_is_not_rendered_to_users():
    """后端原始 JSON 不得直接展示给用户。

    这条契约来自 c0676d5（fix: hide internal json streaming output）：
    当时界面把逐 token 的 `writer_chunk` 直接 appendText 出去，用户看到的
    是一串 `{"plan_name": ...` 的原文，既不可读也没有意义，因此被移除。

    2026-09-24 起首轮草稿会通过 `writer_done_raw` 提前渲染以缩短等待感
    （计划生成含重写回路，实测约 47s）。**契约的核心没有放宽**——
    仍然禁止逐 token 原文，且草稿必须经 `renderPlanDraft` 解析成结构化
    对象后渲染。草稿未经安全检查，额外的安全约束见
    `test_plan_draft_is_marked_unsafe_and_visually_distinct`。
    """
    start = HTML.index("async function streamFrom")
    end = HTML.index("// ═══════════════ RENDER PLAN", start)
    body = HTML[start:end]
    assert "event === 'writer_chunk'" not in body, "逐 token 原始 JSON 不得直接渲染"
    assert "renderPlanResult(resultArea, data)" in body
    # 草稿若渲染，必须走解析函数——不得把 raw 文本直接交给用户
    assert "writer_done_raw" in body, "首轮草稿应提前渲染（见 P0-2 延迟优化）"
    assert "renderPlanDraft(" in body, "草稿必须经 renderPlanDraft 解析后渲染"


def test_plan_draft_is_marked_unsafe_and_visually_distinct():
    """草稿未经安全检查，必须在界面上说清楚，且不能长得像正式结果。

    这不是文案偏好，是安全约束：本项目的前提是"检查通过才交付"。
    若草稿与正式计划看不出区别，用户会直接照着练——一个可能含冲突动作的
    计划就绕过了整个安全门。因此草稿在三个维度上必须与正式结果不同：
        1. 顶部有明确的"检查中 / 请勿照此训练"横幅
        2. 视觉上降级（虚线边框、降饱和），一眼可辨
        3. 不提供任何操作入口（重生成等按钮隐藏）
    """
    start = HTML.index("function renderPlanDraft")
    end = HTML.index("function renderPlanResult", start)
    body = HTML[start:end]
    # 1. 必须警示
    assert "正在做安全检查" in body
    assert "请勿照此训练" in body
    # 2. 结构化渲染，不是原文
    assert "renderPlanDays(" in body
    assert "parseDraftPlan(" in body
    # 3. 不得带操作入口
    assert "plan-actions" not in body
    # 4. CSS 层面确实降级
    assert ".plan-draft {" in HTML
    assert "dashed" in HTML
    assert ".plan-draft .plan-actions { display: none; }" in HTML


def test_analysis_backend_json_stream_is_not_rendered_to_users():
    start = HTML.index("async function analyzeExercise")
    end = HTML.index("// ── File Upload", start)
    body = HTML[start:end]
    assert "event === 'writer_chunk'" not in body
    assert "renderAnalysisResult(resultDiv, data || {})" in body



def test_shared_sse_consumer_enforces_current_terminal_contract():
    start = HTML.index("async function consumeSse")
    end = HTML.index("async function streamFrom", start)
    body = HTML[start:end]
    assert "text/event-stream" in body
    assert "SSE_TERMINAL_EVENTS" in HTML
    assert "['done', 'error', 'cancelled']" in HTML
    assert "decoder.decode()" in body
    assert "if (!terminal) throw new Error" in body
    assert "terminal.event === 'error'" in body
    assert "terminal.event === 'cancelled'" in body
    assert "terminal.data?.message" in body
    assert "terminal.data?.code" in body


def test_all_streaming_actions_reuse_shared_sse_consumer():
    assert HTML.count("consumeSse(") == 4
    for function_name in ("streamFrom", "analyzeExercise", "askQuestion"):
        start = HTML.index(f"function {function_name}")
        end = HTML.find("\n}", start) + 2
        assert "consumeSse(" in HTML[start:end]


def test_qa_only_clears_draft_after_done_terminal():
    start = HTML.index("async function askQuestion")
    body = HTML[start:HTML.index("</script>", start)]
    terminal = body.index("onTerminal:")
    clear = body.index("input.value = ''")
    assert terminal < clear
    assert "if (event !== 'done') return;" in body


# ---------- 训练闭环（训练日志 + 训练分析）----------


def test_training_tabs_are_wired_into_the_tablist():
    """两个新 tab 的按钮与面板必须通过 aria-controls 正确配对。

    activateTab 靠 aria-controls 自动发现面板，配对错了会静默切不过去。
    """
    for name in ("log", "progress"):
        assert f'id="tab-{name}"' in HTML
        assert f'aria-controls="panel-{name}"' in HTML
        assert f'id="panel-{name}"' in HTML
        assert f'aria-labelledby="tab-{name}"' in HTML


def test_training_panels_are_feature_panels():
    """必须复用 feature-panel，否则会被 tab 切换逻辑漏掉。"""
    for name in ("log", "progress"):
        idx = HTML.index(f'id="panel-{name}"')
        assert "feature-panel" in HTML[idx - 200:idx + 100]


def test_training_tabs_do_not_add_a_new_sse_consumer():
    """新功能全部走 JSON 接口——流式消费点数量是不变量。"""
    assert HTML.count("consumeSse(") == 4


def test_athlete_key_is_separate_from_session_id():
    """运动员标识必须独立于会话标识：session 会随「新对话」轮换，
    训练历史需要跨月稳定归属，复用会导致点一次新对话就丢失全部记录。
    """
    assert "ironmind_athlete_key" in HTML
    # 新会话重置逻辑不得触碰运动员标识
    start = HTML.index("function newQaSession")
    end = HTML.find("function ", start + 10)
    body = HTML[start:end if end != -1 else None]
    assert "athleteKey" not in body
    assert "ATHLETE_KEY_STORAGE" not in body


def test_echarts_loader_has_cdn_fallback_and_timeout():
    """图表加载必须具备降级链：双 CDN + 超时，否则离线演示会白屏。"""
    assert "function loadEcharts(" in HTML
    assert "ECHARTS_CDNS" in HTML
    assert "onerror" in HTML
    assert "setTimeout" in HTML


def test_progress_has_dom_fallback_renderer():
    """CDN 不可用时必须有纯 DOM 降级，保证功能闭环不依赖外部资源。"""
    assert "function renderProgressFallback(" in HTML
    start = HTML.index("function renderProgressFallback(")
    end = HTML.index("function ", start + 10)
    body = HTML[start:end]
    assert "fallback-table" in body


def test_training_renderers_avoid_raw_html():
    """新渲染函数一律用 DOM API，与既有契约保持一致。"""
    for fn in ("renderProgressFallback", "renderProgressStats", "renderAdvice",
               "loadLogHistory"):
        start = HTML.index(f"function {fn}(")
        end = HTML.index("\nfunction ", start + 10)
        body = HTML[start:end]
        assert ".innerHTML" not in body, f"{fn} 不得使用 innerHTML"


# ---------- 计划可解释性 ----------


def test_plan_explanation_is_rendered_from_render_plan_result():
    start = HTML.index("function renderPlanResult")
    end = HTML.index("function renderAnalysisResult", start)
    body = HTML[start:end]
    assert "renderPlanExplanation(container, plan.explain)" in body


def test_plan_explanation_renderer_uses_safe_dom():
    start = HTML.index("function renderPlanExplanation")
    end = HTML.index("function renderPlanResult", start)
    body = HTML[start:end]
    assert ".innerHTML" not in body
    assert "el(" in body


def test_plan_explanation_does_not_duplicate_warnings():
    """warnings 与 active_issues 同源，面板里不能再罗列一遍同样的文字，
    否则用户会看到同一个问题出现两次。"""
    start = HTML.index("function renderPlanExplanation")
    end = HTML.index("function renderPlanResult", start)
    body = HTML[start:end]
    assert "已列在计划上方" in body
    assert "safety.active.forEach" not in body


def test_plan_explanation_hides_itself_when_data_absent():
    """旧缓存里的计划没有 explain 字段，面板必须静默隐藏而不是显示出错。"""
    start = HTML.index("function renderPlanExplanation")
    end = HTML.index("function renderPlanResult", start)
    body = HTML[start:end]
    assert "if (!container || !explain) return;" in body


def test_explain_toggle_exposes_expanded_state_to_assistive_tech():
    start = HTML.index("function renderPlanExplanation")
    end = HTML.index("function renderPlanResult", start)
    body = HTML[start:end]
    assert "aria-expanded" in body


# ---------- 视觉规范（容易被后续改动无意破坏，锁住）----------


def test_numeric_displays_use_tabular_figures():
    """数据类数字必须等宽对齐，否则纵向比对时逐位跳动。"""
    assert "font-variant-numeric: tabular-nums" in HTML


def test_composed_empty_state_exists_and_avoids_raw_html():
    """空状态要给出下一步动作，而不是一句「还没有记录」。"""
    assert ".empty-composed" in HTML
    assert "function composedEmpty(" in HTML
    start = HTML.index("function composedEmpty(")
    end = HTML.index("\n}", start)
    assert ".innerHTML" not in HTML[start:end]


def test_advisory_items_use_severity_bar_not_generic_border():
    """建议条目用内嵌严重度色条区分，避免「边框+底色」的通用卡片观感。"""
    start = HTML.index(".advice-item {")
    end = HTML.index(".advice-item .ai-reason", start)
    body = HTML[start:end]
    assert "inset 3px 0 0 0" in body
    assert "border: 1px solid" not in body


def test_interactive_rows_have_focus_and_press_feedback():
    """一排相同的输入框里，用户需要知道焦点在哪；按钮需要按压反馈。"""
    assert ".log-entry-row:focus-within" in HTML
    assert ".log-entry-remove:active" in HTML
    assert ".explain-toggle:active" in HTML


def test_chart_containers_have_depth():
    """大块图表区域不能是纯平坦色块。

    深浅两套主题表达"深度"的方式不同，断言只锁意图、不锁手法：
    深色主题用内阴影（inset）压出凹陷，浅色主题用一档很轻的外阴影
    （0 1px 3px）把卡片从底面上托起来。浅底上再加深色内阴影会把图表框
    涂脏，所以浅色化时换成了外阴影——这条断言也必须跟着改口径，
    否则它锁的就是"必须还是深色主题"，而不是"必须有层次"。
    """
    start = HTML.index(".chart-box {")
    end = HTML.index("}", start)
    block = HTML[start:end]
    assert "radial-gradient" in block
    shadow = next((ln for ln in block.splitlines() if "box-shadow" in ln), "")
    assert shadow, ".chart-box 缺少 box-shadow，会退化成平坦色块"
    assert "none" not in shadow, "box-shadow: none 等于没有层次"


# ---------- 界面声明必须与实际行为一致 ----------


def test_welcome_copy_does_not_claim_knowledge_retrieval():
    """欢迎页不得声称计划生成会做「知识检索」。

    计划生成只检索 exercises 表（VectorSearch + KeywordSearch），
    knowledge_chunks 从未参与；SkillRegistry.get_knowledge_refs() 只有定义、
    零调用点。声称"知识检索"属于说了没做到——与名字不符比没有更伤。
    若将来真的把知识库接进生成链路，再连同这条断一起改回。
    """
    assert "知识检索" not in HTML
    assert "动作检索" in HTML


def test_welcome_library_counts_match_actual_data():
    """欢迎页的库规模数字必须与仓库里的真实数据一致。

    这两个数字曾停在早期版本（39 / 21），与实际相差 5~8 倍。与其为它建接口，
    不如让测试盯着：数据一变，这条立刻失败并指出界面要同步更新。
    """
    import json

    root = Path(__file__).resolve().parent.parent
    exercises = len(json.loads(
        (root / "data/seed_exercises.json").read_text(encoding="utf-8")))
    docs = len(list((root / "data/knowledge").glob("*.md")))

    assert f'<span class="ss-val">{exercises}</span> 个标准动作' in HTML, (
        f"动作库实际有 {exercises} 个，欢迎页上的数字需要同步")
    assert f'<span class="ss-val">{docs}</span> 篇专业文献' in HTML, (
        f"知识库实际有 {docs} 篇，欢迎页上的数字需要同步")


def test_generate_plan_request_carries_athlete_key():
    """生成计划的请求体必须带 athlete_key。

    长期记忆以 athlete_key 为身份。前端若不传，后端就退化成身高体重的 MD5
    指纹——用户改一次体重就换一个身份，先前存的偏好再也读不回来。问答路径
    一直传着，只有生成计划这条漏了，于是两条路径写进了两个不同的身份。
    """
    import re

    m = re.search(r"const body = \{[^}]*\}", HTML)
    assert m, "找不到生成计划的请求体构造"
    assert "athlete_key: athleteKey" in m.group(0), (
        "生成计划必须带 athlete_key，否则长期记忆会落到随体重漂移的指纹上")


def test_habits_input_exists_and_is_collected():
    """运动习惯要能填写并提交。

    这是新加的一类长期记忆。若输入框存在而 getProfile 不收集它，
    用户填了等于没填——界面上看得见、后端收不到，是最难自查的一类漏。
    """
    assert 'id="habits-input"' in HTML, "档案区缺少运动习惯输入框"
    assert "function getProfile()" in HTML
    profile_fn = _fn_body("getProfile")
    assert "habits" in profile_fn, "getProfile 必须收集 habits，否则填了也不会提交"


def test_account_ui_contract():
    """登录界面必须齐全且不改变游客可用性。

    账号是可选增强：不登录也能用完整功能。若登录表单是必经之路，
    这个契约就破了。
    """
    for element_id in ("account-block", "account-label", "account-hint",
                       "auth-form", "auth-username", "auth-password",
                       "auth-status", "btn-login", "btn-register", "btn-logout"):
        assert f'id="{element_id}"' in HTML, f"缺少登录界面元素 #{element_id}"

    # 表单默认收起，游客直接看到的是主界面而不是登录墙
    assert 'id="auth-form" hidden' in HTML

    # 密码框必须是 password 类型，否则明文显示在屏幕上
    assert 'id="auth-password" type="password"' in HTML

    for helper in ("toggleAuthForm", "loadSession", "handleLogin",
                   "handleRegister", "handleLogout", "renderSession"):
        assert f"function {helper}(" in HTML, f"缺少函数 {helper}"


def test_logout_clears_the_cached_user_even_on_failure():
    """登出接口失败时也必须把界面切回游客态。

    否则网络一抖，用户就卡在一个假的"已登录"界面里，看到的还是别人的
    或过期的数据。
    """
    fn = _fn_body("handleLogout")
    assert "currentUser = null" in fn, "登出必须清空本地登录态"
    assert "catch" in fn, "登出接口失败不能中断状态清理"


def test_demo_switch_is_disabled_while_logged_in():
    """演示数据是共享的固定身份，登录态下切换会让人以为数据丢了。"""
    fn = _fn_body("loadDemoData")
    assert "currentUser" in fn, "loadDemoData 必须检查登录态"
    render = _fn_body("renderSession")
    assert "btn-load-demo" in render, "登录态下应隐藏演示按钮"


def test_claim_ui_contract():
    """认领入口必须存在，且登录时不自动认领。

    在别人的电脑上登录时自动认领会把对方的游客数据搬进自己账号，
    因此登录只提示、由用户点按钮决定；注册才自动认领。
    """
    assert 'id="claim-offer"' in HTML
    assert 'id="btn-claim"' in HTML
    for helper in ("claimLocalGuestData", "maybeOfferClaim", "handleClaim"):
        assert f"function {helper}(" in HTML

    login_fn = _fn_body("handleLogin")
    assert "claimLocalGuestData()" not in login_fn, (
        "登录不应自动认领——可能是在别人的设备上登录")

    register_fn = _fn_body("handleRegister")
    assert "claimLocalGuestData()" in register_fn, "注册后应自动认领本机数据"


def test_claim_never_sends_the_demo_key():
    """演示身份是共享的，认领它会把公共数据混进个人账号。"""
    fn = _fn_body("claimLocalGuestData")
    assert "DEMO_ATHLETE_KEY" in fn, "认领前必须排除演示身份"


def test_review_panel_shows_the_actual_risks():
    """被拦下时必须摊开具体风险，而不是只说一句"检测到伤病风险"。

    安全门已经算出了每条冲突（哪个动作、为什么、怎么替换）。只显示一行
    reason 会让用户看到一串审核编号却不知道自己做错了什么——实测中
    这是最像"产品坏了"的一屏。
    """
    fn = _fn_body("renderReviewPending")
    assert "details.issues" in fn or "review.issues" in fn, "审核面板必须渲染 issues"
    assert "severity" in fn, "风险条目要区分严重级别"
    assert "exercise" in fn, "风险条目要带动作名"


def test_review_panel_does_not_promise_a_human_reviewer():
    """不得让用户等待一个不存在的人工审核者。"""
    fn = _fn_body("renderReviewPending")
    assert "等待专业审核" not in fn
    assert "不是故障" in fn, "要明确告知这是安全机制生效，而非产品故障"


def test_review_panel_keeps_internals_out_of_the_ui():
    """内部实现细节只进控制台，不上界面。

    三样东西撤出界面：审核编号（一串 UUID，用户读不懂，还加重"产品坏了"的观感）、
    动作库未收录（我们自己的数据缺口，用户既没法处理也不关心）、禁止项祈使句
    （像系统在训人，且与"下一步"重复）。但排查时要用，所以走 console + data 属性。
    """
    fn = _fn_body("renderReviewPending")
    assert "console.info" in fn, "撤出界面的信息应至少留在控制台，方便排查"
    assert "dataset.reviewId" in fn, "审核编号应挂到 data 属性而不是渲染成文本"
    assert "review-pending-footnote" in fn, "禁止项的实质应改成建议口吻保留"


def test_review_panel_renders_issue_children_as_text():
    """风险文本来自模型输出，必须走 textContent 而非 innerHTML。"""
    fn = _fn_body("renderReviewPending")
    assert "innerHTML" not in fn
    assert "appendText" in fn


def test_plan_generation_uses_the_state_graph_endpoint():
    """计划生成必须走 v2（状态图），而不是命令式编排。

    差别不在"哪个更快"，而在**被安全门拦下之后还能不能救回来**：
    v2 的图带检查点，审核解除后能从中断处接着跑完、服务重启也不丢；
    命令式那一侧创建的审核工件不带线程标识，计划被扣下就再也拿不回来。

    这条契约把"界面走哪条编排路径"钉死，避免它悄悄退回 v1。
    """
    fn = _fn_body("generatePlan")
    assert "/api/v2/plan/stream" in fn, "计划生成应走状态图接口"
    assert "'/api/generate-plan'" not in fn, "不应再退回命令式接口"


def test_plan_request_still_carries_identity():
    """换端点不得弄丢身份字段——长期记忆靠它定归属。"""
    fn = _fn_body("generatePlan")
    assert "athlete_key: athleteKey" in fn
    assert "session_id" in fn


def test_exercise_demo_images_are_opt_in_and_verified_only():
    """图示默认收起、按需加载，且只显示清单里有的动作。

    未收录的动作不显示图，而不是拿一张对不上的图充数——健身场景里
    照着错的图练会受伤。清单本身只含逐张目视核对过的条目。
    """
    for helper in ("loadExerciseImages", "buildExerciseDemo"):
        assert f"function {helper}(" in HTML, f"缺少 {helper}"

    fn = _fn_body("buildExerciseDemo")
    assert "panel.hidden" in fn, "图片面板默认应隐藏"
    assert "loading = 'lazy'" in fn, "图片应懒加载，否则一屏十几个动作会并发请求"
    assert "innerHTML" not in fn, "图片节点必须用 DOM API 构造"
    # 清单里没有就返回 null（不渲染），而不是渲染占位
    assert "return null" in fn


def test_exercise_image_manifest_only_contains_verified_entries():
    """清单里的每个动作都要有真实的图片文件存在。

    清单是构建脚本产出、人工核对后的结果；文件被误删而清单没同步时，
    前端会显示裂图。
    """
    import json

    root = Path(__file__).resolve().parent.parent
    manifest = json.loads((root / "data/exercise_images.json").read_text(encoding="utf-8"))
    assert manifest, "清单不应为空"

    for cn, entry in manifest.items():
        assert entry.get("images"), f"{cn} 没有图片条目"
        for rel in entry["images"]:
            assert (root / "app/static" / rel).exists(), f"{cn} 的图片缺失：{rel}"


def test_auth_gate_exists_with_a_visible_guest_exit():
    """首屏是登录页，但必须提供明写的游客入口。

    游客入口藏着不写，用户就会以为必须先注册——而「不登录也能完整使用」
    是这个项目一直坚持的边界，不该被一次 UI 改动悄悄取消。
    """
    for element_id in ("auth-gate", "gate-username", "gate-password",
                       "gate-login", "gate-register", "gate-guest", "gate-status"):
        assert f'id="{element_id}"' in HTML, f"缺少登录页元素 #{element_id}"

    assert 'id="gate-password" type="password"' in HTML, "密码框必须是 password 类型"
    assert "先随便看看" in HTML, "游客入口的文案要说明这是可以随便逛的"

    for helper in ("showAuthGate", "hideAuthGate", "enterAsGuest", "gateAuth"):
        assert f"function {helper}(" in HTML, f"缺少 {helper}"


def test_first_paint_is_the_login_page_not_the_app():
    """首帧必须是登录页，应用壳默认隐藏。

    原先两处是反的（登录页 hidden、应用壳常驻），结果是未登录用户先看到
    整个应用界面一闪、再被登录页盖住——这也是"登录门像功能页的一部分"
    这个观感的来源。现在把初始态写在 HTML 属性上，不依赖脚本。
    """
    assert 'id="auth-gate"' in HTML
    assert 'id="auth-gate" hidden' not in HTML, "登录页不该默认隐藏：首帧就是它"
    assert 'id="app-shell" hidden' in HTML, "应用壳应默认隐藏，由脚本按登录态放行"


def test_hidden_is_forced_globally_and_scoped_overrides_use_important():
    """全局 [hidden] 带 !important，放行规则必须也是 !important 才压得住。

    这是一处踩过的坑：文件的 `[hidden] { display: none !important }` 会让
    任何普通声明都失效，包括想"把 hidden 元素重新显示出来"的那条。
    漏掉 !important 的表现是首帧全白——登录页被隐藏了，应用却没被放出来，
    而在样式表里检查这条规则完全看不出问题。
    """
    assert re.search(r"\[hidden\]\s*\{\s*display:\s*none\s*!important", HTML), \
        "全局 [hidden] 规则变了，.auth-passed 的放行规则需要重新评估"
    assert ".auth-passed .app-shell[hidden]" in HTML, "缺少首帧放行应用壳的规则"
    rule = HTML[HTML.index(".auth-passed .app-shell[hidden]"):]
    rule = rule[:rule.index("}")]
    assert "!important" in rule, \
        "放行规则漏了 !important，会被全局 [hidden] 规则吃掉，首帧全白"


def test_gate_and_shell_are_toggled_together():
    """开/关登录页时必须同时切两个节点。

    只切一个会留下"门开着但应用也在"或"门关了但底下什么都没有"，
    两者都很难查，而且从截图上分不出来。
    """
    for fn_name in ("showAuthGate", "hideAuthGate"):
        body = _fn_body(fn_name)
        assert "auth-gate" in body, f"{fn_name} 没有操作登录页"
        assert "app-shell" in body, f"{fn_name} 没有操作应用壳"


def test_head_script_decides_first_frame_before_paint():
    """放行标记要在 <head> 里同步读，避免游客刷新时闪一下登录页。

    放到页面底部的主脚本就晚了——那时登录页已经画出来了。
    """
    head = HTML[:HTML.index("</head>")]
    assert "ironmind_guest_pass" in head or "GUEST_PASS_KEY" in head, \
        "head 里没有读游客放行标记"
    assert "auth-passed" in head, "head 里没有为首帧加放行 class"
    assert ".auth-passed" in HTML, "缺少 .auth-passed 的样式规则"


def test_logout_returns_to_the_login_page():
    """退出登录要回登录页，并清掉"上次登录过"的本机提示。

    留着提示会让下次刷新乐观地进应用，再因为会话已失效被弹回来。
    """
    body = _fn_body("handleLogout")
    assert "AUTH_HINT_KEY" in body, "退出时要清掉登录提示"
    assert "showAuthGate()" in body, "退出后应回到登录页"


def test_logout_also_clears_the_guest_pass():
    """登出必须一并清掉游客放行标记，否则刷新会绕过登录页。

    踩过的坑：只清 AUTH_HINT_KEY 时，曾点过游客（guest_pass=1）再登录的用户，
    登出后一刷新就被 <head> 那段脚本当成游客放行了——登出形同虚设。
    """
    body = _fn_body("handleLogout")
    assert "GUEST_PASS_KEY" in body, \
        "登出没清游客标记，刷新会绕过登录页（登出等于没生效）"


def test_login_page_has_login_and_register_modes():
    """登录与注册是两种模式，必须有可见的模式状态。

    原先注册是藏在游客按钮下面的小字链接，新用户填完账号密码点「登录」
    只会得到"用户名或密码不正确"，没有任何"我还没账号"的出口。
    """
    for element_id in ("gate-login", "gate-register", "gate-submit", "gate-title", "gate-sub"):
        assert f'id="{element_id}"' in HTML, f"缺少登录页元素 #{element_id}"
    assert "function setGateMode(" in HTML, "缺少模式切换函数"
    assert "GATE_COPY" in HTML, "缺少按模式区分的文案表"
    # 用 radiogroup：这是"二选一的模式"，不是"切出另一块面板"。
    # 曾经用过 role="tab" + aria-controls="gate-panel"，而那个 id 根本不存在。
    assert 'role="radiogroup"' in HTML, "登录/注册应做成单选组（role=radiogroup）"
    assert 'aria-controls="gate-panel"' not in HTML, \
        "aria-controls 指向了不存在的元素"


def test_tab_activation_is_scoped_to_its_own_tablist():
    """页签切换只能在自己的 tablist 里生效，不能全文档扫描。

    踩过的坑（登录后内容区一片空白）：activateTab 原本扫
    `document.querySelectorAll('[role=\"tab\"]')`，页签容器也用
    `querySelector('[role=\"tablist\"]')` 取"第一个"。登录页那组在 DOM 里更靠前，
    于是——
      ① 主界面的页签一个监听器都绑不上（绑到了登录页那组），点任何页签没反应；
      ② 点一次登录页的页签，主界面每个面板都被置成 hidden，登录进去全空白。
    """
    body = _fn_body("activateTab")
    assert "closest(" in body, "activateTab 没有按所属 tablist 限定范围"
    assert "document.querySelectorAll(" not in body, \
        "activateTab 仍在全文档扫描页签，会被页面上的其他页签组带偏"

    # 页签容器的取法也必须点名，不能靠"第一个"
    assert re.search(r"querySelector\(\"?\.feature-tabs", HTML), \
        "页签容器没有点名主界面那一组，会取到登录页的"


def test_login_mode_selector_reflects_selection():
    """登录页的模式选择器要用 aria-checked 表达选中态。"""
    assert "aria-checked" in _fn_body("setGateMode"), \
        "setGateMode 没有同步选中态"
    assert '.login-tab[aria-checked="true"]' in HTML, \
        "样式没有跟进 aria-checked"


def test_switching_mode_updates_copy_and_password_autocomplete():
    """切模式时标题、主按钮文案、密码框 autocomplete 都要跟着变。

    注册时若仍是 current-password，浏览器会尝试填入一个已有密码，
    而不是提示生成新密码——注册场景下最不该发生的事。
    """
    body = _fn_body("setGateMode")
    for token in ("gate-title", "gate-sub", "gate-submit", "autocomplete", "aria-checked"):
        assert token in body, f"setGateMode 没有更新 {token}"


def test_enter_key_submits_the_current_mode_not_always_login():
    """回车必须按当前模式提交。

    原先两个输入框都硬绑到 gateAuth('login')，切到注册页签后按回车
    仍然去登录，必然得到"用户名或密码不正确"——新用户按回车 100% 撞墙。
    """
    init = HTML[HTML.index("bindEnterToSubmit([") :]
    init = init[: init.index("]);") + 3]
    assert "gateAuth(gateMode)" in init, \
        "登录页的回车没有按当前模式提交"
    assert "gateAuth('login')" not in init, \
        "登录页的回车仍被硬绑到 login"


def test_submit_button_is_wired_to_the_current_mode():
    """主按钮要提交当前模式的表单，而不是固定登录。"""
    assert re.search(
        r"getElementById\('gate-submit'\)[^;]*gateAuth\(gateMode\)", HTML
    ), "主按钮没有按当前模式提交"


def test_register_shows_a_creation_receipt():
    """注册成功后要有明确的"账号已创建"回执。

    注册是自动登录直接进应用的（业界主流：注册完再要求登录一次是重复劳动），
    但用户会直接掉进一个空应用，没有任何"账号建成了"的确认——新账号本来就没数据，
    看起来像没成功。所以主内容区第一眼要有一条回执，并指出下一步。
    """
    for element_id in ("welcome-banner", "welcome-banner-title",
                       "welcome-banner-text", "welcome-banner-close"):
        assert f'id="{element_id}"' in HTML, f"缺少注册回执元素 #{element_id}"
    assert "function showWelcomeBanner(" in HTML, "缺少注册回执的显示逻辑"
    assert "function hideWelcomeBanner(" in HTML, "缺少注册回执的关闭逻辑"

    gate = _fn_body("gateAuth")
    assert "showWelcomeBanner(" in gate, "注册成功后没有显示回执"


def test_register_and_login_say_different_things():
    """注册与登录的成功文案必须区分。

    踩过的坑：登录页那份把两种措辞合并了，注册成功后提示的是"已登录为 X"，
    等于没告诉用户账号是不是真建成了（侧栏那份一直是分开写的）。
    """
    gate = _fn_body("gateAuth")
    assert "已创建并登录为" in gate, "注册成功没有说明「账号已创建」"
    assert "已登录为" in gate, "登录成功文案缺失"


def test_logout_clears_the_registration_receipt():
    """登出要收掉注册回执，否则换个身份进来还挂着"账号已创建"。"""
    assert "hideWelcomeBanner()" in _fn_body("handleLogout"), \
        "登出没有收起注册回执"


def test_guest_pass_is_remembered_locally():
    """选过游客之后刷新不该反复弹登录页——那等于把游客挡在门外。"""
    fn = _fn_body("enterAsGuest")
    assert "localStorage" in fn, "游客放行要记在本机"

    init = HTML[HTML.index("loadSession().then("):]
    init = init[:init.index("});") + 3]
    assert "GUEST_PASS_KEY" in init, "初始化时应读游客放行标记"
    assert "currentUser" in init, "已登录会话应直接放行"


def test_gate_does_not_duplicate_auth_logic():
    """登录页与侧栏共用同一套登录/注册请求，不各写一份。"""
    gate = _fn_body("gateAuth")
    assert "postAuth(" in gate, "登录页应复用 postAuth，而不是自己拼 fetch"
    assert "/api/auth/login" not in gate and "/api/auth/register" not in gate, \
        "端点拼接应收敛在 postAuth 里"


def test_saved_plans_panel_and_save_action_exist():
    """计划要能存到服务端并在界面上找回。

    已保存列表原先挂在侧栏底部（id="saved-plans"），位置在折叠线以下，
    用户存完计划找不到入口；现在它是「我的计划」tab 里的一个面板。
    """
    for element_id in ("panel-saved", "saved-plans-list", "saved-plans-empty"):
        assert f'id="{element_id}"' in HTML
    for helper in ("autoSavePlan", "migrateLocalHistory", "loadSavedPlans",
                   "openSavedPlan", "deleteSavedPlan"):
        assert f"function {helper}(" in HTML, f"缺少 {helper}"

    fn = _fn_body("autoSavePlan")
    # 存不进去要如实返回失败，调用方据此提示——不能假装存好了
    assert "catch" in fn and "return resp.ok" in fn


def test_saved_plan_view_switches_back_to_the_plan_tab():
    """点「查看」必须切回「训练计划」tab。

    计划渲染进的是 panel-plan 里的 #plan-result。若只渲染不切 tab，
    内容是写进了一个隐藏面板，随后的 scrollIntoView 作用在隐藏元素上
    也不会滚动——用户看到的是"点了查看，什么都没发生"。
    """
    fn = _fn_body("openSavedPlan")
    assert "activateTab('tab-plan')" in fn, "查看已保存计划后必须切回训练计划 tab"
    # 切换必须发生在滚动之前，否则滚的还是隐藏元素
    assert fn.index("activateTab('tab-plan')") < fn.index("scrollIntoView"), \
        "先切 tab 再滚动"


def test_saved_plan_view_does_not_offer_saving_again():
    """从历史打开的计划已在库里，不该再显示"保存这份计划"。"""
    fn = _fn_body("openSavedPlan")
    assert "showPlanSaveBarHidden" in fn


def test_exercise_demo_uses_the_static_mount():
    """图片路径必须与后端挂载点一致。

    项目原本没有静态目录挂载（首页是单独 FileResponse 返回的），
    所以"图片放进 app/static 就能被取到"这个假设不成立——
    路径对不上时前端只会看到一片裂图。
    """
    server_py = (Path(__file__).resolve().parent.parent / "app/server.py").read_text(encoding="utf-8")
    assert 'app.mount("/static"' in server_py, "后端需要挂载静态目录"

    fn = _fn_body("buildExerciseDemo")
    assert "'/static/'" in fn, "图片 URL 必须带 /static 前缀以匹配挂载点"


def test_plan_title_is_rendered():
    """计划名必须显示出来。

    模型会给计划起描述性的名字（如「3天低冲击减脂循环计划（膝盖友好版）」），
    但这个字段此前算完就丢——用户从没看到过，从历史里打开某一份时
    也分不清打开的是哪一份。
    """
    fn = _fn_body("renderPlanResult")
    assert "plan_name" in fn, "计划名要渲染"


def test_enter_key_submits_login_forms():
    """登录页与侧栏都要支持回车提交。

    登录页不是 <form>，侧栏那份虽用了 form 但配的是 onsubmit="return false"
    且按钮 type="button"——两处原本都不响应回车，用户输完密码按回车没反应。
    """
    assert "bindEnterToSubmit" in HTML
    init = HTML[HTML.index("bindEnterToSubmit(["):]
    init = init[:init.index("]);") + 3]
    for field in ("gate-username", "gate-password", "auth-username", "auth-password"):
        assert f"'{field}'" in init, f"{field} 应绑定回车提交"
    fn = _fn_body("bindEnterToSubmit")
    assert "preventDefault" in fn, "要阻止隐式提交导致页面刷新"


# ----------------------------------------------------------------------
# 侧栏信息架构
# ----------------------------------------------------------------------

def _sidebar_html():
    start = HTML.index('<aside class="sidebar')
    return HTML[start:HTML.index("</aside>", start)]


def test_account_block_sits_after_the_primary_action():
    """账号区在侧栏最底部，主流程（档案 → 生成计划）排在它前面。

    原先它在品牌正下方、占 110px，还是一个卡片里套一个描边按钮（边框套边框）。
    但登录是一次性动作，把最贵的位置留给它，等于让主流程往下挤。
    """
    sb = _sidebar_html()
    assert sb.index('id="account-block"') > sb.index('id="btn-generate"'), \
        "账号区应在「生成计划」之后（侧栏底部）"
    assert sb.index('id="account-block"') > sb.index('id="profile-editor"'), \
        "账号区应在档案表单之后"


def test_account_actions_are_buttons():
    """账号区的动作是按钮，不是下划线文字链接。

    曾经写成文字链接，理由是"设置项不该和「生成计划」一样重"。但渲染出来
    只是左下角一行孤零零的小字：既不像能点，也和上面的输入框对不齐——
    「退出登录」尤其像掉了样式的残留。改回按钮；不抢主行动靠中性色。
    """
    sb = _sidebar_html()
    assert ".account-link" not in sb, "账号动作不该再用文字链接样式"
    for btn_id in ("btn-show-login", "btn-logout"):
        m = re.search(r'<button class="([^"]*)"[^>]*id="%s"' % btn_id, sb)
        assert m, f"{btn_id} 应仍是 button 元素"
        assert "account-btn" in m.group(1), f"{btn_id} 应该用按钮样式"
    assert ".account-link" not in HTML, "文字链接的样式规则应已删除，别留死代码"


def test_account_actions_share_the_identity_row():
    """动作块必须在 .account-state 这个 grid 内部，才能和身份同一行。

    它曾经是 grid 的兄弟节点，于是 CSS 里那两条
    `grid-column: 3; grid-row: 1` 从来没生效过：动作被甩到第三行，
    孤零零挂在左下角，和上面的身份文字对不齐。结构不对，样式写对也没用。
    """
    sb = _sidebar_html()
    state_at = sb.index('id="account-state"')
    actions_at = sb.index('id="account-actions"')
    hint_at = sb.index('id="account-hint"')
    assert state_at < actions_at < hint_at, \
        "账号动作应嵌在 .account-state 里（hint 之前）"


def test_primary_cta_cannot_be_squashed_by_an_overflowing_sidebar():
    """「生成计划」在侧栏溢出时必须保持完整高度。

    侧栏是「height:100vh + overflow-y:auto」的纵向 flex 容器。内容一旦超出
    视口，flex 项默认被压缩；而这个按钮带 overflow:hidden（扫光要裁剪），
    自动最小尺寸因此失效、能被一路压到 0——实测 57px 高塌成 32px，
    内边距仍是 16+16，文字被裁掉，主行动看起来像"被挡住了"。
    """
    start = HTML.index(".btn-generate {")
    rule = HTML[start:HTML.index("}", start)]
    assert "flex-shrink: 0" in rule, \
        "主按钮必须禁止 flex 压缩，否则侧栏一溢出就被压扁"


def test_profile_summary_is_hidden_on_desktop():
    """桌面端不显示档案摘要卡——它与下面的表单完全重复。

    表单在桌面本来就常显，同样的 180cm/80kg/增肌/每周4练 再列一遍只是噪声，
    而且和表单叠在一起像两张卡片。窄屏保留：那里表单折叠，它是概览与展开入口。
    """
    start = HTML.index(".profile-summary {")
    block = HTML[start:HTML.index("}", start)]
    assert "display: none" in block, "桌面端应隐藏档案摘要"
    # 窄屏媒体查询里必须恢复
    mobile = HTML[HTML.index("@media (max-width: 767px)"):]
    mobile = mobile[:mobile.index(".profile-toggle")]
    assert ".profile-summary" in mobile, "窄屏应恢复摘要卡（它是唯一的展开入口）"


def test_all_profile_fields_live_in_one_group():
    """档案字段全在 #profile-editor 里，折叠只需要一个开关。

    原先训练场景/伤病史/运动习惯散在 #profile-editor 外面，得再靠
    `.sidebar > .field` 单独收一遍——两套机制容易不同步，展开后只出来一半。
    """
    sb = _sidebar_html()
    editor = sb[sb.index('id="profile-editor"'):]
    editor = editor[:editor.index("</div>", editor.index('id="habits-input"'))]
    for field_id in ("height", "weight", "years", "goal", "days",
                     "injuries-input", "habits-input"):
        assert f'id="{field_id}"' in editor, f"{field_id} 应在档案分组内"
    assert 'id="scene-tags"' in editor, "训练场景应在档案分组内"
    assert "profile-expanded" not in HTML, "第二套折叠机制应已移除"


def test_sidebar_scrolls_on_its_own():
    """侧栏约束成视口高度并自己滚动，页面滚动时它留在原地。

    侧栏内容比常见视口高（7 个字段），不这么做的话「生成计划」会被折叠线切掉。
    """
    start = HTML.index(".sidebar {")
    block = HTML[start:HTML.index("}", start)]
    assert "height: 100vh" in block and "overflow-y: auto" in block
    assert "position: sticky" in block


# ----------------------------------------------------------------------
# 档案字段：年龄与性别
#
# 这两个字段只服务于热量目标（Mifflin-St Jeor 的性别常数项男女相差 166 kcal），
# 不参与动作编排。前端必须真的把它们送上去，否则后端只能走粗估退路。
# ----------------------------------------------------------------------

def test_profile_form_collects_age_and_sex():
    assert 'id="age"' in HTML, "档案表单缺少年龄字段"
    assert 'id="sex"' in HTML, "档案表单缺少性别字段"
    # 「不填」必须是可选项且是默认项：营养估算用得上的信息，不值得让用户
    # 先回答一个涉及身份的问题。
    assert '<option value="不填" selected>' in HTML, \
        "性别的默认值应为「不填」，而不是替用户预选一个"


def test_get_profile_sends_age_and_sex():
    fn = _fn_body("getProfile")
    assert "document.getElementById('age')" in fn, "getProfile 没带上年龄"
    assert "document.getElementById('sex')" in fn, "getProfile 没带上性别"


def test_plan_result_renders_the_nutrition_block():
    """热量目标是算出来的，但用户得看得见。"""
    assert "function renderPlanNutrition(" in HTML, "缺少热量块渲染函数"
    assert "renderPlanNutrition(container, plan.nutrition)" in _fn_body("renderPlanResult"), \
        "计划结果区没有渲染 nutrition"


def test_nutrition_block_shows_the_basis_not_just_the_number():
    """依据文本与数字同等重要。

    这个数字是确定性算出来的，但如果界面上只有数字，它对用户就是一个
    无从检验的断言——依据就是它敢做成确定性的理由。
    """
    fn = _fn_body("renderPlanNutrition")

    assert "nutrition-value" in fn and "nutrition-basis" in fn
    assert "nutrition.basis" in fn


def test_explanation_panel_lists_the_habits_that_were_considered():
    """用户问过"我填的习惯到底有没有被用上"，面板要给得出答案。"""
    fn = _fn_body("renderPlanExplanation")

    assert "explain.habits" in fn, "可解释面板没有读取 habits"
    assert "运动习惯" in fn, "可解释面板没有展示运动习惯"
