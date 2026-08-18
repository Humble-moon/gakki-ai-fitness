from pathlib import Path


HTML = Path("app/static/index.html").read_text(encoding="utf-8")


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
    for marker in ("/api/generate-plan", "/api/analyze-exercise", "/api/ask-question"):
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
    for event_name in ("stage", "advice_chunk", "writer_chunk", "factcheck_done", "done"):
        assert event_name in HTML
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


def test_plan_result_uses_safe_dynamic_dom_and_responsive_cards():
    start = HTML.index("function renderPlanResult")
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


def test_plan_history_renders_into_result_layer():
    start = HTML.index("function restorePlan")
    end = HTML.find("// ── Plan Iteration", start)
    body = HTML[start:end]
    assert "plan-result" in body
    assert "renderPlanResult(document.getElementById('plan-result')" in body
    assert "plan-output.innerHTML" not in body


def test_no_result_root_destructive_clear():
    assert "resultArea.innerHTML = ''" not in HTML


def test_plan_mobile_card_copy_is_readable():
    assert ".plan-ex-name { flex: 1 1 100%;" in HTML
    assert ".plan-ex-notes { flex: 1 1 100%;" in HTML
    assert "休息 ${exercise?.rest || '60 秒'}" in HTML
    assert "${exercise?.reps ?? '?'} 次" in HTML




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
