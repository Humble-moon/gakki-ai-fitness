"""训练执行面板的前端契约（独立文件，不与在途改动的契约文件抢锚点）。

钉住两件事，都是真实事故：
1. 「完成训练」保存失败——plan_id 在库里是整数、接口字段是字符串，
   前端直接提交数字 → 422 string_type，用户练完一整场在最后一步被拒。
2. 切走应用/刷新丢进度——训练面板原先纯内存，浏览器回收页面后
   练过的组全部消失。现在进度落本机草稿，任何页签可见续练入口。

契约读源文件（与饮食契约同范式）：结构约定用文本断言钉住，
行为正确性由浏览器实测与 test_training_api 的接口测试覆盖。
"""
import re
from pathlib import Path

HTML = Path("app/static/index.html").read_text(encoding="utf-8")

WORKOUT_BLOCK_START = "let workoutState = null;"
WORKOUT_BLOCK_END = "function formatLogDate("


def _fn_body(name):
    start = re.search(rf"^(?:async )?function {re.escape(name)}\(", HTML, re.M)
    assert start, f"找不到函数 {name}"
    nxt = re.search(r"^(?:async )?function ", HTML[start.end():], re.M)
    end = start.end() + nxt.start() if nxt else len(HTML)
    return HTML[start.start():end]


def _workout_block():
    start = HTML.index(WORKOUT_BLOCK_START)
    end = HTML.index(WORKOUT_BLOCK_END, start)
    return HTML[start:end]


class TestSaveFailureFix:
    def test_plan_id_submitted_as_string(self):
        body = _fn_body("finishWorkout")
        assert "payload.plan_id = String(planMeta.planId)" in body

    def test_error_message_not_generic_double(self):
        body = _fn_body("finishWorkout")
        # 旧写法 (data.error && data.error.message) || '保存失败' 在 422 时
        # 产出「保存失败：保存失败」；必须走可读的错误解释函数。
        assert "workoutSaveErrorMessage(data)" in body
        assert "|| '保存失败'" not in body

    def test_422_detail_translated_to_actionable_text(self):
        body = _fn_body("workoutSaveErrorMessage")
        assert "data.detail" in body
        assert "刷新页面后重试" in body
        # 面向用户的文案不得泄漏字段名/类型等内部细节
        assert "string_type" not in body
        assert "plan_id" not in body

    def test_non_json_response_does_not_leak_parse_error(self):
        body = _fn_body("finishWorkout")
        assert "catch (parseError)" in body

    def test_duration_capped_to_schema_limit(self):
        body = _fn_body("finishWorkout")
        assert "Math.min(1440" in body

    def test_success_clears_draft_before_close(self):
        body = _fn_body("finishWorkout")
        assert body.index("clearWorkoutDraft();") < body.index("closeWorkout(true);")

    def test_failure_keeps_draft(self):
        body = _fn_body("finishWorkout")
        catch = body[body.index("} catch (e) {"):]
        assert "persistWorkoutDraft();" in catch


class TestDraftPersistence:
    def test_draft_key_and_version(self):
        block = _workout_block()
        assert "const WORKOUT_DRAFT_KEY = 'ironmind_workout_draft';" in block
        assert "const WORKOUT_DRAFT_VERSION = 1;" in block

    def test_draft_bound_to_identity_and_freshness(self):
        body = _fn_body("readWorkoutDraft")
        assert "draft.athleteKey !== athleteKey" in body
        assert "WORKOUT_DRAFT_MAX_AGE_MS" in body
        assert "draft.v !== WORKOUT_DRAFT_VERSION" in body

    def test_input_changes_schedule_debounced_save(self):
        body = _fn_body("buildWorkoutSetRow")
        assert body.count("scheduleWorkoutDraftSave();") == 2

    def test_set_completion_persists_immediately(self):
        body = _fn_body("toggleWorkoutSet")
        assert "persistWorkoutDraft();" in body

    def test_pagehide_and_hidden_force_save(self):
        block = _workout_block()
        assert "window.addEventListener('pagehide', () => persistWorkoutDraft());" in block
        vis = block[block.index("document.addEventListener('visibilitychange'"):]
        vis = vis[:vis.index("});")]
        assert "persistWorkoutDraft();" in vis
        assert "clearTimeout(workoutDraftTimer)" in vis

    def test_empty_session_leaves_no_draft(self):
        body = _fn_body("persistWorkoutDraft")
        assert "if (!progress.doneSets && !hasInput) return;" in body

    def test_cross_tab_storage_refresh(self):
        block = _workout_block()
        assert "window.addEventListener('storage'" in block
        assert "event.key === WORKOUT_DRAFT_KEY" in block


class TestResumeEntryPoints:
    def test_banner_in_html_before_tabs_and_hidden_by_default(self):
        assert 'id="workout-resume-banner"' in HTML
        banner = HTML[HTML.index('<div class="resume-banner"'):]
        banner = banner[:banner.index("</div>") + 6]
        assert "hidden" in banner
        assert 'role="status"' in banner
        # 横幅必须在页签之前的第一屏区域：任何页签都看得见
        assert HTML.index('id="workout-resume-banner"') < HTML.index('class="tabs feature-tabs"')

    def test_banner_buttons_bound(self):
        body = _fn_body("initTrainingTabs")
        assert "workout-resume-go" in body
        assert "resumeWorkoutFromDraft" in body
        assert "workout-resume-discard" in body
        assert "discardWorkoutDraft" in body

    def test_banner_hidden_while_panel_open(self):
        body = _fn_body("refreshWorkoutResumeBanner")
        assert "workoutState ? null : readWorkoutDraft()" in body

    def test_banner_refresh_after_identity_ready(self):
        body = _fn_body("initTrainingTabs")
        assert "refreshWorkoutResumeBanner();" in body
        assert body.index("loadSavedProfile();") < body.index("refreshWorkoutResumeBanner();")

    def test_plan_day_button_upgrades_to_resume(self):
        body = _fn_body("renderPlanDays")
        assert "'▸ 继续训练'" in body
        assert "is-resume" in body
        assert "openWorkout(draft.day, draft.planMeta, draft)" in body

    def test_open_workout_accepts_resume_and_auto_resumes_same_session(self):
        body = _fn_body("openWorkout")
        assert "function openWorkout(day, planMeta, resume)" in HTML
        assert "sameSession" in body
        assert "openWorkout(existing.day, existing.planMeta, existing); return;" in body

    def test_different_day_requires_overwrite_confirm(self):
        body = _fn_body("openWorkout")
        assert "开始新训练会覆盖那份进度" in body

    def test_resume_validates_structure_before_applying(self):
        body = _fn_body("openWorkout")
        assert "resume.entries.length === entries.length" in body
        assert "saved?.name === entries[index].name" in body

    def test_restored_values_marked_touched_against_hydration(self):
        body = _fn_body("openWorkout")
        assert "input.dataset.touched = '1'" in body

    def test_discard_requires_confirm(self):
        body = _fn_body("discardWorkoutDraft")
        assert "showConfirmDialog" in body
        assert "window.confirm" not in body
        assert "clearWorkoutDraft();" in body


class TestWallClockCatchUp:
    def test_named_tick_functions(self):
        assert "function tickWorkoutClock()" in HTML
        assert "function tickWorkoutRest()" in HTML
        body = _fn_body("openWorkout")
        assert "setInterval(tickWorkoutClock, 1000)" in body
        rest = _fn_body("startWorkoutRest")
        assert "setInterval(tickWorkoutRest, 250)" in rest

    def test_visible_branch_recomputes_both_clocks(self):
        block = _workout_block()
        vis = block[block.index("document.addEventListener('visibilitychange'"):]
        vis = vis[:vis.index("});")]
        assert "tickWorkoutClock();" in vis
        assert "tickWorkoutRest();" in vis

    def test_stop_rest_zeroes_end_timestamp(self):
        body = _fn_body("stopWorkoutRest")
        assert "workoutState.restEndAt = 0;" in body

    def test_away_pause_rule(self):
        block = _workout_block()
        assert "const WORKOUT_AWAY_PAUSE_MS = 2 * 60 * 60 * 1000;" in block
        body = _fn_body("openWorkout")
        assert "away <= WORKOUT_AWAY_PAUSE_MS" in body
        assert "Date.now() - elapsedMs" in body


class TestCloseIsPause:
    def test_close_persists_and_no_longer_confirms(self):
        body = _fn_body("closeWorkout")
        assert "if (!force) persistWorkoutDraft();" in body
        assert "window.confirm" not in body
        assert "if (!force) refreshWorkoutResumeBanner();" in body


class TestWorkoutBlockHygiene:
    def test_no_innerhtml_or_native_prompt_in_workout_block(self):
        block = _workout_block()
        assert "innerHTML" not in block
        assert "prompt(" not in block

    def test_resume_css_present(self):
        assert ".resume-banner[hidden] { display: none; }" in HTML
        assert ".plan-start-btn.is-resume" in HTML
        assert ".resume-banner-go:hover" in HTML


class TestRestEndAlert:
    """休息到点必须被注意到：原先倒计时归零只是静默回到空闲文案。"""

    def test_natural_end_announces_skip_does_not(self):
        tick = _fn_body("tickWorkoutRest")
        assert "stopWorkoutRest(true)" in tick
        # 后台不烧掉提醒窗口，回前台补算时再宣布
        assert "document.visibilityState === 'visible'" in tick
        build = _fn_body("openWorkout")
        assert "skip.addEventListener('click', () => stopWorkoutRest(false))" in build
        stop = _fn_body("stopWorkoutRest")
        assert "announceRestEnd()" in stop
        assert "休息结束 ✓ 开始下一组" in stop

    def test_three_modes_cycle_and_persist(self):
        block = _workout_block()
        assert "const REST_ALERT_KEY = 'ironmind_rest_alert_mode';" in block
        assert "full: '声音＋振动', silent: '仅振动', off: '仅画面'" in block
        build = _fn_body("openWorkout")
        assert "提醒：" in build and "syncAlertBtn" in build
        assert "localStorage.setItem(REST_ALERT_KEY, next)" in build

    def test_audio_unlocked_on_user_gesture_only(self):
        toggle = _fn_body("toggleWorkoutSet")
        assert "unlockRestAudio();" in toggle
        beep = _fn_body("playRestEndBeep")
        assert "createOscillator" in beep
        # 出声失败不抛断训练
        assert "catch" in beep

    def test_visual_channel_always_on_and_reduced_motion_safe(self):
        assert ".workout-rest.is-ended {" in HTML
        assert "@keyframes rest-end-pulse" in HTML
        stop = _fn_body("stopWorkoutRest")
        assert "classList.add('is-ended')" in stop
        # 文案通道不依赖动画：3 秒后回到记住时长的空闲文案
        assert "restIdleText()" in stop
        idle = _fn_body("restIdleText")
        assert "lastRestSeconds" in idle

    def test_wake_lock_acquired_released_and_reacquired(self):
        build = _fn_body("openWorkout")
        assert "acquireWorkoutWakeLock();" in build
        close = _fn_body("closeWorkout")
        assert "releaseWorkoutWakeLock();" in close
        block = _workout_block()
        vis = block[block.index("document.addEventListener('visibilitychange'"):]
        vis = vis[:vis.index("});")]
        assert "acquireWorkoutWakeLock();" in vis
        acq = _fn_body("acquireWorkoutWakeLock")
        assert "'wakeLock' in navigator" in acq
        assert "if (!workoutState)" in acq  # 请求期间面板被关→释放，不留锁


class TestMobileReachability:
    """44px 命中区下限要真的生效；白字橙底主按钮必须落在 WCAG 大字档。"""

    def test_floor_has_specificity_not_where(self):
        assert "button:not([data-compact]), [role=\"tab\"], #file-upload-area { min-height: 44px; }" in HTML
        assert ":where(button, [role=\"tab\"], #file-upload-area) { min-height: 44px; }" not in HTML

    def test_named_overrides_raised_to_44(self):
        assert ".profile-toggle { min-height: 44px;" in HTML
        assert ".resume-banner-go { order: 3; flex: 1 1 100%; min-height: 44px; }" in HTML
        assert ".resume-banner-close { order: 2; width: 44px; height: 44px; }" in HTML
        assert ".set-check { width: 44px; height: 44px; }" in HTML
        assert ".diet-chip { min-height: 44px; }" in HTML
        assert ".diet-log-food-row .log-entry-remove { width: 44px; height: 44px; font-size: 15px; }" in HTML

    def test_orange_white_ctas_in_large_text_tier(self):
        login = HTML[HTML.index(".login-btn {"):HTML.index(".login-btn:active")]
        assert "font-size: 19px; font-weight: 700;" in login
        assert "#btn-generate { min-height: 48px; padding: 10px 12px; font-size: 19px; }" in HTML