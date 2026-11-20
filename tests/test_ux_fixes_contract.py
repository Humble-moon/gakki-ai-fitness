"""体验修复批次的全局契约（计划等待出口 / 刷新恢复 / 食物补全接线）。

不属于饮食块也不属于训练块的改动落在这里，避免与在途改动的
test_web_frontend_contract.py 抢锚点。契约读源文件，行为正确性由
浏览器实测与接口测试覆盖。
"""
import re
from pathlib import Path

HTML = Path("app/static/index.html").read_text(encoding="utf-8")


def _fn_body(name):
    start = re.search(rf"^(?:async )?function {re.escape(name)}\(", HTML, re.M)
    assert start, f"找不到函数 {name}"
    nxt = re.search(r"^(?:async )?function ", HTML[start.end():], re.M)
    end = start.end() + nxt.start() if nxt else len(HTML)
    return HTML[start.start():end]


class TestPlanWaitExit:
    """47 秒的生成等待必须有出口：取消按钮 + 120 秒硬超时。"""

    def test_cancel_button_exists_and_wired(self):
        assert 'id="btn-cancel-plan"' in HTML
        assert "onclick=\"cancelPlanGeneration()\"" in HTML
        assert "activeRequests.plan?.abort();" in _fn_body("cancelPlanGeneration")

    def test_cancel_visible_only_while_loading(self):
        assert "cancel: 'btn-cancel-plan'" in HTML
        fn = _fn_body("setRequestState")
        assert "cancelBtn.hidden = !isLoading;" in fn

    def test_hard_timeout_120s_with_distinct_message(self):
        fn = _fn_body("generatePlan")
        assert "}, 120000);" in fn
        assert "timedOut = true; controller.abort();" in fn
        assert "生成超过 120 秒已自动停止" in fn
        assert "已取消生成。档案与要求都还在" in fn
        assert "window.clearTimeout(timeoutId);" in fn


class TestRefreshRestoresPlan:
    """刷新后不该对着空态失忆：最近一份已保存计划安静落回结果区。"""

    def test_restore_only_when_welcome_present(self):
        fn = _fn_body("restoreLatestPlanOnWelcome")
        assert fn.count(".welcome-area") >= 2  # 进门前与落位前各查一次
        assert "limit=1" in fn
        assert "renderPlanResult(area, plan)" in fn
        assert "showPlanSaveBarHidden()" in fn
        assert "activateTab" not in fn and "scrollIntoView" not in fn
        assert "生成新计划会替换它" in fn

    def test_restore_runs_after_identity_and_migration(self):
        assert "migrateLocalHistory().then(() => {" in HTML
        init = HTML[HTML.index("migrateLocalHistory().then(() => {"):]
        init = init[:init.index("});")]
        assert "loadSavedPlans();" in init
        assert "restoreLatestPlanOnWelcome();" in init


class TestFoodAutocomplete:
    """食物名靠记忆拼写必然出错：目录补全 + 最近吃过 + 出错行标红。"""

    def test_datalist_wired_to_name_input(self):
        assert '<datalist id="diet-food-options"></datalist>' in HTML
        assert "name.setAttribute('list', 'diet-food-options');" in _fn_body("addDietFoodRow")

    def test_catalog_fetched_once_and_degrades(self):
        fn = _fn_body("ensureDietFoodCatalog")
        assert "/api/diet-foods" in fn
        assert "if (dietFoodCatalog) return dietFoodCatalog;" in fn
        assert "return null;" in fn  # 拉不到退回手填，不拦记录
        assert "ensureDietFoodCatalog();" in _fn_body("initDietTab")
        assert "ensureDietFoodCatalog();" in _fn_body("openDietDrawer")

    def test_recent_chips_from_history(self):
        assert 'id="diet-recent-foods"' in HTML
        collect = _fn_body("collectDietRecentFoods")
        assert "slice(0, 6)" in collect
        assert "collectDietRecentFoods(logs);" in _fn_body("loadDietLogHistory")
        render = _fn_body("renderDietRecentChips")
        assert "addDietFoodRow({ name: item.name, grams: item.grams || '' })" in render
        assert "renderDietRecentChips();" in _fn_body("openDietDrawer")

    def test_server_row_error_highlighted(self):
        post = _fn_body("postDietLog")
        assert "err.foodIndex = data?.error?.food_index;" in post
        hl = _fn_body("highlightDietRowError")
        assert "Number.isInteger(err?.foodIndex)" in hl
        assert "classList.add('is-invalid')" in hl
        assert "highlightDietRowError(e);" in _fn_body("submitDietLog")
        assert ".diet-log-food-row.is-invalid .diet-custom-toggle" in HTML


class TestMinorObservations:
    """次要观察批量修复：原生弹窗清零／焦点圈／进度条合成器化／可读文案。"""

    def test_no_native_confirm_left(self):
        assert "window.confirm" not in HTML
        fn = _fn_body("showConfirmDialog")
        assert "role', 'alertdialog'" in fn
        assert "Escape" in fn
        assert "cancel.focus();" in fn
        assert "e.target === backdrop" in fn
        for caller in ("discardWorkoutDraft", "copyLogToToday", "clearMemory", "openWorkout"):
            assert "showConfirmDialog" in _fn_body(caller), caller

    def test_focus_trapped_and_background_inert(self):
        assert "function trapOverlayFocus(" in HTML
        assert "trapOverlayFocus(overlay)" in _fn_body("openWorkout")
        assert "workoutState.releaseFocusTrap?.();" in _fn_body("closeWorkout")
        assert "setAttribute('inert', '')" in _fn_body("openWorkout")
        assert "removeAttribute('inert')" in _fn_body("closeWorkout")
        assert "dietDrawerFocusTrap = trapOverlayFocus(drawer);" in _fn_body("openDietDrawer")
        assert "dietDrawerFocusTrap?.();" in _fn_body("closeDietDrawer")

    def test_progress_bar_runs_on_compositor(self):
        assert "transition: width" not in HTML
        assert "transform: scaleX(0);" in HTML
        assert "barFill.style.transform = `scaleX(" in _fn_body("updateWorkoutProgress")

    def test_workout_inputs_have_names(self):
        fn = _fn_body("buildWorkoutSetRow")
        assert "组重量（公斤）" in fn and "组次数" in fn

    def test_start_button_solid_ink_white(self):
        block = HTML[HTML.index(".plan-start-btn {"):HTML.index(".plan-start-btn:hover")]
        assert "background: var(--accent-ink);" in block
        assert "color: #fff;" in block

    def test_decorative_marks_hidden_from_readers(self):
        assert '<div class="ec-mark">' not in HTML
        assert "markNode.setAttribute('aria-hidden', 'true');" in _fn_body("composedEmpty")

    def test_heading_level_and_plain_labels(self):
        assert "<h2>Ready to Forge</h2>" in HTML
        assert "<h2>Motion Inspector</h2>" in HTML
        assert ".empty-state h2 {" in HTML
        assert "${exercises.length} 个动作" in HTML
        assert "} EX`" not in HTML
        assert "完成训练」会自动记录到这里" in HTML

    def test_sources_deduped_and_readable(self):
        fn = _fn_body("renderSourceList")
        assert "seen.has(key)" in fn
        assert "readableSourceTitle(source)" in fn
        title = _fn_body("readableSourceTitle")
        assert "replace(/^\\d+[-_]/, '')" in title

    def test_diet_result_anchor_chips(self):
        fn = _fn_body("renderDietResult")
        assert "diet-anchor-chips" in fn
        assert "role', 'navigation'" in fn
        assert "prefers-reduced-motion: reduce" in fn
        assert "'instant' : 'smooth'" in fn

    def test_smallest_fonts_raised(self):
        fc = HTML[HTML.index(".feature-card .fc-desc {"):HTML.index(".feature-card .fc-accent")]
        assert "font-size: 12px;" in fc