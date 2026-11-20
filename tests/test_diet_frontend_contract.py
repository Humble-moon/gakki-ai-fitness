"""饮食功能的前端契约（独立文件，不与在途改动的契约文件抢锚点）。

钉住的是「改饮食界面时最容易顺手破坏」的全局约定与饮食专属安全链：
* consumeSse 计数仍为 4——饮食全 JSON，第 5 个流式消费会破坏既有断言；
* 顶层页签 7 个、饮食页签 aria 四件套齐全、二级页签 scoped；
* 饮食 JS 块禁 innerHTML / 原生弹窗 / .sr-only / 内部术语；
* 自动收录守卫先于收录请求；危机终态有可见出口；
* 生成走 const payload + AbortController 超时 + setBusy；
* 饮食字段不进 PROFILE_FIELD_RULES（与 PlanRequest 的对齐契约不被污染）。
"""
import re
from pathlib import Path

HTML = Path("app/static/index.html").read_text(encoding="utf-8")

DIET_BLOCK_START = "// ═══════════════ 饮食计划 / 饮食记录 / 摄入统计 ═══════════════"


def _fn_body(name):
    start = re.search(rf"^(?:async )?function {re.escape(name)}\(", HTML, re.M)
    assert start, f"找不到函数 {name}"
    nxt = re.search(r"^(?:async )?function ", HTML[start.end():], re.M)
    end = start.end() + nxt.start() if nxt else len(HTML)
    return HTML[start.start():end]


def _diet_block():
    start = HTML.index(DIET_BLOCK_START)
    end = HTML.index("function initTrainingTabs(", start)
    return HTML[start:end]


class TestGlobalGuards:
    def test_consume_sse_count_still_four(self):
        assert HTML.count("consumeSse(") == 4

    def test_first_const_body_still_belongs_to_generate_plan(self):
        first = HTML.index("const body = {")
        starts = list(re.finditer(r"^(?:async )?function (\w+)\(", HTML[:first], re.M))
        assert starts[-1].group(1) == "generatePlan"

    def test_top_level_feature_tabs_are_seven(self):
        tablist = HTML[HTML.index('class="tabs feature-tabs"'):HTML.index("</div>", HTML.index('class="tabs feature-tabs"'))]
        assert tablist.count('role="tab"') == 7
        assert 'id="tab-diet"' in tablist

    def test_profile_field_rules_have_no_diet_fields(self):
        start = HTML.index("const PROFILE_FIELD_RULES = [")
        end = HTML.index("];", start)
        rules = HTML[start:end]
        for diet_id in ("diet-allergies", "diet-avoid", "diet-health", "diet-notes"):
            assert diet_id not in rules


class TestDietTabStructure:
    def test_diet_tab_aria_four_piece(self):
        assert 'id="tab-diet"' in HTML
        assert 'aria-controls="panel-diet"' in HTML
        assert 'id="panel-diet" role="tabpanel" aria-labelledby="tab-diet"' in HTML
        assert 'class="tab-content feature-panel" id="panel-diet"' in HTML

    def test_subtablist_is_scoped_and_complete(self):
        assert 'class="tabs diet-subtabs" role="tablist"' in HTML
        for sub in ("generate", "saved", "log", "stats"):
            assert f'id="tab-diet-{sub}"' in HTML
            assert f'aria-controls="panel-diet-{sub}"' in HTML
            assert f'id="panel-diet-{sub}" role="tabpanel" aria-labelledby="tab-diet-{sub}"' in HTML

    def test_subtab_activation_reuses_scoped_activate_tab(self):
        block = _diet_block()
        assert "activateTab(tab.id)" in block
        assert "closest('[role=\"tab\"]')" in block or 'closest(\'[role="tab"]\')' in block

    def test_generate_button_uses_btn_generate_class(self):
        assert 'class="btn-generate" id="btn-generate-diet"' in HTML

    def test_form_fields_exist(self):
        for element_id in ("diet-allergies", "diet-avoid", "diet-health",
                           "diet-notes", "diet-log-date",
                           "diet-log-foods", "diet-log-history",
                           "diet-stats-pills", "diet-stats-empty",
                           "chart-diet-kcal", "diet-stats-fallback"):
            assert f'id="{element_id}"' in HTML, f"缺少 #{element_id}"
        assert 'name="diet-meal-count"' in HTML
        assert 'name="diet-log-meal-type"' in HTML


class TestHealthTextInputAndSingleNote:
    """健康状况＝手工自由文本（AI 判断）；备注与补充说明合并为一个字段。"""

    def test_health_is_free_text_input_not_chips(self):
        assert 'id="diet-health"' in HTML
        idx = HTML.index('id="diet-health"')
        assert 'maxlength="300"' in HTML[idx:idx + 120]
        assert 'id="diet-diseases"' not in HTML  # 疾病多选 chips 已移除
        assert 'value="糖尿病"' not in HTML

    def test_only_one_note_textarea_remains(self):
        assert 'id="diet-query"' not in HTML
        idx = HTML.index('id="diet-notes"')
        assert 'maxlength="2000"' in HTML[idx:idx + 80]

    def test_prefs_send_health_not_diseases(self):
        fn = _fn_body("getDietPrefs")
        assert "health:" in fn
        assert "diseases" not in fn

    def test_generate_payload_has_no_query_field(self):
        fn = _fn_body("generateDietPlan")
        assert "diet-query" not in fn
        assert "query:" not in fn


class TestDietBlockHygiene:
    def test_no_innerhtml_in_diet_block(self):
        assert "innerHTML" not in _diet_block()

    def test_no_native_prompt_in_diet_block(self):
        assert "prompt(" not in _diet_block()

    def test_no_sr_only_in_diet_block(self):
        assert ".sr-only" not in _diet_block()

    def test_no_internal_jargon_in_diet_block(self):
        block = _diet_block()
        assert "知识检索" not in block
        assert "writer_chunk" not in block

    def test_numbers_use_tabular_nums(self):
        assert ".diet-num { font-variant-numeric: tabular-nums; }" in HTML


class TestGenerationFlow:
    def test_generate_uses_payload_not_body(self):
        fn = _fn_body("generateDietPlan")
        assert "const payload = {" in fn
        assert "const body = {" not in fn

    def test_generate_has_timeout_and_busy(self):
        fn = _fn_body("generateDietPlan")
        assert "new AbortController()" in fn
        assert "60000" in fn
        assert "setBusy('btn-generate-diet'" in fn

    def test_validation_before_clearing_result(self):
        fn = _fn_body("generateDietPlan")
        assert fn.index("validateProfileFields()") < fn.index("replaceChildren()")

    def test_autosave_guard_precedes_fetch(self):
        fn = _fn_body("autoSaveDietPlan")
        guard = fn.index("delivery_status !== 'safe_delivered'")
        request = fn.index("fetch(")
        assert guard < request, "收录守卫必须在请求之前"

    def test_generate_autosave_only_after_safe_delivered(self):
        fn = _fn_body("generateDietPlan")
        assert "data.delivery_status === 'safe_delivered'" in fn
        assert fn.index("data.delivery_status === 'safe_delivered'") < fn.index("autoSaveDietPlan(data)")


class TestTerminalStates:
    def test_all_five_terminals_dispatched(self):
        fn = _fn_body("renderDietTerminal")
        for status in ("safe_delivered", "emergency_stopped",
                       "diet_crisis_stopped", "safety_blocked"):
            assert status in fn
        assert "renderDietGenerationFailed" in fn  # 兜底分支

    def test_crisis_stop_has_visible_next_step(self):
        fn = _fn_body("renderDietCrisisStop")
        assert "next_step" in fn
        assert "matched_signals" in fn
        assert "safety-block-panel" in fn

    def test_emergency_reuses_training_renderer(self):
        fn = _fn_body("renderDietTerminal")
        assert "renderEmergencyStop(container, data.emergency)" in fn

    def test_generation_failed_has_retry_exit(self):
        fn = _fn_body("renderDietGenerationFailed")
        assert "retry.onclick = generateDietPlan" in fn

    def test_safety_blocked_names_foods(self):
        fn = _fn_body("renderDietSafetyBlocked")
        assert "conflicts" in fn and "c?.food" in fn

    def test_safety_blocked_renders_checker_issues_when_no_conflicts(self):
        fn = _fn_body("renderDietSafetyBlocked")
        assert "details.issues" in fn, "审查 danger 无冲突食物时也要给出具体原因"


class TestResultRendering:
    def test_two_day_cards_and_meal_food_rows(self):
        assert "buildDietDayCard" in _fn_body("renderDietResult")
        meal = _fn_body("buildDietMealBlock")
        assert "diet-food-row" in meal
        assert "diet-food-grams" in meal and "diet-food-kcal" in meal

    def test_eat_this_meal_button_logs_to_api(self):
        meal = _fn_body("buildDietMealBlock")
        assert "吃了这餐" in meal
        fn = _fn_body("logMealFromPlan")
        assert "/api/diet-logs" in fn
        assert "todayLocalIso()" in fn

    def test_explanation_is_collapsible_four_sections(self):
        fn = _fn_body("renderDietExplanation")
        for key in ("goal", "energy", "foods", "safety"):
            assert f"'{key}'" in fn
        assert "explain-toggle" in fn and "aria-expanded" in fn

    def test_warnings_rendered_before_plan(self):
        fn = _fn_body("renderDietResult")
        assert fn.index("diet-warnings") < fn.index("buildDietDayCard")


class TestSavedDietPlans:
    def test_rename_is_in_place_not_prompt(self):
        fn = _fn_body("startRenameDietPlan")
        assert "saved-plan-rename-input" in fn
        assert "replaceWith" in fn
        assert "Escape" in fn and "Enter" in fn
        assert "maxLength = 40" in fn
        assert "prompt(" not in fn

    def test_crud_endpoints(self):
        assert "/api/diet-plans" in _fn_body("loadSavedDietPlans")
        assert "method: 'DELETE'" in _fn_body("deleteSavedDietPlan")
        assert "method: 'PATCH'" in _fn_body("renameSavedDietPlan")

    def test_open_switches_to_visible_panel_before_scroll(self):
        fn = _fn_body("openSavedDietPlan")
        assert fn.index("activateTab('tab-diet')") < fn.index("scrollIntoView")


class TestDietLogAndStats:
    def test_log_validation_ranges_match_backend(self):
        fn = _fn_body("readDietFoodRows")
        assert "2000" in fn  # 克数与单食物热量上限
        assert "source: 'custom'" in fn
        assert "12" in fn    # 单餐食物条数上限

    def test_stats_three_layer_degradation(self):
        fn = _fn_body("renderDietChart")
        assert "loadEcharts()" in fn
        assert "renderDietFallback(data)" in fn
        assert "drawDietKcalChart" in _diet_block()
        idx = HTML.index('id="chart-diet-kcal"')
        assert 'class="chart-box"' in HTML[idx:idx + 80]

    def test_chart_gaps_not_connected(self):
        fn = _fn_body("drawDietKcalChart")
        assert "connectNulls: false" in fn
        assert "is_partial" in fn

    def test_fallback_table_marks_missing_days(self):
        fn = _fn_body("renderDietFallback")
        assert "fallback-table" in fn
        assert "图表暂时加载不出来" in fn

    def test_init_binds_and_bootstraps(self):
        fn = _fn_body("initDietTab")
        for btn in ("btn-generate-diet", "btn-add-diet-food",
                    "btn-save-diet-log", "btn-refresh-diet-stats"):
            assert btn in fn
        block = _diet_block()
        assert "DOMContentLoaded" in block
        assert "initDietTab();" in block


class TestDietLogRedesign:
    """记录页重做：今日汇总置顶／单条可折叠／新增走底部抽屉不跳转。"""

    def test_today_summary_precedes_history(self):
        panel = HTML[HTML.index('id="panel-diet-log"'):HTML.index('id="panel-diet-stats"')]
        assert 'id="diet-today-kcal"' in panel
        assert 'id="btn-open-diet-drawer"' in panel
        assert panel.index('diet-today-kcal') < panel.index('diet-log-history')

    def test_records_collapsible_via_details(self):
        fn = _fn_body("loadDietLogHistory")
        assert "el('details', 'diet-rec'" in fn
        assert "el('summary'" in fn
        assert "rec.open = true" in fn  # 今日默认展开

    def test_new_record_is_bottom_drawer_not_navigation(self):
        assert 'id="diet-drawer" role="dialog" aria-modal="true"' in HTML
        assert 'class="diet-drawer-backdrop"' in HTML
        fn = _fn_body("openDietDrawer")
        assert "classList.add('is-open')" in fn
        assert "location" not in fn and "window.open" not in fn
        drawer = HTML[HTML.index('id="diet-drawer"'):HTML.index('id="panel-diet-stats"')]
        for kept_id in ("diet-log-date", "diet-log-foods", "btn-save-diet-log", "diet-log-notes"):
            assert f'id="{kept_id}"' in drawer, "抽屉须保留原表单 id"

    def test_save_closes_drawer_and_refreshes_summary(self):
        fn = _fn_body("saveDietLog")
        assert "closeDietDrawer()" in fn
        assert "loadDietLogHistory()" in fn
        assert "renderDietTodaySummary" in _fn_body("loadDietLogHistory")

    def test_drawer_closes_on_escape_and_backdrop(self):
        fn = _fn_body("initDietTab")
        assert "Escape" in fn
        assert "diet-drawer-backdrop" in fn


class TestVisualUpgradeStep3:
    """步骤三＋记录页三处修正：宏量全量／关闭按钮收敛／食物行输入统一美术。"""

    def test_today_summary_shows_all_four_macros(self):
        fn = _fn_body("renderDietTodaySummary")
        for key in ("kcal", "protein_g", "fat_g", "carbs_g"):
            assert key in fn, f"今日汇总缺少 {key}"
        panel = HTML[HTML.index('id="panel-diet-log"'):HTML.index('id="panel-diet-stats"')]
        assert 'id="diet-today-macros"' in panel
        assert panel.index('diet-today-macros') < panel.index('diet-log-history')

    def test_macro_tiles_built_without_innerhtml(self):
        fn = _fn_body("dietMacroTile")
        assert "innerHTML" not in fn
        assert "diet-today-macro-val" in fn and "diet-today-macro-label" in fn

    def test_drawer_close_is_compact_icon_not_stretched_account_btn(self):
        idx = HTML.index('id="btn-close-diet-drawer"')
        tag = HTML[HTML.rindex('<button', 0, idx):idx]
        assert 'diet-drawer-close' in tag
        assert 'account-btn' not in tag  # flex:1 会把它拉成半行宽大框
        block = HTML[HTML.index('.diet-drawer-close {'):HTML.index('.diet-drawer-close:hover')]
        assert 'flex: 0 0 auto' in block
        assert 'background: transparent' in block and 'border: none' in block

    def test_food_row_inputs_use_recessed_project_style(self):
        start = HTML.index('.diet-log-food-row input[type="text"],\n'
                           '.diet-log-food-row input[type="number"],\n'
                           '.diet-custom-fields input {')
        css = HTML[start:HTML.index('.diet-drawer-close {', start)]
        assert 'background: var(--bg-deep)' in css
        assert 'border-radius: var(--radius-sm)' in css
        assert ':focus' in css and 'box-shadow: 0 0 0 3px var(--forge-amber-dim)' in css

    def test_decorative_empty_state_pulse_removed(self):
        assert ('.empty-state .icon-ring::before,\n'
                '.empty-state .icon-ring::after { animation: none; }') in HTML

    def test_mobile_food_row_remove_stays_on_first_row(self):
        assert '.diet-log-food-row .log-entry-remove { grid-column: 3; grid-row: 1; }' in HTML

    def test_webfont_deferred_until_after_load(self):
        # 解析期不得发现字体 stylesheet：css2 ~121KB 会与 HTML 抢 3G 带宽
        assert 'rel="stylesheet" media="print"' not in HTML
        assert "window.addEventListener('load', function () {" in HTML
        assert "l.href = 'https://fonts.googleapis.com/css2" in HTML
        assert '<noscript><link href="https://fonts.googleapis.com/css2' in HTML

    def test_food_row_remove_is_bare_icon_not_boxed(self):
        block = HTML[HTML.index('.diet-log-food-row .log-entry-remove {'):
                     HTML.index('.diet-log-food-row .log-entry-remove:hover')]
        assert 'border: none' in block and 'background: transparent' in block

    def test_status_and_error_panels_use_hint_typography(self):
        # 原先无规则→浏览器默认 16px 黑字，与全页提示字号脱节
        block = HTML[HTML.index('.status-panel, .error-panel {'):
                     HTML.index('.status-panel { color')]
        assert 'font-size: 12.5px' in block
        assert 'font-family: var(--font-body)' in block
        assert '.status-panel { color: var(--status-safe); }' in HTML
        assert '.error-panel { color: var(--danger); }' in HTML