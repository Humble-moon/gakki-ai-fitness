"""计划可解释性模块的测试（纯函数，无 IO）。"""

import inspect

from src.core import plan_explanation
from src.core.plan_explanation import MAX_MUSCLES_SHOWN, build_explanation


def _plan(skill="muscle_building", subtasks=None):
    return {"skill": skill, "subtasks": subtasks if subtasks is not None else []}


def _retrieved(rows):
    return {"exercises": rows, "knowledge": []}


# ----------------------------------------------------------------------
# 容错：任何一段数据缺失都不能崩
# ----------------------------------------------------------------------

def test_handles_completely_empty_input():
    explain = build_explanation({}, {}, {})
    assert explain["skill"] == {}
    assert explain["retrieval"] == {}
    assert explain["safety"]["rounds"] == 1
    assert explain["safety"]["resolved"] == []


def test_handles_non_dict_inputs():
    """上游传了 None 或异常类型时不能抛错——解释块是展示层的，
    它挂掉不应该连累计划本身。"""
    explain = build_explanation(None, None, None)
    assert isinstance(explain, dict)
    assert explain["skill"] == {}
    assert explain["retrieval"] == {}


# ----------------------------------------------------------------------
# 规划依据
# ----------------------------------------------------------------------

def test_skill_id_is_translated_to_user_facing_label():
    """内部标识不能泄漏到界面上。"""
    explain = build_explanation(_plan(skill="muscle_building"), {}, {})
    assert explain["skill"]["label"] == "增肌计划"
    assert explain["skill"]["id"] == "muscle_building"


def test_unknown_skill_does_not_leak_internal_identifier():
    """新增技能没来得及配标签时，宁可显示通用名也不能把英文标识码丢给用户。"""
    explain = build_explanation(_plan(skill="brand_new_skill"), {}, {})
    assert explain["skill"]["label"] == "通用模板"
    assert "brand_new_skill" not in explain["skill"]["label"]


def test_missing_skill_falls_back_to_generic_label_when_subtasks_exist():
    explain = build_explanation(_plan(skill="", subtasks=["胸部训练"]), {}, {})
    assert explain["skill"]["label"] == "通用模板"


def test_empty_plan_meta_hides_the_whole_section():
    """没有模板也没有子任务时整段隐藏，不编造标签。"""
    assert build_explanation({"skill": "", "subtasks": []}, {}, {})["skill"] == {}


def test_subtasks_are_preserved():
    explain = build_explanation(
        _plan(subtasks=["胸部训练", "背部训练"]), {}, {})
    assert explain["skill"]["subtasks"] == ["胸部训练", "背部训练"]


def test_blank_subtasks_are_dropped():
    explain = build_explanation(
        _plan(subtasks=["胸部训练", "", "   ", None]), {}, {})
    assert explain["skill"]["subtasks"] == ["胸部训练"]


# ----------------------------------------------------------------------
# 动作筛选
# ----------------------------------------------------------------------

def test_splits_semantic_and_structured_retrieval_counts():
    rows = [
        {"name": "杠铃卧推"},
        {"name": "哑铃飞鸟"},
        {"name": "绳索下压", "source": "mcp"},
    ]
    explain = build_explanation({}, _retrieved(rows), {})
    assert explain["retrieval"]["total"] == 3
    assert explain["retrieval"]["semantic"] == 2
    assert explain["retrieval"]["structured"] == 1


def test_collects_and_deduplicates_muscles():
    rows = [
        {"name": "a", "muscles": ["胸大肌", "三头肌"]},
        {"name": "b", "muscles": ["胸大肌", "三角肌"]},
    ]
    explain = build_explanation({}, _retrieved(rows), {})
    assert explain["retrieval"]["muscles"] == ["胸大肌", "三头肌", "三角肌"]
    assert explain["retrieval"]["muscle_total"] == 3


def test_accepts_target_muscles_alias():
    """动作库用 target_muscles，检索结果用 muscles，两种都要认。"""
    rows = [{"name": "a", "target_muscles": ["背阔肌"]}]
    explain = build_explanation({}, _retrieved(rows), {})
    assert explain["retrieval"]["muscles"] == ["背阔肌"]


def test_accepts_muscle_as_plain_string():
    rows = [{"name": "a", "muscles": "胸大肌"}]
    explain = build_explanation({}, _retrieved(rows), {})
    assert explain["retrieval"]["muscles"] == ["胸大肌"]


def test_muscle_list_is_capped():
    rows = [{"name": f"ex{i}", "muscles": [f"肌群{i}"]} for i in range(30)]
    explain = build_explanation({}, _retrieved(rows), {})
    assert len(explain["retrieval"]["muscles"]) == MAX_MUSCLES_SHOWN
    assert explain["retrieval"]["muscle_total"] == 30


def test_empty_retrieval_yields_empty_section():
    assert build_explanation({}, _retrieved([]), {})["retrieval"] == {}


# ----------------------------------------------------------------------
# 安全检查
# ----------------------------------------------------------------------

def test_rounds_counts_the_initial_check_too():
    """rewrite_count=0 表示一次通过，展示成「检查 1 轮」而不是 0 轮。"""
    assert build_explanation({}, {}, {"rewrite_count": 0})["safety"]["rounds"] == 1
    assert build_explanation({}, {}, {"rewrite_count": 2})["safety"]["rounds"] == 3


def test_resolved_issues_are_surfaced():
    """「修掉了什么」是最能建立信任的一条。"""
    result = {"resolved_issues": [{"issue": "深蹲角度不适合当前伤病"}]}
    safety = build_explanation({}, {}, result)["safety"]
    assert safety["resolved"] == ["深蹲角度不适合当前伤病"]


def test_issue_accepts_both_string_and_dict_forms():
    result = {
        "active_issues": ["纯字符串问题", {"issue": "字典形式问题"}],
        "resolved_issues": [{"description": "另一种键名"}],
    }
    safety = build_explanation({}, {}, result)["safety"]
    assert safety["active"] == ["纯字符串问题", "字典形式问题"]
    assert safety["resolved"] == ["另一种键名"]


def test_malformed_issue_entries_are_skipped():
    result = {"active_issues": [None, 123, {}, {"issue": ""}, "有效问题"]}
    assert build_explanation({}, {}, result)["safety"]["active"] == ["有效问题"]


def test_confidence_is_validated():
    assert build_explanation({}, {}, {"confidence": 0.92})["safety"]["confidence"] == 0.92


def test_out_of_range_confidence_is_hidden():
    """越界或非数值的置信度展示出来只会误导用户，宁可隐藏。"""
    for bad in (1.5, -0.2, "high", None, True):
        assert build_explanation({}, {}, {"confidence": bad})["safety"]["confidence"] is None


def test_degraded_provider_is_flagged():
    assert build_explanation({}, {}, {"_degraded": True})["safety"]["degraded"] is True
    assert build_explanation({}, {}, {"provider_degraded": True})["safety"]["degraded"] is True
    assert build_explanation({}, {}, {})["safety"]["degraded"] is False


# ----------------------------------------------------------------------
# 守卫：这是展示层模块，不得影响生成
# ----------------------------------------------------------------------

def test_explanation_is_pure_and_reads_only_its_inputs():
    """不得持有任何外部依赖——它不是流水线的一环，只是把已有事实整理出来。"""
    source = inspect.getsource(plan_explanation)
    for forbidden in ("import requests", "PGClient", "RedisClient", "OpenAI"):
        assert forbidden not in source


def test_explanation_never_mutates_its_inputs():
    plan = _plan(subtasks=["胸部训练"])
    retrieved = _retrieved([{"name": "a", "muscles": ["胸大肌"]}])
    result = {"confidence": 0.9, "rewrite_count": 1, "active_issues": ["x"]}
    plan_snapshot = repr(plan)
    retrieved_snapshot = repr(retrieved)
    result_snapshot = repr(result)

    build_explanation(plan, retrieved, result)

    assert repr(plan) == plan_snapshot
    assert repr(retrieved) == retrieved_snapshot
    assert repr(result) == result_snapshot
