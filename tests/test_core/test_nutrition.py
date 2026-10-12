"""每日热量与蛋白质目标的确定性计算。

这是计划里唯一的「数字结论」，而且用户会真的拿它去用。测试的重点因此不是
"能跑"，而是**数值可手算复现**：每个断言都把公式写出来，改公式必须同时改
这里的推导，不允许出现"没人知道这个数字怎么来的"。
"""

import pytest

from src.core.nutrition import calc_calorie_target
from src.core.plan_finalization import attach_nutrition, normalize_plan


# ----------------------------------------------------------------------
# 公式：手算复现
# ----------------------------------------------------------------------

def test_mifflin_st_jeor_male_bulk_matches_hand_calculation():
    """80kg / 180cm / 30岁 / 男 / 增肌 / 每周 4 练。

    BMR  = 10*80 + 6.25*180 - 5*30 + 5 = 800 + 1125 - 150 + 5 = 1780
    TDEE = 1780 * 1.55（每周 4 练）      = 2759
    目标 = 2759 * 1.15（增肌盈余 15%）   = 3172.85 → 3173
    """
    r = calc_calorie_target(80, 180, 30, "男", "增肌", 4)

    assert r["bmr"] == 1780
    assert r["tdee"] == 2759
    assert r["daily_kcal"] == 3173
    assert r["protein_g"] == round(80 * 1.8)


def test_mifflin_st_jeor_female_cut_matches_hand_calculation():
    """60kg / 165cm / 28岁 / 女 / 减脂 / 每周 3 练。

    BMR  = 600 + 1031.25 - 140 - 161 = 1330.25 → 1330
    TDEE = 1330.25 * 1.375             = 1829.09 → 1829
    目标 = 1829.09 * 0.80（减脂缺口 20%）= 1463.27 → 1463
    """
    r = calc_calorie_target(60, 165, 28, "女", "减脂", 3)

    assert r["bmr"] == 1330
    assert r["tdee"] == 1829
    assert r["daily_kcal"] == 1463
    assert r["protein_g"] == round(60 * 2.0)


def test_sex_constants_differ_by_166_kcal():
    """Mifflin-St Jeor 的男女常数项差 166（+5 vs -161）。

    这条是「为什么要问性别」的全部理由，把它钉死：
    如果哪天有人改了常数，差异会立刻在这里暴露。
    """
    male = calc_calorie_target(80, 180, 30, "男", "增肌", 4)
    female = calc_calorie_target(80, 180, 30, "女", "增肌", 4)

    assert male["bmr"] - female["bmr"] == 166


def test_unfilled_sex_falls_back_to_midpoint_and_says_so():
    """「不填」是合法输入，不是错误。

    它必须①给出可用的估算值，②在依据里如实标注这是两性中值——不能悄悄
    套一个性别，也不能因为缺这一项就拒绝出计划。
    """
    mid = calc_calorie_target(80, 180, 30, "不填", "增肌", 4)
    male = calc_calorie_target(80, 180, 30, "男", "增肌", 4)

    assert mid["sex_known"] is False
    assert male["sex_known"] is True
    # 中值恰好落在男女之间：不在任何一侧之外
    assert 0 < male["bmr"] - mid["bmr"] <= 83
    assert "性别未填" in mid["basis"]


# ----------------------------------------------------------------------
# 活动系数分档
# ----------------------------------------------------------------------

@pytest.mark.parametrize("days,expected_factor_row", [
    (1, 1.2), (2, 1.375), (4, 1.55), (6, 1.725),
])
def test_activity_factor_increases_with_training_days(days, expected_factor_row):
    """活动系数按每周天数分档，天数越多系数越大（单调不减）。"""
    r = calc_calorie_target(80, 180, 30, "男", "增肌", days)

    assert r["tdee"] == round(r["bmr"] * expected_factor_row)


def test_activity_factor_is_monotonic():
    """分档必须单调：天数增加不可能导致热量目标下降。"""
    values = [calc_calorie_target(80, 180, 30, "男", "增肌", d)["tdee"]
              for d in range(0, 8)]

    assert values == sorted(values)


# ----------------------------------------------------------------------
# 退化路径与边界
# ----------------------------------------------------------------------

def test_missing_height_or_age_falls_back_and_labels_the_formula():
    """缺身高或年龄时退回体重粗估，并如实标明用了哪个公式。

    退路存在的意义是让档案没填全的用户仍能拿到计划，但它必须在依据里
    自曝身份——否则用户会把粗估当成标准公式算出来的结果。
    """
    r = calc_calorie_target(80, None, None, "男", "增肌", 4)

    assert r["formula"].startswith("体重系数粗估")
    assert r["sex_known"] is False
    assert r["bmr"] == round(80 * 22)
    assert "Mifflin" not in r["basis"]


def test_basis_always_names_the_formula_so_the_number_is_checkable():
    """依据文本必须出现公式名——这是"这个数怎么来的"唯一的答案。"""
    full = calc_calorie_target(80, 180, 30, "男", "增肌", 4)
    fallback = calc_calorie_target(80, None, None, "男", "增肌", 4)

    assert "Mifflin-St Jeor" in full["basis"]
    assert "1.55" in full["basis"]      # 活动系数也要露出来
    assert fallback["basis"]


@pytest.mark.parametrize("weight,height,age,sex,goal,days", [
    (30, 100, 10, "女", "减脂", 1),
    (200, 250, 100, "男", "增肌", 7),
    (45, 150, 18, "女", "减脂", 0),
])
def test_extreme_but_valid_profiles_still_produce_sane_numbers(
        weight, height, age, sex, goal, days):
    """边界值不能算出负数或 0——那会直接渲染成"每天吃 0 卡"这种荒唐建议。"""
    r = calc_calorie_target(weight, height, age, sex, goal, days)

    assert r["bmr"] > 0
    assert r["tdee"] >= r["bmr"]
    assert r["daily_kcal"] > 0
    assert r["protein_g"] > 0


# ----------------------------------------------------------------------
# 挂载：这是它真正进入计划的地方
# ----------------------------------------------------------------------

def test_normalize_plan_attaches_nutrition_from_profile():
    profile = {"weight": 80, "height": 180, "age": 30, "sex": "男",
               "goal": "增肌", "days_per_week": 4}

    result = normalize_plan({"days": []}, profile=profile)

    assert result["nutrition"]["daily_kcal"] == 3173


def test_nutrition_is_absent_rather_than_faked_without_a_weight():
    """没有体重就不挂。

    宁可整块不显示，也不给一个看着很确定的假数字——这是本项目对
    "编造"的一贯态度。
    """
    assert normalize_plan({"days": []}, profile={}).get("nutrition") is None
    assert normalize_plan({"days": []}, profile=None).get("nutrition") is None


def test_nutrition_is_idempotent_across_rewrites():
    """重写循环会多次归一化，挂载必须幂等——否则每轮都会变一个数。"""
    profile = {"weight": 80, "height": 180, "age": 30, "sex": "男",
               "goal": "增肌", "days_per_week": 4}

    first = normalize_plan({"days": []}, profile=profile)["nutrition"]
    second = normalize_plan({"days": [], "nutrition": first}, profile=profile)["nutrition"]

    assert first == second


def test_attach_nutrition_does_not_blow_up_on_garbage_profile():
    result = {}
    attach_nutrition(result, {})
    assert result == {}


# ----------------------------------------------------------------------
# 单一实现：技能工具不得自己复制一份公式
# ----------------------------------------------------------------------

def test_skill_tool_delegates_to_the_single_implementation():
    """skills/ 里那份同名工具必须与 src/core/nutrition.py 结果一致。

    两份公式各算各的，早晚会漂移——而用户看到的是技能那份还是定稿那份，
    取决于哪条链路先被调用，这种不一致极难排查。
    """
    import importlib.util
    from pathlib import Path

    tool_path = (Path(__file__).resolve().parents[2]
                 / "skills" / "muscle_building" / "scripts" / "tools.py")
    spec = importlib.util.spec_from_file_location("_skill_tools_under_test", tool_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = (80, 180, 30, "男", "增肌", 4)
    assert module.calc_calorie_target(*args) == calc_calorie_target(*args)


# ----------------------------------------------------------------------
# 缓存交付：热量必须按当前档案重算
#
# 缓存指纹（make_user_key）只由身高/体重/目标/伤病组成，而热量还取决于
# 年龄、性别、每周天数。这几项改动不会让缓存失效，于是命中缓存的用户会
# 拿到上一份档案算出来的热量——缓存的是计划，热量是当前档案的派生视图。
# ----------------------------------------------------------------------

def _cacheable_plan(**nutrition):
    """构造一份能通过 safe_cached_result 全部闸门的计划。"""
    return {
        "plan_id": "p1", "user_id": 1, "goal": "增肌", "weeks": 4,
        "sessions_per_week": 4,
        "_persistence_allowed": True, "requires_review": False, "warnings": [],
        "days": [{"day": 1, "focus": "全身", "exercises": []}],
        "nutrition": dict(nutrition),
    }


def test_cached_plan_gets_nutrition_recomputed_from_the_current_profile():
    from src.core.plan_finalization import safe_cached_result

    stale = {"daily_kcal": 1, "protein_g": 1, "basis": "上一份档案算的"}
    profile = {"weight": 80, "height": 180, "age": 30, "sex": "男",
               "goal": "增肌", "days_per_week": 4}

    result = safe_cached_result(_cacheable_plan(**stale), expected_goal="增肌",
                                profile=profile)

    assert result["nutrition"]["daily_kcal"] == 3173, \
        "命中缓存时没有按当前档案重算热量，用户会看到旧档案的数字"


def test_cached_plan_without_profile_keeps_working():
    """不传 profile 的既有调用方（老脚本、测试）不能被打破。"""
    from src.core.plan_finalization import safe_cached_result

    result = safe_cached_result(_cacheable_plan(daily_kcal=1), expected_goal="增肌")

    assert result is not None
