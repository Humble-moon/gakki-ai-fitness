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


# ----------------------------------------------------------------------
# 小数体重 / 身高：填了什么就显示什么
# ----------------------------------------------------------------------

def test_basis_shows_decimal_weight_and_height():
    """依据文案要照实显示小数。

    原先固定用 ``:.0f``：用户填 62.5kg，计划里却写「62kg」。数值全程参与
    BMR 与蛋白质计算（第 61/87 行用的是原始小数），只有这一行展示被取整——
    填的人和显示的数对不上，用户会以为自己填的小数没生效。
    """
    result = calc_calorie_target(62.5, height_cm=172.5, age=30,
                                 sex="男", goal="增肌", days_per_week=4)
    assert "62.5kg" in result["basis"]
    assert "172.5cm" in result["basis"]


def test_basis_omits_trailing_zero_for_integers():
    """整数不带小数点——「80.0kg」比「80kg」难读，没必要为了支持小数而变丑。"""
    result = calc_calorie_target(80, height_cm=180, age=30,
                                 sex="男", goal="增肌", days_per_week=4)
    assert "80kg" in result["basis"]
    assert "180cm" in result["basis"]
    assert "80.0kg" not in result["basis"]


def test_decimal_weight_actually_changes_the_calculation():
    """小数必须真的进入计算，而不只是显示层好看。

    62.5 与 62 的 BMR 相差 5 kcal——这正是「支持小数」的意义所在：
    如果只改显示、计算仍按取整值走，那这个功能就是假的。
    """
    a = calc_calorie_target(62.5, height_cm=172.5, age=30, sex="男",
                            goal="增肌", days_per_week=4)
    b = calc_calorie_target(62, height_cm=172, age=30, sex="男",
                            goal="增肌", days_per_week=4)
    assert a["bmr"] != b["bmr"], "小数没有影响 BMR，说明计算被取整了"


# ----------------------------------------------------------------------
# 饮食侧扩展：宏量 / 护栏 / 日拆分 / 餐分配
# ----------------------------------------------------------------------

from src.core.nutrition import (  # noqa: E402
    allocate_meals,
    apply_energy_guardrails,
    split_day_targets,
    split_macros,
)


def test_calc_returns_macros_without_changing_legacy_keys():
    """增返 fat_g/carbs_g 是 additive：训练侧依赖的旧键值一个都不能变。"""
    r = calc_calorie_target(80, 180, 30, "男", "增肌", 4)
    assert r["daily_kcal"] == 3173 and r["protein_g"] == 144
    # 蛋白 576 kcal → 剩余 2597 → 脂肪 25% = 649.25 → 72g；碳水 1947.75 → 487g
    assert r["fat_g"] == round((3172.85 - 144 * 4) * 0.25 / 9)
    assert r["carbs_g"] == round((3172.85 - 144 * 4) * 0.75 / 4)


@pytest.mark.parametrize("daily,protein", [(3173, 144), (1463, 120), (2000, 100), (500, 144)])
def test_split_macros_energy_conservation(daily, protein):
    """4P + 9F + 4C 必须约等于日目标（三个取整最多偏差 8 kcal）。

    蛋白已超预算时守恒不适用——那时脂肪/碳水归 0，超出部分由上层护栏告知用户。
    """
    m = split_macros(daily, protein)
    if 4 * protein > daily:
        assert m == {"fat_g": 0, "carbs_g": 0}
        return
    reconstructed = 4 * protein + 9 * m["fat_g"] + 4 * m["carbs_g"]
    assert abs(daily - reconstructed) <= 8
    assert m["fat_g"] >= 0 and m["carbs_g"] >= 0


def test_split_macros_never_negative_when_protein_fills_budget():
    m = split_macros(500, 144)  # 蛋白 576 kcal > 500
    assert m == {"fat_g": 0, "carbs_g": 0}


def test_guardrails_block_deficit_for_minors():
    targets = calc_calorie_target(50, 160, 15, "女", "减脂", 3)
    guarded = apply_energy_guardrails(targets, weight_kg=50, age=15, height_cm=160)
    assert guarded["flags"]["deficit_blocked"] is True
    assert guarded["targets"]["daily_kcal"] == guarded["targets"]["tdee"]
    assert guarded["targets"]["delta_pct"] == 0
    assert any("青少年" in w for w in guarded["warnings"])


def test_guardrails_block_deficit_for_low_bmi():
    targets = calc_calorie_target(45, 170, 25, "女", "减脂", 3)  # BMI 15.6
    guarded = apply_energy_guardrails(targets, weight_kg=45, age=25, height_cm=170)
    assert guarded["flags"]["deficit_blocked"] is True
    assert guarded["flags"]["very_low_bmi"] is True
    assert any("BMI 低于 16" in w for w in guarded["warnings"])


def test_guardrails_floor_clamps_and_relabels_delta():
    # 50kg/160cm/30岁/女/减脂/1练：BMR 1189，TDEE 1426.8，目标 1141.4 < 1200
    targets = calc_calorie_target(50, 160, 30, "女", "减脂", 1)
    guarded = apply_energy_guardrails(targets, weight_kg=50, age=30, height_cm=160)
    assert guarded["flags"]["floor_applied"] is True
    assert guarded["flags"]["deficit_blocked"] is False  # BMI 19.5，不属禁缺口人群
    assert guarded["targets"]["daily_kcal"] == 1200
    assert guarded["targets"]["delta_pct"] == round((1200 / targets["tdee"] - 1) * 100)
    assert "安全下限" in guarded["targets"]["basis"]


def test_guardrails_leave_healthy_bulk_untouched_and_do_not_mutate_input():
    targets = calc_calorie_target(80, 180, 30, "男", "增肌", 4)
    snapshot = dict(targets)
    guarded = apply_energy_guardrails(targets, weight_kg=80, age=30, height_cm=180)
    assert targets == snapshot, "护栏不得修改入参"
    assert guarded["warnings"] == []
    assert guarded["flags"] == {"deficit_blocked": False, "floor_applied": False,
                                "very_low_bmi": False}
    assert guarded["targets"]["daily_kcal"] == 3173


def test_guardrails_warn_when_protein_fills_budget():
    # 小体重档案（BMR 1189 → floor 1200），人为把蛋白推到 400g（1600 kcal > 1200）
    targets = calc_calorie_target(50, 160, 30, "女", "维持", 1)
    targets = dict(targets, daily_kcal=1200, protein_g=400, delta_pct=0)
    guarded = apply_energy_guardrails(targets, weight_kg=50, age=30, height_cm=160)
    assert any("占满热量预算" in w for w in guarded["warnings"])
    assert guarded["targets"]["carbs_g"] == 0


@pytest.mark.parametrize("days", range(0, 8))
def test_split_day_targets_weekly_weighted_conservation(days):
    daily = 2500
    split = split_day_targets(daily, days)
    train, rest = split["training_day_kcal"], split["rest_day_kcal"]
    # 取整后周加权均值与日目标偏差 ≤ 1 kcal；未取整前严格守恒
    assert abs((days * train + (7 - days) * rest) / 7 - daily) <= 1
    assert split["weekly_mean_kcal"] == daily
    if 0 < days < 7:
        assert train > daily > rest
    if days == 7:
        assert train == daily and rest < daily  # 休息日模板仍生成


@pytest.mark.parametrize("day", [1463, 2000, 2500, 3173])
@pytest.mark.parametrize("count", [3, 4])
def test_allocate_meals_kcal_sum_is_exact(day, count):
    meals = allocate_meals(day, 144, meal_count=count)
    assert sum(m["kcal_target"] for m in meals) == day
    assert [m["meal"] for m in meals] == (
        ["早餐", "午餐", "晚餐"] if count == 3 else ["早餐", "午餐", "晚餐", "加餐"])
    for m in meals:
        assert abs(m["protein_target_g"] - 144 / count) <= 0.1


def test_allocate_meals_protein_is_evenly_split_not_hardcoded_band():
    """50kg 减脂者日蛋白 100g 分 4 餐 = 25g/餐，不能硬套 30-40g 参考带。"""
    meals = allocate_meals(1500, 100, meal_count=4)
    assert all(m["protein_target_g"] == 25.0 for m in meals)


def test_allocate_meals_rejects_unsupported_count():
    with pytest.raises(ValueError):
        allocate_meals(2000, 144, meal_count=5)


def test_nutrition_module_stays_zero_dependency():
    """模块 docstring 的契约：不 import 项目内任何东西（技能工具委托到此）。"""
    from pathlib import Path
    source = (Path(__file__).resolve().parents[2] / "src" / "core" / "nutrition.py").read_text(encoding="utf-8")
    assert "from src." not in source and "import src" not in source
