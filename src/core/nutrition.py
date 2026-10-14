"""每日热量与蛋白质目标 —— 确定性计算，不经过 LLM。

存在理由：这是计划里唯一一类「数字结论」。数字一旦交给模型生成就无从核对，
而热量目标恰恰是用户会拿去用的东西。所以这里全部用公开公式手算，每一步都能
复现；返回的 ``basis`` 就是给界面展示「这个数是怎么来的」的依据文本。

刻意零依赖：本模块不 import 项目内任何东西。它被计划定稿流程调用，而技能目录
（skills/muscle_building/scripts/tools.py）里那份同名工具委托到这里——两边
共用一份实现，不会各自漂移。
"""

from __future__ import annotations

# 活动系数：按「每周训练天数」分档，取自 Mifflin-St Jeor 常用的活动系数表。
# 从已有的 days_per_week 推导，而不是再加一个「活动量」字段让用户填——
# 用户已经告诉过我们他每周练几天，再问一遍既冗余又容易和训练天数打架。
_ACTIVITY_BY_DAYS = {0: 1.2, 1: 1.2, 2: 1.375, 3: 1.375,
                     4: 1.55, 5: 1.55, 6: 1.725, 7: 1.725}
_DEFAULT_ACTIVITY = 1.375

# 性别常数项。Mifflin-St Jeor 的男女之差就是这一项（+5 vs -161，相差 166 kcal）。
# 「不填」时取两性中值，误差约 ±83 kcal——小于活动系数分档之间的跨度，
# 所以在界面上如实标注为估算区间即可，不必逼用户回答。
_SEX_OFFSET = {"男": 5.0, "male": 5.0, "m": 5.0,
               "女": -161.0, "female": -161.0, "f": -161.0}
_SEX_MIDPOINT = (5.0 + -161.0) / 2

# 目标 → (热量偏移比例, 每公斤蛋白质克数, 中文说明)。中英文键都收：
# 档案里是「增肌/减脂」，技能配置里是 muscle_building/fat_loss，两边都会传进来。
_GOAL_TABLE = {
    "muscle_building": (0.15, 1.8, "增肌期热量盈余 ~15%"),
    "增肌": (0.15, 1.8, "增肌期热量盈余 ~15%"),
    "fat_loss": (-0.20, 2.0, "减脂期热量缺口 ~20%，高蛋白防肌肉流失"),
    "减脂": (-0.20, 2.0, "减脂期热量缺口 ~20%，高蛋白防肌肉流失"),
}


def _num(value) -> str:
    """把身体数据写进「依据」文案：有小数就带小数，没有就不带。

    原先固定用 ``:.0f``，用户填了 62.5kg 却在计划里看到「62kg」，会以为自己
    填的小数没生效——而它全程都参与了 BMR 与蛋白质计算，只有展示这一行被
    取整了。填的人和算的数对不上，是最容易让人怀疑「这功能是不是没做」的那种不一致。

    ``:g`` 让 62.5 显示成 62.5、62.0 显示成 62，两种情况都不别扭。
    """
    return f"{float(value):g}"


def calc_calorie_target(weight_kg: float, height_cm: float | None = None,
                        age: float | None = None, sex: str = "不填",
                        goal: str = "muscle_building",
                        days_per_week: int | None = None) -> dict:
    """估算每日热量与蛋白质目标。

    用 Mifflin-St Jeor 求基础代谢率（BMR），乘活动系数得每日消耗（TDEE），
    再按目标加减热量偏移。

    参数缺失时（没有身高或年龄）退回按体重粗估，并在 ``formula`` 里如实标明——
    退路保留不是为了精度，是为了让档案没填全的用户不至于连计划都生成不了。
    """
    weight_kg = float(weight_kg)
    delta_pct, protein_per_kg, goal_note = _GOAL_TABLE.get(
        (goal or "muscle_building").strip(), (0.0, 1.6, "维持期")
    )

    has_full_inputs = height_cm is not None and age is not None
    if has_full_inputs:
        height_cm = float(height_cm)
        age = float(age)
        offset = _SEX_OFFSET.get((sex or "").strip().lower())
        sex_known = offset is not None
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + (offset if sex_known else _SEX_MIDPOINT)
        formula = "Mifflin-St Jeor"
    else:
        bmr = weight_kg * 22
        sex_known = False
        formula = "体重系数粗估（缺身高或年龄）"

    factor = _DEFAULT_ACTIVITY
    if days_per_week is not None:
        factor = _ACTIVITY_BY_DAYS.get(int(days_per_week), _DEFAULT_ACTIVITY)

    tdee = bmr * factor
    daily = tdee * (1 + delta_pct)

    # 依据文本给界面用：让用户看得懂这个数字怎么来的，而不是接受一个凭空的数
    if has_full_inputs:
        sex_note = "性别已知" if sex_known else "性别未填，按两性中值估算（误差约 ±83 kcal）"
        basis = (f"{formula}｜{_num(weight_kg)}kg / {_num(height_cm)}cm / {_num(age)}岁｜"
                 f"活动系数 {factor}（每周 {days_per_week} 练）｜{sex_note}")
    else:
        basis = f"{formula}｜{_num(weight_kg)}kg｜{goal_note}"

    return {
        "bmr": round(bmr),
        "tdee": round(tdee),
        "daily_kcal": round(daily),
        "protein_g": round(weight_kg * protein_per_kg),
        "delta_pct": round(delta_pct * 100),
        "goal_note": goal_note,
        "formula": formula,
        "sex_known": sex_known,
        "basis": basis,
    }
