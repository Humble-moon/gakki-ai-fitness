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
        # 饮食侧扩展（ additive ）：脂肪/碳水按「蛋白 → 脂肪占剩余 25% → 碳水补齐」
        # 分配。旧键值一律不变——训练计划的 nutrition 字段依赖上面的原有行为。
        **split_macros(daily, weight_kg * protein_per_kg),
    }


# ----------------------------------------------------------------------
# 饮食侧扩展：宏量分配 / 安全护栏 / 训练日拆分 / 餐次分配
#
# 全部仍是零依赖纯函数。与上面的区别：这些只服务饮食管线，训练侧不调用，
# 因此护栏（热量下限、未成年/低体重禁缺口）在这里生效而不改变训练侧既有数字。
# ----------------------------------------------------------------------

# 脂肪占「扣除蛋白后剩余热量」的比例。知识文档 100-yinshi-jihua.md 给的是
# 20-30% 区间，取中值 25%；区间两端留给上层按疾病护栏覆盖（如肾病低蛋白）。
FAT_SHARE_OF_REMAINDER = 0.25

# 人群启发式绝对下限（kcal）。不是临床处方：basis 与 warnings 会如实标注
# 「估算护栏」。真正的临床下限因人而异，所以同时保留 BMR 下限取较大者。
ENERGY_FLOOR_KCAL = 1200
MINOR_AGE = 18          # 未成年禁热量缺口（生长发育期）
LOW_BMI = 18.5          # 低体重禁热量缺口
VERY_LOW_BMI = 16.0     # 强提示专业评估（但不判危机——可能是录入错误）

# 训练日/休息日波动幅度：周加权守恒，多出的热量主要落在碳水（见 split_macros）。
DAY_SWING = 0.15

# 餐次热量比例是**设计常数**（知识文档只给了「每餐 30-40g 蛋白、一天 3-5 餐」，
# 没给热量比例）。basis/explain 必须如实标注为设计口径，不得伪称知识出处。
MEAL_SLOTS = {
    3: (("早餐", 0.30), ("午餐", 0.40), ("晚餐", 0.30)),
    4: (("早餐", 0.28), ("午餐", 0.37), ("晚餐", 0.30), ("加餐", 0.05)),
}


def split_macros(daily_kcal: float, protein_g: float) -> dict:
    """蛋白 → 脂肪（剩余的 25%）→ 碳水补齐。

    剩余热量为 0（蛋白已占满预算）时脂肪/碳水归 0 而不是负数——负碳水会渲染成
    「每天吃 -30g 碳水」这种荒唐建议；是否该触发护栏由上层判断并告知用户。
    """
    protein_kcal = float(protein_g) * 4.0
    remaining = max(0.0, float(daily_kcal) - protein_kcal)
    fat_kcal = remaining * FAT_SHARE_OF_REMAINDER
    carbs_kcal = remaining - fat_kcal
    return {
        "fat_g": round(fat_kcal / 9.0),
        "carbs_g": round(carbs_kcal / 4.0),
    }


def apply_energy_guardrails(targets: dict, *, weight_kg: float,
                            age: float | None = None,
                            height_cm: float | None = None) -> dict:
    """饮食侧安全护栏：禁缺口人群 + 热量下限 + 极低 BMI 提示。

    返回 ``{"targets": 新字典, "warnings": [...], "flags": {...}}``，**不修改入参**。
    顺序：先禁缺口（未成年/低 BMI 把减脂转维持），再钳下限（max(BMR, 1200)）。
    钳制发生时重算 delta_pct 并在 basis 写明，不让用户以为缺口还在。
    """
    out = dict(targets)
    warnings: list[str] = []
    flags = {"deficit_blocked": False, "floor_applied": False, "very_low_bmi": False}

    bmi = None
    if height_cm:
        bmi = float(weight_kg) / ((float(height_cm) / 100.0) ** 2)

    if out.get("delta_pct", 0) < 0:
        if age is not None and float(age) < MINOR_AGE:
            flags["deficit_blocked"] = True
            warnings.append("青少年处于生长发育期，不建议热量缺口；已改为维持热量，"
                            "请咨询医生或营养师后再调整饮食。")
        elif bmi is not None and bmi < LOW_BMI:
            flags["deficit_blocked"] = True
            warnings.append("当前体重偏低（BMI < 18.5），不建议热量缺口；已改为维持热量，"
                            "建议先咨询医生或营养师。")
    if flags["deficit_blocked"]:
        out["daily_kcal"] = out["tdee"]
        out["delta_pct"] = 0
        out["goal_note"] = "维持期（护栏：不建议热量缺口）"
        out["basis"] = out.get("basis", "") + "｜护栏取消热量缺口"

    floor = max(float(out.get("bmr") or 0), ENERGY_FLOOR_KCAL)
    if float(out["daily_kcal"]) < floor:
        flags["floor_applied"] = True
        out["daily_kcal"] = round(floor)
        tdee = float(out.get("tdee") or 0)
        out["delta_pct"] = round((floor / tdee - 1) * 100) if tdee > 0 else 0
        out["basis"] = out.get("basis", "") + f"｜已按安全下限钳制至 {round(floor)} kcal（估算护栏）"
        warnings.append(f"热量目标已按安全下限钳制至 {round(floor)} kcal；"
                        "这是估算护栏而非临床处方，长期低热量饮食请寻求专业指导。")

    if bmi is not None and bmi < VERY_LOW_BMI:
        flags["very_low_bmi"] = True
        warnings.append("BMI 低于 16，强烈建议先做专业营养评估；本计划仅供参考，"
                        "不能替代临床营养诊疗。")

    protein_g = float(out.get("protein_g") or 0)
    if protein_g * 4.0 >= float(out["daily_kcal"]):
        warnings.append("蛋白质目标已占满热量预算，脂肪与碳水记为 0；"
                        "请检查档案填写是否正确，或咨询营养师调整蛋白目标。")

    out.update(split_macros(out["daily_kcal"], protein_g))
    return {"targets": out, "warnings": warnings, "flags": flags}


def split_day_targets(daily_kcal: float, days_per_week: int,
                      swing: float = DAY_SWING) -> dict:
    """训练日/休息日两套热量，周加权守恒。

    训练日 = daily × (1 + k×(7−d)/7)，休息日 = daily × (1 − k×d/7)，
    于是 (d×训练 + (7−d)×休息)/7 恰好 = daily——一周总摄入不因日拆分而漂移。
    多出的热量经 split_macros 主要落在碳水（脂肪只拿剩余的 25%）。
    d=7 时休息日仍生成（模板语义），公式自然退化。
    """
    d = max(0, min(7, int(days_per_week or 0)))
    daily = float(daily_kcal)
    train = daily * (1 + swing * (7 - d) / 7.0)
    rest = daily * (1 - swing * d / 7.0)
    return {
        "training_day_kcal": round(train),
        "rest_day_kcal": round(rest),
        "weekly_mean_kcal": round((d * train + (7 - d) * rest) / 7.0),
        "days_per_week": d,
        "swing_pct": round(swing * 100),
    }


def allocate_meals(day_kcal: float, protein_g: float, meal_count: int = 4) -> list[dict]:
    """把一日目标拆到各餐：热量按比例（最大余数法保证合计精确），蛋白均分。

    蛋白**按日总量均分**而不是硬套「每餐 30-40g」：50kg 减脂者日蛋白 100g 分 4 餐
    只有 25g/餐，硬套参考带会超总量。30-40g 只是增肌大体重者的参考区间。
    """
    slots = MEAL_SLOTS.get(int(meal_count))
    if slots is None:
        raise ValueError("meal_count 仅支持 3 或 4")
    day = float(day_kcal)
    raw = [day * share for _, share in slots]
    base = [int(x) for x in raw]
    leftover = round(day) - sum(base)
    order = sorted(range(len(slots)), key=lambda i: (-(raw[i] - base[i]), i))
    for i in order[:max(0, leftover)]:
        base[i] += 1
    protein_each = float(protein_g) / len(slots)
    return [
        {"meal": name, "kcal_target": base[i], "protein_target_g": round(protein_each, 1)}
        for i, (name, _) in enumerate(slots)
    ]
