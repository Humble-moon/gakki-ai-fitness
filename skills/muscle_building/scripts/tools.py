"""增肌 Skill 工具函数 — 由 SkillLoader 加载到 Skill.tools 字典。"""


def calc_1rm(weight: float, reps: int) -> float:
    """Epley 公式估算单次最大重量。"""
    if reps <= 0 or weight <= 0:
        return 0.0
    return round(weight * (1 + reps / 30), 1)


def calc_bmi(height_cm: float, weight_kg: float) -> dict:
    """计算 BMI 并返回分类。"""
    h = height_cm / 100
    bmi = round(weight_kg / (h * h), 1)
    if bmi < 18.5:
        cat = "偏瘦"
    elif bmi < 24:
        cat = "正常"
    elif bmi < 28:
        cat = "偏重"
    else:
        cat = "肥胖"
    return {"bmi": bmi, "category": cat}


def calc_calorie_target(weight_kg: float, height_cm: float | None = None,
                        age: float | None = None, sex: str = "不填",
                        goal: str = "muscle_building",
                        days_per_week: int | None = None) -> dict:
    """估算每日热量与蛋白质目标（Mifflin-St Jeor + 活动系数）。

    实现在 ``src/core/nutrition.py``，这里只做委托。**不要在本地复制一份公式**：
    同一个数字在技能工具和计划定稿流程里各算各的，两边迟早会漂移，而热量目标
    恰恰是用户会拿去用的东西。

    惰性 import 是刻意的——技能目录不是常规 Python 包，由 SkillLoader 按文件
    路径加载；模块级 import 一旦取不到 src 就会让整个技能加载失败。放进函数体
    后，最坏情况也只是调用这个工具时报错，不影响其它技能。
    """
    from src.core.nutrition import calc_calorie_target as _impl
    return _impl(weight_kg, height_cm=height_cm, age=age, sex=sex,
                 goal=goal, days_per_week=days_per_week)


def check_volume(exercises_per_week: int, sets_per_exercise: int, goal: str = "muscle_building") -> dict:
    """检查每周训练量是否在合理范围。"""
    total_sets = exercises_per_week * sets_per_exercise
    if goal == "muscle_building":
        ok = 10 <= total_sets <= 25
        advise = "增肌每肌群每周 10-25 组为最佳区间（Schoenfeld 2016）"
    elif goal == "fat_loss":
        ok = 12 <= total_sets <= 30
        advise = "减脂期可稍高容量，但不超过 30 组/周以免恢复不足"
    else:
        ok = 8 <= total_sets <= 20
        advise = "维持/入门建议每周 8-20 组"
    return {"total_sets": total_sets, "ok": ok, "advice": advise}


def check_progressive_overload(current_plan: dict, history: list = None) -> dict:
    """检查当前计划是否体现了渐进超负荷原则。"""
    issues = []
    for day in current_plan.get("days", []):
        for ex in day.get("exercises", []):
            reps = ex.get("reps", "")
            if isinstance(reps, str) and "-" not in reps:
                issues.append(f"{ex.get('name', '?')}: 建议用范围次数(如 8-12)而非固定次数")
    return {"ok": len(issues) == 0, "issues": issues,
            "principle": "渐进超负荷需要可追踪的进步路径"}
