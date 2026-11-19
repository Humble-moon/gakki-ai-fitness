"""DietWriter —— 饮食计划的食物选择与确定性营养重算。

分工（与训练侧 writer 同构但更严）：
- LLM 只回答「选哪样食物、吃多少克」；
- **每食物/每餐/每日的热量与宏量一律由 food_catalog 按克数重算**，模型报来的
  任何数字字段都被剥掉。这是「数字结论不交给模型」在饮食侧的落点。

库外食物（清单外名字）不在这里静默接受：重算时标 ``_unknown``，由
diet_service 走确定性替换/摘除——绝不交付一行没有营养依据的食物。
"""

from __future__ import annotations

import logging

from src.agents.output_validation import OutputValidationError, validate_diet_plan
from src.config import REWRITE_MODEL
from src.llm.prompts.diet_writer import (
    build_diet_rewrite_messages,
    build_diet_writer_messages,
)
from src.rag import food_catalog

logger = logging.getLogger(__name__)

_NUTRIENT_KEYS = ("kcal", "protein_g", "fat_g", "carbs_g")


def select_foods(llm, *, profile: dict, day_targets: dict, meal_slots: dict,
                 catalog: list[dict], prefs: dict, knowledge_ctx: str = "",
                 forbidden_names: list[str] | None = None) -> dict:
    """单轮配餐。chat 模型（约束选择任务，非创作）；解析失败抛错交上层重试。"""
    messages = build_diet_writer_messages(
        profile, day_targets, meal_slots, catalog, prefs,
        knowledge_ctx=knowledge_ctx, forbidden_names=forbidden_names,
    )
    raw = llm.chat_with_json_mode(messages)
    return _validated(raw)


def rewrite_with_forbidden(llm, plan: dict, conflicts: list[dict], *,
                           catalog: list[dict], day_targets: dict,
                           meal_slots: dict) -> dict:
    """按冲突清单重写（快模型；重写后必有重算+冲突复检，安全网不变薄）。"""
    messages = build_diet_rewrite_messages(plan, conflicts, catalog, day_targets, meal_slots)
    raw = llm.chat_with_json_mode(messages, model=REWRITE_MODEL)
    return _validated(raw)


def _validated(raw) -> dict:
    if not isinstance(raw, dict) or "raw" in raw or raw.get("_degraded"):
        raise OutputValidationError("diet writer returned unparseable or degraded output")
    return validate_diet_plan(raw)


def recompute_nutrition(plan: dict) -> dict:
    """按克数重算全部营养数字，剥掉模型给的一切数字字段。

    返回新字典（不修改入参）。每个食物重建为
    ``{name, grams, kcal, protein_g, fat_g, carbs_g}``；库外食物保留
    name/grams 并标 ``_unknown=True``（营养字段缺省 0 且不计入合计置信）。
    每餐/每日合计由重算后的食物值累加——合计与明细永远自洽。
    """
    out = dict(plan)
    days = []
    for day in plan.get("days") or []:
        new_day = dict(day)
        meals = []
        day_totals = {key: 0.0 for key in _NUTRIENT_KEYS}
        for meal in day.get("meals") or []:
            new_meal = dict(meal)
            foods = []
            meal_totals = {key: 0.0 for key in _NUTRIENT_KEYS}
            for item in meal.get("foods") or []:
                name = item.get("name") or ""
                grams = item.get("grams") or 0
                nutrients = food_catalog.food_nutrients(name, grams)
                if nutrients is None:
                    foods.append({"name": name, "grams": grams, "_unknown": True,
                                  **{key: 0 for key in _NUTRIENT_KEYS}})
                    continue
                foods.append(nutrients)
                for key in _NUTRIENT_KEYS:
                    meal_totals[key] += nutrients[key]
            new_meal["foods"] = foods
            new_meal.update({key: round(meal_totals[key], 1) for key in _NUTRIENT_KEYS})
            new_meal["kcal"] = round(meal_totals["kcal"])
            for key in _NUTRIENT_KEYS:
                day_totals[key] += meal_totals[key]
            meals.append(new_meal)
        new_day["meals"] = meals
        new_day.update({key: round(day_totals[key], 1) for key in _NUTRIENT_KEYS})
        new_day["total_kcal"] = round(day_totals["kcal"])
        new_day["total_protein_g"] = round(day_totals["protein_g"], 1)
        days.append(new_day)
    out["days"] = days
    return out


def collect_unknown_foods(plan: dict) -> list[dict]:
    """库外食物清单（带定位），供确定性替换/摘除与 warnings 自查覆盖率。"""
    unknown = []
    for day in plan.get("days") or []:
        for meal in day.get("meals") or []:
            for item in meal.get("foods") or []:
                if item.get("_unknown"):
                    unknown.append({"food": item.get("name"),
                                    "day_type": day.get("day_type"),
                                    "meal": meal.get("meal")})
    return unknown


def filter_catalog_for_prefs(catalog: list[dict], prefs: dict) -> list[dict]:
    """配餐前把禁忌食物从清单里滤掉——模型看不到，就不会选。

    过敏原标签 / 疾病食物层规则 / 忌口名字三路过滤；目标层规则（蛋白上限等）
    不在这里处理（那是总量问题，换菜解决不了）。
    """
    from src.core.diet_safety import (
        DISEASE_FOOD_RULES,
        DISEASE_TARGET_RULES,
        resolve_allergens,
        resolve_diseases,
    )

    tags = set(resolve_allergens(prefs.get("allergies")))
    diseases = resolve_diseases(prefs.get("diseases"))
    avoid = set()
    for raw_name in prefs.get("avoid_foods") or []:
        entry = food_catalog.get_food(raw_name)
        avoid.add(entry["name"] if entry else str(raw_name).strip())

    excluded_flags: set[str] = set()
    excluded_tags: set[str] = set()
    excluded_names: set[str] = set(avoid)
    for disease in diseases:
        for rule in (DISEASE_FOOD_RULES.get(disease), DISEASE_TARGET_RULES.get(disease)):
            if not rule:
                continue
            excluded_flags.update(rule.get("exclude_flags", ()))
            excluded_tags.update(rule.get("exclude_allergens", ()))
            excluded_names.update(rule.get("exclude_names", ()))

    kept = []
    for entry in catalog:
        if entry["name"] in excluded_names:
            continue
        if set(entry.get("allergens") or []) & (tags | excluded_tags):
            continue
        if any(entry.get(flag) for flag in excluded_flags):
            continue
        kept.append(entry)
    return kept


# ── 确定性克数校准 ──────────────────────────────────────────
# 真实端到端暴露的缺陷：模型选的食物蛋白基本达标、热量却短三成，
# 而闸门只查安全不查「量够不够」。数字结论既然不交给模型，就要负责到底：
# 食物选择保留，克数由服务端按各餐目标校准。不引入新的 LLM 轮次。
#
# 第二版教训：缺口不能全堆到一种食物上——实测校准出过糙米饭 600g、
# 香蕉 580g、杏仁 115g 的单餐，对糖尿病用户是医学上不当的血糖负荷。
# 现在按类别/密度给每种食物设合理上限，缺口在多个碳水间按现有克数分摊。
CAL_KCAL_TOL_RATIO = 0.08      # 餐次热量容差：max(50, 8%)——配餐凑数口径 ±50 的比例化
CAL_KCAL_TOL_FLOOR = 50.0
CAL_PROTEIN_TOL_G = 5.0        # 蛋白容差 ±5g（与 writer 提示词同一口径）
CAL_MAX_GRAMS = 600.0          # 绝对 ceiling（类别上限通常更紧）
CAL_MIN_GRAMS = 10.0
CAL_REDUCE_FLOOR_G = 20.0      # 减量下限：低于 20g 的行没有意义，不如不减
CAL_FAT_ADD_CAP_G = 15.0       # 脂肪兜底每食物增量上限——补热量不能倒一杯油


def _grams_cap(entry: dict) -> float:
    """单食物一餐的合理克数上限（按类别与能量密度）。

    纯油 15g、坚果 30g（知识文档「每天一小把」）、蛋白粉 60g；
    干重主食/奶酪等高密度食物 120g；熟主食 400g、水果 250g、蔬菜/乳品 300g、
    肉鱼蛋豆 250g。上限是「一餐吃这么多还合理」的常识边界，不是营养处方。
    """
    kcal_per_g = entry["kcal_per_100g"] / 100.0
    fat_per_g = entry["fat_g"] / 100.0
    protein_per_g = entry["protein_g"] / 100.0
    category = entry.get("category")
    if fat_per_g >= 0.8:
        return 15.0
    if category == "脂肪":
        return 30.0
    if protein_per_g >= 0.5:
        return 60.0
    if kcal_per_g >= 2.5:
        return 120.0
    if category == "主食":
        return 400.0
    if category == "水果":
        return 250.0
    if category in ("蔬菜", "乳品"):
        return 300.0
    return 250.0


def _per_gram(name: str) -> dict | None:
    entry = food_catalog.get_food(name)
    if entry is None:
        return None
    return {"kcal": entry["kcal_per_100g"] / 100.0,
            "p": entry["protein_g"] / 100.0,
            "c": entry["carbs_g"] / 100.0,
            "f": entry["fat_g"] / 100.0,
            "cap": _grams_cap(entry)}


def calibrate_portions(plan: dict, meal_slots: dict,
                       top_up_catalog: list[dict] | None = None) -> tuple[dict, bool]:
    """按各餐热量/蛋白目标校准克数，返回 (新计划, 是否动过)。

    顺序即意图：**蛋白先调**（增减蛋白密度最高的食物），**热量后补**
    （先加低蛋白密度碳水，仍不够才 capped 脂肪；过量反向减碳水）。
    反过来先补热量会把蛋白一起撑超——实测模型的蛋白密度大致是对的，
    缺的是碳水和脂肪，所以补热量必须挑低蛋白密度的食物。

    ``top_up_catalog``：禁忌已滤除的食物库条目。现有食物都到合理上限仍
    凑不够热量时，从中补充主食/乳品行（与确定性替换同源：服务端可以在
    安全边界内动食物）。不传则只调克数、不加食物。
    """
    out = dict(plan)
    days = []
    adjusted = False
    for day in plan.get("days") or []:
        slots = {s["meal"]: s for s in (meal_slots or {}).get(day.get("day_type"), [])}
        new_day = dict(day)
        meals = []
        for meal in day.get("meals") or []:
            new_meal = dict(meal)
            foods = [dict(f) for f in meal.get("foods") or []]
            slot = slots.get(meal.get("meal"))
            if slot and foods:
                adjusted = _calibrate_meal(foods, slot, top_up_catalog) or adjusted
            new_meal["foods"] = foods
            meals.append(new_meal)
        new_day["meals"] = meals
        days.append(new_day)
    out["days"] = days
    return out, adjusted


def _calibrate_meal(foods: list[dict], slot: dict,
                    top_up_catalog: list[dict] | None = None) -> bool:
    dens = {f["name"]: _per_gram(f["name"]) for f in foods}
    changed = False

    def cap_of(f: dict) -> float:
        d = dens.get(f["name"])
        return min(d["cap"], CAL_MAX_GRAMS) if d else CAL_MAX_GRAMS

    def totals() -> tuple[float, float]:
        kcal = protein = 0.0
        for f in foods:
            d = dens.get(f["name"])
            if d:
                kcal += f["grams"] * d["kcal"]
                protein += f["grams"] * d["p"]
        return kcal, protein

    def set_grams(f: dict, grams: float) -> None:
        nonlocal changed
        rounded = max(CAL_MIN_GRAMS, min(cap_of(f), round(grams / 5) * 5))
        if rounded != f["grams"]:
            f["grams"] = rounded
            changed = True

    # 0) 先把模型自己选出的离谱份量压回合理上限（杏仁 100g、油 50g 这类）。
    for f in foods:
        if dens.get(f["name"]) and f["grams"] > cap_of(f):
            set_grams(f, cap_of(f))

    # 1) 蛋白：偏差超 ±5g 时动蛋白密度最高且还有上限余量的食物；
    #    增量被上限截断时顺延到下一个候选，减量不低于 20g（再小就没意义）。
    _, protein = totals()
    pgap = float(slot["protein_target_g"]) - protein
    if pgap > CAL_PROTEIN_TOL_G:
        candidates = sorted(
            (f for f in foods if dens.get(f["name"]) and dens[f["name"]]["p"] > 0
             and cap_of(f) - f["grams"] > 0),
            key=lambda f: -dens[f["name"]]["p"])
        for f in candidates:
            if pgap <= CAL_PROTEIN_TOL_G:
                break
            d = dens[f["name"]]
            add = min(pgap / d["p"], cap_of(f) - f["grams"])
            before = f["grams"]
            set_grams(f, f["grams"] + add)
            pgap -= (f["grams"] - before) * d["p"]
    elif pgap < -CAL_PROTEIN_TOL_G:
        candidates = [f for f in foods if dens.get(f["name"]) and dens[f["name"]]["p"] > 0]
        if candidates:
            best = max(candidates, key=lambda f: dens[f["name"]]["p"])
            cut = min((-pgap) / dens[best["name"]]["p"],
                      best["grams"] - CAL_REDUCE_FLOOR_G)
            if cut > 0:
                set_grams(best, best["grams"] - cut)

    # 2) 热量：缺口在多个低蛋白密度碳水间**按现有克数分摊**（先比例、后贪心），
    #    脂肪兜底且增量/总量双 capped；过量则反向按比例减碳水。
    kcal, _ = totals()
    gap = float(slot["kcal_target"]) - kcal
    tol = max(CAL_KCAL_TOL_FLOOR, float(slot["kcal_target"]) * CAL_KCAL_TOL_RATIO)
    if gap > tol:
        pool = [f for f in foods if dens.get(f["name"])
                and dens[f["name"]]["p"] < 0.15 and dens[f["name"]]["c"] >= 0.10
                and cap_of(f) - f["grams"] > 0]
        weight = sum(f["grams"] for f in pool) or 1.0
        for f in pool:  # 比例分摊：大份多担，避免全堆到第一种
            d = dens[f["name"]]
            share = gap * f["grams"] / weight
            add = min(share / d["kcal"], cap_of(f) - f["grams"])
            if add > 0:
                set_grams(f, f["grams"] + add)
        kcal, _ = totals()
        gap = float(slot["kcal_target"]) - kcal
        for f in sorted(pool, key=lambda f: -dens[f["name"]]["c"]):  # 贪心补余量
            if gap <= tol:
                break
            d = dens[f["name"]]
            add = min(gap / d["kcal"], cap_of(f) - f["grams"])
            if add > 0:
                set_grams(f, f["grams"] + add)
                gap -= add * d["kcal"]
        if gap > tol:
            fats = sorted(
                (f for f in foods if dens.get(f["name"]) and dens[f["name"]]["f"] >= 0.30
                 and cap_of(f) - f["grams"] > 0),
                key=lambda f: -dens[f["name"]]["f"])
            for f in fats:
                if gap <= tol:
                    break
                d = dens[f["name"]]
                add = min(gap / d["kcal"], CAL_FAT_ADD_CAP_G, cap_of(f) - f["grams"])
                if add > 0:
                    set_grams(f, f["grams"] + add)
                    gap -= add * d["kcal"]
        # 现有食物都到合理上限仍凑不够：从禁忌已滤除的食物库补主食行。
        # 只补主食（碳水为主、蛋白极少），不会把已调好的蛋白撑超；
        # 与确定性替换同源——服务端可以在安全边界内动食物。
        if gap > tol and top_up_catalog and len(foods) < 8:
            existing = {f["name"] for f in foods}
            candidates = sorted(
                (e for e in top_up_catalog
                 if e["name"] not in existing and e.get("category") == "主食"
                 and 0.3 <= e["kcal_per_100g"] / 100.0 <= 2.5),
                key=lambda e: -e["carbs_g"])
            for entry in candidates[:2]:
                if gap <= tol or len(foods) >= 8:
                    break
                d = _per_gram(entry["name"])
                if d is None:
                    continue
                add = min(gap / d["kcal"], min(d["cap"], CAL_MAX_GRAMS))
                grams = max(CAL_MIN_GRAMS, round(add / 5) * 5)
                foods.append({"name": entry["name"], "grams": grams})
                dens[entry["name"]] = d
                gap -= grams * d["kcal"]
                changed = True
    elif gap < -tol:
        pool = [f for f in foods if dens.get(f["name"]) and dens[f["name"]]["p"] < 0.15
                and f["grams"] > CAL_REDUCE_FLOOR_G]
        weight = sum(f["grams"] for f in pool) or 1.0
        for f in pool:
            if gap >= -tol:
                break
            d = dens[f["name"]]
            share = (-gap) * f["grams"] / weight
            cut = min(share / d["kcal"], f["grams"] - CAL_REDUCE_FLOOR_G)
            if cut > 0:
                set_grams(f, f["grams"] - cut)
                gap += cut * d["kcal"]
    return changed