"""饮食计划生成服务 —— 独立 JSON 链路的胶水层。

先例是 ``training_advice.build_advice``：core 里带 IO 的服务函数，端点用
``run_in_threadpool`` 调它（LLM 全同步，async 端点直跑会冻结事件循环）。
**不进 LangGraph、不加 SSE**：饮食是单轮约束选择任务，图的 checkpoint/thread
纯属开销；consumeSse 契约也禁止第 5 个 SSE 消费。

流程（顺序即安全顺序）：
信号预检（急症 > 硬危机）→ 记忆并集读回 → 护栏/疾病目标覆盖 → 日拆分/餐分配
→ 知识降级检索 → 配餐 → 重算 → 库外处置 → 冲突确定性替换 → ≤2 轮重写
→ 摘除兜底 → 语义审查合并确定性冲突 → 终态闸门 → 可解释 → 交付/拒绝
→ 交付成功才写记忆（仅非空，防过敏史被空值覆盖）。
"""

from __future__ import annotations

import logging

from src.agents import diet_writer
from src.agents.diet_checker import check_diet, merge_deterministic_conflicts
from src.agents.output_validation import OutputValidationError
from src.core import diet_safety
from src.core.diet_explanation import build_diet_explanation
from src.core.diet_finalization import (
    SAFE_DELIVERED_STATUS,
    diet_crisis_stop_payload,
    diet_emergency_stop_payload,
    diet_generation_failed_payload,
    diet_safety_blocked_payload,
    finalize_diet_result,
    prune_conflicting_foods,
    sanitize_diet_warnings,
)
from src.core.nutrition import (
    allocate_meals,
    apply_energy_guardrails,
    calc_calorie_target,
    split_day_targets,
    split_macros,
)
from src.core.plan_finalization import long_term_user_key, persist_long_term_profile
from src.llm.provider import LLMUnavailableError
from src.rag import food_catalog

logger = logging.getLogger(__name__)

MAX_DIET_REWRITES = 2
_KNOWLEDGE_QUERY = "{goal} 饮食计划 蛋白质 每餐分配 热量 买菜清单"
_SOFT_CRISIS_WARNING = (
    "你提到的进食困扰值得被认真帮助：本计划不设热量缺口，以规律三餐为目标；"
    "建议同时寻求注册营养师或心理科医生的专业支持。"
)


def build_diet_plan(profile: dict, diet_prefs: dict, *, llm, knowledge=None,
                    long_term=None, owner_key: str | None = None,
                    query: str = "", meal_count: int = 4) -> dict:
    """生成一份饮食计划终态载荷（safe_delivered / 四种拒绝态之一）。"""
    profile = dict(profile or {})
    diet_prefs = dict(diet_prefs or {})
    meal_count = 3 if int(meal_count or 4) == 3 else 4

    # 1) 信号预检：急症优先于危机；两者载荷都不含任何食物内容。
    signals = diet_safety.scan_diet_signals(
        query=query, injuries=profile.get("injuries"),
        extra_text=diet_prefs.get("notes", ""),
    )
    if signals["emergency"]:
        return diet_emergency_stop_payload(signals["emergency"])
    crisis = signals["crisis"]
    if crisis and crisis["level"] == "hard":
        return diet_crisis_stop_payload(crisis["matched"])

    extra_warnings: list[str] = []
    if signals["trends"]:
        extra_warnings.append(
            f"你提到「{'、'.join(signals['trends'])}」：这类做法证据有限且可能反弹，"
            "本计划不采用极端限制；如想尝试请先咨询营养师。")

    # 2) 记忆并集读回：请求没带 ≠ 没有。过敏是安全关键，绝不静默丢弃。
    effective = _union_prefs(profile, diet_prefs, long_term, owner_key)

    # 3) 目标与护栏（确定性，数字不交给模型）。
    targets = calc_calorie_target(
        profile.get("weight"), profile.get("height"), profile.get("age"),
        profile.get("sex", "不填"), profile.get("goal", "增肌"),
        profile.get("days_per_week"),
    )
    guarded = apply_energy_guardrails(
        targets, weight_kg=profile.get("weight"),
        age=profile.get("age"), height_cm=profile.get("height"))
    targets, flags = guarded["targets"], guarded["flags"]
    extra_warnings += guarded["warnings"]
    if crisis and crisis["level"] == "soft":
        targets = dict(targets, daily_kcal=targets["tdee"], delta_pct=0,
                       goal_note="维持期（关怀护栏：不设热量缺口）")
        targets.update(split_macros(targets["daily_kcal"], targets["protein_g"]))
        flags = dict(flags, deficit_blocked=True)
        extra_warnings.append(_SOFT_CRISIS_WARNING)
    disease = diet_safety.apply_disease_target_rules(
        targets, effective.get("diseases"), profile.get("weight"))
    targets = disease["targets"]
    extra_warnings += disease["warnings"]

    # 4) 日拆分与餐分配（周加权守恒；蛋白均分）。
    days_per_week = profile.get("days_per_week") or 4
    day_split = split_day_targets(targets["daily_kcal"], days_per_week)
    day_targets = {"training_day_kcal": day_split["training_day_kcal"],
                   "rest_day_kcal": day_split["rest_day_kcal"],
                   "protein_g": targets["protein_g"]}
    meal_slots = {
        "训练日": allocate_meals(day_split["training_day_kcal"], targets["protein_g"],
                                 meal_count=meal_count),
        "休息日": allocate_meals(day_split["rest_day_kcal"], targets["protein_g"],
                                 meal_count=meal_count),
    }

    # 5) 知识检索（增强项，降级不阻断）。
    knowledge_ctx, knowledge_refs = _retrieve_knowledge(knowledge, profile.get("goal", "增肌"))

    # 6) 配餐清单先滤禁忌——模型看不到就不会选。
    catalog = list(food_catalog.load_foods())
    kept = diet_writer.filter_catalog_for_prefs(catalog, effective)
    kept_names = {e["name"] for e in kept}
    forbidden = [e["name"] for e in catalog if e["name"] not in kept_names]

    try:
        raw = _select_with_retry(llm, profile, day_targets, meal_slots, kept,
                                 effective, knowledge_ctx, forbidden)
    except LLMUnavailableError as exc:
        logger.warning("[diet_service] LLM 不可用：%s", exc)
        return diet_generation_failed_payload(
            {"_delivery_block_reasons": ["生成服务不可用"], "warnings": extra_warnings})
    except OutputValidationError:
        return diet_generation_failed_payload(
            {"_delivery_block_reasons": ["计划结构不完整"], "warnings": extra_warnings})

    # 7) 重算 → 库外处置 → 冲突替换/重写/摘除。
    plan = diet_writer.recompute_nutrition(raw)
    substitutions: list[str] = []
    plan, substitutions = _resolve_unknown_foods(plan, substitutions)
    checks: list[dict] = []
    rewrite_count = 0
    for attempt in range(MAX_DIET_REWRITES + 1):
        plan = diet_writer.recompute_nutrition(plan)
        conflicts = diet_safety.check_food_conflicts(_located_foods(plan), effective)
        if not conflicts:
            break
        if attempt == MAX_DIET_REWRITES:
            plan, removed = prune_conflicting_foods(plan, [c["food"] for c in conflicts])
            substitutions += [f"{name}（摘除）" for name in removed]
            plan = diet_writer.recompute_nutrition(plan)
            break
        plan, substitutions = _substitute_conflicts(plan, conflicts, effective, substitutions)
        plan = diet_writer.recompute_nutrition(plan)
        if not diet_safety.check_food_conflicts(_located_foods(plan), effective):
            break
        rewrite_count += 1
        try:
            plan = diet_writer.rewrite_with_forbidden(
                llm, plan, conflicts, catalog=kept,
                day_targets=day_targets, meal_slots=meal_slots)
        except OutputValidationError:
            continue  # 重写失败保留确定性替换版本，下一轮继续复检
        except LLMUnavailableError:
            break

    # 7.5) 确定性克数校准：模型选的量可能显著偏目标（实测蛋白达标、热量短三成），
    # 闸门只查安全不查量——数字由服务端负责到底：蛋白先调、热量后补，食物选择不变；
    # 现有食物到合理上限仍不足时，从禁忌已滤除的食物库补主食凑足热量。
    plan, calibrated = diet_writer.calibrate_portions(plan, meal_slots, top_up_catalog=kept)

    # 8) 语义审查 + 确定性冲突独立合并（LLM 判 safe 也拦得住过敏）。
    plan = diet_writer.recompute_nutrition(plan)
    # 校准后仍低于目标一成以上：如实披露缺口，不假装达标（数字可核对的底线）。
    for day in plan.get("days") or []:
        day_target = (day_split["training_day_kcal"] if day.get("day_type") == "训练日"
                      else day_split["rest_day_kcal"])
        day_actual = sum(m.get("kcal", 0) for m in day.get("meals") or [])
        if day_target and day_actual < day_target * 0.9:
            extra_warnings.append(
                f"{day.get('day_type')}按合理单餐份量校准后约 {round(day_actual)} kcal，"
                f"低于目标 {round(day_target)} kcal；可在加餐补充主食与蛋白"
                "（如牛奶+全麦面包），或咨询营养师调整目标。")
    final_conflicts = diet_safety.check_food_conflicts(_located_foods(plan), effective)
    check = check_diet(llm, plan, profile, effective)
    check = merge_deterministic_conflicts(check, final_conflicts)
    checks.append(check)
    if check.get("provider_degraded"):
        # 审查服务降级 = 「这次没生成好」，不是用户身体有冲突；
        # 走 generation_failed 让用户重试，而非 safety_blocked 误导责任归属。
        return diet_generation_failed_payload(
            {"_delivery_block_reasons": ["生成服务降级"], "warnings": extra_warnings})
    if substitutions:
        extra_warnings.append("以下食物因冲突或库外已被替换/摘除：" + "、".join(substitutions))

    # 9) 终态闸门（镜像训练侧六条件 + 双闸门 + 终轮）。
    result = finalize_diet_result(
        plan, checks, rewrite_count,
        provider_degraded=bool(check.get("provider_degraded")),
        expected_goal=profile.get("goal"), extra_warnings=extra_warnings)
    if not result["_delivery_allowed"]:
        if result["_safety_block"]:
            return diet_safety_blocked_payload(result)
        return diet_generation_failed_payload(result)

    # 10) 交付载荷：数字目标 + 可解释 + 剥内部标记后的 warnings。
    delivered = dict(result)
    delivered["delivery_status"] = SAFE_DELIVERED_STATUS
    delivered["warnings"] = sanitize_diet_warnings(result.get("warnings"))
    delivered["daily_targets"] = dict(targets, **day_split)
    delivered["nutrition"] = targets
    delivered["explain"] = build_diet_explanation(
        targets=targets, flags=flags, result=result,
        catalog_total=len(catalog), catalog_kept=len(kept),
        knowledge_refs=knowledge_refs, substitutions=substitutions,
        prefs=effective, days_per_week=days_per_week, calibrated=calibrated)

    # 11) 交付成功才写记忆；写并集后的有效偏好（防请求空值清掉过敏史）。
    if long_term is not None and owner_key:
        try:
            persist_long_term_profile(
                long_term, long_term_user_key(profile, owner_key),
                dict(profile, diet_preferences=effective))
        except Exception as exc:  # noqa: BLE001 - 记忆失败不拖垮交付
            logger.warning("[diet_service] 记忆写入跳过：%s", exc)
    return delivered


# ----------------------------------------------------------------------
# 内部 helpers
# ----------------------------------------------------------------------

def _union_prefs(profile: dict, diet_prefs: dict, long_term, owner_key) -> dict:
    effective = {
        "allergies": list(diet_prefs.get("allergies") or []),
        "diseases": list(diet_prefs.get("diseases") or []),
        "avoid_foods": list(diet_prefs.get("avoid_foods") or []),
        "notes": diet_prefs.get("notes", ""),
    }
    if long_term is None or not owner_key:
        return effective
    try:
        stored = (long_term.get_preferences(long_term_user_key(profile, owner_key))
                  or {}).get("diet_preferences") or {}
    except Exception as exc:  # noqa: BLE001 - 记忆读失败不阻断生成
        logger.warning("[diet_service] 记忆读回跳过：%s", exc)
        return effective
    for key in ("allergies", "diseases", "avoid_foods"):
        merged = list(stored.get(key) or [])
        for item in effective[key]:
            if item not in merged:
                merged.append(item)
        effective[key] = merged
    if not effective["notes"]:
        effective["notes"] = stored.get("notes", "")
    return effective


def _retrieve_knowledge(knowledge, goal: str) -> tuple[str, list[str]]:
    if knowledge is None:
        return "", []
    try:
        chunks = knowledge.search_with_fallback(_KNOWLEDGE_QUERY.format(goal=goal)) or []
    except Exception as exc:  # noqa: BLE001 - 知识检索是增强项
        logger.debug("[diet_service] 知识检索降级：%s", exc)
        return "", []
    refs, parts = [], []
    for chunk in chunks[:3]:
        if not isinstance(chunk, dict):
            continue
        content = str(chunk.get("content") or "")[:400]
        if content:
            parts.append(content)
        source = chunk.get("source") or chunk.get("source_file") or ""
        if source and source not in refs:
            refs.append(str(source))
    return "\n---\n".join(parts), refs


def _select_with_retry(llm, profile, day_targets, meal_slots, kept,
                       effective, knowledge_ctx, forbidden) -> dict:
    try:
        return diet_writer.select_foods(
            llm, profile=profile, day_targets=day_targets, meal_slots=meal_slots,
            catalog=kept, prefs=effective, knowledge_ctx=knowledge_ctx,
            forbidden_names=forbidden)
    except OutputValidationError:
        return diet_writer.select_foods(
            llm, profile=profile, day_targets=day_targets, meal_slots=meal_slots,
            catalog=kept, prefs=effective, knowledge_ctx=knowledge_ctx,
            forbidden_names=forbidden)


def _located_foods(plan: dict) -> list[dict]:
    return [dict(item, day_type=day.get("day_type"), meal=meal.get("meal"))
            for day in plan.get("days") or []
            for meal in day.get("meals") or []
            for item in meal.get("foods") or []]


def _resolve_unknown_foods(plan: dict, substitutions: list[str]) -> tuple[dict, list[str]]:
    """库外食物摘除 + 同餐克数再平衡（绝不交付无宏量食物行）。"""
    unknown = diet_writer.collect_unknown_foods(plan)
    if not unknown:
        return plan, substitutions
    names = {u["food"] for u in unknown}
    out = dict(plan)
    days = []
    for day in plan.get("days") or []:
        new_day = dict(day)
        meals = []
        for meal in day.get("meals") or []:
            kept_foods = [f for f in meal.get("foods") or [] if f.get("name") not in names]
            if not kept_foods:
                continue  # 整餐库外：移除该餐（结构不合格时闸门会拒绝）
            new_meal = dict(meal)
            new_meal["foods"] = kept_foods
            meals.append(new_meal)
        new_day["meals"] = meals
        days.append(new_day)
    out["days"] = days
    substitutions = substitutions + [f"{name}（库外摘除）" for name in sorted(names)]
    return out, substitutions


def _substitute_conflicts(plan: dict, conflicts: list[dict], prefs: dict,
                          substitutions: list[str]) -> tuple[dict, list[str]]:
    """确定性替换优先于 LLM 重写：同类 + 无禁忌 + 宏量接近的首选候选。"""
    tags = diet_safety.resolve_allergens(prefs.get("allergies"))
    diseases = diet_safety.resolve_diseases(prefs.get("diseases"))
    exclude_flags: set[str] = set()
    exclude_tags: set[str] = set(tags)
    for disease in diseases:
        for rule in (diet_safety.DISEASE_FOOD_RULES.get(disease),
                     diet_safety.DISEASE_TARGET_RULES.get(disease)):
            if rule:
                exclude_flags.update(rule.get("exclude_flags", ()))
                exclude_tags.update(rule.get("exclude_allergens", ()))
    replacements: dict[str, str] = {}
    for conflict in conflicts:
        food = conflict["food"]
        if food in replacements:
            continue
        subs = food_catalog.find_substitutes(
            food, exclude_allergens=sorted(exclude_tags),
            exclude_flags=sorted(exclude_flags), limit=3)
        if subs:
            replacements[food] = subs[0]["name"]
    if not replacements:
        return plan, substitutions
    out = dict(plan)
    days = []
    for day in plan.get("days") or []:
        new_day = dict(day)
        meals = []
        for meal in day.get("meals") or []:
            new_meal = dict(meal)
            new_meal["foods"] = [
                dict(f, name=replacements.get(f.get("name"), f.get("name")))
                for f in meal.get("foods") or []
            ]
            meals.append(new_meal)
        new_day["meals"] = meals
        days.append(new_day)
    out["days"] = days
    substitutions = substitutions + [
        f"{old}→{new}" for old, new in sorted(replacements.items())]
    return out, substitutions