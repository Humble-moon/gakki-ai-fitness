"""饮食计划的终态闸门 —— 镜像 plan_finalization 的语义，饮食特有的完整性判据。

## 为什么镜像而不是另起一套

训练侧在这条路径上踩过三次同源的坑（历史检查污染终判、交付误绑可持久化、
issues 非空当阻断），每一次的表现都是「计划永远到不了用户手里」且不报错。
饮食侧复用同一套判据结构，就把这三类错误一次性挡在门外：

- **只看终轮检查**决定 requires_review / warnings；
- **交付闸门与缓存闸门分离**：``_delivery_allowed = hard_ok``，
  ``_persistence_allowed = hard_ok and issues_empty``；
- **warning 永不阻断**（建议性提示随计划交付展示）；
- **急症/危机一票否决**由调用方在闸门之前短路，这里不再重复判定。

## 饮食特有的一条硬判据

训练侧「库外动作只警告不阻断」是因为库外动作仍可执行；饮食侧**库外食物没有
可重算的营养数字**，交付一行没有热量依据的食物违背「数字不交给模型」的立项
原则。所以调用方必须先替换/摘除库外食物；若终态时仍有残留，记入
``_delivery_block_reasons``（完整性问题，走 generation_failed 而非安全拒绝）。

过敏/疾病冲突则相反：**硬阻断**，走 safety_blocked 并点名具体食物。
"""

from __future__ import annotations

from src.agents.output_validation import OutputValidationError, validate_diet_plan
from src.core.plan_finalization import (
    EMERGENCY_STOP_MESSAGE,
    EMERGENCY_STOP_NEXT_STEP,
    EMERGENCY_STOP_STATUS,
    GENERATION_FAILED_STATUS,
    SAFETY_BLOCKED_STATUS,
    _strip_internal_markers,
)

# 与训练侧同值（训练侧为字面量，无集中常量）；前端守卫按同一字符串分流。
SAFE_DELIVERED_STATUS = "safe_delivered"
DIET_CRISIS_STOP_STATUS = "diet_crisis_stopped"

#: 饮食危机终态话术。与急症话术互斥、同优先级层：不评估严重程度、不给任何
#: 饮食计划内容、不教任何限制进食的方法；指向专业支持而非产品内重试。
DIET_CRISIS_STOP_MESSAGE = (
    "你提到的想法（如催吐、绝食、极端低热量或减肥药）可能伤害身体，"
    "我不能为你生成配合这些做法的饮食计划，也没有资格判断它对你是否安全。"
    "与进食相关的痛苦值得被认真帮助——注册营养师、精神科/心理科医生或"
    "进食障碍援助热线都能提供比任何食谱更有用的支持。"
)
DIET_CRISIS_STOP_NEXT_STEP = (
    "先联系专业支持（医院临床营养科/心理科，或当地进食障碍援助热线）。"
    "如果你愿意规律吃饭、恢复体力，回来告诉我，我可以为你配一份不设缺口的维持饮食。"
)


def _issue_text(issue) -> str:
    return issue.get("issue", str(issue)) if isinstance(issue, dict) else str(issue)


def finalize_diet_result(result: dict, checks: list[dict], rewrite_count: int,
                         *, provider_degraded: bool = False,
                         expected_goal: str | None = None,
                         extra_warnings: list[str] | None = None) -> dict:
    """归一一个饮食终态并判定两道闸门。判据结构与训练侧逐条对应。

    ``extra_warnings``：护栏/疾病覆盖/确定性替换/趋势词等非 LLM 来源的提示，
    随计划交付展示，**不进任何闸门**（训练侧 cautions 的同构通道）。
    """
    result = dict(result) if isinstance(result, dict) else {}
    checks = checks if isinstance(checks, list) else []
    final = checks[-1] if checks and isinstance(checks[-1], dict) else {}
    active_issues = final.get("issues") if isinstance(final.get("issues"), list) else []
    active_values = {_issue_text(i) for i in active_issues}

    resolved_issues, seen_resolved = [], set()
    for check in checks[:-1]:
        if not isinstance(check, dict):
            continue
        for issue in check.get("issues") or []:
            value = _issue_text(issue)
            if value not in active_values and value not in seen_resolved:
                seen_resolved.add(value)
                resolved_issues.append(issue)

    warnings = [_issue_text(i) for i in active_issues]
    warnings += [str(w) for w in (extra_warnings or []) if str(w).strip()]
    cautions = [str(c) for c in (final.get("review_cautions") or []) if str(c).strip()]
    warnings += cautions

    try:
        validate_diet_plan(result)
        schema_valid = True
    except OutputValidationError:
        schema_valid = False

    # 饮食完整性：库外食物残留 = 没有营养依据的食物行，不得交付。
    unknown_foods = [
        {"food": item.get("name"), "day_type": day.get("day_type"), "meal": meal.get("meal")}
        for day in result.get("days") or []
        for meal in day.get("meals") or []
        for item in meal.get("foods") or []
        if item.get("_unknown")
    ]
    integrity_ok = not unknown_foods

    is_safe = final.get("is_safe") is True
    issues_empty = final.get("issues") == []
    requires_review = final.get("requires_human_review") is True
    confidence = final.get("confidence")
    confidence_valid = isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
    goal_matches = expected_goal is None or result.get("goal") == expected_goal
    degraded = bool(provider_degraded or result.get("_degraded", False))

    result.update({
        "warnings": warnings,
        "active_issues": active_issues,
        "resolved_issues": resolved_issues,
        "unknown_foods": unknown_foods,
        "cautions": cautions,
        "requires_review": requires_review or not checks or not is_safe,
        "review_reason": final.get("review_reason", ""),
        "review_severity": final.get("review_severity", ""),
        "review_suggestions": list(final.get("review_suggestions") or []),
        "confidence": confidence if confidence_valid else 0,
        "rewrite_count": rewrite_count,
    })

    hard_ok = bool(schema_valid and integrity_ok and is_safe and not requires_review
                   and confidence_valid and goal_matches and not degraded)
    result["_persistence_allowed"] = bool(hard_ok and issues_empty)
    result["_delivery_allowed"] = hard_ok

    safety_block = None
    if is_safe is False or requires_review:
        safety_block = {
            "conflicts": list(final.get("review_conflicts") or []),
            "reason": final.get("review_reason") or "",
            "severity": final.get("review_severity") or "",
        }
    result["_safety_block"] = safety_block

    block_reasons: list[str] = []
    if not schema_valid:
        block_reasons.append("计划结构不完整")
    if not integrity_ok:
        block_reasons.append("存在食物库未收录、无法核对营养的食物")
    if not confidence_valid:
        block_reasons.append("缺少置信度信息")
    if not goal_matches:
        block_reasons.append("计划目标与你的目标不一致")
    if degraded:
        block_reasons.append("生成服务降级")
    result["_delivery_block_reasons"] = block_reasons
    return result


def sanitize_diet_warnings(warnings) -> list[str]:
    """交付载荷剥内部标记（[规则引擎]/触发词/建议人工审核），实质结论保留。

    复用训练侧同一个剥离函数——分类依赖原始标记，展示依赖剥后文本，
    两处各写一份正则早晚会漂移。
    """
    return [_strip_internal_markers(w) for w in (warnings or [])]


def diet_crisis_stop_payload(matched: list[str] | None = None) -> dict:
    """硬危机终态：**不含任何食物内容**，days 显式空（下游收录守卫同训练侧）。"""
    return {
        "delivery_status": DIET_CRISIS_STOP_STATUS,
        "requires_review": False,
        "days": [],
        "plan_name": "已暂停生成饮食计划",
        "goal": None,
        "crisis": {
            "title": "先照顾自己，比配餐更重要",
            "message": DIET_CRISIS_STOP_MESSAGE,
            "matched_signals": list(matched or []),
            "next_step": DIET_CRISIS_STOP_NEXT_STEP,
        },
    }


def diet_emergency_stop_payload(matched: list[str] | None = None) -> dict:
    """急症终态（饮食入口复用同一套急症话术常量，仅计划名改为饮食语境）。"""
    return {
        "delivery_status": EMERGENCY_STOP_STATUS,
        "requires_review": False,
        "days": [],
        "plan_name": "已暂停生成饮食计划",
        "goal": None,
        "emergency": {
            "title": "先停下来，尽快就医",
            "message": EMERGENCY_STOP_MESSAGE,
            "matched_signals": list(matched or []),
            "next_step": EMERGENCY_STOP_NEXT_STEP,
        },
    }


def diet_generation_failed_payload(result: dict | None = None) -> dict:
    """非安全类拒绝：这一次没生成好，重试即可。不含任何食物内容。"""
    result = result or {}
    reasons = [str(r) for r in (result.get("_delivery_block_reasons") or []) if str(r).strip()]
    payload = {
        "delivery_status": GENERATION_FAILED_STATUS,
        "requires_review": False,
        "days": [],
        "plan_name": "这次没能生成饮食计划",
        "goal": None,
        "warnings": sanitize_diet_warnings(result.get("warnings")),
        "failure": {
            "title": "这次没能生成出一份合格的饮食计划",
            "reasons": reasons,
            "message": "生成结果没有通过交付前的检查，可能是模型这一次的输出不完整。",
            "next_step": "直接再生成一次即可；如果连着两次都这样，检查一下档案与饮食偏好有没有漏填。",
        },
    }
    for key in ("rewrite_count", "_persistence_allowed", "_delivery_allowed"):
        if key in result:
            payload[key] = result[key]
    return payload


def diet_safety_blocked_payload(result: dict | None = None) -> dict:
    """安全类拒绝：点名冲突食物与原因，下一步是改填写或咨询专业人士。

    与 generation_failed 的区别在责任归属：这份计划本身与用户身体状况冲突，
    重试大概率同样结果；与送审的区别在有没有出路——本项目没有审核人角色。

    拒绝可能来自两条路：确定性食物冲突（review_conflicts），或语义审查判
    danger（is_safe=False 但无具体冲突食物）。后者必须把审查指出的问题透出，
    否则用户只看到一句空泛的「没有通过安全检查」，无从知道该改什么——
    实测糖尿病 persona 被审查拦下时 conflicts 为空，就是这个缺口。
    """
    result = result or {}
    block = result.get("_safety_block") or {}
    conflicts = block.get("conflicts") or []

    foods, reasons = [], []
    for item in conflicts:
        if not isinstance(item, dict):
            continue
        food = item.get("food")
        if food and food not in foods:
            foods.append(food)
        reason = item.get("reason")
        if reason and reason not in reasons:
            reasons.append(reason)

    issues = sanitize_diet_warnings(
        [_issue_text(i) for i in (result.get("active_issues") or [])])
    issues = [i for i in issues if i.strip()]

    if foods:
        title = "这些食物和你目前的情况有冲突"
        message = (
            f"计划里的「{'、'.join(foods)}」与你填写的过敏/健康状况/忌口冲突"
            f"（{'；'.join(reasons) if reasons else '存在禁忌'}）。"
            "替换之后仍没能凑出一份完整且安全的餐单，所以这次没有生成。"
        )
    elif issues:
        title = "这份餐单没通过安全检查"
        message = (
            "安全检查认为这份餐单不适合你目前的情况："
            + "；".join(issues[:2])
            + "。这次没有生成，调整后再试或先咨询专业人士。"
        )
    else:
        title = "这份餐单没通过安全检查"
        message = block.get("reason") or "这份饮食计划没有通过安全检查，所以这次没有生成。"

    return {
        "delivery_status": SAFETY_BLOCKED_STATUS,
        "requires_review": False,
        "days": [],
        "plan_name": "这次没有生成饮食计划",
        "goal": None,
        "warnings": sanitize_diet_warnings(result.get("warnings")),
        "safety": {
            "title": title,
            "message": message,
            "conflicts": conflicts,
            "prohibited_foods": foods,
            "issues": issues,
            "next_step": "调整过敏/忌口/健康状况的填写，或先咨询医生、临床营养师确认后再试。",
        },
    }


def prune_conflicting_foods(plan: dict, conflict_names) -> tuple[dict, list[str]]:
    """重写预算尽后的确定性兜底：摘除冲突食物行，返回 (新计划, 摘掉的名字)。

    某餐被摘空时整餐移除；某日型餐次少于 3 时该日型结构不再合格——调用方据
    schema 校验决定拒绝（安全冲突未解走 safety_blocked，结构问题走 failed）。
    少几样食物的餐单仍然可以吃，空手而归不行（训练侧 prune 的同源取舍）。
    """
    targets = {n for n in (conflict_names or []) if n}
    if not targets or not isinstance(plan, dict):
        return plan, []
    pruned = dict(plan)
    removed: list[str] = []
    days = []
    for day in plan.get("days") or []:
        new_day = dict(day)
        meals = []
        for meal in day.get("meals") or []:
            kept = [f for f in (meal.get("foods") or []) if f.get("name") not in targets]
            dropped = [f.get("name") for f in (meal.get("foods") or []) if f.get("name") in targets]
            removed.extend(n for n in dropped if n not in removed)
            if kept:
                new_meal = dict(meal)
                new_meal["foods"] = kept
                meals.append(new_meal)
        new_day["meals"] = meals
        days.append(new_day)
    pruned["days"] = days
    return pruned, removed