"""饮食计划的可解释块 —— 「这份饮食是怎么来的」四段式。

与训练侧 ``plan_explanation.build_explanation`` 同构：任何一段数据缺失时该段
为空（lines 为空列表），前端据此隐藏，**绝不编造**。安全检查段读 finalize 已
算好的 active/resolved 差集，不重算——重算会有两处口径，早晚对不上。

四段：目标依据 / 热量公式 / 食物选择 / 安全检查。

一条诚实性约束：餐次热量比例是**设计常数**（知识文档只给了每餐蛋白参考带），
lines 里必须如实标注「设计口径」，不得伪称知识出处；热量下限同理标注「估算护栏」。
"""

from __future__ import annotations


def build_diet_explanation(*, targets: dict, flags: dict | None = None,
                           result: dict | None = None,
                           catalog_total: int = 0, catalog_kept: int = 0,
                           knowledge_refs: list[str] | None = None,
                           substitutions: list[str] | None = None,
                           prefs: dict | None = None,
                           days_per_week: int | None = None,
                           calibrated: bool = False) -> dict:
    flags = flags or {}
    result = result or {}
    prefs = prefs or {}
    return {
        "goal": _explain_goal(targets, flags, prefs, days_per_week),
        "energy": _explain_energy(targets, flags),
        "foods": _explain_foods(catalog_total, catalog_kept, knowledge_refs,
                                substitutions, calibrated),
        "safety": _explain_safety(result),
    }


def _explain_goal(targets: dict, flags: dict, prefs: dict, days_per_week) -> dict:
    lines = []
    goal_note = targets.get("goal_note")
    if goal_note:
        lines.append(f"目标口径：{goal_note}")
    if days_per_week is not None:
        lines.append(f"联动训练：每周 {days_per_week} 练 → 训练日与休息日两套热量，"
                     f"周加权均值等于每日目标（多出的热量主要给训练日碳水）")
    diseases = prefs.get("diseases") or []
    if diseases:
        lines.append(f"健康状况已生效：{'、'.join(diseases)}（总量与食物两层约束）")
    allergies = prefs.get("allergies") or []
    if allergies:
        lines.append(f"过敏/忌口已生效：{'、'.join(allergies)}（冲突食物在配餐前已从清单滤除）")
    if flags.get("deficit_blocked"):
        lines.append("护栏：属禁缺口人群（未成年/低体重/孕期等），已取消热量缺口改为维持")
    return {"title": "目标依据", "lines": lines}


def _explain_energy(targets: dict, flags: dict) -> dict:
    lines = []
    if targets.get("formula"):
        lines.append(f"公式：{targets['formula']}｜BMR {targets.get('bmr')} kcal → "
                     f"TDEE {targets.get('tdee')} kcal → 每日 {targets.get('daily_kcal')} kcal")
    if targets.get("basis"):
        lines.append(f"依据：{targets['basis']}")
    if targets.get("protein_g") is not None:
        lines.append(f"蛋白质 {targets['protein_g']}g/天（按体重与目标取值），"
                     f"脂肪 {targets.get('fat_g')}g（扣除蛋白后剩余热量的 25%），"
                     f"碳水 {targets.get('carbs_g')}g 补齐")
    lines.append("餐次热量比例（早/午/晚/加餐）为设计口径；蛋白按日总量均分到各餐")
    if flags.get("floor_applied"):
        lines.append("热量已按安全下限钳制（估算护栏，非临床处方）")
    return {"title": "热量公式", "lines": lines}


def _explain_foods(catalog_total: int, catalog_kept: int,
                   knowledge_refs, substitutions, calibrated: bool = False) -> dict:
    lines = []
    if catalog_total:
        lines.append(f"食物库 {catalog_total} 条，按你的过敏/疾病/忌口滤除后剩 {catalog_kept} 条供配餐")
    if knowledge_refs:
        lines.append("参考知识：" + "、".join(knowledge_refs))
    else:
        lines.append("本次未检索到补充知识（不影响配餐：热量与宏量由公式与食物库确定）")
    if substitutions:
        lines.append("确定性替换/摘除：" + "、".join(substitutions))
    if calibrated:
        lines.append("克数已按各餐热量与蛋白目标校准（蛋白先调、热量后补）；"
                     "现有食物到合理份量上限仍不足时，由服务端从食物库补充主食凑足热量（禁忌已滤除）")
    lines.append("所有热量与宏量由服务端按克数重算，模型只负责选食物与克数")
    return {"title": "食物选择", "lines": lines}


def _explain_safety(result: dict) -> dict:
    lines = []
    checks_rounds = result.get("rewrite_count")
    if checks_rounds is not None:
        lines.append(f"重写 {checks_rounds} 轮；每轮重写后都有重算与冲突复检")
    confidence = result.get("confidence")
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        lines.append(f"检查把握：{confidence:.2f}")
    resolved = result.get("resolved_issues") or []
    if resolved:
        lines.append(f"已修正 {len(resolved)} 处问题：" +
                     "；".join(str(i.get("issue", i)) if isinstance(i, dict) else str(i)
                               for i in resolved[:3]))
    active = result.get("active_issues") or []
    if active:
        lines.append(f"保留 {len(active)} 条提示（不阻断交付，随计划展示）")
    return {"title": "安全检查", "lines": lines}