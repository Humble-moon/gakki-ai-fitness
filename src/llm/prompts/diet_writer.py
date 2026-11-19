"""
================================================================================
文件角色：DietWriter（饮食计划配餐器）Prompt 模板
================================================================================
- 被调用者：src/core/diet_service.py 的配餐管线（独立 JSON 链路，不进 LangGraph）。
- 调用者：src/agents/diet_writer.py 构造 messages，经 LLMProvider JSON 模式发送。
- 项目角色：饮食计划的食物选择由这个 Prompt 驱动；**一切数字由服务端重算**。

与训练侧 writer.py 同构的三条设计：
1. 约束前置——过敏/疾病/忌口/餐次目标写在 user 消息最前，模型先看到硬约束；
2. 常识内置——蛋白每餐均分、蔬菜每日 300g、±50kcal 容差写死在 prompt，
   不依赖检索质量（检索到的方法论只作补充上下文）；
3. canonical goal 钉死——goal 必须严格等于档案目标，system 末尾再钉一次。

关键差异：训练侧允许模型写 sets/reps（执行参数），饮食侧**禁止模型写任何
热量/宏量数字**——食物库才是数字的唯一来源，模型只回答「选哪样、吃多少克」。
================================================================================
"""

DIET_WRITER_SYSTEM = """你是健身营养配餐员。任务：从给定的食物清单中，为「训练日」和「休息日」各配一日餐食（早餐/午餐/晚餐，清单给出加餐时再加加餐）。

硬约束（违反视为严重错误）：
1. 只输出一个 JSON 对象，不要输出任何解释文字或 markdown 围栏。
2. foods 里每个食物只写 name 和 grams 两个字段。**禁止输出热量、蛋白质、脂肪、碳水等任何数字**——营养数字由服务端按食物库重算，你写的数字会被丢弃。
3. name 必须逐字来自食物清单（含生熟状态括号，如「米饭（蒸）」），不得编造清单外食物，不得改写名字。
4. 每餐蛋白质合计接近该餐 protein_target_g（±5g）；每餐热量合计接近 kcal_target（±50 大卡容差）。靠调整 grams 达成，不要靠换餐次结构。
5. 禁忌清单（过敏原/疾病/忌口）中的食物绝对不得出现，哪怕用户要求。
6. 训练日的碳水总量必须高于休息日（训练日多补碳水）；两日型蛋白质总量相同。
7. 每天蔬菜合计不少于 300g（深色绿叶菜优先）；脂肪来源优先橄榄油/坚果/鱼类，不用油炸做法。
8. 不推荐任何药物、减肥药、泻药或补剂剂量；蛋白粉仅作为清单内普通食物可选。
9. 克数按清单条目的生熟状态计（熟重就是熟重），常见份量参考 default_serving_g。
10. goal 必须严格等于用户档案目标（增肌 或 减脂），不得自行更改。

输出 JSON 结构：
{
  "plan_name": "计划名称（≤40字）",
  "goal": "增肌 或 减脂",
  "days": [
    {"day_type": "训练日",
     "meals": [{"meal": "早餐", "foods": [{"name": "燕麦粥（煮）", "grams": 250}], "note": "可选说明"}]},
    {"day_type": "休息日", "meals": [...]}
  ],
  "notes": "全局说明（可选，含 80/20 弹性原则提醒）",
  "shopping_list": ["买菜清单条目（可选）"]
}

再次强调：goal 严格等于档案目标；foods 只有 name 与 grams；名字逐字来自清单。"""


def _format_catalog(catalog: list[dict]) -> str:
    lines = []
    for entry in catalog:
        flags = [label for key, label in (
            ("high_gi", "高GI"), ("high_purine", "高嘌呤"),
            ("high_sodium", "高钠"), ("raw_or_high_mercury", "生食/高汞"),
        ) if entry.get(key)]
        allergens = "/".join(entry.get("allergens") or []) or "无"
        lines.append(
            f"- {entry['name']}｜每100g: {entry['kcal_per_100g']:.0f}kcal "
            f"蛋白{entry['protein_g']:.1f}g 脂肪{entry['fat_g']:.1f}g 碳水{entry['carbs_g']:.1f}g"
            f"｜过敏原:{allergens}｜标记:{'/'.join(flags) or '无'}"
            f"｜常见份量:{entry['default_serving_g']:.0f}g"
        )
    return "\n".join(lines)


def _format_targets(day_targets: dict, meal_slots: dict) -> str:
    lines = []
    for day_type in ("训练日", "休息日"):
        kcal = day_targets["training_day_kcal" if day_type == "训练日" else "rest_day_kcal"]
        lines.append(f"{day_type}：全天 {kcal} kcal，蛋白 {day_targets['protein_g']}g")
        for slot in meal_slots[day_type]:
            lines.append(f"  - {slot['meal']}：目标 {slot['kcal_target']} kcal，蛋白 {slot['protein_target_g']}g")
    return "\n".join(lines)


def build_diet_writer_messages(
    profile: dict,
    day_targets: dict,
    meal_slots: dict,
    catalog: list[dict],
    prefs: dict,
    knowledge_ctx: str = "",
    forbidden_names: list[str] | None = None,
) -> list[dict]:
    """约束前置组装：禁忌 → 目标 → 清单 → 档案 → 知识补充。

    ``catalog`` 应是服务端已过滤掉禁忌食物后的子集——模型看不到禁忌食物，
    多数过敏场景因此单轮即可交付，不需要重写。
    """
    forbidden = forbidden_names or []
    sections = [
        f"【禁忌清单（绝对不得出现）】{('、'.join(forbidden)) if forbidden else '无'}",
        f"【过敏原标签】{'、'.join(prefs.get('allergen_tags') or []) or '无'}",
        f"【健康状况】{'、'.join(prefs.get('diseases') or []) or '无'}",
        f"【忌口】{'、'.join(prefs.get('avoid_foods') or []) or '无'}",
        "【两日型热量与餐次目标】\n" + _format_targets(day_targets, meal_slots),
        "【食物清单（name 必须逐字取自此处）】\n" + _format_catalog(catalog),
        f"【用户档案】目标:{profile.get('goal')}｜体重:{profile.get('weight')}kg｜"
        f"每周训练:{profile.get('days_per_week')}天｜年龄:{profile.get('age') or '未填'}｜"
        f"性别:{profile.get('sex') or '不填'}",
    ]
    if knowledge_ctx:
        sections.append("【营养方法论参考（补充上下文，与硬约束冲突时以硬约束为准）】\n" + knowledge_ctx)
    sections.append(f"请为档案目标「{profile.get('goal')}」配餐。再次确认：goal 严格等于「{profile.get('goal')}」。")
    return [
        {"role": "system", "content": DIET_WRITER_SYSTEM},
        {"role": "user", "content": "\n\n".join(sections)},
    ]


def build_diet_rewrite_messages(plan: dict, conflicts: list[dict],
                                catalog: list[dict], day_targets: dict,
                                meal_slots: dict) -> list[dict]:
    """重写指令：点名冲突食物 + 禁用词清单 + 同类替代候选。

    与训练侧 rewrite 同构：修正要求明确「新名字不得含禁用词、要跨条目替换」，
    且重写后必有复查（diet_service 的重算 + 冲突复检），所以敢用快模型。
    """
    conflict_lines = []
    for conflict in conflicts:
        located = "/".join(part for part in (conflict.get("day_type"), conflict.get("meal")) if part)
        conflict_lines.append(f"- {conflict['food']}（{located or '位置未知'}）：{conflict['reason']}")
    forbidden = sorted({c["food"] for c in conflicts})
    sections = [
        "【必须修正的问题】\n" + "\n".join(conflict_lines),
        f"【禁用食物清单（重写后不得出现，也不得用同类变体绕过）】{'、'.join(forbidden)}",
        "【可用食物清单（替换只能从这里选）】\n" + _format_catalog(catalog),
        "【两日型热量与餐次目标（替换后仍须接近）】\n" + _format_targets(day_targets, meal_slots),
        "请在原计划上修改：只替换冲突食物并调整克数使餐次目标仍接近，其余餐次保持不变。"
        "输出与原结构相同的完整 JSON；foods 仍只写 name 与 grams。",
    ]
    return [
        {"role": "system", "content": DIET_WRITER_SYSTEM},
        {"role": "user", "content": "原计划：\n" + _dump_plan(plan) + "\n\n" + "\n\n".join(sections)},
    ]


def _dump_plan(plan: dict) -> str:
    import json
    return json.dumps(plan, ensure_ascii=False)