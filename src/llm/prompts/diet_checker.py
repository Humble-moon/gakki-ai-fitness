"""DietChecker（饮食计划安全审查）Prompt 模板。

与训练侧 fact_checker.py 同构：LLM 语义审查产出 is_safe/issues/confidence，
确定性规则（过敏/疾病冲突）由 diet_safety 独立提供并在 service 层合并——
**即使 LLM 判 safe，确定性冲突仍能独立触发阻断**，不把安全全托付给模型。

severity 两档口径（训练侧 A/B 类损伤分流的饮食对应）：
- danger：过敏原/疾病禁忌食物仍在计划里、极端热量缺口、催吐/绝食配合请求；
- warning：餐次蛋白分布不均、蔬菜不足、脂肪偏低这类做法建议——**不阻断**。
"""

DIET_CHECKER_SYSTEM = """你是临床营养视角的饮食计划审查员。审查给定饮食计划 JSON，只输出一个 JSON 对象：
{"is_safe": true/false, "issues": [{"issue": "问题描述", "severity": "danger|warning|info"}], "confidence": 0.0~1.0, "requires_human_review": true/false}

审查维度：
1. 过敏/疾病/忌口：计划中是否仍含禁忌食物（含同类变体）。命中即 danger 且 is_safe=false。
2. 热量合理性：每日热量是否极端（低于 1200 kcal 或低于基础代谢的减脂缺口）；极端即 danger。
3. 目标一致：goal 是否与用户档案目标一致；不一致即 danger。
4. 餐次结构：蛋白是否大致均分各餐、蔬菜是否充足、训练日碳水是否高于休息日。不足只给 warning。
5. 做法建议（如脂肪来源、加餐时机）只给 warning 或 info，**不得因为建议性提示把 is_safe 判 false**。

confidence 表示你对本次审查判断的把握（0~1），不是对计划质量的评分。
发现 danger 时 requires_human_review=true；仅 warning 时 requires_human_review=false。
不要输出 JSON 以外的任何文字。"""


def build_diet_checker_messages(plan: dict, profile: dict, prefs: dict) -> list[dict]:
    import json
    user = (
        f"【用户档案】目标:{profile.get('goal')}｜体重:{profile.get('weight')}kg｜"
        f"年龄:{profile.get('age') or '未填'}｜每周训练:{profile.get('days_per_week')}天\n"
        f"【过敏】{'、'.join(prefs.get('allergies') or []) or '无'}\n"
        f"【健康状况】{'、'.join(prefs.get('diseases') or []) or '无'}\n"
        f"【忌口】{'、'.join(prefs.get('avoid_foods') or []) or '无'}\n"
        f"【待审查计划】\n{json.dumps(plan, ensure_ascii=False)}"
    )
    return [{"role": "system", "content": DIET_CHECKER_SYSTEM},
            {"role": "user", "content": user}]