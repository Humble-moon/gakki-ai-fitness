"""
================================================================================
文件角色：Writer（训练计划生成器）Prompt 模板
================================================================================
- 被调用者：在 Pipeline 中的第三步。Planner 拆任务 → Retriever 搜动作 →
  Writer 生成计划 → FactChecker 校验 → HITL 审核。
- 调用者：编排引擎调用 build_writer_messages() 构造 messages，
  通过 LLMProvider 发送，并要求 JSON 模式输出。
- 项目角色：核心产出模块——最终交付给用户的训练计划由这个 Prompt 模板驱动生成。
================================================================================
"""

# ---------------------------------------------------------------------------
# WRITER_SYSTEM: Writer Agent 的系统提示词
# ---------------------------------------------------------------------------
# 角色定位：训练计划编写专家，负责将检索到的动作编排成结构化的训练计划。
# 输入来源：检索结果（可用动作列表）+ 用户画像 + 训练目标（增肌/减脂）。
# 输出用途：生成一份 JSON 格式的训练计划，包含每天的动作用量。
#
# Prompt 内置了两套训练参数（增肌 vs 减脂），因为这是运动科学常识而非需要
# 检索的外部知识，直接写在 Prompt 中更稳定、不依赖检索质量。
#
# 关键字段说明（输出 JSON 结构）：
#   - plan_name:     计划名称，如 "4天哑铃增肌计划"
#   - goal:          目标类型标签 "增肌" 或 "减脂"
#   - days_per_week: 每周训练天数
#   - days:          数组，每天的内容：
#       - day:       第几天（从 1 开始）
#       - focus:     当天训练重点，如 "推类动作（胸+肩+三头）"
#       - exercises: 当天动作列表：
#           - name:  动作名称（必须来自检索结果，不得编造）
#           - sets:  组数
#           - reps:  次数范围，如 "8-12"
#           - rest:  组间休息时间，如 "60s"
#           - notes: 动作要点提示（可选）
#   - notes:         额外的全局说明（可选）
#
# 安全底线：每个动作必须来自检索结果，不得编造。
# ---------------------------------------------------------------------------
WRITER_SYSTEM = """你是训练计划编写专家。根据检索到的动作库和用户情况，生成结构化训练计划。

增肌计划参数：
- Rep Range: 6-12
- 组间休息: 60-90s
- 每部位每周 10-20 组

减脂计划参数：
- Rep Range: 12-15
- 组间休息: 30-60s
- 可加入超级组/HIIT

【强制约束】你必须严格按照用户画像中指定的 days_per_week 生成对应数量的训练日。
用户说几天就几天，不要自作主张增减。如果用户要求 5 天，days 数组里必须恰好有 5 个元素。

输出 JSON 格式（严格遵守键名）：
{
  "plan_name": "计划名称",
  "goal": "增肌或减脂",
  "days_per_week": <按用户画像中的值>,
  "days": [
    {
      "day": 1,
      "focus": "训练重点",
      "exercises": [
        {"name": "动作名称", "sets": 3, "reps": "8-12", "rest": "60s", "notes": "要点"}
      ]
    }
  ],
  "notes": "其他说明"
}
每个动作必须来自检索结果，不得编造。
"""


def build_writer_messages(retrieved_exercises: list, profile: dict, goal: str) -> list:
    """
    构造发送给 Writer Agent 的消息列表。

    参数：
        retrieved_exercises: list  - 检索系统返回的动作列表，每个动作为一个 dict，
                                    包含 name/equipment/muscles/difficulty 等字段。
                                    这是 Writer 的"素材库"，不能超出此范围编造动作。
        profile: dict              - 用户画像字典，包含身高/体重/训练水平/伤病史/
                                    可用器械等信息，Writer 据此调整训练参数。
        goal: str                  - 训练目标："增肌" / "减脂"，决定使用哪套训练参数。

    返回值：
        list                       - OpenAI 格式的 messages 列表
    """
    days = profile.get("days_per_week", 4)
    injuries = profile.get("injuries", [])
    equipment = profile.get("available_equipment", [])

    # 把关键约束提到最前面，LLM 不会漏掉
    constraints = [
        f"训练日数量：{days} 天（必须恰好生成 {days} 个训练日，每个 day 从 1 到 {days}）",
    ]
    if injuries:
        constraints.append(f"伤病限制：用户有以下伤病 {injuries}，避免涉及这些部位的动作")
    if equipment:
        constraints.append(f"器械限制：只能用 {equipment}，不要推荐需要其他器械的动作")

    user_msg = "\n".join(constraints) + f"\n\n目标：{goal}\n用户画像：{profile}\n可用动作：{retrieved_exercises}"

    return [
        {"role": "system", "content": WRITER_SYSTEM},
        {"role": "user", "content": user_msg}
    ]
