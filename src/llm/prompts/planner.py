"""
================================================================================
文件角色：Planner（任务编排器）Prompt 模板
================================================================================
- 被调用者：在 AI 健身助手的 Pipeline 中，Planner 是整个流程的第一步。
  用户输入 → Planner 分解任务 → Retriever 检索动作 → Writer 生成计划
  → FactChecker 校验安全 → HITL 人工审核。
- 调用者：编排引擎（orchestrator）调用 build_planner_messages() 构造
  messages 列表，然后通过 LLMProvider 发给 LLM。
- 本文件的 Prompt 要求 LLM 输出结构化 JSON，作为后续所有步骤的"任务指令"。
================================================================================
"""

# ---------------------------------------------------------------------------
# PLANNER_SYSTEM: Planner Agent 的系统提示词
# ---------------------------------------------------------------------------
# 角色定位：健身训练计划编排专家，负责"想清楚要做什么"而不是"亲自执行"。
# 输入来源：用户的身体数据（身高、体重、伤病史等）+ 训练目标（增肌/减脂）。
# 输出用途：产出一个 JSON 指令，交给后续的 Retriever / Writer 执行。
# 关键字段说明：
#   - subtasks: 将用户需求拆解为多个子任务，每个子任务对应一个检索维度
#   - retrieval_strategy: 指导 Retriever 使用哪种检索策略（向量/关键词/图谱/全量）
#   - output_format: 最终输出类型的标签（增肌计划/减脂计划/动作分析）
#   - constraints: 硬约束条件，如"仅哑铃""排除肩伤动作"，Writer 必须遵守
# ---------------------------------------------------------------------------
PLANNER_SYSTEM = """你是健身训练计划编排专家。根据用户的身体数据、训练目标和可用器械，分解任务并决定需要检索哪些信息。

输出 JSON 格式：
{
  "subtasks": ["检索推类动作", "检索拉类动作", "检索腿部动作"],
  "retrieval_strategy": "vector" | "keyword" | "graph" | "all",
  "output_format": "增肌计划" | "减脂计划" | "动作分析",
  "constraints": ["仅哑铃动作", "排除肩伤风险动作"]
}
"""


def build_planner_messages(user_input: str, profile: dict,
                          conv_context: str = "", plan_context: str = "") -> list:
    """
    构造发送给 Planner Agent 的消息列表。

    参数：
        user_input: str  - 用户的原始输入，例如 "帮我设计一个增肌计划，只有哑铃"
        profile: dict     - 用户画像字典，通常包含：
                            身高(height)、体重(weight)、训练水平(level)、
                            伤病史(injuries)、可用器械(equipment)、目标(goal)等
        conv_context: str - 多轮对话历史上下文（可选）
        plan_context: str - 上一轮训练计划摘要（可选，用于"修改计划"场景）

    返回值：
        list             - OpenAI 格式的 messages 列表，可直接传给 LLMProvider.chat()

    核心逻辑：
        将 user_input 和 profile 拼接成一条 user 消息，让 LLM 在 system prompt
        的指导下完成"任务拆解"工作。profile 以 Python dict 的字符串形式直接
        拼接，因为 LLM 能很好地理解这种格式。
        如果提供了多轮对话上下文和计划摘要，注入到 user 消息中帮助 LLM
        理解用户的修改意图。
    """
    user_msg = f"用户信息：{profile}\n用户请求：{user_input}"
    if conv_context:
        user_msg = f"{conv_context}\n\n{user_msg}"
    if plan_context:
        user_msg += f"\n\n【当前训练计划（用户可能要修改它）】\n{plan_context}"
    return [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": user_msg}
    ]
