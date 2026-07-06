"""
================================================================================
文件角色：FactChecker（训练安全审查器）Prompt 模板
================================================================================
- 被调用者：Pipeline 中的第四步。Planner → Retriever → Writer → FactChecker
  → HITL 审核。FactChecker 是"最后的 AI 防线"，在交付用户前检查安全隐患。
- 调用者：编排引擎调用 build_fact_checker_messages() 构造 messages，
  然后通过 LLMProvider 发出。输出结果还会传递给 HITLReview 做二次判断。
- 项目角色：安全阀——AI 生成的训练计划如果包含不适合用户的动作（如伤病冲突、
  难度不匹配、训练量超标），在此环节拦截，标记 requires_human_review。
================================================================================
"""

# ---------------------------------------------------------------------------
# FACTCHECKER_SYSTEM: FactChecker Agent 的系统提示词
# ---------------------------------------------------------------------------
# 角色定位：训练安全审查专家，不生成内容，只审查已有内容。
# 输入来源：Writer 生成的完整训练计划 JSON + 用户画像（特别是伤病史和训练水平）。
# 输出用途：产出安全评估 JSON，包含是否安全、具体问题列表、置信度、
#           是否需要人工审核。
#
# 检查维度说明：
#   1. 难度匹配：初学者不应被安排大重量自由重量复合动作（如杠铃深蹲、杠铃硬拉），
#      因为自由重量对核心稳定性和技术熟练度要求高，初学者容易受伤。
#   2. 训练量合理性：单次训练超过 20 组会导致过度疲劳，每周每部位超过 25 组
#      超出肌肉恢复能力，属于过度训练风险。
#   3. 伤病风险：检查计划中的每个动作是否与用户伤病史冲突。
#   4. 器械约束：确认所有动作都需要用户实际拥有的器械。
#
# 关键字段说明（输出 JSON）：
#   - is_safe:               整体判断，true 表示计划安全
#   - issues:                问题列表，每项包含：
#       - exercise:          有问题的动作名称
#       - issue:             问题描述 + 建议替代方案
#       - severity:          "info"（提示）/ "warning"（警告）/ "danger"（危险）
#   - confidence:            0.0~1.0，模型对自己判断的置信度
#   - requires_human_review: true 表示需要转人工审核（由 HITL 模块处理）
# ---------------------------------------------------------------------------
FACTCHECKER_SYSTEM = """你是训练安全审查专家。校验生成的训练计划是否安全合理。

检查项：
1. 动作难度是否匹配用户水平（初学者不推荐大重量自由重量动作）
2. 训练量是否合理（单次最多 20 组，每周每部位最多 25 组）
3. 是否存在已知伤病风险动作
4. 器械约束是否满足

输出 JSON：
{
  "is_safe": true | false,
  "issues": [{"exercise": "杠铃深蹲", "issue": "用户有下背伤史，建议改为高脚杯深蹲", "severity": "warning"}],
  "confidence": 0.0-1.0,
  "requires_human_review": true | false
}
"""


def build_fact_checker_messages(plan: dict, profile: dict) -> list:
    """
    构造发送给 FactChecker Agent 的消息列表。

    参数：
        plan: dict    - Writer 生成的训练计划字典，包含 plan_name / goal /
                        days_per_week / days（动作列表）/ notes 等字段。
        profile: dict - 用户画像字典，FactChecker 主要关注其中的：
                        level（训练水平）、injuries（伤病史）、
                        equipment（可用器械）

    返回值：
        list          - OpenAI 格式的 messages 列表

    核心逻辑：
        将完整计划和用户画像一同提交给 LLM 审查。审查结果传给 HITLReview.check()
        做最终决策——低置信度或有 danger 级别问题时触发人工审核流程。
    """
    return [
        {"role": "system", "content": FACTCHECKER_SYSTEM},
        {"role": "user", "content": f"训练计划：{plan}\n用户画像：{profile}"}
    ]
