"""
===========================================================================
文件角色：规划器 Agent —— LLM 驱动的任务规划 + 技能选择
===========================================================================
- 被谁调用：Orchestrator 在流水线的第 2 步调用 PlannerAgent.plan()
- 调用谁：
    LLMProvider.chat_with_json_mode() → 一次 LLM 调用同时完成技能选择和任务拆解
    SkillRegistry.describe_all()      → 生成可用技能描述注入 prompt
    SkillRegistry.match() / get()     → 仅作 LLM 失败时的降级兜底
    build_planner_messages()          → 构造提示词
- 核心职责：
    1. 一次 LLM 调用完成两件事：选择技能 + 拆解子任务
    2. LLM 返回异常时，降级为关键词匹配兜底
===========================================================================
"""
import logging

from src.llm.provider import LLMProvider
from src.llm.prompts.planner import build_planner_messages
from src.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)


class PlannerAgent:
    """规划器 Agent：LLM 语义理解替代关键词匹配，一次调用完成技能选择+任务拆解。"""

    def __init__(self):
        self.llm = LLMProvider()
        self.skills = SkillRegistry()

    def plan(self, user_input: str, profile: dict,
             conv_context: str = "", plan_context: str = "") -> dict:
        """LLM 驱动的规划：技能选择 + 子任务拆解在同一个 LLM 调用中完成。

        流程：
            1. 将可用技能描述注入 prompt → LLM 产出 skill + subtasks
            2. 如果 LLM 返回的 skill 无效 → 降级为关键词匹配
            3. 从 SkillRegistry 加载该 skill 的检索过滤器和计划模板
        """
        skill_descriptions = self.skills.describe_all()
        messages = build_planner_messages(
            user_input, profile,
            skill_descriptions=skill_descriptions,
            conv_context=conv_context,
            plan_context=plan_context,
        )
        plan = self.llm.chat_with_json_mode(messages)

        # 提取 LLM 选择的 skill，无效则降级关键词匹配
        skill_name = plan.get("skill", "")
        skill = self.skills.get(skill_name)
        if skill is None:
            fallback_name = self.skills.match(user_input)
            logger.warning(
                f"LLM returned invalid skill '{skill_name}', "
                f"falling back to keyword match '{fallback_name}'"
            )
            skill_name = fallback_name
            skill = self.skills.get(skill_name)

        plan["skill"] = skill_name
        plan["skill_config"] = {
            "retrieval_filters": skill.retrieval_filters,
            "plan_template": skill.plan_template,
        }
        return plan
