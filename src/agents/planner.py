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
    """规划器 Agent：LLM 语义选择 + 安全闸门校验。

    v4 升级 (2026-07-29):
      关键词层始终参与——不只在 LLM 失败时兜底。
      安全词命中 → 直接覆盖 LLM 结果，安全不被 LLM 的黑盒决策左右。
    """

    # 安全闸门关键词：无论 LLM 选了哪个 skill，命中即路由到 exercise_analysis
    SAFETY_OVERRIDE = [
        "间盘", "腰突", "半月板", "髌骨", "脱臼", "腱鞘炎", "网球肘",
        "肩峰撞击", "跟腱炎", "TFCC", "手术", "术后", "重建", "炎症",
        "撕裂感", "弹响", "咔咔响", "损伤", "恢复期", "骨折",
        "疼", "痛", "不舒服", "伤到",
    ]

    def __init__(self):
        self.llm = LLMProvider()
        self.skills = SkillRegistry()

    def plan(self, user_input: str, profile: dict,
             conv_context: str = "", plan_context: str = "") -> dict:
        """LLM 驱动的规划 + 安全闸门校验。

        流程:
            1. LLM 产出 skill + subtasks
            2. 关键词层始终运行 → 安全词命中则覆盖 LLM 结果
            3. LLM 返回无效 skill → 关键词匹配兜底
            4. 从 SkillRegistry 加载 skill 配置
        """
        skill_descriptions = self.skills.describe_all()
        messages = build_planner_messages(
            user_input, profile,
            skill_descriptions=skill_descriptions,
            conv_context=conv_context,
            plan_context=plan_context,
        )
        plan = self.llm.chat_with_json_mode(messages)

        llm_skill = plan.get("skill", "")
        keyword_skill = self.skills.match(user_input)

        # === 安全闸门：关键词检测到高危信号 → 无条件覆盖 LLM ===
        safety_hit = any(kw in user_input for kw in self.SAFETY_OVERRIDE)
        if safety_hit and llm_skill != "exercise_analysis":
            logger.info(
                f"Safety gate: LLM chose '{llm_skill}' but safety keywords "
                f"detected in user input, overriding to 'exercise_analysis'"
            )
            llm_skill = "exercise_analysis"

        # === 路由验证：LLM skill 有效性检查 ===
        skill = self.skills.get(llm_skill)
        if skill is None:
            logger.warning(
                f"LLM returned invalid skill '{llm_skill}', "
                f"falling back to keyword match '{keyword_skill}'"
            )
            llm_skill = keyword_skill
            skill = self.skills.get(llm_skill)
        elif llm_skill != keyword_skill and keyword_skill:
            # LLM 和关键词不一致但不是安全问题 → 用 LLM，记录分歧
            logger.debug(
                f"Route divergence: LLM='{llm_skill}', keyword='{keyword_skill}'"
                f" — using LLM"
            )

        # === 置信度标记 ===
        llm_conf = plan.get("confidence", plan.get("skill_confidence", None))
        if llm_conf is not None and llm_conf < 0.6:
            plan["low_confidence"] = True

        plan["skill"] = llm_skill
        plan["_keyword_route"] = keyword_skill  # 调试用
        plan["skill_config"] = {
            "retrieval_filters": skill.retrieval_filters,
            "plan_template": skill.plan_template,
        }
        return plan
