"""
===========================================================================
文件角色：规划器 Agent —— 将用户输入拆解为可执行的子任务并匹配训练技能模板
===========================================================================
- 被谁调用：Orchestrator 在流水线的第 2 步调用 PlannerAgent.plan()
- 调用谁：
    LLMProvider.chat_with_json_mode() → 调用 LLM 进行任务拆解
    SkillRegistry.match() / get()    → 根据用户输入匹配最佳训练技能模板
    build_planner_messages()         → 构造发送给 LLM 的提示词（来自 prompts 模块）
- 核心职责：
    1. 根据用户输入文本，从技能注册表中匹配最合适的训练模板（如"增肌"、"减脂"、"康复"）
    2. 调用 LLM 将用户需求拆解为多个子任务（subtasks）
    3. 将技能模板的检索过滤器和计划模板附加到规划结果中，供下游 Agent 使用
===========================================================================
"""

from src.llm.provider import LLMProvider
from src.llm.prompts.planner import build_planner_messages
from src.skills.registry import SkillRegistry


class PlannerAgent:
    """规划器 Agent：在 Orchestrator 流水线的第 2 步被调用。
    职责：将用户的自然语言需求转换为结构化的子任务列表 + 匹配的训练技能模板。
    输入：用户原始文本 + 画像字典
    输出：包含 subtasks / skill / skill_config 的规划字典"""

    def __init__(self):
        self.llm = LLMProvider()
        self.skills = SkillRegistry()

    def plan(self, user_input: str, profile: dict) -> dict:
        """根据用户输入和画像生成任务规划。

        输入：
            user_input: str — 用户的自由文本输入，如 "为增肌目标生成训练计划"
            profile: dict — 用户画像字典（身高/体重/训练年限/伤病等）
        输出：
            dict — 包含以下字段：
                - "subtasks": list[str] — LLM 拆解出的子任务列表，如 ["胸部训练", "背部训练"]
                - "skill": str — 匹配到的技能模板名称，如 "muscle_building"
                - "skill_config": dict — 该技能的配置（检索过滤器 + 计划模板）

        核心逻辑：
            1. SkillRegistry.match(user_input): 对用户输入做关键词/语义匹配，
               找到最合适的训练模板（如检测到"增肌"→muscle_building，"减脂"→fat_loss）
            2. SkillRegistry.get(skill_name): 获取模板的完整配置，
               包含检索过滤器（如 target_muscle, equipment）和训练计划模板结构
            3. build_planner_messages: 构造 LLM 提示词（含 system prompt + 用户信息）
            4. chat_with_json_mode: 强制 LLM 返回 JSON，保证下游可解析
            5. 将匹配到的 skill 和 skill_config 附加到 LLM 产出的 plan 中，
               这样 Retriever 和 Writer 就能按照该技能的规范进行检索和生成
        """
        # 1. 技能匹配：根据用户输入关键词自动识别训练目标类型
        skill_name = self.skills.match(user_input)
        skill = self.skills.get(skill_name)
        # 2. 构建 LLM 提示词（包含 system prompt + 用户画像 + 需求描述）
        messages = build_planner_messages(user_input, profile)
        # 3. 调用 LLM 拆解任务，chat_with_json_mode 强制返回合法 JSON
        plan = self.llm.chat_with_json_mode(messages)
        # 4. 将技能模板信息注入到规划结果中，供下游使用
        plan["skill"] = skill_name
        plan["skill_config"] = {
            "retrieval_filters": skill.retrieval_filters,
            "plan_template": skill.plan_template
        }
        return plan
