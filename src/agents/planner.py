from src.llm.provider import LLMProvider
from src.llm.prompts.planner import build_planner_messages
from src.skills.registry import SkillRegistry

class PlannerAgent:
    def __init__(self):
        self.llm = LLMProvider()
        self.skills = SkillRegistry()

    def plan(self, user_input: str, profile: dict) -> dict:
        skill_name = self.skills.match(user_input)
        skill = self.skills.get(skill_name)
        messages = build_planner_messages(user_input, profile)
        plan = self.llm.chat_with_json_mode(messages)
        plan["skill"] = skill_name
        plan["skill_config"] = {
            "retrieval_filters": skill.retrieval_filters,
            "plan_template": skill.plan_template
        }
        return plan
