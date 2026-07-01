import uuid
from src.llm.provider import LLMProvider
from src.llm.prompts.writer import build_writer_messages

class WriterAgent:
    def __init__(self):
        self.llm = LLMProvider()

    def write_plan(self, retrieved: dict, profile: dict, plan_config: dict) -> dict:
        goal = profile.get("goal", "增肌")
        messages = build_writer_messages(
            retrieved.get("exercises", []), profile, goal
        )
        plan_json = self.llm.chat_with_json_mode(messages)
        plan_json["plan_id"] = str(uuid.uuid4())[:8]
        plan_json["user_id"] = profile.get("id", 0)
        return plan_json

    def write_analysis(self, exercise_name: str, user_desc: str,
                       retrieved: dict, profile: dict) -> dict:
        prompt = f"""分析动作：{exercise_name}
用户描述：{user_desc}
用户水平：{profile.get('training_years', 1)}年经验
标准动作规范：{retrieved}

输出 JSON：
{{
  "exercise_name": "{exercise_name}",
  "issues_found": ["问题1", "问题2"],
  "severity": "安全" | "注意" | "警告",
  "suggestions": ["改进1", "改进2"],
  "confidence": 0.0-1.0
}}"""
        result = self.llm.chat_with_json_mode([{"role": "user", "content": prompt}])
        result["exercise_name"] = exercise_name
        result.setdefault("issues_found", [])
        result.setdefault("severity", "安全")
        result.setdefault("suggestions", [])
        result.setdefault("confidence", 0.5)
        return result
