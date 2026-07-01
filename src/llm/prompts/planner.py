PLANNER_SYSTEM = """你是健身训练计划编排专家。根据用户的身体数据、训练目标和可用器械，分解任务并决定需要检索哪些信息。

输出 JSON 格式：
{
  "subtasks": ["检索推类动作", "检索拉类动作", "检索腿部动作"],
  "retrieval_strategy": "vector" | "keyword" | "graph" | "all",
  "output_format": "增肌计划" | "减脂计划" | "动作分析",
  "constraints": ["仅哑铃动作", "排除肩伤风险动作"]
}
"""

def build_planner_messages(user_input: str, profile: dict) -> list:
    return [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": f"用户信息：{profile}\n用户请求：{user_input}"}
    ]
