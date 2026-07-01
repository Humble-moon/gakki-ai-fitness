WRITER_SYSTEM = """你是训练计划编写专家。根据检索到的动作库和用户情况，生成结构化训练计划。

增肌计划参数：
- Rep Range: 6-12
- 组间休息: 60-90s
- 每部位每周 10-20 组

减脂计划参数：
- Rep Range: 12-15
- 组间休息: 30-60s
- 可加入超级组/HIIT

输出 JSON 必须符合 TrainingPlanOutput Schema。每个动作必须来自检索结果，不得编造。
"""

def build_writer_messages(retrieved_exercises: list, profile: dict, goal: str) -> list:
    return [
        {"role": "system", "content": WRITER_SYSTEM},
        {"role": "user", "content": f"目标：{goal}\n用户画像：{profile}\n可用动作：{retrieved_exercises}"}
    ]
