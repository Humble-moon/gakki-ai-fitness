from dataclasses import dataclass, field

@dataclass
class Skill:
    name: str
    description: str
    triggers: list
    retrieval_filters: dict = field(default_factory=dict)
    plan_template: str = ""

class SkillRegistry:
    def __init__(self):
        self.skills: dict[str, Skill] = {}
        self._load_builtin()

    def _load_builtin(self):
        self.register(Skill(
            name="muscle_building",
            description="增肌训练计划生成",
            triggers=["增肌", "增重", "变大", "维度", "增肌塑形"],
            retrieval_filters={"rep_range": "6-12", "rest": "60-90s"},
            plan_template="四分化/五分化"
        ))
        self.register(Skill(
            name="fat_loss",
            description="减脂训练计划生成",
            triggers=["减脂", "减重", "瘦", "刷脂", "塑形"],
            retrieval_filters={"rep_range": "12-15", "rest": "30-60s"},
            plan_template="上下肢分化/全身"
        ))
        self.register(Skill(
            name="exercise_analysis",
            description="动作质量分析",
            triggers=["动作", "姿势", "感觉", "疼", "不舒服", "是不是"],
            retrieval_filters={},
            plan_template="分析报告"
        ))

    def register(self, skill: Skill):
        self.skills[skill.name] = skill

    def match(self, user_input: str) -> str | None:
        for name, skill in self.skills.items():
            for trigger in skill.triggers:
                if trigger in user_input:
                    return name
        return "muscle_building"

    def get(self, name: str) -> Skill | None:
        return self.skills.get(name)
