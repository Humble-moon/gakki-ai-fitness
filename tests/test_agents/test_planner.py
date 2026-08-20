import pytest
from src.agents.planner import PlannerAgent

pytestmark = pytest.mark.integration

@pytest.fixture
def planner():
    return PlannerAgent()

class TestPlannerAgent:
    def test_plan_includes_skill(self, planner):
        profile = {"height": 180, "weight": 80, "goal": "增肌",
                   "training_years": 1, "available_equipment": ["哑铃"]}
        result = planner.plan("我想增肌", profile)
        assert "skill" in result
        assert "skill_config" in result

    def test_plan_analysis_triggers_exercise_skill(self, planner):
        profile = {"height": 175, "weight": 70, "goal": "增肌",
                   "training_years": 0.5, "available_equipment": ["哑铃"]}
        result = planner.plan("深蹲时膝盖不舒服", profile)
        assert result["skill"] == "exercise_analysis"

    def test_plan_includes_subtasks(self, planner):
        profile = {"height": 180, "weight": 80, "goal": "增肌",
                   "training_years": 1, "available_equipment": ["哑铃"]}
        result = planner.plan("帮我设计增肌计划", profile)
        assert "subtasks" in result

    def test_plan_records_profile_goal_even_when_safety_route_overrides_skill(self, planner, monkeypatch):
        profile = {"goal": "减脂"}
        monkeypatch.setattr(planner.llm, "chat_with_json_mode", lambda _: {"skill": "fat_loss"})

        result = planner.plan("膝盖疼，想减脂", profile)

        assert result["skill"] == "exercise_analysis"
        assert result["requested_goal"] == "减脂"
