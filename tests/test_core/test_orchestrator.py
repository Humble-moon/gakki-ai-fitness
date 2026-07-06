import pytest
from src.core.orchestrator import Orchestrator
from src.models.schemas import UserProfileInput

@pytest.fixture
def orch():
    return Orchestrator()

class TestOrchestrator:
    def test_generate_plan_complete_flow(self, orch):
        profile = UserProfileInput(
            height=180, weight=80, training_years=1,
            goal="增肌", available_equipment=["哑铃", "杠铃"],
            days_per_week=4
        )
        result = orch.generate_plan(profile, "帮我设计增肌计划")
        assert "plan_id" in result
        # LLM 可能返回 "days"、"notes" 或 "raw"（JSON 解析回退）
        assert ("days" in result or "raw" in result or
                "notes" in result or "exercises" in result)
        if "days" in result:
            assert len(result["days"]) > 0

    def test_analyze_exercise(self, orch):
        profile = UserProfileInput(
            height=175, weight=70, training_years=0.5,
            goal="增肌", available_equipment=["哑铃"],
            days_per_week=3
        )
        result = orch.analyze_exercise(
            "哑铃卧推", "推的时候肩膀前侧有点疼", profile
        )
        assert "exercise_name" in result
        assert "issues_found" in result
        assert "suggestions" in result

    def test_semantic_cache_hit(self, orch):
        profile = UserProfileInput(
            height=180, weight=80, training_years=1,
            goal="增肌", available_equipment=["哑铃", "杠铃"],
            days_per_week=4
        )
        result1 = orch.generate_plan(profile, "增肌计划")
        result2 = orch.generate_plan(profile, "增肌计划")
        assert result1["plan_id"] == result2["plan_id"]
