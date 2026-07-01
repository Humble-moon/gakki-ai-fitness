import pytest
from src.models.schemas import UserProfileInput, ExerciseItem, TrainingDay, TrainingPlanOutput

class TestUserProfileInput:
    def test_valid_profile(self):
        p = UserProfileInput(
            height=180, weight=80, training_years=1.5,
            goal="增肌", available_equipment=["哑铃", "杠铃"],
            days_per_week=4, injuries=["下背痛"]
        )
        assert p.height == 180
        assert p.goal == "增肌"

    def test_invalid_goal_raises(self):
        with pytest.raises(Exception):
            UserProfileInput(
                height=180, weight=80, training_years=1,
                goal="塑形", available_equipment=["哑铃"], days_per_week=3
            )

    def test_height_out_of_range_raises(self):
        with pytest.raises(Exception):
            UserProfileInput(
                height=50, weight=80, training_years=1,
                goal="增肌", available_equipment=["哑铃"], days_per_week=3
            )

class TestTrainingPlanOutput:
    def test_valid_plan(self):
        plan = TrainingPlanOutput(
            plan_id="abc-123", user_id=1, goal="增肌",
            weeks=4, sessions_per_week=4,
            days=[
                TrainingDay(day=1, focus="胸+三头", exercises=[
                    ExerciseItem(name="哑铃卧推", sets=4, reps="8-12", rest="90s")
                ])
            ],
            warnings=[]
        )
        assert plan.weeks == 4
        assert len(plan.days[0].exercises) == 1
