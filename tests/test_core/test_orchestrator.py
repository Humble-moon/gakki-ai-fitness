import pytest
from src.core.orchestrator import Orchestrator
from src.models.schemas import UserProfileInput


@pytest.fixture
def orch():
    return Orchestrator()


class TestOrchestrator:
    def test_generate_plan_complete_flow(self, orch):
        profile = UserProfileInput(height=180, weight=80, training_years=1, goal="增肌", available_equipment=["哑铃", "杠铃"], days_per_week=4)
        result = orch.generate_plan(profile, "帮我设计增肌计划")
        assert "plan_id" in result
        assert "days" in result or "raw" in result or "notes" in result or "exercises" in result

    def test_analyze_exercise(self, orch):
        profile = UserProfileInput(height=175, weight=70, training_years=0.5, goal="增肌", available_equipment=["哑铃"], days_per_week=3)
        result = orch.analyze_exercise("哑铃卧推", "推的时候肩膀前侧有点疼", profile)
        assert "exercise_name" in result
        assert "issues_found" in result
        assert "suggestions" in result

    def test_semantic_cache_hit(self, orch):
        profile = UserProfileInput(height=180, weight=80, training_years=1, goal="增肌", available_equipment=["哑铃", "杠铃"], days_per_week=4)
        result1 = orch.generate_plan(profile, "增肌计划")
        result2 = orch.generate_plan(profile, "增肌计划")
        assert result1["plan_id"] == result2["plan_id"]


class TestStreamingContracts:
    def test_stream_aggregates_advice_and_writer_degraded_and_forwards_context(self):
        orch = Orchestrator()
        profile = UserProfileInput(height=180, weight=80, training_years=1, goal="增肌", available_equipment=["哑铃"], days_per_week=4)

        class FakeStream:
            def __init__(self, text, degraded):
                self.metadata = SimpleNamespace(degraded=degraded)
                self.text = text
            def __iter__(self):
                return iter([self.text])

        orch.writer.llm.chat_stream = lambda *args, **kwargs: FakeStream("建议", True)
        orch.conversation.add_turn = lambda *args: None
        orch.conversation.build_context_for_prompt = lambda *args: "历史上下文"
        orch.conversation.get_plan_state = lambda *args: "旧计划"
        orch.planner.plan = lambda *args, **kwargs: {"skill": "basic", "subtasks": [], "skill_config": {}}
        orch.retriever.retrieve = lambda plan: {"exercises": []}
        writer_kwargs = {}
        def fake_writer_stream(*args, **kwargs):
            writer_kwargs.update(kwargs)
            return iter([("chunk", '{"days": [{"exercises": []}]}'), ("done", {"days": [{"exercises": []}], "_degraded": False})])
        orch.writer.write_plan_stream = fake_writer_stream
        orch.fact_checker.check = lambda *args: {"is_safe": True, "issues": [], "confidence": 0.9}
        persisted = []
        orch._persist_if_safe = lambda *args, **kwargs: persisted.append(True)

        events = list(orch.generate_plan_stream(profile, "修改第二天", session_id="s1"))
        final = events[-1][1]
        assert final["_persistence_allowed"] is False
        assert not persisted
        assert writer_kwargs["plan_context"] == "旧计划"
        assert writer_kwargs["user_query"] == "修改第二天"
