import pytest
from types import SimpleNamespace
from src.agents.writer import WriterAgent


class TestWriterAgent:
    @pytest.mark.integration
    def test_write_analysis_returns_structured_result(self):
        writer = WriterAgent()
        retrieved = {"exercises": []}
        profile = {"training_years": 1, "goal": "增肌", "id": 1}
        result = writer.write_analysis("哑铃卧推", "肩膀疼", retrieved, profile)
        assert result["exercise_name"] == "哑铃卧推"
        assert "issues_found" in result
        assert "severity" in result
        assert "suggestions" in result

    def test_write_plan_stream_forwards_degraded_metadata_and_context(self, monkeypatch):
        writer = WriterAgent()
        captured = {}

        class FakeStream:
            metadata = SimpleNamespace(degraded=True)

            def __iter__(self):
                return iter(['{"days": []}'])

        def fake_chat_stream(messages, **kwargs):
            captured["content"] = messages[-1]["content"]
            return FakeStream()

        monkeypatch.setattr(writer.llm, "chat_stream", fake_chat_stream)
        events = list(writer.write_plan_stream(
            {"exercises": []}, {"id": 7, "goal": "增肌"}, {},
            plan_context="旧计划上下文", user_query="把第二天改成哑铃",
        ))
        assert events[-1][1]["_degraded"] is True
        assert "旧计划上下文" in captured["content"]
        assert "把第二天改成哑铃" in captured["content"]

    def test_rewrite_prompt_requires_profile_goal_without_overwriting_model_goal(self, monkeypatch):
        writer = WriterAgent()
        captured = {}

        def fake_write(messages, **kwargs):
            captured["content"] = messages[-1]["content"]
            return {"goal": "增肌", "days": []}

        monkeypatch.setattr(writer.llm, "chat_with_json_mode", fake_write)
        result = writer.rewrite_plan(
            {"plan_id": "p1", "goal": "减脂", "weeks": 4}, [], {"exercises": []},
            {"id": 1, "goal": "减脂", "days_per_week": 3},
        )

        assert "goal 必须严格等于用户的 canonical goal：减脂" in captured["content"]
        assert result["goal"] == "增肌"

    def test_write_analysis_stream_forwards_degraded_metadata(self, monkeypatch):
        writer = WriterAgent()

        class FakeStream:
            metadata = SimpleNamespace(degraded=True)

            def __iter__(self):
                return iter(['{"issues_found": []}'])

        monkeypatch.setattr(writer.llm, "chat_stream", lambda *args, **kwargs: FakeStream())
        events = list(writer.write_analysis_stream("深蹲", "膝盖不适", {}, {}))
        assert events[-1][1]["_degraded"] is True
