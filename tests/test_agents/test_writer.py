import pytest
from src.agents.writer import WriterAgent

class TestWriterAgent:
    def test_write_analysis_returns_structured_result(self):
        writer = WriterAgent()
        retrieved = {"exercises": []}
        profile = {"training_years": 1, "goal": "增肌", "id": 1}
        result = writer.write_analysis("哑铃卧推", "肩膀疼", retrieved, profile)
        assert "exercise_name" in result
        assert result["exercise_name"] == "哑铃卧推"
        assert "issues_found" in result
        assert "severity" in result
        assert "suggestions" in result
