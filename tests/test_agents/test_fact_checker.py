from src.agents.fact_checker import FactCheckerAgent

def test_fact_checker_returns_structured_result():
    checker = FactCheckerAgent()
    plan = {
        "days": [{"day": 1, "focus": "胸", "exercises": [
            {"name": "杠铃卧推", "sets": 5, "reps": "3-5", "rest": "120s"}
        ]}]
    }
    profile = {"training_years": 0.3, "injuries": [], "goal": "增肌",
               "available_equipment": ["哑铃"]}
    result = checker.check(plan, profile)
    assert "is_safe" in result
    assert "issues" in result
    assert "confidence" in result
