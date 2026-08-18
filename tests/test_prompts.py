import pytest
from src.llm.prompts.planner import build_planner_messages
from src.llm.prompts.fact_checker import build_fact_checker_messages
from src.llm.prompts.writer import build_writer_messages

def test_planner_prompt_includes_profile():
    msgs = build_planner_messages("想增肌", {"height": 180, "goal": "增肌"})
    assert "增肌" in msgs[1]["content"]
    assert msgs[0]["role"] == "system"

def test_factchecker_prompt_includes_plan():
    plan = {"days": [{"day": 1, "exercises": []}]}
    msgs = build_fact_checker_messages(plan, {"injuries": ["下背痛"]})
    assert "下背痛" in msgs[1]["content"]


def test_writer_prompt_matches_training_plan_output_metadata():
    msgs = build_writer_messages([], {"days_per_week": 3}, "增肌")
    system_prompt = msgs[0]["content"]

    assert '"weeks"' in system_prompt
    assert '"sessions_per_week"' in system_prompt
    assert '"days_per_week"' not in system_prompt
