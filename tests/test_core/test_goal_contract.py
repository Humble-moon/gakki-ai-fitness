import pytest

from src.core.goal_contract import (
    GoalConsistencyError,
    plan_goal_issue,
    plan_goal_matches,
    validate_requested_goal,
)


@pytest.mark.parametrize("goal", ["增肌", "减脂"])
def test_validate_requested_goal_returns_supported_goal(goal):
    assert validate_requested_goal(goal) == goal


@pytest.mark.parametrize("goal", [None, 1, "", "塑形"])
def test_validate_requested_goal_rejects_invalid_values(goal):
    with pytest.raises(GoalConsistencyError) as exc:
        validate_requested_goal(goal)
    assert exc.value.code == "GOAL_CONSISTENCY_FAILED"


def test_plan_goal_issue_reports_machine_readable_mismatch_without_mutating_plan():
    plan = {"goal": "增肌"}

    issue = plan_goal_issue(plan, "减脂")

    assert issue == {
        "code": "PLAN_GOAL_MISMATCH",
        "issue": "训练计划目标与用户目标不一致",
        "expected_goal": "减脂",
        "actual_goal": "增肌",
    }
    assert plan == {"goal": "增肌"}


@pytest.mark.parametrize(
    ("plan", "expected_goal", "matches"),
    [
        ({"goal": "增肌"}, "增肌", True),
        ({"goal": "减脂"}, "增肌", False),
        ({}, "增肌", False),
        (None, "增肌", False),
    ],
)
def test_plan_goal_matches_requires_exact_supported_match(plan, expected_goal, matches):
    assert plan_goal_matches(plan, expected_goal) is matches
