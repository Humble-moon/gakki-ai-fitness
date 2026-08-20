"""Deterministic contract for canonical training-plan goals."""

SUPPORTED_GOALS = frozenset({"增肌", "减脂"})


class GoalConsistencyError(ValueError):
    """Raised when a requested or delivered training goal is inconsistent."""

    code = "GOAL_CONSISTENCY_FAILED"


def validate_requested_goal(goal: object) -> str:
    """Return a supported goal or fail closed with a stable error code."""
    if not isinstance(goal, str) or goal not in SUPPORTED_GOALS:
        raise GoalConsistencyError("训练目标必须是增肌或减脂")
    return goal


def plan_goal_issue(plan: dict, expected_goal: str) -> dict | None:
    """Describe an explicit plan goal mismatch without altering model output."""
    expected_goal = validate_requested_goal(expected_goal)
    actual_goal = plan.get("goal") if isinstance(plan, dict) else None
    if actual_goal == expected_goal:
        return None
    return {
        "code": "PLAN_GOAL_MISMATCH",
        "issue": "训练计划目标与用户目标不一致",
        "expected_goal": expected_goal,
        "actual_goal": actual_goal,
    }


def plan_goal_matches(plan: dict, expected_goal: str) -> bool:
    """Return true only for an exact canonical goal match."""
    return plan_goal_issue(plan, expected_goal) is None
