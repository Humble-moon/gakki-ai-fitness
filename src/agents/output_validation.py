"""Fail-closed validation for LLM-produced outputs."""

from pydantic import ValidationError

from src.models.schemas import TrainingPlanOutput


class OutputValidationError(ValueError):
    """Raised when an LLM result does not satisfy its output schema."""


def validate_training_plan(result: dict) -> dict:
    """Validate and return a plan without silently filling missing fields."""
    if not isinstance(result, dict):
        raise OutputValidationError("training plan must be a JSON object")
    try:
        TrainingPlanOutput.model_validate(result)
    except ValidationError as exc:
        raise OutputValidationError("training plan schema validation failed") from exc
    return result
