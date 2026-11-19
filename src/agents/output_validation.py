"""Fail-closed validation for LLM-produced outputs."""

from pydantic import ValidationError

from src.models.schemas import DietPlanOutput, TrainingPlanOutput


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


def validate_diet_plan(result: dict) -> dict:
    """饮食计划同构校验：结构不合格一律抛错，不静默补字段。

    chat_with_json_mode 解析失败时返回 ``{"raw": ...}``——那不是计划，
    在这里会因缺 plan_name/days 直接抛 OutputValidationError，由上层决定
    重试或交付 generation_failed，绝不把散文当计划交付。
    """
    if not isinstance(result, dict):
        raise OutputValidationError("diet plan must be a JSON object")
    try:
        DietPlanOutput.model_validate(result)
    except ValidationError as exc:
        raise OutputValidationError("diet plan schema validation failed") from exc
    return result
