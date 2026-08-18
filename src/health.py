from __future__ import annotations

from typing import Callable


def check_dependencies() -> dict[str, str]:
    """Return local dependency states without opening network connections."""
    checks = {"provider": "configured"}
    try:
        from src.config import LLM_CONFIGS  # type: ignore

        if not LLM_CONFIGS.get("default"):
            checks["provider"] = "down"
    except Exception:
        checks["provider"] = "down"
    return checks


def readiness_checks(
    dependency_checker: Callable[[], dict[str, str]] | None = None,
) -> tuple[dict[str, str], bool]:
    checker = dependency_checker if dependency_checker is not None else check_dependencies
    checks = checker()
    ready = bool(checks) and all(value in {"ok", "configured", "demo"} for value in checks.values())
    return checks, ready


__all__ = ["check_dependencies", "readiness_checks"]
