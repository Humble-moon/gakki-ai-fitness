"""Deterministic plan-finalization helpers shared by every orchestration backend.

This module is the single source of truth for the terminal-state logic of the
training-plan pipeline: cache validation, goal-contract enforcement, schema
validation, persistence gating, and review-payload construction.

It was extracted verbatim from ``Orchestrator`` so that both the legacy
synchronous orchestrator and the LangGraph-based graph pipeline finalize plans
through the *same* deterministic code path (required for a fair comparison in
the thesis). The functions are pure: every dependency is passed explicitly, so
they can be unit-tested with simple stand-ins and reused without constructing a
full ``Orchestrator``.
"""

from src.agents.output_validation import OutputValidationError, validate_training_plan
from src.core.goal_contract import (
    plan_goal_issue,
    plan_goal_matches,
    validate_requested_goal,
)

# Prohibited actions surfaced whenever a plan is held for human review.
REVIEW_PROHIBITED_ACTIONS = [
    "不要开始训练计划",
    "不要加大训练强度",
    "不要自行替换或增加动作",
]


def safe_cached_result(result: dict | None, expected_goal: str | None = None) -> dict | None:
    """Accept only cache entries that already passed the final gate and goal contract."""
    if not isinstance(result, dict) or result.get("_persistence_allowed") is not True:
        return None
    if result.get("requires_review") is not False or result.get("warnings"):
        return None
    if expected_goal is not None and not plan_goal_matches(result, expected_goal):
        return None
    try:
        validate_training_plan(result)
    except OutputValidationError:
        return None
    return result


def check_with_goal_issue(result: dict, check: dict, expected_goal: str) -> dict:
    """Add the deterministic goal issue to a FactChecker result when needed."""
    merged = dict(check) if isinstance(check, dict) else {}
    issues = list(merged.get("issues") or [])
    issue = plan_goal_issue(result, expected_goal)
    if issue:
        issues.append(issue)
        merged["is_safe"] = False
    merged["issues"] = issues
    return merged


def normalize_plan(result: dict, *, profile: dict | None = None,
                   plan_config: dict | None = None) -> dict:
    """Normalize model keys and add only deterministic request metadata."""
    if not isinstance(result, dict):
        return {}
    profile = profile if isinstance(profile, dict) else {}
    plan_config = plan_config if isinstance(plan_config, dict) else {}
    for key in ("weekly_plan", "weekly_schedule", "days", "schedule", "plan"):
        if key in result:
            result["days"] = result.pop(key)
            break

    sessions_per_week = profile.get("days_per_week")
    if sessions_per_week is None:
        sessions_per_week = plan_config.get("sessions_per_week")
    if sessions_per_week is not None:
        result["sessions_per_week"] = sessions_per_week
    elif "sessions_per_week" not in result and "days_per_week" in result:
        result["sessions_per_week"] = result["days_per_week"]

    configured_weeks = plan_config.get("weeks")
    if configured_weeks is not None:
        result["weeks"] = configured_weeks
    else:
        result.setdefault("weeks", None)

    result.pop("days_per_week", None)
    for day in result.get("days", []):
        for ex in day.get("exercises", []):
            if "rest_seconds" in ex and "rest" not in ex:
                ex["rest"] = f"{ex.pop('rest_seconds')}s"
            if "exercise" in ex and "name" not in ex:
                ex["name"] = ex.pop("exercise")
            if "movement" in ex and "name" not in ex:
                ex["name"] = ex.pop("movement")
    return result


def finalize_result(result: dict, checks: list[dict], rewrite_count: int,
                    *, provider_degraded: bool = False,
                    expected_goal: str | None = None) -> dict:
    """Normalize one terminal state and decide whether persistence is allowed."""
    result = dict(result) if isinstance(result, dict) else {}
    checks = checks if isinstance(checks, list) else []
    if expected_goal is not None:
        expected_goal = validate_requested_goal(expected_goal)
        checks = [check_with_goal_issue(result, check, expected_goal) for check in checks]
    final = checks[-1] if checks and isinstance(checks[-1], dict) else {}
    active_issues = final.get("issues") if isinstance(final.get("issues"), list) else []
    active_issue_values = {
        issue.get("issue", str(issue)) if isinstance(issue, dict) else str(issue)
        for issue in active_issues
    }
    resolved_issues = []
    seen_resolved = set()
    for check in checks[:-1]:
        if not isinstance(check, dict):
            continue
        for issue in check.get("issues") or []:
            value = issue.get("issue", str(issue)) if isinstance(issue, dict) else str(issue)
            if value not in active_issue_values and value not in seen_resolved:
                seen_resolved.add(value)
                resolved_issues.append(issue)
    warnings = [
        issue.get("issue", str(issue)) if isinstance(issue, dict) else str(issue)
        for issue in active_issues
    ]
    try:
        validate_training_plan(result)
        schema_valid = True
    except OutputValidationError:
        schema_valid = False
    is_safe = final.get("is_safe") is True
    issues_empty = final.get("issues") == []
    requires_review = any(isinstance(c, dict) and c.get("requires_human_review") is True for c in checks)
    confidence = final.get("confidence")
    confidence_valid = isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
    result.update({"warnings": warnings, "active_issues": active_issues,
                   "resolved_issues": resolved_issues,
                   "requires_review": requires_review or not checks or not is_safe,
                   "review_reason": final.get("review_reason", ""),
                   "review_severity": final.get("review_severity", ""),
                   "review_suggestions": list(final.get("review_suggestions") or []),
                   "confidence": confidence if confidence_valid else 0,
                   "rewrite_count": rewrite_count})
    goal_matches = expected_goal is None or plan_goal_matches(result, expected_goal)
    result["_persistence_allowed"] = bool(schema_valid and is_safe and issues_empty and not requires_review
        and confidence_valid and goal_matches and not provider_degraded and not result.get("_degraded", False))
    return result


def create_review_artifact(review_store, profile: dict, query: str, result: dict):
    """Create and store a pending human-review artifact for an unsafe plan.

    The artifact's issue list combines the final check's active issues with the
    deterministic rule-engine/semantic-match findings (``review_suggestions``),
    so the reviewer sees *why* the plan was held even when the rewrite loop has
    already resolved every LLM issue. Severity prefers the merged deterministic
    verdict (``review_severity``) over per-issue severities.
    """
    issues = [dict(issue) if isinstance(issue, dict) else {"issue": str(issue)}
              for issue in (result.get("active_issues") or [])]
    seen = {issue.get("issue", "") for issue in issues}
    for suggestion in result.get("review_suggestions") or []:
        item = dict(suggestion) if isinstance(suggestion, dict) else {"issue": str(suggestion)}
        if not item.get("issue") or item["issue"] in seen:
            continue
        seen.add(item["issue"])
        item.setdefault("severity", result.get("review_severity") or "warning")
        issues.append(item)
    severity = result.get("review_severity") or next(
        (issue.get("severity") for issue in issues if issue.get("severity")),
        "warning",
    )
    return review_store.create(
        profile_summary={key: profile.get(key) for key in ("goal", "injuries", "training_years") if key in profile},
        query=query,
        issues=issues,
        severity=severity,
        prohibited_actions=REVIEW_PROHIBITED_ACTIONS,
    )


def build_review_pending_payload(result: dict, artifact, thread_id: str | None = None) -> dict:
    """Build the ``review_pending`` delivery payload for a held plan."""
    final_check = result.get("active_issues") or []
    reason = result.get("review_reason") or (
        final_check[0].get("issue") if final_check and isinstance(final_check[0], dict) else "安全检查需要人工确认"
    )
    payload = {
        "delivery_status": "review_pending",
        "requires_review": True,
        "review": {
            "review_id": artifact.review_id,
            "status": artifact.status,
            "created_at": artifact.created_at,
            "reason": reason,
            "issues": artifact.issues,
            "severity": artifact.severity,
            "prohibited_actions": artifact.prohibited_actions,
            "next_step": "请等待专业审核；在审核完成前不要执行或调整训练计划。",
        },
    }
    if thread_id is not None:
        payload["thread_id"] = thread_id
    return payload


def review_pending_result(review_store, profile: dict, query: str, result: dict) -> dict:
    """Deliver only a review summary when the final gate requires human review."""
    if result.get("_persistence_allowed") is True:
        delivered = dict(result)
        delivered["delivery_status"] = "safe_delivered"
        return delivered
    if not result.get("requires_review"):
        return result
    artifact = create_review_artifact(review_store, profile, query, result)
    return build_review_pending_payload(result, artifact)


def make_user_key(profile: dict) -> int:
    """用身体数据组合生成伪用户 ID，无认证场景下的跨会话标识。

    将身高+体重+目标+伤病 MD5 hash 为固定整数。
    使用 hashlib.md5 而非 Python hash() —— 后者在 PYTHONHASHSEED 随机化
    下多进程间不稳定，会导致同一用户在不同 worker 中无法匹配缓存。
    注意：不同用户可能碰撞（1/1M），但在个人项目场景下可接受。
    """
    import hashlib
    raw = (
        f"{profile.get('height', 0)}|{profile.get('weight', 0)}|"
        f"{profile.get('goal', '')}|{','.join(sorted(profile.get('injuries', [])))}"
    )
    return int(hashlib.md5(raw.encode()).hexdigest()[:8], 16) % 1000000


def summarize_plan_for_context(plan: dict) -> str:
    """【私有方法】从训练计划提取摘要，供多轮对话的 plan_state 存储。

    输入：
        plan: dict — 完整的训练计划结果（含 days 列表）
    输出：
        str — "第1天(胸+三头): 杠铃卧推/哑铃飞鸟... 第2天(背+二头): ..." 格式的摘要
    """
    days = plan.get("days", [])
    if not days:
        return ""
    parts = []
    for d in days:
        day_num = d.get("day", "?")
        focus = d.get("focus", "")
        exercises = d.get("exercises", [])
        ex_names = [e.get("name", "?") for e in exercises[:6]]
        ex_str = "/".join(ex_names)
        label = f"第{day_num}天" + (f"({focus})" if focus else "")
        parts.append(f"{label}: {ex_str}")
    return " | ".join(parts)


def persist_if_safe(cache, conversation, long_term, profile: dict, query: str,
                    result: dict, session_id: str | None = None) -> bool:
    """Persist only a fully safe terminal result."""
    if not isinstance(result, dict) or result.get("_persistence_allowed") is not True:
        return False
    if not safe_cached_result(result, expected_goal=profile.get("goal")):
        return False
    if session_id:
        summary = summarize_plan_for_context(result)
        conversation.set_plan_state(session_id, summary)
        conversation.add_turn(session_id, "assistant", summary[:500])
    else:
        cache.set(profile, query, result)
    pseudo_uid = make_user_key(profile)
    long_term.save_preference(pseudo_uid, "profile", profile)
    long_term.save_preference(pseudo_uid, "goal", profile.get("goal", ""))
    long_term.save_preference(pseudo_uid, "equipment", profile.get("available_equipment", []))
    return True
