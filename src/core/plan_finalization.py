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

import re

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


#: 括号补充说明，如「保加利亚分腿蹲（扶支撑）」「上斜哑铃弯举（肱肌专注）」。
#: 中英文括号都算——库里两种写法都有。
_PAREN_NOTE = re.compile(r"[(（][^)）]*[)）]")

#: 常见的姿势/支撑类修饰词。它们加在动作名前只改变执行细节，
#: 不构成一个新动作——库里的基线名通常不带这些前缀。
_MODIFIER_PREFIXES = (
    "站姿", "坐姿", "跪姿", "俯卧", "仰卧", "平躺", "靠墙", "地面", "地板",
    "单臂", "双臂", "单手", "双手", "单腿", "双腿", "负重", "自重", "徒手",
    "上斜", "下斜", "坐姿", "扶墙",
)


def normalize_exercise_name(name: str) -> str:
    """把动作名归一化到"基线名"，用于与动作库比较。

    只做两件保守的事：去掉括号补充说明、剥掉姿势类修饰前缀。
    目的不是模糊匹配，而是把「同一个动作的说法差异」与「根本不存在的
    动作」分开——前者不该阻断交付，后者才该。

    实测依据：`保加利亚分腿蹲（扶支撑）`／`站姿哑铃弯举`／`地面哑铃飞鸟`
    这类名字精确匹配库时为 0，但它们显然指向库里真实存在的
    `保加利亚分腿蹲`／`哑铃弯举`／`哑铃飞鸟`。若一律当作编造动作送审，
    会把正常计划全部扣下。

    而归一化对真正的词素拼接无效：`慢离心哑铃地板卧推`（`慢离心卧推` +
    `哑铃地板卧推` 拼接而成）剥掉修饰前缀后不变，仍然落在库外——
    要抓的正是这种。
    """
    if not name:
        return ""
    normalized = _PAREN_NOTE.sub("", name).strip()
    # 只剥一层前缀：库里的基线名本身可能以这些词开头（如"单臂哑铃卧推"），
    # 多剥会把真实名字剥坏。循环一次即可覆盖"站姿哑铃弯举"这类单层修饰。
    for prefix in _MODIFIER_PREFIXES:
        if normalized.startswith(prefix) and len(normalized) > len(prefix):
            normalized = normalized[len(prefix):]
            break
    return normalized


def collect_unknown_exercises(result: dict, known_names) -> list[str]:
    """找出计划中不在动作库里的动作名（保持出现顺序，去重）。

    为什么需要这道校验：重写回路只做键名归一，**不校验动作名是否存在**。
    实测中首轮计划的动作名都是真的（哑铃卧推等），
    但重写 3 轮后出现了「慢离心哑铃地板卧推」这类库里查不到的名字——
    它们不是凭空编造，而是由真实动作名词素拼接而成，比纯幻觉更难发现。

    **比较前先做名称归一化**（见 :func:`normalize_exercise_name`）：
    否则「保加利亚分腿蹲（扶支撑）」这种"同一动作的另一种写法"会被误判为
    编造动作，把正常计划全扣下。库内名同样归一化后建索引，保证两侧口径一致。

    依赖显式传入 ``known_names`` 而非在此查库：本模块是纯函数模块，
    不做 IO（见模块 docstring）。
    """
    if not known_names:
        return []
    raw = known_names if isinstance(known_names, (set, frozenset)) else set(known_names)
    known = {normalize_exercise_name(n) for n in raw} | set(raw)
    unknown: list[str] = []
    seen: set[str] = set()
    for day in result.get("days") or []:
        if not isinstance(day, dict):
            continue
        for ex in day.get("exercises") or []:
            if not isinstance(ex, dict):
                continue
            name = ex.get("name") or ex.get("exercise") or ""
            if not name or name in seen:
                continue
            seen.add(name)
            if name not in known and normalize_exercise_name(name) not in known:
                unknown.append(name)
    return unknown


def finalize_result(result: dict, checks: list[dict], rewrite_count: int,
                    *, provider_degraded: bool = False,
                    expected_goal: str | None = None,
                    known_exercise_names=None) -> dict:
    """Normalize one terminal state and decide whether persistence is allowed.

    ``known_exercise_names`` 可选：动作库全部动作名的集合。传入时会校验计划中的
    动作名是否存在，把库外动作记为 ``unknown_exercises`` 并强制送审。
    不传则跳过该校验（老调用方保持原行为）。
    """
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
    # 只由**最终那一轮**检查决定是否送审。
    #
    # 这里曾用 any(... for c in checks) 扫描全部历史，后果是：重写回路修好的问题
    # 会永远留在历史里，导致最终检查明明返回 safe=True / issues=[] 的计划依然被扣下。
    # 而首轮草稿几乎总会被找出若干问题（LLM 被要求挑毛病），于是正常交付路径基本
    # 走不到——这与 resolved_issues 单独追踪「已修正问题」的设计意图也自相矛盾。
    #
    # 只看最终轮是安全的：计划层面的问题（某动作与伤病冲突）被重写改掉后，最后一轮
    # 自然不再报；查询层面的问题（用户问的就是危险动作）不随重写改变，最后一轮依然
    # 会报。两种情况的拦截能力都不受影响。
    # 动作库校验：库外动作名——**只提示，不阻断交付**。
    #
    # 曾经这里是强制送审（requires_review=True），实测证明那是错的：
    # 无伤病健康用户的正常计划被扣下两次，理由分别是"5 个动作库中不存在"
    # 与"6 个动作库中不存在"，而被点名的都是 「保加利亚分腿蹲（扶支撑）」
    # 「地面哑铃飞鸟」这类**同一动作的另一种写法**，以及 「帕洛夫推」这类
    # **真实存在、只是本库未收录**的动作。
    #
    # 更根本的原因是：LLM 生成的计划本来就不会严格只用库内动作名，
    # 强制送审等于 100% 的计划都交付不出去（而本项目的 HITL 又没有
    # 定义"审核人"角色，形成死锁）。
    #
    # 定位修正：库外动作是"值得让用户知道"，不是"可能有害"。真正的安全
    # 拦截保持不变——伤病冲突、danger 级建议、低置信度仍然阻断。
    unknown_exercises = collect_unknown_exercises(result, known_exercise_names)
    if unknown_exercises:
        warnings = warnings + [
            f"动作「{name}」不在动作库中，请确认动作名称是否正确" for name in unknown_exercises
        ]

    requires_review = final.get("requires_human_review") is True
    confidence = final.get("confidence")
    confidence_valid = isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
    result.update({"warnings": warnings, "active_issues": active_issues,
                   "resolved_issues": resolved_issues,
                   "unknown_exercises": unknown_exercises,
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
    # 库外动作单独列出：审核者需要直接看到"哪些动作名是模型编造的"，
    # 这比混在 issues 列表里更容易被跳过，而它恰恰是最该被核对的一项。
    unknown = result.get("unknown_exercises")
    if unknown:
        payload["review"]["unknown_exercises"] = list(unknown)
    if thread_id is not None:
        payload["thread_id"] = thread_id
    # 解释块只含检索统计与检查计数，不含被扣下的计划内容，
    # 因此可以安全地一并交给审核方——让审核者知道系统此前检查了几轮、依据是什么。
    if isinstance(result.get("explain"), dict):
        payload["explain"] = result["explain"]
    return payload


def review_pending_result(review_store, profile: dict, query: str, result: dict) -> dict:
    """交付判定：需要人工审核就扣下，否则交付。

    这里曾用 _persistence_allowed 作为交付闸门，但它是「能否缓存」的判据，额外要求
    issues_empty。而 LLM 检查器几乎总会给出若干条建议性提示（实测 4 轮检查分别为
    4/4/3/3 条），issues_empty 实际上永远为假——后果是没有 delivery_status，
    前端 `!== 'safe_delivered'` 直接 return，用户什么都看不到。

    交付与缓存是两件事，必须分开：
        requires_review → 交付闸门（已内含 is_safe 与「至少有一轮有效检查」）
        _persistence_allowed → 缓存闸门（继续要求 issues 清空，宁可少缓存）
    计划上带着的建议性提示会以 warnings 形式一并展示，不会静默丢掉。
    """
    if result.get("requires_review"):
        artifact = create_review_artifact(review_store, profile, query, result)
        return build_review_pending_payload(result, artifact)
    delivered = dict(result)
    delivered["delivery_status"] = "safe_delivered"
    return delivered


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
                    result: dict, session_id: str | None = None,
                    athlete_key: str | None = None) -> bool:
    """Persist only a fully safe terminal result.

    长期记忆的身份用 ``athlete_key``（前端 localStorage 的稳定 UUID），
    缺失时才退回 ``make_user_key(profile)`` 指纹。原因：后者由身高体重哈希
    而成，而增肌减脂期体重必然变化，等于每换一次体重就换一个身份——
    用户会读不到自己先前的偏好，删除数据时也无从定位。

    注意语义缓存**仍然**按 profile 指纹（``cache.set(profile, ...)``）：
    缓存的语义是"同一份身体数据 + 同一目标命中同一份计划"，用指纹是对的，
    把它换掉反而会让缓存失效。身份与缓存键是两件事。
    """
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
    user_key = athlete_key or make_user_key(profile)
    long_term.save_preference(user_key, "profile", profile)
    long_term.save_preference(user_key, "goal", profile.get("goal", ""))
    long_term.save_preference(user_key, "equipment", profile.get("available_equipment", []))
    return True


def long_term_user_key(profile: dict, athlete_key: str | None = None):
    """长期记忆的身份键。集中一处，避免读写两侧各算各的而漂移。

    读写必须用同一个键：写用 athlete_key、读用 profile 指纹的话，
    用户永远读不回自己刚存的偏好。
    """
    return athlete_key or make_user_key(profile)
