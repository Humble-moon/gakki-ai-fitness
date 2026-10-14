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
from src.core.nutrition import calc_calorie_target
from src.core.goal_contract import (
    plan_goal_issue,
    plan_goal_matches,
    validate_requested_goal,
)

# Prohibited actions surfaced whenever a plan is held for human review.
#: 伤病类拦截的禁止事项。这类问题用户无法自行判断，因此连"替换动作"也一并禁止——
#: 换掉的替代动作同样可能与伤病冲突，而用户没有能力判断。
REVIEW_PROHIBITED_ACTIONS = [
    "不要开始训练计划",
    "不要加大训练强度",
    "不要自行替换或增加动作",
]

#: 非伤病类拦截（器械不满足、动作名有误等）的禁止事项。
#: 刻意**去掉**「不要自行替换或增加动作」：这类问题的解法恰恰就是替换——
#: 明细里已经给了替代动作，再禁止替换等于自相矛盾。实测中就出现过
#: 同一屏上「建议替换为俯身哑铃划船」与「不要自行替换或增加动作」并存。
REVIEW_PROHIBITED_ACTIONS_FIXABLE = [
    "不要按原样执行这份计划",
]

#: **确定性**的伤病信号。这三类都由规则引擎或语义匹配产生，文案固定、不含歧义。
#: 只要命中就一定是伤病问题，优先级最高。
_INJURY_STRONG = (
    "[语义匹配]",            # 语义匹配到伤病档案
    "[规则引擎] 伤病",        # 规则引擎的伤病—动作冲突
    "禁止动作:",             # 档案里的 banned_exercises
    "用户查询含伤病",         # 高危关键词命中
)

#: 非伤病类的确定性信号。
_FIXABLE_MARKERS = (
    ("equipment", ("器械约束", "未包含", "器械清单", "需要平凳", "需要可调",
                   "需要上斜凳", "需要TRX", "未列出")),
    ("naming", ("不在动作库", "未收录", "确认动作名称", "动作库中不存在")),
)

#: **宽松**的伤病词。只在没有更强信号时兜底。
_INJURY_LOOSE = ("伤病", "疼痛", "损伤", "急症", "冲突")


def _hold_texts(result: dict) -> str:
    """把这次送审的所有问题文本拼成一整块，供分类扫描。"""
    texts = []
    for issue in (result.get("active_issues") or []):
        texts.append(str(issue.get("issue", "")) if isinstance(issue, dict) else str(issue))
    for item in (result.get("review_suggestions") or []):
        texts.append(str(item.get("issue", "")) if isinstance(item, dict) else str(item))
    texts.append(str(result.get("review_reason") or ""))
    return " ".join(texts)


def classify_hold(result: dict) -> str:
    """判断这次扣下计划的主要原因属于哪一类。

    分类决定两件事：下一步该把用户引向哪里（看医生？补器械？改动作名？），
    以及禁止事项该说多严。写死一套文案的后果实测过——用户因为计划里
    用了一根他没有的 TRX 弹力带，被告知"去骨科或运动医学科评估"。

    **判定顺序是有讲究的，不能简单地"伤病优先"**：器械与动作名那两类问题的
    文案由大模型生成（检查器按提示词检查器械约束），措辞不受控，句子里
    顺带出现"伤病"两个字完全可能。若把宽松的伤病词排在前面，一条
    「器械约束不满足」的提示只要提到伤病就会被判成医疗问题，
    于是又回到"缺根带子去看骨科"。

    所以顺序是：确定性的伤病信号 → 可自行修复类 → 宽松伤病词兜底。
    真实的伤病问题一定带确定性信号（规则引擎与语义匹配都会打固定前缀），
    因此这样排不会漏判；而模糊时落到宽松词，仍然偏保守。
    """
    blob = _hold_texts(result)

    if any(m in blob for m in _INJURY_STRONG):
        return "injury"
    for category, markers in _FIXABLE_MARKERS:
        if any(m in blob for m in markers):
            return category
    if result.get("unknown_exercises"):
        return "naming"
    if any(m in blob for m in _INJURY_LOOSE):
        return "injury"
    return "other"


def hold_guidance(result: dict) -> tuple[list[str], str]:
    """按拦截原因给出（禁止事项, 下一步）。

    分开返回而不是各写各的，是为了让两者永远指向同一件事——
    一个说"去就医"、另一个说"自行替换即可"是最伤信任的组合。
    """
    category = classify_hold(result)

    if category == "injury":
        return REVIEW_PROHIBITED_ACTIONS, (
            "把上面的风险明细带给骨科或运动医学科评估；"
            "在获得专业意见前，不要执行或调整训练计划。")

    if category == "equipment":
        return REVIEW_PROHIBITED_ACTIONS_FIXABLE, (
            "这份计划用到了你的器械清单里没有的器械。"
            "在「编辑资料」里把器械补全（例如是否有平凳、TRX、可调卧推凳），"
            "或直接重新生成一份——系统会按你实际有的器械重新编排。")

    if category == "naming":
        return REVIEW_PROHIBITED_ACTIONS_FIXABLE, (
            "计划里有几个动作名不在动作库中，可能是模型自造的变体。"
            "可以重新生成一份；若你认识这些动作，也可以手动把它们换成"
            "名称规范、你熟悉的同类动作。")

    return REVIEW_PROHIBITED_ACTIONS_FIXABLE, (
        "请先查看上面的问题明细；确认无误后重新生成一份计划。"
        "若反复出现同类问题，可以补充你的训练条件（器械、伤病、时间）后重试。")


def safe_cached_result(result: dict | None, expected_goal: str | None = None,
                       profile: dict | None = None) -> dict | None:
    """Accept only cache entries that already passed the final gate and goal contract.

    传入 ``profile`` 时会重算热量目标再返回。原因是缓存指纹（``make_user_key``）
    只由身高/体重/目标/伤病组成，而热量还取决于年龄、性别、每周天数——这几项
    变化不会造成缓存失效，于是命中缓存的用户会拿到**上一份档案算出来的热量**。
    缓存的是计划，热量是当前档案的派生视图，交付时重算一次最省事也最准确。
    """
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
    return attach_nutrition(result, profile or {})


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


def attach_nutrition(result: dict, profile: dict) -> dict:
    """把每日热量与蛋白质目标挂到计划上（``result["nutrition"]``）。

    与上面覆盖 ``sessions_per_week`` / ``weeks`` 属于同一类事：模型不参与，
    全部由 profile 确定性算出来。热量是计划里唯一的「数字结论」，交给模型生成
    就无从核对——公式和依据文本都写在 src/core/nutrition.py。

    没有体重就不挂：算不出来的时候宁可不显示，也不给一个看着很确定的假数字。
    """
    weight = profile.get("weight")
    if not weight:
        return result
    result["nutrition"] = calc_calorie_target(
        weight_kg=weight,
        height_cm=profile.get("height"),
        age=profile.get("age"),
        sex=profile.get("sex", "不填"),
        goal=profile.get("goal") or "muscle_building",
        days_per_week=profile.get("days_per_week"),
    )
    return result


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
    attach_nutrition(result, profile)
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

    # 功能性受限（活动度/柔韧/僵硬）的做法调整建议：**随计划一起交付**，不进任何闸门。
    #
    # 它和上面那些 warning 一样只是提示，区别在于来源不是 LLM 检查器，而是
    # HITL 的确定性规则/语义层。之所以要在这里明确声明"不参与闸门"：
    # 下面 hard_ok 只看 schema_valid / is_safe / requires_review / confidence，
    # 一旦把 cautions 接进那几个判据，"活动度不好"就会重新变回扣下计划的理由，
    # 死锁照旧——而这正是本次要修的问题。
    cautions = [
        str(c) for c in (final.get("review_cautions") or result.get("review_cautions") or [])
        if str(c).strip()
    ]
    if cautions:
        warnings = warnings + cautions

    requires_review = final.get("requires_human_review") is True
    confidence = final.get("confidence")
    confidence_valid = isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
    result.update({"warnings": warnings, "active_issues": active_issues,
                   "resolved_issues": resolved_issues,
                   "unknown_exercises": unknown_exercises,
                   "cautions": cautions,
                   "requires_review": requires_review or not checks or not is_safe,
                   "review_reason": final.get("review_reason", ""),
                   "review_severity": final.get("review_severity", ""),
                   "review_suggestions": list(final.get("review_suggestions") or []),
                   "confidence": confidence if confidence_valid else 0,
                   "rewrite_count": rewrite_count})
    goal_matches = expected_goal is None or plan_goal_matches(result, expected_goal)
    hard_ok = bool(schema_valid and is_safe and not requires_review
                   and confidence_valid and goal_matches
                   and not provider_degraded and not result.get("_degraded", False))
    # 缓存闸门：在"能交付"之上再要求问题清单干净。缓存是"同一份身体数据命中同一份
    # 计划"的去重产物，宁可少命中也不缓存带瑕疵的版本——这道额外要求是合理的。
    result["_persistence_allowed"] = bool(hard_ok and issues_empty)
    # 交付闸门：只要结果可以安全交付即可。它与缓存闸门只差 issues_empty 一项，
    # 而那一项衡量的是**计划的完成度**，不是**用户是谁**。
    result["_delivery_allowed"] = hard_ok
    return result


def create_review_artifact(review_store, profile: dict, query: str, result: dict,
                           owner_key: str = ""):
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
    prohibited, _ = hold_guidance(result)
    return review_store.create(
        profile_summary={key: profile.get(key) for key in ("goal", "injuries", "training_years") if key in profile},
        query=query,
        issues=issues,
        severity=severity,
        prohibited_actions=prohibited,
        owner_key=owner_key or "",
    )


_INTERNAL_MARKER_RE = re.compile(r"^\s*\[(规则引擎|语义匹配)\]\s*")

#: 交付层的第二道剥离：把只对内部有意义、却会落到用户屏幕上的片段摘掉。
#:
#: 「（触发词: 背）」暴露的是关键词匹配这个实现——用户看到的是"系统按字面比对了
#: 一个『背』字"，比看到一句人话更容易让人怀疑整套判断。「，建议人工审核」则是
#: 内部流程术语，而这一屏本来就写着"需要专业人员确认"，再说一遍像在推诿。
#: 两者都不改变信息量，只改变它看起来像不像一个成品。
_INTERNAL_FRAGMENT_RE = re.compile(r"（触发词:[^）]*）|，建议人工审核")


def _strip_internal_markers(issue):
    """剥掉只对内部有意义的前缀标记与术语。

    ``[规则引擎]`` / ``[语义匹配]`` 标的是这条问题「由哪条通道发现」，
    供 classify_hold 分类与日志排查使用。但审核面板会把 issue 原文直接
    显示给用户，而普通用户不知道「规则引擎」是什么，只会觉得莫名其妙。
    「（触发词: X）」与「建议人工审核」同理。

    因此只在**交付给用户的载荷**这一步剥掉，原始数据保持不动——
    classify_hold 仍要按这些标记分流，剥早了分类就失效了。
    """
    def scrub(text: str) -> str:
        return _INTERNAL_FRAGMENT_RE.sub(
            "", _INTERNAL_MARKER_RE.sub("", text))

    if isinstance(issue, dict):
        cleaned = dict(issue)
        if cleaned.get("issue"):
            cleaned["issue"] = scrub(str(cleaned["issue"]))
        return cleaned
    return scrub(str(issue or ""))


def build_review_pending_payload(result: dict, artifact, thread_id: str | None = None) -> dict:
    """Build the ``review_pending`` delivery payload for a held plan."""
    final_check = result.get("active_issues") or []
    # 兜底那一路直接取了 issue 原文，会带上 [规则引擎] / （触发词: X）这类内部标记，
    # 而它是"风险原因"这一行的正文——比明细列表更显眼，必须同样过一遍剥离。
    reason = result.get("review_reason") or (
        final_check[0].get("issue") if final_check and isinstance(final_check[0], dict) else "安全检查需要人工确认"
    )
    reason = _strip_internal_markers(reason)
    payload = {
        "delivery_status": "review_pending",
        "requires_review": True,
        "review": {
            "review_id": artifact.review_id,
            "status": artifact.status,
            "created_at": artifact.created_at,
            "reason": reason,
            "issues": [_strip_internal_markers(i) for i in (artifact.issues or [])],
            "severity": artifact.severity,
            "prohibited_actions": artifact.prohibited_actions,
            # 下一步按**拦截原因**给，而不是写死一句。
            # 写死那版实测出过事：用户因计划用了一根他没有的 TRX 弹力带，
            # 被告知"去骨科或运动医学科评估"——因为缺根带子去看医生，
            # 既浪费对方时间，也让整套安全机制显得没在理解问题。
            # 措辞也刻意不提"等待审核员"：本版本没有人工审核台（见 5.9 节局限性），
            # 承诺一个不存在的角色会让用户一直等下去。
            "next_step": hold_guidance(result)[1],
            "hold_reason": classify_hold(result),
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


def review_pending_result(review_store, profile: dict, query: str, result: dict,
                          owner_key: str = "") -> dict:
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
        artifact = create_review_artifact(review_store, profile, query, result,
                                          owner_key=owner_key)
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

    【两类闸门，不要混淆】
    这里写三种东西，它们的准入条件本来就不同：

        - **缓存**：要求最严（``_persistence_allowed`` 还要求问题清单干净）。
          它是"同一份身体数据命中同一份计划"的去重产物，少命中无害，存了带
          瑕疵的版本反而会长期污染后续请求。
        - **用户画像**：只要求结果可以安全交付（``_delivery_allowed``）。
          画像记的是**用户是谁**（身高体重目标伤病习惯），这些是用户自己
          填的，与计划的完成度无关。
        - **会话状态**：随缓存一起走。

    合并成一道闸门曾造成一个隐蔽后果：LLM 检查器几乎从不返回空的问题清单
    （见 ``review_pending_result`` 的说明），于是 ``issues_empty`` 长期为假，
    **生成计划这条路上的用户画像几乎从不落盘**——而知识问答路径是直接写的。
    表现就是「问答记住了我，生成计划没记住」，同一件事两条路径行为不一致。

    注意身份键的解析：长期记忆用 ``athlete_key``（前端 localStorage 的稳定
    UUID），缺失时才退回 ``make_user_key(profile)`` 指纹。后者由身高体重哈希
    而成，而增肌减脂期体重必然变化，等于每换一次体重就换一个身份。

    语义缓存**仍然**按 profile 指纹（``cache.set(profile, ...)``）：缓存的
    语义是"同一份身体数据 + 同一目标命中同一份计划"，用指纹是对的。身份与
    缓存键是两件事。
    """
    if not isinstance(result, dict):
        return False
    cache_ok = result.get("_persistence_allowed") is True
    delivery_ok = result.get("_delivery_allowed") is True
    if not (cache_ok or delivery_ok):
        return False

    if cache_ok and safe_cached_result(result, expected_goal=profile.get("goal")):
        if session_id:
            summary = summarize_plan_for_context(result)
            conversation.set_plan_state(session_id, summary)
            conversation.add_turn(session_id, "assistant", summary[:500])
        else:
            cache.set(profile, query, result)

    # 画像在交付闸门下写入，不受缓存闸门牵连。
    if delivery_ok:
        persist_long_term_profile(long_term, long_term_user_key(profile, athlete_key), profile)
    return True


def long_term_user_key(profile: dict, athlete_key: str | None = None):
    """长期记忆的身份键。集中一处，避免读写两侧各算各的而漂移。

    读写必须用同一个键：写用 athlete_key、读用 profile 指纹的话，
    用户永远读不回自己刚存的偏好。
    """
    return athlete_key or make_user_key(profile)


def persist_long_term_profile(long_term, user_key, profile: dict) -> None:
    """把用户画像写入长期记忆。写键集合在此单点定义。

    生成计划（``persist_if_safe``）与知识问答（``Orchestrator`` 问答流）
    两条路径都调用它。此前两条路径各写各的——计划写 3 个键、问答写 2 个，
    而问答那份还引用了未定义变量，实际从未落库。

    ``habits``（运动习惯/爱好）仅在非空时写入：它是可独立更新的记忆，
    若每轮都用空值覆盖，用户填过的习惯会在一次随手问答后被清掉。
    """
    long_term.save_preference(user_key, "profile", profile)
    long_term.save_preference(user_key, "goal", profile.get("goal", ""))
    long_term.save_preference(user_key, "equipment", profile.get("available_equipment", []))
    habits = profile.get("habits") or []
    if habits:
        long_term.save_preference(user_key, "habits", habits)
