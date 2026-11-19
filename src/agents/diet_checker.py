"""DietChecker —— 饮食计划的 LLM 语义审查（确定性冲突在 service 层独立合并）。

失败语义与训练侧 FactChecker 同构：LLM 不可用/解析失败时返回一份
``provider_degraded`` 标记的检查（is_safe=False），让交付闸门 fail-closed，
而不是假装审查通过。
"""

from __future__ import annotations

import logging

from src.config import HITL_CONFIDENCE_THRESHOLD
from src.llm.prompts.diet_checker import build_diet_checker_messages

logger = logging.getLogger(__name__)


def check_diet(llm, plan: dict, profile: dict, prefs: dict) -> dict:
    """一轮语义审查。置信度阈值在本层应用（finalize 只查数值合法性）。"""
    messages = build_diet_checker_messages(plan, profile, prefs)
    try:
        raw = llm.chat_with_json_mode(messages, model="reasoner")
    except Exception as exc:  # noqa: BLE001 - 审查不可用必须 fail-closed
        logger.warning("[diet_checker] LLM 审查不可用：%s", exc)
        return _degraded_check()
    if not isinstance(raw, dict) or "raw" in raw:
        return _degraded_check()
    confidence = raw.get("confidence")
    confidence_valid = isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
    issues = raw.get("issues") if isinstance(raw.get("issues"), list) else []
    has_danger = any(isinstance(i, dict) and i.get("severity") == "danger" for i in issues)
    is_safe = raw.get("is_safe") is True and not has_danger
    requires_review = bool(raw.get("requires_human_review")) or has_danger or not is_safe
    if confidence_valid and confidence < HITL_CONFIDENCE_THRESHOLD:
        requires_review = True
    return {
        "is_safe": is_safe,
        "issues": issues,
        "confidence": confidence if confidence_valid else None,
        "requires_human_review": requires_review,
        "review_conflicts": [],
        "review_reason": "",
        "review_severity": "danger" if has_danger else "",
        "review_cautions": [],
    }


def _degraded_check() -> dict:
    return {
        "is_safe": False,
        "issues": [{"issue": "安全审查服务暂时不可用，本次未能完成审查", "severity": "danger"}],
        "confidence": None,
        "requires_human_review": True,
        "review_conflicts": [],
        "review_reason": "审查服务降级",
        "review_severity": "danger",
        "review_cautions": [],
        "provider_degraded": True,
    }


def merge_deterministic_conflicts(check: dict, conflicts: list[dict]) -> dict:
    """把确定性规则冲突合并进检查：可定位冲突 → danger + is_safe=False。

    即使 LLM 判 safe，这里仍能独立触发阻断（训练侧 R13 的教训：安全网不能
    只接在一条路径上）。冲突明细进 review_conflicts，拒绝载荷据此点名食物。
    """
    if not conflicts:
        return check
    merged = dict(check)
    issues = list(merged.get("issues") or [])
    for conflict in conflicts:
        issues.append({"issue": f"{conflict['food']}：{conflict['reason']}",
                       "severity": "danger", "food": conflict["food"]})
    merged["issues"] = issues
    merged["is_safe"] = False
    merged["requires_human_review"] = True
    merged["review_conflicts"] = list(conflicts)
    merged["review_reason"] = "过敏/疾病/忌口冲突"
    merged["review_severity"] = "danger"
    return merged