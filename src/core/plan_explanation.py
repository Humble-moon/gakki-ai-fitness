"""
计划可解释性 — 把生成过程中算出来但未展示的依据整理成对用户可见的解释块。

存在理由：用户看到一份训练计划时的第一反应不是"好专业"，而是"我凭什么信它"。
这个疑问答不上来，再好的计划也不会被采用。而流水线里其实一路都在产生答案——
检索了多少候选动作、匹配了哪个模板、安全检查发现并修掉了什么——只是这些数据
在返回给前端前全被丢掉了。

这是**展示层**模块：不改变任何生成行为，只把已有事实整理出来。因此它对计划
质量和安全闸门零影响（对应测试见 test_plan_explanation.py 末尾的守卫）。
"""

from __future__ import annotations

# 技能模板的中文标签。内部用英文标识，展示给用户时必须翻译，
# 不能让 muscle_building 这种内部术语泄漏到界面上。
SKILL_LABELS = {
    "muscle_building": "增肌计划",
    "fat_loss": "减脂计划",
    "exercise_analysis": "动作分析",
}

# 展示用的肌群数量上限——列几十个肌群没有意义，反而淹没关键信息
MAX_MUSCLES_SHOWN = 8


def build_explanation(plan_meta: dict, retrieved: dict, result: dict) -> dict:
    """构建计划解释块。

    输入：
        plan_meta: Planner 的产出（含 skill / subtasks）
        retrieved: Retriever 的产出（含 exercises）
        result: 已终结的计划结果（含 active_issues / resolved_issues / confidence 等）

    输出：
        dict — 三段式解释：规划依据 / 动作筛选 / 安全检查。
        任何一段数据缺失时该段为空，前端据此隐藏，绝不编造。

    为什么读 result 而不是重算 issues：finalize_result 已经算过 active/resolved
    的差集，重算会有两处口径，早晚对不上。
    """
    return {
        "skill": _explain_skill(plan_meta),
        "retrieval": _explain_retrieval(retrieved),
        "safety": _explain_safety(result),
    }


def _explain_skill(plan_meta: dict) -> dict:
    """规划依据：匹配了哪个模板、拆成了哪些子任务。

    两项都没有时返回空块（而非"通用模板"这类编造的标签）——宁可整段隐藏，
    也不给用户一个我们自己都不确定的信息。
    """
    if not isinstance(plan_meta, dict):
        return {}

    skill_id = plan_meta.get("skill") or ""
    subtasks = [s for s in (plan_meta.get("subtasks") or []) if isinstance(s, str) and s.strip()]
    if not skill_id and not subtasks:
        return {}

    return {
        "id": skill_id,
        "label": SKILL_LABELS.get(skill_id) or "通用模板",
        "subtasks": subtasks,
    }


def _explain_retrieval(retrieved: dict) -> dict:
    """动作筛选：候选规模、来源构成、覆盖肌群。"""
    if not isinstance(retrieved, dict):
        return {}

    exercises = retrieved.get("exercises")
    if not isinstance(exercises, list) or not exercises:
        return {}

    semantic = 0
    structured = 0
    muscles: list[str] = []
    seen_muscles = set()

    for row in exercises:
        if not isinstance(row, dict):
            continue
        # 路 2（MCP 按肌群精确检索）会显式标记 source="mcp"；
        # 其余来自 AgenticRAG 的语义检索，据此区分两条检索路径的贡献
        if row.get("source") == "mcp":
            structured += 1
        else:
            semantic += 1

        raw = row.get("muscles") or row.get("target_muscles") or []
        if isinstance(raw, str):
            raw = [raw]
        for muscle in raw:
            if isinstance(muscle, str) and muscle.strip() and muscle not in seen_muscles:
                seen_muscles.add(muscle)
                muscles.append(muscle)

    return {
        "total": semantic + structured,
        "semantic": semantic,
        "structured": structured,
        "muscles": muscles[:MAX_MUSCLES_SHOWN],
        "muscle_total": len(muscles),
    }


def _explain_safety(result: dict) -> dict:
    """安全检查：检查了几轮、置信度、修掉了什么、还留着什么。

    「修掉了什么」是最能建立信任的一条——用户看到系统主动发现并纠正过问题，
    才会相信那些没被报出来的部分是检查过的。
    """
    if not isinstance(result, dict):
        return {}

    resolved = _issue_texts(result.get("resolved_issues"))
    active = _issue_texts(result.get("active_issues"))

    return {
        "rounds": int(result.get("rewrite_count") or 0) + 1,
        "confidence": _safe_confidence(result.get("confidence")),
        "resolved": resolved,
        "active": active,
        "degraded": bool(result.get("_degraded") or result.get("provider_degraded")),
    }


def _issue_texts(raw) -> list[str]:
    """把 issue 列表统一成字符串。

    issue 元素既可能是字符串，也可能是 {"issue": "...", ...} 的字典，两种都要兼容。
    """
    if not isinstance(raw, list):
        return []
    texts = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            texts.append(item.strip())
        elif isinstance(item, dict):
            text = item.get("issue") or item.get("description") or ""
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
    return texts


def _safe_confidence(value) -> float | None:
    """置信度只有是合法数值时才有展示意义，否则返回 None 让前端隐藏。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not 0.0 <= float(value) <= 1.0:
        return None
    return round(float(value), 4)
