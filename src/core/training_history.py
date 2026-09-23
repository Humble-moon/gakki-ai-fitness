"""
训练历史上下文 — 把训练日志与已确认的调整渲染成可注入生成 Prompt 的文本。

这是**输入侧的单点真相**：手写编排器（orchestrator.py）与 LangGraph 编排
（graph/nodes.py）都只调用 build_training_context()，不各自拼装。这与
plan_finalization.py 在**输出侧**承担的角色对称——终态逻辑只有一份，
输入上下文也只有一份，两个后端因此不可能漂移。

设计约束：本模块所有对外函数**永不抛异常**。训练历史是生成质量的增强项，
不是必需项；Redis 或 PostgreSQL 不可用时应当退化成「没有历史」的正常生成，
而不是把整个计划生成链路带崩。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from src.core.training_analytics import compute_stats

logger = logging.getLogger(__name__)

# 长期记忆里存放「已确认调整」的偏好键名
ADJUSTMENT_KEY = "training_adjustments"

# 已确认调整的有效期：超过这个天数视为过时（用户水平可能已变化），不再注入
ADJUSTMENT_TTL_DAYS = 42

# 注入 Prompt 的动作条数上限——上下文不是越长越好，太长会稀释关键信息
MAX_EXERCISES_IN_CONTEXT = 6


def build_training_context(athlete_key: str | None, *, store=None, long_term=None,
                           weeks: int = 8) -> str:
    """构建注入生成 Prompt 的训练历史上下文文本。

    输入：
        athlete_key: 稳定运动员标识；为空时直接返回空串
        store / long_term: 依赖注入，便于测试；缺省时惰性构造真实实现
        weeks: 回溯的周数

    输出：
        可直接拼进 Prompt 的文本块；任何异常或无数据时返回空串

    核心逻辑：
        1. 读训练日志 → 2. 聚合统计 → 3. 读已确认调整 → 4. 渲染成紧凑文本
    """
    if not athlete_key:
        return ""

    try:
        store = store if store is not None else _default_store()
        if store is None or not getattr(store, "enabled", True):
            return ""

        logs = store.get_logs(athlete_key, weeks=weeks)
        if not logs:
            return ""

        sections = [_render_stats(compute_stats(logs), weeks)]
        adjustments = load_adjustments(athlete_key, long_term=long_term)
        if adjustments:
            sections.append(_render_adjustments(adjustments))
        return "\n\n".join(s for s in sections if s)
    except Exception as exc:  # 增强项失败绝不能阻断生成
        logger.warning("构建训练历史上下文失败，按无历史处理: %s", exc)
        return ""


def save_adjustments(athlete_key: str | None, items: list[dict] | None,
                     *, long_term=None) -> bool:
    """保存用户确认采纳的调整项。返回是否保存成功。

    存 Redis 而不是新建一张表：长期记忆本来就是「跨会话、用户级、无 TTL」的
    持久层，语义完全匹配；且它已有把偏好渲染进 Prompt 的机制。少一张表就少
    一处 ORM 漂移面和建表点。
    """
    if not athlete_key or not items:
        return False
    try:
        long_term = long_term if long_term is not None else _default_long_term()
        long_term.save_preference(athlete_key, ADJUSTMENT_KEY, items)
        return True
    except Exception as exc:
        logger.warning("保存训练调整失败: %s", exc)
        return False


def load_adjustments(athlete_key: str | None, *, long_term=None,
                     ttl_days: int = ADJUSTMENT_TTL_DAYS) -> list[dict]:
    """读取仍然有效的已确认调整；过期或不存在时返回空列表。

    直接读 Redis 原始值而非 get_preferences()：后者只返回 value、丢掉了
    时间戳，而过期判断恰恰依赖时间戳。
    """
    if not athlete_key:
        return []
    try:
        long_term = long_term if long_term is not None else _default_long_term()
        raw = long_term.redis.get(f"{long_term.prefix}{athlete_key}:pref:{ADJUSTMENT_KEY}")
        if not raw:
            return []

        data = json.loads(raw)
        if not isinstance(data, dict):
            return []

        if _is_expired(data.get("ts"), ttl_days):
            return []

        value = data.get("v", data)
        return value if isinstance(value, list) else []
    except Exception as exc:
        logger.warning("读取训练调整失败: %s", exc)
        return []


# ----------------------------------------------------------------------
# 渲染
# ----------------------------------------------------------------------

def _render_stats(stats: dict, weeks: int) -> str:
    """把统计渲染成「结论先行、细节在后」的紧凑文本。"""
    totals = stats.get("totals") or {}
    if not totals.get("sessions"):
        return ""

    lines = [f"【训练执行情况（最近 {weeks} 周）】"]
    summary = f"共 {totals['sessions']} 次训练，平均完成率 {totals.get('avg_completion', 0) * 100:.0f}%"
    if totals.get("avg_rpe") is not None:
        summary += f"，平均 RPE {totals['avg_rpe']}"
    lines.append(summary + "。")

    exercises = stats.get("exercises") or []
    if exercises:
        lines.append("主要动作进展：")
        for ex in exercises[:MAX_EXERCISES_IN_CONTEXT]:
            lines.append(f"- {_describe_exercise(ex)}")
    return "\n".join(lines)


def _describe_exercise(ex: dict) -> str:
    """描述单个动作的进展，优先说趋势，再说客观指标。

    趋势以估算 1RM 判定，不能只看重量有没有变：重量相同但次数从 10 掉到 8
    同样是退步，只比重量会把它误描述成"维持"，让模型错过减载信号。
    """
    name = ex.get("name", "")
    parts: list[str] = []

    first_w, last_w = ex.get("first_weight"), ex.get("last_weight")
    trend = ex.get("trend_pct")
    if first_w and last_w:
        if trend is None:
            parts.append(f"维持在 {last_w:g}kg")
        elif trend > 0.5:
            parts.append(f"{first_w:g}kg → {last_w:g}kg（力量 +{trend}%）")
        elif trend < -0.5:
            parts.append(f"力量下滑 {abs(trend)}%（以估算 1RM 计，当前 {last_w:g}kg）")
        else:
            parts.append(f"维持在 {last_w:g}kg")

    if ex.get("avg_rpe") is not None:
        parts.append(f"平均 RPE {ex['avg_rpe']}")
    if ex.get("completion") is not None:
        parts.append(f"完成率 {ex['completion'] * 100:.0f}%")
    parts.append(f"练了 {ex.get('sessions', 0)} 次")

    return f"{name}：" + "，".join(parts)


def _render_adjustments(adjustments: list[dict]) -> str:
    """渲染已确认的调整，让模型知道用户此前做过哪些决定。"""
    lines = ["【用户已确认的调整（应在本次计划中延续）】"]
    for item in adjustments[:MAX_EXERCISES_IN_CONTEXT]:
        if not isinstance(item, dict):
            continue
        name = item.get("exercise_name", "")
        suggested = item.get("suggested") or {}
        kind = item.get("type", "")
        weight = suggested.get("weight")
        if weight:
            lines.append(f"- {name}：目标重量 {weight:g}kg（{_kind_label(kind)}）")
        elif suggested.get("sets_delta"):
            lines.append(f"- {name}：组数调整 {suggested['sets_delta']:+d} 组（{_kind_label(kind)}）")
        elif suggested.get("action"):
            lines.append(f"- {name}：建议替换动作（{_kind_label(kind)}）")
    return "\n".join(lines) if len(lines) > 1 else ""


def _kind_label(kind: str) -> str:
    """把内部类型名转成中文标签，避免内部术语泄漏进 Prompt。"""
    return {
        "progress": "加重",
        "deload": "减载",
        "volume_cut": "减量",
        "swap": "替换",
    }.get(kind, kind)


# ----------------------------------------------------------------------
# 惰性默认实现
# ----------------------------------------------------------------------

def _default_store():
    """惰性构造 TrainingLogStore——避免模块导入时就建立数据库连接。"""
    try:
        from src.storage.training_log_store import TrainingLogStore
        return TrainingLogStore()
    except Exception as exc:
        logger.warning("训练日志存储不可用: %s", exc)
        return None


def _default_long_term():
    """惰性构造 LongTermMemory——避免模块导入时就连接 Redis。"""
    from src.memory.long_term import LongTermMemory
    return LongTermMemory()


def _is_expired(ts: str | None, ttl_days: int) -> bool:
    """判断时间戳是否已超过有效期。无法解析时按「未过期」处理（宁可多注入）。"""
    if not ts:
        return False
    try:
        saved = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if saved.tzinfo is None:
            saved = saved.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - saved).days
        return age_days > ttl_days
    except (ValueError, TypeError):
        return False
