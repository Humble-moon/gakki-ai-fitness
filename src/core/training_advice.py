"""
自适应调整建议的生成与暂存。

流程定位：位于「统计 → 建议 → 用户确认 → 反哺生成」的中间两环。

与 HITL 的关系：项目原有的 HITL 是「AI 生成 → 拦截 → 人工审核 → 生效」，
本模块是「系统分析 → 生成建议 → 用户确认 → 生效」——同一个确认语义，
因此沿用 HITL 的 artifact 形状（生成待确认工件 → 用户决策 → 状态流转），
而不是另造一套机制。

关键约束：**用户确认调整 ≠ 安全审核通过**。用户勾选「深蹲 +5kg」后，该重量
只是作为下一次生成的输入；若与伤病冲突，FactChecker 依然会拦下并触发 HITL。
本模块不触碰任何安全闸门。
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.core.training_analytics import compute_stats, rule_engine

logger = logging.getLogger(__name__)

# 建议工件的状态
STATUS_PENDING = "advice_pending"
STATUS_APPLIED = "advice_applied"
STATUS_DISMISSED = "advice_dismissed"


@dataclass(frozen=True)
class AdviceArtifact:
    """一次生成的建议集合。形状对照 hitl/review_store.py 的 ReviewArtifact。"""

    advice_id: str
    athlete_key: str
    status: str
    summary: str
    items: list[dict] = field(default_factory=list)
    created_at: str = ""


class InMemoryAdviceStore:
    """建议工件的内存暂存。

    为什么不持久化：建议是「即时生成、当场确认」的短生命周期对象，用户不确认
    就作废；重启后重新生成的成本极低（规则引擎是确定性的纯函数，不调 LLM）。
    对比 HITL 审核工件必须持久化——审核可能跨越很久，且关系到交付安全。
    """

    def __init__(self):
        self._items: dict[str, AdviceArtifact] = {}

    def create(self, athlete_key: str, items: list[dict], summary: str) -> AdviceArtifact:
        artifact = AdviceArtifact(
            advice_id=f"advice_{uuid.uuid4().hex[:12]}",
            athlete_key=athlete_key,
            status=STATUS_PENDING,
            summary=summary,
            items=list(items),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._items[artifact.advice_id] = artifact
        return artifact

    def get(self, advice_id: str) -> AdviceArtifact | None:
        return self._items.get(advice_id)

    def resolve(self, advice_id: str, accepted_item_ids: list[str]) -> AdviceArtifact | None:
        """按用户勾选的条目 id 结算，返回更新后的工件；不存在时返回 None。

        只接受确实存在于该工件中的 item_id——防止前端传来伪造或过期的 id
        把不相干的调整写进长期记忆。
        """
        artifact = self._items.get(advice_id)
        if artifact is None:
            return None

        accepted = set(accepted_item_ids or [])
        kept = [item for item in artifact.items if item.get("item_id") in accepted]

        updated = AdviceArtifact(
            advice_id=artifact.advice_id,
            athlete_key=artifact.athlete_key,
            status=STATUS_APPLIED if kept else STATUS_DISMISSED,
            summary=artifact.summary,
            items=kept,
            created_at=artifact.created_at,
        )
        self._items[advice_id] = updated
        return updated

    def clear(self) -> None:
        """清空暂存（测试用）。"""
        self._items.clear()


# 模块级单例：apply 时要能找到此前生成的工件
advice_store = InMemoryAdviceStore()


def build_advice(athlete_key: str, log_store, *, weeks: int = 8,
                 store: InMemoryAdviceStore | None = None) -> dict:
    """读取训练日志 → 聚合统计 → 推导建议 → 暂存为待确认工件。

    输入：
        athlete_key: 稳定运动员标识
        log_store: TrainingLogStore 实例
        weeks: 参与分析的回溯周数
        store: 建议暂存，缺省用模块级单例

    输出：
        {"advice_id", "status", "summary", "items": [...], "stats": {...}}

    为什么把 stats 一并返回：前端建议面板要显示「依据什么得出这个结论」，
    把统计数字摆在建议旁边能显著提升可信度，也让面试演示时一眼看出
    建议不是凭空生成的。

    注意本函数只生成、不生效——生效必须经过 resolve_advice()。
    """
    store = store if store is not None else advice_store
    logs = log_store.get_logs(athlete_key, weeks=weeks)
    stats = compute_stats(logs)
    items = rule_engine(stats)

    artifact = store.create(athlete_key, items, _summarize(items, stats))
    return {
        "advice_id": artifact.advice_id,
        "status": artifact.status,
        "summary": artifact.summary,
        "items": artifact.items,
        "stats": stats,
    }


def resolve_advice(advice_id: str, accepted_item_ids: list[str], *,
                   long_term=None, store: InMemoryAdviceStore | None = None) -> dict | None:
    """把用户确认采纳的建议写入长期记忆。返回结算结果；工件不存在返回 None。

    写入的目标是长期记忆而非新建表：它天然是「用户级、跨会话、无 TTL」的
    持久层，且已有把偏好渲染进 Prompt 的通道（见 training_history）。
    """
    from src.core.training_history import save_adjustments

    store = store if store is not None else advice_store
    artifact = store.resolve(advice_id, accepted_item_ids)
    if artifact is None:
        return None

    saved = False
    if artifact.items:
        saved = save_adjustments(artifact.athlete_key, artifact.items, long_term=long_term)

    return {
        "advice_id": artifact.advice_id,
        "status": artifact.status,
        "applied": artifact.items,
        "persisted": saved,
        "note": "以上调整已记录，将在下一次生成计划时作为输入生效；"
                "若与伤病冲突，仍会被安全检查拦下。",
    }


def _summarize(items: list[dict], stats: dict) -> str:
    """生成确定性摘要。不调 LLM——即使模型不可用，建议依然可读可用。"""
    if not items:
        totals = stats.get("totals") or {}
        if not totals.get("sessions"):
            return "还没有训练记录，先记录几次训练再来看看。"
        return "最近训练数据比较平稳，暂时没有需要调整的地方。"

    by_type: dict[str, list[str]] = {}
    for item in items:
        by_type.setdefault(item.get("type", ""), []).append(item.get("exercise_name", ""))

    labels = {"deload": "建议减载", "progress": "可以加重",
              "volume_cut": "建议减量", "swap": "建议替换"}
    parts = [f"{labels.get(k, k)}：{'、'.join(v)}" for k, v in by_type.items()]

    totals = stats.get("totals") or {}
    head = (f"根据最近 {totals.get('sessions', 0)} 次训练记录，"
            f"发现 {len(items)} 处可调整：")
    return head + "；".join(parts) + "。"
