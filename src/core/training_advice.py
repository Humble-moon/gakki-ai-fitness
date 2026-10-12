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
                 store: InMemoryAdviceStore | None = None,
                 long_term=None) -> dict:
    """读取训练日志 → 聚合统计 → 推导建议 → 暂存为待确认工件。

    输入：
        athlete_key: 稳定运动员标识
        log_store: TrainingLogStore 实例
        weeks: 参与分析的回溯周数
        store: 建议暂存，缺省用模块级单例
        long_term: LongTermMemory 实例（可选）。仅用于给周报补一句用户处境，
                   不参与任何数值计算。

    输出：
        {"advice_id", "status", "summary", "summary_source", "summary_detail",
         "items": [...], "stats": {...}}

    为什么把 stats 一并返回：前端建议面板要显示「依据什么得出这个结论」，
    把统计数字摆在建议旁边能显著提升可信度，也让面试演示时一眼看出
    建议不是凭空生成的。

    关于 summary 与 summary_detail：
        summary 永远是短文本（有模型时是 headline，无模型时是规则摘要），
        保证任何前端版本都能显示点什么。summary_detail 是结构化的周报正文，
        仅在模型可用时非空；summary_source 显式标注它来自哪条路径——
        降级不该是静默的，用户有权知道这段话是模型写的还是规则拼的。

    注意本函数只生成、不生效——生效必须经过 resolve_advice()。
    """
    store = store if store is not None else advice_store
    logs = log_store.get_logs(athlete_key, weeks=weeks)
    stats = compute_stats(logs)
    items = rule_engine(stats)

    fallback_summary = _summarize(items, stats)
    profile_context = _profile_context(long_term, athlete_key)
    detail, source = _build_report(items, stats, profile_context)

    # 有结构化正文时，短摘要取 headline 与它保持一致；否则退回规则摘要
    summary = detail["headline"] if detail else fallback_summary

    artifact = store.create(athlete_key, items, summary)
    return {
        "advice_id": artifact.advice_id,
        "status": artifact.status,
        "summary": artifact.summary,
        "summary_source": source,
        "summary_detail": detail,
        "items": artifact.items,
        "stats": stats,
    }


def _build_report(items: list[dict], stats: dict,
                  profile_context: str = "") -> tuple[dict | None, str]:
    """尝试用 LLM 生成结构化周报正文。

    返回值：``(detail, source)``。成功时 detail 为规范化后的字典、source 为
    ``"llm"``；任何环节出问题都返回 ``(None, "rule")``，由调用方回退到
    ``_summarize()``。

    为什么吞掉所有异常：周报是**增强层**。项目已经有一条完整的确定性链路
    （compute_stats → rule_engine → _summarize），它不依赖 LLM。模型不可用、
    超时、返回结构不完整，都不该让用户连规则建议都拿不到——这与
    knowledge_search 里 Re-rank 失败就返回未精排结果的取舍一致。

    LLM 不产生数字：所有数值由 compute_stats 算好后注入提示词，模型只做
    组织与解释（见 prompts/training_report.py）。
    """
    try:
        from src.llm.prompts.training_report import build_training_report_messages
        from src.llm.provider import LLMProvider

        messages = build_training_report_messages(stats, items, profile_context)
        detail = _normalize_report(LLMProvider().chat_with_json_mode(messages))
    except Exception as exc:  # noqa: BLE001 - 增强层失败不阻断主链路
        logger.warning("训练周报生成失败，回退规则摘要：%s", exc)
        return None, "rule"

    if detail is None:
        # 模型返回了内容但结构不可用（例如 JSON 解析失败时的 {"raw": ...}），
        # 同样是降级，不能把半成品当周报端出去
        logger.warning("训练周报返回结构不完整，回退规则摘要")
        return None, "rule"

    return detail, "llm"


def _normalize_report(detail) -> dict | None:
    """校验并规范化模型返回的周报结构；不可用时返回 None。

    结构不合格就整体判失败，而不是尽力修补——残缺的周报比诚实的规则摘要
    更容易误导用户。
    """
    if not isinstance(detail, dict):
        return None

    headline = detail.get("headline")
    if not isinstance(headline, str) or not headline.strip():
        return None

    def _entries(key: str) -> list[dict]:
        raw = detail.get(key)
        if not isinstance(raw, list):
            return []
        out = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            text = entry.get("text")
            if not text:
                continue
            out.append({"text": str(text).strip(),
                        "evidence": str(entry.get("evidence") or "").strip()})
        return out

    raw_next = detail.get("next_week")
    next_week = [str(x).strip() for x in raw_next
                 if str(x).strip()] if isinstance(raw_next, list) else []

    return {
        "headline": headline.strip(),
        "highlights": _entries("highlights"),
        "concerns": _entries("concerns"),
        "next_week": next_week,
    }


def _profile_context(long_term, athlete_key: str) -> str:
    """读取长期记忆里的用户处境，供周报口吻贴合；失败返回空串。

    刻意不抛异常：画像只是让周报更贴合，读不到不影响正文生成。
    """
    if long_term is None:
        return ""
    try:
        return long_term.build_context_for_prompt(athlete_key) or ""
    except Exception as exc:  # noqa: BLE001 - 画像缺失是可选降级
        logger.debug("读取长期记忆失败，周报不含画像上下文：%s", exc)
        return ""


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
