"""Replaceable in-process storage for pending human-review artifacts."""

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4


@dataclass(frozen=True)
class ReviewArtifact:
    review_id: str
    status: str
    created_at: str
    profile_summary: dict
    query: str
    issues: list
    severity: str
    prohibited_actions: list
    #: 工件归属的业务身份键（游客为本机 UUID，登录用户为 u:{id}）。
    #: 默认为空串以兼容本次改动之前落库的旧工件——它们没有归属信息，
    #: 读取侧对这种工件按"仅 uuid4 不可猜"处理（见 app/server.py 的审核路由）。
    owner_key: str = ""


class InMemoryReviewArtifactStore:
    """Single-process artifact store; replace this interface for durable storage."""

    def __init__(self):
        self._artifacts: dict[str, ReviewArtifact] = {}

    def create(self, profile_summary: dict, query: str, issues: list,
               severity: str, prohibited_actions: list,
               owner_key: str = "") -> ReviewArtifact:
        artifact = ReviewArtifact(
            review_id=str(uuid4()),
            status="review_pending",
            created_at=datetime.now(timezone.utc).isoformat(),
            profile_summary=dict(profile_summary or {}),
            query=query or "",
            issues=list(issues or []),
            severity=severity or "warning",
            prohibited_actions=list(prohibited_actions or []),
            owner_key=owner_key or "",
        )
        self._artifacts[artifact.review_id] = artifact
        return artifact

    def get(self, review_id: str) -> ReviewArtifact | None:
        return self._artifacts.get(review_id)

    def delete_by_owner(self, owner_key: str) -> int:
        """删除该身份名下的工件，返回条数。

        只删工件（待处理业务对象），不删已作出的审核决定——后者是
        审计轨迹，不应随用户数据删除。
        """
        if not owner_key:
            return 0
        stale = [rid for rid, art in self._artifacts.items()
                 if getattr(art, "owner_key", "") == owner_key]
        for rid in stale:
            del self._artifacts[rid]
        return len(stale)
