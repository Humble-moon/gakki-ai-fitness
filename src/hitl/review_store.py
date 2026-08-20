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


class InMemoryReviewArtifactStore:
    """Single-process artifact store; replace this interface for durable storage."""

    def __init__(self):
        self._artifacts: dict[str, ReviewArtifact] = {}

    def create(self, profile_summary: dict, query: str, issues: list,
               severity: str, prohibited_actions: list) -> ReviewArtifact:
        artifact = ReviewArtifact(
            review_id=str(uuid4()),
            status="review_pending",
            created_at=datetime.now(timezone.utc).isoformat(),
            profile_summary=dict(profile_summary or {}),
            query=query or "",
            issues=list(issues or []),
            severity=severity or "warning",
            prohibited_actions=list(prohibited_actions or []),
        )
        self._artifacts[artifact.review_id] = artifact
        return artifact

    def get(self, review_id: str) -> ReviewArtifact | None:
        return self._artifacts.get(review_id)
