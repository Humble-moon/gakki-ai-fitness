"""Structures that close the human-review loop for the LangGraph pipeline.

The legacy system creates a ``ReviewArtifact`` when a plan is held for review but
has no way to record the review's outcome — the loop is open-ended. The graph
pipeline completes the loop with these two stores:

* :class:`InMemoryReviewResolutionStore` records *how* a held review was resolved
  (approved / rejected, by whom, with what comment, and when).
* :class:`ReviewThreadIndex` maps a ``review_id`` to the LangGraph ``thread_id``
  whose execution is paused at the review gate, so the resolve endpoint can resume
  the right checkpointed run.

``ReviewArtifact`` itself stays frozen and untouched; resolution is additive state.
"""

from dataclasses import dataclass
from datetime import datetime, timezone

APPROVED = "approved"
REJECTED = "rejected"
VALID_DECISIONS = (APPROVED, REJECTED)


@dataclass(frozen=True)
class ReviewResolution:
    """The recorded outcome of a human review."""
    review_id: str
    decision: str            # "approved" | "rejected"
    reviewer: str
    comment: str
    resolved_at: str         # ISO8601 UTC timestamp


class InMemoryReviewResolutionStore:
    """Single-process resolution store; swap for durable storage in production."""

    def __init__(self):
        self._resolutions: dict[str, ReviewResolution] = {}

    def record(self, resolution: ReviewResolution) -> None:
        self._resolutions[resolution.review_id] = resolution

    def get(self, review_id: str) -> ReviewResolution | None:
        return self._resolutions.get(review_id)


class ReviewThreadIndex:
    """Maps review_id -> LangGraph thread_id for resuming paused runs."""

    def __init__(self):
        self._threads: dict[str, str] = {}

    def register(self, review_id: str, thread_id: str) -> None:
        self._threads[review_id] = thread_id

    def thread_for(self, review_id: str) -> str | None:
        return self._threads.get(review_id)


def make_resolution(review_id: str, decision: str, reviewer: str = "",
                    comment: str = "") -> ReviewResolution:
    """Build a resolution with a current UTC timestamp."""
    return ReviewResolution(
        review_id=review_id,
        decision=decision,
        reviewer=reviewer or "",
        comment=comment or "",
        resolved_at=datetime.now(timezone.utc).isoformat(),
    )
