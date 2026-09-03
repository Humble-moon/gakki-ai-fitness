"""Durable SQLite-backed stores for the human-review loop.

These are drop-in replacements for the in-process stores in ``review_store.py``
and ``review_resolution.py``: same method signatures, same return types, but the
state survives a process restart. That matters for the review loop — a plan held
at the review gate must still be resolvable after the service restarts, and the
audit trail (who approved what, when) must not live in memory.

Design notes:
* One SQLite file (default ``data/hitl_reviews.db``), three tables. The graph
  checkpoints already live in their own SQLite file; keeping review state
  separate means a checkpoint-store migration never touches the audit trail.
* JSON columns for the structured fields (profile summary, issues, prohibited
  actions) keep the schema stable while the artifact shape evolves.
* A single lock serializes writes; every operation opens its own connection, so
  there is no shared cursor state across threads (FastAPI runs handlers across
  a thread pool).
"""

import json
import sqlite3
import threading
from pathlib import Path

from src.hitl.review_resolution import ReviewResolution
from src.hitl.review_store import ReviewArtifact

DEFAULT_DB_PATH = "data/hitl_reviews.db"


class _SqliteBase:
    """Shared connection plumbing for the three review tables."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self._db_path = db_path
        self._lock = threading.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._connect() as conn:
            self._ensure_schema(conn)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        raise NotImplementedError


class SqliteReviewArtifactStore(_SqliteBase):
    """Durable artifact store; same interface as InMemoryReviewArtifactStore."""

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS review_artifacts (
                review_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                profile_summary TEXT NOT NULL,
                query TEXT NOT NULL,
                issues TEXT NOT NULL,
                severity TEXT NOT NULL,
                prohibited_actions TEXT NOT NULL
            )
        """)

    def create(self, profile_summary: dict, query: str, issues: list,
               severity: str, prohibited_actions: list) -> ReviewArtifact:
        from datetime import datetime, timezone
        from uuid import uuid4
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
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO review_artifacts VALUES (?,?,?,?,?,?,?,?)",
                (artifact.review_id, artifact.status, artifact.created_at,
                 json.dumps(artifact.profile_summary, ensure_ascii=False),
                 artifact.query,
                 json.dumps(artifact.issues, ensure_ascii=False),
                 artifact.severity,
                 json.dumps(artifact.prohibited_actions, ensure_ascii=False)),
            )
        return artifact

    def get(self, review_id: str) -> ReviewArtifact | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM review_artifacts WHERE review_id = ?",
                (review_id,),
            ).fetchone()
        if row is None:
            return None
        return ReviewArtifact(
            review_id=row["review_id"],
            status=row["status"],
            created_at=row["created_at"],
            profile_summary=json.loads(row["profile_summary"]),
            query=row["query"],
            issues=json.loads(row["issues"]),
            severity=row["severity"],
            prohibited_actions=json.loads(row["prohibited_actions"]),
        )

    def list_pending(self) -> list[ReviewArtifact]:
        """All artifacts still awaiting review (status = review_pending)."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT review_id FROM review_artifacts WHERE status = 'review_pending' "
                "ORDER BY created_at"
            ).fetchall()
        return [self.get(r["review_id"]) for r in rows]


class SqliteReviewResolutionStore(_SqliteBase):
    """Durable resolution store; same interface as InMemoryReviewResolutionStore."""

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS review_resolutions (
                review_id TEXT PRIMARY KEY,
                decision TEXT NOT NULL,
                reviewer TEXT NOT NULL,
                comment TEXT NOT NULL,
                resolved_at TEXT NOT NULL
            )
        """)

    def record(self, resolution: ReviewResolution) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO review_resolutions VALUES (?,?,?,?,?)",
                (resolution.review_id, resolution.decision, resolution.reviewer,
                 resolution.comment, resolution.resolved_at),
            )

    def get(self, review_id: str) -> ReviewResolution | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM review_resolutions WHERE review_id = ?",
                (review_id,),
            ).fetchone()
        if row is None:
            return None
        return ReviewResolution(
            review_id=row["review_id"],
            decision=row["decision"],
            reviewer=row["reviewer"],
            comment=row["comment"],
            resolved_at=row["resolved_at"],
        )


class SqliteReviewThreadIndex(_SqliteBase):
    """Durable review_id -> thread_id map; survives restarts so the resolve
    endpoint no longer depends on the client passing thread_id back."""

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS review_threads (
                review_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL
            )
        """)

    def register(self, review_id: str, thread_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO review_threads VALUES (?,?)",
                (review_id, thread_id),
            )

    def thread_for(self, review_id: str) -> str | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT thread_id FROM review_threads WHERE review_id = ?",
                (review_id,),
            ).fetchone()
        return row["thread_id"] if row else None
