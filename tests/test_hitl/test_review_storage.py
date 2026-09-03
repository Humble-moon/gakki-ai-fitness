"""SQLite-backed review stores: durability, interface parity, restart survival."""

from src.hitl.review_resolution import make_resolution
from src.hitl.review_storage import (SqliteReviewArtifactStore,
                                     SqliteReviewResolutionStore,
                                     SqliteReviewThreadIndex)


def _make_artifact(store):
    return store.create(
        profile_summary={"height": 178, "injuries": ["腰椎间盘突出"]},
        query="我想练硬拉",
        issues=[{"code": "INJURY_CONFLICT", "detail": "腰椎间盘突出 × 硬拉"}],
        severity="danger",
        prohibited_actions=["审核完成前不要开始执行该计划"],
    )


def test_artifact_roundtrip(tmp_path):
    store = SqliteReviewArtifactStore(str(tmp_path / "hitl.db"))
    artifact = _make_artifact(store)
    fetched = store.get(artifact.review_id)
    assert fetched == artifact
    assert fetched.profile_summary["injuries"] == ["腰椎间盘突出"]
    assert store.get("missing-id") is None


def test_artifact_survives_reopen(tmp_path):
    """A new store instance on the same file sees the old state — restart case."""
    db = str(tmp_path / "hitl.db")
    artifact = _make_artifact(SqliteReviewArtifactStore(db))
    reopened = SqliteReviewArtifactStore(db)
    assert reopened.get(artifact.review_id) is not None
    assert reopened.list_pending()[0].review_id == artifact.review_id


def test_resolution_roundtrip_and_reopen(tmp_path):
    db = str(tmp_path / "hitl.db")
    store = SqliteReviewResolutionStore(db)
    resolution = make_resolution("r-1", "approved", reviewer="ye-long",
                                 comment="计划合理，放行")
    store.record(resolution)
    assert store.get("r-1") == resolution
    assert SqliteReviewResolutionStore(db).get("r-1").decision == "approved"
    assert store.get("missing") is None


def test_thread_index_roundtrip_and_reopen(tmp_path):
    db = str(tmp_path / "hitl.db")
    index = SqliteReviewThreadIndex(db)
    index.register("r-1", "thread-abc")
    assert index.thread_for("r-1") == "thread-abc"
    assert SqliteReviewThreadIndex(db).thread_for("r-1") == "thread-abc"
    assert index.thread_for("missing") is None
    # re-register overwrites rather than duplicates
    index.register("r-1", "thread-def")
    assert index.thread_for("r-1") == "thread-def"
