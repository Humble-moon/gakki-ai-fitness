"""Unit tests for the human-review resolution structures."""

from src.hitl.review_resolution import (
    APPROVED,
    REJECTED,
    InMemoryReviewResolutionStore,
    ReviewThreadIndex,
    make_resolution,
)


def test_make_resolution_records_utc_timestamp():
    res = make_resolution("r1", APPROVED, reviewer="coach", comment="looks safe")
    assert res.review_id == "r1"
    assert res.decision == APPROVED
    assert res.reviewer == "coach"
    assert res.comment == "looks safe"
    assert res.resolved_at  # non-empty ISO8601 timestamp


def test_resolution_store_record_and_get():
    store = InMemoryReviewResolutionStore()
    assert store.get("missing") is None
    res = make_resolution("r1", REJECTED)
    store.record(res)
    assert store.get("r1") is res


def test_resolution_store_last_write_wins():
    store = InMemoryReviewResolutionStore()
    store.record(make_resolution("r1", APPROVED))
    store.record(make_resolution("r1", REJECTED))
    assert store.get("r1").decision == REJECTED


def test_thread_index_register_and_lookup():
    index = ReviewThreadIndex()
    assert index.thread_for("missing") is None
    index.register("r1", "thread-abc")
    assert index.thread_for("r1") == "thread-abc"
