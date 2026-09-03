"""Shared RRF fusion: rank semantics, multi-route boost, key handling."""

from src.rag.fusion import rrf_fuse


def test_single_route_preserves_order():
    docs = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    fused = rrf_fuse([docs], key=lambda d: d["name"])
    assert [d["name"] for d in fused] == ["a", "b", "c"]
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"] > fused[2]["rrf_score"]


def test_dual_hit_outranks_single_hit():
    """Doc at rank 2 in both routes beats rank-1 doc seen by only one route."""
    vec = [{"name": "x"}, {"name": "shared"}, {"name": "v3"}]
    kw = [{"name": "y"}, {"name": "shared"}]
    fused = rrf_fuse([vec, kw], key=lambda d: d["name"])
    assert fused[0]["name"] == "shared"
    # shared: 1/(60+2) + 1/(60+2) ≈ 0.0323 > x: 1/(60+1) ≈ 0.0164
    assert fused[0]["rrf_score"] > 1 / 61


def test_none_keys_and_malformed_docs_skipped():
    vec = [{"name": None}, "not-a-dict", {"name": "ok"}]
    fused = rrf_fuse([vec], key=lambda d: d.get("name"))
    assert [d["name"] for d in fused] == ["ok"]


def test_empty_and_invalid_lists_safe():
    assert rrf_fuse([], key=lambda d: d["name"]) == []
    assert rrf_fuse([None, []], key=lambda d: d["name"]) == []


def test_custom_smoothing_constant():
    docs = [{"name": "a"}, {"name": "b"}]
    fused = rrf_fuse([docs], key=lambda d: d["name"], k=1)
    assert fused[0]["rrf_score"] == 0.5   # 1/(1+1)
    assert fused[1]["rrf_score"] == round(1 / 3, 6)
