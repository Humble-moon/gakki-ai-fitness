"""Optional sparse (lexical) retrieval route and DashScope native embedding mode.

Covers the guarantees that matter for a default-off optional path:
  * defaults leave retrieval behaviour byte-identical to the historical version;
  * gating refuses to engage unless native mode + sparse output are both on;
  * scoring, zero-overlap dropping and top_k truncation behave as documented;
  * the third route folds into the same shared RRF semantics;
  * the native payload carries text_type / instruct / output_type correctly.
"""

import json

import pytest

import src.config as config
import src.rag.embedding as embedding_mod
from src.rag.embedding import EmbeddingService, sparse_dot, _normalize_sparse
from src.rag.fusion import rrf_fuse
from src.rag.sparse_search import SparseSearch


# --------------------------------------------------------------------------
# test doubles
# --------------------------------------------------------------------------
class FakePG:
    """Records every statement; returns canned rows for fetch_all."""

    def __init__(self, rows=None):
        self.rows = list(rows or [])
        self.executed = []
        self.fetch_calls = 0

    def execute(self, query, params=None):
        self.executed.append((query, params or {}))

    def fetch_all(self, query, params=None):
        self.fetch_calls += 1
        return list(self.rows)


class FakeEmb:
    """Stands in for EmbeddingService with a controllable sparse payload."""

    def __init__(self, query_sparse, sparse_enabled=True):
        self._query_sparse = query_sparse
        self.sparse_enabled = sparse_enabled
        self.calls = []

    def embed_with_sparse(self, text, text_type=None):
        self.calls.append((text, text_type))
        return ([0.1, 0.2], self._query_sparse)


def make_search(query_sparse, rows, enabled=True, sparse_enabled=True):
    pg, emb = FakePG(rows), FakeEmb(query_sparse, sparse_enabled)
    search = SparseSearch(pg=pg, emb=emb)
    search.enabled = enabled
    search._table_ready = True  # skip DDL against a real database
    return search, pg, emb


def corpus_row(chunk_id, sparse, title="t", content="c", source="s.md", index=0):
    """One row shaped exactly like the JOIN in SparseSearch._load_all."""
    return (chunk_id, sparse, title, content, source, index)


# --------------------------------------------------------------------------
# defaults: nothing changes unless explicitly opted in
# --------------------------------------------------------------------------
def test_default_config_keeps_compat_mode_and_sparse_off():
    assert config.EMBEDDING_API_MODE == "compatible"
    assert config.EMBEDDING_OUTPUT_TYPE == "dense"
    assert config.SPARSE_RETRIEVAL == "off"

    emb = EmbeddingService()
    assert emb.api_mode == "compatible"
    assert emb.native_enabled is False
    assert emb.sparse_enabled is False
    assert SparseSearch(pg=FakePG(), emb=FakeEmb(None)).enabled is False


def test_native_mode_alone_does_not_enable_sparse():
    """output_type must also contain sparse — native alone is not enough."""
    emb = EmbeddingService(api_mode="native")
    assert emb.native_enabled is True
    assert emb.sparse_enabled is False


def test_sparse_requires_both_native_mode_and_sparse_output(monkeypatch):
    monkeypatch.setattr(embedding_mod, "EMBEDDING_OUTPUT_TYPE", "dense&sparse")
    emb = EmbeddingService(api_mode="native")
    emb.output_type = "dense&sparse"
    assert emb.sparse_enabled is True


# --------------------------------------------------------------------------
# sparse vector maths and wire-format parsing
# --------------------------------------------------------------------------
def test_sparse_dot_overlapping_indices_only():
    assert sparse_dot({1: 0.5, 7: 0.2}, {7: 1.0, 9: 0.3}) == pytest.approx(0.2)


def test_sparse_dot_is_symmetric_and_iterates_shorter_side():
    a, b = {1: 0.4, 2: 0.6}, {2: 0.5, 3: 0.5, 4: 0.1, 5: 0.1}
    assert sparse_dot(a, b) == pytest.approx(sparse_dot(b, a))
    assert sparse_dot(a, b) == pytest.approx(0.3)


@pytest.mark.parametrize("q,d", [(None, {1: 1.0}), ({1: 1.0}, None), ({}, {}), (None, None)])
def test_sparse_dot_empty_sides_mean_no_overlap_not_error(q, d):
    assert sparse_dot(q, d) == 0.0


def test_normalize_sparse_drops_token_and_bad_entries():
    item = {"sparse_embedding": [
        {"index": 7149, "value": 0.829, "token": "风"},
        {"index": "111", "value": "0.5"},        # stringy but coercible
        {"index": None, "value": 0.9},           # dropped
        {"index": 5, "value": None},             # dropped
        "not-a-dict",                            # dropped
    ]}
    assert _normalize_sparse(item) == {7149: 0.829, 111: 0.5}


def test_normalize_sparse_returns_none_when_absent():
    assert _normalize_sparse({}) is None
    assert _normalize_sparse({"sparse_embedding": []}) is None


# --------------------------------------------------------------------------
# gating
# --------------------------------------------------------------------------
def test_available_false_when_switch_off():
    search, _, _ = make_search({1: 1.0}, [], enabled=False)
    assert search.available() is False
    assert search.search("膝盖疼怎么练") == []


def test_available_false_when_embedding_side_not_ready():
    """SPARSE_RETRIEVAL=on without native+sparse must degrade, never raise."""
    search, _, _ = make_search({1: 1.0}, [], enabled=True, sparse_enabled=False)
    assert search.available() is False
    assert search.search("膝盖疼怎么练") == []


def test_available_true_when_all_conditions_met():
    search, _, _ = make_search({1: 1.0}, [], enabled=True, sparse_enabled=True)
    assert search.available() is True


# --------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------
def test_search_ranks_by_sparse_dot_and_shapes_like_other_routes():
    rows = [
        corpus_row("c1", {10: 1.0}, title="膝关节康复", content="膝盖"),
        corpus_row("c2", {99: 1.0}, title="无关", content="饮食"),
        corpus_row("c3", {10: 0.4, 11: 0.4}, title="部分重叠"),
    ]
    search, _, emb = make_search({10: 1.0}, rows)

    results = search.search("膝盖疼")

    # c2 has no lexical overlap → dropped entirely, not ranked last
    assert [r["chunk_id"] for r in results] == ["c1", "c3"]
    assert results[0]["score"] == pytest.approx(1.0)
    assert results[0]["source"] == "sparse"
    assert results[0]["title"] == "膝关节康复"
    assert results[0]["content"] == "膝盖"
    assert emb.calls == [("膝盖疼", "query")]  # query side, asymmetric encoding


def test_search_result_keys_match_vector_and_keyword_routes():
    rows = [corpus_row("c1", {10: 1.0}, title="T", content="C", source="f.md", index=3)]
    search, _, _ = make_search({10: 1.0}, rows)

    (result,) = search.search("q")
    assert set(result) == {"chunk_id", "title", "content", "source_file",
                           "chunk_index", "score", "source"}
    assert result["source_file"] == "f.md"
    assert result["chunk_index"] == 3


def test_search_respects_top_k():
    rows = [corpus_row(f"c{i}", {10: 1.0 - i * 0.01}) for i in range(10)]
    search, _, _ = make_search({10: 1.0}, rows)

    results = search.search("q", top_k=3)
    assert [r["chunk_id"] for r in results] == ["c0", "c1", "c2"]


def test_search_empty_corpus_returns_empty_without_raising():
    search, _, _ = make_search({10: 1.0}, [])
    assert search.search("q") == []


def test_search_survives_query_embedding_failure():
    search, _, emb = make_search({10: 1.0}, [corpus_row("c1", {10: 1.0})])

    def boom(text, text_type=None):
        raise RuntimeError("native embedding 500")

    emb.embed_with_sparse = boom
    assert search.search("q") == []  # degrades to the two existing routes


def test_search_survives_pg_failure():
    search, pg, _ = make_search({10: 1.0}, [corpus_row("c1", {10: 1.0})])

    def boom(query, params=None):
        raise RuntimeError("connection refused")

    pg.fetch_all = boom
    assert search.search("q") == []


def test_jsonb_string_payload_is_parsed():
    """psycopg may hand back JSONB as str depending on driver/version."""
    rows = [corpus_row("c1", json.dumps({10: 0.7}))]
    search, _, _ = make_search({10: 1.0}, rows)

    (result,) = search.search("q")
    assert result["score"] == pytest.approx(0.7)


def test_cache_is_reused_then_dropped_on_invalidate():
    rows = [corpus_row("c1", {10: 1.0})]
    search, pg, _ = make_search({10: 1.0}, rows)

    search.search("q")
    search.search("q")
    assert pg.fetch_calls == 1          # second call served from the process cache

    search.invalidate()
    assert search._cache is None
    search.search("q")
    assert pg.fetch_calls == 2          # cache drop forces exactly one reload


def test_upsert_writes_jsonb_and_invalidates_cache():
    search, pg, _ = make_search({10: 1.0}, [])
    search._cache = {"stale": {}}

    assert search.upsert("c1", {10: 0.5, 12: 0.25}) is True
    assert search._cache is None

    query, params = pg.executed[-1]
    assert "knowledge_chunks_sparse" in query
    assert "ON CONFLICT (chunk_id) DO UPDATE" in query
    assert params["cid"] == "c1"
    assert json.loads(params["payload"]) == {"10": 0.5, "12": 0.25}


def test_upsert_skips_empty_sparse():
    search, pg, _ = make_search(None, [])
    assert search.upsert("c1", None) is False
    assert search.upsert("c1", {}) is False
    assert pg.executed == []


# --------------------------------------------------------------------------
# fusion: the third route uses the exact same shared RRF
# --------------------------------------------------------------------------
def test_three_route_fusion_boosts_doc_hit_by_all_routes():
    vec = [{"chunk_id": "a"}, {"chunk_id": "shared"}]
    kw = [{"chunk_id": "b"}, {"chunk_id": "shared"}]
    sparse = [{"chunk_id": "shared"}, {"chunk_id": "c"}]

    fused = rrf_fuse([vec, kw, sparse], key=lambda d: d["chunk_id"])
    assert fused[0]["chunk_id"] == "shared"
    # three routes → strictly higher than the two-route score for the same ranks
    two_route = rrf_fuse([vec, kw], key=lambda d: d["chunk_id"])
    assert fused[0]["rrf_score"] > two_route[0]["rrf_score"]


def test_knowledge_search_rrf_fusion_accepts_optional_third_route():
    """Constructed without __init__ so no PG/LLM is required."""
    from src.rag.knowledge_search import KnowledgeSearch

    ks = KnowledgeSearch.__new__(KnowledgeSearch)
    vec = [{"chunk_id": "a"}, {"chunk_id": "x"}]
    kw = [{"chunk_id": "b"}]

    dual = ks.rrf_fusion(vec, kw)
    triple = ks.rrf_fusion(vec, kw, [{"chunk_id": "a"}])
    empty_third = ks.rrf_fusion(vec, kw, [])

    assert empty_third == dual                       # [] ≡ two-route behaviour
    assert next(d for d in triple if d["chunk_id"] == "a")["rrf_score"] > \
           next(d for d in dual if d["chunk_id"] == "a")["rrf_score"]


def test_knowledge_search_rrf_fusion_positional_call_still_two_route():
    """The pre-existing call shape rrf_fusion(vec, kw) must keep working."""
    from src.rag.knowledge_search import KnowledgeSearch

    ks = KnowledgeSearch.__new__(KnowledgeSearch)
    fused = ks.rrf_fusion([{"chunk_id": "a"}], [{"chunk_id": "b"}])
    assert [d["chunk_id"] for d in fused] == ["a", "b"]


# --------------------------------------------------------------------------
# native payload construction
# --------------------------------------------------------------------------
class FakeResponse:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body or {}
        self.text = text

    def json(self):
        return self._body


def native_body(n=1, sparse=True):
    embeddings = []
    for i in range(n):
        item = {"text_index": i, "embedding": [0.1, 0.2]}
        if sparse:
            item["sparse_embedding"] = [{"index": 7, "value": 0.9, "token": "膝"}]
        embeddings.append(item)
    return {"output": {"embeddings": embeddings}, "usage": {"total_tokens": 12}}


def patch_native(monkeypatch, response):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(url=url, payload=json, headers=headers, timeout=timeout)
        return response

    monkeypatch.setattr("httpx.post", fake_post)
    monkeypatch.setattr("src.llm.cost_tracker.cost_tracker.record", lambda *a, **k: None)
    return captured


def test_native_payload_carries_text_type_instruct_and_output_type(monkeypatch):
    captured = patch_native(monkeypatch, FakeResponse(body=native_body()))
    monkeypatch.setattr(embedding_mod, "EMBEDDING_API_KEY", "sk-test")
    monkeypatch.setattr(embedding_mod, "EMBEDDING_INSTRUCT_QUERY", "retrieve fitness chunks")

    emb = EmbeddingService(api_mode="native")
    emb.output_type = "dense&sparse"
    dense, sparse = emb.embed_with_sparse("膝盖疼", text_type="query")

    params = captured["payload"]["parameters"]
    assert params["text_type"] == "query"
    assert params["output_type"] == "dense&sparse"
    assert params["instruct"] == "retrieve fitness chunks"
    assert "dimension" not in params            # 1024 is the API default → omitted
    assert captured["payload"]["input"] == {"texts": ["膝盖疼"]}
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert dense == [0.1, 0.2]
    assert sparse == {7: 0.9}


def test_native_omits_absent_optional_params(monkeypatch):
    captured = patch_native(monkeypatch, FakeResponse(body=native_body(sparse=False)))
    monkeypatch.setattr(embedding_mod, "EMBEDDING_API_KEY", "sk-test")
    monkeypatch.setattr(embedding_mod, "EMBEDDING_INSTRUCT_DOCUMENT", "")

    emb = EmbeddingService(api_mode="native")
    dense, sparse = emb.embed_with_sparse("文档正文", text_type="document")

    params = captured["payload"]["parameters"]
    assert params["text_type"] == "document"
    assert "instruct" not in params             # unconfigured → not sent
    assert sparse is None                       # API returned dense only


def test_native_sends_dimension_only_when_non_default(monkeypatch):
    captured = patch_native(monkeypatch, FakeResponse(body=native_body()))
    monkeypatch.setattr(embedding_mod, "EMBEDDING_API_KEY", "sk-test")
    monkeypatch.setattr(embedding_mod, "EMBEDDING_DIM", 512)

    EmbeddingService(api_mode="native").embed("q", text_type="query")
    assert captured["payload"]["parameters"]["dimension"] == 512


def test_native_batches_at_api_limit(monkeypatch):
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(len(json["input"]["texts"]))
        return FakeResponse(body=native_body(n=len(json["input"]["texts"])))

    monkeypatch.setattr("httpx.post", fake_post)
    monkeypatch.setattr("src.llm.cost_tracker.cost_tracker.record", lambda *a, **k: None)
    monkeypatch.setattr(embedding_mod, "EMBEDDING_API_KEY", "sk-test")

    out = EmbeddingService(api_mode="native").embed_batch([f"t{i}" for i in range(23)])
    assert calls == [10, 10, 3]                 # batch size 10 is the hard API limit
    assert len(out) == 23


def test_native_preserves_input_order_via_text_index(monkeypatch):
    body = {"output": {"embeddings": [
        {"text_index": 1, "embedding": [1.0]},
        {"text_index": 0, "embedding": [0.0]},
    ]}, "usage": {"total_tokens": 4}}
    patch_native(monkeypatch, FakeResponse(body=body))
    monkeypatch.setattr(embedding_mod, "EMBEDDING_API_KEY", "sk-test")

    assert EmbeddingService(api_mode="native").embed_batch(["a", "b"]) == [[0.0], [1.0]]


def test_native_failure_raises_instead_of_silently_degrading(monkeypatch):
    """Mixing native query vectors with compatible doc vectors would corrupt
    retrieval silently — so a native failure must surface, and the existing
    keyword fallback in the retrieval chain takes over."""
    patch_native(monkeypatch, FakeResponse(status_code=400, text="InvalidParameter"))
    monkeypatch.setattr(embedding_mod, "EMBEDDING_API_KEY", "sk-test")

    with pytest.raises(RuntimeError, match="400"):
        EmbeddingService(api_mode="native").embed("q", text_type="query")


def test_native_empty_result_raises(monkeypatch):
    patch_native(monkeypatch, FakeResponse(body={"output": {"embeddings": []}}))
    monkeypatch.setattr(embedding_mod, "EMBEDDING_API_KEY", "sk-test")

    with pytest.raises(RuntimeError, match="空结果"):
        EmbeddingService(api_mode="native").embed("q", text_type="query")


def test_compatible_mode_never_calls_native_endpoint(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("compatible mode must not touch the native endpoint")

    monkeypatch.setattr("httpx.post", boom)

    emb = EmbeddingService()
    assert emb.native_enabled is False
    # text_type is accepted but ignored — the compatible API has no such param
    assert emb.instruct_for("query") == ""
