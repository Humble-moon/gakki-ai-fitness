"""Reciprocal Rank Fusion — the shared implementation for all retrieval routes.

RRF merges independently ranked lists without calibrating their raw scores:
``score(d) = sum(1 / (k + rank_i(d)))`` over every route that returned ``d``.
Rank-only fusion is robust to the fact that a vector similarity of 0.9 and a
keyword similarity of 0.9 live on different scales.

This module is the single source of truth for RRF in the project:
* ``KnowledgeSearch.rrf_fusion`` (knowledge chunks, keyed by chunk_id)
* ``AgenticRAG`` exercise fusion (keyed by exercise name)
both delegate here, so the knowledge pipeline and the exercise pipeline use
the exact same fusion semantics.
"""

RRF_K = 60  # smoothing constant; dampens the head of each ranked list


def rrf_fuse(ranked_lists: list[list[dict]], key, k: int = RRF_K) -> list[dict]:
    """Fuse ranked result lists into one list ordered by RRF score.

    Inputs:
        ranked_lists: each inner list is already sorted best-first
        key:          callable(doc) -> hashable identity (chunk_id, name, ...)
        k:            RRF smoothing constant, default 60

    Output:
        list[dict] — copies of the input docs sorted by descending RRF score,
        each annotated with ``rrf_score``. For docs returned by several routes
        the doc payload of the last occurrence wins (routes are passed in the
        same order as the legacy inline implementation).
    """
    scores: dict = {}
    docs: dict = {}
    for ranked in ranked_lists:
        if not isinstance(ranked, list):
            continue
        for rank, doc in enumerate(ranked, start=1):
            if not isinstance(doc, dict):
                continue
            doc_key = key(doc)
            if doc_key is None:
                continue
            try:
                scores[doc_key] = scores.get(doc_key, 0.0) + 1.0 / (k + rank)
            except TypeError:
                continue  # unhashable key (malformed row) → skip, never crash
            docs[doc_key] = doc
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    fused = []
    for doc_key, score in ordered:
        merged = dict(docs[doc_key])
        merged["rrf_score"] = round(score, 6)
        fused.append(merged)
    return fused
