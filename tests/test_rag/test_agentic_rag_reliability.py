from types import SimpleNamespace

import pytest

import src.rag.agentic_rag as agentic_rag_module
from src.rag.agentic_rag import AgenticRAG


def make_rag(vector_results=None, keyword_results=None, eval_results=None):
    rag = AgenticRAG.__new__(AgenticRAG)
    vector_calls = []
    keyword_calls = []
    evaluations = iter(eval_results or [])
    rag.vector = SimpleNamespace(
        search=lambda query, **kwargs: vector_calls.append((query, kwargs))
        or list(vector_results or [])
    )
    rag.keyword = SimpleNamespace(
        search=lambda query, **kwargs: keyword_calls.append((query, kwargs))
        or list(keyword_results or [])
    )
    rag.llm = SimpleNamespace(
        chat_with_json_mode=lambda *args, **kwargs: next(evaluations)
    )
    return rag, vector_calls, keyword_calls


def test_zero_preserves_historical_configured_default_rounds(monkeypatch):
    monkeypatch.setattr(agentic_rag_module, "AGENTIC_RAG_MAX_RETRIES", 2)
    rag, vector_calls, keyword_calls = make_rag(eval_results=[{}, {}])

    assert rag.search("练胸", max_retries=0) == []
    assert [call[0] for call in vector_calls] == ["练胸", "练胸"]
    assert [call[0] for call in keyword_calls] == ["练胸", "练胸"]


def test_negative_max_retries_is_rejected_before_backends_run():
    rag, vector_calls, keyword_calls = make_rag()

    with pytest.raises(ValueError, match="max_retries"):
        rag.search("练胸", max_retries=-1)

    assert vector_calls == []
    assert keyword_calls == []


def test_malformed_backend_rows_are_skipped_with_stable_deduplication():
    first = {"name": "卧推", "source": "vector"}
    duplicate = {"name": "卧推", "source": "keyword"}
    second = {"name": "飞鸟", "source": "keyword"}
    rag, _, _ = make_rag(
        vector_results=[None, {}, {"name": ""}, first],
        keyword_results=[duplicate, {"name": []}, second],
    )

    # RRF 融合会为结果附加 rrf_score 注解，按名称与来源断言融合语义：
    # 双路命中的"卧推"排最前且保留向量路元数据，"飞鸟"次之。
    result = rag.search("练胸", max_retries=1)
    assert [(d["name"], d["source"]) for d in result] == [
        ("卧推", "vector"), ("飞鸟", "keyword")]
    assert result[0]["rrf_score"] > result[1]["rrf_score"]


@pytest.mark.parametrize("malformed_eval", [None, [], "bad", 1])
def test_non_mapping_eval_is_low_quality_and_preserves_current_query(malformed_eval):
    rag, vector_calls, _ = make_rag(eval_results=[malformed_eval])

    assert rag.search("练胸", max_retries=2) == []
    assert [call[0] for call in vector_calls] == ["练胸", "练胸"]


@pytest.mark.parametrize("score", [None, "0.8", "high", [], {}, True])
def test_nonnumeric_score_is_low_quality_without_crashing(score):
    rag, vector_calls, _ = make_rag(
        eval_results=[{"quality_score": score, "rewritten_query": "胸部训练"}]
    )

    assert rag.search("练胸", max_retries=2) == []
    assert [call[0] for call in vector_calls] == ["练胸", "胸部训练"]


@pytest.mark.parametrize("rewrite", [None, 123, [], {}, "", "   "])
def test_invalid_rewrite_preserves_current_query(rewrite):
    rag, vector_calls, _ = make_rag(
        eval_results=[{"quality_score": 0.2, "rewritten_query": rewrite}]
    )

    assert rag.search("练胸", max_retries=2) == []
    assert [call[0] for call in vector_calls] == ["练胸", "练胸"]


def test_malformed_later_rewrite_preserves_last_valid_active_query():
    rag, vector_calls, _ = make_rag(
        eval_results=[
            {"quality_score": 0.2, "rewritten_query": "哑铃胸部训练"},
            {"quality_score": 0.2, "rewritten_query": None},
        ]
    )

    assert rag.search("练胸", max_retries=3) == []
    assert [call[0] for call in vector_calls] == ["练胸", "哑铃胸部训练", "哑铃胸部训练"]


def test_numeric_score_at_threshold_stops_before_another_round():
    row = {"name": "卧推"}
    rag, vector_calls, _ = make_rag(
        vector_results=[row],
        eval_results=[{"quality_score": 0.7, "rewritten_query": "unused"}],
    )

    # RRF 融合附加 rrf_score 注解，按名称断言内容不变。
    assert [d["name"] for d in rag.search("练胸", max_retries=3)] == ["卧推"]
    assert len(vector_calls) == 1
