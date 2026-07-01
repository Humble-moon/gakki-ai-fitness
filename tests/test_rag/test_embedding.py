import pytest
from src.rag.embedding import EmbeddingService
from src.config import EMBEDDING_DIM


@pytest.fixture(scope="module")
def emb():
    return EmbeddingService()


class TestEmbeddingService:
    def test_embed_query_returns_correct_dim(self, emb):
        vec = emb.embed("哑铃卧推")
        assert len(vec) == EMBEDDING_DIM
        assert isinstance(vec[0], float)

    def test_embed_batch(self, emb):
        texts = ["深蹲", "硬拉", "卧推"]
        vectors = emb.embed_batch(texts)
        assert len(vectors) == 3
        assert all(len(v) == EMBEDDING_DIM for v in vectors)

    def test_similarity_same_text(self, emb):
        v1 = emb.embed("增肌训练")
        v2 = emb.embed("增肌训练")
        sim = emb.similarity(v1, v2)
        assert sim > 0.95

    def test_similarity_different_text(self, emb):
        v1 = emb.embed("增肌训练")
        v2 = emb.embed("有氧减脂")
        sim = emb.similarity(v1, v2)
        assert sim < 0.95
