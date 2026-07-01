from sentence_transformers import SentenceTransformer
import numpy as np
from src.config import EMBEDDING_MODEL


class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def embed(self, text: str) -> list:
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list) -> list:
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return vecs.tolist()

    def similarity(self, vec1: list, vec2: list) -> float:
        return float(np.dot(vec1, vec2))
