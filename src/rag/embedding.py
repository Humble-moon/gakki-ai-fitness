"""
=============================================================================
embedding.py — 文本向量化基础服务（阿里云 DashScope 版）
=============================================================================
【项目角色】
    这是整个 RAG 检索体系的最底层服务，为所有上层检索模块提供文本向量化能力。
    使用阿里云 DashScope Embedding API（OpenAI 兼容），无需下载模型、无 HF 依赖。

【相比本地 BGE 模型的优势】
    - 零下载，零缓存，服务秒启动
    - 国内网络直连，稳定低延迟
    - 无需 torch / sentence-transformers 依赖
    - 1024 维向量，比 BGE(512维) 精度更高

【被调用方】
    VectorSearch、KnowledgeSearch、SemanticCache、KnowledgeIngestion
=============================================================================
"""

import numpy as np
from openai import OpenAI
from src.config import (
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
)


class EmbeddingService:
    """文本向量化服务 — 基于 DashScope Embedding API。"""

    def __init__(self):
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = OpenAI(
                api_key=EMBEDDING_API_KEY,
                base_url=EMBEDDING_BASE_URL,
            )
        return self._client

    def embed(self, text: str) -> list:
        """单条文本 → 向量。"""
        resp = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return resp.data[0].embedding

    def embed_batch(self, texts: list) -> list:
        """批量文本 → 向量列表。
        DashScope 单次最多 25 条，这里分批处理。
        """
        all_vecs = []
        batch_size = 20  # 留余量
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
            )
            # 按 index 排序确保顺序
            sorted_data = sorted(resp.data, key=lambda d: d.index)
            all_vecs.extend([d.embedding for d in sorted_data])
        return all_vecs

    def similarity(self, vec1: list, vec2: list) -> float:
        """余弦相似度（DashScope 向量已归一化，等价于点积）。"""
        return float(np.dot(vec1, vec2))
