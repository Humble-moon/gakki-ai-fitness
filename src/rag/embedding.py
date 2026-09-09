"""
=============================================================================
embedding.py — 文本向量化基础服务（阿里云 DashScope 版）
=============================================================================
【项目角色】
    这是整个 RAG 检索体系的最底层服务，为所有上层检索模块提供文本向量化能力。
    使用阿里云 DashScope Embedding API，无需下载模型、无 HF 依赖。

【相比本地 BGE 模型的优势】
    - 零下载，零缓存，服务秒启动
    - 国内网络直连，稳定低延迟
    - 无需 torch / sentence-transformers 依赖
    - 1024 维向量，比 BGE(512维) 精度更高

【两套调用模式 — EMBEDDING_API_MODE】
    compatible（默认）
        OpenAI 兼容端点 /compatible-mode/v1/embeddings，复用 openai SDK 客户端。
        只接受 model / input / dimensions / encoding_format。
    native
        DashScope 原生端点 /api/v1/services/embeddings/text-embedding/text-embedding。
        阿里云官方文档明确：稠密/稀疏向量（output_type）、text_type、instruct
        这三项「仅支持通过 DashScope SDK 及 API 启用」——兼容端点拿不到。
        因此以下三项能力全部挂在 native 模式：
          1. text_type="query" / "document" 非对称编码（检索质量直接杠杆）
          2. instruct 任务指令（官方称通常带来 1%~5% 提升，建议英文撰写）
          3. output_type="dense&sparse" 一次调用同时拿稠密 + 类 SPLADE 稀疏向量

    ⚠️ 切换到 native 会改变向量取值，属于影响检索指标的变更：
       必须重新摄入知识库（query 侧与 document 侧要用同一模式），
       并重跑消融后再与历史数字对比，不可混用两种模式的向量。

【为什么 native 失败不静默降级到 compatible】
    两种模式产出的向量不在同一语义空间（text_type/instruct 会改变编码结果）。
    若查询侧走 native、文档侧因偶发失败落到 compatible，检索会得到看似正常
    实则错乱的结果——这种静默错误比直接失败更难排查。因此 native 调用失败
    一律抛出，由上层检索链路既有的兜底逻辑（关键词检索 / 三段式降级）接管。

【被调用方】
    VectorSearch、KnowledgeSearch、SparseSearch、SemanticCache、KnowledgeIngestion
=============================================================================
"""

import numpy as np
from openai import OpenAI
from src.config import (
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    EMBEDDING_API_MODE,
    EMBEDDING_NATIVE_URL,
    EMBEDDING_INSTRUCT_QUERY,
    EMBEDDING_INSTRUCT_DOCUMENT,
    EMBEDDING_OUTPUT_TYPE,
)
from src.llm.cost_tracker import cost_tracker

# DashScope 原生/兼容两套接口共用的单批上限（text-embedding-v4 批次大小 = 10，
# 单条文本上限 8192 token）。超过即由调用方分批，这里做最后一道保护。
BATCH_LIMIT = 10

# 原生端点超时：连接 5s、整体 30s。embedding 是检索链路的第一跳，
# 超时后应由上层降级到关键词检索，而不是把整个请求拖死。
NATIVE_TIMEOUT = 30.0


class EmbeddingService:
    """文本向量化服务 — 基于 DashScope Embedding API。

    对上层暴露的稳定接口：
        embed(text, text_type=None)            -> list[float]
        embed_batch(texts, text_type=None)     -> list[list[float]]
        embed_with_sparse(text, text_type)     -> (list[float], dict|None)
        embed_batch_with_sparse(texts, tt)     -> list[(list[float], dict|None)]
        similarity(v1, v2)                     -> float

    text_type 取值："query" | "document" | None。
    仅在 native 模式下真正下发给 API；compatible 模式下忽略（API 不支持该参数）。
    """

    def __init__(self, api_mode: str | None = None):
        self._client = None
        # 允许测试注入模式，生产从 config 读
        self.api_mode = (api_mode or EMBEDDING_API_MODE or "compatible").lower()
        self.output_type = (EMBEDDING_OUTPUT_TYPE or "dense").lower()

    # ------------------------------------------------------------------
    # 客户端
    # ------------------------------------------------------------------
    @property
    def client(self):
        if self._client is None:
            self._client = OpenAI(
                api_key=EMBEDDING_API_KEY,
                base_url=EMBEDDING_BASE_URL,
            )
        return self._client

    @property
    def native_enabled(self) -> bool:
        return self.api_mode == "native"

    @property
    def sparse_enabled(self) -> bool:
        """稀疏向量是否可用：必须同时满足 native 模式 + output_type 含 sparse。"""
        return self.native_enabled and "sparse" in self.output_type

    def instruct_for(self, text_type: str | None) -> str:
        """按 text_type 选出对应的任务指令；未配置则返回空串（不下发该参数）。"""
        if text_type == "query":
            return EMBEDDING_INSTRUCT_QUERY or ""
        if text_type == "document":
            return EMBEDDING_INSTRUCT_DOCUMENT or ""
        return ""

    # ------------------------------------------------------------------
    # 对外接口：稠密向量
    # ------------------------------------------------------------------
    def embed(self, text: str, text_type: str | None = None) -> list:
        """单条文本 → 向量。"""
        return self.embed_batch([text], text_type=text_type)[0]

    def embed_batch(self, texts: list, text_type: str | None = None) -> list:
        """批量文本 → 向量列表。单次最多 10 条，超出自动分批。"""
        return [dense for dense, _ in self.embed_batch_with_sparse(texts, text_type)]

    # ------------------------------------------------------------------
    # 对外接口：稠密 + 稀疏
    # ------------------------------------------------------------------
    def embed_with_sparse(self, text: str, text_type: str | None = None):
        """单条文本 → (稠密向量, 稀疏向量 dict|None)。

        稀疏向量形如 {token_index: weight}；不可用时为 None。
        """
        return self.embed_batch_with_sparse([text], text_type)[0]

    def embed_batch_with_sparse(self, texts: list, text_type: str | None = None) -> list:
        """批量文本 → [(稠密向量, 稀疏向量|None), ...]。

        compatible 模式下稀疏项恒为 None（该端点不返回稀疏向量）。
        """
        if not texts:
            return []

        pairs: list = []
        for i in range(0, len(texts), BATCH_LIMIT):
            batch = list(texts[i:i + BATCH_LIMIT])
            if self.native_enabled:
                pairs.extend(self._call_native(batch, text_type))
            else:
                pairs.extend(self._call_compatible(batch))
        return pairs

    # ------------------------------------------------------------------
    # compatible 模式（默认，行为与历史版本一致）
    # ------------------------------------------------------------------
    def _call_compatible(self, batch: list) -> list:
        resp = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        tokens = resp.usage.total_tokens if resp.usage else len("".join(batch)) // 2
        cost_tracker.record(EMBEDDING_MODEL, tokens, extra=f"embed x{len(batch)}")
        # 按 index 排序确保顺序（API 不保证返回顺序与输入一致）
        sorted_data = sorted(resp.data, key=lambda d: d.index)
        return [(d.embedding, None) for d in sorted_data]

    # ------------------------------------------------------------------
    # native 模式：text_type / instruct / dense&sparse
    # ------------------------------------------------------------------
    def _call_native(self, batch: list, text_type: str | None) -> list:
        """调用 DashScope 原生 embedding 端点。

        零新增依赖：openai SDK 本身依赖 httpx，这里直接复用 httpx。
        """
        import httpx

        parameters: dict = {"output_type": self.output_type}
        if text_type in ("query", "document"):
            parameters["text_type"] = text_type
        instruct = self.instruct_for(text_type)
        if instruct:
            parameters["instruct"] = instruct
        # 维度仅在偏离 API 默认值（1024）时下发，避免参数名/取值歧义
        if EMBEDDING_DIM and EMBEDDING_DIM != 1024:
            parameters["dimension"] = EMBEDDING_DIM

        payload = {
            "model": EMBEDDING_MODEL,
            "input": {"texts": batch},
            "parameters": parameters,
        }
        headers = {
            "Authorization": f"Bearer {EMBEDDING_API_KEY}",
            "Content-Type": "application/json",
        }

        try:
            response = httpx.post(
                EMBEDDING_NATIVE_URL,
                json=payload,
                headers=headers,
                timeout=NATIVE_TIMEOUT,
            )
        except httpx.HTTPError as exc:
            # 不静默降级：见模块 docstring 中「向量空间一致性」说明
            raise RuntimeError(f"DashScope 原生 embedding 请求失败: {exc}") from exc

        if response.status_code != 200:
            raise RuntimeError(
                f"DashScope 原生 embedding 返回 {response.status_code}: "
                f"{response.text[:300]}"
            )

        body = response.json()
        embeddings = (body.get("output") or {}).get("embeddings") or []
        if not embeddings:
            raise RuntimeError(f"DashScope 原生 embedding 返回空结果: {str(body)[:300]}")

        # 原生接口用 total_tokens 计费，与兼容接口口径一致
        tokens = (body.get("usage") or {}).get("total_tokens") or len("".join(batch)) // 2
        cost_tracker.record(
            EMBEDDING_MODEL, tokens,
            extra=f"embed_native x{len(batch)} {self.output_type}",
        )

        # 按 text_index 排序，保证与输入顺序一一对应
        ordered = sorted(embeddings, key=lambda e: e.get("text_index", 0))
        return [(item.get("embedding") or [], _normalize_sparse(item)) for item in ordered]

    # ------------------------------------------------------------------
    # 相似度
    # ------------------------------------------------------------------
    def similarity(self, vec1: list, vec2: list) -> float:
        """余弦相似度（DashScope 向量已归一化，等价于点积）。"""
        return float(np.dot(vec1, vec2))


def _normalize_sparse(item: dict) -> dict | None:
    """把 DashScope 的稀疏向量转成 {index: weight} 字典。

    API 返回形如 [{"index": 7149, "value": 0.829, "token": "风"}, ...]，
    token 字段仅用于调试，落库与计算只需要 index/value。
    返回 None 表示该条没有稀疏向量（未开启或模型不支持）。
    """
    raw = item.get("sparse_embedding")
    if not raw:
        return None
    sparse: dict = {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        idx = entry.get("index")
        val = entry.get("value")
        if idx is None or val is None:
            continue
        try:
            sparse[int(idx)] = float(val)
        except (TypeError, ValueError):
            continue
    return sparse or None


def sparse_dot(query_sparse: dict | None, doc_sparse: dict | None) -> float:
    """两个稀疏向量的内积（词法相关性得分）。

    稀疏向量绝大多数维度为 0，只在较短的一侧遍历，复杂度 O(min(|q|, |d|))。
    任一侧为空返回 0.0 —— 语义是「没有词法重叠」，而不是「出错」。
    """
    if not query_sparse or not doc_sparse:
        return 0.0
    if len(query_sparse) > len(doc_sparse):
        query_sparse, doc_sparse = doc_sparse, query_sparse
    return float(sum(w * doc_sparse[i] for i, w in query_sparse.items() if i in doc_sparse))
