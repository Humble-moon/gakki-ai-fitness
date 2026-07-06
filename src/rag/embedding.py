"""
=============================================================================
embedding.py — 文本向量化基础服务
=============================================================================
【项目角色】
    这是整个 RAG 检索体系的最底层服务，为所有上层检索模块提供文本向量化能力。
    被 VectorSearch、KnowledgeSearch、SemanticCache 等模块调用。
    调用 sentence_transformers 库，使用配置中指定的 EMBEDDING_MODEL 模型。

【五层检索关系】
    本模块位于第 0 层（基础层），不直接参与检索，但被第 1~4 层所依赖。
    第 1 层 VectorSearch 使用 embed() 将查询转为向量做余弦相似度检索。
    第 2 层 KnowledgeSearch 使用 embed() 对知识库做向量检索。
    第 5 层 SemanticCache 使用 embed() 存储查询向量用于相似度匹配。

【关键设计决策】
    延迟加载模型（而非 __init__ 中立即加载）的原因：
    - Streamlit 使用 fork 方式启动多进程，如果在 __init__ 时加载 CUDA 模型，
      fork 后的子进程会继承已初始化的 CUDA 上下文导致崩溃。
    - 延迟到首次调用时才加载，确保模型在子进程中独立初始化。
    - torch.set_num_threads(1) 限制了 PyTorch 的并行线程数，避免与 Streamlit
      的线程模型冲突。
    - normalize_embeddings=True 将向量归一化到单位长度，这样相似度计算
      可以直接用点积代替余弦距离，大幅加快检索速度。
=============================================================================
"""

import numpy as np
from src.config import EMBEDDING_MODEL


class EmbeddingService:
    """文本向量化服务类。

    【职责】
        将文本（查询语句、知识片段等）转换为高维向量（embedding），
        并提供向量相似度计算能力。

    【使用场景】
        - VectorSearch.search() 调用 embed() 对用户查询做向量化
        - KnowledgeSearch.vector_search() 调用 embed() 对知识库查询做向量化
        - KnowledgeIngestion.ingest() 调用 embed() 对知识分块做向量化写入 PG
        - SemanticCache 调用 embed() 存储查询向量用于缓存匹配
    """

    def __init__(self):
        # 延迟加载：不在 __init__ 中加载模型，避免 Streamlit fork 子进程崩溃
        # 当 Streamlit 使用 fork 多进程时，子进程会继承父进程的 CUDA 上下文，
        # 如果父进程已初始化 CUDA，子进程的 CUDA 状态会损坏导致段错误。
        self.model = None  # 延迟加载，避免 Streamlit fork 崩溃

    def _ensure_model(self):
        """确保模型已加载（懒加载，首次调用时触发）。

        【为什么放在这里而不是 __init__】
            Streamlit 的 multiprocessing 使用 spawn 或 fork 方式。
            若在 __init__ 中加载模型，fork 后的子进程会拿到已初始化但不可用的 CUDA 上下文，
            导致 "CUDA error: initialization error" 崩溃。
            延迟到实际使用时加载，子进程才会独立初始化模型。
        """
        if self.model is None:
            import torch
            import src.config as cfg
            # 限制 PyTorch 线程数：避免与 Streamlit 线程池争抢 CPU 资源
            torch.set_num_threads(1)
            from sentence_transformers import SentenceTransformer
            # 使用配置文件中指定的模型（如 BGE-M3、text2vec 等）
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            # 运行时自动获取模型输出维度，写入全局配置
            # 避免换模型时手动同步 EMBEDDING_DIM 导致维度不匹配
            actual_dim = self.model.get_sentence_embedding_dimension()
            if cfg.EMBEDDING_DIM != actual_dim:
                cfg.EMBEDDING_DIM = actual_dim
                import logging
                logging.getLogger(__name__).info(
                    f"EMBEDDING_DIM auto-set to {actual_dim} (model: {EMBEDDING_MODEL})"
                )

    def embed(self, text: str) -> list:
        """将单条文本转换为向量。

        输入：
            text: str — 待向量化的文本（用户查询或知识片段）

        输出：
            list[float] — 归一化后的浮点数向量列表（维度由模型决定，通常 768 或 1024）

        核心逻辑：
            1. 确保模型已加载（懒加载）
            2. 调用 sentence_transformers 的 encode 方法
            3. normalize_embeddings=True 将向量归一化为单位向量
               （归一化后余弦相似度 = 点积，计算复杂度从 O(n) 降到 O(1)）
        """
        self._ensure_model()
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list) -> list:
        """批量文本向量化（用于摄入阶段批量处理知识片段）。

        输入：
            texts: list[str] — 待向量化的文本列表

        输出：
            list[list[float]] — 每条文本对应的向量列表

        说明：
            批量编码比逐条编码快 3~5 倍，因为 GPU 可以并行处理。
            摄入流程中一次性编码所有分块时使用此方法。
        """
        self._ensure_model()
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return vecs.tolist()

    def similarity(self, vec1: list, vec2: list) -> float:
        """计算两个向量的余弦相似度。

        输入：
            vec1: list[float] — 向量 A
            vec2: list[float] — 向量 B

        输出：
            float — 余弦相似度，范围 [-1, 1]（因向量已归一化，等价于点积）

        说明：
            因为 embed() 输出的向量已归一化为单位长度，
            余弦相似度 = (A·B)/(|A|*|B|) = A·B/1 = A·B，
            所以直接用 numpy.dot 即可，无需再做除法。
        """
        return float(np.dot(vec1, vec2))
