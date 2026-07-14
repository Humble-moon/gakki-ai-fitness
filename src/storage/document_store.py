"""
文档存储层 — 管理 user_documents + document_chunks 的 CRUD

独立于 knowledge_chunks，确保用户上传文档和公共知识库检索隔离。
"""

import logging
from datetime import datetime
from src.storage.pg import PGClient
from src.rag.embedding import EmbeddingService
from src.rag.knowledge_ingestion import chunk_text

logger = logging.getLogger(__name__)

# 单文件大小上限
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB

# 单个 session 最多保留文档数（自动清理最旧的）
MAX_DOCS_PER_SESSION = 5


class DocumentStore:
    """用户文档的存储与检索。

    职责：
    - save(): 保存文档全文 + 切块 + embedding 写入
    - search(): 向量检索 session 下的文档块
    - cleanup_session(): 清理 session 的所有文档数据
    """

    def __init__(self):
        self.db = PGClient()
        self.embedder = EmbeddingService()

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def save(self, session_id: str, filename: str, file_type: str,
             file_size: int, full_text: str, page_count: int,
             title: str, has_text: bool, parse_error: str) -> int:
        """保存文档并切块入库。返回 document_id。"""
        # 1. 清理旧文档（超过上限则删最旧的）
        self._enforce_limit(session_id)

        # 2. 写入 user_documents
        doc_id = self.db.fetch_one(
            """INSERT INTO user_documents
               (session_id, filename, file_type, file_size, raw_content,
                page_count, title, has_text, parse_error, created_at)
               VALUES (:sid, :fn, :ft, :fs, :rc, :pc, :ti, :ht, :pe, :ts)
               RETURNING id""",
            {"sid": session_id, "fn": filename, "ft": file_type, "fs": file_size,
             "rc": full_text, "pc": page_count, "ti": title, "ht": 1 if has_text else 0,
             "pe": parse_error or "", "ts": datetime.utcnow()},
        ).id

        # 3. 切块
        if has_text and full_text.strip():
            chunks = chunk_text(full_text)
            logger.info(f"Document {doc_id}: {len(chunks)} chunks from {len(full_text)} chars")

            # 4. embedding + 写入 document_chunks
            for i, chunk_text_content in enumerate(chunks):
                try:
                    vec = self.embedder.embed(chunk_text_content)
                except Exception as e:
                    logger.warning(f"Embedding failed for chunk {i} of doc {doc_id}: {e}")
                    continue

                self.db.execute(
                    """INSERT INTO document_chunks
                       (document_id, session_id, chunk_index, content,
                        chunk_type, title_path, page_number, embedding, created_at)
                       VALUES (:did, :sid, :ci, :ct, :ty, :tp, :pn, :emb, :ts)""",
                    {"did": doc_id, "sid": session_id, "ci": i,
                     "ct": self._inject_context(chunk_text_content, title, i + 1, len(chunks)),
                     "ty": "text", "tp": title, "pn": 1, "ts": datetime.utcnow(),
                     "emb": vec},
                )

        return doc_id

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def search(self, query: str, session_id: str, top_k: int = 10) -> list[dict]:
        """向量检索 session 下的文档块。"""
        try:
            vec = self.embedder.embed(query)
        except Exception:
            return []

        rows = self.db.fetch_all(
            """SELECT id, document_id, content, chunk_type, title_path, page_number,
                      embedding <=> (:vec)::vector AS distance
               FROM document_chunks
               WHERE session_id = :sid
               ORDER BY embedding <=> (:vec)::vector
               LIMIT :k""",
            {"vec": str(vec), "sid": session_id, "k": top_k},
        )

        return [
            {"id": r.id, "document_id": r.document_id,
             "content": r.content, "chunk_type": r.chunk_type,
             "title_path": r.title_path, "page_number": r.page_number,
             "score": round(1.0 - min(float(r.distance), 1.0), 4)}
            for r in rows
        ]

    def get_documents_for_session(self, session_id: str) -> list[dict]:
        """获取 session 下已上传的文档列表（元数据，不含全文）。"""
        rows = self.db.fetch_all(
            """SELECT id, filename, file_type, file_size, page_count,
                      title, has_text, parse_error, created_at
               FROM user_documents
               WHERE session_id = :sid
               ORDER BY created_at DESC""",
            {"sid": session_id},
        )
        return [
            {"id": r.id, "filename": r.filename, "file_type": r.file_type,
             "file_size": r.file_size, "page_count": r.page_count,
             "title": r.title, "has_text": bool(r.has_text),
             "parse_error": r.parse_error,
             "created_at": r.created_at.isoformat() if r.created_at else ""}
            for r in rows
        ]

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------

    def cleanup_session(self, session_id: str):
        """删除 session 的所有文档和切块数据。"""
        self.db.execute(
            "DELETE FROM document_chunks WHERE session_id = :sid",
            {"sid": session_id},
        )
        self.db.execute(
            "DELETE FROM user_documents WHERE session_id = :sid",
            {"sid": session_id},
        )

    def _enforce_limit(self, session_id: str):
        """保持每个 session 最多 MAX_DOCS_PER_SESSION 个文档。"""
        count_row = self.db.fetch_one(
            "SELECT COUNT(*) as cnt FROM user_documents WHERE session_id = :sid",
            {"sid": session_id},
        )
        if count_row and count_row.cnt >= MAX_DOCS_PER_SESSION:
            # 删最旧的
            old = self.db.fetch_all(
                """SELECT id FROM user_documents
                   WHERE session_id = :sid ORDER BY created_at ASC
                   LIMIT :n""",
                {"sid": session_id, "n": count_row.cnt - MAX_DOCS_PER_SESSION + 1},
            )
            for row in old:
                self.db.execute(
                    "DELETE FROM document_chunks WHERE document_id = :did",
                    {"did": row.id},
                )
                self.db.execute(
                    "DELETE FROM user_documents WHERE id = :did",
                    {"did": row.id},
                )

    @staticmethod
    def _inject_context(chunk_text_content: str, title: str,
                        chunk_num: int, total: int) -> str:
        """切块文本注入文档标题上下文，提升向量检索匹配质量。"""
        return f"[文档: {title}]\n{chunk_text_content}"
