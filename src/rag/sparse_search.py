"""
=============================================================================
sparse_search.py — 稀疏（词法）检索路：pg_trgm 的学习型替代
=============================================================================
【为什么需要这一路】
    项目现有的关键词检索跑在 PostgreSQL 的 pg_trgm 上（keyword_search.py）。
    trigram 把文本切成连续三字符片段做重合度匹配，这对英文是合理的近似，
    但对中文并不成立——中文没有空格分词，三字片段会大量命中无意义的字符组合。
    评测里已经量到了代价：叠加关键词路后 P@5 被中文 trigram 噪音拉低。

    DashScope 的 text-embedding-v4 / qwen3.7-text-embedding 可以在同一次调用里
    额外返回类 SPLADE 的学习型稀疏向量（output_type="dense&sparse"）。它做的是
    词项级的相关性建模而非字符级重合，正好补上 trigram 在中文上的短板，
    而且不需要引入 Elasticsearch，也不需要给 PG 装 zhparser / pg_jieba 分词扩展
    ——「一个 PostgreSQL 搞定关系数据 + 向量 + 词法检索」的选型逻辑得以保留。

【为什么是旁路表而不是给主表加列】
    knowledge_chunks 是论文评测基线（tag v1.0-thesis）所依赖的表，任何列变更
    都会牵动重新摄入与指标重跑。这里改为独立旁路表 knowledge_chunks_sparse：
      - 开关关闭时这张表根本不会被创建，默认部署零变化；
      - 开关打开时按需建表 + 增量填充，主表结构与既有向量一字未动；
      - 想回退只需把 SPARSE_RETRIEVAL 设回 off，必要时 DROP 这张表即可。
    与 semantic_cache.py 里 semantic_cache_index 的「可选索引表自动创建」同一思路。

【为什么在应用层算内积而不是用 pgvector sparsevec】
    pgvector 0.7+ 提供 sparsevec 类型，但声明列时必须写死向量维度，而稀疏维度
    等于模型词表大小、随模型版本变化；写错即整列不可用。本项目知识库规模是
    数百到数千 chunk（不是百万级），全量载入进程内做内积是微秒~毫秒级开销，
    用一点内存换来「不绑定词表维度、不做 schema 迁移」的确定性，是划算的。
    规模真的上去之后再迁 sparsevec + 稀疏倒排索引，接口不用变。

【降级语义】
    依赖不满足（未开 native 模式 / 未开 sparse 输出 / 表为空 / PG 不可用）时，
    available() 返回 False 或 search() 返回空列表，检索链路退回原有双路，
    绝不因为这一路的问题中断整个检索。

【被谁调用】KnowledgeSearch.search()（第三路，进同一个 RRF）、knowledge_ingestion
=============================================================================
"""

import json
import logging
import threading
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# 与其他检索路保持一致的候选数量：向量 20 + 关键词 20 + 稀疏 20 → RRF
PRE_RANK_TOP = 20


class SparseSearch:
    """稀疏词法检索路（可选，默认关闭）。

    【职责】
        1. 维护旁路表 knowledge_chunks_sparse（chunk_id → 稀疏向量 JSONB）
        2. 对查询做稀疏编码，与库内稀疏向量算内积，返回 Top-K 词法相关结果
        3. 输出结构与 vector_search / keyword_search 对齐，可直接进 rrf_fuse

    【开关】
        SPARSE_RETRIEVAL=on 且 EMBEDDING_API_MODE=native
        且 EMBEDDING_OUTPUT_TYPE 含 sparse，三者同时满足才真正生效。
    """

    TABLE = "knowledge_chunks_sparse"

    def __init__(self, pg=None, emb=None):
        # 延迟导入，避免在 PG / embedding 不可用的环境（如离线单测）里炸掉
        from src.storage.pg import PGClient
        from src.rag.embedding import EmbeddingService
        from src.config import SPARSE_RETRIEVAL

        self.pg = pg or PGClient()
        self.emb = emb or EmbeddingService()
        self.enabled = (SPARSE_RETRIEVAL or "off").lower() == "on"
        self._cache: Optional[Dict[str, dict]] = None
        self._cache_lock = threading.Lock()
        self._table_ready = False

    # ------------------------------------------------------------------
    # 可用性判定
    # ------------------------------------------------------------------
    def available(self) -> bool:
        """这一路是否真的可以参与检索。

        三个条件缺一不可；任一不满足都记一次日志并返回 False，
        让上层安静地退回双路检索，而不是抛异常。
        """
        if not self.enabled:
            return False
        if not self.emb.sparse_enabled:
            logger.warning(
                "SPARSE_RETRIEVAL=on 但 embedding 侧未就绪："
                "需要 EMBEDDING_API_MODE=native 且 EMBEDDING_OUTPUT_TYPE 含 sparse。"
                "本次按关闭处理，检索退回向量+关键词双路。"
            )
            return False
        return True

    # ------------------------------------------------------------------
    # 表管理
    # ------------------------------------------------------------------
    def ensure_table(self) -> None:
        """按需建表。幂等；失败只记日志，不影响主链路。"""
        if self._table_ready:
            return
        try:
            self.pg.execute(
                f"CREATE TABLE IF NOT EXISTS {self.TABLE} ("
                "chunk_id TEXT PRIMARY KEY, "
                "sparse JSONB NOT NULL, "
                "created_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )
            self._table_ready = True
        except Exception as exc:  # 建表失败 → 这一路整体不可用
            logger.warning(f"稀疏检索表创建失败，稀疏路禁用: {exc}")
            self.enabled = False

    def upsert(self, chunk_id: str, sparse: Optional[dict]) -> bool:
        """写入单个 chunk 的稀疏向量。sparse 为空时跳过（不写空行）。"""
        if not sparse:
            return False
        self.ensure_table()
        if not self._table_ready:
            return False
        try:
            self.pg.execute(
                f"INSERT INTO {self.TABLE} (chunk_id, sparse) "
                "VALUES (:cid, CAST(:payload AS JSONB)) "
                "ON CONFLICT (chunk_id) DO UPDATE SET "
                "sparse = EXCLUDED.sparse, created_at = now()",
                {"cid": chunk_id, "payload": json.dumps(sparse, ensure_ascii=False)},
            )
            self.invalidate()
            return True
        except Exception as exc:
            logger.warning(f"稀疏向量写入失败 chunk_id={chunk_id}: {exc}")
            return False

    def bulk_upsert(self, rows: List[tuple]) -> int:
        """批量写入 [(chunk_id, sparse), ...]，返回成功条数。"""
        return sum(1 for cid, sp in rows if self.upsert(cid, sp))

    def count(self) -> int:
        """旁路表里的稀疏向量条数（用于摄入报告与自检）。"""
        self.ensure_table()
        if not self._table_ready:
            return 0
        try:
            rows = self.pg.fetch_all(f"SELECT count(*) FROM {self.TABLE}")
            return int(rows[0][0]) if rows else 0
        except Exception:
            return 0

    def invalidate(self) -> None:
        """清空进程内缓存。知识库重新摄入后必须调用。"""
        with self._cache_lock:
            self._cache = None

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def _load_all(self) -> Dict[str, dict]:
        """把旁路表 JOIN 主表后全量载入进程内缓存。

        缓存结构：{chunk_id: {"sparse": {index: weight}, "title":..., "content":...,
                              "source_file":..., "chunk_index":...}}

        为什么连正文一起缓存、而不是检索命中后再回表：
          1. 省掉一次 IN/ANY 数组绑定查询——SQLAlchemy text() 传 list 给 ANY()
             的行为依赖 DBAPI 适配，项目里没有先例，不值得为它引入不确定性；
          2. 更快：命中后直接组装结果，没有额外往返；
          3. INNER JOIN 天然过滤掉「旁路表有、主表已删」的陈旧行。

        内存代价可控：知识库是数百~数千 chunk、单块约 800 字，全量正文在 MB 量级。
        表内容变更靠 invalidate() 刷新（upsert 会自动调用）。
        """
        with self._cache_lock:
            if self._cache is not None:
                return self._cache

        self.ensure_table()
        if not self._table_ready:
            return {}

        try:
            rows = self.pg.fetch_all(
                f"SELECT s.chunk_id, s.sparse, "
                "k.title, k.content, k.source_file, k.chunk_index "
                f"FROM {self.TABLE} s "
                "JOIN knowledge_chunks k ON k.chunk_id = s.chunk_id"
            )
        except Exception as exc:
            logger.warning(f"稀疏向量载入失败，本次稀疏路返回空: {exc}")
            return {}

        loaded: Dict[str, dict] = {}
        for row in rows:
            chunk_id, raw = row[0], row[1]
            if not chunk_id or not raw:
                continue
            # psycopg 对 JSONB 可能已反序列化为 dict，也可能是字符串
            sparse = raw if isinstance(raw, dict) else _parse_json(raw)
            if not sparse:
                continue
            loaded[str(chunk_id)] = {
                "sparse": _coerce_sparse(sparse),
                "title": row[2] or "",
                "content": row[3] or "",
                "source_file": row[4] or "",
                "chunk_index": row[5],
            }

        with self._cache_lock:
            self._cache = loaded
        return loaded

    def search(self, query: str, top_k: int = PRE_RANK_TOP) -> List[dict]:
        """查询 → 稀疏编码 → 内积排序 → 回表取内容 → 标准结果结构。

        返回结构与 vector_search / keyword_search 对齐，可直接进 rrf_fuse：
            chunk_id, title, content, source_file, chunk_index, score, source
        任何一步失败都返回 []，由上层的双路结果兜住。
        """
        if not self.available():
            return []

        try:
            _, query_sparse = self.emb.embed_with_sparse(query, text_type="query")
        except Exception as exc:
            logger.warning(f"查询稀疏编码失败，稀疏路跳过: {exc}")
            return []
        if not query_sparse:
            return []

        corpus = self._load_all()
        if not corpus:
            # 表是空的（还没跑过稀疏摄入）——这是配置问题不是故障，安静返回
            logger.info("稀疏检索表为空，跳过稀疏路（需先以 native+sparse 模式重新摄入知识库）")
            return []

        from src.rag.embedding import sparse_dot

        scored = [
            (cid, sparse_dot(query_sparse, record["sparse"]))
            for cid, record in corpus.items()
        ]
        # 内积为 0 表示没有词法重叠，直接丢弃，避免把全库塞进候选
        scored = [(cid, s) for cid, s in scored if s > 0.0]
        if not scored:
            return []
        scored.sort(key=lambda kv: kv[1], reverse=True)

        # 正文已随稀疏向量一起缓存，这里直接按得分降序组装成
        # 与 vector_search / keyword_search 完全一致的结果结构
        results: List[dict] = []
        for cid, score in scored[:top_k]:
            record = corpus.get(cid)
            if record is None:
                continue
            results.append({
                "chunk_id": cid,
                "title": record["title"],
                "content": record["content"],
                "source_file": record["source_file"],
                "chunk_index": record["chunk_index"],
                "score": round(score, 6),
                "source": "sparse",
            })
        return results


def _parse_json(raw) -> Optional[dict]:
    """JSONB 取回来是字符串时的兜底解析。"""
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except (TypeError, ValueError):
        return None


def _coerce_sparse(sparse: dict) -> dict:
    """把从 JSONB 读回的稀疏向量键归一化回 int。

    这一步不是可选的：JSON 对象的键只能是字符串，所以 {7149: 0.83} 写进 JSONB
    再读出来一定变成 {"7149": 0.83}。而查询侧的稀疏向量由 _normalize_sparse
    产出、键是 int。两边键类型不一致时 sparse_dot 找不到任何公共索引，
    内积恒为 0 —— 稀疏路会「不报错但永远返回空」，属于最难发现的静默失效。
    """
    coerced: dict = {}
    for key, value in sparse.items():
        try:
            coerced[int(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return coerced
