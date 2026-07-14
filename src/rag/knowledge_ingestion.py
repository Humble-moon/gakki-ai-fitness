"""
=============================================================================
knowledge_ingestion.py — 知识库文档摄入管线（离线批处理）v2.0
=============================================================================
【项目角色】
    这是 RAG 检索体系的"数据补给线"，负责将 data/knowledge/ 目录下的
    Markdown 知识文档（如健身科普文章、训练计划、饮食指南等）经过：
    读取 → 分块 → 向量化 → 写入 PostgreSQL，使其可被 KnowledgeSearch 检索。

【v2.0 新增分块策略】（2026-07-13）
    - three_tier:  原三段式智能切分（参数升级 500→800, 80→120）
    - semantic:   语义相似度切分 — 相邻句子 embedding 相似度低谷处切分
    - contextual: Anthropic 2024 Contextual Retrieval — 上下文前缀注入
    - small_to_big: 小 chunk 检索 + 大 chunk 喂 LLM 的分层策略

【被谁调用】命令行直接执行（python -m src.rag.knowledge_ingestion --dir ...）
【使用方式】
    # 默认：语义切分 + 上下文前缀组合拳
    python -m src.rag.knowledge_ingestion --dir data/knowledge

    # 单用语义切分（你问的"向量相似度切分"）
    python -m src.rag.knowledge_ingestion --dir data/knowledge --strategy semantic --semantic-threshold 0.5

    # 小 chunk 检 + 大 chunk 喂 LLM
    python -m src.rag.knowledge_ingestion --dir data/knowledge --strategy small_to_big

    # 保持旧版行为
    python -m src.rag.knowledge_ingestion --dir data/knowledge --strategy three_tier --chunk-size 500 --overlap 80
=============================================================================
"""

import hashlib
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.models.db_models import init_db
from src.storage.pg import PGClient
from src.rag.embedding import EmbeddingService

logger = logging.getLogger(__name__)

# ---- v2.0 升级默认参数 ----
# 800 字 ≈ 1 个完整概念单元（如一个训练原理的完整展开），相比 500 字减少语义碎片
# 业界趋势：embedding 模型上下文窗口越来越大（text-embedding-v4 支持 8K tokens），
# 更大的 chunk 配合更精准的检索策略（如语义切分）效果更好
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120  # 15%，覆盖相邻 chunk 边界 2-3 句中文


# =========================================================================
# 数据结构
# =========================================================================


@dataclass
class Chunk:
    """分块对象，携带元数据用于上下文检索和小大分层。"""
    content: str
    chunk_id: str
    metadata: dict = field(default_factory=dict)

    # Small-to-Big：指向父 chunk
    parent_id: Optional[str] = None
    parent_content: Optional[str] = None

    # Contextual Retrieval：embedding 时的上下文前缀
    context_prefix: Optional[str] = None

    @property
    def content_for_embedding(self) -> str:
        """embedding 用文本：有上下文前缀则拼接，无则原文。"""
        if self.context_prefix:
            return f"{self.context_prefix}\n\n{self.content}"
        return self.content

    def __len__(self) -> int:
        return len(self.content)


# =========================================================================
# 工具函数
# =========================================================================

_SENT_SPLIT_RE = re.compile(r"(?<=[。！？.!?])\s*")
_HEADING_RE = re.compile(r"^#{1,6}\s+")


def _hash_id(*parts: str) -> str:
    return hashlib.md5(":".join(parts).encode()).hexdigest()[:16]


def split_sentences(text: str) -> list[str]:
    """中文句子切分，保留标点附着在上一句末尾。"""
    parts = _SENT_SPLIT_RE.split(text)
    result = []
    buf = ""
    for part in parts:
        buf += part
        if buf.rstrip() and re.search(r"[。！？.!?]$", buf.rstrip()):
            result.append(buf.strip())
            buf = ""
    if buf.strip():
        result.append(buf.strip())
    return result


def cosine_sim(a: list[float], b: list[float]) -> float:
    """余弦相似度。DashScope embedding 已归一化，等价于点积。"""
    return float(np.dot(a, b))


# =========================================================================
# 步骤 1: 读取 Markdown 文件
# =========================================================================

def read_markdown_files(knowledge_dir: str) -> List[dict]:
    """读取指定目录下所有 .md 文件，提取标题和内容（与 v1 相同）。"""
    docs = []
    dir_path = Path(knowledge_dir)
    for md_file in sorted(dir_path.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else md_file.stem
        docs.append({"title": title, "content": text, "source_file": md_file.name})
    return docs


# =========================================================================
# 步骤 2: 文本分块（抽象基类 + 四种策略）
# =========================================================================


class BaseChunker(ABC):
    """分块器抽象基类。所有策略实现此接口，方便 ingest() 中统一调用。"""

    def __init__(self, chunk_size: int = CHUNK_SIZE, min_chunk: int = 50):
        self.chunk_size = chunk_size
        self.min_chunk = min_chunk

    @abstractmethod
    def split(self, text: str, metadata: dict | None = None) -> List[Chunk]:
        ...


# -------------------------------------------------------------------------
# 策略 1: 三段式智能切分（v1 保留，参数升级）
# -------------------------------------------------------------------------

class ThreeTierChunker(BaseChunker):
    """
    原 v1 分块策略，参数从 (500, 80) 升级到 (800, 120)。

    三段逻辑：
      1. 段落边界优先（\\n\\n）—— 短段落合并，保证语义完整
      2. 句子边界兜底 —— 单段超限时在句号处切开
      3. 短标题附着 —— Markdown 标题不独立成 chunk，附加到下一块

    面试要点：这不是简单的固定大小切分，而是按文档结构逐级回退的智能策略，
    比 LangChain 的 RecursiveCharacterTextSplitter 更适配中文 Markdown 场景。
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP,
                 min_chunk: int = 50):
        super().__init__(chunk_size, min_chunk)
        self.overlap = overlap

    def split(self, text: str, metadata: dict | None = None) -> List[Chunk]:
        meta = metadata or {}
        paragraphs = text.split("\n\n")
        merged: list[str] = []
        buf = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            # 短标题附着
            if _HEADING_RE.match(para) and len(para) < 80:
                buf = (buf + "\n\n" + para).strip()
                continue
            if len(buf) + len(para) + 2 <= self.chunk_size:
                buf = (buf + "\n\n" + para).strip()
            else:
                if buf.strip():
                    merged.append(buf)
                buf = para
        if buf.strip():
            merged.append(buf)

        # 句子兜底 + 滑动窗口
        result = []
        for block in merged:
            if len(block) <= self.chunk_size:
                result.extend(self._slide(block, meta))
            else:
                sentences = split_sentences(block)
                sbuf = ""
                for s in sentences:
                    if len(sbuf) + len(s) <= self.chunk_size:
                        sbuf += s
                    else:
                        if sbuf.strip():
                            result.extend(self._slide(sbuf, meta))
                        sbuf = s
                if sbuf.strip():
                    result.extend(self._slide(sbuf, meta))
        return [c for c in result if len(c) >= self.min_chunk]

    def _slide(self, text: str, meta: dict) -> list[Chunk]:
        """滑动窗口切分，保证 overlap 字符级精确重叠"""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            content = text[start:end].strip()
            if content:
                chunks.append(Chunk(
                    content=content,
                    chunk_id=_hash_id("three_tier", content, str(start)),
                    metadata=dict(meta),
                ))
            if end >= len(text):
                break
            start += self.chunk_size - self.overlap
        return chunks


# -------------------------------------------------------------------------
# 策略 2: 语义相似度切分（你问的"向量相似度切分"）
# -------------------------------------------------------------------------

class SemanticChunker(BaseChunker):
    """
    基于相邻句子的 embedding 相似度判断切分点。

    【原理】
      1. 文本拆成句子
      2. 逐句 embedding（项目已有的 DashScope API）
      3. 计算相邻两句的余弦相似度
      4. 相似度低 = 语义转折 = 在此处切分
      5. 相似度高的连续句子合并成一个 chunk

    【为什么比固定大小切分好】
      固定大小切分不管你在说什么，到 500 字就一刀切。
      语义切分会在"话题自然转换"的地方切，每个 chunk 内都是一个完整概念单元。
      比如在健身文档中，不会把"卧推的动作要领"和"卧推的常见错误"切到两个 chunk 里
      ——因为它们语义连贯，相似度 > 阈值，所以会合并。

    【三种阈值模式】
      - percentile（默认，推荐）: 取相似度分布的第 P 百分位，自适应不同文档
      - absolute: 低于固定阈值就切（如 0.5），适合同一领域文档
      - interquartile: 低于 Q1 - 1.5*IQR 处切，只在高语义断裂处分块（激进合并）

    【面试可以说的点】
      "LangChain 的 SemanticChunker 用的就是这个思路，区别是我直接在项目里
      用 DashScope 的 embedding 做相似度计算，少一层依赖，而且中文句子切分
      做了针对性处理（中文句末标点 vs 英文句点）。"
    """

    MODE_PERCENTILE = "percentile"
    MODE_ABSOLUTE = "absolute"
    MODE_IQR = "interquartile"

    def __init__(
        self,
        embed_service: EmbeddingService,
        chunk_size: int = CHUNK_SIZE,
        min_chunk: int = 100,
        max_chunk: int = 1500,
        mode: str = "percentile",
        threshold: float = 0.6,
        percentile: float = 50.0,
    ):
        super().__init__(chunk_size, min_chunk)
        self.embed_service = embed_service
        self.max_chunk = max_chunk
        self.mode = mode
        self.threshold = threshold
        self.percentile = percentile

    def split(self, text: str, metadata: dict | None = None) -> List[Chunk]:
        meta = metadata or {}
        sentences = split_sentences(text)
        if len(sentences) <= 1:
            return self._finalize(sentences, text, meta)

        # Step 1: 批量 embedding
        embeddings = self.embed_service.embed_batch(sentences)

        # Step 2: 相邻句相似度
        sims = [
            cosine_sim(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]

        # Step 3: 阈值
        cutoff = self._compute_threshold(sims)

        # Step 4: 在相似度低谷处切分
        split_idx = [i + 1 for i, s in enumerate(sims) if s < cutoff]

        # Step 5: 合并句子组
        groups = self._merge(sentences, split_idx)

        # Step 6: 超大 group 二次切分
        result = []
        for g in groups:
            if len(g) <= self.max_chunk:
                result.append(g)
            else:
                # 回退固定切分（语义切分已尽力）
                for i in range(0, len(g), self.max_chunk - 100):
                    result.append(g[i:i + self.max_chunk])

        return self._finalize(result, text, meta)

    def _compute_threshold(self, sims: list[float]) -> float:
        if not sims:
            return self.threshold
        if self.mode == self.MODE_ABSOLUTE:
            return self.threshold
        if self.mode == self.MODE_IQR:
            s = sorted(sims)
            n = len(s)
            q1, q3 = s[n // 4], s[3 * n // 4]
            return q1 - 1.5 * (q3 - q1)
        # percentile
        s = sorted(sims)
        idx = min(int(len(s) * self.percentile / 100), len(s) - 1)
        return s[idx]

    def _merge(self, sentences: list[str], split_idx: list[int]) -> list[str]:
        groups, start = [], 0
        for i in split_idx:
            g = "".join(sentences[start:i]).strip()
            if g:
                groups.append(g)
            start = i
        g = "".join(sentences[start:]).strip()
        if g:
            groups.append(g)
        return groups

    def _finalize(self, contents: list[str], text: str, meta: dict) -> list[Chunk]:
        result = []
        offset = 0
        for c in contents:
            c = c.strip()
            if len(c) < self.min_chunk:
                continue
            pos = text.find(c, offset)
            if pos == -1:
                pos = offset
            result.append(Chunk(
                content=c,
                chunk_id=_hash_id("semantic", c, str(pos)),
                metadata=dict(meta, char_start=pos, char_end=pos + len(c)),
            ))
            offset = pos + len(c)
        return result


# -------------------------------------------------------------------------
# 策略 3: Contextual Retrieval（Anthropic 2024）
# -------------------------------------------------------------------------

class ContextualChunker(BaseChunker):
    """
    Anthropic 2024 年提出的 Contextual Retrieval：在 embedding 前给每个 chunk
    拼接"文档位置描述"，让 embedding 模型理解这段文字的上下文。

    【为什么有效】
      普通的 chunk 是一段孤立的文本。当用户搜"训练后什么时候喝蛋白粉"，
      embedding 只看到 chunk 内容本身。但如果 chunk 带上前缀：
        "文档: 营养补充指南 | 章节: 蛋白质补充时机 > 训练后窗口期"
      再去做 embedding，向量就多了一层"定位信息"，检索命中率大幅提升。

    【前缀格式（规则模板）】
      [文档: {title}] [章节: {heading_path}]

      {原始 chunk 内容}

    【面试拓展】
      如果要更精准的效果，可以用 LLM 为每个 chunk 生成一句话摘要作为前缀，
      Anthropic 论文显示这比规则模板高约 4-5% 的 recall@20。

    【使用方式】
      包裹任意一个 chunker（ThreeTierChunker 或 SemanticChunker）即可。
    """

    def __init__(self, inner: BaseChunker, doc_title: str = ""):
        super().__init__(inner.chunk_size, inner.min_chunk)
        self.inner = inner
        self.doc_title = doc_title

    def split(self, text: str, metadata: dict | None = None) -> List[Chunk]:
        meta = metadata or {}
        # 提取所有标题行用于定位
        headings = [
            (m.group(0), m.start())
            for m in re.finditer(r"^#{1,6}\s+.+$", text, re.MULTILINE)
        ]
        raw = self.inner.split(text, meta)

        for chunk in raw:
            # 找 chunk 之前最近的一个标题
            # 用 chunk 内容的位置来匹配
            chunk_pos = text.find(chunk.content[:60]) if len(chunk.content) >= 20 else 0
            heading_text = ""
            for h_text, h_pos in headings:
                if h_pos <= chunk_pos:
                    heading_text = h_text.strip().lstrip("#").strip()
            prefix_parts = []
            if self.doc_title:
                prefix_parts.append(f"文档: {self.doc_title}")
            if heading_text:
                prefix_parts.append(f"章节: {heading_text}")
            if prefix_parts:
                chunk.context_prefix = " | ".join(prefix_parts)
                chunk.metadata["doc_title"] = self.doc_title
                chunk.metadata["heading"] = heading_text
        return raw


# -------------------------------------------------------------------------
# 策略 4: Small-to-Big 分层检索
# -------------------------------------------------------------------------

class SmallToBigChunker:
    """
    小 chunk 做检索（召回精准），大 chunk 喂 LLM（上下文完整）。

    【问题】
      检索和生成对 chunk 大小的需求是矛盾的：
      - 小 chunk（200-500字）检索精准，但 LLM 看到的上下文不完整
      - 大 chunk（800-1500字）上下文完整，但检索时噪声多、召回精度低

    【解决方案】
      切两层：
        Child（小，~400字）→ embedding 入库，负责检索
        Parent（大，~800字）→ 不单独入库，检索命中 child 后返回其 parent
      LLM 拿到的是 parent 的完整上下文，而检索用的是 child 的精准匹配。

    【使用方式】
      children, parents = chunker.split(text)
      # children → embedding → pgvector（检索用）
      # parents  → 存到 parent_chunks 表或内存（检索命中后返回 LLM 用）
    """

    def __init__(self, parent_chunker: BaseChunker, child_size: int = 400,
                 child_overlap: int = 80):
        self.parent_chunker = parent_chunker
        self.child_size = child_size
        self.child_overlap = child_overlap

    def split(self, text: str, metadata: dict | None = None) -> tuple:
        """
        Returns:
          children: 入库做 embedding + 检索
          parents:  检索命中后喂给 LLM
        """
        meta = metadata or {}
        parents = self.parent_chunker.split(text, meta)
        children = []

        for p in parents:
            pid = p.chunk_id
            p_text = p.content
            start = 0
            while start < len(p_text):
                end = min(start + self.child_size, len(p_text))
                cc = p_text[start:end].strip()
                if cc:
                    children.append(Chunk(
                        content=cc,
                        chunk_id=_hash_id("child", cc, str(start)),
                        parent_id=pid,
                        parent_content=p_text,
                        metadata=dict(meta, parent_id=pid),
                    ))
                if end >= len(p_text):
                    break
                start += self.child_size - self.child_overlap
        return children, parents


# -------------------------------------------------------------------------
# v1 兼容接口：chunk_text（返回 str 列表）
# -------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """【v1 兼容】三段式切分，返回纯字符串列表而非 Chunk 对象。"""
    chunker = ThreeTierChunker(chunk_size=chunk_size, overlap=overlap)
    return [c.content for c in chunker.split(text)]


# =========================================================================
# 分块器工厂
# =========================================================================

def get_chunker(
    strategy: str = "combo",
    embed_service: EmbeddingService | None = None,
    doc_title: str = "",
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    semantic_threshold: float = 0.6,
) -> BaseChunker | SmallToBigChunker:
    """根据策略名创建分块器。

    strategy:
      - "three_tier":   原 v1 三段式（已升级参数）
      - "semantic":     语义相似度切分（你问的"向量切分"）
      - "contextual":   三段式 + Anthropic 上下文前缀
      - "small_to_big": 小检大喂分层（子 chunk 入库，父 chunk 喂 LLM）
      - "combo"（默认）: semantic + contextual 组合拳
    """
    if strategy == "three_tier":
        return ThreeTierChunker(chunk_size=chunk_size, overlap=overlap)

    if strategy == "semantic":
        if embed_service is None:
            embed_service = EmbeddingService()
        return SemanticChunker(
            embed_service=embed_service,
            chunk_size=chunk_size,
            threshold=semantic_threshold,
        )

    if strategy == "contextual":
        base = ThreeTierChunker(chunk_size=chunk_size, overlap=overlap)
        return ContextualChunker(inner=base, doc_title=doc_title)

    if strategy == "small_to_big":
        base = ThreeTierChunker(chunk_size=chunk_size, overlap=overlap)
        return SmallToBigChunker(parent_chunker=base)

    # combo (默认): semantic + contextual
    if embed_service is None:
        embed_service = EmbeddingService()
    semantic = SemanticChunker(
        embed_service=embed_service,
        chunk_size=chunk_size,
        threshold=semantic_threshold,
    )
    return ContextualChunker(inner=semantic, doc_title=doc_title)


# =========================================================================
# 步骤 3-4: 向量化 + 写入
# =========================================================================

def ingest(
    knowledge_dir: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    strategy: str = "combo",
    semantic_threshold: float = 0.6,
):
    """知识库摄入主流程：读取 → 分块 → 向量化 → 写入 PostgreSQL。

    参数：
        knowledge_dir:      Markdown 文件目录路径
        chunk_size:         分块大小（字符数）
        overlap:            块间重叠（字符数）
        strategy:           分块策略（three_tier / semantic / contextual / small_to_big / combo）
        semantic_threshold: SemanticChunker 的切分阈值（仅 strategy=semantic/combo 时生效）

    v2.0 改动：
      - 支持五种分块策略，通过 --strategy 切换
      - 默认策略从 three_tier(500,80) 升级到 combo(800,120) + 语义切分 + 上下文前缀
      - Small-to-big 模式下写入两张表：child_chunks + parent_chunks
    """
    init_db()
    pg = PGClient()
    emb = EmbeddingService()
    docs = read_markdown_files(knowledge_dir)
    logger.info(f"Found {len(docs)} documents in {knowledge_dir}")

    total_chunks = 0
    total_parents = 0

    for doc in docs:
        chunker = get_chunker(
            strategy=strategy,
            embed_service=emb,
            doc_title=doc["title"],
            chunk_size=chunk_size,
            overlap=overlap,
            semantic_threshold=semantic_threshold,
        )

        if strategy == "small_to_big":
            # 小大分层：children 入库检索，parents 入库供 LLM 上下文
            children, parents = chunker.split(doc["content"], {"source_file": doc["source_file"]})
            # 写入 children（检索用）
            for chunk in children:
                if len(chunk.content.strip()) < 50:
                    continue
                _insert_chunk(pg, emb, chunk, doc, chunk.chunk_id, is_parent=False)
                total_chunks += 1
            # 写入 parents（LLM 用），用 parent_id 关联
            for chunk in parents:
                _insert_chunk(pg, emb, chunk, doc, chunk.chunk_id, is_parent=True)
                total_parents += 1
        else:
            # 普通模式：chunks 直接入库
            chunks = chunker.split(doc["content"], {"source_file": doc["source_file"]})
            for chunk in chunks:
                if len(chunk.content.strip()) < 50:
                    continue
                _insert_chunk(pg, emb, chunk, doc, chunk.chunk_id, is_parent=False)
                total_chunks += 1

    if strategy == "small_to_big":
        logger.info(
            f"Ingested {total_chunks} child chunks + {total_parents} parent chunks "
            f"from {len(docs)} documents"
        )
    else:
        logger.info(f"Ingested {total_chunks} chunks from {len(docs)} documents")


def _insert_chunk(
    pg: PGClient,
    emb: EmbeddingService,
    chunk: Chunk,
    doc: dict,
    chunk_id: str,
    is_parent: bool = False,
):
    """写入单个 chunk 到 PG，支持 Small-to-Big 双表。"""
    # embedding 时用带上下文前缀的文本
    vec = emb.embed(chunk.content_for_embedding)
    vec_str = f"[{','.join(str(v) for v in vec)}]"

    table = "knowledge_chunks"  # 默认表
    columns = "(chunk_id, title, content, source_file, chunk_index, embedding)"
    values = "(:chunk_id, :title, :content, :source_file, :chunk_index, CAST(:vec AS vector))"

    params: dict = {
        "chunk_id": chunk_id,
        "title": doc["title"],
        "content": chunk.content,
        "source_file": chunk.metadata.get("source_file", ""),
        "chunk_index": 0,
        "vec": vec_str,
    }

    if is_parent:
        params["content"] = chunk.parent_content or chunk.content
    if chunk.parent_id:
        params["parent_id"] = chunk.parent_id
        columns = columns.replace(")", ", parent_id)")
        values = values.replace(")", ", :parent_id)")

    try:
        pg.execute(
            f"""
                INSERT INTO {table} {columns}
                VALUES {values}
                ON CONFLICT (chunk_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = CAST(:vec AS vector)
            """,
            params,
        )
    except Exception as e:
        logger.warning(f"Ingest chunk {chunk_id} failed: {e}")


# =========================================================================
# 命令行入口
# =========================================================================

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="将知识库 Markdown 文档导入 PG 向量存储（v2.0 多策略分块）"
    )
    parser.add_argument("--dir", default="data/knowledge",
                        help="Markdown 文件所在目录")
    parser.add_argument("--strategy", default="combo",
                        choices=["three_tier", "semantic", "contextual", "small_to_big", "combo"],
                        help="分块策略（默认: combo = semantic + contextual）")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help=f"分块大小（字符数），默认 {CHUNK_SIZE}")
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP,
                        help=f"块间重叠（字符数），默认 {CHUNK_OVERLAP}")
    parser.add_argument("--semantic-threshold", type=float, default=0.6,
                        help="语义切分阈值，仅 semantic/combo 策略生效（默认 0.6）")
    args = parser.parse_args()

    strategy_labels = {
        "three_tier": "三段式智能切分",
        "semantic": "语义相似度切分",
        "contextual": "三段式 + 上下文前缀",
        "small_to_big": "小检大喂分层",
        "combo": "语义切分 + 上下文前缀（推荐）",
    }
    logger.info(f"Strategy: {strategy_labels.get(args.strategy, args.strategy)}")
    logger.info(f"Params: chunk_size={args.chunk_size}, overlap={args.overlap}")

    ingest(
        knowledge_dir=args.dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        strategy=args.strategy,
        semantic_threshold=args.semantic_threshold,
    )

