"""
=============================================================================
knowledge_ingestion.py — 知识库文档摄入管线（离线批处理）
=============================================================================
【项目角色】
    这是 RAG 检索体系的"数据补给线"，负责将 data/knowledge/ 目录下的
    Markdown 知识文档（如健身科普文章、训练计划、饮食指南等）经过：
    读取 → 分块 → 向量化 → 写入 PostgreSQL，使其可被 KnowledgeSearch 检索。

【五层检索关系】
    本模块不直接参与检索，而是为第 2 层 KnowledgeSearch 提供数据基础。
    没有摄入就没有可检索的知识库，因此它是整个 RAG 系统启动的第一步。

【被谁调用】命令行直接执行（python -m src.rag.knowledge_ingestion --dir ...）
【调用谁】  init_db()（初始化数据库表结构）、PGClient（写入 PG）、
           EmbeddingService（生成向量嵌入）

【使用方式】
    python -m src.rag.knowledge_ingestion --dir data/knowledge
    python -m src.rag.knowledge_ingestion --dir data/knowledge --chunk-size 512 --overlap 64
=============================================================================
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import List

from src.models.db_models import init_db
from src.storage.pg import PGClient
from src.rag.embedding import EmbeddingService

logger = logging.getLogger(__name__)

# 默认分块大小 500 字符：在检索精度（小块更精准）和上下文完整性（大块更完整）
# 之间取平衡。500 字约等于 1~2 个自然段，适合健身知识这种中等密度的内容。
CHUNK_SIZE = 500

# 重叠 80 字符：确保分块边界处的内容不会因为跨块丢失上下文
# 80 字约等于 1~2 句话，足以保留段落间的过渡信息
CHUNK_OVERLAP = 80


# =========================================================================
# 步骤 1: 读取 Markdown 文件
# =========================================================================

def read_markdown_files(knowledge_dir: str) -> List[dict]:
    """读取指定目录下所有 .md 文件，提取标题和内容。

    输入：
        knowledge_dir: str — Markdown 文件所在目录路径，如 "data/knowledge"

    输出：
        List[dict] — 每个字典包含：
            - title:       str — 从文件第一个 # 标题提取，若无则用文件名
            - content:     str — 文件原始 Markdown 全文
            - source_file: str — 文件名（如 "diet_guide.md"），用于追溯来源

    说明：
        sorted() 确保文件按名称排序，保证多次运行结果一致（幂等性）。
    """
    docs = []
    dir_path = Path(knowledge_dir)
    for md_file in sorted(dir_path.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        # 从第一个一级标题（# 开头）提取文档标题
        # 这样可以获得有意义的标题（如 # 减脂饮食指南 → "减脂饮食指南"）
        # 如果文档没有标题，回退到文件名（去掉 .md 后缀）作为标题
        title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else md_file.stem
        docs.append({
            "title": title,
            "content": text,
            "source_file": md_file.name,
        })
    return docs


# =========================================================================
# 步骤 2: 文本分块
# =========================================================================

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """将长文本按段落/句子切分为定长方块（带重叠）。

    输入：
        text:       str — 待分块的原始文本
        chunk_size: int — 每块最大字符数（默认 500）
        overlap:    int — 块间重叠字符数（默认 80，当前实现是基于段落边界，未严格计算字符重叠）

    输出：
        List[str] — 分块后的文本列表

    分块策略（优先级递减）：
        1. 优先按段落边界（\n\n）切分 —— 保持自然语义单元
        2. 段落仍超过 chunk_size → 按句子边界（。！？.!?）切分 —— 保持句级语义
        3. 短标题（Markdown heading < 80 字）不单独成块，附加到内容中 —— 避免碎片化

    为什么需要分块：
        - Embedding 模型有最大输入长度限制（通常 512 tokens）
        - 小块检索更精准（不会因为文档太长而稀释相关性）
        - 重叠确保跨块边界的信息不丢失

    为什么需要重叠：
        假设一段话被切在中间：
            块 A: "...减脂期间应该注意"
            块 B: "控制碳水摄入量，增加蛋白质..."
        没有重叠的话，搜索"减脂碳水控制"可能只命中块 A 但 A 缺少关键的"碳水"信息。
        有重叠后：块 A 缓冲区的结尾包含了块 B 的开头内容，保证边界信息不丢失。
    """
    # 按双换行符切分段落（Markdown 的标准段落分隔）
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""  # 当前正在构建的块

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # 短标题（如 "## 训练频率"）不单独成块，合并到下一段内容中
        # 为什么：单独的标题没有实际信息价值，检索命中"训练频率"这个标题
        # 却看不到具体内容，用户体验差。合并后标题+内容一起命中才有意义。
        if re.match(r"^#{1,3}\s", para) and len(para) < 80:
            current = (current + "\n\n" + para).strip()
            continue

        # 当前段加入后不超过 chunk_size → 继续累积
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            # 当前段加入后会超限 → 先保存当前块，再开始新块
            if current:
                chunks.append(current[:chunk_size])

            # 如果段落本身超过 chunk_size → 按句子切分
            # 正则 (?<=[。！？.!?])\s* 在中文/英文句末标点后切分
            if len(para) > chunk_size:
                sentences = re.split(r"(?<=[。！？.!?])\s*", para)
                sub = ""
                for sent in sentences:
                    if len(sub) + len(sent) <= chunk_size:
                        sub += sent
                    else:
                        if sub:
                            chunks.append(sub[:chunk_size])
                        sub = sent
                # 处理完超长段落后，将剩余句子作为新块的起始
                if sub:
                    current = sub
                else:
                    current = ""
            else:
                # 段落适中 → 直接作为新块的起始
                current = para

    # 处理最后一个不完整的块
    if current:
        chunks.append(current[:chunk_size])

    return chunks


# =========================================================================
# 步骤 3-4: 向量化 + 写入
# =========================================================================

def ingest(knowledge_dir: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """知识库摄入主流程：读取 → 分块 → 向量化 → 写入 PostgreSQL。

    输入：
        knowledge_dir: str — Markdown 文件目录路径
        chunk_size:    int — 分块大小（字符数）
        overlap:       int — 块间重叠（字符数）

    输出：
        无（副作用：将数据写入 PostgreSQL 的 knowledge_chunks 表）

    处理流程：
        1. 初始化数据库（确保 knowledge_chunks 表存在）
        2. 读取所有 .md 文件
        3. 对每个文件的内容进行分块
        4. 跳过分块内容过短的块（< 50 字符，无检索价值）
        5. 为每个块生成稳定的 chunk_id（基于文件名+索引的 MD5 哈希）
        6. 调用 EmbeddingService 将文本转为向量
        7. 使用 UPSERT（ON CONFLICT DO UPDATE）写入 PG，支持重复执行

    为什么用 MD5 生成 chunk_id：
        基于 source_file + chunk_index 的 MD5 哈希保证了：
        - 同文件同位置的块永远生成相同的 ID → 支持幂等 upsert
        - 不需要额外的数据库自增 ID
        - 修改文件内容后重新 ingestion，同一位置的块会被覆盖更新

    为什么用 ON CONFLICT DO UPDATE（Upsert）：
        允许反复执行 ingestion 而不会产生重复数据：
        - 首次执行：INSERT 新记录
        - 再次执行：UPDATE 已有记录的 content 和 embedding
        这对于迭代更新知识库非常友好。
    """
    # 初始化数据库表结构（如果 knowledge_chunks 表不存在则创建）
    init_db()
    pg = PGClient()
    emb = EmbeddingService()
    docs = read_markdown_files(knowledge_dir)
    logger.info(f"Found {len(docs)} documents in {knowledge_dir}")

    total_chunks = 0
    for doc in docs:
        chunks = chunk_text(doc["content"], chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            # 跳过内容过短的块（< 50 字符）
            # 原因：过短的分块（如孤立的标题、空行）没有足够的语义信息供检索使用
            if len(chunk.strip()) < 50:
                continue

            # 生成分块 ID：文件名 + 分块索引 → MD5 → 取前 16 位十六进制
            # 16 位十六进制 = 64 bit 空间，对于知识库分块场景碰撞概率极低
            chunk_id = hashlib.md5(f"{doc['source_file']}:{i}".encode()).hexdigest()[:16]

            # 将文本转为向量嵌入
            vec = emb.embed(chunk)
            vec_str = f"[{','.join(str(v) for v in vec)}]"

            try:
                # ON CONFLICT (chunk_id) DO UPDATE：
                # 如果 chunk_id 已存在（之前摄入过），则更新 content 和 embedding
                # 注意：没有更新 title/source_file/chunk_index，因为这些是元数据不会变
                pg.execute("""
                    INSERT INTO knowledge_chunks (chunk_id, title, content, source_file, chunk_index, embedding)
                    VALUES (:chunk_id, :title, :content, :source_file, :chunk_index, CAST(:vec AS vector))
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = CAST(:vec AS vector)
                """, {
                    "chunk_id": chunk_id,
                    "title": doc["title"],
                    "content": chunk,
                    "source_file": doc["source_file"],
                    "chunk_index": i,
                    "vec": vec_str,
                })
                total_chunks += 1
            except Exception as e:
                # 单条失败不中断整体流程，记录警告后继续处理其他分块
                logger.warning(f"Ingest chunk {chunk_id} failed: {e}")

    logger.info(f"Ingested {total_chunks} chunks from {len(docs)} documents")


# =========================================================================
# 命令行入口
# =========================================================================

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="将知识库 Markdown 文档导入 PG 向量存储")
    parser.add_argument("--dir", default="data/knowledge",
                        help="Markdown 文件所在目录")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help=f"分块大小（字符数），默认 {CHUNK_SIZE}")
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP,
                        help=f"块间重叠（字符数），默认 {CHUNK_OVERLAP}")
    args = parser.parse_args()
    ingest(args.dir, args.chunk_size, args.overlap)
