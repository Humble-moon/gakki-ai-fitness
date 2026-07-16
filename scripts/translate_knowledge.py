"""
================================================================================
translate_knowledge.py —— 将爬取的英文文献翻译改写为中文健身科普文章
================================================================================
输入：data/knowledge_fetched/raw/*.md（fetch_knowledge.py 的产出）
输出：data/knowledge/pubmed_*.md（可直接入库的中文知识文档）

处理流程：
  1. 读取所有原始文献 .md 文件
  2. 每篇文献构造翻译+改写提示词
  3. 调用项目 LLM 将学术摘要改写为中文健身科普短文
  4. 保存到 data/knowledge/ 目录
  5. 去重：跳过已翻译过的 PMID

使用方式：
  python scripts/translate_knowledge.py              # 翻译所有
  python scripts/translate_knowledge.py --dry-run    # 预览不调用 LLM
  python scripts/translate_knowledge.py --limit 5    # 只翻译前 5 篇
================================================================================
"""

import json
import logging
import re
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "knowledge_fetched" / "raw"
KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"
TRANSLATED_INDEX = RAW_DIR / "translated_index.json"  # 记录已翻译的 PMID


def load_translated_index() -> set:
    """加载已翻译的 PMID 集合，避免重复翻译。"""
    if TRANSLATED_INDEX.exists():
        return set(json.loads(TRANSLATED_INDEX.read_text(encoding="utf-8")))
    return set()


def save_translated_index(translated: set):
    TRANSLATED_INDEX.write_text(json.dumps(list(translated), ensure_ascii=False), encoding="utf-8")


def parse_raw_article(filepath: Path) -> dict | None:
    """解析原始文献 .md 文件，提取元数据和原文。"""
    text = filepath.read_text(encoding="utf-8")

    article = {
        "filename": filepath.name,
        "title": "",
        "pmid": "",
        "journal": "",
        "year": "",
        "source_url": "",
        "source_type": "",
        "abstract": "",
    }

    # 提取标题（第一行 # xxx）
    title_match = re.search(r"^# (.+)$", text, re.MULTILINE)
    if title_match:
        article["title"] = title_match.group(1).strip()

    # 提取元数据行
    pmid_match = re.search(r"\*\*PMID\*\*:\s*(\d+)", text)
    if pmid_match:
        article["pmid"] = pmid_match.group(1)

    journal_match = re.search(r"\*\*期刊\*\*:\s*(.+?)\s*\*?\*", text)
    if journal_match:
        article["journal"] = journal_match.group(1).strip().rstrip("*").strip()

    year_match = re.search(r"\*\*年份\*\*:\s*(\d{4})", text)
    if year_match:
        article["year"] = year_match.group(1)

    url_match = re.search(r"\*\*URL\*\*:\s*(https?://[^\s]+)", text)
    if url_match:
        article["source_url"] = url_match.group(1)

    source_match = re.search(r"\*\*来源\*\*:\s*(\w+)", text)
    if source_match:
        article["source_type"] = source_match.group(1)

    # 提取摘要正文（--- 之后的内容）
    parts = text.split("---", 2)
    if len(parts) >= 3:
        article["abstract"] = parts[2].strip()
    elif len(parts) == 2:
        article["abstract"] = parts[1].strip()
    else:
        article["abstract"] = text

    # 截断过长摘要（节省 LLM token）
    if len(article["abstract"]) > 2500:
        article["abstract"] = article["abstract"][:2500] + "..."

    return article if article["abstract"] else None


def build_prompt(article: dict) -> str:
    """构建翻译改写提示词：英文学术摘要 → 中文健身科普短文。"""
    return f"""你是一位运动科学科普作家。请将以下英文学术论文摘要改写成一篇中文健身科普短文。

改写要求：
1. 用口语化中文，面向普通健身爱好者（不是研究者）
2. 去掉统计学术语（p值、置信区间、样本量n等），保留核心结论
3. 从结论中提炼 2-3 条可操作的训练建议
4. 250-400 字
5. 在文末标注原文出处

输出格式（严格按此格式）：

---
title: [用中文重新拟一个吸引人的标题，15字以内]
source: PubMed PMID {article.get('pmid', '')}
journal: {article.get('journal', '')}
year: {article.get('year', '')}
---

# [中文标题]

[科普内容，用"你"称呼读者，像教练在讲给学员听]

## 核心结论

- 结论1（一句话，不要数据）
- 结论2（一句话，不要数据）

## 对你的训练有什么启发

1. [可操作建议1]
2. [可操作建议2]

---
*英文原文: {article.get('title', '')}*
*期刊: {article.get('journal', '')} ({article.get('year', '')})*
*PMID: {article.get('pmid', '')}*
*URL: https://pubmed.ncbi.nlm.nih.gov/{article.get('pmid', '')}/*

---

英文摘要原文：
{article.get('abstract', '')}"""


def translate_articles(llm, limit: int = None, dry_run: bool = False):
    """翻译所有原始文献。

    Args:
        llm: LLMProvider 实例
        limit: 最多翻译几篇（None = 全部）
        dry_run: True 时只打印不调用 LLM
    """
    if not RAW_DIR.exists():
        logger.error(f"原始文献目录不存在: {RAW_DIR}")
        logger.error("请先运行: python scripts/fetch_knowledge.py")
        return

    translated = load_translated_index()
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

    # 收集所有待翻译文件（排除 prompts 子目录）
    raw_files = sorted(
        [f for f in RAW_DIR.glob("*.md") if f.is_file()],
        key=lambda f: f.name,
    )

    if not raw_files:
        logger.warning("没有找到任何原始文献 .md 文件")
        return

    logger.info(f"找到 {len(raw_files)} 篇原始文献")

    success_count = 0
    skip_count = 0
    fail_count = 0

    for i, filepath in enumerate(raw_files):
        if limit and i >= limit:
            break

        article = parse_raw_article(filepath)
        if not article or not article["abstract"]:
            logger.warning(f"跳过空文件: {filepath.name}")
            continue

        # 去重检查
        article_id = article.get("pmid") or article["filename"]
        if article_id in translated:
            skip_count += 1
            continue

        logger.info(f"[{i+1}/{len(raw_files)}] 翻译: {article['title'][:60]}...")

        if dry_run:
            logger.info(f"  PMID: {article.get('pmid', 'N/A')}")
            logger.info(f"  摘要长度: {len(article['abstract'])} 字符")
            continue

        try:
            prompt = build_prompt(article)
            response = llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
            )
            content = response.content

            # 保存中文知识文档
            source = article.get("source_type", "pubmed")
            if article.get("pmid"):
                out_filename = f"{source}_{article['pmid']}.md"
            else:
                safe_name = re.sub(r"[^\w\u4e00-\u9fff-]", "_", article["filename"])
                out_filename = f"{source}_{safe_name}"

            out_path = KNOWLEDGE_DIR / out_filename
            out_path.write_text(content, encoding="utf-8")

            # 记录已翻译
            translated.add(article_id)
            save_translated_index(translated)

            success_count += 1
            logger.info(f"  ✓ -> {out_filename}")

            # 限速（避免 API 限流）
            time.sleep(1)

        except Exception as e:
            fail_count += 1
            logger.error(f"  ✗ 翻译失败: {e}")

    print()
    print(f"翻译完成: {success_count} 篇成功, {skip_count} 篇跳过, {fail_count} 篇失败")
    print(f"中文文档位置: {KNOWLEDGE_DIR}")

    if success_count > 0:
        print()
        print("下一步 —— 将新文档摄入向量数据库:")
        print(f"  python -m src.rag.knowledge_ingestion --dir {KNOWLEDGE_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="翻译 PubMed 文献为中文健身科普文章")
    parser.add_argument("--limit", type=int, default=None, help="最多翻译几篇")
    parser.add_argument("--dry-run", action="store_true", help="预览模式，不调用 LLM")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # 初始化 LLMProvider
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from src.llm.provider import LLMProvider
        llm = LLMProvider()
        logger.info(f"LLM 已初始化: model={llm.active_model}")
    except Exception as e:
        if not args.dry_run:
            logger.error(f"LLM 初始化失败: {e}")
            logger.error("请确保 DEEPSEEK_API_KEY 已配置在 .env 文件中")
            sys.exit(1)
        llm = None

    translate_articles(llm, limit=args.limit, dry_run=args.dry_run)
