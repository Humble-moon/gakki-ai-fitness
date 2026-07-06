"""
================================================================================
fetch_knowledge.py —— 专业健身知识爬取管线
================================================================================
数据源（按权威性排序）：
  1. PubMed / PMC — 经同行评审的运动科学文献，通过 Entrez API 免费获取
  2. WHO — 世界卫生组织身体活动指南（中文版）
  3. 中国国家体育总局 — 全民健身指南（中文）

输出：data/knowledge/ 目录下的 Markdown 文件
  爬取原文 → data/knowledge_fetched/raw/ （原始数据）
  LLM 加工后 → data/knowledge/ （可直接入库的 .md 文件）

使用方式：
  # 仅爬取，不翻译
  python scripts/fetch_knowledge.py --fetch-only

  # 爬取 + LLM 翻译改写为中文知识文章
  python scripts/fetch_knowledge.py --translate

  # 仅爬特定源
  python scripts/fetch_knowledge.py --source pubmed
  python scripts/fetch_knowledge.py --source who
================================================================================
"""

import json
import logging
import os
import re
import time
import urllib.parse
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# =========================================================================
# 配置
# =========================================================================

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "knowledge_fetched" / "raw"
KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"

# PubMed 搜索主题 → 对应中文标题前缀
PUBMED_TOPICS = {
    "resistance training hypertrophy": ("增肌训练", "zengji"),
    "protein intake muscle synthesis": ("蛋白质与肌肉合成", "danbaizhi"),
    "carbohydrate exercise performance": ("碳水与运动表现", "tanshui"),
    "exercise injury prevention strength training": ("力量训练伤病预防", "shangbing"),
    "warm up cool down exercise": ("热身与放松", "renshen"),
    "training periodization strength": ("周期化训练", "zhouqihua"),
    "delayed onset muscle soreness recovery": ("肌肉恢复", "huifu"),
    "concurrent training strength endurance": ("混合训练", "hunhe"),
}

WHO_FACT_SHEETS = [
    ("https://www.who.int/zh/news-room/fact-sheets/detail/physical-activity",
     "WHO 身体活动指南"),
]

SPORT_GOV_URLS = [
    "https://www.sport.gov.cn/n315/n331/n401/c785782/content.html",
]

# PubMed API 限速：每秒最多 3 次请求（无 API key 时）
PUBMED_DELAY = 0.35


# =========================================================================
# PubMed 爬取
# =========================================================================

class PubMedFetcher:
    """通过 NCBI Entrez API 搜索和获取 PubMed 文献摘要。

    速率限制：
      - 无 API key: 每秒 3 次请求
      - 有 API key: 每秒 10 次请求
    如需 API key: https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/
    """

    BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, api_key: str = "", tool_name: str = "gakki-fitness-kb"):
        self.api_key = api_key
        self.tool_name = tool_name
        self.session = requests.Session()

    def search(self, query: str, max_results: int = 15) -> list[str]:
        """搜索 PubMed 并返回 PMID 列表。"""
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
            "tool": self.tool_name,
        }
        if self.api_key:
            params["api_key"] = self.api_key

        resp = self.session.get(f"{self.BASE}/esearch.fcgi", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])
        logger.info(f"PubMed search '{query[:50]}...': {len(pmids)} results (total: {data.get('esearchresult', {}).get('count', '?')})")
        return pmids

    def fetch_abstracts(self, pmids: list[str]) -> list[dict]:
        """批量获取摘要。返回 [{"pmid": ..., "title": ..., "abstract": ..., "journal": ..., "year": ...}]。"""
        if not pmids:
            return []

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "xml",
            "tool": self.tool_name,
        }
        if self.api_key:
            params["api_key"] = self.api_key

        resp = self.session.get(f"{self.BASE}/efetch.fcgi", params=params, timeout=30)
        resp.raise_for_status()

        articles = []
        # 简易 XML 解析（避免依赖 lxml）
        text = resp.text
        # 按 <PubmedArticle> 分割
        for article_xml in text.split("<PubmedArticle>"):
            if "</PubmedArticle>" not in article_xml:
                continue
            article_xml = article_xml.split("</PubmedArticle>")[0]

            pmid = self._extract_tag(article_xml, "PMID")
            title = self._extract_tag(article_xml, "ArticleTitle")
            abstract = self._extract_tag(article_xml, "AbstractText")
            journal = self._extract_tag(article_xml, "Title")
            year = self._extract_tag(article_xml, "PubDate")

            # 从 PubDate 中提取年份
            year_match = re.search(r"(\d{4})", year or "")
            if year_match:
                year = year_match.group(1)

            if title and abstract:
                articles.append({
                    "pmid": pmid,
                    "title": self._clean_text(title),
                    "abstract": self._clean_text(abstract),
                    "journal": journal or "",
                    "year": year or "",
                    "source_type": "pubmed",
                })

        return articles

    def _extract_tag(self, xml_str: str, tag: str) -> str:
        m = re.search(f"<{tag}[^>]*>(.*?)</{tag}>", xml_str, re.DOTALL)
        return m.group(1) if m else ""

    def _clean_text(self, text: str) -> str:
        """清理 XML 转义和多余空白。"""
        text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
        text = re.sub(r"\s+", " ", text).strip()
        return text


# =========================================================================
# WHO 爬取
# =========================================================================

class WHOFetcher:
    """获取 WHO 中文资料。"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; gakki-fitness-bot/1.0; educational use)"
        })

    def fetch_fact_sheet(self, url: str, title: str) -> Optional[dict]:
        """获取单篇 WHO 中文 factsheet。返回 {"title": ..., "content": ..., "source_url": ...}"""
        resp = self.session.get(url, timeout=20)
        if resp.status_code != 200:
            logger.warning(f"WHO fetch failed ({resp.status_code}): {url}")
            return None

        html = resp.text
        # 提取标题
        title_match = re.search(r"<h1[^>]*>(.*?)</h1>", html)
        content_title = title_match.group(1) if title_match else title

        # 提取正文：<article> 或 <div class="content"> 或 <main>
        body = ""
        for tag in [r'<article[^>]*>(.*?)</article>',
                     r'<main[^>]*>(.*?)</main>',
                     r'<div[^>]*class="[^"]*content[^"]*"[^>]*>(.*?)</div>']:
            m = re.search(tag, html, re.DOTALL)
            if m:
                body = m.group(1)
                break
        if not body:
            body = html

        # 去除 HTML 标签
        text = _strip_html(body)
        # 去除多余空行
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        # 截取合理长度（WHO factsheet 一般 1000-3000 字）
        if len(text) > 5000:
            text = text[:5000] + "\n\n[... 内容已截断，完整版见来源 URL ...]"

        logger.info(f"WHO fetched: {content_title} ({len(text)} chars)")
        return {
            "title": content_title,
            "content": text,
            "source_url": url,
            "source_type": "who",
        }


# =========================================================================
# 格式化输出
# =========================================================================

def save_raw_article(article: dict, filename: str):
    """保存原始文章到 data/knowledge_fetched/raw/"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    filepath = RAW_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {article['title']}\n\n")
        f.write(f"**来源**: {article.get('source_type', '')} | "
                f"**期刊**: {article.get('journal', '')} | "
                f"**年份**: {article.get('year', '')}\n")
        if article.get("pmid"):
            f.write(f"**PMID**: {article['pmid']} | "
                    f"**URL**: https://pubmed.ncbi.nlm.nih.gov/{article['pmid']}/\n")
        if article.get("source_url"):
            f.write(f"**URL**: {article['source_url']}\n")
        f.write("\n---\n\n")
        f.write(article.get("abstract") or article.get("content", ""))

    return filepath


def build_translation_prompt(article: dict) -> str:
    """构建用于 LLM 翻译改写的提示词。

    不只是翻译，而是从学术摘要改写为"健身科普文章"：
    - 去掉统计学细节（p值、置信区间等）
    - 保留核心结论和实用建议
    - 用口语化中文，面向普通健身爱好者
    - 补充必要的背景知识
    """
    source_type = article.get("source_type", "")

    if source_type == "pubmed":
        return f"""你将一篇英文学术论文摘要改写成中文健身科普短文。

要求：
1. 用口语化中文，面向普通健身爱好者（不是研究者）
2. 去掉统计学术语（p值、置信区间、样本量等），保留核心结论
3. 从结论中提炼 2-3 条可操作的训练建议
4. 250-400 字
5. 格式如下：

---
title: {article['title']}
source: PubMed PMID {article.get('pmid', '')}
journal: {article.get('journal', '')}
year: {article.get('year', '')}
---

# [用中文重新拟一个吸引人的标题]

[科普内容，用"你"称呼读者]

## 核心结论

- 结论1
- 结论2

## 对你的训练有什么启发

1. 可操作建议1
2. 可操作建议2

---
*原文: {article['title']} ({article.get('journal', '')}, {article.get('year', '')})*
*PMID: {article['pmid']}*

---

英文原文摘要：
{article.get('abstract', article.get('content', ''))}"""
    else:
        # WHO 等中文源，只需格式化和提炼
        return f"""将以下内容整理成一篇健身科普文章。

要求：
1. 用口语化中文，像教练在讲给学员听
2. 300-500 字
3. 提炼核心要点，去掉冗余表述

格式：
---
title: {article['title']}
source: {article.get('source_url', '')}
source_type: {source_type}
---

# [保持原标题或重新拟题]

[科普内容]

---

*来源: {article.get('source_url', '')}*

---

原文：
{article.get('abstract') or article.get('content', '')}"""


def _strip_html(html: str) -> str:
    """去除 HTML 标签，保留纯文本。"""
    # 移除 script/style
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL)
    # 移除标签
    text = re.sub(r"<[^>]+>", " ", text)
    # 解码常见 HTML 实体
    text = text.replace("&nbsp;", " ").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'")
    # 压缩空白
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# =========================================================================
# 主流程
# =========================================================================

def fetch_pubmed_articles(pubmed: PubMedFetcher) -> list[dict]:
    """从 PubMed 爬取所有主题的文章。"""
    all_articles = []
    seen_pmids = set()

    for query, (cn_title, file_prefix) in PUBMED_TOPICS.items():
        logger.info(f"--- PubMed: {query} ---")
        try:
            pmids = pubmed.search(query, max_results=8)
            time.sleep(PUBMED_DELAY)
            new_pmids = [p for p in pmids if p not in seen_pmids]
            seen_pmids.update(new_pmids)

            if new_pmids:
                articles = pubmed.fetch_abstracts(new_pmids)
                time.sleep(PUBMED_DELAY)
                for article in articles:
                    article["topic_cn"] = cn_title
                    article["file_prefix"] = file_prefix
                all_articles.extend(articles)
                logger.info(f"  → {len(articles)} articles saved for '{cn_title}'")
        except Exception as e:
            logger.error(f"PubMed topic '{query}' failed: {e}")

    return all_articles


def fetch_who_articles(who: WHOFetcher) -> list[dict]:
    """从 WHO 爬取中文资料。"""
    articles = []
    for url, title in WHO_FACT_SHEETS:
        logger.info(f"--- WHO: {title} ---")
        try:
            article = who.fetch_fact_sheet(url, title)
            if article:
                articles.append(article)
        except Exception as e:
            logger.error(f"WHO '{title}' failed: {e}")
    return articles


def fetch_all_sources() -> list[dict]:
    """爬取所有源，返回格式化后的文章列表。"""
    all_articles = []

    # 1. PubMed
    logger.info("=" * 60)
    logger.info("Phase 1/2: PubMed 运动科学文献")
    logger.info("=" * 60)
    pubmed = PubMedFetcher()
    pubmed_articles = fetch_pubmed_articles(pubmed)
    all_articles.extend(pubmed_articles)
    logger.info(f"PubMed total: {len(pubmed_articles)} articles")

    # 2. WHO
    logger.info("=" * 60)
    logger.info("Phase 2/2: WHO 中文资料")
    logger.info("=" * 60)
    who = WHOFetcher()
    who_articles = fetch_who_articles(who)
    all_articles.extend(who_articles)
    logger.info(f"WHO total: {len(who_articles)} articles")

    return all_articles


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="爬取专业健身知识")
    parser.add_argument("--source", choices=["pubmed", "who", "all"], default="all",
                        help="指定数据源 (default: all)")
    parser.add_argument("--translate", action="store_true",
                        help="同时生成 LLM 翻译提示词文件，供后续 translate_knowledge.py 处理")
    parser.add_argument("--max-pubmed", type=int, default=8,
                        help="每个 PubMed 主题最多取几篇 (default: 8)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    articles = fetch_all_sources()

    # 保存原始文档
    pubmed_count = 0
    who_count = 0
    for i, article in enumerate(articles):
        source = article.get("source_type", "")
        if source == "pubmed":
            pmid = article.get("pmid", f"article_{i}")
            prefix = article.get("file_prefix", "unknown")
            filename = f"pubmed_{prefix}_{pmid}.md"
            pubmed_count += 1
        elif source == "who":
            safe_title = re.sub(r"[^\w\u4e00-\u9fff]", "_", article["title"])[:40]
            filename = f"who_{safe_title}.md"
            who_count += 1
        else:
            continue

        filepath = save_raw_article(article, filename)
        logger.info(f"Saved: {filepath}")

    print()
    print(f"爬取完成！")
    print(f"  PubMed: {pubmed_count} 篇")
    print(f"  WHO: {who_count} 篇")
    print(f"  总计: {pubmed_count + who_count} 篇")
    print(f"  保存位置: {RAW_DIR}")

    if args.translate:
        # 生成翻译提示词文件
        prompts_dir = RAW_DIR / "translation_prompts"
        prompts_dir.mkdir(exist_ok=True)
        for i, article in enumerate(articles):
            prompt = build_translation_prompt(article)
            filename = f"prompt_{article.get('source_type', '')}_{i:03d}.txt"
            with open(prompts_dir / filename, "w", encoding="utf-8") as f:
                f.write(prompt)
        print(f"  翻译提示词: {prompts_dir} ({len(articles)} 个)")
        print()
        print("下一步：将提示词发送给 LLM 翻译，输出存入 data/knowledge/，然后运行：")
        print("  python -m src.rag.knowledge_ingestion --dir data/knowledge")
