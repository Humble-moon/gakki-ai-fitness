"""
================================================================================
build_knowledge_eval.py —— 用 LLM 从 18 篇知识文档生成评测数据集

流程：
  1. 读取 data/knowledge/ 下所有 .md 文档
  2. 每篇文档发给 LLM，生成 2-3 个可验证的事实性问题
  3. 每个问题标注答案来源文档
  4. 保存到 eval/golden_dataset/knowledge_queries.json

设计意图：
  - 出题人是 LLM（不是开发者），避免"自己出题自己评测"的 bias
  - 面试时展示：原始文档+生成 prompt+人工审核=方法论闭环
================================================================================
"""

import json
import logging
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
from openai import OpenAI

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"
OUTPUT_FILE = PROJECT_ROOT / "eval" / "golden_dataset" / "knowledge_queries.json"

# 只取原创的 18 篇中文知识文档（排除 pubmed_ 开头的英文原文）
EXCLUDE_PREFIX = "pubmed_"

GENERATE_PROMPT = """你是一个健身领域的评测专家。下面是一篇健身知识文档的内容。

请基于这篇文档，生成 3 个能用该文档内容回答的事实性问题。

要求：
1. 问题必须是普通健身爱好者会真实问到的问题（口语化、自然）
2. 每个问题必须能在这篇文档中找到明确答案
3. 问题要有实际应用价值，不要问"XX的定义是什么"这类教科书问题
4. 返回纯 JSON 数组，不要包含任何其他内容

返回格式：
[
  {{"question": "问题文本", "answer_source": "文档中的答案段落摘要", "topic": "主题标签"}},
  ...
]

文档标题：{title}

文档内容：
{content}"""


def load_documents():
    docs = {}
    for filepath in sorted(KNOWLEDGE_DIR.glob("*.md")):
        if filepath.name.startswith(EXCLUDE_PREFIX):
            continue
        text = filepath.read_text(encoding="utf-8")
        title_match = re.search(r"^# (.+)$", text, re.MULTILINE)
        title = title_match.group(1) if title_match else filepath.stem
        docs[filepath.name] = {"title": title, "content": text, "file": filepath.name}
    return docs


def generate_questions(doc_title, doc_content, client):
    prompt = GENERATE_PROMPT.format(title=doc_title, content=doc_content[:3000])
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2000,
    )
    raw = response.choices[0].message.content.strip()
    # 清理 markdown 代码块包装
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def main():
    docs = load_documents()
    logger.info(f"加载了 {len(docs)} 篇知识文档")

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    all_questions = []
    qid = 0

    for filename, doc in docs.items():
        logger.info(f"处理: {doc['title']}")
        try:
            qs = generate_questions(doc["title"], doc["content"], client)
            for q in qs:
                qid += 1
                all_questions.append({
                    "id": f"k{qid:03d}",
                    "query": q["question"],
                    "category": "knowledge",
                    "source_doc": filename,
                    "source_title": doc["title"],
                    "relevant_doc_ids": [filename],
                    "answer_summary": q.get("answer_source", ""),
                    "topic": q.get("topic", ""),
                    "eval_type": "knowledge_retrieval",
                    "difficulty": "medium",
                })
            logger.info(f"  -> 生成 {len(qs)} 个问题")
        except Exception as e:
            logger.error(f"  ! 失败 [{doc['title']}]: {e}")
            continue

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(all_questions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"保存 {len(all_questions)} 个问题到 {OUTPUT_FILE}")

    # 打印统计
    topics = {}
    for q in all_questions:
        t = q["topic"]
        topics[t] = topics.get(t, 0) + 1
    logger.info(f"话题分布: {json.dumps(topics, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
