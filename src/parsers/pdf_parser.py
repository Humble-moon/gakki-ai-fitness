"""
PDF 解析器 — 基于 pdfplumber

提取文本 + 表格(Markdown) + 字号信息。
表格转 Markdown 格式嵌入正文，LLM 可直接理解。
"""

import io
from typing import NamedTuple


class ParsedPage(NamedTuple):
    page_number: int
    text: str
    tables_md: list[str]   # 每张表格的 Markdown 表示
    font_sizes: list[float]  # 该页出现的所有字号，降序


class PDFParseResult(NamedTuple):
    pages: list[ParsedPage]
    total_chars: int
    has_text: bool           # False = 可能是扫描件
    error: str | None


def parse_pdf(file_bytes: bytes) -> PDFParseResult:
    """解析 PDF 字节流，返回按页的结构化数据。"""
    import pdfplumber

    pages: list[ParsedPage] = []
    total_chars = 0
    error = None

    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                # 1. 提取纯文本
                text = (page.extract_text() or "").strip()

                # 2. 提取表格 → Markdown
                tables_md = []
                raw_tables = page.extract_tables()
                if raw_tables:
                    for table in raw_tables:
                        md = _table_to_markdown(table)
                        if md:
                            tables_md.append(md)

                # 3. 收集字号
                font_sizes: list[float] = []
                if page.chars:
                    sizes = {round(c.get("size", 0), 1) for c in page.chars if c.get("size")}
                    font_sizes = sorted(sizes, reverse=True)

                total_chars += len(text)
                pages.append(ParsedPage(
                    page_number=i + 1,
                    text=text,
                    tables_md=tables_md,
                    font_sizes=font_sizes,
                ))

    except Exception as e:
        error = f"PDF 解析失败: {e}"

    return PDFParseResult(
        pages=pages,
        total_chars=total_chars,
        has_text=total_chars > 20,  # 少于 20 字符视为无文本（扫描件）
        error=error,
    )


def _table_to_markdown(table: list[list[str | None]]) -> str:
    """将 pdfplumber 提取的二维表格转为 Markdown 表格字符串。"""
    if not table or not table[0]:
        return ""

    # 过滤全空行
    rows = [[str(c or "") for c in row] for row in table if any(c for c in row)]
    if len(rows) < 2:
        return ""

    header = rows[0]
    sep = ["---"] * len(header)
    body = rows[1:]

    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(sep) + "|",
    ]
    for row in body:
        # 补齐列数（某些表格行可能缺少列）
        padded = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(padded[:len(header)]) + " |")

    return "\n".join(lines)


def detect_body_font_size(pages: list[ParsedPage]) -> float:
    """从所有页面的字号中推断正文字号。
    出现频率最高的字号 = 正文字号。
    """
    from collections import Counter
    counter: Counter[float] = Counter()
    for p in pages:
        for s in p.font_sizes:
            counter[s] += 1
    if counter:
        return counter.most_common(1)[0][0]
    return 10.0  # 默认 10pt


def pages_to_full_text(pages: list[ParsedPage]) -> str:
    """将解析后的页面合并为完整文本（含表格）。
    每页标注页码，表格以 Markdown 嵌入。
    """
    parts: list[str] = []
    for p in pages:
        page_header = f"\n--- 第 {p.page_number} 页 ---\n"
        parts.append(page_header + p.text)
        for table_md in p.tables_md:
            parts.append(f"\n[表格 - 第 {p.page_number} 页]\n{table_md}\n")
    return "\n".join(parts)
