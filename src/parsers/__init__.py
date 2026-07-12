"""
文档解析器统一入口。

支持 PDF / Word (.docx) / Markdown (.md) 三种格式。
暴露 parse_file(file_bytes, filename) → ParsedDocument 作为唯一对外接口。
"""

from typing import NamedTuple


class ParsedDocument(NamedTuple):
    """解析后的统一文档结构。"""
    filename: str
    file_type: str          # "pdf" | "docx" | "md"
    full_text: str          # 全文（含 Markdown 表格、页码标注）
    page_count: int         # 页数（PDF）或 1（非 PDF）
    total_chars: int
    title: str              # 文档标题（从内容推测）
    has_text: bool          # 是否有可提取的文字（False = 扫描件）
    error: str | None


def parse_file(file_bytes: bytes, filename: str) -> ParsedDocument:
    """根据文件扩展名选择解析器，返回统一结构。

    输入：
        file_bytes: bytes  — 文件二进制内容
        filename: str      — 原始文件名（用于判断类型）

    输出：
        ParsedDocument — 包含全文、页数、标题等
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "pdf":
        return _parse_pdf(file_bytes, filename)
    elif ext in ("docx", "doc"):
        return _parse_docx(file_bytes, filename)
    elif ext in ("md", "markdown", "txt"):
        return _parse_md(file_bytes, filename)
    else:
        return ParsedDocument(
            filename=filename,
            file_type=ext,
            full_text="",
            page_count=0,
            total_chars=0,
            title=filename,
            has_text=False,
            error=f"不支持的文件格式: .{ext}，支持 PDF/Word/MD",
        )


def _parse_pdf(file_bytes: bytes, filename: str) -> ParsedDocument:
    from src.parsers.pdf_parser import (
        parse_pdf, pages_to_full_text, detect_body_font_size,
    )
    result = parse_pdf(file_bytes)

    title = filename
    if result.has_text and result.pages:
        # 用第一页的第一行非空文本当标题
        first_line = result.pages[0].text.split("\n")[0].strip()
        if first_line and len(first_line) < 100:
            title = first_line

    return ParsedDocument(
        filename=filename,
        file_type="pdf",
        full_text=pages_to_full_text(result.pages) if result.has_text else "",
        page_count=len(result.pages),
        total_chars=result.total_chars,
        title=title,
        has_text=result.has_text,
        error=result.error,
    )


def _parse_docx(file_bytes: bytes, filename: str) -> ParsedDocument:
    from src.parsers.docx_parser import parse_docx
    result = parse_docx(file_bytes)

    return ParsedDocument(
        filename=filename,
        file_type="docx",
        full_text=result.full_text,
        page_count=1,  # Word 不解析页数
        total_chars=result.total_chars,
        title=result.title or filename,
        has_text=result.total_chars > 10,
        error=result.error,
    )


def _parse_md(file_bytes: bytes, filename: str) -> ParsedDocument:
    """MD/TXT 直接读文本，提取第一个 # 标题。"""
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = file_bytes.decode("gbk")
        except Exception:
            text = file_bytes.decode("latin-1")

    # 提取标题
    title = filename
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            break

    return ParsedDocument(
        filename=filename,
        file_type="md",
        full_text=text,
        page_count=1,
        total_chars=len(text),
        title=title,
        has_text=len(text.strip()) > 5,
        error=None,
    )
