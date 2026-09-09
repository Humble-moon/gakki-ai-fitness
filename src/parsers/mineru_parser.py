"""
=============================================================================
mineru_parser.py — MinerU 文档解析后端（可选，默认关闭）
=============================================================================
【为什么要有这个后端】
    pdfplumber 的 extract_tables() 依赖 PDF 里的显式表格线框。体检报告恰恰是
    它的三个盲区叠在一起：扫描件（无文本层）、无框线表格、跨页表格。pdf_parser.py
    里的 NOTE 已经承认「扫描版 PDF 需要 OCR，当前未集成」。

    MinerU（OpenDataLab 开源，github.com/opendatalab/MinerU）做的是
    「理解版面 → 识别内容 → 重建语义」：版面分析模型分区、OCR 识别、
    表格重建为 HTML/Markdown、公式转 LaTeX、自动阅读顺序、标题分级。
    跨页表格与扫描件正是它相对 pdfplumber 的代差优势。
    附带收益：结构化后的文本比原始 PDF 短得多，入库 token 成本显著下降。

【为什么用 CLI 子进程，而不是常驻 HTTP server 或装进项目 venv】
    本项目跑在 16GB 内存的开发机上，同时还要开 PG / Neo4j / Redis / MinIO
    四个容器。三种接入方式的取舍：
      · mineru[all] 装进项目 venv —— 会把 torch 等重依赖拉进主环境，
        影响既有离线测试与「docker compose up 即可跑」的轻量体验；
      · 常驻 mineru-api HTTP server —— 版面分析模型常驻内存，
        在 16GB 机器上与四个容器争资源；
      · CLI 子进程（本实现）—— 用时才起进程、用完即释放，主环境零新增依赖，
        且 CLI 的 -p/-o/-b 参数跨版本比 HTTP 响应结构稳定得多。
    代价是每次调用有进程与模型加载的冷启动开销（秒级）。文件上传 RAG 是
    低频操作，这个代价可以接受；真要高频批处理再切常驻 server。

【隐私边界】
    全程本地：PDF 写入临时目录、子进程本地执行、产物本地读取、退出即清理，
    没有任何一步把用户上传的体检报告发往外部服务。这与项目
    「数据不出本地 Docker / 本机」的整体口径一致。
    官方另有 mineru-open-sdk 云端方案（依赖极轻，只需 httpx），但那需要把
    文档上传到第三方服务——体检报告属《个人信息保护法》下的敏感个人信息，
    本项目不接入该路径。

【降级语义】
    未安装 mineru、执行超时、非零退出、找不到 .md 产物——任何一种情况都返回
    None，由 parsers/__init__.py 回落 pdfplumber。解析后端是增强项，
    绝不能因为它让文件上传功能整体不可用。

【被谁调用】src/parsers/__init__.py 的 _parse_pdf()
=============================================================================
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, NamedTuple, Optional

from src.config import (
    MINERU_CLI,
    MINERU_BACKEND,
    MINERU_TIMEOUT,
    MINERU_MODEL_SOURCE,
)

logger = logging.getLogger(__name__)

# 上传文件在临时目录里统一用这个固定名字。
# 不拿用户原始文件名去拼路径：一方面避免路径穿越与特殊字符，
# 另一方面让 MinerU 的输出子目录名可预期。
STAGED_FILENAME = "upload.pdf"


class MinerUResult(NamedTuple):
    full_text: str
    page_count: int
    total_chars: int
    title: str
    has_text: bool
    error: Optional[str]


def is_available() -> bool:
    """mineru 可执行文件是否就绪。

    只做 which 探测，不触发模型下载、不起进程，因此可以放心地在每次上传时调用。
    """
    return bool(MINERU_CLI) and shutil.which(MINERU_CLI) is not None


def parse_pdf(file_bytes: bytes, filename: str = STAGED_FILENAME) -> Optional[MinerUResult]:
    """用 MinerU 解析 PDF。成功返回 MinerUResult，任何失败返回 None（由调用方降级）。

    流程：
        1. 字节流写入临时目录（固定文件名，不采用用户原始名）
        2. 子进程执行 mineru -p <in> -o <out> -b <backend>
        3. 在输出目录里递归定位 Markdown 产物
        4. 从 content_list.json 推断页数（拿不到则为 0）
        5. finally 清理临时目录
    """
    if not file_bytes:
        return None
    if not is_available():
        logger.info(f"未检测到 {MINERU_CLI} 可执行文件，MinerU 后端不可用")
        return None

    with tempfile.TemporaryDirectory(prefix="gakki-mineru-") as workdir:
        in_path = Path(workdir) / STAGED_FILENAME
        out_dir = Path(workdir) / "out"
        try:
            in_path.write_bytes(file_bytes)
            out_dir.mkdir(parents=True, exist_ok=True)

            completed = _run_cli(in_path, out_dir)
            if completed is None:
                return None
            if not _exit_ok(completed):
                # 非零退出不等于毫无产出（可能只是个别页失败）：
                # 记一条警告后继续尝试读产物，读不到再降级。
                logger.warning(
                    f"MinerU 退出码 {completed.returncode}，仍尝试读取已产出内容。"
                    f"stderr: {(completed.stderr or '')[-300:]}"
                )

            markdown = _locate_markdown(out_dir)
            if markdown is None:
                logger.warning(
                    "MinerU 执行结束但未找到 Markdown 产物，回落 pdfplumber。"
                    f"stderr: {completed.stderr[-300:] if completed.stderr else '(空)'}"
                )
                return None

            full_text = markdown.read_text(encoding="utf-8", errors="replace").strip()
            if not full_text:
                logger.warning("MinerU 产出的 Markdown 为空，回落 pdfplumber")
                return None

            return MinerUResult(
                full_text=full_text,
                page_count=_infer_page_count(out_dir),
                total_chars=len(full_text),
                title=_extract_title(full_text) or filename,
                has_text=True,
                error=None,
            )
        except Exception as exc:  # 任何意外都不应冒泡到上传接口
            logger.warning(f"MinerU 解析异常，回落 pdfplumber: {exc}")
            return None


# ----------------------------------------------------------------------
# 子进程执行
# ----------------------------------------------------------------------
def _run_cli(in_path: Path, out_dir: Path) -> Optional[subprocess.CompletedProcess]:
    """执行 mineru CLI。超时或非零退出返回 None。"""
    cmd = [
        MINERU_CLI,
        "-p", str(in_path),
        "-o", str(out_dir),
        "-b", MINERU_BACKEND,
    ]
    # 继承当前环境并指定模型源。国内默认走 ModelScope，避免 HuggingFace 超时
    # ——这正是项目当初从本地 BGE 切到 DashScope API 的同一个网络问题。
    env = dict(os.environ)
    if MINERU_MODEL_SOURCE:
        env["MINERU_MODEL_SOURCE"] = MINERU_MODEL_SOURCE

    logger.info(f"MinerU 解析中（backend={MINERU_BACKEND}, timeout={MINERU_TIMEOUT}s）...")
    try:
        # 不用 shell=True：参数以列表传递，文件名不参与 shell 解析
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=MINERU_TIMEOUT,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            f"MinerU 执行超过 {MINERU_TIMEOUT}s 未返回，回落 pdfplumber。"
            "纯 CPU 的 pipeline 后端较慢，可提高 MINERU_TIMEOUT 或减少页数。"
        )
        return None
    except OSError as exc:
        logger.warning(f"MinerU 进程启动失败，回落 pdfplumber: {exc}")
        return None


def _exit_ok(completed: subprocess.CompletedProcess) -> bool:
    return completed.returncode == 0


# ----------------------------------------------------------------------
# 产物定位
# ----------------------------------------------------------------------
def _locate_markdown(out_dir: Path) -> Optional[Path]:
    """在输出目录里递归找 Markdown 产物。

    MinerU 的输出层级随版本变化（output/<name>/<backend>/<name>.md 等），
    因此不硬编码路径，而是递归搜索并做稳定性排序：
    优先顶层目录浅的、文件名不含 _content_list / _model 等派生产物的那个。
    """
    candidates: List[Path] = [
        p for p in out_dir.rglob("*.md")
        if p.is_file() and not _is_derived_name(p.name)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: (len(p.parts), p.name))
    return candidates[0]


def _is_derived_name(name: str) -> bool:
    """排除 MinerU 的派生产物（布局/模型/内容清单等中间 JSON 的同名 md）。"""
    lowered = name.lower()
    return any(tag in lowered for tag in ("_content_list", "_model", "_layout", "_origin"))


def _infer_page_count(out_dir: Path) -> int:
    """从 *_content_list.json 的 page_idx 推断页数；拿不到返回 0。

    MinerU 的 Markdown 本身不带页码概念，但 content_list.json 里每个块都有
    page_idx。页数只用于展示与统计，推断失败不影响正文可用性。
    """
    for path in sorted(out_dir.rglob("*content_list.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, list):
            continue
        indices = [
            item.get("page_idx") for item in data
            if isinstance(item, dict) and isinstance(item.get("page_idx"), int)
        ]
        if indices:
            return max(indices) + 1
    return 0


def _extract_title(markdown_text: str) -> str:
    """取第一个 Markdown 标题作为文档标题；没有标题则取首个非空行。"""
    first_line = ""
    for line in markdown_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()[:100]
        if not first_line:
            first_line = stripped
    return first_line[:100]
