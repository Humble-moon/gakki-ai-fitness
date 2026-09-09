"""Optional MinerU parsing backend: availability probe, fallback, artifact discovery.

The contract that matters here is degradation. MinerU is an enhancement for
scanned / borderless / cross-page tables; it must never take down file upload.
Every failure mode — missing binary, timeout, non-zero exit, empty output,
no markdown artifact — has to land back on pdfplumber.
"""

import json
import subprocess
from pathlib import Path

import pytest

import src.parsers as parsers
import src.parsers.mineru_parser as mineru
from src.parsers.pdf_parser import PDFParseResult


PDF_BYTES = b"%PDF-1.4 fake bytes for routing tests"

MARKDOWN = (
    "# 体检报告\n\n"
    "## 基本指标\n\n"
    "| 项目 | 结果 | 参考范围 |\n|---|---|---|\n| 体脂率 | 22% | 15-20% |\n\n"
    "结论：建议控制碳水摄入。\n"
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def write_artifacts(out_dir: Path, md_name="upload.md", md_text=MARKDOWN,
                    page_indices=(0, 1, 2), depth=("upload", "pipeline"),
                    also_derived=True):
    """Materialise the artifact tree MinerU leaves behind."""
    target = out_dir.joinpath(*depth) if depth else out_dir
    target.mkdir(parents=True, exist_ok=True)
    (target / md_name).write_text(md_text, encoding="utf-8")
    if also_derived:
        # derived products must never be mistaken for the main markdown
        (target / "upload_content_list.md").write_text("noise", encoding="utf-8")
        (target / "upload_model.md").write_text("noise", encoding="utf-8")
    if page_indices is not None:
        (target / "upload_content_list.json").write_text(
            json.dumps([{"page_idx": i, "text": "x"} for i in page_indices]),
            encoding="utf-8",
        )
    return target


def fake_cli(md_text=MARKDOWN, returncode=0, stderr="", page_indices=(0, 1, 2),
             depth=("upload", "pipeline"), produce=True, seen=None):
    """Stand in for `mineru -p <in> -o <out> -b <backend>`."""
    def run(cmd, **kwargs):
        assert cmd[0] == mineru.MINERU_CLI
        assert "-p" in cmd and "-o" in cmd and "-b" in cmd
        if seen is not None:
            seen.append({"cmd": cmd, "timeout": kwargs.get("timeout"),
                         "env": kwargs.get("env"), "shell": kwargs.get("shell")})
        if produce:
            write_artifacts(Path(cmd[cmd.index("-o") + 1]), md_text=md_text,
                            page_indices=page_indices, depth=depth)
        return subprocess.CompletedProcess(cmd, returncode, stdout="", stderr=stderr)
    return run


@pytest.fixture
def mineru_on(monkeypatch):
    """Switch the project to the MinerU backend with the binary present."""
    monkeypatch.setattr(parsers, "DOC_PARSER_BACKEND", "mineru")
    monkeypatch.setattr(mineru, "MINERU_CLI", "mineru")
    monkeypatch.setattr(mineru, "MINERU_BACKEND", "pipeline")
    monkeypatch.setattr(mineru, "MINERU_TIMEOUT", 300)
    monkeypatch.setattr(mineru, "MINERU_MODEL_SOURCE", "modelscope")
    monkeypatch.setattr(mineru.shutil, "which", lambda name: "/usr/local/bin/mineru")
    return monkeypatch


def stub_pdfplumber(monkeypatch, text="pdfplumber 输出", has_text=True):
    """Record that the fallback path was taken, and what it returned."""
    calls = []

    def fake_parse(file_bytes):
        calls.append(file_bytes)
        page = type("P", (), {})()
        page.page_number = 1
        page.text = text
        page.tables_md = []
        page.font_sizes = [10.0]
        return PDFParseResult(pages=[page], total_chars=len(text),
                              has_text=has_text, error=None)

    monkeypatch.setattr("src.parsers.pdf_parser.parse_pdf", fake_parse)
    return calls


# --------------------------------------------------------------------------
# defaults
# --------------------------------------------------------------------------
def test_default_backend_is_pdfplumber():
    assert parsers.DOC_PARSER_BACKEND == "pdfplumber"
    assert mineru.MINERU_BACKEND == "pipeline"


def test_default_config_never_touches_mineru(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("pdfplumber backend must not invoke MinerU")

    monkeypatch.setattr(parsers, "DOC_PARSER_BACKEND", "pdfplumber")
    monkeypatch.setattr(mineru.subprocess, "run", boom)
    calls = stub_pdfplumber(monkeypatch)

    doc = parsers.parse_file(PDF_BYTES, "报告.pdf")
    assert doc.file_type == "pdf"
    assert len(calls) == 1


def test_is_available_false_without_binary(monkeypatch):
    monkeypatch.setattr(mineru.shutil, "which", lambda name: None)
    assert mineru.is_available() is False
    assert mineru.parse_pdf(PDF_BYTES) is None


def test_is_available_false_when_cli_unconfigured(monkeypatch):
    monkeypatch.setattr(mineru, "MINERU_CLI", "")
    assert mineru.is_available() is False


# --------------------------------------------------------------------------
# happy path
# --------------------------------------------------------------------------
def test_mineru_result_is_used_when_it_produces_markdown(mineru_on):
    seen = []
    mineru_on.setattr(mineru.subprocess, "run", fake_cli(seen=seen))
    fallback = stub_pdfplumber(mineru_on)

    doc = parsers.parse_file(PDF_BYTES, "体检报告.pdf")

    assert doc.full_text == MARKDOWN.strip()
    assert doc.title == "体检报告"
    assert doc.page_count == 3
    assert doc.has_text is True
    assert doc.error is None
    assert fallback == []                       # never degraded
    assert seen[0]["shell"] is not True          # no shell injection surface
    assert seen[0]["timeout"] == 300
    assert seen[0]["env"]["MINERU_MODEL_SOURCE"] == "modelscope"
    assert seen[0]["cmd"][seen[0]["cmd"].index("-b") + 1] == "pipeline"


def test_staged_filename_is_fixed_not_user_supplied(mineru_on):
    """User filenames never reach the filesystem path — no traversal surface."""
    seen = []
    mineru_on.setattr(mineru.subprocess, "run", fake_cli(seen=seen))
    stub_pdfplumber(mineru_on)

    parsers.parse_file(PDF_BYTES, "../../etc/passwd.pdf")
    in_path = seen[0]["cmd"][seen[0]["cmd"].index("-p") + 1]
    assert Path(in_path).name == mineru.STAGED_FILENAME
    assert ".." not in in_path


def test_temporary_workdir_is_cleaned_up(mineru_on):
    created = []
    real_run = fake_cli()

    def run(cmd, **kwargs):
        created.append(str(Path(cmd[cmd.index("-p") + 1]).parent))
        return real_run(cmd, **kwargs)

    mineru_on.setattr(mineru.subprocess, "run", run)
    stub_pdfplumber(mineru_on)

    parsers.parse_file(PDF_BYTES, "报告.pdf")
    assert created and not Path(created[0]).exists()


# --------------------------------------------------------------------------
# degradation: every failure mode lands on pdfplumber
# --------------------------------------------------------------------------
def test_missing_binary_falls_back(mineru_on):
    mineru_on.setattr(mineru.shutil, "which", lambda name: None)
    fallback = stub_pdfplumber(mineru_on)

    doc = parsers.parse_file(PDF_BYTES, "报告.pdf")
    assert len(fallback) == 1
    assert "pdfplumber 输出" in doc.full_text   # pdfplumber adds page headers


def test_timeout_falls_back(mineru_on):
    def raise_timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))

    mineru_on.setattr(mineru.subprocess, "run", raise_timeout)
    fallback = stub_pdfplumber(mineru_on)

    doc = parsers.parse_file(PDF_BYTES, "报告.pdf")
    assert len(fallback) == 1
    assert doc.file_type == "pdf"


def test_oserror_on_launch_falls_back(mineru_on):
    def raise_oserror(cmd, **kwargs):
        raise OSError("exec format error")

    mineru_on.setattr(mineru.subprocess, "run", raise_oserror)
    fallback = stub_pdfplumber(mineru_on)

    assert len(parsers.parse_file(PDF_BYTES, "报告.pdf").full_text) > 0
    assert len(fallback) == 1


def test_no_markdown_artifact_falls_back(mineru_on):
    mineru_on.setattr(mineru.subprocess, "run", fake_cli(produce=False))
    fallback = stub_pdfplumber(mineru_on)

    parsers.parse_file(PDF_BYTES, "报告.pdf")
    assert len(fallback) == 1


def test_empty_markdown_falls_back(mineru_on):
    mineru_on.setattr(mineru.subprocess, "run", fake_cli(md_text="   \n  "))
    fallback = stub_pdfplumber(mineru_on)

    parsers.parse_file(PDF_BYTES, "报告.pdf")
    assert len(fallback) == 1


def test_nonzero_exit_without_output_falls_back(mineru_on):
    mineru_on.setattr(mineru.subprocess, "run",
                      fake_cli(returncode=1, stderr="CUDA out of memory", produce=False))
    fallback = stub_pdfplumber(mineru_on)

    parsers.parse_file(PDF_BYTES, "报告.pdf")
    assert len(fallback) == 1


def test_nonzero_exit_with_partial_output_is_still_used(mineru_on):
    """A per-page failure should not discard the pages that did parse."""
    mineru_on.setattr(mineru.subprocess, "run",
                      fake_cli(returncode=1, stderr="page 7 failed"))
    fallback = stub_pdfplumber(mineru_on)

    doc = parsers.parse_file(PDF_BYTES, "报告.pdf")
    assert doc.full_text == MARKDOWN.strip()
    assert fallback == []


def test_empty_upload_is_rejected_without_spawning_a_process(mineru_on):
    def boom(*a, **k):
        raise AssertionError("empty bytes must not reach the CLI")

    mineru_on.setattr(mineru.subprocess, "run", boom)
    fallback = stub_pdfplumber(mineru_on)

    parsers.parse_file(b"", "空.pdf")
    assert len(fallback) == 1


# --------------------------------------------------------------------------
# artifact discovery details
# --------------------------------------------------------------------------
def test_locate_markdown_ignores_derived_products(tmp_path):
    write_artifacts(tmp_path, depth=("upload", "pipeline"))
    found = mineru._locate_markdown(tmp_path)
    assert found is not None
    assert found.name == "upload.md"


def test_locate_markdown_prefers_shallowest_path(tmp_path):
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / "deep.md").write_text("deep", encoding="utf-8")
    (tmp_path / "top.md").write_text("top", encoding="utf-8")

    assert mineru._locate_markdown(tmp_path).name == "top.md"


def test_locate_markdown_returns_none_when_absent(tmp_path):
    (tmp_path / "only.json").write_text("{}", encoding="utf-8")
    assert mineru._locate_markdown(tmp_path) is None


def test_infer_page_count_from_content_list(tmp_path):
    write_artifacts(tmp_path, page_indices=(0, 1, 2, 3))
    assert mineru._infer_page_count(tmp_path) == 4


def test_infer_page_count_zero_when_unavailable(tmp_path):
    write_artifacts(tmp_path, page_indices=None)
    assert mineru._infer_page_count(tmp_path) == 0


def test_infer_page_count_survives_malformed_json(tmp_path):
    target = tmp_path / "upload"
    target.mkdir()
    (target / "upload_content_list.json").write_text("{not json", encoding="utf-8")
    assert mineru._infer_page_count(tmp_path) == 0


@pytest.mark.parametrize("text,expected", [
    ("# 体检报告\n\n正文", "体检报告"),
    ("### 膝关节评估\n正文", "膝关节评估"),
    ("没有标题的第一行\n第二行", "没有标题的第一行"),
    ("\n\n# 前置空行\n正文", "前置空行"),
])
def test_extract_title(text, expected):
    assert mineru._extract_title(text) == expected


def test_extract_title_truncates_long_lines():
    assert len(mineru._extract_title("字" * 300)) == 100
