import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_fact_script():
    completed = subprocess.run(
        [sys.executable, "scripts/verify_project_facts.py", "--json"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_fact_script_reports_current_dataset_counts():
    result = run_fact_script()
    assert result["data_counts"]["seed_exercises"] == 338
    assert result["data_counts"]["knowledge_markdown"] == 162
    assert result["eval_counts"]["golden_main"] == 206
    assert result["eval_counts"]["knowledge_subset"] == 54
    assert result["eval_counts"]["conditional_generation"] == 10
    assert result["eval_counts"]["ragas"] == 68


def test_fact_script_has_reviewable_contract_sections():
    result = run_fact_script()
    assert set(("data_counts", "eval_counts", "test_counts", "runtime_contract", "warnings")) <= result.keys()
    assert result["runtime_contract"]["host"] == "localhost"
    assert result["runtime_contract"]["mode"] == "single_user_demo"
    assert isinstance(result["warnings"], list)


def test_readme_declares_localhost_single_user_boundary():
    readme = (ROOT / "README.md").read_text()
    assert "localhost 单用户" in readme
    assert "医疗级" not in readme
    assert "生产 SLA" not in readme
    assert "README 历史声明，当前未独立复核" in readme
    assert "docs/project-fact-baseline.md" in readme


def test_fact_baseline_contains_evidence_and_status_columns():
    baseline = (ROOT / "docs/project-fact-baseline.md").read_text()
    for marker in ("事实", "证据路径", "状态", "统计日期", "对外话术"):
        assert marker in baseline
    for marker in ("data_counts", "eval_counts", "test_counts", "runtime_contract", "warnings"):
        assert marker in baseline


def test_fact_script_does_not_read_environment_files():
    script = (ROOT / "scripts/verify_project_facts.py").read_text()
    assert ".env" not in script
    assert "api_key" not in script.lower()
