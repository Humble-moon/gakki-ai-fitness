import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_manifest_separates_main_and_knowledge_retrieval_sets():
    manifest = json.loads((ROOT / "eval/evaluation_manifest.json").read_text())
    ids = {item["id"] for item in manifest["experiments"]}
    assert "retrieval_main_206" in ids
    assert "retrieval_knowledge_subset_54" in ids


def test_manifest_items_have_required_metadata_and_known_kinds():
    manifest = json.loads((ROOT / "eval/evaluation_manifest.json").read_text())
    required = {"id", "kind", "dataset_size", "source_path", "status", "comparability", "known_limits"}
    kinds = {"retrieval", "generation_conditioned", "safety", "ragas_subset", "load_local"}
    assert manifest["experiments"]
    for item in manifest["experiments"]:
        assert required <= item.keys()
        assert item["kind"] in kinds
        assert isinstance(item["dataset_size"], int) and item["dataset_size"] > 0


def test_metric_validator_accepts_repository_manifest():
    result = subprocess.run(
        [sys.executable, str(ROOT / "eval/scripts/validate_metrics.py"),
         "--manifest", str(ROOT / "eval/evaluation_manifest.json")],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr


def test_metric_validator_rejects_unbounded_recall(tmp_path):
    manifest = {"experiments": [{
        "id": "bad", "kind": "retrieval", "dataset_size": 1,
        "source_path": "x.json", "status": "historical",
        "comparability": "not-comparable", "known_limits": [],
        "metrics": {"recall@10": 1.1},
    }]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    result = subprocess.run(
        [sys.executable, str(ROOT / "eval/scripts/validate_metrics.py"), "--manifest", str(path)],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode != 0


def test_conditioned_generation_is_not_described_as_full_e2e():
    report = (ROOT / "eval/E2E_EVAL_REPORT.md").read_text()
    assert "条件上下文" in report
    assert "不是完整" in report or "不等同" in report


def test_readme_describes_default_pytest_scope_and_eval_entry():
    readme = (ROOT / "README.md").read_text()
    assert "integration/live" in readme
    assert "make eval" in readme


def test_makefile_exposes_reproducible_targets():
    text = (ROOT / "Makefile").read_text()
    for target in ("facts:", "test:", "e2e:", "eval:", "demo-help:"):
        assert target in text
