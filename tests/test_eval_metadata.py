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
    kinds = {
        "retrieval",
        "generation_conditioned",
        "safety",
        "ragas_subset",
        "load_local",
        "routing_fixture",
    }
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


def test_route_comparison_result_contains_comparable_fixture_metadata(tmp_path):
    dataset = tmp_path / "queries.json"
    output = tmp_path / "comparison.json"
    dataset.write_text(json.dumps([
        {"id": "exact", "query": "杠铃深蹲怎么做"},
        {"id": "injury", "query": "深蹲时膝盖疼怎么办"},
        {"id": "fallback", "query": "你好"},
    ]), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "eval/scripts/compare_retrieval_routes.py"),
            "--dataset", str(dataset),
            "--output", str(output),
        ],
        capture_output=True, text=True, cwd=ROOT,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["experiment_type"] == "deterministic-routing-fixture"
    assert payload["is_retrieval_quality_experiment"] is False
    assert payload["dataset_size"] == 3
    assert payload["baseline"]["policy"] == "legacy-route-none"
    assert payload["conditional"]["policy"] == "deterministic-classifier"
    assert payload["metrics"]["fixture_route_coverage"] == 1.0
    assert payload["metrics"]["fixture_changed_route_rate"] == 2 / 3
    assert payload["latency_ms"] is None
    assert payload["provider_calls"] is None
    assert payload["route_distribution"] == {
        "exact_action": 1,
        "fallback": 1,
        "injury_sensitive": 1,
    }
    assert payload["limitations"]


def test_manifest_registers_deterministic_route_fixture():
    manifest = json.loads((ROOT / "eval/evaluation_manifest.json").read_text())
    fixture = next(
        item for item in manifest["experiments"]
        if item["id"] == "conditional_route_fixture"
    )
    assert fixture["kind"] == "routing_fixture"
    assert fixture["source_path"] == "eval/results/conditional-route-fixture.json"
    assert fixture["status"] == "deterministic-fixture"
    assert fixture["comparability"] == "not-a-retrieval-quality-experiment"
    assert fixture["metrics"]["latency_ms"] is None
    assert fixture["metrics"]["provider_calls"] is None


def test_route_result_readme_states_fixture_boundaries():
    text = (ROOT / "eval/results/README.md").read_text(encoding="utf-8")
    assert "非检索质量实验" in text
    assert "latency_ms" in text
    assert "provider_calls" in text
    assert "null" in text
