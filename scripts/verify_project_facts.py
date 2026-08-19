#!/usr/bin/env python3
"""Collect reproducible, read-only project fact counts as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path, warnings: list[str]) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        warnings.append(f"missing: {path.as_posix()}")
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"unreadable: {path.as_posix()} ({exc.__class__.__name__})")
    return None


def _json_length(path: Path, warnings: list[str]) -> int | None:
    value = _load_json(path, warnings)
    if isinstance(value, list):
        return len(value)
    if value is not None:
        warnings.append(f"expected JSON list: {path.as_posix()}")
    return None


def collect_facts(repo_root: Path) -> dict[str, Any]:
    warnings: list[str] = []
    data_counts: dict[str, int | None] = {
        "seed_exercises": _json_length(repo_root / "data/seed_exercises.json", warnings),
        "knowledge_markdown": len(list((repo_root / "data/knowledge").rglob("*.md"))),
    }
    eval_counts: dict[str, int | None] = {
        "golden_main": _json_length(repo_root / "eval/golden_dataset/queries.json", warnings),
        "knowledge_subset": _json_length(repo_root / "eval/golden_dataset/knowledge_queries.json", warnings),
        "conditional_generation": _json_length(repo_root / "eval/e2e_results.json", warnings),
    }
    ragas = _load_json(repo_root / "eval/ragas_results.json", warnings)
    if isinstance(ragas, dict) and isinstance(ragas.get("num_queries"), int):
        eval_counts["ragas"] = ragas["num_queries"]
    else:
        eval_counts["ragas"] = None
        if ragas is not None:
            warnings.append("missing integer num_queries: eval/ragas_results.json")

    test_counts = {
        "pytest_files": len(list((repo_root / "tests").rglob("test_*.py"))),
        "pytest_functions": sum(
            1
            for path in (repo_root / "tests").rglob("test_*.py")
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("def test_") or line.startswith("async def test_")
        ),
    }
    return {
        "data_counts": data_counts,
        "eval_counts": eval_counts,
        "test_counts": test_counts,
        "runtime_contract": {
            "host": "localhost",
            "mode": "single_user_demo",
            "http_port": 8503,
            "sse_endpoints": [
                "/api/generate-plan",
                "/api/analyze-exercise",
                "/api/ask-question",
            ],
        },
        "warnings": warnings
        + [
            "README historical claim of 824 chunks is not independently verified by this inventory",
            "historical evaluation outputs are counted as artifacts, not certified production accuracy",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON (the default)")
    args = parser.parse_args()
    del args
    print(json.dumps(collect_facts(Path(__file__).resolve().parents[1]), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
