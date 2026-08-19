#!/usr/bin/env python3
"""Compare legacy and conditional route selection using a local fixture only.

This script does not execute retrieval, providers, or latency measurements. It
records how deterministic query classification would differ from route=None.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.query_routing import QueryRoute, classify_query


def _route_name(route: QueryRoute) -> str:
    return route.name.lower()


def build_comparison(dataset_path: Path) -> dict:
    raw = dataset_path.read_bytes()
    records = json.loads(raw)
    if not isinstance(records, list) or not records:
        raise ValueError("dataset must be a non-empty JSON array")

    comparisons = []
    distribution: Counter[str] = Counter()
    changed = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("query"), str):
            raise ValueError(f"dataset item {index} must contain a string query")
        route = _route_name(classify_query(record["query"]))
        distribution[route] += 1
        route_argument = None if route == "fallback" else route
        if route_argument is not None:
            changed += 1
        comparisons.append({
            "id": record.get("id", str(index)),
            "query": record["query"],
            "baseline_route": None,
            "conditional_route": route,
            "conditional_route_argument": route_argument,
        })

    size = len(records)
    return {
        "schema_version": "1",
        "experiment_type": "deterministic-routing-fixture",
        "is_retrieval_quality_experiment": False,
        "dataset": str(dataset_path),
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "dataset_size": size,
        "baseline": {
            "policy": "legacy-route-none",
            "description": "All queries preserve the existing route=None call path.",
        },
        "conditional": {
            "policy": "deterministic-classifier",
            "description": "Non-fallback classifications pass an explicit route; fallback preserves route=None.",
        },
        "metrics": {
            "fixture_route_coverage": len(comparisons) / size,
            "fixture_changed_route_rate": changed / size,
        },
        "latency_ms": None,
        "provider_calls": None,
        "route_distribution": dict(sorted(distribution.items())),
        "limitations": [
            "This is not a retrieval quality experiment and executes no retriever.",
            "No relevance, precision, recall, ranking, or answer-quality claim can be made.",
            "Latency and provider calls are not measured and are therefore null.",
            "Metrics describe deterministic fixture routing only.",
        ],
        "queries": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.output.exists():
        print(f"refusing to overwrite existing result: {args.output}", file=sys.stderr)
        return 2
    try:
        result = build_comparison(args.dataset)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"comparison fixture error: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote deterministic routing fixture for {result['dataset_size']} queries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
