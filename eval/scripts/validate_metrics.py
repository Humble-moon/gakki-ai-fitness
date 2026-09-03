#!/usr/bin/env python3
"""Validate evaluation metadata and bounded metrics without touching result files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED = {"id", "kind", "dataset_size", "source_path", "status", "comparability", "known_limits"}
KINDS = {
    "retrieval",
    "generation_conditioned",
    "safety",
    "ablation_compare",
    "ragas_subset",
    "load_local",
    "routing_fixture",
}
BOUNDED_NAMES = ("precision", "recall", "ndcg", "mrr")


def validate(manifest: dict) -> list[str]:
    errors: list[str] = []
    experiments = manifest.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        return ["manifest.experiments must be a non-empty list"]
    for index, item in enumerate(experiments):
        prefix = f"experiments[{index}]"
        missing = REQUIRED - item.keys()
        if missing:
            errors.append(f"{prefix} missing: {', '.join(sorted(missing))}")
        if item.get("kind") not in KINDS:
            errors.append(f"{prefix}.kind is invalid")
        if not isinstance(item.get("dataset_size"), int) or item["dataset_size"] <= 0:
            errors.append(f"{prefix}.dataset_size must be a positive integer")
        metrics = item.get("metrics", {})
        if not isinstance(metrics, dict):
            errors.append(f"{prefix}.metrics must be an object")
            continue
        for name, value in metrics.items():
            if any(token in name.lower() for token in BOUNDED_NAMES):
                if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                    if "recall" in name.lower() and "metric_definition" in item:
                        continue
                    errors.append(f"{prefix}.metrics.{name} must be between 0 and 1")
        if any("recall" in name.lower() and isinstance(value, (int, float)) and value > 1 for name, value in metrics.items()):
            if "metric_definition" not in item:
                errors.append(f"{prefix} requires metric_definition for unbounded recall")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"manifest error: {exc}", file=sys.stderr)
        return 2
    errors = validate(manifest)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"validated {len(manifest['experiments'])} evaluation entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
