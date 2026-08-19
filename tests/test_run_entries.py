from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_entry(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args], cwd=ROOT, capture_output=True, text=True
    )


def test_run_demo_has_safe_modes_and_check():
    result = run_entry("scripts/run_demo.py", "--help")
    assert result.returncode == 0
    assert "--mode" in result.stdout
    assert "--check" in result.stdout


def test_run_demo_check_does_not_print_secrets():
    result = run_entry("scripts/run_demo.py", "--check")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["mode"] in {"demo", "full"}
    assert "api_key" not in result.stdout.lower()
    assert ".env" not in result.stdout


def test_run_e2e_supports_json_summary():
    result = run_entry("scripts/run_e2e.py", "--json")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "passed"
    assert payload["terminal_event"] == "done"
    assert payload["event_counts"]["done"] >= 3
    assert payload["failure_reason"] is None


def test_makefile_has_stage_three_entries():
    makefile = (ROOT / "Makefile").read_text()
    assert "run_demo.py" in makefile
    assert "run_e2e.py" in makefile
