#!/usr/bin/env python3
"""Offline SSE business-flow verification; never treats EOF as success."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx

PAYLOADS = {
    "training_plan": ("/api/generate-plan", {"session_id": "offline", "query": "增肌"}),
    "exercise_analysis": ("/api/analyze-exercise", {"exercise_name": "深蹲"}),
    "knowledge_question": ("/api/ask-question", {"question": "怎么练？"}),
}


def decode_sse(body: str) -> list[dict]:
    events = []
    for frame in body.split("\n\n"):
        for line in frame.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


async def run() -> dict[str, object]:
    from app import server

    class OfflineOrchestrator:
        def generate_plan_stream(self, *args):
            yield "stage", "offline"
            yield "done", {"success": True}

        def analyze_exercise_stream(self, *args):
            yield "stage", "offline"
            yield "done", {"success": True}

        def answer_question_stream(self, *args):
            yield "stage", "offline"
            yield "done", {"success": True}

    server.orch = OfflineOrchestrator()
    counts: Counter[str] = Counter()
    flow_terminals: dict[str, str] = {}
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://offline") as client:
        for name, (path, payload) in PAYLOADS.items():
            response = await client.post(path, json=payload)
            events = decode_sse(response.text)
            counts.update(event["event"] for event in events)
            terminals = [event["event"] for event in events if event["event"] in {"done", "error", "cancelled"}]
            if len(terminals) != 1:
                return {"status": "failed", "terminal_event": terminals[-1] if terminals else None, "event_counts": dict(counts), "failure_reason": f"{name}: missing explicit terminal event"}
            flow_terminals[name] = terminals[0]
    if any(event != "done" for event in flow_terminals.values()):
        return {"status": "failed", "terminal_event": None, "event_counts": dict(counts), "failure_reason": "offline flow did not complete"}
    return {"status": "passed", "terminal_event": "done", "event_counts": dict(counts), "failure_reason": None}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    args = parser.parse_args()
    summary = asyncio.run(run())
    print(json.dumps(summary, ensure_ascii=False) if args.json else summary)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
