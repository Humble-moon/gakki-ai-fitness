#!/usr/bin/env python3
"""Safe localhost demo entry point for the interview build."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def configuration(mode: str) -> dict[str, object]:
    # Importing config loads values, but this report deliberately exposes only presence.
    from src.config import LLM_CONFIGS

    return {
        "mode": mode,
        "provider_configured": bool(LLM_CONFIGS.get("default")),
        "external_services_started": False,
        "keys_required": mode == "full",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the localhost AI Fitness demo")
    parser.add_argument("--mode", choices=("demo", "full"), default="demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8503)
    parser.add_argument("--check", action="store_true", help="check configuration only")
    args = parser.parse_args(argv)

    if args.check:
        print(json.dumps(configuration(args.mode), ensure_ascii=False))
        return 0

    if args.mode == "demo":
        os.environ.setdefault("FITNESS_DEMO_MODE", "1")
        print(f"Starting localhost demo at http://{args.host}:{args.port}", flush=True)
    else:
        report = configuration("full")
        if not report["provider_configured"]:
            print("Full mode requires configured provider credentials; nothing was started.", file=sys.stderr)
            return 2
        print("Full mode configuration is valid; external services are not started automatically.", flush=True)
        return 0

    import uvicorn
    uvicorn.run("app.server:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
