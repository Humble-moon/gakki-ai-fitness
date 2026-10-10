#!/usr/bin/env python3
"""并发压测：静态资源与只读接口的延迟/吞吐。

刻意不压生成类接口——那会真实调用 LLM，既烧钱又会被当日成本上限拦。
静态资源（/static）与健康检查在限流豁免名单里，可以放心打满；
业务接口会计入 per-IP 限流（60 req/min），所以单独统计 429 比例。

用法：python scripts/load_test.py [--concurrency 20] [--requests 200]
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "http://127.0.0.1:8503"


def one(path: str, timeout: int = 15) -> dict:
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(BASE + path, timeout=timeout) as resp:
            resp.read()
            return {"ok": True, "status": resp.status,
                    "ms": (time.perf_counter() - t0) * 1000}
    except urllib.error.HTTPError as e:
        return {"ok": False, "status": e.code, "ms": (time.perf_counter() - t0) * 1000}
    except Exception as e:
        return {"ok": False, "status": None, "ms": (time.perf_counter() - t0) * 1000,
                "err": type(e).__name__}


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(len(s) * p))
    return round(s[idx], 1)


def run_case(label: str, path: str, concurrency: int, total: int) -> dict:
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        t0 = time.perf_counter()
        results = list(pool.map(lambda _: one(path), range(total)))
        wall = time.perf_counter() - t0

    lat = [r["ms"] for r in results]
    codes: dict[str, int] = {}
    for r in results:
        codes[str(r["status"])] = codes.get(str(r["status"]), 0) + 1

    summary = {
        "case": label, "path": path, "concurrency": concurrency, "total": total,
        "wall_s": round(wall, 2),
        "rps": round(total / wall, 1) if wall else None,
        "p50_ms": pct(lat, 0.50), "p95_ms": pct(lat, 0.95), "p99_ms": pct(lat, 0.99),
        "max_ms": round(max(lat), 1) if lat else None,
        "mean_ms": round(statistics.mean(lat), 1) if lat else None,
        "codes": codes,
    }
    print(f"\n【{label}】{path}")
    print(f"  并发 {concurrency} × {total} 请求 | 用时 {summary['wall_s']}s | "
          f"吞吐 {summary['rps']} req/s")
    print(f"  延迟 P50={summary['p50_ms']}ms P95={summary['p95_ms']}ms "
          f"P99={summary['p99_ms']}ms max={summary['max_ms']}ms")
    print(f"  状态码分布: {codes}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--json", default="/tmp/gakki_test/load_test.json")
    args = ap.parse_args()

    cases = [
        ("健康检查（限流豁免）", "/health/live"),
        ("动作图示清单（业务接口，计入限流）", "/api/exercise-images"),
        ("首页 HTML", "/"),
    ]
    # 静态图片：取一个真实存在的文件
    img = next((p for p in (Path("app/static/exercises").rglob("0.jpg"))), None)
    if img:
        rel = "/static/" + str(img.relative_to("app/static"))
        cases.insert(1, ("静态图片（限流豁免）", rel))

    out = [run_case(label, path, args.concurrency, args.requests)
           for label, path in cases]

    Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("\n=== 压测结束 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
