"""
=============================================================================
压测脚本 — serving 层容量评估（QPS / P50 / P95 / P99 / TTFT）
=============================================================================
背景：
  评测体系此前只有离线质量指标（P@K、LLM-as-Judge），没有任何 serving 维度
  数据。本脚本补上"上线后怎么办"的弹药：应用层吞吐、LLM 链路延迟分布、
  流式首字延迟（TTFT）、并发下的错误率。

测什么：
  1. GET /admin/metrics    — 无 LLM 的轻量接口 → 应用层本身的吞吐上限
  2. POST /api/ask-question — 全链路（检索+LLM流式）→ 用户真实体验的延迟
     流式接口关注两个数字：TTFT（首字延迟，决定"体感快慢"）和总时长

用法：
    python eval/load_test.py                    # 全部场景
    python eval/load_test.py --light-only      # 只测轻量接口（不花 LLM 钱）
"""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8503"
OUTPUT = Path(__file__).parent / "load_test_results.json"

QA_PAYLOAD = {
    "height": 178, "weight": 75, "training_years": 1.0,
    "goal": "增肌", "available_equipment": ["哑铃"], "days_per_week": 4,
    "injuries": [], "question": "深蹲和腿举有什么区别，练腿该选哪个",
    "session_id": None,
}


def pct(vals: list, p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(round(p / 100 * len(s) + 0.5)) - 1))
    return s[idx]


def summarize(name: str, latencies: list, errors: int, wall: float, extra: dict = None) -> dict:
    n = len(latencies)
    result = {
        "scenario": name,
        "requests_ok": n,
        "errors": errors,
        "wall_time_sec": round(wall, 2),
        "throughput_rps": round(n / wall, 2) if wall > 0 else 0,
        "latency_ms": {
            "mean": round(statistics.mean(latencies) * 1000, 1) if latencies else 0,
            "p50": round(pct(latencies, 50) * 1000, 1),
            "p95": round(pct(latencies, 95) * 1000, 1),
            "p99": round(pct(latencies, 99) * 1000, 1),
            "max": round(max(latencies) * 1000, 1) if latencies else 0,
        },
    }
    if extra:
        result.update(extra)
    return result


async def bench_light(concurrency: int, total: int) -> dict:
    """轻量接口压测：应用层吞吐（无 LLM）。"""
    latencies, errors = [], 0
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=30, trust_env=False) as client:  # trust_env=False: 绕过系统残留代理
        async def one():
            nonlocal errors
            async with sem:
                t0 = time.perf_counter()
                try:
                    r = await client.get(f"{BASE}/admin/metrics")
                    r.raise_for_status()
                    latencies.append(time.perf_counter() - t0)
                except Exception as e:
                    errors += 1
                    if errors <= 3:
                        print(f"    light request failed: {type(e).__name__}: {e}")

        t0 = time.perf_counter()
        await asyncio.gather(*[one() for _ in range(total)])
        wall = time.perf_counter() - t0

    return summarize(f"light:/admin/metrics c={concurrency}", latencies, errors, wall)


async def bench_qa(concurrency: int, total: int) -> dict:
    """QA 流式接口压测：全链路延迟 + TTFT。"""
    totals, ttfts, errors = [], [], 0
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=180, trust_env=False) as client:  # trust_env=False: 绕过系统残留代理
        async def one():
            nonlocal errors
            async with sem:
                t0 = time.perf_counter()
                ttft = None
                try:
                    async with client.stream(
                        "POST", f"{BASE}/api/ask-question", json=QA_PAYLOAD
                    ) as r:
                        r.raise_for_status()
                        async for _chunk in r.aiter_bytes():
                            if ttft is None:
                                ttft = time.perf_counter() - t0
                    totals.append(time.perf_counter() - t0)
                    if ttft is not None:
                        ttfts.append(ttft)
                except Exception as e:
                    errors += 1
                    print(f"    QA request failed: {type(e).__name__}: {e}")

        t0 = time.perf_counter()
        await asyncio.gather(*[one() for _ in range(total)])
        wall = time.perf_counter() - t0

    extra = {
        "ttft_ms": {
            "mean": round(statistics.mean(ttfts) * 1000, 1) if ttfts else 0,
            "p50": round(pct(ttfts, 50) * 1000, 1),
            "p95": round(pct(ttfts, 95) * 1000, 1),
        }
    }
    return summarize(f"qa:/api/ask-question c={concurrency}", totals, errors, wall, extra)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--light-only", action="store_true")
    args = parser.parse_args()

    results = []

    # ---- 场景 1：轻量接口，应用层吞吐 ----
    for c, n in [(1, 50), (10, 200), (50, 500)]:
        print(f"[light] concurrency={c} total={n} ...")
        r = await bench_light(c, n)
        results.append(r)
        print(f"  -> {r['throughput_rps']} rps, P95={r['latency_ms']['p95']}ms, errors={r['errors']}")

    # ---- 场景 2：QA 全链路（流式），LLM-bound ----
    if not args.light_only:
        for c, n in [(1, 5), (5, 10)]:
            print(f"[qa] concurrency={c} total={n} ...")
            r = await bench_qa(c, n)
            results.append(r)
            print(f"  -> TTFT P50={r['ttft_ms']['p50']}ms, total P95={r['latency_ms']['p95']}ms, "
                  f"errors={r['errors']}")

    OUTPUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults saved to {OUTPUT}")


if __name__ == "__main__":
    asyncio.run(main())
