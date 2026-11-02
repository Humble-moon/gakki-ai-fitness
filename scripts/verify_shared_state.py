#!/usr/bin/env python3
"""验证限流与熔断状态确实跨实例共享。

用法：
    python scripts/verify_shared_state.py

检查两件事：
1. 两个限流器实例对同一客户端共享同一份配额（6 次请求、上限 3 → 只放行 3 次）。
2. 一个实例触发熔断后，另一个实例读得到同样的打开状态。

脚本会自行清理写入的键；Redis 不可用时以非零码退出并说明原因。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import REDIS_HOST, REDIS_PORT  # noqa: E402
from src.security.shared_state import RedisCircuitState, RedisRateLimiter  # noqa: E402


def main() -> int:
    import redis

    try:
        conn = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, decode_responses=True,
            socket_connect_timeout=1.0, socket_timeout=1.0)
        conn.ping()
    except Exception as exc:  # noqa: BLE001
        print(f"Redis 不可用（{REDIS_HOST}:{REDIS_PORT}）：{exc}")
        return 1

    suffix = str(time.time_ns())
    rate_prefix = f"ratelimit:verify-{suffix}"
    circuit_prefix = f"circuit:verify-{suffix}"
    rate_a = RedisRateLimiter(conn=conn, prefix=rate_prefix)
    rate_b = RedisRateLimiter(conn=conn, prefix=rate_prefix)
    circuit_a = RedisCircuitState(conn=conn, prefix=circuit_prefix)
    circuit_b = RedisCircuitState(conn=conn, prefix=circuit_prefix)

    failures: list[str] = []
    try:
        # ---- 1. 共享配额 ----
        results = [
            (rate_a if index % 2 == 0 else rate_b).allow("client", 3, 60)
            for index in range(6)
        ]
        allowed = sum(1 for ok, _ in results if ok)
        print(f"限流：6 次请求分散到两个实例，允许 {allowed} 次（期望 3）")
        if allowed != 3:
            failures.append(f"共享配额不正确：allowed={allowed}")

        # ---- 2. 共享熔断 ----
        for _ in range(3):
            circuit_a.record_failure("default", threshold=3, cooldown=30)
        shared_open = circuit_b.is_open("default")
        print(f"熔断：实例 A 连续失败 3 次，实例 B 读到 is_open={shared_open}（期望 True）")
        if shared_open is not True:
            failures.append(f"共享熔断不正确：is_open={shared_open}")

        circuit_b.record_success("default")
        recovered = circuit_a.is_open("default")
        print(f"熔断：实例 B 记录成功，实例 A 读到 is_open={recovered}（期望 False）")
        if recovered is not False:
            failures.append(f"熔断恢复不正确：is_open={recovered}")
    finally:
        for key in conn.scan_iter(match=f"{rate_prefix}:*"):
            conn.delete(key)
        for key in conn.scan_iter(match=f"{circuit_prefix}:*"):
            conn.delete(key)

    if failures:
        print("\n失败：")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("\n通过：限流与熔断状态在实例间共享。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
