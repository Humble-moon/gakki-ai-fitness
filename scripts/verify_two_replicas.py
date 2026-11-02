#!/usr/bin/env python3
"""双副本限流验证：两个真实进程必须共用同一份每分钟配额。

启动两个 uvicorn 实例（独立端口、共享 Redis），把请求轮流打到两边，
统计放行与 429 的数量。如果限流状态仍在进程内，两个副本会各放行
``RATE_LIMIT`` 次；共享之后总放行数应等于一份配额。

用法：
    python scripts/verify_two_replicas.py

需要本机 Redis 可用；PostgreSQL 等依赖缺失时服务可能起不来，脚本会
打印启动日志并以非零码退出，不会留下后台进程。
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PORTS = (18503, 18504)
LIMIT = 5
REQUESTS = 16


def _wait_ready(port: int, timeout: float = 30.0) -> bool:
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health/live", timeout=1.0) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.4)
    return False


def main() -> int:
    import redis

    from src.config import REDIS_HOST, REDIS_PORT

    try:
        conn = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True,
                           socket_connect_timeout=1.0, socket_timeout=1.0)
        conn.ping()
    except Exception as exc:  # noqa: BLE001
        print(f"Redis 不可用（{REDIS_HOST}:{REDIS_PORT}）：{exc}")
        return 1

    # 清掉上一轮可能残留的窗口，否则限额一开始就是满的。
    for key in conn.scan_iter(match="ratelimit:127.0.0.1*"):
        conn.delete(key)

    env = {
        **os.environ,
        "RATE_LIMIT_PER_MINUTE": str(LIMIT),
        "SHARED_STATE_BACKEND": "redis",
        "LOCAL_OPEN_MODE": "1",
        "AUTO_MIGRATE": "0",
        "LOG_JSON": "0",
        "LOG_LEVEL": "WARNING",
    }
    logs = []
    processes = []
    try:
        for port in PORTS:
            log = tempfile.NamedTemporaryFile(
                prefix=f"replica-{port}-", suffix=".log", delete=False)
            logs.append(log.name)
            processes.append(subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "app.server:app",
                 "--host", "127.0.0.1", "--port", str(port), "--workers", "1"],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            ))

        for port in PORTS:
            if not _wait_ready(port):
                print(f"副本 {port} 在 30 秒内没有就绪。启动日志末尾：")
                print(_tail(logs[PORTS.index(port)]))
                return 1

        import httpx

        allowed = blocked = 0
        with httpx.Client(timeout=5.0) as client:
            for index in range(REQUESTS):
                port = PORTS[index % len(PORTS)]
                response = client.get(f"http://127.0.0.1:{port}/api/auth/session")
                if response.status_code == 429:
                    blocked += 1
                elif response.status_code < 500:
                    allowed += 1
                else:
                    print(f"异常响应 {response.status_code}：{response.text[:200]}")
                    return 1

        print(f"双副本：{REQUESTS} 次请求轮流打到两个进程，"
              f"放行 {allowed} 次、拦截 {blocked} 次（期望放行 {LIMIT} 次）")
        if allowed != LIMIT:
            print("失败：两个副本没有共享同一份配额。")
            return 1
        print("通过：限流状态跨进程共享。")
        return 0
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        for key in conn.scan_iter(match="ratelimit:127.0.0.1*"):
            conn.delete(key)
        for name in logs:
            try:
                os.unlink(name)
            except OSError:
                pass


def _tail(path: str, lines: int = 25) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return "".join(handle.readlines()[-lines:])
    except OSError:
        return "(日志不可读)"


if __name__ == "__main__":
    raise SystemExit(main())
