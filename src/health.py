"""就绪检查。

``/health/ready`` 要回答的是"现在能不能接业务请求"，不是"进程还在不在"。
进程活着但数据库连不上时返回 200，只会让编排器把流量送进一个必然失败的服务。

探测分两步：先用 1 秒的 TCP 连接判断端口是否可达，再发一条最轻量的查询。
先探端口是因为主机不可达时 TCP 重传会挂到几十秒，那会让探针自己变成故障点。
"""

from __future__ import annotations

import socket
from typing import Callable

from src.config import HEALTH_PROBE_DEPENDENCIES

_TCP_PROBE_TIMEOUT = 1.0


def _tcp_reachable(host: str, port: int, timeout: float = _TCP_PROBE_TIMEOUT) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _probe_postgres() -> str:
    from src.config import POSTGRES_HOST, POSTGRES_PORT

    if not _tcp_reachable(POSTGRES_HOST, POSTGRES_PORT):
        return "down"
    try:
        from src.storage.pg import PGClient
        PGClient().fetch_one("SELECT 1")
        return "ok"
    except Exception:  # noqa: BLE001 - 探针只关心结果，不关心异常类型
        return "down"


def _probe_redis() -> str:
    from src.config import REDIS_HOST, REDIS_PORT

    if not _tcp_reachable(REDIS_HOST, REDIS_PORT):
        return "down"
    try:
        from src.storage.redis_client import RedisClient
        RedisClient().get("__health_probe__")
        return "ok"
    except Exception:  # noqa: BLE001
        return "down"


def check_dependencies(probe_dependencies: bool | None = None) -> dict[str, str]:
    """返回各依赖的状态。``probe_dependencies`` 缺省时读配置开关。"""
    if probe_dependencies is None:
        probe_dependencies = HEALTH_PROBE_DEPENDENCIES

    checks = {"provider": "configured"}
    try:
        from src.config import LLM_CONFIGS  # type: ignore

        if not LLM_CONFIGS.get("default"):
            checks["provider"] = "down"
    except Exception:
        checks["provider"] = "down"

    if probe_dependencies:
        checks["postgres"] = _probe_postgres()
        checks["redis"] = _probe_redis()
    return checks


def readiness_checks(
    dependency_checker: Callable[[], dict[str, str]] | None = None,
) -> tuple[dict[str, str], bool]:
    checker = dependency_checker if dependency_checker is not None else check_dependencies
    checks = checker()
    ready = bool(checks) and all(value in {"ok", "configured", "demo"} for value in checks.values())
    return checks, ready


__all__ = ["check_dependencies", "readiness_checks"]
