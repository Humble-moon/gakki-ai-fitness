"""进程内指标注册表：计数器、直方图、仪表盘，按 Prometheus 文本格式导出。

为什么不引 ``prometheus_client``：这里只需要三个原语，而指标标签的集合
是固定的。自己实现可以把标签收敛在少数几个受控位置，避免某天一条
``path="/api/plans/8f3c..."`` 进来，就把时间序列基数撑到失控。

线程安全：FastAPI 的同步端点跑在线程池里，计数与观测都可能并发。
每个指标自带一把锁，临界区只做字典读写。
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict

#: 延迟分桶（秒）。覆盖本地请求的毫秒级到 SSE 流式响应的分钟级。
DEFAULT_BUCKETS: tuple[float, ...] = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
)


def _escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _key(labels: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(k), str(v)) for k, v in labels.items()))


def _render_labels(key: tuple[tuple[str, str], ...], extra: dict[str, str] | None = None) -> str:
    items = list(key) + list((extra or {}).items())
    if not items:
        return ""
    return "{" + ",".join(f'{k}="{_escape(v)}"' for k, v in items) + "}"


class Counter:
    """只增不减的计数器。"""

    def __init__(self, name: str, help_text: str) -> None:
        self.name = name
        self.help = help_text
        self._values: dict[tuple, float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, labels: dict[str, str] | None = None, value: float = 1.0) -> None:
        key = _key(labels or {})
        with self._lock:
            self._values[key] += value

    def value(self, labels: dict[str, str] | None = None) -> float:
        with self._lock:
            return self._values.get(_key(labels or {}), 0.0)

    def items(self) -> list[tuple[dict[str, str], float]]:
        with self._lock:
            return [(dict(key), value) for key, value in self._values.items()]

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        with self._lock:
            for key, value in sorted(self._values.items()):
                lines.append(f"{self.name}{_render_labels(key)} {value}")
        return lines


class Histogram:
    """按固定分桶累计的直方图，输出兼容 Prometheus 的 ``_bucket/_sum/_count``。"""

    def __init__(self, name: str, help_text: str, buckets: tuple[float, ...] = DEFAULT_BUCKETS) -> None:
        self.name = name
        self.help = help_text
        self.buckets = tuple(sorted(buckets))
        self._counts: dict[tuple, list[int]] = defaultdict(lambda: [0] * len(self.buckets))
        self._sums: dict[tuple, float] = defaultdict(float)
        self._totals: dict[tuple, int] = defaultdict(int)
        self._lock = threading.Lock()

    def observe(self, value: float, labels: dict[str, str] | None = None) -> None:
        key = _key(labels or {})
        with self._lock:
            counts = self._counts[key]
            for index, bound in enumerate(self.buckets):
                if value <= bound:
                    counts[index] += 1
                    break
            self._sums[key] += value
            self._totals[key] += 1

    def snapshot(self, labels: dict[str, str] | None = None) -> dict[str, object]:
        key = _key(labels or {})
        with self._lock:
            counts = list(self._counts.get(key, [0] * len(self.buckets)))
            return {
                "buckets": dict(zip(self.buckets, counts, strict=True)),
                "sum": self._sums.get(key, 0.0),
                "count": self._totals.get(key, 0),
            }

    def items(self) -> list[tuple[dict[str, str], dict[str, object]]]:
        with self._lock:
            keys = set(self._counts) | set(self._totals)
        return [(dict(key), self.snapshot(dict(key))) for key in keys]

    def quantile(self, q: float, labels: dict[str, str] | None = None) -> float:
        """从分桶估算分位数。桶内按线性插值，超出最大桶时返回上界。"""
        snap = self.snapshot(labels)
        total = int(snap["count"])
        if total == 0:
            return 0.0
        target = q * total
        cumulative = 0
        previous_bound = 0.0
        previous_cumulative = 0
        for bound, count in snap["buckets"].items():  # type: ignore[union-attr]
            cumulative += count
            if cumulative >= target:
                span = bound - previous_bound
                within = cumulative - previous_cumulative
                if within <= 0:
                    return float(bound)
                return previous_bound + span * ((target - previous_cumulative) / within)
            previous_bound = float(bound)
            previous_cumulative = cumulative
        return float(self.buckets[-1])

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        with self._lock:
            keys = sorted(set(self._counts) | set(self._totals))
            for key in keys:
                counts = self._counts.get(key, [0] * len(self.buckets))
                cumulative = 0
                for bound, count in zip(self.buckets, counts, strict=True):
                    cumulative += count
                    lines.append(
                        f"{self.name}_bucket{_render_labels(key, {'le': str(bound)})} {cumulative}")
                lines.append(
                    f"{self.name}_bucket{_render_labels(key, {'le': '+Inf'})} "
                    f"{self._totals.get(key, 0)}")
                lines.append(f"{self.name}_sum{_render_labels(key)} {self._sums.get(key, 0.0)}")
                lines.append(f"{self.name}_count{_render_labels(key)} {self._totals.get(key, 0)}")
        return lines


class Gauge:
    """可增可减的瞬时值。"""

    def __init__(self, name: str, help_text: str) -> None:
        self.name = name
        self.help = help_text
        self._values: dict[tuple, float] = defaultdict(float)
        self._lock = threading.Lock()

    def set(self, value: float, labels: dict[str, str] | None = None) -> None:
        with self._lock:
            self._values[_key(labels or {})] = value

    def inc(self, labels: dict[str, str] | None = None, value: float = 1.0) -> None:
        with self._lock:
            self._values[_key(labels or {})] += value

    def dec(self, labels: dict[str, str] | None = None, value: float = 1.0) -> None:
        with self._lock:
            self._values[_key(labels or {})] -= value

    def value(self, labels: dict[str, str] | None = None) -> float:
        with self._lock:
            return self._values.get(_key(labels or {}), 0.0)

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} gauge"]
        with self._lock:
            for key, value in sorted(self._values.items()):
                lines.append(f"{self.name}{_render_labels(key)} {value}")
        return lines


# ---------------------------------------------------------------------------
# 项目指标
# ---------------------------------------------------------------------------

HTTP_REQUESTS = Counter(
    "http_requests_total",
    "HTTP 请求数，按方法、路由模板与状态码分组。",
)
HTTP_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP 请求耗时（含流式响应完成时间）。",
)
LLM_CALLS = Counter(
    "llm_calls_total",
    "LLM 调用次数，outcome 取 success / error / circuit_open。",
)
LLM_DURATION = Histogram(
    "llm_call_duration_seconds",
    "单次 LLM 调用耗时。",
)
REVIEW_PENDING = Gauge(
    "review_pending_total",
    "当前仍待人工处理的审核工件数。",
)

_REGISTRY = (HTTP_REQUESTS, HTTP_DURATION, LLM_CALLS, LLM_DURATION, REVIEW_PENDING)


def render_prometheus() -> str:
    """所有指标合并成一份 Prometheus 文本，末尾补一个换行。"""
    lines: list[str] = []
    for metric in _REGISTRY:
        lines.extend(metric.render())
    return "\n".join(lines) + "\n"


def reset() -> None:
    """清空所有指标。仅用于测试，避免用例之间互相污染。"""
    for metric in _REGISTRY:
        if isinstance(metric, Histogram):
            metric._counts.clear()  # noqa: SLF001 - 指标类的受控重置
            metric._sums.clear()  # noqa: SLF001
            metric._totals.clear()  # noqa: SLF001
        else:
            metric._values.clear()  # noqa: SLF001


# ---------------------------------------------------------------------------
# SLI 快照
# ---------------------------------------------------------------------------

def http_sli() -> dict[str, float | int]:
    """全站请求成功率与 P95 延迟。"""
    total = 0
    failed = 0
    for labels, value in HTTP_REQUESTS.items():
        total += int(value)
        status = labels.get("status", "")
        if status.startswith("5") or status.startswith("4"):
            failed += int(value)
    p95 = _http_p95()
    return {
        "requests": total,
        "failed": failed,
        "success_rate": round((total - failed) / total, 4) if total else 1.0,
        "p95_seconds": round(p95, 4),
    }


def _http_p95() -> float:
    """按路由汇总后再取 P95，避免单个高频路由淹没整体分布。"""
    aggregates: dict[tuple[str, str], dict[str, object]] = {}
    for labels, snap in HTTP_DURATION.items():
        method = labels.get("method", "")
        path = labels.get("path", "")
        bucket = aggregates.setdefault((method, path), {
            "buckets": defaultdict(int),
            "count": 0,
        })
        for bound, count in snap["buckets"].items():  # type: ignore[union-attr]
            bucket["buckets"][bound] += count  # type: ignore[index]
        bucket["count"] += int(snap["count"])  # type: ignore[arg-type]

    total = sum(int(a["count"]) for a in aggregates.values())
    if total == 0:
        return 0.0
    target = 0.95 * total
    cumulative = 0
    previous_bound = 0.0
    previous_cumulative = 0
    merged: dict[float, int] = defaultdict(int)
    for agg in aggregates.values():
        for bound, count in agg["buckets"].items():  # type: ignore[union-attr]
            merged[bound] += count  # type: ignore[index]
    for bound in sorted(merged):
        cumulative += merged[bound]
        if cumulative >= target:
            within = cumulative - previous_cumulative
            if within <= 0:
                return float(bound)
            return previous_bound + (bound - previous_bound) * (
                (target - previous_cumulative) / within)
        previous_bound = bound
        previous_cumulative = cumulative
    return float(max(merged, default=0.0))


def llm_sli() -> dict[str, float | int]:
    """LLM 调用失败率（熔断跳过也计入失败侧，因为它同样意味着本次不可用）。"""
    total = 0
    failed = 0
    for labels, value in LLM_CALLS.items():
        total += int(value)
        if labels.get("outcome") != "success":
            failed += int(value)
    return {
        "calls": total,
        "failed": failed,
        "failure_rate": round(failed / total, 4) if total else 0.0,
    }


def sli_snapshot() -> dict[str, object]:
    return {"http": http_sli(), "llm": llm_sli()}


# ---------------------------------------------------------------------------
# HTTP 中间件
# ---------------------------------------------------------------------------

#: 这些前缀不计入业务 SLI：健康探针由编排器高频调用，静态资源量大且无业务含义，
#: 指标端点本身也会被 Prometheus 反复抓取。
_SKIP_PREFIXES = ("/health", "/static", "/admin/metrics", "/metrics")


def _route_path(scope) -> str:
    """优先用路由模板（``/api/plans/{plan_id}``），退化为两级前缀。

    绝不能直接用原始 path：``/api/plans/<uuid>`` 每条都是一组新标签，
    几个用户就能把时间序列基数撑到内存里出不来。
    """
    route = scope.get("route")
    if route is not None:
        template = getattr(route, "path_format", None) or getattr(route, "path", None)
        if template:
            return str(template)
    path = scope.get("path", "/") or "/"
    parts = [part for part in path.strip("/").split("/") if part]
    if len(parts) <= 2:
        return path
    return "/" + "/".join(parts[:2]) + "/{rest}"


class MetricsMiddleware:
    """记录请求总数与耗时。纯 ASGI 实现，SSE 流不会被缓冲。"""

    def __init__(self, app, skip_prefixes: tuple[str, ...] = _SKIP_PREFIXES) -> None:
        self.app = app
        self.skip_prefixes = skip_prefixes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        # 预检请求由 CORS 直接返回 204，计进来会把成功率抬高，也掩盖真实流量。
        if scope.get("method") == "OPTIONS" or path.startswith(self.skip_prefixes):
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        status = {"code": 500}

        async def send_with_status(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_with_status)
        finally:
            labels = {"method": scope.get("method", "GET"), "path": _route_path(scope)}
            HTTP_REQUESTS.inc({**labels, "status": str(status["code"])})
            HTTP_DURATION.observe(time.perf_counter() - started, labels)


__all__ = [
    "Counter", "Gauge", "Histogram", "MetricsMiddleware",
    "HTTP_REQUESTS", "HTTP_DURATION", "LLM_CALLS", "LLM_DURATION", "REVIEW_PENDING",
    "render_prometheus", "reset", "sli_snapshot", "http_sli", "llm_sli",
]
