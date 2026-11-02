"""指标原语、Prometheus 文本格式与 SLI 汇总的单元测试。"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.observability.metrics import (
    Counter,
    Gauge,
    Histogram,
    HTTP_DURATION,
    HTTP_REQUESTS,
    LLM_CALLS,
    MetricsMiddleware,
    http_sli,
    llm_sli,
    render_prometheus,
    reset,
)


def setup_function() -> None:
    reset()


def test_counter_accumulates_per_label_set():
    counter = Counter("demo_total", "demo")
    counter.inc({"path": "/a"})
    counter.inc({"path": "/a"}, value=2)
    counter.inc({"path": "/b"})

    assert counter.value({"path": "/a"}) == 3
    assert counter.value({"path": "/b"}) == 1
    assert counter.value({"path": "/c"}) == 0


def test_histogram_buckets_are_cumulative_in_render():
    histogram = Histogram("demo_seconds", "demo", buckets=(0.1, 1.0))
    histogram.observe(0.05)
    histogram.observe(0.5)
    histogram.observe(5.0)

    rendered = "\n".join(histogram.render())
    assert 'demo_seconds_bucket{le="0.1"} 1.0' not in rendered  # 渲染为整数计数
    assert 'demo_seconds_bucket{le="0.1"} 1' in rendered
    assert 'demo_seconds_bucket{le="1.0"} 2' in rendered
    assert 'demo_seconds_bucket{le="+Inf"} 3' in rendered
    assert "demo_seconds_count 3" in rendered


def test_histogram_quantile_stays_inside_bucket():
    histogram = Histogram("demo_seconds", "demo", buckets=(0.1, 1.0, 10.0))
    for value in (0.01, 0.02, 0.03, 0.04):
        histogram.observe(value)
    histogram.observe(5.0)

    p95 = histogram.quantile(0.95)
    assert 1.0 <= p95 <= 10.0


def test_gauge_set_and_render():
    gauge = Gauge("demo_gauge", "demo")
    gauge.set(3, {"kind": "pending"})
    gauge.dec({"kind": "pending"})
    rendered = "\n".join(gauge.render())
    assert 'demo_gauge{kind="pending"} 2.0' in rendered


def test_render_prometheus_contains_all_registered_metrics():
    HTTP_REQUESTS.inc({"method": "GET", "path": "/", "status": "200"})
    LLM_CALLS.inc({"model": "m", "outcome": "success"})

    text = render_prometheus()
    assert "# TYPE http_requests_total counter" in text
    assert "# TYPE http_request_duration_seconds histogram" in text
    assert "# TYPE llm_calls_total counter" in text
    assert text.endswith("\n")


def test_http_sli_counts_4xx_and_5xx_as_failed():
    HTTP_REQUESTS.inc({"method": "GET", "path": "/a", "status": "200"})
    HTTP_REQUESTS.inc({"method": "GET", "path": "/a", "status": "404"})
    HTTP_REQUESTS.inc({"method": "POST", "path": "/b", "status": "500"})
    HTTP_REQUESTS.inc({"method": "POST", "path": "/b", "status": "200"})

    sli = http_sli()
    assert sli["requests"] == 4
    assert sli["failed"] == 2
    assert sli["success_rate"] == 0.5


def test_http_sli_reports_full_success_when_no_requests():
    sli = http_sli()
    assert sli["requests"] == 0
    assert sli["success_rate"] == 1.0
    assert sli["p95_seconds"] == 0.0


def test_llm_sli_treats_circuit_open_as_failure():
    LLM_CALLS.inc({"model": "m", "outcome": "success"})
    LLM_CALLS.inc({"model": "m", "outcome": "error"})
    LLM_CALLS.inc({"model": "m", "outcome": "circuit_open"})

    sli = llm_sli()
    assert sli["calls"] == 3
    assert sli["failed"] == 2
    assert sli["failure_rate"] == 0.6667


def _client() -> TestClient:
    app = FastAPI()

    @app.get("/api/items/{item_id}")
    def read_item(item_id: str):
        return {"id": item_id}

    @app.get("/health/live")
    def live():
        return {"ok": True}

    app.add_middleware(MetricsMiddleware)
    return TestClient(app)


def test_middleware_records_route_template_not_raw_path():
    client = _client()
    client.get("/api/items/abc")
    client.get("/api/items/def")

    samples = {(labels["path"], labels["status"]) for labels, _ in HTTP_REQUESTS.items()}
    assert ("/api/items/{item_id}", "200") in samples
    # 原始路径不应出现在标签里，否则每个 id 都是一组新时间序列
    assert not any("abc" in path or "def" in path for path, _ in samples)


def test_middleware_skips_health_probes():
    client = _client()
    client.get("/health/live")
    assert HTTP_REQUESTS.items() == []


def test_middleware_records_failed_requests():
    app = FastAPI()

    @app.get("/api/boom")
    def boom():
        raise RuntimeError("boom")

    app.add_middleware(MetricsMiddleware)
    client = TestClient(app, raise_server_exceptions=False)
    client.get("/api/boom")

    samples = {(labels["path"], labels["status"]) for labels, _ in HTTP_REQUESTS.items()}
    assert ("/api/boom", "500") in samples


def test_middleware_duration_is_observed():
    client = _client()
    client.get("/api/items/x")
    snap = HTTP_DURATION.snapshot({"method": "GET", "path": "/api/items/{item_id}"})
    assert snap["count"] == 1
