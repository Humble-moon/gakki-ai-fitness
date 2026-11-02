"""管理端观测接口：Prometheus 文本、SLI 汇总、审计查询。

这里直接调用路由函数，鉴权本身由 ``AdminAuthMiddleware`` 的测试覆盖；
把两件事拆开测，失败时能立刻分辨是"数据不对"还是"门没守住"。
"""

import pytest

from app.server import admin_audit_events, admin_metrics_prometheus, admin_sli
from src.audit import record_event


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_prometheus_endpoint_returns_text_format():
    response = await admin_metrics_prometheus()
    body = response.body.decode("utf-8")
    assert "# TYPE http_requests_total counter" in body
    assert "# TYPE llm_calls_total counter" in body
    assert response.media_type.startswith("text/plain")


@pytest.mark.anyio
async def test_sli_endpoint_has_three_numbers():
    payload = await admin_sli()
    assert set(payload) == {"http", "llm"}
    assert "success_rate" in payload["http"]
    assert "p95_seconds" in payload["http"]
    assert "failure_rate" in payload["llm"]


@pytest.mark.anyio
async def test_audit_endpoint_returns_newest_first():
    record_event("auth.login", actor="u:1")
    record_event("review.resolved", actor="u:1", target_id="r-1")

    payload = await admin_audit_events(limit=10)
    assert payload["total"] == 2
    assert [event["action"] for event in payload["events"]] == [
        "review.resolved", "auth.login"]


@pytest.mark.anyio
async def test_audit_endpoint_filters_by_action():
    record_event("auth.login", actor="u:1")
    record_event("data.exported", actor="u:1")

    payload = await admin_audit_events(limit=10, action="auth.login")
    assert payload["total"] == 1
    assert payload["events"][0]["action"] == "auth.login"
