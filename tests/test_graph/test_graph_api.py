"""API-level tests for the v2 (LangGraph) endpoints.

Follows tests/e2e/test_sse_business_flows.py: drive the real FastAPI app through
``httpx.ASGITransport`` and monkeypatch ``server.graph_runtime`` with an offline
fake runtime (MemorySaver + faked agents). No network, no live LLM.
"""

import httpx
import pytest

from scripts.run_e2e import decode_sse
from tests.test_graph.test_graph_flows import (
    _needs_review_check,
    make_runtime,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _patch(monkeypatch, runtime):
    from app import server
    monkeypatch.setattr(server, "graph_runtime", runtime)
    return server


def _client(server):
    transport = httpx.ASGITransport(app=server.app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.anyio
async def test_v2_plan_delivers_safe(monkeypatch):
    server = _patch(monkeypatch, make_runtime())
    async with _client(server) as client:
        resp = await client.post("/api/v2/plan", json={"query": "帮我增肌"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["delivery_status"] == "safe_delivered"
    assert body["days"]


@pytest.mark.anyio
async def test_v2_plan_goal_mismatch_returns_422(monkeypatch):
    from tests.test_graph.test_graph_flows import valid_plan
    # plan goal 增肌 vs request goal 减脂 → GoalConsistencyError → 422
    server = _patch(monkeypatch, make_runtime(plan=valid_plan(goal="增肌")))
    async with _client(server) as client:
        resp = await client.post("/api/v2/plan", json={"goal": "减脂"})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "GOAL_CONSISTENCY_FAILED"


@pytest.mark.anyio
async def test_v2_plan_interrupt_returns_review_payload(monkeypatch):
    server = _patch(monkeypatch, make_runtime(checks=[_needs_review_check()]))
    async with _client(server) as client:
        resp = await client.post("/api/v2/plan", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["delivery_status"] == "review_pending"
    assert body["thread_id"]
    assert body["review"]["review_id"]


@pytest.mark.anyio
async def test_v2_plan_stream_ends_with_done(monkeypatch):
    server = _patch(monkeypatch, make_runtime())
    async with _client(server) as client:
        resp = await client.post("/api/v2/plan/stream", json={"query": "增肌"})
    events = decode_sse(resp.text)
    terminals = [e["event"] for e in events
                 if e["event"] in {"done", "error", "cancelled"}]
    assert terminals == ["done"]
    # progressive events reuse the legacy names
    names = {e["event"] for e in events}
    assert {"planner_done", "retriever_done", "factcheck_done"} <= names


@pytest.mark.anyio
async def test_v2_resolve_approves_and_delivers(monkeypatch):
    runtime = make_runtime(checks=[_needs_review_check()])
    server = _patch(monkeypatch, runtime)
    async with _client(server) as client:
        plan = (await client.post("/api/v2/plan", json={})).json()
        review_id = plan["review"]["review_id"]

        resp = await client.post(
            f"/api/v2/reviews/{review_id}/resolve",
            json={"decision": "approved", "reviewer": "coach", "comment": "ok"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["resolution"]["decision"] == "approved"
    assert body["delivery"]["delivery_status"] == "review_approved"
    # resolution was recorded
    assert runtime.resolutions.get(review_id).decision == "approved"


@pytest.mark.anyio
async def test_v2_resolve_rejects_and_records(monkeypatch):
    runtime = make_runtime(checks=[_needs_review_check()])
    server = _patch(monkeypatch, runtime)
    async with _client(server) as client:
        plan = (await client.post("/api/v2/plan", json={})).json()
        review_id = plan["review"]["review_id"]
        resp = await client.post(
            f"/api/v2/reviews/{review_id}/resolve",
            json={"decision": "rejected", "reviewer": "coach"})
    assert resp.status_code == 200
    assert resp.json()["delivery"]["delivery_status"] == "review_rejected"


@pytest.mark.anyio
async def test_v2_double_resolve_is_conflict(monkeypatch):
    runtime = make_runtime(checks=[_needs_review_check()])
    server = _patch(monkeypatch, runtime)
    async with _client(server) as client:
        plan = (await client.post("/api/v2/plan", json={})).json()
        review_id = plan["review"]["review_id"]
        first = await client.post(f"/api/v2/reviews/{review_id}/resolve",
                                  json={"decision": "approved"})
        second = await client.post(f"/api/v2/reviews/{review_id}/resolve",
                                   json={"decision": "rejected"})
    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "REVIEW_ALREADY_RESOLVED"


@pytest.mark.anyio
async def test_v2_resolve_unknown_review_is_404(monkeypatch):
    server = _patch(monkeypatch, make_runtime())
    async with _client(server) as client:
        resp = await client.post("/api/v2/reviews/does-not-exist/resolve",
                                 json={"decision": "approved"})
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "REVIEW_NOT_FOUND"


@pytest.mark.anyio
async def test_v2_get_review_shows_artifact_and_resolution(monkeypatch):
    runtime = make_runtime(checks=[_needs_review_check()])
    server = _patch(monkeypatch, runtime)
    async with _client(server) as client:
        plan = (await client.post("/api/v2/plan", json={})).json()
        review_id = plan["review"]["review_id"]

        before = await client.get(f"/api/v2/reviews/{review_id}")
        assert before.status_code == 200
        assert before.json()["resolution"] is None

        await client.post(f"/api/v2/reviews/{review_id}/resolve",
                          json={"decision": "approved"})
        after = await client.get(f"/api/v2/reviews/{review_id}")

    assert after.status_code == 200
    body = after.json()
    assert body["review"]["review_id"] == review_id
    assert body["resolution"]["decision"] == "approved"
