from __future__ import annotations

import httpx
import pytest

from scripts.run_e2e import decode_sse


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client(monkeypatch):
    from app import server

    class FakeOrchestrator:
        def generate_plan_stream(self, *args):
            yield "stage", "plan"
            yield "done", {"success": True}

        def analyze_exercise_stream(self, *args):
            yield "stage", "analysis"
            yield "done", {"success": True}

        def answer_question_stream(self, *args):
            yield "stage", "answer"
            yield "done", {"success": True}

    monkeypatch.setattr(server, "orch", FakeOrchestrator())
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as test_client:
        yield test_client


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/api/generate-plan", {"query": "增肌"}),
        ("/api/analyze-exercise", {"exercise_name": "深蹲"}),
        ("/api/ask-question", {"question": "怎么练？"}),
    ],
)
async def test_business_flows_require_explicit_done(client, path, payload):
    events = decode_sse((await client.post(path, json=payload)).text)
    terminals = [event["event"] for event in events if event["event"] in {"done", "error", "cancelled"}]
    assert terminals == ["done"]


@pytest.mark.anyio
async def test_eof_without_terminal_is_not_success(client, monkeypatch):
    from app import server

    class Incomplete:
        def generate_plan_stream(self, *args):
            yield "stage", "started"

        analyze_exercise_stream = generate_plan_stream
        answer_question_stream = generate_plan_stream

    monkeypatch.setattr(server, "orch", Incomplete())
    response = await client.post("/api/generate-plan", json={})
    events = decode_sse(response.text)
    assert events[-1]["event"] == "error"
    assert events[-1]["data"]["code"] == "STREAM_INCOMPLETE"


def test_decode_sse_preserves_terminal_events():
    events = decode_sse('data: {"event":"cancelled","data":{}}\n\n')
    assert events == [{"event": "cancelled", "data": {}}]
