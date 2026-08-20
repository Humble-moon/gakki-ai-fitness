import httpx
import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client(monkeypatch):
    from app import server

    class FakeOrchestrator:
        def generate_plan_stream(self, *args):
            yield "stage", "offline"
            yield "done", {"success": True}

        def analyze_exercise_stream(self, *args):
            yield "done", {"success": True}

        def answer_question_stream(self, *args):
            yield "done", {"success": True}

    monkeypatch.setattr(server, "orch", FakeOrchestrator())
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as test_client:
        yield test_client


@pytest.mark.anyio
async def test_live_health_is_200(client):
    response = await client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_ready_health_reports_dependency_failure(client, monkeypatch):
    from src import health

    monkeypatch.setattr(health, "check_dependencies", lambda: {"provider": "down"})
    response = await client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["checks"] == {"provider": "down"}


@pytest.mark.anyio
async def test_generate_plan_rejects_invalid_goal(client):
    response = await client.post("/api/generate-plan", json={"goal": "塑形"})
    assert response.status_code == 422


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/api/generate-plan", {"session_id": "offline"}),
        ("/api/analyze-exercise", {"exercise_name": "深蹲"}),
        ("/api/ask-question", {"question": "怎么练？"}),
    ],
)
async def test_all_sse_routes_apply_stream_headers(client, path, payload):
    from app.server import STREAM_HEADERS

    response = await client.post(path, json=payload)
    assert response.status_code == 200
    for name, value in STREAM_HEADERS.items():
        assert response.headers[name] == value


def test_stream_stops_and_closes_upstream_after_first_terminal():
    from app.server import _stream_events

    class Source:
        def __init__(self):
            self.closed = False

        def __iter__(self):
            yield "done", {"success": True}
            yield "stage", "must not escape"

        def close(self):
            self.closed = True

    source = Source()
    frames = list(_stream_events(source))
    assert len(frames) == 1
    assert '"event": "done"' in frames[0]
    assert "must not escape" not in frames[0]
    assert source.closed is True


def test_stream_accepts_review_pending_as_the_single_done_terminal_payload():
    from app.server import _stream_events

    frames = list(_stream_events(iter([
        ("done", {"delivery_status": "review_pending", "review": {"review_id": "review-1"}}),
        ("stage", "must not escape"),
    ])))

    assert len(frames) == 1
    assert '"event": "done"' in frames[0]
    assert '"delivery_status": "review_pending"' in frames[0]
    assert "must not escape" not in frames[0]


def test_stream_generator_exit_closes_upstream_without_yielding():
    from app.server import _stream_events

    class Source:
        def __init__(self):
            self.closed = False

        def __iter__(self):
            yield "stage", "started"
            yield "stage", "later"

        def close(self):
            self.closed = True

    source = Source()
    stream = _stream_events(source)
    assert '"event": "stage"' in next(stream)
    stream.close()
    assert source.closed is True
