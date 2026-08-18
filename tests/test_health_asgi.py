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
async def test_sse_asgi_uses_fake_orchestrator_without_external_services(client):
    response = await client.post("/api/generate-plan", json={"session_id": "offline"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert '"event": "done"' in response.text
