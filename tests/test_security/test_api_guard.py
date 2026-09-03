"""API guard middleware: auth, admin protection, rate limiting."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.security.api_guard import (AdminAuthMiddleware, ApiKeyMiddleware,
                                    RateLimitMiddleware)


def _build_app(*, api_token="", admin_token="", rpm=60):
    app = FastAPI()

    @app.get("/health/live")
    def live():
        return {"status": "ok"}

    @app.get("/api/ping")
    def ping():
        return {"pong": True}

    @app.get("/admin/metrics")
    def metrics():
        return {"metrics": True}

    @app.get("/")
    def index():
        return {"page": True}

    app.add_middleware(AdminAuthMiddleware, admin_token=admin_token,
                       api_token=api_token)
    app.add_middleware(ApiKeyMiddleware, token=api_token)
    app.add_middleware(RateLimitMiddleware, requests_per_minute=rpm)
    return TestClient(app)


# ---------- API key auth ----------

def test_auth_disabled_by_default():
    client = _build_app()
    assert client.get("/api/ping").status_code == 200
    assert client.get("/").status_code == 200


def test_auth_rejects_missing_and_accepts_valid_tokens():
    client = _build_app(api_token="secret-123")
    assert client.get("/api/ping").status_code == 401
    ok_bearer = client.get("/api/ping",
                           headers={"Authorization": "Bearer secret-123"})
    assert ok_bearer.status_code == 200
    ok_header = client.get("/api/ping", headers={"X-API-Key": "secret-123"})
    assert ok_header.status_code == 200
    ok_query = client.get("/api/ping?api_key=secret-123")
    assert ok_query.status_code == 200
    bad = client.get("/api/ping", headers={"X-API-Key": "wrong"})
    assert bad.status_code == 401
    # non-/api routes stay open (static frontend)
    assert client.get("/").status_code == 200


# ---------- admin protection ----------

def test_admin_fails_closed_when_unconfigured():
    client = _build_app()
    resp = client.get("/admin/metrics")
    assert resp.status_code == 403
    assert resp.json()["error"] == "admin_endpoint_disabled"


def test_admin_requires_valid_token():
    client = _build_app(admin_token="admin-secret")
    assert client.get("/admin/metrics").status_code == 401
    ok = client.get("/admin/metrics", headers={"X-Admin-Token": "admin-secret"})
    assert ok.status_code == 200
    bad = client.get("/admin/metrics", headers={"X-Admin-Token": "nope"})
    assert bad.status_code == 401


def test_admin_accepts_api_key_when_configured():
    client = _build_app(api_token="api-secret")
    ok = client.get("/admin/metrics", headers={"X-API-Key": "api-secret"})
    assert ok.status_code == 200


# ---------- rate limiting ----------

def test_rate_limit_returns_429_with_retry_after():
    client = _build_app(rpm=3)
    for _ in range(3):
        assert client.get("/api/ping").status_code == 200
    blocked = client.get("/api/ping")
    assert blocked.status_code == 429
    assert int(blocked.headers["retry-after"]) >= 1
    assert blocked.json()["error"] == "rate_limited"


def test_rate_limit_exempts_health():
    client = _build_app(rpm=1)
    assert client.get("/api/ping").status_code == 200
    # health probes keep flowing even after the budget is spent
    for _ in range(5):
        assert client.get("/health/live").status_code == 200
