"""API guard middleware: optional key auth, admin protection, rate limiting.

Three independent hardening layers, all configured through environment
variables and all safe by default for the single-user local demo:

1. ``ApiKeyMiddleware`` — bearer-token auth for the business API. Disabled when
   ``API_AUTH_TOKEN`` is empty (local demo); when set, every /api/* request must
   carry the token via ``Authorization: Bearer <t>``, ``X-API-Key`` header, or
   ``?api_key=<t>`` (the query form exists because browser EventSource/SSE
   clients cannot set custom headers).

2. ``AdminAuthMiddleware`` — /admin/* endpoints always require the dedicated
   ``ADMIN_TOKEN`` (``X-Admin-Token`` header) or a valid API key. When neither
   is configured the endpoints fail closed with 403: an unauthenticated metrics
   or re-ingest surface is exactly the gap this module exists to close.

3. ``RateLimitMiddleware`` — per-client-IP sliding window (default 60 req/min)
   protecting against accidental hammering and brute force. Health probes are
   exempt so liveness checks never trip the limiter.

Implemented as pure ASGI middleware (not BaseHTTPMiddleware) so SSE streams
pass through without extra buffering.
"""

import json
import os
import threading
import time
from collections import deque


def _json_response(send, status: int, body: dict, extra_headers=None):
    """Send a minimal JSON response from raw ASGI primitives."""
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = [
        (b"content-type", b"application/json; charset=utf-8"),
        (b"content-length", str(len(payload)).encode()),
    ]
    for key, value in (extra_headers or []):
        headers.append((key.encode(), value.encode()))

    async def _send():
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": payload})

    return _send()


def _query_params(scope) -> dict:
    from urllib.parse import parse_qs
    return {k: v[0] for k, v in parse_qs(scope.get("query_string", b"").decode()).items()}


def _header(scope, name: str) -> str | None:
    target = name.lower().encode()
    for key, value in scope.get("headers", []):
        if key == target:
            return value.decode()
    return None


def _client_ip(scope) -> str:
    client = scope.get("client")
    return client[0] if client else "unknown"


class ApiKeyMiddleware:
    """Optional bearer-token auth for /api/* routes."""

    def __init__(self, app, token: str = "", protected_prefix: str = "/api"):
        self.app = app
        self.token = token
        self.protected_prefix = protected_prefix

    def _authorized(self, scope) -> bool:
        auth = _header(scope, "authorization")
        if auth and auth.startswith("Bearer ") and auth[7:] == self.token:
            return True
        if _header(scope, "x-api-key") == self.token:
            return True
        if _query_params(scope).get("api_key") == self.token:
            return True
        return False

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self.token:
            await self.app(scope, receive, send)
            return
        path = scope["path"]
        if not path.startswith(self.protected_prefix):
            await self.app(scope, receive, send)
            return
        if self._authorized(scope):
            await self.app(scope, receive, send)
            return
        await _json_response(send, 401, {
            "error": "unauthorized",
            "message": "请求需要有效的 API 密钥（Authorization: Bearer / X-API-Key / ?api_key=）",
        })


class AdminAuthMiddleware:
    """Protect /admin/* with a dedicated token; fail closed when unconfigured."""

    def __init__(self, app, admin_token: str = "", api_token: str = "",
                 prefix: str = "/admin"):
        self.app = app
        self.admin_token = admin_token
        self.api_token = api_token
        self.prefix = prefix

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith(self.prefix):
            await self.app(scope, receive, send)
            return
        if not self.admin_token and not self.api_token:
            await _json_response(send, 403, {
                "error": "admin_endpoint_disabled",
                "message": "管理端点未配置 ADMIN_TOKEN，已按失败关闭原则拒绝访问；"
                           "在 .env 中设置 ADMIN_TOKEN 后通过 X-Admin-Token 头部访问。",
            })
            return
        supplied_admin = _header(scope, "x-admin-token")
        if self.admin_token and supplied_admin == self.admin_token:
            await self.app(scope, receive, send)
            return
        if self.api_token and _header(scope, "x-api-key") == self.api_token:
            await self.app(scope, receive, send)
            return
        await _json_response(send, 401, {
            "error": "admin_unauthorized",
            "message": "管理端点需要有效的 X-Admin-Token 或 X-API-Key。",
        })


class RateLimitMiddleware:
    """Per-client-IP sliding window rate limiter."""

    def __init__(self, app, requests_per_minute: int = 60,
                 exempt_prefixes: tuple = ("/health",)):
        self.app = app
        self.capacity = max(1, requests_per_minute)
        self.window = 60.0
        self.exempt_prefixes = exempt_prefixes
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def _allow(self, client: str) -> tuple[bool, int]:
        now = time.monotonic()
        with self._lock:
            window = self._hits.setdefault(client, deque())
            while window and now - window[0] > self.window:
                window.popleft()
            if len(window) >= self.capacity:
                retry_after = int(self.window - (now - window[0])) + 1
                return False, max(1, retry_after)
            window.append(now)
            return True, 0

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"].startswith(self.exempt_prefixes):
            await self.app(scope, receive, send)
            return
        allowed, retry_after = self._allow(_client_ip(scope))
        if not allowed:
            await _json_response(send, 429, {
                "error": "rate_limited",
                "message": f"请求过于频繁，请 {retry_after} 秒后重试。",
            }, extra_headers=[("retry-after", str(retry_after))])
            return
        await self.app(scope, receive, send)


def cors_allow_origins() -> list[str]:
    """CORS origin whitelist from config; localhost defaults for the demo."""
    raw = os.getenv(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:8503,http://127.0.0.1:8503,http://localhost:3000",
    )
    return [o.strip() for o in raw.split(",") if o.strip()]
