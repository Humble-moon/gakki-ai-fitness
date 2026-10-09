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
    """Optional bearer-token auth for /api/* routes.

    ``exempt_prefixes`` 存在的理由：登录、注册这些端点必须能在无凭据时访问，
    否则会形成鸡生蛋——要拿令牌得先有令牌。它同时也是全局 API key 与
    「按用户的登录态」两套机制共存的接缝。
    """

    def __init__(self, app, token: str = "", protected_prefix: str = "/api",
                 exempt_prefixes: tuple = ()):
        self.app = app
        self.token = token
        self.protected_prefix = protected_prefix
        self.exempt_prefixes = exempt_prefixes

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
        # CORS 预检不带凭据（浏览器规范如此），拦下它就等于永远走不到 CORS 层，
        # 前端会看到预检 401 而不是真正的跨源策略。预检本身不读业务数据，
        # 放行不降低安全性。
        if scope.get("method") == "OPTIONS":
            await self.app(scope, receive, send)
            return
        path = scope["path"]
        if not path.startswith(self.protected_prefix):
            await self.app(scope, receive, send)
            return
        if any(path.startswith(p) for p in self.exempt_prefixes):
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


class CostLimitMiddleware:
    """当日 LLM 成本超限时拒绝会花钱的请求。

    与 ``RateLimitMiddleware`` 互补：限流管"请求太频繁"（防瞬时冲击），
    本中间件管"今天已经花超了"（防成本在一天内无感累积）。

    拦截范围用**排除法**而不是白名单：默认拦所有会触发生成的请求，
    只放行明确不花钱的端点。这样将来新增端点时默认受保护，
    不会因为忘了加白名单而留下缺口。
    """

    #: 只写本地存储、不调用 LLM 的端点，成本超限时也应可用。
    _FREE_SUFFIXES = ("/training-logs",)

    #: 不消耗 LLM 的前缀。认证端点在列：额度用完时若连登录、登出都被拦，
    #: 用户会被卡在一个既不能用、也退不出的状态——而这两个操作本身不花钱。
    _FREE_PREFIXES = ("/api/auth/",)

    def __init__(self, app, guard, protected_prefix: str = "/api"):
        self.app = app
        self.guard = guard
        self.protected_prefix = protected_prefix

    def _is_free(self, path: str) -> bool:
        return (any(path.endswith(s) for s in self._FREE_SUFFIXES)
                or any(path.startswith(p) for p in self._FREE_PREFIXES))

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope["path"]
        # 只拦写操作：GET 是查询，不产生 LLM 开销
        if (
            scope.get("method") != "POST"
            or not path.startswith(self.protected_prefix)
            or self._is_free(path)
        ):
            await self.app(scope, receive, send)
            return

        allowed, spent = self.guard.check()
        if not allowed:
            await _json_response(send, 429, {
                "error": "daily_cost_limit_reached",
                "message": (
                    f"今日生成额度已用完（已消耗 ¥{spent:.2f} / 上限 "
                    f"¥{self.guard.daily_limit:.2f}），请明天再试。"
                ),
                "spent_today": round(spent, 4),
                "daily_limit": self.guard.daily_limit,
            })
            return
        await self.app(scope, receive, send)


def cors_allow_origins() -> list[str]:
    """CORS origin whitelist from config; localhost defaults for the demo."""
    raw = os.getenv(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:8503,http://127.0.0.1:8503,http://localhost:3000",
    )
    return [o.strip() for o in raw.split(",") if o.strip()]
