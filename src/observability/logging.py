"""结构化日志与请求关联 ID。

为什么要 JSON：单机开发时人眼读纯文本更舒服，但一旦有日志采集系统，
"时间戳 + 级别 + 消息"这种格式要靠正则才能拆。结构化输出让每条日志
带上可过滤的字段，也才能把同一请求的多条日志串起来。

``request_id`` 走 ContextVar 而不是塞进 logger 的 extra：FastAPI 的同步
端点跑在线程池里，参数一路传下去会污染所有业务函数签名。
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone
from uuid import uuid4

from starlette.datastructures import MutableHeaders

from src.config import LOG_JSON, LOG_LEVEL

#: 当前请求的关联 ID。没有请求上下文时是 ``-``（例如启动期日志）。
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-")

#: 入站请求头名；网关或前端可自带，便于与外部链路对齐。
REQUEST_ID_HEADER = "x-request-id"


class JsonFormatter(logging.Formatter):
    """一行一条 JSON，附加 request_id 与异常栈。"""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(
                record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # 业务侧通过 extra 传的字段一并带上，不用改 Formatter。
        for key in ("action", "actor", "target", "status_code", "duration_ms"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False, default=str)


class _SafeStreamHandler(logging.StreamHandler):
    """写流已关闭时静默跳过。

    进程退出阶段（atexit 里的成本汇总）stdout 可能已经被关闭，
    标准 StreamHandler 会抛 ValueError 并在 stderr 打一屏堆栈——
    那不是业务错误，不该淹没真正的退出信息。
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            super().emit(record)
        except (ValueError, OSError):
            pass


def configure_logging() -> None:
    """统一 root 与 uvicorn 的日志出口。

    测试环境直接返回：pytest 自己管理日志捕获，替换 root handler 会
    让 caplog 失效，也会在捕获流关闭后刷出无关堆栈。
    """
    if "pytest" in sys.modules:
        return

    # 日志走 stderr：stdout 留给程序本身的输出。否则 `--json` 之类的
    # CLI 结果会被日志行污染，解析方拿到的就不是纯 JSON 了。
    handler = _SafeStreamHandler(sys.stderr)
    if LOG_JSON:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

    level = getattr(logging, str(LOG_LEVEL).upper(), logging.INFO)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = [handler]
        uvicorn_logger.propagate = False
        uvicorn_logger.setLevel(level)


class RequestIdMiddleware:
    """给每个请求一个关联 ID，并回写到响应头。

    纯 ASGI 实现（而不是 BaseHTTPMiddleware）：后者的流式响应会多一层
    封装，SSE 的首字节延迟会被放大，而这个项目的主力接口全是 SSE。
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        incoming = headers.get(REQUEST_ID_HEADER.encode())
        request_id = (
            incoming.decode("latin-1")[:64] if incoming else uuid4().hex[:12]
        )
        token = request_id_var.set(request_id)

        async def send_with_id(message):
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            request_id_var.reset(token)
