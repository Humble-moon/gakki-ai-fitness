# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# libpq 供 psycopg 使用；build-essential 应对少数需要编译的依赖。
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

# requirements.lock 由 `python -m pip freeze` 生成，锁定全部传递依赖；
# 只有范围约束的 requirements.txt 会让"今天能构建、明天不能"成为常态。
COPY requirements.txt requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

COPY . .

# 非 root 运行：容器逃逸的收益与容器内权限成正比，能不给 root 就不给。
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8503

# 探针走 /health/ready：它检查的正是应用真正依赖的东西，而不是"进程还活着"。
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8503/health/ready || exit 1

# workers=1 是有意的：限流器、熔断器与部分成本计数仍是进程内状态，
# 多 worker 会让同一 IP 的请求落到不同进程、各自计数。要横向扩展，
# 先把这些状态迁到 Redis（见 docs/known-limitations.md）。
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8503", "--workers", "1"]
