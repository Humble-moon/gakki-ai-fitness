"""
===========================================================================
文件角色：核心装饰器工具库 —— 为 Agent 方法提供重试和超时能力
===========================================================================
- 被谁调用：各 Agent 类（PlannerAgent, RetrieverAgent, WriterAgent 等）通过
            @with_retry / @with_timeout 装饰其内部方法
- 调用谁：无外部依赖，仅使用 Python 标准库
- 核心职责：
    1. with_retry: 指数退避重试，应对 LLM API 的瞬时故障（限流、网络波动）
    2. with_timeout: 线程级超时控制，防止 LLM 调用无限挂起拖垮整个请求
===========================================================================
"""

import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)


def with_retry(max_retries: int = 3, backoff: float = 2.0):
    """装饰器工厂：为被装饰函数提供指数退避重试能力。

    输入参数：
        max_retries: int — 最大重试次数（包含首次调用），默认 3 次
        backoff: float — 退避基数，第 n 次重试等待 backoff^n 秒，默认 2.0
    返回：
        decorator — 可应用于任意函数的装饰器

    核心逻辑：
        1. 循环调用原函数，成功则直接返回
        2. 捕获异常 → 记录日志 → 指数退避等待 → 重试
        3. 耗尽重试次数后，抛出最后一次捕获的异常

    使用场景：
        LLM API 调用偶发限流（429）、网关超时（504）、网络抖动等瞬时故障，
        指数退避能避免雪崩效应，给服务端恢复时间。

    线程安全注意：
        本装饰器本身不引入共享状态，但被装饰函数需自行保证线程安全。
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    # 指数退避：backoff^attempt 秒 —— attempt=0→1s, 1→2s, 2→4s
                    wait = backoff ** attempt
                    logger.warning(f"Retry {attempt + 1}/{max_retries} for {func.__name__}: {e}")
                    time.sleep(wait)
            # 所有重试耗尽，抛出最后一次错误（保留原始 traceback）
            raise last_error
        return wrapper
    return decorator


def with_timeout(seconds: int = 60):
    """装饰器工厂：为被装饰函数提供超时控制。

    输入参数：
        seconds: int — 超时阈值（秒），默认 60 秒
    返回：
        decorator — 可应用于任意函数的装饰器

    核心逻辑：
        1. 在新线程中执行原函数（Python 主线程无法被强制中断）
        2. 主线程调用 t.join(seconds) 等待
        3. 超时 → 抛出 TimeoutError；正常完成 → 返回结果或重抛异常

    设计决策 —— 为什么用线程而不是 signal：
        signal 仅限主线程且仅限 Unix，线程方案跨平台兼容，适合 Windows 和 Linux 部署。
        daemon=True 确保超时后守护线程不会阻塞进程退出。

    局限性：
        - 超时后原线程仍在运行（Python 无法安全终止线程），但调用方已收到异常
        - 不适合需要资源清理的场景（数据库事务、文件写入等）
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 用列表作为闭包可变容器，因为 Python 内部函数不能直接赋值外部 nonlocal 变量
            result = [None]
            exception = [None]

            def target():
                """在新线程中执行的函数。结果或异常存储在闭包列表中以传回主线程。"""
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            import threading
            t = threading.Thread(target=target)
            t.daemon = True  # 守护线程：主线程退出时自动终止，不会阻止进程退出
            t.start()
            t.join(seconds)  # 阻塞等待，但最多等 seconds 秒
            if t.is_alive():
                raise TimeoutError(f"{func.__name__} timed out after {seconds}s")
            if exception[0]:
                raise exception[0]
            return result[0]
        return wrapper
    return decorator
