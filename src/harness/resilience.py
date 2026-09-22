"""失败恢复原语：墙钟超时与重试策略。

从 ``src/core/harness.py`` 迁移而来。原文件全仓库零引用，其 ``with_retry``
实现还会**无条件重试所有异常**——401 认证失败、400 参数错误这类逻辑错误
重试三次纯属浪费，且会掩盖真正的配置问题。LLM 层（``src/llm/provider.py``）
已经有一个只重试瞬时故障的正确实现，本模块不再提供第二个竞争实现，
只把可复用的部分保留下来：

* :func:`with_timeout` —— 全项目唯一的墙钟超时能力，LLM 层没有替代品；
* :class:`RetryPolicy` —— 重试策略的声明式描述，供消费方引用而非各自硬编码。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import wraps

from src.harness.config import RETRY_BACKOFF_BASE, RETRY_MAX_ATTEMPTS


class TimeoutExceeded(TimeoutError):
    """被装饰的操作超过墙钟时限。

    继承 ``TimeoutError`` 以便调用方按内置异常类型捕获。
    """


@dataclass(frozen=True)
class RetryPolicy:
    """重试策略描述。

    这里只有**参数**，没有重试循环：重试循环留在消费方（如 LLM 层），
    因为它需要按异常类型决定是否可重试——那是领域知识，不属于通用脚手架。
    """

    max_attempts: int = RETRY_MAX_ATTEMPTS
    backoff_base: float = RETRY_BACKOFF_BASE

    def wait_seconds(self, attempt: int) -> float:
        """第 ``attempt`` 次重试（从 0 开始）前应等待的秒数。"""
        return self.backoff_base ** attempt


DEFAULT_RETRY_POLICY = RetryPolicy()


def with_timeout(seconds: int = 60):
    """装饰器工厂：为被调用函数提供墙钟超时控制。

    实现：在新线程中执行原函数，主线程 ``join(seconds)``；超时即抛出
    :class:`TimeoutExceeded`，不再等待原线程。

    为什么用线程而不是 ``signal.alarm``：
        ``signal`` 只能在主线程使用且仅限 Unix；线程方案跨平台，
        对 FastAPI 这类多线程服务也更实际。

    已知局限（刻意保留，不做隐藏）：
        超时后**原线程仍在后台运行**——Python 无法安全终止线程。
        调用方拿到异常即认为本次调用结束，但被装饰函数若持有数据库事务、
        文件句柄或网络连接，仍需自行保证幂等与清理。因此本装饰器适合
        包裹"幂等的只读/可重算操作"，不适合包裹有副作用的写操作。
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 用单元素列表而非 nonlocal：闭包内赋值 nonlocal 需要额外声明，
            # 列表容器更直观，且与线程间传递结果的方式一致。
            result: list = [None]
            error: list = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except BaseException as exc:  # noqa: BLE001 - 原样透传给主线程
                    error[0] = exc

            worker = threading.Thread(target=target)
            worker.daemon = True  # 超时后不阻塞进程退出
            worker.start()
            worker.join(seconds)

            if worker.is_alive():
                raise TimeoutExceeded(f"{func.__name__} 超过 {seconds}s 未返回")
            if error[0] is not None:
                raise error[0]
            return result[0]

        return wrapper

    return decorator
