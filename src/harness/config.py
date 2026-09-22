"""执行脚手架（harness）的集中配置。

本模块是"模型之外那层脚手架"所有可调参数的**唯一事实源**。分散在各处的
魔法数字集中到这里，好处有三：

1. 调参时不必在 7 个文件里 grep；
2. ``tests/test_harness_contract.py`` 会断言这里的取值与各消费模块实际使用的
   常量一致，任何一处改了而另一处没跟上都会让测试失败（防止文档与代码漂移）；
3. 面试/评审时可以直接回答"你的 agent 预算是多少"，而不是翻代码。

注意：这些值是**当前实现的实际取值**，不是理想值。改动它们会改变行为，
需要重跑评测。
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 重试与退避（消费方：src/llm/provider.py）
# ---------------------------------------------------------------------------

#: LLM 调用失败后的最大尝试次数（含首次调用）。
RETRY_MAX_ATTEMPTS = 3

#: 指数退避基数：第 n 次重试等待 ``RETRY_BACKOFF_BASE ** n`` 秒。
RETRY_BACKOFF_BASE = 2.0


# ---------------------------------------------------------------------------
# 编排循环上限
# ---------------------------------------------------------------------------

#: FactChecker 判定不安全时，允许的最大重写轮次。
#: 与 ``src.graph.state.MAX_REWRITES`` 一致——那是 LangGraph 路径上的同义常量。
MAX_REWRITES = 3

#: LangGraph 单次运行的最大 superstep 数。
#: 最坏情况约 16 步（3 轮重写），60 留了充足余量。
#: 与 ``src.graph.runtime.RECURSION_LIMIT`` 一致。
RECURSION_LIMIT = 60


# ---------------------------------------------------------------------------
# Agent 自主循环预算（消费方：src/harness/loop.py）
# ---------------------------------------------------------------------------

#: 单次自主循环允许的最大工具调用步数。
#: 没有上限的 agent loop 在生产环境等于失控的账单——这是硬性刹车。
AGENT_MAX_STEPS = 10

#: 单次自主循环允许消耗的 token 上限，触顶即中止并交由调用方兜底。
AGENT_TOKEN_BUDGET = 50_000


# ---------------------------------------------------------------------------
# 超时
# ---------------------------------------------------------------------------

#: 单次被装饰操作的默认墙钟超时（秒）。
DEFAULT_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class HarnessConfig:
    """一次执行的完整脚手架配置。

    冻结（frozen）是刻意的：配置在一次运行内不应被中途改写，
    否则 checkpoint 续跑时的行为会与首次运行不一致。
    """

    retry_max_attempts: int = RETRY_MAX_ATTEMPTS
    retry_backoff_base: float = RETRY_BACKOFF_BASE
    max_rewrites: int = MAX_REWRITES
    recursion_limit: int = RECURSION_LIMIT
    agent_max_steps: int = AGENT_MAX_STEPS
    agent_token_budget: int = AGENT_TOKEN_BUDGET
    default_timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    #: "memory" | "sqlite" | "postgres"，见 src/graph/runtime.py
    checkpoint_backend: str = "sqlite"


DEFAULT_CONFIG = HarnessConfig()
