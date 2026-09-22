"""Harness —— 模型之外的执行脚手架层。

这个包本身**不实现业务逻辑**，它做两件事：

1. 给散落在各处的执行能力一个统一的命名与入口，让"脚手架"成为一个
   可以被讨论、被评测、被替换的层，而不是七零八落的工具函数；
2. 集中所有可调的执行预算（重试次数、重写轮次、循环步数、token 上限）。

能力清单见 :data:`CAPABILITIES`。这不是装饰性的文档——它被
``tests/test_harness_contract.py`` 用来断言每个能力对应的模块真实存在，
从而在有人重构掉某个模块时立刻报警，而不是等到线上才发现少了兜底。
"""

from __future__ import annotations

from src.harness.config import (
    AGENT_MAX_STEPS,
    AGENT_TOKEN_BUDGET,
    DEFAULT_CONFIG,
    DEFAULT_TIMEOUT_SECONDS,
    HarnessConfig,
    MAX_REWRITES,
    RECURSION_LIMIT,
    RETRY_BACKOFF_BASE,
    RETRY_MAX_ATTEMPTS,
)
from src.harness.loop import (
    LoopResult,
    ModelTurn,
    StepRecord,
    ToolCall,
    agent_loop_enabled,
    run_agent_loop,
)
from src.harness.resilience import (
    DEFAULT_RETRY_POLICY,
    RetryPolicy,
    TimeoutExceeded,
    with_timeout,
)
from src.harness.tool_calling import PromptToolCallingModel

__all__ = [
    # 配置
    "HarnessConfig",
    "DEFAULT_CONFIG",
    "RETRY_MAX_ATTEMPTS",
    "RETRY_BACKOFF_BASE",
    "MAX_REWRITES",
    "RECURSION_LIMIT",
    "AGENT_MAX_STEPS",
    "AGENT_TOKEN_BUDGET",
    "DEFAULT_TIMEOUT_SECONDS",
    # 失败恢复
    "RetryPolicy",
    "DEFAULT_RETRY_POLICY",
    "with_timeout",
    "TimeoutExceeded",
    # Agent 自主循环
    "run_agent_loop",
    "agent_loop_enabled",
    "LoopResult",
    "ModelTurn",
    "ToolCall",
    "StepRecord",
    "PromptToolCallingModel",
    # 能力清单
    "CAPABILITIES",
]


#: 脚手架能力 -> 承载它的模块。
#:
#: 这些模块**早于本包存在**。本包不搬动它们（搬动会打散 git 历史、打断
#: 既有 import），只是把它们登记在同一张表里，使"harness 由什么构成"
#: 有一个可回答的答案。
CAPABILITIES: dict[str, str] = {
    # 编排与状态
    "graph_orchestration": "src.graph.builder",
    "graph_runtime": "src.graph.runtime",
    "state_schema": "src.graph.state",
    # Agent 自主循环（本包新增）
    "agent_loop": "src.harness.loop",
    # 工具
    "tool_registry": "src.mcp.tool_registry",
    # 上下文
    "conversation_context": "src.memory.conversation",
    "long_term_memory": "src.memory.long_term",
    # 人工闸门
    "human_in_the_loop": "src.hitl.review",
    # 输出约束与校验
    "goal_contract": "src.core.goal_contract",
    "output_validation": "src.agents.output_validation",
    # 失败恢复
    "llm_resilience": "src.llm.provider",
    "timeout": "src.harness.resilience",
}
