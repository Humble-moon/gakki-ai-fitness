"""ReAct 式 Agent 自主循环 —— 让模型自己决定调哪个工具、调几次。

## 为什么要这一层

项目现有的流水线是**固定路径**：Planner → Retriever → Writer → FactChecker
由 ``src/graph/builder.py`` 写死。这对训练计划生成这类确定性任务是正确选择
——路径可预测、可评测、延迟低。

但知识问答这类**开放式任务**的步骤数事先未知：可能要查两次动作库、一次
图谱，也可能一次都不用。固定路径只能把所有工具都调一遍（浪费）或调最少的
（漏掉）。自主循环把"调什么工具"交还给模型。

因此本模块是**可选路径**，不是现有流水线的替代品。默认不启用，
由 ``HARNESS_AGENT_LOOP`` 开关控制（见 :func:`agent_loop_enabled`）。

## 与 LLM 层的关系

``src/llm/provider.py`` 目前**不支持原生 function calling**——全项目没有
``tools=`` 参数，工具都是由 agent 代码直接调用的。因此本循环不依赖任何
特定的工具调用传输方式，而是通过 :class:`ToolCallingModel` 协议注入：

* 生产环境用 :class:`~src.harness.tool_calling.PromptToolCallingModel`
  （把工具 schema 渲染进 prompt，要求模型返回结构化 JSON）；
* 测试用假的 model 对象，从而在不接触网络的前提下验证循环语义。

这样即使将来 provider 支持了原生 function calling，只需换一个 adapter，
循环本身不用动。

## 三个不可省略的设计点

1. **预算硬上限**：``max_steps`` 与 ``token_budget``。没有刹车的 agent loop
   在生产环境是一张失控的账单。
2. **工具错误回喂模型**：工具抛异常时不是中断整个流程，而是把错误作为
   工具结果交回模型，让它自己换策略。这是"能用"和"好用"的分界线。
   连续失败仍会触达步数上限而终止。
3. **每步可选回调**：``on_step`` 让调用方落 checkpoint，使循环可续跑。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from src.harness.config import AGENT_MAX_STEPS, AGENT_TOKEN_BUDGET

logger = logging.getLogger(__name__)

#: 环境变量开关。设为 "1"/"true" 时启用自主循环路径。
AGENT_LOOP_ENV = "HARNESS_AGENT_LOOP"


def agent_loop_enabled() -> bool:
    """自主循环是否启用。默认关闭——确定性流水线仍是默认路径。"""
    return os.environ.get(AGENT_LOOP_ENV, "").strip().lower() in {"1", "true", "yes"}


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """模型请求调用某个工具。"""

    name: str
    args: dict
    #: 同一次回复中多个调用的配对 id；模型不提供时由循环补一个。
    call_id: str = ""


@dataclass
class ModelTurn:
    """模型一次回复：要么给出最终答案，要么请求调用工具。"""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tokens: int = 0


@dataclass
class StepRecord:
    """单步执行记录，供评测统计与调试回溯。"""

    step: int
    tool: str
    args: dict
    ok: bool
    result_preview: str
    error_type: str | None = None


@dataclass
class LoopResult:
    """循环的终态。

    ``exited_reason`` 的取值决定了调用方该怎么处理：

    * ``"completed"``    —— 模型给出了最终答案，可正常返回；
    * ``"max_steps"``    —— 步数耗尽，**没有**最终答案，调用方必须兜底；
    * ``"token_budget"`` —— token 超支，同上。
    """

    content: str
    steps: int
    tokens_used: int
    transcript: list[StepRecord]
    exited_reason: str
    tool_errors: int = 0

    @property
    def completed(self) -> bool:
        return self.exited_reason == "completed"


@runtime_checkable
class ToolCallingModel(Protocol):
    """循环依赖的最小模型接口。

    只要求一个方法，便于测试用几十行假对象替换，也便于将来接入
    provider 的原生 function calling。
    """

    def complete(self, messages: list, tools: list) -> ModelTurn:
        """给定对话与可用工具，返回模型的下一步动作。"""
        ...


class ToolInvoker(Protocol):
    """工具集合的最小接口，与 ``src.mcp.tool_registry.ToolRegistry`` 兼容。"""

    def call(self, name: str, args: dict) -> Any: ...

    def list_tools(self) -> list[dict]: ...


# ---------------------------------------------------------------------------
# 循环
# ---------------------------------------------------------------------------

#: 单个工具结果回喂模型时的截断长度，防止一次返回撑爆上下文窗口。
_RESULT_PREVIEW_CHARS = 500


def run_agent_loop(
    model: ToolCallingModel,
    tools: ToolInvoker,
    messages: list,
    *,
    max_steps: int = AGENT_MAX_STEPS,
    token_budget: int = AGENT_TOKEN_BUDGET,
    on_step: Callable[[StepRecord], None] | None = None,
) -> LoopResult:
    """运行 ReAct 循环直到模型给出答案或触达预算。

    Args:
        model: 见 :class:`ToolCallingModel`。
        tools: 提供 ``call(name, args)`` 与 ``list_tools()`` 的对象。
        messages: OpenAI 格式的消息列表，**会被就地追加**（工具结果与
            模型回复都会写进去），因此调用方若要保留原始输入请先拷贝。
        max_steps: 最大工具调用轮次。
        token_budget: token 上限，累计模型消耗超过即中止。
        on_step: 每完成一次工具调用后调用，用于落 checkpoint。

    Returns:
        :class:`LoopResult`。**始终检查 ``exited_reason``**：非 ``"completed"``
        时 ``content`` 为空串，调用方必须走兜底路径，不能把空串当答案返回。
    """
    transcript: list[StepRecord] = []
    tokens_used = 0
    tool_errors = 0
    available = {t.get("name") for t in tools.list_tools()}

    for step in range(max_steps):
        turn = model.complete(messages, tools.list_tools())
        tokens_used += turn.tokens or 0

        if tokens_used > token_budget:
            logger.warning(
                "[harness.loop] token 预算超支：%d > %d，在第 %d 步中止",
                tokens_used, token_budget, step + 1,
            )
            return LoopResult(
                content="", steps=step + 1, tokens_used=tokens_used,
                transcript=transcript, exited_reason="token_budget",
                tool_errors=tool_errors,
            )

        # 模型没有请求工具 -> 它给出了最终答案，循环结束。
        if not turn.tool_calls:
            return LoopResult(
                content=turn.content or "", steps=step + 1, tokens_used=tokens_used,
                transcript=transcript, exited_reason="completed",
                tool_errors=tool_errors,
            )

        for call in turn.tool_calls:
            ok, payload, error_type = _invoke_tool(tools, available, call)
            if not ok:
                tool_errors += 1

            record = StepRecord(
                step=step + 1,
                tool=call.name,
                args=call.args,
                ok=ok,
                result_preview=str(payload)[:_RESULT_PREVIEW_CHARS],
                error_type=error_type,
            )
            transcript.append(record)

            # 关键：无论成败都把结果交回模型。失败时给的是错误信息而非异常，
            # 模型由此得知"这条路不通"，可以换工具或换参数。
            messages.append({
                "role": "tool",
                "tool_call_id": call.call_id or f"call_{step}_{call.name}",
                "name": call.name,
                "content": str(payload)[:_RESULT_PREVIEW_CHARS],
            })

            if on_step is not None:
                on_step(record)

    logger.warning("[harness.loop] 步数耗尽（%d 步），未得到最终答案", max_steps)
    return LoopResult(
        content="", steps=max_steps, tokens_used=tokens_used,
        transcript=transcript, exited_reason="max_steps", tool_errors=tool_errors,
    )


def _invoke_tool(
    tools: ToolInvoker, available: set, call: ToolCall
) -> tuple[bool, Any, str | None]:
    """调用一个工具，把任何异常收敛成 (False, 错误描述, 异常类型)。

    未知工具名单独处理：模型幻觉出不存在的工具是常见失败模式，
    给出明确提示比抛 KeyError 更容易让它自我纠正。
    """
    if available and call.name not in available:
        return False, {
            "error": "unknown_tool",
            "message": f"没有名为 {call.name} 的工具",
            "available_tools": sorted(t for t in available if t),
        }, "UnknownTool"

    try:
        return True, tools.call(call.name, call.args), None
    except Exception as exc:  # noqa: BLE001 - 故意吞掉并回喂，而非中断循环
        logger.info("[harness.loop] 工具 %s 失败：%s", call.name, exc)
        return False, {
            "error": type(exc).__name__,
            "message": str(exc),
        }, type(exc).__name__
