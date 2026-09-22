"""把现有 LLMProvider 适配成 :class:`~src.harness.loop.ToolCallingModel`。

## 为什么用提示词协议而不是原生 function calling

``src/llm/provider.py`` 目前没有 ``tools=`` 参数，而"给 LLM 的 JSON Schema"
（``src/mcp/tool_registry.py`` v2）实际从未被模型消费过——工具一直是 agent
代码直接调用的。

在这里直接改 provider 加原生 function calling 会动到核心 LLM 路径
（重试、降级链、流式），且**无法在离线环境端到端验证**。相比之下，
提示词协议完全复用现有的 ``chat()`` 与项目已有的 JSON 输出约定，
零 provider 改动，且解析逻辑可以被完整单测覆盖。

代价要说清楚：提示词协议依赖模型听话地输出 JSON，可靠性**低于**原生
function calling（模型可能返回自然语言、可能漏字段）。因此 :meth:`_parse`
把解析失败降级为"当作最终答案"而不是抛异常——宁可提前结束，也不要
在解析错误上空转到预算耗尽。

将来 provider 支持原生 function calling 时，只需另写一个实现同一协议的
adapter，:mod:`src.harness.loop` 一行都不用改。
"""

from __future__ import annotations

import json
import logging

from src.harness.loop import ModelTurn, ToolCall

logger = logging.getLogger(__name__)

#: 注入到对话最前面的工具调用协议说明。
TOOL_PROTOCOL = """\
你可以调用下列工具来获取完成任务所需的信息。

可用工具（JSON Schema）：
{tool_schemas}

严格按以下规则回复，**只输出一个 JSON 对象，不要输出任何其他文字**：

1. 需要调用工具时，输出：
   {{"tool_calls": [{{"name": "工具名", "args": {{...}}}}]}}
   一次可以请求多个工具。

2. 已经可以给出最终答案时，输出：
   {{"answer": "你的最终答案"}}

不要臆造工具名；只能用上面列出的工具。工具调用失败时，你会收到一条
包含错误信息的工具消息，请据此改用其他工具或调整参数，不要重复同样的调用。
"""


def render_tool_schemas(tools: list) -> str:
    """把工具列表渲染成协议说明里的 schema 段。

    参数缺失的工具（``name`` 为空）会被跳过——它们没法被模型调用，
    渲染出来只会浪费上下文。
    """
    cleaned = [t for t in tools if isinstance(t, dict) and t.get("name")]
    if not cleaned:
        return "（当前没有可用工具）"
    return json.dumps(cleaned, ensure_ascii=False, indent=2)


def _strip_code_fence(text: str) -> str:
    """去掉模型习惯性包裹的 ```json ... ``` 围栏。"""
    stripped = text.strip()
    if "```json" in stripped:
        return stripped.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in stripped:
        return stripped.split("```", 1)[1].split("```", 1)[0].strip()
    return stripped


class PromptToolCallingModel:
    """基于提示词协议的工具调用实现。

    只依赖一个 ``chat(messages, temperature=...) -> resp`` 接口，其中
    ``resp`` 需有 ``content`` 与 ``tokens`` 两个属性（与
    ``src/llm/provider.py`` 的 ``LLMResponse`` 一致）。因此测试可以注入
    任意假对象，完全离线。
    """

    def __init__(self, llm, temperature: float = 0.1):
        self.llm = llm
        self.temperature = temperature

    def complete(self, messages: list, tools: list) -> ModelTurn:
        """按协议请求模型决策，并解析成 :class:`ModelTurn`。"""
        resp = self.llm.chat(
            self._with_protocol(messages, tools), temperature=self.temperature
        )
        return self.parse_response(getattr(resp, "content", "") or "",
                                   tokens=getattr(resp, "tokens", 0) or 0)

    @staticmethod
    def _with_protocol(messages: list, tools: list) -> list:
        """在**副本**上注入协议说明，不修改调用方的消息列表。

        循环每轮都会调用 ``complete()``，而 messages 会不断增长——协议
        必须每轮重新注入在最前面，而不是追加到末尾（追加会被后续的
        工具结果淹没）。
        """
        protocol = TOOL_PROTOCOL.format(tool_schemas=render_tool_schemas(tools))
        injected = {"role": "system", "content": protocol}
        # 保留调用方原有的 system 消息（如伤病安全约束），放在协议之后。
        existing_system = [m for m in messages if m.get("role") == "system"]
        rest = [m for m in messages if m.get("role") != "system"]
        return [injected, *existing_system, *rest]

    def format_tool_result(self, call: ToolCall, payload) -> list[dict]:
        """把工具结果包装成模型能读懂的消息。

        **不能用 ``role="tool"``**：那是原生 function calling 的格式，
        OpenAI 兼容服务端要求它与前置 assistant 消息里的 ``tool_calls``
        严格配对；提示词协议下不存在 ``tool_calls``，发出去会被直接拒绝
        （实测 DeepSeek 返回 400: Messages with role 'tool' must be a
        response to a preceding message with 'tool_calls'）。

        这里用一条 ``role="user"`` 消息承载结果，并**显式复述调用了什么、
        传了什么参数**——因为对话里没有 assistant 的 tool_calls 记录，
        不复述的话模型会不知道自己上一轮请求了什么。
        """
        args = json.dumps(call.args, ensure_ascii=False) if call.args else "{}"
        body = str(payload)
        return [{
            "role": "user",
            "content": (
                f"[工具结果] 你刚才调用了 {call.name}({args})，返回如下：\n"
                f"{body}\n\n"
                "请据此继续：需要更多材料就再调用工具，材料够了就直接给出最终答案。"
            ),
        }]

    @staticmethod
    def parse_response(content: str, tokens: int = 0) -> ModelTurn:
        """把模型的一段文本解析成 :class:`ModelTurn`。

        解析策略（全部走"宁可提前结束"的保守分支）：
        * 命中 ``tool_calls`` 且非空 -> 请求调用工具；
        * 命中 ``answer``           -> 最终答案；
        * JSON 结构不对或解析失败   -> 把原文当作最终答案并告警。
          这一条是刻意的：解析失败的常见原因是模型直接用自然语言回答了，
          当成答案至少能正常结束；若当作错误重试，很可能在同一个坑里
          空转到预算耗尽。
        """
        text = (content or "").strip()
        if not text:
            return ModelTurn(content="", tokens=tokens)

        try:
            payload = json.loads(_strip_code_fence(text))
        except json.JSONDecodeError:
            logger.warning(
                "[harness.tool_calling] 模型未按协议返回 JSON，按最终答案处理：%s",
                text[:120],
            )
            return ModelTurn(content=text, tokens=tokens)

        if not isinstance(payload, dict):
            return ModelTurn(content=text, tokens=tokens)

        raw_calls = payload.get("tool_calls")
        if isinstance(raw_calls, list) and raw_calls:
            calls = _coerce_tool_calls(raw_calls)
            if calls:
                return ModelTurn(content=None, tool_calls=calls, tokens=tokens)
            # 所有调用项都格式非法而被丢弃。**不能**返回空 tool_calls——
            # 循环会把它当成"模型给出了最终答案"，从而返回空字符串并
            # 标记为 completed，把一次格式错误伪装成成功。回落成答案更安全。
            logger.warning(
                "[harness.tool_calling] tool_calls 全部格式非法，按最终答案处理：%s",
                raw_calls[:3],
            )
            return ModelTurn(content=text, tokens=tokens)

        if "answer" in payload:
            return ModelTurn(content=str(payload["answer"]), tokens=tokens)

        # 合法 JSON 但没有约定字段——多半是模型自创了格式，回落成答案。
        logger.warning(
            "[harness.tool_calling] JSON 缺少 tool_calls/answer 字段：%s",
            list(payload.keys())[:8],
        )
        return ModelTurn(content=text, tokens=tokens)


def _coerce_tool_calls(raw_calls: list) -> list[ToolCall]:
    """把模型给的调用项收敛成 :class:`ToolCall`，跳过不可用的条目。

    模型可能返回 ``"args": null`` 或把名字写成非字符串；这些都属于
    "格式不对但仍可挽救"，跳过比整条失败更划算。
    """
    calls: list[ToolCall] = []
    for idx, item in enumerate(raw_calls):
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        args = item.get("args")
        calls.append(
            ToolCall(
                name=name.strip(),
                args=args if isinstance(args, dict) else {},
                call_id=str(item.get("id") or f"call_{idx}_{name}"),
            )
        )
    return calls
