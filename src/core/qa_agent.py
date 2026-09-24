"""知识问答的自主循环路径 —— 让模型自己决定查什么。

## 与固定流水线的差别

`Orchestrator.answer_question_stream`（默认路径）每次都走完整五步：
伤病检测 → 知识库检索 → 动作库检索 → 上下文注入 → 生成答案。
其中**知识库与动作库是必然执行的**——哪怕问题是"增肌要练多久"，
也会去查一遍动作库。

本路径把"查什么"交还给模型：它看到问题后自己决定调 `search_knowledge`、
`search_exercises` 还是 `reason_injury`，调几次也由它定。理论上更省、
也更灵活（比如它可以先查、发现不相关、换个词再查）。

代价是**不确定性**：模型可能选错工具、可能多查、可能不查就直接答。
这正是需要 ablation 量化的问题——见 `eval/compare_qa_paths.py`。

## 安全约束不打折

安全检测与安全提示词一律来自 `src/core/qa_safety.py`，与固定流水线
**同一份实现**。这是硬性要求：两条路径若各存一份安全逻辑，
迟早会漂移，而用户看不出自己走的是哪条。

## 默认关闭

由 `HARNESS_AGENT_LOOP` 控制（见 `harness.loop.agent_loop_enabled`）。
固定流水线仍是默认路径——它在延迟上更可预测，且是论文基线的评测对象。
"""

from __future__ import annotations

import logging

from src.core.qa_safety import (
    EMERGENCY_NOTE,
    SAFETY_NOTE,
    build_emergency_messages,
    build_safety_messages,
    detect_emergency,
    detect_safety_concern,
)
from src.harness.loop import LoopResult, ToolInvoker, run_agent_loop
from src.harness.tool_calling import PromptToolCallingModel

logger = logging.getLogger(__name__)

#: 自主循环路径的基础人设。安全约束另行追加（见 build_qa_messages）。
_BASE_SYSTEM = (
    "你是资深健身教练和运动康复专家。你会先用工具查证，再基于查到的材料作答。\n\n"
    "工作方式：\n"
    "1. 判断这个问题需要哪些材料——是概念解释（查知识库）、"
    "具体动作（查动作库），还是伤病相关（查图谱）；\n"
    "2. 调用相应工具获取材料。工具返回不相关时可以换个说法再查一次；\n"
    "3. 材料足够后给出最终答案，不要再无谓地调用工具。\n\n"
    "回答要求：\n"
    "- 先直接给结论，再解释原因，最后给 2-3 条可执行建议\n"
    "- 用自己的话自然回答，不要罗列工具返回的原始内容\n"
    "- 200-350 字，口语化，像教练在聊天\n"
    "- 纯文字段落，不用 markdown\n"
    "- 如果查到的材料不足以回答，就诚实说明，不要编造"
)


def build_qa_messages(question: str, profile: dict) -> list[dict]:
    """组装自主循环的初始消息。

    安全消息**置于最前**：`PromptToolCallingModel` 注入工具协议时会
    保留调用方原有的 system 消息（见其 `_with_protocol`），因此这里
    只需保证顺序正确。
    """
    injuries = profile.get("injuries", [])
    # 急症优先判定：命中急症时**不叠加**普通安全话术。两种规则对"能否给训练
    # 建议"的要求相反——安全规则允许运动康复建议，急症规则禁止任何训练指导，
    # 同时注入会让模型收到自相矛盾的约束。
    is_emergency = detect_emergency(question)
    needs_safety = False if is_emergency else detect_safety_concern(
        question, injuries=injuries
    )

    messages: list[dict] = []

    base = _BASE_SYSTEM
    if is_emergency:
        base += "\n" + EMERGENCY_NOTE
    elif needs_safety:
        base += "\n" + SAFETY_NOTE
    messages.append({"role": "system", "content": base})

    if is_emergency:
        # 硬约束单独再放一条 system——比混在人设里更难被后续指令覆盖。
        messages.extend(build_emergency_messages())
    elif needs_safety:
        messages.extend(build_safety_messages())

    profile_line = (
        f"用户情况：{profile.get('height')}cm, {profile.get('weight')}kg, "
        f"训练{profile.get('training_years', 1)}年\n"
        f"伤病：{injuries}"
    )
    messages.append({"role": "user", "content": f"{profile_line}\n\n用户问题：{question}"})
    return messages


def answer_with_agent_loop(
    llm,
    tools: ToolInvoker,
    question: str,
    profile: dict,
    *,
    temperature: float = 0.3,
    max_steps: int | None = None,
    token_budget: int | None = None,
    on_step=None,
) -> tuple[str, LoopResult]:
    """用自主循环回答一个问题。

    Returns:
        ``(answer, loop_result)``。**调用方必须检查 ``loop_result.completed``**：

        * True  —— ``answer`` 是模型的最终答案；
        * False —— 触达预算上限，模型没给出答案，``answer`` 为空串。
          此时应回退到固定流水线，而不是把空串当答案返回给用户。

        返回 ``LoopResult`` 而非只返回文本，就是为了让调用方**有能力**
        做这个判断——把失败伪装成空答案是最糟的接口设计。
    """
    model = PromptToolCallingModel(llm, temperature=temperature)
    messages = build_qa_messages(question, profile)

    kwargs: dict = {"on_step": on_step}
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    if token_budget is not None:
        kwargs["token_budget"] = token_budget

    result = run_agent_loop(model, tools, messages, **kwargs)

    if not result.completed:
        logger.warning(
            "[qa_agent] 自主循环未产出答案（%s，%d 步，%d token）",
            result.exited_reason, result.steps, result.tokens_used,
        )
    return (result.content if result.completed else ""), result
