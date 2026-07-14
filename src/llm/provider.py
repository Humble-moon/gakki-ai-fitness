"""
================================================================================
文件角色：LLM 调用统一入口（Provider 层）—— 多模型支持
================================================================================
- 被调用者：所有需要调用大语言模型的模块（planner / writer / fact_checker /
  retriever / conversation 摘要等）都通过本模块的 LLMProvider 发出请求。
- 调用者：本模块封装 OpenAI 兼容的客户端，支持同时管理多个模型提供商，
  对外暴露 chat() / chat_stream() / chat_with_json_mode() 三个核心方法。
- 项目角色：底层基础设施层，屏蔽了 API 密钥、base_url、模型名等配置细节，
  让上层业务模块不用关心"用的是哪个模型、怎么连"。

多模型支持：
  通过 .env 配置多个模型别名，每个别名可以有独立的 API 端点：
    LLM_DEFAULT_MODEL=deepseek-chat      # 默认模型
    LLM_DEFAULT_BASE_URL=https://api.deepseek.com
    LLM_DEFAULT_API_KEY=sk-xxx
    LLM_REASONER_MODEL=deepseek-reasoner # 推理模型（可选）
    LLM_REASONER_BASE_URL=https://api.deepseek.com
    LLM_REASONER_API_KEY=sk-xxx

  使用方式：
    llm.chat(messages)                    → 用默认模型
    llm.chat(messages, model="reasoner")  → 用推理模型
    llm.chat(messages, model="fast")      → 用快速模型
    llm.with_model("reasoner").chat(...)  → 链式切换
================================================================================
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Generator
from openai import OpenAI
from src.config import LLM_CONFIGS, LLM_DEFAULT_MODEL, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
from src.llm.cost_tracker import cost_tracker

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """LLM 调用的统一返回值结构。"""
    content: str
    model: str
    tokens: int


class LLMProvider:
    """多模型 LLM 调用提供者。

    核心设计：
      - self._clients: {别名: OpenAI Client}，每个 API 端点一个 client
      - self._models:  {别名: 模型标识}，如 {"default": "deepseek-chat", "reasoner": "deepseek-reasoner"}
      - self._active:  当前活跃的模型别名（默认 "default"）
      - chat() / chat_stream() / chat_with_json_mode() 的 model 参数可以是：
          1. 别名（"default", "reasoner", "fast"）→ 自动切换 client + 模型
          2. 完整模型名（"deepseek-chat"）→ 在默认 client 上用指定模型
          3. None → 使用当前活跃模型
    """

    def __init__(self):
        self._clients: dict[str, OpenAI] = {}
        self._models: dict[str, str] = {}
        self._active: str = "default"

        for alias, cfg in LLM_CONFIGS.items():
            self._clients[alias] = OpenAI(
                api_key=cfg["api_key"],
                base_url=cfg["base_url"],
            )
            self._models[alias] = cfg["model"]

        # 兜底：如果完全没有配置，用旧的 DEEPSEEK_ 变量
        if not self._clients:
            self._clients["default"] = OpenAI(
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL,
            )
            self._models["default"] = "deepseek-chat"

    @property
    def available_models(self) -> dict[str, str]:
        """返回所有可用模型：{别名: 模型标识}。"""
        return dict(self._models)

    @property
    def active_model(self) -> str:
        """当前活跃的模型别名。"""
        return self._active

    def with_model(self, alias: str) -> "LLMProvider":
        """切换活跃模型（链式调用）。

        llm.with_model("reasoner").chat(messages)
        """
        if alias not in self._models:
            available = ", ".join(self._models.keys())
            raise ValueError(f"未知模型别名 '{alias}'，可用: {available}")
        self._active = alias
        return self

    def _resolve(self, model: str | None = None) -> tuple[OpenAI, str]:
        """解析模型参数 → (OpenAI client, 实际模型名)。

        解析逻辑：
          1. model 是别名 → 切换到对应 client + 模型
          2. model 是完整模型名（如 "deepseek-chat"）→ 在默认 client 上用该模型
          3. model 是 None → 使用当前活跃的 client + 模型
        """
        if model is not None and model in self._models:
            # 别名
            alias = model
            return self._clients[alias], self._models[alias]
        elif model is not None:
            # 完整模型名 → 在默认 client 上使用
            return self._clients[self._active], model
        else:
            # 使用活跃模型
            return self._clients[self._active], self._models[self._active]

    # =====================================================================
    # 核心 API：chat / chat_stream / chat_with_json_mode
    # =====================================================================

    def chat(self, messages: list, temperature: float = 0.3,
             model: str = None) -> LLMResponse:
        """非流式对话调用。

        Args:
            messages: OpenAI 格式消息列表
            temperature: 生成随机性 (0=确定, 1=发散)
            model: 模型别名 / 完整模型名 / None(用活跃模型)
        """
        client, actual_model = self._resolve(model)
        resp = client.chat.completions.create(
            model=actual_model,
            messages=messages,
            temperature=temperature,
        )
        tokens = resp.usage.total_tokens if resp.usage else 0
        cost_tracker.record(actual_model, tokens, extra="chat")
        return LLMResponse(
            content=resp.choices[0].message.content,
            model=resp.model,
            tokens=tokens,
        )

    def chat_stream(self, messages: list, temperature: float = 0.3,
                    model: str = None) -> Generator[str, None, None]:
        """流式对话调用，逐 token yield。

        Args:
            messages: OpenAI 格式消息列表
            temperature: 生成随机性
            model: 模型别名 / 完整模型名 / None
        """
        client, actual_model = self._resolve(model)
        stream = client.chat.completions.create(
            model=actual_model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        total_content = []
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                total_content.append(delta.content)
                yield delta.content
        # 流式完成后的成本估算（~1.5 字符/token for 中英文混合）
        output_text = "".join(total_content)
        estimated_tokens = max(1, len(output_text) // 2)
        cost_tracker.record(actual_model, estimated_tokens, extra="stream")

    def chat_with_json_mode(self, messages: list,
                             model: str = None) -> dict:
        """获取 JSON 结构化输出。

        Args:
            messages: OpenAI 格式消息列表（prompt 中需明确要求返回 JSON）
            model: 模型别名 / 完整模型名 / None
        """
        import json
        resp = self.chat(messages, temperature=0.1, model=model)
        try:
            content = resp.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return json.loads(content.strip())
        except json.JSONDecodeError:
            return {"raw": resp.content}
