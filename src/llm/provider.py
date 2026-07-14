"""
================================================================================
文件角色：LLM 调用统一入口（Provider 层）—— 多模型支持 + 弹性降级
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

弹性降级（Resilience）：
  三层防护，面试金句："主链路挂了，200ms 内自动切到备用模型，精度降 5%
  但不中断服务。"

  第一层 — 重试（Retry）：同模型指数退避重试 3 次（1s→2s→4s），
    应对瞬时网络抖动、API 限流（429）。
  第二层 — 降级（Fallback）：重试耗尽后，按 LLM_FALLBACK_CHAIN 顺序
    切换到备用模型（如 deepseek-chat → qwen-turbo）。
  第三层 — 兜底（Graceful Degradation）：所有模型都不可用时，不抛异常，
    返回带 degraded=True 标记的降级响应，由 Orchestrator 层返回给前端。

  配置方式：
    .env 中设置 LLM_FALLBACK_CHAIN=default,fast
    不配置则不做自动降级（仍会重试，但不会切换模型）。
================================================================================
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import Generator

from openai import OpenAI, APIStatusError, APIConnectionError, RateLimitError, APITimeoutError

from src.config import (
    LLM_CONFIGS, LLM_DEFAULT_MODEL, LLM_FALLBACK_CHAIN,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL,
)
from src.llm.cost_tracker import cost_tracker

logger = logging.getLogger(__name__)

# 可重试的异常类型 — 瞬时故障，重试大概率恢复
_RETRYABLE = (APIConnectionError, RateLimitError, APITimeoutError,
              TimeoutError, ConnectionError, OSError)

# 退避参数
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0  # 秒


@dataclass
class LLMResponse:
    """LLM 调用的统一返回值结构。"""
    content: str
    model: str
    tokens: int
    # 降级标记：True 表示本次调用经过了模型降级（主模型挂了，用的备用模型）
    degraded: bool = False
    # 记录实际经过的模型链路，方便排查
    attempted_models: list[str] = field(default_factory=list)


class LLMProvider:
    """多模型 LLM 调用提供者 — 内置重试 + 降级链 + 兜底。

    核心设计：
      - self._clients: {别名: OpenAI Client}，每个 API 端点一个 client
      - self._models:  {别名: 模型标识}
      - self._active:  当前活跃的模型别名（默认 "default"）
      - 重试：指数退避 3 次，仅对网络/限流类异常重试
      - 降级：沿 LLM_FALLBACK_CHAIN 依次尝试
      - 兜底：所有模型都失败时返回降级响应而非抛异常
    """

    def __init__(self):
        self._clients: dict[str, OpenAI] = {}
        self._models: dict[str, str] = {}
        self._active: str = "default"

        for alias, cfg in LLM_CONFIGS.items():
            self._clients[alias] = OpenAI(
                api_key=cfg["api_key"],
                base_url=cfg["base_url"],
                max_retries=0,  # 关闭 SDK 内置重试，由我们的 _call_api_with_retry 统一管理
            )
            self._models[alias] = cfg["model"]

        # 兜底：如果完全没有配置，用旧的 DEEPSEEK_ 变量
        if not self._clients:
            self._clients["default"] = OpenAI(
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL,
                max_retries=0,
            )
            self._models["default"] = "deepseek-chat"

    # ------------------------------------------------------------------
    # 公共属性
    # ------------------------------------------------------------------

    @property
    def available_models(self) -> dict[str, str]:
        """返回所有可用模型：{别名: 模型标识}。"""
        return dict(self._models)

    @property
    def active_model(self) -> str:
        """当前活跃的模型别名。"""
        return self._active

    def with_model(self, alias: str) -> "LLMProvider":
        """切换活跃模型（链式调用）。"""
        if alias not in self._models:
            available = ", ".join(self._models.keys())
            raise ValueError(f"未知模型别名 '{alias}'，可用: {available}")
        self._active = alias
        return self

    def _resolve(self, model: str | None = None) -> tuple[OpenAI, str, str]:
        """解析模型参数 → (OpenAI client, 别名, 实际模型名)。"""
        if model is not None and model in self._models:
            alias = model
            return self._clients[alias], alias, self._models[alias]
        elif model is not None:
            return self._clients[self._active], self._active, model
        else:
            return self._clients[self._active], self._active, self._models[self._active]

    # ------------------------------------------------------------------
    # 降级链
    # ------------------------------------------------------------------

    def _build_fallback_chain(self, primary_alias: str) -> list[str]:
        """构建降级链：主模型在前，备用模型按 LLM_FALLBACK_CHAIN 排在后。

        例：primary="default", LLM_FALLBACK_CHAIN=["default","fast"]
          → ["default", "fast"]（default 是主模型，fast 是备用）

        如果 primary 不在 FALLBACK_CHAIN 中，将 primary 插入最前面。
        """
        chain = list(LLM_FALLBACK_CHAIN) if LLM_FALLBACK_CHAIN else [primary_alias]
        if primary_alias not in chain:
            chain.insert(0, primary_alias)
        # 只保留已配置的模型
        return [a for a in chain if a in self._models]

    # ------------------------------------------------------------------
    # 内部：单次 API 调用（含重试）
    # ------------------------------------------------------------------

    def _call_api_with_retry(self, client: OpenAI, model_name: str,
                             messages: list, temperature: float) -> LLMResponse:
        """向单个模型发请求，带指数退避重试。

        只对网络/限流类瞬时故障重试。逻辑错误（如 API key 无效）
        直接向上抛，不浪费重试次数。
        """
        last_error = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                )
                tokens = resp.usage.total_tokens if resp.usage else 0
                return LLMResponse(
                    content=resp.choices[0].message.content,
                    model=resp.model,
                    tokens=tokens,
                )
            except _RETRYABLE as e:
                last_error = e
                wait = _BACKOFF_BASE ** attempt
                logger.warning(
                    f"[Retry] {model_name} attempt {attempt + 1}/{_MAX_RETRIES} "
                    f"failed: {type(e).__name__}: {e} | waiting {wait}s"
                )
                time.sleep(wait)
            # 非可重试异常（如 401 认证失败、400 参数错误）直接抛出
            except Exception:
                raise

        raise last_error  # type: ignore[misc]

    # ------------------------------------------------------------------
    # 公共 API：chat（非流式）
    # ------------------------------------------------------------------

    def chat(self, messages: list, temperature: float = 0.3,
             model: str = None) -> LLMResponse:
        """非流式对话调用，含重试 + 降级链 + 兜底。

        执行流程：
          1. 解析主模型
          2. 构建降级链（主模型 + 备用模型）
          3. 依次尝试降级链中的每个模型
          4. 全部失败 → 返回降级响应（degraded=True），不抛异常

        Args:
            messages: OpenAI 格式消息列表
            temperature: 生成随机性
            model: 模型别名 / 完整模型名 / None(用活跃模型)
        """
        client, primary_alias, primary_model_name = self._resolve(model)
        chain = self._build_fallback_chain(primary_alias)
        attempted = []
        errors = []

        for i, alias in enumerate(chain):
            cl = self._clients[alias]
            mn = self._models[alias]
            attempted.append(mn)

            if i > 0:
                logger.warning(
                    f"[Fallback] Primary model failed, switching to {mn} "
                    f"(alias={alias}) | errors so far: {[str(e) for e in errors]}"
                )

            try:
                resp = self._call_api_with_retry(cl, mn, messages, temperature)
                resp.attempted_models = attempted
                if i > 0:
                    resp.degraded = True
                    logger.info(f"[Fallback] Succeeded with {mn} after {i} fallback(s)")
                cost_tracker.record(mn, resp.tokens,
                                    extra="fallback" if resp.degraded else "chat")
                return resp
            except Exception as e:
                errors.append(f"{mn}: {type(e).__name__}: {e}")
                logger.error(f"[LLM] {mn} failed: {type(e).__name__}: {e}")
                continue

        # 所有模型都失败了 → 兜底降级响应
        logger.critical(
            f"[LLM] All models exhausted. Chain: {chain}, "
            f"errors: {errors}"
        )
        return LLMResponse(
            content="抱歉，AI 服务暂时不可用，请稍后重试。",
            model="none",
            tokens=0,
            degraded=True,
            attempted_models=attempted,
        )

    # ------------------------------------------------------------------
    # 公共 API：chat_stream（流式）
    # ------------------------------------------------------------------

    def chat_stream(self, messages: list, temperature: float = 0.3,
                    model: str = None) -> Generator[str, None, None]:
        """流式对话调用，含重试 + 降级链。

        注意：流式场景下，如果已经开始输出 token 后中断，无法回退重来。
        因此重试/降级只作用于连接建立阶段（create() 调用），流式输出
        过程中的异常直接向上传播，由 Orchestrator 层处理。

        Args:
            messages: OpenAI 格式消息列表
            temperature: 生成随机性
            model: 模型别名 / 完整模型名 / None
        """
        client, primary_alias, primary_model_name = self._resolve(model)
        chain = self._build_fallback_chain(primary_alias)
        attempted = []
        errors = []

        for i, alias in enumerate(chain):
            cl = self._clients[alias]
            mn = self._models[alias]
            attempted.append(mn)

            if i > 0:
                logger.warning(
                    f"[Fallback:stream] Switching to {mn} after errors: "
                    f"{[str(e) for e in errors]}"
                )

            try:
                # 连接建立阶段含重试
                stream = self._create_stream_with_retry(cl, mn, messages, temperature)
                total_content = []
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        total_content.append(delta.content)
                        yield delta.content

                # 流式成功完成
                output_text = "".join(total_content)
                estimated_tokens = max(1, len(output_text) // 2)
                tag = "stream:fallback" if i > 0 else "stream"
                cost_tracker.record(mn, estimated_tokens, extra=tag)
                return

            except Exception as e:
                errors.append(f"{mn}: {type(e).__name__}: {e}")
                logger.error(f"[LLM:stream] {mn} failed: {type(e).__name__}: {e}")
                continue

        # 所有模型都失败 → yield 降级消息
        logger.critical(
            f"[LLM:stream] All models exhausted. Chain: {chain}, "
            f"errors: {errors}"
        )
        yield "\n\n[AI 服务暂时不可用，请稍后重试]"

    def _create_stream_with_retry(self, client: OpenAI, model_name: str,
                                  messages: list, temperature: float):
        """创建流式连接，含指数退避重试。"""
        last_error = None
        for attempt in range(_MAX_RETRIES):
            try:
                return client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    stream=True,
                )
            except _RETRYABLE as e:
                last_error = e
                wait = _BACKOFF_BASE ** attempt
                logger.warning(
                    f"[Retry:stream] {model_name} attempt {attempt + 1}/{_MAX_RETRIES} "
                    f"failed: {type(e).__name__} | waiting {wait}s"
                )
                time.sleep(wait)
            except Exception:
                raise
        raise last_error  # type: ignore[misc]

    # ------------------------------------------------------------------
    # 公共 API：chat_with_json_mode
    # ------------------------------------------------------------------

    def chat_with_json_mode(self, messages: list,
                             model: str = None) -> dict:
        """获取 JSON 结构化输出。继承 chat() 的降级能力。

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
            result = json.loads(content.strip())
            # 把降级信息透传出去
            if resp.degraded:
                result["_degraded"] = True
            return result
        except json.JSONDecodeError:
            return {"raw": resp.content, "_degraded": resp.degraded}
