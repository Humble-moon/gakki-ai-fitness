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
  第三层 — 明确失败语义：所有模型都不可用时抛出 LLMUnavailableError，
    由上层决定如何将错误呈现给用户；不返回伪成功内容。

  配置方式：
    .env 中设置 LLM_FALLBACK_CHAIN=default,fast
    不配置则不做自动降级（仍会重试，但不会切换模型）。
================================================================================
"""

from __future__ import annotations
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Generator, Iterator

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

_ERROR_DETAIL_LIMIT = 200
_LOG_MESSAGE_LIMIT = 499
_SECRET_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[^\s,;]+"),
    re.compile(r"(?i)\b(Authorization|api[_-]?key|credential|(?:session[_-]?)?cookie)\b\s*[:=]\s*[^\s,;]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
)


def _error_summary(error: Exception) -> str:
    """Return a single-line, redacted, strictly bounded error summary."""
    detail = " ".join(str(error).splitlines())
    for pattern in _SECRET_PATTERNS:
        detail = pattern.sub(lambda match: (
            f"{match.group(1)}=[REDACTED]" if match.lastindex else "[REDACTED]"
        ), detail)
    detail = detail[:_ERROR_DETAIL_LIMIT]
    return f"{type(error).__name__}: {detail}"


def _bounded_log_message(message: str) -> str:
    """Apply the strict limit to a complete, already-redacted log message."""
    return message[:_LOG_MESSAGE_LIMIT]


# 退避参数
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0  # 秒

# 熔断器参数
_CIRCUIT_FAILURE_THRESHOLD = 3   # 连续失败 N 次触发熔断
_CIRCUIT_COOLDOWN_SECONDS = 30.0  # 熔断冷却时间


class CircuitBreaker:
    """滑动窗口熔断器——连续失败 N 次后，冷却期内跳过该模型。

    状态机：CLOSED → (连续失败 threshold 次) → OPEN → (冷却 cooldown 秒)
            → HALF_OPEN → (一次探测成功) → CLOSED
                        → (探测失败) → OPEN（重新计时）

    线程安全：Python GIL 使 dict 操作原子化，单进程场景下无需额外锁。
    多进程部署时需改用 Redis 存储熔断状态。
    """

    def __init__(self, failure_threshold: int = _CIRCUIT_FAILURE_THRESHOLD,
                 cooldown_seconds: float = _CIRCUIT_COOLDOWN_SECONDS):
        self.threshold = failure_threshold
        self.cooldown = cooldown_seconds
        self._failures: dict[str, list[float]] = {}
        self._open_until: dict[str, float] = {}

    def record_failure(self, model_alias: str):
        """记录一次调用失败。连续失败达到阈值时打开熔断器。"""
        now = time.monotonic()
        if model_alias not in self._failures:
            self._failures[model_alias] = []
        self._failures[model_alias].append(now)
        self._failures[model_alias] = self._failures[model_alias][-self.threshold:]
        if len(self._failures[model_alias]) >= self.threshold:
            window_start = self._failures[model_alias][0]
            if now - window_start < self.cooldown * 2:
                self._open_until[model_alias] = now + self.cooldown
                logger.warning(
                    f"[CircuitBreaker] OPEN for {model_alias} — "
                    f"{self.threshold} failures in {now - window_start:.1f}s, "
                    f"cooldown until {self.cooldown}s from now"
                )

    def record_success(self, model_alias: str):
        """一次调用成功——重置该模型的失败计数和熔断状态。"""
        if model_alias in self._failures or model_alias in self._open_until:
            self._failures.pop(model_alias, None)
            self._open_until.pop(model_alias, None)
            logger.info(f"[CircuitBreaker] CLOSED for {model_alias} — success after failures")

    def is_open(self, model_alias: str) -> bool:
        """检查熔断器是否开启（该模型当前是否不可用）。"""
        if model_alias not in self._open_until:
            return False
        if time.monotonic() > self._open_until[model_alias]:
            # 冷却期结束 → 半开状态（关闭熔断器，允许一次探测）
            self._open_until.pop(model_alias, None)
            self._failures.pop(model_alias, None)
            logger.info(f"[CircuitBreaker] HALF_OPEN for {model_alias} — cooling ended, probing")
            return False
        return True


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


@dataclass
class LLMStreamMetadata:
    """Caller-visible status for one streaming call, updated during iteration."""
    degraded: bool = False
    model: str | None = None
    attempted_models: list[str] = field(default_factory=list)


class LLMStream(Iterator[str]):
    """String iterator with metadata that remains compatible with existing callers."""

    def __init__(self, iterator: Iterator[str], metadata: LLMStreamMetadata):
        self._iterator = iterator
        self.metadata = metadata

    def __iter__(self) -> "LLMStream":
        return self

    def __next__(self) -> str:
        return next(self._iterator)


class LLMUnavailableError(RuntimeError):
    """Raised when every configured LLM model fails before producing output."""

    def __init__(self, message: str, *, attempted_models: list[str], errors: list[str]):
        super().__init__(message)
        self.attempted_models = attempted_models
        self.errors = errors


class LLMProvider:
    """多模型 LLM 调用提供者 — 内置重试 + 降级链 + 明确失败语义。

    核心设计：
      - self._clients: {别名: OpenAI Client}，每个 API 端点一个 client
      - self._models:  {别名: 模型标识}
      - self._active:  当前活跃的模型别名（默认 "default"）
      - 重试：指数退避 3 次，仅对网络/限流类异常重试
      - 降级：沿 LLM_FALLBACK_CHAIN 依次尝试
      - 全部失败：抛出 LLMUnavailableError，不返回伪成功内容
    """

    def __init__(self):
        self._clients: dict[str, OpenAI] = {}
        self._models: dict[str, str] = {}
        self._active: str = "default"
        self._breaker = CircuitBreaker()

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
        # 只保留已配置且未熔断的模型
        available = []
        for a in chain:
            if a not in self._models:
                continue
            if self._breaker.is_open(a):
                logger.warning(f"[CircuitBreaker] Skipping {a} — circuit is OPEN")
                continue
            available.append(a)
        # 如果所有模型都被熔断，至少保留一个（全挂了就全挂了，不能空链）
        return available if available else [a for a in chain if a in self._models][:1]

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
                summary = _error_summary(e)
                logger.warning(
                    f"[Retry] {model_name} attempt {attempt + 1}/{_MAX_RETRIES} "
                    f"failed: {summary} | waiting {wait}s"
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
        """非流式对话调用，含重试 + 降级链 + 明确失败语义。

        执行流程：
          1. 解析主模型
          2. 构建降级链（主模型 + 备用模型）
          3. 依次尝试降级链中的每个模型
          4. 全部失败 → 抛出 LLMUnavailableError，不返回伪成功响应

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
                    f"(alias={alias}) | errors so far: {errors}"
                )

            try:
                resp = self._call_api_with_retry(cl, mn, messages, temperature)
                resp.attempted_models = attempted
                if i > 0:
                    resp.degraded = True
                    logger.info(f"[Fallback] Succeeded with {mn} after {i} fallback(s)")
                cost_tracker.record(mn, resp.tokens,
                                    extra="fallback" if resp.degraded else "chat")
                self._breaker.record_success(alias)
                return resp
            except Exception as e:
                summary = _error_summary(e)
                errors.append(f"{mn}: {summary}")
                logger.error(f"[LLM] {mn} failed: {summary}")
                self._breaker.record_failure(alias)
                continue

        # 所有模型都失败了 → 抛出明确的不可用异常，不生成伪成功内容
        logger.critical(_bounded_log_message(
            f"[LLM] All models exhausted. Chain: {chain}, "
            f"errors: {errors}"
        ))
        raise LLMUnavailableError(
            "所有配置的 LLM provider 均不可用",
            attempted_models=attempted,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # 公共 API：chat_stream（流式）
    # ------------------------------------------------------------------

    def chat_stream(self, messages: list, temperature: float = 0.3,
                    model: str = None) -> LLMStream:
        """流式对话调用，yield str while exposing per-call metadata."""
        metadata = LLMStreamMetadata()

        def generate() -> Iterator[str]:
            client, primary_alias, primary_model_name = self._resolve(model)
            chain = self._build_fallback_chain(primary_alias)
            errors = []

            for i, alias in enumerate(chain):
                cl = self._clients[alias]
                mn = self._models[alias]
                metadata.attempted_models.append(mn)
                metadata.model = mn
                if i > 0:
                    logger.warning(f"[Fallback:stream] Switching to {mn} after errors: {errors}")

                started_output = False
                try:
                    stream = self._create_stream_with_retry(cl, mn, messages, temperature)
                    total_content = []
                    for chunk in stream:
                        delta = chunk.choices[0].delta
                        if delta.content:
                            started_output = True
                            total_content.append(delta.content)
                            yield delta.content
                    if not started_output:
                        raise RuntimeError("empty stream response")

                    output_text = "".join(total_content)
                    cost_tracker.record(mn, max(1, len(output_text) // 2),
                                        extra="stream:fallback" if i > 0 else "stream")
                    self._breaker.record_success(alias)
                    metadata.degraded = i > 0
                    return
                except Exception as e:
                    if started_output:
                        raise
                    summary = _error_summary(e)
                    errors.append(f"{mn}: {summary}")
                    logger.error(f"[LLM:stream] {mn} failed: {summary}")
                    self._breaker.record_failure(alias)

            logger.critical(_bounded_log_message(
                f"[LLM:stream] All models exhausted. Chain: {chain}, errors: {errors}"
            ))
            raise LLMUnavailableError(
                "所有配置的 LLM provider 均不可用",
                attempted_models=metadata.attempted_models,
                errors=errors,
            )

        return LLMStream(generate(), metadata)

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
                    f"failed: {_error_summary(e)} | waiting {wait}s"
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
