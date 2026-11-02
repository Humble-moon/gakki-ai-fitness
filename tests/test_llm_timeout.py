"""LLM 调用的超时与总预算。

这些用例守着三条容易被改回去的边界：
1. 客户端必须带 timeout——否则上游不返回时请求会无限挂着；
2. 重试与降级共享同一份墙钟预算——否则"3 次重试 × 多个模型"会叠加成数分钟；
3. 最后一次失败后不能再 sleep——原实现会白等 4 秒。
"""

import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.config import LLM_REQUEST_TIMEOUT, LLM_TOTAL_BUDGET
from src.llm.provider import LLMProvider, LLMUnavailableError


class _FailingClient:
    """chat.completions.create 永远抛可重试异常，并记录被调次数。"""

    def __init__(self):
        self.calls = 0
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        raise ConnectionError("offline fake")


def _bare_provider(aliases=("default", "fallback")):
    """不走 __init__（避免真实 API key 与网络客户端），只装配调用所需字段。"""
    provider = LLMProvider.__new__(LLMProvider)
    provider._clients = {a: object() for a in aliases}
    provider._models = {a: f"model-{a}" for a in aliases}
    provider._active = "default"
    provider._breaker = SimpleNamespace(
        is_open=lambda alias: False,
        record_failure=lambda alias: None,
        record_success=lambda alias: None,
    )
    return provider


def test_clients_are_constructed_with_timeout(monkeypatch):
    monkeypatch.setattr("src.llm.provider.LLM_CONFIGS", {
        "default": {
            "api_key": "sk-test-only",
            "base_url": "https://example.invalid/v1",
            "model": "fake-model",
        },
    })
    provider = LLMProvider()
    assert float(provider._clients["default"].timeout) == pytest.approx(
        LLM_REQUEST_TIMEOUT)


def test_last_attempt_does_not_sleep(monkeypatch):
    provider = _bare_provider()
    client = _FailingClient()
    sleeps = []
    monkeypatch.setattr("src.llm.provider.time.sleep", lambda s: sleeps.append(s))

    with pytest.raises(ConnectionError):
        provider._call_api_with_retry(client, "m", [], 0.1)

    assert client.calls == 3
    # 三次尝试之间只等两次；第三次失败后没有下一次尝试，等待没有意义。
    assert sleeps == [1.0, 2.0]


def test_expired_deadline_skips_all_attempts(monkeypatch):
    provider = _bare_provider()
    client = _FailingClient()

    with pytest.raises(TimeoutError):
        provider._call_api_with_retry(
            client, "m", [], 0.1, deadline=time.monotonic() - 1)

    assert client.calls == 0


def test_no_sleep_when_remaining_budget_is_shorter_than_backoff(monkeypatch):
    provider = _bare_provider()
    client = _FailingClient()
    sleeps = []
    monkeypatch.setattr("src.llm.provider.time.sleep", lambda s: sleeps.append(s))

    with pytest.raises(ConnectionError):
        provider._call_api_with_retry(
            client, "m", [], 0.1, deadline=time.monotonic() + 0.5)

    # 首轮退避 1s，已经超过剩余预算，等待只会让请求更慢。
    assert client.calls == 1
    assert sleeps == []


def test_fallback_chain_shares_one_deadline(monkeypatch):
    provider = _bare_provider()
    monkeypatch.setattr("src.llm.provider.LLM_FALLBACK_CHAIN",
                        ["default", "fallback"])
    seen = []

    def fail(client, model, messages, temperature, deadline=None):
        seen.append(deadline)
        raise ConnectionError("offline fake")

    monkeypatch.setattr(provider, "_call_api_with_retry", fail)

    with pytest.raises(LLMUnavailableError):
        provider.chat([{"role": "user", "content": "固定测试问题"}])

    assert len(seen) == 2
    assert seen[0] is not None and seen[0] == seen[1]


def test_chat_stops_when_total_budget_already_spent(monkeypatch):
    provider = _bare_provider()
    monkeypatch.setattr("src.llm.provider.LLM_FALLBACK_CHAIN",
                        ["default", "fallback"])
    monkeypatch.setattr("src.llm.provider.LLM_TOTAL_BUDGET", 0)
    calls = []

    def should_not_run(*args, **kwargs):
        calls.append(args)
        raise AssertionError("预算耗尽后不应再尝试任何模型")

    monkeypatch.setattr(provider, "_call_api_with_retry", should_not_run)

    with pytest.raises(LLMUnavailableError):
        provider.chat([{"role": "user", "content": "固定测试问题"}])

    assert calls == []


def test_stream_connect_retry_respects_budget(monkeypatch):
    provider = _bare_provider()
    client = _FailingClient()
    sleeps = []
    monkeypatch.setattr("src.llm.provider.time.sleep", lambda s: sleeps.append(s))

    with pytest.raises(TimeoutError):
        provider._create_stream_with_retry(
            client, "m", [], 0.1, deadline=time.monotonic() - 1)

    assert client.calls == 0
    assert sleeps == []


def test_total_budget_default_leaves_room_for_real_traffic():
    # 实测计划生成端到端 47s；预算必须明显大于它，否则正常请求会被误杀。
    assert LLM_TOTAL_BUDGET >= 120
    assert LLM_REQUEST_TIMEOUT >= 30


def test_review_artifact_expires_after_ttl(monkeypatch):
    from src.hitl.review_storage import is_review_expired, review_expires_at

    monkeypatch.setattr(
        "src.hitl.review_storage.REVIEW_ARTIFACT_TTL_SECONDS", 3600)
    old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    fresh = datetime.now(timezone.utc).isoformat()

    assert is_review_expired(SimpleNamespace(created_at=old)) is True
    assert is_review_expired(SimpleNamespace(created_at=fresh)) is False
    assert review_expires_at(fresh) > datetime.now(timezone.utc)


def test_review_expiry_tolerates_unparseable_timestamp():
    from src.hitl.review_storage import is_review_expired

    # 无法解析的时间戳不能把用户挡在门外。
    assert is_review_expired(SimpleNamespace(created_at="not-a-timestamp")) is False
