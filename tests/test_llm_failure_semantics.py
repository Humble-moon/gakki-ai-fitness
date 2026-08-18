import pytest

from src.llm.provider import LLMProvider, LLMResponse, LLMUnavailableError


@pytest.fixture
def provider(monkeypatch):
    provider = LLMProvider.__new__(LLMProvider)
    provider._clients = {"default": object(), "fallback": object()}
    provider._models = {"default": "primary", "fallback": "backup"}
    provider._active = "default"
    provider._breaker = type("Breaker", (), {
        "is_open": lambda self, alias: False,
        "record_failure": lambda self, alias: None,
        "record_success": lambda self, alias: None,
    })()
    monkeypatch.setattr("src.llm.provider.LLM_FALLBACK_CHAIN", ["default", "fallback"])
    return provider


def test_chat_raises_when_all_models_fail(provider, monkeypatch):
    def always_fail(*args, **kwargs):
        raise ConnectionError("offline fake")

    monkeypatch.setattr(provider, "_call_api_with_retry", always_fail)

    with pytest.raises(LLMUnavailableError) as exc_info:
        provider.chat([{"role": "user", "content": "固定脱敏测试问题"}])

    assert exc_info.value.attempted_models == ["primary", "backup"]
    assert exc_info.value.errors
    assert "offline fake" in exc_info.value.errors[0]


def test_chat_marks_fallback_success_as_degraded(provider, monkeypatch):
    calls = []

    def fail_primary_then_success(client, model, messages, temperature):
        calls.append(model)
        if model == "primary":
            raise ConnectionError("primary offline")
        return LLMResponse(content="fake answer", model=model, tokens=2)

    monkeypatch.setattr(provider, "_call_api_with_retry", fail_primary_then_success)

    response = provider.chat([{"role": "user", "content": "固定脱敏测试问题"}])

    assert calls == ["primary", "backup"]
    assert response.content == "fake answer"
    assert response.degraded is True
    assert response.attempted_models == ["primary", "backup"]


def test_stream_raises_when_connection_cannot_start(provider, monkeypatch):
    def always_fail(*args, **kwargs):
        raise ConnectionError("offline fake")

    monkeypatch.setattr(provider, "_create_stream_with_retry", always_fail)

    with pytest.raises(LLMUnavailableError) as exc_info:
        list(provider.chat_stream([{"role": "user", "content": "固定脱敏测试问题"}]))

    assert exc_info.value.attempted_models == ["primary", "backup"]
    assert exc_info.value.errors


def test_stream_does_not_fallback_after_output_started(provider, monkeypatch):
    class Chunk:
        class Choices:
            class Delta:
                content = "partial"
            delta = Delta()
        choices = [Choices()]

    def stream_then_fail(*args, **kwargs):
        yield Chunk()
        raise ConnectionError("mid-stream offline")

    monkeypatch.setattr(provider, "_create_stream_with_retry", stream_then_fail)

    stream = provider.chat_stream([{"role": "user", "content": "固定脱敏测试问题"}])
    assert next(stream) == "partial"
    with pytest.raises(ConnectionError, match="mid-stream offline"):
        next(stream)
