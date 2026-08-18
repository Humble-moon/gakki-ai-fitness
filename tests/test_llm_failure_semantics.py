import logging

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
    assert len(exc_info.value.errors) == 2
    assert exc_info.value.errors[0] == "primary: ConnectionError: offline fake"
    assert exc_info.value.errors[1] == "backup: ConnectionError: offline fake"


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
    assert exc_info.value.errors == [
        "primary: ConnectionError: offline fake",
        "backup: ConnectionError: offline fake",
    ]


def test_chat_bounds_and_flattens_error_diagnostics(provider, monkeypatch):
    def always_fail(*args, **kwargs):
        raise RuntimeError("first line\n" + "x" * 250)

    monkeypatch.setattr(provider, "_call_api_with_retry", always_fail)

    with pytest.raises(LLMUnavailableError) as exc_info:
        provider.chat([{"role": "user", "content": "固定脱敏测试问题"}])

    assert len(exc_info.value.errors) == 2
    for model, error in zip(["primary", "backup"], exc_info.value.errors):
        assert error.startswith(f"{model}: RuntimeError: first line ")
        assert "\n" not in error
        assert len(error) == len(model) + 2 + len("RuntimeError: ") + 200


def test_stream_falls_back_after_empty_chunks(provider, monkeypatch):
    calls = []

    class EmptyChunk:
        class Choices:
            class Delta:
                content = None
            delta = Delta()
        choices = [Choices()]

    class ContentChunk:
        class Choices:
            class Delta:
                content = "backup output"
            delta = Delta()
        choices = [Choices()]

    def empty_then_fail(client, model, messages, temperature):
        calls.append(model)
        if model == "primary":
            def failed_stream():
                yield EmptyChunk()
                raise ConnectionError("primary offline")
            return failed_stream()
        return iter([ContentChunk()])

    monkeypatch.setattr(provider, "_create_stream_with_retry", empty_then_fail)

    assert list(provider.chat_stream(
        [{"role": "user", "content": "固定脱敏测试问题"}]
    )) == ["backup output"]
    assert calls == ["primary", "backup"]




def _chunk(content=None):
    delta = type("Delta", (), {"content": content})()
    choice = type("Choice", (), {"delta": delta})()
    return type("Chunk", (), {"choices": [choice]})()


def _assert_secret_is_redacted(caplog, secret):
    assert secret not in caplog.text
    assert "[REDACTED]" in caplog.text
    assert all("\n" not in record.getMessage() for record in caplog.records)
    assert max(len(record.getMessage()) for record in caplog.records) < 500


def test_retry_logs_use_bounded_redacted_error_summary(provider, monkeypatch, caplog):
    secret = "sk-abcdefghijklmnopqrstuvwxyz012345"

    class FakeCompletions:
        def create(self, **kwargs):
            raise ConnectionError(f"Bearer {secret}\n" + "x" * 1000)

    client = type("Client", (), {
        "chat": type("Chat", (), {"completions": FakeCompletions()})()
    })()
    monkeypatch.setattr("src.llm.provider._MAX_RETRIES", 1)
    monkeypatch.setattr("src.llm.provider.time.sleep", lambda _: None)

    with caplog.at_level(logging.WARNING, logger="src.llm.provider"):
        with pytest.raises(ConnectionError):
            provider._call_api_with_retry(client, "primary", [], 0.1)

    _assert_secret_is_redacted(caplog, secret)


def test_chat_logs_use_bounded_redacted_error_summary(provider, monkeypatch, caplog):
    secret = "credential=top-secret-value"

    def always_fail(*args, **kwargs):
        raise RuntimeError(secret + "\n" + "y" * 1000)

    monkeypatch.setattr(provider, "_call_api_with_retry", always_fail)

    with caplog.at_level(logging.ERROR, logger="src.llm.provider"):
        with pytest.raises(LLMUnavailableError):
            provider.chat([{"role": "user", "content": "固定脱敏测试问题"}])

    _assert_secret_is_redacted(caplog, "top-secret-value")


def test_stream_logs_use_bounded_redacted_error_summary(provider, monkeypatch, caplog):
    secret = "session_cookie=top-secret-cookie"

    def always_fail(*args, **kwargs):
        raise RuntimeError(secret + "\n" + "z" * 1000)

    monkeypatch.setattr(provider, "_create_stream_with_retry", always_fail)

    with caplog.at_level(logging.ERROR, logger="src.llm.provider"):
        with pytest.raises(LLMUnavailableError):
            list(provider.chat_stream([{"role": "user", "content": "固定脱敏测试问题"}]))

    _assert_secret_is_redacted(caplog, "top-secret-cookie")


def test_stream_falls_back_when_primary_returns_zero_chunks(provider, monkeypatch):
    calls = []

    def zero_chunks_then_backup(client, model, messages, temperature):
        calls.append(model)
        if model == "primary":
            return iter(())
        return iter([_chunk("backup output")])

    monkeypatch.setattr(provider, "_create_stream_with_retry", zero_chunks_then_backup)

    stream = provider.chat_stream([{"role": "user", "content": "固定脱敏测试问题"}])
    assert list(stream) == ["backup output"]
    assert calls == ["primary", "backup"]


def test_stream_falls_back_after_metadata_only_and_empty_chunks(provider, monkeypatch):
    calls = []

    def empty_then_backup(client, model, messages, temperature):
        calls.append(model)
        if model == "primary":
            return iter([_chunk(None), _chunk("")])
        return iter([_chunk("backup output")])

    monkeypatch.setattr(provider, "_create_stream_with_retry", empty_then_backup)

    assert list(provider.chat_stream(
        [{"role": "user", "content": "固定脱敏测试问题"}]
    )) == ["backup output"]
    assert calls == ["primary", "backup"]


def test_stream_raises_when_all_models_end_without_text(provider, monkeypatch):
    monkeypatch.setattr(
        provider,
        "_create_stream_with_retry",
        lambda *args, **kwargs: iter([_chunk(None), _chunk("")]),
    )

    with pytest.raises(LLMUnavailableError) as exc_info:
        list(provider.chat_stream([{"role": "user", "content": "固定脱敏测试问题"}]))

    assert exc_info.value.attempted_models == ["primary", "backup"]
    assert all("empty stream response" in error for error in exc_info.value.errors)


def test_stream_exposes_fallback_metadata_without_changing_string_yields(provider, monkeypatch):
    def primary_empty_then_backup(client, model, messages, temperature):
        if model == "primary":
            return iter(())
        return iter([_chunk("backup output")])

    monkeypatch.setattr(provider, "_create_stream_with_retry", primary_empty_then_backup)

    stream = provider.chat_stream([{"role": "user", "content": "固定脱敏测试问题"}])
    assert list(stream) == ["backup output"]
    assert stream.metadata.degraded is True
    assert stream.metadata.model == "backup"
    assert stream.metadata.attempted_models == ["primary", "backup"]
