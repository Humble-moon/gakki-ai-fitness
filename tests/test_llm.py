import pytest

from src.llm.provider import LLMProvider, LLMResponse


class TestLLMResponse:
    def test_response_model(self):
        resp = LLMResponse(content="测试回复", model="deepseek-v3", tokens=100)
        assert resp.content == "测试回复"
        assert resp.model == "deepseek-v3"


class TestLLMProvider:
    def test_chat_sync(self, monkeypatch):
        provider = LLMProvider.__new__(LLMProvider)
        provider._clients = {"default": object()}
        provider._models = {"default": "fake-model"}
        provider._active = "default"
        provider._breaker = type("Breaker", (), {
            "is_open": lambda self, alias: False,
            "record_failure": lambda self, alias: None,
            "record_success": lambda self, alias: None,
        })()
        monkeypatch.setattr("src.llm.provider.LLM_FALLBACK_CHAIN", ["default"])
        monkeypatch.setattr(
            provider,
            "_call_api_with_retry",
            lambda *args, **kwargs: LLMResponse(
                content="fake response", model="fake-model", tokens=2
            ),
        )

        resp = provider.chat(
            messages=[{"role": "user", "content": "固定脱敏测试问题"}],
            temperature=0.1,
        )
        assert resp.content == "fake response"
        assert resp.model == "fake-model"
        assert resp.degraded is False
        assert resp.attempted_models == ["fake-model"]
        assert resp.content is not None
        assert len(resp.content) > 0
        assert resp.model is not None
