import pytest
from src.llm.provider import LLMProvider, LLMResponse

class TestLLMResponse:
    def test_response_model(self):
        resp = LLMResponse(content="测试回复", model="deepseek-v3", tokens=100)
        assert resp.content == "测试回复"
        assert resp.model == "deepseek-v3"

class TestLLMProvider:
    def test_chat_sync(self):
        provider = LLMProvider()
        resp = provider.chat(
            messages=[{"role": "user", "content": "回复一个字：好"}],
            temperature=0.1
        )
        assert resp.content is not None
        assert len(resp.content) > 0
        assert resp.model is not None
