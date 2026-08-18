import os

import pytest


@pytest.mark.live
def test_configured_provider_answers_fixed_sanitized_prompt():
    """Call the configured provider only when explicitly opted in."""
    if os.getenv("RUN_LIVE_LLM_TESTS") != "1":
        pytest.skip("live tests require RUN_LIVE_LLM_TESTS=1")

    from src.llm.provider import LLMProvider

    response = LLMProvider().chat([
        {
            "role": "user",
            "content": "请用一句中文说明热身的目的，不要包含个人信息。",
        }
    ])

    assert response.content.strip()
    assert response.model != "none"
