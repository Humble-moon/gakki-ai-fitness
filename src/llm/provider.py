from dataclasses import dataclass
from openai import OpenAI
from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

@dataclass
class LLMResponse:
    content: str
    model: str
    tokens: int

class LLMProvider:
    def __init__(self):
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )
        self.default_model = "deepseek-chat"

    def chat(self, messages: list, temperature: float = 0.3, model: str = None) -> LLMResponse:
        resp = self.client.chat.completions.create(
            model=model or self.default_model,
            messages=messages,
            temperature=temperature
        )
        return LLMResponse(
            content=resp.choices[0].message.content,
            model=resp.model,
            tokens=resp.usage.total_tokens
        )

    def chat_with_json_mode(self, messages: list, model: str = None) -> dict:
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
