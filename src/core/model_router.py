from src.llm.provider import LLMProvider

class ModelRouter:
    def __init__(self):
        self.llm = LLMProvider()

    def route(self, task_type: str) -> str:
        """Route to appropriate model. Simple tasks use DeepSeek-V3."""
        simple_tasks = ["retrieve", "keyword_search", "cache_lookup"]
        if task_type in simple_tasks:
            return "deepseek-chat"
        return "deepseek-chat"
