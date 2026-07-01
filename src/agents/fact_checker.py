from src.llm.provider import LLMProvider
from src.llm.prompts.fact_checker import build_fact_checker_messages
from src.hitl.review import HITLReview

class FactCheckerAgent:
    def __init__(self):
        self.llm = LLMProvider()
        self.hitl = HITLReview()

    def check(self, plan: dict, profile: dict) -> dict:
        messages = build_fact_checker_messages(plan, profile)
        result = self.llm.chat_with_json_mode(messages)
        result.setdefault("is_safe", True)
        result.setdefault("issues", [])
        result.setdefault("confidence", 0.8)
        result.setdefault("requires_human_review", False)
        review = self.hitl.check(result)
        result["requires_human_review"] = review.needs_review
        result["review_reason"] = review.reason
        result["review_severity"] = review.severity
        return result
