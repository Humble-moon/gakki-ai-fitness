from dataclasses import dataclass
from src.config import HITL_CONFIDENCE_THRESHOLD

@dataclass
class ReviewDecision:
    needs_review: bool
    reason: str
    severity: str  # "safe" | "warning" | "danger"
    suggestions: list

class HITLReview:
    def check(self, fact_check_result: dict) -> ReviewDecision:
        confidence = fact_check_result.get("confidence", 0)
        issues = fact_check_result.get("issues", [])
        has_danger = any(i.get("severity") == "danger" for i in issues)
        has_warning = any(i.get("severity") == "warning" for i in issues)

        if confidence < HITL_CONFIDENCE_THRESHOLD or has_danger:
            return ReviewDecision(
                needs_review=True,
                reason=f"置信度 {confidence:.2f} 低于阈值或有危险建议",
                severity="danger" if has_danger else "warning",
                suggestions=[i["issue"] for i in issues]
            )
        if has_warning:
            return ReviewDecision(
                needs_review=True,
                reason="存在需要确认的警告项",
                severity="warning",
                suggestions=[i["issue"] for i in issues]
            )
        return ReviewDecision(
            needs_review=False, reason="", severity="safe", suggestions=[]
        )
