"""
================================================================================
文件角色：HITL（Human-in-the-Loop，人机协同）审核决策模块
================================================================================
- 被调用者：编排引擎在 FactChecker 输出结果后，调用 HITLReview.check()
  判断该计划是否需要转人工审核。
- 调用者：本模块依赖 config 中的 HITL_CONFIDENCE_THRESHOLD 阈值配置。
- 项目角色：AI 安全管线的最后一环——"AI 说安全不一定安全"。
  这是人机协同的分流决策点：根据置信度和问题严重级别决定是直接返回用户
  还是进入人工审核队列。

Pipeline 位置：
  Planner → Retriever → Writer → FactChecker → HITLReview.check() →
    ├─ needs_review=False → 直接返回给用户
    └─ needs_review=True  → 进入人工审核队列 → 人工确认/修改后返回
================================================================================
"""

from dataclasses import dataclass
from src.config import HITL_CONFIDENCE_THRESHOLD


@dataclass
class ReviewDecision:
    """
    人工审核决策结果数据结构。

    职责：封装 HITL 的判断结果，让编排引擎根据 needs_review 字段分流。

    字段说明：
        needs_review: bool   - 是否需要人工审核。
                              True: 计划进入审核队列，暂停自动交付
                              False: 计划通过，直接返回给用户
        reason: str          - 为什么需要/不需要审核，供前端展示或日志记录
        severity: str        - 严重级别，三档：
                              "safe"    = 无需审核，安全
                              "warning" = 有警告项，建议人工看但非强制
                              "danger"  = 有危险项，强制人工审核
        suggestions: list    - 审核员应关注的问题摘要列表（从 FactChecker 的
                              issues 中提取 issue 字段），方便审核员快速定位
    """
    needs_review: bool
    reason: str
    severity: str
    suggestions: list


class HITLReview:
    """
    HITL 审核决策器，判断 AI 生成的计划是否需要人工介入。

    职责：
    接收 FactChecker 的安全审查结果，根据置信度和问题严重级别做出分流决策。
    这是一个规则引擎（rule-based），不是 LLM 调用——因为决策逻辑足够明确，
    不需要 AI 参与。

    决策规则说明：
    1. 强制审核条件（满足任一即 needs_review=True）：
       a) 置信度低于阈值 HITL_CONFIDENCE_THRESHOLD：
          AI 自己都不确定自己判断得对不对，必须让人工确认。
       b) 存在 severity="danger" 的问题：
          高危动作/伤病冲突，必须人工把关，不能让 AI 承担医疗风险。
       c) 存在 severity="warning" 的问题：
          警告级别也触发审核，但严重级别标记为 "warning" 而非 "danger"，
          这影响了人工审核的优先级判断。

    2. 通过条件：
       置信度 >= 阈值 且 没有 issues 或只有 "info" 级别提示。
       此时 needs_review=False, severity="safe"，直接交付用户。
    """

    def check(self, fact_check_result: dict) -> ReviewDecision:
        """
        根据 FactChecker 输出判断是否需要人工审核。

        参数：
            fact_check_result: dict  - FactChecker Agent 返回的字典，包含：
                                      - confidence: float       (0.0 ~ 1.0)
                                      - is_safe: bool           (LLM 的判断)
                                      - issues: list[dict]      (问题列表)
                                      - requires_human_review: bool (LLM 的建议)

        返回值：
            ReviewDecision           - 包含 needs_review / reason / severity /
                                      suggestions 的决策数据对象

        核心逻辑（按优先级判断）：
        1. 从 fact_check_result 中提取 confidence、issues 列表。
        2. 遍历 issues，检查是否存在 "danger" 或 "warning" 级别的问题。
           用 any() 做惰性短路检查，效率高。
        3. 判断优先级：
           - 置信度 < 阈值 或 有 danger → 强制审核（severity=danger 优先于 warning）
           - 仅有 warning → 也触发审核，但 severity=warning
           - 以上都不满足 → 通过，needs_review=False
        4. 从 issues 中提取所有 issue 描述文本，填充 suggestions 列表。

        为什么需要 HITL 而不直接用 FactChecker 的 is_safe：
        FactChecker 的 is_safe 是 LLM 的单项判断，可能存在"假安全"（模型不够
        谨慎）。HITL 引入额外的置信度阈值和分级策略，对低置信度但 is_safe=True
        的情况也会拦截，实现"宁可多审也不漏审"的安全保守策略。
        """
        # 提取 FactChecker 的关键字段
        confidence = fact_check_result.get("confidence", 0)
        issues = fact_check_result.get("issues", [])

        # 检查是否存在 danger 或 warning 级别的问题
        # 用 any() + 生成器实现惰性短路，遇到 danger 就停止遍历
        has_danger = any(i.get("severity") == "danger" for i in issues)
        has_warning = any(i.get("severity") == "warning" for i in issues)

        # 规则 1：低置信度 或 有危险项 → 强制人工审核
        if confidence < HITL_CONFIDENCE_THRESHOLD or has_danger:
            return ReviewDecision(
                needs_review=True,
                reason=f"置信度 {confidence:.2f} 低于阈值或有危险建议",
                # 有 danger 时 severity 优先标为 "danger"（而非 warning）
                severity="danger" if has_danger else "warning",
                suggestions=[i["issue"] for i in issues]
            )

        # 规则 2：有警告项 → 触发审核，但严重级别为 "warning"
        if has_warning:
            return ReviewDecision(
                needs_review=True,
                reason="存在需要确认的警告项",
                severity="warning",
                suggestions=[i["issue"] for i in issues]
            )

        # 规则 3：无问题或仅 info → 通过
        return ReviewDecision(
            needs_review=False,
            reason="",
            severity="safe",
            suggestions=[]
        )
