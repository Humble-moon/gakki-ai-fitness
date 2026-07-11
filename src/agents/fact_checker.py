"""
===========================================================================
文件角色：事实核查器 Agent —— 对生成的训练计划进行安全审查和 HITL 升级判定
===========================================================================
- 被谁调用：Orchestrator 在流水线的第 5 步调用 FactCheckerAgent.check()
- 调用谁：
    LLMProvider.chat_with_json_mode()  → 调用 LLM 对训练计划做安全检查
    build_fact_checker_messages()       → 构造核查提示词（来自 prompts 模块）
    HITLReview.check()                  → 人工复核升级判定逻辑
- 核心职责：
    1. 调用 LLM 审查生成计划的合理性（动作选择、强度、伤病冲突等）
    2. 通过 HITLReview 规则引擎判定是否需要升级到人工复核
    3. 将审查结果（警告列表 + 置信度 + 是否需人工审查）附加到 plan 中
- 在整个系统中的角色："守门员" —— 在将结果返回给用户前做最后一道安全检查，
  防止 LLM 生成危险的训练建议（如伤病期做高强度动作、新手做高危动作等）
===========================================================================
"""

from src.llm.provider import LLMProvider
from src.llm.prompts.fact_checker import build_fact_checker_messages
from src.hitl.review import HITLReview


class FactCheckerAgent:
    """事实核查器 Agent：在 Orchestrator 流水线的第 5 步被调用。
    职责：两阶段审查 —— 先由 LLM 做语义层面的安全分析，再由规则引擎做 HITL 升级判定。
    LLM 负责"软"问题（如训练动作组合不合理），HITL 负责"硬"判定（如是否需要人工介入）。"""

    def __init__(self):
        self.llm = LLMProvider()
        # HITLReview：Human-In-The-Loop 规则引擎
        # 根据 LLM 安全审查结果 + 预定义危险规则（如伤病+特定动作组合），
        # 判定是否需要将任务升级为人工复核
        self.hitl = HITLReview()

    def check(self, plan: dict, profile: dict) -> dict:
        """对训练计划执行安全审查。

        输入：
            plan: dict — Writer 生成的训练计划（含 days 列表）
            profile: dict — 用户画像（含伤病、训练年限等关键安全信息）
        输出：
            dict — 包含以下字段的结果：
                - "is_safe": bool — 计划是否通过安全检查
                - "issues": list[dict] — 发现的问题列表（每条含 "issue" 描述）
                - "confidence": float — 安全审查的置信度 (0~1)
                - "requires_human_review": bool — HITL 判定的是否需要人工复核
                - "review_reason": str — 需要人工复核的原因
                - "review_severity": str — 复核的严重级别

        两阶段审查说明：
            【阶段 1 — LLM 语义审查】
                build_fact_checker_messages 将训练计划 + 用户画像 + 安全规则
                组装为 prompt，让 LLM 从多个维度检查：
                - 伤病冲突：用户的伤病与计划中的动作是否冲突
                - 强度匹配：训练负荷是否适合用户水平
                - 动作合理性：动作组合是否有过度训练风险
                - 休息恢复：休息时间是否足够
            【阶段 2 — HITL 规则引擎】
                在 LLM 审查结果基础上，应用硬编码的安全规则再做判定。
                例如：检测到伤病 + 高危动作组合 → 自动标记需要人工复核，
                即使 LLM 的 confidence 很高也不会放行。
                这是防御性设计：LLM 可能漏检或低估风险，规则引擎作为最后防线。
        """
        # 阶段 1：构建核查 prompt 并调用 LLM
        messages = build_fact_checker_messages(plan, profile)
        result = self.llm.chat_with_json_mode(messages, model="reasoner")
        # setdefault 确保 LLM 输出不完整时也有合理的默认值
        result.setdefault("is_safe", True)
        result.setdefault("issues", [])
        result.setdefault("confidence", 0.8)
        result.setdefault("requires_human_review", False)

        # 阶段 2：HITL 规则引擎复查（含确定性伤病-动作冲突检测）
        # HITL 在 LLM 结果基础上，还会独立检查 plan + profile 中的伤病冲突
        review = self.hitl.check(result, plan=plan, profile=profile)
        # HITL 的判定覆盖 LLM 的 requires_human_review（以规则引擎为准）
        result["requires_human_review"] = review.needs_review
        result["review_reason"] = review.reason
        result["review_severity"] = review.severity
        return result
