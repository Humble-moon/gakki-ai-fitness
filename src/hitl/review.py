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

from dataclasses import dataclass, field
from src.config import HITL_CONFIDENCE_THRESHOLD


# ---------------------------------------------------------------------------
# 确定性安全规则：伤病关键词 → 高风险动作 → 禁止组合
# 不依赖 LLM，规则引擎直接判定。这是"宁可误报不可漏报"的最后防线。
# ---------------------------------------------------------------------------

# 伤病关键词 → 应禁止的动作模式（substring 匹配）
INJURY_EXERCISE_CONFLICTS = {
    "腰": ["深蹲", "硬拉", "划船", "罗马尼亚", "早安式"],
    "椎": ["深蹲", "硬拉", "划船", "推举", "罗马尼亚"],
    "间盘": ["深蹲", "硬拉", "划船", "推举", "罗马尼亚"],
    "背": ["深蹲", "硬拉", "罗马尼亚", "划船", "推举"],
    "膝": ["深蹲", "箭步蹲", "分腿蹲", "腿举", "跳跃"],
    "半月板": ["深蹲", "箭步蹲", "分腿蹲", "腿举"],
    "髌骨": ["深蹲", "箭步蹲", "腿举"],
    "肩": ["推举", "卧推", "飞鸟", "侧平举", "前平举"],
    "肩袖": ["推举", "卧推", "飞鸟", "侧平举", "前平举"],
    "肩峰": ["推举", "卧推", "飞鸟", "侧平举"],
    "脱臼": ["推举", "卧推", "引体向上", "飞鸟"],
    "肘": ["弯举", "臂屈伸", "窄距卧推", "推举"],
    "网球肘": ["弯举", "臂屈伸", "窄距卧推", "引体向上"],
    "腱鞘": ["弯举", "臂屈伸"],
    "腕": ["弯举", "卧推", "推举", "臂屈伸"],
    "TFCC": ["弯举", "卧推", "推举"],
    "颈": ["深蹲", "推举", "杠铃"],
    "踝": ["提踵", "跳跃", "深蹲", "箭步蹲"],
    "跟腱": ["提踵", "跳跃", "小腿", "跑步", "跳绳"],
    "手术": ["深蹲", "硬拉", "卧推", "推举", "划船"],  # 术后所有大重量复合动作都禁
    "术后": ["深蹲", "硬拉", "卧推", "推举", "划船"],
    "重建": ["深蹲", "硬拉", "卧推", "推举", "划船", "箭步蹲"],
    "炎症": ["深蹲", "硬拉", "卧推", "推举"],
}

# 高危伤病关键词：命中任何一个，无论 LLM 怎么判，直接标记 danger
CRITICAL_INJURY_KEYWORDS = [
    "间盘", "腰突", "半月板", "髌骨", "脱臼", "手术", "术后", "重建",
    "瘫痪", "断裂", "撕裂", "骨折", "TFCC",
]


@dataclass
class ReviewDecision:
    """
    人工审核决策结果数据结构。

    职责：封装 HITL 的判断结果，让编排引擎根据 needs_review 字段分流。

    字段说明：
        needs_review: bool   - 是否需要人工审核
        reason: str          - 为什么需要/不需要审核
        severity: str        - "safe" / "warning" / "danger"
        suggestions: list    - 审核员应关注的问题摘要列表
    """
    needs_review: bool
    reason: str
    severity: str
    suggestions: list


class HITLReview:
    """HITL 审核决策器。两阶段审查：
    阶段 1 — 确定性规则引擎：不依赖 LLM，直接用伤病-动作冲突表判定。
    阶段 2 — LLM 结果复审：根据 FactChecker 的置信度和 severity 做分流。
    """

    def check(self, fact_check_result: dict,
              plan: dict = None, profile: dict = None) -> ReviewDecision:
        """综合 FactChecker LLM 结果 + 确定性规则，判定是否需要人工审核。

        参数：
            fact_check_result: dict — FactChecker Agent 的输出
            plan: dict | None      — 训练计划（用于确定性规则检查）
            profile: dict | None   — 用户画像（用于确定性规则检查）

        返回值：ReviewDecision
        """
        # === 阶段 1：确定性规则引擎 ===
        # 不依赖 LLM 输出，直接用伤病-动作冲突表强制判定。
        # 即使 LLM 漏报，这些规则也能保证危险场景被拦截。
        rule_issues = []
        if plan and profile:
            rule_issues = self._check_conflicts(plan, profile)

        if rule_issues:
            return ReviewDecision(
                needs_review=True,
                reason=f"规则引擎检测到 {len(rule_issues)} 个伤病-动作冲突",
                severity="danger",
                suggestions=rule_issues,
            )

        # === 阶段 2：LLM 结果复审 ===
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

    def _check_conflicts(self, plan: dict, profile: dict) -> list:
        """确定性规则：检查伤病与训练动作的冲突。

        遍历 profile["injuries"] 和 plan["days"][*]["exercises"][*]["name"]，
        用 INJURY_EXERCISE_CONFLICTS 表做 substring 匹配。
        同时检查 query_text 中是否提到了禁止动作（用户可能在询问危险动作）。
        任何冲突都直接返回 danger 级别问题。
        """
        injuries = profile.get("injuries", [])
        if isinstance(injuries, str):
            injuries = [injuries]
        if not injuries:
            return []

        # 收集所有训练动作名
        exercise_names = []
        for day in plan.get("days", []):
            for ex in day.get("exercises", []):
                name = ex.get("name", ex.get("exercise", ""))
                if name:
                    exercise_names.append(name)

        # 同时检查 plan 中的 user_query / focus 文本
        query_text = plan.get("user_query", "")
        if not query_text:
            for day in plan.get("days", []):
                focus = day.get("focus", "")
                if focus:
                    query_text += focus

        issues = []

        for injury in injuries:
            injury_lower = injury.lower() if isinstance(injury, str) else str(injury)
            # 检查每个伤病关键词 → 冲突动作
            for keyword, forbidden_exercises in INJURY_EXERCISE_CONFLICTS.items():
                if keyword in injury_lower or keyword in query_text:
                    # 1. 检查 plan 中的已知动作名
                    for ex_name in exercise_names:
                        for forbidden in forbidden_exercises:
                            if forbidden in ex_name:
                                issues.append(
                                    f"[规则引擎] 伤病「{injury}」与动作「{ex_name}」"
                                    f"存在冲突（触发词: {keyword}），建议人工审核"
                                )
                                break  # 每个动作只报一次

                    # 2. 也检查 query 文本中是否提到了禁止动作
                    # 例如用户问"跟腱炎怎么练小腿"，query 中提到"小腿"→推测为提踵类动作→触发冲突
                    for forbidden in forbidden_exercises:
                        if forbidden in query_text:
                            issues.append(
                                f"[规则引擎] 用户查询含伤病「{injury}」，"
                                f"且提到高风险动作「{forbidden}」（触发词: {keyword}），建议人工审核"
                            )

        # 检查 query_text 中是否包含高危伤病关键词
        for keyword in CRITICAL_INJURY_KEYWORDS:
            if keyword in query_text:
                # 对所有涉及的动作都标记
                for ex_name in exercise_names:
                    issue_text = (
                        f"[规则引擎] 用户描述含高危关键词「{keyword}」，"
                        f"计划中的「{ex_name}」需人工确认安全性"
                    )
                    if issue_text not in issues:
                        issues.append(issue_text)
                if not exercise_names:
                    # 没有具体动作名也要报（高危关键词本身就值得关注）
                    issues.append(
                        f"[规则引擎] 用户描述含高危关键词「{keyword}」，"
                        f"请人工审核训练方案安全性"
                    )
                break  # 只报一次高危

        # 去重
        return list(dict.fromkeys(issues))
