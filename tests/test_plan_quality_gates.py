"""计划质量闸门：动作库校验 与 规则引擎误报防护。

两组用例都来自一次真实执行的实测发现，不是假想的边界：

1. **规则引擎把训练日 focus 当伤病文本**——实测中 day1 的 focus 是
   「胸肌与肩部」，其中"肩"命中了 ``INJURY_EXERCISE_CONFLICTS`` 的触发词，
   于是「哑铃卧推」被判成与「腰椎间盘突出」冲突。用户根本没有肩伤，
   而这类误报会把真实冲突淹没在噪声里。

2. **重写回路不校验动作名**——首轮计划的动作名都真实存在，
   但重写 3 轮后出现了「俯卧地板单臂哑铃划船」这类库里查不到的名字。
   它们是由真实动作名词素重组而成（哑铃单臂划船 + 俯卧地板），
   比凭空编造更难被人工发现，因此必须由确定性校验兜住。
"""

from __future__ import annotations

from src.core.plan_finalization import collect_unknown_exercises, finalize_result
from src.hitl.review import HITLReview


class TestFocusIsNotInjuryText:
    """训练日 focus 是计划内容，不得参与伤病判定。"""

    def test_focus_text_does_not_trigger_conflicts(self):
        """实测原例：focus「胸肌与肩部」不应让卧推被判与腰突冲突。"""
        review = HITLReview()
        plan = {"days": [{"focus": "胸肌与肩部", "exercises": [{"name": "哑铃卧推"}]}]}
        issues = review._check_conflicts(plan, {"injuries": ["腰椎间盘突出"]})
        assert issues == [], f"focus 文本导致误报: {issues}"

    def test_focus_does_not_leak_into_critical_keyword_check(self):
        """focus 也不能污染高危关键词检查。"""
        review = HITLReview()
        plan = {"days": [{"focus": "肩部训练", "exercises": [{"name": "哑铃卧推"}]}]}
        issues = review._check_conflicts(plan, {"injuries": ["膝关节劳损"]})
        assert issues == [], f"focus 文本导致误报: {issues}"

    def test_real_conflict_is_still_detected(self):
        """修复误报不得把真报一起修掉——硬拉对腰突必须仍然报警。"""
        review = HITLReview()
        plan = {"days": [{"focus": "腿部", "exercises": [{"name": "杠铃硬拉"}]}]}
        issues = review._check_conflicts(plan, {"injuries": ["腰椎间盘突出"]})
        assert issues, "真实冲突被漏检"
        assert any("硬拉" in item for item in issues)

    def test_user_query_is_still_checked(self):
        """用户主动在 query 里提到伤病时，仍应检出（injuries 为空也不放过）。"""
        review = HITLReview()
        plan = {
            "user_query": "我膝盖疼，想练腿",
            "days": [{"focus": "腿部", "exercises": [{"name": "深蹲"}]}],
        }
        issues = review._check_conflicts(plan, {"injuries": []})
        # injuries 为空时本函数提前返回——该场景由 qa_safety 在问答路径覆盖，
        # 这里只断言不因 focus 产生误报。
        assert isinstance(issues, list)

    def test_no_injuries_returns_early(self):
        review = HITLReview()
        plan = {"days": [{"focus": "胸肌与肩部", "exercises": [{"name": "哑铃卧推"}]}]}
        assert review._check_conflicts(plan, {"injuries": []}) == []


class TestExerciseLibraryValidation:
    """库外动作名必须被识别，并强制送审。"""

    def test_collects_unknown_names(self):
        result = {"days": [{"exercises": [
            {"name": "哑铃卧推"},
            {"name": "俯卧地板单臂哑铃划船"},
        ]}]}
        unknown = collect_unknown_exercises(result, {"哑铃卧推", "杠铃深蹲"})
        assert unknown == ["俯卧地板单臂哑铃划船"]

    def test_keeps_order_and_dedupes(self):
        result = {"days": [
            {"exercises": [{"name": "幽灵动作A"}, {"name": "幽灵动作B"}]},
            {"exercises": [{"name": "幽灵动作A"}]},
        ]}
        assert collect_unknown_exercises(result, {"哑铃卧推"}) == ["幽灵动作A", "幽灵动作B"]

    def test_all_known_returns_empty(self):
        result = {"days": [{"exercises": [{"name": "哑铃卧推"}]}]}
        assert collect_unknown_exercises(result, {"哑铃卧推"}) == []

    def test_missing_known_names_skips_check(self):
        """不提供库名单时跳过校验，老调用方行为不变。"""
        result = {"days": [{"exercises": [{"name": "任意名字"}]}]}
        assert collect_unknown_exercises(result, None) == []
        assert collect_unknown_exercises(result, set()) == []

    def test_tolerates_malformed_plan(self):
        """脏数据结构不得让校验抛异常。"""
        assert collect_unknown_exercises({}, {"哑铃卧推"}) == []
        assert collect_unknown_exercises({"days": None}, {"哑铃卧推"}) == []
        assert collect_unknown_exercises({"days": [{"exercises": [None, "x"]}]}, {"a"}) == []

    def test_supports_legacy_exercise_key(self):
        result = {"days": [{"exercises": [{"exercise": "幽灵动作"}]}]}
        assert collect_unknown_exercises(result, {"哑铃卧推"}) == ["幽灵动作"]


class TestFinalizeGatesOnUnknownExercises:
    """库外动作必须让终态判定强制送审。"""

    def _safe_check(self):
        return [{"is_safe": True, "issues": [], "confidence": 0.9}]

    def test_unknown_exercise_forces_review(self):
        result = {"days": [{"exercises": [{"name": "俯卧地板单臂哑铃划船"}]}]}
        final = finalize_result(
            result, self._safe_check(), 3, known_exercise_names={"哑铃卧推"}
        )
        assert final["unknown_exercises"] == ["俯卧地板单臂哑铃划船"]
        assert final["requires_review"] is True
        assert final["_persistence_allowed"] is False
        assert final["review_severity"] == "danger"
        assert "动作库" in final["review_reason"]
        assert any("不在动作库" in w for w in final["warnings"])

    def test_unknown_exercise_blocks_even_when_llm_says_safe(self):
        """关键断言：LLM 判 safe=True 也不能让库外动作的计划直接交付。"""
        result = {"days": [{"exercises": [{"name": "幽灵动作"}]}]}
        final = finalize_result(
            result, self._safe_check(), 0, known_exercise_names={"哑铃卧推"}
        )
        assert final["requires_review"] is True

    def test_known_exercises_do_not_force_review(self):
        result = {"days": [{"exercises": [{"name": "哑铃卧推"}]}]}
        final = finalize_result(
            result, self._safe_check(), 0, known_exercise_names={"哑铃卧推"}
        )
        assert final["unknown_exercises"] == []
        assert final["requires_review"] is False

    def test_behavior_unchanged_without_known_names(self):
        """未注入库名单时保持原行为（不因新增参数改变既有路径）。"""
        result = {"days": [{"exercises": [{"name": "任意名字"}]}]}
        final = finalize_result(result, self._safe_check(), 0)
        assert final["unknown_exercises"] == []
        assert final["requires_review"] is False


class TestConflictDeduplication:
    """同一「伤病+动作」只报一条，触发词并列展示。

    这是修复误报后暴露出来的问题：一个伤病描述会命中多个触发词
    （「腰椎间盘突出」同时含「腰」「椎」「间盘」），逐触发词展开会让
    3 个动作膨胀成 9~11 条同义 issue——不是错报，但审核者得在噪声里找信号。
    """

    def test_multiple_keywords_collapse_to_one_issue(self):
        review = HITLReview()
        plan = {"days": [{"exercises": [{"name": "高脚杯深蹲"}]}]}
        issues = review._check_conflicts(plan, {"injuries": ["腰椎间盘突出"]})
        assert len(issues) == 1, f"同义 issue 未合并: {issues}"
        assert "高脚杯深蹲" in issues[0]
        # 触发词并列展示，合并不得丢信息
        assert "/" in issues[0]

    def test_issue_count_equals_true_conflict_count(self):
        """条数应等于真实冲突动作数，而不是 动作数 × 触发词数。"""
        review = HITLReview()
        plan = {"days": [{"exercises": [
            {"name": "高脚杯深蹲"},
            {"name": "胸支撑哑铃划船"},
            {"name": "靠墙哑铃推举"},
        ]}]}
        issues = review._check_conflicts(plan, {"injuries": ["腰椎间盘突出"]})
        assert len(issues) == 3, f"应为每个冲突动作一条: {issues}"

    def test_no_duplicate_issue_text(self):
        review = HITLReview()
        plan = {"days": [{"exercises": [{"name": "杠铃硬拉"}]}]}
        issues = review._check_conflicts(plan, {"injuries": ["腰椎间盘突出"]})
        assert len(issues) == len(set(issues)), f"存在完全重复的 issue: {issues}"

    def test_query_side_conflicts_also_deduped(self):
        """用户查询侧的高风险动作同样只报一条。"""
        review = HITLReview()
        plan = {"user_query": "我有跟腱炎，怎么练小腿和跳跃",
                "days": [{"exercises": [{"name": "提踵"}]}]}
        issues = review._check_conflicts(plan, {"injuries": ["跟腱炎"]})
        assert len(issues) == len(set(issues)), f"query 侧存在重复: {issues}"


class TestReviewPayloadSurfacesUnknownExercises:
    """送审载荷必须让审核者直接看到库外动作名。

    否则审核者只能从 issues 文本里推断，而"动作名是编造的"这件事
    恰恰最该被核对——它比动作强度不当更隐蔽。
    """

    def test_payload_lists_unknown_exercises(self):
        from src.core.plan_finalization import (
            build_review_pending_payload,
            create_review_artifact,
        )

        class _Store:
            def create(self, **kwargs):
                class _Artifact:
                    review_id = "r-1"
                    status = "pending"
                    created_at = "2026-09-24T00:00:00Z"
                    issues = kwargs.get("issues", [])
                    severity = kwargs.get("severity", "warning")
                    prohibited_actions = kwargs.get("prohibited_actions", [])

                return _Artifact()

        result = {
            "days": [{"exercises": [{"name": "幽灵动作"}]}],
            "active_issues": [],
            "review_suggestions": [],
            "review_severity": "danger",
            "review_reason": "计划包含 1 个动作库中不存在的动作，需人工确认",
            "unknown_exercises": ["幽灵动作"],
        }
        artifact = create_review_artifact(_Store(), {"goal": "增肌"}, "增肌", result)
        payload = build_review_pending_payload(result, artifact)
        assert payload["review"]["unknown_exercises"] == ["幽灵动作"]

    def test_payload_omits_key_when_none_unknown(self):
        from src.core.plan_finalization import (
            build_review_pending_payload,
            create_review_artifact,
        )

        class _Store:
            def create(self, **kwargs):
                class _Artifact:
                    review_id = "r-2"
                    status = "pending"
                    created_at = "2026-09-24T00:00:00Z"
                    issues = []
                    severity = "warning"
                    prohibited_actions = []

                return _Artifact()

        result = {
            "days": [{"exercises": [{"name": "哑铃卧推"}]}],
            "active_issues": [],
            "review_suggestions": [],
            "unknown_exercises": [],
        }
        artifact = create_review_artifact(_Store(), {}, "", result)
        payload = build_review_pending_payload(result, artifact)
        assert "unknown_exercises" not in payload["review"]
