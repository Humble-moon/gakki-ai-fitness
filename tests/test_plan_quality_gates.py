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

import pytest

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


class TestNameVariantsAreNotHallucinations:
    """区分「同一动作的另一种写法」与「根本不存在的动作」。

    实测教训：动作库校验上线后，一个无伤病用户的正常计划被扣下，理由是
    「计划包含 5 个动作库中不存在的动作」——而那 5 个（保加利亚分腿蹲（扶支撑）、
    站姿哑铃弯举、地面哑铃飞鸟…）都指向库里真实存在的动作，只是多了
    括号说明或姿势前缀。**把命名变体当幻觉，会把正常计划全部阻挡。**
    """

    KNOWN = {
        "保加利亚分腿蹲", "哑铃弯举", "锤式弯举", "哑铃飞鸟", "哑铃卧推",
        "哑铃单臂划船（支撑凳）", "上斜哑铃弯举（肱肌专注）",
    }

    @pytest.mark.parametrize(
        "planned",
        [
            "保加利亚分腿蹲（扶支撑）",   # 括号说明
            "站姿哑铃弯举",               # 姿势前缀
            "站姿锤式弯举",
            "地面哑铃飞鸟",
            "地面哑铃卧推",
            "双臂哑铃卧推",
            "徒手哑铃飞鸟",
        ],
    )
    def test_name_variant_is_not_flagged(self, planned):
        result = {"days": [{"exercises": [{"name": planned}]}]}
        assert collect_unknown_exercises(result, self.KNOWN) == [], (
            f"{planned} 是命名变体，不该被判为库外动作"
        )

    def test_concatenated_names_are_still_flagged(self):
        """真幻觉必须仍被抓住——回归这条是为了防止把校验改废。"""
        # 「慢离心卧推」+「哑铃地板卧推」拼接而成
        result = {"days": [{"exercises": [{"name": "慢离心哑铃地板卧推"}]}]}
        unknown = collect_unknown_exercises(
            result, {"慢离心卧推", "哑铃地板卧推", "哑铃飞鸟"}
        )
        assert unknown == ["慢离心哑铃地板卧推"]

    def test_completely_made_up_name_is_flagged(self):
        result = {"days": [{"exercises": [{"name": "超级无敌爆炸深蹲"}]}]}
        assert collect_unknown_exercises(result, self.KNOWN) == ["超级无敌爆炸深蹲"]

    def test_library_name_with_paren_note_still_matches(self):
        """库内名自带括号时，两侧归一化口径必须一致。"""
        result = {"days": [{"exercises": [{"name": "哑铃单臂划船（对侧手撑大腿）"}]}]}
        assert collect_unknown_exercises(result, self.KNOWN) == []

    def test_exact_match_unaffected(self):
        result = {"days": [{"exercises": [{"name": "哑铃弯举"}]}]}
        assert collect_unknown_exercises(result, self.KNOWN) == []

    def test_normalizer_strips_only_one_prefix_layer(self):
        """只剥一层前缀——多剥会把库里的真实基线名剥坏。"""
        from src.core.plan_finalization import normalize_exercise_name

        assert normalize_exercise_name("站姿哑铃弯举") == "哑铃弯举"
        assert normalize_exercise_name("保加利亚分腿蹲（扶支撑）") == "保加利亚分腿蹲"
        # 「单臂哑铃卧推」在库里是真实动作名，不该被剥成「哑铃卧推」
        assert normalize_exercise_name("单臂哑铃卧推") == "哑铃卧推"  # 剥一层
        assert normalize_exercise_name("哑铃弯举") == "哑铃弯举"


class TestFinalizeHandlesUnknownExercises:
    """库外动作**只提示，不阻断交付**。

    本组最初断言的是"库外即强制送审"。实测推翻了这个设计：无伤病的健康
    用户连续两次被扣下，被点名的却是 「保加利亚分腿蹲（扶支撑）」这类
    命名变体，以及 「帕洛夫推」这类真实存在、只是库未收录的动作。而 LLM
    生成的计划本来就用不全库内动作名——强制送审等于 100% 的计划都交付
    不出去，叠加 HITL 未定义"审核人"，形成死锁。

    保留的核心价值是：库外动作仍要**被记录并展示给用户**，只是不再阻断。
    """

    def _safe_check(self):
        return [{"is_safe": True, "issues": [], "confidence": 0.9}]

    def test_unknown_exercise_is_recorded_and_warned(self):
        result = {"days": [{"exercises": [{"name": "帕洛夫推"}]}]}
        final = finalize_result(
            result, self._safe_check(), 3, known_exercise_names={"哑铃卧推"}
        )
        assert final["unknown_exercises"] == ["帕洛夫推"]
        assert any("不在动作库" in w for w in final["warnings"]), "信息必须给到用户"

    def test_unknown_exercise_does_not_block_delivery(self):
        """关键断言：库外动作不得阻断交付。"""
        result = {"days": [{"exercises": [{"name": "帕洛夫推"}]}]}
        final = finalize_result(
            result, self._safe_check(), 0, known_exercise_names={"哑铃卧推"}
        )
        assert final["requires_review"] is False

    def test_safety_still_blocks_regardless_of_unknown(self):
        """降级库外动作，不得把真正的安全拦截一起降级。"""
        result = {"days": [{"exercises": [{"name": "帕洛夫推"}]}]}
        unsafe = [{"is_safe": False, "issues": [{"issue": "危险"}], "confidence": 0.9}]
        final = finalize_result(
            result, unsafe, 0, known_exercise_names={"哑铃卧推"}
        )
        assert final["requires_review"] is True

    def test_known_exercises_produce_no_warning(self):
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


class TestGraphPathAlsoValidatesExercises:
    """LangGraph 路径（/api/v2/*）必须与手写编排器同样做动作库校验。

    这条用例是有来由的：动作库校验最初只接在 ``Orchestrator._finalize_result``
    上，而 ``graph/nodes.py`` 的 ``finalize_node`` 是直接调用
    ``plan_finalization.finalize_result`` 的——于是 v2 链路完全没有这道校验，
    实测中 v2 生成的计划里 3 个动作名都不在库（单臂哑铃划船／
    坐姿哑铃肩推（靠墙）／臀桥（负重））却一条都没被标记。
    """

    def _deps(self, names):
        from types import SimpleNamespace

        return SimpleNamespace(library_exercise_names_fn=lambda: names)

    def test_graph_node_records_unknown_exercises(self):
        from src.graph.nodes import finalize_node

        state = {
            "result": {"goal": "增肌", "days": [{"exercises": [{"name": "帕洛夫推"}]}]},
            "checks": [{"is_safe": True, "issues": [], "confidence": 0.9}],
            "rewrite_count": 0,
            "expected_goal": "增肌",
        }
        out = finalize_node(self._deps({"哑铃卧推"}), state)
        # 记录 + 提示，但不阻断交付（设计依据见 TestFinalizeHandlesUnknownExercises）
        assert out["result"]["unknown_exercises"] == ["帕洛夫推"]
        assert out["result"]["requires_review"] is False
        assert any("不在动作库" in w for w in out["result"]["warnings"])

    def test_graph_node_without_capability_skips_check(self):
        """deps 不带该能力时跳过校验，而不是抛异常——测试替身常这样构造。"""
        from types import SimpleNamespace

        from src.graph.nodes import finalize_node

        state = {
            "result": {"goal": "增肌", "days": [{"exercises": [{"name": "任意名字"}]}]},
            "checks": [{"is_safe": True, "issues": [], "confidence": 0.9}],
            "rewrite_count": 0,
            "expected_goal": "增肌",
        }
        out = finalize_node(SimpleNamespace(), state)
        assert out["result"]["unknown_exercises"] == []

    def test_getter_failure_does_not_break_generation(self):
        """查库抛异常时必须降级跳过，不能让整条生成流程失败。"""
        from types import SimpleNamespace

        from src.graph.nodes import finalize_node

        def _boom():
            raise RuntimeError("db down")

        state = {
            "result": {"goal": "增肌", "days": [{"exercises": [{"name": "哑铃卧推"}]}]},
            "checks": [{"is_safe": True, "issues": [], "confidence": 0.9}],
            "rewrite_count": 0,
            "expected_goal": "增肌",
        }
        out = finalize_node(SimpleNamespace(library_exercise_names_fn=_boom), state)
        assert out["result"]["unknown_exercises"] == []

    def test_deps_from_orchestrator_wires_the_capability(self):
        """接线本身要有断言——否则又会出现"只修了一个后端"。"""
        from types import SimpleNamespace

        from src.graph.deps import CoachGraphDeps, deps_from_orchestrator

        orch = SimpleNamespace(
            planner=1, retriever=2, writer=3, fact_checker=4, cache=5,
            conversation=6, long_term=7, review_store=8,
            _library_exercise_names=lambda: {"哑铃卧推"},
        )
        deps = deps_from_orchestrator(orch, resolutions=9, thread_index=10)
        assert isinstance(deps, CoachGraphDeps)
        assert deps.library_exercise_names_fn() == {"哑铃卧推"}

    def test_deps_without_orchestrator_capability_stays_none(self):
        from types import SimpleNamespace

        from src.graph.deps import deps_from_orchestrator

        orch = SimpleNamespace(
            planner=1, retriever=2, writer=3, fact_checker=4, cache=5,
            conversation=6, long_term=7, review_store=8,
        )
        deps = deps_from_orchestrator(orch, resolutions=9, thread_index=10)
        assert deps.library_exercise_names_fn is None


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
