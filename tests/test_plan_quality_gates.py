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
from src.hitl.review import HITLReview, classify_injury_nature


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

    保留的核心价值是：库外动作仍要**被记录**（`unknown_exercises` 结构化清单 +
    `warnings` 原文），只是不再阻断。

    2026-10-10 修正：**记录 ≠ 展示给用户**。这条文案反映的是我们词表的覆盖
    缺口（库里有 8 个俯卧撑变式却没有通用的「标准俯卧撑」），不是用户写错了
    名字——用户既无从"确认"，看到也只会觉得产品不认识俯卧撑。前端 `splitWarnings`
    会把它过滤掉，不再渲染。因此断言保留（字段必须还在，供自查覆盖率），
    但不再声称它是给用户看的。
    """

    def _safe_check(self):
        return [{"is_safe": True, "issues": [], "confidence": 0.9}]

    def test_unknown_exercise_is_recorded_and_warned(self):
        result = {"days": [{"exercises": [{"name": "帕洛夫推"}]}]}
        final = finalize_result(
            result, self._safe_check(), 3, known_exercise_names={"哑铃卧推"}
        )
        assert final["unknown_exercises"] == ["帕洛夫推"]
        assert any("不在动作库" in w for w in final["warnings"]), "原文必须保留在 warnings 里供自查"

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


class TestReviewPayloadTellsTheTruthAboutReview:
    """送审载荷不得承诺一个不存在的人工审核者。

    本版本没有审核台（见 5.9 节局限性）。写"请等待专业审核"会让用户
    一直等下去，演示时被问"谁来点通过"也无法回答。改为把动作交回用户：
    带着风险明细去问专业人士。
    """

    def _payload(self, **over):
        from src.core.plan_finalization import (
            build_review_pending_payload,
            create_review_artifact,
        )

        class _Store:
            def create(self, **kwargs):
                class _Artifact:
                    review_id = "r-copy"
                    status = "pending"
                    created_at = "2026-10-09T00:00:00Z"
                    issues = kwargs.get("issues", [])
                    severity = kwargs.get("severity", "danger")
                    prohibited_actions = kwargs.get("prohibited_actions", [])
                return _Artifact()

        result = {"days": [], "active_issues": [], "review_suggestions": [],
                  "review_severity": "danger", "review_reason": "检测到伤病风险"}
        result.update(over)
        artifact = create_review_artifact(_Store(), {"goal": "减脂"}, "q", result)
        return build_review_pending_payload(result, artifact)

    def test_next_step_does_not_promise_a_waiting_reviewer(self):
        step = self._payload()["review"]["next_step"]
        assert "等待" not in step, f"不得承诺等待审核者：{step}"
        assert "专业" in step, "仍应把用户导向专业人士"

    def test_payload_carries_issue_details_for_display(self):
        """issues 必须随载荷下发——前端要摊开给用户看。

        这些明细是安全门已经算出来的（哪些动作冲突、为什么、替换建议），
        只传一句 reason 等于把最有用的信息丢掉。
        """
        payload = self._payload(active_issues=[
            {"exercise": "深蹲", "issue": "与膝伤冲突", "severity": "danger"},
            {"exercise": "跳跃", "issue": "高冲击", "severity": "warning"},
        ])
        issues = payload["review"]["issues"]
        assert len(issues) == 2
        assert {i.get("severity") for i in issues} == {"danger", "warning"}
        assert any(i.get("exercise") == "深蹲" for i in issues)


class TestHoldGuidanceMatchesTheReason:
    """拦截后的指引必须与**拦截原因**一致。

    这组测试来自一次实测事故：用户没有伤病，计划因为用了一根他没有的
    TRX 弹力带被扣下，而界面告诉他"去骨科或运动医学科评估"——
    因为缺根带子去看医生。原因是 next_step 当时写死成了伤病场景。
    """

    def _guidance(self, **result):
        from src.core.plan_finalization import hold_guidance
        return hold_guidance(result)

    # ---- 截图里的真实载荷，逐字照抄 ----
    EQUIPMENT_CASE = {
        "review_reason": "器械约束不满足：用户可用器械为哑铃、自重、弹力带，未包含TRX/悬挂训练带",
        "active_issues": [
            {"exercise": "TRX双手划船",
             "issue": "器械约束不满足：用户可用器械为哑铃、自重、弹力带，未包含TRX/悬挂训练带；建议替换为俯身哑铃划船或弹力带划船。",
             "severity": "warning"},
        ],
    }

    def test_equipment_hold_does_not_send_the_user_to_a_doctor(self):
        prohibitions, next_step = self._guidance(**self.EQUIPMENT_CASE)
        assert "骨科" not in next_step and "就医" not in next_step, (
            f"器械问题不该让人去看医生：{next_step}")
        assert "器械" in next_step, "应当指向器械清单"

    def test_equipment_hold_allows_replacing_exercises(self):
        """器械问题的解法就是替换动作，此时禁止替换是自相矛盾的。

        实测中同一屏上出现过「建议替换为俯身哑铃划船」与
        「不要自行替换或增加动作」并存。
        """
        prohibitions, _ = self._guidance(**self.EQUIPMENT_CASE)
        assert not any("替换" in p for p in prohibitions), (
            f"器械问题不应禁止替换动作：{prohibitions}")
        assert any("不要按原样执行" in p for p in prohibitions), "仍要提醒别照原样练"

    def test_injury_hold_still_sends_the_user_to_a_doctor(self):
        prohibitions, next_step = self._guidance(
            review_reason="语义匹配检测到伤病风险 (置信度 0.76)",
            active_issues=[{"issue": "[语义匹配] 你的描述与「膝关节骨关节炎」的常见症状相似",
                            "severity": "danger"}])
        assert "骨科" in next_step or "运动医学" in next_step
        assert any("替换" in p for p in prohibitions), "伤病问题必须禁止自行替换"

    def test_equipment_wording_mentioning_injury_is_still_equipment(self):
        """器械类文案由大模型生成，句中出现"伤病"二字不能把它判成医疗问题。

        否则又回到"缺根带子去看骨科"。真实的伤病问题一定带确定性信号
        （规则引擎与语义匹配都有固定前缀），因此不会因此漏判。
        """
        from src.core.plan_finalization import classify_hold
        assert classify_hold({
            "review_reason": "器械约束不满足：计划需要TRX，考虑到你的伤病情况建议改用弹力带",
            "active_issues": [{"issue": "器械约束不满足：未包含TRX；该动作与你的伤病无关"}],
        }) == "equipment"

    def test_strong_injury_signals_win_over_equipment_wording(self):
        """真的伤病不能被器械字样盖过去——那才是会让人受伤的方向。"""
        from src.core.plan_finalization import classify_hold
        assert classify_hold({
            "review_reason": "规则引擎检测到 1 个伤病-动作冲突",
            "active_issues": [
                {"issue": "[规则引擎] 伤病「腰突」与动作「硬拉」存在冲突", "severity": "danger"},
                {"issue": "器械约束不满足：未包含TRX"},
            ],
        }) == "injury"

    def test_unknown_exercises_map_to_naming(self):
        from src.core.plan_finalization import classify_hold
        assert classify_hold({
            "review_reason": "计划包含动作库中不存在的动作",
            "unknown_exercises": ["幽灵动作"],
        }) == "naming"


class TestPayloadActuallyUsesTheGuidance:
    """载荷必须真的用上分类结果——只测分类函数是不够的。

    第一版测试直接调 hold_guidance()，因此把 build_review_pending_payload
    改回写死文案时它们依然全绿：测了零件，没测接线。下面这几条从载荷
    入口进去断言，才真正锁住"用户看到的那句话与拦截原因一致"。
    """

    def _payload(self, result_extra):
        from src.core.plan_finalization import (
            build_review_pending_payload,
            create_review_artifact,
        )

        class _Store:
            def create(self, **kwargs):
                class _A:
                    review_id = "r-wire"
                    status = "pending"
                    created_at = "2026-10-09T00:00:00Z"
                    issues = kwargs.get("issues", [])
                    severity = kwargs.get("severity", "warning")
                    prohibited_actions = kwargs.get("prohibited_actions", [])
                return _A()

        result = {"days": [], "active_issues": [], "review_suggestions": [],
                  "review_severity": "warning", "review_reason": ""}
        result.update(result_extra)
        artifact = create_review_artifact(_Store(), {"goal": "增肌"}, "q", result)
        return build_review_pending_payload(result, artifact)

    def test_equipment_payload_next_step_is_about_equipment(self):
        payload = self._payload({
            "review_reason": "器械约束不满足：未包含TRX/悬挂训练带",
            "active_issues": [{"exercise": "TRX双手划船",
                               "issue": "器械约束不满足：未包含TRX/悬挂训练带",
                               "severity": "warning"}],
        })
        step = payload["review"]["next_step"]
        assert "骨科" not in step, f"器械问题不该让人看医生：{step}"
        assert "器械" in step

    def test_equipment_payload_prohibitions_allow_replacement(self):
        payload = self._payload({
            "review_reason": "器械约束不满足：未包含TRX",
            "active_issues": [{"issue": "器械约束不满足：未包含TRX", "severity": "warning"}],
        })
        assert not any("替换" in p for p in payload["review"]["prohibited_actions"])

    def test_injury_payload_next_step_is_medical(self):
        payload = self._payload({
            "review_reason": "语义匹配检测到伤病风险 (置信度 0.76)",
            "active_issues": [{"issue": "[语义匹配] 与「膝关节骨关节炎」症状相似",
                               "severity": "danger"}],
        })
        assert "骨科" in payload["review"]["next_step"] or "运动医学" in payload["review"]["next_step"]
        assert any("替换" in p for p in payload["review"]["prohibited_actions"])

    def test_payload_reports_the_hold_category(self):
        """载荷带上分类，便于前端或排查时直接看出是哪一类。"""
        payload = self._payload({
            "review_reason": "器械约束不满足：未包含TRX",
            "active_issues": [{"issue": "器械约束不满足：未包含TRX"}],
        })
        assert payload["review"]["hold_reason"] == "equipment"


# ----------------------------------------------------------------------
# 交付层剥离：内部标记与术语不能落到用户屏幕上
# ----------------------------------------------------------------------

def test_delivery_strips_internal_prefix_and_terms():
    """交付载荷里不该出现 [规则引擎] / （触发词: X）/ 建议人工审核。

    这些是给内部路由和排查用的：用户看到"系统按字面比对了一个『背』字"
    比看到一句人话更容易怀疑整套判断；"建议人工审核"则是内部流程术语，
    而这一屏本来就写着"需要专业人员确认"。
    """
    from src.core.plan_finalization import _strip_internal_markers

    raw = ("[规则引擎] 伤病「下背有点酸」与动作「哑铃单臂划船」"
           "存在冲突（触发词: 背），建议人工审核")
    cleaned = _strip_internal_markers(raw)
    assert "[规则引擎]" not in cleaned
    assert "触发词" not in cleaned
    assert "建议人工审核" not in cleaned
    # 实质信息必须留下——剥离的是术语，不是结论
    assert "下背有点酸" in cleaned
    assert "哑铃单臂划船" in cleaned
    assert "冲突" in cleaned


def test_delivery_stripping_also_applies_to_issue_dicts():
    """issue 既可能是字符串也可能是 {issue: ..., severity: ...} 字典。"""
    from src.core.plan_finalization import _strip_internal_markers

    cleaned = _strip_internal_markers(
        {"issue": "[规则引擎] 冲突（触发词: 腰），建议人工审核", "severity": "danger"})
    assert cleaned["severity"] == "danger"
    assert "触发词" not in cleaned["issue"]
    assert "规则引擎" not in cleaned["issue"]


def test_reason_line_is_stripped_too():
    """「风险原因」那一行比明细更显眼，兜底路径也必须剥离。"""
    import inspect
    from src.core import plan_finalization as pf

    src = inspect.getsource(pf.build_review_pending_payload)
    assert "reason = _strip_internal_markers(reason)" in src, \
        "兜底 reason 直接取 issue 原文，会带内部标记，必须剥离"


class TestMobilityLimitationOnlyAdjustsNotBlocks:
    """功能性受限（活动度/柔韧/僵硬）只给做法调整，不阻断交付。

    本组来自一次实测事故：用户在伤病史填「肩膀活动度不好」「脚踝活动度不好」，
    计划被「暂缓交付」，理由是"发现 6 处动作与你的伤病史冲突"——而报出的冲突里，
    触发词是「肩」和「踝」两个**部位词**。

    三重错误叠在一起才致命：
      1. ``INJURY_EXERCISE_CONFLICTS`` 的键全是部位词，而部位词不是伤病——
         「我想练肩」同样命中「肩」；
      2. 活动度差不是禁忌，它恰恰是训练要改善的东西。系统自己算出的处置写着
         "建议限制幅度…或改为绳索夹胸"，内容完全正确，只是被错误地升级成了
         阻断理由；
      3. 本项目的 HITL 没有"审核人"角色，扣下即死锁——用户永远拿不到计划，
         也没有任何人能放行。这才是"体验差"的真正来源。

    加严的那一面同样要守住：真损伤照拦不误。放行的唯一条件是描述里明确
    出现"能力不足"语义（活动度/活动范围/柔韧/僵硬…）且不含任何诊断词。
    """

    @pytest.fixture(autouse=True)
    def _offline_semantic(self, monkeypatch):
        """规则层用例不得依赖 embedding 服务。

        把语义匹配钉成"无命中"，断言才只反映确定性规则层；语义层本身用桩
        单独测（见 TestSemanticLayerRespectsLimitationNature）。
        """
        monkeypatch.setattr(HITLReview, "_match_semantic", lambda self, text: [])

    def _check(self, injuries, exercises):
        review = HITLReview()
        plan = {"days": [{"exercises": [{"name": n} for n in exercises]}]}
        decision = review.check(
            {"is_safe": True, "issues": [], "confidence": 0.9},
            plan=plan, profile={"injuries": injuries},
        )
        return review, decision

    @pytest.mark.parametrize("text,expected", [
        ("肩膀活动度不好", "limitation"),
        ("脚踝活动度不好", "limitation"),
        ("髋关节灵活性差，柔韧性不好", "limitation"),
        ("肩袖撕裂", "structural"),
        ("半月板撕裂，活动度受限", "structural"),  # 诊断词压制受限词
        ("我肩膀疼", "structural"),                  # 只有症状，保持从严
        ("腰椎间盘突出", "structural"),
    ])
    def test_nature_classification(self, text, expected):
        assert classify_injury_nature(text) == expected

    def test_limitation_no_longer_produces_blocking_conflicts(self):
        """实测原例：这两条描述曾报出 5 条 danger 级冲突。"""
        review = HITLReview()
        plan = {"days": [
            {"exercises": [{"name": "慢离心卧推"}, {"name": "上斜哑铃飞鸟"}, {"name": "坐姿推举"}]},
            {"exercises": [{"name": "杠铃深蹲"}, {"name": "提踵"}]},
        ]}
        issues = review._check_conflicts(
            plan, {"injuries": ["肩膀活动度不好", "脚踝活动度不好"]})
        assert issues == [], f"活动度不好不应再产生阻断级冲突: {issues}"

    def test_limitation_still_produces_actionable_cautions(self):
        """不阻断 ≠ 不管。用户要的是"在注意这些的前提下继续生成"。"""
        _review, decision = self._check(["肩膀活动度不好"], ["慢离心卧推"])
        assert decision.needs_review is False
        assert decision.cautions, "受限必须留下做法调整建议"
        assert decision.cautions[0].startswith("[调整]")
        assert "无痛范围" in decision.cautions[0]

    @pytest.mark.parametrize("injury,exercise", [
        ("肩袖撕裂", "慢离心卧推"),
        ("腰椎间盘突出", "杠铃深蹲"),
        ("脚踝扭伤", "杠铃深蹲"),
    ])
    def test_structural_injury_still_blocks(self, injury, exercise):
        """放行受限，绝不能把真损伤一起放行。"""
        _review, decision = self._check([injury], [exercise])
        assert decision.needs_review is True, f"{injury} 必须仍然阻断"
        assert decision.severity == "danger"
        assert decision.suggestions

    def test_mixed_injuries_block_but_keep_limitation_cautions(self):
        """损伤与受限同时存在：以损伤为准阻断，受限的建议也得留着。"""
        _review, decision = self._check(
            ["脚踝活动度不好", "腰椎间盘突出"], ["杠铃深蹲"])
        assert decision.needs_review is True
        assert decision.cautions, "受限那部分仍要给出做法调整"


class TestSemanticLayerRespectsLimitationNature:
    """语义层不得把"活动度不好"读成疾病诊断。

    实测：用户写「肩膀活动度不好」→ 与 shoulder-impingement 档案相似度 0.694
    （danger）；「脚踝活动度不好」→ ankle-sprain 0.703。系统因此判了 danger，
    在用户已经被规则引擎拦下的基础上补了第二刀——只修一层不够，两层都要改。

    这里用桩固定语义匹配结果，避免测试真的去调 embedding 服务。
    """

    _MATCH = [{"profile_id": "shoulder-impingement", "similarity": 0.694,
               "banned_exercises": ["推举", "卧推", "飞鸟"], "severity": "danger"}]

    def _decide(self, monkeypatch, injuries, matches=None):
        review = HITLReview()
        monkeypatch.setattr(
            review, "_match_semantic",
            lambda text: [dict(m) for m in (matches or self._MATCH)])
        return review.check(
            {"is_safe": True, "issues": [], "confidence": 0.9},
            plan={"days": []}, profile={"injuries": injuries},
        )

    def test_limitation_match_is_downgraded_to_caution(self, monkeypatch):
        decision = self._decide(monkeypatch, ["肩膀活动度不好"])
        assert decision.needs_review is False, "活动度不该被当成肩峰撞击来拦"
        assert decision.cautions and decision.cautions[0].startswith("[调整]")

    def test_structural_match_still_blocks(self, monkeypatch):
        decision = self._decide(monkeypatch, ["肩袖撕裂"])
        assert decision.needs_review is True
        assert decision.severity == "danger"

    def test_best_match_is_picked_by_similarity_not_table_order(self, monkeypatch):
        """注释一直写"取相似度最高的匹配"，代码取的却是档案表顺序第一个。

        两条档案同时命中时，扣不扣计划取决于档案表的排列顺序——这是隐式的，
        任何一次档案重排都会悄悄改变安全行为。这里把口径钉在相似度上。
        """
        decision = self._decide(monkeypatch, ["肩袖撕裂"], matches=[
            {"profile_id": "low-sim", "similarity": 0.70,
             "banned_exercises": ["深蹲"], "severity": "warning"},
            {"profile_id": "high-sim", "similarity": 0.95,
             "banned_exercises": ["推举"], "severity": "danger"},
        ])
        assert "推举" in " ".join(decision.suggestions), "应取相似度最高的档案"


class TestCautionsRideAlongWithoutGating:
    """cautions 必须进 warnings 随计划交付，且**不参与任何闸门**。

    这是本次修复的最后一道：判定层放行了，如果 finalize 又把它当问题清单拿去
    算 hard_ok，计划照样交付不出去，等于没修。
    """

    # 形状必须过 TrainingPlanOutput：schema 不过时 schema_valid 为假，
    # _delivery_allowed 本来就该是 False，那条断言会变成假阴性。
    _PLAN = {
        "plan_name": "一周全身训练", "goal": "增肌", "weeks": None,
        "sessions_per_week": 3,
        "days": [{"day": 1, "focus": "全身", "exercises": [
            {"name": "慢离心卧推", "sets": 4, "reps": "8-12", "rest": "90秒"}]}],
        "notes": "循序渐进",
        "plan_id": "plan-caution-test", "user_id": 1,
    }
    _CAUTION = "[调整] 「肩膀活动度不好」不算禁忌，计划照常安排；建议降低活动幅度"

    def _finalize(self, checks):
        return finalize_result(dict(self._PLAN), checks, 0)

    def test_cautions_are_delivered_as_warnings(self):
        final = self._finalize([{"is_safe": True, "issues": [], "confidence": 0.9,
                                 "review_cautions": [self._CAUTION]}])
        assert final["cautions"] == [self._CAUTION]
        assert any("[调整]" in w for w in final["warnings"]), "必须随计划交付给用户"

    def test_cautions_do_not_block_delivery(self):
        """关键断言：带 cautions 的计划必须照常交付。"""
        final = self._finalize([{"is_safe": True, "issues": [], "confidence": 0.9,
                                 "review_cautions": [self._CAUTION]}])
        assert final["requires_review"] is False
        assert final["_delivery_allowed"] is True

    def test_structural_hold_is_unaffected(self):
        """有真拦截时，cautions 不得把闸门顶开。"""
        final = self._finalize([{"is_safe": False, "issues": [{"issue": "危险"}],
                                 "confidence": 0.9,
                                 "review_cautions": ["[调整] 仅供参考"]}])
        assert final["requires_review"] is True
        assert final["_delivery_allowed"] is False
