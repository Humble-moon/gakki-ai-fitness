"""运动急症的检测与话术回归。

补这组用例的动因是一次真实执行：输入「训练时突然胸闷、头晕，眼前发黑，
还要继续练吗？」时，系统答的是呼吸技巧（别憋气、瓦式呼吸），没有建议就医。
根因是安全词表只有"疼/痛/伤"一类词——"胸闷"不含"痛"字，一个词都没命中，
安全话术根本没注入，模型把它当普通提问处理了。

这类漏报的代价是延误就医，所以这里除了测"急症必须触发"，还要测
"急症话术不得与普通安全话术叠加"：两者对"能否给训练建议"的要求相反，
同时注入会让模型收到自相矛盾的约束。

**用例里的不可见字符一律用 \\u 转义书写**：写成字面量的话，一旦在复制、
格式化或编辑器保存中丢失，字符串会退化成"胸闷"本身——用例仍然通过，
却静默失去了对绕过手法的覆盖。
"""

from __future__ import annotations

import pytest

from src.core.qa_safety import (
    EMERGENCY_KEYWORDS,
    EMERGENCY_NOTE,
    EMERGENCY_SYSTEM_MESSAGE,
    SAFETY_NOTE,
    SAFETY_SYSTEM_MESSAGE,
    build_emergency_messages,
    build_safety_messages,
    detect_emergency,
    detect_safety_concern,
)


class TestEmergencyDetection:
    """急症信号必须被识别。"""

    @pytest.mark.parametrize(
        "question",
        [
            # 心血管
            "训练时我突然胸闷，还要继续吗",
            "练完心慌得厉害",
            "感觉心悸，是不是正常",
            "心跳加速停不下来",
            # 神经 / 脑供血
            "训练时头晕",
            "突然眩晕站不稳",
            "眼前发黑",
            "刚才差点晕倒",
            "晕厥过一次",
            "意识模糊",
            # 呼吸
            "呼吸困难",
            "喘不上气",
            "呼吸急促得吓人",
            # 其它体征
            "出冷汗",
            "脸色发白",
            "嘴唇发紫",
        ],
    )
    def test_emergency_signal_is_detected(self, question):
        assert detect_emergency(question) is True, f"急症信号漏检: {question}"

    def test_every_documented_keyword_is_effective(self):
        """词表里写了的词必须真的能命中，防止加词时写错字。"""
        for kw in EMERGENCY_KEYWORDS:
            assert detect_emergency(f"我有点{kw}") is True, f"{kw} 未触发"

    def test_specific_real_world_case_from_test_run(self):
        """实测中漏报的那条原句，必须命中。"""
        assert detect_emergency("训练时我突然胸闷、头晕，眼前发黑，还要继续练吗？")


class TestEmergencyAlsoTriggersSafetyConcern:
    """兜底：急症词虽独立成表，也必须让 detect_safety_concern 返回 True。

    否则将来新增的调用点若只判 detect_safety_concern，急症就会从那条路径漏掉。
    """

    @pytest.mark.parametrize(
        "question",
        ["训练时胸闷", "突然头晕", "心悸得厉害", "喘不上气"],
    )
    def test_emergency_implies_safety_concern(self, question):
        assert detect_safety_concern(question) is True


class TestEmergencyEvasion:
    """绕过手法对急症词同样失效——归一化是共用流水线。"""

    @pytest.mark.parametrize(
        "raw",
        [
            "胸​闷",  # ZERO WIDTH SPACE
            "胸‍闷",  # ZERO WIDTH JOINER
            "胸­闷",  # SOFT HYPHEN
            "胸 闷",  # CJK 间空格
            "胸\U0001F60A闷",  # emoji
            "胸(xiong)闷",  # 括号注音
        ],
    )
    def test_evasion_does_not_bypass_emergency(self, raw):
        assert detect_emergency(raw) is True, f"绕过写法逃逸: {raw!r}"


class TestNoFalsePositive:
    """正常训练问题不应被误判为急症。

    这些词是保守选取的高置信度急症信号，普通提问不该命中。
    """

    @pytest.mark.parametrize(
        "question",
        [
            "增肌怎么吃",
            "深蹲的姿势要领是什么",
            "一周练几次比较合适",
            "硬拉和深蹲哪个更练背",
            "训练容量怎么算",
        ],
    )
    def test_normal_question_is_not_emergency(self, question):
        assert detect_emergency(question) is False, f"误报为急症: {question}"


class TestMessageExclusivity:
    """急症话术与普通安全话术必须互斥且各自自洽。"""

    def test_emergency_message_demands_immediate_care(self):
        content = build_emergency_messages()[0]["content"]
        assert content == EMERGENCY_SYSTEM_MESSAGE
        assert "立即停止" in content
        assert "就医" in content

    def test_emergency_note_forbids_training_advice(self):
        assert "不要提供任何训练动作" in EMERGENCY_NOTE

    def test_two_notes_are_not_interchangeable(self):
        """两种话术对'能否给训练建议'的要求相反，不能混用。"""
        assert EMERGENCY_NOTE != SAFETY_NOTE
        # 安全话术允许康复建议，急症话术禁止训练建议——这是二者不可叠加的原因
        assert "运动康复" in SAFETY_SYSTEM_MESSAGE
        assert "不要提供任何训练动作" in EMERGENCY_NOTE

    def test_builders_return_system_role(self):
        for msg in (build_emergency_messages(), build_safety_messages()):
            assert len(msg) == 1
            assert msg[0]["role"] == "system"


class TestPlannerSafetyGate:
    """Planner 的安全闸门也必须覆盖急症词。

    这是第三张安全词表：`qa_safety` 管问答话术、`review.py` 管规则冲突，
    而 `PlannerAgent.SAFETY_OVERRIDE` 管**计划生成入口的路由**。它原先同样
    只有伤病词（间盘/腰突/疼/痛…），"胸闷"连"痛"字都没有，命不中任何一条，
    于是用户带着急症症状请求生成计划时，系统会照常拆解训练任务。
    """

    def test_planner_reuses_the_shared_emergency_list(self):
        """复用单一事实源，避免第三张表各自漂移。"""
        from src.agents.planner import PlannerAgent
        from src.core.qa_safety import EMERGENCY_KEYWORDS

        assert set(PlannerAgent.EMERGENCY_OVERRIDE) == set(EMERGENCY_KEYWORDS)

    @pytest.mark.parametrize(
        "text",
        ["我训练时胸闷", "最近老头晕", "帮我做个计划，偶尔心悸", "喘不上气怎么练"],
    )
    def test_emergency_text_hits_planner_gate(self, text):
        from src.agents.planner import PlannerAgent

        assert any(kw in text for kw in PlannerAgent.EMERGENCY_OVERRIDE)

    def test_injury_words_still_hit_planner_gate(self):
        """原有伤病词能力不得因这次改动丢失。"""
        from src.agents.planner import PlannerAgent

        for text in ["我腰椎间盘突出", "膝盖疼", "半月板损伤", "刚做完手术"]:
            assert any(kw in text for kw in PlannerAgent.SAFETY_OVERRIDE), text


class TestEmergencyStopsPlanDelivery:
    """急症信号必须阻止计划交付，而不只是改变拆解方向。

    上面那组用例只验到「Planner 认得急症词、会把 skill 换成 exercise_analysis」。
    可换 skill 只是换了拆解方向：下游照常检索、生成、检查，最后交付一份完整
    训练计划。实测把「训练的时候突然胸闷、头晕、眼前发黑，还能继续练吗」发到
    计划接口，拿到的是 **4 天增肌计划**（delivery_status=safe_delivered，
    requires_review=False），而同一句话发到问答入口会被明确挡下并建议就医。
    同一条安全口径，两条路径一处生效一处失效。

    这一组钉住交付侧：命中急症就不给任何训练内容，也不进审核队列——
    审核的语义是「确认之后可能放行」，而急症不是「等一等就能练」的情况。
    """

    EMERGENCY_QUERY = "我训练的时候突然胸闷、头晕、眼前发黑，还能继续练吗"

    def _state(self, query="", injuries=None, result=None):
        return {
            "query": query,
            "profile": {"goal": "增肌", "injuries": injuries or []},
            "result": result or {
                "days": [{"day": 1, "exercises": [{"name": "杠铃卧推"}]}],
            },
            "session_id": "emergency-session",
            "athlete_key": "emergency-athlete",
        }

    def test_deliver_node_withholds_the_plan(self):
        """急症 query 不得产出可执行计划。"""
        from types import SimpleNamespace

        from src.core.plan_finalization import EMERGENCY_STOP_STATUS
        from src.graph.nodes import deliver_node

        payload = deliver_node(
            SimpleNamespace(), self._state(query=self.EMERGENCY_QUERY))["final_payload"]

        assert payload["delivery_status"] == EMERGENCY_STOP_STATUS
        assert payload["delivery_status"] != "safe_delivered"
        assert payload["days"] == [], "急症终态不得携带任何训练内容"
        assert payload["emergency"]["matched_signals"], "要能说清命中了哪句话"
        assert payload["requires_review"] is False, "急症不是「等审核」的情形"

    def test_injuries_field_also_stops_delivery(self):
        """症状写在伤病史字段里同样要拦——只看 query 会漏掉一半入口。"""
        from types import SimpleNamespace

        from src.graph.nodes import deliver_node

        payload = deliver_node(
            SimpleNamespace(), self._state(injuries=["最近训练时胸闷心慌"]))["final_payload"]

        assert payload["delivery_status"] == "emergency_stopped"
        assert payload["days"] == []

    def test_ordinary_injury_is_not_treated_as_emergency(self):
        """伤病不得被误判为急症：那会把「换动作继续练」变成「一律不生成」。"""
        from types import SimpleNamespace

        from src.graph.nodes import deliver_node

        payload = deliver_node(SimpleNamespace(), self._state(
            query="深蹲的时候膝盖有点疼",
            result={"requires_review": True, "days": []}))["final_payload"]

        assert payload["delivery_status"] == "review_pending"

    def test_persist_is_blocked_for_emergency(self):
        """急症结果不得写入缓存与画像。

        缓存会被后续相同请求命中、把计划捞回来，等于绕开这道拦截；画像记的是
        训练偏好，一段急症描述不是偏好。用会抛异常的替身证明这条路径确实
        一个存储都没碰。
        """
        from src.core.plan_finalization import persist_if_safe

        class Untouchable:
            def __getattr__(self, name):
                raise AssertionError(f"急症路径不该触碰存储:{name}")

        wrote = persist_if_safe(
            Untouchable(), Untouchable(), Untouchable(),
            {"goal": "增肌"},
            self.EMERGENCY_QUERY,
            {"_persistence_allowed": True, "_delivery_allowed": True},
        )

        assert wrote is False

    def test_orchestrator_delivery_gate_prefers_emergency(self):
        """命令式后端的交付判定走同一个函数，必须同样先看急症。"""
        from src.core.plan_finalization import review_pending_result

        class NoArtifact:
            def create(self, *args, **kwargs):
                raise AssertionError("急症不该创建审核工件")

        payload = review_pending_result(
            NoArtifact(), {"goal": "增肌", "injuries": []},
            self.EMERGENCY_QUERY, {"days": [], "requires_review": True})

        assert payload["delivery_status"] == "emergency_stopped"

    def test_normal_query_is_unaffected(self):
        """回归保护：正常请求仍按原路交付。"""
        from types import SimpleNamespace

        from src.graph.nodes import deliver_node

        # 交付分支会调用持久化；这个结果的闸门标志都是假值，
        # 所以三个存储只被引用、不会被真的写入。
        deps = SimpleNamespace(cache=None, conversation=None, long_term=None)
        payload = deliver_node(
            deps, self._state(query="想练胸和背，每次 60 分钟"))["final_payload"]

        assert payload["days"], "正常请求必须照常交付计划"
        assert payload["delivery_status"] == "safe_delivered"

    def test_router_sends_emergency_straight_to_delivery(self):
        """急症不进检索与生成。

        拦在交付层也能保证"不给计划"，但用户要等一次完整生成（实测 60~90 秒）
        才看到「请先就医」——对一条就医提示来说，这个等待没有理由。
        """
        from src.graph.routing import route_after_cache

        assert route_after_cache({"emergency_signals": ["胸闷"]}) == "deliver"
        # 反向：普通请求的两条老路不能被这条分流吃掉
        assert route_after_cache({"emergency_signals": [], "cache_hit": None}) == "plan"
        assert route_after_cache(
            {"emergency_signals": [], "cache_hit": {"days": []}}) == "deliver_cached"

    def test_emergency_outranks_cache_hit(self):
        """缓存命中不能把急症挡回去：缓存的同样是训练内容。"""
        from src.graph.routing import route_after_cache

        assert route_after_cache({
            "emergency_signals": ["心悸"],
            "cache_hit": {"days": [{"day": 1, "exercises": [{"name": "深蹲"}]}]},
        }) == "deliver"

    def test_ingest_records_the_signal_once_for_every_consumer(self):
        """信号在 ingest 定下来，路由、交付、持久化看的是同一个判断。"""
        from types import SimpleNamespace

        from src.graph.nodes import ingest_node

        deps = SimpleNamespace(
            conversation=SimpleNamespace(
                add_turn=lambda *a, **k: None,
                build_context_for_prompt=lambda *a, **k: "",
                get_plan_state=lambda *a, **k: "",
            ),
            long_term=SimpleNamespace(build_context_for_prompt=lambda *a, **k: ""),
        )

        hit = ingest_node(deps, {"query": self.EMERGENCY_QUERY,
                                 "profile": {"goal": "增肌"}})
        assert hit["emergency_signals"], "ingest 必须记下令中的信号"

        # 反向：普通请求不能带上信号，否则每条计划都会被拦下
        clean = ingest_node(deps, {"query": "想练胸和背",
                                   "profile": {"goal": "增肌"}})
        assert clean["emergency_signals"] == []
