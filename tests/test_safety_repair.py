"""伤病冲突的自动修复链路。

规则引擎一直能查出「硬拉与腰椎间盘突出冲突」，但过去它的处理方式是
把整份计划扣下等人放行——而这个项目里没有审核人。现在改成：冲突动作
交给重写循环替换，替换不掉才拒绝。

这组用例守三件事：
1. 冲突要变成**可执行**的重写指令（带动作名），而不是一句"建议人工审核"；
2. 冲突存在时必须置 is_safe=False，否则重写循环根本不会启动；
3. 修不掉时终态是拒绝，并且说得出是哪个动作。
"""

from src.agents.fact_checker import FactCheckerAgent
from src.core.plan_finalization import finalize_result, safety_blocked_payload
from src.hitl.review import HITLReview


class _FakeLLM:
    """只返回一份固定的 LLM 审查结果，不联网。"""

    def __init__(self, payload=None):
        self.payload = payload or {
            "is_safe": True, "issues": [], "confidence": 0.9,
            "requires_human_review": False,
        }

    def chat_with_json_mode(self, messages, model=None):
        return dict(self.payload)


def _checker(payload=None):
    checker = FactCheckerAgent.__new__(FactCheckerAgent)
    checker.hitl = HITLReview()
    checker.llm = _FakeLLM(payload)
    return checker


def _plan(*exercise_names):
    # 字段必须齐全：finalize 会跑 validate_training_plan，缺 plan_id / weeks
    # 这类字段的计划会被判成结构不合格，与安全无关。
    return {
        "plan_id": "plan-1", "user_id": 1, "goal": "增肌", "weeks": 4,
        "sessions_per_week": 1,
        "days": [{"day": 1, "focus": "全身",
                  "exercises": [{"name": n, "sets": 3, "reps": "8-12",
                                 "rest": "60秒"}
                                for n in exercise_names]}],
    }


LUMBAR = {"goal": "增肌", "injuries": ["腰椎间盘突出"]}


def test_conflict_becomes_a_repair_instruction():
    """规则引擎的冲突必须变成"替换这个动作"，而不是"建议人工审核"。"""
    result = _checker().check(_plan("杠铃硬拉", "哑铃卧推"), LUMBAR)

    assert result["conflict_exercises"] == ["杠铃硬拉"]
    # 关键：必须让重写循环看到不安全，否则它根本不会跑
    assert result["is_safe"] is False
    # 还有机会修，这一轮不送审
    assert result["requires_human_review"] is False

    repair_issues = [i for i in result["issues"]
                     if i.get("severity") == "danger" and i.get("exercise")]
    assert repair_issues, "冲突必须带着动作名进入 issues，重写才知道改哪个"
    assert repair_issues[0]["exercise"] == "杠铃硬拉"
    assert "替换" in repair_issues[0]["issue"]
    assert "人工审核" not in repair_issues[0]["issue"]


def test_clean_plan_has_no_conflicts():
    """反向保护：没有伤病、没有冲突的计划不该被这条链路影响。

    注意这里不能沿用 LUMBAR——语义匹配会把「腰椎间盘突出」与任何下肢/核心
    相关描述挂钩，那是另一套机制，不是本文件要验证的点。
    """
    result = _checker().check(_plan("哑铃卧推"), {"goal": "增肌", "injuries": []})

    assert result["conflict_exercises"] == []
    assert result["is_safe"] is True
    assert result["requires_human_review"] is False


def test_unrepairable_conflict_is_refused_with_specifics():
    """冲突没修掉时，终态要说得出是哪个动作、哪条伤病。"""
    check = _checker().check(_plan("杠铃硬拉"), LUMBAR)
    result = finalize_result(_plan("杠铃硬拉"), [check], 3)

    # 重写到预算用尽仍在冲突 —— finalize 应当把它标成安全阻断
    assert result["_safety_block"] is not None
    assert result["_delivery_allowed"] is False

    payload = safety_blocked_payload(result)
    assert payload["delivery_status"] == "safety_blocked"
    assert payload["days"] == []
    assert "杠铃硬拉" in payload["safety"]["message"]
    assert "腰椎间盘突出" in payload["safety"]["message"]
    assert payload["safety"]["next_step"]


def test_conflict_is_repaired_by_rewrite_then_delivered():
    """完整链路：规则引擎检出冲突 → 重写换掉动作 → 计划正常交付。

    这条用例走真实的 HITLReview，不喂预设的检查结果——验证的正是"冲突
    能自己修好"，而不是"假装修好了"。
    """
    from tests.test_graph.test_graph_flows import invoke_plan, make_profile, make_runtime

    # 重写结果是"哑铃卧推 + 绳索夹胸"：两者都不在腰突的禁忌表里。
    # 换「划船」不行——划船对腰椎有压力，规则引擎会照样判冲突，那验证的
    # 就不是修复成功而是修复失败了。
    runtime = make_runtime(plan=_plan("杠铃硬拉", "哑铃卧推"),
                           rewrite_out=_plan("哑铃卧推", "绳索夹胸"),
                           fact_checker=_checker())
    state, _, _ = invoke_plan(runtime, make_profile(injuries=["腰椎间盘突出"]))

    assert state["rewrite_count"] >= 1, "冲突没有触发重写"
    payload = state["final_payload"]
    assert payload["delivery_status"] == "safe_delivered"
    names = [ex["name"] for day in payload["days"] for ex in day["exercises"]]
    assert "杠铃硬拉" not in names, "冲突动作被留在了交付的计划里"


def test_prune_removes_only_the_conflicting_exercise():
    """剪枝只摘冲突的那一个，其他动作原样保留。"""
    from src.core.plan_finalization import prune_conflicting_exercises

    plan = _plan("杠铃硬拉", "哑铃卧推", "绳索夹胸")
    pruned, removed = prune_conflicting_exercises(plan, ["杠铃硬拉"])

    assert removed == ["杠铃硬拉"]
    names = [ex["name"] for day in pruned["days"] for ex in day["exercises"]]
    assert names == ["哑铃卧推", "绳索夹胸"]


def test_prune_is_noop_when_name_is_absent():
    """名单里的动作不在计划里时不做任何改动——避免误删同名子串。"""
    from src.core.plan_finalization import prune_conflicting_exercises

    plan = _plan("哑铃卧推")
    pruned, removed = prune_conflicting_exercises(plan, ["杠铃硬拉"])

    assert removed == []
    assert pruned["days"][0]["exercises"] == plan["days"][0]["exercises"]


def test_prune_then_recheck_clears_the_conflict():
    """剪掉冲突动作后复检必须干净——这是"少两个动作总比空手强"的依据。"""
    from src.core.plan_finalization import prune_conflicts_with_recheck

    check = _checker().check(_plan("杠铃硬拉", "哑铃卧推"), LUMBAR)
    assert check["conflict_exercises"] == ["杠铃硬拉"]

    plan, after, pruned = prune_conflicts_with_recheck(
        _plan("杠铃硬拉", "哑铃卧推"), check,
        lambda p: _checker().check(p, LUMBAR))

    assert pruned == ["杠铃硬拉"]
    assert after["conflict_exercises"] == []
    assert after["is_safe"] is True


def test_repair_instruction_reaches_the_rewrite_prompt():
    """重写提示词里必须出现冲突动作名——否则模型不知道该改哪个。"""
    from src.agents.writer import WriterAgent

    captured = {}

    class _CapturingLLM:
        def chat_with_json_mode(self, messages, model=None):
            captured["prompt"] = "\n".join(m.get("content", "") for m in messages)
            return dict(_plan("哑铃卧推"))

    writer = WriterAgent.__new__(WriterAgent)
    writer.llm = _CapturingLLM()

    check = _checker().check(_plan("杠铃硬拉"), LUMBAR)
    writer.rewrite_plan(_plan("杠铃硬拉"), check["issues"], {"exercises": []}, LUMBAR)

    prompt = captured["prompt"]
    assert "杠铃硬拉" in prompt, "重写提示词没有点名冲突动作"
    assert "替换" in prompt
