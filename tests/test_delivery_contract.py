"""交付闸门的跨函数不变量。

这组缺陷有一个共同特征：每个函数的单点测试都是绿的，问题出在两个判据
之间——内部已经算出"不允许交付"，接口却回"可以安全交付"。只测单个函数
永远抓不到，所以这里断言的是跨函数的契约。

三个不变量：
1. 结构校验没过 → 不得标成 safe_delivered，也不得带出可执行计划。
2. 缓存命中 → 必须带 delivery_status（否则前端整屏留白）。
3. 急症 → 必须先于缓存判定，两条编排入口一致。
"""

from types import SimpleNamespace

from src.core.orchestrator import Orchestrator
from src.core.plan_finalization import (
    EMERGENCY_STOP_STATUS,
    GENERATION_FAILED_STATUS,
    finalize_result,
)
from src.graph.nodes import deliver_node
from tests.test_graph.test_graph_flows import (
    invoke_plan,
    make_profile,
    make_runtime,
    safe_check,
    valid_plan,
)

EMERGENCY_QUERY = "训练时突然胸闷、头晕、眼前发黑，还能继续练吗"


class _Recorder:
    """记录型存储替身：不真正落盘，只记下被调用过什么。"""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append(name)
        return record


class _Untouchable:
    """任何属性访问都视为违规——用来证明某条路径确实没碰它。"""

    def __init__(self, label):
        self.label = label

    def __getattr__(self, name):
        raise AssertionError(f"{self.label}不该被访问:{name}")


def _deps():
    return SimpleNamespace(cache=_Recorder(), conversation=_Recorder(),
                           long_term=_Recorder())


def _state(result, query="帮我设计计划", profile=None, session_id="s-1"):
    # deliver_node 读的是字典形态的 profile（图状态里由 ingest 节点填好），
    # 不是 pydantic 模型。
    return {
        "query": query,
        "profile": profile or {"goal": "增肌", "injuries": []},
        "result": result,
        "session_id": session_id,
        "athlete_key": "athlete-1",
    }


def _legacy_profile(**overrides):
    data = {"goal": "增肌", "injuries": []}
    data.update(overrides)
    return SimpleNamespace(model_dump=lambda: dict(data), goal=data["goal"])


def _legacy_orchestrator(*, cache=None, long_term=None):
    orch = object.__new__(Orchestrator)
    orch.cache = cache if cache is not None else _Untouchable("缓存")
    orch.long_term = long_term if long_term is not None else _Recorder()
    return orch


# ---------------------------------------------------------------------------
# 1. 结构校验失败不得被标成安全交付
# ---------------------------------------------------------------------------

def test_structurally_invalid_plan_is_not_delivered_as_safe():
    """内部判定"不许交付"时，终态也不得自称 safe_delivered。

    ``_delivery_allowed`` 因 schema_valid=False 为假，而 ``requires_review``
    只反映安全类问题，这里仍为 False。此前交付节点只判后者，于是这份结构
    不合法的计划被标成"可以安全交付"——两个判据口径相反。
    """
    broken = {"days": "not a list", "confidence": 0.9}
    result = finalize_result(broken, [safe_check()], 0)

    assert result["_delivery_allowed"] is False
    assert result["requires_review"] is False, "用例前提：这属于非安全类阻断"

    payload = deliver_node(_deps(), _state(result))["final_payload"]
    assert payload["delivery_status"] == GENERATION_FAILED_STATUS
    assert payload["days"] == [], "不允许交付时不得带出可执行计划"
    assert payload["failure"]["reasons"], "必须告诉用户是哪一项没过"


def test_degraded_generation_is_not_delivered_as_safe():
    """生成过程降级同样属于"这次没生成好"，不是"计划可能有害"。"""
    result = finalize_result(valid_plan(), [safe_check()], 0, provider_degraded=True)
    assert result["_delivery_allowed"] is False
    assert result["requires_review"] is False

    payload = deliver_node(_deps(), _state(result))["final_payload"]
    assert payload["delivery_status"] == GENERATION_FAILED_STATUS
    assert payload["days"] == []


def test_warned_but_valid_plan_still_delivers():
    """反向保护：带建议性警告但结构合格的计划必须照常交付。

    交付判据从 requires_review 换成 _delivery_allowed 时，最怕的是顺手把
    "问题清单不干净"也带进交付条件——那会让绝大多数计划交付不出去。
    """
    warned = safe_check(issues=[{"issue": "建议增加后链训练", "severity": "warning"}])
    result = finalize_result(valid_plan(), [warned], 0)
    assert result["_delivery_allowed"] is True
    assert result["_persistence_allowed"] is False, "带警告不该进缓存"

    payload = deliver_node(_deps(), _state(result))["final_payload"]
    assert payload["delivery_status"] == "safe_delivered"
    assert payload["days"] == valid_plan()["days"]


# ---------------------------------------------------------------------------
# 2. 缓存命中必须带交付状态
# ---------------------------------------------------------------------------

def test_cache_hit_payload_carries_delivery_status():
    cached = valid_plan() | {"_persistence_allowed": True,
                             "requires_review": False, "warnings": []}
    runtime = make_runtime(cache_return=cached)
    state, _, _ = invoke_plan(runtime, make_profile())

    payload = state["final_payload"]
    assert payload["delivery_status"] == "safe_delivered"
    assert payload["days"] == cached["days"]
    assert "delivery_status" not in cached, "交付状态不该被写回缓存对象"


# ---------------------------------------------------------------------------
# 3. 急症先于缓存：两条入口口径一致
# ---------------------------------------------------------------------------

def test_emergency_beats_cache_in_graph_path():
    """缓存里有一份合格计划，也不能在急症描述下交付出去。"""
    cached = valid_plan() | {"_persistence_allowed": True,
                             "requires_review": False, "warnings": []}
    runtime = make_runtime(cache_return=cached)
    state, _, _ = invoke_plan(runtime, make_profile(), query=EMERGENCY_QUERY)

    payload = state["final_payload"]
    assert payload["delivery_status"] == EMERGENCY_STOP_STATUS
    assert payload["days"] == []
    assert "plan_id" not in payload, "急症终态不得携带任何已生成的计划"


def test_emergency_beats_cache_in_legacy_sync():
    """命令式同步入口：急症必须在查缓存之前判定。

    用"一碰就报错"的缓存替身证明这条路径根本没查缓存——只要它查了，
    这份训练计划就会被递到用户手上。
    """
    orch = _legacy_orchestrator(cache=_Untouchable("缓存"))
    payload = orch.generate_plan(_legacy_profile(), EMERGENCY_QUERY)

    assert payload["delivery_status"] == EMERGENCY_STOP_STATUS
    assert payload["days"] == []


def test_emergency_beats_cache_in_legacy_stream():
    """命令式流式入口：同上，且终态事件必须是 done + 急症载荷。"""
    orch = _legacy_orchestrator(cache=_Untouchable("缓存"))
    events = list(orch.generate_plan_stream(_legacy_profile(), EMERGENCY_QUERY))

    assert events[-1][0] == "done"
    payload = events[-1][1]
    assert payload["delivery_status"] == EMERGENCY_STOP_STATUS
    assert payload["days"] == []
    assert not any(event == "cache_hit" for event, _ in events)


def test_emergency_in_injuries_field_also_blocks():
    """症状写在档案的伤病字段里同样要拦——只看 query 会漏一半入口。"""
    orch = _legacy_orchestrator(cache=_Untouchable("缓存"))
    payload = orch.generate_plan(
        _legacy_profile(injuries=["最近胸闷，偶尔头晕"]), "帮我设计增肌计划")

    assert payload["delivery_status"] == EMERGENCY_STOP_STATUS
