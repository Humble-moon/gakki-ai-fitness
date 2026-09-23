"""
双后端训练上下文注入的一致性守卫。

项目有两套并行编排后端（手写 Orchestrator 与 LangGraph 状态图），必须行为一致。
训练历史的注入链很长：prompts → agents → 编排器/图节点 → 状态声明，任何一环漏改
都会让某个后端静默地拿不到上下文，而**不会报错**——这类偏差只有靠契约测试才能
在提交前发现。

本文件锁两件事：
    1. 两个后端产出的上下文逐字符相同；
    2. 注入链上每个环节都确实接受并转发 training_context。
"""

import inspect
from types import SimpleNamespace

import pytest

from src.agents.planner import PlannerAgent
from src.agents.writer import WriterAgent
from src.core.orchestrator import Orchestrator
from src.graph.nodes import ingest_node, plan_node, write_node
from src.llm.prompts.planner import build_planner_messages
from src.llm.prompts.writer import build_writer_messages

STUB_CONTEXT = "【训练执行情况（最近 8 周）】\n共 12 次训练，平均完成率 92%。"


# ----------------------------------------------------------------------
# 两后端产出必须一致
# ----------------------------------------------------------------------

def test_both_backends_inject_identical_context(monkeypatch):
    """手写编排与 LangGraph 注入的训练上下文必须逐字符相同。"""
    monkeypatch.setattr("src.core.training_history.build_training_context",
                        lambda key, **kw: STUB_CONTEXT if key else "")

    v1 = Orchestrator._training_context("athlete-1")
    v2 = ingest_node(SimpleNamespace(), {
        "profile": {"goal": "增肌"}, "athlete_key": "athlete-1",
    })["training_context"]

    assert v1 == v2 == STUB_CONTEXT


def test_both_backends_yield_empty_without_athlete_key(monkeypatch):
    """没有 athlete_key 时两个后端都应得到空串，而不是 None 或异常。"""
    monkeypatch.setattr("src.core.training_history.build_training_context",
                        lambda key, **kw: STUB_CONTEXT if key else "")

    assert Orchestrator._training_context(None) == ""
    assert ingest_node(SimpleNamespace(), {
        "profile": {"goal": "增肌"},
    })["training_context"] == ""


# ----------------------------------------------------------------------
# LangGraph 状态声明守卫
# ----------------------------------------------------------------------

def test_coach_state_declares_training_context_and_athlete_key():
    """CoachState 必须声明这两个键。

    LangGraph 对**未声明**的 state 键会静默丢弃、不抛异常——漏声明的表现是
    「图后端拿不到训练历史」而不是报错，极难排查。这条测试就是防它的。
    """
    from src.graph.state import CoachState

    annotations = CoachState.__annotations__
    assert "training_context" in annotations
    assert "athlete_key" in annotations


def test_build_inputs_carries_athlete_key_into_graph_state():
    from src.graph.events import build_inputs

    inputs = build_inputs({"goal": "增肌"}, query="q", athlete_key="athlete-9")
    assert inputs["athlete_key"] == "athlete-9"
    # 不传时必须是 None（而不是缺失），否则 .get 之外的下游取值会 KeyError
    assert build_inputs({"goal": "增肌"}, query="q")["athlete_key"] is None


# ----------------------------------------------------------------------
# 注入链每一环都要转发
# ----------------------------------------------------------------------

def test_plan_node_forwards_training_context_to_planner():
    captured = {}

    def fake_plan(user_input, prof, conv_context="", plan_context="",
                  training_context=""):
        captured["training_context"] = training_context
        return {}

    deps = SimpleNamespace(planner=SimpleNamespace(plan=fake_plan))
    plan_node(deps, {"query": "q", "profile": {}, "training_context": STUB_CONTEXT})
    assert captured["training_context"] == STUB_CONTEXT


def test_write_node_forwards_training_context_to_writer():
    captured = {}

    def fake_write(retr, prof, cfg, plan_context="", user_query="",
                   training_context=""):
        captured["training_context"] = training_context
        yield ("done", {})

    deps = SimpleNamespace(writer=SimpleNamespace(write_plan_stream=fake_write))
    list(write_node(deps, {"retrieved": {}, "profile": {}, "plan": {},
                           "training_context": STUB_CONTEXT}))
    assert captured["training_context"] == STUB_CONTEXT


@pytest.mark.parametrize("obj,name", [
    (PlannerAgent.plan, "PlannerAgent.plan"),
    (WriterAgent.write_plan, "WriterAgent.write_plan"),
    (WriterAgent.write_plan_stream, "WriterAgent.write_plan_stream"),
    (build_planner_messages, "build_planner_messages"),
    (build_writer_messages, "build_writer_messages"),
])
def test_every_stage_of_the_chain_accepts_training_context(obj, name):
    """链上任何一环漏了参数，注入都会在该环节静默断掉。"""
    assert "training_context" in inspect.signature(obj).parameters, (
        f"{name} 必须接受 training_context，否则训练历史注入在此处断裂"
    )


@pytest.mark.parametrize("obj,name", [
    (PlannerAgent.plan, "PlannerAgent.plan"),
    (WriterAgent.write_plan, "WriterAgent.write_plan"),
    (WriterAgent.write_plan_stream, "WriterAgent.write_plan_stream"),
    (build_planner_messages, "build_planner_messages"),
    (build_writer_messages, "build_writer_messages"),
])
def test_new_parameter_has_a_default_so_callers_need_not_change(obj, name):
    """带默认值是为了不破坏既有调用方（CLI、脚本、测试替身）。"""
    param = inspect.signature(obj).parameters["training_context"]
    assert param.default == "", f"{name} 的 training_context 默认值应为空串"


@pytest.mark.parametrize("obj,name", [
    (Orchestrator.generate_plan, "Orchestrator.generate_plan"),
    (Orchestrator.generate_plan_stream, "Orchestrator.generate_plan_stream"),
])
def test_orchestrators_accept_athlete_key(obj, name):
    """编排器接收的是 athlete_key 而非上下文本身——读取由它自己负责。

    这条区分很重要：如果编排器改为接收渲染好的文本，两个后端就各自需要
    一份读取逻辑，正是本方案要避免的漂移源。
    """
    params = inspect.signature(obj).parameters
    assert "athlete_key" in params, f"{name} 必须接受 athlete_key"
    assert params["athlete_key"].default is None, "默认必须是 None（未登录/无历史）"
    assert "training_context" not in params, "编排器不应接收渲染好的上下文"


# ----------------------------------------------------------------------
# 安全闸门不受影响
# ----------------------------------------------------------------------

def test_training_context_does_not_touch_finalization_gates():
    """训练上下文只作为生成输入，不得改变终态判定。

    这条守卫的含义：finalize_result 的签名里**不应该**出现 training_context——
    一旦有人把它接进持久化门控，就等于让用户数据绕过了安全检查。
    """
    from src.core.plan_finalization import finalize_result

    params = inspect.signature(finalize_result).parameters
    assert "training_context" not in params
    assert "athlete_key" not in params
