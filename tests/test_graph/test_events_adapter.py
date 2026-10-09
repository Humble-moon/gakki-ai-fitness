"""Tests for the LangGraph → (event, data) SSE adapter."""

import pytest

from src.graph.events import build_inputs, graph_stream_events
from tests.test_graph.test_graph_flows import (
    _needs_review_check,
    make_profile,
    make_runtime,
    valid_plan,
)


def run_events(runtime, profile=None, query="测试"):
    profile = profile or make_profile()
    thread_id = runtime.new_thread_id()
    config = runtime.config_for(thread_id)
    inputs = build_inputs(profile, query=query, thread_id=thread_id)
    return list(graph_stream_events(runtime, inputs, config))


def test_happy_path_event_sequence_ends_with_done():
    runtime = make_runtime()
    events = run_events(runtime)

    names = [event for event, _ in events]
    assert names[-1] == "done"
    # progressive events reuse the legacy protocol names, in order
    for expected in ["planner_done", "retriever_done", "writer_chunk", "factcheck_done"]:
        assert expected in names
    assert names.index("planner_done") < names.index("retriever_done")
    assert names.index("retriever_done") < names.index("factcheck_done")
    # terminal payload is the delivered plan
    final_event, final_payload = events[-1]
    assert final_payload["delivery_status"] == "safe_delivered"


def test_interrupt_surfaces_review_payload_as_done():
    runtime = make_runtime(checks=[_needs_review_check()])
    events = run_events(runtime)
    final_event, final_payload = events[-1]
    assert final_event == "done"
    assert final_payload["delivery_status"] == "review_pending"
    assert final_payload["review"]["review_id"]


def test_goal_mismatch_emits_single_error_terminal():
    runtime = make_runtime(plan=valid_plan(goal="增肌"))
    events = run_events(runtime, profile=make_profile(goal="减脂"))
    terminals = [(e, d) for e, d in events if e in {"done", "error"}]
    assert len(terminals) == 1
    event, payload = terminals[0]
    assert event == "error"
    assert payload["code"] == "GOAL_CONSISTENCY_FAILED"


def test_llm_unavailable_emits_dependency_error():
    runtime = make_runtime(planner_raises=True)
    events = run_events(runtime)
    terminals = [(e, d) for e, d in events if e in {"done", "error"}]
    assert terminals == [("error", {"code": "DEPENDENCY_UNAVAILABLE",
                                    "message": "演示依赖暂时不可用"})]


def test_cache_hit_emits_cache_hit_then_done():
    cached = valid_plan() | {"_persistence_allowed": True,
                             "requires_review": False, "warnings": []}
    runtime = make_runtime(cache_return=cached)
    events = run_events(runtime)
    names = [event for event, _ in events]
    assert "cache_hit" in names
    assert names[-1] == "done"
    assert events[-1][1] == cached


# ===========================================================================
# 状态图与命令式编排的事件对齐
# ===========================================================================
class _LongTermWithMemory:
    """长期记忆替身：既能读（开场白要用），也能写（persist_if_safe 要用）。"""

    def __init__(self, context=""):
        self.context = context
        self.read_keys = []
        self.written = []

    def build_context_for_prompt(self, key):
        self.read_keys.append(key)
        return self.context

    def save_preference(self, user_id, key, value):
        self.written.append((user_id, key))


def _runtime_with_advice(text="你好，我是你的教练。", memory="", **make_kwargs):
    """给图装上"有 LLM 的 writer"与"有内容的长期记忆"。

    make_runtime 的替身刻意不带 llm，因此开场白那段默认走不到；要验证它，
    必须显式补上——这也正是它此前一直没被发现缺失的原因。
    """
    from src.agents.planner import PlannerAgent  # noqa: F401  (仅确保导入顺序稳定)
    from types import SimpleNamespace

    runtime = make_runtime(**make_kwargs)
    captured = {"messages": []}

    def chat_stream(messages, temperature=0):
        captured["messages"].append(messages)
        for piece in text:
            yield piece

    runtime.deps.writer.llm = SimpleNamespace(chat_stream=chat_stream)
    runtime.deps.long_term = _LongTermWithMemory(memory)
    return runtime, captured


def test_advice_event_reaches_the_stream():
    """开场白必须流式发出——命令式编排有、状态图此前没有。

    缺了它，前端那个"教练先说说你的情况"的区域在 v2 上永远是空的。
    """
    runtime, captured = _runtime_with_advice("你好呀，先看看你的数据。")
    events = run_events(runtime)
    names = [e for e, _ in events]

    assert "advice_chunk" in names, "开场白要有流式分片"
    assert "advice_done" in names, "开场白要有结束事件（前端据此收尾）"
    joined = "".join(d for e, d in events if e == "advice_chunk")
    assert joined == "你好呀，先看看你的数据。"
    assert names.index("advice_done") < names.index("planner_done"), "暖场应在规划之前"
    assert captured["messages"], "提示词确实被送进了模型"


def test_advice_prompt_carries_cross_session_memory():
    """开场白提示词要带上跨会话记忆，否则"这个用户之前来过"就无从谈起。"""
    runtime, captured = _runtime_with_advice(memory="伤病史：['膝盖偶尔疼']")
    run_events(runtime)
    content = captured["messages"][0][0]["content"]
    assert "膝盖偶尔疼" in content, "记忆没有进提示词"
    assert "[这个用户之前来过" in content


def test_ingest_reads_long_term_with_the_same_key_it_writes():
    """读记忆与写记忆必须用同一个身份键。

    身份键算错的表现很隐蔽：写入成功、读取也"成功"，只是读到的是另一个人
    （或另一份空数据）——用户看到的就是"它又忘了"。
    """
    from src.core.plan_finalization import long_term_user_key

    runtime, _ = _runtime_with_advice()
    run_events(runtime)
    lt = runtime.deps.long_term

    assert lt.read_keys, "ingest 应当读取长期记忆"
    expected = long_term_user_key(make_profile().model_dump(), None)
    assert lt.read_keys[0] == expected, "读侧身份键与 long_term_user_key 不一致"


def test_writer_done_raw_carries_the_full_draft():
    """初稿全文要单独发一次——前端据此渲染草稿预览。

    只发 writer_chunk 的话，前端得自己拼接并猜测"写完了没有"，
    而"写完"这件事只有 post-write 的节点知道。
    """
    runtime, _ = _runtime_with_advice()
    events = run_events(runtime)
    raw = [d for e, d in events if e == "writer_done_raw"]
    assert raw, "缺少 writer_done_raw 事件"
    assert raw[0] == "生成中", f"应为完整初稿文本，实际 {raw[0]!r}"


def test_plan_with_warnings_still_gets_a_delivery_status():
    """交付带建议性警告时，终态载荷必须带上 safe_delivered。

    这是最常见的交付形态：LLM 检查器几乎从不返回空的问题清单。
    deliver 节点原先用缓存闸门（还要求清单干净）判断要不要标记交付，
    于是这类计划的载荷里**没有 delivery_status**，而前端写的是
    ``if (data?.delivery_status !== 'safe_delivered') return;``
    ——计划有内容，用户却什么都看不到。

    既有测试没抓到它，是因为替身用的都是"零警告"的检查结果，
    这条路径从未被走到。
    """
    from tests.test_graph.test_graph_flows import safe_check

    warned = safe_check(issues=[{"issue": "建议增加后链训练", "severity": "warning"}])
    runtime = make_runtime(checks=[warned])
    events = run_events(runtime)
    final_event, payload = events[-1]

    assert final_event == "done"
    assert payload.get("days"), "计划内容应当在载荷里"
    assert payload.get("delivery_status") == "safe_delivered", (
        "带警告的交付也必须标记为 safe_delivered，否则前端静默不渲染")
    assert payload.get("_persistence_allowed") is False, "带警告不该进缓存（这条不变）"


def test_warned_plan_still_writes_the_user_profile():
    """带警告的交付同样要记住用户——记忆与缓存是两件事。"""
    from tests.test_graph.test_graph_flows import safe_check

    warned = safe_check(issues=[{"issue": "建议增加后链训练", "severity": "warning"}])
    runtime, _ = _runtime_with_advice(checks=[warned])
    run_events(runtime)
    written = runtime.deps.long_term.written
    assert written, "交付带警告时画像也必须落盘"
    assert {k for _, k in written} >= {"profile", "goal"}
