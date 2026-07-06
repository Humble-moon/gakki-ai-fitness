"""Task 14（MCP 协议）和 Task 15（A2A + 记忆 + HITL + Skill）的测试。"""

import json
from unittest.mock import patch, MagicMock

# ---------- ExerciseMCPServer（MCP 动作库服务）----------

from src.mcp.exercise_server import ExerciseMCPServer


def test_exercise_mcp_search_by_muscle():
    server = ExerciseMCPServer()
    results = server.call_tool("search_by_muscle", {"muscle": "胸大肌"})
    assert len(results) >= 1
    assert any(e["name"] == "哑铃卧推" for e in results)


def test_exercise_mcp_search_by_equipment():
    server = ExerciseMCPServer()
    results = server.call_tool("search_by_equipment", {"equipment": "哑铃"})
    assert len(results) >= 2
    assert all("哑铃" in e["equipment"] for e in results)


def test_exercise_mcp_search_by_difficulty():
    server = ExerciseMCPServer()
    results = server.call_tool("search_by_difficulty", {"difficulty": "中级"})
    assert len(results) >= 2
    assert all(e["difficulty"] == "中级" for e in results)


def test_exercise_mcp_get_exercise_detail_found():
    server = ExerciseMCPServer()
    results = server.call_tool("get_exercise_detail", {"name": "杠铃深蹲"})
    assert len(results) == 1
    assert results[0]["name"] == "杠铃深蹲"


def test_exercise_mcp_get_exercise_detail_not_found():
    """v2: 未找到动作抛出 McpToolError 而非返回 []。"""
    from src.mcp.exercise_server import McpToolError
    server = ExerciseMCPServer()
    try:
        server.call_tool("get_exercise_detail", {"name": "不存在"})
        assert False, "Expected McpToolError"
    except McpToolError as e:
        assert e.tool_name == "get_exercise_detail"


def test_exercise_mcp_unknown_tool():
    """v2: 未知工具抛出 McpToolError 而非返回 []。"""
    from src.mcp.exercise_server import McpToolError
    server = ExerciseMCPServer()
    try:
        server.call_tool("unknown_tool", {})
        assert False, "Expected McpToolError"
    except McpToolError as e:
        assert "unknown_tool" in str(e)


def test_exercise_mcp_list_tools():
    server = ExerciseMCPServer()
    tools = server.list_tools()
    assert len(tools) == 4
    tool_names = [t["name"] for t in tools]
    assert "search_by_muscle" in tool_names
    assert "search_by_equipment" in tool_names
    assert "search_by_difficulty" in tool_names
    assert "get_exercise_detail" in tool_names


# ---------- ToolRegistry（使用模拟数据库依赖）----------

@patch("src.mcp.tool_registry.GraphSearch")
def test_tool_registry_call_mcp_tool(mock_graph):
    from src.mcp.tool_registry import ToolRegistry
    registry = ToolRegistry()
    results = registry.call("search_by_equipment", {"equipment": "杠铃"})
    assert len(results) >= 1
    assert all("杠铃" in e["equipment"] for e in results)


@patch("src.mcp.tool_registry.GraphSearch")
def test_tool_registry_list_tools(mock_graph):
    from src.mcp.tool_registry import ToolRegistry
    registry = ToolRegistry()
    tools = registry.list_tools()
    assert len(tools) == 7
    tool_names = [t["name"] for t in tools]
    assert "search_by_muscle" in tool_names
    assert "graph_multi_hop" in tool_names
    assert "graph_injury_risk" in tool_names
    assert "graph_reason_pain" in tool_names


@patch("src.mcp.tool_registry.GraphSearch")
def test_tool_registry_call_unknown_tool(mock_graph):
    """v2: 未知工具抛出 McpToolError(code=-32601) 而非返回 None。"""
    from src.mcp.tool_registry import ToolRegistry, McpToolError
    registry = ToolRegistry()
    try:
        registry.call("nonexistent", {})
        assert False, "Expected McpToolError"
    except McpToolError as e:
        assert e.code == -32601


# ---------- SkillRegistry ----------

from src.skills.registry import SkillRegistry


def test_skill_registry_match_muscle_building():
    registry = SkillRegistry()
    result = registry.match("我想增肌，给我个计划")
    assert result == "muscle_building"


def test_skill_registry_match_fat_loss():
    registry = SkillRegistry()
    result = registry.match("我要减脂，太胖了")
    assert result == "fat_loss"


def test_skill_registry_match_exercise_analysis():
    registry = SkillRegistry()
    result = registry.match("我做卧推的时候感觉肩膀疼")
    assert result == "exercise_analysis"


def test_skill_registry_match_fallback():
    registry = SkillRegistry()
    result = registry.match("今天天气不错")
    assert result == "muscle_building"  # 默认回退


def test_skill_registry_get_existing():
    registry = SkillRegistry()
    skill = registry.get("fat_loss")
    assert skill is not None
    assert skill.name == "fat_loss"
    assert "减脂" in skill.triggers


def test_skill_registry_get_nonexistent():
    registry = SkillRegistry()
    skill = registry.get("nonexistent")
    assert skill is None


# ---------- MessageBus (A2A) ----------

from src.a2a.messaging import MessageBus, Task, TaskStatus, Artifact


def test_message_bus_send_and_get():
    bus = MessageBus()
    task1 = Task(task_id="t1", from_agent="user", to_agent="coach",
                 task_type="plan_generation", payload={"goal": "增肌"})
    task2 = Task(task_id="t2", from_agent="user", to_agent="nutritionist",
                 task_type="diet_plan", payload={"goal": "减脂"})
    bus.send(task1)
    bus.send(task2)

    coach_tasks = bus.get_for_agent("coach")
    assert len(coach_tasks) == 1
    assert coach_tasks[0].task_id == "t1"
    assert coach_tasks[0].status == TaskStatus.PENDING

    nutritionist_tasks = bus.get_for_agent("nutritionist")
    assert len(nutritionist_tasks) == 1
    assert nutritionist_tasks[0].task_id == "t2"


def test_task_complete():
    task = Task(task_id="t3", from_agent="user", to_agent="coach",
                task_type="plan", payload={})
    task.complete()
    assert task.status == TaskStatus.COMPLETED


def test_task_fail():
    task = Task(task_id="t4", from_agent="user", to_agent="coach",
                task_type="plan", payload={})
    task.fail()
    assert task.status == TaskStatus.FAILED


def test_task_add_artifact():
    task = Task(task_id="t5", from_agent="user", to_agent="coach",
                task_type="plan", payload={})
    artifact = Artifact(artifact_id="a1", artifact_type="training_plan",
                        content={"exercises": ["深蹲", "卧推"]})
    task.add_artifact(artifact)
    assert len(task.artifacts) == 1
    assert task.artifacts[0].artifact_type == "training_plan"


def test_message_bus_get_for_agent_skips_completed():
    bus = MessageBus()
    task1 = Task(task_id="t6", from_agent="user", to_agent="coach",
                 task_type="plan", payload={})
    task2 = Task(task_id="t7", from_agent="user", to_agent="coach",
                 task_type="plan", payload={})
    task2.complete()
    bus.send(task1)
    bus.send(task2)

    pending = bus.get_for_agent("coach")
    assert len(pending) == 1
    assert pending[0].task_id == "t6"


# ---------- HITLReview ----------

from src.hitl.review import HITLReview


def test_hitl_review_low_confidence():
    review = HITLReview()
    result = review.check({"confidence": 0.5, "issues": []})
    assert result.needs_review is True
    assert result.severity == "warning"


def test_hitl_review_high_confidence():
    review = HITLReview()
    result = review.check({"confidence": 0.9, "issues": []})
    assert result.needs_review is False
    assert result.severity == "safe"


def test_hitl_review_danger_issue():
    review = HITLReview()
    result = review.check({
        "confidence": 0.8,
        "issues": [
            {"issue": "危险动作建议", "severity": "danger"},
        ]
    })
    assert result.needs_review is True
    assert result.severity == "danger"
    assert "危险动作建议" in result.suggestions


def test_hitl_review_warning_issue():
    review = HITLReview()
    result = review.check({
        "confidence": 0.8,
        "issues": [
            {"issue": "可能不太适合", "severity": "warning"},
        ]
    })
    assert result.needs_review is True
    assert result.severity == "warning"
    assert "可能不太适合" in result.suggestions


# ---------- LongTermMemory（使用 Redis）----------

from src.memory.long_term import LongTermMemory


def test_long_term_memory_save_and_get_preferences():
    memory = LongTermMemory()
    memory.redis.flushdb()
    memory.save_preference(999, "goal", "增肌")
    memory.save_preference(999, "equipment", ["哑铃", "杠铃"])
    prefs = memory.get_preferences(999)
    assert prefs.get("goal") == "增肌"
    assert prefs.get("equipment") == ["哑铃", "杠铃"]
    memory.redis.flushdb()


def test_long_term_memory_get_injury_history_empty():
    memory = LongTermMemory()
    memory.redis.flushdb()
    injuries = memory.get_injury_history(999)
    assert injuries == []
    memory.redis.flushdb()


def test_long_term_memory_record_feedback():
    memory = LongTermMemory()
    memory.redis.flushdb()
    memory.record_feedback(999, "plan_001", 5, "很有效果")
    # 直接读取 key 进行校验
    key = "memory:user:999:feedback:plan_001"
    raw = memory.redis.get(key)
    assert raw is not None
    feedback = json.loads(raw)
    assert feedback["rating"] == 5
    assert feedback["plan_id"] == "plan_001"
    assert feedback["comment"] == "很有效果"
    memory.redis.flushdb()


def test_long_term_memory_build_context_for_prompt():
    memory = LongTermMemory()
    memory.redis.flushdb()
    memory.save_preference(999, "goal", "增肌")
    memory.save_preference(999, "injuries", ["肩袖损伤"])
    ctx = memory.build_context_for_prompt(999)
    assert "增肌" in ctx
    assert "肩袖损伤" in ctx
    memory.redis.flushdb()
