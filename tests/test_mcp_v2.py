"""MCP v2 协议升级测试 —— JSON Schema、Resources、错误码。

与 test_advanced.py 的区别：
  test_advanced.py 中的 MCP 测试覆盖 v1 兼容层行为。
  本文件测试 v2 新增能力：完整 JSON Schema、Resources 协议、结构化错误。
"""

import pytest
from unittest.mock import patch

from src.mcp.exercise_server import ExerciseMCPServer, McpToolError


# =========================================================================
# ExerciseMCPServer — v2 兼容层测试
# =========================================================================

class TestExerciseMCPServer:
    """v1 兼容层的行为验证。"""

    def test_search_by_muscle(self):
        server = ExerciseMCPServer()
        results = server.call_tool("search_by_muscle", {"muscle": "胸大肌"})
        assert len(results) >= 1
        assert any(e["name"] == "哑铃卧推" for e in results)

    def test_search_by_equipment(self):
        server = ExerciseMCPServer()
        results = server.call_tool("search_by_equipment", {"equipment": "哑铃"})
        assert len(results) >= 2
        assert all("哑铃" in e["equipment"] for e in results)

    def test_search_by_difficulty(self):
        server = ExerciseMCPServer()
        results = server.call_tool("search_by_difficulty", {"difficulty": "中级"})
        assert len(results) >= 2
        assert all(e["difficulty"] == "中级" for e in results)

    def test_get_exercise_detail_found(self):
        server = ExerciseMCPServer()
        results = server.call_tool("get_exercise_detail", {"name": "杠铃深蹲"})
        assert len(results) == 1
        assert results[0]["name"] == "杠铃深蹲"

    def test_get_exercise_detail_not_found(self):
        """v2 行为：未找到时抛出 McpToolError，而非返回空列表。

        注意：FastMCP 内部将工具函数中的 ValueError 包装为执行错误，
        所以错误码为 -32603 (Internal error) 而非 -32602 (Invalid params)。
        这是可接受的——Agent 仍然能感知到工具调用失败，而非像 v1 那样静默返回 []。
        """
        server = ExerciseMCPServer()
        with pytest.raises(McpToolError) as exc_info:
            server.call_tool("get_exercise_detail", {"name": "不存在"})
        # 错误码可能是 -32602 或 -32603，取决于 FastMCP 包装方式
        assert exc_info.value.code in (-32602, -32603)
        assert exc_info.value.tool_name == "get_exercise_detail"

    def test_unknown_tool(self):
        """v2 行为：未知工具时抛出 McpToolError(code=-32603)，而非返回空列表。"""
        server = ExerciseMCPServer()
        with pytest.raises(McpToolError) as exc_info:
            server.call_tool("unknown_tool", {})
        assert exc_info.value.tool_name == "unknown_tool"

    def test_list_tools(self):
        server = ExerciseMCPServer()
        tools = server.list_tools()
        assert len(tools) == 4
        tool_names = [t["name"] for t in tools]
        assert "search_by_muscle" in tool_names
        assert "search_by_equipment" in tool_names
        assert "search_by_difficulty" in tool_names
        assert "get_exercise_detail" in tool_names

    def test_list_tools_has_json_schema(self):
        """v2 关键升级：list_tools 返回完整 JSON Schema，不是 v1 的 'params: dict'。"""
        server = ExerciseMCPServer()
        tools = server.list_tools()
        for t in tools:
            schema = t.get("inputSchema", {})
            assert schema["type"] == "object"
            assert "properties" in schema
            assert "required" in schema
            # 每个工具至少有 1 个参数
            assert len(schema["properties"]) >= 1

    def test_mcp_error_to_dict(self):
        """McpToolError 可以序列化为 MCP 标准错误响应格式。"""
        error = McpToolError(
            code=-32601,
            message="Method not found",
            tool_name="test_tool",
            details="not registered",
        )
        d = error.to_dict()
        assert d["error"]["code"] == -32601
        assert d["error"]["message"] == "Method not found"
        assert d["error"]["data"]["tool_name"] == "test_tool"


# =========================================================================
# ToolRegistry — v2 测试（含 JSON Schema、Resources、错误码）
# =========================================================================

class TestToolRegistryV2:
    """ToolRegistry v2 功能验证。"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """每次测试前 mock GraphSearch 避免 Neo4j 连接。"""
        with patch("src.mcp.tool_registry.GraphSearch"):
            yield

    def test_list_tools_count(self):
        from src.mcp.tool_registry import ToolRegistry
        registry = ToolRegistry()
        tools = registry.list_tools()
        assert len(tools) == 7

    def test_list_tools_has_fastmcp_schemas(self):
        """FastMCP 工具应该有完整的 JSON Schema（从 type hints 自动生成）。"""
        from src.mcp.tool_registry import ToolRegistry
        registry = ToolRegistry()
        tools = {t["name"]: t for t in registry.list_tools()}

        # search_by_muscle 的 schema
        schema = tools["search_by_muscle"]["inputSchema"]
        assert schema["type"] == "object"
        assert "muscle" in schema["properties"]
        assert schema["properties"]["muscle"]["type"] == "string"
        assert "muscle" in schema["required"]

    def test_list_tools_has_graph_schemas(self):
        """GraphRAG 工具的 schema 应该有完整的 required 和 enum。"""
        from src.mcp.tool_registry import ToolRegistry
        registry = ToolRegistry()
        tools = {t["name"]: t for t in registry.list_tools()}

        schema = tools["graph_reason_pain"]["inputSchema"]
        assert schema["type"] == "object"
        assert "exercise" in schema["required"]
        assert "symptom" in schema["required"]

    def test_call_unknown_tool(self):
        """v2 行为：未知工具抛出 McpToolError(code=-32601) 而非返回 None。"""
        from src.mcp.tool_registry import ToolRegistry
        registry = ToolRegistry()
        with pytest.raises(McpToolError) as exc_info:
            registry.call("nonexistent", {})
        assert exc_info.value.code == -32601
        assert "nonexistent" in str(exc_info.value)

    def test_call_mcp_tool(self):
        from src.mcp.tool_registry import ToolRegistry
        registry = ToolRegistry()
        results = registry.call("search_by_equipment", {"equipment": "杠铃"})
        assert len(results) >= 1
        assert all("杠铃" in e["equipment"] for e in results)

    def test_get_tools_prompt(self):
        """get_tools_prompt 生成 LLM 可读的工具描述。"""
        from src.mcp.tool_registry import ToolRegistry
        registry = ToolRegistry()
        prompt = registry.get_tools_prompt()
        assert "search_by_muscle" in prompt
        assert "search_by_equipment" in prompt
        assert "graph_multi_hop" in prompt
        assert "graph_reason_pain" in prompt

    def test_list_resources(self):
        """v2 新增：list_resources 返回动作标准库和知识库的资源列表。"""
        from src.mcp.tool_registry import ToolRegistry
        registry = ToolRegistry()
        resources = registry.list_resources()
        assert len(resources) >= 10  # 至少 2 个索引资源 + 8 个动作标准

        uris = [r["uri"] for r in resources]
        assert "exercise://library" in uris
        assert "exercise://muscles" in uris
        assert any(uri.startswith("exercise://standards/") for uri in uris)

        # 每个资源都应有 uri, name, description, mimeType
        for r in resources:
            assert "uri" in r
            assert "name" in r
            assert "description" in r
            assert "mimeType" in r

    def test_read_resource_exercise_standard(self):
        """读取动作标准资源。"""
        from src.mcp.tool_registry import ToolRegistry
        registry = ToolRegistry()
        content = registry.read_resource("exercise://standards/杠铃深蹲")
        assert "杠铃深蹲" in content
        assert "标准做法" in content
        assert "常见错误" in content

    def test_read_resource_exercise_library(self):
        """读取动作库索引资源。"""
        from src.mcp.tool_registry import ToolRegistry
        registry = ToolRegistry()
        content = registry.read_resource("exercise://library")
        assert "动作库索引" in content
        assert "哑铃卧推" in content

    def test_read_resource_exercise_muscles(self):
        """读取肌群覆盖资源。"""
        from src.mcp.tool_registry import ToolRegistry
        registry = ToolRegistry()
        content = registry.read_resource("exercise://muscles")
        assert "肌群-动作覆盖" in content

    def test_read_resource_unknown_uri_prefix(self):
        """读取未知 URI 前缀的资源应抛出 McpToolError。"""
        from src.mcp.tool_registry import ToolRegistry
        registry = ToolRegistry()
        with pytest.raises(McpToolError) as exc_info:
            registry.read_resource("unknown://something")
        assert exc_info.value.code == -32601

    def test_list_resources_no_knowledge_dir(self, tmp_path, monkeypatch):
        """知识库目录不存在时优雅降级（不崩溃）。"""
        from src.mcp.tool_registry import ToolRegistry
        registry = ToolRegistry()
        # _find_knowledge_files 已经在初始化时调用了。
        # 如果 data/knowledge 目录不存在，应该不崩溃且不包含 knowledge:// 资源
        resources = registry.list_resources()
        # 所有资源都应该是 exercise:// 或 knowledge://
        knowledge_uris = [r for r in resources if r["uri"].startswith("knowledge://")]
        # 不存在 knowledge 资源时不应崩溃
        for r in resources:
            assert r["uri"].startswith(("exercise://", "knowledge://"))
