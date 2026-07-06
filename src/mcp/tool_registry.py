"""
================================================================================
文件角色：工具注册中心（Tool Registry）—— MCP 协议兼容版
================================================================================
- 被调用者：编排引擎（orchestrator）通过 ToolRegistry 统一调度所有工具。
  上层代码只需 registry.call("search_by_muscle", {"muscle": "胸"})，
  无需关心工具来自 MCP Server 还是 GraphRAG。
- 调用者：本模块聚合了三类底层工具模块：
  1. ExerciseMCPServer  - 动作库的 MCP 查询接口（FastMCP 实现）
  2. GraphSearch        - 知识图谱的多跳检索 + 伤病推理
  3. KnowledgeBase      - 18 篇健身知识文档的 Resources 暴露
- 项目角色：工具层的"门面"（Facade 模式）——统一注册、统一调度、统一接口。
  让 LLM Agent 的 function calling 只需对接这一层。

升级 v2 —— 完整 MCP 协议：
  v1 的 list_tools() 返回 {"name": "x", "params": "dict"}，LLM 不知道参数该填什么。
  v2 返回完整的 JSON Schema：
    {
      "name": "search_by_muscle",
      "description": "按目标肌群搜索训练动作",
      "inputSchema": {
        "type": "object",
        "properties": {
          "muscle": {
            "type": "string",
            "description": "目标肌群名称",
            "enum": ["胸大肌", "背阔肌", ...]
          },
          "difficulty": {
            "type": "string",
            "enum": ["初级", "中级", "高级"]
          }
        },
        "required": ["muscle"]
      }
    }

  新增 MCP Resources 支持：
    - list_resources() → 列出所有可用的知识资源（动作标准 + 知识库文档）
    - read_resource(uri) → 按 URI 精确读取资源内容
    这让 Agent 可以从 "盲目检索" 变成 "有目的地查阅"。

  新增结构化错误：
    v1: 未知工具返回 None，Agent 不知道失败原因
    v2: 抛出 McpToolError 带错误码，Agent 可根据错误类型决策
================================================================================
"""

from __future__ import annotations
import glob as glob_mod
import logging
import os
from pathlib import Path
from typing import Any

from src.mcp.exercise_server import (
    ExerciseMCPServer,
    McpToolError,
    mcp as exercise_mcp,
)
from src.graphrag.search import GraphSearch

logger = logging.getLogger(__name__)


# =========================================================================
# 工具 JSON Schema 手动定义 —— GraphRAG 工具
# =========================================================================
# FastMCP 工具（exercise_server.py 中的 4 个）通过 type hints 自动生成 schema。
# GraphRAG 工具因为依赖 Neo4j 连接，不适合用 @mcp.tool() 装饰器（初始化时序问题），
# 这里手动定义其 JSON Schema，保持统一格式。
# =========================================================================

_GRAPH_TOOL_SCHEMAS = [
    {
        "name": "graph_multi_hop",
        "description": (
            "多跳组合查询：同时按器械和肌群搜索训练动作。"
            "利用知识图谱中的关系链（Exercise→REQUIRES→Equipment + Exercise→TARGETS→Muscle），"
            "找到同时满足器械条件和肌群目标的动作。"
            "适用场景：用户说'用哑铃练胸的动作有哪些?'"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "equipment": {
                    "type": "string",
                    "description": "器材名称，如 '哑铃'、'杠铃'、'自重'",
                },
                "target": {
                    "type": "string",
                    "description": "目标肌群，如 '胸'、'背'、'腿'",
                },
            },
            "required": ["equipment", "target"],
        },
    },
    {
        "name": "graph_injury_risk",
        "description": (
            "查询某个训练动作的已知伤病风险。"
            "利用知识图谱中的 Exercise→MAY_CAUSE→Injury 关系链，"
            "返回该动作可能导致的伤病列表。"
            "适用场景：用户问'深蹲会不会伤膝盖?'"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "exercise": {
                    "type": "string",
                    "description": "动作名称（精确匹配），如 '杠铃深蹲'、'杠铃硬拉'",
                },
            },
            "required": ["exercise"],
        },
    },
    {
        "name": "graph_reason_pain",
        "description": (
            "伤病因果推理：分析某动作与用户症状之间的关联，给出可能的原因和康复建议。"
            "推理链：动作 → MAY_CAUSE → 伤病 → RECOVERED_BY → 康复动作 + 应避免的动作。"
            "适用场景：用户说'我做卧推的时候肩膀疼，怎么回事?'"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "exercise": {
                    "type": "string",
                    "description": "引起症状的动作，如 '杠铃卧推'",
                },
                "symptom": {
                    "type": "string",
                    "description": "症状描述，如 '肩膀疼'、'腰酸'",
                },
            },
            "required": ["exercise", "symptom"],
        },
    },
]


class ToolRegistry:
    """工具注册中心，聚合管理所有可供 LLM Agent 调用的工具和资源。

    v2 升级：
      1. list_tools() 返回完整的 JSON Schema（含 type/enum/required/description）
      2. call() 失败时抛出 McpToolError 而非静默返回 None
      3. 新增 list_resources() / read_resource() 支持 MCP Resources 协议
      4. 内置健身知识库文档作为 Resources 暴露
    """

    # ---- 知识库资源 URI 前缀 ----
    KNOWLEDGE_URI_PREFIX = "knowledge://doc/"

    def __init__(self):
        """初始化工具注册中心。

        创建底层模块实例并注册所有工具。
        - ExerciseMCPServer: 包装了 FastMCP 实例，4 个动作库工具
        - GraphSearch: Neo4j 知识图谱，3 个图检索工具
        """
        self.exercise_mcp = ExerciseMCPServer()
        self.graph_search = GraphSearch()

        # ---- 工具调度表 ----
        # key=工具名, value=callable(params dict) -> result
        # 每个函数内部负责参数解构和错误检查
        self._tools: dict[str, Any] = {}
        self._register_tools()

        # ---- 工具 Schema 索引 ----
        # 合并 FastMCP 自动生成的 schema + 手动定义的 GraphRAG schema
        self._tool_schemas: dict[str, dict] = {}
        self._build_schema_index()

    # =====================================================================
    # 工具注册
    # =====================================================================

    def _register_tools(self):
        """注册所有工具到 self._tools 字典。

        分为三类：
          - 动作库工具（4 个）：委托给 ExerciseMCPServer / FastMCP
          - 图谱工具（3 个）：委托给 GraphSearch 实例
        """
        self._tools = {
            # --- 动作库工具（委托给 ExerciseMCPServer，其内部处理 FastMCP 返回格式解包）---
            "search_by_muscle": lambda p: self.exercise_mcp.call_tool(
                "search_by_muscle", p
            ),
            "search_by_equipment": lambda p: self.exercise_mcp.call_tool(
                "search_by_equipment", p
            ),
            "search_by_difficulty": lambda p: self.exercise_mcp.call_tool(
                "search_by_difficulty", p
            ),
            "get_exercise_detail": lambda p: self.exercise_mcp.call_tool(
                "get_exercise_detail", p
            ),

            # --- 知识图谱工具 ---
            "graph_multi_hop": lambda p: self.graph_search.multi_hop_search(
                p.get("equipment", ""), p.get("target", "")
            ),
            "graph_injury_risk": lambda p: self.graph_search.find_injury_risks(
                p.get("exercise", "")
            ),
            "graph_reason_pain": lambda p: self.graph_search.reason_about_pain(
                p.get("exercise", ""), p.get("symptom", "")
            ),
        }

    def _build_schema_index(self):
        """构建工具名 → JSON Schema 的索引。

        合并两个来源：
          1. FastMCP 自动生成的 schema（exercise_server.py 中 @mcp.tool() 装饰器）
             FastMCP 的 Tool 对象将参数 schema 存储在 .parameters 属性中，
             格式为 {"type": "object", "properties": {...}, "required": [...]}
          2. 手动定义的 GraphRAG schema（本文件 _GRAPH_TOOL_SCHEMAS）
        """
        # 从 FastMCP 提取 schema
        for tool_name, tool_info in exercise_mcp._tool_manager._tools.items():
            self._tool_schemas[tool_name] = {
                "name": tool_name,
                "description": tool_info.description or "",
                "inputSchema": (
                    tool_info.parameters
                    if hasattr(tool_info, "parameters")
                    else {"type": "object", "properties": {}}
                ),
            }

        # 合并 GraphRAG 手动定义的 schema
        for schema in _GRAPH_TOOL_SCHEMAS:
            self._tool_schemas[schema["name"]] = schema

    # =====================================================================
    # 公开 API：list_tools() —— 返回完整 JSON Schema
    # =====================================================================

    def list_tools(self) -> list[dict]:
        """列出所有已注册工具及其完整的 JSON Schema 定义。

        返回格式（MCP 标准）：
          [
            {
              "name": "search_by_muscle",
              "description": "按目标肌群搜索训练动作...",
              "inputSchema": {
                "type": "object",
                "properties": {
                  "muscle": {"type": "string", "description": "...", "enum": [...]},
                  "difficulty": {"type": "string", "enum": [...]}
                },
                "required": ["muscle"]
              }
            },
            ...
          ]

        与 v1 的区别：
          v1: [{"name": "search_by_muscle", "params": "dict"}, ...]
          v2: 完整的 JSON Schema，LLM 可以精确理解每个参数的类型、可选值、必填性
        """
        return list(self._tool_schemas.values())

    # =====================================================================
    # 公开 API：call() —— 带结构化错误
    # =====================================================================

    def call(self, tool_name: str, params: dict):
        """统一工具调用入口。

        Args:
            tool_name: 工具名称（self._tools 中的 key）
            params: 工具参数字典

        Returns:
            工具执行结果（类型取决于具体工具）

        Raises:
            McpToolError: 当工具不存在或执行失败时，携带标准 JSON-RPC 错误码。
                          Agent 可根据 error.code 决定下一步：
                            -32601 → 工具不存在，换一个工具
                            -32602 → 参数不对，修正参数重试
                            -32603 → 内部错误，放弃并报告用户


        与 v1 的区别：
          v1: 工具不存在时返回 None（静默失败）
          v2: 抛出 McpToolError（携带错误码，Agent 可感知失败原因）
        """
        if tool_name not in self._tools:
            available = ", ".join(sorted(self._tools.keys()))
            raise McpToolError(
                code=-32601,
                message="Method not found",
                tool_name=tool_name,
                details=f"可用工具: {available}",
            )

        try:
            result = self._tools[tool_name](params)
            return result
        except McpToolError:
            raise
        except ValueError as e:
            raise McpToolError(
                code=-32602,
                message="Invalid params",
                tool_name=tool_name,
                details=str(e),
            ) from e
        except Exception as e:
            logger.error(f"Tool '{tool_name}' failed: {e}")
            raise McpToolError(
                code=-32603,
                message="Internal error",
                tool_name=tool_name,
                details=str(e),
            ) from e

    # =====================================================================
    # 公开 API：Resources —— MCP Resources 协议支持
    # =====================================================================

    def list_resources(self) -> list[dict]:
        """列出所有可用的知识资源（MCP Resources 协议）。

        返回两类资源：
          1. 动作标准库资源（exercise:// URI）
             - exercise://library        — 动作库全量索引
             - exercise://muscles        — 肌群-动作覆盖表
             - exercise://standards/{name} — 单个动作的完整标准
          2. 健身知识库资源（knowledge://doc/ URI）
             - knowledge://doc/{filename} — 每篇知识文档的完整内容

        Returns:
            list[dict]: 每项包含 uri, name, description, mimeType
        """
        resources = []

        # ---- 动作标准库资源 ----
        resources.extend([
            {
                "uri": "exercise://library",
                "name": "动作库全量索引",
                "description": "按肌群分组列出所有可用训练动作，作为动作库目录",
                "mimeType": "text/markdown",
            },
            {
                "uri": "exercise://muscles",
                "name": "肌群-动作覆盖表",
                "description": "列出所有目标肌群及其对应的训练动作数量",
                "mimeType": "text/markdown",
            },
        ])

        # 动态列出每个动作的标准规范资源
        from src.mcp.exercise_server import EXERCISE_LIBRARY
        for ex in EXERCISE_LIBRARY:
            resources.append({
                "uri": f"exercise://standards/{ex['name']}",
                "name": f"{ex['name']} - 标准规范",
                "description": (
                    f"{ex['type']}动作 | {ex['difficulty']} | "
                    f"器械: {ex['equipment']} | "
                    f"肌群: {', '.join(ex['target_muscles'])}"
                ),
                "mimeType": "text/markdown",
            })

        # ---- 健身知识库资源 ----
        knowledge_files = self._find_knowledge_files()
        for filepath in knowledge_files:
            filename = os.path.basename(filepath)
            doc_name = filename.replace(".md", "").replace("_", " ").replace("-", " ")
            resources.append({
                "uri": f"knowledge://doc/{filename}",
                "name": doc_name,
                "description": f"健身知识文档: {doc_name}",
                "mimeType": "text/markdown",
            })

        return resources

    def read_resource(self, uri: str) -> str:
        """按 URI 读取资源内容（MCP Resources 协议）。

        Args:
            uri: 资源 URI，支持以下格式：
                 - exercise://library
                 - exercise://muscles
                 - exercise://standards/{动作名}
                 - knowledge://doc/{文件名}.md

        Returns:
            str: 资源内容（markdown 格式）

        Raises:
            McpToolError: 资源不存在时抛出（错误码 -32601）

        使用场景：
            Agent 在对话开始时调用 list_resources() 浏览可用知识 →
            然后对相关资源调用 read_resource(uri) 精确获取内容 →
            将内容作为上下文注入 LLM prompt。
        """
        # 委托给 exercise_server 的 FastMCP Resource 处理
        if uri.startswith("exercise://"):
            import asyncio as _asyncio
            try:
                result = _asyncio.run(exercise_mcp.read_resource(uri))
                # FastMCP 返回 list[ReadResourceContents]，提取文本内容
                if isinstance(result, list):
                    texts = []
                    for item in result:
                        if hasattr(item, "content"):
                            texts.append(item.content)
                        elif isinstance(item, str):
                            texts.append(item)
                    return "\n".join(texts)
                return str(result)
            except Exception as e:
                raise McpToolError(
                    code=-32601,
                    message="Resource not found",
                    tool_name="read_resource",
                    details=f"URI: {uri} — {e}",
                ) from e

        # 知识库文档资源
        if uri.startswith(self.KNOWLEDGE_URI_PREFIX):
            filename = uri[len(self.KNOWLEDGE_URI_PREFIX):]
            return self._read_knowledge_file(filename)

        raise McpToolError(
            code=-32601,
            message="Resource not found",
            tool_name="read_resource",
            details=f"未知 URI 前缀: {uri}",
        )

    # =====================================================================
    # 私有方法：知识库文件查找和读取
    # =====================================================================

    def _find_knowledge_files(self) -> list[str]:
        """扫描 data/knowledge/ 目录，返回所有 .md 文件路径。"""
        # 从项目根目录搜索
        project_root = Path(__file__).parent.parent.parent
        knowledge_dir = project_root / "data" / "knowledge"
        if not knowledge_dir.exists():
            return []
        return sorted(glob_mod.glob(str(knowledge_dir / "*.md")))

    def _read_knowledge_file(self, filename: str) -> str:
        """读取单篇知识库文档。

        安全性：只允许读取 .md 文件，防止路径遍历攻击。
        """
        # 安全检查：文件名只能是 .md
        if not filename.endswith(".md") or ".." in filename or "/" in filename or "\\" in filename:
            raise McpToolError(
                code=-32602,
                message="Invalid params",
                tool_name="read_resource",
                details=f"非法文件名: {filename}",
            )

        project_root = Path(__file__).parent.parent.parent
        filepath = project_root / "data" / "knowledge" / filename
        if not filepath.exists():
            raise McpToolError(
                code=-32601,
                message="Resource not found",
                tool_name="read_resource",
                details=f"知识库文件不存在: {filename}",
            )

        return filepath.read_text(encoding="utf-8")

    # =====================================================================
    # 便捷方法：获取工具描述（供 LLM system prompt 注入）
    # =====================================================================

    def get_tools_prompt(self) -> str:
        """生成工具列表的 LLM 可读描述，用于注入 system prompt。

        这是 list_tools() JSON Schema 的"人类可读版"，
        适合放在 prompt 中而非 function calling 的 tools 参数。

        Returns:
            str: 每个工具的名称 + 描述 + 参数列表
        """
        lines = ["可用工具："]
        for schema in self._tool_schemas.values():
            name = schema["name"]
            desc = schema.get("description", "")
            props = schema.get("inputSchema", {}).get("properties", {})
            required = schema.get("inputSchema", {}).get("required", [])

            param_strs = []
            for pname, pinfo in props.items():
                req_mark = " (必填)" if pname in required else ""
                enum_vals = pinfo.get("enum", [])
                enum_hint = f" 可选值: {enum_vals}" if enum_vals else ""
                param_strs.append(f"    {pname}: {pinfo.get('type', '?')}{req_mark}{enum_hint}")

            lines.append(f"\n  {name}")
            lines.append(f"    {desc}")
            lines.extend(param_strs)

        return "\n".join(lines)
