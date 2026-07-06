"""
================================================================================
文件角色：动作标准库 MCP 服务（Exercise MCP Server）
================================================================================
- 被调用者：ToolRegistry 将本服务器的工具注册到统一工具调度表中。
  编排引擎通过 ToolRegistry.call("search_by_muscle", {...}) 间接调用本模块。
- 调用者：本模块是底层数据+工具层，不调用其他业务模块，仅依赖内置的
  EXERCISE_LIBRARY 静态数据。
- 项目角色：充当"动作百科全书"——提供按肌肉/器械/难度筛选动作的标准查询接口。

升级 v2 —— 完整 MCP 协议实现：
  v1 是"字典+lambda 注册表"，只借用了 MCP 的命名约定。
  v2 使用 FastMCP (官方 Python SDK)，实现完整 MCP 协议：
    - 第 1 层：Capability Negotiation（initialize 握手时公告 tools + resources 能力）
    - 第 2 层：Transport（支持 stdio 独立进程 或 in-process 直接调用）
    - 第 3 层：JSON-RPC 2.0 消息格式（由 FastMCP 自动处理）
    - 第 4 层：Lifecycle（initialize → initialized → shutdown，由 SDK 管理）
    - 第 5 层：Primitives ——
        Tools: @mcp.tool() 装饰器，type hints → JSON Schema（enum/required/type）
        Resources: @mcp.resource() 装饰器，URI 寻址的动作标准文档

  关键升级点：
    1. JSON Schema 从 "params": "dict" 升级为完整 schema（enum、required、type）
       → LLM function calling 准确率大幅提升
    2. Resources 机制让 Agent 可以先 list_resources() 浏览可用知识，
       再精确 read_resource(uri) 获取内容，替代盲目检索
    3. 结构化错误：MCP 协议规范了错误码（INVALID_PARAMS / METHOD_NOT_FOUND 等）
       → Agent 可以根据错误类型决定下一步（修正参数 vs 换工具 vs 放弃）
    4. 支持独立进程运行（python run_mcp_server.py）→ Claude Desktop 可接入
================================================================================
"""

from __future__ import annotations
import asyncio as _asyncio
from dataclasses import dataclass
from typing import Literal

from mcp.server.fastmcp import FastMCP


# =========================================================================
# McpToolError —— MCP 标准错误
# =========================================================================
# 定义在此处而非 tool_registry.py，避免循环导入。
# tool_registry.py 从本模块导入。

@dataclass
class McpToolError(Exception):
    """MCP 工具调用错误，携带标准 JSON-RPC 错误码。

    JSON-RPC 2.0 标准错误码：
        -32700: Parse error
        -32600: Invalid request
        -32601: Method not found     → Agent 知道该换一个工具
        -32602: Invalid params       → Agent 知道该修正参数
        -32603: Internal error       → Agent 知道该放弃重试
    """

    code: int
    message: str
    tool_name: str
    details: str = ""

    def __str__(self):
        return f"[{self.code}] {self.message} — tool={self.tool_name}: {self.details}"

    def to_dict(self) -> dict:
        """序列化为 MCP 标准错误响应格式。"""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "data": {
                    "tool_name": self.tool_name,
                    "details": self.details,
                },
            }
        }

# ---------------------------------------------------------------------------
# FastMCP 实例 —— 整个 MCP 协议的核心
# ---------------------------------------------------------------------------
# FastMCP 封装了 MCP 协议的完整生命周期：
#   - 启动时自动公告 tools + resources 能力
#   - 处理 JSON-RPC 消息序列化/反序列化
#   - 管理 initialize → initialized 握手流程
#   - 支持 stdio / SSE / streamable-HTTP 三种传输方式
# ---------------------------------------------------------------------------
mcp = FastMCP("GakkiFitnessExerciseServer")

# ---------------------------------------------------------------------------
# EXERCISE_LIBRARY: 动作标准库（静态数据）
# ---------------------------------------------------------------------------
# 每个动作的字段说明：
#   - name:       动作名称（中文），也是唯一标识
#   - equipment:  所需器械类型
#   - target_muscles: 目标肌肉列表（解剖学名称）
#   - difficulty: 难度等级
#   - type:       动作类型（复合=多关节/孤立=单关节）
#   - description: 动作标准做法描述
#   - common_mistakes: 常见错误
# ---------------------------------------------------------------------------
EXERCISE_LIBRARY = [
    {
        "name": "哑铃卧推",
        "equipment": "哑铃",
        "target_muscles": ["胸大肌", "三角肌前束", "肱三头肌"],
        "difficulty": "初级",
        "type": "复合",
        "description": "仰卧于平板凳上，双手各持哑铃置于胸部两侧，向上推举至手臂伸直但不锁死，顶峰收缩胸肌，缓慢下放至胸部两侧。全程保持肩胛骨收紧，避免耸肩。",
        "common_mistakes": ["手肘过度外展（应保持 45° 左右夹角）", "耸肩代偿", "下放过快失去张力"],
    },
    {
        "name": "杠铃深蹲",
        "equipment": "杠铃",
        "target_muscles": ["股四头肌", "臀大肌", "腘绳肌"],
        "difficulty": "中级",
        "type": "复合",
        "description": "杠铃置于斜方肌上方，双脚与肩同宽，脚尖略微外展。下蹲时保持核心收紧、背部挺直、膝盖与脚尖方向一致。下蹲至大腿与地面平行或稍低，然后发力站起。",
        "common_mistakes": ["膝盖内扣", "背部拱起（脊柱屈曲）", "脚跟抬起", "深度不足（半蹲）"],
    },
    {
        "name": "引体向上",
        "equipment": "自重",
        "target_muscles": ["背阔肌", "肱二头肌"],
        "difficulty": "中级",
        "type": "复合",
        "description": "双手正握（掌心朝前）单杠，比肩稍宽。从悬垂状态发力，想象肘部向身体两侧下拉，下巴过杠。缓慢下放至手臂完全伸展。",
        "common_mistakes": ["摆动借力", "只用手臂拉（未调动背部）", "半程动作"],
    },
    {
        "name": "哑铃侧平举",
        "equipment": "哑铃",
        "target_muscles": ["三角肌中束"],
        "difficulty": "初级",
        "type": "孤立",
        "description": "站立位，双手各持哑铃置于体侧。肘部微屈并保持该角度不变，从身体两侧平举哑铃至与肩同高，顶峰停顿后缓慢下放。肩部发力带动手臂，而非用手臂甩动。",
        "common_mistakes": ["哑铃举得过高（超过肩部导致斜方肌代偿）", "肘部弯曲角度变化", "身体摆动借力"],
    },
    {
        "name": "杠铃硬拉",
        "equipment": "杠铃",
        "target_muscles": ["腘绳肌", "臀大肌", "竖脊肌"],
        "difficulty": "高级",
        "type": "复合",
        "description": "杠铃置于脚前，双脚与髋同宽。屈髋屈膝下蹲握杠，保持背部挺直、核心收紧。发力时脚跟蹬地，同时伸髋伸膝将杠铃拉起，全程杠铃贴近身体。站直后肩胛骨收紧，缓慢下放。",
        "common_mistakes": ["背部拱起（极易导致腰椎间盘突出）", "杠铃远离身体", "先伸膝后伸髋（动作分解）"],
    },
    {
        "name": "绳索下压",
        "equipment": "绳索",
        "target_muscles": ["肱三头肌"],
        "difficulty": "初级",
        "type": "孤立",
        "description": "面对龙门架，双手握住绳索手柄，肘部固定于身体两侧。向下推压手柄至手臂伸直，顶峰收缩肱三头肌，缓慢返回至前臂与地面平行的起始位。全程大臂不动。",
        "common_mistakes": ["大臂前后摆动（借力）", "身体前倾过多", "动作幅度过大拉伤肘关节"],
    },
    {
        "name": "哑铃弯举",
        "equipment": "哑铃",
        "target_muscles": ["肱二头肌"],
        "difficulty": "初级",
        "type": "孤立",
        "description": "站立或坐姿，双手各持哑铃，掌心朝前。肘部固定于身体两侧，向上弯举哑铃至肩部高度，顶峰收缩肱二头肌，缓慢下方至手臂接近伸直。",
        "common_mistakes": ["身体摆动借力", "下放过快", "肘部前移（肩部代偿）"],
    },
    {
        "name": "腿举",
        "equipment": "腿举机",
        "target_muscles": ["股四头肌", "臀大肌"],
        "difficulty": "初级",
        "type": "复合",
        "description": "坐于腿举机上，双脚与肩同宽置于踏板上。解除安全杠后，弯曲膝盖至约 90°，然后发力推踏板至腿部接近伸直但不锁死膝盖。全程下背部紧贴靠垫。",
        "common_mistakes": ["膝盖完全锁死（关节受力过大）", "下背离开靠垫", "膝盖内扣"],
    },
]

# ---------------------------------------------------------------------------
# 可用的枚举值常量 —— 用于 JSON Schema 的 enum 约束
# ---------------------------------------------------------------------------
VALID_MUSCLES = sorted(set(
    m for ex in EXERCISE_LIBRARY for m in ex["target_muscles"]
))
VALID_EQUIPMENT = sorted(set(ex["equipment"] for ex in EXERCISE_LIBRARY))
VALID_DIFFICULTY = ["初级", "中级", "高级"]
VALID_EXERCISE_NAMES = sorted(ex["name"] for ex in EXERCISE_LIBRARY)


# =========================================================================
# MCP Tools —— @mcp.tool() 装饰器
# =========================================================================
# 每个装饰器自动做三件事：
#   1. 从 Python type hints 生成 JSON Schema（包括 enum、required、type）
#   2. 注册到 MCP 的 tools/list 响应中
#   3. 关联到 tools/call 的调度逻辑
#
# 对比 v1：
#   v1: {"name": "search_by_muscle", "params": "dict"}
#        → LLM 不知道 params 里该填什么
#   v2: {
#     "name": "search_by_muscle",
#     "inputSchema": {
#       "type": "object",
#       "properties": {
#         "muscle": {"type": "string", "description": "...", "enum": [...]},
#         "difficulty": {"type": "string", "enum": ["初级","中级","高级"]}
#       },
#       "required": ["muscle"]
#     }
#   } → LLM 知道精确的可用值、必填字段、参数含义
# =========================================================================


@mcp.tool()
def search_by_muscle(
    muscle: str,
    difficulty: Literal["初级", "中级", "高级"] | None = None,
) -> list[dict]:
    """按目标肌群搜索训练动作。

    适用场景：用户说"给我推荐练胸的动作"或"有什么练背的动作推荐"。

    Args:
        muscle: 目标肌群名称，如 "胸大肌"、"背阔肌"、"股四头肌" 等。
                支持子串模糊匹配，搜"胸"能命中"胸大肌"。
        difficulty: 可选难度筛选。"初级"适合新手/"中级"适合有训练经验者/"高级"适合老手。
    """
    q = muscle.lower()
    results = [
        ex for ex in EXERCISE_LIBRARY
        if any(q in m.lower() for m in ex["target_muscles"])
    ]
    if difficulty:
        results = [ex for ex in results if ex["difficulty"] == difficulty]
    return results


@mcp.tool()
def search_by_equipment(equipment: str) -> list[dict]:
    """按可用器械搜索训练动作。

    适用场景：用户说"我只有哑铃，能练什么动作？"或"家里只有弹力带，能练什么？"

    Args:
        equipment: 器械名称，如 "哑铃"、"杠铃"、"自重"、"绳索"、"腿举机"。
                   支持子串模糊匹配。
    """
    q = equipment.lower()
    return [ex for ex in EXERCISE_LIBRARY if q in ex["equipment"].lower()]


@mcp.tool()
def search_by_difficulty(
    difficulty: Literal["初级", "中级", "高级"],
) -> list[dict]:
    """按难度等级搜索训练动作。

    适用场景：用户问"有哪些适合新手的动作？"或"我是初学者，该从什么动作开始？"

    Args:
        difficulty: 难度等级。"初级"=适合新手/"中级"=需要一定训练基础/"高级"=需要掌握正确技术。
    """
    return [ex for ex in EXERCISE_LIBRARY if ex["difficulty"] == difficulty]


@mcp.tool()
def get_exercise_detail(name: str) -> list[dict]:
    """获取某个动作的完整技术详情。

    适用场景：用户问"硬拉应该怎么做？"或"卧推的正确姿势是什么？"

    返回字段（列表中的单元素）：
        name: 动作名
        equipment: 所需器械
        target_muscles: 目标肌群
        difficulty: 难度
        type: 复合/孤立
        description: 标准做法描述（分步骤）
        common_mistakes: 常见错误列表

    Args:
        name: 动作名称（精确匹配），如 "杠铃深蹲"、"引体向上"、"哑铃卧推"。
    """
    for ex in EXERCISE_LIBRARY:
        if ex["name"] == name:
            return [ex]
    raise ValueError(f"未找到动作 '{name}'，可用动作: {', '.join(VALID_EXERCISE_NAMES)}")


# =========================================================================
# MCP Resources —— @mcp.resource() 装饰器
# =========================================================================
# Resources 是 MCP 的 "GET 端点" —— 用 URI 寻址的只读数据。
# 与 Tools 的区别：
#   - Tools 是 "POST"：LLM 决定何时调用、传什么参数
#   - Resources 是 "GET"：App/Agent 在对话前就加载好上下文
#   - Resources 有明确的 URI，可以被 list 和 read
#
# 实战价值：
#   Agent 接到问题后，先 resources/list 看看有哪些知识领域可用 →
#   然后精确 resources/read 需要的文档 →
#   而不是盲目把所有文档 chunk 扔进 prompt。
# =========================================================================


@mcp.resource("exercise://standards/{name}")
def exercise_standard(name: str) -> str:
    """获取某个动作的标准执行规范。

    返回该动作的完整标准：做法描述 + 常见错误。
    URI 示例：exercise://standards/杠铃深蹲
    """
    for ex in EXERCISE_LIBRARY:
        if ex["name"] == name:
            return (
                f"=== {ex['name']} ===\n"
                f"类型: {ex['type']} | 难度: {ex['difficulty']} | 器械: {ex['equipment']}\n"
                f"目标肌群: {', '.join(ex['target_muscles'])}\n\n"
                f"标准做法:\n{ex['description']}\n\n"
                f"常见错误:\n" + "\n".join(f"  - {m}" for m in ex['common_mistakes'])
            )
    return f"未找到动作: {name}"


@mcp.resource("exercise://library")
def exercise_library_index() -> str:
    """动作库全量索引 —— 按肌群分组列出所有可用动作。

    供 Agent 在对话开始时加载，作为"目录"使用。
    """
    by_muscle: dict[str, list[str]] = {}
    for ex in EXERCISE_LIBRARY:
        for m in ex["target_muscles"]:
            by_muscle.setdefault(m, []).append(ex["name"])

    lines = ["# 动作库索引 (共 {} 个动作)".format(len(EXERCISE_LIBRARY)), ""]
    for muscle, exercises in sorted(by_muscle.items()):
        lines.append(f"## {muscle}")
        for name in exercises:
            ex = next(e for e in EXERCISE_LIBRARY if e["name"] == name)
            lines.append(f"  - {name} ({ex['equipment']}, {ex['difficulty']})")
        lines.append("")
    return "\n".join(lines)


@mcp.resource("exercise://muscles")
def muscle_groups() -> str:
    """列出所有目标肌群及对应的训练动作数量。

    供 Agent 在推荐动作时参考，了解知识覆盖范围。
    """
    from collections import Counter
    counts = Counter(
        m for ex in EXERCISE_LIBRARY for m in ex["target_muscles"]
    )
    lines = ["# 肌群-动作覆盖", ""]
    for muscle, count in counts.most_common():
        lines.append(f"  - {muscle}: {count} 个动作")
    return "\n".join(lines)


# =========================================================================
# 兼容层：ExerciseMCPServer
# =========================================================================
# 保留 v1 的 class 接口，确保 ToolRegistry 和测试代码无需修改即可运行。
# 内部委托给 FastMCP 实例的 call_tool。
# =========================================================================

class ExerciseMCPServer:
    """动作标准库 MCP 服务器（兼容 v1 API 的包装器）。

    内部使用 FastMCP 实例来管理工具注册和 JSON Schema 生成，
    但对外暴露与 v1 完全相同的 list_tools() / call_tool() 同步接口。

    如果想用完整 MCP 协议（独立进程、JSON-RPC 传输），
    请使用 python run_mcp_server.py 启动 stdio 服务器。
    """

    def list_tools(self) -> list:
        """列出所有可用工具及其完整的 JSON Schema 定义。

        与 v1 的区别：
            v1: {"name": "search_by_muscle", "params": "dict"}
            v2: 包含完整的 inputSchema（type, properties, required, enum 等）
        """
        return _get_tools_schema()

    def call_tool(self, tool_name: str, params: dict) -> list | dict:
        """执行工具调用。

        v2 变更：
            - 未知工具: 抛出 McpToolError(code=-32601) 而非返回 []
            - 参数无效: 抛出 McpToolError(code=-32602) 而非返回 []
            - 内部错误: 抛出 McpToolError(code=-32603) 而非静默吞错
            这让 Agent 能感知失败原因并做出对应决策。
        """
        try:
            result = _asyncio.run(_call_tool_async(tool_name, params))
            return result
        except McpToolError:
            raise
        except ValueError as e:
            raise McpToolError(
                code=-32602, message="Invalid params",
                tool_name=tool_name, details=str(e)
            ) from e
        except Exception as e:
            raise McpToolError(
                code=-32603, message="Internal error",
                tool_name=tool_name, details=str(e)
            ) from e


async def _call_tool_async(tool_name: str, params: dict):
    """异步调用 FastMCP 工具并解包返回结果。

    FastMCP 的 call_tool() 返回格式取决于工具是否有 output schema：
      - 有 output schema（如 -> list[dict]）：返回 (list[TextContent], dict)
        其中 dict 格式为 {'result': <实际返回值>}
      - 无 output schema（如 -> dict）：返回 list[TextContent]
        需要从 TextContent.text 中解析 JSON

    本函数统一解包，返回原始 Python 对象（list 或 dict）。
    """
    import json

    result = await mcp.call_tool(tool_name, arguments=params)

    # 情况 1: 元组格式 (content_list, structured_output)
    if isinstance(result, tuple) and len(result) == 2:
        structured = result[1]
        # structured 格式为 {'result': <实际数据>}
        if isinstance(structured, dict) and "result" in structured:
            return structured["result"]
        return structured

    # 情况 2: 列表格式 [TextContent, ...]
    if isinstance(result, list):
        text = result[0].text if result else ""
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text

    return result


def _get_tools_schema() -> list:
    """从 FastMCP 实例提取完整的工具 JSON Schema 列表。

    FastMCP 的 Tool 对象将参数 schema 存储在 .parameters 属性中
    （而非 .inputSchema），格式为:
      {"type": "object", "properties": {...}, "required": [...]}
    这些 schema 由 @mcp.tool() 装饰器从 Python type hints 自动生成。
    """
    schemas = []
    for tool_name, tool_info in mcp._tool_manager._tools.items():
        schema = {
            "name": tool_name,
            "description": tool_info.description or "",
            "inputSchema": (
                tool_info.parameters
                if hasattr(tool_info, "parameters")
                else {"type": "object", "properties": {}}
            ),
        }
        schemas.append(schema)
    return schemas
