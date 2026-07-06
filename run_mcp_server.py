"""
================================================================================
run_mcp_server.py —— 完整的 MCP 协议服务器启动器
================================================================================
角色：将 gakki-ai-fitness 的工具和知识资源作为标准 MCP 服务器暴露，
     支持通过 stdio/SSE/HTTP 三种传输方式被外部 MCP 客户端连接。

真正的 MCP 协议：
  这个文件启动的是真正的 MCP 服务器 —— 它通过 stdio 管道接收 JSON-RPC 2.0
  消息，处理 tools/list、tools/call、resources/list、resources/read 等请求，
  返回标准 JSON-RPC 响应。可以被 Claude Desktop、MCP Inspector 或任何
  MCP 兼容的客户端连接。

与项目内部的 ToolRegistry 的关系：
  - ToolRegistry（tool_registry.py）：项目内部使用的工具门面。
    Agent 在同一进程内通过 Python 函数调用使用工具。
  - run_mcp_server.py：对外暴露的 MCP 协议接口。
    外部 MCP 客户端通过 JSON-RPC 消息发现和调用工具。
    两者共享相同的底层工具实现（ExerciseMCPServer + GraphSearch）。

为什么不直接用 ToolRegistry 做 MCP Server？
  ToolRegistry 中有 GraphSearch 工具依赖 Neo4j 连接。如果作为 stdio 服务器
  启动，需要确保 Neo4j 可用。这里的实现做了防御：GraphRAG 工具仅在运行时
  能连接 Neo4j 时才注册，无法连接时优雅降级。

启动方式：
  # stdio 模式（用于 Claude Desktop 集成）
  python run_mcp_server.py

  # HTTP 模式（用于 Web 客户端）
  python run_mcp_server.py --transport streamable-http --port 8504

  # SSE 模式
  python run_mcp_server.py --transport sse --port 8504

Claude Desktop 配置示例（claude_desktop_config.json）：
  {
    "mcpServers": {
      "gakki-fitness": {
        "command": "python",
        "args": ["run_mcp_server.py"],
        "cwd": "E:/gakki-ai-fitness"
      }
    }
  }

MCP 完整协议在此文件中的体现：
  - 传输层: stdio / SSE / Streamable HTTP（由 FastMCP 自动处理）
  - JSON-RPC: 所有消息的编解码（由 FastMCP 自动处理）
  - 生命周期: initialize → initialized → ... → shutdown（由 FastMCP 管理）
  - 原语:
      Tools:     由 @mcp.tool() 装饰器定义 → tools/list + tools/call
      Resources: 由 @mcp.resource() 装饰器定义 → resources/list + resources/read
================================================================================
"""

import argparse
import logging
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

# =========================================================================
# 创建 FastMCP 服务器实例
# =========================================================================
# FastMCP("name") 在 MCP 协议中的作用：
#   1. 服务器身份标识 —— 在 initialize 响应中作为 serverInfo.name 返回
#   2. 工具命名空间 —— 所有 @mcp.tool() 注册到此实例
#   3. 传输绑定 —— mcp.run() 启动时绑定到指定传输方式
# =========================================================================
mcp = FastMCP("GakkiFitnessMCP")

# =========================================================================
# 导入并注册核心工具
# =========================================================================
# 从 exercise_server 导入 FastMCP 定义的工具。
# 注意：exercise_server.py 中已经用 @exercise_mcp.tool() 注册了 4 个工具，
# 但那是另一个 FastMCP 实例。这里我们直接复用函数逻辑。
# =========================================================================

from src.mcp.exercise_server import (  # noqa: E402
    EXERCISE_LIBRARY,
    get_exercise_detail,
    search_by_difficulty,
    search_by_equipment,
    search_by_muscle,
)

# 直接在本服务器的 FastMCP 实例上注册工具
# 这些工具会出现在 tools/list 响应中
mcp.tool()(search_by_muscle)
mcp.tool()(search_by_equipment)
mcp.tool()(search_by_difficulty)
mcp.tool()(get_exercise_detail)

logger = logging.getLogger(__name__)


# =========================================================================
# GraphRAG 工具 —— 条件注册
# =========================================================================
# GraphSearch 依赖 Neo4j，如果 Neo4j 不可用则不注册，优雅降级。
# MCP 协议支持在 tools/list 中只列出当前可用的工具，客户端会据此调整行为。
# =========================================================================

def _try_register_graph_tools():
    """尝试注册 GraphRAG 图谱检索工具。Neo4j 不可用时优雅降级。"""
    try:
        from src.graphrag.search import GraphSearch
        gs = GraphSearch()

        @mcp.tool()
        def graph_multi_hop(equipment: str, target: str) -> list:
            """多跳组合查询：同时按器械和肌群搜索训练动作。利用知识图谱中的关系链（动作→器械 + 动作→肌群）找到符合条件的动作。适用场景："用哑铃练胸的动作有哪些？"

            Args:
                equipment: 器材名称，如 '哑铃'、'杠铃'、'自重'
                target: 目标肌群，如 '胸'、'背'、'腿'
            """
            return gs.multi_hop_search(equipment, target)

        @mcp.tool()
        def graph_injury_risk(exercise: str) -> list:
            """查询某动作的已知伤病风险。适用场景："深蹲会不会伤膝盖？"

            Args:
                exercise: 动作名称（精确匹配），如 '杠铃深蹲'
            """
            return gs.find_injury_risks(exercise)

        @mcp.tool()
        def graph_reason_pain(exercise: str, symptom: str) -> dict:
            """伤病因果推理：分析动作与症状的关联，给出可能原因和康复建议。推理链：动作→伤病→康复动作+应避免动作。适用场景："我做卧推肩膀疼，怎么回事？"

            Args:
                exercise: 引起症状的动作，如 '杠铃卧推'
                symptom: 症状描述，如 '肩膀疼'
            """
            return gs.reason_about_pain(exercise, symptom)

        logger.info("GraphRAG 工具已注册 (Neo4j 可用)")
    except Exception as e:
        logger.warning(f"GraphRAG 工具未注册 (Neo4j 不可用): {e}")


# =========================================================================
# 知识库 Resources
# =========================================================================

def _register_knowledge_resources():
    """将 data/knowledge/ 下的文档注册为 MCP Resources。

    MCP Resources 协议：
      - Client 发 resources/list → Server 返回可用资源列表（URI + 描述）
      - Client 发 resources/read {uri: "knowledge://xxx"} → Server 返回内容

    实际效果：外部 MCP 客户端（如 Claude Desktop）可以先浏览健身知识库的
    文档目录，然后精确读取感兴趣的文档，而不是把所有内容塞进 prompt。
    """
    knowledge_dir = Path(__file__).parent / "data" / "knowledge"
    if not knowledge_dir.exists():
        logger.warning("知识库目录不存在，跳过 Resource 注册")
        return

    for md_file in sorted(knowledge_dir.glob("*.md")):
        file_uri = f"knowledge://doc/{md_file.name}"

        # 闭包陷阱：需要在工厂函数中捕获 md_file 的值
        def make_reader(path):
            def reader() -> str:
                return path.read_text(encoding="utf-8")
            return reader

        reader_fn = make_reader(md_file)
        doc_name = md_file.stem.replace("_", " ").replace("-", " ")

        mcp.resource(file_uri, name=doc_name, description=f"健身知识文档: {doc_name}")(reader_fn)

    logger.info(f"已注册 {len(list(knowledge_dir.glob('*.md')))} 个知识库 Resources")


# =========================================================================
# 主入口
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GakkiFitness MCP Server —— AI 健身私教工具和知识库"
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="传输方式 (默认: stdio，用于 Claude Desktop 集成)",
    )
    parser.add_argument(
        "--port", type=int, default=8504,
        help="HTTP/SSE 模式的监听端口 (默认: 8504)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="HTTP/SSE 模式的监听地址 (默认: 127.0.0.1)",
    )
    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger.info(f"启动 GakkiFitness MCP Server (transport={args.transport})")

    # 条件注册 GraphRAG 工具
    _try_register_graph_tools()

    # 注册知识库 Resources
    _register_knowledge_resources()

    # 启动 MCP 协议服务器
    # mcp.run() 内部做的事：
    #   1. 绑定到指定传输层（stdio/SSE/HTTP）
    #   2. 开始监听 JSON-RPC 消息
    #   3. 等待客户端 initialize 握手
    #   4. 处理 tools/list、tools/call、resources/list、resources/read 等请求
    #   5. 直到收到 shutdown 或进程被终止
    logger.info("MCP 服务器已就绪，等待客户端连接...")
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    elif args.transport == "streamable-http":
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
