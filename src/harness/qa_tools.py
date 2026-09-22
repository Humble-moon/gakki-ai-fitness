"""知识问答的工具门面 —— 把领域检索能力暴露给自主循环。

## 为什么需要这一层

`src/harness/loop.py` 只要求工具集满足两个方法（`call` / `list_tools`），
这在设计上是为了让循环与具体工具解耦。但问答链路的三类检索能力分散在
不同模块、签名各异：

* 知识库检索 —— `KnowledgeSearch.search_with_fallback(query)`，参数是裸字符串；
* 动作库检索 —— `RetrieverAgent.retrieve(plan_dict)`，参数是一个 dict plan；
* 图谱推理   —— `ToolRegistry.call("graph_reason_pain", {...})`，得先知道工具名。

这三者没有一个统一的"给模型看的名字 + JSON Schema + 统一入参"表面。
本模块就是补这一层：**它不做检索，只做适配**——把三种入参风格收敛成
`{"query": "..."}` 这类模型能理解的形状。

## 与 ToolRegistry 的分工

`src/mcp/tool_registry.py` 是**面向外部 MCP 客户端**的完整协议实现
（9 个工具、JSON-RPC 错误码）。本门面是**面向本次问答**的裁剪视图：
只暴露对回答问题真正有用的三类能力，且把参数收敛得更简单。

直接复用 ToolRegistry 也不是不行，但那样模型要在 9 个动作库工具里自己
挑（`search_by_muscle` / `search_by_equipment` / `search_by_difficulty`…），
多数问题其实只需要"按语义查一下"。门面把默认路径收敛成一条，降低模型
选错的概率。需要精确查询时模型仍可显式指名——见下面 `search_exercises`
的 `mode` 参数。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: 回喂给模型的单条结果上限。循环层还会再截断一次（500 字符），
#: 这里先按"条数"收口，避免一次检索把几十条塞进上下文。
MAX_ITEMS_PER_CALL = 5


def _preview(text: Any, limit: int = 160) -> str:
    """把任意结果压成一行短摘要，供模型阅读。"""
    s = "" if text is None else str(text).replace("\n", " ").strip()
    return s[:limit]


class QAToolKit:
    """知识问答场景下的工具集，满足 `harness.loop.ToolInvoker` 协议。

    三个协作者都是**注入**的，便于测试替换：

    Args:
        knowledge: 提供 ``search_with_fallback(query) -> list[dict]``。
        retriever: 提供 ``retrieve(plan) -> {"exercises": [...], "knowledge": [...]}``。
        tool_registry: 提供 ``call(name, params)``，用于图谱推理。
            可为 None（图谱不可用时），此时 `reason_injury` 会返回明确错误
            而不是抛异常——循环会把错误回喂模型让它换个思路。
    """

    def __init__(self, knowledge: Any, retriever: Any, tool_registry: Any = None):
        self.knowledge = knowledge
        self.retriever = retriever
        self.tool_registry = tool_registry

    # ------------------------------------------------------------------
    # ToolInvoker 协议
    # ------------------------------------------------------------------

    def list_tools(self) -> list[dict]:
        """返回给模型看的工具清单（MCP 风格 schema）。"""
        tools = [
            {
                "name": "search_knowledge",
                "description": (
                    "检索健身知识库，返回相关科普片段。回答'为什么''怎么做'"
                    "这类需要解释的问题时使用。"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "检索用的关键词或短句"}
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "search_exercises",
                "description": (
                    "检索训练动作库，返回动作名称、目标肌群与要领。"
                    "需要给出具体训练动作建议时使用。"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "动作名或训练部位"},
                        "mode": {
                            "type": "string",
                            "enum": ["semantic", "by_muscle"],
                            "description": (
                                "semantic（默认）按语义查；"
                                "by_muscle 按目标肌群精确查，需要 query 填肌群名"
                            ),
                        },
                    },
                    "required": ["query"],
                },
            },
        ]

        # 图谱工具只在注册表可用时暴露——不可用时列出来只会让模型白试一次。
        if self.tool_registry is not None:
            tools.append({
                "name": "reason_injury",
                "description": (
                    "基于'动作→肌肉→伤病'知识图谱做多跳推理。"
                    "仅在问题涉及疼痛、伤病或不适时使用。"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "exercise": {"type": "string", "description": "相关动作名"},
                        "symptom": {"type": "string", "description": "症状描述"},
                    },
                    "required": ["exercise", "symptom"],
                },
            })

        return tools

    def call(self, name: str, args: dict) -> Any:
        """统一工具调用入口。

        返回值一律是**可 JSON 序列化的 dict**，因为循环会把结果转成字符串
        回喂模型。抛异常表示"这条路真的断了"（如未知工具名），循环会捕获
        并回喂错误而非中断——见 `harness.loop._invoke_tool`。
        """
        args = args or {}

        if name == "search_knowledge":
            return self._search_knowledge(args)
        if name == "search_exercises":
            return self._search_exercises(args)
        if name == "reason_injury":
            return self._reason_injury(args)

        raise ValueError(f"未知工具: {name}")

    # ------------------------------------------------------------------
    # 具体实现
    # ------------------------------------------------------------------

    def _search_knowledge(self, args: dict) -> dict:
        query = str(args.get("query") or "").strip()
        if not query:
            return {"error": "empty_query", "message": "query 不能为空"}

        chunks = self.knowledge.search_with_fallback(query) or []
        return {
            "count": len(chunks),
            "items": [
                {
                    "text": _preview(c.get("text") or c.get("content")),
                    "source": _preview(c.get("source") or c.get("doc_id"), 60),
                    "score": c.get("rerank_score") or c.get("rrf_score"),
                }
                for c in chunks[:MAX_ITEMS_PER_CALL]
            ],
        }

    def _search_exercises(self, args: dict) -> dict:
        query = str(args.get("query") or "").strip()
        if not query:
            return {"error": "empty_query", "message": "query 不能为空"}

        mode = str(args.get("mode") or "semantic")

        # by_muscle 走 ToolRegistry 的精确查询；其余走 Retriever 的语义检索。
        if mode == "by_muscle" and self.tool_registry is not None:
            raw = self.tool_registry.call("search_by_muscle", {"muscle": query})
            items = raw if isinstance(raw, list) else []
            return {
                "count": len(items),
                "mode": "by_muscle",
                "items": [
                    {
                        "name": _preview(e.get("name"), 60),
                        "muscles": e.get("target_muscles"),
                        "equipment": _preview(e.get("equipment"), 40),
                    }
                    for e in items[:MAX_ITEMS_PER_CALL]
                    if isinstance(e, dict)
                ],
            }

        # 复用 Retriever 的既有流水线：传一个最小 plan，语义检索该 query。
        retrieved = self.retriever.retrieve({"subtasks": [query], "skill_config": {}})
        exercises = (retrieved or {}).get("exercises", []) or []
        return {
            "count": len(exercises),
            "mode": "semantic",
            "items": [
                {
                    "name": _preview(e.get("name"), 60),
                    "muscles": e.get("target_muscles") or e.get("muscles"),
                    "source": _preview(e.get("source"), 40),
                }
                for e in exercises[:MAX_ITEMS_PER_CALL]
                if isinstance(e, dict)
            ],
        }

    def _reason_injury(self, args: dict) -> dict:
        if self.tool_registry is None:
            return {
                "error": "graph_unavailable",
                "message": "知识图谱当前不可用，请改用 search_knowledge 检索伤病相关文档",
            }

        exercise = str(args.get("exercise") or "").strip()
        symptom = str(args.get("symptom") or "").strip()
        if not exercise or not symptom:
            return {
                "error": "missing_argument",
                "message": "reason_injury 需要 exercise 与 symptom 两个参数",
            }

        result = self.tool_registry.call(
            "graph_reason_pain", {"exercise": exercise, "symptom": symptom}
        )
        # 图谱返回结构随实现变化，这里不做假设，原样透传给模型。
        return {"result": result}
