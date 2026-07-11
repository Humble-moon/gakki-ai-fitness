"""
===========================================================================
文件角色：检索器 Agent —— 根据 Planner 产出的子任务，从多源检索训练动作数据
===========================================================================
- 被谁调用：Orchestrator 在流水线的第 3 步调用 RetrieverAgent.retrieve()
- 调用谁：
    AgenticRAG.search() → 语义向量检索 + 自校正循环，获取与子任务最相关的动作
    ToolRegistry.call("search_by_muscle") → 通过 MCP 协议按肌肉名称精确检索动作数据
- 核心职责：
    1. 遍历 Planner 产出的每个子任务，调用 AgenticRAG 进行语义检索
    2. 从 Plan 中提取目标肌群，通过 MCP 工具按肌肉名称补充检索
    3. 合并 RAG 和 MCP 两路结果，统一去重后返回
- 数据流：
    Planner 输出 plan dict → RetrieverAgent.retrieve() → 返回 {"exercises": [...], "knowledge": [...]}
    → 传递给 WriterAgent 用于生成训练计划
===========================================================================
"""

from src.rag.agentic_rag import AgenticRAG
from src.mcp.tool_registry import ToolRegistry


class RetrieverAgent:
    """检索器 Agent：在 Orchestrator 流水线的第 3 步被调用。
    职责：根据规划结果从动作库（AgenticRAG）和外部数据源（MCP）检索训练动作。
    采用双路检索策略：语义检索（RAG）+ 结构化查询（MCP），互补提高召回率。"""

    def __init__(self):
        # AgenticRAG：支持自校正循环的智能 RAG 检索器
        # 如果初次检索结果不够，会自动改写查询词重新检索
        self.agentic_rag = AgenticRAG()
        # ToolRegistry：MCP 工具注册表，提供 search_by_muscle 等结构化查询能力
        self.tools = ToolRegistry()

    def retrieve(self, plan: dict) -> dict:
        """根据规划结果执行多源检索。

        输入：
            plan: dict — Planner 产出的规划字典，包含：
                - "subtasks": list[str] — 子任务列表（如 ["胸部训练", "背部训练"]）
                - "skill_config.retrieval_filters": dict — 技能模板定义的过滤条件
        输出：
            dict — 包含 "exercises" 和 "knowledge" 两个列表：
                - "exercises": 检索到的训练动作列表（每项含 name, source, muscles 等）
                - "knowledge": 关联的健身知识片段

        双路检索策略说明：
            【路 1 — AgenticRAG 语义检索】
                遍历每个子任务文本 → 调用 AgenticRAG.search() → 语义向量匹配
                → 适用于"概念级"匹配（如 "增肌阶段的胸部训练"）
            【路 2 — MCP 结构化检索】
                从子任务中提取目标肌群 → 按肌肉名称精确查询外部数据库
                → 适用于"精确"查询（如直接搜"胸肌"的所有动作）
            两路合并可兼顾语义覆盖面和精确匹配，提高最终计划的动作丰富度。
        """
        results = {"exercises": [], "knowledge": []}

        # === 路 1：AgenticRAG 语义检索 ===
        # 对每个子任务逐一检索。AgenticRAG 内部含自校正循环：
        # 检索 → 评估质量 → 不够好就改写查询 → 再检索，直到满意或达到最大轮次
        for subtask in plan.get("subtasks", []):
            filters = plan.get("skill_config", {}).get("retrieval_filters", {})
            rag_results = self.agentic_rag.search(subtask, filters=filters)
            results["exercises"].extend(rag_results)

        # === 路 2：MCP 按肌肉名称精确检索 ===
        # 从子任务文本中提取目标肌群（推→胸/肩/三头，拉→背/二头，腿→腿/臀）
        # 然后对每个肌群调用 MCP 工具精确查询
        body_parts = self._extract_body_parts(plan)
        for part in body_parts:
            mcp_results = self.tools.call("search_by_muscle", {"muscle": part})
            if mcp_results:
                # 统一字段名格式，标记来源为 "mcp"
                results["exercises"].extend(
                    [{"name": r["name"], "source": "mcp",
                      "muscles": r.get("target_muscles", r.get("muscles", [])),
                      "equipment": r["equipment"], "difficulty": r["difficulty"],
                      "type": r["type"]}
                     for r in mcp_results]
                )
        return results

    def _extract_body_parts(self, plan: dict) -> list:
        """【私有方法】从 Planner 产出的子任务文本中提取目标肌群名称。

        输入：
            plan: dict — Planner 产出的规划字典
        输出：
            list[str] — 目标肌群列表，如 ["胸", "肩", "三头", "背", "二头"]

        映射规则：
            - 子任务含 "推" → 胸、肩、三头（推类动作共同发力的肌群）
            - 子任务含 "拉" → 背、二头（拉类动作共同发力的肌群）
            - 子任务含 "腿" → 腿、臀
        兜底：如果没有匹配到任何肌群，默认返回 ["胸", "背", "腿"]，确保检索不空

        设计意图：
            Planner 的 LLM 拆解输出可能不含明确的肌肉名称（如只写"上肢训练"），
            通过训练动作分类（推/拉/腿）反向推导目标肌群，确保 MCP 检索有锚点。
        """
        parts_map = {
            "推": ["胸", "肩", "三头"],
            "拉": ["背", "二头"],
            "腿": ["腿", "臀"],
        }
        subtasks_str = "".join(plan.get("subtasks", []))
        parts = []
        for key, vals in parts_map.items():
            if key in subtasks_str:
                parts.extend(vals)
        # 兜底：如果完全没有匹配，返回最常见的三大肌群，确保检索不会空跑
        return parts if parts else ["胸", "背", "腿"]
