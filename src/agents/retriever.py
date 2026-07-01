from src.rag.agentic_rag import AgenticRAG
from src.mcp.tool_registry import ToolRegistry

class RetrieverAgent:
    def __init__(self):
        self.agentic_rag = AgenticRAG()
        self.tools = ToolRegistry()

    def retrieve(self, plan: dict) -> dict:
        results = {"exercises": [], "knowledge": []}
        for subtask in plan.get("subtasks", []):
            filters = plan.get("skill_config", {}).get("retrieval_filters", {})
            rag_results = self.agentic_rag.search(subtask, filters=filters)
            results["exercises"].extend(rag_results)
        body_parts = self._extract_body_parts(plan)
        for part in body_parts:
            mcp_results = self.tools.call("search_by_muscle", {"muscle": part})
            if mcp_results:
                results["exercises"].extend(
                    [{"name": r["name"], "source": "mcp",
                      "muscles": r["muscles"], "equipment": r["equipment"],
                      "difficulty": r["difficulty"], "type": r["type"]}
                     for r in mcp_results]
                )
        return results

    def _extract_body_parts(self, plan: dict) -> list:
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
        return parts if parts else ["胸", "背", "腿"]
