from src.mcp.exercise_server import ExerciseMCPServer
from src.graphrag.search import GraphSearch
from src.rag.vector_search import VectorSearch

class ToolRegistry:
    def __init__(self):
        self.exercise_mcp = ExerciseMCPServer()
        self.graph_search = GraphSearch()
        self.vector_search = VectorSearch()
        self._register_tools()

    def _register_tools(self):
        self.tools = {
            "search_by_muscle": lambda p: self.exercise_mcp.call_tool("search_by_muscle", p),
            "search_by_equipment": lambda p: self.exercise_mcp.call_tool("search_by_equipment", p),
            "search_by_difficulty": lambda p: self.exercise_mcp.call_tool("search_by_difficulty", p),
            "get_exercise_detail": lambda p: self.exercise_mcp.call_tool("get_exercise_detail", p),
            "graph_multi_hop": lambda p: self.graph_search.multi_hop_search(
                p.get("equipment", ""), p.get("target", "")),
            "graph_injury_risk": lambda p: self.graph_search.find_injury_risks(p.get("exercise", "")),
            "graph_reason_pain": lambda p: self.graph_search.reason_about_pain(
                p.get("exercise", ""), p.get("symptom", "")),
        }

    def list_tools(self) -> list:
        return [{"name": k, "params": "dict"} for k in self.tools]

    def call(self, tool_name: str, params: dict):
        if tool_name in self.tools:
            return self.tools[tool_name](params)
        return None
