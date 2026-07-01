"""Exercise Standard Library MCP Server.
Provides standardized exercise library queries: filter by equipment, muscle, difficulty."""

EXERCISE_LIBRARY = [
    {"name": "哑铃卧推", "equipment": "哑铃", "muscles": ["胸大肌", "三角肌前束", "肱三头肌"],
     "difficulty": "初级", "type": "复合"},
    {"name": "杠铃深蹲", "equipment": "杠铃", "muscles": ["股四头肌", "臀大肌", "腘绳肌"],
     "difficulty": "中级", "type": "复合"},
    {"name": "引体向上", "equipment": "自重", "muscles": ["背阔肌", "肱二头肌"],
     "difficulty": "中级", "type": "复合"},
    {"name": "哑铃侧平举", "equipment": "哑铃", "muscles": ["三角肌中束"],
     "difficulty": "初级", "type": "孤立"},
    {"name": "杠铃硬拉", "equipment": "杠铃", "muscles": ["腘绳肌", "臀大肌", "竖脊肌"],
     "difficulty": "高级", "type": "复合"},
    {"name": "绳索下压", "equipment": "绳索", "muscles": ["肱三头肌"],
     "difficulty": "初级", "type": "孤立"},
    {"name": "哑铃弯举", "equipment": "哑铃", "muscles": ["肱二头肌"],
     "difficulty": "初级", "type": "孤立"},
    {"name": "腿举", "equipment": "腿举机", "muscles": ["股四头肌", "臀大肌"],
     "difficulty": "初级", "type": "复合"},
]

class ExerciseMCPServer:
    """Simulated MCP Server interface: tools/list + tools/call"""

    def list_tools(self) -> list:
        return [
            {"name": "search_by_muscle", "description": "按目标肌肉搜索动作",
             "parameters": {"muscle": "string"}},
            {"name": "search_by_equipment", "description": "按器械搜索动作",
             "parameters": {"equipment": "string"}},
            {"name": "search_by_difficulty", "description": "按难度搜索动作",
             "parameters": {"difficulty": "string"}},
            {"name": "get_exercise_detail", "description": "获取动作详情",
             "parameters": {"name": "string"}},
        ]

    def call_tool(self, tool_name: str, params: dict) -> list:
        if tool_name == "search_by_muscle":
            muscle = params.get("muscle", "").lower()
            return [e for e in EXERCISE_LIBRARY
                    if any(muscle in m.lower() for m in e["muscles"])]
        elif tool_name == "search_by_equipment":
            equip = params.get("equipment", "").lower()
            return [e for e in EXERCISE_LIBRARY if equip in e["equipment"].lower()]
        elif tool_name == "search_by_difficulty":
            diff = params.get("difficulty", "")
            return [e for e in EXERCISE_LIBRARY if e["difficulty"] == diff]
        elif tool_name == "get_exercise_detail":
            name = params.get("name", "")
            for e in EXERCISE_LIBRARY:
                if e["name"] == name:
                    return [e]
            return []
        return []
