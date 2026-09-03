"""
===========================================================================
exercise_server.py — MCP Exercise Server v2 (数据库接入)
===========================================================================
v2 更新 (2026-07-28):
  - 工具从硬编码动作切换到 PG 数据库查询（当前语料 338 动作）
  - 保留 EXERCISE_LIBRARY 仅作 PG 不可用时的降级演示数据（3 个动作）
  - 新增 _db_search() 统一查询入口
v2.1 更新 (2026-09-03):
  - 旧版 exercise://library、exercise://muscles、exercise://standards/*
    资源也切换为数据库优先，与工具层口径一致
===========================================================================

FastMCP server exposing exercise library as Tools + Resources.

Run standalone:  python run_mcp_server.py
"""

import json
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from src.storage.pg import PGClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fallback 演示数据 — 仅用于 PG 不可用时的降级
# ---------------------------------------------------------------------------
FALLBACK_EXERCISES = [
    {
        "name": "哑铃卧推",
        "equipment": "哑铃",
        "target_muscles": ["胸大肌", "三角肌前束", "肱三头肌"],
        "difficulty": "初级",
        "type": "复合",
        "description": "仰卧平板凳，双手持哑铃，推举至手臂伸直，顶峰收缩，缓慢下放。",
        "common_mistakes": ["手肘过度外展", "耸肩代偿"],
    },
    {
        "name": "杠铃深蹲",
        "equipment": "杠铃",
        "target_muscles": ["股四头肌", "臀大肌", "腘绳肌"],
        "difficulty": "中级",
        "type": "复合",
        "description": "杠铃置斜方肌上方，双脚与肩同宽，下蹲至大腿平行地面。",
        "common_mistakes": ["膝盖内扣", "背部拱起"],
    },
    {
        "name": "引体向上",
        "equipment": "自重",
        "target_muscles": ["背阔肌", "肱二头肌"],
        "difficulty": "中级",
        "type": "复合",
        "description": "正握单杠，比肩稍宽，下拉至下巴过杠，缓慢下放。",
        "common_mistakes": ["摆动借力", "只用手臂拉"],
    },
]

# Kept as a public alias for callers that enumerate offline resource metadata.
EXERCISE_LIBRARY = FALLBACK_EXERCISES

EXERCISE_TOOL_SCHEMAS = [
    {
        "name": "search_by_muscle",
        "description": "按目标肌群检索训练动作，如'胸大肌'、'臀大肌'。",
        "inputSchema": {"type": "object", "properties": {
            "muscle": {"type": "string", "description": "目标肌群名称"},
            "limit": {"type": "integer", "default": 10},
        }, "required": ["muscle"]},
    },
    {
        "name": "search_by_equipment",
        "description": "按器械检索训练动作，如'哑铃'、'杠铃'、'自重'、'壶铃'。",
        "inputSchema": {"type": "object", "properties": {
            "equipment": {"type": "string", "description": "器械名称"},
            "limit": {"type": "integer", "default": 10},
        }, "required": ["equipment"]},
    },
    {
        "name": "search_by_difficulty",
        "description": "按难度检索训练动作，如'初级'、'中级'、'高级'。",
        "inputSchema": {"type": "object", "properties": {
            "difficulty": {"type": "string", "description": "难度等级"},
            "limit": {"type": "integer", "default": 10},
        }, "required": ["difficulty"]},
    },
    {
        "name": "get_exercise_detail",
        "description": "按名称获取单个动作的详细信息。",
        "inputSchema": {"type": "object", "properties": {
            "name": {"type": "string", "description": "动作名称"},
        }, "required": ["name"]},
    },
]

# ---------------------------------------------------------------------------
# 数据库查询层
# ---------------------------------------------------------------------------

class _ExerciseQuery:
    """统一的 exercise 查询接口——优先查 PG，不可用时降级到 fallback。"""

    def __init__(self):
        self._pg = None
        self._pg_available = None

    @property
    def pg(self) -> PGClient:
        if self._pg is None:
            try:
                self._pg = PGClient()
                # 验证连接：尝试查询
                self._pg.fetch_all("SELECT 1")
                self._pg_available = True
            except Exception as e:
                logger.warning(f"PG unavailable, using fallback data: {e}")
                self._pg_available = False
        return self._pg

    @property
    def use_db(self) -> bool:
        if self._pg_available is None:
            _ = self.pg  # trigger connection check
        return self._pg_available

    def search(self, query: str = None, muscle: str = None,
               equipment: str = None, difficulty: str = None,
               limit: int = 20) -> list[dict]:
        """从 PG 检索匹配动作。不可用时降级到 fallback。"""
        if not self.use_db:
            return self._fallback_search(query, muscle, equipment, difficulty, limit)

        conditions = ["1=1"]
        params = {"limit": limit}

        if query:
            conditions.append("(name ILIKE :q1 OR target_muscles ILIKE :q2)")
            params["q1"] = f"%{query}%"
            params["q2"] = f"%{query}%"
        if muscle:
            conditions.append("target_muscles ILIKE :muscle")
            params["muscle"] = f"%{muscle}%"
        if equipment:
            conditions.append("equipment ILIKE :equipment")
            params["equipment"] = f"%{equipment}%"
        if difficulty:
            conditions.append("difficulty = :difficulty")
            params["difficulty"] = difficulty

        where = " AND ".join(conditions)
        sql = f"""
            SELECT name, equipment, target_muscles, difficulty,
                   exercise_type AS type, description, common_errors
            FROM exercises
            WHERE {where}
            ORDER BY name
            LIMIT :limit
        """
        try:
            rows = self.pg.fetch_all(sql, params)
            return [
                {
                    "name": r[0],
                    "equipment": r[1],
                    "target_muscles": r[2].split(",") if isinstance(r[2], str) else r[2] or [],
                    "difficulty": r[3],
                    "type": r[4],
                    "description": r[5],
                    "common_mistakes": r[6].split(",") if isinstance(r[6], str) else r[6] or [],
                }
                for r in rows
            ]
        except Exception as e:
            logger.warning(f"PG query failed, falling back: {e}")
            return self._fallback_search(query, muscle, equipment, difficulty, limit)

    def get_by_name(self, name: str) -> dict | None:
        """按名称精确查询单个动作。"""
        if not self.use_db:
            for ex in FALLBACK_EXERCISES:
                if name == ex["name"]:
                    return ex
            return None
        try:
            rows = self.pg.fetch_all(
                "SELECT name, equipment, target_muscles, difficulty, exercise_type, description, common_errors "
                "FROM exercises WHERE name = :name",
                {"name": name}
            )
            if rows:
                r = rows[0]
                return {
                    "name": r[0], "equipment": r[1],
                    "target_muscles": r[2].split(",") if isinstance(r[2], str) else r[2] or [],
                    "difficulty": r[3], "type": r[4], "description": r[5],
                    "common_mistakes": r[6].split(",") if isinstance(r[6], str) else r[6] or [],
                }
        except Exception as e:
            logger.warning(f"PG query failed: {e}")
        return None

    def list_all(self, limit: int = 100) -> list[dict]:
        """列出所有动作。"""
        if not self.use_db:
            return FALLBACK_EXERCISES[:limit]
        try:
            rows = self.pg.fetch_all(
                "SELECT name, equipment, target_muscles, difficulty FROM exercises ORDER BY name LIMIT :limit",
                {"limit": limit}
            )
            return [
                {"name": r[0], "equipment": r[1],
                 "target_muscles": r[2].split(",") if isinstance(r[2], str) else r[2] or [],
                 "difficulty": r[3]}
                for r in rows
            ]
        except Exception:
            return FALLBACK_EXERCISES[:limit]

    @staticmethod
    def _fallback_search(query: str = None, muscle: str = None,
                         equipment: str = None, difficulty: str = None,
                         limit: int = 20) -> list[dict]:
        results = []
        for ex in FALLBACK_EXERCISES:
            if query and query not in ex["name"]:
                continue
            if muscle and muscle not in str(ex.get("target_muscles", [])):
                continue
            if equipment and ex.get("equipment") != equipment:
                continue
            if difficulty and ex.get("difficulty") != difficulty:
                continue
            results.append(ex)
        return results[:limit]


# 全局查询实例
_query = _ExerciseQuery()


# ---------------------------------------------------------------------------
# 兼容层 — tool_registry.py 依赖的 McpToolError + ExerciseMCPServer
# ---------------------------------------------------------------------------

@dataclass
class McpToolError(Exception):
    """MCP 工具调用错误（标准 JSON-RPC 错误码）。"""

    code: int
    message: str
    tool_name: str = ""
    details: str = ""

    def __str__(self) -> str:
        return f"[{self.code}] {self.message} — tool={self.tool_name}: {self.details}"

    def to_dict(self) -> dict:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "data": {"tool_name": self.tool_name, "details": self.details},
            }
        }


class ExerciseMCPServer:
    """MCP Exercise Server 兼容包装——v2 底层已切到 PG 查询。

    保留此类是为了 tool_registry.py 的向后兼容。
    tool_registry 依赖 call_tool() / read_resource() 接口。
    """

    def list_tools(self) -> list[dict]:
        """列出四个旧版兼容工具及其完整 JSON Schema。"""
        return EXERCISE_TOOL_SCHEMAS.copy()

    def call_tool(self, tool_name: str, params: dict) -> list[dict]:
        """调用 MCP 工具（兼容旧接口）。"""
        tools = {
            "search_by_muscle": lambda p: _query.search(muscle=p.get("muscle"), limit=p.get("limit", 10)),
            "search_by_equipment": lambda p: _query.search(equipment=p.get("equipment"), limit=p.get("limit", 10)),
            "search_by_difficulty": lambda p: _query.search(difficulty=p.get("difficulty"), limit=p.get("limit", 10)),
            "get_exercise_detail": lambda p: _query.get_by_name(p.get("name", "")),
            "search_exercises": lambda p: _query.search(query=p.get("query"), limit=p.get("limit", 10)),
            "list_all_exercises": lambda p: _query.list_all(limit=p.get("limit", 50)),
        }
        if tool_name not in tools:
            raise McpToolError(
                code=-32601,
                message="Method not found",
                tool_name=tool_name,
                details=f"Unknown tool: {tool_name}",
            )
        try:
            result = tools[tool_name](params)
            if tool_name == "get_exercise_detail" and result is None:
                raise ValueError(f"未找到动作 '{params.get('name', '')}'")
            return result if isinstance(result, list) else [result]
        except McpToolError:
            raise
        except ValueError as exc:
            raise McpToolError(
                code=-32602,
                message="Invalid params",
                tool_name=tool_name,
                details=str(exc),
            ) from exc
        except Exception as exc:
            raise McpToolError(
                code=-32603,
                message="Internal error",
                tool_name=tool_name,
                details=str(exc),
            ) from exc

    def read_resource(self, uri: str) -> str:
        """读取旧版 exercise:// Markdown 资源及新版 exercises:// JSON。

        旧版资源同样走数据库优先：PG 可用时按库内全部动作渲染，
        不可用时降级到 EXERCISE_LIBRARY 演示数据，与工具层口径一致。
        """
        if uri == "exercise://library":
            exercises = _query.list_all(limit=400) or list(EXERCISE_LIBRARY)
            by_muscle: dict[str, list[dict]] = {}
            for exercise in exercises:
                for muscle in exercise["target_muscles"]:
                    by_muscle.setdefault(muscle, []).append(exercise)
            lines = [f"# 动作库索引 (共 {len(exercises)} 个动作)", ""]
            for muscle, exs in sorted(by_muscle.items()):
                lines.append(f"## {muscle}")
                lines.extend(
                    f"  - {ex['name']} ({ex['equipment']}, {ex['difficulty']})"
                    for ex in exs
                )
                lines.append("")
            return "\n".join(lines)
        if uri == "exercise://muscles":
            exercises = _query.list_all(limit=400) or list(EXERCISE_LIBRARY)
            counts = Counter(
                muscle
                for exercise in exercises
                for muscle in exercise["target_muscles"]
            )
            return "\n".join(
                ["# 肌群-动作覆盖", ""]
                + [f"  - {muscle}: {count} 个动作" for muscle, count in counts.most_common()]
            )
        if uri.startswith("exercise://standards/"):
            name = uri.removeprefix("exercise://standards/")
            exercise = _query.get_by_name(name) or next(
                (ex for ex in EXERCISE_LIBRARY if ex["name"] == name), None)
            if exercise is None:
                raise McpToolError(-32601, "Resource not found", "read_resource", uri)
            mistakes = "\n".join(f"  - {item}" for item in exercise["common_mistakes"])
            return (
                f"=== {exercise['name']} ===\n"
                f"类型: {exercise['type']} | 难度: {exercise['difficulty']} | 器械: {exercise['equipment']}\n"
                f"目标肌群: {', '.join(exercise['target_muscles'])}\n\n"
                f"标准做法:\n{exercise['description']}\n\n常见错误:\n{mistakes}"
            )
        if uri == "exercises://all":
            return json.dumps(_query.list_all(limit=200), ensure_ascii=False)
        if uri.startswith("exercises://"):
            name = uri.split("://", 1)[1]
            detail = _query.get_by_name(name)
            return json.dumps(detail, ensure_ascii=False) if detail else "{}"
        raise McpToolError(-32601, "Resource not found", "read_resource", uri)


# Resource-capable compatibility export used by ToolRegistry.
exercise_mcp = ExerciseMCPServer()

# ---------------------------------------------------------------------------
# MCP Tools — 由 run_mcp_server.py 注册
# ---------------------------------------------------------------------------

def search_by_muscle(muscle: str, limit: int = 10) -> list[dict]:
    """按目标肌群检索训练动作，如'胸大肌'、'臀大肌'。"""
    return _query.search(muscle=muscle, limit=limit)


def search_by_equipment(equipment: str, limit: int = 10) -> list[dict]:
    """按器械检索训练动作，如'哑铃'、'杠铃'、'自重'、'壶铃'。"""
    return _query.search(equipment=equipment, limit=limit)


def search_by_difficulty(difficulty: str, limit: int = 10) -> list[dict]:
    """按难度检索训练动作，如'初级'、'中级'、'高级'。"""
    return _query.search(difficulty=difficulty, limit=limit)


def get_exercise_detail(name: str) -> dict | None:
    """按名称获取单个动作的详细信息。"""
    return _query.get_by_name(name)


def list_all_exercises(limit: int = 50) -> list[dict]:
    """列出所有训练动作。"""
    return _query.list_all(limit=limit)


def search_exercises(query: str, limit: int = 10) -> list[dict]:
    """通用搜索：支持名称/肌群/器械的模糊匹配。"""
    return _query.search(query=query, limit=limit)
