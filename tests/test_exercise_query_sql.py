"""动作库查询的 SQL 构造回归。

补这组用例的动因是一次真实排查：`target_muscles` 是 **json 列**，
而查询里直接写了 `target_muscles ILIKE :muscle`。PostgreSQL 没有
`json ~~* unknown` 这个操作符，于是**整条 SQL 报错 → 静默降级到只有
3 条演示数据的 FALLBACK_EXERCISES**。实测 `search_by_muscle("胸")`
只返回 1 条、"臀" 只匹配到演示数据里的那一条。

更大的连带影响：`query` 分支把 `name ILIKE` 与 `target_muscles ILIKE`
写在同一个 OR 里，所以按名称搜索也一起被打坏。

**这个 bug 能长期存活，是因为现有测试跑不到它**：
`tests/test_advanced.py` 与 `tests/test_mcp_v2.py` 都有 autouse fixture
把 `_query._pg_available` 置为 False，所有 MCP 用例都在 3 条演示数据上跑。
本文件用两种方式补上覆盖：
  1. 离线：桩掉 pg 层、捕获真实执行的 SQL，断言它不含坏模式（默认就跑）
  2. 集成：连真实数据库验证返回量（标 integration，默认跳过）
"""

from __future__ import annotations

import pytest

from src.mcp import exercise_server

# ---------------------------------------------------------------------------
# 第 1 层：SQL 构造（离线，默认执行）
# ---------------------------------------------------------------------------

class _CapturingPG:
    """记录 SQL、返回空结果集的桩，用来观察查询到底发了什么。"""

    def __init__(self):
        self.sqls: list[str] = []

    def fetch_all(self, sql, params=None):
        self.sqls.append(sql)
        return []


@pytest.fixture
def captured_query(monkeypatch):
    """让 _query 走"数据库可用"分支，但把 pg 换成捕获桩。"""
    pg = _CapturingPG()
    q = exercise_server._ExerciseQuery()
    monkeypatch.setattr(q, "_pg", pg, raising=False)
    monkeypatch.setattr(q, "_pg_available", True, raising=False)
    return q, pg


class TestNoIlikeOnJsonColumn:
    """json 列绝不能直接 ILIKE——这是把一个功能整条打坏的写法。"""

    def test_muscle_search_does_not_ilike_json_column(self, captured_query):
        q, pg = captured_query
        q.search(muscle="胸大肌", limit=10)
        assert pg.sqls, "未发出 SQL"
        sql = pg.sqls[0]
        assert "target_muscles ILIKE" not in sql, (
            "又对 json 列用 ILIKE 了——PostgreSQL 无 json ~~* unknown 操作符，"
            "整条查询会报错并静默降级到 3 条演示数据"
        )
        assert "jsonb_array_elements_text" in sql, "应按 json 数组元素展开比较"

    def test_text_search_does_not_ilike_json_column(self, captured_query):
        """query 分支把 name 与 target_muscles 写在同一 OR 里，一起被打坏。"""
        q, pg = captured_query
        q.search(query="卧推", limit=10)
        sql = pg.sqls[0]
        assert "target_muscles ILIKE" not in sql
        assert "jsonb_array_elements_text" in sql

    def test_query_branch_still_matches_names(self, captured_query):
        """修好肌群那半边时，不得把按名称搜的那半边删掉。"""
        q, pg = captured_query
        q.search(query="卧推", limit=10)
        assert "name ILIKE" in pg.sqls[0]

    def test_equipment_search_keeps_plain_ilike(self, captured_query):
        """equipment 是 varchar，ILIKE 本来就合法，不该被误改成 jsonb 展开。"""
        q, pg = captured_query
        q.search(equipment="哑铃", limit=10)
        sql = pg.sqls[0]
        assert "equipment ILIKE" in sql
        assert "jsonb_array_elements_text" not in sql


# ---------------------------------------------------------------------------
# 第 2 层：真实数据库（集成，默认跳过）
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestAgainstRealDatabase:
    """连真实库验证——降级路径的 3 条演示数据无法暴露这类问题。"""

    def test_muscle_search_returns_more_than_fallback_size(self):
        from src.mcp.exercise_server import FALLBACK_EXERCISES, search_by_muscle

        results = search_by_muscle("臀", limit=30)
        assert len(results) > len(FALLBACK_EXERCISES), (
            f"只返回了 {len(results)} 条——疑似又降级到 {len(FALLBACK_EXERCISES)} 条演示数据"
        )

    def test_muscle_search_results_actually_carry_that_muscle(self):
        from src.mcp.exercise_server import search_by_muscle

        results = search_by_muscle("股四头肌", limit=10)
        assert results, "按肌群检索返回空"
        for item in results:
            muscles = item.get("target_muscles") or []
            assert any("股四头肌" in str(m) for m in muscles), (
                f"{item.get('name')} 的目标肌群不含所查部位: {muscles}"
            )

    def test_query_search_matches_names(self):
        from src.mcp.exercise_server import search_exercises

        results = search_exercises("卧推", limit=10)
        assert results, "按名称搜索返回空"
        assert any("卧推" in (item.get("name") or "") for item in results)
