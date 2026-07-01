import pytest
from src.graphrag.search import GraphSearch
from src.graphrag.builder import GraphBuilder


@pytest.fixture(scope="module")
def gs():
    builder = GraphBuilder()
    # Seed minimal test data
    exercises = [
        {
            "name": "哑铃卧推",
            "exercise_type": "复合",
            "difficulty": "初级",
            "equipment": "哑铃",
            "target_muscles": ["胸大肌", "三角肌前束"],
        },
        {
            "name": "杠铃深蹲",
            "exercise_type": "复合",
            "difficulty": "中级",
            "equipment": "杠铃",
            "target_muscles": ["股四头肌", "臀大肌"],
        },
    ]
    builder.build_from_seed(exercises)
    return GraphSearch()


class TestGraphSearch:
    def test_find_exercises_by_muscle(self, gs):
        results = gs.find_exercises_by_muscle("胸")
        assert len(results) > 0
        assert any("卧推" in r["name"] for r in results)

    def test_multi_hop(self, gs):
        results = gs.multi_hop_search("哑铃", "胸")
        assert len(results) > 0

    def test_equipment_lookup(self, gs):
        results = gs.find_equipment_for_exercise("哑铃卧推")
        assert any(r["equipment"] == "哑铃" for r in results)
