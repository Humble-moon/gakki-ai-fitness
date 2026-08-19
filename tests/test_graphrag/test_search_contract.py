import pytest

from src.graphrag.search import GraphSearch


class FakeClient:
    def __init__(self, results=None, error=None):
        self.results = results if results is not None else []
        self.error = error
        self.calls = []

    def query(self, query, params=None):
        self.calls.append((query, params))
        if self.error:
            raise self.error
        return self.results


def graph_search(fake):
    search = GraphSearch.__new__(GraphSearch)
    search.neo4j = fake
    return search


@pytest.mark.parametrize(
    "method,args,field",
    [
        ("find_exercises_by_muscle", ("",), "muscle"),
        ("multi_hop_search", ("", "胸"), "equipment"),
        ("multi_hop_search", ("哑铃", ""), "target"),
    ],
)
def test_empty_search_terms_are_rejected_without_query(method, args, field):
    fake = FakeClient(results=[{"unexpected": "row"}])
    search = graph_search(fake)

    with pytest.raises(ValueError, match=field):
        getattr(search, method)(*args)
    assert fake.calls == []


def test_multi_hop_query_uses_valid_two_hop_cypher_contract():
    fake = FakeClient(results=[])
    search = graph_search(fake)

    assert search.multi_hop_search("哑铃", "胸") == []
    query, params = fake.calls[0]

    assert query.count("MATCH") == 2
    assert "(e:Exercise)-[:REQUIRES]->(eq:Equipment)" in query
    assert "MATCH (e)-[:TARGETS]->(m:Muscle)" in query
    assert "WHERE eq.name CONTAINS $equipment" in query
    assert "WHERE m.name CONTAINS $target" in query
    assert "collect(DISTINCT m.name) AS muscles" in query
    assert "collect(DISTINCT eq.name) AS equipment" in query
    assert params == {"equipment": "哑铃", "target": "胸"}


@pytest.mark.parametrize("limit", [0, -1])
def test_non_positive_limit_is_rejected(limit):
    fake = FakeClient()
    search = graph_search(fake)

    with pytest.raises(ValueError, match="limit"):
        search.find_exercises_by_muscle("胸", limit=limit)
    assert fake.calls == []


def test_reason_about_pain_requires_non_empty_exercise_and_symptom():
    search = graph_search(FakeClient())

    with pytest.raises(ValueError, match="exercise"):
        search.reason_about_pain("", "肩膀疼")
    with pytest.raises(ValueError, match="symptom"):
        search.reason_about_pain("卧推", "")


def test_reason_about_pain_deduplicates_rehab_in_stable_order():
    fake = FakeClient()
    search = graph_search(fake)
    search.find_injury_risks = lambda exercise: [
        {"injury": "肩袖损伤"},
        {"injury": "肩峰撞击"},
    ]
    search.find_rehab_exercises = lambda injury: [
        {"rehab_exercise": "肩外旋", "avoid_exercises": []},
        {"rehab_exercise": "肩胛稳定", "avoid_exercises": []},
    ] if injury == "肩袖损伤" else [
        {"rehab_exercise": "肩胛稳定", "avoid_exercises": []},
        {"rehab_exercise": "肩外旋", "avoid_exercises": []},
    ]

    result = search.reason_about_pain("卧推", "肩膀疼")

    assert result["suggested_rehab"] == ["肩外旋", "肩胛稳定"]


def test_neo4j_errors_are_not_silently_converted_to_empty_results():
    error = RuntimeError("database unavailable")
    search = graph_search(FakeClient(error=error))

    with pytest.raises(RuntimeError, match="database unavailable"):
        search.find_exercises_by_muscle("胸")
