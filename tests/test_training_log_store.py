"""逐组明细的清洗与解析：用户输入落库前的边界处理。"""

from src.storage.training_log_store import _clean_sets_detail, _parse_sets_detail


def test_clean_keeps_only_numeric_fields_and_renumbers_sets():
    cleaned = _clean_sets_detail([
        {"set": 9, "weight": "50", "reps": "10", "completed": False, "hack": "x"},
        {"weight": -5, "reps": 999999},
        "bad",
    ])
    assert cleaned[0] == {"set": 1, "weight": 50.0, "reps": 10, "completed": False}
    # 负重量归零，超范围次数截到上限
    assert cleaned[1]["weight"] == 0.0
    assert cleaned[1]["reps"] == 1000


def test_clean_caps_number_of_sets():
    assert len(_clean_sets_detail([{"weight": 1, "reps": 1}] * 200)) == 100


def test_clean_handles_non_list():
    assert _clean_sets_detail(None) == []
    assert _clean_sets_detail({"weight": 1}) == []


def test_parse_accepts_string_or_list():
    assert _parse_sets_detail('[{"weight": 1}]') == [{"weight": 1}]
    assert _parse_sets_detail([{"weight": 1}]) == [{"weight": 1}]
    assert _parse_sets_detail(None) == []
    assert _parse_sets_detail("not json") == []
