"""Exercise catalog: data-driven loading, longest-first matching, fallbacks."""

import json
from unittest.mock import patch

from src.rag import exercise_catalog


def _clear_cache():
    exercise_catalog.load_exercise_names.cache_clear()


def test_longest_first_matching_from_seed(tmp_path):
    _clear_cache()
    seed = [{"name": "深蹲"}, {"name": "杠铃深蹲"}, {"name": "硬拉"}]
    with patch.object(exercise_catalog, "_names_from_pg", return_value=[]), \
         patch.object(exercise_catalog, "_SEED_PATH", tmp_path / "seed.json"):
        (tmp_path / "seed.json").write_text(json.dumps(seed), encoding="utf-8")
        names = exercise_catalog.load_exercise_names()
        assert names[0] == "杠铃深蹲"  # longer name first
        assert exercise_catalog.extract_exercise_name("做杠铃深蹲膝盖疼") == "杠铃深蹲"
        assert exercise_catalog.extract_exercise_name("硬拉后腰不舒服") == "硬拉"
        assert exercise_catalog.extract_exercise_name("今天吃什么") is None
    _clear_cache()


def test_pg_names_take_precedence(tmp_path):
    _clear_cache()
    with patch.object(exercise_catalog, "_names_from_pg",
                      return_value=["泽奇深蹲", "深蹲"]):
        names = exercise_catalog.load_exercise_names()
        assert "泽奇深蹲" in names  # only reachable via data loading
        assert exercise_catalog.extract_exercise_name("泽奇深蹲怎么做") == "泽奇深蹲"
    _clear_cache()


def test_builtin_fallback_when_all_sources_fail():
    _clear_cache()
    with patch.object(exercise_catalog, "_names_from_pg", return_value=[]), \
         patch.object(exercise_catalog, "_names_from_seed", return_value=[]):
        names = exercise_catalog.load_exercise_names()
        assert "深蹲" in names
    _clear_cache()


def test_deduplication():
    _clear_cache()
    with patch.object(exercise_catalog, "_names_from_pg",
                      return_value=["深蹲", "深蹲", "", "硬拉"]):
        names = exercise_catalog.load_exercise_names()
        assert names.count("深蹲") == 1
        assert "" not in names
    _clear_cache()
