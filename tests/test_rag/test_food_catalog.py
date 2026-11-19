"""食物库数据质量与 catalog 行为测试。

头号正确性风险是录入错误（生熟混淆、宏量与能量对不上），所以 Atwater 一致性
逐条锁死：能量应约等于 4P + 9F + 4C - 2×fiber（纤维按 2 kcal/g 发酵产能），
偏差 >10% 即 CI 红。其余锁别名归一、熟态优先、缓存不可变、过敏封闭词表。
"""

import json
from pathlib import Path

import pytest

from src.rag import food_catalog as fc

SEED_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "seed_foods.json"


@pytest.fixture(autouse=True)
def _fresh_cache():
    fc.load_foods.cache_clear()
    fc._name_index.cache_clear()
    fc._alias_map.cache_clear()
    yield
    fc.load_foods.cache_clear()
    fc._name_index.cache_clear()
    fc._alias_map.cache_clear()


def _seed_foods():
    payload = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    return payload["foods"]


class TestSeedDataQuality:
    def test_atwater_consistency_within_ten_percent(self):
        """能量与宏量必须自洽——录入错一个数量级在这里就炸，不靠人眼。"""
        for item in _seed_foods():
            expected = (
                4 * item.get("protein_g", 0)
                + 9 * item.get("fat_g", 0)
                + 4 * item.get("carbs_g", 0)
                - 2 * item.get("fiber_g", 0)
            )
            kcal = item["kcal_per_100g"]
            deviation = abs(kcal - expected) / kcal
            assert deviation <= 0.10, (
                f"{item['name']}: 标 {kcal} kcal 但宏量算出 {expected:.1f} kcal"
                f"（偏差 {deviation:.1%}）——疑似生熟状态或录入错误"
            )

    def test_names_unique_and_state_present(self):
        names = [it["name"] for it in _seed_foods()]
        assert len(names) == len(set(names))
        for item in _seed_foods():
            assert item.get("state") in ("生", "熟", "干"), item["name"]
            assert item.get("default_serving_g", 0) > 0, item["name"]

    def test_allergens_use_closed_vocabulary(self):
        for item in _seed_foods():
            for allergen in item.get("allergens", []):
                assert allergen in fc.FOOD_ALLERGENS, f"{item['name']}: 未知过敏原 {allergen}"

    def test_categories_use_closed_set(self):
        for item in _seed_foods():
            assert item["category"] in fc.FOOD_CATEGORIES, item["name"]

    def test_aliases_never_shadow_another_canonical_name(self):
        names = {it["name"] for it in _seed_foods()}
        for item in _seed_foods():
            for alias in item.get("aliases", []):
                assert alias not in names or alias == item["name"], (
                    f"{item['name']} 的别名 {alias} 遮蔽了另一个规范名"
                )

    def test_raw_and_cooked_rice_differ_by_state(self):
        """生熟差三倍是本项目食物库的立项教训，必须有成对条目锁住口径。"""
        raw = fc.get_food("大米（生）")
        cooked = fc.get_food("米饭（蒸）")
        assert raw and cooked
        assert raw["kcal_per_100g"] > 2.5 * cooked["kcal_per_100g"]

    def test_seed_covers_all_six_categories(self):
        categories = {it["category"] for it in _seed_foods()}
        assert categories == set(fc.FOOD_CATEGORIES)

    def test_seed_size_in_expected_band(self):
        assert 100 <= len(_seed_foods()) <= 200


class TestLookup:
    def test_alias_normalization(self):
        assert fc.normalize_food_name("马铃薯") == "土豆（蒸）"
        assert fc.normalize_food_name("西红柿") == "番茄"
        assert fc.normalize_food_name(" 鸡胸 ") == "鸡胸"

    def test_prefix_match_prefers_cooked_state(self):
        hit = fc.get_food("鸡胸")
        assert hit is not None and hit["name"] == "鸡胸肉（煮）"

    def test_unknown_food_returns_none(self):
        assert fc.get_food("炸鸡架") is None
        assert fc.get_food("") is None
        assert fc.food_nutrients("不存在食物", 100) is None

    def test_get_food_returns_copy_not_cached_object(self):
        first = fc.get_food("米饭（蒸）")
        first["kcal_per_100g"] = 9999
        assert fc.get_food("米饭（蒸）")["kcal_per_100g"] == 116

    def test_load_foods_is_cached_and_immutable_by_contract(self):
        assert fc.load_foods() is fc.load_foods()

    def test_food_nutrients_scales_by_grams(self):
        nut = fc.food_nutrients("米饭（蒸）", 150)
        entry = fc.get_food("米饭（蒸）")
        assert nut["name"] == "米饭（蒸）" and nut["grams"] == 150.0
        assert nut["kcal"] == 174  # 116 × 1.5，整数倍不受浮点舍入影响
        assert nut["protein_g"] == round(entry["protein_g"] * 1.5, 1)
        assert nut["fat_g"] == round(entry["fat_g"] * 1.5, 1)
        assert nut["carbs_g"] == round(entry["carbs_g"] * 1.5, 1)

    def test_food_nutrients_rejects_negative_grams(self):
        assert fc.food_nutrients("米饭（蒸）", -10) is None

    def test_longest_first_name_order(self):
        names = fc.load_food_names()
        assert list(names) == sorted(names, key=len, reverse=True)


class TestSubstitutes:
    def test_excludes_allergen_and_keeps_category(self):
        subs = fc.find_substitutes("花生（生）", exclude_allergens=["花生"])
        assert subs, "脂肪类里应还有不含花生的替代"
        for entry in subs:
            assert entry["category"] == "脂肪"
            assert "花生" not in entry["allergens"]

    def test_excludes_dairy_for_lactose(self):
        subs = fc.find_substitutes("牛奶（全脂）", exclude_allergens=["乳"])
        for entry in subs:
            assert "乳" not in entry["allergens"]

    def test_excludes_high_gi_for_diabetes(self):
        subs = fc.find_substitutes("米饭（蒸）", exclude_flags=["high_gi"])
        assert subs
        for entry in subs:
            assert entry["category"] == "主食"
            assert not entry["high_gi"]

    def test_unknown_source_returns_empty(self):
        assert fc.find_substitutes("炸鸡架") == []

    def test_substitutes_are_copies(self):
        subs = fc.find_substitutes("米饭（蒸）")
        subs[0]["kcal_per_100g"] = 1
        assert fc.get_food(subs[0]["name"])["kcal_per_100g"] != 1


class TestFallback:
    def test_missing_seed_falls_back_to_builtin(self, monkeypatch, tmp_path):
        monkeypatch.setattr(fc, "_SEED_PATH", tmp_path / "absent.json")
        fc.load_foods.cache_clear()
        fc._name_index.cache_clear()
        fc._alias_map.cache_clear()
        foods = fc.load_foods()
        assert foods, "seed 缺失时内置兜底不能为空"
        assert fc.get_food("米饭（蒸）") is not None

    def test_corrupt_seed_falls_back_to_builtin(self, monkeypatch, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        monkeypatch.setattr(fc, "_SEED_PATH", bad)
        fc.load_foods.cache_clear()
        fc._name_index.cache_clear()
        fc._alias_map.cache_clear()
        assert fc.load_foods()