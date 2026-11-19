"""DietWriter 测试：假 LLM、不联网。

核心断言三类：① 模型报来的营养数字一律被服务端重算覆盖；② 解析失败/降级/
结构不合格 fail-closed 抛错；③ 配餐前禁忌食物已从清单滤掉（模型看不到就不会选）。
"""

import pytest

from src.agents import diet_writer as dw
from src.agents.output_validation import OutputValidationError
from src.config import REWRITE_MODEL
from src.rag import food_catalog


class _FakeLLM:
    """固定返回一份配餐结果，记录 messages 与 model，不联网。"""

    def __init__(self, payload=None):
        self.payload = payload if payload is not None else _valid_plan()
        self.calls = []

    def chat_with_json_mode(self, messages, model=None):
        self.calls.append({"messages": messages, "model": model})
        return dict(self.payload) if isinstance(self.payload, dict) else self.payload


def _valid_plan():
    return {
        "plan_name": "增肌饮食计划",
        "goal": "增肌",
        "days": [
            {"day_type": "训练日", "meals": [
                {"meal": "早餐", "foods": [{"name": "燕麦粥（煮）", "grams": 250},
                                           {"name": "鸡蛋（煮）", "grams": 100}]},
                {"meal": "午餐", "foods": [{"name": "米饭（蒸）", "grams": 200},
                                           {"name": "鸡胸肉（煮）", "grams": 150}]},
                {"meal": "晚餐", "foods": [{"name": "糙米饭（蒸）", "grams": 150},
                                           {"name": "西兰花", "grams": 200}]},
                {"meal": "加餐", "foods": [{"name": "牛奶（全脂）", "grams": 250}]},
            ]},
            {"day_type": "休息日", "meals": [
                {"meal": "早餐", "foods": [{"name": "燕麦粥（煮）", "grams": 200}]},
                {"meal": "午餐", "foods": [{"name": "米饭（蒸）", "grams": 150},
                                           {"name": "豆腐（北）", "grams": 150}]},
                {"meal": "晚餐", "foods": [{"name": "红薯（蒸）", "grams": 150},
                                           {"name": "菠菜", "grams": 200}]},
                {"meal": "加餐", "foods": [{"name": "苹果", "grams": 200}]},
            ]},
        ],
        "notes": "80% 按计划吃，20% 留给社交外食。",
        "shopping_list": ["鸡胸肉", "燕麦", "西兰花"],
    }


def _targets():
    return {"training_day_kcal": 2800, "rest_day_kcal": 2400, "protein_g": 144}


def _slots():
    from src.core.nutrition import allocate_meals
    return {"训练日": allocate_meals(2800, 144), "休息日": allocate_meals(2400, 144)}


class TestSelectFoods:
    def test_returns_validated_plan_and_prompt_carries_constraints(self):
        llm = _FakeLLM()
        catalog = food_catalog.load_foods()
        plan = dw.select_foods(
            llm, profile={"goal": "增肌", "weight": 80, "days_per_week": 4},
            day_targets=_targets(), meal_slots=_slots(), catalog=list(catalog),
            prefs={"allergen_tags": ["花生"], "diseases": [], "avoid_foods": []},
            forbidden_names=["花生（生）"],
        )
        assert plan["plan_name"] == "增肌饮食计划"
        user_msg = llm.calls[0]["messages"][1]["content"]
        assert "花生（生）" in user_msg and "禁忌清单" in user_msg
        assert "米饭（蒸）" in user_msg
        system_msg = llm.calls[0]["messages"][0]["content"]
        assert "禁止输出热量" in system_msg

    @pytest.mark.parametrize("payload", [
        {"raw": "散文不是计划"},
        {"_degraded": True, "plan_name": "x", "goal": "增肌", "days": []},
        {"plan_name": "x", "goal": "增肌", "days": []},
        "not a dict",
    ])
    def test_fail_closed_on_unparseable_or_invalid(self, payload):
        with pytest.raises(OutputValidationError):
            dw.select_foods(_FakeLLM(payload), profile={"goal": "增肌"},
                            day_targets=_targets(), meal_slots=_slots(),
                            catalog=[], prefs={})

    def test_single_day_type_rejected(self):
        plan = _valid_plan()
        plan["days"] = [plan["days"][0], plan["days"][0]]
        with pytest.raises(OutputValidationError):
            dw.select_foods(_FakeLLM(plan), profile={"goal": "增肌"},
                            day_targets=_targets(), meal_slots=_slots(),
                            catalog=[], prefs={})

    def test_duplicate_meal_rejected(self):
        plan = _valid_plan()
        plan["days"][0]["meals"][1]["meal"] = "早餐"
        with pytest.raises(OutputValidationError):
            dw.select_foods(_FakeLLM(plan), profile={"goal": "增肌"},
                            day_targets=_targets(), meal_slots=_slots(),
                            catalog=[], prefs={})

    def test_zero_grams_rejected(self):
        plan = _valid_plan()
        plan["days"][0]["meals"][0]["foods"][0]["grams"] = 0
        with pytest.raises(OutputValidationError):
            dw.select_foods(_FakeLLM(plan), profile={"goal": "增肌"},
                            day_targets=_targets(), meal_slots=_slots(),
                            catalog=[], prefs={})


class TestRecompute:
    def test_model_numbers_are_stripped_and_recomputed(self):
        plan = _valid_plan()
        plan["days"][0]["meals"][0]["foods"][0]["kcal"] = 99999
        plan["days"][0]["meals"][0]["foods"][0]["protein_g"] = 99999
        plan["days"][0]["meals"][0]["kcal"] = 1
        out = dw.recompute_nutrition(plan)
        food = out["days"][0]["meals"][0]["foods"][0]
        expected = food_catalog.food_nutrients("燕麦粥（煮）", 250)
        assert food["kcal"] == expected["kcal"] != 99999
        assert food["protein_g"] == expected["protein_g"]
        assert set(food) == {"name", "grams", "kcal", "protein_g", "fat_g", "carbs_g"}

    def test_meal_and_day_totals_are_self_consistent(self):
        out = dw.recompute_nutrition(_valid_plan())
        for day in out["days"]:
            assert day["total_kcal"] == round(sum(m["kcal"] for m in day["meals"]))
            for meal in day["meals"]:
                assert meal["kcal"] == round(sum(f["kcal"] for f in meal["foods"]))
                assert meal["protein_g"] == round(sum(f["protein_g"] for f in meal["foods"]), 1)

    def test_unknown_food_flagged_not_silently_accepted(self):
        plan = _valid_plan()
        plan["days"][0]["meals"][0]["foods"].append({"name": "炸鸡架", "grams": 100})
        out = dw.recompute_nutrition(plan)
        unknown = dw.collect_unknown_foods(out)
        assert unknown == [{"food": "炸鸡架", "day_type": "训练日", "meal": "早餐"}]
        flagged = out["days"][0]["meals"][0]["foods"][-1]
        assert flagged["_unknown"] is True and flagged["kcal"] == 0

    def test_input_not_mutated(self):
        plan = _valid_plan()
        snapshot = repr(plan)
        dw.recompute_nutrition(plan)
        assert repr(plan) == snapshot

    def test_alias_name_normalized_to_canonical(self):
        plan = _valid_plan()
        plan["days"][0]["meals"][1]["foods"][0]["name"] = "马铃薯"
        out = dw.recompute_nutrition(plan)
        assert out["days"][0]["meals"][1]["foods"][0]["name"] == "土豆（蒸）"


class TestRewrite:
    def test_rewrite_uses_fast_model_and_names_forbidden(self):
        llm = _FakeLLM()
        conflicts = [{"food": "花生（生）", "reason": "含过敏原「花生」",
                      "meal": "加餐", "day_type": "训练日"}]
        dw.rewrite_with_forbidden(llm, _valid_plan(), conflicts,
                                  catalog=list(food_catalog.load_foods()),
                                  day_targets=_targets(), meal_slots=_slots())
        assert llm.calls[0]["model"] == REWRITE_MODEL
        user_msg = llm.calls[0]["messages"][1]["content"]
        assert "禁用食物清单" in user_msg and "花生（生）" in user_msg


class TestCatalogFilter:
    def test_allergen_and_disease_and_avoid_filtered_before_prompt(self):
        prefs = {"allergies": ["花生过敏"], "diseases": ["糖尿病", "乳糖不耐"],
                 "avoid_foods": ["带鱼"]}
        kept = dw.filter_catalog_for_prefs(list(food_catalog.load_foods()), prefs)
        names = {e["name"] for e in kept}
        assert "花生（生）" not in names and "花生酱" not in names
        assert "牛奶（全脂）" not in names and "希腊酸奶（无糖）" not in names
        assert "米饭（蒸）" not in names and "馒头（蒸）" not in names  # high_gi
        assert "带鱼" not in names
        assert "鸡胸肉（煮）" in names and "糙米饭（蒸）" in names

    def test_target_level_disease_does_not_remove_foods(self):
        """肾病是蛋白总量问题，换菜解决不了——清单不该被滤空。"""
        kept = dw.filter_catalog_for_prefs(list(food_catalog.load_foods()),
                                           {"diseases": ["肾病"]})
        assert len(kept) == len(food_catalog.load_foods())

    def test_empty_prefs_keeps_everything(self):
        kept = dw.filter_catalog_for_prefs(list(food_catalog.load_foods()), {})
        assert len(kept) == len(food_catalog.load_foods())