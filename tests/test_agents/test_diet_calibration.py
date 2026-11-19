"""确定性克数校准：模型选食物，数字由服务端负责到底。

真实端到端暴露的缺陷：LLM 选的克数蛋白基本达标、热量却短 33%，
闸门只查安全不查「选的量够不够目标」，于是增肌者拿到一份缺口餐单。
校准规则（纯确定性，不引入新的 LLM 轮次）：
* 蛋白先调：偏差 >5g 时增减蛋白密度最高的食物；
* 热量后补：缺口超容差（max(50, 8%)）时先加低蛋白密度碳水（≤600g/食物），
  仍不够才动脂肪（每食物增量 ≤15g，避免一杯油）；过量则反向减碳水；
* 食物选择不变——只动克数；容差内的餐次原样保留。
"""
from __future__ import annotations

from src.agents import diet_writer as dw
from src.core.nutrition import allocate_meals
from src.rag import food_catalog


def _meal(meal, foods):
    return {"meal": meal, "foods": [{"name": n, "grams": g} for n, g in foods]}


def _plan(meals, day_type="训练日"):
    return {"plan_name": "x", "goal": "增肌",
            "days": [{"day_type": day_type, "meals": meals}]}


def _slots(kcal, protein, day_type="训练日", meal_count=4):
    return {day_type: allocate_meals(kcal, protein, meal_count=meal_count)}


def _kcal(meal):
    return sum(food_catalog.food_nutrients(f["name"], f["grams"])["kcal"]
               for f in meal["foods"] if food_catalog.get_food(f["name"]))


def _protein(meal):
    return sum(food_catalog.food_nutrients(f["name"], f["grams"])["protein_g"]
               for f in meal["foods"] if food_catalog.get_food(f["name"]))


def _slots_for(plan):
    # 测试里直接按餐次给目标，不走 allocate（隔离被测函数）
    return {"训练日": [{"meal": m["meal"], "kcal_target": m["_kcal_target"],
                        "protein_target_g": m["_protein_target"]}
                       for m in plan["days"][0]["meals"]]}


def _with_targets(meals):
    """meals: (meal_dict, kcal_target, protein_target) 列表。"""
    plan = _plan([m for m, _, _ in meals])
    for m, kcal, protein in meals:
        m["_kcal_target"] = kcal
        m["_protein_target"] = protein
    return plan


class TestCalorieTopUp:
    def test_undershoot_topped_up_with_carbs(self):
        meal = _meal("午餐", [("米饭（蒸）", 150), ("鸡胸肉（煮）", 100)])
        plan = _with_targets([(meal, 600, 35)])
        out, adjusted = dw.calibrate_portions(plan, _slots_for(plan))
        m = out["days"][0]["meals"][0]
        assert adjusted
        assert abs(_kcal(m) - 600) <= max(50, 600 * 0.08)
        assert abs(_protein(m) - 35) <= 6
        names = [f["name"] for f in m["foods"]]
        assert names == ["米饭（蒸）", "鸡胸肉（煮）"], "食物选择不得改变"
        rice = next(f for f in m["foods"] if f["name"] == "米饭（蒸）")
        assert 150 < rice["grams"] <= 400

    def test_staple_capped_at_400(self):
        meal = _meal("午餐", [("米饭（蒸）", 150), ("鸡胸肉（煮）", 100)])
        plan = _with_targets([(meal, 2000, 35)])
        out, _ = dw.calibrate_portions(plan, _slots_for(plan))
        for f in out["days"][0]["meals"][0]["foods"]:
            assert f["grams"] <= 400

    def test_gap_distributed_across_carbs(self):
        """缺口按现有克数分摊到多种碳水，而不是全堆到第一种（600g 米饭教训）。"""
        meal = _meal("午餐", [("米饭（蒸）", 100), ("全麦面包", 100), ("鸡胸肉（煮）", 100)])
        plan = _with_targets([(meal, 1000, 35)])
        out, _ = dw.calibrate_portions(plan, _slots_for(plan))
        foods = {f["name"]: f["grams"] for f in out["days"][0]["meals"][0]["foods"]}
        assert foods["米饭（蒸）"] > 100
        assert foods["全麦面包"] > 100, "第二种碳水也应分担缺口"
        assert foods["米饭（蒸）"] < 400, "不应一种食物独吞到上限"

    def test_fat_fallback_capped(self):
        meal = _meal("午餐", [("米饭（蒸）", 600), ("鸡胸肉（煮）", 100), ("橄榄油", 10)])
        plan = _with_targets([(meal, 1500, 35)])
        out, _ = dw.calibrate_portions(plan, _slots_for(plan))
        foods = {f["name"]: f["grams"] for f in out["days"][0]["meals"][0]["foods"]}
        assert foods["米饭（蒸）"] <= 400
        assert foods["橄榄油"] <= 15, "纯油一餐 ≤15g，不能倒一杯油"

    def test_overshoot_reduces_carbs(self):
        meal = _meal("午餐", [("米饭（蒸）", 500), ("鸡胸肉（煮）", 100)])
        plan = _with_targets([(meal, 400, 35)])
        out, _ = dw.calibrate_portions(plan, _slots_for(plan))
        m = out["days"][0]["meals"][0]
        assert abs(_kcal(m) - 400) <= max(50, 400 * 0.08)
        rice = next(f for f in m["foods"] if f["name"] == "米饭（蒸）")
        assert rice["grams"] < 500
        assert rice["grams"] >= 20

    def test_within_tolerance_untouched(self):
        rice = food_catalog.get_food("米饭（蒸）")
        chicken = food_catalog.get_food("鸡胸肉（煮）")
        target = round(rice["kcal_per_100g"] * 2 + chicken["kcal_per_100g"])
        protein = round(rice["protein_g"] * 2 + chicken["protein_g"], 1)
        meal = _meal("午餐", [("米饭（蒸）", 200), ("鸡胸肉（煮）", 100)])
        plan = _with_targets([(meal, target, protein)])
        out, adjusted = dw.calibrate_portions(plan, _slots_for(plan))
        assert not adjusted
        assert [f["grams"] for f in out["days"][0]["meals"][0]["foods"]] == [200, 100]


class TestAbsurdPortionsClamped:
    """模型自己选出的离谱份量也要压回合理上限（实测杏仁 115g、香蕉 580g）。"""

    def test_nuts_clamped_to_30(self):
        meal = _meal("加餐", [("杏仁", 100), ("米饭（蒸）", 150)])
        plan = _with_targets([(meal, 500, 20)])
        out, _ = dw.calibrate_portions(plan, _slots_for(plan))
        foods = {f["name"]: f["grams"] for f in out["days"][0]["meals"][0]["foods"]}
        assert foods["杏仁"] == 30

    def test_pure_oil_clamped_to_15(self):
        meal = _meal("午餐", [("橄榄油", 50), ("米饭（蒸）", 200)])
        plan = _with_targets([(meal, 400, 10)])
        out, _ = dw.calibrate_portions(plan, _slots_for(plan))
        foods = {f["name"]: f["grams"] for f in out["days"][0]["meals"][0]["foods"]}
        assert foods["橄榄油"] == 15

    def test_protein_powder_clamped_to_60(self):
        meal = _meal("加餐", [("乳清蛋白粉", 200), ("米饭（蒸）", 150)])
        plan = _with_targets([(meal, 600, 50)])
        out, _ = dw.calibrate_portions(plan, _slots_for(plan))
        foods = {f["name"]: f["grams"] for f in out["days"][0]["meals"][0]["foods"]}
        assert foods["乳清蛋白粉"] == 60

    def test_fruit_clamped_to_250(self):
        meal = _meal("早餐", [("香蕉", 580), ("鸡蛋（煮）", 100)])
        plan = _with_targets([(meal, 600, 30)])
        out, _ = dw.calibrate_portions(plan, _slots_for(plan))
        foods = {f["name"]: f["grams"] for f in out["days"][0]["meals"][0]["foods"]}
        assert foods["香蕉"] <= 250


class TestProteinAdjustment:
    def test_protein_short_increases_protein_food(self):
        meal = _meal("午餐", [("米饭（蒸）", 200), ("鸡胸肉（煮）", 100)])
        plan = _with_targets([(meal, 700, 55)])
        out, _ = dw.calibrate_portions(plan, _slots_for(plan))
        m = out["days"][0]["meals"][0]
        assert abs(_protein(m) - 55) <= 6
        chicken = next(f for f in m["foods"] if f["name"] == "鸡胸肉（煮）")
        assert chicken["grams"] > 100

    def test_protein_over_reduces_protein_food(self):
        meal = _meal("午餐", [("米饭（蒸）", 200), ("鸡胸肉（煮）", 300)])
        plan = _with_targets([(meal, 600, 35)])
        out, _ = dw.calibrate_portions(plan, _slots_for(plan))
        m = out["days"][0]["meals"][0]
        assert abs(_protein(m) - 35) <= 6
        chicken = next(f for f in m["foods"] if f["name"] == "鸡胸肉（煮）")
        assert chicken["grams"] < 300
        assert chicken["grams"] >= 10


class TestStructure:
    def test_input_not_mutated(self):
        meal = _meal("午餐", [("米饭（蒸）", 150), ("鸡胸肉（煮）", 100)])
        plan = _with_targets([(meal, 700, 35)])
        dw.calibrate_portions(plan, _slots_for(plan))
        assert plan["days"][0]["meals"][0]["foods"][0]["grams"] == 150

    def test_three_meal_slots(self):
        meals = [_meal("早餐", [("燕麦粥（煮）", 250)]),
                 _meal("午餐", [("米饭（蒸）", 150), ("鸡胸肉（煮）", 100)]),
                 _meal("晚餐", [("红薯（蒸）", 200), ("三文鱼", 100)])]
        plan = _plan(meals)
        slots = {"训练日": allocate_meals(2400, 140, meal_count=3)}
        out, _ = dw.calibrate_portions(plan, slots)
        assert len(out["days"][0]["meals"]) == 3
        for m in out["days"][0]["meals"]:
            assert all(10 <= f["grams"] <= 600 for f in m["foods"])

    def test_unknown_food_left_alone(self):
        meal = _meal("午餐", [("米饭（蒸）", 150), ("库外神秘食物", 100)])
        plan = _with_targets([(meal, 700, 35)])
        out, _ = dw.calibrate_portions(plan, _slots_for(plan))
        foods = {f["name"]: f["grams"] for f in out["days"][0]["meals"][0]["foods"]}
        assert foods["库外神秘食物"] == 100

    def test_top_up_adds_staple_when_existing_capped(self):
        """现有食物都到合理上限仍凑不够热量时，从（禁忌已滤除的）食物库补主食。"""
        meal = _meal("午餐", [("米饭（蒸）", 400), ("鸡胸肉（煮）", 250)])
        plan = _with_targets([(meal, 1200, 40)])
        catalog = [food_catalog.get_food("全麦面包"), food_catalog.get_food("红薯（蒸）")]
        out, adjusted = dw.calibrate_portions(plan, _slots_for(plan), top_up_catalog=catalog)
        m = out["days"][0]["meals"][0]
        added = [f["name"] for f in m["foods"] if f["name"] not in ("米饭（蒸）", "鸡胸肉（煮）")]
        assert added, "应补充主食行"
        assert set(added) <= {e["name"] for e in catalog}, "只能从给定（已滤禁忌）清单补"
        assert abs(_kcal(m) - 1200) <= max(50, 1200 * 0.08)
        assert adjusted

    def test_no_top_up_without_catalog(self):
        meal = _meal("午餐", [("米饭（蒸）", 400), ("鸡胸肉（煮）", 250)])
        plan = _with_targets([(meal, 1200, 40)])
        out, _ = dw.calibrate_portions(plan, _slots_for(plan))
        names = [f["name"] for f in out["days"][0]["meals"][0]["foods"]]
        assert names == ["米饭（蒸）", "鸡胸肉（煮）"], "不传清单则只调克数不加食物"