"""饮食安全规则层测试。

重点不是"能跑"，而是三类真实失败模式：
① 危机词表误报——把正当求助者（「压力大总暴食想规律饮食」）挡在门外；
② 危机词表漏报/被绕过——零宽字符、否定语境、第三人称；
③ 过敏硬约束被降级或去重膨胀——过敏性休克级后果不允许 warning 待遇。
"""

import pytest

from src.core import diet_safety as ds
from src.core import qa_safety


class TestSingleSource:
    def test_emergency_keywords_are_the_same_object(self):
        """急症词表单一事实源：饮食侧不得复制第二份。"""
        assert ds.EMERGENCY_KEYWORDS is qa_safety.EMERGENCY_KEYWORDS

    def test_public_negation_helpers_share_private_implementation(self):
        assert qa_safety.find_affirmed_keyword("没有胸闷", ("胸闷",)) is False
        assert qa_safety.find_affirmed_keyword("胸闷得厉害", ("胸闷",)) is True
        assert qa_safety.negated_at("没有胸闷", 2) is True


class TestCrisisTwoTiers:
    def test_hard_crisis_extreme_request(self):
        result = ds.detect_diet_crisis("教我催吐，我想一天只吃 500 大卡")
        assert result is not None and result["level"] == "hard"
        assert result["matched"]

    @pytest.mark.parametrize("text", [
        "每天控制在 600 大卡以内",
        "一天只吃一顿饭",
        "推荐减肥药给我",
        "怎么绝食瘦得快",
    ])
    def test_hard_crisis_variants(self, text):
        result = ds.detect_diet_crisis(text)
        assert result is not None and result["level"] == "hard", text

    def test_normal_calorie_target_is_not_crisis(self):
        assert ds.detect_diet_crisis("每天控制在 2000 大卡左右") is None
        assert ds.detect_diet_crisis("我想减脂，每天吃 1500 大卡") is None

    def test_soft_signal_help_seeker_is_not_rejected(self):
        """最该帮的用户：提及状况但无极端请求 → 软档，计划照常交付。"""
        result = ds.detect_diet_crisis("最近压力大总是暴食，想规律饮食")
        assert result is not None and result["level"] == "soft"

    def test_status_without_request_is_soft_not_hard(self):
        result = ds.detect_diet_crisis("我有时吃完会抠吐，很痛苦")
        assert result is not None and result["level"] == "soft"

    def test_negation_does_not_trigger(self):
        assert ds.detect_diet_crisis("我没有催吐想法") is None
        assert ds.detect_diet_crisis("没有暴食倾向，饮食规律") is None

    def test_negation_overturned_by_turning_word(self):
        result = ds.detect_diet_crisis("没有暴食，但最近又开始催吐了")
        assert result is not None  # 转折推翻否定

    def test_third_person_request_is_not_hard(self):
        result = ds.detect_diet_crisis("朋友问我怎么催吐，我该怎么劝他")
        assert result is None or result["level"] != "hard"

    def test_zero_width_bypass_fails(self):
        result = ds.detect_diet_crisis("教我催\u200b吐")
        assert result is not None and result["level"] == "hard"

    def test_cjk_gap_bypass_fails(self):
        result = ds.detect_diet_crisis("怎么绝 食")
        assert result is not None and result["level"] == "hard"

    def test_pinyin_paren_bypass_fails(self):
        result = ds.detect_diet_crisis("教我催(tu)吐")
        assert result is not None and result["level"] == "hard"

    def test_trend_words_are_warnings_not_crisis(self):
        assert ds.detect_diet_crisis("我想试试轻断食减肥") is None
        assert ds.detect_diet_trends("我想试试轻断食减肥") == ["轻断食"]
        assert ds.detect_diet_crisis("生酮饮食期间能增肌吗") is None

    def test_empty_text_is_none(self):
        assert ds.detect_diet_crisis("") is None
        assert ds.detect_diet_crisis(None) is None


class TestSignalScan:
    def test_emergency_detected_with_negation_awareness(self):
        signals = ds.scan_diet_signals(query="训练时胸闷头晕")
        assert signals["emergency"]
        assert ds.scan_diet_signals(query="没有胸闷")["emergency"] == []

    def test_emergency_and_crisis_both_reported_priority_left_to_caller(self):
        signals = ds.scan_diet_signals(query="胸闷，还想教我催吐")
        assert signals["emergency"] and signals["crisis"]["level"] == "hard"

    def test_injuries_text_also_scanned(self):
        signals = ds.scan_diet_signals(injuries=["晕厥史"])
        assert signals["emergency"]


class TestAllergenAndDiseaseResolution:
    def test_free_text_maps_to_closed_tags(self):
        assert ds.resolve_allergens(["花生过敏", "对牛奶过敏"]) == ["花生", "乳"]
        assert ds.resolve_allergens(["乳糖不耐受"]) == ["乳"]

    def test_direct_tags_pass_through(self):
        assert ds.resolve_allergens(["麸质", "花生"]) == ["麸质", "花生"]

    def test_disease_synonyms_normalize(self):
        assert ds.resolve_diseases(["血糖高", "尿酸高", "怀孕"]) == ["糖尿病", "痛风", "孕期"]

    def test_unknown_allergen_text_yields_nothing(self):
        assert ds.resolve_allergens(["不知道过敏什么"]) == []


class TestFoodConflicts:
    PREFS = {"allergies": ["花生过敏"],
             "diseases": ["糖尿病", "乳糖不耐"],
             "avoid_foods": ["带鱼"]}

    def test_allergen_disease_and_avoid_all_locatable(self):
        foods = [{"name": "花生（生）", "grams": 15, "meal": "加餐", "day_type": "训练日"},
                 {"name": "牛奶（全脂）", "grams": 250, "meal": "早餐"},
                 {"name": "米饭（蒸）", "grams": 150, "meal": "午餐"},
                 {"name": "带鱼", "grams": 120, "meal": "晚餐"}]
        conflicts = ds.check_food_conflicts(foods, self.PREFS)
        kinds = {(c["food"], c["kind"]) for c in conflicts}
        assert ("花生（生）", "allergen") in kinds
        assert ("牛奶（全脂）", "disease") in kinds      # 乳糖不耐 → 乳
        assert ("米饭（蒸）", "disease") in kinds        # 糖尿病 → high_gi
        assert ("带鱼", "avoid") in kinds
        located = [c for c in conflicts if c["food"] == "花生（生）"][0]
        assert located["meal"] == "加餐" and located["day_type"] == "训练日"

    def test_conflicts_deduped_by_food_kind_reason(self):
        foods = [{"name": "牛奶（全脂）", "grams": 250},
                 {"name": "牛奶（全脂）", "grams": 250}]
        conflicts = ds.check_food_conflicts(foods, {"diseases": ["乳糖不耐"]})
        assert len(conflicts) == 1

    def test_unknown_food_is_flagged_for_substitution(self):
        conflicts = ds.check_food_conflicts([{"name": "炸鸡架", "grams": 100}], {})
        assert conflicts[0]["kind"] == "unknown"

    def test_clean_plan_has_no_conflicts(self):
        foods = [{"name": "鸡胸肉（煮）", "grams": 120},
                 {"name": "糙米饭（蒸）", "grams": 150},
                 {"name": "西兰花", "grams": 150}]
        assert ds.check_food_conflicts(foods, self.PREFS) == []

    def test_pregnancy_excludes_liver_and_high_mercury(self):
        foods = [{"name": "猪肝（鲜）", "grams": 75},
                 {"name": "金枪鱼（水浸罐头）", "grams": 100}]
        conflicts = ds.check_food_conflicts(foods, {"diseases": ["怀孕"]})
        assert {c["food"] for c in conflicts} == {"猪肝（鲜）", "金枪鱼（水浸罐头）"}
        assert all(c["kind"] == "disease" for c in conflicts)


class TestDiseaseTargetRules:
    def test_kidney_disease_caps_protein_before_meal_allocation(self):
        targets = {"daily_kcal": 2000, "protein_g": 160, "tdee": 2200,
                   "delta_pct": 0, "basis": "x"}
        result = ds.apply_disease_target_rules(targets, ["肾功能不全"], weight_kg=80)
        assert result["targets"]["protein_g"] == 64  # 80 × 0.8
        assert result["applied"] == ["肾病"]
        assert result["warnings"]
        assert targets["protein_g"] == 160, "不得修改入参"

    def test_kidney_cap_does_not_raise_protein(self):
        targets = {"daily_kcal": 2000, "protein_g": 50, "tdee": 2200, "delta_pct": 0}
        result = ds.apply_disease_target_rules(targets, ["肾病"], weight_kg=80)
        assert result["targets"]["protein_g"] == 50

    def test_pregnancy_blocks_deficit(self):
        targets = {"daily_kcal": 1500, "protein_g": 120, "tdee": 2000,
                   "delta_pct": -25, "basis": "x", "goal_note": "减脂"}
        result = ds.apply_disease_target_rules(targets, ["孕期"], weight_kg=70)
        assert result["targets"]["daily_kcal"] == 2000
        assert result["targets"]["delta_pct"] == 0
        assert "疾病护栏" in result["targets"]["basis"]

    def test_no_disease_leaves_targets_untouched(self):
        targets = {"daily_kcal": 2000, "protein_g": 144, "tdee": 2200, "delta_pct": 0}
        result = ds.apply_disease_target_rules(targets, [], weight_kg=80)
        assert result["targets"] == targets and result["warnings"] == []