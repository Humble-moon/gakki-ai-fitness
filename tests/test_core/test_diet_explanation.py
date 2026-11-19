"""饮食可解释块测试：四段齐全、缺数据为空不编造、设计口径如实标注。"""

from src.core.diet_explanation import build_diet_explanation


def _targets():
    return {"formula": "Mifflin-St Jeor", "bmr": 1780, "tdee": 2759,
            "daily_kcal": 3173, "protein_g": 144, "fat_g": 72, "carbs_g": 487,
            "goal_note": "增肌期热量盈余 ~15%", "basis": "Mifflin-St Jeor｜80kg / 180cm / 30岁"}


class TestSections:
    def test_four_sections_present_with_titles(self):
        explain = build_diet_explanation(targets=_targets(), days_per_week=4,
                                         catalog_total=124, catalog_kept=100,
                                         result={"rewrite_count": 1, "confidence": 0.9,
                                                 "resolved_issues": [{"issue": "花生冲突已替换"}],
                                                 "active_issues": [{"issue": "脂肪偏低"}]})
        assert set(explain) == {"goal", "energy", "foods", "safety"}
        assert explain["goal"]["title"] == "目标依据"
        assert any("每周 4 练" in line for line in explain["goal"]["lines"])
        assert any("设计口径" in line for line in explain["energy"]["lines"])
        assert any("124" in line and "100" in line for line in explain["foods"]["lines"])
        assert any("已修正 1 处" in line for line in explain["safety"]["lines"])

    def test_missing_data_yields_empty_lines_not_fabrication(self):
        explain = build_diet_explanation(targets={})
        assert explain["goal"]["lines"] == []
        assert explain["safety"]["lines"] == []
        assert explain["energy"]["lines"] == ["餐次热量比例（早/午/晚/加餐）为设计口径；蛋白按日总量均分到各餐"]

    def test_safety_reads_finalize_diff_not_recomputed(self):
        result = {"rewrite_count": 2, "confidence": 0.85,
                  "resolved_issues": [{"issue": "a"}, {"issue": "b"}],
                  "active_issues": []}
        explain = build_diet_explanation(targets=_targets(), result=result)
        safety = explain["safety"]["lines"]
        assert any("重写 2 轮" in line for line in safety)
        assert any("已修正 2 处" in line for line in safety)
        assert not any("保留" in line for line in safety)

    def test_guardrail_and_disease_notes_surface(self):
        explain = build_diet_explanation(
            targets=_targets(), flags={"deficit_blocked": True, "floor_applied": True},
            prefs={"diseases": ["糖尿病"], "allergies": ["花生过敏"]}, days_per_week=3)
        goal_lines = explain["goal"]["lines"]
        assert any("禁缺口人群" in line for line in goal_lines)
        assert any("糖尿病" in line for line in goal_lines)
        assert any("估算护栏" in line for line in explain["energy"]["lines"])

    def test_knowledge_refs_and_substitutions_listed(self):
        explain = build_diet_explanation(
            targets=_targets(), catalog_total=124, catalog_kept=110,
            knowledge_refs=["100-yinshi-jihua.md"], substitutions=["花生（生）→杏仁（干）"])
        foods = explain["foods"]["lines"]
        assert any("100-yinshi-jihua.md" in line for line in foods)
        assert any("花生（生）→杏仁（干）" in line for line in foods)

    def test_no_knowledge_states_fallback_honestly(self):
        explain = build_diet_explanation(targets=_targets(), catalog_total=124,
                                         catalog_kept=124, knowledge_refs=[])
        assert any("未检索到补充知识" in line for line in explain["foods"]["lines"])