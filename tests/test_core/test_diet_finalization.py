"""饮食终态闸门测试：镜像训练侧三次踩坑点的不变量。

① 只看终轮（历史修好的问题不得继续扣计划）；② 交付与缓存闸门分离
（带建议性 warning 的计划必须能交付）；③ warning 永不阻断；
④ 饮食特有：库外食物残留不得交付；过敏冲突硬阻断走 safety_blocked。
"""


from src.core import diet_finalization as df
from src.core import plan_finalization as pf


def _plan(**overrides):
    plan = {
        "plan_name": "增肌饮食计划", "goal": "增肌",
        "days": [
            {"day_type": "训练日", "meals": [
                {"meal": "早餐", "foods": [{"name": "燕麦粥（煮）", "grams": 250,
                                            "kcal": 170, "protein_g": 6.0,
                                            "fat_g": 3.5, "carbs_g": 30.0}]},
                {"meal": "午餐", "foods": [{"name": "米饭（蒸）", "grams": 200,
                                            "kcal": 232, "protein_g": 5.2,
                                            "fat_g": 0.6, "carbs_g": 51.8}]},
                {"meal": "晚餐", "foods": [{"name": "鸡胸肉（煮）", "grams": 150,
                                            "kcal": 225, "protein_g": 42.0,
                                            "fat_g": 5.3, "carbs_g": 0.0}]},
            ]},
            {"day_type": "休息日", "meals": [
                {"meal": "早餐", "foods": [{"name": "燕麦粥（煮）", "grams": 200,
                                            "kcal": 136, "protein_g": 4.8,
                                            "fat_g": 2.8, "carbs_g": 24.0}]},
                {"meal": "午餐", "foods": [{"name": "米饭（蒸）", "grams": 150,
                                            "kcal": 174, "protein_g": 3.9,
                                            "fat_g": 0.5, "carbs_g": 38.9}]},
                {"meal": "晚餐", "foods": [{"name": "豆腐（北）", "grams": 150,
                                            "kcal": 147, "protein_g": 18.3,
                                            "fat_g": 7.2, "carbs_g": 2.3}]},
            ]},
        ],
    }
    plan.update(overrides)
    return plan


def _check(is_safe=True, issues=None, confidence=0.9, requires_review=False, **extra):
    check = {"is_safe": is_safe, "issues": issues if issues is not None else [],
             "confidence": confidence, "requires_human_review": requires_review}
    check.update(extra)
    return check


class TestGateInvariants:
    def test_clean_plan_delivers_and_persists(self):
        result = df.finalize_diet_result(_plan(), [_check()], 0, expected_goal="增肌")
        assert result["_delivery_allowed"] is True
        assert result["_persistence_allowed"] is True

    def test_warnings_do_not_block_delivery_but_block_cache(self):
        """历史第三次踩坑点：issues 非空当阻断 → 所有计划都交付不出去。"""
        issues = [{"issue": "晚餐脂肪偏低，可加一小把坚果", "severity": "warning"}]
        result = df.finalize_diet_result(_plan(), [_check(issues=issues)], 0,
                                         expected_goal="增肌")
        assert result["_delivery_allowed"] is True
        assert result["_persistence_allowed"] is False
        assert result["warnings"] == ["晚餐脂肪偏低，可加一小把坚果"]

    def test_only_final_round_decides_review(self):
        """历史第一次踩坑点：历史检查污染终判 → 修好的问题永远扣计划。"""
        history = [_check(is_safe=False,
                          issues=[{"issue": "花生与过敏冲突", "severity": "danger"}],
                          requires_review=True)]
        final = _check()
        result = df.finalize_diet_result(_plan(), history + [final], 1,
                                         expected_goal="增肌")
        assert result["_delivery_allowed"] is True
        assert result["requires_review"] is False
        assert result["resolved_issues"] == [{"issue": "花生与过敏冲突", "severity": "danger"}]

    def test_no_checks_means_not_deliverable(self):
        result = df.finalize_diet_result(_plan(), [], 0, expected_goal="增肌")
        assert result["_delivery_allowed"] is False
        assert result["requires_review"] is True

    def test_danger_blocks_delivery(self):
        result = df.finalize_diet_result(
            _plan(), [_check(is_safe=False, requires_review=True,
                             review_conflicts=[{"food": "花生（生）", "reason": "含过敏原「花生」"}],
                             review_reason="过敏冲突", review_severity="danger")],
            0, expected_goal="增肌")
        assert result["_delivery_allowed"] is False
        assert result["_safety_block"]["conflicts"][0]["food"] == "花生（生）"

    def test_goal_mismatch_blocks(self):
        result = df.finalize_diet_result(_plan(goal="减脂"), [_check()], 0,
                                         expected_goal="增肌")
        assert result["_delivery_allowed"] is False
        assert "计划目标与你的目标不一致" in result["_delivery_block_reasons"]

    def test_degraded_blocks(self):
        result = df.finalize_diet_result(_plan(), [_check()], 0,
                                         provider_degraded=True, expected_goal="增肌")
        assert result["_delivery_allowed"] is False
        assert "生成服务降级" in result["_delivery_block_reasons"]

    def test_missing_confidence_blocks(self):
        result = df.finalize_diet_result(_plan(), [_check(confidence=None)], 0,
                                         expected_goal="增肌")
        assert result["_delivery_allowed"] is False
        assert result["confidence"] == 0

    def test_extra_warnings_deliver_without_blocking(self):
        result = df.finalize_diet_result(
            _plan(), [_check()], 0, expected_goal="增肌",
            extra_warnings=["青少年处于生长发育期，不建议热量缺口"])
        assert result["_delivery_allowed"] is True
        assert "青少年处于生长发育期，不建议热量缺口" in result["warnings"]


class TestDietIntegrity:
    def test_unknown_food_residual_blocks_delivery(self):
        """库外食物没有可重算营养数字——绝不交付无宏量食物行。"""
        plan = _plan()
        plan["days"][0]["meals"][0]["foods"].append(
            {"name": "炸鸡架", "grams": 100, "_unknown": True,
             "kcal": 0, "protein_g": 0, "fat_g": 0, "carbs_g": 0})
        result = df.finalize_diet_result(plan, [_check()], 0, expected_goal="增肌")
        assert result["_delivery_allowed"] is False
        assert result["unknown_foods"][0]["food"] == "炸鸡架"
        assert "存在食物库未收录、无法核对营养的食物" in result["_delivery_block_reasons"]

    def test_broken_schema_blocks(self):
        result = df.finalize_diet_result({"plan_name": "x"}, [_check()], 0,
                                         expected_goal="增肌")
        assert result["_delivery_allowed"] is False
        assert "计划结构不完整" in result["_delivery_block_reasons"]


class TestPayloads:
    def test_crisis_payload_has_no_food_content(self):
        payload = df.diet_crisis_stop_payload(["教我催吐"])
        assert payload["delivery_status"] == "diet_crisis_stopped"
        assert payload["days"] == []
        assert payload["crisis"]["matched_signals"] == ["教我催吐"]
        assert payload["crisis"]["next_step"]
        assert "foods" not in payload and "meals" not in payload

    def test_emergency_payload_reuses_training_message_constants(self):
        payload = df.diet_emergency_stop_payload(["胸闷"])
        assert payload["delivery_status"] == pf.EMERGENCY_STOP_STATUS
        assert payload["emergency"]["message"] == pf.EMERGENCY_STOP_MESSAGE
        assert payload["days"] == []

    def test_safety_blocked_names_foods_and_gives_exit(self):
        result = df.finalize_diet_result(
            _plan(), [_check(is_safe=False, requires_review=True,
                             review_conflicts=[{"food": "牛奶（全脂）", "reason": "乳糖不耐"}],
                             review_reason="疾病冲突")], 0, expected_goal="增肌")
        payload = df.diet_safety_blocked_payload(result)
        assert payload["delivery_status"] == "safety_blocked"
        assert payload["days"] == []
        assert "牛奶（全脂）" in payload["safety"]["message"]
        assert payload["safety"]["prohibited_foods"] == ["牛奶（全脂）"]
        assert payload["safety"]["next_step"]

    def test_safety_blocked_without_conflicts_surfaces_checker_issues(self):
        """审查判 danger 但无确定性冲突食物时，不能只给一句空泛的「没通过」。
        实测糖尿病 persona 被拦下时 conflicts 为空——审查指出的问题必须透出。"""
        result = df.finalize_diet_result(
            _plan(), [_check(is_safe=False, requires_review=True,
                             issues=[{"issue": "糖尿病用户单餐血糖负荷过高", "severity": "danger"}])],
            0, expected_goal="增肌")
        payload = df.diet_safety_blocked_payload(result)
        assert payload["safety"]["conflicts"] == []
        assert "血糖负荷" in payload["safety"]["message"]
        assert payload["safety"]["issues"], "审查问题必须随载荷透出供前端渲染"
        assert payload["safety"]["title"] == "这份餐单没通过安全检查"
        assert payload["safety"]["next_step"]

    def test_generation_failed_carries_reasons_and_no_food(self):
        result = df.finalize_diet_result(_plan(goal="减脂"), [_check()], 0,
                                         expected_goal="增肌")
        payload = df.diet_generation_failed_payload(result)
        assert payload["delivery_status"] == "generation_failed"
        assert payload["days"] == []
        assert payload["failure"]["reasons"]

    def test_internal_markers_stripped_in_payload_not_in_result(self):
        result = df.finalize_diet_result(
            _plan(), [_check(issues=[{"issue": "[规则引擎] 牛奶与乳糖不耐冲突（触发词: 乳）"}])],
            0, expected_goal="增肌")
        assert "[规则引擎]" in result["warnings"][0]  # 原始数据保留（分类依赖它）
        payload = df.diet_generation_failed_payload(result)
        assert "[规则引擎]" not in payload["warnings"][0]
        assert "触发词" not in payload["warnings"][0]
        assert "牛奶与乳糖不耐冲突" in payload["warnings"][0]


class TestPrune:
    def test_prune_removes_conflicting_foods_and_empty_meals(self):
        plan = _plan()
        plan["days"][0]["meals"][0]["foods"] = [
            {"name": "花生（生）", "grams": 15}, {"name": "燕麦粥（煮）", "grams": 250}]
        pruned, removed = df.prune_conflicting_foods(plan, ["花生（生）"])
        assert removed == ["花生（生）"]
        assert [f["name"] for f in pruned["days"][0]["meals"][0]["foods"]] == ["燕麦粥（煮）"]

    def test_meal_emptied_is_dropped(self):
        plan = _plan()
        plan["days"][0]["meals"][0]["foods"] = [{"name": "花生（生）", "grams": 15}]
        pruned, _ = df.prune_conflicting_foods(plan, ["花生（生）"])
        assert [m["meal"] for m in pruned["days"][0]["meals"]] == ["午餐", "晚餐"]

    def test_prune_noop_without_targets(self):
        plan = _plan()
        pruned, removed = df.prune_conflicting_foods(plan, [])
        assert pruned is plan and removed == []