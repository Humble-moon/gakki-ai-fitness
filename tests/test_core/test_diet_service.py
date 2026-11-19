"""diet_service 全流程测试：假 LLM / 假知识 / 假记忆，不联网。

覆盖三类 persona（健康/过敏/危机）+ 记忆并集读回 + 空值不覆盖过敏史 +
降级路径 + 重写与摘除兜底。安全顺序（急症 > 硬危机 > 软危机不拒绝）逐条钉死。
"""


from src.core import diet_service as svc
from src.llm.provider import LLMUnavailableError


def _plan(foods_by_meal=None):
    """两日型 × 三餐的合法计划；foods_by_meal 可注入冲突食物。"""
    default = {
        "早餐": [{"name": "燕麦粥（煮）", "grams": 250}, {"name": "鸡蛋（煮）", "grams": 100}],
        "午餐": [{"name": "米饭（蒸）", "grams": 200}, {"name": "鸡胸肉（煮）", "grams": 150}],
        "晚餐": [{"name": "糙米饭（蒸）", "grams": 150}, {"name": "西兰花", "grams": 200}],
    }
    meals_map = foods_by_meal or default
    days = []
    for day_type in ("训练日", "休息日"):
        days.append({"day_type": day_type, "meals": [
            {"meal": meal, "foods": [dict(f) for f in foods]}
            for meal, foods in meals_map.items()]})
    return {"plan_name": "测试饮食计划", "goal": "增肌", "days": days,
            "notes": "", "shopping_list": []}


class _ServiceFakeLLM:
    """按 system 角色分流：审查员 → checker payload；其余 → writer/rewrite。"""

    def __init__(self, writer=None, checker=None, rewrite=None, unavailable=False):
        self.writer = writer if writer is not None else _plan()
        self.checker = checker if checker is not None else {
            "is_safe": True, "issues": [], "confidence": 0.9,
            "requires_human_review": False}
        self.rewrite = rewrite
        self.unavailable = unavailable
        self.calls = []

    def chat_with_json_mode(self, messages, model=None):
        if self.unavailable:
            raise LLMUnavailableError("模拟不可用", attempted_models=["fake"],
                                      errors=["fake"], reason="unavailable")
        system = messages[0]["content"]
        user = messages[1]["content"]
        self.calls.append({"system": system[:20], "model": model, "user": user[:40]})
        if "审查员" in system:
            return dict(self.checker)
        if "必须修正的问题" in user and self.rewrite is not None:
            return dict(self.rewrite)
        return dict(self.writer)


class _FakeKnowledge:
    def __init__(self, fail=False):
        self.fail = fail

    def search_with_fallback(self, query):
        if self.fail:
            raise RuntimeError("embedding 服务抖动")
        return [{"content": "每餐保证 30-40g 蛋白质。", "source": "100-yinshi-jihua.md"}]


class _FakeLongTerm:
    def __init__(self, stored=None):
        self.stored = stored or {}
        self.writes = []

    def get_preferences(self, key):
        return dict(self.stored)

    def save_preference(self, key, name, value):
        self.writes.append((key, name, value))
        self.stored[name] = value


PROFILE = {"height": 175, "weight": 70, "age": 25, "sex": "男",
           "goal": "增肌", "days_per_week": 4, "injuries": []}


class TestPersonas:
    def test_healthy_persona_delivers_with_linked_targets(self):
        memory = _FakeLongTerm()
        result = svc.build_diet_plan(PROFILE, {}, llm=_ServiceFakeLLM(),
                                     knowledge=_FakeKnowledge(), long_term=memory,
                                     owner_key="g1_test")
        assert result["delivery_status"] == "safe_delivered"
        targets = result["daily_targets"]
        assert targets["training_day_kcal"] > targets["rest_day_kcal"]
        assert targets["weekly_mean_kcal"] == targets["daily_kcal"]
        assert set(result["explain"]) == {"goal", "energy", "foods", "safety"}
        assert result["nutrition"]["daily_kcal"] == targets["daily_kcal"]
        assert any(name == "diet_preferences" for _, name, _ in memory.writes)

    def test_undersized_portions_calibrated_before_delivery(self):
        """真实 E2E 回归：模型选的量蛋白达标、热量短三成，闸门曾照常放行。
        交付前必须经确定性克数校准——食物选择不变，各餐热量回到目标容差内。"""
        short = _plan({
            "早餐": [{"name": "燕麦粥（煮）", "grams": 150}, {"name": "鸡蛋（煮）", "grams": 100},
                     {"name": "牛奶（全脂）", "grams": 250}, {"name": "香蕉", "grams": 100}],
            "午餐": [{"name": "米饭（蒸）", "grams": 200}, {"name": "鸡胸肉（煮）", "grams": 100},
                     {"name": "全麦面包", "grams": 50}, {"name": "橄榄油", "grams": 10}],
            "晚餐": [{"name": "红薯（蒸）", "grams": 150}, {"name": "三文鱼", "grams": 80},
                     {"name": "全麦面包", "grams": 50}, {"name": "橄榄油", "grams": 10}],
        })
        result = svc.build_diet_plan(PROFILE, {}, llm=_ServiceFakeLLM(writer=short),
                                     meal_count=3)
        assert result["delivery_status"] == "safe_delivered"
        targets = result["daily_targets"]
        for day in result["days"]:
            target = (targets["training_day_kcal"] if day["day_type"] == "训练日"
                      else targets["rest_day_kcal"])
            actual = sum(m["kcal"] for m in day["meals"])
            assert abs(actual - target) <= 100, f"{day['day_type']} {actual} vs {target}"
        assert any("校准" in line for line in result["explain"]["foods"]["lines"])

    def test_emergency_short_circuits_before_any_food(self):
        memory = _FakeLongTerm()
        result = svc.build_diet_plan(PROFILE, {"notes": "训练时胸闷头晕"},
                                     llm=_ServiceFakeLLM(), long_term=memory,
                                     owner_key="g1_test")
        assert result["delivery_status"] == "emergency_stopped"
        assert result["days"] == [] and memory.writes == []

    def test_hard_crisis_refuses_with_care_message(self):
        memory = _FakeLongTerm()
        result = svc.build_diet_plan(PROFILE, {"notes": "教我催吐，一天只吃 500 大卡"},
                                     llm=_ServiceFakeLLM(),
                                     long_term=memory, owner_key="g1_test")
        assert result["delivery_status"] == "diet_crisis_stopped"
        assert result["days"] == [] and result["crisis"]["next_step"]
        assert memory.writes == []

    def test_soft_crisis_delivers_no_deficit_with_care(self):
        cutting_plan = _plan()
        cutting_plan["goal"] = "减脂"
        result = svc.build_diet_plan(dict(PROFILE, goal="减脂"),
                                     {"notes": "最近压力大总是暴食，想规律饮食"},
                                     llm=_ServiceFakeLLM(writer=cutting_plan))
        assert result["delivery_status"] == "safe_delivered"
        assert result["daily_targets"]["delta_pct"] == 0
        assert any("专业支持" in w for w in result["warnings"])

    def test_trend_word_warns_but_delivers(self):
        result = svc.build_diet_plan(PROFILE, {"notes": "我想试试轻断食"},
                                     llm=_ServiceFakeLLM())
        assert result["delivery_status"] == "safe_delivered"
        assert any("轻断食" in w for w in result["warnings"])


class TestHealthFreeText:
    """健康状况＝手工自由文本：已知疾病确定性识别走硬规则，否定不误判，
    记忆并集读回；其余身体限制交给模型（prompt 渲染在 writer 测试钉）。"""

    def test_free_text_extracts_disease_and_applies_target_rule(self):
        result = svc.build_diet_plan(PROFILE, {"health": "肾功能不全，蛋白要控制"},
                                     llm=_ServiceFakeLLM())
        assert result["delivery_status"] == "safe_delivered"
        assert result["daily_targets"]["protein_g"] == 56  # 70kg × 0.8g/kg
        assert any("肾病" in w for w in result["warnings"])

    def test_negated_condition_not_extracted(self):
        result = svc.build_diet_plan(PROFILE, {"health": "没有肾病，身体挺健康"},
                                     llm=_ServiceFakeLLM())
        assert result["daily_targets"]["protein_g"] > 56
        assert not any("肾病" in w for w in result["warnings"])

    def test_stored_health_diseases_union_when_request_empty(self):
        memory = _FakeLongTerm(stored={"diet_preferences": {
            "health": "肾病", "diseases": ["肾病"]}})
        result = svc.build_diet_plan(PROFILE, {}, llm=_ServiceFakeLLM(),
                                     long_term=memory, owner_key="g1_test")
        assert result["daily_targets"]["protein_g"] == 56  # 请求没带 ≠ 没有


class TestAllergyHardBlock:
    def test_conflicting_food_deterministically_substituted(self):
        bad = _plan({"早餐": [{"name": "花生（生）", "grams": 15},
                              {"name": "燕麦粥（煮）", "grams": 250}],
                     "午餐": [{"name": "米饭（蒸）", "grams": 200},
                              {"name": "鸡胸肉（煮）", "grams": 150}],
                     "晚餐": [{"name": "糙米饭（蒸）", "grams": 150},
                              {"name": "西兰花", "grams": 200}]})
        result = svc.build_diet_plan(PROFILE, {"allergies": ["花生过敏"]},
                                     llm=_ServiceFakeLLM(writer=bad))
        assert result["delivery_status"] == "safe_delivered"
        names = {f["name"] for d in result["days"] for m in d["meals"] for f in m["foods"]}
        assert "花生（生）" not in names
        assert any("→" in w for w in result["warnings"])

    def test_memory_union_readback_blocks_even_when_request_empty(self):
        memory = _FakeLongTerm({"diet_preferences": {"allergies": ["花生过敏"]}})
        bad = _plan({"早餐": [{"name": "花生（生）", "grams": 15},
                              {"name": "燕麦粥（煮）", "grams": 250}],
                     "午餐": [{"name": "米饭（蒸）", "grams": 200},
                              {"name": "鸡胸肉（煮）", "grams": 150}],
                     "晚餐": [{"name": "糙米饭（蒸）", "grams": 150},
                              {"name": "西兰花", "grams": 200}]})
        result = svc.build_diet_plan(PROFILE, {}, llm=_ServiceFakeLLM(writer=bad),
                                     long_term=memory, owner_key="g1_test")
        names = {f["name"] for d in result["days"] for m in d["meals"] for f in m["foods"]}
        assert "花生（生）" not in names

    def test_empty_request_does_not_erase_stored_allergies(self):
        memory = _FakeLongTerm({"diet_preferences": {"allergies": ["花生过敏"]}})
        svc.build_diet_plan(PROFILE, {"allergies": []}, llm=_ServiceFakeLLM(),
                            long_term=memory, owner_key="g1_test")
        written = [v for _, name, v in memory.writes if name == "diet_preferences"]
        assert written and "花生过敏" in written[-1]["allergies"]

    def test_unresolvable_conflict_ends_safety_blocked_naming_food(self):
        # 乳品整类都含乳过敏原：替换候选滤空、重写仍冲突、摘除后餐次不足 → 拒绝
        bad = _plan({"早餐": [{"name": "牛奶（全脂）", "grams": 250}],
                     "午餐": [{"name": "酸奶（无糖）", "grams": 150}],
                     "晚餐": [{"name": "奶酪（切达）", "grams": 20}]})
        result = svc.build_diet_plan(PROFILE, {"allergies": ["牛奶过敏"]},
                                     llm=_ServiceFakeLLM(writer=bad, rewrite=bad))
        assert result["delivery_status"] in ("safety_blocked", "generation_failed")
        assert result["days"] == []


class TestDegradation:
    def test_llm_unavailable_returns_generation_failed_not_500(self):
        result = svc.build_diet_plan(PROFILE, {}, llm=_ServiceFakeLLM(unavailable=True))
        assert result["delivery_status"] == "generation_failed"
        assert result["failure"]["reasons"]

    def test_checker_degraded_returns_generation_failed(self):
        result = svc.build_diet_plan(PROFILE, {}, llm=_ServiceFakeLLM(checker={"raw": "散文"}))
        assert result["delivery_status"] == "generation_failed"
        assert "生成服务降级" in result["failure"]["reasons"]

    def test_knowledge_failure_does_not_block_delivery(self):
        result = svc.build_diet_plan(PROFILE, {}, llm=_ServiceFakeLLM(),
                                     knowledge=_FakeKnowledge(fail=True))
        assert result["delivery_status"] == "safe_delivered"
        assert any("未检索到补充知识" in line
                   for line in result["explain"]["foods"]["lines"])

    def test_memory_failure_does_not_block_delivery(self):
        class _BrokenMemory(_FakeLongTerm):
            def get_preferences(self, key):
                raise RuntimeError("redis down")

            def save_preference(self, key, name, value):
                raise RuntimeError("redis down")

        result = svc.build_diet_plan(PROFILE, {}, llm=_ServiceFakeLLM(),
                                     long_term=_BrokenMemory(), owner_key="g1_test")
        assert result["delivery_status"] == "safe_delivered"


class TestRewriteAndPrune:
    def test_rewrite_loop_resolves_conflict(self):
        bad = _plan({"早餐": [{"name": "牛奶（全脂）", "grams": 250},
                              {"name": "燕麦粥（煮）", "grams": 250}],
                     "午餐": [{"name": "米饭（蒸）", "grams": 200},
                              {"name": "鸡胸肉（煮）", "grams": 150}],
                     "晚餐": [{"name": "糙米饭（蒸）", "grams": 150},
                              {"name": "西兰花", "grams": 200}]})
        # 乳糖不耐：乳品替换候选存在（豆浆不在乳品 allergens）→ 确定性替换即可
        result = svc.build_diet_plan(PROFILE, {"health": "乳糖不耐"},
                                     llm=_ServiceFakeLLM(writer=bad))
        assert result["delivery_status"] == "safe_delivered"
        names = {f["name"] for d in result["days"] for m in d["meals"] for f in m["foods"]}
        assert "牛奶（全脂）" not in names

    def test_unknown_food_removed_and_rebalanced_out(self):
        bad = _plan()
        bad["days"][0]["meals"][0]["foods"].append({"name": "炸鸡架", "grams": 100})
        result = svc.build_diet_plan(PROFILE, {}, llm=_ServiceFakeLLM(writer=bad))
        assert result["delivery_status"] == "safe_delivered"
        assert not any(f.get("_unknown") for d in result["days"]
                       for m in d["meals"] for f in m["foods"])
        assert any("库外摘除" in w for w in result["warnings"])

    def test_goal_mismatch_from_llm_is_rejected(self):
        bad = _plan()
        bad["goal"] = "减脂"
        result = svc.build_diet_plan(PROFILE, {}, llm=_ServiceFakeLLM(writer=bad))
        assert result["delivery_status"] == "generation_failed"
        assert "计划目标与你的目标不一致" in result["failure"]["reasons"]