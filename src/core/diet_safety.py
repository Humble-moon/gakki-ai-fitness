"""饮食侧安全规则层 —— 纯函数、确定性、独立于 LLM。

镜像 ``src/hitl/review.py`` 的范式（确定性冲突表 + 独立词表），但面向「吃」：
过敏原、疾病饮食禁忌、进食障碍/极端节食信号。删掉 LLM 这一层仍然有效。

## 三条处置分级（与训练侧分流哲学同源，但饮食有自己的硬度）

1. **硬危机**（极端限制请求 + 意图）→ 拒绝生成饮食计划，独立话术，载荷不含
   任何食物内容。与急症同优先级层、话术互斥。
2. **软信号**（提及暴食/厌食/节食反弹但无极端请求）→ **不拒绝**。这是最该帮的
   用户：强制无缺口 + 关怀提示 + 建议专业支持，计划照常交付。单档「命中即拒绝」
   会把「我最近压力大总是暴食，想规律饮食」挡在门外——那是产品失职。
3. **过敏/疾病冲突＝硬阻断，永不降级为 warning**。过敏性休克是即刻人身伤害，
   与「库外动作名只提示」有本质区别：可定位→确定性替换→替换不掉才拒绝交付。

## 单一事实源

急症词表、归一化、否定语境全部 import 自 ``qa_safety``，不在这里复制第二份。
``emergency_signals_in``（训练计划入口）是裸 substring 不做否定判断；饮食侧
刻意走 ``find_affirmed_keyword``——「我没有催吐想法」不该被判成危机。
"""

from __future__ import annotations

import re

from src.core.qa_safety import (
    EMERGENCY_KEYWORDS,
    find_affirmed_keyword,
    negated_at,
    normalize_question,
)
from src.rag.food_catalog import get_food

# ----------------------------------------------------------------------
# 过敏原：用户自由文本 → 封闭标签（food_catalog.FOOD_ALLERGENS）
# ----------------------------------------------------------------------

ALLERGEN_KEYWORDS: dict[str, str] = {
    "花生": "花生", "落花生": "花生",
    "坚果": "坚果", "杏仁": "坚果", "核桃": "坚果", "腰果": "坚果",
    "榛子": "坚果", "开心果": "坚果", "巴旦木": "坚果",
    "麸质": "麸质", "小麦": "麸质", "大麦": "麸质", "黑麦": "麸质", "乳糜泻": "麸质",
    "大豆": "大豆", "黄豆": "大豆", "豆制品": "大豆", "大豆蛋白": "大豆",
    "鸡蛋": "蛋", "蛋清": "蛋", "蛋黄": "蛋", "蛋白过敏": "蛋",
    "牛奶": "乳", "乳糖": "乳", "乳制品": "乳", "奶酪": "乳", "芝士": "乳",
    "酸奶": "乳", "奶油": "乳", "奶粉": "乳",
    "鱼": "鱼", "三文鱼": "鱼", "带鱼": "鱼", "鳕鱼": "鱼", "鲈鱼": "鱼",
    "虾": "虾蟹", "蟹": "虾蟹", "螃蟹": "虾蟹", "海鲜": "虾蟹",
    "贝类": "贝类", "生蚝": "贝类", "牡蛎": "贝类", "扇贝": "贝类",
    "蛤蜊": "贝类", "鱿鱼": "贝类",
    "芝麻": "芝麻",
}

# ----------------------------------------------------------------------
# 疾病：目标层（改总量，换菜解决不了）+ 食物层（换菜能解决）
# ----------------------------------------------------------------------

DISEASE_SYNONYMS: dict[str, str] = {
    "糖尿病": "糖尿病", "血糖高": "糖尿病", "高血糖": "糖尿病", "二型糖尿病": "糖尿病",
    "痛风": "痛风", "尿酸高": "痛风", "高尿酸": "痛风",
    "高血压": "高血压", "血压高": "高血压",
    "肾病": "肾病", "肾炎": "肾病", "肾功能不全": "肾病", "肾衰竭": "肾病", "尿毒症": "肾病",
    "孕期": "孕期", "怀孕": "孕期", "妊娠": "孕期", "备孕": "孕期",
    "哺乳期": "哺乳期",
    "乳糖不耐": "乳糖不耐", "乳糖不耐受": "乳糖不耐", "乳糖酶缺乏": "乳糖不耐",
    "乳糜泻": "乳糜泻", "麸质过敏": "乳糜泻", "麸质不耐": "乳糜泻",
}

#: 目标层规则：蛋白上限（g/kg）与禁缺口。必须在 allocate_meals 之前覆盖总量。
DISEASE_TARGET_RULES: dict[str, dict] = {
    "肾病": {"protein_per_kg_cap": 0.8,
             "note": "肾病需限制蛋白总量，已按 0.8g/kg 上限覆盖蛋白目标"},
    "孕期": {"no_deficit": True, "exclude_flags": ("raw_or_high_mercury",),
             "exclude_names": ("猪肝（鲜）",),
             "note": "孕期不建议热量缺口，并排除生食/高汞鱼与动物肝脏"},
    "哺乳期": {"no_deficit": True,
               "note": "哺乳期不建议热量缺口"},
}

#: 食物层规则：排除标记 / 排除过敏原 / 排除名字。
DISEASE_FOOD_RULES: dict[str, dict] = {
    "糖尿病": {"exclude_flags": ("high_gi",), "note": "糖尿病需控制高 GI 主食"},
    "痛风": {"exclude_flags": ("high_purine",), "note": "痛风需排除高嘌呤食物"},
    "高血压": {"exclude_flags": ("high_sodium",), "note": "高血压需排除高钠食物"},
    "乳糖不耐": {"exclude_allergens": ("乳",), "note": "乳糖不耐需排除乳制品"},
    "乳糜泻": {"exclude_allergens": ("麸质",), "note": "乳糜泻需严格排除麸质"},
}

# ----------------------------------------------------------------------
# 危机词表：硬 / 软两档 + 趋势词（只 warning）
# ----------------------------------------------------------------------

#: 硬危机：极端限制请求 + 意图。命中即拒绝生成。
DIET_CRISIS_HARD: tuple[str, ...] = (
    "教我催吐", "怎么催吐", "催吐方法", "如何催吐", "催吐教程",
    "怎么绝食", "绝食减肥", "绝食瘦身", "饿瘦自己",
    "减肥药推荐", "推荐减肥药", "哪种减肥药", "减肥药哪个好",
    "泻药减肥", "利尿剂减肥",
    "一天只吃一顿", "每天只吃一顿", "一天只吃一个苹果", "只吃苹果减肥",
    "暴食后催吐", "吃完就抠吐",
    "怎么得厌食症", "如何患上厌食症",
)

#: 硬危机意图正则：极端低热量数字请求 / 第一人称催吐绝食意图。
#: 数字用命名组 kcal——单位是第 4 组，靠 lastindex 取数会取到单位而非数字。
_CRISIS_INTENT_PATTERNS = (
    re.compile(r"(一天|每天|每日).{0,6}(只吃|就吃|控制在|不超过|低于)[^\d]{0,6}(?P<kcal>\d{3,4})(?!\d)\s*(大卡|千卡|kcal|cal|卡)"),
    re.compile(r"(教我|怎么|如何|帮我|我想|我要).{0,4}(催吐|抠吐|绝食|饿瘦)"),
)
#: 数字阈值：低于此值的每日请求判硬危机（正常减脂缺口不会到这个量级）。
_CRISIS_KCAL_THRESHOLD = 1000

#: 软信号：状况词但无极端请求 → 不拒绝，无缺口 + 关怀 + 专业支持建议。
DIET_CRISIS_SOFT: tuple[str, ...] = (
    "暴食", "厌食", "进食障碍", "节食反弹", "减肥反弹", "催吐", "抠吐",
    "食物焦虑", "热量焦虑", "不敢吃饭", "害怕吃饭", "吃完就后悔",
    "暴食症", "厌食症",
)

#: 趋势词：只给 warning，不判危机（知识库本就有生酮文档，阻断它自相矛盾）。
DIET_TREND_WORDS: tuple[str, ...] = (
    "断食", "轻断食", "生酮", "酮症", "碳水循环", "极低碳水", "零碳水",
)

#: 第三人称标记：意图正则命中但主语是别人时不判硬危机（「朋友想学催吐」是求助）。
_THIRD_PERSON_MARKERS = ("朋友", "家人", "别人", "他人", "他", "她", "网友", "患者", "同事", "孩子")


def _third_person_at(text: str, start: int) -> bool:
    prefix = text[max(0, start - 6):start]
    return any(marker in prefix for marker in _THIRD_PERSON_MARKERS)


def _affirmed_positions(text: str, keywords):
    """词表中所有未被否定的命中位置 (keyword, index)。"""
    for keyword in keywords:
        start = 0
        while True:
            index = text.find(keyword, start)
            if index < 0:
                break
            if not negated_at(text, index):
                yield keyword, index
            start = index + len(keyword)


def resolve_allergens(raw_items) -> list[str]:
    """用户过敏输入（自由文本或标签）→ 封闭过敏原标签列表（去重保序）。"""
    tags: list[str] = []
    for item in raw_items or []:
        text = normalize_question(str(item or ""))
        if not text:
            continue
        if text in ("麸质", "花生", "坚果", "大豆", "蛋", "乳", "鱼", "虾蟹", "贝类", "芝麻"):
            tags.append(text)
            continue
        for keyword, tag in ALLERGEN_KEYWORDS.items():
            if keyword in text and tag not in tags:
                tags.append(tag)
    return tags


#: 自由文本健康自述的分句符。共享否定 helper 的窗口遇分句标点即判非否定，
#: 整句匹配会把「有痛风，没有高血压」里的高血压误识别——先分句再逐句判断。
_DISEASE_CLAUSE_SPLIT = re.compile(r"[，。、；;！!？?\n\r]+")


def resolve_diseases(raw_items) -> list[str]:
    """用户健康状况输入（自由文本或标签）→ 规范疾病名列表（去重保序）。

    健康状况已改为手工自由文本，同一段里可能混着肯定与否定
    （「有痛风，没有高血压」）。先按分句符切开，再逐句走共享否定判断——
    否定落在本句内才判得准；「没有糖尿病」不该被识别成糖尿病。
    封闭词表外的身体限制（如脂肪肝）不在此识别，交给配餐模型判断。
    """
    names: list[str] = []
    for item in raw_items or []:
        text = normalize_question(str(item or ""))
        if not text:
            continue
        for clause in _DISEASE_CLAUSE_SPLIT.split(text):
            clause = clause.strip()
            if not clause:
                continue
            if clause in DISEASE_TARGET_RULES or clause in DISEASE_FOOD_RULES:
                if clause not in names:
                    names.append(clause)
                continue
            for keyword, _idx in _affirmed_positions(clause, DISEASE_SYNONYMS.keys()):
                canonical = DISEASE_SYNONYMS[keyword]
                if canonical not in names:
                    names.append(canonical)
    return names


def detect_diet_crisis(text: str) -> dict | None:
    """饮食危机两档检测。返回 ``{"level": "hard"|"soft", "matched": [...]}`` 或 None。

    走 normalize_question（零宽/拼音/空格/emoji 绕过失效）+ 否定语境
    （「我没有催吐想法」不触发）+ 第三人称排除（「朋友想学催吐」不判硬危机）。
    硬档优先于软档。
    """
    normalized = normalize_question(text or "")
    if not normalized:
        return None

    hard_matched = [kw for kw, idx in _affirmed_positions(normalized, DIET_CRISIS_HARD)
                    if not _third_person_at(normalized, idx)]
    for pattern in _CRISIS_INTENT_PATTERNS:
        for match in pattern.finditer(normalized):
            if negated_at(normalized, match.start()) or _third_person_at(normalized, match.start()):
                continue
            kcal = match.groupdict().get("kcal")
            if kcal is not None and int(kcal) >= _CRISIS_KCAL_THRESHOLD:
                continue
            hard_matched.append(match.group(0))
    if hard_matched:
        return {"level": "hard", "matched": list(dict.fromkeys(hard_matched))}

    soft_matched = [kw for kw in DIET_CRISIS_SOFT
                    if find_affirmed_keyword(normalized, (kw,))]
    if soft_matched:
        return {"level": "soft", "matched": list(dict.fromkeys(soft_matched))}
    return None


def detect_diet_trends(text: str) -> list[str]:
    """趋势词命中列表（只 warning，不阻断）。

    重叠命中去重：「轻断食」命中时不再单列它的子串「断食」，否则提示会重复。
    """
    normalized = normalize_question(text or "")
    matched = [kw for kw in DIET_TREND_WORDS if find_affirmed_keyword(normalized, (kw,))]
    matched.sort(key=len, reverse=True)
    kept: list[str] = []
    for kw in matched:
        if not any(kw in longer for longer in kept):
            kept.append(kw)
    return kept


def scan_diet_signals(query: str = "", injuries=None, extra_text: str = "") -> dict:
    """饮食生成入口的信号预检：急症（复用同一常量）+ 危机 + 趋势。

    急症优先于危机由调用方（diet_service）决定；这里只负责把三类信号都检出来，
    且急症/危机都做否定语境判断（训练计划入口的裸 substring 不适用于饮食文本）。
    """
    combined = " ".join(part for part in (
        query or "", extra_text or "", " ".join(injuries or []),
    ) if part)
    normalized = normalize_question(combined)
    emergency = [kw for kw in EMERGENCY_KEYWORDS
                 if find_affirmed_keyword(normalized, (kw,))]
    return {
        "emergency": list(dict.fromkeys(emergency)),
        "crisis": detect_diet_crisis(combined),
        "trends": detect_diet_trends(combined),
    }


def check_food_conflicts(foods, prefs: dict) -> list[dict]:
    """逐食物检查过敏/疾病/忌口/库外冲突（可定位到具体食物）。

    ``foods`` 为 ``[{"name", "grams", "meal"?, "day_type"?}, ...]``；
    ``prefs`` 为 ``{"allergies": [...], "diseases": [...], "avoid_foods": [...]}``
    （过敏/疾病接受自由文本，内部归一）。

    返回冲突列表，每条含 kind（allergen/disease/avoid/unknown）、food、reason、
    以及定位信息（meal/day_type）。按 (food, kind, reason) 去重——同一食物因多个
    触发词命中不膨胀成多条（训练侧 R14 的教训）。
    """
    allergen_tags = set(resolve_allergens(prefs.get("allergies")))
    diseases = resolve_diseases(prefs.get("diseases"))
    avoid_names: set[str] = set()
    for raw_name in prefs.get("avoid_foods") or []:
        if not raw_name:
            continue
        entry = get_food(raw_name)
        avoid_names.add(entry["name"] if entry else normalize_question(str(raw_name)))

    food_rules: list[tuple[str, dict]] = []
    for disease in diseases:
        rule = DISEASE_FOOD_RULES.get(disease)
        if rule:
            food_rules.append((disease, rule))
        target_rule = DISEASE_TARGET_RULES.get(disease) or {}
        if target_rule.get("exclude_flags") or target_rule.get("exclude_names"):
            food_rules.append((disease, target_rule))

    conflicts: list[dict] = []
    seen: set[tuple] = set()
    for item in foods or []:
        name = (item or {}).get("name") or ""
        entry = get_food(name)
        located = {"meal": item.get("meal"), "day_type": item.get("day_type")}
        if entry is None:
            key = (name, "unknown", "库外食物")
            if key not in seen:
                seen.add(key)
                conflicts.append({"kind": "unknown", "food": name,
                                  "reason": "食物库未收录，无法核对营养数字", **located})
            continue
        reasons: list[tuple[str, str]] = []
        for tag in entry["allergens"]:
            if tag in allergen_tags:
                reasons.append(("allergen", f"含过敏原「{tag}」"))
        for disease, rule in food_rules:
            for flag in rule.get("exclude_flags", ()):
                if entry.get(flag):
                    reasons.append(("disease", f"{disease}：{rule.get('note', flag)}"))
            for tag in rule.get("exclude_allergens", ()):
                if tag in entry["allergens"]:
                    reasons.append(("disease", f"{disease}：{rule.get('note', tag)}"))
            if entry["name"] in rule.get("exclude_names", ()):
                reasons.append(("disease", f"{disease}：{rule.get('note', entry['name'])}"))
        if entry["name"] in avoid_names:
            reasons.append(("avoid", "用户忌口"))
        for kind, reason in reasons:
            key = (entry["name"], kind, reason)
            if key in seen:
                continue
            seen.add(key)
            conflicts.append({"kind": kind, "food": entry["name"],
                              "reason": reason, **located})
    return conflicts


def apply_disease_target_rules(targets: dict, diseases, weight_kg: float) -> dict:
    """疾病目标层覆盖：肾病蛋白上限 / 孕期哺乳期禁缺口。

    返回 ``{"targets": 新字典, "warnings": [...], "applied": [...]}``，不修改入参。
    必须在 allocate_meals 之前调用——总量超标靠换菜解决不了。
    """
    from src.core.nutrition import split_macros  # 局部 import 避免环状依赖

    out = dict(targets)
    warnings: list[str] = []
    applied: list[str] = []
    for disease in resolve_diseases(diseases):
        rule = DISEASE_TARGET_RULES.get(disease)
        if not rule:
            continue
        cap = rule.get("protein_per_kg_cap")
        if cap is not None:
            capped = round(float(weight_kg) * cap)
            if capped < out.get("protein_g", 0):
                out["protein_g"] = capped
                applied.append(disease)
                warnings.append(rule["note"] + f"（{capped}g/天）；请遵医嘱定期复查。")
        if rule.get("no_deficit") and out.get("delta_pct", 0) < 0:
            out["daily_kcal"] = out["tdee"]
            out["delta_pct"] = 0
            out["goal_note"] = "维持期（疾病护栏：不建议热量缺口）"
            out["basis"] = out.get("basis", "") + "｜疾病护栏取消热量缺口"
            applied.append(disease)
            warnings.append(rule["note"] + "；饮食调整请先咨询医生或临床营养师。")
    if applied:
        out.update(split_macros(out["daily_kcal"], out.get("protein_g", 0)))
    return {"targets": out, "warnings": warnings, "applied": applied}