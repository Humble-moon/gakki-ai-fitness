#!/usr/bin/env python3
"""安全识别的分类测试。

覆盖评审关心的六类输入：正常需求、已知伤病、疑似急性损伤、急症、
否定表达、多伤病组合。只测确定性层（关键词表、伤病性质分类、伤病—动作
冲突表），不调用模型与 embedding，因此结果可复现，也不产生调用费用。

急症层按严格二分类评估：误报会让用户被一句"立即就医"吓退，漏报会延误
就医，两个方向都算错。一般安全话术层在设计上是"宁可过度触发"，因此只
报告触发率，不判对错。

用法：
    python eval/safety_classification_eval.py
    python eval/safety_classification_eval.py --json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.qa_safety import detect_emergency, detect_safety_concern
from src.hitl.review import HITLReview, classify_injury_nature

# ---------------------------------------------------------------------------
# 用例
# ---------------------------------------------------------------------------

#: 必须判为急症。用真实口语，不照抄词表——照抄等于自测。
EMERGENCY_POSITIVE = [
    "训练时突然胸闷，喘不上气",
    "刚才做完深蹲头晕，眼前发黑",
    "心跳特别快，还有点恶心",
    "跑完步胸口闷得厉害",
    "练到一半突然心慌手抖",
    "今天训练时晕倒了，现在还没缓过来",
    "呼吸急促，感觉吸不进气",
    "胸口像被压着，左边胳膊也发麻",
    "运动中失去意识了几秒钟",
    "突然心悸，坐着也觉得心跳很重",
    "训练时眼前一黑差点摔倒",
    "喘不过气，嘴唇有点发紫",
]

#: 不得判为急症。包含正常需求、普通伤病、否定表达与历史病史。
EMERGENCY_NEGATIVE = [
    # 正常需求
    "我想增肌，每周练四天怎么安排",
    "减脂期碳水应该吃多少",
    "怎么提高卧推重量",
    "深蹲的时候膝盖有点响，正常吗",
    "昨天练完腿今天肌肉酸痛",
    "跑步后小腿紧绷，想放松一下",
    # 普通伤病 / 亚急性
    "腰椎间盘突出，还能硬拉吗",
    "半月板损伤，想避开深蹲",
    "肩膀活动度不好，卧推要注意什么",
    "脚踝之前崴过，现在有点肿",
    # 否定表达
    "训练时没有胸闷，只是有点累",
    "我没有头晕，就是想问下深蹲姿势",
    "没有心脏问题，可以练高强度吗",
    "不存在呼吸困难，就是喘得比别人粗",
    "从来没晕过，想试试大重量",
    "胸口不闷也不疼，只是想加练",
    # 历史病史
    "以前有腰突，现在完全好了",
    "之前崴过脚，已经恢复半年了",
    "小时候骨折过，现在没有影响",
    "旧伤早就好了，可以正常训练",
]

#: 伤病性质分类。
INJURY_NATURE_CASES = [
    ("腰椎间盘突出", "structural"),
    ("半月板撕裂", "structural"),
    ("肩袖撕裂术后", "structural"),
    ("膝关节骨关节炎", "structural"),
    ("脚踝扭伤", "structural"),
    ("肋骨骨折", "structural"),
    ("颈椎病", "structural"),
    ("膝盖积液", "structural"),
    ("肩膀活动度不好", "limitation"),
    ("脚踝活动度受限", "limitation"),
    ("髋部僵硬", "limitation"),
    ("柔韧性差", "limitation"),
    ("关节活动范围小", "limitation"),
    ("背部容易紧", "limitation"),
]

#: 伤病—动作冲突：(伤病, 计划中的动作, 是否应在规则层判为阻断级冲突)
CONFLICT_CASES = [
    ("腰椎间盘突出", "硬拉", True),
    ("腰椎间盘突出", "卧推", False),
    ("膝关节骨关节炎", "深蹲", True),
    ("膝关节骨关节炎", "哑铃弯举", False),
    ("半月板撕裂", "跑步", True),
    ("肩袖撕裂", "杠铃推举", True),
    ("肩膀活动度不好", "卧推", False),
    ("脚踝活动度受限", "深蹲", False),
]


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------

def _binary_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) else 0.0,
        "false_negative_rate": round(fn / (fn + tp), 4) if (fn + tp) else 0.0,
        "accuracy": round((tp + tn) / (tp + fp + fn + tn), 4),
    }


def evaluate_emergency() -> dict:
    tp = fp = fn = tn = 0
    errors = []
    for text in EMERGENCY_POSITIVE:
        hit = detect_emergency(text)
        if hit:
            tp += 1
        else:
            fn += 1
            errors.append({"type": "漏报", "text": text})
    for text in EMERGENCY_NEGATIVE:
        hit = detect_emergency(text)
        if hit:
            fp += 1
            errors.append({"type": "误报", "text": text})
        else:
            tn += 1
    return {"metrics": _binary_metrics(tp, fp, fn, tn), "errors": errors}


def evaluate_injury_nature() -> dict:
    correct = 0
    errors = []
    confusion: dict[str, dict[str, int]] = {}
    for text, expected in INJURY_NATURE_CASES:
        actual = classify_injury_nature(text)
        confusion.setdefault(expected, {}).setdefault(actual, 0)
        confusion[expected][actual] += 1
        if actual == expected:
            correct += 1
        else:
            errors.append({"text": text, "expected": expected, "actual": actual})
    return {
        "accuracy": round(correct / len(INJURY_NATURE_CASES), 4),
        "total": len(INJURY_NATURE_CASES),
        "correct": correct,
        "confusion": confusion,
        "errors": errors,
    }


def _plan_with(exercise: str) -> dict:
    return {"days": [{"exercises": [{"name": exercise}]}], "user_query": ""}


def evaluate_conflicts() -> dict:
    review = HITLReview()
    tp = fp = fn = tn = 0
    errors = []
    for injury, exercise, expected in CONFLICT_CASES:
        danger = review._check_conflicts(
            _plan_with(exercise), {"injuries": [injury]})
        actual = bool(danger)
        if expected and actual:
            tp += 1
        elif expected and not actual:
            fn += 1
            errors.append({"type": "漏报", "injury": injury, "exercise": exercise})
        elif not expected and actual:
            fp += 1
            errors.append({"type": "误报", "injury": injury, "exercise": exercise,
                           "issues": [d.get("message", "") for d in danger]})
        else:
            tn += 1
    metrics = _binary_metrics(tp, fp, fn, tn)
    return {"metrics": metrics, "errors": errors}


def safety_trigger_rate() -> dict:
    """一般安全话术的触发率，只作描述，不判对错。"""
    groups = {
        "正常需求": [
            "我想增肌，每周练四天怎么安排",
            "减脂期碳水应该吃多少",
            "怎么提高卧推重量",
            "深蹲的时候膝盖有点响，正常吗",
            "昨天练完腿今天肌肉酸痛",
            "跑步后小腿紧绷，想放松一下",
        ],
        "否定表达": [
            "训练时没有胸闷，只是有点累",
            "我没有头晕，就是想问下深蹲姿势",
            "胸部不疼，只是想加练",
            "旧伤早就好了，可以正常训练",
        ],
        "多伤病": [
            "有腰椎间盘突出和膝盖关节炎",
            "半月板损伤，还有肩袖损伤",
            "肩周炎和网球肘",
            "脚踝扭伤后遗症，膝盖积液",
        ],
    }
    out = {}
    for name, cases in groups.items():
        hits = sum(1 for t in cases if detect_safety_concern(t))
        out[name] = {"hit": hits, "total": len(cases),
                     "rate": round(hits / len(cases), 4)}
    return out


def main() -> int:
    result = {
        "emergency": evaluate_emergency(),
        "injury_nature": evaluate_injury_nature(),
        "conflict": evaluate_conflicts(),
        "safety_trigger_rate": safety_trigger_rate(),
    }

    if "--json" in sys.argv:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    def pct(x: float) -> str:
        return f"{x * 100:.1f}%"

    print("=" * 62)
    print("安全识别分类测试（确定性层，不调用模型）")
    print("=" * 62)

    m = result["emergency"]["metrics"]
    print("\n[1] 急症识别（12 正 / 20 负）")
    print(f"    准确率 {pct(m['accuracy'])}  精确率 {pct(m['precision'])}  "
          f"召回率 {pct(m['recall'])}  F1 {m['f1']:.3f}")
    print(f"    误报率 {pct(m['false_positive_rate'])}  "
          f"漏报率 {pct(m['false_negative_rate'])}")
    for err in result["emergency"]["errors"]:
        print(f"    {err['type']}: {err['text']}")

    n = result["injury_nature"]
    print(f"\n[2] 伤病性质分类（{n['total']} 条）")
    print(f"    准确率 {pct(n['accuracy'])}（{n['correct']}/{n['total']}）")
    for err in n["errors"]:
        print(f"    误判: {err['text']} → {err['actual']}（期望 {err['expected']}）")

    c = result["conflict"]["metrics"]
    print(f"\n[3] 伤病—动作冲突（{len(CONFLICT_CASES)} 组）")
    print(f"    准确率 {pct(c['accuracy'])}  精确率 {pct(c['precision'])}  "
          f"召回率 {pct(c['recall'])}")
    for err in result["conflict"]["errors"]:
        print(f"    {err['type']}: {err['injury']} × {err['exercise']}")

    print("\n[4] 一般安全话术触发率（描述性，不判对错）")
    for name, v in result["safety_trigger_rate"].items():
        print(f"    {name}: {v['hit']}/{v['total']}（{pct(v['rate'])}）")

    out = ROOT / "eval" / "results" / "safety_classification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
