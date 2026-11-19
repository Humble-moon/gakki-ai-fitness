"""饮食摄入统计 —— 纯函数、无 IO、不经过 LLM（照 training_analytics 范式）。

把饮食记录变成「摄入 vs 目标」的对照：连续日序列（空日 has_data=False、
数值 None 而不是 0，缺口不被折线跨过）、达标率、蛋白命中率、估算占比。

日类型判定用**实际训练日志日期**（当天有训练记录=训练日），与计划模板用
days_per_week 是两问两答：模板回答「打算怎么吃」，统计回答「实际练了没」。
没有训练历史时标「未分类」并用每日均值目标，不假装知道哪天练了。

``today`` 参数为了测试不依赖运行时钟；当天序列标 is_partial（与训练分析
「未结束的本周不算整周」同源教训：当天还没过完，摄入天然偏低）。
"""

from __future__ import annotations

from datetime import date, timedelta

# 摄入达标带：±10%。计划构建容差是 ±50 kcal（知识文档），但那是「配餐凑数」
# 的容差；真实进食波动更大，用比例带才不至于把正常一天判成不达标。
KCAL_BAND_RATIO = 0.10
PROTEIN_HIT_RATIO = 0.90


def compute_diet_stats(logs: list[dict], targets: dict | None = None,
                       training_dates=None, today: date | None = None,
                       window_days: int = 14) -> dict:
    today = today or date.today()
    window_days = max(1, min(int(window_days), 90))
    start = today - timedelta(days=window_days - 1)
    classified = training_dates is not None
    training_set = {d if isinstance(d, str) else d.isoformat()
                    for d in (training_dates or [])}

    by_date: dict[str, dict] = {}
    for log in logs or []:
        day = str(log.get("log_date") or "")
        if not day:
            continue
        bucket = by_date.setdefault(day, {"kcal": 0.0, "protein_g": 0.0, "custom_kcal": 0.0})
        bucket["kcal"] += float(log.get("kcal") or 0)
        bucket["protein_g"] += float(log.get("protein_g") or 0)
        bucket["custom_kcal"] += float(log.get("custom_kcal") or 0)

    days = []
    logged_kcal, logged_protein, within, protein_hits, custom_total = [], [], 0, 0, 0.0
    band_evaluated = protein_evaluated = 0
    for offset in range(window_days):
        day = start + timedelta(days=offset)
        key = day.isoformat()
        bucket = by_date.get(key)
        if classified:
            day_type = "训练日" if key in training_set else "休息日"
        else:
            day_type = "未分类"
        target_kcal = target_protein = None
        if targets:
            if day_type == "训练日":
                target_kcal = targets.get("training_day_kcal")
            elif day_type == "休息日":
                target_kcal = targets.get("rest_day_kcal")
            else:
                target_kcal = targets.get("daily_kcal")
            target_protein = targets.get("protein_g")
        entry = {
            "date": key,
            "day_type": day_type,
            "has_data": bucket is not None,
            "kcal": round(bucket["kcal"]) if bucket else None,
            "protein_g": round(bucket["protein_g"], 1) if bucket else None,
            "custom_kcal": round(bucket["custom_kcal"]) if bucket else None,
            "target_kcal": target_kcal,
            "target_protein_g": target_protein,
            "kcal_within_band": None,
            "protein_hit": None,
            "is_partial": day == today,
        }
        if bucket and target_kcal:
            band = max(50.0, target_kcal * KCAL_BAND_RATIO)
            entry["kcal_within_band"] = abs(bucket["kcal"] - target_kcal) <= band
            within += 1 if entry["kcal_within_band"] else 0
            band_evaluated += 1
        if bucket and target_protein:
            entry["protein_hit"] = bucket["protein_g"] >= target_protein * PROTEIN_HIT_RATIO
            protein_hits += 1 if entry["protein_hit"] else 0
            protein_evaluated += 1
        if bucket:
            logged_kcal.append(bucket["kcal"])
            logged_protein.append(bucket["protein_g"])
            custom_total += bucket["custom_kcal"]
        days.append(entry)

    total_kcal = sum(logged_kcal)
    summary = {
        "window_days": window_days,
        "days_logged": len(logged_kcal),
        "total_kcal": round(total_kcal),
        "avg_kcal": round(total_kcal / len(logged_kcal)) if logged_kcal else None,
        "avg_protein_g": round(sum(logged_protein) / len(logged_protein), 1) if logged_protein else None,
        # 达标率按「有目标可判定的天数」算：没给档案时是 None（无从判定），
        # 不是 0（假装全不达标）。
        "adherence_rate": round(within / band_evaluated, 2) if band_evaluated else None,
        "protein_hit_rate": round(protein_hits / protein_evaluated, 2) if protein_evaluated else None,
        "custom_share": round(custom_total / total_kcal, 2) if total_kcal else None,
        "classified": classified,
    }
    return {"days": days, "summary": summary}