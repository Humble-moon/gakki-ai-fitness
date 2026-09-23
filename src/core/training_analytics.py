"""
训练数据统计聚合与自适应调整规则引擎。

纯函数模块：不做任何 IO，输入是 TrainingLogStore.get_logs() 的返回结构，
输出可直接喂给前端图表与建议面板。

为什么把「算」和「读」分开：规则引擎的每一条分支都能被单元测试穷尽覆盖，
不需要数据库也不需要 LLM。这让「调整建议为什么可信」这个问题有确定性答案——
LLM 只负责把结论润色得更自然，不参与得出结论。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

# ----------------------------------------------------------------------
# 阈值常量（集中定义，便于调参与测试）
# ----------------------------------------------------------------------

# 判定为「进步空间」的 RPE 上限。
# 依据：RPE 与「还能做几次」(RIR) 的对应关系是 RIR ≈ 10 - RPE，RPE 7.0 相当于
# 还能做 3 次，仍属明显有余力；增肌训练的常用有效强度区间是 RIR 1~3，
# 完成度高且 RIR ≥ 3 时加重量是通行做法。
PROGRESS_MAX_RPE = 7.0
# 判定为「需要减载」的 RPE 下限
DELOAD_MIN_RPE = 8.0
# 判定为「需要减载」的近期 1RM 下滑幅度
DELOAD_DROP_PCT = -3.0
# 「近期」窗口大小：减载判断只看最近 N 次记录。
# 不能用全期平均——12 周里前半段的低负荷会把均值拉下来，导致「最近明显在
# 下滑且很吃力」的动作反而判不出减载，而它恰恰是最该减载的那个。
RECENT_WINDOW = 3
# 判定为「完成度不足」的完成率上限
VOLUME_CUT_MAX_COMPLETION = 0.6
# 判定为「一直跳过」的最近出现次数上限
SWAP_MAX_SESSIONS = 1

# 建议幅度的硬上限，防止规则跑飞给出危险增量。
# 注意：小重量动作会突破这个上限——哑铃最小配重差就是 2.5kg，对 20kg 的
# 动作意味着 +12.5%，这是健身房配重的物理限制，不是 bug。上限约束的是
# 「按比例递增」这条路径；按步进递增时以实际步进为准。
MAX_PROGRESS_PCT = 5.0
DELOAD_PCT = -10.0
WEIGHT_STEP_KG = 2.5


# ----------------------------------------------------------------------
# 基础计算
# ----------------------------------------------------------------------

def estimate_1rm(weight: float, reps: int) -> float:
    """用 Epley 公式估算单次最大重量（1RM）。

    公式：1RM = weight * (1 + reps / 30)

    为什么用估算值而不是直接比重量：同样是 60kg，做 12 次和做 8 次代表的力量
    水平不同。只看重量会在「重量没变但次数涨了」时误判为停滞，估算 1RM 把
    次数变化折算进去了。

    自重动作（weight=0）无意义，返回 0.0。
    """
    if not weight or weight <= 0 or not reps or reps <= 0:
        return 0.0
    return round(weight * (1 + reps / 30.0), 2)


def volume_of(sets: int, reps: int, weight: float) -> float:
    """单条明细的训练容量 = 组数 × 次数 × 重量。自重动作返回 0。"""
    if not sets or not reps or not weight or weight <= 0:
        return 0.0
    return float(sets) * float(reps) * float(weight)


def week_start_of(day: date) -> date:
    """取该日所在 ISO 周的周一，作为周聚合的桶标签。"""
    return day - timedelta(days=day.weekday())


def round_to_step(weight: float, step: float = WEIGHT_STEP_KG) -> float:
    """把重量归整到最近的 step 倍数（默认 2.5kg，健身房最小配重片）。"""
    if step <= 0:
        return round(weight, 2)
    return round(round(weight / step) * step, 2)


# ----------------------------------------------------------------------
# 统计聚合
# ----------------------------------------------------------------------

def compute_stats(logs: list[dict]) -> dict:
    """把训练日志聚合成分层统计，直接对应前端图表的数据源。

    输入：
        logs: TrainingLogStore.get_logs() 的返回（按日期倒序，含 entries）

    输出：
        {
          "totals":    {sessions, volume, avg_completion, avg_rpe},
          "weekly":    [{week_start, sessions, volume, completion, avg_rpe}],
          "exercises": [{name, sessions, first_weight, last_weight, trend_pct,
                         avg_rpe, completion, series:[{date, weight, est_1rm,
                         volume, rpe, completed}]}]
        }

    空输入返回结构完整的空壳，而不是 None——让前端渲染逻辑不必到处判空。
    """
    empty = {
        "totals": {"sessions": 0, "volume": 0.0, "avg_completion": 0.0, "avg_rpe": None},
        "weekly": [],
        "exercises": [],
    }
    if not logs:
        return empty

    weekly: dict[str, dict] = {}
    exercises: dict[str, dict] = {}
    total_volume = 0.0
    completion_hits = 0
    entry_count = 0
    rpe_values: list[float] = []

    for log in logs:
        day = _as_date(log.get("log_date"))
        if day is None:
            continue
        entries = log.get("entries") or []

        wk = week_start_of(day).isoformat()
        bucket = weekly.setdefault(wk, {
            "week_start": wk, "sessions": 0, "volume": 0.0,
            "_completed": 0, "_entries": 0, "_rpe": [],
        })
        bucket["sessions"] += 1

        session_rpe = log.get("session_rpe")
        if isinstance(session_rpe, (int, float)):
            rpe_values.append(float(session_rpe))

        for entry in entries:
            weight = float(entry.get("weight") or 0.0)
            sets = int(entry.get("sets") or 0)
            reps = int(entry.get("reps") or 0)
            completed = bool(entry.get("completed", True))
            name = (entry.get("exercise_name") or "").strip()
            vol = volume_of(sets, reps, weight)

            # ---- 全局计数 ----
            entry_count += 1
            completion_hits += 1 if completed else 0
            total_volume += vol
            entry_rpe = entry.get("rpe")
            if isinstance(entry_rpe, (int, float)):
                rpe_values.append(float(entry_rpe))

            # ---- 周桶 ----
            bucket["volume"] += vol
            bucket["_entries"] += 1
            bucket["_completed"] += 1 if completed else 0
            if isinstance(entry_rpe, (int, float)):
                bucket["_rpe"].append(float(entry_rpe))

            # ---- 动作桶 ----
            if not name:
                continue
            ex = exercises.setdefault(name, {
                "name": name, "_series": [], "_completed": 0, "_entries": 0, "_rpe": [],
            })
            ex["_series"].append({
                "date": day.isoformat(),
                "weight": weight,
                "est_1rm": estimate_1rm(weight, reps),
                "volume": vol,
                "rpe": float(entry_rpe) if isinstance(entry_rpe, (int, float)) else None,
                "completed": completed,
            })
            ex["_entries"] += 1
            ex["_completed"] += 1 if completed else 0
            if isinstance(entry_rpe, (int, float)):
                ex["_rpe"].append(float(entry_rpe))

    return {
        "totals": {
            "sessions": len(logs),
            "volume": round(total_volume, 2),
            "avg_completion": round(completion_hits / entry_count, 4) if entry_count else 0.0,
            "avg_rpe": round(sum(rpe_values) / len(rpe_values), 1) if rpe_values else None,
        },
        "weekly": [_finalize_week(b) for b in sorted(weekly.values(), key=lambda x: x["week_start"])],
        "exercises": sorted(
            (_finalize_exercise(e) for e in exercises.values()),
            key=lambda x: x["sessions"], reverse=True,
        ),
    }


def _finalize_week(bucket: dict) -> dict:
    """把周桶的内部计数字段折算成对外字段。"""
    n = bucket["_entries"]
    rpes = bucket["_rpe"]
    return {
        "week_start": bucket["week_start"],
        "sessions": bucket["sessions"],
        "volume": round(bucket["volume"], 2),
        "completion": round(bucket["_completed"] / n, 4) if n else 0.0,
        "avg_rpe": round(sum(rpes) / len(rpes), 1) if rpes else None,
    }


def _finalize_exercise(ex: dict) -> dict:
    """把动作桶折算成对外字段，并按日期正序整理进步曲线。"""
    series = sorted(ex["_series"], key=lambda s: s["date"])
    n = ex["_entries"]
    rpes = ex["_rpe"]

    # 趋势只在「有重量的动作」上计算；自重动作的 est_1rm 恒为 0，比较无意义
    weighted = [s for s in series if s["est_1rm"] > 0]
    trend_pct = None
    first_w = last_w = None
    if len(weighted) >= 2:
        first_1rm, last_1rm = weighted[0]["est_1rm"], weighted[-1]["est_1rm"]
        if first_1rm > 0:
            trend_pct = round((last_1rm - first_1rm) / first_1rm * 100, 1)
        first_w = weighted[0]["weight"]
        last_w = weighted[-1]["weight"]
    elif weighted:
        first_w = last_w = weighted[0]["weight"]

    recent_rpes = [s["rpe"] for s in series[-RECENT_WINDOW:] if s.get("rpe") is not None]

    return {
        "name": ex["name"],
        "sessions": len({s["date"] for s in series}),
        "first_weight": first_w,
        "last_weight": last_w,
        "trend_pct": trend_pct,
        "avg_rpe": round(sum(rpes) / len(rpes), 1) if rpes else None,
        "recent_rpe": round(sum(recent_rpes) / len(recent_rpes), 1) if recent_rpes else None,
        "completion": round(ex["_completed"] / n, 4) if n else 0.0,
        "last_weight_raw": weighted[-1]["weight"] if weighted else None,
        "recent_1rm": [s["est_1rm"] for s in weighted[-RECENT_WINDOW:]],
        "series": [
            {"date": s["date"], "weight": s["weight"], "est_1rm": s["est_1rm"],
             "volume": round(s["volume"], 2), "rpe": s["rpe"], "completed": s["completed"]}
            for s in series
        ],
    }


# ----------------------------------------------------------------------
# 规则引擎
# ----------------------------------------------------------------------

def rule_engine(stats: dict, plan: dict | None = None) -> list[dict]:
    """从统计数据推导调整建议。纯函数、无 IO、无 LLM，可完全离线运行。

    规则优先级（同一动作只取第一条命中的建议，避免自相矛盾）：
        1. deload      近 3 次估算 1RM 下滑超阈值且 RPE 偏高 → 减重 10%
        2. volume_cut  完成率过低（多次没练完）            → 组数 -1
        3. progress    余力充足且完成度高、趋势不降        → 加 2.5%~5%
        4. swap        近期几乎不出现（一直被跳过）        → 建议替换动作

    plan 参数保留用于把建议对齐到当前计划的动作，当前实现只依赖 stats，
    因此传 None 也能完整工作。

    每条建议都带确定性 reason 文案：即使 LLM 全线不可用，建议依然能产出。
    """
    items: list[dict] = []
    for ex in stats.get("exercises", []):
        for check in (_check_deload, _check_volume_cut, _check_progress, _check_swap):
            item = check(ex)
            if item:
                items.append(item)
                break
    return items


def _make_item(ex: dict, kind: str, suggested: dict, delta_pct: float,
               reason: str, severity: str) -> dict:
    """组装一条建议。item_id 必须稳定，apply 时靠它校验条目是否仍有效。"""
    return {
        "item_id": f"{ex['name']}:{kind}",
        "exercise_name": ex["name"],
        "type": kind,
        "current": {
            "weight": ex.get("last_weight_raw"),
            "trend_pct": ex.get("trend_pct"),
        },
        "suggested": suggested,
        "delta_pct": round(delta_pct, 1),
        "reason": reason,
        "severity": severity,
        "confidence": _confidence(ex),
    }


def _confidence(ex: dict) -> float:
    """数据点越多，建议越可信。上限 0.9，避免给出绝对化的承诺。"""
    return round(min(0.9, 0.5 + 0.06 * ex.get("sessions", 1)), 2)


def _check_deload(ex: dict) -> dict | None:
    """规则 1：力量下滑 + 高疲劳 → 减载。

    两个条件同时成立才触发，只看 RPE 会把「今天心情差」误判成需要减载。
    """
    recent = ex.get("recent_1rm") or []
    rpe = ex.get("recent_rpe")
    if len(recent) < 3 or rpe is None:
        return None
    peak = max(recent[:-1])
    if peak <= 0:
        return None
    drop = (recent[-1] - peak) / peak * 100
    if drop > DELOAD_DROP_PCT or rpe < DELOAD_MIN_RPE:
        return None

    cur_w = ex.get("last_weight_raw") or 0.0
    new_w = round_to_step(cur_w * (1 + DELOAD_PCT / 100))
    return _make_item(
        ex, "deload", {"weight": new_w}, DELOAD_PCT,
        f"最近三次估算 1RM 从 {peak}kg 降到 {recent[-1]}kg（{drop:.1f}%），"
        f"近期 RPE {rpe} 偏高，建议减载一周让身体恢复",
        "warn",
    )


def _check_volume_cut(ex: dict) -> dict | None:
    """规则 2：完成率过低 → 减组数。

    练不完通常不是意志力问题，而是给自己的量超过了当前恢复能力。
    """
    completion = ex.get("completion")
    if completion is None or completion >= VOLUME_CUT_MAX_COMPLETION:
        return None
    return _make_item(
        ex, "volume_cut", {"sets_delta": -1}, -25.0,
        f"完成率只有 {completion * 100:.0f}%，多数情况下没能按计划练完，"
        f"建议先减 1 组把完成度拉回来，恢复能力跟上后再加量",
        "warn",
    )


def _check_progress(ex: dict) -> dict | None:
    """规则 3：余力充足 → 加重。

    要求趋势必须算得出来且不为负：trend 为 None 说明加权数据不足两个点，
    此时「完成得轻松」可能只是还没练够次数，不足以支撑加重建议；
    趋势为负说明在退步，也不该加重。
    """
    rpe = ex.get("avg_rpe")
    completion = ex.get("completion")
    trend = ex.get("trend_pct")
    if rpe is None or completion is None:
        return None
    if rpe > PROGRESS_MAX_RPE or completion < 0.9:
        return None
    if trend is None or trend < 0:
        return None

    cur_w = ex.get("last_weight_raw") or 0.0
    if cur_w <= 0:
        return None          # 自重动作无法按重量递增
    delta = min(MAX_PROGRESS_PCT, WEIGHT_STEP_KG / cur_w * 100)
    new_w = round_to_step(cur_w * (1 + delta / 100))
    if new_w <= cur_w:
        new_w = round_to_step(cur_w + WEIGHT_STEP_KG)
    return _make_item(
        ex, "progress", {"weight": new_w}, (new_w - cur_w) / cur_w * 100,
        f"平均 RPE 只有 {rpe}，完成率 {completion * 100:.0f}%，"
        f"还有余力，建议加重到 {new_w}kg（{WEIGHT_STEP_KG}kg 递增）",
        "info",
    )


def _check_swap(ex: dict) -> dict | None:
    """规则 4：一直被跳过 → 建议替换。

    频繁跳过的动作往往是用户不喜欢或器械抢不到，硬留在计划里只会拉低完成率。
    """
    if ex.get("sessions", 0) > SWAP_MAX_SESSIONS:
        return None
    return _make_item(
        ex, "swap", {"action": "replace"}, 0.0,
        f"最近只练了 {ex.get('sessions', 0)} 次，说明这个动作很难执行到位，"
        f"建议换成同肌群的替代动作，而不是继续留在计划里拉低完成率",
        "notice",
    )


# ----------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------

def _as_date(value) -> date | None:
    """把日志里的日期字段统一成 date；无法解析返回 None。"""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not value:
        return None
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None
