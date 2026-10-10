#!/usr/bin/env python3
"""真实 LLM 端到端马拉松测试。

对多种用户画像走完整 SSE 生成链路，覆盖安全分支、器械分支、伤病分支，
记录耗时、事件序列、交付状态、动作图示覆盖，并追踪当日 LLM 成本。

用法：python scripts/marathon_test.py [--rounds N] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8503"

# 覆盖矩阵：正常路径 / 伤病 / 器械不足 / 组合 / 安全拦截 / 极端输入
PROFILES = [
    # --- 正常路径 ---
    ("增肌·健身房·无伤病", {
        "height": 178, "weight": 82, "training_years": 3, "goal": "增肌",
        "available_equipment": ["杠铃", "哑铃", "龙门架", "深蹲架", "平凳"],
        "days_per_week": 4, "injuries": [], "habits": ["晚上练"],
        "query": "突破卧推平台期", "session_id": "m01", "athlete_key": "mar-a"}),
    ("减脂·健身房·无伤病", {
        "height": 170, "weight": 88, "training_years": 2, "goal": "减脂",
        "available_equipment": ["杠铃", "哑铃", "跑步机", "龙门架"],
        "days_per_week": 5, "injuries": [], "habits": ["早上练"],
        "query": "想减到75公斤", "session_id": "m02", "athlete_key": "mar-b"}),
    ("增力诉求·健身房·进阶", {
        # 目标枚举只有「增肌/减脂」（前后端一致），力量诉求通过 query 表达。
        "height": 175, "weight": 90, "training_years": 5, "goal": "增肌",
        "available_equipment": ["杠铃", "深蹲架", "卧推凳", "硬拉台"],
        "days_per_week": 4, "injuries": [], "habits": [],
        "query": "深蹲想上200公斤", "session_id": "m03", "athlete_key": "mar-c"}),

    # --- 伤病分支 ---
    ("增肌·健身房·膝盖不适", {
        "height": 180, "weight": 85, "training_years": 3, "goal": "增肌",
        "available_equipment": ["杠铃", "哑铃", "腿举机"],
        "days_per_week": 4, "injuries": ["膝盖偶尔不舒服"], "habits": [],
        "query": "练腿日膝盖有点疼", "session_id": "m04", "athlete_key": "mar-d"}),
    ("增肌·健身房·肩伤", {
        "height": 176, "weight": 78, "training_years": 4, "goal": "增肌",
        "available_equipment": ["哑铃", "龙门架", "平凳"],
        "days_per_week": 4, "injuries": ["左肩有旧伤，推举时疼"], "habits": [],
        "query": "想练胸但肩不舒服", "session_id": "m05", "athlete_key": "mar-e"}),
    ("减脂·居家·腰伤", {
        "height": 172, "weight": 80, "training_years": 1, "goal": "减脂",
        "available_equipment": ["哑铃", "自重", "瑜伽垫"],
        "days_per_week": 3, "injuries": ["腰椎间盘突出"], "habits": [],
        "query": "腰不好能练什么", "session_id": "m06", "athlete_key": "mar-f"}),

    # --- 器械分支（历史截图问题） ---
    ("增肌·居家·器械不足", {
        "height": 178, "weight": 82, "training_years": 3, "goal": "增肌",
        "available_equipment": ["哑铃", "自重", "弹力带"],
        "days_per_week": 4, "injuries": [], "habits": ["晚上练"],
        "query": "在家练增肌", "session_id": "m07", "athlete_key": "mar-g"}),
    ("增肌·徒手·零器械", {
        "height": 175, "weight": 70, "training_years": 0, "goal": "增肌",
        "available_equipment": [], "days_per_week": 3, "injuries": [], "habits": [],
        "query": "什么都没有怎么练", "session_id": "m08", "athlete_key": "mar-h"}),
    ("减脂·居家·无伤病", {
        "height": 165, "weight": 70, "training_years": 1, "goal": "减脂",
        "available_equipment": ["哑铃", "自重", "弹力带"],
        "days_per_week": 3, "injuries": [], "habits": [],
        "query": "", "session_id": "m09", "athlete_key": "mar-i"}),
    ("塑形诉求·瑜伽垫·仅自重", {
        "height": 162, "weight": 55, "training_years": 1, "goal": "减脂",
        "available_equipment": ["瑜伽垫"], "days_per_week": 4,
        "injuries": [], "habits": [], "query": "想练核心和臀",
        "session_id": "m10", "athlete_key": "mar-j"}),

    # --- 组合分支 ---
    ("增肌·居家·膝伤+器械不足", {
        "height": 170, "weight": 75, "training_years": 2, "goal": "增肌",
        "available_equipment": ["哑铃", "弹力带"],
        "days_per_week": 3, "injuries": ["右膝半月板损伤"], "habits": [],
        "query": "在家练但膝盖不好", "session_id": "m11", "athlete_key": "mar-k"}),
    ("减脂·健身房·高血压", {
        "height": 168, "weight": 92, "training_years": 0, "goal": "减脂",
        "available_equipment": ["跑步机", "哑铃", "龙门架"],
        "days_per_week": 3, "injuries": ["高血压"], "habits": [],
        "query": "医生让我运动", "session_id": "m12", "athlete_key": "mar-l"}),

    # --- 安全拦截路径 ---
    ("康复·严重伤病", {
        "height": 175, "weight": 68, "training_years": 0, "goal": "增肌",
        "available_equipment": ["哑铃"], "days_per_week": 3,
        "injuries": ["刚做完前交叉韧带重建手术一个月"], "habits": [],
        "query": "术后想恢复训练", "session_id": "m13", "athlete_key": "mar-m"}),
    ("异常·心血管症状", {
        "height": 172, "weight": 85, "training_years": 2, "goal": "减脂",
        "available_equipment": ["哑铃"], "days_per_week": 3,
        "injuries": [], "habits": [],
        "query": "跑步时胸闷心慌还能继续吗", "session_id": "m14", "athlete_key": "mar-n"}),

    # --- 极端/边界输入 ---
    ("极端·超长query", {
        "height": 175, "weight": 75, "training_years": 2, "goal": "增肌",
        "available_equipment": ["杠铃", "哑铃"], "days_per_week": 4,
        "injuries": [], "habits": [],
        "query": "我想增肌" + "，还要注意饮食和休息" * 40,
        "session_id": "m15", "athlete_key": "mar-o"}),
    ("极端·特殊字符query", {
        "height": 175, "weight": 75, "training_years": 2, "goal": "增肌",
        "available_equipment": ["杠铃"], "days_per_week": 4,
        "injuries": [], "habits": [],
        "query": "<script>alert(1)</script>'; DROP TABLE users;-- 🏋️ emoji 测试",
        "session_id": "m16", "athlete_key": "mar-p"}),
    ("极端·零训练年限", {
        "height": 180, "weight": 60, "training_years": 0, "goal": "增肌",
        "available_equipment": ["哑铃", "平凳"], "days_per_week": 3,
        "injuries": [], "habits": [], "query": "太瘦了想变壮",
        "session_id": "m17", "athlete_key": "mar-q"}),
    ("极端·每天练7天", {
        "height": 178, "weight": 80, "training_years": 3, "goal": "增肌",
        "available_equipment": ["杠铃", "哑铃", "深蹲架"], "days_per_week": 7,
        "injuries": [], "habits": ["每天都想练"], "query": "想天天练",
        "session_id": "m18", "athlete_key": "mar-r"}),

    # --- 老年/特殊人群 ---
    ("老年·50岁减脂", {
        "height": 170, "weight": 85, "training_years": 0, "goal": "减脂",
        "available_equipment": ["跑步机", "哑铃"], "days_per_week": 3,
        "injuries": ["轻度关节炎"], "habits": [], "query": "50岁了还能练吗",
        "session_id": "m19", "athlete_key": "mar-s"}),
    ("女性·产后恢复", {
        "height": 163, "weight": 65, "training_years": 1, "goal": "减脂",
        "available_equipment": ["瑜伽垫", "哑铃", "弹力带"],
        "days_per_week": 3, "injuries": ["产后腹直肌分离"], "habits": [],
        "query": "产后恢复训练", "session_id": "m20", "athlete_key": "mar-t"}),
]

CST = timezone(timedelta(hours=8))


def cost_now() -> float | None:
    try:
        import redis
        r = redis.Redis(host="localhost", port=6380, decode_responses=True)
        key = f"cost:guard:daily:{datetime.now(CST).strftime('%Y-%m-%d')}"
        v = r.get(key)
        return float(v) if v else 0.0
    except Exception:
        return None


def stream(payload: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(
        BASE + "/api/v2/plan/stream",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    events: list[dict] = []
    t0 = time.time()
    first_at = None
    http_status = None
    err = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            http_status = resp.status
            for raw in resp:
                s = raw.decode("utf-8", "replace").strip()
                if not s.startswith("data: "):
                    continue
                try:
                    ev = json.loads(s[6:])
                except Exception:
                    continue
                if first_at is None:
                    first_at = time.time() - t0
                events.append(ev)
    except urllib.error.HTTPError as e:
        http_status = e.code
        try:
            err = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            err = str(e)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    return {"elapsed": round(time.time() - t0, 1),
            "first_event": round(first_at, 1) if first_at else None,
            "http_status": http_status, "error": err, "events": events}


def analyze(res: dict) -> dict:
    evs = res["events"]
    names = [e.get("event") for e in evs]
    terminals = [n for n in names if n in {"done", "error", "cancelled"}]
    done = next((e for e in evs if e.get("event") == "done"), None)
    data = (done or {}).get("data") or {}
    days = data.get("days") or []
    ex_names = [x.get("name") for d in days for x in (d.get("exercises") or [])]

    try:
        img_map = json.load(open(ROOT / "data" / "exercise_images.json"))
    except Exception:
        img_map = {}
    with_img = [n for n in ex_names if n in img_map]

    return {
        "elapsed": res["elapsed"],
        "first_event": res["first_event"],
        "http_status": res["http_status"],
        "error": res["error"],
        "event_kinds": sorted(set(names)),
        "event_count": len(evs),
        "terminals": terminals,
        "terminal_ok": len(terminals) == 1 and terminals[0] == "done",
        "delivery_status": data.get("delivery_status"),
        "plan_id": data.get("plan_id"),
        "days": len(days),
        "exercise_count": len(ex_names),
        "with_image": len(with_img),
        "image_coverage": round(len(with_img) / len(ex_names) * 100, 1) if ex_names else None,
        "held_reason": data.get("hold_reason") or data.get("review_reason"),
        "next_step": (data.get("next_step") or "")[:200],
        "prohibited": data.get("prohibited_actions") or [],
        "dup_names": len(ex_names) - len(set(ex_names)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1, help="每个画像重复轮数")
    ap.add_argument("--json", default="/tmp/gakki_test/marathon.json")
    ap.add_argument("--only", default=None, help="逗号分隔的画像序号（1起）")
    args = ap.parse_args()

    idx = None
    if args.only:
        idx = {int(x) for x in args.only.split(",")}

    results = []
    start = time.time()
    cost0 = cost_now()
    print(f"=== 马拉松开始 {datetime.now(CST):%H:%M:%S} | 成本基线 ¥{cost0} ===", flush=True)

    total = len(PROFILES) * args.rounds
    n = 0
    for rnd in range(args.rounds):
        for i, (label, prof) in enumerate(PROFILES, 1):
            if idx and i not in idx:
                continue
            n += 1
            payload = dict(prof)
            payload["session_id"] = f"{prof['session_id']}-r{rnd}"
            t0 = time.time()
            res = stream(payload)
            a = analyze(res)
            a["label"] = label
            a["round"] = rnd
            a["cost_after"] = cost_now()
            results.append(a)
            flag = "OK " if a["terminal_ok"] else "!! "
            print(f"[{n}/{total}] {flag}{label} | {a['elapsed']}s | "
                  f"首事件{a['first_event']}s | status={a['delivery_status']} | "
                  f"动作{a['exercise_count']} 图示{a['image_coverage']}% | "
                  f"成本¥{a['cost_after']}", flush=True)
            if a["error"]:
                print(f"        错误: {a['error'][:200]}", flush=True)
            if not a["terminal_ok"]:
                print(f"        终止事件: {a['terminals']} | 事件种类: {a['event_kinds']}", flush=True)
            Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2))

    dur = time.time() - start
    ok = sum(1 for r in results if r["terminal_ok"])
    print(f"\n=== 马拉松结束 用时 {dur/60:.1f} 分钟 | {ok}/{len(results)} 正常终止 ===", flush=True)
    print(f"=== 成本 ¥{cost0} → ¥{cost_now()} ===", flush=True)

    # 汇总异常
    bad = [r for r in results if not r["terminal_ok"]]
    if bad:
        print("\n异常清单：", flush=True)
        for r in bad:
            print(f"  - {r['label']}: {r['terminals']} err={str(r['error'])[:120]}", flush=True)
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
