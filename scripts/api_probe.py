#!/usr/bin/env python3
"""API 边界、异常与安全探针。

对全部路由发非法/越权/边界请求，验证：
  - 参数校验是否拦截（不把坏输入透传到 LLM）
  - 未认证访问是否被拒
  - 越权读写他人数据是否被拒
  - 注入类输入是否被安全处理
所有用例都用「应当快速失败」的输入，避免触发真实 LLM 生成。

用法：python scripts/api_probe.py [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8503"

RESULTS: list[dict] = []


def call(method: str, path: str, body=None, headers=None, timeout=30,
         cookie: str | None = None) -> dict:
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    if cookie:
        h["Cookie"] = cookie
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            return {"status": resp.status, "body": raw[:400],
                    "headers": dict(resp.headers), "elapsed": round(time.time() - t0, 2)}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        return {"status": e.code, "body": raw[:400],
                "headers": dict(e.headers), "elapsed": round(time.time() - t0, 2)}
    except Exception as e:
        return {"status": None, "body": f"{type(e).__name__}: {e}",
                "headers": {}, "elapsed": round(time.time() - t0, 2)}


def check(name: str, expect: str, got: dict, detail: str = "") -> None:
    """expect: 期望的状态类别。"""
    st = got["status"]
    ok = {
        "2xx": st is not None and 200 <= st < 300,
        "4xx": st is not None and 400 <= st < 500,
        "401": st == 401,
        "403": st == 403,
        "404": st == 404,
        "422": st == 422,
        "reject": st is not None and 400 <= st < 500,
    }.get(expect, False)
    RESULTS.append({"case": name, "expect": expect, "status": st,
                    "pass": ok, "body": got["body"][:200], "detail": detail,
                    "elapsed": got["elapsed"]})
    print(f"{'PASS' if ok else 'FAIL'} [{expect:>5} → {st}] {name}"
          + (f" | {detail}" if detail and not ok else ""), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="/tmp/gakki_test/api_probe.json")
    args = ap.parse_args()

    print("=== 健康检查 ===", flush=True)
    check("GET /health/live", "2xx", call("GET", "/health/live"))
    check("GET /health/ready", "2xx", call("GET", "/health/ready"))

    print("\n=== 参数校验：坏输入不应透传到 LLM ===", flush=True)
    # 类型错误
    check("generate-plan 身高为字符串", "reject",
          call("POST", "/api/generate-plan", {"height": "abc", "weight": 80}))
    check("generate-plan 天数为负数", "reject",
          call("POST", "/api/generate-plan", {"height": 175, "weight": 75, "days_per_week": -5}))
    check("generate-plan 天数为 0", "reject",
          call("POST", "/api/generate-plan", {"height": 175, "weight": 75, "days_per_week": 0}))
    check("generate-plan 天数超范围(99)", "reject",
          call("POST", "/api/generate-plan", {"height": 175, "weight": 75, "days_per_week": 99}))
    check("generate-plan 身高为 0", "reject",
          call("POST", "/api/generate-plan", {"height": 0, "weight": 75}))
    check("generate-plan 身高 999cm", "reject",
          call("POST", "/api/generate-plan", {"height": 999, "weight": 75}))
    check("generate-plan 体重负数", "reject",
          call("POST", "/api/generate-plan", {"height": 175, "weight": -10}))
    check("generate-plan 非法目标枚举", "reject",
          call("POST", "/api/generate-plan", {"goal": "变成超级赛亚人"}))
    check("generate-plan 器械为字符串而非数组", "reject",
          call("POST", "/api/generate-plan", {"available_equipment": "哑铃"}))
    check("generate-plan 空 body", "reject",
          call("POST", "/api/generate-plan", None))

    # v2 流式同样校验
    check("v2/plan/stream 天数负数", "reject",
          call("POST", "/api/v2/plan/stream", {"height": 175, "weight": 75, "days_per_week": -1}))
    check("v2/plan 身高为 0", "reject",
          call("POST", "/api/v2/plan", {"height": 0, "weight": 75}))

    # 其它接口
    check("analyze-exercise 缺动作名", "reject",
          call("POST", "/api/analyze-exercise", {}))
    check("analyze-exercise 空动作名", "reject",
          call("POST", "/api/analyze-exercise", {"exercise_name": ""}))
    check("ask-question 缺问题", "reject",
          call("POST", "/api/ask-question", {}))
    check("ask-question 空问题", "reject",
          call("POST", "/api/ask-question", {"question": ""}))

    print("\n=== 认证与越权 ===", flush=True)
    # 本产品支持游客模式：不带身份时返回 400 athlete_key_required（要求补身份），
    # 而不是 401。这里断言的是「必须被拒」，不锁死具体状态码。
    check("未登录访问数据导出", "reject", call("GET", "/api/user-data/export"))
    check("未登录删除数据", "reject", call("DELETE", "/api/user-data"))
    check("未登录看训练统计", "reject", call("GET", "/api/training-stats"))
    check("未登录看计划列表", "reject", call("GET", "/api/plans"))
    check("未登录看训练日志", "reject", call("GET", "/api/training-logs"))
    check("未登录写训练日志", "reject",
          call("POST", "/api/training-logs", {"entries": []}))

    print("\n=== 登录流程 ===", flush=True)
    import random
    uname = f"probe_{random.randint(100000, 999999)}"
    r = call("POST", "/api/auth/register", {"username": uname, "password": "ProbePass123!"})
    check("注册新用户", "2xx", r, r["body"][:120])
    cookie = r["headers"].get("set-cookie", "").split(";")[0] if r["headers"].get("set-cookie") else None

    r2 = call("POST", "/api/auth/login", {"username": uname, "password": "ProbePass123!"})
    check("登录成功", "2xx", r2)
    cookie = r2["headers"].get("set-cookie", "").split(";")[0] if r2["headers"].get("set-cookie") else cookie

    check("重复注册同名用户", "reject",
          call("POST", "/api/auth/register", {"username": uname, "password": "ProbePass123!"}))
    check("错误密码登录", "reject",
          call("POST", "/api/auth/login", {"username": uname, "password": "WrongPass!"}))
    check("不存在用户登录", "reject",
          call("POST", "/api/auth/login", {"username": "no_such_user_xyz", "password": "x"}))
    check("空密码注册", "reject",
          call("POST", "/api/auth/register", {"username": f"p{random.randint(1,99999)}", "password": ""}))
    check("SQL注入式用户名", "reject",
          call("POST", "/api/auth/login", {"username": "' OR '1'='1", "password": "x"}))

    if cookie:
        print(f"\n=== 已登录会话（{uname}）===", flush=True)
        check("查会话", "2xx", call("GET", "/api/auth/session", cookie=cookie))
        check("已登录导出数据", "2xx", call("GET", "/api/user-data/export", cookie=cookie))
        check("已登录看计划列表", "2xx", call("GET", "/api/plans", cookie=cookie))

        # 计划保存要求「已交付」的计划——这里给一个空计划，验证它被正确拒绝。
        r = call("POST", "/api/plans", {"title": "探针计划", "content": {"days": []}}, cookie=cookie)
        check("保存未交付计划被拒", "reject", r, r["body"][:120])
        try:
            pid = json.loads(r["body"]).get("plan_id") or json.loads(r["body"]).get("id")
        except Exception:
            pid = None
        if pid:
            check("读取自己的计划", "2xx", call("GET", f"/api/plans/{pid}", cookie=cookie))
            check("未登录读该计划", "reject", call("GET", f"/api/plans/{pid}"))
            check("删除自己的计划", "2xx", call("DELETE", f"/api/plans/{pid}", cookie=cookie))
            check("删除不存在的计划", "404", call("DELETE", f"/api/plans/{pid}", cookie=cookie))

        check("查看不存在的计划", "404", call("GET", "/api/plans/999999999", cookie=cookie))

    print("\n=== 注入与畸形输入 ===", flush=True)
    check("query 含 XSS", "2xx",
          call("POST", "/api/ask-question",
               {"question": "<script>alert(1)</script>", "athlete_key": "probe-x"}),
          "应为安全处理而非 500")
    check("超长 query(100k)", "reject",
          call("POST", "/api/ask-question",
               {"question": "啊" * 100000, "athlete_key": "probe-x"}))
    check("路径穿越 /api/plans/../", "reject",
          call("GET", "/api/plans/..%2F..%2Fetc%2Fpasswd"))
    check("exercise-images 无参", "2xx", call("GET", "/api/exercise-images"))
    # 该接口设计为返回全量清单（前端自行按名查找），不接收 name 参数——
    # 传了也会被忽略。这里断言的是「不因多余参数而报错」。
    check("exercise-images 带多余参数仍可用", "2xx",
          call("GET", "/api/exercise-images?name=" + urllib.parse.quote("不存在的动作xyz")))

    print("\n=== 管理端 ===", flush=True)
    # 未配置 ADMIN_TOKEN 时按失败关闭原则一律拒绝（403），这是有意的安全设计。
    check("未认证访问 /admin/metrics", "403", call("GET", "/admin/metrics"))
    check("未认证访问 /admin/reingest", "403",
          call("POST", "/admin/reingest?strategy=combo"))

    print("\n=== 静态资源缓存 ===", flush=True)
    r = call("GET", "/")
    cc = r["headers"].get("cache-control", "")
    ok = "no-cache" in cc.lower() or "no-store" in cc.lower()
    RESULTS.append({"case": "首页不缓存", "expect": "no-cache", "status": r["status"],
                    "pass": ok, "body": cc, "detail": cc, "elapsed": r["elapsed"]})
    print(f"{'PASS' if ok else 'FAIL'} [首页缓存头] {cc}", flush=True)

    print("\n=== 404 与错误码格式 ===", flush=True)
    r = call("GET", "/api/nonexistent-endpoint")
    check("未知路由 404", "404", r)
    r = call("GET", "/api/v2/reviews/nonexistent-review-id")
    check("不存在的审核工件", "404", r)

    Path(args.json).write_text(json.dumps(RESULTS, ensure_ascii=False, indent=2))
    passed = sum(1 for r in RESULTS if r["pass"])
    print(f"\n=== 探针结束 {passed}/{len(RESULTS)} 通过 ===", flush=True)
    fails = [r for r in RESULTS if not r["pass"]]
    if fails:
        print("\n未通过用例：", flush=True)
        for r in fails:
            print(f"  - {r['case']} | 期望{r['expect']} 实际{r['status']} | {r['body'][:150]}", flush=True)
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
