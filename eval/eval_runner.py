import json
import time
import argparse
from src.core.orchestrator import Orchestrator
from src.models.schemas import UserProfileInput

def run_eval(limit: int = None):
    with open("eval/test_queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
    if limit:
        queries = queries[:limit]

    orch = Orchestrator()
    results = []
    total_latency = 0
    cache_hits = 0

    for q in queries:
        profile = UserProfileInput(
            height=180, weight=80, training_years=1,
            goal=q.get("goal", "增肌"),
            available_equipment=q.get("equipment", ["哑铃"]),
            days_per_week=q.get("days", 4)
        )
        start = time.time()
        if "exercise" in q:
            result = orch.analyze_exercise(q["exercise"], q.get("desc", ""), profile)
        else:
            result = orch.generate_plan(profile, q["query"])
        latency = time.time() - start
        total_latency += latency
        if latency < 0.5:
            cache_hits += 1

        checks_passed = 0
        checks_total = len(q.get("checks", []))
        for check in q.get("checks", []):
            if check == "has_plan_id" and "plan_id" in result:
                checks_passed += 1
            elif check == "has_exercises":
                days_data = result.get("days", [])
                if any(len(d.get("exercises", [])) > 0 for d in days_data):
                    checks_passed += 1
            elif check == "has_issues" and result.get("issues_found"):
                checks_passed += 1
            elif check == "has_suggestions" and result.get("suggestions"):
                checks_passed += 1
            elif check == "cache_hit" and latency < 0.5:
                checks_passed += 1

        results.append({
            "id": q["id"], "latency_ms": round(latency * 1000),
            "checks": f"{checks_passed}/{checks_total}",
            "passed": checks_passed == checks_total
        })

    avg_latency = total_latency / len(queries) * 1000 if queries else 0
    pass_count = sum(1 for r in results if r["passed"])
    print(f"Total: {len(queries)} | Passed: {pass_count} | Avg latency: {avg_latency:.0f}ms | Cache hits: {cache_hits}")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['id']}: {r['latency_ms']}ms ({r['checks']})")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_eval(args.limit)
