#!/usr/bin/env python3
"""End-to-end demo: the LangGraph human-review loop survives a process restart.

This demonstrates the checkpoint-persistence story for the thesis:

  Phase 1 (one Python process) builds the coach graph against a SQLite
  checkpointer, generates a plan that the safety gate escalates to human review,
  and pauses at the review gate. The paused state is written to SQLite, and the
  thread_id / review_id are handed off via a small JSON file. The process then
  exits — simulating a crash or a deploy.

  Phase 2 (a *fresh* Python process) rebuilds the graph against the SAME SQLite
  file with brand-new in-memory collaborators, proves the run is still paused at
  ``review_gate``, then resumes it with ``Command(resume=...)`` and delivers the
  approved plan. Nothing about the original request needs to be re-submitted —
  it all lives in the checkpoint.

Run:  .venv/bin/python scripts/demo_graph_hitl.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DB_PATH = str(ROOT / "data" / "demo_graph_checkpoints.db")
HANDOFF_PATH = str(ROOT / "data" / "demo_handoff.json")


# ---------------------------------------------------------------------------
# Shared fake collaborators (offline, deterministic)
# ---------------------------------------------------------------------------
def _build_fake_deps():
    from types import SimpleNamespace
    from src.graph.deps import CoachGraphDeps
    from src.hitl.review_resolution import InMemoryReviewResolutionStore, ReviewThreadIndex
    from src.hitl.review_store import InMemoryReviewArtifactStore

    plan = {"plan_id": "demo-plan", "user_id": 1, "goal": "增肌", "weeks": 4,
            "sessions_per_week": 1,
            "days": [{"day": 1, "focus": "全身", "exercises": []}]}
    needs_review = {"is_safe": True, "issues": [], "requires_human_review": True,
                    "confidence": .9, "review_reason": "伤病冲突需要人工确认",
                    "review_severity": "warning"}

    def write_plan_stream(retrieved, profile, plan_config, plan_context="", user_query=""):
        yield ("chunk", "生成中")
        yield ("done", json.loads(json.dumps(plan)))

    return CoachGraphDeps(
        planner=SimpleNamespace(plan=lambda *a, **k: {"skill": "demo", "subtasks": [],
                                                      "skill_config": {}}),
        retriever=SimpleNamespace(retrieve=lambda *a, **k: {"exercises": []}),
        writer=SimpleNamespace(write_plan_stream=write_plan_stream,
                               rewrite_plan=lambda *a, **k: dict(plan)),
        fact_checker=SimpleNamespace(check=lambda *a, **k: dict(needs_review)),
        cache=SimpleNamespace(get=lambda *a, **k: None, set=lambda *a, **k: None),
        conversation=SimpleNamespace(add_turn=lambda *a, **k: None,
                                     build_context_for_prompt=lambda *a, **k: "",
                                     get_plan_state=lambda *a, **k: "",
                                     set_plan_state=lambda *a, **k: None),
        long_term=SimpleNamespace(save_preference=lambda *a, **k: None),
        review_store=InMemoryReviewArtifactStore(),
        resolutions=InMemoryReviewResolutionStore(),
        thread_index=ReviewThreadIndex(),
    )


def _sqlite_checkpointer():
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def phase1() -> int:
    """Generate a plan that is held for review; pause and persist the checkpoint."""
    import uuid
    from src.graph.builder import build_coach_graph
    from src.graph.events import build_inputs
    from src.models.schemas import UserProfileInput

    deps = _build_fake_deps()
    graph = build_coach_graph(deps, _sqlite_checkpointer())
    profile = UserProfileInput(height=180, weight=80, training_years=2, goal="增肌",
                               available_equipment=["哑铃", "杠铃"], days_per_week=4,
                               injuries=["膝盖疼"])
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}
    state = graph.invoke(build_inputs(profile, query="我想增肌但膝盖疼",
                                      thread_id=thread_id), config)

    assert "__interrupt__" in state, "expected the run to pause at the review gate"
    review_id = state["review_id"]
    Path(HANDOFF_PATH).write_text(json.dumps({"thread_id": thread_id,
                                              "review_id": review_id}),
                                  encoding="utf-8")
    print(f"[phase 1] 计划被安全门拦截，进入人工审核。")
    print(f"[phase 1] thread_id = {thread_id}")
    print(f"[phase 1] review_id = {review_id}")
    print(f"[phase 1] checkpoint 已写入 {DB_PATH}；进程随即退出（模拟重启）。")
    return 0


def phase2() -> int:
    """In a fresh process, resume the paused run from the same SQLite checkpoint."""
    from langgraph.types import Command
    from src.graph.builder import build_coach_graph

    handoff = json.loads(Path(HANDOFF_PATH).read_text(encoding="utf-8"))
    thread_id, review_id = handoff["thread_id"], handoff["review_id"]

    # Brand-new collaborators; only the SQLite checkpoint is shared across the restart.
    deps = _build_fake_deps()
    graph = build_coach_graph(deps, _sqlite_checkpointer())
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}

    snapshot = graph.get_state(config)
    assert "review_gate" in (snapshot.next or ()), "run should still be paused at review_gate"
    print(f"[phase 2] 新进程读取到暂停的执行，仍停在 review_gate。")

    state = graph.invoke(Command(resume={"decision": "approved", "review_id": review_id,
                                         "reviewer": "教练", "comment": "已评估，可执行"}),
                         config)
    delivery = state.get("final_payload") or {}
    assert delivery.get("delivery_status") == "review_approved", delivery
    print(f"[phase 2] 审核通过，计划交付：delivery_status = {delivery['delivery_status']}")
    print("[phase 2] ✅ checkpoint 跨进程重启恢复成功。")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] == "--phase1":
        return phase1()
    if args and args[0] == "--phase2":
        return phase2()

    # Orchestrator: run each phase in its own process to prove cross-restart recovery.
    for leftover in (DB_PATH, HANDOFF_PATH):
        Path(leftover).unlink(missing_ok=True)

    exe = sys.executable
    print("=" * 68)
    print("阶段 1：生成计划 → 触发人工审核 → 写入 checkpoint → 退出进程")
    print("=" * 68)
    subprocess.run([exe, __file__, "--phase1"], check=True)

    print()
    print("=" * 68)
    print("阶段 2：全新进程 → 读取同一 SQLite → 恢复审核 → 交付计划")
    print("=" * 68)
    subprocess.run([exe, __file__, "--phase2"], check=True)

    print()
    print("演示完成：人工审核闭环 + checkpoint 持久化均工作正常。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
