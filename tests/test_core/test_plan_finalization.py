"""Direct unit tests for the extracted plan-finalization pure functions.

Mirrors tests/test_core/test_orchestrator_safety_gate.py but targets the shared
``src.core.plan_finalization`` functions instead of the Orchestrator delegators,
so the LangGraph pipeline's terminal-state layer is covered independently.
"""

from src.core import plan_finalization as pf
from src.hitl.review_store import InMemoryReviewArtifactStore


class StoreSpy:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
        return record

    def assert_empty(self):
        assert self.calls == []


def valid_plan():
    return {
        "plan_id": "plan-1", "user_id": 1, "goal": "增肌", "weeks": 4,
        "sessions_per_week": 1,
        "days": [{"day": 1, "focus": "全身", "exercises": []}],
    }


def safe_check(**overrides):
    check = {"is_safe": True, "issues": [], "requires_human_review": False, "confidence": .9}
    check.update(overrides)
    return check


def test_finalize_is_fail_closed_for_missing_or_invalid_checks():
    result = pf.finalize_result(valid_plan(), [{}], 0)
    assert result["_persistence_allowed"] is False
    assert result["requires_review"] is True
    cache, conversation, long_term = StoreSpy(), StoreSpy(), StoreSpy()
    assert pf.persist_if_safe(cache, conversation, long_term, {}, "q", result) is False
    cache.assert_empty(); conversation.assert_empty(); long_term.assert_empty()


def test_safe_result_persists_session_memory_without_regular_cache():
    result = pf.finalize_result(valid_plan(), [safe_check()], 0)
    cache, conversation, long_term = StoreSpy(), StoreSpy(), StoreSpy()
    assert pf.persist_if_safe(cache, conversation, long_term, {"goal": "增肌"}, "q", result, "session") is True
    cache.assert_empty()
    assert [name for name, _, _ in conversation.calls] == ["set_plan_state", "add_turn"]
    assert [name for name, _, _ in long_term.calls] == ["save_preference"] * 3


def test_safe_result_persists_regular_cache_and_long_term_memory():
    result = pf.finalize_result(valid_plan(), [safe_check()], 0)
    cache, conversation, long_term = StoreSpy(), StoreSpy(), StoreSpy()
    assert pf.persist_if_safe(cache, conversation, long_term, {"goal": "增肌"}, "q", result) is True
    assert [name for name, _, _ in cache.calls] == ["set"]
    conversation.assert_empty()
    assert [name for name, _, _ in long_term.calls] == ["save_preference"] * 3


def test_review_pending_payload_and_artifact_creation():
    store = InMemoryReviewArtifactStore()
    check = safe_check(
        requires_human_review=True,
        issues=[{"issue": "膝盖风险", "severity": "danger"}],
        review_reason="伤病冲突需要人工确认",
        review_severity="danger",
    )
    result = pf.finalize_result(valid_plan(), [check], 0)
    artifact = pf.create_review_artifact(store, {"goal": "增肌", "injuries": ["膝盖疼"]}, "膝盖疼还能深蹲吗", result)
    delivered = pf.build_review_pending_payload(result, artifact)

    assert result["requires_review"] is True
    assert delivered["delivery_status"] == "review_pending"
    assert delivered["review"]["review_id"] == artifact.review_id
    assert delivered["review"]["reason"] == "伤病冲突需要人工确认"
    assert delivered["review"]["prohibited_actions"] == pf.REVIEW_PROHIBITED_ACTIONS
    assert "thread_id" not in delivered  # omitted unless explicitly supplied
    assert store.get(artifact.review_id) is artifact


def test_review_payload_includes_thread_id_when_supplied():
    store = InMemoryReviewArtifactStore()
    result = pf.finalize_result(valid_plan(), [safe_check(requires_human_review=True,
                                                          issues=[{"issue": "x", "severity": "warning"}])], 0)
    artifact = pf.create_review_artifact(store, {"goal": "增肌"}, "q", result)
    delivered = pf.build_review_pending_payload(result, artifact, thread_id="thread-123")
    assert delivered["thread_id"] == "thread-123"


def test_review_artifact_carries_rule_engine_findings_and_merged_severity():
    """规则引擎拦截（LLM issues 已被重写解决）时，审核工件必须携带冲突明细与
    合并后的严重级别，而不是回退到 warning/空清单。"""
    store = InMemoryReviewArtifactStore()
    check = safe_check(
        requires_human_review=True,
        issues=[],
        review_reason="规则引擎检测到 2 个伤病-动作冲突",
        review_severity="danger",
        review_suggestions=[
            "[规则引擎] 伤病「腰椎间盘突出」与动作「杠铃深蹲」存在冲突（触发词: 腰椎），建议人工审核",
            "[规则引擎] 用户查询含伤病「腰椎间盘突出」，且提到高风险动作「硬拉」（触发词: 腰椎），建议人工审核",
        ],
    )
    result = pf.finalize_result(valid_plan(), [check], 2)
    artifact = pf.create_review_artifact(
        store, {"goal": "增肌", "injuries": ["腰椎间盘突出"]}, "想练硬拉", result)
    delivered = pf.build_review_pending_payload(result, artifact)

    assert artifact.severity == "danger"
    assert delivered["review"]["severity"] == "danger"
    assert len(artifact.issues) == 2
    assert all(issue["severity"] == "danger" for issue in artifact.issues)
    assert "杠铃深蹲" in artifact.issues[0]["issue"]


def test_review_artifact_deduplicates_suggestions_matching_active_issues():
    """LLM 阶段升级审核时，suggestions 与 active issues 同源，不得重复计入。"""
    store = InMemoryReviewArtifactStore()
    check = safe_check(
        requires_human_review=True,
        issues=[{"issue": "强度过高", "severity": "warning"}],
        review_reason="存在需要确认的警告项",
        review_severity="warning",
        review_suggestions=["强度过高"],
    )
    result = pf.finalize_result(valid_plan(), [check], 0)
    artifact = pf.create_review_artifact(store, {"goal": "增肌"}, "q", result)
    assert artifact.severity == "warning"
    assert artifact.issues == [{"issue": "强度过高", "severity": "warning"}]


def test_review_pending_result_marks_safe_delivered():
    store = InMemoryReviewArtifactStore()
    delivered = pf.review_pending_result(store, {"goal": "增肌"}, "q",
                                         pf.finalize_result(valid_plan(), [safe_check()], 0))
    assert delivered["delivery_status"] == "safe_delivered"
    assert delivered["days"] == valid_plan()["days"]
    assert "review" not in delivered


def test_degraded_provider_result_cannot_persist():
    result = pf.finalize_result(valid_plan(), [safe_check()], 0, provider_degraded=True)
    assert result["_persistence_allowed"] is False
    cache, conversation, long_term = StoreSpy(), StoreSpy(), StoreSpy()
    assert pf.persist_if_safe(cache, conversation, long_term, {}, "q", result) is False
    cache.assert_empty()


def test_cached_result_must_be_safe_and_schema_valid():
    assert pf.safe_cached_result({"days": [], "_persistence_allowed": False}) is None
    unsafe = dict(valid_plan(), _persistence_allowed=True, requires_review=False, warnings=["x"])
    assert pf.safe_cached_result(unsafe) is None
    safe = dict(valid_plan(), _persistence_allowed=True, requires_review=False, warnings=[])
    assert pf.safe_cached_result(safe) == safe


def test_goal_mismatch_cache_entry_is_a_miss():
    mismatched = dict(valid_plan(), goal="减脂", _persistence_allowed=True,
                      requires_review=False, warnings=[])
    assert pf.safe_cached_result(mismatched, expected_goal="增肌") is None


def test_finalization_adds_goal_issue_and_blocks_persistence():
    result = pf.finalize_result(valid_plan() | {"goal": "减脂"}, [safe_check()], 0,
                                expected_goal="增肌")
    assert result["_persistence_allowed"] is False
    assert result["active_issues"][-1]["code"] == "PLAN_GOAL_MISMATCH"


def test_successful_rewrite_separates_resolved_from_active_issues():
    initial_issue = {"issue": "深蹲对当前伤情不安全", "severity": "warning"}
    result = pf.finalize_result(valid_plan(),
                                [safe_check(is_safe=False, issues=[initial_issue]), safe_check()], 1)
    assert result["active_issues"] == []
    assert result["resolved_issues"] == [initial_issue]
    assert result["warnings"] == []
    assert result["_persistence_allowed"] is True


def test_normalize_plan_uses_deterministic_metadata_only():
    normalized = pf.normalize_plan(
        {"plan_id": "p", "user_id": 1, "goal": "增肌", "days_per_week": 5,
         "days": [{"day": 1, "focus": "全身", "exercises": []}]},
        profile={"days_per_week": 3}, plan_config={"weeks": 8})
    assert normalized["sessions_per_week"] == 3
    assert normalized["weeks"] == 8

    unknown_weeks = pf.normalize_plan({"days_per_week": 3, "days": []},
                                      profile={"days_per_week": 3}, plan_config={})
    assert unknown_weeks["weeks"] is None


def test_make_user_key_is_stable():
    profile = {"height": 180, "weight": 80, "goal": "增肌", "injuries": ["下背痛"]}
    assert pf.make_user_key(profile) == pf.make_user_key(dict(profile))
    assert 0 <= pf.make_user_key(profile) < 1000000


def test_summarize_plan_for_context():
    plan = {"days": [{"day": 1, "focus": "胸", "exercises": [{"name": "卧推"}, {"name": "飞鸟"}]}]}
    assert pf.summarize_plan_for_context(plan) == "第1天(胸): 卧推/飞鸟"
    assert pf.summarize_plan_for_context({"days": []}) == ""
