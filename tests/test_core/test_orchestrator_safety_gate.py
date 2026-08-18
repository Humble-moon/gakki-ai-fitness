from types import SimpleNamespace

from src.core.orchestrator import Orchestrator


class StoreSpy:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
        return record

    def assert_empty(self):
        assert self.calls == []


def make_orch():
    orch = Orchestrator.__new__(Orchestrator)
    orch.cache = StoreSpy()
    orch.conversation = StoreSpy()
    orch.long_term = StoreSpy()
    return orch


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


def test_finalization_is_fail_closed_for_missing_or_invalid_checks():
    orch = make_orch()
    result = orch._finalize_result(valid_plan(), [{}], 0)
    assert result["_persistence_allowed"] is False
    assert result["requires_review"] is True
    assert orch._persist_if_safe({}, "q", result) is False
    orch.cache.assert_empty()
    orch.conversation.assert_empty()
    orch.long_term.assert_empty()


def test_safe_result_persists_session_memory_without_regular_cache():
    orch = make_orch()
    result = orch._finalize_result(valid_plan(), [safe_check()], 0)
    assert orch._persist_if_safe({"goal": "增肌"}, "q", result, "session") is True
    orch.cache.assert_empty()
    assert [name for name, _, _ in orch.conversation.calls] == ["set_plan_state", "add_turn"]
    assert [name for name, _, _ in orch.long_term.calls] == ["save_preference"] * 3


def test_safe_result_persists_regular_cache_and_long_term_memory():
    orch = make_orch()
    result = orch._finalize_result(valid_plan(), [safe_check()], 0)
    assert orch._persist_if_safe({"goal": "增肌"}, "q", result) is True
    assert [name for name, _, _ in orch.cache.calls] == ["set"]
    orch.conversation.assert_empty()
    assert [name for name, _, _ in orch.long_term.calls] == ["save_preference"] * 3


def test_review_required_stream_result_is_not_persisted():
    orch = make_orch()
    result = orch._finalize_result(valid_plan(), [safe_check(requires_human_review=True)], 0)
    assert result["requires_review"] is True
    assert orch._persist_if_safe({}, "q", result, "session") is False
    orch.cache.assert_empty(); orch.conversation.assert_empty(); orch.long_term.assert_empty()


def test_degraded_provider_result_cannot_persist():
    orch = make_orch()
    result = orch._finalize_result(valid_plan(), [safe_check()], 0, provider_degraded=True)
    assert result["_persistence_allowed"] is False
    assert orch._persist_if_safe({}, "q", result) is False
    orch.cache.assert_empty()


def test_cached_result_must_be_safe_and_schema_valid_before_returning():
    assert Orchestrator._safe_cached_result({"days": [], "_persistence_allowed": False}) is None
    unsafe = dict(valid_plan(), _persistence_allowed=True, requires_review=False, warnings=["x"])
    assert Orchestrator._safe_cached_result(unsafe) is None
    safe = dict(valid_plan(), _persistence_allowed=True, requires_review=False, warnings=[])
    assert Orchestrator._safe_cached_result(safe) == safe


def test_schema_invalid_result_is_fail_closed():
    orch = make_orch()
    result = orch._finalize_result({"days": []}, [safe_check()], 0)
    assert result["_persistence_allowed"] is False
    assert orch._persist_if_safe({}, "q", result) is False
    orch.cache.assert_empty(); orch.conversation.assert_empty(); orch.long_term.assert_empty()


def test_degraded_fact_check_including_rewrite_cannot_authorize_persistence():
    orch = make_orch()
    orch.planner = SimpleNamespace(plan=lambda *args, **kwargs: {"skill_config": {}})
    orch.retriever = SimpleNamespace(retrieve=lambda plan: {"exercises": []})
    orch.bus = SimpleNamespace(send=lambda task: None)
    orch.writer = SimpleNamespace(
        write_plan=lambda *args: valid_plan(),
        rewrite_plan=lambda *args: valid_plan(),
    )
    checks = iter([
        safe_check(is_safe=False, issues=[{"issue": "rewrite"}], _degraded=True),
        safe_check(),
    ])
    orch.fact_checker = SimpleNamespace(check=lambda *args: next(checks))
    persisted = []
    orch._persist_if_safe = lambda *args, **kwargs: persisted.append(args[2].get("_persistence_allowed"))
    profile = SimpleNamespace(model_dump=lambda: {"goal": "增肌"}, goal="增肌")

    result = orch.generate_plan(profile, "q")

    assert result["rewrite_count"] == 1
    assert result["_persistence_allowed"] is False
    assert persisted == [False]
