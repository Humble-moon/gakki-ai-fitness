"""解释块接入流水线的测试（伪造 Agent，不调 LLM，不连数据库）。"""

from types import SimpleNamespace

from src.core.orchestrator import Orchestrator
from src.hitl.review_store import InMemoryReviewArtifactStore
from src.models.schemas import UserProfileInput


class StoreSpy:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
        return record


def _plan():
    return {
        "plan_id": "plan-1", "user_id": 1, "goal": "增肌", "weeks": 4,
        "sessions_per_week": 1,
        "days": [{"day": 1, "focus": "全身", "exercises": []}],
    }


def _check(is_safe=True, issues=None, needs_review=False, confidence=0.9):
    return {"is_safe": is_safe, "issues": issues or [],
            "requires_human_review": needs_review, "confidence": confidence}


def _orchestrator(checks):
    """构造只跑得动 generate_plan 的最小编排器。checks 按调用顺序弹出。"""
    orch = Orchestrator.__new__(Orchestrator)
    queue = list(checks)

    def check(*args, **kwargs):
        return queue.pop(0) if queue else _check()

    orch.planner = SimpleNamespace(plan=lambda *a, **k: {
        "skill": "muscle_building",
        "subtasks": ["胸部训练", "背部训练"],
        "skill_config": {},
    })
    orch.retriever = SimpleNamespace(retrieve=lambda p: {"exercises": [
        {"name": "杠铃卧推", "muscles": ["胸大肌", "三头肌"]},
        {"name": "高位下拉", "muscles": ["背阔肌"]},
        {"name": "哑铃侧平举", "muscles": ["三角肌"], "source": "mcp"},
    ], "knowledge": []})
    orch.bus = SimpleNamespace(send=lambda t: None)
    orch.writer = SimpleNamespace(
        write_plan=lambda *a, **k: _plan(),
        rewrite_plan=lambda *a, **k: _plan(),
    )
    orch.fact_checker = SimpleNamespace(check=check)
    orch.cache = StoreSpy()
    orch.conversation = StoreSpy()
    orch.long_term = StoreSpy()
    orch.review_store = InMemoryReviewArtifactStore()
    return orch


def _profile():
    return UserProfileInput(height=178, weight=80, training_years=2,
                            goal="增肌", available_equipment=["杠铃"],
                            days_per_week=4, injuries=[])


# ----------------------------------------------------------------------
# 解释块确实被接上了
# ----------------------------------------------------------------------

def test_delivered_plan_carries_explanation():
    orch = _orchestrator([_check()])
    result = orch.generate_plan(_profile(), "增肌")

    explain = result.get("explain")
    assert isinstance(explain, dict)
    assert explain["skill"]["label"] == "增肌计划"
    assert explain["skill"]["subtasks"] == ["胸部训练", "背部训练"]


def test_explanation_reports_retrieval_breakdown():
    orch = _orchestrator([_check()])
    explain = orch.generate_plan(_profile(), "增肌")["explain"]

    assert explain["retrieval"]["total"] == 3
    assert explain["retrieval"]["semantic"] == 2
    assert explain["retrieval"]["structured"] == 1
    assert "胸大肌" in explain["retrieval"]["muscles"]


def test_explanation_records_safety_rounds_and_confidence():
    orch = _orchestrator([_check()])
    safety = orch.generate_plan(_profile(), "增肌")["explain"]["safety"]

    assert safety["rounds"] == 1
    assert safety["confidence"] == 0.9
    assert safety["degraded"] is False


def test_explanation_surfaces_issues_that_rewriting_fixed():
    """首轮有问题、重写后干净——被修掉的那条要出现在解释里。"""
    orch = _orchestrator([
        _check(is_safe=False, issues=[{"issue": "深蹲角度与膝盖伤病冲突"}], confidence=0.5),
        _check(),
    ])
    explain = orch.generate_plan(_profile(), "增肌")["explain"]

    assert explain["safety"]["rounds"] == 2
    assert explain["safety"]["resolved"] == ["深蹲角度与膝盖伤病冲突"]
    assert explain["safety"]["active"] == []


# ----------------------------------------------------------------------
# 重写修好的问题不应阻塞交付
# ----------------------------------------------------------------------

def test_plan_that_passed_after_rewrite_is_delivered():
    """首轮有问题、重写后完全通过的计划必须能正常交付。

    这曾是一个真实缺陷：finalize_result 用 any(... for c in checks) 扫描全部
    历史检查，于是首轮的那些问题会永远留着，把最终明明安全的计划扣成人工审核。
    由于首轮草稿几乎总会被找出问题，正常交付路径实际上走不到。
    """
    orch = _orchestrator([
        _check(is_safe=False, issues=[{"issue": "首轮问题"}], needs_review=True, confidence=0.5),
        _check(),   # 重写后完全通过
    ])
    result = orch.generate_plan(_profile(), "增肌")

    assert result["requires_review"] is False
    assert result["_persistence_allowed"] is True
    assert result["delivery_status"] == "safe_delivered"
    # 但被修掉的问题不能就此消失——它要出现在解释里，让用户知道系统检查过
    assert result["explain"]["safety"]["resolved"] == ["首轮问题"]
    assert result["explain"]["safety"]["rounds"] == 2


def test_unresolved_problem_in_final_round_still_blocks_delivery():
    """反面：最终轮仍有问题的计划依然被拦下——修复没有削弱拦截能力。

    注意要喂满 4 条检查：重写回路最多调用 1 次初始检查 + 3 次重写后的复检
    （MAX_RETRIES=3）。给少了会在队列耗尽后回退成安全默认值，测不出想要的场景。
    """
    unsafe = [_check(is_safe=False, issues=[{"issue": "仍未解决的问题"}],
                     needs_review=True, confidence=0.6) for _ in range(4)]
    orch = _orchestrator(unsafe)
    result = orch.generate_plan(_profile(), "增肌")

    # 走审核路径返回的是审核载荷，计划本身被扣下不返回
    assert result["delivery_status"] == "review_pending"
    assert result["requires_review"] is True
    assert "review" in result
    assert "未解决的问题" in result["review"]["reason"] or result["review"]["issues"]


def test_clean_first_draft_is_delivered_normally():
    """对照组：首轮就干净的计划能正常交付——说明上面那条不是路径问题。"""
    orch = _orchestrator([_check()])
    result = orch.generate_plan(_profile(), "增肌")

    assert result["requires_review"] is False
    assert result["delivery_status"] == "safe_delivered"
