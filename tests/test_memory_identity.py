"""长期记忆身份链路的回归测试。

「系统记不住用户」是三个独立缺陷叠加的结果，其中两个长期隐形：

1. ``_answer_question_fixed_stream`` 结尾写长期记忆时引用了未定义的
   ``pseudo_uid``——每条问答流都在收尾处抛 ``NameError``，用户拿不到
   ``done`` 事件，记忆也从未落库；
2. 前端 ``generate-plan`` 不传 ``athlete_key``，身份退化成身高体重的 MD5
   指纹，用户改一次体重就再也读不回自己的偏好。

两者都不会让既有测试变红，因为此前**没有任何测试把真实问答流 drain 到结束**
——``tests/test_qa_dispatch.py`` 全都把该方法替换成假实现。本文件补上这个
缺口：跑真实生成器，并断言身份键。
"""

from __future__ import annotations

import fnmatch
import json
from types import SimpleNamespace

from src.core import plan_finalization as pf
from src.core.orchestrator import Orchestrator
from src.memory.long_term import LongTermMemory
from src.models.schemas import UserProfileInput


# ============================================================
# 测试替身
# ============================================================
class StoreSpy:
    """记录调用的万能替身（同 tests/test_core/test_plan_finalization.py）。"""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
        return record


class LongTermSpy:
    """长期记忆替身：分开记录读侧与写侧用到的身份键。"""

    def __init__(self):
        self.writes = []         # [(user_key, pref_key)]
        self.context_calls = []  # 读侧传入的身份键

    def build_context_for_prompt(self, user_id):
        self.context_calls.append(user_id)
        return ""

    def save_preference(self, user_id, key, value):
        self.writes.append((user_id, key))


class FakeRedis:
    """最小的 Redis 替身，够 LongTermMemory 的读取路径用。"""

    def __init__(self, store=None):
        self.store = dict(store or {})
        self.conn = self  # get_preferences 走 self.redis.conn.scan_iter

    def scan_iter(self, match=None, count=20):
        return [k for k in list(self.store) if fnmatch.fnmatch(k, match)]

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, ex=None):
        self.store[key] = value

    def delete(self, key):
        self.store.pop(key, None)


def _lt_with(store):
    """用给定 Redis 内容构造一个绕开真实连接的 LongTermMemory。"""
    lt = LongTermMemory.__new__(LongTermMemory)
    lt.prefix = "memory:user:"
    lt.redis = FakeRedis(store)
    return lt


def _pref(value):
    return json.dumps({"v": value, "ts": "2026-10-01T00:00:00Z"})


def profile(**overrides):
    base = dict(height=175, weight=70, training_years=2, goal="增肌",
                available_equipment=["哑铃"], days_per_week=3)
    base.update(overrides)
    return UserProfileInput(**base)


def make_question_orch(long_term):
    """构造一个能真实跑完整问答流的 Orchestrator（不触网、不碰数据库）。"""
    orch = Orchestrator.__new__(Orchestrator)
    orch.writer = SimpleNamespace(llm=SimpleNamespace(
        chat_stream=lambda messages, temperature=0: iter(["建议", "规律训练"])))
    orch.knowledge = SimpleNamespace(search_with_fallback=lambda q: [])
    orch.retriever = SimpleNamespace(
        tools=SimpleNamespace(call=lambda *a, **k: {}),
        retrieve=lambda ctx: {})
    orch.long_term = long_term
    orch.documents = SimpleNamespace(
        search=lambda *a, **k: [], get_documents_for_session=lambda sid: [])
    orch.conversation = SimpleNamespace(
        add_turn=lambda *a, **k: None,
        build_context_for_prompt=lambda *a, **k: "")
    return orch


def valid_plan():
    return {"plan_id": "plan-1", "user_id": 1, "goal": "增肌", "weeks": 4,
            "sessions_per_week": 1,
            "days": [{"day": 1, "focus": "全身", "exercises": []}]}


def safe_check(**overrides):
    check = {"is_safe": True, "issues": [], "requires_human_review": False,
             "confidence": .9}
    check.update(overrides)
    return check


# ============================================================
# 1. 问答流必须能走完，且写在 athlete_key 下
# ============================================================
def test_question_stream_reaches_done_and_persists_under_athlete_key():
    """真实 drain 问答流：必须走到 done，且记忆写在 athlete_key 下。

    这是当前唯一能钉住 ``pseudo_uid`` 的测试。修复前该生成器在 ``yield done``
    之前抛 NameError——用户收到半截回答后流异常中断，记忆从未落库。
    """
    long_term = LongTermSpy()
    orch = make_question_orch(long_term)

    events = list(orch._answer_question_fixed_stream(
        "增肌期怎么安排训练量", profile(injuries=["腰椎间盘突出"]),
        athlete_key="athlete-abc"))

    assert events[-1][0] == "done", (
        "问答流必须走到 done；中断说明收尾处又抛异常了")
    assert long_term.writes, "长期记忆必须落盘（修复前问答路径从未写入过）"
    assert {uid for uid, _ in long_term.writes} == {"athlete-abc"}, (
        "身份必须是传入的 athlete_key，而不是身高体重指纹")
    assert {k for _, k in long_term.writes} >= {"profile", "goal"}


def test_question_path_reads_and_writes_the_same_key():
    """同一次问答里，读记忆与写记忆必须用同一个身份键。

    读用一个键、写用另一个，是「记不住」的经典形态：用户刚说过偏好，
    下一轮就读不回来。
    """
    long_term = LongTermSpy()
    orch = make_question_orch(long_term)

    list(orch._answer_question_fixed_stream(
        "增肌期怎么安排训练量", profile(), athlete_key="athlete-xyz"))

    assert long_term.context_calls == ["athlete-xyz"]
    assert {uid for uid, _ in long_term.writes} == {"athlete-xyz"}


# ============================================================
# 2. 两条路径必须写到同一个身份下
# ============================================================
def test_plan_and_question_paths_persist_under_the_same_key():
    """生成计划与知识问答必须写到同一个身份下，否则两条路径各记各的。"""
    long_term = LongTermSpy()
    result = pf.finalize_result(valid_plan(), [safe_check()], 0)
    pf.persist_if_safe(StoreSpy(), StoreSpy(), long_term,
                       profile().model_dump(), "q", result,
                       athlete_key="athlete-shared")

    orch = make_question_orch(long_term)
    list(orch._answer_question_fixed_stream(
        "训练量怎么安排", profile(), athlete_key="athlete-shared"))

    assert {uid for uid, _ in long_term.writes} == {"athlete-shared"}


# ============================================================
# 3. 身份不随体重漂移
# ============================================================
def test_identity_does_not_change_when_weight_changes():
    """改体重不能换身份——这正是「统一长期记忆身份」要解决的问题。"""
    key_before = pf.long_term_user_key(profile(weight=80).model_dump(), "athlete-stable")
    key_after = pf.long_term_user_key(profile(weight=72).model_dump(), "athlete-stable")
    assert key_before == key_after == "athlete-stable"

    # 没有 athlete_key 时才退化成指纹（游客的兜底路径），此时确实随体重变。
    # 这条断言是反向证据：说明「为什么必须有 athlete_key」而不是靠指纹。
    assert pf.long_term_user_key(profile(weight=80).model_dump()) != \
           pf.long_term_user_key(profile(weight=72).model_dump())


# ============================================================
# 4. 伤病：死读路径的修复
# ============================================================
def test_injury_history_falls_back_to_profile_snapshot():
    """pref:injuries 从未被写过时，伤病史必须能从画像快照读回来。

    修复前 get_injury_history 只读 pref:injuries，而全项目没有任何地方写过
    这个键——一条死读路径，伤病永远返回空列表。
    """
    lt = _lt_with({
        "memory:user:athlete-1:pref:profile":
            _pref({"goal": "增肌", "injuries": ["腰椎间盘突出"]}),
    })
    assert lt.get_injury_history("athlete-1") == ["腰椎间盘突出"]


def test_explicit_injury_key_still_wins_over_snapshot():
    """显式写了 pref:injuries 时以它为准（兼容旧数据），不被画像快照覆盖。"""
    lt = _lt_with({
        "memory:user:athlete-1:pref:injuries": _pref(["肩袖损伤"]),
        "memory:user:athlete-1:pref:profile": _pref({"injuries": ["腰椎间盘突出"]}),
    })
    assert lt.get_injury_history("athlete-1") == ["肩袖损伤"]


def test_missing_injuries_returns_empty_list():
    """没有任何伤病数据时返回空列表，而不是报错或返回 None。"""
    lt = _lt_with({})
    assert lt.get_injury_history("athlete-none") == []


# ============================================================
# 5. 运动习惯/爱好
# ============================================================
def test_habits_are_persisted_and_injected_into_context():
    """运动习惯要能落盘，并出现在注入提示词的上下文里。"""
    spy = LongTermSpy()
    pf.persist_long_term_profile(spy, "athlete-1",
                                 profile(habits=["晨练", "跑步"]).model_dump())
    assert ("athlete-1", "habits") in spy.writes

    lt = _lt_with({"memory:user:athlete-1:pref:habits": _pref(["晨练", "跑步"])})
    ctx = lt.build_context_for_prompt("athlete-1")
    assert "运动习惯" in ctx
    assert "晨练" in ctx


def test_empty_habits_do_not_overwrite_existing_ones():
    """画像里没有 habits 时不能把它清空。

    否则用户在问答里随手提交一次画像，就会抹掉先前填过的运动习惯。
    """
    spy = LongTermSpy()
    pf.persist_long_term_profile(spy, "athlete-1", {"goal": "增肌"})
    assert ("athlete-1", "habits") not in spy.writes


# ============================================================
# 6. 提示词里同一事实只出现一次
# ============================================================
def test_context_does_not_repeat_injuries():
    """伤病在注入上下文里只应出现一次。

    profile 快照本身含 injuries，若再把整个 profile dump 一遍，
    模型会收到同一事实的两份拷贝——既浪费上下文，也容易被误读为强调。
    """
    lt = _lt_with({
        "memory:user:athlete-1:pref:profile":
            _pref({"goal": "增肌", "injuries": ["腰椎间盘突出"]}),
    })
    ctx = lt.build_context_for_prompt("athlete-1")
    assert ctx.count("腰椎间盘突出") == 1
    assert "伤病史" in ctx
