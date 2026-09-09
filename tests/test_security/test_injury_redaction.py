"""Direct-identifier redaction on the outbound LLM path.

The load-bearing guarantee here is asymmetric, and the tests are organised
around it: identifiers must leave the machine stripped, while medical semantics
must survive byte-for-byte. If semantics were stripped too, FactChecker's LLM
review would lose its basis for judging contraindications and HITL's semantic
matching (15 injury profiles, cosine threshold 0.68) would collapse — i.e.
full masking would trade away the safety property the project cannot concede.
"""

import pytest

import src.config as config
import src.security.injury_redaction as redaction
from src.security.injury_redaction import (
    redact_text, redact_messages, restore_text, summarize, is_enabled,
)


RAW_INJURY_TEXT = (
    "患者：张伟，手机 13812345678，2025年3月18日在北京协和医院做的微创手术，"
    "病历号 A12345678。我是腰椎 L4-L5 间盘突出，现在左腿还有点麻，能做硬拉吗？"
)


# --------------------------------------------------------------------------
# default: off means byte-identical outbound text
# --------------------------------------------------------------------------
def test_disabled_by_default():
    assert config.LLM_INJURY_REDACTION == "off"
    assert is_enabled() is False


def test_messages_pass_through_untouched_when_disabled():
    messages = [{"role": "user", "content": RAW_INJURY_TEXT}]
    out, mapping = redact_messages(messages)
    assert out is messages            # same object, zero work
    assert mapping == {}


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(redaction, "LLM_INJURY_REDACTION", "on")
    assert is_enabled() is True
    return monkeypatch


# --------------------------------------------------------------------------
# identifiers are stripped
# --------------------------------------------------------------------------
@pytest.mark.parametrize("text,label", [
    ("联系 13812345678", "手机号"),
    ("联系 13912345678 和 15887654321", "手机号"),
    ("身份证 110101199003074512", "证件号"),
    ("身份证 11010119900307451X", "证件号"),
    ("卡号 6222021234567890123", "银行卡"),
    ("邮箱 zhangwei@example.com", "邮箱"),
    ("病历号: A12345678", "病历号"),
    ("就诊号 B-9981", "病历号"),
    ("2025年3月18日做的手术", "日期"),
    ("2025-03-18 做的手术", "日期"),
    ("2025/03/18 做的手术", "日期"),
    ("患者：张伟", "姓名"),
    ("病人:李小明 复查", "姓名"),
])
def test_identifier_is_stripped(text, label):
    out, mapping = redact_text(text)
    assert any(label in ph for ph in mapping), mapping
    # the raw identifier no longer appears anywhere in the outbound text
    for original in mapping.values():
        assert original not in out


def test_all_identifiers_stripped_from_realistic_input():
    out, mapping = redact_text(RAW_INJURY_TEXT)
    for secret in ("13812345678", "2025年3月18日", "A12345678", "张伟"):
        assert secret not in out
    assert len(mapping) >= 4


# --------------------------------------------------------------------------
# medical semantics must survive — this is the whole point of not masking
# --------------------------------------------------------------------------
@pytest.mark.parametrize("term", [
    "腰椎", "L4-L5", "间盘突出", "硬拉", "左腿", "麻", "微创手术",
])
def test_medical_semantics_survive_redaction(term):
    out, _ = redact_text(RAW_INJURY_TEXT)
    assert term in out


def test_redacted_text_still_triggers_hitl_keyword_matching():
    """HITL's rule engine does substring matching on the raw query text.

    Redaction lives on the LLM egress path, so the engine keeps seeing the
    original. This test pins the other half of that contract: even the
    *redacted* text still carries every keyword the conflict table needs,
    so masking can never silently disarm the safety layer.
    """
    from src.hitl.review import INJURY_EXERCISE_CONFLICTS

    raw = "患者：张伟，手机13812345678，我膝盖半月板损伤，还能做深蹲吗"
    out, mapping = redact_text(raw)

    assert mapping, "identifiers should have been stripped"
    assert "13812345678" not in out

    # The conflict table keys on short injury tokens ("膝"), not on the full
    # colloquial phrase, so match by substring rather than by an exact key.
    knee_keys = [kw for kw in INJURY_EXERCISE_CONFLICTS if kw in out and "膝" in kw]
    assert knee_keys, f"no knee key survived redaction; matched={list(INJURY_EXERCISE_CONFLICTS)}"
    forbidden = INJURY_EXERCISE_CONFLICTS[knee_keys[0]]
    assert any("深蹲" in exercise for exercise in forbidden)
    # and the user's own mention of the risky movement is still visible too
    assert "深蹲" in out


def test_pure_training_text_is_left_alone():
    """No identifiers → no rewrite at all, mapping stays empty."""
    text = "我想增肌，一周四练，家里有哑铃和杠铃，膝盖偶尔不舒服"
    out, mapping = redact_text(text)
    assert out == text
    assert mapping == {}


# --------------------------------------------------------------------------
# false positives: ordinary narration must not be eaten
# --------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "昨天我去了医院检查，医生说没大问题",      # institution-ish narration
    "我是医生，想给自己安排训练",              # occupation, not an identifier
    "深蹲时膝盖不要内扣，保持与脚尖同向",      # core coaching cue
    "每组 8-12 次，做 4 组，组间休息 90 秒",   # numbers that are not identifiers
    "2025年我开始系统训练",                    # year alone is allowed (HIPAA)
    "上个月膝盖有点疼",                        # relative time is clinically useful
    "训练年限 3 年，身高 175，体重 70",        # body stats are not identifiers
])
def test_ordinary_text_is_not_redacted(text):
    out, mapping = redact_text(text)
    assert out == text, f"unexpected redaction: {mapping}"
    assert mapping == {}


def test_short_digit_runs_are_not_treated_as_cards():
    out, mapping = redact_text("负重 100 公斤，做了 12345 次")
    assert mapping == {}
    assert out == "负重 100 公斤，做了 12345 次"


def test_phone_pattern_requires_exactly_eleven_digits():
    # 12 digits starting with 13 must not be silently truncated into a match
    out, mapping = redact_text("编号 138123456789")
    assert "手机号" not in "".join(mapping)


# --------------------------------------------------------------------------
# placeholders, numbering and restoration
# --------------------------------------------------------------------------
def test_distinct_values_get_distinct_numbered_placeholders():
    out, mapping = redact_text("联系 13812345678 或 15887654321")
    assert "【手机号1】" in out and "【手机号2】" in out
    assert mapping["【手机号1】"] == "13812345678"
    assert mapping["【手机号2】"] == "15887654321"


def test_repeated_value_reuses_one_placeholder():
    out, mapping = redact_text("手机 13812345678，再说一次 13812345678")
    assert out.count("【手机号1】") == 2
    assert "【手机号2】" not in out
    assert len([k for k in mapping if "手机号" in k]) == 1


def test_restore_round_trips_every_identifier():
    out, mapping = redact_text(RAW_INJURY_TEXT)
    restored = restore_text(out, mapping)
    for secret in ("13812345678", "2025年3月18日", "A12345678"):
        assert secret in restored
    assert "【" not in restored


def test_restore_is_noop_without_mapping():
    assert restore_text("原文", {}) == "原文"
    assert restore_text("", {"【手机号1】": "x"}) == ""


def test_restore_leaves_unrelated_text_untouched():
    out = restore_text("训练计划：深蹲 3x10", {"【手机号1】": "138"})
    assert out == "训练计划：深蹲 3x10"


# --------------------------------------------------------------------------
# message-level behaviour
# --------------------------------------------------------------------------
def test_redact_messages_does_not_mutate_input(enabled):
    messages = [
        {"role": "system", "content": "你是健身教练"},
        {"role": "user", "content": RAW_INJURY_TEXT},
    ]
    snapshot = [dict(m) for m in messages]

    out, mapping = redact_messages(messages)

    assert messages == snapshot            # caller's list untouched
    assert out is not messages
    assert out[0]["content"] == "你是健身教练"   # nothing to strip → same dict
    assert "13812345678" not in out[1]["content"]
    assert mapping


def test_redact_messages_handles_non_string_and_missing_content(enabled):
    messages = [
        {"role": "user"},
        {"role": "tool", "content": None},
        {"role": "user", "content": ["multimodal", "parts"]},
        "not-a-dict",
    ]
    out, mapping = redact_messages(messages)
    assert out == messages
    assert mapping == {}


def test_redact_messages_merges_across_messages(enabled):
    messages = [
        {"role": "user", "content": "手机 13812345678"},
        {"role": "assistant", "content": "好的"},
        {"role": "user", "content": "复查日期 2025年3月18日"},
    ]
    out, mapping = redact_messages(messages)
    assert "【手机号1】" in out[0]["content"]
    assert "【日期1】" in out[2]["content"]
    assert set(mapping) == {"【手机号1】", "【日期1】"}


def test_summarize_reports_counts_without_leaking_originals(enabled):
    _, mapping = redact_text(RAW_INJURY_TEXT)
    summary = summarize(mapping)
    assert summary.get("手机号") == 1
    assert summary.get("日期") == 1
    blob = repr(summary)
    for secret in ("13812345678", "张伟", "A12345678"):
        assert secret not in blob


def test_redact_text_tolerates_non_string():
    assert redact_text(None) == (None, {})
    assert redact_text("") == ("", {})


# --------------------------------------------------------------------------
# provider wiring: strip on egress, restore on the non-streaming return
# --------------------------------------------------------------------------
def _provider(monkeypatch, captured, reply_content):
    from src.llm.provider import LLMProvider, LLMResponse

    provider = LLMProvider()
    monkeypatch.setattr(provider, "_resolve", lambda model=None: (None, "default", "m"))
    monkeypatch.setattr(provider, "_build_fallback_chain", lambda alias: ["default"])
    provider._clients = {"default": object()}
    provider._models = {"default": "m"}

    class FakeBreaker:
        def record_success(self, *a, **k):
            pass

        def record_failure(self, *a, **k):
            pass

    provider._breaker = FakeBreaker()

    def fake_call(client, model_name, messages, temperature):
        captured.extend(messages)
        return LLMResponse(content=reply_content, model=model_name, tokens=10)

    monkeypatch.setattr(provider, "_call_api_with_retry", fake_call)
    monkeypatch.setattr("src.llm.cost_tracker.cost_tracker.record", lambda *a, **k: None)
    return provider


def test_chat_strips_identifiers_before_sending(enabled, monkeypatch):
    captured = []
    provider = _provider(monkeypatch, captured, reply_content="建议避免硬拉")

    provider.chat([{"role": "user", "content": RAW_INJURY_TEXT}])

    assert len(captured) == 1
    sent = captured[0]["content"]
    assert "13812345678" not in sent
    assert "张伟" not in sent
    assert "腰椎" in sent and "L4-L5" in sent      # semantics preserved


def test_chat_restores_placeholders_in_the_response(enabled, monkeypatch):
    captured = []
    provider = _provider(
        monkeypatch, captured,
        reply_content="已收到你的联系方式【手机号1】，建议避免硬拉",
    )

    resp = provider.chat([{"role": "user", "content": "手机 13812345678，能练腿吗"}])

    assert "【手机号1】" not in resp.content
    assert "13812345678" in resp.content


def test_chat_is_untouched_when_redaction_disabled(monkeypatch):
    captured = []
    provider = _provider(monkeypatch, captured, reply_content="ok")

    provider.chat([{"role": "user", "content": RAW_INJURY_TEXT}])

    assert captured[0]["content"] == RAW_INJURY_TEXT   # byte-identical
