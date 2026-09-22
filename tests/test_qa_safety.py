"""问答安全模块的直测。

这些用例此前并不存在——安全逻辑内联在 `answer_question_stream` 里，
只被端到端用例间接覆盖。抽取到 `src/core/qa_safety.py` 时补上，
使得"归一化行为与抽取前一致"这件事有回归保护：如果将来有人动了
零宽字符表或正则顺序，这里会立刻失败。
"""

from __future__ import annotations

import pytest

from src.core.qa_safety import (
    SAFETY_KEYWORDS,
    SAFETY_NOTE,
    SAFETY_SYSTEM_MESSAGE,
    build_safety_messages,
    detect_safety_concern,
    normalize_question,
)


class TestNormalizeQuestion:
    """归一化必须拆掉常见的绕过写法。"""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            # CJK 字符间空格
            ("硬 拉", "硬拉"),
            ("膝 盖 疼", "膝盖疼"),
            # 零宽字符（逐字验证六种）
            ("硬​拉", "硬拉"),
            ("硬‌拉", "硬拉"),
            ("硬‍拉", "硬拉"),
            ("硬﻿拉", "硬拉"),
            ("硬­拉", "硬拉"),
            ("硬⁠拉", "硬拉"),
            # 括号注音
            ("(xi)盖疼", "盖疼"),
            ("膝盖(teng)疼", "膝盖疼"),
            # emoji
            ("膝盖😊疼", "膝盖疼"),
            ("深蹲💪", "深蹲"),
        ],
    )
    def test_strips_evasion_forms(self, raw, expected):
        assert normalize_question(raw) == expected

    def test_leaves_plain_text_untouched(self):
        assert normalize_question("增肌应该怎么吃") == "增肌应该怎么吃"

    def test_non_cjk_spaces_are_preserved(self):
        """只该去掉 CJK 之间的空格，英文句子的空格要留着。"""
        assert normalize_question("bench press") == "bench press"

    def test_empty_input(self):
        assert normalize_question("") == ""

    def test_combined_evasion(self):
        """多种绕过手法叠加时仍能还原。"""
        assert normalize_question("硬​ 拉😊") == "硬拉"


class TestDetectSafetyConcern:
    def test_keyword_hit(self):
        assert detect_safety_concern("膝盖疼怎么办") is True

    @pytest.mark.parametrize("kw", ["疼", "痛", "骨折", "撕裂", "脱臼", "麻"])
    def test_individual_keywords(self, kw):
        assert detect_safety_concern(f"我{kw}了") is True

    def test_no_keyword_no_injury(self):
        assert detect_safety_concern("增肌怎么吃") is False

    def test_existing_injury_always_triggers(self):
        """即使当前问题与伤病无关，有伤病史也必须走安全话术。"""
        assert detect_safety_concern("今天练什么", injuries=["腰椎间盘突出"]) is True

    def test_empty_injury_list_does_not_trigger(self):
        assert detect_safety_concern("今天练什么", injuries=[]) is False

    def test_evasion_is_caught_after_normalisation(self):
        """绕过写法不该逃过检测——这是归一化存在的全部意义。"""
        assert detect_safety_concern("膝​盖 疼") is True

    def test_all_documented_keywords_are_effective(self):
        """词表里每个词都应真的能触发。防止加了词却写错。"""
        for kw in SAFETY_KEYWORDS:
            assert detect_safety_concern(f"有点{kw}") is True, f"{kw} 未触发"


class TestSemanticLayer:
    def test_matcher_hit_triggers(self):
        assert detect_safety_concern(
            "膝盖咔咔响", semantic_matcher=lambda q: [{"profile_id": "knee"}]
        ) is True

    def test_matcher_miss_does_not_trigger(self):
        assert detect_safety_concern(
            "增肌怎么吃", semantic_matcher=lambda q: None
        ) is False

    def test_matcher_returning_empty_list_does_not_trigger(self):
        assert detect_safety_concern("增肌", semantic_matcher=lambda q: []) is False

    def test_matcher_exception_is_swallowed(self):
        """embedding 服务挂掉不该阻塞问答——关键词层已提供基础保护。"""
        def boom(_):
            raise RuntimeError("embedding 服务不可用")

        assert detect_safety_concern("膝盖咔咔响", semantic_matcher=boom) is False

    def test_keyword_hit_skips_semantic_layer(self):
        """关键词已命中时不该再调 embedding——省一次网络往返。"""
        called = []

        def spy(q):
            called.append(q)
            return None

        assert detect_safety_concern("膝盖疼", semantic_matcher=spy) is True
        assert called == []


class TestSafetyMessages:
    def test_system_message_is_emitted(self):
        msgs = build_safety_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == SAFETY_SYSTEM_MESSAGE

    def test_system_message_resists_roleplay(self):
        """硬约束必须显式覆盖角色扮演类绕过。"""
        assert "假装" in SAFETY_SYSTEM_MESSAGE
        assert "忽略" in SAFETY_SYSTEM_MESSAGE

    def test_system_message_forbids_diagnosis(self):
        assert "医疗诊断" in SAFETY_SYSTEM_MESSAGE

    def test_note_requires_stopping_and_referral(self):
        assert "停止训练" in SAFETY_NOTE
        assert "咨询医生" in SAFETY_NOTE

    def test_note_disclaims_medical_authority(self):
        assert "不能替代专业医疗诊断" in SAFETY_NOTE


class TestExtractionParity:
    """抽取后的行为必须与内联版本逐字一致。

    这段是"重构没改变行为"的守卫：把原 orchestrator 里的实现复制过来
    对照，防止将来有人"顺手优化"掉某个替换步骤。
    """

    @staticmethod
    def _original_inline(question: str) -> str:
        import re as _re
        n = question
        n = n.replace("​", "").replace("‌", "")
        n = n.replace("‍", "").replace("﻿", "")
        n = n.replace("­", "").replace("⁠", "")
        n = _re.sub(r"\([a-zA-Z1-4]+\)", "", n)
        n = _re.sub(
            r"(?<=[一-鿿㐀-䶿])\s+(?=[一-鿿㐀-䶿])",
            "", n,
        )
        n = _re.sub(r"[\U0001F300-\U0001FFFF]", "", n)
        return n

    @pytest.mark.parametrize(
        "sample",
        [
            "膝盖疼怎么办",
            "硬 拉",
            "硬​拉",
            "(xi)盖疼",
            "膝盖😊疼",
            "增肌怎么吃",
            "　全角空格　测试",
            "bench press 100kg",
        ],
    )
    def test_matches_original(self, sample):
        assert normalize_question(sample) == self._original_inline(sample)
