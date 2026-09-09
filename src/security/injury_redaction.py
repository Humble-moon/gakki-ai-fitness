"""
=============================================================================
injury_redaction.py — 出网文本的直接标识符剥离（可选，默认关闭）
=============================================================================
【要解决的问题】
    用户输入会包含伤病自由文本，例如
    「我是腰椎 L4-L5 间盘突出，2025年3月18日在北京协和医院做的微创，
      患者：张伟，手机 13812345678，现在左腿还有点麻」。
    这段文本会随 prompt 发往外部 LLM API（也会随 embedding 请求发往阿里云）。
    其中真正构成隐私风险的不是「腰椎间盘」这个诊断，而是能把这条记录
    关联到某个具体自然人的**直接标识符**：姓名、手机号、精确就诊日期、
    病历号、证件号。

【为什么不做「全遮蔽成 INJURY_TYPE_BACK_003」】
    这是最直觉但最糟的方案。FactChecker 的 LLM 语义审查要靠「腰椎 / 间盘 /
    突出」这些词才能判断硬拉是否禁忌；HITL 的 _match_semantic 要把伤病描述
    做 embedding、与 15 个伤病档案算余弦相似度（阈值 0.68）。
    一旦把医学语义抹掉换成不透明 ID：
      · LLM 安全审查直接失去判断依据 → 漏报率上升；
      · 语义匹配的向量变成同一个常量 → HITL 语义路整体失效。
    也就是说，全遮蔽会**用隐私收益换掉安全能力**，而安全恰恰是这个项目
    不可让步的部分。所以这里采用「最小必要披露」：剥离直接标识符，
    完整保留医学语义。诊断信息本身不在 HIPAA 的 18 类标识符之列，
    「腰椎 L4-L5 突出」这样的描述在国内有数百万人，单独不构成识别。

【为什么接在 LLMProvider 出口，而不是数据流上游】
    src/hitl/review.py 的 _check_conflicts() 用原始 query_text 做 substring
    匹配、_match_semantic() 用原始文本算相似度——这两条都是**本地**安全机制，
    不出网，因此完全不需要脱敏；一旦在上游脱敏，它们反而会失效。
    把脱敏放在「文本即将离开本机」的最后一跳（LLMProvider），
    既覆盖了所有出网调用，又让本地安全逻辑继续看到完整原文。

【当前实现的边界 — 只做格式高度确定的标识符】
    覆盖：证件号、银行卡号、手机号、邮箱、病历号/就诊号、精确日期、
          带标签前缀的显式姓名（「患者：张伟」）。
    不覆盖：机构名、医师名、详细地址。
    原因是这三类需要真正的命名实体识别。用正则做中文 NER 的误伤率不可接受
    ——例如「昨天我去了医院检查」会被机构模式吃掉，「我是医生」会被医师模式
    吃掉，而这些恰恰是 FactChecker 需要看到的叙述。宁可少剥一类，
    也不把安全审查的输入改坏。要补齐这三类，正确的做法是接入 NER 模型
    或用一次廉价 LLM 调用做识别，而不是加更多正则。

【开关】
    LLM_INJURY_REDACTION=on 才生效；默认 off，出网文本与历史行为完全一致。
    开启会改变送给 LLM 的 prompt，因此属于影响生成质量与评测指标的变更，
    启用后应重跑相关评测再与基线对比。

【流式路径的已知限制】
    chat_stream() 只在发送侧脱敏，不对响应做还原：占位符可能跨 chunk 被切断
    （「【手」+「机号1】」），逐块替换会漏。要支持流式还原需要在出口做
    跨 chunk 缓冲，代价是首 token 延迟——与流式的目的冲突，故不做。
    非流式 chat() 则在返回前完整还原。
=============================================================================
"""

import logging
import re
from typing import Dict, List, Tuple

from src.config import LLM_INJURY_REDACTION

logger = logging.getLogger(__name__)

# 规则顺序敏感：先长且更具体的模式，避免短模式抢先命中长模式的片段。
# 数字类模式一律加 (?<!\d) / (?!\d) 边界：否则 19 位银行卡会被 18 位证件号
# 模式吃掉前 18 位、只留下 1 位残尾，既漏剥了卡号又污染了文本。
_REDACTION_RULES: List[Tuple[str, re.Pattern]] = [
    ("证件号", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
    ("银行卡", re.compile(r"(?<!\d)\d{16,19}(?!\d)")),
    ("手机号", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    ("邮箱", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("病历号", re.compile(
        r"(?:病历号|病案号|就诊号|住院号|门诊号)\s*[:：#]?\s*[A-Za-z0-9][A-Za-z0-9\-]{2,}"
    )),
    ("日期", re.compile(
        r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日"
        r"|\d{4}-\d{1,2}-\d{1,2}"
        r"|\d{4}/\d{1,2}/\d{1,2}"
    )),
    # 只剥离带显式标签的姓名。裸姓名（「张伟说」）无法与常用词可靠区分，
    # 强行匹配会大面积误伤正文。
    ("姓名", re.compile(r"(?:患者|病人|姓名|本人)\s*[:：]\s*[一-龥]{2,4}")),
]


def is_enabled() -> bool:
    return (LLM_INJURY_REDACTION or "off").lower() == "on"


def redact_text(text: str) -> Tuple[str, Dict[str, str]]:
    """剥离文本中的直接标识符。

    返回 (脱敏后文本, {占位符: 原文})。占位符形如「【手机号1】」，带序号，
    因此同一段文本里出现多个不同手机号也能被精确还原；相同原文复用同一编号。

    未命中任何规则时返回原文与空映射（调用方可据此跳过还原步骤）。
    """
    if not text or not isinstance(text, str):
        return text, {}

    mapping: Dict[str, str] = {}
    counters: Dict[str, int] = {}
    result = text

    for label, pattern in _REDACTION_RULES:
        if not pattern.search(result):
            continue

        def _replace(match, _label=label):
            original = match.group(0)
            # 同一原文复用已分配的占位符，避免同一个人被替换成两个编号
            for placeholder, seen in mapping.items():
                if seen == original and placeholder.startswith(f"【{_label}"):
                    return placeholder
            counters[_label] = counters.get(_label, 0) + 1
            placeholder = f"【{_label}{counters[_label]}】"
            mapping[placeholder] = original
            return placeholder

        result = pattern.sub(_replace, result)

    return result, mapping


def redact_messages(messages: List[dict]) -> Tuple[List[dict], Dict[str, str]]:
    """对 OpenAI 格式消息列表脱敏，返回新列表与合并映射。

    不修改入参（返回浅拷贝 + 新 content），避免调用方持有的 messages
    被就地改写——Orchestrator 会在多处复用同一份消息列表。
    开关关闭时原样返回，零开销。
    """
    if not is_enabled() or not messages:
        return messages, {}

    redacted: List[dict] = []
    merged: Dict[str, str] = {}
    for message in messages:
        if not isinstance(message, dict):
            redacted.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content:
            redacted.append(message)
            continue
        new_content, mapping = redact_text(content)
        if not mapping:
            redacted.append(message)
            continue
        copied = dict(message)
        copied["content"] = new_content
        redacted.append(copied)
        for placeholder, original in mapping.items():
            # 不同消息里出现同名占位符但原文不同时，保留第一次的映射；
            # 还原只是体验优化，不追求跨消息的全局一致性。
            merged.setdefault(placeholder, original)

    if merged:
        logger.info(f"出网文本脱敏：替换 {len(merged)} 处直接标识符")
    return redacted, merged


def restore_text(text: str, mapping: Dict[str, str]) -> str:
    """把占位符还原为原文，用于将 LLM 响应呈现给用户。

    还原是纯字符串替换，不会引入任何新信息，因此是安全的。
    mapping 为空或 text 不含占位符时原样返回。
    """
    if not text or not mapping:
        return text
    result = text
    for placeholder, original in mapping.items():
        if placeholder in result:
            result = result.replace(placeholder, original)
    return result


def summarize(mapping: Dict[str, str]) -> Dict[str, int]:
    """按类别统计剥离数量，用于日志与审计（不含任何原文）。"""
    counts: Dict[str, int] = {}
    for placeholder in mapping:
        label = placeholder.strip("【】")
        counts[label.rstrip("0123456789")] = counts.get(label.rstrip("0123456789"), 0) + 1
    return counts
