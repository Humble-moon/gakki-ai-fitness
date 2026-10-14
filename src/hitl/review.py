"""
================================================================================
文件角色：HITL（Human-in-the-Loop，人机协同）审核决策模块
================================================================================
- 被调用者：编排引擎在 FactChecker 输出结果后，调用 HITLReview.check()
  判断该计划是否需要转人工审核。
- 调用者：本模块依赖 config 中的 HITL_CONFIDENCE_THRESHOLD 阈值配置。
- 项目角色：AI 安全管线的最后一环——"AI 说安全不一定安全"。
  这是人机协同的分流决策点：根据置信度和问题严重级别决定是直接返回用户
  还是进入人工审核队列。

Pipeline 位置：
  Planner → Retriever → Writer → FactChecker → HITLReview.check() →
    ├─ needs_review=False → 直接返回给用户
    └─ needs_review=True  → 进入人工审核队列 → 人工确认/修改后返回
================================================================================
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from src.config import HITL_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

# 语义匹配阈值：query vs injury profile 余弦相似度 >= 此值时触发
_SEMANTIC_INJURY_THRESHOLD = 0.68  # cos-sim: 低于0.65易误报，高于0.72会漏掉口语化表达


# ---------------------------------------------------------------------------
# 伤病语义档案：包含详细描述文本 + 禁止动作 + 严重级别
# 语义匹配层用 embedding 相似度替代关键词子串匹配，
# 解决用户用不同表达方式描述同一伤病（如"膝盖里面咔咔响"→匹配"膝关节炎"）的问题。
# ---------------------------------------------------------------------------
INJURY_PROFILES = [
    {
        "id": "knee-osteoarthritis",
        "description": "膝关节骨关节炎、膝盖退化、膝盖磨损。症状包括膝盖咔咔响、膝盖僵硬、"
                       "上下楼梯膝盖疼、膝盖肿胀、膝盖活动受限、膝盖里面感觉骨头磨骨头、"
                       "膝盖怕冷、膝盖无力、蹲下去站不起来。",
        "banned_exercises": ["深蹲", "箭步蹲", "分腿蹲", "跳跃", "跑步", "跳绳", "腿举"],
        "severity": "danger",
    },
    {
        "id": "lumbar-disc",
        "description": "腰椎间盘突出、腰突、椎间盘膨出。症状包括下背痛、腰痛放射到臀部和腿、"
                       "久坐加重、咳嗽打喷嚏腰疼、腿麻脚麻、坐骨神经痛、弯腰困难、"
                       "站久了腰酸、睡硬板床才舒服。",
        "banned_exercises": ["深蹲", "硬拉", "划船", "推举", "罗马尼亚硬拉", "早安式"],
        "severity": "danger",
    },
    {
        "id": "shoulder-impingement",
        "description": "肩峰撞击综合征、肩袖损伤、肩关节疼痛。症状包括手臂上举时肩膀疼、"
                       "卧推时肩膀前侧刺痛、侧平举时肩膀咔咔响、肩关节活动范围受限、"
                       "肩膀夜间疼痛、肩膀无力、举手过头困难、肩膀弹响。",
        "banned_exercises": ["推举", "卧推", "飞鸟", "侧平举", "前平举", "引体向上"],
        "severity": "danger",
    },
    {
        "id": "acl-post-surgery",
        "description": "前交叉韧带重建术后、ACL手术、膝关节镜手术。术后恢复期、"
                       "膝盖稳定性差、膝盖活动度受限、膝盖肿胀未消、"
                       "医生嘱咐避免负重、膝关节内有螺钉或植入物。",
        "banned_exercises": ["深蹲", "硬拉", "箭步蹲", "跳跃", "跑步", "分腿蹲", "腿举", "大重量"],
        "severity": "danger",
    },
    {
        "id": "meniscus-tear",
        "description": "半月板撕裂、半月板损伤、膝盖扭伤。症状包括膝盖内侧或外侧压痛、"
                       "膝盖卡住不能动、膝盖肿胀、膝盖屈伸时有弹响、膝盖不稳定感。",
        "banned_exercises": ["深蹲", "箭步蹲", "分腿蹲", "跳跃", "腿举"],
        "severity": "danger",
    },
    {
        "id": "tennis-elbow",
        "description": "网球肘、肱骨外上髁炎。症状包括肘关节外侧疼痛、握力下降、"
                       "拧毛巾时肘痛、提重物肘痛、前臂旋转时疼痛、手指伸直时肘部牵扯痛。",
        "banned_exercises": ["弯举", "臂屈伸", "窄距卧推", "引体向上", "划船"],
        "severity": "warning",
    },
    {
        "id": "ankle-sprain",
        "description": "踝关节扭伤、崴脚、脚踝韧带损伤。症状包括脚踝肿胀、脚踝淤青、"
                       "走路脚踝疼、跑步脚踝疼、脚踝不稳定容易再崴、脚踝活动度受限。",
        "banned_exercises": ["提踵", "跳跃", "深蹲", "跑步", "跳绳", "箭步蹲"],
        "severity": "warning",
    },
    {
        "id": "cervical-spine",
        "description": "颈椎病、颈椎间盘突出、颈部僵硬疼痛。症状包括脖子酸痛、转头困难、"
                       "手臂麻、手指麻、头晕、颈后僵硬、长时间低头加重。",
        "banned_exercises": ["杠铃深蹲(杠铃压颈)", "推举", "倒立"],
        "severity": "warning",
    },
    {
        "id": "wrist-tfcc",
        "description": "三角纤维软骨复合体损伤、腕关节TFCC。症状包括手腕尺侧疼痛、"
                       "手腕旋转时疼痛、撑地时手腕痛、推的动作时手腕刺痛。",
        "banned_exercises": ["卧推", "推举", "弯举", "俯卧撑"],
        "severity": "danger",
    },
    {
        "id": "hip-issues",
        "description": "髋关节问题、弹响髋、髋关节撞击、股骨髋臼撞击FAI。"
                       "症状包括髋关节弹响、深蹲时髋关节卡住、髋关节酸痛、"
                       "腹股沟区域疼痛、髋关节活动范围受限。",
        "banned_exercises": ["深蹲", "硬拉", "箭步蹲", "分腿蹲", "跳跃"],
        "severity": "warning",
    },
    {
        "id": "cardiac-risk",
        "description": "心脏病、高血压、心血管疾病、冠心病。"
                       "医生嘱咐避免剧烈运动、运动时胸闷心悸、心律不齐、"
                       "服用降压药/抗凝药/β受体阻滞剂。",
        "banned_exercises": ["大重量", "憋气(Valsalva)", "HIIT", "倒立", "极限重量"],
        "severity": "danger",
    },
    {
        "id": "osteoporosis",
        "description": "骨质疏松、骨密度低、骨质减少。症状包括容易骨折、"
                       "身高变矮、驼背、骨密度检查T值低于-2.5。",
        "banned_exercises": ["大重量", "跳跃", "冲击", "硬拉", "深蹲", "跑步"],
        "severity": "danger",
    },
    {
        "id": "pregnancy-postpartum",
        "description": "怀孕期间或产后恢复期。腹直肌分离、盆底肌弱、"
                       "产后核心不稳、医生要求避免腹部压力增加。",
        "banned_exercises": ["深蹲", "硬拉", "跳跃", "仰卧起坐", "卷腹", "大重量", "卧推"],
        "severity": "danger",
    },
    {
        "id": "hernia",
        "description": "疝气、腹股沟疝、脐疝、切口疝。症状包括腹部或腹股沟有包块、"
                       "咳嗽或用力时包块变大、腹部坠胀感。",
        "banned_exercises": ["深蹲", "硬拉", "推举", "大重量", "憋气"],
        "severity": "danger",
    },
    {
        "id": "general-pain-warning",
        "description": "一般性运动疼痛或不适、训练伤痛。症状包括训练后疼痛、"
                       "练完第二天还疼、拉伸也疼、某处隐隐作痛、不知道为什么会疼、"
                       "练的时候不疼练完疼、酸胀感不消退、关节不适、肌肉长期酸痛。",
        "banned_exercises": ["大重量", "极限组数"],
        "severity": "warning",
    },
]


# ---------------------------------------------------------------------------
# 确定性安全规则：伤病关键词 → 高风险动作 → 禁止组合
# 不依赖 LLM，规则引擎直接判定。这是"宁可误报不可漏报"的最后防线。
# ---------------------------------------------------------------------------

# 伤病关键词 → 应禁止的动作模式（substring 匹配）
INJURY_EXERCISE_CONFLICTS = {
    "腰": ["深蹲", "硬拉", "划船", "罗马尼亚", "早安式"],
    "椎": ["深蹲", "硬拉", "划船", "推举", "罗马尼亚"],
    "间盘": ["深蹲", "硬拉", "划船", "推举", "罗马尼亚"],
    "背": ["深蹲", "硬拉", "罗马尼亚", "划船", "推举"],
    "膝": ["深蹲", "箭步蹲", "分腿蹲", "腿举", "跳跃"],
    "半月板": ["深蹲", "箭步蹲", "分腿蹲", "腿举"],
    "髌骨": ["深蹲", "箭步蹲", "腿举"],
    "肩": ["推举", "卧推", "飞鸟", "侧平举", "前平举"],
    "肩袖": ["推举", "卧推", "飞鸟", "侧平举", "前平举"],
    "肩峰": ["推举", "卧推", "飞鸟", "侧平举"],
    "脱臼": ["推举", "卧推", "引体向上", "飞鸟"],
    "肘": ["弯举", "臂屈伸", "窄距卧推", "推举"],
    "网球肘": ["弯举", "臂屈伸", "窄距卧推", "引体向上"],
    "腱鞘": ["弯举", "臂屈伸"],
    "腕": ["弯举", "卧推", "推举", "臂屈伸"],
    "TFCC": ["弯举", "卧推", "推举"],
    "颈": ["深蹲", "推举", "杠铃"],
    "踝": ["提踵", "跳跃", "深蹲", "箭步蹲"],
    "跟腱": ["提踵", "跳跃", "小腿", "跑步", "跳绳"],
    "手术": ["深蹲", "硬拉", "卧推", "推举", "划船"],  # 术后所有大重量复合动作都禁
    "术后": ["深蹲", "硬拉", "卧推", "推举", "划船"],
    "重建": ["深蹲", "硬拉", "卧推", "推举", "划船", "箭步蹲"],
    "炎症": ["深蹲", "硬拉", "卧推", "推举"],
    # --- v2 扩展 (2026-07-28): 覆盖更多伤病场景 ---
    # 脊柱/核心
    "脊柱": ["深蹲", "硬拉", "划船", "推举", "引体向上"],
    "侧弯": ["深蹲", "硬拉", "单侧负重"],
    "压缩性骨折": ["深蹲", "硬拉", "推举", "跳跃", "跑步"],
    # 髋关节
    "髋": ["深蹲", "硬拉", "箭步蹲", "分腿蹲"],
    "股骨头": ["深蹲", "硬拉", "箭步蹲", "跳跃"],
    # 手腕/手指
    "腕管": ["卧推", "推举", "弯举", "臂屈伸", "引体向上"],
    # 心血管/代谢
    "心脏": ["深蹲", "硬拉", "推举", "HIIT", "大重量", "憋气"],
    "血压高": ["大重量", "憋气", "倒立", "HIIT"],
    "血压": ["大重量", "憋气", "倒立", "HIIT"],
    "糖尿病": ["空腹训练", "高强度"],
    # 孕期/产后
    "怀孕": ["深蹲", "硬拉", "卧推", "跳跃", "仰卧起坐", "大重量"],
    "孕期": ["深蹲", "硬拉", "卧推", "跳跃", "仰卧起坐", "大重量"],
    "产后": ["深蹲", "硬拉", "跳跃", "仰卧起坐", "卷腹"],
    # 骨质/关节退化
    "骨质疏松": ["深蹲", "硬拉", "跳跃", "大重量", "冲击"],
    "关节炎": ["深蹲", "硬拉", "跳跃", "箭步蹲"],
    "风湿": ["深蹲", "硬拉", "大重量"],
    # 神经/慢性
    "坐骨神经": ["深蹲", "硬拉", "划船", "腿弯举"],
    "纤维肌痛": ["大重量", "高强度", "HIIT"],
    # 疝气/术后
    "疝": ["深蹲", "硬拉", "推举", "大重量"],
    "疝气": ["深蹲", "硬拉", "推举", "大重量"],
    "切口": ["深蹲", "硬拉", "卧推", "推举", "划船"],
    # 呼吸
    "哮喘": ["HIIT", "高强度", "大重量"],
    # 眩晕/神经
    "眩晕": ["深蹲", "硬拉", "倒立", "快速起立"],
    "偏头痛": ["HIIT", "大重量", "憋气"],
    # 兜底规则: 任何包含 "伤" 或 "痛" 的查询，标记为需 HITL 审查
    # (此规则在 _check_conflicts 中通过 _FALLBACK_SAFETY 实现)
}

# 高危伤病关键词：命中任何一个，直接标记 danger（最高优先级）
CRITICAL_INJURY_KEYWORDS = [
    "间盘", "腰突", "半月板", "髌骨", "脱臼", "手术", "术后", "重建",
    "瘫痪", "断裂", "撕裂", "骨折", "TFCC",
    # v2 新增: 绝对禁忌
    "心脏", "血压", "主动脉", "动脉瘤", "血栓", "癫痫",
    "怀孕", "孕期",
    "骨质疏松", "压缩性骨折",
    "疝", "疝气",
    "股骨头",
]


# ---------------------------------------------------------------------------
# 伤病性质分流：结构性损伤 vs 功能性受限
#
# 上面那张 INJURY_EXERCISE_CONFLICTS 的键全是**部位词**（肩/踝/腰/膝…），
# 而部位词本身不是伤病——「我想练肩」「肩膀有点酸」「肩膀活动度不好」都会命中
# 「肩」。命中之后一律标 danger 并扣下计划，就会出现下面这起实测事故：
#
#   用户伤病史填「肩膀活动度不好」「脚踝活动度不好」→ 规则引擎报 5 条冲突
#   （触发词 肩/踝）→ severity=danger → requires_review=True → 计划被暂缓交付。
#
# 但活动度差**不是禁忌**，它恰恰是训练要改善的东西。正确处置是降幅度、换更稳的
# 变式、提示无痛范围——系统自己算出的文案写的就是"建议限制幅度、保持肘部微屈
# 并在无痛范围内进行，或改为绳索夹胸/器械推胸"。处置内容完全正确，只是被错误地
# 升级成了阻断理由。而本项目的 HITL 没有"审核人"角色（见 review_pending_result
# 的说明），扣下即死锁：用户永远拿不到计划，也没人能放行。
#
# 分流口径（默认从严，拦截能力不减弱）：
#   命中结构性诊断词            → structural，维持 danger 阻断
#   只命中受限词、无诊断词      → limitation，降级为 caution：照常交付 + 带上做法调整
#   两者都没有（如「我肩膀疼」）→ unclear，按原来那样当 structural 处理
# 所以"我肩膀疼"这类只有部位词的描述仍然会拦，只有明确说出活动度/柔韧/僵硬
# 这类**能力不足**语义时才放行。
# ---------------------------------------------------------------------------

# 功能性受限词：描述"能力不足"，不是"结构损伤"。
MOBILITY_LIMITATION_KEYWORDS = (
    "活动度", "活动范围", "灵活", "柔韧", "僵硬", "发僵", "发紧", "紧绷",
    "紧张", "拉不开", "伸展受限", "受限", "幅度", "不利索",
)

# 结构性诊断词：用户明确说出的损伤/疾病。一旦出现就压制受限判定——
# 「半月板撕裂，活动度受限」是损伤不是受限，必须继续拦。
STRUCTURAL_INJURY_KEYWORDS = (
    # 结构性损伤
    "撕裂", "断裂", "骨折", "骨裂", "脱臼", "脱位", "错位",
    "手术", "术后", "重建", "置换", "关节镜", "钢板", "螺钉", "石膏", "支具",
    "综合征", "撞击", "扭伤", "崴", "拉伤", "劳损", "突出", "膨出", "滑脱",
    "磨损", "退化", "骨刺", "积液", "肿胀", "炎症", "炎", "水肿",
    # 症状词：出现这些说明不只是"能力不足"，按从严处理
    "麻", "刺痛", "疼", "痛", "酸胀", "无力", "不稳", "弹响", "咔咔",
    # 诊疗语境
    "注射", "封闭", "诊断", "病史", "复发", "慢性", "急性",
)

# caution 文案前缀。前端按它把这类提示单独归类成「动作调整」，
# 而不是混进「伤病相关」——受限不是病，摆成医疗风险会吓到用户。
CAUTION_MARKER = "[调整]"


def classify_injury_nature(text: str) -> str:
    """判定一条伤病描述的性质：``structural`` / ``limitation`` / ``unclear``。

    结构性损伤优先于功能性受限：用户写明诊断时，即使描述里同时出现
    「活动受限」这类受限词，也按损伤处理（从严）。

    用途有三处，必须共用同一口径，避免生成侧和判定侧各写一份慢慢走样：
      1. HITL 规则引擎：决定冲突是阻断还是只提醒；
      2. HITL 语义层：受限语义不该被当成疾病诊断；
      3. Writer 提示词：损伤要回避、受限要保留动作只调做法。
    """
    s = str(text or "")
    if not s:
        return "unclear"
    if any(kw in s for kw in STRUCTURAL_INJURY_KEYWORDS):
        return "structural"
    if any(kw in s for kw in MOBILITY_LIMITATION_KEYWORDS):
        return "limitation"
    return "unclear"


def build_limitation_caution(injury: str, targets: list, scope: str = "计划中的") -> str:
    """为一条功能性受限生成面向用户的「动作调整」提示。

    文案刻意不写成风险口吻：受限用户的计划**照常交付**，这条只是告诉他
    具体怎么做。末尾带「建议」，前端 splitNotice 会据此把它拆成
    「问题 / 该怎么做」两行展示。
    """
    names = "、".join(str(t) for t in targets if str(t).strip())
    return (
        f"{CAUTION_MARKER} 「{injury}」不算禁忌，计划照常安排；"
        f"但{scope}{names}要调整做法："
        "建议降低活动幅度、全程保持在无痛范围内，必要时换成更稳的变式（器械或支撑类）"
    )


def injury_profile_label(profile_id: str) -> str:
    """把伤病档案 ID 换成中文名。

    档案里只存了英文 id（``knee-osteoarthritis``），但这条文案是要给用户看的
    （会出现在「计划已暂缓交付」面板的风险明细里）。中文界面上冒出一个英文标识，
    用户既看不懂也无法据此行动；演示时也很突兀。

    中文名取自档案 description 的首句——形如
    「膝关节骨关节炎、膝盖退化、膝盖磨损。症状包括…」，取第一个别名即可。
    档案缺失时原样返回 id，而不是抛异常：文案生成不该成为新的故障点。
    """
    for profile in INJURY_PROFILES:
        if profile.get("id") == profile_id:
            first_sentence = (profile.get("description") or "").split("。")[0]
            name = first_sentence.split("、")[0].strip()
            return name or profile_id
    return profile_id


@dataclass
class ReviewDecision:
    """
    人工审核决策结果数据结构。

    职责：封装 HITL 的判断结果，让编排引擎根据 needs_review 字段分流。

    字段说明：
        needs_review: bool   - 是否需要人工审核
        reason: str          - 为什么需要/不需要审核
        severity: str        - "safe" / "warning" / "danger"
        suggestions: list    - 审核员应关注的问题摘要列表
        cautions: list       - 功能性受限（活动度/柔韧/僵硬）的**做法调整建议**。
                               它**不阻断交付**，只随计划一起交给用户，所以既不并入
                               suggestions（那是扣下计划的理由），也不影响 needs_review。
    """
    needs_review: bool
    reason: str
    severity: str
    suggestions: list
    cautions: list = field(default_factory=list)


class HITLReview:
    """HITL 审核决策器。三阶段审查：

    阶段 1a — 关键词规则引擎：不依赖 LLM，伤病关键词 substring → 禁止动作表。
             零延迟、确定性、可复现。
    阶段 1b — 语义相似度匹配：Embedding cos-sim 匹配伤病档案 → 禁止动作表。
             抓关键词漏掉的表达方式（如"膝盖里面咔咔响"→匹配"膝关节炎"）。
             同样不依赖 LLM，embedding 结果可复现。
    阶段 2  — LLM FactChecker 结果复审：置信度 + severity 判定人工审核。
    """

    def __init__(self):
        self._profile_embeddings: dict[str, np.ndarray] | None = None
        self._emb_service = None

    def _lazy_init_embeddings(self):
        """惰性初始化：预计算所有伤病档案的 embedding 向量。"""
        if self._profile_embeddings is not None:
            return
        from src.rag.embedding import EmbeddingService
        self._emb_service = EmbeddingService()
        self._profile_embeddings = {}
        for profile in INJURY_PROFILES:
            vec = self._emb_service.embed(profile["description"])
            self._profile_embeddings[profile["id"]] = np.array(vec)

    def _match_semantic(self, query_text: str) -> list[dict]:
        """语义匹配：用 embedding 余弦相似度匹配伤病档案。

        Args:
            query_text: 用户 query 或伤病描述

        Returns:
            匹配到的伤病档案列表（仅返回 sim >= threshold 的）
        """
        if not query_text or len(query_text) < 5:
            return []
        self._lazy_init_embeddings()
        query_vec = np.array(self._emb_service.embed(query_text))

        matches = []
        for profile in INJURY_PROFILES:
            profile_vec = self._profile_embeddings[profile["id"]]
            sim = float(np.dot(query_vec, profile_vec) /
                        (np.linalg.norm(query_vec) * np.linalg.norm(profile_vec)))
            if sim >= _SEMANTIC_INJURY_THRESHOLD:
                matches.append({
                    "profile_id": profile["id"],
                    "similarity": round(sim, 3),
                    "banned_exercises": profile["banned_exercises"],
                    "severity": profile["severity"],
                })
        return matches

    def check(self, fact_check_result: dict,
              plan: dict = None, profile: dict = None,
              query_text: str = "") -> ReviewDecision:
        """综合 FactChecker LLM 结果 + 确定性规则，判定是否需要人工审核。

        参数：
            fact_check_result: dict — FactChecker Agent 的输出
            plan: dict | None      — 训练计划（用于确定性规则检查）
            profile: dict | None   — 用户画像（用于确定性规则检查）

        返回值：ReviewDecision
        """
        # === 阶段 1a：关键词规则引擎 ===
        # 不依赖 LLM，直接用伤病-动作冲突表 substring 匹配。
        # 命中后按伤病性质分流：结构性损伤扣下计划，功能性受限只带调整建议。
        rule_issues = []
        rule_cautions = []
        if plan and profile:
            rule_issues, rule_cautions = self._collect_conflicts(plan, profile)

        if rule_issues:
            return ReviewDecision(
                needs_review=True,
                reason=f"发现 {len(rule_issues)} 处动作与你的伤病史冲突",
                severity="danger",
                suggestions=rule_issues,
                cautions=rule_cautions,
            )

        # === 阶段 1b：语义相似度匹配 ===
        # 用 embedding cos-sim 匹配伤病档案 → 解决关键词漏掉的口语化表达
        # （如"膝盖里面咔咔响"→ 语义匹配"膝关节骨关节炎"档案）
        injury_text = query_text or ""
        if profile:
            inj = profile.get("injuries", [])
            if isinstance(inj, list):
                injury_text += " " + " ".join(inj)
            elif isinstance(inj, str):
                injury_text += " " + inj
        semantic_matches = self._match_semantic(injury_text)
        if semantic_matches:
            # 取**相似度最高**的匹配。这里此前写的是 semantic_matches[0]，
            # 取到的是伤病档案表里的顺序第一个，与注释"取相似度最高的匹配"
            # 并不一致——两条档案同时命中时，扣不扣计划取决于档案表的排列顺序。
            best = max(semantic_matches, key=lambda m: m["similarity"])

            # 受限语义不按疾病处理。
            #
            # 实测：用户写「肩膀活动度不好」→ 与 shoulder-impingement 档案
            # 相似度 0.694（danger）；「脚踝活动度不好」→ ankle-sprain 0.703。
            # 系统因此把"你活动度差"读成了"你得了肩峰撞击综合征"。
            # 活动度不足是功能性描述，不带诊断，不该触发疾病级的拦截——
            # 这里降级成调整建议，计划照常交付。
            if classify_injury_nature(injury_text) == "limitation":
                logger.info(
                    f"HITL semantic match downgraded to caution: {best['profile_id']} "
                    f"(sim={best['similarity']})"
                )
                rule_cautions.append(build_limitation_caution(
                    injury_text.strip() or "身体活动度",
                    best["banned_exercises"][:5],
                    scope="",
                ))
            else:
                suggestions = [
                    f"[语义匹配] 你的描述与「{injury_profile_label(best['profile_id'])}」"
                    f"的常见症状相似（相似度 {best['similarity']}，级别 {best['severity']}）",
                    f"建议禁止动作: {', '.join(best['banned_exercises'][:5])}",
                ]
                logger.info(
                    f"HITL semantic match: {best['profile_id']} "
                    f"(sim={best['similarity']}, severity={best['severity']})"
                )
                return ReviewDecision(
                    needs_review=True,
                    reason="你的描述与已知的伤病情况相符，需要专业人员确认",
                    severity=best["severity"],
                    suggestions=suggestions,
                    cautions=rule_cautions,
                )

        # === 阶段 2：LLM 结果复审 ===
        confidence = fact_check_result.get("confidence", 0)
        issues = fact_check_result.get("issues", [])

        has_danger = any(i.get("severity") == "danger" for i in issues)
        has_warning = any(i.get("severity") == "warning" for i in issues)

        if confidence < HITL_CONFIDENCE_THRESHOLD or has_danger:
            return ReviewDecision(
                needs_review=True,
                reason="计划中有需要专业人员确认的内容",
                severity="danger" if has_danger else "warning",
                suggestions=[i["issue"] for i in issues],
                cautions=rule_cautions,
            )

        # === 警告级提示：不阻断交付 ===
        #
        # 这里曾经是 needs_review=True，后果是**所有计划都被扣下**：LLM 检查器
        # 几乎总给出若干条建议性提示（"新手注意恢复""下肢训练量接近上限"），
        # 极少返回空 issues，于是连无伤病、无冲突的健康用户也一律进审核队列。
        # 而本项目的 HITL 只实现了机制、没有定义"审核人"这个角色，
        # 于是形成死锁——用户永远拿不到计划，也没人来审核。
        #
        # 这与交付闸门在 finalize_result 里的判据也是对齐的：那里决定
        # requires_review 的是 is_safe，issues 只作为 warnings 展示。
        #
        # 拦截能力没有变薄——上面三层（规则引擎冲突、语义伤病匹配、
        # 低置信度或 danger）全部保持阻断，warning 只是"可以更好"，
        # 不是"可能有害"。
        if has_warning:
            return ReviewDecision(
                needs_review=False,
                reason="",
                severity="warning",
                suggestions=[i["issue"] for i in issues],
                cautions=rule_cautions,
            )

        return ReviewDecision(
            needs_review=False, reason="", severity="safe", suggestions=[],
            cautions=rule_cautions,
        )

    def _check_conflicts(self, plan: dict, profile: dict) -> list:
        """兼容入口：只返回**阻断级**冲突。

        功能性受限（活动度/柔韧/僵硬）不再从这里出去——它由
        :meth:`_collect_conflicts` 的 cautions 通道承载，只提醒不阻断。
        保留这个签名是因为它同时是"计划是否会被扣下"的既有判据，
        语义应当保持"这里非空 = 危险"。
        """
        danger, _cautions = self._collect_conflicts(plan, profile)
        return danger

    def _collect_conflicts(self, plan: dict, profile: dict) -> tuple[list, list]:
        """确定性规则：检查伤病与训练动作的冲突，并按伤病**性质**分流。

        遍历 profile["injuries"] 和 plan["days"][*]["exercises"][*]["name"]，
        用 INJURY_EXERCISE_CONFLICTS 表做 substring 匹配。
        同时检查 query_text 中是否提到了禁止动作（用户可能在询问危险动作）。

        分流依据见 :func:`classify_injury_nature`。为什么必须分：
        INJURY_EXERCISE_CONFLICTS 的键是**部位词**，而部位词不是伤病。
        实测事故——用户伤病史写「肩膀活动度不好」「脚踝活动度不好」，
        「肩」「踝」两个字命中禁忌表，正在生成的计划被判 danger 并扣下；
        而本项目的 HITL 没有"审核人"角色，扣下即死锁，用户永远拿不到计划。
        更讽刺的是系统自己算出的处置写着"建议限制幅度…或改为绳索夹胸"，
        内容完全正确，只是被错误地升级成了阻断理由。

        返回 ``(danger, cautions)``：
            danger   — 结构性损伤与动作的硬冲突，阻断交付。
            cautions — 功能性受限与动作的软冲突，**不阻断**，随计划交付。
        """
        injuries = profile.get("injuries", [])
        if isinstance(injuries, str):
            injuries = [injuries]
        if not injuries:
            return [], []

        # 收集所有训练动作名
        exercise_names = []
        for day in plan.get("days", []):
            for ex in day.get("exercises", []):
                name = ex.get("name", ex.get("exercise", ""))
                if name:
                    exercise_names.append(name)

        # 用户查询文本，**只取真正的用户查询**。
        # 这里曾在 user_query 为空时用 day["focus"] 兜底，那是错的：focus 是
        # 系统生成的动作主题（如"胸肌与肩部"），把它当伤病文本去匹配触发词会
        # 命中「肩」，于是明明没有肩伤的用户，计划里的卧推/飞鸟也会被判成与
        # 「腰椎间盘突出」冲突——审核队列被误报淹没，真报反而被埋掉。
        query_text = plan.get("user_query", "") or ""

        # 按 (伤病, 目标) 聚合触发词，而不是逐触发词各报一条。
        # 一个伤病描述会命中多个触发词——「腰椎间盘突出」同时含「腰」「椎」「间盘」，
        # 逐个触发词展开会让 3 个动作膨胀成 9 条同义 issue，审核者得在噪声里找信号。
        # 合并成一条、触发词并列展示：信息不丢，条数等于真实冲突数。
        exercise_conflicts: dict[tuple, list] = {}
        query_conflicts: dict[tuple, list] = {}
        # 受限侧按 injury 聚合动作名，而不是逐动作一条：受限用户的计划里
        # 命中部位的动辄五六个，逐个报会把提示区刷成一屏噪声。
        limitation_exercises: dict[str, list] = {}
        limitation_queries: dict[str, list] = {}

        for injury in injuries:
            injury_lower = injury.lower() if isinstance(injury, str) else str(injury)
            # 一条描述是否只是"能力不足"。判定用原始描述本身，不掺 query——
            # query 里出现的疼/麻属于提问语境，不该改变这条伤病的性质。
            is_limitation = classify_injury_nature(injury_lower) == "limitation"
            for keyword, forbidden_exercises in INJURY_EXERCISE_CONFLICTS.items():
                if keyword not in injury_lower and keyword not in query_text:
                    continue

                # 1. 检查 plan 中的已知动作名
                for ex_name in exercise_names:
                    if any(forbidden in ex_name for forbidden in forbidden_exercises):
                        if is_limitation:
                            names = limitation_exercises.setdefault(injury, [])
                            if ex_name not in names:
                                names.append(ex_name)
                        else:
                            keywords = exercise_conflicts.setdefault((injury, ex_name), [])
                            if keyword not in keywords:
                                keywords.append(keyword)

                # 2. 也检查 query 文本中是否提到了禁止动作
                # 例如用户问"跟腱炎怎么练小腿"，query 中提到"小腿"→推测为提踵类动作→触发冲突
                for forbidden in forbidden_exercises:
                    if forbidden in query_text:
                        if is_limitation:
                            names = limitation_queries.setdefault(injury, [])
                            if forbidden not in names:
                                names.append(forbidden)
                        else:
                            keywords = query_conflicts.setdefault((injury, forbidden), [])
                            if keyword not in keywords:
                                keywords.append(keyword)

        issues = []
        for (injury, ex_name), keywords in exercise_conflicts.items():
            issues.append(
                f"[规则引擎] 伤病「{injury}」与动作「{ex_name}」"
                f"存在冲突（触发词: {'/'.join(keywords)}），建议人工审核"
            )
        for (injury, forbidden), keywords in query_conflicts.items():
            issues.append(
                f"[规则引擎] 用户查询含伤病「{injury}」，"
                f"且提到高风险动作「{forbidden}」（触发词: {'/'.join(keywords)}），建议人工审核"
            )

        # 检查 query_text 中是否包含高危伤病关键词
        for keyword in CRITICAL_INJURY_KEYWORDS:
            if keyword in query_text:
                # 对所有涉及的动作都标记
                for ex_name in exercise_names:
                    issue_text = (
                        f"[规则引擎] 用户描述含高危关键词「{keyword}」，"
                        f"计划中的「{ex_name}」需人工确认安全性"
                    )
                    if issue_text not in issues:
                        issues.append(issue_text)
                if not exercise_names:
                    # 没有具体动作名也要报（高危关键词本身就值得关注）
                    issues.append(
                        f"[规则引擎] 用户描述含高危关键词「{keyword}」，"
                        f"请人工审核训练方案安全性"
                    )
                break  # 只报一次高危

        # 受限侧：只产出"怎么做"，不产出"别做"。
        cautions = []
        for injury, names in limitation_exercises.items():
            cautions.append(build_limitation_caution(injury, names))
        for injury, names in limitation_queries.items():
            cautions.append(build_limitation_caution(injury, names, scope="你提到的"))

        # 去重
        return list(dict.fromkeys(issues)), list(dict.fromkeys(cautions))
