"""教练开场白（建议）的提示词与上下文拼装。

这段逻辑原先内联在 ``Orchestrator._build_advice_prompt`` 里，只有命令式编排
用得上。状态图这一侧此前没有对应能力——于是两条编排路径产生的事件流不一致：
命令式会先来一段"你好呀，先说说你的情况……"的暖场，状态图直接进规划。

把它抽出来共享，是为了让两种编排在**同一份提示词**下发声。若各存一份，
两边会各自漂移，而用户看到的只是"这个版本话变少了"，很难察觉是漂移。
"""

from __future__ import annotations


def build_advice_prompt(profile: dict, query: str) -> str:
    """构建教练开场白的 LLM 提示词。

    输入：
        profile: dict — 用户画像字典
        query: str — 用户补充描述
    输出：
        str — 引导模型以教练口吻做初步分析（BMI、训练阶段、伤病关注、
              训练原则）的提示词，用于生成前的暖场。
    """
    equipment = "、".join(profile.get("available_equipment", []))
    injuries_raw = profile.get("injuries", [])
    injuries = injuries_raw[0] if injuries_raw else "无"
    habits_raw = profile.get("habits", [])
    habits = "、".join(habits_raw) if habits_raw else "无"
    return f"""你是一位经验丰富的健身教练。用户刚刚输入了以下信息，请先给一个简短、友好、专业的初步分析。

用户信息：
- 身高 {profile.get('height')}cm，体重 {profile.get('weight')}kg
- 训练 {profile.get('training_years')} 年
- 目标：{profile.get('goal')}
- 训练场景对应的可用器械：{equipment}
- 每周 {profile.get('days_per_week')} 练
- 伤病情况：{injuries}
- 运动习惯：{habits}
- 补充说明：{query or '无'}

要求：
1. 先打招呼，认可用户的基础和目标
2. 点评用户的数据（BMI是否合理、训练年限处于什么阶段、器械是否够用）
3. 如果用户有伤病，认真分析并给出规避建议；如果没有伤病也要提一下注意预防
4. 给 2-3 条针对该场景的实用训练原则
5. 最后用一句"接下来我为你生成具体的训练计划"过渡
6. 总共 150-200 字，用口语化、有温度的语气，像一个真正的教练在对话
7. 用中文回复，不要用 markdown 格式，就是纯文字段落"""


def build_advice_context(long_term_context: str = "",
                         plan_context: str = "",
                         query: str = "") -> str:
    """拼装开场白的附加上下文（跨会话记忆 + 上一版计划）。

    与 ``build_advice_prompt`` 分开，是因为这两段是**条件性**的：
    新用户既没有历史画像也没有旧计划，此时应拼出空串，而不是留下
    「[这个用户之前来过]：」这样一句空标题——那会让模型以为自己漏看了什么。
    """
    parts: list[str] = []
    if long_term_context:
        parts.append(f"\n\n[这个用户之前来过，他的历史画像]：\n{long_term_context}")
    if plan_context:
        parts.append(
            f"\n\n用户之前已经有了一个训练计划：\n{plan_context}\n"
            f"用户现在说：{query}\n请结合这个上下文给出建议。"
        )
    return "".join(parts)


def build_advice_messages(profile: dict, query: str,
                          long_term_context: str = "",
                          plan_context: str = "") -> list[dict]:
    """把提示词与上下文组装成可直接送进 ``chat_stream`` 的消息列表。"""
    content = (build_advice_prompt(profile, query)
               + build_advice_context(long_term_context, plan_context, query))
    return [{"role": "user", "content": content}]
