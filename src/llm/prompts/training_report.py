"""
================================================================================
文件角色：训练周报（Training Report）Prompt 模板
================================================================================
- 被调用者：``src/core/training_advice.py`` 的 ``_build_report()``。
- 调用者：``build_training_report_messages()`` 构造 messages，经 LLMProvider
  以 JSON 模式发送。失败时调用方回退到 ``_summarize()`` 的确定性模板。
- 项目角色：增强层而非必需层。项目里已经有一条完整的确定性链路
  （``compute_stats`` → ``rule_engine`` → ``_summarize``），它不依赖 LLM、
  模型全挂也能产出可读结论。这个 Prompt 只负责把那条链路的结论**换个说法**，
  让用户更容易读懂。

命名注意：本模块与 ``prompts/advice.py`` 无关。后者是**计划生成前的教练
暖场**（"你好呀，先说说你的情况"），在 orchestrator 与状态图两条编排路径里
共用。两者都叫 advice 只是历史命名撞车，不要合并。

【核心约束：LLM 不产生任何数字】
所有数值由 ``compute_stats()`` 确定性算好后注入下方事实清单，模型只做组织
与解释。周报里出现的每个数字都必须能在事实清单里找到原文——健身场景里错误
的数字会误导训练，这条约束是硬线，不是风格偏好。
================================================================================
"""

# ---------------------------------------------------------------------------
# 输出的 JSON 结构（供调用方与前端参照）
#
#   headline:      一句话概括本周期（不得含事实清单之外的数字）
#   highlights:    进步亮点，每项 {"text": 结论, "evidence": 支撑该结论的原始数字}
#   concerns:      需要注意的问题，结构同上；没有问题时给空数组
#   next_week:     下周可执行的具体建议（3 条以内，动词开头）
#
# evidence 字段是这套设计的关键：它要求模型把「结论」和「依据」分开写，
# 用户与前端都能拿 evidence 回到图表核对，模型胡说会立刻暴露。
# ---------------------------------------------------------------------------
TRAINING_REPORT_SYSTEM = """你是一位健身教练，负责把学员的训练数据总结成一份简短的周报。

【最重要的规则】你只能使用下方「训练事实」里给出的数字。
- 禁止计算新的数字，禁止估算，禁止推断未给出的数值。
- 引用数字时必须与事实清单完全一致（包括单位与正负号）。
- 事实清单里没有的信息，就不要提；不要为了内容完整而编造。

输出要求：
1. 语气像教练对学员说话，专业但不啰嗦，不用 markdown 标记。
2. highlights 只在确有进步时给；concerns 只在确有问题时给。
   两项都可以是空数组——数据平稳时如实说平稳，不要硬凑。
3. next_week 给 1-3 条，每条是一个可执行动作（如"深蹲维持当前重量再加一次"），
   不要写"继续加油"这类没有信息量的句子。
4. 数据量太少（如只有 1 次训练）时，headline 要如实说明结论仅供参考。

输出 JSON 格式（严格遵守键名与嵌套结构）：
{{
  "headline": "一句话概括",
  "highlights": [
    {{"text": "结论", "evidence": "事实清单里支撑它的原始数字"}}
  ],
  "concerns": [
    {{"text": "结论", "evidence": "事实清单里支撑它的原始数字"}}
  ],
  "next_week": ["具体建议"]
}}
"""


def _render_facts(stats: dict, items: list[dict], weeks_shown: int = 4,
                  exercises_shown: int = 5) -> str:
    """把聚合统计渲染成给模型看的纯文本事实清单。

    只注入聚合结果，不注入原始日志——token 不随训练次数增长，且模型
    看不到能自行组合出新数字的原材料。

    参数：
        stats: compute_stats() 的输出（totals / weekly / exercises）
        items: rule_engine() 已推导出的调整建议，作为既定结论一并给出
        weeks_shown: 逐周明细最多展示几周（取最近的）
        exercises_shown: 动作明细最多展示几个（按训练次数降序）
    """
    lines: list[str] = []

    totals = stats.get("totals") or {}
    lines.append("【总体】")
    lines.append(f"- 统计区间内训练 {totals.get('sessions', 0)} 次")
    volume = totals.get("volume")
    if volume:
        lines.append(f"- 总训练容量 {volume} kg")
    completion = totals.get("avg_completion")
    if completion:
        lines.append(f"- 平均完成率 {round(completion * 100, 1)}%")
    avg_rpe = totals.get("avg_rpe")
    if avg_rpe is not None:
        lines.append(f"- 平均 RPE {avg_rpe}")

    weekly = stats.get("weekly") or []
    if weekly:
        lines.append("")
        lines.append("【逐周】")
        for w in weekly[-weeks_shown:]:
            parts = [f"{w.get('week_start')} 起：{w.get('sessions', 0)} 次"]
            if w.get("volume"):
                parts.append(f"容量 {round(w['volume'], 2)} kg")
            if w.get("completion") is not None:
                parts.append(f"完成率 {round(w['completion'] * 100, 1)}%")
            if w.get("avg_rpe") is not None:
                parts.append(f"RPE {w['avg_rpe']}")
            lines.append("- " + "，".join(parts))

    exercises = stats.get("exercises") or []
    if exercises:
        lines.append("")
        lines.append("【动作】")
        for ex in exercises[:exercises_shown]:
            parts = [f"练了 {ex.get('sessions', 0)} 次"]
            if ex.get("first_weight") is not None and ex.get("last_weight") is not None:
                parts.append(f"重量 {ex['first_weight']} → {ex['last_weight']} kg")
            if ex.get("trend_pct") is not None:
                parts.append(f"估算 1RM 趋势 {ex['trend_pct']}%")
            if ex.get("completion") is not None:
                parts.append(f"完成率 {round(ex['completion'] * 100, 1)}%")
            if ex.get("recent_rpe") is not None:
                parts.append(f"近期 RPE {ex['recent_rpe']}")
            lines.append(f"- {ex.get('name')}：" + "，".join(parts))

    if items:
        lines.append("")
        lines.append("【规则引擎已推导出的调整建议（直接沿用，不要另起炉灶）】")
        labels = {"deload": "建议减载", "volume_cut": "建议减量",
                  "progress": "可以加重", "swap": "建议替换"}
        for item in items:
            kind = labels.get(item.get("type", ""), item.get("type", ""))
            reason = item.get("reason") or ""
            lines.append(f"- {kind}｜{item.get('exercise_name', '')}：{reason}")

    if not stats.get("totals", {}).get("sessions"):
        lines.append("")
        lines.append("【注意】当前没有任何训练记录，请如实说明还没有足够数据可供分析。")

    return "\n".join(lines)


def build_training_report_messages(stats: dict, items: list[dict],
                                   profile_context: str = "") -> list:
    """构造训练周报的消息列表。

    参数：
        stats: dict            - compute_stats() 的聚合结果，是周报唯一的数字来源。
        items: list            - rule_engine() 已推导的调整建议，作为既定结论注入，
                                 避免模型与规则引擎给出互相矛盾的说法。
        profile_context: str   - 长期记忆渲染出的用户画像（目标/伤病等），可选。
                                 用于让口吻贴合用户处境，不参与数字生成。

    返回值：
        list                   - OpenAI 格式的 messages 列表
    """
    facts = _render_facts(stats, items)

    user_msg = f"【训练事实】\n{facts}"
    if profile_context:
        user_msg += f"\n\n【学员情况】\n{profile_context}"
    user_msg += "\n\n请根据以上事实生成周报 JSON。再次提醒：只能引用上面出现过的数字。"

    return [
        {"role": "system", "content": TRAINING_REPORT_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
