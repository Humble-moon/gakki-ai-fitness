import pytest
from src.llm.prompts.planner import build_planner_messages
from src.llm.prompts.fact_checker import build_fact_checker_messages
from src.llm.prompts.writer import build_writer_messages

def test_planner_prompt_includes_profile():
    msgs = build_planner_messages("想增肌", {"height": 180, "goal": "增肌"})
    assert "增肌" in msgs[1]["content"]
    assert msgs[0]["role"] == "system"

def test_factchecker_prompt_includes_plan():
    plan = {"days": [{"day": 1, "exercises": []}]}
    msgs = build_fact_checker_messages(plan, {"injuries": ["下背痛"]})
    assert "下背痛" in msgs[1]["content"]


def test_writer_prompt_matches_training_plan_output_metadata():
    msgs = build_writer_messages([], {"days_per_week": 3}, "增肌")
    system_prompt = msgs[0]["content"]

    assert '"weeks"' in system_prompt
    assert '"sessions_per_week"' in system_prompt
    assert '"days_per_week"' not in system_prompt


def test_writer_prompt_requires_canonical_goal():
    messages = build_writer_messages([], {"days_per_week": 3, "goal": "减脂"}, "减脂")

    assert "canonical goal：减脂" in messages[0]["content"]
    assert "不得输出其他 goal" in messages[0]["content"]


def test_planner_prompt_preserves_profile_goal_separately_from_skill():
    messages = build_planner_messages("膝盖疼，想减脂", {"goal": "减脂"})

    assert '"requested_goal"' in messages[0]["content"]
    assert "即使安全路由到动作分析，也必须保留" in messages[0]["content"]


# ----------------------------------------------------------------------
# 运动习惯与跨会话画像
#
# 习惯此前只落进长期记忆、只进「训练建议」提示词，生成计划的 Writer 一个字
# 都没用到——用户填了「跑步、瑜伽」，计划里完全看不出来。下面两条锁住它真的
# 进了生成提示词，而不只是"参数加了"。
# ----------------------------------------------------------------------

def test_writer_prompt_turns_habits_into_an_explicit_constraint():
    messages = build_writer_messages(
        [], {"days_per_week": 4, "habits": ["晨练", "跑步"]}, "增肌"
    )
    content = messages[1]["content"]

    assert "运动习惯" in content
    assert "晨练" in content and "跑步" in content
    # 约束要出现在最前面的约束区，而不是被淹没在末尾的用户画像 dump 里
    assert content.index("运动习惯") < content.index("可用动作")


def test_writer_prompt_omits_habits_section_when_empty():
    """没填习惯时不该多出一行空约束——那是噪声，还会稀释真正的安全约束。"""
    messages = build_writer_messages([], {"days_per_week": 4}, "增肌")

    assert "运动习惯" not in messages[1]["content"]


def test_long_term_memory_reaches_both_planner_and_writer():
    """跨会话画像必须同时进 Planner 和 Writer。

    只读不用的链路此前真实存在过：ingest 把记忆读出来，却只喂给教练开场白，
    规划与生成两个环节都拿不到。
    """
    long_term = "运动习惯：['瑜伽']\n伤病史：['腰突']"

    planner_msgs = build_planner_messages("增肌", {"goal": "增肌"},
                                          long_term_context=long_term)
    writer_msgs = build_writer_messages([], {"days_per_week": 4, "goal": "增肌"},
                                        "增肌", long_term_context=long_term)

    for msgs, who in ((planner_msgs, "Planner"), (writer_msgs, "Writer")):
        content = msgs[1]["content"]
        assert "瑜伽" in content and "腰突" in content, f"{who} 没拿到跨会话记忆"
        # 记忆比本次表单旧，必须交代冲突时以表单为准，否则会拿旧数据覆盖新提交
        assert "以本次表单为准" in content, f"{who} 缺少新旧冲突的裁决规则"
