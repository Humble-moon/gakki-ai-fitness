import streamlit as st
from src.core.orchestrator import Orchestrator
from src.models.schemas import UserProfileInput

st.set_page_config(page_title="AI 健身私教", page_icon="💪", layout="wide")
st.title("💪 AI 健身私教")

# Sidebar: User profile
with st.sidebar:
    st.header("身体数据")
    height = st.number_input("身高 (cm)", 150, 220, 180)
    weight = st.number_input("体重 (kg)", 35, 180, 80)
    years = st.slider("训练年限", 0.0, 15.0, 1.0, 0.5)
    goal = st.selectbox("目标", ["增肌", "减脂"])
    equipment = st.multiselect(
        "可用器械", ["哑铃", "杠铃", "绳索", "腿举机", "自重"],
        default=["哑铃", "杠铃"]
    )
    days = st.slider("每周训练天数", 1, 7, 4)
    injuries = st.multiselect(
        "伤病史", ["下背痛", "肩伤", "膝伤", "手腕伤", "无"], default=["无"]
    )
    if "无" in injuries:
        injuries = []

# Main area: Tabs
tab1, tab2, tab3 = st.tabs(["训练计划", "动作分析", "知识问答"])

with tab1:
    st.header("生成训练计划")
    query = st.text_area("补充说明", "帮我设计增肌计划，重点练胸和背")
    if st.button("生成计划", type="primary", key="gen_plan"):
        with st.spinner("AI 正在为你编排计划..."):
            profile = UserProfileInput(
                height=height, weight=weight, training_years=years,
                goal=goal, available_equipment=equipment, days_per_week=days,
                injuries=injuries
            )
            orch = Orchestrator()
            result = orch.generate_plan(profile, query)
            st.success(f"计划生成完成 | 置信度: {result.get('confidence', 0):.0%}")
            if result.get("warnings"):
                st.warning("⚠️ 安全提示：\n" + "\n".join(f"- {w}" for w in result["warnings"]))
            if result.get("requires_review"):
                st.info("此计划含低置信度建议，请谨慎执行")
            days_data = result.get("days", [])
            if days_data:
                for day in days_data:
                    with st.expander(f"第{day['day']}天 - {day['focus']}", expanded=True):
                        for ex in day.get("exercises", []):
                            st.markdown(f"**{ex['name']}** — {ex.get('sets', 3)}组 × {ex.get('reps', '8-12')}次 | 休息{ex.get('rest', '60s')}")
                            if ex.get("notes"):
                                st.caption(ex["notes"])
            else:
                st.json(result)

with tab2:
    st.header("动作分析")
    ex_name = st.text_input("动作名称", "哑铃卧推")
    ex_desc = st.text_area("描述你的训练感受", "推的时候右边肩膀前侧有点疼")
    if st.button("分析动作", type="primary", key="analyze"):
        with st.spinner("分析中..."):
            profile = UserProfileInput(
                height=height, weight=weight, training_years=years,
                goal=goal, available_equipment=equipment, days_per_week=days,
                injuries=injuries
            )
            orch = Orchestrator()
            result = orch.analyze_exercise(ex_name, ex_desc, profile)
            st.markdown(f"严重程度: **{result.get('severity', '未知')}**")
            st.subheader("发现的问题")
            for issue in result.get("issues_found", []):
                st.markdown(f"- {issue}")
            st.subheader("改进建议")
            for sug in result.get("suggestions", []):
                st.markdown(f"- {sug}")

with tab3:
    st.header("知识问答")
    question = st.text_area("输入你的健身问题", "硬拉的时候下背酸正常吗？")
    if st.button("提问", type="primary", key="ask"):
        st.info("知识问答功能开发中，将使用 GraphRAG 知识图谱推理回答。敬请期待！")
