import streamlit as st

from src.showcase import inject_showcase_css
from src.ui import get_library, get_llm_client, initialize_state, show_notice

st.set_page_config(
    page_title="自适应教学决策台",
    page_icon=":material/school:",
    layout="wide",
    # Let Streamlit keep the evidence panel open on desktop while collapsing it
    # on narrow screens. A forced-expanded sidebar covered the entire mobile
    # teaching form on first load.
    initial_sidebar_state="auto",
)
initialize_state()
inject_showcase_css()
show_notice()

with st.sidebar:
    st.caption(":material/school:　Teaching Agent")
    st.header("自适应教学")
    st.caption("根据学生的真实回答，决定下一步教什么、怎么教。")
    library = get_library()
    client = get_llm_client()
    if client.available:
        st.badge("LLM 已连接", icon=":material/cloud_done:", color="green")
        st.caption(f"{client.model} · {len(library.skills)} 个 Skill")
    else:
        st.badge("LLM 未连接", icon=":material/cloud_off:", color="orange")
        st.caption("请配置 API 后再进行真实教学。")
    st.caption("v6 · 会话与评估自动保存到本地 JSON")
    with st.popover("功能导航", icon=":material/route:", width="stretch"):
        st.markdown("**1　实时教学**")
        st.caption("根据学生回答调整教学方式")
        st.markdown("**2　过程回放**")
        st.caption("查看逐轮回答与学习证据")
        st.markdown("**3　Agent 评估**")
        st.caption("比较不同教学策略的表现")

navigation = st.navigation(
    [
        st.Page("app_pages/live_teaching.py", title="实时教学", icon=":material/forum:"),
        st.Page("app_pages/skill_library.py", title="Skill Library", icon=":material/library_books:"),
        st.Page("app_pages/session_replay.py", title="过程回放", icon=":material/timeline:"),
        st.Page("app_pages/evaluation_lab.py", title="Agent 评估", icon=":material/experiment:"),
    ],
    position="top",
)
navigation.run()
