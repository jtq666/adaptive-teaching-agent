from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from src.models import TeachingSession
from src.ui import STATUS_LABELS, action_label, decision_mode_label, get_store, page_header, set_notice

PHASE_LABELS = {
    "diagnosis": "诊断理解",
    "instruction": "提供支架",
    "practice": "练习应用",
    "repair": "修复误解",
    "transfer": "迁移验证",
    "completed": "教学完成",
    "paused": "暂时暂停",
}

page_header(
    "过程回放",
    "选择一段已保存会话，复核学生状态变化、Skill 切换和终止依据。",
    eyebrow="教学证据档案 · 每一轮都可追溯",
    icon="timeline",
)

store = get_store()
with st.container(horizontal=True, vertical_alignment="center", horizontal_alignment="distribute"):
    with st.popover("导入会话 JSON", icon=":material/upload:"):
        uploaded = st.file_uploader("选择以前导出的会话", type=["json"], label_visibility="collapsed")
        if st.button("导入并保存", type="primary", disabled=uploaded is None, width="stretch"):
            try:
                if uploaded is None:
                    raise ValueError("请选择 JSON 文件")
                imported = TeachingSession.model_validate_json(uploaded.getvalue())
                imported = store.import_session(imported)
                set_notice(f"已导入“{imported.goal.topic}”会话。")
                st.rerun()
            except Exception as exc:
                st.error(f"无法导入：{exc}", icon=":material/error:")
    with st.popover("回收站", icon=":material/restore_from_trash:"):
        trashed = store.list_trash()
        if trashed:
            restore_id = st.selectbox("选择待恢复会话", trashed)
            if st.button("恢复会话", icon=":material/restore:", width="stretch"):
                store.restore(restore_id)
                set_notice("会话已从回收站恢复")
                st.rerun()
        else:
            st.caption("回收站为空")
    st.caption("每轮自动保存 · 支持导入、导出、复制、归档与可恢复删除")

with st.container(border=True):
    filter_row = st.container(horizontal=True, vertical_alignment="bottom")
    with filter_row:
        search = st.text_input(
            "搜索会话",
            placeholder="主题、学生或会话 ID",
            key="replay_search",
            icon=":material/search:",
        )
        status_filter = st.selectbox("状态", ["全部", "教学中", "已达成", "已暂停"], key="replay_status")
        course_options = ["全部", *store.list_courses(include_archived=True)]
        course_filter = st.selectbox("课程", course_options, key="replay_course")

status_reverse = {value: key for key, value in STATUS_LABELS.items()}
page_size = 20
metadata, total_records = store.list_metadata(
    page=1,
    page_size=page_size,
    query=search,
    course="" if course_filter == "全部" else course_filter,
    status="" if status_filter == "全部" else status_reverse[status_filter],
    include_archived=True,
)
page_count = max(1, (total_records + page_size - 1) // page_size)
page = st.number_input("页码", min_value=1, max_value=page_count, value=1, step=1)
if page != 1:
    metadata, total_records = store.list_metadata(
        page=int(page),
        page_size=page_size,
        query=search,
        course="" if course_filter == "全部" else course_filter,
        status="" if status_filter == "全部" else status_reverse[status_filter],
        include_archived=True,
    )
records = []
for item in metadata:
    updated_time = datetime.fromisoformat(item["updated_at"]).strftime("%m-%d %H:%M")
    status = item.get("status", "active").replace("SessionStatus.", "").lower()
    label = (
        f"{updated_time}　{item.get('display_title') or item['topic']}　·　{item['student_name']}　·　"
        f"{STATUS_LABELS.get(status, status)}　·　#{item['session_id'][:6]}"
    )
    records.append((label, item["session_id"], item))
st.caption(f"共 {total_records} 个有效会话 · 每页 20 条 · 第 {page}/{page_count} 页（正文按选中后加载）")
if not records:
    st.info("没有匹配的会话。清除搜索词或调整筛选条件。", icon=":material/search_off:")
    st.stop()

labels = {label: session_id for label, session_id, _ in records}
if st.session_state.get("selected_replay") not in labels:
    st.session_state.selected_replay = next(iter(labels))
selected_label = st.selectbox("选择会话", list(labels), key="selected_replay")
session = store.load(labels[selected_label])


title_row = st.container(horizontal=True, vertical_alignment="center", horizontal_alignment="distribute")
with title_row:
    st.subheader(f"{session.display_title or session.goal.topic} · {session.profile.name}")
    with st.container(horizontal=True, vertical_alignment="center"):
        if str(session.status) == "active" and st.button("继续这次教学", type="primary", icon=":material/play_arrow:"):
            st.session_state.teaching_session = session
            set_notice("已恢复会话，可以从最后一轮继续。")
            st.switch_page("app_pages/live_teaching.py")
        with st.popover("编辑信息", icon=":material/edit:"):
            with st.form(f"edit_session_{session.session_id}", border=False):
                display_title = st.text_input(
                    "展示名称",
                    value=session.display_title or session.goal.topic,
                    key=f"edit_topic_{session.session_id}",
                )
                st.caption("只修改展示名称；原始教学目标与学生状态保持不可变。")
                submitted = st.form_submit_button(
                    "保存修改",
                    type="primary",
                    icon=":material/save:",
                    width="stretch",
                )
            if submitted:
                try:
                    updated_session = store.update_metadata(session.session_id, display_title=display_title)
                    active = st.session_state.get("teaching_session")
                    if active is not None and active.session_id == session.session_id:
                        st.session_state.teaching_session = updated_session
                    set_notice("会话展示名称已更新。", ":material/edit_note:")
                    st.rerun()
                except Exception as exc:
                    st.error(f"无法保存修改：{exc}", icon=":material/error:")
        if st.button("复制", icon=":material/content_copy:"):
            clone = store.duplicate(session.session_id)
            set_notice(f"已创建副本 #{clone.session_id[:6]}")
            st.rerun()
        archive_label = "取消归档" if session.archived_at else "归档"
        if st.button(archive_label, icon=":material/unarchive:" if session.archived_at else ":material/archive:"):
            was_archived = bool(session.archived_at)
            store.archive(session.session_id, not was_archived)
            set_notice("会话已恢复到活动列表" if was_archived else "会话已归档")
            st.rerun()
        with st.popover("删除会话", icon=":material/delete:"):
            st.warning(
                f"将“{session.display_title or session.goal.topic}”移入回收站，可稍后恢复。",
                icon=":material/warning:",
            )
            confirmed = st.checkbox(
                "我确认删除这份会话档案",
                key=f"confirm_delete_session_{session.session_id}",
            )
            if st.button(
                "移入回收站",
                type="primary",
                icon=":material/delete:",
                disabled=not confirmed,
                width="stretch",
            ):
                store.delete(session.session_id)
                active = st.session_state.get("teaching_session")
                if active is not None and active.session_id == session.session_id:
                    st.session_state.teaching_session = None
                set_notice("会话已移入回收站。", ":material/delete:")
                st.rerun()

teaching_rounds = session.answered_rounds()
with st.container(horizontal=True):
    st.metric("教学轮数", teaching_rounds, border=True)
    st.metric("最终掌握度", f"{session.state.average_mastery():.0%}", border=True)
    st.metric("Skill 切换", sum(bool(turn.switch_reason) for turn in session.turns), border=True)
    st.metric("状态", STATUS_LABELS[str(session.status)], border=True)

mastery_rows = []
for turn in session.turns:
    for point, value in turn.state_after.mastery.items():
        mastery_rows.append({"轮次": turn.round_index, "知识点": point, "掌握度": value})
if mastery_rows:
    chart_data = pd.DataFrame(mastery_rows)
    st.subheader("掌握度随教学轮次变化")
    st.line_chart(chart_data, x="轮次", y="掌握度", color="知识点", x_label="教学轮次", y_label="掌握度")

st.subheader("决策轨迹")
trajectory = pd.DataFrame(
    [
        {
            "轮次": turn.round_index,
            "内容 Skill": turn.content_skill_id or "通用诊断模式",
            "教学策略": turn.strategy_skill_id or "未记录",
            "阶段": PHASE_LABELS.get(turn.phase, turn.phase),
            "动作": action_label(turn.action_type),
            "决策来源": decision_mode_label(turn.decision_mode),
            "平均掌握变化": turn.state_after.average_mastery() - turn.state_before.average_mastery(),
            "发生切换": bool(turn.switch_reason),
            "终止判断": turn.stop_decision,
        }
        for turn in session.turns
    ]
)
st.dataframe(
    trajectory,
    hide_index=True,
    column_config={
        "平均掌握变化": st.column_config.NumberColumn(format="%+.1%"),
        "发生切换": st.column_config.CheckboxColumn(),
        "内容 Skill": st.column_config.TextColumn(width="large"),
        "教学策略": st.column_config.TextColumn(width="large"),
    },
)

st.subheader("逐轮证据")
for turn in session.turns:
    with st.expander(
        f"第 {turn.round_index} 轮 · {action_label(turn.action_type)} · {turn.content_skill_id or '通用诊断'} × {turn.strategy_skill_id or '未记录'}",
        expanded=turn.round_index == len(session.turns),
        icon=":material/route:",
    ):
        top = st.container(horizontal=True)
        with top:
            st.badge(f"第 {turn.round_index} 轮", color="blue")
            st.caption(f"内容 Skill：{turn.content_skill_id or '通用诊断模式'}")
            st.caption(f"教学策略：{turn.strategy_skill_id or '未记录'}")
            st.caption(f"阶段：{PHASE_LABELS.get(turn.phase, turn.phase)}")
            st.caption(action_label(turn.action_type))
        if turn.student_message:
            st.markdown(f"**学生：** {turn.student_message}")
        st.markdown(f"**教师：** {turn.teacher_message}")
        st.caption(f"选择理由：{turn.selection_reason}")
        st.caption(f"决策来源：{decision_mode_label(turn.decision_mode)}　|　触发规则：{turn.policy_rule or '无'}")
        if turn.candidate_skill_ids:
            st.caption("候选 Skill：" + " → ".join(turn.candidate_skill_ids))
        if turn.switch_reason:
            st.info(turn.switch_reason)
        state_table = pd.DataFrame(
            [
                {
                    "知识点": point,
                    "轮前": turn.state_before.mastery.get(point, 0.0),
                    "轮后": turn.state_after.mastery.get(point, 0.0),
                    "变化": turn.state_after.mastery.get(point, 0.0) - turn.state_before.mastery.get(point, 0.0),
                }
                for point in turn.state_after.mastery
            ]
        )
        st.dataframe(
            state_table,
            hide_index=True,
            column_config={
                "轮前": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="percent"),
                "轮后": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="percent"),
                "变化": st.column_config.NumberColumn(format="%+.2f"),
            },
        )

st.download_button(
    "导出完整会话 JSON",
    data=session.model_dump_json(indent=2),
    file_name=f"session_{session.session_id}.json",
    mime="application/json",
    icon=":material/download:",
)
