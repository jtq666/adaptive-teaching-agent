from __future__ import annotations

from datetime import datetime
from html import escape

import pandas as pd
import streamlit as st

from src.models import TeachingSession
from src.ui import STATUS_LABELS, action_label, get_library, get_store, page_header, set_notice

PHASE_LABELS = {
    "diagnosis": "诊断理解",
    "instruction": "提供支架",
    "practice": "练习应用",
    "repair": "修复误解",
    "transfer": "迁移验证",
    "completed": "教学完成",
    "paused": "暂时暂停",
}


def skill_name(skill_id: str | None) -> str:
    if not skill_id:
        return "通用教学"
    try:
        return get_library().get(skill_id).name
    except (KeyError, ValueError):
        return skill_id


def action_reason(turn: object) -> str:
    action_type = str(getattr(turn, "action_type", ""))
    reasons = {
        "diagnostic": "学生表达了困惑或不确定，先确认具体卡点。",
        "scaffold": "学生方向基本正确，但步骤或表达还不清楚，先把任务拆小。",
        "correction": "学生明确表达了错误概念，先用对比把错误模型纠正过来。",
        "correct": "学生明确表达了错误概念，先用对比把错误模型纠正过来。",
        "transfer": "当前理解证据已经足够，换一个情境检查能否迁移。",
        "subject_instruction": "当前没有明确困难，继续讲清当前知识点。",
        "explain": "学生需要建立当前概念之间的联系，先进行简短讲解。",
    }
    reason = reasons.get(action_type, "根据学生本轮回答调整下一步教学。")
    audit = getattr(turn, "generation_audit", {}) or {}
    evidence = str(audit.get("evidence", "")).strip()
    if evidence and evidence not in reason:
        reason = f"{reason} 学生证据：{evidence}"
    return reason


page_header(
    "过程回放",
    "只看教学是怎样根据学生回答变化的。",
    eyebrow="教学过程 · 关键证据",
    icon="timeline",
)

store = get_store()
with st.popover("会话管理", icon=":material/more_horiz:"):
    with st.expander("导入会话", expanded=False, icon=":material/upload:"):
        uploaded = st.file_uploader("选择以前导出的会话", type=["json"], label_visibility="collapsed")
        if st.button("导入并保存", type="primary", disabled=uploaded is None, width="stretch"):
            try:
                if uploaded is None:
                    raise ValueError("请选择 JSON 文件")
                imported = store.import_session(TeachingSession.model_validate_json(uploaded.getvalue()))
                set_notice(f"已导入“{imported.goal.topic}”会话。")
                st.rerun()
            except Exception as exc:
                st.error(f"无法导入：{exc}", icon=":material/error:")
    with st.expander("回收站", expanded=False, icon=":material/restore_from_trash:"):
        trashed = store.list_trash()
        if trashed:
            restore_id = st.selectbox("选择待恢复会话", trashed)
            if st.button("恢复会话", icon=":material/restore:", width="stretch"):
                store.restore(restore_id)
                set_notice("会话已恢复")
                st.rerun()
        else:
            st.caption("回收站为空")

with st.container(border=True):
    filter_row = st.container(horizontal=True, vertical_alignment="bottom")
    with filter_row:
        search = st.text_input("搜索会话", placeholder="主题、学生或会话 ID", key="replay_search", icon=":material/search:")
        status_filter = st.selectbox("状态", ["全部", "教学中", "已达成", "已暂停"], key="replay_status")
        course_options = ["全部", *store.list_courses(include_archived=True)]
        course_filter = st.selectbox("课程", course_options, key="replay_course")

status_reverse = {value: key for key, value in STATUS_LABELS.items()}
metadata, total_records = store.list_metadata(
    page=1,
    page_size=20,
    query=search,
    course="" if course_filter == "全部" else course_filter,
    status="" if status_filter == "全部" else status_reverse[status_filter],
    include_archived=True,
)
if not metadata:
    st.info("没有匹配的会话。清除搜索词或调整筛选条件。", icon=":material/search_off:")
    st.stop()

labels: dict[str, str] = {}
for item in metadata:
    updated_time = datetime.fromisoformat(item["updated_at"]).strftime("%m-%d %H:%M")
    status = item.get("status", "active").replace("SessionStatus.", "").lower()
    labels[
        f"{updated_time} · {item.get('display_title') or item['topic']} · {item['student_name']} · "
        f"{STATUS_LABELS.get(status, status)}"
    ] = item["session_id"]

st.caption(f"共 {total_records} 个会话 · 选择一份查看关键教学证据")
if st.session_state.get("selected_replay") not in labels:
    st.session_state.selected_replay = next(iter(labels))
selected_label = st.selectbox("选择会话", list(labels), key="selected_replay")
session = store.load(labels[selected_label])

with st.container(horizontal=True, vertical_alignment="center", horizontal_alignment="distribute"):
    st.subheader(f"{session.display_title or session.goal.topic} · {session.profile.name}")
    if str(session.status) == "active" and st.button("继续教学", type="primary", icon=":material/play_arrow:"):
        st.session_state.teaching_session = session
        set_notice("已恢复会话，可以从最后一轮继续。")
        st.switch_page("app_pages/live_teaching.py")

with st.popover("更多操作", icon=":material/more_horiz:"):
    with st.form(f"edit_session_{session.session_id}", border=False):
        display_title = st.text_input("展示名称", value=session.display_title or session.goal.topic)
        submitted = st.form_submit_button("保存名称", type="primary", icon=":material/save:")
    if submitted:
        try:
            store.update_metadata(session.session_id, display_title=display_title)
            set_notice("会话展示名称已更新。")
            st.rerun()
        except Exception as exc:
            st.error(f"无法保存修改：{exc}", icon=":material/error:")
    if st.button("复制会话", icon=":material/content_copy:"):
        clone = store.duplicate(session.session_id)
        set_notice(f"已创建副本 #{clone.session_id[:6]}")
        st.rerun()
    archive_label = "取消归档" if session.archived_at else "归档会话"
    if st.button(archive_label, icon=":material/archive:"):
        store.archive(session.session_id, not bool(session.archived_at))
        set_notice("会话归档状态已更新")
        st.rerun()
    with st.expander("删除会话", expanded=False, icon=":material/delete:"):
        confirmed = st.checkbox("确认移入回收站", key=f"confirm_delete_{session.session_id}")
        if st.button("移入回收站", type="primary", disabled=not confirmed, width="stretch"):
            store.delete(session.session_id)
            set_notice("会话已移入回收站")
            st.rerun()
    st.download_button(
        "导出完整 JSON",
        data=session.model_dump_json(indent=2),
        file_name=f"session_{session.session_id}.json",
        mime="application/json",
        icon=":material/download:",
        width="stretch",
    )

teaching_rounds = session.answered_rounds()
with st.container(horizontal=True):
    st.metric("教学轮数", teaching_rounds, border=True)
    st.metric("最终掌握度", f"{session.state.average_mastery():.0%}", border=True)
    st.metric("Skill 切换", sum(bool(turn.switch_reason) for turn in session.turns), border=True)
    st.metric("状态", STATUS_LABELS[str(session.status)], border=True)

st.subheader("关键教学轨迹")
trajectory = pd.DataFrame(
    [
        {
            "轮次": turn.round_index,
            "内容 Skill": skill_name(turn.content_skill_id),
            "教学动作": action_label(turn.action_type),
            "阶段": PHASE_LABELS.get(turn.phase, turn.phase),
            "掌握度": turn.state_after.average_mastery(),
        }
        for turn in session.turns
    ]
)
st.dataframe(
    trajectory,
    hide_index=True,
    column_config={"掌握度": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="percent")},
)

action_changes: list[dict[str, str | int]] = []
previous_turn = None
for turn in session.turns:
    if previous_turn is not None and turn.action_type != previous_turn.action_type:
        evidence = next(
            (item for item in reversed(turn.state_after.evidence) if item.round_index == turn.round_index),
            None,
        )
        action_changes.append(
            {
                "轮次": turn.round_index,
                "学生触发": turn.student_message or "首轮建立教学情境",
                "动作变化": f"{action_label(previous_turn.action_type)} → {action_label(turn.action_type)}",
                "本轮证据": evidence.reason if evidence else "暂无单独证据说明",
                "Agent 依据": action_reason(turn),
            }
        )
    previous_turn = turn

with st.expander("为什么改变动作", expanded=True, icon=":material/psychology_alt:"):
    if action_changes:
        st.caption("这里只展示动作发生变化的轮次，便于复核学生回答如何触发下一步教学。")
        cards = []
        for index, change in enumerate(action_changes):
            previous_action, current_action = str(change["动作变化"]).split(" → ", 1)
            accent = "orange" if index % 2 else "teal"
            cards.append(
                """
                <article class="replay-change replay-change--%s">
                  <div class="replay-change-round">第<br><strong>%s</strong>轮</div>
                  <div class="replay-change-body">
                    <div class="replay-change-path">
                      <span class="replay-action replay-action--muted">%s</span>
                      <span class="replay-arrow">→</span>
                      <span class="replay-action replay-action--active">%s</span>
                    </div>
                    <div class="replay-change-row"><span>学生回答</span><p>%s</p></div>
                    <div class="replay-change-row"><span>为什么</span><p>%s</p></div>
                    <div class="replay-change-row replay-change-row--evidence"><span>本轮证据</span><p>%s</p></div>
                  </div>
                </article>
                """
                % (
                    accent,
                    escape(str(change["轮次"])),
                    escape(previous_action),
                    escape(current_action),
                    escape(str(change["学生触发"])),
                    escape(str(change["Agent 依据"])),
                    escape(str(change["本轮证据"])),
                )
            )
        st.html(
            """
            <style>
              .replay-change-list { display: flex; flex-direction: column; gap: 10px; }
              .replay-change { display: grid; grid-template-columns: 54px 1fr; gap: 14px; padding: 14px 16px; border: 1px solid #d9e4e1; border-radius: 12px; background: #fbfdfc; }
              .replay-change--teal { border-left: 4px solid #0f766e; }
              .replay-change--orange { border-left: 4px solid #d8893e; }
              .replay-change-round { padding-top: 2px; color: #6b7c78; font-size: 12px; line-height: 1.25; text-align: center; }
              .replay-change-round strong { color: #173b3d; font-size: 20px; }
              .replay-change-path { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }
              .replay-action { display: inline-block; padding: 4px 9px; border-radius: 7px; font-size: 13px; font-weight: 700; }
              .replay-action--muted { color: #56706d; background: #edf3f1; }
              .replay-action--active { color: #0f625d; background: #dff1eb; }
              .replay-arrow { color: #d8893e; font-size: 18px; font-weight: 700; }
              .replay-change-row { display: grid; grid-template-columns: 66px 1fr; gap: 10px; padding-top: 7px; border-top: 1px solid #edf1ef; }
              .replay-change-row span { color: #6b7c78; font-size: 12px; }
              .replay-change-row p { margin: 0; color: #294947; font-size: 13px; line-height: 1.55; }
              .replay-change-row--evidence p { color: #56706d; }
              @media (max-width: 700px) { .replay-change { grid-template-columns: 42px 1fr; padding: 12px; } .replay-change-row { grid-template-columns: 54px 1fr; } }
            </style>
            <div class="replay-change-list">%s</div>
            """ % "".join(cards),
            unsafe_allow_javascript=False,
        )
    else:
        st.info("当前会话还没有发生教学动作变化。", icon=":material/info:")

st.subheader("逐轮教学")
for turn in session.turns:
    before = turn.state_before.average_mastery()
    after = turn.state_after.average_mastery()
    latest_evidence = next(
        (item for item in reversed(turn.state_after.evidence) if item.round_index == turn.round_index),
        None,
    )
    with st.expander(
        f"第 {turn.round_index} 轮 · {action_label(turn.action_type)} · {skill_name(turn.content_skill_id)}",
        expanded=turn.round_index == len(session.turns),
        icon=":material/chat:",
    ):
        st.caption(f"阶段：{PHASE_LABELS.get(turn.phase, turn.phase)} · 掌握度 {before:.0%} → {after:.0%}")
        if turn.student_message:
            st.markdown(f"**学生：** {turn.student_message}")
        st.markdown(f"**教师：** {turn.teacher_message}")
        if latest_evidence:
            st.caption(f"本轮证据：{latest_evidence.evidence_level} · {latest_evidence.reason}")
        st.caption(f"动作依据：{action_reason(turn)}")
        if turn.switch_reason:
            st.info("根据学生状态，教学内容或方式在本轮发生了调整。", icon=":material/swap_horiz:")
