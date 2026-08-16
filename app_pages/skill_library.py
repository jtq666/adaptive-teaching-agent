from __future__ import annotations

import streamlit as st
import yaml  # type: ignore[import-untyped]

from src.skills import TeachingSkill
from src.ui import get_library, page_header

TYPE_LABELS = {
    "subject": "学科教学",
    "strategy": "自适应教学",
    "diagnostic": "诊断提问",
    "scaffold": "分层提示",
    "correction": "误解纠正",
    "transfer": "迁移验证",
}

TRIGGER_LABELS = {
    "initial": "首次教学",
    "followup": "正常跟进",
    "confusion": "学生困惑",
    "misconception": "存在误解",
}


def course_label(skill: TeachingSkill) -> str:
    if skill.skill_type != "subject":
        return "跨课程通用"
    if skill.courses:
        aliases = {"算法设计": "程序设计"}
        return aliases.get(skill.courses[0], skill.courses[0])
    source = skill.source_course or ""
    if any(token in source for token in ("算法", "程序", "代码")):
        return "程序设计"
    if any(token in source for token in ("数学", "微积分", "导数", "积分")):
        return "高等数学"
    if any(token in source for token in ("物理", "力学", "牛顿", "动量")):
        return "大学物理"
    return source or "其他"


def matches(skill: TeachingSkill, query: str, course: str, skill_type: str) -> bool:
    searchable = " ".join(
        [
            skill.skill_id,
            skill.name,
            skill.source_course or "",
            *skill.trigger,
            *skill.goal,
            *skill.applicable_when,
            *skill.topic_tags,
            *skill.required_prior_knowledge,
        ]
    ).lower()
    return (
        (not query or query.lower() in searchable)
        and (course == "全部课程" or course_label(skill) == course)
        and (skill_type == "全部类型" or TYPE_LABELS.get(skill.skill_type, skill.skill_type) == skill_type)
    )


library = get_library()
subject_count = len(library.by_type("subject"))
generic_count = len(library.skills) - subject_count

page_header(
    "Skill Library",
    "查看 Agent 的教学工具箱：学科 Skill 提供内容依据，统一自适应 Skill 选择本轮教学动作。",
    eyebrow=f"{len(library.skills)} 个当前可用 Skill · 来源、触发条件与版本可审计",
    icon="library_books",
)

with st.container(horizontal=True):
    st.metric("全部 Skill", len(library.skills), border=True)
    st.metric("学科教学", subject_count, border=True)
    st.metric("通用自适应", generic_count, border=True)

st.info(
    "实时教学只使用 adaptive_teaching_v1：模型根据学生原话选择讲解、诊断、分层提示、"
    "误解纠正或迁移验证。旧的策略 Skill 仅为历史会话和评估数据保留。",
    icon=":material/account_tree:",
)

with st.popover("导入 Skill", icon=":material/upload_file:"):
    st.caption("上传 UTF-8 YAML。系统会先校验和预览，不会覆盖已有 Skill。")
    uploaded = st.file_uploader(
        "选择 Skill YAML",
        type=["yaml", "yml"],
        max_upload_size=1,
        key="skill_import_file",
    )
    if uploaded is not None:
        try:
            raw_skill = uploaded.getvalue()
            preview = library.validate_import(raw_skill)
            conflict = preview.skill_id in library.by_id
            st.success("格式与必填字段校验通过", icon=":material/check_circle:")
            st.write(f"**{preview.name}**")
            st.caption(f"Skill ID：{preview.skill_id} · 类型：{TYPE_LABELS.get(preview.skill_type, preview.skill_type)}")
            st.write("触发条件：" + "；".join(preview.trigger[:2]))
            if preview.skill_type == "subject":
                st.caption(
                    "硬约束："
                    f"课程 {' / '.join(preview.courses)} · "
                    f"主题标签 {' / '.join(preview.topic_tags)} · "
                    f"前置证据 {' / '.join(preview.required_prior_knowledge)}"
                )
            new_id = ""
            if conflict:
                st.warning("该 Skill ID 已存在。原文件不会被覆盖，请输入一个新的 ID。")
                new_id = st.text_input(
                    "新的 Skill ID",
                    value=f"{preview.skill_id}_v2",
                    key="skill_import_new_id",
                )
            if st.button("确认导入", type="primary", icon=":material/add:", width="stretch"):
                imported = library.import_skill(raw_skill, new_skill_id=new_id or None)
                get_library.clear()
                st.toast(f"已导入 {imported.name}", icon=":material/check_circle:")
                st.rerun()
        except (ValueError, FileExistsError) as exc:
            st.error(str(exc), icon=":material/error:")

with st.popover("用户 Skill 管理", icon=":material/manage_accounts:"):
    user_ids = library.user_skill_ids()
    archived_ids = library.list_archived_user_skills()
    st.caption("内置 Skill 永远只读；用户导入的 Skill 通过版本 ID 管理，不覆盖历史会话引用。")
    if user_ids:
        managed_id = st.selectbox("选择用户 Skill", user_ids, key="managed_skill_id")
        with st.container(horizontal=True):
            if st.button("归档用户 Skill", icon=":material/archive:"):
                library.archive_user_skill(managed_id)
                get_library.clear()
                st.rerun()
            if st.button("移入回收站", icon=":material/delete:"):
                library.delete_user_skill(managed_id)
                get_library.clear()
                st.rerun()
    else:
        st.caption("还没有用户导入版本。")
    if archived_ids:
        restore_id = st.selectbox("选择归档版本", archived_ids, key="restore_skill_id")
        if st.button("恢复用户 Skill", icon=":material/restore:"):
            library.restore_user_skill(restore_id)
            get_library.clear()
            st.rerun()

filters = st.container(horizontal=True, vertical_alignment="bottom")
with filters:
    query = st.text_input(
        "搜索 Skill", placeholder="输入名称、主题或 Skill ID", icon=":material/search:"
    ) or ""
    course = st.selectbox(
        "课程",
        ["全部课程", "程序设计", "高等数学", "大学物理", "跨课程通用"],
        index=0,
    ) or "全部课程"
    skill_type = st.selectbox("类型", ["全部类型", *TYPE_LABELS.values()], index=0) or "全部类型"

visible = [skill for skill in library.skills if matches(skill, query, course, skill_type)]
st.caption(f"找到 {len(visible)} 个 Skill。点击条目查看触发条件、前置条件和执行步骤。")

for skill in visible:
    kind = TYPE_LABELS.get(skill.skill_type, skill.skill_type)
    with st.expander(
        f"{skill.name}　·　{kind}",
        icon=":material/menu_book:" if skill.skill_type == "subject" else ":material/route:",
    ):
        with st.container(horizontal=True):
            st.badge(course_label(skill), color="blue")
            st.badge(kind, color="green" if skill.skill_type != "subject" else "gray")
            st.badge("内置只读" if library.is_builtin(skill.skill_id) else "用户版本", color="gray" if library.is_builtin(skill.skill_id) else "violet")
            st.caption(f"Skill ID：{skill.skill_id}　·　版本 {skill.version}")

        if skill.added_reason:
            st.info(f"为什么新增：{skill.added_reason}", icon=":material/lightbulb:")

        if skill.skill_type == "subject":
            with st.container(border=True):
                st.markdown("**Agent 执行前的四项硬约束**")
                constraint_columns = st.columns(2, gap="large")
                constraint_columns[0].caption("课程")
                constraint_columns[0].write(" / ".join(skill.courses))
                constraint_columns[0].caption("教学目标标签")
                constraint_columns[0].write(" / ".join(skill.topic_tags))
                constraint_columns[1].caption("允许触发阶段")
                constraint_columns[1].write(
                    " / ".join(TRIGGER_LABELS.get(item, item) for item in skill.trigger_states)
                )
                constraint_columns[1].caption("至少一项前置证据")
                constraint_columns[1].write(" / ".join(skill.required_prior_knowledge))
                st.caption(
                    f"适用掌握区间：{skill.mastery_range[0]:.2f}–{skill.mastery_range[1]:.2f}；"
                    f"当平均掌握度达到 {skill.prerequisite_mastery_bypass:.2f} 时，可由状态证据替代画像中的前置知识。"
                )

        overview, execution = st.columns(2, gap="large")
        with overview:
            st.markdown("**什么时候使用**")
            conditions = skill.applicable_when or skill.trigger
            for item in conditions:
                st.markdown(f"- {item}")
            st.markdown("**使用前需要什么**")
            for item in skill.preconditions:
                st.markdown(f"- {item}")
        with execution:
            st.markdown("**执行步骤**")
            for index, item in enumerate(skill.procedure, 1):
                st.markdown(f"{index}. {item}")
            st.markdown("**如何判断有效**")
            for item in skill.verification:
                st.markdown(f"- {item}")

        if skill.source_video:
            st.caption(
                f"来源：{skill.source_video}"
                + (f" · {skill.source_timestamp}" if skill.source_timestamp else "")
            )
        st.download_button(
            "导出 YAML",
            data=yaml.safe_dump(skill.model_dump(exclude_none=True), allow_unicode=True, sort_keys=False),
            file_name=f"{skill.skill_id}.yaml",
            mime="application/x-yaml",
            icon=":material/download:",
            key=f"export_skill_{skill.skill_id}",
        )

if not visible:
    st.warning("没有符合当前筛选条件的 Skill。请清空搜索词或调整课程与类型。")
