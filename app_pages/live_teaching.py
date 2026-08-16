from __future__ import annotations

from typing import Literal, cast

import streamlit as st

from src.agent import HybridTeachingAgent
from src.config import get_agent_settings
from src.demo_reply_generator import DemoReplyGenerator
from src.llm import LLMUnavailableError
from src.models import ConversationTurn, StudentProfile, StudentState, TeachingGoal
from src.ui import (
    STATUS_LABELS,
    action_label,
    build_agent,
    decision_mode_label,
    get_library,
    inject_live_chat_css,
    page_header,
    reset_teaching_session,
    set_notice,
)


def live_agent() -> HybridTeachingAgent:
    """Use the real LLM with the lightweight live-demo review budget."""

    return build_agent(fast_demo=True)

PRESETS = {
    "二分查找边界": {
        "course": "程序设计",
        "topic": "二分查找边界条件",
        "objective": "从区间不变量独立推导循环条件和边界更新",
        "points": "区间定义,循环不变量,边界更新",
        "prior": "while 循环,有序数组",
    },
    "导数极限定义": {
        "course": "高等数学",
        "topic": "导数的极限定义",
        "objective": "从平均变化率理解瞬时变化率，并解释极限如何定义导数",
        "points": "平均变化率,极限思想,瞬时变化率",
        "prior": "函数,斜率,平均变化率,平均速度",
    },
    "牛顿第一定律": {
        "course": "大学物理",
        "topic": "牛顿第一定律",
        "objective": "用公交车急刹车解释惯性，并区分保持运动与改变运动状态",
        "points": "惯性,合力与运动变化",
        "prior": "速度,力,运动,惯性",
    },
}
# The programming example remains available for legacy sessions and the
# required cross-course evaluation set, but it is intentionally not the
# recommended live-demo path: its notation-heavy boundary reasoning hides
# the adaptive teaching loop from a first-time reviewer.
DEMO_PRESET_NAMES = ("牛顿第一定律", "导数极限定义")
DEFAULT_PRESET_NAME = "牛顿第一定律"

RESPONSE_MODE_LABELS = {
    "open": "开放回答",
    "single_choice": "单选回答",
    "fill_blank": "填空回答",
    "numeric": "数值回答",
}

RESPONSE_PREFERENCE_LABELS = {
    "auto": "Agent 自适应选择",
    "open": "开放回答",
    "single_choice": "单选题",
    "fill_blank": "填空题",
    "numeric": "数值题",
}

PHASE_LABELS = {
    "diagnosis": "诊断理解",
    "instruction": "提供支架",
    "practice": "练习应用",
    "repair": "修复误解",
    "transfer": "迁移验证",
    "completed": "教学完成",
    "paused": "暂时暂停",
}
PHASE_ORDER = ["diagnosis", "instruction", "practice", "repair", "transfer"]

CONFIDENCE_CHOICE_CUES = ("我能说明", "我还不能说明", "我会", "我不会", "我不确定", "需要提示")


def response_mode_label(value: str) -> str:
    return RESPONSE_MODE_LABELS.get(value, "开放回答")


def phase_label(value: str) -> str:
    return PHASE_LABELS.get(value, value)


def has_reliable_choice_options(step) -> bool:
    """Only render choices that test course content, including old saved sessions."""
    if step is None or step.response_mode != "single_choice":
        return True
    texts = [option.text.strip() for option in step.options]
    return (
        2 <= len(texts) <= 4
        and len(set(texts)) == len(texts)
        and all(texts)
        and not all(any(cue in text for cue in CONFIDENCE_CHOICE_CUES) for text in texts)
    )

EVIDENCE_LABELS = {
    "none": "暂无有效证据",
    "partial": "部分理解",
    "correct": "正确回答",
    "explained": "说明了依据",
    "transfer": "迁移成功",
}


def apply_selected_preset() -> None:
    preset_name = st.session_state.get("preset_name")
    if preset_name not in DEMO_PRESET_NAMES:
        preset_name = DEFAULT_PRESET_NAME
        # Widget callbacks execute before the rerun, so this safely repairs
        # browser sessions created before the control became required.
        st.session_state.preset_name = preset_name
    preset = PRESETS[preset_name]
    st.session_state.start_course = preset["course"]
    st.session_state.start_topic = preset["topic"]
    st.session_state.start_objective = preset["objective"]
    st.session_state.start_points = preset["points"]
    st.session_state.start_prior = preset["prior"]
    # These fields are knowledge-point-specific. Keeping values from the
    # previous preset would silently apply binary-search labels to a physics
    # or calculus session.
    st.session_state.start_mastery_overrides = ""
    st.session_state.start_history = ""


def ensure_start_defaults(preset: dict[str, str]) -> None:
    preset_marker = "|".join((preset["course"], preset["topic"], preset["points"]))
    if st.session_state.get("_preset_defaults_marker") != preset_marker:
        st.session_state.start_mastery_overrides = ""
        st.session_state.start_history = ""
        st.session_state._preset_defaults_marker = preset_marker
    defaults = {
        "start_course": preset["course"],
        "start_topic": preset["topic"],
        "start_objective": preset["objective"],
        "start_points": preset["points"],
        "start_student_name": "演示学生",
        "start_level": "中等",
        "start_mastery": 0.35,
        "start_prior": preset["prior"],
        "start_mastery_overrides": "",
        "start_history": "",
        "start_preferences": ["具体例子", "逐步提示"],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if st.session_state.get("start_level") not in {"基础薄弱", "中等", "较好"}:
        st.session_state.start_level = "中等"


def execution_role_label(turn: ConversationTurn) -> str:
    content_id = turn.skill_plan.content_skill_id if turn.skill_plan else turn.content_skill_id
    strategy_id = turn.skill_plan.strategy_skill_id if turn.skill_plan else turn.strategy_skill_id
    if turn.selected_skill_id == strategy_id:
        return f"教学策略（{strategy_id or '通用策略'}）"
    if turn.selected_skill_id == content_id:
        return f"内容讲解（{content_id or '通用内容'}）"
    return f"执行 Skill（{turn.selected_skill_id}）"


def skill_switch_message(previous: ConversationTurn, current: ConversationTurn) -> str:
    previous_plan = previous.skill_plan
    current_plan = current.skill_plan
    content_before = previous_plan.content_skill_id if previous_plan else previous.content_skill_id
    content_after = current_plan.content_skill_id if current_plan else current.content_skill_id
    strategy_before = previous_plan.strategy_skill_id if previous_plan else previous.strategy_skill_id
    strategy_after = current_plan.strategy_skill_id if current_plan else current.strategy_skill_id
    plan_changes = []
    if content_before != content_after:
        plan_changes.append(f"内容 Skill：{content_before or '通用'} → {content_after or '通用'}")
    if strategy_before != strategy_after:
        plan_changes.append(f"教学策略：{strategy_before or '未指定'} → {strategy_after or '未指定'}")
    if plan_changes:
        role_change = ""
        if previous.selected_skill_id != current.selected_skill_id:
            role_change = (
                f"；本轮主执行角色：{execution_role_label(previous)} → "
                f"{execution_role_label(current)}"
            )
        return "**Skill 已切换 · 教学方案组成** · " + "；".join(plan_changes) + role_change
    if previous.selected_skill_id != current.selected_skill_id:
        return (
            "**Skill 已切换 · 本轮主执行角色** · "
            f"{execution_role_label(previous)} → {execution_role_label(current)}；"
            "内容 Skill × 教学策略组合保持不变。"
        )
    return "**执行方式已调整** · " + current.switch_reason


page_header(
    "实时教学",
    "老师每次只问一个问题，并根据你的回答决定下一步。",
    eyebrow="自适应教学",
    icon="forum",
)
inject_live_chat_css()
st.caption(
    "现场推荐先演示牛顿第一定律，再演示导数极限定义；"
    "二分查找保留在 Skill Library 和评估集，不作为默认演示入口。"
)
max_rounds = int(get_agent_settings().get("max_rounds", 8))

session = st.session_state.teaching_session
if session is None:
    library = get_library()
    if st.session_state.get("preset_name") not in DEMO_PRESET_NAMES:
        st.session_state.preset_name = DEFAULT_PRESET_NAME
    preset_name = st.segmented_control(
        "从示例开始",
        list(DEMO_PRESET_NAMES),
        required=True,
        key="preset_name",
        on_change=apply_selected_preset,
    )
    preset = PRESETS[preset_name if preset_name in DEMO_PRESET_NAMES else DEFAULT_PRESET_NAME]
    ensure_start_defaults(preset)
    skill_options = [skill.skill_id for skill in library.skills]
    with st.form("start_session", border=True):
        st.subheader("开始一段教学")
        st.caption("先填写最必要的信息；知识点状态、历史和 Skill 限制都在高级设置中。")
        setup_columns = st.columns([0.7, 1.3], gap="medium", vertical_alignment="bottom")
        with setup_columns[0]:
            course = st.selectbox(
                "课程",
                ["程序设计", "高等数学", "大学物理"],
                key="start_course",
                accept_new_options=True,
                help="可以输入新课程；若没有匹配的学科 Skill，系统会明确提示并进入通用诊断模式。",
            )
        with setup_columns[1]:
            topic = st.text_input("教学主题", key="start_topic", placeholder="例如：牛顿第一定律中的惯性")
        objective = st.text_area(
            "希望学生最终学会什么",
            height=88,
            key="start_objective",
            placeholder="例如：能解释惯性不是维持运动的力，并能迁移到新情境",
        )
        level = st.segmented_control(
            "学生基础",
            ["基础薄弱", "中等", "较好"],
            required=True,
            key="start_level",
        )
        st.caption("当前基础版统一使用开放回答，Agent 根据学生原话更新状态。")

        with st.expander("高级设置", icon=":material/tune:"):
            st.caption("这些设置用于自定义课程、恢复历史或限制本次可用 Skill。演示时通常无需修改。")
            points_text = st.text_input("知识点（逗号分隔）", key="start_points", help="每个知识点都会单独维护 0–1 掌握度。")
            student_name = st.text_input("学生姓名", key="start_student_name")
            initial_mastery = st.slider(
                "初始掌握度",
                0.0,
                1.0,
                step=0.05,
                key="start_mastery",
                help="这是可观察学习证据的起点，不是真实能力概率。",
            )
            mastery_overrides = st.text_input(
                "逐知识点掌握度（可选）",
                key="start_mastery_overrides",
                placeholder="例如：惯性:0.35, 合力与运动变化:0.20",
            )
            prior_text = st.text_input("已有知识（逗号分隔）", key="start_prior")
            preferences = st.pills(
                "学习偏好",
                ["具体例子", "图形", "逐步提示", "代码演示", "反例"],
                selection_mode="multi",
                key="start_preferences",
            )
            available_skill_ids = st.multiselect(
                "限制本次可用 Skill（留空表示全部匹配）",
                skill_options,
                default=[],
                key="start_available_skills",
                help="限制本次会话可调用的 Skill，便于对比和答辩演示。",
            )
            history_text = st.text_area(
                "历史对话（可选）",
                height=80,
                key="start_history",
                placeholder="学生：我总是混淆两个概念。\n教师：请先说说它们分别表示什么。",
                help="每行一轮，使用“学生：”或“教师：”开头；历史内容只作为上下文，不会重新计入本轮轮次。",
            )
        submitted = st.form_submit_button("开始教学", type="primary", icon=":material/play_arrow:", width="stretch")

    demo_notes = {
        "牛顿第一定律": "推荐首演：公交车急刹车的生活情境直观，最容易看出“困惑 → 诊断 → 分层提示”。",
        "导数极限定义": "推荐第二个演示：从平均速度过渡到瞬时速度，适合展示同一 Agent 在高等数学中的迁移。",
    }
    with st.container(border=True):
        st.markdown(f"**现场演示建议 · {preset_name}**")
        st.caption(demo_notes[preset_name])
        st.caption("操作顺序：开始教学 → 先输入困惑 → 再使用“AI 推荐演示回答”选择部分理解或正确回答 → 切到教师/答辩视图查看 Skill 变化。")

    if submitted:
        points = [item.strip() for item in points_text.replace("，", ",").split(",") if item.strip()]
        missing = [label for label, value in (("教学主题", topic.strip()), ("教学目标", objective.strip()), ("知识点", points), ("学生姓名", student_name.strip())) if not value]
        if missing:
            st.error("请补全：" + "、".join(missing), icon=":material/error:")
        else:
            try:
                override_map = {}
                for item in mastery_overrides.replace("，", ",").split(","):
                    if ":" in item:
                        point, raw_value = item.split(":", 1)
                        try:
                            override_map[point.strip()] = max(0.0, min(1.0, float(raw_value.strip())))
                        except ValueError:
                            raise ValueError(f"逐知识点掌握度格式错误：{item.strip()}") from None
                history = []
                for line in history_text.splitlines():
                    text = line.strip()
                    if not text:
                        continue
                    if text.startswith("学生："):
                        history.append({"role": "student", "content": text.removeprefix("学生：").strip()})
                    elif text.startswith("教师："):
                        history.append({"role": "teacher", "content": text.removeprefix("教师：").strip()})
                    else:
                        history.append({"role": "student", "content": text})
                goal = TeachingGoal(
                    course=str(course).strip(),
                    topic=topic.strip(),
                    objective=objective.strip(),
                    knowledge_points=points,
                    success_criteria=["能解释关键原理", "能完成迁移验证"],
                )
                profile = StudentProfile(
                    name=student_name.strip(),
                    level=cast(Literal["基础薄弱", "中等", "较好"], level),
                    prior_knowledge=[item.strip() for item in prior_text.replace("，", ",").split(",") if item.strip()],
                    learning_preferences=list(preferences or []),
                    response_preference="open",
                )
                state = StudentState(
                    mastery={point: override_map.get(point, initial_mastery) for point in points},
                    next_focus=f"诊断“{points[0]}”的真实理解",
                )
                with st.spinner("正在诊断起点并选择首个 Teaching Skill…"):
                    st.session_state.teaching_session = live_agent().start_session(
                        goal,
                        profile,
                        state,
                        history=history,
                        available_skill_ids=available_skill_ids,
                    )
                set_notice("教学会话已创建，并自动保存首轮记录。")
                st.rerun()
            except Exception as exc:
                st.error(f"无法开始教学：{exc}", icon=":material/error:")
else:
    # Streamlit removes widgets that are not rendered after a session starts.
    # Keep the form's key fields available for a rerun that only changes the
    # student/teacher view; this also prevents AppTest and browser back/forward
    # navigation from resurrecting a stale segmented-control widget.
    st.session_state.setdefault("start_level", st.session_state.teaching_session.profile.level)
    session = st.session_state.teaching_session
    teaching_rounds = session.answered_rounds()
    teacher_view_active = st.session_state.get("live_view_mode") == "教师/答辩视图"
    with st.sidebar:
        st.space("small")
        st.subheader("学习进度")
        if session.teaching_route:
            route = session.teaching_route
            route_total = len(route.steps)
            st.progress(
                route.completed_count() / route_total,
                text=f"教学路线 {route.completed_count()} / {route_total}",
            )
            st.caption(f"当前目标：{route.current_step().title}")
        st.caption(f"已完成 {teaching_rounds} 次回答")
        if session.state.misconceptions:
            st.warning("当前还有一个需要澄清的理解点。", icon=":material/psychology_alt:")
        if teacher_view_active:
            st.metric(
                "平均证据指数",
                f"{session.state.average_mastery():.0%}",
                chart_data=list(session.state.mastery.values()),
                chart_type="bar",
            )
            st.badge(f"当前阶段：{phase_label(session.state.phase)}", icon=":material/route:", color="blue")
            for point, value in session.state.mastery.items():
                knowledge_state = session.state.knowledge_states.get(point)
                level = knowledge_state.last_evidence_level if knowledge_state else "none"
                st.progress(value, text=f"{point}　{value:.0%} · {EVIDENCE_LABELS.get(level, level)}")
            st.caption(f"内部关注点：{session.state.next_focus}")
            current = session.turns[-1]
            with st.container(border=True):
                st.caption("本轮教学方案")
                content_id = current.content_skill_id or current.support_skill_id or "未匹配学科 Skill"
                strategy_id = current.strategy_skill_id or "学科 Skill 自带讲解策略"
                st.caption("教什么 · 内容 Skill")
                st.code(content_id, language=None)
                st.caption("怎么教 · 教学策略")
                st.code(strategy_id, language=None)

    header = st.container(horizontal=True, vertical_alignment="center", horizontal_alignment="distribute")
    with header:
        st.subheader(f"{session.goal.topic} · {session.profile.name}")
        st.badge(
            f"历史回答 {teaching_rounds} 次 · 本阶段 {session.rounds_in_current_run} / {max_rounds} 轮",
            color="blue",
        )
        status_label = STATUS_LABELS[str(session.status)]
        st.badge(status_label, color="green" if str(session.status) == "success" else "orange")
        with st.popover("会话操作", icon=":material/more_horiz:"):
            st.caption("当前会话已自动保存，可以安全地开始新任务。")
            st.download_button(
                "导出当前会话",
                data=session.model_dump_json(indent=2),
                file_name=f"session_{session.session_id}.json",
                mime="application/json",
                icon=":material/download:",
                width="stretch",
            )
            st.button("保存并新建会话", icon=":material/restart_alt:", on_click=reset_teaching_session, width="stretch")

    current = session.turns[-1]
    view_options = ["学生视图", "教师/答辩视图"]
    if st.session_state.get("live_view_mode") not in view_options:
        st.session_state.live_view_mode = "学生视图"
    view_mode = st.segmented_control(
        "查看模式",
        view_options,
        required=True,
        key="live_view_mode",
    )
    teacher_view = view_mode == "教师/答辩视图"
    if not teacher_view:
        with st.container(border=True):
            if session.teaching_route:
                route = session.teaching_route
                step = route.current_step()
                st.caption(f"当前目标 · {step.title}")
                st.progress(route.completed_count() / len(route.steps))
                st.write(step.learning_target)
            elif current.micro_step:
                st.caption(f"当前目标 · {current.micro_step.focus}")
            if current.micro_step:
                st.caption(f"回答方式：{response_mode_label(current.micro_step.response_mode)}")
    if teacher_view:
        st.subheader("本轮决策", help="Agent 根据最新学生回答更新状态，再选择本轮教学策略。")
        st.caption("以下内容用于答辩和调试；学生视图只保留教师话语与回答控件。")
        if session.teaching_route:
            with st.container(border=True):
                st.caption(
                    "会话教学路线 · "
                    + ("真实 LLM 规划" if session.teaching_route.source == "llm" else "目标字段回退")
                )
                route_row = st.container(horizontal=True, vertical_alignment="center")
                with route_row:
                    for route_step in session.teaching_route.steps:
                        color = "green" if route_step.status == "completed" else "blue" if route_step.status == "active" else "gray"
                        icon = ":material/check:" if route_step.status == "completed" else ":material/radio_button_checked:" if route_step.status == "active" else ":material/radio_button_unchecked:"
                        st.badge(route_step.title, color=cast(Literal["green", "blue", "gray"], color), icon=icon)
                st.caption(f"当前路线约束：{session.teaching_route.current_step().learning_target}")
        decision_flow = st.columns([1, 1, 1], border=True, vertical_alignment="top")
        with decision_flow[0]:
            st.badge("学生信号", icon=":material/psychology:", color="orange")
            if current.state_after.understanding_signals:
                st.write(current.state_after.understanding_signals[-1])
            else:
                st.write("等待更多回答证据")
        with decision_flow[1]:
            st.badge("本轮教学策略", icon=":material/route:", color="green")
            st.markdown(f"**{action_label(current.action_type)}**")
            st.caption(f"阶段：{phase_label(current.phase)}")
            if current.skill_plan:
                st.caption("内容 Skill")
                st.code(current.skill_plan.content_skill_id or "通用", language=None)
                st.caption("教学策略")
                st.code(current.skill_plan.strategy_skill_id or "学科内置", language=None)
            else:
                st.caption(f"策略 Skill：{current.selected_skill_id}")
        with decision_flow[2]:
            if current.action_type.startswith("terminate_"):
                st.badge("终止依据", icon=":material/flag:", color="orange")
                st.write(current.stop_decision or current.policy_rule)
            else:
                st.badge("下一步目标", icon=":material/flag:", color="blue")
                if current.micro_step:
                    st.write(f"本轮只确认：{current.micro_step.focus}")
                else:
                    st.write(current.state_after.next_focus)

        if current.switch_reason and len(session.turns) > 1:
            previous_turn = session.turns[-2]
            st.success(
                skill_switch_message(previous_turn, current),
                icon=":material/swap_horiz:",
            )

        if current.micro_step:
            with st.container(border=True):
                st.caption("本轮单步教学上下文")
                micro_columns = st.columns(3)
                with micro_columns[0]:
                    st.markdown("**本轮只解决**")
                    st.write(current.micro_step.focus)
                with micro_columns[1]:
                    st.markdown("**当前情境**")
                    st.write(current.micro_step.context)
                with micro_columns[2]:
                    st.markdown("**你只需要回答**")
                    st.write(current.micro_step.requested_target)
                st.caption(
                    f"回答方式：{response_mode_label(current.micro_step.response_mode)}"
                    + (f" · {current.micro_step.input_hint}" if current.micro_step.input_hint else "")
                )
                if session.profile.response_preference != "auto":
                    st.caption(
                        f"已按你的偏好固定为：{RESPONSE_PREFERENCE_LABELS[session.profile.response_preference]}"
                    )
                if (
                    current.micro_step.response_mode == "single_choice"
                    and not has_reliable_choice_options(current.micro_step)
                ):
                    st.warning(
                        "本轮没有生成出可靠的知识选项，因此不会把“我会/我不会”当作学习证据。"
                        "请在下方重新生成；该操作不增加轮次，也不改变掌握度。",
                        icon=":material/refresh:",
                    )
                if current.micro_step.response_mode == "single_choice" and current.micro_step.options:
                    st.caption("选项仅用于降低回答门槛；Agent 仍会根据你的选择和后续解释判断掌握度。")
        if current.student_message:
            if current.state_after.evidence:
                latest_evidence = current.state_after.evidence[-1]
                st.caption(
                    f"本轮状态证据：{latest_evidence.knowledge_point} · "
                    f"{EVIDENCE_LABELS.get(latest_evidence.evidence_level, latest_evidence.evidence_level)} · "
                    f"{latest_evidence.signal_type} · {latest_evidence.reason}"
                )
            else:
                st.caption("本轮状态证据：暂无可映射证据；本轮不提升掌握度，继续收集回答。")

    chat_heading = st.container(horizontal=True, vertical_alignment="center")
    with chat_heading:
        st.subheader("教学对话")
        st.caption("像聊天一样学习：每次只回答当前这一问，提交后 Agent 才会继续")

    for turn_index, turn in enumerate(session.turns):
        if turn.student_message:
            with st.chat_message("user", avatar=":material/person:"):
                st.write(turn.student_message)
        with st.chat_message("assistant", avatar=":material/school:"):
            visible_message = (
                turn.teacher_message
                if teacher_view
                else HybridTeachingAgent.student_visible_message(session, turn)
            )
            st.write(visible_message)
            if teacher_view and turn is current:
                with st.container(horizontal=True, vertical_alignment="center"):
                    st.badge(
                        f"内容 Skill · {turn.content_skill_id or '通用诊断模式'}",
                        color="gray",
                    )
                    st.badge(
                        f"教学策略 · {turn.strategy_skill_id or '通用策略'}",
                        color="green",
                    )
                    st.badge(f"教学动作 · {action_label(turn.action_type)}", color="blue")
            if teacher_view and turn.switch_reason and turn_index > 0 and turn is not current:
                previous_turn = session.turns[turn_index - 1]
                st.success(
                    skill_switch_message(previous_turn, turn),
                    icon=":material/swap_horiz:",
                )
            if teacher_view and turn is current:
                with st.expander(
                    "专业决策证据",
                    expanded=False,
                    icon=":material/manage_search:",
                ):
                    st.write(turn.selection_reason)
                    if turn.micro_step:
                        st.caption("单步教学计划")
                        st.json(turn.micro_step.model_dump(mode="json"), expanded=False)
                    if turn.teacher_review:
                        review = turn.teacher_review
                        review_status = "通过" if review.valid else "未通过/已回退"
                        st.caption(
                            f"输出复核：{review_status} · "
                            f"单步={review.one_step} · 情境唯一={review.one_context} · "
                            f"单问题={review.one_question} · 事实一致={review.fact_consistent} · "
                            f"上下文连续={review.same_context} · "
                            f"回答模式={review.response_mode_valid} · 选项={review.options_valid}"
                        )
                        if review.issues:
                            st.warning("；".join(review.issues))
                    if turn.generation_audit:
                        st.caption("生成审计：")
                        st.json(turn.generation_audit, expanded=False)
                    if turn.llm_trace:
                        st.caption("LLM 调用审计：")
                        st.dataframe(
                            [trace.model_dump(mode="json") for trace in turn.llm_trace],
                            hide_index=True,
                            column_config={
                                "latency_ms": st.column_config.NumberColumn("耗时（ms）"),
                                "attempts": st.column_config.NumberColumn("尝试次数"),
                            },
                        )
                    evidence_row = st.container(horizontal=True)
                    with evidence_row:
                        st.badge(f"决策来源 · {decision_mode_label(turn.decision_mode)}", color="blue")
                        if turn.policy_rule:
                            st.badge(f"规则 · {turn.policy_rule}", color="gray")
                    if turn.candidate_skill_ids:
                        st.info(
                            "**内容 Skill 资格符合：** "
                            + "、".join(f"`{skill_id}`" for skill_id in turn.candidate_skill_ids)
                            + "\n\n这里的“资格符合”只表示它适合提供学科内容，"
                            "最终采用的教学策略以上方绿色标签为准。",
                            icon=":material/menu_book:",
                        )
                    st.caption(
                        f"本轮教学方案：内容 Skill `{turn.content_skill_id or '通用诊断模式'}` × "
                        f"教学策略 `{turn.strategy_skill_id or '通用策略'}`。"
                    )
                    if turn.fallback_reason:
                        st.warning(turn.fallback_reason)
                    if turn.candidate_audit:
                        passed = [item for item in turn.candidate_audit if item.get("passed")]
                        rejected = [item for item in turn.candidate_audit if not item.get("passed")]
                        st.caption(f"内容 Skill 资格检查：{len(passed)} 个符合，{len(rejected)} 个不符合")
                        audit_rows = [
                            {
                                "Skill": item["skill_id"],
                                "资格判断": "符合" if item.get("passed") else "不符合",
                                "课程": (
                                    "—" if item.get("checks", {}).get("course") is None
                                    else "通过" if item.get("checks", {}).get("course") else "拒绝"
                                ),
                                "目标": (
                                    "—" if item.get("checks", {}).get("goal") is None
                                    else "通过" if item.get("checks", {}).get("goal") else "拒绝"
                                ),
                                "触发": (
                                    "—" if item.get("checks", {}).get("trigger") is None
                                    else "通过" if item.get("checks", {}).get("trigger") else "拒绝"
                                ),
                                "前置": (
                                    "—" if item.get("checks", {}).get("precondition") is None
                                    else "通过" if item.get("checks", {}).get("precondition") else "拒绝"
                                ),
                                "规则分": item.get("score", 0.0),
                                "依据": "；".join(item.get("reasons", [])),
                            }
                            for item in turn.candidate_audit
                        ]
                        st.dataframe(audit_rows, hide_index=True)
                    if turn.state_after.evidence:
                        latest = turn.state_after.evidence[-1]
                        st.caption(
                            f"状态证据：{latest.knowledge_point} · "
                            f"{EVIDENCE_LABELS.get(latest.evidence_level, latest.evidence_level)} · "
                            f"{latest.signal_type} · {latest.reason}"
                        )
                    if turn.student_message and turn.state_before.mastery:
                        st.caption("掌握度变化：")
                        changes = st.container(horizontal=True, gap="small")
                        for point, after_value in turn.state_after.mastery.items():
                            before_value = turn.state_before.mastery.get(point, after_value)
                            delta = after_value - before_value
                            if abs(delta) < 0.0005:
                                continue
                            knowledge_state = turn.state_after.knowledge_states.get(point)
                            level = knowledge_state.last_evidence_level if knowledge_state else "none"
                            with changes:
                                st.metric(
                                    point,
                                    f"{after_value:.0%}",
                                    delta=f"{delta:+.1%} · {EVIDENCE_LABELS.get(level, level)}",
                                )
                        if not any(
                            abs(after_value - turn.state_before.mastery.get(point, after_value)) >= 0.0005
                            for point, after_value in turn.state_after.mastery.items()
                        ):
                            st.caption("本轮没有产生可确认的掌握度提升；系统保留回答证据，等待下一次验证。")
                    if turn.switch_reason:
                        st.info(turn.switch_reason)
                    st.write(f"下一关注点：{turn.state_after.next_focus}")
                    if turn.state_after.understanding_signals:
                        st.write("理解信号：" + "；".join(turn.state_after.understanding_signals))
                    if turn.state_after.misconceptions:
                        st.write("误解证据：")
                        for item in turn.state_after.misconceptions:
                            st.caption(f"{item.label}（累计 {item.count} 次）— {item.evidence}")
                    st.write(f"终止判断：{turn.stop_decision}")

    if str(session.status) == "active":
        suggestion_key = f"ai_demo_replies_{session.session_id}_{teaching_rounds}"
        with st.expander("AI 推荐演示回答", expanded=False, icon=":material/auto_awesome:"):
            st.caption(
                "根据当前教师问题、教学路线和最近对话，由已配置的 LLM 生成。"
                "它只帮你准备学生回答，不会自动提交。"
            )
            if st.button(
                "生成 3 条 AI 推荐回答",
                icon=":material/auto_awesome:",
                key=f"generate_demo_replies_{session.session_id}_{teaching_rounds}",
            ):
                try:
                    with st.spinner("正在根据当前上下文生成推荐回答…"):
                        suggestions = DemoReplyGenerator(live_agent().llm).generate(session)
                    st.session_state[suggestion_key] = [item.model_dump(mode="json") for item in suggestions]
                    st.session_state[f"{suggestion_key}_selected"] = suggestions[0].suggestion_id
                    set_notice("已生成当前问题的 AI 推荐回答。", ":material/auto_awesome:")
                    st.rerun()
                except LLMUnavailableError as exc:
                    st.error(str(exc), icon=":material/cloud_off:")
                except Exception as exc:
                    st.error(f"推荐回答生成失败，本轮不会改变学生状态：{exc}", icon=":material/error:")

            raw_suggestions = st.session_state.get(suggestion_key, [])
            if raw_suggestions:
                suggestion_map = {item["suggestion_id"]: item for item in raw_suggestions}
                suggestion_ids = list(suggestion_map)
                selected_id = st.radio(
                    "选择一条回答",
                    suggestion_ids,
                    key=f"{suggestion_key}_selected",
                    format_func=lambda item_id: suggestion_map[item_id]["label"],
                )
                selected = suggestion_map[selected_id]
                st.info(selected["reply"], icon=":material/chat:")
                if teacher_view:
                    st.caption(f"演示信号（仅教师视图）：{selected.get('intended_signal', '未标注')}")
                if st.button(
                    "使用所选回答",
                    type="primary",
                    icon=":material/send:",
                    key=f"use_demo_reply_{session.session_id}_{teaching_rounds}",
                ):
                    st.session_state.pending_demo_reply = selected["reply"]
                st.caption("点击后会像真实学生提交一样进入 Agent；只有提交后才会更新状态和轮次。")
        pending_demo_reply = st.session_state.pop("pending_demo_reply", None)
        response_mode = current.micro_step.response_mode if current.micro_step else "open"
        submitted_reply = pending_demo_reply
        if submitted_reply is None and response_mode == "single_choice" and current.micro_step:
            options = current.micro_step.options
            if not has_reliable_choice_options(current.micro_step):
                st.error(
                    "当前没有可提交的知识选项。重新生成成功前，本轮不会写入学生回答。",
                    icon=":material/block:",
                )
                if st.button(
                    "重新生成知识选项",
                    type="primary",
                    icon=":material/refresh:",
                    key=f"regenerate_choice_{session.session_id}_{teaching_rounds}",
                ):
                    try:
                        with st.spinner("正在重新生成与当前知识点匹配的选项…"):
                            st.session_state.teaching_session = live_agent().regenerate_current_turn(session)
                        set_notice("已重新生成当前题目；轮次和学生状态保持不变。")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"重新生成失败，原会话仍已保留：{exc}", icon=":material/error:")
                if st.button(
                    "本轮改用开放回答",
                    icon=":material/edit:",
                    key=f"fallback_open_{session.session_id}_{teaching_rounds}",
                    help="只改变当前题型，不改变你为后续教学选择的回答偏好。",
                ):
                    try:
                        with st.spinner("正在保留当前情境并改写为开放问题…"):
                            st.session_state.teaching_session = live_agent().regenerate_current_turn(
                                session,
                                response_mode_override="open",
                            )
                        set_notice("当前轮已改用开放回答；学生状态和轮次保持不变。")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"切换回答方式失败，原会话仍已保留：{exc}", icon=":material/error:")
            else:
                option_labels = [f"{option.option_id}. {option.text}" for option in options]
                selected_label = st.radio(
                    "请选择最符合你当前理解的一项",
                    option_labels,
                    key=f"choice_{session.session_id}_{teaching_rounds}",
                )
                if st.button(
                    "提交选择",
                    type="primary",
                    icon=":material/check:",
                    key=f"submit_choice_{session.session_id}_{teaching_rounds}",
                ):
                    selected_index = option_labels.index(selected_label)
                    selected_option = options[selected_index]
                    submitted_reply = f"选择 {selected_option.option_id}：{selected_option.text}"
        elif submitted_reply is None and response_mode in {"fill_blank", "numeric"}:
            answer = st.text_input(
                "请输入答案",
                key=f"guided_answer_{session.session_id}_{teaching_rounds}",
                placeholder=current.micro_step.input_hint if current.micro_step else "输入答案",
            )
            if st.button(
                "提交答案",
                type="primary",
                icon=":material/check:",
                key=f"submit_answer_{session.session_id}_{teaching_rounds}",
            ):
                if answer.strip():
                    submitted_reply = answer.strip()
                else:
                    st.warning("请先输入答案，再提交。", icon=":material/edit:")
        elif submitted_reply is None:
            prompt = st.chat_input("输入学生的真实回答…", key="student_reply", submit_mode="disable")
            submitted_reply = prompt
        if submitted_reply:
            try:
                with st.spinner("正在分析回答、更新状态并选择下一步…"):
                    st.session_state.teaching_session = live_agent().handle_student_message(session, submitted_reply.strip())
                set_notice("已记录学生回答和本轮决策。")
                st.rerun()
            except Exception as exc:
                st.error(f"本轮未提交成功，原会话仍已保留：{exc}", icon=":material/error:")
    elif str(session.status) == "success":
        st.success(f"教学成功：{session.termination_reason}", icon=":material/task_alt:")
    else:
        st.warning(f"教学暂停：{session.termination_reason}", icon=":material/pause_circle:")
        max_round_stop = current.policy_rule == "max_rounds"
        st.caption(
            "已保存最后一轮的学习证据，可从当前关注点继续。"
            if max_round_stop
            else "纠错后仍未出现改善证据，建议换一种讲法继续。"
        )
        with st.container(horizontal=True, vertical_alignment="center"):
            if st.button(
                "从当前进度继续" if max_round_stop else "换一种讲法继续",
                type="primary",
                icon=":material/replay:",
                help=(
                    "创建补充会话，继承当前掌握度和证据。"
                    if max_round_stop
                    else "创建补救会话，继承当前掌握度和证据，但重置连续失败计数。"
                ),
            ):
                st.session_state.teaching_session = live_agent().resume_session(
                    session,
                    reset_misconceptions=not max_round_stop,
                )
                set_notice("已在原会话中继续：历史对话、路线和学习证据均已保留。")
                st.rerun()
            st.button(
                "开始全新任务",
                icon=":material/add_circle:",
                on_click=reset_teaching_session,
            )
