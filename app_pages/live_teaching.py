from __future__ import annotations

from typing import Literal, cast

import streamlit as st

from src.agent import HybridTeachingAgent
from src.config import get_agent_settings
from src.demo_reply_generator import DEMO_SIGNAL_LABELS, DEMO_TARGET_LABELS, DemoReplyGenerator
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
        "points": "导数的极限定义",
        "prior": "函数,斜率,平均变化率,平均速度",
        "skill_id": "derivative_intro_via_slope_limit_v1",
    },
    "牛顿第一定律": {
        "course": "大学物理",
        "topic": "牛顿第一定律",
        "objective": "用公交车急刹车解释惯性，并区分保持运动与改变运动状态",
        "points": "惯性与运动状态变化",
        "prior": "速度,力,运动,惯性",
        "skill_id": "newtons_first_law_via_engineering_examples_v1",
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

DIFFICULTY_LABELS = {
    "concept_misconception": "概念误解",
    "symbol_notation": "符号/公式困难",
    "calculation": "计算/代入困难",
    "task_comprehension": "题意理解困难",
    "unknown": "暂未判断",
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
        st.session_state.start_available_skills = [preset["skill_id"]]
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
        "start_available_skills": [preset["skill_id"]],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if st.session_state.get("start_level") not in {"基础薄弱", "中等", "较好"}:
        st.session_state.start_level = "中等"


def execution_role_label(turn: ConversationTurn) -> str:
    content_id = turn.skill_plan.content_skill_id if turn.skill_plan else turn.content_skill_id
    strategy_id = turn.skill_plan.strategy_skill_id if turn.skill_plan else turn.strategy_skill_id
    if turn.selected_skill_id == strategy_id:
        return f"教学策略（{strategy_id or '内容 Skill 自主讲解'}）"
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
        plan_changes.append(f"内容：{content_before or '通用'} → {content_after or '通用'}")
    if strategy_before != strategy_after:
        plan_changes.append(f"策略：{strategy_before or '未指定'} → {strategy_after or '未指定'}")
    if plan_changes:
        return "**Skill 已切换** · " + "；".join(plan_changes)
    if previous.selected_skill_id != current.selected_skill_id:
        return f"**Skill 已切换** · {execution_role_label(previous)} → {execution_role_label(current)}"
    return "**执行方式已调整**"


page_header(
    "实时教学",
    "老师根据你的回答自主组织教学，并根据状态决定下一步。",
    eyebrow="自适应教学",
    icon="forum",
)
inject_live_chat_css()
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
        with st.expander("高级设置", icon=":material/tune:"):
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
        "牛顿第一定律": "生活情境直观，适合展示诊断、支架、纠错和迁移。",
        "导数极限定义": "从平均变化率过渡到瞬时变化率，适合展示数学迁移。",
    }
    with st.container(border=True):
        st.markdown(f"**现场演示建议 · {preset_name}**")
        st.caption(demo_notes[preset_name])

    if submitted:
        points = [item.strip() for item in points_text.replace("，", ",").split(",") if item.strip()]
        missing = [label for label, value in (("教学主题", topic.strip()), ("教学目标", objective.strip()), ("知识点", points), ("学生姓名", student_name.strip())) if not value]
        if missing:
                    st.error("请补全：" + "、".join(missing), icon=":material/error:")
        else:
            try:
                preset_skill_id = preset.get("skill_id")
                if preset_skill_id and available_skill_ids == [preset_skill_id] and (
                    str(course).strip() != preset["course"] or topic.strip() != preset["topic"]
                ):
                    # A manually edited course/topic should not keep a stale
                    # demo-only Skill restriction from the old preset.
                    available_skill_ids = []
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
    # Keep the form's key field available for browser back/forward navigation.
    st.session_state.setdefault("start_level", st.session_state.teaching_session.profile.level)
    session = st.session_state.teaching_session
    teaching_rounds = session.answered_rounds()
    with st.sidebar:
        st.space("small")
        st.subheader("学习进度")
        if session.teaching_route:
            route = session.teaching_route
            route_total = len(route.steps)
            st.progress(
                route.completed_count() / route_total,
                text=f"路线 {route.completed_count()}/{route_total} · {route.current_step().title}",
            )
        if session.state.misconceptions:
            st.warning("当前还有一个需要澄清的理解点。", icon=":material/psychology_alt:")
        st.metric(
            "平均证据指数",
            f"{session.state.average_mastery():.0%}",
            chart_data=list(session.state.mastery.values()),
            chart_type="bar",
        )
        st.badge(f"当前阶段：{phase_label(session.state.phase)}", icon=":material/route:", color="blue")
        with st.expander("掌握度详情", expanded=False, icon=":material/insights:"):
            for point, value in session.state.mastery.items():
                knowledge_state = session.state.knowledge_states.get(point)
                level = knowledge_state.last_evidence_level if knowledge_state else "none"
                st.progress(value, text=f"{point}　{value:.0%} · {EVIDENCE_LABELS.get(level, level)}")

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
    with st.container(border=True):
        if session.teaching_route:
            route = session.teaching_route
            step = route.current_step()
            st.progress(
                route.completed_count() / len(route.steps),
                text=f"进度 {route.completed_count()}/{len(route.steps)}",
            )
            st.caption(f"当前教学目标：{step.title}")
        elif current.micro_step:
            st.caption(f"当前教学目标：{current.micro_step.focus}")
        if current.micro_step and current.micro_step.response_mode != "open":
            st.caption(f"回答方式：{response_mode_label(current.micro_step.response_mode)}")

    chat_heading = st.container(horizontal=True, vertical_alignment="center")
    with chat_heading:
        st.subheader("教学对话")

    for turn_index, turn in enumerate(session.turns):
        if turn.student_message:
            with st.chat_message("user", avatar=":material/person:"):
                st.write(turn.student_message)
        with st.chat_message("assistant", avatar=":material/school:"):
            st.write(turn.teacher_message)
            content_id = (
                turn.content_skill_id
                or (turn.skill_plan.content_skill_id if turn.skill_plan else None)
                or "未匹配内容 Skill"
            )
            with st.container(horizontal=True, vertical_alignment="center"):
                st.badge(f"内容 Skill · {content_id}", color="gray")
                st.badge(f"教学动作 · {action_label(turn.action_type)}", color="green")
            if turn.switch_reason and turn_index > 0:
                previous_turn = session.turns[turn_index - 1]
                st.success(
                    skill_switch_message(previous_turn, turn),
                    icon=":material/swap_horiz:",
                )
            with st.expander(
                f"本轮教学证据 · {action_label(turn.action_type)}",
                expanded=False,
                icon=":material/manage_search:",
            ):
                if turn.state_after.evidence:
                    latest = turn.state_after.evidence[-1]
                    st.caption(
                        f"证据：{latest.knowledge_point} · "
                        f"{EVIDENCE_LABELS.get(latest.evidence_level, latest.evidence_level)}"
                    )
                st.caption(
                    "困难识别："
                    + DIFFICULTY_LABELS.get(turn.difficulty_type, turn.difficulty_type or "暂未判断")
                )
                if turn.student_message and turn.state_before.mastery:
                    changes = st.container(horizontal=True, gap="small")
                    has_change = False
                    for point, after_value in turn.state_after.mastery.items():
                        before_value = turn.state_before.mastery.get(point, after_value)
                        delta = after_value - before_value
                        if abs(delta) < 0.0005:
                            continue
                        has_change = True
                        knowledge_state = turn.state_after.knowledge_states.get(point)
                        level = knowledge_state.last_evidence_level if knowledge_state else "none"
                        with changes:
                            st.metric(
                                point,
                                f"{after_value:.0%}",
                                delta=f"{delta:+.1%} · {EVIDENCE_LABELS.get(level, level)}",
                            )
                    if not has_change:
                        st.caption("掌握度暂未变化，等待下一次验证。")
                with st.expander("查看详细记录", expanded=False):
                    st.caption(f"选择理由：{turn.selection_reason or '根据当前状态自动选择。'}")
                    st.caption(f"决策来源：{decision_mode_label(turn.decision_mode)}")
                    if turn.micro_step:
                        st.caption(f"关注：{turn.micro_step.focus}")
                        st.caption(f"情境：{turn.micro_step.context}")
                        st.caption(f"回答方式：{response_mode_label(turn.micro_step.response_mode)}")
                    st.caption(f"下一关注点：{turn.state_after.next_focus}")
                    if turn.state_after.understanding_signals:
                        st.caption("理解信号：" + "；".join(turn.state_after.understanding_signals))
                    if turn.state_after.misconceptions:
                        st.caption(
                            "误解证据："
                            + "；".join(f"{item.label}（{item.count} 次）" for item in turn.state_after.misconceptions)
                        )
                if turn.action_type.startswith("terminate_"):
                    st.caption(f"终止判断：{turn.stop_decision or turn.policy_rule}")

    if str(session.status) == "active":
        with st.expander("AI 推荐演示回答", expanded=False, icon=":material/auto_awesome:"):
            target_signal = st.selectbox(
                "本次希望演示的状态",
                list(DEMO_TARGET_LABELS),
                format_func=lambda value: DEMO_TARGET_LABELS[value],
                key=f"demo_target_signal_{session.session_id}_{teaching_rounds}",
            )
            st.caption("这是给演示者的生成提示，不会强制 Agent 采用对应动作；提交后仍由模型正常判断。")
            suggestion_key = (
                f"ai_demo_replies_{session.session_id}_{teaching_rounds}_{target_signal}"
            )
            if st.button(
                "生成 3 条 AI 推荐回答",
                icon=":material/auto_awesome:",
                key=f"generate_demo_replies_{session.session_id}_{teaching_rounds}",
            ):
                try:
                    with st.spinner("正在根据当前上下文生成推荐回答…"):
                        suggestions = DemoReplyGenerator(live_agent().llm).generate(
                            session,
                            target_signal=target_signal,
                        )
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
                    format_func=lambda item_id: (
                        f"{DEMO_SIGNAL_LABELS.get(suggestion_map[item_id].get('intended_signal', ''), '演示回答')}"
                        f" · {suggestion_map[item_id]['label']}"
                    ),
                )
                selected = suggestion_map[selected_id]
                st.caption(
                    "模型标注的演示意图："
                    + DEMO_SIGNAL_LABELS.get(selected.get("intended_signal", ""), "未标注")
                )
                st.info(selected["reply"], icon=":material/chat:")
                if st.button(
                    "使用所选回答",
                    type="primary",
                    icon=":material/send:",
                    key=f"use_demo_reply_{session.session_id}_{teaching_rounds}",
                ):
                    st.session_state.pending_demo_reply = selected["reply"]
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
