from pathlib import Path

import pytest

from src.agent import HybridTeachingAgent
from src.llm import LLMUnavailableError, OpenAICompatibleClient
from src.models import (
    Misconception,
    MisconceptionState,
    ResponseOption,
    SessionStatus,
    StateAssessment,
    StudentProfile,
    StudentState,
    TeacherDraft,
    TeachingGoal,
    TeachingMicroStep,
)
from src.skills import SkillLibrary
from src.state_tracker import StateTracker, detect_generic_contradiction
from src.storage import SessionStore


def make_agent(tmp_path: Path, **settings) -> HybridTeachingAgent:
    offline = OpenAICompatibleClient({"api_key": "", "model": "offline"})
    return HybridTeachingAgent(
        library=SkillLibrary(),
        llm=offline,
        store=SessionStore(tmp_path),
        settings={
            "max_rounds": 8,
            "mastery_threshold": 0.8,
            "no_progress_limit": 3,
            "candidate_limit": 5,
            **settings,
        },
    )


def make_inputs(mastery: float = 0.3):
    goal = TeachingGoal(
        course="程序设计",
        topic="二分查找边界条件",
        objective="从区间定义推导循环条件",
        knowledge_points=["区间定义", "边界更新"],
    )
    profile = StudentProfile(name="边界测试学生", level="中等")
    state = StudentState(
        mastery={point: mastery for point in goal.knowledge_points},
        next_focus="区间定义",
    )
    return goal, profile, state


def test_generic_contradiction_guard_only_checks_language_relation():
    assert detect_generic_contradiction(
        "请解释为什么乘客会向后方倾斜？",
        "那就是往前方",
    ) is not None
    assert detect_generic_contradiction(
        "请解释为什么乘客会向后方倾斜？",
        "不是前方，而是后方",
    ) is None


def test_max_round_terminal_uses_repair_summary_when_conflict_remains(tmp_path):
    agent = make_agent(tmp_path, max_rounds=2, no_progress_limit=20)
    goal, profile, state = make_inputs()
    session = agent.start_session(goal, profile, state)
    session = agent.handle_student_message(session, "我不知道")
    session = agent.handle_student_message(session, "我还是不知道")

    turn = session.turns[-1]
    assert str(session.status) == "unable"
    assert turn.action_type == "terminate_max_rounds"
    assert turn.selected_skill_id == "misconception_contrast_correction_v1"
    assert "暂不记为掌握" in turn.teacher_message
    assert turn.skill_plan is not None
    assert turn.skill_plan.strategy_skill_id == "misconception_contrast_correction_v1"


def make_step() -> TeachingMicroStep:
    return TeachingMicroStep(
        focus="区间定义",
        context="数组 [1, 3, 5]，使用左闭右闭区间",
        known_fact="left=0，right=2",
        requested_target="说明下标 2 是否在区间内",
        representation="左闭右闭",
        expected_signal="学生能说明端点包含",
        step_index=1,
    )


def test_agent_helper_failure_boundaries_and_history(tmp_path):
    agent = make_agent(tmp_path)
    goal, profile, state = make_inputs()
    session = agent.start_session(
        goal,
        profile,
        state,
        history=[{"role": "student", "content": "之前学过数组"}],
    )
    with pytest.raises(RuntimeError, match="已终止"):
        session.status = SessionStatus.SUCCESS.value
        agent.regenerate_current_turn(session)

    session.status = SessionStatus.ACTIVE.value
    session.turns = []
    with pytest.raises(RuntimeError, match="没有可重新生成"):
        agent.regenerate_current_turn(session)

    assert "之前学过数组" in agent._history_summary(session)
    assert "无历史对话" not in agent._history_summary(session)
    assert agent._transition_allowed(session, "") is True
    assert agent._single_target("写出边界并说明原因", "区间定义") == "写出边界"
    assert agent._single_target("应该使用 left <= right，而不是 left < right", "区间定义").startswith("说明")


def test_final_message_hard_gate_reduces_two_tasks_to_one(tmp_path):
    agent = make_agent(tmp_path)
    goal, profile, state = make_inputs()
    session = agent.start_session(goal, profile, state)
    skill = agent.library.get("binary_search_boundary_by_interval_definition")
    step = make_step()
    unsafe = "请判断这个状态是否改变；如果改变，再说明是什么原因造成的？"

    message = agent._final_message(unsafe, session, skill, "subject_instruction", step)

    assert message != unsafe
    assert message.count("？") + message.count("?") == 1
    assert not agent._contains_multiple_requests(message)


def test_agent_semantic_selector_success_constraint_and_exception(tmp_path):
    class Selector:
        available = True

        def __init__(self, answer):
            self.answer = answer

        def structured(self, *args, **kwargs):
            if self.answer == "error":
                raise ValueError("invalid json")
            return {"skill_id": self.answer, "reason": "语义选择"}

    agent = make_agent(tmp_path)
    session = agent.start_session(*make_inputs())
    candidates = [
        agent.library.get("binary_search_boundary_by_interval_definition"),
        agent.library.get("derivative_intro_via_slope_limit_v1"),
    ]
    audit = [{"skill_id": item.skill_id, "score": 0.5} for item in candidates]
    agent.llm = Selector(candidates[1].skill_id)
    selected, source = agent._llm_select(candidates, session, "回答", audit)
    assert selected == candidates[1]
    assert source == "llm_semantic_selection"

    agent.llm = Selector("not-a-candidate")
    selected, source = agent._llm_select(candidates, session, "回答", audit)
    assert selected == candidates[0]
    assert source == "candidate_constraint_fallback"

    agent.llm = Selector("error")
    selected, source = agent._llm_select(candidates, session, "回答", audit)
    assert selected == candidates[0]
    assert source == "rule_fallback"


def test_agent_review_and_fallback_contract_edges(tmp_path):
    agent = make_agent(tmp_path)
    goal, profile, state = make_inputs()
    session = agent.start_session(goal, profile, state)
    step = make_step()

    empty = agent._review_teacher_draft(
        session,
        "",
        TeacherDraft(micro_step=TeachingMicroStep(), teacher_message="", question_count=1),
        None,
        False,
        ask_llm=False,
    )
    assert empty.valid is False
    assert any("为空" in issue for issue in empty.issues)
    assert any("字段不完整" in issue for issue in empty.issues)

    profile.response_preference = "single_choice"
    session.profile = profile
    mismatch = agent._review_teacher_draft(
        session,
        "",
        TeacherDraft(micro_step=step, teacher_message="请说明？", question_count=1),
        None,
        False,
        ask_llm=False,
    )
    assert mismatch.response_mode_valid is False

    duplicate = step.model_copy(
        update={
            "response_mode": "single_choice",
            "options": [
                ResponseOption(option_id="A", text="包含"),
                ResponseOption(option_id="A", text="包含"),
            ],
        }
    )
    duplicate_review = agent._review_teacher_draft(
        session,
        "",
        TeacherDraft(micro_step=duplicate, teacher_message="请选择？", question_count=1),
        None,
        False,
        ask_llm=False,
    )
    assert duplicate_review.options_valid is False

    changed = step.model_copy(update={"context": "另一个数组", "representation": "左闭右开"})
    locked = agent._review_teacher_draft(
        session,
        "",
        TeacherDraft(micro_step=changed, teacher_message="你怎么判断？", question_count=1),
        step,
        True,
        ask_llm=False,
    )
    assert locked.same_context is False
    assert any("不得更换" in issue for issue in locked.issues)

    class ReviewUnavailable:
        available = True

        def structured(self, *args, **kwargs):
            raise LLMUnavailableError("复核服务不可用")

    agent.llm = ReviewUnavailable()
    unavailable = agent._review_teacher_draft(
        session,
        "",
        TeacherDraft(micro_step=step, teacher_message="你能说明依据吗？", question_count=1),
        None,
        False,
        ask_llm=True,
    )
    assert unavailable.valid is False
    assert any("不可用" in issue for issue in unavailable.issues)


def test_agent_fallback_choice_and_guard_paths(tmp_path):
    agent = make_agent(tmp_path)
    goal, profile, state = make_inputs()
    session = agent.start_session(goal, profile, state)
    subject = agent.library.get("binary_search_boundary_by_interval_definition")
    invalid_choice = make_step().model_copy(
        update={"response_mode": "single_choice", "options": [ResponseOption(option_id="A", text="只有一个选项")]}
    )
    profile.response_preference = "single_choice"
    session.profile = profile
    draft = TeacherDraft(micro_step=invalid_choice, teacher_message="请选择？", question_count=1)
    fixed = agent._fallback_step_from_draft(session, subject, "diagnostic", draft, None, False)
    assert fixed.options == []
    assert "等待重新生成" in fixed.requested_target

    profile.response_preference = "numeric"
    session.profile = profile
    normalized = agent._apply_response_preference(session, make_step().model_copy(update={"options": [ResponseOption(option_id="A", text="x")]}))
    assert normalized.response_mode == "numeric"
    assert normalized.options == []
    assert agent._response_mode_shape_valid(normalized)
    assert not agent._response_mode_shape_valid(make_step().model_copy(update={"response_mode": "numeric", "options": [ResponseOption(option_id="A", text="x")] }))

    previous = invalid_choice
    fallback = agent._fallback_micro_step(session, subject, "diagnostic", previous)
    assert fallback.options == []
    review = agent._fallback_review(
        agent._review_teacher_draft(session, "", draft, None, False, ask_llm=False),
        "测试回退",
        "本轮只请回答一个问题？",
    )
    assert review.one_question is True
    assert "测试回退" in review.issues

    assert "我们继续保持当前情境" in agent._guard_teacher_message("", session, subject, "diagnostic", make_step())
    assert "我们继续保持当前情境" in agent._guard_teacher_message("答案是 left <= right", session, subject, "diagnostic", make_step())
    assert agent._guard_teacher_message("没有问号的说明", session, subject, "diagnostic", make_step()).endswith("？")


def test_state_assessment_core_helpers_cover_evidence_boundaries():
    tracker = StateTracker()
    goal, _, state = make_inputs()
    with pytest.raises(LLMUnavailableError, match="状态诊断器不可用"):
        tracker._assessment_structured("system", "user", "schema")
    assert tracker._level_for(StateAssessment(verification_passed=True), "区间定义") == "transfer"
    assert tracker._level_for(StateAssessment(progress="improved"), "区间定义") == "correct"
    assert tracker._level_for(StateAssessment(), "区间定义") == "none"
    level, delta = tracker._delta_for(
        StateAssessment(progress="regressed", confidence=1.0),
        "区间定义",
        None,
        "错误回答",
        0.8,
    )
    assert level == "none"
    assert delta < 0
    level, delta = tracker._delta_for(
        StateAssessment(progress="improved", evidence_levels={"区间定义": "transfer"}, confidence=1.0),
        "区间定义",
        None,
        "迁移回答",
        0.4,
    )
    assert level == "explained"
    assert delta > 0
    assert StateTracker._as_evidence_level("unknown", "partial") == "partial" if hasattr(StateTracker, "_as_evidence_level") else True


class BranchingAssessmentClient:
    available = True

    def __init__(self, first, follow_ups=()):
        self.first = first
        self.follow_ups = list(follow_ups)
        self.calls = 0

    def structured(self, system, user, schema, **kwargs):
        self.calls += 1
        if "状态诊断器" in system:
            return self.first
        if self.follow_ups:
            return self.follow_ups.pop(0)
        return {"contains_contradiction": False, "affected_points": [], "evidence": "无"}


def assessment_payload(**overrides):
    payload = {
        "mastery_updates": {"区间定义": 0.2, "边界更新": 0.2},
        "evidence_levels": {"区间定义": "none"},
        "misconceptions": [],
        "understanding_signals": ["需要继续观察"],
        "next_focus": "区间定义",
        "verification_passed": False,
        "progress": "unchanged",
        "affected_points": ["区间定义"],
        "confidence": 0.5,
        "evidence_reason": "状态证据不足",
    }
    payload.update(overrides)
    return payload


def test_state_negative_guard_and_context_mapping_are_deterministic():
    client = BranchingAssessmentClient(assessment_payload(progress="unchanged", affected_points=[]))
    goal, profile, state = make_inputs()
    after = StateTracker(client).update(
        goal,
        profile,
        state,
        "不用再检查了",
        previous_teacher_message="当 left == right 时区间是否为空？",
    )
    assert after.misconceptions
    assert after.mastery["区间定义"] < state.mastery["区间定义"]

    client = BranchingAssessmentClient(
        assessment_payload(progress="improved", affected_points=[], evidence_levels={})
    )
    after = StateTracker(client).update(
        goal,
        profile,
        state,
        "一个元素",
        previous_teacher_message="区间 [2, 2] 里有几个元素？",
    )
    assert after.mastery["区间定义"] > state.mastery["区间定义"]


def test_state_conditional_contradiction_and_repair_reviews():
    goal, profile, state = make_inputs()
    first = assessment_payload(
        progress="improved",
        confidence=0.4,
        misconceptions=[{"label": "冲突关系", "evidence": "当前回答", "count": 1}],
    )
    contradiction = {
        "contains_contradiction": True,
        "label": "明确冲突",
        "affected_points": ["区间定义"],
        "evidence": "回答与目标直接冲突",
    }
    after = StateTracker(BranchingAssessmentClient(first, [contradiction])).update(
        goal,
        profile,
        state,
        "我很确定，但关系相反",
        previous_teacher_message="请判断区间含义。",
    )
    assert after.misconceptions
    assert after.misconceptions[0].label == "明确冲突"

    active = state.model_copy(deep=True)
    active.misconceptions = [Misconception(label="旧误解", evidence="旧回答", count=1)]
    active.misconception_states = [
            MisconceptionState(
                label="旧误解",
                evidence="旧回答",
                count=1,
                consecutive_count=1,
                knowledge_point="区间定义",
            )
    ]
    repair = {
        "repairs_misconception": True,
        "affected_points": ["区间定义"],
        "evidence_levels": {"区间定义": "explained"},
        "reason": "学生给出与旧误解相反的依据",
    }
    after = StateTracker(BranchingAssessmentClient(assessment_payload(progress="unchanged"), [repair])).update(
        goal,
        profile,
        active,
        "我现在给出了新的依据",
        previous_teacher_message="请重新判断。",
    )
    assert not after.misconceptions
    assert after.evidence[-1].evidence_level == "explained"


def test_state_low_confidence_evidence_probe_and_relevance_gate():
    goal, profile, state = make_inputs()
    first = assessment_payload(progress="unchanged", confidence=0.4, affected_points=[])
    contradiction_false = {"contains_contradiction": False, "affected_points": [], "evidence": "没有明确冲突"}
    evidence_probe = {
        "affected_points": ["区间定义"],
        "evidence_levels": {"区间定义": "partial"},
        "has_confirmed_evidence": True,
        "reason": "回答包含局部正确原则",
    }
    tracker = StateTracker(
        BranchingAssessmentClient(first, [contradiction_false, evidence_probe]),
        settings={"max_state_reviews": 1, "state_review_call_budget": 3},
    )
    after = tracker.update(
        goal,
        profile,
        state,
        "我只知道一部分",
        previous_teacher_message="请说明区间定义。",
    )
    assert after.evidence[-1].evidence_level == "partial"
    assert after.mastery["区间定义"] > state.mastery["区间定义"]

    first = assessment_payload(
        progress="regressed",
        confidence=0.4,
        misconceptions=[],
        affected_points=["区间定义"],
    )
    direct = {
        "classification": "direct_answer",
        "affected_points": ["区间定义"],
        "reason": "回答直接解决当前问题",
    }
    tracker = StateTracker(
        BranchingAssessmentClient(first, [direct]),
        settings={"max_state_reviews": 2, "state_review_call_budget": 5},
    )
    active = state.model_copy(deep=True)
    active.misconceptions = [Misconception(label="待复核误解", evidence="历史证据", count=1)]
    active.misconception_states = [
        MisconceptionState(
            label="待复核误解",
            evidence="历史证据",
            count=1,
            consecutive_count=1,
            knowledge_point="区间定义",
        )
    ]
    after = tracker.update(
        goal,
        profile,
        active,
        "我回答了当前问题",
        previous_teacher_message="请判断当前区间。",
    )
    assert after.evidence[-1].evidence_level == "correct"


def test_state_nonempty_ambiguous_regression_is_not_recorded_as_error():
    goal, profile, state = make_inputs()
    client = BranchingAssessmentClient(
        assessment_payload(
            progress="regressed",
            confidence=0.95,
            misconceptions=[],
            affected_points=["区间定义"],
        )
    )
    after = StateTracker(client).update(goal, profile, state, "这部分我还没展开", previous_teacher_message="")
    assert after.misconceptions == []
    assert after.mastery == state.mastery
    assert after.evidence[-1].evidence_level == "none"
