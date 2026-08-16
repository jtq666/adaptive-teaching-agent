import json
from concurrent.futures import ThreadPoolExecutor

from src.agent import HybridTeachingAgent
from src.evaluation import load_cases
from src.llm import OpenAICompatibleClient
from src.models import StudentProfile, StudentState, TeachingGoal, TeachingSession
from src.skills import SkillLibrary
from src.storage import SessionStore


def inputs():
    goal = TeachingGoal(
        course="程序设计",
        topic="二分查找边界",
        objective="从区间定义推导边界",
        knowledge_points=["区间定义", "循环不变量", "边界更新"],
    )
    profile = StudentProfile(name="测试学生", prior_knowledge=["while 循环"])
    state = StudentState(mastery={point: 0.35 for point in goal.knowledge_points})
    return goal, profile, state


def test_hard_filter_emits_pass_and_reject_audit():
    goal, profile, state = inputs()
    candidates, audit = SkillLibrary().candidates(goal, state, profile=profile)
    assert candidates
    assert all(skill.skill_type == "subject" for skill in candidates)
    assert any(item["passed"] for item in audit)
    assert any(not item["passed"] for item in audit)


def test_hard_filter_rejects_cross_course_keyword_false_positives():
    goal = TeachingGoal(
        course="大学物理",
        topic="电磁感应",
        objective="理解磁通量变化如何产生感应电动势",
        knowledge_points=["磁通量", "感应电动势"],
    )
    state = StudentState(mastery={point: 0.3 for point in goal.knowledge_points})
    candidates, audit = SkillLibrary().candidates(
        goal,
        state,
        profile=StudentProfile(prior_knowledge=["力", "速度"]),
    )
    assert candidates == []
    rejected = {item["skill_id"]: item for item in audit}
    assert "课程不匹配" in rejected["derivative_limit_definition_v1"]["reasons"]
    assert "教学目标不匹配" in rejected["newtons_first_law_via_engineering_examples_v1"]["reasons"]


def test_hard_filter_rejects_unrelated_topic_in_same_course_and_agent_falls_back(tmp_path):
    goal = TeachingGoal(
        course="程序设计",
        topic="递归与回溯",
        objective="理解递归终止条件和调用栈",
        knowledge_points=["递归终止条件", "调用栈"],
    )
    profile = StudentProfile(prior_knowledge=["循环", "数组"])
    state = StudentState(mastery={point: 0.3 for point in goal.knowledge_points})
    library = SkillLibrary()
    candidates, audit = library.candidates(goal, state, profile=profile)
    assert candidates == []
    binary = next(item for item in audit if item["skill_id"] == "binary_search_boundary_by_interval_definition")
    assert "教学目标不匹配" in binary["reasons"]

    agent = HybridTeachingAgent(
        library=library,
        llm=OpenAICompatibleClient({"api_key": ""}),
        store=SessionStore(tmp_path),
    )
    session = agent.start_session(goal, profile, state)
    assert session.turns[-1].selected_skill_id == "diagnostic_questioning_v1"
    assert "无学科 Skill" in session.turns[-1].fallback_reason


def test_shared_knowledge_point_cannot_jump_to_adjacent_newton_law():
    goal = TeachingGoal(
        course="大学物理",
        topic="牛顿第一定律与惯性",
        objective="区分维持运动和改变运动状态所需的力",
        knowledge_points=["惯性", "合力", "运动状态"],
    )
    profile = StudentProfile(prior_knowledge=["速度", "力", "运动"])
    state = StudentState(mastery={"惯性": 0.35, "合力": 0.5, "运动状态": 0.45})
    candidates, audit = SkillLibrary().candidates(
        goal,
        state,
        "力不是维持速度的原因，合力为零时物体可以保持匀速直线运动。",
        profile=profile,
    )
    assert [skill.skill_id for skill in candidates] == [
        "newtons_first_law_via_engineering_examples_v1"
    ]
    second_law = next(item for item in audit if item["skill_id"] == "newtons_second_law_intro_v1")
    assert "教学目标不匹配" in second_law["reasons"]


def test_topic_title_prevents_neighboring_skill_from_shared_objective_term():
    goal = TeachingGoal(
        course="大学物理",
        topic="牛顿第二定律与多力合成",
        objective="通过受力分析得到合力，并用 F=ma 判断加速度",
        knowledge_points=["受力分析", "合力", "F=ma", "加速度方向"],
    )
    state = StudentState(mastery={point: 0.35 for point in goal.knowledge_points})
    candidates, audit = SkillLibrary().candidates(
        goal,
        state,
        profile=StudentProfile(prior_knowledge=["力", "质量", "加速度"]),
    )
    assert [skill.skill_id for skill in candidates] == ["newtons_second_law_intro_v1"]
    first_law = next(
        item for item in audit if item["skill_id"] == "newtons_first_law_via_engineering_examples_v1"
    )
    assert "教学目标不匹配" in first_law["reasons"]


def test_prerequisite_tag_is_not_enough_to_admit_transition_skill():
    goal = TeachingGoal(
        course="大学物理",
        topic="牛顿第三定律与作用力反作用力",
        objective="按施力物体和受力物体识别作用力反作用力对",
        knowledge_points=["相互作用", "作用力反作用力", "受力物体", "等大反向"],
    )
    state = StudentState(mastery={point: 0.35 for point in goal.knowledge_points})
    candidates, audit = SkillLibrary().candidates(
        goal,
        state,
        profile=StudentProfile(prior_knowledge=["力", "牛顿定律", "相互作用"]),
    )
    assert [skill.skill_id for skill in candidates] == ["newtons_third_law_content_formula_v1"]
    transition = next(
        item for item in audit if item["skill_id"] == "transition_from_newton_to_momentum_v1"
    )
    assert "教学目标不匹配" in transition["reasons"]


def test_transition_skill_requires_both_declared_goal_topics():
    goal = TeachingGoal(
        course="大学物理",
        topic="动量守恒、系统边界与外力冲量",
        objective="先选系统并检查外力冲量，再判断总动量是否守恒",
        knowledge_points=["系统边界", "外力冲量", "动量守恒", "空间均匀性"],
    )
    state = StudentState(mastery={point: 0.35 for point in goal.knowledge_points})
    candidates, audit = SkillLibrary().candidates(
        goal,
        state,
        profile=StudentProfile(prior_knowledge=["动量", "守恒"]),
    )
    assert [skill.skill_id for skill in candidates] == ["momentum_conservation_spatial_uniformity_v1"]
    transition = next(
        item for item in audit if item["skill_id"] == "transition_from_newton_to_momentum_v1"
    )
    assert "教学目标不匹配" in transition["reasons"]


def test_hard_filter_enforces_observable_prerequisite_evidence():
    goal, _, state = inputs()
    library = SkillLibrary()
    missing, audit = library.candidates(goal, state, profile=StudentProfile(prior_knowledge=[]))
    assert missing == []
    binary = next(item for item in audit if item["skill_id"] == "binary_search_boundary_by_interval_definition")
    assert "前置条件或掌握区间不满足" in binary["reasons"]

    matched, _ = library.candidates(
        goal,
        state,
        profile=StudentProfile(prior_knowledge=[]),
        student_message="我会 while 循环，也知道数组下标从 0 开始。",
    )
    assert [skill.skill_id for skill in matched] == ["binary_search_boundary_by_interval_definition"]


def test_every_frozen_evaluation_case_has_an_eligible_labeled_subject_skill():
    library = SkillLibrary()
    for case in load_cases():
        candidates, _ = library.candidates(
            case.goal,
            StudentState(mastery=case.initial_mastery),
            profile=case.profile,
        )
        assert candidates, case.case_id
        assert any(skill.skill_id in case.acceptable_skills for skill in candidates), case.case_id


def test_every_evaluation_case_has_complete_and_consistent_expectations():
    library = SkillLibrary()
    known_skill_ids = {skill.skill_id for skill in library.skills}
    allowed_switches = {"diagnostic", "scaffold", "correction", "transfer"}
    required_responses = {"confused", "partial", "correct", "transfer"}
    cases = load_cases()
    assert len(cases) == 18
    for case in cases:
        assert set(case.initial_mastery) == set(case.goal.knowledge_points), case.case_id
        assert case.true_misconceptions, case.case_id
        assert case.expected_focus.strip(), case.case_id
        assert set(case.acceptable_skills) <= known_skill_ids, case.case_id
        assert set(case.expected_switch_types) <= allowed_switches, case.case_id
        assert required_responses <= set(case.responses), case.case_id
        assert all(case.responses[key].strip() for key in required_responses), case.case_id
        assert 0 <= case.pretest_score <= 100, case.case_id


def test_state_evidence_only_changes_affected_point(tmp_path):
    agent = HybridTeachingAgent(
        llm=OpenAICompatibleClient({"api_key": ""}),
        store=SessionStore(tmp_path),
    )
    session = agent.start_session(*inputs())
    before = dict(session.state.mastery)
    session = agent.handle_student_message(session, "left 和 right 的区间定义我不明白")
    changed = [point for point in before if session.state.mastery[point] != before[point]]
    assert changed == ["区间定义"]
    assert session.state.evidence[-1].knowledge_point == "区间定义"
    assert session.state.knowledge_states["区间定义"].confidence > 0


def test_four_stage_failure_policy(tmp_path):
    agent = HybridTeachingAgent(
        llm=OpenAICompatibleClient({"api_key": ""}),
        store=SessionStore(tmp_path),
        settings={"max_rounds": 8, "mastery_threshold": 0.8, "no_progress_limit": 4},
    )
    session = agent.start_session(*inputs())
    observed = []
    for answer in ("我不懂区间", "还是不懂", "依然不会"):
        session = agent.handle_student_message(session, answer)
        observed.append(session.turns[-1].action_type)
    assert observed == ["diagnostic", "scaffold", "correction"]


def test_store_duplicate_archive_trash_restore_and_index(tmp_path):
    store = SessionStore(tmp_path)
    session = TeachingSession(goal=inputs()[0], profile=inputs()[1], state=inputs()[2])
    store.save(session)
    clone = store.duplicate(session.session_id)
    assert clone.session_id != session.session_id
    store.archive(clone.session_id)
    visible, total = store.list_metadata()
    assert total == 1
    assert visible[0]["session_id"] == session.session_id
    assert store.delete(session.session_id)
    assert session.session_id in store.list_trash()
    restored = store.restore(session.session_id)
    assert restored.session_id == session.session_id


def test_import_collision_creates_copy(tmp_path):
    store = SessionStore(tmp_path)
    session = TeachingSession(goal=inputs()[0], profile=inputs()[1], state=inputs()[2])
    store.save(session)
    imported = store.import_session(session)
    assert imported.session_id != session.session_id
    assert "导入副本" in imported.display_title


def test_store_atomic_parallel_writes_have_valid_json(tmp_path):
    store = SessionStore(tmp_path)
    sessions = [TeachingSession(goal=inputs()[0], profile=inputs()[1], state=inputs()[2]) for _ in range(25)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(store.save, sessions))
    for session in sessions:
        payload = json.loads((tmp_path / f"{session.session_id}.json").read_text(encoding="utf-8"))
        assert payload["schema_version"] == 5


def test_legacy_session_migrates_without_semantic_mutation(tmp_path):
    session = TeachingSession(goal=inputs()[0], profile=inputs()[1], state=inputs()[2])
    payload = session.model_dump(mode="json")
    payload.pop("schema_version")
    payload.pop("display_title")
    (tmp_path / f"{session.session_id}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    loaded = SessionStore(tmp_path).load(session.session_id)
    assert loaded.schema_version == 5
    assert loaded.display_title == loaded.goal.topic
