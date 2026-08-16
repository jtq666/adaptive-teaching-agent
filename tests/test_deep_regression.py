from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agent import HybridTeachingAgent
from src.evaluation import METHODS, DeterministicStudent, EvaluationRunner, load_cases
from src.llm import LLMUnavailableError, OpenAICompatibleClient
from src.models import StudentProfile, StudentState, TeachingGoal, TeachingSession
from src.skills import SkillLibrary
from src.state_tracker import StateTracker
from src.storage import SessionStore


def binary_inputs(mastery: float = 0.35):
    goal = TeachingGoal(
        course="程序设计",
        topic="二分查找边界条件",
        objective="从区间不变量推导边界更新",
        knowledge_points=["区间定义", "循环不变量", "边界更新"],
    )
    profile = StudentProfile(name="深度测试学生", level="中等", prior_knowledge=["while循环", "数组"])
    state = StudentState(
        mastery={point: mastery for point in goal.knowledge_points},
        next_focus="诊断区间定义",
    )
    return goal, profile, state


def offline_agent(tmp_path: Path, **settings) -> HybridTeachingAgent:
    return HybridTeachingAgent(
        library=SkillLibrary(),
        llm=OpenAICompatibleClient({"api_key": "", "model": "offline"}),
        store=SessionStore(tmp_path),
        settings={
            "max_rounds": 8,
            "mastery_threshold": 0.8,
            "no_progress_limit": 3,
            "candidate_limit": 5,
            **settings,
        },
    )


def test_llm_client_retries_then_succeeds(monkeypatch):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise TimeoutError("temporary timeout")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="恢复成功"))]
        )

    client = OpenAICompatibleClient({"api_key": "", "retries": 2, "max_tokens": 77})
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr("src.llm.time.sleep", lambda _: None)
    assert client.chat("system", "user") == "恢复成功"
    assert len(calls) == 3
    assert calls[-1]["max_tokens"] == 77


def test_llm_client_without_key_fails_fast_without_network():
    client = OpenAICompatibleClient({"api_key": ""})
    with pytest.raises(LLMUnavailableError, match="未配置"):
        client.chat("system", "user")


def test_llm_client_retry_exhaustion_reports_last_error(monkeypatch):
    client = OpenAICompatibleClient({"api_key": "", "retries": 2})
    client.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: (_ for _ in ()).throw(ConnectionError("断网")))
        )
    )
    monkeypatch.setattr("src.llm.time.sleep", lambda _: None)
    with pytest.raises(LLMUnavailableError, match="断网"):
        client.chat("system", "user")


def test_structured_output_extracts_json_and_rejects_plain_text():
    client = OpenAICompatibleClient({"api_key": ""})
    client.chat = lambda *_, **__: '说明文字```json\n{"skill_id":"x","reason":"ok"}\n```'
    assert client.structured("s", "u", "schema")["skill_id"] == "x"
    client.chat = lambda *_, **__: "没有结构化对象"
    with pytest.raises(ValueError, match="JSON"):
        client.structured("s", "u", "schema")


def test_structured_output_repairs_invalid_literal_backslashes():
    client = OpenAICompatibleClient({"api_key": ""})
    client.chat = lambda *_, **__: r'{"reason":"\(text)"}'
    assert client.structured("s", "u", "schema")["reason"] == "\\(text)"


def test_llm_retry_policy_only_retries_transient_failures():
    class TimeoutErrorLike(Exception):
        pass

    class AuthErrorLike(Exception):
        pass

    assert OpenAICompatibleClient._is_retryable(TimeoutErrorLike()) is True
    assert OpenAICompatibleClient._is_retryable(AuthErrorLike()) is False

    rate_limit = Exception("rate")
    rate_limit.status_code = 429
    assert OpenAICompatibleClient._is_retryable(rate_limit) is True


def test_skill_library_empty_and_missing_id_errors_are_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="Skill Library 为空"):
        SkillLibrary(tmp_path)
    with pytest.raises(KeyError, match="Skill 不存在"):
        SkillLibrary().get("does_not_exist")


class ExtremeAssessmentClient:
    available = True

    def structured(self, *_, **__):
        return {
            "mastery_updates": {"区间定义": 99, "循环不变量": -99, "边界更新": 0.9},
            "misconceptions": [],
            "understanding_signals": ["极端输入"],
            "next_focus": "验证边界",
            "verification_passed": False,
            "progress": "improved",
            "affected_points": ["区间定义"],
        }


def test_state_tracker_limits_single_round_mastery_deltas():
    goal, profile, before = binary_inputs(0.4)
    after = StateTracker(ExtremeAssessmentClient()).update(goal, profile, before, "部分回答")
    # Mastery is now updated from the generic evidence level rather than
    # trusting an arbitrary absolute score returned by the LLM.
    assert after.mastery["区间定义"] == pytest.approx(0.603, abs=0.001)
    assert after.mastery["循环不变量"] == pytest.approx(0.4)
    assert after.mastery["边界更新"] == pytest.approx(0.4)
    assert all(0 <= value <= 1 for value in after.mastery.values())


def test_agent_survives_empty_ranked_candidates(tmp_path, monkeypatch):
    library = SkillLibrary()
    monkeypatch.setattr(library, "candidates", lambda *_, **__: ([], []))
    agent = HybridTeachingAgent(
        library=library,
        llm=OpenAICompatibleClient({"api_key": ""}),
        store=SessionStore(tmp_path),
        settings={"max_rounds": 8, "mastery_threshold": 0.8, "no_progress_limit": 3},
    )
    session = agent.start_session(*binary_inputs())
    assert session.turns[0].selected_skill_id == "diagnostic_questioning_v1"
    assert session.turns[0].candidate_skill_ids == []


class SemanticSelectionClient:
    available = True

    def structured(self, *_, schema_hint="", **__):
        schema = schema_hint or (str(_[2]) if len(_) >= 3 else "")
        if "skill_id" in schema:
            return {"skill_id": "not_a_candidate", "reason": "故意返回未知候选"}
        raise ValueError("use heuristic state tracking")

    def chat(self, *_, **__):
        return "请说明你判断区间边界的依据是什么？"


def test_semantic_selector_cannot_escape_candidate_set(tmp_path):
    agent = HybridTeachingAgent(
        library=SkillLibrary(),
        llm=SemanticSelectionClient(),
        store=SessionStore(tmp_path),
        settings={"max_rounds": 8, "mastery_threshold": 0.8, "no_progress_limit": 3, "candidate_limit": 5},
    )
    session = agent.start_session(*binary_inputs())
    assert session.turns[0].selected_skill_id == session.turns[0].candidate_skill_ids[0]
    assert session.turns[0].decision_mode == "rule_margin_selection"


def test_max_round_guard_records_terminal_snapshot(tmp_path):
    agent = offline_agent(tmp_path, max_rounds=2, no_progress_limit=99)
    session = agent.start_session(*binary_inputs())
    session = agent.handle_student_message(session, "我给出了一点信息但还不完整")
    session = agent.handle_student_message(session, "继续尝试")
    assert str(session.status) == "unable"
    assert session.turns[-1].action_type == "terminate_max_rounds"
    assert session.turns[-1].policy_rule == "max_rounds"
    assert session.turns[-1].stop_decision == session.termination_reason
    assert session.turns[-1].state_after == session.state
    assert "最大教学轮数 2" in session.termination_reason


def test_hundred_sessions_are_unique_and_roundtrip_under_parallel_writes(tmp_path):
    store = SessionStore(tmp_path)

    def create_and_save(index: int) -> str:
        goal, profile, state = binary_inputs((index % 10) / 10)
        session = TeachingSession(goal=goal, profile=profile, state=state)
        store.save(session)
        return session.session_id

    with ThreadPoolExecutor(max_workers=12) as pool:
        session_ids = list(pool.map(create_and_save, range(100)))
    assert len(set(session_ids)) == 100
    assert len(store.list_sessions()) == 100
    assert all(store.load(session_id).session_id == session_id for session_id in session_ids)


def test_evaluation_is_bitwise_reproducible_for_same_seed():
    cases = load_cases()
    first = EvaluationRunner(seed=314159).run(cases)
    second = EvaluationRunner(seed=314159).run(cases)
    first_data = first.model_dump(mode="json", exclude={"generated_at"})
    second_data = second.model_dump(mode="json", exclude={"generated_at"})
    assert first_data == second_data


def test_evaluation_metrics_obey_bounds_and_pairing_invariants():
    cases = load_cases()
    report = EvaluationRunner(seed=271828).run(cases)
    bounded = [
        "misconception_precision", "misconception_recall", "misconception_f1",
        "mastery_mae", "focus_accuracy", "skill_selection_accuracy", "switch_accuracy",
        "termination_accuracy", "behavior_quality", "direct_answer_violation_rate",
        "normalized_gain", "learning_efficiency", "transfer_accuracy",
    ]
    assert len(report.case_results) == len(cases) * len(METHODS)
    for case in cases:
        rows = [row for row in report.case_results if row.case_id == case.case_id]
        assert {row.method for row in rows} == set(METHODS)
        assert {row.pretest_score for row in rows} == {case.pretest_score}
    for row in report.case_results:
        assert 1 <= row.rounds <= 8
        assert 0 <= row.posttest_score <= 100
        assert all(
            (-1 <= getattr(row, metric) <= 1 if metric in {"normalized_gain", "learning_efficiency"}
             else 0 <= getattr(row, metric) <= 1)
            for metric in bounded
        )
        assert len(row.selected_skills) == row.rounds
        assert len(row.action_types) == row.rounds
        assert len(row.teacher_messages) == row.rounds
        assert all(0 <= score <= 1 for score in row.behavior_dimensions.values())
    for comparison in report.paired_comparisons:
        assert comparison["ci_low"] <= comparison["mean_difference"] <= comparison["ci_high"]


def test_paired_statistics_cluster_repeated_seeds_by_case():
    runner = EvaluationRunner(seed=99)
    report = runner.run(load_cases()[:1])
    duplicated = []
    for row in report.case_results:
        duplicated.append(row)
        clone = row.model_copy(deep=True)
        clone.simulation_seed += 1
        duplicated.append(clone)
    comparisons = runner._paired_comparisons(duplicated)
    tests = runner._statistical_tests(duplicated)
    assert all(item["n_pairs"] == 1 for item in comparisons)
    assert all(item["mcnemar_b"] + item["mcnemar_c"] <= 1 for item in tests)


def test_confidence_interval_empty_and_edge_inputs():
    runner = EvaluationRunner(seed=7)
    assert runner._bootstrap_mean_ci([], seed=7) == (0.0, 0.0)
    assert runner._wilson_ci(0, 0) == (0.0, 0.0)
    low, high = runner._wilson_ci(10, 10)
    assert 0 <= low <= high <= 1


def test_diagnostic_action_has_small_positive_simulated_gain():
    case = load_cases()[0]
    student = DeterministicStudent(case)
    before = sum(student.mastery.values())
    student.respond("diagnostic", METHODS[0], aligned=True)
    assert sum(student.mastery.values()) > before
