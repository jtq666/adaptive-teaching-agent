import csv
import io
from collections import Counter

from src.evaluation import (
    HUMAN_RATING_FIELDS,
    METHODS,
    DeterministicStudent,
    EvaluationRunner,
    _fuzzy_match,
    load_cases,
    summarize_human_annotations,
    validate_human_annotation_csv,
)
from src.models import EvaluationReport, StudentProfile, StudentState, TeachingGoal, TeachingSession
from src.storage import EvaluationStore, SessionStore


def test_fuzzy_label_matching_is_domain_neutral():
    assert _fuzzy_match("光合作用能量转换", "光合作用中的能量转换")
    assert _fuzzy_match("文艺复兴人文主义", "人文主义")
    assert _fuzzy_match("HTTP状态码", "http 状态码")
    assert not _fuzzy_match("细胞呼吸", "古典诗歌")


def test_session_roundtrip(tmp_path):
    store = SessionStore(tmp_path)
    session = TeachingSession(
        goal=TeachingGoal(course="物理", topic="惯性", objective="理解惯性", knowledge_points=["惯性"]),
        profile=StudentProfile(name="小明"),
        state=StudentState(mastery={"惯性": 0.4}),
    )
    store.save(session)
    loaded = store.load(session.session_id)
    assert loaded.model_dump() == session.model_dump()


def test_session_update_and_delete_crud(tmp_path):
    store = SessionStore(tmp_path)
    session = TeachingSession(
        goal=TeachingGoal(course="物理", topic="惯性", objective="理解惯性", knowledge_points=["惯性"]),
        profile=StudentProfile(name="小明"),
        state=StudentState(mastery={"惯性": 0.4}),
    )
    store.save(session)
    updated = store.update_metadata(session.session_id, display_title="牛顿第一定律复习")
    assert updated.display_title == "牛顿第一定律复习"
    assert store.load(session.session_id).goal.topic == "惯性"
    assert store.delete(session.session_id) is True
    assert store.delete(session.session_id) is False


def test_session_metadata_filters_index_rebuild_and_corrupt_isolation(tmp_path):
    store = SessionStore(tmp_path)
    session = TeachingSession(
        goal=TeachingGoal(course="高等数学", topic="导数", objective="理解导数", knowledge_points=["极限"]),
        profile=StudentProfile(name="小红"),
        state=StudentState(mastery={"极限": 0.4}),
    )
    store.save(session)
    rows, total = store.list_metadata(course="高等数学", status="active", query="小红")
    assert total == 1 and rows[0]["session_id"] == session.session_id
    assert store.list_courses() == ["高等数学"]
    store.index_path.write_text("not json", encoding="utf-8")
    assert store.list_metadata()[1] == 1
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{bad", encoding="utf-8")
    store.rebuild_index()
    assert (tmp_path / ".corrupt" / corrupt.name).exists()


def test_evaluation_store_index_is_lazy_and_archive_recoverable(tmp_path):
    report = EvaluationRunner(seed=11).run(load_cases()[:1])
    store = EvaluationStore(tmp_path)
    imported = store.import_report(report)
    rows, total = store.list_report_metadata(query=imported.name)
    assert total == 1 and rows[0]["case_count"] == 1
    store.archive(imported.name)
    assert store.list_report_metadata()[1] == 0
    store.restore_archive(imported.name)
    assert store.list_report_metadata()[1] == 1


def test_session_store_rejects_unsafe_import_id(tmp_path):
    store = SessionStore(tmp_path)
    session = TeachingSession(
        session_id="../escape",
        goal=TeachingGoal(course="物理", topic="惯性", objective="理解惯性", knowledge_points=["惯性"]),
        profile=StudentProfile(name="小明"),
        state=StudentState(mastery={"惯性": 0.4}),
    )
    try:
        store.save(session)
    except ValueError as exc:
        assert "不安全" in str(exc)
    else:
        raise AssertionError("路径穿越式会话 ID 应被拒绝")


def test_evaluation_has_eighteen_cases_and_two_baselines():
    cases = load_cases()
    report = EvaluationRunner().run(cases)
    assert sum(case.split == "development" for case in cases) == 6
    assert sum(case.split == "held_out" for case in cases) == 12
    assert {case.goal.course for case in cases} == {"高等数学", "大学物理", "程序设计"}
    assert all(sum(case.goal.course == course for case in cases) == 6 for course in {case.goal.course for case in cases})
    assert all(case.case_id.startswith("heldout_") for case in cases if case.split == "held_out")
    assert len(report.case_results) == 54
    assert len({item.case_id for item in report.case_results}) == 18
    assert report.methods == METHODS
    assert len(report.summary) == 3
    adaptive = next(item for item in report.summary if item["method"] == METHODS[0])
    fixed = next(item for item in report.summary if item["method"] == METHODS[1])
    generic = next(item for item in report.summary if item["method"] == METHODS[2])
    assert adaptive["decision_quality"] > fixed["decision_quality"] > generic["decision_quality"]
    assert adaptive["normalized_gain"] > generic["normalized_gain"]
    assert fixed["normalized_gain"] > generic["normalized_gain"]
    assert adaptive["learning_efficiency"] > generic["learning_efficiency"]
    assert fixed["learning_efficiency"] > generic["learning_efficiency"]
    assert all(0 <= item["transfer_accuracy"] <= 1 for item in report.summary)
    assert report.statistical_tests
    assert fixed["transfer_accuracy"] > 0
    assert report.evaluation_protocol["success_rule"]
    assert all(result.action_types for result in report.case_results)
    assert report.failure_case["metrics"]["success"] is False
    assert report.paired_comparisons
    assert all(item["metric"] != "behavior_proxy" for item in report.paired_comparisons)
    assert all("ci_low" in item and "ci_high" in item for item in report.paired_comparisons)
    assert all(item["n_pairs"] == 18 for item in report.paired_comparisons)
    assert all(item["cases"] == 18 for item in report.summary)


def test_blind_annotation_export_hides_method_names():
    runner = EvaluationRunner()
    report = runner.run(load_cases()[:1])
    blind = runner.human_annotation_csv(report)
    key = runner.human_annotation_csv(report, reveal_key=True)
    assert "教师回复" in blind
    assert "自适应混合 Agent" not in blind
    assert "自适应混合 Agent" in key


def test_human_annotation_validation_and_weighted_kappa_attachment(tmp_path):
    fields = ["样本ID", "评审员", *HUMAN_RATING_FIELDS]
    lines = [",".join(fields)]
    for sample in ("case-a-1", "case-b-1"):
        lines.append(",".join([sample, "评审员甲", *(["4"] * len(HUMAN_RATING_FIELDS))]))
        lines.append(",".join([sample, "评审员乙", *(["4"] * len(HUMAN_RATING_FIELDS))]))
    rows = validate_human_annotation_csv("\n".join(lines))
    summary = summarize_human_annotations(rows)
    assert summary["sample_count"] == 2
    assert summary["weighted_cohen_kappa"] == 1.0
    store = EvaluationStore(tmp_path)
    attachment = store.save_human_review("evaluation_current.json", rows, summary)
    assert attachment.exists()
    assert "evaluation_current.json" in attachment.read_text(encoding="utf-8")


def test_human_annotation_rejects_missing_or_invalid_scores():
    with_error = "样本ID,评审员,知识正确性\ncase,甲,6"
    try:
        validate_human_annotation_csv(with_error)
    except ValueError as exc:
        assert "缺少字段" in str(exc)
    else:
        raise AssertionError("不完整的人工标注表应被拒绝")

    fields = ["样本ID", "评审员", *HUMAN_RATING_FIELDS]
    invalid = ",".join(["case", "甲", "0", *(["4"] * (len(HUMAN_RATING_FIELDS) - 1))])
    try:
        validate_human_annotation_csv(",".join(fields) + "\n" + invalid)
    except ValueError as exc:
        assert "1–5" in str(exc)
    else:
        raise AssertionError("超出量表范围的人工标注应被拒绝")


def test_full_blind_annotation_samples_twelve_paired_cases_per_method():
    runner = EvaluationRunner()
    report = runner.run(load_cases())
    rows = list(csv.DictReader(io.StringIO(runner.human_annotation_csv(report))))
    assert len(rows) == 36
    counts = Counter(row["盲法系统编号"] for row in rows)
    assert sorted(counts.values()) == [12, 12, 12]
    cases_by_code = {
        code: {row["案例"] for row in rows if row["盲法系统编号"] == code}
        for code in counts
    }
    assert all(len(case_ids) == 12 for case_ids in cases_by_code.values())
    assert len({tuple(sorted(case_ids)) for case_ids in cases_by_code.values()}) == 1


def test_student_gain_function_is_method_blind():
    case = load_cases()[0]
    averages = []
    for method in METHODS:
        student = DeterministicStudent(case)
        student.respond("subject_instruction", method, aligned=True)
        averages.append(sum(student.mastery.values()) / len(student.mastery))
    assert averages[0] == averages[1] == averages[2]


def test_repeated_teaching_action_has_method_blind_diminishing_returns():
    values = [DeterministicStudent._diminishing_multiplier(index) for index in range(1, 7)]
    assert values == sorted(values, reverse=True)
    assert values[0] == 1.0
    assert values[-1] == 0.3


def test_all_methods_share_the_same_hidden_transfer_threshold():
    case = load_cases()[0]
    outcomes = []
    for action in ("transfer", "fixed_verification", "generic_verification"):
        student = DeterministicStudent(case, seed=17)
        student.mastery = {point: 0.71 for point in student.mastery}
        student.initial_average = 0.71
        student.respond(action, "method name must be ignored", aligned=True)
        outcomes.append(student.transfer_passed)
    assert outcomes == [True, True, True]


def test_generic_instruction_can_eventually_correct_a_misconception():
    case = load_cases()[0]
    student = DeterministicStudent(case, seed=23)
    for _ in range(60):
        student.respond("generic", "method name must be ignored", aligned=False)
    assert student.misconceptions == []


def test_markdown_report_contains_required_sections():
    runner = EvaluationRunner()
    report = runner.run(load_cases()[:1])
    markdown = runner.to_markdown(report)
    assert "方法对比" in markdown
    assert "典型案例" in markdown
    assert "实验协议与公平性" in markdown
    assert "固定单 Skill" in markdown


def test_evaluation_archive_import_load_and_delete_bundle(tmp_path):
    report = EvaluationRunner(seed=11).run(load_cases()[:1])
    store = EvaluationStore(tmp_path)
    json_path = store.import_report(EvaluationReport.model_validate(report.model_dump()))
    assert store.list_reports() == [json_path]
    assert store.load(json_path.name).seed == 11

    token = json_path.stem.removeprefix("evaluation_")
    for name in (
        f"evaluation_{token}.csv",
        f"evaluation_{token}.md",
        f"annotation_blind_{token}.csv",
        f"annotation_key_{token}.csv",
    ):
        (tmp_path / name).write_text("test", encoding="utf-8")
    removed = store.delete_bundle(json_path.name)
    assert len(removed) == 5
    assert not store.list_reports()
    assert json_path.name in store.list_trash()
    restored = store.restore_bundle(json_path.name)
    assert len(restored) == 5
    assert store.list_reports()


def test_evaluation_bundle_archive_and_restore_moves_all_files(tmp_path):
    report = EvaluationRunner(seed=19).run(load_cases()[:1])
    store = EvaluationStore(tmp_path)
    json_path = store.import_report(report)
    token = json_path.stem.removeprefix("evaluation_")
    for name in (
        f"evaluation_{token}.csv",
        f"evaluation_{token}.md",
        f"annotation_blind_{token}.csv",
        f"annotation_key_{token}.csv",
    ):
        (tmp_path / name).write_text("test", encoding="utf-8")
    moved = store.archive(json_path.name)
    assert len(moved) == 5
    assert store.list_archived() == [json_path.name]
    assert not store.list_reports()
    restored = store.restore_archive(json_path.name)
    assert len(restored) == 5
    assert store.list_reports() == [json_path]
