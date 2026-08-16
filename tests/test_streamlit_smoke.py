import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.evaluation import EvaluationRunner, load_cases
from src.models import ResponseOption
from src.ui import get_evaluation_store, get_library, get_llm_client, get_store

APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


@pytest.fixture(autouse=True)
def isolate_streamlit_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("TEACHING_AGENT_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("TEACHING_AGENT_EVALUATION_DIR", str(tmp_path / "evaluations"))
    get_llm_client.clear()
    get_library.clear()
    get_store.clear()
    get_evaluation_store.clear()
    yield
    get_llm_client.clear()
    get_library.clear()
    get_store.clear()
    get_evaluation_store.clear()


def button(result, label):
    return next(item for item in result.button if item.label == label)


def open_active_session(session):
    result = AppTest.from_file(str(APP))
    result.session_state["teaching_session"] = session
    result.session_state["live_view_mode"] = "教师/答辩视图"
    return result.run(timeout=30)


def test_streamlit_entrypoint_and_all_pages_load():
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "实时教学"
    for page, title in {
        "app_pages/live_teaching.py": "实时教学",
        "app_pages/skill_library.py": "Skill Library",
        "app_pages/session_replay.py": "过程回放",
        "app_pages/evaluation_lab.py": "Agent 评估",
    }.items():
        result = app.switch_page(page).run(timeout=30)
        assert not result.exception
        assert result.title[0].value == title


def test_skill_library_lists_and_filters_all_skills():
    app = AppTest.from_file(str(APP)).run(timeout=30)
    app = app.switch_page("app_pages/skill_library.py").run(timeout=30)
    assert not app.exception
    assert len(app.status) == 14
    assert any("找到 14 个 Skill" in item.value for item in app.caption)

    course_filter = next(item for item in app.selectbox if item.label == "课程")
    app = course_filter.set_value("跨课程通用").run(timeout=30)
    assert not app.exception
    assert len(app.status) == 4
    assert any("找到 4 个 Skill" in item.value for item in app.caption)


def test_live_teaching_form_and_four_stage_chat():
    result = AppTest.from_file(str(APP)).run(timeout=30)
    result = button(result, "开始教学").click().run(timeout=30)
    assert not result.exception
    session = result.session_state["teaching_session"]
    assert session.turns[0].micro_step is not None
    assert any("查看模式" in item.label for item in result.segmented_control)
    assert any("当前目标" in item.value for item in result.markdown) or any(
        "当前目标" in item.value for item in result.caption
    )
    assert session.teaching_route is not None
    assert len(session.teaching_route.steps) >= 2
    assert not any("内部关注点" in item.value for item in result.caption)
    assert not any("当前还没有足够的学生回答证据" in item.value for item in result.markdown)
    assert not any("专业决策证据" in item.value for item in result.expander)
    result = next(item for item in result.segmented_control if item.label == "查看模式").set_value(
        "教师/答辩视图"
    ).run(timeout=30)
    assert any("本轮只解决" in item.value for item in result.markdown)
    assert any("本轮单步教学上下文" in item.value for item in result.caption)
    expected = [
        ("我不知道，left 和 right 总是混淆。", "diagnostic_questioning_v1"),
        ("还是不明白，我记不住应该写哪个。", "scaffolded_hint_ladder_v1"),
        ("依然不会，我觉得循环结束就不用检查元素。", "misconception_contrast_correction_v1"),
    ]
    for reply, skill_id in expected:
        result = open_active_session(session)
        result = result.chat_input[0].set_value(reply).run(timeout=30)
        assert not result.exception
        session = result.session_state["teaching_session"]
        assert session.turns[-1].selected_skill_id == skill_id
        assert any(f"教学策略 · {skill_id}" in item.value for item in result.markdown)
        if len(session.turns) == 2:
            assert any("本轮主执行角色" in item.value for item in result.success)
            assert any("教学方案组成" in item.value for item in result.success)
        if len(session.turns) > 1:
            assert any("Skill 已切换" in item.value for item in result.success)
    assert session.state.evidence


def test_live_choice_mode_renders_and_submits_button():
    app = AppTest.from_file(str(APP)).run(timeout=30)
    app = button(app, "开始教学").click().run(timeout=30)
    session = app.session_state["teaching_session"]
    current_step = session.turns[-1].micro_step
    assert current_step is not None
    current_step.response_mode = "single_choice"
    current_step.options = [
        ResponseOption(option_id="A", text="端点包含"),
        ResponseOption(option_id="B", text="端点不包含"),
    ]
    before = len(session.turns)
    app = open_active_session(session)
    assert not app.exception
    assert app.radio
    assert any(item.label == "提交选择" for item in app.button)
    app = button(app, "提交选择").click().run(timeout=30)
    assert not app.exception
    updated = app.session_state["teaching_session"]
    assert len(updated.turns) == before + 1
    assert "选择" in updated.turns[-1].student_message


def test_live_new_session_uses_open_response_mode():
    app = AppTest.from_file(str(APP)).run(timeout=30)
    app = button(app, "开始教学").click().run(timeout=30)
    assert not app.exception
    session = app.session_state["teaching_session"]
    assert session.profile.response_preference == "open"
    assert session.turns[-1].micro_step is not None
    assert session.turns[-1].micro_step.response_mode == "open"
    assert app.chat_input


@pytest.mark.parametrize(
    ("preset", "topic_token", "replies"),
    [
        (
            "导数极限定义",
            "导数",
            [
                "我不知道为什么导数需要极限。",
                "还是不明白，我只会背公式。",
                "依然不会，我觉得割线斜率就是切线斜率。",
            ],
        ),
        (
            "牛顿第一定律",
            "牛顿第一定律",
            [
                "我不知道，物体不受力为什么还能运动。",
                "还是不明白，我觉得运动必须有力维持。",
                "依然不会，只要物体运动就一定有向前的合力。",
            ],
        ),
    ],
)
def test_demo_subjects_show_visible_skill_switches_and_evidence(preset, topic_token, replies):
    app = AppTest.from_file(str(APP)).run(timeout=30)
    app = app.segmented_control[0].set_value(preset).run(timeout=30)
    app = button(app, "开始教学").click().run(timeout=30)
    assert not app.exception
    session = app.session_state["teaching_session"]
    assert topic_token in session.goal.topic

    expected_skills = [
        "diagnostic_questioning_v1",
        "scaffolded_hint_ladder_v1",
        "misconception_contrast_correction_v1",
    ]
    for reply, expected_skill in zip(replies, expected_skills, strict=True):
        app = open_active_session(session)
        app = app.chat_input[0].set_value(reply).run(timeout=30)
        assert not app.exception
        session = app.session_state["teaching_session"]
        turn = session.turns[-1]
        assert turn.selected_skill_id == expected_skill
        assert turn.switch_reason

        assert turn.state_after.evidence
        assert any(f"教学策略 · {expected_skill}" in item.value for item in app.markdown)
        assert any("Skill 已切换" in item.value and "→" in item.value for item in app.success)
        assert any("适合提供学科内容" in item.value for item in app.info)
        assert any("本轮教学方案" in item.value for item in app.caption)

    # The correction must remain answerable. Pausing on the same rerun would
    # show a correction message and immediately remove the input box.
    assert str(session.status) == "active"
    assert app.chat_input
    app = open_active_session(session)
    app = app.chat_input[0].set_value("纠正后我仍然不会，也无法解释理由。").run(timeout=30)
    assert not app.exception
    session = app.session_state["teaching_session"]
    assert str(session.status) == "unable"
    assert session.turns[-1].action_type == "terminate_no_improvement"
    assert session.turns[-1].policy_rule == "no_improvement_after_correction"
    assert any("已暂停" in item.value for item in app.markdown)
    assert button(app, "换一种讲法继续")
    assert button(app, "开始全新任务")
    previous_session_id = session.session_id
    previous_turn_count = len(session.turns)
    previous_mastery = session.state.mastery.copy()
    app = button(app, "换一种讲法继续").click().run(timeout=30)
    recovered = app.session_state["teaching_session"]
    assert str(recovered.status) == "active"
    assert recovered.session_id == previous_session_id
    assert len(recovered.turns) == previous_turn_count + 1
    assert recovered.state.mastery == previous_mastery
    assert recovered.state.no_progress_rounds == 0
    assert app.chat_input

    replay = AppTest.from_file(str(APP)).run(timeout=30).switch_page("app_pages/session_replay.py").run(timeout=30)
    assert not replay.exception
    session_picker = next(item for item in replay.selectbox if item.label == "选择会话")
    original_label = next(option for option in session_picker.options if previous_session_id[:6] in option)
    replay = session_picker.set_value(original_label).run(timeout=30)
    trajectory = replay.dataframe[0].value
    assert expected_skills[-1] in trajectory["教学策略"].tolist()


def test_binary_search_is_preserved_for_legacy_data_but_hidden_from_default_demo():
    source = (APP.parent / "app_pages" / "live_teaching.py").read_text(encoding="utf-8")
    assert '"二分查找边界": {' in source
    assert 'DEMO_PRESET_NAMES = ("牛顿第一定律", "导数极限定义")' in source
    assert '"prior": "速度,力,运动,惯性"' in source
    assert '"prior": "函数,斜率,平均变化率,平均速度"' in source


def test_high_mastery_transfer_success_is_visible_and_disables_chat():
    app = AppTest.from_file(str(APP)).run(timeout=30)
    app = next(item for item in app.selectbox if item.label == "课程").set_value("程序设计").run(timeout=30)
    app = next(item for item in app.text_input if item.label == "教学主题").set_value("二分查找边界条件").run(timeout=30)
    app = next(item for item in app.text_area if item.label == "希望学生最终学会什么").set_value("从区间不变量推导边界更新").run(timeout=30)
    app = next(item for item in app.text_input if item.label == "知识点（逗号分隔）").set_value("区间定义,边界更新").run(timeout=30)
    mastery = next(item for item in app.slider if item.label == "初始掌握度")
    app = mastery.set_value(0.85).run(timeout=30)
    app = button(app, "开始教学").click().run(timeout=30)
    session = app.session_state["teaching_session"]

    app = open_active_session(session)
    app = app.chat_input[0].set_value(
        "换一个新题同样可以，因为先定义区间，所以我能推出边界更新并解释区别。"
    ).run(timeout=30)
    assert not app.exception
    session = app.session_state["teaching_session"]
    assert str(session.status) == "active"
    assert session.state.transfer_verified is False
    assert session.turns[-1].selected_skill_id == "transfer_verification_v1"

    app = open_active_session(session)
    app = app.chat_input[0].set_value(
        "在新数组里也先定义闭区间；因为 left==right 时还有一个候选，"
        "所以使用 left<=right，并按循环不变量更新边界。"
    ).run(timeout=30)
    assert not app.exception
    session = app.session_state["teaching_session"]
    assert str(session.status) == "success"
    assert session.state.transfer_verified is True
    assert session.turns[-1].selected_skill_id == "transfer_verification_v1"
    app = open_active_session(session)
    assert any("教学策略 · transfer_verification_v1" in item.value for item in app.markdown)
    assert any("已达成" in item.value for item in app.markdown)
    assert not app.chat_input


def test_live_preset_and_empty_goal_validation():
    result = AppTest.from_file(str(APP)).run(timeout=30)
    result = result.segmented_control[0].set_value("导数极限定义").run(timeout=30)
    assert "导数" in next(item for item in result.text_input if item.label == "教学主题").value
    topic = next(item for item in result.text_input if item.label == "教学主题")
    topic.set_value("")
    result = button(result, "开始教学").click().run(timeout=30)
    assert result.error


def test_live_preset_recovers_from_deselected_legacy_state():
    app = AppTest.from_file(str(APP))
    app.session_state["preset_name"] = None

    app = app.run(timeout=30)

    assert not app.exception
    assert app.session_state["preset_name"] == "牛顿第一定律"
    preset_control = next(item for item in app.segmented_control if item.label == "从示例开始")
    assert preset_control.value == "牛顿第一定律"


def test_live_view_and_demo_controls_recover_from_none_state():
    seed = AppTest.from_file(str(APP)).run(timeout=30)
    seed = button(seed, "开始教学").click().run(timeout=30)
    session = seed.session_state["teaching_session"]

    student_view = AppTest.from_file(str(APP))
    student_view.session_state["teaching_session"] = session
    student_view.session_state["live_view_mode"] = None
    student_view = student_view.run(timeout=30)
    assert not student_view.exception
    assert student_view.session_state["live_view_mode"] == "学生视图"

    teacher_view = AppTest.from_file(str(APP))
    teacher_view.session_state["teaching_session"] = session
    teacher_view.session_state["live_view_mode"] = "教师/答辩视图"
    teacher_view = teacher_view.run(timeout=30)
    assert not teacher_view.exception
    assert any(item.label == "生成 3 条 AI 推荐回答" for item in teacher_view.button)


def test_replay_page_search_duplicate_archive_and_continue():
    live = AppTest.from_file(str(APP)).run(timeout=30)
    live = button(live, "开始教学").click().run(timeout=30)
    session = live.session_state["teaching_session"]
    replay = AppTest.from_file(str(APP))
    replay.session_state["selected_replay"] = None
    replay = replay.run(timeout=30).switch_page("app_pages/session_replay.py").run(timeout=30)
    assert not replay.exception
    search = next(item for item in replay.text_input if item.label == "搜索会话")
    replay = search.set_value(session.session_id).run(timeout=30)
    assert not replay.exception
    assert any(item.label == "复制" for item in replay.button)
    assert any(item.label == "归档" for item in replay.button)
    assert any(item.label == "继续这次教学" for item in replay.button)


def test_evaluation_quick_mode_renders_and_exports():
    report = EvaluationRunner().run(load_cases()[:2])
    app = AppTest.from_file(str(APP))
    app.session_state["evaluation_report"] = report.model_dump(mode="json")
    app = app.run(timeout=30).switch_page("app_pages/evaluation_lab.py").run(timeout=30)
    assert not app.exception
    assert any(item.label == "运行模式" for item in app.segmented_control)
    assert app.download_button


def test_live_demo_fill_export_and_reset_buttons():
    app = AppTest.from_file(str(APP)).run(timeout=30)
    app = button(app, "开始教学").click().run(timeout=30)
    session = app.session_state["teaching_session"]
    app = open_active_session(session)
    assert any(item.label == "生成 3 条 AI 推荐回答" for item in app.button)
    export = next(item for item in app.download_button if item.label == "导出当前会话")
    app = export.click().run(timeout=30)
    assert not app.exception
    app = button(app, "保存并新建会话").click().run(timeout=30)
    assert app.session_state["teaching_session"] is None


def test_live_continue_button_keeps_the_same_session_and_history():
    app = AppTest.from_file(str(APP)).run(timeout=30)
    app = button(app, "开始教学").click().run(timeout=30)
    session = app.session_state["teaching_session"]
    session.status = "unable"
    session.termination_reason = "达到最大教学轮数 8。"
    session.rounds_in_current_run = 8
    session.turns[-1].policy_rule = "max_rounds"
    original_id = session.session_id
    original_turns = len(session.turns)
    original_route_index = session.teaching_route.current_index

    app = open_active_session(session)
    assert any(item.label == "从当前进度继续" for item in app.button)
    app = button(app, "从当前进度继续").click().run(timeout=30)
    resumed = app.session_state["teaching_session"]
    assert not app.exception
    assert resumed.session_id == original_id
    assert len(resumed.turns) == original_turns + 1
    assert resumed.rounds_in_current_run == 0
    assert resumed.teaching_route.current_index == original_route_index
    assert resumed.status == "active"


def test_replay_import_collision_edit_copy_archive_delete_restore():
    live = AppTest.from_file(str(APP)).run(timeout=30)
    live = button(live, "开始教学").click().run(timeout=30)
    session = live.session_state["teaching_session"]
    payload = session.model_dump_json().encode("utf-8")

    replay = AppTest.from_file(str(APP)).run(timeout=30).switch_page("app_pages/session_replay.py").run(timeout=30)
    replay = replay.file_uploader[0].set_value(("session.json", payload, "application/json")).run(timeout=30)
    replay = button(replay, "导入并保存").click().run(timeout=30)
    assert not replay.exception

    replay = AppTest.from_file(str(APP)).run(timeout=30).switch_page("app_pages/session_replay.py").run(timeout=30)
    count_before = len(replay.selectbox[3].options) if len(replay.selectbox) > 3 else 0
    replay = button(replay, "复制").click().run(timeout=30)
    assert not replay.exception

    replay = AppTest.from_file(str(APP)).run(timeout=30).switch_page("app_pages/session_replay.py").run(timeout=30)
    replay = button(replay, "归档").click().run(timeout=30)
    assert not replay.exception
    replay = AppTest.from_file(str(APP)).run(timeout=30).switch_page("app_pages/session_replay.py").run(timeout=30)
    assert any(item.label in {"归档", "取消归档"} for item in replay.button)
    assert count_before >= 0


def test_evaluation_import_clear_archive_and_restore_buttons():
    report = EvaluationRunner(seed=23).run(load_cases()[:1])
    payload = report.model_dump_json().encode("utf-8")
    app = AppTest.from_file(str(APP)).run(timeout=30).switch_page("app_pages/evaluation_lab.py").run(timeout=30)
    app = app.file_uploader[0].set_value(("evaluation.json", payload, "application/json")).run(timeout=30)
    app = button(app, "导入并载入").click().run(timeout=30)
    assert app.session_state["evaluation_report"] is not None
    app = button(app, "清除页面结果").click().run(timeout=30)
    assert app.session_state["evaluation_report"] is None

    app = AppTest.from_file(str(APP)).run(timeout=30).switch_page("app_pages/evaluation_lab.py").run(timeout=30)
    app = button(app, "载入报告").click().run(timeout=30)
    assert app.session_state["evaluation_report"] is not None
    if any(item.label == "归档报告" for item in app.button):
        app = button(app, "归档报告").click().run(timeout=30)
        assert not app.exception
        app = AppTest.from_file(str(APP)).run(timeout=30).switch_page("app_pages/evaluation_lab.py").run(timeout=30)
        assert any(item.label == "取消归档报告" for item in app.button)
        app = button(app, "取消归档报告").click().run(timeout=30)
        assert not app.exception

    app = AppTest.from_file(str(APP)).run(timeout=30).switch_page("app_pages/evaluation_lab.py").run(timeout=30)
    report_name = get_evaluation_store().list_reports()[0].name
    app = button(app, "确认移入回收站").click().run(timeout=30)
    assert report_name in get_evaluation_store().list_trash()
    app = AppTest.from_file(str(APP)).run(timeout=30).switch_page("app_pages/evaluation_lab.py").run(timeout=30)
    app = button(app, "恢复评估报告").click().run(timeout=30)
    assert get_evaluation_store().list_reports()


def test_replay_edit_delete_and_restore_dialog_buttons():
    live = AppTest.from_file(str(APP)).run(timeout=30)
    live = button(live, "开始教学").click().run(timeout=30)
    session_id = live.session_state["teaching_session"].session_id
    replay = AppTest.from_file(str(APP)).run(timeout=30).switch_page("app_pages/session_replay.py").run(timeout=30)

    name_input = next(item for item in replay.text_input if item.label == "展示名称")
    replay = name_input.set_value("答辩演示会话").run(timeout=30)
    replay = button(replay, "保存修改").click().run(timeout=30)
    assert get_store().load(session_id).display_title == "答辩演示会话"

    replay = AppTest.from_file(str(APP)).run(timeout=30).switch_page("app_pages/session_replay.py").run(timeout=30)
    confirm = next(item for item in replay.checkbox if item.label == "我确认删除这份会话档案")
    replay = confirm.set_value(True).run(timeout=30)
    replay = button(replay, "移入回收站").click().run(timeout=30)
    assert session_id in get_store().list_trash()

    replay = AppTest.from_file(str(APP)).run(timeout=30).switch_page("app_pages/session_replay.py").run(timeout=30)
    replay = button(replay, "恢复会话").click().run(timeout=30)
    assert get_store().load(session_id).session_id == session_id


def test_evaluation_all_views_and_download_buttons():
    report = EvaluationRunner(seed=31).run(load_cases()[:2])
    expected_downloads = {
        "完整 JSON",
        "总体 CSV",
        "正式 Markdown 报告",
        "八维盲评表",
        "解盲密钥（单独保管）",
    }
    for label in expected_downloads:
        app = AppTest.from_file(str(APP))
        app.session_state["evaluation_report"] = report.model_dump(mode="json")
        app = app.run(timeout=30).switch_page("app_pages/evaluation_lab.py").run(timeout=30)
        assert expected_downloads <= {item.label for item in app.download_button}
        current = next(item for item in app.download_button if item.label == label)
        app = current.click().run(timeout=30)
        assert not app.exception

    for view in ["结果总览", "统计检验", "行为盲评", "逐案例", "验收证据"]:
        app = AppTest.from_file(str(APP))
        app.session_state["evaluation_report"] = report.model_dump(mode="json")
        app = app.run(timeout=30).switch_page("app_pages/evaluation_lab.py").run(timeout=30)
        control = next(item for item in app.segmented_control if item.label == "查看评估内容")
        app = control.set_value(view).run(timeout=30)
        assert not app.exception


def test_evaluation_empty_archive_filter_and_quick_run_button():
    report = EvaluationRunner(seed=37).run(load_cases()[:1])
    get_evaluation_store().import_report(report)
    app = AppTest.from_file(str(APP)).run(timeout=30).switch_page("app_pages/evaluation_lab.py").run(timeout=30)
    archive_filter = next(item for item in app.text_input if item.label == "筛选归档")
    app = archive_filter.set_value("no-such-report").run(timeout=30)
    assert not app.exception
    assert any("没有匹配" in item.value for item in app.caption)

    app = AppTest.from_file(str(APP)).run(timeout=30).switch_page("app_pages/evaluation_lab.py").run(timeout=30)
    app = button(app, "运行快速演示").click().run(timeout=60)
    assert not app.exception
    generated = app.session_state["evaluation_report"]
    assert len(generated["case_results"]) == 54


def test_separate_browser_sessions_do_not_share_teaching_state():
    first = AppTest.from_file(str(APP)).run(timeout=30)
    first = button(first, "开始教学").click().run(timeout=30)
    second = AppTest.from_file(str(APP)).run(timeout=30)
    assert first.session_state["teaching_session"] is not None
    assert second.session_state["teaching_session"] is None


def test_skill_library_import_conflict_rename_search_and_export_buttons():
    source = APP.parent / "data" / "skills" / "diagnostic_questioning_v1.yaml"
    imported_id = "diagnostic_questioning_ui_acceptance_v1"
    target = Path(os.environ["TEACHING_AGENT_SKILL_DIR"]) / f"{imported_id}.yaml"
    try:
        app = AppTest.from_file(str(APP)).run(timeout=30).switch_page("app_pages/skill_library.py").run(timeout=30)
        assert not app.exception
        assert any("Agent 执行前的四项硬约束" in item.value for item in app.markdown)
        app = app.file_uploader[0].set_value(
            (source.name, source.read_bytes(), "application/x-yaml")
        ).run(timeout=30)
        new_id = next(item for item in app.text_input if item.label == "新的 Skill ID")
        app = new_id.set_value(imported_id).run(timeout=30)
        app = button(app, "确认导入").click().run(timeout=30)
        assert not app.exception
        assert target.exists()

        search = next(item for item in app.text_input if item.label == "搜索 Skill")
        app = search.set_value(imported_id).run(timeout=30)
        assert any(imported_id in item.value for item in app.caption)
        export = next(item for item in app.download_button if item.label == "导出 YAML")
        app = export.click().run(timeout=30)
        assert not app.exception
    finally:
        target.unlink(missing_ok=True)
        get_library.clear()
