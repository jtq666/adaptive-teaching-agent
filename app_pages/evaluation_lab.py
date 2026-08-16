from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st

from src.evaluation import (
    METHOD_DESCRIPTIONS,
    EvaluationRunner,
    load_cases,
    summarize_human_annotations,
    validate_human_annotation_csv,
)
from src.models import EvaluationReport
from src.ui import get_evaluation_store, get_library, page_header, set_notice

METHOD_COLORS = {
    "自适应混合 Agent": "#176B87",
    "固定单 Skill": "#D89B3C",
    "无 Skill 通用 Agent": "#8A99A1",
}
METRIC_LABELS = {
    "decision_quality": "决策质量",
    "behavior_quality": "行为代理分",
    "normalized_gain": "标准化增益",
    "transfer_accuracy": "迁移率",
}
DIMENSION_NAMES = {
    "knowledge_correctness_proxy": "知识正确性代理",
    "clarity_proxy": "清晰度代理",
    "targeting_proxy": "针对性代理",
    "promotes_thinking": "促进思考",
    "answer_non_revealing": "不直接给答案",
    "actionability": "可执行性",
    "coherence": "上下文连贯",
    "tutor_tone": "教师语气",
}


@st.cache_data(max_entries=1)
def cached_cases():
    return load_cases()


page_header(
    "Agent 评估",
    "比较自适应 Agent、固定 Skill 和无 Skill 基线，并明确区分模拟证据与真实教学效果。",
    eyebrow="评估实验室 · 证据先于结论",
    icon="experiment",
)

with st.expander("先看实验边界", expanded=False, icon=":material/science:"):
    st.write("三种方法共享目标、画像、初始真实状态、回答模板、状态诊断器和最多 8 轮预算。")
    for method, description in METHOD_DESCRIPTIONS.items():
        st.markdown(f"**{method}** · {description}")
    st.caption("学习变化函数不读取方法名称；结果同时报告点估计、置信区间和失败案例。")

evidence_row = st.container(horizontal=True)
with evidence_row:
    st.badge("策略仿真 · 当前可运行", color="green", icon=":material/science:")
    st.badge("真实模型稳定性 · 需在线脚本", color="blue", icon=":material/cloud:")
    st.badge("人工盲评 · 导入两人标注后完成", color="orange", icon=":material/groups:")
st.caption("当前快速/完整评估只回答策略闭环是否可复现；它不会自动升级为真实课堂效果结论。")

cases = cached_cases()
existing_report = st.session_state.evaluation_report
evaluation_store = get_evaluation_store()


with st.container(horizontal=True, vertical_alignment="center"):
    with st.popover("历史评估", icon=":material/history:"):
        archive_query = st.text_input("筛选归档", placeholder="输入日期、模式或文件名")
        report_rows, total_reports = evaluation_store.list_report_metadata(
            page=1,
            page_size=20,
            query=archive_query,
        )
        report_names = [str(row["file_name"]) for row in report_rows]
        if report_names:
            if st.session_state.get("selected_evaluation") not in report_names:
                st.session_state.selected_evaluation = report_names[0]
            selected_report = st.selectbox(
                "选择归档",
                report_names,
                key="selected_evaluation",
                format_func=lambda name: name.removeprefix("evaluation_").removesuffix(".json"),
            )
            st.caption(f"显示 {len(report_names)} / {total_reports} 份可复核报告（仅读取索引）")
            with st.container(horizontal=True):
                if st.button("载入报告", type="primary", icon=":material/folder_open:"):
                    try:
                        loaded = evaluation_store.load(selected_report)
                        st.session_state.evaluation_report = loaded.model_dump(mode="json")
                        set_notice("历史评估已载入。", ":material/history:")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"无法载入报告：{exc}", icon=":material/error:")
                with st.popover("删除档案", icon=":material/delete:"):
                    st.warning(
                        "将 JSON、CSV、Markdown 和盲评材料成组移入回收站，可稍后恢复。",
                        icon=":material/warning:",
                    )
                    if st.button(
                        "确认移入回收站",
                        type="primary",
                        icon=":material/delete:",
                        width="stretch",
                    ):
                        removed = evaluation_store.delete_bundle(selected_report)
                        st.session_state.evaluation_report = None
                        set_notice(f"已将 {len(removed)} 个评估文件移入回收站。", ":material/delete:")
                        st.rerun()
                if st.button("归档报告", icon=":material/archive:"):
                    moved = evaluation_store.archive(selected_report)
                    set_notice(f"已成组归档 {len(moved)} 个评估文件")
                    st.rerun()
        else:
            st.caption(
                "没有匹配的评估报告，请清除筛选词。"
                if archive_query.strip()
                else "还没有历史评估。运行一次完整评估后会自动归档。"
            )

    with st.popover("导入评估 JSON", icon=":material/upload:"):
        uploaded_report = st.file_uploader(
            "选择评估 JSON",
            type=["json"],
            label_visibility="collapsed",
            key="evaluation_upload",
        )
        if st.button(
            "导入并载入",
            type="primary",
            disabled=uploaded_report is None,
            icon=":material/upload_file:",
            width="stretch",
        ):
            try:
                if uploaded_report is None:
                    raise ValueError("请选择评估 JSON 文件")
                imported = EvaluationReport.model_validate_json(uploaded_report.getvalue())
                evaluation_store.import_report(imported)
                st.session_state.evaluation_report = imported.model_dump(mode="json")
                set_notice("评估报告已导入并归档。", ":material/upload_file:")
                st.rerun()
            except Exception as exc:
                st.error(f"无法导入报告：{exc}", icon=":material/error:")

    with st.popover("评估回收站", icon=":material/restore_from_trash:"):
        trashed_reports = evaluation_store.list_trash()
        if trashed_reports:
            restore_report = st.selectbox("选择待恢复报告", trashed_reports)
            if st.button("恢复评估报告", icon=":material/restore:", width="stretch"):
                evaluation_store.restore_bundle(restore_report)
                set_notice("评估报告已恢复")
                st.rerun()
        else:
            st.caption("回收站为空")

    with st.popover("已归档报告", icon=":material/inventory_2:"):
        archived_reports = evaluation_store.list_archived()
        if archived_reports:
            archived_report = st.selectbox("选择待取消归档报告", archived_reports)
            if st.button("取消归档报告", icon=":material/unarchive:", width="stretch"):
                restored = evaluation_store.restore_archive(archived_report)
                set_notice(f"已恢复 {len(restored)} 个归档文件")
                st.rerun()
        else:
            st.caption("没有已归档报告")

    st.caption("评估报告自动归档，可随时载入、导出或移入回收站。")

with st.container(border=True):
    evaluation_mode = st.segmented_control(
        "运行模式",
        ["快速演示", "完整评估"],
        default="快速演示",
        required=True,
    )
    design_runs = len(cases) * 3 if evaluation_mode == "快速演示" else len(cases) * 3 * 3 * 5
    row = st.container(horizontal=True, vertical_alignment="center")
    with row:
        st.metric("测试案例", len(cases), border=True)
        st.metric("开发 / 留出", "6 / 12", border=True)
        st.metric("评估单元", design_runs, border=True)
        run_label = "运行快速演示" if evaluation_mode == "快速演示" else "运行完整评估"
        run_clicked = st.button(
            f"重新{run_label}" if existing_report else run_label,
            type="primary",
            icon=":material/play_arrow:",
        )
        if existing_report and st.button("清除页面结果", icon=":material/close:"):
            st.session_state.evaluation_report = None
            set_notice("已清除页面中的评估结果，历史导出文件仍保留。", ":material/delete_sweep:")
            st.rerun()
    st.caption(f"离线运行，不消耗 LLM API · 预计 {design_runs} 个配对方法单元。")

if run_clicked:
    with st.status(f"正在运行 {design_runs} 个可复现评估单元…", expanded=True) as status:
        try:
            st.write(f"加载 {len(get_library().skills)} 个 Teaching Skill")
            st.write(f"运行 {len(cases)} 个案例 × 3 种方法")
            st.write("计算 bootstrap、置换检验、Holm 校正、McNemar 与盲评材料")
            generated = EvaluationRunner(get_library()).run(cases, mode="quick" if evaluation_mode == "快速演示" else "full")
            st.session_state.evaluation_report = generated.model_dump(mode="json")
            status.update(label="评估完成", state="complete", expanded=False)
            set_notice("评估完成，结果和盲评材料已经生成。")
            st.rerun()
        except Exception as exc:
            status.update(label="评估未完成", state="error", expanded=True)
            st.error(f"评估失败，已有结果未被覆盖：{exc}", icon=":material/error:")

raw_report = st.session_state.evaluation_report
if not raw_report:
    st.info("运行后可查看总体差异、统计区间、八维行为代理、逐案例轨迹和盲评材料。", icon=":material/info:")
    case_df = pd.DataFrame(
        [
            {
                "案例": case.title,
                "课程": case.goal.course,
                "数据划分": "开发集" if case.split == "development" else "冻结留出集",
                "前测": case.pretest_score,
                "主要困难": case.true_misconceptions[0],
            }
            for case in cases
        ]
    )
    st.dataframe(case_df, hide_index=True)
    st.stop()

report = EvaluationReport.model_validate(raw_report)
if (
    not report.summary
    or "simulation_runs" not in report.summary[0]
    or not report.case_results
    or "knowledge_correctness_proxy" not in report.case_results[0].behavior_dimensions
    or "single_step_contract_rate" not in report.summary[0]
):
    st.warning("当前页面缓存的是旧版评估结果，请点击“重新运行评估”升级统计口径。", icon=":material/update:")
    st.stop()

summary_df = pd.DataFrame(report.summary)
adaptive = summary_df[summary_df["method"] == "自适应混合 Agent"].iloc[0]
fixed = summary_df[summary_df["method"] == "固定单 Skill"].iloc[0]
runner = EvaluationRunner(get_library(), seed=report.seed)

result_bar = st.container(horizontal=True, vertical_alignment="center")
with result_bar:
    st.badge(f"随机种子 {report.seed}", icon=":material/casino:", color="blue")
    st.caption(f"生成时间：{report.generated_at}")
    with st.popover("导出结果", icon=":material/download:"):
        st.download_button(
            "完整 JSON",
            data=json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
            file_name="teacher_agent_evaluation.json",
            mime="application/json",
            icon=":material/data_object:",
            width="stretch",
        )
        st.download_button(
            "总体 CSV",
            data=summary_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="teacher_agent_summary.csv",
            mime="text/csv",
            icon=":material/table_view:",
            width="stretch",
        )
        st.download_button(
            "正式 Markdown 报告",
            data=runner.to_markdown(report),
            file_name="teacher_agent_evaluation.md",
            mime="text/markdown",
            icon=":material/description:",
            width="stretch",
        )
        st.download_button(
            "八维盲评表",
            data=runner.human_annotation_csv(report).encode("utf-8-sig"),
            file_name="teacher_response_blind_annotation.csv",
            mime="text/csv",
            icon=":material/rate_review:",
            width="stretch",
        )
        st.download_button(
            "解盲密钥（单独保管）",
            data=runner.human_annotation_csv(report, reveal_key=True).encode("utf-8-sig"),
            file_name="teacher_response_annotation_key.csv",
            mime="text/csv",
            icon=":material/key:",
            width="stretch",
        )

view = st.segmented_control(
    "查看评估内容",
    ["结果总览", "统计检验", "行为盲评", "逐案例", "验收证据"],
    default="结果总览",
    required=True,
    key="evaluation_view",
)

if view == "结果总览":
    with st.container(horizontal=True):
        st.metric("决策质量", f"{adaptive['decision_quality']:.1%}", border=True)
        st.metric("标准化学习增益", f"{adaptive['normalized_gain']:.1%}", border=True)
        st.metric("单位轮次增益", f"{adaptive['learning_efficiency']:.1%}", border=True)
        st.metric("迁移正确率", f"{adaptive['transfer_accuracy']:.1%}", border=True)
    with st.container(horizontal=True):
        st.metric("单步契约通过", f"{adaptive['single_step_contract_rate']:.1%}", border=True)
        st.metric("上下文连续", f"{adaptive['context_continuity_rate']:.1%}", border=True)
        st.metric("选项有效", f"{adaptive['option_validity_rate']:.1%}", border=True)
        st.metric("平均调用", f"{adaptive['mean_llm_calls']:.1f} 次/轮", border=True)

    if fixed["normalized_gain"] > adaptive["normalized_gain"]:
        st.warning(
            f"固定 Skill 讲满 {fixed['mean_rounds']:.0f} 轮，标准化增益更高；自适应方法平均 "
            f"{adaptive['mean_rounds']:.1f} 轮终止。请分别比较单位轮次效率、决策质量与迁移率，"
            "不要把任一指标解释为全面领先。",
            icon=":material/balance:",
        )
    else:
        st.info(
            f"自适应方法的标准化增益比固定 Skill 高 "
            f"{adaptive['normalized_gain'] - fixed['normalized_gain']:.1%}；"
            f"迁移率分别为 {adaptive['transfer_accuracy']:.1%} 与 {fixed['transfer_accuracy']:.1%}。"
            "各指标可能给出不同排序，结论应结合配对区间和失败案例。",
            icon=":material/balance:",
        )

    chart_rows = [
        {"方法": item["method"], "指标": METRIC_LABELS[metric], "得分": item[metric]}
        for item in report.summary
        for metric in METRIC_LABELS
    ]
    chart = (
        alt.Chart(pd.DataFrame(chart_rows))
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("指标:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("得分:Q", title="得分（0–1）", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "方法:N",
                scale=alt.Scale(domain=list(METHOD_COLORS), range=list(METHOD_COLORS.values())),
            ),
            xOffset="方法:N",
            tooltip=["方法", "指标", alt.Tooltip("得分:Q", format=".3f")],
        )
        .properties(title="同一套指标下的三种策略表现", height=340)
    )
    st.altair_chart(chart)

    compact = summary_df.rename(
        columns={
            "method": "方法",
            "state_f1": "状态 F1",
            "mastery_mae": "掌握误差↓",
            "decision_quality": "决策质量",
            "behavior_quality": "行为代理分",
            "normalized_gain": "标准化增益",
            "learning_efficiency": "单位轮次增益",
            "mean_rounds": "平均轮数",
            "transfer_accuracy": "迁移率",
            "success_rate": "成功率",
            "single_step_contract_rate": "单步契约通过",
            "context_continuity_rate": "上下文连续",
            "option_validity_rate": "选项有效",
            "evidence_mapping_accuracy": "证据映射",
        }
    )
    compact = compact[["方法", "状态 F1", "掌握误差↓", "决策质量", "行为代理分", "标准化增益", "单位轮次增益", "平均轮数", "迁移率", "成功率", "单步契约通过", "上下文连续", "选项有效", "证据映射"]]
    st.dataframe(
        compact,
        hide_index=True,
        column_config={
            column: st.column_config.NumberColumn(format="percent")
            for column in ["状态 F1", "决策质量", "行为代理分", "标准化增益", "单位轮次增益", "迁移率", "成功率", "单步契约通过", "上下文连续", "选项有效", "证据映射"]
        },
    )

    success_col, failure_col = st.columns(2)
    with success_col.container(border=True, height="stretch"):
        st.subheader("典型成功")
        st.code(report.successful_case["case_id"], language=None)
        st.write(report.successful_case["reason"])
    with failure_col.container(border=True, height="stretch"):
        st.subheader("真实失败")
        st.code(report.failure_case["case_id"], language=None)
        st.write(report.failure_case["reason"])

elif view == "统计检验":
    st.subheader("配对效应与不确定性")
    comparison_df = pd.DataFrame(report.paired_comparisons)
    comparison_display = comparison_df.assign(
        **{
            "95% CI": comparison_df.apply(lambda row: f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]", axis=1),
            "结论": comparison_df.apply(
                lambda row: "方向稳定" if row["ci_low"] > 0 or row["ci_high"] < 0 else "证据不足（CI 跨 0）",
                axis=1,
            ),
        }
    ).rename(
        columns={
            "baseline": "对比基线",
            "metric": "指标",
            "n_pairs": "配对案例",
            "adaptive_mean": "自适应均值",
            "baseline_mean": "基线均值",
            "mean_difference": "均值差",
            "hedges_g_paired": "配对 Hedges g",
            "win_rate": "胜率",
        }
    )
    st.dataframe(
        comparison_display[["对比基线", "指标", "配对案例", "自适应均值", "基线均值", "均值差", "95% CI", "配对 Hedges g", "胜率", "结论"]],
        hide_index=True,
        column_config={"胜率": st.column_config.NumberColumn(format="percent")},
    )
    st.caption("先在案例内汇总画像与种子，再以 18 个案例为独立配对单位 bootstrap 5000 次；同时报告区间、效应量和校正后检验。")
    if report.statistical_tests:
        st.subheader("稳健性检验")
        test_df = pd.DataFrame(report.statistical_tests).rename(
            columns={
                "baseline": "对比基线",
                "metric": "指标",
                "permutation_p": "配对置换 p",
                "holm_adjusted_p": "Holm 校正 p",
                "mcnemar_b": "迁移 自适应胜",
                "mcnemar_c": "迁移 基线胜",
                "mcnemar_exact_p": "McNemar 精确 p",
            }
        )
        st.dataframe(test_df, hide_index=True)

    ci_table = summary_df.assign(
        **{
            "增益 95% CI": summary_df.apply(lambda row: f"[{row['gain_ci_low']:.3f}, {row['gain_ci_high']:.3f}]", axis=1),
            "成功率 95% CI": summary_df.apply(lambda row: f"[{row['success_ci_low']:.3f}, {row['success_ci_high']:.3f}]", axis=1),
        }
    ).rename(columns={"method": "方法", "normalized_gain": "标准化增益", "success_rate": "成功率"})
    st.dataframe(ci_table[["方法", "标准化增益", "增益 95% CI", "成功率", "成功率 95% CI"]], hide_index=True)

    st.subheader("按课程分层")
    stratified = pd.DataFrame(report.stratified_summary).rename(
        columns={"course": "课程", "method": "方法", "cases": "案例数", "normalized_gain": "标准化增益", "decision_quality": "决策质量", "transfer_accuracy": "迁移率"}
    )
    st.dataframe(stratified, hide_index=True)

elif view == "行为盲评":
    st.subheader("教学行为八维规则代理")
    dimension_rows = []
    for method in report.methods:
        method_results = [item for item in report.case_results if item.method == method]
        dimension_row: dict[str, object] = {"方法": method}
        for key, label in DIMENSION_NAMES.items():
            dimension_row[label] = sum(item.behavior_dimensions.get(key, 0) for item in method_results) / len(method_results)
        dimension_rows.append(dimension_row)
    st.dataframe(
        pd.DataFrame(dimension_rows),
        hide_index=True,
        column_config={label: st.column_config.NumberColumn(format="percent") for label in DIMENSION_NAMES.values()},
    )
    st.info(
        "规则代理分只用于自动回归。正式结论应由至少两名评审使用盲评表独立标注，并报告一致性。",
        icon=":material/groups:",
    )
    proxy_values = [item["behavior_quality"] for item in report.summary]
    if proxy_values and max(proxy_values) >= 0.95:
        st.warning(
            "当前行为代理出现天花板效应，不能用来证明某种方法教学质量更高。"
            "页面只作描述展示，配对显著性和效应量分析已排除该代理指标。",
            icon=":material/vertical_align_top:",
        )
    st.warning(report.human_evaluation_status, icon=":material/pending_actions:")
    with st.expander("导入两名评审的盲评结果", expanded=False, icon=":material/upload_file:"):
        st.caption(
            "请使用下载的盲评表填写八个 1–5 分维度；每个样本至少由两名不同评审员独立评分。"
            "导入结果会作为独立附件保存，不会覆盖自动评估报告。"
        )
        annotation_upload = st.file_uploader(
            "选择已完成的盲评 CSV",
            type=["csv"],
            key="human_annotation_upload",
        )
        if st.button(
            "校验并保存人工评审",
            type="primary",
            disabled=annotation_upload is None,
            key="save_human_annotation",
            icon=":material/fact_check:",
        ):
            try:
                if annotation_upload is None:
                    raise ValueError("请选择人工盲评 CSV")
                rows = validate_human_annotation_csv(annotation_upload.getvalue())
                human_summary = summarize_human_annotations(rows)
                report_ref = str(st.session_state.get("selected_evaluation") or "evaluation_current.json")
                attachment = evaluation_store.save_human_review(report_ref, rows, human_summary)
                st.session_state.human_review_summary = human_summary
                st.success(
                    f"人工评审已保存为独立附件：{attachment.name}；加权 Cohen’s κ = "
                    f"{human_summary['weighted_cohen_kappa']:.3f}。",
                    icon=":material/task_alt:",
                )
            except (ValueError, OSError) as exc:
                st.error(f"人工评审未保存：{exc}", icon=":material/error:")
    if st.session_state.get("human_review_summary"):
        human_summary = st.session_state.human_review_summary
        st.subheader("已导入的人工一致性结果")
        metric_row = st.container(horizontal=True)
        with metric_row:
            st.metric("样本数", human_summary["sample_count"], border=True)
            st.metric("加权 Cohen’s κ", f"{human_summary['weighted_cohen_kappa']:.3f}", border=True)
            st.metric("分歧率", f"{human_summary['disagreement_rate']:.1%}", border=True)
        st.caption("人工评分作为独立附件保存；自动 proxy 分不因人工导入而被改写。")
    with st.container(horizontal=True):
        st.download_button(
            "下载盲评表",
            data=runner.human_annotation_csv(report).encode("utf-8-sig"),
            file_name="teacher_response_blind_annotation.csv",
            mime="text/csv",
            icon=":material/rate_review:",
        )
        st.download_button(
            "下载解盲密钥",
            data=runner.human_annotation_csv(report, reveal_key=True).encode("utf-8-sig"),
            file_name="teacher_response_annotation_key.csv",
            mime="text/csv",
            icon=":material/key:",
        )
    st.caption("请将解盲密钥与标注表分开交给不同人员保管。")

elif view == "逐案例":
    st.subheader("逐案例结果与动作轨迹")
    case_rows = [
        {
            "案例": item.case_id,
            "方法": item.method,
            "前测": item.pretest_score,
            "后测": item.posttest_score,
            "标准化增益": item.normalized_gain,
            "单位轮次增益": item.learning_efficiency,
            "轮数": item.rounds,
            "状态 F1": item.misconception_f1,
            "Skill 准确率": item.skill_selection_accuracy,
            "切换准确率": item.switch_accuracy,
            "迁移通过": item.transfer_accuracy,
            "动作轨迹": " → ".join(item.action_types),
        }
        for item in report.case_results
    ]
    case_result_df = pd.DataFrame(case_rows)
    gain_chart = (
        alt.Chart(case_result_df)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            y=alt.Y("案例:N", title=None),
            x=alt.X("标准化增益:Q", title="标准化学习增益", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("方法:N", scale=alt.Scale(domain=list(METHOD_COLORS), range=list(METHOD_COLORS.values()))),
            yOffset="方法:N",
            tooltip=["案例", "方法", "前测", "后测", alt.Tooltip("标准化增益:Q", format=".3f")],
        )
        .properties(title="逐案例学习增益", height=310)
    )
    st.altair_chart(gain_chart)
    st.dataframe(
        case_result_df,
        hide_index=True,
        column_config={
            column: st.column_config.NumberColumn(format="percent")
            for column in ["标准化增益", "单位轮次增益", "状态 F1", "Skill 准确率", "切换准确率", "迁移通过"]
        }
        | {"动作轨迹": st.column_config.TextColumn(width="large")},
    )

else:
    st.subheader("考核要求覆盖证据")
    coverage = pd.DataFrame(
        [
            ["学生状态显式维护", "掌握度、误解、证据、连续次数、理解信号、下一关注点", "实时教学 / 回放"],
            ["Skill 动态选择", "候选列表、决策来源、主辅 Skill、选择理由", "实时教学每轮证据卡"],
            ["切换与约束", "连续错误守卫、切换原因、禁止直接给答案守卫", "实时教学 / JSON"],
            ["终止判断", "掌握阈值 + 迁移通过；轮次或误解停滞终止", "实时教学 / 回放"],
            ["量化评估", "18 案例（6 开发 + 12 留出）、2 基线、配对统计、盲评导出", "Agent 评估"],
        ],
        columns=["要求", "可核验证据", "展示位置"],
    )
    st.dataframe(coverage, hide_index=True)
    st.subheader("论文依据")
    st.markdown(
        "- [MRBench / AI Tutor 八维评估 taxonomy](https://aclanthology.org/2025.naacl-long.57/)\n"
        "- [MathTutorBench：开放式教学能力 benchmark](https://aclanthology.org/2025.emnlp-main.11/)\n"
        "- [BEA 2023：教师回复的自动评测与人工成对评价](https://aclanthology.org/2023.bea-1.64/)\n"
        "- [Beyond normalized gain：前测差异与增益解释](https://doi.org/10.1103/PhysRevPhysEducRes.20.010123)\n"
        "- [约束式教师回复的人评维度](https://aclanthology.org/2024.sigdial-1.11/)"
    )
