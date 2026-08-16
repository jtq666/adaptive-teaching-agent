from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import random
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from src.agent import HybridTeachingAgent
from src.config import get_agent_settings, get_path
from src.llm import OpenAICompatibleClient
from src.models import (
    EvaluationCase,
    EvaluationReport,
    MethodCaseResult,
    StudentState,
)
from src.skills import SkillLibrary
from src.state_tracker import StateTracker
from src.storage import EvaluationStore, _atomic_json

METHODS = ["自适应混合 Agent", "固定单 Skill", "无 Skill 通用 Agent"]

METHOD_DESCRIPTIONS = {
    METHODS[0]: "每轮更新学生状态，经候选过滤后动态选择或切换 Skill，并执行迁移验证。",
    METHODS[1]: "全程固定使用首轮排名最高的学科 Skill；末轮仍获得一次验证机会，但不允许切换 Skill。",
    METHODS[2]: "不读取 Skill Library，使用通用解释与提问；末轮同样获得一次通用验证机会。",
}

HUMAN_RATING_FIELDS = (
    "知识正确性",
    "清晰度",
    "针对性",
    "促进思考",
    "不直接给答案",
    "上下文连贯",
    "教师语气",
    "可执行性",
)


def validate_human_annotation_csv(raw: bytes | str) -> list[dict[str, str]]:
    """Validate an independently completed blind-rating attachment.

    The automatic report is intentionally not modified by this operation.  A
    rating row is valid only when it identifies a sample and rater and gives
    integer 1--5 scores for all eight dimensions.  This keeps missing or
    malformed ratings from silently becoming a teaching-quality conclusion.
    """
    if isinstance(raw, bytes):
        if len(raw) > 5_000_000:
            raise ValueError("人工标注文件不能超过 5MB")
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("人工标注文件必须使用 UTF-8 编码") from exc
    else:
        text = raw
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = set(reader.fieldnames or [])
    required = {"样本ID", "评审员", *HUMAN_RATING_FIELDS}
    missing = sorted(required - fieldnames)
    if missing:
        raise ValueError("人工标注缺少字段：" + "、".join(missing))
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row_index, row in enumerate(reader, start=2):
        sample_id = str(row.get("样本ID", "")).strip()
        rater = str(row.get("评审员", "")).strip()
        if not sample_id or not rater:
            raise ValueError(f"第 {row_index} 行缺少样本 ID 或评审员")
        pair = (sample_id, rater)
        if pair in seen:
            raise ValueError(f"第 {row_index} 行与已有标注重复：{sample_id}/{rater}")
        seen.add(pair)
        normalized = {"样本ID": sample_id, "评审员": rater}
        for field in HUMAN_RATING_FIELDS:
            raw_score = str(row.get(field, "")).strip()
            try:
                score = int(raw_score)
            except ValueError as exc:
                raise ValueError(f"第 {row_index} 行的“{field}”必须是 1–5 的整数") from exc
            if score < 1 or score > 5:
                raise ValueError(f"第 {row_index} 行的“{field}”必须在 1–5 之间")
            normalized[field] = str(score)
        rows.append(normalized)
    if not rows:
        raise ValueError("人工标注文件没有数据行")
    return rows


def summarize_human_annotations(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Compute weighted Cohen's kappa and disagreement without scipy."""
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["样本ID"]].append(row)
    paired = {sample: items for sample, items in grouped.items() if len({item["评审员"] for item in items}) >= 2}
    if not paired:
        raise ValueError("至少需要同一批样本的两名不同评审员标注")

    def weighted_kappa(left: list[int], right: list[int]) -> float:
        n = len(left)
        if not n:
            return 0.0
        categories = range(1, 6)
        observed = sum(1.0 - ((a - b) / 4.0) ** 2 for a, b in zip(left, right, strict=True)) / n
        left_marginal = {category: left.count(category) / n for category in categories}
        right_marginal = {category: right.count(category) / n for category in categories}
        expected = sum(
            left_marginal[a] * right_marginal[b] * (1.0 - ((a - b) / 4.0) ** 2)
            for a in categories
            for b in categories
        )
        return 1.0 if expected >= 1.0 else (observed - expected) / max(1e-9, 1.0 - expected)

    kappas: dict[str, float] = {}
    disagreements: list[float] = []
    for dimension in HUMAN_RATING_FIELDS:
        left_scores: list[int] = []
        right_scores: list[int] = []
        for items in paired.values():
            raters = sorted({item["评审员"] for item in items})[:2]
            first = next(item for item in items if item["评审员"] == raters[0])
            second = next(item for item in items if item["评审员"] == raters[1])
            left_scores.append(int(first[dimension]))
            right_scores.append(int(second[dimension]))
        kappas[dimension] = round(weighted_kappa(left_scores, right_scores), 3)
        disagreements.extend(float(left != right) for left, right in zip(left_scores, right_scores, strict=True))
    return {
        "sample_count": len(paired),
        "row_count": len(rows),
        "rater_count": len({row["评审员"] for row in rows}),
        "weighted_cohen_kappa": round(mean(kappas.values()), 3),
        "dimension_kappa": kappas,
        "disagreement_rate": round(mean(disagreements), 3) if disagreements else 0.0,
        "status": "已完成双人盲评" if len({row["评审员"] for row in rows}) >= 2 else "待完成双人盲评",
    }


class _NoOpSessionStore:
    """评估运行不写入人工教学回放目录。"""

    @staticmethod
    def save(session):
        return Path(f"{session.session_id}.json")


def load_cases(path: Path | None = None) -> list[EvaluationCase]:
    source = path or get_path("cases")
    with source.open("r", encoding="utf-8") as handle:
        development = [EvaluationCase.model_validate(item) for item in json.load(handle)]
    held_out_source = source.with_name(f"{source.stem}_heldout{source.suffix}")
    if not held_out_source.exists():
        if path is None:
            raise FileNotFoundError(f"缺少冻结留出集：{held_out_source}")
        held_out: list[EvaluationCase] = []
    else:
        with held_out_source.open("r", encoding="utf-8") as handle:
            held_out = [EvaluationCase.model_validate(item) for item in json.load(handle)]
    for case in development:
        case.split = "development"
        if case.data_version == "dev-v1":
            case.data_version = "development-v1"
    for case in held_out:
        case.split = "held_out"
        if case.data_version == "dev-v1":
            case.data_version = "heldout-v1"
    combined = development + held_out
    ids = [case.case_id for case in combined]
    if len(ids) != len(set(ids)):
        raise ValueError("评估案例 ID 重复，开发集与留出集必须独立")
    return combined


def _fuzzy_match(left: str, right: str) -> bool:
    """Match labels without relying on a fixed list of subject keywords.

    Evaluation labels are short, so word tokenizers are unreliable for Chinese
    and mixed-language courses.  Normalized character n-grams provide a small,
    deterministic and domain-neutral similarity check instead.
    """

    def normalize(value: str) -> str:
        return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", value).lower()

    def ngrams(value: str, size: int = 2) -> set[str]:
        if len(value) <= size:
            return {value} if value else set()
        return {value[index : index + size] for index in range(len(value) - size + 1)}

    normalized_left = normalize(left)
    normalized_right = normalize(right)
    if not normalized_left or not normalized_right:
        return normalized_left == normalized_right
    if normalized_left == normalized_right:
        return True
    if min(len(normalized_left), len(normalized_right)) >= 2 and (
        normalized_left in normalized_right or normalized_right in normalized_left
    ):
        return True
    left_grams = ngrams(normalized_left)
    right_grams = ngrams(normalized_right)
    union = left_grams | right_grams
    return bool(union) and len(left_grams & right_grams) / len(union) >= 0.5


def _prf(predicted: list[str], truth: list[str]) -> tuple[float, float, float]:
    if not predicted and not truth:
        return 1.0, 1.0, 1.0
    matches = sum(any(_fuzzy_match(item, expected) for expected in truth) for item in predicted)
    precision = matches / len(predicted) if predicted else 0.0
    recall = matches / len(truth) if truth else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


SIMULATOR_PROFILES = {
    "cautious": {"gain": 0.82, "noise": 0.012},
    "balanced": {"gain": 1.0, "noise": 0.008},
    "fast_transfer": {"gain": 1.12, "noise": 0.010},
}


class DeterministicStudent:
    def __init__(self, case: EvaluationCase, profile: str = "balanced", seed: int = 0):
        self.case = case
        self.mastery = deepcopy(case.initial_mastery)
        self.initial_average = mean(case.initial_mastery.values())
        self.misconceptions = list(case.true_misconceptions)
        self.transfer_passed = False
        self.transfer_attempts = 0
        self.turn_count = 0
        self.profile = profile
        self.rng = random.Random(seed)
        self.action_counts: dict[str, int] = defaultdict(int)

    @staticmethod
    def _action_family(action_type: str) -> str:
        if action_type in {"transfer", "fixed_verification", "generic_verification"}:
            return "verification"
        return action_type

    @staticmethod
    def _diminishing_multiplier(repetition: int) -> float:
        """Method-blind diminishing returns for repeating the same teaching action."""
        return max(0.30, 1.0 - 0.18 * max(0, repetition - 1))

    def respond(self, action_type: str, method: str, aligned: bool) -> str:
        del method  # public compatibility argument; the simulator is method-blind
        self.turn_count += 1
        family = self._action_family(action_type)
        self.action_counts[family] += 1
        # 学习变化只由教学动作及其与当前困难的匹配程度决定，不能按方法名称预设优势。
        # The simulator is calibrated to make an eight-turn teaching budget
        # capable of reaching the protocol threshold on strong starting
        # cases, while weak cases can still remain unresolved. These are
        # frozen method-blind parameters, not tuned from a method's result.
        base_gain = 0.040
        bonus = 0.0
        if aligned:
            bonus += 0.050
        if action_type == "diagnostic":
            bonus += 0.015
        elif action_type == "scaffold":
            bonus += 0.040
        elif action_type == "correction" and self.misconceptions:
            bonus += 0.065
        elif action_type in {"transfer", "fixed_verification", "generic_verification"}:
            bonus += 0.015
        elif action_type == "subject_instruction":
            bonus += 0.050
        elif action_type == "generic":
            bonus += 0.035

        params = SIMULATOR_PROFILES[self.profile]
        diminishing = self._diminishing_multiplier(self.action_counts[family])
        gain = max(
            0.0,
            (base_gain + bonus) * params["gain"] * diminishing
            + self.rng.uniform(-params["noise"], params["noise"]),
        )
        # Evidence improves one hidden knowledge point at a time.  A fluent
        # answer about one concept must not raise every point in the goal.
        target_point = min(
            self.mastery,
            key=lambda point: (self.mastery[point], list(self.mastery).index(point)),
        )
        self.mastery[target_point] = min(0.96, self.mastery[target_point] + gain)
        average = mean(self.mastery.values())
        resolution_threshold = {
            "correction": 0.48,
            "subject_instruction": 0.54,
            "generic": 0.58,
        }.get(action_type)
        if self.misconceptions and resolution_threshold is not None and self.mastery[target_point] >= resolution_threshold:
            self.misconceptions.clear()
        if self.turn_count == 1 and self.misconceptions and average < 0.62:
            return self.case.responses["confused"]
        # All methods face the same hidden transfer threshold. Different
        # thresholds would bake the expected ranking into the simulator.
        verification_threshold = 0.72 if action_type in {
            "transfer", "fixed_verification", "generic_verification"
        } else None
        if verification_threshold is not None and self.mastery[target_point] >= verification_threshold:
            self.transfer_attempts += 1
            if self.initial_average < 0.30 and self.transfer_attempts == 1:
                return self.case.responses["partial"]
            self.transfer_passed = True
            return self.case.responses["transfer"]
        if aligned and self.turn_count > 1 and action_type in {"scaffold", "correction", "subject_instruction", "fixed_verification"}:
            # Once the current point has accumulated enough hidden evidence,
            # expose a correct explanation instead of emitting the same
            # partial template forever. This is method-blind: every method
            # receives the same response rule, shared initial state and noise.
            if self.mastery[target_point] >= 0.35:
                return self.case.responses["correct"]
        if average < 0.46:
            # A low baseline should produce an initial confusion signal, not
            # an endless identical loop. Once an aligned action has been
            # observed, the method-blind simulator exposes partial evidence so
            # the agent can test whether the learner is improving. This keeps
            # the evaluation sensitive to diagnosis/scaffold/correction while
            # preserving the same hidden state and noise for all methods.
            if aligned and self.turn_count > 1:
                return self.case.responses["partial"]
            return self.case.responses["confused"]
        if average < 0.73:
            return self.case.responses["partial"]
        return self.case.responses["correct"]


class EvaluationRunner:
    def __init__(self, library: SkillLibrary | None = None, seed: int | None = None):
        self.library = library or SkillLibrary()
        self.settings = get_agent_settings()
        self.seed = int(seed if seed is not None else self.settings.get("evaluation_seed", 20260809))
        random.seed(self.seed)

    @staticmethod
    def _case_data_manifest(source: Path | None = None) -> dict[str, str]:
        """Record immutable case-file fingerprints alongside every report."""
        case_source = source or get_path("cases")
        files = [case_source]
        held_out = case_source.with_name(f"{case_source.stem}_heldout{case_source.suffix}")
        if held_out.exists():
            files.append(held_out)
        return {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
            if path.exists()
        }

    def run(self, cases: list[EvaluationCase] | None = None, mode: str = "quick") -> EvaluationReport:
        cases = cases or load_cases()
        max_rounds = int(self.settings.get("max_rounds", 8))
        results: list[MethodCaseResult] = []
        profiles = ["balanced"] if mode == "quick" else list(SIMULATOR_PROFILES)
        seeds = [self.seed] if mode == "quick" else [self.seed + offset for offset in range(5)]
        for case in cases:
            for profile in profiles:
                for simulation_seed in seeds:
                    for method in METHODS:
                        results.append(self._run_case(case, method, profile, simulation_seed))
        summary = self._summarize(results)
        paired_comparisons = self._paired_comparisons(results)
        statistical_tests = self._statistical_tests(results)
        stratified_summary = self._stratify(results, cases)
        adaptive = [item for item in results if item.method == METHODS[0]]
        successful = max(adaptive, key=lambda item: (item.normalized_gain, item.behavior_quality))
        failure = min(
            adaptive,
            key=lambda item: (item.success, item.transfer_accuracy, item.normalized_gain, item.misconception_f1),
        )
        report = EvaluationReport(
            seed=self.seed,
            methods=METHODS,
            evaluation_protocol={
                "fairness": [
                    "同一案例的三种方法共享教学目标、学生画像、初始真实状态、噪声序列和回答模板",
                    f"三种方法均运行最多 {max_rounds} 个有效教学轮次，并至少获得一次验证机会",
                    "三种方法使用同一个隐藏迁移通过阈值；模拟器不读取方法名称、可接受 Skill 或预期切换标签",
                    "所有方法共享同一重复动作边际收益递减函数，避免重复同一种讲法获得无限线性增益",
                    "快速/完整仿真模式固定随机种子且不调用外部 API；它只验证策略闭环，不代替真实模型稳定性",
                    "极低起点学生首次迁移尝试可能失败，用于检验 Agent 是否会遵守终止条件",
                ],
                "method_definitions": METHOD_DESCRIPTIONS,
                "success_rule": "仿真学生的逐知识点题目级代理分与迁移验证均达到协议阈值；不是现实能力概率",
                "caveat": "学生模拟器用于可复现的策略对照，不替代真实学习者实验。",
                "evidence_grade": "strategy_simulation",
                "mode": mode,
                "evidence_levels": ["策略仿真", "真实模型稳定性（需冻结输出或在线运行）", "人工盲评（当前未完成）"],
                "case_data_versions": sorted({case.data_version for case in cases}),
                "case_data_sha256": self._case_data_manifest(),
                "simulation_parameters": {
                    "profile_parameters": SIMULATOR_PROFILES,
                    "posttest_item_thresholds": {"partial": 0.42, "correct": 0.62, "mastered": 0.78},
                    "success_is_not_guaranteed": True,
                },
                "posttest_definition": "逐知识点阈值题目级规则代理；没有把隐藏 mastery 直接乘以 100 当作后测",
                "shared_checkpoints": [4, 8, "自然终止"],
                "design": f"{len(cases)} cases × {len(profiles)} simulator profiles × {len(seeds)} shared seeds × 3 methods",
                "uncertainty": "先在每个案例内汇总画像与随机种子，再以案例为独立配对单位进行 5000 次 cluster bootstrap/符号置换，报告 95% CI、配对 Hedges g 与 Holm 校正。",
                "behavior_rubric": [
                    "知识正确性", "清晰度", "针对性", "促进思考",
                    "不直接给答案", "上下文连贯", "教师语气", "可执行性",
                ],
            },
            case_results=results,
            summary=summary,
            paired_comparisons=paired_comparisons,
            stratified_summary=stratified_summary,
            successful_case={
                "case_id": successful.case_id,
                "reason": (
                    "该案例达到协议定义的完整成功条件。"
                    if successful.success
                    else "当前冻结模拟参数下没有案例同时达到掌握与迁移成功条件；该案例仅作为自适应组最高表现样本。"
                ),
                "metrics": successful.model_dump(),
            },
            failure_case={
                "case_id": failure.case_id,
                "reason": (
                    "该案例在轮次预算内未同时满足掌握阈值与迁移验证，按终止规则判为未成功。"
                    if not failure.success
                    else "该案例虽达成目标，但学习增益或误解识别相对最低，提示触发条件仍需细化。"
                ),
                "metrics": failure.model_dump(),
            },
            statistical_tests=statistical_tests,
        )
        self.save(report)
        return report

    def _run_case(self, case: EvaluationCase, method: str, simulator_profile: str = "balanced", simulation_seed: int = 0) -> MethodCaseResult:
        mastery_threshold = float(self.settings.get("mastery_threshold", 0.8))
        student = DeterministicStudent(case, simulator_profile, simulation_seed)
        selected: list[str] = []
        action_types: list[str] = []
        student_contexts: list[str] = []
        teacher_messages: list[str] = []
        predicted_state = StudentState(mastery=deepcopy(case.initial_mastery), next_focus=case.expected_focus)
        observed_focuses = [predicted_state.next_focus]
        state_precision_scores: list[float] = []
        state_recall_scores: list[float] = []
        state_f1_scores: list[float] = []
        mastery_errors: list[float] = []
        success = False
        declared_status = "unable"
        previous_response = "尚未回答"
        checkpoint_gains: dict[int, float] = {}
        latency_samples: list[float] = []
        call_samples: list[float] = []
        fallback_count = 0
        contract_checks: list[float] = []
        continuity_checks: list[float] = []
        option_checks: list[float] = []
        evidence_mapping_checks: list[float] = []
        concern_handling_checks: list[float] = []

        def item_scores() -> dict[str, float]:
            """Question-level deterministic proxy, never presented as real testing."""
            return {
                point: round(
                    100.0 if score >= 0.78 else 75.0 if score >= 0.62 else 50.0 if score >= 0.42 else 25.0,
                    1,
                )
                for point, score in student.mastery.items()
            }

        def current_gain() -> float:
            posttest = mean(item_scores().values())
            return max(-1.0, min(1.0, (posttest - case.pretest_score) / max(1.0, 100 - case.pretest_score)))

        def record_checkpoint() -> None:
            if student.turn_count in {4, 8}:
                checkpoint_gains[student.turn_count] = current_gain()

        def record_state_snapshot(state: StudentState) -> None:
            precision, recall, f1 = _prf(
                [item.label for item in state.misconceptions],
                list(student.misconceptions),
            )
            state_precision_scores.append(precision)
            state_recall_scores.append(recall)
            state_f1_scores.append(f1)
            mastery_errors.append(
                mean(abs(state.mastery.get(point, 0.0) - truth) for point, truth in student.mastery.items())
            )

        if method == METHODS[0]:
            offline_llm = OpenAICompatibleClient(settings={"api_key": "", "model": "offline"})
            agent = HybridTeachingAgent(library=self.library, llm=offline_llm, store=_NoOpSessionStore())
            session = agent.start_session(case.goal, case.profile, predicted_state)
            for _ in range(int(self.settings.get("max_rounds", 8))):
                turn = session.turns[-1]
                selected.append(turn.selected_skill_id)
                action_types.append(turn.action_type)
                student_contexts.append(previous_response)
                teacher_messages.append(turn.teacher_message)
                has_concern = bool(HybridTeachingAgent._student_concern(previous_response))
                concern_handling_checks.append(
                    float(not has_concern or bool(turn.generation_audit.get("concern_addressed")))
                )
                if turn.teacher_review is not None:
                    contract_checks.append(float(turn.teacher_review.valid))
                else:
                    # Offline strategy simulation intentionally has no LLM
                    # review call, but its deterministic fallback still has a
                    # persisted micro-step contract. Score that contract from
                    # observable structure instead of treating “no reviewer”
                    # as a failed teacher response.
                    step = turn.micro_step
                    contract_checks.append(
                        float(
                            bool(
                                step
                                and step.focus.strip()
                                and step.context.strip()
                                and step.requested_target.strip()
                                and turn.teacher_message.count("？") + turn.teacher_message.count("?") == 1
                            )
                        )
                    )
                continuity_checks.append(float(bool(turn.teacher_review is None or turn.teacher_review.same_context)))
                option_checks.append(
                    float(turn.micro_step.response_mode != "single_choice" or bool(turn.micro_step.options))
                    if turn.micro_step
                    else 1.0
                )
                fallback_count += int(bool(turn.fallback_reason or turn.generation_audit.get("fallback_reason")))
                for trace in turn.llm_trace:
                    latency_samples.append(float(trace.latency_ms))
                    call_samples.append(float(trace.attempts))
                evidence_before = len(session.state.evidence)
                aligned = self._is_instruction_aligned(turn.action_type, student)
                response = student.respond(turn.action_type, method, aligned)
                record_checkpoint()
                previous_response = response
                session = agent.handle_student_message(session, response)
                evidence_mapping_checks.extend(
                    float(item.knowledge_point in case.goal.knowledge_points)
                    for item in session.state.evidence[evidence_before:]
                )
                observed_focuses.append(session.state.next_focus)
                record_state_snapshot(session.state)
                if session.status != "active":
                    break
            predicted_state = session.state
            declared_status = str(session.status)
            success = student.transfer_passed and mean(student.mastery.values()) >= mastery_threshold
        else:
            tracker = StateTracker(OpenAICompatibleClient(settings={"api_key": "", "model": "offline"}))
            fixed_candidates, _ = self.library.candidates(
                case.goal,
                predicted_state,
                profile=case.profile,
                include_generic=False,
                limit=1,
            )
            if not fixed_candidates:
                raise ValueError(f"评估案例 {case.case_id} 没有满足硬约束的固定学科 Skill")
            subject = fixed_candidates[0]
            rounds = int(self.settings.get("max_rounds", 8))
            for index in range(rounds):
                is_verification_round = index == rounds - 1
                if method == METHODS[1]:
                    skill_id = subject.skill_id
                    action = "fixed_verification" if is_verification_round else "subject_instruction"
                    message = (
                        f"仍按照 {subject.name}，请用一个新情境验证你能否独立应用。"
                        if is_verification_round
                        else f"继续按照 {subject.name} 的固定步骤讲解，并请学生回答当前问题。"
                    )
                    aligned = self._is_instruction_aligned(action, student)
                else:
                    skill_id = "general_tutor_without_skill"
                    action = "generic_verification" if is_verification_round else "generic"
                    message = (
                        "请在一个新情境中尝试应用这个知识点，并说明理由。"
                        if is_verification_round
                        else "我再解释一遍这个知识点。请尝试回答。"
                    )
                    aligned = self._is_instruction_aligned(action, student)
                selected.append(skill_id)
                action_types.append(action)
                student_contexts.append(previous_response)
                teacher_messages.append(message)
                concern_handling_checks.append(float(not HybridTeachingAgent._student_concern(previous_response)))
                response = student.respond(action, method, aligned)
                record_checkpoint()
                previous_response = response
                predicted_state = tracker.update(
                    case.goal, case.profile, predicted_state, response, action
                )
                observed_focuses.append(predicted_state.next_focus)
                record_state_snapshot(predicted_state)
            declared_status = (
                "success"
                if predicted_state.mastery
                and all(value >= mastery_threshold for value in predicted_state.mastery.values())
                and predicted_state.transfer_verified
                else "unable"
            )
            success = student.transfer_passed and mean(student.mastery.values()) >= mastery_threshold

        precision = mean(state_precision_scores) if state_precision_scores else 0.0
        recall = mean(state_recall_scores) if state_recall_scores else 0.0
        f1 = mean(state_f1_scores) if state_f1_scores else 0.0
        mastery_mae = mean(mastery_errors) if mastery_errors else 0.0
        focus_accuracy = float(
            any(
                token and token in focus
                for focus in observed_focuses
                for token in (case.expected_focus, case.expected_focus[:2])
            )
        )
        valid_selections = sum(
            skill_id in case.acceptable_skills or action in case.expected_switch_types
            for skill_id, action in zip(selected, action_types, strict=True)
        )
        selection_accuracy = valid_selections / len(selected) if selected else 0.0
        observed_switches = set(action_types) & set(case.expected_switch_types)
        switch_accuracy = len(observed_switches) / len(set(case.expected_switch_types)) if case.expected_switch_types else 1.0
        expected_success = mean(student.mastery.values()) >= mastery_threshold and student.transfer_passed
        termination_accuracy = float((declared_status == "success") == expected_success)
        violations = sum(
            any(token in message for token in ("答案是", "直接答案", "照抄这段", "完整代码如下"))
            for message in teacher_messages
        )
        questions = sum("？" in message or "?" in message or "请" in message for message in teacher_messages)
        message_count = max(1, len(teacher_messages))
        non_revealing = 1 - violations / message_count
        question_rate = questions / message_count
        text = " ".join(teacher_messages)
        supportive = float(any(token in text for token in ("请", "我们", "先", "试", "依据")))
        contextual = float(any(case.goal.topic[:2] in message for message in teacher_messages))
        complete = sum(message.rstrip().endswith(("。", "？", "?", "！", "：")) for message in teacher_messages) / message_count
        # The automatic score is deliberately a proxy.  For the adaptive
        # path, use the independent structured output gate; do not treat a
        # handful of subject-specific phrases as a correctness proof.
        reviewed_fact_checks = [
            float(turn.teacher_review.fact_consistent)
            for turn in session.turns
            if turn.teacher_review is not None
        ] if method == METHODS[0] else []
        correctness_proxy = mean(reviewed_fact_checks) if reviewed_fact_checks else 0.0
        behavior_dimensions = {
            "knowledge_correctness_proxy": correctness_proxy,
            "clarity_proxy": complete,
            "targeting_proxy": contextual,
            "promotes_thinking": question_rate,
            "answer_non_revealing": non_revealing,
            "actionability": question_rate,
            "coherence": max(0.0, 1.0 - max(0, len(set(action_types)) - 4) * 0.1),
            "tutor_tone": supportive,
            "student_question_handling": mean(concern_handling_checks) if concern_handling_checks else 0.0,
        }
        behavior_quality = mean(behavior_dimensions.values())
        posttest_item_scores = item_scores()
        posttest = round(mean(posttest_item_scores.values()), 1)
        gain = (posttest - case.pretest_score) / max(1.0, 100 - case.pretest_score)
        learning_efficiency = gain / max(1, len(selected))
        p95_latency = (
            sorted(latency_samples)[min(len(latency_samples) - 1, max(0, math.ceil(len(latency_samples) * 0.95) - 1))]
            if latency_samples
            else 0.0
        )
        return MethodCaseResult(
            case_id=case.case_id,
            method=method,
            rounds=len(selected),
            misconception_precision=round(precision, 3),
            misconception_recall=round(recall, 3),
            misconception_f1=round(f1, 3),
            mastery_mae=round(mastery_mae, 3),
            focus_accuracy=round(focus_accuracy, 3),
            skill_selection_accuracy=round(selection_accuracy, 3),
            switch_accuracy=round(switch_accuracy, 3),
            termination_accuracy=round(termination_accuracy, 3),
            behavior_quality=round(behavior_quality, 3),
            behavior_dimensions={key: round(value, 3) for key, value in behavior_dimensions.items()},
            direct_answer_violation_rate=round(violations / max(1, len(teacher_messages)), 3),
            pretest_score=case.pretest_score,
            posttest_score=posttest,
            normalized_gain=round(max(-1.0, min(1.0, gain)), 3),
            learning_efficiency=round(max(-1.0, min(1.0, learning_efficiency)), 3),
            transfer_accuracy=float(student.transfer_passed),
            success=success,
            declared_status=declared_status,
            selected_skills=selected,
            action_types=action_types,
            student_contexts=student_contexts,
            teacher_messages=teacher_messages,
            notes=[
                "这是策略仿真，不是完整教学效果实验；学习增益函数不读取方法名称",
                "三种方法共享状态诊断器、初始状态、轮次预算与验证机会",
                "前后测为逐知识点题目级规则代理分；正式结论需结合真实题目和盲法人工标注",
                "隐藏学生状态按知识点逐点更新，未使用可接受 Skill 或预期切换标签",
            ],
            split=case.split,
            simulator_profile=simulator_profile,
            simulation_seed=simulation_seed,
            checkpoint_gain_4=round(checkpoint_gains.get(4, current_gain()), 3),
            checkpoint_gain_8=round(checkpoint_gains.get(8, current_gain()), 3),
            posttest_items=posttest_item_scores,
            single_step_contract_rate=round(mean(contract_checks) if contract_checks else 0.0, 3),
            context_continuity_rate=round(mean(continuity_checks) if continuity_checks else 0.0, 3),
            student_question_handling_rate=round(mean(concern_handling_checks) if concern_handling_checks else 0.0, 3),
            option_validity_rate=round(mean(option_checks) if option_checks else 1.0, 3),
            llm_fallback_rate=round(fallback_count / max(1, len(selected)), 3),
            mean_latency_ms=round(mean(latency_samples) if latency_samples else 0.0, 1),
            p95_latency_ms=round(p95_latency, 1),
            mean_llm_calls=round(mean(call_samples) if call_samples else 0.0, 2),
            evidence_mapping_accuracy=round(mean(evidence_mapping_checks) if evidence_mapping_checks else 1.0, 3),
            unreasonable_switch_rate=0.0,
        )

    def _is_aligned(
        self,
        skill_id: str,
        action_type: str,
        case: EvaluationCase,
        student: DeterministicStudent,
    ) -> bool:
        if skill_id in case.acceptable_skills:
            return True
        if action_type == "correction" and student.misconceptions:
            return True
        if action_type == "transfer" and mean(student.mastery.values()) >= 0.68:
            return True
        return action_type in case.expected_switch_types

    @staticmethod
    def _is_instruction_aligned(action_type: str, student: DeterministicStudent) -> bool:
        """Method-blind simulator policy; it never reads acceptable skill labels."""
        if student.misconceptions:
            return action_type in {"diagnostic", "scaffold", "correction", "subject_instruction", "fixed_verification"}
        if mean(student.mastery.values()) >= 0.68:
            return action_type in {"transfer", "fixed_verification", "generic_verification"}
        return action_type in {"scaffold", "subject_instruction", "generic"}

    def _statistical_tests(self, results: list[MethodCaseResult]) -> list[dict[str, Any]]:
        """Case-clustered paired sign permutation with Holm correction and McNemar counts."""
        grouped: dict[str, dict[str, list[MethodCaseResult]]] = defaultdict(lambda: defaultdict(list))
        for result in results:
            grouped[result.method][result.case_id].append(result)
        rows: list[dict[str, Any]] = []
        for baseline in METHODS[1:]:
            keys = sorted(set(grouped[METHODS[0]]) & set(grouped[baseline]))
            adaptive_gain = {
                key: mean(item.normalized_gain for item in grouped[METHODS[0]][key]) for key in keys
            }
            baseline_gain = {
                key: mean(item.normalized_gain for item in grouped[baseline][key]) for key in keys
            }
            diffs = [adaptive_gain[key] - baseline_gain[key] for key in keys]
            observed = abs(mean(diffs)) if diffs else 0.0
            rng = random.Random(self.seed + len(rows) + 900)
            samples = 5000
            extreme = sum(
                abs(mean(value * (1 if rng.random() > 0.5 else -1) for value in diffs)) >= observed
                for _ in range(samples)
            ) if diffs else samples
            p_value = (extreme + 1) / (samples + 1)
            adaptive_transfer = {
                key: mean(item.transfer_accuracy for item in grouped[METHODS[0]][key]) >= 0.5 for key in keys
            }
            baseline_transfer = {
                key: mean(item.transfer_accuracy for item in grouped[baseline][key]) >= 0.5 for key in keys
            }
            b = sum(adaptive_transfer[key] and not baseline_transfer[key] for key in keys)
            c = sum(baseline_transfer[key] and not adaptive_transfer[key] for key in keys)
            mcnemar_p = 1.0 if b + c == 0 else min(1.0, 2 * sum(math.comb(b + c, i) for i in range(min(b, c) + 1)) / 2 ** (b + c))
            rows.append({"baseline": baseline, "metric": "normalized_gain", "permutation_p": p_value,
                         "mcnemar_b": b, "mcnemar_c": c, "mcnemar_exact_p": mcnemar_p})
        ordered = sorted(enumerate(rows), key=lambda item: item[1]["permutation_p"])
        running = 0.0
        for rank, (index, row) in enumerate(ordered):
            adjusted = min(1.0, float(row["permutation_p"]) * (len(rows) - rank))
            running = max(running, adjusted)
            rows[index]["holm_adjusted_p"] = running
        return rows

    @staticmethod
    def _bootstrap_mean_ci(values: list[float], seed: int, samples: int = 5000) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        rng = random.Random(seed)
        estimates = sorted(
            mean(rng.choice(values) for _ in values)
            for _ in range(samples)
        )
        return estimates[int(0.025 * samples)], estimates[min(samples - 1, int(0.975 * samples))]

    @staticmethod
    def _wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
        if total == 0:
            return 0.0, 0.0
        rate = successes / total
        denominator = 1 + z * z / total
        center = (rate + z * z / (2 * total)) / denominator
        margin = z * ((rate * (1 - rate) / total + z * z / (4 * total * total)) ** 0.5) / denominator
        return max(0.0, center - margin), min(1.0, center + margin)

    def _summarize(self, results: list[MethodCaseResult]) -> list[dict[str, Any]]:
        grouped: dict[str, list[MethodCaseResult]] = defaultdict(list)
        for item in results:
            grouped[item.method].append(item)
        summary = []
        for method_index, (method, items) in enumerate(grouped.items()):
            by_case: dict[str, list[MethodCaseResult]] = defaultdict(list)
            for item in items:
                by_case[item.case_id].append(item)
            case_gains = [mean(result.normalized_gain for result in case_items) for case_items in by_case.values()]
            case_success = [mean(float(result.success) for result in case_items) for case_items in by_case.values()]
            gain_low, gain_high = self._bootstrap_mean_ci(case_gains, self.seed + method_index)
            success_low, success_high = self._bootstrap_mean_ci(case_success, self.seed + 50 + method_index)
            summary.append(
                {
                    "method": method,
                    "cases": len(by_case),
                    "simulation_runs": len(items),
                    "state_f1": round(mean(item.misconception_f1 for item in items), 3),
                    "mastery_mae": round(mean(item.mastery_mae for item in items), 3),
                    "decision_quality": round(
                        mean((item.skill_selection_accuracy + item.switch_accuracy + item.termination_accuracy) / 3 for item in items), 3
                    ),
                    "behavior_quality": round(mean(item.behavior_quality for item in items), 3),
                    "normalized_gain": round(mean(item.normalized_gain for item in items), 3),
                    "gain_ci_low": round(gain_low, 3),
                    "gain_ci_high": round(gain_high, 3),
                    "transfer_accuracy": round(mean(item.transfer_accuracy for item in items), 3),
                    "learning_efficiency": round(mean(item.learning_efficiency for item in items), 3),
                    "mean_rounds": round(mean(item.rounds for item in items), 2),
                    "success_rate": round(mean(float(item.success) for item in items), 3),
                    "success_ci_low": round(success_low, 3),
                    "success_ci_high": round(success_high, 3),
                    "single_step_contract_rate": round(mean(item.single_step_contract_rate for item in items), 3),
                    "context_continuity_rate": round(mean(item.context_continuity_rate for item in items), 3),
                    "student_question_handling_rate": round(
                        mean(item.student_question_handling_rate for item in items), 3
                    ),
                    "option_validity_rate": round(mean(item.option_validity_rate for item in items), 3),
                    "llm_fallback_rate": round(mean(item.llm_fallback_rate for item in items), 3),
                    "mean_latency_ms": round(mean(item.mean_latency_ms for item in items), 1),
                    "p95_latency_ms": round(mean(item.p95_latency_ms for item in items), 1),
                    "mean_llm_calls": round(mean(item.mean_llm_calls for item in items), 2),
                    "evidence_mapping_accuracy": round(mean(item.evidence_mapping_accuracy for item in items), 3),
                }
            )
        return summary

    def _paired_comparisons(self, results: list[MethodCaseResult]) -> list[dict[str, Any]]:
        by_method: dict[str, dict[str, list[MethodCaseResult]]] = {
            method: defaultdict(list) for method in METHODS
        }
        for item in results:
            by_method[item.method][item.case_id].append(item)
        comparisons: list[dict[str, Any]] = []
        metric_getters = {
            "normalized_gain": lambda item: item.normalized_gain,
            "learning_efficiency": lambda item: item.learning_efficiency,
            "decision_quality": lambda item: (
                item.skill_selection_accuracy + item.switch_accuracy + item.termination_accuracy
            ) / 3,
            "transfer_accuracy": lambda item: item.transfer_accuracy,
        }
        for baseline_index, baseline in enumerate(METHODS[1:]):
            common = sorted(set(by_method[METHODS[0]]) & set(by_method[baseline]))
            for metric_index, (metric, getter) in enumerate(metric_getters.items()):
                adaptive_values = [
                    mean(getter(item) for item in by_method[METHODS[0]][case_id]) for case_id in common
                ]
                baseline_values = [
                    mean(getter(item) for item in by_method[baseline][case_id]) for case_id in common
                ]
                differences = [left - right for left, right in zip(adaptive_values, baseline_values, strict=True)]
                low, high = self._bootstrap_mean_ci(
                    differences,
                    self.seed + 100 + baseline_index * 10 + metric_index,
                )
                if len(differences) > 1 and stdev(differences) > 0:
                    correction = 1 - 3 / (4 * len(differences) - 5)
                    effect_size = correction * mean(differences) / stdev(differences)
                else:
                    effect_size = None
                comparisons.append(
                    {
                        "baseline": baseline,
                        "metric": metric,
                        "n_pairs": len(common),
                        "adaptive_mean": round(mean(adaptive_values), 3),
                        "baseline_mean": round(mean(baseline_values), 3),
                        "mean_difference": round(mean(differences), 3),
                        "ci_low": round(low, 3),
                        "ci_high": round(high, 3),
                        "hedges_g_paired": round(effect_size, 3) if effect_size is not None else None,
                        "win_rate": round(
                            mean(1.0 if value > 0 else 0.5 if value == 0 else 0.0 for value in differences),
                            3,
                        ),
                    }
                )
        return comparisons

    @staticmethod
    def _stratify(results: list[MethodCaseResult], cases: list[EvaluationCase]) -> list[dict[str, Any]]:
        course_by_case = {case.case_id: case.goal.course for case in cases}
        split_by_case = {case.case_id: case.split for case in cases}
        grouped: dict[tuple[str, str, str], list[MethodCaseResult]] = defaultdict(list)
        for item in results:
            grouped[(course_by_case[item.case_id], split_by_case[item.case_id], item.method)].append(item)
        return [
            {
                "course": course,
                "split": split,
                "method": method,
                "cases": len(items),
                "normalized_gain": round(mean(item.normalized_gain for item in items), 3),
                "decision_quality": round(
                    mean(
                        (item.skill_selection_accuracy + item.switch_accuracy + item.termination_accuracy) / 3
                        for item in items
                    ),
                    3,
                ),
                "transfer_accuracy": round(mean(item.transfer_accuracy for item in items), 3),
            }
            for (course, split, method), items in grouped.items()
        ]

    @staticmethod
    def human_annotation_csv(report: EvaluationReport, reveal_key: bool = False) -> str:
        method_codes = list(report.methods)
        random.Random(report.seed + 701).shuffle(method_codes)
        code_by_method = {method: f"系统{chr(65 + index)}" for index, method in enumerate(method_codes)}
        output = io.StringIO()
        if reveal_key:
            writer = csv.DictWriter(output, fieldnames=["盲法系统编号", "真实方法"])
            writer.writeheader()
            for method in report.methods:
                writer.writerow({"盲法系统编号": code_by_method[method], "真实方法": method})
            return output.getvalue()

        fields = [
            "样本ID", "盲法系统编号", "案例", "轮次", "学生上下文", "教师回复",
            "知识正确性", "清晰度", "针对性", "促进思考", "不直接给答案",
            "上下文连贯", "教师语气", "可执行性", "评审员", "备注",
        ]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        # Keep human comparison paired: use the same case sample for every
        # method and export at most 12 replies per method. One reply per case is
        # selected first, avoiding hundreds of correlated turns being presented
        # as independent human-rating evidence.
        available_case_ids = sorted({item.case_id for item in report.case_results})
        case_rng = random.Random(report.seed + 702)
        case_rng.shuffle(available_case_ids)
        selected_case_ids = available_case_ids[:12]
        rows = []
        row_rng = random.Random(report.seed + 703)
        grouped: dict[tuple[str, str], list[dict[str, str | int]]] = defaultdict(list)
        for item in report.case_results:
            code = code_by_method[item.method]
            for index, response in enumerate(item.teacher_messages):
                grouped[(item.method, item.case_id)].append(
                    {
                        "样本ID": f"{item.case_id}-{index + 1}-{code}",
                        "盲法系统编号": code,
                        "案例": item.case_id,
                        "轮次": index + 1,
                        "学生上下文": item.student_contexts[index] if index < len(item.student_contexts) else "",
                        "教师回复": response,
                    }
                )
        for method in report.methods:
            method_rows: list[dict[str, str | int]] = []
            for case_id in selected_case_ids:
                candidates = grouped.get((method, case_id), [])
                if candidates:
                    method_rows.append(candidates[row_rng.randrange(len(candidates))])
            # Quick/custom reports can contain fewer than 12 distinct cases.
            # Fill from unused replies while keeping sample IDs unique.
            if len(method_rows) < 12:
                used = {str(row["样本ID"]) for row in method_rows}
                extras = [
                    row
                    for (row_method, _), candidates in grouped.items()
                    if row_method == method
                    for row in candidates
                    if str(row["样本ID"]) not in used
                ]
                row_rng.shuffle(extras)
                method_rows.extend(extras[: 12 - len(method_rows)])
            rows.extend(method_rows[:12])
        row_rng.shuffle(rows)
        writer.writerows(rows)
        return output.getvalue()

    def save(self, report: EvaluationReport) -> dict[str, Path]:
        override = os.getenv("TEACHING_AGENT_EVALUATION_DIR", "").strip()
        output = Path(override) if override else get_path("evaluations")
        output.mkdir(parents=True, exist_ok=True)
        stem = report.generated_at.replace(":", "").replace("+", "_").replace("-", "") + f"_{os.urandom(3).hex()}"
        json_path = output / f"evaluation_{stem}.json"
        csv_path = output / f"evaluation_{stem}.csv"
        md_path = output / f"evaluation_{stem}.md"
        annotation_path = output / f"annotation_blind_{stem}.csv"
        annotation_key_path = output / f"annotation_key_{stem}.csv"
        _atomic_json(json_path, report.model_dump(mode="json"))
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report.summary[0].keys()))
            writer.writeheader()
            writer.writerows(report.summary)
        md_path.write_text(self.to_markdown(report), encoding="utf-8")
        annotation_path.write_text(self.human_annotation_csv(report), encoding="utf-8-sig")
        annotation_key_path.write_text(
            self.human_annotation_csv(report, reveal_key=True), encoding="utf-8-sig"
        )
        EvaluationStore(output).register_report(json_path.name, report)
        return {
            "json": json_path,
            "csv": csv_path,
            "markdown": md_path,
            "annotation": annotation_path,
            "annotation_key": annotation_key_path,
        }

    @staticmethod
    def to_markdown(report: EvaluationReport) -> str:
        best_label = (
            "成功案例"
            if report.successful_case.get("metrics", {}).get("success")
            else "最高表现案例（本轮无完整成功案例）"
        )
        lines = [
            "# 教师 Agent 自动评估报告",
            "",
            f"生成时间：{report.generated_at}　固定随机种子：{report.seed}",
            "",
            "## 方法对比",
            "",
            "| 方法 | 状态F1 | 掌握误差↓ | 决策质量 | 行为代理（描述） | 标准化增益 | 迁移正确率 | 成功率 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for item in report.summary:
            lines.append(
                f"| {item['method']} | {item['state_f1']:.3f} | {item['mastery_mae']:.3f} | "
                f"{item['decision_quality']:.3f} | {item['behavior_quality']:.3f} | "
                f"{item['normalized_gain']:.3f} | {item['transfer_accuracy']:.3f} | {item['success_rate']:.3f} |"
            )
        lines.extend(
            [
                "",
                "## 实验协议与公平性",
                "",
                *[f"- {item}" for item in report.evaluation_protocol.get("fairness", [])],
                f"- 成功判据：{report.evaluation_protocol.get('success_rule', '')}",
                f"- 局限：{report.evaluation_protocol.get('caveat', '')}",
                f"- 不确定性：{report.evaluation_protocol.get('uncertainty', '')}",
                "",
                "## Agent 质量门（自动 proxy）",
                "",
                "| 方法 | 单步契约 | 上下文连续 | 学生疑问承接 | 选项有效 | 证据映射 | 平均 LLM 调用 |",
                "|---|---:|---:|---:|---:|---:|---:|",
                *[
                    f"| {item['method']} | {item.get('single_step_contract_rate', 0):.3f} | "
                    f"{item.get('context_continuity_rate', 0):.3f} | "
                    f"{item.get('student_question_handling_rate', 0):.3f} | "
                    f"{item.get('option_validity_rate', 0):.3f} | "
                    f"{item.get('evidence_mapping_accuracy', 0):.3f} | {item.get('mean_llm_calls', 0):.2f} |"
                    for item in report.summary
                ],
                "",
                "## 配对比较（自适应 − 基线）",
                "",
                "| 基线 | 指标 | 均值差 | 95% CI | 配对 Hedges g | 胜率 |",
                "|---|---|---:|---:|---:|---:|",
                *[
                    f"| {item['baseline']} | {item['metric']} | {item['mean_difference']:.3f} | "
                    f"[{item['ci_low']:.3f}, {item['ci_high']:.3f}] | "
                    f"{item['hedges_g_paired'] if item['hedges_g_paired'] is not None else 'NA'} | "
                    f"{item['win_rate']:.3f} |"
                    for item in report.paired_comparisons
                ],
                "",
                "## 典型案例",
                "",
                f"- {best_label}：`{report.successful_case['case_id']}`。{report.successful_case['reason']}",
                f"- 失败案例：`{report.failure_case['case_id']}`。{report.failure_case['reason']}",
                f"- 数据版本：{', '.join(report.evaluation_protocol.get('case_data_versions', []))}",
                f"- 数据 SHA-256：{report.evaluation_protocol.get('case_data_sha256', {})}",
                "",
                "## 说明",
                "",
                f"本报告包含 {len(set(item.case_id for item in report.case_results))} 个案例，覆盖开发集与完全留出集。"
                "三种方法共享初始状态、模拟器画像和随机种子；自动行为分仅是 proxy。",
                "行为代理存在明显天花板效应，仅用于发现格式、泄题和缺问句等回归异常；"
                "不对该代理分计算方法优劣的显著性或效应量。",
                f"人工盲评状态：{report.human_evaluation_status}",
                "结论边界：结果只支持模拟环境中的决策有效性与稳健性，不构成真实课堂因果效果证据。",
            ]
        )
        return "\n".join(lines)
