from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class SessionStatus(str, Enum):
    ACTIVE = "active"
    SUCCESS = "success"
    UNABLE = "unable"


ResponseMode = Literal["open", "single_choice", "fill_blank", "numeric"]
ResponsePreference = Literal["auto", "open", "single_choice", "fill_blank", "numeric"]
TeachingPhase = Literal["diagnosis", "instruction", "practice", "repair", "transfer", "completed", "paused"]
RouteStepStatus = Literal["pending", "active", "completed"]
RouteStepKind = Literal["knowledge", "transfer"]


class TeachingGoal(BaseModel):
    course: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    knowledge_points: list[str] = Field(min_length=1)
    success_criteria: list[str] = Field(default_factory=list)


class StudentProfile(BaseModel):
    name: str = "学生"
    level: Literal["基础薄弱", "中等", "较好"] = "中等"
    prior_knowledge: list[str] = Field(default_factory=list)
    learning_preferences: list[str] = Field(default_factory=list)
    response_preference: ResponsePreference = "auto"


class Misconception(BaseModel):
    label: str
    evidence: str = ""
    count: int = Field(default=1, ge=1)


class StateEvidence(BaseModel):
    """A traceable observation supporting one state update."""

    student_quote: str
    knowledge_point: str
    signal_type: Literal["positive", "partial", "negative", "empty", "transfer"]
    evidence_level: Literal["none", "partial", "correct", "explained", "transfer"] = "partial"
    round_index: int = Field(default=0, ge=0)
    reason: str = ""


class KnowledgeState(BaseModel):
    mastery: float = Field(default=0.3, ge=0.0, le=1.0)
    confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    last_evidence_level: Literal["none", "partial", "correct", "explained", "transfer"] = "none"
    evidence: list[StateEvidence] = Field(default_factory=list)
    updated_at: str = Field(default_factory=now_iso)


class MisconceptionState(Misconception):
    knowledge_point: str = ""
    consecutive_count: int = Field(default=1, ge=0)
    improving: bool = False


class StudentState(BaseModel):
    mastery_model_version: str = "evidence-v2"
    mastery: dict[str, float] = Field(default_factory=dict)
    misconceptions: list[Misconception] = Field(default_factory=list)
    understanding_signals: list[str] = Field(default_factory=list)
    next_focus: str = "诊断当前理解"
    transfer_verified: bool = False
    no_progress_rounds: int = Field(default=0, ge=0)
    knowledge_states: dict[str, KnowledgeState] = Field(default_factory=dict)
    evidence: list[StateEvidence] = Field(default_factory=list)
    misconception_states: list[MisconceptionState] = Field(default_factory=list)
    phase: TeachingPhase = "diagnosis"

    @field_validator("mastery")
    @classmethod
    def clamp_mastery(cls, value: dict[str, float]) -> dict[str, float]:
        return {key: round(max(0.0, min(1.0, float(score))), 3) for key, score in value.items()}

    def average_mastery(self) -> float:
        return sum(self.mastery.values()) / len(self.mastery) if self.mastery else 0.0


class ResponseOption(BaseModel):
    """A learner-facing option; no answer key is stored in the domain model."""

    option_id: str = Field(min_length=1, max_length=8)
    text: str = Field(min_length=1)
    misconception_hint: str = ""


class QuestionContract(BaseModel):
    """The learner-facing contract for exactly one answerable teaching step."""

    focus: str = ""
    context: str = ""
    requested_target: str = ""
    response_mode: ResponseMode = "open"
    option_ids: list[str] = Field(default_factory=list)
    expected_signal: str = ""
    allowed_next_phase: TeachingPhase | None = None
    valid: bool = True


class SkillPlan(BaseModel):
    """Separates subject content from the pedagogical strategy used this turn."""

    content_skill_id: str | None = None
    strategy_skill_id: str | None = None
    content_skill_reason: str = ""
    strategy_reason: str = ""
    candidate_content_skill_ids: list[str] = Field(default_factory=list)
    candidate_strategy_skill_ids: list[str] = Field(default_factory=list)
    content_switch: bool = False
    strategy_switch: bool = False


class LLMCallTrace(BaseModel):
    operation: str
    model: str = ""
    attempts: int = 0
    latency_ms: int = 0
    success: bool = False
    fallback: str = ""
    error_class: str = ""


class GenerationRevision(BaseModel):
    revision_index: int = Field(default=1, ge=1)
    teacher_message: str = ""
    reason: str = ""
    created_at: str = Field(default_factory=now_iso)


class TeachingMicroStep(BaseModel):
    """A single, auditable teaching step shared by planning and generation."""

    focus: str = ""
    context: str = ""
    known_fact: str = ""
    requested_target: str = ""
    representation: str = ""
    expected_signal: str = ""
    step_index: int = Field(default=0, ge=0)
    response_mode: ResponseMode = "open"
    options: list[ResponseOption] = Field(default_factory=list)
    input_hint: str = ""


class TeachingRouteStep(BaseModel):
    """One stable curriculum target; teaching strategies may vary inside it."""

    step_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    knowledge_point: str = Field(min_length=1)
    learning_target: str = Field(min_length=1)
    evidence_requirement: Literal["correct", "explained", "transfer"] = "correct"
    kind: RouteStepKind = "knowledge"
    status: RouteStepStatus = "pending"


class TeachingRoute(BaseModel):
    """Persisted lesson route that prevents turn-by-turn topic drift."""

    steps: list[TeachingRouteStep] = Field(min_length=1)
    current_index: int = Field(default=0, ge=0)
    source: Literal["llm", "goal_fallback"] = "goal_fallback"
    created_at: str = Field(default_factory=now_iso)

    def current_step(self) -> TeachingRouteStep:
        index = min(self.current_index, len(self.steps) - 1)
        return self.steps[index]

    def completed_count(self) -> int:
        return sum(step.status == "completed" for step in self.steps)


class TeacherDraft(BaseModel):
    """Structured LLM output before it is allowed into the student-facing UI."""

    micro_step: TeachingMicroStep
    teacher_message: str = ""
    introduced_symbols: list[str] = Field(default_factory=list)
    introduced_values: list[str] = Field(default_factory=list)
    question_count: int = Field(default=0, ge=0)

    @field_validator("introduced_symbols", "introduced_values", mode="before")
    @classmethod
    def _normalize_audit_values(cls, value: Any) -> list[str]:
        """Keep audit metadata tolerant of numeric JSON values from an LLM."""
        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError("审计字段必须是数组")
        return [str(item) for item in value]


class TeacherReview(BaseModel):
    """Structured quality gate for one-step teacher output."""

    valid: bool = False
    one_step: bool = False
    one_context: bool = False
    one_question: bool = False
    fact_consistent: bool = False
    same_context: bool = True
    answer_leakage: bool = False
    response_mode_valid: bool = True
    options_valid: bool = True
    issues: list[str] = Field(default_factory=list)
    revised_message: str = ""


class AgentDecision(BaseModel):
    primary_skill_id: str
    support_skill_id: str | None = None
    selection_reason: str
    action_type: str
    teacher_message: str
    expected_signal: str = ""
    switch_reason: str = ""
    decision_mode: str = "rule_fallback"
    candidate_skill_ids: list[str] = Field(default_factory=list)
    policy_rule: str = ""
    candidate_audit: list[dict[str, Any]] = Field(default_factory=list)
    fallback_reason: str = ""
    should_stop: bool = False
    status: SessionStatus = SessionStatus.ACTIVE
    stop_reason: str = ""
    micro_step: TeachingMicroStep | None = None
    teacher_review: TeacherReview | None = None
    generation_audit: dict[str, Any] = Field(default_factory=dict)
    phase: TeachingPhase = "diagnosis"
    skill_plan: SkillPlan | None = None
    question_contract: QuestionContract | None = None
    llm_trace: list[LLMCallTrace] = Field(default_factory=list)


class ConversationTurn(BaseModel):
    round_index: int = Field(ge=1)
    student_message: str = ""
    teacher_message: str
    selected_skill_id: str
    support_skill_id: str | None = None
    selection_reason: str
    action_type: str
    state_before: StudentState
    state_after: StudentState
    switch_reason: str = ""
    decision_mode: str = "rule_fallback"
    candidate_skill_ids: list[str] = Field(default_factory=list)
    policy_rule: str = ""
    candidate_audit: list[dict[str, Any]] = Field(default_factory=list)
    fallback_reason: str = ""
    stop_decision: str = "继续教学"
    micro_step: TeachingMicroStep | None = None
    teacher_review: TeacherReview | None = None
    generation_audit: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=now_iso)
    phase: TeachingPhase = "diagnosis"
    content_skill_id: str | None = None
    strategy_skill_id: str | None = None
    skill_plan: SkillPlan | None = None
    question_contract: QuestionContract | None = None
    generation_revisions: list[GenerationRevision] = Field(default_factory=list)
    llm_trace: list[LLMCallTrace] = Field(default_factory=list)


class TeachingSession(BaseModel):
    model_config = ConfigDict(use_enum_values=True, validate_default=True)

    session_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    schema_version: int = Field(default=5, ge=1)
    display_title: str = ""
    archived_at: str | None = None
    deleted_at: str | None = None
    goal: TeachingGoal
    profile: StudentProfile
    state: StudentState
    turns: list[ConversationTurn] = Field(default_factory=list)
    status: SessionStatus = SessionStatus.ACTIVE
    termination_reason: str = ""
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)
    available_skill_ids: list[str] = Field(default_factory=list)
    skill_snapshot: dict[str, str] = Field(default_factory=dict)
    imported_history: list[dict[str, str]] = Field(default_factory=list)
    teaching_route: TeachingRoute | None = None
    # Counts answers since the latest continuation.  The full conversation
    # remains in ``turns``; this separate counter keeps the 8-turn budget from
    # immediately terminating a genuinely resumed session.
    rounds_in_current_run: int = Field(default=0, ge=0)

    def answered_rounds(self) -> int:
        """Initial teacher opening is not a student-answer round."""
        return sum(bool(turn.student_message.strip()) for turn in self.turns)


class StateAssessment(BaseModel):
    mastery_updates: dict[str, float] = Field(default_factory=dict)
    misconceptions: list[Misconception] = Field(default_factory=list)
    understanding_signals: list[str] = Field(default_factory=list)
    next_focus: str = ""
    verification_passed: bool = False
    progress: Literal["improved", "unchanged", "regressed"] = "unchanged"
    affected_points: list[str] = Field(default_factory=list)
    evidence_levels: dict[str, Literal["none", "partial", "correct", "explained", "transfer"]] = Field(
        default_factory=dict
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_reason: str = ""


class EvaluationCase(BaseModel):
    case_id: str
    title: str
    goal: TeachingGoal
    profile: StudentProfile
    initial_mastery: dict[str, float]
    true_misconceptions: list[str]
    expected_focus: str
    acceptable_skills: list[str]
    expected_switch_types: list[str]
    pretest_score: float = Field(ge=0, le=100)
    responses: dict[str, str]
    pretest_items: dict[str, float] = Field(default_factory=dict)
    transfer_items: list[str] = Field(default_factory=list)
    data_version: str = "dev-v1"
    split: Literal["development", "held_out"] = "development"
    profile_group: str = "balanced"


class MethodCaseResult(BaseModel):
    case_id: str
    method: str
    rounds: int
    misconception_precision: float
    misconception_recall: float
    misconception_f1: float
    mastery_mae: float
    focus_accuracy: float
    skill_selection_accuracy: float
    switch_accuracy: float
    termination_accuracy: float
    behavior_quality: float
    behavior_dimensions: dict[str, float] = Field(default_factory=dict)
    direct_answer_violation_rate: float
    pretest_score: float
    posttest_score: float
    normalized_gain: float
    learning_efficiency: float = 0.0
    transfer_accuracy: float
    success: bool
    declared_status: str = "unable"
    selected_skills: list[str]
    action_types: list[str] = Field(default_factory=list)
    student_contexts: list[str] = Field(default_factory=list)
    teacher_messages: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    split: str = "development"
    simulator_profile: str = "balanced"
    simulation_seed: int = 0
    checkpoint_gain_4: float = 0.0
    checkpoint_gain_8: float = 0.0
    posttest_items: dict[str, float] = Field(default_factory=dict)
    single_step_contract_rate: float = 0.0
    context_continuity_rate: float = 0.0
    student_question_handling_rate: float = 0.0
    option_validity_rate: float = 0.0
    llm_fallback_rate: float = 0.0
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    mean_llm_calls: float = 0.0
    evidence_mapping_accuracy: float = 0.0
    unreasonable_switch_rate: float = 0.0


class EvaluationReport(BaseModel):
    schema_version: int = 2
    generated_at: str = Field(default_factory=now_iso)
    seed: int
    methods: list[str]
    evaluation_protocol: dict[str, Any] = Field(default_factory=dict)
    case_results: list[MethodCaseResult]
    summary: list[dict[str, Any]]
    paired_comparisons: list[dict[str, Any]] = Field(default_factory=list)
    stratified_summary: list[dict[str, Any]] = Field(default_factory=list)
    successful_case: dict[str, Any]
    failure_case: dict[str, Any]
    human_evaluation_status: str = "待完成：不得将自动代理分解释为人工教学质量结论"
    statistical_tests: list[dict[str, Any]] = Field(default_factory=list)
