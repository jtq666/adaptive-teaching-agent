"""Internal roles coordinated by the public Teaching Agent.

The application exposes one ``HybridTeachingAgent``.  These roles separate
diagnosis, decision, response generation and quality review without creating
independent agents with competing state or outputs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

from src.models import (
    AgentDecision,
    StudentProfile,
    StudentState,
    TeacherDraft,
    TeacherReview,
    TeachingGoal,
    TeachingMicroStep,
    TeachingSession,
)
from src.skills import TeachingSkill
from src.state_tracker import StateTracker

GenerationResult = TypeVar("GenerationResult")


class StudentDiagnosisRole:
    """Produce one updated learner state from the latest learner response."""

    role_id = "student_diagnosis"

    def __init__(self, tracker: StateTracker):
        self.tracker = tracker

    def execute(
        self,
        goal: TeachingGoal,
        profile: StudentProfile,
        state: StudentState,
        student_message: str,
        current_skill_id: str,
        *,
        round_index: int,
        previous_teacher_message: str,
        active_knowledge_point: str = "",
    ) -> StudentState:
        return self.tracker.update(
            goal,
            profile,
            state,
            student_message,
            current_skill_id,
            round_index=round_index,
            previous_teacher_message=previous_teacher_message,
            active_knowledge_point=active_knowledge_point,
        )


class TeachingDecisionRole:
    """Choose one Skill plan and one next teaching action."""

    role_id = "skill_decision"

    def __init__(self, executor: Callable[[TeachingSession, str, bool], AgentDecision]):
        self.executor = executor

    def execute(
        self,
        session: TeachingSession,
        student_message: str,
        *,
        initial: bool,
    ) -> AgentDecision:
        return self.executor(session, student_message, initial)


class TeacherResponseRole(Generic[GenerationResult]):
    """Generate one student-facing response from an already selected plan."""

    role_id = "teacher_response_generation"

    def __init__(
        self,
        executor: Callable[
            [TeachingSession, TeachingSkill, TeachingSkill | None, str, str],
            GenerationResult,
        ],
    ):
        self.executor = executor

    def execute(
        self,
        session: TeachingSession,
        primary: TeachingSkill,
        support: TeachingSkill | None,
        action_type: str,
        student_message: str,
    ) -> GenerationResult:
        return self.executor(session, primary, support, action_type, student_message)


class OutputQualityReviewRole:
    """Review a draft; it cannot change learner state or select Skills."""

    role_id = "output_quality_review"

    def __init__(self, executor: Callable[..., TeacherReview]):
        self.executor = executor

    def execute(
        self,
        session: TeachingSession,
        student_message: str,
        draft: TeacherDraft,
        previous_step: TeachingMicroStep | None,
        context_locked: bool,
        *,
        ask_llm: bool = True,
    ) -> TeacherReview:
        return self.executor(
            session,
            student_message,
            draft,
            previous_step,
            context_locked,
            ask_llm=ask_llm,
        )


ROLE_PIPELINE: tuple[str, ...] = (
    StudentDiagnosisRole.role_id,
    TeachingDecisionRole.role_id,
    TeacherResponseRole.role_id,
    OutputQualityReviewRole.role_id,
)


def role_pipeline_audit() -> dict[str, Any]:
    """Return stable public audit metadata without exposing prompts or secrets."""
    return {
        "architecture": "single_agent_four_internal_roles",
        "roles": list(ROLE_PIPELINE),
        "single_state_owner": True,
        "single_action_output": True,
    }
