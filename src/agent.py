from __future__ import annotations

import re
import time
from copy import deepcopy
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Protocol, cast

from src.config import get_agent_settings
from src.lesson_planner import LessonPlanner
from src.llm import LLMUnavailableError, OpenAICompatibleClient
from src.models import (
    AgentDecision,
    ConversationTurn,
    GenerationRevision,
    LLMCallTrace,
    QuestionContract,
    SessionStatus,
    SkillPlan,
    StudentProfile,
    StudentState,
    TeacherDraft,
    TeacherReview,
    TeachingGoal,
    TeachingMicroStep,
    TeachingPhase,
    TeachingSession,
    now_iso,
)
from src.roles import (
    OutputQualityReviewRole,
    StudentDiagnosisRole,
    TeacherResponseRole,
    TeachingDecisionRole,
    role_pipeline_audit,
)
from src.skills import SkillLibrary, TeachingSkill
from src.state_tracker import StateTracker, has_negative_signal, has_prompt_injection
from src.storage import SessionStore

GENERIC_SKILLS = {
    "diagnostic": "diagnostic_questioning_v1",
    "scaffold": "scaffolded_hint_ladder_v1",
    "correction": "misconception_contrast_correction_v1",
    "transfer": "transfer_verification_v1",
}


class SessionStoreLike(Protocol):
    def save(self, session: TeachingSession) -> Any: ...


@dataclass
class MessageGeneration:
    message: str
    micro_step: TeachingMicroStep | None = None
    review: TeacherReview | None = None
    audit: dict[str, Any] | None = None
    fallback_reason: str = ""


class HybridTeachingAgent:
    def __init__(
        self,
        library: SkillLibrary | None = None,
        llm: OpenAICompatibleClient | None = None,
        store: SessionStoreLike | None = None,
        settings: dict[str, Any] | None = None,
    ):
        self.library = library or SkillLibrary()
        self.llm = llm or OpenAICompatibleClient()
        self.store = store or SessionStore()
        self.settings = settings or get_agent_settings()
        self.tracker = StateTracker(self.llm, settings=self.settings)
        self.lesson_planner = LessonPlanner(self.llm)
        self.diagnosis_role = StudentDiagnosisRole(self.tracker)
        self.decision_role = TeachingDecisionRole(self._decide)
        self.response_role: TeacherResponseRole[MessageGeneration] = TeacherResponseRole(
            self._generate_teacher_message
        )
        self.review_role = OutputQualityReviewRole(self._review_teacher_draft)
        self._teacher_draft_retry = False

    def start_session(
        self,
        goal: TeachingGoal,
        profile: StudentProfile,
        initial_state: StudentState,
        history: list[dict[str, str]] | None = None,
        available_skill_ids: list[str] | None = None,
    ) -> TeachingSession:
        for point in goal.knowledge_points:
            initial_state.mastery.setdefault(point, 0.3)
        allowed = list(dict.fromkeys(available_skill_ids or []))
        snapshot = {
            skill.skill_id: skill.version
            for skill in self.library.skills
            if not allowed or skill.skill_id in allowed
        }
        session = TeachingSession(
            goal=goal,
            profile=profile,
            state=initial_state,
            imported_history=list(history or []),
            available_skill_ids=allowed,
            skill_snapshot=snapshot,
            teaching_route=self.lesson_planner.build(goal),
        )
        if session.teaching_route:
            initial_state.next_focus = session.teaching_route.current_step().learning_target
        decision = self.decision_role.execute(session, student_message="", initial=True)
        initial_state.phase = decision.phase
        self._append_turn(session, "", decision, initial_state, initial_state)
        self.store.save(session)
        return session

    @staticmethod
    def _coerce_session(session: TeachingSession) -> TeachingSession:
        """Rehydrate sessions that survived a Streamlit module hot reload.

        Streamlit can retain an object built by the previous import of
        ``src.models``.  It may look structurally correct, but its nested
        ``StudentState`` is a different Python class, so constructing a new
        ``ConversationTurn`` can fail Pydantic validation.  Revalidating the
        serialized payload keeps the agent safe for both the UI and direct
        callers.
        """
        if isinstance(session, TeachingSession) and isinstance(session.state, StudentState):
            return session
        return TeachingSession.model_validate(session.model_dump(mode="python"))

    def handle_student_message(self, session: TeachingSession, message: str) -> TeachingSession:
        session = self._coerce_session(session)
        if session.status != SessionStatus.ACTIVE.value and session.status != SessionStatus.ACTIVE:
            raise RuntimeError("教学会话已终止，请新建会话")
        before = session.state.model_copy(deep=True)
        started = time.perf_counter()
        calls_before = getattr(self.llm, "total_calls", None)
        current_skill = session.turns[-1].selected_skill_id if session.turns else ""
        previous_teacher_message = session.turns[-1].teacher_message if session.turns else ""
        after = self.diagnosis_role.execute(
            session.goal,
            session.profile,
            before,
            message,
            current_skill,
            round_index=session.answered_rounds() + 1,
            previous_teacher_message=previous_teacher_message,
        )
        session.state = after
        if session.teaching_route:
            # The assessor can mention a future concept in a broad answer.
            # That is evidence for diagnosis, not permission to advance the
            # curriculum.  The route can move only when the answer matches the
            # step that was active before this turn.
            route_before = session.teaching_route.model_copy(deep=True)
            allow_route_advance = self._route_transition_allowed(
                session,
                message,
                route_before,
            )
            session.teaching_route = self.lesson_planner.sync(
                session.teaching_route,
                after,
                allow_advance=allow_route_advance,
            )
            after.next_focus = session.teaching_route.current_step().learning_target

        terminal = self._terminal_decision(session, message)
        decision = terminal or self.decision_role.execute(session, message, initial=False)
        calls_after = getattr(self.llm, "total_calls", None)
        if isinstance(calls_before, int) and isinstance(calls_after, int):
            call_delta = max(0, calls_after - calls_before)
            if call_delta:
                decision.llm_trace.append(
                    LLMCallTrace(
                        operation="teaching_turn",
                        model=str(getattr(self.llm, "model", "")),
                        attempts=call_delta,
                        latency_ms=round((time.perf_counter() - started) * 1000),
                        success=True,
                    )
                )
        if terminal and current_skill and current_skill != decision.primary_skill_id:
            decision.switch_reason = (
                f"终止规则触发前完成最后一次策略调整，Skill 从 {current_skill} "
                f"切换为 {decision.primary_skill_id}。"
            )
        session.state.phase = self._phase_for_decision(decision)
        after.phase = session.state.phase
        self._append_turn(session, message, decision, before, after)
        session.rounds_in_current_run += 1
        if decision.should_stop:
            # TeachingSession serializes enum values for stable JSON/UI labels.
            session.status = decision.status.value  # type: ignore[assignment]
            session.termination_reason = decision.stop_reason
        session.updated_at = now_iso()
        self.store.save(session)
        return session

    def resume_session(
        self,
        session: TeachingSession,
        *,
        reset_misconceptions: bool = False,
    ) -> TeachingSession:
        """Resume the same persisted session after a pause.

        Continuation deliberately keeps the session id, turns, route and
        evidence.  Only the current run's round budget is reset, so the next
        answer is allowed a fresh eight-turn window while the complete audit
        trail remains visible in replay.
        """

        session = self._coerce_session(session)

        if session.status != SessionStatus.UNABLE.value and session.status != SessionStatus.UNABLE:
            raise RuntimeError("只有已暂停的教学会话可以继续")
        session.status = SessionStatus.ACTIVE.value  # type: ignore[assignment]
        session.termination_reason = ""
        session.rounds_in_current_run = 0
        session.state.no_progress_rounds = 0
        if reset_misconceptions:
            for misconception in session.state.misconceptions:
                misconception.count = 1
            for misconception in session.state.misconception_states:
                misconception.count = 1
                misconception.consecutive_count = 1
                misconception.improving = False
        if session.teaching_route:
            session.state.next_focus = session.teaching_route.current_step().learning_target

        before = session.state.model_copy(deep=True)
        decision = self.decision_role.execute(session, "", initial=False)
        session.state.phase = self._phase_for_decision(decision)
        after = session.state.model_copy(deep=True)
        self._append_turn(session, "", decision, before, after)
        session.updated_at = now_iso()
        self.store.save(session)
        return session

    @staticmethod
    def _phase_for_decision(decision: AgentDecision) -> TeachingPhase:
        if decision.should_stop:
            return "completed" if decision.status == SessionStatus.SUCCESS else "paused"
        if decision.action_type == "subject_instruction" and decision.support_skill_id == GENERIC_SKILLS["diagnostic"]:
            return "diagnosis"
        return cast(
            TeachingPhase,
            {
                "diagnostic": "diagnosis",
                "scaffold": "instruction",
                "correction": "repair",
                "transfer": "transfer",
            }.get(
                decision.action_type,
                "practice" if decision.action_type == "subject_instruction" else "instruction",
            ),
        )

    def _skill_plan(
        self,
        session: TeachingSession,
        primary: TeachingSkill,
        support: TeachingSkill | None,
        candidate_ids: list[str],
        previous_skill_id: str,
    ) -> SkillPlan:
        subject = primary if primary.skill_type == "subject" else support if support and support.skill_type == "subject" else None
        strategy = primary if primary.skill_type != "subject" else support if support and support.skill_type != "subject" else None
        previous = self.library.get(previous_skill_id) if previous_skill_id else None
        previous_subject = previous if previous and previous.skill_type == "subject" else None
        previous_strategy = previous if previous and previous.skill_type != "subject" else None
        if session.turns and session.turns[-1].skill_plan:
            previous_plan = session.turns[-1].skill_plan
            previous_subject = self.library.get(previous_plan.content_skill_id) if previous_plan.content_skill_id else None
            previous_strategy = self.library.get(previous_plan.strategy_skill_id) if previous_plan.strategy_skill_id else None
        strategy_candidates = [
            skill.skill_id
            for skill in self.library.skills
            if skill.skill_type != "subject"
            and (not session.available_skill_ids or skill.skill_id in session.available_skill_ids)
        ]
        return SkillPlan(
            content_skill_id=subject.skill_id if subject else None,
            strategy_skill_id=strategy.skill_id if strategy else None,
            content_skill_reason=(subject.name if subject else "未匹配学科 Skill，使用通用教学") if subject else "",
            strategy_reason=(strategy.name if strategy else "未匹配教学策略，使用通用教学") if strategy else "",
            candidate_content_skill_ids=[skill_id for skill_id in candidate_ids if self.library.get(skill_id).skill_type == "subject"],
            candidate_strategy_skill_ids=strategy_candidates,
            content_switch=bool(subject and previous_subject and subject.skill_id != previous_subject.skill_id),
            strategy_switch=bool(strategy and previous_strategy and strategy.skill_id != previous_strategy.skill_id),
        )

    @staticmethod
    def _question_contract(step: TeachingMicroStep | None) -> QuestionContract | None:
        if step is None:
            return None
        return QuestionContract(
            focus=step.focus,
            context=step.context,
            requested_target=step.requested_target,
            response_mode=step.response_mode,
            option_ids=[option.option_id for option in step.options],
            expected_signal=step.expected_signal,
            valid=bool(step.focus and step.requested_target),
        )

    def regenerate_current_turn(
        self,
        session: TeachingSession,
        response_mode_override: str | None = None,
    ) -> TeachingSession:
        """Regenerate only the latest teacher prompt without changing learner state."""
        session = self._coerce_session(session)
        if session.status != SessionStatus.ACTIVE.value and session.status != SessionStatus.ACTIVE:
            raise RuntimeError("教学会话已终止，不能重新生成当前题目")
        if not session.turns:
            raise RuntimeError("当前会话没有可重新生成的教师题目")
        current = session.turns[-1]
        primary = self.library.get(current.selected_skill_id)
        support = self.library.get(current.support_skill_id) if current.support_skill_id else None
        context_session = session.model_copy(deep=True)
        context_session.turns = context_session.turns[:-1]
        if response_mode_override is not None:
            context_session.profile.response_preference = response_mode_override  # type: ignore[assignment]
        generation = self._generate_teacher_message(
            context_session,
            primary,
            support,
            current.action_type,
            current.student_message,
        )
        current.teacher_message = generation.message
        current.micro_step = generation.micro_step
        current.teacher_review = generation.review
        current.generation_audit = generation.audit or {}
        if response_mode_override is not None:
            current.generation_audit["response_mode_override"] = response_mode_override
        current.fallback_reason = generation.fallback_reason
        current.question_contract = self._question_contract(generation.micro_step)
        current.generation_revisions.append(
            GenerationRevision(
                revision_index=len(current.generation_revisions) + 1,
                teacher_message=generation.message,
                reason="学生请求重新生成当前题目",
            )
        )
        session.updated_at = now_iso()
        self.store.save(session)
        return session

    def _terminal_decision(self, session: TeachingSession, student_message: str = "") -> AgentDecision | None:
        threshold = float(self.settings.get("mastery_threshold", 0.8))
        point_threshold = float(self.settings.get("point_mastery_threshold", 0.70))
        evidence_levels = {
            point: {item.evidence_level for item in knowledge.evidence}
            for point, knowledge in session.state.knowledge_states.items()
        }
        evidence_complete = all(
            any(level in {"correct", "explained", "transfer"} for level in evidence_levels.get(point, set()))
            for point in session.state.mastery
        )
        explained = any(
            level in {"explained", "transfer"}
            for levels in evidence_levels.values()
            for level in levels
        )
        mastered = (
            bool(session.state.mastery)
            and session.state.average_mastery() >= threshold
            and all(value >= point_threshold for value in session.state.mastery.values())
            and evidence_complete
            and explained
        )
        transfer_prompt_delivered = bool(
            session.turns
            and session.turns[-1].action_type == "transfer"
        )
        if self._simple_mode_enabled() and not transfer_prompt_delivered:
            # An answer mentioning a new problem is not enough to verify
            # transfer when the previous teacher turn did not ask a transfer
            # question. The simplified flow first presents that task, then
            # judges the student's response to it.
            mastered = False
        if mastered and session.state.transfer_verified:
            preserved_plan = self._preserved_skill_plan(
                session,
                strategy_skill_id=GENERIC_SKILLS["transfer"],
                strategy_reason="掌握证据充分后进行独立迁移验证",
            )
            return AgentDecision(
                primary_skill_id=GENERIC_SKILLS["transfer"],
                selection_reason="目标知识点均达到掌握阈值，且迁移验证已经通过。",
                action_type="terminate_success",
                teacher_message="你已经能够解释关键原理并迁移到新情境。本轮教学完成，我建议稍后再做一道延迟练习巩固。",
                expected_signal="教学目标达成",
                should_stop=True,
                status=SessionStatus.SUCCESS,
                stop_reason="所有知识点掌握度不低于阈值，迁移题通过。",
                decision_mode="termination_guard",
                candidate_skill_ids=[GENERIC_SKILLS["transfer"]],
                policy_rule="mastery >= threshold AND transfer_verified",
                skill_plan=preserved_plan,
            )

        max_rounds = int(self.settings.get("max_rounds", 8))
        no_progress_limit = int(self.settings.get("no_progress_limit", 3))
        repeated = max(
            (item.consecutive_count for item in session.state.misconception_states),
            default=max((item.count for item in session.state.misconceptions), default=0),
        )
        previous_skill = session.turns[-1].selected_skill_id if session.turns else ""
        correction_was_delivered = previous_skill == GENERIC_SKILLS["correction"]
        # The current student message is assessed before its turn is appended.
        answered_rounds = session.rounds_in_current_run + 1
        repeated_after_correction = (
            correction_was_delivered
            and session.state.no_progress_rounds >= no_progress_limit
            and repeated >= no_progress_limit
        )
        if answered_rounds >= max_rounds or repeated_after_correction:
            reached_limit = answered_rounds >= max_rounds
            reason = (
                f"达到最大教学轮数 {max_rounds}。"
                if reached_limit
                else f"误解纠正后仍未改善（当前连续 {repeated} 轮）。"
            )
            current = session.turns[-1] if session.turns else None
            current_skill = current.selected_skill_id if current else GENERIC_SKILLS["diagnostic"]
            current_content_skill = (
                current.skill_plan.content_skill_id
                if current and current.skill_plan and current.skill_plan.content_skill_id
                else current.content_skill_id
                if current and current.content_skill_id
                else None
            )
            latest_evidence = session.state.evidence[-1] if session.state.evidence else None
            needs_repair = bool(session.state.misconception_states or session.state.misconceptions) or bool(
                latest_evidence and latest_evidence.signal_type in {"negative", "empty"}
            )
            if reached_limit:
                if needs_repair:
                    teacher_message = (
                        f"本轮已达到设定的 {max_rounds} 轮上限。刚才的回答与当前情境中的关键关系仍有冲突，"
                        f"因此暂不记为掌握。已保留这条误解证据；下次从“{session.state.next_focus}”开始，"
                        "先重新核对当前情境中的判断依据，再进入新的知识点。"
                    )
                    selection_reason = (
                        "达到轮数上限前检测到仍需修复的理解冲突；终止回复先使用误解纠正策略，"
                        "再保留状态并提供后续路径。"
                    )
                    terminal_strategy = GENERIC_SKILLS["correction"]
                    # The final turn is still a terminal audit record, but its
                    # visible teaching role is correction rather than a generic
                    # subject explanation. The subject Skill remains attached
                    # as support so the content/strategy split stays explicit.
                    current_skill = terminal_strategy
                    terminal_plan = SkillPlan(
                        content_skill_id=current_content_skill,
                        strategy_skill_id=terminal_strategy,
                        content_skill_reason="终止前保留当前学科情境",
                        strategy_reason="达到轮数上限前先总结并纠正当前理解冲突",
                        candidate_content_skill_ids=[current_content_skill] if current_content_skill else [],
                        candidate_strategy_skill_ids=[terminal_strategy],
                        content_switch=False,
                        strategy_switch=bool(
                            current
                            and (
                                not current.skill_plan
                                or current.skill_plan.strategy_skill_id != terminal_strategy
                            )
                        ),
                    )
                else:
                    teacher_message = (
                        f"本轮已达到设定的 {max_rounds} 轮上限。你刚才的回答已作为最新学习证据保存；"
                        f"当前还需要继续验证“{session.state.next_focus}”。之后可以从这个关注点继续，而不必重新开始。"
                    )
                    selection_reason = "达到轮数上限；保留最后一轮状态证据并暂停，避免把轮数限制误报为学习失败。"
                    terminal_plan = self._preserved_skill_plan(
                        session,
                        strategy_reason="达到轮数上限，保留上一轮教学方案以便继续会话",
                    )
                action_type = "terminate_max_rounds"
                policy_rule = "max_rounds"
            else:
                teacher_message = (
                    "本轮先暂停。纠正之后同一理解困难仍没有出现可确认的改善证据，"
                    f"我已保存当前状态；下一次建议从“{session.state.next_focus}”换一种讲法继续。"
                )
                selection_reason = "纠错后仍未改善；暂停并保留误解证据，等待换一种教学路径。"
                action_type = "terminate_no_improvement"
                policy_rule = "no_improvement_after_correction"
                terminal_plan = self._preserved_skill_plan(
                    session,
                    strategy_skill_id=GENERIC_SKILLS["correction"],
                    strategy_reason="纠错后仍未改善，暂停并保留当前教学方案",
                )
            return AgentDecision(
                primary_skill_id=current_skill,
                selection_reason=selection_reason,
                action_type=action_type,
                teacher_message=teacher_message,
                expected_signal="学生获得明确后续路径",
                should_stop=True,
                status=SessionStatus.UNABLE,
                stop_reason=reason,
                decision_mode="termination_guard",
                candidate_skill_ids=[],
                policy_rule=policy_rule,
                support_skill_id=current_content_skill if reached_limit and needs_repair else None,
                skill_plan=terminal_plan,
            )
        return None

    @staticmethod
    def _latest_plan(session: TeachingSession) -> SkillPlan | None:
        for turn in reversed(session.turns):
            if turn.skill_plan is not None:
                return turn.skill_plan
        return None

    def _preserved_skill_plan(
        self,
        session: TeachingSession,
        *,
        strategy_skill_id: str | None = None,
        strategy_reason: str = "",
    ) -> SkillPlan:
        """Keep content and teaching strategy roles stable at terminal turns."""
        previous = self._latest_plan(session)
        content_id = previous.content_skill_id if previous else None
        previous_strategy = previous.strategy_skill_id if previous else None
        if content_id is None and session.turns:
            latest = session.turns[-1]
            content_id = latest.content_skill_id
            previous_strategy = previous_strategy or latest.strategy_skill_id
        selected_strategy = strategy_skill_id if strategy_skill_id is not None else previous_strategy
        content_reason = previous.content_skill_reason if previous else "沿用当前学科情境"
        selected_reason = strategy_reason or (previous.strategy_reason if previous else "沿用当前教学策略")
        return SkillPlan(
            content_skill_id=content_id,
            strategy_skill_id=selected_strategy,
            content_skill_reason=content_reason,
            strategy_reason=selected_reason,
            candidate_content_skill_ids=[content_id] if content_id else [],
            candidate_strategy_skill_ids=[selected_strategy] if selected_strategy else [],
            content_switch=False,
            strategy_switch=bool(previous and previous.strategy_skill_id != selected_strategy),
        )

    def _route_transition_allowed(
        self,
        session: TeachingSession,
        student_message: str,
        route: Any,
    ) -> bool:
        """Permit route progress only for evidence about the active step."""
        if not session.turns or not route or not route.steps:
            return False
        latest = session.state.evidence[-1] if session.state.evidence else None
        previous_step = session.turns[-1].micro_step
        if (
            latest is None
            or latest.student_quote.strip() != student_message.strip()
            or previous_step is None
        ):
            return False
        current = route.current_step()
        if current.kind == "transfer":
            return latest.signal_type == "transfer" and latest.evidence_level == "transfer"
        levels = {"correct": 1, "explained": 2, "transfer": 3}
        observed = levels.get(latest.evidence_level, 0)
        required = levels.get(current.evidence_requirement, 1)
        matches_step = self._focus_terms_match(latest.knowledge_point, current.knowledge_point)
        if (
            self.settings.get("simple_teaching_mode", False)
            and latest.signal_type in {"positive", "transfer"}
            and observed >= required
            and matches_step
        ):
            # In the simplified live loop, evidence for the active step is
            # enough to advance. Requiring lexical overlap with the previous
            # generated question made correct natural-language answers look
            # incomplete and trapped the learner in the same micro-step.
            return True
        return (
            latest.signal_type in {"positive", "transfer"}
            and observed >= required
            and matches_step
            and self._answer_references_step(student_message, previous_step)
        )

    def _simple_mode_enabled(self) -> bool:
        return bool(self.settings.get("simple_teaching_mode", False))

    @staticmethod
    def _student_concern(message: str) -> dict[str, str]:
        """Extract a concrete learner question that must be answered first."""
        text = re.sub(r"\s+", "", message or "")
        if not text:
            return {}
        question_cues = (
            "？", "?", "是不是", "是否", "为什么", "怎么", "到底", "算不算",
            "有没有", "有吗", "能不能", "哪个", "什么",
        )
        uncertainty_cues = ("我不确定", "我有点不确定", "搞不清", "不明白", "不懂", "不清楚")
        if not any(cue in text for cue in (*question_cues, *uncertainty_cues)):
            return {}
        if "合力" in text and any(cue in text for cue in ("零", "算不算", "有没有", "有吗")):
            kind = "force_zero"
        elif any(cue in text for cue in ("脚底", "摩擦力", "地板")):
            kind = "foot_friction"
        elif any(cue in text for cue in ("受到力", "有没有力", "到底有没有力", "受力")):
            kind = "force_presence"
        elif any(cue in text for cue in ("前倾", "向前", "往前")) and "为什么" in text:
            kind = "inertia_forward"
        else:
            kind = "generic"
        return {"kind": kind, "text": message.strip()}

    @staticmethod
    def _is_abstract_concern_question(question: str) -> bool:
        text = re.sub(r"\s+", "", question or "")
        return any(
            cue in text
            for cue in ("判断依据", "这一步", "当前关注点", "一个区别", "下一步推理", "只说明")
        )

    @staticmethod
    def _is_generic_concern_answer(answer: str) -> bool:
        text = re.sub(r"\s+", "", answer or "")
        if len(text) < 6:
            return True
        return any(
            cue in text
            for cue in ("先把你刚才卡住的一个区别说清楚", "你的回答会作为我们继续学习的依据", "我们先一步一步来")
        )

    def _context_has_floor_friction(self, session: TeachingSession) -> bool:
        context = "\n".join(turn.teacher_message for turn in session.turns)
        return any(cue in context for cue in ("脚底", "地板", "摩擦力", "摩擦"))

    def _concern_answer_is_sufficient(self, concern: dict[str, str], answer: str) -> bool:
        text = re.sub(r"\s+", "", answer or "")
        if self._is_generic_concern_answer(text):
            return False
        kind = concern.get("kind", "generic")
        if kind == "force_zero":
            return (
                "合力" in text
                and ("速度" in text or "加速度" in text)
                and any(cue in text for cue in ("不为零", "不能为零", "不等于零", "不为0"))
            )
        if kind == "foot_friction":
            return "摩擦" in text and any(cue in text for cue in ("脚", "地板", "地面"))
        if kind == "force_presence":
            return "力" in text and any(cue in text for cue in ("受到", "摩擦", "合力"))
        if kind == "inertia_forward":
            return any(cue in text for cue in ("惯性", "上半身", "前倾", "原来的速度"))
        return True

    def _concern_fallback_payload(
        self,
        session: TeachingSession,
        concern: dict[str, str],
        route_step: Any,
    ) -> dict[str, str]:
        """Create a short answer-first bridge when model output is too abstract."""
        kind = concern.get("kind", "generic")
        if kind == "force_zero" and self._context_has_floor_friction(session):
            direct_answer = (
                "先看速度变化：刹车时乘客的脚和下半身随车减速，说明乘客的运动状态在变化，"
                "因此有加速度，乘客受到的合力不为零。"
            )
            question = "现在只判断：这个使乘客减速的水平力主要来自哪里？"
        elif kind in {"force_zero", "force_presence", "foot_friction"}:
            direct_answer = (
                "要先区分题设。若题目明确说乘客没有受到水平方向的力，水平方向合力就是零；"
                "若脚底和地板之间存在摩擦力，乘客就会受到向后的水平力，不能把两种情形混在一起。"
            )
            question = "当前情境中，题目有没有说明脚底和地板之间存在摩擦力？"
        elif kind == "inertia_forward":
            direct_answer = (
                "前倾不是因为受到一个向前的推力，而是脚部先受到车地板的摩擦力并随车减速，"
                "上半身暂时保持原来的向前运动状态。"
            )
            question = "现在只判断：前倾主要说明身体哪一部分没有立即跟着车减速？"
        else:
            direct_answer = "先回答你刚才提出的具体疑问，不急着进入下一步；关键要看当前情境中给出的受力和运动变化。"
            focus = route_step.knowledge_point if route_step else self._focus_label(session)
            question = f"关于“{focus}”，题目中哪一个条件最直接决定你的判断？"
        return {
            "student_concern": concern.get("text", ""),
            "direct_answer": direct_answer,
            "feedback": direct_answer,
            "question": question,
            "context": route_step.title if route_step else session.goal.topic,
            "known_fact": "先回应学生的明确疑问，再提出一个最小判断问题",
            "expected_signal": "学生能回答当前最小问题",
        }

    def _decide_simple(self, session: TeachingSession, student_message: str, initial: bool) -> AgentDecision:
        """Run the small live-teaching loop used by the real-time UI.

        The original pipeline is intentionally kept below this method for
        compatibility with archived sessions and evaluation tests.  New live
        turns use a simpler contract: choose one content/strategy Skill,
        briefly respond to the learner's last message, then ask one question.
        """
        prompt_injection = has_prompt_injection(student_message)
        selection_message = "" if prompt_injection else student_message
        ranked, candidate_audit = self.library.candidates(
            session.goal,
            session.state,
            selection_message,
            profile=session.profile,
            limit=int(self.settings.get("candidate_limit", 5)),
            include_generic=False,
        )
        if session.available_skill_ids:
            allowed = set(session.available_skill_ids)
            ranked = [skill for skill in ranked if skill.skill_id in allowed]
            candidate_audit = [item for item in candidate_audit if item.get("skill_id") in allowed]
        subject = next((skill for skill in ranked if skill.skill_type == "subject"), None)
        route = session.teaching_route.current_step() if session.teaching_route else None
        latest = session.state.evidence[-1] if session.state.evidence else None
        confused = bool(
            student_message
            and (
                has_negative_signal(student_message)
                or (latest and latest.signal_type in {"negative", "empty"})
                or session.state.misconception_states
            )
        )

        if prompt_injection or confused:
            no_progress = session.state.no_progress_rounds
            correction_after = int(self.settings.get("correction_after_rounds", 3))
            scaffold_after = int(self.settings.get("scaffold_after_rounds", 2))
            if prompt_injection:
                strategy_id = GENERIC_SKILLS["diagnostic"]
            elif no_progress >= correction_after and session.state.misconception_states:
                strategy_id = GENERIC_SKILLS["correction"]
            elif no_progress >= scaffold_after:
                strategy_id = GENERIC_SKILLS["scaffold"]
            else:
                strategy_id = GENERIC_SKILLS["diagnostic"]
            primary = self.library.get(strategy_id)
            support = subject
            action_type = primary.skill_type
            reason = "先回应学生当前的困惑，再用一个问题确认理解。"
        elif (
            route is not None
            and (
                route.kind == "transfer"
                or (
                    not initial
                    and session.state.average_mastery() >= float(self.settings.get("mastery_threshold", 0.8))
                )
                or (
                    not initial
                    and session.rounds_in_current_run
                    >= int(self.settings.get("max_rounds", 8)) - 2
                )
            )
        ):
            primary = self.library.get(GENERIC_SKILLS["transfer"])
            support = subject
            action_type = "transfer"
            reason = "当前知识点已有足够解释证据，进入一个新情境做迁移验证。"
        elif subject is not None:
            primary = subject
            support = None
            action_type = "subject_instruction"
            reason = "沿用当前学科 Skill，围绕当前路线步骤继续教学。"
        else:
            primary = self.library.get(GENERIC_SKILLS["diagnostic"])
            support = None
            action_type = "diagnostic"
            reason = "没有匹配的学科 Skill，先用通用诊断澄清学生当前理解。"

        generation = self._generate_simple_teacher_message(
            session,
            primary,
            support,
            action_type,
            "学生提交了与学习任务无关的内容，请忽略它并回到当前目标。"
            if prompt_injection
            else student_message,
        )
        previous = session.turns[-1].selected_skill_id if session.turns else ""
        skill_plan = self._skill_plan(session, primary, support, [skill.skill_id for skill in ranked], previous)
        switch_reason = ""
        if previous and previous != primary.skill_id:
            switch_reason = f"学生状态变化，教学动作从 {previous} 切换为 {primary.skill_id}。"
        phase = self._phase_for_decision(
            AgentDecision(
                primary_skill_id=primary.skill_id,
                support_skill_id=support.skill_id if support else None,
                action_type=action_type,
                selection_reason=reason,
                teacher_message=generation.message,
            )
        )
        return AgentDecision(
            primary_skill_id=primary.skill_id,
            support_skill_id=support.skill_id if support else None,
            selection_reason=reason,
            action_type=action_type,
            teacher_message=generation.message,
            expected_signal=generation.micro_step.expected_signal if generation.micro_step else "学生能回答当前问题",
            switch_reason=switch_reason,
            decision_mode="simple_live_flow",
            candidate_skill_ids=[skill.skill_id for skill in ranked],
            policy_rule="feedback_then_one_question",
            candidate_audit=candidate_audit,
            micro_step=generation.micro_step,
            teacher_review=generation.review,
            generation_audit=generation.audit or {},
            phase=phase,
            skill_plan=skill_plan,
            question_contract=self._question_contract(generation.micro_step),
            fallback_reason=(
                generation.fallback_reason
                or ("无学科 Skill 通过硬过滤，已回退到通用诊断。" if subject is None else "")
            ),
        )

    def _generate_simple_teacher_message(
        self,
        session: TeachingSession,
        primary: TeachingSkill,
        support: TeachingSkill | None,
        action_type: str,
        student_message: str,
    ) -> MessageGeneration:
        route_step = session.teaching_route.current_step() if session.teaching_route else None
        previous_turn = session.turns[-1] if session.turns else None
        previous_question = previous_turn.teacher_message if previous_turn else "无"
        route_target = route_step.learning_target if route_step else session.state.next_focus or session.goal.objective
        concern = self._student_concern(student_message)
        if self.llm.available:
            try:
                data = self.llm.structured(
                    (
                        "你是一名真正和学生对话的教师。你的任务不是收集审计证据，而是帮助学生理解当前问题。"
                        "如果学生上一句话包含明确疑问或不确定，必须先直接回答这个疑问，再提出一个最小追问；"
                        "不能只说‘我们先澄清一下’，也不能把问题改写成抽象的‘判断依据’。"
                        "先根据学生上一句话给出一到两句直接、具体的反馈；如果学生困惑，先澄清困惑中的关键区别。"
                        "然后只提出一个学生可以回答的问题。反馈可以解释当前误解，但不要一次讲完整节课，"
                        "不要模拟学生，不要出现 Skill、路线、掌握度、证据、内部流程等词。"
                        "必须延续当前情境；学生已经回答正确时，承认正确并推进到下一个认知任务，不能换词重复原问题。"
                        "如果当前是迁移验证，必须换一个新情境。"
                        "若学生问的是‘速度是否变化、是否有加速度、合力是否为零’，优先建立‘速度变化→有加速度→合力不为零’的链条；"
                        "若当前情境已给出脚底与地板摩擦力，就不能回答成‘乘客没有水平力’。"
                    ),
                    (
                        f"课程：{session.goal.course}\n主题：{session.goal.topic}\n教学目标：{session.goal.objective}\n"
                        f"当前路线目标：{route_target}\n当前教学动作：{action_type}\n"
                        f"学生状态：{session.state.model_dump_json()}\n"
                        f"上一轮教师话语：{previous_question}\n学生刚才的话：{student_message or '尚未回答'}\n"
                        f"学生明确疑问（若为空则没有）：{concern.get('text', '无')}\n"
                        "请输出反馈和一个问题。"
                    ),
                    '{"student_concern":"学生明确提出的疑问或空字符串","direct_answer":"必须先回答该疑问；无明确疑问时可为空","feedback":"给学生的简短反馈或澄清","question":"唯一一个具体、可回答的问题","context":"当前情境","known_fact":"本轮可用的已知条件","expected_signal":"希望学生表现出的理解"}',
                    temperature=float(self.settings.get("temperature", 0.2)),
                )
                return self._build_simple_generation(session, data, action_type, student_message)
            except (LLMUnavailableError, ValueError, TypeError, KeyError):
                pass
        return self._simple_fallback_generation(session, action_type, route_step, student_message)

    def _build_simple_generation(
        self,
        session: TeachingSession,
        data: dict[str, Any],
        action_type: str,
        student_message: str = "",
        enforce_concern: bool = True,
    ) -> MessageGeneration:
        route_step = session.teaching_route.current_step() if session.teaching_route else None
        focus = route_step.knowledge_point if route_step else self._focus_label(session)
        concern = self._student_concern(student_message)
        direct_answer = str(data.get("direct_answer", "")).strip()
        feedback = direct_answer if concern and direct_answer else str(data.get("feedback", "")).strip()
        question = str(data.get("question", "")).strip()
        if not question:
            raise ValueError("简单教师回复缺少问题")
        if enforce_concern and concern and (
            not direct_answer
            or not self._concern_answer_is_sufficient(concern, direct_answer)
            or self._is_abstract_concern_question(question)
        ):
            fallback = self._build_simple_generation(
                session,
                self._concern_fallback_payload(session, concern, route_step),
                action_type,
                student_message,
                enforce_concern=False,
            )
            fallback.fallback_reason = "answer_first_concern_guard"
            fallback.audit = {**(fallback.audit or {}), "fallback_reason": fallback.fallback_reason}
            return fallback
        feedback = re.sub(r"[？?]+", "。", feedback).strip()
        if route_step is not None and route_step.kind == "transfer":
            prefix = "再换一个" if session.turns and session.turns[-1].action_type == "transfer" else "换一个"
            question = f"请{prefix}不同于刚才的新情境，说明“{focus}”如何表现"
        elif session.turns:
            previous = session.turns[-1].teacher_message
            previous_question = re.split(r"[。！？!?]\s*", previous.strip())[-1]
            similarity = SequenceMatcher(None, question, previous_question).ratio()
            if question in previous or previous_question in question or similarity >= 0.72:
                if concern:
                    fallback_payload = self._concern_fallback_payload(session, concern, route_step)
                    question = fallback_payload["question"]
                elif any(cue in student_message for cue in ("有没有受到力", "到底有没有力", "受到力")):
                    question = "现在只判断：身体在水平方向是否受到向后的力"
                else:
                    question = f"请只说明“{focus}”这一次最关键的一个区别"
        question = re.sub(r"[。！？!?]+", "。", question).strip().rstrip("。") + "？"
        context = str(data.get("context", "")).strip() or (route_step.title if route_step else session.goal.topic)
        known_fact = str(data.get("known_fact", "")).strip()
        expected = str(data.get("expected_signal", "")).strip() or "学生能回答当前问题"
        step = TeachingMicroStep(
            focus=focus,
            context=context,
            known_fact=known_fact,
            requested_target=question.rstrip("？"),
            representation="用自己的话回答",
            expected_signal=expected,
            step_index=(session.turns[-1].micro_step.step_index + 1 if session.turns and session.turns[-1].micro_step else 1),
            response_mode="open",
            input_hint="请用自己的话回答",
        )
        message = f"{feedback}\n\n{question}" if feedback else question
        return MessageGeneration(
            message=message.strip(),
            micro_step=step,
            review=TeacherReview(
                valid=True,
                one_step=True,
                one_context=True,
                one_question=True,
                fact_consistent=True,
                same_context=True,
                answer_leakage=False,
                issues=[],
            ),
            audit={
                "architecture": "single_agent_simple_flow",
                "flow": ["understand_student", "brief_feedback", "one_question"],
                "action_type": action_type,
                "llm_generated": True,
                "student_facing_feedback": bool(feedback),
                "student_concern": concern.get("text", ""),
                "student_concern_kind": concern.get("kind", ""),
                "answer_first": bool(concern),
                "concern_addressed": bool(concern and direct_answer) or not concern,
            },
        )

    def _simple_fallback_generation(
        self,
        session: TeachingSession,
        action_type: str,
        route_step: Any,
        student_message: str,
    ) -> MessageGeneration:
        focus = route_step.knowledge_point if route_step else self._focus_label(session)
        concern = self._student_concern(student_message)
        if concern:
            payload = self._concern_fallback_payload(session, concern, route_step)
        elif action_type == "transfer":
            feedback = "前面的解释已经完成，现在换一个情境检查你能否迁移这个理解。"
            question = f"请换一个不同于刚才的新情境，说明“{focus}”如何表现？"
        elif action_type in {"diagnostic", "correction"}:
            feedback = "我们先把你刚才卡住的一个区别说清楚，再继续往下。"
            question = f"请只用一句话说明你对“{focus}”目前最确定的一个判断。"
        else:
            feedback = "你的回答会作为我们继续学习的依据。"
            target = route_step.learning_target if route_step else f"说明“{focus}”"
            question = f"请用自己的话说明：{target}？"
        if not concern:
            payload = {
                "feedback": feedback,
                "question": question,
                "context": route_step.title if route_step else session.goal.topic,
                "known_fact": "",
                "expected_signal": "学生能回答当前问题",
            }
        return self._build_simple_generation(session, payload, action_type, student_message, enforce_concern=False)

    def _decide(self, session: TeachingSession, student_message: str, initial: bool) -> AgentDecision:
        if self._simple_mode_enabled():
            return self._decide_simple(session, student_message, initial)
        prompt_injection = has_prompt_injection(student_message)
        selection_message = "" if prompt_injection else student_message
        ranked, candidate_audit = self.library.candidates(
            session.goal,
            session.state,
            selection_message,
            profile=session.profile,
            limit=int(self.settings.get("candidate_limit", 5)),
            include_generic=False,
        )
        if session.available_skill_ids:
            allowed = set(session.available_skill_ids)
            ranked = [skill for skill in ranked if skill.skill_id in allowed]
            candidate_audit = [item for item in candidate_audit if item.get("skill_id") in allowed]
        if prompt_injection:
            subject_skill, selection_mode = (ranked[0], "security_guard") if ranked else (None, "security_guard")
        else:
            subject_skill, selection_mode = (
                self._llm_select(ranked, session, student_message, candidate_audit) if ranked else (None, "rule_fallback")
            )
        no_subject_match = subject_skill is None

        forced_type: str | None = None
        scaffold_after = int(self.settings.get("scaffold_after_rounds", 2))
        correction_after = int(self.settings.get("correction_after_rounds", 3))
        if prompt_injection:
            forced_type = "diagnostic"
        elif no_subject_match:
            forced_type = "diagnostic"
        elif not initial:
            explicit_confusion = has_negative_signal(student_message)
            latest_signal = session.state.evidence[-1].signal_type if session.state.evidence else ""
            active_misconception = bool(session.state.misconception_states)
            # The semantic state assessor is authoritative here. A learner may
            # state a misconception confidently without saying “I don't know”,
            # so routing must also honor the LLM-produced active misconception.
            explicit_failure = (
                latest_signal in {"negative", "empty"}
                or explicit_confusion
                or active_misconception
            )
            route_is_transfer = bool(session.teaching_route and session.teaching_route.current_step().kind == "transfer")
            if self.settings.get("simple_teaching_mode", False) and route_is_transfer:
                forced_type = "transfer"
            elif session.state.average_mastery() >= float(self.settings.get("mastery_threshold", 0.8)):
                forced_type = "transfer"
            elif (
                session.state.no_progress_rounds >= correction_after
                and explicit_failure
                and active_misconception
            ):
                forced_type = "correction"
            elif session.state.no_progress_rounds >= scaffold_after and explicit_failure:
                forced_type = "scaffold"
            elif explicit_failure:
                forced_type = "diagnostic"

        if forced_type:
            primary = self.library.get(GENERIC_SKILLS[forced_type])
            support = subject_skill
            reason = self._generic_reason(forced_type, session.state)
            decision_mode = "deterministic_guard"
            policy_rule = {
                "diagnostic": "student_explicitly_confused_or_llm_misconception",
                "scaffold": f"no_progress_rounds >= {scaffold_after}",
                "correction": f"no_progress_rounds >= {correction_after}",
                "transfer": "average_mastery >= threshold",
            }[forced_type]
            if prompt_injection:
                reason = "检测到学生输入包含提示注入式元指令；不执行该指令，回到教学目标并索取学习证据。"
                decision_mode = "security_guard"
                policy_rule = "prompt_injection_guard"
        else:
            assert subject_skill is not None
            primary = subject_skill
            support = self.library.get(GENERIC_SKILLS["diagnostic"]) if initial else None
            reason = (
                f"“{primary.name}”与课程、目标和当前关注点匹配；"
                + ("辅以诊断提问确认起点。" if support else "继续沿用学科 Skill 推进。")
            )
            decision_mode = selection_mode
            policy_rule = "course_goal_trigger_precondition_ranking"

        previous = session.turns[-1].selected_skill_id if session.turns else ""
        switch_reason = ""
        if previous and previous != primary.skill_id:
            switch_reason = f"学生状态变化，Skill 从 {previous} 切换为 {primary.skill_id}。"
        # The visible teaching action follows the strategy role when one is
        # attached to a content Skill. Previously the subject Skill always
        # won here, so a diagnostic strategy was displayed as "学科讲解".
        strategy_skill = (
            primary
            if primary.skill_type != "subject"
            else support
            if support and support.skill_type != "subject"
            else None
        )
        action_type = strategy_skill.skill_type if strategy_skill else "subject_instruction"
        generation_message = (
            "学生提交了与学习任务无关的元指令；请简短提醒其回到当前知识点，并提出一个诊断问题。"
            if prompt_injection
            else student_message
        )
        generation = self.response_role.execute(session, primary, support, action_type, generation_message)
        generation = self._apply_simple_teaching_flow(session, generation)
        expected = primary.student_signals[0] if primary.student_signals else "学生能解释下一步推理"
        previous = session.turns[-1].selected_skill_id if session.turns else ""
        skill_plan = self._skill_plan(session, primary, support, [skill.skill_id for skill in ranked], previous)
        if session.turns:
            changes: list[str] = []
            if skill_plan.content_switch:
                previous_content = session.turns[-1].skill_plan.content_skill_id if session.turns[-1].skill_plan else "通用"
                changes.append(f"内容 Skill：{previous_content} → {skill_plan.content_skill_id}")
            if skill_plan.strategy_switch:
                previous_strategy = session.turns[-1].skill_plan.strategy_skill_id if session.turns[-1].skill_plan else "未指定"
                changes.append(f"教学策略：{previous_strategy} → {skill_plan.strategy_skill_id}")
            if changes:
                switch_reason = "学生状态变化，教学方案切换（" + "；".join(changes) + "）。"
            elif previous != primary.skill_id:
                switch_reason = (
                    f"执行入口从 {previous} → {primary.skill_id}；内容 Skill 与教学策略角色保持不变。"
                )
        phase = self._phase_for_decision(
            AgentDecision(
                primary_skill_id=primary.skill_id,
                support_skill_id=support.skill_id if support else None,
                action_type=action_type,
                selection_reason=reason,
                teacher_message=generation.message,
            )
        )
        return AgentDecision(
            primary_skill_id=primary.skill_id,
            support_skill_id=support.skill_id if support else None,
            selection_reason=reason,
            action_type=action_type,
            teacher_message=generation.message,
            expected_signal=expected,
            switch_reason=switch_reason,
            decision_mode=decision_mode,
            candidate_skill_ids=[skill.skill_id for skill in ranked],
            policy_rule=policy_rule,
            candidate_audit=candidate_audit,
            micro_step=generation.micro_step,
            teacher_review=generation.review,
            generation_audit=generation.audit or {},
            phase=phase,
            skill_plan=skill_plan,
            question_contract=self._question_contract(generation.micro_step),
            fallback_reason=(
                "检测到提示注入式输入，已隔离原文本并使用安全诊断回复"
                if prompt_injection
                else "无学科 Skill 通过硬过滤，回退到通用诊断" if no_subject_match else generation.fallback_reason
            ),
        )

    def _apply_simple_teaching_flow(
        self,
        session: TeachingSession,
        generation: MessageGeneration,
    ) -> MessageGeneration:
        """Keep the live lesson focused on one small, visible progression.

        The existing generation and audit pipeline remains available for
        compatibility, but it used to turn every repeated target into another
        evidence request. In live teaching that produced a polished loop
        without a change in the learner's cognitive task. This small policy
        layer makes the route authoritative: a completed route step advances,
        and the transfer step always asks for a genuinely new situation.
        """
        if not self.settings.get("simple_teaching_mode", False) or generation.micro_step is None:
            return generation
        route = session.teaching_route
        if route is None or not route.steps:
            return generation

        current = route.current_step()
        step = generation.micro_step.model_copy(deep=True)
        changed = False
        if current.kind == "transfer":
            focus = current.knowledge_point or step.focus or session.goal.topic
            step.focus = focus
            step.requested_target = f"请换一个不同于刚才的新情境，说明“{focus}”如何表现？"
            step.step_index = max(step.step_index, 1)
            generation.message = self._natural_message(session, step)
            changed = True
        elif session.turns and session.turns[-1].micro_step is not None:
            previous = session.turns[-1].micro_step
            latest = session.state.evidence[-1] if session.state.evidence else None
            same_focus = self._focus_terms_match(previous.focus, current.knowledge_point)
            positive = bool(
                latest
                and latest.signal_type in {"positive", "transfer"}
                and latest.evidence_level in {"correct", "explained", "transfer"}
            )
            if same_focus and positive and self._repeats_previous_target(previous, step):
                # One follow-up can deepen the current step; after that, move
                # to an application/comparison task instead of collecting more
                # synonyms for the same explanation.
                step.focus = current.knowledge_point
                step.requested_target = (
                    f"请说明如果当前情境中的关键条件改变，关于“{current.knowledge_point}”的判断会怎样变化？"
                )
                step.step_index = max(step.step_index, previous.step_index + 1)
                generation.message = self._natural_message(session, step)
                changed = True

        if not changed:
            return generation
        audit = dict(generation.audit or {})
        audit["simple_teaching_flow"] = "route_step_or_transfer"
        audit["simple_teaching_flow_changed_output"] = True
        generation.audit = audit
        generation.micro_step = step
        generation.review = generation.review
        return generation

    def _llm_select(
        self,
        candidates: list[TeachingSkill],
        session: TeachingSession,
        student_message: str,
        candidate_audit: list[dict[str, Any]] | None = None,
    ) -> tuple[TeachingSkill | None, str]:
        if not candidates:
            return None, "rule_fallback"
        if not self.llm.available:
            return candidates[0], "rule_fallback"
        audit_by_id = {str(item.get("skill_id")): item for item in candidate_audit or []}
        scores = sorted(
            (float(audit_by_id.get(skill.skill_id, {}).get("score", 0.0)) for skill in candidates),
            reverse=True,
        )
        selector_margin = float(self.settings.get("semantic_selector_margin", 0.05))
        if len(candidates) == 1 or len(scores) >= 2 and scores[0] - scores[1] >= selector_margin:
            return candidates[0], "rule_margin_selection"
        try:
            data = self.llm.structured(
                "你是教学 Skill 选择器。只能从候选列表中选择，不生成教学内容。",
                (
                    f"目标：{session.goal.model_dump_json()}\n状态：{session.state.model_dump_json()}\n"
                    f"历史对话：{self._history_summary(session)}\n学生回答：{student_message}\n候选：\n"
                    + "\n".join(skill.prompt_summary() for skill in candidates)
                ),
                '{"skill_id":"候选ID","reason":"选择理由"}',
                temperature=0.0,
            )
            chosen = str(data.get("skill_id", ""))
            selected = next((skill for skill in candidates if skill.skill_id == chosen), None)
            if selected is None:
                return candidates[0], "candidate_constraint_fallback"
            return selected, "llm_semantic_selection"
        except (LLMUnavailableError, ValueError, TypeError):
            return candidates[0], "rule_fallback"

    def _generate_teacher_message(
        self,
        session: TeachingSession,
        primary: TeachingSkill,
        support: TeachingSkill | None,
        action_type: str,
        student_message: str,
    ) -> MessageGeneration:
        previous_step = session.turns[-1].micro_step if session.turns else None
        transition_allowed = self._transition_allowed(session, student_message)
        # A positive answer does not automatically mean "start a new topic".
        # If the declared route is still on the same knowledge point, keep the
        # current scene and representation and move only one rung on the
        # evidence ladder.  The route itself is the authority for topic
        # changes; this prevents a short answer such as “有一个元素” from
        # causing the LLM to regenerate the opening question.
        same_route_step = bool(
            previous_step
            and session.teaching_route
            and self._focus_terms_match(
                session.teaching_route.current_step().knowledge_point,
                previous_step.focus,
            )
        )
        context_locked = bool(previous_step and (not transition_allowed or same_route_step))
        fallback_code = "llm_unavailable"
        fallback_reason = "LLM 不可用，已使用通用单步回退"
        llm_error_stage = ""
        self._teacher_draft_retry = False
        if self.llm.available:
            try:
                llm_error_stage = "teacher_draft"
                draft: TeacherDraft | None = None
                draft_error: ValueError | TypeError | None = None
                retry_limit = max(0, int(self.settings.get("structured_output_retries", 2)))
                for attempt in range(retry_limit + 1):
                    try:
                        draft = self._llm_teacher_draft(
                            session,
                            primary,
                            support,
                            action_type,
                            student_message,
                            previous_step,
                            context_locked,
                        )
                        break
                    except (ValueError, TypeError) as exc:
                        draft_error = exc
                        if attempt < retry_limit:
                            self._teacher_draft_retry = True
                            llm_error_stage = "teacher_draft_retry"
                if draft is None:
                    if draft_error is not None:
                        raise draft_error
                    raise ValueError("教师草稿为空")
                llm_error_stage = "deterministic_review"
                deterministic_review = self.review_role.execute(
                    session,
                    student_message,
                    draft,
                    previous_step,
                    context_locked,
                    ask_llm=False,
                )
                # A clean deterministic contract does not need a second LLM
                # judge. Only ambiguous/high-risk drafts consume the semantic
                # review budget; this keeps a normal turn at draft + state
                # diagnosis rather than stacking redundant reviewers.
                if deterministic_review.valid and (
                    context_locked or bool(self.settings.get("fast_demo_mode", False))
                ):
                    review = deterministic_review
                else:
                    llm_error_stage = "semantic_review"
                    review = self.review_role.execute(
                        session,
                        student_message,
                        draft,
                        previous_step,
                        context_locked,
                    )
                audit: dict[str, Any] = {
                    **role_pipeline_audit(),
                    "context_locked": context_locked,
                    "response_preference": session.profile.response_preference,
                    "response_mode": draft.micro_step.response_mode,
                    "option_count": len(draft.micro_step.options),
                    "draft_question_count": draft.question_count,
                    "introduced_symbols": draft.introduced_symbols,
                    "introduced_values": draft.introduced_values,
                    "review_issues": review.issues,
                    "teacher_draft_retry": self._teacher_draft_retry,
                    "fast_demo_mode": bool(self.settings.get("fast_demo_mode", False)),
                }
                if not review.valid:
                    if bool(self.settings.get("fast_demo_mode", False)):
                        # In live-demo mode an invalid draft is never shown to
                        # the learner. The deterministic fallback is safe and
                        # avoids stacking semantic review + repair calls on a
                        # slow provider. Full evaluation keeps the complete
                        # repair chain below.
                        fallback_step = self._fallback_step_from_draft(
                            session,
                            primary,
                            action_type,
                            draft,
                            previous_step,
                            context_locked,
                        )
                        fallback_message = self._offline_message(
                            session,
                            primary,
                            action_type,
                            fallback_step,
                        )
                        audit["fallback"] = "fast_demo_deterministic_fallback"
                        fallback_review = self._fallback_review(
                            review,
                            "快速演示模式：单步复核未通过，已使用安全回退",
                            fallback_message,
                        )
                        return MessageGeneration(
                            message=fallback_message,
                            micro_step=fallback_step,
                            review=fallback_review,
                            audit=audit,
                            fallback_reason="快速演示模式：教师草稿未通过确定性单步复核，已安全回退",
                        )
                    llm_error_stage = "teacher_draft_repair"
                    repaired = self._repair_teacher_draft(
                        session,
                        student_message,
                        draft,
                        review,
                        previous_step,
                        context_locked,
                    )
                    if repaired is not None:
                        llm_error_stage = "repaired_draft_review"
                        repaired_review = self.review_role.execute(
                            session,
                            student_message,
                            repaired,
                            previous_step,
                            context_locked,
                        )
                        if repaired_review.valid:
                            repeated_target = bool(
                                previous_step
                                and self._repeats_previous_target(previous_step, repaired.micro_step)
                            )
                            if repeated_target:
                                safe_step = self._advance_repeated_target_for_student(
                                    session,
                                    repaired.micro_step,
                                    student_message,
                                )
                                audit["fallback"] = "repeated_target_guard"
                                safe_message = self._offline_message(session, primary, action_type, safe_step)
                                return MessageGeneration(
                                    message=safe_message,
                                    micro_step=safe_step,
                                    review=self._fallback_review(
                                        repaired_review,
                                        "已拦截重复微步骤，改为同一知识点的判断依据",
                                        safe_message,
                                    ),
                                    audit=audit,
                                    fallback_reason="已拦截重复微步骤，改为同一知识点的判断依据",
                                )
                            audit["repair_applied"] = True
                            audit["review_issues"] = repaired_review.issues
                            return MessageGeneration(
                                message=self._final_message(
                                    repaired.teacher_message,
                                    session,
                                    primary,
                                    action_type,
                                    repaired.micro_step,
                                ),
                                micro_step=repaired.micro_step,
                                review=repaired_review,
                                audit=audit,
                            )
                    fallback_step = self._fallback_step_from_draft(
                        session,
                        primary,
                        action_type,
                        draft,
                        previous_step,
                        context_locked,
                    )
                    completed_revisit = self._completed_focus_revisit_reason(session, draft.micro_step)
                    fallback_step_revisit = self._completed_focus_revisit_reason(session, fallback_step)
                    if completed_revisit or fallback_step_revisit:
                        fallback_step = self._advance_completed_definition(session, fallback_step)
                    fallback_guarded_revisit = (
                        bool(completed_revisit)
                        or bool(fallback_step_revisit)
                        or "当前情境中的判断依据" in fallback_step.requested_target
                    )
                    audit["fallback"] = (
                        "completed_focus_revisit"
                        if completed_revisit or fallback_guarded_revisit
                        else "review_repair_failed"
                    )
                    fallback_message = self._offline_message(session, primary, action_type, fallback_step)
                    fallback_review = self._fallback_review(
                        review,
                        "结构化复核未通过，已使用单步通用回退",
                        fallback_message,
                    )
                    return MessageGeneration(
                        message=fallback_message,
                        micro_step=fallback_step,
                        review=fallback_review,
                        audit=audit,
                        fallback_reason=(
                            "已拦截已完成知识点的重复定义，改为当前情境中的依据说明"
                            if completed_revisit or fallback_guarded_revisit
                            else "教师话语未通过单步结构化复核，已回退到通用单步提问"
                        ),
                    )
                repeated_target = bool(
                    previous_step and self._repeats_previous_target(previous_step, draft.micro_step)
                )
                if repeated_target:
                    safe_step = self._advance_repeated_target_for_student(
                        session,
                        draft.micro_step,
                        student_message,
                    )
                    audit["fallback"] = "repeated_target_guard"
                    safe_message = self._offline_message(session, primary, action_type, safe_step)
                    return MessageGeneration(
                        message=safe_message,
                        micro_step=safe_step,
                        review=self._fallback_review(
                            review,
                            "已拦截重复微步骤，改为同一知识点的判断依据",
                            safe_message,
                        ),
                        audit=audit,
                        fallback_reason="已拦截重复微步骤，改为同一知识点的判断依据",
                    )
                completed_revisit = self._completed_focus_revisit_reason(session, draft.micro_step)
                if completed_revisit:
                    safe_step = self._advance_completed_definition(session, draft.micro_step)
                    audit["fallback"] = "completed_focus_revisit"
                    safe_message = self._offline_message(session, primary, action_type, safe_step)
                    return MessageGeneration(
                        message=safe_message,
                        micro_step=safe_step,
                        review=self._fallback_review(
                            review,
                            "已拦截已完成知识点的重复定义，改为当前情境中的依据说明",
                            safe_message,
                        ),
                        audit=audit,
                        fallback_reason="已拦截已完成知识点的重复定义，改为当前情境中的依据说明",
                    )
                return MessageGeneration(
                    message=self._final_message(
                        draft.teacher_message,
                        session,
                        primary,
                        action_type,
                        draft.micro_step,
                    ),
                    micro_step=draft.micro_step,
                    review=review,
                    audit=audit,
                )
            except LLMUnavailableError:
                fallback_code = "llm_unavailable"
                fallback_reason = "真实 LLM 请求失败，已使用通用单步回退"
            except ValueError:
                fallback_code = "llm_invalid_structured_output"
                fallback_reason = "真实 LLM 返回的结构化输出无法解析，已使用通用单步回退"
            except TypeError:
                fallback_code = "llm_schema_error"
                fallback_reason = "真实 LLM 输出字段不符合结构约束，已使用通用单步回退"
        fallback_step = self._fallback_micro_step(session, primary, action_type, previous_step)
        if context_locked and previous_step:
            # An API/schema failure must not make the learner answer the exact
            # opening question again.  Keep the same context and representation
            # but advance one generic evidence rung, just like the normal
            # repeated-target guard does.
            fallback_step = self._advance_repeated_target_for_student(
                session,
                previous_step,
                student_message,
            )
        fallback_revisit = self._completed_focus_revisit_reason(session, fallback_step)
        if fallback_revisit:
            fallback_step = self._advance_completed_definition(session, fallback_step)
        return MessageGeneration(
            message=self._offline_message(session, primary, action_type, fallback_step),
            micro_step=fallback_step,
            review=None,
            audit={
                **role_pipeline_audit(),
                "context_locked": context_locked,
                "response_preference": session.profile.response_preference,
                "response_mode": fallback_step.response_mode,
                "option_count": len(fallback_step.options),
                "fallback": "completed_focus_revisit" if fallback_revisit else fallback_code,
                "llm_error_stage": llm_error_stage or None,
                "teacher_draft_retry": self._teacher_draft_retry,
            },
            fallback_reason=(
                "已拦截已完成知识点的重复定义，改为当前情境中的依据说明"
                if fallback_revisit
                else fallback_reason
            ),
        )

    def _llm_teacher_draft(
        self,
        session: TeachingSession,
        primary: TeachingSkill,
        support: TeachingSkill | None,
        action_type: str,
        student_message: str,
        previous_step: TeachingMicroStep | None,
        context_locked: bool,
    ) -> TeacherDraft:
        previous_json = previous_step.model_dump_json() if previous_step else "无"
        continuity = (
            "必须原样沿用上一小步的 context 和 representation，只做最小追问。"
            if context_locked
            else "可以规划下一小步，但仍只能使用一个情境；不要提前讲后续分支。"
        )
        route_step = session.teaching_route.current_step() if session.teaching_route else None
        route_contract = route_step.model_dump_json() if route_step else "未建立教学路线"
        draft = TeacherDraft.model_validate(
            self.llm.structured(
                (
                    "你是自适应教学 Agent 的微步骤规划器和教师话语生成器。"
                    "你必须先规划一个且只有一个可验证的教学微步骤，再把它表达成学生能理解的教师话语。"
                    "不得写完整课程，不得模拟学生，不得直接泄露最终答案。"
                    "一个微步骤只能有一个 focus、一个 context、一个 known_fact 和一个 requested_target。"
                    "不得在同一轮同时讲多个目标值、多个数组、多个区间、多个假设分支或多种表示法。"
                    "首轮或需要建立情境时，优先给出一个学生能直接指认的具体对象、数值、公式、现象或代码片段；"
                    "不要只说‘在这个主题中’。教师话语不能把正确结论写进问题里再让学生解释。"
                    "如果要学生预测或描述现象，context 和 known_fact 只能给条件，不能提前写出该现象或结论。"
                    "学生可见的话语不得出现‘本轮只处理’、‘当前还没有足够的学生回答证据’、"
                    "‘专业决策证据’、‘答辩演示输入建议’等内部审计用语；这些只属于教师视图。"
                    "必须核对数字、比较符号、下标和结论之间的事实一致性。"
                    "教师话语必须只提出一个最终可回答的问题。"
                    "当前教学路线步骤是课程边界：focus 必须是该步骤的 knowledge_point，"
                    "内容只能服务于该步骤的 learning_target；不得自行跳到其他知识点。"
                    "根据教学阶段选择 response_mode：初次诊断和迁移验证优先 open；"
                    "当学生卡住、需要降低回答门槛或隔离一个误解时，可以选择 single_choice、fill_blank 或 numeric。"
                    "single_choice 必须给 2 到 4 个互斥选项，选项只提供不同思路，不标注正确答案；"
                    "选项必须是当前知识问题的不同判断或解释，不能只是‘我会不会/我是否需要提示’的信心自评；"
                    "fill_blank 和 numeric 不得生成 options。任何模式都不能把答案写入教师话语。"
                ),
                (
                    f"教学目标：{session.goal.model_dump_json()}\n"
                    f"学生画像：{session.profile.model_dump_json()}\n"
                    f"学生状态：{session.state.model_dump_json()}\n"
                    f"历史对话：{self._history_summary(session)}\n"
                    f"学生刚才回答：{student_message or '尚未回答'}\n"
                    f"当前 Skill：{primary.generation_summary()}\n"
                    f"辅助 Skill：{support.generation_summary() if support else '无'}\n"
                    f"上一轮微步骤：{previous_json}\n"
                    f"当前教学路线步骤：{route_contract}\n"
                    f"上下文门控：{continuity}\n"
                    f"学生指定的回答方式：{session.profile.response_preference}。如果不是 auto，必须严格使用该方式。\n"
                    f"当前唯一关注点（只用于规划，不要原样展示）：{session.state.next_focus}\n"
                    f"当前动作类型：{action_type}\n"
                    "请输出一个结构化微步骤和对应的面向学生的话语。"
                ),
                (
                    '{"micro_step":{"focus":"唯一关注点","context":"唯一情境",'
                    '"known_fact":"已给事实","requested_target":"学生只需回答的一个目标",'
                    '"representation":"当前表示法","expected_signal":"期望信号","step_index":1,'
                    '"response_mode":"open","options":[],"input_hint":"请用自己的话回答"},'
                    '"teacher_message":"只包含一个情境和一个最终问题的教师话语",'
                    '"introduced_symbols":[],"introduced_values":[],"question_count":1}'
                ),
                temperature=float(self.settings.get("temperature", 0.2)),
            )
        )
        return self._normalize_draft(session, draft, previous_step, context_locked)

    @staticmethod
    def _focus_label(session: TeachingSession) -> str:
        if session.teaching_route:
            return session.teaching_route.current_step().knowledge_point
        raw = session.state.next_focus.strip()
        matches = [point for point in session.goal.knowledge_points if point and point in raw]
        if len(matches) > 1 and session.turns and session.turns[-1].micro_step:
            previous_focus = session.turns[-1].micro_step.focus
            alternatives = [point for point in matches if point not in previous_focus]
            if alternatives:
                return alternatives[0]
        if matches:
            return matches[0]
        if session.turns and session.turns[-1].micro_step and session.state.evidence:
            latest = session.state.evidence[-1]
            previous_focus = session.turns[-1].micro_step.focus
            if (
                latest.evidence_level in {"correct", "explained", "transfer"}
                and latest.knowledge_point in session.goal.knowledge_points
                and not HybridTeachingAgent._focus_terms_match(latest.knowledge_point, previous_focus)
            ):
                return latest.knowledge_point
        return raw or (session.goal.knowledge_points[0] if session.goal.knowledge_points else session.goal.topic)

    @classmethod
    def _normalize_draft(
        cls,
        session: TeachingSession,
        draft: TeacherDraft,
        previous_step: TeachingMicroStep | None,
        context_locked: bool,
    ) -> TeacherDraft:
        normalized = draft.model_copy(deep=True)
        if context_locked and previous_step:
            # A locked step is an explicit deterministic contract. The model
            # may still improve the wording, but it cannot silently replace
            # the context or formalism before the learner has completed it.
            normalized.micro_step = previous_step.model_copy(deep=True)
        else:
            normalized.micro_step.focus = cls._focus_label(session)
        preference = session.profile.response_preference
        if preference != "auto":
            normalized.micro_step.response_mode = preference
            if preference != "single_choice":
                normalized.micro_step.options = []
        normalized.question_count = normalized.teacher_message.count("？") + normalized.teacher_message.count("?")
        return normalized

    def _review_teacher_draft(
        self,
        session: TeachingSession,
        student_message: str,
        draft: TeacherDraft,
        previous_step: TeachingMicroStep | None,
        context_locked: bool,
        *,
        ask_llm: bool = True,
    ) -> TeacherReview:
        message = draft.teacher_message.strip()
        question_count = message.count("？") + message.count("?")
        issues: list[str] = []
        if not message:
            issues.append("教师话语为空")
        if question_count != 1:
            issues.append("教师话语必须只有一个最终问题")
        if draft.question_count != question_count:
            issues.append("结构化 question_count 与教师话语不一致")
        required = {
            "focus": draft.micro_step.focus,
            "context": draft.micro_step.context,
            "known_fact": draft.micro_step.known_fact,
            "requested_target": draft.micro_step.requested_target,
            "representation": draft.micro_step.representation,
            "expected_signal": draft.micro_step.expected_signal,
        }
        if any(not str(value).strip() for value in required.values()):
            issues.append("微步骤字段不完整")
        response_mode_valid = True
        options_valid = True
        response_mode = draft.micro_step.response_mode
        preferred_mode = session.profile.response_preference
        if preferred_mode != "auto" and response_mode != preferred_mode:
            response_mode_valid = False
            issues.append(f"学生已指定回答方式为 {preferred_mode}，当前草稿却是 {response_mode}")
        if response_mode == "single_choice":
            options = draft.micro_step.options
            option_ids = [option.option_id.strip() for option in options]
            option_texts = [option.text.strip() for option in options]
            if not 2 <= len(options) <= 4:
                options_valid = False
                issues.append("单选题必须有 2 到 4 个选项")
            if len(set(option_ids)) != len(option_ids) or any(not value for value in option_ids):
                options_valid = False
                issues.append("单选题选项编号必须唯一且非空")
            if len(set(option_texts)) != len(option_texts) or any(not value for value in option_texts):
                options_valid = False
                issues.append("单选题选项文本必须唯一且非空")
            confidence_cues = ("我能说明", "我还不能说明", "我会", "我不会", "我不确定", "需要提示")
            if len(options) >= 2 and all(any(cue in text for cue in confidence_cues) for text in option_texts):
                options_valid = False
                issues.append("单选题选项变成了信心自评，不是当前知识问题的判断")
        elif draft.micro_step.options:
            response_mode_valid = False
            options_valid = False
            issues.append(f"{response_mode} 模式不应携带选择题选项")
        branch_cues = (
            "再换一种",
            "另一种情况",
            "另外一种",
            "同时考虑",
            "分别讨论",
            "再看一个",
            "接着看另一个",
            "有两种情况",
            "有三种情况",
        )
        one_context = not any(cue in message for cue in branch_cues)
        if not one_context:
            issues.append("同一轮出现多个情境或分支")
        representation_text = "\n".join(
            [
                message,
                draft.micro_step.context,
                draft.micro_step.requested_target,
                draft.micro_step.representation,
            ]
        )
        if self._contains_multiple_representations(representation_text):
            one_context = False
            issues.append("同一轮同时引入多种区间表示法")
        if self._contains_multiple_requests(message) or self._contains_multiple_requests(draft.micro_step.requested_target):
            issues.append("同一轮包含多个子问题或教学任务")
        focus_consistent = not self._question_focus_mismatch(session, draft)
        if not focus_consistent:
            issues.append("教师最终问题与持久化微步骤 focus 不一致")
        if self._contains_internal_task_language(message) or self._contains_internal_task_language(
            draft.micro_step.requested_target
        ):
            issues.append("教师话语泄露了内部任务描述或包含无效问句标点")
        if self._contains_multiple_value_scenarios(
            "\n".join([message, draft.micro_step.context, draft.micro_step.known_fact])
        ):
            issues.append("同一轮重新定义了变量或引入了第二组数值情境")
        if self._repeats_previous_question(session, message):
            issues.append("教师问题与上一轮高度重复，未形成新的教学推进")
        if (
            not context_locked
            and previous_step is not None
            and self._repeats_previous_target(previous_step, draft.micro_step)
        ):
            issues.append("本轮教学目标与上一轮重复，未形成新的微步骤")
        completed_revisit = self._completed_focus_revisit_reason(session, draft.micro_step)
        if completed_revisit:
            issues.append(completed_revisit)
        if self._focus_stage_mismatch(draft.micro_step.focus, draft.micro_step.requested_target):
            issues.append("当前教学焦点仍是概念理解，问题却提前进入操作或推导")
        same_context = True
        if context_locked and previous_step:
            same_context = (
                draft.micro_step.context.strip() == previous_step.context.strip()
                and draft.micro_step.representation.strip() == previous_step.representation.strip()
            )
            if not same_context:
                issues.append("学生尚未完成当前小步，不得更换情境或表示法")
            if session.turns and self._has_conflicting_comparison(
                session.turns[-1].teacher_message,
                message,
            ):
                issues.append("当前小步尚未完成，教师话语引入了互斥判断分支")
        answer_leakage = self._contains_answer_leakage(message) or self._contains_answer_leakage(
            draft.micro_step.requested_target
        )
        if answer_leakage:
            issues.append("教师话语疑似直接泄露答案")

        deterministic_valid = not issues
        if not ask_llm:
            return TeacherReview(
                valid=deterministic_valid,
                one_step=deterministic_valid,
                one_context=one_context,
                one_question=question_count == 1,
                fact_consistent=focus_consistent and deterministic_valid,
                same_context=same_context,
                answer_leakage=answer_leakage,
                response_mode_valid=response_mode_valid,
                options_valid=options_valid,
                issues=issues,
            )

        try:
            result = TeacherReview.model_validate(
                self.llm.structured(
                    (
                        "你是教学输出质量复核器。你必须检查教师话语是否只完成一个教学微步骤。"
                        "重点检查：情境是否唯一、事实是否自洽、是否保持上一轮情境和表示法、"
                        "是否只有一个问题、是否引入了第二个目标或分支、是否直接泄露答案。"
                        "同时检查 response_mode 是否适合当前教学阶段：open 用于解释和迁移，"
                        "single_choice 只能有 2 到 4 个互斥选项，fill_blank/numeric 不应有选项。"
                     "选项不能暗示哪个是正确答案，也不能把选项设计成多个问题。"
                     "如果学生指定了非 auto 的回答方式，必须严格检查当前草稿是否遵守该方式。"
                         "必须比较当前微步骤 focus 与 requested_target 的语义是否一致；"
                         "如果 focus 只要求理解一个概念，而 requested_target 实际要求另一个规则、计算或后续步骤，valid 必须为 false。"
                     "不能因为整段只有一个问号，就把多个情境判为单步。"
                        "如果教师话语先给出‘应该使用某个条件/某个值，而不是另一个’，再询问原因，也属于答案泄露或多任务。"
                        "如果 requested_target 要学生判断、预测或描述一个结果，而 context、known_fact 或教师话语"
                        "已经陈述了这个结果，也必须判为答案泄露。"
                    ),
                    (
                        f"教学目标：{session.goal.model_dump_json()}\n"
                        f"学生刚才回答：{student_message or '尚未回答'}\n"
                        f"上一轮微步骤：{previous_step.model_dump_json() if previous_step else '无'}\n"
                        f"当前微步骤：{draft.micro_step.model_dump_json()}\n"
                        f"学生指定回答方式：{preferred_mode}\n"
                         f"教师话语：{message}\n"
                         "请先检查 focus、context、known_fact、requested_target 四者是否形成同一个可验证微步骤。"
                         "如果存在事实矛盾、focus 与 requested_target 语义错位、多个情境、多个目标、无理由切换表示法或答案泄露，valid 必须为 false。"
                        "revised_message 只在可以修复时填写，并且仍然只能有一个问题。"
                    ),
                    (
                        '{"valid":true,"one_step":true,"one_context":true,"one_question":true,'
                        '"fact_consistent":true,"same_context":true,"answer_leakage":false,'
                        '"response_mode_valid":true,"options_valid":true,'
                        '"issues":[],"revised_message":""}'
                    ),
                    temperature=0.0,
                )
            )
        except (LLMUnavailableError, ValueError, TypeError):
            return TeacherReview(
                valid=False,
                one_step=False,
                one_context=one_context,
                one_question=question_count == 1,
                fact_consistent=False,
                same_context=same_context,
                answer_leakage=answer_leakage,
                response_mode_valid=response_mode_valid,
                options_valid=options_valid,
                issues=issues + ["LLM 结构化复核不可用"],
            )
        merged_issues = list(dict.fromkeys(issues + result.issues))
        result.issues = merged_issues
        result.one_context = result.one_context and one_context
        result.one_question = result.one_question and question_count == 1
        result.fact_consistent = result.fact_consistent and focus_consistent
        result.same_context = result.same_context and same_context
        result.answer_leakage = result.answer_leakage or answer_leakage
        result.response_mode_valid = result.response_mode_valid and response_mode_valid
        result.options_valid = result.options_valid and options_valid
        result.valid = (
            result.valid
            and deterministic_valid
            and result.one_step
            and result.one_context
            and result.one_question
            and result.fact_consistent
            and result.same_context
            and result.response_mode_valid
            and result.options_valid
            and not result.answer_leakage
        )
        return result

    def _repair_teacher_draft(
        self,
        session: TeachingSession,
        student_message: str,
        draft: TeacherDraft,
        review: TeacherReview,
        previous_step: TeachingMicroStep | None,
        context_locked: bool,
    ) -> TeacherDraft | None:
        try:
            repaired = TeacherDraft.model_validate(
                self.llm.structured(
                    (
                        "你是教学话语修复器。只修复当前微步骤，不重新规划课程。"
                        "保留一个 focus、一个 context、一个 known_fact、一个 requested_target，"
                        "教师话语只能有一个最终问题。"
                    ),
                    (
                        f"原始草稿：{draft.model_dump_json()}\n"
                        f"复核问题：{review.model_dump_json()}\n"
                        f"上一轮微步骤：{previous_step.model_dump_json() if previous_step else '无'}\n"
                        f"上下文是否锁定：{context_locked}\n"
                        f"学生回答：{student_message or '尚未回答'}\n"
                        "请修复事实、分支、回答模式或连续性问题；如果上下文锁定，必须沿用原 context 和 representation。"
                    ),
                    (
                        '{"micro_step":{"focus":"唯一关注点","context":"唯一情境",'
                        '"known_fact":"已给事实","requested_target":"学生只需回答的一个目标",'
                        '"representation":"当前表示法","expected_signal":"期望信号","step_index":1,'
                        '"response_mode":"open","options":[],"input_hint":"请用自己的话回答"},'
                        '"teacher_message":"修复后的单步教师话语",'
                        '"introduced_symbols":[],"introduced_values":[],"question_count":1}'
                    ),
                    temperature=0.0,
                )
            )
            return self._normalize_draft(session, repaired, previous_step, context_locked)
        except (LLMUnavailableError, ValueError, TypeError):
            return None

    @staticmethod
    def _contains_answer_leakage(message: str) -> bool:
        direct_answer_cues = ("答案是", "最终答案", "正确选项是", "直接写成", "直接得到", "你只要记住")
        answer_pattern = re.compile(r"(?:所以|因此|故)\s*[A-Za-z\u4e00-\u9fff]+\s*=\s*[-+]?\d")
        prescribed_value = re.compile(
            r"(?:应该|应当|必须|要用|使用)\s*[`A-Za-z\u4e00-\u9fff_]+\s*(?:<=|>=|<|>|=)"
        )
        comparison_conclusion = re.compile(r"(?:应该|应当|必须).*(?:而不是|而非)")
        return (
            any(cue in message for cue in direct_answer_cues)
            or bool(answer_pattern.search(message))
            or bool(prescribed_value.search(message))
            or bool(comparison_conclusion.search(message))
        )

    @staticmethod
    def _contains_internal_task_language(message: str) -> bool:
        """Reject rubric/prompt language that is not addressed to a learner."""
        learner_task_pattern = re.compile(
            r"(?:学生|学习者)(?:需要|应该|应当|要|必须|用自己的话|需要判断)"
        )
        meta_cues = (
            "本轮只请",
            "教学目标是",
            "当前任务是",
            "评估目标是",
            "请学生",
            "答辩演示输入建议",
            "专业决策证据",
        )
        dangling_question = re.search(r"[，,、；;：:]\s*[？?]\s*$", message)
        return bool(
            learner_task_pattern.search(message)
            or any(cue in message for cue in meta_cues)
            or dangling_question
        )

    @staticmethod
    def _contains_multiple_requests(text: str) -> bool:
        cues = (
            "并解释",
            "并说明",
            "并判断",
            "并举例",
            "并指出",
            "并说出",
            "并回答",
            "并给出",
            "并找出",
            "并列出",
            "同时回答",
            "再说明",
            "再判断",
            "据此解释为什么",
            "以及为什么",
            "以及说明",
        )
        if any(cue in text for cue in cues):
            return True
        if re.search(r"(?:以及|同时|另外).{0,40}(?:为什么|如何|怎样|什么|是否|哪一个|多少)", text):
            return True
        # A model can hide two tasks behind one final question mark, for
        # example: "判断合力方向。这个合力产生了什么影响？".  Detect the
        # discourse structure instead of storing any subject answer.
        clauses = [item.strip() for item in re.split(r"[。；;\n]+", text) if item.strip()]
        task_verbs = ("判断", "选择", "写出", "指出", "计算", "求出", "说明", "解释", "描述")
        interrogatives = ("为什么", "如何", "什么", "是否", "哪一个", "多少", "怎样")
        if len(clauses) >= 2:
            for index, clause in enumerate(clauses[:-1]):
                # “这说明你已经理解……” is explanatory feedback, not a
                # second request.  Strip that discourse use before looking
                # for a task verb so a correct answer is not rejected as a
                # multi-question turn.
                task_clause = re.sub(r"(?:这|它|上述|这一点)(?:也|就|便)?说明", "", clause)
                if any(verb in task_clause for verb in task_verbs) and any(
                    marker in later
                    for later in clauses[index + 1 :]
                    for marker in interrogatives
                ):
                    return True
        return False

    @staticmethod
    def _contains_multiple_value_scenarios(text: str) -> bool:
        """Detect generic variable redefinition or a second numeric scenario.

        This intentionally does not know any subject vocabulary.  It only
        checks the structure of assignments and transition cues such as
        ``例如`` or ``换成``.  A first sentence may introduce several values;
        the guard triggers when a named value is redefined or a second value
        is introduced after a scenario transition.
        """
        assignment = re.compile(
            r"(?P<name>[A-Za-z_][A-Za-z0-9_]*(?:\[[^\]]+\])?|[\u4e00-\u9fff]{1,12})"
            r"\s*(?:=|是|为|等于|设为)\s*"
            r"(?P<value>\[[^\]]*\]|[-+]?\d+(?:\.\d+)?|[A-Za-z_][A-Za-z0-9_]*)"
        )
        assignments = [
            (match.group("name").lower(), match.group("value"))
            for match in assignment.finditer(text)
        ]
        values_by_name: dict[str, set[str]] = {}
        for name, value in assignments:
            values_by_name.setdefault(name, set()).add(value)
        if any(len(values) > 1 for values in values_by_name.values()):
            return True

        transition_cues = ("比如", "例如", "假设", "换成", "改为", "再看", "现在换")
        first_assignment_end = min(
            (match.end() for match in assignment.finditer(text)),
            default=-1,
        )
        if first_assignment_end < 0:
            return False
        return any(
            text.find(cue, first_assignment_end) >= 0
            and any(match.start() > text.find(cue, first_assignment_end) for match in assignment.finditer(text))
            for cue in transition_cues
        )

    @staticmethod
    def _focus_stage_mismatch(focus: str, requested_target: str) -> bool:
        """Reject a generic jump from concept recognition to later operations.

        This is a teaching-language constraint, not a subject answer rule.  It
        keeps a concept-definition step from asking for calculation, update,
        proof, or application before the learner has supplied evidence for the
        current concept.  The same guard applies to mathematics, physics,
        programming, or a user-imported Skill.
        """
        concept_cues = ("定义", "概念", "含义", "意义", "是什么", "表示什么", "理解")
        operation_cues = (
            "计算",
            "求出",
            "更新",
            "推导",
            "证明",
            "写出下一步",
            "如何应用",
            "怎么做",
            "下一步",
        )
        normalized_focus = re.sub(r"\s+", "", focus)
        normalized_target = re.sub(r"\s+", "", requested_target)
        return (
            any(cue in normalized_focus for cue in concept_cues)
            and any(cue in normalized_target for cue in operation_cues)
            and not any(cue in normalized_target for cue in ("概念", "定义", "含义", "意义"))
        )

    @staticmethod
    def _repeats_previous_question(session: TeachingSession, message: str) -> bool:
        """Reject duplicate teacher questions without knowing the subject."""
        if not session.turns:
            return False

        def normalize(value: str) -> str:
            return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", value).lower()

        current = normalize(message)
        previous = normalize(session.turns[-1].teacher_message)
        if not current or not previous:
            return False
        return current == previous or SequenceMatcher(None, current, previous).ratio() >= 0.94

    @staticmethod
    def _repeats_previous_target(
        previous_step: TeachingMicroStep,
        current_step: TeachingMicroStep,
    ) -> bool:
        """Detect semantic repetition even when the teacher rephrases it."""
        def normalize(value: str) -> str:
            normalized = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", value).lower()
            # Remove discourse wrappers so that "请用自己的话解释…" and
            # "先回答一个问题：解释…" are compared by their actual target.
            for cue in (
                "请用自己的话说明",
                "请用自己的话解释",
                "请先用自己的话说明",
                "请先用自己的话解释",
                "先回答一个问题",
                "请说明",
                "请解释",
                "请回答",
                "你能说明",
                "你能解释",
                "用自己的话说明",
                "用自己的话解释",
            ):
                normalized = normalized.replace(cue, "")
            return normalized.replace("一下", "").replace("吗", "")

        previous = normalize(previous_step.requested_target)
        current = normalize(current_step.requested_target)
        if not previous or not current:
            return False
        if previous == current or SequenceMatcher(None, previous, current).ratio() >= 0.86:
            return True
        # When the focus is unchanged, a lower similarity threshold catches a
        # genuine rephrasing such as adding/removing the concrete scene while
        # retaining the same cognitive question. This is still language-level
        # repetition detection; it does not know the subject answer.
        return (
            HybridTeachingAgent._focus_terms_match(previous_step.focus, current_step.focus)
            and SequenceMatcher(None, previous, current).ratio() >= 0.74
        )

    @staticmethod
    def _is_definition_only_target(target: str) -> bool:
        """Return whether a question merely asks to redefine a concept.

        This is intentionally a language-level guard.  It does not know any
        subject terminology or answer key; it only distinguishes a definition
        request from application, comparison, explanation, or verification.
        """
        normalized = re.sub(r"\s+", "", target)
        definition_cues = (
            "是什么意思",
            "是什么",
            "定义",
            "含义",
            "意义",
            "概念",
            "用自己的话说明",
            "用自己的话解释",
        )
        progress_cues = (
            "为什么",
            "依据",
            "应用",
            "举例",
            "对比",
            "验证",
            "迁移",
            "新情境",
            "如何",
            "怎么做",
            "更新",
            "判断",
            "保持",
            "下一步",
        )
        return bool(
            normalized
            and any(cue in normalized for cue in definition_cues)
            and not any(cue in normalized for cue in progress_cues)
        )

    @classmethod
    def _completed_focus_revisit_reason(
        cls,
        session: TeachingSession,
        step: TeachingMicroStep,
    ) -> str:
        """Reject a definition-only revisit after positive evidence.

        The state assessor may correctly mark the learner's latest answer as
        positive while still emitting a stale or broad ``next_focus``.  Without
        this gate the teacher can move backwards and ask the definition of an
        already demonstrated point.  A negative answer or active misconception
        is deliberately exempt: in that case re-diagnosing the point is valid.
        """
        if not session.turns or not step.focus.strip() or not cls._is_definition_only_target(step.requested_target):
            return ""
        latest = session.state.evidence[-1] if session.state.evidence else None
        if latest is None or latest.signal_type not in {"positive", "transfer"}:
            return ""
        if latest.evidence_level not in {"correct", "explained", "transfer"}:
            return ""
        if session.state.misconceptions or session.state.misconception_states:
            return ""

        matched_points = [
            point
            for point in session.goal.knowledge_points
            if point and cls._focus_terms_match(point, step.focus)
        ]
        if not matched_points:
            return ""
        strong_levels = {"correct", "explained", "transfer"}
        for point in matched_points:
            knowledge = session.state.knowledge_states.get(point)
            if knowledge and (
                knowledge.last_evidence_level in strong_levels
                or any(item.evidence_level in strong_levels for item in knowledge.evidence)
            ):
                return "已完成知识点再次被问定义，未形成应用或验证推进"
            if any(
                item.knowledge_point == point and item.evidence_level in strong_levels
                for item in session.state.evidence
            ):
                return "已完成知识点再次被问定义，未形成应用或验证推进"
        return ""

    @staticmethod
    def _contains_multiple_representations(text: str) -> bool:
        """Reject two interval forms without encoding a subject answer."""
        normalized = re.sub(r"\s+", "", text).lower()
        forms: set[str] = set()
        for label, key in (
            ("左闭右闭", "closed_closed"),
            ("左闭右开", "closed_open"),
            ("左开右闭", "open_closed"),
            ("左开右开", "open_open"),
        ):
            if label in text:
                forms.add(key)
        if "区间" in text:
            if re.search(r"\[(?:left|low),(?:right|high)\]", normalized):
                forms.add("closed_closed")
            if re.search(r"\[(?:left|low),(?:right|high)\)", normalized):
                forms.add("closed_open")
        return len(forms) > 1

    @classmethod
    def _single_target(cls, target: str, focus: str) -> str:
        cleaned = target.strip()
        if cls._contains_multiple_requests(cleaned):
            clauses = [item.strip() for item in re.split(r"[。；;\n]+", cleaned) if item.strip()]
            if clauses:
                cleaned = clauses[0]
        for cue in (
            "并解释",
            "并说明",
            "并判断",
            "并举例",
            "并指出",
            "并说出",
            "并回答",
            "并给出",
            "并找出",
            "并列出",
            "同时回答",
            "再说明",
            "再判断",
        ):
            if cue in cleaned:
                cleaned = cleaned.split(cue, 1)[0].strip()
                break
        if re.search(r"(?:应该|应当|必须|要用|使用).*(?:<=|>=|<|>|=|而不是|而非)", cleaned):
            return f"说明“{focus}”的一个关键含义"
        return cleaned or f"说明“{focus}”的一个判断"

    @staticmethod
    def _transition_allowed(session: TeachingSession, student_message: str) -> bool:
        if not session.turns:
            return True
        latest = session.state.evidence[-1] if session.state.evidence else None
        if latest is None or latest.student_quote.strip() != student_message.strip():
            return False
        if latest.signal_type not in {"positive", "transfer"}:
            return False
        if latest.evidence_level not in {"partial", "correct", "explained", "transfer"}:
            return False
        if latest.evidence_level == "transfer":
            return True
        previous_step = session.turns[-1].micro_step
        if previous_step is None:
            return True
        strong_evidence = latest.evidence_level in {"correct", "explained"}
        if strong_evidence and HybridTeachingAgent._focus_terms_match(
            latest.knowledge_point,
            previous_step.focus,
        ):
            return True

        # A strong assessor result can explicitly point to the next declared
        # knowledge point even when the natural-language next_focus contains
        # both the old invariant and the new boundary-update phrase.
        next_focus_points = [
            point
            for point in session.goal.knowledge_points
            if point and point in session.state.next_focus
        ]
        if strong_evidence and any(
            not HybridTeachingAgent._focus_terms_match(point, previous_step.focus)
            for point in next_focus_points
        ):
            return True
        if (
            latest.signal_type == "positive"
            and latest.evidence_level in {"partial", "correct", "explained"}
            and HybridTeachingAgent._focus_terms_match(latest.knowledge_point, previous_step.focus)
            and HybridTeachingAgent._answer_references_step(student_message, previous_step)
        ):
            return True
        return latest.evidence_level in {"explained", "transfer"}

    @staticmethod
    def _answer_references_step(student_message: str, step: TeachingMicroStep) -> bool:
        """Detect generic textual evidence that a learner addressed the step.

        This is deliberately domain-neutral.  It compares shared identifiers,
        numbers and CJK bigrams from the learner answer with the current
        micro-step instead of storing subject-specific answer patterns.
        """
        def signatures(text: str) -> set[str]:
            compact = re.sub(r"\s+", "", text.lower())
            result = set(re.findall(r"[a-z_][a-z0-9_]*|\d+(?:\.\d+)?", compact))
            cjk = re.findall(r"[\u4e00-\u9fff]", compact)
            result.update("".join(cjk[index : index + 2]) for index in range(len(cjk) - 1))
            return result

        answer_signatures = signatures(student_message)
        prompt_signatures = signatures(
            " ".join((step.focus, step.context, step.known_fact, step.requested_target))
        )
        return len(answer_signatures & prompt_signatures) >= 2

    @staticmethod
    def _focus_terms_match(left: str, right: str) -> bool:
        def normalize(value: str) -> str:
            return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", value).lower()

        normalized_left = normalize(left)
        normalized_right = normalize(right)
        return bool(
            normalized_left
            and normalized_right
            and (
                normalized_left == normalized_right
                or normalized_left in normalized_right
                or normalized_right in normalized_left
            )
        )

    @classmethod
    def _question_focus_mismatch(cls, session: TeachingSession, draft: TeacherDraft) -> bool:
        """Detect a final question that targets another declared knowledge point."""
        final_question = re.split(r"[。！!]\s*", draft.teacher_message.strip())[-1]
        mentioned_points = [
            point
            for point in session.goal.knowledge_points
            if point and cls._focus_terms_match(point, final_question)
        ]
        if not mentioned_points:
            return False
        return not any(
            cls._focus_terms_match(point, draft.micro_step.focus)
            for point in mentioned_points
        )

    @staticmethod
    def _has_conflicting_comparison(previous_message: str, current_message: str) -> bool:
        """Detect a new opposite comparison while a micro-step is locked.

        This is a domain-neutral language guard: it does not know the answer to
        any subject question. It only prevents replacing ``x > y`` with
        ``x < y`` (or another opposite operator) before the learner has
        completed the current step.
        """
        pattern = re.compile(
            r"(?P<left>[A-Za-z_][A-Za-z0-9_]*(?:\[[^\]]+\])?)\s*"
            r"(?P<operator><=|>=|==|!=|<|>)\s*"
            r"(?P<right>[A-Za-z_][A-Za-z0-9_]*|-?\d+(?:\.\d+)?)"
        )

        def claims(text: str) -> dict[tuple[str, str], set[str]]:
            result: dict[tuple[str, str], set[str]] = {}
            for match in pattern.finditer(text):
                pair = (
                    match.group("left").lower(),
                    match.group("right").lower(),
                )
                result.setdefault(pair, set()).add(match.group("operator"))
            return result

        previous = claims(previous_message)
        current = claims(current_message)
        return any(
            pair in current and any(operator != other for operator in operators for other in current[pair])
            for pair, operators in previous.items()
        )

    @classmethod
    def _fallback_step_from_draft(
        cls,
        session: TeachingSession,
        skill: TeachingSkill,
        action_type: str,
        draft: TeacherDraft,
        previous_step: TeachingMicroStep | None,
        context_locked: bool,
    ) -> TeachingMicroStep:
        if context_locked and previous_step:
            # Staying on the same curriculum target must not mean asking the
            # identical question again. Keep the context and representation,
            # but move one rung down/up the generic evidence ladder.
            return cls._advance_repeated_target(session, previous_step)
        candidate = draft.micro_step
        completed_revisit = cls._completed_focus_revisit_reason(session, candidate)
        if completed_revisit:
            return cls._advance_completed_definition(session, candidate)
        values = (
            candidate.focus,
            candidate.context,
            candidate.known_fact,
            candidate.representation,
            candidate.expected_signal,
        )
        branch_cues = ("再换一种", "另一种情况", "另外一种", "同时考虑", "分别讨论", "再看一个")
        if (
            all(value.strip() for value in values)
            and not any(cue in candidate.context for cue in branch_cues)
            and not cls._contains_multiple_representations("\n".join(values))
            and not cls._question_focus_mismatch(session, draft)
            and not cls._focus_stage_mismatch(candidate.focus, candidate.requested_target)
            and not cls._contains_internal_task_language(candidate.requested_target)
            and not (
                previous_step is not None
                and cls._repeats_previous_target(previous_step, candidate)
            )
        ):
            candidate.requested_target = cls._single_target(candidate.requested_target, candidate.focus)
            candidate = cls._apply_response_preference(session, candidate)
            if session.profile.response_preference == "auto" and not cls._response_mode_shape_valid(candidate):
                candidate.response_mode = "open"
                candidate.options = []
                candidate.input_hint = "请用自己的话回答"
            elif candidate.response_mode == "single_choice" and not cls._response_mode_shape_valid(candidate):
                candidate.options = []
                candidate.requested_target = cls._choice_generation_wait_target(candidate.focus)
                candidate.input_hint = "请点击“重新生成知识选项”"
            return candidate
        fallback = cls._fallback_micro_step(
            session,
            skill,
            action_type,
            previous_step if context_locked else None,
        )
        if previous_step is not None and cls._repeats_previous_target(previous_step, fallback):
            fallback = cls._advance_repeated_target(session, fallback)
        # The generic fallback can itself choose the stale focus. Apply the
        # same guard after fallback construction, not only to the LLM draft.
        if cls._completed_focus_revisit_reason(session, fallback):
            return cls._advance_completed_definition(session, fallback)
        return fallback

    @classmethod
    def _advance_completed_definition(
        cls,
        session: TeachingSession,
        step: TeachingMicroStep,
    ) -> TeachingMicroStep:
        """Turn a stale definition fallback into a generic evidence step."""
        progressed = step.model_copy(deep=True)
        progressed.requested_target = f"请说明“{progressed.focus}”在当前情境中的判断依据"
        if (
            session.turns
            and session.turns[-1].micro_step is not None
            and cls._repeats_previous_target(session.turns[-1].micro_step, progressed)
        ):
            return cls._advance_repeated_target(session, progressed)
        progressed = cls._apply_response_preference(session, progressed)
        if session.profile.response_preference == "single_choice" and not cls._response_mode_shape_valid(progressed):
            progressed.options = []
            progressed.requested_target = cls._choice_generation_wait_target(progressed.focus)
            progressed.input_hint = "请点击“重新生成知识选项”"
        return progressed

    @classmethod
    def _advance_repeated_target(
        cls,
        session: TeachingSession,
        step: TeachingMicroStep,
    ) -> TeachingMicroStep:
        """Keep the current topic while replacing an unchanged question."""
        progressed = step.model_copy(deep=True)
        focus = progressed.focus or cls._focus_label(session)
        previous_target = (
            session.turns[-1].micro_step.requested_target
            if session.turns and session.turns[-1].micro_step
            else ""
        )
        if "一个判断依据" in previous_target:
            progressed.requested_target = f"请指出当前情境中支持“{focus}”的一个具体事实"
        elif "一个具体事实" in previous_target:
            progressed.requested_target = f"请说明这个事实怎样支持你对“{focus}”的判断"
        elif "怎样支持" in previous_target:
            progressed.requested_target = f"请在当前情境中用“{focus}”完成一个最小应用"
        elif cls._is_definition_only_target(progressed.requested_target):
            progressed.requested_target = f"请说明“{focus}”在当前情境中的判断依据"
        else:
            progressed.requested_target = f"请只说明“{focus}”这一步的一个判断依据"
        progressed.step_index = max(progressed.step_index, 1) + 1
        progressed = cls._apply_response_preference(session, progressed)
        if progressed.response_mode == "single_choice":
            progressed.options = []
            progressed.requested_target = cls._choice_generation_wait_target(focus)
            progressed.input_hint = "请点击“重新生成知识选项”"
        return progressed

    @classmethod
    def _advance_repeated_target_for_student(
        cls,
        session: TeachingSession,
        step: TeachingMicroStep,
        student_message: str,
    ) -> TeachingMicroStep:
        """Honor an explicit learner request without changing the topic.

        This remains domain-neutral: it changes only the evidence request,
        while preserving the locked context and representation.  In
        particular, asking for an example should not cause an abstract
        definition to be repeated or a subject-specific answer to be invented.
        """
        progressed = cls._advance_repeated_target(session, step)
        normalized = student_message.lower()
        example_cues = ("具体例子", "具体例", "举例", "举个例子", "实例", "example")
        if any(cue in normalized for cue in example_cues):
            progressed.requested_target = f"请用一个具体例子说明“{progressed.focus}”的含义"
            progressed.step_index = max(progressed.step_index, step.step_index + 1)
            progressed = cls._apply_response_preference(session, progressed)
            if progressed.response_mode == "single_choice":
                progressed.options = []
                progressed.requested_target = cls._choice_generation_wait_target(progressed.focus)
                progressed.input_hint = "请点击“重新生成知识选项”"
        return progressed

    @staticmethod
    def _apply_response_preference(
        session: TeachingSession,
        step: TeachingMicroStep,
    ) -> TeachingMicroStep:
        preference = session.profile.response_preference
        if preference != "auto":
            step.response_mode = preference
            if preference != "single_choice":
                step.options = []
        return step

    @staticmethod
    def _choice_generation_wait_target(focus: str) -> str:
        """Describe a blocked choice turn without inventing domain answers."""
        return f"等待重新生成与“{focus}”匹配的知识选项"

    @staticmethod
    def _response_mode_shape_valid(step: TeachingMicroStep) -> bool:
        if step.response_mode == "single_choice":
            ids = [option.option_id.strip() for option in step.options]
            texts = [option.text.strip() for option in step.options]
            confidence_cues = ("我能说明", "我还不能说明", "我会", "我不会", "我不确定", "需要提示")
            return (
                2 <= len(step.options) <= 4
                and len(set(ids)) == len(ids)
                and len(set(texts)) == len(texts)
                and all(ids)
                and all(texts)
                and not all(any(cue in text for cue in confidence_cues) for text in texts)
            )
        return step.response_mode in {"open", "fill_blank", "numeric"} and not step.options

    @staticmethod
    def _fallback_micro_step(
        session: TeachingSession,
        skill: TeachingSkill,
        action_type: str,
        previous_step: TeachingMicroStep | None,
    ) -> TeachingMicroStep:
        if previous_step:
            previous = HybridTeachingAgent._apply_response_preference(
                session, previous_step.model_copy(deep=True)
            )
            if previous.response_mode == "single_choice" and not HybridTeachingAgent._response_mode_shape_valid(previous):
                previous.options = []
                previous.requested_target = HybridTeachingAgent._choice_generation_wait_target(previous.focus)
                previous.input_hint = "请点击“重新生成知识选项”"
            return previous
        focus = HybridTeachingAgent._focus_label(session)
        response_mode = session.profile.response_preference
        if response_mode == "auto":
            response_mode = "open"
        requested_target = (
            HybridTeachingAgent._choice_generation_wait_target(focus)
            if response_mode == "single_choice"
            else f"请用自己的话说明“{focus}”是什么意思"
        )
        return TeachingMicroStep(
            focus=focus,
            context=f"当前主题：{session.goal.topic}",
            known_fact="当前还没有足够的学生回答证据。",
            requested_target=requested_target,
            representation="沿用当前课程中的表示法",
            expected_signal=skill.student_signals[0] if skill.student_signals else "学生能说明一步判断",
            step_index=1,
            response_mode=response_mode,
            options=[],
            input_hint=(
                "请点击“重新生成知识选项”"
                if response_mode == "single_choice"
                else "请用自己的话回答"
            ),
        )

    @staticmethod
    def _fallback_review(review: TeacherReview, reason: str, message: str) -> TeacherReview:
        return TeacherReview(
            valid=False,
            one_step=True,
            one_context=True,
            one_question=message.count("？") + message.count("?") == 1,
            fact_consistent=True,
            same_context=review.same_context,
            answer_leakage=False,
            response_mode_valid=True,
            options_valid=True,
            issues=list(dict.fromkeys(review.issues + [reason])),
        )

    def _final_message(
        self,
        message: str,
        session: TeachingSession,
        skill: TeachingSkill,
        action_type: str,
        micro_step: TeachingMicroStep | None = None,
    ) -> str:
        guarded = self._guard_teacher_message(message, session, skill, action_type, micro_step)
        # The model can hide two tasks behind one question mark (for example,
        # "是否改变；如果改变，为什么").  Review metadata is useful for
        # audit, but the learner-facing boundary must be deterministic: never
        # display such a draft even if a semantic reviewer accepted it.
        if (
            self._contains_multiple_requests(guarded)
            or guarded.count("？") + guarded.count("?") != 1
        ):
            safe_step = micro_step.model_copy(deep=True) if micro_step is not None else None
            if safe_step is not None:
                safe_step.requested_target = self._single_target(safe_step.requested_target, safe_step.focus)
            guarded = self._offline_message(session, skill, action_type, safe_step)
            if self._contains_multiple_requests(guarded) or guarded.count("？") + guarded.count("?") != 1:
                focus = (safe_step.focus if safe_step is not None else self._focus_label(session)) or session.goal.topic
                guarded = f"我们继续保持当前情境。请只说明“{focus}”的一个判断依据？"
        return guarded

    @classmethod
    def _guard_teacher_message(
        cls,
        message: str,
        session: TeachingSession,
        skill: TeachingSkill,
        action_type: str,
        micro_step: TeachingMicroStep | None = None,
    ) -> str:
        direct_answer_cues = ("答案是", "最终答案", "正确选项是", "直接写成", "直接得到", "你只要记住")
        internal_opening_cues = (
            "继续聚焦",
            "当前关注点",
            "真实理解",
            "下一步目标",
            "本轮只处理",
            "当前还没有足够的学生回答证据",
            "专业决策证据",
            "答辩演示输入建议",
        )
        answer_pattern = re.compile(r"(?:所以|因此|故)\s*[A-Za-z\u4e00-\u9fff]\s*=\s*[-+]?\d")
        if (
            not message
            or any(cue in message for cue in direct_answer_cues)
            or answer_pattern.search(message)
            or any(cue in message for cue in internal_opening_cues)
            or cls._contains_internal_task_language(message)
        ):
            return cls._offline_message(session, skill, action_type, micro_step)
        cleaned = message.removeprefix("教师：").strip()
        if session.turns:
            # A later turn should sound like a continuation, not like a new
            # lesson.  This is a generic discourse guard and leaves the
            # model's subject content untouched.
            cleaned = re.sub(
                r"^我们先从“[^”]+”开始[。；;]\s*",
                "我们继续保持当前情境。",
                cleaned,
                count=1,
            )
        if cls._contains_multiple_requests(cleaned):
            return cls._offline_message(session, skill, action_type, micro_step)
        if cleaned.count("？") + cleaned.count("?") > 1:
            return cls._offline_message(session, skill, action_type, micro_step)
        latest_signal = session.state.evidence[-1].signal_type if session.state.evidence else "partial"
        if latest_signal not in {"positive", "transfer"}:
            praise_patterns = (
                "你说得对", "你答对了", "回答正确", "很好，", "这个结论本身没错", "你已经正确",
            )
            for pattern in praise_patterns:
                cleaned = cleaned.replace(pattern, "我们先检验这个结论")
        if not any(marker in cleaned for marker in ("？", "?")):
            cleaned = cleaned.rstrip("，。； ") + "。你能说明下一步的依据吗？"
        final_message = cls._truncate_complete(cleaned)
        # The review model may revise the draft after the first check. Validate
        # the final bounded string as well so the UI always preserves the
        # one-step/one-question contract.
        if final_message.count("？") + final_message.count("?") > 1:
            return cls._offline_message(session, skill, action_type, micro_step)
        return final_message

    @staticmethod
    def _truncate_complete(message: str, limit: int = 480) -> str:
        """Bound UI output without cutting a sentence or question in half."""
        if len(message) <= limit:
            return message
        prefix = message[:limit]
        boundaries = [prefix.rfind(marker) for marker in ("？", "?", "。", "！", "\n")]
        cut = max(boundaries)
        if cut >= limit // 2:
            shortened = prefix[: cut + 1].strip()
        else:
            shortened = prefix[: limit - 18].rstrip("，；： ") + "……"
        if not any(marker in shortened for marker in ("？", "?")):
            shortened += "你能说明判断依据吗？"
        return shortened

    @classmethod
    def student_visible_message(cls, session: TeachingSession, turn: ConversationTurn) -> str:
        """Render a safe student message without rewriting persisted history."""
        message = str(turn.teacher_message or "").strip()
        internal_cues = (
            "本轮只处理",
            "当前还没有足够的学生回答证据",
            "专业决策证据",
            "答辩演示输入建议",
            "当前关注点",
            "下一步目标",
        )
        if message and not any(cue in message for cue in internal_cues):
            return message
        if turn.micro_step is not None:
            # The opening turn is already stored in the session by the time
            # the page renders it. Do not let that stored turn make the first
            # student-visible question sound like a continuation of itself.
            context_session = session
            if turn is session.turns[0] and not turn.student_message:
                context_session = session.model_copy(deep=True)
                context_session.turns = []
            return cls._natural_message(context_session, turn.micro_step)
        return (
            f"我们先从“{session.goal.topic}”开始。"
            f"先回答一个问题：你能用自己的话说明“{session.goal.topic}”是什么意思？"
        )

    @classmethod
    def _offline_message(
        cls,
        session: TeachingSession,
        skill: TeachingSkill,
        action_type: str,
        micro_step: TeachingMicroStep | None = None,
    ) -> str:
        step = micro_step or cls._fallback_micro_step(session, skill, action_type, None)
        return cls._natural_message(session, step)

    @staticmethod
    def _natural_message(session: TeachingSession, step: TeachingMicroStep) -> str:
        focus = step.focus or session.goal.topic
        if step.response_mode == "single_choice" and not HybridTeachingAgent._response_mode_shape_valid(step):
            return (
                f"本轮需要用单选题处理“{focus}”，但暂时没有生成出可靠的知识选项。"
                "请点击“重新生成知识选项”再试一次。"
            )
        # Model-generated planning fields are facts/context, not student-facing
        # questions. Normalize punctuation before adding the one final prompt,
        # otherwise a stray question mark in `context` would violate the
        # one-question contract of the fallback itself.
        context = (step.context or f"{session.goal.topic}中的一个基础问题").replace("？", "。").replace("?", "。")
        known_fact = step.known_fact.strip().replace("？", "。 ").replace("?", ". ")
        target = HybridTeachingAgent._single_target(
            step.requested_target.strip() or f"请说明“{focus}”的含义",
            focus,
        )
        target = target.replace("？", "。 ").replace("?", ". ")
        target = target.rstrip("。！？? .，,：:；;") + "？"
        context = context.removeprefix("当前主题：").strip()
        if "新情境" in target or "换一个" in target:
            opening = "现在换一个不同的新情境。"
        else:
            opening = (
                "我们继续保持当前情境。"
                if session.turns
                else f"我们先从“{context.rstrip('。！？? ')}”开始。"
            )
        internal_facts = ("当前还没有足够的学生回答证据", "无历史对话", "无（首轮）")
        if known_fact and known_fact not in context and not any(item in known_fact for item in internal_facts):
            opening += f"已知，{known_fact.rstrip('。！？? ')}。"
        target = target.removeprefix("请").lstrip()
        message = f"{opening}先回答一个问题：{target}"
        if session.turns and HybridTeachingAgent._repeats_previous_question(session, message):
            return f"我们保持刚才的情境，换一种问法：你能说明“{focus}”这一步的判断依据吗？"
        return message

    @staticmethod
    def _history_summary(session: TeachingSession, limit: int = 4) -> str:
        parts = []
        for item in session.imported_history[-limit:]:
            role = "学生" if item.get("role") in {"student", "user"} else "教师"
            content = str(item.get("content", "")).strip()
            if content:
                parts.append(f"{role}：{content}")
        for turn in session.turns[-limit:]:
            if turn.student_message:
                parts.append(f"学生：{turn.student_message}")
            parts.append(f"教师：{turn.teacher_message}（Skill={turn.selected_skill_id}）")
        return "｜".join(parts) if parts else "无历史对话"

    @staticmethod
    def _generic_reason(skill_type: str, state: StudentState) -> str:
        reasons = {
            "diagnostic": "学生表达困惑，先定位知识缺口和错误来源。",
            "scaffold": "上一轮未形成进展，降低任务粒度并提供一层提示。",
            "correction": "误解持续出现，切换到对比与反例纠正，避免重复讲解。",
            "transfer": "掌握度已接近阈值，切换到新情境验证是否真正理解。",
        }
        return f"{reasons[skill_type]} 当前关注点：{state.next_focus}"

    @staticmethod
    def _append_turn(
        session: TeachingSession,
        student_message: str,
        decision: AgentDecision,
        before: StudentState,
        after: StudentState,
    ) -> None:
        plan = deepcopy(decision.skill_plan)
        if plan is None:
            plan = HybridTeachingAgent._preserved_skill_plan_for_turn(session, decision)
        session.turns.append(
            ConversationTurn(
                round_index=len(session.turns) + 1,
                student_message=student_message,
                teacher_message=decision.teacher_message,
                selected_skill_id=decision.primary_skill_id,
                support_skill_id=decision.support_skill_id,
                selection_reason=decision.selection_reason,
                action_type=decision.action_type,
                state_before=deepcopy(before),
                state_after=deepcopy(after),
                switch_reason=decision.switch_reason,
                decision_mode=decision.decision_mode,
                candidate_skill_ids=decision.candidate_skill_ids,
                policy_rule=decision.policy_rule,
                candidate_audit=decision.candidate_audit,
                fallback_reason=decision.fallback_reason,
                stop_decision=decision.stop_reason or "继续教学",
                micro_step=deepcopy(decision.micro_step),
                teacher_review=deepcopy(decision.teacher_review),
                generation_audit=deepcopy(decision.generation_audit),
                phase=decision.phase,
                content_skill_id=plan.content_skill_id if plan else None,
                strategy_skill_id=plan.strategy_skill_id if plan else None,
                skill_plan=plan,
                question_contract=deepcopy(decision.question_contract),
                llm_trace=deepcopy(decision.llm_trace),
                generation_revisions=[
                    GenerationRevision(
                        revision_index=1,
                        teacher_message=decision.teacher_message,
                        reason="首轮生成",
                    )
                ],
            )
        )

    @staticmethod
    def _preserved_skill_plan_for_turn(
        session: TeachingSession,
        decision: AgentDecision,
    ) -> SkillPlan | None:
        """Build a role-safe plan for legacy/custom decisions without one."""
        previous = HybridTeachingAgent._latest_plan(session)
        if previous is not None:
            return previous.model_copy(deep=True)
        return SkillPlan(
            content_skill_id=None,
            strategy_skill_id=decision.primary_skill_id or None,
            content_skill_reason="未指定学科 Skill",
            strategy_reason=decision.selection_reason,
            candidate_strategy_skill_ids=[decision.primary_skill_id],
        )
