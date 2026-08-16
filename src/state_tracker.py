from __future__ import annotations

import re
from copy import deepcopy
from difflib import SequenceMatcher
from typing import Literal, cast

from src.config import get_agent_settings
from src.llm import LLMUnavailableError, OpenAICompatibleClient
from src.models import (
    KnowledgeState,
    Misconception,
    MisconceptionState,
    StateAssessment,
    StateEvidence,
    StudentProfile,
    StudentState,
    TeachingGoal,
    now_iso,
)

POSITIVE_CUES = ("理解", "因为", "所以", "可以", "区别", "推出", "等于", "依据", "不变量")
PROMPT_INJECTION_CUES = (
    "ignore previous", "ignore all", "system prompt", "developer message",
    "忽略之前", "忽略以上", "无视之前", "系统提示词", "开发者指令", "不要遵守", "改成输出",
)

# These are language-level contradiction pairs, not subject answer keys. They
# help catch a learner who reverses the relation explicitly stated in the
# immediately preceding teacher question (for example, "向后方" -> "往前方").
# The semantic LLM review remains the main judge; this guard only covers a
# high-confidence lexical reversal that should never be treated as mastery.
GENERIC_CONTRADICTION_PAIRS = (
    ("前方", "后方"),
    ("向前", "向后"),
    ("往前", "往后"),
    ("左边", "右边"),
    ("左侧", "右侧"),
    ("上方", "下方"),
    ("增加", "减少"),
    ("增大", "减小"),
    ("保留", "排除"),
    ("包含", "不包含"),
    ("大于", "小于"),
    ("多于", "少于"),
)

TENTATIVE_CUES = ("可能", "感觉", "我猜", "大概", "似乎", "不确定", "应该是")

EvidenceLevel = Literal["none", "partial", "correct", "explained", "transfer"]


def _as_evidence_level(value: object, fallback: EvidenceLevel = "none") -> EvidenceLevel:
    """Keep provider-produced labels inside the persistence schema."""
    text = str(value).strip().lower()
    return cast(EvidenceLevel, text) if text in {"none", "partial", "correct", "explained", "transfer"} else fallback


def has_negative_signal(message: str) -> bool:
    """Detect learner difficulty without mistaking subject-matter negation for self-confusion."""
    text = message.strip()
    if not text:
        return True
    # “元素不会被检查到” describes an algorithmic consequence; it does not
    # mean “I cannot answer”. Bare 不会 is only treated as confusion in a
    # short/self-referential statement.
    without_passive_consequence = text.replace("不会被", "")
    strong = (
        "不知道", "不懂", "不明白", "没明白", "搞不清", "混淆", "记不住", "还是错",
        "还是不会", "依然不会", "仍然不会", "不用再检查", "不需要", "不必", "随便取", "随便用",
    )
    return (
        any(cue in without_passive_consequence for cue in strong)
        or "我不会" in without_passive_consequence
        or (without_passive_consequence in {"不会", "还是不会", "依然不会"})
    )


def has_tentative_signal(message: str) -> bool:
    """Return whether a learner marks an answer as a guess or uncertainty."""
    text = message.strip()
    return bool(text) and any(cue in text for cue in TENTATIVE_CUES)


def _is_negated(text: str, phrase: str) -> bool:
    """Check common Chinese negation immediately attached to a phrase."""
    return any(
        marker in text
        for marker in (
            f"不是{phrase}",
            f"并非{phrase}",
            f"不{phrase}",
            f"没有{phrase}",
            f"不会{phrase}",
        )
    )


def detect_generic_contradiction(reference: str, response: str) -> tuple[str, str] | None:
    """Detect a clear lexical reversal between a prompt and a reply.

    This deliberately does not encode what is correct for any course. It only
    reports that the response uses the opposite member of a generic language
    pair found in the teacher's current prompt. Negated alternatives such as
    ``不是前方，而是后方`` are not falsely flagged when the opposite phrase is
    explicitly negated.
    """
    prompt = re.sub(r"\s+", "", reference or "")
    answer = re.sub(r"\s+", "", response or "")
    if not prompt or not answer:
        return None
    for left, right in GENERIC_CONTRADICTION_PAIRS:
        for anchor, opposite in ((left, right), (right, left)):
            if anchor in prompt and opposite in answer and not _is_negated(answer, opposite):
                return anchor, opposite
    return None


def has_prompt_injection(message: str) -> bool:
    lowered = message.strip().lower()
    return any(cue in lowered for cue in PROMPT_INJECTION_CUES)


class StateTracker:
    EVIDENCE_LEVELS = frozenset({"none", "partial", "correct", "explained", "transfer"})

    def __init__(self, llm: OpenAICompatibleClient | None = None, settings: dict | None = None):
        self.llm = llm
        self.settings = get_agent_settings()
        if settings:
            self.settings.update(settings)

    def _level_for(self, assessment: StateAssessment, point: str) -> EvidenceLevel:
        level = assessment.evidence_levels.get(point, "")
        if level in self.EVIDENCE_LEVELS:
            return _as_evidence_level(level)
        if assessment.verification_passed:
            return "transfer"
        if assessment.progress == "improved":
            # Backward-compatible interpretation for older/custom LLM schemas
            # that only return progress. New prompts are required to provide a
            # per-point evidence level.
            return "correct"
        return "none"

    @staticmethod
    def _shared_context_tokens(previous_teacher_message: str, student_message: str) -> set[str]:
        """Find generic lexical anchors shared by the prompt and answer.

        This is only a conservative evidence-mapping fallback when a model
        returns no point mapping. It does not contain subject answers: the
        answer must share a variable/concept token with the immediately
        preceding teacher prompt, and the current focus is the only point that
        can be updated.
        """
        def tokens(text: str) -> set[str]:
            compact = re.sub(r"\s+", "", text.lower())
            ascii_tokens = set(re.findall(r"[a-z_][a-z0-9_]{1,}", compact))
            han = "".join(re.findall(r"[\u4e00-\u9fff]", compact))
            han_tokens = {han[index : index + 2] for index in range(max(0, len(han) - 1))}
            return ascii_tokens | han_tokens

        stop_tokens = {
            "回答", "问题", "当前", "学生", "内容", "说明", "判断", "理解", "知识", "什么", "可以",
        }
        return (tokens(previous_teacher_message) & tokens(student_message)) - stop_tokens

    @classmethod
    def _has_generic_repair_evidence(cls, state: StudentState, message: str) -> bool:
        """Recognize a concise correction without encoding subject answers."""
        if not state.misconceptions or has_negative_signal(message):
            return False
        contrast_cues = ("不是", "而是", "不应该", "不能", "不会", "原因是", "只有", "因为", "不成立")
        if not any(cue in message for cue in contrast_cues):
            return False
        misconception_text = " ".join(
            f"{item.label} {item.evidence}" for item in state.misconceptions
        )
        return bool(cls._shared_context_tokens(misconception_text, message))

    def _assessment_structured(
        self,
        system: str,
        user: str,
        schema_hint: str,
        temperature: float = 0.0,
    ) -> dict:
        """Call the state model under one per-turn semantic-review budget."""
        if getattr(self, "_assessment_call_count", 0) >= getattr(self, "_assessment_call_budget", 5):
            raise LLMUnavailableError("本轮状态复核预算已用尽，跳过额外语义复核")
        self._assessment_call_count = getattr(self, "_assessment_call_count", 0) + 1
        if self.llm is None:
            raise LLMUnavailableError("状态诊断器不可用")
        return self.llm.structured(system, user, schema_hint, temperature=temperature)

    def _delta_for(
        self,
        assessment: StateAssessment,
        point: str,
        previous: KnowledgeState | None,
        quote: str,
        current_mastery: float,
    ) -> tuple[EvidenceLevel, float]:
        level = self._level_for(assessment, point)
        if level == "transfer" and not assessment.verification_passed:
            level = "explained"
        targets = dict(
            self.settings.get(
                "mastery_evidence_targets",
                {"partial": 0.50, "correct": 0.68, "explained": 0.82, "transfer": 0.92},
            )
        )
        confidence = max(0.0, min(1.0, assessment.confidence))
        confidence_factor = 0.55 + 0.35 * confidence
        if assessment.progress == "regressed":
            return level, float(self.settings.get("mastery_regression_delta", -0.06)) * (0.60 + 0.40 * confidence)
        if level in {"partial", "correct", "explained", "transfer"}:
            target = float(targets.get(level, 0.0))
            delta = max(0.0, target - (previous.mastery if previous else current_mastery)) * confidence_factor
            if previous and previous.evidence:
                last_quote = " ".join(previous.evidence[-1].student_quote.split()).lower()
                normalized_quote = " ".join(quote.split()).lower()
                if normalized_quote and normalized_quote == last_quote:
                    delta *= float(self.settings.get("mastery_repeat_novelty", 0.35))
            return level, delta
        return level, 0.0

    def update(
        self,
        goal: TeachingGoal,
        profile: StudentProfile,
        state: StudentState,
        student_message: str,
        current_skill_id: str = "",
        round_index: int = 0,
        previous_teacher_message: str = "",
        active_knowledge_point: str = "",
    ) -> StudentState:
        before = deepcopy(state)
        assessment = self._assess(
            goal,
            profile,
            state,
            student_message,
            previous_teacher_message=previous_teacher_message,
        )
        # In the live single-step flow, an LLM may mention a future concept in
        # ``affected_points`` even when the current question is about another
        # point. Prompts discourage that behavior, but prompts are not a
        # state-integrity boundary. Keep evidence and mastery updates scoped to
        # the active route step; legacy/evaluation callers without an active
        # point retain the broader historical behavior.
        if active_knowledge_point in goal.knowledge_points:
            reported_points = list(assessment.affected_points)
            assessment.affected_points = [
                point for point in reported_points if point == active_knowledge_point
            ]
            assessment.mastery_updates = {
                point: value
                for point, value in assessment.mastery_updates.items()
                if point == active_knowledge_point
            }
            assessment.evidence_levels = {
                point: value
                for point, value in assessment.evidence_levels.items()
                if point == active_knowledge_point
            }
            if reported_points and not assessment.affected_points:
                # The answer was mapped outside the active micro-step. Do not
                # turn that mismatch into evidence or a misconception on the
                # current point; wait for a response to the actual question.
                assessment.progress = "unchanged"
                assessment.misconceptions = []
                assessment.understanding_signals = ["回答涉及其他知识点，暂不更新当前小步"]
                assessment.evidence_reason = "状态更新被单步路线边界拦截"
        # Apply the generic repair gate again at the state boundary. The
        # provider may time out or return an inconsistent intermediate review;
        # a clear contrastive correction with shared context must still be
        # allowed to resolve the active misconception deterministically.
        if self._has_generic_repair_evidence(state, student_message):
            active_points = [
                item.knowledge_point
                for item in state.misconception_states
                if item.knowledge_point in goal.knowledge_points
            ]
            assessment.progress = "improved"
            assessment.misconceptions = []
            assessment.affected_points = list(
                dict.fromkeys(active_points or assessment.affected_points or goal.knowledge_points[:1])
            )
            assessment.evidence_levels = {
                point: "explained" for point in assessment.affected_points
            }
            assessment.understanding_signals = ["回答包含与当前活动误解相反的解释"]
            assessment.evidence_reason = "回答用对比或因果关系纠正了当前活动误解"
        if current_skill_id not in {
            "transfer_verification_v1",
            "fixed_verification",
            "generic_verification",
        }:
            # Asking for a transfer task is not evidence of passing one. A
            # verification result is admissible only after a verification
            # action was actually issued on the preceding teacher turn.
            assessment.verification_passed = False
        updated = state.model_copy(deep=True)

        affected = [point for point in assessment.affected_points if point in goal.knowledge_points]
        if not assessment.affected_points and (self.llm is None or not self.llm.available):
            # The local heuristic has no semantic point mapper, so it keeps
            # its documented first-focus fallback. A live LLM assessment with
            # no affected_points is different: it is uncertainty, not evidence
            # for the first knowledge point, and must not move mastery.
            affected = goal.knowledge_points[:1]
        evidence_points = (
            affected
            if active_knowledge_point in goal.knowledge_points
            else affected or goal.knowledge_points[:1]
        )

        # The transition below resolves active misconceptions when this turn
        # contains improvement evidence. Canonicalize the visible explanation
        # so an LLM cannot say "not fully resolved" while structured state has
        # already removed the misconception.
        resolvable_labels = {
            item.label
            for item in state.misconception_states
            if assessment.progress == "improved"
            and (not item.knowledge_point or item.knowledge_point in affected)
        }
        if assessment.progress == "improved" and resolvable_labels:
            points = "、".join(affected)
            assessment.evidence_reason = (
                f"回答对“{points}”给出了正确解释；状态一致性约束确认相关活动误解已改善并解除。"
            )
        quote = student_message.strip()
        evidence_by_point: dict[str, tuple[EvidenceLevel, float]] = {}
        for point in goal.knowledge_points:
            old = updated.mastery.get(point, 0.3)
            if point not in affected:
                continue
            previous_knowledge = updated.knowledge_states.get(point)
            level, delta = self._delta_for(assessment, point, previous_knowledge, quote, old)
            evidence_by_point[point] = (level, delta)
            updated.mastery[point] = round(max(0.0, min(1.0, old + delta)), 3)

        signal_type: Literal["positive", "partial", "negative", "empty", "transfer"] = (
            "empty" if not quote else "transfer" if assessment.verification_passed else
            "positive" if assessment.progress == "improved" else
            "negative" if assessment.progress == "regressed" else "partial"
        )
        for point in evidence_points:
            level, _ = evidence_by_point.get(point, ("none", 0.0))
            evidence = StateEvidence(
                student_quote=quote or "学生未作答",
                knowledge_point=point,
                signal_type=signal_type,
                evidence_level=level,
                round_index=round_index,
                reason=assessment.evidence_reason or f"状态判断：{assessment.progress}",
            )
            updated.evidence.append(evidence)
            previous_knowledge = updated.knowledge_states.get(
                point, KnowledgeState(mastery=before.mastery.get(point, 0.3), confidence=0.25)
            )
            previous_knowledge.mastery = updated.mastery[point]
            previous_knowledge.last_evidence_level = level
            if point in affected:
                previous_knowledge.confidence = round(
                    min(1.0, previous_knowledge.confidence + 0.12 * assessment.confidence), 3
                )
            previous_knowledge.evidence = [*previous_knowledge.evidence[-4:], evidence]
            previous_knowledge.updated_at = now_iso()
            updated.knowledge_states[point] = previous_knowledge

        existing = {item.label: item for item in updated.misconceptions}
        previous_states = {item.label: item for item in updated.misconception_states}
        observed_labels = {item.label for item in assessment.misconceptions}
        resolved_labels = resolvable_labels
        for label in resolved_labels:
            existing.pop(label, None)
            previous_states.pop(label, None)
            observed_labels.discard(label)
        remaining_assessment_misconceptions = [
            item for item in assessment.misconceptions if item.label not in resolved_labels
        ]
        if assessment.progress == "improved" and not remaining_assessment_misconceptions:
            # A correct explanation is evidence that the active misconception
            # affecting this turn has improved. Remove it from the current-state
            # list instead of accumulating it forever as if it were still active.
            if not state.misconception_states:
                # Compatibility for schema-v1 sessions that did not yet store
                # knowledge-point associations for misconceptions.
                existing.clear()
        else:
            for item in remaining_assessment_misconceptions:
                if item.label in existing:
                    if assessment.progress != "improved":
                        existing[item.label].count += 1
                    existing[item.label].evidence = item.evidence or existing[item.label].evidence
                else:
                    existing[item.label] = item

        updated.misconceptions = list(existing.values())
        misconception_states: list[MisconceptionState] = []
        for item in updated.misconceptions:
            previous_misconception = previous_states.get(item.label)
            was_observed = item.label in observed_labels
            if assessment.progress == "improved":
                consecutive = 0
            elif was_observed:
                consecutive = (previous_misconception.consecutive_count if previous_misconception else 0) + 1
            else:
                consecutive = 0
            misconception_states.append(
                MisconceptionState(
                    label=item.label,
                    evidence=item.evidence,
                    count=item.count,
                    consecutive_count=consecutive,
                    knowledge_point=(
                        previous_misconception.knowledge_point
                        if previous_misconception
                        else affected[0]
                        if affected
                        else ""
                    ),
                    improving=assessment.progress == "improved",
                )
            )
        updated.misconception_states = misconception_states
        updated.understanding_signals = assessment.understanding_signals
        updated.next_focus = assessment.next_focus or updated.next_focus
        updated.transfer_verified = updated.transfer_verified or assessment.verification_passed

        if updated.average_mastery() > before.average_mastery():
            updated.no_progress_rounds = 0
        else:
            updated.no_progress_rounds += 1

        return StudentState.model_validate(updated.model_dump())

    def _assess(
        self,
        goal: TeachingGoal,
        profile: StudentProfile,
        state: StudentState,
        message: str,
        previous_teacher_message: str = "",
    ) -> StateAssessment:
        max_state_reviews = int(self.settings.get("max_state_reviews", 2))
        self._assessment_call_count = 0
        self._assessment_call_budget = max(
            1,
            min(5, int(self.settings.get("state_review_call_budget", 2))),
        )
        if has_prompt_injection(message):
            return StateAssessment(
                mastery_updates=dict(state.mastery),
                misconceptions=[],
                understanding_signals=["检测到与学习任务无关的指令文本，未作为掌握证据"],
                next_focus=state.next_focus,
                verification_passed=False,
                progress="unchanged",
                affected_points=[],
                confidence=0.95,
                evidence_reason="提示注入守卫：忽略元指令，只保留教学任务",
            )
        if self.llm and self.llm.available:
            try:
                data = self._assessment_structured(
                    (
                        "你是学生状态诊断器。必须把上一轮教师问题、教学动作和学生回答放在一起判断，"
                        "不能只按关键词或回答长度打分。学生用一个词或一句短话准确回答当前的最小问题时，"
                        "应判为 improved，而不是因为没有长篇解释就判为 partial；只有与问题冲突、明确错误、"
                        "或明确无法作答时才判为 regressed。流利、完整或自信的表达不等于正确；如果其中的"
                        "核心断言与教学目标或当前概念直接冲突，必须判为 regressed 并记录误解。请只依据对话证据，"
                        "不要替学生补写未说出的理解，"
                        "也不要把尚未回答的知识点标记为 affected_points。对每个 affected_points 给出 evidence_levels："
                        "partial 表示方向部分正确，correct 表示正确回答当前问题，explained 表示说明了依据或纠正了误解，"
                        "transfer 表示在新情境中独立应用。如果学生没有逐字复述教师问题，但明确解释了当前活动"
                        "知识点或纠正了已有误解，也应判为 improved。掌握度数值只是兼容字段，必须以证据等级作为更新依据。"
                    ),
                    (
                        f"教学目标：{goal.model_dump_json()}\n学生画像：{profile.model_dump_json()}\n"
                        f"原状态：{state.model_dump_json()}\n"
                        f"上一轮教师问题/教学动作：{previous_teacher_message.strip() or '无（首轮）'}\n"
                        f"学生回答：{message}"
                    ),
                    (
                        '{"mastery_updates":{"知识点":0到1},"evidence_levels":{"知识点":"none|partial|correct|explained|transfer"},"misconceptions":'
                        '[{"label":"误解","evidence":"回答证据","count":1}],'
                        '"understanding_signals":["信号"],"next_focus":"下一关注点",'
                        '"verification_passed":false,"progress":"improved|unchanged|regressed",'
                        '"affected_points":["只能填写教学目标中的知识点"],'
                        '"confidence":0到1,"evidence_reason":"判断依据"}'
                    ),
                )
                assessment = StateAssessment.model_validate(data)
                # A tentative answer is useful diagnostic evidence, but it is
                # not the same as a verified explanation. Keep it at partial
                # progress so the teacher asks for a reason instead of praising
                # a guess as if the concept were mastered.
                if (
                    has_tentative_signal(message)
                    and assessment.progress == "improved"
                    and not assessment.verification_passed
                ):
                    assessment.progress = "unchanged"
                    assessment.evidence_levels = {
                        point: "partial" for point in assessment.affected_points if point in goal.knowledge_points
                    }
                    assessment.understanding_signals = ["回答方向可能正确，但仍带有猜测，尚缺少解释依据"]
                    assessment.evidence_reason = "学生使用不确定表达，先记录为部分证据，继续索取理由"
                    assessment.confidence = min(assessment.confidence, 0.55)
                # Explicit inability cues are a deterministic safety boundary:
                # a provider must not turn “I don't know / stop checking” into
                # neutral evidence merely because its semantic label was
                # conservative.  This is a generic learner-signal rule, not a
                # subject answer key, and still maps only to the current focus.
                if has_negative_signal(message) and assessment.progress != "regressed":
                    affected = [
                        point for point in assessment.affected_points if point in goal.knowledge_points
                    ] or (
                        [state.next_focus]
                        if state.next_focus in goal.knowledge_points
                        else goal.knowledge_points[:1]
                    )
                    assessment.progress = "regressed"
                    assessment.affected_points = affected
                    assessment.evidence_levels = {point: "none" for point in affected}
                    assessment.misconceptions = assessment.misconceptions or [
                        Misconception(
                            label="当前知识点尚未形成可观察理解",
                            evidence=message.strip(),
                            count=1,
                        )
                    ]
                    assessment.understanding_signals = ["学生明确表达无法继续当前问题"]
                    assessment.evidence_reason = "学生明确表达无法继续当前问题，暂不将其视为中性证据"
                # Explicit inability/stop signals are already high-confidence
                # generic evidence. Do not let a later low-confidence semantic
                # adjudicator replace that regression with a neutral default.
                # This also keeps the normal path within the intended
                # one-primary-review budget.
                if has_negative_signal(message):
                    assessment.affected_points = [
                        point for point in assessment.affected_points if point in goal.knowledge_points
                    ] or ([state.next_focus] if state.next_focus in goal.knowledge_points else goal.knowledge_points[:1])
                    assessment.progress = "regressed"
                    assessment.evidence_levels = {point: "none" for point in assessment.affected_points}
                    assessment.misconceptions = assessment.misconceptions or [
                        Misconception(
                            label="当前知识点尚未形成可观察理解",
                            evidence=message.strip(),
                            count=1,
                        )
                    ]
                    assessment.understanding_signals = ["学生明确表达无法继续当前问题"]
                    assessment.evidence_reason = "学生明确表达无法继续当前问题，暂不将其视为中性证据"
                    return assessment
                if not assessment.affected_points and message.strip() and previous_teacher_message.strip():
                    focus = (
                        state.next_focus
                        if state.next_focus in goal.knowledge_points
                        else goal.knowledge_points[0]
                    )
                    shared_tokens = self._shared_context_tokens(previous_teacher_message, message)
                    if shared_tokens:
                        # Context alignment is evidence that the answer is
                        # about the current focus, not evidence that it is
                        # correct. The model's progress/evidence level still
                        # determines the mastery update.
                        assessment.affected_points = [focus]
                        assessment.evidence_levels.setdefault(
                            focus,
                            "correct" if assessment.progress == "improved" else "none",
                        )
                        assessment.evidence_reason = assessment.evidence_reason or (
                            "回答与上一轮问题共享当前情境锚点，映射到当前关注知识点"
                        )
                # Some providers return a compatibility misconception object
                # with count=0 alongside an improved assessment. It is already
                # marked as inactive and should not trigger another expensive
                # review call.
                if not state.misconceptions and assessment.progress == "improved" and (
                    assessment.confidence < 0.65 or bool(assessment.misconceptions)
                ):
                    assessment.misconceptions = [
                        item for item in assessment.misconceptions if item.count > 0
                    ]
                    if message.strip():
                        # An independent adversarial pass checks that a fluent
                        # answer is not actually a confident misconception.
                        try:
                            contradiction = self._assessment_structured(
                                (
                                    "你是独立的反误解评审器。检查学生当前回答是否包含与教学目标或当前知识"
                                    "关系直接冲突的核心断言。不要因为表达完整、语气自信或包含因果词就判正确；"
                                    "也不要因为回答简短或没有展开全部步骤就判错误。只有明确的概念冲突才设为"
                                    "contains_contradiction=true。"
                            ),
                            (
                                f"教学目标：{goal.model_dump_json()}\n"
                                f"当前状态：{state.model_dump_json()}\n"
                                f"上一轮教师问题/教学动作：{previous_teacher_message.strip() or '无'}\n"
                                f"学生回答：{message}"
                            ),
                                (
                                    '{"contains_contradiction":true或false,'
                                    '"label":"明确的误解标签；无冲突时为空",'
                                    '"affected_points":["只能填写教学目标中的知识点"],'
                                    '"evidence":"回答中的冲突依据"}'
                                ),
                                temperature=0.0,
                            )
                            if bool(contradiction.get("contains_contradiction")):
                                points = [
                                    point for point in contradiction.get("affected_points", [])
                                    if point in goal.knowledge_points
                                ] or assessment.affected_points or goal.knowledge_points[:1]
                                label = str(contradiction.get("label", "当前概念关系出现冲突")).strip()
                                assessment.progress = "regressed"
                                assessment.evidence_levels = {point: "none" for point in points}
                                assessment.affected_points = points
                                assessment.misconceptions = [
                                    Misconception(
                                        label=label or "当前概念关系出现冲突",
                                        evidence=str(contradiction.get("evidence", message)).strip() or message,
                                        count=1,
                                    )
                                ]
                                assessment.understanding_signals = ["独立反误解评审发现回答中的概念冲突"]
                                assessment.evidence_reason = assessment.misconceptions[0].evidence
                        except (LLMUnavailableError, ValueError, TypeError):
                            pass
                if (
                    max_state_reviews >= 2
                    and not state.misconceptions
                    and assessment.progress == "improved"
                    and assessment.misconceptions
                    and assessment.confidence < 0.65
                ):
                    # Progress and a remaining misconception can coexist, but
                    # only when the current answer explicitly contains a
                    # separate error. Incomplete or short answers must not be
                    # promoted to misconceptions by the state model.
                    try:
                        misconception_review = self._assessment_structured(
                            (
                                "你是误解证据复核器。只保留学生当前回答中明确表达或明确展示的错误概念。"
                                "如果回答只是简短、部分完成、没有展开理由，confirmed_labels 必须为空；"
                                "不要因为没有回答教师问题的全部部分就新增误解。"
                            ),
                            (
                                f"上一轮教师问题：{previous_teacher_message.strip() or '无'}\n"
                                f"学生当前回答：{message}\n"
                                f"初步误解：{[item.model_dump(mode='json') for item in assessment.misconceptions]}"
                            ),
                            '{"confirmed_labels":["只填写当前回答明确展示的误解标签"],"reason":"依据"}',
                            temperature=0.0,
                        )
                        if "confirmed_labels" in misconception_review:
                            labels = {str(label) for label in misconception_review.get("confirmed_labels", [])}
                            assessment.misconceptions = [
                                item for item in assessment.misconceptions if item.label in labels
                            ]
                    except (LLMUnavailableError, ValueError, TypeError):
                        pass
                if max_state_reviews >= 1 and not state.misconceptions and assessment.progress == "unchanged" and message.strip() and assessment.confidence < 0.65:
                    # A broad assessor can be overly cautious and return
                    # “unchanged” for a fluent misconception. Run the same
                    # semantic contradiction boundary here so an explicit
                    # conceptual conflict can still enter the diagnostic
                    # branch; absence of a conflict remains unchanged.
                    try:
                        contradiction = self._assessment_structured(
                            (
                                "你是概念冲突边界评审器。结合教学目标、上一轮教师问题和学生回答，"
                                "只判断回答是否包含与当前知识关系直接冲突的核心断言。若回答只是简短、"
                                "部分完成或没有展开理由，不要判为冲突；只有明确错误、否定必要条件或重复"
                                "与目标相反的关系时，contains_contradiction=true。"
                            ),
                            (
                                f"教学目标：{goal.model_dump_json()}\n"
                                f"上一轮教师问题/教学动作：{previous_teacher_message.strip() or '无'}\n"
                                f"学生回答：{message}\n"
                                f"当前状态评审：{assessment.model_dump_json()}"
                            ),
                            (
                                '{"contains_contradiction":true或false,'
                                '"label":"明确的误解标签；无冲突时为空",'
                                '"affected_points":["只能填写教学目标中的知识点"],'
                                '"evidence":"冲突依据"}'
                            ),
                            temperature=0.0,
                        )
                        if bool(contradiction.get("contains_contradiction")):
                            points = [
                                point
                                for point in contradiction.get("affected_points", [])
                                if point in goal.knowledge_points
                            ] or assessment.affected_points or goal.knowledge_points[:1]
                            label = str(contradiction.get("label", "当前概念关系出现冲突")).strip()
                            assessment.progress = "regressed"
                            assessment.evidence_levels = {point: "none" for point in points}
                            assessment.affected_points = points
                            assessment.misconceptions = [
                                Misconception(
                                    label=label or "当前概念关系出现冲突",
                                    evidence=str(contradiction.get("evidence", message)).strip() or message,
                                    count=1,
                                )
                            ]
                            assessment.understanding_signals = ["独立概念冲突边界评审发现回答中的明确错误"]
                            assessment.evidence_reason = assessment.misconceptions[0].evidence
                    except (LLMUnavailableError, ValueError, TypeError):
                        pass
                if (
                    max_state_reviews >= 2
                    and not state.misconceptions
                    and previous_teacher_message.strip()
                    and message.strip()
                    and assessment.progress != "improved"
                    and assessment.confidence < 0.65
                ):
                    # A single semantic pass can be overly conservative when
                    # a learner corrects a misconception in compact language.
                    # Ask the same real LLM for an independent adjudication with
                    # the immediate question and the first assessment visible.
                    # This remains semantic judging; no subject-specific answer
                    # list is used here.
                    try:
                        adjudicated = self._assessment_structured(
                            (
                                "你是第二位独立的学生状态评审员。复核第一位评审是否误把学生的"
                                "简短但正确的回答判成 partial。必须结合上一轮教师问题判断："
                                "如果学生回答已经直接解决了该问题或纠正了当前误解，progress 必须为 improved；"
                                "如果仍与问题冲突或明确错误，才保留 regressed。不要因回答短而降级，"
                                "也不要把未涉及的知识点放入 affected_points，并为涉及的知识点标出 evidence_levels。"
                                "学生只要明确纠正当前活动误解，即使没有复述教师问题，也属于 improved。"
                            ),
                            (
                                f"教学目标：{goal.model_dump_json()}\n"
                                f"原状态：{state.model_dump_json()}\n"
                                f"上一轮教师问题/教学动作：{previous_teacher_message.strip()}\n"
                                f"学生回答：{message}\n"
                                f"第一位评审结果：{assessment.model_dump_json()}"
                            ),
                            (
                            '{"mastery_updates":{"知识点":0到1},"evidence_levels":{"知识点":"none|partial|correct|explained|transfer"},"misconceptions":'
                                '[{"label":"误解","evidence":"回答证据","count":1}],'
                                '"understanding_signals":["信号"],"next_focus":"下一关注点",'
                                '"verification_passed":false,"progress":"improved|unchanged|regressed",'
                                '"affected_points":["只能填写教学目标中的知识点"],'
                                '"confidence":0到1,"evidence_reason":"复核依据"}'
                            ),
                            temperature=0.0,
                        )
                        assessment = StateAssessment.model_validate(adjudicated)
                    except (LLMUnavailableError, ValueError, TypeError):
                        pass
                if (
                    max_state_reviews >= 2
                    and not state.misconceptions
                    and previous_teacher_message.strip()
                    and message.strip()
                    and assessment.progress != "improved"
                    and assessment.confidence < 0.65
                ):
                    # Use a deliberately small semantic gate after the full
                    # rubric. Complex scoring sometimes labels a concise but
                    # correct conceptual repair as merely “partial”. This gate
                    # asks only whether the reply demonstrates real progress;
                    # it contains no course-specific answer keys or phrases.
                    try:
                        progress_gate = self._assessment_structured(
                            (
                                "你是独立的语义进步裁决器，只判断学生这句话是否展示了真实概念进步。"
                                "结合上一轮教师问题和当前活动误解：若回答明确纠正了旧误解、给出了正确原则、"
                                "或直接回答了当前微问题，shows_correct_progress=true；不要因为没有完全展开"
                                "所有步骤、没有覆盖全部教学目标或措辞简短而判 false。一轮回答只要在一个相关"
                                "知识点上给出正确原则，或明确纠正当前活动误解，就属于进步，不等于整课已经掌握。"
                                "即使回答没有复述教师问题，只要它对当前活动知识点给出连贯正确解释，也应视为进步。"
                                "若内容仍错误、回避问题"
                                "或没有学习证据才为 false。请额外区分：学生没有展开理由不等于不会；只有明确说"
                                "不知道、无法回答，或回答与问题事实相冲突，才算失败。"
                            ),
                            (
                                f"教学目标：{goal.model_dump_json()}\n"
                                f"上一轮教师问题/教学动作：{previous_teacher_message.strip()}\n"
                                f"当前活动误解：{[item.model_dump(mode='json') for item in state.misconceptions]}\n"
                                f"学生回答：{message}"
                            ),
                            (
                                '{"shows_correct_progress":true或false,'
                                '"corrects_active_misconception":true或false,'
                                '"contains_correct_relevant_principle":true或false,'
                                '"has_relevant_learning_evidence":true或false,'
                                '"explicitly_unable":true或false,'
                                '"contradicts_question":true或false,'
                                '"relation_to_active_misconception":"corrects|repeats|unrelated|unclear",'
                                '"affected_points":["只能填写教学目标中的知识点"],'
                                '"evidence_levels":{"知识点":"correct|explained"},'
                                '"reason":"简短语义依据"}'
                            ),
                            temperature=0.0,
                        )
                        semantic_progress = (
                            bool(progress_gate.get("shows_correct_progress"))
                            or bool(progress_gate.get("corrects_active_misconception"))
                            or bool(progress_gate.get("contains_correct_relevant_principle"))
                            or bool(progress_gate.get("has_relevant_learning_evidence"))
                            or progress_gate.get("relation_to_active_misconception") == "corrects"
                        )
                        if semantic_progress:
                            gate_points = [
                                point
                                for point in progress_gate.get("affected_points", [])
                                if point in goal.knowledge_points
                            ]
                            assessment.progress = "improved"
                            assessment.verification_passed = False
                            assessment.affected_points = gate_points or assessment.affected_points or goal.knowledge_points[:1]
                            assessment.evidence_levels = {
                                point: _as_evidence_level(
                                    progress_gate.get("evidence_levels", {}).get(point, "correct"),
                                    "correct",
                                )
                                for point in assessment.affected_points
                            }
                            assessment.misconceptions = []
                            reason = str(progress_gate.get("reason", "")).strip() or "语义复核确认回答展示了真实进步"
                            assessment.understanding_signals = [reason]
                            assessment.evidence_reason = reason
                    except (LLMUnavailableError, ValueError, TypeError):
                        pass
                if (
                    max_state_reviews >= 2
                    and previous_teacher_message.strip()
                    and message.strip()
                    and assessment.progress == "regressed"
                    and assessment.confidence < 0.65
                ):
                    # A negative first-pass label is not sufficient evidence of
                    # a misconception. This generic boundary check distinguishes
                    # an answered sub-question from an explicit failure or a
                    # factual contradiction, without using course answer keys.
                    try:
                        relevance = self._assessment_structured(
                            (
                                "你是学习证据边界裁决器。只判断学生回答相对于上一轮教师问题的关系，"
                                "不要判断学科答案细节。若教师一次问了多个方面，而学生只准确回答其中一个"
                                "明确子问题，标记为 direct_answer 或 partial_relevant，不能标记为 explicit_failure。"
                                "没有展开理由不等于不会。只有学生明确表示不知道、无法回答、完全回避学习任务，"
                                "或回答与问题事实直接冲突，才标记为 explicit_failure 或 contradiction。"
                            ),
                            (
                                f"教学目标：{goal.model_dump_json()}\n"
                                f"上一轮教师问题/教学动作：{previous_teacher_message.strip()}\n"
                                f"学生回答：{message}\n"
                                f"当前评审：{assessment.model_dump_json()}"
                            ),
                            (
                                '{"classification":"direct_answer|partial_relevant|unrelated|'
                                'explicit_failure|contradiction",'
                                '"affected_points":["只能填写教学目标中的知识点"],'
                                '"reason":"边界判断依据"}'
                            ),
                            temperature=0.0,
                        )
                        classification = str(relevance.get("classification", ""))
                        points = [
                            point for point in relevance.get("affected_points", []) if point in goal.knowledge_points
                        ]
                        # Relevance to the teacher's question is not evidence
                        # that a factually contradicted answer is correct. A
                        # wrong answer can directly answer the question too;
                        # preserve the explicit misconception and let the
                        # dedicated repair judge decide whether it was fixed.
                        explicit_contradiction = assessment.progress != "improved" and bool(
                            assessment.misconceptions
                        )
                        if classification in {"direct_answer", "partial_relevant"} and not explicit_contradiction:
                            assessment.progress = "improved" if classification == "direct_answer" else "unchanged"
                            assessment.misconceptions = []
                            assessment.affected_points = points or assessment.affected_points or goal.knowledge_points[:1]
                            assessment.evidence_levels = {
                                point: "correct" if classification == "direct_answer" else "partial"
                                for point in assessment.affected_points
                            }
                            assessment.evidence_reason = str(relevance.get("reason", "")).strip() or (
                                "回答直接解决了当前问题中的一个子问题"
                                if classification == "direct_answer"
                                else "回答涉及当前问题，但尚未覆盖全部要求"
                            )
                    except (LLMUnavailableError, ValueError, TypeError):
                        pass
                if message.strip() and state.misconceptions:
                    # Judge repair against the active misconception directly.
                    # This prevents a valid correction from being rejected just
                    # because the learner did not mirror the previous question.
                    # It intentionally runs even after the broad assessor says
                    # improved: the active-misconception judge is the final
                    # consistency check for the transition out of a repair
                    # stage, rather than allowing one conservative or
                    # adversarial pass to keep a repaired misconception alive.
                    try:
                        repair = self._assessment_structured(
                            (
                                "你是活动误解修复裁决器。忽略学生是否逐字回答上一轮教师问题，只比较"
                                "当前学生回答与历史活动误解：如果学生明确否定旧误解、给出正确原则或解释"
                                "为什么旧说法不成立，repairs_misconception=true；如果重复旧误解、明确错误、"
                                "只说不知道或完全无关，才为 false。不要因回答简短而判 false。"
                            ),
                            (
                                f"教学目标：{goal.model_dump_json()}\n"
                                f"历史活动误解：{[item.model_dump(mode='json') for item in state.misconceptions]}\n"
                                f"学生当前回答：{message}"
                            ),
                            (
                                '{"repairs_misconception":true或false,'
                                '"affected_points":["只能填写教学目标中的知识点"],'
                                '"evidence_levels":{"知识点":"correct|explained"},'
                                '"reason":"修复判断依据"}'
                            ),
                            temperature=0.0,
                        )
                        if (
                            not bool(repair.get("repairs_misconception"))
                            and int(self.settings.get("max_repair_reviews", 1)) >= 2
                        ):
                            repair = self._assessment_structured(
                                (
                                    "你是第二位独立的误解修复评审员。把历史误解当作一个待检验的命题，"
                                    "逐句比较学生当前回答：如果学生明确表达了与旧命题相反的正确关系，"
                                    "或说明旧命题为什么不成立，repairs_misconception=true。不要要求学生复述"
                                    "上一轮问题，也不要把没有展开全部步骤当作没有修复。"
                                ),
                                (
                                    f"教学目标：{goal.model_dump_json()}\n"
                                    f"历史活动误解：{[item.model_dump(mode='json') for item in state.misconceptions]}\n"
                                    f"学生当前回答：{message}"
                                ),
                                (
                                    '{"repairs_misconception":true或false,'
                                    '"affected_points":["只能填写教学目标中的知识点"],'
                                    '"evidence_levels":{"知识点":"correct|explained"},'
                                    '"reason":"第二位评审依据"}'
                                ),
                                temperature=0.0,
                            )
                        if (
                            not bool(repair.get("repairs_misconception"))
                            and int(self.settings.get("max_repair_reviews", 1)) >= 3
                        ):
                            # A final arbiter is only used when two focused
                            # repair checks disagree with the observed answer.
                            # It sees the immediate question and the broad
                            # assessment as well, which reduces false
                            # negatives for compact but fully corrective
                            # explanations without adding a subject answer key.
                            repair = self._assessment_structured(
                                (
                                    "你是最终的误解修复仲裁器。根据历史误解、上一轮教师问题、当前学生回答和"
                                    "初步状态评审，判断是否出现了可观察的修复证据。只要学生明确给出与旧误解"
                                    "相反的条件、正确原则或旧说法不成立的理由，repairs_misconception=true；"
                                    "不要要求逐字复述问题、完整覆盖所有知识点或写出长篇推导。只有重复旧误解、"
                                    "明确错误、表示不会或完全无关时才为 false。"
                                ),
                                (
                                    f"教学目标：{goal.model_dump_json()}\n"
                                    f"上一轮教师问题：{previous_teacher_message.strip() or '无'}\n"
                                    f"历史活动误解：{[item.model_dump(mode='json') for item in state.misconceptions]}\n"
                                    f"当前学生回答：{message}\n"
                                    f"初步状态评审：{assessment.model_dump_json()}"
                                ),
                                (
                                    '{"repairs_misconception":true或false,'
                                    '"affected_points":["只能填写教学目标中的知识点"],'
                                    '"evidence_levels":{"知识点":"correct|explained"},'
                                    '"reason":"最终仲裁依据"}'
                                ),
                                temperature=0.0,
                            )
                        if bool(repair.get("repairs_misconception")):
                            repair_points = [
                                point for point in repair.get("affected_points", []) if point in goal.knowledge_points
                            ]
                            active_points = [
                                item.knowledge_point
                                for item in state.misconception_states
                                if item.knowledge_point in goal.knowledge_points
                            ]
                            assessment.progress = "improved"
                            assessment.misconceptions = []
                            assessment.affected_points = (
                                repair_points
                                or active_points
                                or assessment.affected_points
                                or goal.knowledge_points[:1]
                            )
                            assessment.evidence_levels = {
                                point: _as_evidence_level(
                                    repair.get("evidence_levels", {}).get(point, "explained"),
                                    "explained",
                                )
                                for point in assessment.affected_points
                            }
                            assessment.understanding_signals = [
                                str(repair.get("reason", "回答明确纠正了当前活动误解"))
                            ]
                            assessment.evidence_reason = assessment.understanding_signals[0]
                    except (LLMUnavailableError, ValueError, TypeError):
                        pass
                if assessment.progress == "unchanged" and message.strip() and assessment.confidence < 0.65:
                    positive_levels = {"partial", "correct", "explained", "transfer"}
                    if not any(level in positive_levels for level in assessment.evidence_levels.values()):
                        # Separate partial evidence from whole-misconception
                        # repair. A learner can correctly state one principle
                        # while still needing help on the original error; that
                        # evidence should move only the relevant knowledge
                        # point instead of freezing the whole state.
                        try:
                            evidence_probe = self._assessment_structured(
                                (
                                    "你是独立的学习证据提取器。不要判断整题是否掌握，也不要因为回答相关或"
                                    "措辞流利就判正确；只提取学生当前回答中明确说对的局部原则。若回答包含一条"
                                    "正确关系、条件、步骤或对旧说法的部分纠正，给对应知识点标记 partial、correct"
                                    "或 explained；若没有可确认的正确学习证据，所有等级填 none。不要把未涉及的"
                                    "知识点加入 affected_points，也不要用学科关键词代替语义判断。"
                                ),
                                (
                                    f"教学目标：{goal.model_dump_json()}\n"
                                    f"上一轮教师问题/教学动作：{previous_teacher_message.strip() or '无'}\n"
                                    f"学生回答：{message}\n"
                                    f"当前状态评审：{assessment.model_dump_json()}"
                                ),
                                (
                                    '{"affected_points":["只能填写教学目标中的知识点"],'
                                    '"evidence_levels":{"知识点":"none|partial|correct|explained"},'
                                    '"has_confirmed_evidence":true或false,"reason":"证据依据"}'
                                ),
                                temperature=0.0,
                            )
                            extracted_levels = {
                                point: _as_evidence_level(
                                    evidence_probe.get("evidence_levels", {}).get(point, "none")
                                )
                                for point in evidence_probe.get("affected_points", [])
                                if point in goal.knowledge_points
                            }
                            if any(level in positive_levels for level in extracted_levels.values()):
                                assessment.evidence_levels.update(extracted_levels)
                                assessment.affected_points = list(
                                    dict.fromkeys([*assessment.affected_points, *extracted_levels])
                                )
                                assessment.understanding_signals = assessment.understanding_signals or [
                                    str(evidence_probe.get("reason", "回答包含局部正确学习证据"))
                                ]
                                assessment.evidence_reason = str(
                                    evidence_probe.get("reason", "回答包含局部正确学习证据")
                                )
                        except (LLMUnavailableError, ValueError, TypeError):
                            pass
                if self._has_generic_repair_evidence(state, message):
                    active_points = [
                        item.knowledge_point
                        for item in state.misconception_states
                        if item.knowledge_point in goal.knowledge_points
                    ]
                    assessment.progress = "improved"
                    assessment.misconceptions = []
                    assessment.affected_points = list(
                        dict.fromkeys(active_points or assessment.affected_points or goal.knowledge_points[:1])
                    )
                    assessment.evidence_levels = {
                        point: "explained" for point in assessment.affected_points
                    }
                    assessment.understanding_signals = ["回答包含与当前活动误解相反的解释"]
                    assessment.evidence_reason = "回答用对比或因果关系纠正了当前活动误解"
                assessment.affected_points = [
                    point for point in assessment.affected_points if point in goal.knowledge_points
                ]
                # A negative state requires an explicit misconception or an
                # explicit inability signal. A non-empty but incomplete answer
                # without either is ambiguous evidence, not a regression.
                if assessment.progress == "regressed" and not assessment.misconceptions and message.strip():
                    assessment.progress = "unchanged"
                    assessment.evidence_levels = {
                        point: "none" for point in assessment.affected_points
                    }
                    assessment.understanding_signals = assessment.understanding_signals or [
                        "回答尚不充分，暂不判定为退步"
                    ]
                    assessment.evidence_reason = assessment.evidence_reason or "回答不完整，但没有明确错误或无法作答证据"
                return assessment
            except (LLMUnavailableError, ValueError, TypeError):
                pass
        return self._heuristic_assessment(goal, state, message)

    @staticmethod
    def _heuristic_assessment(goal: TeachingGoal, state: StudentState, message: str) -> StateAssessment:
        stripped = message.strip()
        negative = has_negative_signal(stripped)
        positive_count = sum(cue in stripped for cue in POSITIVE_CUES)
        has_causal_explanation = any(cue in stripped for cue in ("因为", "所以", "依据")) and len(stripped) >= 12
        improved = (positive_count >= 2 or has_causal_explanation) and not negative

        # The offline path is deliberately generic: map evidence to the
        # current focus and to knowledge-point names that actually occur in
        # the learner's answer.  It never boosts every point merely because a
        # response was fluent.
        normalized = "".join(stripped.split()).lower()
        mentioned: list[str] = []
        for point in goal.knowledge_points:
            point_text = "".join(point.split()).lower()
            if not point_text:
                continue
            direct = point_text in normalized
            character_overlap = len(set(point_text) & set(normalized)) / max(1, len(set(point_text)))
            reordered = SequenceMatcher(None, point_text, normalized).ratio() >= 0.42
            if direct or character_overlap >= 0.75 or reordered:
                mentioned.append(point)
        focus = state.next_focus if state.next_focus in goal.knowledge_points else goal.knowledge_points[0]
        affected_points = list(dict.fromkeys(mentioned or [focus]))
        updates: dict[str, float] = {}
        for point in affected_points:
            old = state.mastery.get(point, 0.3)
            updates[point] = old + (0.16 if improved else -0.04 if negative else 0.0)

        misconceptions = []
        signals = []
        if negative:
            label = "当前知识点的理解出现困难"
            misconceptions.append(Misconception(label=label, evidence=stripped or "学生未作答"))
            signals.append("学生明确表达困惑或无法作答")
            focus = state.next_focus or goal.knowledge_points[0]
            progress: Literal["improved", "unchanged", "regressed"] = "regressed" if stripped else "unchanged"
        elif improved:
            signals.append("学生能够使用因果关系解释关键概念")
            # Keep the next focus inside the goal's declared knowledge-point
            # set. A generic phrase such as “继续检验下一步推理” is useful in
            # prose but cannot drive candidate filtering or per-point evidence
            # updates. Choose the next unfinished point deterministically; the
            # order comes from the user-provided goal, not a subject answer key.
            remaining = [
                point
                for point in goal.knowledge_points
                if point not in affected_points and state.mastery.get(point, 0.0) < 0.70
            ]
            focus = remaining[0] if remaining else "继续验证当前理解"
            progress = "improved"
        else:
            signals.append("学生给出部分相关信息，但解释尚不完整")
            focus = state.next_focus or goal.knowledge_points[0]
            progress = "unchanged"

        verification = improved and any(
            token in stripped for token in ("新题", "新数组", "新情境", "迁移", "换一个", "另一个", "变式", "同样")
        )
        evidence_level: EvidenceLevel = "transfer" if verification else "explained" if improved else "none"
        return StateAssessment(
            mastery_updates=updates,
            misconceptions=misconceptions,
            understanding_signals=signals,
            next_focus=focus,
            verification_passed=verification,
            progress=progress,
            affected_points=affected_points,
            evidence_levels={point: evidence_level for point in affected_points},
            confidence=0.75 if improved or negative else 0.5,
            evidence_reason=(
                "回答包含可解释的因果关系" if improved else
                "回答明确表达困惑或否定" if negative else
                "回答只有部分相关信息"
            ),
        )
