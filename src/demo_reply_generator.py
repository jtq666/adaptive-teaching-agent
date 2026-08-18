"""Generate learner-side demo replies from the live teaching context.

This helper is intentionally separate from the teaching Agent.  It does not
make a teaching decision, update state, or submit a reply.  It only asks the
configured LLM for a few plausible learner utterances so a presenter can
exercise the real Agent without typing every answer by hand.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from src.llm import LLMUnavailableError, OpenAICompatibleClient
from src.models import TeachingSession

DemoTargetSignal = Literal["auto", "diagnostic", "scaffold", "misconception", "correct", "transfer"]

DEMO_TARGET_LABELS = {
    "auto": "自动生成不同类型",
    "diagnostic": "诊断倾向：表达具体卡点",
    "scaffold": "分层提示倾向：只回答下一步",
    "misconception": "误解倾向：表达错误观点",
    "correct": "正确理解倾向：给出完整依据",
    "transfer": "迁移倾向：换情境应用",
}
DEMO_SIGNAL_LABELS = {
    **DEMO_TARGET_LABELS,
    "confused": "表达困惑",
    "partial": "部分理解",
}

TARGET_BEHAVIOR = {
    "diagnostic": "学生明确说出自己卡在哪里，但不要直接给出完整结论。",
    "scaffold": "学生只完成当前问题的下一小步，表现出需要进一步提示。",
    "misconception": "学生必须表达一个明确但错误的知识观点，不能只是说‘我不会’。",
    "correct": "学生给出与当前知识目标一致的完整、简短依据。",
    "transfer": "学生把当前知识迁移到题目要求的新情境中，给出自己的判断和理由。",
}

# These are learner utterances that have already been accepted by the real
# physics/calculus demo runs.  They are references for the presenter helper,
# not rules for the teaching Agent.
VERIFIED_DEMO_REPLIES: dict[str, dict[str, list[str]]] = {
    "newtons_first_law_via_engineering_examples_v1": {
        "diagnostic": [
            "我不确定，物体没有受到水平方向的力，为什么还会继续向前运动？",
        ],
        "scaffold": [
            "急刹车时我可能会向前倾，但我不太确定是因为惯性还是因为别的，我卡在怎么把保持运动和改变运动状态分开说。",
        ],
        "misconception": [
            "我认为只要物体在运动，就一定需要一个向前的力来维持运动。",
        ],
        "correct": [
            "现在我明白了：合力为零时，物体会保持静止或匀速直线运动；惯性是保持这种状态的性质，不是力。",
        ],
        "transfer": [
            "公交车原来停着，突然向前加速时，我会相对车向后仰，因为身体想保持原来的静止状态，而车向前加速。",
            "公交车匀速向前行驶时突然向右转弯，我会相对车向左偏，因为身体想保持原来向前的直线运动状态。",
            "电梯原来静止，突然向上加速时，身体想保持原来的静止状态，所以会感觉被压向地板。",
        ],
    },
    "derivative_intro_via_slope_limit_v1": {
        "diagnostic": [
            "我不确定h在几何上对应什么，也不清楚平均变化率怎样变成瞬时变化率。",
        ],
        "scaffold": [
            "这个不会啊。",
            "我看不懂。",
        ],
        "correct": [
            "时间区间无限缩小，平均变化率就越来越接近某一点的瞬时变化率。",
        ],
    },
    "derivative_limit_definition_v1": {
        "diagnostic": [
            "我不确定h在几何上对应什么，也不清楚平均变化率怎样变成瞬时变化率。",
        ],
        "scaffold": ["这个不会啊。", "我看不懂。"],
        "correct": [
            "时间区间无限缩小，平均变化率就越来越接近某一点的瞬时变化率。",
        ],
    },
}


def available_demo_targets(
    session: TeachingSession,
    mastery_threshold: float = 0.8,
) -> list[DemoTargetSignal]:
    """Return demo answer types supported by the current learning evidence.

    This only filters the presenter helper.  It does not select the Agent's
    teaching action or change the learner state.
    """

    state = session.state
    route_step = session.teaching_route.current_step() if session.teaching_route else None
    has_misconception = bool(
        state.misconceptions
        or state.misconception_states
        or state.misconception_confirmed
        or state.current_difficulty == "concept_misconception"
    )
    has_operational_difficulty = state.current_difficulty in {
        "symbol_notation",
        "calculation",
        "task_comprehension",
    }
    has_diagnostic_need = state.phase == "diagnosis" or state.current_difficulty == "unknown"
    has_transfer_evidence = bool(
        state.phase == "transfer"
        or (route_step is not None and route_step.kind == "transfer")
        or state.average_mastery() >= mastery_threshold
    )

    allowed: list[DemoTargetSignal] = ["auto"]
    if has_diagnostic_need:
        allowed.append("diagnostic")
    if state.phase in {"instruction", "repair"} or has_operational_difficulty or state.no_progress_rounds > 0:
        allowed.append("scaffold")
    if has_misconception or state.phase == "repair":
        allowed.append("misconception")
    allowed.append("correct")
    if has_transfer_evidence:
        allowed.append("transfer")
    return allowed


def get_verified_demo_replies(
    session: TeachingSession,
    target_signal: DemoTargetSignal,
) -> list[str]:
    """Return stable, previously verified replies for the presenter shortcut.

    These replies are deliberately kept outside :class:`DemoReplyGenerator`.
    The AI recommendation panel must contain only the current LLM response;
    this helper powers a separate, clearly labelled rehearsal panel.
    """

    if target_signal == "auto":
        return []
    current = session.turns[-1]
    skill_id = current.content_skill_id or (
        current.skill_plan.content_skill_id if current.skill_plan else ""
    )
    return list(VERIFIED_DEMO_REPLIES.get(skill_id, {}).get(target_signal, []))


class DemoReplySuggestion(BaseModel):
    """One selectable learner reply; the signal is for the presenter only."""

    # 这是模型返回的内部标识，不展示给学生；允许带有目标类型前缀，
    # 例如 misconception_1，避免无意义的格式错误阻断演示回答。
    suggestion_id: str = Field(min_length=1, max_length=40)
    label: str = Field(min_length=1, max_length=30)
    reply: str = Field(min_length=1, max_length=240)
    intended_signal: Literal[
        "confused",
        "partial",
        "diagnostic",
        "scaffold",
        "correct",
        "misconception",
        "transfer",
    ]

    @field_validator("reply")
    @classmethod
    def validate_reply(cls, value: str) -> str:
        value = " ".join(value.split()).strip()
        if not value:
            raise ValueError("推荐回答不能为空")
        return value


class DemoReplySet(BaseModel):
    """Validated LLM response for the current teaching turn."""

    suggestions: list[DemoReplySuggestion] = Field(min_length=3, max_length=4)


class DemoReplyGenerator:
    """Generate context-grounded demonstration replies for live demos."""

    def __init__(self, llm: OpenAICompatibleClient):
        self.llm = llm

    def generate(
        self,
        session: TeachingSession,
        target_signal: DemoTargetSignal = "auto",
        _strict_retry: bool = False,
    ) -> list[DemoReplySuggestion]:
        if not self.llm.available:
            raise LLMUnavailableError("未配置 LLM_API_KEY，无法生成 AI 推荐回答")

        current = session.turns[-1]
        recent_turns = session.turns[-4:]
        history = [
            {
                "round": turn.round_index,
                "teacher": turn.teacher_message,
                "student": turn.student_message,
            }
            for turn in recent_turns
        ]
        current_step = current.micro_step.model_dump(mode="json") if current.micro_step else {}
        route_step = (
            session.teaching_route.current_step().model_dump(mode="json")
            if session.teaching_route
            else {}
        )
        target_instruction = (
            "三条回答分别覆盖：表达困惑、部分理解、正确或有依据的理解；"
            "每条回答的 intended_signal 必须准确标注它实际表现的学生回答类型。"
            if target_signal == "auto"
            else (
                f"三条回答的目标类型必须全部是 {target_signal}，不能混入其他类型。"
                f"目标行为要求：{TARGET_BEHAVIOR[target_signal]}"
                + (
                    "这是一次严格重试，上一版混入了其他类型；请只返回目标类型的三条回答。"
                    if _strict_retry
                    else ""
                )
            )
        )
        data = self.llm.structured(
            (
                "你是一个教学演示辅助器，不是教师，也不负责判断答案。"
                "请根据当前教师问题，生成三条学生可能会说的、可以直接提交的短回答。"
                "三条回答必须围绕同一个当前情境和同一个问题，不能换题、补充新知识点或替教师讲解。"
                "回答要像真实学生，不要使用‘选项A/B’、不要写分析标签、不要直接复制教师问题。"
                + target_instruction
                + "不要编造当前上下文中没有的数字、公式、实验条件或结论。"
            ),
            (
                f"课程与目标：{session.goal.model_dump_json()}\n"
                    f"当前路线步骤：{route_step}\n"
                    f"当前单步教学上下文：{current_step}\n"
                    f"当前教师话语：{current.teacher_message}\n"
                    f"最近对话：{history}\n"
                    "请只返回 suggestions 数组；suggestion_id 使用简短且互不重复的字符串，"
                    "每项包含 suggestion_id、label、reply、intended_signal。"
            ),
            (
                '{"suggestions":[{"suggestion_id":"demo_1",'
                '"label":"演示回答一",'
                '"reply":"用学生口吻生成第一条当前情境回答",'
                '"intended_signal":"diagnostic|scaffold|misconception|correct|transfer|confused|partial"},'
                '{"suggestion_id":"demo_2",'
                '"label":"演示回答二",'
                '"reply":"用学生口吻生成第二条当前情境回答",'
                '"intended_signal":"diagnostic|scaffold|misconception|correct|transfer|confused|partial"},'
                '{"suggestion_id":"demo_3",'
                '"label":"演示回答三",'
                '"reply":"用学生口吻生成第三条当前情境回答",'
                '"intended_signal":"diagnostic|scaffold|misconception|correct|transfer|confused|partial"}]}'
            ),
            temperature=0.25,
        )
        result = DemoReplySet.model_validate(data)
        suggestions = self._deduplicate(result.suggestions)
        if target_signal == "auto":
            return suggestions
        mismatches = [
            item.intended_signal
            for item in suggestions
            if item.intended_signal != target_signal
        ]
        if mismatches:
            if not _strict_retry:
                return self.generate(session, target_signal, _strict_retry=True)
            raise ValueError(
                f"模型未能生成三条“{DEMO_TARGET_LABELS[target_signal]}”回答，"
                "请点击重新生成"
            )
        return suggestions

    @staticmethod
    def _deduplicate(items: list[DemoReplySuggestion]) -> list[DemoReplySuggestion]:
        unique: list[DemoReplySuggestion] = []
        seen: set[str] = set()
        for item in items:
            normalized = item.reply.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            unique.append(item)
        if len(unique) < 3:
            raise ValueError("LLM 推荐回答存在重复，未达到三条可选回答")
        return unique


def suggestion_rows(items: list[DemoReplySuggestion]) -> list[dict[str, Any]]:
    """Return a UI-safe representation without exposing internal signal labels."""

    return [{"id": item.suggestion_id, "label": item.label, "reply": item.reply} for item in items]
