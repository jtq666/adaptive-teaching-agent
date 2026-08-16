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
    "diagnostic": "教学诊断：定位学生卡点",
    "scaffold": "分层提示：降低下一步难度",
    "misconception": "误解纠正：暴露错误观点",
    "correct": "正确理解：给出完整依据",
    "transfer": "迁移验证：换情境应用",
}
DEMO_SIGNAL_LABELS = {
    **DEMO_TARGET_LABELS,
    "confused": "表达困惑",
    "partial": "部分理解",
}


class DemoReplySuggestion(BaseModel):
    """One selectable learner reply; the signal is for the presenter only."""

    suggestion_id: str = Field(min_length=1, max_length=12)
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
    """Generate context-grounded demonstration replies with one LLM call."""

    def __init__(self, llm: OpenAICompatibleClient):
        self.llm = llm

    def generate(
        self,
        session: TeachingSession,
        target_signal: DemoTargetSignal = "auto",
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
            "如果当前问题适合暴露典型误解，可以用一条自然的错误理解替代‘表达困惑’。"
            if target_signal == "auto"
            else (
                f"本次演示目标是“{DEMO_TARGET_LABELS[target_signal]}”。"
                "三条回答都要围绕当前教师问题，用真实学生口吻自然地表现这个目标，"
                "不要直接说‘我要触发某个教学动作’，也不要改变当前知识点。"
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
                "请只返回 suggestions 数组，每项包含 suggestion_id、label、reply、intended_signal。"
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
            temperature=0.35,
        )
        result = DemoReplySet.model_validate(data)
        return self._deduplicate(result.suggestions)

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
