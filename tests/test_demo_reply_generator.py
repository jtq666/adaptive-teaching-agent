from pathlib import Path

import pytest

from src.agent import HybridTeachingAgent
from src.demo_reply_generator import DemoReplyGenerator
from src.llm import LLMUnavailableError, OpenAICompatibleClient
from src.models import StudentProfile, StudentState, TeachingGoal
from src.skills import SkillLibrary
from src.storage import SessionStore


class DemoReplyClient(OpenAICompatibleClient):
    def __init__(self, payload):
        self.client = object()
        self.model = "demo-reply-test"
        self.payload = payload
        self.last_user = ""

    @property
    def available(self):
        return True

    def structured(self, system, user, schema_hint, temperature=0.0):
        self.last_user = user
        return self.payload


def make_session(tmp_path: Path):
    agent = HybridTeachingAgent(
        library=SkillLibrary(),
        llm=OpenAICompatibleClient(settings={"api_key": ""}),
        store=SessionStore(tmp_path),
        settings={"max_rounds": 8, "mastery_threshold": 0.8, "no_progress_limit": 3},
    )
    return agent.start_session(
        TeachingGoal(
            course="大学物理",
            topic="牛顿第一定律",
            objective="理解惯性与运动状态变化",
            knowledge_points=["惯性", "合力与运动变化"],
        ),
        StudentProfile(name="演示学生", level="中等"),
        StudentState(mastery={"惯性": 0.3, "合力与运动变化": 0.3}),
    )


def payload():
    return {
        "suggestions": [
            {
                "suggestion_id": "confused",
                "label": "我还不确定",
                "reply": "我还不确定，能不能再给我一个生活中的例子？",
                "intended_signal": "confused",
            },
            {
                "suggestion_id": "partial",
                "label": "我理解了一部分",
                "reply": "我觉得惯性和物体原来的运动状态有关，但还说不清力的作用。",
                "intended_signal": "partial",
            },
            {
                "suggestion_id": "correct",
                "label": "我试着解释",
                "reply": "惯性是物体保持原来运动状态的性质，合力才会让运动状态发生变化。",
                "intended_signal": "correct",
            },
        ]
    }


def test_generator_uses_current_context_and_returns_three_selectable_replies(tmp_path):
    client = DemoReplyClient(payload())
    suggestions = DemoReplyGenerator(client).generate(make_session(tmp_path))

    assert len(suggestions) == 3
    assert {item.intended_signal for item in suggestions} == {"confused", "partial", "correct"}
    assert "当前教师话语" in client.last_user
    assert "惯性" in client.last_user


def test_generator_rejects_duplicate_replies(tmp_path):
    duplicate_payload = payload()
    duplicate_payload["suggestions"][1]["reply"] = duplicate_payload["suggestions"][0]["reply"]
    duplicate_payload["suggestions"][2]["reply"] = duplicate_payload["suggestions"][0]["reply"]
    client = DemoReplyClient(duplicate_payload)

    with pytest.raises(ValueError, match="重复"):
        DemoReplyGenerator(client).generate(make_session(tmp_path))


def test_generator_requires_real_llm():
    client = OpenAICompatibleClient(settings={"api_key": ""})
    generator = DemoReplyGenerator(client)
    with pytest.raises(LLMUnavailableError, match="无法生成"):
        generator.generate(
            # The availability check runs before the session is inspected.
            object()  # type: ignore[arg-type]
        )
