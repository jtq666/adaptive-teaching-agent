"""Real-LLM regression for short answers that are correct in dialogue context."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import HybridTeachingAgent  # noqa: E402
from src.llm import OpenAICompatibleClient  # noqa: E402
from src.models import StudentProfile, StudentState, TeachingGoal  # noqa: E402
from src.skills import SkillLibrary  # noqa: E402
from src.storage import SessionStore  # noqa: E402


def main() -> None:
    client = OpenAICompatibleClient()
    if not client.available:
        raise SystemExit("未配置真实 LLM_API_KEY，拒绝以离线规则替代验收")

    goal = TeachingGoal(
        course="程序设计",
        topic="二分查找边界条件",
        objective="理解左闭右闭区间定义，并由此推导循环条件",
        knowledge_points=["区间定义", "循环不变量", "边界更新"],
    )
    profile = StudentProfile(name="上下文回归学生", level="中等", prior_knowledge=["数组", "while循环"])
    initial = StudentState(
        mastery={point: 0.35 for point in goal.knowledge_points},
        next_focus="区间定义",
    )
    answers = ["有元素", "1个元素只有", "1个下标啊"]

    with TemporaryDirectory(prefix="teaching-agent-context-") as directory:
        agent = HybridTeachingAgent(
            library=SkillLibrary(),
            llm=client,
            store=SessionStore(Path(directory)),
        )
        session = agent.start_session(goal, profile, initial)
        # This regression targets contextual state judging, so keep the
        # student answer sequence aligned with one explicit teacher question.
        # The initial LLM-generated message is still checked for being a
        # single question; only the next prompt is fixed for reproducibility.
        assert session.turns[0].teacher_message.count("？") + session.turns[0].teacher_message.count("?") <= 1
        assert session.turns[0].micro_step is not None
        controlled_question = (
            "在左闭右闭区间 [left, right] 中，若 left = 2、right = 2，区间里还有几个元素？"
        )
        session.turns[0].teacher_message = controlled_question
        initial_average = session.state.average_mastery()
        turns = []
        for answer in answers:
            # Keep the semantic probe aligned with the same minimal question;
            # the generated next teacher message is still checked below, but
            # it must not change the meaning of this state-judging regression.
            session.turns[-1].teacher_message = controlled_question
            session = agent.handle_student_message(session, answer)
            turn = session.turns[-1]
            assert turn.micro_step is not None, turns
            turns.append(
                {
                    "student": answer,
                    "skill": turn.selected_skill_id,
                    "action": turn.action_type,
                    "signal": session.state.evidence[-1].signal_type,
                    "question": session.turns[-2].teacher_message if len(session.turns) > 1 else "",
                    "micro_step": turn.micro_step.model_dump(mode="json"),
                    "response_mode": turn.micro_step.response_mode,
                    "option_count": len(turn.micro_step.options),
                    "generation_audit": dict(turn.generation_audit),
                    "understanding_signals": list(session.state.understanding_signals),
                    "misconceptions": [item.label for item in session.state.misconceptions],
                    "mastery": dict(session.state.mastery),
                }
            )
            assert turn.action_type != "correction", turns[-1]
            assert session.state.evidence[-1].signal_type != "negative", turns[-1]
            assert not session.state.misconceptions, turns[-1]
            assert turn.teacher_message.count("？") + turn.teacher_message.count("?") <= 1, turns[-1]
        assert session.state.average_mastery() > initial_average, {
            "initial": initial_average,
            "final": session.state.average_mastery(),
            "turns": turns,
        }

    artifact = {"mode": "real_api_context_short_answer", "model": client.model, "turns": turns}
    output = Path(__file__).resolve().parents[1] / ".e2e-runtime" / "online-context-acceptance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"真实 API 上下文短回答验收通过：{len(turns)} 轮，结果已写入 {output}")


if __name__ == "__main__":
    main()
