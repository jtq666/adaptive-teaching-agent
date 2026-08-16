"""Small real-API acceptance for the single-call live teaching path.

This script intentionally refuses to run in offline mode.  It checks the
observable contract rather than exact generated wording, because natural
language may vary between real model calls.
"""

# The script is also executable directly from its own directory.
# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import GENERIC_SKILLS, HybridTeachingAgent
from src.config import get_llm_settings
from src.llm import OpenAICompatibleClient
from src.models import StudentProfile, StudentState, TeachingGoal
from src.storage import SessionStore


def run_case(
    client: OpenAICompatibleClient,
    goal: TeachingGoal,
    profile: StudentProfile,
    replies: list[str],
) -> list[dict[str, object]]:
    with TemporaryDirectory() as directory:
        agent = HybridTeachingAgent(llm=client, store=SessionStore(Path(directory)))
        calls_before = client.total_calls
        session = agent.start_session(goal, profile, StudentState())
        initial_calls = client.total_calls - calls_before
        assert initial_calls >= 1
        rows: list[dict[str, object]] = []
        for answer in ["", *replies]:
            if answer:
                session = agent.handle_student_message(session, answer)
            turn = session.turns[-1]
            assert turn.content_skill_id
            assert agent.library.get(turn.content_skill_id).skill_type == "subject"
            assert turn.strategy_skill_id == GENERIC_SKILLS["adaptive"]
            assert turn.generation_audit.get("architecture") == "single_llm_adaptive_turn"
            assert turn.generation_audit.get("fallback") is False
            if answer:
                assert len(turn.llm_trace) == 1
            assert len(turn.teacher_message) <= 100
            assert turn.teacher_message.count("？") + turn.teacher_message.count("?") <= 1
            rows.append(
                {
                    "student": answer,
                    "content_skill": turn.content_skill_id,
                    "action": turn.action_type,
                    "difficulty": turn.difficulty_type,
                    "reply_length": len(turn.teacher_message),
                    "llm_calls": len(turn.llm_trace) if answer else initial_calls,
                }
            )
        return rows


def main() -> None:
    client = OpenAICompatibleClient(get_llm_settings())
    if not client.available:
        raise SystemExit("未配置真实 API：本验收拒绝使用离线回退")

    derivative = run_case(
        client,
        TeachingGoal(
            course="高等数学",
            topic="导数的极限定义",
            objective="从平均变化率理解瞬时变化率，并解释极限如何定义导数",
            knowledge_points=["平均变化率", "极限思想", "瞬时变化率"],
        ),
        StudentProfile(prior_knowledge=["函数", "斜率", "平均变化率", "平均速度"]),
        ["时间区间无限缩小", "这个不会啊", "我看不懂"],
    )
    derivative_actions = [row["action"] for row in derivative[1:]]
    assert "correct" not in derivative_actions
    assert all(row["action"] in {"diagnose", "scaffold", "explain"} for row in derivative[1:])

    physics = run_case(
        client,
        TeachingGoal(
            course="大学物理",
            topic="牛顿第一定律",
            objective="用公交车急刹车解释惯性，并区分保持运动与改变运动状态",
            knowledge_points=["惯性", "合力与运动变化"],
        ),
        StudentProfile(prior_knowledge=["速度", "力", "运动", "惯性"]),
        ["物体不受力为什么还能继续运动？", "运动必须有力维持。", "合力为零时保持匀速直线运动。"],
    )
    assert physics[1]["action"] != "correct"
    assert physics[2]["action"] == "correct"
    assert physics[3]["action"] != "correct"

    payload = {
        "architecture": "single_llm_adaptive_turn",
        "model": client.model,
        "derivative": derivative,
        "physics": physics,
    }
    output = Path(__file__).resolve().parents[1] / "output" / "evaluations" / "online_adaptive_teaching_acceptance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"已保存：{output}")


if __name__ == "__main__":
    main()
