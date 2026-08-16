"""真实 API 单步教学契约验收，不替换或改写模型生成的教师问题。

运行：python scripts/online_single_step_acceptance.py
"""

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

SAFE_GUARD_FALLBACKS = {
    "repeated_target_guard",
    "completed_focus_revisit",
    "fast_demo_deterministic_fallback",
}


def main() -> None:
    client = OpenAICompatibleClient()
    if not client.available:
        raise SystemExit("未配置真实 LLM_API_KEY，拒绝以离线规则替代验收")

    goal = TeachingGoal(
        course="程序设计",
        topic="二分查找边界条件",
        objective="理解区间定义并逐步验证边界更新",
        knowledge_points=["区间定义", "循环不变量", "边界更新"],
    )
    profile = StudentProfile(name="真实单步验收学生", level="中等", prior_knowledge=["数组", "循环"])
    state = StudentState(
        mastery={point: 0.35 for point in goal.knowledge_points},
        next_focus="区间定义",
    )
    answers = [
        "我不确定，请给我一个具体例子。",
        "我还是不太确定，请继续保持这个例子。",
        "我能说出一部分，但还不能解释依据。",
    ]
    turns: list[dict[str, object]] = []

    with TemporaryDirectory(prefix="teaching-agent-single-step-") as directory:
        agent = HybridTeachingAgent(
            library=SkillLibrary(),
            llm=client,
            store=SessionStore(Path(directory)),
        )
        session = agent.start_session(goal, profile, state)
        for answer in answers:
            previous = session.turns[-1]
            previous_step = previous.micro_step
            assert previous_step is not None
            session = agent.handle_student_message(session, answer)
            latest = session.turns[-1]
            current_step = latest.micro_step
            assert current_step is not None
            message = latest.teacher_message
            assert message.count("？") + message.count("?") <= 1, message
            assert not agent._contains_internal_task_language(message), message
            assert not agent._contains_multiple_value_scenarios(message), message
            assert not agent._focus_stage_mismatch(
                current_step.focus,
                current_step.requested_target,
            ), current_step.model_dump()
            if latest.generation_audit.get("context_locked"):
                assert current_step.context == previous_step.context
                assert current_step.representation == previous_step.representation
            turns.append(
                {
                    "student": answer,
                    "focus": current_step.focus,
                    "requested_target": current_step.requested_target,
                    "teacher_message": message,
                    "context_locked": latest.generation_audit.get("context_locked"),
                    "generation_audit": dict(latest.generation_audit),
                    "fallback_reason": latest.fallback_reason,
                    "review_issues": latest.teacher_review.issues if latest.teacher_review else [],
                }
            )

    safe_guard_count = sum(
        1
        for turn in turns
        if turn["fallback_reason"]
        and turn["generation_audit"].get("fallback") in SAFE_GUARD_FALLBACKS
    )
    fallback_count = sum(
        1
        for turn in turns
        if turn["fallback_reason"]
        and turn["generation_audit"].get("fallback") not in SAFE_GUARD_FALLBACKS
    )
    assert fallback_count == 0, {
        "fallback_count": fallback_count,
        "safe_guard_count": safe_guard_count,
        "turns": turns,
    }

    output = PROJECT_ROOT / ".e2e-runtime" / "online-single-step-acceptance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "mode": "real_api_single_step_contract",
                "model": client.model,
                "fallback_count": fallback_count,
                "safe_guard_count": safe_guard_count,
                "turns": turns,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"真实 API 单步契约验收通过：{len(turns)} 轮，结果已写入 {output}")


if __name__ == "__main__":
    main()
