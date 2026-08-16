"""真实 API 验收：牛顿第一定律路线连续性与单任务契约。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import HybridTeachingAgent  # noqa: E402
from src.models import StudentProfile, StudentState, TeachingGoal  # noqa: E402
from src.skills import SkillLibrary  # noqa: E402
from src.storage import SessionStore  # noqa: E402


def answer_for(session) -> str:
    step = session.turns[-1].micro_step
    prompt = f"{step.focus} {step.context} {step.requested_target}" if step else ""
    if "参考系" in prompt:
        return "惯性参考系是牛顿第一定律成立的参考系；匀速直线运动的汽车可以近似看作惯性参考系。"
    if "合力" in prompt or "运动变化" in prompt:
        if "公交车受到" in prompt:
            return "公交车受到的合力方向与运动方向相反，所以速度逐渐减小，运动状态发生改变。"
        return "刹车时乘客受到的合力向后，它使乘客的速度减小，因此改变了乘客的运动状态。"
    if "方向" in prompt or "倾" in prompt or "刹车" in prompt:
        return "公交车刹车时身体会相对车厢向前，因为身体倾向于保持原来的运动状态。"
    return "惯性是物体保持原来静止或匀速直线运动状态的性质，它不是维持运动的一种力。"


def main() -> None:
    goal = TeachingGoal(
        course="大学物理",
        topic="牛顿第一定律",
        objective="区分惯性与力，并能解释合力如何改变运动状态",
        knowledge_points=["惯性", "合力与运动变化", "惯性参考系"],
    )
    state = StudentState(
        mastery={point: 0.35 for point in goal.knowledge_points},
        next_focus="惯性",
    )
    records: list[dict] = []
    with TemporaryDirectory(prefix="teaching-agent-physics-") as directory:
        agent = HybridTeachingAgent(library=SkillLibrary(), store=SessionStore(Path(directory)))
        if not agent.llm.available:
            raise SystemExit("未配置真实 LLM_API_KEY，拒绝使用离线规则替代验收")
        session = agent.start_session(
            goal,
            StudentProfile(name="真实物理验收学生", level="基础薄弱", learning_preferences=["具体例子"]),
            state,
        )
        assert session.teaching_route is not None
        previous_target = ""
        for index in range(5):
            reply = "我不知道，物体不受力时为什么还能继续运动。" if index == 0 else answer_for(session)
            session = agent.handle_student_message(session, reply)
            turn = session.turns[-1]
            step = turn.micro_step
            assert step is not None
            assert not agent._contains_multiple_requests(turn.teacher_message), turn.teacher_message
            assert not agent._contains_multiple_requests(step.requested_target), step.requested_target
            assert turn.teacher_message.count("？") + turn.teacher_message.count("?") <= 1
            if previous_target:
                assert step.requested_target != previous_target, {
                    "round": turn.round_index,
                    "target": step.requested_target,
                }
            previous_target = step.requested_target
            records.append(
                {
                    "round": turn.round_index,
                    "student": reply,
                    "route_step": session.teaching_route.current_step().title,
                    "focus": step.focus,
                    "requested_target": step.requested_target,
                    "teacher_message": turn.teacher_message,
                    "content_skill": turn.content_skill_id,
                    "strategy_skill": turn.strategy_skill_id,
                    "fallback": turn.fallback_reason,
                }
            )

    output = PROJECT_ROOT / ".e2e-runtime" / "online-physics-route-acceptance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"mode": "real_api_physics_route", "turns": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"真实物理路线验收通过：{len(records)} 轮，结果已写入 {output}")


if __name__ == "__main__":
    main()
