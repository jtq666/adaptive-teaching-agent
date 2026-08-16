"""可直接跑通的物理答辩演示脚本（只接受真实 API）。

现场输入顺序与本脚本的 ``ANSWERS`` 完全一致。脚本本身不替模型选择
教学动作，只检查真实回合是否保存了内容 Skill、动作、证据和调用审计。
"""

# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.agent import GENERIC_SKILLS, HybridTeachingAgent
from src.config import get_llm_settings
from src.llm import OpenAICompatibleClient
from src.models import StudentProfile, StudentState, TeachingGoal
from src.storage import SessionStore

CONTENT_SKILL = "newtons_first_law_via_engineering_examples_v1"
ANSWERS = [
    "我不确定，物体没有受到水平方向的力，为什么还会继续向前运动？",
    "我认为只要物体在运动，就一定需要一个向前的力来维持运动。",
    "现在我明白了：合力为零时，物体会保持静止或匀速直线运动；惯性是保持这种状态的性质，不是力。",
    "公交车突然加速时，我会相对车向后仰，因为身体想保持原来的静止状态，而车向前加速。",
    "公交车向右转弯时，我会相对车向左偏，因为身体想保持原来向前的直线运动状态。",
    "电梯突然向上加速时，我会感觉被压向地板，因为身体想保持原来的静止状态。",
    "不会。电梯匀速上升时合力为零，身体保持匀速运动，不会因为加速而有额外的压迫感。",
]


def main() -> None:
    client = OpenAICompatibleClient(get_llm_settings())
    if not client.available:
        raise SystemExit("未配置真实 API；物理答辩演示拒绝使用离线回退")

    goal = TeachingGoal(
        course="大学物理",
        topic="牛顿第一定律：公交车急刹车与惯性",
        objective="用公交车急刹车解释惯性，并区分保持运动与改变运动状态",
        knowledge_points=["惯性与运动状态变化"],
        success_criteria=["能解释急刹车现象", "能把惯性迁移到转弯情境"],
    )
    profile = StudentProfile(
        name="物理演示学生",
        level="中等",
        prior_knowledge=["速度", "力", "运动", "惯性"],
        learning_preferences=["具体例子", "逐步提示"],
    )

    with TemporaryDirectory(prefix="demo-physics-") as directory:
        agent = HybridTeachingAgent(llm=client, store=SessionStore(Path(directory)))
        session = agent.start_session(
            goal,
            profile,
            StudentState(mastery={point: 0.35 for point in goal.knowledge_points}),
            available_skill_ids=[CONTENT_SKILL],
        )
        records: list[dict[str, object]] = []
        for index, answer in enumerate(ANSWERS, start=1):
            session = agent.handle_student_message(session, answer)
            turn = session.turns[-1]
            assert turn.content_skill_id == CONTENT_SKILL
            assert turn.strategy_skill_id == GENERIC_SKILLS["adaptive"]
            assert turn.generation_audit.get("architecture") == "single_llm_adaptive_turn"
            if turn.generation_audit.get("fallback"):
                raise SystemExit(
                    f"第{index}轮真实 API 未返回有效教学结果："
                    f"{turn.generation_audit.get('fallback_error', '请检查 API 余额或密钥')}"
                )
            assert len(turn.llm_trace) == 1
            assert len(turn.teacher_message) <= 100
            assert turn.teacher_message.count("？") + turn.teacher_message.count("?") <= 1
            row = {
                "round": index,
                "student": answer,
                "action": turn.action_type,
                "difficulty": turn.difficulty_type,
                "teacher": turn.teacher_message,
                "mastery": dict(turn.state_after.mastery),
                "status": str(session.status),
            }
            records.append(row)
            print(f"第{index}轮 | {turn.action_type:8} | {turn.difficulty_type:20} | {turn.teacher_message}")
            if str(session.status) != "active":
                break

    actions = [str(row["action"]) for row in records]
    assert "correct" in actions, f"没有触发误解纠正：{actions}"
    assert "transfer" in actions, f"没有触发迁移验证：{actions}"
    correction_index = actions.index("correct")
    assert any(action != "correct" for action in actions[correction_index + 1:]), actions
    output = ROOT / "output" / "evaluations" / "demo_physics_run.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"demo": "physics", "model": client.model, "answers": ANSWERS, "turns": records},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"物理完整演示通过：已触发 correct、transfer；结果：{output}")


if __name__ == "__main__":
    main()
