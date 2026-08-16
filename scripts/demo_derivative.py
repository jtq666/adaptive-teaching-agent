"""可直接跑通的导数答辩演示脚本（只接受真实 API）。

现场输入顺序与本脚本的 ``ANSWERS`` 完全一致，重点展示模型承接公式困难、
连续降低任务粒度，并把导数概念迁移到位移函数。
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

CONTENT_SKILL = "derivative_intro_via_slope_limit_v1"
ANSWERS = [
    "时间区间无限缩小。",
    "这个不会啊。",
    "我看不懂。",
]


def main() -> None:
    client = OpenAICompatibleClient(get_llm_settings())
    if not client.available:
        raise SystemExit("未配置真实 API；导数答辩演示拒绝使用离线回退")

    goal = TeachingGoal(
        course="高等数学",
        topic="导数的极限定义：从平均变化率到瞬时变化率",
        objective="从平均变化率理解瞬时变化率，并解释极限如何定义导数",
        knowledge_points=["导数的极限定义"],
        success_criteria=["能写出导数的极限定义", "能迁移到位移函数"],
    )
    profile = StudentProfile(
        name="导数演示学生",
        level="中等",
        prior_knowledge=["函数", "斜率", "平均变化率", "平均速度"],
        learning_preferences=["逐步提示", "符号解释"],
    )

    with TemporaryDirectory(prefix="demo-derivative-") as directory:
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

    actions = [str(row["action"]) for row in records]
    assert actions.count("scaffold") >= 2, f"没有连续触发分层提示：{actions}"
    assert "correct" not in actions, f"把符号困难误判成概念纠正：{actions}"
    output = ROOT / "output" / "evaluations" / "demo_derivative_run.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"demo": "derivative", "model": client.model, "answers": ANSWERS, "turns": records},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"导数完整演示通过：已触发连续 scaffold 且未误判为 correct；结果：{output}")


if __name__ == "__main__":
    main()
