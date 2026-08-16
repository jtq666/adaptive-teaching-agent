"""使用真实 LLM API 验收多轮教学语义，不写入正式 output。

运行：python scripts/online_semantic_acceptance.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

# Allow the documented ``python scripts/online_semantic_acceptance.py`` command
# to resolve the project package without requiring an editable installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main() -> None:
    from src.agent import HybridTeachingAgent
    from src.models import StudentProfile, StudentState, TeachingGoal
    from src.skills import SkillLibrary
    from src.storage import SessionStore

    def answer_for_current_step(current_session) -> str:
        step = current_session.turns[-1].micro_step
        if step is None:
            return "我会根据当前搜索区间来判断边界是否包含。"
        prompt = f"{step.focus} {step.requested_target}".lower().replace(" ", "")
        if "循环不变量" in prompt or "不变量" in prompt:
            return "如果目标值存在，那么每一轮循环开始前，它一定仍然位于当前搜索区间 [left, right] 中。"
        if ">target" in prompt or "大于目标" in prompt:
            return "应该更新为 right = mid - 1，因为 nums[mid] 已经检查过且大于目标值，应排除 mid。"
        if "<target" in prompt or "小于目标" in prompt:
            return "应该更新为 left = mid + 1，因为 nums[mid] 已经检查过且小于目标值，应排除 mid。"
        if "边界更新" in prompt:
            return "边界更新就是排除已经检查且不可能包含目标值的位置，同时保持搜索区间不变量。"
        return "因为 left == right 时左闭右闭区间仍然包含一个元素，所以它不能被当成空区间。"

    goal = TeachingGoal(
        course="程序设计",
        topic="二分查找边界条件",
        objective="从区间定义推导循环条件和边界更新",
        knowledge_points=["区间定义", "循环不变量", "边界更新"],
    )
    initial = StudentState(
        mastery={point: 0.35 for point in goal.knowledge_points},
        next_focus="区间定义",
    )
    with TemporaryDirectory() as directory:
        agent = HybridTeachingAgent(
            library=SkillLibrary(),
            store=SessionStore(Path(directory)),
        )
        if not agent.llm.available:
            raise SystemExit("未配置 LLM_API_KEY，无法运行在线语义验收")
        session = agent.start_session(
            goal,
            StudentProfile(name="在线验收学生", level="中等", prior_knowledge=["while循环", "数组"]),
            initial,
        )

        session = agent.handle_student_message(session, "不用再检查了")
        assert session.state.mastery["区间定义"] < 0.35
        assert session.state.mastery["循环不变量"] == 0.35
        assert session.state.mastery["边界更新"] == 0.35
        assert session.state.misconceptions
        assert session.turns[-1].action_type == "diagnostic"

        before = dict(session.state.mastery)
        session = agent.handle_student_message(session, answer_for_current_step(session))
        assert session.state.mastery["区间定义"] > before["区间定义"]
        # The single-step contract updates the current focus only; the next
        # knowledge point is deliberately left for a later teacher turn.
        assert session.state.mastery["循环不变量"] == before["循环不变量"]
        assert session.state.mastery["边界更新"] == before["边界更新"]
        assert not session.state.misconceptions
        assert session.state.no_progress_rounds == 0
        assert session.turns[-1].selected_skill_id == "binary_search_boundary_by_interval_definition"

        before_boundary = session.state.mastery["边界更新"]
        before_next_step = dict(session.state.mastery)
        session = agent.handle_student_message(session, answer_for_current_step(session))
        latest = session.turns[-1]
        changed_points = [
            point for point, value in session.state.mastery.items() if value > before_next_step[point]
        ]
        # Which adjacent point is current depends on the live model's semantic
        # focus choice. The invariant is that at most the current focus moves,
        # and at least one relevant point receives evidence.
        assert changed_points
        assert len(changed_points) == 1
        assert session.state.mastery["边界更新"] >= before_boundary
        assert latest.teacher_message.rstrip().endswith(("？", "?", "。", "！"))
        assert "如果 righ" not in latest.teacher_message
        assert "这个结论本身没错" not in latest.teacher_message

    print("在线语义验收通过：状态方向、知识点映射、误解消退、Skill 回切和回复完整性均正常")


if __name__ == "__main__":
    main()
