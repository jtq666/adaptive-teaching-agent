"""真实 API 回归：验证连续正确回答后微步骤不会重复或回退。

运行：python scripts/online_progression_acceptance.py
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


def main() -> None:
    client = OpenAICompatibleClient()
    if not client.available:
        raise SystemExit("未配置真实 LLM_API_KEY，拒绝以离线规则替代验收")

    goal = TeachingGoal(
        course="程序设计",
        topic="二分查找边界条件",
        objective="从区间定义逐步推导循环不变量和边界更新",
        knowledge_points=["区间定义", "循环不变量", "边界更新"],
    )
    profile = StudentProfile(name="真实推进验收学生", level="中等", prior_knowledge=["数组", "循环"])
    state = StudentState(
        mastery={point: 0.35 for point in goal.knowledge_points},
        next_focus="区间定义",
    )
    answer_bank = {
        "interval": "区间定义就是规定当前搜索范围，以及 left 和 right 两个边界是否包含在范围内。",
        "initial_bounds": "初始时 left = 0，right = 4，因为数组下标从 0 到 4，左闭右闭区间包含两端。",
        "boundary_meaning": "left 表示当前搜索范围的左端，right 表示当前搜索范围的右端；左闭右闭时两端都包含。",
        "single_element": "当 left = 2、right = 2 时，左闭右闭区间只包含下标 2 对应的一个元素。",
        "equality_loop": "当 left == right 时，区间仍包含一个元素，所以循环条件必须允许继续检查这个位置，例如使用 left <= right。",
        "match_case": "当 nums[mid] 等于 target 时，目标已经找到，查找应当结束，不需要继续更新 left 或 right。",
        "invariant": "如果目标值存在，那么每一轮循环开始前，它一定仍然位于当前搜索区间 [left, right] 中。",
        "right": "应该更新为 right = mid - 1，因为 nums[mid] 已经检查过且大于目标值，应排除 mid。",
        "left": "应该更新为 left = mid + 1，因为 nums[mid] 已经检查过且小于目标值，应排除 mid。",
        "boundary": "边界更新就是根据比较结果排除已经检查且不可能包含目标值的位置，同时保持搜索区间不变量。",
    }

    def answer_for_prompt(focus: str, prompt: str, fallback_index: int) -> str:
        normalized = prompt.lower().replace(" ", "")
        if "初始" in prompt and ("下标" in prompt or "索引" in prompt or "具体" in prompt):
            return answer_bank["initial_bounds"]
        if (
            "left == right" in normalized
            or "left=right" in normalized
            or "left和right相等" in normalized
        ):
            return answer_bank["equality_loop"]
        if "等于 target" in prompt or "等于target" in normalized or "== target" in prompt:
            return answer_bank["match_case"]
        if "仍然满足" in prompt or "保持循环不变量" in prompt:
            return answer_bank["invariant"]
        if ">target" in normalized or "大于目标" in prompt:
            return answer_bank["right"]
        if "<target" in normalized or "小于目标" in prompt:
            return answer_bank["left"]
        if "target 大于 mid" in prompt or "target大于mid" in normalized:
            return answer_bank["left"]
        if "target 小于 mid" in prompt or "target小于mid" in normalized:
            return answer_bank["right"]
        if "目标值比" in prompt and "小" in prompt:
            return answer_bank["right"]
        if "目标值比" in prompt and "大" in prompt:
            return answer_bank["left"]
        if "包含几个元素" in prompt or "包含哪些" in prompt or "[2, 2]" in prompt:
            return answer_bank["single_element"]
        if "是否包含" in prompt or ("right" in normalized and "包含" in prompt):
            return answer_bank["boundary_meaning"]
        if "left" in prompt and "right" in prompt and ("分别" in prompt or "含义" in prompt or "代表" in prompt):
            return answer_bank["boundary_meaning"]
        if "循环不变量" in prompt or "不变量" in prompt or "不变量" in focus:
            return answer_bank["invariant"]
        if "区间" in focus or "定义" in focus or "区间定义" in prompt:
            return answer_bank["interval"]
        if "right" in normalized or "右边界" in prompt:
            return answer_bank["right"]
        if "left" in normalized or "左边界" in prompt:
            return answer_bank["left"]
        if "循环不变量" in prompt or "不变量" in prompt:
            return answer_bank["invariant"]
        if (
            "下一轮搜索区间" in prompt
            or "下一步" in prompt and "区间" in prompt
            or "搜索区间" in prompt and "新" in prompt
        ):
            if ">" in prompt or "大于" in prompt:
                return answer_bank["right"]
            if "<" in prompt or "小于" in prompt:
                return answer_bank["left"]
        if "区间定义" in prompt or ("区间" in prompt and "包含" in prompt):
            return answer_bank["interval"]
        return answer_bank["boundary"] if fallback_index >= 2 else answer_bank["invariant"]
    turns: list[dict[str, object]] = []

    with TemporaryDirectory(prefix="teaching-agent-progression-") as directory:
        agent = HybridTeachingAgent(library=SkillLibrary(), store=SessionStore(Path(directory)))
        session = agent.start_session(goal, profile, state)
        previous_message = session.turns[-1].teacher_message
        for index in range(4):
            previous_step = session.turns[-1].micro_step
            assert previous_step is not None
            answer = answer_for_prompt(previous_step.focus, previous_step.requested_target, index)
            session = agent.handle_student_message(session, answer)
            latest = session.turns[-1]
            step = latest.micro_step
            assert step is not None
            assert step.requested_target != previous_step.requested_target, {
                "round": latest.round_index,
                "previous_target": previous_step.requested_target,
                "current_target": step.requested_target,
                "answer": answer,
            }
            assert latest.teacher_message != previous_message, {
                "round": latest.round_index,
                "message": latest.teacher_message,
            }
            assert not agent._focus_stage_mismatch(step.focus, step.requested_target), {
                "round": latest.round_index,
                "focus": step.focus,
                "target": step.requested_target,
                "message": latest.teacher_message,
            }
            assert not agent._contains_internal_task_language(latest.teacher_message), latest.teacher_message
            assert latest.teacher_message.count("？") + latest.teacher_message.count("?") <= 1, latest.teacher_message
            turns.append(
                {
                    "round": latest.round_index,
                    "student": answer,
                    "focus": step.focus,
                    "requested_target": step.requested_target,
                    "teacher_message": latest.teacher_message,
                    "context_locked": latest.generation_audit.get("context_locked"),
                    "fallback_reason": latest.fallback_reason,
                    "next_focus": session.state.next_focus,
                    "fallback_code": latest.generation_audit.get("fallback"),
                    "latest_evidence": (
                        session.state.evidence[-1].model_dump(mode="json")
                        if session.state.evidence
                        else None
                    ),
                    "knowledge_levels": {
                        point: knowledge.last_evidence_level
                        for point, knowledge in session.state.knowledge_states.items()
                    },
                }
            )
            previous_message = latest.teacher_message

    output = PROJECT_ROOT / ".e2e-runtime" / "online-progression-acceptance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"mode": "real_api_progression", "model": client.model, "turns": turns}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"真实 API 连续推进验收通过：{len(turns)} 轮，结果已写入 {output}")


if __name__ == "__main__":
    main()
