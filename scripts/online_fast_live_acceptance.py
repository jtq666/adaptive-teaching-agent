"""Measure one real-API live teaching turn with the fast demo budget."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import HybridTeachingAgent  # noqa: E402
from src.config import get_agent_settings  # noqa: E402
from src.llm import OpenAICompatibleClient  # noqa: E402
from src.models import StudentProfile, StudentState, TeachingGoal  # noqa: E402
from src.skills import SkillLibrary  # noqa: E402
from src.storage import SessionStore  # noqa: E402


def main() -> None:
    client = OpenAICompatibleClient()
    if not client.available:
        raise SystemExit("快速实时验收失败：未配置 LLM_API_KEY")
    settings = get_agent_settings()
    settings.update(
        {
            "fast_demo_mode": True,
            "max_state_reviews": 1,
            "state_review_call_budget": 1,
            "semantic_selector_margin": 0.0,
            "structured_output_retries": 0,
        }
    )
    with TemporaryDirectory(prefix="teaching-agent-fast-live-") as directory:
        agent = HybridTeachingAgent(
            library=SkillLibrary(),
            llm=client,
            store=SessionStore(Path(directory) / "sessions"),
            settings=settings,
        )
        session = agent.start_session(
            TeachingGoal(
                course="大学物理",
                topic="牛顿第一定律",
                objective="理解惯性与运动状态变化",
                knowledge_points=["惯性", "合力与运动变化"],
            ),
            StudentProfile(name="快速演示学生", level="中等", response_preference="open"),
            StudentState(mastery={"惯性": 0.3, "合力与运动变化": 0.3}),
        )
        calls_before = client.total_calls
        started = time.perf_counter()
        session = agent.handle_student_message(session, "我不知道物体不受力时为什么还能继续运动。")
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        calls = client.total_calls - calls_before
        if calls > 3:
            raise SystemExit(f"快速实时验收失败：单轮调用 {calls} 次，超过 3 次预算")
        print(f"快速实时验收通过：单轮 {calls} 次 LLM 调用，耗时 {elapsed_ms} ms，状态 {session.status}")


if __name__ == "__main__":
    main()
