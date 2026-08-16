"""Run one real-API acceptance check for AI demo replies."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import HybridTeachingAgent  # noqa: E402
from src.demo_reply_generator import DemoReplyGenerator  # noqa: E402
from src.llm import LLMUnavailableError, OpenAICompatibleClient  # noqa: E402
from src.models import StudentProfile, StudentState, TeachingGoal  # noqa: E402
from src.skills import SkillLibrary  # noqa: E402
from src.storage import SessionStore  # noqa: E402


def main() -> None:
    client = OpenAICompatibleClient()
    if not client.available:
        raise SystemExit("真实 API 验收失败：未配置 LLM_API_KEY")
    runtime_dir = Path(__file__).resolve().parents[1] / ".e2e-runtime" / "online-demo-reply"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    agent = HybridTeachingAgent(
        library=SkillLibrary(),
        llm=client,
        store=SessionStore(runtime_dir / "sessions"),
    )
    session = agent.start_session(
        TeachingGoal(
            course="大学物理",
            topic="牛顿第一定律",
            objective="理解惯性与合力对运动状态的作用",
            knowledge_points=["惯性", "合力与运动变化"],
        ),
        StudentProfile(name="AI 演示学生", level="中等", response_preference="open"),
        StudentState(mastery={"惯性": 0.3, "合力与运动变化": 0.3}),
    )
    try:
        suggestions = DemoReplyGenerator(client).generate(session)
    except LLMUnavailableError as exc:
        raise SystemExit(f"真实 API 验收失败：{exc}") from exc
    if len(suggestions) < 3 or len({item.reply for item in suggestions}) != len(suggestions):
        raise SystemExit("真实 API 验收失败：推荐回答不足三条或存在重复")
    artifact = {
        "teacher_message": session.turns[-1].teacher_message,
        "suggestions": [item.model_dump(mode="json") for item in suggestions],
        "model": client.model,
        "llm_calls": client.total_calls,
    }
    output = Path(__file__).resolve().parents[1] / ".e2e-runtime" / "online-demo-reply-acceptance.json"
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"真实 AI 推荐回答验收通过：{len(suggestions)} 条，调用 {client.total_calls} 次")
    print(output)


if __name__ == "__main__":
    main()
