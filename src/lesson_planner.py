"""Create and advance a bounded teaching route for one public Agent.

The route contains curriculum targets, not subject answers.  It is generated
once from the user-provided goal and then constrains all turn-level decisions.
"""

from __future__ import annotations

import re

from src.llm import LLMUnavailableError, OpenAICompatibleClient
from src.models import (
    StudentState,
    TeachingGoal,
    TeachingRoute,
    TeachingRouteStep,
)


class LessonPlanner:
    def __init__(self, llm: OpenAICompatibleClient):
        self.llm = llm

    def build(self, goal: TeachingGoal) -> TeachingRoute:
        if self.llm.available:
            try:
                data = self.llm.structured(
                    (
                        "你是教学路线规划器。根据用户明确给出的知识点，生成一条简短、递进的教学路线。"
                        "每个知识点只能出现一次，不得增加新的学科知识点，不得写问题、答案、例题或教学话语。"
                        "learning_target 只描述该步骤希望学生表现出的能力。"
                    ),
                    (
                        f"课程与教学目标：{goal.model_dump_json()}\n"
                        "请严格按 knowledge_points 的先后关系规划；最后增加一个独立迁移验证步骤。"
                    ),
                    (
                        '{"steps":[{"knowledge_point":"原知识点",'
                        '"learning_target":"学生在这一步应表现出的单一能力",'
                        '"evidence_requirement":"correct或explained"}]}'
                    ),
                    temperature=0.0,
                )
                targets = self._validated_targets(goal, data)
                if targets:
                    return self._route(goal, targets, source="llm")
            except (LLMUnavailableError, ValueError, TypeError, KeyError):
                pass
        return self._route(goal, {}, source="goal_fallback")

    @staticmethod
    def _validated_targets(goal: TeachingGoal, data: dict) -> dict[str, tuple[str, str]]:
        allowed = set(goal.knowledge_points)
        targets: dict[str, tuple[str, str]] = {}
        raw_steps = data.get("steps", [])
        if not isinstance(raw_steps, list):
            return {}
        for item in raw_steps:
            if not isinstance(item, dict):
                continue
            point = str(item.get("knowledge_point", "")).strip()
            target = str(item.get("learning_target", "")).strip()
            if point not in allowed or point in targets or not target:
                continue
            if any(marker in target for marker in ("？", "?", "答案是", "正确答案")):
                continue
            # Route progression and final mastery are intentionally separate:
            # a correct answer may unlock the next curriculum step, while the
            # session still requires explanatory and transfer evidence before
            # declaring success.
            targets[point] = (target, "correct")
        return targets if set(targets) == allowed else {}

    @staticmethod
    def _route(
        goal: TeachingGoal,
        targets: dict[str, tuple[str, str]],
        *,
        source: str,
    ) -> TeachingRoute:
        steps: list[TeachingRouteStep] = []
        for index, point in enumerate(goal.knowledge_points, start=1):
            target, requirement = targets.get(
                point,
                (f"能够说明并应用“{point}”", "correct"),
            )
            steps.append(
                TeachingRouteStep(
                    step_id=f"knowledge_{index}",
                    title=point,
                    knowledge_point=point,
                    learning_target=target,
                    evidence_requirement=requirement,  # type: ignore[arg-type]
                    status="active" if index == 1 else "pending",
                )
            )
        steps.append(
            TeachingRouteStep(
                step_id="transfer",
                title="迁移验证",
                knowledge_point=goal.knowledge_points[-1],
                learning_target=f"在新情境中独立运用“{goal.objective}”",
                evidence_requirement="transfer",
                kind="transfer",
                status="pending",
            )
        )
        return TeachingRoute(steps=steps, source=source)  # type: ignore[arg-type]

    @classmethod
    def sync(
        cls,
        route: TeachingRoute,
        state: StudentState,
        *,
        allow_advance: bool = True,
    ) -> TeachingRoute:
        """Advance only when the current route target has matching evidence.

        ``allow_advance`` is a second deterministic gate owned by the Agent.
        The state assessor may mention a future concept, but that must not
        silently move the lesson route while the learner is still working on
        the active step.  The default stays ``True`` for standalone callers.
        """
        if not route.steps:
            return route
        if not allow_advance:
            return route
        current = route.current_step()
        completed = False
        if current.kind == "transfer":
            completed = state.transfer_verified
        elif state.evidence:
            latest = state.evidence[-1]
            levels = {"correct": 1, "explained": 2, "transfer": 3}
            required = levels.get(current.evidence_requirement, 1)
            observed = levels.get(latest.evidence_level, 0)
            completed = (
                latest.signal_type in {"positive", "transfer"}
                and observed >= required
                and cls._same_point(latest.knowledge_point, current.knowledge_point)
            )
        if completed:
            current.status = "completed"
            if route.current_index < len(route.steps) - 1:
                route.current_index += 1
                route.current_step().status = "active"
        return route

    @staticmethod
    def _same_point(left: str, right: str) -> bool:
        def normalize(value: str) -> str:
            return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", value).lower()

        a, b = normalize(left), normalize(right)
        return bool(a and b and (a == b or a in b or b in a))
