from src.lesson_planner import LessonPlanner
from src.models import StateEvidence, StudentState, TeachingGoal


class RouteLLM:
    available = True

    def structured(self, *args, **kwargs):
        return {
            "steps": [
                {
                    "knowledge_point": "惯性",
                    "learning_target": "能区分惯性与力",
                    "evidence_requirement": "explained",
                },
                {
                    "knowledge_point": "合力与运动变化",
                    "learning_target": "能判断合力如何改变运动状态",
                    "evidence_requirement": "correct",
                },
            ]
        }


def goal() -> TeachingGoal:
    return TeachingGoal(
        course="大学物理",
        topic="牛顿第一定律",
        objective="解释惯性并应用到新情境",
        knowledge_points=["惯性", "合力与运动变化"],
    )


def test_lesson_route_is_built_once_from_declared_knowledge_points():
    route = LessonPlanner(RouteLLM()).build(goal())  # type: ignore[arg-type]

    assert route.source == "llm"
    assert [step.title for step in route.steps] == ["惯性", "合力与运动变化", "迁移验证"]
    assert route.current_step().knowledge_point == "惯性"
    assert route.steps[-1].kind == "transfer"


def test_route_advances_only_on_matching_sufficient_evidence():
    route = LessonPlanner(RouteLLM()).build(goal())  # type: ignore[arg-type]
    state = StudentState(mastery={"惯性": 0.5, "合力与运动变化": 0.3})
    state.evidence.append(
        StateEvidence(
            student_quote="合力会改变运动状态",
            knowledge_point="合力与运动变化",
            signal_type="positive",
            evidence_level="correct",
        )
    )
    LessonPlanner.sync(route, state)
    assert route.current_step().knowledge_point == "惯性"

    state.evidence[-1] = StateEvidence(
        student_quote="物体会保持原来的运动状态",
        knowledge_point="惯性",
        signal_type="positive",
        evidence_level="correct",
    )
    LessonPlanner.sync(route, state)
    assert route.current_step().knowledge_point == "合力与运动变化"


def test_route_deterministic_gate_can_hold_matching_evidence():
    route = LessonPlanner(RouteLLM()).build(goal())
    state = StudentState(mastery={"惯性": 0.5, "合力与运动变化": 0.3})
    state.evidence.append(
        StateEvidence(
            student_quote="物体会保持原来的运动状态",
            knowledge_point="惯性",
            signal_type="positive",
            evidence_level="correct",
        )
    )

    LessonPlanner.sync(route, state, allow_advance=False)

    assert route.current_step().knowledge_point == "惯性"


def test_route_rejects_llm_added_knowledge_points_and_falls_back():
    class HallucinatingLLM(RouteLLM):
        def structured(self, *args, **kwargs):
            return {"steps": [{"knowledge_point": "题目外知识", "learning_target": "越界目标"}]}

    route = LessonPlanner(HallucinatingLLM()).build(goal())  # type: ignore[arg-type]
    assert route.source == "goal_fallback"
    assert {step.knowledge_point for step in route.steps[:-1]} == set(goal().knowledge_points)


def test_multi_task_guard_detects_two_tasks_hidden_behind_one_question_mark():
    from src.agent import HybridTeachingAgent

    assert HybridTeachingAgent._contains_multiple_requests(
        "判断乘客受到的合力方向是向前还是向后。这个合力对运动状态产生了什么影响？"
    )
    assert HybridTeachingAgent._contains_multiple_requests(
        "请说明合力方向与运动方向的关系，以及运动状态如何变化？"
    )
    assert not HybridTeachingAgent._contains_multiple_requests(
        "刹车时乘客受到向后的合力。请说明这个力为什么会让速度减小？"
    )
    assert not HybridTeachingAgent._contains_multiple_requests(
        "你回答得很准确，这说明你已经理解了关键区别。请换一个新情境说明它如何表现？"
    )
    assert HybridTeachingAgent._single_target(
        "说明公交车受到的合力是否为零。合力对运动状态产生了什么影响",
        "合力与运动变化",
    ) == "说明公交车受到的合力是否为零"
