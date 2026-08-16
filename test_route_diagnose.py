"""精确诊断：逐轮检查路线推进的每一个条件"""

import sys, os
sys.path.insert(0, os.getcwd())

from src.agent import HybridTeachingAgent
from src.models import StudentProfile, StudentState, TeachingGoal

agent = HybridTeachingAgent()
goal = TeachingGoal(
    course="大学物理",
    topic="牛顿第一定律",
    objective="用公交车急刹车解释惯性，并区分保持运动与改变运动状态",
    knowledge_points=["惯性", "合力与运动变化"],
)
profile = StudentProfile(name="测试", level="中等")
state = StudentState(
    mastery={"惯性": 0.3, "合力与运动变化": 0.3},
    next_focus="惯性",
)

session = agent.start_session(goal, profile, state)

# 路线初始状态
route = session.teaching_route
print("=== 初始路线 ===")
for i, step in enumerate(route.steps):
    print(f"  [{i}] {step.knowledge_point} | 要求={step.evidence_requirement} | 状态={step.status}")
print(f"  current_index={route.current_index}")

answers = [
    "惯性就是物体保持原来运动状态的性质，比如公交车突然刹车人会往前倾",
    "我会继续向前倾，因为车停了但我没有受到向后的力，身体还保持着原来向前运动的状态，这就是惯性",
    "惯性是物体的固有属性，合力为零时物体保持匀速直线运动或静止",
]

for turn_num, answer in enumerate(answers, start=2):
    print(f"\n{'='*60}")
    print(f"=== 轮{turn_num}: 学生回答: \"{answer[:50]}...\" ===")
    print(f"{'='*60}")

    # 路线状态（回答前）
    print(f"\n  [回答前] 路线 current_index={session.teaching_route.current_index}")
    for i, step in enumerate(session.teaching_route.steps):
        marker = " <-- 当前" if i == session.teaching_route.current_index else ""
        print(f"    [{i}] {step.knowledge_point} | 状态={step.status}{marker}")

    # 证据状态（回答前）
    print(f"  [回答前] 证据数={len(session.state.evidence)}")
    if session.state.evidence:
        prev_ev = session.state.evidence[-1]
        print(f"    上一条: [{prev_ev.signal_type}/{prev_ev.evidence_level}] {prev_ev.knowledge_point}")

    # 执行
    session = agent.handle_student_message(session, answer)
    turn = session.turns[-1]

    # 路线状态（回答后）
    print(f"\n  [回答后] 路线 current_index={session.teaching_route.current_index}")
    for i, step in enumerate(session.teaching_route.steps):
        marker = " <-- 当前" if i == session.teaching_route.current_index else ""
        print(f"    [{i}] {step.knowledge_point} | 状态={step.status}{marker}")

    # 最新证据
    print(f"  [回答后] 证据数={len(session.state.evidence)}")
    if session.state.evidence:
        latest_ev = session.state.evidence[-1]
        print(f"    最新: [{latest_ev.signal_type}/{latest_ev.evidence_level}] {latest_ev.knowledge_point}")
        print(f"    原文: \"{latest_ev.student_quote[:60]}...\"")

    # focus_label
    focus = agent._focus_label(session)
    print(f"  [焦点] _focus_label = \"{focus}\"")

    # 教师回复
    print(f"  [教师] \"{turn.teacher_message[:80]}...\"")
    print(f"  [教师] Skill={turn.selected_skill_id}, 阶段={turn.phase}")

    # 掌握度
    print(f"  [掌握度] {session.state.mastery}")
    print(f"  [下一关注点] {session.state.next_focus}")
