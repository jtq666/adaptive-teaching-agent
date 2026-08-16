"""端到端功能测试：创建会话 → 多轮交互 → 状态追踪 → 终止条件"""

import sys, os
sys.path.insert(0, os.getcwd())

from src.agent import HybridTeachingAgent
from src.config import get_agent_settings
from src.models import (
    StudentProfile, StudentState, TeachingGoal, SessionStatus
)

# ============================================================
# 测试 1: 会话创建
# ============================================================
print("=" * 60)
print("测试 1: 会话创建")
print("=" * 60)

agent = HybridTeachingAgent()
goal = TeachingGoal(
    course="大学物理",
    topic="牛顿第一定律",
    objective="用公交车急刹车解释惯性，并区分保持运动与改变运动状态",
    knowledge_points=["惯性", "合力与运动变化"],
)
profile = StudentProfile(name="测试学生", level="中等")
initial_state = StudentState(
    mastery={"惯性": 0.3, "合力与运动变化": 0.3},
    next_focus="惯性",
)

session = agent.start_session(goal, profile, initial_state)
print(f"  会话ID: {session.session_id[:8]}...")
print(f"  状态: {session.status}")
print(f"  轮次数: {len(session.turns)}")
print(f"  教学路线: {[s.knowledge_point for s in session.teaching_route.steps]}")
print(f"  当前路线步骤: {session.teaching_route.current_step().knowledge_point}")

if session.turns:
    first_turn = session.turns[0]
    print(f"  首轮教师话语: {first_turn.teacher_message[:80]}...")
    print(f"  首轮Skill: {first_turn.selected_skill_id}")
    print(f"  首轮动作类型: {first_turn.action_type}")
    print(f"  首轮阶段: {first_turn.phase}")

assert session.status == SessionStatus.ACTIVE.value, f"期望active，实际{session.status}"
assert len(session.turns) == 1, f"期望1轮，实际{len(session.turns)}"
print("  [OK] 会话创建成功\n")

# ============================================================
# 测试 2: 多轮教学交互
# ============================================================
print("=" * 60)
print("测试 2: 多轮教学交互")
print("=" * 60)

student_answers = [
    "惯性就是物体保持原来运动状态的性质，比如公交车突然刹车，人会往前倾",
    "我觉得是因为人有惯性所以往前倾，合力改变了运动状态",
    "惯性是物体的固有属性，合力为零时物体保持匀速直线运动或静止",
    "当公交车刹车时，脚受到摩擦力停下了，但上半身由于惯性继续向前运动",
    "合力不为零才会改变运动状态，惯性本身不会改变运动，只是抵抗改变",
]

for i, answer in enumerate(student_answers):
    turn_num = i + 1
    print(f"\n--- 第 {turn_num + 1} 轮 ---")
    print(f"  学生回答: {answer[:60]}...")

    before_mastery = dict(session.state.mastery)
    before_evidence = len(session.state.evidence)

    session = agent.handle_student_message(session, answer)

    after_mastery = dict(session.state.mastery)
    after_evidence = len(session.state.evidence)
    current_turn = session.turns[-1]

    print(f"  状态变化: {before_mastery} → {after_mastery}")
    print(f"  证据数: {before_evidence} → {after_evidence}")
    print(f"  教师回复: {current_turn.teacher_message[:80]}...")
    print(f"  选择Skill: {current_turn.selected_skill_id}")
    print(f"  动作类型: {current_turn.action_type}")
    print(f"  阶段: {current_turn.phase}")
    print(f"  决策模式: {current_turn.decision_mode}")

    if current_turn.switch_reason:
        print(f"  切换原因: {current_turn.switch_reason}")

    if current_turn.fallback_reason:
        print(f"  回退原因: {current_turn.fallback_reason}")

    if current_turn.skill_plan:
        plan = current_turn.skill_plan
        print(f"  Skill方案: 内容={plan.content_skill_id or '无'}, 策略={plan.strategy_skill_id or '无'}")

    print(f"  会话状态: {session.status}")

    # 检查是否已经终止
    if session.status != SessionStatus.ACTIVE.value:
        print(f"  [WARN] 会话已终止: {session.termination_reason}")
        break

print(f"\n  最终掌握度: {session.state.mastery}")
print(f"  平均掌握度: {session.state.average_mastery():.3f}")
print(f"  总轮数: {len(session.turns)}")
print(f"  会话状态: {session.status}")
print("  [OK] 多轮交互完成\n")

# ============================================================
# 测试 3: 状态追踪详细检查
# ============================================================
print("=" * 60)
print("测试 3: 状态追踪详细检查")
print("=" * 60)

print(f"  知识点掌握度:")
for point, mastery in session.state.mastery.items():
    print(f"    {point}: {mastery:.3f}")

print(f"\n  知识点详细状态:")
for point, kstate in session.state.knowledge_states.items():
    print(f"    {point}:")
    print(f"      mastery={kstate.mastery:.3f}, confidence={kstate.confidence:.3f}")
    print(f"      last_evidence={kstate.last_evidence_level}")
    print(f"      evidence_count={len(kstate.evidence)}")

print(f"\n  误解列表: {[m.label for m in session.state.misconceptions]}")
print(f"  活动误解: {[m.label for m in session.state.misconception_states]}")
print(f"  当前焦点: {session.state.next_focus}")
print(f"  迁移验证: {session.state.transfer_verified}")
print(f"  无进展轮数: {session.state.no_progress_rounds}")

print(f"\n  证据链 ({len(session.state.evidence)} 条):")
for ev in session.state.evidence:
    print(f"    轮{ev.round_index}: [{ev.signal_type}/{ev.evidence_level}] {ev.knowledge_point} - \"{ev.student_quote[:50]}...\"")

print("  [OK] 状态追踪检查完成\n")

# ============================================================
# 测试 4: 终止条件
# ============================================================
print("=" * 60)
print("测试 4: 终止条件")
print("=" * 60)

settings = get_agent_settings()
mastery_threshold = settings.get("mastery_threshold", 0.8)
max_rounds = settings.get("max_rounds", 8)
no_progress_limit = settings.get("no_progress_limit", 3)

print(f"  配置: mastery_threshold={mastery_threshold}, max_rounds={max_rounds}, no_progress_limit={no_progress_limit}")
print(f"  当前: 平均掌握度={session.state.average_mastery():.3f}, 轮数={len(session.turns)}")

if session.status == SessionStatus.ACTIVE.value:
    print(f"  会话仍在进行中 (可能需要更多轮次)")
elif session.status == SessionStatus.SUCCESS.value:
    print(f"  成功终止 [OK]")
elif session.status == SessionStatus.UNABLE.value:
    print(f"  暂停/无法完成 [WARN]")
    print(f"  终止原因: {session.termination_reason}")

print("  [OK] 终止条件检查完成\n")

# ============================================================
# 测试 5: 会话恢复
# ============================================================
print("=" * 60)
print("测试 5: 会话恢复")
print("=" * 60)

if session.status == SessionStatus.UNABLE.value:
    try:
        resumed = agent.resume_session(session)
        print(f"  恢复后状态: {resumed.status}")
        print(f"  轮次预算重置: {resumed.rounds_in_current_run}")
        print(f"  新轮数: {len(resumed.turns)}")
        print("  [OK] 会话恢复成功\n")
    except Exception as e:
        print(f"  [FAIL] 会话恢复失败: {e}\n")
else:
    print(f"  会话状态为 {session.status}，跳过恢复测试\n")

# ============================================================
# 测试 6: 再生当前轮
# ============================================================
print("=" * 60)
print("测试 6: 再生当前轮")
print("=" * 60)

try:
    regenerated = agent.regenerate_current_turn(session)
    last_turn = regenerated.turns[-1]
    print(f"  再生后教师话语: {last_turn.teacher_message[:80]}...")
    print(f"  修订历史: {len(last_turn.generation_revisions)} 条")
    for rev in last_turn.generation_revisions:
        print(f"    修订{rev.revision_index}: {rev.reason}")
    print("  [OK] 再生功能正常\n")
except Exception as e:
    print(f"  [FAIL] 再生失败: {e}\n")

print("=" * 60)
print("全部功能测试完成")
print("=" * 60)
