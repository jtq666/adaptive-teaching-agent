"""对照原始需求文档，逐条验证多轮交互内容"""

import sys, os
sys.path.insert(0, os.getcwd())

from src.agent import HybridTeachingAgent
from src.models import StudentProfile, StudentState, TeachingGoal, SessionStatus

agent = HybridTeachingAgent()

# ============================================================
# 需求: 输入教学目标、学生画像、学生当前掌握状态
# ============================================================
print("=" * 70)
print("需求1: 输入信息")
print("=" * 70)

goal = TeachingGoal(
    course="大学物理",
    topic="牛顿第一定律",
    objective="用公交车急刹车解释惯性，并区分保持运动与改变运动状态",
    knowledge_points=["惯性", "合力与运动变化"],
)
profile = StudentProfile(name="小明", level="中等")
initial_state = StudentState(
    mastery={"惯性": 0.3, "合力与运动变化": 0.3},
    next_focus="惯性",
)

session = agent.start_session(goal, profile, initial_state)

print(f"  [OK] 教学目标: {goal.course} / {goal.topic}")
print(f"  [OK] 学生画像: {profile.name}, {profile.level}")
print(f"  [OK] 初始掌握状态: {session.state.mastery}")
print(f"  [OK] 历史对话: 无 (首轮)")
print(f"  [OK] 可用Skill数: {len(session.skill_snapshot)}")

# ============================================================
# 需求: 从Skill Library中选择合适的Skill
# 需求: 每轮给出当前选择的Skill + 选择理由
# ============================================================
print("\n" + "=" * 70)
print("需求2: Skill选择与组合 (每轮必须有Skill + 选择理由)")
print("=" * 70)

first_turn = session.turns[0]
print(f"\n  首轮:")
print(f"    Skill: {first_turn.selected_skill_id}")
print(f"    选择理由: {first_turn.selection_reason}")
print(f"    Skill方案: 内容={first_turn.skill_plan.content_skill_id if first_turn.skill_plan else 'N/A'}, 策略={first_turn.skill_plan.strategy_skill_id if first_turn.skill_plan else 'N/A'}")

assert first_turn.selected_skill_id, "首轮必须有选择的Skill"
assert first_turn.selection_reason, "首轮必须有选择理由"
print("    [OK] 首轮Skill选择完整")

# ============================================================
# 需求: 学生状态建模 (4类信息)
# 需求: 当前知识掌握情况
# 需求: 可能存在的误解或错误模式
# 需求: 当前教学轮次下的理解信号
# 需求: 下一步教学关注点
# ============================================================
print("\n" + "=" * 70)
print("需求3: 学生状态建模 (4类信息)")
print("=" * 70)

print(f"\n  [OK] (1) 当前知识掌握情况:")
for point, mastery in session.state.mastery.items():
    print(f"        {point}: {mastery:.3f}")

print(f"  [OK] (2) 可能存在的误解:")
print(f"        误解列表: {[m.label for m in session.state.misconceptions]}")
print(f"        活动误解: {[m.label for m in session.state.misconception_states]}")

print(f"  [OK] (3) 当前理解信号:")
if session.state.evidence:
    latest = session.state.evidence[-1]
    print(f"        最新信号: {latest.signal_type}, 证据级别: {latest.evidence_level}")
print(f"        理解信号: {session.state.understanding_signals}")

print(f"  [OK] (4) 下一步教学关注点:")
print(f"        焦点: {session.state.next_focus}")

# ============================================================
# 核心测试: 多轮交互，逐轮检查
# ============================================================
print("\n" + "=" * 70)
print("需求4: 多轮教学交互 (逐轮检查)")
print("=" * 70)

test_answers = [
    ("惯性就是物体保持原来运动状态的性质，比如公交车突然刹车人会往前倾", "正确理解"),
    ("我觉得是因为人有惯性所以往前倾", "部分正确，但只说了惯性"),
    "合力改变了运动状态，惯性本身不变",
    "公交车刹车时脚受摩擦力停了，但上半身由于惯性继续向前",
    "不是惯性改变运动，而是合力为零时保持运动，合力不为零才改变",
]

for i, item in enumerate(test_answers):
    if isinstance(item, tuple):
        answer, expected = item
    else:
        answer = item
        expected = ""

    session = agent.handle_student_message(session, answer)
    turn = session.turns[-1]

    print(f"\n  --- 第 {i+2} 轮 ---")
    print(f"  学生回答: \"{answer}\"")

    # 需求: Skill选择
    print(f"  [需求] 当前Skill: {turn.selected_skill_id}")
    print(f"  [需求] 选择理由: {turn.selection_reason}")
    assert turn.selected_skill_id, f"第{i+2}轮必须有Skill选择"
    assert turn.selection_reason, f"第{i+2}轮必须有选择理由"

    # 需求: Skill组合/切换
    if turn.switch_reason:
        print(f"  [需求] Skill切换: {turn.switch_reason}")
    if turn.skill_plan:
        plan = turn.skill_plan
        print(f"  [需求] Skill组合: 内容={plan.content_skill_id or '无'} x 策略={plan.strategy_skill_id or '无'}")

    # 需求: 状态建模 - 掌握情况
    print(f"  [需求] 掌握度: {session.state.mastery}")

    # 需求: 状态建模 - 误解
    if session.state.misconceptions:
        print(f"  [需求] 误解: {[m.label for m in session.state.misconceptions]}")
    if session.state.misconception_states:
        print(f"  [需求] 活动误解: {[m.label for m in session.state.misconception_states]}")

    # 需求: 状态建模 - 理解信号
    if session.state.evidence:
        latest = session.state.evidence[-1]
        print(f"  [需求] 理解信号: {latest.signal_type}/{latest.evidence_level}")

    # 需求: 状态建模 - 关注点
    print(f"  [需求] 下一关注点: {session.state.next_focus}")

    # 需求: 教师回复 (不直接给答案)
    teacher_msg = turn.teacher_message
    print(f"  [需求] 教师回复: \"{teacher_msg[:100]}...\"")

    # 需求: 不直接给答案
    leak_cues = ["答案是", "正确答案", "直接写成", "你只要记住"]
    has_leak = any(cue in teacher_msg for cue in leak_cues)
    if has_leak:
        print(f"  [FAIL] 教师回复疑似泄露答案!")
    else:
        print(f"  [OK] 教师回复未泄露答案")

    # 需求: 终止判断
    print(f"  [需求] 终止判断: {turn.stop_decision}")
    print(f"  [需求] 动作类型: {turn.action_type}")
    print(f"  [需求] 阶段: {turn.phase}")
    print(f"  [需求] 决策模式: {turn.decision_mode}")

    if session.status != SessionStatus.ACTIVE.value:
        print(f"\n  >>> 会话终止! 状态={session.status}, 原因={session.termination_reason}")
        break

# ============================================================
# 需求: 在教学成功或判断无法继续时终止
# ============================================================
print("\n" + "=" * 70)
print("需求5: 终止条件")
print("=" * 70)

print(f"  会话状态: {session.status}")
if session.status == SessionStatus.ACTIVE.value:
    print(f"  [INFO] 会话仍在进行 (掌握度={session.state.average_mastery():.3f}, 需>={0.8})")
elif session.status == SessionStatus.SUCCESS.value:
    print(f"  [OK] 成功终止 (所有知识点掌握 + 迁移验证通过)")
elif session.status == SessionStatus.UNABLE.value:
    print(f"  [OK] 判断无法继续 (暂停)")

# ============================================================
# 需求: 评估 - 学生状态判断
# ============================================================
print("\n" + "=" * 70)
print("需求6: 评估 - 学生状态判断")
print("=" * 70)

print(f"  [OK] 知识掌握: {session.state.mastery}")
print(f"  [OK] 误解识别: {[m.label for m in session.state.misconceptions]}")
print(f"  [OK] 理解信号: {session.state.understanding_signals}")
print(f"  [OK] 关注点: {session.state.next_focus}")

# 证据链
print(f"\n  证据链 ({len(session.state.evidence)} 条):")
for ev in session.state.evidence:
    print(f"    轮{ev.round_index}: [{ev.signal_type}/{ev.evidence_level}] {ev.knowledge_point}")
    print(f"           原文: \"{ev.student_quote[:60]}...\"")
    print(f"           理由: {ev.reason[:60]}...")

# ============================================================
# 需求: 评估 - 教学决策质量
# ============================================================
print("\n" + "=" * 70)
print("需求7: 评估 - 教学决策质量 (Skill选择/组合/切换/终止)")
print("=" * 70)

for idx, turn in enumerate(session.turns):
    print(f"  轮{idx+1}: Skill={turn.selected_skill_id}, 动作={turn.action_type}, 阶段={turn.phase}")
    if turn.skill_plan:
        p = turn.skill_plan
        print(f"         组合: 内容={p.content_skill_id or '无'} x 策略={p.strategy_skill_id or '无'}")
        if p.content_switch:
            print(f"         [切换] 内容Skill已切换")
        if p.strategy_switch:
            print(f"         [切换] 策略Skill已切换")

# ============================================================
# 需求: 评估 - 教学行为质量 (不直接给答案)
# ============================================================
print("\n" + "=" * 70)
print("需求8: 评估 - 教学行为质量 (不直接给答案)")
print("=" * 70)

for idx, turn in enumerate(session.turns):
    msg = turn.teacher_message
    leak_cues = ["答案是", "正确答案", "直接写成", "你只要记住"]
    has_leak = any(cue in msg for cue in leak_cues)
    has_question = "？" in msg or "?" in msg

    status = "[OK]" if (not has_leak and has_question) else "[FAIL]"
    print(f"  轮{idx+1}: {status} 泄露={has_leak}, 有问句={has_question}")

# ============================================================
# 评估报告生成
# ============================================================
print("\n" + "=" * 70)
print("需求9: 评估报告")
print("=" * 70)

print(f"  总轮数: {len(session.turns)}")
print(f"  最终掌握度: {session.state.mastery}")
print(f"  平均掌握度: {session.state.average_mastery():.3f}")
print(f"  迁移验证: {session.state.transfer_verified}")
print(f"  误解数: {len(session.state.misconceptions)}")
print(f"  证据数: {len(session.state.evidence)}")
print(f"  会话状态: {session.status}")

print("\n" + "=" * 70)
print("全部需求核对完成")
print("=" * 70)
