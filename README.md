# 基于 Teaching Skill Library 的自适应教学 Agent

> 当前版本先由真实 LLM 生成并持久化一条教学路线，再执行可审计的逐轮决策链：**状态诊断 → 路线门控 → Skill 选择 → 单步生成 → 语义复核 → 唯一下一步动作**。初始教师开场不计入学生作答轮。

## V5 深度整改说明

- 快速演示运行 18 案例 × 3 方法；完整评估进一步组合 3 种学生模拟画像和 5 个共享种子，共 810 个方法单元。
- 6 个原案例为开发集，12 个单独冻结的新情境案例为留出集；模拟器不读取方法名、可接受 Skill 或预期切换标签。
- 自动教学行为分是独立规则 proxy，人工盲评默认明确标记为“待完成”；项目不声称真实课堂因果效果。
- 会话 schema 升级到 v5，掌握度模型升级到 `evidence-v2`；会话保存 `TeachingRoute`，每轮记录内容 Skill、教学策略、阶段、调用审计和生成修订；暂停后“从当前进度继续”保留原会话 ID、完整对话和路线，只重置新的轮次预算。
- 掌握度是可观察证据指数：`partial=0.50 / correct=0.68 / explained=0.82 / transfer=0.92` 向目标收敛，而不是固定加分或真实能力概率。
- 评估明确拆成策略仿真、真实模型稳定性、人工盲评三种证据等级；没有真实双人标注时不生成教学质量分数。
- 会话支持展示名称、复制、归档、导出和可恢复删除；历史教学目标与状态证据默认不可原地篡改。
- 历史列表使用轻量索引、每页 20 条和选中后懒加载。测试使用隔离目录，不再污染正式档案。

严格复现依赖可执行 `pip install -r requirements-lock.txt`；覆盖率验收执行 `pytest --cov=src --cov-report=term-missing --cov-fail-under=92 -q`。提交前先运行 `python scripts/quality_gate.py` 核对案例、测试数量、覆盖率门槛和最新评估，再运行 `powershell -ExecutionPolicy Bypass -File scripts/prepare_submission.ps1` 完成质量审计，最后运行 `powershell -ExecutionPolicy Bypass -File scripts/build_submission.ps1` 生成白名单式交付包。打包脚本不会删除历史数据，只复制源码、文档、一个演示会话和最新一套 810 单元完整评估，并拒绝 `.env`、缓存和密钥文件。

华东师范大学夏令营综合考察题（二）的完整实现。系统根据教学目标、学生画像、当前掌握状态和历史对话，逐轮选择或切换 Teaching Skill，并且每次只输出下一步教学动作，等待真实学生回答。

## 已实现能力

- 显式维护知识掌握、误解、当前理解信号和下一关注点；
- 掌握度采用通用证据模型：LLM 对实际涉及的知识点标注 `partial / correct / explained / transfer`，再结合置信度和重复证据衰减更新；不信任模型随意返回的绝对分数，也不会更新未被回答涉及的知识点；
- 掌握度是可解释的证据估计分，不冒充真实概率；当前更新对有效解释证据更敏感，但仍保留 0–1 边界和 0.8 终止阈值。LLM 未明确映射知识点时只记录不确定证据，不猜测更新第一个知识点；
- 教师生成遵守对话连续性：学生只给出短回答时沿用当前例子、符号和数值，只有证据足够才推进到新情境；状态诊断与误解修复分别复核，避免“相关但错误”和“短但正确”混淆；
- 教师话语先生成并保存 `TeachingMicroStep`，再生成学生可见文本；每轮审计唯一关注点、唯一情境、已知事实、回答目标和当前表示法。结构化复核会拦截多个情境、多个子问题、事实矛盾、无理由换表示法和答案泄露，修复失败后使用保留上下文的通用单步回退；
- 新会话采用开放回答基础版，保证“真实回答 → 状态更新 → Skill 调整 → 下一问题”主闭环稳定、易演示；旧会话中的单选、填空和数值题数据仍可读取与回放，但不作为当前现场演示入口；
- 会话开始时先生成 `TeachingRoute`：每个用户提供的知识点对应一个稳定课程步骤，正确证据允许进入下一步骤，最终成功仍额外要求解释性证据和独立迁移验证；
- 每轮分别选择“内容 Skill × 教学策略”，并记录内容切换、策略切换和约束修正；无匹配内容 Skill 时明确进入通用诊断模式；内置 Skill 只读，用户 Skill 在独立目录中版本化。
- 每轮展示候选 Skill、课程/目标/触发/前置四项硬约束、决策来源、选择理由、状态变化和终止判断；
- 困惑持续时依次使用分层提示、误解纠正，并在掌握后切换到迁移验证；
- 达到掌握阈值且回答上一轮真实迁移题后成功终止；第三次停滞先执行纠错，学生回答纠错后仍无改善或达到八轮时暂停；
- 教学会话完整保存为 JSON，可在界面中回放；
- 会话档案支持搜索分页、导入导出、继续教学、展示名、复制、归档和回收站恢复；
- 18 个案例（6 个开发案例 + 12 个独立冻结留出案例）与 2 个基线方法的可复现量化评估；
- 评估报告自动归档，支持历史筛选、JSON 导入、全格式导出、成组归档和回收站恢复。
- 真实 LLM 负责语义诊断、误解识别、知识点映射、微步骤规划和回复生成；程序只保留提示注入、候选越权、掌握度边界、迁移验证、单步契约和终止条件等通用安全约束。典型轮次控制在 2–3 次 LLM 调用，低置信度或冲突轮才增加复核，单轮总预算为 45 秒。
- 实时教学默认启用简化推进模式：路线步骤完成后立即进入下一步，同一知识点最多做一次补充追问，迁移验证必须切换到新情境；复杂审计数据仍保留在教师视图和回放中，但不再主导学生可见问题。

## 快速开始

```powershell
cd D:\桌面\华师智能教育考核\考核2
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# 编辑 .env，填入 OpenAI-compatible API；正式演示默认走真实模型
streamlit run streamlit_app.py
```

正式答辩优先使用真实 API，侧栏应显示“LLM 已连接”；只有网络故障排查或自动化回归才设置 `TEACHING_AGENT_OFFLINE=1`。真实浏览器验收可在服务启动后运行：

```powershell
python -m pip install -r requirements-e2e.txt
playwright install chromium
python scripts/browser_acceptance.py
python scripts/online_multicase_acceptance.py --trials 2 --retries 2
```

使用真实 API 验证状态语义、Skill 回切和 AI 推荐演示回答（数据只写入临时目录）：

```powershell
python scripts/online_semantic_acceptance.py
python scripts/online_physics_route_acceptance.py
python scripts/online_demo_reply_acceptance.py
python scripts/online_fast_live_acceptance.py
```

使用真实 API 验证教师话语的单步契约，不替换或改写模型生成的问题：

```powershell
python scripts/online_single_step_acceptance.py
```

访问 `http://localhost:8501`。四页应用分别为：

1. **实时教学**：现场推荐先选择“牛顿第一定律”，再选择“导数极限定义”；二分查找保留在 Skill Library 和评估集，不作为默认演示入口。填写课程、主题、目标和学生基础后开始开放问答；知识点状态、历史和 Skill 限制收在“高级设置”。不想手动打字时，展开“AI 推荐演示回答”，由真实 LLM 根据当前问题生成三条回答，选择并确认后才提交；
2. **Skill Library**：查看、筛选、安全导入和导出 Teaching Skill；内置 Skill 只读，用户 Skill 可归档、恢复和移入回收站；
3. **过程回放**：复核 Skill 切换、状态轨迹和终止依据；
4. **Agent 评估**：快速运行 54 个方法单元，或完整运行 810 个方法单元，并导出 JSON、CSV、Markdown 和盲评材料。

实时教学页默认使用轻量复核模式：仍调用真实 LLM，普通回答轮优先控制在 2 次调用；如果确定性单步检查失败，会安全回退，不把不合格教师话语展示给学生。完整评估仍保留完整语义复核链。

会话是可继续更新的教学记录，提供完整 CRUD；评估报告作为不可变实验档案，不允许原地篡改指标，但支持创建/导入、读取、导出、成组归档和可恢复删除。Skill Library 是版本化教学资产；界面允许经过校验的 YAML 导入和单项导出，但禁止静默覆盖已有 Skill。

单独运行评估：

```powershell
python scripts/run_evaluation.py --mode full
```

运行测试：

```powershell
python -m pytest -q
```

## 系统架构

系统对外只有一个 `HybridTeachingAgent` 和一份统一的 `TeachingSession`，内部拆分为四个专业角色。四个角色不相互聊天、不各自保存状态，也不会产生互相竞争的教师回复；统一编排器按固定顺序调用它们，并且每次学生回答后只提交一个最终教学动作。

```text
新建会话：教学目标 → LLM 教学路线 → 持久化 TeachingRoute
   ↓
学生回答
   ↓
角色 1：状态诊断器 ──→ 掌握度 / 误解 / 理解信号 / 下一关注点
   ↓
角色 2：教学决策器 ──→ 候选硬过滤 + 内容 Skill × 教学策略
   ↓
角色 3：教师回复生成器 ──→ 一个 TeachingMicroStep + 一条教师话语
   ↓
角色 4：质量复核器 ──→ 通过 / 修复 / 拒绝
   ↓
统一 Teaching Agent ──→ 唯一下一步动作 / 成功终止 / 暂停
```

每轮 `generation_audit` 保存 `single_agent_four_internal_roles` 架构标识、角色列表、单一状态所有者和单动作输出约束，便于回放与答辩核验。评估 Agent 与实时教学链路隔离，不参与学生状态更新或教学回复生成。

核心公开接口：

```python
agent = HybridTeachingAgent()
session = agent.start_session(
    goal, profile, initial_state,
    history=None,
    available_skill_ids=None,
)
session = agent.handle_student_message(session, "学生的真实回答")
```

`TeachingSession` 保存整条 `TeachingRoute`；`turns` 保存每轮 `state_before`、`state_after`、候选 Skill、内容/策略 Skill、阶段、决策来源、理由、教师动作、切换原因、终止判断、`TeachingMicroStep`、`TeacherReview` 和 LLM 调用审计。会话 schema 当前为 v5，旧 v1/v2/v3/v4 会话读取时自动迁移；`rounds_in_current_run` 用于让续接会话获得新的轮次预算，而不删除历史。`StateEvidence` 额外保存学生原句、知识点、信号类型和证据等级；`KnowledgeState` 保存当前掌握度、置信度、最近证据和更新时间。

## Skill Library

`data/skills/` 包含第一题复制的 10 个学科 Skill，以及第二题新增的 4 个通用 Skill；用户导入版本保存到 `data/skills_custom/`，不覆盖内置文件：

| 新增 Skill | 新增原因 |
|---|---|
| 基于证据的诊断提问 | 原 Skill 缺少回答含糊时的状态定位方法 |
| 不直接给答案的分层提示 | 学生首次受阻时需要逐层降低难度 |
| 对比与反例驱动的误解纠正 | 原 Skill 记录失败模式，但没有统一纠错流程 |
| 新情境迁移与终止验证 | 为跨 Skill 成功终止提供统一证据 |

每个新增 YAML 都包含 `added_reason` 和 `applicable_when`，说明新增原因与适用场景。

## 配置

`.env`：

```dotenv
LLM_API_KEY=
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
TEACHING_AGENT_OFFLINE=0
TEACHING_AGENT_SESSION_DIR=
TEACHING_AGENT_EVALUATION_DIR=
TEACHING_AGENT_SKILL_DIR=
TEACHING_AGENT_CUSTOM_SKILL_DIR=
TEACHING_AGENT_CASES_PATH=
```

`config/settings.yaml` 控制最大轮数、平均/逐知识点掌握阈值、连续失败终止阈值、提示/纠错切换轮次、
模型温度、教师回复事实复核、候选数量、评估随机种子、LLM 超时/重试、状态复核调用预算和 evidence-v2 证据目标。
其中 `agent.simple_teaching_mode: true` 用于实时教学的简化路线推进；关闭后可回到兼容旧会话的完整微步骤守卫模式。

LLM 未配置、超时或返回非法 JSON 时，系统自动使用规则模式继续运行；OpenAI-compatible 客户端会重试、校验结构化 JSON，并修复字符串中的非法字面反斜杠。不会因为 API 故障丢失会话；真实 API 验收会单独记录回退和重试情况。

## 评估方法

开发集包含高等数学、大学物理、程序设计各 2 个案例；另一个冻结文件提供每门课程 4 个新情境，共 12 个留出案例。所有方法共享相同初始状态、模拟画像、随机种子、8 轮预算、迁移阈值和重复动作边际收益函数，对比：

- 自适应混合 Agent；
- 固定单 Skill Agent；
- 无 Skill Library 的通用教师 Agent。

指标包括逐轮状态诊断、决策质量、单步契约通过率、上下文连续率、选项有效率、LLM 回退率、调用耗时、证据映射、八维教学行为 proxy、题目级前后测、标准化学习增益、单位轮次效率和迁移正确率。画像和种子先在案例内汇总，再以 18 个案例为独立配对单位计算 cluster bootstrap 95% CI、配对置换检验、Holm 校正、McNemar 检验和配对 Hedges g。详细定义及量化结果见 [评估报告.md](评估报告.md) 与 [评估方法依据.md](评估方法依据.md)。

## 项目结构

```text
考核2/
├── streamlit_app.py          # Streamlit 入口
├── app_pages/                # 实时教学 / Skill Library / 回放 / 评估
├── src/                      # Agent、状态、Skill、LLM、评估、存储
├── data/skills/              # 10 个原 Skill + 4 个新增 Skill
├── data/evaluation_cases.json
├── data/evaluation_cases_heldout.json
├── output/sessions/          # 会话 JSON（运行时生成）
├── output/evaluations/       # 评估 JSON/CSV/Markdown
├── .e2e-runtime/             # 真实 API 与前端验收证据
├── release/                  # 当前规范提交包与 SHA-256 清单
├── tests/
├── config/settings.yaml
├── 评估报告.md
├── 评估方法依据.md
├── 评委关注点.md
└── 答辩演示流程.md
```
