# 基于 Teaching Skill Library 的自适应教学 Agent

> 当前实时教学采用“单次自适应教学输出”：一次真实 LLM 调用同时返回内容 Skill、教学动作、困难判断、证据、掌握度和教师回复。程序只校验内容 Skill 存在、保存状态并展示结果；旧多角色链路仅用于历史会话和评估兼容。

## V6 模型优先改造说明

- 快速演示运行 18 案例 × 3 方法；完整评估进一步组合 3 种学生模拟画像和 5 个共享种子，共 810 个方法单元。
- 6 个原案例为开发集，12 个单独冻结的新情境案例为留出集；模拟器不读取方法名、可接受 Skill 或预期切换标签。
- 自动教学行为分是独立规则 proxy，人工盲评默认明确标记为“待完成”；项目不声称真实课堂因果效果。
- 会话 schema 升级到 v6，掌握度模型升级到 `evidence-v2`；会话保存 `TeachingRoute`，每轮记录内容 Skill、统一自适应教学 Skill、五种教学动作、阶段和调用审计；旧会话读取时自动补齐新字段。
- 掌握度是可观察证据指数：`partial=0.50 / correct=0.68 / explained=0.82 / transfer=0.92` 向目标收敛，而不是固定加分或真实能力概率。
- 评估明确拆成策略仿真、真实模型稳定性、人工盲评三种证据等级；没有真实双人标注时不生成教学质量分数。
- 会话支持展示名称、复制、归档、导出和可恢复删除；历史教学目标与状态证据默认不可原地篡改。
- 历史列表使用轻量索引、每页 20 条和选中后懒加载。测试使用隔离目录，不再污染正式档案。

严格复现依赖可执行 `pip install -r requirements-lock.txt`；覆盖率验收执行 `pytest --cov=src --cov-report=term-missing --cov-fail-under=85 -q`。提交前先运行 `python scripts/quality_gate.py` 核对案例、覆盖率门槛和最新评估，再运行 `powershell -ExecutionPolicy Bypass -File scripts/prepare_submission.ps1` 完成质量审计，最后运行 `powershell -ExecutionPolicy Bypass -File scripts/build_submission.ps1` 生成白名单式交付包。自动化测试只作为核心回归，不设固定数量门槛；85% 只是基础健康线，不是题目要求。打包脚本不会删除历史数据，只复制源码、运行数据、一个演示会话和最新一套 810 单元完整评估，并拒绝 `.env`、缓存和密钥文件。

华东师范大学夏令营综合考察题（二）的完整实现。系统根据教学目标、学生画像、当前掌握状态和历史对话，逐轮选择或切换 Teaching Skill；具体怎么解释、举例和追问由真实 LLM 自主完成。

## 已实现能力

- 显式维护知识掌握、误解、当前理解信号和下一关注点；
- 掌握度采用通用证据模型：LLM 对实际涉及的知识点标注 `partial / correct / explained / transfer`，再结合置信度和重复证据衰减更新；不信任模型随意返回的绝对分数，也不会更新未被回答涉及的知识点；
- 掌握度是可解释的证据估计分，不冒充真实概率；当前更新对有效解释证据更敏感，但仍保留 0–1 边界和 0.8 终止阈值。LLM 未明确映射知识点时只记录不确定证据，不猜测更新第一个知识点；
- 教师回复由真实 LLM 根据当前 Skill、状态和历史对话自主组织；系统保存原始回复与教学记录，不用固定话术替代模型；
- 新会话采用开放回答基础版，保证“真实回答 → 状态更新 → Skill 调整 → 下一轮模型教学”主闭环稳定、易演示；旧会话中的单选、填空和数值题数据仍可读取与回放，但不作为当前现场演示入口；
- 会话开始时先生成 `TeachingRoute`：每个用户提供的知识点对应一个稳定课程步骤，正确证据允许进入下一步骤，最终成功仍额外要求解释性证据和独立迁移验证；
- 实时回合由一个模型同时返回内容 Skill 和教学动作；内容 Skill 与教学动作分开显示，但不再由多个策略 Skill 互相抢决策权。统一自适应教学 Skill 的动作只有 `explain / diagnose / scaffold / correct / transfer`。
- 每轮展示内容 Skill、教学动作和简短证据；程序只校验返回的内容 Skill 合法，教学表达不由固定模板接管；
- 模型根据学生原话判断是讲解、诊断、分层提示、误解纠正还是迁移验证；程序不根据关键词、连续失败次数或轮数强行切换动作。
- 达到掌握阈值且回答上一轮真实迁移题后成功终止；第三次停滞先执行纠错，学生回答纠错后仍无改善或达到八轮时暂停；
- 教学会话完整保存为 JSON，可在界面中回放；
- 会话档案支持搜索分页、导入导出、继续教学、展示名、复制、归档和回收站恢复；
- 18 个案例（6 个开发案例 + 12 个独立冻结留出案例）与 2 个基线方法的可复现量化评估；
- 评估报告自动归档，支持历史筛选、JSON 导入、全格式导出、成组归档和回收站恢复。
- 真实 LLM 在实时教学中负责语义诊断、内容 Skill 选择、教学动作选择和回复生成；程序只保留内容 Skill 合法性、状态保存和最大轮次终止边界。每个实时回合只进行一次模型决策调用。
- 实时教学默认使用单次自适应教学模式；无 API 时才进入离线兼容回退，真实 API 验收不会接受离线结果。

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
python scripts/online_adaptive_teaching_acceptance.py
python scripts/demo_physics.py
python scripts/demo_derivative.py
```

答辩现场直接按 `scripts/demo_physics.py` 和 `scripts/demo_derivative.py` 的输入顺序操作。物理脚本会跑通 `correct → transfer → success`；导数脚本会跑通连续 `scaffold`。两套脚本都强制真实 API，余额、网络或密钥异常时直接失败，不使用离线结果冒充通过。

使用真实 API 验证物理路线、Skill 切换和掌握度证据：

```powershell
python scripts/online_physics_route_acceptance.py
```

访问 `http://localhost:8501`。四页应用分别为：

1. **实时教学**：现场推荐先选择“牛顿第一定律”，再选择“导数极限定义”；两个预设自动使用对应内容 Skill 和单个完整知识目标，保证演示上下文稳定。二分查找保留在 Skill Library 和评估集，不作为默认演示入口。填写课程、主题、目标和学生基础后开始开放问答；知识点状态、历史和 Skill 限制收在“高级设置”。不想手动打字时，展开“AI 推荐演示回答”，由真实 LLM 根据当前问题生成三条回答，选择并确认后才提交；
2. **Skill Library**：查看、筛选、安全导入和导出 Teaching Skill；内置 Skill 只读，用户 Skill 可归档、恢复和移入回收站；
3. **过程回放**：复核 Skill 切换、状态轨迹和终止依据；
4. **Agent 评估**：快速运行 54 个方法单元，或完整运行 810 个方法单元，并导出 JSON、CSV、Markdown 和盲评材料。

实时教学页默认使用轻量模式：仍调用真实 LLM，普通回答轮优先控制在 2 次调用；模型自主组织教师回复，程序记录状态、Skill 和掌握度证据。完整评估仍保留完整语义复核链。

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

实时教学对外只有一个 `HybridTeachingAgent` 和一份统一的 `TeachingSession`。一次模型调用直接完成“理解学生 → 选择内容 Skill → 选择教学动作 → 生成回复”；程序只做内容 Skill 合法性检查、状态写入和终止边界控制。

```text
新建会话：教学目标 → 本地生成稳定路线 → 持久化 TeachingRoute
   ↓
学生回答
   ↓
一次自适应教学调用 ──→ 内容 Skill + 教学动作 + 困难 + 证据 + 回复
   ↓
统一 Teaching Agent ──→ 校验内容 Skill / 保存状态 / 展示回复
```

每轮 `generation_audit` 保存 `single_llm_adaptive_turn` 架构标识、模型动作和调用次数，便于回放与答辩核验。评估 Agent 与实时教学链路隔离，不参与学生状态更新或教学回复生成。

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

`TeachingSession` 保存整条 `TeachingRoute`；`turns` 保存每轮 `state_before`、`state_after`、内容 Skill、教学动作、困难类型、阶段和 LLM 调用审计。会话 schema 当前为 v6，旧会话读取时自动迁移；`StateEvidence` 保存学生原句、知识点、信号类型和证据等级；`KnowledgeState` 保存当前掌握度、置信度、最近证据和更新时间。

## Skill Library

`data/skills/` 包含学科 Skill，以及一个统一的 `adaptive_teaching_v1` 教学 Skill；旧通用 Skill 文件保留用于历史会话回放和兼容评估。用户导入版本保存到 `data/skills_custom/`，不覆盖内置文件：

| 新增 Skill | 新增原因 |
|---|---|
| 基于证据的诊断提问 | 原 Skill 缺少回答含糊时的状态定位方法 |
| 不直接给答案的分层提示 | 学生首次受阻时需要逐层降低难度 |
| 对比与反例驱动的误解纠正 | 原 Skill 记录失败模式，但没有统一纠错流程 |
| 新情境迁移与终止验证 | 为跨 Skill 成功终止提供统一证据 |

实时教学新增的统一 Skill 为 `adaptive_teaching_v1`，其内部动作只有 `explain / diagnose / scaffold / correct / transfer`。旧的四个策略 YAML 仍保留在库中，目的是让历史会话和旧评估可以读取，不会进入新的实时模型决策链。每个新增 YAML 都包含 `added_reason` 和 `applicable_when`，说明新增原因与适用场景。

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
其中 `agent.simple_teaching_mode: true` 启用实时单次自适应教学；关闭后可回到兼容旧会话和评估的完整微步骤链路。

LLM 未配置、超时或返回非法 JSON 时，系统自动使用规则模式继续运行；OpenAI-compatible 客户端会重试、校验结构化 JSON，并修复字符串中的非法字面反斜杠。不会因为 API 故障丢失会话；真实 API 验收会单独记录回退和重试情况。

## 评估方法

开发集包含高等数学、大学物理、程序设计各 2 个案例；另一个冻结文件提供每门课程 4 个新情境，共 12 个留出案例。所有方法共享相同初始状态、模拟画像、随机种子、8 轮预算、迁移阈值和重复动作边际收益函数，对比：

- 自适应混合 Agent；
- 固定单 Skill Agent；
- 无 Skill Library 的通用教师 Agent。

指标包括逐轮状态诊断、决策质量、单步契约通过率、上下文连续率、选项有效率、LLM 回退率、调用耗时、证据映射、八维教学行为 proxy、题目级前后测、标准化学习增益、单位轮次效率和迁移正确率。画像和种子先在案例内汇总，再以 18 个案例为独立配对单位计算 cluster bootstrap 95% CI、配对置换检验、Holm 校正、McNemar 检验和配对 Hedges g。运行 `python scripts/run_evaluation.py --mode full` 可生成结果文件。

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
├── scripts/                  # 评估、验收和两个现场演示脚本
├── tests/                    # 自动化回归
└── config/settings.yaml
```
