# Adaptive Teaching Agent

基于 Teaching Skill Library 的自适应教学 Agent，服务于华东师范大学夏令营综合考察题（二）。

项目的核心不是“回答一道题”，而是：

> 学生回答 → Agent 判断学习状态 → 选择下一步教法 → 生成教师回复 → 保存证据并继续教学。

系统使用真实 OpenAI-compatible LLM 完成自然语言理解和教师回复生成，使用确定性程序规则负责 Skill 合法性、状态范围、会话持久化和终止边界。

## 项目亮点

- **实时自适应教学**：根据学生回答识别困惑、概念误解、符号困难等情况，再选择讲解、诊断、分层提示、误解纠正或迁移验证。
- **内容与教法分离**：内容 Skill 负责“教什么”，教学动作负责“怎么教”，同一套 Agent 框架可以复用到物理、高等数学和程序设计。
- **证据化学习状态**：保存知识点掌握度、理解信号、误解、证据等级、下一关注点和逐轮状态快照。
- **可解释和可回放**：每轮记录内容 Skill、教学动作、学生证据、状态变化、调用审计和终止依据。
- **公平评估**：使用 18 个案例、3 种教学方法、共享学生初始条件和随机种子，比较自适应 Agent、固定方法和无 Skill 通用教师。
- **数据安全**：`.env`、缓存、运行时会话和批量评估输出默认不提交到 GitHub。

## 运行环境

- Python 3.11 或 3.12
- Streamlit 1.59+
- OpenAI-compatible API（OpenAI、DeepSeek 或其他兼容服务）
- Windows PowerShell、macOS/Linux shell 均可运行

## 快速启动

在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
Copy-Item .env.example .env
```

然后编辑 `.env`，至少填写：

```dotenv
LLM_API_KEY=你的_API_密钥
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
TEACHING_AGENT_OFFLINE=0
```

启动应用：

```powershell
python -m streamlit run streamlit_app.py
```

浏览器打开：<http://localhost:8501>

侧栏显示“LLM 已连接”后，再进行真实模型教学演示。API 不可用时可以设置 `TEACHING_AGENT_OFFLINE=1` 做本地排障，但离线回退不能冒充真实 API 验收结果。

## 三个主要页面

### 1. 实时教学

建议用“牛顿第一定律”或“导数极限定义”开始：

1. 选择课程和教学主题；
2. 设置学生基础水平和初始掌握度；
3. 开始开放回答；
4. 输入学生真实回答；
5. 观察 Agent 的教学动作、内容 Skill、状态证据和下一步问题；
6. 必要时进入教师/答辩视图查看详细决策证据。

现场推荐演示的路径是：

```text
学生困惑 → 继续诊断 → 暴露概念误解 → 误解纠正 → 新情境迁移验证
```

### 2. Skill Library

用于查看和管理教学 Skill：

- 查看内置学科 Skill；
- 查看统一自适应教学 Skill；
- 导入经过校验的自定义 Skill；
- 导出、归档、恢复和回收站管理。

内置 Skill 默认只读，用户导入 Skill 不会静默覆盖已有版本。

### 3. 过程回放

选择已经保存的会话，可以查看：

- 学生每轮原话；
- 教师每轮回复；
- 内容 Skill 和教学动作；
- 状态更新和学习证据；
- Skill/教学动作变化；
- 成功、暂停或无法继续的原因。

## Agent 的工作方式

实时模式优先使用一次结构化 LLM 调用，返回：

- 内容 Skill；
- 教学动作；
- 学生困难类型；
- 学习证据和证据等级；
- 涉及的知识点；
- 掌握度建议；
- 下一关注点；
- 教师回复和下一道问题。

程序随后检查：

- 内容 Skill 是否存在并且允许使用；
- 掌握度是否在 0～1 范围内；
- 证据是否映射到真实涉及的知识点；
- 是否满足路线推进和终止条件；
- API 异常或非法 JSON 是否需要回退。

掌握度是“可观察学习证据指数”，不是学生真实能力的绝对概率。证据大致分为：

```text
partial   部分理解
correct   正确应用
explained 能够解释
transfer  能够迁移
```

一次答对不会自动代表完全掌握；成功终止还需要充分证据和独立迁移验证。

## Agent 评估

评估页面评估的是 Agent 的整体教学能力，而不是只评估学生状态。核心看四件事：

| 关注点 | 要回答的问题 |
|---|---|
| 状态判断 | Agent 看懂学生了吗？有没有识别误解？ |
| 教学决策 | Agent 选的下一步教法合适吗？ |
| 教学行为 | 回复准确、清楚、有针对性且不直接泄露答案吗？ |
| 教学效果 | 学生前后有没有提升，能否迁移到新情境？ |

### 三种对比方法

1. **自适应 Agent**：根据学生状态调整教学动作；
2. **固定单 Skill Agent**：固定使用一种教学方式；
3. **无 Skill 通用 Agent**：使用同一模型进行普通教师式教学，但不使用 Skill Library。

三种方法共享相同的案例、学生初始状态、回答条件、迁移题、轮次预算和随机条件，保证比较公平。

### 案例和评估单元

- 18 个案例：6 个开发案例 + 12 个留出案例；
- 快速评估：18 个案例 × 3 种方法 = 54 个评估单元；
- 完整评估：18 × 3 种方法 × 3 组学生画像 × 5 个种子 = 810 个评估单元。

“评估单元”就是一组独立的“案例 + 方法 + 学生条件”实验。

### 离线评估和真实 API 验收

批量评估默认使用可复现的模拟学生，原因是：

- 相同条件下可以公平比较三种方法；
- 不受模型随机回复、网络和 API 延迟影响；
- 可以快速重复大量实验；
- 结果能够复查和回归测试。

离线评估可以检查状态判断、教学决策、状态变化、迁移结果和终止逻辑，但不能直接证明真实课堂效果。真实 API 验收单独检查模型在线调用、上下文连续、输出格式、回退和响应稳定性。人工盲评则用于评价教师回复质量；未完成真实双人标注时，项目不会生成虚构的人工分数。

## 评估命令

快速评估或完整评估：

```powershell
python scripts/run_evaluation.py --mode quick
python scripts/run_evaluation.py --mode full
```

真实 API 演示脚本：

```powershell
python scripts/demo_physics.py
python scripts/demo_derivative.py
```

真实 API 多轮验收：

```powershell
python scripts/online_physics_route_acceptance.py
python scripts/online_multicase_acceptance.py --trials 2 --retries 2
python scripts/online_fast_live_acceptance.py
```

## 测试和质量检查

运行自动化测试：

```powershell
python -m pytest -q
```

覆盖率检查：

```powershell
python -m pytest --cov=src --cov-report=term-missing --cov-fail-under=85 -q
```

质量门禁：

```powershell
python scripts/quality_gate.py
```

测试默认使用隔离临时目录，不应把测试会话和评估结果写入正式 `output/`。

## 项目结构

```text
考核2/
├── streamlit_app.py          # Streamlit 入口
├── app_pages/                # 实时教学、Skill Library、回放、评估
├── src/                      # Agent、模型、状态、LLM、Skill、存储、评估
├── data/skills/              # 内置 Teaching Skill
├── data/evaluation_cases*.json # 开发集和留出集案例
├── config/settings.yaml      # 轮数、阈值、模型和评估配置
├── scripts/                  # 评估、演示、在线验收和质量检查
├── tests/                    # 单元、集成和 Streamlit 测试
├── requirements.txt          # 常规依赖
├── requirements-lock.txt     # 可复现依赖
├── .env.example              # 无密钥配置模板
└── 考核2项目报告.pdf         # 项目报告
```

运行时生成的会话、评估结果、缓存、覆盖率文件和 `.env` 不应上传到公开仓库。

## 配置说明

`.env` 中的主要配置：

```dotenv
LLM_API_KEY=
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
TEACHING_AGENT_OFFLINE=0
```

`config/settings.yaml` 控制最大轮数、掌握阈值、连续停滞阈值、模型温度、请求超时、评估随机种子和证据目标。

## 当前边界

- 自动评估主要验证模拟环境中的教学决策和稳定性，不等同于真实课堂因果效果；
- 自动行为分是回归用的 proxy，不能替代人工教师评价；
- LLM 输出具有自然波动，真实 API 验收需要保留失败、重试和回退记录；
- 18 个案例可以支持工程验证，但不能单独证明跨课程长期泛化；
- 真实学生实验、双人盲评和长期学习追踪仍是后续工作。

## 答辩演示建议

1. 用 PPT/HTML 先讲清楚“学生回答 → 状态判断 → 教学决策 → 下一步教学”；
2. 进入实时教学，用牛顿第一定律展示困惑、误解纠正和迁移；
3. 打开过程回放，展示每轮证据和状态变化；
4. 进入 Agent 评估，运行快速评估，展示三种方法和四类核心指标；
5. 最后主动说明离线评估、真实 API 验收和人工盲评的边界。

## 许可证和安全说明

本项目用于课程考核、教学 Agent 研究和演示。请勿将 `.env`、API 密钥、真实学生隐私数据或运行时会话上传到公开仓库。
