# 真实 API 多案例验收报告

验收日期：2026-08-16
模型：`deepseek-chat`
运行命令：`python scripts/online_multicase_acceptance.py --trials 1 --retries 2`

## V6 当前实时教学验收（单次自适应输出）

当前实时教学已改为每个回合一次真实 API 调用，同时返回内容 Skill、教学动作、困难类型、证据、掌握度和教师回复。程序不再根据关键词、连续失败次数或轮数覆盖模型动作；只校验内容 Skill 合法、保存状态并展示回复。旧的多角色数据和策略 Skill 仅用于历史会话、离线兼容与评估回归。

专项命令：

```powershell
python scripts/online_adaptive_teaching_acceptance.py
```

本次真实 `deepseek-chat` 验收覆盖两个答辩案例：

- 导数：`explain → scaffold → scaffold → scaffold`；学生依次回答“时间区间无限缩小”“这个不会啊”“我看不懂”，没有进入误解纠正，也没有切换到无关物理例子。
- 物理：学生先提问“物体不受力为什么还能继续运动”，随后明确断言“运动必须有力维持”，最后说明“合力为零时保持匀速直线运动”；真实模型在第二个回答进入 `correct`，第三个回答回到 `explain`。
- 首轮和每个学生回答回合均为 1 次自适应教学调用；教师回复不超过 100 字且最多保留一个问题。

脱敏结果保存在 `output/evaluations/online_adaptive_teaching_acceptance.json`。自然语言输出会随模型采样变化，验收只检查动作边界、内容 Skill 合法性、单次调用和回复长度，不要求逐字一致。

## V6 两个答辩演示脚本验收

2026-08-16 使用真实 `deepseek-chat` 跑通：

```powershell
python scripts\demo_physics.py
python scripts\demo_derivative.py
```

- 物理脚本：真实输入触发 `correct → explain/transfer → transfer`，掌握度达到阈值后会显示 `success`；结果保存到 `output/evaluations/demo_physics_run.json`。
- 导数脚本：真实输入连续触发 3 轮 `scaffold`，困难类型为 `symbol_notation`，没有误判为 `correct`；结果保存到 `output/evaluations/demo_derivative_run.json`。
- 两个脚本均检查内容 Skill、`adaptive_teaching_v1`、每轮 1 次 API 调用、教师回复不超过 100 字，并在 API 回退时直接失败。

以下旧版多案例报告保留作历史记录，不代表 V6 实时教学链路的当前架构。

## 验收规模

- 1 次完整复跑，另有 1 次无重试压力运行作为失败样本记录；
- 覆盖程序设计 1 个、高等数学 2 个、大学物理 5 个案例；
- 共 8 条真实多轮教学路径、24 条真实学生回答步骤；
- 108 次真实结构化 API 调用；
- 0 次额外自由文本生成调用（当前教师话语主路径使用结构化草稿）；
- API 或结构化解析回退失败：0 次。

每个案例均执行“创建目标 → 生成首轮教学 → 输入错误或困惑回答 → 诊断 Skill → 输入正确解释 → 回切学科 Skill”，并检查 8 项预期：首轮 Skill 合法、错误后进入诊断、错误后不涨掌握度、识别误解、正确后回切学科 Skill、正确后掌握度提升、清除当前误解、教师继续提出可回答问题。8 条路径共 64 项检查，复跑失败 0 项。

无重试压力运行曾出现 1 个真实模型误判：牛顿第二定律案例中，学生已经给出“F 是全部外力的矢量合力、应先受力分析”的完整正确解释，但一次状态诊断输出为 `partial`，没有立即回切学科 Skill。该次失败已保留在终端记录；独立复跑通过，说明它是模型输出波动而不是确定性路由必现错误，仍作为真实 API 稳定性风险保留。

## 三轮 Skill 轨迹

| 课程案例 | 三轮共同轨迹 |
|---|---|
| 程序设计：二分查找边界 | `binary_search_boundary_by_interval_definition → diagnostic_questioning_v1 → binary_search_boundary_by_interval_definition` |
| 高等数学：导数极限定义 | `derivative_intro_via_slope_limit_v1 → diagnostic_questioning_v1 → derivative_limit_definition_v1` |
| 大学物理：牛顿第一定律 | `newtons_first_law_via_engineering_examples_v1 → diagnostic_questioning_v1 → newtons_first_law_via_engineering_examples_v1` |
| 高等数学：变上限积分 | `variable_upper_limit_integration_v1 → diagnostic_questioning_v1 → variable_upper_limit_integration_v1` |
| 大学物理：牛顿第二定律 | `newtons_second_law_intro_v1 → diagnostic_questioning_v1 → newtons_second_law_intro_v1` |
| 大学物理：牛顿第三定律 | `newtons_third_law_content_formula_v1 → diagnostic_questioning_v1 → newtons_third_law_content_formula_v1` |
| 大学物理：动量守恒条件 | `momentum_conservation_spatial_uniformity_v1 → diagnostic_questioning_v1 → momentum_conservation_spatial_uniformity_v1` |
| 大学物理：牛顿定律到动量 | `transition_from_newton_to_momentum_v1 → diagnostic_questioning_v1 → transition_from_newton_to_momentum_v1` |

## 真实测试发现并修复的问题

首次稳定性测试曾出现：牛顿第一定律目标在“惯性”掌握度仍为 0.35 时跳到牛顿第二定律 Skill。根因是主题硬过滤把两个章节共享的知识点“合力”当成了主题匹配证据。

修复后，主题硬过滤只使用明确的教学主题和教学目标作为意图锚点；知识点仍用于状态更新，但不能单独授权跨章节 Skill。

扩充案例后又发现两类“正确回答被判 partial”的问题，分别出现在牛顿第二定律和动量守恒。根因是复杂状态评分要求学生一次覆盖过多目标。当前增加了独立的 LLM 语义进步裁决：只要回答在某个相关知识点上给出正确原则，或明确纠正当前误解，就判为本轮进步并只更新涉及知识点，不等同于整课掌握。该裁决不包含学科答案白名单。

另针对截图中的短回答场景执行了独立回归：“有元素” → “1个元素只有” → “1个下标啊”均由真实 LLM 结合上一轮教师问题判为正向证据，没有误触发纠错 Skill，掌握度从 0.35 逐步提升到 0.60。状态诊断增加了独立 LLM 复核层：首轮判断为 partial 或 regressed 且存在上一轮教师问题时，复核器会重新检查学生回答是否已经解决当前微问题，避免把正确的短回答仅因长度不足判为无进展。项目没有为“一个元素”“一个下标”等学科答案建立短语白名单。

## 前端验收

真实 Chromium 验收现在强制检查页面出现“LLM 已连接”；如果应用进入离线规则模式，脚本会直接失败。前端已完成真实回答提交、Skill 切换、Skill 检索、会话改名/复制/归档和结果导出入口检查。

本次追加了语义结构门禁：真实浏览器会检查首轮和后续教师话语，若一轮同时出现“左闭右闭”和“左闭右开”等多个区间表示法，则直接失败并保存页面证据。该门禁用于捕获“字段都存在但教学话语混题”的问题，不依赖学科答案键。

最新真实 API 浏览器验收命令：

```powershell
python scripts/browser_acceptance.py --output .e2e-runtime/screenshots/real-teaching-semantic-gate-20260816
```

结果：通过；截图保存在 `.e2e-runtime/screenshots/real-teaching-semantic-gate-20260816/`，覆盖 1366×768、1920×1080 和 390×844。

## 原始证据

完整脱敏轨迹保存在 `.e2e-runtime/online-api-multicase.json`，包含每轮 Skill、动作类型、教师回复、掌握状态和状态证据，不包含 API Key；本次同时保存 `.e2e-runtime/online-context-acceptance.json`，记录 3 轮单步上下文连续性验收。

该验收证明当前配置下真实模型调用与混合决策链能够完成上述案例；它仍不是大规模真实学生教学效果实验。

## 单步微步骤与上下文连续性回归

在单步决策链重构后，使用真实 `deepseek-chat` 运行：

```powershell
python scripts/online_progression_acceptance.py
```

最新真实 API 连续推进结果为 4 轮通过。每轮均保存 `TeachingMicroStep` 和生成审计；连续路径中未出现重复 `requested_target`，焦点能够从“区间定义”进入“循环不变量”和“边界更新”。其中真实模型曾把已有充分证据的知识点重新生成为定义题，确定性门控将其改为“请说明……在当前情境中的判断依据”，审计标记为 `completed_focus_revisit`，原始失败没有被静默删除。该结果是针对上下文连续和重复教学目标的定向回归，不替代完整多案例压力测试。

本次核心自动化回归为 150 项，另有结构化复核用例覆盖 Agent 编排、状态更新、Skill 切换和真实对话路径。

本轮还执行了两条真实 API 连续路径：大学物理 5 轮路线验收通过；程序设计 4 轮连续推进验收通过。对短回答“我不确定，请给我一个具体例子”的回归中，系统保持原情境、不重复开场问题，并生成“请用一个具体例子说明当前知识点”的单一请求；若课程先决知识不足，则明确进入通用诊断模式，不伪造学科 Skill。

## 现场推荐案例回归

为避免现场演示被二分查找的符号细节干扰，实时教学默认入口只保留“牛顿第一定律”和“导数极限定义”；二分查找仍保留在 Skill Library、历史会话和跨课程评估中。

本轮使用真实 `deepseek-chat` 分别创建两个预设会话：

- 牛顿第一定律：首轮命中 `newtons_first_law_via_engineering_examples_v1`，学生表达“物体不受力时为什么还能继续运动”后切换到 `diagnostic_questioning_v1`，继续表达“运动必须有力维持”后切换到 `scaffolded_hint_ladder_v1`；公交车急刹车情境保持不变。
- 导数极限定义：首轮命中 `derivative_limit_definition_v1`，教师保持同一汽车、位置函数和时间区间；学生暂时答非所问时，系统继续追问平均变化率，没有提前跳到新的极限表达式。

真实快速实时验收结果：单轮 3 次 LLM 调用，耗时约 4.9 秒，状态保持 `active`；AI 推荐演示回答真实生成 3 条且无重复。前端回归、浏览器验收和核心 150 项回归均通过。

本轮专项复核确认：物理急刹车中，学生问“合力是否为零”时，系统先建立“速度变化→加速度→合力不为零”的判断链，并将减速方向约束为向后；高数导数中，学生问“是不是取一个特别小的数”时，系统先区分固定小区间与趋近于零的极限，再继续追问。真实模型曾生成错误方向、重复问题和隐藏双问题，均已由确定性守卫拦截并补充回归测试。

本轮同时将实时教学收敛为单一对话页面：不再切换学生视图/教师视图，每条教师消息直接展示本轮内容 Skill、教学策略和状态证据；同时补齐两个预设的前置知识字段，避免默认现场会话无故退回通用诊断。

设计系统整改后又完成一轮真实浏览器教学：单一对话页面展示“本轮学习卡”，每条教师消息下方直接显示“内容 Skill × 教学策略 × 教学动作”，困惑回答后保持原情境与回答目标。最新验收截图保存在 `.e2e-runtime/screenshots/revisit-guard-20260816/`，包含 1920×1080、1366×768 和 390×844 视图，以及 Skill 切换、困惑回答和回放/评估入口证据。
