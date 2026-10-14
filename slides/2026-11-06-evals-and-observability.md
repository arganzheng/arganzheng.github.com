---
layout: slides
title: "评测、可观测与可追溯：怎么知道改了之后更好了"
subtitle: "系列精华 · 七篇正文每篇一页：没有评测集的改动是猜测"
permalink: /slides/evals-and-observability.html
series: evals-and-observability
date: 2026-11-06 23:30:00 +0800
author: arganzheng
description: "《评测、可观测与可追溯》系列的分享用幻灯片：评测集从真实流量采、规则 / judge / 人各评什么与 judge 的四类偏差、单步 / RAG / agent / 多轮的指标矩阵、回归门禁与供应商静默升级、trace 与 OTel GenAI 约定、录制回放与失败分类、运行时 guardrails 与物理世界的仿真。"
theme: white
transition: slide
---

## 这个系列的三句话主张

> **没有评测集的改动是猜测**；**judge 是仪器，要先被测量**；**不可复现就录下来**。

<aside class="notes" markdown="1">
总纲：/evaluation-observability-and-traceability.html。89% 的团队有可观测、只有 52% 有评测——差距是事故来源。
</aside>

---

## 几篇怎么连起来

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 170}}}%%
flowchart TB
    E1["01 评测集<br/>AI 应用的单元测试"] --> E2["02 评分<br/>规则 / judge / 人"] --> E3["03 评什么<br/>指标矩阵"] --> E4["04 回归与门禁<br/>CI、静默升级、A/B、在线评测"]
    E5["05 trace<br/>span、OTel GenAI 约定"] --> E6["06 可追溯<br/>录制回放、决策点日志、失败分类"]
    E5 -- "评测集从 trace 里采" --> E1
    E6 -- "bad case → 用例" --> E1
    E4 & E6 --> E7["07 运行时可靠性与物理世界的仿真<br/>评测发现的失效模式做成运行时检查"]
```

---

## 01 · 评测集：AI 应用的单元测试

**结论**：从**真实流量按类型采**（高频 / 边界 / 失败过的）+ 红队，不按流量比例；四字段、期望可检验、含状态；**几十条起、跑 k 次**；hold-out、六元组绑定、随 bad case 长。

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 180}}}%%
flowchart LR
    L["生产流量（trace）"] -- "按<b>类型</b>采样，不按流量比例" --> S["候选用例"]
    R["红队 / 安全用例<br/>（流量里不会自然出现，专门写）"] --> S
    S --> A["标注：期望是<b>分布</b>不是值<br/>通过率 ≥ 80%、必含 / 必不含、schema"]
    A --> T{"分层"}
    T --> T1["冒烟集：几十条，每次改动都跑"]
    T --> T2["全量集：几百到几千条，发布前跑"]
    T --> T3["hold-out：不给调 prompt 的人看，防过拟合评测集"]
    T1 & T2 --> G["门禁（第四篇）<br/>绑定 prompt 版本 + 模型快照"]
    G -- "线上 bad case" --> S
    G -. "模型升级 → 依赖变了，全量重跑" .-> T2

```

<aside class="notes" markdown="1">
原文 /eval-sets-the-unit-tests-of-ai-applications.html。
</aside>

<!-- v -->

### 要点

| 量 | 数 |
|---|---|
| 有可观测 vs 有评测的团队 | 89% vs **52%** |
| 重复次数 k | 3–5 |
| hold-out | 20–30% |
| 合成集 | 只做冷启动——问法贴合原文，recall 高估 |

- 「等评测集完善再用」——几十条就能分方向；拖延是最常见的失败
- 「按流量比例采」——难例几乎没有，退化看不见

---

## 02 · 评分：规则、LLM-as-judge 与人

**结论**：**能规则不 judge、能 judge 不人**；judge 四类偏差（风格最大）；**报 kappa 不报一致率**（未校正机遇，虚高 33–41 pp）；一致 ≠ 正确；八步校准。

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 190}}}%%
flowchart LR
    O["一条输出"] --> R{"能用<b>规则</b>判吗？<br/>字段齐不齐、枚举、容差、引用 id 是否存在"}
    R -- "能：确定、免费" --> RR["规则打分"]
    R -- "不能：「解释清楚吗」「忠实吗」" --> J["<b>LLM-as-judge</b> 打分<br/>rubric、pairwise 换序去偏"]
    J --> C{"judge 校准过吗？"}
    C -- "抽 30–50 条与人比，一致率 ≥ 85%" --> OK["可用"]
    C -- "一致率不够" --> H["<b>人</b>：重标、改 rubric、换 judge 模型"]
    H --> J
    OK -. "每次改 judge 的 prompt / 模型重新校准；<br/>人之间也要量一致率（两个 80% 正确的人只有 68% 一致）" .-> C

```

<aside class="notes" markdown="1">
原文 /scoring-rules-llm-as-a-judge-and-humans.html。
</aside>

<!-- v -->

### 要点

| 量 | 数 |
|---|---|
| 一致率 → kappa | 虚高 33–41 pp；排名移 14 位 |
| 重测一致 vs 位置偏差 | > 0.95 与 > 0.10 并存——稳定但有偏 |
| 风格偏差 | 0.10–0.76 |
| 中档模型 + 去偏 vs 前沿模型 | 胜，且便宜 **15×** |
| 门槛 | κ ≥ 0.61 |

- 「judge 一致率 88% 可做门禁」——报 kappa；「用最大的模型当 judge」——去偏策略比大小重要

---

## 03 · 评什么：指标矩阵

**结论**：**形态 × 维度**——单步 / RAG / agent / 多轮 / 代码 × 结果 / 过程 / 成本 / 安全；**过程与结果并列**；多轮以**会话**为单位、模拟器要校准；生产看 **pass^k** 不看 pass@k。

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 360}}}%%
flowchart LR
    F{"应用形态"} --> S["<b>单步</b><br/>结果：准确率 / F1 / rubric<br/>形式：格式遵循率、schema 通过率、拒答正确率"]
    F --> RG["<b>RAG</b><br/>检索侧：recall@k、nDCG<br/>生成侧：faithfulness、引用准确<br/>（分开评，L3 第七篇）"]
    F --> AG["<b>agent</b><br/>结果之外看过程：步数、越权、<br/>工具选择、部分结果、成本<br/>（L4 第九篇）"]
    F --> MT["<b>多轮</b><br/>策略遵循、轮数、目标达成、<br/>上下文丢失点"]
    S & RG & AG & MT --> X["所有形态都并列报：<br/>成本 / 任务 · p95 延迟 · 安全（红队通过率）<br/>按分布看（跑 n 次），按类型分层"]

```

<aside class="notes" markdown="1">
原文 /what-to-evaluate-metrics-for-single-step-rag-agent-and-multi-turn.html。
</aside>

<!-- v -->

### 要点

| 形态 | 结果 | 过程 | 成本 | 安全 |
|---|---|---|---|---|
| 单步 | 正确 / 格式 | — | token | 拒答 |
| RAG | 忠实 / 完整 | recall@k、引用 | 检索 + 生成 | 权限泄漏 |
| agent | 任务完成 | 步数、越权、重复 | token × 步 | 不可逆动作 |
| 多轮 | 会话目标 | 每轮 + 全局 | 会话 token | |

- 分层报告；质量 vs 成本图
- 「pass@5 = 生产可靠性」——生产看 pass^k（k 次全过）

---

## 04 · 回归与门禁：改动 = 发布

**结论**：分层 ≥ 基线 − 噪声、逐条 diff；**钉快照 + 探针集每日**（探针集是模型指纹，发现供应商静默升级）；影子 → 灰度 → 回滚；A/B 一因素；隐式反馈带 trace id；**季度校验离线—在线相关性**。

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 180}}}%%
flowchart LR
    CH["改动：prompt · 模型 / effort · schema · 工具集 · 检索配置 · harness 版本 · judge / rubric"] --> V["新版本（不可变）+ 六元组记录"]
    V --> G{"门禁：评测集 × k<br/>分层指标 ≥ 基线 − 噪声阈值<br/>逐条 diff · 成本 / 延迟不退化 · 红队集通过"}
    G -->|"不过"| CH
    G -->|"过"| S["影子模式（高风险改动）<br/>新旧并跑 · 只记录不生效 · 比差异"]
    S --> C["灰度 canary 1–5%<br/>trace 绑定版本 · 在线指标 · 隐式反馈"]
    C -->|"退化"| RB["回滚：标签指回旧版本"]
    C -->|"正常"| P["全量 production"]
    P --> M["监控 + 探针集每日跑<br/>供应商静默升级 · 分布漂移"]
    M -->|"漂移"| G
    AB["A/B：两个版本并行分流<br/>按版本聚合在线指标"] -.-> C

    classDef step fill:#fff7e0,stroke:#c98a00,stroke-width:2px,color:#222
    classDef gate fill:#eef6ff,stroke:#5b8fd6,color:#222
    class CH,V,S,C,P,M,RB,AB step
    class G gate
```

<aside class="notes" markdown="1">
原文 /regression-gates-silent-model-updates-ab-and-online-evaluation.html。
</aside>

<!-- v -->

### 要点

| 规则 | 为什么 |
|---|---|
| 改 rubric 后重量基线 | 尺子变了 |
| 灰度忽略缓存冷启动 | 第一轮全写是预期 |
| 短期信在线、长期修离线 | 在线有噪声，离线可复现 |

- 「没部署就不会变」——供应商静默升级；「离线涨了在线一定涨」——季度校验

---

## 05 · trace：记什么、按什么约定

**结论**：三级树（会话 / 轮 / 调用）+ 事件 + 子会话链接；OTel `gen_ai.*` 全部 Development 状态、迁独立仓库、属性改名（`system` → `provider.name`）、**钉 opt-in**；业务属性用 `app.*`；元数据全量、内容采样、bad case 全量。

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 400}}}%%
flowchart TB
    T["trace = 一次会话 / 一个任务<br/>trace_id · 用户 · prompt 版本 · 模型快照 · 总成本"]
    T --> S1["span：第 1 步"] & S2["span：第 2 步"] & S3["span：第 k 步"]
    S2 --> M["模型调用 span<br/>完整输入（或可重建的引用）、输出、<br/>四类 token、effort、TTFT、stop_reason"]
    S2 --> TO["工具调用 span<br/>名称、参数、结果引用、耗时、<br/>沙箱决定、审批人与时间"]
    S2 --> RT["检索 span<br/>改写前后的查询、过滤条件、两路候选、<br/>RRF / rerank 排序、最终块 id"]
    M & TO & RT -. "评测集从这里采；bad case 带 trace_id；<br/>供应商静默升级从指标曲线看出来" .-> U["下游用途"]

```

<aside class="notes" markdown="1">
原文 /tracing-spans-opentelemetry-genai-conventions-and-tooling.html。
</aside>

<!-- v -->

### 要点

| 项 | 数 |
|---|---|
| OTel 语义约定 | v1.42.0（2026-06）移到独立仓库 |
| 工具选择 | 按评测闭环 / 自托管 |
| trace 是什么 | 会话日志的投影 |

- 「只记前 500 字省空间」——回放与复现都做不了；内容进对象存储
- 「自造 `gen_ai.prompt.version`」——命名空间冲突；用 `app.*`

---

## 06 · 可追溯：从坏结果追到那一步并变成用例

**结论**：**录制回放**（输入哈希守卫，非模型改动零成本回归）、**决策点日志**、**失败分类十一类**（「技术对产品错」单列）、版本 diff 同轴、反馈带 trace id 进队列、事后分析五段。

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 170}}}%%
flowchart LR
    B["一个坏结果（用户 👎 / 存疑 / 客服工单）"] -- "反馈绑定 trace_id" --> TR["打开完整轨迹"]
    TR --> D["决策点日志：第几步、看到了什么、为什么这么选"]
    D --> C["失败打类（检索没找到 / 幻觉 / 越权 / 忘目标 / …）"]
    C --> F{"改哪一侧？"}
    F -- "应用侧：工具、解析、卫士阈值" --> RP["<b>录制回放</b>：模型输出录下来，改完应用重放，不再花模型的钱"]
    F -- "模型侧：prompt、模型、effort" --> EV["变成评测用例，进门禁"]
    RP & EV --> V["版本 diff：变更时间线与指标曲线对齐"]

```

<aside class="notes" markdown="1">
原文 /traceability-record-replay-decision-logs-failure-taxonomy-and-feedback.html。
</aside>

<!-- v -->

### 要点

| 失败类 | 药在哪 |
|---|---|
| 检索没找到 / 找到没用 | 检索层 |
| prompt 约束丢失 | 上下文层 |
| 工具返回错 / 编造工具结果 | 工具与 mask |
| 模型能力不足 | 模型（十一类里的少数） |
| 技术对、产品错 | 产品层 |

- 「幻觉 = 模型问题」——多数的药在检索、prompt、产品层
- 「回放全过就上线」——输入变了要报警重跑

---

## 07 · 运行时可靠性与物理世界的仿真

**结论**：guardrails **同步只放高风险**；高风险断言必须有依据（不要模型自报置信度）；fallback 四级；仿真六层、**闭环才测决策**、sim-to-real、放大长尾、影子、SIL / HIL；数字与物理世界同构。

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 460}}}%%
flowchart TB
    U["请求"] --> I["输入侧检查：注入特征、越界请求、敏感信息（只降概率，不是防线）"]
    I --> LLM["模型 / agent"]
    LLM --> O["输出侧检查：schema、引用是否存在、置信度、幻觉检测、敏感词"]
    O -- "通过" --> Rr["返回用户"]
    O -- "不通过" --> FB["fallback：拒答 + 说明 / 降级到更保守的 prompt 或模型 / 转人工"]
    E["上线前的评测（第一到四篇）发现的失效模式"] -. "做成运行时检查" .-> O

```

<aside class="notes" markdown="1">
原文 /runtime-reliability-and-simulation-for-the-physical-world.html。
</aside>

<!-- v -->

### 要点

| 数字世界 | 物理世界 |
|---|---|
| 评测集 | 场景库 |
| 回放 | 仿真回灌 |
| 影子模式 | 影子驾驶 |
| 灰度 | 分批 OTA |
| 覆盖率 = 新场景比例 | 同 |

- 「让模型输出置信度」——不可靠；要依据

---

## 一个闭环

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 200}}}%%
flowchart LR
    T["trace（05）<br/>会话日志的投影"] --> S["采样成评测集（01）"] --> J["评分：规则 / judge / 人（02）<br/>指标矩阵（03）"] --> G["门禁（04）<br/>改动 = 发布"]
    T --> B["bad case → 失败分类 → 用例（06）"] --> S
    G --> R["运行时检查（07）<br/>评测发现的失效模式"] --> T
```

---

## 常见误区（一）

- 「有 trace 就够了」——看得见不等于能判断
- 「按流量比例采评测集」——难例看不见
- 「等评测集完善再用」——几十条就能分方向
- 「judge 一致率 88% 可做门禁」——报 kappa
- 「用最大的模型当 judge」——去偏比大小重要
- 「每轮平均分代表会话质量」——以会话为单位
{: .fragments}

---

## 常见误区（二）

- 「pass@5 = 生产可靠性」——pass^k
- 「没部署就不会变」——静默升级，探针集
- 「只记前 500 字」——回放做不了
- 「幻觉 = 模型问题」——药多在别处
- 「让模型输出置信度」——要依据
{: .fragments}

---

## 下一步

- **同一路线**：《生产与运维》——门禁之后的发布、SLO 与 runbook；《产品与体验》——量「有用」不是「用了」；《工具、Agent 与 harness》第 9 篇——轨迹评测
- **算法侧**：《经典机器学习》第 10 篇——校准、judge 一致率、多重比较；《后训练》第 8 篇——benchmark 的三关
- 原文总纲：`/evaluation-observability-and-traceability.html`；通关自测在系列总结
