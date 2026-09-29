---
layout: slides
title: "生产化与运营：让 AI 应用可靠、可控、可持续地跑"
subtitle: "系列精华 · 七篇正文每篇一页，每篇预防一类事故"
permalink: /slides/production-and-operations.html
series: production-and-operations
date: 2026-10-05
author: arganzheng
description: "《生产化与运营》系列的分享用幻灯片：模型网关的七项职责、从账单到每任务成本、TTFT 到 agent 多步的延迟分解、注入 / 供应链 / 数据泄漏的五层防御、审计链与 AI Act 时间表、改动 = 发布的开关与 runbook、数据飞轮与物理世界的分批 OTA。"
theme: white
transition: slide
---

## 这个系列的三句话主张

> **四个指标并列**（质量、成本、延迟、安全）；**防御在架构不在措辞**；**改动 = 发布，依赖也会变**。

<aside class="notes" markdown="1">
总纲：/production-and-operations-for-ai-applications.html。每篇从一类事故出发。
</aside>

---

## 几篇怎么连起来

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 170}}}%%
flowchart TB
    O1["01 模型网关<br/>所有调用经过的一层：密钥、路由、归因"] --> O2["02 成本工程<br/>从账单到每任务成本"]
    O1 --> O3["03 延迟工程<br/>TTFT 到 agent 的多步"]
    O1 --> O4["04 安全<br/>注入、供应链、数据泄漏"]
    O1 & O4 --> O5["05 治理与合规<br/>审计链、保留、AI Act、内容标识"]
    O2 & O3 & O4 --> O6["06 发布工程<br/>开关、钉版本、runbook、SLO"]
    O6 --> O7["07 数据飞轮与物理世界的上线"]
```

---

## 01 · 模型网关：所有调用经过的一层

**预防的事故**：key 散落、锁死供应商、无归因。**结论**：七项职责——密钥、路由 / fallback、限流 / 配额、缓存、归因、日志 / 脱敏、抽象；**不做业务逻辑、不改请求体、不默认语义缓存**、要高可用。

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 420}}}%%
flowchart TB
    subgraph APPS["十二个服务"]
        direction LR
        A1["客服"] --- A2["搜索"] --- A3["报告 agent"] --- A4["…"]
    end
    APPS -- "逻辑模型名 chat-default<br/>网关凭据（按服务 / 租户发）" --> GW["<b>模型网关</b><br/>① 认证与密钥 ② 路由与 fallback ③ 限流与配额<br/>④ 成本归因 ⑤ 日志 / trace ⑥ 缓存 ⑦ 内容检查"]
    GW -- "供应商 key 只在这里" --> P1["OpenAI"] & P2["Anthropic"] & P3["自托管 vLLM"]
    GW -. "限流时统一退避，不是十二个服务各自重试" .-> P1

```

<aside class="notes" markdown="1">
原文 /model-gateway-the-layer-every-call-goes-through.html。
</aside>

<!-- v -->

### 要点

| 形态 | 例 |
|---|---|
| 自托管 | LiteLLM |
| 托管市场 | OpenRouter |
| 边缘 | Cloudflare AI Gateway |
| 抽象的边界 | 推理状态与缓存不能抽象掉 |

- 「网关做语义缓存省钱」——相似不等于同答案；默认关
- 「网关注入 request_id 到 system」——毁缓存前缀

---

## 02 · 成本工程：从账单到每任务成本

**预防的事故**：账单吓一跳。**结论**：归因三视图（用户 / 功能 / 模型）；预算 80 / 95 / 100% + **六级降级梯子**；九种手段各有量级；**按任务算 p50 / p95**；单位经济学三决策；FinOps 节奏。

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 180}}}%%
flowchart LR
    B["月底账单：一个总数"] -- "没有归因就没有优化" --> T["每次调用打标签：<br/>租户 · 功能 · prompt 版本 · 模型快照 · 任务 id"]
    T --> L["按天的账本：标签 × 五项 usage × 美元<br/>（未缓存输入 / 缓存读 / 缓存写 / 输出含思考 / 工具费）"]
    L --> V["按<b>任务</b>看，不按调用看：<br/>p50 与 p95 的每任务成本"]
    V --> F["发现：5% 的任务跑到 40 步 × 100K 上下文，<br/>花了 60% 的钱（长尾占大头）"]
    F --> A["药：每任务预算与降级（第三章）、缓存与排列、<br/>小模型分流、批处理……九种手段各有量级（第四章）"]

```

<aside class="notes" markdown="1">
原文 /cost-engineering-from-the-bill-to-cost-per-task.html。
</aside>

<!-- v -->

### 要点

| 手段 | 量级 |
|---|---|
| 级联 | 50–80% |
| 缓存输入 | 50–90% |
| Batch API | 50% |
| 长尾 | 5% 的任务花 60% |

- 价值 / 成本 < 3 重做；定价覆盖 p95（重度用户 5–10×）
- 「月底看账单」——长尾与跳变；每小时异常归因；「评测太贵先砍」——一次事故更贵

---

## 03 · 延迟工程：从 TTFT 到 agent 的多步

**预防的事故**：用户等。**结论**：**七段分解各有药**；感知 ≠ 绝对、流式三层；p95 预算分段 + 四层超时；agent 上 **减步数 > 并行 > 级联 > 每步更快 > 后台**。

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 460}}}%%
flowchart TB
    Q["请求"] --> A["应用侧准备：检索 1.2 s（rerank 50 个候选）"]
    A --> N["网络 + 排队：1 s（峰值时段）"]
    N --> P["prefill / TTFT：2.5 s（60K 上下文，前缀有时间戳 → 缓存未命中）"]
    P --> D["输出：3 s（400 token，不流式）"]
    D --> U["用户看到：p95 = 9 s，但 4 s 时已经关了页面"]
    A -. "rerank 候选 50 → 20、两路召回并行" .-> A
    P -. "去掉时间戳让缓存命中" .-> P
    D -. "流式：首字 1 s 就出现" .-> D
    style U fill:#fde8e8,stroke:#c0392b

```

<aside class="notes" markdown="1">
原文 /latency-engineering-from-ttft-to-multi-step-agents.html。
</aside>

<!-- v -->

### 要点

| 段 | 药 |
|---|---|
| 排队 | 限流、优先级 |
| prefill | **缓存命中是最大杠杆** |
| 思考 | effort 选档 |
| decode | 输出长度约束、流式 |
| 工具 | 并行、超时 |
| 多步 | 减步数 |
| 网络 | 就近、连接复用 |

- TTFT ≈ 1 s 的预算例；三个数字一起报（TTFT / TPOT / 总）
- 「慢就换快模型」——七段各有药；「硬砍 max_tokens」——截断；配 prompt 约束与结构化

---

## 04 · 安全：注入、供应链与数据泄漏

**预防的事故**：EchoLeak、MCP 投毒、沙箱逃逸。**结论**：OWASP 十条里七条已有机制；四个事件的链路**逐环有层拦**；**五层防御**；**MCP server 是权限主体**；密钥七条；红队进门禁。

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 170}}}%%
flowchart LR
    IN["① 输入侧：来源标记与分区<br/>外部内容标为数据 · 分区标记 · 过滤（只降概率）"] --> M["模型"]
    M --> P["② 权限与沙箱（L4 第五篇四层）<br/>注入让模型想做 · 四层让它做不到"]
    P --> O["③ 输出处理<br/>渲染前脱敏链接 / 图片 / HTML · 执行前检查命令 / SQL · CSP"]
    O --> D["④ 数据流控制<br/>读了敏感数据的会话不能对外发 · 跨资源边界要审批 · 工具间数据流"]
    D --> R["⑤ 监控与红队<br/>异常动作检测 · 注入用例进评测集 · 持续红队"]
    R -.->|"发现新链路"| IN

    classDef step fill:#fff7e0,stroke:#c98a00,stroke-width:2px,color:#222
    class IN,M,P,O,D,R step
```

<aside class="notes" markdown="1">
原文 /security-prompt-injection-supply-chain-and-data-exfiltration.html。
</aside>

<!-- v -->

### 要点

| 事件 | 链路 | 拦在哪 |
|---|---|---|
| EchoLeak | 七环：邮件 → 检索 → 注入 → 读 → 拼 URL → 渲染 → 外传 | 任一环：权限、输出检查、CSP |
| Cursor 三 CVE | 同模式「读 → 写不该写的文件」 | 配置 / MCP 清单 / 沙箱策略 deny 写 |
| MCP 描述投毒 | 架构性 | 钉哈希、再批准、独立凭据 |

- 「加注入分类器就安全了」——EchoLeak 绕过了；「工作区可写没问题」——配置文件要 deny

---

## 05 · 治理与合规：审计链、保留、AI Act 与内容标识

**预防的事故**：审计拿不出、数据被训、罚款、未标识。**结论**：**审计链 = 会话日志 + trace + 决策点 + 保留 / 不可篡改 / 导出**；数据流图核合同；AI Act 推迟的只是高风险；标识一套两格式；责任矩阵。

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 300}}}%%
flowchart TB
    Q["合规部门：上月这条回复依据什么给出的？"] --> O["输出 → trace_id → 会话 / 任务（L4 第三篇的事件日志）"]
    O --> M["模型 id + 快照、prompt 版本"]
    O --> R["检索到的文档 id 与当时的权限"]
    O --> A["谁批准了这版 prompt 上线（发布记录）"]
    M & R & A --> E["审计链 = 前几层已经记下的东西串起来，<br/>治理只是定期抽一条<b>验证</b>证据拿得出来"]
    E -. "还要回答：数据存在哪、存多久、供应商能不能拿去训练；<br/>AI 生成内容有没有按法规标识" .-> Q

```

<aside class="notes" markdown="1">
原文 /governance-and-compliance-audit-retention-ai-act-and-labeling.html。
</aside>

<!-- v -->

### 要点

| 法规 | 日期 | 罚则 |
|---|---|---|
| AI Act GPAI 义务 | 2026-08-02 | 1,500 万 € / 3% |
| Article 50 标识 | 2026-08 | |
| Annex III 高风险 | 2027-12-02（推迟） | |
| Annex I | 2028-08-02 | |
| 中国生成内容标识 | 2025-09-01 | |

- 「有 trace 就有审计链」——要完整内容、保留期、不可篡改、导出；「供应商网页说不训练」——核合同

---

## 06 · 发布工程：改动 = 发布

**结论**：**十类算发布**（prompt、模型、工具、schema、检索索引……）；开关三态 + **kill switch 上线前存在并演练**；钉一切；依赖变更进同一流程；五本 runbook；四类 SLO。

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 460}}}%%
flowchart TB
    C["任何会改变模型行为的改动<br/>prompt · 模型快照 · effort · schema · 工具集 · 检索配置 · harness 版本"] -- "改动 = 发布" --> V["不可变版本 + 标签<br/>staging / canary / production"]
    V --> G["门禁：评测集（L5 第四篇）"]
    G -- "过" --> CN["灰度：特性开关按租户 / 比例放量"]
    CN -- "在线指标正常" --> PR["production"]
    CN -- "回归" --> RB["一键回滚到上一个版本"]
    PR -. "供应商别名指向新快照：钉住快照 + 依赖变更单独发布，<br/>两个变化不叠在一起" .-> V
    K["kill switch：供应商宕机 40 分钟 → 关掉 AI 功能走降级路径<br/>runbook 写清谁在什么信号下按"] -.-> PR

```

<aside class="notes" markdown="1">
原文 /release-engineering-for-ai-features-switches-pinning-runbooks-and-slos.html。
</aside>

<!-- v -->

### 要点

| 规则 | |
|---|---|
| 控制台改动 | 禁止 |
| 浮动别名 | 禁止——静默升级 |
| 回滚 | 前确认（缓存、状态） |
| 弃用 | 是有截止日的发布 |

- 「小改动直接上」——改动 = 发布；「出事再做 kill switch」——上线前存在并演练

---

## 07 · 数据飞轮与物理世界的上线

**预防的事故**：系统不变好；车队更新出事。**结论**：飞轮六环三判据；**微调在末端做行为优化**（bad case 不直接微调，筛过的好例子）；四道门；物理世界**分阶段准入 + 安全案例 + 分批 OTA**；高风险 agent 的发布应更像 OTA。

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 180}}}%%
flowchart LR
    U["使用<br/>真实流量"] --> F["反馈<br/>隐式（采纳 / 修改 / 重试 / 放弃）· 显式（点踩）· 系统（错误 / 卫士 / 拒答）<br/>全部带 trace id"]
    F --> Q["bad case 队列<br/>owner · SLA · 分类 · 去重"]
    Q --> E["评测集<br/>每周补 · hold-out · 版本绑定"]
    E --> I["改进<br/>prompt · 检索 · 工具描述 · 策略 · 路由"]
    I --> G["门禁 → 影子 → 灰度<br/>（L5 第四篇 · 第六篇）"]
    G --> U
    E -->|"几千条真实（输入，理想输出）对"| FT["微调小模型<br/>行为优化：编译规则、降成本、提一致性"]
    FT --> G
    U -->|"授权 · 隔离 · 脱敏 · 可退出"| D["用户数据治理（第五篇）"]
    D -.-> F

    classDef step fill:#fff7e0,stroke:#c98a00,stroke-width:2px,color:#222
    classDef gate fill:#eef6ff,stroke:#5b8fd6,color:#222
    class U,F,Q,E,I,FT step
    class G,D gate
```

<aside class="notes" markdown="1">
原文 /data-flywheel-and-deploying-to-the-physical-world.html。
</aside>

<!-- v -->

### 要点

| 节奏 | |
|---|---|
| 周 | 新用例进评测集 |
| 月 | 发布 |
| 季 | 分布变化审视 |

- 「bad case 直接微调」——学进错误；「OTA 一次全量」——分批、准入、停止条件

---

## 三句话在七篇里

| 主张 | 落点 |
|---|---|
| **四个指标并列** | 网关归因（01）、每任务成本（02）、七段延迟（03）、安全进门禁（04）——任何一个单独优化都会伤另一个 |
| **防御在架构不在措辞** | 网关不改请求体（01）、五层防御（04）、MCP 是权限主体（04）、审计链不可篡改（05） |
| **改动 = 发布，依赖也会变** | 十类算发布（06）、钉一切与探针（06）、弃用是发布（06）、分批 OTA（07） |

---

## 常见误区（一）

- 「每个服务自己拿 key 调模型」——散落、无归因、无 fallback
- 「网关做语义缓存省钱」——默认关
- 「月底看账单」——每小时归因
- 「按平均用户定价」——覆盖 p95
- 「慢就换快模型」——七段各有药
- 「加注入分类器就安全了」——五层
{: .fragments}

---

## 常见误区（二）

- 「MCP server 装了就完」——权限主体
- 「有 trace 就有审计链」——保留、不可篡改、导出
- 「AI Act 延期了停项目」——推迟的只是高风险
- 「小改动直接上」——改动 = 发布
- 「浮动别名方便」——静默升级
- 「bad case 直接微调」——学进错误
{: .fragments}

---

## 下一步

- **同一路线**：《评测与可观测》——门禁的另一半；《模型作为组件》第 6 篇——客户端纪律；《工具、Agent 与 harness》第 5 篇——沙箱与权限的 agent 侧
- **Infra 侧**：《AI 平台工程》第 7 篇——网关之下的配额与路由
- 原文总纲：`/production-and-operations-for-ai-applications.html`；通关自测在系列总结
