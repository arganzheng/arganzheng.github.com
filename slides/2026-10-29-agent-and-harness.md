---
layout: slides
title: "工具、Agent 与 harness：模型怎么从回答变成做事"
subtitle: "系列精华 · 九篇正文每篇一页：循环之外的一切都是 harness"
permalink: /slides/agent-and-harness.html
series: agent-and-harness
date: 2026-10-29 23:30:00 +0800
author: arganzheng
description: "《工具、Agent 与 harness》系列的分享用幻灯片：最小循环的四个卫士、MCP 与 tool search、事件溯源的运行时与 durable execution、长任务的压缩与子 agent、权限四层与沙箱、Codex / DeepSeek Harness / Claude Code / OpenHarness 源码对照、多 agent、记忆与 human-in-the-loop、可靠性与评测。"
theme: white
transition: slide
---

## 这个系列的三句话主张

> **循环之外的一切都是 harness**；**权限范围是错误的上限**；**能自动验证到什么程度，就能自主到什么程度**。

<aside class="notes" markdown="1">
总纲：/tools-agents-and-harness.html。
</aside>

---

## 几篇怎么连起来

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 160}}}%%
flowchart TB
    A1["01 最小循环"] --> A2["02 工具与 MCP"]
    A1 --> A3["03 运行时"]
    A1 --> A4["04 上下文与子 agent"]
    A2 & A3 --> A5["05 权限与沙箱"]
    A1 & A2 & A3 & A4 & A5 --> A6["06 源码对照"]
    A4 --> A7["07 多 agent"]
    A3 & A5 --> A8["08 记忆与人审"]
    A5 & A7 & A8 --> A9["09 可靠性、评测与运营"]
```

---

## 01 · 最小 agent 循环：从 40 行到生产级

**40 行循环 + 四个卫士 + 两个出口才够生产——能用工作流就不用 agent**

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 460}}}%%
flowchart LR
    A["① 组装上下文"] --> B["② 调模型"] --> C{"③ 解析输出"}
    C --> D["④ 权限 → 沙箱"] --> E["⑤ 写回结果"] --> A
    C --> END1["出口一<br/>正常结束"]
    G["卫士<br/>步数/token/美元/时间"] -.-> A
    G -- "耗尽" --> END2["出口二<br/>部分结果"]
    G -- "触发" --> END3["出口三<br/>降级给人"]
    style END2 fill:#fff3e0,stroke:#c98a00
    style END3 fill:#fde8e8,stroke:#c0392b

```

<aside class="notes" markdown="1">
原文 /anatomy-of-the-agent-loop.html。
各篇关键词：01 40 行 + 四个卫士 + 出口；02 工具从哪来、怎么被发现；03 会话日志 / durable execution；05 权限范围是错误的上限；06 Codex / DeepSeek Harness / Claude Code / OpenHarness；07 编排 / handoff / A2A。
③ 解析输出分两类：最终回答 → 出口一；tool_call → ④ 权限 → 沙箱执行。
③ 解析输出分两类：最终回答 → 出口一；tool_call → ④ 权限 → 沙箱执行。
卫士每圈查步数 / token / 美元 / 时间。
③ 的输出若是 tool_call 走 ④⑤ 循环，最终回答走出口一。
</aside>

<!-- v -->

## 要点

**四个卫士都在循环里；编排器不升级**

```python
while True:                                   # 四个卫士在这里：步数 / 预算 / 超时 / 重复
    resp = model(messages, tools)
    if not resp.tool_calls: return resp.text  # 出口一：模型停
    for call in resp.tool_calls:              # 编排器：审批 → 沙箱首试 → 不升级
        result = execute(call)                # 一切「模型可见」的都要「已记录」
        messages.append(tool_result(call, result))
```

<aside class="notes" markdown="1">
补充要点：编排器原则：审批 → 沙箱首试 → `never` / `on-request` 不升级；「模型可见 ⟺ 已记录」
</aside>

---

## 02 · 工具调用与 MCP：工具从哪来、怎么被发现

**工具按 schema 进注册表；tool search 按需加载、PTC 让中间结果不进上下文**

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 460}}}%%
flowchart TB
    subgraph SRC["工具从哪来"]
        direction LR
        S1["应用函数"] --- S2["MCP server"] --- S3["供应商内置"] --- S4["技能 / 脚本"]
    end
    SRC --> REG["工具注册表：schema"]
    REG -- "全塞上下文" --> CTX1["② 定义层"]
    REG -- "tool search" --> CTX2["按需加载"]
    CTX1 & CTX2 --> MD["模型：返回 tool_call"]
    MD -- "程序化调用" --> EX["执行层"]

```

<aside class="notes" markdown="1">
原文 /tool-calling-mcp-tool-search-and-programmatic-tool-calling.html。
</aside>

<!-- v -->

## 要点

**工具按 schema 进注册表；PTC 让中间结果不进上下文**

| 项 | 数 |
|---|---|
| 治理 | AAIF；SDK 十亿下载 |
| 授权 | DCR → CIMD；`resource` 绑定受众 |
| 何时用 tool search | 二十个以上工具 |
| PTC | 四家都有 |


<!-- v -->


### 常见误区

- 「MCP 只是一个有会话的 RPC」——已改为无状态核心

<!-- v -->

### 传统调用 vs 程序化工具调用（PTC）

![传统调用每个中间结果都回上下文；PTC 写一段程序调多个工具，只有最终结果进上下文](/img/in-post/tool-calling-mcp-traditional-vs-ptc.svg){: style="max-height: 340px"}

- 中间结果不进上下文——长工具链省掉的是上下文预算与延迟，四家都有

---

## 03 · agent 运行时：第三种服务形态

**会话日志只追加是核心——resume / fork / replay 都是同一条流的重放**

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 170}}}%%
flowchart LR
    E1["1 用户"] --> E2["2 模型"] --> E3["3 结果"] --> E4["4 审批"] --> E5["5 模型"] --> E6["6 压缩"] --> E7["7 …"]
    E3 -.-> D["当时上下文"]
    E4 -.-> F["新会话"]
    E7 -.-> C["当前上下文"]
```

<aside class="notes" markdown="1">
原文 /agent-runtime-sessions-persistence-and-durable-execution.html。
resume / fork / replay 都是同一条流的重放。
E3 → 当时上下文 与 E7 → 当前上下文 都是重放；E4 → 新会话是分叉。
链条是会话日志（只追加）；E3→当时上下文、E7→当前上下文是重放，E4→新会话是分叉。
</aside>

<!-- v -->

## 要点

**日志只追加——resume / fork / replay 是同一条流**

| 项 | 数 |
|---|---|
| Agents API | 2026-09-10；九家沙箱；无额外费；失败响应 −86% |
| 100 个并发会话在等审批 | 不能占 100 个 worker——挂起释放 |
| 托管的代价 | 绑供应商模型、日志在供应商侧 |


---

## 04 · 长任务的上下文管理与子 agent

**隔离 → 卸载 → 清理 → 压缩；子 agent 只回摘要，但多 agent 约 15 倍 token**

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 190}}}%%
flowchart LR
    A["<b>隔离</b>：子 agent 自带上下文<br/>只回摘要，代价≈0"] --> B["<b>卸载</b>：大返回存外面<br/>上下文留定位符"]
    B --> C["<b>清理</b>：修剪过期<br/>工具输出"]
    C --> D["<b>压缩</b>：模型调用摘要旧历史<br/>最贵、有损、缓存失效"]
    style A fill:#eefaf0,stroke:#4d9a5c
    style D fill:#fde8e8,stroke:#c0392b

```

<aside class="notes" markdown="1">
原文 /long-horizon-context-management-and-subagents.html。
卸载 = DeepSeek Harness 的 spill；清理 = compaction-tool-result-pruner；压缩 = Codex compact* / Claude Code auto-compact。
</aside>

<!-- v -->

## 要点

**子 agent 只回摘要——隔离换上下文，不省钱**

| 规则 | 为什么 |
|---|---|
| 「无 > 10K token 的项」 | 任何单项超过就 spill 到文件 |
| subagent vs agent teams | 前者回摘要、后者共享目标 |
| 串行任务 | **不拆**子 agent——没有并行收益，多派发与摘要开销 |

<aside class="notes" markdown="1">
补充要点：「子 agent 只回摘要所以便宜」——token 不进主窗口，但都在账单上
</aside>

---

## 05 · 权限、沙箱与安全边界：压低错误的上限

**权限四层缺一不可——权限档、审批、执行策略、沙箱；deny 高于一切**

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 150}}}%%
flowchart LR
    CALL["模型返回<br/>工具调用"] --> L1{"① 权限档"}
    L1 -->|"超档"| DENY["拒绝"]
    L1 -->|"在档内"| L2{"② 审批 · ③ 执行"}
    L2 -->|"forbidden"| DENY
    L2 -->|"需审批"| ASK["请人批"]
    ASK -->|"拒绝"| DENY
    ASK -->|"批准"| SB["④ 沙箱"]
    L2 -->|"自动"| SB
    SB -->|"成功"| OUT["结果写回"]
    SB -. "被拦 → 升级（下图）" .-> ASK

    classDef step fill:#fff7e0,stroke:#c98a00,stroke-width:2px,color:#222
    classDef dec fill:#eef6ff,stroke:#5b8fd6,color:#222
    classDef deny fill:#fdecea,stroke:#c0392b,color:#222
    class CALL,ASK,SB,OUT step
    class L1,L2 dec
    class DENY deny
```

<!-- v -->

### 沙箱拦住以后：升级回到审批

**「never / on-request」决定拦截后是死路还是新审批——升级不是绕开沙箱，是重新问人**

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 200}}}%%
flowchart TB
    SB["④ 沙箱执行"] -->|"被沙箱拦住"| ESC{"允许升级？"}
    ESC -->|"never / on-request：不"| DNO["返回沙箱拒绝说明"]
    ESC -->|"是：新审批"| ASK["挂起请人批"]
    ASK -->|"拒绝"| DENY["拒绝，结果送回模型"]
    ASK -->|"批准"| SB

    classDef step fill:#fff7e0,stroke:#c98a00,stroke-width:2px,color:#222
    classDef dec fill:#eef6ff,stroke:#5b8fd6,color:#222
    classDef deny fill:#fdecea,stroke:#c0392b,color:#222
    class SB,ASK step
    class ESC dec
    class DNO,DENY deny
```

<aside class="notes" markdown="1">
原文 /permissions-sandboxes-and-security-boundaries-for-agents.html。调用示例：shell: rm -rf ./build · write: src/x.py · http: DELETE /volumes/7；审批可选 Guardian 模型先审；沙箱在受限的文件系统 / 网络 / 进程里执行。
</aside>

<!-- v -->

## 要点

**命令文本看不出效果——所以审批后还要沙箱**

| 层 | 例 |
|---|---|
| 权限档 | `:read_only` / `:workspace` / `:danger_full_access` |
| 审批策略 | allow / prompt / forbidden 带 `match` / `not_match`；策略化授权对抗审批疲劳 |
| 执行策略 | 命令文本看不出脚本的真实效果——所以还要沙箱 |
| 沙箱 | macOS Seatbelt / Linux Landlock + seccomp / Windows |


---

## 06 · 源码级对照：四个 harness 各怎么选

**四个 harness 换一样付一样——状态、压缩、策略语言、沙箱、模型耦合五个轴**

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 300}}}%%
flowchart TB
    H["harness 五组件：循环 · 工具协议 · 运行时 · 上下文管理 · 权限沙箱"]
    H --> C1["<b>Codex</b>（Rust）<br/>单核工作区 · 状态在对象 · 策略语言"]
    H --> C2["<b>DeepSeek Harness</b>（TS）<br/>一切插件 · 日志即状态"]
    C1 --> C3["<b>Claude Code</b>（TS，闭源）<br/>Agent SDK · 模式审批 · auto-compact"]
    C2 --> C4["<b>OpenHarness</b><br/>Claude Code 的开源复刻"]

```

<aside class="notes" markdown="1">
原文 /coding-agent-harness-comparison-codex-deepseek-harness-claude-code.html。
Codex 三平台原生沙箱；DeepSeek Harness 可读事件、沙箱是接缝；OpenHarness 读它看结构。
</aside>

<!-- v -->

## 要点

**五轴对照：状态、压缩、策略、沙箱、耦合**

| | Codex | DeepSeek Harness | Claude Code | OpenHarness |
|---|---|---|---|---|
| 实现 | Rust 单核 | Python 轻量 | 闭源 + SDK | TS 一切皆插件 |
| 状态 | 对象 | 日志驱动 | 对象 + 加密压缩 | 插件 |
| 权限 | `execpolicy` Starlark | 规则 | 六步 deny 优先 | 插件 |
| 沙箱 | 原生三平台 | 接缝 | 原生 | 接缝 |
| 模型耦合 | 深 | 浅（能拉起别家） | 深 | 浅；Minimal 模式 |

---

## 07 · 多 agent：编排、handoff 与 A2A

**多 agent 只为隔离、并行、专业化——代价 15 倍 token、缓存不共享**

![orchestrator-workers 并行综合、handoff 移交控制权、层级经理管小组——三种多 agent 模式](/img/in-post/multi-agent-orchestration-three-patterns.svg){: style="max-height: 360px"}

<aside class="notes" markdown="1">
原文 /multi-agent-orchestration-handoff-and-a2a.html。三层评测。
</aside>

<!-- v -->

## 要点

**多 agent 只为隔离、并行、专业化——代价 15 倍**

| 量 | 数 |
|---|---|
| 多 agent 成本 | 约 15 倍 token |
| 缓存 | 只有分叉前的前缀共享 |
| A2A | 组织内先用子 agent |

<!-- v -->


### 常见误区

- 「并行 agent 共享缓存」——只有分叉前的前缀
- 「现在就要接 A2A」——组织内先用子 agent

<!-- v -->

### handoff 的一条时间线

![客服 handoff：分流 agent 判断退款 → transfer_to_refund 移交控制权与上下文，超阈值再移交人工](/img/in-post/multi-agent-orchestration-handoff-timeline.svg){: style="max-height: 340px"}

- handoff 移交的是**控制权 + 上下文**——移交什么上下文是设计点，不是默认值

---

## 08 · memory 与 human-in-the-loop

**记忆是检索（写 / 取 / 忘、上限 25 KB）；人机分工六级、每级换验证方式**

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 480}}}%%
flowchart TB
    S["一次会话结束"] --> W{"写什么？"}
    W -- "偏好、坑、进度" --> ST["存储：带来源与时间戳"]
    W -- "能重查的" --> X["不写"]
    ST --> R["<b>取</b>：进上下文 ③ 层<br/>（硬上限 25 KB）"]
    ST --> F["<b>忘</b>：过期 / 冲突 / 要求 → 删"]
    R -. "错记忆跨会话当事实" .-> S

```

<aside class="notes" markdown="1">
原文 /agent-memory-and-human-in-the-loop.html。
写：下次会用到、且别处不易取回。
</aside>

<!-- v -->

## 要点

**不可逆止于提案；六级分工各换验证**

| 级 | 人做什么 | 验证靠什么 |
|---|---|---|
| 1 | 逐步确认 | 人 |
| 2–3 | 批准计划 / 抽检 | 人 + 规则 |
| 4–5 | 例外处理 | 结构（本体、类型、权限）+ 测试 |
| 6 | 不在环 | 全自动验证 + 可逆 |

<aside class="notes" markdown="1">
补充要点：提案先于落库；风险五级
</aside>

<!-- v -->


### 常见误区

- 「直接发送给客户 = 无人在环成功」——不可逆且不能验证

---

## 09 · 可靠性、评测与运营

**十类失败只有一类是异常；轨迹评测看六维——不只完成率**

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 460}}}%%
flowchart TB
    R["agent 跑任务"] --> O{"结束方式"}
    O -- "回答" --> V["结果校验"]
    O -- "任一到顶" --> P["交付部分结果"]
    O -- "卫士触发" --> H["降级给人"]
    V & P & H --> TR["每步进 trace"]
    TR --> C["失败打类"]
    C --> E["坏轨迹 → 评测用例"]
    E -. "改动先过门禁" .-> R

```

<aside class="notes" markdown="1">
原文 /agent-reliability-evaluation-and-operations.html。
十类失败每类有药；轨迹评测看过程。
十类失败每类有药。
</aside>

<!-- v -->

## 要点

**同一基准三协议三个数——看六维不只完成率**

| 量 | 数 |
|---|---|
| Terminal-Bench | 90.6 / 30.0 / 31.2——同一基准三种协议三个数 |
| demo 到生产 | 十二行清单 |
| 改工具 | 录制回放，不重跑模型回归 |


<!-- v -->


### 常见误区

- 「拒绝率降到零是好事」——审批疲劳或规则太松

---

## 三句话在九篇里

| 主张 | 落点 |
|---|---|
| **循环之外的一切都是 harness** | 卫士与出口（01）、工具发现（02）、会话日志（03）、压缩与子 agent（04）、权限（05）、四家对照（06） |
| **权限范围是错误的上限** | 四层缺一不可（05）、deny 高于一切、沙箱看真实效果、基础设施三环（05）、越权是评测项（09） |
| **能验证到什么程度就能自主到什么程度** | 六级分工每级换验证（08）、本体前移验证、不可逆不无人（08）、轨迹六维（09） |

---

## 常见误区（一）

- 「循环只要模型停就停」——四个卫士
- 「所有多步任务都做成 agent」——能工作流就工作流
- 「tool search 一定提高缓存命中率」——看绝对量
- 「用 Web 框架直接跑 agent」——durable execution
- 「审批等待占着 worker 没问题」——挂起释放
- 「子 agent 只回摘要所以便宜」——15 倍
{: .fragments}

---

## 常见误区（二）

- 「审批够了不需要沙箱」——命令文本看不出效果
- 「hook 返回 allow 就放行」——deny 优先
- 「有最好的 harness」——每个选择换一样付一样
- 「多 agent 更聪明」——只为隔离、并行、专业化
- 「记忆越多越好」——上限、来源、遗忘
- 「完成率 85% 就可靠」——六维
{: .fragments}

---

## 下一步

- **同一路线**：《Prompt 与上下文工程》——第 4 篇压缩的应用侧；《检索与知识》——agentic retrieval 与本体；《评测与可观测》——轨迹评测与 trace；《生产与运维》——运营 agent
- **Infra 侧**：《RL 后训练 Infra》第 6 篇——训练 agent 的 rollout 是同一个循环
- 原文总纲：`/tools-agents-and-harness.html`；通关自测在系列总结
