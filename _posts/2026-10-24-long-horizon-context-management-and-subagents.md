---
layout: post
series: agent-and-harness
title: "工具、Agent 与 harness（04）：长任务的上下文管理与子 agent"
subtitle: "Long-Horizon Context Management and Subagents"
tags: [AI, LLM, AI-Application, Agent, Harness, Context Engineering, Subagent]
catalog: true
---

L2 第四篇讲了长任务上下文管理的**策略**：隔离 → 卸载 → 清理 → 压缩，代价递增；复述对抗漂移。这一篇看**实现**——这些策略在 harness 的运行时里是什么样的代码与配置：Codex 的 `compact*` 模块族、DeepSeek Harness 的 `compaction` 与 `spill` 包组、`plan` / `todo` / `goal` 三个包、Claude Code 的 auto-compact 与子 agent。然后讲隔离手段里最重要的一种——**子 agent**——的三种形态与它们的取舍：Claude Code 的 subagent 与 agent teams、Codex 的 `multi_agents`、DeepSeek Harness 的七种子 agent 后端——其中包括把 Claude Code 与 Codex 作为子 agent 拉起。

一个反复出现的数字值得先记住：Anthropic 的多 agent 研究系统报告 agent 用的 token 约是聊天的 4 倍、多 agent 系统约是聊天的 15 倍。子 agent 是隔离上下文最有效的手段，也是成本放大最快的手段——两面要一起看。

本篇要回答的核心问题是：

> **L2 的卸载 / 清理 / 压缩在 Codex 与 DeepSeek Harness 里各是什么模块，怎么配合？[^q0] plan / todo / goal 作为复述机制怎么实现？[^q1] 子 agent 有哪三种形态，各在什么情况下用、成本多少？[^q2]**

## 一、总览

### 1. 策略到实现

| L2 的策略 | Codex | DeepSeek Harness | Claude Code |
|---|---|---|---|
| 卸载（大结果出上下文留引用） | 工具返回的截断与 `view_image` 一类按需读取 | **`spill` 包组**：存储服务 + 本地后端 + 结果策略，返回带取回指引的定位符 | 子 agent 隔离大量读取；工具返回截断 |
| 清理（删已消费的工具返回） | 会话记忆压缩优先于摘要（L2 第四篇） | `compaction-tool-result-pruner`：压缩前先修剪超大工具输出 | 微压缩旧工具返回 |
| 压缩（摘要替换历史） | `compact.rs`、`compact_remote_v2*.rs`（服务端加密）、`compact_token_budget.rs`、`compact_model_fallback.rs` | `compaction` 包组：`compaction-basic` 自动 + `/compact` 命令 + `compaction-image-offload` | auto-compact 约 83.5% |
| 复述（目标回末尾） | `plan` 工具处理器、`get_context_remaining` 让模型看剩余预算 | `plan`（记录式规划）、`todo`（`todo_write` 工具）、`goal`（会话目标） | todo 工具、CLAUDE.md 重注入 |
| 隔离（子 agent） | `multi_agents` / `multi_agents_v2` 处理器、`agent-roles` crate | `subagent` 包组七种后端 | subagent / agent teams |

### 2. 本文的章节安排

第二章卸载与清理的实现；第三章压缩的实现；第四章复述；第五章子 agent 的三种形态；第六章子 agent 的成本与取舍；第七章实践建议。

## 二、卸载与清理

### 1. DeepSeek Harness 的 `spill`

`spill/` 是一个独立包组——"把全文存到模型上下文之外，返回一个带取回指引的定位符"——分三个包：存储服务（`spill`）、本地文件系统后端（`spill-local`）、结果策略（决定什么时候溢出）。它是 L2 第四篇"卸载"原则的一等实现：工具返回超过阈值不进上下文，模型看到的是"内容已存为 `spill://…`，共 N 行，前几行如下，用 `read` 取回"。作为独立包组的意义是**可替换**：把本地后端换成对象存储、把策略换成按工具类型的阈值，不动循环。

### 2. Codex 的截断

Codex 没有单独的 spill 概念，工具返回在处理器里按上限截断（shell 输出的行数与字节上限），大内容（文件、图片）用专门的工具按需读取（`view_image`、文件读取工具的范围参数）。仓库规范里对"模型可见上下文"有硬规则："不能有无界的项——注入模型上下文的一切都要有有界大小与硬上限；**没有大于 10K token 的项**；能超过 1K token 的新单项按 P0 审查。"这是"卸载"作为代码审查规则的形态。

### 3. 清理

DeepSeek Harness 的 `compaction-tool-result-pruner` 在压缩之前先修剪超大的工具输出——"这样要压缩的东西更少"，正是 L2 第四篇"先清理再压缩"的顺序。Codex 的会话记忆压缩（用已结构化的任务状态替代摘要、多数情况不调模型）是另一种"压缩前的便宜一步"。Claude Code 的微压缩同理。

## 三、压缩

### 1. Codex 的模块族

`core/src` 下的 `compact.rs`（本地压缩：用模型生成摘要）、`compact_remote_history.rs` 与 `compact_remote_v2*.rs`（服务端压缩：调 Responses 的 compact 端点拿加密 item——L2 第四篇；v2 有单独的尝试逻辑与图片预算）、`compact_token_budget.rs`（触发阈值与预算计算）、`compact_model_fallback.rs`（压缩用的模型不可用时的回退）。触发点在 turn loop 的边界（第一篇），压缩结果作为一条 rollout 记录进日志（第三篇）。

### 2. DeepSeek Harness 的 `compaction` 包组

包组的说明："让长对话在接近模型上下文上限时继续工作：随 token 压力累积自动把旧历史浓缩成摘要，按需用 `/compact`，超大工具输出可以先修剪，图片能力不足的路由把图片换成占位符。"五个包：`compaction`（接缝）、`compaction-basic`（自动压缩）、`command-compact`（`/compact` 命令）、`compaction-tool-result-pruner`（修剪）、`compaction-image-offload`（图片卸载）。SDK 文档里提到默认组合"带显式组合的语义检查点策略"——压缩策略是配置出来的，不是写死的。压缩摘要作为一个会话事件进日志（"模型可见 ⟺ 已记录"），有独立的 token 上限。

### 3. 共同点

两家都把压缩做成**多个可组合的步骤**（修剪 / 会话记忆 → 摘要 → 回退），都把结果**写进日志**，都在**思考链结束的边界**触发（L1 第三篇）。差别在摘要形态（Codex 服务端加密 vs DeepSeek Harness 可读事件）与可替换性（Codex 的模块是代码、DeepSeek Harness 的是插件）——第六篇对照。

## 四、复述

### 1. 三个包

DeepSeek Harness 把 Manus 的 todo.md（L2 第四篇）做成三个一等功能：`plan/`（"记录式规划"——计划作为事件进日志）、`todo/`（`todo_write` 工具——模型维护任务列表）、`goal/`（"会话目标"——整个会话的目标，与 `schedule` 的定时跟进配合）。它们的共同作用是**把目标与进度作为结构化事件保持在模型可见的近端**，对抗 L2 第一篇讲的漂移；同时是压缩的现成材料（摘要里直接带任务状态）。

### 2. Codex 的 `plan` 与 `get_context_remaining`

`tools/handlers/plan.rs` 让模型写与更新计划；`get_context_remaining.rs` 让模型**看到自己还剩多少上下文预算**——这是一个有意思的设计：把预算变成模型的输入，让它自己决定何时收敛、何时卸载。Codex 的会话记忆压缩靠的就是这些结构化的任务状态。

### 3. Claude Code

todo 工具与压缩后从磁盘重注入的 CLAUDE.md（L2 第一篇）：项目级的常驻目标靠重注入，任务级的进度靠 todo。

## 五、子 agent 的三种形态

### 1. 为什么

三个理由：**上下文隔离**（探索性搜索读几十个文件，主窗口只要摘要——L2 第一篇的 Claude Code 时间线）、**并行**（几个独立子任务同时跑）、**专业化**（不同的系统提示、工具集、权限、甚至模型）。没有这三个理由之一就不要用——第六章讲代价。

### 2. Claude Code：subagent 与 agent teams

| | subagent | agent teams（实验，v2.1.178+） |
|---|---|---|
| 上下文 | 自己的窗口；结果回调用者 | 自己的窗口；完全独立 |
| 通信 | 只向主 agent 报告结果 | 队友之间直接通信 |
| 协调 | 主 agent 管全部工作 | 共享任务列表，自协调 |
| 适合 | 只要结果的聚焦任务 | 需要讨论与协作的复杂工作 |
| token | 较低：结果摘要回主上下文 | 较高：每个队友是独立的 Claude 实例 |

subagent 的定义有三种方式：程序化（Agent SDK 的 `agents` 参数）、文件（`.claude/agents/*.md`，前置元数据 `description` / `tools` / `model` / `permissionMode`）、内置（`general-purpose`、`Explore`、`Plan`）；主 agent 按每个 subagent 的 `description` 决定是否派发——又是"描述是 prompt"。subagent 继承父会话的权限，多数内置的用受限工具集。agent teams 默认关闭（`CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`），文档明写"在会话恢复、任务协调、关闭行为上有已知限制"——第七篇讲多 agent 时再回来。

### 3. Codex：`multi_agents`

`tools/handlers/multi_agents*.rs`（含 v2）是 Codex 的子 agent 工具处理器，`agent-roles` crate 定义角色，`agent-graph-store` 存 agent 之间的关系图；GPT-5.6 的 ultra 模式与 Responses 的 multi-agent beta（"一次请求内运行并发子 agent 并综合"）是它在 API 层的对应（L1 第三篇）。

### 4. DeepSeek Harness：七种后端

`subagent/` 包组把"委派"做成一个接缝加多种后端：

| 后端 | 含义 |
|---|---|
| `subagent-spawn-in-process` | 进程内全新的子 agent（隔离的上下文） |
| `subagent-fork-in-process` | 进程内**继承历史**的子 agent（带着父会话的上下文分叉——事件溯源让 fork 成为复制前缀） |
| `subagent-in-process-driver` | 进程内驱动 |
| `subagent-dsh-sdk` | 通过 SDK 起一个进程外的 dsh |
| `subagent-acp` | 通过 ACP（自动化协议）起外部 agent |
| **`subagent-claude-code`** | 把 **Claude Code** 作为子 agent 拉起 |
| **`subagent-codex`** | 把 **Codex** 作为子 agent 拉起 |

最后两个值得停一下：一个 harness 可以把另外两家的 harness 作为自己的子 agent——委派一个子任务给 Claude Code 或 Codex，拿回结果。`hooks/` 包组还有 Claude Code / Codex 的 hooks 桥接。这说明 2026 年 harness 之间的**互操作已经是事实**：它们共享 MCP、共享 SKILL.md、共享 AGENTS.md（L2 第六篇），现在还能互相调用。"该用哪个 harness"的问题正在变成"该怎么组合"。

### 5. 子 agent 的上下文设计

无论哪种形态，子 agent 的效果取决于两段文本：**派发指令**（任务、边界、期望的返回格式——"返回文件路径列表与每处一句话说明，不要贴代码"）与**返回的摘要**（主 agent 拿到的全部）。子 agent 自己也是一个 agent：有自己的预算、卫士、权限（Claude Code 的 subagent 可以指定 `permissionMode` 与工具子集——一个只读的 code-reviewer 是常见配置）。fork 型子 agent 继承历史所以启动时上下文就大，适合"接着当前状态并行做一件事"；spawn 型从零开始，适合"独立探索"。

## 六、子 agent 的成本与取舍

### 1. 15 倍

Anthropic 的数字：agent 约 4 倍于聊天、多 agent 约 15 倍。原因：每个子 agent 有自己的系统提示、工具定义、读取的内容（都是输入 token），并行的子 agent 之间无法共享缓存前缀（各自的历史不同），主 agent 还要读所有摘要。子 agent 的 token 不因"结果只回摘要"而消失——它只是不进主窗口，账单上都在。

### 2. 什么时候值得

| 情形 | 值得吗 |
|---|---|
| 探索性搜索（读几十个文件找线索） | 值得：主窗口省下几万 token 的噪声 |
| 几个真正独立的子任务 | 值得：并行省墙钟时间 |
| 需要不同权限 / 工具 / 模型的角色（只读审查、危险操作隔离） | 值得：安全边界 |
| 串行的、每步依赖上一步的任务 | 不值得：没有并行收益，多出派发与摘要的开销 |
| 结果需要完整过程（不是摘要） | 不值得：摘要丢信息 |
| 为了"看起来像团队" | 不值得 |

### 3. 调试

子 agent 让 trace 变成一棵树（会话 → 子会话 → 步骤——第九篇、L5），失败可能发生在子 agent 里而主 agent 只看到"任务失败"或一个误导性的摘要。每个子 agent 的完整轨迹要能从日志追到（DeepSeek Harness 的"发现每个创建的子 agent"是 `subagent` 接缝的一部分）。

## 七、实践建议

1. **给运行时接 L2 的三条线**：卸载阈值（spill 或截断 + 定位符）、清理（压缩前修剪）、压缩（在思考链边界、写进日志、有回退模型）。
2. **让模型看到预算**：提供一个 `get_context_remaining` 类工具或在系统提示里注入剩余预算，观察模型是否更早收敛。
3. **把目标做成事件**：plan / todo / goal 进日志、每步可见，作为压缩的材料。
4. **子 agent 先做只读的探索型**：spawn、受限工具、返回结构化摘要；比较主窗口 token 与端到端成本。
5. **每个子 agent 有自己的预算与卫士**，轨迹可从父会话追到。
6. **试一次跨 harness 委派**：用 DeepSeek Harness 的 `subagent-claude-code` 或自己的 ACP 桥接，把一个子任务交给另一家 harness，看结果格式与成本。

## 八、本文小结

- 策略到实现：卸载——DeepSeek Harness 的 `spill` 包组（存储 + 后端 + 策略，返回定位符）、Codex 的截断与"无大于 10K token 的项"规则；清理——`compaction-tool-result-pruner`、Codex 会话记忆压缩、Claude Code 微压缩；压缩——Codex `compact*` 模块族（本地 / 远程加密 v2 / 预算 / 回退）、DeepSeek Harness `compaction` 五个包（自动 / 命令 / 修剪 / 图片卸载）、Claude Code 83.5%；都是多步可组合、写进日志、在思考链边界触发。
- 复述：DeepSeek Harness 的 `plan` / `todo` / `goal` 三个包把目标与进度做成模型近端的事件；Codex 的 `plan` 与 `get_context_remaining`（让模型看到剩余预算）；Claude Code 的 todo 与 CLAUDE.md 重注入。
- 子 agent 三种形态：Claude Code 的 subagent（结果回主 agent、token 低）与 agent teams（独立、互通、共享任务列表、token 高、实验）；Codex 的 `multi_agents` / `agent-roles` 与 GPT-5.6 ultra；DeepSeek Harness 七种后端——spawn / fork（继承历史）/ 驱动 / SDK / ACP / **Claude Code / Codex**——harness 互操作已是事实。
- 成本：多 agent 约 15 倍 token；只在隔离、并行、专业化三个理由之一成立时用；子 agent 的效果取决于派发指令与返回摘要；trace 是树。

## 九、自测

1. 一个 agent 的工具经常返回 30K token 的日志。Codex 的"无大于 10K token 的项"规则与 DeepSeek Harness 的 `spill` 各会怎么处理？

   <details markdown="1"><summary>答案</summary>
   Codex：处理器按上限截断输出（行数 / 字节），超出部分不进上下文，模型需要时用范围参数或专门工具再读——规则是代码审查层面的硬约束。DeepSeek Harness：`spill` 的结果策略判断超阈值，存储服务把全文存到本地后端，上下文里只留带取回指引的定位符（`spill://…`、行数、预览），模型用读取工具按需取回；后端与策略可替换。两者都是 L2 的"卸载"。详见[第二章](#二卸载与清理)。
   </details>

2. 为什么 `compaction-tool-result-pruner` 要在压缩**之前**运行？这对应 L2 的哪条原则？

   <details markdown="1"><summary>答案</summary>
   先修剪超大的、已被消费的工具输出，要压缩的历史就少，摘要调用更便宜、损失更小；且修剪是可恢复的（内容可再取）而摘要是有损的。对应 L2 第四篇"代价递增：清理在压缩之前"。详见[第二章](#二卸载与清理)、[第三章](#三压缩)。
   </details>

3. Codex 的 `get_context_remaining` 工具把什么变成了模型的输入？预期效果是什么？

   <details markdown="1"><summary>答案</summary>
   把剩余的上下文预算变成模型可查的信息，让模型自己决定何时收敛、何时卸载或总结，而不是被运行时在阈值处强制压缩。预期是模型在预算紧张时主动写计划 / 缩短输出，压缩发生得更少、更可控——把一部分上下文管理从运行时交给了模型。详见[第四章](#四复述)。
   </details>

4. 一个任务：把 12 个微服务的 README 各总结成三句话并汇总。用 subagent 还是 agent teams？spawn 还是 fork？为什么？

   <details markdown="1"><summary>答案</summary>
   subagent（结果只需摘要回主 agent，不需要子任务之间讨论），spawn（每个子任务独立、不需要父会话的历史，从零开始上下文最小）；12 个并行，主 agent 汇总 12 条三句话摘要。agent teams 与 fork 都会多花 token 而没有收益。详见[第五章](#五子-agent-的三种形态)、[第六章](#六子-agent-的成本与取舍)。
   </details>

5. DeepSeek Harness 能把 Claude Code 与 Codex 作为子 agent 拉起。这依赖哪些已有的互操作基础？对"选哪个 harness"这个问题意味着什么？

   <details markdown="1"><summary>答案</summary>
   依赖：三家都是 MCP client（共享工具）、都读 SKILL.md（共享技能）、都读项目指令文件（AGENTS.md / CLAUDE.md）、Claude Code 与 Codex 都有可脚本化的无头 / SDK 入口与 hooks（`hooks/` 包组有两家的桥接）、ACP 一类自动化协议。意味着 harness 之间不是互斥的选择——可以用一个作为运行时与日志中心、把子任务委派给另一个更擅长的；问题从"选哪个"变成"怎么组合"。详见[第五章](#五子-agent-的三种形态)。
   </details>

## 下一篇

[权限、沙箱与安全边界](/permissions-sandboxes-and-security-boundaries-for-agents.html)

[^q0]: 卸载：DeepSeek Harness 的 `spill` 包组（存储服务 `spill` + 本地后端 `spill-local` + 结果策略，全文存到上下文外、返回带取回指引的定位符，各部分可替换）；Codex 在处理器里按上限截断、大内容用专门工具按需读，并有代码审查规则"无无界项、无大于 10K token 的项、超 1K 的新项 P0 审查"。清理：DeepSeek Harness `compaction-tool-result-pruner` 压缩前先修剪超大工具输出；Codex 的会话记忆压缩用结构化任务状态替代摘要；Claude Code 微压缩。压缩：Codex `compact.rs`（本地）、`compact_remote_history` / `compact_remote_v2*`（服务端加密 item、图片预算）、`compact_token_budget`（阈值）、`compact_model_fallback`（回退）；DeepSeek Harness `compaction` 包组五个包（接缝、`compaction-basic` 自动、`command-compact`、修剪器、`compaction-image-offload`），策略可配置。共同点：多步可组合、结果写进日志、在思考链边界触发。详见[第二章](#二卸载与清理)、[第三章](#三压缩)。

[^q1]: 复述是把目标与进度作为结构化事件保持在模型可见的近端，对抗漂移并作为压缩材料。DeepSeek Harness 三个包：`plan`（记录式规划，计划是日志事件）、`todo`（`todo_write` 工具，模型维护任务列表）、`goal`（会话目标，与 `schedule` 定时跟进配合）。Codex：`tools/handlers/plan.rs` 让模型写与更新计划，`get_context_remaining.rs` 让模型看到剩余上下文预算并自己决定何时收敛——把一部分上下文管理交给模型；会话记忆压缩靠这些结构化状态多数情况不调模型。Claude Code：todo 工具管任务级进度，CLAUDE.md 压缩后从磁盘重注入管项目级目标。详见[第四章](#四复述)。

[^q2]: 三种形态。Claude Code 的 subagent（自己的窗口、结果只回主 agent、主 agent 管协调、token 较低；程序化 / 文件 / 内置三种定义，按 `description` 派发，继承权限可限工具与模式）与 agent teams（各自独立、直接互通、共享任务列表、每个是独立实例、token 高、实验特性有已知限制）。Codex 的 `multi_agents` / `multi_agents_v2` 处理器、`agent-roles`、`agent-graph-store`，API 层对应 GPT-5.6 ultra 与 Responses multi-agent beta。DeepSeek Harness `subagent` 包组七种后端：进程内 spawn（全新）、fork（继承历史，事件溯源让分叉是复制前缀）、驱动、`dsh-sdk` 进程外、ACP、以及把 Claude Code 与 Codex 作为子 agent 拉起——互操作已是事实。用的理由只有隔离、并行、专业化；成本约 15 倍聊天（Anthropic），子 agent 的 token 不因只回摘要而消失、并行的无法共享缓存；串行任务、需要完整过程的任务不用；效果取决于派发指令与返回摘要；每个子 agent 有自己的预算与卫士，trace 是树。详见[第五章](#五子-agent-的三种形态)、[第六章](#六子-agent-的成本与取舍)。
