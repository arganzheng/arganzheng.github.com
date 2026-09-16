---
layout: post
title: 工具、Agent 与 harness：模型怎么从回答变成做事（总纲）
subtitle: "Tools, Agents and the Harness: From Answering to Acting"
tags: [AI, LLM, AI-Application, Agent, Harness, MCP, Coding Agent]
catalog: true
---


## 内容简介

《工具、Agent 与 harness：模型怎么从回答变成做事》是一组共九篇的系列文章，是[《AI 应用工程师学习地图》](/ai-application-engineer-learning-roadmap.html)第四层（L4）的正文。前三层讲的模型只**回答**：它读上下文、给出文本。这一层讲它怎么**做事**——调用工具、执行命令、修改文件、操作业务对象——以及让这件事可靠所需要的一切：一个循环、一套协议、一个运行时、一组权限、一层沙箱、一个持久化的会话、几个子 agent、一份记忆、一个把人放在合适位置的机制。业内把模型之外的这一切叫 **harness**（马具）。

2026 年是 harness 成为独立产品的一年。OpenAI 在 9 月 10 日把驱动 Codex 的 harness 作为 **Agents API** 公测——"OpenAI 托管并维护 harness，你选择 agent 的计算环境"，并强调 harness 建在开源的 Codex 代码库上（`codex-rs`，五十多万行 Rust）。DeepSeek 在 8 月 13 日发布 **DeepSeek Harness**（`dsh`）开发者预览，MIT 许可，口号是"一切都是插件"：模型、工具、技能、会话、沙箱、存储、循环、调度、UI 全部是可替换的 Cordis 插件。Anthropic 的 **Claude Code** 把自己的 harness 以 Agent SDK 的形式开放，权限系统、hooks、子 agent、技能都可编程。学术界与社区有 **OpenHarness** 一类开源实现。四个 harness 在同一批问题上给出了不同答案：循环放在哪、工具怎么注册、上下文怎么压、权限怎么判、沙箱用什么、会话怎么存、子 agent 怎么起——本系列第六篇把它们放在源码层面对照。

它回答的问题是：

> **一个 agent 循环的最小形态是什么，成熟的 harness 在它之上加了什么？工具调用的协议（MCP）2026 年长什么样，为什么出现了 tool search 与程序化工具调用？agent 运行时为什么是一种新的服务形态？权限与沙箱怎么设计才能让 PocketOS 那样的事不发生？Codex、DeepSeek Harness、Claude Code、OpenHarness 各自怎么解这些问题？多 agent 什么时候值得？人该站在环内、环上还是环外？**

九篇沿一条线走。**第一篇**解剖最小的 agent 循环——模型读上下文、返回工具调用、应用执行、结果写回、再来——用 Codex 的三层循环（`submission_loop` → turn loop → 工具编排器）与 DeepSeek Harness 的 `agent-loop`（领取 prompt → 在会话日志上开一个 turn → 组装 → 流式 → 分发工具 → 追加）做实例，说明终止条件、循环卫士（重复工具调用提醒、单次工具超时）为什么必要。**第二篇**讲工具调用的协议：MCP 的 2026-07-28 版（无状态核心、多轮往返请求、扩展框架——Tasks、Apps、Skills over MCP——授权加固、由 Linux 基金会下的 Agentic AI Foundation 治理）、工具描述、tool search（定义按需加载）与程序化工具调用（模型写一段程序调用多个工具、只返回结果——Anthropic 的 PTC、OpenAI GPT-5.6 的 Responses PTC、Cloudflare 的 Code Mode、DeepSeek Harness 的 Code 模式与 `ptc-runtime`、Codex 的 `code-mode` crate）。**第三篇**讲运行时——为什么 agent 是传统服务与模型 serving 之外的第三种服务形态：事件溯源的会话日志（DeepSeek Harness 的"模型可见 ⟺ 已记录"不变量、resume / fork / replay 在同一事件流上）、Codex 的 rollout 与 thread-store、LangGraph 的 checkpoint、Temporal 一类 durable execution、OpenAI Agents API 的托管运行时与九家沙箱合作方、挂起等人、事件流推前端。**第四篇**讲长任务的上下文管理与子 agent：L2 第四篇的卸载 / 清理 / 压缩在运行时里的实现（Codex 的 `compact*` 模块、DeepSeek Harness 的 `compaction` 与 `spill` 包组）、plan / todo / goal 作为复述机制、子 agent 的三种形态（Claude Code 的 subagent 与 agent teams、Codex 的 `multi_agents`、DeepSeek Harness 能把 Claude Code 与 Codex 作为子 agent 拉起）。**第五篇**讲权限、沙箱与安全边界——本系列最重要的一篇：Codex 的权限档（`read_only` / `workspace` / `danger_full_access`）、审批策略、`execpolicy` 的 Starlark 前缀规则（allow / prompt / forbidden）、Guardian 审查、三平台沙箱（macOS Seatbelt、Linux bubblewrap + Landlock + seccomp、Windows 受限令牌）；Claude Code 权限判定的六步顺序（hooks → deny → ask → 模式 → allow → 回调）与五种模式；DeepSeek Harness 的 `read-only` / `workspace-write` / `danger-full-access` 与审批策略插件；云沙箱；工具返回里的注入；PocketOS 事件的五环在这一层如何逐环被防住。**第六篇**是源码级对照——语言与架构、循环、工具注册、上下文、权限、沙箱、会话、子 agent、扩展模型、UI 面、模型耦合、基准模式——以及 Agents API（托管）、Agents SDK（库）、Harness（自托管产品）三种交付形态的取舍。**第七篇**讲多 agent：什么时候需要、orchestrator-workers / handoff / 层级三种模式、Claude Code agent teams 与 GPT-5.6 的 ultra 模式、A2A 协议、成本（Anthropic 的研究系统约 15 倍 token）与调试代价、"先用一个 agent 加更好的工具"。**第八篇**讲 memory 与 human-in-the-loop：记忆的写入与遗忘（Anthropic 的 memory tool、Claude Code 自动记忆 25 KB、Codex 的 `memories`）、人从环内退到环上、环外的路径与每一级需要的验证、审批粒度、`request_user_input` 与 MCP elicitation、本体驱动的业务 agent 作为业务侧的 harness。**第九篇**讲可靠性、评测与运营：失败分类学、幂等工具、预算、轨迹评测（SWE-bench Verified、Terminal-Bench 4.0、τ-bench）、trace、每任务成本、从 demo 到生产的清单。

系列的组织原则：**harness 的每一个组件都是对 L1 某条失效模式的防线**。循环卫士防死循环，权限与沙箱防越界（PocketOS），会话日志防"任务跑到一半丢了"，压缩防上下文腐化，子 agent 防预算耗尽，人机分工防不可逆的错误。九篇的每一篇都标出它防的是哪一条、代价是什么。


## 为什么写这个系列？

### harness 决定了同一个模型能走多远

"同样的模型，更好的 harness，更好的 agent"——DeepSeek 把 harness 作为独立产品发布时的理由，也是 OpenAI 把 Codex 的 harness 拆成 Agents API 的理由。2026 年的前沿模型能力相近，产品之间的差距越来越多地来自模型之外：工具集设计得好不好、上下文压得对不对、权限判得准不准、任务中断了能不能续、人在哪一步介入。一个只会调 API 的团队与一个会建 harness 的团队，用同一个模型做出的产品差一个量级。

### 源码公开了，可以对照着讲

这是写这个系列的时机。Codex 的 `codex-rs` 是公开的 Rust 代码库（`core`、`sandboxing`、`execpolicy`、`hooks`、`app-server`、`code-mode` 等几十个 crate），DeepSeek Harness 是公开的 TypeScript 单体仓库（`packages/core` 的 `agent-loop` / `session` / `tools`、`sandbox` / `compaction` / `spill` / `subagent` / `guard` / `ptc-runtime` 等包组，附架构文档与 Cordis 教程），Claude Code 的 Agent SDK 文档把权限判定顺序、子 agent 与 agent teams 的差别、hooks 的生命周期写得很具体。三家在同一批问题上的不同答案，是理解"harness 是什么"最好的教材——比任何框架教程都好。

### 事故告诉我们缺哪一格会怎样

L1 第一篇讲过的三个事件——Replit 的 agent 在代码冻结期删库、PocketOS 的 Cursor agent 用一个无关文件里权限过宽的 token 在 9 秒内删掉生产卷与备份、OpenAI 约 700 个测试中的 agent 入侵 Hugging Face——每一个都能在 harness 的组件表上找到缺的那一格：环境隔离、最小权限、确认门、可回滚、沙箱出口。本系列第五篇逐环讲这些格子怎么填。

### 现有材料的断层

- **框架文档**（LangGraph、各家 Agent SDK）教你用它的 API，不讲它为什么这样设计、别家为什么不这样；
- **供应商的发布博客**各讲自己的 harness，没有横向对照；
- **安全研究**讲 prompt injection 与沙箱逃逸，很少与 agent 的权限模型放在一起讲；
- **"agent 教程"**多停在 ReAct 循环与工具调用，不讲运行时、持久化、权限、子 agent 这些让循环能上生产的部分。

本系列想填补的是从"能写一个 ReAct 循环"到"能为一个业务场景设计 harness——选交付形态、定工具集、建会话日志、写权限规则、配沙箱、决定人站在哪、用轨迹评测证明它可靠"之间的那段路。


## 适合哪些读者？

### 要做 agent 产品的应用工程师

全部九篇。第一、五、九篇是最小必读。

### 平台工程师，要为团队提供 agent 运行时

第三、五、六篇：运行时形态、权限与沙箱、三种交付形态的取舍。

### 想读懂 Codex / DeepSeek Harness / Claude Code 的开发者

第六篇是入口，第一、四、五篇给每个子系统的展开。

### 业务侧的 AI 应用负责人

第五、八篇：agent 能做什么错事、人该站在哪、本体驱动的业务 agent 怎样把验证前移。


## 系列的整体主线

| 篇 | 主题 | 建立的东西 | 防 L1 的哪条失效 | 代价 |
|---|---|---|---|---|
| 1 | 最小循环 | 循环的形态、终止条件、卫士；Codex 与 DeepSeek Harness 的循环实现 | 非确定性（死循环） | 步数与预算上限 |
| 2 | 工具与 MCP | MCP 2026-07-28；描述；tool search；程序化工具调用 | 指令遵循（用错工具）；上下文（定义占预算） | 协议复杂度 |
| 3 | 运行时 | 第三种服务形态；事件溯源会话日志；durable execution；托管运行时 | "任务丢了"；供应商变更（锁定） | 持久化与恢复的工程 |
| 4 | 上下文与子 agent | 运行时里的卸载 / 清理 / 压缩；复述；三种子 agent | 上下文标称 ≠ 有效；预算耗尽 | 子 agent 的 token 与调试 |
| 5 | 权限与沙箱 | 权限档、审批策略、执行策略语言、三平台沙箱、注入 | **越界** | 摩擦与审批疲劳 |
| 6 | 源码对照 | 四个 harness 的十二维对照；三种交付形态 | —（综合） | — |
| 7 | 多 agent | 三种模式；A2A；成本 | 预算耗尽（多 agent 的 15 倍） | token 与调试 |
| 8 | memory 与人 | 记忆的写入 / 遗忘；环内 → 环上 → 环外；本体 harness | 越界（不可逆）；知识截止（记忆） | 人的时间 |
| 9 | 可靠性与评测 | 失败分类学；幂等；轨迹评测；trace；每任务成本 | 全部（验证） | 评测的工程 |

三条贯穿的线：

- **循环之外的一切都是 harness。** 模型只做两件事：决定下一步、读结果。校验、权限、执行、配对、截断、持久化、恢复、压缩、隔离、审批、评测——全是应用侧的。第一篇立这个框，后面八篇填。
- **验证能自动化到什么程度，agent 就能自主到什么程度。** 代码有测试所以 coding agent 最先成熟；业务动作没有测试，所以要把验证前移到本体（第八篇）；不可逆动作永远要确认门（第五篇）。
- **权限范围是错误的上限。** 模型的判断永远不是 100%，让错误不变成灾难的是权限、隔离、确认、可回滚（第五篇）；harness 的价值首先是把错误的上限压低，其次才是让模型更能干。


## 章节结构与分章导读

### 1. 最小 agent 循环的解剖

> 核心问题：**agent 循环的最小形态是什么？它在哪些地方会失控？Codex 与 DeepSeek Harness 的循环各分几层、每层管什么？**

从一个 40 行的循环开始：组装上下文 → 调模型 → 有工具调用则执行并写回 → 否则结束；加上四个卫士——步数上限、token 预算、重复调用检测、单次工具超时。然后看两个生产实现：Codex 的三层——`submission_loop`（会话生命周期，接收 `Op`）→ turn loop（一次用户输入触发：组装 prompt、调 Responses API、处理输出项）→ 工具编排器（`tools/orchestrator.rs`：审批 → 选沙箱 → 首次尝试 → 失败时的升级路径），客户端与会话之间是一对操作 / 事件队列（app-server 的 JSON-RPC 之上）；DeepSeek Harness 的 `agent-loop`——领取排队的 prompt → 在会话日志上开一个 turn → 经 `system-prompt` 组装请求前缀、从日志派生历史 → 流式调模型 → 经工具注册表分发 → 把每个模型可见的事实追加回日志再派生下一步；`guard` 包组的两个卫士：重复工具调用提醒（模型重复完全相同的调用时提醒它换方法或结束）与单次工具超时。Anthropic《Building effective agents》里"工作流 vs agent"的区分作为设计起点：能用固定流程的不要用循环。

### 2. 工具调用与 MCP：协议、tool search 与程序化工具调用

> 核心问题：**MCP 2026-07-28 版改了什么，为什么？工具描述怎么写才让模型用对？tool search 与程序化工具调用各解决什么问题？**

L1 第二篇讲了工具调用的往返协议，这一篇讲工具的**生态协议**。MCP（Model Context Protocol）2025 年 12 月由 Anthropic 捐给 Linux 基金会下的 Agentic AI Foundation（OpenAI、Block 共同创始），TypeScript 与 Python SDK 各过十亿次下载；2026-07-28 版是发布以来最大的修订：**无状态核心**（`initialize` 握手取消，版本与能力协商放进每个请求的 `_meta` 与 `MCP-Protocol-Version` 头，Streamable HTTP 去掉会话 id——能在普通 HTTP 基础设施上横向扩展）、**多轮往返请求**替代服务端发起的请求、**扩展框架**（反向 DNS 标识、独立版本：Tasks——长任务的轮询与持久句柄、MCP Apps——对话内渲染的 UI、Skills over MCP、企业托管授权）、**授权加固**（RFC 9207 的 `iss` 校验、RFC 8707 的 `resource` 参数绑定受众、动态客户端注册弃用改为客户端元数据文档 CIMD），Sampling / Roots / Logging 十二个月弃用窗口，注册表仍在预览。然后是三个工程问题：工具描述的写法（L1 第二篇的延伸，Codex 的 `agents_md` 与 DeepSeek Harness 的 `system-prompt` 怎样把工具说明拼进系统提示）；**tool search**——几十个 MCP 工具的完整 schema 占几千 token，Codex（`tools/handlers/tool_search.rs`）、Claude Code（MCP 工具 schema 延迟加载）、OpenAI Responses 都做成按需加载，代价是这些定义不在缓存前缀里（L2 第五篇）；**程序化工具调用**（PTC）——让模型写一段程序，在沙箱里调用多个工具、处理中间结果、只把最终输出送回上下文：Anthropic 的 programmatic tool calling、OpenAI GPT-5.6 在 Responses 里的 PTC（同时让它兼容零数据保留）、Cloudflare 的 Code Mode、DeepSeek Harness 的 Code 模式（工具经 Code Mode SDK 暴露、一个 TypeScript 程序组合多步）与 `ptc-runtime` 包组（模型写一个程序调用宿主函数、只返回打印输出与返回值）、Codex 的 `code-mode` 系列 crate——它解决的是 L2 的预算问题：中间结果不进上下文。

### 3. agent 运行时：会话、持久化与 durable execution

> 核心问题：**为什么 agent 运行时是一种新的服务形态？事件溯源的会话日志解决什么？durable execution 是什么？托管运行时（Agents API）与自托管各换了什么？**

地图里那张三列表（传统服务 / 模型 serving / agent 运行时）在这里展开：一次调用分钟到小时、几十次模型调用与工具执行交替；每一步都要持久化，崩溃、重启、等人之后从断点续跑；控制流由模型决定、运行时只给边界；可能挂起几小时等审批；并发单位是会话。**事件溯源的会话日志**是这一形态的核心数据结构——DeepSeek Harness 把它作为不变量："任何进入模型请求的东西都必须能从会话日志重建；一个新的模型可见输入需要一个会话事件"，append-only 的 `SessionEvent` 日志是唯一事实来源，resume、fork、search、replay 在同一事件流上操作，Trajectory 视图按来源检视每一条；Codex 的 `rollout` 与 `thread-store`、`app-server` 之上的操作 / 事件队列是同一思想的 Rust 实现。**durable execution**：LangGraph 把 agent 建成带 checkpoint 的状态图、Temporal 一类工作流引擎把每步持久化、崩溃后从 checkpoint 继续——agent 运行时最接近的老概念。**托管运行时**：OpenAI Agents API（2026-09-10 公测，无额外费用、只收 token 与工具费）把会话状态、编排、上下文压缩、恢复放在 OpenAI 侧，你选沙箱（OpenAI 托管、自己的基础设施、或 Blaxel / Cloudflare / Daytona / DigitalOcean / E2B / Modal / Oracle / Runloop / Vercel 九家合作方），一次 API 调用创建能跑几天的 agent；它与 Agents SDK（库，控制权在你的进程里）是两种交付形态，第六篇比较。挂起与唤醒（等人、等外部事件）、事件流推前端、中断与取消、并发会话的隔离。

### 4. 长任务的上下文管理与子 agent

> 核心问题：**L2 讲的卸载 / 清理 / 压缩在运行时里怎么实现？plan / todo / goal 是什么？三种子 agent 各在什么情况下用？**

L2 第四篇讲了策略，这一篇看实现：Codex 的 `core/src/compact*.rs` 系列（本地压缩、远程压缩 v2、token 预算、图片预算、模型回退）；DeepSeek Harness 的 `compaction` 包组（自动压缩、`/compact` 命令、工具结果修剪器先修剪再压缩、图片卸载为占位符）与 `spill` 包组（把全文存到上下文之外、返回一个带取回指引的定位符——L2 第四篇"卸载"的实现）。**复述**的实现：Codex 的 `plan` 工具与 `get_context_remaining`，DeepSeek Harness 的 `plan` / `todo` / `goal` 三个包（记录式规划、todo_write 工具、会话目标）——Manus 的 todo.md 在这里是一等功能。**子 agent**的三种形态：Claude Code 的 subagent（自己的上下文窗口、结果只回主 agent、token 较低）与 agent teams（各自独立、互相直接通信、共享任务列表、每个是独立实例、token 较高，v2.1.178 起免设置步骤但仍是实验特性）；Codex 的 `multi_agents` / `multi_agents_v2` 工具处理器与 `agent-roles` crate；DeepSeek Harness 的 `subagent` 包组——进程内 fork（继承历史）、进程内 spawn（全新）、进程外驱动，以及 `subagent-claude-code` / `subagent-codex` / `subagent-acp`：**一个 harness 把另外两个 harness 作为子 agent 拉起**，这是 2026 年 harness 之间的互操作已经成为事实的证据。什么时候用子 agent（隔离探索性上下文、并行独立子任务、专业化角色），什么时候不用（任务本来就是串行的、结果需要完整过程）。

### 5. 权限、沙箱与安全边界

> 核心问题：**权限档、审批策略、执行策略、沙箱四层各管什么？Codex、Claude Code、DeepSeek Harness 各怎么判一个工具调用能不能执行？PocketOS 的五环在这一层怎么逐环被防住？**

这是本系列最重要的一篇。四层：**权限档**（能碰什么）——Codex 的 `PermissionProfile`：`:read_only`、`:workspace`（只写工作区根目录、`.git` 与 `.codex` 只读、网络可选受限）、`:danger_full_access`；DeepSeek Harness 的 `read-only` / `workspace-write` / `danger-full-access`；Claude Code 的 allow / deny / ask 规则（`Bash(rm *)` 一类作用域规则）。**审批策略**（什么时候问人）——Codex 的 `AskForApproval` 与 `execpolicy`：Starlark 写的前缀规则 `prefix_rule(pattern=[...], decision=allow|prompt|forbidden, justification, match, not_match)`，`match` / `not_match` 是加载时校验的单元测试；Guardian 审查（一个模型审另一个模型的高风险调用）；Claude Code 判定的六步顺序——hooks 先跑（可直接拒绝）→ deny 规则（即使 `bypassPermissions` 也生效）→ ask 规则 → 当前模式（`bypassPermissions` 放行、`acceptEdits` 放行文件操作、`plan` 把写操作一律送回调、`dontAsk` 拒绝所有未决）→ allow 规则 → `canUseTool` 回调；DeepSeek Harness 的审批策略插件。**沙箱**（执行时真正拦住）——Codex 三平台：macOS Seatbelt（`sandbox-exec` + SBPL 策略文件，写根目录之外只读）、Linux bubblewrap + Landlock + seccomp（分离的文件系统策略走 bubblewrap、旧路径走 Landlock）、Windows 受限令牌 + ACL + Job Objects，网络经代理策略；DeepSeek Harness 的 `sandbox-local` / `sandbox-policy` / `sandbox-windows-acl`；云沙箱（E2B、Modal、Daytona 一类 microVM / 容器）。**输入边界**——工具返回是不可信输入，间接注入从这里进来（L1 第二篇、L6 的分层防御）。然后把 PocketOS 的五环放到四层上：无关文件里的 token（权限档：agent 不该读到它 / 凭据不该在工作区）、权限过宽（最小权限）、staging 能碰生产（环境隔离）、无确认（审批策略：删除类命令 `prompt` 或 `forbidden`）、备份同卷（可回滚性）——每一环在一个成熟的 harness 里都有对应的格子。审批疲劳与"策略化授权"（从逐个批到批一类）。

### 6. 源码级对照：Codex、DeepSeek Harness、Claude Code、OpenHarness

> 核心问题：**四个 harness 在十二个维度上各怎么选？为什么？Agents API、Agents SDK、自托管 harness 三种交付形态怎么取舍？**

一张十二行的对照表：语言与架构（Codex：Rust 工作区几十个 crate、`core` 为重心；DeepSeek Harness：TypeScript 单体仓库、几十个 `@deepseek-ai/dsh-*` 包、Cordis 插件树、profile / bundle / patch 三层配置；Claude Code：TypeScript，产品闭源、Agent SDK 与文件系统约定开放；OpenHarness：Python 开源、轻量、`oh` 一条命令）、循环（三层 vs 日志驱动 vs SDK 封装）、工具注册（`tools/registry.rs` + 路由 + 并行 vs 作用域注册表 + 守卫执行管线 vs 内置 + MCP + hooks）、上下文（`compact*` vs `compaction` + `spill` vs auto-compact 83.5%）、权限与审批（`execpolicy` Starlark + Guardian vs 审批策略插件 vs 六步判定）、沙箱（三平台原生 vs `sandbox` 包组 vs 工作目录与命令限制 + 云沙箱）、会话（`rollout` / `thread-store` vs 事件溯源 JSONL + 检查点 vs 会话文件）、子 agent（`multi_agents` vs 七种后端含 Claude Code / Codex vs subagent / agent teams）、扩展模型（Codex 的 plugins / hooks / skills / MCP vs 一切皆插件 + 四种模式 vs hooks / skills / plugins / MCP）、UI 面（TUI / app-server JSON-RPC / exec vs web / headless / sdk / acp 四个 profile vs CLI / IDE / SDK）、模型耦合（Responses API vs 多供应商含 pi-ai 目录 vs Claude 专用）、基准（DeepSeek Harness 的 Minimal 模式——只留 shell 与文件编辑器，为公平比较模型）。三种交付形态：**Agents API**（托管：会话、编排、压缩、恢复在 OpenAI 侧，你选沙箱，零基础设施，绑定 OpenAI 模型；早期用户报告 4 倍延迟降低、60% 每任务成本降低、86% 失败响应减少）、**Agents SDK / Agent SDK**（库：控制权在你的进程，自己管持久化）、**自托管 harness**（DeepSeek Harness 一类：全部可替换、多模型、自己运维）。每个维度写"为什么这样选"，最后一节是"从每家学什么"。

### 7. 多 agent：编排、handoff 与 A2A

> 核心问题：**什么时候需要多 agent？三种模式各适合什么？A2A 是什么？成本与调试代价有多大？**

先说不需要的情况：多数任务一个 agent 加更好的工具就够，多 agent 是为了三件事——上下文隔离、并行、专业化。三种模式：orchestrator-workers（一个主 agent 分解任务、多个子 agent 并行、主 agent 综合——Anthropic 的多 agent 研究系统，代价是约 15 倍于单次对话的 token）、handoff（OpenAI Agents SDK 的模式：控制权在 agent 之间移交、每个 agent 有自己的工具与指令）、层级（GPT-5.6 的 ultra 模式与 Responses 的 multi-agent beta：一次请求内并发子 agent 并综合）。Claude Code 的 agent teams（互相直接通信、共享任务列表）与 subagent（只回主 agent）的取舍表。**A2A**（Agent2Agent，Google 2025 年 6 月捐给 Linux 基金会）：跨厂商 agent 之间的任务协议——agent card、任务生命周期、与 MCP 的分工（MCP 是 agent 到工具，A2A 是 agent 到 agent）。成本、调试（多条轨迹的 trace）、失败传播、以及"先用一个 agent"的纪律。

### 8. memory 与 human-in-the-loop

> 核心问题：**记忆怎么写、怎么忘？人从环内退到环上、环外各需要什么？审批的粒度怎么定？本体驱动的业务 agent 怎样把验证前移？**

记忆是一种特殊的检索（L3）：会话内记忆就是上下文，跨会话记忆要存、要取、要忘。实现：Anthropic 的 memory tool（模型自己驱动的文件式记忆 `/memories`）、Claude Code 的自动记忆（上限 200 行 / 25 KB，压缩后重注入）、Codex 的 `memories` crate、DeepSeek Harness 的 `goal` / `schedule`（会话目标与定时跟进）；写入策略（什么值得记）、遗忘（过期、冲突、用户要求删除）、记忆的权限（L3 第四篇）。**人机分工**：地图里那张六级表（单步问答 → 短循环 → 长循环 → 策略化授权 → 后台异步 → 无人在环）展开——每升一级把哪类验证从人手里交给系统；审批粒度按风险分级（只读自动、写入确认、不可逆二次确认）；`request_user_input`（Codex 的处理器、MCP 2026 的 elicitation）作为"agent 主动问人"的协议；后台 agent 的异步交付（提 PR、生成草稿）与 on-the-loop 的监控形态。**本体驱动的业务 agent**（Palantir AIP 一类）作为业务侧的 harness：动作有类型、校验、权限，提案先暂存再审批——把"业务动作没有测试可跑"这个问题用结构解决（L3 第六篇讲了本体本身，这里讲它作为 harness）。"绕过障碍"要被识别为需要人介入的信号而不是自主性。

### 9. 可靠性、评测与运营 agent

> 核心问题：**agent 的失败有哪几类？怎么设计幂等的工具与预算？轨迹怎么评？从 demo 到生产要过哪些关？**

失败分类学：幻觉的动作、死循环、预算耗尽、工具半成功、权限拒绝、上下文腐化、注入、中断未恢复——每类的检测与应对。幂等的工具设计（L1 第六篇的延伸）、预算（步数、token、美元、时间）、循环检测（DeepSeek Harness 的 guard 是运行时内置的例子）、部分结果的交付。**轨迹评测**：结果指标（任务完成率）之外看过程——步数、成本、是否越权、是否走弯路；公开基准（SWE-bench Verified、Terminal-Bench 2.1 → 4.0 分数从 90 掉到 31 的饱和—加难循环、τ-bench 的多轮工具使用）与它们的局限（脚手架不同不可横比——L1 第五篇）；自己的轨迹评测集从 trace 里采。trace 的结构（会话 → 任务 → 步骤三级、每步的完整输入输出、决策点日志——L5 展开）。每任务成本的分布（p50 / p95）。从 demo 到生产的清单：权限档与审批策略、沙箱、会话持久化、压缩与卸载、预算与卫士、幂等工具、trace、轨迹评测集、fallback、人在哪一级。

### 10. 系列总结与通关自测

一张总表回顾九篇各自回答的问题与必记的数字，逐篇的核心结论与常见误解，贯穿全系列的几条线，然后是三段自测——判断与计算、跨篇综合、面试题——加一份"读过 / 掌握 / 能教人"的判据。


## 贯穿全系列的实践线

本系列不设配套实验。每篇末尾有一节**实践建议**：

| 篇 | 实践建议的内容 |
|---|---|
| 1 | 手写一个带四个卫士的 40 行循环；读一遍 Codex `tools/orchestrator.rs` 或 DeepSeek Harness `agent-loop` 的 README |
| 2 | 给你的工具集做一次描述审计；工具超过 20 个时评估 tool search；有多步数据处理时试 PTC |
| 3 | 把会话改成 append-only 事件日志；验证崩溃后能从日志恢复；决定托管还是自托管 |
| 4 | 给运行时接上 L2 的三条线；把探索性任务交给子 agent 并比较 token |
| 5 | 写权限档与审批策略（删除类 forbidden / prompt）；把执行放进沙箱；用 PocketOS 五环自检 |
| 6 | 用对照表给自己的 harness 打分，找出空格 |
| 7 | 数一数你的任务里真正需要并行或隔离的比例，再决定多 agent |
| 8 | 定每类动作的风险级与审批粒度；给后台 agent 配监控面板 |
| 9 | 建轨迹评测集；写失败分类；把每任务成本分布上仪表盘 |


## 阅读路径建议

### 第一遍怎么读

第一篇；第二篇的 MCP 一节与 PTC 一节；第三篇的会话日志一节；第五篇全文；第六篇的对照表；第九篇的清单。约两个半小时。

### 完整学习路径

1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9，做每篇自测，最后做第十篇。

### 按角色读

- 做 coding agent / 平台：6 → 5 → 3 → 4 → 2；
- 做业务 agent：8 → 5 → 3 → 9；
- 只想读懂源码：6 → 1 → 5 → 4。


## 本系列的边界

- **不讲 agent 能力的训练**：工具调用数据、多轮环境、RL——属于算法地图 L5；RL 的 rollout 基础设施属于 Infra 地图 09。
- **不讲检索**：agentic retrieval 属于 L3；本系列只把检索当作一类工具。
- **不讲评测方法论本身**：judge、评测集运营属于 L5；本系列只讲 agent 专属的轨迹评测。
- **不讲 prompt injection 的完整防御体系**：属于 L6；本系列讲它作为权限与沙箱设计的输入。
- **不讲物理世界的 agent**：自动驾驶的 harness 在地图里作为对照案例，本系列只在第八篇的人机分工里引用它的 SAE 分级。


## 前置要求与说明

### 前置要求

- L1（工具调用协议、失效模式、成本）、L2（上下文预算与压缩）、L3（检索作为工具、本体）；
- 后端基础：进程与沙箱、消息队列、状态机、事件溯源这些概念不需要解释。

### 版本与事实基线

以 **2026 年 9 月中旬**的公开材料与源码为准：

| 对象 | 版本 / 来源 |
|---|---|
| OpenAI Codex | `openai/codex` 主线（2026-09-16 的快照，`codex-rs` 工作区）；引用 crate 与模块路径，不引用行号 |
| DeepSeek Harness | `deepseek-ai/deepseek-harness` 主线（2026-09-15 的快照，v0.1 developer preview，2026-08-13 发布，MIT）；引用包组与包名；文档站与《DeepSeek Harness Architecture》 |
| Claude Code / Agent SDK | 文档站 2026-09（权限、子 agent、agent teams v2.1.178+、hooks、settingSources） |
| OpenHarness | `HKUDS/OpenHarness`（2026-04 起，MIT） |
| MCP | 规范 2026-07-28（前一版 2025-11-25）；Agentic AI Foundation |
| A2A | Linux 基金会项目（2025-06 起） |
| OpenAI Agents API | 2026-09-10 公测；Agents SDK |
| Anthropic | 《Building effective agents》（2024-12）、多 agent 研究系统（2025-06）、memory tool、programmatic tool calling、Agent SDK |
| 事件 | Replit（2025-07）、PocketOS（2026-04）、OpenAI–Hugging Face（2026-07） |
| 基准 | SWE-bench Verified、Terminal-Bench 2.1 / 3.0 / 4.0、τ-bench |

两个开源 harness 都在快速迭代（DeepSeek Harness 明确写着"会有破坏兼容的变更"），文中的模块与包名以上述快照为准，读者读到时可能已变；机制层面的结论比路径稳定。


## 章节目录

1. [最小 agent 循环的解剖：从 40 行到生产级](/anatomy-of-the-agent-loop.html)
2. [工具调用与 MCP：协议、tool search 与程序化工具调用](/tool-calling-mcp-tool-search-and-programmatic-tool-calling.html)
3. [agent 运行时：会话、持久化与 durable execution](/agent-runtime-sessions-persistence-and-durable-execution.html)
4. [长任务的上下文管理与子 agent](/long-horizon-context-management-and-subagents.html)
5. [权限、沙箱与安全边界](/permissions-sandboxes-and-security-boundaries-for-agents.html)
6. [源码级对照：Codex、DeepSeek Harness、Claude Code 与 OpenHarness](/coding-agent-harness-comparison-codex-deepseek-harness-claude-code.html)
7. [多 agent：编排、handoff 与 A2A](/multi-agent-orchestration-handoff-and-a2a.html)
8. [memory 与 human-in-the-loop](/agent-memory-and-human-in-the-loop.html)
9. [可靠性、评测与运营 agent](/agent-reliability-evaluation-and-operations.html)
10. [系列总结与通关自测](/agent-and-harness-series-recap-and-self-test.html)


## 最终目标

读完本系列，面对一个要让模型"做事"的场景，读者应该能够：

| 追问 | 答案来自 |
|---|---|
| 这个任务需要循环吗，还是固定工作流就够？循环的卫士设在哪？ | 第一篇 |
| 工具用 MCP 还是自定义？几十个工具时怎么办？中间结果怎么不进上下文？ | 第二篇 |
| 任务中断后怎么续？会话存什么？托管还是自托管运行时？ | 第三篇 |
| 跑到第 40 步上下文满了怎么办？什么该交给子 agent？ | 第四篇 |
| 这个工具调用该自动执行、问人还是禁止？沙箱用什么？PocketOS 的五环我防住了几环？ | 第五篇 |
| Codex、DeepSeek Harness、Claude Code 各怎么解这些问题，我该学谁？ | 第六篇 |
| 需要多个 agent 吗？用哪种模式？代价多少？ | 第七篇 |
| 记忆存什么、忘什么？人站在哪一级？业务动作没有测试怎么验证？ | 第八篇 |
| 这个 agent 可靠吗？轨迹怎么评？上生产前还缺什么？ | 第九篇 |

这就是 L4 要建立的能力：**把 agent 当作一个有运行时、有权限、有沙箱、有会话、有预算、有人在合适位置的系统来设计**，而不是一个会调工具的 prompt。
