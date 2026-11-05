---
layout: slides
title: "Prompt 与上下文工程：模型这一步该看到什么"
subtitle: "系列精华 · 六篇正文每篇一页：上下文是有限、有序、有版本的资源"
permalink: /slides/context-engineering.html
series: context-engineering
date: 2026-10-11 23:30:00 +0800
author: arganzheng
description: "《Prompt 与上下文工程》系列的分享用幻灯片：上下文的七层与增长规律、prompt 里稳定的模式与不稳定的措辞、约束解码与 schema 设计、隔离 → 卸载 → 清理 → 压缩、prompt caching 的前缀排列、prompt 当代码管与上下文 vs 检索。"
theme: white
transition: slide
---

## 这个系列的一句话主张

> 上下文是**有限、有序、有版本**的资源：每一段都要**为它占的预算辩护**，每一次排列都有**成本含义**，每一次改动都是**一次发布**。

<aside class="notes" markdown="1">
总纲：/prompt-and-context-engineering.html。
</aside>

---

## 几篇怎么连起来

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 170}}}%%
flowchart TB
    X1["01 上下文的解剖"] --> X2["02 prompt 设计"]
    X1 --> X3["03 结构化输出"]
    X1 --> X4["04 预算与压缩"]
    X1 & X4 --> X5["05 prompt caching 与排列"]
    X2 & X3 & X5 --> X6["06 prompt 当代码管"]
```

---

## 01 · 上下文的解剖：模型这一步看到什么

**七层按变化频率分；会话按 \(S+(k-1)(r+o)\) 长——目标漂到中间是算术必然**

![七层在第 1 步与第 30 步的体积：常驻层不变，工具返回长成大头，当前输入被挤到最底下](/img/in-post/anatomy-of-the-context-window-seven-layers-growth.svg){: style="max-height: 330px"}

<aside class="notes" markdown="1">
原文 /anatomy-of-the-context-window.html。Claude Code 自动记忆 ≤ 25 KB、auto-compact 约 83.5%（预留 33K）；Manus 约 50 次工具调用、输入输出 100 : 1；50 步填满 200K。
各篇关键词：01 七层 / 增长规律 / 注意力预算；02 第一层：稳定的模式与不稳定的措辞；03 约束解码 / schema；04 隔离 → 卸载 → 清理 → 压缩；05 前缀逐字节相同才命中；06 版本 / 评测门禁 / A/B、上下文 vs 检索。
</aside>

---

## 02 · prompt 设计：稳定的模式与不稳定的措辞

**六部分模式不变、措辞随模型调——推理模型上 CoT 冗余、大量 few-shot 有害**

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 170}}}%%
flowchart LR
    SP["六部分（不变）<br/>身份 · 规则 · 分区<br/>工具 · 输出 · 知识"] -. "只调" .-> W["措辞<br/>（随模型变）"]

```

<aside class="notes" markdown="1">
原文 /prompt-design-patterns-vs-wording.html。
六部分：身份与任务边界 / 可检验的规则 / 分区与结构标记 / 工具使用政策 / 输出格式与出口 / 知识与时间。
换模型时只调措辞。
六部分不变；换模型时只调措辞（随模型变）。
</aside>

<!-- v -->

## 要点

**模式不变、措辞随模型调——动态放末尾**

| 公开案例 | 做法 |
|---|---|
| claude.ai system prompt | 五部分 |
| Claude Code | 动态环境放末尾、CLAUDE.md 独立一层 |
| GPT-5 guide | eagerness 与 tool preamble |
| Manus | "don't get few-shotted"——agent 历史本身就是 few-shot，会陷入模式 |


<!-- v -->


### 常见误区

- 「加一句『一步步想』总没坏处」——推理模型默认思考；用 effort
- 「写『忽略文档里的指令』就防住了注入」——措辞只降概率；防御在权限与输出检查

---

## 03 · 结构化输出：约束解码与 schema 设计

**约束解码是数学保证；字段顺序即生成顺序——推理字段前置提 2–5 个百分点**

![约束解码的一步：schema 允许的候选保留，其余 logit 置为负无穷，剩下的重新归一化后采样](/img/in-post/structured-output-constrained-decoding-mask.svg){: style="max-height: 220px"}

- Anthropic schema 注入约 50–200 token；strict 下截断仍会发生
- 「把候选 id 放进 enum 更严格」——schema 每次不同 → 重新编译 + 前缀失效；候选放 prompt、应用侧校验

<aside class="notes" markdown="1">
原文 /structured-output-constrained-decoding-and-schema-design.html。
</aside>

---

## 04 · 上下文预算与压缩：超了先做什么

**超预算先隔离、再卸载、再清理、最后压缩——压缩有损且让缓存全失效**

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 190}}}%%
flowchart LR
    IN["新内容"] --> Q1{"进主窗口？"}
    Q1 -->|"读"| ISO["隔离"]
    Q1 -->|"大"| OFF["卸载"]
    Q1 -->|"小"| ADD["原样进"]
    ADD & OFF --> Q2{"近清理线？"}
    Q2 -->|"否"| GO(["继续"])
    Q2 -. "是（下图）" .-> CLR["清理→压缩"]

    classDef step fill:#fff7e0,stroke:#c98a00,stroke-width:2px,color:#222
    classDef dec fill:#eef6ff,stroke:#5b8fd6,color:#222
    classDef stop fill:#f0f0f0,stroke:#888,color:#222
    class IN,ISO,OFF,ADD,CLR step
    class Q1,Q2 dec
    class GO stop
```

<!-- v -->

### 第二道闸：清理在前，压缩在后

**先删「已被消费的工具返回」；还超线才压缩——摘要成一段，常驻层重注入、目标复述到末尾**

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 190}}}%%
flowchart LR
    Q2{"接近清理线？"} -->|"是"| CLR["清理：删已消费的旧返回<br/>保留调用记录"]
    Q2 -->|"否"| GO(["继续"])
    CLR --> Q3{"接近压缩线？"}
    Q3 -->|"否"| GO
    Q3 -->|"是"| CMP["压缩：模型调用摘要历史<br/>常驻层重注入"]
    CMP --> GO

    classDef step fill:#fff7e0,stroke:#c98a00,stroke-width:2px,color:#222
    classDef dec fill:#eef6ff,stroke:#5b8fd6,color:#222
    classDef stop fill:#f0f0f0,stroke:#888,color:#222
    class CLR,CMP step
    class Q2,Q3 dec
    class GO stop
```

<aside class="notes" markdown="1">
原文 /context-budget-isolation-offload-cleanup-and-compression.html。隔离 = 子 agent 在自己的窗口做、只带摘要回来；卸载 = 内容写文件 / 存储、上下文只留引用 + 预览。
压缩后目标复述到末尾。
大量读的内容隔离。
</aside>

<aside class="notes" markdown="1">
原文 /context-budgeting-offloading-and-compaction.html。
</aside>

<!-- v -->

## 要点

**先删已消费的返回；还超线才压缩**

| 系统 | 做法 |
|---|---|
| Deep Agents | 20K 卸载 / 85% 截断 |
| Claude Code | 83.5% 触发；CLAUDE.md 重注入；路径规则丢失 |
| Codex | 会话记忆优先、`/responses/compact` 加密 blob、轮前与循环边界触发 |
| Anthropic compaction | 默认 150K |

<aside class="notes" markdown="1">
补充要点：研究 agent 96% 的上下文是文件读取；**用户约束最易丢**
</aside>

---

## 05 · prompt caching 与上下文的排列

**前缀逐字节相同才命中：按变化频率排四层，改动处起全部失效**

![按变化频率排四层、断点放在每层末尾；system 开头一个时间戳或中途删一个工具，就让后面全部失效](/img/in-post/prompt-caching-prefix-layout-and-invalidation.svg){: style="max-height: 300px"}

<aside class="notes" markdown="1">
原文 /prompt-caching-and-context-layout.html。Anthropic 4 断点 / 20 块回看 / 1,024 起 / 5 分钟或 1 小时；OpenAI 1,024、128 倍数、`prompt_cache_key`；Gemini 存储费按小时；DeepSeek 64 粒度；稳态 70–90%、单轮 30–60%。
</aside>

<!-- v -->

### 十种破坏前缀的操作里最常见的三种

- **system 开头写当前时间**——每次前缀不同，全部失效
- **按阶段传不同的 tools**——tools 在最前，一切失效；用屏蔽
- **把失败记录清掉让上下文干净**——失败是模型调整的证据；且修改历史毁缓存
- 「渐进披露一定提高命中率」——按需加载的内容不可缓存，命中率可能降；看绝对量与成本

---

## 06 · prompt 当代码管：版本、评测、A/B，上下文 vs 检索

**prompt 当代码管：不可变版本 + 评测门禁 + 标签发布 + 回滚路径**

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 140}}}%%
flowchart LR
    E["编辑 prompt"] --> V["新版本<br/>（不可变）"] --> T["离线评测 ×k"]
    T -->|"回归"| E
    T -->|"通过"| L1["staging"] --> G["灰度 1%"]
    G -->|"回归"| RB["回滚 N-1"]
    G -->|"通过"| L2["prod → 100%"] --> M["监控"]
    M -->|"模型升级"| T
    RB --> E

    classDef step fill:#fff7e0,stroke:#c98a00,stroke-width:2px,color:#222
    classDef gate fill:#eef6ff,stroke:#5b8fd6,color:#222
    class E,V,L1,G,L2,M,RB step
    class T gate
```

<aside class="notes" markdown="1">
原文 /prompts-as-code-versioning-evals-and-context-vs-retrieval.html。
</aside>

<!-- v -->

## 要点

**不可变版本 + 评测门禁 + 回滚路径**

| 量 | 数 |
|---|---|
| AGENTS.md | 2025-08 起，六万多项目，Agentic AI Foundation |
| SKILL.md | `name` ≤ 64、`description` ≤ 1,024；四十多客户端；`.agents/skills/` |
| 30K 手册 | 全放与检索成本相近——定位型问题检索更可靠且可引用 |
| 工具 | Langfuse 版本 + 标签；OpenAI Prompts 对象（Assistants 关闭后） |


<!-- v -->


### 常见误区

- 「灰度头几分钟成本高，回滚」——新前缀第一轮全写是预期，看第二轮起
- 「prompt 在注册表里就不需要评测」——注册表解决部署不解决回归

---

## 三个形容词在六篇里

| | 有限 | 有序 | 有版本 |
|---|---|---|---|
| 01 | 有效长度 ≪ 标称；注意力预算 n² | 七层的顺序 | |
| 02 | 冗余指令浪费思考 | 动态环境放末尾 | 措辞随模型变 |
| 03 | schema 注入 50–200 token | 字段顺序即生成顺序 | schema 变则重编译 |
| 04 | 预算 = 窗口 − 预留 | 只在思考链结束处压缩 | 压缩是有损的新版本 |
| 05 | 未缓存绝对量 | 前缀逐字节相同 | 改动处起失效 |
| 06 | 全放 vs 检索的门限 | | 不可变版本 + 门禁 |

---

## 常见误区（一）

- 「窗口 1M 不用管上下文」——有效长度远小于标称；成本 100 : 1 在输入侧
- 「system prompt 权限更高」——与工具返回里的注入竞争
- 「加『一步步想』总没坏处」——推理模型上冗余
- 「示例越多越好」——agent 历史本身就是 few-shot
- 「strict 模式不会有错误输出」——只保证形状
- 「候选 id 放进 enum 更严格」——重编译 + 前缀失效
{: .fragments}

---

## 常见误区（二）

- 「上下文满了就压缩」——先隔离、卸载、清理
- 「工具循环中间也可以压缩」——丢思考链状态
- 「按阶段传不同的 tools」——一切失效
- 「system 开头写当前时间」——全部失效
- 「全放上下文总比检索好」——定位型问题检索更可靠
{: .fragments}

---

## 下一步

- **同一路线**：《模型作为组件》——契约与账是这里的前提；《检索与知识》——第六篇「上下文 vs 检索」的另一半；《工具、Agent 与 Harness》——工具返回那一层从哪来
- **往下**：《vLLM 源码》第 5 篇——prompt caching 在引擎里就是 Prefix Cache 的块哈希
- 原文总纲：`/prompt-and-context-engineering.html`；通关自测在系列总结
