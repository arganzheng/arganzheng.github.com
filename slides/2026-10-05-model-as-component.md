---
layout: slides
title: "模型作为组件：契约、失效模式与选型"
subtitle: "系列精华 · 六篇正文每篇一页：给一个没有规格书的组件写规格书"
permalink: /slides/model-as-component.html
series: model-as-component
date: 2026-10-05
author: arganzheng
description: "《模型作为组件》系列的分享用幻灯片：七条失效性质、四家 API 的共同骨架、推理模型的 thinking / effort / 跨轮状态、一次调用的成本与延迟账、榜单之外的选型、客户端的重试 / 超时 / 幂等 / 限流。"
theme: white
transition: slide
---

## 这个系列的一句话主张

> 模型 API **没有供应商替你写的规格书**，应用工程师要自己写——**失效模式、契约、账、选型判据、客户端纪律**，五部分缺一不可。

<aside class="notes" markdown="1">
总纲：/model-as-a-component.html。
</aside>

---

## 几篇怎么连起来

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 170}}}%%
flowchart TB
    C1["01 失效模式<br/>七条可检测的性质"] --> C2["02 API 契约（一）<br/>消息、工具、结构化输出、流式"]
    C2 --> C3["03 API 契约（二）<br/>推理模型：thinking、effort、跨轮状态"]
    C2 & C3 --> C4["04 成本与延迟的账<br/>四种输入价、TTFT 分解"]
    C1 & C4 --> C5["05 选型<br/>榜单失真、自己的评测集、弃用周期"]
    C1 & C2 & C4 --> C6["06 客户端工程<br/>重试、超时、幂等、限流、流式解析"]
```

---

## 01 · 失效模式：把「模型会出错」拆成七条可检测的性质

**结论**：非确定性、幻觉、指令遵循与 prompt 敏感、**上下文标称 ≠ 有效**、知识截止、供应商侧变更、越界——每条对应后面某一层的应对。

```mermaid
flowchart TB
    CTX["组装上下文<br/>system · 历史 · 检索 · 工具定义"] --> PRE["prefill：模型读上下文"]
    PRE --> SAMP["逐 token 采样"]
    SAMP --> OUT["输出：文本 / JSON / tool_call"]
    OUT --> ACT["执行动作（有工具时）"]
    F4["④ 上下文：标称 ≠ 有效<br/>中间遗忘、腐化"] -.-> PRE
    F5["⑤ 知识截止"] -.-> PRE
    F3["③ 指令遵循 / prompt 敏感"] -.-> PRE
    F1["① 非确定性<br/>采样随机 + batch 不变性缺失"] -.-> SAMP
    F2["② 幻觉<br/>高概率续写 ≠ 真"] -.-> SAMP
    F6["⑥ 供应商变更<br/>权重 / 默认值 / tokenizer"] -.-> PRE
    F6 -.-> SAMP
    F7["⑦ 越界<br/>权限范围 = 错误上限"] -.-> ACT

    classDef stage fill:#fff7e0,stroke:#c98a00,stroke-width:2px,color:#222
    classDef fail fill:#fdecea,stroke:#c0392b,color:#222
    class CTX,PRE,SAMP,OUT,ACT stage
    class F1,F2,F3,F4,F5,F6,F7 fail
```

<aside class="notes" markdown="1">
原文 /llm-failure-modes-nondeterminism-hallucination-and-context.html。
</aside>

<!-- v -->

### 要点

| 性质 | 一个数 |
|---|---|
| 非确定性 | batch 不变性缺失：1000 次贪心 18 种结果 → 修好后 1 种；公开 API 做不到 |
| 上下文有效长度 | NoLiMa 32K 处 11/13 模型掉到基线一半以下；GPT-4o 99.3% → 69.7% |
| 幻觉的后果 | Air Canada 812.02 加元——源材料正确、输出编造、公司仍负责 |
| 越界 | PocketOS 9 秒删库；约 700 个 agent 入侵 Hugging Face |

- 「system prompt 是权限更高的通道」——它只是上下文里一段被标记的文本，与工具返回里的注入竞争

---

## 02 · API 契约（一）：四家 API 的共同骨架

**结论**：**块列表进、块列表出**；**工具调用是协议不是功能**（校验、权限、执行、配对、截断在应用侧）；结构化输出**保证语法不保证语义**；服务端状态默认存储。

```mermaid
sequenceDiagram
    participant App as 应用（你的代码）
    participant API as 模型 API
    participant Tool as 工具（你的函数 / 外部服务）
    App->>API: 请求：messages + tools（name、description、JSON schema）
    API-->>App: 响应：assistant 消息，含 1..n 个 tool_call（id、name、arguments）
    Note over App: 校验参数 → 权限检查 → 决定执行哪些
    App->>Tool: 执行（可并行）
    Tool-->>App: 结果 / 错误
    App->>API: 请求：完整历史 + 每个 tool_call 对应一条 tool result（引用 id）
    API-->>App: 响应：最终回答，或又一批 tool_call
    Note over App,API: 循环直到模型不再返回 tool_call，或预算耗尽
```

<aside class="notes" markdown="1">
原文 /llm-api-contract-messages-tools-structured-output-and-streaming.html。
</aside>

<!-- v -->

### 要点

| 项 | OpenAI | Anthropic |
|---|---|---|
| 工具参数 | `arguments` 是字符串 | `input` 是对象 |
| 服务端状态 | Responses 默认 `store: true` 30 天；Conversation 无 TTL | 无状态 |
| 缓存顺序 | tools → system → messages | 同 |
| 变更 | Assistants API 2026-08-26 关闭 | |

- 「用了 strict 模式就不会有错误输出」——只保证 schema 形状；值可以编造；拒答会被伪装成答案
- 「服务端状态省 token 钱」——每轮模型仍读完整上下文，输入照样计费

---

## 03 · API 契约（二）：推理模型加了什么维度

**结论**：思考**按输出价计费、占 TTFT、`max_tokens` 含它**；effort 是**倾向不是预算**，选档靠三条曲线（质量 / 延迟 / 成本 vs 档位）取饱和最低档；思考是**跨轮状态，只搬运不读写**。

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 200}}}%%
flowchart LR
    IN["输入：完整历史（含上一轮的 thinking / reasoning 状态）+ 本轮 user 或 tool result"] --> THINK["思考阶段<br/>生成 thinking token：不可见 / 摘要<br/>按输出价计费 · 占 TTFT"]
    THINK --> OUT["可见输出<br/>文本 或 tool_call"]
    OUT --> STATE["返回给应用的状态<br/>thinking block（Anthropic）/ reasoning item（OpenAI）/ thought signature（Gemini）"]
    STATE --> NEXT["下一轮请求必须携带它<br/>（DeepSeek 例外：不携带）"]
    EFF["effort / thinking_level"] -.->|"控制思考 token 的量级"| THINK
    NEXT -.-> IN

    classDef stage fill:#fff7e0,stroke:#c98a00,stroke-width:2px,color:#222
    classDef ctl fill:#eef6ff,stroke:#5b8fd6,color:#222
    class IN,THINK,OUT,STATE,NEXT stage
    class EFF ctl
```

<aside class="notes" markdown="1">
原文 /reasoning-models-as-components-thinking-effort-and-state.html。
</aside>

<!-- v -->

### 要点

| 量 | 数 |
|---|---|
| 同任务 `low` → `high` | 账单 \$0.0075 → \$0.0445（**6 倍**） |
| 历史校验 | 校验 thinking block 之前的历史——编辑历史受约束 |
| Chat Completions | 自 GPT-5.4 起不支持带 effort 的工具调用 |
| DeepSeek | `reasoning_content` 不能送回 |

- 「把模型名从 4.6 改成 5 就迁移完了」——thinking 默认开启、采样参数 400、tokenizer 多 30%、`max_tokens` 含思考
- 「effort 越高越好」——简单任务高档可能过度思考

---

## 04 · 成本与延迟的账

**结论**：五项成本公式；**缓存读一次回本**（写 1.25× + 读 0.1× = 1.35× < 2×）；多轮不缓存二次增长、缓存后约 15%；**TTFT = 排队 + prefill + 思考，总时长由输出 × TPOT 主导**。

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 170}}}%%
flowchart LR
    Q["排队<br/>供应商侧的负载 · 你的限流队列"] --> P["prefill<br/>读完整上下文，∝ 未缓存输入长度<br/>命中缓存的前缀跳过"]
    P --> T["思考<br/>∝ 思考 token 数 × 每 token 时间<br/>由 effort 决定"]
    T --> F(["首个可见 token（TTFT 在此结束）"])
    F --> D["decode<br/>每 token 一步，速度由内存带宽决定<br/>总时长 ∝ 输出 token 数"]
    D --> E(["最后一个 token（总时长）"])

    classDef seg fill:#fff7e0,stroke:#c98a00,stroke-width:2px,color:#222
    classDef mark fill:#f0f0f0,stroke:#888,color:#222
    class Q,P,T,D seg
    class F,E mark
```

<aside class="notes" markdown="1">
原文 /token-cost-and-latency-ledger-for-llm-applications.html。
</aside>

<!-- v -->

### 要点

| 价 | 相对输入价 |
|---|---|
| 输出 | 4–6× |
| 缓存读 | 0.1×（Fable 5.1 0.025×、DeepSeek 0.02×） |
| 缓存写 | 1.25×（1 小时 TTL 2×） |
| 长上下文 | OpenAI > 272K：**整个请求**输入 2×、输出 1.5× |
| 峰时 / Batch | DeepSeek 峰时 ×2；Batch API 50% |

- 价目跨三个量级：\$10 → \$0.15 → 读 \$0.003 每百万
- 「长上下文加价只对超出部分」——300K 输入全部 2×

---

## 05 · 选型：榜单只缩范围

**结论**：选型用**自己的评测集（30–50 条起）**；闭源 vs 自托管是运营问题；**级联分流不漏难题**（路由会漏）；**弃用周期是选型维度**。

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 170}}}%%
flowchart LR
    A["全部候选（20+）"] -->|"硬约束过滤：数据驻留 · 上下文 · 多模态 · 延迟 · 推理"| C["可行候选（8–12）"]
    C -->|"榜单与同类经验，只做缩小范围"| E["短名单（3–5）"]
    E --> F["在自己的评测集上跑<br/>每模型 × 每 effort 档 × k 次"]
    F --> G{"过质量门限？"}
    G -->|"否"| H["放宽约束 / 加检索或微调（L3）/ 改任务形态（L7）"]
    G -->|"是"| I["按每任务成本排序"]
    I -->|"供应商风险：弃用周期 · 替代路径 · 多云"| K["选定主模型 + 替代模型<br/>写进配置，不写进代码"]

    classDef step fill:#fff7e0,stroke:#c98a00,stroke-width:2px,color:#222
    classDef dec fill:#eef6ff,stroke:#5b8fd6,color:#222
    class A,C,E,F,I,K,H step
    class G dec
```

<aside class="notes" markdown="1">
原文 /model-selection-beyond-leaderboards.html。
</aside>

<!-- v -->

### 要点

| 量 | 数 |
|---|---|
| Arena 失真 | Llama 4 前 27 个私测变体；两家各约 20% Arena 数据 vs 83 个开源模型共 29.7% |
| 大小模型价差 | 5–20 倍；70% 简单请求级联省 63% |
| 自托管 | 满载 \$0.8 / 百万；利用率 30% 时翻三倍 + Infra 团队 |
| 弃用通知 | OpenAI 3–6 个月、Anthropic ≥ 1 年、**DeepSeek 4 天** |

- 「路由比级联省钱所以用路由」——只省 2–3 个百分点却会漏难题

---

## 06 · 客户端工程：调用侧要处理哪些失败

**结论**：错误分四类**只重试「再试可能不同」的**；**四层超时**；幂等问题在工具上；限流按 RPM / TPM 与层级；发前数 token；熔断 + 分层 fallback。

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 180}}}%%
flowchart LR
    S(["业务层发起调用"]) --> CNT["数 token<br/>超过上下文或预算 → 先压缩 / 拒绝，不发"]
    CNT --> SLOT["取并发槽 / 令牌<br/>按模型 · 按租户的限流器"]
    SLOT --> SEND["发送（流式或非流式）<br/>四层超时：连接 · 首事件 · 事件间 · 总"]
    SEND --> OK{"成功？"}
    OK -->|"是"| PARSE["解析：拼接增量 · 工具参数等 done · 读 usage · 读 stop_reason"]
    PARSE --> DONE(["返回业务层 + 记 trace"])
    OK -->|"否"| CLS{"错误分类"}
    CLS -->|"可重试：429 / 5xx / 529 / 超时 / 连接断"| BACK["退避（指数 + 抖动，遵守 Retry-After）<br/>重试预算未耗尽？"]
    BACK -->|"是"| SLOT
    BACK -->|"否"| FB
    CLS -->|"不可重试：400 · 401 · 配额不足 · 内容过滤"| FB{"有 fallback？"}
    FB -->|"换模型 / 换供应商 / 降级"| SLOT
    FB -->|"无"| FAIL(["失败返回业务层 + 记 trace"])

    classDef step fill:#fff7e0,stroke:#c98a00,stroke-width:2px,color:#222
    classDef dec fill:#eef6ff,stroke:#5b8fd6,color:#222
    classDef stop fill:#f0f0f0,stroke:#888,color:#222
    class CNT,SLOT,SEND,PARSE,BACK step
    class OK,CLS,FB dec
    class S,DONE,FAIL stop
```

<aside class="notes" markdown="1">
原文 /llm-client-engineering-retries-timeouts-streaming-and-rate-limits.html。
</aside>

<!-- v -->

### 要点

| 纪律 | 规则 |
|---|---|
| 重试 | full jitter $$t_n = \min(t_{max}, \text{random}(0, t_0 2^n))$$；重试预算 10% |
| `insufficient_quota` | 也是 429，**不可重试** |
| 超时 | 连接 / 首字节 / 字节间 / 总——一个总超时不够 |
| 流式 | 超过一分钟的生成一律流式 |
| 计数 | Sonnet 5 同文本多 30% token；中文每字 1–2 token |
| 取消 | 用户关页面要传播到上游，否则供应商继续生成并计费 |

---

## 五部分规格书

| 部分 | 一句话 | 篇 |
|---|---|---|
| 失效模式 | 七条性质，每条可检测 | 01 |
| 契约 | 块列表进出；工具调用是协议；思考是只搬运的状态 | 02、03 |
| 账 | 五项成本；读一次回本；TTFT 三段 | 04 |
| 选型判据 | 自己的评测集；级联；弃用周期 | 05 |
| 客户端纪律 | 四类错误、四层超时、预算、传播取消 | 06 |

---

## 常见误区（一）

- 「temperature = 0 加 seed 就能复现」——公开 API 的 batch 随负载变
- 「源材料没错机器人就不会说错」——幻觉是高概率续写不是转述
- 「标称 1M 就能放 1M」——32K 处就掉一半
- 「system prompt 权限更高」——只是一段文本
- 「strict 模式不会有错误输出」——只保证形状
- 「服务端状态省 token 钱」——只改变谁传历史
{: .fragments}

---

## 常见误区（二）

- 「effort 越高越好」——账单 6 倍、可能过度思考
- 「缓存写入加价所以不划算」——读一次回本
- 「Arena 第一就是最好」——只用来缩范围
- 「自托管 \$0.8 比 API 便宜」——满载数字
- 「429 了就重试」——`insufficient_quota` 不可重试
- 「用户关了页面请求就结束了」——不传播则继续计费
{: .fragments}

---

## 下一步

- **同一路线**：《Prompt 与上下文工程》——契约之上，模型这一步该看到什么；《生产与运维》——客户端纪律的服务端一半
- **往下**：《vLLM 源码》——API 另一端的引擎；《高效推理与压缩》——为什么输出价是输入的 4–6 倍
- 原文总纲：`/model-as-a-component.html`；通关自测在系列总结
