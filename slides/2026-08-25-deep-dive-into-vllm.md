---
layout: slides
title: "大模型推理系统揭秘：从 vLLM 看 LLM Serving 核心技术"
subtitle: "系列精华 · 十四篇正文逐篇看图，第十五篇总结与自测"
permalink: /slides/deep-dive-into-vllm.html
series: deep-dive-into-vllm
date: 2026-08-25 23:30:00 +0800
author: arganzheng
description: "大模型推理系统揭秘系列分享：从 Serving 难点与指标，到调度、KV、GPU、多卡、PD、模型与请求适配、硬件抽象、未来架构、源码里的请求路径和 SGLang 对照。"
theme: white
transition: slide
---

## 大模型推理系统揭秘

### 从动态请求到推理操作系统

**十四篇正文 · 一条请求路径 · 一切围绕动态请求、KV 状态与 GPU 资源**

```mermaid
%% 图：一个 LLM 请求：Prefill 之后逐步 Decode，直到 EOS 或长度上限
flowchart LR
  IN["请求进入"] --> PF["Prompt Processing / Prefill"]
  PF --> T1["生成第一个 Token"]
  T1 --> DS["Decode Step：再生成一个 Token"]
  DS --> Q{"EOS 或达到长度上限？"}
  Q -- "否" --> DS
  Q -- "是" --> FIN["请求结束"]
```

<aside class="notes" markdown="1">
系列总纲：[/deep-dive-into-vllm.html](/deep-dive-into-vllm.html)。第 15 篇是系列总结与通关自测。主线：Serving 难点 → 指标 → 系统全景 → 调度、内存、执行（含投机解码）→ 多卡和 PD → 输入、输出、模型、硬件适配 → 未来与 SGLang 对照；源码落在讲到它的那一篇里（03 的四个域与调用链、06 的翻译层）。
</aside>

---

## 这个系列回答一个问题

> 一个持续生成的请求，如何在 SLO、KV 显存与 GPU 算力之间被系统持续协调？

- KV Cache 是随生成增长的状态；调度的单位是 token。
- 01–14：定义问题、建立指标、逐层拆系统并落到源码，再用第二个引擎检验分析框架。
- 目标不是最大 batch，而是在 SLO 约束下选择合适的执行规模。

<aside class="notes" markdown="1">
系列总纲：[/deep-dive-into-vllm.html](/deep-dive-into-vllm.html)。第 15 篇：[/deep-dive-into-vllm-series-recap-and-self-test.html](/deep-dive-into-vllm-series-recap-and-self-test.html)。
</aside>

<!-- v -->

### 版本：基于 vLLM v0.27.1，之后的变化

| 变化 | 版本 | 涉及 |
|---|---|---|
| Model Runner V2 成为默认，V1 只作回退 | v0.29 | 03 · 06 · 10 |
| 持有 KV 的等待请求单独排队、优先调度 | v0.31 | 04 · 08 |
| TP all-reduce 默认走 FlashInfer | v0.29 | 07 |
| `max_num_batched_tokens` 默认 8192 → 16384 | v0.28 | 04 · 09 |
| 删除 Medusa 与旧 Ray executor | v0.33 rc | 03 · 06 |

<aside class="notes" markdown="1">
版本提示 @2026-11-12。全系列源码钉在 v0.27.1（2026-08-11）；截至 v0.32.0（2026-10-18），分层架构（API Server → AsyncLLM → EngineCore → Executor → Worker）与进程间通信不变，变化集中在 ModelRunner 层：v0.29.0 起 `vllm/v1/worker/gpu/` 的 Model Runner V2 是默认实现，有自己的 CUDA Graph 管理、采样器、logits processor 接口和投机解码子包；V1 的 `gpu_model_runner.py` 被宣布弃用，v0.32.0 中仍在。另有 prefix cache 的 `NONE_HASH` 默认改为确定值（v0.29，第 05 篇）。完整清单见总纲「版本与硬件基线」：[/deep-dive-into-vllm.html#版本与硬件基线](/deep-dive-into-vllm.html#版本与硬件基线)。
</aside>

---

## 全景：一个问题被不断放大

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 300}}}%%
flowchart TB
    subgraph S1["一台机器"]
        direction LR
        A["01 为什么难<br/>02 怎么衡量"] --> B["03 一个请求的路径"] --> C["04 调度 · 05 KV Cache<br/>06 GPU 执行与投机解码"]
    end
    subgraph S2["扩展与变化"]
        direction LR
        D["07 Multi-GPU<br/>08 PD 分离"] ~~~ E["请求级：09 输入 · 10 输出"] --> F["部署级：11 模型 · 12 硬件"]
    end
    subgraph S3["演进与验收"]
        direction LR
        G["13 下一站"] --> H["14 对照 SGLang"]
    end
    S1 --> S2 --> S3
```

<!-- v -->

### 单机段与变化段的分界：改不改输出

| 机制 | 输出分布 | 归属 |
|---|---|---|
| 投机解码 | 不变（拒绝采样保证无损） | 06 执行效率 |
| 采样参数、结构化输出 | 改变「输出什么」 | 10 输出在变 |
| multi-LoRA、多模态 | 每个请求带进来的东西不同 | 09 输入在变 |

**变化段按粒度排：请求级（09、10，同一 batch 逐请求不同）→ 部署级（11、12，实例启动时定下）。**

<aside class="notes" markdown="1">
01 [/why-llm-serving-is-hard.html](/why-llm-serving-is-hard.html)；02 [/how-to-measure-llm-serving.html](/how-to-measure-llm-serving.html)；03 [/vllm-request-lifecycle-overview.html](/vllm-request-lifecycle-overview.html)；04 [/scheduler-batch-and-fairness.html](/scheduler-batch-and-fairness.html)；05 [/kv-cache-memory-core.html](/kv-cache-memory-core.html)；06 [/gpu-execution-kernels-and-graphs.html](/gpu-execution-kernels-and-graphs.html)；07 [/multi-gpu-scaling-strategies.html](/multi-gpu-scaling-strategies.html)；08 [/prefill-decode-disaggregation.html](/prefill-decode-disaggregation.html)；09 [/request-shapes-multi-lora-and-multimodal.html](/request-shapes-multi-lora-and-multimodal.html)；10 [/output-changes-sampling-and-structured-output.html](/output-changes-sampling-and-structured-output.html)；11 [/model-adaptation-architecture.html](/model-adaptation-architecture.html)；12 [/hardware-abstraction-and-portability.html](/hardware-abstraction-and-portability.html)；13 [/future-of-serving-infra.html](/future-of-serving-infra.html)；14 [/vllm-vs-sglang.html](/vllm-vs-sglang.html)。系列总结：[/deep-dive-into-vllm-series-recap-and-self-test.html](/deep-dive-into-vllm-series-recap-and-self-test.html)。
</aside>

---

## 01 · 为什么 LLM Serving 比传统推理难

**接纳时不知道要跑几轮 forward，KV 又跨步常驻、越长越大——请求有了生命周期**

```mermaid
%% 图：一个 LLM 请求：Prefill 之后逐步 Decode，直到 EOS 或长度上限
flowchart LR
  IN["请求进入"] --> PF["Prompt Processing / Prefill"]
  PF --> T1["生成第一个 Token"]
  T1 --> DS["Decode Step：再生成一个 Token"]
  DS --> Q{"EOS 或达到长度上限？"}
  Q -- "否" --> DS
  Q -- "是" --> FIN["请求结束"]
```

<aside class="notes" markdown="1">
原文：[/why-llm-serving-is-hard.html](/why-llm-serving-is-hard.html)。输入长度可变不是 LLM 独有的问题（传统 Serving 用 padding / bucketing 规整），LLM 只是跨度大了几个数量级。真正改变服务对象的是两点：一个请求要循环多少次「Forward → 采样」，要到生成 EOS / `max_tokens` / stop 才知道，接纳时无法规划；KV Cache 从 Prefill 起一直占着显存并逐步追加，直到请求结束才释放。两点合起来，请求就有了接纳 → Prefill → 若干轮 Decode → 完成（或被抢占、重算）的生命周期，系统每一步都要重新决策。由此引出三个变化：动态执行、带状态推理、Prefill / Decode 混合 workload。贯穿全系列的例子：Llama-3-70B、8×H100，每 token KV 320 KB，2050 token prompt + 300 token 输出约 734 MB。
</aside>

<!-- v -->

### Prefill 与 Decode 争用不同资源

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 170}}}%%
%% 图：Prefill 与 Decode 的数据流对照
flowchart TB
  subgraph PP["Prefill：N 个 Prompt Token 并行，Compute-Bound 倾向"]
    direction LR
    PIN["<b>input_ids</b><br/>$$[t_1, \dots, t_N]$$"] --> PQKV["<b>QKV Proj</b><br/>Q、K、V 各 N 行"]
    PQKV -- "K、V 批量写入" --> PKV["<b>KV Cache</b><br/>+N 个位置"]
    PQKV -- "Q、K、V" --> PATT["<b>Attention</b><br/>因果掩码，K、V 是刚算出的 N 个"]
    PATT --> PLOG["最后 1 个位置的 Logits → 第 1 个 Token"]
  end
  subgraph PD["Decode：1 个新 Token，Memory-Bound 倾向"]
    direction LR
    DIN["<b>input_ids</b><br/>$$[t_L]$$"] --> DQKV["<b>QKV Proj</b><br/>q、k、v 各 1 行"]
    DQKV -- "k、v 追加" --> DKV["<b>KV Cache</b><br/>+1 个位置"]
    DQKV -- "q" --> DATT["<b>Attention</b><br/>q 对全部 L 个位置"]
    DKV -- "读出 L 个 K、V" --> DATT
    DATT --> DLOG["Logits → 下一个 Token"]
  end
  PP ~~~ PD
```

**Prefill 偏计算密集，Decode 偏内存带宽；两者竞争同一组 GPU 资源。**

<aside class="notes" markdown="1">
原文：[/why-llm-serving-is-hard.html](/why-llm-serving-is-hard.html)。Decode 从 batch 1 到 64，单步时间从 5.3 ms 到 7.0 ms；batch 256 时 KV 读取达到 24 GB。
</aside>

<!-- v -->

### 四个核心问题与四个横切约束

| 核心问题 | 主要机制 | 篇 |
|---|---|---|
| 这一轮谁执行、执行多少？ | Continuous Batching、Token Budget、Chunked Prefill、抢占 | 04 |
| 历史状态放哪、如何复用？ | PagedAttention、Prefix Cache、GQA / MLA、KV 量化 | 05 |
| 这一轮如何算得更快？ | FlashAttention、CUDA Graph、融合、量化、投机解码 | 06 |
| 一张卡不够，如何扩展？ | TP、PP、EP、CP、DP；PD 分离 | 07 · 08 |

**横切约束，从请求到芯片：输入在变 09 · 输出在变 10 · 模型在变 11 · 硬件在变 12**

<aside class="notes" markdown="1">
原文：[/why-llm-serving-is-hard.html#六阅读地图后续内容如何展开](/why-llm-serving-is-hard.html#六阅读地图后续内容如何展开)。对应 01 的表 4。代码落点：Scheduler、KVCacheManager / BlockPool、ModelRunner / Attention Backend、Executor / distributed。横切约束：一个请求可以带自己的 LoRA、带图片（09）；每个请求有自己的采样参数、可能要求符合 JSON schema（10）；MLA、MoE、MTP 等新结构（11）；同一套 Serving 逻辑跑在不同芯片上（12）。
</aside>

---

## 02 · 如何衡量：吞吐、延迟与 Goodput

**吞吐算系统做了多少，延迟算用户等多久，Goodput 算 SLO 内有效工作**

![一次请求的 TTFT、TPOT、ITL 与 E2E 时间线](/img/in-post/how-to-measure-llm-serving-request-timeline.svg){: style="max-height: 420px"}

<aside class="notes" markdown="1">
原文：[/how-to-measure-llm-serving.html](/how-to-measure-llm-serving.html)。指标覆盖延迟（TTFT、TPOT、ITL、E2E）、吞吐（Tokens/s、Requests/s、Goodput）、效率和质量；对比 benchmark 前需先统一统计口径。
</aside>

<!-- v -->

### 指标定位到哪一段，参数就动哪一个

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 10, "rankSpacing": 36, "padding": 10}}}%%
%% 图：TTFT 变差时的排查顺序与第一个要试的参数
flowchart LR
    S{"TTFT 变差"}
    S --> T{"queue_time 与 prefill_time<br/>谁占大头？"}
    T -->|queue_time| Q{"waiting_by_reason<br/>与 KV 用量、running 数"}
    Q -->|"capacity，KV 用量 ≈ 1"| Q1["<b>KV 池不够</b><br/>kv_cache_dtype=fp8 或扩副本"]
    Q -->|"capacity，running = max_num_seqs"| Q2["<b>并发上限</b><br/>max_num_seqs ↑（先看 TPOT 余量）"]
    Q -->|"capacity，KV 与 running 都有余量"| Q4["<b>预算被长 prompt 吃光</b><br/>long_prefill_token_threshold"]
    T -->|prefill_time| P{"前缀命中率下降了？"}
    P -->|是| P1["<b>前缀没复用上</b><br/>固定前缀、KV 容量 ↑"]
    P -->|否| P2["<b>prefill 算量本身</b><br/>预算 ↑、TP ↑、PD 分离"]
```

**TTFT 先拆排队与 prefill：排队长是容量或预算，prefill 长是算得多或算得慢。**

<aside class="notes" markdown="1">
原文：[/how-to-measure-llm-serving.html#3-排查顺序先拆-ttft-和-itl再碰参数](/how-to-measure-llm-serving.html#3-排查顺序先拆-ttft-和-itl再碰参数)。两个分叉点：TTFT 先拆成排队（`request_queue_time_seconds`）和 prefill（`request_prefill_time_seconds`），两段原因不重叠——排队长是连第一次调度都没拿到，卡在 KV 池、并发上限或预算；prefill 长是要算的多或算得慢。ITL 先分均值和 P99——均值高是每一步都重（batch 与上下文），只有 P99 高是个别步突然变重（prefill 块或抢占）。第五章有 22 个参数（含 v0.27.1 默认值）与 14 行症状表。原图是一张树，这里按 TTFT、ITL 拆成两页；排队分支里还有一片叶子：`waiting_by_reason` 为 `deferred` 时是 LoRA 预算，`max_loras` ↑。
</aside>

<!-- v -->

### ITL / TPOT：先分均值和 P99

```mermaid
%% 图：ITL / TPOT 变差时的排查顺序与第一个要试的参数
flowchart LR
    S{"ITL / TPOT 变差"}
    S --> I{"均值高，还是只有 P99 高？"}
    I -->|均值| I1["<b>每步太重</b><br/>max_num_seqs ↓<br/>KV fp8、TP ↑"]
    I -->|P99| R{"抢占速率 > 0？"}
    R -->|是| R1["<b>重算式抢占</b><br/>KV 容量 ↑、watermark"]
    R -->|否| R2["<b>大 prefill 块混进 decode 步</b><br/>max_num_batched_tokens ↓<br/>或 PD 分离"]
```

**均值高是每步都重，只有 P99 高是个别步突然变重。**

<aside class="notes" markdown="1">
原文同上。均值高：batch 大、上下文长，每步读的 KV 多，`max_num_seqs` ↓、`kv_cache_dtype=fp8`、TP ↑，或低并发时开投机解码。P99 尖刺：先看 `num_preemptions` 速率；非零是重算式抢占（V1 没有 swap），加 KV 容量或 `--watermark 0.01–0.05`；为零是大 prefill 块和 decode 进了同一步，压 `max_num_batched_tokens` 或 PD 分离。
</aside>

<!-- v -->

### 一个参数，两个方向

| 调大 | 变好 | 变坏 |
|---|---|---|
| `max_num_batched_tokens` 2048 → 8192 | 总吞吐 20.3k → 21.4k tok/s（+5%） | 同一步 64 个 decode 流的 ITL 99 → 376 ms |
| `max_num_seqs` 128 → 256 | Output tok/s +54% | TPOT 11.8 → 15.4 ms，越过 15 ms SLO 后 Goodput 断崖 |

**每个参数都是在两个指标之间换：一步多装一点，摊薄权重读取，这一步也更长。**

<aside class="notes" markdown="1">
原文：[/how-to-measure-llm-serving.html#4-五组需要算账的取舍](/how-to-measure-llm-serving.html#4-五组需要算账的取舍)。算例：Llama-3-70B、8×H100、TP=8，每步 ≈ t_decode(B) + 45 µs × prefill token 数。预算那组的 Output Tokens/s 反而从 764 降到 329——"吞吐变高"要说清是哪个吞吐。H100 默认 `max_num_seqs` 1024 在 70B / TP=8 上碰不到，真正限制 batch 的是 KV 池：每卡约 51 GB、约 540 条 2350 token 的请求，那时 TPOT ≈ 23.5 ms；有 TPOT SLO 的服务应显式把 `max_num_seqs` 设到约 200–250。`kv_cache_dtype=fp8` 是少有的两个指标一起变好（B=256 时 TPOT 15.4 → 11.8 ms，容量翻倍），代价在精度。V1 只有重算式抢占，`--swap-space` / `--preemption-mode` 已不存在。
</aside>

<!-- v -->

### 线上看什么：`/metrics` 的 8 个面板

| # | 面板 |
|---:|---|
| 1 | 请求与 token 流量（按 `finished_reason`） |
| 2 | TTFT 分位数与排队 |
| 3 | ITL 与 TPOT（P99 / P50） |
| 4 | 队列：running、`waiting_by_reason` |
| 5 | KV 用量与抢占速率 |
| 6 | Prefix cache 命中率 |
| 7 | 每步 token 数与请求形状 |
| 8 | 效率：MFU 估算、投机接受率 |

<aside class="notes" markdown="1">
原文：[/how-to-measure-llm-serving.html#六线上监控metrics-里该看什么](/how-to-measure-llm-serving.html#六线上监控metrics-里该看什么)。面板 2、3、4、5 就是排查顺序图的四个分叉点。服务端口径与 `vllm bench` 差四处：TTFT 偏小（不含回送首 token 的网络与 SSE）；ITL 按引擎步记，投机一步出 3 个 token 也只记一个样本；Queueing Time 只算第一次，抢占后的再等待算进 prefill / decode；服务端没有 Goodput，直方图回不到单个请求，只能逐指标估达标率，要真 Goodput 在网关或请求日志里逐请求算。旧名 `vllm:gpu_cache_usage_perc` 已换成 `vllm:kv_cache_usage_perc`。告警：SLO（TTFT P99 / TPOT P95 持续 5 分钟）、容量（`waiting_by_reason{capacity}` 持续非零）、抢占（速率 > 0 且 KV 用量 > 0.95）、前缀命中率低于前一天同时段 70%、错误与 abort 比例。
</aside>

---

## 03 · 鸟瞰 vLLM：4+1 视图落到一次请求

**控制面在 CPU 只动元数据，数据面在 GPU 只动张量；边界上只过两条消息**

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 220}}}%%
flowchart TB
    subgraph CP["控制面 · CPU · 只处理元数据"]
        direction LR
        API["<b>API Server + AsyncLLM</b><br/>tokenize、流式输出"]
        EngineCore["<b>EngineCore</b><br/>step() 主循环"]
        Scheduler["<b>Scheduler</b><br/>每轮给谁几个 token"]
        KV["<b>KVCacheManager + BlockPool</b><br/>请求 → block id"]
        API --> EngineCore --> Scheduler --> KV
    end
    subgraph DP["数据面 · GPU · 只处理张量"]
        direction LR
        Worker["<b>GPU Worker × N</b><br/>ModelRunner：准备输入、前向、采样"]
        Mem["<b>GPU 显存</b><br/>权重分片 · KV Cache · 激活"]
        Worker --> Mem
    end
    CP <==>|"↓ SchedulerOutput：请求 id、token 数、新 block id<br/>↑ ModelRunnerOutput：采样出的 token id"| DP
    classDef cp fill:#eaf2fb,stroke:#5b8fd6,color:#1d3557
    classDef dp fill:#fdf0e6,stroke:#d9822b,color:#5a2e0e
    class API,EngineCore,Scheduler,KV cp
    class Worker,Mem dp
```

<aside class="notes" markdown="1">
原文：[/vllm-request-lifecycle-overview.html#二逻辑视图静态系统拓扑自顶向下](/vllm-request-lifecycle-overview.html#二逻辑视图静态系统拓扑自顶向下)。向下的 `SchedulerOutput`：首次调度的请求以 `NewRequestData` 带 prompt token ids、采样参数和 block ids，已在 Worker 上的请求以 `CachedRequestData` 只带增量；向上的 `ModelRunnerOutput` 主体是 `req_ids` 与 `sampled_token_ids`，只有 `output_rank` 回传。从不过边界的是权重、激活和 KV Cache：KVCacheManager 管的「块」只是 `block_id`、`ref_cnt` 与可选哈希，是显存里 KV 张量的下标；分配、释放、前缀复用、抢占都是整数运算。收益三条：CPU 工作（如填语法 bitmask、异步调度下一轮）与 GPU 前向重叠；边界消息大小与模型大小无关；调度状态只有一份，换 Uni / Multiproc / Ray 只换中间的 Executor。Executor 不是进程，是 EngineCore 构造时创建的对象：`UniProc` 本进程直接调，`Multiproc` 经共享内存 MessageQueue 广播，`Ray` 走 actor。
</aside>

<!-- v -->

### 4+1：进程视图

**Executor 是引擎内对象：DP 定 EngineCoreProc 数、TP×PP 定 WorkerProc 数**

![进程视图：API Server、EngineCoreProc、Executor 与 WorkerProc 的职责和通信](/img/in-post/vllm-request-lifecycle-process-view.svg){: style="max-height: 390px"}

<aside class="notes" markdown="1">
原文：[/vllm-request-lifecycle-overview.html](/vllm-request-lifecycle-overview.html)。Kruchten 4+1：逻辑、开发、进程、部署/物理四个视图，加上串联它们的场景。vLLM v0.27.1：world_size=1 默认选 UniProcExecutor，Worker 在 EngineCore 进程内初始化，没有 WorkerProc；多进程模式下 Executor 进程内调度 WorkerProc。DP size 对应 EngineCoreProc 数；每个 EngineCoreProc 内按 TP×PP×PCP rank 启动 WorkerProc；EP 复用已有 rank、不新增进程。（版本提示：v0.29 起默认是 Model Runner V2，`vllm/v1/worker/gpu/`；本页讲的是 v0.27.1 的 V1 路径。）
</aside>

<!-- v -->

### 执行层三层：Executor → Worker → ModelRunner

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 260}}}%%
graph TB
    EX["<b>Executor</b><br/><i>执行抽象：单设备或多设备模型调用</i>"]
    EX --> W0 & W1 & WN
    subgraph W0["Worker 0 (GPU 0)"]
        direction TB
        M0["ModelRunner"] --- S0["Model Weights（分片）<br/>KV Cache"]
    end
    subgraph W1["Worker 1 (GPU 1)"]
        direction TB
        M1["ModelRunner"] --- S1["Model Weights（分片）<br/>KV Cache"]
    end
    subgraph WN["Worker N (GPU N)"]
        direction TB
        MN["ModelRunner"] --- SN["Model Weights（分片）<br/>KV Cache"]
    end
    NC["NCCL：TP all-reduce / PP 激活收发"]
    W0 & W1 & WN <-.-> NC
```

**Executor 定交给哪些 Worker，Worker 管一张卡，ModelRunner 跑完一轮。**

<aside class="notes" markdown="1">
原文：[/vllm-request-lifecycle-overview.html#24-executorworker-与-modelrunner模型执行的设备抽象](/vllm-request-lifecycle-overview.html#24-executorworker-与-modelrunner模型执行的设备抽象)。Executor：`UniProcExecutor`、`MultiprocExecutor`、`RayDistributedExecutor`、`RayExecutorV2`（继承 `MultiprocExecutor`，只把进程拉起方式换成 Ray）、`ExecutorWithExternalLauncher`——用不用 Ray 是部署方式的差异，不是执行模型的差异。Worker：初始化设备与 TP / PP 通信组、加载权重分片、测显存余量并分配 KV Cache、把 `execute_model()` 转给 ModelRunner。ModelRunner（`GPUModelRunner`）常驻一份 `InputBatch`，每轮准备输入（`_prepare_inputs()`）、前向（能用 CUDA Graph 时走捕获好的图）、采样（`sample_tokens()`）。
</aside>

<!-- v -->

### 4+1：开发视图与包依赖

![vLLM v0.27.1 的包图：运行时调用方向](/img/in-post/vllm-request-lifecycle-package-view.svg){: style="max-height: 440px"}

**实线是一条请求从入口驱动到 kernel 的调用方向，虚线是主干之外的直接依赖。**

<aside class="notes" markdown="1">
原文：[/vllm-request-lifecycle-overview.html#四开发视图模块在源码树的哪里](/vllm-request-lifecycle-overview.html#四开发视图模块在源码树的哪里)。调用方向：entrypoints → v1.engine → v1.core 与 v1.executor；v1.executor → v1.worker → model_executor（含 v1/attention）与 distributed；model_executor → csrc。import 关系并不分层：v0.27.1 里这些目录两两之间有 38 个方向存在 import，量最大的是 model_executor → distributed（270 处，并行线性层与 collective）、model_executor → v1/attention（126）、model_executor → v1/worker（83）、distributed → v1/core（61，KV connector 读调度输出），多数是引用数据结构或注册表，不是运行时反向调用。
</aside>

<!-- v -->

### 开发视图：源码与 wheel 制品

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 480}}}%%
%% 图：vLLM v0.27.1（CUDA）的源码与 wheel 制品：Python 源码原样打包，csrc 与 FlashAttention 经 CMake 编译成扩展模块
flowchart LR
    subgraph SRC["源码（vllm 仓库 + CMake 拉取的依赖）"]
        PY["<b>vllm/ 下的 .py</b><br/>entrypoints · v1 · model_executor · distributed"]
        CS["<b>csrc/libtorch_stable/</b><br/>layernorm、量化、cache、sampler、all-reduce"]
        CM["<b>csrc/libtorch_stable/moe/</b><br/>moe_align_sum · topk_softmax"]
        CA["<b>csrc/cumem_allocator.cpp</b><br/>sleep mode 的显存分配器"]
        FA["<b>vllm-project/flash-attention</b><br/>CMake FetchContent，固定 commit"]
    end
    subgraph WHL["wheel 里的制品"]
        W1["<b>vllm/ 下的 .py</b><br/>原样打包"]
        W2["<b>vllm/_C_stable_libtorch.abi3.so</b><br/>注册 torch 自定义算子"]
        W3["vllm/_moe_C_stable_libtorch.abi3.so"]
        W4["vllm/cumem_allocator.abi3.so"]
        W5["<b>vllm/vllm_flash_attn/</b><br/>_vllm_fa2_C · _vllm_fa3_C"]
    end
    PY -->|"setup.py 收集"| W1
    CS -->|"CMake · nvcc"| W2
    CM -->|"CMake · nvcc"| W3
    CA -->|"CMake"| W4
    FA -->|"CMake · nvcc"| W5
    classDef src fill:#f4f4f4,stroke:#888,color:#333
    classDef art fill:#f2eefb,stroke:#7a68a6,color:#2e2450
    class PY,CS,CM,CA,FA src
    class W1,W2,W3,W4,W5 art
```

**Python 源码原样打包；csrc 与 FlashAttention 经 CMake 编译成扩展模块。**

<aside class="notes" markdown="1">
原文同上，第四章第 2 节。CUDA 平台上是 `_C_stable_libtorch` / `_moe_C_stable_libtorch`，名字不带 `_stable_libtorch` 的 `vllm._C` 只在 ROCm 和 CPU 上构建。`vllm/platforms/cuda.py` 在模块顶层导入 `_C_stable_libtorch`，所以 API Server、EngineCoreProc 也会加载这个 .so；真正在 GPU 上调用 kernel 的只有 WorkerProc。
</aside>

<!-- v -->

### 4+1：部署视图

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 260}}}%%
%% 图：两机 TP=8 × PP=2：节点 0 跑 API Server、EngineCoreProc 和 PP stage 0，节点 1 以 --headless 只拉起 PP stage 1 的 8 个 WorkerProc
flowchart LR
    subgraph N0["节点 0 · --node-rank 0"]
        API["<b>API Server 进程</b>"]
        ECP["<b>EngineCoreProc</b><br/>Scheduler · MultiprocExecutor"]
        S0["<b>WorkerProc rank 0–7</b><br/>PP stage 0 · TP=8"]
    end
    subgraph N1["节点 1 · --node-rank 1 --headless"]
        HL["<b>headless 启动进程</b><br/>只拉起本机 Worker"]
        S1["<b>WorkerProc rank 8–15</b><br/>PP stage 1 · TP=8"]
    end
    API <--> ECP
    ECP -->|"本机共享内存"| S0
    ECP -->|"跨机 ZMQ"| S1
    HL --> S1
    S0 ==>|"PP 激活"| S1
    S1 -.->|"output_rank = 8 回传"| ECP
    classDef cp fill:#eaf2fb,stroke:#5b8fd6,color:#1d3557
    classDef dp fill:#fdf0e6,stroke:#d9822b,color:#5a2e0e
    class API,ECP,HL cp
    class S0,S1 dp
```

**两机 TP=8 × PP=2：节点 0 有完整的控制面，节点 1 只有 WorkerProc。**

<aside class="notes" markdown="1">
原文：[/vllm-request-lifecycle-overview.html#五部署物理视图进程怎样落到-gpu-和节点上](/vllm-request-lifecycle-overview.html#五部署物理视图进程怎样落到-gpu-和节点上)，文中按单卡、单机 TP=4、两机 TP×PP、单机 DP=2 × TP=2、多 API Server 各画了一张部署图。rank 按 TP 最内层编号，rank 0–7 是 PP stage 0、占满节点 0；output_rank = world_size − tp_size = 8 在节点 1。DP=2 时不指定 `--api-server-count` 默认起 2 个 API Server，另有一个 DPCoordinator 进程。EP 在既有 DP×TP rank 上执行，不额外创建进程。
</aside>

<!-- v -->

### 四个域：给源码里的每个对象定位

![核心数据对象全景图：四个域](/img/in-post/source-code-request-walkthrough-four-domains.svg){: style="max-height: 440px"}

**只有显存域跨请求共享；跨进程只传整数 `block_id`，`KVCacheBlock` 对象不出 EngineCore。**

<aside class="notes" markdown="1">
原文：[/vllm-request-lifecycle-overview.html#3-核心数据对象四个域](/vllm-request-lifecycle-overview.html#3-核心数据对象四个域)。请求域 `Request`（token ids、采样参数、状态机状态、块哈希）；调度域 `SchedulerOutput` / `ModelRunnerOutput`；显存域 `KVCacheBlock`（`block_id`、`ref_cnt`、哈希）与 Worker 侧 `BlockTable`；模型域是权重与激活。下面两页是串起它们的两把钥匙：调度器一侧、Worker 一侧各一张图。
</aside>

<!-- v -->

### 两把钥匙 · 调度器一侧

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 160}}}%%
%% 图：调度器一侧：request_id 找到 KVCacheBlock，跨进程只剩 block_id
flowchart LR
    subgraph SB["EngineCoreProc：Scheduler / KVCacheManager"]
        direction TB
        REQ["<b>Request</b><br/>• request_id、status、num_computed_tokens<br/>• output_token_ids、block_hashes"]
        R2B["<b>req_to_blocks</b><br/>request_id → list#91;KVCacheBlock#93;"]
        BLK["<b>KVCacheBlock</b><br/>block_id、ref_cnt、block_hash"]
        POOL["<b>BlockPool</b><br/>• blocks#91;block_id#93;：全部块<br/>• block hash → 块：前缀缓存表"]
    end
    SO["<b>SchedulerOutput</b><br/>• 每步一份，广播给每个 Worker<br/>• 只带 block_id 整数，不带 KVCacheBlock 对象"]
    REQ -- "request_id" --> R2B --> BLK
    BLK -- "block_id 是下标" --> POOL
    REQ -. "block_hashes 查前缀命中" .-> POOL
    R2B -- "只取出 block_id" --> SO
    classDef state fill:#e8f1ff,stroke:#3b6fb6;
    classDef msg fill:#fff4e0,stroke:#d9822b;
    classDef mem fill:#e9f7e9,stroke:#3a8a3a;
    class REQ,R2B state;
    class SO msg;
    class BLK,POOL mem;
```

<aside class="notes" markdown="1">
原文同上，第二章第 3 节第 1 点。`request_id` 经 `req_to_blocks` 找到 `KVCacheBlock` 列表；`block_id` 就是 `BlockPool.blocks` 的下标；`block_hashes` 查前缀命中。`ref_cnt` 和 `block_hash` 只在这一侧有意义（能否复用、何时释放），写进 `SchedulerOutput` 的只有新请求分到的块号与老请求新增的块号——这就是"只传 `block_id` 整数"的那一刀，所以 Prefix Cache 命中与否 Worker 完全不知情。
</aside>

<!-- v -->

### 两把钥匙 · Worker 一侧

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 160}}}%%
%% 图：Worker 一侧：block_id 落到 InputBatch 的行与 BlockTable，派生出 slot_mapping
flowchart LR
    SO2["<b>SchedulerOutput</b><br/>• 新请求：prompt token ids + block_ids<br/>• 已有请求：new_block_ids 增量"]
    subgraph WK["WorkerProc：GPUModelRunner"]
        direction TB
        CRS["<b>CachedRequestState</b><br/>• 每个请求一份<br/>• block_ids、num_computed_tokens、output_token_ids"]
        IB["<b>InputBatch 的一行</b><br/>• req_id → 行号<br/>• token_ids_cpu#91;row, :#93;"]
        BT["<b>BlockTable</b><br/>• 每个 KV cache group 一张<br/>• block_table#91;row, j#93; = block_id"]
        SM["<b>slot_mapping</b><br/>• slot = block_id × block_size + offset<br/>• 由 Triton kernel 在 GPU 上算"]
        KV["<b>KV Cache 张量（GPU HBM）</b><br/>kv_cache#91;block_id#93;#91;offset#93;"]
    end
    SO2 --> CRS
    CRS --> IB
    IB --> BT
    BT --> SM
    SM --> KV
    BT -.-> KV
    classDef state fill:#e8f1ff,stroke:#3b6fb6;
    classDef msg fill:#fff4e0,stroke:#d9822b;
    classDef mem fill:#e9f7e9,stroke:#3a8a3a;
    class CRS,IB,BT state;
    class SO2 msg;
    class SM,KV mem;
```

<aside class="notes" markdown="1">
原文同上，第二章第 3 节第 2 点。原图的边标注：`SchedulerOutput` 按请求更新增量 → `CachedRequestState` 占 `InputBatch` 一行 → 该行的块号进 `BlockTable` → 拷到 GPU 算出 `slot_mapping` → 写入新的 K、V；虚线是 attention kernel 按 `BlockTable` 的行读 KV。`CachedRequestState` 是 `Request` 的只读镜像，靠每步 `CachedRequestData` 的增量维护、从不反向同步；`InputBatch` 每个在跑的请求占一行，请求结束后尾行搬进空行，行号每步都可能变，`BlockTable` 与 token ids 跟着同步搬行——`request_id` 才是跨进程的稳定身份。`slot = block_id × block_size + offset` 由 Triton kernel 在 GPU 上算；每步的四个函数（`_update_states()`、`_prepare_inputs()`、`_build_attention_metadata()`、选图）见第六篇第二章。
</aside>

<!-- v -->

### 控制面开销：一步 decode 里 Python 占多少

![单次 Decode 迭代：Python 控制面与 GPU 数据面](/img/in-post/source-code-request-walkthrough-decode-iteration-timeline.svg){: style="max-height: 420px"}

**schedule ~0.05 ms + prepare ~0.1 ms，对十毫秒量级的 forward 约 1%，且被流水线化。**

<aside class="notes" markdown="1">
原文：[/vllm-request-lifecycle-overview.html#1-两个面控制面与数据面的分离](/vllm-request-lifecycle-overview.html#1-两个面控制面与数据面的分离)。四个阶段：schedule（CPU）→ prepare（CPU）→ model forward（GPU，纯 decode 时一次 CUDA Graph replay）→ sample（GPU → CPU）。GPU 跑第 N 步时 CPU 已在准备第 N+1 步。图中毫秒数是估算，不是源码常量。四个阶段各落在哪些函数、一步之内谁等谁、唯一的同步点在哪，见[第六篇第二章第 3 节](/gpu-execution-kernels-and-graphs.html#3-一步之内的时序cpu-与-gpu-在哪里等谁)；EngineCore 一侧的主循环见[第四篇第五章第 5 节](/scheduler-batch-and-fairness.html#5-调度循环跑在哪里enginecoreproc-的三个线程与-step)。
</aside>

<!-- v -->

### 放大：同一条生命周期，落到组件和关键方法上

![从 HTTP 请求到 GPU kernel 的时间顺序链](/img/in-post/source-code-request-walkthrough-request-to-kernel-chain.svg){: style="max-height: 440px"}

**十一步、三道边界：③ ZMQ 与 ⑤ 共享内存队列是进程边界，⑦ 是 Python → CUDA。**

<aside class="notes" markdown="1">
原文：[/vllm-request-lifecycle-overview.html#放大同一条生命周期落到组件和关键方法上](/vllm-request-lifecycle-overview.html#放大同一条生命周期落到组件和关键方法上)。① `create_chat_completion()` → ② `AsyncLLM.add_request()` + `InputProcessor` tokenize → ③ ZMQ + msgspec 进 EngineCore → ④ `Scheduler.schedule()` → ⑤ `Executor.execute_model()` 经 `rpc_broadcast_mq` 到 Worker（也是控制面与数据面的边界；单卡 `UniProcExecutor` 时退化为函数调用）→ ⑥ `GPUModelRunner` 准备张量 → ⑦ `model(input_ids, positions)` → ⑧ attention kernel → ⑨ 可选 CUDA Graph 重放 → ⑩ 采样结果 D2H → ⑪ `update_from_output()`、detokenize、SSE。三段分别活在 API Server、EngineCoreProc、WorkerProc 里，彼此只靠消息队列衔接；每个进程里常驻的循环与调用栈在机制篇：API Server 的协程见[第十篇第六章第 1 节](/output-changes-sampling-and-structured-output.html#1-解析在哪个进程里跑由哪些参数打开)，EngineCoreProc 的三个线程与 `step()` 见[第四篇第五章第 5 节](/scheduler-batch-and-fairness.html#5-调度循环跑在哪里enginecoreproc-的三个线程与-step)，WorkerProc 的循环与 Python 怎样调到 kernel 见[第六篇第二章](/gpu-execution-kernels-and-graphs.html#二forward-之前scheduleroutput-如何变成-gpu-张量)。（版本提示：执行侧走的是 v0.27.1 的 V1 `GPUModelRunner`；v0.29 起调试断点要打在 `vllm/v1/worker/gpu/model_runner.py`。）
</aside>

<!-- v -->

### 4+1：token 如何穿过 Serving 栈

![Client、API Server、EngineCore 与 GPU 之间的 token 数据流](/img/in-post/vllm-request-lifecycle-token-dataflow.svg){: style="max-height: 410px"}

**跨进程传递的是请求元数据、token IDs 与调度结果，不是完整模型状态。**

<aside class="notes" markdown="1">
原文：[/vllm-request-lifecycle-overview.html](/vllm-request-lifecycle-overview.html)。文本在 API Server tokenize，进程边界上传 token IDs；采样结果返回后在 API Server detokenize。
</aside>

<!-- v -->

### 4+1：一个 token 的回程

![ModelRunner、EngineCore 主线程、输出线程、output_handler 与 SSE 的 token 回程](/img/in-post/vllm-request-lifecycle-token-return.svg){: style="max-height: 410px"}

**主线程立即回到 schedule；序列化走输出线程，detokenize 与 stop 检查在 API Server 进程。**

<aside class="notes" markdown="1">
原文：[/vllm-request-lifecycle-overview.html](/vllm-request-lifecycle-overview.html)。stop 字符命中后 abort_requests_async 回到 EngineCoreProc；RequestOutputCollector 合并 deltas，再由 generate() 发 SSE。
</aside>

---

## 04 · Scheduler：每轮按 token 推进请求

**每轮结合 token budget、请求状态与 KV 空间，决定运行哪些请求、每个请求推进多少 token。**

```mermaid
%% 图：Scheduler 的全貌：Token Budget、KV Cache、Mixed Batch
flowchart TB
    S["<b>Scheduler</b>"] --> TB["<b>Token Budget</b>"]
    TB --> A["<b>Request A</b><br/>256 · Prefill"] & B["<b>Request B</b><br/>1 · Decode"] & C["<b>Request C</b><br/>5 · Spec Decode"]
    A & B & C --> M["<b>Mixed Batch</b>"]
    classDef card fill:#ffffff,stroke:#a9b6c4,color:#243447
    classDef compute fill:#eef5ff,stroke:#5880b4,color:#1d3557
    classDef memory fill:#eff9f4,stroke:#4b9477,color:#1f4a37
    classDef request fill:#e9f4ee,stroke:#70a582,color:#1f4a37
    classDef waiting fill:#fff5e6,stroke:#d79a45,color:#5a3a0e,stroke-dasharray:5 4
    class S card
    class TB compute
    class A,B,C request
    class M memory
```

<aside class="notes" markdown="1">
原文：[/scheduler-batch-and-fairness.html](/scheduler-batch-and-fairness.html)。Continuous Batching、Chunked Prefill 与 Mixed Batch 都是同一资源调度模型的结果；KV 不足时按策略抢占并重算。没有 prefill / decode 两个阶段：每轮让 `num_computed_tokens` 追上 `num_tokens_with_spec`（prompt + 已生成 + 投机 token）。两者是精确等式：普通 decode 时 `num_tokens_with_spec = num_computed_tokens + 1`（上一轮采出的 token 正是这一轮的输入），带 k 个 draft 时是 `+ 1 + k`；默认开启的异步调度下用 `num_output_placeholders` 记已调度、结果未回的 token，`num_tokens_with_spec + num_output_placeholders = num_computed_tokens + 1`。调度循环跑在 EngineCoreProc 的主线程里，另有输入、输出两个线程收发 ZMQ 消息，见[第五章第 5 节](/scheduler-batch-and-fairness.html#5-调度循环跑在哪里enginecoreproc-的三个线程与-step)。
</aside>

---

## 05 · KV Cache：管理增长中的状态

**`block_size`=16；2,050 token 占 129 块，Prefix Cache 只复用满块**

![KV Cache 从 Prefill 写入、Decode 追加到完成或抢占归还的生命周期](/img/in-post/kv-cache-memory-core-kv-cache-lifecycle.svg){: style="max-height: 400px"}

<aside class="notes" markdown="1">
原文：[/kv-cache-memory-core.html](/kv-cache-memory-core.html)。固定大小的块按需分配；请求持 block table。示例中 2,000 token 前缀对应 125 个整块，可由后续请求复用。Prefix Cache 两条容易忽略的规则：命中最多到 `num_tokens − 1`（最后一个 token 要重算出 logits，按块对齐后是最后一整块，2048 token 完全一致也只命中 127 块）；命中的块若还在空闲队列（`ref_cnt == 0`），也算进本次分配的容量，容量不够时一块都不 touch。驱逐：LRU，同一请求链尾先走，没写满的块插队首最先被分走。命中率塌掉的五种负载：前缀开头不稳定、工作集比块池大（LRU 轮转下接近零命中）、高负载、抢占与前缀无感的多实例路由、整块粒度。
</aside>

<!-- v -->

### 混合 KV：滑窗组只留窗口里的块

![同一个请求在全注意力组和滑窗组里的块表随时间的变化](/img/in-post/kv-cache-memory-core-hybrid-block-tables.svg){: style="max-height: 400px"}

**Gemma-3-27B 一个 32K 请求：不区分窗口 15.5 GiB → 窗口感知约 3 GiB（5.2×）**

<aside class="notes" markdown="1">
原文：[/kv-cache-memory-core.html#五混合-kv一个模型里有几种注意力](/kv-cache-memory-core.html#五混合-kv一个模型里有几种注意力)。图里用 64 token 的窗口示意（Gemma-3 是 1024）。Gemma-3-27B 62 层：10 层全注意力 + 52 层滑窗 → 7 个 KV cache group（1 + 6，每个滑窗组 9 或 8 层，最多 15.38% 填充）、10 个 `KVCacheTensor`；页大小统一后所有组共用一个 `BlockPool`。滑窗组把窗口外的块换成 `null_block` 归还，哈希保留。15 GiB 的池：不区分窗口一个 32K 请求都放不下（0.97），窗口感知放 5 个。代价：分组填充、cascade attention 用不上、KV connector 要支持 HMA，否则自动关掉混合管理器。
</aside>

<!-- v -->

### KV 量化：FP8 让同一块池装两倍

| 场景（每卡约 50 GiB KV 池） | BF16 | FP8（E4M3） |
|---|---:|---:|
| 每卡块池 | 81,920 块 | 163,840 块 |
| 2,350 token 的请求 | 557 | 1,114 → `max_num_seqs` 1,024 封顶 |
| 8K 上下文 | 160 | 320 |
| 32K 上下文 | 40 | 80 |

**每 token 每卡 40 KiB → 20 KiB：并发翻倍，decode 每步读的 KV 也减半。**

<aside class="notes" markdown="1">
原文：[/kv-cache-memory-core.html#3-kv-量化每个元素占几个字节](/kv-cache-memory-core.html#3-kv-量化每个元素占几个字节)。Llama-3-70B / 8×H100、TP8 每卡 1 个 KV 头。scale：`fp8` 按张量，每层一对 `k_scale` / `v_scale` 标量从 checkpoint 读，没有就是 1.0 并告警（风险最大）；`*_per_token_head` 写 cache 时由 kernel 动态算，scale 从 KV 池里划（多约 3%）。后端：FlashAttention 只接受 `fp8` / `fp8_e4m3`（FA3 + Hopper 或 FA4 + Blackwell），FlashInfer 还有 `fp8_e5m2`、`nvfp4`，per-token-head 只有 Triton。PD 两端 `kv_cache_dtype` 必须一致。
</aside>

---

## 06 · GPU 执行：减少等待、搬运与轮次

**Decode 权重读取下界约 5.3 ms、实测约 10 ms；decode 占总时 97%**

![标准 Attention 与 FlashAttention 的 HBM 读写和分块计算](/img/in-post/gpu-execution-kernels-and-graphs-standard-vs-flash.svg){: style="max-height: 400px"}

<aside class="notes" markdown="1">
原文：[/gpu-execution-kernels-and-graphs.html](/gpu-execution-kernels-and-graphs.html)。Prefill 2,050 token 约 92 ms；300 步 decode 约 3,000 ms，占总时间 97%。四种浪费是 CPU launch、HBM 等待、数值过胖与解码轮次过多。
</aside>

<!-- v -->

### GPU 上的四种浪费

| 浪费形态 | 症状 | 对策 |
|---|---|---|
| GPU 在等 CPU 发指令 | kernel 之间有气泡 | CUDA Graph |
| GPU 在等 HBM 送数据 | 算力闲置、访存打满 | FlashAttention、算子融合 |
| 搬的每个数太胖 | 带宽被低信息密度的数据占满 | FP8 / INT8 / INT4 量化 |
| 轮次本身太多 | 每轮只产出 1 个 token | 投机解码 |

**Prefill 2050 token ≈ 92 ms 占 3%，300 步 decode ≈ 3000 ms 占 97%——优化都冲着 Decode。**

<aside class="notes" markdown="1">
原文：[/gpu-execution-kernels-and-graphs.html#一总览四种浪费与一笔账](/gpu-execution-kernels-and-graphs.html#一总览四种浪费与一笔账)。TP=8 下每卡 17.6 GB 权重、3.35 TB/s，一次 decode 的权重读取下界约 5.3 ms；加上 KV 读取、80 层 × 2 次 All-Reduce 和 kernel 开销，实测一步约 10 ms。量化与减少搬运次数正交、可叠加。
</aside>

<!-- v -->

### forward 之前：SchedulerOutput 如何变成 GPU 张量

![SchedulerOutput → GPU 输入的映射](/img/in-post/source-code-request-walkthrough-scheduler-output-to-gpu.svg){: style="max-height: 420px"}

**`slot_mapping` 说 KV 写去哪，`block_table` 说能读哪些块；整步唯一的同步点在采样结果 D2H。**

<aside class="notes" markdown="1">
原文：[/gpu-execution-kernels-and-graphs.html#二forward-之前scheduleroutput-如何变成-gpu-张量](/gpu-execution-kernels-and-graphs.html#二forward-之前scheduleroutput-如何变成-gpu-张量)。四步：① `_update_states()` 增量同步跨步常驻的 `InputBatch`；② `_prepare_inputs()` 拼 `input_ids` / `positions`、pinned buffer 异步 H2D，Triton kernel 由 block table 算 `slot_mapping`（Prefix Cache 命中的那段根本不在本轮 token 里）；③ `_build_attention_metadata()`；④ 选 FULL / PIECEWISE CUDA Graph 或 eager 并 pad。`execute_model()` 全程异步入队，同步点只有 `sample_tokens()` 里 `_bookkeeping_sync()` 的 `transfer_event.synchronize()`；async scheduling 把它也移到独立 copy stream 上。三个字段对应三处设计约束：分块存储、Prefix Cache 共享、CUDA Graph 形状固定。同一章第 3–5 节还有：一步四个阶段的源码位置与 CPU / GPU 时序、WorkerProc 的常驻循环（`worker_busy_loop`，`execute_model` 只是一个方法名）、Python 怎样经编译扩展调到 C++ / CUDA / Triton kernel。
</aside>

<!-- v -->

### 投机解码：少跑几轮 forward

**无损：只改每步决定的位置数；batch 1、接受长度 2.5 约 2.1×，batch 128 每步慢 1.9 倍**

![Proposer、Scheduler 与 Target 在投机解码中的角色和约束](/img/in-post/decoding-extensions-spec-decode-roles.svg){: style="max-height: 400px"}

<aside class="notes" markdown="1">
原文：[/gpu-execution-kernels-and-graphs.html#六能不能少跑几轮模型-投机解码](/gpu-execution-kernels-and-graphs.html#六能不能少跑几轮模型-投机解码)。投机解码用拒绝采样保证输出分布不变，所以归在执行优化；它改变每步决定的位置数（1 → 1 + K），牵动调度预算、KV lookahead 预留、回滚与 CUDA Graph。batch × (1 + K) 超过约 300 token / 步 / 卡后验证不再免费。（版本提示：v0.29 起 Model Runner V2 有自己的一套 speculator 与拒绝采样，`vllm/v1/worker/gpu/spec_decode/`，机制相同；Medusa 在 v0.33 预发布版中已删除。）
</aside>

<!-- v -->

### 投机解码在调度器里：进预算、预留 lookahead、按接受数回滚

![连续两步的 KV 槽位（K=3）](/img/in-post/decoding-extensions-spec-kv-slots.svg){: style="max-height: 400px"}

**「KV 回滚」就是 `num_computed_tokens -= num_rejected`：被拒槽位不清理、不释放，下一步直接覆盖。**

<aside class="notes" markdown="1">
原文：[/gpu-execution-kernels-and-graphs.html#4-调度器与-kvdraft-进预算预留-lookahead-槽位按接受数回滚](/gpu-execution-kernels-and-graphs.html#4-调度器与-kvdraft-进预算预留-lookahead-槽位按接受数回滚)。① draft 进 token 预算：带 K 个 draft 的请求本轮推进 1+K 个 token，预算紧时 draft 被截断；chunked prefill 中的请求不调度 draft。② `allocate_slots(..., num_lookahead_tokens)` 为 draft 模型下一步写 KV 预留槽位：EAGLE / MTP / draft model 为 K，n-gram 为 0；例子里 K=3、每卡每 token 40 KB，只预留 120 KB，代价是更早碰到 watermark。③ step 后按接受数回滚，正确性靠 `num_computed_tokens` 这个游标，prefix cache 也以它为界，脏槽位不会被命中。EAGLE / MTP 开启时 prefix cache 命中的最后一块会被主动丢掉（125 块变 124 块），留一个位置真正跑 forward 产生 hidden state。model runner 一次 forward 算 K+1 个位置，`RejectionSampler` 得到 `[batch, K+1]`，被拒位置填 -1。
</aside>

<!-- v -->

### 收益有临界点：每步约 300 token / 卡

| batch | K | 每步 token | 验证免费？ | 每步（估） |
|---:|---:|---:|---|---|
| 1 | 3 | 4 | 是 | ≈ 10 ms |
| 32 | 3 | 128 | 是 | ≈ 10–11 ms |
| 64 | 3 | 256 | 接近临界 | ≈ 12 ms |
| 128 | 3 | 512 | 否，≈ 1.7× 临界 | ≈ 17 ms |
| 128 | 0（不开） | 128 | —— | ≈ 10 ms |

**batch=128 时每步慢 1.9 倍、只换来 2.5 倍步数减少——系统吞吐反降。**

<aside class="notes" markdown="1">
原文：[/gpu-execution-kernels-and-graphs.html#7-收益与代价的量化](/gpu-execution-kernels-and-graphs.html#7-收益与代价的量化)。临界点 $$T_\text{ridge} \approx 295$$ token / 步 / 卡（989 TFLOPS 对 3.35 TB/s）；40% MFU 时还要更低。batch=1、300 个输出 token：EAGLE K=3 接受长度 2.5 → 120 步 × 12 ms ≈ 1450 ms，≈ 2.1×；接受长度 1.6 → 1.3×；n-gram 抄 RAG 原文、接受长度 3.5 → 3.5×。不该开：每步 token 已近临界、输出很短、请求需要 `min_p` / `logit_bias`（开投机时在前端被拒）、高温高熵文本。动态 K（`num_speculative_tokens_per_batch_size`）在大 batch 时降到 0，但 DP > 1 时被关掉；draft 只在最后一个 PP rank；PD 下异步拉 KV 的那一步不预留 lookahead。
</aside>

---

## 07 · Multi-GPU：按瓶颈切分模型与请求

**TP/PP/EP/CP 解决装不下，DP 解决要更多吞吐且在最外层**

| 遇到的问题 | 策略 | 切的是什么 |
|---|---|---|
| 单卡放不下一个 Linear 层 | TP | 层内的矩阵，按行 / 列切 |
| 单层放得下，整个模型太大 | PP | 按层切，分段流水 |
| MoE 的 Expert 太多 | EP | 按 Expert 切 |
| 上下文太长，KV 装不下 | CP | 按 token 序列切 |
| 装得下，但要更多吞吐 | DP | 不切，整个模型复制一份 |

<aside class="notes" markdown="1">
原文：[/multi-gpu-scaling-strategies.html](/multi-gpu-scaling-strategies.html)。TP / PP / EP / CP 是被迫拆分，DP 的前提恰恰是装得下；生产部署先用前四种把模型塞进一组卡，再用 DP 把这组卡整体复制 N 份。
</aside>

<!-- v -->

### rank 布局：TP=2 × PP=2 × DP=2，两台 4 卡机器

![TP=2 × PP=2 × DP=2 在两台 4 卡机器上的 rank 布局](/img/in-post/multi-gpu-scaling-strategies-tp-pp-dp-rank-layout.svg){: style="max-height: 400px"}

**TP 组是相邻的卡、走 NVLink；PP 组隔一个 TP 组；DP 组隔一整个副本、跨机。**

<aside class="notes" markdown="1">
原文：[/multi-gpu-scaling-strategies.html](/multi-gpu-scaling-strategies.html)。vLLM 的 `initialize_model_parallel()` 按 DP × PP × PCP × TP 排布 rank，TP 在最内层。TP 组 [0,1] [2,3] [4,5] [6,7]：激活 All-Reduce，每层 2 次、在关键路径上；PP 组 [0,2] [1,3] [4,6] [5,7]：`hidden_states` 的 send / recv，每个 microbatch 1 次；DP 组 [0,4] [1,5] [2,6] [3,7]：不传激活，只在 MoE + EP 时 All-to-All 跨副本。每个 DP 副本一个 EngineCore，各自拉起 TP × PP = 4 个 WorkerProc。
</aside>

<!-- v -->

### TP 的代价：All-Reduce 在每层的关键路径上

![All-Reduce 位于每层的关键路径](/img/in-post/multi-gpu-scaling-strategies-all-reduce-critical-path.svg){: style="max-height: 300px"}

1. 每层 All-Reduce 两次：Attention 后、MLP 后各一次
2. 每层通信量 $$2V$$，$$V = B \times S \times H \times \text{sizeof(dtype)}$$
3. 只适合 NVLink / NVSwitch 互联的同机多卡

<aside class="notes" markdown="1">
原文：[/multi-gpu-scaling-strategies.html#5-tp-的代价通信](/multi-gpu-scaling-strategies.html#5-tp-的代价通信)。列切的输出分片正好是行切的输入分片，中间不用通信，只有行切的部分和要在末尾求和；Attention 和 MLP 各一对「列切 → 行切」，所以每层两次。算例：batch 32、Decode $$S=1$$、$$H=8192$$、BF16，一次 512 KiB；80 层 × 2 次 = 160 次，一步约 80 MiB 逻辑通信量；Ring All-Reduce 单卡实际传输约 $$2\frac{N-1}{N}V$$，TP=8 时约 $$1.75V$$。所有卡要等 All-Reduce 完成才能算下一层，对带宽和延迟都敏感，跨机时改用 PP、DP、EP 减少高频同步。
</aside>

---

## 08 · PD 分离：状态交接需要明确契约

**匹配≠就绪、算完≠可回收；分离换来配置自由与交接成本**

![Prefill 与 Decode 分离下的 KV 传输、执行与资源干扰时间线](/img/in-post/prefill-decode-disaggregation-overlap-timelines.svg){: style="max-height: 400px"}

<aside class="notes" markdown="1">
原文：[/prefill-decode-disaggregation.html](/prefill-decode-disaggregation.html)。分离后 P、D 两侧仍各自管理本地计算与内存；Connector 管状态交接，外围编排连接请求与池级资源。
</aside>

---

## 09 · 输入在变：multi-LoRA 与多模态

**8 个 rank-16 槽位静态占 1.44 GB/卡，与实际加载数无关**

![一个 kernel 按排序后的 token 索引 gather 不同 LoRA adapter 的权重行](/img/in-post/request-shapes-multi-lora-and-multimodal-lora-gather.svg){: style="max-height: 390px"}

<aside class="notes" markdown="1">
原文：[/request-shapes-multi-lora-and-multimodal.html](/request-shapes-multi-lora-and-multimodal.html)。变化段的第一篇，属于请求级变化：同一个 batch 里每个请求都可能带自己的 LoRA、带几张图片，引擎必须逐请求处理；模型类经 `SupportsLoRA` / `SupportsMultiModal` 声明能力，引擎据此启用对应路径（完整机制在 11）。LoRA FLOPs 只多 0.2%，但每步多 1,120 次 kernel launch；图片贵在持续驻留的 KV，而不是短命的 encoder 输出。
</aside>

<!-- v -->

### 多模态：图片 token 的 KV 成本

![占位区间、embedding 掩码、encoder 输出下标与 chunk 窗口对齐](/img/in-post/request-shapes-multi-lora-and-multimodal-placeholder-align.svg){: style="max-height: 400px"}

**一张图 encoder 输出 9–22 MB，它占的 KV 184–438 MB，约 20 倍**

<aside class="notes" markdown="1">
原文：[/request-shapes-multi-lora-and-multimodal.html](/request-shapes-multi-lora-and-multimodal.html)。图片示例 576 或 1,369 个 token；encoder 输出短命，KV 在生成期间持续驻留。
</aside>

<!-- v -->

### 从 messages 到 token id

| 步 | 函数 | 做什么 |
|---:|---|---|
| 1 | `OnlineRenderer.render_chat()` | 校验 `tool_choice` 与启动参数 |
| 2 | `parse_chat_messages()` | 图片进 `mm_data`，留下占位 |
| 3 | `safe_apply_chat_template()` | 选模板，Jinja 渲染出文本 |
| 4 | `tokenize_prompts_async()` | 文本 → token id |
| 5 | `_process_multimodal()` | 一个图片占位 → N 个 |
| 6 | `Parser.adjust_request()` | 按 `tool_choice` 写约束 |
| 7 | `InputProcessor.process_inputs()` | 并入停止 token |

**引擎只看到 `prompt_token_ids`；模板就是训练格式，配错不报错。**

<aside class="notes" markdown="1">
原文：[/request-shapes-multi-lora-and-multimodal.html#2-渲染流水线从-http-请求到-enginecorerequest](/request-shapes-multi-lora-and-multimodal.html#2-渲染流水线从-http-请求到-enginecorerequest)。全在 API Server 进程、用 CPU，`--api-server-count` 加的就是这一段的并行度。原文这里是一张九个方框的流程图，幻灯片上改成表。模板四级来源：`--chat-template` → 不带 `tools` 时的 processor 模板 → `tokenizer_config.json` → vLLM 兜底文件；都没有就报 `ChatTemplateResolutionError`。`add_special_tokens` 默认关（模板已写 BOS）；多模态处理器在分词之后把一个占位换成 N 个；`InputProcessor` 把 `generation_config.json` 的 `eos_token_id` 并进停止集合。
</aside>

<!-- v -->

### 实例：工具段占 232 个 token 里的 132 个

```text
<|im_start|>system
You are Qwen, created by Alibaba Cloud. ...
# Tools
<tools>{"type": "function", "function": {"name": "get_weather", ...}}</tools>
...<|im_end|>
<|im_start|>user
深圳现在多少度？<|im_end|>
<|im_start|>assistant
<tool_call>{"name": "get_weather", "arguments": {"city": "深圳"}}</tool_call><|im_end|>
...
<|im_start|>assistant
```

**Qwen2.5-7B-Instruct 两轮对话加一个工具：232 token，不带 `tools` 只有 100。**

<aside class="notes" markdown="1">
原文：[/request-shapes-multi-lora-and-multimodal.html#4-实例两轮对话加一个工具渲染出来是什么](/request-shapes-multi-lora-and-multimodal.html#4-实例两轮对话加一个工具渲染出来是什么)。节选，完整渲染见原文。工具定义与格式说明在最前面，`tools` 相同的请求共享这段前缀；改一个工具描述或调换顺序，后面整段都要重算。`<|im_start|>` / `<|im_end|>` 是特殊 token；`<tool_call>` 是单个普通 token，所以 `skip_special_tokens` 不会删掉它。排查用 `POST /tokenize`（加 `"return_token_strs": true`）：它走同一个 `preprocess_chat()`，返回的就是引擎会看到的那串。
</aside>

<!-- v -->

### 模板与模型不配：五种坏法

1. 模板不渲染工具：工具从不被调用，HTTP 200
2. 轮次标记属于别的家族：被切成普通 token，停不下来
3. `add_generation_prompt=false`：模型替用户接着说
4. 思考开关传给不认识它的模板：被静默丢掉
5. 调用格式与 `--tool-call-parser` 不配：调了工具，客户端收不到

<aside class="notes" markdown="1">
原文：[/request-shapes-multi-lora-and-multimodal.html#5-模板与模型不配时会怎样](/request-shapes-multi-lora-and-multimodal.html#5-模板与模型不配时会怎样)。第 2 条：Llama 3 的 `<|eot_id|>` 在 Qwen2.5 词表里被切成 7 个普通 token；模板结束一轮的 token 不在停止集合里也一样停不下来。第 4 条：Qwen3 的 `enable_thinking` 只能经 `chat_template_kwargs` 进模板，`resolve_chat_template_kwargs()` 过滤掉模板不引用的变量。第 5 条：Qwen2.5 的模板教 `<tool_call>` 包 JSON，对应 `hermes`；配成 `llama3_json` 构造时找不到 `<|python_tag|>`，每个请求都失败。没打工具定义时用 `examples/tool_chat_template_*.jinja`。
</aside>

---

## 10 · 输出在变：采样参数与结构化输出

**同一 batch 每行按自己的采样参数；grammar 变成每步一张 16 KB / 行的 bitmask**

![结构化输出生效的四个位置](/img/in-post/decoding-extensions-structured-output-positions.svg){: style="max-height: 400px"}

<aside class="notes" markdown="1">
原文：[/output-changes-sampling-and-structured-output.html](/output-changes-sampling-and-structured-output.html)。采样参数改分布本身、结构化输出改分布的支撑集；参数按 batch 维张量向量化，penalties 每步搬历史输出；grammar 在调度器侧异步编译，与投机解码同开时逐个 draft 位置对账；TP 下每个 rank 都拿到完整 logits、各自采样，GrammarOutput 广播到所有 worker。（版本提示：v0.29 起默认的 Model Runner V2 有独立采样器 `vllm/v1/worker/gpu/sample/` 和新的 logits processor 接口；结构化输出的调度器侧不变。）
</aside>

<!-- v -->

### 两类输出变化落在哪几层

| 层 | 采样语义 | 结构化输出 |
|---|---|---|
| API | 校验参数；开投机时拒绝 `min_p` / `logit_bias` | 校验 schema、选后端 |
| Scheduler | —— | 异步编译 grammar、每步填 bitmask、推进 FSM |
| KV Cache | —— | —— |
| ModelRunner / Sampler | 每请求参数按行向量化 | GPU 上按行打掩码 |
| 多卡 | TP 每 rank 各自采样；PP 末 stage 采样 | `GrammarOutput` 广播给所有 worker |
| PD 分离 | 参数随请求交接 | P、D 各编译一次，都落在 TTFT 上 |

<aside class="notes" markdown="1">
原文：[/output-changes-sampling-and-structured-output.html#3-变化落在哪几层](/output-changes-sampling-and-structured-output.html#3-变化落在哪几层)。KV Cache 一行是空的，这本身就是结论：输出语义的变化不影响状态怎么存，不同参数、不同 grammar 产生的都是普通 token，KV 与 prefix cache 不需要多一个键（对比 09：LoRA 和多模态都要往块哈希里加 `extra_keys`）。代价：采样语义几乎为零，值得算的是 penalties（每步两张 `[B, V+1]` int64 直方图 + 历史输出上传）和 `seed`；结构化输出几乎全在 CPU 与 TTFT——编译、每步填 bitmask、FSM 推进，GPU 只有一个掩码 kernel。batch=64 全开 JSON：每步 H2D 64 × 16 KB = 1 MB。
</aside>

<!-- v -->

### 多卡与 PD 下：谁采样、grammar 在哪编译

- **TP**：logits all-gather 到每个 rank，各 rank 用同一种子各自采样，只有 `output_rank` 回复
- **PP**：只有末 stage 采样；异步调度下把采出的 token 广播回其他 stage
- **DP**：每个 DP rank 自己的调度器编译 grammar，同一 schema 打到 4 个 rank 编译 4 次
- **PD**：P、D 两侧各编译一次 grammar；D 上编译排在 KV 拉取之前，串行

<aside class="notes" markdown="1">
原文：[/output-changes-sampling-and-structured-output.html#四多卡与-pd-分离下的适配](/output-changes-sampling-and-structured-output.html#四多卡与-pd-分离下的适配)。TP：`lm_head` 按词表切，`Platform.use_all_gather()` 默认 True，CUDA 平台每个 rank 都得到完整 `[num_tokens, V]`；`sample_tokens(grammar_output)` 经共享内存一次广播；各 rank 自己采是为了让下一步输入直接留在本 rank 的 `InputBatch`，省一次广播，前提是 `set_random_seed(model_config.seed)` 对齐随机状态。例子 batch=64、TP=8：完整 bf16 logits 约 16 MB、每卡贡献 2 MB，all-gather 在 NVLink 上几十微秒。`external_launcher` 下每个 TP rank 一个调度器，异步编译会让各 rank 状态转换时刻不同而死锁，所以改同步编译。PD（NIXL toy proxy）：P 收到 `max_tokens=1` 的请求，同样等 grammar 就绪才调度、用掩码采出一个会被丢弃的 token；P 上的编译落在 $$Q_P$$，D 上的串在 $$T_{\text{handoff}}$$ 前。缓解：xgrammar 缓存让重复 schema 几乎免费；proxy 可在发往 P 的请求里去掉结构化输出参数。
</aside>

<!-- v -->

### 同一串 token → content / reasoning / tool_calls

```mermaid
%% 图：一步采出的 token 在 API Server 进程里变成 OpenAI 响应字段
flowchart LR
  core["<b>EngineCore</b><br/>新 token id"]
  subgraph api["API Server 进程"]
    direction LR
    det["<b>OutputProcessor</b><br/>detokenize"]
    rp{"推理已结束？"}
    rs["<b>ReasoningParser</b><br/>切出 reasoning"]
    tp["<b>ToolParser</b><br/>抠出 tool_calls"]
    dm["<b>DeltaMessage</b><br/>三个字段"]
    det --> rp
    rp -- "否" --> rs
    rp -- "是" --> tp
    rs -- "标记之后" --> tp
    rs --> dm
    tp --> dm
  end
  core -- "ZMQ" --> det
```

**解析在 API Server；约束在前、解析在后。**

<aside class="notes" markdown="1">
原文：[/output-changes-sampling-and-structured-output.html#2-一个-delta-怎么流过解析器](/output-changes-sampling-and-structured-output.html#2-一个-delta-怎么流过解析器)，图按原图简化（原图还列出各方框里的方法名，以及最后写成 SSE chunk 那一步）。`DelegatingParser` 把流分成推理阶段与工具阶段：reasoning parser 按 token id 找结束标记（`deepseek_r1` 找 `</think>`；prompt 末尾已有 `</think>`，如 Qwen3 关思考，则从第一个 token 起推理已结束），切换那一步工具解析器的累计文本从结束标记之后重新开始，所以思考里写的 `<tool_call>` 不算调用。工具解析要 `--enable-auto-tool-choice` 与 `--tool-call-parser` 一起开；`hermes` 遇到坏 JSON 整段退回 `content`、HTTP 200；响应字段是 `reasoning`，`reasoning_content` 是旧名。解析是 API Server 里的纯 Python，每个 delta 重扫全文。
</aside>

<!-- v -->

### `tool_choice` 落在采样与解析两层

| `tool_choice` | 采样时（EngineCore） | 解析时（API Server） |
|---|---|---|
| `none` | 不约束 | 全部是 `content` |
| `auto` | 不约束 | 事后抠，坏了退回 `content` |
| `auto` + `strict: true` | structural tag，`<tool_call>` 触发 | 调用部分已保证合法 |
| `required` | structural tag，至少一个调用 | 同一个 `hermes` 解析器 |
| 点名函数 | structural tag，恰好一个调用 | 同上，`finish_reason: stop` |

**要格式保证就约束采样；`auto` 不标 `strict` 时格式全靠模型。**

<aside class="notes" markdown="1">
原文：[/output-changes-sampling-and-structured-output.html#6-和结构化输出的关系约束在前解析在后](/output-changes-sampling-and-structured-output.html#6-和结构化输出的关系约束在前解析在后)。`Parser.adjust_request()` 在请求进引擎前把约束写进 `SamplingParams.structured_outputs`：解析器声明了 `structural_tag_model` 且 `VLLM_ENFORCE_STRICT_TOOL_CALLING` 为真（默认）时走 xgrammar structural tag，约束的就是模板教的格式；否则退回 JSON schema。约束一旦写上，就付结构化输出的全部成本（grammar 编译落在 TTFT 上、每步填 bitmask）。推理模型下约束与解析都从 `</think>` 之后开始，由同一个 `is_reasoning_end_streaming()` 定位。
</aside>

<!-- v -->

### 流式输出一个工具调用：20 个 token

| 步 | 新 token | 本步发出的 delta |
|---:|---|---|
| 1–8 | `<tool_call>` … `_weather` | 无：名字的右引号还没到 |
| 9 | `",` | 函数名 `get_weather` + 调用 id |
| 13–18 | ` {"` … `"}}\n` | 六段 `arguments` 片段，拼起来是 `{"city": "广州"}` |
| 19 | `</tool_call>` | 无 |
| 20 | <code>&lt;&#124;im_end&#124;&gt;</code> | 空 delta，`finish_reason: "tool_calls"` |

**名字整块发，参数按文本差分发；客户端拼完才是合法 JSON。**

<aside class="notes" markdown="1">
原文：[/output-changes-sampling-and-structured-output.html#5-实例流式输出一个工具调用](/output-changes-sampling-and-structured-output.html#5-实例流式输出一个工具调用)。Qwen2.5-7B-Instruct，`--enable-auto-tool-choice --tool-call-parser hermes`，`tool_choice` 缺省为 `auto`，不开投机解码。第 10–12 步（` "` `arguments` `":`）不发：后面还没有值。第 18 步一个 token 同时收尾参数对象、外层对象和换行，JSON 完整后才剥掉外层的 `}`。停止 token `<|im_end|>` 不进任何字段；流里出现过工具调用且不是点名函数，`finish_reason` 从 `stop` 改写成 `tool_calls`。
</aside>

---

## 11 · 模型适配：变化落在哪一层

**模型变化按「算什么 / 状态长什么样 / 怎么算」落到模型层、运行时层、算子层**

| 模型变化 | 模型层 | 运行时层 | 算子层 |
|---|---|---|---|
| 激活、归一化 | 组合公共层 | 不动 | 缺算子时加 `CustomOp` |
| MLA | MLA attention 层 | `MLAAttentionSpec` | MLA 专用后端 |
| Mamba / 混合 | `IsHybrid` 声明 | `MambaSpec` 定长状态 | mamba 算子切图 |
| MoE | `FusedMoE` + 路由 | EP 组、EPLB | fused MoE、All-to-All |
| MTP | `DeepSeekMTP` 类 | 回退 `num_computed_tokens` | `RejectionSampler` |
| 新量化 | 不改 | 只有 KV 量化要改 | `LinearMethodBase` 子类 |

<aside class="notes" markdown="1">
原文：[/model-adaptation-architecture.html#2-新模型差异如何向下游扩散](/model-adaptation-architecture.html#2-新模型差异如何向下游扩散)。四种外部变化里的第三种，进入部署级变化：模型在实例启动时就定下，引擎要在接口层把它隔离。表从上往下越来越贵：前两行停在模型层或只多一个后端选项；MLA、Mamba 改了"状态长什么样"，调度器算显存的方式跟着变；MoE、MTP 改了"一步怎么执行"，牵动并行和调度。MTP 不是事务式回滚：被拒位置的 KV 留在原地，下一步覆盖；MoE 也不往 `SchedulerOutput` 加字段。09 用到的 `SupportsLoRA` / `SupportsMultiModal` 就是这里的能力声明，在两个时间点检查：启动时注册表在子进程里查类、写进 `_ModelInfo`（不支持 PP 却开 PP 启动即报错），Worker 加载后对实例做 `isinstance`（要不要包 LoRA 层、挂 EPLB）。
</aside>

<!-- v -->

### 接入链路：主进程只解析配置，Worker 才建模型

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 200}}}%%
%% 图：一个模型从 checkpoint 到第一次 forward：主进程只解析配置，Worker 进程才 import 模型、建结构、装权重
flowchart TB
    subgraph P1["引擎主进程：不 import 模型文件"]
        direction LR
        CFG["<b>ModelConfig</b><br/>读 config.json 与命令行"]
        INS["<b>inspect_model_cls</b><br/>子进程里做 Protocol 检查"]
        INFO["<b>_ModelInfo 能力位</b><br/>不支持 PP 却开 PP → 启动即报错"]
        CFG --> INS --> INFO
    end
    subgraph P2["每个 Worker 进程：建模型、装权重"]
        direction LR
        INIT["<b>initialize_model</b><br/>这时才 import 模型文件"]
        LW["<b>load_weights</b><br/>AutoWeightsLoader + WeightsMapper"]
        WRAP["<b>按能力包装</b><br/>isinstance：LoRA、EPLB、CUDA Graph"]
        SPEC["<b>get_kv_cache_spec</b><br/>逐层收集 KVCacheSpec"]
        INIT --> LW --> WRAP --> SPEC
    end
    P1 --> P2
```

<aside class="notes" markdown="1">
原文：[/model-adaptation-architecture.html#1-接入链路全景](/model-adaptation-architecture.html#1-接入链路全景)，原图每个方框还列了字段，这里只留方框标题与一句话。Worker 里在 `initialize_model` 之前是 `GPUModelRunner.load_model` → `get_model_loader`，`load_weights` 之后有加载后处理（`track_weights_loading`、`process_weights_after_loading`）。主进程按 `architectures` 查注册表，`_ModelInfo` 按源码 hash 缓存，未命中才在子进程里 import 模型类做 Protocol 检查；Worker 里 `resolve_model_cls` 才真正 import 模型文件，`load_weights` 经 `AutoWeightsLoader` + `WeightsMapper` 改名，每个参数的 `weight_loader` 切出本 rank 的一片；有参数没被加载就报错。新模型的验收阶梯 10 级，每级只加一个变量。
</aside>

<!-- v -->

### MLA：状态表示变了，运行时层最贵

![模型适配中的 KV Cache 与 latent 表示变化](/img/in-post/model-adaptation-kv-vs-latent.svg){: style="max-height: 400px"}

**MLA 每 token 存 512 维 latent + 64 维 RoPE key，`get_num_kv_heads()` 返回 1。**

<aside class="notes" markdown="1">
原文：[/model-adaptation-architecture.html#1-mla从注意力优化到-kv-cache-协议扩展](/model-adaptation-architecture.html#1-mla从注意力优化到-kv-cache-协议扩展)。MLA 落在三层：模型层的 `DeepseekV2MLAAttention` + `MultiHeadLatentAttentionWrapper`，运行时层的 `MLAAttentionSpec`（1 个"头" × 576 维），算子层的一族 MLA 后端，prefill、decode 两条计算路径。
</aside>

---

## 12 · 硬件解耦：Platform 是边界，不是万能胶

**核心依赖抽象能力而非具体芯片；OOT 靠主仓库的稳定契约**

![Platform 按硬件与能力派发 Attention Backend、Worker 与算子实现](/img/in-post/hardware-abstraction-platform-dispatch.svg){: style="max-height: 400px"}

<aside class="notes" markdown="1">
原文：[/hardware-abstraction-and-portability.html](/hardware-abstraction-and-portability.html)。四种外部变化的最后一种：输入、输出、模型都还是软件侧的变化，硬件也是部署级、实例启动时定下。Platform 是硬件能力中心，不是所有底层组件的唯一父类；Serving Core、抽象契约、平台实现、硬件运行时与 Kernel 形成多层边界。
</aside>

---

## 13 · Serving 的下一站：分布式智能操作系统

**四转变：配置→自动计划、缓存→分布式状态、执行→多阶段、目标→Goodput**

![AI Serving Operating System 的计算、状态与编排层次](/img/in-post/future-of-serving-infra-ai-serving-os-layers.svg){: style="max-height: 400px"}

<aside class="notes" markdown="1">
原文：[/future-of-serving-infra.html](/future-of-serving-infra.html)。未来系统还要管理可编排计算、可迁移状态、可恢复请求、异构资源和动态执行计划；vLLM 当前代表 Runtime 层。
</aside>

---

## 14 · vLLM 与 SGLang：同一请求，两种状态组织

| 维度 | vLLM v0.27.1 | SGLang v0.5.18 |
|---|---|---|
| 出发点 | PagedAttention：显存碎片 → 块 | RadixAttention：前缀复用 → 树 |
| 调度 | 一个调度器，token 预算，无阶段 | 每 rank 一份调度器，prefill 优先 |
| KV | block hash、`block_table` | RadixCache、`req_to_token` |
| 进程 | SchedulerOutput 跨进程到 WorkerProc | Scheduler 与 ModelRunner 同进程 |

**一句话：vLLM 是“一个调度器 + 块”，SGLang 是“N 个相同调度器 + 树”。**

<aside class="notes" markdown="1">
原文：[/vllm-vs-sglang.html](/vllm-vs-sglang.html)。两边功能面已重叠；结构差异来自状态归属与最初要解决的问题，而非单一 benchmark 排名。版本以正文标注的 vLLM v0.27.1 与 SGLang v0.5.18 为准。
</aside>

---

## 同一个请求的五笔账

| 篇 | 这个请求遭遇了什么 | 数字 |
|---|---|---|
| 05 | KV 切成块，125 个 system prompt 块可复用 | 147 块 / 734 MB |
| 04 | 没有 prefill 阶段，被持续发放 token 额度 | chunk=512 分 5 段 |
| 06 | 97% 的时间花在 300 步 decode 上 | 92 ms + 3,000 ms |
| 07 | 每 token 每步在 8 卡间同步 | 约 2.5 MB、~0.09 ms |
| 08 | 拆成 PD 两池，KV 跨节点搬一次 | 641 MB ≈ 13 ms |

**没有哪一个数字能单独说明问题，合在一起才是系统。**

<aside class="notes" markdown="1">
系列总结第三章第 1 节：[/deep-dive-into-vllm-series-recap-and-self-test.html#1-一个请求的五笔账与各环节耗时量级](/deep-dive-into-vllm-series-recap-and-self-test.html#1-一个请求的五笔账与各环节耗时量级)。同一节还有一张 Llama-2-7B / A100 单卡口径的各环节耗时量级表：权重读取 13.5 GB ÷ 2.0 TB/s ≈ 6.6 ms；decode 一步 batch 1 为 8–12 ms、batch 32 为 10–18 ms（不是 ×32），吞吐 ~100 → ~2,000 tok/s；ITL 稳态下约等于 TPOT，反映的是波动。
</aside>

---

## 两条判断贯穿十四篇

- **KV 是一切约束的源头**：并发上限、抢占、前缀复用、Decode 带宽与 PD 交接。
- **调度单位是 token**：budget、chunk、投机 lookahead 与 encoder 预算都围绕 token 推进。

<aside class="notes" markdown="1">
系列总结：[/deep-dive-into-vllm-series-recap-and-self-test.html](/deep-dive-into-vllm-series-recap-and-self-test.html)。第 15 篇回顾十四篇正文并提供通关自测。
</aside>

---

## 常见误区

- “最大 batch 就是目标”——目标是在 SLO 下的 Goodput。
- “TTFT 等于 Prefill 时间”——TTFT 包含排队。
- “Prefix Cache 按 token 复用”——复用单位是完整块。
- “CUDA Graph 把 forward 融成超级 kernel”——它重放 kernel 序列，主要减少 launch 开销。
- “投机解码总能加速”——batch 与 draft 负载可能抵消验证收益。
- “图片贵在 encoder”——持续驻留的图片 KV 更大。
{: .fragments}

<aside class="notes" markdown="1">
对应原文：[/how-to-measure-llm-serving.html](/how-to-measure-llm-serving.html)、[/kv-cache-memory-core.html](/kv-cache-memory-core.html)、[/gpu-execution-kernels-and-graphs.html](/gpu-execution-kernels-and-graphs.html)、[/request-shapes-multi-lora-and-multimodal.html](/request-shapes-multi-lora-and-multimodal.html)。
</aside>

---

## 下一步

- **系列总结**：第 15 篇 [/deep-dive-into-vllm-series-recap-and-self-test.html](/deep-dive-into-vllm-series-recap-and-self-test.html)
- **往下**：《GPU Kernel 工程》与《通信与互联》——继续看 FlashAttention、PagedAttention、All-Reduce 与 KV 传输。
- **往旁**：《扩散模型推理 Infra》与《RL 后训练 Infra》——比较另一种 Serving 工作负载及训练中的推理引擎。
- **回看总纲**：[/deep-dive-into-vllm.html](/deep-dive-into-vllm.html)

<aside class="notes" markdown="1">
正文顺序为 01–14；系列总结与自测编号为 15。所有结论与数字以对应正文为准。
</aside>
