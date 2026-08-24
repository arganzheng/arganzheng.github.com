---
layout: post
title: 大模型推理系统揭秘（01）：为什么 LLM Serving 比传统 DL 推理难？
tags: [AI, AI-infra, 大模型推理]
catalog: true
---

> **NOTE** 本文基于 vLLM v0.27.1（tag `6e448d0`, 2026-08-11）源码深度剖析。文中所有文件路径、类名和行号均以该版本为准；vLLM 迭代很快，阅读时请以你手上的版本对照。


传统深度学习推理通常围绕一个相对稳定的计算过程展开：输入一批形状相近的数据，执行一次前向传播，然后返回结果。

LLM Serving 则完全不同。一个请求可能携带几十个，也可能携带数万个输入 token；模型还需要逐个生成长度未知的输出 token。在这个过程中，系统必须动态管理请求、显存、KV Cache、批次和 GPU 计算资源。

因此，LLM Serving 的难点并不只是“模型更大”或“参数更多”，而是它改变了推理服务的基本执行模型：

> 从一次性、静态、可预测的前向计算，转变为持续进行、动态变化、状态不断增长的自回归计算。

整体来说，有如下三方面的变化和挑战：

1. 传统深度学习推理与 LLM Serving 存在巨大的范式差异
2. Prefill 与 Decode 是两种完全不同的 GPU Workload
3. 吞吐与时延之间难以避免的系统权衡

## 1. 范式转移：服务对象从“一次计算”变成“持续生成过程”

理解 LLM Serving 的第一步，不是从某个具体优化技术开始，而是先看清它与典型传统 DL Serving 在**服务对象**上的差异：**从一次 Forward 到持续生成**。

许多传统深度学习推理任务都可以抽象为“一次请求对应一次前向计算”：

```text
请求进入 → 组 Batch → 执行一次 Forward → 返回结果
```

即使输入 shape 存在一定变化，系统通常也可以通过 padding、bucketing 或预编译的执行形态，将请求规整为有限几种执行模式。在推理过程中，中间激活一般即可释放。

LLM Serving 则不同。一个请求通常要经历：

```text
请求进入
  ↓
Prompt Processing / Prefill
  ↓
生成第一个 Token
  ↓
Decode Step 1
  ↓
Decode Step 2
  ↓
Decode Step ...
  ↓
EOS 或达到长度上限
```

它的输入和输出都具有动态性：

- **输入长度不固定**：可能是几十个 token，也可能是数万甚至更长的上下文；
- **输出长度不确定**：模型何时生成 EOS，事先无法准确确定；
- **服务时长不可直接预知**：请求可能只生成几个 token，也可能持续数千个 token；
- **中间状态需要跨多个 step 保留**：Prefill 产生的 KV Cache 会被后续 Decode 持续访问。

因此，LLM Serving 的核心变化并不只是模型更大，而是服务对象从“一次性计算任务”变成了**带有动态生命周期和持久中间状态的生成任务**。

| 维度 | 典型传统 DL Serving | LLM Serving |
|---|---|---|
| 服务单位 | 一次或少数几次 Forward | 多轮自回归生成 |
| 输入长度 | 通常可规整为有限几种形态 | 可变，可能非常长 |
| 输出长度 | 通常较容易确定 | 由生成过程动态决定 |
| 中间状态 | 多为临时激活 | KV Cache 跨 Decode Step 持续存在 |
| Batch 语义 | 请求级静态或动态 Batch | 面向持续执行的 Continuous Batching |
| 内存模式 | 相对静态 | KV Cache 动态增长和回收 |
| 服务时长 | 相对容易估计 | 与输出长度、调度和资源竞争有关 |
| 主要挑战 | 高效执行一次计算 | 计算、调度和状态管理协同优化 |

这并不意味着传统 Serving 的技术全部失效，而是意味着系统不能再只依赖以下假设：

- 输入和执行 shape 基本固定；
- 一次 Forward 即可完成请求；
- 请求服务时长容易估计；
- 执行过程中或者执行后无需保存中间状态；
- 内存可以按请求一次性分配。

LLM Serving 需要在显存预算、请求生命周期、生成进度和服务等级约束之间进行动态决策。这是后续调度、缓存和执行优化的共同背景。


## 2. Prefill 与 Decode：两种完全不同的 GPU Workload

LLM 推理通常分为 Prefill 与 Decode 两个阶段：

```text
──[       Prefill       ]──[D][D][D][D][D][D][D]──▶
                            ↑  ↑  ↑  ↑  ↑  ↑  ↑
                  每步生成 1 个 Token
```

Prefill 主要决定**第一个 Token 何时到达**，Decode 则决定**后续 Token 的生成速度和稳定性**。二者不仅执行顺序不同，计算规模、并行方式和硬件瓶颈也明显不同。

```text
┌──────────────────────────── Prefill 数据流 ─────────────────────────────┐
│                                                                         │
│  input_ids: [t₁, t₂, t₃, ..., tₙ]    (N 个 Prompt Token 并行输入)      │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────┐  Q: [N, H, d]                                             │
│  │  QKV    │  K: [N, Hkv, d]   ──写入──▶  KV Cache (N 个新位置)         │
│  │  Proj   │  V: [N, Hkv, d]   ──写入──▶  (批量写入 Prompt)             │
│  └─────────┘                                                            │
│       │                                                                 │
│  Attention: 逻辑上计算 Q × Kᵀ 和 Attention × V                          │
│             实际通常由 FlashAttention 等 Kernel 分块完成                │
│       │                                                                 │
│  只取最后 1 个 Position 的 Logits → Sampling → 第 1 个 Output Token     │
│                                                                         │
│  特征: Compute-Bound 倾向，高 GPU 利用率，大矩阵 GEMM                   │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────── Decode 数据流 ──────────────────────────────┐
│                                                                         │
│  input_ids: [tₙ₊ₖ]                 (1 个新 Token 输入)                   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────┐  q: [1, H, d]                                             │
│  │  QKV    │  k: [1, Hkv, d]   ──追加──▶  KV Cache (1 个新位置)         │
│  │  Proj   │  v: [1, Hkv, d]   ──追加──▶  (增量写入)                    │
│  └─────────┘                                                            │
│       │                                                                 │
│  Attention: q × K_historyᵀ → × V       (读取全部历史 KV)                │
│       │                                                                 │
│  Logits → Sampling → 下一个 Output Token                                │
│                                                                         │
│  特征: Memory-Bound 倾向，单请求并行度低，大量 HBM 读取                  │
└─────────────────────────────────────────────────────────────────────────┘
```

可以把两者的差异概括为：

> Prefill 一次处理多个 Prompt Token，并批量写入 KV Cache；Decode 每个 Step 只处理一个新 Token，却要读取此前累积的历史 KV Cache。

| 维度 | Prefill | Decode |
|---|---|---|
| 单步输入 Token 数 | N 个 Prompt Token | 每个请求 1 个新 Token |
| 单请求序列并行度 | 高 | 低 |
| 跨请求并行度 | 可通过 Batch 提升 | 依赖 Continuous Batching |
| KV Cache 操作 | 批量写入 N 个位置 | 追加 1 个位置，读取历史 KV |
| Attention 序列相关计算 | O(N² · d) | O(L · d)，L 为当前序列长度 |
| 总计算量 | O(N·d² + N²·d) | O(d² + L·d) per Step |
| 典型瓶颈 | Compute、长序列 Attention | HBM 带宽、KV Cache、通信 |
| 常见优化 | FlashAttention、Chunked Prefill | PagedAttention、FlashInfer、CUDA Graph |
| CUDA Graph | 需视执行形态而定 | 适合形状和路径相对稳定的场景 |

在多请求服务中，Prefill 和 Decode 会竞争同一组 GPU 资源：

- 长 Prefill 可能占据大量计算资源，导致 Decode 延迟抖动；
- Decode 请求过多，可能挤压 Prefill 的计算机会；
- 新请求不断进入，会同时消耗计算预算和 KV Cache 容量。

这也是以下机制存在的原因：

- Continuous Batching；
- Token Budget；
- Chunked Prefill；
- 抢占与重计算；
- Prefill/Decode 分离。

## 3. 吞吐与时延：无法同时最优的权衡

扩大 Batch 通常可以提高吞吐，因为更多请求能够共享一次权重读取和 GPU 计算。

但 Batch 增大也会带来：

- 更长的计算等待；
- 更激烈的显存带宽竞争；
- 更高的单请求延迟；
- 更严重的尾延迟；
- 更大的 KV Cache 压力。

```mermaid
xychart-beta
    title "Batch Size 对吞吐和延迟的影响"
    x-axis "Batch Size" [1, 2, 4, 8, 16, 32]
    y-axis "相对值" 0 --> 100
    line "吞吐" [12, 25, 45, 68, 85, 92]
    line "单请求延迟" [10, 16, 25, 39, 60, 88]
```

一般而言：

- **吞吐**：先随 Batch 增大而增长，随后进入饱和；
- **单请求延迟**：通常随 Batch 增大而上升；
- **尾延迟**：在资源接近饱和时可能快速恶化。

因此，系统目标不是简单地追求最大 Batch，而是在 SLO 约束下选择合适的执行规模：

```text
吞吐最大化
      ▲
      │      可接受区间
      │    ┌──────────┐
      │   /            \
      │  /              \
      └────────────────────▶
          延迟 / 尾延迟约束
```

调度器需要同时考虑：

- 当前活跃请求数；
- Prompt 长度；
- 已生成 token 数；
- KV Cache 占用；
- 请求优先级；
- TTFT、ITL 和 P99 等 SLO。

这也是 Token Budget、Chunked Prefill、抢占和 Admission Control 等机制出现的原因。


## 4. 小结：LLM Serving 的三个根本变化

到这里，可以将 LLM Serving 相对于传统深度学习推理的变化概括为三个方面。

### 变化一：从静态计算变成动态执行

传统推理通常可以预先确定输入形状、Batch 大小和计算过程。

LLM Serving 中：

- Prompt 长度不确定；
- 输出长度不确定；
- 请求完成时间不确定；
- 每轮活跃请求集合不确定。

因此，系统需要持续调度，而不是只在请求到达时进行一次 Batch 组装。

### 变化二：从无状态推理变成带状态推理

传统推理通常只需要处理输入、模型参数和临时激活。

LLM Serving 还必须管理每个请求不断增长的 KV Cache。KV Cache 的位置、容量、复用和回收都会直接影响：

- 最大并发数；
- 可支持的上下文长度；
- 显存利用率；
- 请求是否需要抢占；
- 系统整体吞吐。

### 变化三：从单一 Workload 变成 Prefill 与 Decode 的混合 Workload

Prefill 更偏向计算密集型，Decode 更偏向访存密集型。

这意味着：

- 适合 Prefill 的优化，不一定适合 Decode；
- 提高 GPU 算力利用率，不一定能降低 Decode 延迟；
- 只优化 Kernel，不一定能解决排队问题；
- 只优化 KV Cache，也不一定能改善长 Prompt 的 TTFT。


## 5. 阅读地图：后续内容如何展开

后续章节将围绕 LLM Serving 的四个核心问题展开。

| # | 核心问题 | 主要机制 | 主要代码落点 | 后续章节 |
|---|---|---|---|---|
| 一 | 请求来了，**这一轮谁执行、执行多少？** | Continuous Batching、Token Budget、Chunked Prefill、抢占 | `Scheduler` | 调度 |
| 二 | 历史状态**放在哪里、如何复用？** | PagedAttention、Prefix Cache、GQA/MLA、KV 量化 | `KVCacheManager` / `BlockPool` | KV Cache |
| 三 | 这一轮**如何计算得更快？** | FlashAttention、CUDA Graph、算子融合、量化、投机解码 | `ModelRunner` / Attention Backend | 执行优化 |
| 四 | 一张卡不够，**如何扩展？** | TP、PP、EP、CP、DP、集合通信 | `Executor` / `distributed` | 多卡与集群 |   |

四问之外还有两个**横切约束**：模型在变、硬件在变——它们不新增问题，但要求上面四个答案在剧烈变化的外部环境里保持稳定。

## 6. 一个贯穿全文的例子

抽象的讨论容易滑走，所以从这里开始，本文会**反复回到同一个具体请求**。后面每一章都会带着它算一笔账。

> **场景设定**
>
> - **模型**：Llama-3-70B，FP16，80 层，`hidden=8192`，64 个 Q head / 8 个 KV head（GQA-8），`head_dim=128`
> - **硬件**：8 × H100 80GB，TP=8，机内 NVLink
> - **请求**：2000 token 的 system prompt + 50 token 的用户提问，生成 300 token
>   - prompt 合计 **2050** token，结束时序列长度 **2350** token

先把两个最基础的量算出来，后面各章都要用：

| 量 | 计算 | 结果 |
|---|---|---|
| 每 token 每层的 KV | `2(K,V) × 8 kv_head × 128 dim × 2 B` | **4 KB** |
| 每 token 的 KV（80 层） | `4 KB × 80` | **320 KB** |
| ↳ TP=8 时每张卡承担 | `320 KB ÷ 8` | 40 KB |
| 这个请求最终的 KV 总量 | `2350 × 320 KB` | **约 734 MB** |
| 权重每卡 | `141 GB ÷ 8` | 17.6 GB |


