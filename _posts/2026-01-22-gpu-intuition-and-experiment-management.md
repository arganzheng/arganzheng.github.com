---
layout: post
series: algorithm-tooling
title: "算法工程师的工具箱（06）：GPU 直觉与实验管理——两个上限、四块显存、能复现"
subtitle: "GPU Intuition and Experiment Management: Two Ceilings, Four Memory Buckets, and Reproducibility"
tags: [AI, LLM, PyTorch, Python]
catalog: true
updated: 2026-09-20
---

不写 kernel 的算法工程师也需要一份 GPU 直觉——不是为了优化它，而是为了**解释**：一次训练为什么慢、一次推理为什么快不起来、一个 OOM 从哪里来、我改的这个结构把瓶颈推向了哪边。这份直觉只有两个数字（算力、带宽）和一张表（显存的四块）。本篇讲它们，然后讲读 `torch.profiler` 的表，最后讲一件与 GPU 无关但同样决定实验能不能算数的事：怎么让三个月前的结果能复现。

全篇的核心问题是：

> **不写 kernel，能不能解释一次训练为什么慢、一次推理为什么快不起来、一个 OOM 从哪里来？[^q0] 三个月后能不能复现今天这次实验？[^q1]**

## 一、总览

### 1. 本文的组织方式

前半（二到四章）回答"为什么快为什么慢"：先给 GPU 的两个上限与它们的比值（第二章），用它解释 decode 与 prefill 为什么一个受带宽限制、一个受算力限制（第三章），再把显存分成四块并给出 OOM 的归因顺序（第四章）。第五章把"慢"落到具体算子上——kernel、stream 与 profiler 表怎么读。后半（第六章）换一个问题：怎么让今天的实验三个月后还能复现。两半的共同点是都只到"读数字、算数字"为止，不写 kernel、不碰 CUDA。

### 2. 两个数字一张表

| | 数字 | 含义 |
|---|---|---|
| 算力 | H100 SXM bf16 稠密约 989 TFLOPS | 每秒能做多少 FLOPs |
| 带宽 | HBM3 3.35 TB/s | 每秒能从显存搬多少字节 |
| ridge | 989e12 / 3.35e12 ≈ 295 FLOP / 字节 | 每搬一个字节做多于 295 次运算 → 受算力限制；少于 → 受带宽限制 |
| 显存四块 | 权重 · 梯度与优化器状态 · 激活 · KV cache | OOM 先问落在哪一块 |

### 3. 本文的章节安排

| 章 | 主题 | 内容 |
|---|---|---|
| 二 | 两个上限 | 算力、带宽、算术强度、ridge point |
| 三 | decode 与 prefill | 为什么 decode 是 memory-bound、batch 大才快；prefill 与训练是 compute-bound；MFU |
| 四 | 显存的四块 | 训练与推理各是哪块大；KV cache；OOM 归因 |
| 五 | kernel、stream 与 profiler | 三个概念；读一张 profiler 表 |
| 六 | 实验管理 | 最小记录的七项；工具各管哪项；随机性 |
| 七 | 自测 | 五道题 |


## 二、两个上限

### 1. 算力与带宽

GPU 有两个硬指标。**算力**：每秒能做多少次浮点运算，H100 SXM 上 bf16 稠密矩阵乘约 989 TFLOPS（$$989 \times 10^{12}$$ FLOP/s）。**显存带宽**：每秒能从 HBM（显存）搬多少字节到计算单元，H100 是 3.35 TB/s。

任何一段计算要么受前者限制（**compute-bound**：算不过来，搬数据的时间被算的时间盖住），要么受后者限制（**memory-bound**：算得飞快但数据搬不过来，计算单元在等）。取决于它的**算术强度**（arithmetic intensity）——每搬一个字节做多少次 FLOP。

### 2. ridge point

两个上限之比是分界线：

$$
\text{ridge} = \frac{989 \times 10^{12}\ \text{FLOP/s}}{3.35 \times 10^{12}\ \text{字节/s}} \approx 295\ \text{FLOP/字节}
$$

算术强度高于 295 的操作受算力限制，低于的受带宽限制。这就是 **roofline 模型**的全部内容：一条水平线（算力上限）和一条斜线（带宽 × 强度），操作的性能被压在两条线下面。画出来（两轴都是对数坐标，H100 的数字）：

```text
可达性能
(TFLOP/s)
  989 ┤                                 ┌──────────────────────────  算力上限 989 TFLOPS
      │                               ／ ▲                 ▲
      │                             ／   │ decode B=512    │ prefill / 训练
      │                           ／     │ (compute-bound) │ 强度 ≈ 数千
  100 ┤                         ／
      │                       ／  ▲ decode B=128
      │                     ／    │ (memory-bound：性能 = 3.35 TB/s × 强度)
   10 ┤                   ／
      │                 ／
      │               ／  斜线 = 带宽 × 算术强度
    1 ┤             ／
      │           ／ ▲ decode B=1：强度 1，性能 ≈ 3.35 TFLOP/s = 峰值的 0.3%
      └───────────┴──────────┴──────────┴──────────┴─────────► 算术强度 (FLOP/字节)
                  1          10         100   ridge≈295   1000
```

一个操作的算术强度决定它落在横轴的哪个位置；落在斜线段上的是 memory-bound，往右挪（提高强度）性能线性上升；落到水平段上的是 compute-bound，再挪也不会更快。L4《Transformer 与 LLM》第二篇把整个模型逐层放到这张图上；Infra 05 系列第一篇从硬件侧讲同一件事。算法工程师需要的是用它判断：**我改的这个结构把瓶颈往哪边推了**。

## 三、decode 与 prefill

### 1. decode 是 memory-bound

生成一个 token 要把**全部权重读一遍**（8B 模型 bf16 是 16.06 GB），每个权重只做 2 次 FLOP（乘、加，L0 第一篇的 $$2N$$）。算术强度 $$= 2N / 2N = 1$$ FLOP/字节，远低于 295。所以 batch = 1 时一个 token 的时间**下限**是搬权重的时间：

$$
\frac{16.06\ \text{GB}}{3.35\ \text{TB/s}} \approx 4.8\ \text{ms} \quad \Rightarrow \quad \approx 209\ \text{token/s}
$$

再快就要换更高带宽的卡或量化权重（读的字节少一半），**模型多聪明都没用**。

把 batch 加大：权重读一次被 $$B$$ 个 token 共享，强度变成 $$B$$ FLOP/字节；算的时间 $$2NB / 989\text{T}$$ 随 $$B$$ 线性涨，搬的时间不变。算几个点：

| batch | 带宽时间 | 算力时间 | 瓶颈 | 吞吐 |
|---:|---:|---:|---|---:|
| 1 | 4.8 ms | 0.0 ms | memory-bound | 209 token/s |
| 32 | 4.8 ms | 0.5 ms | memory-bound | 6,675 token/s |
| 128 | 4.8 ms | 2.1 ms | memory-bound | 26,700 token/s |
| 512 | 4.8 ms | 8.3 ms | compute-bound | 61,582 token/s |

batch 从 1 到 128，时间几乎不变（都是 4.8 ms），吞吐涨 128 倍——**这就是"为什么 batch 大才快"**。到 batch ≈ 295（ridge）之后才开始受算力限制。推理系统（vLLM 一类）的核心工作就是把尽量多的请求凑成一个大 batch；Infra 08 系列讲它。

### 2. prefill 与训练是 compute-bound

prefill 把 prompt 的 4096 个 token 一起过模型。算力用第二章那条 $$2N$$ 的规则：每个 token 过一次模型要对每个参数做一次乘加（2 FLOP），$$N = 8.03 \times 10^9$$ 个参数、4096 个 token，就是 $$2 \times 8.03 \times 10^9 \times 4096 \approx 6.6 \times 10^{13} = 66$$ TFLOP。搬运还是 16 GB 权重——只读一次，被 4096 个 token 共享——所以算术强度是 decode 的 4096 倍（$$\approx 4096$$ FLOP/字节），远高于 295。时间由算力决定：$$66 / 989 \approx 67$$ ms。训练同理（每个 batch 几百万 token 一起过），所以训练的效率指标是 **MFU**（Model FLOPs Utilization：实际有效 FLOPs / 峰值 FLOPs），好的训练能到 40–50%——剩下的一半花在通信、数据加载、小算子、等待上。

### 3. 用它判断结构改动

- 把 FFN 做大（$$d_{ff}$$ 翻倍）：参数与 FLOPs 同比涨，decode 每 token 多读一倍权重——memory-bound 的时间翻倍；
- 换成 MoE（每 token 只激活一部分专家）：FLOPs 降、但**全部专家的权重都要在显存里**，decode 读的字节不降反可能升——这是 MoE 推理的特有困难（L4 第五篇）；
- 上下文变长：attention 的 $$QK^T$$ 与 $$T^2$$ 成正比，KV cache 与 $$T$$ 成正比（下一章）；
- 量化到 4 bit：decode 读的字节降到 1/4，memory-bound 的时间也降到 1/4——这是量化对推理有效的根本原因（L6）。

## 四、显存的四块

### 1. 表

| 块 | 大小由什么决定 | 训练 | 推理 |
|---|---|---|---|
| 权重 | 参数量 × 字节 / 参数 | bf16 2 字节 | bf16 2 字节；量化后 1 或 0.5 |
| 梯度与优化器状态 | 可训练参数量 × 14 字节（上一篇：梯度 2 + fp32 主权重 4 + AdamW 两个矩 8）。两者放在一块是因为它们都只在训练时存在、都随**可训练**参数量线性增长——LoRA 把这一块从 8B 缩到 41.9M 的原因就在这里 | 全量微调时的大头 | 无 |
| 激活 | batch × 序列长度 × 层数 × hidden；反向要保存 | 长序列大 batch 时的大头；checkpointing 可换 | 只有当前层，很小 |
| KV cache | batch × 序列长度 × 层数 × $$2 \times n_{kv} \times d_{head}$$ × 字节 | 无 | 长上下文、高并发时的大头；GQA / MLA 就是为了压它 |

前三块上一篇讲过。第四块 **KV cache** 是推理特有的：生成第 $$t$$ 个 token 时要看前面所有 token 的 key 与 value（L0 第四篇：条件不变、缓存有效），所以把它们存下来。每个 token、每层存 $$2 \times n_{kv} \times d_{head}$$ 个数；Llama-3-8B（$$n_{kv} = 8$$、$$d_{head} = 128$$、32 层、bf16）：每个 token $$2 \times 8 \times 128 \times 32 \times 2 = 131$$ KB，一个 8K 的上下文 1 GB，并发 64 个这样的请求 64 GB——**比权重还大**。这就是 GQA（Llama-3 用 8 个 kv 头而不是 32 个，KV cache 缩到 1/4）与 MLA（DeepSeek）的动机。精确公式在 L4 第三篇。

### 2. OOM 归因

一个 OOM，先问它落在哪一块——用上一篇的表加上这里的 KV cache，按"改了什么之后开始 OOM"倒推：

- **参数量没变、batch 没变、只是序列变长了** → 训练时是激活（与序列长度成正比，长序列下还有 $$T^2$$ 的 attention 分数矩阵），推理时是 KV cache（每个 token 131 KB，8K 上下文 1 GB）。对策：训练开 gradient checkpointing 或减 batch；推理减并发或换 GQA / 量化 KV。
- **换了优化器**（比如 SGD → AdamW，或加了 8-bit 优化器又拿掉） → 优化器状态。AdamW 每个可训练参数 8 字节，SGD 是 0。
- **加了 LoRA 还是 OOM** → 不是参数与状态的问题——LoRA 已经把那一块缩到几十 MB——看激活。LoRA 不减少激活：前向经过的层一样多、保存的中间量一样大。对策同第一条。
- **训练能跑、生成时 OOM** → 两种可能：忘了 `torch.no_grad()`，每一步的中间量都被保存等 backward；或者并发太高，KV cache 把显存吃满。
- **第一步就 OOM、什么都没改** → 权重本身放不下：算一下参数量 × 字节 / 参数，看这张卡够不够；不够就换更小的模型、量化、或多卡（上一篇）。

这张表能解释绝大多数 OOM；剩下的用第五章的 profiler 看每个算子的显存变化。

## 五、kernel、stream 与 profiler

### 1. 三个概念

- **kernel** 是 GPU 上执行的一个函数——一次矩阵乘、一次 softmax、一次逐元素加。PyTorch 的每个算子对应一个或几个 kernel。
- **kernel launch 有固定开销**（几微秒）。一个大矩阵乘几毫秒，launch 开销可忽略；一个 $$[32, 128]$$ 的逐元素加几微秒，launch 开销与计算本身相当。所以**小算子多了 GPU 会空转**——这是 `torch.compile` 与 CUDA Graph 做算子融合的动机，也是小模型、小 batch 时 GPU 利用率低的原因。
- **stream** 是 kernel 的执行队列。CPU 把 kernel 扔进队列就继续往下走（异步），GPU 在后面慢慢执行。所以 **`time.time()` 测出来的不是 GPU 时间**——要 `torch.cuda.synchronize()` 等 GPU 做完再计时，或者用 profiler。`loss.item()` 会隐式同步（第三篇），这也是它拖慢训练的原因。

### 2. 读一张 profiler 表

```python
with torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    one_step()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

输出每个算子的时间、调用次数、显存变化。在 CPU 上对第三篇的小 Transformer 跑一步（没有 GPU 就看 CPU 时间，读法相同）：

```text
模型 0.84 M 参数, batch 32 × seq 128; 一步 44 ms（CPU）
Name                                              Self CPU %    Self CPU     # of Calls
aten::mm                                            25.30%     13.504ms          43
aten::_scaled_dot_product_flash_attention_backward  15.82%      8.443ms           4
aten::gelu_backward                                 10.43%      5.566ms           4
aten::addmm                                         10.15%      5.416ms           8
aten::_scaled_dot_product_flash_attention           9.79%      5.227ms           4
aten::gelu                                          7.10%      3.788ms           4
aten::native_layer_norm_backward                    2.85%      1.519ms           9
前三个算子占比: aten::mm 25%, attention_backward 16%, gelu_backward 10%
```

读法：

- **矩阵乘（`mm` / `addmm` / `bmm`）与 attention 应占大头**——这里 `mm` 25% + `addmm` 10% + attention 前后向 26% ≈ 60%，正常。如果 `copy_`、`to`、`contiguous` 或几千个小算子占大头，就是形状转换或 launch 开销在吃时间。
- **反向约是前向的两倍**：`_scaled_dot_product_flash_attention_backward` 8.4 ms vs 前向 5.2 ms，`gelu_backward` 5.6 vs 3.8——L3 第一篇讲的"反向 = 2 × 前向"在表里直接可见。
- `mm` 调用 43 次：4 层 × 每层几个线性层 × 前向 + 反向两个 GEMM。数一数就知道模型结构。

会读它意味着能回答"这一步 300 ms 花在哪"——是 attention、是 FFN 的 GEMM、是数据加载等 GPU 空转（表里 GPU 时间加起来远小于墙钟时间）、还是几千个小算子的 launch 开销。到这里为止；怎么让那个算子快起来，是 Infra 05 系列的事。

## 六、实验管理

### 1. 最小记录的七项

三个月后能不能复现今天这次实验，取决于当时记了什么。最小记录是一行：

```text
run id · commit · 配置文件 · 数据版本 · seed · 环境 · 指标
```

| 需求 | 工具 | 最小做法 |
|---|---|---|
| 记录指标与曲线 | W&B、MLflow、TensorBoard | 每次实验一个 run；记 loss、学习率、梯度范数、评测指标、吞吐；同一张图上叠多个 run 对比 |
| 管理配置 | Hydra / OmegaConf，或 `dataclass` + YAML | 所有超参数进配置文件，命令行只覆盖个别项；配置随 run 一起记录 |
| 代码版本 | git | 每次实验记录 commit hash；有未提交改动时记录 diff 或拒绝启动 |
| 数据版本 | 数据文件的 hash 或 `datasets` 的 revision | 数据变了就是另一个实验 |
| 环境 | `pip freeze` / 锁文件、CUDA 与驱动版本、容器镜像 | 随 run 记录 |
| 随机性 | `seed` 参数 + `torch.manual_seed` 等 | 多 seed 报均值与方差；知道有些 kernel 本身不确定 |
| 产物 | checkpoint、评测输出、生成样本 | 命名含 run id；评测输出保存到能做第二篇那种错误分析的粒度 |

这一行齐了，"复现三个月前的结果"就是重跑一条命令。少了任何一项，那次实验的结论都只是"当时好像是这样"。

### 2. 一次记录长什么样

```text
seed 0 两次: 3.237898 vs 3.237898 → 一致
seed 1:      3.378334 → 与 seed 0 差 0.1404，这就是'单个数字不算结论'的原因
记录写到 out/run-20260914-205033.json：run id · commit · 配置 · 数据版本 · seed · 环境 · 指标
{"run_id": "20260914-205033", "commit": "a9531a3", "metrics": {"loss_seed0": 3.2379, "loss_seed1": 3.3783}}
```

三件事：同 seed 两次结果**完全一致**（CPU 上；GPU 上某些 kernel——比如 atomic add 的 scatter——本身非确定，需要 `torch.use_deterministic_algorithms(True)` 且接受变慢）；换个 seed 差 0.14——**20 步的训练里 seed 的影响比很多"方法改进"都大**，这是 L0 第八篇"多 seed 报均值与标准差"在工程上的原因；记录是一个 JSON，七项齐全。

### 3. 与方法论的关系

这一章是横切"实验方法论"的物质基础——方法论讲"怎么设计实验才能得出可信的结论"，这里讲"用什么工具把实验记下来"。工具很便宜（W&B 一行 `wandb.init`、Hydra 一个装饰器），贵的是习惯：**每次实验先想"三个月后我怎么复现它"**。

配套代码：本文的数字（batch 与吞吐的几个点、profiler 表、一次实验记录）由 [`algorithm-tooling/05_profiler_and_record.py`](https://github.com/arganzheng/ai-learning-labs/blob/main/algorithm-tooling/05_profiler_and_record.py) 产生，CPU 可跑；复现时去拉它，读本文不需要。

## 七、自测

1. 一张卡带宽 2 TB/s、算力 500 TFLOPS：ridge 是多少？一个 70B bf16 模型 batch 1 decode 的时间下限？

   <details markdown="1"><summary>答案</summary>

   250 FLOP/字节；$$140\ \text{GB} / 2\ \text{TB/s} = 70$$ ms。

   </details>

2. 把 8B 模型量化到 int4（0.5 字节 / 参数），batch 1 decode 的下限变成多少？为什么量化对推理有效而对训练用处不大？

   <details markdown="1"><summary>答案</summary>

   $$4.0\ \text{GB} / 3.35 = 1.2$$ ms；训练是 compute-bound，读权重的字节不是瓶颈，且反向需要高精度的梯度。

   </details>

3. Llama-3-8B 一个 32K 上下文的 KV cache 多大？如果 $$n_{kv}$$ 是 32 而不是 8 呢？

   <details markdown="1"><summary>答案</summary>

   $$131\ \text{KB} \times 32768 = 4.3$$ GB；$$n_{kv} = 32$$ 时 17 GB——比权重还大，这就是 GQA 的理由。

   </details>

4. profiler 表里 `aten::copy_` 占了 40%，最可能的原因是什么？

   <details markdown="1"><summary>答案</summary>

   大量非连续张量的 `contiguous()` / dtype 转换 / device 拷贝——形状变换或 `.to()` 太多。

   </details>

5. 两次"同样配置"的实验结果差 0.5 个点，在下结论之前要先排除什么？

   <details markdown="1"><summary>答案</summary>

   seed（跑几个 seed 看方差）、数据版本、环境（库版本、非确定 kernel）——七项里有没有哪一项其实不同。

   </details>

[^q0]: 能。GPU 有算力与带宽两个上限，比值 ridge $$\approx 295$$ FLOP/字节（H100）。decode 每生成一个 token 要读全部权重、每个权重只做 2 次运算，算术强度约 1，远低于 ridge——memory-bound，batch = 1 时 16 GB 权重 / 3.35 TB/s $$\approx 4.8$$ ms 是下限；batch 加大到 128 时间几乎不变、吞吐涨 128 倍，这就是「batch 大才快」。训练与 prefill 是 compute-bound，慢通常是 MFU 低——通信、数据加载、小算子、等待。OOM 从四块显存里找：权重、梯度与优化器状态、激活、KV cache。详见[第二](#二两个上限)至[四章](#四显存的四块)。
[^q1]: 能，记七项——代码版本、配置、数据版本、环境、seed、硬件、结果——放进一次 `log()`，三个月后按它重跑。详见[第六章](#六实验管理)。
