---
layout: post
series: transformer-and-llm
title: "Transformer 与 LLM（09）：长上下文的成本与结构手段"
subtitle: "The Cost of Long Context: KV Cache, Quadratic Prefill and Structural Remedies"
tags: [Transformer, LLM, AI, AI-Infra]
catalog: true
date: 2026-04-04 14:00:00
---

> **本篇在系列中的位置。** 第二段的第五篇。第 07 篇讲了位置能不能外推，第 08 篇讲了 KV cache 的账，本篇讲另一组问题——"用得起吗"：上下文拉长后 KV cache、prefill 与中间量各涨成什么函数，sliding window、全局/局部交错、attention sink、稀疏 attention 又把成本改成了什么函数。完整地图见[总纲](/transformer-and-llm-for-infra-engineers.html)。

[《Transformer 与 LLM（07）：位置编码与外推》](/positional-encoding-and-long-context.html)回答了模型凭什么知道一个 token 在第几个位置、为什么用 8K 训练的 RoPE 模型不能直接推理 32K。但即使位置编码完全没问题，上下文长度 $$s$$ 仍然受另一组限制：它同时进入 KV cache 的一次项和 attention 算量的二次项。Llama-3-70B 在 128K 上下文下，每个 token 花在 attention 上的算量（344 GFLOPs）已经超过了花在全部权重上的算量（141 GFLOPs）。这些限制与位置怎么编码无关，改变它们要改 attention "看哪些 token"。

本篇要回答的核心问题是：

> **一个 128K 的请求到底贵在哪里？[^q0] 滑窗、交错、sink、稀疏各把这笔账改成了什么？[^q1]**

## 一、总览：一个线性项、一个二次项、一个中间量

### 1. 本文的思路

先把 full attention 下长上下文的三项成本算出来（第二章），再看四类结构手段各自把哪一项从什么函数改成什么函数（第三章），最后落到推理系统上：并发数、TTFT、chunked prefill 与序列并行（第四章）。

### 2. 数字基线

模型同上一篇：Llama-3-8B、Llama-3-70B、DeepSeek-V3；硬件以 H100 SXM 为准（BF16 dense 989 TFLOPS，3.35 TB/s，80 GB）。所有 FLOPs 与字节数都是理论下界，不是实测。

### 3. 本文的章节安排

| 章 | 主题 | 内容 |
|---|---|---|
| 二 | 长上下文的成本 | attention 算量与权重算量的交叉点、128K prefill 的 11 秒、KV cache 线性项、s² 的 logits |
| 三 | 缩短成本的结构手段 | sliding window、全局/局部交错、attention sink、稀疏 attention；每种手段改成了什么函数 |
| 四 | 对 Infra 的影响汇总 | 并发数、TTFT、chunked prefill、序列并行 |
| 五 | 实践 | `llm_cost.py` 上下文长度扫描 |
| 六 | 本文小结 |  |
| 七 | 自测 | 2 道题 |

Table: 本文的章节安排

## 二、长上下文的成本

位置编码决定了模型**能不能**处理长上下文；这一章算它**要花多少**。第六篇的两个基本公式重新写在这里：矩阵乘 $$[m, k] \times [k, n]$$ 是 $$2mkn$$ FLOPs，因此每参数每 token 2 FLOPs；attention 对上下文 $$s$$ 的部分，每层每 token $$QK^\top$$ 与 $$PV$$ 各 $$2 \cdot n_h \cdot d_{head} \cdot s = 2ds$$，合计 $$4ds$$。

### 1. 每 token 的 attention 算量与权重算量的交叉点

Llama-3-8B 的权重 GEMM 部分每 token $$2 \times (8.03 - 0.53)\text{B} \approx 15.0$$ GFLOPs（embedding 查表不算 GEMM，lm_head 算）。attention 部分每层 $$4 \times 4096 \times s = 16384\,s$$，32 层 $$0.524 \times 10^6 \cdot s$$ FLOPs：

```text title="attention 算量占比随上下文变化"
                      Llama-3-8B                       Llama-3-70B
上下文 s      attention/token   占比           attention/token   占比
  8192          4.3 GFLOPs      22.2%            21.5 GFLOPs     13.2%
 32768         17.2 GFLOPs      53.4%            85.9 GFLOPs     37.9%
131072         68.7 GFLOPs      82.1%           343.6 GFLOPs     70.9%
权重部分       15.0 GFLOPs                       141 GFLOPs
```

Llama-3-70B 在 128K 下：$$4 \times 8192 \times 131072 \times 80 \approx 344$$ GFLOPs，是权重 141 GFLOPs 的 2.4 倍。交叉点（attention 等于权重）在 8B 约 28.6K、70B 约 53.8K——超过这个长度，模型每生成一个 token 的主要算量就不再是"跑一遍权重"，而是"看一遍上下文"。

这个交叉点对 decode 的 Roofline 判断有直接影响。第六篇的结论是 decode 时权重 GEMM 的算术强度约等于 batch 大小 $$B$$（BF16），$$B = 1$$ 时距 H100 的 ridge point 295 差两个数量级，是 memory-bound。attention 对 KV cache 的读取也是 memory-bound 的，而且它**不随 batch 摊薄**——每个请求有自己的 KV cache，$$B$$ 个请求就读 $$B$$ 份。上下文 8K、batch 64 时 8B 模型每步要读 $$128\,\text{KiB} \times 8192 \times 64 = 64$$ GiB 的 KV cache，是权重（16 GB）的四倍。长上下文下 decode 的瓶颈从"读权重"变成"读 KV cache"。

### 2. prefill 的二次项：一个 128K 请求的 11 秒

prefill 要对 $$s$$ 个 token 各算一遍。权重项与 $$s$$ 成正比，attention 项与 $$s^2$$ 成正比（利用因果掩码，只算下三角，取 $$s^2/2$$）：

$$
\text{FLOPs}_{prefill}(s) = 2N_{gemm} \cdot s + 4dL \cdot \frac{s^2}{2}
$$

Llama-3-8B、$$s = 131072$$：

$$
\underbrace{15.0 \times 10^9 \times 131072}_{\approx 2.0\ \text{PFLOP}} + \underbrace{0.524 \times 10^6 \times \frac{131072^2}{2}}_{\approx 4.5\ \text{PFLOP}} \approx 6.5\ \text{PFLOP}
$$

H100 BF16 989 TFLOPS，按 60% MFU 算 593 TFLOPS，$$6.5 \times 10^{15} / 593 \times 10^{12} \approx 11$$ s。同一个模型 8K 的 prefill 约 0.14 PFLOP、0.24 s；128K 是 8K 的 16 倍长度、46 倍算量、46 倍时间。二次项已经占了 70%。

对 70B，128K prefill 约 41 PFLOP（权重 18.5 + attention 22.5），单卡 60% MFU 要 69 s；即便 8 卡 TP 完美线性，也接近 9 s。这就是 TTFT（time to first token）在长上下文下的量级：不是调度问题，是算量问题（60% 是经验效率，按峰值算的物理下界是 6.6 s / 41 s；第六篇第六章说明了两者的区别）。

### 3. KV cache 的线性项

第八篇的公式：每 token 的 KV cache 字节数为 $$2 \cdot L \cdot n_{kv} \cdot d_{head} \cdot \text{bytes}$$。

| 模型 | bytes/token | 8K | 32K | 128K |
|---|---|---|---|---|
| Llama-3-8B | 128 KiB | 1.0 GiB | 4.0 GiB | 16 GiB |
| Llama-3-70B | 320 KiB | 2.5 GiB | 10.0 GiB | 40 GiB |
| DeepSeek-V3 (MLA) | 68.6 KiB | 0.54 GiB | 2.1 GiB | 8.6 GiB |

Table: 三个模型在 8K、32K、128K 上下文下的 KV cache

Llama-3-8B 一个 128K 请求的 KV cache 是它权重（16.06 GB）的大小；70B 一个 128K 请求 40 GiB，是单张 H100 一半的显存。对 Infra 来说，长上下文的 KV cache 意味着**并发数**的上限：H100 放下 8B 权重后剩约 64 GB，8K 上下文可以放 64 个请求的 KV，128K 只能放 4 个。

### 4. 中间量：s² 的 logits 矩阵

还有一项不在 FLOPs 和 KV cache 里，但在实现上更致命：attention 分数矩阵 $$QK^\top$$ 本身是 $$s \times s$$。128K 时单个 head 的 logits 就是 $$131072^2 \times 2\,\text{B} = 32$$ GiB（BF16），32 个 head 就是 1 TiB。它不可能物化。FlashAttention（Dao 等 2022）把 softmax 拆成分块的 online softmax，logits 只在 SRAM 里存一个块，从来不写回 HBM——长上下文可行的前提不是显存大，而是 attention kernel 从不物化 $$s \times s$$。同样，训练时 attention 的中间激活如果物化，反向传播也需要它，这是 FlashAttention 对长上下文训练的意义。

## 三、缩短成本的结构手段

上面的成本函数是 full attention 的：每个 token 看所有前面的 token。改变"看哪些"就改变了函数形式。

### 1. sliding window attention

Mistral 7B（Jiang 等 2023）让每个 token 只看前面 $$W = 4096$$ 个 token。attention 算量从 $$4dLs$$ 变成 $$4dL \cdot \min(s, W)$$，KV cache 从 $$s$$ 个 token 变成 $$\min(s, W)$$ 个——两者都从 $$O(s)$$ 变成 $$O(W)$$，即常数。Mistral 7B 的每 token KV cache 与 Llama-3-8B 相同（32 层、8 个 KV 头、$$d_{head} = 128$$，128 KiB），但上限是 $$4096 \times 128\,\text{KiB} = 512$$ MiB，无论输入多长；attention 算量上限 $$4 \times 4096 \times 4096 \times 32 \approx 2.1$$ GFLOPs/token，不到权重的 15%。

实现上 KV cache 变成一个环形缓冲区（rolling buffer），位置 $$n$$ 的 K、V 写到槽 $$n \bmod W$$，第 $$W + 1$$ 个 token 覆盖第 1 个。vLLM 对 sliding window 模型的 block 分配就是按这个做的。

代价是信息只能通过层间传递向远处流动：第 $$\ell$$ 层的 token 能间接看到 $$\ell \cdot W$$ 之内的信息（每层扩一个窗口），32 层理论上是 131072，但每一跳都有损。事实上 Mistral 后续的模型（Mistral Large、Mixtral 的部分版本）取消了滑窗，回到 full attention，说明纯滑窗在长程精确检索任务上有代价。

### 2. 全局层与局部层交错

Gemma 2（2024）把两种层交错排列：奇数层用 $$W = 4096$$ 的局部 attention，偶数层用 full attention，1:1 交错。这是一个折中：一半的层保留了远距离精确取回的能力，另一半的 KV cache 与算量被封顶。

KV cache 变成：

$$
\text{KV}(s) = \frac{L}{2} \cdot b \cdot \min(s, W) + \frac{L}{2} \cdot b \cdot s
$$

其中 $$b$$ 是每层每 token 的 KV 字节数。$$s \gg W$$ 时约为 full attention 的一半。attention 算量同理，$$s \gg W$$ 时约减半。它没有改变函数的阶（仍然 $$O(s)$$ 的 KV 与 $$O(s^2)$$ 的 prefill），只是把系数减半；如果全局层占 $$1/k$$，系数就变成 $$1/k$$。这类设计在 2024–2025 年成为长上下文模型的常见选择，全局层比例从 1:1 到 1:5 不等。

### 3. attention sink 与 StreamingLLM

StreamingLLM（Xiao 等 2023）观察到一个现象：在 full attention 训练的模型里，大量的 attention 分数会集中到序列最开头的几个 token 上，无论那几个 token 是什么内容——它们是 softmax 的"泄洪口"（attention sink）：当一个 query 与所有 key 都不相关时，softmax 仍然要把概率分配出去，模型学会了把这些概率倒进开头几个 token。

这解释了为什么朴素的滑窗（丢掉最早的 token）会让 full attention 训练的模型崩溃：sink 被丢了，softmax 的概率没地方去。StreamingLLM 的做法是永远保留开头 4 个 token 的 K、V，再加一个滑动窗口。KV cache 是 $$b \cdot L \cdot (4 + \min(s, W))$$，与滑窗同阶。它使一个 full attention 训练的模型可以在不微调的情况下处理无限长的流式输入——但代价与滑窗一样，窗口之外的信息丢失了，它是"流式稳定"而非"长上下文理解"。

对位置编码有一个细节：保留 sink 并滑动窗口后，位置用的是**cache 内的相对位置**（sink 是 0–3，窗口内从 4 开始连续编号），而不是原始文本中的位置；否则 RoPE 的相对距离会超过训练长度，回到第七篇第四章的外推问题。以 $$W = 6$$、当前正在生成第 10003 个 token 为例，cache 里的内容与它们用的 RoPE 位置是：

```text title="StreamingLLM：cache 槽与 RoPE 位置"
原始位置   0   1   2   3 | 4 … 9996 | 9997 9998 9999 10000 10001 10002 | 10003
           └ sink 保留 ┘   └ 已丢弃┘  └─────── 窗口 W=6 ─────────────┘ 新 token
cache 槽   0   1   2   3               4    5    6     7     8     9
RoPE 位置  0   1   2   3               4    5    6     7     8     9      10
```

新 token 与最早的窗口 token（t9997）之间的相对距离在 RoPE 看来是 $$10 - 4 = 6$$——这一项本来就是 $$10003 - 9997 = 6$$，窗口内的相对距离重编号前后不变；真正被改变的是 sink 与新 token 的距离：原文里是 10003，重编号后是 10。所有相对距离都被控制在 $$W + 4$$ 之内，永远不会超出训练长度。这是 sink + 滑窗特有的处理；普通滑窗（不保留 sink）不需要重编号，用原始位置即可——窗口内的相对差值本来就 $$\le W$$。

### 4. 稀疏 attention 的形态

更一般的做法是让每个 query 只看上下文中的一个子集，子集的选法有多种：

- **块稀疏**（block-sparse）：把 K、V 按块（例如 64 个 token）分组，每个 query 块只看部分 key 块——固定模式（局部块 + 若干全局块，如 Longformer、BigBird）或按块的粗粒度分数动态选（top-k 块）。稀疏度 $$\rho$$（保留的块占比）下算量约为 $$4dLs \cdot \rho$$；KV cache 通常不减（需要保留所有块以便选择），除非配合淘汰策略。
- **DeepSeek 的 NSA**（Native Sparse Attention，Yuan 等 2025）一类：把上下文压成粗粒度的块表示、按块选 top-k 精细 attention、再加一个滑窗，三路结果加权，并在训练时就使用这种结构，让 kernel 形态对齐硬件的块粒度。这一类方法的算量是 $$O(s \cdot k_{blocks} \cdot B_{size})$$ 加压缩部分，接近线性。

这里只提形态，不展开。关键在于任何稀疏方法的实现都要处理**索引与不规则访存**——被选中的 K、V 块散落在 HBM 中，kernel 要 gather，这与 dense attention "顺序读一整段"是完全不同的访存模式，也是稀疏 attention 的理论算量节省常常兑现不到实测的原因。

### 5. 对比：每种手段把成本改成了什么函数

以每层每 token 的 KV 字节 $$b$$、模型层数 $$L$$、上下文 $$s$$、窗口 $$W$$、全局层占比 $$1/k$$、稀疏度 $$\rho$$ 表示：

```text title="五种手段的 KV cache 与算量函数"
手段                    KV cache（每请求）                   attention 算量（每 token）
full attention          b · L · s                            4 d L · s
sliding window (W)      b · L · min(s, W)                    4 d L · min(s, W)
全局/局部交错 (1/k)     b · L · [ s/k + (1 − 1/k) min(s, W) ] 4 d L · [ s/k + (1 − 1/k) min(s, W) ]
sink + window           b · L · (4 + min(s, W))              4 d L · (4 + min(s, W))
块稀疏（保留 ρ）        b · L · s（一般不减）                4 d L · s · ρ  + 选择开销
MLA（DeepSeek-V3）      (d_c + d_h^R) · L · s（系数减 57 倍） 与 full attention 同阶（吸收后 FLOPs 更高）
```

值得注意的是最后一行：MLA 减的是 KV cache 的**系数**（从 3.81 MiB 到 68.6 KiB），不改变它对 $$s$$ 的线性依赖，也不减少 attention 算量；而滑窗改的是**阶**（从 $$s$$ 到常数），但丢信息。两者正交，可以叠加。

## 四、对 Infra 的影响汇总

把前两章的成本落到系统上，长上下文带来的影响集中在五处。

**线性项：KV cache 决定并发数。** 每请求 KV cache $$= b \cdot L \cdot s$$，Llama-3-70B 在 128K 是 40 GiB。给定显存预算，最大并发数与上下文长度成反比；这也是 PagedAttention（vLLM）按 block 而不是按最大长度预分配 KV 的原因——大多数请求用不满 128K，但只要**允许** 128K，静态分配就得按 128K 留。GQA（8B/70B 的 8 个 KV 头）与 MLA（DeepSeek-V3 的 68.6 KiB/token）是从模型结构上减这一项的系数。

**二次项：prefill 算量。** 128K 的 prefill 在 8B 上是 6.5 PFLOP、11 s，其中 attention 占 70%。这一项无法靠 batch 摊薄，因为它本身就是 compute-bound 的（算术强度远高于 ridge point）。减少它只能靠减少算量本身：滑窗、交错、稀疏，或者 prefix caching（同一个 system prompt 的 KV 只算一次）。

**TTFT：单请求的算量下界。** 用户感知的首 token 延迟至少等于 prefill 时间。一个 128K 请求在单卡 8B 上的 TTFT 按峰值算至少 6.6 s、按 60% MFU 经验估约 11 s，要压到 1 s 以内需要 7–11 张卡并行处理同一个请求——这是序列并行的动机之一。

**chunked prefill 的必要性。** 如果调度器让一个 128K 请求一次性 prefill，它会独占 GPU 约 11 s，期间所有正在 decode 的请求全部停顿——它们的 token 间延迟从几十毫秒跳到 11 s。chunked prefill（Sarathi-Serve，Agrawal 等 2023；vLLM 与 SGLang 默认启用）把长 prefill 切成若干个 chunk（例如每次 2K–8K token），每个调度步里让一个 prefill chunk 与若干 decode 请求拼成一个 batch。decode 请求的 KV 读取是 memory-bound、prefill chunk 是 compute-bound，两者拼在一起恰好能同时用满带宽与算力。代价是长请求自己的 TTFT 略微变长，换来其他请求的延迟稳定。两种调度下同一段时间内 GPU 上发生的事对比如下（P = 128K 请求的 prefill，D = 已在 decode 的请求各出一个 token）：

```text title="不分块与 chunked prefill 的时间轴"
时间 →      0 s                                     11 s
不分块      [PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP][D][D][D][D]…
decode 请求  ←──────── 停顿 11 s，没有新 token ────────→ 恢复，每步几十 ms

chunked     [P2K+D][P2K+D][P2K+D][P2K+D] … [P2K+D][D][D]…
decode 请求  每步都出 token，步长略增（batch 里多了一个 compute-bound 的 chunk）
128K 请求    TTFT 从 11 s 变成 11 s + 若干 decode 步的开销
```

**序列并行 / context parallel 的动机。** 当单个请求的 KV cache（70B 的 40 GiB）或激活（128K 时每层的 hidden state 就是 $$131072 \times 8192 \times 2\,\text{B} = 2$$ GiB）放不进一张卡、或 TTFT 要求单请求必须由多卡并行时，就需要把**序列维度**切到多张卡上。TP 切的是 head 维度（第八篇），每张卡仍要处理全部 $$s$$ 个 token；序列并行切的是 token 维度，每张卡处理 $$s/P$$ 个 token，但 attention 需要所有 token 的 K、V——Ring Attention（Liu 等 2023）让 K、V 块在卡之间环形传递，每张卡对每个到达的 K、V 块做一次局部 attention 并用 online softmax 合并。它引入了新的通信项（每层传一遍全部 K、V），是长上下文训练与超长请求推理的标准手段。

最后一个跨章节的提醒：位置编码的选择会限制以上所有手段。滑窗与 sink 依赖"cache 内相对位置"的重新编号；YaRN 的温度要乘进 cos/sin 表；Llama 3.1 的分段缩放要在 kernel 之前的 inv_freq 计算里实现。推理引擎里 `rope_scaling` 字段解析错误是长上下文精度问题的常见根源之一——数学上只差一个分段规则，效果上是 32K 之后 perplexity 是否发散。

## 五、实践：llm_cost.py 上下文长度扫描

在第五、六、十二篇脚本的骨架上新增 `context_scan(cfg, gpu, ctxs)`，输出"上下文长度 → KV cache、prefill FLOPs、attention 占比"：

```python title="llm_cost.py：context_scan"
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    name: str
    hidden: int
    layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    vocab: int
    tie_embeddings: bool = False
    mla_rank: Optional[int] = None   # MLA 的 d_c；None 表示 MHA/GQA
    rope_dim: Optional[int] = None   # MLA 解耦 RoPE 的 d_h^R

LLAMA3_8B = ModelConfig("Llama-3-8B", 4096, 32, 32, 8, 128, 14336, 128256)
LLAMA3_70B = ModelConfig("Llama-3-70B", 8192, 80, 64, 8, 128, 28672, 128256)

@dataclass
class GPU:
    name: str
    hbm_bytes: float
    bandwidth: float   # bytes/s
    bf16_flops: float  # FLOP/s

H100 = GPU("H100 SXM", 80e9, 3.35e12, 989e12)
A100 = GPU("A100 80GB", 80e9, 2.0e12, 312e12)

GiB = 2 ** 30

def param_count(cfg: ModelConfig) -> dict:
    """dense 模型参数量（第五篇的公式，重给以便独立运行）。"""
    d, dh = cfg.hidden, cfg.head_dim
    attn = d * cfg.n_heads * dh + 2 * d * cfg.n_kv_heads * dh + cfg.n_heads * dh * d
    ffn = 3 * d * cfg.d_ff
    per_layer = attn + ffn + 2 * d
    emb = cfg.vocab * d
    head = 0 if cfg.tie_embeddings else cfg.vocab * d
    total = cfg.layers * per_layer + d + emb + head
    return {"per_layer": per_layer, "embedding": emb, "lm_head": head, "total": total}

def kv_bytes_per_token(cfg: ModelConfig, dtype_bytes: int = 2) -> int:
    """每 token 的 KV cache 字节数（第八篇），支持 MLA。"""
    if cfg.mla_rank is not None:
        return cfg.layers * (cfg.mla_rank + cfg.rope_dim) * dtype_bytes
    return 2 * cfg.layers * cfg.n_kv_heads * cfg.head_dim * dtype_bytes

def weight_flops_per_token(cfg: ModelConfig) -> float:
    """权重 GEMM 部分：每参数每 token 2 FLOPs，embedding 查表不计。"""
    p = param_count(cfg)
    return 2.0 * (p["total"] - p["embedding"])

def attn_flops_per_token(cfg: ModelConfig, ctx: int) -> float:
    """对上下文 ctx 做一次 attention（QK^T 与 PV）：每层 4·d·ctx。"""
    return 4.0 * cfg.n_heads * cfg.head_dim * ctx * cfg.layers

def forward_flops_per_token(cfg: ModelConfig, ctx: int) -> float:
    """decode 一个 token、上下文 ctx 时的前向 FLOPs（第六篇）。"""
    return weight_flops_per_token(cfg) + attn_flops_per_token(cfg, ctx)

def prefill_flops(cfg: ModelConfig, ctx: int, causal: bool = True) -> tuple:
    """一次 ctx 长度 prefill 的总 FLOPs，返回 (权重项, attention 项)。"""
    w = weight_flops_per_token(cfg) * ctx
    a = attn_flops_per_token(cfg, ctx) * ctx
    if causal:
        a /= 2
    return w, a

def context_scan(cfg: ModelConfig, gpu: GPU, ctxs, mfu: float = 0.6) -> None:
    """上下文长度 → KV cache、prefill FLOPs、attention 占比。"""
    kv = kv_bytes_per_token(cfg)
    w_tok = weight_flops_per_token(cfg)
    print(f"{cfg.name}: KV {kv/1024:.1f} KiB/token, weights {w_tok/1e9:.1f} GFLOPs/token")
    print(f"{'ctx':>8} {'KV cache':>10} {'prefill':>12} {'prefill@'+str(int(mfu*100))+'%':>12} "
          f"{'attn/token':>12} {'attn share':>10}")
    for s in ctxs:
        w, a = prefill_flops(cfg, s)
        t = (w + a) / (gpu.bf16_flops * mfu)
        a_tok = attn_flops_per_token(cfg, s)
        share = a_tok / (w_tok + a_tok)
        print(f"{s:>8d} {kv*s/GiB:>8.1f} G {(w+a)/1e15:>9.2f} PF {t:>10.2f} s "
              f"{a_tok/1e9:>9.1f} GF {share*100:>9.1f}%")

if __name__ == "__main__":
    for cfg in (LLAMA3_8B, LLAMA3_70B):
        context_scan(cfg, H100, [8192, 32768, 131072])
        print()
```

输出：

```text title="上下文长度扫描的输出"
Llama-3-8B: KV 128.0 KiB/token, weights 15.0 GFLOPs/token
     ctx   KV cache      prefill  prefill@60%   attn/token attn share
    8192      1.0 G      0.14 PF       0.24 s       4.3 GF      22.2%
   32768      4.0 G      0.77 PF       1.30 s      17.2 GF      53.4%
  131072     16.0 G      6.47 PF      10.90 s      68.7 GF      82.1%

Llama-3-70B: KV 320.0 KiB/token, weights 139.0 GFLOPs/token
     ctx   KV cache      prefill  prefill@60%   attn/token attn share
    8192      2.5 G      1.23 PF       2.07 s      21.5 GF      13.4%
   32768     10.0 G      5.96 PF      10.05 s      85.9 GF      38.2%
  131072     40.0 G     40.74 PF      68.65 s     343.6 GF      71.2%
```

8B 的三列与第二章一致：128K 时 16 GiB、6.5 PFLOP、11 s、attention 占 82%。70B 的权重项脚本给出 139 GFLOPs（$$2 \times (70.55 - 1.05)$$B），正文沿用总纲取整的 141，差异 1.5%，不影响任何结论；70B 的 prefill 时间是"单卡等效"，实际至少要 2 张 H100 才放得下权重。要加 DeepSeek-V3，传入 `mla_rank=512, rope_dim=64` 即可得到 68.6 KiB/token 与 128K 的 8.6 GiB；它的权重 FLOPs 项需要第十篇的 MoE 字段（激活 37B → 74 GFLOPs），attention 项按 128 头、q/k 192 维、v 128 维手算是每层 $$2 \times 128 \times (192 + 128) \cdot s = 81920\,s$$，61 层约 $$5.0 \times 10^6 \cdot s$$（未吸收的朴素形式）。

## 六、本文小结

上下文长度是对 Infra 成本最敏感的维度。本篇的结论：

1. 长上下文的成本有一个线性项（KV cache）、一个二次项（prefill 的 attention）和一个不能物化的中间量（$$s \times s$$ logits）。Llama-3-70B 128K 每 token attention 344 GFLOPs 超过权重 141 GFLOPs；Llama-3-8B 128K prefill 6.5 PFLOP、60% MFU 约 11 s。滑窗让单序列 KV 与每步 decode attention 限于 $$O(W)$$，整段 prefill attention 则由 $$O(s^2)$$ 变成 $$O(sW)$$ 但丢信息；全局/局部交错把系数变成 $$1/k$$；sink + 滑窗让 full attention 模型能流式运行；MLA 减 KV 的系数不改阶。
2. 对 Infra：KV cache 决定并发数，prefill 二次项决定 TTFT 下界，chunked prefill 是为了不让一个 128K 请求独占 GPU 11 s，序列并行是为了把单请求的 KV、激活与 TTFT 分到多卡。

本篇算出的数字汇总（theoretical，BF16，H100 60% MFU）：

```text title="本篇数字汇总：三个模型的长上下文成本"
                              Llama-3-8B        Llama-3-70B       DeepSeek-V3
权重 FLOPs/token              15.0 GFLOPs       141 GFLOPs        74 GFLOPs
KV bytes/token                128 KiB           320 KiB           68.6 KiB
KV cache   8K / 32K / 128K    1.0 / 4.0 / 16 GiB  2.5 / 10 / 40 GiB  0.54 / 2.1 / 8.6 GiB
attention FLOPs/token  8K     4.3 GFLOPs        21.5 GFLOPs       41 GFLOPs
                       32K    17.2 GFLOPs       85.9 GFLOPs       164 GFLOPs
                       128K   68.7 GFLOPs       344 GFLOPs        655 GFLOPs
attention 占比  8K / 32K / 128K  22% / 53% / 82%  13% / 38% / 71%   36% / 69% / 90%
prefill FLOPs   8K / 32K / 128K  0.14 / 0.77 / 6.5 PF  1.24 / 6.0 / 41 PF   —
prefill 时间（单卡等效）128K   约 11 s           约 69 s           —
attention = 权重 的交叉点      约 28.6K          约 53.8K          —
```

DeepSeek-V3 的 attention FLOPs 按未吸收的朴素形式（128 头、q/k 192 维、v 128 维）计算，吸收后的形式访存更少但 FLOPs 更高，第八篇有讨论；它的 prefill 总量需要第十篇 MoE 的激活参数量才能完整给出。

配套代码：[`transformer-and-llm/llm_cost_04_long_context.py`](https://github.com/arganzheng/ai-learning-labs/blob/main/transformer-and-llm/llm_cost_04_long_context.py)；RoPE 的 NumPy 实现与三种缩放的波长表在 [`rope_numpy.py`](https://github.com/arganzheng/ai-learning-labs/blob/main/transformer-and-llm/rope_numpy.py)。

## 七、自测


1. Llama-3-8B、128K 上下文：每 token 的 attention FLOPs 与权重 FLOPs 各多少？attention = 权重的交叉点在哪？

   <details markdown="1"><summary>答案</summary>

   attention 68.7 GFLOPs、权重 15.0 GFLOPs（82% 是 attention）；交叉点约 28.6K token。

   </details>

2. 一个 128K 的请求在单卡 H100 上 prefill 要多久？这个数字对推理系统意味着什么？

   <details markdown="1"><summary>答案</summary>

   6.5 PFLOP / (989 T × 60% MFU) ≈ 11 s（峰值下 6.6 s 才是物理下界，60% 是经验效率）；且这 11 秒里 GPU 被一个请求独占——chunked prefill 与序列并行就是为它设计的。

   </details>


[^q0]: 贵在三处：KV cache 随 $$s$$ 线性增长（Llama-3-8B 128K 一个请求 16 GiB，与权重一样大），决定并发数；prefill 的 attention 随 $$s^2$$ 增长（8B 128K 约 6.5 PFLOP，单卡 60% MFU 约 11 s，attention 占 70%），决定 TTFT；$$s \times s$$ 的 logits 128K 时单头 32 GiB，只能靠 FlashAttention 不物化。详见[第二章](#二长上下文的成本)。
[^q1]: 滑窗把单序列 KV 与每步 decode attention 从 $$O(s)$$ 改成 $$O(W)$$，整段 prefill attention 从 $$O(s^2)$$ 改成 $$O(sW)$$，但窗口外的信息只能逐层间接传递；全局/局部交错不改阶，只把系数变成全局层占比 $$1/k$$；sink + 滑窗让 full attention 训练的模型能稳定处理流式输入，代价同滑窗；块稀疏减算量但一般不减 KV，并引入不规则访存。MLA 改的是 KV 的系数而非阶，可与上面叠加。详见[第三章](#三缩短成本的结构手段)。
