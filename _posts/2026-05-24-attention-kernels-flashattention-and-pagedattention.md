---
layout: post
title: "GPU Kernel 工程（08）：Attention Kernel——FlashAttention 与 PagedAttention"
subtitle: "Attention Kernels: FlashAttention and PagedAttention from Derivation to Code"
tags: [CUDA, Triton, GPU, AI, AI-Infra]
catalog: true
---

> 本文是[《GPU Kernel 工程：从 CUDA 执行模型到 FlashAttention》](/gpu-kernel-engineering.html)系列的第 8 篇（共十篇）。上一篇：[Triton：块级编程与编译器的边界](/triton-block-level-programming.html)　下一篇：[量化与融合 kernel](/quantization-and-fused-kernels.html)

前七篇分别处理了 GPU 的硬件结构与 Roofline、CUDA 执行模型、访存合并、shared memory 与 reduction（其中包括 online softmax）、GEMM 的分块、Tensor Core 与 CUTLASS、Triton。这一篇是它们的汇合点：attention 同时包含两个 GEMM（$$QK^T$$ 与 $$PV$$）、一个逐行的 reduction（softmax），以及推理时特有的内存访问模式（分页的 KV cache）。它是 Transformer 推理里最重要、也最难写好的 kernel。

总纲给这一篇提的核心问题是：

> **一个序列长度 4k、head dim 128 的 attention，标准实现和 FlashAttention 分别读写多少 HBM？decode 阶段每生成一个 token 要读多少 KV cache，这决定了什么？**

全文遵循同一个方法论：先算理论上应该多快（字节数 / FLOPs / Roofline），再看实现，再解释差距。硬件基线仍然是 A100 SXM 80GB 的公开标称值：HBM2e 约 2.0 TB/s，BF16 Tensor Core 约 312 TFLOPS（dense），ridge point 约 156 FLOP/byte；Hopper 相关内容随文标注 H100（约 3.35 TB/s、约 989 TFLOPS、ridge 约 295）。没有 GPU 可供实测，所有性能数字要么是可推导的理论下界，要么用"通常能达到"的量级区间给出。

本篇不讨论 KV cache 的分页**管理**（block 的分配、换入换出、prefix 共享属于推理引擎调度层），只讨论分页之后 kernel 如何**访问**它。


## 一、先算账：标准 attention 读写多少 HBM

### 1. 记号与两个需要重述的结论

单个 head 的 attention 定义为：

$$
S = \frac{QK^T}{\sqrt{d}}, \quad P = \mathrm{softmax}(S), \quad O = PV
$$

其中 $$Q, K, V \in \mathbb{R}^{N \times d}$$，$$S, P \in \mathbb{R}^{N \times N}$$。$$N$$ 是序列长度，$$d$$ 是 head dim。本文的默认例子是 $$N = 4096$$、$$d = 128$$、BF16。

本篇会反复用到两个前面篇章的结论，这里自足地重新给出。

**Roofline。** 一个 kernel 的算术强度 $$I = \text{FLOPs} / \text{bytes}$$（对 HBM 而言），设备的 ridge point 是峰值算力除以峰值带宽。A100 BF16 的 ridge 是 $$312 / 2.0 \approx 156$$ FLOP/byte：$$I$$ 低于 156 的 kernel 受带宽限制（memory-bound），时间下界是字节数 / 2.0 TB/s；高于 156 的受算力限制（compute-bound），时间下界是 FLOPs / 312 TFLOPS。

**Online softmax。** 对一行 $$x_1, \dots, x_N$$，朴素 softmax 要三遍：先求最大值 $$m$$，再求 $$l = \sum_i e^{x_i - m}$$，最后写出 $$e^{x_i - m} / l$$。online softmax 把前两遍合成一遍，维护一对运行统计量 $$(m, l)$$，每来一个新元素（或一批新元素）就更新：

$$
m_{\text{new}} = \max(m, x_i), \qquad l_{\text{new}} = l \cdot e^{m - m_{\text{new}}} + e^{x_i - m_{\text{new}}}
$$

关键性质是：**已经累积好的 $$l$$ 可以在最大值变化时用一个因子 $$e^{m - m_{\text{new}}}$$ 修正**，不需要重新看一遍旧数据。FlashAttention 的全部数学，就是把这个修正因子同时施加到 $$O$$ 上。

### 2. 标准实现：物化 S 与 P

标准的 attention 实现（就是 PyTorch 里 `q @ k.transpose(-1, -2)`、`softmax`、`@ v` 三行代码所做的事）要把 $$S$$ 和 $$P$$ 完整地写进 HBM 再读回来。数一下字节：

- $$Q, K, V, O$$ 各是 $$4096 \times 128 \times 2$$ B = **1 MiB**；
- $$S = QK^T$$ 有 $$N^2 = 4096^2 = 16{,}777{,}216$$（约 16M）个元素，BF16 下是 **32 MiB**，若用 FP32 存放则是 64 MiB；
- 第一个 GEMM 写 $$S$$（32 MiB），softmax 读 $$S$$ 写 $$P$$（32 + 32 MiB），第二个 GEMM 读 $$P$$（32 MiB）：**与 $$N^2$$ 成正比的流量至少 $$4 \times 32 = 128$$ MiB**；
- 加上 $$Q, K, V, O$$ 的 4 MiB，共约 **132 MiB**。如果 softmax 之前还要加 mask、之后还有 dropout，每一步都再来一次 32 MiB 的读写。

FLOPs 呢？两个 GEMM 形状分别是 $$[N, d] \times [d, N]$$ 和 $$[N, N] \times [N, d]$$，各 $$2 N^2 d$$，合计：

$$
\text{FLOPs} = 4 N^2 d = 4 \times 16.78\text{M} \times 128 \approx 8.6\ \text{GFLOP}
$$

算术强度：

$$
I \approx \frac{8.6 \times 10^9}{132 \times 1.05 \times 10^6\ \text{B}} \approx 62\ \text{FLOP/byte}
$$

62 低于 A100 的 ridge 156：**标准 attention 在 A100 上是 memory-bound 的**。它的时间下界由 HBM 决定：$$138\ \text{MB} / 2.0\ \text{TB/s} \approx 69\ \mu s$$；而如果只看算力，$$8.6\ \text{G} / 312\ \text{T} \approx 28\ \mu s$$。中间那 40 µs 的差距，就是 $$S$$ 和 $$P$$ 在 HBM 上往返的代价。$$N$$ 越大情况越糟：$$N^2$$ 项随 $$N$$ 平方增长，而 $$Q, K, V, O$$ 只线性增长，序列到 32k 时 $$S$$ 一个矩阵就是 2 GiB。

这就是 FlashAttention 要解决的问题：**不是 FLOPs 太多，而是一个 $$N \times N$$ 的中间结果不该被写出去。**


## 二、FlashAttention：分块 + online softmax + 不物化 S

### 1. 分块

把 $$Q$$ 沿行切成大小为 $$B_r$$ 的块 $$Q_1, \dots, Q_{N/B_r}$$，把 $$K, V$$ 沿行切成大小为 $$B_c$$ 的块 $$K_1, \dots, K_{N/B_c}$$ 与 $$V_1, \dots, V_{N/B_c}$$。对于 $$Q$$ 的第 $$i$$ 块，它需要的输出是：

$$
O_i = \mathrm{softmax}\big( Q_i K^T \big) V = \frac{\sum_j e^{S_{ij} - m_i} V_j}{\sum_j \mathbf{1}^T e^{S_{ij} - m_i}}, \qquad S_{ij} = Q_i K_j^T \in \mathbb{R}^{B_r \times B_c}
$$

这里 $$m_i$$ 是每一行的全局最大值（一个 $$B_r$$ 维向量），$$e^{\cdot}$$ 逐元素。分子和分母都是**对 $$j$$ 的求和**——这正是 online softmax 能处理的形式。

### 2. 每个 tile 做什么

一个 thread block 拿到 $$Q_i$$（放进 shared memory 或直接进寄存器），维护三个运行量：行最大值 $$m_i \in \mathbb{R}^{B_r}$$（初值 $$-\infty$$）、行分母 $$l_i \in \mathbb{R}^{B_r}$$（初值 0）、未归一化的输出累加器 $$O_i \in \mathbb{R}^{B_r \times d}$$（初值 0）。然后依次遍历 $$K_j, V_j$$：

1. 把 $$K_j$$ 载入 shared memory，用 Tensor Core 算 $$S_{ij} = Q_i K_j^T$$，结果留在寄存器里（FP32）；
2. 求 $$S_{ij}$$ 每行的最大值 $$\tilde m_{ij}$$，更新 $$m_{\text{new}} = \max(m_i, \tilde m_{ij})$$；
3. 算 $$\tilde P_{ij} = e^{S_{ij} - m_{\text{new}}}$$（寄存器里，逐元素 exp），行和 $$\tilde l_{ij}$$；
4. **rescale 并累加**：

$$
l_i \leftarrow l_i \cdot e^{m_i - m_{\text{new}}} + \tilde l_{ij}, \qquad
O_i \leftarrow \mathrm{diag}\!\left(e^{m_i - m_{\text{new}}}\right) O_i + \tilde P_{ij} V_j
$$

5. $$m_i \leftarrow m_{\text{new}}$$。

遍历完所有 $$j$$ 后，$$O_i \leftarrow O_i / l_i$$（逐行除），写回 HBM。

对照第一节的 online softmax 递推，$$l_i$$ 的更新一模一样；$$O_i$$ 的更新只是把标量 $$e^{x_i - m_{\text{new}}}$$ 换成了矩阵 $$\tilde P_{ij}$$，把"加一个数"换成了"加一个 GEMM 结果 $$\tilde P_{ij} V_j$$"。旧的 $$O_i$$ 是以旧最大值 $$m_i$$ 为基准算出来的，最大值变大后需要乘上 $$e^{m_i - m_{\text{new}}} \le 1$$ 修正——这就是 `diag(...)`。

整个过程中，$$S_{ij}$$ 和 $$\tilde P_{ij}$$ 只在寄存器与 shared memory 中存在，从未写到 HBM。$$N \times N$$ 的矩阵被压成了 $$B_r \times B_c$$ 的 tile，用完即弃。

### 3. HBM 流量：O(N²d²/M)

现在数 FlashAttention 的字节数。$$Q$$ 读一次、$$O$$ 写一次，各 1 MiB。$$K, V$$ 呢？每个 $$Q$$ 块都要把 $$K, V$$ 从头到尾读一遍，所以 $$K, V$$ 各被读 $$N / B_r$$ 次：

$$
\text{bytes}_{\text{FA}} = \underbrace{2 N d}_{Q, O} + \underbrace{\frac{N}{B_r} \cdot 2 N d}_{K, V} \ \text{（元素数）}
$$

$$B_r$$ 由 shared memory 大小 $$M$$ 决定：一个 tile 要同时放下 $$Q_i, K_j, V_j$$ 和 $$S_{ij}$$（或它们中的一部分），Dao 等 2022 论文的选择是 $$B_c = M / (4d)$$，$$B_r = \min(B_c, d)$$，于是 $$N/B_r \approx 4Nd/M$$，K/V 的流量约为 $$4Nd/M \cdot 2Nd = 8N^2d^2/M$$，即 **$$\Theta(N^2 d^2 / M)$$**。对比标准实现的 $$\Theta(N^2)$$：多了一个 $$d^2/M$$ 的因子，$$d = 128$$、$$M = 100$$ KB 时 $$d^2 / M = 16384 / 102400 \approx 0.16$$，再除以 FP32 的 4 字节或 BF16 的 2 字节，就是数量级上的节省。

代入具体数字更直观。取 FlashAttention-2 常用的 $$B_r = 128$$：

- $$K, V$$ 各被读 $$4096 / 128 = 32$$ 次，每次 1 MiB，共 **64 MiB**；
- 加 $$Q$$ 读、$$O$$ 写 2 MiB，合计 **66 MiB**；
- 标准实现是 132 MiB，**约 2 倍**；$$B_r = 256$$ 时 K/V 只读 16 次，合计 34 MiB，约 4 倍。

"只有 2 倍"看起来不够惊艳，但注意两件事。第一，标准实现的 128 MiB 是 $$N^2$$ 项，FlashAttention 的 64 MiB 是 $$N^2 d / B_r$$ 项——$$B_r$$ 随硬件（shared memory、寄存器）增大而增大，$$N$$ 翻倍时前者翻 4 倍、后者也翻 4 倍，但常数因子相差 $$B_r / d \times$$（字节比）。第二，更实际的一点：**这里 K、V 总共只有 2 MiB，A100 有 40 MB L2**——32 次"重读"绝大部分命中 L2，真正到 HBM 的流量接近 4 MiB 的下界。标准实现的 32 MiB 中间矩阵则不可能待在 L2 里。因此对 4k 序列，FlashAttention 实际接近 memory 下界的 $$4\ \text{MiB} / 2.0\ \text{TB/s} \approx 2\ \mu s$$，远低于计算下界 28 µs：**它把 attention 从 memory-bound 推进了 compute-bound 区域**。这才是 IO-aware 的真正收益。

算一下算术强度确认：以 66 MiB 计，$$I = 8.6\ \text{G} / 69\ \text{MB} \approx 125$$；以实际 HBM 流量 4 MiB 计，$$I \approx 2000$$。无论哪种口径都在 156 附近或远高于它。

顺便说明一下循环顺序对这笔账的影响。上面按 "外层遍历 $$Q$$ 块、内层遍历 $$K, V$$ 块" 来数，这是 FlashAttention-2 的顺序；FlashAttention-1 论文里是反过来的——外层遍历 $$K_j, V_j$$（每块只从 HBM 载入一次），内层遍历 $$Q_i$$，于是 $$Q$$ 以及运行量 $$O_i, m_i, l_i$$ 要被反复从 HBM 读进来、更新、写回去，共 $$N / B_c$$ 遍。两种顺序的渐近流量都是 $$\Theta(N^2 d^2 / M)$$，但 v2 的顺序让 $$O_i$$ 全程留在寄存器，少了一份 $$N / B_c$$ 倍的 $$O$$ 读写，也少了它带来的同步——这是下一节 FA2 第一项改动的动机。

### 4. 一个重要修正：FA 不省 FLOPs，甚至略多

FlashAttention 的 FLOPs 比标准实现**略多**：每个 tile 多了 $$B_r \times d$$ 次乘法做 rescale（$$\mathrm{diag}(e^{m - m_{\text{new}}}) O_i$$），以及最后的 $$O_i / l_i$$。这些相对于 $$4N^2 d$$ 的 GEMM 只是 $$O(N^2 d / B_c)$$ 的小项，但确实存在。它的收益全部来自 IO，不来自算法上少算了什么。

还有一件容易被忽略的事：$$N^2$$ 次 exp。Tensor Core 极快而 SFU（special function unit，做 exp、rsqrt 等）很慢。A100 每 SM 每周期能做 1024 次 dense BF16 FMA（Tensor Core），但 SFU 只有 16 次/周期。每个 $$S$$ 元素对应 $$4d = 512$$ FLOP = 256 次 FMA，用 Tensor Core 需要 $$256/1024 = 0.25$$ 周期；1 次 exp 需要 $$1/16 \approx 0.06$$ 周期——**exp 占到了 matmul 时间的 1/4**。H100 上 Tensor Core 快了 3 倍多而 SFU 没有同比例提升（Shah 等 2024 给的数字是约 989 TFLOPS matmul 对约 3.9 TFLOPS 特殊函数），这个比例进一步恶化到接近 1:2。所以从 FlashAttention-2 起，"减少非矩阵运算"和"让 exp 与 matmul 重叠"成了主要优化方向，而不是继续省 HBM 流量。


## 三、FlashAttention-2 与 FlashAttention-3

### 1. FlashAttention-2（Dao 2023）：并行化与 warp 分工

FlashAttention-1 的外层循环遍历 $$K, V$$ 块，内层遍历 $$Q$$ 块，每个 thread block 负责一个 (batch, head)。这有两个问题：$$O_i, m_i, l_i$$ 要在 HBM 上反复读写（因为内层换 $$Q$$ 块）；并行度只有 batch × heads，长序列、小 batch 时（推理常态）填不满 108 个 SM。

FA2 做了三处改动：

**外循环改为 Q 块，并按 (batch, head, Q block) 三维并行。** 每个 thread block 负责一个 $$Q_i$$，遍历全部 $$K_j, V_j$$。$$O_i, m_i, l_i$$ 全程留在寄存器，只在最后写一次。grid 大小变为 $$\text{batch} \times \text{heads} \times N / B_r$$：单序列、32 head、4k 长度、$$B_r = 128$$ 时有 $$32 \times 32 = 1024$$ 个 block，足够填满 SM。这就是上面第二节按 $$Q_i$$ 讨论的版本。

**warp 分工从 "split K" 改为 "split Q"。** 一个 block 里通常有 4 个 warp。FA1 让 4 个 warp 各持有 $$K_j$$ 的一段（沿 $$B_c$$ 切），每个 warp 算出 $$S_{ij}$$ 的一个列切片；但 softmax 是按**行**归约的，行最大值和行和需要跨 4 个 warp 通过 shared memory 同步——每个 tile 都要 `__syncthreads()` 加 shared 读写。FA2 让 4 个 warp 各持有 $$Q_i$$ 的若干行（$$B_r = 128$$ 时每 warp 32 行，或用 16 行的 mma 形状），每个 warp 独立算出自己那些行对完整 $$K_j$$ 的 $$S$$、独立做行 softmax、独立累加 $$O$$——**warp 之间零通信**。代价是 $$K_j, V_j$$ 要被 4 个 warp 各读一遍（从 shared memory，不是 HBM），但这比每 tile 一次跨 warp 归约便宜得多。

**延迟 rescale，减少非 matmul 运算。** 上一节的算法每个 tile 都对 $$O_i$$ 乘 $$\mathrm{diag}(e^{m - m_{\text{new}}})$$，FA1 还在每个 tile 里除 $$l$$。FA2 只做前者（这是不可省的），除法只在最后做一次；同时用 exp2 替代 exp、把 $$1/\sqrt d$$ 和 $$\log_2 e$$ 预先合并成一个 scale 乘进 $$S$$，让每个元素只剩一次乘法和一次 `ex2` 指令。

FA2 在 A100 上通常能达到 BF16 峰值的 50%–73%（论文数字），即前向大约 160–230 TFLOPS 量级；这已经是 GEMM 级别的效率。

### 2. FlashAttention-3（Shah 等 2024，Hopper only）

FA3 只针对 sm_90。它把 Hopper 的三项硬件特性（wgmma、TMA、更大的 shared memory）用在 attention 上，并用 warp specialization 解决 exp 慢的问题：

- **wgmma + TMA**：$$QK^T$$ 与 $$PV$$ 用 warpgroup 级的异步 `wgmma`（形状 `m64nNk16`），$$K_j, V_j$$ 用 TMA 一条指令搬进 shared memory，地址计算与搬运不再占用线程；
- **producer/consumer warp specialization**：一个 producer warpgroup 只负责发 TMA 加载、维护 pipeline 的 barrier；两个 consumer warpgroup 做计算。生产者用 `setmaxnreg` 让出寄存器给消费者；
- **两个 consumer warpgroup 的 ping-pong**：当 warpgroup 1 在做 $$S = QK^T$$ 的 GEMM 时，warpgroup 2 在做上一 tile 的 softmax（exp、max、sum）；然后互换。Tensor Core 与 SFU 同时忙碌，把上一节说的 "exp 占 matmul 时间的一半" 隐藏掉；
- **块内 GEMM–softmax 流水**：单个 warpgroup 内也把第 $$j$$ 块的 softmax 与第 $$j+1$$ 块的 $$QK^T$$ 重叠（软件流水，需要多一套 $$S$$ 的寄存器）；
- **FP8**：$$Q, K, V$$ 用 FP8 e4m3 喂给 FP8 wgmma（H100 上约 1979 TFLOPS 标称）；为控制精度做块级量化（per-block scale）以及"非相干处理"（用随机正交矩阵把离群值摊平）；FP8 wgmma 要求 $$V$$ 是 k-major 布局，需要在 shared memory 里做一次转置/布局重排。

FA3 论文报告 H100 上 BF16 前向达到 ~740 TFLOPS（约 75% 峰值），FP8 接近 1.2 PFLOPS。sm_80 上用不了这些，A100 上的最优解仍然是 FA2。

### 3. 反向传播的 recompute

训练时反向需要 $$P$$ 来算 $$dV = P^T dO$$ 和 $$dS$$。标准实现把 $$N^2$$ 的 $$P$$ 存下来（4k 序列、32 head、batch 8 就是 8 GiB）。FlashAttention 只保存 $$O$$ 与每行的 logsumexp：

$$
L_i = m_i + \log l_i
$$

反向时按同样的分块重新算 $$S_{ij} = Q_i K_j^T$$，然后 $$P_{ij} = e^{S_{ij} - L_i}$$——因为 $$L_i$$ 已知，一步到位，不需要再做 online 归约。代价是反向多算一次 $$QK^T$$（$$2N^2 d$$ FLOPs），换来 $$N^2$$ 的显存节省。这是典型的"用 FLOPs 换字节"：反向从 5 个 GEMM 变成 5 个 GEMM 加一次重算，总 FLOPs 从 $$2.5\times$$ 前向变为约 $$2.5\times$$ 前向再加 0.5，但全部 compute-bound，实际时间反而比读写 $$P$$ 更短。前向输出的 `lse` 就是为此保留的接口，后面 Triton 版会照此存下 $$L_i$$。

### 4. FlashAttention 源码的结构

FlashAttention-2 的 CUDA 源码大致位于仓库的 `csrc/flash_attn/src/` 目录。核心文件是 `flash_fwd_kernel.h`，其中 `compute_attn_1rowblock` 这个 device 函数就是"一个 thread block 处理一个 $$Q$$ 行块"的实现（本机没有源码，以下凭对代码结构的了解描述，不给行号）：

1. 用 `binfo`（`BlockInfo`）从 `cu_seqlens_q/k` 取得当前序列的起止，算出这个 $$Q$$ 块需要遍历的 KV 块范围 `n_block_min..n_block_max`（因果或 sliding window 时缩小范围）；
2. 用 CuTe 的 `Tensor` 与 `Layout` 把 gmem 上的 $$Q, K, V$$ 切成 tile，`cp.async` 搬到 smem 的双缓冲；
3. 主循环分两段：先处理需要 mask 的块（对角块，`n_masking_steps` 次），再处理不需要 mask 的块；两段共用同一套 "gemm → softmax rescale → gemm" 逻辑，只是 mask 那段多了 `apply_mask_causal`；
4. `flash::gemm` 用 `mma.sync m16n8k16` 算 $$S$$，`Softmax` 结构体（在 `softmax.h`）里的 `softmax_rescale_o` 完成 $$m, l, O$$ 的更新——正是第二节步骤 2–4；
5. $$P$$ 从 FP32 累加器转成 BF16 后**直接作为下一个 mma 的 A 操作数**（fragment 布局是兼容的，不需要经过 shared memory）；
6. 收尾：$$O / l$$，写 $$O$$ 与 $$L$$。

后面第七节的 CUDA 骨架就是这个结构的浓缩。


## 四、推理的两种形态：prefill 与 decode

### 1. prefill：Q 很长，GEMM 主导

prefill 处理 prompt 的全部 token，$$Q$$ 有 $$N_q$$ 行、$$K, V$$ 有 $$N_k = N_q$$ 行（或加上已缓存的前缀）。这就是训练前向的形状，$$4N_q N_k d$$ 的 FLOPs 配 $$O(N d)$$ 级的 HBM 流量，compute-bound。FlashAttention-2/3 是为这个场景设计的，效率与 GEMM 同级。

现代推理引擎还有一种混合形态：chunked prefill 把一个长 prompt 切成若干 chunk，每步只 prefill 一个 chunk，并与其他请求的 decode token 拼进同一个 batch。此时一个 batch 里既有 $$N_q$$ 为几百上千的请求，也有 $$N_q = 1$$ 的请求，它们的 $$K, V$$ 都是 "已缓存的前缀 + 本步新 token"。这正是 `cu_seqlens`（第六节）与 "统一 prefill/decode 的 kernel"（第七节）存在的原因：kernel 不能假设 $$Q$$ 是长是短，只能按每个序列各自的 `query_len` 与 `context_len` 工作。

### 2. decode：Q 只有一行，每 token 读全部 KV

decode 每步只生成一个 token：$$Q$$ 是 $$1 \times d$$（每个 head），但要对整个上下文的 $$K, V$$ 做 attention。数一下 Llama-3-8B（32 层、8 个 KV head、$$d_{\text{head}} = 128$$、BF16）每个 token 的 KV cache：

$$
32 \times 8 \times 128 \times 2\ (K, V) \times 2\ \text{B} = 131072\ \text{B} = 128\ \text{KiB}
$$

上下文长度 $$s$$ 时，每生成一个 token 要读 $$128\ \text{KiB} \times s$$：$$s = 4096$$ 是 512 MiB，$$s = 8192$$ 是 1 GiB。

它的算术强度低到什么程度？对一个 KV head、一个 token 位置，读 $$K$$ 和 $$V$$ 各 $$d$$ 个元素（BF16 共 $$4d$$ 字节），做 $$QK$$ 的 $$2d$$ FLOPs 加 $$PV$$ 的 $$2d$$ FLOPs。若 $$g$$ 个 query head 共享这一份 KV（GQA，Llama-3-8B 是 32/8 = 4），FLOPs 乘 $$g$$：

$$
I_{\text{decode}} = \frac{4 d g}{4 d} = g\ \text{FLOP/byte（BF16 KV）}
$$

KV cache 用 FP8 存时分母减半，$$I \approx 2g$$。所以 decode attention 的算术强度约为 $$1\sim 2 \cdot g$$ FLOP/byte，$$g = 4$$ 时是 4–8，与 A100 的 ridge 156 相差一个半数量级以上——**彻底 memory-bound**，无论 kernel 写得多好，时间都等于 KV 字节数 / 带宽。

这回答了核心问题的第二半。一步 decode 的时间下界是：

$$
t_{\text{decode}} \ge \frac{\text{权重字节}}{\text{BW}} + \frac{\text{batch} \times s \times 128\ \text{KiB}}{\text{BW}}
$$

Llama-3-8B 的 BF16 权重约 16 GB，在 A100 上读一遍约 8 ms。KV 那一项随 batch 与上下文线性增长：batch 1、$$s = 4096$$ 时 512 MiB 只占权重的 3%；但当 $$\text{batch} \times s$$ 超过 $$16\ \text{GB} / 128\ \text{KiB} \approx 131\text{k}$$ 个 token（例如 batch 32、$$s = 4096$$）时，**KV 的读取时间超过权重的读取时间**。这就是为什么推理引擎在长上下文、大 batch 时会把 KV cache 量化到 FP8、为什么 GQA/MQA 直接缩小 $$g$$ 倍的 KV、以及为什么 decode attention kernel 的全部目标就是"以接近峰值带宽的速度把 KV 读一遍"。

### 3. GQA 在 kernel 层的含义

GQA 让 $$g$$ 个 query head 共享一个 KV head，在参数与 KV cache 层面的节省是显然的。kernel 层的问题是：**怎么保证 KV 真的只从 HBM 读一次，而不是 $$g$$ 个 query head 各读一次？**

做法是把共享同一 KV head 的 $$g$$ 个 query head 放进同一个 thread block（或同一个 tile）。对 decode，$$g$$ 个 $$1 \times d$$ 的 query 恰好可以 pack 成一个 $$g \times d$$ 的矩阵，作为 mma 的 M 维——原本 $$M = 1$$ 的 GEMV 变成 $$M = g$$ 的 GEMM，$$K_j$$ 载入一次被 $$g$$ 行复用，算术强度乘 $$g$$。vLLM 的 Triton unified attention 正是这么做的：`BLOCK_M = 16`（或 `num_queries_per_kv` 向上取到 2 的幂）、`BLOCK_Q = BLOCK_M // num_queries_per_kv`——一个 tile 的 16 行由 `BLOCK_Q` 个 token 位置 × $$g$$ 个 query head 拼成。对 prefill，FA2 里 GQA 的处理是 grid 仍按 query head 展开、kernel 内部用 `h_k = h_q / g` 映射到 KV head，K/V 的复用交给 L2；FA3 与 FlashInfer 则会显式把同一 KV head 的 query head 打包进同一 tile。


## 五、PagedAttention：分页 KV cache 的 kernel 侧

### 1. block table 间接寻址

Kwon 等 2023 的 PagedAttention 把每个序列的 KV cache 切成固定大小的 block（vLLM 默认每 block 16 个 token，也常用 32），按需分配，物理上不连续；一张 block table 记录 "序列 $$s$$ 的第 $$b$$ 个逻辑块在第几个物理块"。kernel 读第 $$t$$ 个 token 的 K 时：

$$
\text{physical} = \text{block\_table}[s][\lfloor t / B \rfloor], \qquad \text{addr} = \text{physical} \times \text{block\_stride} + \text{head} \times \text{head\_stride} + (t \bmod B) \times \ldots
$$

即多一次查表（一个 int32 读，几乎总在 L1/L2 里），然后按物理块地址读连续的 $$B$$ 个 token。对 kernel 而言，KV 不再是一个 $$[N, d]$$ 的连续矩阵，而是 $$N/B$$ 个通过指针数组间接访问的 $$[B, d]$$ 小矩阵——这也是为什么 block size 不能太小：每个 block 内的读是连续合并的，块间是随机跳转。

block size 的取舍可以用字节数说清楚。一个 block、一个 KV head 的 K 是 $$B \times d \times 2$$ 字节，$$B = 16$$、$$d = 128$$ 时是 4 KiB（V 同样 4 KiB），正好是 32 条 128 字节 cache line——每次跳转之后至少有 4 KiB 的连续读，DRAM 的页开销与 TLB 开销被摊薄到可以忽略。$$B$$ 再大一些（32、64）对 kernel 更友好（tile 边界与 block 边界更容易对齐，查表次数更少），但调度层的内部碎片会增加：每个序列最后一个块平均浪费 $$B/2$$ 个 token 的空间。16 与 32 是两边都能接受的折中；FlashAttention 与 FlashInfer 的 paged 接口对 block size 的要求也大致在这个范围（FA 要求是 16 的倍数，与其 tile 的 k16 对齐有关）。

### 2. vLLM 的 CUDA kernel：paged_attention_v1

vLLM 的 `csrc/attention/attention_kernels.cuh` 里的 `paged_attention_kernel`（改写自 FasterTransformer 的 decoder masked MHA）是最早的 PagedAttention 实现。它是一个 decode 专用 kernel（$$Q$$ 一行），不用 Tensor Core。以 v0.20.0 为准，模板参数与 grid：

```cpp
// csrc/attention/attention_kernels.cuh (vLLM v0.20.0)
// Grid: (num_heads, num_seqs, max_num_partitions).
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE,
          bool IS_BLOCK_SPARSE,
          int PARTITION_SIZE = 0>  // Zero means no partitioning.
__device__ void paged_attention_kernel(
    float* __restrict__ exp_sums,    // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,  // [num_seqs, num_heads, max_num_partitions]
    scalar_t* __restrict__ out,      // [num_seqs, num_heads, max_num_partitions, head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads, head_size, block_size]
    const int num_kv_heads, const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq, ...)
```

`paged_attention_v1.cu` 的 launcher 用 `dim3 grid(num_heads, num_seqs, 1)`、`NUM_THREADS = 128`：**一个 thread block 处理一个 (seq, head)**，4 个 warp。kernel 内部分三段。

**第一段：QK。** 线程被组织成 thread group，`THREAD_GROUP_SIZE = max(32 / BLOCK_SIZE, 1)`：BLOCK_SIZE = 16 时每组 2 个线程，一个 warp 的 16 个 group 正好对应一个 block 的 16 个 token——**每个 warp 一次处理一个 KV block**，4 个 warp 轮转（`block_idx += NUM_WARPS`）。每个 group 负责一个 token 的完整 $$d = 128$$ 维点积，组内线程各拿一半维度，每次读 16 字节（`VEC_SIZE = 16 / (THREAD_GROUP_SIZE × sizeof(scalar_t))` = 4 个 BF16）：

```cpp
// csrc/attention/attention_kernels.cuh (vLLM v0.20.0), QK 段节选
const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
     block_idx += NUM_WARPS) {
  const int64_t physical_block_number =
      static_cast<int64_t>(block_table[block_idx]);
  for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
    const int physical_block_offset = (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    K_vec k_vecs[NUM_VECS_PER_THREAD];
#pragma unroll
    for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
      const cache_t* k_ptr = k_cache + physical_block_number * kv_block_stride +
                             kv_head_idx * kv_head_stride + physical_block_offset * x;
      const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
      const int offset1 = (vec_idx * VEC_SIZE) / x;
      const int offset2 = (vec_idx * VEC_SIZE) % x;
      k_vecs[j] = *reinterpret_cast<const K_vec*>(k_ptr + offset1 * BLOCK_SIZE * x + offset2);
    }
    float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs[thread_group_offset], k_vecs);
    if (thread_group_offset == 0) {
      const bool mask = token_idx >= seq_len;
      logits[token_idx - start_token_idx] = mask ? 0.f : qk;
      qk_max = mask ? qk_max : fmaxf(qk_max, qk);
    }
  }
}
```

注意 K cache 的布局 `[num_blocks, num_kv_heads, head_size/x, block_size, x]`，其中 `x = 16 / sizeof(cache_t)`（BF16 时 8）。它把 head dim 拆成 `head_size/x` 组，每组 `x` 个元素，**同一 token 的 `x` 个连续元素放在一起**（最内维），而 `block_size` 个 token 在倒数第二维。这样一个 thread group 读一个 token 的第 `offset1` 组时，读到的是 16 字节连续数据；同一个 warp 的 16 个 group 读 16 个 token 的同一组，地址是连续的 16 × 16 = 256 字节——正好是 2 条 cache line，完美合并。这是 KV cache 布局要服从 kernel 访存模式的直接例子，也是为什么这个布局被叫作 "x-major"。

`Qk_dot::dot` 在组内做 shuffle 归约得到完整点积（`attention_utils.cuh`）。每个 group 的 0 号线程把 logit 写进 shared memory 的 `logits[]`（FP32），并更新自己的 `qk_max`。

**第二段：softmax。** 先 warp 内 shuffle 归约 `qk_max`，再通过 `red_smem` 跨 warp 归约得到全序列最大值；然后所有线程 strided 遍历 `logits[]`，原地替换为 $$e^{x - m}$$ 并累加 `exp_sum`（`block_sum`），再乘 `inv_sum` 归一化。这是标准的 shared memory 两级 reduction，不是 online softmax——因为一个 block 拿到了整个序列的 logits（最长 `padded_max_seq_len × 4` 字节的 shared memory，这也限制了 v1 能处理的序列长度）。

**第三段：PV。** 每个 warp 再次遍历自己负责的 KV block，读 V（布局 `[num_blocks, num_kv_heads, head_size, block_size]`，token 在最内维，每线程读 16 字节即 8 个 token 的同一个 head dim），乘上 `logits[]` 里对应的概率，累加到寄存器 `accs[NUM_ROWS_PER_THREAD]`（每线程负责 head dim 的 8 行）。最后 warp 内 shuffle 归约、跨 warp 通过 shared memory 二分归约、warp 0 写出。

### 3. v2：partition 就是 split-KV

v1 的并行度是 `num_seqs × num_heads` 个 block。decode 时 batch 8、32 head 只有 256 个 block，每个 128 线程；108 个 SM 每个能驻留 2048 线程（16 个这样的 block），也就是**只用到了硬件并发能力的 15% 左右**，带宽拉不满。而 memory-bound kernel 的唯一目标就是拉满带宽。

`paged_attention_v2.cu` 的解法是 `PARTITION_SIZE = 512`：把序列按 512 个 token 切成 `max_num_partitions` 段，grid 变成 `(num_heads, num_seqs, max_num_partitions)`，每个 block 只处理自己那 512 个 token（32 个 KV block），各自算出局部的 $$(m, l, O)$$ 写进 `max_logits`、`exp_sums`、`tmp_out`。然后 `paged_attention_v2_reduce_kernel` 合并：

```cpp
// csrc/attention/attention_kernels.cuh (vLLM v0.20.0), reduce kernel 节选
// 全局最大值 max_logit 已由两级归约得到
float global_exp_sum = 0.0f;
for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
  float l = shared_max_logits[i];
  float rescaled_exp_sum = exp_sums_ptr[i] * expf(l - max_logit);
  global_exp_sum += rescaled_exp_sum;
  shared_exp_sums[i] = rescaled_exp_sum;
}
__syncthreads();
global_exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], global_exp_sum);
const float inv_global_exp_sum = __fdividef(1.0f, global_exp_sum + 1e-6f);
// Aggregate tmp_out to out.
for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
  float acc = 0.0f;
  for (int j = 0; j < num_partitions; ++j) {
    acc += to_float(tmp_out_ptr[j * HEAD_SIZE + i]) * shared_exp_sums[j] * inv_global_exp_sum;
  }
  from_float(out_ptr[i], acc);
}
```

这就是 online softmax 的**合并公式**：两段各有 $$(m_a, l_a, O_a)$$、$$(m_b, l_b, O_b)$$（$$O$$ 已归一化或未归一化都可以推），合并后 $$m = \max(m_a, m_b)$$，$$l = l_a e^{m_a - m} + l_b e^{m_b - m}$$，$$O = (O_a l_a e^{m_a - m} + O_b l_b e^{m_b - m}) / l$$。它与第二节 tile 间的 rescale 是同一个公式，只是两个操作数换成了两个并行 block 的结果。

这个思路在 FlashAttention 仓库里叫 **Flash-decoding**（split-KV）：decode 时 $$\text{batch} \times \text{heads}$$ 不够填满 SM，就把 KV 序列切成 $$S$$ 段，$$S$$ 倍的 block 并行，最后一个小 kernel 合并。FA2/FA3 的 `num_splits` 参数、FlashInfer 的 split-KV plan、vLLM Triton kernel 的 "3D" 模式（`NUM_SEGMENTS_PER_SEQ`）都是同一个东西。vLLM 还有一个通用的 `merge_attn_states` kernel（`csrc/attention/merge_attn_states.cu`）专门做两段结果的合并，接口是 `(prefix_output, prefix_lse, suffix_output, suffix_lse)`——以 logsumexp 形式传 $$(m, l)$$，用 `p_scale = e^{lse_p - lse_{max}} / (e^{lse_p - lse_{max}} + e^{lse_s - lse_{max}})` 加权。它被用在 cascade attention（多个请求共享前缀时，前缀部分算一次、后缀各算一次再 merge）和 chunked prefill 中。

顺带一句版本事实：在 v0.20.0 的 CUDA 路径上，V1 引擎的 attention 已经交给 FlashAttention、FlashInfer 或 Triton 后端；这套 `paged_attention_v1/v2` 主要还在 ROCm 等路径（`rocm_aiter_fa.py`）里使用。但它是理解"分页 KV 如何被 kernel 访问"最直接的教材。


## 六、变长 batch、因果掩码与 sliding window

### 1. cu_seqlens：无 padding 的 packed 布局

推理时一个 batch 里各请求长度不同。如果按最长的 pad 成 `[B, N_max, d]`，短序列的 padding 位置白白浪费计算和显存。FlashAttention 的 `varlen` 接口改用 **packed 布局**：所有序列的 token 首尾相接排成 `[total_tokens, H, d]`，另给一个前缀和数组：

```text
seq_lens   = [3, 5, 2]
cu_seqlens = [0, 3, 8, 10]      # 长度 B+1，cu_seqlens[i] 是第 i 个序列的起点
```

kernel 里一个 block 拿到序列编号 $$b$$ 后，用 `cu_seqlens[b]` 和 `cu_seqlens[b+1]` 找到自己 $$Q$$ 的起止行、用 `cu_seqlens_k` 找到 $$K, V$$ 的起止；grid 的 Q-block 维要覆盖 $$\sum_b \lceil N_b / B_r \rceil$$。vLLM 的 Triton kernel 用 `query_start_len_ptr` 上的二分查找（`find_seq_idx`）把一个一维的 `program_id(0)` 映射回 `(seq_idx, q_block_local_idx)`；FA2 的 `flash_attn_varlen_func` 则把 grid 按 `max_seqlen_q / B_r × batch` 开、越界的 block 直接返回。vLLM 调用 FA 时，`k`/`v` 传的是整个 paged KV cache，KV 侧的长度用 `seqused_k` 而非 `cu_seqlens_k` 给出，再配 `block_table` 做分页寻址（`vllm/v1/attention/backends/flash_attn.py`）——这是 vLLM 维护的 vllm-flash-attn fork 加进 FA 的能力。

### 2. 因果掩码在分块中的处理

因果 attention 里第 $$i$$ 行只看 $$j \le i$$ 列。对分块实现，这意味着 $$Q$$ 块 $$i$$ 只需要遍历 $$K$$ 块 $$j \le i$$（上三角的块整个跳过），**FLOPs 与 KV 读取量都省一半**；只有对角块（$$j = i$$，或 $$B_r \ne B_c$$ 时的几个跨对角块）内部才需要逐元素 mask。所以实现上主循环分成两段：先跑 off-diagonal 的块（无 mask，纯 GEMM + softmax），再跑对角块（有 mask）。把 mask 分支从主循环里拿掉，让编译器为绝大多数迭代生成无分支的代码，这是 FA2 和 Triton tutorial 都采用的结构。

在 vLLM 这类 "context + 新 query" 的形态下，第 $$q$$ 个新 token 的绝对位置是 `context_len + q`，它能看到的 key 范围是 `[0, context_len + q]`；一个 $$Q$$ 块能看到的最长前缀是块内最后一行的位置加一，kernel 据此算出要遍历的 tile 数（`compute_tile_loop_bounds` 里的 `max_seq_prefix_len`）。

### 3. sliding window

sliding window（Mistral 等模型用，窗口 $$W$$）让第 $$i$$ 行只看 $$[i - W + 1, i]$$。分块上就是**同时裁掉左边的块**：只遍历与 $$[i_{\min} - W + 1, i_{\max}]$$ 相交的 KV 块（$$i_{\min}, i_{\max}$$ 是本 $$Q$$ 块的行范围），块内再逐元素 mask 掉 $$j < i - W + 1$$ 的部分。vLLM 的 `compute_tile_loop_bounds` 里 `tile_start = max(0, first_allowed_key // TILE_SIZE)`、`tile_end = (last_allowed_key // TILE_SIZE) + 1` 就是这个裁剪；FA2 的 `window_size=(left, right)` 参数在 `n_block_min` 上做同样的事。上下文远长于 $$W$$ 时，每个 $$Q$$ 块只做 $$W / B_c$$ 次 tile 迭代，attention 成本从 $$O(N^2)$$ 降为 $$O(NW)$$。

一个细节：window 裁剪后，一行的某个 tile 可能整行被 mask（全 $$-\infty$$），此时 $$m_{\text{new}}$$ 若仍是 $$-\infty$$ 会让 $$e^{S - m_{\text{new}}}$$ 变成 NaN。vLLM 的 `softmax_step` 里 `m_j = tl.where(m_j > -inf, m_j, 0.0)` 就是防这个。后面自己写的 kernel通过保证"每行访问的第一个 tile 至少有一个合法列"来规避，但在窗口场景下必须显式处理。


## 七、Triton 版 FlashAttention 与 CUDA 版的结构对照

### 1. Triton tutorial 06 的结构

Triton 官方 tutorial 06（fused attention）是最容易读懂的 FlashAttention-2 实现，结构如下：

- `_attn_fwd`：一个 program 处理一个 (batch, head, $$Q$$ 块)，载入 $$q$$，初始化 `m_i = -inf`、`l_i`、`acc`；因果模式下调两次 `_attn_fwd_inner`——`STAGE=1` 跑 off-diagonal 块（无 mask），`STAGE=2` 跑对角块（有 mask）；最后 `acc / l_i`，写 `O` 与 `lse`；
- `_attn_fwd_inner`：`for start_n in range(lo, hi, BLOCK_N)`，每步 `tl.dot(q, k)` → mask → `m_ij = max(m_i, max(qk))` → `p = exp2(qk - m_ij)` → `alpha = exp2(m_i - m_ij)` → `l_i = l_i * alpha + sum(p)` → `acc = acc * alpha[:, None]` → `acc = tl.dot(p.to(bf16), v, acc)` → `m_i = m_ij`；
- 两个数值技巧：用 `tl.math.exp2` 而非 `exp`（直接映射到 `ex2.approx` 指令，省一次乘法），并把 $$\log_2 e$$ 提前乘进 `qk_scale = sm_scale * 1.44269504`，这样 $$e^{x \cdot s} = 2^{x \cdot s \cdot \log_2 e}$$，全程在 log2 域工作；`lse` 输出时再换回自然对数。

它和 CUDA 版的对应关系：Triton 的 `tl.dot(q, k)` 对应 CuTe 的 `gemm(tiled_mma, tSrQ, tSrK, acc_s)`；`tl.max(qk, 1)` 对应 `quad_allreduce_` 那种 warp 内按行的 shuffle 归约；`acc * alpha[:, None]` 对应 `softmax_rescale_o`；`p.to(bf16)` 后直接 `tl.dot(p, v)` 对应 CUDA 里把 $$S$$ 的累加器 fragment 重解释为下一个 mma 的 A fragment。Triton 隐藏了 shared memory 的搬运、双缓冲、`cp.async`、fragment 布局与 bank conflict，代价是 FA3 那种 producer/consumer warp specialization 与两个 warpgroup 的 ping-pong 只能依赖编译器自动调度（Triton 3.x 在 Hopper 上能自动用 wgmma 与 TMA，也在推进自动 warp specialization），程序员没有像 CUDA 那样的手工控制权。

### 2. 实践一：完整的 Triton FlashAttention 前向（因果 + GQA）

下面是本篇的第一个实践：一个支持因果掩码与 GQA 的 FlashAttention-2 风格前向 kernel。输入布局 `[B, H, N, d]`，$$K, V$$ 的 head 数 $$H_{kv}$$ 可以小于 $$H_q$$，query head `h` 映射到 KV head `h // g`（$$g = H_q / H_{kv}$$）。要求 `BLOCK_M % BLOCK_N == 0`、head dim 为 2 的幂。

```python
# flash_attn_triton.py
import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453


@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q,
    K_base, V_base, stride_kn, stride_vn,
    offs_m, offs_d, qk_scale,
    lo, hi, N_CTX,
    BLOCK_N: tl.constexpr,
    MASKED: tl.constexpr,
):
    # 遍历 [lo, hi) 范围内的 KV 块；lo 是 BLOCK_N 的倍数
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_mask = offs_n < N_CTX
        # K^T tile: [HEAD_DIM, BLOCK_N]
        k = tl.load(K_base + offs_n[None, :] * stride_kn + offs_d[:, None],
                    mask=kv_mask[None, :], other=0.0)
        # S = Q K^T，FP32 累加，已含 1/sqrt(d) 与 log2(e)
        qk = tl.dot(q, k) * qk_scale
        if MASKED:
            allowed = (offs_m[:, None] >= offs_n[None, :]) & kv_mask[None, :]
        else:
            allowed = kv_mask[None, :]
        qk = tl.where(allowed, qk, float("-inf"))
        # online softmax：新的行最大值、本块概率、修正因子
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        # V tile: [BLOCK_N, HEAD_DIM]；P 转成 V 的 dtype 后直接进第二个 GEMM
        v = tl.load(V_base + offs_n[:, None] * stride_vn + offs_d[None, :],
                    mask=kv_mask[:, None], other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij
    return acc, l_i, m_i


@triton.jit
def _attn_fwd(
    Q, K, V, O, Lse,
    stride_qz, stride_qh, stride_qm,
    stride_kz, stride_kh, stride_kn,
    stride_vz, stride_vh, stride_vn,
    stride_oz, stride_oh, stride_om,
    H_Q, N_CTX, sm_scale,
    GROUP: tl.constexpr,        # H_Q // H_KV
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H_Q
    off_hq = off_hz % H_Q
    off_hk = off_hq // GROUP     # GQA：query head -> kv head

    Q_base = Q + off_z * stride_qz + off_hq * stride_qh
    K_base = K + off_z * stride_kz + off_hk * stride_kh
    V_base = V + off_z * stride_vz + off_hk * stride_vh
    O_base = O + off_z * stride_oz + off_hq * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_mask = offs_m < N_CTX
    q = tl.load(Q_base + offs_m[:, None] * stride_qm + offs_d[None, :],
                mask=q_mask[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.4426950408889634

    if IS_CAUSAL:
        diag_lo = start_m * BLOCK_M
        # 阶段一：对角线左侧的块，无 mask
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, K_base, V_base, stride_kn, stride_vn,
            offs_m, offs_d, qk_scale, 0, diag_lo, N_CTX,
            BLOCK_N=BLOCK_N, MASKED=False)
        # 阶段二：对角块，逐元素 mask
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, K_base, V_base, stride_kn, stride_vn,
            offs_m, offs_d, qk_scale, diag_lo, tl.minimum(diag_lo + BLOCK_M, N_CTX), N_CTX,
            BLOCK_N=BLOCK_N, MASKED=True)
    else:
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, K_base, V_base, stride_kn, stride_vn,
            offs_m, offs_d, qk_scale, 0, N_CTX, N_CTX,
            BLOCK_N=BLOCK_N, MASKED=False)

    acc = acc / l_i[:, None]
    # logsumexp（自然对数），供反向 recompute 使用
    lse = (m_i + tl.math.log2(l_i)) * 0.6931471805599453
    tl.store(Lse + off_hz * N_CTX + offs_m, lse, mask=q_mask)
    tl.store(O_base + offs_m[:, None] * stride_om + offs_d[None, :],
             acc.to(O.dtype.element_ty), mask=q_mask[:, None])


def flash_attn_fwd(q, k, v, causal=True, sm_scale=None,
                   BLOCK_M=128, BLOCK_N=64, num_warps=8, num_stages=3):
    """q: [B, Hq, N, d]; k, v: [B, Hkv, N, d]; 返回 (o, lse)。"""
    B, Hq, N, D = q.shape
    Hkv = k.shape[1]
    assert k.shape == v.shape and k.shape[0] == B and k.shape[2] == N and k.shape[3] == D
    assert Hq % Hkv == 0 and D in (16, 32, 64, 128, 256)
    assert BLOCK_M % BLOCK_N == 0
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    o = torch.empty_like(q)
    lse = torch.empty((B, Hq, N), device=q.device, dtype=torch.float32)
    grid = (triton.cdiv(N, BLOCK_M), B * Hq)
    _attn_fwd[grid](
        q, k, v, o, lse,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        Hq, N, sm_scale,
        GROUP=Hq // Hkv, HEAD_DIM=D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, IS_CAUSAL=causal,
        num_warps=num_warps, num_stages=num_stages,
    )
    return o, lse
```

几处值得逐行核对的地方：

- **grid 与索引**：`program_id(1)` 遍历 $$B \times H_q$$，`off_hk = off_hq // GROUP` 完成 GQA 映射；`Lse` 是 `[B, Hq, N]` 连续张量，所以 `off_hz * N_CTX + offs_m` 正确。
- **不会出现 NaN**：因果模式下每行访问的第一个 tile 是 `start_n = 0`（阶段一）或 `start_n = diag_lo`（阶段二，此时阶段一为空），而 `diag_lo <= offs_m` 对块内所有行成立且 `diag_lo < N_CTX`，所以每行至少有一个合法列，`m_ij` 有限。后续 tile 整行被 mask 时 `p = exp2(-inf) = 0`、`alpha = exp2(0) = 1`，安全。`N_CTX` 以外的 padding 行不写回。
- **$$P$$ 直接进第二个 GEMM**：`p.to(v.dtype)` 后 `tl.dot(p, v, acc)`，$$S$$、$$P$$ 从未离开寄存器。
- **`tl.dot(a, b, acc)`** 的三参数形式把累加融合进 mma，Triton 3.x 支持。
- **数值域**：`qk_scale` 已含 $$\log_2 e$$，全程 `exp2`；输出 `lse` 时乘 $$\ln 2$$ 换回自然对数，与 FlashAttention 的 `softmax_lse` 语义一致。

### 3. 正确性与性能对照

对照 PyTorch 的 `F.scaled_dot_product_attention`（PyTorch 2.5 起支持 `enable_gqa=True`，v2.10.0 当然支持）：

```python
def check(B=2, Hq=32, Hkv=8, N=4096, D=128, causal=True, dtype=torch.bfloat16):
    torch.manual_seed(0)
    q = torch.randn(B, Hq, N, D, device="cuda", dtype=dtype)
    k = torch.randn(B, Hkv, N, D, device="cuda", dtype=dtype)
    v = torch.randn(B, Hkv, N, D, device="cuda", dtype=dtype)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=True)
    out, lse = flash_attn_fwd(q, k, v, causal=causal)
    # BF16 输出、FP32 累加，两边 P 都在 BF16 下喂入第二个 GEMM，
    # 逐元素误差量级约 1e-2，故 atol 从默认的 1e-5 放宽到 1e-2
    torch.testing.assert_close(out, ref, rtol=1.6e-2, atol=1e-2)
    # lse 与 FP32 参考对照：逐 (batch, head) 物化一个 N x N 的 S（64 MiB），
    # 避免一次性分配 [B, Hq, N, N]
    g = Hq // Hkv
    causal_mask = torch.ones(N, N, device="cuda", dtype=torch.bool).triu(1)
    for b in range(B):
        for h in range(Hq):
            s = (q[b, h].float() @ k[b, h // g].float().T) / math.sqrt(D)
            if causal:
                s = s.masked_fill(causal_mask, float("-inf"))
            torch.testing.assert_close(lse[b, h], torch.logsumexp(s, dim=-1),
                                       rtol=1e-3, atol=1e-3)
    print("ok")


def bench(B=1, Hq=32, Hkv=8, N=4096, D=128, causal=True, dtype=torch.bfloat16):
    q = torch.randn(B, Hq, N, D, device="cuda", dtype=dtype)
    k = torch.randn(B, Hkv, N, D, device="cuda", dtype=dtype)
    v = torch.randn(B, Hkv, N, D, device="cuda", dtype=dtype)
    flops = 4 * B * Hq * N * N * D * (0.5 if causal else 1.0)
    ms_tri = triton.testing.do_bench(lambda: flash_attn_fwd(q, k, v, causal=causal))
    ms_ref = triton.testing.do_bench(
        lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=True))
    print(f"triton : {ms_tri:.3f} ms  {flops / ms_tri / 1e9:.1f} TFLOPS")
    print(f"sdpa   : {ms_ref:.3f} ms  {flops / ms_ref / 1e9:.1f} TFLOPS")


if __name__ == "__main__":
    check()
    bench()
```

`lse` 的对照代码里每个 `s` 是一个 `[N, N]` 的 FP32 矩阵——就是本文第一节说不该物化的那 64 MiB；测试代码恰好用最笨的方式把它物化出来做参考，这没问题，因为测试只跑一次。容差方面：输出 `out` 是 BF16，两边的 $$P$$ 都在 BF16 下喂给第二个 GEMM，但块的划分不同（SDPA 后端的 tile 大小与本 kernel 不一定一致），累加顺序不同带来的差异在 $$10^{-2}$$ 量级，所以 `atol` 从系列默认的 `1e-5` 放宽到 `1e-2`；`lse` 是 FP32 累加得到的标量，与 FP32 参考对照可以用严格得多的 `1e-3`。如果 `out` 的误差明显超过 `1e-2`，通常不是容差问题，而是 mask 边界或 GQA 映射写错了。

关于结果的预期：A100 上 `N = 4096`、causal、32 head、$$d = 128$$，FLOPs 为 $$4 \times 32 \times 4096^2 \times 128 / 2 \approx 137$$ GFLOP，compute 下界 $$137 / 312 \approx 0.44$$ ms。SDPA 在 A100 上走 FlashAttention-2 后端，文献与经验中通常达到 50%–70% 的 BF16 峰值，即 0.6–0.9 ms 的量级；未调参的 Triton 版本通常落在 FA2 的 70%–90%，自动调优 `BLOCK_M/BLOCK_N/num_warps/num_stages` 后可以接近。差距来源是第七篇讨论过的那些：Triton 的软件流水与寄存器分配不如手写 CuTe 精细、对角块的 mask 分支、以及 $$P$$ 转 BF16 的布局转换。

### 4. vLLM 的 triton_unified_attention：一个 kernel 统一 prefill 与 decode

vLLM 的 `vllm/v1/attention/ops/triton_unified_attention.py` 把上面这个 kernel 推广到了推理引擎需要的全部形态——分页 KV、变长、GQA 打包、sliding window、split-KV——而且**用同一个 kernel 同时处理 prefill 和 decode**。它的 grid 是 `(total_num_q_blocks, num_kv_heads[, num_segments])`，`program_id(1)` 是 **KV head** 而不是 query head，一个 tile 的 `BLOCK_M` 行由 `BLOCK_Q` 个 token × `num_queries_per_kv` 个 query head 拼成：

```python
# vllm/v1/attention/ops/triton_unified_attention.py (vLLM v0.20.0), 节选
q_block_global_idx = tl.program_id(0)
kv_head_idx = tl.program_id(1)
segm_idx = tl.program_id(2) if IS_3D else 0

(seq_idx, q_block_local_idx, cur_batch_in_all_start_index,
 cur_batch_query_len, seq_len) = resolve_seq_and_query_len(
    query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q)

offs_m = tl.arange(0, BLOCK_M)
offs_d = tl.arange(0, HEAD_SIZE_PADDED)
offs_t = tl.arange(0, TILE_SIZE)
query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

query_offset_0 = cur_batch_in_all_start_index + query_pos
query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
```

`offs_m // num_queries_per_kv` 是 token 位置、`offs_m % num_queries_per_kv` 是组内的 query head——这就是第四节说的"把 $$g$$ 个 query head pack 进 mma 的 M 维"。host 侧 `BLOCK_M = 16 if num_queries_per_kv <= 16 else next_power_of_2(num_queries_per_kv)`，`BLOCK_Q = BLOCK_M // num_queries_per_kv`：Llama-3-8B（$$g = 4$$）时一个 tile 是 4 个 token × 4 个 head；decode 的 `cur_batch_query_len = 1` 时 tile 里只有 4 行有效，其余被 `query_mask_0` 屏蔽，但 $$K, V$$ 无论如何只读一次。

KV 侧的分页寻址在主循环里：

```python
# 同文件，主循环节选
for j in range(loop_lo, loop_hi):
    seq_offset = j * TILE_SIZE + offs_t
    tile_mask = seq_offset < max_seq_prefix_len
    physical_block_idx = tl.load(
        block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
    ).to(tl.int64)
    k_offset = (physical_block_idx[None, :] * stride_k_cache_0
                + kv_head_idx * stride_k_cache_2
                + offs_d[:, None] * stride_k_cache_3
                + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1)
    K_load = tl.load(key_cache_ptr + k_offset,
                     mask=dim_mask[:, None] & tile_mask[None, :], other=0.0)
    ...
    S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
    S += scale * tl.dot(Q, K)
    S = tl.where(query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf"))
    M, L, P, alpha = softmax_step(S, M, L)
    acc = acc * alpha[:, None]
    acc += tl.dot(P.to(V.dtype), V)
```

一个 `TILE_SIZE`（prefill 32、decode 16）个 token 的 tile 可能跨多个 KV block，所以 `physical_block_idx` 是一个向量：**每个 token 各自查表**，然后 `seq_offset % BLOCK_SIZE` 给出块内偏移——这是 Triton 里表达 block table 间接寻址最直接的写法（gather 式 load）。`softmax_step` 就是本文的 online softmax 一步。prefill/decode 的统一在于：`cur_batch_query_len` 是多少行、`context_len` 是多少、要遍历多少 tile，全部是运行时量，由 `cu_seqlens`（`query_start_len_ptr`）与 `seq_lens` 决定；kernel 代码不区分两种形态。

decode 时它还有 "3D" 模式（`IS_3D`）：当 batch 里没有 prefill 且序列数不超过阈值时，grid 加第三维 `num_par_softmax_segments`，每个 program 只处理自己 segment 内的 tile（`compute_tile_loop_bounds` 里的 `loop_lo/loop_hi` 裁剪），把未归一化的 `acc` 和 `(M, L)` 写进 `segm_*` 缓冲，再由 `reduce_segments` kernel 合并——与 paged_attention_v2 的 partition + reduce 完全同构。

### 5. 实践二：CUDA 版核心循环骨架

CUDA 版本这里只给核心循环的骨架（**不是完整可编译代码**，省略了 include、shared memory 的搬运与双缓冲、fragment 的具体寄存器映射与边界处理）。目的是标出 FA2 的关键决策在 CUDA 层面对应到哪些指令与布局：

```cpp
// 骨架：一个 block 4 个 warp，处理 BLOCK_M = 64 行 Q（每 warp 16 行），
// 每次迭代处理 BLOCK_N = 64 列 K/V，d = 128。mma.sync m16n8k16 BF16。
// 每个 warp 持有：
//   Q fragment  : 16 x 128，A 操作数，整个 kernel 常驻寄存器（8 个 k16 片）
//   S/P 累加器  : 16 x 64  FP32，8 个 n8 片，每线程 4 个 float/片 -> 32 个 float
//   O 累加器    : 16 x 128 FP32，16 个 n8 片 -> 64 个 float
//   m[2], l[2]  : 本线程负责的两行（mma 累加器布局里每线程覆盖 row=lane/4 和 row+8）
__device__ void attn_1rowblock_warp(/* ... */) {
  float acc_o[16][4] = {0};                    // O 累加器 fragment
  float m_row[2] = {-INFINITY, -INFINITY};
  float l_row[2] = {0.f, 0.f};

  for (int n_block = n_block_min; n_block < n_block_max; ++n_block) {
    // (0) K_j, V_j 已由 cp.async 搬进 smem（双缓冲，另一组在飞）
    // (1) S = Q K^T：8 个 n8 片 x 8 个 k16 片 = 64 条 mma
    float acc_s[8][4] = {0};
    for (int kk = 0; kk < 8; ++kk)             // 沿 d 的 k16 片
      for (int nn = 0; nn < 8; ++nn)           // 沿 BLOCK_N 的 n8 片
        mma_m16n8k16_bf16(acc_s[nn], frag_q[kk], ldmatrix_k(smem_k, nn, kk));

    // (2) mask（仅对角块）+ 行最大值：本线程 4 个 float/片 覆盖两行各两列，
    //     先在线程内取 max，再与同一行的另外 3 个 lane 用 shfl_xor(1), (2) 归约
    float m_new[2];
    for (int r = 0; r < 2; ++r) {
      float mx = m_row[r];
      for (int nn = 0; nn < 8; ++nn) mx = fmaxf(mx, fmaxf(acc_s[nn][2*r], acc_s[nn][2*r+1]));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
      m_new[r] = mx;
    }
    // (3) P = exp2(S*scale - m_new)，行和；同时用 alpha 缩放 O 与 l（延迟归一化）
    for (int r = 0; r < 2; ++r) {
      float alpha = exp2f(m_row[r] - m_new[r]);
      l_row[r] *= alpha;
      for (int nn = 0; nn < 16; ++nn) { acc_o[nn][2*r] *= alpha; acc_o[nn][2*r+1] *= alpha; }
      float rs = 0.f;
      for (int nn = 0; nn < 8; ++nn) {
        acc_s[nn][2*r]   = exp2f(acc_s[nn][2*r]   * scale_log2e - m_new[r]);
        acc_s[nn][2*r+1] = exp2f(acc_s[nn][2*r+1] * scale_log2e - m_new[r]);
        rs += acc_s[nn][2*r] + acc_s[nn][2*r+1];
      }
      rs += __shfl_xor_sync(0xffffffff, rs, 1);
      rs += __shfl_xor_sync(0xffffffff, rs, 2);
      l_row[r] += rs;  m_row[r] = m_new[r];
    }
    // (4) P 累加器 -> A fragment：m16n8k16 的 C 布局（每线程 2 行 x 2 列）与
    //     A 布局（每线程 2 行 x 2x2 列）在 n8 片两两拼接后恰好对齐，
    //     只需把相邻两个 n8 片的 FP32 pack 成 4 个 bf16x2，无需经过 smem
    uint32_t frag_p[4][4];                      // 4 个 k16 片（BLOCK_N = 64 = 4 x 16）
    pack_c_to_a_bf16(acc_s, frag_p);
    // (5) O += P V：4 个 k16 片 x 16 个 n8 片 = 64 条 mma
    for (int kk = 0; kk < 4; ++kk)
      for (int nn = 0; nn < 16; ++nn)
        mma_m16n8k16_bf16(acc_o[nn], frag_p[kk], ldmatrix_v_trans(smem_v, nn, kk));
    // (6) __syncthreads() + 切换 smem 双缓冲、发起下一块的 cp.async
  }
  // 收尾：O /= l（每行一次除法），写回；lse = (m + log2(l)) * ln2
}
```

标出几个复用点：(a) $$Q$$ 的 A fragment 全程常驻寄存器，这是 "外循环遍历 Q 块" 的直接后果；(b) 步骤 (4) 的 C→A fragment 复用是 FA2 能在寄存器内完成 $$S \to P \to PV$$ 的关键——`m16n8k16` 的累加器布局中每线程持有第 `lane/4` 行与第 `lane/4 + 8` 行的两对相邻元素，与 A 操作数布局中每线程持有的位置重合，只差一次 FP32→BF16 的 pack；(c) 行归约只需 `shfl_xor` 1 和 2 两步，因为 mma 布局里一行的 8 列分布在同一 quad 的 4 个 lane 上（"split Q" 让归约不出 warp）；(d) $$V$$ 需要转置载入（`ldmatrix.trans`），因为 $$PV$$ 的 B 操作数要求 k-major——这就是 FA3 在 FP8 下需要显式重排 $$V$$ 布局的原因（FP8 的 `ldmatrix` 没有对应的转置形式）。


## 八、后端生态与 vLLM 的选择

到 2026 年 5 月，生产环境里的 attention kernel 主要来自以下几家：

- **FlashAttention-2 / 3**（Dao AI Lab）：`flash_attn_func` / `flash_attn_varlen_func`，CUDA + CuTe 手写；FA2 支持 sm_80+，FA3 只支持 sm_90。原版不支持分页 KV，vLLM 维护的 **vllm-flash-attn** fork 给 varlen 接口加了 `block_table` 与 `seqused_k` 参数（上一节引用的调用），并保留 FA3 的 `scheduler_metadata`（提前算好 split 策略，便于 CUDA graph）。
- **FlashInfer**：面向推理的 attention 库，JIT 编译 kernel 变体，`plan()`/`run()` 两段式接口（`plan` 在 CPU 侧做 split-KV 的负载均衡与 workspace 分配，`run` 只发 kernel），decode 与 paged prefill 是它的专长；vLLM 的 `flashinfer.py` 后端用 `BatchPrefillWithPagedKVCacheWrapper` 与 `BatchDecodeWithPagedKVCacheWrapper`。
- **xFormers**：Meta 的 `memory_efficient_attention`，早期 vLLM 的默认后端之一，现在主要用于视觉模型等场景。
- **cuDNN attention**：NVIDIA 在 cuDNN 8.9+ 提供的 fused attention（Hopper 上很强），是 PyTorch `scaled_dot_product_attention` 的后端之一（与 FlashAttention-2 后端、memory-efficient 后端、math 后端并列，由 `torch.backends.cuda` 的开关和形状约束决定走哪个）。
- **Triton 实现**：vLLM 的 `triton_unified_attention.py`（上一节）、SGLang 的 Triton 后端、Triton tutorial 06 及其衍生。可移植性最好（ROCm 直接可用），性能在 Ampere 上通常落后手写 FA2 一到三成，在 Hopper 上落后 FA3 更多。
- **PyTorch 的 `scaled_dot_product_attention`**：不是一个独立 kernel 而是一个分发器，按输入的 dtype、head dim、是否有 mask、是否需要梯度等条件在 FlashAttention-2、cuDNN、memory-efficient（xFormers 派生）与 math 四个后端之间选择；`torch.nn.attention.sdpa_kernel` 可以强制指定。本篇实践里的 `F.scaled_dot_product_attention(..., is_causal=True, enable_gqa=True)` 在 A100 上通常走 FlashAttention-2 后端，所以它既是正确性参考也是一个有意义的性能基线。

选择的原则可以压缩成三条：prefill 追求 Tensor Core 利用率，优先 FA3（sm_90）或 FA2；decode 追求带宽利用率与并行度，split-KV 的策略质量比 GEMM 效率更重要，FlashInfer 与 FA3 的 scheduler 在这里下了最多功夫；需要非标准特性（新的 mask 形状、bias、KV 量化格式）时，Triton 版本的修改成本远低于 CUDA 版本，这是它在生产系统里一直有一席之地的原因。

vLLM v0.20.0 的选择逻辑在 `vllm/platforms/cuda.py`：用户可用 `--attention-backend`（或环境变量 `VLLM_ATTENTION_BACKEND`）显式指定；不指定时按设备能力给出优先级列表，逐个调用各后端类的 `validate_configuration` 检查 dtype、head size、block size、KV cache 量化等约束，取第一个通过的。sm_80/sm_90 上的默认顺序是 FLASH_ATTN → FLASHINFER → TRITON_ATTN → FLEX_ATTENTION；Blackwell（compute capability 10.x）上把 FLASHINFER 提到最前。FLASH_ATTN 后端内部再由 `fa_utils.get_flash_attn_version` 决定 FA 版本：sm_90 优先 FA3，其余用 FA2（ALiBi 等 FA3 不支持的特性会回退到 FA2）。


## 九、小结

本篇把前七篇的工具用在了一个 kernel 上。回到核心问题：

**标准 vs FlashAttention 的 HBM 读写（N = 4096、d = 128、单 head BF16）**：标准实现要物化 $$S$$ 与 $$P$$，$$N^2$$ 项的流量至少 128 MiB、合计约 132 MiB，算术强度约 62 FLOP/byte，在 A100 上 memory-bound；FlashAttention 只读 $$Q$$、写 $$O$$ 各 1 MiB，$$K, V$$ 各重读 $$N / B_r$$ 次，$$B_r = 128$$ 时 64 MiB，理论 HBM 流量 $$\Theta(N^2 d^2 / M)$$，而且这些重读大多命中 L2，实际 HBM 流量接近 4 MiB 的下界——attention 因此从 memory-bound 变成 compute-bound。它不省 FLOPs（还略多），省的是 IO。

**decode 每 token 读多少 KV、这决定了什么**：Llama-3-8B 每 token 128 KiB，上下文 $$s$$ 时读 $$128\ \text{KiB} \times s$$；算术强度约 $$g$$ FLOP/byte（GQA 组大小），彻底 memory-bound。一步 decode 的时间下界是 (权重字节 + batch × s × 128 KiB) / 带宽：batch × s 超过约 131k token 时 KV 项超过权重项。这决定了 KV 量化、GQA、split-KV 这些手段的价值。

```text
标准 attention vs FlashAttention（N=4096, d=128, 单 head, BF16, A100）
                        标准实现                 FlashAttention (B_r=128)
Q/K/V/O                  4 MiB                    Q 读 1 MiB + O 写 1 MiB
S (N^2)                  写 32 + 读 32 MiB         不物化
P (N^2)                  写 32 + 读 32 MiB         不物化（寄存器内）
K/V 重读                  --                       32 次 x 2 MiB = 64 MiB（多数命中 L2）
合计（HBM 口径）           ~132 MiB                 ~66 MiB（实际 HBM 接近 4 MiB）
FLOPs                    8.6 GFLOP                8.6 GFLOP + rescale 小项
算术强度                  ~62 FLOP/byte             >=125（实际 ~2000）
Roofline                 memory-bound (~69 us)     compute-bound (~28 us 下界)

FlashAttention 三代
        并行维度                    warp 分工       非 matmul 优化              硬件
FA1     (batch, head)，外循环 K/V    split-K         --                          sm_80
FA2     (batch, head, Q block)      split-Q         延迟归一化、exp2            sm_80+
FA3     同 FA2 + warp specialization  producer/consumer, 2 warpgroup ping-pong
                                    GEMM-softmax 重叠、FP8 块量化 + 布局重排       sm_90 only
反向    只存 O 与 L = m + log l，重算 P = exp(S - L)：省 N^2 显存，多 2N^2d FLOPs

prefill vs decode（Llama-3-8B, BF16）
              Q 行数      每 token 读 KV    算术强度           瓶颈
prefill       N_q（长）    --               ~4N_q d / (...)高   Tensor Core
decode        1           128 KiB x s       ~g (=4)            HBM 带宽
decode 下界    (16 GB + batch x s x 128 KiB) / BW；batch x s > ~131k 时 KV 项占主导

vLLM PagedAttention kernel（v0.20.0, csrc/attention）
v1      grid (heads, seqs)，128 线程，每 warp 一个 KV block，thread group 16 B 向量读 K
        K 布局 [blocks, kv_heads, d/x, block_size, x]，x = 16/sizeof(T)
        logits 放 shared (FP32)，两级 reduction softmax，再遍历 V 累加
v2      + PARTITION_SIZE=512，grid 加第三维，reduce kernel 用 online softmax 合并公式
merge_attn_states  两段 (O, lse) 的通用合并；Flash-decoding / split-KV 同构

后端（vLLM v0.20.0 CUDA 默认优先级）
FLASH_ATTN（sm_90 用 FA3，其余 FA2；vllm-flash-attn 加 block_table）
  -> FLASHINFER（JIT, plan/run，decode 专长；Blackwell 上优先）
  -> TRITON_ATTN（triton_unified_attention，prefill/decode 统一）
  -> FLEX_ATTENTION
```

方法论上，这一篇再次验证了系列的主线：**先数字节、再数 FLOPs、放到 Roofline 上看瓶颈在哪，再决定优化方向**。FlashAttention 的每一代都对应瓶颈的一次转移——从 HBM 流量（FA1）到并行度与 warp 同步（FA2）再到 SFU 与 Tensor Core 的重叠（FA3）；decode 的瓶颈从来都是带宽，所以 PagedAttention、split-KV、GQA 打包做的都是同一件事：让 KV 只读一次、让足够多的 SM 同时读。


## 下一篇

[量化与融合 kernel](/quantization-and-fused-kernels.html)
