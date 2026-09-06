---
layout: post
title: "大规模训练工程（02）：并行策略全景——每种并行切的是哪种状态"
subtitle: "A Map of Parallelism: Which State Does Each Strategy Shard"
tags: [Megatron, DeepSpeed, torchtitan, Distributed Training, Parallelism, AI, AI-Infra]
catalog: true
---

> 本文是《大规模训练工程：从并行策略到容错恢复》系列的第 2 篇。上一篇：[训练任务的状态解剖：显存账与 MFU](/training-state-anatomy-memory-and-mfu.html)；下一篇：[三个框架：Megatron-LM、DeepSpeed 与 torchtitan 的架构对比与源码导读](/megatron-deepspeed-torchtitan-architecture-and-source-guide.html)。

上一篇算出了一个数字：混合精度 + Adam 下，每个参数在训练时要占 16 字节。一个 70B 的模型光是参数、梯度和优化器状态就是 1.13 TB，还没算激活；405B 是 6.5 TB。任何一张 80 GB 的卡都放不下其中的零头。所以这些字节必须被切开放到很多卡上——**怎么切**，就是并行策略的全部内容。

习惯上并行策略被当作一组 API：DDP、FSDP、`ColwiseParallel`、`Schedule1F1B`、`--tensor-model-parallel-size 8`。这样理解的问题是你记住了十几个名字，却回答不了"这个配置为什么慢"。换一个角度：训练的状态只有四样——参数、梯度、优化器状态、激活——每一种并行策略无非是决定**这四样东西的哪一样、沿哪个维度、切到哪一组卡上**，然后为这个决定付出**一种特定形态的通信**。DP 什么都不切、只切数据，代价是每步一次梯度 all-reduce；ZeRO-3 把三种常驻状态都切了，代价是每步多一次参数 all-gather；TP 切矩阵，代价是每层四次激活大小的集合通信；PP 切层，代价是气泡；CP 切序列，代价是 K/V 在卡间流动；EP 切专家，代价是 all-to-all。

把每种策略写成同一个五元组——**切哪种状态 / 每 step 通信量 / 走哪条链路 / 能否与计算重叠 / 适用条件**——它们就可以放在同一张表里比较，而"该用几维并行、每维多少"这个问题也就变成了一道有公式可套的算术题。本篇就是把这张表填满。

本篇的核心问题：

> **每种并行都在"复制"和"切分"之间做交换：复制多占显存，切分多花通信。给定一个模型和一个集群的拓扑（节点内 NVLink、节点间 InfiniBand），每个维度的通信量是多少、走哪条链路、和计算能不能重叠？这决定了它该放在几维并行的哪一层。**

本篇不碰任何一个框架的进程组代码（那是下一篇），只在每一节末尾指出 PyTorch 2.13.0 与 Megatron Core 0.18.0 里对应实现的位置，方便读者按图索骥。通信原语（all-reduce、all-gather、reduce-scatter、all-to-all、send/recv）只用它们的**语义和每卡通信量**，耗时当作由链路带宽决定的黑盒。


## 一、总览

### 1. 符号与第一篇的结论

本篇沿用第一篇的记账符号，用到的复述如下：

```text
N            参数量（个数）。混合精度 + Adam 下每参数 16 字节：
             bf16 参数 2 + bf16 梯度 2 + fp32 主参数 4 + fp32 一阶矩 4 + fp32 二阶矩 4
             → 常驻状态 16N 字节（fp32 累加梯度则 18N）
N_d N_t N_p N_c N_e     数据 / 张量 / 流水 / 上下文 / 专家并行度；总卡数 N = N_t · N_c · N_p · N_d
s  b  h  a  L           序列长、micro-batch 大小、隐藏维、注意力头数、层数
m                       每个 DP 副本每 step 的 micro-batch 数（梯度累积步数）
```

**通信量的口径**：本篇所有"通信量"指**一张卡在一个 step 内发出的字节数**（全双工链路上收发同时进行、数量相等）。对 $$n$$ 个参与者、有效载荷 $$S$$ 字节，五个原语的每卡通信量是：

```text
all-reduce(S)                 2(n-1)/n · S  ≈ 2S       S 为每卡 buffer 大小
all-gather(S)                   (n-1)/n · S  ≈  S       S 为拼接后的总大小
reduce-scatter(S)               (n-1)/n · S  ≈  S       S 为输入总大小
all-to-all(S)                   (n-1)/n · S  ≈  S       S 为每卡输入总大小；n(n-1) 条独立的流
send/recv(S)                              S             点对点
```

以及一条后面反复使用的等式：**all-reduce = reduce-scatter + all-gather**，两半各 $$\approx S$$，合计 $$\approx 2S$$。用 N 计量时，$$N$$ 个 bf16 梯度做一次 all-reduce，每卡搬 $$\approx 2N$$ 个元素，即 $$4N$$ 字节；本篇按 ZeRO 论文的习惯把它写成"$$2N$$ 的通信量"，单位是元素，乘以 dtype 字节数才是字节。

### 2. 并行是状态的放置方案

把四种状态、六种并行放在同一张图上：

```text
                   参数        梯度        优化器状态      激活
                 ─────────   ─────────   ────────────   ────────────────────
DP (DDP)         复制         复制→归约    复制            按 batch 切（天然）
ZeRO-1           复制         复制→归约    切 1/N_d        同上
ZeRO-2           复制         切 1/N_d     切 1/N_d        同上
ZeRO-3 / FSDP    切 1/N_d     切 1/N_d     切 1/N_d        同上；前向前临时 all-gather 一层参数
TP               切 1/N_t     切 1/N_t     切 1/N_t        按 h（头 / FFN 列）切 1/N_t，层边界处完整
SP（Megatron）    —            —            —               层边界处也按 s 切 1/N_t
CP               复制         复制→归约    复制            按 s 切 1/N_c，含注意力本身
PP               切 1/N_p     切 1/N_p     切 1/N_p        只持有本 stage 的层；同时在途 ≤ N_p 个 micro-batch
EP               专家切 1/N_e  同           同              token 按路由结果 all-to-all 到专家所在卡
```

读这张表的方式：**每一行"切"的格子越多，显存越省；每一个"切"都对应一种通信**。DP 那一行只在"梯度→归约"处有通信；ZeRO-3 在参数一格多了一次 all-gather；TP 的三个"切"是免费的（矩阵切开后梯度和优化器状态自然跟着切），它的代价在激活那一格——层内切开的激活要在层边界处拼回来。PP 同理，三个状态的"切"不产生通信，代价是 stage 之间传激活和气泡。

这张图还揭示了另一件事：**DP 系的策略（DP/ZeRO/FSDP）与模型并行系的策略（TP/PP/CP/EP）是正交的**。前者决定"同一个参数的几份副本之间如何分工"，后者决定"一份模型如何被切开"。任何一个真实配置都是两者的乘积：模型被 TP/PP 切成 $$N_t N_p$$ 份，每份再有 $$N_d N_c$$ 个副本，副本之间用 ZeRO 的某一级分片。

### 3. 五元组与两条链路

每种并行的五元组里，"走哪条链路"和"能否重叠"是决定它放在哪一层的关键。集群有两种链路：节点内 8 张卡之间的 NVLink（H100 标称单向 450 GB/s，8 卡集合通信通常能用到每卡 200–300 GB/s 量级）和节点之间每卡一张的 InfiniBand（NDR 标称单向 50 GB/s）。两者带宽差 5–10 倍。

```text
                通信量（每卡每 step）                     形态          在关键路径上？
──────────────  ────────────────────────────────────    ──────────    ─────────────────────────
TP              4 × 2 × (激活 s·b·h) × 每 stage 层数 × m   集合          是；下一步 GEMM 等它 → 必须 NVLink
CP (ring)       3 × (N_c-1) × 每卡 K/V 块 × 层数 × m        点对点环      否；与注意力分块计算重叠 → 可跨节点
PP              2 × 层边界激活 × m                          点对点        否；调度掩盖 → 跨节点
DP / ZeRO       2N_local（ZeRO-3 为 3N_local）              集合          否；与反向重叠 → 跨节点，但量大
EP              4 × 路由 token × 层数 × m                    all-to-all    是；专家计算等它 → 尽量节点内
```

"通信量大"和"必须快链路"是两件事。TP 的通信量在数值上常常是最大的一项，但真正把它锁在节点内的是**它在关键路径上**：column-parallel 的输出经过 all-reduce 才能进下一个 GEMM，没有东西可以和它重叠。DP 的通信量也不小，但它是一大块可以在反向传播期间慢慢发的数据，对延迟不敏感，只要带宽够。PP 的通信量最小、又不在关键路径上，所以它是最适合放在最慢链路上的那一维。

### 4. 本文的章节安排

```text
第二章  数据并行与 ZeRO      DP 的 2N；ZeRO-1/2/3 各切什么；3N 的推导；FSDP1 vs FSDP2；HSDP
第三章  张量并行与序列并行    列切 + 行切的配对；每层 2 + 2 次 all-reduce；为什么不出节点；SP 把 all-reduce 拆成 AG + RS
第四章  上下文并行           为什么 TP/SP 不够；Ring Attention 与 Ulysses 的通信形态；GQA 的影响
第五章  流水线并行           GPipe → 1F1B → interleaved → zero-bubble；气泡率 (p-1)/m 的推导；通信量最小
第六章  专家并行             专家的状态形态；all-to-all 通信量；负载不均的双重代价；EP 与 DP/TP 的组合
第七章  组合与实例           五元组大表；组合顺序 TP → CP → PP → DP 及原因；Llama 3 405B 代入；原语对应表；三框架对照
第八章  小结                 要点、源码位置、train-ledger 的 ledger/parallel.py
```


## 二、数据并行与 ZeRO：切优化器状态、梯度、参数

### 1. DP：什么都不切，通信 2N

数据并行是最朴素的放置方案：每张卡持有全部 $$16N$$ 字节的常驻状态，各自处理不同的数据，反向结束后把梯度求平均，再各自做完全相同的优化器步骤。每卡显存 $$16N$$ 与 $$N_d$$ 无关——DP 不省显存，只加吞吐。

通信只有一处：$$N$$ 个 bf16 梯度的 all-reduce。按第一章的口径每卡搬 $$\frac{2(N_d-1)}{N_d}N \approx 2N$$ 个元素、$$4N$$ 字节。70B 模型是 282 GB——每张卡、每一步。这个数字与 $$N_d$$ 几乎无关（8 卡是 $$1.75N$$，1024 卡是 $$1.998N$$），这是 ring / 分层 all-reduce 带宽最优的结果，也是 DP 能扩展到上千卡的原因。

它能扩展的另一个原因是**重叠**：反向传播是从最后一层往前算的，某一层的梯度算完就可以开始 all-reduce，不必等整个反向结束。DDP 把参数按反向顺序分成若干 bucket（默认 25 MiB 一桶，PyTorch 2.13.0 `torch/csrc/distributed/c10d/reducer.hpp` 的 `kDefaultBucketBytesCap`），一个桶内的梯度都就位就发出去。只要每层的通信时间小于该层的反向计算时间，DP 的通信就被完全藏住，step 时间与单卡一样。这个条件在 $$N_d$$ 增大时不会变差（通信量不随 $$N_d$$ 变），只在**计算变少**时变差——micro-batch 太小、或者模型被 TP/PP 切得太碎，每层的反向时间就盖不住它的梯度通信了。

### 2. ZeRO-1 / ZeRO-2：切优化器状态与梯度，通信量不变

DP 的浪费在优化器状态：$$N_d$$ 张卡各存一份完全相同的 fp32 主参数、一阶矩、二阶矩（$$12N$$ 字节），然后各自做一遍完全相同的更新。ZeRO（Rajbhandari et al. 2020）的观察是这份状态可以按参数切成 $$N_d$$ 份，每张卡只负责更新自己那 $$1/N_d$$ 的参数。

**ZeRO-1** 只切优化器状态。每卡显存从 $$16N$$ 降到 $$4N + 12N/N_d$$。通信上，梯度 all-reduce 被拆成两半：先 **reduce-scatter**，让每张卡拿到自己负责的那 $$1/N_d$$ 梯度的归约结果（$$\approxN$$ 元素）；各自更新自己的参数分片后，再 **all-gather** 把更新后的 bf16 参数拼回每张卡（$$\approxN$$ 元素）。合计仍是 $$2N$$，与 DP 完全相同——因为 all-reduce 本来就等于 reduce-scatter + all-gather，ZeRO-1 只是在两者之间插入了优化器步骤。

**ZeRO-2** 再切梯度。观察是：reduce-scatter 之后，每张卡只需要自己负责的那 $$1/N_d$$ 梯度，其余 $$(N_d-1)/N_d$$ 可以在 reduce-scatter 完成的瞬间释放。于是梯度显存也降到 $$2N/N_d$$，每卡 $$2N + 14N/N_d$$。通信量仍是 $$2N$$——什么都没多。实现上要求梯度按 bucket 在反向过程中即时 reduce-scatter 并释放，而不是攒到反向结束。

Megatron 的分布式优化器（`megatron/core/optimizer/distrib_optimizer.py`）就是 ZeRO-1 这一级，`--use-distributed-optimizer` 打开；它的 reduce-scatter 与 all-gather 都以 bucket 为单位与计算重叠（下一篇展开）。

### 3. ZeRO-3：切参数，通信 2N → 3N

前两级不碰参数：bf16 参数 $$2N$$ 在每张卡上是完整的，因为前向和反向都要用完整的参数做 GEMM。**ZeRO-3** 把参数也切成 $$N_d$$ 份，每卡只常驻 $$2N/N_d$$；用到哪一层时再临时把那一层的参数 **all-gather** 回来，用完立即释放。每卡显存降到 $$16N/N_d$$——所有常驻状态都被 $$N_d$$ 均分。

代价是通信量从 $$2N$$ 涨到 $$3N$$。逐项数：

```text
前向       每层执行前 all-gather 该层参数            全部层合计   ≈ N      ← 新增
反向       每层反向前再次 all-gather 该层参数        全部层合计   ≈ N      ← 新增（前向后释放了）
反向       每层梯度算完 reduce-scatter               全部层合计   ≈ N      ← 原有
优化器后   不再需要 all-gather 参数                                 0       ← 原 ZeRO-1 的那次省掉了：
                                                                          下一步前向本来就要 all-gather
────────────────────────────────────────────────────────────────────────
合计                                                              3N
```

推导里有一个容易漏的抵消：ZeRO-1/2 在优化器步骤后有一次参数 all-gather，ZeRO-3 把它省掉了，因为下一个 step 的前向本来就要逐层 all-gather；所以净增只有反向那一次。如果前向后**不释放**参数（FSDP 的 `reshard_after_forward=False`），反向那次 all-gather 也省掉，通信回到 $$2N$$，代价是完整的 bf16 参数 $$2N$$ 常驻——这时它的显存是 $$2N + 14N/N_d$$，与 ZeRO-2 相同。**ZeRO-3 的 $$3N$$ 是用 $$1N$$ 的通信换 $$2N(1 - 1/N_d)$$ 的显存**。

另一点：这 $$3N$$ 的每一份都能重叠。前向的 all-gather 可以预取（算第 $$i$$ 层时 all-gather 第 $$i+1$$ 层），反向同样；reduce-scatter 与 DP 一样跟着反向走。所以 ZeRO-3 在带宽充足时 step 时间接近 DP，只是"带宽充足"的门槛比 DP 高 50%。

### 4. 每卡显存表：70B、N_d = 64

把 70.6B 参数（Llama 3 70B）、$$N_d = 64$$ 代入，字节数以 GB（$$10^9$$）计：

```text
               参数 (bf16)   梯度 (bf16)   优化器 (fp32×3)   常驻合计      每 step 通信（元素）     通信形态
─────────────  ───────────   ───────────   ───────────────   ──────────    ────────────────────    ─────────────────────────
DP (ZeRO-0)    141.2         141.2         847.2             1129.6  GB    2N                      all-reduce
ZeRO-1         141.2         141.2          13.2              295.6  GB    2N                      reduce-scatter + all-gather
ZeRO-2         141.2           2.2          13.2              156.6  GB    2N                      同上，梯度即时释放
ZeRO-3           2.2           2.2          13.2               17.6  GB    3N                      AG (fwd) + AG (bwd) + RS
```

三级之间的显存台阶分别是 $$12N$$、$$2N$$、$$2N$$——优化器状态是最大的一块，所以 ZeRO-1 一步就拿掉了 74%，这也是为什么 Megatron 长期只做到 ZeRO-1 而不觉得亏。表里没有激活：$$s = 8192$$、$$b = 1$$ 时一层 34sbh ≈ 2.3 GB，80 层 180 GB，ZeRO 一个字节都不帮它——切激活是 TP/SP/CP/PP 和重计算的事。

### 5. FSDP1 与 FSDP2：FlatParameter 与 per-parameter DTensor

FSDP 是 ZeRO-3 在 PyTorch 里的原生实现，有两代。

**FSDP1**（`torch/distributed/fsdp/fully_sharded_data_parallel.py` 的 `FullyShardedDataParallel`）把一个 wrap 单元（通常是一个 Transformer 层）内的所有参数**拍平拼接**成一个一维的 `FlatParameter`（`torch/distributed/fsdp/_flat_param.py`，管理它的是 `FlatParamHandle`），再把这个大一维张量均匀切成 $$N_d$$ 段。好处是每层只有一次 all-gather 和一次 reduce-scatter，通信效率最高；坏处是原始参数的形状、dtype、`requires_grad` 都被抹掉了——同一个 `FlatParameter` 里的参数必须同 dtype、要一起冻结或一起训练，和 TP 组合时要专门适配，checkpoint 里的分片也是"一维大张量的第 $$k$$ 段"，不对应任何一个具体参数。`torch/distributed/fsdp/api.py` 的 `ShardingStrategy` 枚举把 ZeRO 三级映射为 `FULL_SHARD`（ZeRO-3）、`SHARD_GRAD_OP`（ZeRO-2）、`NO_SHARD`（DDP），另有 `HYBRID_SHARD`（见下节）。

**FSDP2**（`torch/distributed/fsdp/_fully_shard/` 下的 `fully_shard()`）放弃了拍平：每个参数**各自**沿第 0 维切成 $$N_d$$ 份，切开后的分片是一个 `DTensor`（placement 为 `Shard(0)`）。一次 `fully_shard(module)` 调用把该 module 的参数编成一个通信组 `FSDPParamGroup`（`_fsdp_param_group.py`），组内每个参数对应一个 `FSDPParam`（`_fsdp_param.py`，维护 sharded / unsharded 两种状态的切换）。前向前 `FSDPParamGroup.unshard()` 把组内所有参数的分片拷进一个连续 buffer 做**一次** all-gather（`_fsdp_collectives.py` 的 `foreach_all_gather`），拷出后各参数恢复原形状；前向后 `reshard()` 释放；反向后 `post_backward()` 把梯度拷进连续 buffer 做一次 reduce-scatter（`foreach_reduce`）。所以通信次数与 FSDP1 相同，但参数的身份保留了：不同参数可以不同 dtype、可以单独冻结、每个参数的分片是一个自描述的 DTensor——这让 FSDP2 能与 TP 的 DTensor 自然组合（一个参数同时有 `Shard(0)` 的 FSDP 维和 `Shard(1)` 的 TP 维），checkpoint 也能按参数名重分片（第五篇）。`_fsdp_api.py` 的 `MixedPrecisionPolicy` 决定 all-gather 出来的参数用什么 dtype 计算、reduce-scatter 用什么 dtype 归约；`reshard_after_forward` 参数就是第 3 节里 $$3N$$ 与 $$2N$$ 之间的开关，它还可以是一个整数——前向后不是完全释放而是重分片到一个更小的组（例如节点内 8 卡），让反向的 all-gather 只在节点内做。

两代的显存与通信量相同，差别在**可组合性**：FSDP2 是 torchtitan 与 PyTorch 原生 TP/PP/CP 组合的基础，下一篇的对照会反复用到。

### 6. HSDP：节点内切、节点间复制

ZeRO-3 的 $$3N$$ 通信全在 DP 组上；当 $$N_d$$ 跨越几十个节点时，这 $$3N$$ 走的是 InfiniBand，而且 all-gather 是在关键路径附近的（预取深度有限）。**HSDP**（Hybrid Sharded Data Parallel）把 DP 维拆成两层：在一个 $$N_s$$ 卡的分片组内做 ZeRO-3（通常 $$N_s = 8$$ 或几个节点），分片组之间做普通 DP 复制（$$N_r = N_d / N_s$$ 个副本）。

```text
显存      16N / N_s                                            ← 只被分片组均分，N_s 小则显存大
通信      分片组内：AG + AG + RS ≈ 3N                 走 NVLink（N_s = 8 时）
          副本间：  reduce-scatter 之后的梯度分片做 all-reduce
                    每卡 2(N_r-1)/N_r × N/N_s ≈ 2N/N_s          走 IB，量小了 N_s 倍
```

它的取舍很直接：把大头 $$3N$$ 挪到快链路上，跨节点只剩 $$2N/N_s$$，代价是显存只省 $$N_s$$ 倍。70B 用 $$N_s = 8$$ 每卡要 141 GB，放不下；$$N_s = 64$$ 才是 17.6 GB。所以 HSDP 的 $$N_s$$ 是"刚好放得下"的最小值，不是越小越好。FSDP2 里 HSDP 由传入的 2D `DeviceMesh` 决定：`fully_shard(module, mesh=mesh_2d)` 时参数 placement 为 `(Replicate(), Shard(0))`，`_fsdp_common.py` 的 `HSDPMeshInfo` 同时持有 shard 与 replicate 两个进程组；`_fsdp_api.py` 的 `DataParallelMeshDims` 则允许在一个更高维的 SPMD mesh 上指定哪些维是 shard、哪些是 replicate。


## 三、张量并行与序列并行

### 1. 列切与行切的配对

张量并行把一个线性层的权重矩阵切开。对 $$Y = XA$$（Megatron 的记法，$$X$$ 是 $$[\text{tokens}, h]$$ 的激活，$$A$$ 是 $$[h, h']$$ 的权重），有两种切法：

```text
列切（column-parallel）  A = [A_1 | A_2 | … | A_t]     每卡持 A_i（h × h'/t）
                         输入 X 完整（每卡一份）；输出 Y_i = X A_i 是 Y 的第 i 列块
                         → 输入不需通信，输出按列分布在各卡

行切（row-parallel）     A = [A_1 ; A_2 ; … ; A_t]     每卡持 A_i（h/t × h'）
                         输入 X 必须按列切成 X_i；输出 Y = Σ_i X_i A_i 是 t 个部分和
                         → 输入按列分布即可，输出需要 all-reduce
```

Megatron-LM（Shoeybi et al. 2019）的洞见是把两者**配对**：MLP 的第一个线性层列切、第二个行切。列切的输出 $$Y_i$$ 正好是行切需要的按列分布的输入 $$X_i$$，中间的 GELU 是逐元素的、不需要完整向量——于是整个 MLP 只在末尾做一次 all-reduce。注意力同理：Q/K/V 投影列切（每卡持 $$a/N_t$$ 个头，头之间的计算天然独立），输出投影行切，末尾一次 all-reduce。

```text
      X ──┬── [列切 Q/K/V] ── 每卡 a/t 个头的注意力 ── [行切 O] ──┐
          │                                                    ├── all-reduce ── + 残差 ── LayerNorm
          └────────────────────────────────────────────────────┘
      X ──┬── [列切 W1] ── GELU ── [行切 W2] ──┐
          │                                   ├── all-reduce ── + 残差 ── LayerNorm
          └───────────────────────────────────┘
```

状态上，TP 是最"干净"的切分：权重被切成 $$1/N_t$$，它的梯度和优化器状态自然也是 $$1/N_t$$，**不需要任何额外通信来维持分片**——这与 ZeRO 形成对比，ZeRO 为了维持参数分片每步要 all-gather 两次。LayerNorm 的权重、偏置这类小参数在 TP 组内是复制的。

### 2. 每层 2 + 2 次 all-reduce 与通信量

前向每层两次 all-reduce（注意力后一次、MLP 后一次），每次的载荷是一个完整的激活张量 $$s \cdot b \cdot h$$ 个元素。反向也是两次：行切层前向的 all-reduce 在反向是恒等（梯度直接分发），而列切层前向的恒等（输入广播）在反向变成 all-reduce（各卡对 $$X$$ 的梯度要求和）——两者互为共轭，所以反向的 all-reduce 出现在与前向不同的位置，但次数相同。每层四次，每次每卡 $$\frac{2(N_t-1)}{N_t} \cdot sbh \cdot 2$$ 字节。

代入 Llama 3 405B（$$h = 16384$$，$$s = 8192$$，$$b = 1$$，bf16）：一个激活张量 268 MB，$$N_t = 8$$ 的一次 all-reduce 每卡搬 470 MB，每层四次 1.88 GB；一个 PP stage 约 8 层、每 step 16 个 micro-batch，每卡每 step **约 220 GB** 走 NVLink。这是所有并行维度里数值上最大的一项。

这 220 GB 与 $$N_t$$ 几乎无关（系数 $$2(N_t-1)/N_t$$），但每卡的**计算**随 $$N_t$$ 减少——TP 越大，同样的通信配越少的计算，通信占比线性上升。这是 TP 不能无限加大的第一个原因。

### 3. 为什么 TP 不出节点

第二个原因更硬：这四次 all-reduce **在关键路径上**。行切层的输出必须归约完才能加残差、过 LayerNorm、进下一个子层；列切层反向对 $$X$$ 的梯度必须归约完才能继续往前传。没有别的计算可以填进这段等待（Megatron 的 `LinearWithGradAccumulationAndAsyncCommunication` 能把反向里对输入的梯度 all-reduce 与对权重的梯度 GEMM 重叠，能藏一部分但不是全部）。

所以 TP 的每一次通信都直接加在 step 时间上，它对**延迟**和**带宽**都敏感。8 卡 NVLink 上一次几百 MB 的 all-reduce 是一两毫秒；跨 IB 是十几毫秒；乘以每层四次、几十层、十几个 micro-batch，差别是 step 时间的几倍。这就是 TP 几乎只在 NVLink 域内使用、$$N_t \le 8$$（NVL72 一类机器上可以更大）成为惯例的原因。放到第一章的五元组里：TP **切参数、梯度、优化器状态与层内激活；每层 $$4 \times 2sbh$$ 元素；NVLink；不可重叠；节点内**。

### 4. SP：把 all-reduce 拆成 all-gather + reduce-scatter，激活再降 N_t 倍

TP 切了层**内**的激活（Q/K/V、FFN 中间态按头 / 按列分布），但层**边界**处的激活——LayerNorm 的输入输出、dropout、残差流——在每张 TP 卡上是完整复制的：$$sbh$$ 个元素、$$N_t$$ 份。Korthikanti et al. 2022（Megatron 的激活重计算论文）算过，一层激活 $$sbh(34 + 5as/h)$$ 里，TP 切不到的部分是 $$10sbh$$，随 $$N_t$$ 增大它的占比越来越高。

**序列并行**（Megatron 意义下的 SP，注意与后面的 CP 区分）把这部分沿序列维切成 $$N_t$$ 份：LayerNorm、dropout、残差加法都是逐 token 的，切开序列不影响结果。切开后进入列切线性层前要把序列拼回来（**all-gather**），行切线性层的输出原本要 all-reduce、现在改成 **reduce-scatter**（归约的同时按序列切开），正好落回序列并行的布局：

```text
  无 SP：   LN ──── X（完整）───→ [列切] … [行切] ─── all-reduce ───→ + 残差 ─── LN
  有 SP：   LN ── X_i（s/t）── all-gather ──→ [列切] … [行切] ── reduce-scatter ──→ + 残差 ── LN
            ↑ 每卡 1/t                                                         ↑ 每卡 1/t
```

通信量：一次 all-gather $$\approx S$$ 加一次 reduce-scatter $$\approx S$$，等于一次 all-reduce 的 $$\approx 2S$$——**分文未加**。反向对称（all-gather 的反向是 reduce-scatter，reduce-scatter 的反向是 all-gather）。收益是那 $$10sbh$$ 也被 $$N_t$$ 均分，一层的激活变成 $$\frac{sbh}{N_t}(34 + 5as/h)$$，整层都被切了。因为不花钱，Megatron 里 SP 总是随 TP 一起开（`--sequence-parallel`），torchtitan 的 TP 也默认带 SP。它顺带还改变了 PP 的载荷：层边界处的张量现在是 $$sbh/N_t$$ 而不是 $$sbh$$，Megatron `megatron/core/pipeline_parallel/schedules.py` 的 `get_tensor_shapes()` 在 `sequence_parallel` 打开时把序列长除以 TP 大小，正是这一点。

### 5. 实现的位置

PyTorch 2.13.0 用 DTensor 表达 TP：`torch/distributed/tensor/parallel/style.py` 的 `ColwiseParallel` 把 `nn.Linear` 的权重按 `Shard(0)` 分布（PyTorch 的 `Linear` 存的是 $$A^T$$，第 0 维就是输出维，对应 Megatron 的列切）、输入为 `Replicate()`、输出为 `Shard(-1)`；`RowwiseParallel` 把权重按 `Shard(1)` 分布、输入 `Shard(-1)`、输出 `Replicate()`——从 `Shard(-1)` 到 `Replicate()` 的 redistribute 就是那次 all-reduce，由 DTensor 自动插入。`SequenceParallel` 让 LayerNorm / RMSNorm / Dropout 在序列维 `Shard(1)` 的输入上运行，与前后两个线性层之间的 all-gather / reduce-scatter 同样由 redistribute 生成。`api.py` 的 `parallelize_module()` 把一张 `{子模块名: ParallelStyle}` 的计划应用到模型上。这套实现的特点是**通信是从 placement 推导出来的，不是手写的**。

Megatron Core 0.18.0 则是手写的：`megatron/core/tensor_parallel/layers.py` 的 `ColumnParallelLinear`（`gather_output` 控制是否在输出处 all-gather）与 `RowParallelLinear`（`input_is_parallel` 表示输入已按列分布）；`mappings.py` 里每种通信是一个 `autograd.Function`——`_CopyToModelParallelRegion`（前向恒等、反向 all-reduce）、`_ReduceFromModelParallelRegion`（前向 all-reduce、反向恒等）这一对是无 SP 的 TP；`_GatherFromSequenceParallelRegion` 与 `_ReduceScatterToSequenceParallelRegion` 这一对是有 SP 的 TP。共轭关系在类名里就写明了。


## 四、上下文并行

### 1. 为什么 TP/SP 还不够

TP + SP 把一层的激活切成 $$1/N_t$$，但 $$N_t \le 8$$。当 $$s$$ 从 8K 长到 128K 时，激活线性增长 16 倍，注意力的 $$s^2$$ 项（即使有 FlashAttention 不物化分数矩阵，计算量仍是 $$s^2$$）增长 256 倍；单层 $$34sbh/8$$ 在 $$h = 16384$$、$$s = 131072$$ 时是 9 GB，80 层无论怎么重计算都放不下。需要一个能超过 8、能跨节点的维度来切序列——这就是**上下文并行**（CP）。

CP 与 SP 的区别：SP 只切层边界处那些逐 token 的算子，注意力本身仍在完整序列上算（all-gather 拼回来了）；CP 把**注意力本身**也沿序列切开，每张卡只持有 $$s/N_c$$ 个 token 的 Q/K/V，全程不拼回完整序列。困难在于注意力不是逐 token 的：每个 query 要看到全部 key/value。两种做法解决这个困难。

### 2. Ring Attention：K/V 沿环流动

Ring Attention（Liu et al. 2023）让每张卡固定持有自己那块 Q，把 K/V 块沿环传递：第 $$j$$ 步用来自第 $$(i-j) \bmod N_c$$ 张卡的 K/V 块算一个局部注意力，同时把手上的 K/V 块发给下一张卡、接收上一张卡的。$$N_c - 1$$ 步后每块 Q 看过了全部 K/V；局部结果用 online-softmax 的方式合并（与 FlashAttention 分块的合并方式相同）。

```text
  卡 0: Q_0  K/V_0 → K/V_3 → K/V_2 → K/V_1        每步：算 attn(Q_0, K/V_j)，同时收发下一块
  卡 1: Q_1  K/V_1 → K/V_0 → K/V_3 → K/V_2
  卡 2: Q_2  K/V_2 → K/V_1 → K/V_0 → K/V_3
  卡 3: Q_3  K/V_3 → K/V_2 → K/V_1 → K/V_0
```

通信是**点对点的 send/recv**，每步传一个 K/V 块。每卡每层：前向接收 $$N_c - 1$$ 个 K/V 块；反向再接收一遍 K/V（重算局部注意力需要）并传递累积的 dK/dV，约为前向的两倍。每个 K/V 块的大小是 $$\frac{s}{N_c} \cdot b \cdot 2 \cdot h_{kv} \cdot 2$$ 字节（K 和 V 各一份，$$h_{kv}$$ 是 K/V 的总维度：MHA 下等于 $$h$$，GQA 下是 $$\text{kv\_heads} \times \text{head\_dim}$$；再有 TP 时除以 $$N_t$$）。合计：

$$
V_{\text{ring}} \approx 3\,(N_c - 1)\cdot\frac{s\,b}{N_c}\cdot 2h_{kv}\cdot 2 \;\approx\; 12\,s\,b\,h_{kv}\ \text{字节／层}
$$

注意这个量**与 $$N_c$$ 几乎无关**：不论切成几份，每张卡都要把整个序列的 K/V 看一遍。它的好处在别处：通信是点对点、可以与当前块的注意力计算完全重叠（算第 $$j$$ 块时收第 $$j+1$$ 块），只要一块的注意力计算时间大于一块 K/V 的传输时间。所以 Ring Attention 可以跨节点、$$N_c$$ 可以很大。因果 mask 下还有一个负载均衡问题：靠后的 Q 块要算更多的 K/V 块，靠前的少；标准做法是把序列按"头尾配对"的方式分块（第 $$i$$ 张卡持有第 $$i$$ 块和第 $$2N_c - 1 - i$$ 块），让每张卡的计算量相等。

### 3. Ulysses：按头 all-to-all

DeepSpeed-Ulysses（Jacobs et al. 2023）换一个思路：注意力对**头**是独立的。每张卡持有 $$s/N_c$$ 个 token 的全部头，算 Q/K/V 投影后做一次 **all-to-all**，变成持有全部 $$s$$ 个 token 的 $$a/N_c$$ 个头——这时每张卡可以对自己的头做完整的、不需要任何通信的注意力；算完再 all-to-all 回到按序列切的布局，进输出投影。

```text
      按序列切 [s/c, b, a·d]  ── all-to-all(Q)、(K)、(V) ──→  按头切 [s, b, a/c·d]  ── 局部完整注意力
                                                                       │
      按序列切 [s/c, b, a·d]  ←──────── all-to-all(O) ────────────────┘
```

前向四次 all-to-all（Q、K、V、O），反向四次，每次载荷是一个激活张量 $$\frac{s}{N_c} b h$$：

$$
V_{\text{ulysses}} \approx 8\cdot\frac{N_c - 1}{N_c}\cdot\frac{s\,b\,h}{N_c}\cdot 2 \;\approx\; \frac{16\,s\,b\,h}{N_c}\ \text{字节／层}
$$

与 Ring 相反，Ulysses 的通信量随 $$N_c$$ **下降**——$$N_c$$ 越大每卡持有的序列越短，all-to-all 搬的就越少。但它有一个硬约束：$$N_c$$ 不能超过头数（GQA 下是 K/V 头数，否则要复制 K/V 头），而且 all-to-all 在关键路径上、$$N_c(N_c-1)$$ 条流同时打满链路，跨节点时对网络的压力比 ring 的点对点大得多。

### 4. 两者对比与 GQA 的影响

```text
                      Ring Attention                       Ulysses
──────────────────    ────────────────────────────────     ──────────────────────────────────
切的状态              Q/K/V/激活 沿 s 切 1/N_c               同，但注意力内部临时按头切
通信形态              send/recv 环，N_c - 1 步                all-to-all，每层 4 + 4 次
每卡每层通信量        ≈ 12 s b h_kv（与 N_c 无关）              ≈ 16 s b h / N_c
能否重叠              是（与分块注意力计算流水）               否（关键路径）
N_c 上限              无                                     ≤ 头数（GQA：≤ K/V 头数）
与 TP 组合            K/V 块再除以 N_t                         头数再除以 N_t，上限更紧
```

GQA 是分水岭。MHA 下 $$h_{kv} = h$$，Ring 每层 $$12sbh$$ 对 Ulysses 的 $$16sbh/N_c$$，$$N_c = 8$$ 时 Ring 多搬 6 倍；GQA 下 $$h_{kv}$$ 只有 $$h$$ 的 $$1/8$$ 到 $$1/16$$（Llama 3 405B：128 个头、8 个 K/V 头，$$h_{kv} = h/16$$），Ring 的通信量随之缩到 $$0.75\,sbh$$，反而比 Ulysses 少了；同时 Ulysses 的 $$N_c$$ 上限被压到 8 个 K/V 头再除以 $$N_t$$——TP = 8 时它只剩 1。所以 GQA + TP 的大模型上 Ring 是唯一可行的选择，Llama 3 用的正是它。放进五元组：CP **切激活（含注意力）；Ring 每层 $$\approx 12sbh_{kv}$$，Ulysses $$\approx 16sbh/N_c$$；可跨节点；Ring 可重叠、Ulysses 不可；长序列必需**。

PyTorch 2.13.0 里 CP 是实验 API：`torch/distributed/tensor/experimental/_context_parallel/_attention.py` 的 `context_parallel()` 上下文管理器把 SDPA 替换为 ring 版本（`_templated_ring_attention()`），K/V 块的传递方式有两种 `_RingRotater`——`_AllToAllRotater`（逐步点对点交换）与 `_AllGatherRotater`（一次 all-gather 全部 K/V，换通信量换步数），由 `set_rotate_method()` 选择；因果负载均衡在 `_load_balancer.py` 的 `_HeadTailLoadBalancer`。Megatron 的 CP 在 Transformer Engine 的注意力内核里实现，`parallel_state.py` 单独维护 CP 进程组，梯度归约用 `get_data_parallel_group(with_context_parallel=True)`——这一点第七章组合时会用到：**CP 的各卡持有同一份参数的副本，梯度要在 DP × CP 上归约**。


## 五、流水线并行

### 1. 按层切，用 micro-batch 填满

流水线并行把 $$L$$ 层切成 $$N_p$$ 段（stage），每段放在一组卡上，激活按顺序从 stage 0 流到 stage $$N_p - 1$$，梯度反向流回。状态上它与 TP 一样干净：参数、梯度、优化器状态都被切成 $$1/N_p$$，不需要通信来维持；激活方面，每张卡只持有自己 stage 的层的激活。通信只有 stage 边界处的激活（前向）与激活的梯度（反向），是**点对点** send/recv，载荷是一个层边界处的张量（$$sbh$$，有 SP 时 $$sbh/N_t$$）——所有并行维度里最小的通信量，而且不在关键路径上（下面讲的调度就是为了让它不在）。

代价是一个新的东西：**气泡**。一个 batch 从 stage 0 进入到 stage $$N_p - 1$$ 出来之前，后面的 stage 没事可做；反向同理。把 batch 切成 $$m$$ 个 micro-batch 依次送入，让不同 stage 同时处理不同的 micro-batch，才能填满流水线。

### 2. GPipe 与气泡率 (p − 1)/m 的推导

记 $$p = N_p$$，每个 micro-batch 在一个 stage 上前向耗时 $$t_f$$、反向 $$t_b$$（通常 $$t_b \approx 2t_f$$）。GPipe（Huang et al. 2019）的调度是先把 $$m$$ 个 micro-batch 的前向全部做完，再做全部反向：

```text
 p = 4, m = 4      时间 →
 stage 0   F0 F1 F2 F3 ·  ·  ·  ·  ·  ·  B3 B2 B1 B0
 stage 1   ·  F0 F1 F2 F3 ·  ·  ·  ·  B3 B2 B1 B0 ·
 stage 2   ·  ·  F0 F1 F2 F3 ·  ·  B3 B2 B1 B0 ·  ·
 stage 3   ·  ·  ·  F0 F1 F2 F3 B3 B2 B1 B0 ·  ·  ·
                    ↑ 前向填充 p-1 格          ↑ 反向排空 p-1 格
```

看任何一个 stage：它做了 $$m$$ 次前向和 $$m$$ 次反向，有用时间 $$m(t_f + t_b)$$。但整条流水线从第一个前向开始到最后一个反向结束的总时间，等于 stage 0 的时间线长度：前向阶段最后一个 micro-batch 要在 stage $$p-1$$ 做完，stage 0 才能开始反向，中间 stage 0 空等 $$(p-1)t_f$$；反向阶段对称，空等 $$(p-1)t_b$$。所以：

$$
T_{\text{total}} = m(t_f + t_b) + (p-1)(t_f + t_b), \qquad
\frac{T_{\text{bubble}}}{T_{\text{ideal}}} = \frac{(p-1)(t_f+t_b)}{m(t_f+t_b)} = \frac{p-1}{m}
$$

气泡占**总时间**的比例是 $$\frac{p-1}{m + p - 1}$$。$$p = 16$$、$$m = 16$$ 时是 48%——一半时间在等；要把它压到 10% 以下需要 $$m \ge 9(p-1) = 135$$。**气泡率只与 $$p$$ 和 $$m$$ 有关，与模型大小、卡的快慢无关**，这是 PP 的根本约束：$$m$$ 由 global batch 除以 $$N_d$$ 再除以 micro-batch 大小决定，不能随意加大；$$p$$ 由显存决定，不能随意减小。

### 3. 1F1B：不减气泡，减显存

GPipe 的另一个问题是显存：stage 0 在开始反向前要保存全部 $$m$$ 个 micro-batch 的激活。1F1B（PipeDream-Flush，Narayanan et al. 2021）在进入稳态后交替做一次前向、一次反向，让每个 micro-batch 的激活尽早被反向消费掉：

```text
 p = 4, m = 8
 stage 0   F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4 ·  B5 ·  B6 ·  B7
 stage 1   ·  F0 F1 F2 B0 F3 B1 F4 B2 F5 B3 F6 B4 F7 B5 ·  B6 ·  B7 ·
 stage 2   ·  ·  F0 F1 B0 F2 B1 F3 B2 F4 B3 F5 B4 F6 B5 F7 B6 ·  B7 ·  ·
 stage 3   ·  ·  ·  F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7 ·  ·  ·
           ←warm-up→←────────────── 稳态：1F1B ──────────────→←cool-down→
```

气泡的总量没有变（仍是前向填充 $$p-1$$ 格、反向排空 $$p-1$$ 格，$$\frac{p-1}{m}$$），但任一时刻每个 stage 在途的 micro-batch 数不超过 $$p$$（stage $$i$$ 是 $$p - i$$），激活显存从 $$O(m)$$ 降到 $$O(p)$$，与 $$m$$ 无关——这让 $$m$$ 可以放大去压气泡。1F1B 是所有生产框架的默认调度。

### 4. Interleaved 1F1B：用更多 stage 换更小的气泡

气泡 $$(p-1)(t_f + t_b)$$ 里的 $$t_f$$、$$t_b$$ 是**一个 stage** 的前向 / 反向时间。如果把每张卡上的层再切成 $$v$$ 段（virtual stage / model chunk），让流水线有 $$pv$$ 个 stage、每张卡轮流负责其中 $$v$$ 个不相邻的 stage，那么每个 stage 只有原来 $$1/v$$ 的层，$$t_f$$、$$t_b$$ 缩小 $$v$$ 倍，而填充和排空的格数由卡数 $$p$$ 决定不变：

$$
\frac{T_{\text{bubble}}}{T_{\text{ideal}}} = \frac{(p-1)(t_f + t_b)/v}{m(t_f + t_b)} = \frac{p-1}{v\,m}
$$

```text
 p = 2 卡, v = 2 → 4 个 stage：卡 0 持 stage {0, 2}，卡 1 持 stage {1, 3}
 卡 0   F0⁰ F1⁰ F0² F1² B1² B0² B1⁰ B0⁰ …      上标为 stage 编号；同一 micro-batch 在卡 0 上进出两次
 卡 1   ·   F0¹ F1¹ F0³ F1³ B1³ B0³ B1¹ B0¹ …
```

代价：stage 边界从 $$p - 1$$ 个变成 $$pv - 1$$ 个，PP 的点对点通信量乘以 $$v$$；调度更复杂；每张卡在途的激活也多了。Megatron 的 `--num-layers-per-virtual-pipeline-stage` 就是这个 $$v$$。$$p = 16$$、$$m = 16$$、$$v = 8$$ 时气泡从 48% 降到 10.5%。

### 5. Zero-bubble：把反向拆成两半

Qi et al. 2023 的观察是反向本身可以拆成两个独立的部分：对**输入**的梯度 $$B$$（下一个 stage 等的是它）和对**权重**的梯度 $$W$$（谁都不等它，只要在优化器步骤前算完）。1F1B 把两者绑在一起，$$W$$ 无谓地占据了关键路径。把 $$W$$ 拆出来、塞进原本的气泡里，气泡就能大幅缩小；再配合把优化器步骤的同步点后移，理论上可以做到零气泡（ZB-H2），代价是激活显存更高；ZB-H1 在与 1F1B 相同的显存下把气泡减到约三分之一。ZB-V 用 $$v = 2$$ 的 V 形 stage 分配（卡 $$i$$ 持 stage $$i$$ 和 $$2p - 1 - i$$）在 $$t_f = t_B = t_W$$ 的假设下达到零气泡；DeepSeek-V3 的 DualPipe 进一步让前向与反向的两个方向在流水线上同时流动，用双份参数换更多的重叠机会。

这些调度都不改变 PP 的状态放置与通信总量（DualPipe 的参数双份除外），只改变时间轴上的排列。它们的共同前提是 $$W$$ 可以延后——这要求框架把 `backward` 拆成 `backward_input` 与 `backward_weight` 两个可以分别调用的操作，是 `torch.distributed.pipelining` 与 Megatron 都专门做过的事。

### 6. 通信量与链路

PP 的通信量是所有维度里最小的：每个 micro-batch 在每个 stage 边界前向传一个 $$sbh/N_t$$（有 SP）的张量、反向传一个同样大小的梯度。Llama 3 405B、$$N_t = 8$$：一个张量 33.5 MB，16 个 micro-batch、前向 + 反向，每卡每 step **1 GB**，是 TP 的 1/220。它又是点对点、不在关键路径上（在 1F1B 稳态下 stage $$i$$ 发出前向激活后立即开始做一个反向，接收方也有自己的活干），所以它是最适合放在跨节点链路上、甚至跨机柜链路上的维度。放进五元组：PP **切参数、梯度、优化器状态、激活（按层）；每 step $$2m \cdot sbh/N_t$$；IB；可重叠；气泡率 $$(p-1)/(vm)$$ 要求 $$m \gg p$$**。

### 7. 实现的位置

PyTorch 2.13.0 的 `torch/distributed/pipelining/schedules.py` 把调度做成了可枚举的类：单 stage 每卡的 `ScheduleGPipe` 与 `Schedule1F1B`（基类 `PipelineScheduleSingle`）；多 stage 每卡的 `ScheduleInterleaved1F1B`、`ScheduleLoopedBFS`、`ScheduleInterleavedZeroBubble`（论文的 ZB-H1 / ZB1P）、`ScheduleZBVZeroBubble`（ZB-V，要求每卡恰好两个 stage）、`ScheduleDualPipeV`（基类 `PipelineScheduleMulti`，运行时是 `_PipelineScheduleRuntime`，它把调度表达成一张按时间步排列的动作表——前向、反向输入、反向权重、send、recv——逐步执行）；`get_schedule_class()` 按名字取类。每个 stage 是 `stage.py` 的 `PipelineStage`，负责 send/recv 与形状推断。Megatron 0.18.0 的 `megatron/core/pipeline_parallel/schedules.py` 是过程式写法：`forward_backward_pipelining_without_interleaving()` 是 1F1B，`forward_backward_pipelining_with_interleaving()` 是 interleaved 1F1B，`get_forward_backward_func()` 按 PP 与 VP 大小选择其一；点对点在 `p2p_communication.py`。两种写法的对照是下一篇的内容。


## 六、专家并行

### 1. MoE 的状态形态

MoE 把每层的 FFN 换成 $$E$$ 个专家，每个 token 由路由器选 $$k$$ 个（通常 $$k = 1$$ 或 2）专家处理。参数量随 $$E$$ 线性增长，但每个 token 的计算量只与 $$k$$ 有关——这是 MoE 的全部吸引力，也是它的状态形态与 dense 模型的全部差别：**参数很多、每个参数被很少的 token 用到**。

对放置来说这意味着：专家参数不能像 dense 参数那样靠 TP 切——一个专家的 FFN 矩阵本来就不大（与 dense 的 FFN 同尺寸），切成 8 份每份的 GEMM 太小、效率差；也不适合完全靠 ZeRO-3 切——每层前向要 all-gather 全部 $$E$$ 个专家的参数，但每张卡的 token 只用其中 $$k$$ 个。自然的方案是**专家并行**：把 $$E$$ 个专家分到 $$N_e$$ 张卡上，每卡 $$E/N_e$$ 个；专家参数、梯度、优化器状态各切 $$1/N_e$$，不需要通信维持；非专家部分（注意力、embedding）在 EP 组内是复制的，仍靠 DP/ZeRO 或 TP 切。

### 2. all-to-all 通信量

代价是 token 要去专家所在的卡。每层前向两次 **all-to-all**：dispatch 把每个 token 发给它选中的 $$k$$ 个专家所在的卡，combine 把专家的输出发回 token 所在的卡；反向再两次。每次的载荷是本卡的 token 数 × $$k$$ × $$h$$（每个 token 被复制 $$k$$ 份），每卡通信量 $$\frac{N_e - 1}{N_e}$$ 倍：

$$
V_{\text{EP}} \approx 4 \cdot \frac{N_e - 1}{N_e}\cdot \frac{s\,b}{N_c}\cdot k\,h \cdot 2\ \text{字节／层}
$$

与 TP 的每层 $$8sbh$$ 量级相同（$$k = 2$$ 时正好相等），但形态完全不同：TP 的 all-reduce 是环状流水化的、每步只与邻居通信；all-to-all 是 $$N_e(N_e - 1)$$ 条独立的流同时发生，跨节点时同时压满所有网卡，没有环可以借力。它也在关键路径上——专家的 GEMM 要等 token 到齐。所以 EP 与 TP 一样偏爱节点内，但它比 TP 更能容忍跨节点（可以按专家分块流水化，DeepSeek-V3 的 DeepEP 与 Megatron 的 `MoEFlexTokenDispatcher` 都在做这个），而 $$N_e$$ 常常需要大于 8（专家数 64 到 256 时每卡只放几个专家才划算）。

### 3. 负载不均：通信与计算的双重代价

上面的通信量假设 token 均匀分到各专家。路由器不保证这一点：热门专家收到的 token 可能是平均值的几倍。不均衡在两个地方付费：

- **通信**：all-to-all 的每卡时间由**收到最多 token 的那张卡**决定，其余卡等它；不均衡度 $$\rho$$（最忙专家的负载 / 平均负载）直接乘在通信时间上。
- **计算**：最忙的专家所在的卡要算 $$\rho$$ 倍的 GEMM，同一 step 内其他卡等它——这是一个结构性的 straggler，每层都发生。

两种缓解各有代价：**容量因子**（capacity factor）给每个专家设上限，超出的 token 被丢弃（不经过专家，走残差）或溢出到次选专家，通信和计算的上界确定了，但训练信号被截断；**辅助损失**（load balancing loss）鼓励路由器均匀分配，代价是与主目标冲突。Megatron 的 `megatron/core/transformer/moe/moe_utils.py` 里 `switch_load_balancing_loss_func()` 与 `get_capacity()` 分别对应这两种手段。放进五元组时 EP 的通信量要带上 $$\rho$$：**切专家的参数、梯度、优化器状态与路由后的激活；每层 $$4 \cdot \rho \cdot \frac{sb}{N_c} k h$$；all-to-all；不可重叠（可分块流水）；尽量节点内，负载均衡是前提**。

### 4. EP 与 DP/TP 的组合

EP 的进程组与 DP 组是**同一批卡的不同用法**：一个 EP 组的 $$N_e$$ 张卡处理的是 $$N_e$$ 个不同的 micro-batch（它们本来是 DP 副本），只是专家层在它们之间交换 token。所以 EP 不增加总卡数，$$N_e$$ 从 $$N_d$$ 里划出来：专家参数的 DP 组大小是 $$N_d / N_e$$（Megatron 称为 expert data parallel），非专家参数的 DP 组仍是 $$N_d$$。ZeRO 对两组参数分别按各自的 DP 组分片——这是本篇 `ledger/parallel.py` 里 MoE 那几行除法的来源。Megatron 0.18.0 的 `parallel_state.py` 里 EP 是 rank 排布 `"tp-cp-ep-dp-pp"` 中的一维，`megatron/core/transformer/moe/token_dispatcher.py` 里 `MoEAlltoAllTokenDispatcher` 是本节描述的 all-to-all 实现，`MoEAllGatherTokenDispatcher` 是 $$N_e$$ 很小时的替代（all-gather 全部 token、各卡挑自己专家的）。EP 与 TP 的组合（专家再做 TP）在 Megatron 里也支持，但如第 1 节所说通常不划算，多数 MoE 配置让专家层的 TP 为 1。


## 七、组合：多维并行与 Llama 3 405B

### 1. 五元组大表

把前五章收进一张表。通信量一栏是每卡每 step，$$N_{\text{local}} = N / (N_t N_p)$$ 是本卡持有的参数份额；"激活"指一个层边界处的张量 $$\frac{s}{N_c} b h$$ 个元素。

```text
策略         切哪种状态                        每 step 通信量（每卡）                        链路      重叠     适用条件
──────────   ───────────────────────────────  ──────────────────────────────────────────  ───────   ──────   ──────────────────────────────
DP (DDP)     无（切数据）                       all-reduce 2N_local                          IB        是       模型放得下一张卡；N_d 任意
ZeRO-1       优化器状态 1/N_d                   RS N + AG N = 2N_local                       IB        是       优化器状态是显存大头（74%）
ZeRO-2       + 梯度 1/N_d                      同上 2N_local                                IB        是       梯度需即时归约释放
ZeRO-3/FSDP  + 参数 1/N_d                      AG N + AG N + RS N = 3N_local                IB        是       带宽充足；不 reshard 则 2N
HSDP         参数/梯度/优化器 1/N_s              组内 3N_local（NVLink）+ 组间 AR 2N_local/N_s   NVLink+IB 是       中等模型，跨节点带宽紧
TP           参数/梯度/优化器 1/N_t + 层内激活    4 × AR(激活) × 层数/stage × m ≈ 8·sbh·L_s·m/N_c   NVLink    否       N_t ≤ 8；GEMM 够大
SP           层边界激活再切 1/N_t               与 TP 相同（AR → AG + RS）                     NVLink    否       总是随 TP 打开
CP (Ring)    激活（含注意力）1/N_c               3(N_c-1) × K/V 块 ≈ 12·sb·h_kv·L_s·m            IB        是       长序列；GQA 下极便宜
CP (Ulysses) 同上                              8 次 all-to-all ≈ 16·sbh·L_s·m/N_c            IB        否       N_c ≤ K/V 头数 / N_t
PP           参数/梯度/优化器/激活 按层 1/N_p     2 × 激活/N_t × m × v（send/recv）              IB        是       m ≫ p；气泡 (p-1)/(vm)
EP           专家参数/梯度/优化器 1/N_e           4 × ρ × all-to-all(token·k·h) × L_s × m        IB/NVLink 部分     MoE；负载均衡；N_e 从 N_d 划出
```

看这张表的两个维度。**按通信量**：TP 最大、DP/ZeRO 次之（但只与 $$N_{\text{local}}$$ 成正比、与 $$N_d$$ 无关）、CP 与 EP 视模型而定、PP 最小。**按链路要求**：TP 与 EP 在关键路径上必须快链路；CP（Ring）、PP、DP 都能重叠，可以跨节点。

### 2. 组合顺序：TP → CP → PP → DP 及原因

多维并行是这些策略的乘积，卡数 $$N = N_t N_c N_p N_d$$。选择每一维的大小有一个几乎固定的顺序，它来自上表的"链路"和"重叠"两栏：

```text
第一步  TP     受节点内卡数限制（≤ 8），受 GEMM 效率限制（h/N_t 不能太小）
               选 TP 是为了把一层放进一张卡（层内激活 + 该层参数），以及把层边界激活切 1/N_t（SP）
第二步  CP     序列长到 TP/SP 切完仍放不下一层激活时才需要；Ring 可跨节点，N_c 由 s 决定
第三步  PP     把 L 层的参数 + 优化器状态放进 N_t × N_p 张卡；越小越好（气泡），
               但受"每卡显存放得下 L/N_p 层的 16N/(N_t N_p) + 在途激活"约束
第四步  DP     用满剩余的卡：N_d = N / (N_t N_c N_p)；ZeRO-1 几乎总是打开（不花通信）
               如果 PP 放不下、或 N_p 太大气泡不可接受，用 ZeRO-3/FSDP 代替一部分 PP
```

为什么是这个顺序？TP 必须最内层，因为它的通信不可重叠、必须走 NVLink，节点内 8 张卡是唯一的候选。CP 其次：Ring 的 K/V 块流动可以跨节点，但它与 TP 组的相互作用最紧（K/V 块要除以 $$N_t$$），且它决定每卡的序列长度、进而决定后面所有维度的激活大小。PP 再外：它的通信量最小、可重叠、点对点，能放在最慢的链路上；它是**确定每卡显存能否放下参数**的最后一道闸。DP 最外：它不切模型，只是把整个模型（已被 TP/CP/PP 切好的一份）复制 $$N_d$$ 次；从状态的角度看，一个 DP 副本 = 一份完整模型，所以它逻辑上包着其他所有维度。

要区分两件事：**逻辑嵌套**与**物理 rank 排布**。逻辑上 DP 最外，但在把 rank 映射到物理卡时，框架把**通信量最大、最不能等的维度放在编号最相邻（同节点）的卡上**，通信量最小、最能等的维度放在最远的卡上——Megatron `parallel_state.py` 的 `initialize_model_parallel()` 默认 rank 顺序是 `"tp-cp-ep-dp-pp"`（`RankGenerator` 按它生成各进程组）：TP 变化最快（同节点），PP 变化最慢（最远），DP 在两者之间。也就是说物理上 PP 而不是 DP 被放到最外层的链路上，因为 PP 只有 1 GB 而 DP 有十几 GB。"TP 最内、DP 最外"说的是决策顺序与状态嵌套，"PP 最远"说的是链路分配——两者不矛盾，都是从上表推出来的。

### 3. Llama 3 405B 代入

Llama 3 论文（Dubey et al. 2024）给出的 405B 训练配置有三档（Table 5）：8K GPU 与 16K GPU 上 $$s = 8192$$，TP = 8、CP = 1、PP = 16、DP = 64 或 128；长上下文阶段 $$s = 131072$$，TP = 8、**CP = 16**、PP = 16、DP = 4（四维乘积 8192 张卡）。（本系列总纲把两档合写成 "TP=8/CP=16/PP=16/DP=128"，乘起来是 262144 张卡，不是 16384——正确的是上面两档；正文以论文为准。）模型：126 层、$$h = 16384$$、128 个头、8 个 K/V 头、词表 128256，$$N = 405 \times 10^9$$。

**状态放置**（16K GPU、$$s = 8192$$ 档，ZeRO-1 式的分布式优化器）：

```text
每卡参数份额       N_local = 405B / (8 × 16) = 3.16B
bf16 参数          2 × 3.16B = 6.3 GB
bf16 梯度          6.3 GB
优化器 (fp32 × 3)  12 × 3.16B / N_d = 38 GB / 128 = 0.3 GB           ← ZeRO-1 把 38 GB 切成 0.3 GB
常驻合计           ≈ 12.9 GB                                           （二进制单位 12.06 GiB）
每层激活（每卡）    34 × 8192 × 16384 / 8 = 570 MB（FlashAttention，SP 已切 1/N_t，b = 1）
每 stage 层数      126 / 16 ≈ 8 层；1F1B 下 stage 0 在途 ≤ 16 个 micro-batch → 峰值 8 × 16 × 570 MB ≈ 73 GB
```

最后一行说明了两件事：常驻状态只有 13 GB，80 GB 显存的大头是**在途激活**；以及为什么 405B 需要选择性重计算或更小的 $$v$$ 才能把激活压进 80 GB（第四篇）。

**通信量**（global batch 16M token → 每 step 2048 个序列，每个 DP 副本 16 个，$$b = 1$$ 则 $$m = 16$$；6N 算力 $$3.9 \times 10^{22}$$ FLOP，16384 × 989 TFLOPS 标称 × 41% MFU 下 step 约 5.9 s）：

```text
TP   4 × AR(268 MB) × 8 层 × 16 mb  =  4 × 470 MB × 126  ≈ 220 GB   NVLink   ← 220 GB / 5.9 s ≈ 37 GB/s 每卡平均
PP   2 × 33.5 MB × 16 mb            ≈  1.0 GB                IB
DP   RS(6.3 GB) + AG(6.3 GB)        ≈ 12.6 GB                IB       ← 与反向重叠；12.6 GB / 50 GB/s = 0.25 s ≪ 5.9 s
CP   —（N_c = 1）
```

长上下文档（$$s = 131072$$，CP = 16，每卡仍是 8192 个 token，$$m$$ 取 32 为例）：

```text
TP   不变的每卡序列长 → 每 mb 每层与上面相同，× 32 mb  ≈ 441 GB   NVLink
CP   K/V 块 = 8192 × 2 × (1024 / 8) × 2 B = 4 MB；3 × 15 × 4 MB × 8 层 × 32 mb ≈ 47 GB   IB，与注意力重叠
     （若用 Ulysses：8 × 15/16 × 33.5 MB × 8 × 32 ≈ 63 GB，且 N_c = 16 > K/V 头数 / N_t = 1，不可行）
PP   2 × 33.5 MB × 32                                                  ≈ 2.1 GB    IB
DP   N_d × N_c = 64 个副本，RS + AG 仍 ≈ 12.6 GB                                    IB
气泡 (16-1)/(32+15) = 32%（v = 1）；v = 8 时 (15/8)/(32+15/8) = 5.5%
```

CP 那一行是 GQA 的功劳：K/V 总维度 1024 只有 $$h$$ 的 1/16，再被 TP 切 8 份，每卡每块 K/V 只有 4 MB，16 倍的序列长度只多了 47 GB 的**可重叠**跨节点通信。这正是"CP 放在 TP 之外、PP 之内"的实例——它跨节点，但通信量和形态都比 DP 温和。

气泡一行则解释了为什么 405B 的 PP = 16 不是灾难：$$m = 16$$、$$v = 1$$ 的气泡是 48%，与论文报告的 38–43% MFU 不相容，所以实际调度里 stage 必须再切（interleaved，$$v > 1$$）或 $$m$$ 更大；论文第 3.3 节也确实描述了对流水线调度的修改。这是下一篇比较两种 1F1B 实现、第四篇讨论 micro-batch 与 $$v$$ 取值的起点。

### 4. 三框架对照

三个框架实现的是同一张表，但各自覆盖的格子与默认取向不同（源码细节在下一篇）：

```text
                 Megatron Core 0.18.0                    DeepSpeed 0.19.2                     torchtitan（PyTorch 原生 API）
──────────────   ─────────────────────────────────────   ──────────────────────────────────   ────────────────────────────────────
DP/ZeRO          DDP + 分布式优化器（ZeRO-1）；            ZeRO-1/2/3 全部；offload；             FSDP2 fully_shard（ZeRO-3）；
                 Megatron-FSDP（ZeRO-3）作为新选项          ZeRO++ 的分层与量化通信                 reshard_after_forward 调 2N/3N；HSDP 由 2D mesh
TP + SP          ColumnParallelLinear / RowParallelLinear  依赖 Megatron 的层或自带 autotp          ColwiseParallel / RowwiseParallel / SequenceParallel
                 手写 mappings.py 的通信                                                           通信由 DTensor redistribute 推导
CP               Transformer Engine 的 ring attention；    DeepSpeed-Ulysses（按头 all-to-all）     torch.distributed.tensor.experimental 的
                 dp-cp 组归约梯度                                                                  context_parallel（ring）
PP               schedules.py 的 1F1B / interleaved         pipe/ 引擎，1F1B                       torch.distributed.pipelining 的 Schedule* 类
                 过程式写法                                                                       声明式动作表
EP               token_dispatcher.py（all-to-all / all-gather / flex）  MoE 层 + expert parallel 组   实验性
进程组           parallel_state.py，order "tp-cp-ep-dp-pp"  groups.py                             DeviceMesh（device_mesh.py），维度命名
```

三者对同一格子的选择差异，多数可以从五元组解释：Megatron 长期只做 ZeRO-1 是因为 ZeRO-1 拿掉了 74% 的显存却不花通信，其余靠 TP/PP 解决；DeepSpeed 从 ZeRO-3 出发是因为它对用户模型零侵入；torchtitan 用 FSDP2 + DTensor 是因为 per-parameter 分片让各维度可以自由组合。

### 5. 原语 ↔ 并行对应表

最后把通信原语与并行策略的对应关系收成一张表，本系列只用到这五个原语的语义与通信量：

```text
原语             语义                                    每卡通信量        服务于
──────────────   ─────────────────────────────────────   ───────────────  ───────────────────────────────────────────
all-reduce       所有卡的 buffer 求和，结果每卡一份         ≈ 2S             DP 梯度；TP（无 SP）每层 2 + 2 次；HSDP 组间梯度
all-gather       每卡一段，拼成完整的一份给每卡             ≈ S              ZeRO-1/2 更新后的参数；ZeRO-3/FSDP 前向与反向前的参数；SP 进列切层前的激活
reduce-scatter   每卡一份完整输入，求和后每卡拿一段         ≈ S              ZeRO-1/2/3 的梯度；SP 行切层后的激活
all-to-all       第 i 卡的第 j 块发给第 j 卡（转置）         ≈ S；n(n-1) 条流  EP 的 token dispatch / combine；Ulysses 的 Q/K/V/O 转置
send / recv      一对一                                  S                PP 的 stage 边界激活与梯度；Ring Attention 的 K/V 块
```

一个记法：**all-reduce 给复制的状态用，all-gather / reduce-scatter 给分片的状态用，all-to-all 给按路由或按维度转置的激活用，send/recv 给流水和环用**。看到一个训练任务的通信 profile 里各原语的占比，就能反推它的并行配置。


## 八、本文小结

### 1. 要点回顾

- 并行策略是**状态的放置方案**：四种状态（参数、梯度、优化器状态、激活）× 六种切法，每一个"切"对应一种通信。DP 系（DP/ZeRO/FSDP）决定同一参数的副本之间如何分工，模型并行系（TP/PP/CP/EP）决定一份模型如何切开，两者正交，真实配置是乘积。
- **DP** 什么都不切、通信 $$2N$$，与 $$N_d$$ 无关、可与反向重叠。**ZeRO-1** 切优化器状态（拿掉 74% 显存）、**ZeRO-2** 再切梯度，通信仍是 $$2N$$（all-reduce = reduce-scatter + all-gather）。**ZeRO-3** 切参数，通信 $$3N$$：前向 all-gather 新增 $$N$$，反向 all-gather 新增 $$N$$，省掉优化器后的 all-gather $$N$$，净增 $$N$$；不 reshard 则回到 $$2N$$、显存等于 ZeRO-2。
- **FSDP1** 把一层拍平成 `FlatParameter`；**FSDP2** 每参数一个 `Shard(0)` 的 DTensor，通信次数相同、可组合性质变。**HSDP** 组内 ZeRO-3 走 NVLink、组间 all-reduce 只剩 $$2N/N_s$$，代价是显存只省 $$N_s$$ 倍。
- **TP** 列切 + 行切配对，每层前向 2 次、反向 2 次 all-reduce，载荷是激活 $$sbh$$；参数/梯度/优化器状态的切分免费；通信在关键路径上、不可重叠，所以锁在 NVLink 内、$$N_t \le 8$$。**SP** 把 all-reduce 拆成 all-gather + reduce-scatter，通信不变、层边界激活再切 $$1/N_t$$，总是随 TP 打开。
- **CP** 切注意力本身。Ring Attention 点对点传 K/V，每层 $$\approx 12sbh_{kv}$$、与 $$N_c$$ 无关、可重叠、可跨节点、$$N_c$$ 无上限；Ulysses 每层 8 次 all-to-all $$\approx 16sbh/N_c$$，随 $$N_c$$ 下降但 $$N_c \le$$ K/V 头数 $$/ N_t$$。GQA + TP 下 Ring 是唯一选择。
- **PP** 按层切，通信最小（$$2m$$ 个层边界张量，点对点、可重叠），代价是气泡 $$\frac{p-1}{m}$$（相对理想时间）、$$\frac{p-1}{m+p-1}$$（占总时间）。1F1B 不减气泡、把激活从 $$O(m)$$ 降到 $$O(p)$$；interleaved 用 $$v$$ 个 virtual stage 把气泡降到 $$\frac{p-1}{vm}$$、通信乘 $$v$$；zero-bubble 把反向拆成 $$B$$ 与 $$W$$、用 $$W$$ 填气泡。
- **EP** 切专家，每层 4 次 all-to-all、载荷 token × $$k$$ × $$h$$，$$N_e(N_e-1)$$ 条流、在关键路径上；负载不均 $$\rho$$ 同时乘在通信与计算上；$$N_e$$ 从 $$N_d$$ 里划出，专家参数的 DP 组是 $$N_d/N_e$$。
- **组合顺序** TP → CP → PP → DP：TP 不可重叠必须节点内；CP 决定每卡序列长；PP 是放下参数的最后一道闸、通信最小可放最远；DP 用满其余的卡、逻辑上包着一切。物理 rank 排布把 PP 放最远（Megatron 默认 `"tp-cp-ep-dp-pp"`）。
- **Llama 3 405B**（TP 8 / PP 16 / DP 128，$$s$$ = 8K）：每卡常驻 13 GB、在途激活可达 73 GB；每 step TP 220 GB NVLink、DP 12.6 GB IB、PP 1 GB IB；$$m = 16$$、$$v = 1$$ 的气泡 48% 与 38–43% MFU 不相容，stage 必须再切。长上下文档 CP = 16 只多 47 GB 可重叠的 IB 通信，GQA 是原因。

### 2. 本篇涉及的源码位置

| 路径 | 内容 |
|---|---|
| PyTorch 2.13.0 `torch/distributed/fsdp/_fully_shard/_fully_shard.py` | `fully_shard()`：per-parameter `Shard(0)` DTensor 分片；`mesh` 为 2D 时是 HSDP；`reshard_after_forward` 在 $$2N$$ 与 $$3N$$ 之间切换（可为整数：重分片到更小的组）；`FSDPModule` |
| PyTorch 2.13.0 `torch/distributed/fsdp/_fully_shard/_fsdp_param.py`、`_fsdp_param_group.py` | `FSDPParam`（单个参数的 sharded / unsharded 状态）；`FSDPParamGroup` 的 `unshard()` / `reshard()` / `post_backward()`；`FSDPCommContext` 的 all-gather / reduce-scatter / all-reduce 三条 stream |
| PyTorch 2.13.0 `torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py`、`_fsdp_api.py`、`_fsdp_common.py` | `foreach_all_gather()`、`foreach_reduce()`（一组参数拼成一次集合通信）；`MixedPrecisionPolicy`、`DataParallelMeshDims`；`HSDPMeshInfo` |
| PyTorch 2.13.0 `torch/distributed/fsdp/fully_sharded_data_parallel.py`、`_flat_param.py`、`api.py` | FSDP1：`FullyShardedDataParallel`、`FlatParameter` / `FlatParamHandle`、`ShardingStrategy`（`FULL_SHARD` / `SHARD_GRAD_OP` / `NO_SHARD` / `HYBRID_SHARD`） |
| PyTorch 2.13.0 `torch/distributed/tensor/parallel/style.py`、`api.py` | `ColwiseParallel`（权重 `Shard(0)`、输出 `Shard(-1)`）、`RowwiseParallel`（权重 `Shard(1)`、输入 `Shard(-1)`、输出 `Replicate()`）、`SequenceParallel`；`parallelize_module()` |
| PyTorch 2.13.0 `torch/distributed/pipelining/schedules.py` | `ScheduleGPipe`、`Schedule1F1B`（`PipelineScheduleSingle`）；`ScheduleInterleaved1F1B`、`ScheduleLoopedBFS`、`ScheduleInterleavedZeroBubble`、`ScheduleZBVZeroBubble`、`ScheduleDualPipeV`（`PipelineScheduleMulti` / `_PipelineScheduleRuntime`）；`get_schedule_class()` |
| PyTorch 2.13.0 `torch/distributed/tensor/experimental/_context_parallel/_attention.py`、`_load_balancer.py` | `context_parallel()`、`_templated_ring_attention()`、`_AllToAllRotater` / `_AllGatherRotater`、`set_rotate_method()`；`_HeadTailLoadBalancer` |
| PyTorch 2.13.0 `torch/distributed/device_mesh.py` | `DeviceMesh`、`init_device_mesh()`；`DeviceMesh.__getitem__` 按名取子 mesh、`_flatten()` 合并维度——HSDP 与多维组合的底座 |
| PyTorch 2.13.0 `torch/csrc/distributed/c10d/reducer.hpp` | `kDefaultBucketBytesCap`（DDP 25 MiB bucket） |
| Megatron Core 0.18.0 `megatron/core/tensor_parallel/layers.py` | `ColumnParallelLinear`（`gather_output`）、`RowParallelLinear`（`input_is_parallel`）、`LinearWithGradAccumulationAndAsyncCommunication`（反向 all-reduce 与权重梯度 GEMM 重叠） |
| Megatron Core 0.18.0 `megatron/core/tensor_parallel/mappings.py` | `_CopyToModelParallelRegion` / `_ReduceFromModelParallelRegion`（无 SP 的共轭对）；`_GatherFromSequenceParallelRegion` / `_ReduceScatterToSequenceParallelRegion`（SP 的共轭对） |
| Megatron Core 0.18.0 `megatron/core/parallel_state.py` | `initialize_model_parallel()` 默认 `order="tp-cp-ep-dp-pp"`、`RankGenerator`；`get_data_parallel_group(with_context_parallel=True)` |
| Megatron Core 0.18.0 `megatron/core/pipeline_parallel/schedules.py` | `forward_backward_pipelining_without_interleaving()`（1F1B）、`forward_backward_pipelining_with_interleaving()`、`get_forward_backward_func()`；`get_tensor_shapes()`（SP 时 PP 载荷除以 $$N_t$$） |
| Megatron Core 0.18.0 `megatron/core/transformer/moe/token_dispatcher.py`、`moe_utils.py` | `MoEAlltoAllTokenDispatcher`、`MoEAllGatherTokenDispatcher`、`MoEFlexTokenDispatcher`；`switch_load_balancing_loss_func()`、`get_capacity()` |
| train-ledger `ledger/parallel.py` | 本篇增量，见下 |

### 3. train-ledger 本篇增量：ledger/parallel.py

第一篇的账本给出一个模型的四类状态字节数；本篇给它加上并行维度：输入 `ParallelConfig(tp, pp, dp, cp, ep, zero_stage, …)`，输出每卡常驻状态与每 step 各并行维度、各原语的通信字节数，以及 PP 气泡率。它只依赖第一篇的 `ledger.model.ModelSpec` 与 `ledger.memory.state_bytes()`，不依赖 torch。

```python
"""train-ledger / ledger/parallel.py -- parallelism as state placement.

Input : ModelSpec + StateBytes (from article 1) + ParallelConfig
Output: per-GPU state bytes and per-step communication bytes, broken down
        by parallel dimension and by collective type.

Interface relied upon from article 1 (kept minimal on purpose):
  ledger.model.ModelSpec   fields: layers, hidden, heads, vocab, seq_len,
                           ffn_hidden, params  (optional: kv_heads)
  ledger.memory.StateBytes fields: params_bytes, grads_bytes, optim_bytes
  ledger.memory.state_bytes(model, precision, optimizer) -> StateBytes
No torch dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ledger.model import ModelSpec
from ledger.memory import StateBytes, state_bytes

# --------------------------------------------------------------------------
# collectives: bytes each rank sends (== receives) for a payload of S bytes.
# S follows the usual convention: all_reduce -> per-rank buffer; all_gather /
# reduce_scatter / all_to_all -> total (concatenated) size; send_recv -> message.
# --------------------------------------------------------------------------
def coll_bytes(kind: str, n: int, S: float) -> float:
    if n <= 1 and kind != "send_recv":
        return 0.0
    if kind == "all_reduce":
        return 2 * (n - 1) / n * S
    if kind in ("all_gather", "reduce_scatter", "all_to_all"):
        return (n - 1) / n * S
    if kind == "send_recv":
        return S
    raise ValueError(kind)


@dataclass
class MoESpec:
    num_experts: int = 1
    topk: int = 1
    expert_fraction: float = 0.0     # share of params living in experts
    capacity_factor: float = 1.0     # >1 models padding / imbalance (rho)


@dataclass
class ParallelConfig:
    tp: int = 1
    pp: int = 1
    dp: int = 1
    cp: int = 1
    ep: int = 1
    zero_stage: int = 0              # 0 = DDP, 1/2/3 = ZeRO-1/2/3 (3 == FSDP)
    sequence_parallel: bool = True   # Megatron SP inside the TP group
    cp_kind: str = "ring"            # "ring" | "ulysses"
    reshard_after_forward: bool = True   # ZeRO-3 / FSDP2 only
    hsdp_shard: int | None = None    # HSDP: shard group size (None = flat)
    micro_batch: int = 1
    num_microbatches: int = 1
    vp: int = 1                      # virtual stages per PP rank (interleaved 1F1B)
    act_bytes: int = 2               # bf16 activations
    moe: MoESpec = field(default_factory=MoESpec)

    @property
    def world(self) -> int:
        return self.tp * self.pp * self.dp * self.cp

    @property
    def grad_replicas(self) -> int:
        """ranks holding a replica of the same parameter shard: gradients are
        reduced over dp x cp (Megatron's dp-cp group)."""
        return self.dp * self.cp


@dataclass
class Comm:
    dim: str                         # "tp" | "cp" | "pp" | "dp" | "ep"
    kind: str                        # collective type
    link: str                        # "nvlink" | "ib"
    bytes: float                     # per GPU per step (sent)
    overlap: str                     # "yes" | "partial" | "no"


@dataclass
class Placement:
    per_gpu: StateBytes
    comms: list[Comm]
    bubble_fraction: float           # PP bubble as share of total step time
    tokens_per_gpu_per_mb: int

    def total_bytes(self, dim: str | None = None) -> float:
        return sum(c.bytes for c in self.comms if dim is None or c.dim == dim)

    def by_kind(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for c in self.comms:
            out[c.kind] = out.get(c.kind, 0.0) + c.bytes
        return out


def _shard_factors(cfg: ParallelConfig) -> tuple[float, float, float]:
    """(params, grads, optim) divisors from the ZeRO stage over the DP replicas."""
    n = cfg.grad_replicas if cfg.hsdp_shard is None else cfg.hsdp_shard
    p = n if cfg.zero_stage >= 3 else 1
    g = n if cfg.zero_stage >= 2 else 1
    o = n if cfg.zero_stage >= 1 else 1
    return p, g, o


def per_gpu_state(model: ModelSpec, st: StateBytes, cfg: ParallelConfig) -> StateBytes:
    """model-parallel (tp, pp, ep) divides every byte; ZeRO divides by stage."""
    fp, fg, fo = _shard_factors(cfg)
    mp = cfg.tp * cfg.pp
    dense, expert = 1.0 - cfg.moe.expert_fraction, cfg.moe.expert_fraction

    def split(total: float, zero_div: float) -> float:
        # expert params are additionally split by EP; their DP group is dp/ep
        d = total * dense / mp / zero_div
        e = total * expert / (mp * cfg.ep) / max(zero_div / cfg.ep, 1.0)
        return d + e

    return StateBytes(
        params_bytes=int(split(st.params_bytes, fp)),
        grads_bytes=int(split(st.grads_bytes, fg)),
        optim_bytes=int(split(st.optim_bytes, fo)),
    )


def activation_bytes_per_layer(model: ModelSpec, cfg: ParallelConfig, flash: bool = True) -> float:
    """Korthikanti et al. 2022: s*b*h*(34 + 5*a*s/h) bytes per layer, divided by
    TP (with SP the whole term) and by CP; FlashAttention drops the 5*a*s/h term."""
    s, b, h, a = model.seq_len, cfg.micro_batch, model.hidden, model.heads
    per = s * b * h * 34.0
    if not flash:
        per += 5.0 * a * s * s * b
    return per / (cfg.tp * cfg.cp)


def pp_bubble_fraction(p: int, m: int, vp: int = 1) -> float:
    """1F1B / GPipe: (p-1) idle slots against m useful ones per stage.
    Interleaved 1F1B with vp chunks shrinks each slot by 1/vp."""
    if p <= 1:
        return 0.0
    bubble = (p - 1) / vp
    return bubble / (m + bubble)


def place(model: ModelSpec, st: StateBytes, cfg: ParallelConfig) -> Placement:
    L_stage = model.layers / cfg.pp
    m = cfg.num_microbatches
    tok = model.seq_len * cfg.micro_batch // cfg.cp            # tokens per GPU per micro-batch
    S_act = tok * model.hidden * cfg.act_bytes                 # one activation tensor
    comms: list[Comm] = []

    # ---- TP: 2 collectives fwd + 2 bwd per layer on an activation-sized tensor
    if cfg.tp > 1:
        per_call = coll_bytes("all_reduce", cfg.tp, S_act)     # AG + RS pair costs the same
        total = 4 * per_call * L_stage * m
        if cfg.sequence_parallel:
            comms.append(Comm("tp", "all_gather", "nvlink", total / 2, "partial"))
            comms.append(Comm("tp", "reduce_scatter", "nvlink", total / 2, "partial"))
        else:
            comms.append(Comm("tp", "all_reduce", "nvlink", total, "no"))

    # ---- CP: ring passes K,V blocks (and dK,dV in bwd); Ulysses all-to-alls Q,K,V,O
    if cfg.cp > 1:
        kv_heads = getattr(model, "kv_heads", None) or model.heads
        head_dim = model.hidden // model.heads
        if cfg.cp_kind == "ring":
            kv_block = tok * 2 * (kv_heads * head_dim / cfg.tp) * cfg.act_bytes   # K and V, this rank's heads
            per_layer = 3 * (cfg.cp - 1) * kv_block            # fwd: K,V ; bwd: K,V + dK,dV
            comms.append(Comm("cp", "send_recv", "ib", per_layer * L_stage * m, "yes"))
        else:
            per_layer = 8 * coll_bytes("all_to_all", cfg.cp, S_act / cfg.tp)
            comms.append(Comm("cp", "all_to_all", "ib", per_layer * L_stage * m, "no"))

    # ---- PP: activations cross stage boundaries; with SP the tensor is 1/tp of S_act
    if cfg.pp > 1:
        S_pp = S_act / (cfg.tp if cfg.sequence_parallel else 1)
        comms.append(Comm("pp", "send_recv", "ib", 2 * S_pp * m * cfg.vp, "yes"))

    # ---- DP / ZeRO / FSDP / HSDP on this GPU's own (tp, pp) shard of the model
    n = cfg.grad_replicas
    local_params = st.params_bytes / (cfg.tp * cfg.pp)         # bf16 params on this rank
    local_grads = st.grads_bytes / (cfg.tp * cfg.pp)
    if n > 1:
        shard_n = cfg.hsdp_shard or n
        repl_n = n // shard_n
        if cfg.zero_stage == 0:
            comms.append(Comm("dp", "all_reduce", "ib", coll_bytes("all_reduce", n, local_grads), "yes"))
        else:
            comms.append(Comm("dp", "reduce_scatter", "ib", coll_bytes("reduce_scatter", shard_n, local_grads), "yes"))
            ag = 1 if cfg.zero_stage < 3 or not cfg.reshard_after_forward else 2
            comms.append(Comm("dp", "all_gather", "ib", ag * coll_bytes("all_gather", shard_n, local_params), "yes"))
            if repl_n > 1:                                     # HSDP: reduce across replica groups
                comms.append(Comm("dp", "all_reduce", "ib",
                                  coll_bytes("all_reduce", repl_n, local_grads / shard_n), "yes"))

    # ---- EP: dispatch + combine all-to-all, fwd and bwd, on routed tokens
    if cfg.ep > 1 and cfg.moe.num_experts > 1:
        routed = tok * model.hidden * cfg.act_bytes * cfg.moe.topk * cfg.moe.capacity_factor
        per_layer = 4 * coll_bytes("all_to_all", cfg.ep, routed)
        comms.append(Comm("ep", "all_to_all", "ib", per_layer * L_stage * m, "partial"))

    return Placement(per_gpu_state(model, st, cfg), comms,
                     pp_bubble_fraction(cfg.pp, m, cfg.vp), tok)


def fmt(b: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(b) < 1024 or unit == "TB":
            return f"{b:8.2f} {unit}"
        b /= 1024
    return f"{b:.2f} TB"


def report(model: ModelSpec, cfg: ParallelConfig, precision: str = "bf16", optimizer: str = "adam") -> None:
    st = state_bytes(model, precision, optimizer)
    pl = place(model, st, cfg)
    print(f"{model.name}: tp={cfg.tp} cp={cfg.cp} pp={cfg.pp} dp={cfg.dp} ep={cfg.ep} "
          f"zero={cfg.zero_stage} sp={cfg.sequence_parallel} mb={cfg.micro_batch} m={cfg.num_microbatches} "
          f"-> {cfg.world} GPUs")
    g = pl.per_gpu
    print(f"  per-GPU state : params {fmt(g.params_bytes)}  grads {fmt(g.grads_bytes)}  "
          f"optim {fmt(g.optim_bytes)}  total {fmt(g.params_bytes + g.grads_bytes + g.optim_bytes)}")
    print(f"  activation/layer/GPU (flash, 1 mb): {fmt(activation_bytes_per_layer(model, cfg))}"
          f"   layers/stage {model.layers / cfg.pp:.1f}   tokens/GPU/mb {pl.tokens_per_gpu_per_mb}")
    print(f"  PP bubble fraction: {pl.bubble_fraction:.3f}")
    for c in pl.comms:
        print(f"  {c.dim:3s} {c.kind:15s} {c.link:7s} {fmt(c.bytes)}  overlap={c.overlap}")
    print(f"  total per GPU per step: {fmt(pl.total_bytes())}"
          f"  (nvlink {fmt(sum(c.bytes for c in pl.comms if c.link == 'nvlink'))},"
          f" ib {fmt(sum(c.bytes for c in pl.comms if c.link == 'ib'))})")


if __name__ == "__main__":
    from ledger.model import llama3_405b, llama3_70b

    m405 = llama3_405b()
    # Llama 3 405B, 16K GPUs, s = 8192 (Table 5 of the Llama 3 paper): TP=8 CP=1 PP=16 DP=128
    report(m405, ParallelConfig(tp=8, cp=1, pp=16, dp=128, zero_stage=1,
                                micro_batch=1, num_microbatches=16))
    # long-context stage, s = 131072: TP=8 CP=16 PP=16 DP=4
    m405_long = llama3_405b(); m405_long.seq_len = 131072
    report(m405_long, ParallelConfig(tp=8, cp=16, pp=16, dp=4, zero_stage=1,
                                     micro_batch=1, num_microbatches=32))
    # same model, pure ZeRO-3 / FSDP over 16K GPUs
    report(m405, ParallelConfig(dp=16384, zero_stage=3, micro_batch=1, num_microbatches=1))
    # 70B on 1024 GPUs, three placements
    for cfg in (ParallelConfig(tp=8, pp=8, dp=16, zero_stage=1, micro_batch=1, num_microbatches=32),
                ParallelConfig(tp=8, pp=4, dp=32, zero_stage=1, micro_batch=1, num_microbatches=16, vp=2),
                ParallelConfig(dp=1024, zero_stage=3, hsdp_shard=64, micro_batch=1, num_microbatches=1)):
        report(llama3_70b(), cfg)
```

几点实现说明。通信量按第一章的口径计算：`coll_bytes` 返回每卡**发出**的字节数，all-reduce 用 $$2(n-1)/n$$、其余三个集合原语用 $$(n-1)/n$$。TP 的 all-gather + reduce-scatter 对与一次 all-reduce 等价，所以 SP 打开时只是把同一份量拆成两行记。CP 的 K/V 块用 `kv_heads`（GQA）并除以 TP；`ModelSpec` 若没有 `kv_heads` 字段则退化为 MHA。DP 一段用的是本卡的 $$N_{\text{local}}$$——TP/PP 切了之后 DP 组内交换的只是本卡那一份；`grad_replicas` 是 $$N_d \times N_c$$，因为 CP 各卡持有同一份参数。HSDP 通过 `hsdp_shard` 打开，多出一行组间 all-reduce。气泡率返回的是占总时间的份额 $$\frac{(p-1)/v}{m + (p-1)/v}$$。这里的激活估算只给一层、一个 micro-batch 的数，在途 micro-batch 数与重计算策略是第四篇的内容。

运行输出（`python -m ledger.parallel`，`ledger/model.py` 里的 `llama3_405b()` / `llama3_70b()` 是第一篇定义的模型规格，`kv_heads=8`；字节数为二进制单位）：

```text
llama3-405b: tp=8 cp=1 pp=16 dp=128 ep=1 zero=1 sp=True mb=1 m=16 -> 16384 GPUs
  per-GPU state : params     5.91 GB  grads     5.91 GB  optim   283.49 MB  total    12.09 GB
  activation/layer/GPU (flash, 1 mb):   544.00 MB   layers/stage 7.9   tokens/GPU/mb 8192
  PP bubble fraction: 0.484
  tp  all_gather      nvlink    110.25 GB  overlap=partial
  tp  reduce_scatter  nvlink    110.25 GB  overlap=partial
  pp  send_recv       ib          1.00 GB  overlap=yes
  dp  reduce_scatter  ib          5.86 GB  overlap=yes
  dp  all_gather      ib          5.86 GB  overlap=yes
  total per GPU per step:   233.22 GB  (nvlink   220.50 GB, ib    12.72 GB)
llama3-405b: tp=8 cp=16 pp=16 dp=4 ep=1 zero=1 sp=True mb=1 m=32 -> 8192 GPUs
  per-GPU state : params     5.91 GB  grads     5.91 GB  optim   566.97 MB  total    12.37 GB
  activation/layer/GPU (flash, 1 mb):   544.00 MB   layers/stage 7.9   tokens/GPU/mb 8192
  PP bubble fraction: 0.319
  tp  all_gather      nvlink    220.50 GB  overlap=partial
  tp  reduce_scatter  nvlink    220.50 GB  overlap=partial
  cp  send_recv       ib         44.30 GB  overlap=yes
  pp  send_recv       ib          2.00 GB  overlap=yes
  dp  reduce_scatter  ib          5.81 GB  overlap=yes
  dp  all_gather      ib          5.81 GB  overlap=yes
  total per GPU per step:   498.92 GB  (nvlink   441.00 GB, ib    57.92 GB)
llama3-405b: tp=1 cp=1 pp=1 dp=16384 ep=1 zero=3 sp=True mb=1 m=1 -> 16384 GPUs
  per-GPU state : params    47.25 MB  grads    47.25 MB  optim   283.49 MB  total   377.98 MB
  activation/layer/GPU (flash, 1 mb):     4.25 GB   layers/stage 126.0   tokens/GPU/mb 8192
  PP bubble fraction: 0.000
  dp  reduce_scatter  ib        755.91 GB  overlap=yes
  dp  all_gather      ib          1.48 TB  overlap=yes
  total per GPU per step:     2.21 TB  (nvlink     0.00 B, ib     2.21 TB)
llama3-70b: tp=8 cp=1 pp=8 dp=16 ep=1 zero=1 sp=True mb=1 m=32 -> 1024 GPUs
  per-GPU state : params     2.05 GB  grads     2.05 GB  optim   788.50 MB  total     4.88 GB
  activation/layer/GPU (flash, 1 mb):   272.00 MB   layers/stage 10.0   tokens/GPU/mb 8192
  PP bubble fraction: 0.179
  tp  all_gather      nvlink    140.00 GB  overlap=partial
  tp  reduce_scatter  nvlink    140.00 GB  overlap=partial
  pp  send_recv       ib          1.00 GB  overlap=yes
  dp  reduce_scatter  ib          1.93 GB  overlap=yes
  dp  all_gather      ib          1.93 GB  overlap=yes
  total per GPU per step:   284.85 GB  (nvlink   280.00 GB, ib     4.85 GB)
llama3-70b: tp=8 cp=1 pp=4 dp=32 ep=1 zero=1 sp=True mb=1 m=16 -> 1024 GPUs
  per-GPU state : params     4.11 GB  grads     4.11 GB  optim   788.50 MB  total     8.98 GB
  activation/layer/GPU (flash, 1 mb):   272.00 MB   layers/stage 20.0   tokens/GPU/mb 8192
  PP bubble fraction: 0.086
  tp  all_gather      nvlink    140.00 GB  overlap=partial
  tp  reduce_scatter  nvlink    140.00 GB  overlap=partial
  pp  send_recv       ib          1.00 GB  overlap=yes
  dp  reduce_scatter  ib          3.98 GB  overlap=yes
  dp  all_gather      ib          3.98 GB  overlap=yes
  total per GPU per step:   288.96 GB  (nvlink   280.00 GB, ib     8.96 GB)
llama3-70b: tp=1 cp=1 pp=1 dp=1024 ep=1 zero=3 sp=True mb=1 m=1 -> 1024 GPUs
  per-GPU state : params     2.05 GB  grads     2.05 GB  optim    12.32 GB  total    16.43 GB
  activation/layer/GPU (flash, 1 mb):     2.12 GB   layers/stage 80.0   tokens/GPU/mb 8192
  PP bubble fraction: 0.000
  dp  reduce_scatter  ib        129.36 GB  overlap=yes
  dp  all_gather      ib        258.73 GB  overlap=yes
  dp  all_reduce      ib          3.85 GB  overlap=yes
  total per GPU per step:   391.94 GB  (nvlink     0.00 B, ib   391.94 GB)
```

每一组都对应正文的一处。前两组是第七章第 3 节的 Llama 3 两档配置，TP 的 220 GB、DP 的 12.6 GB、PP 的 1 GB、CP 的 47 GB（二进制 44.3 GiB）、48% 与 32% 的气泡都从这里来。第三组是"如果 405B 只用 FSDP"：常驻状态只有 377 MB，但一层激活 4.25 GB × 126 层根本放不下，而且每卡每 step 要在 IB 上搬 2.2 TB——$$3N$$ 的通信作用在**整个模型**上，而 TP × PP = 128 把它缩到了 1/128；这就是为什么 ZeRO-3 不能替代模型并行去训 405B。后三组是 70B 在 1024 卡上的三种放置：TP 8 / PP 8 与 TP 8 / PP 4 的 NVLink 流量一样（TP 通信只与总层数和 micro-batch 数有关），差别在 PP 的气泡（18% 对 8.6%，后者用了 $$v = 2$$）和每卡显存（4.9 GB 对 9.0 GB）；纯 HSDP（分片组 64 卡）常驻 16.4 GB 放得下，但 IB 上 392 GB 的通信是前两者的 40–80 倍——它能否被反向藏住，取决于第四篇要测的重叠效率。

这个脚本给出的是**每卡显存与通信量**，还没有时间。把通信量换成时间需要链路带宽与重叠效率，把气泡率换成 MFU 需要 micro-batch 大小与重计算策略——这些是第四篇的内容。而在此之前，先要看清三个框架各自是怎么把本篇的每一行变成代码的。

> **一个 bf16 参数在 Megatron-LM、DeepSpeed、torchtitan 里各自存在哪里、什么时候被 all-gather、什么时候被释放、它的 fp32 主副本在哪张卡上？**


## 下一篇

[三个框架：Megatron-LM、DeepSpeed 与 torchtitan 的架构对比与源码导读](/megatron-deepspeed-torchtitan-architecture-and-source-guide.html)
