---
layout: post
title: 大规模训练工程：从 Megatron 到容错（总纲）
subtitle: Large-Scale Training Engineering, from Megatron to Fault Tolerance
tags: [Megatron, DeepSpeed, Distributed Training, AI, AI-Infra]
catalog: true
---


## 内容简介

《大规模训练工程：从 Megatron 到容错》是一组共八篇的系列文章，面向已经会用 PyTorch 写训练循环、了解 DDP、准备把训练任务从几张卡放大到几百几千张卡的工程师，系统讲解一个大规模训练任务是如何被**配置、跑满、并且长期跑住**的。

它回答的问题是：

> **一个千卡训练任务，怎么配、怎么跑满、怎么跑一个月不倒？**

这三个问题对应三种完全不同的工程：

```text
怎么配     显存账 · 并行策略 · 框架选型          —— 静态的资源规划问题
怎么跑满   MFU · 通信/计算重叠 · micro-batch      —— 稳态的性能工程问题
怎么不倒   checkpoint · 容错 · 稳定性 · 可观测    —— 长时的可靠性工程问题
```

单卡训练只有第一个问题的影子（模型放得下吗），多卡训练把第二个问题带进来（通信是不是白等了），而千卡级、持续几周的训练把第三个问题变成主角：Llama 3 405B 在 16K 张 H100 上训练，54 天的统计窗口里发生了 466 次任务中断，其中 419 次是意外中断——平均每三小时一次。在这个尺度上，"能跑起来"和"能跑完"之间隔着本系列的后半部分。

系列的组织原则来自训练引擎自身的结构：**训练引擎围绕状态组织**。一个训练任务的全部状态就是四样东西——参数、梯度、优化器状态、激活值。所谓并行策略，是决定这四样东西**放在哪张卡上**；所谓 checkpoint，是把其中三样**写到持久存储**；所谓容错，是在一部分卡消失后**重建它们**；所谓稳定性，是保证它们的**数值不跑飞**。把这四样东西的字节数、位置、生命周期追踪清楚，就是理解训练引擎的全部。

系列以三个框架为源码阅读对象：**Megatron-LM**（Megatron Core）、**DeepSpeed** 和 **torchtitan**。它们代表了三种不同的设计取向——Megatron 从模型结构出发切分，DeepSpeed 从优化器状态出发切分，torchtitan 用 PyTorch 原生的 DTensor 与 FSDP2 把切分做成可组合的——对照着读，比只读其中一个更容易看清"哪些是必然的，哪些是选择"。


## 为什么写这个系列？

### 训练任务的成本让"跑满"成为硬指标

一千张 H100 每小时的租金以万美元计。同一个任务，MFU（Model FLOPs Utilization）从 35% 提到 45%，意味着同样的预算多训练 30% 的 token，或者提前一周结束。这不是"锦上添花"的优化，而是训练团队每天都在盯的数字。

但 MFU 不是调几个参数就能上去的。它取决于并行维度的搭配、micro-batch 的大小、激活重计算的策略、通信与计算的重叠程度、数据加载是否跟得上、以及有没有一张卡在拖慢所有人。这些因素互相牵制：TP 越大通信越多，PP 越深气泡越大，激活重计算省显存但多算 30% 的前向。要在这个多维空间里找到好的配置，靠试是试不出来的，必须先会算。本系列第一到四篇就是教这套算法。

### 千卡任务的故障是常态，不是异常

一张 GPU 的年故障率也许只有百分之几，但一千张卡放在一起，故障率是它的一千倍。Llama 3 论文给出的数字是：16K 张 H100，54 天，419 次意外中断，其中约 78% 归因于确认或疑似的硬件问题，GPU 本身（含 HBM）占了 58.7%。同一篇论文报告他们仍然维持了超过 90% 的有效训练时间，而且只有 3 次需要显著的人工介入。

这两组数字之间的差距，就是容错工程的价值：checkpoint 写得够快够频繁、故障能被自动检测、任务能被自动重启、坏卡能被自动隔离、重启后数据顺序和随机状态能精确恢复。每一环都是具体的工程问题，有具体的数学（多久存一次 checkpoint 最优？）和具体的源码（`torch.distributed.checkpoint` 的异步保存是怎么实现的？）。本系列第五、六篇讨论这些。

### 三个框架的取舍值得对照着看

Megatron-LM、DeepSpeed、torchtitan 都能把一个几百亿参数的模型训起来，但它们回答"状态放在哪"这个问题的方式不同：

```text
Megatron-LM    按模型结构切：TP 切矩阵、PP 切层、SP/CP 切序列，DP 在最外层
               强项是极致性能与 NVIDIA 硬件特性的紧密集成（Transformer Engine、FP8）
DeepSpeed      按优化器状态切：ZeRO-1/2/3 逐级分片优化器状态、梯度、参数
               强项是对用户模型代码的低侵入与 offload 能力
torchtitan     用 PyTorch 原生原语组合：DTensor、FSDP2、TP、PP、CP 都是 torch.distributed 的公开 API
               强项是可读性与可组合性，是 PyTorch 分布式功能的参考实现
```

这三种取向不是谁对谁错，而是不同约束下的不同选择。只读一个框架，容易把它的选择当成必然；对照着读，才能分清哪些是分布式训练本身的要求（例如优化器状态一定要以 fp32 存在某处），哪些是框架的历史包袱或设计偏好。

### 现有材料的断层

- **论文**（Megatron-LM 的三篇、ZeRO、PaLM、Llama 3、torchtitan）给出了方法和数字，但不讲怎么配、怎么排障；
- **框架文档**告诉你每个参数是什么意思，不告诉你在给定模型和集群下应该选什么；
- **博客和教程**大多停在"用 DeepSpeed 训一个 7B 模型"，很少触及 checkpoint 格式、故障率数学、straggler、silent data corruption；
- **一线团队的经验**散落在论文的附录、issue 的讨论和内部文档里，没有被系统整理。

本系列想填补的是从"能在 8 卡上跑通 FSDP"到"能为一个千卡任务做出配置决策、并对它的长期运行负责"之间的那段路。


## 适合哪些读者？

### 准备把训练任务放大的工程师

你已经在 8 张或几十张卡上用 DDP、FSDP 或 DeepSpeed 训过模型，任务能跑。现在模型更大、卡更多、时间更长，你需要回答：用哪种并行、每个维度多少、显存够不够、MFU 能到多少、checkpoint 怎么存、挂了怎么恢复。本系列是为这个阶段准备的。

### 训练平台与训练基础设施团队

你负责的不是某一个模型，而是让所有训练任务在集群上跑得快、跑得稳。你需要理解引擎对资源层提出了什么要求——checkpoint 的 I/O 模式、故障检测的信号、弹性伸缩的接口、需要采集哪些指标——才能设计出合适的调度、存储和监控。本系列第四到八篇是直接材料。

### 想读懂 Megatron / DeepSpeed / torchtitan 源码的开发者

这三个项目的核心目录是本系列的源码阅读线。读完后你应该能在 `megatron/core/` 或 `torchtitan/` 里定位任何一个并行策略、checkpoint 策略或训练循环环节的实现，并有能力修改它。

### 做训练性能分析与调优的工程师

你需要判断一个训练任务的 MFU 为什么只有 30%：是气泡、是通信没重叠、是激活重计算过度、是数据加载拖后腿、还是某张卡慢。本系列的显存账、通信量估算和可观测方法是这项工作的工具。


## 系列的整体主线

八篇文章按"先算清、再配好、再跑稳"的顺序推进：

```text
第一篇：训练任务的状态解剖 —— 显存账与 MFU 定义
        ↓  四种状态的字节数 · 每参数 16–18 字节 · 激活的公式 · MFU 与 HFU
第二篇：并行策略全景 —— 每种并行切的是哪种状态，通信量多少
        ↓  DP/ZeRO/FSDP · TP · PP · SP/CP · EP · 通信量与适用条件
第三篇：三个框架 —— Megatron-LM、DeepSpeed、torchtitan 的架构对比与源码导读
        ↓  进程组组织 · 状态的实际存放 · 训练循环 · 各自的取舍
第四篇：千卡配置实战 —— 并行搭配、micro-batch、激活重计算、MFU 调优
        ↓  从模型与集群规格推出配置 · 逐项排除 MFU 损失
第五篇：分布式 checkpoint —— 格式、异步保存、重分片恢复
        ↓  状态如何落盘 · 存多久一次 · 换并行配置怎么加载
第六篇：容错与弹性 —— 故障率数学、straggler、SDC、弹性训练
        ↓  MTBF 与有效训练时间 · torchrun 与 torchft · 坏卡检测与隔离
第七篇：训练稳定性与数据管线 —— loss spike、梯度范数、数据混合、流式加载
        ↓  数值不跑飞 · 数据顺序可恢复 · 数据加载不成为瓶颈
第八篇：长时训练的可观测与运维 —— 指标、日志、hang 排查、值班手册
           看什么 · 怎么告警 · 卡住了怎么查
```

前四篇是**静态与稳态**：任务开始前怎么算、怎么配，跑起来后怎么跑满。后四篇是**长时**：几周的运行中会发生什么、怎么让它们不致命。

四条交织的线索：

```text
状态线：参数 · 梯度 · 优化器状态 · 激活值 —— 在哪张卡、多少字节、什么时候存、怎么恢复
算账线：显存账 → 通信量 → MFU → checkpoint 间隔 → 有效训练时间
框架线：Megatron-LM · DeepSpeed · torchtitan 在每个主题下的实现对照
运维线：日志与指标 → 故障检测 → 自动恢复 → 值班手册
```


## 章节结构与分章导读

### 1. 训练任务的状态解剖：显存账与 MFU

第一篇建立全系列的记账框架。它不讨论任何并行策略，只回答一个问题：**一个训练任务在一张卡上占多少字节、每秒做多少 FLOP，以及这两个数字的理论上限是什么？**

这一篇会讨论：

- 训练的四种状态：参数、梯度、优化器状态、激活值，以及它们在一个 step 内的生命周期——哪些常驻、哪些在前向产生反向消费、哪些只在优化器步骤出现；
- 混合精度下的字节数推导：bf16 参数 2 字节 + bf16 梯度 2 字节 + fp32 主参数 4 字节 + Adam 的 fp32 一阶矩 4 字节 + 二阶矩 4 字节 = **每参数 16 字节**；若梯度以 fp32 累加再多 2 字节到 18 字节；一个 70B 模型光这部分就是 1.1–1.3 TB，任何一张卡都放不下——这就是所有并行策略的出发点；
- 激活值的估算：Transformer 每层的激活字节数近似为 $$sbh\left(34 + 5\frac{as}{h}\right)$$（序列长 $$s$$、批大小 $$b$$、隐藏维 $$h$$、注意力头数 $$a$$），来自 Megatron 的激活重计算论文；其中随 $$s^2$$ 增长的项来自注意力分数矩阵，使用 FlashAttention 后基本消失；
- 显存之外的开销：CUDA context、NCCL buffer、PyTorch Caching Allocator 的碎片与保留、临时 buffer——为什么 80 GB 的卡实际可用只有 70 多 GB；
- 算力账：每个参数每个 token 前向 2 FLOP、反向 4 FLOP，合计 $$6N$$；注意力部分随序列长度增加的额外项；
- MFU 的定义（来自 PaLM 论文）：观测到的吞吐（token/s）与在硬件峰值 FLOPS 下的理论最大吞吐之比，**不计**激活重计算的额外 FLOP；HFU（Hardware FLOPs Utilization）则计入重计算——为什么开了重计算后 HFU 会好看而 MFU 不会；
- 当前的参考水平：Megatron-LM 论文在 A100 集群上报告过 52% 的 MFU，Llama 3 405B 在 8K–16K 张 H100 上报告 38%–43%；千卡 H100 上 dense 模型做到 40% 以上是好成绩，做不到 30% 说明有明确的问题；
- Megatron-LM 的 `megatron/training/theoretical_memory_usage.py` 与 `training.py` 里的 FLOPs 计算函数：把公式与源码对上。

核心问题是：

> **一个 70B 参数、序列长 8192 的模型，在一张 80 GB 的 H100 上，参数与优化器状态要多少字节？一层的激活要多少？每个 token 要多少 FLOP？把它们与 80 GB 和 989 TFLOPS 放在一起，就知道后面每一篇要解决的是什么。**

实践：写一个显存与 MFU 计算器脚本，输入模型结构（层数、隐藏维、头数、词表、序列长）、批大小、精度、优化器和硬件规格，输出各类状态的字节数与 step 时间下限。它是全系列的第一件工具，后面每一篇都会给它加功能。

### 2. 并行策略全景：每种并行切的是哪种状态

第二篇按"切哪种状态、代价是什么通信"的框架把所有并行策略过一遍。它把并行策略当作**状态的放置方案**而不是一组 API。

这一篇会覆盖：

- 数据并行（DP）：复制全部状态、切数据；梯度 all-reduce 的通信量为每 step 约 $$2\Psi$$ 字节每卡（$$\Psi$$ 为参数字节数）；梯度 bucket 与反向计算的重叠；
- ZeRO 的三级分片：Stage 1 切优化器状态、Stage 2 再切梯度、Stage 3 再切参数；每卡显存从 $$16\Psi$$ 降到 $$16\Psi/N_d$$，代价是 Stage 3 的通信量从 $$2\Psi$$ 涨到 $$3\Psi$$（前向 all-gather、反向 all-gather 与 reduce-scatter）；
- FSDP 作为 ZeRO-3 的 PyTorch 原生实现：FSDP1 的 FlatParameter 与 FSDP2（`fully_shard`）的 per-parameter DTensor 分片；HSDP（Hybrid Sharded Data Parallel）在节点内分片、节点间复制的动机；
- 张量并行（TP）：切矩阵的行或列，Megatron 式的 column-parallel 与 row-parallel 线性层配对；每层前向两次 all-reduce、反向两次；为什么 TP 几乎只在 NVLink 域内（8 卡）使用；
- 序列并行（SP，Megatron 意义下）：把 TP 域内 LayerNorm 与 dropout 的激活沿序列切开，all-reduce 变成 all-gather 加 reduce-scatter，通信量不变、激活显存下降；
- 上下文并行（CP）：沿序列维切注意力本身，Ring Attention 与 DeepSpeed-Ulysses 两种做法的通信形态；长上下文训练为什么必须有它；
- 流水线并行（PP）：按层切、以 micro-batch 流水；GPipe、1F1B、interleaved 1F1B、zero-bubble 调度；气泡率 $$\frac{p-1}{m}$$ 的推导（$$p$$ 为 stage 数、$$m$$ 为 micro-batch 数）；PP 的通信量最小（只传相邻 stage 之间的激活），所以放在跨节点的慢链路上；
- 专家并行（EP）：MoE 的专家分布在不同卡上，token 通过 all-to-all 路由；EP 与 DP、TP 的组合方式；负载不均衡对通信和计算的双重影响；
- 多维并行的组合顺序：为什么通常是 TP 最内（NVLink）、CP 次之、PP 再外、DP 最外；Llama 3 405B 的 TP=8、CP=16、PP=16、DP=128 是这套逻辑的一个实例；
- 通信原语与各并行的对应表：all-reduce、all-gather、reduce-scatter、all-to-all、点对点 send/recv 各服务于谁——本系列只用到它们的语义和通信量，不讨论其实现。

核心问题是：

> **每种并行都在"复制"和"切分"之间做交换：复制多占显存，切分多花通信。给定一个模型和一个集群的拓扑（节点内 NVLink、节点间 InfiniBand），每个维度的通信量是多少、走哪条链路、和计算能不能重叠？这决定了它该放在几维并行的哪一层。**

实践：给计算器加上并行维度：输入 TP/PP/DP/CP/EP 和 ZeRO 级别，输出每卡显存与每 step 各类通信量。

### 3. 三个框架：Megatron-LM、DeepSpeed、torchtitan 的架构对比与源码导读

第三篇进入源码。三个框架实现了第二篇的同一套策略，这一篇看它们各自怎么组织进程组、怎么存放状态、怎么写训练循环。

这一篇会覆盖：

- **Megatron-LM / Megatron Core**：`megatron/core/parallel_state.py` 如何初始化并管理 TP、PP、DP、CP、EP 各个进程组；`tensor_parallel/layers.py` 与 `mappings.py` 的 column/row-parallel 线性层与通信原语；`pipeline_parallel/schedules.py` 与 `p2p_communication.py` 的调度与点对点通信；`distributed/distributed_data_parallel.py` 与 `param_and_grad_buffer.py` 的梯度 bucket 与重叠；`optimizer/distrib_optimizer.py` 的分布式优化器（Megatron 版的 ZeRO-1）；`megatron/training/training.py` 的训练循环；
- **DeepSpeed**：`deepspeed/runtime/engine.py` 的 `DeepSpeedEngine` 如何包裹用户模型；`deepspeed/runtime/zero/stage_1_and_2.py` 与 `stage3.py` 的分片实现；`partition_parameters.py` 如何在 Stage 3 下用 hook 在前向前 all-gather 参数、前向后释放；`deepspeed/runtime/pipe/` 的流水线引擎；JSON 配置驱动的设计及其代价；
- **torchtitan**：`torchtitan/train.py` 与 `trainer.py` 的训练循环；`torchtitan/distributed/parallel_dims.py` 用 `DeviceMesh` 定义多维并行；`fsdp.py`、`tensor_parallel.py`、`pipeline_parallel.py`、`context_parallel/` 各自对 PyTorch 原生 API 的调用；`torchtitan/components/` 里的 checkpointer、optimizer、metrics、data；
- 三者的共同底座——PyTorch 分布式：`torch/distributed/device_mesh.py` 的 DeviceMesh、`torch/distributed/tensor/` 的 DTensor 及其 `parallel/` 子包（TP 的原生实现）、`torch/distributed/fsdp/_fully_shard/` 的 FSDP2、`torch/distributed/pipelining/` 的 `stage.py` 与 `schedules.py`；
- 对照阅读：同一个"前向时把分片参数 all-gather 回来"的动作，在 DeepSpeed Stage 3、FSDP1、FSDP2 里分别是怎么实现的，差别在哪；同一个 1F1B 调度在 Megatron 与 `torch.distributed.pipelining` 里的两种写法；
- 各自的取舍：Megatron 的性能与侵入性（模型必须用它的层写）、DeepSpeed 的易用与调试难度、torchtitan 的清晰与功能覆盖面；以及 Megatron 近年把 FSDP 引入 `megatron/core/distributed/fsdp/` 所反映的趋势。

核心问题是：

> **一个 bf16 参数在这三个框架里各自存在哪里、什么时候被 all-gather、什么时候被释放、它的 fp32 主副本在哪张卡上？把这条链追清楚，三个框架的架构差异就全部显现了。**

实践：在 8 卡上分别用三个框架跑同一个小模型，用 `torch.cuda.memory_stats` 与 profiler 验证每卡显存与通信模式是否与第一、二篇的计算一致。

### 4. 千卡配置实战：并行搭配、micro-batch、激活重计算与 MFU 调优

第四篇把前三篇的知识变成决策过程：给定模型规格与集群规格，推出一组配置，然后逐项排查 MFU 损失。

这一篇会覆盖：

- 配置的推导顺序：先定 TP（受限于节点内卡数与显存）→ 再定 PP（把参数与优化器状态放下）→ 再定 DP（用满剩余的卡）→ 再定 CP（序列长度需要时）；每一步用第一、二篇的计算器验证显存与通信；
- global batch、micro-batch 与梯度累积：global batch 由算法决定，micro-batch 由显存与 PP 气泡共同决定，梯度累积步数是二者之商；micro-batch 过小导致 GEMM 效率低，过大导致激活放不下或 PP 气泡大；
- 激活重计算的三种策略：全量重计算（多花约一次前向的算力）、选择性重计算（只重算注意力部分，Megatron 论文的主张）、按层重计算若干层；`torch.utils.checkpoint` 与 Megatron `--recompute-*` 参数族；激活 offload 到 CPU 的场景；
- 通信与计算的重叠：DP 梯度 reduce 与反向计算的重叠、TP 通信与 GEMM 的重叠、PP 的点对点与计算的重叠；重叠失败的常见原因（bucket 太小、stream 同步、CPU 侧发射太慢）；
- MFU 损失的逐项拆解：PP 气泡、通信未重叠、重计算、数据加载等待、kernel 效率（GEMM 形状不友好）、CPU 开销、straggler；用 profiler 时间线给每一项标价；
- 融合与低精度：Transformer Engine 的 FP8、融合的 LayerNorm/RoPE/交叉熵——它们在 MFU 上的贡献与代价；
- `torch.compile` 在训练中的位置：torchtitan 的 per-block compile；与 FSDP2、TP 的兼容性；
- 长时任务的配置纪律：配置写进版本控制、变更留痕、每次改动前后各跑一段基准。

核心问题是：

> **一个 70B dense 模型，1024 张 H100（128 节点 × 8 卡），序列长 8192，global batch 4M token。TP、PP、DP 各多少？micro-batch 多大？要不要重计算？预期 MFU 多少？跑出来只有 32%，缺的 10 个点去了哪里？**

实践：用 torchtitan 在 8 卡上跑 Llama 3 8B，对比不同 TP/FSDP/重计算配置的 MFU，用 profiler 找到每种配置的主要损失项；再用计算器把结论外推到 1024 卡的配置。

### 5. 分布式 checkpoint：格式、异步保存与重分片恢复

第五篇讲状态的持久化。checkpoint 是容错的基础，它的两个核心指标是**保存时间**（决定能存多频繁）和**恢复的灵活性**（能不能换并行配置加载）。

这一篇会覆盖：

- checkpoint 里有什么：模型参数、优化器状态（是参数的两到三倍）、学习率调度器、数据加载器的位置、随机数状态、迭代计数；缺任何一样都无法精确恢复；
- 单文件 checkpoint 为什么不可行：405B 模型的完整状态约 6 TB，写一次要多久；rank 0 汇总的模式在千卡下的瓶颈；
- 分片 checkpoint：每个 rank 写自己持有的分片；`torch.distributed.checkpoint`（DCP）的模型——`torch/distributed/checkpoint/` 下的 `state_dict_saver.py`、`planner.py`、`filesystem.py`、`metadata.py`：Planner 决定谁写什么、Storage 负责 I/O、metadata 记录每个张量的全局形状与分片位置；
- 重分片（resharding）：DCP 加载时按 metadata 与当前分片布局的交集读取，允许保存与加载用不同的 TP/PP/DP；`resharding.py`；这是能在故障后用不同卡数恢复的前提；
- Megatron 的 `megatron/core/dist_checkpointing/`：ShardedTensor 抽象、`strategies/` 下的多种后端、`optimizer.py` 对分布式优化器状态的处理、与 DCP 格式的互通；DeepSpeed 的 `deepspeed/runtime/checkpoint_engine/` 与 ZeRO checkpoint 的 universal checkpoint 转换；
- 异步保存：先把状态拷到 CPU pinned memory（staging），再由后台线程或进程写入存储，训练继续；DCP 的 `async_save` 与 `staging.py`、`_async_executor.py`；Megatron 的 `--async-save`；异步保存的显存与主机内存代价；
- 多级存储：本地 NVMe 作为一级、并行文件系统或对象存储作为二级；本地 checkpoint 让重启从邻居节点拿数据而不是从远端存储；
- 存多久一次：Young 公式 $$\tau_{opt} \approx \sqrt{2\delta M}$$（$$\delta$$ 为一次 checkpoint 的开销时间、$$M$$ 为平均故障间隔）；把 Llama 3 的每三小时一次故障代入，得出异步保存为什么是必需的；
- checkpoint 的校验与清理：写完后的完整性校验、保留策略、与训练框架版本的兼容性。

核心问题是：

> **一个 405B 模型、6 TB 状态的 checkpoint，同步写要停训练几分钟？异步写代价是什么？故障后剩 15 个节点而不是 16 个，能不能直接加载？**

实践：给 torchtitan 训练加 DCP 异步 checkpoint，测量保存对 step 时间的影响；用不同的 FSDP/TP 配置加载同一份 checkpoint，验证 loss 曲线精确衔接。

### 6. 容错与弹性：故障率数学、straggler、SDC 与弹性训练

第六篇讨论长时训练最主要的敌人：硬件会坏，而且会以各种方式坏。

这一篇会覆盖：

- 故障率的数学：单卡的 MTBF 除以卡数就是集群的 MTBF；有效训练时间 = 有用训练时间 / 总时间，由故障频率、检测时间、重启时间、checkpoint 间隔和回退损失共同决定；Llama 3 论文的 419 次意外中断、78% 硬件、58.7% GPU、超过 90% 有效训练时间、3 次人工介入——把这些数字放进公式里；
- 故障的分类：显式故障（进程崩溃、XID 错误、NCCL 报错）、隐式故障（hang、变慢）、静默故障（结果错但没有任何报错）；三类需要完全不同的检测手段；
- 检测：NCCL watchdog 与 `TORCH_NCCL_*` 超时参数；心跳；训练循环级的 step 时间告警；框架层的 rank 监控（NVIDIA Resiliency Extension 的 `fault_tolerance` 模块、Megatron 的 `megatron/training/ft_integration.py`）；
- 重启：`torchrun`（`torch/distributed/run.py`）与 `torch/distributed/elastic/` 的 agent、rendezvous（`elastic/rendezvous/` 下的 c10d 与 etcd 后端）、`--max-restarts`；进程组重建的成本；进程内重启（in-process restart，NVIDIA Resiliency Extension 的 `inprocess` 模块、Megatron 的 `inprocess_restart.py`）如何跳过进程拉起和 CUDA 初始化，把重启时间从分钟压到秒；
- 弹性训练：卡数变化后怎么继续——torchft 的思路：把 DP 的各个副本组视为可独立失败的单元，用 Lighthouse 做 quorum，一组挂了其余组继续，坏组恢复后从活着的组拉参数；它与 DDP/FSDP/HSDP 的配合；torchtitan 的 `experiments/torchft/`；LocalSGD/DiLoCo 等异步方法在容错语境下的位置；
- 坏卡隔离：从故障日志到节点的排除列表；`gpu_sniff_test.py` 这类启动前的硬件自检；调度器的配合（本系列只讲引擎侧提出的要求）；
- straggler：一张慢卡拖慢整个同步训练；成因（降频、HBM 错误纠正、PCIe 降速、CPU 侧数据加载慢、网络链路差）；检测方法（每 rank 的计算时间与通信等待时间对比，Megatron 的 straggler 检测与 NVIDIA Resiliency Extension）；Meta 2025 年关于 straggler 的 what-if 分析论文给出的结论：不少 straggler 不是硬件问题而是数据与调度不均衡；
- silent data corruption（SDC）：GPU 算错但不报错，损失是错误的梯度污染全部副本；Llama 3 统计中 SDC 占 1.4%；检测方法——冗余计算、Megatron 的 `megatron/core/rerun_state_machine.py` 通过重跑一个 iteration 比对结果来发现不确定性、周期性的确定性校验；`fault_injector.py` 用于演练；
- 确定性：可复现的训练是所有校验的前提；`torch.use_deterministic_algorithms`、NCCL 的算法选择、数据顺序的可复现。

核心问题是：

> **一千张卡平均每几小时坏一张。每次故障从发现到恢复训练要多久？其中检测、重启、加载 checkpoint、回退重算各占多少？把有效训练时间从 85% 提到 95%，最该缩短的是哪一段？**

实践：在 8 卡训练中注入故障（kill 一个 rank、人为拖慢一个 rank、篡改一个梯度），验证检测与恢复路径；用 torchrun 的弹性重启和 torchft 的副本组分别演练。

### 7. 训练稳定性与数据管线：loss spike、梯度范数、数据混合与流式加载

第七篇讨论硬件之外的两类长时问题：数值上模型会跑飞，数据上管线要跟得上、并且可恢复。两者被放在一起，因为 loss spike 的排查往往最终落到"那个 batch 里有什么"。

这一篇会覆盖：

- loss spike 的现象与成因：突然的 loss 跳升、有时自行恢复有时发散；成因包括学习率过高、bf16 精度不足、注意力 logit 增长、坏数据、优化器状态的时间性问题；
- 预防手段：梯度裁剪（范数阈值通常为 1.0，`megatron/core/optimizer/clip_grads.py` 里跨 TP/PP/DP 的全局范数计算）、warmup、z-loss、QK-LayerNorm、embedding 不做 weight decay、fp32 累加与 fp32 优化器状态；
- 处理手段：PaLM 论文的做法——回退到 spike 前约 100 步的 checkpoint，跳过之后的 200–500 个 batch；这依赖第五篇的 checkpoint 与本篇的数据顺序可复现；
- 必须记录的数值信号：loss、梯度范数、参数范数、学习率、loss scale（fp16 时）、跳过的迭代数、梯度中零的数量（Megatron 的 `num_zeros_in_grad`）、注意力 logit 的最大值；它们的正常形态与异常形态；
- tokenization 与预处理：离线 tokenize 成二进制索引格式（Megatron 的 `megatron/core/datasets/indexed_dataset.py` 及配套的预处理脚本），还是在线 tokenize；两者对 CPU、存储与可复现性的影响；
- 数据混合：多来源按权重采样（`blended_dataset.py`、`blended_megatron_dataset_builder.py`）；多阶段训练中权重的变化；混合比例如何在 checkpoint 中记录以便恢复；
- shuffle 与打包：epoch 级 shuffle 索引的生成与保存；文档打包成定长序列，`packed_seq_params.py` 与 cu_seqlens 让注意力不跨文档；序列长度课程（DeepSpeed 的 `data_pipeline/`）；
- 流式加载：数据在对象存储上时的流式读取（Megatron datasets 的对象存储支持、torchtitan 的 `components/data/` 与 `hf_datasets/`、Hugging Face datasets 的 streaming、MosaicML StreamingDataset）；预取深度、worker 数与 pinned memory；
- 可恢复的数据加载器：恢复后必须从精确的位置继续、不重复不遗漏；有状态的 DataLoader 与 checkpoint 中数据位置的记录；多 rank 下的一致性；
- 数据加载对 MFU 的影响：如何判断是数据在等 GPU 还是 GPU 在等数据。

核心问题是：

> **第 137,000 步 loss 从 2.1 跳到 4.8。是数据、学习率、还是数值精度？要回答这个问题，需要哪些信号在事前就被记录下来，需要哪些状态能被精确回放？**

实践：给 torchtitan 训练加上完整的数值信号记录；构造一个坏 batch 触发 spike，用 checkpoint 回退与跳过 batch 演练恢复；验证数据加载器在中断恢复后的顺序精确一致。

### 8. 长时训练的可观测与运维：从指标到 hang 排查

最后一篇讲把一个训练任务当作**服务**来运维：看什么、怎么告警、出问题怎么查、值班手册长什么样。

这一篇会覆盖：

- 三层指标：任务层（loss、梯度范数、学习率、token/s、TFLOPS/GPU、MFU、step 时间及其抖动）、进程层（各 rank 的计算时间与通信等待时间、显存峰值与碎片、CPU 与数据加载队列）、硬件层（GPU 温度、功耗、频率、XID 错误、HBM 纠错计数、网络错包与重传）；
- 指标采集的实现：Megatron 的 `timers.py` 与日志、torchtitan 的 `components/metrics.py`、`torch.cuda.memory_stats`、DCGM 导出的硬件指标；写入 TensorBoard/W&B 与 Prometheus 的分工；
- 每 rank 的可见性：千卡下必须能按 rank 看数据，否则找不到 straggler 和异常节点；日志的 rank 标记、聚合与抽样；
- hang 的排查：hang 是长时训练最常见也最难查的故障——所有进程都活着但没有进展；NCCL Flight Recorder（`torch/distributed/flight_recorder/` 与 c10d 的 `FlightRecorder.hpp`，通过 `TORCH_NCCL_TRACE_BUFFER_SIZE`、`TORCH_NCCL_DUMP_ON_TIMEOUT` 启用）记录每个集合通信的发起与完成状态，从中找出哪个 rank 没有到达哪次通信；`py-spy dump` 全部 rank 看 Python 栈；`TORCH_DISTRIBUTED_DEBUG`；常见成因（某 rank 的条件分支不一致、数据加载器在某 rank 上耗尽、死锁的点对点通信）；
- 性能回归的排查：step 时间慢慢变长的常见原因（显存碎片、日志过多、数据加载队列、温度降频）；对比 profiler 时间线；只在少数 rank 上开 Nsight Systems 的做法；
- 告警设计：什么该叫醒人（loss NaN、连续故障、有效训练时间跌破阈值、checkpoint 失败），什么只需记录；
- 运维流程：开训前的检查清单（硬件自检、NCCL 带宽测试、小规模 dry-run、checkpoint 恢复演练）、值班手册、故障复盘模板；
- 成本视角：把每一次故障、每一个 MFU 百分点换算成 GPU 小时与金额，用它决定优化的优先级。

核心问题是：

> **凌晨三点告警：step 时间从 12 秒变成 40 秒，没有报错。十分钟内你要判断是 straggler、数据、通信、还是硬件降频。你需要的每一个信号，在开训前有没有采集？**

实践：为练手项目搭起完整的指标面板与告警规则；用 Flight Recorder 排查一次人为制造的 hang；写出这个项目的值班手册。本篇最后给出全系列总结。


## 贯穿全系列的实践线

系列的练手项目是**一套"训练账本"（ledger）工具与一个从 8 卡长到可外推千卡的 Llama 风格训练配置**。选这个形态是因为千卡集群不是每个读者都能拿到的，但计算与方法是可以在 8 卡上验证、再用数学外推的。项目的每一部分对应一篇：

```text
第一篇    显存与 MFU 计算器              四类状态字节数 · 6N 算力 · step 时间下限
第二篇    计算器加并行维度               TP/PP/DP/CP/EP · ZeRO 级别 · 每 step 通信量
第三篇    三框架对照运行                 同一小模型 · 显存与通信模式与计算对账
第四篇    8 卡 Llama 3 8B 调优           配置矩阵 · MFU 对比 · profiler 拆解 · 外推到 1024 卡
第五篇    DCP 异步 checkpoint            保存开销测量 · 换并行配置加载 · loss 曲线衔接
第六篇    故障注入与恢复演练             kill rank · 拖慢 rank · 篡改梯度 · torchrun 与 torchft
第七篇    稳定性信号与数据管线           数值信号记录 · 坏 batch 演练 · 数据顺序精确恢复
第八篇    指标面板、hang 排查与值班手册   三层指标 · Flight Recorder · 告警规则
```

到第八篇结束，读者手上有：一个能对任意模型与集群规格给出显存、通信量与 MFU 预期的计算器；一份在 8 卡上验证过、并有外推依据的训练配置；一套包含 checkpoint、故障恢复、数值监控和告警的运行方案。它不是一个千卡任务，但它是接手一个千卡任务时需要带的全部东西。

与它平行的源码阅读线：

```text
第一篇    Megatron-LM  megatron/training/theoretical_memory_usage.py · training.py 的 FLOPs 计算
第二篇    PyTorch  torch/distributed/fsdp/_fully_shard/ · torch/distributed/tensor/parallel/ · torch/distributed/pipelining/schedules.py
第三篇    Megatron-LM  megatron/core/parallel_state.py · tensor_parallel/ · pipeline_parallel/ · distributed/ · optimizer/distrib_optimizer.py
          DeepSpeed    deepspeed/runtime/engine.py · zero/stage_1_and_2.py · zero/stage3.py · zero/partition_parameters.py · pipe/
          torchtitan   torchtitan/train.py · trainer.py · distributed/parallel_dims.py · distributed/{fsdp,tensor_parallel,pipeline_parallel}.py
第四篇    torchtitan  distributed/activation_checkpoint.py · compile.py；PyTorch  torch/utils/checkpoint.py；Megatron-LM  megatron/core/pipeline_parallel/schedules.py
第五篇    PyTorch  torch/distributed/checkpoint/{state_dict_saver,planner,filesystem,resharding,staging}.py
          Megatron-LM  megatron/core/dist_checkpointing/ · megatron/training/checkpointing.py；torchtitan  components/checkpointer/
第六篇    PyTorch  torch/distributed/run.py · torch/distributed/elastic/agent/server/ · elastic/rendezvous/
          torchft  torchft/manager.py · process_group.py；Megatron-LM  megatron/core/rerun_state_machine.py · megatron/training/{ft_integration,inprocess_restart}.py
第七篇    Megatron-LM  megatron/core/optimizer/clip_grads.py · megatron/core/datasets/{indexed_dataset,blended_dataset,gpt_dataset}.py
          torchtitan  components/data/；DeepSpeed  deepspeed/runtime/data_pipeline/
第八篇    PyTorch  torch/distributed/flight_recorder/ · torch/csrc/distributed/c10d/FlightRecorder.hpp
          Megatron-LM  megatron/core/timers.py；torchtitan  components/metrics.py
```


## 阅读路径建议

### 完整学习路径

```text
1 → 2 → 3 → 4 → 5 → 6 → 7 → 8
```

### 只关心"怎么配、怎么跑满"

```text
1 → 2 → 3 → 4
```

前四篇自成一段。读完能为一个任务做出配置决策并诊断 MFU 损失，但不涉及长时运行。

### 训练平台与基础设施团队

```text
1 → 2 → 5 → 6 → 8
```

第一、二篇建立状态与通信的概念；第五、六、八篇是引擎对存储、调度、监控提出的全部要求。第三、四篇可以只读 torchtitan 部分。

### 已经在跑长时任务、被故障与 spike 困扰

```text
5 → 6 → 7 → 8
```

后四篇可以独立阅读，前提是已经理解自己任务的并行配置和状态分布。

### 主要目标是读懂框架源码

```text
1 → 2 → 3 → 5
```

第三篇是主体；第一、二篇提供读它所需的概念，第五篇覆盖三个框架中最容易被忽视的 checkpoint 子系统。


## 本系列的边界

本系列只讨论训练引擎这一层：状态如何被切分、放置、持久化、恢复与监控。以下内容与它紧邻，但不在范围内：

- **集合通信的实现**：NCCL 的算法（Ring、Tree）、protocol、channel、拓扑探测、RDMA 与 GPUDirect、nccl-tests 与调优参数。本系列只使用 all-reduce、all-gather、reduce-scatter、all-to-all、send/recv 的**语义和通信量**，把它们的耗时当作由链路带宽决定的黑盒。
- **kernel 内部**：GEMM、attention、融合算子的实现与优化。本系列把 kernel 效率当作 MFU 损失的一个来源，讨论如何测量它的份额，不讨论如何提高它。
- **模型的算量与显存推导**：Transformer 每一层的 FLOP 与激活字节数的逐项推导。本系列第一篇直接使用它们的结论公式（$$6N$$、激活估算式），不从矩阵形状重新推导。
- **算法与训练配方**：学习率、batch size 的选择、数据配比的算法依据、评测。本系列第七篇讨论 loss spike 的**工程**处理与数据管线的**工程**实现，不讨论它们对模型质量的影响。
- **集群资源层**：Kubernetes/Slurm 的调度、gang scheduling、GPU 切分、并行文件系统与对象存储的选型与配置。本系列第五、六篇给出引擎对它们的要求，不讨论它们自身。
- **推理引擎**：训练完成的模型如何被服务。
- **后训练特有的系统问题**：RLHF 中生成与训练的协同、多模型的调度。本系列以预训练为主线，SFT 在系统层面与预训练同构。


## 前置要求与说明

### 前置要求

- 会用 PyTorch 写完整的训练循环：模型、优化器、学习率调度、混合精度（`torch.autocast`、bf16）；
- 用过 `DistributedDataParallel`，知道 `torchrun` 怎么启动多进程、rank 与 world size 是什么、进程组是什么；
- 知道 all-reduce、all-gather、reduce-scatter、all-to-all 的语义（不要求了解实现）；
- 了解 Transformer decoder 的结构：embedding、注意力、MLP、LayerNorm 各有什么参数；
- 能读 Python 源码，能用 `torch.profiler` 或 Nsight Systems 看时间线；
- 至少一台 8 卡 GPU 机器（Ampere 或更新）用于实践；千卡级的内容以计算外推和公开数据为主。

不要求：

- 用过 Megatron-LM、DeepSpeed 或 torchtitan；
- 有千卡任务的运维经验；
- 了解 NCCL 内部或 CUDA 编程。

### 框架与版本基线

- PyTorch 2.x（2.4 及之后；正文源码片段取自 v2.13.0 源码树）；重点使用 FSDP2（`fully_shard`）、DTensor、`torch.distributed.checkpoint`、`torch.distributed.pipelining`、torchrun 与 Flight Recorder；
- Megatron-LM 以 `main` 分支的 Megatron Core 为准，源码路径以 `megatron/core/` 与 `megatron/training/` 为主；Megatron Core 的模块划分近年相对稳定，但参数名和默认值变化较快，正文随文标注；
- DeepSpeed 以 `master` 分支为准，源码路径以 `deepspeed/runtime/` 为主；
- torchtitan 以 `main` 分支为准；它演进很快，目录结构（`torchtitan/distributed/`、`torchtitan/components/`、`torchtitan/experiments/`）以写作时为准；
- torchft 与 NVIDIA Resiliency Extension 处于快速迭代期，正文只讨论它们的设计而不依赖具体接口；
- 硬件以 **H100 SXM（80 GB HBM3，BF16 dense 约 989 TFLOPS）** 为默认分析对象，节点内 8 卡 NVLink、节点间 InfiniBand；给出的公开 MFU 与故障数字均注明来源（Megatron-LM 论文、PaLM 论文、Llama 3 论文），实测会因集群与版本而异。

### 关于"千卡"

正文中的"千卡"指 1024 张 GPU 量级（128 节点 × 8 卡），这是当前 dense 模型预训练的常见规模；万卡级的任务在方法上与之相同，差别在故障率再高一个量级、通信拓扑更复杂，正文会在相关位置标注。


## 章节目录

1. [训练任务的状态解剖：显存账与 MFU](/training-state-anatomy-memory-and-mfu.html)
2. [并行策略全景：每种并行切的是哪种状态](/parallelism-landscape-dp-tp-pp-cp-ep.html)
3. [三个框架：Megatron-LM、DeepSpeed、torchtitan 的架构对比与源码导读](/megatron-deepspeed-torchtitan-architecture.html)
4. [千卡配置实战：并行搭配、micro-batch、激活重计算与 MFU 调优](/thousand-gpu-configuration-and-mfu-tuning.html)
5. [分布式 checkpoint：格式、异步保存与重分片恢复](/distributed-checkpointing-async-save-and-resharding.html)
6. [容错与弹性：故障率数学、straggler、SDC 与弹性训练](/fault-tolerance-elasticity-stragglers-and-sdc.html)
7. [训练稳定性与数据管线：loss spike、梯度范数、数据混合与流式加载](/training-stability-and-data-pipeline.html)
8. [长时训练的可观测与运维：从指标到 hang 排查](/long-running-training-observability-and-operations.html)


## 最终目标

读完这套系列之后，面对任何一个大规模训练任务——无论是自己配的、别人交接的、还是出了问题要接手排查的——读者应该能够回答：

```text
每张卡上放了哪些状态、多少字节？           → 第一篇：显存账
每个 step 走了多少通信、走哪条链路？        → 第二篇：并行策略的通信量
这个框架把参数与优化器状态存在哪里？        → 第三篇：三框架的状态放置
MFU 应该是多少？实际差在哪一项？            → 第四篇：损失拆解
checkpoint 多久存一次？换卡数能不能加载？    → 第五篇：Young 公式与重分片
坏一张卡会损失多少时间？怎么缩短？          → 第六篇：故障率与恢复路径
loss 跳了，回退到哪一步、跳过哪些数据？      → 第七篇：稳定性信号与数据回放
卡住了，哪个 rank 没到哪次通信？            → 第八篇：Flight Recorder 与排查流程
```

最终目标是三种能力：

1. **配置能力**：给定模型与集群规格，算出显存、通信量与 MFU 预期，做出并行与 checkpoint 配置，并解释每个决定的理由；
2. **诊断能力**：面对 MFU 偏低、step 时间抖动、hang、loss spike，用信号与数字而不是猜测定位原因；
3. **运维能力**：为一个持续数周的训练任务设计 checkpoint、容错、监控与告警方案，让它在硬件持续出故障的前提下把有效训练时间维持在 90% 以上。

这是 AI-Infra 引擎层里"训练"这一半的全部：把模型、kernel 与通信组织成一个能跑一个月不倒的系统。
