---
layout: post
title: "PyTorch 深度实践（09）：分布式 PyTorch"
subtitle: "Distributed Training in PyTorch: Collectives, DDP, FSDP, TP, PP, CP and EP"
tags: [PyTorch, AI, AI-Infra]
catalog: true
---

> 本文是[《PyTorch 深度实践：从 Tensor 到深度学习运行时》](/deep-dive-into-pytorch.html)系列的第九篇（共十篇）。上一篇：[性能优化与调试](/pytorch-performance-optimization-and-debugging.html)　下一篇：[PyTorch 的工程体系：一次改动如何安全地到达用户](/pytorch-engineering-system.html)

前八篇都在一张卡上。第八篇末尾算过一笔账：Adam 训练下每个参数的静态显存是 16 字节，7B 参数的模型仅参数、梯度和优化器状态就要 112 GB，激活值还没算。一张 80 GB 的卡放不下。即使放得下，第八篇案例里那个 38M 参数的小模型在单卡上跑到 2207 samples/s 之后，GPU 已经饱和——再要快，只能加卡。

这一篇回答加卡之后的问题：

> **当一张卡放不下模型或跑不完数据时，PyTorch 如何把计算和状态切分到多个设备，并让通信与计算重叠？**

分布式训练的资料通常按 API 组织：DDP 一章、FSDP 一章、张量并行一章、流水线并行一章。这样读完会记住一堆包装类，却答不出"为什么 FSDP 比 DDP 多 50% 通信量"或者"张量并行为什么只能在节点内做"。本文换一条主线：

> **每种并行策略，都是对训练中的五类状态——数据、参数、梯度、优化器状态、激活值——各自做一个决定：复制还是分片。每个决定对应一种集合通信原语和一个通信时机；所有决定加起来，决定了显存占用和通信量。**

DDP、ZeRO 的三个阶段、FSDP、张量并行、流水线并行、上下文并行、专家并行，在这条主线上是同一张表的不同行。掌握了表的结构，新出现的并行策略也能立刻归位。

本篇主要从训练出发，把每种并行的**通用机制**（切什么、通信什么、正向反向各通信几次、通信量怎么算）讲透，再加上训练特有的部分（梯度同步、状态分片、反向重叠、流水线调度）。推理特有的内容本文不做介绍。


## 一、总览：为什么需要多卡

### 1. 两个极限

单卡训练有两个独立的上限：

```text
吞吐极限    GPU 已饱和（第八篇第三章 §2.9），每秒处理的样本数到顶，训练时间只能靠加卡缩短
容量极限    模型状态 + 激活值超过显存，一张卡根本放不下
```

两个极限需要不同的解法。吞吐极限只需要**更多的卡各算一部分数据**，模型本身不用动；容量极限则必须**把模型状态或计算切开放到多张卡上**。前者是数据并行，后者是各种形式的模型并行。真实的大模型训练两个极限同时碰到，所以两类策略要组合使用。

### 2. 五类状态，两种选择

一次训练 step 涉及五类状态。对每一类，多卡训练都要回答同一个问题——每张卡持有完整的一份（**复制**），还是只持有 1/N（**分片**）。这里 **N 表示参与某个并行维度的进程数**（只有一个维度时就是总进程数 world size），**P 表示模型参数量**（个数，乘以每参数字节数才是字节）：

| 状态 | 大小 | 复制的含义 | 分片的含义 |
|---|---|---|---|
| 数据（一个 batch） | ∝ batch | 每卡算同样的数据，无意义 | 每卡算不同的 1/N，这是"并行"的来源 |
| 参数 | P | 每卡一份完整模型 | 每卡持有 1/N，用到时临时聚合 |
| 梯度 | P | 每卡各算一份，必须同步 | 每卡只保留自己负责的 1/N |
| 优化器状态 | 2P（Adam 的 m、v） | 每卡各自更新，结果相同 | 每卡只更新自己负责的 1/N 参数 |
| 激活值 | ∝ batch × 序列长度 × hidden × 层数 | 每卡持有自己那份数据的激活 | 按序列或按 hidden 维切开 |

复制和分片各有代价，而且代价刚好互补：

```text
复制    显存：N 份            通信：需要让 N 份保持一致 → 同步（all-reduce）
分片    显存：1/N             通信：用到时要凑齐完整的一份 → 聚合（all-gather）、用完再分发（reduce-scatter）
```

于是每个"复制还是分片"的决定，都同时决定了三件事：**这类状态占多少显存、需要哪种通信原语、通信发生在 step 的哪个时刻**。这就是本文的主线。

### 3. 符号约定

后文的显存和通信量计算反复用到以下符号，集中声明一次：

```text
P       模型参数量（个）。bf16 参数占 2P 字节，fp32 占 4P 字节；Adam + 混合精度的静态状态共 16P 字节（第八篇第四章 §2）
N       某个并行维度上的进程数（并行度）；只有一个维度时等于 world size
B       每个进程一次处理的序列数（per-rank batch）
S       序列长度（token 数）
H       hidden 维度
L       Transformer 层数
K       流水线并行的 stage 数
M       流水线并行的 micro-batch 数
α, β    一次通信的固定延迟、单位字节的传输时间（β = 1/带宽），第二章 §4
```

一个 Transformer 层的激活值大约是 B × S × H 的若干倍（不做 checkpointing 时约 34 倍，第五章会用到），所以"激活值"在公式里都写成 ∝ B·S·H。

### 4. 三层结构

从底向上，分布式训练由三层组成：

```text
运行时与工程        torchrun 启动 · 数据切分 · Checkpoint · 拓扑 · 性能分析 · 故障排查
        ↑
并行策略            DDP · ZeRO / FSDP · TP · PP · CP · EP · 它们的组合
        ↑
通信底座            进程 / Rank / 进程组 · 集合通信原语 · NCCL · 成本模型
```

并行策略是集合通信原语的组合方式；理解原语的语义和成本之后，每种策略的显存和通信量都能自己推出来。运行时那层是让策略在真实集群上跑起来、跑得快、出了问题能查的工程部分。

### 5. 本文的章节安排

```text
二    通信底座：进程、进程组与集合通信
      1  SPMD 执行模型
      2  进程组与后端
      3  集合通信原语
      4  成本模型：α + β、Ring 与 Tree
      5  通信也是异步的

三    并行策略：复制还是分片
      1  DDP：复制模型、切分数据、同步梯度
      2  ZeRO 与 FSDP：分片状态
      3  TP：切分一层内部（含 Sequence Parallel）
      4  PP：切分层与层之间（含训练调度）
      5  CP：切分序列
      6  EP：切分 Expert
      7  组合与选择：训练 vs 推理
      8  统一表

四    运行时与工程：启动、数据、Checkpoint、拓扑、性能与故障

五    完整案例：把第八篇的 Transformer block 扩到 8 卡、再扩到 4 机
六    Java 对照
七    小结
```


## 二、通信底座：进程、进程组与集合通信

### 1. SPMD 执行模型

PyTorch 分布式训练采用 **SPMD**（Single Program, Multiple Data）模型：**同一份 Python 脚本被启动 N 次，成为 N 个独立进程，每个进程绑定一张 GPU**。没有 master 进程调度 worker，没有中心节点持有全局状态；每个进程执行相同的代码，只是处理的数据不同、持有的状态分片不同。

几个基本概念：

```text
World Size    进程总数
Rank          进程的全局编号 0 ~ World Size-1，唯一标识一个进程
Local Rank    进程在本机内的编号，通常直接用作 GPU 编号：torch.cuda.set_device(local_rank)
Node          一台机器；多机训练时 World Size = 节点数 × 每节点 GPU 数
```

一个最小的分布式脚本：

```python
import os, torch, torch.distributed as dist

dist.init_process_group(backend="nccl")             # 从环境变量读取 RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT
rank, world = dist.get_rank(), dist.get_world_size()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

t = torch.ones(4, device="cuda") * rank             # 每个进程的数据不同
dist.all_reduce(t)                                  # 所有进程调用同一个集合通信
print(rank, t)                                      # 每个进程都得到 0+1+...+(world-1)

dist.destroy_process_group()
```

用 `torchrun --nproc_per_node=8 script.py` 启动，就会有 8 个进程运行这段代码。启动机制第四章 §1 细讲。

SPMD 的关键后果是：**代码中每一处集合通信，都必须被所有参与的进程以相同的顺序调用**。如果 rank 0 走了 `if` 分支多调用了一次 `all_reduce`，而其他 rank 没有，rank 0 会永远等待——这是分布式训练中最常见的 hang 成因，第四章 §6 讨论。

### 2. 进程组与后端

`init_process_group` 创建**默认进程组**（World），包含所有进程。集合通信总是在某个进程组内进行；不指定 `group` 参数时用默认组。

进程组由**通信后端**实现：

| 后端 | 设备 | 用途 |
|---|---|---|
| **NCCL** | GPU | NVIDIA 集合通信库，GPU 训练的唯一实际选择；直接走 NVLink / PCIe / InfiniBand，数据不经过 CPU |
| **Gloo** | CPU（也支持 GPU 但慢） | CPU 训练、调试、以及少数需要在 CPU 上做的控制面通信（如 `monitored_barrier`） |
| **MPI** | 两者 | 需要自行编译，HPC 环境使用 |

后端决定了原语的实现方式和性能，不影响 Python 层的语义。

**子进程组**用于让一部分进程参与通信。多维并行（第三章 §7）大量依赖它：例如 16 张卡做"2 路数据并行 × 8 路张量并行"，需要 2 个各含 8 卡的张量并行组和 8 个各含 2 卡的数据并行组。

```python
tp_group = dist.new_group(ranks=[0, 1, 2, 3, 4, 5, 6, 7])       # 必须所有进程都调用，即使自己不在组里
```

手工管理这些组容易出错，PyTorch 2.x 提供了 **DeviceMesh** 把进程按多维网格组织：

```python
from torch.distributed.device_mesh import init_device_mesh

mesh = init_device_mesh("cuda", mesh_shape=(2, 8), mesh_dim_names=("dp", "tp"))
#   tp →   0  1  2  3  4  5  6  7
#   dp ↓   8  9 10 11 12 13 14 15
mesh["tp"].get_group()     # 本进程所在的 tp 组（同一行）
mesh["dp"].get_group()     # 本进程所在的 dp 组（同一列）
```

DeviceMesh 是 FSDP2、TP、CP API 的共同输入，第三章会反复用到。

### 3. 集合通信原语

集合通信（Collective Communication）是所有并行策略的构件。和点对点通信（一个进程发给另一个）不同，集合通信由组内**所有进程同时参与**，每个进程贡献一部分输入、得到一部分输出。

以 4 个进程为例，每个进程持有一个数据块，方框表示该 rank 上的数据：

```text
broadcast（一份数据发给所有人）
  rank 0  [A]              [A]
  rank 1  [ ]      →       [A]
  rank 2  [ ]              [A]
  rank 3  [ ]              [A]

reduce（所有人的数据归约到一个人）             all_reduce（归约后所有人都拿到结果）
  rank 0  [A₀]         [A₀+A₁+A₂+A₃]              rank 0  [A₀]         [ΣA]
  rank 1  [A₁]    →    [  ]                       rank 1  [A₁]    →    [ΣA]
  rank 2  [A₂]         [  ]                       rank 2  [A₂]         [ΣA]
  rank 3  [A₃]         [  ]                       rank 3  [A₃]         [ΣA]

all_gather（每人一片，拼成完整数据，所有人都拿到）
  rank 0  [A₀]              [A₀ A₁ A₂ A₃]
  rank 1  [A₁]      →       [A₀ A₁ A₂ A₃]
  rank 2  [A₂]              [A₀ A₁ A₂ A₃]
  rank 3  [A₃]              [A₀ A₁ A₂ A₃]

reduce_scatter（归约后每人只拿自己负责的那一段）
  rank 0  [a₀ b₀ c₀ d₀]              [Σa]
  rank 1  [a₁ b₁ c₁ d₁]      →       [Σb]
  rank 2  [a₂ b₂ c₂ d₂]              [Σc]
  rank 3  [a₃ b₃ c₃ d₃]              [Σd]

all_to_all（矩阵转置：第 i 段发给 rank i）
  rank 0  [a₀ b₀ c₀ d₀]              [a₀ a₁ a₂ a₃]
  rank 1  [a₁ b₁ c₁ d₁]      →       [b₀ b₁ b₂ b₃]
  rank 2  [a₂ b₂ c₂ d₂]              [c₀ c₁ c₂ c₃]
  rank 3  [a₃ b₃ c₃ d₃]              [d₀ d₁ d₂ d₃]

barrier    所有人到齐后才继续
send/recv  点对点，一个 rank 发给另一个
```

对应的 API（`torch.distributed` 命名空间）：

```python
dist.broadcast(t, src=0)
dist.all_reduce(t, op=dist.ReduceOp.SUM)                  # 原地，op 可为 SUM / AVG / MAX / MIN / PRODUCT
dist.all_gather_into_tensor(out, t)                       # out.shape[0] == N * t.shape[0]
dist.reduce_scatter_tensor(out, t)                        # t.shape[0] == N * out.shape[0]
dist.all_to_all_single(out, t)
dist.barrier()
dist.send(t, dst=1);  dist.recv(t, src=0)
```

对照上图，每个原语在并行策略中的用途：

| 原语 | 主线中的角色 | 用在哪 |
|---|---|---|
| broadcast | 让复制的状态初始一致 | DDP 构造时分发参数 |
| all_reduce | 让复制的状态保持一致 | DDP 同步梯度；TP 合并部分和 |
| all_gather | 把分片的状态临时凑齐 | FSDP 前向 / 反向前聚合参数；Sequence Parallel |
| reduce_scatter | 把完整结果分回各分片 | FSDP 反向后归约梯度；Sequence Parallel |
| all_to_all | 重新划分维度 | EP 的 token 分发；Ulysses 序列并行 |
| send / recv | 相邻 stage 传递 | PP；CP 的 Ring Attention |

一个恒等式贯穿全文：

> **all_reduce = reduce_scatter + all_gather**

先做 reduce_scatter，每个 rank 得到归约结果的 1/N；再做 all_gather，把这 N 个 1/N 拼起来，每个 rank 都有完整的归约结果。NCCL 的 Ring all-reduce 内部正是这样实现的（下一节）。这个恒等式是理解 ZeRO / FSDP 的钥匙：**DDP 用 all_reduce 同步梯度；FSDP 把它拆成两半，只做 reduce_scatter 那一半，从而每个 rank 只保留 1/N 的梯度**。它也是 Sequence Parallel 的钥匙：TP 边界上的 all_reduce 拆成两半后，中间那段的激活值就是分片的。

### 4. 成本模型：α + β、Ring 与 Tree

#### 4.1 α + β 模型

一次点对点传输的时间用 **α + β 模型**近似：

```text
T(n) = α + β · n        α：延迟（固定开销，微秒级）    β：每字节传输时间 = 1 / 带宽    n：字节数
```

小消息由 α 主导，大消息由 β·n 主导。这和第八篇 Kernel launch 的固定成本是同一个结构：**消息越小越浪费，所以要把小消息合并成大消息**——这就是 DDP 梯度桶（第三章 §1.2）存在的理由，也是 FSDP 分片单元不能太小（第三章 §2.4）的理由。

#### 4.2 Ring all-reduce 分步走一遍

集合通信的成本取决于算法。最常用的 **Ring** 算法把 N 个进程排成环，每个进程只和左右邻居通信。以 N=4、数据切成 4 块为例，all_reduce 的 reduce_scatter 阶段：

```text
初始              第 1 步：每人把一块发给右邻并累加收到的       第 2 步                    第 3 步（结束）
rank 0 [a₀ b₀ c₀ d₀]   发 a₀ → rank 1，收 d₃：[a₀ b₀ c₀ d₀+d₃]      收 c₂+c₃ → [.. c₀+c₂+c₃ ..]   收 b₁+b₂+b₃ → 持有 Σb
rank 1 [a₁ b₁ c₁ d₁]   发 b₁ → rank 2，收 a₀：[a₀+a₁ b₁ c₁ d₁]      收 d₃+d₀ → [.. d₀+d₁+d₃]      收 c₀+c₂+c₃ → 持有 Σc
rank 2 [a₂ b₂ c₂ d₂]   发 c₂ → rank 3，收 b₁：[a₂ b₁+b₂ c₂ d₂]      收 a₀+a₁ → [a₀+a₁+a₂ ..]      收 d₀+d₁+d₃ → 持有 Σd
rank 3 [a₃ b₃ c₃ d₃]   发 d₃ → rank 0，收 c₂：[a₃ b₃ c₂+c₃ d₃]      收 b₁+b₂ → [.. b₁+b₂+b₃ ..]   收 a₀+a₁+a₂ → 持有 Σa
```

N−1 步后，每个 rank 持有**一块**的完整归约结果——这正是 reduce_scatter 的输出。再做 N−1 步 all_gather（同样的环，只是传递而不累加），每个 rank 就有全部四块的归约结果。

对大小为 n 字节的数据，成本：

```text
reduce_scatter 阶段    N-1 步，每步每个 rank 发送并接收 n/N 字节
all_gather 阶段        N-1 步，每步每个 rank 发送并接收 n/N 字节

all_reduce 总时间  ≈ 2(N-1) · α  +  2 · (N-1)/N · β · n
```

带宽项 `2(N-1)/N · β · n` 在 N 很大时趋近 `2βn`——**与进程数无关**。这是 Ring 算法的价值：每个 rank 收发的总字节数约为数据量的 2 倍，不随 N 增长；单独的 all_gather 和 reduce_scatter 各约 1 倍。本文后面所有"通信量"都用这个口径：**每个 rank 收发的字节数**。

#### 4.3 Tree、NVLS 与协议

延迟项 `2(N-1)α` 随 N 线性增长，几百卡时几十微秒的 α 累积成毫秒。NCCL 因此提供其他算法：

```text
Ring      带宽最优，延迟 O(N)；大消息、小规模的默认选择
Tree      双二叉树，延迟 O(log N)，带宽略低；大规模或小消息时 NCCL 自动切换
NVLS      NVLink SHARP：Hopper 起 NVSwitch 可以在交换机内做归约，GPU 只需发一次收一次；节点内 all_reduce 的最快路径
CollNet   把 Tree / NVLS 与 IB 交换机的 SHARP 归约结合，跨节点
```

NCCL 还按消息大小选择**协议**：`LL`（8 字节数据 + 标志位一起发，延迟最低，小消息）、`LL128`（128 字节粒度，NVLink 上兼顾延迟与带宽）、`Simple`（大块传输，带宽最高，大消息）。算法和协议都由 NCCL 根据拓扑和消息大小自动选，`NCCL_ALGO` / `NCCL_PROTO` 可以强制指定，通常只用于基准实验。

#### 4.4 带宽层级

β 由链路决定，层级差异巨大：

```text
NVLink（节点内 GPU 间，H100）       约 900 GB/s 双向，实际 all_reduce 总线带宽 300～450 GB/s
PCIe Gen5 x16                       约 64 GB/s 单向；GPU 与 CPU、或没有 NVLink 的 GPU 之间
InfiniBand NDR（节点间，每网卡）      400 Gb/s ≈ 50 GB/s；一台 8 卡机器通常配 8 张网卡，每 GPU 一张
以太网 100 GbE                       约 12 GB/s
```

**节点内和节点间的带宽差近一个数量级**。这一个事实决定了第三章中"TP 只在节点内做"、"PP 用于跨节点"和 HSDP 的设计。

`nccl-tests` 报告两个带宽：**algbw**（算法带宽）= 数据量 / 时间，是用户视角的速度；**busbw**（总线带宽）= algbw × 2(N−1)/N（all_reduce 时），换算成链路实际承载的流量，用来和硬件规格比较。看 busbw 接近 NVLink 或 IB 的标称值，说明通信库和硬件都正常；看 algbw 才知道自己的 all_reduce 要多久。

### 5. 通信也是异步的

第八篇建立的异步模型在这里延伸：**NCCL 的集合通信也是 GPU 上的 Kernel**，在 Profiler 里以 `ncclDevKernel_AllReduce_...` 之类的名字出现，运行在 PyTorch 为通信专门创建的 CUDA Stream 上，与计算 Stream 并行。

```python
work = dist.all_reduce(t, async_op=True)     # 立即返回 Work 句柄，通信在后台进行
...                                          # 这里可以继续提交计算 Kernel
work.wait()                                  # 让当前 Stream 等待通信完成（是 Stream 间的依赖，不阻塞 CPU）
```

`async_op=False`（默认）等价于调用后立刻 `wait()`——注意即使如此 CPU 也**不**阻塞，只是让计算 Stream 排在通信之后。真正的 CPU 阻塞只发生在 `.item()`、`synchronize()` 这类第八篇讨论过的同步点。

这个机制是"通信与计算重叠"的基础：反向传播还在算后面几层的梯度时，前面几层的梯度已经在通信 Stream 上做 all_reduce。两条 Stream 同时占用 GPU 的不同资源（SM 算力 vs 网络/拷贝引擎），互不阻塞。第三章 §1.2 和 §2.5 分别是 DDP 和 FSDP 对它的运用。

两个限制：

- 通信 Kernel 也占用少量 SM（NCCL 默认每个 channel 用一个 SM 做数据搬运，常见配置下共占用几个到十几个 SM），与计算 Kernel 竞争，重叠期间计算会慢几个百分点；
- "放到另一条 Stream"不等于重叠：如果下一步计算**依赖**通信结果（TP 的 all_reduce 就是这样），再多的 Stream 也只能等。真正的重叠要求重新安排依赖，让通信的输出不是紧接着的那步计算的输入——DDP 的桶、FSDP 的 prefetch、PP 的 micro-batch、异步 TP 的分块（第三章 §3.7）都是这个原则的不同实现。


## 三、并行策略：复制还是分片

### 1. DDP：复制模型、切分数据、同步梯度

#### 1.1 只分片数据

`DistributedDataParallel` 是最简单的并行策略，五类状态的决定是：

```text
数据        分片    每个 rank 处理 batch 的 1/N
参数        复制
梯度        复制    各自算，然后 all_reduce 求平均，使所有 rank 的梯度一致
优化器状态  复制    各自更新，因为梯度一致、初始参数一致，更新后参数仍一致
激活值      各自    随本 rank 的数据
```

数学上，N 个 rank 各算 batch/N 个样本的梯度再取平均，与单卡算整个 batch 的梯度完全等价。所以 DDP 训练在数值上等同于用 N 倍 batch 的单卡训练（浮点归约顺序的差异除外），学习率等超参数应按大 batch 调整。

```python
from torch.nn.parallel import DistributedDataParallel as DDP

model = Block(...).cuda(local_rank)
model = DDP(model, device_ids=[local_rank])           # 构造时从 rank 0 broadcast 参数，保证初始一致
optimizer = torch.optim.AdamW(model.parameters())

for x in loader:                                       # loader 用 DistributedSampler 切分数据（第四章 §2）
    loss = model(x).pow(2).mean()
    loss.backward()                                    # 反向过程中 DDP 自动 all_reduce 梯度
    optimizer.step()                                   # 每个 rank 各自更新，结果一致
    optimizer.zero_grad()
```

训练循环和单卡几乎一样，梯度同步藏在 `backward()` 里。

#### 1.2 Reducer：把 all_reduce 藏进反向

DDP 的核心组件是 C++ 实现的 **Reducer**。它在构造时给每个参数注册一个 autograd hook（`Tensor.register_post_accumulate_grad_hook`，挂在第三篇计算图末端那个把梯度累积进 `.grad` 的节点上），当某个参数的梯度在反向中算完，hook 通知 Reducer。

如果每个参数算完就单独 all_reduce，会有几百到几千次小消息，被第二章 §4.1 的 α 项吃掉。Reducer 把参数分成**桶（Bucket）**，默认每桶 25 MB（`bucket_cap_mb`），一个桶内所有参数的梯度都就位后，对整个桶发起一次异步 all_reduce：

```text
反向传播（计算 Stream）      layer L → layer L-1 → ... → layer 1
                                 │           │
桶就位                        bucket 0 ready   bucket 1 ready   ...
                                 ↓            ↓
通信 Stream                      [all_reduce bucket 0][all_reduce bucket 1]...[all_reduce bucket k]
                                                                                       ↑
backward() 返回前                                                             等待所有桶完成
```

反向是从最后一层往前算的，所以桶按参数注册顺序的**逆序**划分，让最先算完梯度的参数落在第一个桶。第一次迭代 Reducer 会记录实际的梯度就位顺序，据此重建桶的划分。

这就是 DDP 的"通信与计算重叠"：除最后一个桶外，所有通信都隐藏在反向计算之后。反向越长、桶越多，隐藏得越好；模型很小时反向本身很短，通信藏不住，加速比下降。桶大小是权衡：桶越大，α 项越少，但第一个桶就位得越晚、最后一个桶暴露得越多。

`gradient_as_bucket_view=True` 让参数的 `.grad` 直接是桶内存的视图，省一次拷贝和一份梯度显存。`static_graph=True` 告诉 DDP 计算图每次迭代相同，可以跳过未使用参数的检查，并允许一次反向中多次调用同一模块的参数正确归约。

#### 1.3 显存账：DDP 不省显存

DDP 复制了参数、梯度和优化器状态，每个 rank 的静态显存与单卡相同：

```text
每 rank 静态显存 = 16P 字节（Adam + 混合精度，第八篇第四章 §2）
7B 模型 → 每卡 112 GB    → 无论多少张卡，DDP 都放不下
```

DDP 解决的**只是吞吐极限**。它降低的是每卡的 batch，从而降低激活值显存，但静态部分一字节不少。

通信量：每 step 一次梯度 all_reduce，每 rank 收发约 2 倍梯度字节数（fp32 梯度即 2 × 4P = 8P 字节；后文简写为 **2P 量级**，指以状态大小为单位的倍数）。

#### 1.4 通信压缩：`register_comm_hook`

DDP 允许替换桶的通信逻辑：

```python
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks, powerSGD_hook

model.register_comm_hook(state=None, hook=default_hooks.bf16_compress_hook)     # 梯度 cast 成 bf16 再 all_reduce，通信量减半

state = powerSGD_hook.PowerSGDState(process_group=None, matrix_approximation_rank=1, start_powerSGD_iter=1000)
model.register_comm_hook(state, powerSGD_hook.powerSGD_hook)                     # 低秩近似，通信量可降一个数量级，有精度代价
```

前者几乎无损（归约在 bf16 上做，累加误差略大于 fp32）；后者是有损压缩，适合带宽极度受限（跨数据中心、以太网）的场景。hook 拿到的是整个桶的 Tensor，返回一个 Future——自定义的通信策略都从这里进。

#### 1.5 几个必须知道的细节

**梯度累积**：如果每次 `backward()` 都同步，累积 4 次就通信 4 次。`no_sync()` 关闭中间几次的同步：

```python
for i, x in enumerate(loader):
    ctx = model.no_sync() if (i + 1) % 4 != 0 else contextlib.nullcontext()
    with ctx:
        model(x).pow(2).mean().backward()             # 前 3 次只累积，不通信
    if (i + 1) % 4 == 0:
        optimizer.step(); optimizer.zero_grad()      # 第 4 次 backward 同步累积后的梯度
```

**未使用的参数**：Reducer 等待桶内所有参数的梯度就位。如果某次前向有参数没参与计算（条件分支跳过了某个子模块），它的梯度永远不会来，桶永远不就位，`backward()` 挂起。`find_unused_parameters=True` 让 DDP 在每次前向后遍历计算图找出未使用的参数并标记为就位，代价是每步多一次图遍历。更好的做法是让模型结构静态。

**不等长的输入**：各 rank 的数据量不同时，先跑完的 rank 退出循环，其他 rank 的 all_reduce 永远等不到它。`Join` 上下文让先结束的 rank 继续参与"影子"集合通信直到所有人结束：

```python
from torch.distributed.algorithms.join import Join
with Join([model]):
    for x in loader: ...
```

生产中更常见的做法是让 `DistributedSampler` 保证等长（第四章 §2）。

**BatchNorm**：每个 rank 只看到 batch/N 的样本，统计量是局部的。`nn.SyncBatchNorm.convert_sync_batchnorm(model)` 让 BN 在 all_reduce 统计量之后归一化。LayerNorm 按样本归一化，没有这个问题。

**随机性**：Dropout 等操作在各 rank 上应使用**不同**的随机数（否则相当于没有增加样本多样性），而参数初始化应**相同**（DDP 构造时的 broadcast 保证了这一点，即使各 rank 初始化不同）。

**`torch.compile`**：`torch.compile(DDP(model))` 时，Dynamo 的 DDPOptimizer 会在桶边界插入 graph break（第七篇），让编译后的图仍能在反向中触发 Reducer 的 hook。否则整张图的反向作为一个整体，所有梯度同时就位，桶的重叠机制失效。

#### 1.6 何时够用

模型能放进单卡、且激活值留有余量时，DDP 是首选：实现简单，通信量最小（2P 量级），几乎无额外计算开销。第五章案例的第一步就是它。超出单卡容量，进入下一节。

### 2. ZeRO 与 FSDP：分片状态

#### 2.1 ZeRO 的三级分片

DDP 中的冗余显而易见：N 个 rank 持有 N 份完全相同的参数、梯度和优化器状态。**ZeRO**（Zero Redundancy Optimizer，DeepSpeed 提出）的思路是逐级消除这些冗余，分三个阶段，每个阶段多分片一类状态：

| 阶段 | 分片的状态 | 每 rank 静态显存（字节） | N=8 时（以 16P 为基准） | 通信量 |
|---|---|---|---|---|
| DDP | 无 | 16P | 16P | 2P |
| **ZeRO-1** | 优化器状态 | 4P + 12P/N | 5.5P | 2P |
| **ZeRO-2** | + 梯度 | 2P + 14P/N | 3.75P | 2P |
| **ZeRO-3** | + 参数 | 16P/N | 2P | 3P |

（16P 的构成：bf16 参数 2 + bf16 梯度 2 + fp32 主参数 4 + Adam 的 m、v 各 4 = 16 字节/参数。优化器状态这里指 fp32 主参数 + m + v 共 12P。）

每级的逻辑：

**ZeRO-1**：优化器状态是最大的一块（12P）。每个 rank 只负责更新 1/N 的参数，只需持有这 1/N 的优化器状态。但更新需要**完整**的梯度对应段——刚好是 reduce_scatter 的输出。所以把 DDP 的 all_reduce 换成 reduce_scatter（每个 rank 拿到自己负责段的归约梯度），更新自己那段参数，再 all_gather 把更新后的参数发给所有人。通信量 P + P = 2P，与 DDP 相同。

**ZeRO-2**：既然只更新 1/N 的参数，那么其余 (N−1)/N 的梯度在 reduce_scatter 之后就没用了，可以立刻释放。梯度显存从 2P 降到 2P/N。通信量不变。

**ZeRO-3**：参数也分片。每个 rank 只持有 1/N 的参数，前向算到某一层时 all_gather 这层的完整参数，算完释放；反向同样再 all_gather 一次，算完梯度后 reduce_scatter。通信量：前向 all_gather P + 反向 all_gather P + reduce_scatter P = **3P**，比 DDP 多 50%。

结论用主线表达：**ZeRO-1/2 只改变了"归约后的梯度给谁"，是 all_reduce 恒等式的直接应用，不增加通信；ZeRO-3 把参数也分片，多出的 P 是"用到时凑齐"的代价**。

PyTorch 中 ZeRO-1 对应 `ZeroRedundancyOptimizer`（配合 DDP 使用），ZeRO-3 对应 **FSDP**（Fully Sharded Data Parallel）。ZeRO-2 对应 FSDP 的 `reshard_after_forward=False` 模式（§2.4）。生产中 FSDP 是主要选择。

#### 2.2 FSDP 的执行流程

以一个 4 层模型、每层作为一个分片单元为例，FSDP 一个 step 的时间线：

```text
前向
  layer 1:  all_gather(参数₁) → 计算 → 释放完整参数₁（保留分片）
  layer 2:  all_gather(参数₂) → 计算 → 释放
  layer 3:  ...
  layer 4:  ...
反向
  layer 4:  all_gather(参数₄) → 计算梯度 → 释放参数₄ → reduce_scatter(梯度₄) → 只保留自己的 1/N
  layer 3:  ...
  layer 2:  ...
  layer 1:  ...
优化器
  每个 rank 更新自己持有的 1/N 参数（fp32 主参数和 m、v 也只有这 1/N）
```

任何时刻，显存中只有**一层**（或 prefetch 时两层）的完整参数，其余都是分片。峰值静态显存 ≈ 16P/N + 最大一层的完整参数。这就是 FSDP 能训练超过单卡容量的模型的原因。

每个分片单元在 step 中经历一个状态循环：

```text
sharded ──all_gather──► unsharded ──计算──► (前向后 reshard) sharded ──all_gather──► unsharded ──反向计算──► reshard + reduce_scatter ──► sharded
```

FSDP2 用两条专用 Stream 驱动这个循环：一条 all-gather Stream，一条 reduce-scatter Stream；all_gather 的输出先落到一块连续的通信缓冲区，再按参数切成视图交给模块。梯度的 reduce_scatter 完成后，本地分片梯度累加到 DTensor 参数的 `.grad` 上——优化器看到的就是普通的分片参数和分片梯度。

#### 2.3 FSDP2：`fully_shard`

PyTorch 有两代 FSDP 实现。第一代 `FullyShardedDataParallel`（FSDP1）是一个包装类，把被包装模块的所有参数拍平成一个大 `FlatParameter` 再切分；第二代 **FSDP2** 以 `fully_shard` 函数为入口，按参数逐个切分，用 **DTensor** 表示分片后的参数。FSDP2 是 2.4 以后的推荐路径，本文以它为主线。

```python
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh

mesh = init_device_mesh("cuda", (world_size,))
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

with torch.device("meta"):                             # 先在 meta 设备上构造，不分配真实内存
    model = Transformer(...)
for block in model.blocks:
    fully_shard(block, mesh=mesh, mp_policy=mp)        # 每个 block 是一个分片单元
fully_shard(model, mesh=mesh, mp_policy=mp)            # 根模块：处理 embedding、输出层等剩余参数
model.to_empty(device="cuda")                          # 分片后再分配：每卡只分配自己的 1/N
model.init_weights()                                   # 各 rank 初始化自己的分片（需要模型提供确定性初始化）

optimizer = torch.optim.AdamW(model.parameters())      # 参数已是 DTensor，优化器只更新本地分片
```

meta 设备初始化是大模型的必要步骤：7B 模型 fp32 参数 28 GB，如果先在 CPU 上物化再分片，8 个进程会同时占用 224 GB 内存。

调用 `fully_shard(module)` 之后：

- `module` 的类型不变（FSDP1 会包一层 wrapper，改变 `model.xxx` 的访问路径），但被就地混入了前向/反向 hook，并获得 `set_*` 系列控制方法；
- `module` 的每个参数被替换为 `DTensor`，在 mesh 的 dp 维上按 dim 0 分片：`param.to_local()` 拿到本地分片，`param.full_tensor()` 触发 all_gather 得到完整参数；
- 前向 hook 负责 all_gather 和释放，反向 hook 负责 all_gather、释放和 reduce_scatter。

**DTensor** 是 PyTorch 2.x 的分布式 Tensor 抽象：一个逻辑上完整的 Tensor，附带一个 **Placement** 描述它在 DeviceMesh 每一维上是 `Shard(dim)`、`Replicate()` 还是 `Partial()`（各 rank 持有待归约的部分和）。FSDP2 的参数是 `Shard(0)`；TP 的参数按列或按行 `Shard`；两者组合就是 2D 的 Placement。DTensor 上的算子会根据输入的 Placement 自动插入需要的通信（例如两个 `Partial` 相加不需要通信，`Partial` 转 `Replicate` 需要 all_reduce），并推导输出的 Placement。DTensor 让"复制还是分片"从策略的隐含约定变成了 Tensor 元数据的一部分，§3 的 TP 和第四章 §3 的 Checkpoint 都建立在它上面。

FSDP2 相对 FSDP1 的实际差别：

```text
参数表示      逐参数 DTensor                  vs  FlatParameter（把一组参数拍平拼接）
模块类型      不变，无 wrapper                vs  FullyShardedDataParallel 包装类
state_dict    直接是 DTensor，无需特殊上下文   vs  需要 state_dict_type 上下文切换 full / sharded / local
显存          确定性释放，峰值更低            vs  依赖 recordStream，释放时机不确定
灵活性        同一单元内可混合 frozen 参数、不同 dtype 参数    vs  FlatParameter 要求同 dtype、同 requires_grad
```

#### 2.4 分片单元与 wrap 策略

`fully_shard` 施加在哪些模块上，决定了分片单元的粒度，这是 FSDP 最重要的性能决定：

```text
单元太大（整个模型一个单元）    all_gather 一次拿到全部参数 → 峰值显存 = 完整模型，FSDP 失去意义
单元太小（每个 Linear 一个单元） 通信碎成上千次小 all_gather → 被 α 项吃掉，无法重叠
合适的粒度                      Transformer block：参数量足够大（几十 MB 以上），数量适中（几十个）
```

上面代码里对每个 block 调用 `fully_shard`，再对根模块调用一次，是标准做法。根模块那次 `fully_shard` 管理不属于任何 block 的参数（embedding、最终 LayerNorm、输出投影）。

`reshard_after_forward` 参数控制前向后是否释放完整参数：

```text
True（默认）     前向后释放，反向再 all_gather 一次    → ZeRO-3，通信 3P，显存最省
False            前向后保留到反向                      → ZeRO-2，通信 2P，显存多 2P（bf16 完整参数）
整数 k           前向后重新分片到 k 个 rank（而非 N）  → 节点内保留、节点间释放的折中
```

#### 2.5 重叠：prefetch

按 §2.2 的时间线，每层计算前要等 all_gather 完成，通信不重叠。FSDP 用 **prefetch** 解决：在计算第 i 层时，就在 all-gather Stream 上发起第 i+1 层的 all_gather。

```text
计算 Stream      [layer 1 计算    ][layer 2 计算    ][layer 3 计算    ]
all-gather 流  [AG₁][AG₂          ][AG₃            ][AG₄            ]
reduce-scatter 流                                     ...反向时 [RS₄][RS₃]...
```

代价是显存中同时存在两层的完整参数。FSDP2 默认隐式 prefetch 下一层（按上一次迭代记录的执行顺序）；显式控制：

```python
for i, block in enumerate(model.blocks):
    if i + 1 < len(model.blocks):
        block.set_modules_to_forward_prefetch([model.blocks[i + 1]])
    if i > 0:
        block.set_modules_to_backward_prefetch([model.blocks[i - 1]])
```

重叠是否真的发生，要在 Profiler 时间线里看 NCCL Kernel 是否与计算 Kernel 并排（第四章 §5）。

#### 2.6 梯度累积与 `torch.compile`

FSDP2 的梯度累积用 `set_requires_gradient_sync`：

```python
for i, x in enumerate(loader):
    model.set_requires_gradient_sync((i + 1) % 4 == 0)      # False：反向后不 reduce_scatter，梯度以完整形态累积在本地
    model(x).pow(2).mean().backward()
    if (i + 1) % 4 == 0:
        optimizer.step(); optimizer.zero_grad()
```

不同步的那几步，梯度必须以**未分片**的形态保留（因为还没归约），显存多出 2P/N × (N−1) 量级；`set_reshard_after_backward(False)` 可以进一步在累积期间保留完整参数，省掉重复的 all_gather，用显存换通信。

与 `torch.compile` 组合的推荐方式是**先编译每个 block，再 `fully_shard`**：FSDP2 的 hook 在模块边界，天然是 graph break 的位置；编译的图在 block 内部，不跨越通信。

#### 2.7 通信量与 HSDP

FSDP 每 step 通信 3P，其中 2P 是 all_gather 参数（bf16，`param_dtype`），P 是 reduce_scatter 梯度（`reduce_dtype`，fp32 时字节数翻倍）。以 7B 模型、8 卡为例：

```text
all_gather × 2     2 × 7e9 × 2 B = 28 GB
reduce_scatter     7e9 × 4 B     = 28 GB（fp32 归约）
每 rank 每 step    56 GB
NVLink 300 GB/s    ≈ 190 ms
IB 50 GB/s/GPU     ≈ 1.1 s
```

节点内 190 ms 可以藏在几秒的计算里；跨节点 1.1 s 就很难藏。**HSDP**（Hybrid Sharded Data Parallel）用一个 2D mesh 折中：

```python
mesh = init_device_mesh("cuda", (num_nodes, gpus_per_node), mesh_dim_names=("replicate", "shard"))
fully_shard(block, mesh=mesh)    # 2D mesh：在 shard 维分片，在 replicate 维复制
```

参数在节点**内**分片（all_gather 和 reduce_scatter 走 NVLink），在节点**间**复制：每个节点 reduce_scatter 之后，各 rank 只对自己持有的 **1/8 梯度分片**做跨节点 all_reduce。跨 IB 的流量因此从 FSDP 的 3P 降到 2P/8，而且可以按层与反向重叠。用主线的话说：同一类状态在不同的 mesh 维上做不同的决定。代价是每个节点持有完整的一份状态，显存不再随节点数下降。

#### 2.8 CPU offload 与显存的再一次交换

FSDP 允许把分片后的参数、梯度和优化器状态放到 CPU 内存，只在计算时搬到 GPU：

```python
from torch.distributed.fsdp import CPUOffloadPolicy
fully_shard(block, mesh=mesh, offload_policy=CPUOffloadPolicy())
```

这是第八篇第四章 §9 提到的 ZeRO-Offload：GPU 显存降到只剩激活值和当前层参数，代价是每层参数经过 PCIe 往返，PCIe 带宽（64 GB/s）比 NVLink 低一个数量级，通常只在显存实在不够、又不能加卡时使用。

#### 2.9 混合精度策略

`MixedPrecisionPolicy` 与第四篇的 `autocast` 不同，它作用在**参数存储**层面：

```text
param_dtype     all_gather 时把 fp32 分片 cast 成 bf16 再通信 → 通信量减半，计算用 bf16
reduce_dtype    reduce_scatter 用的 dtype；fp32 更稳定，代价是梯度通信量翻倍
本地分片        始终是 fp32 主参数，优化器在 fp32 上更新（第八篇第三章 §2.6 的理由）
```

它比 autocast 更彻底（不需要每个算子判断是否 cast），且与 FSDP 的通信天然结合。两者可以叠加。

### 3. TP：切分一层内部

DDP 和 FSDP 都是**数据并行**：每个 rank 处理不同的数据，对**同一个完整模型**做前向和反向。FSDP 分片的只是状态的存储，计算时仍然要把一层的参数凑齐——所以单层的参数和它的激活值必须放进一张卡。当单层大到放不下（超大 hidden 维），或者 FSDP 的 3P 通信在跨节点时藏不住，就需要切分**计算本身**。

**张量并行**（Tensor Parallel，TP）把一个 Linear 层的权重矩阵切开，TP 组内的每个 rank 算一部分输出、**处理同一份数据**。

#### 3.1 两种切法：列并行与行并行

一个 Linear 层 Y = XW，X 是 [tokens, H_in]，W 是 [H_in, H_out]。切 W 有两种方向：

```text
列并行（Colwise）：按输出维切
   W = [W₀ | W₁ | W₂ | W₃]         每个 rank 持有 W 的 H_out/N 列
   Yᵢ = X Wᵢ                       输入 X 完整（复制），输出 Yᵢ 是 Y 的第 i 段列
   Y = [Y₀ | Y₁ | Y₂ | Y₃]         各 rank 的输出互不重叠，拼起来才是 Y
   通信：无

行并行（Rowwise）：按输入维切
   W = [W₀ ; W₁ ; W₂ ; W₃]         每个 rank 持有 W 的 H_in/N 行
   Zᵢ = Xᵢ Wᵢ                      输入 Xᵢ 是 X 的第 i 段列（分片），输出 Zᵢ 是完整形状的部分和
   Z = Z₀ + Z₁ + Z₂ + Z₃           各 rank 的输出必须相加
   通信：all_reduce
```

列并行的输入是复制的、输出是分片的；行并行的输入是分片的、输出是部分和。**列并行的输出分片形状恰好是行并行需要的输入分片**——所以两者可以直接相连而不需要中间通信：

```text
X（复制） ──列并行 W₁──► Yᵢ（分片） ──逐元素激活──► gelu(Yᵢ)（仍分片） ──行并行 W₂──► Zᵢ（部分和） ──all_reduce──► Z（复制）
```

这正是 Transformer MLP 的结构：fc1（H → 4H）列并行，gelu 逐元素在分片上独立算，fc2（4H → H）行并行，末尾一次 all_reduce。中间 4H 维的激活从头到尾都是分片的，**不需要凑齐**。整个 MLP 前向只通信一次。

#### 3.2 反向也要通信：f 和 g

训练不只有前向。把列并行和行并行的边界看成两个算子 f 和 g（Megatron-LM 的记法）：

```text
          f                                 g
X ──────────► [列并行 → 本地计算 → 行并行] ──────────► Z

f   前向：恒等（X 已复制，直接用）              反向：all_reduce（每个 rank 算出的 ∂L/∂X 只是自己分片贡献的部分，要求和）
g   前向：all_reduce（部分和 → 完整 Z）         反向：恒等（∂L/∂Z 已复制，每个 rank 直接用）
```

f 和 g 互为**共轭**：一个前向通信、反向不通信，另一个反过来。所以一个 MLP 在训练中通信两次：前向 g 处一次，反向 f 处一次。Attention 同理（下一节）。一个 Transformer 层 = Attention + MLP，**训练中每层每 step 共 4 次 all_reduce**（推理只有前向，每层 2 次）。

权重的梯度不需要通信：每个 rank 持有 Wᵢ，∂L/∂Wᵢ 只依赖本地的输入分片和输出梯度，算完就是最终值。用主线的话说，TP 的参数、梯度、优化器状态都是分片的，且分片之间**没有冗余**，所以不需要归约。

#### 3.3 Attention 与 Embedding 的切分

**Attention 按 head 切**。q、k、v 的三个投影是列并行，每个 rank 得到 heads/N 个 head 的 q、k、v（H/N 列刚好是 heads/N 个 head 拼起来）；attention 计算在 head 之间独立，各 rank 本地完成；输出投影 proj 是行并行，末尾 all_reduce。要求 head 数能被 N 整除；GQA 时 kv head 数也要能整除。

这里有一个工程细节：第八篇案例把 q、k、v 合并成一个 `qkv` Linear（3H 列）。对它做列并行，每个 rank 拿到的是 3H/N 列——是 q 的一段、k 的一段、v 的一段**交错**在一起，本地 `split(H, dim=-1)` 会切错。要么把 qkv 拆成三个 Linear，要么在切分时按 [q 段, k 段, v 段] 的顺序重排权重（Megatron 的做法）。第五章案例会先做这个改动。

**Embedding 按 vocab 切**（行并行的变体）：每个 rank 持有词表的 1/N 行，查表时不在自己范围内的 token 输出 0，然后 all_reduce——只有一个 rank 贡献非零值，求和等于查表结果。

**输出层与 loss**：输出投影 H → V 列并行，logits 按 vocab 维分片为 [B, S, V/N]。logits 是训练中最大的单个激活（V 通常 32k～256k），all_gather 它代价很高。`loss_parallel` 直接在分片的 logits 上算 cross-entropy：每个 rank 算本地 vocab 段的 exp 和，all_reduce 一个 [B, S] 的标量场得到 softmax 分母，再各自算自己那段的 loss。通信量从 B·S·V 降到 B·S。

#### 3.4 PyTorch 的 TP API

```python
from torch.distributed.tensor.parallel import (
    parallelize_module, ColwiseParallel, RowwiseParallel, SequenceParallel, PrepareModuleInput, loss_parallel
)
from torch.distributed.tensor import Shard, Replicate

tp_mesh = mesh["tp"]

parallelize_module(model, tp_mesh, {
    "tok_embeddings": RowwiseParallel(input_layouts=Replicate()),         # vocab 切分
    "output":         ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),   # logits 保持 vocab 分片
})
for block in model.blocks:
    parallelize_module(block, tp_mesh, {
        "attn.wq":   ColwiseParallel(),
        "attn.wk":   ColwiseParallel(),
        "attn.wv":   ColwiseParallel(),
        "attn.wo":   RowwiseParallel(),
        "mlp.fc1":   ColwiseParallel(),
        "mlp.fc2":   RowwiseParallel(),
    })
    block.attn.n_heads //= tp_mesh.size()          # 本地只有 heads/N 个 head，view 时用本地数

with loss_parallel():
    loss = F.cross_entropy(logits, targets)        # logits 是 vocab 分片的 DTensor
```

`parallelize_module` 把指定子模块的参数替换成对应 Placement 的 DTensor（`ColwiseParallel` → 权重 `Shard(0)`，`RowwiseParallel` → `Shard(1)`，注意 PyTorch 的 Linear 权重是 [out, in]），并在模块的输入/输出边界按 `input_layouts` / `output_layouts` 插入通信。默认 `use_local_output=True`，模块输出是普通的本地 Tensor，所以 attention 内部的 view / transpose 按本地 shape 写即可——这就是 `n_heads //= N` 那行的原因。

#### 3.5 通信量与适用范围

TP 每次 all_reduce 的数据是一层的输入/输出激活 [B, S, H]：

```text
每次 all_reduce    B × S × H × 2 字节（bf16）
每层每 step        4 次（前向 2、反向 2）→ 8 · B·S·H 字节，ring 下每 rank 实际收发约 2 × 8 · B·S·H · (N-1)/N
每 step 总量       × L 层

例：B=8, S=4096, H=4096, L=32 → 每次 all_reduce 268 MB，每 step 逻辑通信量 32 × 4 × 268 MB ≈ 34 GB
```

与 FSDP 的关键区别：**FSDP 通信参数（∝ P），TP 通信激活（∝ B·S·H·L）**。更要紧的是时机：TP 的 all_reduce 在计算的**关键路径**上——下一个算子的输入依赖它，无法用 Stream 重叠（第二章 §5 的第二个限制）。34 GB 在 NVLink 300 GB/s 下是 110 ms 的纯暴露时间；在 IB 上是 700 ms 且加上 4L 次跨节点延迟。所以：

```text
TP 只在 NVLink 范围内做，TP 度 ≤ 节点内 GPU 数（8）
TP 度越大，每卡的 GEMM 越小（[tokens, H] × [H, 4H/N]），GPU 利用率下降——N=8 通常已是效率下限
```

用主线表达 TP：

```text
参数 / 梯度 / 优化器状态    层内分片，分片间无冗余，不需要归约
激活值                      中间激活分片（4H 维、head 维），层的输入/输出复制（all_reduce 后每个 rank 都有完整值）
数据                        复制，TP 组内所有 rank 处理同一份数据
```

#### 3.6 Sequence Parallel：把复制的激活也切掉

TP 下层的输入/输出激活是复制的。LayerNorm、Dropout、残差相加作用在这些复制的激活上，N 个 rank 算了 N 遍一样的东西，还各存了一份。**Sequence Parallel**（Megatron-LM 的 SP，与 CP 不同）把这些区域的激活按**序列维**切分，每个 rank 只持有 S/N 个 token 的 LayerNorm 输入输出。

进出 TP 区域时的转换，正是那个恒等式：

```text
原来  g：all_reduce（部分和 → 完整）                     f：恒等
现在  g：reduce_scatter（部分和 → 序列分片，每 rank 只拿 S/N 个 token 的完整和）
      f：all_gather（序列分片 → 完整，进入列并行前凑齐所有 token）
反向  g 的反向是 all_gather，f 的反向是 reduce_scatter
```

reduce_scatter + all_gather 的总通信量与一次 all_reduce 相同，所以 SP **不增加通信**，却把 TP 区域外所有激活的显存降到 1/N。它总是与 TP 一起开。

```python
parallelize_module(block, tp_mesh, {
    "attn_norm": SequenceParallel(),                                          # LayerNorm 输入按 Shard(1)（序列维）
    "attn":      PrepareModuleInput(input_layouts=Shard(1), desired_input_layouts=Replicate()),   # f：all_gather
    "attn.wq":   ColwiseParallel(), "attn.wk": ColwiseParallel(), "attn.wv": ColwiseParallel(),
    "attn.wo":   RowwiseParallel(output_layouts=Shard(1)),                    # g：reduce_scatter 而不是 all_reduce
    "mlp_norm":  SequenceParallel(),
    "mlp":       PrepareModuleInput(input_layouts=Shard(1), desired_input_layouts=Replicate()),
    "mlp.fc1":   ColwiseParallel(), "mlp.fc2": RowwiseParallel(output_layouts=Shard(1)),
})
```

#### 3.7 异步 TP：让关键路径上的通信也能重叠

§3.5 说 TP 的 all_reduce 无法重叠，这在"整块通信、整块计算"的粒度上是对的。**异步 TP**（Async TP / 微流水线）把 all_gather + 矩阵乘、矩阵乘 + reduce_scatter 各拆成若干块，块间流水：

```text
不拆     [all_gather 全部        ][matmul 全部          ]
拆 4 块  [AG₀][AG₁][AG₂][AG₃]
              [mm₀][mm₁][mm₂][mm₃]        ← 收到第 0 块就开始算第 0 块，通信只暴露第一块
```

PyTorch 通过 **对称内存**（`torch.distributed._symmetric_memory`，节点内 GPU 直接读写彼此显存）实现块间的细粒度传输，Inductor 在编译时识别 SP 的 all_gather → matmul 和 matmul → reduce_scatter 模式并做替换：

```python
from torch.distributed._symmetric_memory import enable_symm_mem_for_group
enable_symm_mem_for_group(tp_mesh.get_group().group_name)
torch._inductor.config._micro_pipeline_tp = True
model = torch.compile(model)
```

这是第七篇编译器与本篇通信的交汇点：图优化的对象不再只是算子，也包括通信。推理引擎在前向图上做的是同一件事。

### 4. PP：切分层与层之间

**流水线并行**（Pipeline Parallel，PP）把模型按层分成 K 段（stage），每段放在一张卡（或一个 TP 组）上，数据像流水线一样依次经过：

```text
stage 0（卡 0）   embedding + layer 1-8
stage 1（卡 1）   layer 9-16
stage 2（卡 2）   layer 17-24
stage 3（卡 3）   layer 25-32 + 输出层 + loss
```

通信只有相邻 stage 之间的 send/recv：前向传边界激活 [B, S, H]，反向传它的梯度。每个 micro-batch 每个边界 2 × B·S·H × 2 字节，与 TP 的每层 4 次 all_reduce 比是零头，且点对点、不需要所有 rank 同步——**可以跨节点**。这是 PP 相对 TP 的优势。

用主线表达 PP：

```text
参数 / 梯度 / 优化器状态    按层分片，每 stage 只有自己的层，不需要任何归约
激活值                      按层分片，只有 stage 边界的激活需要传输
数据                        切成 micro-batch 依次流过所有 stage
```

#### 4.1 气泡

问题是**气泡（bubble）**：stage 1 必须等 stage 0 算完才能开始，反向同理。如果一个 batch 整体流过，任何时刻只有一个 stage 在工作，利用率 1/K。解法是把 batch 切成 M 个 **micro-batch**，让多个 micro-batch 在不同 stage 上同时流动。

#### 4.2 训练调度的演进

**GPipe**：所有 micro-batch 先做完前向，再做反向：

```text
K=4, M=4，F=前向，B=反向（B 通常约 2 倍 F 的时长，图中按等长画）
stage 0   F₀ F₁ F₂ F₃ .  .  .  .  .  .  B₃ B₂ B₁ B₀
stage 1   .  F₀ F₁ F₂ F₃ .  .  .  .  B₃ B₂ B₁ B₀ .
stage 2   .  .  F₀ F₁ F₂ F₃ .  .  B₃ B₂ B₁ B₀ .  .
stage 3   .  .  .  F₀ F₁ F₂ F₃ B₃ B₂ B₁ B₀ .  .  .

气泡占总时间的比例 ≈ (K-1) / (M+K-1)        M=4,K=4 → 43%     M=32,K=4 → 9%
```

M 越大气泡越小，但 GPipe 要把 M 个 micro-batch 的激活全部保存到反向开始——激活显存 ∝ M。

**1F1B**：进入稳态后，每个 stage 做一次前向就紧接着做一次（更早的 micro-batch 的）反向：

```text
stage 0   F₀ F₁ F₂ F₃ B₀ F₄ B₁ F₅ B₂ F₆ B₃ ...
stage 1   .  F₀ F₁ F₂ B₀ F₃ B₁ F₄ B₂ F₅ B₃ ...
stage 2   .  .  F₀ F₁ B₀ F₂ B₁ F₃ B₂ F₄ B₃ ...
stage 3   .  .  .  F₀ B₀ F₁ B₁ F₂ B₂ F₃ B₃ ...
```

气泡与 GPipe 相同，但任一时刻每个 stage 最多持有 K 个 micro-batch 的激活（而不是 M 个），**显存不随 M 增长**——于是可以放心增大 M 来压气泡。这是训练 PP 的默认调度。

**Interleaved 1F1B**：每个 rank 持有 v 段**不连续**的层（如 rank 0 持有 layer 1-4 和 17-20），相当于虚拟 stage 数变成 vK，气泡缩小到 1/v，代价是 stage 边界数变成 v 倍、P2P 通信量也 v 倍。

**Zero Bubble**：把反向拆成两半——对输入的梯度 B（关键路径，下游 stage 等它）和对权重的梯度 W（不在关键路径上，任何时候算都行）。W 被填进原来的气泡里，理论上可以把气泡压到零，代价是调度复杂、显存更高。

```text
调度              气泡          激活显存（每 stage）     通信
GPipe             (K-1)/(M+K-1) ∝ M                     每边界 2 次/micro-batch
1F1B              同上          ∝ K                     同上
Interleaved 1F1B  上者 / v      ∝ K（略高）             × v
Zero Bubble       → 0           更高                    同 1F1B
```

#### 4.3 PyTorch 的 PP API

`torch.distributed.pipelining`（2.4 起以 prototype 状态进入主库）提供 stage 抽象和上述调度。手工切分是最可控的方式：

```python
from torch.distributed.pipelining import PipelineStage, Schedule1F1B, ScheduleInterleaved1F1B

# 每个 rank 只构造自己那段模型；其他 stage 的层根本不存在于本进程
layers_per_stage = len(model.blocks) // num_stages
stage_mod = StageModule(model, start=rank * layers_per_stage, end=(rank + 1) * layers_per_stage,
                        has_embedding=(rank == 0), has_head=(rank == num_stages - 1))
stage = PipelineStage(stage_mod, stage_index=rank, num_stages=num_stages, device=device)
schedule = Schedule1F1B(stage, n_microbatches=16, loss_fn=loss_fn)

for x, y in loader:
    if rank == 0:                    schedule.step(x)                          # 第一个 stage 喂输入
    elif rank == num_stages - 1:     schedule.step(target=y, losses=losses)    # 最后一个 stage 算 loss
    else:                            schedule.step()                           # 中间 stage 只传递
    optimizer.step(); optimizer.zero_grad()
```

`schedule.step` 内部按调度表执行 micro-batch 的前向、反向和 stage 间的 send/recv；micro-batch 的梯度在本 stage 内累积，最后由普通的优化器更新。

也可以让框架自动切分——`pipeline()` 用 `torch.export` 追踪整个模型（第七篇的导出机制），在指定的模块边界切开：

```python
from torch.distributed.pipelining import pipeline, SplitPoint
pipe = pipeline(model, mb_args=(example_x,), split_spec={"blocks.8": SplitPoint.BEGINNING, "blocks.16": SplitPoint.BEGINNING, "blocks.24": SplitPoint.BEGINNING})
stage = pipe.build_stage(rank, device)
```

代价是模型必须可追踪，且完整模型要先构造出来（可以在 meta 设备上）。

#### 4.4 负载均衡

stage 划分不均匀时，最慢的 stage 决定节奏，其他 stage 等它。第一段有 embedding、最后一段有输出层和 loss（V 维的大矩阵乘），按层数平分往往不均匀；输出层的 logits 也让最后一段显存更高。常见做法是首尾 stage 少放一两层。1F1B 下还要考虑第一段保存激活最多（等最后一段的反向回来）。

PP 的代价不是通信而是气泡和负载不均。推理侧的 PP 面对另一组问题：没有反向所以没有 1F1B 的调度问题，但请求长度动态变化，气泡更难消除；KV Cache 按 stage 分布。

### 5. CP：切分序列

TP 和 PP 切的都是参数。当序列很长（32k、128k 以上）时，瓶颈变成**激活值**：每层激活 ∝ B·S·H，attention 的 score 矩阵 ∝ S²（SDPA 不物化它，但计算量仍 ∝ S²）。即使 B=1，S=128k 时一层的激活也是 GB 级；FSDP 不分片激活，TP 只分到 1/8。

**上下文并行**（Context Parallel，CP）把序列切成 N 段，每个 rank 持有 S/N 个 token 的全部激活。用主线表达：

```text
激活值                      按序列维分片，每 rank 持有 S/N 个 token
参数 / 梯度 / 优化器状态    复制（CP 组内），像 DDP 一样在反向中归约
数据                        同一批序列，每个 rank 处理其中一段
```

Transformer 中除了 attention，所有算子都是逐 token 的（Linear、LayerNorm、gelu 都不跨 token），序列分片后各 rank 独立算，不需要通信。**只有 attention 需要看到全部 token 的 K 和 V**。

#### 5.1 Ring Attention

每个 rank 持有自己那段的 Q、K、V。计算本地 Q 对全部 K、V 的 attention，分 N 步：第 j 步用当前手里的 K、V 块算一块部分 attention，同时把这块 K、V 发给右邻、从左邻收下一块；N 步后每个 rank 的 Q 见过了所有 K、V：

```text
step 0   rank i 用 K_i, V_i        算 Q_i 对块 i 的 attention          同时 K_i,V_i → rank i+1
step 1   rank i 用 K_{i-1}, V_{i-1}  累加 Q_i 对块 i-1 的 attention     同时传下一块
...
step N-1 完成
```

"分块算、在线累加"依赖 **online softmax**：每块得到局部的 max 和 exp 和，合并时按 max 差重新缩放——与 FlashAttention 在 SRAM 分块时的技巧完全相同，只是分块跨越的是 GPU 而不是显存层级。K、V 的传输（send/recv）与当前块的 attention 计算重叠，只要每块的计算时间大于传输时间，通信就被隐藏。

通信量：每层前向每 rank 收发 (N−1)/N × 2 × B·(S/N)·H × 2 字节 × N 块 ≈ 4·B·S·H·(N−1)/N 字节（K 和 V 各一份），反向再传一次 K、V 加上它们的梯度。与 TP 同一量级，但**可以重叠**，且 GQA 下 K、V 的 head 数少，通信量随之减少。

因果掩码带来负载不均：序列后段的 token 要 attend 的 key 更多，按顺序切块时最后一个 rank 的计算量是第一个的近 N 倍。解决办法是**zigzag 切分**：把序列切成 2N 块，rank i 拿第 i 块和第 2N−1−i 块，每个 rank 的因果计算量相等。

另一条路线是 **Ulysses**（DeepSpeed）：在 attention 前用 all_to_all 把"序列分片"转成"head 分片"，attention 按 head 本地算完，再 all_to_all 转回序列分片。通信量更少但 CP 度受 head 数限制。两者可以叠加。

#### 5.2 PyTorch 的 CP API

```python
from torch.distributed.tensor.experimental import context_parallel
from torch.distributed.tensor.experimental._attention import set_rotate_method

cp_mesh = mesh["cp"]
set_rotate_method("alltoall")          # 或 "allgather"：K/V 块在 ring 上的传递方式

with context_parallel(cp_mesh, buffers=[x, position_ids], buffer_seq_dims=[1, 1], no_restore_buffers={x, position_ids}):
    loss = model(x, position_ids).pow(2).mean()      # 上下文内 SDPA 被替换为 ring attention；输入按序列维自动切分
    loss.backward()
```

`context_parallel` 是一个上下文管理器：进入时把 `buffers` 沿指定维切成本地分片，并把 `F.scaled_dot_product_attention` 替换成 ring 版本；模型代码不用改。API 在 `experimental` 命名空间，接口可能变化；生产训练框架（torchtitan）用的就是它。

CP 组内参数是复制的，所以 CP 通常与 FSDP 共用一个 mesh 维：FSDP 在 dp × cp 展平后的维度上分片参数，梯度归约自然覆盖了 CP 组。

推理侧的 CP 有不同的形态：Prefill 阶段切分序列，Decode 阶段切分 KV Cache，两者的通信模式与训练侧的 Ring Attention 不同，本文不展开。

### 6. EP：切分 Expert

MoE（Mixture of Experts）模型的 MLP 由 E 个 expert 组成，每个 token 由 router 选 top-k 个 expert 计算。参数量 ∝ E，但每个 token 的计算量只 ∝ k。E=64、k=2 时参数是稠密模型的 32 倍而计算只有 2 倍——这正是 MoE 的价值，也是它的分布式难点：**参数太多放不下，但每个 expert 的计算又太小不值得 TP**。

**专家并行**（Expert Parallel，EP）把 E 个 expert 分到 N 个 rank，每个 rank 持有 E/N 个 expert 的完整参数：

```text
参数（expert）              按 expert 分片，每 rank E/N 个完整 expert
参数（attention 等稠密部分） 不受 EP 影响，由 DP / TP 决定
激活值                      token 被路由到 expert 所在的 rank 计算，再送回来
```

一个 MoE 层的执行：

```text
router        每个 token 算出 top-k expert 及权重（本地）
dispatch      all_to_all：把 token 的 hidden 发到它选中的 expert 所在 rank
expert 计算   每个 rank 对收到的 token 跑本地 expert（grouped GEMM，E/N 个 expert 各一个小矩阵乘）
combine       all_to_all：结果按原顺序送回 token 所属的 rank，按权重加和
```

这是第二章 §3 中 all_to_all 的主要用途。通信量：每个 token 发 k 份 hidden 出去再收 k 份回来，每层 2 × k × B·S·H × 2 字节，与 TP 同一量级，且与 TP 一样在关键路径上——所以 EP 也偏好 NVLink，跨节点时要靠分块流水（先到的 token 先算）重叠。

训练特有的两个问题：

**负载不均**：router 是学出来的，热门 expert 收到的 token 可能是冷门的几十倍，持有它的 rank 成为 straggler。训练时加**负载均衡辅助 loss**鼓励均匀路由，或设 **capacity factor** 限制每个 expert 每 step 最多接收的 token 数（超出的 token 被丢弃、直接走残差）。

**两套数据并行组**：EP 通常与 DP 组合——例如 64 卡，EP=8、DP=8。稠密参数（attention）在全部 64 卡上复制，梯度在 64 卡的组里归约；expert 参数只在 8 个 DP 副本间复制（每个 expert 存在于 8 张卡上），梯度在**另一个** 8 卡的组里归约。两类参数走不同的进程组，FSDP 也要分别施加在两组参数上。这是 EP 在训练中比推理多出来的复杂度。

router 的具体算法、capacity factor 的取舍、grouped GEMM 与 token 重排的 kernel 实现属于模型与算子层的话题，本文只关注 EP 的通信与状态分布。

### 7. 组合与选择

#### 7.1 训练 vs 推理：同一组策略，不同的重心

同一组并行策略在推理中也全部用得上，但重心不同。本文的主线放在训练上，两者的差别集中在几点：

| | 训练 | 推理 |
|---|---|---|
| 显存构成 | 16P 静态状态 + 激活值 | 2P 参数 + KV Cache |
| 数据并行的成本 | DDP 复制 16P，需梯度同步；FSDP 分片状态，通信 3P | 复制 2P 参数，**零通信**——所以 DP 在推理中永远是最外层的免费倍增器 |
| 通信的对象 | 参数 / 梯度（∝ P，FSDP）和激活（∝ B·S·H，TP） | 只有激活 |
| 消息大小 | 大（Prefill 式，整个 batch 的激活） | Decode 时 S=1，消息 KB 级，α 主导 → `CustomAllreduce` 等低延迟实现 |
| PP 的难点 | 反向调度、激活显存（1F1B） | 请求动态到达、气泡难消除、KV Cache 按 stage 分布 |
| 重叠的余地 | 反向传播是天然的重叠窗口（DDP 桶、FSDP prefetch） | 前向的关键路径短，重叠靠分块流水 |
| 序列切分 | CP 切激活，与 FSDP 共用维度 | Prefill CP、Decode 切 KV Cache |

推理没有优化器状态、没有反向，所以 ZeRO / FSDP 那一整节在推理中没有对应物；反过来，KV Cache 的分布和 decode 的小消息优化在训练中没有对应物。

#### 7.2 决策顺序

训练配置的经验顺序，从内到外：

```text
1. 单层放不下、或激活太大而 FSDP 通信藏不住   → TP（+ SP），节点内，度 ≤ 8
2. 序列太长                                    → CP，与 FSDP 共用维度
3. MoE                                         → EP，通常 EP × TP = 节点内卡数
4. 模型状态放不下                              → FSDP（节点内） / HSDP（跨节点）
5. 跨节点带宽不够、层数多                       → PP 跨节点，micro-batch 数 ≥ 4K
6. 剩下的所有卡                                → 数据并行维度，最外层
```

推理侧的决策顺序与此对照：第 4 步不存在（参数复制是免费的），第 6 步变成"DP 多实例"。

#### 7.3 多维并行的 mesh

各维度组合成 DeviceMesh，从内到外的顺序要与拓扑对齐——**最内层的维度（TP）必须落在同一节点**。以 2 机 16 卡为例：

```python
mesh = init_device_mesh("cuda", (2, 8), mesh_dim_names=("dp", "tp"))
for block in model.blocks:
    parallelize_module(block, mesh["tp"], tp_plan)           # 先 TP：节点内 8 卡切分每一层
    fully_shard(block, mesh=mesh["dp"])                      # 再 FSDP：跨节点 2 路分片 + 数据并行
fully_shard(model, mesh=mesh["dp"])
```

参数变成 2D DTensor：在 tp 维按列/行 `Shard`，在 dp 维按 dim 0 `Shard`。加上 PP（`("pp", "dp", "tp")`）、CP（`("dp", "cp", "tp")`，FSDP 用 `mesh["dp", "cp"]` 展平的维）就是所谓 4D 并行。`torchrun` 的 rank 分配是节点内连续的，`init_device_mesh` 按 rank 顺序填 mesh，所以最后一维自然落在节点内——mesh 维度的顺序写反了，TP 会跨节点，性能差一个数量级。

### 8. 统一表

把本章所有策略放进一张表。P 为参数量，N 为该并行维度的度，通信量以状态大小为单位：

| 策略 | 数据 | 参数 | 梯度 | 优化器状态 | 激活 | 通信原语 | 通信时机 | 每 rank 静态显存 | 通信量 / step |
|---|---|---|---|---|---|---|---|---|---|
| DDP | 分片 | 复制 | 复制→归约 | 复制 | 各自 | all_reduce | 反向中，按桶 | 16P | 2P |
| ZeRO-1 | 分片 | 复制 | 复制→归约 | 分片 | 各自 | reduce_scatter + all_gather | 反向末、更新后 | 4P + 12P/N | 2P |
| ZeRO-2 / FSDP `reshard_after_forward=False` | 分片 | 复制（前向后保留） | 分片 | 分片 | 各自 | all_gather + reduce_scatter | 前向前、反向后 | 2P + 14P/N | 2P |
| ZeRO-3 / FSDP | 分片 | 分片 | 分片 | 分片 | 各自 | all_gather ×2 + reduce_scatter | 每层前向前、反向前后 | 16P/N | 3P |
| HSDP | 分片 | 节点内分片、节点间复制 | 同上 + 节点间归约 | 节点内分片 | 各自 | 节点内 AG/RS，节点间 AR（对分片） | 同上 | 16P/N_shard | 3P 节点内 + 2P/N_shard 节点间 |
| TP (+SP) | 复制 | 层内分片 | 层内分片 | 层内分片 | 中间分片；边界复制（SP：序列分片） | all_reduce（SP：RS + AG） | 每层前向 2 次、反向 2 次，关键路径 | 16P/N | 8·B·S·H·L |
| PP | micro-batch | 按层分片 | 按层分片 | 按层分片 | 按层分片 | send/recv | stage 边界 | 16P/N | 4·B·S·H·(K−1)，最小 |
| CP | 序列分段 | 复制 | 复制→归约 | 复制 | 序列分片 | send/recv（ring）或 all_to_all | 每层 attention 内，可重叠 | 16P（与 FSDP 共用维度时 16P/N） | ≈ 4·B·S·H·L（K、V） |
| EP | 分片 | expert 分片、稠密部分复制 | expert 在 EP 内无冗余 | 同参数 | token 路由到 expert | all_to_all ×2 | 每个 MoE 层，关键路径 | expert 部分 /N | 4·k·B·S·H·L_moe |

读这张表的方式：**先看"参数"列决定了显存能否放下，再看"通信原语"和"通信时机"列决定通信能否被计算隐藏**。任何新策略，只要填出它的行，性能特征就清楚了。


## 四、运行时与工程

### 1. 启动：`torchrun`

SPMD 需要有人把同一脚本启动 N 次、告诉每个进程它的 rank、并让它们找到彼此。这就是 `torchrun`（`torch.distributed.run`）：

```bash
# 单机 8 卡
torchrun --nproc_per_node=8 train.py

# 2 机 16 卡：每台机器各执行一次，node_rank 不同
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
         --rdzv_backend=c10d --rdzv_endpoint=node0:29400 train.py
```

`torchrun` 在每台机器上 fork 出 `nproc_per_node` 个进程，为每个进程设置环境变量：

```text
RANK              全局 rank
LOCAL_RANK        本机内 rank
WORLD_SIZE        总进程数
LOCAL_WORLD_SIZE  本机进程数
MASTER_ADDR / MASTER_PORT    rank 0 所在地址，用于初始化时的 rendezvous
```

`init_process_group()` 不带参数时读这些变量。所有进程通过 `MASTER_ADDR:MASTER_PORT` 上的 TCPStore 交换 NCCL 的通信 ID，之后的通信不再经过它。

`torchrun` 还提供弹性能力（`--max_restarts`）：某个进程失败时杀掉所有进程、从 checkpoint 重启。这反映了 NCCL 训练的故障模型——**任何一个 rank 挂掉，集合通信就无法完成，整个作业必须重启**。没有"部分失败继续运行"的选项（第六章会与微服务对比）。

### 2. 数据切分与随机性

数据并行要求每个 rank 看到不同的数据。`DistributedSampler` 把数据集的索引按 rank 交错切分：

```python
sampler = DistributedSampler(dataset, shuffle=True)               # 自动读取 rank / world_size
loader = DataLoader(dataset, batch_size=per_rank_batch, sampler=sampler, num_workers=4, pin_memory=True)

for epoch in range(epochs):
    sampler.set_epoch(epoch)                                       # 否则每个 epoch 的 shuffle 顺序相同
    for x in loader: ...
```

多维并行时，sampler 的 `num_replicas` 和 `rank` 应该是**数据并行维度**的度和编号（`mesh["dp"].size()`、`mesh["dp"].get_local_rank()`），而不是 world size——同一 TP 组或 PP 组内的 rank 必须拿到相同的数据。

两个陷阱：

- `set_epoch` 忘了调，每个 epoch 数据顺序一样；
- 各 rank 的数据量必须**相等**（`DistributedSampler` 默认 `drop_last=False` 时会填充到相等）。如果 rank 0 多跑一个 batch，它会多调用一次 `backward` 里的 all_reduce，其他 rank 已经进入下一个 epoch 或退出——hang（§6）。

随机性：参数初始化各 rank 应相同（DDP 的 broadcast 或 FSDP 的确定性 `init_weights` 保证），Dropout 等应不同（各 rank 用 `seed + rank`；TP 组内则必须相同，否则复制的激活不一致）。数据加载的随机性由 sampler 的 `seed` 控制，各 rank 必须相同，否则切分不互补。

### 3. Checkpoint

单卡 checkpoint 是 `torch.save(model.state_dict())`。分布式下问题变复杂：FSDP 的参数是分片的，每个 rank 只有 1/N；TP 的参数按列/行切开。三种选择：

```text
Full state_dict      在 rank 0 上 all_gather 出完整参数再保存    → 一份文件；需要 rank 0 有足够 CPU 内存放下整个模型；保存慢
Sharded state_dict   每个 rank 保存自己的分片                    → N 份文件；快；但加载时并行度必须相同
DCP                  torch.distributed.checkpoint：保存 DTensor 及其 Placement 元数据    → 加载时可以 reshard 到不同并行度
```

**DCP**（Distributed Checkpoint）是推荐方案。它保存每个 DTensor 的本地分片和全局 Placement，加载时根据新的 mesh 重新切分——8 卡训练的 checkpoint 可以在 16 卡上恢复，或从 FSDP+TP 的布局加载到纯 FSDP，或在单卡上加载做推理。

```python
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

# 保存：每个 rank 并行写自己的分片
model_sd, optim_sd = get_state_dict(model, optimizer)               # 处理 FSDP / TP 的 DTensor，与并行策略无关
dcp.save({"model": model_sd, "optim": optim_sd}, checkpoint_id="ckpt/step_1000")

# 加载：可以是不同的并行配置
model_sd, optim_sd = get_state_dict(model, optimizer)
dcp.load({"model": model_sd, "optim": optim_sd}, checkpoint_id="ckpt/step_1000")
set_state_dict(model, optimizer, model_state_dict=model_sd, optim_state_dict=optim_sd)
```

`dcp.async_save` 先把分片拷到 CPU 再在后台线程写盘，训练只停顿拷贝的时间。`dcp_to_torch_save` 可以把 DCP 格式转成单文件的 `torch.save` 格式供非分布式加载。

优化器状态是 checkpoint 中最大的部分（12P 中的 8P 是 Adam 的 m、v）。7B 模型完整 checkpoint 约 100 GB，写盘时间和存储带宽是大规模训练中真实的工程约束。

### 4. 多机拓扑

第二章 §4.4 给出了带宽层级。真实机器上的拓扑：

```text
节点内    8 张 GPU 通过 NVSwitch 全互联（DGX/HGX），任意两卡间 NVLink 带宽相同
          没有 NVSwitch 的机器：部分卡对之间 NVLink 直连，其余走 PCIe，带宽不均匀
          nvidia-smi topo -m 查看矩阵：NV# 表示 NVLink 链路数，PIX/PXB/PHB/SYS 表示经过 PCIe 的不同层级
节点间    每 GPU 一张 IB 网卡，GPU 与网卡在同一 PCIe switch 下时可以 GPUDirect RDMA（数据不经 CPU 内存）
```

NCCL 启动时探测拓扑，据此构建 ring 和 tree。`NCCL_DEBUG=INFO` 打印它的决定：

```text
NCCL INFO Channel 00/08 :  0  1  2  3  4  5  6  7          ← 一条 ring 的顺序
NCCL INFO NET/IB : Using [0]mlx5_0:1/IB ...                  ← 用了哪张网卡
NCCL INFO Connected all rings, Connected all trees
```

常用环境变量：

```text
NCCL_SOCKET_IFNAME=eth0      指定 TCP 用哪个网络接口（初始化和 Gloo 用；多网卡机器常见问题）
NCCL_IB_HCA=mlx5             指定 IB 网卡
NCCL_P2P_DISABLE=1           禁用 GPU 直连（排查 NVLink 硬件问题时）
NCCL_ALGO / NCCL_PROTO       强制算法（Ring/Tree/NVLS）和协议（LL/LL128/Simple），见第二章 §4.3
NCCL_NET_GDR_LEVEL           GPUDirect RDMA 的开启条件
```

多维并行的 mesh 布局必须与拓扑对齐（第三章 §7.3）。通信性能不达预期时，排查顺序是：先用 `nvidia-smi topo -m` 确认物理拓扑 → 用 `NCCL_DEBUG=INFO` 确认 NCCL 识别到的拓扑与之一致 → 用 nccl-tests 测裸通信带宽，区分是通信库问题还是应用问题 → 按消息规模判断是 α 主导还是 β 主导，决定优化方向。

### 5. 通信性能分析

第八篇的工具在分布式下继续可用，多了几个要看的东西。

**Profiler 时间线**中 NCCL Kernel 出现在独立的 Stream 泳道上，名字形如 `ncclDevKernel_AllGather_RING_LL` 或 `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL`。判断重叠：

```text
重叠成功      通信泳道上的 NCCL Kernel 与计算泳道上的 Kernel 在时间上并排
重叠失败      NCCL Kernel 期间计算泳道空白 → 计算在等通信，或 NCCL Kernel 与计算 Kernel 交替而不重叠
```

**扩展效率**是评估分布式性能的核心指标：

```text
扩展效率 = 吞吐(N 卡) / (N × 吞吐(1 卡))
```

低于 90% 时要找原因，通常按以下顺序：

```text
1. 通信没有重叠            Profiler 中 NCCL Kernel 期间计算空白 → 检查 prefetch、桶大小、find_unused_parameters
2. 通信量超过带宽能藏的量   通信时间 > 反向计算时间 → 换策略（FSDP→HSDP）、降低 reduce_dtype、增大 per-rank batch
3. Straggler               某个 rank 慢，其他 rank 在 all_reduce 上等它 → Profiler 中大量 NCCL 时间但实际传输很短
                           原因：数据加载不均、某卡降频、CPU 争抢、NUMA 绑定错误、EP 的路由不均
4. per-rank batch 太小      加卡后每卡 batch 变小 → 回到第八篇的 launch-bound
5. 跨节点带宽                NCCL_DEBUG 确认是否走了 IB；nvidia-smi topo 确认 GPU 与网卡的亲和
```

NCCL 自带 `nccl-tests`（`all_reduce_perf` 等）可以在不跑模型的情况下测量集群的原始集合通信带宽（第二章 §4.4 的 busbw），是排除"网络本身有问题"的第一步。

### 6. 故障排查：hang

分布式训练的主要故障形态不是报错，而是**所有进程静止不动**——某些 rank 在等一个永远不会到来的集合通信。三类成因：

```text
集合通信不匹配    某个 rank 多调或少调了一次集合通信；或调用顺序不同；或 Tensor 形状不同
                  常见来源：条件分支、数据量不等（§2）、只在 rank 0 做的 logging 里含集合通信、异常在某个 rank 被吞掉
计算图不一致      DDP 中某个 rank 有参数未使用（第三章 §1.5）
硬件 / 网络       某张卡挂了、IB 链路断了、NCCL 内部错误
```

排查工具：

```python
dist.init_process_group(backend="nccl", timeout=timedelta(minutes=10))   # 默认 10 分钟后抛异常而非永远等待
```

```bash
TORCH_DISTRIBUTED_DEBUG=DETAIL        # 在每次集合通信前校验所有 rank 的调用是否一致（很慢，仅调试用）
NCCL_DEBUG=WARN                       # NCCL 层面的错误
py-spy dump --pid <pid>               # 对每个 rank 打印 Python 栈，比较各 rank 停在哪一行
```

**Flight Recorder** 是 2.x 新增的工具：NCCL 后端记录最近若干次集合通信的元信息（哪个 rank、哪个原语、什么形状、是否完成），超时时 dump 出来：

```bash
TORCH_NCCL_TRACE_BUFFER_SIZE=2000 TORCH_NCCL_DUMP_ON_TIMEOUT=1 TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/tmp/nccl_trace_rank_ torchrun ...
torchfrtrace --prefix /tmp/nccl_trace_rank_          # 分析：哪个 rank 缺了哪次调用，或形状不匹配
```

它直接回答"谁没来"，比对着 N 个 rank 的日志猜要快得多。

### 7. 为什么加卡不线性

大纲里的问题，现在可以系统回答。扩展效率低于 1 的原因，按主线分为四组：

```text
通信时间          带宽项 2βn 不随 N 减少；延迟项 2(N-1)α 随 N 增加；只有重叠部分是免费的
同步等待          BSP 模型下每步所有 rank 必须到齐，最慢的决定速度；rank 越多，出现慢 rank 的概率越大
计算效率          总 batch 不变时 per-rank batch = batch/N，Kernel 变小、GPU 利用率下降（第八篇第三章 §1）；TP 度越大 GEMM 越小
算法效率          总 batch 随 N 增大时，超过临界 batch 后每步的收益递减，需要更多 step 才能收敛——这是优化理论问题，不是系统问题
```

前三组是本文的范围，第四组是为什么"卡多了 loss 反而降得慢"的解释——分布式系统做到了线性吞吐，但每个样本的价值下降了。工程上用学习率缩放、warmup 和更长的训练来补偿。


## 五、完整案例：从 8 卡到 4 机

接着第八篇的案例。终点是：12 层 Transformer block，H=512，约 38M 参数，B=64，bf16 + compile + SDPA，单卡 29 ms/step，2207 samples/s，峰值显存 8.4 GB。GPU 已饱和。数字为示意，比例关系反映真实规律。

### 1. 第一步：DDP，8 卡

模型放得下（8.4 GB），要的是吞吐。DDP：

```python
model = torch.compile(DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True))
```

每卡 B=64，总 batch 512：

```text
8 卡 DDP    step: 30.5 ms    吞吐: 16800 samples/s    扩展效率: 95%    每卡峰值显存: 8.6 GB
```

通信账：fp32 梯度 38M × 4 B = 151 MB，ring all_reduce 每卡收发 2 × 7/8 × 151 ≈ 264 MB，NVLink 下约 0.8 ms，全部藏在约 19 ms 的反向里。多出的 1.5 ms 来自最后一个桶无法重叠、以及 NCCL Kernel 占用的少量 SM。Profiler 确认：NCCL Kernel 与反向 Kernel 并排，只有末尾 0.3 ms 的通信暴露。

显存只多了桶缓冲（`gradient_as_bucket_view=True` 时 `.grad` 就是桶的视图，几乎不额外占用），静态部分不变——DDP 不省显存（第三章 §1.3）。

### 2. 第二步：模型放大，单卡放不下

把模型放大到 H=4096、L=32、S=4096，约 6.4B 参数（加 embedding 约 7B）。静态显存 16P = 112 GB。DDP 在任何卡数下都是每卡 112 GB——不可行。换 FSDP2：

```python
mesh = init_device_mesh("cuda", (8,))
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
with torch.device("meta"):
    model = Transformer(H=4096, L=32)
for block in model.blocks:
    fully_shard(block, mesh=mesh, mp_policy=mp)
fully_shard(model, mesh=mesh, mp_policy=mp)
model.to_empty(device="cuda"); model.init_weights()
```

先算显存。静态 16P/8 = 14 GB，prefetch 时两层完整 bf16 参数 0.8 GB。激活值：不做 checkpointing 时 Transformer 每层每 token 约 34 × H 字节（bf16，SDPA 不物化 score 矩阵），H=4096 时 136 KB；每卡 B=8、S=4096 共 32k token，32 层 → **143 GB**，远超显存。第八篇第四章 §7 说 checkpointing 在那个案例里"不值得"，这里结论反过来：每个 block 做 checkpointing，只保存 block 输入（每层每 token 2H = 8 KB），激活值降到约 8.6 GB，加上重算时一层的完整激活 4.5 GB，共约 13 GB。代价是前向多算一遍，约增加 33% 计算。

```text
8 卡 FSDP   每卡 B=8    总 token 8 × 8 × 4096 = 262k / step
            step: 4.2 s     吞吐: 62k tokens/s     每卡峰值显存: 31 GB
            静态 14 GB + prefetch 0.8 GB + 激活 13 GB + workspace 与碎片约 3 GB
```

通信账：all_gather 2 × 14 GB（bf16 参数）+ reduce_scatter 28 GB（fp32 梯度）= 56 GB / rank / step，NVLink 下约 190 ms，占 4.2 s 的 4.5%。Profiler 时间线中 NCCL Kernel 与计算并排，暴露的只有第一层前向的 all_gather 和最后一层反向的 reduce_scatter，合计不到 40 ms。**通信已隐藏，瓶颈回到计算**：`aten::mm` 占 78%，Tensor Core 利用率接近上限。

对照单卡：单卡放不下，扩展效率无法定义；但 8 卡的 GPU 利用率与第八篇单卡饱和时相当，FSDP 的 3P 通信没有成为代价。

### 3. 第三步：4 机 32 卡

再加 3 台机器。总 batch 保持 64 个序列不变（第四章 §7 的算法约束：总 batch 不能无限加大），每卡 B 从 8 降到 2。直接把 mesh 扩到 32：

```text
32 卡 FSDP   每卡 B=2   step: 1.7 s    吞吐: 154k tokens/s    扩展效率（对 8 卡）: 62%    每卡峰值显存: 10 GB
```

效率掉了近四成。通信账：每卡收发的 56 GB **不随卡数减少**（第二章 §4.2 的带宽项），但每卡的计算随 B 缩到 1/4，只剩约 1.05 s；而 ring 现在跨节点，最慢一段是 IB，每 GPU 50 GB/s，56 GB 要 1.1 s——**通信时间超过了计算时间，无论怎么重叠都藏不住**。Profiler 确认：计算泳道有大段空白在等 all_gather 和 reduce_scatter。

用 HSDP：节点内分片，节点间复制：

```python
mesh = init_device_mesh("cuda", (4, 8), mesh_dim_names=("replicate", "shard"))
fully_shard(block, mesh=mesh, mp_policy=mp)
```

```text
32 卡 HSDP   每卡 B=2   step: 1.12 s    吞吐: 234k tokens/s    扩展效率: 94%    每卡峰值显存: 21 GB
```

通信账重算：节点内 all_gather 2 × 7/8 × 14 GB + reduce_scatter 7/8 × 28 GB ≈ 49 GB，走 NVLink 约 160 ms；跨节点 all_reduce 的是每卡持有的 1/8 梯度分片，3.5 GB × 2 × 3/4 ≈ 5.3 GB，走 IB 约 0.1 s。两者合计 0.26 s，远小于 1.05 s 的计算，可以完全重叠。代价：每个节点持有完整的一份状态，静态显存从 3.5 GB 回到 14 GB——这里 21 GB 有余量，可以接受。

### 4. 再往后：TP 与 CP 何时进场

如果模型再大一倍（14B，静态 224 GB，节点内 8 卡分片后每卡 28 GB，加激活约 45 GB），HSDP 仍可行；到 70B（静态 1.1 TB，节点内分片后每卡 140 GB）就放不下了，这时引入 TP=8 让节点内 8 卡切分每一层：每卡参数 2P/8，FSDP 再在跨节点的 dp 维上分片——mesh 变成 `("dp", "tp")`，参数是 2D DTensor（第三章 §7.3）。TP 的代价是每层 4 次关键路径上的 all_reduce，B=2、S=4096、H=8192 时每次 134 MB，NVLink 下 0.4 ms，80 层共 130 ms，占 step 的几个百分点，可以接受；换成跨节点则不可接受。

如果序列从 4k 拉到 128k，激活值（即使 checkpointing 后）∝ S 增长 32 倍，attention 计算 ∝ S² 增长 1000 倍，单卡放不下一个序列——CP 进场，把序列切到 8 卡，与 FSDP 共用 mesh 维度（第三章 §5.2）。

### 5. 优化报告

| 改动 | 吞吐 | 每卡显存 | 通信/step/rank | 暴露的通信 | 代价 |
|---|---|---|---|---|---|
| 单卡（第八篇终点） | 2207 samples/s | 8.4 GB | 0 | 0 | — |
| 8 卡 DDP | 16800 samples/s（95%） | 8.6 GB | 264 MB | 0.3 ms | 总 batch 变 512，需调学习率 |
| 放大到 7B：8 卡 FSDP + checkpointing | 62k tokens/s | 31 GB | 56 GB | 40 ms | 通信 3P 全部隐藏；重算 +33% 计算 |
| 32 卡 FSDP | 154k tokens/s（62%） | 10 GB | 56 GB（跨 IB） | 0.6 s | 通信不随卡数减少，计算随 batch 减少 |
| 32 卡 HSDP | 234k tokens/s（94%） | 21 GB | 49 GB 内 + 5.3 GB 间 | 0.05 s | 显存不随节点数下降 |

每一步的决策依据都是第三章 §8 那张表的两列：**先看每卡显存放不放得下，再看通信时间能不能被计算隐藏**。第三步还展示了第四章 §7 的第一组和第三组原因同时发生：卡数翻四倍，通信量不变而计算量缩到四分之一，两条曲线交叉，扩展效率断崖式下跌。


## 六、Java 工程师如何理解分布式 PyTorch

Java 工程师做过分布式系统，但 PyTorch 分布式训练和微服务、消息队列所在的那个"分布式"在几个基本假设上相反。先说相反的，再说相通的。

### 1. SPMD vs 微服务

微服务架构中不同进程运行**不同的代码**、承担不同角色，通过 RPC 异步交互，任何一个服务的失败应当被隔离。SPMD 中所有进程运行**同一份代码**，通过集合通信**同步**交互，任何一个进程失败则全体失败。

```text
                微服务                              SPMD 训练
角色            异构：gateway / order / payment       同构：N 个 rank 跑同一脚本
交互            RPC，请求-响应，异步                   集合通信，所有人同时参与，BSP 同步
失败模型        部分失败，熔断、重试、降级              任一失败 → 全体重启（从 checkpoint）
一致性          最终一致，幂等，补偿                   每步强一致：所有 rank 的参数逐位相同
协调者          注册中心、配置中心                     没有；rendezvous 之后各 rank 对等
```

最接近 SPMD 的 Java 世界的东西是 MPI 风格的 HPC 程序，或者 Spark/Flink 的一个 stage 内所有 task 的执行——同一段代码在所有分区上跑，stage 边界做 shuffle 同步。

### 2. 复制 vs 分片：数据库的类比

本文的主线在数据库领域有精确对应：

```text
复制（DDP）           读副本：每个节点一份完整数据，写入时同步（all_reduce ≈ 同步复制）；扩展读吞吐，不扩展容量
分片（FSDP / TP）     分库分表：每个节点 1/N 数据，跨分片查询要聚合（all_gather ≈ scatter-gather 查询）；扩展容量，代价是跨分片通信
HSDP                  分片 + 每个分片多副本：Kafka 的 partition × replica、Elasticsearch 的 shard × replica
TP 的列/行并行        按列分区的表做 join：分区键对齐时 join 不需要 shuffle（列并行 → 行并行无通信），否则要 shuffle（all_reduce）
EP                    按 key 路由到不同分片处理（all_to_all ≈ shuffle by key），热 key 导致的分片倾斜 ≈ expert 负载不均
```

Kafka consumer group 是 DistributedSampler 的对应物：一个 topic 的 partition 被 group 内的 consumer 互不重叠地分走。`set_epoch` 忘调，就像每次 rebalance 都用同一种分配。

### 3. 集合通信 vs MapReduce / 并发原语

```text
all_reduce         MapReduce 的 combine + reduce 后再广播；Spark 的 treeReduce + broadcast
reduce_scatter     shuffle 到 reducer：每个 key 的所有值归到一个 reducer
all_gather         collect() 后广播给所有 executor
all_to_all         shuffle 本身：每个 mapper 的第 i 段发给 reducer i
barrier            CyclicBarrier / Phaser
Work.wait()        CompletableFuture.join()，但发生在 GPU Stream 上而非线程上
流水线并行         SEDA / Disruptor 的多阶段流水线：每个阶段一个处理器，micro-batch 是流过的事件，气泡是阶段间的空转
```

Ring all_reduce 的"每个节点只和邻居通信、带宽项与 N 无关"，与 Chord/Cassandra 一致性哈希环的"每个节点只需知道少数邻居"是同一类设计：用 O(N) 步的顺序通信换掉 O(N²) 的全连接。

### 4. hang vs 分布式死锁

Java 中的死锁是两个线程互相等对方持有的锁，`jstack` 看到 `BLOCKED`。集合通信 hang 是 N 个进程中有人没来开会，其他人无限等待——不是循环等待，是**缺席**。诊断思路相同：拿到所有参与者的栈（`py-spy dump` ≈ `jstack`），比较谁停在哪里。Flight Recorder 相当于一个记录了每次"开会"的参与者名单和议题的日志——比 `jstack` 更进一步，直接指出谁缺席了哪一次。

`TORCH_DISTRIBUTED_DEBUG=DETAIL` 对应 `-Xcheck:jni` 或 `-ea` 这类"开了很慢但能抓到错误"的模式。

### 5. Checkpoint vs 快照

DCP 的 sharded checkpoint + reshard 能力，对应分布式存储的快照与 rebalance：Elasticsearch 的 snapshot 可以 restore 到不同 shard 数的集群；Kafka 的 partition 可以 reassign。区别是训练 checkpoint 是**唯一的容错手段**——没有 WAL、没有副本自动切换，失败就回滚到上一个 checkpoint 重来。checkpoint 间隔是"重算多少"与"写盘多少"的权衡，与数据库 checkpoint 间隔的权衡结构相同。

### 6. 扩展效率 vs Amdahl / USL

第四章 §7 的四组原因，Java 工程师在 Universal Scalability Law 里见过：线性项是并行部分，α 是串行部分（不能重叠的通信、同步等待），β 是一致性代价（rank 越多、越可能等最慢的）。分布式训练的 USL 曲线和数据库连接池、线程池的曲线形状相同，只是横轴是 GPU 数。


## 七、本文小结

### 1. 一条主线

```text
五类状态          数据 · 参数 · 梯度 · 优化器状态 · 激活值
两种选择          复制（显存 N 份，需同步）  vs  分片（显存 1/N，需聚合与分发）
三个后果          显存占用 · 通信原语 · 通信时机
一个恒等式        all_reduce = reduce_scatter + all_gather
```

DDP 全复制只分数据；ZeRO 三级逐个把优化器状态、梯度、参数分片；FSDP 是 ZeRO-3 的 PyTorch 实现，用 3P 通信换 16P/N 显存；TP 在层内分片、通信激活、每层训练 4 次 all_reduce 在关键路径上、必须 NVLink，SP 把它边界上的复制激活也切掉；PP 按层分片、通信最小、代价是气泡，1F1B 让显存不随 micro-batch 数增长；CP 切序列、只有 attention 通信、可重叠；EP 切 expert、all_to_all 路由 token。HSDP 和多维并行是在 DeviceMesh 的不同维上对同一状态做不同决定。

### 2. 三层结构

```text
通信底座    SPMD · 进程组 · 集合通信原语 · α+β 与 Ring / Tree 成本模型 · 通信是异步 Kernel、重叠要求重排依赖
并行策略    第三章 §8 那张表
运行时      torchrun · DistributedSampler · DCP · 拓扑对齐 · 扩展效率 · Flight Recorder
```

### 3. 两个判断

面对任何分布式配置，先问两个问题：

```text
每卡显存放不放得下？      看表的"参数"列：16P（DDP）→ 16P/N（FSDP / TP / PP）；激活看 CP
通信能不能被计算隐藏？    通信量 / 带宽  vs  计算时间；藏不住就换策略（FSDP → HSDP → TP 进节点内 → PP 跨节点）
```

### 4. 与前几篇的连接

```text
第三篇 autograd hook              → DDP Reducer 和 FSDP 的反向 hook 都挂在它上面
第四篇 autocast / 主参数           → MixedPrecisionPolicy：fp32 分片、bf16 通信与计算
第七篇 graph break / 导出          → DDP + torch.compile 在桶边界切图；pipeline() 用 export 切 stage；异步 TP 由 Inductor 改写通信
第八篇 异步 Stream / 同步点        → 通信 Stream 与计算 Stream 的重叠；Work.wait() 是 Stream 依赖而非 CPU 阻塞
第八篇 launch-bound                → 加卡后 per-rank batch 变小，瓶颈回到 CPU 侧
第八篇 16 B/参数                   → 本文所有显存账的基准
第八篇 Activation Checkpointing    → 单卡案例里不值得，7B 案例里必需
第八篇 CPU 内存 / PCIe             → CPU offload 的约束
```

### 5. 本篇涉及的源码位置

本篇讨论的机制在源码中的位置（对应第一篇第四章 §3 的代码地图）：

| 路径 | 内容 |
|---|---|
| `torch/distributed/distributed_c10d.py` | `init_process_group`、集合通信的 Python API、进程组管理 |
| `torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp`、`ProcessGroupGloo.cpp`、`Work.hpp` | NCCL / Gloo 后端；异步通信的 `Work` 句柄 |
| `torch/csrc/distributed/c10d/reducer.cpp`、`torch/nn/parallel/distributed.py` | DDP：Reducer 的梯度桶与 all-reduce 触发；Python 包装 |
| `torch/distributed/fsdp/_fully_shard/` | FSDP2：`fully_shard`、分片单元、预取、all-gather / reduce-scatter 的调度 |
| `torch/distributed/tensor/`、`torch/distributed/device_mesh.py` | DTensor 与 `DeviceMesh` |
| `torch/distributed/tensor/parallel/` | TP：`ColwiseParallel`、`RowwiseParallel`、`SequenceParallel`、`loss_parallel` |
| `torch/distributed/pipelining/`、`torch/distributed/tensor/experimental/_attention.py` | PP 的 stage 与调度；CP 的 `context_parallel` |
| `torch/distributed/checkpoint/`、`torch/distributed/run.py`、`torch/distributed/elastic/` | 分布式 Checkpoint；`torchrun` 与弹性启动 |

到这里，PyTorch 的执行系统从单卡讲到了多机。剩下最后一个问题：这样一个横跨 Python、C++、CUDA、编译器和分布式运行时的框架，如何保证每次改动不破坏正确性和性能？

> **一个复杂深度学习框架如何测试、构建和演进？**


## 下一篇

[PyTorch 的工程体系：一次改动如何安全地到达用户](/pytorch-engineering-system.html)
