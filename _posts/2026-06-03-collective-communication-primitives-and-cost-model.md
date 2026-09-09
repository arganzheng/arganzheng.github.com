---
layout: post
series: communication-and-interconnect
title: "通信与互联（01）：集合通信原语与代价模型——α-β 模型与 ring all-reduce"
subtitle: "Collective Communication Primitives and the Alpha-Beta Cost Model: Deriving Ring All-Reduce"
tags: [NCCL, RDMA, GPU, AI, AI-Infra]
catalog: true
---

总纲把这个系列要回答的问题定为一句话：一次 all_reduce 从调用到完成，数据在 PCIe、NVLink、InfiniBand 上是怎么流动的，为什么有时候是带宽的问题、有时候是延迟的问题。要回答它，先得有一把尺子。没有尺子，nccl-tests 打出来的 `busbw 23.1 GB/s` 只是一个数字，profiler 里 `ncclDevKernel_AllReduce` 的 145 µs 也只是一个数字，你不知道它们是好是坏、离上限多远、差的那部分该去哪一层找。

这一篇造这把尺子。它不碰任何硬件、不读 NCCL 的源码，只做三件事：把集合通信的**原语**定义清楚，把描述一条链路的 **α-β 模型**建立起来，再把 **ring** 和 **tree** 两个最基本的 all_reduce 算法在这个模型上推导出理论时间。推导的产物是几个公式和一个几十行的 Python 脚本，后面六篇的每一个实测数字都要拿来和它比。

尺子的刻度是两本账。**带宽的账**问：这次通信一共要在链路上搬多少字节，链路每秒能搬多少；**延迟的账**问：这次通信要分几步走，每一步有多少与字节数无关的固定开销。同一个 all_reduce，1 GB 的梯度几乎只有第一本账，64 KB 的 decode 张量几乎只有第二本账。分不清这两本账，就会在延迟主导的场景里升级网卡，在带宽主导的场景里合并小消息——两者都不会有任何效果。

本篇的核心问题就是把这两本账各算一遍：

> **8 张卡做一次 1 GB 的 all_reduce，链路单向 25 GB/s，ring 算法理论上要多久？改成 64 KB 呢？这两个数字为什么分别对带宽和延迟敏感？**

回答它需要的全部材料是：ring all_reduce 走 $$2(n-1)$$ 步、每个 rank 收发 $$\frac{2(n-1)}{n}S$$ 字节，以及一个 α 的量级。有了这三样，答案是两行算术；没有它们，答案只能靠测。本篇的目标是让读者在没有机器的情况下也能写出这两行算术，并且知道实测值应该落在哪个范围、超出范围时该怀疑哪一层。

依照系列惯例，本篇的性能数字全部是**理论下界或典型量级**，不是实测。链路带宽用公开标称值（InfiniBand HDR 单向 25 GB/s、NDR 单向 50 GB/s；NVLink 与 PCIe 的数字下一篇展开），每步延迟 α 用数量级估计（InfiniBand 上一步 5–20 µs、NVLink 上几 µs），文中会反复标注"典型量级、非实测"。第六篇会用 nccl-tests 把这些数字换成你手上机器的真实值。


## 一、总览：三个变量、两本账、一条曲线

### 1. 问题的形状

任何一次集合通信都可以用四个量描述：参与者数 $$n$$、数据量 $$S$$（字节）、每步固定开销 $$\alpha$$（秒）、链路带宽 $$\beta$$（字节/秒）。前两个由上层决定——多少张卡、多大的张量；后两个由硬件和软件栈决定——什么链路、什么协议、多少层软件参与一次握手。这一篇的任务是给出从这四个量到时间 $$T$$ 的函数：

```text
上层决定                            底层决定
─────────────────────────           ────────────────────────────────
n   参与者数（DP/TP/EP 的度）        α   每一步的固定开销：握手、kernel 启动、proxy 响应
S   字节数（张量大小 × dtype）        β   链路每秒能搬多少字节：NVLink / PCIe / IB 的单向速率
        │                                       │
        └──────────────┬────────────────────────┘
                       ▼
            算法：把 (n, S) 变成"走几步、每步搬多少字节"
                       │
                       ▼
            T = 步数 × α  +  每 rank 搬运字节数 / β
                 ─────────     ───────────────────
                 延迟的账          带宽的账
```

算法是中间那一层：ring 把 all_reduce 变成 $$2(n-1)$$ 步、每步 $$S/n$$ 字节；tree 把它变成 $$2\log_2 n$$ 步、每步更大的块。同一个 $$(n, S, \alpha, \beta)$$，不同算法给出不同的 $$T$$，而且两本账上的优劣往往相反——这是后面第五、六章的主题。

### 2. 一条曲线

把 $$T$$ 换算成带宽（$$S/T$$ 的某个归一化，第七章讲 algbw 与 busbw 的区别）、以 $$S$$ 为横轴画出来，就是 nccl-tests 输出的那条曲线，它的形状对整个系列都重要：

```text
busbw
  ▲
  │                                ┌─────────────────── 平台：由 β 与算法带宽效率决定
  │                          ╭─────╯                    （带宽的账）
  │                     ╭────╯
  │                ╭────╯
  │           ╭────╯       ← 拐点附近：S* ≈ 步数 × α × β / 效率
  │      ╭────╯              （两本账相当）
  │ ╭────╯
  │─╯   ← 小消息端：T ≈ 步数 × α，与 S 无关；busbw 随 S 线性上升
  └──────────────────────────────────────────────────────▶ S（对数轴）
   8 KB     128 KB      2 MB       32 MB      512 MB
```

左端的斜坡是延迟主导：时间几乎不随 $$S$$ 变，所以带宽随 $$S$$ 线性增长；右端的平台是带宽主导：时间随 $$S$$ 线性增长，带宽饱和在链路速率乘以算法效率。拐点的位置由 $$\alpha$$、$$\beta$$、步数共同决定，对 8 卡 IB 上的 ring all_reduce 大约在 MB 量级（第五章算）。第六章会讲这条曲线的每种异常形状对应哪一层的问题，本篇先把它"应该长什么样"算出来。

### 3. 本文的章节安排

```text
第二章  集合通信原语        八个原语的语义、组合与对偶关系、每个原语每 rank 至少搬多少字节
第三章  训练与推理的通信模式   DP / FSDP / TP / PP / MoE / 推理 TP / PD 分离各用哪个原语、多大、几个参与者
第四章  α-β 模型            一条消息的 T = α + S/β；拐点 S* = αβ；模型忽略了什么
第五章  ring all_reduce      reduce_scatter + all_gather 的完整推导、4 rank 逐步图、核心问题的两个数字
第六章  tree all_reduce      log n 的延迟、朴素二叉树为什么只有一半带宽、double binary tree、Ring vs Tree 对照表
第七章  algbw 与 busbw        nccl-tests 的两个带宽怎么算、各原语的系数、为什么只有 busbw 能和链路比
第八章  分层与多级算法        节点内快、节点间慢：两级 all_reduce 的代价、为什么平坦 ring 跨节点吃亏
第九章  消息大小的谱          几十 KB / 几十 MB / GB 三个量级各在曲线哪一段、各自的对策与检查项
第十章  小结                 要点、公式速查、源码位置、comm-probe 的 cost_model.py
```


## 二、集合通信原语：语义、组合与下界

### 1. 八个原语

集合通信（collective）是一组进程（rank）**同时参与**、语义上一次完成的数据交换。与点对点 send/recv 的区别在于：每个 rank 调用同一个函数，参数里没有"发给谁"，数据的流向由原语的定义决定。下面用 $$n = 4$$、每个 rank 持有一段数据的方式给出八个原语的语义。记 rank $$i$$ 的输入为 $$x_i$$，$$\oplus$$ 为归约算子（sum、max、min、prod、avg 之一）。

```text
broadcast（root = 0）                 reduce（root = 0）
  r0: x0     →  r0: x0                 r0: x0     →  r0: x0⊕x1⊕x2⊕x3
  r1: -      →  r1: x0                 r1: x1     →  r1: -
  r2: -      →  r2: x0                 r2: x2     →  r2: -
  r3: -      →  r3: x0                 r3: x3     →  r3: -

all_reduce                            all_gather
  r0: x0     →  r0: x0⊕x1⊕x2⊕x3        r0: x0     →  r0: [x0 x1 x2 x3]
  r1: x1     →  r1: x0⊕x1⊕x2⊕x3        r1: x1     →  r1: [x0 x1 x2 x3]
  r2: x2     →  r2: x0⊕x1⊕x2⊕x3        r2: x2     →  r2: [x0 x1 x2 x3]
  r3: x3     →  r3: x0⊕x1⊕x2⊕x3        r3: x3     →  r3: [x0 x1 x2 x3]

reduce_scatter（每个 x_i 切成 4 块 x_i[0..3]）
  r0: x0[0] x0[1] x0[2] x0[3]   →  r0: x0[0]⊕x1[0]⊕x2[0]⊕x3[0]
  r1: x1[0] x1[1] x1[2] x1[3]   →  r1: x0[1]⊕x1[1]⊕x2[1]⊕x3[1]
  r2: x2[0] x2[1] x2[2] x2[3]   →  r2: x0[2]⊕x1[2]⊕x2[2]⊕x3[2]
  r3: x3[0] x3[1] x3[2] x3[3]   →  r3: x0[3]⊕x1[3]⊕x2[3]⊕x3[3]

all_to_all（第 i 个 rank 的第 j 块发给第 j 个 rank，等于矩阵转置）
  r0: x0[0] x0[1] x0[2] x0[3]   →  r0: x0[0] x1[0] x2[0] x3[0]
  r1: x1[0] x1[1] x1[2] x1[3]   →  r1: x0[1] x1[1] x2[1] x3[1]
  r2: x2[0] x2[1] x2[2] x2[3]   →  r2: x0[2] x1[2] x2[2] x3[2]
  r3: x3[0] x3[1] x3[2] x3[3]   →  r3: x0[3] x1[3] x2[3] x3[3]

scatter（root = 0）                   gather（root = 0）
  r0: x[0] x[1] x[2] x[3] → r0: x[0]    r0: x0  →  r0: [x0 x1 x2 x3]
                            r1: x[1]    r1: x1  →  r1: -
                            r2: x[2]    r2: x2  →  r2: -
                            r3: x[3]    r3: x3  →  r3: -

send / recv                            点对点：一个 rank 发、一个 rank 收，必须配对
```

几点约定后面会反复用到。**S 指什么**：对 broadcast、reduce、all_reduce，$$S$$ 是每个 rank 的 buffer 大小，也是最终结果的大小；对 all_gather，$$S$$ 指拼接后的总大小（每 rank 输入 $$S/n$$）；对 reduce_scatter，$$S$$ 指输入总大小（每 rank 输出 $$S/n$$）；对 all_to_all，$$S$$ 是每 rank 的输入总量（也等于输出总量）。这与 nccl-tests 打印的 `size` 列一致，第七章核对源码时会看到它是怎么算的。**归约的顺序**：$$\oplus$$ 在浮点上不满足结合律，不同算法、不同 rank 数会给出比特级不同的结果，这是第六篇"结果不稳定"一类问题的来源，此处只需记住 all_reduce 不保证各 rank 结果的比特一致性——NCCL 实际上保证了（所有 rank 收到同一份归约结果），但不同次运行、不同算法之间不保证。

### 2. 组合与对偶

八个原语之间有两组关系。第一组是**组合**：

```text
all_reduce      = reduce_scatter + all_gather        ← ring all_reduce 的基础，第五章
all_reduce      = reduce + broadcast                 ← tree all_reduce 的基础，第六章
all_gather      = n 次 broadcast（每次换一个 root）
reduce_scatter  = reduce + scatter
all_to_all      = n 次 scatter（每次换一个 root）
```

第一条是本篇最重要的等式。reduce_scatter 之后每个 rank 持有结果的 $$1/n$$（且已经归约完），再 all_gather 把这 $$n$$ 段拼起来，每个 rank 就有了完整结果。它的重要性在于两个子操作都是**带宽最优**的：每个 rank 只需收发 $$\frac{n-1}{n}S$$ 字节，而不需要任何 rank 接收全部 $$n$$ 份数据。

第二条给出另一种分解：先把所有数据归约到一个 root，再从 root 广播出去。它的步数可以做到 $$O(\log n)$$，但 root 要接收 $$n$$ 份数据，除非用树形流水化，否则带宽差。

第二组关系是**对偶**：把一个原语的数据流反过来、把"复制"换成"归约"，就得到另一个：

```text
broadcast   ⟷  reduce            一对多复制  ⟷  多对一归约
scatter     ⟷  gather            一对多分发  ⟷  多对一收集
all_gather  ⟷  reduce_scatter    多对多复制  ⟷  多对多归约
```

对偶的意义是实现上的：一个 all_gather 算法把箭头反过来、在接收端加一个 $$\oplus$$ 就是 reduce_scatter 算法，两者的步数和字节数完全一样。所以后面分析 ring 的 reduce_scatter 阶段时，all_gather 阶段的代价可以直接复用。

### 3. 每个原语至少要搬多少字节

在任何算法之前，先问一个与算法无关的问题：完成这个原语，**每个 rank 至少要接收多少字节**？这给出带宽账的下界，任何算法都不可能低于它。

- broadcast：每个非 root 的 rank 必须收到全部 $$S$$，下界 $$S$$；
- reduce：root 必须收到其他 $$n-1$$ 个 rank 的贡献。朴素做法 root 接收 $$(n-1)S$$；用树或链把归约分摊出去，每个 rank 接收 $$S$$ 即可（每个中间节点接收、归约、上传），下界 $$S$$；
- all_gather：每个 rank 已有 $$S/n$$，还缺 $$\frac{n-1}{n}S$$，下界 $$\frac{n-1}{n}S$$；
- reduce_scatter：每个 rank 最终持有 $$S/n$$ 的归约结果，这一段需要其他 $$n-1$$ 个 rank 各贡献 $$S/n$$，共 $$\frac{n-1}{n}S$$，且归约可以边收边做，下界 $$\frac{n-1}{n}S$$；
- all_reduce：可以证明（Patarasuk 与 Yuan 2009 年的结果）任何 all_reduce 算法每个 rank 至少收发 $$\frac{2(n-1)}{n}S$$——直觉上就是 reduce_scatter 与 all_gather 的下界相加，两者缺一不可；
- all_to_all：每个 rank 要发出 $$\frac{n-1}{n}S$$（留一块给自己）、接收同样多，下界 $$\frac{n-1}{n}S$$。

```text
原语              每 rank 接收字节数下界      备注
──────────────    ───────────────────────    ───────────────────────────────
broadcast         S                          非 root 都要拿到全部
reduce            S                          流水化后中间节点各收 S
all_reduce        2(n-1)/n · S               → 2S（n 大时），reduce_scatter + all_gather
all_gather        (n-1)/n · S                → S
reduce_scatter    (n-1)/n · S                → S
all_to_all        (n-1)/n · S                → S；但是 n-1 个不同的目标
```

这张表就是第七章 busbw 系数的来源：nccl-tests 把测得的 $$S/T$$ 乘上这一列的系数，得到"链路实际承载的流量速率"，它才能和链路带宽比。

注意 all_to_all 的一行：字节数下界与 all_gather 相同，但流量模式完全不同——all_gather 的每一块要发给所有人，可以走环流水化；all_to_all 的每一块只发给一个特定的人，$$n$$ 个 rank 之间是 $$n(n-1)$$ 条不同的流。这意味着它无法从"绕环一圈"里得到好处，而在跨节点时会同时压满所有链路，是 MoE 训练最难对付的通信模式。


## 三、训练与推理需要哪些原语

### 1. 训练：五种并行的通信模式

并行策略本身不在本系列范围内，这里只把每一种策略**产生的通信**当作输入记下来：用哪个原语、消息多大、几个参与者、在不在关键路径上。以一个 7B 参数、hidden 4096、32 层、bf16 的 dense 模型和一个 MoE 模型为例，数字取量级。

```text
策略      原语                     每次消息大小（量级）                   参与者 n           关键路径
──────    ────────────────────     ──────────────────────────────────    ─────────────    ──────────────
DP        all_reduce（梯度）         DDP 按 bucket 发，默认 25 MiB 一桶      DP 度：8 ~ 数千    反向中可重叠
                                   （kDefaultBucketBytesCap）
                                   整体梯度 7B × 2 B = 14 GB（fp32 28 GB）
FSDP      all_gather（参数）         一层参数：~200 M × 2 B ≈ 400 MB         分片组：8 ~ 数百    前向/反向都在，可预取重叠
          reduce_scatter（梯度）     同上                                                    反向中重叠
TP        all_reduce（激活）         tokens × hidden × 2 B：                TP 度：2 ~ 8       每层 2 次，几乎无法重叠
                                   4096 tokens × 4096 × 2 B = 32 MB
PP        send / recv（激活）        micro-batch × seq × hidden × 2 B：      2（点对点）        阶段之间，靠 1F1B 调度隐藏
                                   1 × 4096 × 4096 × 2 B = 32 MB
MoE / EP  all_to_all（token）        tokens × top-k × hidden × 2 B 分散到    EP 度：8 ~ 64      每层 2 次（分发 + 收回）
                                   n 个目标
```

几个值得注意的量。**DP 的 all_reduce 是带宽账的典型**：25 MiB 的桶、总量十几 GB，反向传播一边算一边发，通信时间能否被计算掩盖取决于带宽而不是延迟。`kDefaultBucketBytesCap = 25 * 1024 * 1024` 定义在 PyTorch 的 `torch/csrc/distributed/c10d/reducer.hpp`（以 PyTorch 2.12 为准），第一个桶更小（`kDefaultFirstBucketBytes` 为 1 MiB），目的是让反向刚开始就有东西可发。

**TP 的 all_reduce 在关键路径上**。Megatron 风格的张量并行每一层有两次 all_reduce（attention 输出和 MLP 输出），下一步计算依赖它的结果，没有东西可以和它重叠。所以 TP 一般不出节点：8 卡 NVLink 上一次 32 MB 的 all_reduce 是几百 µs 量级，跨 IB 就是几 ms，乘以每层两次、几十层，差距直接体现在迭代时间上。

**FSDP 把 all_reduce 拆成了 reduce_scatter + all_gather**——正好是第二章的第一条组合等式。它的通信总量与 DP 相同（每个参数被 all_gather 一次、梯度被 reduce_scatter 一次，合起来还是 $$\frac{2(n-1)}{n}S$$），只是把两半分开放在了前向和反向。

**MoE 的 all_to_all** 消息大小取决于路由结果，每次不一样，且 $$n(n-1)$$ 条流同时打满链路；它是目前训练里最考验网络的模式，后面几篇多处会回到它。

### 2. 推理：两种几乎相反的模式

推理侧的通信模式只有两种，但两者对通信层的要求几乎相反。

**TP all_reduce**。prefill 阶段与训练的 TP 一样是几十 MB 的激活；decode 阶段每一步只处理每个序列一个 token，消息是 batch × hidden × 2 B。batch 8、hidden 8192 是 128 KB；batch 32 是 512 KB。每层两次，一个 80 层的模型每生成一个 token 要做 160 次这样的 all_reduce。这是**延迟账的典型**：按第五章的模型，8 卡 NVLink 上 128 KB 的 ring all_reduce 时间 97% 是 α，带宽多少无关紧要。第七篇的 custom all-reduce 就是为这个区间设计的。

**PD 分离的 KV 传输**。prefill 实例算完一个请求的 KV cache 后要把它搬到 decode 实例。每个 token 每层的 KV 大小是 $$2 \times \text{kv\_heads} \times \text{head\_dim} \times \text{bytes}$$，乘以层数再乘以序列长度。以 80 层、8 个 KV head（GQA）、head_dim 128、bf16 为例：每 token $$2 \times 8 \times 128 \times 2 \times 80 = 327{,}680$$ 字节，即 320 KB；一个 4096 token 的 prompt 是 1.28 GB。它是点对点的、目标动态、不需要归约、要与 decode 的计算完全解耦——这些特征让它更适合单边 RDMA WRITE 而不是集合通信（第七篇）。它是**带宽账的典型**：1.28 GB 在 50 GB/s 的 NDR 上是 26 ms，α 几乎可以忽略。

### 3. 消息大小的谱

把上面所有消息放到一条对数轴上：

```text
  64 KB     128 KB    512 KB      2 MB        25 MiB     32 MB     400 MB      1.3 GB     14 GB
   │          │         │          │            │          │          │           │          │
   │  decode TP all_reduce（batch 8 ~ 32）       │  DDP bucket │  TP/PP 激活（prefill/训练）  │
   │◄─────────────────────────►│                │◄──────────►│◄────────►│ FSDP 一层 │ KV cache │ 全部梯度
   │                           │                                                    （P2P）    （分桶发）
   │◄──── 延迟主导 ──────────►│◄── 拐点区 ──►│◄──────────────── 带宽主导 ─────────────────────►
   （8 卡 IB ring 的拐点约 2 MB，第五章算；NVLink 上拐点向右移，见第九章）
```

三个量级、三种账。第九章会把每一段的对策列出来：延迟主导的段合并消息或换算法与协议，带宽主导的段看链路与算法效率，拐点区两本账都要算。


## 四、α-β 模型

### 1. 一条消息：T = α + S/β

Hockney 在 1994 年给出的模型把一条点对点消息的传输时间写成两项之和：

$$
T(S) = \alpha + \frac{S}{\beta}
$$

$$\alpha$$（秒）是与消息大小无关的固定开销，$$\beta$$（字节/秒）是稳态下每秒能搬多少字节。这两个参数的物理含义在 GPU 通信里各自摊开是这样的：

```text
α 里面有什么                                    β 里面有什么
──────────────────────────────────────         ──────────────────────────────────────
kernel 启动、参数下发                             链路的物理速率（NVLink 一条链路、IB 一个端口）
两端的握手：接收方准备好了没有（flag / credit）      链路的协议效率（PCIe 包头、IB 报文头、NCCL 协议的 flag 开销）
线缆与交换机上的传播时延（每跳几百 ns ~ µs）          实际能同时用上几条链路（channel 数）
跨机时 CPU proxy 线程发现请求、提交 RDMA 的响应时间   端到端最慢的一段（PCIe → NIC → 网络 → NIC → PCIe 的最小值）
接收方轮询到完成标志的时间
```

模型说的是：一条链路在 $$S \to 0$$ 时时间趋于 $$\alpha$$，在 $$S \to \infty$$ 时带宽趋于 $$\beta$$。它对单条消息是一个相当好的近似，实测的 $$T(S)$$ 曲线通常就是一条直线加一个截距。

### 2. 拐点 S* = αβ

两项相等的消息大小是这个模型最有用的一个数：

$$
\alpha = \frac{S^*}{\beta} \quad\Longrightarrow\quad S^* = \alpha \beta
$$

小于 $$S^*$$ 的消息，时间的一半以上是 α；大于 $$S^*$$ 的消息，时间的一半以上是 $$S/\beta$$。这个数只依赖链路，与算法无关，是判断"这条链路上多大的消息才算大"的标尺。几种链路的典型量级（**数量级估计，非实测**；α 取的是一次 NCCL 步骤级的端到端固定开销，不是裸链路的传播时延）：

```text
链路                     β（单向）        α（每步，典型量级）    S* = αβ
──────────────────       ────────────    ───────────────────    ────────────
NVLink（节点内，NVSwitch）  ~200 GB/s      ~3 µs                  ~600 KB
                          （8 卡 ring 每 rank 通常能用到的量级；标称 H100 单向 450 GB/s）
PCIe 4.0 x16              32 GB/s         ~5 µs                  ~160 KB
InfiniBand HDR            25 GB/s         5 ~ 20 µs               125 KB ~ 500 KB
InfiniBand NDR            50 GB/s         5 ~ 20 µs               250 KB ~ 1 MB
```

有一个反直觉的地方值得先说：**链路越快，拐点越大**。NDR 的 α 与 HDR 差不多（延迟主要来自软件与握手，不是线速），但 β 翻倍，所以同样一条 256 KB 的消息在 HDR 上是带宽主导、在 NDR 上接近拐点。换更快的网卡对小消息帮助不大，这就是原因。

### 3. 模型忽略了什么

α-β 模型足够简单，也因此漏掉了几件后面几篇要补的事：

- **多条链路并行**。一张 GPU 有 18 条 NVLink、一台机器有 8 张网卡；NCCL 用多个 channel 同时走多条路，等效 β 是它们之和。模型里的 β 应理解为"这次通信实际用上的总带宽"，而不是单条链路。
- **双向**。NVLink、PCIe、IB 都是全双工，一个 rank 同时发和收互不抢带宽。本篇所有 β 都指**单向**，ring 的每个 rank 同时在发和在收，所以只算一个方向即可。
- **拷贝与归约**。数据到达接收端后还要做 $$\oplus$$、可能要在 buffer 之间拷贝；这些消耗 GPU 的访存带宽而不是链路带宽，在 NVLink 这种链路速度接近显存速度一个数量级以内的场合不能完全忽略。
- **拥塞与竞争**。多个流共享一条链路或一台交换机时 β 会下降、α 会抖动；all_to_all 和多任务共享网络时尤其明显。
- **同步**。集合通信要所有 rank 都到了才能开始，最慢的 rank 决定开始时间；模型里的 $$T$$ 是从"所有人都到了"起算的。straggler 问题不在模型里，但在 profiler 里非常常见。

这些都是模型与实测之间的差距来源，也就是"比一比"要解释的东西。本篇先把无差距的理论值算出来。

### 4. 集合通信在模型里

把 α-β 模型用到集合通信上只需要一步：算法决定这次通信分几步、每一步每个 rank 搬多少字节；每一步付一次 α，字节数除以 β 累加。写成通式：

$$
T_{\text{coll}} = (\text{步数}) \cdot \alpha + \frac{\text{每 rank 收发的字节数}}{\beta}
$$

这里隐含了一个假设：每一步所有 rank 同时在收发，链路是全双工的，所以时间由单个 rank 单方向的字节数决定。步数是延迟账，字节数是带宽账。接下来两章分别对 ring 和 tree 数这两个量。


## 五、ring all_reduce 的推导

### 1. 朴素做法为什么不行

最直接的 all_reduce：所有 rank 把 $$S$$ 发给 rank 0，rank 0 归约后广播回去。带宽账：rank 0 要接收 $$(n-1)S$$、再发出 $$(n-1)S$$，8 卡 1 GB 就是 7 GB 进、7 GB 出，全部压在一张卡的一条链路上，其他 7 张卡的链路几乎闲着。时间 $$\approx 2(n-1)S/\beta$$，随 $$n$$ 线性变差。延迟账倒是不错——2 步。

ring 的思想是把 rank 0 的活分给所有人：数据切成 $$n$$ 块，每个 rank 负责归约其中一块，负责的那块归约完了再传给所有人。这样每一步所有 rank 的链路都在满负荷工作，没有人闲着。

### 2. reduce_scatter 阶段：4 个 rank 逐步看

把 rank 排成环，rank $$r$$ 只向 rank $$(r+1) \bmod n$$ 发送、只从 rank $$(r-1) \bmod n$$ 接收。每个 rank 的数据切成 $$n$$ 块，记 rank $$r$$ 的第 $$k$$ 块为 $$c_k^{(r)}$$。以 $$n = 4$$ 为例，初始状态：

```text
          块 0        块 1        块 2        块 3
rank 0    c0(0)       c1(0)       c2(0)       c3(0)
rank 1    c0(1)       c1(1)       c2(1)       c3(1)
rank 2    c0(2)       c1(2)       c2(2)       c3(2)
rank 3    c0(3)       c1(3)       c2(3)       c3(3)
```

规则：第 $$s$$ 步（$$s = 1, \dots, n-1$$）rank $$r$$ 把自己手上第 $$(r - s + 1) \bmod n$$ 块的**当前部分和**发给下游，同时从上游收到第 $$(r - s) \bmod n$$ 块的部分和，加到自己那块上。用 `{0,1}` 表示"已累加了 rank 0 和 rank 1 的贡献"：

```text
第 1 步   r0 → r1 发块 0    r1 → r2 发块 1    r2 → r3 发块 2    r3 → r0 发块 3
          rank 0:  块 3 = {3,0}
          rank 1:  块 0 = {0,1}
          rank 2:  块 1 = {1,2}
          rank 3:  块 2 = {2,3}

第 2 步   r0 → r1 发块 3{3,0}   r1 → r2 发块 0{0,1}   r2 → r3 发块 1{1,2}   r3 → r0 发块 2{2,3}
          rank 0:  块 2 = {2,3,0}
          rank 1:  块 3 = {3,0,1}
          rank 2:  块 0 = {0,1,2}
          rank 3:  块 1 = {1,2,3}

第 3 步   r0 → r1 发块 2{2,3,0}   r1 → r2 发块 3{3,0,1}   r2 → r3 发块 0{0,1,2}   r3 → r0 发块 1{1,2,3}
          rank 0:  块 1 = {1,2,3,0}   ← 完整
          rank 1:  块 2 = {2,3,0,1}   ← 完整
          rank 2:  块 3 = {3,0,1,2}   ← 完整
          rank 3:  块 0 = {0,1,2,3}   ← 完整
```

上面每步只列了发生变化的那一块。把第 3 步结束时 4 个 rank 手上全部 16 块的状态摊开看，能看出 ring 的流水线结构——每一列（同一块）在 4 个 rank 上恰好是 1、2、3、4 份贡献的"阶梯"，完整的那份落在对角线上：

```text
第 3 步结束（reduce_scatter 完成）时的完整状态；✓ = 已含全部 4 个 rank 的贡献

          块 0           块 1           块 2           块 3
rank 0    {0}            {1,2,3,0} ✓    {2,3,0}        {3,0}
rank 1    {0,1}          {1}            {2,3,0,1} ✓    {3,0,1}
rank 2    {0,1,2}        {1,2}          {2}            {3,0,1,2} ✓
rank 3    {0,1,2,3} ✓    {1,2,3}        {2,3}          {3}

每列：4 份部分和分别累加了 1/2/3/4 个 rank 的贡献
每行：只有对角线上的 ✓ 块进入 all_gather 阶段，其余 3 块是早已发出的旧部分和
```

$$n - 1 = 3$$ 步之后，每个 rank 手上恰好有一块是完整归约结果：rank $$r$$ 持有第 $$(r+1) \bmod n$$ 块。这就是 reduce_scatter 的语义（块的归属是轮转的，不影响后续）。每一步每个 rank 发出一块 $$S/n$$ 字节、收进一块 $$S/n$$ 字节，两个方向同时进行。代价：

$$
T_{\text{RS}} = (n-1)\,\alpha + (n-1)\cdot\frac{S/n}{\beta} = (n-1)\,\alpha + \frac{n-1}{n}\cdot\frac{S}{\beta}
$$

### 3. all_gather 阶段

接下来的 $$n-1$$ 步，每个 rank 把手上刚完成的那块沿环传下去，收到的完整块直接覆盖本地对应位置（不再归约），再把它继续传给下游：

```text
第 4 步   r0 → r1 发块 1    r1 → r2 发块 2    r2 → r3 发块 3    r3 → r0 发块 0
          rank 0:  块 0、块 1 完整
          rank 1:  块 1、块 2 完整
          rank 2:  块 2、块 3 完整
          rank 3:  块 3、块 0 完整

第 5 步   r0 → r1 发块 0    r1 → r2 发块 1    r2 → r3 发块 2    r3 → r0 发块 3
          每个 rank 有 3 块完整

第 6 步   r0 → r1 发块 3    r1 → r2 发块 0    r2 → r3 发块 1    r3 → r0 发块 2
          每个 rank 有 4 块完整   ← all_reduce 完成
```

用 rank × 块 的网格看这三步，完整块（#）从 reduce_scatter 留下的对角线开始，每步沿环向下游多铺一格，直到铺满：

```text
                第 3 步后      第 4 步后      第 5 步后      第 6 步后
      块         0 1 2 3        0 1 2 3        0 1 2 3        0 1 2 3
rank 0           . # . .        # # . .        # # . #        # # # #
rank 1           . . # .        . # # .        # # # .        # # # #
rank 2           . . . #        . . # #        . # # #        # # # #
rank 3           # . . .        # . . #        # . # #        # # # #

#  = 已持有完整归约结果的块      每步每 rank 新增 1 块，共 n-1 = 3 步
```

步数与字节数与 reduce_scatter 阶段完全相同（第二章说的对偶关系）：

$$
T_{\text{AG}} = (n-1)\,\alpha + \frac{n-1}{n}\cdot\frac{S}{\beta}
$$

### 4. 总代价与它的两个极限

两段相加：

$$
T_{\text{ring}} = 2(n-1)\,\alpha + \frac{2(n-1)}{n}\cdot\frac{S}{\beta}
$$

每个 rank 收发的字节数是 $$\frac{2(n-1)}{n}S$$——正是第二章给出的 all_reduce 下界，所以 ring 是带宽最优的。看两个极限：

- **$$n \to \infty$$ 时带宽项 $$\to 2S/\beta$$**，与参与者数无关。1000 张卡和 8 张卡做同样大小的 all_reduce，带宽项几乎一样（$$\frac{2 \cdot 999}{1000} = 1.998$$ 对 $$\frac{2 \cdot 7}{8} = 1.75$$）。这是 ring 在大消息上无可替代的原因：加机器不会让每张卡的通信量变多。
- **延迟项 $$2(n-1)\alpha$$ 随 $$n$$ 线性增长**。8 张卡 14 步，1024 张卡 2046 步；α 取 10 µs 时分别是 140 µs 和 20 ms。这是 ring 在小消息、大规模上失败的原因：一个 64 KB 的消息在 1024 卡的环上要走 20 ms，比数据本身的传输时间大四个量级。

$$\frac{2(n-1)}{n}$$ 这个系数后面会反复出现：它是 busbw 的校正因子（第七章），也是 NCCL 调优模型里 ring 的步数系数（下一节）。

### 5. 核心问题的两个数字

现在回答总纲的核心问题。取 $$n = 8$$，$$\beta = 25$$ GB/s（HDR 单向标称，$$25 \times 10^9$$ B/s），$$\alpha = 10$$ µs（IB 上一步的典型量级，非实测）。

**S = 1 GB（$$10^9$$ 字节）**：

$$
\begin{aligned}
\text{延迟项} &= 2 \times 7 \times 10\ \mu\text{s} = 140\ \mu\text{s} \\
\text{带宽项} &= \frac{14}{8} \times \frac{10^9}{25 \times 10^9}\ \text{s} = 1.75 \times 40\ \text{ms} = 70\ \text{ms} \\
T_{\text{ring}} &\approx 70.14\ \text{ms}
\end{aligned}
$$

延迟项占 0.2%。α 就算估错 5 倍（50 µs）也只让总时间变化 1%。这个数字只对 β 敏感：换 NDR（50 GB/s）时间减半；链路只跑出 80% 带宽，时间长 25%。

**S = 64 KB（65,536 字节）**：

$$
\begin{aligned}
\text{延迟项} &= 140\ \mu\text{s} \\
\text{带宽项} &= 1.75 \times \frac{65536}{25 \times 10^9}\ \text{s} = 1.75 \times 2.62\ \mu\text{s} = 4.6\ \mu\text{s} \\
T_{\text{ring}} &\approx 145\ \mu\text{s}
\end{aligned}
$$

带宽项占 3%。这个数字只对 α 和步数敏感：换 NDR 几乎没有变化（少 2.3 µs）；把 α 从 10 µs 降到 5 µs 时间几乎减半；把 8 卡改成 4 卡（6 步）时间从 145 µs 降到 63 µs。

两个数字对不同变量敏感的原因，就是 $$T_{\text{ring}}$$ 两项在 $$S$$ 上的不同阶：第一项是常数，第二项与 $$S$$ 成正比。$$S$$ 差 15,000 倍，两项的比例从 3% : 97% 翻到 97% : 3%。

顺便算出 algbw 与 busbw（定义见第七章）：1 GB 时 algbw $$= 10^9 / 0.07014 \approx 14.3$$ GB/s，busbw $$= 14.3 \times 1.75 \approx 24.9$$ GB/s，几乎就是链路的 25 GB/s；64 KB 时 algbw $$\approx 0.45$$ GB/s，busbw $$\approx 0.79$$ GB/s，只有链路的 3%。这两个 busbw 数字就是第六章 nccl-tests 曲线两端的理论值。

### 6. ring 的集体拐点：S* = nαβ

第四章的拐点 $$S^* = \alpha\beta$$ 是单条消息的。对整个 ring all_reduce，令两项相等：

$$
2(n-1)\,\alpha = \frac{2(n-1)}{n}\cdot\frac{S^*_{\text{ring}}}{\beta} \quad\Longrightarrow\quad S^*_{\text{ring}} = n\,\alpha\,\beta
$$

拐点随 $$n$$ 线性右移。8 卡 IB（α = 10 µs，β = 25 GB/s）的拐点是 $$8 \times 250\ \text{KB} = 2$$ MB；64 卡是 16 MB；1024 卡是 256 MB。也就是说在 1024 卡的平坦 ring 上，一个 25 MiB 的 DDP bucket 仍然是延迟主导的——这就是为什么大规模训练必须用 tree 或分层算法，第六、八章讲。

把 $$T_{\text{ring}}(S)$$ 画在双对数坐标上，两项各自是一条直线（延迟项水平、带宽项斜率 1），曲线就是两条渐近线的"圆角拼接"，拐点是它们的交点：

![ring all_reduce 的 T(S) 双对数曲线：延迟项水平渐近线与带宽项斜线在 S* = nαβ 处相交；n 变大整条曲线左端抬高、拐点右移，链路变快只压低右半段](/img/in-post/collective-communication-primitives-and-cost-model-alpha-beta-regimes.svg)

三条曲线的差别正是两本账的差别：n 从 8 到 64，左端平台抬高 9 倍（步数 ×9）、拐点从 2 MB 右移到 16 MB，而右端几乎不动（$$\frac{2(n-1)}{n}$$ 只从 1.75 到 1.97）；换成 NVLink，右半段整体下压 8 倍（β ×8），左端平台只降到 42 µs（α 从 10 µs 到 3 µs）——所以 64 KB 的消息在两种链路上都在平台段，链路带宽对它无关紧要。

反过来，拐点也是一个实用的诊断量：从 nccl-tests 曲线上读出拐点位置，除以 $$n\beta$$，就反推出这台机器上一步的 α。


## 六、tree all_reduce 与 double binary tree

### 1. 二叉树的延迟：2 log₂ n

ring 的延迟随 $$n$$ 线性增长，因为环上每一步只能把信息往前传一格。要在 $$O(\log n)$$ 步内让每个 rank 拿到所有人的贡献，需要每一步让"知道的人"翻倍——这就是树。用第二章的第二条组合等式：all_reduce = reduce + broadcast。reduce 阶段数据沿二叉树向根归约，broadcast 阶段结果从根向叶广播。以 $$n = 8$$、一棵深度为 3 的二叉树为例：

```text
              r0                 reduce：叶 → 根，3 步；每个内部节点收两个孩子的数据、加上自己、发给父亲
              │                  broadcast：根 → 叶，3 步；每个内部节点收父亲的结果、转发给两个孩子
              r4
           ┌──┴──┐               （这是 NCCL 实际使用的形状：rank 0 为根、只有一个孩子，
          r2     r6                其余节点按编号二进制里最低位 1 的位置决定层级；奇数 rank 全是叶子）
        ┌─┴─┐  ┌─┴─┐
       r1   r3 r5   r7
```

步数 $$2\lceil\log_2 n\rceil$$：8 卡 6 步（对 14 步），1024 卡 20 步（对 2046 步）。1024 卡、α = 10 µs 时延迟项从 20 ms 降到 200 µs，差 100 倍。这是 tree 存在的全部理由。

### 2. 朴素二叉树的带宽账：为什么只有一半

延迟账赢了，看带宽账。把数据切成小块流水化（NCCL 的做法），reduce 阶段一个内部节点每处理一块要**从两个孩子各收一块**、发一块给父亲；整个阶段它的入向链路要承载 $$2S$$，出向 $$S$$。broadcast 阶段反过来：入向 $$S$$，出向 $$2S$$。叶子节点在 reduce 阶段只发不收（入向 0，出向 $$S$$），broadcast 阶段只收不发。

时间由最忙的链路决定。reduce 阶段瓶颈是内部节点的入向 $$2S$$，broadcast 阶段是内部节点的出向 $$2S$$，两阶段相加：

$$
T_{\text{tree,naive}} \approx 2\lceil\log_2 n\rceil\,\alpha + \frac{4S}{\beta}
$$

对比 ring 的带宽项 $$\frac{2(n-1)}{n}\frac{S}{\beta} \approx 2S/\beta$$，朴素二叉树只有**一半**的带宽。原因很直观：树上一半的 rank 是叶子，它们在每个阶段只用了链路的一个方向；另一半是内部节点，它们的一个方向要扛两倍流量。链路是全双工的，但树的结构让每个方向都只有一半的节点在用。

### 3. double binary tree：两棵互补的树

补救的办法是 2019 年 NCCL 2.4 引入的 double binary tree：建**两棵**树，第二棵的结构使得**第一棵里的叶子在第二棵里是内部节点、反之亦然**；数据切成两半，各走一棵树。

```text
树 A（处理前半数据 S/2）                树 B（处理后半数据 S/2；编号平移一位）
        r0                                      r1
        │                                       │
        r4                                      r5
     ┌──┴──┐                                 ┌──┴──┐
    r2     r6            ← 内部节点          r3     r7           ← 内部节点
  ┌─┴─┐  ┌─┴─┐                             ┌─┴─┐  ┌─┴─┐
  r1  r3 r5  r7          ← 叶子             r2  r4 r6  r0         ← 叶子

  A 的内部节点 {0,2,4,6} 在 B 里全是叶子；A 的叶子 {1,3,5,7} 在 B 里全是内部节点
```

（$$n$$ 为偶数时 NCCL 的第二棵树就是第一棵树把 rank 编号平移一位得到的，奇数时用镜像；见 NCCL 2.28.9 `src/graph/trees.cc` 里的 `ncclGetBtree` 与 `ncclGetDtree`，第四篇展开。）

现在算每个 rank 的链路负载。对树 A 来说 rank 3 是叶子，reduce 阶段入向 0、出向 $$S/2$$；对树 B 它是内部节点，入向 $$2 \times S/2 = S$$、出向 $$S/2$$。两棵树同时进行，rank 3 在 reduce 阶段的入向总量是 $$S$$、出向 $$S$$。反过来 rank 2 在 A 里是内部节点、在 B 里是叶子，账目一样。broadcast 阶段对称，也是入向 $$S$$、出向 $$S$$。每个 rank 都如此（两个根只有一个孩子，负载更轻，不构成瓶颈）。把单棵树和两棵树的每个方向的负载并排列出来，就能看到"补回一半"是怎么发生的：

| 阶段 | 方向 | 单棵树：叶子 | 单棵树：内部节点 | 双树：任一 rank（一棵里是叶、另一棵里是内部） |
|---|---|---|---|---|
| reduce | 入向 | 0 | **2S** ← 瓶颈 | 0 + 2·(S/2) = S |
| reduce | 出向 | S | S | S/2 + S/2 = S |
| broadcast | 入向 | S | S | S/2 + S/2 = S |
| broadcast | 出向 | 0 | **2S** ← 瓶颈 | 0 + 2·(S/2) = S |
| 合计（瓶颈方向） | | | 2S + 2S = **4S** | S + S = **2S** |

单棵树里每个阶段都有一半 rank 的一个方向空着（叶子的入向、内部节点的出向），另一半 rank 的对应方向扛 $$2S$$；双树让每个 rank 在两棵树里各扮一个角色，每个方向的负载都被拉平到 $$S$$。于是：

$$
T_{\text{tree,double}} \approx 2\lceil\log_2 n\rceil\,\alpha + \frac{2S}{\beta}
$$

带宽项回到了 $$2S/\beta$$，与 ring 的极限相同。两棵互补的树把每个 rank 两个方向的链路都用满了，代价是每个 rank 要同时维护两组连接、两条流水线。

### 4. Ring vs Tree：两本账对照

```text
                     Ring                                  Double Binary Tree
──────────────────   ─────────────────────────────────    ─────────────────────────────────
步数（延迟账）         2(n-1)                                2 ⌈log₂ n⌉
每 rank 字节数         2(n-1)/n · S                          ≈ 2S（两棵树各 S/2，进出各 S）
（带宽账）             n=8: 1.75 S    n=1024: 1.998 S         与 n 无关
带宽效率              最优（等于下界）                         接近最优，比 ring 差 n/(n-1)
                                                            NCCL 调优模型另乘 0.92 经验系数
实现复杂度            每 rank 一进一出、一条流水线              每 rank 两棵树、多组连接
适合的区间            大消息、小规模；节点内                    小消息、大规模；跨节点

数字（α = 10 µs，β = 25 GB/s；非实测）
  n=8,    S=1 GB     140 µs + 70 ms   ≈ 70.1 ms             60 µs + 80 ms   ≈ 80.1 ms      ring 胜
  n=8,    S=64 KB    140 µs + 4.6 µs  ≈ 145 µs              60 µs + 5.2 µs  ≈ 65 µs        tree 胜
  n=1024, S=1 GB     20.5 ms + 80 ms  ≈ 100 ms              200 µs + 80 ms  ≈ 80.2 ms      tree 胜
  n=1024, S=64 KB    20.5 ms + 5 µs   ≈ 20.5 ms             200 µs + 5 µs   ≈ 205 µs       tree 胜 100 倍
```

结论：8 卡以内、消息大，ring 赢一点；规模一大或消息一小，tree 赢很多。ring 的唯一优势是带宽账上那个 $$\frac{n-1}{n}$$，$$n = 8$$ 时值 12.5%，$$n = 64$$ 时只剩 1.6%。所以实际系统里 tree 是跨节点的默认，ring 是节点内的默认——NCCL 的选择大致就是这样，只是它还有 NVLS、CollNet 等更多选项，第四篇讲。

### 5. 看一看：NCCL 的调优模型里就是这几个公式

本篇不读 NCCL 内部，但有一处值得先指一下，因为它证明上面推的东西不是纸面练习。NCCL 2.28.9 的 `src/graph/tuning.cc` 里 `ncclTopoTuneModel` 为每种（原语 × 算法 × 协议）估算时间时，步数正是这样定义的：

```cpp
// src/graph/tuning.cc, ncclTopoTuneModel (NCCL 2.28.9)
int nsteps = coll == ncclFuncAllReduce ? 2*(nRanks-1) :
  coll == ncclFuncReduceScatter || coll == ncclFuncAllGather ? nRanks-1 :
  nRanks;
// ...
if (a == NCCL_ALGO_TREE && coll == ncclFuncAllReduce) busBw = std::min(busBw*.92, ...);
// ...
} else if (a == NCCL_ALGO_TREE) {
  if (coll == ncclFuncAllReduce) {
    comm->latencies[coll][a][p] +=
      2 * ((nRanks/nNodes-1) * intraLat + log2i(nNodes) * interLat);
  }
```

ring all_reduce 走 $$2(n-1)$$ 步、reduce_scatter 与 all_gather 走 $$n-1$$ 步，与第五章一致；tree 的带宽乘 0.92 的经验系数，延迟按"节点内链 + 节点间树"算——后一条是第八章分层算法的内容。NCCL 用这套 α-β 模型给每个候选算法打分、选最快的那个，`NCCL_ALGO` / `NCCL_PROTO` 环境变量（在 `tuning.cc` 与 `src/enqueue.cc` 里以 `ncclGetEnv("NCCL_ALGO")` 读取）可以覆盖这个选择。第四篇会把这张调优表整个读一遍。


## 七、algbw 与 busbw

### 1. 两个定义与 nccl-tests 的源码

nccl-tests 对每一个消息大小打印两个带宽。**algbw**（algorithm bandwidth）是最直接的定义：

$$
\text{algbw} = \frac{S}{T}
$$

"用户看到的带宽"——传了 $$S$$ 字节的数据、花了 $$T$$ 秒。**busbw**（bus bandwidth）把 algbw 乘上第二章那张表里的系数，换算成"每个 rank 的链路上实际流过的字节速率"。以 nccl-tests 2.18.3 的 `src/all_reduce.cu` 的 `AllReduceGetBw` 为准：

```cpp
// nccl-tests src/all_reduce.cu
void AllReduceGetBw(size_t count, size_t typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(2*(nranks - 1)))/((double)nranks);
  *busBw = baseBw * factor;
}
```

`/ 1.0E9` 说明 nccl-tests 的 GB/s 是 $$10^9$$ 字节每秒，与网卡的 Gb/s 换算一致（HDR 200 Gb/s = 25 GB/s），与 GiB 无关；`factor` 就是 $$\frac{2(n-1)}{n}$$。其他几个原语（各在同名 `.cu` 文件的 `*GetBw` 函数）：

```cpp
// src/all_gather.cu  AllGatherGetBw
double baseBw = (double)(count * typesize * nranks) / 1.0E9 / sec;   // S 是 n 份拼接后的总量
double factor = ((double)(nranks - 1))/((double)nranks);

// src/reduce_scatter.cu  ReduceScatterGetBw
double baseBw = (double)(count * typesize * nranks) / 1.0E9 / sec;   // S 是输入总量
double factor = ((double)(nranks - 1))/((double)nranks);

// src/alltoall.cu  AlltoAllGetBw
double baseBw = (double)(count * nranks * typesize) / 1.0E9 / sec;
double factor = ((double)(nranks-1))/((double)(nranks));

// src/broadcast.cu  BroadcastGetBw
double factor = 1;

// src/reduce.cu  ReduceGetBw
*busBw = baseBw;                                                      // factor 1
```

注意 all_gather、reduce_scatter、all_to_all 三个的 `baseBw` 里多乘了 `nranks`：这里的 `count` 是每个 rank 的那一份（由各文件的 `*GetCollByteCount` 算出，例如 `AllGatherGetCollByteCount` 把 `sendcount` 设为 `count/nranks`），乘回 `nranks` 后 $$S$$ 是总量，与第二章的约定一致。`common.cu` 的 `BenchTime` 在计时后调用 `args->collTest->getBw(...)`，传入的 `nranks` 是 `nProcs * nThreads * nGpus`，即 communicator 的总 rank 数。

### 2. 各原语的系数表

```text
原语              nccl-tests 里的 S                busbw / algbw       n=8 时     n→∞
──────────────    ─────────────────────────────    ─────────────────   ────────   ─────
all_reduce        每 rank 的 buffer                2(n-1)/n            1.75       2
all_gather        n 份拼接后的总量                  (n-1)/n             0.875      1
reduce_scatter    输入总量（n × 每 rank 输出）        (n-1)/n             0.875      1
all_to_all        每 rank 输入总量                  (n-1)/n             0.875      1
broadcast         buffer 大小                      1                   1          1
reduce            buffer 大小                      1                   1          1
```

系数就是第二章"每 rank 至少接收多少字节"那一列除以 $$S$$。它假设的是**带宽最优算法**下每个 rank 的链路流量；用其他算法（比如朴素树）实际流量会更多，但 busbw 仍按这个系数算——所以 busbw 是"等效 ring 流量"的口径，不是链路的实测流量。

### 3. 为什么只有 busbw 能与链路带宽比较

拿第五章的数字看。8 卡 1 GB 的 ring all_reduce 理论 70.14 ms，algbw = 14.3 GB/s。这个数字与 25 GB/s 的链路没有直接关系：把 8 卡换成 2 卡，同样的链路、同样跑满，$$T = 2 \times 1 \times 10\ \mu\text{s} + 1.0 \times 40\ \text{ms} \approx 40$$ ms，algbw 变成 25 GB/s；换成 64 卡，$$T = 126 \times 10\ \mu\text{s} + 1.97 \times 40\ \text{ms} \approx 80$$ ms，algbw 变成 12.5 GB/s。链路一直是那条链路、一直是满的，algbw 却随 $$n$$ 变。

busbw 把 $$\frac{2(n-1)}{n}$$ 乘回去：2 卡 25.0 GB/s，8 卡 24.9 GB/s，64 卡 24.6 GB/s——**只要链路跑满、延迟项可忽略，busbw 就等于链路单向带宽，与 n 无关**。把这几组数并排（S = 1 GB，β = 25 GB/s，α = 10 µs，ring，非实测）：

| n | 步数 × α | 每 rank 字节数 / β | T | algbw = S/T | 系数 2(n-1)/n | busbw |
|---|---|---|---|---|---|---|
| 2 | 20 µs | 1.00 × 40 ms | 40.0 ms | 25.0 GB/s | 1.00 | 25.0 GB/s |
| 8 | 140 µs | 1.75 × 40 ms | 70.1 ms | 14.3 GB/s | 1.75 | 24.9 GB/s |
| 64 | 1.26 ms | 1.97 × 40 ms | 80.0 ms | 12.5 GB/s | 1.97 | 24.6 GB/s |
| 1024 | 20.5 ms | 1.998 × 40 ms | 100.4 ms | 10.0 GB/s | 1.998 | 19.9 GB/s |

algbw 一列随 $$n$$ 从 25 掉到 10，busbw 一列在 25 附近不动。（1024 卡的 ring 会降到 19.9 GB/s，但那 20% 是 2046 步的 α 账，不是链路的账；换 tree 就回到 24.9。）这就是它能与硬件标称值直接比的原因：nccl-tests 的 busbw 平台是 23 GB/s、链路是 25 GB/s，你立刻知道链路效率 92%；如果平台是 12 GB/s，你知道差了一倍，该去查路径、GDR、channel 数（第六篇）。

两个数字各有用处。algbw 回答用户的问题："我的 1 GB 梯度多久能同步完"——它就是 $$S/T$$，与你关心的张量直接相关。busbw 回答工程师的问题："链路跑满了没有"。看曲线时看 busbw，算迭代时间时用 algbw。

一个补充：在节点内用 NVSwitch 的 NVLS 或跨节点用 SHARP 时，归约在交换机里完成，每个 rank 实际只需发一份、收一份（$$\approx S$$ 而不是 $$2S$$），此时按 $$\frac{2(n-1)}{n}$$ 算出来的 busbw 会**超过**链路单向带宽。这不是测错，是 busbw 的 ring 等效口径在非 ring 算法下的表现。第四篇讲 NVLS 时会回到这一点；第八章的两级算法算例里也会看到同样的现象。

### 4. 测一测、比一比：从曲线上能读出什么

把本篇的模型和 nccl-tests 的曲线对起来，有四个可以直接读出的量。第六篇会展开工具用法，这里给出读法：

- **右端平台高度 vs β**：平台是 busbw 的饱和值，理论上等于链路单向带宽乘以协议效率。8 卡 HDR 上应在 20–23 GB/s（"通常能达到"，非实测）；平台远低于此，是路径或配置问题，不是算法问题。
- **左端平台 vs 步数 × α**：最小消息的时间 $$T_{\min} \approx (\text{步数}) \times \alpha$$。8 卡 ring 在 IB 上若 $$T_{\min} = 140$$ µs，则 α ≈ 10 µs；若是 400 µs，α ≈ 30 µs，该查跨 NUMA、proxy 线程、协议选择。
- **拐点位置 vs nαβ**：曲线达到平台一半高度的 $$S$$ 大约是 $$n\alpha\beta$$。已知 $$n$$ 和 β，反推 α；与左端平台反推的 α 对不上，说明中间段有额外开销（比如算法切换点选得不好）。
- **不同 n 的曲线平台是否重合**：按 busbw 画，2 卡与 8 卡的平台应该基本重合；不重合说明拓扑上某些 rank 之间的路径比其他 rank 差（比如跨了 PCIe switch 或 NUMA）。

这四条是本篇给第六篇"比一比"的检查项，它们不需要机器，只需要模型。


## 八、分层与多级算法

### 1. 两个 β、两个 α

到目前为止的模型假设所有链路一样。真实的集群不是：节点内 8 张卡通过 NVSwitch 互联，每张卡单向几百 GB/s、一步几 µs；节点之间每张卡一张 400 Gb/s 网卡，单向 50 GB/s、一步十几 µs。两者带宽差 4–8 倍，延迟差 3–5 倍。

一个跨 4 个节点、32 张卡的平坦 ring 会怎样？环上 32 条边中有 4 条跨节点（每个节点进一条、出一条），28 条在节点内。ring 的每一步所有 rank 同时收发，一步的时间由**最慢的那条边**决定；带宽项是 $$\frac{2(n-1)}{n}\frac{S}{\beta_{\min}}$$，$$\beta_{\min}$$ 是节点间链路。更糟的是每个节点只有一条出边跨节点，8 张网卡只用了 1 张——除非 NCCL 建多条环让不同的环从不同网卡出去（它确实这么做，这就是 channel，第四篇）。

分层的思路是让数据在快的链路上多走、在慢的链路上少走。

### 2. 两级 all_reduce 的代价

$$N$$ 个节点、每节点 $$p$$ 张卡（$$n = Np$$）。节点内参数 $$(\alpha_i, \beta_i)$$，节点间 $$(\alpha_e, \beta_e)$$。三步：

```text
第 1 步   节点内 reduce_scatter      p 个 rank 的 ring，S 字节      → 每张卡持有 S/p 的节点内归约结果
第 2 步   节点间 all_reduce          每张卡与其他节点上同一位置的卡    → p 个并行的 N-rank ring，各 S/p 字节
                                    组成一个 N-rank 的环，走自己的网卡
第 3 步   节点内 all_gather          p 个 rank 的 ring，S 字节      → 每张卡拿到完整结果
```

把 $$N = 4$$ 个节点、每节点 $$p = 8$$ 张卡排成 4 × 8 的网格（行 = 节点，列 = 卡在节点内的位置），三步分别是"横着走"和"竖着走"：

```text
            卡0    卡1    卡2    卡3    卡4    卡5    卡6    卡7
          ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
节点 0    │◀─── 第 1 步 / 第 3 步：节点内 8-rank ring，NVLink ───▶│
          ├──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
节点 1    │◀──────────────── 节点内 ring，S 字节 ────────────────▶│
          ├──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
节点 2    │◀──────────────── 节点内 ring，S 字节 ────────────────▶│
          ├──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
节点 3    │◀──────────────── 节点内 ring，S 字节 ────────────────▶│
          └──┬───┴──┬───┴──┬───┴──┬───┴──┬───┴──┬───┴──┬───┴──┬───┘
             ▲      ▲      ▲      ▲      ▲      ▲      ▲      ▲
             ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼
          第 2 步：每一列 4 张卡组成一个 4-rank ring，各搬 S/8，
                   走该列各卡自己的网卡（IB）；8 条 ring 同时进行
```

第 1 步之后第 $$j$$ 列的卡持有的是本节点第 $$j$$ 块（$$S/p$$）的节点内归约结果，所以第 2 步只需要**同一列**的 4 张卡之间做 all_reduce；4 个节点各 8 张网卡，每张网卡恰好服务一列。

代价：

$$
T_{\text{2-level}} = \underbrace{2\left[(p-1)\,\alpha_i + \frac{p-1}{p}\frac{S}{\beta_i}\right]}_{\text{节点内两段}} + \underbrace{2(N-1)\,\alpha_e + \frac{2(N-1)}{N}\cdot\frac{S/p}{\beta_e}}_{\text{节点间}}
$$

关键在节点间那一项的分子：$$S/p$$ 而不是 $$S$$。节点间要搬的总字节数没有少（$$p$$ 张卡各搬 $$S/p$$），但被摊到了 $$p$$ 张网卡上，每张网卡只需承载 $$1/p$$。

用数字比一比。$$N = 4$$，$$p = 8$$，$$S = 1$$ GB，$$\beta_e = 25$$ GB/s，$$\alpha_e = 10$$ µs，$$\beta_i = 200$$ GB/s（8 卡 ring 每 rank 在 NVSwitch 上通常能用到的量级，非实测），$$\alpha_i = 3$$ µs：

```text
两级
  节点内 reduce_scatter   7 × 3 µs + 7/8 × 1 GB / 200 GB/s   =  21 µs + 4.4 ms
  节点间 all_reduce       6 × 10 µs + 6/4 × 125 MB / 25 GB/s  =  60 µs + 7.5 ms
  节点内 all_gather       7 × 3 µs + 7/8 × 1 GB / 200 GB/s   =  21 µs + 4.4 ms
  合计                                                        ≈ 16.4 ms

平坦 ring（32 rank，每步受节点间链路限制，每节点一张网卡承载全部跨节点流量）
  62 × 10 µs + 62/32 × 1 GB / 25 GB/s                        = 0.6 ms + 77.5 ms ≈ 78 ms
```

差 4.7 倍。差距的来源就是 $$S/p$$：节点间那一段从 77.5 ms 降到 7.5 ms，代价是节点内多走两段共 8.8 ms。当 $$\beta_i \gg \beta_e$$ 时这笔交换永远划算。

第七章说过 busbw 在非 ring 算法下会超过链路带宽，这里就是一个例子：两级算法的 algbw $$= 1\ \text{GB} / 16.4\ \text{ms} \approx 61$$ GB/s，busbw $$= 61 \times \frac{62}{32} \approx 118$$ GB/s，远超网卡的 25 GB/s——因为大部分字节走的是 NVLink。

### 3. 延迟账上的分层

分层对延迟账同样有效，且更明显。上面的两级算法步数是 $$2(p-1) + 2(N-1) = 14 + 6 = 20$$，平坦 ring 是 62；如果节点间用 tree，步数是 $$2(p-1) + 2\log_2 N = 14 + 4 = 18$$。更重要的是**哪些步付的是小 α、哪些步付的是大 α**：两级算法里 14 步付 $$\alpha_i = 3$$ µs、6 步付 $$\alpha_e = 10$$ µs，合计 102 µs；平坦 ring 的 62 步里虽然只有 8 步真正跨节点，但每一步所有 rank 要同步，实际每步都按最慢的算，合计 620 µs。

第六章第 5 节引用的 NCCL tree 延迟公式 $$2\,[(p-1)\,\alpha_i + \log_2 N \cdot \alpha_e]$$ 正是这个结构：节点内一条链（$$p - 1$$ 步）、节点间一棵树（$$\log_2 N$$ 步），reduce 与 broadcast 各一遍。NCCL 的 tree 算法本身就是分层的。

### 4. 分层的代价与前提

分层不是免费的。它假设"同一位置的卡"之间有独立的网卡（每 GPU 一张 NIC，这是第二篇讲的 rail-optimized 设计的原因），假设节点内的链路确实比节点间快得多（PCIe 机器上 $$\beta_i$$ 可能只有 $$\beta_e$$ 的一两倍，分层的收益就小），也假设 $$p$$ 张网卡能同时跑满（NIC 与 GPU 的 PCIe 亲和不对时不能，第二、三篇）。另外，节点内两段各自要付 $$(p-1)\alpha_i$$，小消息时这一项不可忽略：64 KB 的 all_reduce 在两级算法下是 $$14 \times 3 + 6 \times 10 = 102$$ µs，仍是纯延迟主导，分层帮不了它——它需要的是更少的步数（tree）、更低的 α（协议、custom kernel）或者干脆不做这么小的通信（合并）。


## 九、把消息大小的谱放到曲线上

### 1. 三个量级、三种账

回到第三章那条对数轴，现在可以给每一段填上两本账的比例。以 8 卡为单位，两种链路（IB HDR：α = 10 µs、β = 25 GB/s；NVLink：α = 3 µs、β = 200 GB/s；均为典型量级、非实测），ring all_reduce：

```text
消息            场景                        IB HDR 8 卡                    NVLink 8 卡
                                           T          延迟占比            T          延迟占比
─────────────   ───────────────────────    ─────────  ────────           ─────────  ────────
64 KB           decode TP，batch 4         145 µs     97%                43 µs      98%
128 KB          decode TP，batch 8         149 µs     94%                43 µs      97%
512 KB          decode TP，batch 32        176 µs     80%                47 µs      90%
2 MB            ring 拐点（IB）             287 µs     49%                60 µs      70%
25 MiB          DDP 一个 bucket             1.98 ms    7%                 271 µs     15%
32 MB           prefill / 训练 TP 激活      2.5 ms     6%                 336 µs     13%
400 MB          FSDP 一层 all_gather        （用 (n-1)/n 系数）14 ms      1.8 ms
1 GB            大梯度 / KV cache 量级      70 ms      0.2%               8.8 ms     0.5%
```

三个区间的性质：

- **几十到几百 KB（decode TP）**：两种链路上都是 90% 以上的延迟。换更快的链路几乎无效；有效的是减少步数（tree、one-shot 的 custom all-reduce）、降低每步的 α（LL 协议、绕开 kernel 启动与 proxy）、或减少通信次数。第七篇的 custom all-reduce 把 8 卡 128 KB 从 NCCL 的几十 µs 降到十几 µs，省的全是 α。
- **几 MB 到几十 MB（DDP bucket、TP/PP 激活）**：IB 上处于拐点附近到带宽主导之间，两本账都要看；NVLink 上已经是带宽主导。DDP 把桶设成 25 MiB 而不是 1 MB，就是为了把梯度同步从拐点左边推到右边——8 卡 IB 上 25 个 1 MiB 的消息要 $$25 \times (140 + 73)\ \mu\text{s} \approx 5.3$$ ms，一个 25 MiB 的消息只要 1.98 ms。
- **几百 MB 到 GB（FSDP 大层、KV cache、整体梯度）**：纯带宽主导。看的是链路速率、算法效率、能否用上所有网卡、GDR 是否生效、能否与计算重叠。α 估错一个量级对结果没有影响。

### 2. 一次训练迭代的通信账

把模型用到一个具体任务上。7B dense 模型、bf16 梯度 14 GB、64 卡 DDP（8 节点 × 8 卡）、每卡一张 NDR 网卡（50 GB/s 单向）、两级 ring：

```text
节点间：每张卡承载 14 GB / 8 = 1.75 GB，N = 8 的 ring，带宽项 2×7/8 × 1.75 GB / 50 GB/s = 61 ms
节点内：两段，各 7/8 × 14 GB / 200 GB/s = 61 ms，合计 122 ms      ← 与节点间同量级！
延迟：  14 × 3 µs + 14 × 10 µs ≈ 0.2 ms，忽略
合计：  ≈ 183 ms
```

这个算例暴露了分层的一个盲点：节点内的两段各要在 NVLink 上搬 $$\frac{7}{8} \times 14$$ GB，与节点间的时间相当。真实的 NCCL 用多 channel 让节点内多条 ring 并行，实际 $$\beta_i$$ 能用到更高（H100 NVSwitch 上 all_reduce 的 busbw 通常能到 300–400 GB/s 量级，非实测），节点内的份额会缩小；但"节点内不是免费的"这一点在 FSDP 这种通信量与 DP 相同、却不能完全重叠的场合会体现出来。反向传播如果是 500 ms，183 ms 的通信能否藏进去，取决于第五篇讲的重叠机制。

### 3. 比一比：从一个实测数字反推该查哪一层

本篇没有实测，但可以给出拿到实测数字后的判断步骤，这是模型作为尺子的用法：

1. **算理论值**。用 `cost_model.py`（下一章）输入 $$n$$、$$S$$、链路标称 β、量级 α，得到 $$T_{\text{model}}$$ 与 busbw。
2. **判断区间**。看延迟占比：> 80% 是延迟主导，< 20% 是带宽主导，中间是拐点区。
3. **带宽主导时**，比 busbw：实测 / 标称 ≥ 80% 正常；50–80% 查协议效率与 channel 数；< 50% 查路径（是否走了 PCIe 而非 NVLink、是否走了 Socket 而非 IB、GDR 是否生效）——第二、三、六篇。
4. **延迟主导时**，比 $$T / \text{步数}$$ 反推 α：IB 上一步 5–20 µs、NVLink 上 2–5 µs 正常；显著高于此查协议选择、跨 NUMA、proxy 线程、框架侧是否有额外同步——第四、五、六篇。
5. **拐点区**，两个都算；先确认 β 正常再看 α，因为 β 的问题更容易定位。
6. **模型不适用的情形**：all_to_all（流量模式不同）、有 straggler（所有 rank 等最慢的）、通信与计算重叠时被抢了 SM——这些在 profiler 里表现为通信时间远大于模型值且方差大。

前两步不需要机器，是本篇能给的；后四步每一步都指向后面某一篇。


## 十、本文小结

### 1. 要点回顾

- 集合通信有八个原语；all_reduce = reduce_scatter + all_gather 是 ring 的基础，all_reduce = reduce + broadcast 是 tree 的基础；broadcast/reduce、scatter/gather、all_gather/reduce_scatter 互为对偶，实现上可以互相翻转。
- 每个原语每 rank 至少接收的字节数：all_reduce $$\frac{2(n-1)}{n}S$$，all_gather / reduce_scatter / all_to_all $$\frac{n-1}{n}S$$，broadcast / reduce $$S$$。这一列就是 busbw 的系数。
- 训练：DP 的梯度 all_reduce（25 MiB 桶，带宽账）、FSDP 的 all_gather / reduce_scatter（几百 MB 一层）、TP 的 all_reduce（几十 MB、关键路径、不出节点）、PP 的 send/recv、MoE 的 all_to_all（$$n(n-1)$$ 条流）。推理：decode TP 的 all_reduce（几十到几百 KB，延迟账）、PD 分离的 KV 传输（GB 级点对点，带宽账）。
- α-β 模型：$$T = \alpha + S/\beta$$；单消息拐点 $$S^* = \alpha\beta$$；链路越快拐点越大，换快网卡对小消息无效。
- ring all_reduce：$$T = 2(n-1)\alpha + \frac{2(n-1)}{n}\frac{S}{\beta}$$。带宽项 $$\to 2S/\beta$$ 与 $$n$$ 无关（大消息最优），延迟项随 $$n$$ 线性增长（小消息、大规模失败）。集体拐点 $$S^* = n\alpha\beta$$，8 卡 IB 约 2 MB。
- 核心问题：8 卡、25 GB/s、α = 10 µs，1 GB 约 70 ms（延迟占 0.2%，只对 β 敏感），64 KB 约 145 µs（带宽占 3%，只对 α 与步数敏感）。
- tree：延迟 $$2\lceil\log_2 n\rceil\alpha$$；朴素二叉树带宽只有 ring 一半（叶子只用链路一个方向、内部节点一个方向扛两倍）；double binary tree 用两棵互补的树各走一半数据，把带宽补回 $$2S/\beta$$。
- algbw = $$S/T$$ 随 $$n$$ 变，回答"我的张量多久同步完"；busbw = algbw × 系数，链路跑满时等于链路单向带宽、与 $$n$$ 无关，是唯一能与硬件标称值直接比的数字。非 ring 算法（NVLS、两级）下 busbw 可超过链路带宽。
- 分层：节点内 reduce_scatter → 节点间 all_reduce（$$S/p$$，走 $$p$$ 张网卡）→ 节点内 all_gather；节点间流量每网卡减到 $$1/p$$，4 节点 32 卡的算例快 4.7 倍。NCCL 的 tree 本身就是"节点内链 + 节点间树"的分层结构。
- 消息大小的谱：几十 KB 是纯延迟（减步数、降 α、合并），几 MB 到几十 MB 是拐点区（两本账都算），几百 MB 以上是纯带宽（链路、效率、多网卡、重叠）。

### 2. 公式速查

```text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
α-β 模型
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
单条消息            T = α + S/β                          拐点 S* = αβ
集合通信通式         T = 步数 × α + 每 rank 字节数 / β      β 为单向；全双工下收发不互抢

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
算法                 步数（延迟账）          每 rank 字节数（带宽账）         拐点
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ring all_reduce      2(n-1)                 2(n-1)/n · S  → 2S             S* = nαβ
ring all_gather      n-1                    (n-1)/n · S   → S              S* = nαβ
ring reduce_scatter  n-1                    (n-1)/n · S   → S              S* = nαβ
朴素二叉树 all_reduce 2⌈log₂n⌉               4S（瓶颈链路）
double binary tree   2⌈log₂n⌉               2S
两级 ring（N 节点×p）  2(p-1) 内 + 2(N-1) 外   2(p-1)/p · S / β_i + 2(N-1)/N · (S/p) / β_e
朴素 reduce+bcast    2                      (n-1)S 进 + (n-1)S 出（root）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
nccl-tests 的两个带宽（GB/s = 10^9 B/s）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
algbw = S / T
busbw = algbw × 系数     all_reduce 2(n-1)/n    all_gather / reduce_scatter / all_to_all (n-1)/n
                         broadcast / reduce 1
S 的口径                 all_gather 为拼接后总量；reduce_scatter 为输入总量；其余为每 rank buffer

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
典型量级（非实测；单向）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
β     IB HDR 25 GB/s · IB NDR 50 GB/s · PCIe 4.0 x16 32 GB/s · NVLink 8 卡 ring 每 rank ~200 GB/s
α     IB 上一步 5–20 µs · NVLink 上一步 2–5 µs
8 卡  IB ring 拐点 ~2 MB · NVLink ring 拐点 ~5 MB
从曲线反推   α ≈ T_min / 步数 ；  β ≈ busbw 平台 / 协议效率 ；  α ≈ 拐点 S / (nβ)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
核心问题（n=8, β=25 GB/s, α=10 µs）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1 GB     140 µs + 70 ms  ≈ 70.1 ms     algbw 14.3 · busbw 24.9 GB/s     带宽主导，对 β 敏感
64 KB    140 µs + 4.6 µs ≈ 145 µs      algbw 0.45 · busbw 0.79 GB/s     延迟主导，对 α 与步数敏感
```

### 3. 本篇涉及的源码与工具位置

| 路径 | 内容 |
|---|---|
| nccl-tests `src/all_reduce.cu` | `AllReduceGetBw`：algbw = S/T，busbw 系数 $$2(n-1)/n$$；`AllReduceGetCollByteCount` |
| nccl-tests `src/all_gather.cu`、`src/reduce_scatter.cu`、`src/alltoall.cu` | `AllGatherGetBw`、`ReduceScatterGetBw`、`AlltoAllGetBw`：baseBw 乘 `nranks` 得总量，系数 $$(n-1)/n$$ |
| nccl-tests `src/broadcast.cu`、`src/reduce.cu` | `BroadcastGetBw`、`ReduceGetBw`：系数 1 |
| nccl-tests `src/common.cu` | `BenchTime` 计时后调用 `collTest->getBw(...)`；`nranks = nProcs * nThreads * nGpus` |
| NCCL 2.28.9 `src/graph/tuning.cc` | `ncclTopoTuneModel`：ring 步数 $$2(n-1)$$ / $$n-1$$，tree 带宽 ×0.92，tree 延迟 $$2[(p-1)\alpha_i + \log_2 N\,\alpha_e]$$；读取 `NCCL_ALGO` / `NCCL_PROTO` |
| NCCL 2.28.9 `src/graph/trees.cc` | `ncclGetBtree`、`ncclGetDtree`：单棵与 double binary tree 的构造 |
| PyTorch `torch/csrc/distributed/c10d/reducer.hpp` | `kDefaultBucketBytesCap = 25 MiB`、`kDefaultFirstBucketBytes = 1 MiB`（DDP bucket 默认值） |
| comm-probe `cost_model.py` | 本篇增量，见下 |

### 4. comm-probe 本篇增量：cost_model.py

comm-probe 的第一个工具是本篇公式的可执行版本。输入 $$n$$、$$S$$、α、β 与算法，输出理论时间、延迟占比、algbw 与 busbw。它不联网、不需要 GPU，后面六篇的每一个实测数字都拿它做参照。

```python
#!/usr/bin/env python3
"""comm-probe / cost_model.py -- alpha-beta cost model for collectives.

T = steps * alpha + bytes_per_rank / beta
algbw = S / T ; busbw = algbw * factor(coll, n)   (factor follows nccl-tests src/*.cu)
"""
import argparse, math

UNITS = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30}

def parse_size(s):
    s = s.strip().lower().rstrip("b")
    for u, mult in UNITS.items():
        if s.endswith(u):
            return int(float(s[:-1]) * mult)
    return int(float(s))

def busbw_factor(coll, n):
    if coll == "all_reduce":
        return 2 * (n - 1) / n
    if coll in ("all_gather", "reduce_scatter", "all_to_all"):
        return (n - 1) / n
    return 1.0                                  # broadcast / reduce

def ring(coll, n, S, alpha, beta):
    """returns (latency_part, bandwidth_part) for a ring on one link of bandwidth beta."""
    if coll == "all_reduce":
        steps, nbytes = 2 * (n - 1), 2 * (n - 1) / n * S
    elif coll in ("all_gather", "reduce_scatter", "all_to_all"):
        steps, nbytes = n - 1, (n - 1) / n * S
    else:                                       # broadcast / reduce: pipelined chain
        steps, nbytes = n - 1, S
    return steps * alpha, nbytes / beta

def tree(coll, n, S, alpha, beta):
    """double binary tree all_reduce: 2*ceil(log2 n) steps, 2S per rank."""
    if coll != "all_reduce":
        raise SystemExit("tree model implemented for all_reduce only")
    return 2 * math.ceil(math.log2(n)) * alpha, 2 * S / beta

def ring_allreduce_time(n, S, alpha_us, beta_gbps):
    """importable helper for later tools (nccl_log_reader.py, plot_sweep.py): returns T in us."""
    lat, bw = ring("all_reduce", n, S, alpha_us * 1e-6, beta_gbps * 1e9)
    return (lat + bw) * 1e6

def two_level(coll, n, S, alpha, beta, ppn, alpha_i, beta_i):
    """intra reduce_scatter -> inter ring all_reduce on S/ppn -> intra all_gather."""
    if coll != "all_reduce":
        raise SystemExit("2level model implemented for all_reduce only")
    l1, b1 = ring("reduce_scatter", ppn, S, alpha_i, beta_i)       # == all_gather cost
    l2, b2 = ring("all_reduce", n // ppn, S / ppn, alpha, beta)
    return 2 * l1 + l2, 2 * b1 + b2

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, required=True, help="number of ranks")
    p.add_argument("--size", required=True, help="S, e.g. 64K, 25M, 1G (total bytes)")
    p.add_argument("--alpha", type=float, default=10.0, help="per-step latency, us")
    p.add_argument("--beta", type=float, default=25.0, help="link bandwidth, GB/s (1e9 B/s)")
    p.add_argument("--algo", choices=["ring", "tree", "2level"], default="ring")
    p.add_argument("--coll", default="all_reduce",
                   choices=["all_reduce", "all_gather", "reduce_scatter",
                            "all_to_all", "broadcast", "reduce"])
    p.add_argument("--ppn", type=int, default=8, help="ranks per node (2level)")
    p.add_argument("--alpha-intra", type=float, default=3.0, help="intra-node alpha, us")
    p.add_argument("--beta-intra", type=float, default=200.0, help="intra-node beta, GB/s")
    a = p.parse_args()

    S, alpha, beta = parse_size(a.size), a.alpha * 1e-6, a.beta * 1e9
    if a.algo == "ring":
        lat, bw = ring(a.coll, a.n, S, alpha, beta)
    elif a.algo == "tree":
        lat, bw = tree(a.coll, a.n, S, alpha, beta)
    else:
        lat, bw = two_level(a.coll, a.n, S, alpha, beta, a.ppn,
                            a.alpha_intra * 1e-6, a.beta_intra * 1e9)
    T = lat + bw
    algbw = S / T / 1e9
    busbw = algbw * busbw_factor(a.coll, a.n)
    print(f"{a.coll:14s} {a.algo:6s} n={a.n:<4d} S={a.size:>5s}  T={T*1e6:11.1f} us  "
          f"lat={100*lat/T:5.1f}%  algbw={algbw:7.2f}  busbw={busbw:7.2f} GB/s")

if __name__ == "__main__":
    main()
```

几点实现说明。`parse_size` 里 `1G` 是 $$2^{30}$$ 字节，与 nccl-tests `-b`/`-e` 参数的口径一致，所以下面表里 1G 的时间是 75.3 ms 而不是正文用 $$10^9$$ 算的 70 ms；β 的单位是 $$10^9$$ B/s，与 nccl-tests 和网卡的口径一致。`ring` 函数对 broadcast / reduce 用的是流水化链（$$n-1$$ 步、每 rank $$S$$ 字节），与 NCCL 的 ring broadcast 一致。`tree` 只实现了 double binary tree 的 all_reduce；`two_level` 假设每张卡有自己的节点间链路。

一组示例运行（默认 α = 10 µs、β = 25 GB/s；`2level` 默认节点内 α = 3 µs、β = 200 GB/s）：

```text
$ for s in 64K 2M 25M 1G; do ./cost_model.py --n 8 --size $s; done
all_reduce     ring   n=8    S=  64K  T=      144.6 us  lat= 96.8%  algbw=   0.45  busbw=   0.79 GB/s
all_reduce     ring   n=8    S=   2M  T=      286.8 us  lat= 48.8%  algbw=   7.31  busbw=  12.80 GB/s
all_reduce     ring   n=8    S=  25M  T=     1975.0 us  lat=  7.1%  algbw=  13.27  busbw=  23.23 GB/s
all_reduce     ring   n=8    S=   1G  T=    75301.9 us  lat=  0.2%  algbw=  14.26  busbw=  24.95 GB/s

$ ./cost_model.py --n 8 --size 64K --algo tree
all_reduce     tree   n=8    S=  64K  T=       65.2 us  lat= 92.0%  algbw=   1.00  busbw=   1.76 GB/s
$ ./cost_model.py --n 1024 --size 64K;  ./cost_model.py --n 1024 --size 64K --algo tree
all_reduce     ring   n=1024 S=  64K  T=    20465.2 us  lat=100.0%  algbw=   0.00  busbw=   0.01 GB/s
all_reduce     tree   n=1024 S=  64K  T=      205.2 us  lat= 97.4%  algbw=   0.32  busbw=   0.64 GB/s

$ ./cost_model.py --n 32 --size 1G;  ./cost_model.py --n 32 --size 1G --algo 2level
all_reduce     ring   n=32   S=   1G  T=    83835.0 us  lat=  0.7%  algbw=  12.81  busbw=  24.82 GB/s
all_reduce     2level n=32   S=   1G  T=    17550.3 us  lat=  0.6%  algbw=  61.18  busbw= 118.54 GB/s

$ ./cost_model.py --n 8 --size 128K --alpha 3 --beta 200          # NVLink 上的 decode TP all_reduce
all_reduce     ring   n=8    S= 128K  T=       43.1 us  lat= 97.3%  algbw=   3.04  busbw=   5.32 GB/s
$ ./cost_model.py --n 8 --size 1G --coll all_gather
all_gather     ring   n=8    S=   1G  T=    37651.0 us  lat=  0.2%  algbw=  28.52  busbw=  24.95 GB/s
```

每一行都能在正文里找到对应：第一组是第五章的核心问题与第九章的谱（2 MB 处延迟占比 48.8%，正是拐点）；第二组是第六章 Ring vs Tree 表里的 1024 卡；第三组是第八章两级算法的 4.7 倍与 busbw 超过链路的现象；最后两行分别是第七篇要处理的 decode 场景，以及 all_gather 的 busbw 与 all_reduce 一样落在链路带宽上、algbw 却不同。

后面几篇对这个脚本的使用方式是：第二篇用 `nvbandwidth` 与 `ib_write_bw` 测出你机器上真实的 β，第六篇用 nccl-tests 最小消息的时间反推 α，把默认值换掉；然后每一条实测曲线都与它输出的理论值比，差距就是那一篇要解释的东西。

模型里的 α 和 β 目前都是量级估计。下一篇给它们填上真实的数字：一台 8 卡服务器内部有哪些链路、每条多快、GPU 到 GPU 和 GPU 到网卡的路径怎么选，以及为什么两张看起来一样的卡之间的带宽可以差一个量级。

> **`nvidia-smi topo -m` 里 GPU0 到 GPU1 是 `NV12`、到 NIC0 是 `PIX`、到 NIC4 是 `SYS`。这三个词各自意味着什么带宽和什么路径？为什么 NCCL 会为 GPU0 选 NIC0 而不是 NIC4？**


## 下一篇

[硬件互联：PCIe、NVLink、NVSwitch 与网络拓扑](/hardware-interconnect-pcie-nvlink-and-topology.html)
