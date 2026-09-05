---
layout: post
title: 通信与互联：从 NCCL 到 RDMA（总纲）
subtitle: Communication and Interconnect for AI-Infra, from NCCL to RDMA
tags: [NCCL, RDMA, GPU, AI, AI-Infra]
catalog: true
---


## 内容简介

《通信与互联：从 NCCL 到 RDMA》是一组共七篇的系列文章，面向已经跑过多卡训练或多卡推理、准备向下进入通信层的工程师，系统讲解 GPU 之间的数据是怎么流动的：经过哪些硬件链路、由哪些软件层驱动、每一层的代价是什么、出问题时到哪里去看。

它回答的问题是：

> **一次 all_reduce 从调用到完成，数据在 PCIe、NVLink、InfiniBand 上是怎么流动的？为什么有时候是带宽的问题，有时候是延迟的问题？**

站在框架的层面看，通信是一行代码：`dist.all_reduce(t)`。它返回得很快，然后在某个时刻"完成"。Profiler 里它是一段 `ncclDevKernel_AllReduce_*` 的时间条；nccl-tests 给它一个 `busbw` 数字；训练日志偶尔告诉你它 timeout 了。这些信息都是真的，但都停在一个没有展开的前提上——**这行代码触发了什么**。

这个系列把这条路径展开。一次跨节点的 all_reduce，数据至少经过下面这些层，每一层都有自己的带宽、延迟和失败方式：

```text
dist.all_reduce(t)                    Python，进入 ProcessGroupNCCL
  → ncclAllReduce(...)                NCCL host 侧：选算法、选协议、切 channel、入队
    → ncclDevKernel_AllReduce_*       NCCL 设备侧：每个 channel 一个 block，按 ring/tree 收发
      → NVLink / NVSwitch              节点内：GPU 直接读写对端显存
      → PCIe → NIC → InfiniBand        节点间：proxy 线程驱动 RDMA，GPUDirect 绕开主机内存
        → 对端 NIC → PCIe → GPU        再走一遍反向的路
```

带宽和延迟是这条路径上两种不同的账。一个 512 MB 的梯度 all_reduce，时间几乎全部花在数据搬运上，看的是链路带宽和算法的带宽效率；一个 decode 阶段几十 KB 的张量并行 all_reduce，时间几乎全部花在同步、握手和 kernel 启动上，带宽多少无关紧要。同一个 API，同一套 NCCL，两种完全不同的瓶颈——分不清这两种账，就分不清什么时候该换算法、什么时候该换硬件、什么时候什么都不用换。

系列贯穿的方法是：**先用代价模型算出理论时间，再用 nccl-tests 和 profiler 测出实际时间，再用拓扑、算法和协议解释差距，再动手缩小差距**。


## 为什么写这个系列？

### 单卡之外的一切都建立在通信之上

模型一旦放不进一张卡，通信就成了主线而不是配角。数据并行需要 all_reduce 或 reduce_scatter + all_gather 同步梯度；张量并行每一层前向都要一次 all_reduce；流水线并行靠 send/recv 传激活；MoE 靠 all_to_all 分发 token；FSDP 和 ZeRO 每一步都要把分片的参数 all_gather 回来。推理侧同样：张量并行的每一层都有 all_reduce，PD 分离要把 KV cache 从 prefill 节点搬到 decode 节点。

这些通信的**模式**各不相同——消息大小从几十 KB 到几 GB，参与者从 2 张卡到几千张卡，有的在关键路径上、有的可以和计算重叠——但它们的**底座**是同一套：NCCL、NVLink、PCIe、InfiniBand 或 RoCE、RDMA。这一层自成体系，值得独立于训练和推理来学。

### "通信占了 40%"之后呢？

一份 profiler 报告说通信占了迭代时间的 40%，这个数字本身不说明任何事情。可能的原因至少有：

- 消息太小，每次通信的固定开销（延迟）占主导，应该合并（bucket）或换协议；
- 消息够大，但算法带宽效率低，比如在 NVSwitch 机器上跑了不该跑的 Ring；
- 拓扑没有被正确识别，NCCL 走了 PCIe 而不是 NVLink，或者走了主机内存而不是 GPUDirect；
- 网卡与 GPU 不在同一个 PCIe switch 或 NUMA 节点下，跨了 CPU 的 root complex；
- 通信本身很快，但没有和计算重叠，或者重叠被一次不必要的同步打断；
- 某一张卡慢（straggler），所有人都在等它。

每一种原因对应完全不同的处理方式，而区分它们需要知道数据实际走了哪条路、每一段的理论上限是多少。这套判断能力是本系列要建立的。

### NCCL 几乎是黑盒，但它不该是

NCCL 是所有 NVIDIA GPU 上分布式训练和推理的事实标准，它的行为由几十个环境变量控制，源码约几万行 C++ 和 CUDA。大多数人对它的了解止于 `NCCL_DEBUG=INFO` 和几条"据说有用"的环境变量。但它的核心机制并不神秘：拓扑探测、图搜索、算法与协议的选择、channel 的切分、proxy 线程与网络插件——每一个都能在源码里找到对应的文件，都能用日志和 nccl-tests 验证。

读懂 NCCL 的价值不只是调参。它是理解一切上层通信设计的参照：PyTorch 的 ProcessGroupNCCL 为什么要有 watchdog 线程，vLLM 为什么要绕开 NCCL 自己写 all-reduce，为什么 NVLink 机器上的 all_reduce 几乎不受消息大小影响而 IB 上不是——答案都在 NCCL 的设计取舍里。

### 现有材料的断层

- **NCCL 官方文档**覆盖了 API 和环境变量，但不讲机制，也不讲为什么；
- **RDMA / InfiniBand 的教材**面向 HPC 与存储网络，不讲 GPU，也不讲 NCCL 如何使用 verbs；
- **PyTorch 分布式文档**讲怎么用 DDP 和 FSDP，把通信当作已经工作的黑盒；
- **论文**（ring all-reduce、double binary tree、NVLink SHARP）给出了算法和硬件的原理，但没有人把它们串成从 API 到链路的一条线；
- **博客与调优帖子**分享环境变量组合，往往缺少"为什么这样设置"以及"什么时候不该这样设置"。

本系列要填的是从"会调 `dist.all_reduce`"到"能说出这次调用的数据走了哪里、为什么是这个耗时、出了问题去哪里看"之间的那段路。


## 适合哪些读者？

### 跑过多卡训练、想弄清通信在做什么的工程师

你用过 DDP 或 FSDP，配过 `NCCL_SOCKET_IFNAME`，见过 `NCCL WARN` 和 `Watchdog caught collective operation timeout`。你想知道的是：**那些环境变量各自控制路径上的哪一段，那个 timeout 背后真正发生了什么**。

### 做推理系统、需要处理张量并行和 PD 分离的工程师

张量并行的 all_reduce 在 decode 阶段是延迟敏感的小消息，PD 分离的 KV 传输是带宽敏感的大块搬运，两者对通信层的要求几乎相反。本系列第七篇专门处理推理侧，前六篇是理解它的基础。

### 负责 GPU 集群网络与调度的平台工程师

你需要决定网卡怎么配、GPU 与 NIC 的亲和怎么绑、RoCE 网络要不要开 PFC 和 ECN、一个任务的 8 张卡放在哪台机器上有区别吗。这些决定的依据全部在本系列的第二、三、六篇里。

### 准备读或改 NCCL、PyTorch c10d、vLLM 通信层源码的开发者

NCCL 的 `src/`、PyTorch 的 `torch/csrc/distributed/c10d/`、vLLM 的 `vllm/distributed/` 是本系列的源码阅读对象。读完后你应该能读懂它们的主路径，并知道一个新的传输后端或一个新的 all-reduce 实现应该接在哪里。


## 系列的整体主线

七篇文章按"从抽象代价到物理链路，再回到软件栈"的顺序推进：

```text
第一篇：集合通信原语与代价模型 —— 建立分析框架：α-β 模型与 ring all-reduce 的推导
        ↓
第二篇：硬件互联 —— PCIe、NVLink、NVSwitch、IB/RoCE 的拓扑与带宽，NUMA 与亲和
        ↓
第三篇：RDMA 与 GPUDirect —— 绕过 CPU 和主机内存的数据通路
        ↓
第四篇：NCCL 架构 —— 拓扑探测、channel、算法与协议，一次 ncclAllReduce 的完整路径
        ↓
第五篇：PyTorch 的通信栈 —— ProcessGroupNCCL、stream/event 语义、异步与重叠
        ↓
第六篇：nccl-tests、调优与排障 —— 带宽曲线的读法、环境变量、hang 与 timeout
        ↓
第七篇：推理侧的通信 —— custom all-reduce、KV 传输（NIXL / UCX / Mooncake）
```

三条交织的线索：

```text
代价线：α-β 模型 → 链路带宽与延迟 → 算法带宽效率 → 协议开销 → 实测曲线 → 小消息与大消息的两种账
路径线：NVLink / PCIe / IB → RDMA verbs → NCCL transport → ProcessGroupNCCL → vLLM 通信后端
排障线：nvidia-smi topo → ibstat / ib_write_bw → NCCL_DEBUG → Flight Recorder → 决策树
```

前三篇建立"硬件能做到什么"的上限，第四篇讲 NCCL 如何逼近这个上限，第五篇讲框架如何使用 NCCL 而不浪费它，第六篇把前五篇变成可操作的测量与排障方法，第七篇把这套方法用到推理的两个特殊场景上。

每一篇都有同样的四段结构：

```text
算一算      用代价模型给出这一层的理论上限：这段路径最快多久、最多多少字节每秒
看一看      读源码或日志，弄清这一层实际怎么做决定：拓扑文件、NCCL 日志、c10d 的 C++
测一测      用工具测出实际数字：nvbandwidth、ib_write_bw、nccl-tests、profiler
比一比      解释理论与实测的差距，给出这一层的排障检查项
```

"两种账"的区分会贯穿每一篇。带宽的账看链路速率、算法的带宽效率、协议开销；延迟的账看步数、握手次数、kernel 启动、proxy 线程的响应。同一个问题在两本账上的答案往往相反——比如更多 channel 提高带宽却增加小消息的延迟，Tree 减少延迟却在某些拓扑上损失带宽——本系列会在每一处选择上把两本账都算一遍。


## 章节结构与分章导读

### 1. 集合通信原语与代价模型：α-β 模型与 ring all-reduce

第一篇不碰任何硬件和 NCCL，只建立整个系列的分析框架：**给定 n 个参与者、S 字节的数据、一条带宽为 β、延迟为 α 的链路，一次集合通信理论上最快需要多久？**

这一篇会覆盖：

- 集合通信原语的定义与语义：broadcast、reduce、all_reduce、all_gather、reduce_scatter、all_to_all、scatter/gather、send/recv；它们之间的组合关系（all_reduce = reduce_scatter + all_gather）；
- 训练和推理各自需要哪些原语：数据并行的梯度 all_reduce、FSDP 的 all_gather / reduce_scatter、张量并行的 all_reduce、流水线的 send/recv、MoE 的 all_to_all、PD 分离的点对点 KV 传输；
- α-β 模型（Hockney 模型）：一条消息的传输时间 $$T = \alpha + S/\beta$$，α 是延迟（固定开销），β 是带宽；两项在什么消息大小下相等，这个拐点决定了"延迟主导"还是"带宽主导"；
- ring all-reduce 的推导：reduce_scatter 与 all_gather 各 $$n-1$$ 步，每步每个 rank 收发 $$S/n$$ 字节，总时间

$$
T_{\text{ring}} = 2(n-1)\,\alpha + \frac{2(n-1)}{n}\cdot\frac{S}{\beta}
$$

  带宽项随 n 增大趋于 $$2S/\beta$$，与参与者数量无关，这是 ring 在大消息上最优的原因；延迟项随 n 线性增长，这是 ring 在小消息、大规模上失败的原因；每个 rank 实际收发的字节数是 $$\frac{2(n-1)}{n} S$$，这个系数后面会反复出现；
- tree all-reduce：延迟 $$O(\log n)$$，朴素二叉树的带宽只有一半，double binary tree 如何用两棵互补的树把带宽补回来；
- algbw 与 busbw：nccl-tests 报告的两个带宽，前者是 $$S/T$$，后者对 all_reduce 乘上 $$2(n-1)/n$$ 校正为"链路实际承载的流量"，为什么只有 busbw 才能和硬件带宽直接比较；
- 分层与多级：节点内快、节点间慢时，两级算法的代价怎么算；
- 消息大小的谱：从 decode 阶段几十 KB 的 TP all_reduce，到几百 MB 的梯度 bucket，到几 GB 的 KV cache 传输，各落在 α-β 模型的哪一段。

核心问题是：

> **8 张卡做一次 1 GB 的 all_reduce，链路单向 25 GB/s，ring 算法理论上要多久？改成 64 KB 呢？这两个数字为什么分别对带宽和延迟敏感？**

实践：写一个几十行的代价模型脚本，输入 n、S、α、β 和算法，输出理论时间与 busbw；后面每一篇的实测数字都要和它比。

### 2. 硬件互联：PCIe、NVLink、NVSwitch 与网络拓扑

第二篇给第一篇的 α 和 β 填上真实数字。一台 8 卡服务器内部和一个多机集群之间，有哪些链路、各自多快、GPU 到 GPU 的路径怎么选，这些决定了通信的物理上限。

这一篇会覆盖：

- PCIe：lane、代际与带宽——x16 的理论单向带宽 PCIe 3.0 约 16 GB/s、4.0 约 32 GB/s、5.0 约 64 GB/s（实测通常 80–90%）；root complex、PCIe switch、P2P 事务；为什么跨 root complex 的 P2P 可能慢甚至不可用；
- NVLink 与 NVSwitch：每代 NVLink 的链路数与每 GPU 总带宽——A100 第三代 12 链路合计 600 GB/s、H100 第四代 18 链路合计 900 GB/s、Blackwell 第五代 1.8 TB/s（均为双向合计，单向减半）；NVSwitch 让节点内任意两卡全带宽互通；NVLink 上的 all_reduce 为什么几乎不受消息大小影响；
- 网卡与网络：InfiniBand 的代际与端口速率——HDR 200 Gb/s（约 25 GB/s）、NDR 400 Gb/s（约 50 GB/s）；RoCE v2 在以太网上承载 RDMA，需要 PFC / ECN 保证无损；一台 8 卡 H100 服务器通常配 8 张 400 Gb/s 网卡，每卡一张，为什么是这个比例；
- 拓扑的读法：`nvidia-smi topo -m` 输出的矩阵与 `NV#` / `PIX` / `PXB` / `PHB` / `NODE` / `SYS` 六个等级各自的含义；`lspci -tv` 看 PCIe 树；`nvidia-smi topo -mp` 看 GPU 与 NIC 的亲和；
- NUMA 与亲和性：CPU socket、内存节点、PCIe 设备的归属；进程绑核（`numactl`）为什么会影响通信性能；NCCL 如何读取并使用亲和信息，`NCCL_IGNORE_CPU_AFFINITY` 在什么情况下需要；
- 主机内存在路径上的位置：pinned memory、staging buffer，什么时候数据必须经过主机内存，什么时候可以绕过；
- 集群级拓扑：fat-tree、rail-optimized 设计，为什么同一个 rail 上的 GPU 通信更快，这对任务调度意味着什么；
- 测带宽的工具：`nvbandwidth`、`p2pBandwidthLatencyTest`、`ib_write_bw` / `ib_read_bw`（perftest），各测哪一段。

核心问题是：

> **`nvidia-smi topo -m` 里 GPU0 到 GPU1 是 `NV12`、到 NIC0 是 `PIX`、到 NIC4 是 `SYS`。这三个词各自意味着什么带宽和什么路径？为什么 NCCL 会为 GPU0 选 NIC0 而不是 NIC4？**

实践：写一个脚本解析 `nvidia-smi topo -m`、`lspci` 和 `/sys/bus/pci/devices/*/numa_node`，画出本机的 GPU–PCIe switch–NIC–NUMA 拓扑图，并用 `nvbandwidth` 与 `ib_write_bw` 测出每段链路的实际带宽，填入第一篇的代价模型。

### 3. RDMA 与 GPUDirect：绕过 CPU 和主机内存的数据通路

第三篇讲节点间通信的软件基础。TCP/IP 的路径上有内核协议栈、多次拷贝和 CPU 参与，跑不满 400 Gb/s 的网卡；RDMA 让网卡直接读写应用内存，GPUDirect RDMA 让它直接读写显存。这一篇讲这两件事怎么工作、怎么用、怎么坏。

这一篇会覆盖：

- 为什么需要 RDMA：内核旁路（kernel bypass）、零拷贝、CPU 卸载；TCP 路径与 RDMA 路径的对比；
- verbs 编程模型：device、Protection Domain、Memory Region 注册（`ibv_reg_mr`）、Queue Pair（RC / UC / UD）、Completion Queue、Work Request 与 Work Completion；
- 两类操作：单边的 RDMA WRITE / READ（对端 CPU 不参与）与双边的 SEND / RECV；NCCL 为什么主要用 RDMA WRITE；
- 连接建立：QP 状态机、需要交换的信息（QPN、LID / GID、rkey、地址），带外交换用 TCP 还是 `rdma_cm`；
- InfiniBand 与 RoCE v2 的差别：链路层、寻址（LID vs GID）、拥塞控制（信用机制 vs PFC + ECN / DCQCN）；RoCE 上 `NCCL_IB_GID_INDEX` 为什么经常需要手工指定；
- 内存注册的代价：pin 页、建立地址翻译，为什么注册慢、注册多少、注册缓存（MR cache）为什么存在；
- GPUDirect RDMA：让网卡 DMA 直接访问显存，需要 `nvidia-peermem` 内核模块（或 DMA-BUF 路径）；PCIe 拓扑对它的限制——GPU 和 NIC 在同一 PCIe switch 下最优，跨 root complex 可能不可用或很慢；
- GDRCopy：用 CPU 直接读写显存映射的小数据，NCCL 用它降低小消息延迟；
- GPUDirect 家族的其余成员：GPUDirect P2P（节点内 GPU 互访）、GPUDirect Storage（NVMe 直读显存）；
- 诊断工具：`ibstat`、`ibv_devinfo`、`show_gids`、`rdma link`、`ib_write_bw` 加 `--use_cuda` 测 GPUDirect 路径；`lsmod | grep peermem`。

核心问题是：

> **一段显存里的数据要发到另一台机器的显存，走 TCP、走 RDMA 不开 GPUDirect、走 RDMA 开 GPUDirect，分别经过几次拷贝、经过哪些 PCIe 链路？各自的带宽上限是多少？**

实践：用 `libibverbs` 写一个最小的 RDMA WRITE 程序（约两三百行 C），两台机器之间传一块 buffer；然后把 buffer 换成 `cudaMalloc` 的显存，验证 GPUDirect RDMA 路径；用 `ib_write_bw` 对照带宽，并观察 GPU 与 NIC 亲和不同时的差距。

### 4. NCCL 架构：拓扑探测、channel、算法与协议

第四篇是全系列的中心。它沿着一次 `ncclAllReduce` 从初始化到 kernel 结束的完整路径，讲 NCCL 如何把前三篇的硬件能力组织成一次集合通信：

```text
ncclCommInitRank
  bootstrap        TCP 交换地址，所有 rank 互相认识                    src/bootstrap.cc
  topo detect      读 /sys 与 NVML，建出 GPU/NIC/PCIe/CPU 的树         src/graph/xml.cc · topo.cc
  paths            算出任意两设备间的路径类型与带宽                     src/graph/paths.cc
  search           在拓扑上搜出若干条 ring 与 tree                     src/graph/search.cc · rings.cc · trees.cc
  connect          为每个 channel 建立 transport（P2P/SHM/NET/...）    src/transport/*.cc · src/graph/connect.cc
  tuning           为每种算法×协议×消息大小估算时间，填调优表           src/graph/tuning.cc
ncclAllReduce
  enqueue          查调优表选算法与协议，切 channel，准备 kernel 参数    src/enqueue.cc
  launch           一个 kernel，nChannels 个 block，各跑一条 ring/tree  src/device/all_reduce.h · primitives.h
  proxy            CPU 线程替 GPU 提交与轮询网络请求（跨机时）          src/proxy.cc
```

这一篇会覆盖：

- 初始化：`ncclGetUniqueId` 与 bootstrap（TCP，`NCCL_SOCKET_IFNAME` 控制走哪张网卡）、`ncclCommInitRank`、每个 rank 如何得知其他 rank 的位置；
- 拓扑探测：从 `/sys` 和 NVML 读出 GPU、NIC、PCIe switch、CPU 的树（`src/graph/xml.cc`、`topo.cc`），计算两两之间的路径与带宽（`paths.cc`）；`NCCL_TOPO_DUMP_FILE` 把它导出来看；
- 图搜索：在拓扑上搜索 ring 和 tree（`search.cc`、`rings.cc`、`trees.cc`），目标是最大化带宽；`NCCL_GRAPH_DUMP_FILE`；
- transport 选择：P2P（NVLink / PCIe 直接访问对端显存）、SHM（经主机共享内存）、NET（IB / Socket），以及 CollNet 和 NVLS；`NCCL_P2P_LEVEL`、`NCCL_NET_GDR_LEVEL` 用与 `nvidia-smi topo` 相同的等级词（`LOC` / `NVL` / `PIX` / `PXB` / `PHB` / `SYS`）控制某种路径最多能跨多远；
- channel：一个 communicator 有多个 channel，每个 channel 是一条独立的 ring 或 tree、对应 kernel 的一个 block、拥有自己的 buffer 和连接；channel 数决定并行度，`NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS`；
- 算法：Ring、Tree、CollNet（借助 IB SHARP 在网络中归约）、NVLS（借助 NVSwitch 归约，Hopper 起）、PAT（大规模 all_gather / reduce_scatter）；`NCCL_ALGO` 覆盖自动选择；
- 协议：Simple（数据与 flag 分离、需要内存屏障、带宽最高、延迟最高）、LL（8 字节数据搭 8 字节 flag、一次 16 字节原子写、无需屏障、带宽效率 50%、延迟最低）、LL128（128 字节里 120 字节数据、效率约 94%、依赖 NVLink 的写顺序保证）；`NCCL_PROTO`；
- 调优表：NCCL 按消息大小、rank 数、拓扑估算每种算法与协议组合的时间，选最快的一种（`src/graph/tuning.cc`，master 分支已迁到 `src/tuning/`）；tuner 插件接口；
- enqueue 与 kernel：`src/enqueue.cc` 把集合操作切成任务、分配到 channel、启动一个 kernel；设备侧 `src/device/all_reduce.h` 与 `primitives.h` / `prims_simple.h` / `prims_ll.h` / `prims_ll128.h` 是每个 block 执行的收发原语；
- proxy 线程：GPU kernel 不能直接驱动网卡，`src/proxy.cc` 里的 CPU 线程替它提交 RDMA 请求、轮询完成；这条 CPU–GPU 协作路径是很多性能问题和 hang 的源头；
- group 语义：`ncclGroupStart` / `ncclGroupEnd` 把多个操作合成一次启动，send/recv 为什么必须成对放在 group 里。

核心问题是：

> **同一次 8 卡 all_reduce，NCCL 在 NVSwitch 机器上选了 NVLS + Simple，在 PCIe 机器上选了 Ring + LL128，跨 32 台机器时选了 Tree。它是根据什么做出这三个不同决定的？强行用 `NCCL_ALGO=Ring` 会付出什么？**

实践：用 `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING` 跑一次 all_reduce，逐行解读日志里的拓扑、ring/tree 结构、channel 数与算法协议选择，并与第一、二篇的理论预测对照；导出 `NCCL_TOPO_DUMP_FILE` 和 `NCCL_GRAPH_DUMP_FILE`，改写拓扑文件观察 NCCL 的决策如何变化。

### 5. PyTorch 的通信栈：ProcessGroupNCCL、stream 语义与计算通信重叠

第五篇向上一层，讲 PyTorch 如何使用 NCCL。这一篇只讲通信层的视角：一次 `dist.all_reduce` 在 c10d 里经过了什么、NCCL kernel 在哪条 stream 上、"异步"到底意味着什么、重叠为什么会失效。并行策略本身（DDP 的 bucket 划分、FSDP 的分片规则）不在范围内。

这一篇会覆盖：

- c10d 的分层：Python 的 `torch.distributed.distributed_c10d` → C++ 的 `ProcessGroup` / `Backend` → `ProcessGroupNCCL`；Store（TCPStore）承担 rendezvous 与 `ncclUniqueId` 的分发；
- ProcessGroupNCCL 的对象模型：每个 device 一个 `ncclComm`，每个 collective 返回一个 `WorkNCCL`，`Work` 上记录了起止 CUDA event；communicator 的懒创建与 `ncclCommInitRankConfig` 的非阻塞初始化；
- stream 语义：NCCL 操作在 ProcessGroupNCCL 自己的内部 stream 上执行，与当前计算 stream 通过 event 建立依赖；`work.wait()` 是 stream 级等待，不阻塞 CPU；什么时候才真的阻塞 CPU；
- 输入 tensor 的生命周期：为什么需要 `recordStream` 或 `TORCH_NCCL_AVOID_RECORD_STREAMS` 的替代方案，Caching Allocator 与跨 stream 使用的冲突；
- `async_op=True` 与重叠：通信 kernel 与计算 kernel 在不同 stream 上并发的条件——SM 资源、`NCCL_NTHREADS`、`TORCH_NCCL_HIGH_PRIORITY`；重叠被打断的常见原因（隐式同步、`.item()`、显存分配触发的 `cudaFree`）；
- 合并（coalescing）：`all_reduce_coalesced`、`_coalescing_manager`、DDP 的 bucket 为什么能把延迟主导的小消息变成带宽主导的大消息；
- 函数式集合通信：`torch.distributed._functional_collectives` 让通信成为可编译的算子，`wait_tensor` 的语义，`torch.compile` 如何对通信做重排；
- 错误处理与超时：`TORCH_NCCL_ASYNC_ERROR_HANDLING`、`TORCH_NCCL_BLOCKING_WAIT`、watchdog 线程如何检测 timeout、检测到之后做什么（abort communicator、抛异常、`ncclCommAbort`）；heartbeat monitor（`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`）监视 watchdog 自己；
- 与 NCCL 之外后端的对照：Gloo（CPU）、UCC、以及 vLLM 用 ctypes 直接调用 NCCL 的 PyNccl 为什么绕开 ProcessGroupNCCL；
- 对称内存（`torch/csrc/distributed/c10d/symm_mem/`）：让 GPU 直接读写对端显存的低延迟通信原语，它和 NCCL 的关系。

核心问题是：

> **`work = dist.all_reduce(t, async_op=True)` 返回时，通信开始了吗？`work.wait()` 返回时，通信完成了吗？在此期间修改 `t` 会发生什么？**

实践：写一个 micro-benchmark，在两条 stream 上让一个 GEMM 与一个 all_reduce 重叠，用 `torch.profiler` 的 trace 验证重叠是否真的发生；逐一引入 `.item()`、显存碎片、不当的 `wait()` 位置，观察重叠如何消失。

### 6. nccl-tests、调优与排障：从带宽曲线到 hang

第六篇把前五篇变成一套可操作的方法。它分两半：**性能**——怎么测、怎么读、怎么调；**正确性**——hang、timeout、死锁、结果错误怎么排。

这一篇会覆盖：

- nccl-tests 的用法：`all_reduce_perf` 等各原语的测试程序；`-b` / `-e` / `-f` 扫描消息大小，`-g` 每进程 GPU 数，`-n` / `-w` 迭代与预热，`-c` 校验结果；`mpirun` 与 `torchrun` 两种启动方式；
- 带宽曲线的读法：以消息大小为横轴、busbw 为纵轴，小消息端的平台是延迟主导（看 α），大消息端的平台是带宽主导（看 β），拐点位置与第一篇的 α-β 模型对照；曲线"该长什么样"——NVLink 节点内、单机 PCIe、跨机 IB 各一条参考线；
- 常见异常形状：大消息端上不去（链路没走对、GDR 没开、channel 太少）、小消息端太高（协议选错、跨了 NUMA）、中段有凹陷（算法切换点选得不好）、多机比单机慢很多（网卡亲和、NCCL_IB_HCA 没限定、走了 Socket）；
- 调优参数与它们各自作用的层：`NCCL_ALGO` / `NCCL_PROTO`（算法与协议）、`NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` / `NCCL_NTHREADS`（并行度）、`NCCL_BUFFSIZE`（channel buffer）、`NCCL_P2P_LEVEL` / `NCCL_NET_GDR_LEVEL` / `NCCL_P2P_DISABLE` / `NCCL_SHM_DISABLE`（路径选择）、`NCCL_IB_HCA` / `NCCL_IB_GID_INDEX` / `NCCL_IB_TC` / `NCCL_IB_QPS_PER_CONNECTION` / `NCCL_IB_SPLIT_DATA_ON_QPS`（网络）、`NCCL_SOCKET_IFNAME` / `NCCL_SOCKET_NTHREADS` / `NCCL_NSOCKS_PERTHREAD`（Socket 与 bootstrap）、`NCCL_CROSS_NIC`、`NCCL_NVLS_ENABLE`、`NCCL_COLLNET_ENABLE`；哪些应该动、哪些几乎永远不该动；
- 日志：`NCCL_DEBUG=WARN` 是生产默认、`INFO` 看决策、`TRACE` 看每次调用；`NCCL_DEBUG_SUBSYS` 按子系统过滤；`NCCL_DEBUG_FILE` 按 rank 分文件；
- hang 的分类：集合通信参数不一致（某个 rank 的 tensor 大小或 dtype 不同）、调用顺序不一致（某个 rank 多发或少发了一次集合通信）、send/recv 没有配对、两个 communicator 交叉等待、一个 rank 崩溃而其他 rank 在等、网络真的断了；
- 排查工具：`py-spy dump` 看每个 rank 的 Python 栈、`gdb -p` 看 C++ 栈、`cuda-gdb` 看 kernel 是否在自旋；PyTorch 的 Flight Recorder（`TORCH_NCCL_TRACE_BUFFER_SIZE`、`TORCH_NCCL_DUMP_ON_TIMEOUT`、`torch/distributed/flight_recorder/fr_trace.py` 分析工具）把所有 rank 最近的集合通信记录汇总对齐，直接指出哪个 rank 在哪一次操作上掉了队；`TORCH_NCCL_DESYNC_DEBUG`；
- timeout 的语义：`init_process_group(timeout=...)` 到底约束什么、为什么 checkpoint 保存或数据加载卡住会表现为 NCCL timeout；
- 正确性问题：结果不一致（浮点归约顺序、`NCCL_ALGO` 不同导致数值不同）、NaN 的来源与 `TORCH_NCCL_NAN_CHECK`、多 communicator 与多 stream 下的数据竞争；
- 一棵决策树：从现象（慢 / hang / 错）到检查项到处理方式，大致是这个形状：

```text
慢    大消息 busbw 低      → 路径：topo -m 等级 · NCCL_DEBUG 里的 transport · GDR 是否生效 · channel 数
      小消息延迟高         → 协议与算法 · 跨 NUMA · proxy 线程被抢占 · 框架侧没有合并
      多机远差于单机       → 网卡亲和 · NCCL_IB_HCA · 是否回落到 Socket · RoCE 的 PFC/ECN
hang  所有 rank 停在同一处 → 网络或某个 rank 崩溃：看 dmesg · ibstat · 进程是否还在
      各 rank 停在不同处   → 调用不一致：Flight Recorder 对齐序号 · py-spy 看栈
      只有部分 rank 停     → send/recv 不配对 · 多 communicator 交叉等待
错    结果不稳定           → 浮点归约顺序 · 算法差异 · 数据竞争（多 stream 未同步）
      NaN                  → TORCH_NCCL_NAN_CHECK 定位首次出现的 rank 与操作
```

核心问题是：

> **一个 64 卡训练任务在第 3000 步 hang 住，所有 rank 的日志都停在 all_reduce。是谁的问题、是哪一次 all_reduce、为什么会等到 timeout 才暴露？**

实践：在本机与两机环境跑完整的 nccl-tests 扫描并画出带宽曲线，与代价模型的预测对齐；人为制造三种 hang（参数不一致、顺序不一致、send/recv 不配对），用 py-spy 和 Flight Recorder 分别定位。

### 7. 推理侧的通信：custom all-reduce 与 KV 传输

第七篇把前六篇的方法用到推理系统的两个特殊场景上。它们的通信模式与训练截然不同：张量并行的 all_reduce 是 decode 阶段每层一次的几十 KB 小消息，延迟就是一切；PD 分离的 KV 传输是点对点的大块搬运，要在不干扰计算的前提下跑满链路。

这一篇会覆盖：

- decode 阶段 TP all_reduce 的账：batch 小、hidden 维度固定，每层一次 all_reduce 只有几十到几百 KB，按第一篇的模型这是纯延迟主导；NCCL 在这个区间的固定开销（kernel 启动、proxy、协议握手）有多大；
- custom all-reduce：vLLM 的 `csrc/custom_all_reduce.cuh` 用 CUDA IPC（`cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`）让每张卡直接读写对端显存，one-shot（`cross_device_reduce_1stage`：每张卡读所有人的数据自己归约）与 two-shot（`cross_device_reduce_2stage`：先 reduce_scatter 再 all_gather）两种 kernel 各适合什么消息大小；用 flag 做跨卡同步而不用 NCCL；为什么只在节点内、NVLink 全互联、消息小于一个阈值时启用；
- 后端的选择链：`vllm/distributed/device_communicators/cuda_communicator.py` 里 all_reduce 的调度顺序——对称内存、quick reduce（ROCm）、FlashInfer、custom all-reduce、最后回落到 PyNccl；`vllm/distributed/parallel_state.py` 的 `GroupCoordinator` 如何管理 TP / PP / DP 组；
- PyNccl：`pynccl_wrapper.py` 用 ctypes 直接加载 `libnccl.so`（`VLLM_NCCL_SO_PATH`）调用 NCCL，为什么推理引擎不直接用 ProcessGroupNCCL——stream 控制、CUDA Graph 捕获、避免 watchdog；
- CUDA Graph 与通信：all_reduce 被捕获进 graph 需要满足什么条件，custom all-reduce 的 `register_graph_buffers` 在做什么；
- 对称内存与 NVLS 在推理上的应用：PyTorch 的 symmetric memory、NCCL 的 NVLS，与 custom all-reduce 各自的适用区间；
- PD 分离的 KV 传输：要传多少字节（每 token 每层 KV 的大小乘以序列长度乘以层数）、什么时候传（prefill 结束后一次性还是按层流水）、传给谁（TP 度不同的 prefill 与 decode 实例之间的映射）；
- KV 传输层的选择：NIXL（NVIDIA 的推理数据传输库，后端包括 UCX、GPUDirect Storage 等，vLLM 的 `NixlConnector` 在 `vllm/distributed/kv_transfer/kv_connector/v1/nixl/`）、UCX（统一通信框架，可在 RDMA / TCP / 共享内存 / CUDA IPC 之间自动选路）、Mooncake Transfer Engine（`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/`）；vLLM 的 `KVConnector` 抽象如何把 scheduler 侧的决策与 worker 侧的传输分开；
- 为什么 KV 传输不用 NCCL：点对点、动态目标、不需要归约、需要与计算完全解耦——单边 RDMA WRITE / READ 是更自然的原语；
- 推理侧的排障：TP all_reduce 延迟异常时先看走了哪个后端；KV 传输慢时先看是否走了 GPUDirect、GPU 与 NIC 亲和是否正确、注册内存是否命中缓存。

核心问题是：

> **8 卡 TP 的 decode，每层一次 128 KB 的 all_reduce，NCCL 要 30 微秒，custom all-reduce 要 10 微秒。这 20 微秒省在哪里？为什么这个方法不能用在训练的梯度同步上？**

实践：在 vLLM 上用同一模型分别开关 custom all-reduce，用 profiler 测每层 all_reduce 的延迟并与 nccl-tests 对照；搭一个两实例的 PD 分离部署，用 NIXL 传 KV，测传输带宽并与第三篇的 `ib_write_bw` 结果比较。本篇最后给出全系列总结。


## 贯穿全系列的实践线

系列的练手项目是一套**通信诊断工具集**，取名 comm-probe。选它是因为通信层的学习几乎全部依赖测量：不亲手测出每段链路的带宽、不亲手复现一次 hang，前面讲的一切都只是名词。它逐篇生长：

```text
第一篇    cost_model.py              α-β 模型 · ring/tree 预测 · algbw 与 busbw 换算
第二篇    topo_map.py                解析 nvidia-smi topo / lspci / NUMA · 画拓扑图 · 记录 nvbandwidth 与 ib_write_bw 实测
第三篇    rdma_write.c               libibverbs 最小 RDMA WRITE · 显存版（GPUDirect RDMA）· 与 ib_write_bw 对照
第四篇    nccl_log_reader.py         解析 NCCL_DEBUG=INFO 日志 · 提取 ring/tree/channel/算法协议决策 · 与预测比对
第五篇    overlap_bench.py           计算与通信重叠的 micro-benchmark · profiler trace 检查 · 重叠失效的复现集
第六篇    sweep.sh · hang_lab/       nccl-tests 扫描与画图 · 三种 hang 的复现与定位剧本 · 排障决策树
第七篇    tp_ar_bench.py · kv_xfer/  vLLM 各 all_reduce 后端延迟对照 · PD 分离 KV 传输带宽测量
```

到第七篇结束，读者手上有一套能在任何一台新机器上跑一遍的工具：先画出拓扑，再测每段链路，再跑 nccl-tests 与理论对照，再检查框架侧的重叠与后端选择。它不是一个通信库，但每一次"通信慢了"或"通信卡了"，都能用它在一小时内把问题定位到某一层。

与它平行的源码阅读线（NCCL 以 2.27–2.29 的源码树为准；master 分支近期把 `enqueue.cc` 移入 `src/enqueue/`、`graph/tuning.cc` 移入 `src/tuning/`、`transport/net_ib.cc` 拆为目录，随文标注）：

```text
第一篇    nccl-tests  src/all_reduce.cu · src/common.cu（algbw / busbw 的计算）
第二篇    NCCL  src/graph/xml.cc · src/graph/topo.cc · src/graph/paths.cc（拓扑的发现与路径带宽）
第三篇    NCCL  src/transport/net_ib.cc · src/misc/ibvwrap.cc · src/misc/gdrwrap.cc（verbs 与 GDRCopy 的封装）
第四篇    NCCL  src/init.cc · src/bootstrap.cc · src/graph/{search,rings,trees,connect,tuning}.cc
                src/transport/{p2p,shm,net,coll_net,nvls}.cc · src/enqueue.cc · src/proxy.cc · src/group.cc
                src/device/{all_reduce.h,primitives.h,prims_simple.h,prims_ll.h,prims_ll128.h}
第五篇    PyTorch  torch/csrc/distributed/c10d/ProcessGroupNCCL.{hpp,cpp} · NCCLUtils.cpp · Work.cpp · TCPStore.cpp
                   torch/distributed/distributed_c10d.py · torch/distributed/_functional_collectives.py
                   torch/csrc/distributed/c10d/symm_mem/
第六篇    PyTorch  torch/csrc/distributed/c10d/FlightRecorder.cpp · torch/distributed/flight_recorder/fr_trace.py
          NCCL  src/debug.cc · src/misc/param.cc（环境变量如何被读取）
第七篇    vLLM  csrc/custom_all_reduce.cuh · vllm/distributed/device_communicators/{custom_all_reduce,cuda_communicator,pynccl,pynccl_wrapper,symm_mem}.py
                vllm/distributed/parallel_state.py · vllm/distributed/kv_transfer/kv_connector/v1/{base.py,nixl/,mooncake/}
```


## 阅读路径建议

### 完整学习路径

```text
1 → 2 → 3 → 4 → 5 → 6 → 7
```

### 训练方向，主要关心多机性能与稳定性

```text
1 → 2 → 3 → 4 → 5 → 6
```

第七篇可以略过，但其中 custom all-reduce 一节对理解"NCCL 在小消息上的固定开销"有帮助。

### 推理方向，关心 TP 延迟与 PD 分离

```text
1 → 2 → 4 → 7 → 3 → 6
```

先建立代价模型与拓扑直觉，读 NCCL 架构后直接进第七篇；KV 传输涉及 RDMA 时再回第三篇，遇到问题时读第六篇。

### 平台与集群方向，负责网络与调度

```text
2 → 3 → 6 → 1 → 4
```

重点是硬件拓扑、RDMA 网络配置、nccl-tests 验收与排障；第一篇和第四篇帮助理解引擎为什么对拓扑和亲和有那些要求。

### 只想排查一次具体的 hang 或性能问题

```text
6 → 4 → 5
```

第六篇的决策树可以直接用；解释决策树背后的"为什么"需要第四、五篇。


## 本系列的边界

本系列只讨论通信层：数据如何在 GPU 之间、节点之间流动，由哪些软硬件承载，代价是多少，如何测量与排障。以下内容与它紧邻，但不在范围内：

- **并行策略的设计**：数据并行、张量并行、流水线并行、专家并行、FSDP / ZeRO 的分片规则与配置选择。本系列只把它们产生的通信模式（哪种原语、多大消息、多少参与者、在不在关键路径上）作为输入，不讨论怎么切分模型。
- **训练系统的容错与弹性**：节点失败后的恢复、checkpoint、弹性伸缩。本系列第六篇讨论如何定位一次 hang，但不讨论定位之后如何自动恢复训练。
- **推理引擎的调度与 KV cache 管理**：continuous batching、分页管理、prefix caching、PD 分离的调度策略。本系列第七篇只讨论 KV 从一张卡搬到另一张卡的传输层，不讨论何时搬、搬谁。
- **kernel 内部**：NCCL 的设备侧原语会读源码，vLLM 的 custom all-reduce kernel 会讲结构，但不讨论如何优化一个 CUDA kernel 的访存与占用率。
- **通用网络知识**：TCP/IP、以太网交换、路由。假设读者作为后端工程师已经具备；本系列只讲 RDMA 与 GPU 相关的部分。
- **集群网络的物理设计与运维**：交换机选型、布线、fat-tree 的层数与超额订阅比。第二篇会在拓扑一节提及它们对通信的影响，但不展开。
- **NVIDIA 之外的通信栈**：AMD 的 RCCL 与 CUDA 版 NCCL 高度对应，华为 HCCL、Intel oneCCL 结构类似；正文在相关位置提及，不展开。


## 前置要求与说明

### 前置要求

- 跑过多卡程序：用 DDP 或 FSDP 训练过模型，或用 vLLM 跑过张量并行推理；知道 `torchrun`、rank、world size 是什么，配过至少一次 `NCCL_SOCKET_IFNAME` 之类的环境变量；
- 理解 CUDA 的基本执行模型：kernel、stream、event、host 与 device 内存、异步执行与同步的含义；
- 能读 C 和 C++：NCCL 是 C++ 与 CUDA，RDMA 的 verbs 接口是 C；不要求写过大型 C++ 项目；
- 能读 Python：PyTorch 与 vLLM 的通信层 Python 部分是源码阅读对象；
- Linux 基础：`/sys`、PCI 设备、内核模块、NUMA 的基本概念；
- 硬件访问：至少一台多卡机器（理想是 8 卡 NVLink / NVSwitch 服务器）；第三篇与第六篇的多机部分需要两台以上带 InfiniBand 或 RoCE 网卡的机器，没有条件时以阅读和单机实验为主，正文会标注哪些实验需要多机。

不要求：

- 写过 RDMA 程序；
- 读过 NCCL 源码；
- 了解 InfiniBand 或 RoCE 的网络配置。

### 硬件与版本基线

- 硬件：以 **8 卡 A100 / H100 加 NVSwitch 的服务器** 为节点内的默认分析对象（A100 NVLink 双向合计 600 GB/s，H100 900 GB/s；PCIe 4.0 / 5.0 x16），节点间以 **InfiniBand HDR 200 Gb/s 或 NDR 400 Gb/s、每 GPU 一张网卡** 为默认；RoCE v2 的差异随文标注；Blackwell（第五代 NVLink、NVL72）在涉及处标注。文中带宽数字为公开标称值，且明确区分"单向"与"双向合计"；实测会因固件、拓扑、功耗与配置有差异；
- 软件：NCCL 2.27–2.29（源码路径以此为准，master 分支的目录重组随文标注）、nccl-tests 主线、CUDA 12.x、PyTorch 2.x（2.4 及之后，c10d 的环境变量名以近期源码树为准）、vLLM 主线、rdma-core 与 `libibverbs`、`nvidia-peermem` 或 DMA-BUF；
- 版本敏感处随文标注：NCCL 的算法集合（NVLS、PAT 的加入版本）与 `NCCL_ALGO` 的可选值、ProcessGroupNCCL 的 `TORCH_NCCL_*` 环境变量（早期版本无 `TORCH_` 前缀）、Flight Recorder 的接口、vLLM all_reduce 后端的选择顺序、NIXL 与 Mooncake 的接口。

### 关于替代硬件

本系列以 NVIDIA GPU 为主线。AMD 平台上 RCCL 的 API 与 NCCL 一致、Infinity Fabric 对应 NVLink，vLLM 的 quick reduce（`csrc/quickreduce/`）是它的 custom all-reduce 对应物；正文在相关位置提及差异，但不展开。国产加速器及其通信库不在范围内。


## 章节目录

1. [集合通信原语与代价模型：α-β 模型与 ring all-reduce](/collective-communication-primitives-and-cost-model.html)
2. [硬件互联：PCIe、NVLink、NVSwitch 与网络拓扑](/hardware-interconnect-pcie-nvlink-and-topology.html)
3. [RDMA 与 GPUDirect：绕过 CPU 和主机内存的数据通路](/rdma-and-gpudirect.html)
4. [NCCL 架构：拓扑探测、channel、算法与协议](/nccl-architecture-topology-channels-algorithms-and-protocols.html)
5. [PyTorch 的通信栈：ProcessGroupNCCL、stream 语义与计算通信重叠](/pytorch-communication-stack-processgroupnccl-and-streams.html)
6. [nccl-tests、调优与排障：从带宽曲线到 hang](/nccl-tests-tuning-and-debugging-hangs.html)
7. [推理侧的通信：custom all-reduce 与 KV 传输](/inference-communication-custom-all-reduce-and-kv-transfer.html)


## 最终目标

读完这套系列之后，面对任何一次通信——无论是训练日志里的一次 all_reduce timeout、profiler 里一段比预期长的通信 kernel、还是推理服务里 TP 延迟的一次异常——读者应该能够回答：

```text
这次通信传了多少字节、多少参与者？理论上要多久？   → 第一篇：α-β 模型与算法带宽效率
它是延迟主导还是带宽主导？                        → 第一篇：消息大小与拐点
数据走了哪条物理链路？这条链路的上限是多少？        → 第二篇：拓扑、NVLink / PCIe / IB 的带宽
跨机时经过了主机内存吗？GPUDirect 生效了吗？        → 第三篇：RDMA 路径与 GDR 的条件
NCCL 为什么选了这个算法和协议？切了几个 channel？   → 第四篇：拓扑探测、图搜索、调优表
框架侧有没有浪费它？重叠发生了吗？                 → 第五篇：stream / event 语义与重叠条件
实测曲线和理论差在哪里？该动哪个环境变量？           → 第六篇：nccl-tests 的读法与调优参数的层次
hang 住了，是谁、在哪一次操作上、为什么？          → 第六篇：Flight Recorder 与决策树
推理的小消息 all_reduce 为什么要绕开 NCCL？         → 第七篇：custom all-reduce 与固定开销
KV cache 该用什么传、能传多快？                   → 第七篇：NIXL / UCX / Mooncake 与单边 RDMA
```

最终目标是三种能力：

1. **阅读能力**：读懂 NCCL 的主路径、PyTorch c10d 的 ProcessGroupNCCL、vLLM 的通信后端与 KV connector，理解每个设计决定背后的硬件原因；
2. **诊断能力**：面对一次慢的或卡住的通信，用拓扑、代价模型、日志和 Flight Recorder 把问题定位到具体的一层，而不是靠试环境变量；
3. **决策能力**：为一个训练或推理任务判断通信的理论上限、选择合适的算法与传输路径、给平台提出拓扑与亲和的要求，并知道什么时候该自己写一个通信原语。

这是单卡之外一切系统的底座，也是训练与推理两条路径唯一共享的一层。
