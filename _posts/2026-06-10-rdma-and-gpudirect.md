---
layout: post
title: "通信与互联（03）：RDMA 与 GPUDirect——绕过 CPU 和主机内存的数据通路"
subtitle: "RDMA and GPUDirect: Bypassing the CPU and Host Memory"
tags: [NCCL, RDMA, GPU, AI, AI-Infra]
catalog: true
---

> 本文是[《通信与互联：从 NCCL 到 RDMA》](/communication-and-interconnect-for-ai-infra.html)系列的第 3 篇（共八篇）。上一篇：[硬件互联：PCIe、NVLink、NVSwitch 与网络拓扑](/hardware-interconnect-pcie-nvlink-and-topology.html)　下一篇：[NCCL 架构：拓扑探测、channel、算法与协议](/nccl-architecture-topology-channels-algorithms-and-protocols.html)

上一篇给出了节点间链路的物理上限：一张 NDR InfiniBand 网卡 400 Gb/s，约 50 GB/s 单向；一张 HDR 200 Gb/s，约 25 GB/s。也给出了它挂在哪里：每张 GPU 配一张网卡，两者在同一个 PCIe switch 下（`nvidia-smi topo -m` 里的 `PIX` 或 `PXB`），走 PCIe 4.0 x16（单向约 32 GB/s）或 PCIe 5.0 x16（单向约 64 GB/s）。这些数字是链路的能力，不是软件能拿到的带宽。软件拿到多少，取决于数据从显存到网线之间经过了什么。

TCP/IP 的路径上有内核协议栈、至少两次内存拷贝和一个必须参与每个包收发的 CPU。这条路径在 10 Gb/s 时代设计得很合理，到 400 Gb/s 时代它的每一段都成了瓶颈：拷贝吃掉内存带宽，协议处理吃掉 CPU 核，中断和上下文切换吃掉延迟。RDMA（Remote Direct Memory Access）把这三件事全部从路径上拿掉：网卡直接读写应用的内存，内核不参与数据面，CPU 只负责下发请求和收完成通知。GPUDirect RDMA 再进一步，让"应用的内存"可以是显存——网卡的 DMA 引擎直接读写 GPU 的 HBM，主机内存彻底退出数据面。

这一篇讲这两件事怎么工作、怎么用、怎么坏。它是系列里离硬件最近的一篇软件文章：verbs 接口是 C，NCCL 的 `src/transport/net_ib.cc` 是它唯一的网络传输实现（Socket 传输是回退路径），后面第四篇的 proxy 线程、第六篇的 `NCCL_IB_*` 环境变量、第七篇的 KV 传输，都建立在本篇的概念上。读完本篇，`NCCL_DEBUG=INFO` 日志里 `GPU Direct RDMA Enabled for GPU 0 / HCA 0` 和 `NET/IB : Got completion ... status=12` 这两行应该都能读懂。

本篇要回答总纲提出的核心问题：

> **一段显存里的数据要发到另一台机器的显存，走 TCP、走 RDMA 不开 GPUDirect、走 RDMA 开 GPUDirect，分别经过几次拷贝、经过哪些 PCIe 链路？各自的带宽上限是多少？**

依照系列惯例，本文的性能数字要么是标称值与可推导的理论上限，要么以"通常能达到"的区间给出，非实测。带宽数字每次出现都标注"单向"或"双向合计"。源码以 NCCL 2.28.9 的 `src/transport/net_ib.cc`、`src/misc/ibvwrap.cc`、`src/misc/gdrwrap.cc` 为准（2.29 起 `net_ib.cc` 拆成了 `src/transport/net_ib/` 目录，路径以你手上的版本为准）。


## 一、总览：三条路径与两本账

### 1. 从显存到对端显存的三条路径

把"一段显存发到另一台机器的显存"这件事画出来，三条路径的差别一目了然：

```text
路径 A：TCP/IP
  GPU HBM ─PCIe→ 主机内存(staging) ─CPU memcpy→ 内核 socket buffer ─NIC DMA→ 网线
  网线 ─NIC DMA→ 内核 socket buffer ─CPU memcpy→ 主机内存(staging) ─PCIe→ GPU HBM
  拷贝：每端 2 次（PCIe 拷贝 + CPU 拷贝）  CPU：协议栈 + 拷贝，每个包都参与

路径 B：RDMA，不开 GPUDirect
  GPU HBM ─PCIe→ 主机 pinned buffer ─NIC DMA(PCIe)→ 网线
  网线 ─NIC DMA(PCIe)→ 主机 pinned buffer ─PCIe→ GPU HBM
  拷贝：每端 1 次（GPU↔主机的 PCIe 拷贝）  CPU：只下发请求、轮询完成，不碰数据

路径 C：RDMA + GPUDirect RDMA
  GPU HBM ─NIC DMA(经 PCIe switch)→ 网线 ─NIC DMA(经 PCIe switch)→ GPU HBM
  拷贝：0 次  主机内存：不在数据面上  CPU：只下发请求、轮询完成
```

三条路径的**数据面**差别是拷贝次数和 PCIe 跳数，**控制面**差别是 CPU 参与的深度。RDMA 把 CPU 参与的控制面从"每个包"降到"每个 Work Request"——对 NCCL 这种一次 RDMA WRITE 搬几十 KB 到几 MB 的用法，就是每个消息一次（分段、重传、ACK 都由网卡在协议层完成），GPUDirect 把数据面从"两跳 PCIe 加一次主机内存往返"降到"PCIe switch 内部一跳"。第二章会给每条路径算出带宽上限，那张表是本篇核心问题的答案。

### 2. 两本账在这一层的形态

**带宽的账**：路径上每一段的带宽取最小值。路径 A 的最小值通常是 CPU 拷贝能力（一个核 memcpy 大约 10–15 GB/s，且拷贝在内存总线上是读加写两倍流量）；路径 B 的最小值是 GPU 的 PCIe 链路，因为同一条链路既要承担 GPU 到主机的拷贝、又要和网卡 DMA 争抢主机内存带宽；路径 C 的最小值是 min(网卡, PCIe x16)——A100 配 PCIe 4.0 x16 单向 32 GB/s，配 HDR 25 GB/s 正好塞得下；H100 配 PCIe 5.0 x16 单向 64 GB/s，配 NDR 50 GB/s 也塞得下。这不是巧合，是每一代"GPU + 网卡"的搭配都按 PCIe 不成为瓶颈来配的。

**延迟的账**：一次 RDMA WRITE 的单向延迟在同一交换机下通常 1–2 µs（InfiniBand）到 2–4 µs（RoCE v2），TCP 同机房 RTT 通常 15–50 µs。差一个数量级，来源是内核协议栈、中断、上下文切换和拷贝。但 RDMA 的延迟里还有两项 TCP 没有的固定开销：**连接建立**（QP 状态机要走三步、带外交换连接信息）和**内存注册**（pin 页、建翻译表，GB 级 buffer 要几十到几百 ms）。这两项决定了 RDMA 程序的结构：连接与注册在初始化时一次做完，数据面只做 post 与 poll。NCCL 的 MR cache、`ncclCommRegister`，第七篇 KV 传输层预注册整个 KV pool，都是这本账的产物。

### 3. 本文的章节安排

```text
二、为什么需要 RDMA           TCP 路径的拷贝与 CPU 代价；kernel bypass / zero copy / CPU offload；三条路径对照表（核心问题）
三、verbs 编程模型            device → PD → MR / CQ / QP；lkey 与 rkey；RC / UC / UD；WR 与 WC；post 与 poll
四、单边与双边操作             RDMA WRITE / READ 与 SEND / RECV；NCCL 为什么用 RDMA WRITE + IMM；net_ib.cc 的 FIFO 机制
五、连接建立                  QP 状态机；交换什么信息；TCP 还是 rdma_cm；NCCL 的多 QP 与 adaptive routing
六、InfiniBand 与 RoCE v2      链路层、LID 与 GID、信用流控与 PFC + ECN；NCCL_IB_GID_INDEX；NCCL_IB_TC / TIMEOUT / RETRY_CNT
七、内存注册的代价与 MR cache   pin 页与地址翻译；注册为什么慢；NCCL 的 ncclIbMrCache；ncclCommRegister
八、GPUDirect RDMA            网卡 DMA 到显存的机制；nvidia-peermem 与 DMA-BUF；NCCL_NET_GDR_LEVEL / GDR_READ；flush；PCIe 拓扑限制
九、GDRCopy 与 GPUDirect 家族   CPU 直接读写显存映射；NCCL 用它做什么；GPUDirect P2P / Storage
十、测一测与比一比             ibstat / ibv_devinfo / show_gids / rdma link / ib_write_bw --use_cuda；检查清单
十一、本文小结                要点、排障检查项、源码位置、comm-probe 的 rdma_write.c
```


## 二、为什么需要 RDMA

### 1. TCP 发送一段显存的完整路径

先把路径 A 每一步的成本列出来。假设应用要把 GPU 上 1 GB 的数据经 TCP 发出去：

```text
步骤                              执行者          经过的总线             内存流量（读+写）
1  cudaMemcpy D2H 到 staging       GPU DMA         GPU PCIe x16          1 GB 写主机内存
2  send(fd, staging, ...)          CPU             内存总线              读 1 GB + 写 1 GB（拷入 socket buffer）
3  协议栈：分段、校验、TCP 状态     CPU             —                      每个 MSS 一次（约 1500 或 9000 字节）
4  NIC DMA 从 skb 取数据           NIC DMA         NIC PCIe x16          读 1 GB
接收侧对称：NIC DMA 写 skb → CPU 拷到用户 buffer → cudaMemcpy H2D
```

数据面每端至少两次搬运（步骤 1 和 2），主机内存被访问四次（1 写、2 读、2 写、4 读）。控制面上 CPU 要为每一个 MSS 做协议处理——即便有 TSO / GRO / LRO 这些卸载，每个 socket 仍然只能由一个核驱动，多流并发才能用上多核。

### 2. 算一算：CPU 与内存能不能喂饱 400 Gb/s

400 Gb/s 是 50 GB/s 单向。只看步骤 2 的拷贝：一个核的 `memcpy` 通常 10–15 GB/s，喂 50 GB/s 需要 4–5 个核**只做拷贝**，加上协议处理（经验上一个核在启用卸载后能驱动 10–30 Gb/s 的 TCP 流），把一张 400 Gb/s 网卡跑满需要 15–40 个核。一台 8 卡服务器有 8 张这样的网卡，两个 CPU socket 合计通常 96–192 个核，全拿来做通信也未必够，而这些核本来该做数据加载和 Python 侧调度。

再看内存带宽。一个 socket 8 通道 DDR5-4800 的理论带宽约 300 GB/s。单张网卡满速收发时，步骤 1 到 4 加起来在主机内存上产生 4 × 50 = 200 GB/s 的流量（收发各 100 GB/s），8 张网卡是 1.6 TB/s——超出两个 socket 内存带宽的两倍以上。结论是硬的：**TCP 路径在 400 Gb/s 上不是"慢一点"，而是原理上跑不满**，因为它让主机内存和 CPU 成为每个字节都要经过的中转站。

延迟的账也一样。TCP 单向延迟里，中断、软中断、拷贝、系统调用、协议处理各占几微秒，一次 RTT 在同机房通常 15–50 µs。对一次 decode 阶段几十 KB 的 all_reduce，α 项就是全部。

### 3. RDMA 的三个承诺

RDMA 通过三个设计把上面每一项成本移走：

- **kernel bypass（内核旁路）**：数据面的所有操作——提交发送、提交接收、轮询完成——由用户态直接写网卡的门铃寄存器和读网卡写回的完成队列完成，不经过系统调用和内核协议栈。内核只在控制面参与：打开设备、分配保护域、注册内存、建立连接。
- **zero copy（零拷贝）**：网卡的 DMA 引擎直接从应用注册过的 buffer 读数据、直接写到对端应用注册过的 buffer。没有 socket buffer，没有中间拷贝。前提是 buffer 必须**事先注册**（第七章），网卡要知道它的物理地址且它不能被换出。
- **CPU offload（CPU 卸载）**：可靠传输——分段、重传、顺序、拥塞控制——全部由网卡硬件完成。CPU 每个**消息**（不是每个包）只做两件事：post 一个 Work Request，poll 一个 Work Completion。一个核能驱动一张 400 Gb/s 网卡，NCCL 的 proxy 线程就是这样一个核。

三者合起来，路径 A 的四次主机内存访问变成路径 B 的两次（GPU 拷进来、网卡读出去），CPU 从每包参与变成每消息参与。再加上 GPUDirect RDMA，主机内存访问变成零次。

### 4. 三条路径的对照表

这是本篇核心问题的答案。默认节点：8 卡 H100、PCIe 5.0 x16、每 GPU 一张 NDR 400 Gb/s 网卡，GPU 与网卡在同一 PCIe switch 下；括号里是 A100 + PCIe 4.0 + HDR 的数字。所有带宽为单向理论上限，非实测。

| | 路径 A：TCP | 路径 B：RDMA，无 GDR | 路径 C：RDMA + GPUDirect RDMA |
|---|---|---|---|
| 每端拷贝次数 | 2（PCIe D2H/H2D + CPU memcpy） | 1（PCIe D2H/H2D 到 pinned staging） | 0 |
| 主机内存在数据面上 | 是，每字节访问 4 次 | 是，每字节访问 2 次 | 否 |
| 每端经过的 PCIe 链路 | GPU x16 一次 + NIC x16 一次，都上行到 root complex | GPU x16 一次 + NIC x16 一次，都上行到 root complex | GPU x16 与 NIC x16 各一次，在 PCIe switch 内转发，不到 root complex |
| CPU 参与 | 每个 MSS：协议栈 + 拷贝 | 每个消息：post / poll | 每个消息：post / poll |
| 单流带宽上限 | 受单核限制，通常几十 Gb/s | min(NIC 50 GB/s, PCIe 5.0 64 GB/s, 主机内存) ≈ 50 GB/s（A100/PCIe 4.0：32 GB/s 的 PCIe 被 D2H 和 NIC DMA 共享主机内存，实际不到 25 GB/s） | min(NIC 50 GB/s, PCIe 5.0 64 GB/s) = 50 GB/s（A100/HDR：min(25, 32) = 25 GB/s） |
| 8 卡 8 网卡合计 | 受 CPU 核数与内存带宽限制，很难接近 8 × 50 GB/s | 主机内存流量 8 × 2 × 50 = 800 GB/s 收发合计，接近或超过两 socket 内存带宽 | 8 × 50 = 400 GB/s，主机内存流量 0 |
| 单向延迟量级 | 15–50 µs（RTT 量级） | 2–5 µs + 一次 PCIe 拷贝 | 1–2 µs（IB）/ 2–4 µs（RoCE） |

三点解读：

- **路径 B 在 PCIe 4.0 上是不够的**。A100 的 PCIe 4.0 x16 单向 32 GB/s，如果 GPU 到主机的拷贝与网卡从主机的 DMA 都要走主机内存，两条 PCIe 链路都在用，但主机内存成了汇聚点；再加上 NCCL 非 GDR 路径下是 GPU kernel 直接往主机 buffer 写（第四篇会讲 proxy 与 staging buffer），实际能到网线上的带宽通常明显低于 HDR 的 25 GB/s。这就是 GPUDirect RDMA 不是"锦上添花"而是"必需"的原因。
- **路径 C 的上限被 PCIe 而不是网卡限制**。H100 + NDR 是 min(50, 64)，PCIe 有 14 GB/s 余量；如果把 NDR 网卡插到 PCIe 4.0 的机器上，上限立刻降到 32 GB/s，网卡浪费三分之一。
- **路径 C 要求 GPU 和网卡在同一 PCIe switch 下**。跨 root complex 时 DMA 要穿过 CPU 的 PCIe 控制器和 socket 间互联（UPI / Infinity Fabric），带宽可能只有几 GB/s，有的平台干脆不支持 P2P 事务，NCCL 会因为 `NCCL_NET_GDR_LEVEL` 默认值而拒绝开 GDR（第八章）。

后面的章节按"怎么用（三、四、五章）→ 怎么配（六章）→ 代价在哪（七章）→ GPU 怎么接进来（八、九章）→ 怎么测（十章）"展开。


## 三、verbs 编程模型

### 1. 对象层级

RDMA 的用户态接口叫 **verbs**，由 rdma-core 的 `libibverbs` 提供，头文件 `<infiniband/verbs.h>`。它不是 socket 那种"一个 fd 搞定一切"的接口，而是一组显式的对象，层级如下：

```text
ibv_device                网卡（HCA）本身，ibv_get_device_list() 枚举，名字如 mlx5_0
  └─ ibv_context          打开设备得到的句柄，ibv_open_device()；所有后续对象挂在它下面
       ├─ ibv_pd          Protection Domain，ibv_alloc_pd()；隔离边界：MR 与 QP 必须属于同一 PD 才能配合使用
       │    ├─ ibv_mr     Memory Region，ibv_reg_mr()；一段被 pin 住、网卡知道物理地址的内存，附带 lkey / rkey
       │    └─ ibv_qp     Queue Pair，ibv_create_qp()；一条连接 = 一个发送队列 + 一个接收队列
       └─ ibv_cq          Completion Queue，ibv_create_cq()；网卡把完成通知写到这里，可被多个 QP 共享
```

一个端口（port）上还有 `ibv_port_attr`（`ibv_query_port()`：链路层类型、LID、MTU、状态）和 GID 表（`ibv_query_gid()`）。这些是建连接时要交换的信息，第五章讲。

NCCL 的封装在 `src/misc/ibvwrap.cc`：每个 verbs 函数有一个 `wrap_ibv_*` 版本，用 `dlopen` 加载 `libibverbs.so`（`src/misc/ibvsymbols.cc` 里的 `LOAD_SYM_VERSION` 按符号版本取 `ibv_reg_mr_iova2` 与 `ibv_reg_dmabuf_mr`），这样 NCCL 的二进制不需要链接时依赖 rdma-core，机器上没有 IB 时自动回退到 Socket 传输。`net_ib.cc` 里 `ncclIbInitCommDevBase` 一段能看到典型的初始化顺序：`wrap_ibv_alloc_pd`（每设备一个 PD，引用计数共享），然后 `wrap_ibv_create_cq`，CQ 深度是 `2*MAX_REQUESTS*ncclParamIbQpsPerConn()`。

### 2. Memory Region：lkey 与 rkey

`ibv_reg_mr(pd, addr, length, access)` 把一段虚拟地址区间交给网卡。它做三件事（细节在第七章）：把这些页 pin 在物理内存里、把虚拟到物理的映射写进网卡的翻译表、返回两个 32 位的 key：

- **lkey**（local key）：本地 post Work Request 时放在 `ibv_sge.lkey` 里，告诉网卡"这段本地地址属于哪个 MR"；
- **rkey**（remote key）：交给对端，对端在 RDMA WRITE / READ 的 `wr.rdma.rkey` 里带上它，本地网卡收到后据此校验"对端是否有权访问这段内存"。

`access` 标志决定权限：`IBV_ACCESS_LOCAL_WRITE`（本地接收要写）、`IBV_ACCESS_REMOTE_WRITE`（允许对端 RDMA WRITE）、`IBV_ACCESS_REMOTE_READ`（允许对端 RDMA READ）、`IBV_ACCESS_REMOTE_ATOMIC`。`net_ib.cc` 的 `ncclIbRegMrDmaBufInternal2` 用的是四个全开，再视 `NCCL_IB_PCI_RELAXED_ORDERING` 加上 `IBV_ACCESS_RELAXED_ORDERING`（允许 PCIe 写乱序到达以提高带宽，需要 `ibv_reg_mr_iova2`）。

rkey 是 RDMA 安全模型的全部：拿到 rkey + 地址的人就能读写那段内存，网卡只校验 rkey 与 PD、地址范围、权限位是否匹配。所以 rkey 只在受信任的网络里通过带外通道交换，NCCL 通过 bootstrap TCP socket 交换。

### 3. Queue Pair 的三种类型

QP 是 RDMA 里"连接"的载体，由一个 Send Queue 和一个 Receive Queue 组成，各自绑定一个 CQ（可以是同一个）。类型决定传输语义：

```text
类型  全称                     可靠性       连接        支持的操作                        典型用途
RC    Reliable Connected       可靠、有序   点对点       SEND/RECV、RDMA WRITE/READ、原子    NCCL、MPI、存储；本篇全部内容
UC    Unreliable Connected     不可靠、有序 点对点       SEND/RECV、RDMA WRITE               很少用
UD    Unreliable Datagram      不可靠       无连接       SEND/RECV（≤ MTU）                  管理流量、IPoIB、某些 RPC
```

RC 是唯一同时提供可靠性和单边操作的类型，NCCL 只用 RC：`ncclIbCreateQp` 里 `qpInitAttr.qp_type = IBV_QPT_RC`。RC 的代价是每对通信方要一个 QP，n 个节点全互联需要 O(n²) 个 QP，每个 QP 在网卡上占用上下文（几百字节到几 KB 的片上或主机内存缓存）。对 NCCL 这不是问题——ring / tree 里每个 rank 只和少数邻居通信；对全互联的 all_to_all 则是，这是 MoE 训练在超大规模上会碰到的一个 RDMA 层问题。

一个 QP 的容量在创建时定死：`cap.max_send_wr` / `max_recv_wr`（队列深度）、`max_send_sge` / `max_recv_sge`（每个 WR 最多几个分散聚合段）、`max_inline_data`（多小的数据可以直接塞进 WR 而不用 DMA）。NCCL 的选择是 `max_send_wr = 2*MAX_REQUESTS`（因为一次发送可能拆成一个 RDMA_WRITE 加一个 RDMA_WRITE_WITH_IMM，下一章讲）、`max_recv_wr = MAX_REQUESTS`、SGE 各 1、inline 默认 0（`NCCL_IB_USE_INLINE`）。

### 4. Work Request、Completion Queue 与轮询

数据面只有三个动作：

```text
ibv_post_send(qp, &send_wr, &bad_wr)    把一个或一串 ibv_send_wr 放进 Send Queue，敲门铃
ibv_post_recv(qp, &recv_wr, &bad_wr)    把一个 ibv_recv_wr 放进 Receive Queue，为对端的 SEND 准备落地 buffer
ibv_poll_cq(cq, n, wc_array)            从 CQ 取出最多 n 个 ibv_wc，非阻塞，返回实际取到的个数
```

`ibv_send_wr` 的关键字段：`wr_id`（64 位，原样回到 `ibv_wc.wr_id`，用来把完成和请求对上）、`opcode`（`IBV_WR_SEND` / `IBV_WR_RDMA_WRITE` / `IBV_WR_RDMA_WRITE_WITH_IMM` / `IBV_WR_RDMA_READ` / 原子）、`sg_list` + `num_sge`（本地 buffer 列表，每段 addr / length / lkey）、`send_flags`（`IBV_SEND_SIGNALED` 要不要产生完成、`IBV_SEND_INLINE` 数据是否内联、`IBV_SEND_FENCE`）、`wr.rdma.remote_addr` + `wr.rdma.rkey`（单边操作的目标）、`imm_data`（32 位立即数，随 WITH_IMM 一起送到对端的完成里）、`next`（链成一串一次 post）。

`ibv_wc` 的关键字段：`wr_id`、`status`（`IBV_WC_SUCCESS` 或错误码，`12` 是 `IBV_WC_RETRY_EXC_ERR`，NCCL 日志里最常见的那个）、`opcode`（`IBV_WC_RDMA_WRITE`、`IBV_WC_RECV`、`IBV_WC_RECV_RDMA_WITH_IMM`）、`byte_len`、`imm_data`、`qp_num`。

两个必须知道的规则：

- **接收要先 post**。对端的 SEND（以及 RDMA_WRITE_WITH_IMM）到达时，本地 Receive Queue 里必须已有一个 recv WR 等着，否则网卡回 RNR NAK（Receiver Not Ready），发送方按 `rnr_retry` 重试，重试用尽即 `IBV_WC_RNR_RETRY_EXC_ERR`。NCCL 的 `ncclIbIrecv` 在每个 QP 上 `wrap_ibv_post_recv` 一个 0 SGE 的 recv WR，就是为了接对端的 WITH_IMM。
- **不 signaled 的 send 也占队列**。Send Queue 里的 WR 直到某个它之后的 signaled WR 完成被 poll 出来才算释放。全部用 unsignaled 会把队列填满且永远清不掉。`net_ib.cc` 的 `ncclIbPostFifo` 里那段长注释讲的就是这个坑：它周期性地给 FIFO 写请求加 `IBV_SEND_SIGNALED`，保证队列会被排空。

轮询是纯用户态自旋，没有系统调用；也可以 `ibv_req_notify_cq` + `ibv_get_cq_event` 走中断等待，延迟高几微秒但不烧 CPU。NCCL 的 proxy 线程用自旋轮询（`ncclIbTest` 里的 `wrap_ibv_poll_cq(cq, 4, wcs, &wrDone)`），这是它需要一个专用核的原因，也是"proxy 线程被抢占导致通信慢"的根源——第四篇再讲。


## 四、单边与双边操作

### 1. 两类语义

verbs 提供两类数据操作：

**双边（two-sided）：SEND / RECV**。发送方 post SEND，接收方事先 post RECV 指定落地 buffer；数据到达后双方各得到一个完成。语义与 socket 的 send/recv 接近，但接收方要预先准备 buffer，且不知道对方要发多大——只能按最大值准备。接收方 CPU 参与每一个消息（post RECV、poll WC）。

**单边（one-sided）：RDMA WRITE / RDMA READ**。发起方直接指定对端的地址和 rkey，网卡把数据写到（或从）对端内存。**对端的 CPU 和软件完全不参与**，也不会得到任何完成通知——数据就静静地出现在它的内存里。发起方得到一个本地完成，表示数据已经离开本地（WRITE）或已经到达本地（READ）。

单边操作的优点正是它的缺点：对端不知道数据到了。要通知对端，有三种办法：

1. **RDMA WRITE WITH IMMEDIATE**：变体 `IBV_WR_RDMA_WRITE_WITH_IMM`，写数据的同时附带 32 位立即数，对端会消耗一个 recv WR 并产生一个 `IBV_WC_RECV_RDMA_WITH_IMM` 完成，`wc.imm_data` 里是那 32 位。对端不需要为数据准备 recv buffer（recv WR 可以 0 SGE），只需要 poll。
2. **数据里带 flag**：把 buffer 的最后几个字节当作标志位，接收方轮询内存。依赖 RDMA WRITE 的"最后一个字节最后到达"顺序保证，NCCL 的 LL / LL128 协议就是这种思路，第四篇讲。
3. **再发一个 SEND**：多一次操作，延迟多一个 α。

### 2. NCCL 为什么主要用 RDMA WRITE

读 `net_ib.cc` 的 `ncclIbMultiSend`，NCCL 发送数据的核心就是一段 RDMA WRITE：

```cpp
// src/transport/net_ib.cc, ncclIbMultiSend（2.28.9，有删节）
for (int r=0; r<nreqs; r++) {
  struct ibv_send_wr* wr = comm->wrs+r;
  // ...
  wr->opcode = IBV_WR_RDMA_WRITE;
  wr->send_flags = 0;                          // 数据 WR 不产生完成
  wr->wr.rdma.remote_addr = slots[r].addr;     // 接收方在 FIFO 里告诉我们的目标地址
  wr->next = wr + 1;
}
// Write size as immediate data. ...
uint32_t immData = 0;
if (nreqs == 1) immData = reqs[0]->send.size;
// ...
struct ibv_send_wr* lastWr = comm->wrs+nreqs-1;
if (nreqs > 1 || (comm->ar && reqs[0]->send.size > ncclParamIbArThreshold())) {
  // When using ADAPTIVE_ROUTING, send the bulk of the data first as an
  // RDMA_WRITE, then a 0-byte RDMA_WRITE_WITH_IMM to trigger a remote completion.
  lastWr++;
  // ...
}
lastWr->wr_id = wr_id;
lastWr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
lastWr->imm_data = immData;
lastWr->send_flags = IBV_SEND_SIGNALED;        // 只有最后一个 WR 产生完成
```

它选 RDMA WRITE 而不是 SEND，原因有四：

- **接收方不需要为每个消息 post 带 buffer 的 RECV**。SEND 要求接收方预先 post 一个足够大的 recv buffer，且 buffer 大小必须按最大消息准备；RDMA WRITE 的目标地址由接收方在 FIFO 里精确告知，数据直接落到它最终该在的位置（GDR 下就是 GPU 的 channel buffer）。
- **接收方 CPU 不参与数据面**。NCCL 的接收方是 GPU kernel 在轮询 channel buffer 的 flag（Simple 协议）或数据自带的 flag（LL / LL128），proxy 线程只需知道"这次接收完成了"以更新计数。RDMA WRITE 让数据直接进显存，proxy 什么也不用搬。
- **用 immediate 传大小**。`immData = reqs[0]->send.size`：接收方在 `ncclIbTest` 里 poll 到 `IBV_WC_RECV_RDMA_WITH_IMM` 时，`req->recv.sizes[0] = wc->imm_data`，从而得知实际收了多少字节——这就是 NCCL 检测接收完成的机制。多个请求合并（multi-send，`nreqs > 1`）时，立即数放不下多个大小，NCCL 改为再 RDMA WRITE 一个 `remSizesFifo` 数组到对端。
- **与 adaptive routing 兼容**。IB 网络开 adaptive routing 后包可能走不同路径乱序到达，`RDMA_WRITE_WITH_IMM` 的完成要等它之前所有数据落地才产生（RC 语义保证），所以大于 `NCCL_IB_AR_THRESHOLD`（默认 8192 字节）的消息拆成"一个大 RDMA_WRITE + 一个 0 字节 RDMA_WRITE_WITH_IMM"，让立即数在数据全部到达后才触发对端完成。

RDMA READ 在 NCCL 里只用于一个特殊目的：GDR 接收后的 **flush**（`ncclIbIflush`，`wr.opcode = IBV_WR_RDMA_READ`）。第八章解释为什么需要它。

### 3. 接收方怎么告诉发送方"往哪写"

单边 WRITE 需要发送方知道对端的地址和 rkey。NCCL 的做法是一个**反向的 FIFO**（`struct ncclIbSendFifo`）：

```text
接收方 ncclIbIrecv                          发送方 ncclIbIsend
  1. 为每个 QP post 一个 0 SGE 的 recv WR
  2. ncclIbPostFifo：把 {addr, size, rkeys[], tag, idx} ─RDMA WRITE→ 发送方的 comm->fifo[slot]
                                              3. 自旋等待 slots[0].idx == fifoHead+1（对端的 FIFO 元素到了）
                                              4. 校验 tag，ncclIbMultiSend：RDMA WRITE 数据到 slots[r].addr / rkeys
  5. ncclIbTest 轮询 CQ，收到 RECV_RDMA_WITH_IMM，记录 imm_data 为大小
```

`ncclIbSendFifo` 每个元素 64 字节（`static_assert` 保证 32 字节对齐），含 `addr`、`size`、`rkeys[NCCL_IB_MAX_DEVS_PER_NIC]`（合并多网卡时每设备一个 rkey）、`nreqs`、`tag`、`idx`。这是一个"接收方驱动"（receiver-driven）的设计：接收方先说"我准备好了，这里是地址"，发送方才写。它也是 NCCL 的一种流控——发送方永远不会写到接收方没准备好的地方，channel buffer 里的 `NCCL_STEPS` 个 slot 通过这个 FIFO 循环复用。

在两本账上：一次数据传输的延迟是"FIFO 写（一次 RDMA WRITE，约 1–2 µs）+ 数据写 + 完成"，比纯 SEND/RECV 多一个 α，但 NCCL 的 FIFO 是流水化的——接收方总是提前 post 好多个 slot，稳态下 FIFO 写不在关键路径上。带宽上，每次数据传输额外的开销只有 64 字节的 FIFO 元素和一个 0 字节的 IMM，可以忽略。


## 五、连接建立

### 1. QP 状态机

一个新建的 QP 处于 RESET 状态，不能收发。要能通信，它必须按顺序经过三次 `ibv_modify_qp`：

```text
RESET ──INIT──→ INIT ──RTR──→ RTR ──RTS──→ RTS
       本地属性          需要对端信息         本地重传参数
       pkey_index        path_mtu             timeout
       port_num          dest_qp_num          retry_cnt
       qp_access_flags   rq_psn               rnr_retry
                         ah_attr（LID 或 GID）  sq_psn
                         max_dest_rd_atomic     max_rd_atomic
                         min_rnr_timer
```

- **INIT**：设本地端口、分区 key、访问权限。到 INIT 后就可以 `ibv_post_recv`（这是为什么要在建连接前就 post 好 recv WR）。
- **RTR（Ready To Receive）**：填入对端的 QPN 和地址（Address Vector `ah_attr`：IB 用 `dlid`，RoCE 用 `grh.dgid` + `sgid_index`）、路径 MTU、期待的起始 PSN（Packet Sequence Number）。到 RTR 后可以接收对端的数据。
- **RTS（Ready To Send）**：设本地的重传参数与发送起始 PSN。到 RTS 后可以 `ibv_post_send`。

NCCL 的三步在 `net_ib.cc` 里是三个函数：`ncclIbCreateQp`（create + INIT）、`ncclIbRtrQp`、`ncclIbRtsQp`。`ncclIbRtsQp` 里 `qpAttr.timeout = ncclParamIbTimeout()`（`NCCL_IB_TIMEOUT`，默认 20）、`qpAttr.retry_cnt = ncclParamIbRetryCnt()`（`NCCL_IB_RETRY_CNT`，默认 7）、`rnr_retry = 7`（7 表示无限重试）。`timeout` 的单位是 $$4.096\,\mu s \times 2^{\text{timeout}}$$：20 对应约 4.3 秒，即一个包发出后 4.3 秒没收到 ACK 才算一次超时，重试 7 次约 30 秒后网卡报 `IBV_WC_RETRY_EXC_ERR`（status 12）。NCCL 日志里 `Got completion from peer ... with status=12` 意味着对端在 30 秒内没有回应——链路断了、对端进程死了、或者 RoCE 网络丢包严重到重传都救不回来。

### 2. 需要交换什么

RTR 需要对端的信息，所以在 INIT 和 RTR 之间必须有一次**带外交换**。双方各自把下面这些打包发给对方：

```text
QPN                本地 QP 的编号（qp->qp_num），对端填到 dest_qp_num
LID                InfiniBand 的本地标识（16 位，子网管理器分配），对端填到 ah_attr.dlid
GID                128 位全局标识（RoCE 必需；IB 跨子网时用），对端填到 ah_attr.grh.dgid
MTU                本端口的 active_mtu，双方取最小
PSN                起始包序号（可以都用 0）
rkey + 地址        如果对端要做单边操作，还要告诉它我的 buffer 地址和 rkey
```

NCCL 的 `struct ncclIbConnectionMetadata` 就是这个包：`qpInfo[]`（每个 QP 的 `qpn` 加 ECE 扩展信息）、`devs[]`（每个物理设备的 `lid`、`ib_port`、`mtu`、`link_layer`、`gid`、`fifoRkey`）、`fifoAddr`（FIFO 的地址，接收方要往这里 RDMA WRITE）、`tc`、`sl`。

### 3. 带外通道：TCP 还是 rdma_cm

交换这些信息有两条路：

- **自己用 TCP**：双方本来就有 IP 连通性（bootstrap 也要用），开一个 socket 把 metadata 结构体发过去。简单、可控、与网络类型无关。
- **`librdmacm`**：rdma-core 提供的连接管理库，用类似 socket 的 API（`rdma_create_id` / `rdma_resolve_addr` / `rdma_connect` / `rdma_accept`），走 IB 的 CM 协议或 RoCE 的 UDP 端口 4791 交换信息，并自动做路径解析、把 QP 推到 RTS。perftest 的 `ib_write_bw -R` 用它；UCX、libfabric 在底层也用它。

NCCL 选择前者：`ncclIbConnect` 里 `ncclSocketInit` + `ncclSocketConnect` 到 `ncclIbListen` 监听的地址，然后 `ncclSocketProgress` 把 `ncclIbConnectionMetadata` 发过去；`ncclIbAccept` 对称地收。这个 socket 是 NCCL 的 bootstrap 基础设施（`src/misc/socket.cc`，走 `NCCL_SOCKET_IFNAME` 指定的网卡）的一部分，第四篇讲 bootstrap 时会再见到它。选 TCP 的理由是 NCCL 本来就有 socket 层、要处理多设备合并（`NCCL_IB_MERGE_NICS`）与多 QP 这些 `rdma_cm` 不直接支持的场景。整个过程是非阻塞状态机（`enum ncclIbCommState`），因为 NCCL 要同时建几百条连接。

延迟的账：一条 RC 连接的建立包含 TCP 三次握手、一次 metadata 往返、三次 `ibv_modify_qp`（每次几十微秒，涉及内核与固件），总计通常在毫秒级。8 卡 × 32 节点的 ring 加 tree，每个 rank 要建的连接数是 channel 数 × 邻居数，几十到上百条，这是 `ncclCommInitRank` 在大规模上要花几秒到几十秒的原因之一。

### 4. NCCL 的多 QP 与 adaptive routing

一条 RC QP 在单一路径上传输，网络的 ECMP / adaptive routing 是按流（QP）做哈希的，一个 QP 在某一时刻只走一条路径。当一条链路拥塞时，单 QP 的带宽就被那条链路限制。`NCCL_IB_QPS_PER_CONNECTION`（`ncclParamIbQpsPerConn`，默认 1）让每条 NCCL 连接建多个 QP，发送时轮转：`ncclIbMultiSend` 把每个消息按 `nqps` 切成 128 字节对齐的块（`align = 128`，"so that LL and LL128 protocols still work"），依次 post 到 `comm->base.qps[qpIndex]`，`qpIndex` 循环递增。`NCCL_IB_SPLIT_DATA_ON_QPS`（默认 0）控制是否把**同一个**消息切到所有 QP 上：为 0 时只用 `nDataQps`（等于合并的设备数）个 QP 承载数据、其余 QP 空闲轮转；为 1 时切到全部 `nqps` 上。

两本账：多 QP 在拥塞的 fat-tree 上能把多路径带宽用起来（带宽的账赢）；但每个消息被切成更多更小的 WR，每个 WR 有固定开销，小消息的延迟略增，且 QP 上下文占用增加（延迟的账输）。RoCE 网络上 ECMP 哈希不均时 `NCCL_IB_QPS_PER_CONNECTION=2` 或 `4` 是常见的调优项；IB 上通常靠交换机的 adaptive routing 就够了。

`NCCL_IB_ADAPTIVE_ROUTING`（默认 -2 表示自动：IB 链路层开、RoCE 关）决定 `comm->ar`，进而决定是否把大消息拆成"RDMA_WRITE + 0 字节 WITH_IMM"两个 WR——上一章讲过原因：AR 下乱序到达，立即数必须单独在最后发。`NCCL_IB_AR_THRESHOLD` 默认 8192 字节，小于它的消息不拆。


## 六、InfiniBand 与 RoCE v2

### 1. 同一套 verbs，两种链路

RDMA 的 verbs 接口与传输语义（RC / UC / UD、WRITE / READ / SEND）是 InfiniBand 规范定义的；RoCE（RDMA over Converged Ethernet）把 IB 的传输层原样搬到以太网上：

```text
                  InfiniBand                        RoCE v2
链路层            IB 链路层，专用交换机               以太网，普通以太网交换机
网络层            IB 网络层（GRH 可选）               IPv4 / IPv6（IB 传输报文封装在 UDP 4791 里）
寻址              LID（16 位，子网管理器 SM 分配）     GID = IPv6 地址或 IPv4 映射地址（RoCE 没有 LID）
路由              SM 计算的转发表，子网内 LID 转发     普通 IP 路由，可跨三层
无损保证          链路层信用（credit-based）流控        PFC（Priority Flow Control）逐跳暂停
拥塞控制          IB CC（较少启用）+ adaptive routing  ECN 标记 + DCQCN（网卡侧速率控制）
MTU               256 B … 4096 B（active_mtu，通常 4096） 同以太网，通常 1024 … 4096 B 的 IB MTU
管理              需要 SM（opensm 或交换机内置）        不需要 SM；靠 DHCP / 静态 IP 与 ARP
```

对上层软件——包括 NCCL——两者几乎透明：同一个 `libibverbs`，同一个 `mlx5` 驱动，同一份 `net_ib.cc`。差别集中在两处：**地址向量的填法**与**网络配置的正确性**。

### 2. 寻址：LID 还是 GID

IB 子网内，`ah_attr.is_global = 0`，只填 `dlid`；RoCE 没有 LID，必须 `is_global = 1`，填 `grh.dgid`（对端 GID）、`grh.sgid_index`（本端用哪个 GID 发）、`hop_limit`、`traffic_class`。`ncclIbRtrQp` 的分支正是按 `info->link_layer == IBV_LINK_LAYER_ETHERNET` 走 GRH 路径；IB 路径下若两端 subnet prefix 不同，还会用 FLID（IB 路由器场景，`NCCL_IB_ROUTABLE_FLID_GID_INDEX`）。

RoCE 的 GID 表是问题的来源。一个端口的 GID 表通常有多个条目：每个 IP 地址（IPv4 与 IPv6、每个 VLAN）各有 RoCE v1 和 RoCE v2 两个 GID。`show_gids`（Mellanox OFED 自带脚本）能列出来：

```text
DEV     PORT  INDEX  GID                                       IPv4            VER   DEV
mlx5_0  1     0      fe80:0000:0000:0000:xxxx:xxff:fexx:xxxx                   v1    eth0
mlx5_0  1     1      fe80:0000:0000:0000:xxxx:xxff:fexx:xxxx                   v2    eth0
mlx5_0  1     2      0000:0000:0000:0000:0000:ffff:0a00:0105  10.0.1.5        v1    eth0
mlx5_0  1     3      0000:0000:0000:0000:0000:ffff:0a00:0105  10.0.1.5        v2    eth0
```

要跨三层路由必须用 RoCE v2 的 IPv4 映射 GID（这里是索引 3）。选错索引的后果是连接建立能成功（metadata 通过 TCP 交换不受影响）但第一个 RDMA 包发不到对端，最终 `IBV_WC_RETRY_EXC_ERR`。

### 3. `NCCL_IB_GID_INDEX` 为什么经常要手工指定

`net_ib.cc` 的 `ncclIbGetGidIndex` 是 NCCL 的选择逻辑：IB 链路层直接用索引 0（或 FLID 索引）；RoCE 下先看 `NCCL_IB_GID_INDEX`（`ncclParamIbGidIndex`，默认 -1），用户指定了就用，否则从索引 1 开始遍历 GID 表，用 `ncclUpdateGidIndex` 按三个条件筛选：地址族（`NCCL_IB_ADDR_FAMILY`，默认 IPv4）、RoCE 版本（`NCCL_IB_ROCE_VERSION_NUM`，默认 2）、以及可选的地址前缀（`NCCL_IB_ADDR_RANGE`）。近几年版本的自动选择在"一个端口一个 IPv4"的简单配置下通常能选对；需要手工指定的情形是：

- 端口有多个 IP（多 VLAN、bond 的子接口），自动选择可能挑到不可路由的那个；
- 需要用 IPv6 而没设 `NCCL_IB_ADDR_FAMILY`；
- 老版本 NCCL（2.18 之前）没有自动选择，默认索引 0 是 RoCE v1 link-local，跨三层必挂——这是 "RoCE 一定要设 `NCCL_IB_GID_INDEX=3`" 这条经验的来源；
- 混用不同厂商的网卡，GID 表布局不同。

检查方法：`show_gids` 找到目标 IP 对应的 v2 行，把 INDEX 填给 `NCCL_IB_GID_INDEX`；用 `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=NET` 看 `ncclIbRtrQp` 的 TRACE 或错误时打印的 `localGid` / `remoteGids`（`ncclIbTest` 出错时在 RoCE 链路上会把两端 GID 一起打出来，就是给这个用的）。

### 4. 无损与拥塞：信用流控 vs PFC + ECN

RDMA 的 RC 传输假设**网络基本不丢包**：它的重传是 go-back-N（丢一个包，之后的全部重发），一次超时要等 $$4.096\,\mu s \times 2^{\text{timeout}}$$ 量级，丢包率千分之一就能把带宽打掉一半以上。

InfiniBand 的链路层信用机制保证不丢：接收方按 buffer 空间发信用，发送方只在有信用时发包，逐跳、逐 VL（Virtual Lane）执行，硬件级、无需配置。拥塞靠 adaptive routing 分散和（较少启用的）IB CC 处理。

以太网天生会丢包，RoCE v2 要靠两层机制凑出无损：

- **PFC**（802.1Qbb）：接收方 buffer 快满时向上游发 PAUSE 帧，按优先级（通常 RoCE 流量放在优先级 3）暂停。逐跳、有效，但有 head-of-line blocking 和 PFC 风暴 / 死锁的风险，大规模部署很谨慎。
- **ECN + DCQCN**：交换机在队列超过阈值时给包打 ECN 标记（不丢），接收方网卡收到标记后向发送方回 CNP（Congestion Notification Packet），发送方网卡按 DCQCN 算法降速。这是网卡侧的拥塞控制，需要交换机的 ECN 阈值、网卡的 DCQCN 参数与 PFC 配合调好。

这些都是网络侧配置，NCCL 唯一能影响的是**流量分类**：`NCCL_IB_TC`（`ncclParamIbTc`，默认 -1，在 `ncclIbRtrQp` 里填到 `grh.traffic_class`）决定 RoCE 包的 DSCP 值（TC 的高 6 位），交换机按它映射到优先级队列；`NCCL_IB_SL`（`ncclParamIbSl`）对应 IB 的 Service Level。集群网络为 RoCE 配置的无损优先级如果对应 DSCP 26（TC = 104），NCCL 必须设 `NCCL_IB_TC=104` 才能落到无损队列——否则流量走有损队列，表现为多机带宽随规模急剧下降、日志里偶发 status 12。`NCCL_IB_FIFO_TC` 可以给控制面的 FIFO 写单独指定一个 TC（比如更高优先级），避免控制消息排在大数据后面。

带宽的账上，配置正确的 RoCE v2 与同速率 IB 的大消息带宽相近（都能到线速的 90% 以上）；延迟的账上，RoCE 通常多 1–2 µs（以太网交换机的转发延迟与 UDP 封装），而且在拥塞下抖动更大。这是"IB 更省心、RoCE 更便宜但需要网络团队"这条经验的技术内容。


## 七、内存注册的代价与 MR cache

### 1. 注册做了什么

网卡的 DMA 引擎用物理地址（准确说是 IOMMU 后的总线地址）访问内存，而应用手里是虚拟地址。`ibv_reg_mr` 要把这个鸿沟填平，三步：

1. **pin 页**：调用内核把区间内的每一页 `get_user_pages`，标记为不可换出、不可迁移。否则网卡按旧物理地址写，数据落到不知道谁的内存里。
2. **建翻译表**：把每一页的物理地址写进网卡的 **MTT**（Memory Translation Table，mlx5 上通过 UMR 或直接 firmware 命令）。网卡收到带虚拟地址与 rkey 的请求时，先查 MPT（Memory Protection Table，按 key 索引，存权限与范围）再查 MTT 翻译。翻译表存在主机内存里，网卡上有 cache（类似 TLB），miss 时要多一次 PCIe 读。
3. **返回 key**：lkey / rkey 是 MPT 的索引加一个 8 位的 key 版本。

反注册（`ibv_dereg_mr`）逆过来：清 MTT、unpin。两者都是系统调用加 firmware 命令，慢。

### 2. 算一算：注册要多久，注册多少

注册的成本与**页数**成正比。以 4 KB 页计，1 GB 有 262144 页；每页 pin 加写 MTT 通常 1–3 µs 量级（与内核版本、IOMMU、是否 THP 有关），1 GB 的注册通常要**几百毫秒**。用 2 MB 大页能把页数除以 512，时间降到几毫秒。GPU 显存注册（GPUDirect RDMA）走 `nvidia-peermem` 的 2 MB 页表，也是这个量级。

翻译表的容量也有账：一张网卡的 MTT 条目上限通常在几百万到上亿的量级（`ibv_devinfo -v` 的 `max_mr_size` / `max_mr`），主机内存 4 KB 页注册 1 TB 需要 2.7 亿条目，会碰到上限；显存以 2 MB 页注册 80 GB 只需 4 万条。

这就是为什么 RDMA 程序绝不在数据面上做注册：一次 16 MB 的传输在 50 GB/s 下只要 0.3 ms，而注册这 16 MB（4096 页）就要几 ms。**注册必须在初始化时做，且做一次用很多次**。两种策略：

- **预注册的 staging buffer**：申请一大块、注册一次，数据发送前拷进去。多一次拷贝，但注册成本为零。NCCL 的 channel buffer（`NCCL_BUFFSIZE`，默认 4 MB 每 channel 每协议）就是这样：在连接建立时注册一次，后面所有传输都经它中转。
- **注册用户 buffer + 缓存**：直接注册应用的 buffer，避免拷贝，用 cache 避免重复注册。适合同一块 buffer 反复使用的场景（训练的梯度 buffer、推理的 KV pool）。

### 3. NCCL 的 MR cache

`net_ib.cc` 里每个网卡设备有一个 `struct ncclIbMrCache`（`slots`、`capacity`、`population`），`ncclIbRegMrDmaBufInternal2` 是所有注册的入口：

```cpp
// src/transport/net_ib.cc, ncclIbRegMrDmaBufInternal2（2.28.9，有删节）
uintptr_t addr = (uintptr_t)data & -pageSize;                         // 按页对齐
size_t pages = ((uintptr_t)data + size - addr + pageSize-1)/pageSize;
std::lock_guard<std::mutex> lock(ncclIbDevs[base->ibDevN].mutex);
for (int slot=0; /*true*/; slot++) {
  if (slot == cache->population || addr < cache->slots[slot].addr) {  // 没命中：注册
    // ... 扩容 cache
    if (fd != -1) {                                                    // DMA-BUF 路径
      NCCLCHECK(wrap_ibv_reg_dmabuf_mr(&mr, base->pd, offset, pages*pageSize, addr, fd, flags));
    } else if (relaxedOrdering) {
      NCCLCHECK(wrap_ibv_reg_mr_iova2(&mr, base->pd, (void*)addr, pages*pageSize, addr, flags));
    } else {
      NCCLCHECK(wrap_ibv_reg_mr(&mr, base->pd, (void*)addr, pages*pageSize, flags));
    }
    // ... 插入 slot（按地址有序），refs = 1
    return ncclSuccess;
  } else if ((addr >= cache->slots[slot].addr) &&
      ((addr-cache->slots[slot].addr)/pageSize+pages) <= cache->slots[slot].pages) {
    cache->slots[slot].refs += 1;                                      // 命中：加引用
    *mhandle = cache->slots[slot].mr;
    return ncclSuccess;
  }
}
```

cache 是一个按起始地址排序的数组，查找是线性扫描（条目数不多，通常几十个）；命中的条件是请求区间完全落在某个已注册区间内；`ncclIbDeregMr` 减引用，到 0 才真正 `wrap_ibv_dereg_mr`。这个 cache 服务的对象是 NCCL 内部的 channel buffer 与用户注册的 buffer，`regIsGlobal = 1`（`ncclIbGetPhysProperties`）表示一次注册对该设备上的所有连接有效。

### 4. 用户 buffer 注册：`ncclCommRegister`

NCCL 2.19 起提供 `ncclCommRegister(comm, buff, size, &handle)` / `ncclCommDeregister`（`src/register/register.cc`），让应用把自己的 buffer 预先注册到 communicator 上。注册后，集合通信若发现输入输出 buffer 已注册，可以让网卡（或 NVLink 对端）直接读写用户 buffer，跳过 channel buffer 的中转——这是 NCCL 的"user buffer registration"，对大消息能省掉一次显存内拷贝并减少 kernel 的工作量。

实现是两级：`ncclRegister` 维护 communicator 级的 `ncclRegCache`（结构与 `ncclIbMrCache` 几乎相同：按地址排序、区间包含即命中、引用计数），真正到网络层的注册发生在第一次使用时，由 `src/transport/net.cc` 的 `ncclNetLocalRegisterBuffer` / `ncclNetGraphRegisterBuffer` 通过 proxy 调用网络插件的 `regMrDmaBuf`（DMA-BUF 可用时）或 `regMr`（回退到 peermem）。开关是 `NCCL_LOCAL_REGISTER`（默认 1）与 `NCCL_GRAPH_REGISTER`（CUDA Graph 捕获时自动注册，默认 1）。PyTorch 侧对应 `TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK` 之类的机制把 Caching Allocator 分出的大块自动注册——第五篇再讲。

延迟的账提醒一句：注册是同步的、慢的，`ncclCommRegister` 应该在训练循环之前调用；在第一次集合通信时才触发的注册会让那一次通信明显变慢，profiler 里会看到一个孤立的长通信。


## 八、GPUDirect RDMA

### 1. 网卡怎么 DMA 到显存

PCIe 上任何设备都可以向另一个设备的 BAR（Base Address Register）窗口发起读写事务——这是 PCIe P2P，上一篇讲过。GPU 把一部分显存映射进它的 BAR1（A100 / H100 的 BAR1 大小与显存相当，需要 Large BAR 支持），网卡的 DMA 引擎向 BAR1 里的地址发 PCIe 读写，就直接访问了显存。数据在 PCIe switch 内从网卡的下行口转到 GPU 的下行口，不上 root complex，不进主机内存。

问题在于软件：`ibv_reg_mr` 拿到一个地址，要 pin 页、查物理地址。显存的虚拟地址在 CPU 的页表里没有物理页——它是 GPU 的地址空间。所以 RDMA 驱动需要一个"如何把这个地址翻译成 BAR1 里的总线地址"的插件，有两条路：

- **`nvidia-peermem` 内核模块**（旧名 `nv_peer_mem`）：向 RDMA 子系统注册一个 peer memory client；`ibv_reg_mr` 收到显存地址时，`ib_core` 发现不是普通内存，调 `nvidia-peermem` → NVIDIA 驱动的 `nvidia_p2p_get_pages` 拿到 BAR1 里的物理页（2 MB 粒度）→ 写进 MTT。对应用透明：还是 `ibv_reg_mr(pd, cudaMalloc 的指针, ...)`。要求 `nvidia-peermem` 模块已加载（`lsmod | grep peermem`），且它是 out-of-tree 的，随 NVIDIA 驱动一起编译。
- **DMA-BUF**：Linux 内核标准的跨设备 buffer 共享机制。应用调 `cuMemGetHandleForAddressRange(&fd, ptr, size, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0)` 让 NVIDIA 驱动为这段显存导出一个 dma-buf 文件描述符，再 `ibv_reg_dmabuf_mr(pd, offset, length, iova, fd, access)` 让网卡驱动 attach 这个 dma-buf、拿到映射。不需要 `nvidia-peermem`，走内核主线 API（内核 5.12+，rdma-core 1.12+ 即 `IBVERBS_1.12` 符号版本，CUDA 11.7+，NVIDIA 开源内核模块或 R515+ 驱动）。这是未来的方向，也是容器与不能加载第三方模块的环境里的唯一选择。

NCCL 两条路都支持，探测逻辑在 `net_ib.cc`：

- `ncclIbGdrSupport`：检查 `/sys/kernel/mm/memory_peers/nv_mem/version`、`/sys/kernel/mm/memory_peers/nv_mem_nc/version` 或 `/sys/module/nvidia_peermem/version` 三个路径之一是否存在（`ibGdrSupportInitOnce`），存在则 `props->ptrSupport |= NCCL_PTR_CUDA`。
- `ncclIbDmaBufSupport`：用 `fd = -1` 试调一次 `ibv_reg_dmabuf_mr`，返回 `EOPNOTSUPP` / `EPROTONOSUPPORT` 说明内核或驱动不支持，其他错误（`EBADF`）说明支持；支持则 `props->ptrSupport |= NCCL_PTR_DMABUF`。
- `src/init.cc` 的 `dmaBufSupported` 再检查 GPU 侧：`NCCL_DMABUF_ENABLE`（默认 1）、CUDA 驱动 ≥ 11.7、`CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED` 为真。日志 `DMA-BUF is available on GPU device 0` 就是它打的。
- `src/transport/net.cc` 注册 channel buffer 时，若 `resources->useDmaBuf` 则 `cuMemGetHandleForAddressRange` + `regMrDmaBuf`，否则 "FALL-THROUGH to nv_peermem GDR path" 调 `regMr`。用户 buffer 注册（`ncclNetLocalRegisterBuffer` 一路）同样先试 DMA-BUF、失败 `goto peermem`。

`NCCL_IB_DATA_DIRECT`（默认 1）是更新的一条路：ConnectX-8 一类网卡的 "data direct" 能力，通过 `mlx5dv_reg_dmabuf_mr` 加 `MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT` 标志，让 NIC 用与 GPU 直连的专用 PCIe 路径。以你手上的网卡为准。

### 2. `NCCL_NET_GDR_LEVEL`：距离多远还开 GDR

网卡 DMA 显存虽然在 PCIe 电气上总是可能的，性能却强烈依赖路径。NCCL 在 `src/graph/paths.cc` 的 `ncclTopoCheckGdr` 里按拓扑距离决定要不要开：

```cpp
// src/graph/paths.cc, ncclTopoCheckGdr（2.28.9，有删节）
if (net->net.gdrSupport == 0) return ncclSuccess;   // 网卡不支持（peermem / dmabuf 都没有）
if (gpu->gpu.gdrSupport == 0) return ncclSuccess;   // GPU 不支持
if (read) { /* NCCL_NET_GDR_READ 的逻辑，见下 */ }
int netGdrLevel = PATH_PXB;                          // 默认：最多跨一个 PCIe switch 层级
NCCLCHECK(ncclGetLevel(&ncclTopoUserGdrLevel, NULL, "NCCL_NET_GDR_LEVEL"));
if (ncclTopoUserGdrLevel != -2) netGdrLevel = ncclTopoUserGdrLevel;
int distance = gpu->paths[NET][n].type;
// ... PXN：用中转 GPU 到网卡的距离代替；C2C 平台特殊处理
if (distance > netGdrLevel) {
  INFO(NCCL_GRAPH|NCCL_NET,"GPU Direct RDMA Disabled for GPU %d / HCA %lx (distance %d > %d)", ...);
  return ncclSuccess;
}
*gdrMode = ncclTopoGdrModeDefault;
INFO(NCCL_GRAPH|NCCL_NET,"GPU Direct RDMA Enabled for GPU %d / HCA %lx (distance %d <= %d), read %d mode %s", ...);
```

`distance` 是上一篇 `nvidia-smi topo -m` 的那套等级（`LOC` < `NVL` < `PIX` < `PXB` < `PHB` < `SYS`，第四篇讲 `paths.cc` 怎么算出来），默认阈值 `PATH_PXB`：GPU 与网卡在同一 PCIe switch 下（`PIX`）或经过多层 switch 但不经过 CPU（`PXB`）才开 GDR；要经过 CPU root complex（`PHB`）或跨 socket（`SYS`）就不开，退回主机内存 staging。原因是第二章表格最后一行：跨 root complex 的 P2P 事务在很多平台上带宽只有几 GB/s，甚至比走主机内存还慢，或者干脆不工作（IOMMU / ACS 配置）。`NCCL_NET_GDR_LEVEL=SYS` 可以强制开，只应在确认硬件路径可用（用 `ib_write_bw --use_cuda` 测过）后使用。

日志里那一行 `GPU Direct RDMA Enabled for GPU 0 / HCA 0 (distance 2 <= 3)` 就是这个函数打的；看到 `Disabled ... (distance 4 > 3)` 说明 GPU 与网卡的亲和不对，去看 `nvidia-smi topo -mp`。

### 3. `NCCL_NET_GDR_READ`：发送方向为什么默认更谨慎

GDR 有两个方向：接收（网卡**写**显存）和发送（网卡**读**显存）。`ncclTopoCheckGdr` 的 `read` 参数区分它们，`NCCL_NET_GDR_READ`（`ncclParamNetGdrRead`，默认 -2）控制发送方向：

- 0：发送永不用 GDR（数据先由 GPU 写到主机 buffer，网卡再从主机读）；
- 1：发送也用 GDR；
- -2（默认）：Ampere（`cudaCompCap >= 80`）及之后无条件开；之前的架构只在该 GPU 有 NVLink 连接（或单卡）时开。

原因是 PCIe 的读比写贵：写是 posted（发出即完成），读是 non-posted（要等数据回来），网卡对显存的读需要多个 outstanding 请求才能填满带宽，早期 GPU 的 PCIe 读响应能力有限，GDR 读在有其他 PCIe 流量时反而慢于 GPU 自己把数据写到主机。Ampere 之后不再是问题，默认全开。

### 4. flush：为什么收完还要读一次

网卡把数据 DMA 写进显存后，向 CPU（proxy 线程）写完成；proxy 更新 flag，GPU kernel 看到 flag 去读数据。PCIe 上"数据写到 GPU"与"完成写到 CPU"是两条不同的路径，PCIe 的顺序模型不保证 GPU 看到 flag 时数据已经全部落在 HBM 里（可能还在 GPU 的 PCIe 入口 buffer 里）。解决办法是 proxy 在通知 GPU 之前对刚收到的 buffer 做一次 **RDMA READ**（本地 loopback QP，读回几个字节）：READ 的响应必须等之前所有到达该地址的写都可见，这样 READ 完成即意味着数据已就位。

这就是 `ncclIbIflush` 的 `IBV_WR_RDMA_READ` 与 `gpuFlush.qp`（一个连到自己的 QP，`ncclIbAccept` 里 `ncclIbRtrQp(..., rCommDev->gpuFlush.qp.qp->qp_num, ...)` 目标 QPN 是自己）。`rComm->flushEnabled` 在 GDR 可用且 `NCCL_GDR_FLUSH_DISABLE` 为 0 时置 1。`paths.cc` 的 `ncclTopoNeedFlush` 进一步决定要不要真的 flush：`cudaCompCap >= 90`（Hopper）时不需要，Hopper 的 PCIe 实现保证了顺序；C2C 平台数据走 PCIe、flag 走 C2C 的组合要强制 flush；`NCCL_NET_FORCE_FLUSH` 可强制。GDRCopy 提供了另一种 flush 方式（`NCCL_GDRCOPY_FLUSH_ENABLE`，用 CPU 经 BAR1 读一下显存），下一章讲。

延迟的账：一次 loopback RDMA READ 约 1–2 µs，在每个接收的关键路径上。这是 Ampere 上跨机小消息比 Hopper 多出一两微秒的原因之一。

### 5. PCIe 拓扑对 GDR 的限制，再说一遍

把上一篇的拓扑知识和本章合起来，GDR 路径可用且高效的条件是：

- GPU 与网卡在同一 PCIe switch 下（`PIX`），或经多层 switch 但不过 CPU（`PXB`）；这是 8 卡 8 网卡机器的标准设计，每对 GPU + NIC 挂在一个 switch 上；
- 主板 BIOS 关闭了 ACS（Access Control Services）在 switch 下行口上的 P2P 重定向，或 IOMMU 配置允许 P2P；ACS 开着时 P2P 事务被强制送到 root complex 做检查，带宽骤降——这是"topo 显示 PIX 但 GDR 很慢"的最常见原因；
- 网卡与 GPU 的 PCIe 代际与宽度匹配：任何一端 x8 或降代都成为瓶颈（`lspci -vv` 看 `LnkSta`）；
- 一张网卡只服务一张 GPU；两张 GPU 共享一张 NDR 时每张只有 25 GB/s。

NCCL 在 GPU 找不到近的网卡时有一个补救：PXN（PCI × NVLink），让数据先经 NVLink 到一张离网卡近的 GPU，再由它做 GDR。`ncclTopoCheckGdr` 里 `distance == PATH_PXN` 那段就是用中转 GPU 的距离来判断。第四篇讲 channel 与 transport 时会回到 PXN。


## 九、GDRCopy 与 GPUDirect 家族

### 1. GDRCopy：CPU 直接读写显存

GPUDirect RDMA 解决的是"网卡访问显存"；还有一个更小的问题：**CPU 怎么低延迟地读写显存里的几个字节**。`cudaMemcpy` 走 DMA 引擎，一次调用固定开销几微秒到十几微秒，对 8 字节的 flag 太贵。GDRCopy（`libgdrapi`，内核模块 `gdrdrv`）把一段显存通过 BAR1 映射到 CPU 的虚拟地址空间，CPU 之后用普通的 load / store 访问它——一次 PCIe 事务，不到 1 µs：

```text
gdr_open()                          打开 /dev/gdrdrv
gdr_pin_buffer(g, dptr, size, ...)  pin 这段显存，拿到 BAR1 里的物理地址
gdr_map(g, handle, &va, size)       mmap 到 CPU 虚拟地址
gdr_copy_to_mapping / from_mapping  CPU 写 / 读（内部是带 write-combining 优化的 memcpy）
```

代价是 BAR1 空间（A100 / H100 够大）和 CPU 对 BAR1 的写是 posted、读是 non-posted（读较慢，几百 ns 到 1 µs）。

### 2. NCCL 用它做什么

`src/misc/gdrwrap.cc` 用 `dlopen("libgdrapi.so")` 加载上面这些符号，`ncclGdrCudaCalloc` 是封装：分配一块显存、pin、映射，返回 GPU 指针与 CPU 指针各一个。开关 `NCCL_GDRCOPY_ENABLE`（`src/init.cc`，**默认 0**），开了以后三处使用：

- `NCCL_GDRCOPY_FIFO_ENABLE`（默认 1，在 GDRCopy 开启的前提下）：把 host 到 device 的 work FIFO（`comm->workFifoBuf`，kernel 参数队列）放在显存里，CPU 通过 GDRCopy 写，GPU 读本地显存而不是跨 PCIe 读主机内存；
- `NCCL_GDRCOPY_SYNC_ENABLE`（`src/transport/net.cc`，默认 1）：proxy 线程更新的接收 tail（`gdcSync`）放在显存里，GPU kernel 轮询本地 HBM 而不是主机内存，省掉 GPU 每次轮询跨 PCIe 的读延迟（约 1 µs）；
- `NCCL_GDRCOPY_FLUSH_ENABLE`（默认 0）：用 CPU 经 BAR1 读一下显存代替 RDMA READ 做 flush。

这些都是**延迟的账**上的优化：把 GPU 与 proxy 之间的同步变量放到"读的一方本地"，让轮询不跨 PCIe。对小消息（decode 阶段的 TP all_reduce、LL 协议）有效，对大消息带宽没有影响。默认关闭是因为需要 `gdrdrv` 模块且收益随平台而异；第七篇讨论推理侧延迟时会再评估它。

### 3. GPUDirect 家族的其余成员

GPUDirect 是一个品牌，下面是一族"让 X 直接访问显存、绕开主机内存"的技术：

```text
成员                    谁访问显存         经过什么                  本系列的位置
GPUDirect P2P           另一张 GPU         NVLink 或 PCIe P2P        第二篇：NVLink / PCIe 拓扑；第四篇：NCCL 的 P2P transport；第七篇：CUDA IPC 与 custom all-reduce
GPUDirect RDMA          网卡               PCIe P2P（BAR1）          本篇
GPUDirect Storage       NVMe / 存储网卡    PCIe P2P（cuFile API）    第七篇提及：NIXL 的 GDS 后端；数据加载与 checkpoint 不在系列范围内
GPUDirect Async         GPU 自己触发网卡   GPU 写网卡门铃            NCCL 的设备侧网络（device-side networking，`ncclNetDeviceHandle_t`）方向，本系列不展开
```

GPUDirect P2P 与 RDMA 的底层机制是同一个：PCIe 事务直达 GPU 的 BAR1（NVLink 是 NVIDIA 私有链路，另有一套地址翻译）。GDRCopy 也是 BAR1 映射的应用。理解了"显存可以映射到 PCIe 地址空间、任何设备都能对它发事务"这一件事，整个家族就清楚了。


## 十、测一测与比一比：诊断工具

### 1. 设备与链路状态：`ibstat`、`ibv_devinfo`、`rdma link`

```bash
ibstat                       # 每个 HCA 每个端口：State (Active/Down)、Physical state (LinkUp)、Rate (400)、LID、link_layer
ibv_devinfo -v               # verbs 视角：fw_ver、max_qp、max_mr、max_mr_size、active_mtu、port state；-v 才有 MR 上限
rdma link                    # iproute2 的 rdma 子命令：link mlx5_0/1 state ACTIVE physical_state LINK_UP netdev eth0
rdma dev                     # 设备列表与 node_guid
```

要看的字段：`State: Active`（不是 `Initializing`，那表示 IB 子网管理器没跑起来或 RoCE 的以太网口没 UP）；`Rate` 与预期一致（400 而不是 200 或 100，降速通常是线缆或光模块问题）；`Link layer: InfiniBand` 还是 `Ethernet` 决定后面一切配置；`active_mtu`（IB 通常 4096，RoCE 取决于以太网 MTU，9000 的以太网 MTU 才能用 4096 的 IB MTU）。

### 2. RoCE 的 GID 表：`show_gids`

上一章已给出输出格式。要确认的是：目标 IP 对应的 RoCE v2 行的 INDEX 就是 `NCCL_IB_GID_INDEX` 该填的值（或者确认 NCCL 的自动选择在 `NCCL_DEBUG_SUBSYS=NET` 日志里选到了它）。多网卡机器每张网卡都要看，且各网卡的索引可能不同——`NCCL_IB_GID_INDEX` 是全局的一个值，如果不同网卡的正确索引不同，只能靠自动选择或统一网络配置。没有 `show_gids` 脚本时可以直接读 `/sys/class/infiniband/mlx5_0/ports/1/gids/*` 与 `gid_attrs/types/*`。

### 3. 带宽与延迟：perftest 的 `ib_write_bw` 与 `--use_cuda`

perftest 是 RDMA 层的 nccl-tests。三个最有用的测试：

```bash
# 服务端与客户端各跑一条，-d 指定网卡，-x 指定 GID 索引（RoCE 必需），-F 忽略 CPU 频率警告
# 主机内存 → 主机内存：这是网卡与网络的上限
server$ ib_write_bw -d mlx5_0 -x 3 -F --report_gbits -s 1048576 -n 10000
client$ ib_write_bw -d mlx5_0 -x 3 -F --report_gbits -s 1048576 -n 10000 <server-ip>

# GPU 显存 → GPU 显存：这是 GPUDirect RDMA 路径；--use_cuda=N 指定 GPU 编号；perftest 需以 CUDA 支持编译
server$ ib_write_bw -d mlx5_0 -x 3 -F --report_gbits -s 1048576 --use_cuda=0
client$ ib_write_bw -d mlx5_0 -x 3 -F --report_gbits -s 1048576 --use_cuda=0 <server-ip>

# 延迟：ib_write_lat 报单向延迟的 typical / average / 99% 分位
client$ ib_write_lat -d mlx5_0 -x 3 -F -s 8 <server-ip>

# DMA-BUF 路径（较新 perftest）：--use_cuda_dmabuf；多 QP：-q 4；用 rdma_cm 建连：-R
```

预期数字（非实测，"通常能达到"）：NDR 400 Gb/s 的主机内存测试 1 MB 消息约 370–390 Gb/s（46–49 GB/s）；`--use_cuda` 在 GPU 与网卡同 switch 且 PCIe 5.0 下应与主机内存测试相差不到 5%；PCIe 4.0（A100 + NDR）会被 PCIe 卡在约 25–28 GB/s 即 200–220 Gb/s；`ib_write_lat` 在 IB 上 8 字节单向延迟通常 1–2 µs，RoCE 2–4 µs。

**观察亲和差距**：把 `--use_cuda=0` 换成一张跨 root complex 的 GPU（`nvidia-smi topo -mp` 里与 `mlx5_0` 是 `SYS` 的那张），带宽通常掉到几 GB/s 甚至测试失败——这就是第二章表格里"跨 root complex 可能不可用或很慢"的实证，也是 `NCCL_NET_GDR_LEVEL` 默认不允许 `SYS` 的理由。

### 4. 模块与权限：`lsmod`、`ulimit`

```bash
lsmod | grep -E "peermem|nv_peer|gdrdrv"       # nvidia_peermem（GDR，peermem 路径）；gdrdrv（GDRCopy）
cat /sys/module/nvidia_peermem/version         # NCCL 的 ncclIbGdrSupport 就是查这个路径
ulimit -l                                      # memlock 上限：注册内存要 pin，必须 unlimited；容器里常见的坑
ls /dev/infiniband/                            # uverbs0 ... rdma_cm；容器要挂进来
lspci -vv -s <NIC bdf> | grep -E "LnkSta|LnkCap"   # 网卡实际协商的 PCIe 代际与宽度
```

`ulimit -l` 太小时 `ibv_reg_mr` 返回 `ENOMEM`，NCCL 日志是 `ibv_reg_mr failed`；容器里 `/dev/infiniband` 没挂、`nvidia-peermem` 在宿主机没加载、`IPC_LOCK` capability 没给，是三个最常见的"NCCL 回落到 Socket"原因。

### 5. 比一比：这一层的检查清单

理论与实测有差距时，按顺序：

```text
1  网卡状态           ibstat：Active？Rate 对？link_layer 是预期的？
2  GDR 可用           lsmod | grep peermem 或 NCCL 日志 "DMA-BUF is available"；日志 "GPU Direct RDMA Enabled"
3  GPU–NIC 亲和       nvidia-smi topo -mp：每张 GPU 到它的网卡是 PIX/PXB；NCCL 日志里每个 rank 选的 HCA 是近的那张
4  PCIe 链路          lspci LnkSta：GPU 与 NIC 都是预期代际 × x16；ACS 关闭
5  RDMA 层带宽        ib_write_bw 主机内存版 ≈ 线速 90%+；--use_cuda 版 ≈ 主机内存版；否则问题在 RDMA 层以下
6  RoCE 配置          show_gids 索引对；NCCL_IB_TC 对应无损队列的 DSCP；交换机 PFC/ECN 已开；ib_write_bw 无重传（perftest 不直接报，看网卡计数器 rdma statistics / ethtool -S 的 np_cnp_sent、rp_cnp_handled、out_of_sequence）
7  多网卡选择         NCCL_IB_HCA 限定到正确的网卡集合（"^" 排除，"=" 精确匹配；见 net_ib.cc 对 NCCL_IB_HCA 的解析）；确认没混入管理网口
8  内存注册           ulimit -l unlimited；NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=NET 里没有 ibv_reg_mr failed；大 buffer 首次通信慢是注册在数据面上
9  超时与重传         日志 status=12（RETRY_EXC）→ 对端不可达或严重丢包；status=13（RNR_RETRY_EXC）→ 对端没 post recv（通常是对端 hang 了）
```

这 9 条中前 5 条不需要 NCCL，用系统工具和 perftest 就能做；这也是它们应该先做的原因——NCCL 之上的一切问题，都以"RDMA 层本身能跑到线速"为前提。


## 十一、本文小结

### 1. 要点回顾

```text
为什么 RDMA       TCP 路径每字节访问主机内存 4 次、每个 MSS 要 CPU 参与；400 Gb/s 需要 15–40 个核与超出内存带宽的流量，原理上跑不满
                  RDMA 三个承诺：kernel bypass（数据面无系统调用）、zero copy（网卡直接读写注册过的 buffer）、CPU offload（可靠传输在网卡上）
三条路径          TCP：每端 2 次拷贝；RDMA 无 GDR：每端 1 次 PCIe 拷贝，主机内存仍在数据面；RDMA + GDR：0 拷贝，PCIe switch 内一跳
                  GDR 上限 = min(NIC, PCIe x16)：H100/PCIe 5.0/NDR = 50 GB/s；A100/PCIe 4.0/HDR = 25 GB/s；A100 配 NDR 被 PCIe 4.0 卡在 32 GB/s
verbs 对象        device → context → PD → {MR(lkey/rkey), QP(SQ+RQ), CQ}；NCCL 只用 RC QP；数据面 = post_send / post_recv / poll_cq
单边 vs 双边      RDMA WRITE/READ 对端 CPU 不参与、需要对端地址 + rkey；SEND/RECV 对端要先 post recv
                  NCCL 用 RDMA WRITE + WITH_IMM：接收方经 FIFO 告知地址与 rkey，发送方写数据、最后一个 WR 带立即数（大小）触发对端完成
连接建立          RESET → INIT → RTR（要对端 QPN + LID/GID + MTU + PSN）→ RTS（timeout/retry）；NCCL 用自己的 TCP socket 交换 ncclIbConnectionMetadata
                  NCCL_IB_TIMEOUT=20 → 4.3 s 一次超时 × NCCL_IB_RETRY_CNT=7 → 约 30 s 后 status=12
IB vs RoCE v2     同一 verbs、同一 net_ib.cc；IB 用 LID + 信用流控 + SM；RoCE 用 GID（IPv4 映射）+ PFC/ECN/DCQCN + 普通 IP
                  RoCE 要选对 GID 索引（show_gids 的 v2 行）与 TC（NCCL_IB_TC 落到无损队列）
内存注册          pin 页 + 写 MTT，每页微秒级，GB 级 buffer 百毫秒级；必须在初始化时做；NCCL 的 ncclIbMrCache 与 ncclCommRegister 都是为此
GPUDirect RDMA    网卡经 PCIe P2P 访问 GPU BAR1；nvidia-peermem（ibv_reg_mr 透明）或 DMA-BUF（cuMemGetHandleForAddressRange + ibv_reg_dmabuf_mr）
                  NCCL_NET_GDR_LEVEL 默认 PXB：跨 root complex 不开；NCCL_NET_GDR_READ 默认 Ampere+ 开；Ampere 上接收后需 RDMA READ flush，Hopper 不需要
GDRCopy           CPU 经 BAR1 映射直接读写显存，亚微秒；NCCL_GDRCOPY_ENABLE 默认 0；用于 work FIFO、同步 tail、可选 flush；延迟的账
诊断              ibstat / ibv_devinfo / rdma link / show_gids / ib_write_bw [--use_cuda] / ib_write_lat / lsmod | grep peermem / ulimit -l
```

### 2. 排障检查项

这一层出问题时先看什么（详细版在第十章第 5 节）：

- **多机带宽远低于单机、或 NCCL 日志出现 `NET/Socket`**：RDMA 没被使用。查 `ibstat` 状态、`/dev/infiniband` 是否可见、`ulimit -l`、`NCCL_IB_DISABLE` 没被误设、`NCCL_IB_HCA` 是否把网卡排除了。
- **日志有 `GPU Direct RDMA Disabled ... (distance N > 3)`**：GPU 与网卡亲和不对或 peermem / DMA-BUF 不可用。查 `nvidia-smi topo -mp`、`lsmod | grep peermem`、`DMA-BUF is available` 日志。不要直接设 `NCCL_NET_GDR_LEVEL=SYS`，先用 `ib_write_bw --use_cuda` 证明跨 root complex 路径能跑。
- **RoCE 上连接建立成功但传数据即 status=12**：GID 索引或 TC 错、或网络没配无损。`show_gids` 对索引；`NCCL_IB_TC` 对 DSCP；看网卡的 CNP / out_of_sequence 计数器。
- **status=12 偶发、伴随带宽抖动**：丢包与重传。RoCE 查 PFC / ECN；IB 查线缆误码（`ibqueryerrors`）；两者都可试 `NCCL_IB_QPS_PER_CONNECTION=2` 分散路径。
- **status=13（RNR）**：对端没有 post recv，通常对端 rank 已 hang 或崩溃，不是网络问题——转第六篇的 hang 排查。
- **第一次集合通信特别慢、之后正常**：内存注册在数据面上发生。大 buffer 用 `ncclCommRegister` 预注册（PyTorch 侧的对应机制第五篇讲）；检查是否用了大页。
- **A100 + NDR 网卡只跑到 200 Gb/s 出头**：这是 PCIe 4.0 x16 单向 32 GB/s 的上限（扣掉协议开销约 28 GB/s），不是故障。
- **`ib_write_bw` 主机内存版就不到线速**：问题在 NCCL 与 GPU 之下——网卡 PCIe 降速（`LnkSta`）、固件、MTU、线缆。先解决它。

### 3. 本篇涉及的源码与工具位置

以 NCCL 2.28.9 源码树为准；2.29 起 `src/transport/net_ib.cc` 拆成了 `src/transport/net_ib/` 目录，函数名基本不变。

| 位置 | 内容 |
|---|---|
| `src/transport/net_ib.cc` `ncclIbIsend` / `ncclIbMultiSend` | 发送路径：等 FIFO、构造 `IBV_WR_RDMA_WRITE` 链 + 末尾 `IBV_WR_RDMA_WRITE_WITH_IMM`、按 QP 轮转 post |
| `src/transport/net_ib.cc` `ncclIbIrecv` / `ncclIbPostFifo` | 接收路径：每 QP post 0 SGE recv WR；把 `ncclIbSendFifo`（addr / size / rkeys / tag / idx）RDMA WRITE 到发送方 |
| `src/transport/net_ib.cc` `ncclIbTest` | 轮询 CQ；`IBV_WC_RECV_RDMA_WITH_IMM` 的 `imm_data` 作为接收大小；错误时打印 status、opcode、两端 GID |
| `src/transport/net_ib.cc` `ncclIbIflush` | GDR 接收后的 flush：对自身 loopback QP 发 `IBV_WR_RDMA_READ` |
| `src/transport/net_ib.cc` `ncclIbCreateQp` / `ncclIbRtrQp` / `ncclIbRtsQp` | QP 状态机三步；RC 类型；IB 用 dlid、RoCE 用 GRH；timeout / retry_cnt 来源 |
| `src/transport/net_ib.cc` `ncclIbConnect` / `ncclIbAccept` / `ncclIbListen` | 用 `ncclSocket*` 交换 `ncclIbConnectionMetadata`；非阻塞状态机 `ncclIbCommState` |
| `src/transport/net_ib.cc` `ncclIbGetGidIndex` | RoCE 下 GID 索引的自动选择；`NCCL_IB_GID_INDEX` / `NCCL_IB_ROCE_VERSION_NUM` / `NCCL_IB_ADDR_FAMILY` |
| `src/transport/net_ib.cc` `ncclIbRegMrDmaBufInternal2` / `ncclIbMrCache` / `ncclIbDeregMr` | 注册缓存：按页对齐、区间包含命中、引用计数；`ibv_reg_mr` / `ibv_reg_mr_iova2` / `ibv_reg_dmabuf_mr` 三条路 |
| `src/transport/net_ib.cc` `ncclIbGdrSupport` / `ncclIbDmaBufSupport` / `ncclIbGetPhysProperties` | 探测 peermem（`/sys/module/nvidia_peermem/version` 等）与 DMA-BUF（试调 `ibv_reg_dmabuf_mr`）；填 `ptrSupport` |
| `src/transport/net_ib.cc` 顶部与各处 `NCCL_PARAM(...)` | `NCCL_IB_GID_INDEX`、`IB_TIMEOUT`、`IB_RETRY_CNT`、`IB_SL`、`IB_TC`、`IB_AR_THRESHOLD`、`IB_ADAPTIVE_ROUTING`、`IB_PCI_RELAXED_ORDERING`、`IB_USE_INLINE`、`IB_FIFO_TC`、`IB_QPS_PER_CONNECTION`、`IB_SPLIT_DATA_ON_QPS`、`IB_DISABLE`、`IB_MERGE_NICS`、`GDR_FLUSH_DISABLE`；`NCCL_IB_HCA` 由 `ncclGetEnv` 读取 |
| `src/misc/ibvwrap.cc` / `src/misc/ibvsymbols.cc` | `wrap_ibv_*` 封装；`dlopen` libibverbs，按 `IBVERBS_1.8` / `IBVERBS_1.12` 取 `ibv_reg_mr_iova2` / `ibv_reg_dmabuf_mr` |
| `src/misc/gdrwrap.cc` | `dlopen` libgdrapi：`gdr_open` / `gdr_pin_buffer` / `gdr_map` / `gdr_copy_to_mapping`；`ncclGdrCudaCalloc` |
| `src/init.cc` `initGdrCopy` / `dmaBufSupported` | `NCCL_GDRCOPY_ENABLE`（默认 0）、`NCCL_GDRCOPY_FIFO_ENABLE`、`NCCL_DMABUF_ENABLE`（默认 1）与 `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED` |
| `src/graph/paths.cc` `ncclTopoCheckGdr` / `ncclTopoNeedFlush` | `NCCL_NET_GDR_LEVEL`（默认 PXB）、`NCCL_NET_GDR_READ`（默认 -2）、`NCCL_NET_GDR_C2C`、`NCCL_NET_FORCE_FLUSH`；Hopper 不 flush |
| `src/transport/net.cc` 发送/接收 `proxySetup` 一路、`ncclNetLocalRegisterBuffer` | channel buffer 与用户 buffer 的注册：先 `cuMemGetHandleForAddressRange` + `regMrDmaBuf`，回退 `regMr`；`NCCL_GDRCOPY_SYNC_ENABLE` / `GDRCOPY_FLUSH_ENABLE` |
| `src/register/register.cc` `ncclCommRegister` / `ncclRegister` | 用户 buffer 注册的 communicator 级缓存 `ncclRegCache`；`NCCL_LOCAL_REGISTER` |
| 工具 | `ibstat`、`ibv_devinfo -v`、`rdma link`、`show_gids`、`ib_write_bw` / `ib_write_lat`（perftest，`--use_cuda`、`-x`、`-q`、`-R`）、`lsmod`、`ulimit -l`、`lspci -vv` |

### 4. comm-probe 本篇增量：`rdma_write.c`

**它做什么**：两台机器之间用 `libibverbs` 建一条 RC 连接，客户端把一块 buffer 用 `IBV_WR_RDMA_WRITE_WITH_IMM` 写到服务端的 buffer，服务端从 CQ 的 `IBV_WC_RECV_RDMA_WITH_IMM` 得知完成并校验内容；报告耗时与带宽。编译时加 `-DUSE_CUDA` 把 buffer 换成 `cudaMalloc` 的显存，走 GPUDirect RDMA。它复现的是 NCCL `net_ib.cc` 的最小骨架：PD / MR / CQ / RC QP、TCP 带外交换、三步状态机、WRITE + IMM、poll。

**输入输出**：

```text
编译   gcc -O2 rdma_write.c -o rdma_write -libverbs
       nvcc -O2 -DUSE_CUDA -x c rdma_write.c -o rdma_write_cuda -libverbs -lcuda      # 显存版
运行   server$ ./rdma_write -d mlx5_0 -g 3 -s 268435456                 # -g GID 索引（RoCE 必填；IB 填 0 或省略）
       client$ ./rdma_write -d mlx5_0 -g 3 -s 268435456 <server-ip>     # -s 字节数
输出   [client] wrote 268435456 bytes in 5.6 ms → 47.9 GB/s (383 Gb/s)   # 数字示意，非实测
       [server] got IMM=268435456, verify OK
对照   ib_write_bw -d mlx5_0 -x 3 -s 268435456 [--use_cuda=0]；不同 GPU（PIX vs SYS）各跑一遍，记录到 topo_map.py 的输出里
```

**关键实现片段**（约 110 行，去掉了参数解析、TCP 辅助函数 `tcp_listen_accept` / `tcp_connect` / `xchg`、错误检查宏与资源释放；完整版约 250 行）：

```c
// rdma_write.c -- comm-probe 第 3 篇：最小 RDMA WRITE（RC QP，TCP 带外交换）
#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

struct conn_info {               // 带外交换的全部内容：QPN、LID、GID、buffer 地址、rkey
  uint32_t qpn; uint16_t lid; uint8_t gid[16]; uint64_t addr; uint32_t rkey;
} __attribute__((packed));

static struct ibv_context *open_dev(const char *name) {
  int n; struct ibv_device **list = ibv_get_device_list(&n); struct ibv_context *ctx = NULL;
  for (int i = 0; i < n; i++)
    if (!name || !strcmp(ibv_get_device_name(list[i]), name)) { ctx = ibv_open_device(list[i]); break; }
  ibv_free_device_list(list);
  return ctx;
}

static struct ibv_wc poll_one(struct ibv_cq *cq) {  // 自旋轮询到一个完成；NCCL 的 ncclIbTest 做同样的事
  struct ibv_wc wc; int n;
  do { n = ibv_poll_cq(cq, 1, &wc); } while (n == 0);
  if (n < 0 || wc.status != IBV_WC_SUCCESS) {
    fprintf(stderr, "wc status %d (%s)\n", wc.status, ibv_wc_status_str(wc.status)); exit(1);
  }
  return wc;
}

int main(int argc, char **argv) {
  const char *dev = "mlx5_0", *server_ip = NULL; int gid_idx = 0, port = 1; size_t size = 1 << 28;
  // ... 解析 -d -g -s 与可选的 server-ip（客户端）

  struct ibv_context *ctx = open_dev(dev);
  struct ibv_port_attr pattr; ibv_query_port(ctx, port, &pattr);
  union ibv_gid gid; ibv_query_gid(ctx, port, gid_idx, &gid);
  struct ibv_pd *pd = ibv_alloc_pd(ctx);

  // 1. buffer：主机内存，或 -DUSE_CUDA 时的显存（GPUDirect RDMA）
  void *buf;
#ifdef USE_CUDA
  cudaMalloc(&buf, size);                                    // 设备指针
#else
  buf = aligned_alloc(4096, size);
#endif
  // 2. 注册：peermem 路径下显存指针可以直接传给 ibv_reg_mr；DMA-BUF 路径见下文
  struct ibv_mr *mr = ibv_reg_mr(pd, buf, size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  if (!mr) { perror("ibv_reg_mr"); return 1; }             // 显存注册失败 → 检查 lsmod | grep peermem

  // 3. CQ 与 RC QP
  struct ibv_cq *cq = ibv_create_cq(ctx, 16, NULL, NULL, 0);
  struct ibv_qp_init_attr qia = { .send_cq = cq, .recv_cq = cq, .qp_type = IBV_QPT_RC,
    .cap = { .max_send_wr = 16, .max_recv_wr = 16, .max_send_sge = 1, .max_recv_sge = 1 } };
  struct ibv_qp *qp = ibv_create_qp(pd, &qia);

  // 4. RESET → INIT
  struct ibv_qp_attr a = { .qp_state = IBV_QPS_INIT, .pkey_index = 0, .port_num = port,
    .qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE };
  ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);

  // 服务端：先 post 一个 0 SGE 的 recv WR，用来接 WRITE_WITH_IMM 的立即数（INIT 状态即可 post recv）
  if (!server_ip) { struct ibv_recv_wr rwr = { .wr_id = 2 }, *bad; ibv_post_recv(qp, &rwr, &bad); }

  // 5. 带外交换：TCP 发送本地 conn_info，接收对端的（NCCL 用 bootstrap socket 交换 ncclIbConnectionMetadata）
  struct conn_info me = { .qpn = qp->qp_num, .lid = pattr.lid, .addr = (uintptr_t)buf, .rkey = mr->rkey }, peer;
  memcpy(me.gid, gid.raw, 16);
  int sock = server_ip ? tcp_connect(server_ip, 18515) : tcp_listen_accept(18515);
  xchg(sock, &me, &peer, sizeof me);

  // 6. INIT → RTR：填入对端 QPN 与地址向量。IB 子网内用 dlid；RoCE（或指定了 GID）必须用 GRH
  memset(&a, 0, sizeof a);
  a.qp_state = IBV_QPS_RTR; a.path_mtu = pattr.active_mtu; a.dest_qp_num = peer.qpn; a.rq_psn = 0;
  a.max_dest_rd_atomic = 1; a.min_rnr_timer = 12;
  a.ah_attr.port_num = port; a.ah_attr.dlid = peer.lid;
  if (pattr.link_layer == IBV_LINK_LAYER_ETHERNET || gid_idx > 0) {
    a.ah_attr.is_global = 1; memcpy(a.ah_attr.grh.dgid.raw, peer.gid, 16);
    a.ah_attr.grh.sgid_index = gid_idx; a.ah_attr.grh.hop_limit = 64;   // 此处还可设 grh.traffic_class（对应 NCCL_IB_TC）
  }
  ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                        IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);

  // 7. RTR → RTS：本地重传参数。timeout=14 → 4.096us * 2^14 ≈ 67 ms；NCCL 用 20（≈ 4.3 s）与 retry_cnt 7
  memset(&a, 0, sizeof a);
  a.qp_state = IBV_QPS_RTS; a.timeout = 14; a.retry_cnt = 7; a.rnr_retry = 7; a.sq_psn = 0; a.max_rd_atomic = 1;
  ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                        IBV_QP_MAX_QP_RD_ATOMIC);

  if (server_ip) {
    // 8a. 客户端：填充 buffer（显存版用 cudaMemset），post 一个 RDMA WRITE WITH IMM，立即数放字节数
    struct ibv_sge sge = { .addr = (uintptr_t)buf, .length = (uint32_t)size, .lkey = mr->lkey };
    struct ibv_send_wr wr = { .wr_id = 1, .sg_list = &sge, .num_sge = 1, .opcode = IBV_WR_RDMA_WRITE_WITH_IMM,
      .send_flags = IBV_SEND_SIGNALED, .imm_data = htonl((uint32_t)size) }, *bad;
    wr.wr.rdma.remote_addr = peer.addr; wr.wr.rdma.rkey = peer.rkey;   // 对端在带外交换里给的地址与 rkey
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    ibv_post_send(qp, &wr, &bad);
    poll_one(cq);                                            // 本地完成：数据已被对端网卡 ACK
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("[client] wrote %zu bytes in %.3f ms -> %.1f GB/s\n", size, ms, size / ms / 1e6);
  } else {
    // 8b. 服务端：什么都不用做，等 CQ 里出现 RECV_RDMA_WITH_IMM——这就是"对端 CPU 不参与"
    struct ibv_wc wc = poll_one(cq);
    printf("[server] opcode %d imm %u bytes\n", wc.opcode, ntohl(wc.imm_data));   // 期待 IBV_WC_RECV_RDMA_WITH_IMM
    // ... 校验 buffer 内容（显存版先 cudaMemcpy D2H）
  }
  // ... 一次 TCP 握手确保双方都完成后再 ibv_destroy_qp / ibv_destroy_cq / ibv_dereg_mr / ibv_dealloc_pd / ibv_close_device
  return 0;
}
```

几点与正文的对应：`conn_info` 就是 `ncclIbConnectionMetadata` 的最小子集；服务端提前 post 的 0 SGE recv 对应 `ncclIbIrecv` 里的 `wrap_ibv_post_recv`；`IBV_WR_RDMA_WRITE_WITH_IMM` + `IBV_SEND_SIGNALED` 对应 `ncclIbMultiSend` 的 `lastWr`；`poll_one` 对应 `ncclIbTest`；RTR 里 IB 与 RoCE 的分支对应 `ncclIbRtrQp`。与 NCCL 的差别是这里没有 FIFO——地址与 rkey 在建连接时一次性交换，因为只写一块 buffer。

**显存版需要改什么**：上面的代码在 `nvidia-peermem` 加载的机器上，`-DUSE_CUDA` 编译后不需要改任何 verbs 调用——`ibv_reg_mr` 接受 `cudaMalloc` 返回的指针，这是 peermem 路径"对应用透明"的含义。若没有 peermem（或想验证 DMA-BUF 路径），把第 2 步换成：

```c
#ifdef USE_CUDA_DMABUF
  int fd; CUresult r = cuMemGetHandleForAddressRange(&fd, (CUdeviceptr)buf, size,
                                                     CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
  if (r != CUDA_SUCCESS) { fprintf(stderr, "no DMA-BUF support (driver/kernel/GPU)\n"); return 1; }
  // offset 0、长度 size、iova 用 buf 自身的地址：这样对端 RDMA WRITE 的 remote_addr 仍是 buf 的设备指针
  struct ibv_mr *mr = ibv_reg_dmabuf_mr(pd, 0, size, (uint64_t)buf, fd,
                                        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  close(fd);                                                 // MR 持有对 dma-buf 的引用，fd 可以关
#endif
```

这正是 NCCL `src/transport/net.cc` 的做法：`cuMemGetHandleForAddressRange` → `regMrDmaBuf` → `close(dmabuf_fd)`；`net_ib.cc` 的 `ncclIbRegMrDmaBufInternal2` 用 `addr` 同时作为 `iova`，让远端地址空间与设备指针一致。DMA-BUF 版需要 `ibv_reg_dmabuf_mr` 存在（rdma-core ≥ 34，`IBVERBS_1.12`）、CUDA ≥ 11.7、内核 ≥ 5.12 且 NVIDIA 驱动支持——`cudaDeviceGetAttribute(..., cudaDevAttrDmaBufSupported ...)` 或 CUDA driver API 的 `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED` 可以先查。

**要观察的三件事**：（1）主机内存版与 `ib_write_bw` 的数字应接近；（2）显存版在 `PIX` 亲和下应与主机内存版接近，换到 `SYS` 亲和的 GPU 后带宽下降甚至 `ibv_reg_mr` 或 WRITE 失败——这就是 `NCCL_NET_GDR_LEVEL` 默认值的实证；（3）把 `-s` 从 8 字节扫到 256 MB，画出的曲线小消息端是 α（约 2 µs 一次 WRITE + 完成），大消息端逼近 min(NIC, PCIe)，与第一篇 `cost_model.py` 的 α-β 拐点对照：$$S^* = \alpha \beta \approx 2\,\mu s \times 50\ \text{GB/s} = 100\ \text{KB}$$——小于 100 KB 的消息在这条链路上是延迟主导的。

下一篇进入 NCCL 本身。本篇讲清了网卡能做什么、一次 RDMA WRITE 怎么发出去；第四篇讲 NCCL 如何把 8 卡 × N 节点的 GPU、NVLink、PCIe、网卡组织成若干条 ring 和 tree，为每条建立 P2P / SHM / NET 传输，选算法、选协议、切 channel，让 proxy 线程替 GPU 驱动本篇的 `ncclIbIsend` / `ncclIbIrecv`：

> **同一次 8 卡 all_reduce，NCCL 在 NVSwitch 机器上选了 NVLS + Simple，在没有 NVSwitch 的 NVLink 机器上选了 Ring + LL128，在纯 PCIe 机器上只剩 Simple / LL，跨 32 台机器时选了 Tree。它是根据什么做出这三个不同决定的？强行用 `NCCL_ALGO=Ring` 会付出什么？**


## 下一篇

[NCCL 架构：拓扑探测、channel、算法与协议](/nccl-architecture-topology-channels-algorithms-and-protocols.html)
