---
layout: post
title: "通信与互联（04）：NCCL 架构——拓扑探测、channel、算法与协议"
subtitle: "NCCL Architecture: Topology Detection, Channels, Algorithms and Protocols"
tags: [NCCL, RDMA, GPU, AI, AI-Infra]
catalog: true
---

> 本文是[《通信与互联：从 NCCL 到 RDMA》](/communication-and-interconnect-for-ai-infra.html)系列的第 4 篇（共八篇）。上一篇：[RDMA 与 GPUDirect：绕过 CPU 和主机内存的数据通路](/rdma-and-gpudirect.html)　下一篇：[PyTorch 的通信栈：ProcessGroupNCCL、stream 语义与计算通信重叠](/pytorch-communication-stack-processgroupnccl-and-streams.html)

前三篇把"硬件能做到什么"的上限摆出来了：[第一篇](/collective-communication-primitives-and-cost-model.html)给出 α-β 模型和 ring all_reduce 的 $$T_{\text{ring}} = 2(n-1)\,\alpha + \frac{2(n-1)}{n}\cdot\frac{S}{\beta}$$；[第二篇](/hardware-interconnect-pcie-nvlink-and-topology.html)给 α 和 β 填上数字——NVLink 每 GPU 双向合计 600/900 GB/s（A100/H100），PCIe 4.0 x16 单向约 32 GB/s，IB NDR 单向 50 GB/s——并用 `nvidia-smi topo -m` 的 `NV#`/`PIX`/`PXB`/`PHB`/`SYS` 描述任意两个设备之间的路径；[第三篇](/rdma-and-gpudirect.html)讲清了跨机时网卡如何绕过 CPU 和主机内存直接读写显存。这些都是"能力"。本篇讲的是 NCCL 如何把这些能力**组织**成一次集合通信：它怎么知道机器长什么样、怎么决定数据走哪条路、怎么把一个 all_reduce 切成多少条并行的流、用哪种算法和协议、以及 GPU 上的 kernel 与 CPU 上的线程如何配合把字节送上网卡。

NCCL 对大多数使用者是一个黑盒：`ncclCommInitRank` 之后它就"能用"，`NCCL_DEBUG=INFO` 之后它打出几百行看不懂的日志。但它的结构并不复杂，可以压缩成两段路径。**初始化期**（`ncclCommInitRank`）做的是一次性的、与消息无关的决策：探测拓扑、算路径、搜 ring 和 tree、决定 channel 数、为每一种算法×协议组合填一张预估时间的表。**执行期**（`ncclAllReduce`）做的是与消息相关的决策：查那张表选出最快的算法×协议、决定用几个 channel、把工作切好、启动一个 kernel，需要跨机时再由 CPU 上的 proxy 线程替 kernel 驱动网卡。本篇按这个顺序把两段路径走一遍，每一步都指到 NCCL 2.28.9 源码里对应的文件和函数。

和前几篇一样，"两本账"贯穿全文。带宽的账在 NCCL 里对应三样东西：图搜索算出的每 channel 带宽、channel 数、协议的有效载荷比例（Simple 接近 100%，LL128 约 94%，LL 50%）。延迟的账对应另外三样：算法的步数（ring 是 $$2(n-1)$$ 步、tree 是 $$O(\log n)$$ 步）、每步的硬件延迟（NVLink 上零点几到几微秒、网络上十几微秒）、以及 kernel 启动与 proxy 响应的固定开销。NCCL 的调优模型本质上就是把这两本账相加取最小，本篇第八章会把它的常数表逐项摊开。

本篇要回答总纲提出的核心问题：

> **同一次 8 卡 all_reduce，NCCL 在 NVSwitch 机器上选了 NVLS + Simple，在没有 NVSwitch 的 NVLink 机器上选了 Ring + LL128，在纯 PCIe 机器上只剩 Simple / LL，跨 32 台机器时选了 Tree。它是根据什么做出这三个不同决定的？强行用 `NCCL_ALGO=Ring` 会付出什么？**

读完本篇你应该能拿着一份 `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING` 的日志，说出每一行是初始化的哪一步打出来的、NCCL 在这台机器上看到了什么拓扑、搜出了几条什么样的 ring、为这次调用选了哪个算法和协议、为什么。本篇的 comm-probe 增量 `nccl_log_reader.py` 就是把这件事自动化。

依照系列惯例，本文的性能数字要么来自源码里的模型常数（`src/graph/tuning.cc`、`src/graph/topo.h`），要么是可推导的理论值或"通常能达到"的区间，全部**非实测**。源码路径以 NCCL 2.28.9（2025-11 发布）为准；2.29 起 `src/transport/net_ib.cc` 拆成了目录，更新的版本可能继续调整目录结构，阅读时以你手上的版本对照。


## 一、总览：一次 ncclAllReduce 的两段路径

### 1. 初始化期与执行期

把总纲里的那张图展开一层，标上本篇要读的函数：

```text
ncclCommInitRank                                  src/init.cc: ncclCommInitRankDev → initTransportsRank
  bootstrap        用 uniqueId 里的地址连上 root，建 TCP 环，       src/bootstrap.cc: bootstrapInit / bootstrapAllGather
                   两次 AllGather 交换 peerInfo 与图信息
  topo detect      读 /sys 与 NVML 建 XML，节点内各 rank 的 XML 融合   src/graph/xml.cc · topo.cc: ncclTopoGetSystem
  paths            BFS 算出 GPU/NIC/CPU 两两之间的路径类型与带宽      src/graph/paths.cc: ncclTopoComputePaths
  search           按 pattern 搜 ring / tree / nvls / collnet 图     src/graph/search.cc: ncclTopoCompute
  connect          节点内图拼成全局 ring 与 double binary tree       src/graph/connect.cc: ncclTopoPreset / ncclTopoPostset
  channels         决定 nChannels，复制 channel，建 p2p 调度表        src/graph/connect.cc · src/channel.cc · src/init.cc
  tuning           为每个 (函数, 算法, 协议) 填 latency 与 bandwidth    src/graph/tuning.cc: ncclTopoTuneModel
  proxy            起 proxy service 线程（progress 线程按需再起）      src/proxy.cc: ncclProxyCreate

ncclAllReduce                                     src/collectives.cc → src/enqueue.cc: ncclEnqueueCheck
  group            隐式 ncclGroupStart/End，任务入队                 src/group.cc: ncclGroupEndInternal → groupLaunch
  prepare          按大小聚合任务，查表选 算法×协议，定 channel 数      src/enqueue.cc: ncclPrepareTasks → getAlgoInfo
  (lazy connect)   首次用到某算法时才真正建 transport 连接            src/transport/generic.cc: ncclTransportRingConnect …
  plan             把工作切到 channel，生成 kernel 参数与 proxyOp     src/enqueue.cc: scheduleCollTasksToPlan
  launch           一个 kernel，nChannels 个 block                    src/enqueue.cc: ncclLaunchKernel → cuLaunchKernelEx
  kernel           每个 block 跑一条 ring/tree，用 primitives 收发     src/device/all_reduce.h · primitives.h · prims_*.h
  proxy progress   CPU 线程替 GPU 提交 isend/irecv、轮询完成          src/proxy.cc: ncclProxyProgress · src/transport/net.cc
```

初始化期的决策全部与消息大小无关，所以它可以慢（几百毫秒到几秒），但只做一次。执行期每次调用都要走，所以它必须快——`ncclAllReduce` 在 host 侧的路径是查表和填结构体，没有搜索。

### 2. 两本账在 NCCL 里对应什么

调优模型（`src/graph/tuning.cc`）给每一种 (集合操作, 算法, 协议) 组合两个数：`latencies[coll][algo][proto]`（微秒）和 `bandwidths[coll][algo][proto]`（GB/s，已经换算成算法带宽）。执行期 `ncclTopoGetAlgoTime` 用它们算预估时间：

$$
T(\text{algo}, \text{proto}, S) = \text{lat} \times \text{latCount} + \frac{S}{1000 \times \text{bw}}
$$

这就是第一篇的 $$\alpha + S/\beta$$，只不过 α 和 β 是按算法、协议和这台机器的拓扑分别算出来的。带宽的账决定 `bw`：图搜索的每 channel 带宽 × channel 数 × 协议效率 × 算法的 busbw 到 algbw 的换算比。延迟的账决定 `lat`：基础延迟 + 步数 × 每步硬件延迟。第八章会逐项拆。

### 3. 版本与目录说明

本篇以 NCCL 源码树的 v2.28.9-1 tag 为准。这个版本的 `src/` 里除了本篇主要读的 `init.cc`、`bootstrap.cc`、`graph/`、`transport/`、`enqueue.cc`、`group.cc`、`proxy.cc`、`device/` 之外，还有几个较新的目录：`src/register/`（用户 buffer 注册，让 kernel 直接读写用户 buffer 而不经中转 buffer）、`src/scheduler/`（对称内存路径的调度）、`src/ras/`（RAS 子系统，运行时诊断）、`src/plugin/`（net / tuner / profiler / env 四类插件的加载）、`src/symmetric` 与 `src/device/symmetric/`（对称内存 kernel）。它们不在本篇主线上，涉及处顺带提及。

### 4. 本文的章节安排

```text
二、初始化        uniqueId 里有什么；bootstrap 怎么建环；NCCL_SOCKET_IFNAME 控制哪一段
三、拓扑探测      XML 树的节点类型；从 /sys 与 NVML 读什么；链路带宽常数；NCCL_TOPO_DUMP_FILE 样例
四、路径计算      BFS 与 path type 合成规则；LOC…SYS 词表；NCCL_P2P_LEVEL / NCCL_NET_GDR_LEVEL 如何用它
五、图搜索        ncclTopoGraph 的输入输出；两个 pass 的搜索；double binary tree；NCCL_GRAPH_DUMP_FILE 样例
六、transport 与 channel   P2P/SHM/NET/CollNet 的选择顺序；lazy connect；channel 是什么；nChannels 怎么定；两本账
七、算法与协议    ncclAlgoStr 的七种算法；Simple / LL / LL128 的机制与效率；哪些组合会被禁用
八、调优模型      ncclTopoTuneModel 的常数表；ncclTopoGetAlgoTime；回答核心问题；NCCL_ALGO/PROTO 与 tuner 插件
九、执行期        ncclEnqueueCheck → group → plan → 一个 kernel；primitives；proxy 线程；send/recv 配对
十、读日志        INFO 日志的格式与逐行解读；改拓扑文件观察决策变化；排障检查清单
十一、小结        要点、检查项、源码位置、comm-probe 增量 nccl_log_reader.py
```


## 二、初始化：uniqueId 与 bootstrap

### 1. ncclGetUniqueId 里装的是什么

`ncclUniqueId` 是一个 128 字节的不透明结构（`NCCL_UNIQUE_ID_BYTES`，`src/nccl.h.in`）。`ncclGetUniqueId`（`src/init.cc`）调用 `bootstrapGetUniqueId`（`src/bootstrap.cc`）：由 rank 0 所在进程创建一个监听 socket，把它的地址和一个随机 magic 写进这 128 字节。这就是为什么它必须由一个进程生成、再通过带外渠道（MPI broadcast、PyTorch 的 TCPStore、文件）发给所有 rank——它是所有 rank 找到彼此的唯一线索。

监听在哪张网卡上由两件事决定：如果设置了 `NCCL_COMM_ID`（形如 `ip:port`），`bootstrapNetInit` 用 `ncclFindInterfaceMatchSubnet` 找同一子网的接口；否则 `ncclFindInterfaces` 按 `NCCL_SOCKET_IFNAME`（`src/misc/socket.cc`）挑选。`NCCL_SOCKET_IFNAME` 影响的只是 **bootstrap 和 Socket transport** 走哪张网卡，不影响 IB 数据面（IB 的选择由 `NCCL_IB_HCA` 控制，见第六篇）。多网卡机器上 bootstrap 走错网卡（比如走到 docker0 或没有路由的管理网）是"初始化 hang"的经典原因，日志里对应的一行是 `Bootstrap: Using eth0:10.0.0.1<0>`。

### 2. bootstrap 环与 AllGather

`bootstrapInit`（`src/bootstrap.cc`）的流程：

1. 每个 rank 创建自己的监听 socket，把地址发给 root（uniqueId 里的那个地址）。rank 数超过 `NCCL_UID_STAGGER_THRESHOLD`（默认 256）时按 `NCCL_UID_STAGGER_RATE` 错开连接时间，避免几千个 rank 同时砸向 root。
2. root 收齐所有 rank 的地址后，把 rank $$r+1$$ 的地址发给 rank $$r$$。于是每个 rank 只知道自己的下一跳，形成一个 TCP **环**。
3. 之后所有 `bootstrapAllGather` 都在这个环上做：每个 rank 把数据发给 next、从 prev 收，转 $$n-1$$ 圈。这是 $$O(n)$$ 步的朴素实现，只在初始化用，不追求性能。

bootstrap 还提供 `bootstrapSend/Recv`（点对点，按需建 socket）和 `bootstrapIntraNodeAllGather`（节点内子集），后面拓扑融合和 transport 连接信息交换都靠它们。默认走 TCP；`NCCL_OOB_NET_ENABLE=1` 可以改走 net 插件（比如 IB），这是新版本加的。

### 3. initTransportsRank 的两次 AllGather

`initTransportsRank`（`src/init.cc`）开头的注释直接写了它的骨架：

```cpp
// We use 2 AllGathers
// 1. { peerInfo, comm, compCap}
// 2. { nChannels, graphInfo, topoRanks }
```

第一次 AllGather 交换 `ncclPeerInfo`：rank、cudaDev、busId（PCI 地址）、hostHash、pidHash、NCCL 版本、是否支持 GDR、shmDev 等。收齐后每个 rank 能算出 `nNodes`（有几个不同的 hostHash）、检查版本一致（不一致直接 `WARN("Mismatched NCCL version detected")` 退出）、检查有没有两个 rank 用了同一张 GPU。

第一次和第二次之间是本地计算：拓扑探测、路径、图搜索。第二次 AllGather 交换每个 rank 搜出来的图信息（每种算法的 `nChannels`、`bwIntra/bwInter`、`typeIntra/typeInter`、`crossNic`）和 `ncclTopoRanks`（我在节点内 ring 里的 prev/next、tree 的 parent/child）。收齐后所有 rank 对每种图取**最小 channel 数、最小带宽、最大路径类型**——保证所有 rank 的调优模型一致，否则不同 rank 可能选出不同的算法，那就是 hang。然后 `ncclTopoPostset` 把各节点的局部图拼成全局 ring 和 tree。

整个初始化的最后是 `ncclTopoTuneModel`、`devCommSetup`（把 channel、peer 连接信息拷到设备内存）和一次节点内 barrier。


## 三、拓扑探测：从 /sys 和 NVML 到 XML 树

### 1. 节点类型与链路类型

NCCL 内部的拓扑是一个图，节点类型六种，链路类型和路径类型共用一套编号（`src/graph/topo.h`）：

```text
节点类型   GPU  PCI(桥/switch)  NVS(NVSwitch)  CPU(实为 NUMA 域)  NIC  NET(NIC 上的一个端口/设备)

链路类型   LINK_LOC=0  LINK_NVL=1  LINK_C2C=3  LINK_PCI=4  LINK_SYS=9  LINK_NET=10
路径类型   PATH_LOC=0 PATH_NVL=1 PATH_NVB=2 PATH_C2C=3 PATH_PIX=4 PATH_PXB=5 PATH_P2C=6 PATH_PXN=7 PATH_PHB=8 PATH_SYS=9 PATH_NET=10 PATH_DIS=11
字符串     "LOC" "NVL" "NVB" "C2C" "PIX" "PXB" "P2C" "PXN" "PHB" "SYS" "NET" "DIS"   (topo.cc: topoPathTypeStr)
```

每条链路带一个带宽 `bw`（GB/s）。链路类型与路径类型"尽量对齐"（源码注释原话），所以 `NVL` 既是"一条 NVLink"也是"经 NVLink 的路径"，`PIX`/`PXB`/`PHB`/`SYS` 则只有路径含义——它们是 PCIe 路径按经过什么合成出来的。这套词表就是第二篇 `nvidia-smi topo -m` 用的那套，多出的几个是 NCCL 自己的：`NVB`（经一个中间 GPU 的 NVLink 两跳）、`C2C`（Grace-Hopper 的 CPU-GPU 链路）、`P2C`（GPU 经 C2C 到 CPU 再经 PCIe 到 NIC）、`PXN`（GPU 经 NVLink 到另一个 GPU 再到它的 NIC）。

### 2. 探测流程

`ncclTopoGetSystem`（`src/graph/topo.cc`）：

1. 如果设置了 `NCCL_TOPO_FILE`，从文件读 XML；否则尝试 `/var/run/nvidia-topologyd/virtualTopology.xml`（虚拟化环境下由 nvidia-topologyd 提供）；都没有就从空的 `<system version="1">` 开始。
2. **只探测本进程管的那张 GPU**：`ncclTopoFillGpu`（`src/graph/xml.cc`）从 busId 出发，沿 `/sys/class/pci_bus/…` 和 `/sys/bus/pci/devices/…` 向上走到 root complex，每一级创建一个 `<pci>` 节点，读 `class`、`vendor`、`device`、`link_speed`、`link_width`；到顶后挂到对应 NUMA 节点的 `<cpu>` 下（`numaid`、`affinity` 来自 `/sys/devices/system/node/nodeN/cpumap`，`arch`/`vendor`/`familyid`/`modelid` 来自 `/proc/cpuinfo` 之类）。GPU 本身用 NVML 查 `sm`（计算能力）和 NVLink：对每条 link 调 `ncclNvmlDeviceGetNvLinkRemotePciInfo` 得到对端 busId，按对端合并计数写成 `<nvlink target="…" count="…" tclass="…"/>`，`tclass` 是对端 PCI class——`0x068000` 是 NVSwitch、`0x03` 是 GPU、`0x068001` 是 CPU（C2C）。
3. 网卡：`ncclTopoProcessNet` 向 net 插件（默认 IB 或 Socket）要设备列表和属性（`getProperties`：pciPath、speed、port、guid、gdrSupport、maxComms），`ncclTopoFillNet` 沿 pciPath 同样建出 `<pci>` 链，末端挂 `<nic><net …/></nic>`。这一步还做 NIC fusion（`NCCL_NET_MERGE_LEVEL`），把同一物理卡的多个端口合成一个虚拟设备。
4. `ncclTopoTrimXml` 删掉没有 `keep="1"` 标记的分枝（从文件导入时多余的部分）。
5. **节点内融合**：每个 rank 只有自己那张 GPU 的树，通过 `bootstrapIntraNodeAllGather` 把同一 host 上所有 rank 的 XML 收齐，`ncclTopoFuseXml` 按 busId 合并成整机的树。这就是为什么每个 rank 看到的拓扑是整机的，而不只是自己那张卡。
6. 如果设置了 `NCCL_TOPO_DUMP_FILE` 且本 rank 是 `NCCL_TOPO_DUMP_FILE_RANK`（默认 0），把融合后的 XML 写到文件。
7. `ncclTopoGetSystemFromXml` 把 XML 转成 `ncclTopoSystem` 图结构。

注意第 6 步：`initTransportsRank` 里 dump 是单独调了一次 `ncclTopoGetSystem(comm, NULL, dumpXmlFile)`，与真正建图那次分开。所以 dump 出来的是 **NCCL 看到的原始拓扑**，不含后面 trim 掉的不可达 GPU。

### 3. 一个精简的 topo XML 样例

下面是一台 8 卡 A100 + NVSwitch + 每 GPU 一张 HDR 网卡的机器 dump 出来的样子（**示意**，删掉了大部分重复分枝和无关属性；属性名以 `src/graph/xml.cc` / `topo.cc` 读取的为准）：

```text
<system version="1">
  <cpu host_hash="0x..." numaid="0" affinity="ffff,ffffffff" arch="x86_64" vendor="AuthenticAMD" familyid="23" modelid="49">
    <pci busid="0000:40:00.0" class="0x060400" vendor="0x1022" device="0x1483" link_speed="16.0 GT/s PCIe" link_width="16">
      <pci busid="0000:41:00.0" class="0x060400" link_speed="16.0 GT/s PCIe" link_width="16">     <!-- PCIe switch -->
        <pci busid="0000:47:00.0" class="0x030200" vendor="0x10de" device="0x20b0" link_speed="16.0 GT/s PCIe" link_width="16">
          <gpu dev="0" sm="80" rank="0" gdr="1">
            <nvlink target="0000:c4:00.0" count="2" tclass="0x068000"/>   <!-- 到 NVSwitch 0，2 条 link -->
            <nvlink target="0000:c5:00.0" count="2" tclass="0x068000"/>
            <!-- ... 共 6 个 NVSwitch，12 条 link -->
          </gpu>
        </pci>
        <pci busid="0000:43:00.0" class="0x020700" vendor="0x15b3" device="0x101b" link_speed="16.0 GT/s PCIe" link_width="16">
          <nic>
            <net name="mlx5_0" dev="0" speed="200000" port="1" latency="0.000000" guid="0x..." maxconn="131072" gdr="1"/>
          </nic>
        </pci>
      </pci>
    </pci>
  </cpu>
  <cpu numaid="1" ...> ... </cpu>   <!-- 其余 GPU/NIC 分枝略 -->
</system>
```

读它的要点：GPU 和 NIC 挂在同一个 `<pci>`（PCIe switch）下，所以它们之间的路径是 `PIX`；两个 `<cpu>` 是两个 NUMA 域，跨域的路径是 `SYS`；`<nvlink>` 的 `count` 决定 NVLink 带宽；`sm="80"` 决定 NVLink 每 link 的带宽常数和调优表的档位。

### 4. 链路带宽是怎么填的

`ncclTopoGetSystemFromXml` 建图时按 `src/graph/topo.h` 里的常数给链路带宽（全部是 NCCL 自己的**保守估计**，单位 GB/s，不是硬件标称）：

```text
NVLink 每 link      SM60 18.0 · SM70 20.0 · SM80 20.0 · SM86 12.0 · SM90 20.6 · SM100 40.1     (ncclTopoNVLinkBw)
                    GPU→NVS/GPU 链路带宽 = count × 每 link 带宽：A100 12×20 = 240，H100 18×20.6 ≈ 371，B200 18×40.1 ≈ 722
PCIe                width × speed / 80，speed 单位 100 Mbps/lane：Gen3 x16 = 16×60/80 = 12，Gen4 = 24，Gen5 = 48   (ncclTopoAddPci)
                    PCI_BW = 12.0 是 Gen3 x16 的基准，多处判断用它
CPU 间（QPI/UPI）   Intel BDW 6 · SKL 10 · SRP 22 · ERP 40；AMD 16；ARM 6；POWER9 32                 (ncclTopoGetInterCpuBw)
                    Intel 上 P2P 流量走 64B TLP，额外乘 6/5 的开销 (INTEL_P2P_OVERHEAD)
NET                 speed(Mbps) / 8000：HDR 200000 → 25，NDR 400000 → 50；未知时按 10 Gbps          (ncclTopoAddNet)
LOC_BW = 5000       自己到自己
```

和第二篇的标称值比：A100 NVLink 单向标称 300 GB/s，NCCL 用 240；PCIe 4.0 x16 单向标称 32 GB/s，NCCL 用 24。NCCL 有意用"通常能达到"的数，因为这些数只用来**排序和比较**（走哪条路、搜几条 ring），不用来做绝对预测。日志里 `=== System : maxBw 240.0 totalBw 240.0 ===` 打的就是这套数。


## 四、路径计算：path type 与带宽

### 1. BFS 与 path type 的合成规则

`ncclTopoComputePaths`（`src/graph/paths.cc`）对每一个 CPU、GPU、NET、NVS 节点调 `ncclTopoSetPaths`，从该节点出发做一次 BFS，给图里其他节点写上"到它的路径"：经过的链路列表、路径带宽（沿途链路带宽的最小值）、路径类型。类型的合成规则就几条：

```cpp
// src/graph/paths.cc: ncclTopoSetPaths（节选）
int type = link->type == LINK_NET ? LINK_LOC : link->type;            // 起点是链路类型
if (node->type == PCI && remNode->type == PCI) type = PATH_PXB;        // 桥到桥：跨了多个 PCIe switch
if (link->type == LINK_PCI && (node->type == CPU || link->remNode->type == CPU)) type = PATH_PHB;  // 经过 CPU
if (node->type == GPU && path->type == PATH_NVL && type == PATH_NVL && remPath->count > 1) type = PATH_NVB;  // 经中间 GPU 的 NVLink
remPath->type = std::max(path->type, type);                           // 整条路径取最"远"的一段
```

"取最大"是关键：路径类型是有序的，`LOC < NVL < NVB < C2C < PIX < PXB < P2C < PXN < PHB < SYS < NET`，一条路径的类型由它最差的一段决定。经过 CPU 就至少是 `PHB`；跨了两个 CPU（走了 `LINK_SYS`）就是 `SYS`。还有一条限制：只允许把 GPU 当作**一跳**中转（`path->count > 1` 时不再经 GPU 转发），这就是 `NVB` 的来源——没有 NVSwitch 的 NVLink 机器（比如 4 卡 NVLink 桥接）上两卡之间可能没有直连，经另一张卡转一跳。

### 2. LOC…SYS 词表与 nvidia-smi topo 的对应

第二篇 `nvidia-smi topo -m` 的六个等级与 NCCL 词表一一对应：`NV#`↔`NVL`、`PIX`、`PXB`、`PHB`、`NODE`（同 CPU 不同 root complex）在 NCCL 里没有单独等级、归入 `PHB`、`SYS`↔`SYS`。NCCL 多出的 `NVB`/`C2C`/`P2C`/`PXN` 是 `nvidia-smi` 不区分的。所以看懂第二篇那张矩阵，就能预测 NCCL 会给每对设备打什么类型。

`NCCL_DEBUG_SUBSYS=GRAPH` 下 `ncclTopoPrintPaths` 会把每个节点到所有 GPU/NET 的路径打出来，格式是 `GPU/0-0-7000 (2/240.0/NVL)`——括号里是跳数/带宽/类型。

### 3. P2P 与 GDR 的 level 判定

路径算好之后，两个判定函数用路径类型做决定，它们的环境变量共用这套词表：

**`ncclTopoCheckP2p`**：两个 GPU 之间能否用 P2P transport（直接读写对端显存）。默认门限 `p2pLevel = PATH_PXB`——路径类型 ≤ PXB 才走 P2P，经过 CPU（PHB/SYS）不走。AMD CPU 且只有 2 张 GPU 时放宽到 `SYS`。用户可以用 `NCCL_P2P_LEVEL`（值是 `LOC`/`NVL`/`PIX`/`PXB`/`PHB`/`SYS`，或旧式数字）覆盖，`NCCL_P2P_DISABLE=1` 等价于 `LOC`（谁也不 P2P）。判定为不能 P2P 的 GPU 对，路径会被 `addInterStep` 改写成"经本地 CPU 中转"，后面选 transport 时就落到 SHM。此外还要通过 NVML 查 `cudaDeviceCanAccessPeer`——如果拓扑说有 NVLink 而 NVML 说不能 P2P，NCCL 会 `WARN("P2P is disabled between NVLINK connected GPUs …")`，这通常是硬件或驱动问题，`NCCL_IGNORE_DISABLED_P2P=1` 只是压住警告。

**`ncclTopoCheckGdr`**：某个 GPU 到某个 NET 设备能否用 GPUDirect RDMA。前提是 NIC 和 GPU 都报告 `gdrSupport`（第三篇的 `nvidia-peermem` 或 DMA-BUF）。然后看距离：默认 `netGdrLevel = PATH_PXB`，`NCCL_NET_GDR_LEVEL` 覆盖。距离 > 门限就打 `GPU Direct RDMA Disabled for GPU %d / HCA %lx (distance %d > %d)`，此后这条 GPU→NIC 路径被改写成经 CPU（数据先拷到主机内存再发，第三篇讲过这条路多两次 PCIe 传输）。发送方向（`read=1`，NIC 从显存读）还受 `NCCL_NET_GDR_READ` 控制，默认 Ampere 之前只在有 NVLink 时开。

这两个门限是"比一比"阶段最常查的东西：日志里没有 `GDRDMA` 字样、或者 `GPU Direct RDMA Disabled … distance 8 > 5`，就是 GPU 和 NIC 跨了 root complex（PHB=8），要么换 NIC 亲和，要么明确接受 `NCCL_NET_GDR_LEVEL=SYS` 的代价。

### 4. PXN：借邻居的 NIC

`ncclTopoComputePaths` 最后一段处理 PXN（PCI × NVLink）：如果 GPU g 到 NIC n 的路径不好（比如要过 CPU），但同节点另一张 GPU p 到 n 是 PXB 以内且 g 到 p 有 NVLink，就把 g→n 的路径改成 g→p→n，类型 `PXN`。数据先经 NVLink 到 p 的显存，再由 p 的 proxy 发到 n。这让 rail-optimized 网络（第二篇）上每个 GPU 都能用"自己 rail"的 NIC，也让多个 GPU 的流量在一张 NIC 上聚合。`NCCL_PXN_DISABLE=1` 关掉。日志里的表现是 `via NET/IB/0(1)/GDRDMA`——括号里的 1 是中转 rank——以及 `Connected all rings, use ring PXN 1 GDR 1`。


## 五、图搜索：在拓扑上找 ring 与 tree

### 1. ncclTopoGraph：输入与输出

`initTransportsRank` 为每种算法准备一个 `ncclTopoGraph`（`src/include/graph.h`）并调 `ncclTopoCompute`（`src/graph/search.cc`）：

```text
id  pattern                            minChannels        maxChannels
0   NCCL_TOPO_PATTERN_RING (4)         1                  MAXCHANNELS/2 = 32
1   NCCL_TOPO_PATTERN_BALANCED_TREE(1) ringGraph.nChannels ringGraph.nChannels   （tree 的 channel 数被钉成和 ring 一样）
2   NCCL_TOPO_PATTERN_TREE (3), collNet=1   同上          同上                    （CollNetChain，仅 collnet 可用时搜）
4   NCCL_TOPO_PATTERN_COLLNET_DIRECT(6)     1              MAXCHANNELS             （仅 collnet 可用时搜）
3   NCCL_TOPO_PATTERN_NVLS (5)         1                  MAXCHANNELS             （仅 NVLS 可用时搜）
```

输出是 `nChannels`、每 channel 的 `bwIntra`（节点内）与 `bwInter`（出节点，经 NIC）、`typeIntra/typeInter`（用到的最差路径类型）、`sameChannels`、以及每个 channel 的 GPU 顺序 `intra[c*ngpus + i]` 和进出 NIC `inter[2c]`/`inter[2c+1]`。**搜索只在本节点的拓扑上做**：一条"channel"在这里是节点内 GPU 的一个排列，加上从哪个 NIC 进、从哪个 NIC 出。节点间怎么接是下一节 `connect.cc` 的事。

三种 tree pattern 的区别是 NIC 流量怎么分：`TREE` 所有 NIC 流量都经第一个 GPU；`SPLIT_TREE` 父方向在第一个 GPU、子方向在第二个；`BALANCED_TREE` 父和一个子在第一个 GPU、另一个子在第二个。Hopper 及之后（`ccMin >= 90`）搜索时会把 `BALANCED_TREE` 退化成 `TREE`——调优模型里对应 `maxTreePattern == NCCL_TOPO_PATTERN_TREE` 时 tree 带宽乘 0.85。

### 2. 搜索过程

`ncclTopoCompute` 的目标是**在给定每 channel 带宽的前提下，找到尽可能多的 channel**，然后再试着提高带宽。带宽不是连续搜的，而是从一个离散表里取：

```cpp
// src/graph/search.cc
float speedArrayIntra[]     = { 40.0, 30.0, 20.0, 18.0, 15.0, 12.0, 10.0, 9.0, 7.0, 6.0, 5.0, 4.0, 3.0 };
float speedArrayInter[]     = { 48.0, 30.0, 28.0, 24.0, 20.0, 18.0, 15.0, 12.0, 10.0, 9.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.4, 1.2, 0.24, 0.12 };
float sm90SpeedArrayIntra[] = { 60.0, 50.0, 40.0, 30.0, 24.0, 20.0, 15.0, 12.0, 11.0, 6.0, 3.0 };
float sm90SpeedArrayInter[] = { 48.0, 45.0, 42.0, 40.0, 30.0, 24.0, 22.0, 20.0, 17.5, 15.0, 12.0, 6.0, 3.0, 2.4, 1.2, 0.24, 0.12 };
// sm100 系列更高，略
```

流程（简化）：

1. 从表里第一个不超过 `system->maxBw` 的速度开始，把 `bwIntra = bwInter = speed`。
2. `ncclTopoSearchRec` 做深度优先回溯：从第一个 GPU（或先选一个 NIC）出发，每一步沿路径类型 ≤ `typeIntra` 且剩余带宽 ≥ `speed` 的链路走到下一个 GPU（`ncclTopoFollowPath` 会从沿途链路上"扣掉"这份带宽），ring 要求走完所有 GPU 回到起点，tree 要求走完所有 GPU。找到一条就记为一个 channel，继续找下一条，直到带宽扣完或到 `maxChannels`。搜索有步数预算（`NCCL_SEARCH_TIMEOUT` 等），到了就放弃这个分枝。
3. `time == -1` 表示找到了"完美解"（channel 数 × 带宽 ≥ `totalBw`），停止。否则**第一个 pass** 依次放宽：允许各 channel 不同（`sameChannels=0`）→ 换更简单的 tree pattern → 允许更差的 `typeIntra`/`typeInter` → 允许 crossNic（进出不同 NIC，`NCCL_CROSS_NIC`）→ 降一档速度重来。
4. 有解后进入**第二个 pass**：`ncclTopoDupChannels` 在带宽够高时（`bwIntra >= 25`）把 channel 复制一倍、每条带宽减半；然后尝试升一档速度看能不能保持同样的 channel 数。
5. 完全找不到时退化为"按 GPU 编号顺序的 1 条 ring，带宽 0.1"，并打 `Could not find a path for pattern %d, falling back to simple order`。

结果由 `ncclTopoPrintGraph` 打出：`Pattern 4, crossNic 0, nChannels 12, bw 20.000000/20.000000, type NVL/PIX, sameChannels 1`，后面每行一个 channel 的 GPU 顺序（多机时首尾带 NET）。以 8 卡 A100 + NVSwitch 为例，GPU→NVS 链路 240 GB/s，`speedArrayIntra` 从 40 开始：40 一条只能放 6 条（240/40）、不到 totalBw；降到 20 可以放 12 条 = 240，命中 `nChannels*bwInter >= totalBw`，得到 **12 条 20 GB/s 的 ring**。这就是 A100 机器日志里常见的 `nChannels 12, bw 20.0/20.0` 的来源。

### 3. 从节点内图到全局 ring 与 tree：connect.cc

`ncclTopoPreset`（`src/graph/connect.cc`）在第二次 AllGather 前，把本节点每个 channel 的 `intra[]` 翻译成"我在这条 ring 里的 prev/next、这条 ring 在本节点的入口 rank（`ringRecv`）和出口 rank（`ringSend`）、tree 的本节点父/子候选"，装进 `ncclTopoRanks` 发出去。

`ncclTopoPostset` 收齐所有节点后：

- **`connectRings`**：把节点 $$n$$ 的出口接到节点 $$n+1$$ 的入口，所有节点的局部路径就串成一个全局环。`NCCL_CROSS_NIC=2` 且解带 crossNic 时，奇数节点把相邻两个 channel 的 ring 交换，让相邻 ring 走不同 rail。
- **`connectTrees`**：调 `ncclGetDtree` 在**节点**之间建 double binary tree（每个节点是树的一个顶点），再把节点内的 GPU 链接进去——节点内是一条链（`intra[]` 顺序），链头（`treeToParent`）负责和父节点通信，`treeToChild0/1` 负责和两个子节点通信。日志 `Tree 0 : -1 -> 0 -> 1/8/-1` 的格式是 `up -> me -> down[0]/down[1]/down[2]`。
- **channel 复制**：`nChannels = min(MAXCHANNELS, nChannels*2)`——每个搜出来的 channel 变两个，第二份跑另一棵树（double binary tree 的两棵），ring 则完全相同。这是为什么 12 变 24。Hopper 多机且节点内带宽高时还会再翻一倍到 16 以上。
- 最后按 `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS`（旧名 `NCCL_MIN_NRINGS`/`NCCL_MAX_NRINGS`，`ncclMinNchannels`/`ncclMaxNchannels`）和 `ncclConfig_t` 的 `minCTAs`/`maxCTAs` 裁剪或复制，`ncclBuildRings` 从 prev/next 数组重建每条 ring 并校验它确实回到起点、包含所有 rank，rank 0 打出 `Channel %02d/%02d : 0 1 2 3 …`（`src/graph/rings.cc: dumpLine`）。

### 4. double binary tree

`src/graph/trees.cc` 只有一百多行，值得读一遍。`ncclGetBtree` 用位运算建一棵"叶节点和内部节点交替"的二叉树：rank 的最低置位比特决定它在第几层，父节点是把该比特清零再置高一位，两个子节点是加减半个比特。`ncclGetDtree` 用它建两棵树：偶数个节点时第二棵是第一棵的**镜像**（rank $$r$$ 映射到 $$n-1-r$$），奇数时是**平移一位**。效果是：在第一棵树里是叶子的节点，在第二棵里是内部节点，反之亦然。第一篇讲过朴素二叉树的带宽只有一半（叶子只发不收或只收不发），两棵互补的树各承担一半数据，把带宽补回来——这就是第一篇 tree 一节的实现。

树的深度记在 `channel->tree.depth = nRanks/nNodes - 1 + log2i(nNodes)`：节点内链的长度加节点间树的高度。8 卡 × 32 节点：$$7 + 5 = 12$$。

### 5. NCCL_GRAPH_DUMP_FILE 样例

`ncclTopoDumpGraphs` 在 `NCCL_GRAPH_DUMP_FILE` 设置时把五种图写成 XML（`ncclTopoGetXmlFromGraph`）。两机 16 卡的样例（**示意**，只保留 ring 图的前两个 channel）：

```text
<graphs version="1">
  <graph id="0" pattern="4" crossnic="0" nchannels="12" speedintra="20" speedinter="20" latencyinter="0" typeintra="NVL" typeinter="PIX" samechannels="1">
    <channel>
      <net dev="0"/>
      <gpu dev="0"/> <gpu dev="1"/> <gpu dev="2"/> <gpu dev="3"/> <gpu dev="4"/> <gpu dev="5"/> <gpu dev="6"/> <gpu dev="7"/>
      <net dev="0"/>
    </channel>
    <channel>
      <net dev="1"/>
      <gpu dev="1"/> <gpu dev="0"/> <gpu dev="3"/> <gpu dev="2"/> <gpu dev="5"/> <gpu dev="4"/> <gpu dev="7"/> <gpu dev="6"/>
      <net dev="1"/>
    </channel>
    <!-- ... -->
  </graph>
  <graph id="1" pattern="1" ...> ... </graph>   <!-- tree；id 4/2 是 collnet direct/chain；id 3 是 nvls，每个 channel 只列一个 gpu（head） -->
</graphs>
```

这个文件可以改了再用 `NCCL_GRAPH_FILE` 喂回去（`ncclTopoCompute` 开头就检查它），跳过搜索。`nchannels`、`speedintra/speedinter`、`typeintra/typeinter` 会直接进调优模型——第十章会用它做实验。


## 六、transport 与 channel

### 1. 四种 transport 与选择顺序

`src/transport.cc` 里 `ncclTransports[]` 的顺序就是尝试顺序：

```cpp
struct ncclTransport* ncclTransports[NTRANSPORTS+1] = {
  &p2pTransport,      // src/transport/p2p.cc   同机 GPU 直接读写对端显存（NVLink 或 PCIe P2P）
  &shmTransport,      // src/transport/shm.cc   同机，经主机共享内存中转
  &netTransport,      // src/transport/net.cc   跨机，经 net 插件（IB / Socket / 第三方）
  &collNetTransport,  // src/transport/coll_net.cc  跨机，交换机内归约（IB SHARP）
  &profilerTransport  // 不是真 transport
};
```

`selectTransport` 对一对 (我, peer) 按顺序调各 transport 的 `canConnect`，第一个说行的就用。P2P 的 `canConnect` 依赖第四章的 `ncclTopoCheckP2p`；SHM 要求同 host（`hostHash` 相同）且未 `NCCL_SHM_DISABLE`；NET 总是可以。所以同机默认 P2P，P2P 不行（跨 root complex 且没放宽 `NCCL_P2P_LEVEL`）落 SHM，跨机 NET。每个 transport 有 `send`/`recv` 两个 `ncclTransportComm`，各含 `setup`（分配 buffer、准备连接信息）、`connect`（拿到对端信息后建立连接）、`proxyProgress`（proxy 线程的推进函数，只有 NET/CollNet 有实质内容）等回调。

日志里每条连接一行，格式随 transport（`src/transport/p2p.cc` / `shm.cc` / `net.cc` 的 `INFO(NCCL_INIT|…)`）：

```text
Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM                 同进程内是 P2P/direct pointer；多进程用 cuMem 导出句柄；老路径是 P2P/IPC
Channel 00/0 : 0[0] -> 2[2] via P2P/indirect/1[1]         NVB：经 GPU1 中转
Channel 00 : 0[0] -> 4[4] via SHM/direct/direct            经共享内存；CE 表示用 copy engine 搬
Channel 00/0 : 7[7] -> 8[0] [send] via NET/IB/0/GDRDMA     经 IB 设备 0，开了 GPUDirect RDMA
Channel 00/0 : 15[7] -> 0[0] [receive] via NET/IB/0/GDRDMA
Channel 00/0 : 3[3] -> 8[0] [send] via NET/IB/0(1)/GDRDMA  PXN：经 rank 1 的 proxy 用它的 NIC 0
```

`%d[%d]` 是 `rank[nvmlDev]`。没有 `/GDRDMA` 后缀就是数据要经主机内存——这是"多机远差于单机"时第一个要查的字样。

NVLS（`src/transport/nvls.cc`）不在这个数组里，它不是点对点 transport：`ncclNvlsInit` 检查 `NCCL_NVLS_ENABLE`（默认 2 = 自动）、驱动是否有 `cuMulticastCreate`、设备是否 `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`，然后 `ncclNvlsSetup` 用 CUDA 的 multicast object 建一块所有 GPU 共享的多播地址空间，NVSwitch 在这块地址上做归约与广播。CollNet 同样单独走 `ncclCollNetSetup`，需要 collnet 插件（SHARP）且节点数 ≥ `NCCL_COLLNET_NODE_THRESHOLD`（默认 2）。

### 2. 连接何时建立：lazy connect

2.28 默认 `comm->runtimeConn = cuMemSupport && NCCL_RUNTIME_CONNECT(默认 1)`。开着时，`initTransportsRank` 只对每个 channel 调 `setupChannel`（分配 channel 结构、填 ring 的 userRanks），**不**建 ring/tree 连接；`ncclNvlsSetup`、`ncclCollNetSetup` 也只做资源准备。真正的 `ncclTransportRingConnect` / `ncclTransportTreeConnect` / `ncclNvlsTreeConnect` / `ncclTransportPatConnect`（`src/transport/generic.cc`）在**第一次有集合操作用到该算法时**才调——`ncclPrepareTasks` 发现 `comm->initAlgoChannels[algo] == false` 就标记 `algoNeedConnect`，`groupLaunch` 在 `ncclCollPreconnect` 里补建。所以日志里 `Connected all rings, use ring PXN 0 GDR 1` 和 `Connected all trees` 出现在 `Init COMPLETE` 之后、第一次 all_reduce 之前，是正常的；第一次 all_reduce 比后面慢几十到几百毫秒也是正常的（这对 PyTorch 侧的 timeout 有影响，第五篇会提）。

send/recv 的连接则一直是按需的：`ncclSend/ncclRecv` 第一次碰到某个 peer 时，`p2pTaskAppend` 把 `connectSend/connectRecv[peer]` 的位图置上，`groupLaunch` 通过 `ncclP2PPreconnectFunc` → `ncclTransportP2pSetup` 建立。

### 3. channel 是什么

一个 channel（`struct ncclChannel`，`src/include/comm.h`；设备侧 `ncclDevChannel`）拥有：

- **一条 ring 与一棵 tree 的拓扑**：`ring.prev/next/userRanks`、`tree.up/down[3]`，以及 `collnetDirect`、`collnetChain`、`nvls` 结构（`src/include/device.h`）。
- **到每个 peer 的连接**：`peers[peer]->send[NCCL_MAX_CONNS]` / `recv[NCCL_MAX_CONNS]`，每个 `ncclConnector` 里是 `ncclConnInfo`：三种协议各一块 buffer（`buffs[NCCL_NUM_PROTOCOLS]`）、`head`/`tail` 两个 64 位计数器、`connFifo`（GPU 告诉 proxy 每一步多少字节）。连接是**每 channel、每 peer、每方向**独立的。
- **kernel 里的一个 block**：`ncclLaunchKernel` 的 grid 是 `countOneBits(plan->channelMask)`，每个 block 在 `ncclKernelMain` 里用 `channelMask` 算出自己的 `channelId`，把该 channel 的结构拷进 shared memory，然后只沿这条 ring/tree 工作。
- **一份 proxy 工作**：跨机时每个 channel 在 proxy 线程里有自己的 `ncclProxyArgs`，独立推进。

所以 channel 是 NCCL 的**并行度单位**：一次 all_reduce 的 S 字节被切成 nChannels 份，每份在自己的 ring 上独立跑完整个算法，互不同步。它同时也是**资源单位**：每个 channel 每个 peer 每方向的 buffer 是 Simple 4 MiB + LL 512 KiB + LL128 约 4.7 MiB（`DEFAULT_BUFFSIZE`、`DEFAULT_LL_BUFFSIZE`、`DEFAULT_LL128_BUFFSIZE`，`src/init.cc`），32 个 channel 的 ring 就是 32 × 2 方向 × 约 9 MiB ≈ 600 MiB 显存。这是 `NCCL_MAX_NCHANNELS` 除了性能之外的另一个用途。

### 4. nChannels 怎么定

三层决定：

1. **图搜索**给出 ring 的 `nChannels`（第五章，由节点内带宽 / 每 channel 速度决定，A100 NVSwitch 是 12）。
2. **`ncclTopoPostset`** 翻倍（12 → 24），Hopper 多机可能再翻倍，然后被 `NCCL_MAX_NCHANNELS`/`maxCTAs` 截断、`NCCL_MIN_NCHANNELS`/`minCTAs` 补齐。这是 `comm->nChannels`，日志 `%d coll channels, %d collnet channels, %d nvls channels, %d p2p channels, %d p2p channels per peer` 的第一个数。NVLS 的 channel 数另算（`NVLS_NCHANNELS_SM90 = 16`，`NCCL_NVLS_NCHANNELS` 覆盖）；p2p 的 channel 数是 `ncclTopoComputeP2pChannels` 按 NVLink 带宽 / 每 NIC 2 条算出后向上取 2 的幂（`NCCL_MIN_P2P_NCHANNELS`/`NCCL_MAX_P2P_NCHANNELS`）。
3. **执行期按消息大小缩减**：`topoGetAlgoInfo`（`src/enqueue.cc`）从 `comm->nChannels` 开始，`while (nBytes < nc * nt * threadThreshold) nc--`。`nt` 是该算法×协议的最大线程数（Simple 512 或 256、LL 512、LL128 640，`NCCL_NTHREADS`/`NCCL_LL128_NTHREADS` 覆盖），`threadThreshold` 是每线程至少要分到的字节数（`NCCL_LL_THREAD_THRESHOLD = 8`，LL128 8，Simple 64，ring+LL 再乘 nRanks；`NCCL_THREAD_THRESHOLDS` 覆盖）。Simple 一个 channel 至少要 512 × 64 = 32 KB 才值得开——1 MB 的消息最多用 32 个 channel，64 KB 只用 2 个。channel 数够小之后再减线程数。日志 `AllReduce: 65536 Bytes -> Algo Tree proto LL channel{Lo..Hi}={0..1}` 里的 `channel{Lo..Hi}` 就是这一步的结果。

### 5. buffer 与 NCCL_STEPS

每条连接的 Simple buffer（默认 4 MiB，`NCCL_BUFFSIZE` 覆盖）被切成 `NCCL_STEPS = 8` 个 slot，每个 512 KiB。发送方写第 `step % 8` 个 slot 然后把 `tail` 加一；接收方看到 `tail` 前进就读，读完把 `head` 加一；发送方在 `head + 8 > step` 之前不能再写。这是一个深度为 8 的流水线：任意时刻最多 8 个 slot 在飞。跨机时 proxy 线程也按同样的 `head`/`tail` 协议与 GPU 交互（第九章）。`NCCL_BUFFSIZE` 调大只对"链路带宽 × 往返延迟 > 8 × 512 KiB"的场景有意义，比如长距离 RoCE；一般不动。

### 6. 两本账：channel 数的取舍

**带宽的账**：一个 block（一个 channel）的 NVLink 吞吐是有限的——搜索给的每 channel 20 GB/s 就是经验上限，`perChMaxRingLL128Bws`/`perChMaxTreeBws` 表里也是每 channel 20–40 GB/s 量级。要把 240 GB/s 的 NVLink 或 8 张 NIC 打满，就必须有足够多的 channel。大消息时 nChannels 越大越好，直到链路饱和。

**延迟的账**：每个 channel 是一个 block，多一个 block 就多一份 shared memory 加载、多一份 `head/tail` 握手、多占一个 SM（与计算 kernel 抢）；跨机时每个 channel 是一份独立的 proxy 工作、一组独立的 QP。消息小到每 channel 分不到几十 KB 时，切分本身的固定开销超过并行收益。所以 `topoGetAlgoInfo` 按大小缩 channel。`NCCL_MIN_NCHANNELS` 抬高下限的常见后果是小消息延迟变差、以及 SM 占用增加打断计算通信重叠（第五篇）。

一个可操作的判断：如果 nccl-tests 曲线大消息端上不去而拓扑正确，看日志里 `coll channels` 是不是被 `NCCL_MAX_NCHANNELS` 或 `maxCTAs` 压低了；如果小消息延迟异常高，看 `channel{Lo..Hi}` 是不是被 `NCCL_MIN_NCHANNELS` 撑大了。


## 七、算法与协议

### 1. 七种算法

`src/init.cc` 定义了名字，顺序就是 `NCCL_ALGO_*` 的编号：

```cpp
const char* ncclAlgoStr[NCCL_NUM_ALGORITHMS] = { "Tree", "Ring", "CollNetDirect", "CollNetChain", "NVLS", "NVLSTree", "PAT" };
const char* ncclProtoStr[NCCL_NUM_PROTOCOLS] = { "LL", "LL128", "Simple" };
```

```text
Tree           节点间 double binary tree + 节点内链；all_reduce 先 reduce 到根再 broadcast 下来（runTreeUpDown / runTreeSplit）
               延迟 O(log nNodes)，带宽比 ring 低（模型里 ×0.92，Hopper 上再 ×0.85）
Ring           第一篇的 ring：reduce_scatter + all_gather 各 n-1 步（runRing）；带宽最优，延迟 O(n)
CollNetDirect  节点内每个 GPU 直接和"head"交换，head 通过 collnet 插件（IB SHARP）在交换机里归约；最多 8 GPU/节点
CollNetChain   节点内是一条链，链头走 collnet
NVLS           NVLink SHARP：Hopper 起，NVSwitch 内做归约；数据写到多播地址，交换机归约后多播回来
               约 2.18 引入；单机直接用，多机与 collnet 组合（NVLS 节点内 + SHARP 节点间）
NVLSTree       节点内 NVLS + 节点间 tree；多机且没有 SHARP 时的 NVLS 路径
PAT            Parallel Aggregated Trees（约 2.23 引入）：只用于 all_gather / reduce_scatter，只在每节点 1 GPU 时启用
               （ncclPatEnable: nNodes == nRanks），log 步数的 Bruck 类算法，解决大规模下 ring 的 O(n) 延迟
```

哪些算法能跑哪些集合操作在 `ncclTopoTuneModel` 里写死：broadcast/reduce 只有 Ring；all_gather/reduce_scatter 只有 Ring、PAT、NVLS、CollNetDirect；all_reduce 除 PAT 外都可以。总纲提到的"NVLS、PAT 的加入版本"以你手上版本的 `ncclAlgoStr` 为准——这一行 grep 一下就知道有几种。

### 2. 三种协议的机制

协议决定"一个 GPU 怎么知道对端已经写完了一块数据"。三种协议是三种同步机制，代价不同。

**Simple**（`src/device/prims_simple.h`）。数据和控制分离：发送方把数据写进对端 buffer 的一个 slot（512 KiB），然后**内存屏障**（`fence_acq_rel_sys`，系统范围的 acquire-release fence），再把 `tail` 计数器加一（`st_relaxed_sys_global`）。接收方轮询 `tail`（`ld_volatile_global`），看到前进了就知道整个 slot 的数据已经可见。一次同步覆盖 512 KiB，摊销后同步开销接近零，**带宽效率 ≈ 100%**。代价是那个 fence：它要等本 GPU 之前所有的写对系统可见（跨 PCIe/NVLink/网卡），在 NVLink 上是微秒级，再加上接收方轮询到 flag 的往返，是三种协议里**延迟最高**的。

**LL**（Low Latency，`src/device/prims_ll.h`）。把 flag 和数据打包在一起：

{% raw %}
```cpp
union ncclLLFifoLine {
  struct { uint32_t data1; uint32_t flag1; uint32_t data2; uint32_t flag2; };  // 8 字节数据 + 8 字节 flag
  uint64_t v[2];
  int4 i4;
};
// 发送：一条 16 字节的 volatile store，数据和 flag 原子地一起落地
asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(&dst->i4), "r"((uint32_t)val), "r"(flag), "r"((uint32_t)(val >> 32)), "r"(flag) : "memory");
// 接收：反复 16 字节 volatile load，直到两个 flag 都等于期望值
do {
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(data1), "=r"(flag1), "=r"(data2), "=r"(flag2) : "l"(&src->i4) : "memory");
} while ((flag1 != flag) || (flag2 != flag));
```
{% endraw %}

16 字节的 store 在 GPU 上是原子的，所以接收方看到 flag 对了，同一行里的 8 字节数据一定也对了——**不需要 fence**，也不需要单独的 `tail` 往返。flag 就是 step 编号（`NCCL_LL_FLAG(step+1)`），单调递增，接收方不必清零。代价显而易见：每 16 字节只有 8 字节是数据，**带宽效率 50%**。它是**延迟最低**的协议，用于小消息。`ncclLLFifoLine` 的注释还解释了 flag 为什么放在数据**后面**：网络可能不是原子地送达 16 字节，但 IB/RDMA 保证 8 字节原子、socket 保证顺序，flag 在后就不会"flag 到了数据没到"。

**LL128**（`src/device/prims_ll128.h`）。LL 的思路，但把粒度放大到 128 字节：一个 warp 的 8 个线程各持 16 字节组成一个 128 字节的 line，其中 **120 字节数据 + 8 字节 flag**（`NCCL_LL128_LINESIZE 128`、`NCCL_LL128_DATAELEMS 15`，即 15 个 8 字节数据 + 1 个 8 字节 flag；`flagThread = (tid % 8) == 7`，每 8 个线程里第 8 个的高 8 字节存 flag）。接收方只检查 flag 那 8 字节。这依赖一个硬件保证：**128 字节的写在 NVLink 上按顺序、整体可见**——flag 可见时同一 line 的 120 字节一定可见。PCIe 不给这个保证，所以 LL128 只在 NVLink 路径上启用（下一小节）；发送到网络时 proxy 在 sysmem 里要逐 line 检查 flag（`sendProxyProgress` 里对 LL128 的特殊处理）。**带宽效率 120/128 = 93.75%**（调优模型用 0.92），延迟介于 LL 和 Simple 之间。它需要 `__threadfence`（Hopper 上 `__threadfence_system`）只在跨 step 边界处做一次（`postSend`），比 Simple 每 slot 一次 fence 便宜的原因是它的 step 更小、且数据自身带 flag 不需要接收方额外一次 `tail` 往返。

### 3. 效率与延迟排序

```text
协议      有效载荷      同步方式                         延迟   带宽    典型消息区间（非实测，以 tuning 表为准）
LL        8/16 = 50%    16 B 原子 store 自带 flag，无 fence   最低   最低    几十 KB 以下
LL128     120/128 ≈ 94% 128 B line 自带 flag，NVLink 顺序保证  中     接近    几十 KB ～ 几十 MB（NVLink 节点内常是默认）
Simple    ≈ 100%        512 KiB slot + fence + head/tail     最高   最高    大消息；PCIe / 网络路径上的默认
```

调优表里的硬件延迟常数（`hwLatencies`，微秒，NVLink 上 Ring 的 LL/LL128/Simple）是 0.6 / 1.9 / 3.4；PCIe 上 1.0 / 2.5 / 5.7；网络上 2.7 / 4.0 / 14.0——这是"每一步"的延迟，ring 要乘 $$2(n-1)$$。

### 4. 哪些组合会被禁用

`ncclTopoTuneModel` 最后有一段 enable/disable 逻辑，决定 `comm->bandwidths[coll][algo][proto]` 是否被清零（清零 = 永不选）：

- `NCCL_PROTO` / `NCCL_ALGO` 环境变量：`parseList` 支持 `ring,tree`、`^LL128`（排除）、`allreduce:tree;broadcast:ring`（按集合操作分别指定）。rank 0 会打 `NCCL_ALGO set by environment to …` 和一张 `Enabled NCCL Func/Proto/Algo Matrix`。
- **LL128 默认"条件启用"**（`protoEnable = 2`）：要求节点内路径类型 ≤ `NVB`（必须是 NVLink）、节点间路径类型 ≤ `PXB`（Hopper 起放宽到 `PXN`/`P2C`，`NCCL_LL128_C2C`）、所有 GPU 计算能力相同且 ≥ 7.0。**纯 PCIe 机器上 LL128 是关的**。
- 单机禁 NVLSTree；没有 collnet 时禁 CollNetDirect/CollNetChain 和多机的 NVLS；没有 NVSwitch 禁 CollNetDirect；NVLS/NVLSTree 只有 Simple；CollNet 只有 Simple；PAT 只有 Simple。

所以"NCCL 为什么不用 LL128"的答案几乎总是路径类型：日志里 `type PIX/PIX` 或 `type PHB/…` 就是原因。


## 八、调优模型：NCCL 如何估算时间

### 1. ncclTopoTuneModel：给每个组合填 lat 与 bw

这个函数（`src/graph/tuning.cc`）在初始化最后、连接建好之后调用一次，输入是四张图的 `nChannels/bwIntra/bwInter/typeIntra/typeInter`、`nRanks`、`nNodes`、计算能力档位（Volta/Ampere/Hopper/Blackwell），输出两张三维表。它的常数全在 `ncclTunerConstantsDefaults`，这里摘要：

```text
baseLatencies[algo][proto]  (µs)      Tree {6.8, 14.0, 8.4}  Ring {6.6, 14.0, 8.4}  CollNet* {0}  NVLS* {0}  PAT {8.0}
hwLatencies[hw][algo][proto] (µs)
   NVLINK   Tree {0.6, 1.25, 4.0}  Ring {0.6, 1.9, 3.4}  CollNetDirect {-,-,3.7}  CollNetChain {-,-,2.8}  NVLS {-,-,25}  NVLSTree {-,-,25}  PAT {-,-,4.0}
   PCI      Tree {1.0, 1.9, 4.0}   Ring {1.0, 2.5, 5.7}  同上 3.7 / 2.8            NVLS 不可用                              PAT 4.0
   NET      Tree {5.0, 8.5, 14}    Ring {2.7, 4.0, 14.0} CollNetDirect 31          CollNetChain 30   NVLS 18   NVLSTree 20.9  PAT 14
llMaxBws[cpu/gpu][nNodes 档]  (GB/s)  Volta/Intel {39, 39, 20.4}  Ampere/AMD {87.7, 22.5, 19}  Hopper {141, 45, 35}  Blackwell 翻倍
perChMaxRingLL128Bws          Hopper {36.7, 36.7, 36.7}    perChMaxTreeBws  Hopper {38.7, 41.4, 36.0}   （每 channel 上限）
nvlsEfficiency                Hopper 0.85  Blackwell 0.74
```

对每个 (coll, algo, proto)，带宽这样算（`busBw` 是链路承载的流量，最后换成算法带宽）：

```text
bw      = nNodes <= 2 ? graph.bwIntra : graph.bwInter          （多机时瓶颈在网络）
busBw   = graph.nChannels × bw
Ring+LL     busBw = min(llMaxBw, busBw × 0.5)                   LL 50%，且有绝对上限
Ring+LL128  busBw = min(busBw × 0.92, nChannels × perChMaxRingLL128Bw)
Tree(all_reduce)  busBw = min(busBw × 0.92, nChannels × perChMaxTreeBw)；Tree+LL ÷3.8；Tree+LL128 ×7/9（单机）或 ×120/128
Tree 且 pattern==TREE（Hopper）  ×0.85
NVLS    intraBw = bwIntra × nvlsEfficiency × (nCh-1)/nCh × 2（all_reduce 两阶段流水）；bw = min(intraBw, interBw)
PAT     ×0.75
算法带宽 = busBw × ratio：Ring/NVLS/NVLSTree 乘 nRanks/nsteps（all_reduce 是 n/(2(n-1))，即第一篇 busbw→algbw 的反向）；Tree/CollNet 乘 0.5
```

延迟这样算（`intraLat` 取 NVLINK 或 PCI 表，`interLat` 取 NET 表再加图的 `latencyInter`，Simple 再加一次 flush 延迟）：

```text
Ring   lat = base + (nsteps - nInterSteps) × intraLat + nInterSteps × interLat
           nsteps = 2(nRanks-1)，nInterSteps = 2(nNodes-1)；多机时 intraLat 至少是 netOverhead（Intel 1 µs、AMD 2 µs，Simple ×3）
Tree   lat = base + 2 × ((nRanks/nNodes - 1) × intraLat + log2(nNodes) × interLat)
CollNetDirect  lat = base + 2 × (min(1, ppn-1) × intraLat + (ppn-1) × 0.4) + interLat
NVLS   lat = intraLat (25)，多机再 + interLat
NVLSTree  lat = base + intraLat + 2 × log2(nNodes) × interLat
PAT    lat = base + log2(nNodes) × interLat/3.5 + nRanks × 2.8
```

rank 0 在 `NCCL_DEBUG_SUBSYS=TUNING` 下把整张表打出来，格式 `%8.1f/%6.1f` 即 `延迟/带宽`：

```text
  Algorithm   |                    Tree                   |                    Ring                   |             CollNetDirect          |
  Protocol    |       LL |    LL128 |   Simple |       LL |    LL128 |   Simple |       LL |    LL128 |   Simple |
 Max NThreads |      512 |      640 |      512 |      512 |      640 |      512 |        0 |        0 |      640 |
    Broadcast |     ...（Broadcast/Reduce 只有 Ring 非零；AllGather/ReduceScatter 略）
    AllReduce |    15.2/ 43.9 |    31.5/186.7 |    64.4/220.8 |    22.4/ 50.1 |    40.6/252.3 |    56.0/274.3 |     0.0/  0.0 |     0.0/  0.0 |     0.0/  0.0 |
```

（**示意**，AllReduce 一行是按上面的公式对"8 卡 A100 单机、ring 图 24 channel × 20 GB/s"算出来的：Ring+Simple busBw 480 → algbw 480 × 8/14 = 274.3，lat 8.4 + 14 × 3.4 = 56.0；Tree+Simple busBw min(480 × 0.92, 24 × 24) = 441.6 → × 0.5 = 220.8，lat 8.4 + 2 × 7 × 4.0 = 64.4；Ring+LL 受 llMaxBw 87.7 限制 → 87.7 × 8/14 = 50.1。你的机器上这张表的数字会不同，但格式一样，`nccl_log_reader.py` 会直接解析它。）

### 2. ncclTopoGetAlgoTime 与执行期的选择

执行期 `getAlgoInfo`（`src/enqueue.cc`）对每个可用的 (algo, proto) 调 `ncclTopoGetAlgoTime`：

```cpp
ncclResult_t ncclTopoGetAlgoTime(struct ncclComm* comm, int coll, int algorithm, int protocol, size_t nBytes, int numPipeOps, float* time) {
  float bw = comm->bandwidths[coll][algorithm][protocol];
  float lat = comm->latencies[coll][algorithm][protocol];
  if (bw == 0) { *time = -1.0; return ncclSuccess; }                       // 被禁用
  int logSize = log2i(nBytes>>6);
  if (algorithm == NCCL_ALGO_TREE && coll == ncclFuncAllReduce && logSize >= 0 && logSize < 23) bw *= treeCorrectionFactor[protocol][logSize];
  if (algorithm == NCCL_ALGO_RING && protocol == NCCL_PROTO_SIMPLE && comm->nNodes > 1
      && coll == ncclFuncAllReduce && nBytes/(comm->nChannels*comm->nRanks) >= 64) {
    lat *= comm->minCompCap < 80 ? 1.9 : 1.4; // Plateau effect of ring
  }
  int latCount = algorithm == NCCL_ALGO_RING ? numPipeOps : DIVUP(numPipeOps, NCCL_MAX_DEV_WORK_BATCH_COLLS);
  *time = lat * latCount + nBytes / (1000 * bw);                            // µs：bw 是 GB/s，nBytes/1000/bw
  return ncclSuccess;
}
```

`treeCorrectionFactor` 是一张 23 项的经验表（64 B 到 256 MB，按 2 的幂），中等大小时把 tree 的带宽打 0.4～0.7 的折——源码注释承认"Trees are not perfectly sticking to the model for medium sizes"。`numPipeOps` 是同一 group 里聚合的操作数：ring 每个操作都付一次延迟，tree 因为流水线每 `NCCL_MAX_DEV_WORK_BATCH_COLLS` 个才付一次。

`topoGetAlgoInfo` 取时间最小的组合，`NCCL_ALGO_PROTO_IGNORE`（-1）的跳过。如果一个都没有（比如 `NCCL_ALGO=NVLS` 但机器不支持），报 `Error : no algorithm/protocol available for function … NCCL_ALGO was set to …`——这是错误配置最直接的表现。有 tuner 插件时先让插件改 `collCostTable` 再选。

### 3. 回答核心问题

下面的手算是**简化模型**：只用上一节的基础公式，忽略 `treeCorrectionFactor`、多机 Ring+Simple 的 plateau 修正以及与消息大小相关的协议修正，目的是看清三台机器上各组合的相对位置；真实的调优表以 `NCCL_DEBUG_SUBSYS=TUNING` 打印出来的为准。

三台机器，同一个 8 卡（或 8 卡×N 节点）all_reduce，用上面的公式走一遍。所有数字都是模型输出（非实测），图的参数写明假设。

**机器一：8 卡 H100 + NVSwitch（单机）。** 可用算法：Ring、Tree、NVLS（有 NVSwitch、Hopper、`nvlsSupport`）。假设 ring 图 nChannels = 16、bwIntra = 24（H100 常见量级）、NVLS 图 nChannels = 8（每 GPU 一个 head）、bwIntra = 60（`sm90SpeedArrayIntra` 首项）。

```text
Ring+Simple   busBw = 16×24 = 384 → algbw = 384×8/14 = 219 GB/s；lat = 8.4 + 14×3.4 = 56.0 µs
Ring+LL128    busBw = min(384×0.92, 16×36.7) = 353 → 202 GB/s；lat = 14 + 14×1.9 = 40.6 µs
Ring+LL       busBw = min(141, 384×0.5) = 141 → 80.6 GB/s；lat = 14 + 14×0.6 = 22.4 µs
NVLS+Simple   intraBw = 60×0.85×7/8×2 = 89.3；busBw = 8×89.3 = 714 → algbw = 714×8/14 = 408 GB/s；lat = 25 µs
```

带宽的账 NVLS 大胜：ring 里每个 GPU 要经 NVLink 收发 $$\frac{2(n-1)}{n}S = 1.75S$$，而且要串行走 $$2(n-1)$$ 步；NVLS 把归约交给 NVSwitch，每个 GPU 只需把自己的份交给多播地址、再收一次结果，NVLink 上每卡的字节量仍约 $$2S$$，但一步完成、没有逐跳等待，链路能持续跑满，模型用 `nvlsEfficiency × 2` 表达这一点。延迟的账 NVLS 也不差：25 µs 是一次多播归约的固定开销，而 Ring+Simple 的 14 步累计 56 µs。所以只有 Ring+LL（22.4 µs）在极小消息上能赢：解 $$22.4 + S/80.6\text{e}3 < 25 + S/408\text{e}3$$ 得 $$S \lesssim 260\ \text{KB}$$。**模型结论：几百 KB 以下 Ring+LL（或 Tree+LL），以上全部 NVLS+Simple。** 这就是总纲说的"NVSwitch 机器上选了 NVLS + Simple"。

**机器二："PCIe 机器"。** 要分两种情况，因为 LL128 的启用条件是节点内路径 ≤ NVB：

- 如果是 **NVLink 桥接但没有 NVSwitch**（比如 4 卡两两 NVLink 桥、或 8 卡无 NVSwitch 的 HGX 老拓扑），`typeIntra = NVL/NVB`，NVLS 不可用（没有 NVS 节点），LL128 可用。假设 ring 图 nChannels = 4、bwIntra = 24（部分 GPU 对之间要经 NVB 转一跳，带宽被分摊）：Ring+Simple 96×8/14 = 55 GB/s，lat 56 µs；Ring+LL128 min(96×0.92, 4×20)=80 → 46 GB/s，lat 40.6 µs。Ring+LL128 在 $$40.6 + S/46\text{e}3 < 56 + S/55\text{e}3$$ 即 $$S \lesssim 4.3\ \text{MB}$$ 时赢——DDP 的 25 MB bucket 会落到 Simple，几 MB 以下的 TP all_reduce 落到 LL128。**这是"PCIe 机器上 Ring + LL128"的真实含义：有 NVLink 但没有 NVSwitch。**
- 如果是**纯 PCIe**（GPU 间只有 PCIe switch 或 CPU），`typeIntra ≥ PIX`，**LL128 被禁用**，`intraHw = PCI`，NVLS 不可用。Ring+Simple（lat 8.4 + 14×5.7 = 88 µs）对大消息，Ring/Tree+LL 对小消息。此时 Tree 的每步延迟（PCI 表 Tree Simple 4.0 < Ring 5.7）和更少的步数让 Tree 在中等消息上更有竞争力。

**机器三：32 节点 × 8 卡 = 256 rank，节点内 NVSwitch，节点间 IB。** 多机时 `bw = bwInter`，假设 ring 图 nChannels = 8（搜出 4 条 × 翻倍）、bwInter = 12（每 NIC 分到的每 channel 带宽），Hopper。

```text
Ring+Simple   busBw = 96 → algbw = 96 × 256/510 = 48.2 GB/s
              lat = 8.4 + (510-62)×3.4 + 62×(14+0) = 8.4 + 1523 + 868 = 2400 µs，大消息再 ×1.4（plateau）= 3360 µs
Tree+Simple   busBw = min(96×0.92, 8×36.0) = 88.3，×0.85（Hopper TREE pattern）= 75.1 → algbw = 37.5 GB/s
              lat = 8.4 + 2×(7×4.0 + 5×14) = 204 µs
Tree+LL       lat = 6.8 + 2×(7×0.6 + 5×5.0) = 65 µs；bw 小
Ring+LL       lat = 14 + 448×0.6 + 62×2.7 = 450 µs
```

延迟的账差了一个数量级：ring 要走 510 步，其中 62 步过网络；tree 只要节点内 7 步 + 节点间 $$\log_2 32 = 5$$ 步、再乘 2。带宽的账 ring 好 30%。交叉点：$$204 + S/37.5\text{e}3 = 3360 + S/48.2\text{e}3$$ 得 $$S \approx 533\ \text{MB}$$（还没算 `treeCorrectionFactor` 在 100 MB 附近的 0.8 折，算上后交叉点略前移）。**25 MB 的 DDP bucket、几十 MB 的梯度分片都在 Tree 的区间，只有整 GB 的消息 Ring 才回来。** 这就是"跨 32 台机器时选了 Tree"。

三个决定用的是同一个公式，不同的只是输入：机器一的输入里有 NVLS 图且 NVLink 延迟低；机器二的输入里 `typeIntra` 决定 LL128 开不开、有没有 NVS 决定 NVLS 有没有；机器三的输入里 `nNodes = 32` 让 ring 的步数项爆掉。

### 4. 强行 NCCL_ALGO=Ring 的代价

把上面三台机器的 Ring 与自动选择相减：

```text
机器一（H100 NVSwitch）  1 GB   NVLS+Simple 25 + 2451 = 2476 µs   vs  Ring+Simple 56 + 4566 = 4622 µs     慢 1.9×（带宽的账：NVLink 流量 1.75S vs ~2S/n）
                         64 KB  Ring+LL 22.4 + 0.8 = 23 µs          vs  Ring+LL 同上                            无代价（本来就是 Ring）
机器三（32 节点）        25 MB  Tree+Simple 204 + 667 = 871 µs     vs  Ring+Simple 3360 + 519 = 3879 µs      慢 4.5×（延迟的账：510 步）
                         64 KB  Tree+LL 65 + ~2 = 67 µs             vs  Ring+LL 450 + ~1 = 451 µs             慢 6.7×
                         1 GB   Ring 3360 + 20747 = 24107 µs        vs  Tree 204 + 26667 = 26871 µs           Ring 反而快 10%
```

结论：`NCCL_ALGO=Ring` 在单机 NVSwitch 上损失的是带宽（放弃了交换机内归约），在大规模多机上损失的是延迟（$$O(n)$$ 步），只有"大规模 + 整 GB 消息"这一种情形它是对的。它常被当作"稳定性开关"使用（排除 NVLS/CollNet 的兼容性问题、或让不同 rank 数下的浮点归约顺序一致），这时要知道付的是哪本账。同理，`NCCL_PROTO=Simple` 是在放弃小消息的延迟，`NCCL_PROTO=LL` 是在放弃一半带宽。

### 5. NCCL_ALGO / NCCL_PROTO 的语法与 tuner 插件

`parseList`（`src/graph/tuning.cc`）的语法：元素用逗号分隔，`^` 前缀表示排除，`函数名:` 前缀限定集合操作，分号分隔多组，第一组可以没有前缀。源码注释里的例子：

```text
NCCL_ALGO="ring,collnetdirect;allreduce:tree,collnetdirect;broadcast:ring"
NCCL_PROTO="LL,Simple;allreduce:^LL"
NCCL_PROTO="^LL128;allreduce:LL128"
```

名字不区分大小写，必须是 `ncclAlgoStr`/`ncclProtoStr` 里的（`Tree Ring CollNetDirect CollNetChain NVLS NVLSTree PAT`；`LL LL128 Simple`），写错直接 `Unrecognized element token`。

不想改环境变量而想用自己的规则，用 **tuner 插件**：`NCCL_TUNER_PLUGIN=<so 名>`（`src/plugin/tuner.cc`），接口在 `src/include/plugin/tuner/tuner_v5.h`，只有三个函数：

```c
// ext-tuner/example/nccl/tuner_v5.h（节选）
ncclResult_t (*init)(void** ctx, uint64_t commId, size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction,
                     ncclNvlDomainInfo_v5_t* nvlDomainInfo, ncclTunerConstants_v5_t* constants);
ncclResult_t (*getCollInfo)(void* context, ncclFunc_t collType, size_t nBytes,
                            int numPipeOps, float** collCostTable, int numAlgo, int numProto,
                            int regBuff, int* nChannels);
ncclResult_t (*finalize)(void* context);
```

`init` 拿到的 `constants` 就是上面那张 `ncclTunerConstantsDefaults`，插件可以改它（比如把某个 `hwLatencies` 改成你实测的）；`getCollInfo` 拿到 NCCL 已经算好的 `collCostTable[algo][proto]`（就是 `ncclTopoGetAlgoTime` 的输出），可以改任何一格、或直接把不想要的置为 `NCCL_ALGO_PROTO_IGNORE`，还可以返回 `nChannels`。`ext-tuner/example/` 是一个读 CSV 配置的完整实现，`nccl_tuner.conf` 的每一行是 `collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff`——按消息大小区间指定算法协议。它比环境变量精细得多，也是第六篇调优时"先测出曲线、再把交叉点写进配置"的落地方式。


## 九、执行期：enqueue、kernel 与 proxy

### 1. 从 ncclAllReduce 到任务队列

`ncclAllReduce`（`src/collectives.cc`）填一个 `ncclInfo` 然后调 `ncclEnqueueCheck`（`src/enqueue.cc`）。后者的骨架：

```cpp
NCCLCHECK(ncclGroupStartInternal());        // 隐式 group：每个集合调用都在一个 group 里
NCCLCHECKGOTO(ncclCommEnsureReady(info->comm), ret, fail);
NCCLCHECKGOTO(ArgsCheck(info), ret, fail);   // 参数检查：count、datatype、op、指针
INFO(NCCL_COLL,"%s: opCount %lx sendbuff %p recvbuff %p count %zu datatype %d op %d root %d comm %p [nranks=%d] stream %p", ...);
NCCLCHECKGOTO(taskAppend(info->comm, info), ret, fail);   // 只是入队
// ...
NCCLCHECK(ncclGroupEndInternal());          // 如果这是最外层 group，这里触发真正的调度与启动
```

`taskAppend` 把集合操作变成一个 `ncclTaskColl` 放进 `comm->planner.collSorter`（按大小排序），把 send/recv 变成 `ncclTaskP2p` 挂到对应 peer 上。**此时没有任何 GPU 工作发生。**如果用户没有显式 `ncclGroupStart`，`ncclGroupDepth` 从 1 减到 0，`ncclGroupEndInternal` 立刻执行 `groupLaunch`。如果在显式 group 里，就等用户的 `ncclGroupEnd`。

### 2. 聚合、选算法、分 channel

`groupLaunch`（`src/group.cc`）依次：为需要预连接的 p2p peer 建连接（异步线程）→ 对每个 comm 调 `ncclPrepareTasks` → 需要时 `ncclCollPreconnect` 补建该算法的连接 → `ncclTasksRegAndEnqueue` → `doLaunches`。

`ncclPrepareTasks`（`src/enqueue.cc`）：

1. 把任务按 (函数, 归约操作, 数据类型) 分桶。同一桶里**大小相差不到 4 倍的任务聚合**成一个 `agg` 去查调优表（`while (aggEnd->trafficBytes < 4*aggBeg->trafficBytes)`），这样一个 group 里 10 个 1 MB 的 all_reduce 按"10 个 1 MB 流水"而不是"1 个 10 MB"或"10 个独立的"来估时间——`numPipeOps` 就是这里传进去的。
2. `getAlgoInfo` → `topoGetAlgoInfo` 选出算法、协议、`nMaxChannels`、`nWarps`（第八章第 2 节、第六章第 4 节）。
3. 第一次用到某算法时标记需要连接。

{% raw %}`scheduleCollTasksToPlan` 再把任务放进 `ncclKernelPlan`：每个任务占 `channelLo..channelHi` 一段连续 channel，`calcCollChunking` 算每个 channel 每步搬多少（chunk），需要网络的生成 `ncclProxyOp`。rank 0 打 `AllReduce: %ld Bytes -> Algo %s proto %s channel{Lo..Hi}={%d..%d}`。一个 plan 就是一次 kernel 启动能装下的工作（受 `NCCL_WORK_ARGS_BYTES` 等限制），装不下就多个 plan、多次启动。{% endraw %}

### 3. 一个 kernel、nChannels 个 block

`ncclLaunchKernel`（`src/enqueue.cc`）：

```cpp
int nChannels = countOneBits(plan->channelMask);
dim3 grid = {(unsigned)nChannels, 1, 1};
dim3 block = {(unsigned)plan->threadPerBlock, 1, 1};
// ... Hopper 起可设 cluster（NCCL_CGA_CLUSTER_SIZE）与 MEM_SYNC_DOMAIN
CUCHECKGOTO(cuLaunchKernelEx(&launchConfig, fn, nullptr, extra), ret, do_return);
```

kernel 名形如 `ncclDevKernel_AllReduce_Sum_f32_RING_LL128`（`src/device/generate.py` 生成，profiler 里看到的就是它）。`ncclKernelMain`（`src/device/common.h`）里每个 block：用 `channelMask` 算自己是第几个置位 → 得到 `channelId`；warp 0 把 `ncclKernelComm` 拷进 shared memory、warp 1 拷本 channel 的 `ncclDevChannel`、其余 warp 加载工作批（`loadWorkBatchToShmem`）；然后按 `funcId` 跳到 `RunWorkBatch<coll, ty, redop, algo, proto>::run()`，批里还有下一批就继续，直到 `nextBatchIx == -1`。**同一个 kernel 可以顺序执行多个集合操作**（一个 group 里的），这是 group 减少 kernel 启动次数的机制。

### 4. 设备侧原语

`src/device/all_reduce.h` 的 `runRing` 就是第一篇的 ring 算法逐字翻译：

```cpp
Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0> prims(tid, nthreads, &ring->prev, &ring->next, work->sendbuff, work->recvbuff, ...);
for (elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
  prims.directSend(offset, offset, nelem);                            // step 0：把自己的一块发给 next
  for (int j = 2; j < nranks; ++j) prims.directRecvReduceDirectSend(offset, offset, nelem);   // k-2 步：收、归约、发
  prims.directRecvReduceCopyDirectSend(offset, offset, nelem, /*postOp=*/true);               // 第 k-1 步：得到最终结果，写回并发出
  for (int j = 1; j < nranks - 1; ++j) prims.directRecvCopyDirectSend(offset, offset, nelem);  // k-2 步：all_gather 转发
  prims.directRecv(offset, nelem);                                    // 最后一块
}
```

`Primitives` 模板按 `Proto` 分派到 `prims_simple.h` / `prims_ll.h` / `prims_ll128.h`，`Fan` 说明收几路发几路（ring 是 1 收 1 发，tree 是最多 3 收 1 发）。每个原语内部是"等对端就绪（`waitPeer`/`readLL`）→ 从 recv buffer 读、与本地数据归约、写到 send buffer 或用户 buffer（`reduceCopy`）→ 通知对端（`postPeer`/flag）"。`direct` 前缀表示可能直接读写对端（或用户）buffer 而不经中转（`NCCL_P2P_READ`/`NCCL_P2P_WRITE`、`src/register/` 的注册 buffer）。

Tree 的 all_reduce（`runTreeUpDown` / `runTreeSplit`）是两段：先沿 `down[]` 收、归约、往 `up` 发（reduce），根拿到全量后再沿 `down[]` 广播。`nthreads` 在 tree 上被分成两半分别跑上行和下行（`runTreeSplit`），这是为什么 Tree 的线程数总是 `NCCL_MAX_NTHREADS`。

### 5. proxy 线程：GPU 不能驱动网卡

GPU kernel 能直接读写对端显存（P2P）、能读写主机内存（SHM），但**不能提交 RDMA 请求**——verbs 的 `ibv_post_send` 是 CPU 侧的库调用，需要 CPU 写 QP 的 doorbell（第三篇）。所以跨机时每张 GPU 需要一个 CPU 线程替它做这件事，这就是 proxy。

`ncclProxyCreate`（`src/proxy.cc`）在初始化时起 `ncclProxyService` 线程（处理连接建立等控制消息，本地 rank 通过 socket/UDS 与它通话），进度线程 `ncclProxyProgress` 在第一次需要时由 service 线程创建。进度线程的主循环：

```cpp
do {
  int idle = 1;
  ncclResult_t ret = progressOps(proxyState, state, state->active, &idle);   // 对每个活跃的 ProxyArgs 调其 progress 函数
  // ...
  if (idle || !state->active || (++proxyOpAppendCounter == ncclParamProgressAppendOpFreq())) {
    ret = ncclProxyGetPostedOps(proxyState, &added);                          // 从 GPU 侧 enqueue 时投递的 proxyOp 队列取新工作
    if (added == 0) sched_yield();
  }
} while ((state->stop == 0 || (state->stop == 1 && state->active)) && __atomic_load_n(proxyState->abortFlag, __ATOMIC_ACQUIRE) == 0);
```

NET transport 的 `sendProxyProgress`（`src/transport/net.cc`）对每个 channel 的每个 sub 维护三个游标 `posted`（已告诉 GPU 可以写的 slot）、`transmitted`（已交给网卡的）、`done`（网卡已完成的），每一轮：

1. `posted < done + NCCL_STEPS` 时向前推 `posted`，通过 `sendMem->head` 告诉 GPU 有新 slot 可写；
2. 看 `recvMem->tail`（GPU 写完一个 slot 后推进），如果 `connFifo[slot].size != -1` 说明 GPU 填好了这个 slot（LL/LL128 在 sysmem 时还要逐 line 验 flag），调 `ncclNet->isend`；
3. 对 `done < transmitted` 的请求调 `ncclNet->test`，完成了就 `sendHead` 前进，GPU 可以复用该 slot。

接收侧 `recvProxyProgress` 对称：先 `irecv` 把 slot 交给网卡，完成后（GDR 时可能还要 `iflush` 确保数据对 GPU 可见）推 `recvMem->tail` 让 GPU 去读。所有 GPU–proxy 之间的同步都靠 `head`/`tail` 这两个 64 位计数器，位于 GPU 能读写的内存里（`gdcSync` 开着时用 GDRCopy 让 CPU 直接写显存里的计数器，省一次 PCIe 往返——第三篇讲的 GDRCopy 用途）。

为什么它是 hang 和性能问题的常见源头：

- 它是**一个 CPU 线程**在为整张 GPU 的所有 channel 轮询。被抢占（CPU 超卖、绑核不当、`NCCL_PROXY_CPUSET` 没设而落在忙核上）就直接表现为小消息延迟抖动和大消息带宽掉一截。`[Proxy Progress] Device %d CPU core %d` 这行日志告诉你它跑在哪个核。
- GPU 侧 kernel 在 `waitPeer` 里自旋等 `head`/`tail`，proxy 不推进它就永远等。所以"网络断了 / 对端 rank 死了 / QP 出错"在 GPU 上的表现是 kernel 自旋不退出，profiler 里是一条无限长的 `ncclDevKernel_*`——这是第六篇 hang 排查里 `cuda-gdb` 看到"kernel 在自旋"的来源。
- 出错时 proxy 把 `asyncResult` 置位、打 `[Proxy Progress] … [Progress Thread]`，但 kernel 侧要靠 `abortFlag` 才会退出——`ncclCommAbort` 做的就是置这个 flag。
- `NCCL_PROXY_DUMP_SIGNAL` 可以指定一个信号，收到时 `ncclDumpProxyState` 把所有活跃 proxyOp 的游标打出来——每个 channel 停在 `posted/transmitted/done` 的哪一步，一看便知是 GPU 没写、网卡没发、还是对端没收。

节点内 P2P/SHM 不需要 proxy 推进数据，但 SHM 用 copy engine 的模式和 P2P 的某些注册路径也会用到 proxy 的控制面。

### 6. group 语义与 send/recv 配对

`ncclGroupStart/End` 的语义是"这之间的所有调用作为一个整体调度"。三个后果：

1. **聚合**：同一 comm 的多个集合操作进同一个 plan，一次 kernel 启动（第 3 节）；多个 comm（一个进程管多张 GPU）的操作并行启动。DDP 把多个 bucket 的 all_reduce 放进一个 group、多 GPU 单进程的场景都靠这个。
2. **调优时的流水线折扣**：`numPipeOps` > 1 时 tree 的延迟只算一次。
3. **send/recv 必须成对放在 group 里**。`ncclSend` 和 `ncclRecv` 各自只是入队；如果 rank A 调 `ncclSend(to B)` 时不在 group 里，`ncclGroupEndInternal` 立刻启动一个只有 send 的 kernel，这个 kernel 会在 `waitPeer` 里等 B 的接收 buffer 就绪；而 B 要等 A 的 send kernel 返回后才会调 `ncclRecv`——如果 B 同时也在 `ncclSend(to A)` 里等 A 收，两边都在等对方先收，**死锁**。放在同一个 group 里，`ncclP2pSchedule`（`src/init.cc`）会把本轮所有 peer 的 send 和 recv 按固定的轮次表（每轮和一个 peer 互发互收）排进同一个 kernel 的不同 channel，同时进行，不存在"先后"。这也是为什么 `ncclSend/ncclRecv` 的连接是按 peer 懒建的，且 `p2pnChannels` 要是 2 的幂——channel 到 peer 的映射是位运算（`ncclP2pChannelForPart`）。

第六篇 hang 分类里"send/recv 没有配对"就是第 3 条；第五篇会看到 PyTorch 的 `batch_isend_irecv` 和 `_coalescing_manager` 就是为了把这些调用包进一个 group。


## 十、看一看与测一测：读一份 INFO 日志

### 1. 日志格式

`src/debug.cc` 决定每行的前缀。`NCCL_DEBUG` 取 `VERSION`/`WARN`/`INFO`/`ABORT`/`TRACE`；`NCCL_DEBUG_SUBSYS` 是逗号分隔的子系统列表，可用值 `INIT COLL P2P SHM NET GRAPH TUNING ENV ALLOC CALL PROXY NVLS BOOTSTRAP REG PROFILE RAS ALL`，前缀 `^` 表示排除；默认掩码是 `INIT|BOOTSTRAP|ENV`。每个环境变量被 `NCCL_PARAM` 宏（`src/include/param.h` → `src/misc/param.cc: ncclLoadParam`）第一次读取时都会打一行 `%s set by environment to %lld`（子系统 ENV），所以 `NCCL_DEBUG_SUBSYS=ENV` 是"我设的变量到底生效了没"的直接答案。`NCCL_DEBUG_FILE` 可以按 `%h`（hostname）、`%p`（pid）分文件。`~/.nccl.conf` 与 `/etc/nccl.conf`（`NCCL_CONF_FILE`）里的键值会被当成环境变量读入。

INFO 行的格式：

```text
hostname:pid:tid [cudaDev] NCCL INFO <消息>
```

`tid` 值得留意：主线程、proxy service 线程、progress 线程、异步连接线程的 tid 不同，`Connected all rings` 常由另一个 tid 打出（异步预连接线程）。

### 2. 逐行解读

下面是两机 16 卡 A100 + NVSwitch + 每 GPU 一张 HDR 网卡、`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING,ENV` 下 rank 0 的关键行（**示意**，按各 `INFO()` 的格式串拼出，删去了大量重复行；`…` 是省略）：

```text
hostA:12345:12345 [0] NCCL INFO Bootstrap: Using eth0:10.0.0.1<0>
hostA:12345:12345 [0] NCCL INFO NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_1:1/IB … [7]mlx5_7:1/IB [RO]; OOB eth0:10.0.0.1<0>
hostA:12345:12345 [0] NCCL INFO Using network IB
hostA:12345:12345 [0] NCCL INFO ncclCommInitRank comm 0x55d0c0 rank 0 nranks 16 cudaDev 0 nvmlDev 0 busId 7000 commId 0x9a3f… - Init START
hostA:12345:12345 [0] NCCL INFO === System : maxBw 240.0 totalBw 240.0 ===
hostA:12345:12345 [0] NCCL INFO CPU/0-0 (1/2/0)
hostA:12345:12345 [0] NCCL INFO + PCI[24.0] - PCI/0-0-40000
hostA:12345:12345 [0] NCCL INFO               + PCI[24.0] - GPU/0-0-47000 (0)
hostA:12345:12345 [0] NCCL INFO                             + NVL[240.0] - NVS/0-0-0
hostA:12345:12345 [0] NCCL INFO               + PCI[24.0] - NIC/0-0-43000
hostA:12345:12345 [0] NCCL INFO                             + NET[25.0] - NET/0-0-0 (0/…/1/25.000000)
hostA:12345:12345 [0] NCCL INFO ==========================================
hostA:12345:12345 [0] NCCL INFO GPU Direct RDMA Enabled for GPU 0 / HCA 0 (distance 4 <= 5), read 1 mode Default
hostA:12345:12345 [0] NCCL INFO NVLS multicast support is not available on dev 0 (NVLS_NCHANNELS 0)
hostA:12345:12345 [0] NCCL INFO Pattern 4, crossNic 0, nChannels 12, bw 20.000000/20.000000, type NVL/PIX, sameChannels 1
hostA:12345:12345 [0] NCCL INFO  0 : NET/0-0-0 GPU/0-0-47000 GPU/0-0-4e000 GPU/0-0-87000 … GPU/0-0-ce000 NET/0-0-0
hostA:12345:12345 [0] NCCL INFO Pattern 1, crossNic 0, nChannels 12, bw 20.000000/20.000000, type NVL/PIX, sameChannels 1
hostA:12345:12345 [0] NCCL INFO comm 0x55d0c0 rank 0 nRanks 16 nNodes 2 localRanks 8 localRank 0 MNNVL 0
hostA:12345:12345 [0] NCCL INFO Channel 00/24 :    0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
hostA:12345:12345 [0] NCCL INFO Channel 01/24 :    0   3   2   1   5   4   7   6   8  11  10   9  13  12  15  14
hostA:12345:12345 [0] NCCL INFO Ring 00 : 15 -> 0 -> 1
hostA:12345:12345 [0] NCCL INFO Tree 0 : -1 -> 0 -> 1/8/-1
hostA:12345:12345 [0] NCCL INFO Tree 12 : 8 -> 0 -> 1/-1/-1
hostA:12345:12345 [0] NCCL INFO Trees [0] 1/8/-1->0->-1 [1] 1/8/-1->0->-1 … [12] 1/-1/-1->0->8 …
hostA:12345:12345 [0] NCCL INFO threadThresholds 8/8/64 | 128/8/64 | 512 | 512
hostA:12345:12345 [0] NCCL INFO   Algorithm   |                    Tree                   |                    Ring                   | …
hostA:12345:12345 [0] NCCL INFO   Protocol    |       LL |    LL128 |   Simple |       LL |    LL128 |   Simple | …
hostA:12345:12345 [0] NCCL INFO     AllReduce |    25.2/ 11.3 |    48.5/225.0 |    92.4/220.8 |    36.2/ 12.0 |    75.2/235.5 |   131.6/256.0 | …
hostA:12345:12345 [0] NCCL INFO 24 coll channels, 24 collnet channels, 0 nvls channels, 32 p2p channels, 32 p2p channels per peer
hostA:12345:12345 [0] NCCL INFO ncclCommInitRank comm 0x55d0c0 rank 0 nranks 16 cudaDev 0 nvmlDev 0 busId 7000 commId 0x9a3f… - Init COMPLETE
hostA:12345:12401 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM
hostA:12345:12401 [0] NCCL INFO Channel 00/0 : 15[7] -> 0[0] [receive] via NET/IB/0/GDRDMA
hostA:12345:12401 [0] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
hostA:12345:12401 [0] NCCL INFO Channel 00/0 : 0[0] -> 8[0] [send] via NET/IB/0/GDRDMA
hostA:12345:12401 [0] NCCL INFO Connected all trees
hostA:12345:12345 [0] NCCL INFO AllReduce: 1073741824 Bytes -> Algo Ring proto Simple channel{Lo..Hi}={0..23}
hostA:12345:12345 [0] NCCL INFO AllReduce: 26214400 Bytes -> Algo Tree proto LL128 channel{Lo..Hi}={0..23}
hostA:12345:12345 [0] NCCL INFO AllReduce: 65536 Bytes -> Algo Tree proto LL channel{Lo..Hi}={0..1}
```

逐段解释：

- **Bootstrap 与网络**：`Bootstrap: Using eth0:…` 是 bootstrap 走的接口（第二章）；`NET/IB : Using [0]mlx5_0:1/IB …` 列出 IB 插件看到的所有设备（`[序号]设备名:端口/链路层`，`[RO]` 是 relaxed ordering）；`Using network IB` 是最终选定的 net 插件——如果这里是 `Socket`，跨机性能会差一个数量级，先查 IB 插件为什么没加载。
- **拓扑打印**（`ncclTopoPrint`）：`maxBw/totalBw` 是 GPU 出口的最大/总带宽；缩进的树就是第三章的图，`PCI[24.0]` 是 Gen4 x16、`NVL[240.0]` 是 12 link × 20、`NET[25.0]` 是 HDR。GPU 后括号里是 rank，NET 后括号是 `(collSupport/asic/port/bw)`。
- **GDR 判定**：`distance 4 <= 5` 即 `PIX ≤ PXB`，开了 GPUDirect RDMA；`read 1` 表示发送方向也开。如果这里是 `Disabled … distance 8 > 5`，说明 GPU 与 NIC 跨了 root complex。
- **NVLS**：A100 上不可用，H100 上会是 `is available … (NVLS_NCHANNELS 16)`。
- **图搜索结果**：`Pattern 4`（ring）12 条、每条 20/20 GB/s、节点内 NVL、节点间 PIX、各 channel 相同；每行一条 channel 的 GPU 顺序，首尾是进出 NIC（同一个，`crossNic 0`）。`Pattern 1` 是 balanced tree，channel 数被钉成和 ring 一样。
- **全局 ring**：`Channel 00/24 : 0 1 2 … 15` 是 rank 0 视角的第 0 条 ring 的完整顺序（`dumpLine`），共 24 条（12 × 2）；channel 12 与 channel 0 相同（复制）。`Ring 00 : 15 -> 0 -> 1` 是本 rank 在这条 ring 上的 prev/next——15 在另一台机器上，说明这条 ring 在这里跨机。
- **全局 tree**：`Tree 0 : -1 -> 0 -> 1/8/-1`：rank 0 是第一棵树的根（up=-1），子节点是本机的 1（节点内链的下一个）和 hostB 的 8（节点间树的子）。`Tree 12 : 8 -> 0 -> 1/-1/-1`：第二棵树里 rank 0 的父是 8——double binary tree 的镜像。`Trees` 一行是所有 channel 的汇总，格式 `[c] down0/down1/down2->me->up`。
- **channel 与阈值**：`24 coll channels … 32 p2p channels`；`threadThresholds 8/8/64 | 128/8/64 | 512 | 512` 依次是 Tree 的 LL/LL128/Simple、Ring 的 LL/LL128/Simple（Ring+LL 是 8 × 16 rank = 128）、CollNetDirect、CollNetChain。`[Proxy Service] Device 0 CPU core 12`（本例中省略）告诉你 proxy 线程落在哪个核。
- **调优表**：格式见第八章，这里是两机 16 卡的版本（`hw = NET`，`nNodes ≤ 2` 时带宽仍用 `bwIntra`）。Ring+Simple 的 131.6 µs 是 $$8.4 + (30-2) \times 3.4 + 2 \times 14$$——30 步里 2 步跨网络（每步 14 µs）、28 步在 NVLink 上（每步 3.4 µs）；带宽 $$480 \times 16/30 = 256$$。Tree+Simple 的 92.4 µs 是 $$8.4 + 2 \times (7 \times 4.0 + 1 \times 14)$$。Ring+LL 的带宽被多机的 `llMaxBw`（AMD CPU、2 节点档 22.5）压到 12.0——这就是为什么多机 LL 只会出现在极小消息上。你的日志里以实际打印为准，`nccl_log_reader.py` 直接用它算预测。
- **连接**（`Init COMPLETE` 之后、另一个 tid）：节点内 `P2P/CUMEM`，跨机 `NET/IB/0/GDRDMA`；`Connected all rings, use ring PXN 0 GDR 1` 说明所有 ring 连接都用上了 GDR、没有用 PXN。
- **算法选择**（`TUNING` 子系统，rank 0）：1 GB → Ring+Simple 用满 24 channel；25 MB → Tree+LL128（两机时 tree 的延迟优势在几十 MB 还在）；64 KB → Tree+LL 只用 2 个 channel。这三行就是核心问题在这台机器上的答案。

### 3. 改拓扑文件，观察决策变化

把 `NCCL_TOPO_DUMP_FILE` 导出的 XML 改一改再用 `NCCL_TOPO_FILE` 喂回去，是理解决策链最直接的实验（在单机上也能做；改的是 NCCL 对机器的**认知**，不是机器本身，所以性能结果不作数，只看日志里决策怎么变）：

```text
实验                                     改动                                              预期看到
去掉 NVLink                              删掉所有 <nvlink …/>                                type PIX/… 或 PHB；nChannels 降到 2～4；LL128 消失（Enabled 矩阵里为 0）；小消息选 LL
把 NIC 挪到另一个 NUMA                   把 <nic> 所在 <pci> 剪到另一个 <cpu> 下              GPU Direct RDMA Disabled … distance 9 > 5；连接行没有 /GDRDMA
把 NVLink count 减半                     count="2" → "1"                                     NVL[120.0]；ring nChannels 从 12 降到 6（×2 后 12）
```

同样，`NCCL_GRAPH_DUMP_FILE` 导出的图改 `nchannels`、`speedintra` 后用 `NCCL_GRAPH_FILE` 喂回去，可以直接看到调优表随 channel 数与带宽的变化，而不必重新搜索。

### 4. 比一比：这一层的检查清单

理论与实测有差距时，先按下面的顺序看日志（第六篇会把它并入完整的决策树）：

```text
1  Using network 是不是 IB（或你期望的插件）？               不是 → 插件加载 / NCCL_NET / NCCL_IB_HCA / 权限
2  Bootstrap: Using 的接口对不对？                            错 → NCCL_SOCKET_IFNAME（只影响 bootstrap 与 Socket 数据面）
3  拓扑打印里 NVL 带宽 = link 数 × 每 link？PCI 是 Gen4/5 x16？  少了 → 硬件/驱动/容器把设备藏了；nvidia-smi topo -m 对照
4  GPU Direct RDMA Enabled？连接行有 /GDRDMA？                 没有 → GPU/NIC 亲和（distance > 5）或 nvidia-peermem 没加载；确认后再考虑 NCCL_NET_GDR_LEVEL
5  节点内连接是 P2P 还是 SHM？                                SHM → 跨 root complex 且 NCCL_P2P_LEVEL 默认 PXB；或 NVML 报 P2P 不可用
6  Pattern 4 的 nChannels × bw 是否接近 maxBw？type 是否 NVL？  不是 → 图搜索退化；看有没有 "falling back to simple order"
7  coll channels 数是否被 NCCL_MAX_NCHANNELS / maxCTAs 压低？   是 → 大消息带宽上限被砍
8  Enabled Func/Proto/Algo Matrix：LL128 是否为 0？NVLS 是否为 0？  是 → 路径类型不满足（PIX 以上）/ 驱动不支持；不要硬开
9  AllReduce: N Bytes -> Algo … proto … channel{Lo..Hi}：与调优表算出来的最小值一致？  不一致 → 有 tuner 插件或 NCCL_ALGO/PROTO 在生效
10 [Proxy Progress] Device d CPU core c：这个核是否被绑给了别的忙进程？  是 → NCCL_PROXY_CPUSET 或调度器亲和
```


## 十一、本文小结

### 1. 要点回顾

```text
两段路径      ncclCommInitRank 做与消息无关的决策（拓扑→路径→图→channel→调优表），ncclAllReduce 只查表、切 channel、启一个 kernel
bootstrap     uniqueId 是 rank 0 的监听地址 + magic；TCP 环上做 AllGather；NCCL_SOCKET_IFNAME 只管这一段和 Socket 数据面
拓扑          每 rank 探测自己的 GPU（/sys 走 PCIe 树，NVML 查 NVLink）+ net 插件报 NIC，节点内 XML 融合；NCCL_TOPO_DUMP_FILE 导出
带宽常数      NVLink 每 link 20/20.6/40.1（A100/H100/B200），PCIe = width×speed/80（Gen4 x16 = 24），NET = Mbps/8000；都是保守估计
路径          BFS，类型取沿途最差一段：LOC<NVL<NVB<C2C<PIX<PXB<P2C<PXN<PHB<SYS；NCCL_P2P_LEVEL / NCCL_NET_GDR_LEVEL 默认 PXB
图搜索        在节点内找 nChannels 条每条 bw 的 ring/tree，离散速度表，两个 pass；connect.cc 拼成全局 ring 与 double binary tree，channel ×2
transport     P2P → SHM → NET → CollNet 顺序尝试；NVLS 单独走 multicast；2.28 默认 lazy connect，首次用到某算法才连
channel       一条 ring/tree + 每 peer 每方向的 buffer/连接 + kernel 的一个 block + 一份 proxy 工作；nChannels 由图×2、env 裁剪、按消息大小缩
算法          Tree Ring CollNetDirect CollNetChain NVLS NVLSTree PAT（ncclAlgoStr 顺序）
协议          LL 8+8 字节原子 store 无 fence 50%；LL128 120/128 依赖 NVLink 写序 ~94%，仅 NVLink 路径启用；Simple 512 KiB slot + fence ≈100%
调优          T = lat×latCount + S/(1000×bw)；lat = base + 步数×每步硬件延迟；bw = nCh×每 ch 带宽×协议效率×算法换算；常数在 tuning.cc
核心问题      NVSwitch 机：NVLS 交换机内归约，NVLink 流量 ~2S/n vs ring 1.75S，延迟 25 µs 也不高 → NVLS+Simple（几百 KB 以上）
              有 NVLink 无 NVSwitch：NVLS 不可用、LL128 可用 → Ring+LL128（几 MB 以下）；纯 PCIe：LL128 被禁 → Ring/Tree + Simple/LL
              32 节点：ring 510 步 vs tree 2×(7+5) 步，延迟差一个量级，带宽差 30% → Tree 直到几百 MB
强行 Ring     单机 NVSwitch 大消息慢约 2×（带宽账）；32 节点中小消息慢 4～7×（延迟账）；只在大规模 + GB 级消息上正确
执行期        group 聚合 → 4 倍以内的任务合并查表 → 一个 kernel nChannels 个 block → 每 block 一条 ring/tree 用 primitives 收发
proxy         GPU 不能 post RDMA；CPU 线程按 head/tail 与 GPU 同步，替它 isend/irecv/test；单线程、被抢占即抖动；kernel 自旋等它
send/recv     必须同 group：否则 send kernel 等 recv、recv 在 send 返回后才发起 → 死锁
```

### 2. 排障检查项 / 决策要点

```text
现象                          先看                                                  再看
多机远差于单机                Using network 是否 IB；连接行是否 /GDRDMA；GDR distance   NCCL_IB_HCA、PXN、NIC 亲和、crossNic
单机大消息上不去              拓扑打印 NVL 带宽；Pattern 4 nChannels×bw；coll channels 数   NCCL_MAX_NCHANNELS / maxCTAs、是否落到 SHM
小消息延迟高                  选了什么 proto（LL128 是否被禁）；channel{Lo..Hi} 是否过大    NCCL_MIN_NCHANNELS、proxy 线程所在核、跨 NUMA
NCCL_ALGO/PROTO 不生效        ENV 子系统有没有 "set by environment"；Enabled 矩阵         tuner 插件 NCCL_TUNER_PLUGIN 是否加载
"no algorithm/protocol available"  NCCL_ALGO/PROTO 指定了机器不支持的组合               去掉 env 或按 ncclAlgoStr 拼写
初始化 hang                   Bootstrap: Using 的接口；NCCL_COMM_ID 子网                  防火墙、NCCL_SOCKET_IFNAME、NCCL_OOB_NET_ENABLE
第一次集合操作特别慢          正常：lazy connect（Connected all rings 在 Init COMPLETE 之后）  PyTorch 侧 timeout 是否够
kernel 长时间自旋             proxy 是否还活着；NCCL_PROXY_DUMP_SIGNAL 看游标              对端 rank 是否存活；网络；send/recv 是否配对
决策：要不要设 NCCL_ALGO      先用 TUNING 日志看自动选择与调优表；用 nccl-tests 扫描确认交叉点   再写进 tuner 配置而不是全局 env
```

### 3. 本篇涉及的源码与工具位置

| 主题 | 路径（NCCL 2.28.9） | 函数 / 符号 |
|---|---|---|
| 入口与初始化骨架 | `src/init.cc` | `ncclGetUniqueId`、`ncclCommInitRank`、`ncclCommInitRankDev`、`initTransportsRank`、`computeBuffSizes`、`ncclAlgoStr`、`ncclProtoStr` |
| bootstrap | `src/bootstrap.cc`、`src/misc/socket.cc` | `bootstrapNetInit`、`bootstrapGetUniqueId`、`bootstrapInit`、`bootstrapAllGather`、`bootstrapIntraNodeAllGather`；`NCCL_SOCKET_IFNAME`、`NCCL_COMM_ID`、`NCCL_OOB_NET_ENABLE` |
| 拓扑 XML | `src/graph/xml.cc`、`src/graph/xml.h` | `ncclTopoFillGpu`、`ncclTopoFillNet`、`ncclTopoTrimXml`、`ncclTopoFuseXml`、`ncclTopoGetXmlFromFile`、`ncclTopoDumpXmlToFile` |
| 拓扑图 | `src/graph/topo.cc`、`src/graph/topo.h` | `ncclTopoGetSystem`、`ncclTopoGetSystemFromXml`、`ncclTopoAddPci/Nic/Net/Cpu/NvLinks`、`ncclTopoPrint`、`ncclTopoNVLinkBw`、`topoPathTypeStr`、带宽常数；`NCCL_TOPO_FILE`、`NCCL_TOPO_DUMP_FILE`、`NCCL_TOPO_DUMP_FILE_RANK`、`NCCL_IGNORE_CPU_AFFINITY` |
| 路径 | `src/graph/paths.cc` | `ncclTopoComputePaths`、`ncclTopoSetPaths`、`ncclTopoCheckP2p`、`ncclTopoCheckGdr`、`ncclGetLevel`、`ncclTopoTrimSystem`、`ncclTopoComputeP2pChannels`；`NCCL_P2P_LEVEL`、`NCCL_P2P_DISABLE`、`NCCL_NET_GDR_LEVEL`、`NCCL_NET_GDR_READ`、`NCCL_PXN_DISABLE`、`NCCL_IGNORE_DISABLED_P2P`、`NCCL_MIN/MAX_P2P_NCHANNELS` |
| 图搜索 | `src/graph/search.cc`、`src/include/graph.h` | `ncclTopoCompute`、`ncclTopoSearchRec`、`ncclTopoDupChannels`、`ncclTopoPrintGraph`、`ncclTopoDumpGraphs`、`speedArray*`、`NCCL_TOPO_PATTERN_*`；`NCCL_GRAPH_FILE`、`NCCL_GRAPH_DUMP_FILE`、`NCCL_GRAPH_DUMP_FILE_RANK`、`NCCL_CROSS_NIC` |
| ring / tree 拼接 | `src/graph/connect.cc`、`rings.cc`、`trees.cc` | `ncclTopoPreset`、`ncclTopoPostset`、`connectRings`、`connectTrees`、`ncclMinNchannels`、`ncclMaxNchannels`、`ncclBuildRings`、`dumpLine`、`ncclGetBtree`、`ncclGetDtree`；`NCCL_MIN_NCHANNELS`、`NCCL_MAX_NCHANNELS` |
| 调优 | `src/graph/tuning.cc` | `ncclTopoTuneModel`、`ncclTopoGetAlgoTime`、`parseList`、`ncclTunerConstantsDefaults`、`treeCorrectionFactor`；`NCCL_ALGO`、`NCCL_PROTO`、`NCCL_NTHREADS`、`NCCL_LL128_NTHREADS`、`NCCL_THREAD_THRESHOLDS`、`NCCL_NET_OVERHEAD`、`NCCL_PAT_ENABLE`、`NCCL_LL128_C2C` |
| tuner 插件 | `src/plugin/tuner.cc`、`src/include/plugin/tuner/tuner_v5.h`、`ext-tuner/example/` | `ncclTunerPluginLoad`、`init`/`getCollInfo`/`finalize`、`nccl_tuner.conf`；`NCCL_TUNER_PLUGIN` |
| transport | `src/transport.cc`、`src/transport/{p2p,shm,net,coll_net,nvls,generic}.cc` | `ncclTransports[]`、`selectTransport`、`ncclTransportP2pSetup`、`ncclTransportRingConnect`、`ncclTransportTreeConnect`、`ncclTransportPatConnect`、`ncclNvlsInit`、`ncclNvlsSetup`、`sendProxyProgress`、`recvProxyProgress`；`NCCL_SHM_DISABLE`、`NCCL_NVLS_ENABLE`、`NCCL_NVLS_NCHANNELS`、`NCCL_COLLNET_ENABLE`、`NCCL_COLLNET_NODE_THRESHOLD`、`NCCL_RUNTIME_CONNECT` |
| channel 与 buffer | `src/channel.cc`、`src/include/device.h`、`src/include/comm.h` | `initChannel`、`ncclChannel`、`ncclConnInfo`、`ncclRing`、`ncclTree`、`NCCL_STEPS`、`MAXCHANNELS`、`ncclLLFifoLine`、`NCCL_LL128_*`；`NCCL_BUFFSIZE`、`NCCL_LL_BUFFSIZE`、`NCCL_LL128_BUFFSIZE` |
| enqueue | `src/enqueue.cc`、`src/collectives.cc` | `ncclEnqueueCheck`、`taskAppend`、`ncclPrepareTasks`、`getAlgoInfo`、`topoGetAlgoInfo`、`updateCollCostTable`、`scheduleCollTasksToPlan`、`calcCollChunking`、`ncclLaunchKernel`；`NCCL_CGA_CLUSTER_SIZE`、`NCCL_WORK_ARGS_BYTES` |
| group | `src/group.cc` | `ncclGroupStart/End`、`ncclGroupEndInternal`、`groupLaunch`、`ncclCollPreconnect`、`doLaunches`、`ncclGroupDepth`；`NCCL_LAUNCH_MODE` |
| proxy | `src/proxy.cc` | `ncclProxyCreate`、`ncclProxyService`、`ncclProxyProgress`、`progressOps`、`ncclProxyGetPostedOps`、`ncclProxySaveOp`、`ncclDumpProxyState`；`NCCL_PROXY_CPUSET`、`NCCL_PROXY_DUMP_SIGNAL`、`NCCL_PROGRESS_APPENDOP_FREQ`、`NCCL_PROXY_APPEND_BATCH_SIZE` |
| 设备侧 | `src/device/common.h`、`all_reduce.h`、`primitives.h`、`prims_simple.h`、`prims_ll.h`、`prims_ll128.h`、`generate.py` | `ncclKernelMain`、`loadWorkBatchToShmem`、`runRing`、`runTreeUpDown`、`runTreeSplit`、`Primitives<…>`、`waitPeer`、`postPeer`、`readLL`、`storeLL`、`flagThread` |
| 日志与参数 | `src/debug.cc`、`src/misc/param.cc`、`src/include/param.h` | `ncclDebugInit`、`ncclDebugLog`、`NCCL_PARAM`、`ncclLoadParam`、`ncclGetEnv`；`NCCL_DEBUG`、`NCCL_DEBUG_SUBSYS`、`NCCL_DEBUG_FILE`、`NCCL_CONF_FILE` |
| 较新目录 | `src/register/`、`src/scheduler/`、`src/ras/`、`src/plugin/` | 用户 buffer 注册；对称内存调度；RAS；net / tuner / profiler / env 插件加载 |

### 4. comm-probe 本篇增量：nccl_log_reader.py

**做什么**：读一份（或多个 rank 拼在一起的）`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING` 日志，提取：网络插件、bootstrap 接口、拓扑摘要（maxBw、GDR 判定）、图搜索结果（每种 pattern 的 nChannels/bw/type）、全局 ring（`Channel xx/yy`）与 tree（`Trees`）、channel 数、每条连接用的 transport、调优表（每个算法×协议的 lat/bw）、以及每次集合操作的算法协议决策。然后对每次决策，用调优表算出 NCCL 自己的预估时间，再用第一篇 `cost_model.py` 的 α-β 模型（输入第二篇填的链路 α、β）算理论时间，三者并列，供第六篇与 nccl-tests 实测对照。

**输入**：日志文件路径；可选 `--alpha-us`、`--beta-gbps`（给 `cost_model.py`）。**输出**：一段结构化摘要（也可 `--json`）。

关键实现（Python 3，约 90 行，正则全部对应本篇引用的 `INFO()` 格式串）：

```python
#!/usr/bin/env python3
"""nccl_log_reader.py: parse NCCL_DEBUG=INFO logs (INIT,GRAPH,TUNING) and compare with cost_model."""
import re, sys, json, argparse, collections
from cost_model import ring_allreduce_time   # 第一篇：T = 2(n-1)α + 2(n-1)/n · S/β，单位 µs / GB/s

LINE   = re.compile(r'^(?P<host>\S+):(?P<pid>\d+):(?P<tid>\d+) \[(?P<dev>\d+)\] NCCL INFO (?P<msg>.*)$')
PATS = {   # 与源码 INFO() 格式串一一对应（文件: 函数）
  'net':      re.compile(r'^Using network (\S+)'),                                             # init.cc: commAlloc
  'boot':     re.compile(r'^Bootstrap: Using (\S+)'),                                          # bootstrap.cc: bootstrapNetInit
  'system':   re.compile(r'^=== System : maxBw ([\d.]+) totalBw ([\d.]+) ==='),                 # topo.cc: ncclTopoPrint
  'gdr':      re.compile(r'^GPU Direct RDMA (Enabled|Disabled) for GPU (\d+) / HCA (\S+) \(distance (\d+) [<>]=? (\d+)\)'),  # paths.cc
  'graph':    re.compile(r'^Pattern (\d+), crossNic (\d+), nChannels (\d+), bw ([\d.]+)/([\d.]+), type (\w+)/(\w+), sameChannels (\d)'),  # search.cc
  'comm':     re.compile(r'^comm \S+ rank (\d+) nRanks (\d+) nNodes (\d+) localRanks (\d+)'),  # init.cc
  'ring':     re.compile(r'^Channel (\d+)/(\d+) :((?:\s+\d+)+)$'),                              # rings.cc: dumpLine
  'trees':    re.compile(r'^Trees((?: \[\d+\] -?\d+/-?\d+/-?\d+->\d+->-?\d+)+)'),               # init.cc
  'nchan':    re.compile(r'^(\d+) coll channels, (\d+) collnet channels, (\d+) nvls channels, (\d+) p2p channels'),  # init.cc
  'conn':     re.compile(r'^Channel (\d+)/\d+ : (\d+)\[\d+\] -> (\d+)\[\d+\](?: \[(send|receive)\])? via (\S+)'),   # p2p.cc / net.cc
  'connshm':  re.compile(r'^Channel (\d+) : (\d+)\[\d+\] -> (\d+)\[\d+\] via (SHM\S*)'),        # shm.cc
  'connected':re.compile(r'^Connected (all rings|all trees|NVLS tree|binomial trees)(, use ring PXN (\d) GDR (\d))?'),  # generic.cc / nvls.cc
  'env':      re.compile(r'^(NCCL_\w+) set by environment to (.+)$'),                          # param.cc / tuning.cc / paths.cc
  'algohdr':  re.compile(r'^  Algorithm   \|(.*)$'),                                            # tuning.cc: ncclTopoTuneModel
  'protohdr': re.compile(r'^  Protocol    \|(.*)$'),
  'tunerow':  re.compile(r'^\s*(Broadcast|Reduce|AllGather|ReduceScatter|AllReduce) \|(.*)$'),
  'decision': re.compile(r'^(AllReduce|AllGather|ReduceScatter|Broadcast|Reduce): (\d+) Bytes -> Algo (\w+) proto (\w+) channel\{Lo\.\.Hi\}=\{(\d+)\.\.(\d+)\}'),  # enqueue.cc
}
PATTERN_NAME = {1: 'BalancedTree', 2: 'SplitTree', 3: 'Tree', 4: 'Ring', 5: 'NVLS', 6: 'CollNetDirect'}

def parse(path):
    r = collections.defaultdict(list); r['env'] = {}; r['tuning'] = {}; algos = protos = None
    for raw in open(path, errors='replace'):
        m = LINE.match(raw.rstrip('\n'))
        if not m: continue
        msg = m['msg']
        for key, pat in PATS.items():
            g = pat.match(msg)
            if not g: continue
            if key == 'env':       r['env'][g[1]] = g[2]
            elif key == 'graph':   r['graphs'].append(dict(pattern=PATTERN_NAME.get(int(g[1]), g[1]), crossNic=int(g[2]), nChannels=int(g[3]),
                                                         bwIntra=float(g[4]), bwInter=float(g[5]), typeIntra=g[6], typeInter=g[7], same=int(g[8])))
            elif key == 'ring':    r['rings'].append((int(g[1]), int(g[2]), [int(x) for x in g[3].split()]))
            elif key == 'trees':   r['trees'] = re.findall(r'\[(\d+)\] (-?\d+)/(-?\d+)/(-?\d+)->(\d+)->(-?\d+)', g[1])
            elif key == 'conn':    r['conns'].append(dict(ch=int(g[1]), src=int(g[2]), dst=int(g[3]), dir=g[4] or 'p2p', via=g[5]))
            elif key == 'connshm': r['conns'].append(dict(ch=int(g[1]), src=int(g[2]), dst=int(g[3]), dir='p2p', via=g[4]))
            elif key == 'algohdr': algos = [a for a in re.findall(r'\b(Tree|Ring|CollNetDirect|CollNetChain|NVLS|NVLSTree|PAT)\b', g[1])]
            elif key == 'protohdr':protos = list(dict.fromkeys(re.findall(r'\b(LL128|LL|Simple)\b', g[1])))   # 表按 3 个算法一块打印，每块重复协议头
            elif key == 'tunerow' and algos and protos:      # 每格 "lat/bw"，按 algos × protos 展开
                cells = re.findall(r'([\d.]+)/\s*([\d.]+)', g[2])
                tab = r['tuning'].setdefault(g[1], {})
                for i, (lat, bw) in enumerate(cells):
                    a, p = algos[i // len(protos)], protos[i % len(protos)]
                    tab[(a, p)] = (float(lat), float(bw))
            elif key == 'decision':r['decisions'].append(dict(func=g[1], bytes=int(g[2]), algo=g[3], proto=g[4], chLo=int(g[5]), chHi=int(g[6])))
            else:                  r[key].append(g.groups())
            break
    return r

def nccl_model_time_us(tab, algo, proto, nbytes):
    lat, bw = tab.get((algo, proto), (None, 0.0))    # tuning.cc: ncclTopoGetAlgoTime（忽略 treeCorrectionFactor / plateau）
    return None if not bw else lat + nbytes / (1000.0 * bw)

def report(r, alpha_us, beta_gbps):
    nranks = int(r['comm'][0][1]) if r['comm'] else None
    print(f"network={r['net'][0][0] if r['net'] else '?'}  bootstrap={r['boot'][0][0] if r['boot'] else '?'}  nRanks={nranks}  nNodes={r['comm'][0][2] if r['comm'] else '?'}")
    for g in r['graphs']:  print(f"graph {g['pattern']:<14} nChannels={g['nChannels']:<3} bw={g['bwIntra']}/{g['bwInter']} type={g['typeIntra']}/{g['typeInter']} crossNic={g['crossNic']}")
    if r['nchan']:         print(f"channels coll/collnet/nvls/p2p = {'/'.join(r['nchan'][0])}")
    via = collections.Counter(c['via'].split('/')[0] + ('/GDRDMA' if 'GDRDMA' in c['via'] else '') for c in r['conns'])
    print("transports:", dict(via), " rings:", len(r['rings']), " gdr:", [(e[0], e[1], e[3]) for e in r['gdr']][:4])
    for k, v in r['env'].items(): print(f"env {k}={v}")
    tab = r['tuning'].get('AllReduce', {})
    for d in r['decisions']:
        chosen = nccl_model_time_us(tab, d['algo'], d['proto'], d['bytes'])
        best = min(((nccl_model_time_us(tab, a, p, d['bytes']) or 1e18, a, p) for (a, p) in tab), default=(None, '?', '?'))
        ab = ring_allreduce_time(nranks, d['bytes'], alpha_us, beta_gbps) if nranks else None
        print(f"{d['func']} {d['bytes']:>12} B -> {d['algo']}+{d['proto']} ch[{d['chLo']}..{d['chHi']}] | nccl-model {chosen and round(chosen,1)} µs"
              f" (table-min {best[1]}+{best[2]} {round(best[0],1) if best[0] else '?'}) | alpha-beta ring {ab and round(ab,1)} µs")

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('log'); ap.add_argument('--alpha-us', type=float, default=3.4); ap.add_argument('--beta-gbps', type=float, default=240.0); ap.add_argument('--json', action='store_true')
    a = ap.parse_args(); r = parse(a.log)
    if a.json: print(json.dumps({k: (v if not isinstance(v, dict) or k != 'tuning' else {f: {f'{x}+{y}': z for (x, y), z in t.items()} for f, t in v.items()}) for k, v in r.items()}, default=str, indent=1))
    else: report(r, a.alpha_us, a.beta_gbps)
```

三个对照列的含义：`nccl-model` 是 NCCL 用自己的调优表对**它选的**组合算出的时间；`table-min` 是同一张表里的最小值及其组合——两者不同就说明有 `NCCL_ALGO`/`NCCL_PROTO` 或 tuner 插件在干预；`alpha-beta ring` 是第一篇模型用第二篇实测的 α、β 算的 ring 理论时间——它与 `nccl-model` 的差距是"NCCL 的经验常数 vs 你机器的实际链路"，与第六篇 nccl-tests 实测的差距是"模型 vs 现实"。三列摆在一起，哪一层出了问题一眼可见。

下一篇往上走一层：PyTorch 是如何使用 NCCL 的。本篇讲清了 `ncclAllReduce` 返回时 kernel 只是被放进了 stream，那么框架侧的"异步"、"重叠"、"wait"到底各自意味着什么？

> **`work = dist.all_reduce(t, async_op=True)` 返回时，通信开始了吗？`work.wait()` 返回时，通信完成了吗？在此期间修改 `t` 会发生什么？**


## 下一篇

[PyTorch 的通信栈：ProcessGroupNCCL、stream 语义与计算通信重叠](/pytorch-communication-stack-processgroupnccl-and-streams.html)
