---
layout: post
title: "AI 平台工程（05）：网络与存储——RDMA 进容器、并行文件系统与 checkpoint I/O"
subtitle: "Networking and Storage: RDMA in Containers, Parallel File Systems and Checkpoint I/O"
tags: [Kubernetes, GPU, RDMA, Storage, AI, AI-Infra]
catalog: true
---

> 本文是《AI 平台工程：资源层与交付层》系列的第 5 篇。上一篇：[GPU 共享与切分：MIG、时间片、MPS 与 HAMi](/gpu-sharing-and-partitioning-mig-mps-hami.html)；下一篇：[Serving 平台：从 InferenceService 到 llm-d](/serving-platforms-kserve-triton-ray-serve-llm-d.html)。

一个 8 节点 64 卡的训练任务在裸机上跑通了，`all_reduce_perf` 的大消息 busbw 接近网卡的标称值。同一个镜像、同一套 NCCL 环境变量搬到 Kubernetes 上，Pod 全部 Running，任务也在正常推进，只是 step time 慢了两倍多。再跑一次 `all_reduce_perf`，busbw 只剩裸机的三分之一。日志里没有报错，`nvidia-smi` 显示八张卡都在，`ibstat` 在宿主机上也一切正常。

这种"没有错误、只是慢"的故障在平台层非常典型，因为它不是某个组件坏了，而是**引擎想走的那条路平台没有铺**。NCCL 需要直接打开 `/dev/infiniband/uverbs*`、在网卡上注册显存、通过 RDMA 网卡的 IP 完成握手；而 Kubernetes 默认给每个 Pod 的只有一张 veth 网卡和一个 overlay 地址。NCCL 发现没有 RDMA 设备，就安静地退回 TCP socket，用那张 veth 把几十 GB 的梯度搬来搬去——它做了它该做的事，慢是唯一的症状。

存储侧的故障形态是同一类：训练任务每半小时写一次 checkpoint，在单机 NVMe 上几十秒写完；上了集群改写共享文件系统，一次 checkpoint 十几分钟，训练在这十几分钟里停着。没有报错，只是慢。而推理服务扩容一个副本要拉 140 GB 权重，从对象存储拉要八分钟，"扩容"这个动作的大部分时间花在了 I/O 上。

本篇讲这两条"喂饱 GPU"的管道在 Kubernetes 上怎么铺：网络这一半是让 RDMA 设备和 GPUDirect RDMA 在容器里可用、可验证；存储这一半是按三类负载（数据集、checkpoint、权重）选方案，并把 checkpoint I/O 的带宽需求算清楚。本篇要回答总纲提出的核心问题：

> **一个 8 节点 64 卡的训练任务，`nccl-tests` 在容器里测出的 all_reduce 带宽只有裸机的三分之一。从 Pod 的网络配置、device plugin 的资源分配、NCCL 的环境变量三个层面，各自可能出了什么问题？**

依照系列惯例，本篇不解释 NCCL 为什么选某条路径、不讲 RDMA verbs 与 IB/RoCE 的协议差别，只讨论如何让 NCCL 在容器里能选到 RDMA 这条路、如何从日志确认它选了。源码与 CRD 以 Multus CNI v4.3.0、k8s-rdma-shared-dev-plugin v1.5.4、NVIDIA Network Operator v26.7.0、NVIDIA GPU Operator v26.7.0 为准；NCCL 2.28.9 与 nccl-tests 2.18.3 只用到环境变量名、日志行与命令行；PyTorch v2.13.0 只用到 `torch.distributed.checkpoint` 的公开 API。存储产品（Lustre、Storage Scale、WEKA、BeeGFS、JuiceFS、Alluxio、Fluid、GPUDirect Storage）按公开文档描述定位，本篇不给任何实测数字。


## 一、总览

### 1. 引擎的需求

训练框架与推理引擎在网络和存储这一层的需求，可以从 NCCL 的初始化和 PyTorch 的 checkpoint 路径倒推出来：

```text
需求                                 来源                                    不满足时的表现
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
进程能打开 RDMA 设备文件               NCCL ncclIbInit 调 ibv_get_device_list   "NET/IB : No device found." → 回落 Socket
                                       枚举 /dev/infiniband/uverbs*
网卡能直接读写显存（GPUDirect RDMA）    nvidia_peermem 或 DMA-BUF                "GPU Direct RDMA Disabled" → 数据经 host 内存中转
一个能路由到对端的 RDMA 侧 IP          NCCL 的 OOB 握手、RoCE 的 GID 选择       握手失败，或 GID 选到不通的地址
进程能锁定内存（memlock）              RDMA 注册内存需要 pin                    ibv_reg_mr 失败（NCCL 日志 "Call to ibv_reg_mr failed with error …"）
GPU 与网卡的 PCIe 亲和                 NCCL 按拓扑距离决定是否启用 GDR          跨 NUMA 的网卡被 GDR_LEVEL 排除
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
数据集：小文件随机读、可缓存           dataloader 多进程读                       GPU 等 I/O，利用率锯齿
checkpoint：大块顺序写、突发           每 N 步一次，写完才继续                   训练停顿；写不完就下一次开始
权重：一次写多次读、并发读             推理副本扩容                              扩容时间被 I/O 主导
```

前五条是网络，后三条是存储。它们有一个共同点：**引擎自己检测能力、自己降级**。NCCL 找不到 IB 设备就用 socket，找不到 peermem 就走 host 内存；PyTorch 的 `dcp.save` 写不完就等。降级不报错，所以平台层必须主动验证"引擎走的是不是我们铺的那条路"。

### 2. K8s 的空缺

原生 Kubernetes 在这一层留下四个洞：

- **一个 Pod 一张网卡**。CNI 规范本身允许多网卡，但 kubelet 只调用一个 CNI 插件、只关心一个 Pod IP；Service、NetworkPolicy、DNS 都围绕这一个 IP。RDMA 网卡要作为"第二张网卡"进入 Pod，需要一个元插件在 kubelet 之下把多个 CNI 调用串起来——这是 Multus 的位置。
- **设备文件不进容器**。容器运行时默认不挂 `/dev/infiniband/*`；Kubernetes 没有"RDMA 设备"这种资源类型，只有 device plugin 这个通用扩展点。谁来发现 HCA、谁决定哪个 Pod 能用、挂哪些设备文件——这是 RDMA device plugin 的位置。
- **内核模块不归 K8s 管**。OFED/DOCA 驱动、`nvidia_peermem`、`rdma_cm` 这些模块要在每个节点上装好并且版本匹配；K8s 没有"节点级驱动"的对象。GPU Operator 用驱动容器解决了 GPU 驱动，Network Operator 用同样的方式解决网卡驱动，两者还要协调顺序。
- **存储只有 PV/PVC 的接口**。CSI 把"挂一个卷"标准化了，但 K8s 不知道这个卷是 Lustre 还是 NFS、带宽多少、能不能被 64 个 Pod 同时写。选型、带宽规划、缓存策略都在 CSI 之上，由平台决定。

### 3. 平台的机制（全局图）

```text
                         ┌─────────────────────────────────────────────────────────────┐
                         │ Pod（训练 rank）                                            │
                         │  eth0 (veth, overlay IP)        net1 (RDMA 侧 IP)           │
                         │  /dev/infiniband/{uverbs0,rdma_cm,...}   /dev/nvidia*       │
                         │  NCCL_IB_HCA / NCCL_SOCKET_IFNAME / NCCL_NET_GDR_LEVEL       │
                         └────┬──────────────┬──────────────────┬─────────────────────┘
                              │              │                  │
   谁给的 ──────────  默认 CNI       Multus + NAD          device plugin
                    (Calico/Cilium…) (host-device/macvlan/    (rdma-shared 或 sriov)
                                      ipoib 三种接入)          经 kubelet Allocate 挂设备文件
                              │              │                  │
   谁装的 ──────────  集群自带      Network Operator NicClusterPolicy：ofedDriver（DOCA-OFED 驱动容器）·
                                   rdmaSharedDevicePlugin / sriovDevicePlugin · secondaryNetwork（multus, cniPlugins, ipoib）· nvIpam
                                            │
                                   GPU Operator ClusterPolicy：driver.rdma.enabled（nvidia-peermem sidecar）· gds · gdrcopy
                                            │
   节点上 ──────────  mlx5_core / ib_uverbs / rdma_cm / nvidia / nvidia_peermem · rdma system netns shared|exclusive
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
   存储 ──────────── CSI 挂载：并行文件系统 PVC（POSIX, 高吞吐） · 对象存储 + 缓存层（JuiceFS/Alluxio/Fluid） · 节点本地 NVMe
                    PyTorch DCP：dcp.save（每 rank 写分片）/ dcp.async_save（先 stage 到 host 内存再后台写）
```

### 4. 本文的章节安排

```text
二、为什么 CNI overlay 不够          veth/overlay 与 /dev/infiniband + GPUDirect RDMA 的两条路径图
三、第二张网卡：Multus 与 NAD        thick/thin · NetworkAttachmentDefinition 与 networks 注解 · host-device / macvlan / IPoIB 三种接入
四、把 RDMA 设备给容器               shared device plugin 的配置与 Allocate · SR-IOV device plugin · 对照表 · netns shared/exclusive · IPC_LOCK
五、Network Operator 与 GPU Operator NicClusterPolicy 字段 · driver.rdma · nvidia-peermem vs DMA-BUF · 部署顺序
六、容器内验证与排障                 ibv_devinfo · rdma link · nccl-tests · NCCL_DEBUG=INFO 判读 · 常见坑表 · 核心问题的三层排查表
七、存储：三类需求与方案定位         数据集 / checkpoint / 权重 × 并行文件系统 / 对象存储 / 缓存层 · CSI 挂载
八、checkpoint I/O 的算术            70B 的状态有多大 · 聚合带宽 · 分片与异步如何降需求 · DCP 的 API
九、推理侧的权重分发                 本地 NVMe 缓存 · 权重进镜像 · P2P · GPUDirect Storage 何时值得
十、代价与边界                       每个机制引入的新问题与不该用的场景
十一、本文小结                       要点 · 四栏表 · 源码/CRD 位置 · mini-platform 本篇增量
```


## 二、为什么 CNI overlay 不够

### 1. 默认路径：veth + overlay

默认 CNI（Calico、Cilium、Flannel 等）给每个 Pod 创建一对 veth，一端在 Pod 的网络命名空间里叫 `eth0`，另一端接在宿主机的网桥或路由表上；跨节点流量再经 VXLAN/IPIP 封装或 BGP 路由。这条路径的每个环节都是内核 TCP/IP 协议栈：

```text
GPU 显存 ──cudaMemcpy──▶ host 内存 ──send()──▶ 内核协议栈 ──veth──▶ 宿主机 ──封装──▶ 物理网卡 ──▶ 对端（反向再来一遍）
```

对 NCCL 的 socket transport 来说，这条路径的带宽上限由 CPU 拷贝和协议栈决定，通常远低于网卡线速；延迟在几十微秒量级。一张 400 Gb/s 的网卡走这条路，能用到的带宽可能只有几分之一，而且要消耗大量 CPU 核。这就是开头那个"三分之一"的来源之一。

### 2. NCCL 想走的路：`/dev/infiniband` + GPUDirect RDMA

NCCL 的 IB transport 需要的是另一条路：

```text
GPU 显存 ◀──DMA──▶ RDMA 网卡（HCA）──▶ 对端 HCA ◀──DMA──▶ 对端 GPU 显存
   │                   ▲
   │  ibv_reg_mr 注册显存（需要 nvidia_peermem 或 DMA-BUF）
   └── 用户态 verbs 库经 /dev/infiniband/uverbsN 与内核 ib_uverbs 通信；rdma_cm 做连接管理
```

内核协议栈完全不参与数据搬运，CPU 只负责下发工作请求。要走这条路，容器里必须有三样东西：

1. **设备文件**：`/dev/infiniband/uverbsN`（对应某个 HCA 端口）、`/dev/infiniband/rdma_cm`，以及 `umadN`/`issmN` 这类管理设备。NCCL 2.28.9 的 `src/transport/net_ib.cc` 在 `ncclIbInit` 里调 `ibv_get_device_list` 枚举设备；容器里没有这些文件，就是 `NET/IB : No device found.`，然后按 `src/plugin/net.cc` 里内部插件的顺序（IB 之后是 Socket）回落到 `NET/Socket`。
2. **GPUDirect RDMA 的内核支持**：`ibGdrSupportInitOnce`（同文件）检查 `/sys/module/nvidia_peermem/version` 或旧的 `nv_mem` 路径；`ncclIbDmaBufSupport` 试探 `ibv_reg_dmabuf_mr` 是否可用。两者都没有，网卡就只能读写 host 内存，数据要在 GPU 与 host 之间多拷一次。
3. **一个 RDMA 侧的 IP**：NCCL 的 IB transport 用一个 IP 接口做带外（OOB）握手（`ncclFindInterfaces`，受 `NCCL_SOCKET_IFNAME` 影响）；RoCE 还要用网卡的 IP 决定 GID（`ncclIbGetGidIndex`）。这个 IP 通常来自 IPoIB 子接口或 RoCE 网卡本身，而不是 overlay。

### 3. 两条路径的对照图

```text
                 overlay（默认 CNI）                          RDMA 直通（Multus + device plugin）
                ────────────────────────────────────         ──────────────────────────────────────────
Pod 网络         eth0 (veth) · Pod CIDR 地址                  eth0 保留 + net1（host-device/macvlan/ipoib）· RDMA 侧地址
设备文件         无                                           /dev/infiniband/{uverbsN,rdma_cm,umadN,...}
数据路径         GPU→host→内核栈→veth→封装→网卡               GPU 显存 ⇄ HCA（GPUDirect RDMA）
CPU 参与         每字节拷贝 + 协议处理                        只下发工作请求
NCCL 日志        "Using network Socket"                       "Using network IB" + "via NET/IB/N/GDRDMA"
隔离             由 NetworkPolicy 管                          RDMA 流量绕过 NetworkPolicy
节点前提         无                                           OFED/DOCA 驱动 · nvidia_peermem 或 DMA-BUF · netns 模式匹配
```

第三章到第五章就是把右边那一列一项项铺出来。


## 三、第二张网卡：Multus 与 NetworkAttachmentDefinition

### 1. Multus 的位置：thin 与 thick

Multus CNI 是一个"元插件"：kubelet 只认它一个 CNI，它再把 ADD/DEL 调用分发给默认网络插件和若干附加网络插件（delegate）。v4.3.0 有两种部署形态（`docs/thick-plugin.md`）：

- **thin**：`/opt/cni/bin/multus` 是一个完整的二进制，每次 CNI 调用都由它直接读 kubeconfig、查 API、调 delegate；部署文件 `deployments/multus-daemonset.yml`。
- **thick**：节点上的 `multus-shim` 只负责把 CNI 参数通过 unix socket 转给常驻的 `multus-daemon`，后者取 `NetworkAttachmentDefinition`、算 `RuntimeConfig`、调 delegate；部署文件 `deployments/multus-daemonset-thick.yml`，daemon 配置默认在 `/etc/cni/net.d/multus.d/daemon-config.json`。

thick 的好处是有了一个常驻进程可以做缓存、指标和更复杂的逻辑，代价是多一个 DaemonSet 和一次本地 RPC。Network Operator 部署的是它自己打包的 Multus 镜像（`secondaryNetwork.multus`），形态以 Operator 版本为准。

Multus 的配置选项（`docs/configuration.md`）里和本篇相关的有：`clusterNetwork` 或 `delegates` 指定默认网络；`namespaceIsolation` 限制 Pod 只能引用本 namespace 的 NAD；`readinessindicatorfile` 让 Multus 等默认网络就绪再工作——GPU 训练 Pod 起不来时，先看这个文件是否存在。

### 2. `NetworkAttachmentDefinition` 与三个注解

附加网络用 CRD `NetworkAttachmentDefinition`（API 组 `k8s.cni.cncf.io/v1`，Network Plumbing WG 的事实标准）描述，`spec.config` 就是一段 CNI 配置 JSON。Pod 用注解引用它。Multus v4.3.0 `pkg/k8sclient/k8sclient.go` 里定义了三个关键注解：

```text
k8s.v1.cni.cncf.io/networks          Pod 上：要挂哪些附加网络。逗号分隔的名字，或 JSON 数组（可带 interface、ips、mac、gateway）
k8s.v1.cni.cncf.io/resourceName      NAD 上：这个网络要消耗哪个 device plugin 资源；Multus 据此把 kubelet 分配的设备信息传给 delegate
k8s.v1.cni.cncf.io/network-status    Pod 上（Multus 写回）：每个接口的名字、IP、MAC、是否默认路由
```

`networks` 注解跨 namespace 引用写成 `<namespace>/<name>`；`resourceName` 把 NAD 与第四章的 device plugin 关联起来——Pod 请求了 `nvidia.com/hostdev: 1`，kubelet 分配了某个 PCI 设备，Multus 从 kubelet 的设备分配记录（checkpoint 文件或 PodResources API）读到结果，再以 `deviceID` 传给 `host-device` 或 `sriov` CNI，让 CNI 知道要把哪张网卡搬进 Pod。

### 3. 三种接入方式

RDMA 网卡进 Pod 的第二张网卡有三种常见接法。它们决定的是**IP 接口**怎么进 Pod；RDMA 设备文件由第四章的 device plugin 负责，两者是正交的。

```text
接入方式       CNI 类型       进 Pod 的是什么                       独占性              适用
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
host-device    host-device    整个 PF（或一个 VF）netdev 被 move 进   一个 netdev 一个 Pod  SR-IOV VF；或每节点 Pod 数 ≤ 网卡数的训练
                              Pod netns，宿主机上看不到它了
macvlan        macvlan        在以太网 PF 上创建的 macvlan 子接口，   多个 Pod 共享 PF      RoCE；多个 Pod 共享一张网卡的 IP 平面
                              自己的 MAC 与 IP
IPoIB          ipoib          在 IB PF 上创建的 IPoIB 子接口          多个 Pod 共享 PF      InfiniBand；NCCL 只用它做握手，数据走 verbs
（第四种）     sriov          分配到的 VF，带 VLAN/MAC 配置           一个 VF 一个 Pod      需要强隔离或 exclusive netns 的场景
```

Network Operator 用三个 CRD 把前三种包装成"填几个字段就生成 NAD"：

- `HostDeviceNetwork`（`api/v1alpha1/hostdevicenetwork_types.go`）：`spec.networkNamespace`、`spec.resourceName`、`spec.ipam`；生成的 NAD 模板在 `manifests/state-hostdevice-network/0010-hostdevice-net-cr.yml`，`type: host-device`，并带 `k8s.v1.cni.cncf.io/resourceName` 注解指向 `sriovDevicePlugin` 上报的资源；
- `MacvlanNetwork`（`macvlannetwork_types.go`）：`spec.master`、`spec.mode`、`spec.mtu`、`spec.ipam`；
- `IPoIBNetwork`（`ipoibnetwork_types.go`）：`spec.master`、`spec.ipam`；NAD 模板 `manifests/state-ipoib-network/0010-ipoib-net-cr.yaml`，`type: ipoib`。

`ipam` 字段是一段 JSON，Network Operator 的例子用它自己的 `nv-ipam`（`{"type":"nv-ipam","poolName":"my-pool"}`，由 `NicClusterPolicy.spec.nvIpam` 部署），也可以用 `whereabouts` 或 `static`。

直接写 NAD 也完全可以，这是 mini-platform 的做法（第十一章 `net/nad-ipoib.yaml`，IB 集群 + shared device plugin 的组合）：

```yaml
apiVersion: k8s.cni.cncf.io/v1
kind: NetworkAttachmentDefinition
metadata:
  name: rdma-net
  namespace: default
spec:
  config: '{
    "cniVersion": "0.3.1",
    "name": "rdma-net",
    "type": "ipoib",
    "master": "ibs3f1",
    "ipam": {
      "type": "whereabouts",
      "range": "192.168.100.0/24"
    }
  }'
```

`master` 是宿主机上的 IPoIB 接口名，各节点要一致（或按节点分别建 NAD）。RoCE 集群把 `type` 换成 `macvlan`、`master` 换成以太网接口、加 `"mode": "bridge"`。host-device 变体（`net/nad-hostdevice.yaml`）不写 `master`，而是加注解 `k8s.v1.cni.cncf.io/resourceName: nvidia.com/hostdev`，由 CNI 从 device plugin 的分配结果里得知要搬哪张网卡——它要求 `NicClusterPolicy` 里配了 `sriovDevicePlugin`（资源名 `nvidia.com/hostdev`，第四章第 2 节），Pod 也要请求这个资源。Pod 侧（片段）：

```yaml
metadata:
  annotations:
    k8s.v1.cni.cncf.io/networks: rdma-net
```

多张网卡（每 GPU 一张的 8 卡节点）写成 `rdma-net-0,rdma-net-1,...` 或 JSON 数组并指定 `interface` 名，Pod 里就会出现 `net1`…`net8`。


## 四、把 RDMA 设备给容器

### 1. RDMA shared device plugin：一个 HCA 给很多 Pod

k8s-rdma-shared-dev-plugin v1.5.4 的模型很简单：把宿主机上的每个 RDMA 网卡（按 selector 过滤后）当作一个"共享设备"，对外上报 `rdmaHcaMax` 个同名资源副本；任何 Pod 请求一个副本，就把这组网卡的**全部** RDMA 字符设备挂进容器。配置文件（默认 `/k8s-rdma-shared-dev-plugin/config.json`，由 ConfigMap 挂入）的结构在 `pkg/types/types.go`：

```text
UserConfigList
  periodicUpdateInterval   秒；0 关闭；不设默认 60（README）
  configList[]             每项一个资源
    resourceName           资源名后缀，如 rdma_shared_device_a
    resourcePrefix         默认 "rdma"（resources_manager.go 的 rdmaHcaResourcePrefix）→ 资源名 rdma/rdma_shared_device_a
    rdmaHcaMax             上报多少个副本；每个 Pod 请求 1 就是"最多 rdmaHcaMax 个 Pod 共用"
    devices[]              按接口名选网卡（等价于 selectors.ifNames）
    selectors              vendors / deviceIDs / drivers / ifNames / linkTypes；同一 selector 内 OR，selector 之间 AND
```

Network Operator 的例子（`example/crs/mellanox.com_v1alpha1_nicclusterpolicy_cr.yaml`）是 `resourceName: rdma_shared_device_a`，`rdmaHcaMax: 63`，`selectors.vendors: ["15b3"]`、`deviceIDs: ["101b"]`——15b3 是 Mellanox 的 PCI vendor ID，101b 是 ConnectX-6 的 device ID。这就是总纲里提到的 `rdma/rdma_shared_device_a` 的出处。

分配时发生什么（`pkg/resources/server.go`）：`Allocate` 对每个容器返回同一份 `deviceSpec`——非 CDI 模式下是 `ContainerAllocateResponse.Devices`，CDI 模式下（`useCdi`）是以 `nvidia.com` 为前缀的 CDI 注解。`deviceSpec` 由 `getDevicesSpec` 汇总所有匹配网卡的 `GetRdmaSpec()`，而 `pkg/resources/rdma_device_spec.go` 的 `Get` 用 `utils.GetRdmaDevices(pciAddress)` 从 PCI 地址找到 RDMA 设备再列出其字符设备，每个以 `HostPath == ContainerPath`、权限 `rwm` 挂入。`VerifyRdmaSpec` 要求 `rdma_cm`、`umad`、`uverbs` 三类设备都在（`requiredRdmaDevices`）。也就是说，Pod 请求 `rdma/rdma_shared_device_a: 1` 之后，容器里会出现该资源下所有网卡的 `/dev/infiniband/uverbsN`、`umadN`、`issmN` 与共享的 `/dev/infiniband/rdma_cm`。

两点常被忽略：

- 它**没有**做 GPU 亲和。一个 8 卡 8 网卡的节点配成一个资源，Pod 拿到的是 8 张网卡的设备文件，由 NCCL 自己按 PCIe 距离选卡；想让 Pod 只看到某几张，要拆成多个 `configList` 项（用 `ifNames` 分组）并请求对应的资源名。
- 它**要求** RDMA 子系统处于 shared netns 模式：`resources_manager.go` 的 `ValidateRdmaSystemMode` 调 `netlink.RdmaSystemGetNetnsMode`，返回 `exclusive` 就报错退出（见本章第 4 节）。

### 2. SR-IOV device plugin：一个 VF 给一个 Pod

另一条路是把物理网卡切成 SR-IOV 虚拟功能（VF），每个 VF 有独立的 PCI 地址、独立的 RDMA 设备，由 SR-IOV network device plugin 作为独立资源上报，一个 Pod 拿一个。Network Operator 通过 `NicClusterPolicy.spec.sriovDevicePlugin` 部署它，配置格式是那个插件自己的 `resourceList`（`example/crs/mellanox.com_v1alpha1_nicclusterpolicy_cr-full.yaml`）：

```json
{
  "resourceList": [
    {
      "resourcePrefix": "nvidia.com",
      "resourceName": "hostdev",
      "selectors": { "vendors": ["15b3"], "isRdma": true }
    }
  ]
}
```

`isRdma: true` 让插件在分配 VF 时一并挂入该 VF 的 RDMA 字符设备。VF 的创建本身不在这个插件里——由 SR-IOV Network Operator 的 `SriovNetworkNodePolicy`（`numVfs`、`nicSelector`、`isRdma`，`example/sriov-network/`）或节点脚本完成；SR-IOV network device plugin 与 SR-IOV Network Operator 不在本篇的检出范围内，字段以 Network Operator v26.7.0 的示例为准。

### 3. 两种插件的对照

```text
                    rdma-shared-dev-plugin                          SR-IOV device plugin（+ VF）
─────────────────────────────────────────────────────────────────────────────────────────────────────────
资源粒度            一个 PF（或一组 PF）× rdmaHcaMax 个副本           一个 VF 一个资源单位
Pod 拿到什么        该组所有 PF 的 /dev/infiniband/*                  自己那个 VF 的 /dev/infiniband/*（isRdma）
隔离                无：同节点 Pod 共享 HCA 的队列、带宽、错误域      PCI 级：VF 有自己的队列与 QP 配额；带宽仍共享物理端口
netns 模式          要求 shared                                       通常配 exclusive（rdma-cni 把 VF 的 RDMA 设备搬进 Pod netns）
IP 接口             正交：另配 macvlan/ipoib/host-device              通常 sriov CNI 直接把 VF netdev 搬进 Pod
每节点 Pod 数上限   rdmaHcaMax（软限制）                              VF 数（硬件与固件上限；每 PF 的 VF 数以 NVIDIA 文档为准）
GPU 亲和            靠 NCCL 自选或拆多个资源                          可以按 PF 分资源池，Pod 请求对应池
虚拟机 / 迁移       不涉及                                            VF 直通的 VM 无法热迁移；容器场景无此约束但 VF 重配要清空 Pod
适用                训练集群：节点独占（一个 8 卡任务占满节点），       多租户共享节点、要审计流量、要 exclusive netns 的场景；
                    简单可靠，NCCL 自己看到全部网卡                    每 Pod 一两张卡的推理或开发环境
```

训练集群的主流做法是 shared plugin：训练任务通常独占节点，隔离是多余的，而 NCCL 看到全部网卡反而更好选。SR-IOV 的价值在多租户和网络策略执行，代价是 VF 配置、固件依赖与更多的运维面。

### 4. RDMA 子系统的 netns 模式：shared 与 exclusive

Linux RDMA 子系统有一个全局开关，决定 RDMA 设备与网络命名空间的关系（iproute2 的 `rdma` 工具，属通用知识）：

```bash
rdma system show                 # netns shared | exclusive
rdma system set netns shared     # 所有 netns 都能看到所有 RDMA 设备
rdma system set netns exclusive  # RDMA 设备只属于一个 netns，需显式 rdma dev set <dev> netns <ns>
```

- **shared**（默认）：容器只要有设备文件就能用任何 HCA；RDMA 流量与 netns 无关，所以 NetworkPolicy 管不到它。shared-dev-plugin 只在这个模式下工作。
- **exclusive**：每个 RDMA 设备归属一个 netns；配合 SR-IOV，rdma-cni 在 Pod 创建时把 VF 对应的 RDMA 设备 move 进 Pod 的 netns，其他 Pod 即使有设备文件也看不到它。切换到 exclusive 要求当前没有 RDMA 设备被 netns 之外的用户占用，实践中在节点初始化、加载驱动之后立即设置；Network Operator 的 OFED 驱动容器挂了 `/etc/modprobe.d/ib_core.conf`（`manifests/state-ofed-driver/0050_ofed-driver-ds.yaml`），`ib_core` 的模块参数也可以固定这个模式（参数名以内核/OFED 文档为准）。

### 5. 容器还需要什么：`IPC_LOCK` 与 memlock

RDMA 注册内存需要把页锁定，容器默认的 `RLIMIT_MEMLOCK` 通常很小。所有官方示例（rdma-shared-dev-plugin `example/test-hca-pod.yaml`、Network Operator `example/rdma-gpu-test-pod1.yml`）都给容器加了 `securityContext.capabilities.add: ["IPC_LOCK"]`。少了它，`ibv_reg_mr` 会在注册大块内存时失败，NCCL 的表现是初始化时报 `Call to ibv_reg_mr failed with error …`（`src/misc/ibvwrap.cc` 的 `IBV_PTR_CHECK_ERRNO`）或直接回落。这是第六章排查清单里最容易漏的一项。


## 五、Network Operator 与 GPU Operator 的配合

### 1. `NicClusterPolicy`：一个 CR 部署整条网络栈

NVIDIA Network Operator v26.7.0 的核心 CRD 是 `NicClusterPolicy`（API 组 `mellanox.com/v1alpha1`，`api/v1alpha1/nicclusterpolicy_types.go`）。它是集群级（`scope=Cluster`）单例，控制器只处理名为 `nic-cluster-policy` 的那一个实例（`pkg/consts/consts.go` 的 `NicClusterPolicyResourceName`，`controllers/nicclusterpolicy_controller.go` 据此忽略其他名字）。`NicClusterPolicySpec` 的字段与本篇相关的部分：

```text
spec.ofedDriver                  OFEDDriverSpec：DOCA-OFED 驱动容器（image/repository/version，upgradePolicy、探针、env、
                                 certConfig/repoConfig、forcePrecompiled、terminationGracePeriodSeconds）
spec.rdmaSharedDevicePlugin      DevicePluginSpec：镜像 + config（第四章的 JSON 原样放进去）+ useCdi
spec.sriovDevicePlugin           DevicePluginSpec：镜像 + config（resourceList JSON）+ useCdi
spec.secondaryNetwork            SecondaryNetworkSpec：multus（MultusSpec，可带 config）· cniPlugins（containernetworking 插件集）· ipoib（IPoIB CNI）
spec.nvIpam                      NVIPAMSpec：nv-ipam 控制器与 CNI，enableWebhook
spec.ibKubernetes                IBKubernetesSpec：IB 分区（PKey/GUID）管理，pKeyGUIDPoolRangeStart/End、ufmSecret
spec.nicFeatureDiscovery         NICFeatureDiscoverySpec：配合 NFD 打网卡标签
spec.nicConfigurationOperator    NicConfigurationOperatorSpec：网卡固件参数
spec.docaTelemetryService        DOCATelemetryServiceSpec
spec.nodeAffinity / tolerations  DaemonSet 的调度约束
```

注意 v26.7.0 **没有** `docaDriver` 字段：DOCA-OFED 驱动容器仍由 `ofedDriver` 配置（类型注释写的是 "DOCA-OFED Driver Container"），示例镜像是 `nvcr.io/nvidia/mellanox/doca-driver:doca3.5.0-26.07-0.7.7.0-0`。`ofedDriver` 的 `Env` 可以传 `docs/mofed-container-env-vars.md` 列出的变量，例如 `UNLOAD_THIRD_PARTY_RDMA_MODULES`。`OFEDDriverSpec` 里的 `upgradePolicy`（`autoUpgrade`、`maxParallelUpgrades`、`drain`、`safeLoad`、`waitForCompletion`）决定驱动升级时怎么排空节点——驱动升级要重载 `mlx5_core`，所有用着网卡的 Pod 都会断，这是训练集群最需要谨慎的运维动作之一（`docs/host-ofed.md` 与 `docs/automatic-ofed-upgrade.md`）。

一份最小的 `NicClusterPolicy`（mini-platform 的 `net/nicclusterpolicy.yaml`）：

```yaml
apiVersion: mellanox.com/v1alpha1
kind: NicClusterPolicy
metadata:
  name: nic-cluster-policy
spec:
  ofedDriver:
    image: doca-driver
    repository: nvcr.io/nvidia/mellanox
    version: doca3.5.0-26.07-0.7.7.0-0
    forcePrecompiled: false
    upgradePolicy:
      autoUpgrade: true
      maxParallelUpgrades: 1
      drain:
        enable: true
        force: true
        deleteEmptyDir: true
        timeoutSeconds: 300
  rdmaSharedDevicePlugin:
    image: k8s-rdma-shared-dev-plugin
    repository: nvcr.io/nvidia/mellanox
    version: network-operator-v26.7.0
    config: |
      {
        "configList": [
          {
            "resourceName": "rdma_shared_device_a",
            "rdmaHcaMax": 63,
            "selectors": {
              "vendors": ["15b3"],
              "linkTypes": ["infiniband"]
            }
          }
        ]
      }
  secondaryNetwork:
    multus:
      image: multus-cni
      repository: nvcr.io/nvidia/mellanox
      version: network-operator-v26.7.0
    cniPlugins:
      image: plugins
      repository: nvcr.io/nvidia/mellanox
      version: network-operator-v26.7.0
    ipoib:
      image: ipoib-cni
      repository: nvcr.io/nvidia/mellanox
      version: network-operator-v26.7.0
  nvIpam:
    image: nvidia-k8s-ipam
    repository: nvcr.io/nvidia/mellanox
    version: network-operator-v26.7.0
    enableWebhook: false
```

镜像 tag 来自仓库示例，实际以随 Operator 发布的 values 为准。`rdmaSharedDevicePlugin.config` 与 `sriovDevicePlugin.config` 都是"原样透传"的字符串，Operator 只负责生成 ConfigMap 和 DaemonSet（`manifests/state-rdma-shared-device-plugin/`），不校验里面的 JSON——写错了要去 device plugin 的日志里找。

### 2. GPU Operator 的一半：`driver.rdma`

GPUDirect RDMA 需要 GPU 驱动侧的 `nvidia_peermem` 模块（把显存作为 RDMA 可注册的 peer memory 暴露给 `ib_core`），它属于 GPU 驱动包而不是网卡驱动。GPU Operator v26.7.0 的 `ClusterPolicy`（`api/nvidia/v1/clusterpolicy_types.go`）在 `DriverSpec` 里有：

```text
spec.driver.rdma.enabled          GPUDirectRDMASpec.Enabled：在驱动 DaemonSet 里保留 nvidia-peermem-ctr sidecar，
                                  并给驱动容器设 GPU_DIRECT_RDMA_ENABLED=true（controllers/object_controls.go）
spec.driver.rdma.useHostMofed     GPUDirectRDMASpec.UseHostMOFED：OFED 装在宿主机而不是 Network Operator 容器里时，
                                  把 /usr/src 从宿主机挂进来编 peermem，并设 USE_HOST_MOFED=true
spec.gds.enabled                  GPUDirectStorageSpec：nvidia-fs 驱动容器（实验性），device plugin 加 GDS_ENABLED 与 MOFED_ENABLED
                                  以注入 /dev/nvidia-fs 与 /dev/infiniband
spec.gdrcopy.enabled              GDRCopySpec：gdrdrv 驱动
```

驱动 DaemonSet 模板 `assets/state-driver/0500_daemonset.yaml` 里可以看到这套机制的形状：`nvidia-peermem-ctr` 容器运行 `nvidia-driver reload_nvidia_peermem`，探针是 `nvidia-driver probe_nvidia_peermem`；它挂了 `/run/mellanox/drivers`（`mountPropagation: HostToContainer`）——这正是 Network Operator 的 OFED 容器把驱动树发布到宿主机的路径（`manifests/state-ofed-driver/0050_ofed-driver-ds.yaml` 同样挂 `/run/mellanox/drivers`）。`transformPeerMemoryContainer` 在 `rdma.enabled` 为假时把这个 sidecar 从 DaemonSet 里删掉。validator（`cmd/nvidia-validator/main.go`）用 `lsmod | grep nvidia_peermem` 确认模块加载并写 `nvidia-peermem-ready` 状态文件。

### 3. 两个 Operator 的顺序

两者的依赖关系是：**nvidia_peermem 要针对当前的 OFED 内核模块编译并在其后加载**。所以推荐顺序是 Network Operator 的 OFED 驱动先就位，GPU Operator 的驱动容器再启动；OFED 升级重载后，`nvidia-peermem-ctr` 的 `reload_nvidia_peermem` 负责重新加载。Network Operator 用节点标签 `network.nvidia.com/operator.mofed.wait` 表达"OFED 尚未就绪"（`controllers/mofed_wait_labels.go` 设置；`pkg/nodeinfo/attributes.go` 的 `NodeLabelWaitOFED`），GPU Operator 在 `internal/nodeinfo/attributes.go` 里定义了同名常量；这个标签也出现在 Network Operator Helm values 的 `configDaemonNodeSelector` 里。两个 Operator 各自的 Helm 参数与安装顺序以 NVIDIA 文档为准，本篇只说明机制。

### 4. `nvidia_peermem` 与 DMA-BUF

GPUDirect RDMA 有两条内核路径，NCCL 2.28.9 两条都探测：

```text
                     nvidia_peermem                                   DMA-BUF
──────────────────────────────────────────────────────────────────────────────────────────────────────────
机制                 NVIDIA 驱动附带的模块，向 ib_core 注册 peer       内核通用的 dma-buf 框架：CUDA 导出显存为 dma-buf fd，
                     memory client，ib_core 注册显存时回调它             网卡驱动用 ibv_reg_dmabuf_mr 注册
需要什么             nvidia_peermem 模块（GPU Operator driver.rdma）    支持 dma-buf 的内核、驱动、rdma-core、开源 GPU 内核模块或
                     + OFED 的 ib_core                                  同等支持（版本以 NVIDIA 文档为准）
NCCL 如何探测        ibGdrSupportInitOnce 查 /sys/module/nvidia_peermem/  ncclIbDmaBufSupport 试探 ibv_reg_dmabuf_mr；
                     version（net_ib.cc）→ ptrSupport |= NCCL_PTR_CUDA    NCCL_DMABUF_ENABLE（init.cc，默认 1）控制是否使用
容器里怎么看         宿主机 lsmod | grep nvidia_peermem                  NCCL_DEBUG=INFO 里 GDRDMA 仍然出现，但 lsmod 无 peermem
运维                 OFED 升级后要重载；版本要匹配                       无额外模块，但对内核与驱动版本的下限更高
```

对平台来说两者是可替换的：只要 NCCL 日志里出现 `via NET/IB/N/GDRDMA` 就说明 GPUDirect RDMA 生效了；`paths.cc` 的 `GPU Direct RDMA Disabled for GPU … (distance N > M)` 则说明模块在但拓扑距离超过了 `NCCL_NET_GDR_LEVEL`。


## 六、容器内验证与排障

### 1. 验证清单

在一个请求了 RDMA 资源的 Pod 里，按这个顺序验证，每一步不通就不用看下一步：

```text
步骤  命令                                          期望                                        不满足说明
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
1     ls /dev/infiniband                             uverbsN umadN issmN rdma_cm                 device plugin 没分配到；查 resources.limits 与节点 Allocatable
2     ibv_devinfo                                    列出 HCA，state PORT_ACTIVE，link_layer     设备文件在但 verbs 库读不到：驱动/rdma-core 版本；端口 DOWN 查交换机
3     rdma link                                      每个 HCA 端口 state ACTIVE，netdev 对应关系  exclusive 模式下看不到设备：netns 模式与插件不匹配
4     ip -d addr                                     除 eth0 外有 net1（IPoIB/macvlan/host-device）   Multus 注解没生效：看 network-status 注解与 multus 日志
5     ulimit -l                                      unlimited                                   缺 IPC_LOCK；ibv_reg_mr 会失败
6     ib_write_bw（perftest，两 Pod 之间）           接近网卡线速                                 单纯 RDMA 层就慢：GID/路由/交换机；与 NCCL 无关
7     all_reduce_perf + NCCL_DEBUG=INFO              "Using network IB"、"via NET/IB/…/GDRDMA"    见第 3、4 节
```

第 1～5 步在没有 GPU 的节点上也能做，是平台层的责任；第 6 步验证 RDMA 网络本身；第 7 步才轮到 NCCL。

### 2. 容器里跑 nccl-tests

nccl-tests 2.18.3 的多机模式靠 MPI（`src/common.cu` 用 `MPI_Comm_rank/size` 定 rank、`MPI_Bcast` 分发 `ncclUniqueId`），mini-platform 的 `net/nccl-tests-job.yaml` 用一个 Indexed Job 起两个 Pod：index 1 跑 sshd，index 0 等对端就绪后 `mpirun`。生产上这一步由 MPI Operator 或 Kubeflow Trainer 的 MPI runtime 自动化，这里手写是为了看清每一层。关键片段：

```yaml
apiVersion: v1
kind: Service
metadata:
  name: nccl-tests
spec:
  clusterIP: None                   # headless：Pod 得到 nccl-tests-<index>.nccl-tests 的稳定 DNS
  selector:
    job-name: nccl-tests
---
apiVersion: batch/v1
kind: Job
metadata:
  name: nccl-tests
spec:
  completionMode: Indexed
  completions: 2
  parallelism: 2
  template:
    metadata:
      annotations:
        k8s.v1.cni.cncf.io/networks: rdma-net     # 第三章的 NAD；Pod 里出现 net1
    spec:
      subdomain: nccl-tests
      restartPolicy: Never
      containers:
      - name: nccl
        image: registry.example.com/nccl-tests:2.18.3-mpi   # 自建：nccl-tests make MPI=1 + openssh
        securityContext:
          capabilities:
            add: ["IPC_LOCK"]
        resources:
          limits:
            nvidia.com/gpu: 8
            rdma/rdma_shared_device_a: 1
        env:
        - name: NCCL_DEBUG
          value: INFO
        - name: NCCL_DEBUG_SUBSYS
          value: INIT,NET
        - name: NCCL_SOCKET_IFNAME
          value: net1                 # Multus 挂进来的 RDMA 侧接口，不是 eth0
        # - name: NCCL_IB_HCA
        #   value: mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
      # ... 省略：ssh 密钥 Secret 的挂载、按 JOB_COMPLETION_INDEX 分支的 command（1 起 sshd，0 跑 mpirun）、
      #     podAntiAffinity 让两 Pod 落在不同节点
```

index 0 最终执行的命令与裸机一致：

```bash
mpirun -np 16 -N 8 -H nccl-tests-0.nccl-tests:8,nccl-tests-1.nccl-tests:8 \
  -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS -x NCCL_SOCKET_IFNAME -x NCCL_IB_HCA -x LD_LIBRARY_PATH \
  ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 1 -n 20 -w 5 -c 0 -T 600
```

`-T 600` 让网络不通时得到一个错误而不是一个永远不退出的 Job。

**没有 RDMA 网卡的集群**（总纲允许的"host network 模拟"）：同一份 Job 去掉 `rdma/...` 请求与 `networks` 注解，改成 `hostNetwork: true` 与 `hostIPC: true`，`NCCL_SOCKET_IFNAME` 指向宿主机的物理网卡名。这样测出的是 socket 路径的数字，意义在于走通"两 Pod 的 NCCL 能互相发现并完成 all_reduce"这一整条流程，并把 `NET/Socket` 的日志形态看一遍——之后在真 RDMA 集群上看到 `NET/Socket` 就知道是回落了。如果宿主机有 RDMA 网卡但没装 device plugin，再加一个 `hostPath: /dev/infiniband` 的挂载与 `privileged: true`，就是"不经 device plugin 的直通"，只适合验证，不适合生产。

### 3. 读 `NCCL_DEBUG=INFO`：只看"走了哪条路"

本篇只需要 INFO 日志里的这几行（`NCCL_DEBUG_SUBSYS=INIT,NET` 足够）：

```text
行                                                                    出处（NCCL 2.28.9）             含义
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_1:1/IB … ; OOB net1:192.168.100.11<0>  src/transport/net_ib.cc   IB transport 初始化成功；列出全部 HCA、每个的链路层
                                                                                                         （IB 或 RoCE）与 OOB 接口。设备数 < 期望 → 少挂了网卡
NET/IB : No device found.                                             src/transport/net_ib.cc          容器里没有 uverbs 设备 → 之后会看到下一行的 Socket
NET/Socket : Using [0]eth0:10.244.2.6<0>                              src/transport/net_socket.cc      socket transport 用的接口。若这是最终选择，就是回落
Using network IB   /   Using network Socket                           src/init.cc                      最终选择。看到 Socket 就不用再看带宽了
Channel 00/0 : 0[0] -> 8[0] [send] via NET/IB/0/GDRDMA                src/transport/net.cc             每个跨机连接用的网卡编号；有 /GDRDMA 说明 GPUDirect RDMA 生效，
                                                                                                         没有 /GDRDMA 就是经 host 内存中转；(PCI) 后缀是 GDR 的 PCI 模式
GPU Direct RDMA Disabled for GPU 3 / HCA … (distance 5 > 3)           src/graph/paths.cc               peermem/DMA-BUF 在但拓扑距离超过 NCCL_NET_GDR_LEVEL
```

只要第一行列出了正确数量的 HCA、`Using network IB`、每个 channel 都有 `GDRDMA`，网络这一层就是通的；再慢就不是平台的问题，而是第三章之外的调优范围。

### 4. 常见的坑

```text
现象                                       原因                                                    怎么定位 / 修
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Using network Socket；带宽是裸机的几分之一   Pod 没请求 RDMA 资源，或节点 Allocatable 为 0            ls /dev/infiniband；kubectl describe node 看 rdma/* 资源；
                                            → 容器无 uverbs → "No device found" → 回落                device plugin 日志（selector 没匹配到网卡最常见）
Using network IB 但 HCA 数少于预期           NCCL_IB_HCA 写错：名字是 mlx5_N 不是 netdev 名；          看 "NET/IB : Using" 那一行列了哪些；ibv_devinfo 对照；
                                            写成 ^ 排除语法反了；或 shared plugin 的 selector 只选了部分   NCCL_IB_HCA 支持前缀匹配、^ 排除与 = 精确（net_ib.cc）
初始化卡住或报 No IP interface found         NCCL_SOCKET_IFNAME 指向 Pod 里不存在的接口                容器里 ip addr 看真实名字（Multus 默认叫 net1，不是宿主机名）
                                            （复制了宿主机上的 ens1f0）
握手慢、偶发 timeout                        NCCL_SOCKET_IFNAME 没设，NCCL 选了 eth0（overlay）做 OOB；   显式设为 RDMA 侧接口；确认两 Pod 的 net1 互通（ping）
                                            overlay 的 MTU/NAT 有时会拖慢或阻断
RoCE：ibv_devinfo 正常但 NCCL 连不上         GID 选错：IPv4 vs IPv6、RoCE v1 vs v2                     2.28.9 默认按 NCCL_IB_ROCE_VERSION_NUM=2 与 NCCL_IB_ADDR_FAMILY
                                                                                                    自动选（ncclIbGetGidIndex）；不对时用 NCCL_IB_GID_INDEX 固定；
                                                                                                    show_gids（MLNX_OFED/DOCA 附带脚本）看每个 index 对应的地址
IB 通、无 GDRDMA、带宽只有一半左右           nvidia_peermem 没加载 / 版本不匹配 / GPU Operator 未开 rdma   宿主机 lsmod；ClusterPolicy spec.driver.rdma.enabled；
                                                                                                    validator 的 nvidia-peermem-ready 状态
部分 GPU 的 GDRDMA 被 Disabled               网卡与 GPU 跨 NUMA/跨 PCIe switch                          nvidia-smi topo -m；这是硬件布局或 NCCL_NET_GDR_LEVEL 的问题
Call to ibv_reg_mr failed with error …       缺 IPC_LOCK / memlock 太小                               ulimit -l；securityContext.capabilities.add IPC_LOCK
Pod Pending，事件 Insufficient rdma/…         rdmaHcaMax 太小，或 device plugin 未在该节点运行            调大 rdmaHcaMax；看 DaemonSet 的 nodeSelector 与 NFD 标签
rdma-shared-dev-plugin 启动即退出             节点处于 exclusive netns 模式                             rdma system show；ValidateRdmaSystemMode 只接受 shared
```

### 5. 回答核心问题：三层排查表

回到开头：8 节点 64 卡，容器里 all_reduce 带宽只有裸机的三分之一。按总纲的三个层面：

```text
层面                  可能的问题                                       证据在哪                                    修在哪
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Pod 的网络配置        没有第二张网卡：忘了 networks 注解，或 NAD 不在      kubectl get pod -o yaml 看 network-status；   加注解；NAD 放对 namespace 或关 namespaceIsolation；
                      Pod 的 namespace（namespaceIsolation）             容器里 ip addr 只有 eth0                     检查 multus 的 readinessindicatorfile
                      第二张网卡在但不通：IPAM 冲突、子网不同、交换机 VLAN  两 Pod 之间 ping net1；ib_write_bw            修 NAD 的 ipam；IPoIB 的 pkey；RoCE 的 VLAN
                      NCCL 的 OOB 走了 overlay                          日志 "OOB eth0:10.244…"                      NCCL_SOCKET_IFNAME=net1
device plugin 的分配  Pod 根本没请求 rdma/…；或请求了但节点上报 0          ls /dev/infiniband 为空；describe node        加 resources.limits；修 selector（vendors/deviceIDs/
                                                                                                                    ifNames）；确认 DaemonSet 跑在这些节点
                      只分到了一部分网卡（selector 按 ifNames 只选了几张） "NET/IB : Using" 列出的 HCA 少于 8           一个资源覆盖全部网卡，或按 GPU 亲和拆资源并让 Pod 请求全部
                      少了 IPC_LOCK                                    "Call to ibv_reg_mr failed"                  securityContext
                      节点是 exclusive 模式但用了 shared plugin           plugin 起不来 / 容器里 rdma link 为空          统一 netns 模式与插件类型
NCCL 的环境变量       NCCL_IB_HCA 名字错或排除反了                         "Using" 行 HCA 数不对                        按 ibv_devinfo 的名字写；注意 ^ 与 = 语义
                      NCCL_SOCKET_IFNAME 是宿主机的接口名                  "No IP interface found"                     改为容器内名字
                      NCCL_NET_GDR_LEVEL 太严或 peermem 没加载             "GPU Direct RDMA Disabled" / 无 GDRDMA        开 driver.rdma.enabled；按拓扑调 GDR_LEVEL
                      RoCE 的 GID 自动选择不对                             连接错误、或带宽极低                          NCCL_IB_GID_INDEX 固定
```

"三分之一"这个数字本身也有信息量：如果是 socket 回落，通常比三分之一还差得多（而且 CPU 打满）；如果是三分之一到一半，更像是 IB 通了但没有 GDRDMA（数据多经一次 host 内存），或者 8 张网卡只用上了 2～3 张（selector 只选了部分、或 NCCL_IB_HCA 写漏了）。先看 `Using network` 那一行，再数 `NET/IB : Using` 列出的设备，两个信息就能把范围缩到一层。


## 七、存储：三类需求与方案定位

### 1. 三类负载，三种 I/O 画像

训练与推理对存储提出三类完全不同的要求，混在一个"共享存储"上是大多数存储问题的根源：

```text
负载            访问模式                          规模                         对存储的要求
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
数据集读取      海量小文件（或大文件内随机块）、    TB～PB；每 epoch 全量读一遍     元数据性能（open/stat）· 随机读 IOPS · 可缓存
                多进程并发随机读、只读                                            （数据不变，节点本地缓存收益高）
checkpoint 写   每 rank 一个大文件顺序写、突发、     每次几百 GB～几 TB；每 30 分钟   聚合写吞吐（第八章算）· 写完即可，不要求低延迟
                写完之后很少读（恢复时读一次）       一次                          · 保留最近 N 份就删
权重分发        一次写、多次读；推理扩容时 N 个副本  每个模型几十～几百 GB           并发读吞吐 · 冷启动延迟 · 热点文件（同一份权重）
                同时读同一份文件
```

### 2. 方案定位

```text
方案类别               代表（公开文档）                     接口      擅长                              不擅长 / 代价
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
并行文件系统           Lustre · IBM Storage Scale (GPFS) ·   POSIX     高聚合吞吐、大文件顺序读写、        小文件元数据（Lustre 尤甚）；运维重；
                       WEKA · BeeGFS                                  多客户端并发写 → checkpoint 的首选   价格高；需要专用客户端/内核模块
对象存储               S3 及兼容实现（MinIO、云厂商）       S3 API    便宜、无限扩展、跨集群共享、          不是 POSIX；小对象延迟高；
                                                                     权重与数据集的"源"                    直接当训练读路径太慢
缓存 / 加速层          JuiceFS · Alluxio · Fluid（K8s 数据编排）  POSIX（FUSE/CSI） 把对象存储包装成有本地缓存的文件系统；   FUSE 开销；缓存一致性；
                                                                     数据集读与权重分发的首选                首次读仍受对象存储限制；元数据服务是新的单点
节点本地 NVMe          hostPath / local PV / emptyDir         POSIX     最高带宽与 IOPS，零网络              不共享、随节点丢失；只能做缓存或暂存
NFS                    通用 NFS 服务                         POSIX     简单、通用                          单服务器带宽上限；不适合 checkpoint 突发
```

把三类负载映射到方案：

- **数据集**：源放对象存储，前面放缓存层（JuiceFS/Alluxio/Fluid），缓存落节点本地 NVMe。数据只读且反复读，缓存命中率高；小文件用 JuiceFS 这类有独立元数据引擎的方案比直接 S3 好几个量级（定性判断，以各产品文档为准）。
- **checkpoint**：并行文件系统直写，或先落本地 NVMe 再异步上传对象存储（第八章）。关键是聚合写吞吐，元数据不重要。
- **权重**：源在对象存储，分发靠缓存层或节点本地缓存 + 预热（第九章）。

### 3. K8s 里的挂载：CSI

所有这些在 Pod 里的样子都一样：一个 PVC。差别在 StorageClass 的 provisioner 和参数。JuiceFS CSI Driver 的 provisioner 是 `csi.juicefs.com`，StorageClass 参数通过 Secret 指向元数据引擎与对象存储（字段名以 JuiceFS CSI Driver 文档为准）；mini-platform 的 `storage/juicefs-pvc.yaml`：

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: juicefs-ckpt
provisioner: csi.juicefs.com
reclaimPolicy: Retain
parameters:
  csi.storage.k8s.io/provisioner-secret-name: juicefs-secret
  csi.storage.k8s.io/provisioner-secret-namespace: kube-system
  csi.storage.k8s.io/node-publish-secret-name: juicefs-secret
  csi.storage.k8s.io/node-publish-secret-namespace: kube-system
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ckpt
  namespace: default
spec:
  accessModes: ["ReadWriteMany"]
  storageClassName: juicefs-ckpt
  resources:
    requests:
      storage: 10Ti
```

`ReadWriteMany` 是训练 checkpoint 的硬要求——64 个 Pod 同时写同一个目录。并行文件系统（Lustre、Storage Scale、WEKA、BeeGFS）各有自己的 CSI 驱动，也有很多集群直接在节点上挂好文件系统再以 `hostPath` 进 Pod——后者绕过了 CSI 的生命周期管理，但对静态、全集群共享的并行文件系统是常见做法。

两个平台层的注意点：

- **FUSE 客户端跑在哪**。JuiceFS/Alluxio 的 CSI 通常在节点上以 DaemonSet 或 per-PVC 的 mount Pod 运行 FUSE 进程；它的 CPU/内存是平台开销，而且升级 CSI 或 mount Pod 重启期间，用着这个卷的训练 Pod 的 I/O 可能中断（各产品的平滑升级能力以其文档为准）。
- **缓存盘的容量与驱逐**。节点本地 NVMe 做缓存时，一个节点上跑几个任务、每个任务的数据集多大、缓存驱逐策略是什么，决定命中率；缓存目录用 `hostPath` 还是 `local` PV 也影响 Pod 的可迁移性。


## 八、checkpoint I/O 的算术

### 1. 一个 70B 模型的训练状态有多大

混合精度 + Adam 的训练状态，每个参数：

```text
BF16 参数                 2 字节
FP32 主参数（master）      4 字节
Adam 一阶动量 m（FP32）    4 字节
Adam 二阶动量 v（FP32）    4 字节
─────────────────────────────────
                          14 字节 / 参数
```

70B 参数 × 14 字节 ≈ 980 GB，取整就是 **1 TB 量级**（不含 dataloader 状态、RNG 状态和可选的梯度）。这个数字与并行策略无关：无论 TP/PP/DP 怎么切，整个任务的状态总量就是这么多，只是分布在不同 rank 上。ZeRO/FSDP 把优化器状态分片后，每个 rank 持有的是 1/DP 份，合起来仍是 1 TB。

### 2. 聚合带宽

总纲的设定：每 30 分钟一次，要求 1 分钟内写完。

$$
B_{\text{agg}} = \frac{1\,\text{TB}}{60\,\text{s}} \approx 16.7\,\text{GB/s} \approx 133\,\text{Gb/s}
$$

分摊到 8 个节点，每节点 ≈ 2.1 GB/s 的持续写；分摊到 64 个 rank，每 rank ≈ 260 MB/s。这三个数字分别对应存储系统的聚合吞吐、每节点网卡/客户端吞吐、每个进程的写吞吐。

为什么是 1 分钟而不是随便多久：同步 checkpoint 期间训练是停的。每 30 分钟训练停 1 分钟，约 3% 的时间开销；如果写要 10 分钟，每 40 分钟里只有 30 分钟在训练，有效吞吐掉四分之一。这就是开头那个"一次 checkpoint 十几分钟"的成本——它直接从任务级的 MFU 里扣。反过来，为什么是 30 分钟：间隔决定故障时丢多少进度，64 卡 30 分钟 = 32 卡·小时的重算。checkpoint 频率是"写开销"与"重算损失"之间的权衡，两者都和存储带宽挂钩。

### 3. 分片 checkpoint：让每个 rank 写自己的那份

早期的做法是 rank 0 `all_gather` 全部状态再 `torch.save`——1 TB 先汇聚到一个节点再由一个进程写，单节点网卡与单进程写吞吐成为瓶颈，而且需要 1 TB 的 host 内存。PyTorch 的 Distributed Checkpoint（DCP）改成**每个 rank 只写自己持有的分片**：

- `torch.distributed.checkpoint.save(state_dict, checkpoint_id=path)`（`torch/distributed/checkpoint/state_dict_saver.py`）：各 rank 调用 planner 生成写计划，`ShardedTensor`/`DTensor` 只保存本地分片，由 coordinator rank 去重后各自写文件；
- `FileSystemWriter(path, single_file_per_rank=True, sync_files=True, thread_count=1, per_thread_copy_ahead=10_000_000)`（`filesystem.py`）：默认每 rank 一个文件加一个全局 `.metadata`；`thread_count` 决定每个 rank 的 I/O 并发，`sync_files` 决定是否 `fsync`；
- `torch.distributed.checkpoint.state_dict.get_state_dict(model, optimizers, options=StateDictOptions(...))` 把 FSDP/DDP 模型与优化器的状态取成 DCP 能处理的形式。

分片 checkpoint 不改变总量（还是 1 TB），但把写压力从一个进程摊到 64 个进程、从一个节点摊到 8 个节点，让"每 rank 260 MB/s"成为一个每个进程都能轻松做到的数字，瓶颈回到存储系统的聚合吞吐上。它还带来一个平台层的后果：**恢复时 rank 数可以变**——DCP 按元数据重新分片（`resharding.py`），16 节点存的 checkpoint 可以用 8 节点恢复。这对第三篇讲的弹性与抢占很重要。

### 4. 异步 checkpoint：把 1 分钟变成 1 秒

同步写的 1 分钟里 GPU 是空转的。异步 checkpoint 把"写"拆成两步：先把状态从显存拷到 host 内存（stage），训练立刻继续；再由后台线程/进程把 host 内存里的副本写到存储（upload）。

- `torch.distributed.checkpoint.async_save(state_dict, checkpoint_id=..., storage_writer=..., async_checkpointer_type=AsyncCheckpointerType.THREAD, async_stager=None)`（`state_dict_saver.py`）：`AsyncCheckpointerType` 有 `THREAD` 与 `PROCESS` 两种；返回一个 `Future`，或者在提供了支持异步 stage 的 stager 时返回 `AsyncSaveResponse`（`staging_completion` 与 `upload_completion` 两个 future）。
- `FileSystemWriter` 本身实现了 `BlockingAsyncStager`（`staging.py`）：stage 是同步的 D2H 拷贝（可用 pinned memory 加速），upload 在后台。
- `DefaultStager(StagingOptions(use_pinned_memory=True, use_shared_memory=True, use_async_staging=True, use_non_blocking_copy=True))`：把 stage 也做成异步，训练几乎不停。

于是算术变成：

```text
                       同步 dcp.save                 异步 dcp.async_save
────────────────────────────────────────────────────────────────────────────────────────────────
训练停顿                写完为止：1 TB / B_agg          D2H 拷贝：每 rank ~15.6 GB（1 TB / 64）经 PCIe 到 host 内存，
                                                       秒级（PCIe 带宽以硬件为准）；异步 stage 时更短
对存储的带宽要求        16.7 GB/s（1 分钟写完）         只需在下一次 checkpoint 之前写完：1 TB / 30 min ≈ 0.56 GB/s
                                                       聚合；实际留余量，取 2～3 GB/s
新增的资源需求          无                              每 rank ~15.6 GB pinned host 内存 × 8 rank/节点 ≈ 125 GB/节点；
                                                       后台写与训练争 CPU 与网卡（若存储走同一张网卡）
故障窗口                写完即持久                       host 内存里的副本在节点宕机时丢失；上一份仍在
```

异步 checkpoint 把存储带宽需求降了一个量级以上，代价是每节点上百 GB 的 host 内存和"最新一份可能还没落盘"的语义。平台要做的是：给训练 Pod 留够 host 内存（`resources.requests.memory` 要算上这一份）、存储网与 RDMA 网分开（否则后台写会和 all_reduce 抢带宽）、并监控 upload 的完成时间是否一直小于 checkpoint 间隔——一旦超过就会积压。

### 5. `dcp-bench.py`：测这两个数字

mini-platform 的 `storage/dcp-bench.py` 是一个 torchrun 脚本，在指定路径（挂进 Pod 的 PVC）上对同一份人造状态字典分别跑 `dcp.save` 与 `dcp.async_save`，报告每次的墙钟时间与聚合吞吐。它是工具，不附带任何结果：

```python
#!/usr/bin/env python3
"""mini-platform/storage/dcp-bench.py — 对比 dcp.save 与 dcp.async_save 的写入耗时。

用法（每个 rank 写 GB_PER_RANK GB 的人造状态，键带 rank 后缀以模拟分片）：
  torchrun --nnodes=$NNODES --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=$MASTER:29500 \
      dcp-bench.py --path /ckpt/bench --gb-per-rank 4 --threads 4 --rounds 2
输出：每轮 save / async_save 的 stage 时间、总时间、聚合 GB/s（以 rank 0 统计，各 rank 用 barrier 对齐）。
"""
import argparse
import os
import shutil
import time

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemWriter


def make_state(gb: float, rank: int, device: torch.device) -> dict:
    # 每个 rank 的键唯一，DCP 不会跨 rank 去重，因此每个 rank 写自己的全部数据（模拟 FSDP/ZeRO 分片）
    n = int(gb * (1 << 30) // 2)  # bf16
    chunks = 8
    per = n // chunks
    return {f"shard_r{rank}_{i}": torch.randn(per, dtype=torch.bfloat16, device=device) for i in range(chunks)}


def barrier_time() -> float:
    dist.barrier()
    return time.perf_counter()


def run_sync(state, path, threads):
    writer = FileSystemWriter(path, thread_count=threads, sync_files=True)
    t0 = barrier_time()
    dcp.save(state, storage_writer=writer)
    t1 = barrier_time()
    return t1 - t0


def run_async(state, path, threads):
    writer = FileSystemWriter(path, thread_count=threads, sync_files=True)
    t0 = barrier_time()
    fut = dcp.async_save(state, storage_writer=writer)
    t_stage = time.perf_counter() - t0  # 训练线程被阻塞的时间（stage）
    # v2.13.0：返回 Future 或 AsyncSaveResponse（有 upload_completion）
    upload = getattr(fut, "upload_completion", fut)
    upload.result()
    t1 = barrier_time()
    return t_stage, t1 - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--gb-per-rank", type=float, default=2.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", rank % torch.cuda.device_count()) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    state = make_state(args.gb_per_rank, rank, device)
    total_gb = args.gb_per_rank * world

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    log(f"world={world} backend={backend} total={total_gb:.1f} GB threads={args.threads} path={args.path}")
    for r in range(args.rounds):
        for mode in ("sync", "async"):
            sub = os.path.join(args.path, f"{mode}-{r}")
            if rank == 0:
                shutil.rmtree(sub, ignore_errors=True)
                os.makedirs(sub, exist_ok=True)
            dist.barrier()
            if mode == "sync":
                t = run_sync(state, sub, args.threads)
                log(f"round {r} sync     total={t:7.2f}s  agg={total_gb / t:6.2f} GB/s")
            else:
                t_stage, t = run_async(state, sub, args.threads)
                log(f"round {r} async    stage={t_stage:6.2f}s  total={t:7.2f}s  agg={total_gb / t:6.2f} GB/s")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

用它能回答三个问题：这个 PVC 的聚合写吞吐是多少（同步模式的 `agg`）；异步模式下训练实际被阻塞多久（`stage`）；线程数（`--threads`，即 `FileSystemWriter.thread_count`）和每 rank 数据量怎样影响这两个数字。把 `--gb-per-rank` 设成真实任务每 rank 的状态大小、`--nnodes` 设成真实节点数，得到的就是这个集群上这个任务的 checkpoint 时间下界。


## 九、推理侧的权重分发

### 1. 问题：扩容时间被 I/O 主导

一个 70B BF16 模型的权重约 140 GB。推理副本扩容时，从对象存储拉到本地、加载进显存，这一步的时间由"读 140 GB 的吞吐"决定：1 GB/s 要两分多钟，200 MB/s 要十几分钟。而扩容往往是并发的——高峰来了要同时起 4 个副本，4 个副本同时读同一份文件，对象存储或文件系统的热点就出现了。第六篇讲扩缩容时会把这一段计入"扩容时间分解"；本篇讨论平台能做什么。

### 2. 三种方案与 GPUDirect Storage

```text
方案                 做法                                            适合                           代价 / 边界
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
节点本地 NVMe 缓存    权重第一次读时落到节点 NVMe（缓存层 CSI 自动做，    同一模型在同一批节点上反复扩缩       首个副本仍要冷读；缓存占盘；节点池变动
                      或 DaemonSet 预热），后续副本直接本地读             （最常见的情况）                     后要重新预热；缓存驱逐策略
权重进镜像            把权重打进容器镜像的一层，靠镜像仓库与节点镜像缓存   模型固定、版本少、节点数多            镜像几十～上百 GB：仓库压力、拉取时间、
                      分发（第二篇的镜像预热与 P2P 分发同样适用）         （边缘/固定部署）                    版本更新要重打镜像；镜像层不共享权重
P2P 分发              节点之间互相传（Dragonfly 这类 P2P 镜像/文件分发，   大规模并发扩容（几十副本同时起）      需要额外的 P2P 组件；首个种子仍是冷读；
                      或缓存层自带的节点间读）                                                              占用节点间网络（要与 RDMA 网分开）
GPUDirect Storage    存储 → 显存的 DMA 路径（nvidia-fs 驱动，GPU Operator  单节点本地 NVMe 或支持 GDS 的并行     需要 GDS 支持的文件系统与驱动；引擎要用
（GDS）              spec.gds.enabled 部署），绕过 host 内存             文件系统上、CPU 成为瓶颈时           cuFile API 读；GPU Operator 里标为实验性
```

选择顺序通常是：先用缓存层 + 本地 NVMe（覆盖大多数场景），并发扩容规模大了加 P2P，只有模型固定且节点多才考虑权重进镜像。GDS 解决的是另一个瓶颈——当 NVMe 已经很快而 host 内存拷贝与 CPU 成为限制时，它把存储读直接送进显存；对"从对象存储拉权重"这类受网络限制的场景没有帮助。GDS 是否值得用，看 `nvidia-smi` 和 `top`：如果加载权重时 CPU 打满而 NVMe 还有余量，才是它的场景。


## 十、代价与边界

每个机制都在填一个洞，也都挖了新的：

```text
机制                          引入的新问题                                            不该用 / 要小心的场景
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Multus 第二张网卡              Pod 有两个网络平面，Service/NetworkPolicy/DNS 只管一个； 只有一两台 RDMA 节点的集群——手工 hostNetwork 更简单；
                              IPAM 是新组件（nv-ipam/whereabouts）要运维；               Pod 数超过 IPAM 池
                              CNI 链多一跳，Pod 创建变慢
rdma-shared-dev-plugin        无隔离：一个 Pod 的错误流量影响同节点其他 Pod；            多租户共享节点且需要审计/限速——用 SR-IOV；
                              rdmaHcaMax 是软限制，不是带宽配额；                       需要 exclusive netns 的安全要求
                              要求 shared netns，与 exclusive 的方案互斥
SR-IOV device plugin          VF 数受固件限制；VF 配置改动要清空节点；                   训练集群（节点独占，隔离多余、VF 的 QP 配额可能反而限制 NCCL）；
                              需要 SR-IOV Network Operator 或等价的节点配置             没有专人维护固件/驱动矩阵
exclusive netns               全节点一个模式；与 shared plugin 互斥；                   需要同节点混跑 shared 与 exclusive 需求的场景（不可能）
                              切换需要节点上没有在用的 RDMA 设备
Network Operator + 驱动容器   驱动升级 = 重载 mlx5_core = 所有用网卡的 Pod 断开；       宿主机已经由别的流程管 OFED（用 useHostMofed 并放弃 Operator 管驱动）；
                              与 GPU Operator 的顺序依赖；两个 Operator 的版本矩阵       内核版本不在预编译镜像列表里（forcePrecompiled 会失败）
GPUDirect RDMA（peermem）      驱动/OFED/内核三方版本耦合；peermem 与 OFED 的加载顺序    GPU 与网卡跨 NUMA 的机器——GDR 被 NCCL 自己关掉，装了也没用
DMA-BUF                       对内核/驱动版本下限高                                    老内核、老驱动
容器内 nccl-tests 验证        需要 MPI 或等价的多进程启动；两 Pod 要跨节点              没有 gang scheduling 时两 Pod 可能落同一节点（测不到网络）
并行文件系统                  运维重、专用客户端、内核模块与节点 OS 耦合；贵            小集群、预算有限——对象存储 + 缓存层 + 本地 NVMe 暂存
缓存层（JuiceFS/Alluxio/Fluid）FUSE 开销；元数据服务是新单点；缓存一致性；             频繁写的工作负载（缓存层对写的加速有限）
                              mount Pod 重启影响所有使用者
异步 checkpoint               每节点上百 GB pinned host 内存；后台写抢 CPU/网卡；        host 内存本来就紧张的节点（例如 CPU offload 的训练）；
                              "最新一份可能未落盘"                                     存储网与 RDMA 网共用一张网卡
权重进镜像                    镜像巨大，仓库与拉取压力                                 模型频繁更新、多版本共存
GPUDirect Storage             文件系统与驱动支持面窄；实验性                           网络受限的加载场景（没有帮助）
```

两条更宏观的边界：

- **RDMA 流量绕过 K8s 的网络模型**。NetworkPolicy、Service mesh、eBPF 数据面对 verbs 流量一无所知。安全与审计要在 IB 分区（pkey，Network Operator 的 `ibKubernetes`）、RoCE 的 VLAN、或 SR-IOV + exclusive netns 这一层做，不能指望 K8s 原生对象。
- **存储选型不是 K8s 的问题**。CSI 只是接口；带宽、并发、元数据这些属性全在存储系统本身。平台能做的是把三类负载分开、把需求算清楚（第八章）、把缓存放对地方，而不是靠一个 StorageClass 解决所有问题。


## 十一、本文小结

### 1. 要点回顾

```text
为什么不够      默认 CNI 给 Pod 一张 veth 走 overlay，数据经 host 内存与内核协议栈；NCCL 需要 /dev/infiniband 设备文件、
                peermem/DMA-BUF 注册显存、一个 RDMA 侧 IP 做握手。缺任何一项都不报错，只是回落：
                "NET/IB : No device found." → "Using network Socket"
第二张网卡      Multus（thin 或 thick）当元插件；NetworkAttachmentDefinition（k8s.cni.cncf.io/v1）装 CNI 配置；
                Pod 注解 k8s.v1.cni.cncf.io/networks 引用；NAD 注解 k8s.v1.cni.cncf.io/resourceName 关联 device plugin；
                Multus 写回 network-status。三种接入：host-device（整卡/VF 独占）、macvlan（RoCE 共享）、ipoib（IB 共享）；
                Network Operator 的 HostDeviceNetwork / MacvlanNetwork / IPoIBNetwork 生成 NAD
设备进容器      rdma-shared-dev-plugin：configList[].resourceName/rdmaHcaMax/selectors → rdma/<name> 资源，Allocate 挂该组
                网卡全部 uverbs/umad/issm + rdma_cm，要求 netns shared，无隔离、无 GPU 亲和；
                SR-IOV device plugin：一 VF 一 Pod，isRdma 挂 VF 的 RDMA 设备，通常配 exclusive netns；
                容器要 IPC_LOCK；rdma system set netns shared|exclusive 是节点级全局开关
两个 Operator   NicClusterPolicy（mellanox.com/v1alpha1）：ofedDriver（DOCA-OFED 容器，无 docaDriver 字段）、rdmaSharedDevicePlugin /
                sriovDevicePlugin（config 原样透传）、secondaryNetwork.{multus,cniPlugins,ipoib}、nvIpam；
                ClusterPolicy spec.driver.rdma.{enabled,useHostMofed} 管 nvidia-peermem sidecar；OFED 先、GPU 驱动后；
                nvidia_peermem 与 DMA-BUF 二选一即可，NCCL 都能探测
验证            ls /dev/infiniband → ibv_devinfo → rdma link → ip addr → ulimit -l → ib_write_bw → nccl-tests；
                INFO 日志几行：NET/IB : Using（数 HCA、看 OOB 接口）· Using network IB/Socket · via NET/IB/N/GDRDMA ·
                GPU Direct RDMA Disabled（拓扑）
核心问题        Pod 网络（注解、NAD、IPAM、OOB 接口）/ device plugin（未请求、selector、部分网卡、IPC_LOCK、netns 模式）/
                NCCL 环境（IB_HCA 名字与 ^= 语义、SOCKET_IFNAME 容器内名字、NET_GDR_LEVEL、RoCE 的 GID_INDEX）
存储            三类负载：数据集（小文件随机读，缓存）· checkpoint（大块顺序写，聚合吞吐）· 权重（一写多读，并发）；
                并行文件系统给 checkpoint，对象存储 + 缓存层给数据集与权重，本地 NVMe 做缓存与暂存；CSI 只是接口
算术            70B × 14 B/参数 ≈ 1 TB；30 分钟一次、1 分钟写完 → 16.7 GB/s 聚合、2.1 GB/s 每节点、260 MB/s 每 rank；
                分片（dcp.save + FileSystemWriter）把压力摊到每个 rank；异步（dcp.async_save）把停顿变成 D2H 拷贝、
                带宽需求降到 1 TB / 30 min ≈ 0.56 GB/s，代价是每节点 ~125 GB host 内存
权重分发        本地 NVMe 缓存（默认）→ P2P（大规模并发扩容）→ 权重进镜像（模型固定）；GDS 只在 CPU 成为瓶颈时值得
```

### 2. 引擎需求 → K8s 空缺 → 平台机制 → 代价

```text
引擎需求                          K8s 的空缺                       平台机制                                     代价
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
NCCL 要一个 RDMA 侧 IP 做握手      一 Pod 一网卡，只有 overlay        Multus + NetworkAttachmentDefinition          第二个网络平面；IPAM 组件；Pod 创建慢
                                                                     （host-device / macvlan / ipoib）
NCCL 要打开 /dev/infiniband/*      没有 RDMA 资源类型，设备不进容器   rdma-shared-dev-plugin（rdma/<name>）或        shared 无隔离；SR-IOV 运维重；
                                                                     SR-IOV device plugin（一 VF 一 Pod）           netns 模式全节点二选一
网卡要直接读写显存                 内核模块不归 K8s 管                Network Operator ofedDriver + GPU Operator      三方版本耦合；驱动升级断网；
                                                                     driver.rdma（nvidia_peermem）或 DMA-BUF          跨 NUMA 时无效
注册内存要 pin                     容器默认 memlock 很小              securityContext IPC_LOCK                       放宽了一项隔离
"走了哪条路"要可验证               K8s 只知道 Pod Running             容器内清单 + NCCL_DEBUG=INFO 判读           需要 MPI/多 Pod 启动；要人读日志
checkpoint 大块突发写              PVC 不表达带宽                     并行文件系统 RWX PVC；DCP 分片 + 异步            存储贵/运维重；异步要 host 内存
数据集小文件随机读                 同上                               对象存储 + 缓存层（JuiceFS/Alluxio/Fluid）+ NVMe   FUSE 开销；元数据单点；缓存一致性
权重并发读                         同上                               本地 NVMe 缓存 / P2P / 权重进镜像 / GDS           缓存预热；P2P 组件；镜像巨大
```

### 3. 本篇涉及的源码与 CRD 位置

```text
项目                         路径                                                    关键符号 / 内容
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Multus CNI v4.3.0            pkg/k8sclient/k8sclient.go                              networkAttachmentAnnot（k8s.v1.cni.cncf.io/networks）·
                                                                                     resourceNameAnnot（…/resourceName）· defaultNetAnnot；
                                                                                     GetNetAttachDef；getKubernetesDelegate 读 resourceName
                             docs/thick-plugin.md · deployments/multus-daemonset{,-thick}.yml  thin/thick 两种形态；daemon-config.json
                             docs/configuration.md                                   clusterNetwork/delegates · namespaceIsolation · readinessindicatorfile
                             docs/how-to-use.md · examples/macvlan-pod.yml           NAD 与 networks 注解的写法（JSON 形式含 interface/ips/gateway）
k8s-rdma-shared-dev-plugin   pkg/types/types.go                                      UserConfigList.configList/periodicUpdateInterval；UserConfig.
v1.5.4                                                                               resourceName/resourcePrefix/rdmaHcaMax/devices/selectors；
                                                                                     Selectors.vendors/deviceIDs/drivers/ifNames/linkTypes
                             pkg/resources/resources_manager.go                      rdmaHcaResourcePrefix = "rdma"；ValidateRdmaSystemMode（拒绝 exclusive）
                             pkg/resources/server.go                                 Allocate（Devices 或 CDI 注解）；resourceName = prefix/name；
                                                                                     rdmaHcaMax 个 Device；cdiResourcePrefix = "nvidia.com"
                             pkg/resources/rdma_device_spec.go · pkg/utils/utils.go  requiredRdmaDevices（rdma_cm/umad/uverbs）；Get；VerifyRdmaSpec；GetRdmaDevices
                             deployment/k8s/base/configmap.yaml · example/test-hca-pod.yaml  配置示例；IPC_LOCK 与 rdma/hca_shared_devices_a 请求
NVIDIA Network Operator      api/v1alpha1/nicclusterpolicy_types.go                  NicClusterPolicySpec.ofedDriver/rdmaSharedDevicePlugin/sriovDevicePlugin/
v26.7.0                                                                              secondaryNetwork/nvIpam/ibKubernetes/…；OFEDDriverSpec；DevicePluginSpec.useCdi；
                                                                                     SecondaryNetworkSpec.multus/cniPlugins/ipoib；DriverUpgradePolicySpec
                             api/v1alpha1/groupversion_info.go                       mellanox.com/v1alpha1
                             api/v1alpha1/{hostdevicenetwork,ipoibnetwork,macvlannetwork}_types.go  networkNamespace/resourceName/master/mode/mtu/ipam
                             manifests/state-hostdevice-network/0010-hostdevice-net-cr.yml  生成的 NAD：type host-device + resourceName 注解
                             manifests/state-ipoib-network/0010-ipoib-net-cr.yaml     生成的 NAD：type ipoib + master
                             manifests/state-ofed-driver/0050_ofed-driver-ds.yaml     /run/mellanox/drivers · /dev/infiniband · /etc/modprobe.d/ib_core.conf
                             manifests/state-rdma-shared-device-plugin/               ConfigMap 与 DaemonSet（hostNetwork，挂 /dev）
                             example/crs/*.yaml · example/*.yml · example/sriov-network/  NicClusterPolicy 示例；rdma-gpu-test-pod；SR-IOV 的 isRdma
                             controllers/mofed_wait_labels.go · pkg/nodeinfo/attributes.go  network.nvidia.com/operator.mofed.wait
                             docs/host-ofed.md · docs/mofed-container-env-vars.md    宿主机 OFED 与 Operator 共存；驱动容器 env
NVIDIA GPU Operator v26.7.0  api/nvidia/v1/clusterpolicy_types.go                    DriverSpec.GPUDirectRDMA（json rdma）：Enabled/UseHostMOFED；
                                                                                     GPUDirectStorageSpec（gds）；GDRCopySpec（gdrcopy）
                             controllers/object_controls.go                          transformPeerMemoryContainer；GPU_DIRECT_RDMA_ENABLED / USE_HOST_MOFED /
                                                                                     MOFED_ENABLED / GDS_ENABLED
                             assets/state-driver/0500_daemonset.yaml                 nvidia-peermem-ctr（reload_nvidia_peermem / probe_nvidia_peermem）· nvidia-fs-ctr ·
                                                                                     /run/mellanox/drivers 挂载
                             cmd/nvidia-validator/main.go                            nvidia-peermem-ready；lsmod 检查
                             internal/nodeinfo/attributes.go                         NodeLabelWaitOFED
NCCL 2.28.9（仅日志与变量）   src/transport/net_ib.cc                                 NCCL_IB_HCA（^/= 语义）· NCCL_IB_GID_INDEX · NCCL_IB_ROCE_VERSION_NUM ·
                                                                                     NCCL_IB_ADDR_FAMILY · ncclIbGetGidIndex · ibGdrSupportInitOnce（nvidia_peermem）·
                                                                                     ncclIbDmaBufSupport · "NET/IB : Using …; OOB …" · "NET/IB : No device found." ·
                                                                                     "NET/IB : No IP interface found."
                             src/transport/net_socket.cc · src/misc/socket.cc         "NET/Socket : Using …"；NCCL_SOCKET_IFNAME
                             src/init.cc · src/plugin/net.cc                          "Using network %s"；NCCL_NET；NCCL_DMABUF_ENABLE；内部插件顺序 IB → Socket
                             src/transport/net.cc · src/graph/paths.cc               "via NET/%s/%d/GDRDMA"；NCCL_NET_GDR_LEVEL；"GPU Direct RDMA Enabled/Disabled"
nccl-tests 2.18.3            src/common.cu                                           参数表；MPI 的使用；-T 超时
PyTorch v2.13.0              torch/distributed/checkpoint/state_dict_saver.py         save · async_save · AsyncCheckpointerType（THREAD/PROCESS）· AsyncSaveResponse
                             torch/distributed/checkpoint/filesystem.py               FileSystemWriter（single_file_per_rank/sync_files/thread_count/per_thread_copy_ahead）
                             torch/distributed/checkpoint/staging.py                  BlockingAsyncStager · DefaultStager · StagingOptions
                             torch/distributed/checkpoint/state_dict.py               get_state_dict · StateDictOptions（full_state_dict/cpu_offload）
                             torch/distributed/checkpoint/resharding.py               恢复时按元数据重新分片
```

### 4. mini-platform 本篇增量：`net/` 与 `storage/`

```text
mini-platform/
├── net/
│   ├── nad-ipoib.yaml               NetworkAttachmentDefinition（k8s.cni.cncf.io/v1）：type ipoib + master + whereabouts IPAM
│   ├── nad-hostdevice.yaml          变体：type host-device + resourceName 注解（需 sriovDevicePlugin）；注释里给出 macvlan 变体
│   ├── rdma-plugin-configmap.yaml   rdma-shared-dev-plugin 的 config.json（configList：resourceName/rdmaHcaMax/selectors）
│   ├── nicclusterpolicy.yaml        第五章的最小 NicClusterPolicy（mellanox.com/v1alpha1）
│   └── nccl-tests-job.yaml          Indexed Job × 2 + headless Service：RDMA 资源请求、networks 注解、IPC_LOCK、NCCL_DEBUG；
│                                    注释给出 hostNetwork/hostIPC 变体（无 RDMA 集群）与 hostPath /dev/infiniband 变体
└── storage/
    ├── juicefs-pvc.yaml             StorageClass（csi.juicefs.com）+ RWX PVC；可替换为任何 RWX CSI
    └── dcp-bench.py                 第八章的 torchrun 脚本：dcp.save vs dcp.async_save 的 stage/总时间/聚合 GB/s
```

`rdma-plugin-configmap.yaml` 只在不用 Network Operator、直接部署 device plugin 的 DaemonSet 时需要（`deployment/k8s/base/`）；用了 `NicClusterPolicy` 就由 Operator 生成。三步验证的剧本：

```text
1  kubectl apply -f net/nicclusterpolicy.yaml；等 rdma-shared-dp-ds 与 mofed 的 Pod Ready；
   kubectl describe node <n> | grep rdma/ 看到 Allocatable
2  kubectl apply -f net/nad-ipoib.yaml（RoCE 换 macvlan 变体；SR-IOV 换 nad-hostdevice.yaml）
   kubectl apply -f net/nccl-tests-job.yaml；kubectl logs job/nccl-tests -c nccl（index 0）
   先找 "NET/IB : Using" 数设备，再找 "Using network"，再找 "GDRDMA"；最后看大消息的 busbw
3  kubectl apply -f storage/juicefs-pvc.yaml；把 PVC 挂到一个 torchrun Job 的 /ckpt；
   torchrun … dcp-bench.py --path /ckpt/bench --gb-per-rank <真实每 rank 状态 GB>
   同步的 agg 与第八章的 16.7 GB/s 比，异步的 stage 与训练 step time 比
```

到这里，资源层的四篇讲完了：GPU 怎么进容器、任务怎么被调度、卡怎么切、网络和存储怎么喂。下一篇进入交付层：一个 vLLM 进程加一个 Service 只是最简单的推理服务，多节点副本、扩缩容、PD 分离、灰度都要 Serving 平台来补。

> **一个 TP=4 的 70B 模型服务，晚高峰要从 2 副本扩到 6 副本，每个副本从调度到能接流量要 8 分钟。扩缩容指标选什么、阈值定多少、提前多久触发，才能在高峰到来前就绪而不在平时浪费 16 张卡？**


## 下一篇

[Serving 平台：从 InferenceService 到 llm-d](/serving-platforms-kserve-triton-ray-serve-llm-d.html)
