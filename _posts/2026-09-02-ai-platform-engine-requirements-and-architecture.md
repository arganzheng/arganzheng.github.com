---
layout: post
title: "AI 平台工程（01）：引擎的需求清单与平台的整体架构"
subtitle: "What Engines Demand from the Platform, and the Platform's Two Layers"
tags: [Kubernetes, GPU, MLOps, AI, AI-Infra]
catalog: true
---

> 本文是《AI 平台工程：资源层与交付层》系列的第 1 篇。上一篇：[总纲](/ai-platform-engineering.html)；下一篇：[容器里的 GPU：驱动、CUDA、device plugin 与镜像](/gpu-in-containers-driver-cuda-device-plugin.html)。

一个刚装好的 Kubernetes 集群，三台 worker 每台插着一张 GPU。提交一个只有十几行的 Pod，`resources.limits` 里写 `nvidia.com/gpu: 1`，它一直 Pending。`kubectl describe pod` 的 Events 里只有一行：`0/4 nodes are available: 3 Insufficient nvidia.com/gpu`。三张卡明明在那里，`nvidia-smi` 在宿主机上能看到，调度器却说"不够"。

这一行事件是本系列的起点，因为它精确地暴露了原生 Kubernetes 与 AI 负载之间的第一道缝：**调度器不知道什么是 GPU**。它只知道节点的 `status.allocatable` 里有没有一个叫 `nvidia.com/gpu` 的键、值是几；没人往里写这个键，值就是零，零小于一，所以 Insufficient。要让这行事件消失，得装 device plugin——这是第二篇的事。但装完之后，缝并没有合上，只是换了位置：32 个 Pod 的训练任务调度了 30 个就卡住、两张卡在同一台机器上却走了 PCIe、推理服务的 HPA 盯着 CPU 利用率永远不扩容、一张 80 GB 的卡跑一个 7B 模型剩下 60 GB 没人能用。每一道缝后面都有一个组件，每个组件的存在都是因为引擎提出了某个原生 Kubernetes 没打算满足的要求。

所以在装任何组件之前，值得先把这些要求列出来。训练框架要什么？推理引擎要什么？两者要的东西哪些一致、哪些冲突？原生 Kubernetes 能满足哪些、满足不了哪些？满足不了的那些，分别由哪个组件、在哪一层去填？这一篇不装任何东西，只做这份清单，并用它把后面七篇的组件放到一张图上。

本篇的核心问题：

> **一个 4 节点 32 卡的训练任务和一个 TP=2、副本数动态变化的推理服务，各自对平台提出的需求列成一张表，哪几条是原生 Kubernetes 满足不了的？**

全文的版本锚点：Kubernetes v1.37.0；引擎侧只引用 PyTorch 2.13.0 的 `torchrun` 环境变量与 rendezvous 语义、vLLM v0.23.0 的指标名与启动参数，不进入任何引擎内部实现。平台组件的版本随全景图逐一标注，全部发布于本文日期之前。


## 一、总览

### 1. 引擎的需求

平台之上跑的是两类引擎，它们对平台的要求可以各自压缩成一句话。

**训练框架**：给我 $$W$$ 个进程，每个进程一张卡，让它们**同时**起来、能互相找到、之间有高带宽网络，再给我一个能周期性写几百 GB 的地方——然后别打断我。这句话里的每个词都是一条需求：

```text
W 个进程同时起来          → 整数个 GPU；全员到齐才能开始（gang）
互相找到                  → 稳定的地址（MASTER_ADDR）与编号（RANK / WORLD_SIZE）
高带宽网络                → 节点内 NVLink、节点间 RDMA；拓扑接近
周期性写几百 GB            → checkpoint 的突发顺序写带宽
别打断我                  → 抢占要考虑上次 checkpoint 之后的损失；一个进程挂了要整组重启
```

**推理引擎**：给我一张（或几张）显存够放权重加 KV cache 的卡，让我把权重加载完再接流量，副本多少按**我内部的**队列长度来定，请求别用轮询发给我。同样拆开：

```text
显存够放                  → 显存是硬约束，不能超卖；小模型只需要一张卡的一部分
加载完再接流量            → 扩容时间由权重加载决定，就绪探针要等它
按我内部的队列长度         → 扩缩容信号是 vllm:num_requests_waiting 这类引擎指标，不是 CPU
别轮询                    → 路由要知道每个副本的 KV cache 与队列状态
```

第二、三章把这两句话展开成两张需求清单，第四章把两张清单放在一起看它们的冲突。

### 2. K8s 的空缺

Kubernetes 的核心对象——Pod、Deployment、Service、Job——建立在四个假设上：工作负载是**单 Pod 独立**的、资源是**可细分可压缩**的 CPU 与内存、网络是**一张 overlay 网卡**、服务的负载信号是**CPU 利用率**。AI 负载逐条违背这四个假设，于是有了四类空缺：

```text
假设                          AI 负载的现实                          空缺
────────────────────────────  ────────────────────────────────────  ──────────────────────────────
Pod 之间独立，逐个调度          训练是 W 个 Pod 强耦合，缺一不可         gang scheduling、队列与配额、拓扑
资源是 CPU / 内存，可细分       GPU 是整数个、有型号、有拓扑、不能超卖    设备发现与属性、共享与切分
一张 overlay 网卡              NCCL 需要直接打开 RDMA 设备             第二张网卡、RDMA 设备直通
Service 轮询 + HPA 看 CPU      副本状态在引擎内部；请求长短差百倍       多 Pod 副本、引擎指标扩缩容、状态感知路由
```

第五章逐条说明这些空缺在原生对象上具体表现为什么，哪些只是"缺个插件"，哪些是模型层面的缺失。

### 3. 平台的机制（全局图）

填这些空缺的组件，按"引擎之下"与"引擎之侧"分成两层。资源层的输入是裸节点与引擎的资源请求，输出是一组配好 GPU、网络、存储、能互相找到的 Pod；交付层的输入是这些 Pod，输出是一个有地址、有 SLA、有配额、有账单的服务。

```text
                           请求（OpenAI 兼容协议）
                                  │
交付层   ┌────────────────────────▼────────────────────────────────────────────────────────┐
         │ 模型网关   Gateway API Inference Extension v1.6.0（InferencePool）+ EPP（llm-d-router v0.10.0）│  第七篇
         │ Serving   KServe v0.20.0 · llm-d v0.9.0 · LeaderWorkerSet v0.10.0 · KEDA v2.20.2           │  第六篇
         │ 可观测    DCGM Exporter 4.6.0-4.8.3 · vllm:* 指标 · OpenCost v1.121.1                        │  第八篇
         └────────────────────────┬────────────────────────────────────────────────────────┘
                                  │  一组能接流量的 Pod  /  一组已就位的训练 Pod
资源层   ┌────────────────────────▼────────────────────────────────────────────────────────┐
         │ 任务表达   Kubeflow Trainer v2.3.0（TrainJob）· KubeRay v1.7.0（RayJob / RayService）         │  第三篇
         │ 调度      Kueue v0.19.2（配额闸门）· Volcano v1.15.2（批调度器）                               │  第三篇
         │ 切分      MIG（GPU Operator MIG Manager）· 时间片 / MPS（device plugin）· HAMi v2.10.0          │  第四篇
         │ 网络存储   Network Operator v26.7.0 · Multus v4.3.0 · RDMA shared dev plugin v1.5.4 · CSI     │  第五篇
         │ 设备      GPU Operator v26.7.0 · k8s-device-plugin v0.20.0 · DRA（Kubernetes v1.37.0）        │  第二篇
         └────────────────────────┬────────────────────────────────────────────────────────┘
                                  │  裸节点：驱动、GPU、网卡、本地盘
                            ┌─────▼─────┐
                            │  节点池    │      同位替代：Slurm（Slinky slurm-operator v1.2.2）· Ray
                            └───────────┘
```

第六章展开这张图：每一层的输入输出、组件为什么落在这一层、Slurm 与 Ray 为什么是"同位替代"而不是"另一层"。

### 4. 本文的章节安排

```text
第二章  训练任务的形态与需求     torchrun 的进程组与 rendezvous；MASTER_ADDR / WORLD_SIZE / RANK；一个进程挂掉为什么全停；四条资源需求
第三章  推理服务的形态与需求     副本的三种形态；prefill 与 decode；显存硬约束；扩容时间；负载信号在引擎内部
第四章  两组矛盾的需求          独占 vs 共享、拓扑 vs 弹性、批处理 vs 长驻；混部为什么难；两者的共同点
第五章  原生 Kubernetes 的空缺   kube-scheduler 的逐 Pod 模型；device plugin 的计数模型；一张网卡；Service 与 HPA 的假设
第六章  平台的两层拆分          资源层与交付层的输入输出；组件全景图按层落位；Slurm 与 Ray 的位置
第七章  核心问题：两张需求表     32 卡训练任务与 TP=2 推理服务各一张表，"原生 K8s 能否满足"列与补缺的篇目
第八章  全系列术语表            gang、cohort、ResourceFlavor、MIG profile、InferencePool、EPP、TTFT/TPOT、goodput、分配率与使用率
第九章  代价与边界              四栏表；每叠一层的代价；本系列的边界；托管服务替你做了什么
第十章  小结                    要点、源码与 CRD 位置、mini-platform 的第一批文件
```


## 二、训练任务的形态与需求

### 1. torchrun 启动的进程组

一个多机训练任务在平台眼里是**一组进程**。以 PyTorch 为例，每台机器上运行一个 `torchrun`（PyTorch 2.13.0 的 `torch/distributed/run.py`），它是一个 agent：本机起 `--nproc-per-node` 个 worker 进程，各绑一张卡，然后盯着它们。`--nnodes` 台机器上的 agent 通过 rendezvous 相互发现，共同构成一个大小为 $$W = \text{nnodes} \times \text{nproc\_per\_node}$$ 的进程组。

```text
节点 0                        节点 1                        节点 2                        节点 3
torchrun (agent)              torchrun (agent)              torchrun (agent)              torchrun (agent)
 ├─ worker RANK 0  GPU 0       ├─ worker RANK 8  GPU 0       ├─ worker RANK 16 GPU 0       ├─ worker RANK 24 GPU 0
 ├─ worker RANK 1  GPU 1       ├─ worker RANK 9  GPU 1       ├─ ...                         ├─ ...
 ├─ ...                        ├─ ...                        │                              │
 └─ worker RANK 7  GPU 7       └─ worker RANK 15 GPU 7       └─ worker RANK 23 GPU 7       └─ worker RANK 31 GPU 7
                                            ▲
                          rendezvous：--rdzv-backend=c10d --rdzv-endpoint=<节点 0 地址>:29400 --rdzv-id=<任务 ID>
```

`run.py` 的模块文档把这个结构定义为三层：`Worker` 是训练进程，`LocalWorkerGroup` 是一台机器上的 worker 集合，`WorkerGroup` 是全部节点 worker 的并集。平台需要理解的只有一件事：**这 $$W$$ 个进程在逻辑上是一个不可分的整体**，它们的编号由 rendezvous 统一分配，它们的通信拓扑在起来的那一刻就固定了。

### 2. rendezvous 与三个环境变量

rendezvous 是"$$W$$ 个进程如何互相找到并达成一致编号"的协议。`torchrun` 用 `--rdzv-backend`（默认推荐 `c10d`，即 PyTorch 自带的 TCP store）、`--rdzv-endpoint`（store 所在的地址与端口）、`--rdzv-id`（本任务的唯一 ID）三个参数定位它。所有节点的 agent 连到同一个 endpoint、报同一个 ID，`torch/distributed/elastic/rendezvous/dynamic_rendezvous.py` 里的 `_DistributedRendezvousOpExecutor` 等到参与者数量达到 `RendezvousSettings.min_nodes`（默认等于 `--nnodes`），关闭这一轮 rendezvous，给每个节点分配一个 `group_rank`。

之后每个 agent 为它的 worker 注入一组环境变量。`run.py` 的文档列了十几个，平台关心的是这三个加两个：

```text
MASTER_ADDR    RANK 0 所在节点的地址；进程组用它初始化 c10d TCP store（后续 NCCL 的通信也从这个 store 起步）
MASTER_PORT    MASTER_ADDR 上的端口
WORLD_SIZE     进程组的总大小 W
RANK           本进程在 0..W-1 中的全局编号
LOCAL_RANK     本进程在本节点上的编号（0..nproc_per_node-1），代码里 device_ids=[int(os.environ["LOCAL_RANK"])]
```

这五个变量是训练框架与平台之间**最窄的接口**。任何一个把训练任务搬进 Kubernetes 的方案——Kubeflow Trainer 的 `TrainJob`、Volcano 的 `vcjob`、手写的 Indexed Job——本质上都在做同一件事：让每个 Pod 起来时能算出自己的 `RANK`，能解析出 `MASTER_ADDR` 指向的那个 Pod。Kubeflow Trainer 2.3.0 在 `pkg/constants/constants.go` 里定义的 `PET_NNODES`、`PET_MASTER_ADDR` 等常量正是这些变量的 `torchrun` 别名（`PET_` 前缀由 `torchrun` 识别为参数）。这要求平台提供两样东西：**稳定的 DNS 名**（Pod 重启后地址不变，否则 `MASTER_ADDR` 失效）与**稳定的序号**（Pod 与 `RANK` 的映射不变）。原生 Kubernetes 用 headless Service 加 StatefulSet 或 Indexed Job 可以做到，但这只是必要条件，第三章会看到还缺什么。

### 3. 为什么一个进程挂掉整个任务就停

分布式训练的每一步都以集合通信结束：所有 $$W$$ 个进程各算一份梯度，然后 all-reduce 求平均，**每个进程都要等到其他 $$W-1$$ 个进程的数据到齐才能继续**。这是算法上的同步点，与框架无关。一个进程挂了，其余 $$W-1$$ 个进程会卡在下一次集合通信上，直到超时——在那之前它们占着 GPU 什么都不做。

`torchrun` 把这个事实做成了显式的策略。`run.py` 文档的 "Failure Modes" 一节写明：$$n$$ 个 worker 中任意 $$k \le n$$ 个失败，**所有** worker 被停止并重启，最多 `--max-restarts` 次。实现上，`torch/distributed/elastic/agent/server/api.py` 的 `SimpleElasticAgent._invoke_run()` 每隔 `monitor_interval` 调用 `_monitor_workers()`，一旦返回 `WorkerState.FAILED` 或 `UNHEALTHY`，就调用 `_restart_workers()`——停掉本地整组、重新 rendezvous、重新分配 `RANK` 与 `WORLD_SIZE`、重新启动。因为所有节点的 agent 共享同一个 rendezvous，一个节点重启 worker 组会让其他节点在下一轮 rendezvous 中察觉并跟着重启。至于 agent 自身或整个节点消失，文档明确写：由任务管理器决定是让整个任务失败（"gang semantics"）还是补一个节点——**这一步是平台的责任**。

对平台的推论有三条：

- **全员到齐才算开始**。$$W$$ 个 Pod 里只起了 $$W-1$$ 个，起来的那些在 rendezvous 里等着，占着 GPU 不干活。等待的时间越长浪费越大，如果最后一个 Pod 永远等不到资源，就是死锁的开端（第三篇）。
- **失败的单位是任务，不是 Pod**。Kubernetes Job 的 `backoffLimit` 数的是单个 Pod 的重启次数；训练任务需要的是"任何一个 Pod 失败 → 整组 Pod 一起重建"。
- **重启后要能从 checkpoint 继续**。`--max-restarts` 之内的重启由 `torchrun` 处理，但重启后的进程是全新的，它们的状态来自上一次 checkpoint。checkpoint 多久写一次，决定一次故障丢多少进度，也决定平台抢占它的代价。

### 4. 四条资源需求

把上面的形态翻译成对硬件的要求：

**整数个 GPU，且每个进程一张**。训练进程的显存占用（参数、梯度、优化器状态、激活）是按整卡规划的，没有"半张卡"的用法；也没有两个训练进程共享一张卡的用法——NCCL 的通信假定每个 rank 独占它的设备。对 4 节点 32 卡任务，需求是"4 个 Pod，每个 Pod 恰好 8 张卡，且这 8 张卡在同一节点上"。

**节点间高带宽、低延迟网络**。数据并行的 all-reduce 每步要在节点间交换与梯度同量级的数据，一个 70B 模型的 bf16 梯度是 140 GB 量级，按 ring all-reduce 每个节点收发约 $$2 \times$$ 这个量的 $$(N-1)/N$$。走 TCP overlay 的几十 Gbps 与走 RDMA 的几百 Gbps，差的是这一步要几秒还是几十秒——直接反映在 MFU 上。此外节点间的**拓扑**有区别：同一机柜或同一 spine 下的节点之间带宽高、跨 spine 要争抢上行链路。这两条构成第五篇（RDMA 进容器）和第三篇（拓扑感知调度）的需求来源。

**周期性的大块顺序写（checkpoint）**。一个训练任务每隔十几到几十分钟把完整状态写一遍，量级从几十 GB（小模型）到 1 TB 级（70B 加 Adam 状态）。写的时候训练暂停或降速，所以要求的是**突发写带宽**：1 TB 在 1 分钟内写完是约 17 GB/s 的聚合写入。平均带宽很低——大部分时间不写——这与按平均带宽选型的存储直觉相反。

**持续的小块随机读（数据集）**。训练数据是海量小文件或大文件内的随机偏移读，每步读一个 batch，量不大但持续、随机、并发（每个 rank 各自读）。它要的是 IOPS 与元数据性能，与 checkpoint 要的吞吐是两种存储画像，第五篇会讨论它们能不能用同一套存储。

### 5. 训练任务的生命周期

最后是时间维度。训练任务是**有限时长的批处理**：提交、排队、被调度、运行数小时到数周、结束或失败、释放全部资源。它的运行时间由数据量和算力决定，与外部请求无关；它在排队时不占资源，在运行时独占资源，中间没有"部分运行"的状态（弹性训练是例外，但改变 $$W$$ 要重新 rendezvous，是重启而非平滑扩缩）。

```text
提交 ──▶ 排队（等配额 / 等资源凑齐）──▶ 全员调度 ──▶ rendezvous ──▶ 运行 ──▶ 完成 / 失败 ──▶ 释放全部 GPU
                                                      │                  │
                                                      │                  └─ 每 T 分钟写一次 checkpoint
                                                      └─ 任一 Pod 失败 → 整组重启 → 从 checkpoint 恢复
```

这个生命周期对平台的含义是：**排队是正常状态**，需要队列、配额和公平性；**运行中被打断的代价可量化**——上次 checkpoint 之后的所有算力白费——所以抢占策略要知道这个代价；**结束后资源整体归还**，适合按任务而非按时间切片记账。


## 三、推理服务的形态与需求

### 1. 一个副本的三种形态

推理服务的基本单位是**副本**（replica）：一个能独立服务完整请求的引擎实例。副本的物理形态取决于模型放不放得进一张卡：

```text
形态                     模型规模（bf16 权重）     K8s 对象          备注
──────────────────────  ──────────────────────  ───────────────  ─────────────────────────────────────
一个进程，一张卡          ≲ 60 GB（7B–30B）         Pod × 1           最简单；一张 80 GB 卡放权重 + KV cache
一个进程，一节点多卡       60 GB–600 GB（70B–405B）  Pod × 1，多 GPU    TP 在节点内走 NVLink；--tensor-parallel-size=N
多个 Pod，一个副本        单节点放不下                Pod × M（一组）    TP / PP 跨节点；M 个 Pod 是一个整体，缺一不可
```

第三种形态与训练任务的形状一样：一组 Pod 强耦合、同时起、同时停、有 leader 有 worker。但它是**长驻服务**而不是批处理，需要滚动升级、按组重启、按组扩缩。Deployment 的副本是单 Pod，StatefulSet 的序号是单 Pod——原生对象里没有"多 Pod 一个副本"的概念，这是 LeaderWorkerSet 存在的原因（第六篇）。

本文核心问题里的 TP=2 服务属于第二种：每个副本是一个 Pod，请求 2 张卡，最好在 NVLink 相连的一对上。

### 2. prefill 与 decode

一个 LLM 请求在引擎里分两个阶段。**prefill** 把整段 prompt 一次算完，生成第一个 token，是计算密集的——几千个 token 并行通过模型，GPU 的算力被填满。**decode** 之后每步只生成一个 token，每步都要把全部权重从显存读一遍，是显存带宽密集的——GPU 算力大部分时间空着，瓶颈是每秒能读多少次权重。

这个差别对平台有两个后果。第一，同一张卡上 prefill 与 decode 混跑时互相干扰：一个长 prompt 的 prefill 会让正在 decode 的所有请求那一步变慢，用户看到的是输出突然卡一下。第二，两个阶段的最优硬件不同：prefill 要算力，decode 要显存带宽和容量。PD 分离（prefill 与 decode 跑在不同的 Pod 组里，中间传 KV cache）把它们拆开，各自扩缩——这意味着**一个服务有两种副本、两套扩缩容指标、副本之间还要传数据**，第六篇和第七篇会处理它的部署与路由形态。本文只需要知道：推理服务的"副本"可能不止一种。

对应两个阶段的两个延迟指标贯穿交付层三篇：**TTFT**（time to first token，prefill 阶段加排队时间）与 **TPOT**（time per output token，decode 每步的时间）。vLLM v0.23.0 在 `vllm/v1/metrics/loggers.py` 的 `PrometheusStatLogger` 里以 `vllm:time_to_first_token_seconds` 和 `vllm:inter_token_latency_seconds` 两个直方图暴露它们。

### 3. 显存是硬约束

一个副本的显存由两部分组成：**权重**（模型大小 × 每参数字节，固定）与 **KV cache**（每个在处理的 token 的 key/value，随并发与上下文长度增长）。引擎启动时按 `--gpu-memory-utilization`（vLLM v0.23.0 `vllm/config/cache.py` 的 `CacheConfig.gpu_memory_utilization`，默认 0.92）预留这一比例的显存，权重占掉一块，**剩下的全部划给 KV cache**。KV cache 的大小决定同时能服务多少请求；KV cache 满了，新请求进等待队列，`vllm:num_requests_waiting` 上升，TTFT 拉长。

这带来三条平台约束：

- **不能超卖**。显存不像 CPU 可以时间片轮转，也不像内存可以 swap。两个引擎进程被放到同一张卡上、各自按 92% 预留，第二个直接 OOM。平台要么给整卡，要么用有显存隔离的切分方式（第四篇）。
- **模型越大，权重占比越高，能服务的并发越低**。一张 80 GB 的卡跑 7B bf16 模型，权重 14 GB、KV cache 可用约 60 GB；跑 30B 模型，权重 60 GB、KV cache 只剩十几 GB。同样的卡，后者的副本数要多好几倍才能撑住同样的并发。容量规划不能只数卡。
- **小模型浪费大**。7B 模型 14 GB 权重，如果流量不高，KV cache 用不到 60 GB，这张卡的大部分显存与算力空着。这是切分（MIG / HAMi / 时间片）的需求来源——但切分与"不能超卖"直接冲突，第四篇的全部内容是在这两者之间找位置。

### 4. 扩容一个副本要多久

一个 Web 服务的副本扩容以秒计：调度、拉一个几百 MB 的镜像、进程起来、就绪探针通过。推理副本的扩容多了一个主项：**把几十到几百 GB 的权重从存储读进显存**。

```text
阶段                       典型耗时              取决于
────────────────────────  ──────────────────  ────────────────────────────────────
调度 + 等 GPU 空出           秒到分钟             有没有空卡；要不要等节点扩容（云上加节点是分钟级）
拉镜像                     分钟级（10 GB 级镜像）  镜像大小；节点有没有缓存；registry 带宽
加载权重到显存              分钟级（70B ≈ 140 GB） 存储读带宽（对象存储 / PFS / 本地 NVMe）；节点网卡
引擎初始化（profiling、图捕获）几十秒到分钟          模型与引擎配置
预热                       几十秒               第一批请求触发的编译、缓存
```

一个 70B 模型的副本从"决定扩容"到"能接流量"以分钟计，常见的数字是 5–10 分钟。这有两个后果。第一，**扩缩容必须提前触发**：等队列已经排起来再扩，新副本就绪时高峰已经过了；这要求预测式或阈值很低的扩容策略（第六篇）。第二，**缩容到零很贵**：省下的是空闲时的 GPU 费用，付出的是下一个请求要等几分钟。就绪探针要等到权重加载与预热完成才能返回成功，否则请求会被路由到一个还在加载的副本上。

### 5. 负载信号在引擎内部

Kubernetes 原生的 HPA 以 CPU 与内存利用率为默认信号。对推理服务这两个信号都无意义：GPU 上的 kernel 在跑时 CPU 几乎空闲，内存（宿主机内存）也与负载无关。GPU 利用率（DCGM 的 `DCGM_FI_DEV_GPU_UTIL`）看起来相关，但它只表示"这段时间内有没有 kernel 在跑"，一个 decode 请求就能让它接近 100%，与"还能接多少请求"无关（第八篇）。

真正能说明负载的数字在引擎内部。vLLM v0.23.0 的 `vllm/v1/metrics/loggers.py` 里 `PrometheusStatLogger` 定义的指标中，与扩缩容和路由直接相关的是：

```text
vllm:num_requests_running          正在 decode 的请求数
vllm:num_requests_waiting          在等待队列里的请求数           ← 扩容的主信号
vllm:kv_cache_usage_perc           KV cache 使用比例 0–1           ← 饱和度；路由的副本选择依据
vllm:prefix_cache_hits / _queries  prefix cache 命中与查询次数     ← 同前缀请求去同副本的收益
vllm:num_preemptions               因显存不足被抢占（重算）的请求数
vllm:time_to_first_token_seconds   TTFT 直方图
vllm:inter_token_latency_seconds   TPOT 直方图
vllm:request_queue_time_seconds    请求在队列里等了多久
```

平台要做的是把这些指标从每个副本采出来、聚合、接到扩缩容控制器（KEDA 的 Prometheus scaler 或 HPA 的 external metrics）和路由器（Endpoint Picker 直接抓每个副本的 `/metrics`）。这条链路上每个环节——采集周期、聚合方式、阈值——都会影响扩容延迟和路由质量。指标由引擎定义、由平台消费，是引擎与交付层之间的接口，第六、七、八篇反复用到。

### 6. 推理服务的生命周期

与训练相反，推理服务是**长驻**的：它的运行时间由业务决定，负载随时间波动，副本数随负载变化，版本要在不停服的情况下更新。

```text
部署 ──▶ 副本就绪（加载权重）──▶ 接流量 ──▶ 随负载扩缩 ──▶ 滚动升级（新旧版本并存）──▶ ... ──▶ 下线
                                        │
                                        └─ 低谷时缩到最少副本或零；高峰前提前扩容
```

它对平台的含义：**没有"结束"**，成本是持续的，空闲副本的费用与忙时一样；**扩缩是常态**，每一次扩容都是一次几分钟的冷启动；**升级是有状态的**——一个副本在处理流式请求时被终止，用户看到的是回复中断，所以替换副本要等在途请求完成。


## 四、两组矛盾的需求

### 1. 对照表

把第二、三章的需求并排放：

```text
维度              训练任务                                  推理服务
───────────────  ────────────────────────────────────────  ────────────────────────────────────────
资源粒度          整卡，整节点，越多越好                        整卡或一部分卡；小模型只要一张卡的几分之一
调度单位          W 个 Pod 一组，全员到齐才开始                 每个副本独立就绪；多 Pod 副本时组内 gang
拓扑              极度敏感：节点内 NVLink、节点间同机柜           TP 在节点内敏感；副本之间无关，越分散越好（容灾）
时长              有限：小时到周                                无限：长驻
资源变化          运行中不变（改 W 要重启）                       副本数随负载分钟级变化
被打断的代价       丢上次 checkpoint 之后的进度                   丢在途请求；一个副本重启是分钟级冷启动
负载信号          无（跑满就是目标）                              引擎内部的队列长度与 KV cache 使用率
网络              节点间 RDMA 带宽是瓶颈                         节点间流量小；南北向的请求流量与流式长连接
存储              周期性突发写 + 持续随机读                       启动时一次性大块顺序读（权重）
排队              正常状态，可以等几小时                          不能等：排队直接是用户看到的 TTFT
成本视角          按任务：卡数 × 时长                           按服务：副本数 × 时长，折算到每百万 token
```

### 2. 三对矛盾

**独占 vs 共享**。训练要整卡整节点独占，NCCL 假定它是设备唯一的用户；推理的小模型要几个服务共享一张卡才划算。同一个集群里两者共存，切分策略只能按节点池划：这几台切、那几台不切，改一台机器的 MIG 配置要清空它上面的所有进程。

**拓扑 vs 弹性**。训练要 32 张卡尽量聚在拓扑上相近的位置，宁可多等一会儿也不要跨 spine；推理要副本随时能扩出来，哪里有空卡就用哪里，分散反而好。一个偏好"凑齐再给"的调度器与一个偏好"有就给"的调度器，在同一个资源池上要做的决定相反。

**批处理 vs 长驻**。训练的资源用完归还，适合排队、配额、公平分享；推理常驻，占着的资源不会自己释放，只能靠缩容。训练排队时被抢占是可接受的（丢一次 checkpoint 的进度），推理副本被抢占是用户可见的故障。混部时谁能抢谁、抢的代价怎么算，是配额与优先级设计的核心。

### 3. 共同点

矛盾之下有三个共同点，它们是平台设计能统一的部分：

- **两者都要"多 Pod 一个单位"**。训练的 $$W$$ 个 Pod、跨节点推理副本的 $$M$$ 个 Pod，都需要同时调度、同时销毁、稳定的组内寻址。这是 gang 语义与 LeaderWorkerSet / JobSet 这类"Pod 组"抽象共用的基础。
- **两者都要 GPU 有属性**。训练要同型号同拓扑，推理要按显存大小选卡。device plugin 的"计数"模型对两者都不够，DRA 的属性选择对两者都有用。
- **两者的成本都要落到 GPU 时间上**。训练按任务算、推理按 token 算，最后都要分摊到"哪张卡在哪段时间归谁"。可观测与成本的底层是同一份数据。

这也是为什么本系列把两者放在同一个平台里讨论：资源层的大部分机制对两者通用，差别集中在调度策略与交付层。


## 五、原生 Kubernetes 的假设与空缺

### 1. kube-scheduler：一个 Pod 一个 Pod 地调度

kube-scheduler 的工作单位是**单个 Pod**。它从队列里取一个 Pod，跑一遍 Filter（哪些节点放得下）与 Score（放哪个最好），选一个节点绑定，然后取下一个。它不知道这个 Pod 与另外 31 个 Pod 是一组，也不会因为第 31 个 Pod 放不下就撤回前 30 个的绑定。

Filter 阶段对 GPU 的判断在 `pkg/scheduler/framework/plugins/noderesources/fit.go`（Kubernetes v1.37.0）的 `Fits()` 与 `fitsRequest()`：把 Pod 请求的每种资源与节点 `allocatable` 减去已分配量比较，任一不足就返回一个 `InsufficientResource`，`Reason` 字段对扩展资源就是 `Insufficient <资源名>`。开头那行 `Insufficient nvidia.com/gpu` 就是这里生成的。对调度器来说，`nvidia.com/gpu` 只是一个字符串键加一个整数——与 `example.com/foo` 没有任何区别。

这个模型对训练任务的后果是：32 个 Pod 逐个调度，前 30 个成功、后 2 个 Pending，前 30 个占着 240 张卡等待；另一个任务的 Pod 恰好拿走了剩下的卡，两个任务都凑不齐、都不释放——死锁。原生的缓解手段只有 `PriorityClass` 与抢占，但抢占也是按单 Pod 决定的。**gang scheduling 必须由一个知道"组"的调度器或准入控制器实现**，这是第三篇。

kube-scheduler 也不知道拓扑。它有 `nodeAffinity` 与 `podAffinity`，可以表达"这些 Pod 要在有某个 label 的节点上"，但不能表达"这 4 个 Pod 要在**同一个**机柜里，哪个机柜都行"——`podAffinity` 的 `topologyKey` 只能把 Pod 往已有 Pod 所在的域聚，第一个 Pod 落哪里是随机的，而且它是逐 Pod 判断，不会为了整组的拓扑退回重选。

### 2. device plugin：计数模型

节点上 `allocatable` 里的 `nvidia.com/gpu` 是 device plugin 上报的。机制的核心是两个 gRPC 调用（Kubernetes v1.37.0 `staging/src/k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1/api.proto`）：`ListAndWatch` 把设备列表流式推给 kubelet，kubelet 据此更新节点 `status.capacity` / `status.allocatable`；`Allocate` 在 Pod 启动时被调用，返回要注入容器的设备文件与环境变量。kubelet 侧的 `pkg/kubelet/cm/devicemanager/manager.go` 的 `ManagerImpl` 维护每个资源名的设备集合与已分配映射。

这个模型的表达能力是**每种资源名一个整数**。它能说"这个节点有 8 个 `nvidia.com/gpu`"，不能说"其中 4 个是 A100、4 个是 H100"、"这 8 个的 NVLink 拓扑是什么"、"每个有多少显存"。Pod 侧也只能说"我要 2 个"，不能说"我要 2 个有 NVLink 直连的"或"我要 1 个显存 ≥ 40 GB 的"。要区分型号只能靠节点 label 加 `nodeSelector`，把属性从设备挪到节点上，前提是节点内的卡同构。

它也不能共享：一个设备 ID 只能分配给一个容器。要让两个 Pod 用同一张卡，device plugin 只能"撒谎"——把一张卡上报成 N 个虚拟设备（时间片模式），代价是没有任何隔离。

DRA（`resource.k8s.io/v1`，Kubernetes v1.37.0 的 `staging/src/k8s.io/api/resource/v1/types.go`）用 `ResourceSlice` 发布带属性的设备、用 `ResourceClaim` 里的 CEL 表达式按属性选择，是对这个模型的替代。device plugin 与 DRA 的机制、边界与当前状态是第二篇的内容；本文只需要记住：**默认情况下 GPU 在 Kubernetes 里是一个不透明的整数**。

### 3. 一张网卡

CNI 给每个 Pod 一张 veth 网卡，接到 overlay 或 underlay 网络。NCCL 要的不是这张网卡：它要打开 `/dev/infiniband/*` 下的 RDMA 设备、注册 GPU 显存做 GPUDirect RDMA、绕过内核网络栈。这两件事默认的 Pod 都做不到——设备文件不在容器里，RDMA 设备的网络命名空间归属也要配置。给 Pod 第二张网卡（Multus）、把 RDMA 设备作为扩展资源分配给 Pod（RDMA device plugin）、在宿主机上装对的驱动（Network Operator），三件事缺一个 NCCL 就回退到 TCP，带宽掉一个数量级而**不报错**。第五篇处理这条链。

存储侧原生的 PVC 与 CSI 抽象足够：checkpoint 写到 PVC、数据集从 PVC 读。缺的不是接口而是**容量规划**：Kubernetes 不知道这个卷要在 1 分钟内吃下 1 TB 的突发写，也不知道扩容 6 个副本时会有 6 个 Pod 同时读同一份 140 GB 的权重。这是存储选型的问题，第五篇给算术。

### 4. Deployment、Service 与 HPA

交付侧的三个原生对象各有一个不成立的假设。

**Deployment** 假设副本是一个 Pod，Pod 之间可互换、无序、独立就绪。跨节点的 TP 副本是 $$M$$ 个 Pod、有 leader / worker 角色、组内要稳定寻址、要一起滚动。StatefulSet 给了序号与稳定 DNS，但序号是 Pod 级的，没有"组"。

**Service** 假设后端可互换，kube-proxy 按轮询或随机分发。推理副本不可互换：一个副本 KV cache 已满、另一个空着，轮询会把请求送到满的那个排队；同一前缀的请求去同一副本能命中 prefix cache，轮询完全不管。请求还是流式长连接，时长差百倍，连接数均衡不等于负载均衡。Gateway API 的 `HTTPRoute` 能按路径与 header 路由、能按权重分流，但选哪个后端 Pod 还是 Service 的事——除非把后端换成 `InferencePool`（第七篇）。

**HPA** 假设信号是 CPU / 内存，或通过 metrics API 接入的自定义指标。前者对推理无意义；后者要一条从引擎 `/metrics` 到 Prometheus 到 metrics adapter 的管线，HPA 自身不提供。HPA 也不能缩到零。KEDA 补的正是这两条（第六篇）。

### 5. 空缺清单

汇总起来，原生 Kubernetes 在 AI 负载面前有三种性质的空缺：

```text
性质            表现                                                    填法
─────────────  ──────────────────────────────────────────────────────  ─────────────────────────────
缺一个插件      GPU 不被识别（Insufficient nvidia.com/gpu）；RDMA 设备不在容器里    device plugin / Operator，机制不变
缺一个概念      "一组 Pod"（gang、多 Pod 副本）；"设备的属性"；"配额与队列"          新 CRD + 控制器，或调度器扩展；改变对象模型
缺一个信号      扩缩容看不到引擎队列；路由看不到副本状态；成本看不到 token           指标管线 + 扩展点（KEDA、EPP、OpenCost）
```

第一类装上就好，第二类要改用新的对象来表达工作负载，第三类要把引擎与平台之间的信息通道建起来。后面七篇大致按这个难度递进。


## 六、平台的两层拆分

### 1. 资源层：输入与输出

资源层的任务是**把裸节点变成引擎能直接用的运行环境**。

```text
输入   节点：装了驱动的 GPU、RDMA 网卡、本地 NVMe；引擎的请求：N 张某型号的卡、M 个 Pod 一组、要 RDMA、要挂某个卷
输出   一组已调度、已启动的 Pod，容器里看得到 GPU 与 RDMA 设备、挂上了存储、彼此能以稳定 DNS 名互访、环境变量齐全
```

它内部又分四个子层，自下而上对应第二到第五篇：

- **设备层**（第二篇）：让容器看到 GPU。驱动与 CUDA 的兼容契约、Container Toolkit 把设备文件与用户态库注入容器、device plugin 或 DRA 把设备变成可调度的资源、GPU Operator 把这一切装到每个节点、镜像里放什么不放什么。GPU Operator v26.7.0 的 `ClusterPolicy`（`api/nvidia/v1/clusterpolicy_types.go` 的 `ClusterPolicySpec`）是这一层的总开关。
- **调度层**（第三篇）：决定哪组 Pod 在什么时候上哪些节点。Kueue v0.19.2 是"调度器之前的配额闸门"——`ClusterQueue` / `LocalQueue` / `ResourceFlavor` / cohort（`apis/kueue/v1beta2/clusterqueue_types.go` 的 `ClusterQueueSpec.CohortName`）决定一个 Workload 何时被准入，准入后交给 kube-scheduler；Volcano v1.15.2 是"一个更懂批处理的调度器"——`PodGroup`（`MinMember`）与 `Queue`（`capability` / `deserved` / `guarantee`）加 action-plugin 流水线，替换 kube-scheduler 做 gang 与公平分享。任务的表达由 Kubeflow Trainer v2.3.0 的 `TrainJob` / `TrainingRuntime`（`pkg/apis/trainer/v1alpha1`，基于 JobSet）与 KubeRay v1.7.0 的 `RayJob`（`ray-operator/apis/ray/v1`）承担。
- **切分层**（第四篇）：让一张卡被多个 Pod 用。MIG 是硬件分区（GPU Operator 的 MIG Manager 按 `nvidia.com/mig.config` 节点标签配置）、时间片与 MPS 是 device plugin 的 `sharing` 配置（k8s-device-plugin v0.20.0 的 `api/config/v1`）、HAMi v2.10.0 是软件层的显存与算力限制。
- **网络与存储层**（第五篇）：Network Operator v26.7.0 的 `NicClusterPolicy` 装 RDMA 驱动栈、Multus v4.3.0 给 Pod 第二张网卡、k8s-rdma-shared-dev-plugin v1.5.4 把 RDMA 设备作为资源分配；存储通过 CSI 接并行文件系统、对象存储与缓存层。

### 2. 交付层：输入与输出

交付层的任务是**把一组能跑的引擎 Pod 变成一个可以卖给用户的服务**。

```text
输入   资源层交出的引擎 Pod（或声明式的"我要 N 个这样的副本"）；请求流量；租户与配额策略
输出   一个稳定的服务端点：按模型名路由、按租户限流、副本随负载伸缩、有 TTFT/TPOT 的 SLO、有按 token 的账单
```

三个子层自内向外对应第六到第八篇：

- **Serving 层**（第六篇）：副本的生命周期。LeaderWorkerSet v0.10.0 的 `LeaderWorkerSet`（`api/leaderworkerset/v1/leaderworkerset_types.go` 的 `LeaderWorkerSetSpec.LeaderWorkerTemplate`）表达多 Pod 副本；KServe v0.20.0 的 `InferenceService`（`serving.kserve.io/v1beta1`）面向传统模型、`LLMInferenceService`（`serving.kserve.io/v1alpha1`，`pkg/apis/serving/v1alpha1/llm_inference_service_types.go`）面向 LLM 并内置 llm-d 的架构；llm-d v0.9.0 是 vLLM 与 Kubernetes 社区共同推动的 LLM Serving 栈；KubeRay 的 `RayService` 把 Ray Serve 放到 Kubernetes 上；KEDA v2.20.2 的 `ScaledObject`（`apis/keda/v1alpha1/scaledobject_types.go`，`triggers` 字段接 Prometheus scaler）按引擎指标扩缩容。
- **网关层**（第七篇）：请求的路由与治理。Gateway API Inference Extension v1.6.0 的 `InferencePool`（`inference.networking.k8s.io/v1`，`api/v1/inferencepool_types.go`，`EndpointPickerRef` 字段指向 EPP）替代 Service 作为 `HTTPRoute` 的后端；Endpoint Picker（EPP）在 v1.6.0 已迁出到 llm-d-router v0.10.0（`cmd/epp`、`pkg/epp`），按每个副本的队列、KV cache 使用率与已加载的 LoRA 选目标；`InferenceObjective` 在 llm-d-router 的 `apix/v1alpha2` 表达请求优先级。多租户、配额、灰度建立在这个网关之上。
- **可观测与成本层**（第八篇）：DCGM Exporter 4.6.0-4.8.3 的硬件指标（`etc/default-counters.csv` 里的 `DCGM_FI_DEV_GPU_UTIL`、`DCGM_FI_DEV_FB_USED`；`DCGM_FI_PROF_SM_ACTIVE` 在同一文件中默认注释掉，需要打开）、引擎的 `vllm:*` 指标、网关的 token 计数，三层拼成从硬件到请求的链条；OpenCost v1.121.1（`pkg/costmodel` 的 allocation 逻辑含 GPU 字段）按 label 把 GPU 成本分摊到团队与模型。

### 3. 两层之间的接口

两层的分界线是**"Pod 能跑了"**。资源层不知道 Pod 里跑的是训练还是推理、服务的是谁、值多少钱；交付层不知道 GPU 是怎么分出来的、网卡是怎么进容器的。它们通过三样东西耦合：

```text
资源请求        交付层用 K8s 的资源语言（nvidia.com/gpu: 2、nodeSelector、ResourceClaim）向资源层要卡；资源层不解释用途
Pod 组抽象      LeaderWorkerSet / JobSet 定义"哪些 Pod 是一组"，调度层（Kueue / Volcano）据此做 gang；两层共用这个概念
指标            DCGM（资源层产出）与 vllm:*（引擎产出）在可观测层汇合；成本分摊要同时用到"谁占了卡"与"谁发了请求"
```

训练任务只走资源层：`TrainJob` 提交、Kueue 准入、Pod 起来、跑完释放，交付层的三篇对它几乎没有内容（除了第八篇的任务级可观测与成本）。推理服务两层都走：先由资源层给出 Pod，再由交付层包装成服务。这也是为什么总纲说前四篇按"一个训练任务从提交到跑起来"推进、后三篇按"一个推理请求从进入到计费"推进。

### 4. 组件全景图按层落位

把总览里的图展开成表，每个组件标注它填的是第五章哪类空缺、解决第二、三章哪条需求：

```text
层        组件（版本）                                   填的空缺                    解决的需求                          篇
────────  ──────────────────────────────────────────  ─────────────────────────  ────────────────────────────────  ──
设备      NVIDIA GPU Operator v26.7.0                  缺插件：驱动、Toolkit、插件      容器看到 GPU；节点标签有型号           2
          k8s-device-plugin v0.20.0                    缺插件：GPU 作为资源            整数个 GPU 可请求                    2
          DRA（Kubernetes v1.37.0，resource.k8s.io/v1） 缺概念：设备属性              按显存 / 型号 / 拓扑选卡               2
          NVIDIA Container Toolkit v1.20.0             缺插件：设备注入容器            容器内有 /dev/nvidia* 与 libcuda      2
调度      Kueue v0.19.2                                缺概念：队列、配额、gang 准入     全员到齐才开始；团队配额与借用          3
          Volcano v1.15.2                              缺概念：批调度器                gang、公平分享、拓扑                  3
          Kubeflow Trainer v2.3.0                      缺概念：训练任务对象            RANK / MASTER_ADDR 注入；整组重启       3
          KubeRay v1.7.0                               缺概念：Ray 集群对象            Ray 任务与服务的表达                  3、6
切分      MIG（GPU Operator MIG Manager）               缺概念：一张卡的一部分           小模型共卡，硬件隔离                  4
          时间片 / MPS（device plugin sharing）          缺概念：一张卡的一部分           开发环境共卡，无 / 弱隔离              4
          HAMi v2.10.0                                 缺概念：按显存与算力比例请求       小模型共卡，软件隔离                  4
网络存储   NVIDIA Network Operator v26.7.0              缺插件：RDMA 驱动栈             节点间 RDMA                        5
          Multus CNI v4.3.0                            缺概念：第二张网卡              RDMA 网卡进 Pod                     5
          k8s-rdma-shared-dev-plugin v1.5.4            缺插件：RDMA 设备作为资源         /dev/infiniband 进容器              5
          CSI（PFS / 对象存储 / 缓存层）                  缺规划：突发写与并发读带宽        checkpoint 写、数据集读、权重分发       5
Serving   LeaderWorkerSet v0.10.0                      缺概念：多 Pod 一副本            跨节点 TP / PP 副本；PD 分离            6
          KServe v0.20.0                               缺概念：推理服务对象            InferenceService / LLMInferenceService  6
          llm-d v0.9.0                                 缺概念：LLM Serving 栈          PD 分离、KV 感知路由的整体方案          6、7
          KEDA v2.20.2                                 缺信号：按引擎指标扩缩            vllm:num_requests_waiting 驱动扩容    6
网关      Gateway API Inference Extension v1.6.0        缺概念：InferencePool           副本不可互换的后端集合                 7
          llm-d-router v0.10.0（EPP）                   缺信号：按副本状态选后端          KV cache / 队列 / LoRA 感知路由        7
可观测     DCGM Exporter 4.6.0-4.8.3                    缺信号：硬件指标                分配率 vs 使用率                    8
          OpenCost v1.121.1                            缺信号：成本分摊                每团队、每模型、每百万 token 的成本      8
```

这张表是全系列的"机制线"。每一篇会把它所在的行展开成一章。

### 5. 同位替代：Slurm 与 Ray

两个系统经常被问到"它们在这张图上哪里"。答案是：它们不是图上的一层，而是**资源层调度部分的替代实现**——所以叫同位替代。

**Slurm** 是 HPC 集群二十年的标准答案。`sbatch` 提交、partition 是队列、GRES 表达 GPU、`--exclusive` 独占节点、backfill 调度填空隙、拓扑插件按交换机层级放任务——第三章训练任务的全部需求它都原生满足，gang 是它的默认语义而非扩展。它的短板恰好是推理侧：没有 Service、没有滚动升级、没有自动扩缩容、容器化是后来加的。HPC 出身的训练团队选 Slurm 是自然的，问题出在同一个集群还要跑推理服务的时候。Slinky 项目的 slurm-operator v1.2.2（`api/v1beta1` 的 `Controller`、`NodeSet`、`LoginSet` 等 CRD）把 Slurm 的控制面与计算节点作为 Kubernetes 对象来运行，是两个世界的一种桥接方式；第三篇会把它与 Kueue、Volcano 放在同一个核心问题下对照。

**Ray** 是另一个方向：它自己有调度器（placement group、资源标签、actor），自己管进程组，自己做弹性。Ray Train 跑训练、Ray Serve 跑推理，都在 Ray 的抽象里。放到 Kubernetes 上，KubeRay v1.7.0 的 `RayCluster` / `RayJob` / `RayService`（`ray-operator/apis/ray/v1` 的 `RayClusterSpec` 等）申请节点级资源，Ray 在拿到的节点里再做细粒度调度——**两层调度器**。它适合 Python 逻辑重、多阶段流水线的场景；代价是 Ray 拿到的资源在 Kubernetes 看来已经"用掉了"，两层之间的利用率信息不通。

两者的定位可以用一句话概括：Slurm 替代的是第三篇的调度层（并且顺带替代了任务表达），Ray 替代的是第三篇的调度层加第六篇的 Serving 层（但都在自己的边界内）。本系列以 Kubernetes 生态为主线，两者只在第三篇与第六篇作对照。


## 七、核心问题：两张需求表

### 1. 4 节点 32 卡的训练任务

设定：一个 70B 级模型的预训练，4 台 8 卡节点，节点内 TP=8、节点间 DP=4；每 30 分钟写一次 checkpoint；所属团队配额 32 卡，另一团队配额也是 32 卡，共用一个 64 卡池。

```text
#   需求                                                   原生 Kubernetes 能否满足                                          补缺篇
──  ───────────────────────────────────────────────────  ──────────────────────────────────────────────────────────────  ─────
1   4 个 Pod，每个恰好 8 张卡，全在一个节点上                 部分：resources.limits 能表达 8，但 nvidia.com/gpu 要先由插件上报    2
2   32 张卡型号一致（都是 H100，不能混 A100）                部分：靠 nodeSelector 按节点标签选；标签要 GFD 打；设备本身无属性      2
3   8 张卡之间 NVLink 全连接（整节点独占即可保证）             是（整节点独占时）；非整节点时插件不保证给出拓扑最优的 8 张             2、4
4   4 个 Pod 同时调度，凑不齐就一个也不占                     否：kube-scheduler 逐 Pod 调度，会出现 30/32 占着等                3
5   4 个节点在同一机柜 / 同一 spine 下                       否：podAffinity 不能表达"任意一个同域"，也不为整组回退               3
6   团队配额 32 卡；对方空闲时可借用、对方要用时归还              否：ResourceQuota 是硬上限，没有队列、借用与归还                   3
7   Pod 与 RANK 的稳定映射；MASTER_ADDR 在 worker 起来前可解析   是：Indexed Job + headless Service；但需要控制器注入变量           3
8   任一 Pod 失败，整组重建并从 checkpoint 恢复               否：Job 的 backoffLimit 是单 Pod 语义，没有组级重启                 3
9   被抢占时优先抢刚写完 checkpoint 的任务                    否：PriorityClass 抢占是单 Pod、不知道 checkpoint                  3
10  节点间 RDMA；NCCL 不回退到 TCP                          否：CNI 只给一张 veth；/dev/infiniband 不在容器里                    5
11  checkpoint 突发写：1 TB 级在 1 分钟内落盘                接口是（PVC）；带宽否：K8s 不做存储容量规划                          5
12  数据集持续随机读，32 个 rank 并发                        接口是（PVC）；性能取决于存储                                       5
13  任务结束整体释放 32 张卡                                 是：Job 完成即释放                                                —
14  任务级 GPU 利用率、每 rank 的 step 时间、排队时长          否：需要 DCGM + 按任务聚合的指标管线                                8
15  成本按任务分摊到团队                                     否：需要成本模型与 label 分摊                                       8
```

15 条里原生完全满足的只有第 13 条。第 1、2、3、7、11、12 条是"接口有、能力缺"——Kubernetes 提供了字段，但字段背后的东西（设备上报、节点标签、存储带宽）要平台补。其余 8 条是模型层面的缺失，集中在调度（4–9）与网络（10）。

### 2. TP=2、副本数动态变化的推理服务

设定：一个 30B 级模型，bf16 权重约 60 GB，TP=2 放在两张 80 GB 卡上；白天 6 个副本、夜间 1 个；同一模型名下要灰度一个 FP8 版本；两个租户各有 TPM 配额。

```text
#   需求                                                   原生 Kubernetes 能否满足                                          补缺篇
──  ───────────────────────────────────────────────────  ──────────────────────────────────────────────────────────────  ─────
1   每副本一个 Pod、2 张卡、同节点、最好 NVLink 相连            部分：2 张卡能请求；是否 NVLink 相连取决于插件的分配策略               2、4
2   显存独占：这 2 张卡上不能有别的进程                       是（整卡分配时）；一旦切分共享，隔离由切分方案决定                     4
3   小流量时段该模型用不满 2 张 80 GB 卡，想让别人用剩余部分     否：device plugin 不能把一张卡的一部分给别人                        4
4   副本数按 vllm:num_requests_waiting 扩缩                 否：HPA 默认只看 CPU / 内存；自定义指标要自建管线                    6
5   高峰前提前扩容（一个副本就绪要 5 分钟以上）                 否：HPA 是反应式的；需要预测或低阈值策略                             6
6   夜间缩到 1 甚至 0，且第一个请求能接受几分钟冷启动            部分：缩到 1 可以；缩到 0 HPA 不支持                               6
7   就绪探针等到权重加载与预热完成                             是：readinessProbe 打引擎的 /health                                6
8   扩容时 6 个 Pod 并发读同一份 60 GB 权重                    接口是；带宽否                                                    5
9   滚动升级时等在途流式请求完成再终止旧副本                    部分：terminationGracePeriodSeconds + preStop；引擎要配合 drain      6
10  请求不轮询：按每副本的 KV cache 使用率与队列长度选副本        否：Service 是轮询 / 随机；不知道副本状态                           7
11  同前缀请求去同一副本（prefix cache 命中）                  否：Service 无会话亲和到内容                                       7
12  按请求体的 model 字段路由到对应模型的副本集合                否：HTTPRoute 按路径 / header 路由，不解析 body                    7
13  BF16 与 FP8 两个版本按 9:1 分流，按 header 定向            部分：HTTPRoute 权重与 header 匹配可做；后端仍要能区分版本            7
14  租户 A 每分钟 100 万 token、租户 B 30 万；超了排队而非拒绝    否：没有 token 计数，没有租户概念，没有排队                          7
15  TTFT / TPOT 分位数、goodput、每百万 token 成本            否：需要引擎指标 + 网关计数 + 成本模型                              8
```

15 条里原生满足的是第 2（整卡时）、第 7 两条，"接口有、能力缺"的是第 1、6、8、9、13 条，其余 8 条是模型层面的缺失，集中在扩缩容（4–6）与路由（10–14）。

### 3. 哪几条是原生满足不了的

两张表合起来，原生 Kubernetes **模型层面**满足不了的需求可以归为五组，每组对应一篇：

```text
组                                  训练表          推理表          机制                                  篇
──────────────────────────────────  ─────────────  ─────────────  ───────────────────────────────────  ──
GPU 是不透明整数：无属性、无拓扑、不可分   1、2、3         1、3           device plugin 上报 + GFD 标签；DRA；切分   2、4
没有"一组 Pod"：无 gang、无组级重启      4、7、8         —              Kueue / Volcano；TrainJob；LWS        3、6
没有队列、配额借用、拓扑与抢占策略         5、6、9         —              ClusterQueue / cohort / TAS；PodGroup  3
一张网卡；存储只有接口没有规划            10、11、12      8              Multus + RDMA plugin；CSI 与存储选型     5
交付信号缺失：扩缩容、路由、租户、成本      14、15         4–6、10–15      KEDA；InferencePool + EPP；DCGM + OpenCost  6、7、8
```

值得注意的是训练与推理的空缺**几乎不重叠**：训练卡在调度与网络，推理卡在扩缩容与路由。只有"GPU 是不透明整数"是两者共同的空缺——这也是为什么第二篇是两条阅读路径共同的第一站。

反过来看，原生满足的部分也不少：Pod 的资源请求语法、headless Service 的稳定 DNS、Indexed Job 的序号、readinessProbe、PVC 抽象、HTTPRoute 的权重分流。平台组件几乎都建立在这些原生能力之上而非绕开它们——Kueue 用的是 Job 的 `suspend` 字段，LeaderWorkerSet 生成的是 StatefulSet，InferencePool 挂在 HTTPRoute 下面。**扩展而非替换**是这一层生态的共同选择，它的代价在第九章。


## 八、全系列术语表

后续七篇会反复使用下面这些词，各篇的总览会复述用到的部分：

```text
术语                    含义                                                                        首次展开
──────────────────────  ────────────────────────────────────────────────────────────────────────  ────────
gang（gang scheduling）  一组 Pod 要么全部调度要么全部不调度；Volcano PodGroup 的 minMember、Kueue 的 Workload 准入   第三篇
cohort                  Kueue 中一组 ClusterQueue 的集合，成员之间可以借用彼此的空闲配额（ClusterQueueSpec.CohortName） 第三篇
ResourceFlavor          Kueue 中资源的"口味"：一组节点标签与污点对应一种硬件（如 H100 与 A100 各一个 flavor）           第三篇
Workload                Kueue 为每个被管理的 Job 创建的准入对象，记录它要多少配额、被哪个 ClusterQueue 准入            第三篇
TAS                     Kueue 的 Topology-Aware Scheduling：Topology CRD（v1beta1）定义层级，Workload 按层级紧凑放置    第三篇
PodGroup / Queue        Volcano 的调度单位与队列；Queue 有 capability / deserved / guarantee 三个配额语义             第三篇
TrainJob                Kubeflow Trainer 的训练任务对象，引用 TrainingRuntime，底层生成 JobSet                     第三篇
MIG profile             一张 GPU 切成的硬件分区规格，如 1g.5gb、3g.20gb；device plugin 上报为 nvidia.com/mig-1g.5gb    第四篇
时间片 / MPS             device plugin 的 sharing 配置：一张卡上报为 N 个副本；时间片无隔离，MPS 合并 CUDA context      第四篇
HAMi                    软件层 GPU 切分：nvidia.com/gpumem、nvidia.com/gpucores 表达显存与算力比例                   第四篇
GPUDirect RDMA          网卡直接读写 GPU 显存，NCCL 跨节点通信不经过主机内存                                        第五篇
NetworkAttachmentDefinition  Multus 的 CRD，定义 Pod 可以额外挂的一张网卡                                       第五篇
LeaderWorkerSet（LWS）   一个 leader 加 N 个 worker 作为一个副本单元的 Kubernetes 对象；多节点推理的标准表达            第六篇
PD 分离                  prefill 与 decode 跑在不同的 Pod 组里，中间传 KV cache；两组独立扩缩                        第六篇
InferencePool           Gateway API Inference Extension 的 CRD：一组同模型同硬件的 Pod，替代 Service 作为路由后端     第七篇
EPP（Endpoint Picker）   Envoy ext_proc 外部处理器，按每个副本的队列、KV cache、LoRA 状态选目标 Pod；llm-d-router 实现   第七篇
InferenceObjective      表达请求优先级的扩展 API（llm-d-router apix/v1alpha2）                                      第七篇
TTFT / TPOT             首 token 延迟（排队 + prefill）/ 每输出 token 时间（decode 每步）；vllm:time_to_first_token_seconds、vllm:inter_token_latency_seconds  第三章
goodput                 单位时间内满足 SLO（TTFT 与 TPOT 都在阈值内）的请求或 token 数；与吞吐的差是"跑了但没用"的部分     第八篇
分配率 / 使用率           分配率：被 Pod 请求占用的 GPU 比例；使用率：DCGM_FI_PROF_SM_ACTIVE 这类真实负载；差距是平台改进空间  第八篇
DCGM_FI_DEV_GPU_UTIL     "有 kernel 在跑"的时间比例，不是算力利用率；与 SM_ACTIVE 的区别是第八篇的起点                  第八篇
```


## 九、代价与边界

### 1. 引擎需求 → K8s 空缺 → 平台机制 → 代价

全系列每篇都会有这样一张四栏表，本篇给的是全局版，后面各篇展开各自的行：

```text
引擎需求                    K8s 空缺                    平台机制                              代价
─────────────────────────  ─────────────────────────  ───────────────────────────────────  ─────────────────────────────────────
容器看到 GPU，版本匹配        GPU 不被识别                 GPU Operator + device plugin / DRA     驱动、Toolkit、插件、CUDA 四层版本契约；每层一个升级窗口
W 个 Pod 全员到齐            逐 Pod 调度                  Kueue 准入 / Volcano 批调度             排队时间换无死锁；两套调度器的对象模型要学一套
同机柜、同 NVLink 域          调度器不知拓扑               TAS / 拓扑插件 + 节点标签               等更久换更快；标签来源与准确性成为新依赖
团队配额、空闲借用            ResourceQuota 是硬上限        ClusterQueue / cohort / Queue          借来的资源可能被收回（抢占）；配额策略本身要运维
小模型共卡                  一卡只能给一个容器             MIG / 时间片 / MPS / HAMi              隔离与利用率的取舍；改配置要清空整卡；NCCL 不能跨 MIG
NCCL 走 RDMA                一张 veth 网卡               Multus + RDMA plugin + Network Operator  多一套网络配置与驱动栈；回退到 TCP 不报错、难发现
checkpoint 突发写            PVC 无带宽语义               PFS / 对象存储 + 缓存层                  按峰值带宽买存储；缓存一致性
多 Pod 一副本                Deployment 是单 Pod          LeaderWorkerSet / JobSet                滚动升级以组为单位，一次替换 M 个 Pod
按引擎指标扩缩               HPA 看 CPU                  KEDA + Prometheus                       指标管线的延迟叠加在冷启动上；阈值调参
不轮询、感知 KV cache         Service 轮询                 InferencePool + EPP                     多一跳 ext_proc；EPP 自身要高可用；抓指标有开销
按 token 限流与计费          没有租户与 token 概念          网关的配额与计量                          预扣与结算的复杂性；流式断开的计费争议
分配率与使用率的差距          没有 GPU 指标                DCGM + vllm:* + OpenCost                 高基数指标的存储成本；分摊规则的争议
```

### 2. 每叠一层的代价

上表的"代价"列有三种共同模式，值得单独说。

**版本契约的层数**。一个能用 GPU 的 Pod 依赖驱动、Container Toolkit、device plugin、CUDA Runtime 四层各自的版本，加上 Kubernetes 自身、Operator、调度器扩展、CRD 的 API 版本。每一层都有自己的发布节奏与兼容矩阵。总纲的版本基线表里有二十多个项目，这不是本系列的选择，而是这一层生态的现状：升级任何一个都要检查它与相邻层的契约。

**两套对象模型**。用了 Kueue 就要理解 `Workload` 与 `ClusterQueue`，用了 Volcano 就要理解 `PodGroup` 与 `Queue`，用了 LeaderWorkerSet 就不能直接用 Deployment 的滚动升级参数。每个 CRD 都是对原生对象的一次包装，排障时要穿过这层包装找到底下的 Pod。第五章说这一层生态选择了"扩展而非替换"，好处是原生工具（`kubectl describe`、事件、标签）都还能用，代价是**对象数量翻倍**——一个 `TrainJob` 背后是 JobSet、Job、Pod 三层，一个 `LLMInferenceService` 背后是 LeaderWorkerSet、StatefulSet、Pod、InferencePool、HTTPRoute。

**信号延迟的叠加**。推理扩容的链路是：引擎指标每 N 秒被 Prometheus 抓一次 → KEDA 每 M 秒查一次 → HPA 决定 → 调度 → 拉镜像 → 加载权重 → 就绪。每一段的延迟相加才是"从负载上升到新副本可用"的时间。平台每加一个环节，这个数字就多一项；第六篇的核心问题就是把它分解并压缩。

### 3. 什么时候不该上这一层

平台层不是免费的，几种情况下它的代价大于收益：

- **单团队、固定负载、几台机器**。三台 8 卡机跑一个团队的训练，`torchrun` 直接上 SSH 或用 Slurm 就够了，gang 与配额没有多租户就没有意义。
- **只有推理、只有小模型、副本数固定**。一个 Deployment 加 Service 加 HPA（就算看 CPU 也无所谓，因为不扩）可能足够；InferencePool 与 EPP 的收益在副本多、请求长短悬殊、prefix 重复率高时才明显。
- **云上托管服务能接受黑盒**。各云的 GPU 节点池替你做了第二篇（驱动与插件），托管推理服务替你做了第六、七篇的大部分；代价是排障时看不到内部、换厂商时经验不迁移。本系列讨论的是自建路线，托管服务在相关处会标注"替你做了哪一步"。

### 4. 本系列的边界

本系列只讨论**引擎之下的资源层**与**引擎之侧的交付层**，以下内容作为"需求来源"一句带过，不展开：

- **引擎内部**：训练框架的并行策略与状态切分、checkpoint 的格式与异步写法；推理引擎的请求调度、KV cache 管理、PD 分离在引擎内的实现。本文第二、三章用到的只是它们对外的形状——进程组、环境变量、显存构成、指标名。
- **通信与网络协议内部**：NCCL 的算法与 channel、RDMA verbs、InfiniBand 与 RoCE 的差别。第五篇只讨论如何让 RDMA 设备在容器里可用、如何验证 NCCL 走了正确的路径。
- **GPU 硬件与 kernel 性能**：第四篇讨论 MIG 时用到"SM 与显存带宽被分区"这个事实，不展开它对 kernel 的影响。
- **Kubernetes 一般原理**：Pod、Deployment、CRD、Operator、CNI、CSI 假定已知。
- **数据平台、实验管理、应用层**：数据怎么来、模型怎么评、RAG 与 Agent 怎么写，都在本系列之外。

边界的另一侧是硬件：本系列以 **Kubernetes 上的 NVIDIA GPU** 为主线；AMD 有对应的 device plugin 与 Operator，机制高度对应，在第二、四篇提及差异；其他加速器不在范围内。


## 十、本文小结

### 1. 要点回顾

- 训练任务是 $$W$$ 个强耦合的进程：`torchrun` 的 agent 通过 rendezvous 互相发现，注入 `MASTER_ADDR` / `MASTER_PORT` / `WORLD_SIZE` / `RANK` / `LOCAL_RANK`。每一步以集合通信结束，任一进程失败其余全部卡住，所以 `SimpleElasticAgent._invoke_run()` 的策略是任一 worker 失败则整组重启；agent 或节点消失时"整个任务失败还是补节点"由平台决定。
- 训练的四条资源需求：整数个 GPU 且每进程一张；节点间高带宽低延迟且拓扑相近；周期性突发顺序写（checkpoint，1 TB 级在 1 分钟内）；持续小块随机读（数据集）。它是有限时长的批处理，排队是常态，被打断的代价是上次 checkpoint 之后的进度。
- 推理副本有三种形态（单卡、单 Pod 多卡、多 Pod 一副本），prefill 计算密集、decode 显存带宽密集。显存是硬约束——权重加 KV cache 不能超卖；扩容以分钟计，主项是权重加载；负载信号在引擎内部：`vllm:num_requests_waiting`、`vllm:kv_cache_usage_perc`、`vllm:time_to_first_token_seconds`、`vllm:inter_token_latency_seconds`。
- 两类负载的需求在独占 vs 共享、拓扑 vs 弹性、批处理 vs 长驻三处直接冲突；共同点是都要"多 Pod 一个单位"、都要 GPU 有属性、成本都落到 GPU 时间。
- 原生 Kubernetes 的空缺分三种：缺一个插件（GPU 与 RDMA 设备不被识别）、缺一个概念（一组 Pod、设备属性、队列与配额、多 Pod 副本）、缺一个信号（引擎指标驱动的扩缩容、路由、成本）。kube-scheduler 逐 Pod 调度，`noderesources/fit.go` 的 `Fits()` 只把 `nvidia.com/gpu` 当整数比较；device plugin 的 `ListAndWatch` / `Allocate` 只能上报计数。
- 平台分两层：资源层（设备、调度、切分、网络存储）把裸节点变成能跑的 Pod；交付层（Serving、网关、可观测）把 Pod 变成有 SLA 与账单的服务。分界线是"Pod 能跑了"；两层通过资源请求、Pod 组抽象、指标三样东西耦合。训练只走资源层，推理两层都走。
- Slurm 与 Ray 是资源层调度部分的同位替代：Slurm 原生满足训练的全部需求但缺推理侧；Ray 自带调度器与 Serve，放到 Kubernetes 上是两层调度。
- 核心问题的两张表：32 卡训练任务 15 条需求中原生完全满足 1 条，模型层面缺失集中在调度（gang、拓扑、配额、组级重启、抢占策略）与网络（RDMA）；TP=2 推理服务 15 条中原生满足 2 条，缺失集中在扩缩容（指标、提前、缩零）与路由（副本状态、model 字段、租户配额）。两者唯一共同的空缺是"GPU 是不透明整数"。
- 平台每叠一层的代价：版本契约的层数、对象模型翻倍、信号延迟叠加。单团队固定负载、只有小模型固定副本、或能接受托管黑盒时，不该上这一层。

### 2. 本篇涉及的源码与 CRD 位置

| 路径 | 内容 |
|---|---|
| PyTorch 2.13.0 `torch/distributed/run.py` | 模块文档：`--nnodes` / `--nproc-per-node` / `--rdzv-backend` / `--rdzv-endpoint` / `--rdzv-id` / `--max-restarts`；环境变量 `MASTER_ADDR`、`MASTER_PORT`、`WORLD_SIZE`、`RANK`、`LOCAL_RANK`、`LOCAL_WORLD_SIZE`、`TORCHELASTIC_RUN_ID`；"Failure Modes" 与 "Membership Changes" 两节 |
| PyTorch 2.13.0 `torch/distributed/elastic/agent/server/api.py` | `SimpleElasticAgent._invoke_run()`（监控循环：`SUCCEEDED` → `_exit_barrier()`；`FAILED` / `UNHEALTHY` → `_restart_workers()` 或停止）、`_rendezvous()`、`_monitor_workers()`；`local_elastic_agent.py` 的 `LocalElasticAgent` 是本机实现 |
| PyTorch 2.13.0 `torch/distributed/elastic/rendezvous/dynamic_rendezvous.py` | `RendezvousSettings.min_nodes` / `max_nodes`、`_DistributedRendezvousOpExecutor`；`rendezvous/api.py` 的 `RendezvousHandler.next_rendezvous()`；`c10d_rendezvous_backend.py` 为默认后端 |
| vLLM v0.23.0 `vllm/v1/metrics/loggers.py` | `PrometheusStatLogger`：`vllm:num_requests_running`、`vllm:num_requests_waiting`、`vllm:kv_cache_usage_perc`、`vllm:prefix_cache_hits` / `vllm:prefix_cache_queries`、`vllm:num_preemptions`、`vllm:time_to_first_token_seconds`、`vllm:inter_token_latency_seconds`、`vllm:request_queue_time_seconds` |
| vLLM v0.23.0 `vllm/config/cache.py`、`vllm/config/parallel.py` | `CacheConfig.gpu_memory_utilization`（默认 0.92）；`ParallelConfig.tensor_parallel_size` |
| Kubernetes v1.37.0 `pkg/scheduler/framework/plugins/noderesources/fit.go` | `Fits()`、`fitsRequest()`、`InsufficientResource`（`Reason` 为 `Insufficient <资源名>`） |
| Kubernetes v1.37.0 `staging/src/k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1/api.proto` | `ListAndWatch`、`GetPreferredAllocation`、`Allocate` 三个 rpc；`pkg/kubelet/cm/devicemanager/manager.go` 的 `ManagerImpl.Allocate()` / `GetCapacity()` |
| Kubernetes v1.37.0 `staging/src/k8s.io/api/resource/v1/types.go` | DRA 的 `ResourceSlice`、`ResourceClaim`、`ResourceClaimTemplate`、`DeviceClass`（第二篇展开） |
| GPU Operator v26.7.0 `api/nvidia/v1/clusterpolicy_types.go` | `ClusterPolicySpec` |
| Kueue v0.19.2 `apis/kueue/v1beta2/clusterqueue_types.go`、`resourceflavor_types.go`、`apis/kueue/v1beta1/topology_types.go` | `ClusterQueueSpec.CohortName`、`ReclaimWithinCohort` / `BorrowWithinCohort`；`ResourceFlavorSpec.NodeLabels`；`Topology` CRD |
| Volcano v1.15.2 `staging/src/volcano.sh/apis/pkg/apis/scheduling/v1beta1/types.go`、`pkg/scheduler/plugins/gang/gang.go` | `PodGroupSpec.MinMember`；`QueueSpec.Capability` / `Deserved` / `Guarantee`；gang 插件 |
| Kubeflow Trainer v2.3.0 `pkg/apis/trainer/v1alpha1/trainjob_types.go`、`trainingruntime_types.go`、`pkg/constants/constants.go` | `TrainJobSpec.RuntimeRef`；`TorchMLPolicySource.NumProcPerNode`、`NumNodes`；`TorchEnvNumNodes = "PET_NNODES"` 等 torchrun 环境变量常量 |
| KubeRay v1.7.0 `ray-operator/apis/ray/v1/` | `RayClusterSpec`、`RayJobSpec`、`RayServiceSpec` |
| Slinky slurm-operator v1.2.2 `api/v1beta1/` | `ControllerSpec`、`NodeSetSpec`、`LoginSetSpec` |
| LeaderWorkerSet v0.10.0 `api/leaderworkerset/v1/leaderworkerset_types.go` | `LeaderWorkerSetSpec.LeaderWorkerTemplate`；`api/disaggregatedset/v1/` 为 PD 分离的独立 API group |
| KServe v0.20.0 `pkg/apis/serving/v1alpha1/llm_inference_service_types.go` | `LLMInferenceService`（`serving.kserve.io/v1alpha1`） |
| Gateway API Inference Extension v1.6.0 `api/v1/inferencepool_types.go` | `InferencePool`（`inference.networking.k8s.io/v1`）、`EndpointPickerRef`；EPP 实现在 llm-d-router v0.10.0 `cmd/epp`、`pkg/epp`；`InferenceObjective` 在 llm-d-router `apix/v1alpha2` |
| KEDA v2.20.2 `apis/keda/v1alpha1/scaledobject_types.go`、`pkg/scalers/prometheus_scaler.go` | `ScaledObjectSpec.Triggers`；Prometheus scaler |
| DCGM Exporter 4.6.0-4.8.3 `etc/default-counters.csv` | `DCGM_FI_DEV_GPU_UTIL`、`DCGM_FI_DEV_FB_USED` 默认开启；`DCGM_FI_PROF_SM_ACTIVE` 默认注释 |
| OpenCost v1.121.1 `pkg/costmodel/allocation.go` | allocation 合并逻辑中的 GPU 字段（第八篇展开） |

### 3. mini-platform 本篇增量

练手项目 `mini-platform/` 是一个从零搭起的最小 AI 平台，八篇各加一批文件。本篇只做两件事：把集群搭起来，提交一个注定 Pending 的 GPU Pod 并读懂它的事件。**不装任何 GPU 相关组件**——第二篇装上 GPU Operator 之后，同一份 manifest 会变成 Running，前后对照是理解 device plugin 的最短路径。

`cluster/nodes.md`——节点清单与约束：

```text
mini-platform / cluster/nodes.md -- node inventory for the whole series

角色            数量   建议规格                            说明
──────────────  ────  ─────────────────────────────────  ──────────────────────────────────────────────
control-plane    1    4 vCPU / 8 GB，无 GPU                 kubeadm init；带 node-role.kubernetes.io/control-plane 污点
gpu-worker       3    每台 1 张 NVIDIA GPU（云上单卡按需实例即可）  kubeadm join；本篇不装驱动之外的任何 GPU 组件

硬件对后续篇目的影响：
- 第二、三、六、七、八篇：任意 NVIDIA GPU 即可（T4 / L4 / A10 都行）
- 第四篇 MIG 部分：需要 A100 或 H100；没有则只做 HAMi 与时间片
- 第五篇 RDMA 部分：需要 IB 或 RoCE 网卡的实例；没有则用 host network 做配置走读

Kubernetes 版本：v1.37.0（DRA 的 resource.k8s.io/v1 已 GA；第二篇会用到）
容器运行时：containerd（第二篇 Container Toolkit 与 CDI 的配置以它为准）
CNI：任意（Calico / Cilium / Flannel）；第五篇加 Multus 时保留它作为默认网络

kubeadm 的最小命令序列（控制面节点）：
  kubeadm init --kubernetes-version v1.37.0 --pod-network-cidr <CNI 要求的网段>
  # 安装 CNI
  kubeadm token create --print-join-command        # 在三台 worker 上执行输出的 join 命令

本篇结束时的预期状态：
  kubectl get nodes                                 # 4 个 Ready
  kubectl get node <gpu-worker> -o jsonpath='{.status.allocatable}'
  # 输出里只有 cpu / memory / ephemeral-storage / pods / hugepages-*，没有 nvidia.com/gpu
```

`probes/pending-gpu-pod.yaml`——一个请求 GPU 的最小 Pod：

```yaml
# mini-platform / probes/pending-gpu-pod.yaml
# 在裸集群（未装 device plugin）上提交，预期 Pending；第二篇装完 GPU Operator 后同一份 manifest 预期 Running。
apiVersion: v1
kind: Pod
metadata:
  name: gpu-probe
  namespace: default
  labels:
    app: gpu-probe
spec:
  restartPolicy: Never
  containers:
    - name: probe
      image: nvidia/cuda:12.8.1-base-ubuntu22.04
      command: ["nvidia-smi", "-L"]
      resources:
        limits:
          nvidia.com/gpu: 1
```

两个细节值得注意。第一，`nvidia.com/gpu` 只写在 `limits` 里：扩展资源要求 `requests` 与 `limits` 相等，只写 `limits` 时 Kubernetes 会把 `requests` 设为同值。第二，命令是 `nvidia-smi -L`，第二篇里它会列出容器内可见的那张卡；本篇它不会有机会运行。

提交并观察：

```bash
kubectl apply -f probes/pending-gpu-pod.yaml
kubectl get pod gpu-probe            # STATUS 一直是 Pending
kubectl describe pod gpu-probe
```

`describe` 输出末尾的 Events 形态如下（**示意**，节点数与措辞随集群与版本略有差异）：

```text
Events:
  Type     Reason            Age   From               Message
  ----     ------            ----  ----               -------
  Warning  FailedScheduling  12s   default-scheduler  0/4 nodes are available: 1 node(s) had untolerated taint
                                                      {node-role.kubernetes.io/control-plane: }, 3 Insufficient nvidia.com/gpu.
                                                      preemption: 0/4 nodes are available: 1 Preemption is not helpful for
                                                      scheduling, 3 No preemption victims found for incoming pod.
```

逐段解读：

- `0/4 nodes are available`：Filter 阶段没有一个节点通过。`FailedScheduling` 事件由 kube-scheduler 发出，Pod 会留在调度队列里周期性重试，所以 `Age` 会不断刷新而事件只有一条。
- `1 node(s) had untolerated taint {node-role.kubernetes.io/control-plane: }`：控制面节点因污点被排除，与 GPU 无关。
- `3 Insufficient nvidia.com/gpu`：三台 worker 都被 `noderesources/fit.go` 的 `fitsRequest()` 判为该扩展资源不足。注意这里的措辞是"不足"而不是"不存在"——对调度器来说，节点 `allocatable` 里没有这个键与这个键等于 0 是同一回事。三台机器上各有一张物理 GPU，但没有任何组件把它上报给 kubelet。
- `preemption: ... No preemption victims found`：调度器尝试了抢占，但抢占只能腾出**已被其他 Pod 占用**的资源；这个资源在节点上根本没有登记，杀掉谁都没用。

验证根因：

```bash
kubectl get nodes -o custom-columns='NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu'
# GPU 列全部为 <none>
```

这就是第五章第 2 节说的"默认情况下 GPU 在 Kubernetes 里是一个不透明的整数"的零值形态。下一篇装上 GPU Operator，`allocatable` 里出现 `nvidia.com/gpu: 1`，同一个 Pod 被调度、`nvidia-smi -L` 打印出一张卡——然后我们再问：它是怎么知道容器里该有哪些设备文件和哪几个 `.so` 的。

清理：

```bash
kubectl delete -f probes/pending-gpu-pod.yaml
```


## 下一篇

[容器里的 GPU：驱动、CUDA、device plugin 与镜像](/gpu-in-containers-driver-cuda-device-plugin.html)。本篇的 `Insufficient nvidia.com/gpu` 是从调度器视角看到的空缺；下一篇从节点视角把它填上：内核驱动、用户态库、CUDA Runtime 与容器运行时四层的版本契约，Container Toolkit 与 CDI 如何把设备注入容器，device plugin 的 `ListAndWatch` / `Allocate` 如何把 GPU 变成 `allocatable` 里的一个整数，GPU Operator 的 `ClusterPolicy` 如何把这一切装到每个节点，以及 DRA 用 `ResourceClaim` 与 CEL 表达式如何让这个整数重新有属性。核心问题：宿主机驱动 535、镜像里 CUDA 12.4 的 PyTorch、代码调用了 CUDA 12.4 新增的 API，这个组合能跑吗？
