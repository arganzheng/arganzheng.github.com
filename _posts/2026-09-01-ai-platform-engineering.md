---
layout: post
title: AI 平台工程：资源层与交付层（总纲）
subtitle: "AI Platform Engineering: the Resource Layer and the Delivery Layer"
tags: [Kubernetes, MLOps, GPU, AI, AI-Infra]
catalog: true
---


## 内容简介

《AI 平台工程：资源层与交付层》是一组共八篇的系列文章，面向已经跑过训练任务或推理服务、准备把它们放到一个多人共用的 GPU 集群上运行和交付的工程师，系统讲解引擎之下和引擎之侧的两层基础设施：**资源层**把 GPU、网络、存储组织成可调度的资源池，承载引擎运行；**交付层**把引擎包装成有 SLA、有配额、有账单、可观测的服务。

它回答的问题是：

> **一个 GPU 集群如何被切分、调度和喂饱？一个训好的模型如何变成一个可运维的服务？**

站在引擎的层面看，平台是一个"给我八张卡、一个 RDMA 网口和一个能读 checkpoint 的路径，剩下的我自己来"的黑盒。这个视角是对的，但它掩盖了一件事：平台的每一个设计决定，都是被引擎的某个需求推出来的。

- 训练框架要求 `WORLD_SIZE` 个进程**同时**起来，缺一个就全部等着——所以有 gang scheduling；
- NCCL 要求进程能直接打开 RDMA 设备、注册 GPU 显存——所以有 device plugin、SR-IOV、GPUDirect RDMA 的一整套配置；
- 推理引擎的 KV cache 占满显存后请求就排队——所以扩缩容不能看 CPU 利用率，要看 `vllm:num_requests_waiting` 这类引擎指标；
- 一个 7B 模型的推理服务只用得上一张卡的三分之一——所以有 MIG、时间片和 HAMi 这类切分方案；
- 训练每半小时写一次几百 GB 的 checkpoint——所以存储要按突发写带宽而不是平均带宽选型。

这个系列采用的组织方式就是这条因果链：**每一篇先说引擎提出了什么要求，再说平台用什么机制满足，再说这个机制的代价和边界**。

```text
资源层    容器里的 GPU    → 驱动 / CUDA 兼容矩阵、device plugin、DRA、镜像
          任务调度        → gang scheduling、队列与配额、拓扑感知；Volcano / Kueue / Slurm / Ray
          GPU 切分        → MIG、时间片、MPS、HAMi：隔离与利用率的取舍
          网络与存储      → RDMA 在 K8s 里的配置、并行文件系统与对象存储、checkpoint I/O
交付层    Serving 平台    → KServe / Triton / Ray Serve / llm-d，LLM 服务的形态与扩缩容
          模型网关        → 路由、多租户、配额与限流
          可观测与成本    → 从 DCGM 到引擎指标，从利用率到账单
```

系列以 **Kubernetes 上的 NVIDIA GPU 集群**为主线，因为这是当前开源生态最活跃、组件最齐全的组合。Slurm 和 Ray 作为同位替代品在调度篇中对比，不单独展开。


## 为什么写这个系列？

### 引擎跑得快不等于集群用得好

一个训练任务的 MFU 到了 45%，一个推理服务的吞吐到了硬件极限的 80%——这是引擎层的成绩。但从集群层面看，还有另一组数字：这个集群有多少张卡处于分配状态、多少张真正在算、多少任务在排队等资源、排队的原因是配额不够还是碎片太多、每张卡每小时花了多少钱、这些钱分摊到哪个团队。

这两组数字经常互相矛盾。一个把单卡吞吐做到极限的推理服务，可能独占了一张 80 GB 的卡却只用了 20 GB 显存；一个 MFU 很高的训练任务，可能因为等 gang scheduling 凑齐 64 张卡而空转了两个小时。**引擎的目标是把给定的资源用满，平台的目标是让给出去的资源尽可能少而够用**。后者是本系列的主题。

### K8s 不是为 AI 负载设计的

Kubernetes 的核心假设是：工作负载是长驻的、无状态的、单 Pod 独立的、资源需求是 CPU 和内存这样可压缩或可细分的。AI 负载几乎逐条违背这些假设：

- 训练任务是**有限时长**的批处理，而且是**多 Pod 强耦合**的——一个 Pod 起不来其他 Pod 都白等；
- GPU 是**整数个、不可压缩、不能超卖**的资源，默认调度器对它只会计数，不知道 A100 和 H100 的区别，也不知道两张卡之间是 NVLink 还是 PCIe；
- 分布式训练对**网络拓扑**极度敏感，同一机柜和跨机柜的 all_reduce 带宽差几倍，默认调度器完全不考虑；
- 推理服务需要**多 Pod 作为一个副本**（张量并行跨节点），Deployment 和 StatefulSet 都没有这个概念。

所以 K8s 上的 AI 平台，实际上是在原生对象之上叠了一层又一层的扩展：device plugin 和 DRA 解决"GPU 是什么"，Volcano 和 Kueue 解决"多 Pod 一起调度"，LeaderWorkerSet 解决"多 Pod 一个副本"，Multus 和 RDMA device plugin 解决"第二张网卡"。每一层扩展都在填一个原生假设的洞。理解这些洞在哪里，比记住每个组件的安装命令重要得多。

### 交付层的问题在引擎之外

一个 vLLM 进程能服务一个模型。一个平台要服务几十个模型、几百个租户、多种硬件、不同的 SLA：有的模型要 PD 分离，有的一张卡就够；有的租户要保证延迟，有的只要便宜；同一个模型有 FP8 和 BF16 两个版本要做 A/B；一个新版本上线要灰度和回滚；每个请求要计 token 数记账。这些问题都不在引擎内部，引擎也不应该关心它们。它们构成了交付层：Serving 平台管副本的生命周期，网关管请求的路由和配额，可观测管指标、日志和账单。

这一层和传统微服务的交付有相似的骨架，但每个环节都被 LLM 的特性扭曲了：请求是流式的、时长差几十倍、成本按 token 计而不是按次计、扩容一个副本要几分钟加载权重而不是几秒。直接套用微服务的经验会踩很多坑，本系列的交付层三篇就是为这些坑而写。

### 现有材料的断层

- **各组件的官方文档**（GPU Operator、Kueue、KServe、Triton）分别讲得都很清楚，但每一份都假设读者已经知道自己为什么需要它，也不讲组件之间怎么配合；
- **云厂商的托管方案**把这一层封装掉了，用起来方便，但出问题时看不到内部，换厂商时经验不能迁移；
- **MLOps 类材料**大多面向传统机器学习，讲实验跟踪和特征存储，对 GPU 调度和 LLM 服务几乎没有覆盖；
- **HPC 社区的 Slurm 经验**在训练调度上很成熟，但很少和 K8s 生态对话。

本系列想填补的是从"能在一台机器上跑起来引擎"到"能设计和运维一个多租户 GPU 平台"之间的那段路。


## 适合哪些读者？

### 跑过训练或推理、准备接手集群的工程师

你已经在一台或几台机器上跑过分布式训练，或者部署过 vLLM 这类推理服务，知道 `torchrun`、`NCCL_DEBUG`、`--tensor-parallel-size` 是什么。现在这些东西要上到一个几十到几百张卡、多个团队共用的集群上，你要回答"任务为什么 Pending"、"为什么这个任务被调度到了跨机柜的节点上"、"怎么让两个小模型共用一张卡"这类问题。

### 从后端或云原生方向转入 AI 平台的工程师

你熟悉 K8s 的基本对象、Operator 模式、Prometheus 和 Gateway API，做过微服务的交付和运维。你需要的是：GPU 作为资源和 CPU 有什么本质区别，AI 负载在调度、网络、存储上的特殊要求是什么，LLM 服务的扩缩容、路由和计费为什么不能照搬微服务的做法。本系列会在每一处对比"通用云原生的做法"和"AI 负载下为什么不够"。

### 做 MLOps / LLMOps 的工程师

你负责把模型变成服务：选 Serving 框架、配扩缩容、接网关、做灰度、看指标、算成本。本系列的交付层三篇是直接工具；资源层四篇让你知道底下的 GPU 是怎么被分出来的，这决定了服务的成本结构和扩容速度。

### 引擎开发者，想知道平台对引擎提出了什么要求

反向的视角同样有用：一个训练框架怎样才算"平台友好"（可弹性、可抢占、checkpoint 可恢复），一个推理引擎需要暴露哪些指标才能被正确扩缩容、被网关正确路由。本系列每篇的"引擎的需求"一节，也是引擎开发者的接口清单。


## 系列的整体主线

八篇文章分两部分。前四篇是资源层，按"一个训练任务从提交到跑起来要经过的层"自下而上推进；后三篇是交付层，按"一个推理请求从进入到计费要经过的层"自外向内推进。第一篇是两者共同的起点：

```text
第一篇：引擎的需求清单与平台的整体架构 —— 建立组织视角
        ↓
第二篇：容器里的 GPU —— 驱动、CUDA、device plugin、DRA、镜像
        ↓
第三篇：AI 任务调度 —— gang scheduling、队列与配额、拓扑感知；Volcano / Kueue / Slurm / Ray
        ↓
第四篇：GPU 共享与切分 —— MIG、时间片、MPS、HAMi
        ↓
第五篇：网络与存储 —— RDMA 进容器、并行文件系统与对象存储、checkpoint I/O
        ↓
第六篇：Serving 平台 —— KServe / Triton / Ray Serve / llm-d，LLM 服务的形态与扩缩容
        ↓
第七篇：模型网关与多租户 —— 路由、配额、限流、灰度
        ↓
第八篇：可观测、成本与 FinOps —— 从 DCGM 到 token 账单
```

三条交织的线索：

```text
引擎线：训练框架的进程组与 checkpoint → 推理引擎的显存与请求队列 → 两者对平台接口的要求
机制线：device plugin → 调度器扩展 → 切分与隔离 → 第二张网卡 → CRD 与 Operator → 网关扩展 → 指标管线
取舍线：隔离 vs 利用率 → 排队 vs 碎片 → 拓扑 vs 等待时间 → 冷启动 vs 常驻成本 → 精确计费 vs 开销
```

每一篇都有同样的结构：

```text
引擎的需求      训练框架 / 推理引擎在这一层提出了什么要求，不满足会发生什么
K8s 的空缺      原生 Kubernetes 为什么满足不了
平台的机制      填这个空缺的组件是什么、怎么工作、怎么配
代价与边界      这个机制引入了什么新问题，什么场景下不该用
```


## 章节结构与分章导读

### 1. 引擎的需求清单与平台的整体架构

第一篇建立整个系列的组织视角。它不装任何组件，只回答一个问题：**训练框架和推理引擎到底对平台提出了哪些要求？**

这一篇会讨论：

- 训练任务的形态：`torchrun` 启动的进程组、rendezvous、`MASTER_ADDR` / `WORLD_SIZE` / `RANK` 的语义，为什么任何一个进程挂掉整个任务就停；
- 训练任务的资源需求：整数个 GPU、节点间高带宽低延迟网络、周期性的大块顺序写（checkpoint）、持续的小块随机读（数据集）；
- 推理服务的形态：一个模型副本可能是一个进程、一个 Pod、或跨节点的一组 Pod；prefill 与 decode 两个阶段对资源的不同要求；
- 推理服务的资源需求：显存是硬约束（权重 + KV cache），扩容一个副本的时间由权重加载决定，负载指标在引擎内部而不在 CPU 上；
- 两类负载对平台的矛盾要求：训练要独占和拓扑，推理要共享和弹性；训练是有限时长的批处理，推理是长驻服务；
- 平台的两层拆分：资源层（容器运行时、调度、网络、存储）和交付层（Serving、网关、可观测），每层的输入输出是什么；
- 组件全景图：NVIDIA GPU Operator、device plugin / DRA、Volcano / Kueue、Kubeflow Trainer / KubeRay、LeaderWorkerSet、KServe / llm-d、Gateway API Inference Extension、DCGM Exporter，各自落在哪一层、解决哪条需求；
- 同位替代：Slurm 集群和 Ray 集群在这张图上的位置，为什么 HPC 出身的团队和云原生出身的团队会选不同的路。

核心问题是：

> **一个 4 节点 32 卡的训练任务和一个 TP=2、副本数动态变化的推理服务，各自对平台提出的需求列成一张表，哪几条是原生 Kubernetes 满足不了的？**

这一篇也会划清本系列的边界：引擎内部的机制（并行策略、KV cache 管理、调度算法）被当作已知的需求来源，不展开；NCCL 和 RDMA 的协议内部被当作黑盒，只讨论如何让它们在容器里工作。

实践：搭起系列贯穿使用的最小集群（三到四个节点，每节点至少一张 GPU，可以是云上按需实例），装好 K8s，先不装任何 GPU 相关组件，提交一个请求 GPU 的 Pod 观察它为什么 Pending。

### 2. 容器里的 GPU：驱动、CUDA、device plugin 与镜像

第二篇解决最基础的问题：一个容器怎样才能用上 GPU。这个问题看起来只是"装个插件"，实际上涉及内核驱动、用户态库、容器运行时和 K8s 四层之间的版本契约。

这一篇会覆盖：

- GPU 软件栈的分层：内核态驱动（`nvidia.ko`）、用户态驱动库（`libcuda.so`）、CUDA Runtime（`libcudart.so`）、cuDNN / NCCL 等库；哪些在宿主机、哪些在容器里、边界为什么划在那里；
- CUDA 兼容规则：向后兼容（新驱动跑旧 Toolkit，无条件成立）、minor version compatibility（CUDA 11 起，同一大版本内旧驱动跑新 Toolkit，要求 SASS 而非 PTX）、forward compatibility（跨大版本，需要 `cuda-compat` 包，仅数据中心 GPU）；CUDA 12.x 要求驱动 >= 525、CUDA 13.x 要求 >= 580 这类基线怎么查；
- 为什么 PyTorch 的 wheel 自带 CUDA Runtime 但不带驱动，`torch.version.cuda` 与 `nvidia-smi` 显示的 CUDA 版本为什么可以不同；
- NVIDIA Container Toolkit：`nvidia-container-runtime` 作为 OCI runtime hook 把驱动库和设备文件注入容器；`NVIDIA_VISIBLE_DEVICES` 与 `NVIDIA_DRIVER_CAPABILITIES`；CDI（Container Device Interface）作为新一代的注入方式；
- Kubernetes device plugin 机制：`ListAndWatch` 与 `Allocate` 两个 gRPC 接口，节点上报 `nvidia.com/gpu` 扩展资源，Pod 用 `resources.limits` 请求；这套机制的局限——只能计数、不能表达属性、不能跨 Pod 共享；
- NVIDIA GPU Operator：用 `ClusterPolicy` CRD 统一管理驱动容器、Container Toolkit、device plugin、GPU Feature Discovery、DCGM Exporter、MIG Manager，节点标签（`nvidia.com/gpu.product` 等）从哪里来；
- Dynamic Resource Allocation（DRA）：Kubernetes 1.34 GA 的 `resource.k8s.io/v1` API——`DeviceClass`、`ResourceSlice`、`ResourceClaim`、`ResourceClaimTemplate`，用 CEL 表达式按属性选设备；它相对 device plugin 解决了什么、NVIDIA 的 DRA driver 处于什么阶段；
- 镜像：官方 `nvidia/cuda` 基础镜像的 `base` / `runtime` / `devel` 三种变体；PyTorch、vLLM 官方镜像的层结构；多阶段构建把 10 GB 的镜像压到 3 GB；镜像拉取时间对训练启动和推理扩容的影响，以及镜像预热和 P2P 分发。

核心问题是：

> **宿主机驱动 535、镜像里 CUDA 12.4 的 PyTorch、代码里调用了 CUDA 12.4 新增的 API——这个组合能跑吗？如果宿主机驱动是 470 呢？答案取决于三条兼容规则中的哪一条适用。**

实践：在集群上装 GPU Operator，验证 Pod 能看到 GPU；故意构造一个驱动与 CUDA 版本不匹配的镜像，读懂报错；用 DRA 的 `ResourceClaim` 按显存大小选一张卡。

### 3. AI 任务调度：gang scheduling、队列与拓扑感知

第三篇进入调度。默认的 kube-scheduler 一个 Pod 一个 Pod 地调度，对训练任务来说这是致命的：32 个 Pod 调度了 30 个，剩下 2 个等不到资源，前 30 个占着 GPU 空转，其他任务也进不来——死锁。

这一篇会覆盖：

- gang scheduling 的定义：一组 Pod 要么全部调度要么全部不调度；`minMember` 语义；为什么它必须在调度器层面实现而不能靠应用层重试；
- Volcano：`PodGroup` 与 `Queue` CRD，`vcjob` 的任务模板，调度器的 action（enqueue / allocate / preempt / reclaim / backfill）与 plugin（gang / priority / drf / binpack / capacity）流水线；Queue 的 `capability` / `deserved` / `guarantee` 三个配额语义；
- Kueue：不替换调度器而是做准入控制——`Workload` 被 `LocalQueue` 提交到 `ClusterQueue`，配额按 `ResourceFlavor` 划分，`cohort` 之间借用配额；Job 被 `suspend` 直到配额准入；`AdmissionCheck` 与集群自动扩缩容的配合；
- 两者的设计哲学差别：Volcano 是"一个更懂批处理的调度器"，Kueue 是"调度器之前的配额闸门"；什么场景选哪个，能不能一起用；
- 拓扑感知：为什么同一台机器内的 8 张卡要一起给，为什么同一机柜的节点要一起给；Kueue 的 Topology-Aware Scheduling（`Topology` CRD、`podset-required-topology` / `podset-preferred-topology` 注解）；Volcano 的网络拓扑感知调度；节点标签从哪里来（NVIDIA 的拓扑标签、云厂商的 placement group）；
- 抢占与优先级：训练任务被抢占的代价（丢失自上次 checkpoint 以来的进度）如何影响抢占策略；`PriorityClass` 与队列优先级的交互；
- 训练任务的 K8s 表达：Kubeflow Trainer 的 `TrainJob` / `TrainingRuntime`（基于 `JobSet`），它如何生成 `PodGroup` 与调度器对接；`torchrun` 的 rendezvous 在 K8s 里怎么落地（headless Service 与稳定的 DNS 名）；
- Slurm 的对照：`sbatch` / `srun`、partition、GRES、`--exclusive`、backfill 调度——这套东西在 HPC 里已经做了二十年，K8s 生态在重新发明哪些部分；Slinky 项目（`slurm-operator`、`slurm-bridge`）把 Slurm 搬进 K8s 的两种方式；
- Ray 的对照：Ray 自己的调度器（placement group、资源标签）与 K8s 调度器的分工；KubeRay 的 `RayCluster` / `RayJob` / `RayService`，Ray autoscaler 与 K8s 节点自动扩缩容的两层关系。

核心问题是：

> **两个团队各有 16 卡的配额，A 团队提交了一个 32 卡的任务，B 团队的卡空着。在 Volcano、Kueue 和 Slurm 里，这个任务分别会怎样？借用、抢占、等待三种行为各自的配置是什么？**

实践：装 Kueue，建两个 `ClusterQueue` 组成 cohort，用 Kubeflow Trainer 提交一个 2 节点 PyTorch DDP 任务，观察它从 suspended 到 admitted 的全过程；再用 Volcano 跑同一个任务，对比两者的对象模型。

### 4. GPU 共享与切分：MIG、时间片、MPS 与 HAMi

第四篇处理利用率问题。一张 80 GB 的 H100 跑一个 7B 模型的推理服务，显存用了 20 GB、算力峰值不到 30%——剩下的部分怎么给别人用？这个问题在推理和开发环境里非常普遍，而 device plugin 的默认答案是"不能"。

这一篇会覆盖：

- 共享的四个层次：不隔离（多进程直接共用）、时间片（驱动层轮转）、MPS（进程合并到一个 CUDA context，可限显存和算力比例）、MIG（硬件分区，独立的 SM、L2 和显存带宽）；各自的隔离强度、故障域和性能开销；
- MIG 的机制：Ampere 及之后的数据中心 GPU 支持；GPU Instance 与 Compute Instance；A100 40 GB 的 profile（`1g.5gb` × 7、`2g.10gb` × 3、`3g.20gb` × 2、`7g.40gb` × 1）与 H100 的对应；分区几何不能任意组合；改变 MIG 配置要清空 GPU 上的所有进程；
- MIG 在 K8s 里：GPU Operator 的 MIG Manager 用 `nvidia.com/mig.config` 节点标签驱动配置；device plugin 的 `single` 与 `mixed` 策略——前者上报 `nvidia.com/gpu`，后者上报 `nvidia.com/mig-1g.5gb` 这类细分资源；
- 时间片与 MPS 在 device plugin 里的配置：`sharing.timeSlicing` 与 `sharing.mps` 把一张卡上报为 N 个副本；两者互斥且节点级生效；MPS 支持仍标为实验性、不能与 MIG 同用；时间片没有显存隔离——一个进程 OOM 会影响同卡所有进程；
- HAMi：CNCF 孵化项目，用 `nvidia.com/gpumem` / `nvidia.com/gpucores` 表达细粒度请求，通过 CUDA API 拦截（`libvgpu.so`）在软件层强制显存上限和算力比例；它与 NVIDIA 官方 device plugin 互斥；HAMi-DRA 子项目向 DRA 迁移；
- 商业 vGPU（NVIDIA vGPU / vCS）与虚拟机场景的简要对照，为什么容器场景下它不是主流；
- 切分对引擎的影响：一个 MIG 实例上的推理吞吐不是整卡的 1/7 而可能更低或更高（取决于负载是 memory-bound 还是 compute-bound）；NCCL 不支持跨 MIG 实例通信，训练几乎不用切分；
- 决策树：训练不切；推理服务按模型大小和 SLA 选 MIG（要隔离）或 HAMi / MPS（要弹性）；开发环境和 notebook 用时间片。

核心问题是：

> **同一张 A100 上跑三个小模型的推理服务，用 MIG `3g.20gb` + `3g.20gb`、用 HAMi 按显存切三份、用时间片开三个副本——三种方案在隔离性、总吞吐、故障影响范围上各自怎样？哪种方案下一个服务的 OOM 会拖垮另外两个？**

实践：在一张 GPU 上分别配置 MIG（若硬件支持）和 HAMi，各部署两个推理服务，用压测对比总吞吐和尾延迟；故意让其中一个服务 OOM，观察另一个是否受影响。

### 5. 网络与存储：RDMA 进容器、并行文件系统与 checkpoint I/O

第五篇讲两种"喂饱 GPU"的管道。多机训练的速度上限由节点间网络决定，训练能否长期运行由 checkpoint 能否按时写完决定。这两件事在单机上都不存在，上了集群就成为主要瓶颈。

这一篇会覆盖：

- 为什么 K8s 默认网络不够：CNI 给每个 Pod 一张 veth 网卡走 overlay，NCCL 需要的是直接访问 RDMA 设备（`/dev/infiniband/*`）和 GPUDirect RDMA（网卡直接读写显存）；
- 第二张网卡的机制：Multus CNI 让 Pod 挂多个网络；`NetworkAttachmentDefinition`；host-device / macvlan / IPoIB 几种接入方式；
- 把 RDMA 设备给容器：RDMA shared device plugin（多个 Pod 共享一个 HCA，上报 `rdma/rdma_shared_device_a` 这类资源）与 SR-IOV device plugin（每个 Pod 一个 VF，隔离更强）；RDMA 子系统的 shared / exclusive 网络命名空间模式；
- NVIDIA Network Operator：用 `NicClusterPolicy` 统一部署 OFED / DOCA 驱动容器、device plugin、Multus 与 IPAM；它与 GPU Operator 如何配合启用 GPUDirect RDMA（DMA-BUF 或 `nvidia-peermem`）；
- 验证与排障：容器内 `ibv_devinfo`、`nccl-tests` 的 all_reduce 带宽、`NCCL_DEBUG=INFO` 里判断走了 IB 还是 socket；常见的坑——Pod 没拿到 RDMA 设备而回退到 TCP、`NCCL_IB_HCA` 选错网卡、`NCCL_SOCKET_IFNAME` 指向 overlay 网卡；
- 存储的三类需求：数据集读取（海量小文件、随机读、可缓存）、checkpoint 写入（大块顺序写、突发、对延迟不敏感但对吞吐敏感）、模型权重分发（一次写多次读、推理扩容时的并发读）；
- 存储方案的定位：并行文件系统（Lustre、GPFS / Storage Scale、WEKA、BeeGFS）提供 POSIX 和高吞吐；对象存储（S3 及兼容实现）便宜、可扩展但不是 POSIX；缓存层（JuiceFS、Alluxio、Fluid）把对象存储包装成有本地缓存的文件系统；K8s 里用 CSI 挂载；
- checkpoint I/O 的算术：一个 70B 模型的完整训练状态（BF16 参数 + FP32 主参数 + Adam 两个动量）约 1 TB 量级，每 30 分钟写一次、要求在 1 分钟内写完，对应多少聚合带宽；分布式 checkpoint（每个 rank 写自己的分片）与异步 checkpoint（先拷到主机内存再后台写）如何把这个需求降下来；
- 推理侧的权重加载：扩容一个副本要从存储读几十到几百 GB，节点本地 NVMe 缓存、镜像内嵌权重、P2P 分发各自的适用场景；GPUDirect Storage 是什么、什么时候值得用。

核心问题是：

> **一个 8 节点 64 卡的训练任务，`nccl-tests` 在容器里测出的 all_reduce 带宽只有裸机的三分之一。从 Pod 的网络配置、device plugin 的资源分配、NCCL 的环境变量三个层面，各自可能出了什么问题？**

实践：为集群配置 Multus + RDMA device plugin（若有 RDMA 网卡；否则用 host network 模拟），在容器内跑 `nccl-tests` 并与裸机对照；挂一个 JuiceFS 或 S3 CSI 卷，测 PyTorch 分布式 checkpoint 的写入吞吐，对比同步与异步保存。

### 6. Serving 平台：从 InferenceService 到 llm-d

第六篇转入交付层。一个 vLLM 进程加一个 Service 就是最简单的推理服务，但它没有扩缩容、没有多节点副本、没有灰度、没有 PD 分离。Serving 平台把这些补上。

这一篇会覆盖：

- 推理服务的部署形态：单 Pod 单卡、单 Pod 多卡（TP 在节点内）、多 Pod 一副本（TP / PP 跨节点）、PD 分离（prefill 和 decode 是不同的 Pod 组）；每种形态用什么 K8s 对象表达；
- LeaderWorkerSet：把"一个 leader 加 N 个 worker"作为一个副本单元，稳定的 DNS 名、组级别的重启策略、组级别的滚动升级；vLLM 多节点部署的标准做法；`DisaggregatedSet` 对 PD 分离的支持；
- KServe：`InferenceService` CRD 面向传统模型（predictor / transformer / explainer 三段式，Knative 或 raw Deployment 两种模式）；`LLMInferenceService` CRD 面向 LLM，内置 llm-d 的架构——路由、调度器（Endpoint Picker）、prefill / decode 分离；两个 CRD 的分工；
- Triton Inference Server：model repository 与 `config.pbtxt`，多后端（TensorRT、ONNX、PyTorch、Python、vLLM、TensorRT-LLM），dynamic batching 与 ensemble；它作为"通用推理服务器"与 vLLM 这类"LLM 引擎"的关系——Triton 可以把 vLLM 作为后端；
- Ray Serve：以 Python 代码而非 YAML 描述服务拓扑，deployment 的副本与自动扩缩容，多模型组合；KubeRay 的 `RayService` 如何把它放到 K8s 上；适合什么场景（多阶段流水线、Python 逻辑重）；
- llm-d：vLLM 社区与 K8s 社区共同推动的 LLM Serving 栈——基于 Gateway API Inference Extension 的推理网关、KV cache 感知的路由、PD 分离部署、多节点推理；它与 KServe 的关系；
- 扩缩容：为什么 CPU 利用率对推理服务没有意义；HPA 接自定义指标（`vllm:num_requests_waiting`、`vllm:kv_cache_usage_perc`）；KEDA 的 `ScaledObject`；缩容到零与冷启动时间的矛盾；PD 分离下 prefill 和 decode 按不同指标独立扩缩；
- 模型权重的加载：`storageInitializer` 从 S3 / PVC / HuggingFace 拉取权重，初始化容器与主容器的分工，扩容时间的分解（调度 + 拉镜像 + 拉权重 + 加载到显存 + 预热）。

核心问题是：

> **一个 TP=4 的 70B 模型服务，晚高峰要从 2 副本扩到 6 副本，每个副本从调度到能接流量要 8 分钟。扩缩容指标选什么、阈值定多少、提前多久触发，才能在高峰到来前就绪而不在平时浪费 16 张卡？**

实践：用 KServe 的 `LLMInferenceService`（或 LeaderWorkerSet + 自建 Service）部署一个多副本 vLLM 服务，接 KEDA 按 `vllm:num_requests_waiting` 扩缩容，压测观察扩容延迟；若资源允许，部署一个 PD 分离的形态并对比 TTFT。

### 7. 模型网关与多租户：路由、配额与灰度

第七篇讲请求进入平台后的第一站。一个模型网关在传统 API 网关的基础上，要多做几件 LLM 特有的事：按模型名路由到不同后端、按 KV cache 状态选副本、按 token 而不是按请求限流、对流式响应计费。

这一篇会覆盖：

- 为什么需要专门的推理网关：Service 的轮询负载均衡对 LLM 请求是错的——一个副本 KV cache 已满而另一个空闲，轮询会让请求在满的那个排队；prefix caching 使得"同一前缀的请求去同一副本"有明显收益；
- Gateway API Inference Extension：`InferencePool`（一组同模型同硬件的 Pod，替代 Service 作为 `HTTPRoute` 的后端）与 Endpoint Picker（外部处理器，按每个副本的队列长度、KV cache 使用率、已加载的 LoRA 选择目标）；`InferenceObjective` 表达请求的优先级；Envoy 的 `ext_proc` 机制；Istio、Envoy Gateway、kgateway 等实现；
- OpenAI 兼容协议作为事实标准：`/v1/chat/completions`、流式 SSE 响应、`model` 字段作为路由键；网关在多后端间做协议归一；
- 多租户：租户识别（API key、JWT、命名空间），每租户每模型的配额（RPM、TPM、并发数），优先级与抢占（高优先级租户的请求是否可以让低优先级的排队），公平性；
- 限流与计费的粒度问题：请求进来时不知道要生成多少 token，按输入 token 预扣、按输出 token 结算；流式响应中途断开怎么计；`usage` 字段的可靠性；
- 模型版本与灰度：同一模型名后面挂 BF16 和 FP8 两个版本按比例分流，按租户或按 header 定向，回滚；LoRA adapter 的动态加载与按 adapter 路由；
- 多集群与多区域：模型在多个集群部署时的路由（`InferencePoolImport` 这类跨集群机制处于什么阶段），就近访问与容量溢出；
- 网关自身的容量：LLM 请求的长连接和流式响应让网关的连接数和内存占用远高于普通 API，网关自己的扩缩容和高可用。

核心问题是：

> **两个租户共用一个 70B 模型的 4 个副本，A 租户的配额是 B 的三倍。当两者同时打满时，网关应该按什么规则决定哪个请求排队、排在哪个副本上？"配额"在这里指的是 GPU 时间、token 数还是请求数？**

实践：在第六篇的服务前面部署一个实现了 Gateway API Inference Extension 的网关，配 `InferencePool` 和 Endpoint Picker，对比轮询与 KV cache 感知路由下的 TTFT 分布；为两个 API key 配不同的 TPM 配额并验证限流行为。

### 8. 可观测、成本与 FinOps

最后一篇讲平台的反馈回路。前七篇建的东西如果看不见运行状态、算不清成本，就无法改进。这一篇把从硬件到请求的指标串成一条线，再把指标变成账单。

这一篇会覆盖：

- 指标的四层：硬件（DCGM Exporter：`DCGM_FI_DEV_GPU_UTIL`、`DCGM_FI_PROF_SM_ACTIVE`、`DCGM_FI_DEV_FB_USED`、温度、功耗、XID 错误）、容器与 Pod（kube-state-metrics、cAdvisor，GPU 分配状态）、引擎（vLLM 的 `vllm:*` 指标族：TTFT、TPOT、队列长度、KV cache 使用率、prefix cache 命中率；训练框架的 step time、loss、MFU）、请求（网关的 token 计数、租户、模型版本）；
- 为什么 `DCGM_FI_DEV_GPU_UTIL` 不能说明利用率——它只表示这段时间内有 kernel 在跑；`SM_ACTIVE` 与 `DRAM_ACTIVE` 这类 profiling 指标才接近真实负载；分配率、使用率、有效利用率三个数字的差距是平台改进的空间；
- 指标管线：Prometheus 的采集与联邦、高基数问题（每请求一个 label 会撑爆时序库）、OpenTelemetry 的 trace 如何把一个请求从网关到引擎串起来、日志里的 NCCL 和 XID 错误如何告警；
- 训练任务的可观测：任务级的 GPU 利用率时间线、每个 rank 的 step time 分布（找 straggler）、checkpoint 写入耗时、通信与计算的比例；
- 推理服务的 SLO：TTFT 和 TPOT 的分位数、goodput（满足 SLO 的请求吞吐）、每副本的饱和点；从 SLO 反推容量规划；
- 成本模型：GPU 小时单价 × 分配时长是基础；按团队 / 项目 / 模型分摊（Kubernetes label 与队列作为分摊维度）；OpenCost / Kubecost 这类工具对 GPU 的支持程度；闲置成本（分配了但没用）、排队成本（任务等待的机会成本）、冷启动成本；
- 推理的单位经济：每百万 token 的成本怎么算——GPU 单价、吞吐、利用率三者的乘积；量化、PD 分离、批大小对这个数字的影响；与 API 厂商定价的对比；
- FinOps 的反馈回路：把成本数据回流到配额设置（哪个队列长期闲置）、切分策略（哪些服务该上 MIG）、扩缩容参数（缩容到零能省多少）；
- 容量规划：从历史负载曲线到采购或预留决策，训练与推理的错峰复用。

核心问题是：

> **一个 64 卡集群上月账单 X 元，DCGM 显示平均分配率 85%、平均 `SM_ACTIVE` 35%。这 50 个百分点的差距分别来自哪里——排队等 gang、训练的通信等待、推理的低峰空转、开发环境的长期占用？每一项对应本系列哪一篇的机制？**

实践：为集群部署 DCGM Exporter + Prometheus + Grafana，做一张同时显示分配率、`SM_ACTIVE`、任务队列长度和推理 TTFT 的看板；用 label 给前几篇部署的任务和服务分摊成本，算出每百万 token 的成本。本篇最后给出全系列总结。


## 贯穿全系列的实践线

系列的练手项目是**在一个小 K8s 集群上从零搭一个能跑训练任务和推理服务的最小 AI 平台**。它不追求生产级的高可用，但每一层都用真实组件、每一个决定都能对照到前面讨论的引擎需求：

```text
第一篇    裸 K8s 集群 · 一个 Pending 的 GPU Pod            三到四节点，先不装任何 GPU 组件
第二篇    GPU Operator · 版本不匹配的复现 · DRA 初试        Pod 能看到 GPU，读懂兼容报错
第三篇    Kueue + Kubeflow Trainer · Volcano 对照           2 节点 DDP 任务从 suspended 到 admitted
第四篇    MIG 或 HAMi 切分 · 两个共卡的推理服务             压测对比吞吐，验证隔离
第五篇    Multus + RDMA device plugin · nccl-tests          存储 CSI · 分布式 checkpoint 吞吐
第六篇    LLMInferenceService 或 LWS · KEDA 扩缩容          多副本 vLLM，测扩容延迟
第七篇    Inference Gateway · InferencePool · 租户配额      对比轮询与 KV 感知路由
第八篇    DCGM + Prometheus + Grafana · 成本分摊            一张看板，一份账单
```

到第八篇结束，读者手上有一个能提交训练任务、能部署和扩缩推理服务、能按租户限流、能看到利用率和成本的平台。它的每个组件都是生产环境里在用的，差别只在规模和高可用配置。硬件要求会在每篇标注：大部分内容用云上三到四台单卡实例就能完成；MIG 需要 A100 / H100 这类数据中心 GPU；RDMA 部分需要带 IB 或 RoCE 网卡的实例，没有的话用 host network 完成配置走读。

与它平行的源码与文档阅读线。平台层的"源码"更多是 CRD 定义、Operator 的 reconcile 逻辑和官方设计文档：

```text
第二篇    k8s-device-plugin  cmd/nvidia-device-plugin · api/config/v1；kubernetes  KEP-4381（DRA structured parameters）
第三篇    kueue  apis/kueue/v1beta2 · pkg/scheduler；volcano  pkg/scheduler/plugins/gang；trainer  pkg/runtime
第四篇    k8s-device-plugin  internal/rm（MIG 与 sharing 的资源上报）；HAMi  pkg/scheduler · libvgpu 的拦截点
第五篇    network-operator  NicClusterPolicy CRD；k8s-rdma-shared-dev-plugin；nccl-tests；PyTorch torch/distributed/checkpoint
第六篇    kserve  pkg/apis/serving/v1alpha1/llm_inference_service_types.go；lws  api/leaderworkerset/v1；llm-d 架构文档
第七篇    gateway-api-inference-extension  api/v1（InferencePool）· pkg/epp（Endpoint Picker 的调度插件）
第八篇    dcgm-exporter 的默认指标表；vLLM  docs/design/metrics.md；OpenCost 的 GPU 计费逻辑
```


## 阅读路径建议

### 完整学习路径

```text
1 → 2 → 3 → 4 → 5 → 6 → 7 → 8
```

### 只负责训练集群

```text
1 → 2 → 3 → 5 → 8
```

跳过切分和交付层。第五篇的网络部分和第三篇的拓扑感知是训练集群的核心。

### 只负责推理平台 / LLMOps

```text
1 → 2 → 4 → 6 → 7 → 8
```

跳过训练调度和 RDMA。第四篇的切分决定推理的成本结构，第六到八篇是交付层全部。

### 云原生背景，想快速理解 AI 负载的特殊之处

```text
1 → 3 → 4 → 6
```

每篇的"引擎的需求"和"K8s 的空缺"两节是重点，机制部分可以快速翻过。

### 引擎开发者，想知道平台需要什么接口

```text
1 → 3（gang 与 checkpoint 恢复）→ 6（扩缩容指标）→ 7（路由所需的状态暴露）→ 8（指标规范）
```


## 本系列的边界

本系列只讨论引擎之下的资源层和引擎之侧的交付层。以下内容与它紧邻，但不在范围内：

- **引擎内部的机制**：训练框架的并行策略与状态切分、推理引擎的请求调度与 KV cache 管理、PD 分离在引擎内的实现。本系列把它们当作已知的需求来源——例如"TP 需要节点内高带宽"、"KV cache 满了请求会排队"——只讨论平台如何响应这些需求。
- **通信与网络协议的内部**：NCCL 的算法与 channel、RDMA 的 verbs 编程、InfiniBand 与 RoCE 的协议差异。本系列第五篇只讨论如何让 RDMA 设备在容器里可用、如何验证 NCCL 走了正确的路径，不解释 NCCL 为什么选了这条路径。
- **GPU 硬件与 kernel**：SM、显存层次、kernel 的性能分析。第四篇讨论 MIG 时会用到"SM 与显存带宽被分区"这个事实，但不展开它对 kernel 性能的影响。
- **Kubernetes 本身**：Pod、Deployment、Service、CRD、Operator 模式、CNI、CSI 的一般原理。本系列假设读者已经掌握，只讲 AI 负载下的特殊之处。
- **数据平台**：数据采集、清洗、湖仓、特征存储。第五篇只讨论训练数据读取和 checkpoint 写入对存储的要求，不讨论数据本身怎么来。
- **实验管理与模型仓库**：实验跟踪、模型版本与血缘、评测流水线。它们是 MLOps 的一部分，但与 GPU 资源和服务交付关系较远，本系列只在第七篇的版本路由处提及模型仓库作为版本来源。
- **应用层**：RAG、Agent 框架、prompt 管理。它们在网关之上，属于应用开发。


## 前置要求与说明

### 前置要求

- 了解 Kubernetes 的基本对象和扩展机制：Pod、Deployment、StatefulSet、Job、Service、ConfigMap、PVC；知道 CRD 和 Operator 是什么，读过一两个 CRD 的 spec；用过 Helm；
- 跑过至少一个训练任务或推理服务：知道 `torchrun` 或 `torch.distributed` 的启动方式和 `WORLD_SIZE` / `RANK` 的含义，或者部署过 vLLM 这类引擎并调用过它的 OpenAI 兼容接口；
- 知道 GPU 显存是硬约束、多机训练靠 NCCL 通信、模型权重是几十到几百 GB 的文件——不需要理解这些东西的内部；
- 基本的 Linux 网络与存储概念：网卡、网络命名空间、块设备与文件系统、NFS；
- 用过 Prometheus 和 Grafana，或至少知道指标、label、PromQL 是什么；
- 一个可用的 K8s 集群：**至少三个节点、每节点至少一张 NVIDIA GPU**（云上按需实例即可，几十小时的用量能完成大部分实践）；MIG 和 RDMA 相关实践对硬件有额外要求，随文标注。

不要求：

- 写过 Kubernetes Operator 或调度器插件；
- 了解 Volcano、Kueue、KServe 中的任何一个；
- 用过 Slurm 或 Ray；
- 会写 CUDA 或读过引擎源码。

### 组件与版本基线

- Kubernetes **1.34 及之后**（DRA 的 `resource.k8s.io/v1` 在此版本 GA）；device plugin 路径对更早版本同样适用，随文标注；
- NVIDIA 侧：GPU Operator 25.x / 26.x、k8s-device-plugin 0.17 及之后、驱动 R570 及之后；CUDA 12.x 为默认基线，CUDA 13.x 的驱动要求随文标注；
- 调度：Kueue 0.14 及之后（Topology-Aware Scheduling 进入 beta）、Volcano 1.11 及之后、Kubeflow Trainer 2.x（`TrainJob` API，取代 Training Operator v1 的 `PyTorchJob`）、KubeRay 1.x；
- 交付：KServe 0.16 及之后（`LLMInferenceService` 从 0.16 起提供，正文以 0.20 附近为准）、LeaderWorkerSet 0.7 及之后（`DisaggregatedSet` 从 0.9 起随包提供）、Gateway API Inference Extension 1.0 及之后（`InferencePool` v1）、Triton Inference Server 2.x、KEDA 2.x；
- 可观测：DCGM Exporter 3.x / 4.x、Prometheus 2.x / 3.x、OpenTelemetry Collector；
- 引擎作为被服务对象：PyTorch 2.x、vLLM 0.x 主线，正文只使用它们对外暴露的接口（启动参数、指标、OpenAI 兼容 API），不依赖内部实现。

这一层的组件版本变化比引擎更快，尤其是 DRA、Inference Extension 和 llm-d 这几处仍在快速演进。正文的原则是：**先讲机制和取舍，再讲当前的 API 形态**，API 变化时机制部分仍然成立。版本敏感处随文标注。

### 关于云厂商托管服务与其他硬件

本系列以自建 K8s 集群为主线，因为托管服务（各云的 K8s GPU 节点池、托管推理服务）把本系列要讨论的大部分机制封装掉了。正文会在相关位置说明托管服务替你做了哪一步、代价是什么。AMD GPU 有对应的 device plugin 和 Operator，机制与 NVIDIA 侧高度对应，会在第二篇和第四篇提及差异；国产加速器不在范围内。


## 章节目录

1. [引擎的需求清单与平台的整体架构](/ai-platform-engine-requirements-and-architecture.html)
2. [容器里的 GPU：驱动、CUDA、device plugin 与镜像](/gpu-in-containers-driver-cuda-device-plugin.html)
3. [AI 任务调度：gang scheduling、队列与拓扑感知](/ai-job-scheduling-gang-queue-topology.html)
4. [GPU 共享与切分：MIG、时间片、MPS 与 HAMi](/gpu-sharing-and-partitioning-mig-mps-hami.html)
5. [网络与存储：RDMA 进容器、并行文件系统与 checkpoint I/O](/rdma-networking-storage-and-checkpoint-io.html)
6. [Serving 平台：从 InferenceService 到 llm-d](/serving-platforms-kserve-triton-ray-serve-llm-d.html)
7. [模型网关与多租户：路由、配额与灰度](/model-gateway-multi-tenancy-and-quota.html)
8. [可观测、成本与 FinOps](/ai-platform-observability-cost-and-finops.html)


## 最终目标

读完这套系列之后，面对一个 GPU 集群上的任何异常——任务 Pending、训练变慢、服务超时、账单超支——读者应该能够沿着平台的层次追问下去：

```text
Pod 为什么看不到 GPU？                       → 第二篇：驱动、Container Toolkit、device plugin 的链条
任务为什么 Pending？配额够却调度不上？          → 第三篇：gang、队列、拓扑约束、碎片
两个服务共卡时为什么互相影响？                  → 第四篇：隔离层次与故障域
多机训练为什么比单机慢这么多？                  → 第五篇：RDMA 是否生效、NCCL 走了哪条路
checkpoint 为什么写不完？                     → 第五篇：存储带宽与 checkpoint 形态
服务为什么扩容慢、扩了还是超时？                → 第六篇：扩缩容指标、冷启动、副本形态
请求为什么在一个副本上排队而另一个空着？         → 第七篇：路由策略与 Endpoint Picker
利用率 35% 的差距去了哪里？钱花在哪个团队？      → 第八篇：指标层次与成本分摊
```

最终目标是三种能力：

1. **设计能力**：面对一组训练和推理负载，能选出合适的调度器、切分策略、网络与存储方案、Serving 形态和网关，并说清每个选择的取舍；
2. **排障能力**：面对集群上的异常，能从引擎报错定位到平台的哪一层、哪个组件、哪项配置；
3. **运营能力**：能建立从硬件到请求的可观测体系，把利用率和成本的差距分解到具体原因，并用它驱动配额、切分和扩缩容策略的调整。

这是把引擎变成可共享的资源池和可交付的服务的那一层——引擎再快，没有它也到不了用户手里。
