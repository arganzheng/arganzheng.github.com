---
layout: post
series: ai-platform-engineering
title: "AI 平台工程（09）：系列总结与通关自测"
subtitle: "AI Platform Engineering: Series Recap and Final Self-Test"
tags: [Kubernetes, MLOps, GPU, AI, AI-Infra]
catalog: true
date: 2026-09-22 20:00:00 +0800
---

八篇正文回答了一个问题：**一个 GPU 集群如何被切分、调度和喂饱，一个训好的模型如何变成一个可运维的服务**。第一篇从一个 Pending 的 GPU Pod 出发列出引擎对平台的需求清单；第二到五篇是资源层——容器里的 GPU、一组 Pod 的调度、一张卡的切分、第二张网卡与 checkpoint 的带宽；第六到八篇是交付层——副本的形态与扩缩容、请求的路由与配额、从 DCGM 到账单的反馈回路。

本文不讲新内容，做三件事：把八篇压成一张表与八段回顾，把贯穿八篇的几条线拎出来，然后给一套三段式的通关自测——判断与计算、跨篇综合、面试题。各篇末尾的自测检验的是"这一篇读懂了没有"，这里检验的是"八篇能不能连起来用"：面对一个 Pending、一次变慢、一份超支的账单，能不能沿着层次追问下去。

> **读完这八篇，你应该能回答哪些问题？[^q0] 哪些数字与结论必须能脱口而出？[^q1] 怎么判断自己是"读过"还是"掌握"了？[^q2]**

## 一、总览：系列回答的问题与主线

系列的一句话主张是：**平台的每一个设计决定都是被引擎的某个需求推出来的，而每个决定都有代价**。每篇同一个骨架——引擎的需求 → K8s 的空缺 → 平台的机制 → 代价与边界；同一条主线：Kubernetes 的四个假设（Pod 独立、资源可细分、一张 overlay 网卡、HPA 看 CPU）被 AI 负载逐条违背，于是有了 device plugin、Kueue / Volcano、MIG / HAMi、Multus、LeaderWorkerSet、InferencePool、DCGM 映射这一层又一层的扩展；每一层都填一个洞，也都挖一个新的。

| 篇 | 回答的问题 | 一句话结论 | 必记的数字 / 公式 |
|---|---|---|---|
| [第一篇：需求清单与整体架构](/ai-platform-engine-requirements-and-architecture.html) | 32 卡训练任务与 TP=2 推理服务各列一张需求表，哪几条原生 Kubernetes 满足不了？ | 空缺分缺插件、缺概念、缺信号三种；训练卡在调度与网络，推理卡在扩缩容与路由，唯一共同的空缺是"GPU 是不透明整数"；平台分资源层与交付层，分界线是"Pod 能跑了" | 训练 15 条原生满足 1 条、推理 15 条满足 2 条；80 GB 卡 × 0.92：7B 权重 14 GB → KV 约 60 GB，30B 权重 60 GB → KV 约 14 GB；扩容一个副本 5–10 分钟 |
| [第二篇：容器里的 GPU](/gpu-in-containers-driver-cuda-device-plugin.html) | 驱动 580 + CUDA 13.1 镜像 + 调用 13.1 新 API 能跑吗？驱动 570 呢？ | 四层栈、三条规则：向后兼容无条件；minor version 同大版本内旧驱动跑新 Toolkit（新驱动 API 与 PTX JIT 除外）；forward 跨大版本只有数据中心 GPU + cuda-compat 包 | 基线 12.x ≥ 525.60.13、13.x ≥ 580.65.06；错误码 35 / 36 / 222；device plugin 只计数、无属性、不跨 Pod 共享；DRA `resource.k8s.io/v1` 自 1.34 GA；10 GB 镜像冷拉 2–3 分钟 |
| [第三篇：AI 任务调度](/ai-job-scheduling-gang-queue-topology.html) | A、B 各 16 卡配额，A 提交 32 卡任务，B 空闲——Volcano、Kueue、Slurm 里各会怎样？ | Volcano 在 Pod 之后做节点级 gang，Kueue 在 Pod 之前做配额级 gang；默认值下 Volcano 与 Kueue 借了不还、Slurm 不借 | `guarantee ≤ deserved ≤ capability`；`nominalQuota` / `borrowingLimit` / `lendingLimit` + 三个抢占开关；抢占代价 $$N_{gpu} \times (T_{since\_ckpt} + T_{restart})$$；30/32 死锁 |
| [第四篇：GPU 共享与切分](/gpu-sharing-and-partitioning-mig-mps-hami.html) | 同一张 A100 跑三个小模型：MIG `3g.20gb × 2`、HAMi 切三份、时间片三副本——隔离、吞吐、故障域各怎样？ | 四档取舍都是"device plugin 把一张卡复制成 N 个逻辑设备"的不同底座；时间片无隔离、HAMi 软件隔离但故障域整卡、MIG 连 Xid 都隔离但几何刚性；训练不切 | A100 7 个计算 slice、8 个显存 slice，`3g.20gb × 2` 只能放两个服务；`3g.20gb` = 3/7 算力 + 4/8 带宽；HAMi `deviceSplitCount` 默认 10；时间片下 OOM 会蔓延 |
| [第五篇：网络与存储](/rdma-networking-storage-and-checkpoint-io.html) | 容器里 `nccl-tests` 只有裸机三分之一——Pod 网络、device plugin、NCCL 环境变量三层各可能错在哪？ | NCCL 找不到 RDMA 就静默回落 Socket，缺任何一项都不报错；Multus 给第二张网卡、RDMA device plugin 给 `/dev/infiniband`、`IPC_LOCK` 与 peermem / DMA-BUF 给 GDR | 14 字节/参数 → 70B ≈ 1 TB；30 分钟一次、1 分钟写完 → 16.7 GB/s 聚合、2.1 GB/s 每节点、260 MB/s 每 rank；异步 → 0.56 GB/s，代价每节点约 125 GB pinned 内存；`NCCL_NET_GDR_LEVEL` 默认 `PXB` |
| [第六篇：Serving 平台](/serving-platforms-kserve-triton-ray-serve-llm-d.html) | TP=4 的 70B 服务晚高峰 2 → 6 副本、就绪 8 分钟——指标、阈值、提前多久？ | CPU 无意义；领先指标 `num_requests_running` / `kv_cache_usage_perc`，`waiting` 只兜底；纯反应式阈值 = 峰值过配；可预测的高峰用 cron 提前 | 阈值 36/副本（饱和点 64 的 56%）→ 峰值 10 副本而非 6；cron 提前 12 分钟 + running 56 + waiting 4 + 缩容 15 分钟窗口 ≈ 5 卡时/天，对比常驻 328 卡时/天；信号延迟约 1 分钟 |
| [第七篇：模型网关与多租户](/model-gateway-multi-tenancy-and-quota.html) | 两租户共用 4 个副本、A 配额是 B 三倍、同时打满——按什么规则排队、排在哪个副本？"配额"是什么？ | 外层按 RPM / TPM / 并发 429，内层 EPP 按 priority 严格优先、同级 round-robin，无按权重公平；选副本与租户无关；配额是 token + 并发 + RPM，GPU 时间只做内部成本 | 64 会话 × 8k 对 4 副本 × 160k：前缀亲和让 TTFT 从 0.5 s 级到几十 ms；预扣 = 输入估算 + min(max_completion_tokens, 上限)；打分权重前缀 3、队列 2、KV 2、LoRA 1；EPP 每 50 ms 抓一次 `/metrics` |
| [第八篇：可观测、成本与 FinOps](/ai-platform-observability-cost-and-finops.html) | 分配率 85%、`SM_ACTIVE` 35%，50 个点去了哪里？各对应哪篇？ | 四层指标靠 `pod` / `namespace` join；`GPU_UTIL` 只表示"有 kernel 在跑"；$$E \approx A \times U$$；按分配计费让闲置有主 | 50 个点：推理低峰 ~15、dev ~10、通信等待 ~10、排队占位 ~5、checkpoint ~3、冷启动 ~2、测量上限 ~5；每百万 token 成本 U 从 100% 到 40% 贵 2.5 倍（1.39 → 3.47 美元） |

### 1. 本文的章节安排

| 章 | 内容 |
|---|---|
| 二 | 逐篇回顾：核心问题、结论、必记、常见误解 |
| 三 | 贯穿八篇的五条线：引擎需求驱动机制、"不透明整数"与 device plugin 技巧、checkpoint、扩容时间、隔离 vs 利用率 |
| 四 | 常见误区表 |
| 五 | 通关自测：A 判断与计算 10 题、B 跨篇综合 5 题、C 面试题 7 题、D 掌握判据 |
| 六 | 下一步 |

## 二、逐篇回顾

### 1. 第一篇：引擎的需求清单与平台的整体架构

**核心问题**：一个 4 节点 32 卡的训练任务和一个 TP=2、副本数动态变化的推理服务，各自对平台提出的需求列成一张表，哪几条是原生 Kubernetes 满足不了的？

**结论**：训练任务是 $$W$$ 个强耦合的进程，rendezvous 等不齐就超时重建、任一进程失败整组重启，所以要整数个 GPU、全员到齐、节点间 RDMA 且拓扑相近、周期性突发写 checkpoint。推理副本的显存是硬约束——预留后权重占一块、剩下全给 KV cache，不能超卖；扩容以分钟计、主项是权重加载；负载信号只在引擎内部（`vllm:num_requests_waiting`、`vllm:kv_cache_usage_perc`、TTFT / ITL）。原生 Kubernetes 的空缺分三种：缺插件（GPU、RDMA 设备）、缺概念（一组 Pod、设备属性、队列配额、多 Pod 副本）、缺信号（引擎指标驱动的扩缩容、路由、成本）。训练卡在调度与网络、推理卡在扩缩容与路由，唯一共同的是"GPU 是不透明整数"。平台分两层：资源层把裸节点变成能跑的 Pod，交付层把 Pod 变成有 SLA 与账单的服务；训练只走资源层，推理两层都走。

**必记**：

- 训练 15 条需求原生完全满足 1 条（任务结束整体释放）；推理 15 条满足 2 条（整卡时显存独占、readinessProbe）。
- 80 GB 卡按 0.92 预留 73.6 GB：7B bf16 权重 14 GB → KV 约 60 GB；30B 权重 60 GB → KV 约 14 GB。
- 扩容一个 70B 副本 5–10 分钟：调度 + 拉镜像（10 GB 级）+ 加载权重（140 GB）+ 初始化 + 预热。
- 三对矛盾：独占 vs 共享、拓扑 vs 弹性、批处理 vs 长驻；三个共同点：多 Pod 一个单位、GPU 要有属性、成本落到 GPU 时间。
- 每叠一层的代价：版本契约的层数、对象模型翻倍（一个 `TrainJob` 背后是 JobSet、Job、Pod）、信号延迟叠加。

**常见误解**："`Insufficient nvidia.com/gpu` 说明卡不够"——对调度器来说 `allocatable` 里没有这个键与等于 0 是同一回事，三张卡在那里但没人上报。另一个："HPA 看 GPU 利用率就行"——`DCGM_FI_DEV_GPU_UTIL` 一个 decode 请求就能推到接近 100%，与还能接多少请求无关。

### 2. 第二篇：容器里的 GPU：驱动、CUDA、device plugin 与镜像

**核心问题**：宿主机驱动 580（原生 CUDA 13.0）、镜像里 CUDA 13.1 编译的 PyTorch、代码调用了 13.1 新增的 API——能跑吗？宿主机驱动是 570 呢？

**结论**：四层软件要互相接受：内核驱动与用户态驱动库在宿主机同版本、由 Container Toolkit 挂进容器，CUDA Runtime 与上层库随镜像或 wheel——`nvidia-smi` 显示的是驱动支持的上限，`torch.version.cuda` 是 Toolkit 版本，两者不同是常态。三条规则：向后兼容无条件；minor version 同大版本内旧驱动跑新 Toolkit，新增驱动 API 返回 36、PTX JIT 报 222；forward 跨大版本需 cuda-compat 包，仅数据中心 GPU，否则容器起不来或 35。580 + 13.1 能跑，只有那一处新 API 报 36；570 + 13.x 只有数据中心卡装 compat 包才行。device plugin 用 `ListAndWatch` / `Allocate` 把 GPU 变成一个整数——只计数、无属性、不跨 Pod 共享；GPU Operator 用 `ClusterPolicy` 装成一套；DRA 用 `ResourceSlice` 发布属性、`ResourceClaim` 加 CEL 按属性选设备，但 NVIDIA 的 DRA driver 在 GPU Operator v26.7.0 中仍是实验性。镜像是第五个参与者：PyTorch / vLLM 镜像 3–12 GB，绝大部分是 CUDA 库。

**必记**：

- minor version 基线：11.x ≥ 450.80.02、12.x ≥ 525.60.13、13.x ≥ 580.65.06；错误码 35 `InsufficientDriver`、36 `CallRequiresNewerDriver`、222 PTX 工具链不支持。
- Toolkit v1.20.0 默认 `auto` → jit-cdi；`deviceListStrategy` 四种：`envvar` / `volume-mounts` / `cdi-annotations` / `cdi-cri`。
- 节点标签三个来源：NFD `pci-10de.present` → Operator `gpu.present` / `gpu.deploy.*` → GFD `gpu.product` / `gpu.memory` / `cuda.driver.major`；标签是字符串，只能 `In` 枚举不能比大小。
- 10 GB 镜像冷拉 2–3 分钟；多阶段构建 devel → base 去掉 5 GB 以上；32 个 Pod 同时拉 12 GB 镜像是 registry 瞬时 400 GB 出口。
- v1.37 的 `DRAExtendedResource` GA 让 `nvidia.com/gpu: 1` 可由 DRA 满足；同节点 device plugin 与 DRA driver 同时上报会重复计数。

**常见误解**："镜像里带上 `libcuda.so` 就不用管节点驱动了"——用户态驱动库必须与内核驱动同版本，`nvidia/cuda` 的 base / runtime / devel 都不含它。另一个："驱动升级是节点级小事"——它要卸载内核模块，节点上所有 GPU 进程先停，对 30 天的训练任务是一次 checkpoint 恢复。

### 3. 第三篇：AI 任务调度：gang scheduling、队列与拓扑感知

**核心问题**：两个团队各有 16 卡配额，A 提交了一个 32 卡任务，B 的卡空着。Volcano、Kueue、Slurm 里这个任务分别会怎样？借用、抢占、等待各自的配置是什么？

**结论**：kube-scheduler 逐 Pod 决定，32 卡任务调度了 30 个就卡住——已起的 Pod 占着卡等 rendezvous 超时、重建、再等，两个大任务互相卡成活锁；"要么全起要么都不起"只能在调度 / 准入层做。Volcano 换掉调度器：Job → PodGroup（`minMember`）→ Queue，`allocate` 模拟全部 task、`JobReady` 才 Commit；`guarantee ≤ deserved ≤ capability` 表达保底 / 可借可收 / 上限，`reclaim` 收回超出 `deserved` 的部分，但 gang 插件不允许把 Job 打到 `minMember` 以下，满 gang 任务要靠 v1.15 的 `gangreclaim` 才收得回。Kueue 不换调度器：webhook 置 `suspend=true`，Workload → LocalQueue → ClusterQueue，cohort 内按 `borrowingLimit` / `lendingLimit` 借用，`reclaimWithinCohort` 把整个 Workload 驱逐回队列。差别在介入时机：Volcano 在 Pod 之后做节点级 gang，Kueue 在 Pod 之前做配额级 gang，后者的盲区是节点级放置，TAS 补这一块。拓扑 `hard` 约束换可预期的带宽，代价是等更久。Slurm 默认 `GrpTRES` 硬配额根本不借。

**必记**：

- 不配置时的行为：Volcano 借了不还（`reclaimable` 默认 true 但默认 actions 不含 `reclaim`）、Kueue 借了不还（默认 `reclaimWithinCohort: Never`）、Slurm 不借（`AssocGrpGRES`）。
- Kueue 三个抢占开关：`withinClusterQueue`、`reclaimWithinCohort`、`borrowWithinCohort`；不借是 `borrowingLimit: 0` 或不设 cohort，Workload 停在 `ExceedsMaxQuota`。
- 抢占代价 $$N_{gpu} \times (T_{since\_ckpt} + T_{restart})$$：30 分钟间隔平均丢 15 分钟，256 卡任务被抢一次平均损失 64 卡时；grace period 至少覆盖一次 checkpoint 写入。
- 三层优先级：Pod 级 `PriorityClass`、Workload 级 `WorkloadPriorityClass`、队列级；训练 Pod 一律同一个 `PriorityClass`，差异只在 Workload 层表达。
- Kubeflow Trainer：`TrainJob` → JobSet，注入 `PET_*`，`MASTER_ADDR = <job>-node-0-0.<job>`；K8s v1.37 的 WAS（`scheduling.k8s.io/v1beta1` PodGroup）是原生 gang 雏形，只解决同时性。

**常见误解**："配额就是硬上限"——配额系统能表达保底、可借、可收回，把它当硬上限用是最常见的浪费来源；32 卡任务是常态时"两个 16 卡队列"本身就是错的设计。另一个："Kueue 准入了就一定能跑"——准入是配额加法，节点实况可能放不下（碎片、taint、非 Kueue 的 Pod），要 TAS 或 `waitForPodsReady`。

### 4. 第四篇：GPU 共享与切分：MIG、时间片、MPS 与 HAMi

**核心问题**：同一张 A100 上跑三个小模型的推理服务，用 MIG `3g.20gb` + `3g.20gb`、用 HAMi 按显存切三份、用时间片开三个副本——隔离性、总吞吐、故障影响范围各怎样？哪种方案下一个服务的 OOM 会拖垮另外两个？

**结论**："安全"有三个含义——显存、算力、故障隔离——四层机制做到的程度完全不同：时间片是驱动层轮转，无任何隔离；MPS 合并 CUDA context，有软显存上限与 SM 比例但只能 1/N 均分、server 是单点、仍标实验性；HAMi 用 `libvgpu.so` 拦截 CUDA 驱动 API，按 MB 的 `gpumem` 与 `gpucores` 节流，OOM 被限制在越界容器内，但不切带宽与 L2、故障域仍是整卡；MIG 是硬件分区，Xid 都隔离在单个 GI 内，代价是几何枚举、改配置要清空 GPU、实例间无 NCCL。K8s 侧全是同一个技巧：device plugin 把一张卡复制成 N 个逻辑设备。切分对引擎的影响由瓶颈决定：decode memory-bound 看显存 slice 比例，prefill compute-bound 看计算 slice 比例；HAMi / MPS 不切带宽，总吞吐通常更高但延迟随邻居波动。时间片下一个服务的 OOM 会拖垮另外两个；题设 `3g.20gb × 2` 用尽 8 个显存 slice，只能放两个服务。决策树：训练不切；有 SLA 的推理用 MIG、要弹性用 HAMi；开发环境用时间片；一个节点池一种策略。

**必记**：

- A100 7 个计算 slice、8 个显存 slice；`3g.20gb × 2` 用 6 个计算 + 8 个显存 slice，浪费 1 个计算 slice；`3g.20gb` = 3/7 算力 + 4/8 带宽；`1g.5gb` 拿到 1/7 SM、1/8 显存与带宽。
- 7B FP16 权重 14 GB 在 A100 40 GB 上至少 `3g.20gb`；INT4 约 4 GB 可勉强进 `1g.5gb` 但 KV 几乎没空间。
- `migStrategy: single` 全部上报 `nvidia.com/gpu` 且要求全节点几何一致；`mixed` 每种 profile 一个资源名 `nvidia.com/mig-<profile>`。
- 时间片与 MPS 互斥、节点级生效、`replicas ≥ 2`、`.shared` 后缀；HAMi 与官方 device plugin 互斥，`deviceSplitCount` 默认 10，三个服务各约 13000 MB。
- vLLM 的 `--gpu-memory-utilization` 在 MIG 与 HAMi 下自动相对配额，在时间片与 MPS 下要手动压到约 1/N。

**常见误解**："1/7 的 MIG 实例就是 1/7 的吞吐"——小 batch 下整卡也没用满带宽，7 个 `1g` 实例各跑一个小 batch 服务总吞吐可以接近甚至高于整卡，代价是每请求延迟高得多。另一个："HAMi 有显存隔离所以两个服务互不影响"——它防"邻居多分了显存"，不防"邻居把 GPU 弄挂了"。

### 5. 第五篇：网络与存储：RDMA 进容器、并行文件系统与 checkpoint I/O

**核心问题**：一个 8 节点 64 卡的训练任务，`nccl-tests` 在容器里测出的 all_reduce 带宽只有裸机的三分之一。从 Pod 的网络配置、device plugin 的资源分配、NCCL 的环境变量三个层面，各自可能出了什么问题？

**结论**：带宽只剩三分之一几乎一定是 NCCL 回落到了 Socket 或 GDR 没开——引擎自己检测能力、自己降级，缺任何一项都不报错，日志里只有 `NET/IB : No device found.` → `Using network Socket`。默认 CNI 只给 Pod 一张 veth 走 overlay，RDMA 要作为第二张网卡进 Pod：Multus 当元插件、`NetworkAttachmentDefinition` 装 CNI 配置、Pod 注解引用，host-device / macvlan / ipoib 三种接入。`/dev/infiniband` 要经 device plugin 进容器：rdma-shared-dev-plugin 上报 `rdma/<name>`（要求 netns shared，无隔离、无 GPU 亲和）或 SR-IOV 一 VF 一 Pod；容器要 `IPC_LOCK`；GDR 需要 nvidia-peermem 或 DMA-BUF。NCCL 侧 `NCCL_IB_HCA`、`NCCL_SOCKET_IFNAME`、RoCE 的 `GID_INDEX`、`NCCL_NET_GDR_LEVEL` 任一错了都是同一症状。存储侧三类负载画像不同：数据集小文件随机读靠缓存层，checkpoint 大块顺序写靠并行文件系统的聚合吞吐，权重一写多读靠本地 NVMe 缓存；CSI 只是接口。checkpoint I/O 是算术题：分片让每个 rank 写自己那份，异步把停顿变成 D2H 拷贝、把带宽需求降一个量级。

**必记**：

- 训练状态 14 字节/参数（BF16 参数 2 + FP32 主参数 4 + Adam 两个动量 8）：70B ≈ 1 TB，与并行策略无关。
- 30 分钟一次、1 分钟写完：16.7 GB/s ≈ 133 Gb/s 聚合，每节点 2.1 GB/s，每 rank 260 MB/s；每 30 分钟停 1 分钟约 3% 开销。
- 异步 `dcp.async_save`：带宽只需 1 TB / 30 min ≈ 0.56 GB/s（留余量取 2–3 GB/s），代价每 rank 约 15.6 GB pinned 内存、每节点约 125 GB；聚合带宽只跟状态总量挂钩，405B 同步要 95 GB/s。
- 70B BF16 权重 140 GB：1 GB/s 两分多钟，200 MB/s 十几分钟；分发顺序本地 NVMe 缓存 → P2P → 权重进镜像；GDS 只在 CPU 成为瓶颈时值得。
- `NCCL_NET_GDR_LEVEL` 默认 `PXB`，日志 `GPU Direct RDMA Disabled ... (distance N > 3)`；修法是 GPU 与 RDMA 分配带拓扑亲和，不是放宽 `GDR_LEVEL`。

**常见误解**："Pod 全 Running、训练在推进，说明网络没问题"——NCCL 回落 TCP 不报错，慢是唯一症状，平台必须主动验证。另一个："checkpoint 写不完是存储太慢"——同步要 16.7 GB/s，异步只要 0.56 GB/s，问题往往是形态而不是带宽。

### 6. 第六篇：Serving 平台：从 InferenceService 到 llm-d

**核心问题**：一个 TP=4 的 70B 模型服务，晚高峰要从 2 副本扩到 6 副本，每个副本从调度到能接流量要 8 分钟。扩缩容指标选什么、阈值定多少、提前多久触发，才能在高峰到来前就绪而不在平时浪费 16 张卡？

**结论**：一个副本是"一份完整权重加服务它的进程组"；Pod 数为 1 用 Deployment，多 Pod 一副本用 LeaderWorkerSet（`RecreateGroupOnPodRestart`，注入 `LWS_LEADER_ADDRESS`，scale 子资源只看 leader），PD 分离用两组对象或 `DisaggregatedSet`。KServe 的 `LLMInferenceService` 按形状生成 Deployment / LWS + InferencePool + EPP + HTTPRoute，把 llm-d 包成一个 CRD。扩缩容的核心是 CPU 无意义：领先指标是 `vllm:num_requests_running` 与 `kv_cache_usage_perc`，`waiting` 非零时已经饱和，只能兜底。扩容时间分五段——调度、拉镜像、拉权重、加载到显存、预热——正是 8 分钟，再加约 1 分钟信号延迟；决策时刻现有副本必须还有 $$r \times (T_{signal} + T_{ready})$$ 的余量，倒推出每副本阈值 36。但领先阈值等于峰值过配：HPA 在 350 并发时算出 10 副本而非 6。可预测的高峰用 cron 提前 12 分钟抬地板，反应式阈值抬到 56 只管意外流量，缩容 15 分钟窗口 + 每 5 分钟 1 个，合计每天约 5 卡时。缩零对 70B 是把 8 分钟冷启动暴露给用户，只适合冷启动 30 秒以内的小模型。

**必记**：

- 阈值公式 $$N \cdot C - D(t) \ge r (T_{signal} + T_{ready})$$：N = 2、C = 64、r = 6.25/分钟、9 分钟 → D ≤ 72 → 每副本 36（56%）；换成 KV 是 36 / 110 ≈ 0.33（依赖平均上下文 4k）。
- 三种方案的代价：反应式阈值 36 → 峰值 10 副本、多 16 卡 × 3.5 h ≈ 56 卡时/天；常驻 6 → 328 卡时/天；cron + 阈值 56 + 慢缩容 ≈ 5 卡时/天。
- 信号延迟 ≈ Prometheus 抓取 30 s + KEDA `pollingInterval` 15 s + HPA 同步 15 s ≈ 1 分钟。
- 五段量级：云上加节点 3–10 分钟；未缓存镜像 1–2 分钟；权重对象存储 1 GB/s 2–3 分钟 / 并行文件系统 30 秒 / 本地缓存 ≈ 0；加载 1–3 分钟；预热 1–3 分钟。
- `startupProbe` 60 × 30 s——大模型加载以半小时为上限配探针；HPA 不能缩零（`HPAScaleToZero` 仍是 alpha），缩零靠 KEDA `minReplicaCount: 0`；权重四条路径 `pvc://`、`s3://` / `hf://`、`oci://`、`LocalModelCache`。

**常见误解**："阈值没调好才过配"——为 9 分钟就绪预留的 44% headroom 在稳态高峰也被保留，是反应式扩缩容的结构性代价。另一个："缩容阈值接近扩容阈值更灵敏"——新副本刚就绪负载就落回去、被缩掉、再涨再扩，抖动 8 分钟一周期。

### 7. 第七篇：模型网关与多租户：路由、配额与灰度

**核心问题**：两个租户共用一个 70B 模型的 4 个副本，A 租户的配额是 B 的三倍。当两者同时打满时，网关应该按什么规则决定哪个请求排队、排在哪个副本上？"配额"在这里指的是 GPU 时间、token 数还是请求数？

**结论**：轮询错在两处：不看 KV 满不满——请求一旦送到副本就迁不走；不看缓存在哪——prefix cache 在副本本地，同一会话落到另一个副本就是一次全量 prefill。GIE v1.6.0 只定义 `InferencePool`（替代 Service 作 `HTTPRoute` 后端）与 ext_proc 协议，选副本的逻辑在 llm-d-router 的 EPP：parse → 模型名重写 → `InferenceObjective` 取 priority → fairness ID → flow control → filter → 加权打分 → picker，用 `x-gateway-destination-endpoint` 告诉网关选了哪个 Pod。租户分两层：外层认证层识别租户、剥掉伪造的 `x-llm-d-*` 头、按租户 × 模型做 RPM / TPM / 并发限流，超配额直接 429；内层按 priority 严格优先、同级 round-robin——v0.10 没有按权重的公平，A 的"三倍"只能靠外层的桶体现。排在哪个副本与租户无关：前缀亲和 → `utilization-filter` 丢掉 KV > 0.9 或队列 > 4 的 → 加权分。"配额"对租户是 token（输入 / 输出 / 命中分开）+ 并发 + RPM，GPU 时间到达时不可知、只做内部成本。要在空间上隔离两个租户就得拆两个池——隔离 vs 利用率在网关层的形态。

**必记**：

- 数值例子：4 副本各 160k token KV、64 会话 × 8k、prefill 16k token/s：轮询命中率 < 1/4、未命中一轮 0.5 s；前缀亲和每副本 128k 在容量内、TTFT 几十 ms。
- 请求数是错的量纲：500 token 与 30k token 差 60 倍。
- 预扣 = 输入估算 + min(`max_completion_tokens`, 单请求上限)；按 `usage` 结算；流式必须强制 `stream_options.include_usage`；排队中断开退全部预扣，生成中断开按已生成计。
- EPP 默认打分权重前缀 3、队列 2、KV 2、LoRA 1；`/metrics` 每 50 ms 抓一次；`defaultRequestTTL` 60 s；429 的 `rejected-*` 没消耗 GPU 可重试。
- `endpointPickerRef.failureMode` 默认 `FailClose`（EPP 挂了流量全断）；灰度池间用 `HTTPRoute` 权重会打散前缀缓存，池内 `InferenceModelRewrite` 只对 LoRA 版本成立。

**常见误解**："给 A 更高 priority 就实现了 3:1"——那是严格优先，不是 3:1；比例只能在外层配额上落地。另一个："`sessionAffinity: ClientIP` 能代替前缀亲和"——亲和的键应该是请求内容的前缀哈希，一个 NAT 后面的几千个用户会全落到一个副本。

### 8. 第八篇：可观测、成本与 FinOps

**核心问题**：一个 64 卡集群上月账单 X 元，DCGM 显示平均分配率 85%、平均 `SM_ACTIVE` 35%。这 50 个百分点的差距分别来自哪里？每一项对应本系列哪一篇的机制？

**结论**：指标分四层——硬件（DCGM）、容器 Pod（kube-state-metrics、Kueue）、引擎（`vllm:*`、训练约定集）、请求（EPP 的 `llm_d_epp_*`，按 `fairness_id`）——任何一层替代不了另一层，四层靠 `pod` / `namespace` join，DCGM 的 `--kubernetes` 映射（查 kubelet pod-resources）是链条的关键一环。`DCGM_FI_DEV_GPU_UTIL` 只表示"有至少一个 kernel 在跑"，一个 SM 与 132 个 SM、NCCL 自旋等待都是 100%；`SM_ACTIVE`（默认 CSV 里被注释）才接近负载，decode 要用 `DRAM_ACTIVE` 复核。三个数字：分配率 A、使用率 U、有效利用率 $$E \approx A \times U$$；`1 − A` 是没给出去的，`A − E` 是给出去没算的。成本 = 单价 × 分配时长，**按分配计费**——只有这样闲置才会出现在某个团队的账上；使用率作展示列施压。每百万 token 成本三个因子里平台只能动 U。FinOps 是回路：队列使用率 → 改配额（第三篇），`FB_USED` / `SM_ACTIVE` 双低 → 改切分（第四篇），日夜周期 → 改 KEDA（第六篇），按月闭环。文末把八篇归为一个最小平台、八张四栏表、一组数字，以及引擎线、机制线、取舍线三条线的终点。

**必记**：

- 85% / 35% → U ≈ 41%；50 个点：推理低峰 ~15、开发环境 ~10、通信等待 ~10、排队占位 ~5、checkpoint ~3、冷启动 ~2、测量上限 ~5（不可回收）；`1 − A` 的 15 个点：碎片 ~6、坏节点 ~3、配额过紧 ~6。
- 训练 MFU 40% 对应 SM 忙 40–50%，`SM_ACTIVE` 长期 40–50% 是训练的正常值；开发环境是分配率 100%、`SM_ACTIVE` 接近 0。
- $$C_{1M} = \frac{P_{GPU} \times N_{GPU}}{T \times 3600 \times U} \times 10^6$$：2.5 美元/卡时、TP=4、2000 token/s → 满载 1.39 美元，U = 40% 时 3.47 美元。
- OpenCost v1.121.1：`GPUHours = request × hours`，使用率用 `GR_ENGINE_ACTIVE` 而非 `SM_ACTIVE`，`gpuIdleCost = 资产 − 分配`（只算 `1 − A`）；无队列、无 token、MIG 资源名不匹配。
- DCGM Exporter 默认 30 秒采样；成本核算要 13 个月保留；`fairness_id` 超过 1000 折叠为 `other`；共卡时 DCGM 不能按进程拆活跃度，MIG 的 `GPU_I_ID` 归因准确。

**常见误解**："GPU 利用率 78% 说明集群用得不错"——`GPU_UTIL` 把通信等待报成 100%，换 `SM_ACTIVE` 只有 35%。另一个："按使用率计费更公平"——它让闲置成为没人负责的公共成本；按分配计费才让 dev 团队看到 4% 的 U 对着 12,500 美元的账单。

## 三、贯穿全系列的几条线

### 1. 引擎的需求推出平台的机制

这是总纲的第一句话，也是八篇共同的骨架。第一篇把它写成两张表：`torchrun` 的 rendezvous 等不齐就超时 → 第三篇的 gang；NCCL 要直接打开 `/dev/infiniband` 并注册显存 → 第五篇的 Multus 与 RDMA device plugin；vLLM 的 KV cache 满了请求就排队、负载信号只在 `/metrics` 上 → 第六篇按 `num_requests_running` 扩缩、第七篇按 `kv_cache_usage_perc` 与前缀选副本、第八篇把 `vllm:*` 作为四层指标里引擎那一层；一个 7B 服务只用整卡的三分之一 → 第四篇的切分；每半小时写一次几百 GB → 第五篇按突发写带宽选存储。

这条线在第七、八篇有了回头的方向：平台对引擎提出"暴露哪些指标"的需求。GIE 的 model server protocol 要求引擎给出 `TotalQueuedRequests`、`TotalRunningRequests`、`KVCacheUtilization` 三个 gauge 与 OpenAI 接口，满足它就能被 EPP 正确路由；第八篇要求训练框架暴露每 rank 的 step 时间、checkpoint 耗时等约定指标，否则 straggler 的根因查不到硬件层。第一篇说"每篇的引擎的需求一节也是引擎开发者的接口清单"，到第八篇这份清单闭合成双向的接口。

### 2. "GPU 是不透明整数"：一个洞被填了四次

两张需求表唯一共同的空缺，贯穿了资源层四篇与第八篇。第二篇给出它的零值形态：`allocatable` 里没有 `nvidia.com/gpu` 与等于 0 是同一回事；device plugin 填上一个整数，但只计数、无属性、不跨 Pod 共享，DRA 用 `ResourceSlice` 与 CEL 让这个整数重新有属性。第三篇的调度器仍只对这个整数做减法，`Fits()` 不知道 8 张卡要在同一台机器，拓扑要靠 HyperNode 或 TAS 的标签体系外加。第四篇把同一个洞变成技巧：所有共享方案在 K8s 侧都是 device plugin 把一张卡复制成 N 个逻辑设备，隔离强度由底座决定；HAMi 的 `gpumem` 是对"不能表达 20 GB"的正面回答。第五篇把技巧用到网卡上：`rdma/<name>` 也是一个 device plugin 上报的整数，同样没有 GPU 亲和，所以 `NCCL_NET_GDR_LEVEL` 会在 GPU 与网卡跨 PCIe root complex 时把 GDR 关掉。第八篇要把这个整数还原成"哪张卡在哪段时间归谁"：DCGM 的 `--kubernetes` 映射经 pod-resources API 把 UUID 对到 Pod，共卡时归因不准、MIG 的 `GPU_I_ID` 准确——切分方式（第四篇）直接决定账单（第八篇）的粒度。

### 3. checkpoint：一个数字在四篇里的四种用法

第一篇把它列为训练的四条资源需求之一：1 TB 级在 1 分钟内落盘，被打断的代价是上次 checkpoint 之后的进度。第三篇用它定抢占策略：代价 $$N_{gpu} \times (T_{since\_ckpt} + T_{restart})$$，抢占以整任务为单位，`terminationGracePeriodSeconds` 至少覆盖一次 checkpoint 写入，承诺不被抢的队列可以拉长间隔。第五篇把"一次写入要多久"算出来：14 字节/参数，70B ≈ 1 TB，同步 1 分钟写完要 16.7 GB/s，异步只要 0.56 GB/s 但每节点约 125 GB pinned 内存——第三篇的 grace period 该填多少秒，答案在这里；DCP 恢复时 rank 数可变，又是第三篇弹性与抢占的前提。第八篇把它变成看板上的一条竖条与账单上的 ~3 个点：竖条宽度 × 频率 > 5% 时间就是一个"改异步、扩带宽、拉长间隔"的 PR。四篇说的是同一件事的四面：需求、策略、算术、度量。

### 4. "8 分钟"：扩容时间的分解与压缩

第一篇给出量级：一个 70B 副本从决定扩容到能接流量 5–10 分钟，主项是权重加载，所以扩缩容必须提前、缩零很贵。第二篇负责其中一段：10 GB 镜像冷拉 2–3 分钟，几十个 Pod 同时拉时十分钟也不罕见，对策是统一基底、多阶段构建、预热、P2P。第五篇负责另一段：140 GB 权重从对象存储 1 GB/s 要两分多钟，本地 NVMe 缓存、权重进镜像、P2P 各有适用场景。第六篇把五段列成表并算出后果：调度 + 拉镜像 + 拉权重 + 加载到显存 + 预热 ≈ 8 分钟，再加约 1 分钟信号延迟，倒推出阈值 36、峰值过配到 10 副本、cron 提前 12 分钟；前三段是平台能优化的，后两段是引擎的。第八篇把它记成冷启动成本：(就绪 − 调度) × 卡数 × 单价 × 次数，在 50 个点里占约 2 个点，并把"缩零省的常驻成本"与"每次冷启动的代价"的比值作为该不该缩零的判据。

### 5. 隔离 vs 利用率：同一个取舍出现在四层

这是总纲取舍线的第一项。第三篇：gang 让集群更空——32 卡任务凑齐之前空出来的 20 张卡不能给别的大任务，`StrictFIFO` 与 `BestEffortFIFO` 是大任务饥饿与小任务插队的二选一；拓扑 `hard` 约束换可预期的带宽，代价是等几小时。第四篇是它最直接的形态：从时间片到 MIG 隔离越强、总利用率越低，`3g.20gb × 2` 浪费一个计算 slice 且只能放两个服务，HAMi 总吞吐更高但故障域整卡。第五篇：rdma-shared-dev-plugin 多 Pod 共享一个 HCA 无隔离，SR-IOV 一 VF 一 Pod 隔离强但 VF 数受固件限制。第七篇把它搬到网关层：两个租户共用 4 个副本时租户隔离在时间（排队顺序）上而不在空间上；要 B 的突发不影响 A 的 TTFT 就得拆成两个池，代价是 A 空闲时 B 用不上那 3 个副本。第八篇给了它一个统一的单位：每一段隔离付出的利用率都出现在 `1 − A`（MIG 几何浪费、碎片）或 `A − E`（低峰空转）里，并被换算成钱。

| 概念 | 出现的篇 | 关系 |
|---|---|---|
| 引擎的需求 → 平台的机制 | 一至八 | 一列清单；三、五满足训练；四、六、七满足推理；七、八反过来对引擎提出指标接口 |
| `nvidia.com/gpu` 不透明整数 | 一、二、三、四、五、八 | 一发现；二填计数、DRA 加属性；三只做减法；四复制成 N 份；五用同一技巧给网卡；八还原成归属 |
| checkpoint | 一、三、五、八 | 一需求；三抢占代价与 grace period；五 14 字节/参数与带宽算术；八竖条与 ~3 个点 |
| 扩容时间 8 分钟 | 一、二、五、六、八 | 一量级；二镜像段；五权重段；六五段分解与阈值；八冷启动成本 |
| `vllm:*` 引擎指标 | 一、六、七、八 | 一列为需求；六扩缩容信号；七路由与 model server protocol；八四层指标的引擎层 |
| 隔离 vs 利用率 | 三、四、五、七、八 | 三 gang 与拓扑；四四档共享；五 shared 与 SR-IOV；七两个池；八换成 `1 − A` 与 `A − E` |
| 三个数字 A / U / E | 三、四、六、八 | 八定义；三管 `1 − A` 的碎片与配额；四管低峰切分与开发环境；六管推理低峰空转 |

## 四、常见误区

| 误区 | 为什么错 | 正确的说法 | 出处 |
|---|---|---|---|
| `Insufficient nvidia.com/gpu` 说明卡不够 | 调度器把"键不存在"与"等于 0"当同一回事 | 没有组件把 GPU 上报给 kubelet；装 device plugin | [第一篇](/ai-platform-engine-requirements-and-architecture.html) |
| `nvidia-smi` 与 `torch.version.cuda` 的版本不一致就有问题 | 前者是驱动支持的上限，后者是 Toolkit 版本 | 新驱动跑旧 Toolkit 是规则一，无条件成立 | [第二篇](/gpu-in-containers-driver-cuda-device-plugin.html) |
| 旧驱动跑新 CUDA 装个 cuda-compat 包就行 | forward compat 仅数据中心 GPU 与受支持驱动分支 | 同大版本内靠 minor version（12.x ≥ 525.60.13）；消费级卡跨大版本只能升驱动 | [第二篇](/gpu-in-containers-driver-cuda-device-plugin.html) |
| Pod 起不来靠应用层重试解决 gang | 已起的 Pod 占着卡等同伴，重试只是反复占卡、释放、再占 | 要么全起要么都不起只能在调度 / 准入层做 | [第三篇](/ai-job-scheduling-gang-queue-topology.html) |
| 配了 cohort 借用，借出去的卡自然会收回 | Kueue 默认 `reclaimWithinCohort: Never`，Volcano 默认 actions 不含 `reclaim` | 默认值下两者都借了不还；要显式配回收，Volcano 还要 `gangreclaim` 绕过 gang 保护 | [第三篇](/ai-job-scheduling-gang-queue-topology.html) |
| MIG `3g.20gb × 2` 还能塞一个 `1g.5gb` | 两个 `3g.20gb` 已用尽 8 个显存 slice | 只能放两个服务；浪费的是 1 个计算 slice | [第四篇](/gpu-sharing-and-partitioning-mig-mps-hami.html) |
| HAMi 有显存配额，所以共卡服务互不影响 | 它拦截 `cudaMalloc`，不切带宽与 L2，故障域仍是整卡 | OOM 被限制在越界容器内，Xid 三个全挂；要故障隔离用 MIG | [第四篇](/gpu-sharing-and-partitioning-mig-mps-hami.html) |
| 多机训练慢但没报错，不是网络问题 | NCCL 找不到 RDMA 设备就静默回落 Socket | `NCCL_DEBUG=INFO` 看 `Using network IB/Socket` 与 `/GDRDMA` | [第五篇](/rdma-networking-storage-and-checkpoint-io.html) |
| checkpoint 写不完只能买更快的存储 | 同步 1 分钟写完要 16.7 GB/s，异步只要 0.56 GB/s | 先改形态：分片 + `dcp.async_save`，代价是每节点约 125 GB pinned 内存 | [第五篇](/rdma-networking-storage-and-checkpoint-io.html) |
| 推理服务的扩缩容阈值调准了就不会过配 | 为 9 分钟就绪预留的 headroom 在稳态高峰也被保留 | 反应式的结构性代价；可预测的高峰用 cron 提前，阈值抬到接近饱和 | [第六篇](/serving-platforms-kserve-triton-ray-serve-llm-d.html) |
| 给 A 更高 priority 就是 3:1 的配额 | flow control 是严格优先 + 同级 round-robin，无加权公平 | 比例只能在外层 TPM / 并发桶上体现；饱和时 B 先撞 429 | [第七篇](/model-gateway-multi-tenancy-and-quota.html) |
| DCGM 报 GPU 利用率 78%，集群用得不错 | `GPU_UTIL` 只表示有 kernel 在跑，NCCL 自旋也是 100% | 看 `SM_ACTIVE`；三个数字 A、U、E 分开算；按分配而非使用率计费 | [第八篇](/ai-platform-observability-cost-and-finops.html) |

## 五、通关自测

### A. 判断与计算（10 题）

1. 宿主机驱动 560（原生 CUDA 12.6），镜像里是 CUDA 12.8 编译的 PyTorch——能跑吗？换成 CUDA 13.0 的镜像、消费级显卡呢？

   <details markdown="1"><summary>答案</summary>

   12.8：同大版本、560 ≥ 525.60.13，minor version compatibility 成立，能跑（调用 12.8 新增驱动 API 处返回 36、需要 PTX JIT 处报 222 除外）；13.0：跨大版本，消费级卡没有 forward compat，容器启动报 `unsatisfied condition: cuda>=13.0` 或运行时 35。

   </details>

2. 一个 128 卡任务每 60 分钟存一次 checkpoint、重启要 10 分钟，被抢占一次平均浪费多少 GPU 小时？若 grace period 覆盖一次 checkpoint 写入呢？

   <details markdown="1"><summary>答案</summary>

   $$128 \times (30 + 10)$$ 分钟 ≈ 85 GPU 小时；grace period 让任务收到 SIGTERM 先存盘，损失压到 $$T_{restart}$$：128 × 10 分钟 ≈ 21 GPU 小时。

   </details>

3. A100 40 GB 切成 `1g.5gb × 7`，用了几个计算 slice、几个显存 slice，浪费什么？一个 7B FP16 服务能放进去吗？

   <details markdown="1"><summary>答案</summary>

   7 个计算 slice 全部用尽、7 个显存 slice，浪费 1 个显存 slice——与 `3g.20gb × 2` 浪费 1 个计算 slice 正好相反；7B FP16 权重 14 GB 放不进 5 GB，至少要 `3g.20gb`，INT4 约 4 GB 可勉强进 `1g.5gb` 但 KV 几乎没空间。

   </details>

4. 一个 13B 模型、16 卡 2 节点训练，每 20 分钟同步 checkpoint 一次、要求 1 分钟写完：状态多大、聚合写带宽多少？改异步后带宽多少、每节点要多少 pinned 内存？

   <details markdown="1"><summary>答案</summary>

   13B × 14 字节 ≈ 182 GB；同步 182 GB / 60 s ≈ 3.0 GB/s 聚合、每节点 1.5 GB/s；异步只需 182 GB / 20 min ≈ 0.15 GB/s，代价每 rank 182 / 16 ≈ 11.4 GB、每节点 8 rank ≈ 91 GB pinned host 内存——小集群跑大模型时 host 内存最吃紧。

   </details>

5. 第六篇的服务改成常驻 3 副本，饱和点 64、爬升 6.25 并发/分钟、就绪 + 信号 9 分钟，`num_requests_running` 的每副本阈值取多少？峰值 350 并发时 HPA 会算出几个副本？

   <details markdown="1"><summary>答案</summary>

   $$3 \times 64 - D \ge 6.25 \times 9 \approx 56$$ → D ≤ 136 → 每副本 ≤ 45（饱和点的 71%，比 2 副本时的 36 高，余量摊在更多副本上）；峰值 ceil(350 / 45) = 8 副本、32 张卡，仍比需要的 6 副本过配。

   </details>

6. H100 2.5 美元/卡时，一个 TP=2 的服务满载输出 1500 token/s，平均负载是满载的 50%——每百万输出 token 成本多少？满载时呢？

   <details markdown="1"><summary>答案</summary>

   $$C_{1M} = \frac{2.5 \times 2}{1500 \times 3600 \times 0.5} \times 10^6 \approx 1.85$$ 美元；满载 U = 1 时约 0.93 美元。平台能动的只有 U。

   </details>

7. 集群分配率 90%、全部卡的 `SM_ACTIVE` 平均 45%：已分配卡上的使用率 U 是多少？"没给出去"与"给出去没算"各多少个点？

   <details markdown="1"><summary>答案</summary>

   $$U = E / A = 45 / 90 = 50\%$$；`1 − A` = 10 个点没给出去（碎片、坏节点、配额过紧），`A − E` = 45 个点给出去没算（低峰空转、开发环境、通信等待、排队占位、checkpoint、冷启动、测量上限）。

   </details>

8. 租户 TPM 余量 10,000，一个请求输入估算 2,000 token、`max_completion_tokens` 8192、单请求上限 4096：预扣多少？结束时 `usage` 为 2,100 + 900，退还多少、余量回到多少？

   <details markdown="1"><summary>答案</summary>

   预扣 = 2,000 + min(8192, 4096) = 6,096，余量 3,904；实际 3,000，退还 3,096，余量 7,000。若是流式请求且网关没强制 `stream_options.include_usage`，就拿不到 `usage`，一个 token 都记不到。

   </details>

9. 4 个副本各能放 160k token 的 KV，负载变成 96 个会话 × 8k 上下文、prefill 16k token/s：前缀亲和下每副本要放多少 token，放得下吗？未命中一轮的 prefill 多久？

   <details markdown="1"><summary>答案</summary>

   96 × 8k / 4 = 192k > 160k，放不下，LRU 会淘汰、部分轮次未命中；至少 5 副本（≈ 154k/副本）才全部命中；未命中一轮 8k / 16k = 0.5 s，命中接近 0——这时该扩容（第六篇）而不是换路由策略。

   </details>

10. Kueue 里 A、B 同一 cohort，A `nominalQuota` 16、`borrowingLimit` 8，B `nominalQuota` 16、`lendingLimit` 16。A 提交 24 卡任务与 32 卡任务各会怎样？Volcano 里等价的配置是什么？

    <details markdown="1"><summary>答案</summary>

    24 卡：16 + 借 8 = 24 → admitted、unsuspend；32 卡：32 > 16 + 8 → Pending，reason `ExceedsMaxQuota`。Volcano：Queue A `deserved: 16, capability: 24`，24 卡任务 `enqueue` 通过、`allocate` 用到 B 的空闲即运行，32 卡任务 `minResources` 超过 `realCapability` 永远 Pending。

    </details>

### B. 跨篇综合（5 题）

1. 一个 32 卡 `TrainJob` 提交后 Kueue 显示 `Admitted True`，Pod 却全部 Pending，事件是 `Insufficient nvidia.com/gpu`。可能在哪几层？各怎么查？

   <details markdown="1"><summary>答案</summary>

   第三篇：Kueue 是配额级 gang，准入 ≠ 节点放得下——碎片、taint、非 Kueue 管理的 Pod 占了卡，长期解用 TAS 或 `waitForPodsReady`；第二篇：某些节点 device plugin 没上报（驱动容器与内核不匹配、Toolkit 模式与插件 `deviceListStrategy` 不一致），查节点 `allocatable`；第四篇：节点池切成了 MIG `mixed`，资源名是 `nvidia.com/mig-<profile>` 而不是 `nvidia.com/gpu`，或 HAMi 替换了官方插件——配额加法与资源名要一致，一个节点池一种策略。

   </details>

2. 三个小模型用 HAMi 共一张 80 GB 卡，月底要按团队分账、按 SLA 决定是否迁 MIG。哪几篇的结论一起用？

   <details markdown="1"><summary>答案</summary>

   第四篇：HAMi 隔离显存不隔离带宽与 L2、故障域整卡，有 SLA 的服务才值得迁 MIG（要数据中心卡且模型大小与 profile 匹配）；第八篇：共卡时 DCGM 不能按进程拆 `SM_ACTIVE`，只能按份额平摊或按 `FB_USED` 比例估，MIG 实例有独立 `GPU_I_ID` 归因准确——这是 MIG 在多租户计费上更受欢迎的原因；账单按 `gpumem` 份额而非按卡算；第三篇：切分后队列配额要按 profile 或显存 MB 设，`ResourceFlavor` 按节点池标签把共享池与独占池分开。

   </details>

3. Kueue 开了 `reclaimWithinCohort: Any`，训练任务每 30 分钟同步 `dcp.save` 一次、64 卡 8 节点、状态 1 TB、存储聚合写 16.7 GB/s。`terminationGracePeriodSeconds` 至少多少？改成 `dcp.async_save` 后呢？看板上怎么看它有没有生效？

   <details markdown="1"><summary>答案</summary>

   第三篇：grace period 要覆盖一次 checkpoint 写入 + 余量，否则被抢占的损失翻倍；第五篇：同步写 1 TB / 16.7 GB/s = 60 s，所以至少 60 s 加余量；异步只需等 D2H stage（每 rank 15.6 GB，秒级）就能退出，但 host 内存里的副本随节点一起消失——异步保护的是训练不停顿，不是抢占时的最新一份，grace period 仍要留到 upload 完成或接受回到上一份；第八篇：checkpoint 竖条宽度 × 频率 > 5% 时间就该改异步，`kueue_admission_wait_time_seconds` × 卡数是排队成本。

   </details>

4. 要把一个 70B TP=4 副本的 8 分钟就绪压到 3 分钟以内，前几篇各能贡献哪一段？剩下的谁负责？

   <details markdown="1"><summary>答案</summary>

   第二篇：10 GB 镜像冷拉 2–3 分钟 → 节点预拉、多阶段构建瘦身、P2P，压到 0；第五篇：140 GB 权重从对象存储 1 GB/s 2–3 分钟 → 本地 NVMe 缓存或并行文件系统（5 GB/s 30 秒）；第六篇：`pvc://` 直挂、`LocalModelCache`、`oci://` modelcar 把拉权重拿掉，持久化 compile cache 缩短预热，llm-d 的 Fast Model Actuation 用 sleep / wake 做秒级 hot start；剩下加载到显存 1–3 分钟与 CUDA graph 捕获是引擎侧的；第八篇：冷启动成本 = (就绪 − 调度) × 卡数 × 单价 × 次数，压短它同时改变缩零的判据。

   </details>

5. 租户 B 抱怨 TTFT p95 超 SLO。怎么用第七、八篇的指标判断是 B 自己超配额、池整体饱和、还是路由问题？各改哪篇的旋钮？

   <details markdown="1"><summary>答案</summary>

   第八篇：以 EPP 侧为准，`llm_d_epp_request_ttft_seconds{fairness_id="B"}` 是用户看到的延迟；第七篇：看 429 的 `x-llm-d-request-dropped-reason`——`rejected-saturated` 是池饱和、外层 TPM 429 是 B 超配额；池饱和时对比各副本 `vllm:kv_cache_usage_perc` 是否长期不齐——不齐是路由问题（`utilization-filter` 阈值、前缀亲和把太多会话压到一个副本），齐且都高是容量问题；容量问题改第六篇的饱和点与 KEDA 阈值或加副本，B 的优先级改 `InferenceObjective` 或拆独立 `InferencePool`（第七篇），B 超配额则是配额设计本身。

   </details>

### C. 面试题（7 题）

1. 让你为三个团队共用的 64 卡集群设计训练调度与配额，你会选 Kueue 还是 Volcano？借用、回收、拓扑各怎么配？

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 先说介入时机——Volcano 换调度器在 Pod 之后做节点级 gang，Kueue 不换调度器在 Pod 之前做配额级 gang；混合集群倾向 Kueue，纯训练且要 binpack / NUMA 倾向 Volcano；(2) 配额不是硬上限：Kueue 每队列 `nominalQuota` + cohort，生产队列 `lendingLimit` 小、实验队列全开，`reclaimWithinCohort: Any`；Volcano 用 `guarantee ≤ deserved ≤ capability` 并显式加 `reclaim` / `gangreclaim`；(3) 默认值陷阱：两者默认借了不还；(4) 拓扑：TP 组 `podset-required-topology` 同机，DP / PP 组 `preferred` 同 rack，标签来源必须正确；(5) 抢占以 Workload 为单位，`terminationGracePeriodSeconds` 覆盖一次 checkpoint，训练 Pod 统一 `PriorityClass`。
   **追问方向**：Kueue 准入后节点放不下怎么办（TAS / `waitForPodsReady`）；`StrictFIFO` vs `BestEffortFIFO`；为什么 K8s 生态没有真 backfill。
   **好答案与一般答案的区别**：一般答案比功能列表；好答案说出两者介入时机的差别、三个系统不配置时的默认行为，并把 checkpoint 间隔与抢占策略绑在一起。

   </details>

2. 一张 80 GB 的 H100 上跑几个小模型的推理服务，MIG、HAMi、时间片怎么选？给出判据与代价。

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 训练或需要 NCCL 一律不切；(2) 三条隔离各做到什么程度——时间片无隔离、OOM 蔓延；HAMi 显存与算力软隔离、不切带宽与 L2、故障域整卡；MIG 硬件分区连 Xid 都隔离；(3) 有 SLA 用 MIG，但几何是枚举——A100 7 个计算 slice、8 个显存 slice，`3g.20gb × 2` 只能放两个，改几何要清空 GPU；`3g.20gb` 是 3/7 算力 + 4/8 带宽，对 decode 密集更划算；(4) 模型大小参差、要弹性用 HAMi，接受故障域整卡；(5) 开发 / notebook 用时间片；一个节点池一种策略，用 `ResourceFlavor` 分派；(6) 切分前先用 `SM_ACTIVE` 与 `FB_USED` 确认负载互补，账单按 GI 或 `gpumem` 算。
   **追问方向**：vLLM 的 `--gpu-memory-utilization` 在各方案下怎么设；MPS 什么时候比 HAMi 合适；HAMi 与官方 device plugin 为什么互斥。
   **好答案与一般答案的区别**：一般答案说"MIG 隔离好、HAMi 灵活"；好答案把三种隔离拆开、给出几何的具体数字，并说明切分决定了计费粒度。

   </details>

3. 多机训练在 K8s 上比裸机慢两倍多、没有任何报错，你的排查顺序是什么？

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 先假设 NCCL 回落 Socket 或 GDR 没开——引擎自己降级不报错；(2) `NCCL_DEBUG=INFO` 看 `Using network IB` 还是 `Socket`、连接行有没有 `/GDRDMA`、有没有 `GPU Direct RDMA Disabled ... distance > 3`；(3) 容器里 `ls /dev/infiniband`、`ibv_devinfo`、`ulimit -l`、`ib_write_bw --use_cuda` 两 Pod 间测裸带宽，再到 nccl-tests；(4) 三层归因：Pod 网络（Multus 注解、NAD 的 `resourceName`、`NCCL_SOCKET_IFNAME`）、device plugin（没请求 `rdma/<name>`、selectors、netns 模式、`IPC_LOCK`、peermem / DMA-BUF）、NCCL 环境变量（`NCCL_IB_HCA`、RoCE 的 `GID_INDEX`、`NET_GDR_LEVEL`）；(5) GPU 与网卡分配没有 PCIe 亲和是容器场景特有的坑，修分配而不是放宽 `GDR_LEVEL`；(6) 网络没问题再看拓扑（第三篇是否跨 spine）与 checkpoint 停顿（第五篇的竖条）。
   **追问方向**：shared 与 SR-IOV plugin 怎么选；Network Operator 与 GPU Operator 的部署顺序；为什么 NetworkPolicy 管不到 RDMA 流量。
   **好答案与一般答案的区别**：一般答案说"检查网络配置"；好答案先从日志行判定回落到了哪一层，再按三层逐项对照，并知道每一项不满足时的具体症状。

   </details>

4. 为一个 70B TP=4 的推理服务设计扩缩容：指标、阈值、提前量、缩容、缩零，并说明每个数字怎么来。

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 对象是 LeaderWorkerSet 的 `/scale`，KEDA `ScaledObject` 用 `AverageValue` 让阈值是每副本的；(2) CPU 与 `GPU_UTIL` 无意义，领先用 `vllm:num_requests_running` 或 `kv_cache_usage_perc`，`waiting` 只做兜底；(3) 先压测出饱和点 C 与历史曲线的爬升速率 r，没有这两个数任何阈值都是猜的；(4) 反应式阈值由 $$N C - D \ge r (T_{signal} + T_{ready})$$ 倒推，例子里 36（56%），代价是峰值过配到 10 副本；(5) 可预测的高峰用 cron 提前 $$T_{signal} + T_{ready}$$ + 余量（约 12 分钟）抬地板，反应式阈值抬到接近饱和（56）只管意外流量；(6) 缩容 15 分钟稳定窗口 + 每 5 分钟 1 个，因为缩错的代价是 8 分钟；(7) 70B 不缩零，`minReplicaCount` 2；(8) 同时压 8 分钟本身：`pvc://` / `LocalModelCache`、镜像预热。
   **追问方向**：PD 分离两侧各按什么指标扩；KV 阈值与并发阈值的换算依赖什么；`startupProbe` 怎么配才不会杀掉还在加载的容器。
   **好答案与一般答案的区别**：一般答案说"接 KEDA 按队列长度扩"；好答案给出阈值公式、说清领先阈值等于峰值过配这个结构性代价，并用 cron 与慢缩容绕开它。

   </details>

5. 为什么 LLM 服务不能用 Service 轮询？模型网关的"配额"应该是什么量纲？两个租户 3:1 怎么落地？

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 轮询不看 KV 满不满——请求一旦送到副本就迁不走；不看缓存在哪——64 会话 × 8k 的例子里 TTFT 差一个数量级；(2) `InferencePool` 替代 Service 作 `HTTPRoute` 后端，EPP 经 ext_proc 每请求选副本：前缀亲和 → `utilization-filter` → 加权分（前缀 3、队列 2、KV 2、LoRA 1）；(3) 配额量纲：请求数错（500 与 30k token 差 60 倍）、GPU 时间到达时不可知，对租户暴露 token（输入 / 输出 / 命中分开）+ 并发 + RPM；(4) 预扣 = 输入估算 + min(`max_completion_tokens`, 上限)，按 `usage` 结算，流式强制 `include_usage`；(5) 3:1 只能在外层 TPM / 并发桶上体现，内层是严格优先 + 同级 round-robin；(6) 要空间隔离就拆两个池，代价是隔离 vs 利用率。
   **追问方向**：近似与精确前缀索引的差别；`FailClose` 与 `FailOpen`；灰度为什么会打散缓存。
   **好答案与一般答案的区别**：一般答案说"用智能路由、按 token 限流"；好答案把选副本的流水线、配额的三种量纲与预扣结算的时序说具体，并指出 3:1 在开源栈里落在哪一层。

   </details>

6. 集群分配率 85%、`SM_ACTIVE` 35%，老板问钱花到哪去了、下个月怎么改。你怎么回答？

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 先纠正尺度：`GPU_UTIL` 把通信等待报成 100%，`SM_ACTIVE` 才接近负载，且它低估 memory-bound 的 decode；(2) 三个数字：$$E \approx A \times U$$，U ≈ 41%，`1 − A` 15 个点是没给出去的，`A − E` 50 个点是给出去没算的；(3) 分解并归篇：推理低峰 ~15（第六篇）、dev ~10（第四篇时间片 + 第三篇空闲回收）、通信等待 ~10（第五篇 RDMA 是否生效，其余是引擎的）、排队占位 ~5（第三篇）、checkpoint ~3（第五篇）、冷启动 ~2、测量上限 ~5 不可回收；碎片 ~6、坏节点 ~3、配额过紧 ~6；(4) 按分配计费让每一项有主，闲置作展示列，dev 的 4% U 对着 12,500 美元最有说服力；(5) 回路：月初账单与分解表 → 改配额 / 切分 / KEDA 参数 → 月末同一组 recording rules 复核；(6) 训练与推理错峰是 E 的日夜周期能否填平的关键。
   **追问方向**：DCGM 的 Pod 映射怎么来、共卡时归因误差；OpenCost 缺什么；每百万 token 成本怎么算、为什么与 API 只比量级。
   **好答案与一般答案的区别**：一般答案报一个利用率；好答案把差距拆成可测量、可归篇、有 owner 的项，并把"不可回收"的部分明确标出来。

   </details>

7. Pod 调度成功但 `torch.cuda.is_available()` 返回 `False`，或容器起不来报 `unsatisfied condition: cuda>=13.1`。这条链上有哪几层，怎么定位？为什么很多集群把驱动固定在 LTSB 分支？

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 四层栈：内核驱动与 `libcuda.so` 在宿主机同版本，CUDA Runtime 随 wheel / 镜像，上层库随框架——四个角色互不知情，只靠三条兼容规则维系；(2) 先对 `nvidia-smi` 的 CUDA Version（驱动上限）与 `torch.version.cuda`（Toolkit）：同大版本查 minor version 基线，跨大版本看是否数据中心卡 + cuda-compat；(3) `unsatisfied condition` 是 Container Toolkit 读 `NVIDIA_REQUIRE_CUDA` 把检查前移到了容器创建；(4) 容器内没设备则查 Toolkit 模式（legacy / cdi / jit-cdi）与 device plugin `deviceListStrategy` 是否一致——"插件给了环境变量、运行时只认 CDI 注解"是典型组合；(5) 用 GFD 的 `cuda.driver.major` 标签把 CUDA 13 镜像只调度到驱动 ≥ 580 的节点；(6) 驱动升级要卸载内核模块、节点上所有 GPU 进程先停，对长训练是一次 checkpoint 恢复——所以固定 LTSB，靠 minor version 与 forward compat 消化应用侧升级。
   **追问方向**：为什么 PyTorch wheel 带 cudart 不带 libcuda；DRA 与 device plugin 双轨的重复计数；镜像体积对扩容时间的影响。
   **好答案与一般答案的区别**：一般答案说"驱动版本不匹配，升驱动"；好答案说出四层各归谁管、三条规则各适用于哪个组合，并解释为什么生产上是应用侧适配驱动而不是反过来。

   </details>

### D. 掌握判据

| 水平 | 表现 |
|---|---|
| 读过 | 能说出八篇各讲什么；知道 gang、cohort、MIG profile、Multus、LeaderWorkerSet、InferencePool、EPP、`SM_ACTIVE` 这些名词 |
| 掌握 | A 组能不翻书算出 8 题以上；B 组能说出每题用了哪几篇的什么；面对一个 Pending、一次变慢、一份账单，能说出它卡在哪一层、查哪个指标、改哪篇的旋钮 |
| 能教人 | C 组每题能给出全部要点并预判追问；能解释八篇里每个反直觉结论（默认值下借了不还、1/7 的实例不是 1/7 的吞吐、领先阈值等于峰值过配、按分配计费才公平）为什么成立，并能说出每个机制的代价 |

通关标准：A 组至少 8 题、B 组至少 4 题、C 组每题能说出一半以上要点。没过的部分回到第二章对应篇的"必记"，再回该篇正文的"核心问题"章。

## 六、下一步

八篇讨论的是引擎之下的资源层与引擎之侧的交付层，总纲划出的边界之外有这些方向：

- **引擎内部的机制**——训练框架的并行策略与 checkpoint 的格式，在[《大规模训练工程：从并行策略到容错恢复》](/large-scale-training-from-parallelism-to-fault-tolerance.html)；推理引擎的请求调度与 KV cache 管理，在[《大模型推理系统揭秘：从 vLLM 看 LLM Serving Infra 核心技术》](/deep-dive-into-vllm.html)。本系列把它们当作已知的需求来源。
- **通信与网络协议的内部**——NCCL 的算法、RDMA verbs、IB 与 RoCE 的差别，在[《通信与互联：从 NCCL 到 RDMA》](/communication-and-interconnect-for-ai-infra.html)；第五篇只讨论如何让 NCCL 在容器里选到 RDMA、如何从日志确认它选了。
- **GPU 硬件与 kernel**——SM、显存层次、kernel 性能分析，在[《GPU Kernel 工程：从 CUDA 执行模型到 FlashAttention》](/gpu-kernel-engineering.html)；第四篇只用到"SM 与显存带宽被分区"这个事实。
- **本系列在整张路径上的位置**（L5）见[《AI-Infra 工程师学习地图》](/ai-infra-learning-roadmap.html)；网关之上的应用层属于[《AI 应用工程师学习地图》](/ai-application-engineer-learning-roadmap.html)；三张地图如何拼在一起见[《AI 全栈学习地图》](/ai-fullstack-learning-roadmap.html)。

回到总纲：[《AI 平台工程：资源层与交付层》](/ai-platform-engineering.html)。

[^q0]: 八个：训练框架与推理引擎各对平台要什么、原生 Kubernetes 缺什么（两张 15 条的需求表、三种空缺、两层拆分）；一个容器怎样用上 GPU、版本契约怎么查（四层栈、三条规则、device plugin 与 DRA、镜像）；一组 Pod 怎样一起拿到一组 GPU、多团队怎么借与还（gang、Volcano 与 Kueue 的介入时机、借用 / 回收 / 等待的配置、拓扑、抢占代价）；一张卡怎样安全地给多个负载（四档隔离、MIG 几何、HAMi、决策树）；NCCL 为什么在容器里变慢、checkpoint 要多少带宽（Multus、RDMA plugin、GDR、三层排查、14 字节/参数的算术）；一个副本是几个 Pod、按什么扩缩、提前多久（LWS、KServe / llm-d、阈值公式、cron）；请求怎么路由、配额是什么量纲（InferencePool + EPP、token 记账、3:1 落在哪层）；利用率的差距去了哪里、钱怎么分（四层指标、三个数字、50 个点的分解、按分配计费、每百万 token 成本）。详见[第二章](#二逐篇回顾)。
[^q1]: 训练 15 条原生满足 1 条、推理满足 2 条；minor version 基线 12.x ≥ 525.60.13、13.x ≥ 580.65.06，错误码 35 / 36 / 222；`guarantee ≤ deserved ≤ capability`，默认值下 Volcano 与 Kueue 借了不还、Slurm 不借；抢占代价 $$N_{gpu} \times (T_{since\_ckpt} + T_{restart})$$；A100 7 个计算 slice、8 个显存 slice，`3g.20gb` = 3/7 算力 + 4/8 带宽；14 字节/参数 → 70B ≈ 1 TB → 16.7 GB/s 同步、0.56 GB/s 异步、每节点约 125 GB pinned；扩容 8 分钟 + 1 分钟信号，阈值 36 → 峰值 10 副本，cron 提前 12 分钟 ≈ 5 卡时/天 vs 常驻 328；64 会话 × 8k 对 4 × 160k，未命中 0.5 s；打分权重 3 / 2 / 2 / 1；$$E \approx A \times U$$，85% / 35% → 50 个点的分解；$$C_{1M}$$ 从 1.39 到 3.47 美元。详见[第一章](#一总览系列回答的问题与主线)、[第三章](#三贯穿全系列的几条线)。
[^q2]: 用第五章的三段自测：A 组 10 题判断与计算（至少 8 题）、B 组 5 题跨篇综合（至少 4 题）、C 组 7 道面试题（每题说出一半以上要点）；D 组的表给出"读过 / 掌握 / 能教人"三级的表现。详见[第五章](#五通关自测)。
