---
layout: post
title: "AI 平台工程（08）：可观测、成本与 FinOps"
subtitle: "Observability, Cost and FinOps: from DCGM to the Token Bill"
tags: [Kubernetes, GPU, Observability, FinOps, AI, AI-Infra]
catalog: true
---

> 本文是[《AI 平台工程：资源层与交付层》](/ai-platform-engineering.html)系列的第 8 篇（共八篇）。上一篇：[模型网关与多租户：路由、配额与灰度](/model-gateway-multi-tenancy-and-quota.html)。

月初，财务把上个月的 GPU 账单转给平台组：64 张 H100，按云上的小时价折下来是一个七位数。附带一个问题：这些钱花得值不值？平台组打开 Grafana，能给出的数字是两个——DCGM 报的平均"GPU 利用率"78%，节点上 `nvidia.com/gpu` 的平均分配率 85%。看起来不错。但换一个指标，`DCGM_FI_PROF_SM_ACTIVE` 的集群平均只有 35%。三个数字之间差了几十个百分点，而没有一张看板能说清这几十个点分别去了哪里、归哪个团队、对应前七篇里的哪个机制。

这就是本篇要处理的问题。前七篇搭起来的东西——GPU Operator、Kueue、MIG 与 HAMi、RDMA 与 checkpoint 存储、Serving 平台、推理网关——每一层都在做一个取舍：隔离换利用率、排队换碎片、冷启动换常驻成本。取舍做得对不对，只有两种办法知道：看运行状态，算成本。看不见就无法改进，算不清就无法说服任何人改配额、改切分、改扩缩容参数。可观测与 FinOps 是平台的反馈回路，没有它，前七篇的机制只是一次性的配置。

这一层的难点不在于工具——Prometheus、Grafana、OpenTelemetry 都是现成的——而在于 AI 负载把每一个环节都扭曲了。"利用率"这个词在 GPU 上至少有三个互不相等的定义；一张卡的指标要归到一个 Pod、一个队列、一个租户，需要穿过 kubelet 的设备分配记录；推理请求的成本按 token 计而不是按次计，成本模型要从 GPU 小时换算到每百万 token；训练任务的"空转"藏在通信等待里，DCGM 的 `GPU_UTIL` 把它报成 100%。通用的云成本工具对 GPU 的支持也刚起步——OpenCost 到 v1.121.1 才有 `GPUAllocation` 这样的结构，且用的是 `GR_ENGINE_ACTIVE` 而不是 `SM_ACTIVE`。

所以本篇分三段走。先把从硬件到请求的**四层指标**串成一条线，说清每一层从哪采、叫什么名字、哪些名字会骗人；再用这些指标定义**分配率、使用率、有效利用率**三个数，把它们之间的差距分解到具体原因；最后把指标变成**账单**——GPU 小时怎么分摊到团队、每百万 token 的成本怎么算、成本数据怎么回流到配额、切分与扩缩容的参数。作为末篇，最后一章是全系列的总结与目录。

本篇要回答总纲提出的核心问题：

> **一个 64 卡集群上月账单 X 元，DCGM 显示平均分配率 85%、平均 `SM_ACTIVE` 35%。这 50 个百分点的差距分别来自哪里——排队等 gang、训练的通信等待、推理的低峰空转、开发环境的长期占用？每一项对应本系列哪一篇的机制？**

版本以 NVIDIA DCGM Exporter 4.6.0-4.8.3、vLLM v0.23.0（只用指标名）、Kueue v0.19.2、OpenCost v1.121.1、llm-d-router v0.10.0（Endpoint Picker 的指标）、Kubernetes v1.37.0 为准。kube-state-metrics、cAdvisor、Prometheus 与 OpenTelemetry 只用通行的指标名与概念，不引用其源码。GPU 单价与吞吐数字全部是标注为假设的算例，不是任何供应商的报价或实测。本篇不引用其他系列的文章；训练框架与推理引擎的内部只作为指标来源，不展开。


## 一、总览

### 1. 引擎的需求

引擎对可观测层的要求，和它对资源层的要求一样具体：

- **训练框架**要求平台知道"这个任务此刻在算还是在等"。它自己知道 step 时间、loss、MFU、每个 rank 的前向 / 反向 / 通信时间，但这些数字只在它的日志与 TensorBoard 里；平台需要它们以指标形式暴露（每 rank 一份、带 rank 与 job 标签），才能回答"这 32 张卡里哪一张在拖后腿"。反过来，训练框架不知道自己所在的卡温度多高、有没有 XID、NVLink 有没有重传——这些只有 DCGM 知道。**两边的数据要在同一个时间轴上对齐**，否则 straggler 的根因永远查不到硬件层。
- **推理引擎**要求平台按它的指标而不是 CPU 利用率做决策。vLLM 暴露的 `vllm:num_requests_waiting`、`vllm:kv_cache_usage_perc`、TTFT / TPOT 直方图是扩缩容与路由的输入（第六、七篇），也是 SLO 的定义域。它还要求平台把 token 数记账：`vllm:prompt_tokens_total` 与 `vllm:generation_tokens_total` 按模型累计，但**按租户**的计数只能在网关做——引擎不知道请求属于谁。
- **两类引擎共同要求**：一张卡的硬件指标能归到一个 Pod、一个队列、一个团队。没有这条映射，利用率只能按节点看，账单只能按集群算。

不满足会发生什么：训练任务变慢时只能看到"GPU 利用率 100%"（NCCL kernel 自旋也算 100%），查不出是通信还是硬件；推理服务扩容依据错误的信号，晚高峰超时而平时浪费；账单只有一个总数，每个团队都说"不是我用的"。

### 2. K8s 的空缺

Kubernetes 自带的可观测面对 GPU 几乎是空白：

- **kubelet 与 cAdvisor 不采 GPU 指标**。cAdvisor 的 `container_cpu_usage_seconds_total`、`container_memory_working_set_bytes` 对 GPU 容器同样有效，但没有任何 GPU 对应物；kube-state-metrics 的 `kube_pod_container_resource_requests{resource="nvidia_com_gpu"}` 只知道**请求了几张**，不知道用了多少。
- **扩展资源是不透明的整数**。`nvidia.com/gpu: 2` 在 API 里就是数字 2，哪两张卡、什么型号、MIG 还是整卡、在 K8s 对象里看不到；只有 kubelet 的 pod-resources API（`k8s.io/kubelet/pkg/apis/podresources/v1` 的 `List`）能把 Pod / 容器与设备 ID 对上，而它是一个节点本地的 Unix socket，不是集群级 API。
- **HPA 与 Metrics Server 只认 CPU 和内存**。自定义指标要经过 Prometheus Adapter 或 KEDA（第六篇）。
- **没有成本概念**。K8s 不知道一个节点每小时多少钱，更不知道一张卡多少钱；Pod 的 `requests` 是调度输入，不是账单输入。
- **没有跨请求的追踪**。一个推理请求经过网关、Endpoint Picker、引擎三跳，K8s 层面没有任何东西把它们关联起来。

### 3. 平台的机制

填这些空缺的组件与它们之间的数据流：

```text
  层        采集器 / 来源                                   指标前缀 / 名字                        归属标签怎么来
  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  硬件      DCGM Exporter（DaemonSet，每节点一个）           DCGM_FI_DEV_* · DCGM_FI_PROF_*         --kubernetes：查 kubelet pod-resources
            读驱动 / NVML / 硬件性能计数器                                                            socket，把 gpu/UUID 映射到 pod/namespace/container
  容器 Pod  kube-state-metrics · cAdvisor（kubelet 内置）    kube_pod_* · kube_node_* · container_*  K8s 对象自带 namespace / pod / label
            Kueue / Volcano 控制器的 /metrics               kueue_pending_workloads 等             cluster_queue / local_queue 标签
  引擎      vLLM /metrics（每副本）· 训练进程或 sidecar      vllm:* · 训练框架自定义（step_time 等）  Prometheus 服务发现给 pod / namespace；
                                                                                                   训练侧自带 rank / job 标签
  请求      Endpoint Picker（llm-d-router）· 网关的访问日志   llm_d_epp_* · inference_objective_*    model_name / target_model_name /
                                                                                                   fairness_id（租户）/ priority
                                     │
                                     ▼
            Prometheus（agent 模式采集 → 中心存储；recording rules 算三个数）
            OpenTelemetry Collector（trace：网关 → EPP → 引擎；日志：NCCL / XID 模式匹配）
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
        Grafana 看板          Alertmanager 告警        成本引擎（OpenCost / allocate.py）
        四层同屏              XID · 队列 · TTFT p95     GPU 小时 × 单价 → 按 label / 队列分摊
                                                        → 每百万 token 成本 → 回流到配额 / 切分 / 扩缩容
```

四层的边界是**谁能采**：硬件层只有 DCGM 能采（它读驱动），容器层只有 kubelet 与 kube-state-metrics 能采（它们读 K8s 对象），引擎层只有引擎自己能采（队列长度、KV cache 在进程内），请求层只有网关能采（租户身份在 HTTP 头里）。任何一层都替代不了另一层，四层指标要在 Prometheus 里靠 `pod` / `namespace` 这组公共标签 join 起来——这是 DCGM Exporter 的 `--kubernetes` 映射为什么是整条链的关键一环。

### 4. 本文的章节安排

```text
二、四层指标            四层指标表 · DCGM 的字段与标签 · 为什么 GPU_UTIL 不是利用率 · 容器层 · vllm:* · 训练框架 · EPP 的请求层指标
三、三个数字            分配率 / 使用率 / 有效利用率的定义 · PromQL · 三者差距的含义
四、指标管线            采集与 agent 模式 · 高基数 · OpenTelemetry trace · 日志告警（NCCL / XID）
五、训练任务的可观测    任务级 GPU 时间线 · 每 rank step 时间 · checkpoint 耗时 · 通信 / 计算比 · 引擎作为黑盒
六、推理服务的 SLO      TTFT / TPOT 分位数 · goodput · 每副本饱和点 · 从 SLO 反推容量
七、成本模型            GPU 小时 × 单价 · 分摊维度 · OpenCost 的 GPU 支持 · 闲置 / 排队 / 冷启动三种隐性成本
八、每百万 token 的成本  公式 · 量化 / PD 分离 / 批大小的影响 · 与 API 定价的量级对比
九、FinOps 回路与容量规划  成本 → 配额 / 切分 / 扩缩容参数 · 从历史曲线到采购 · 训练与推理错峰
十、核心问题            50 个百分点的分解表 · 每项对应哪一篇
十一、代价与边界        采集开销 · 基数 · 归因的误差 · 成本模型的假设
十二、本文小结          要点 · 四栏表 · 源码位置 · 练手项目 obs/ 与 cost/
十三、系列总结          读者手上有什么 · 三条线 · 三种能力 · 系列目录
```


## 二、四层指标：从 DCGM 到网关

### 1. 四层指标表

```text
层      指标（精确名）                                             来源                     用途
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
硬件    DCGM_FI_DEV_GPU_UTIL                                      DCGM Exporter 默认表     "有 kernel 在跑"的时间比例；不是利用率（第 3 节）
        DCGM_FI_PROF_GR_ENGINE_ACTIVE                             默认表                    图形引擎活跃比例；OpenCost 用它当 GPU 使用率
        DCGM_FI_PROF_SM_ACTIVE · DCGM_FI_PROF_SM_OCCUPANCY        默认表中被注释，需自行开启  SM 至少有一个 warp / 驻留 warp 比例；本篇的"使用率"
        DCGM_FI_PROF_PIPE_TENSOR_ACTIVE                           默认表                    Tensor core 活跃比例；与 MFU 同趋势
        DCGM_FI_PROF_DRAM_ACTIVE                                  默认表                    HBM 接口活跃比例；decode 阶段的主指标
        DCGM_FI_DEV_FB_USED · DCGM_FI_DEV_FB_FREE · FB_RESERVED    默认表                    显存占用（MiB），驱动视角
        DCGM_FI_DEV_GPU_TEMP · DCGM_FI_DEV_MEMORY_TEMP             默认表                    温度
        DCGM_FI_DEV_POWER_USAGE · DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION  默认表               功耗（W）与累计能耗（mJ）
        DCGM_FI_DEV_SM_CLOCK · DCGM_FI_DEV_MEM_CLOCK               默认表                    时钟；降频的直接证据
        DCGM_FI_DEV_XID_ERRORS                                    默认表                    最近一次 XID 码；非 0 即告警
        DCGM_FI_DEV_PCIE_REPLAY_COUNTER · DCGM_FI_PROF_PCIE_TX/RX_BYTES  默认表             PCIe 重放与流量
        DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL                        默认表                    NVLink 总带宽计数；CRC / REPLAY 错误计数在表中被注释
        DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS · ROW_REMAP_FAILURE  默认表                 HBM 行重映射；ECC 字段在表中被注释
容器    kube_pod_container_resource_requests{resource="nvidia_com_gpu"}  kube-state-metrics   请求了几张卡（分配率的分子）
Pod     kube_node_status_allocatable{resource="nvidia_com_gpu"}   kube-state-metrics        节点可分配卡数（分配率的分母）
        kube_pod_status_phase · kube_pod_labels                   kube-state-metrics        只算 Running 的；label 作为分摊维度
        container_cpu_usage_seconds_total · container_memory_working_set_bytes  cAdvisor    dataloader 的 CPU、主机内存（异步 checkpoint 的暂存）
        kueue_pending_workloads{cluster_queue,status}             Kueue v0.19.2             排队深度；status=active / inadmissible
        kueue_admitted_active_workloads · kueue_admission_wait_time_seconds  Kueue          在跑的任务数；从提交到准入的等待
        kueue_cluster_queue_resource_usage / _nominal_quota / _resource_reservation  Kueue   队列用量 / 名义配额 / 预留，按 flavor 与 resource
引擎    vllm:num_requests_running · vllm:num_requests_waiting     vLLM v0.23.0              在跑 / 排队的请求数（gauge）
        vllm:kv_cache_usage_perc                                  vLLM                      KV cache 占用（0–1）
        vllm:time_to_first_token_seconds（histogram）              vLLM                      TTFT
        vllm:inter_token_latency_seconds · vllm:request_time_per_output_token_seconds  vLLM  ITL（逐 token）与每请求 TPOT
        vllm:e2e_request_latency_seconds · vllm:request_queue_time_seconds ·             vLLM  端到端、排队、prefill、decode 各阶段耗时
        vllm:request_prefill_time_seconds · vllm:request_decode_time_seconds
        vllm:prompt_tokens_total · vllm:generation_tokens_total   vLLM（Counter 加 _total）   token 吞吐；每百万 token 成本的分母
        vllm:prefix_cache_queries_total · vllm:prefix_cache_hits_total  vLLM                prefix cache 命中率 = hits / queries
        vllm:num_preemptions_total · vllm:request_success_total{finished_reason}  vLLM       抢占次数；完成数按结束原因
        训练框架：step 时间 · loss · MFU · 每 rank 阶段耗时 · checkpoint 时长  框架自定义     第五章；名字由训练侧约定
请求    llm_d_epp_request_total{model_name,target_model_name,fairness_id,priority}  EPP     按租户（fairness_id）与模型的请求数
        llm_d_epp_request_input_tokens · llm_d_epp_request_output_tokens（histogram） EPP    按租户与模型的 token 分布；_sum 即累计 token
        llm_d_epp_request_ttft_seconds · llm_d_epp_request_streaming_tpot_seconds  EPP       网关视角的 TTFT / TPOT（含网关与路由开销）
        llm_d_epp_average_kv_cache_utilization · llm_d_epp_average_queue_size · llm_d_epp_ready_endpoints  EPP  池级视图
```

表里的名字全部在对应版本检出中 grep 到；DCGM 的"默认表"指 `dcgm-exporter etc/default-counters.csv`（与 `etc/dcp-metrics-included.csv` 的启用项相同）。两点先说明：**vLLM 的 Counter 在 Prometheus 暴露时带 `_total` 后缀**（源码里 `name="vllm:prompt_tokens"`，抓到的是 `vllm:prompt_tokens_total`，`docs/design/metrics.md` 里两种写法并存）；**EPP 的 `inference_objective_*` 与 `inference_pool_*` 前缀在 llm-d-router v0.10.0 已标 Deprecated**，`pkg/epp/metrics/metrics.go` 的注释指向 `llm_d_epp_*` 替代（`llm_d_router_metrics.go`），本篇用新名。

### 2. 硬件层：DCGM Exporter 的字段与标签

DCGM Exporter 以 DaemonSet 跑在每个 GPU 节点上，通过 DCGM host engine 读 NVML 与硬件性能计数器，按 `--collectors`（环境变量 `DCGM_EXPORTER_COLLECTORS`，`pkg/cmd/app.go` 的 `CLIFieldsFile`）指定的 CSV 决定暴露哪些字段，默认 `/etc/dcgm-exporter/default-counters.csv`（`internal/pkg/appconfig/types.go` 的 `DefaultCollectorsFile`）。CSV 每行是"DCGM 字段名, Prometheus 类型, 帮助文本"，以 `#` 开头的行是注释。**默认表里 `DCGM_FI_PROF_SM_ACTIVE` 与 `DCGM_FI_PROF_SM_OCCUPANCY` 是被注释掉的**，ECC 与 NVLink 错误计数也是；要用它们，Helm 的 `customMetrics` 值要给出**完整**的 CSV（values.yaml 注释明确说"Must be the complete list and is not additive"）。第十二章的 `dcgm-values.yaml` 就是这么做的。

每条指标的标签由 `internal/pkg/rendermetrics/render_metrics.go` 的实体组分支决定。GPU 实体（`FE_GPU`）固定带 `gpu`（索引）、`UUID`、`pci_bus_id`、`device`（`nvidia0` 一类）、`modelName`、`hostname`；MIG 实例额外带 `GPU_I_PROFILE` 与 `GPU_I_ID`。**Kubernetes 归属标签**由 `--kubernetes`（`DCGM_EXPORTER_KUBERNETES`，Helm 的 DaemonSet 模板固定设为 `true`）开启：`internal/pkg/transformation/kubernetes.go` 的 `PodMapper` 通过挂载在 `/var/lib/kubelet/pod-resources` 的 socket（values 的 `kubeletPath`）调用 kubelet pod-resources API 的 `PodResourcesListerClient.List`，拿到每个 Pod 每个容器分到的设备 ID 列表，再按 `--kubernetes-gpu-id-type`（`uid` 或 `device-name`）与 DCGM 看到的 GPU 对上，给指标加上 `pod`、`namespace`、`container` 三个属性（`internal/pkg/transformation/const.go` 的 `podAttribute` 等；`--use-old-namespace` 时是 `pod_name` / `pod_namespace` / `container_name`）。kubelet 侧这份记录来自 device manager 的分配状态（`kubernetes pkg/kubelet/cm/devicemanager/manager.go` 的 `ManagerImpl.GetDevices(podUID, containerName)`）。

再往上一层是 **Pod label**：`--kubernetes-enable-pod-labels`（Helm `kubernetes.enablePodLabels: true`，会一并创建读 Pod 的 ClusterRole）让 PodMapper 用 kube client 读 Pod 对象的 labels，经 `utils.SanitizeLabelName` 把 `.` `/` `-` 换成 `_` 后直接作为指标标签（`team` → `team`，`app.kubernetes.io/name` → `app_kubernetes_io_name`）；只有与保留名冲突时（如 Pod 上有一个叫 `gpu` 的 label）才加 `pod_label_` 前缀（`availablePodLabelName`）。`kubernetes.podLabelAllowlistRegex` 限定哪些 label 进指标——不限定的话每个 Pod 的全部 label 都变成时序标签，第四章会算这笔基数账。`--kubernetes-enable-pod-uid` 加 `pod_uid`；`kubernetesDRA.enabled` 让它同时识别 DRA 分配的设备（`dra.go`，标签 `dra_claim_name` 等），对应第二篇的 `ResourceClaim` 路线。

这一步的产物是本篇全部归因的基础：一条 `DCGM_FI_PROF_SM_ACTIVE{gpu="3",UUID="GPU-…",hostname="node-7",pod="train-ddp-0",namespace="team-a",container="pytorch",team="a"} 0.41` 同时回答了"哪张卡、在哪个节点、给了谁、属于哪个团队、此刻算得多满"。

### 3. 为什么 GPU_UTIL 不是利用率

`DCGM_FI_DEV_GPU_UTIL` 的帮助文本是"GPU utilization (in %)"，它的定义（以 NVIDIA 文档为准）是**过去采样周期内有至少一个 kernel 在 GPU 上执行的时间比例**。一个只用一个 SM 跑 kernel 的进程和一个用满 132 个 SM 的进程，在这个指标上都是 100%；一个在 NCCL all_reduce 上自旋等对端的 rank，也是 100%。它回答的是"GPU 是否空闲"，不是"GPU 用了多少"。对推理服务尤其误导：decode 阶段 kernel 连续发射但每个 kernel 只做很小的矩阵乘，`GPU_UTIL` 接近 100% 而 Tensor core 活跃度可能不到 20%。

`DCGM_FI_PROF_*` 系列来自硬件性能计数器（DCGM 的 profiling 模块，Ampere 及之前的卡依赖额外的 proprietary 包，见 dcgm-exporter README），是按周期统计的比例：

```text
DCGM_FI_PROF_GR_ENGINE_ACTIVE      图形 / 计算引擎有工作的周期比例；比 GPU_UTIL 精细一档，但仍不区分"用了几个 SM"
DCGM_FI_PROF_SM_ACTIVE             所有 SM 上"至少有一个 warp 驻留"的周期比例的平均；一个 kernel 只占一半 SM 时约 50%
DCGM_FI_PROF_SM_OCCUPANCY          驻留 warp 数 / 最大 warp 数；SM_ACTIVE 高而 OCCUPANCY 低 = kernel 很多但每个都小
DCGM_FI_PROF_PIPE_TENSOR_ACTIVE    Tensor core 管线活跃周期比例；训练与 prefill 的主指标，直接对应 MFU 的趋势
DCGM_FI_PROF_DRAM_ACTIVE           HBM 接口收发数据的周期比例；decode 是 memory-bound，这个指标才反映它是否吃满带宽
```

哪一个是"利用率"取决于负载：训练与 prefill 看 `TENSOR_ACTIVE`，decode 看 `DRAM_ACTIVE`，跨负载比较用 `SM_ACTIVE`——它对两类负载都有意义，也是总纲核心问题里用的那个。本篇第三章的三个数字用 `SM_ACTIVE`；OpenCost 用的是 `GR_ENGINE_ACTIVE`（第七章），两者数值不同，做对账时要知道各自看的是什么。一个实用的经验：**`GPU_UTIL` 高而 `SM_ACTIVE` 低，是"分配了但没算满"的最直接证据**，多数情况下是通信等待（训练）或小批量 decode（推理）。

### 4. 容器与 Pod 层

容器层不采 GPU 的运行数据，但它是分配率的来源与所有分摊维度的来源。kube-state-metrics 把 Pod 的 `resources.requests` 展开为 `kube_pod_container_resource_requests{namespace,pod,container,node,resource,unit}`，`nvidia.com/gpu` 被规范化为 `resource="nvidia_com_gpu"`（斜杠与点换成下划线，这是 kube-state-metrics 的通行做法，OpenCost 的查询 `queryFmtGPUsRequested` 也这么写）；节点侧是 `kube_node_status_allocatable{resource="nvidia_com_gpu"}` 与 `kube_node_status_capacity`。MIG 的 `mixed` 策略（第四篇）会让资源名变成 `nvidia_com_mig_1g_5gb` 一类，分配率要按 MIG 实例数而不是物理卡数算，或者换用第三章基于 DCGM 的算法。HAMi 的 `nvidia.com/gpumem` / `gpucores` 则要按显存或算力份额折算，本篇的 PromQL 只处理整卡与 MIG 两种。

Kueue 的控制器指标（`pkg/metrics/metrics.go`，Subsystem 为 `kueue`）是排队成本的来源。`kueue_pending_workloads{cluster_queue,status}` 的 `status` 有 `active`（在准入队列里）与 `inadmissible`（尝试过准入失败、等集群条件变化再试）两个值；`kueue_admission_wait_time_seconds{cluster_queue,priority_class}` 是从创建或重新入队到准入的直方图；`kueue_cluster_queue_resource_usage{cohort,cluster_queue,flavor,resource}` 与 `kueue_cluster_queue_nominal_quota` 之比就是**队列级的配额使用率**——第九章"哪个队列长期闲置"看的是它；`kueue_cluster_queue_resource_reservation` 与 `kueue_cluster_queue_resource_pending`（后者是尚未分配到 flavor 的待处理请求量）区分了"已预留"与"在排队"。这些指标全部带 `cluster_queue` 标签，是把成本按队列分摊的第二条维度（第一条是 Pod label）。

### 5. 引擎层：vllm:* 与训练框架

vLLM 的指标在 `vllm/v1/metrics/loggers.py` 的 `PrometheusStatLogger` 里定义，全部带 `model_name` 与 `engine` 两个标签（DP 多引擎时每个引擎一份，`create_metric_per_engine`）。按用途分三组：

```text
组        指标                                                  类型 / 说明
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
状态      vllm:num_requests_running · vllm:num_requests_waiting   Gauge；扩缩容（第六篇）与路由（第七篇）的输入
          vllm:num_requests_waiting_by_reason{reason}             Gauge；等 KV 块还是等其他资源
          vllm:kv_cache_usage_perc                                Gauge，0–1；"KV-cache usage. 1 means 100 percent usage."
          vllm:engine_sleep_state{sleep_state}                    Gauge；缩容到"睡眠"而非零的形态
延迟      vllm:time_to_first_token_seconds                        Histogram；TTFT
          vllm:inter_token_latency_seconds                        Histogram；逐 token 的间隔（docs 称之为 TPOT）
          vllm:request_time_per_output_token_seconds              Histogram；每请求平均 TPOT
          vllm:e2e_request_latency_seconds                        Histogram；端到端
          vllm:request_queue_time_seconds · request_prefill_time_seconds · request_decode_time_seconds · request_inference_time_seconds
                                                                  Histogram；一个请求的时间分解
吞吐 /    vllm:prompt_tokens_total · vllm:generation_tokens_total  Counter；rate() 即 token/s
计数      vllm:iteration_tokens_total                             Histogram；每次调度迭代处理的 token 数（批大小的代理）
          vllm:prefix_cache_queries_total · vllm:prefix_cache_hits_total   Counter；命中率 = rate(hits)/rate(queries)
          vllm:request_success_total{finished_reason}             Counter；stop / length / abort
          vllm:num_preemptions_total                              Counter；KV 不够时的抢占，非 0 说明显存压力
          vllm:request_prompt_tokens · vllm:request_generation_tokens   Histogram；每请求输入 / 输出长度分布
```

三点使用上的提醒。**分位数从 `_bucket` 算**：`histogram_quantile(0.95, sum(rate(vllm:time_to_first_token_seconds_bucket[5m])) by (le, model_name))`，桶边界在 `loggers.py` 里硬编码（TTFT 从 0.001 s 起），超出最大桶的值会被截到最大桶，长上下文 prefill 的 p99 可能因此失真。**`_sum / _count` 给平均值**，对 SLO 没用但对成本有用——`rate(vllm:request_generation_tokens_sum[1h]) / rate(vllm:request_generation_tokens_count[1h])` 是平均输出长度。**`engine` 标签在单引擎时恒为 `0`**，聚合时记得 `sum by (model_name)`。

训练框架没有统一的指标规范。平台侧的做法是**把训练进程当黑盒，只约定一组它必须暴露的指标名**，由训练脚本用 `prometheus_client` 暴露，或写本地 JSONL 由节点上的 sidecar 转换。本系列练手项目约定的最小集合（名字是本系列自定的，不是任何框架的原生名）：`train_step_seconds{job,rank}`（gauge，最近一步耗时）、`train_step_total{job,rank}`（counter）、`train_last_step_timestamp_seconds{job}`（hang 检测）、`train_loss{job}`、`train_tokens_per_second{job}`、`train_ckpt_seconds{job,phase}`（save / load）。第五章据此讲训练侧的可观测。

### 6. 请求层：EPP 与网关

请求层的指标出自 Endpoint Picker。llm-d-router v0.10.0 的 `pkg/epp/metrics/llm_d_router_metrics.go` 以 `llm_d_epp` 为 Subsystem 定义了一组带租户维度的指标：`llm_d_epp_request_total`、`llm_d_epp_request_error_total`、`llm_d_epp_request_duration_seconds`、`llm_d_epp_request_input_tokens`、`llm_d_epp_request_output_tokens`、`llm_d_epp_request_cached_tokens`、`llm_d_epp_request_ttft_seconds`、`llm_d_epp_request_streaming_tpot_seconds`、`llm_d_epp_request_streaming_itl_seconds`，标签为 `model_name`、`target_model_name`、`fairness_id`、`priority`（部分加 `streaming`）。`fairness_id` 来自请求头 `x-llm-d-inference-fairness-id`（`pkg/epp/metadata/consts.go` 的 `FlowFairnessIDKey`，旧名 `x-gateway-inference-fairness-id`），是第七篇多租户里的租户标识——**按租户计 token 的数据源就是 `llm_d_epp_request_input_tokens_sum` 与 `llm_d_epp_request_output_tokens_sum` 按 `fairness_id` 求和**。池级视图是 `llm_d_epp_average_kv_cache_utilization`、`llm_d_epp_average_queue_size`、`llm_d_epp_average_running_requests`、`llm_d_epp_ready_endpoints`，它们是 EPP 从各副本的 `vllm:*` 拉来再聚合的，是引擎层指标在请求层的投影。

网关（Envoy / Istio / kgateway）自身的访问日志与 `envoy_cluster_upstream_rq_*` 一类指标补充了 EPP 看不到的部分：被限流拒绝的请求（第七篇的配额）不会到 EPP，只在网关有记录；流式响应中途断开的请求，EPP 记录的 `output_tokens` 是断开前已生成的数。**租户账单以网关侧为准、引擎侧为对账**：两边的 token 总数差不应超过几个百分点，差得多说明有请求绕过网关直连了引擎——这本身就是一个要告警的事。


## 三、三个数字：分配率、使用率、有效利用率

### 1. 定义

```text
分配率 A   = 已分配的 GPU 数 / 集群可分配的 GPU 数                 调度视角："给出去了多少"
使用率 U   = 已分配 GPU 上 SM_ACTIVE 的平均                        引擎视角："给出去的算得多满"
有效利用率 E = 全部 GPU 上 SM_ACTIVE 的平均 ≈ A × U                平台视角："买来的算了多少"
```

三个数都按时间平均（一天、一月），也都可以按 namespace、队列、团队切分。它们之间的两段差距是平台可改进空间的两个部分：`1 − A` 是**没给出去的**（碎片、预留、隔离的坏节点、配额设置过紧），`A − E = A(1 − U)` 是**给出去了但没算的**（排队占位、通信等待、低峰空转、开发环境挂着）。总纲核心问题里的 85% 与 35% 就是 A 与 E；由此 U = 35 / 85 ≈ 41%，第十章把 `A − E = 50` 个点分解到原因。

### 2. PromQL

分配率有两种算法。基于 kube-state-metrics（要求 Pod 处于 Running）：

```text
platform:gpu_allocated:count =
  sum(
    kube_pod_container_resource_requests{resource="nvidia_com_gpu"}
    * on (namespace, pod) group_left ()
    (kube_pod_status_phase{phase="Running"} == 1)
  )
platform:gpu_allocatable:count = sum(kube_node_status_allocatable{resource="nvidia_com_gpu"})
platform:gpu_allocated_ratio   = platform:gpu_allocated:count / platform:gpu_allocatable:count
```

基于 DCGM 的 Pod 映射（不依赖 kube-state-metrics，MIG 下按实例计）：

```text
platform:gpu_allocated_ratio:dcgm =
  count(DCGM_FI_DEV_GPU_UTIL{pod!=""}) / count(DCGM_FI_DEV_GPU_UTIL)
```

后者的逻辑是"一张卡的指标带了 `pod` 标签就是分配了"，它与前者的差在于：一个 Pending 或 Terminating 的 Pod 不会出现在 pod-resources 里，但 `requests` 已经记在 kube-state-metrics 上；两者相减就是**调度中的卡**。使用率与有效利用率：

```text
platform:gpu_sm_active:avg_allocated = avg(DCGM_FI_PROF_SM_ACTIVE{pod!=""})         # U
platform:gpu_sm_active:avg_all       = avg(DCGM_FI_PROF_SM_ACTIVE)                   # E
```

按团队切分，在 `avg by (team)` 或 `count by (namespace)` 上做同样的事；按队列切分，要把 Pod 的 `kueue.x-k8s.io/queue-name` label（Kueue 用它把 Job 送进 `LocalQueue`）放进 DCGM 的 `podLabelAllowlistRegex`，它会以 `kueue_x_k8s_io_queue_name` 的名字出现在 DCGM 指标上。第十二章的 `prometheus-rules.yaml` 把这些做成 recording rules，`grafana-dashboard.json` 把 A、U、E 三条线画在同一张图上。

### 3. 差距的含义

三个数在一天里的形状比它们的均值更有信息：

- **A 高而平、E 有明显日夜周期**：推理服务按峰值常驻，低峰空转——第六篇的扩缩容与缩零，或第四篇的切分把低峰的卡让出来。
- **A 有阶梯、每个阶梯的前几分钟 E 为零**：训练任务启动期（拉镜像、加载 checkpoint、rendezvous）——第二篇的镜像体积与第五篇的 checkpoint 读吞吐。
- **A 长期 100%、`kueue_pending_workloads` 不为零、E 在 40% 附近**：卡是满的但算得不满，配额把任务挡在外面而里面的任务在等通信——第三篇的配额与第五篇的网络，两边都有事做。
- **某些 namespace 的 A 全天不变、E 接近零**：开发环境与 notebook 占着卡——第四篇的时间片，或第三篇的配额与超时回收。
- **U 在某个 job 上显著低于同类 job**：straggler 或数据管线——第五章。

三个数字是**诊断的入口而不是结论**：每一种形状指向一个章节，进了那一章才有具体的机制可调。


## 四、指标管线

### 1. 采集、联邦与 agent 模式

一个 64 卡的集群，DCGM Exporter 默认每 30 秒（`DCGM_EXPORTER_INTERVAL` 默认 30000 ms）暴露每张卡二三十个字段，加上每个 vLLM 副本上百条时序（直方图每个桶一条）、kube-state-metrics 与 Kueue，总量在几万条时序的量级，单个 Prometheus 轻松承受。规模上去以后（几百节点、几千副本）两条路：**联邦**——每个集群或每个区一个 Prometheus，中心 Prometheus 通过 `/federate` 只拉聚合过的 recording rules（三个数字、按团队的 GPU 小时），原始时序留在边缘；**agent 模式**——边缘 Prometheus 以 `--agent` 运行，不存储、只抓取并通过 remote write 推到中心的长期存储（Thanos、Mimir、VictoriaMetrics 一类）。成本核算需要**至少 13 个月**的保留期（同比对账），这是长期存储的理由；而告警只需要几小时的窗口，放在边缘就近评估。

Prometheus 的 ServiceMonitor（dcgm-exporter Helm 的 `serviceMonitor.enabled`、`interval`、`relabelings`）是接入点。一个常用的 relabel：把 `__meta_kubernetes_pod_node_name` 写进 `nodename`，这样 DCGM 的 `hostname` 与 kube-state-metrics 的 `node` 能 join。

### 2. 高基数

时序数 = 指标数 × 标签值组合数。AI 负载在三处容易失控：

- **Pod label 全量进 DCGM 指标**。Pod 上常有十几个 label（`app`、`pod-template-hash`、`controller-revision-hash`、`batch.kubernetes.io/job-name`……），其中 `pod-template-hash` 与 `job-name` 每次部署都变。不设 `podLabelAllowlistRegex`，每张卡每次调度到新 Pod 就产生一组新时序，旧的变成"陈旧但仍在索引里"的时序。**只放分摊需要的 label**（`team`、`project`、`kueue.x-k8s.io/queue-name`），`pod` 本身已经是唯一标识，不需要更多。
- **请求级标签**。给 `llm_d_epp_request_total` 加 `request_id`、`user_id` 或完整的 `model` 字符串，每个请求一条时序，时序库在小时级内被撑爆。llm-d-router v0.10.0 的 `pkg/epp/metrics/cardinality.go` 对此有内建的保护：`maxModelLabelValues` 与 `maxFairnessLabelValues` 都是 1000，超过上限的 `model_name` / `fairness_id` 值统一折叠为 `other`（`overflowValue`），`boundFairnessID` / `boundModels` 在每次记录时调用。这意味着**租户数超过 1000 时账单里会出现一个 `other`**——做多租户计费的平台要么把租户 ID 映射到更粗的计费单元，要么把按租户计费移到日志管线（第 4 节）。
- **每 rank 的训练指标**。1024 个 rank × 每 rank 十几个指标是可以接受的（一两万条），但如果再乘上 `step` 作为标签就不行——step 是值不是标签，这是 Prometheus 与 TensorBoard 的分工边界。

一个经验的阈值：单个 Prometheus 的活跃时序控制在几百万以内；每加一个标签之前，先算它的取值个数。

### 3. OpenTelemetry：把一个请求串起来

指标回答"总体怎样"，trace 回答"这一个请求为什么慢"。一个推理请求的路径是网关（Envoy）→ `ext_proc` 调 EPP → EPP 选副本 → Envoy 转发到引擎 → 引擎排队、prefill、decode、流式返回。每一跳都可能是延迟的来源：网关限流排队、EPP 的调度插件耗时（`llm_d_epp_plugin_duration_seconds{extension_point,plugin_type,plugin_name}` 与 `llm_d_epp_scheduler_e2e_duration_seconds` 是它的指标面）、引擎的 `request_queue_time_seconds`。把它们对齐需要一个跨进程的 trace context：Envoy 与 EPP 支持 W3C `traceparent` 头的传播，vLLM 有 OpenTelemetry 导出（以 v0.23.0 文档为准，启动参数决定是否开启）。OpenTelemetry Collector 收三方的 span，送到 Tempo / Jaeger 一类后端，一个请求的瓦片图上就能看到三跳各占多少。

trace 的采样率要低（1%–5%）——每个请求一个 span 树的存储成本远高于指标；但**慢请求要全采**：tail-based sampling 在 Collector 里按"端到端超过 SLO"的条件保留。trace 与指标的关系是：指标发现 TTFT p95 越线 → trace 找到越线请求都卡在哪一跳 → 回到那一跳的指标看原因。

### 4. 日志：NCCL 与 XID 的告警

两类故障只在日志里出现，指标是它们的滞后信号。

**XID** 是驱动报告的 GPU 错误码，权威来源是内核日志（`dmesg` / `journalctl -k` 里的 `NVRM: Xid (PCI:…): 79, …`）。DCGM 的 `DCGM_FI_DEV_XID_ERRORS` 只报最近一个码，作为告警触发足够（`> 0` 即 page），作为复盘不够。做法是节点上一个日志采集器（Fluent Bit、Vector、OTel Collector 的 filelog receiver）对 `NVRM: Xid` 做正则匹配，产出一条带 `node`、`pci_bus_id`、`xid` 的事件；再由 Alertmanager 或一个小控制器执行**隔离动作**——给节点打 `cordon`、在 Kueue 的 `ResourceFlavor` 上通过 taint 把它摘出资源池（第三篇），让正在跑的 gang 任务按第五篇的容错路径重启到别的节点。XID 的不同码含义不同（以 NVIDIA 文档为准），常见的 79（GPU 掉线）、48 / 63 / 64（显存错误）、13 / 31 / 43（应用侧非法访问，多为软件 bug）要在告警规则里分级：前两类 page 并隔离，第三类只 record。

**NCCL** 的错误在训练进程的 stderr 里：`NCCL WARN` 后接 `Cuda failure` / `unhandled system error` / `Socket ... connect failed`，以及 PyTorch watchdog 的 `Watchdog caught collective operation timeout`。它们在指标上表现为 `train_last_step_timestamp_seconds` 停止前进（hang）或任务重启计数增加，但日志里的那行才说明**哪个 rank、哪条链路**。采集方式同上，正则匹配 `NCCL WARN` 与 `Watchdog caught`，标签带 `job`、`pod`（即 rank 的位置），告警按 job 聚合抑制——一次 hang 会让全部 rank 同时报超时，不要发 256 条。

日志管线还承担一件本篇关心的事：**按租户的精确计费**。当租户数超过 EPP 指标的基数上限、或者需要按请求逐条对账时，网关的访问日志（每请求一行，含租户、模型、输入 / 输出 token 数、状态、耗时）进对象存储，成本核算从日志聚合而不是从 Prometheus 拉。指标算趟数，日志算账。


## 五、训练任务的可观测

### 1. 任务级的 GPU 时间线

训练任务是一组 Pod，平台看它的第一张图是**这个 job 的全部卡在时间轴上的 `SM_ACTIVE` / `TENSOR_ACTIVE` 热力图**：横轴时间、纵轴 rank（或 `pod` + `gpu`），颜色是活跃度。用 `job` label（Kubeflow Trainer 的 `TrainJob` 会给 Pod 打 `jobset.sigs.k8s.io/jobset-name` 一类的 label，进 DCGM 的 allowlist）过滤 DCGM 指标即可。它一眼能看出四种形态：

```text
形态                                     含义                                       对应
全体在启动后 N 分钟才亮                    拉镜像 · 加载 checkpoint · rendezvous       第二篇镜像体积；第五篇存储读吞吐
全体周期性同时变暗数十秒                    checkpoint 写入（同步保存）                  第五篇的异步 checkpoint
一两行持续比其他行暗                        straggler：该 rank 慢，其余在等它            第五篇网络（该节点链路）；硬件层降频 / XID
全体亮度均匀但只有 40%                      通信 / 计算比高，或 kernel 小                引擎内部（并行策略 / 批大小），平台层只能提供 RDMA
```

这张图的价值是把"任务 MFU 只有 30%"从一个数字变成一种形状；形状决定该找平台还是找算法。

### 2. 每 rank 的 step 时间

同步训练里每一步的时间由最慢的 rank 决定，所以要看的不是平均 step 时间而是**各 rank step 时间的分布**：`histogram_quantile(0.95, …)` 与 p50 的比值超过 1.2 就有 rank 在拖。训练进程按第二章第 5 节的约定暴露 `train_step_seconds{job,rank}`，`topk(3, train_step_seconds{job="x"})` 直接给出嫌疑 rank；再用 `rank → pod → hostname / gpu` 的映射（DCGM 的 `pod` 标签反查）落到具体的卡，看它的 `DCGM_FI_DEV_SM_CLOCK`（降频）、`DCGM_FI_DEV_PCIE_REPLAY_COUNTER`（PCIe 重放）、`DCGM_FI_DEV_XID_ERRORS`。straggler 的平台侧根因通常就是这三个之一，或者是该节点的 RDMA 网卡（第五篇的 IB 计数器，不在 DCGM 里）。

平台不解释 step 时间为什么是 4.8 秒而不是 4.5 秒——那是引擎内部；平台负责的是"为什么 rank 517 是 6 秒而其他是 4.8 秒"。

### 3. checkpoint 耗时与通信 / 计算比

`train_ckpt_seconds{job,phase="save"}` 每次保存记一个值，与第五篇的算术对照：70B 模型约 1 TB 的状态、要求 1 分钟写完，对应约 17 GB/s 的聚合写带宽；如果实测是 4 分钟，要么存储带宽不够，要么没做分布式 / 异步保存。它在 GPU 时间线上就是"全体周期性变暗"的那些竖条，宽度乘以频率乘以卡数就是 checkpoint 的 GPU 小时成本（第七章）。

通信 / 计算比平台层没有直接指标——NCCL kernel 与计算 kernel 都算 `SM_ACTIVE`。间接的办法有两个：`DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` 与 `SM_ACTIVE` 的比值（通信 kernel 不用 Tensor core，比值越低通信占比越高）；`DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL` 与 `DCGM_FI_PROF_PCIE_TX_BYTES` 的流量形状（通信阶段有流量、计算阶段没有）。精确的分解要靠训练框架的 timer 输出——那是引擎侧的事，平台只要求它把"通信等待时间占比"作为一个指标暴露出来（约定名 `train_comm_wait_ratio{job}`），然后在第十章的分解表里用它。

### 4. 引擎作为黑盒的边界

平台对训练任务的可观测到此为止：硬件层全采、Pod 层全采、引擎层只采**约定的最小指标集**。不去解析训练框架的日志格式、不去读 TensorBoard 的事件文件、不去猜并行策略。原因是训练框架多、版本变化快、每个团队的脚本都不同；平台若依赖它们的内部格式，每次框架升级都要跟着改。约定一个指标接口（名字、标签、语义）并写进任务提交的规范里，是平台与引擎之间正确的边界——这也是第一篇"引擎的需求清单"反过来的那一面：平台对引擎的需求清单。


## 六、推理服务的 SLO 与容量

### 1. TTFT、TPOT 与 goodput

推理服务的 SLO 由两个分位数定义：TTFT p95 ≤ X（用户看到第一个字要多久）、TPOT p95 ≤ Y（之后每个字的间隔）。两者对资源的要求相反：TTFT 由 prefill 决定，是 compute-bound，要 Tensor core；TPOT 由 decode 决定，是 memory-bound，要 HBM 带宽；批大小增大让 TPOT 变差但吞吐变好。**goodput** 是单位时间内**同时满足两个 SLO** 的请求数——它才是容量的度量，吞吐（token/s）不是。一个把批大小推到极限的副本吞吐最高，但 TPOT 越线，goodput 反而可能低于批小一点的配置。

从 vLLM 指标算 goodput 没有直接的办法（引擎不知道 SLO），有两条近似：一是用 `vllm:e2e_request_latency_seconds_bucket{le="X"}` 的计数除以 `_count`，得到"端到端在 X 秒内完成"的比例，乘以 `rate(vllm:request_success_total)` 得到近似 goodput；二是在 EPP 或网关侧按请求判定——llm-d-router 的 `llm_d_epp_request_ttft_seconds` 与 `llm_d_epp_request_streaming_tpot_seconds` 两个直方图带 `fairness_id`，可以按租户算各自的 SLO 达成率，`x-llm-d-slo-ttft-ms` 头（`TTFTSLOHeaderKey`）甚至允许请求自带 SLO。以 EPP 侧为准，因为它测的是**用户看到的**延迟，包含了网关与路由的开销。

### 2. 每副本的饱和点

一个副本的负载—延迟曲线有一个拐点：在拐点之前加请求，TTFT 与 TPOT 几乎不变、吞吐线性上升；过了拐点，`vllm:num_requests_waiting` 开始非零、`vllm:kv_cache_usage_perc` 接近 1、TTFT 陡增。**饱和点就是拐点处的并发数或 token/s**，它是这个模型、这种硬件、这组引擎参数下的常数，用压测测一次，之后用来做两件事：

- **扩缩容阈值**（第六篇）：让 KEDA 按 `vllm:num_requests_waiting > 0 持续 N 秒`或 `kv_cache_usage_perc > 0.8` 扩容，阈值取饱和点的 70%–80%，留出扩容所需的几分钟；
- **容量规划**：需要的副本数 = 峰值 goodput 需求 / 单副本饱和点 goodput，向上取整再加冗余。

饱和点随三件事变：量化（FP8 的 KV cache 更小、饱和点更高）、PD 分离（prefill 与 decode 各自的饱和点独立，第八章）、prefix cache 命中率（`vllm:prefix_cache_hits_total / queries_total` 高时 prefill 负担轻）。任何一项变了都要重测。

### 3. 从 SLO 反推容量

把上面两节合起来就是容量规划的算术。以第六篇核心问题的 70B / TP=4 服务为例（数字是假设）：单副本饱和点 goodput 为 40 请求/s（在 TTFT ≤ 1 s、TPOT ≤ 50 ms 下），晚高峰需求 200 请求/s → 5 副本 → 20 张卡；平时需求 60 请求/s → 2 副本 → 8 张卡。差额 12 张卡 × 每天 6 小时低峰 = 72 GPU 小时/天，这就是缩容能省的钱（第七章第 4 节）；而每次扩容要 8 分钟，8 分钟内新增流量会越线——所以要提前触发，提前的代价是这 8 分钟的 12 张卡。SLO、饱和点、扩容时间、单价四个数决定了扩缩容参数，没有一个是拍脑袋的。


## 七、成本模型

### 1. 基础公式

```text
GPU 成本 = Σ_(卡, 时间) 单价(卡型, 计费方式) × 分配时长
```

分配时长按 Pod 的 `requests` 与生命周期算，不按使用率算——**一张卡分给了你就是你的成本，不管你算了没算**。这是 FinOps 里最重要的一条约定，因为只有这样闲置才会出现在某个团队的账上、才有人去处理它；按使用率计费会让闲置成为"公共成本"，没人负责。使用率作为第二列展示（"你花了 1000 GPU 小时，平均 SM_ACTIVE 30%"），用来施加压力，不用来定价。

单价是分层的：整卡按卡型（H100 / A100 / L4）每小时一个价；MIG 实例按 profile 折算（`1g.10gb` 大约是整卡的 1/7，是否严格按比例由平台决定）；HAMi 的显存 / 算力份额按比例；节点级的 CPU、内存、本地盘要么摊进 GPU 单价，要么单列。计费方式（按需 / 预留 / 自建折旧）让同一种卡有几个价——OpenCost 的定价模型里 `GPU` 与 `SpotGPU` 是两个键（`pkg/cloud/models/models.go` 的 `CustomPricing`），自建集群按折旧摊到小时。

### 2. 分摊维度

成本要能按团队、项目、模型、环境切分，维度来自 K8s 对象：

```text
维度        来源                                              说明
────────────────────────────────────────────────────────────────────────────────────────────────────
namespace   Pod 所在的 namespace                                最粗但最可靠；很多平台一个团队一个 namespace
Pod label   team / project / cost-center 一类的 label            要进 DCGM 的 podLabelAllowlistRegex 与 kube_pod_labels
队列        kueue.x-k8s.io/queue-name label；Volcano 的 queue     训练侧的天然维度；配额与账单用同一个键
模型        vLLM 的 model_name；InferencePool 名                  推理侧的天然维度；同一模型多版本时加 target_model_name
租户        EPP 的 fairness_id                                   请求层；只有 token 数，没有 GPU 小时——要靠第八章的换算
```

一条卡的 GPU 小时先按 `pod` 归到 Pod，再按 Pod 的 label 归到团队；一个推理副本的 GPU 小时归到模型，再按该模型各租户的 token 占比**二次分摊**到租户。共享成本（DCGM Exporter、Prometheus、网关自己占的资源）按各团队 GPU 小时的比例分摊，或者作为平台成本单列——OpenCost 的 `SharedCost` / `shareIdle` 参数就是这两种选择。

### 3. OpenCost 的 GPU 支持

OpenCost v1.121.1 的分配模型（`core/pkg/opencost/allocation.go` 的 `Allocation`）对 GPU 有这几个字段：`GPUHours`、`GPUCost`、`GPUCostAdjustment`、`GPUCostIdle`，以及 v1.121 引入的 `GPUAllocation` 结构（`GPUDevice`、`GPUModel`、`GPUUUID`、`IsGPUShared`、`GPUUsageAverage`、`GPURequestAverage`；旧的顶层 `GPURequestAverage` / `GPUUsageAverage` 已标 deprecated）。计算链路在 `pkg/costmodel/allocation_helpers.go`：

- **分配**：`applyGPUsAllocated` 用 kube-state-metrics 的 `kube_pod_container_resource_requests{resource="nvidia_com_gpu"}`（`modules/prometheus-source/pkg/prom/metricsquerier.go` 的 `queryFmtGPUsRequested`）或 OpenCost 自己导出的 `container_gpu_allocation`（`queryFmtGPUsAllocated`），`GPUHours = 请求数 × 小时`——注释写明"GPUHours reflects the full reserved GPU allocation (request × hours)"，即按分配不按使用；
- **定价**：`applyNodeCostPerGPUHr` 读 `node_gpu_hourly_cost{node,instance_type,provider_id}`（OpenCost 自己从云厂商定价 API 或 `configs/*.json` 的自定义价生成，`default.json` 里 `"GPU": "0.95"`），然后 `GPUCost = GPUHours × CostPerGPUHr`；
- **使用**：`applyGPUUsageAvg` / `applyGPUUsageMax` 从 `DCGM_FI_PROF_GR_ENGINE_ACTIVE{container!=""}` 按 `container, pod, namespace, pod_uid` 聚合（`queryFmtGPUsUsageAvg`），填进 `GPUUsageAverage`；它依赖 DCGM Exporter 开了 `--kubernetes` 映射且 Prometheus 里有 `pod_uid` 标签（`--kubernetes-enable-pod-uid`）；
- **共享**：`applyGPUUsageShared` 识别 `nvidia_com_gpu_shared` 资源名（GFD `renameByDefault=true` 时时间片副本的名字，第四篇）置 `IsGPUShared`；节点侧 `costmodel.go` 读 `nvidia.com/gpu.count` 标签与 `nvidia.com/gpu.shared` 容量区分物理卡数与副本数；
- **闲置**：`costmodel.go` 里 `gpuIdleCost = assetTotal.TotalGPUCost() − allocTotal.TotalGPUCost()`，即节点上 GPU 的总成本减去分配出去的，就是"没给出去"的那部分（`1 − A`）；分配出去但没算的那部分（`A(1 − U)`）不在 OpenCost 的闲置定义里，要用 `GPUUsageAverage` 自己算。

查询接口是 `/allocation/compute`（`pkg/costmodel/router.go`），参数 `window`、`aggregate`（支持 `namespace`、`label:team` 这样的形式，`core/pkg/opencost/allocationprops.go`）、`includeIdle`、`shareIdle`、`idleByNode`。**边界**：GPU 使用率用的是 `GR_ENGINE_ACTIVE` 而非 `SM_ACTIVE`；MIG 实例按 `nvidia_com_mig_*` 资源名不会被 `queryFmtGPUsRequested` 的 `resource="nvidia_com_gpu"` 过滤器匹配到（以 v1.121.1 源码为准），MIG 集群要么改资源名要么自己算；没有队列维度（它不认识 Kueue）；没有 token 维度。第十二章的 `allocate.py` 补这三样，其余照搬 OpenCost 的思路。

### 4. 三种隐性成本

账单上不会单列、但三个数字里能算出来的成本：

- **闲置成本**：`Σ 单价 × 分配时长 × (1 − SM_ACTIVE)`，按团队算。它是 `A − E` 那 50 个点的货币化，第十章分解它。
- **排队成本**：任务等资源的时间不占 GPU，但占人的时间与项目进度。度量是 `kueue_admission_wait_time_seconds` 的和乘以任务请求的卡数——"如果配额够，这些 GPU 小时本来可以在这段时间里产出"。它与闲置成本是同一枚硬币的两面：闲置在 A 团队的卡，就是 B 团队在排的队。两者同时高，说明配额（第三篇）切得不对或 cohort 借用没开。
- **冷启动成本**：推理副本从调度到就绪的时间里卡已分配但没服务；训练任务启动期同理。`(就绪时间 − 调度时间) × 卡数 × 单价 × 次数`。缩容到零省的是常驻成本，付的是每次冷启动——两者的比值决定该不该缩零（第六篇）。


## 八、每百万 token 的成本

### 1. 公式

推理的单位经济只有一个数：每百万输出 token 的成本。

$$
C_{1M} = \frac{P_{\text{GPU}} \times N_{\text{GPU}}}{T \times 3600 \times U} \times 10^{6}
$$

$$P_{\text{GPU}}$$ 是每卡每小时单价，$$N_{\text{GPU}}$$ 是一个副本的卡数，$$T$$ 是该副本满载时的输出吞吐（token/s），$$U$$ 是实际负载相对满载的比例（即副本的平均"忙"程度，不是 `SM_ACTIVE`）。三个因子分别由采购（价）、引擎（吞吐）、平台（利用率）决定——**平台能动的只有 $$U$$**，但它常常是三个里最差的那个。

用第六章的算例（全部为假设）：H100 每卡每小时 2.5 美元，TP=4 的 70B 服务一个副本 10 美元/小时；满载输出吞吐 2,000 token/s（假设值，随模型、量化、批大小变化很大）；则满载时 $$C_{1M} = 10 / (2000 \times 3600) \times 10^6 \approx 1.39$$ 美元。$$U = 40\%$$（按峰值常驻、平均负载四成）时 $$C_{1M} \approx 3.47$$ 美元。**同一套硬件、同一个引擎，利用率从 100% 掉到 40%，每百万 token 贵 2.5 倍**。输入 token 通常另计一个更低的价（prefill 吞吐远高于 decode），公式同构。

### 2. 三个杠杆

- **量化**：FP8 / INT8 权重让同一张卡的权重占用减半、KV cache 空间增加、decode 的带宽需求下降，$$T$$ 上升（幅度以引擎实测为准，通常在 1.3–2 倍之间）；代价是精度评测与两个版本的灰度（第七篇）。$$C_{1M}$$ 按 $$T$$ 的倍数下降。
- **PD 分离**：prefill 与 decode 放在不同的 Pod 组（第六篇的 `DisaggregatedSet` 形态），各自按自己的饱和点扩缩。它不一定降低单副本的 $$T$$，但**把 goodput 提上去**——decode 不再被突发的长 prefill 打断，TPOT 稳定，同样的卡数能承诺更严的 SLO；也让两侧可以用不同的卡型（prefill 用算力强的、decode 用带宽大的），改变 $$P_{\text{GPU}}$$ 的构成。
- **批大小**：并发上限（vLLM 的 `--max-num-seqs`、`--max-num-batched-tokens`，以 v0.23.0 文档为准）越大 $$T$$ 越高，但 TPOT 随之上升；在 SLO 内取最大批就是第六章的饱和点。`vllm:iteration_tokens_total` 直方图显示实际每步的 token 数，与配置的上限比就知道批有没有填满——填不满是负载不够（$$U$$ 的问题），不是引擎的问题。

### 3. 与 API 定价的量级对比

把 $$C_{1M}$$ 与 API 厂商的每百万 token 报价放在一起，只能比**量级**：厂商报价里有毛利、有多租户的高利用率、有自研 kernel 与专用硬件，也有大量的免费额度与阶梯价；自建的数字里没有人力、没有机房、没有网络出口。一个粗略的判断标准（作为假设而非结论）：自建 $$C_{1M}$$ 在满载时低于 API 报价一个量级左右，才可能在 40%–60% 的真实利用率下与 API 打平。低于这个差距，自建的理由要从数据合规、延迟、定制模型里找，不是成本。这个对比每季度做一次——API 价格在降，硬件价格也在降，结论会变。


## 九、FinOps 回路与容量规划

### 1. 成本回流到配置

FinOps 的"回路"指成本数据改变前七篇的参数，而不是只出一张报表：

```text
观察（第三、七章的数字）                                  动作                                   改的是哪篇的机制
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
某 ClusterQueue 的 resource_usage / nominal_quota 长期 < 30%   缩它的 nominalQuota；开 cohort 借用；    第三篇：ClusterQueue 配额与 cohort
且其他队列 pending_workloads 持续非零                          或把 borrowingLimit 放开
某推理服务 FB_USED 长期 < 30 GB、SM_ACTIVE < 25%              迁到 MIG 3g.40gb 或 HAMi 按显存切；       第四篇：切分与隔离
                                                             整卡让给训练
某服务 E 有明显日夜周期、低峰 U < 15%                          KEDA 阈值下调、minReplicaCount 降到 1     第六篇：扩缩容与冷启动
                                                             或 0；缩零前算冷启动成本
某 namespace 全天 A 不变、E ≈ 0                               开发环境改时间片；notebook 加空闲超时回收    第四篇时间片；第三篇配额与超时
训练 job 的 checkpoint 竖条宽度 × 频率 > 5% 时间               异步 checkpoint；存储带宽扩容；间隔拉长      第五篇：checkpoint I/O
某租户 token 占比高但 SLO 达成率低                             提高该租户的 priority 或独立 InferencePool    第七篇：租户优先级与配额
```

每一行的左边是一条 recording rule 或一个看板面板，右边是一个 PR。回路的周期按月：月初出账单与分解表，月中改配置，月末看三个数字的变化。

### 2. 容量规划

采购或预留的决策来自历史负载曲线的三个统计量：**A 的 p95**（多少卡在 95% 的时间里是分配出去的——这是"基础容量"，适合预留 / 自建）、**A 的峰值与 p95 的差**（突发部分，适合按需）、**`kueue_pending_workloads` 的积分**（被压抑的需求——容量不够时排队会把真实需求藏起来，规划时要把它加回去）。

训练与推理的错峰是 GPU 平台特有的机会：推理的低峰（夜间）正是训练可以借卡的时段。Kueue 的 cohort 借用与抢占（第三篇）能把推理队列夜间释放的配额借给训练队列，前提是训练任务能被抢占并从 checkpoint 恢复（第五篇），且推理的扩缩容真的在夜间缩了副本（第六篇）。做到这一步，E 的日夜周期会被填平，A 与 E 的差距缩小——这是本系列八篇机制全部到位后才能出现的形状。


## 十、核心问题：50 个百分点去了哪里

回到总纲的问题：64 卡、A = 85%、E = 35%。先把两个数的关系说清：15 个点没分配出去（`1 − A`），50 个点分配了但没算（`A − E`）。下面的分解是**一种典型的构成**，每个团队的比例不同，但每一项都有一个可以测量的指标——分解的意义在于每一项都能被单独观察、单独归到一篇：

```text
去向                          典型占比   怎么测（Prometheus）                                                           对应篇 / 机制
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
A − E = 50 个点（分配了但没算）
推理低峰空转                  ~15       推理 namespace 的 (A − E) 在夜间 / 周末的积分；vllm:num_requests_running 接近 0     第六篇：KEDA 阈值 · minReplica · 缩零；
                                        而 Pod 仍 Running                                                                  第四篇：低峰改切分
开发环境与 notebook 长期占用   ~10       dev namespace 的 A 全天恒定、SM_ACTIVE < 5%；FB_USED 只有 CUDA context 的几 GB         第四篇：时间片；第三篇：配额 + 空闲超时
训练的通信与数据等待           ~10       训练 job 的 TENSOR_ACTIVE / SM_ACTIVE 比值低；train_comm_wait_ratio；                第五篇：RDMA 是否生效 · NCCL 走了哪条路；
                                        NVLINK / PCIE 流量呈周期性                                                          引擎内部的并行策略（本系列边界外）
排队占位（部分调度的 gang）    ~5        没有 gang 时：job 的部分 Pod Running 而其余 Pending，Running 的 SM_ACTIVE = 0；        第三篇：gang scheduling · Kueue 准入
                                        有 Kueue 时应为 0，否则查 AdmissionCheck 与 Pod 启动失败
checkpoint 写入停顿           ~3        train_ckpt_seconds × 频率 × 卡数；GPU 时间线上的周期性竖条                            第五篇：异步 DCP · 存储写带宽
启动与冷启动                  ~2        Pod 从 Scheduled 到 Ready 的时间 × 卡数（kube_pod_status_ready 与 scheduled 时间差）；   第二篇：镜像体积 · 预热；第五篇：权重加载；
                                        推理副本扩容期间 vllm:* 尚无数据                                                    第六篇：扩容时间分解
测量上限（不可回收）           ~5        memory-bound 的 decode 与小 batch 下 SM_ACTIVE 本来到不了 100%；                     不属于任何一篇：换 DRAM_ACTIVE 看 decode，
                                        DRAM_ACTIVE 高而 SM_ACTIVE 低                                                       这部分不是浪费
1 − A = 15 个点（没分配出去）
碎片                          ~6        节点上 allocatable − requests 的余量之和，且这些余量凑不出任何 pending 任务的需求         第三篇：binpack · 拓扑感知 · 队列的 flavor 划分
隔离的坏节点                  ~3        cordoned / tainted 节点上的 GPU 数；XID 告警后未恢复的节点                            第二篇：驱动 · 第五章第 4 节的隔离动作
预留与配额过紧                ~6        kueue_cluster_queue_nominal_quota − resource_usage 的和大于 0 且 pending 非零           第三篇：配额与 cohort 借用
```

四个观察。第一，**推理低峰与开发环境合起来占了一半**，它们是最容易回收的——参数级的改动（KEDA 阈值、时间片、空闲超时），不需要改架构。第二，**训练的通信等待是平台与引擎共同的责任**：平台保证 RDMA 生效、拓扑对齐（第三、五篇），剩下的比例由并行策略决定，平台看板上要把这两部分分开（RDMA 没生效时 `NVLINK_BANDWIDTH_TOTAL` 低而 PCIe 流量高，是平台的问题；RDMA 生效仍等待，是引擎的问题）。第三，**有 5 个点不是浪费**——用 `SM_ACTIVE` 做统一尺度的代价是它低估 memory-bound 负载，decode 服务要用 `DRAM_ACTIVE` 复核；把"不可回收"的部分明确标出来，团队才不会追一个到不了的目标。第四，**`1 − A` 那 15 个点里有 6 个是配额设置**，不是硬件——配额过紧的队列在账单上表现为"闲置"，在 Kueue 上表现为别的队列在排队，这两个信号要放在同一张看板上才会被同时看到。

分解完之后，每一项都成了前面某一篇的一个待办。这就是可观测层的作用：它自己不省一张卡，但它决定了前七篇的哪个开关该动。


## 十一、代价与边界

可观测与成本核算不是免费的，也不是精确的。

**采集开销**。DCGM 的 `PROF_*` 字段占用硬件性能计数器，与 Nsight 一类 profiler 互斥——做 kernel 级 profiling 的那几张卡要临时关掉 DCGM 的 profiling 字段。DCGM Exporter 默认 30 秒的采样对成本核算足够，对捕捉几秒的 straggler 抖动不够；缩到 5 秒会让 DCGM host engine 的 CPU 占用上升，也让时序量翻六倍。vLLM 的 `/metrics` 每次抓取要序列化几百条时序，QPS 很高的副本上抓取本身有可测的开销；训练进程暴露指标要么占一个端口（千个 rank 千个目标），要么写文件让 sidecar 转（多一个进程）。

**基数与保留**。第四章的三条基数陷阱每一条都有真实的事故记录。分摊维度越细、保留越长，存储越贵；13 个月的原始 DCGM 时序在几百节点的集群上是 TB 级，必须靠 recording rules 降采样后再长期保留，而降采样后就查不到当时某张卡的抖动了。**告警用原始、账单用聚合**，两边的数字要能对上，对不上时以原始为准。

**归因的误差**。DCGM 的 `pod` 标签来自 kubelet 的 pod-resources API，一张卡在两个 Pod 之间切换的那几十秒里可能归错；时间片与 HAMi 共卡时（第四篇）一张卡的 `SM_ACTIVE` 归到多个 Pod，DCGM 4.x 用 `vgpu` 属性区分副本但**没有按进程拆分活跃度的能力**——共卡的使用率只能按份额平摊或按 `FB_USED` 比例估。MIG 实例有独立的 `GPU_I_ID`，归因是准确的；这也是 MIG 在多租户计费上比软件切分更受欢迎的原因之一。EPP 的 `fairness_id` 超过 1000 个折叠为 `other`。OpenCost 用 `GR_ENGINE_ACTIVE`，与本篇的 `SM_ACTIVE` 数值不同。

**成本模型的假设**。单价是输入不是输出——自建集群的折旧口径、电费、机房、网络分摊都是财务的约定，换一种口径每百万 token 成本可以差一倍；"按分配计费"是一个政策选择，它让闲置有主，但也让被平台调度到坏节点上等了两小时的任务替平台买单——这类情况要靠 `GPUCostAdjustment` 一类的人工冲销。每百万 token 成本里的 $$T$$ 依赖压测条件（输入 / 输出长度分布），真实流量与压测分布不同时它就不准。

**边界**：可观测层不解释引擎内部——MFU 为什么是 42%、KV cache 为什么碎片化、哪个 kernel 慢，是训练与推理引擎自己的 profiler 的事；平台只把"哪张卡、哪个任务、什么时候、多满"这四个问题回答到 Pod 级。它也不替代财务系统——它产出的是按团队的 GPU 小时与 token 数，定价、结算、预算是财务的流程。


## 十二、本文小结

### 1. 要点回顾

```text
四层指标      硬件（DCGM_FI_DEV_* / PROF_*，每卡一份）· 容器 Pod（kube_pod_* / kueue_*）· 引擎（vllm:* / 训练约定集）· 请求（llm_d_epp_*，按 fairness_id）
              四层靠 pod / namespace 标签 join；DCGM 的 --kubernetes 映射（kubelet pod-resources）是链条的关键一环
GPU_UTIL      只表示"有 kernel 在跑"；SM_ACTIVE（通用）· TENSOR_ACTIVE（训练 / prefill）· DRAM_ACTIVE（decode）才接近负载；
              SM_ACTIVE / SM_OCCUPANCY 在默认 CSV 中被注释，要用 customMetrics 全量覆盖
三个数字      A 分配率（kube-state-metrics 或 count(DCGM{pod!=""})）· U 已分配卡的 SM_ACTIVE 均值 · E 全部卡的均值 ≈ A × U；
              1 − A 是没给出去的，A − E 是给出去没算的；日夜周期 / 阶梯 / 恒定各指向一篇
管线          agent 模式 + 中心长期存储（成本要 13 个月）；基数：Pod label allowlist、请求级标签禁止、EPP 的 1000 上限折叠为 other；
              OTel trace 网关 → EPP → 引擎，tail-based 保留慢请求；XID / NCCL 走日志正则 → 告警 → 自动隔离
训练          job 级 SM_ACTIVE 热力图的四种形状；每 rank step 时间 topk 找 straggler → 落到 SM_CLOCK / PCIE_REPLAY / XID；
              checkpoint 竖条 = GPU 小时；通信比用 TENSOR/SM 比值与链路流量间接看；引擎是黑盒，只约定最小指标集
推理          SLO = TTFT p95 + TPOT p95；goodput 以 EPP 侧为准；饱和点压测一次，决定 KEDA 阈值与副本数；量化 / PD 分离 / 命中率变了要重测
成本          按分配计费（闲置有主）；维度 namespace / label / 队列 / 模型 / 租户；OpenCost：GPUHours = request × hours，
              GPUCost = GPUHours × node_gpu_hourly_cost，使用率用 GR_ENGINE_ACTIVE，闲置 = 资产 − 分配；无队列、无 token、MIG 需自算
每百万 token   C = P × N / (T × 3600 × U) × 1e6；平台只能动 U；U 从 100% 到 40% 贵 2.5 倍；与 API 只比量级
回路          队列使用率 → 配额；FB_USED / SM_ACTIVE 双低 → 切分；日夜周期 → KEDA / 缩零；dev 恒定 → 时间片 + 超时；按月闭环
50 个点        推理低峰 ~15 · dev ~10 · 通信等待 ~10 · 排队占位 ~5 · checkpoint ~3 · 冷启动 ~2 · 测量上限 ~5（不可回收）；
              1 − A 的 15：碎片 ~6 · 坏节点 ~3 · 配额过紧 ~6
```

### 2. 引擎需求 → K8s 空缺 → 平台机制 → 代价

```text
引擎的需求                         K8s 的空缺                          平台的机制                                      代价
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
硬件状态能归到任务                  扩展资源是不透明整数；cAdvisor 无 GPU   DCGM Exporter --kubernetes 查 pod-resources；      共卡时归因不准；Pod 切换窗口误差；
                                                                       Pod label allowlist 进指标                        label 基数
"在算还是在等"可区分               GPU_UTIL 语义只是"非空闲"              PROF_* 字段（SM / TENSOR / DRAM_ACTIVE）            占用性能计数器，与 profiler 互斥；
                                                                                                                        memory-bound 负载被 SM_ACTIVE 低估
按引擎指标决策与定 SLO              HPA 只认 CPU / 内存                   vllm:* 直方图 + KEDA / Prometheus Adapter；          直方图桶固定；每副本几百条时序
                                                                       EPP 的 llm_d_epp_* 作为用户视角
按租户计 token                     无请求级观测                          EPP fairness_id；网关访问日志对账                   基数上限 1000 折叠；日志管线另建
一个请求跨三跳可追踪               无                                    OTel trace + tail-based sampling                    存储成本高，只能低采样
训练 hang / straggler 可定位        无                                    约定的 train_* 指标 + 每 rank 标签 + 日志正则         千 rank 千目标或 sidecar；引擎需配合暴露
GPU 有价、闲置有主                  无成本概念                            按分配计费 × 分层单价；OpenCost 或自建分摊              单价是约定；被动闲置需人工冲销；
                                                                                                                        OpenCost 缺队列 / token / MIG
成本能回流到配置                    无                                    看板 → 每月改 Kueue 配额 / MIG / KEDA 参数            需要人来闭环；改动有滞后
```

### 3. 本篇涉及的源码位置

```text
项目                        路径                                                    关键符号 / 内容
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
DCGM Exporter 4.6.0-4.8.3   etc/default-counters.csv · etc/dcp-metrics-included.csv  默认字段表；SM_ACTIVE / SM_OCCUPANCY / ECC / NVLINK 错误计数为注释行
                            pkg/cmd/app.go                                          CLIFieldsFile("collectors") · CLIKubernetes · CLIKubernetesEnablePodLabels ·
                                                                                    CLIKubernetesEnablePodUID · CLIKubernetesGPUIDType · CLIKubernetesPodLabelAllowlistRegex；
                                                                                    环境变量 DCGM_EXPORTER_KUBERNETES / _ENABLE_POD_LABELS / _INTERVAL / _COLLECTORS
                            internal/pkg/appconfig/types.go · const.go              DefaultCollectorsFile；KubernetesGPUIDType 的 uid / device-name
                            internal/pkg/transformation/kubernetes.go               PodMapper · listPods（PodResourcesListerClient.List）· availablePodLabelName · toDeviceToPodsDRA
                            internal/pkg/transformation/const.go                    podAttribute / namespaceAttribute / containerAttribute / uidAttribute / vgpuAttribute / podLabelPrefix / dra_*
                            internal/pkg/rendermetrics/render_metrics.go            FE_GPU 标签：gpu · UUID · pci_bus_id · device · modelName · GPU_I_PROFILE · GPU_I_ID · hostname
                            internal/pkg/utils/utils.go                             SanitizeLabelName
                            deployment/values.yaml · templates/daemonset.yaml       kubernetes.enablePodLabels / podLabelAllowlistRegex / rbac · kubeletPath · customMetrics · serviceMonitor · arguments
Kubernetes v1.37.0          pkg/kubelet/cm/devicemanager/manager.go                 ManagerImpl.GetDevices(podUID, containerName)（pod-resources List 的设备来源）
vLLM v0.23.0                vllm/v1/metrics/loggers.py                              PrometheusStatLogger：全部 vllm:* 名字与桶；labelnames = [model_name, engine]；create_metric_per_engine
                            vllm/v1/metrics/perf.py                                 vllm:estimated_flops_per_gpu_total 等估算指标（本篇未用）
                            docs/design/metrics.md                                  Counter 的 _total 后缀；ITL 即 TPOT 的说明
Kueue v0.19.2               pkg/metrics/metrics.go                                  kueue_pending_workloads{cluster_queue,status} · admitted_active_workloads · admission_wait_time_seconds ·
                                                                                    cluster_queue_resource_usage / _nominal_quota / _resource_reservation / _resource_pending
                            pkg/constants/constants.go                              KueueName = "kueue"（Subsystem）
llm-d-router v0.10.0        pkg/epp/metrics/llm_d_router_metrics.go                 LLMDRouterEndpointPickerSubsystem = "llm_d_epp"；request_total · request_input/output_tokens ·
                                                                                    request_ttft_seconds · request_streaming_tpot_seconds · average_kv_cache_utilization · ready_endpoints · plugin_duration_seconds
                            pkg/epp/metrics/metrics.go                              inference_objective_* / inference_pool_*（Deprecated 注释）
                            pkg/epp/metrics/cardinality.go                          maxModelLabelValues / maxFairnessLabelValues = 1000 · overflowValue = "other" · boundFairnessID
                            pkg/epp/metadata/consts.go                              FlowFairnessIDKey · TTFTSLOHeaderKey
OpenCost v1.121.1           core/pkg/opencost/allocation.go                         Allocation：GPUHours · GPUCost · GPUCostAdjustment · GPUCostIdle · GPUAllocation{IsGPUShared, GPUUsageAverage, GPURequestAverage}
                            core/pkg/opencost/allocationprops.go                    AllocationLabelProp（"label:"）
                            pkg/costmodel/allocation_helpers.go                     applyGPUsAllocated（GPUHours = request × hours）· applyGPUUsageAvg / Max · applyGPUUsageShared · applyNodeCostPerGPUHr（GPUCost）
                            pkg/costmodel/costmodel.go                              nvidia.com/gpu.count / gpu.shared 的节点解析；gpuIdleCost = 资产 − 分配
                            pkg/costmodel/metrics.go · router.go · aggregation.go   node_gpu_hourly_cost · container_gpu_allocation；/allocation/compute 的 aggregate / includeIdle / shareIdle
                            modules/prometheus-source/pkg/prom/metricsquerier.go    queryFmtGPUsRequested（nvidia_com_gpu）· queryFmtGPUsUsageAvg（DCGM_FI_PROF_GR_ENGINE_ACTIVE）· queryFmtNodeCostPerGPUHr
                            pkg/cloud/models/models.go · configs/default.json       CustomPricing 的 GPU / SpotGPU 键；默认 "GPU": "0.95"
```

### 4. 练手项目：mini-platform 的 obs/ 与 cost/

本篇给 `mini-platform/` 加两个目录。硬件要求与前几篇相同（有 GPU 节点即可；MIG 相关的分配率算法需要 MIG 卡）。

**`obs/dcgm-values.yaml`**——dcgm-exporter Helm chart（与 4.6.0-4.8.3 镜像同版本）的 values 片段，开 Pod label 映射并把 `SM_ACTIVE` / `SM_OCCUPANCY` 从注释里放出来。`customMetrics` 必须是完整表，这里只列本篇用到的字段（省略号处按 `etc/default-counters.csv` 补齐其余需要的行）：

```yaml
# mini-platform/obs/dcgm-values.yaml —— helm upgrade -i dcgm-exporter gpu-helm-charts/dcgm-exporter -n gpu-operator -f obs/dcgm-values.yaml
image:
  tag: 4.6.0-4.8.3-distroless
arguments:
  - "--kubernetes-gpu-id-type"
  - "uid"
kubernetes:
  enablePodLabels: true
  enablePodUID: true              # OpenCost 的 DCGM 查询按 pod_uid 聚合
  podLabelAllowlistRegex:
    - "^team$"
    - "^project$"
    - "^kueue\\.x-k8s\\.io/queue-name$"
    - "^jobset\\.sigs\\.k8s\\.io/jobset-name$"
    - "^app\\.kubernetes\\.io/name$"
  rbac:
    create: true
serviceMonitor:
  enabled: true
  interval: 15s
  relabelings:
    - sourceLabels: [__meta_kubernetes_pod_node_name]
      targetLabel: nodename
      action: replace
customMetrics: |
  # 完整表（非增量）。以下为本篇用到的字段；其余按 etc/default-counters.csv 补齐
  DCGM_FI_DEV_SM_CLOCK,  gauge, SM clock frequency (in MHz).
  DCGM_FI_DEV_MEM_CLOCK, gauge, Memory clock frequency (in MHz).
  DCGM_FI_DEV_GPU_TEMP,    gauge, GPU temperature (in C).
  DCGM_FI_DEV_MEMORY_TEMP, gauge, Memory temperature (in C).
  DCGM_FI_DEV_POWER_USAGE, gauge, Power draw (in W).
  DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION, counter, Total energy consumption since boot (in mJ).
  DCGM_FI_DEV_PCIE_REPLAY_COUNTER, counter, Total number of PCIe retries.
  DCGM_FI_DEV_GPU_UTIL,      gauge, GPU utilization (in %).
  DCGM_FI_DEV_MEM_COPY_UTIL, gauge, Memory utilization (in %).
  DCGM_FI_DEV_XID_ERRORS,    gauge, Value of the last XID error encountered.
  DCGM_FI_DEV_FB_FREE, gauge, Framebuffer memory free (in MiB).
  DCGM_FI_DEV_FB_USED, gauge, Framebuffer memory used (in MiB).
  DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL, gauge, Total number of NVLink bandwidth counters for all lanes.
  DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS, counter, Number of remapped rows for uncorrectable errors
  DCGM_FI_DEV_ROW_REMAP_FAILURE, gauge, Whether remapping of rows has failed
  DCGM_FI_PROF_GR_ENGINE_ACTIVE,   gauge, Ratio of time the graphics engine is active.
  DCGM_FI_PROF_SM_ACTIVE,          gauge, The ratio of cycles an SM has at least 1 warp assigned.
  DCGM_FI_PROF_SM_OCCUPANCY,       gauge, The ratio of number of warps resident on an SM.
  DCGM_FI_PROF_PIPE_TENSOR_ACTIVE, gauge, Ratio of cycles the tensor (HMMA) pipe is active.
  DCGM_FI_PROF_DRAM_ACTIVE,        gauge, Ratio of cycles the device memory interface is active sending or receiving data.
  DCGM_FI_PROF_PCIE_TX_BYTES, gauge, The rate of data transmitted over the PCIe bus in bytes per second.
  DCGM_FI_PROF_PCIE_RX_BYTES, gauge, The rate of data received over the PCIe bus in bytes per second.
```

装完后 `curl` 任一 DCGM Pod 的 `:9400/metrics`，应当看到 `DCGM_FI_PROF_SM_ACTIVE{gpu="0",UUID="GPU-…",…,pod="…",namespace="…",container="…",team="…"}`——`team` 只在 Pod 打了这个 label 时出现，没有 GPU Pod 的卡上 `pod=""`。GPU Operator 管理的 DCGM Exporter 用 `ClusterPolicy.spec.dcgmExporter` 的对应字段（第二篇），机制相同。

**`obs/prometheus-rules.yaml`**——三个数字的 recording rules 与三条告警（`monitoring.coreos.com/v1` 的 `PrometheusRule`，Prometheus Operator 通行 CRD）：

{% raw %}
```yaml
# mini-platform/obs/prometheus-rules.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: mini-platform-gpu
  namespace: monitoring
  labels:
    release: kube-prometheus-stack
spec:
  groups:
    - name: platform.gpu.three-numbers
      interval: 1m
      rules:
        - record: platform:gpu_allocatable:count
          expr: sum(kube_node_status_allocatable{resource="nvidia_com_gpu"})
        - record: platform:gpu_allocated:count
          expr: |
            sum(
              kube_pod_container_resource_requests{resource="nvidia_com_gpu"}
              * on (namespace, pod) group_left ()
              (kube_pod_status_phase{phase="Running"} == 1)
            )
        - record: platform:gpu_allocated_ratio
          expr: platform:gpu_allocated:count / platform:gpu_allocatable:count
        - record: platform:gpu_allocated_ratio:dcgm
          expr: count(DCGM_FI_DEV_GPU_UTIL{pod!=""}) / count(DCGM_FI_DEV_GPU_UTIL)
        - record: platform:gpu_sm_active:avg_allocated
          expr: avg(DCGM_FI_PROF_SM_ACTIVE{pod!=""})
        - record: platform:gpu_sm_active:avg_all
          expr: avg(DCGM_FI_PROF_SM_ACTIVE)
        # 按团队与队列：分配卡数与已分配卡的 SM_ACTIVE
        - record: platform:gpu_allocated:count_by_team
          expr: count by (team) (DCGM_FI_DEV_GPU_UTIL{pod!=""})
        - record: platform:gpu_sm_active:avg_by_team
          expr: avg by (team) (DCGM_FI_PROF_SM_ACTIVE{pod!=""})
        - record: platform:gpu_allocated:count_by_queue
          expr: count by (kueue_x_k8s_io_queue_name) (DCGM_FI_DEV_GPU_UTIL{pod!="", kueue_x_k8s_io_queue_name!=""})
        - record: platform:gpu_sm_active:avg_by_queue
          expr: avg by (kueue_x_k8s_io_queue_name) (DCGM_FI_PROF_SM_ACTIVE{pod!="", kueue_x_k8s_io_queue_name!=""})
        # 队列配额使用率（Kueue）
        - record: platform:kueue_quota_usage_ratio
          expr: |
            sum by (cluster_queue, resource) (kueue_cluster_queue_resource_usage)
            / sum by (cluster_queue, resource) (kueue_cluster_queue_nominal_quota)
        # 推理：TTFT p95 与 token 速率
        - record: platform:vllm_ttft_seconds:p95
          expr: histogram_quantile(0.95, sum by (le, model_name) (rate(vllm:time_to_first_token_seconds_bucket[5m])))
        - record: platform:vllm_generation_tokens:rate5m
          expr: sum by (model_name) (rate(vllm:generation_tokens_total[5m]))
    - name: platform.gpu.alerts
      rules:
        - alert: GpuXidError
          expr: DCGM_FI_DEV_XID_ERRORS > 0
          for: 0m
          labels:
            severity: page
          annotations:
            summary: "XID {{ $value }} on {{ $labels.hostname }} gpu {{ $labels.gpu }} (pod {{ $labels.namespace }}/{{ $labels.pod }})"
            runbook: "cordon 节点；dmesg 确认 XID；通知该 pod 所属 job 重启"
        - alert: KueueQueueDepthHigh
          expr: sum by (cluster_queue) (kueue_pending_workloads{status="active"}) > 3
          for: 30m
          labels:
            severity: record
          annotations:
            summary: "{{ $labels.cluster_queue }} 有 {{ $value }} 个 workload 排队超过 30 分钟"
        - alert: InferenceTTFTP95High
          expr: platform:vllm_ttft_seconds:p95 > 2
          for: 10m
          labels:
            severity: page
          annotations:
            summary: "{{ $labels.model_name }} TTFT p95 = {{ $value }}s，超过 SLO 2s"
        - alert: GpuAllocatedButIdle
          expr: |
            (avg by (namespace, pod) (DCGM_FI_PROF_SM_ACTIVE{pod!=""}) < 0.02)
            and on (namespace, pod) (kube_pod_status_phase{phase="Running"} == 1)
          for: 2h
          labels:
            severity: record
          annotations:
            summary: "{{ $labels.namespace }}/{{ $labels.pod }} 占卡 2 小时而 SM_ACTIVE < 2%"
```
{% endraw %}

**`obs/grafana-dashboard.json`**——一张看板，面板按第三章的思路排：三个数字同屏、按团队分解、训练热力图、推理 SLO、队列。这里只给面板列表与各自的 PromQL（完整 JSON 由读者在 Grafana 里按此建好后导出；`legendFormat` 里的模板已按 Jekyll 要求包在 raw 里）：

{% raw %}
```json
{
  "title": "mini-platform / GPU 四层看板",
  "panels": [
    {"title": "分配率 A / 使用率 U / 有效利用率 E", "type": "timeseries",
     "targets": [
       {"expr": "platform:gpu_allocated_ratio", "legendFormat": "A 分配率"},
       {"expr": "platform:gpu_sm_active:avg_allocated", "legendFormat": "U 已分配卡 SM_ACTIVE"},
       {"expr": "platform:gpu_sm_active:avg_all", "legendFormat": "E 全部卡 SM_ACTIVE"}]},
    {"title": "GPU_UTIL vs SM_ACTIVE vs TENSOR_ACTIVE（集群均值）", "type": "timeseries",
     "targets": [
       {"expr": "avg(DCGM_FI_DEV_GPU_UTIL) / 100", "legendFormat": "GPU_UTIL"},
       {"expr": "avg(DCGM_FI_PROF_SM_ACTIVE)", "legendFormat": "SM_ACTIVE"},
       {"expr": "avg(DCGM_FI_PROF_PIPE_TENSOR_ACTIVE)", "legendFormat": "TENSOR_ACTIVE"},
       {"expr": "avg(DCGM_FI_PROF_DRAM_ACTIVE)", "legendFormat": "DRAM_ACTIVE"}]},
    {"title": "按团队：分配卡数", "type": "timeseries",
     "targets": [{"expr": "platform:gpu_allocated:count_by_team", "legendFormat": "{{team}}"}]},
    {"title": "按团队：已分配卡的 SM_ACTIVE", "type": "timeseries",
     "targets": [{"expr": "platform:gpu_sm_active:avg_by_team", "legendFormat": "{{team}}"}]},
    {"title": "训练 job 的 GPU 时间线（SM_ACTIVE 热力图）", "type": "heatmap",
     "targets": [{"expr": "DCGM_FI_PROF_SM_ACTIVE{jobset_sigs_k8s_io_jobset_name=\"$job\"}",
                  "legendFormat": "{{pod}}/{{gpu}}"}]},
    {"title": "Kueue：排队与配额使用率", "type": "timeseries",
     "targets": [
       {"expr": "sum by (cluster_queue) (kueue_pending_workloads{status=\"active\"})", "legendFormat": "pending {{cluster_queue}}"},
       {"expr": "platform:kueue_quota_usage_ratio{resource=\"nvidia.com/gpu\"}", "legendFormat": "usage/quota {{cluster_queue}}"}]},
    {"title": "推理：TTFT p95 / TPOT p95", "type": "timeseries",
     "targets": [
       {"expr": "platform:vllm_ttft_seconds:p95", "legendFormat": "TTFT p95 {{model_name}}"},
       {"expr": "histogram_quantile(0.95, sum by (le, model_name) (rate(vllm:request_time_per_output_token_seconds_bucket[5m])))",
        "legendFormat": "TPOT p95 {{model_name}}"}]},
    {"title": "推理：排队请求 / KV cache 占用（按副本）", "type": "timeseries",
     "targets": [
       {"expr": "vllm:num_requests_waiting", "legendFormat": "waiting {{pod}}"},
       {"expr": "vllm:kv_cache_usage_perc", "legendFormat": "kv {{pod}}"}]},
    {"title": "推理：输出 token/s 与 prefix cache 命中率", "type": "timeseries",
     "targets": [
       {"expr": "platform:vllm_generation_tokens:rate5m", "legendFormat": "tok/s {{model_name}}"},
       {"expr": "sum by (model_name) (rate(vllm:prefix_cache_hits_total[5m])) / sum by (model_name) (rate(vllm:prefix_cache_queries_total[5m]))",
        "legendFormat": "hit {{model_name}}"}]},
    {"title": "请求层：按租户的 token 速率（EPP）", "type": "timeseries",
     "targets": [{"expr": "sum by (fairness_id) (rate(llm_d_epp_request_output_tokens_sum[5m]))", "legendFormat": "{{fairness_id}}"}]},
    {"title": "硬件：XID / 降频 / 温度", "type": "table",
     "targets": [
       {"expr": "DCGM_FI_DEV_XID_ERRORS > 0", "legendFormat": "xid"},
       {"expr": "DCGM_FI_DEV_SM_CLOCK < 1300", "legendFormat": "clock"},
       {"expr": "DCGM_FI_DEV_GPU_TEMP > 85", "legendFormat": "temp"}]}
  ],
  "templating": {"list": [{"name": "job", "type": "query",
    "query": "label_values(DCGM_FI_PROF_SM_ACTIVE, jobset_sigs_k8s_io_jobset_name)"}]}
}
```
{% endraw %}

看板上先看第一块：三条线之间的两段空隙就是第十章要分解的东西。把前几篇部署的训练任务（第三篇的 DDP）与推理服务（第六、七篇的 vLLM + 网关）跑起来，第一块会出现明显的形状——训练任务启动的几分钟里 A 上升而 E 不动，推理服务在压测停止后 A 不变而 E 落到接近零。

**`cost/allocate.py`**——从 Prometheus 拉一个时间窗内按 label 的分配时序与 token 计数，配一张单价表，输出每团队的 GPU 小时、成本、闲置比例，以及每模型 / 每租户的每百万 token 成本。它做的是 OpenCost 分配模型的一个子集（按分配计费、按 label 聚合），加上 OpenCost 没有的三样：`SM_ACTIVE` 作为使用率、队列维度、token 维度。

{% raw %}
```python
#!/usr/bin/env python3
"""mini-platform/cost/allocate.py —— 按 label 分摊 GPU 成本，并算每百万 token 成本。

用法：
  python3 allocate.py --prom http://prometheus:9090 --start 2026-09-01T00:00:00Z --end 2026-09-08T00:00:00Z \
      --team-label team --prices prices.json [--tenant-metric llm_d_epp_request_output_tokens_sum]

prices.json（单价是假设值，不是报价）：
  {"default": 2.5, "NVIDIA H100 80GB HBM3": 2.5, "NVIDIA A100-SXM4-80GB": 1.4,
   "mig": {"1g.10gb": 0.14, "3g.40gb": 0.43}}
数据源：DCGM Exporter（--kubernetes 与 pod label 已开）、vLLM /metrics、EPP /metrics，均已被 Prometheus 抓取。
"""
import argparse
import json
import sys
import urllib.parse
import urllib.request
from collections import defaultdict


def query_range(prom, expr, start, end, step):
    q = urllib.parse.urlencode({"query": expr, "start": start, "end": end, "step": step})
    with urllib.request.urlopen(f"{prom}/api/v1/query_range?{q}", timeout=120) as resp:
        body = json.load(resp)
    if body.get("status") != "success":
        raise RuntimeError(f"prometheus error for {expr!r}: {body}")
    return body["data"]["result"]


def integrate(values, step_s):
    """把 (ts, value) 采样序列按 step 积分成 '值 × 小时'。缺失的采样点视为 0。"""
    total = 0.0
    for _, v in values:
        try:
            total += float(v) * step_s / 3600.0
        except ValueError:  # NaN
            pass
    return total


def price_for(series_labels, prices):
    profile = series_labels.get("GPU_I_PROFILE", "")
    if profile:
        return prices.get("mig", {}).get(profile, prices["default"] / 7.0)
    return prices.get(series_labels.get("modelName", ""), prices["default"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prom", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--step", default="5m")
    ap.add_argument("--team-label", default="team")
    ap.add_argument("--prices", required=True)
    ap.add_argument("--tenant-metric", default="llm_d_epp_request_output_tokens_sum",
                    help="EPP 侧按租户的输出 token 累计（histogram 的 _sum）；为空则跳过按租户分摊")
    a = ap.parse_args()
    step_s = int(a.step.rstrip("m")) * 60 if a.step.endswith("m") else int(a.step.rstrip("s"))
    prices = json.load(open(a.prices))
    tl = a.team_label

    # 1. 每张已分配的卡：是否分配（1/0）、SM_ACTIVE、归属。按 UUID + pod 保留一条时序，
    #    这样一张卡先后给两个 Pod 时各自只算自己那段。
    alloc = query_range(
        a.prom,
        f'max by (UUID, GPU_I_PROFILE, modelName, namespace, pod, {tl}, kueue_x_k8s_io_queue_name) '
        f'(DCGM_FI_DEV_GPU_UTIL{{pod!=""}} >= bool 0)',
        a.start, a.end, a.step)
    active = query_range(
        a.prom,
        f'avg by (UUID, namespace, pod) (DCGM_FI_PROF_SM_ACTIVE{{pod!=""}})',
        a.start, a.end, a.step)
    active_by_key = {(s["metric"].get("UUID"), s["metric"].get("namespace"), s["metric"].get("pod")): s["values"]
                     for s in active}
    total_gpus = query_range(a.prom, "count(DCGM_FI_DEV_GPU_UTIL)", a.start, a.end, a.step)
    cluster_gpu_hours = integrate(total_gpus[0]["values"], step_s) if total_gpus else 0.0

    team = defaultdict(lambda: {"gpu_hours": 0.0, "cost": 0.0, "busy_gpu_hours": 0.0})
    queue = defaultdict(lambda: {"gpu_hours": 0.0, "cost": 0.0})
    model_cost = defaultdict(float)  # 推理服务按 app.kubernetes.io/name 视为模型；按需改成自己的 label
    for s in alloc:
        m = s["metric"]
        hours = integrate(s["values"], step_s)
        if hours == 0:
            continue
        cost = hours * price_for(m, prices)
        key = (m.get("UUID"), m.get("namespace"), m.get("pod"))
        busy = integrate(active_by_key.get(key, []), step_s)  # Σ SM_ACTIVE × Δt：'忙的 GPU 小时'
        t = m.get(tl, "unlabeled")
        team[t]["gpu_hours"] += hours
        team[t]["cost"] += cost
        team[t]["busy_gpu_hours"] += min(busy, hours)
        q = m.get("kueue_x_k8s_io_queue_name")
        if q:
            queue[q]["gpu_hours"] += hours
            queue[q]["cost"] += cost
        model_cost[m.get("namespace", "") + "/" + m.get("pod", "").rsplit("-", 1)[0]] += cost

    # 2. 每模型的 token 数（引擎侧），用于每百万 token 成本
    gen = query_range(a.prom, "sum by (namespace, pod, model_name) (increase(vllm:generation_tokens_total[%s]))" % a.step,
                      a.start, a.end, a.step)
    tokens_by_model = defaultdict(float)
    cost_by_model = defaultdict(float)
    for s in gen:
        m = s["metric"]
        tok = sum(float(v) for _, v in s["values"] if v not in ("NaN", "+Inf"))
        tokens_by_model[m.get("model_name", "?")] += tok
        cost_by_model[m.get("model_name", "?")] += model_cost.get(m.get("namespace", "") + "/" + m.get("pod", "").rsplit("-", 1)[0], 0.0)

    # 3. 每租户的 token 占比（网关侧），把模型成本二次分摊到租户
    tenant_tokens = defaultdict(float)
    if a.tenant_metric:
        try:
            ten = query_range(a.prom, "sum by (fairness_id, model_name) (increase(%s[%s]))" % (a.tenant_metric, a.step),
                              a.start, a.end, a.step)
            for s in ten:
                m = s["metric"]
                tenant_tokens[(m.get("fairness_id", "?"), m.get("model_name", "?"))] += \
                    sum(float(v) for _, v in s["values"] if v not in ("NaN", "+Inf"))
        except Exception as e:  # EPP 未部署时跳过
            print(f"# tenant breakdown skipped: {e}", file=sys.stderr)

    # 4. 输出
    allocated_hours = sum(t["gpu_hours"] for t in team.values())
    print(f"# window {a.start} .. {a.end}; cluster GPU-hours {cluster_gpu_hours:.1f}; "
          f"allocated {allocated_hours:.1f} (A={allocated_hours / cluster_gpu_hours:.1%})"
          if cluster_gpu_hours else "# no DCGM data")
    print(f"{'team':<16}{'gpu_hours':>12}{'cost':>12}{'U(sm_active)':>14}{'idle_share':>12}")
    for t, v in sorted(team.items(), key=lambda kv: -kv[1]["cost"]):
        u = v["busy_gpu_hours"] / v["gpu_hours"] if v["gpu_hours"] else 0.0
        print(f"{t:<16}{v['gpu_hours']:>12.1f}{v['cost']:>12.2f}{u:>14.1%}{1 - u:>12.1%}")
    if queue:
        print(f"\n{'cluster/local queue':<32}{'gpu_hours':>12}{'cost':>12}")
        for q, v in sorted(queue.items(), key=lambda kv: -kv[1]["cost"]):
            print(f"{q:<32}{v['gpu_hours']:>12.1f}{v['cost']:>12.2f}")
    if tokens_by_model:
        print(f"\n{'model':<40}{'M tokens':>12}{'cost':>12}{'$/M tok':>10}")
        for mname, tok in sorted(tokens_by_model.items()):
            c = cost_by_model[mname]
            per_m = c / (tok / 1e6) if tok else float("nan")
            print(f"{mname:<40}{tok / 1e6:>12.2f}{c:>12.2f}{per_m:>10.2f}")
    if tenant_tokens:
        print(f"\n{'tenant':<24}{'model':<32}{'M tokens':>10}{'share':>8}{'cost':>10}")
        by_model_total = defaultdict(float)
        for (tenant, mname), tok in tenant_tokens.items():
            by_model_total[mname] += tok
        for (tenant, mname), tok in sorted(tenant_tokens.items()):
            share = tok / by_model_total[mname] if by_model_total[mname] else 0.0
            print(f"{tenant:<24}{mname:<32}{tok / 1e6:>10.2f}{share:>8.1%}{share * cost_by_model[mname]:>10.2f}")


if __name__ == "__main__":
    main()
```
{% endraw %}

脚本的四步对应第七、八章的模型：按 `UUID + pod` 积分得到每段分配的 GPU 小时（按分配计费）；按卡型与 MIG profile 取单价；用 `SM_ACTIVE` 的积分算每团队的"忙的 GPU 小时"与闲置比例；用 `vllm:generation_tokens_total` 的增量算每模型的每百万 token 成本，再用 EPP 的 `fairness_id` 把模型成本按 token 占比分到租户。跑在前几篇的集群上，结果的形状（不给假测量，读者代入自己的数字）应当是：训练团队 GPU 小时多、U 在 40%–60%；推理服务 U 随负载变化大，压测时段每百万 token 成本比空闲时段低一个数量级；`unlabeled` 一行如果不为零，说明有 GPU Pod 没打 `team` label——这是准入策略（ValidatingAdmissionPolicy 或网关的 webhook）该拦下的。

到这里，`mini-platform/` 的八个增量合在一起（随文给出、由读者自行保存成对应文件，不是一个已发布的软件包）：`cluster/` 与 `probes/` 是裸集群与第一个 Pending 的 GPU Pod；`gpu/` 装上 GPU Operator 与 DRA；`sched/` 用 Kueue 与 Volcano 跑通两队列 cohort 下的 DDP；`share/` 在一张卡上放两个服务并验证隔离；`net/` 与 `storage/` 把 RDMA 与 checkpoint 存储接进容器；`serve/` 与 `gateway/` 把 vLLM 变成有扩缩容、有路由、有配额的服务；`obs/` 与 `cost/` 让全部这些有一张看板和一份账单。


## 十三、系列总结

八篇文章从一个 Pending 的 GPU Pod 出发，走到一份按团队与租户分摊的账单。回头看，读者手上应当有三样东西。

**一个最小平台**。`mini-platform/` 的每个目录都是一个生产环境在用的组件的最小配置：GPU Operator 的 `ClusterPolicy` 与 DRA 的 `DeviceClass` / `ResourceClaim`（第二篇），Kueue 的 `ResourceFlavor` / `ClusterQueue` / `LocalQueue` 与 Kubeflow Trainer 的 `TrainJob`（第三篇），MIG 的 `mig.config` 与 HAMi 的 `gpumem` / `gpucores`（第四篇），Multus 的 `NetworkAttachmentDefinition` 与 RDMA device plugin、JuiceFS CSI 与 DCP 写吞吐（第五篇），LeaderWorkerSet 与 `LLMInferenceService`、KEDA 的 `ScaledObject`（第六篇），`InferencePool` 与 Endpoint Picker、两租户的 TPM 配额（第七篇），DCGM 的 Pod 映射、三个数字的 recording rules 与 `allocate.py`（本篇）。它离生产差的是规模与高可用，不是机制。

**一张机制表**。每篇一张"引擎需求 → K8s 空缺 → 平台机制 → 代价"四栏表，八张合起来就是这一层的全景：K8s 的哪条假设被 AI 负载违背了、哪个组件填了这个洞、填洞的代价是什么。接手一个别人搭的平台时，把它的组件清单对到这八张表上，就知道它填了哪些洞、没填哪些、每个选择付了什么代价。

**一组数字**。分配率、使用率、有效利用率，每百万 token 成本，每团队的 GPU 小时与闲置比例，每队列的配额使用率与排队时间。它们把前七篇的每一个取舍变成可以度量的东西，也把"平台好不好"从一个印象变成一张分解表。

三条贯穿全系列的线，各自的终点：

```text
引擎线   训练框架的进程组与 checkpoint → 推理引擎的显存与请求队列 → 两者对平台接口的要求
         第一篇列出需求清单 → 第三篇的 gang 与第五篇的 RDMA / checkpoint I/O 满足训练 → 第四、六、七篇的切分 / 扩缩容 / 路由满足推理
         → 第八篇：引擎的指标（vllm:* · 约定的 train_*）成为平台决策的输入，平台反过来对引擎提出"暴露哪些指标"的需求
机制线   device plugin → 调度器扩展 → 切分与隔离 → 第二张网卡 → CRD 与 Operator → 网关扩展 → 指标管线
         第二篇 device plugin / DRA → 第三篇 Kueue / Volcano → 第四篇 MIG / HAMi → 第五篇 Multus / RDMA plugin / CSI
         → 第六篇 LWS / KServe / KEDA → 第七篇 InferencePool / EPP → 第八篇 DCGM 映射 · recording rules · 成本分摊；
         每一层都在填 K8s 原生假设的一个洞，每篇一张四栏表
取舍线   隔离 vs 利用率 → 排队 vs 碎片 → 拓扑 vs 等待时间 → 冷启动 vs 常驻成本 → 精确计费 vs 开销
         第四篇的 MIG / 时间片 → 第三篇的 gang / 配额 / cohort → 第三、五篇的 TAS 与 RDMA → 第六篇的缩零与扩容时间
         → 第八篇：每个取舍都对应三个数字里的一段差距，成本把它们换成同一个单位，FinOps 回路按月重做这些取舍
```

总纲提出的三种能力，现在可以逐条对照：

1. **设计能力**：面对一组训练与推理负载，按第一篇的需求清单选调度器（Kueue 还是 Volcano，第三篇）、切分策略（MIG / HAMi / 时间片，第四篇）、网络与存储（RDMA 接入方式与存储三类需求，第五篇）、Serving 形态（单 Pod / LWS / PD 分离，第六篇）与网关（InferencePool + EPP，第七篇），并用每篇的四栏表说清每个选择的代价——第一到七篇。
2. **排障能力**：Pod 看不到 GPU 查第二篇的四层栈；任务 Pending 查第三篇的 gang / 配额 / 拓扑；共卡互相影响查第四篇的隔离层次；多机变慢查第五篇的 RDMA 路径；扩容慢查第六篇的时间分解；请求排在满的副本上查第七篇的路由；每一条都从本篇的四层指标进入——第二到八篇。
3. **运营能力**：建立从 DCGM 到网关的四层指标，把分配率与有效利用率的差距分解到具体原因，按团队 / 队列 / 租户分摊成本，算出每百万 token 的成本，并用它驱动配额、切分与扩缩容参数的按月调整——本篇，以及它回指的每一篇。

平台的每一个设计决定都是被引擎的某个需求推出来的，这是总纲的第一句话，也是全系列的方法：面对集群上的任何异常——Pending、变慢、超时、账单超支——先问"引擎在这一层要什么、K8s 为什么给不了、平台用什么给的、代价是什么"，答案就在三条线的交点上。

**系列目录**

- 总纲：[AI 平台工程：资源层与交付层](/ai-platform-engineering.html)

1. [引擎的需求清单与平台的整体架构](/ai-platform-engine-requirements-and-architecture.html)
2. [容器里的 GPU：驱动、CUDA、device plugin 与镜像](/gpu-in-containers-driver-cuda-device-plugin.html)
3. [AI 任务调度：gang scheduling、队列与拓扑感知](/ai-job-scheduling-gang-queue-topology.html)
4. [GPU 共享与切分：MIG、时间片、MPS 与 HAMi](/gpu-sharing-and-partitioning-mig-mps-hami.html)
5. [网络与存储：RDMA 进容器、并行文件系统与 checkpoint I/O](/rdma-networking-storage-and-checkpoint-io.html)
6. [Serving 平台：从 InferenceService 到 llm-d](/serving-platforms-kserve-triton-ray-serve-llm-d.html)
7. [模型网关与多租户：路由、配额与灰度](/model-gateway-multi-tenancy-and-quota.html)
8. [可观测、成本与 FinOps](/ai-platform-observability-cost-and-finops.html)
