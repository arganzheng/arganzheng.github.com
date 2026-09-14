---
layout: post
title: RL 后训练基础设施：rollout 与训练如何共享一组 GPU（总纲）
subtitle: "RL Post-Training Infrastructure: How Rollout and Training Share the Same GPUs"
tags: [RL, verl, vLLM, Megatron, Distributed Training, AI, AI-Infra]
catalog: true
---


## 内容简介

《RL 后训练基础设施：rollout 与训练如何共享一组 GPU》是一组共八篇的系列文章，面向已经理解大规模训练引擎（并行策略、FSDP / Megatron、checkpoint）和 LLM 推理引擎（vLLM 的调度、KV cache、权重加载）的工程师，系统讲解 RLHF / GRPO 一类**在线强化学习后训练**的系统是如何组织的：一个 RL 训练步里的**生成（rollout）、打分（reward）、更新（train）**三段各花多少算力与显存，训练框架与推理引擎怎样在同一组 GPU 上共存或分离，每一步的新权重怎样从训练器的分片布局搬进推理引擎，异步与 off-policy 怎样把 GPU 的等待压掉，Agent 训练的沙箱集群与长尾轨迹怎样调度，以及 verl 这个当前的事实标准框架怎样把这些机制组织成一个系统。

它回答的问题是：

> **一个 RL 后训练任务，同时是一个推理服务和一个训练任务。这两样东西怎样共享一组 GPU，而不让任何一方在等另一方？**

这个问题在预训练和 SFT 里不存在：那里只有训练，模型从数据里读 token；在线 RL 里 token 是模型**自己生成**的，每一步先要跑一次推理——批量、长序列、decode 为主——再拿生成的样本训练，然后把新权重送回推理引擎生成下一批。生成是 memory-bound 的 decode，训练是 compute-bound 的 GEMM，两者对 GPU 的用法相反，对显存的要求互斥（一个要 KV cache，一个要参数 + 梯度 + 优化器状态），而且每一步之间有一道同步的墙。公开报告里 rollout 占 RL 训练墙钟时间的 60–80%，GPU 平均利用率常在 30% 以下——**优化 RL 后训练 = 让训练循环里的推理引擎不闲着，也不让训练器等它**。

系列的组织原则来自这个负载自身的结构：**RL 训练的一步是三个形态不同的作业加两次同步**。生成是一个 serving 作业，打分是一次或几次前向（或一个外部服务），训练是一个常规的分布式训练步；两次同步是"样本从推理引擎到训练器"与"权重从训练器到推理引擎"。把三个作业的算力、显存、时间和两次同步的字节数与耗时追踪清楚，就是理解 RL 基础设施的全部；系统形态（共置 / 分离 / 异步）、权重同步的传输方式、Agent 环境的调度，都是在这张账上做交换。

系列以一个框架为源码阅读对象：**verl**（ByteDance Seed 发起、社区维护，HybridFlow 论文的开源实现）。选它不是因为它唯一，而是因为它是当前的事实标准，且**一套代码覆盖了本系列要讲的全部形态**——共置 / 分离 / 异步三种 trainer 模式、FSDP 与 Megatron 两种训练后端、vLLM 与 SGLang 两种推理后端、NCCL / NIXL / 增量三种权重同步方式、agent loop 与沙箱网关。每一篇的机制都能在 verl 里找到对应的实现，8 卡实践也全部用它。第七篇末尾用一节对照两个取向相反的框架——**slime**（智谱，只绑 Megatron + SGLang、参数原样透传、刻意不做抽象，GLM 系列的 RL 框架）与 **AReaL**（蚂蚁，从第一天起就是完全异步的设计，论文把异步的算法修正写得最清楚）——看同一个问题上它们与 verl 的分歧点在哪，以此分清哪些是这类负载的必然，哪些是 verl 的选择。OpenRLHF 作为最早的 Ray + vLLM 开源实现只在历史脉络里提及。


## 为什么写这个系列？

### RL 后训练是当前增长最快的一类 AI-Infra 负载

2025 年以后，前沿模型的能力增量主要来自后训练里的强化学习：可验证奖励的推理模型（DeepSeek-R1 一族）、多轮工具调用的 Agent 训练、以及在此之上的蒸馏。它们的算力开销已经不是"预训练之后的一点微调"：一个推理模型的 RL 训练，一步 rollout 几千万到几亿 token，总算力到 $$10^{22}$$–$$10^{23}$$ FLOPs，与一个小模型的预训练相当；Agent 训练每一步还要驱动几千个沙箱容器跑测试。集群里为 RL 预留的 GPU 份额在快速上升，而它的系统问题——推理引擎与训练框架的共存、权重同步、异步、环境调度——在预训练和推理服务两条主线里都没有对应物。

### 它是两条主线的组合，但组合本身是新问题

这个方向的每个**组件**都已在别处覆盖：训练器是 FSDP / Megatron，推理引擎是 vLLM / SGLang，通信是 NCCL，调度是 Ray / Kubernetes。这曾是它没有单列一个系列的理由。但把它们放进一个循环之后出现的问题，任何一个组件里都没有：

```text
显存归属      同一张卡上，训练状态（16 字节/参数）与 KV cache 轮流占据显存，切换要多久？
权重布局      FSDP 的 DTensor 分片、Megatron 的 TP/PP/EP 分片 → vLLM 的 TP/EP 分片（可能还要量化成 FP8），怎么搬？
同步的墙      严格 on-policy 要求"生成完 → 训练 → 同步权重 → 再生成"，长尾的一条轨迹让所有 GPU 等它
异步的代价    生成用 k 步前的权重，样本过期，重要性比修正的边界在哪？
环境          Agent 轨迹里八成 token 来自环境，环境的 CPU 小时与模型的 GPU 小时同量级，两个集群怎么配比？
```

这些问题的答案决定了一个 RL 集群的利用率是 30% 还是 70%。它们是编排问题，但编排不等于简单。

### 框架已经收敛到可以写"机制"而不是"实现"

2024 年这个领域的框架各自为战、接口每月都变。到本系列写作时（2026 年 9 月初），情况已经不同：verl 的 HybridFlow 编程模型（单控制器 + 多 worker 组）成了多数框架的共同形态；"共置 / 分离 / 异步"三种系统形态都有了生产实现并且可以在同一个框架里切换；权重同步的传输方式（NCCL 广播、CUDA IPC、NIXL / Mooncake、增量同步）与推理引擎的显存让渡接口（vLLM 的 sleep / wake_up、SGLang 的 memory saver）已经稳定；异步训练的算法修正（重要性比、staleness 上限、部分 rollout）在 AReaL、verl、slime 里的写法基本一致。机制稳定、实现仍在变——正是适合写系列的时候：正文以机制为主，源码只在必要处引用，版本随文标注。

### 现有材料的断层

- **论文**（HybridFlow、AReaL、DeepSeek-R1、Kimi k1.5 的系统章节）给出了设计与数字，但不讲一个具体的任务怎么配、卡怎么分、慢在哪；
- **框架文档**告诉你每个配置项是什么，不告诉你在给定模型、GPU 数与 rollout 长度下应该选共置还是分离、rollout 与训练的 GPU 比例多少；
- **博客与教程**大多停在"用 verl 跑通一个 GSM8K 的 GRPO"，很少触及权重同步的字节数、显存切换的耗时、异步的 staleness 边界、沙箱集群的调度；
- **一线团队的经验**散落在各框架的 release note、issue 与内部文档里。

本系列想填补的是从"能跑通一个 7B 的 GRPO"到"能为一个几百卡的 RL 任务做出系统形态与资源配比的决策、并解释它的利用率为什么是这个数"之间的那段路。


## 适合哪些读者？

### 负责 RL 训练任务的系统工程师

你的团队在做推理模型或 Agent 的 RL 训练，任务能跑，但 GPU 利用率低、一步的墙钟时间大半在等 rollout、权重同步一次要几十秒、规模再扩大就 OOM 或 hang。你需要知道时间去了哪、每一项的理论下界是多少、哪种系统形态对你的负载合适。本系列是为这个阶段准备的。

### 训练平台与调度团队

你负责让 RL 任务和预训练、SFT 任务在同一个集群上跑。RL 任务对平台的要求与它们不同：一个任务里有两类形态相反的 GPU 负载、可能还有一个几千容器的沙箱集群、一个 reward 服务、一个 Ray 集群；它的弹性、checkpoint 与故障恢复语义也不同。本系列第二、六、八篇是引擎对平台提出的要求。

### 想读懂 verl 源码的开发者

verl 的核心目录是本系列的源码阅读线。读完后你应该能在 `verl/trainer/ppo/`、`verl/workers/`、`verl/single_controller/` 与 checkpoint engine 里定位任何一个系统环节——rollout 引擎的启动与显存让渡、权重同步、样本缓冲、异步控制、agent loop——并有能力修改它；读 slime 或 AReaL 时也知道该找什么。

### 做后训练算法、想知道系统在限制什么的算法工程师

你在设计 RL 配方，需要知道哪些算法选择对系统有多大代价：$$G$$ 与回答长度决定 rollout 的 KV 量与长尾，价值模型多占一个模型的显存，严格 on-policy 与异步之间差多少倍的吞吐，多轮 Agent 环境为什么让异步成为必需。本系列第一、二、五、六篇是这些问题的系统侧答案；算法本身（目标函数、奖励、配方）在算法地图的[《后训练：从 SFT 到可验证奖励》](/post-training-from-sft-to-verifiable-rewards.html)。


## 系列的整体主线

八篇文章按"先算账、再定形态、再解决形态带来的三个问题、最后读源码与配置"的顺序推进：

```text
第一篇：负载画像 —— 一步 RL 里发生什么：三段作业的算力、显存与时间账
        ↓  生成 / 打分 / 训练的 FLOPs 与 MFU · 四个模型的显存 · KV cache 的量 · 为什么 rollout 占一半以上
第二篇：系统形态 —— 共置、分离与异步：三种拓扑的利用率、气泡与正确性代价
        ↓  单控制器与 worker 组 · 时分复用 vs 空分复用 · rollout : train 的 GPU 配比
第三篇：共置 —— 训练器与推理引擎在同一组 GPU 上共存
        ↓  显存归属切换 · sleep / wake_up 与 memory saver · CUDA graph 与 KV 池的重建代价
第四篇：权重同步 —— 从训练分片到推理分片
        ↓  布局映射与重分片 · NCCL 广播 / CUDA IPC / NIXL · 量化传输与增量同步 · 一次同步多少秒
第五篇：异步与 off-policy —— 把同步的墙拆掉之后要补什么
        ↓  staleness 上限 · 部分 rollout · 重要性比与训推不一致 · 样本缓冲与流式数据
第六篇：Agentic rollout —— 多轮、工具、沙箱与环境服务
        ↓  agent loop · 几千容器的沙箱集群 · 长尾轨迹调度 · reward 服务化 · 环境账与 GPU 账
第七篇：verl 源码导读 —— 从一个 GRPO 配置追到每个 worker；末尾对照 slime 与 AReaL
        ↓  单控制器 · worker 组 · rollout server · checkpoint engine · 样本通路 · 两个反例的分歧点
第八篇：配置、可观测与排障 —— 从一张卡的比例到一条 hang 的排查
           GPU 配比 · 两套并行配置 · 全步 MFU · RL 状态的 checkpoint · 确定性 · 常见故障
```

前两篇是**账与形态**：一步里每样东西花多少，三种形态各自把 GPU 用到几成。中间四篇是**形态带来的四个问题**：共置要切显存、任何形态都要同步权重、异步要修正、Agent 要调度环境。后两篇是**源码与运维**。

三条交织的线索：

```text
账本线：FLOPs · 显存 · 字节数 · 秒 —— 每篇都把新引入的机制记进同一张账
形态线：共置 / 分离 / 异步 —— 每个机制在三种形态下各是什么样
框架线：每个机制在 verl 里的实现位置；第七篇对照 slime 与 AReaL
```


## 章节结构与分章导读

### 1. 负载画像：一步 RL 里发生什么

第一篇建立全系列的记账框架。它不讨论任何系统形态，只回答一个问题：**一个 GRPO / PPO 训练步里，生成、打分、训练三段各做多少 FLOP、占多少显存、在什么 MFU 下花多少秒？**

这一篇会讨论：

- 一步的结构：$$B$$ 个 prompt × $$G$$ 条回答 × 平均长度 $$\bar L$$ 的 rollout；参考模型与（若有）奖励模型 / 价值模型的前向；策略的前向 + 反向；两次同步——样本进训练器、权重回推理引擎；
- FLOPs 的拆分：生成 $$2N$$、参考前向 $$2N$$、RM 前向 $$2N_{RM}$$、策略训练 $$6N$$，合计约 $$12N$$ 每 token；PPO 再加价值网络约 $$8N$$；**按 FLOPs 训练占一半、生成只占六分之一**；
- 时间的拆分：训练与前向是大 GEMM，MFU 40% 上下；生成是 decode，每步一个 token、memory-bound，对话长度下 MFU 只有 10% 上下，长思维链下 KV 读取占了带宽的大半、只剩个位数；再加长尾——同一波里最长的回答决定这一波何时结束——**按时间生成占一半以上**，HybridFlow（verl）与 OpenRLHF 的论文都报告 rollout 占 60–80%；
- 显存的四份：训练状态（bf16 参数 + 梯度 + fp32 主参数与 Adam 状态，每参数 16 字节，来自大规模训练系列第一篇的公式）、参考模型与 RM 的 bf16 权重、推理引擎的权重副本、KV cache；对话 RLHF 的 4096 条 × 1300 token 在 Llama-3-8B 规格下是 680 GiB 的 KV，推理模型 8K–32K 的回答是它的 8 到 64 倍——**任何一张卡都放不下全部，rollout 必须分波或分卡**；
- 三段作业的形态对比：生成是 serving（请求、批、KV 池、连续批处理），训练是分布式训练步（并行、梯度、优化器），打分要么是 prefill 形态的前向、要么是外部服务（规则验证器、代码沙箱、生成式 RM）——它们对 GPU 的用法为什么互斥；
- 推理模型与 Agent 把账放大了什么：$$\bar L$$ 从 1K 到 32K 让 rollout 的 token 数与 KV 多一到两个数量级、墙钟从分钟到几十分钟；Agent 轨迹里八成 token 来自环境，训练要过它们的前向却不产生梯度，每个有效 token 的代价是单轮的 5 倍；
- 公开配方的账：DeepSeekMath / R1 的 GRPO、Kimi k1.5、Qwen3 的后训练报告里能推出的 rollout 规模、GPU 数与步时间。

核心问题是：

> **一个 8B 策略模型，$$B = 512$$、$$G = 16$$、$$\bar L = 8\text{K}$$，在 64 张 H100 上做 GRPO。一步生成多少 token、多少 KV、要分几波？三段各要多少 GPU·秒？GPU 平均利用率的理论上限是多少？——这一步之后每一篇要解决的，就是实际值与这个上限之间的差。**

实践：写一个 RL 一步账本脚本，输入模型规格、$$B / G / \bar L$$、硬件与各段的 MFU 假设，输出三段的 FLOPs、显存、KV、时间与利用率上限。它是全系列的第一件工具，后面每一篇给它加一个维度。

### 2. 系统形态：共置、分离与异步

第二篇讨论三个形态相反的作业怎样放到一组 GPU 上。它把系统形态当作**在利用率、显存和正确性之间的三种交换**，而不是三个框架的三个默认值。

这一篇会覆盖：

- HybridFlow 的编程模型（verl 的论文）：**单控制器**负责 RL 算法的数据流（一步里谁先谁后、样本怎么流），**多个 worker 组**（actor、rollout、ref、critic、RM）各自用 SPMD 方式跑自己的分布式计算；这个模型为什么让"换算法"与"换系统形态"互不干扰，以及它的代价（控制器与 worker 之间的数据搬运）；
- **共置（colocate）**：所有角色在同一组 GPU 上**时分复用**——生成时显存归推理引擎，训练时归训练器；优点是每个阶段都能用满全部 GPU、权重同步在卡内完成；代价是显存要在两种状态之间切换、阶段之间无法重叠、长尾拖住所有卡；
- **分离（disaggregate）**：rollout 与训练在不同的 GPU 池上**空分复用**；优点是两种负载各用适合自己的并行配置、可以重叠；代价是配比要事先定好、权重同步要跨机器、配比不对就有一方空转；
- **异步（async）**：分离之上再去掉每步的同步墙——训练用 $$k$$ 步前的权重生成的样本，生成不等训练；GPU 利用率最高，代价是 off-policy 与它的算法修正（第五篇）；
- 三种形态的利用率模型：把第一篇的三段时间代入，共置的利用率 = 各段 MFU 的时间加权，分离的利用率取决于配比与两侧时间是否相等，异步趋近两侧各自的 MFU；一个简单的数学模型给出每种形态在不同 $$\bar L$$ 方差下的期望利用率；
- rollout : train 的 GPU 配比怎么定：由两侧的时间比决定，而时间比随训练推进变化（回答变长、KV 变多），所以生产系统里配比是可调的——verl 的动态资源调度（trainer 节点的 GPU 按策略在 rollout 与训练之间切换）；
- verl 的三种 trainer 模式——`sync`（共置同步）、`colocate_async`（共置异步）、`separate_async`（分离异步）——与上面三种形态的对应；它们共享一套控制流、样本缓冲与指标，所以换形态只是换配置。

核心问题是：

> **同样 64 张卡、同样的 GRPO 配置，共置、分离（48 : 16）与异步三种形态下，一步的墙钟时间与 GPU 利用率各是多少？回答长度的方差从小变大时，哪种形态先撑不住？**

实践：给账本加系统形态维度：输入形态与配比，输出各阶段的 GPU 利用率、气泡与一步的墙钟；用第一篇的公开配方验证。

### 3. 共置：训练器与推理引擎在同一组 GPU 上共存

第三篇进入共置形态的核心机制：**同一张卡上的显存怎样在训练状态与 KV cache 之间切换，切一次要多久。**

这一篇会覆盖：

- 显存归属的问题：训练器需要每参数 16 字节加激活，推理引擎需要权重副本加尽可能大的 KV 池；两者加起来远超 80 GB，所以生成时训练状态要让出去、训练时 KV 池要让出去；
- 推理引擎的让渡接口：vLLM 的 **sleep mode**——level 1 把权重拷到 CPU、释放 KV 池，level 2 连权重也丢弃（下次 wake_up 后重新同步）；它建立在 vLLM 的 `CuMemAllocator`（用 CUDA 虚拟内存 API 把物理页从虚拟地址上摘下再挂回）之上；SGLang 的 `torch_memory_saver` 走同一条路；
- 训练器一侧的让渡：FSDP 的参数与优化器状态 offload 到 CPU、`torch.cuda.empty_cache`、Megatron 的 offload 路径；两侧让渡的顺序与 pinned memory 的用量；
- 切换的代价：一次 sleep + wake_up 要搬多少字节（8B 的 16 GB 权重 PCIe 往返约 1 秒；70B 的 140 GB 是十几秒），CUDA graph 是否要重新捕获、KV 池是否要重新分配与预热、prefix cache 是否失效；每步两次切换在步时间里占多少；
- 共置下的权重同步：训练器与推理引擎在同一进程或同一台机器——`update_weights` 走 CUDA IPC 或进程内张量传递，不走网络；verl 的 `ActorRolloutRefWorker`——同一个 worker 同时持有训练器与推理引擎——与 hybrid engine 的实现；
- 显存碎片与 allocator：训练与推理两套 allocator 在同一块显存上交替，`expandable_segments`、mem pool 的用法与坑；
- 共置形态的适用边界：模型多大、$$\bar L$$ 多长时切换代价开始不可忽视，何时应换分离。

核心问题是：

> **一个 32B 模型在 8 卡共置。每步开始生成前要把 16 字节/参数的训练状态搬走、把 KV 池建起来，训练前再反过来。每个动作搬多少字节、走哪条链路、要几秒？步时间 5 分钟时这是 2% 还是 20%？**

实践：在 8 卡上用 verl 跑一个共置的 GRPO，用 `torch.cuda.memory_stats` 与 profiler 记录每次切换前后的显存归属与耗时，与计算对账；对比 sleep level 1 与 2 的差别。

### 4. 权重同步：从训练分片到推理分片

第四篇讨论任何形态都绕不开的一步：**训练器更新后的权重怎样进入推理引擎。**它是 RL 系统里最"Infra"的一段——纯粹的布局映射与字节搬运。

这一篇会覆盖：

- 问题的两半：**布局**（训练器按 FSDP 的 DTensor 或 Megatron 的 TP / PP / EP 切分，推理引擎按自己的 TP / EP 切分，两边的切法、rank 数、甚至参数名都不同）与**传输**（字节从训练卡到推理卡走哪条链路）；
- 名字与形状的映射：Hugging Face 的参数名 ↔ Megatron 的融合权重（QKV 合并、gate/up 合并、专家堆叠）；Megatron-Bridge / mbridge 做的就是这层映射；FSDP 一侧 `full_state_dict` 与 sharded state dict 的区别，为什么"先 all-gather 成完整权重再广播"在大模型上不可行；
- 传输方式：共置下的 CUDA IPC；分离下的 **NCCL 广播**（训练 rank 0 → 所有推理 rank，或按分片点对点）、**NIXL / Mooncake** 的 RDMA 传输、经磁盘或对象存储的 checkpoint 中转；各自的带宽、对进程组的要求、与训练 / 推理各自 NCCL 通信的冲突；
- 分桶与流水：权重按 bucket 逐个 gather → 传输 → 加载，让三步重叠、并把峰值显存限制在一个 bucket；bucket 大小的选择；
- **量化传输**：推理引擎用 FP8 / MXFP4 权重时，同步路径上要做在线量化——scale 怎么算、与推理侧的量化格式怎样对齐；
- **增量同步**：每步只有一部分权重变化明显（LoRA 时更极端——只传 adapter）；verl 0.9 的 `delta_sharded` checkpoint engine 让每个训练 rank 只对自己的分片做字节 diff、只传变化的位置，报告 7B–72B 上比全量 NCCL 广播快 1.9–3.1 倍；
- 一次同步多少秒：8B 的 16 GB 在 NVLink 内几秒、跨机 InfiniBand 几十秒；405B 的 810 GB、DeepSeek-V3 规格的 671B MoE 是分钟级——与步时间相比值不值、异步下同步频率怎么定；
- 正确性：同步中途推理引擎不能生成（或必须知道自己在用哪个版本）；MTP 草稿模型、LoRA 合并、量化 scale 这些"权重之外的权重"漏同步的后果。

核心问题是：

> **训练器是 Megatron TP=4、PP=2、EP=8 的 671B MoE，推理引擎是 vLLM TP=8、EP=4 的 FP8 副本，两边在不同机器。一次同步要做哪几步映射、传多少字节、走哪条链路、至少几秒？增量同步能省多少？**

实践：不依赖框架，写一个最小的 FSDP2 → vLLM 权重同步：从 DTensor 分片 gather 一个 bucket、按名字映射、经 NCCL 广播、用 `collective_rpc` 加载进 vLLM；测量各段耗时，与账本对照。

### 5. 异步与 off-policy：把同步的墙拆掉之后要补什么

第五篇讨论异步形态引入的算法与系统问题。它的前提是第二篇的结论：只要回答长度有方差，同步形态就有大量的等待；拆掉同步墙能换回利用率，但样本不再严格 on-policy。

这一篇会覆盖：

- staleness 的定义与控制：训练用第 $$t$$ 步的权重更新，样本由第 $$t-k$$ 步的权重生成；$$k$$ 的上限（AReaL 的 `max_head_offpolicyness`、verl 的 `max_off_policy_threshold` 与 `drop` / `wait` 两种策略）；$$k = 1$$–$$4$$ 在多数报告里几乎无损；
- **部分 rollout**（Kimi k1.5、AReaL）：给每步的生成设 token 或时间上限，没生成完的序列暂停、下一步用新权重继续——一条轨迹跨越多个策略版本；它把长尾截平的代价是同一条序列的不同段来自不同策略；
- 算法修正：重要性比 $$\rho = \pi_\theta / \pi_{old}$$ 与 clip 在 off-policy 下的行为；AReaL 的 decoupled PPO loss（把"生成策略"与"近端策略"分开）；按 token 记录 $$\log \pi_{old}$$ 与策略版本号的系统开销；
- **训推不一致**（training-inference mismatch）：推理引擎（vLLM / SGLang，可能 FP8、不同的 kernel 与 reduce 顺序）算出的 $$\log \pi$$ 与训练器（FSDP / Megatron，bf16）算出的不同，本身就是一种 off-policy；TIS（truncated importance sampling）等修正；为什么要监控两侧 logprob 的差、以及 verl / slime 把它做成了测试项；
- 样本通路：rollout 完成一条就进**样本缓冲**（replay buffer），训练器取够一个 batch 就更新；缓冲的淘汰规则（过期、DAPO 式过滤掉全对 / 全错的组、失败的 rollout）；流式数据加载与"变长 global batch"；
- 异步下的权重同步频率与方式：不再是"每步一次、全员停下"，而是推理引擎在请求之间换权重、或多个推理实例轮流更新；正在生成的请求怎么办——中断重来、还是记下版本继续；
- 异步的可观测：off-policy 指标（staleness 分布、$$\rho$$ 的分布、被裁掉的比例）、缓冲区深度、两侧的等待时间；
- verl 的 `colocate_async` / `separate_async` 里这些机制的实现：replay buffer、staleness 策略、流式 dataloader、异步 checkpoint 恢复；AReaL 的完全异步设计（rollout controller + 样本缓冲 + 版本管理）作为原型在第七篇对照。

核心问题是：

> **从同步换成 $$k \le 2$$ 的异步，一步的墙钟从 12 分钟降到 5 分钟，但 reward 曲线的斜率变缓了。是 staleness、是训推不一致、还是缓冲区淘汰规则？要区分这三个原因，需要事前记录哪些信号？**

实践：在 8 卡上用 verl 的 `colocate_async` 模式跑一个小模型的 GRPO，对比 $$k = 0, 1, 2, 4$$ 的吞吐与 reward 曲线；记录两侧 logprob 的差并人为放大（推理侧换 FP8）观察影响。

### 6. Agentic rollout：多轮、工具、沙箱与环境服务

第六篇讨论 rollout 不再是"推理引擎批量生成"、而是"模型与环境交替几十轮"时系统的变化。它是第五篇异步的最强动机：环境的耗时方差比生成长度的方差大得多。

这一篇会覆盖：

- agent loop 的结构：模型生成一段 → 解析出工具调用 → 环境执行（几毫秒到几分钟）→ 结果追加进上下文 → 再生成；一条轨迹几万 token、几十轮，其中八成 token 是环境返回的；
- 推理引擎侧的要求：**前缀缓存**让每轮的追加只 prefill 新增部分（累计 prefill 从二次降到线性）；多轮 token 的连续性（chat template 在轮与轮之间的边界处理，verl 的 continuous token 机制）；一条轨迹在几十轮之间保持 KV 还是重新 prefill；
- **沙箱集群**：代码类任务每条轨迹要起一个容器、跑测试套件、重复十几轮——一步 rollout 是几万次容器执行；容器的启动时间、镜像分发、状态回收、并发上限、隔离与安全；环境的 CPU·小时与模型的 GPU·小时同量级——**Agent RL 的集群里 GPU 旁边有一个与之匹配的 CPU 集群**；
- 环境服务化：环境与 reward 做成独立服务（OpenAI / Anthropic 兼容的接口、会话网关），推理引擎与训练器都不直接碰它；verl 的 agent loop 与 uni-agent 网关走的就是这条路（slime 的 `slime/agent` 模块同理）；"黑盒 agent"——把 Claude Code、mini-SWE-agent 一类现成的 harness 直接接进 RL 训练；
- 长尾调度：一步里 95% 的轨迹早已完成、5% 卡在一个 10 分钟的测试上；同步形态下 GPU 在等，异步形态下轨迹完成一条就进缓冲；轨迹级的超时、重试与丢弃策略；变长 global batch；
- reward 的形态：规则验证器（CPU，快）、代码执行（沙箱，慢且方差大）、生成式 RM（另一个 LLM 的推理服务，共置或独立）——三种在系统上完全不同的东西怎样进同一条通路；
- 环境账与 GPU 账：一步 SWE-bench 规模的 rollout 要多少 CPU 核·小时、多少并发容器才能让墙钟与训练匹配；沙箱并发数是新的配比变量。

核心问题是：

> **500 个代码任务 × $$G = 8$$ × 20 轮，每轮跑一次几十秒的测试。一步 rollout 要多少次容器执行、多少 CPU·小时、并发多少个沙箱才能在 30 分钟内完成？GPU 这一侧在这 30 分钟里做了什么？**

实践：搭一个最小的沙箱环境（容器化的代码执行 + 规则 reward），用 verl 的 agent loop 在小模型上跑多轮 RL；测量每步的环境时间分布与 GPU 空闲比例，对比同步与异步。

### 7. verl 源码导读：从一个 GRPO 配置追到每个 worker

第七篇进入源码。前六篇的每个机制在 verl 里都有实现，这一篇按"一个配置文件怎样变成一组 Ray worker、一步训练怎样在它们之间流动"的顺序把它们串起来，末尾用一节看两个取向相反的框架在哪里与 verl 分道。

这一篇会覆盖：

- **控制流**：`verl/trainer/ppo/` 的单控制器训练循环——v1 统一 trainer 下 `sync` / `colocate_async` / `separate_async` 共享一套控制流、replay buffer 与指标；从 `main_ppo` 的入口读到一步里 generate → reward → old logprob / ref → advantage → update 的每个阶段各由哪个 worker 组执行、数据以什么形态（tensordict）在控制器与 worker 之间搬运；
- **worker 组织**：`verl/single_controller/` 的 `RayWorkerGroup` 与 `@register` 分发装饰器（单控制器怎样把一次调用广播到一组 SPMD 进程并收回结果）；`verl/workers/` 的角色（actor / rollout / ref / critic / RM）与 model engine 抽象（FSDP、Megatron、VeOmni 后端在同一个 `TrainingWorker` 接口后面）；Ray 的资源池与 placement group 怎样表达共置与分离；
- **rollout 的接入**：server 模式——vLLM / SGLang 作为独立服务启动，agent loop 与 reward 通过 HTTP 调它；`sleep` / `wake_up` 在哪里被调用、CUDA graph 与 KV 池的重建发生在哪一步；
- **权重同步**：checkpoint engine 的三条路（NCCL 广播、NIXL / Mooncake、`delta_sharded`）在代码里怎样被选择、分桶与流水在哪一层、量化传输的 scale 在哪算；Megatron-Bridge 的参数名映射怎样被调用；
- **样本通路与异步**：replay buffer 的入队 / 淘汰 / 取 batch、staleness 的 `drop` / `wait`、流式 dataloader、异步 checkpoint 恢复——第五篇的机制各落在哪个类；
- **agent loop**：多轮 rollout 的循环、工具调用的解析、continuous token 的边界处理、uni-agent 网关的会话；
- **对照一节**：**slime** 与 verl 的分歧点——不做后端抽象、只绑 Megatron + SGLang 并原样透传参数（`--sglang-*`），训练 / rollout / Data Buffer 三段通路；为什么"少一层抽象"在这类系统里可以是优点，代价是什么。**AReaL** 与 verl 的分歧点——从第一天就是完全异步：rollout controller、样本缓冲、`generate` / `update_weights` 两类请求交错、部分 rollout 与版本管理是核心而不是可选项；它的 decoupled PPO loss 为什么必须与系统一起设计。OpenRLHF 作为最早的 Ray + vLLM 实现在脉络里提一句：哪些设计被 verl 继承。

核心问题是：

> **一个 bf16 参数从优化器更新完成，到推理引擎用它生成下一个 token，在 verl 里经过哪些函数、哪些进程、哪条链路？把这条链追清楚，前六篇的机制就全部落到了代码上；再问 slime 和 AReaL 在这条链的哪一段做了不同的选择，就知道哪些是必然、哪些是取舍。**

实践：在 8 卡上用 verl 跑一个小模型的 GRPO，打开 Ray dashboard 与 profiler，把一步的时间线对到本篇讲的每个函数上；把第三、四、五篇量到的切换、同步、等待各落在时间线的哪一段标出来。

### 8. 配置、可观测与排障：从一张卡的比例到一条 hang 的排查

最后一篇把前七篇变成决策与运维：给定模型、GPU 数与任务形态，推出系统形态与配置；跑起来后看什么、坏了怎么查。

这一篇会覆盖：

- 配置的推导顺序：先按第一篇算三段的时间 → 按第二篇选形态与配比 → 训练侧的并行配置（FSDP / Megatron 的 TP / PP / EP / CP，与预训练相同的逻辑但 batch 形态不同——变长、按 token 数打包）→ 推理侧的并行配置（vLLM 的 TP / EP / DP、KV 池大小、最大并发）→ 权重同步方式；MoE 模型两侧都要 EP，长上下文训练侧要 CP；
- 全步 MFU：把三段的 FLOPs 除以全部 GPU 的峰值 × 墙钟，得到整个 RL 步的利用率——它才是要优化的数字；逐项拆解损失（rollout 长尾、切换、同步、训练 MFU、环境等待）；
- **RL 状态的 checkpoint**：除了训练状态，还有样本缓冲、数据集位置、rollout 引擎里正在生成的请求、异步下的策略版本表；恢复时重放还是丢弃进行中的 rollout；异步 trainer 的恢复语义；
- **确定性**：RL 的 reward 曲线对随机性极敏感，排障的前提是两次运行能对齐——vLLM 的确定性推理、训练侧的确定性算法、采样种子；"bitwise 对齐的 reward 曲线"是可以做到的，代价是什么；
- 常见故障：跨引擎的 NCCL 进程组 hang（训练与推理各自的通信在同一批卡上交叉）、权重同步中途的显存峰值 OOM、sleep / wake_up 之后的 CUDA graph 失效、训推 logprob 差异过大导致的 NaN 与 reward 崩塌、沙箱泄漏耗尽 CPU 集群、Ray 的对象存储溢出；每一类的信号与排查路径；
- 可观测：三段各自的 token/s 与 GPU 时间、每步的切换与同步耗时、off-policy 指标、rollout 长度分布与长尾、MoE 的专家负载均衡、环境的排队与失败率；哪些该告警；
- 与平台的接口：一个 RL 任务向调度器申请的是"两类 GPU 池 + 一个 CPU 池 + 若干服务"，gang scheduling、弹性与抢占的语义与预训练任务不同；本篇只给出引擎侧的要求。

核心问题是：

> **凌晨两点告警：reward 曲线从上升变成平台，步时间没变，没有报错。十分钟内你要判断是 staleness 涨了、是训推不一致、是某个沙箱池挂了导致 reward 全为零、还是权重同步漏了一部分参数。你需要的每一个信号，在开训前有没有采集？**

实践：为练手项目搭起完整的指标面板；人为制造三类故障（拖慢一个推理实例、关掉一半沙箱、跳过一个 bucket 的权重同步）并用面板定位；写出这个任务的配置推导记录与值班手册。本篇最后给出全系列总结。


## 贯穿全系列的实践线

系列的练手线是**一张能算到几百卡的账，和一个在 8 卡上能验证它的 GRPO 配置**。几百卡的 RL 集群不是每个读者都能拿到的，但账是可以在 8 卡上验证、再用数学外推的。各篇末尾的"实践"是给读者的动手建议，对应的是：

```text
第一篇    RL 一步账本                 三段 FLOPs · 显存 · KV · 时间 · 利用率上限（配套脚本）
第二篇    账本加系统形态              共置 / 分离 / 异步 · 配比 · 气泡与墙钟
第三篇    8 卡共置 GRPO               切换耗时与显存归属对账 · sleep level 对比
第四篇    最小的 FSDP2 → vLLM 权重同步 gather · 映射 · NCCL 广播 · collective_rpc · 各段耗时
第五篇    异步对比实验                k = 0 / 1 / 2 / 4 · 吞吐与 reward · 训推 logprob 差
第六篇    最小沙箱环境与多轮 RL       容器化执行 · 环境时间分布 · GPU 空闲比例
第七篇    verl 一步的时间线对函数     Ray dashboard · profiler · 切换 / 同步 / 等待各落在哪一段
第八篇    面板、故障注入与值班手册     全步 MFU · 三类故障定位 · 配置推导记录
```

只有第一篇的账本有配套脚本（`ai-learning-labs` 的 `rl-post-training-infra/`），后面各篇把新引入的机制直接用公式与表格记进这张账，不再单独给脚本——它们要验证的东西（切换耗时、同步秒数、staleness 曲线）都要在真实 GPU 上量，纸面模型给出的是量之前该期待的数字。到第八篇结束，读者手上有：一套能对任意模型、任务形态与 GPU 数给出三段时间、利用率上限与推荐系统形态的算法；一份在 8 卡上验证过、有外推依据的 GRPO 配置；一套包含权重同步、异步控制、环境调度、checkpoint 与告警的运行方案。

与它平行的源码阅读线：

```text
第一篇    verl  verl/trainer/ppo/ 的一步控制流（只看阶段划分）
第二篇    verl  verl/trainer/ppo/v1/ 三种 trainer 模式 · verl/single_controller/ · Ray 资源池
第三篇    vLLM  vllm/device_allocator/cumem.py · sleep / wake_up 路径；SGLang  torch_memory_saver
          verl  verl/workers/ 的 hybrid engine 与 sleep / wake_up 调用点
第四篇    verl  checkpoint engine（NCCL / NIXL / delta_sharded）· 权重名映射；Megatron-Bridge
          vLLM  collective_rpc 与 load_weights 路径
第五篇    verl  replay buffer 与 staleness 策略 · 流式 dataloader · 异步 checkpoint 恢复
第六篇    verl  agent loop · continuous token · uni-agent 网关
第七篇    verl  main_ppo 入口 · single_controller/ · workers/ · trainer/ppo/v1/；对照 slime 的三段通路与 AReaL 的 rollout controller
第八篇    verl  off-policy 与 MoE 负载指标 · 异步 checkpoint 恢复；vLLM  确定性推理
```


## 阅读路径建议

### 第一遍怎么读（全栈 / 新手读者）

```text
1 → 2
```

约 2 小时。第一篇把一步 RL 的生成、打分、训练三段的算力、显存与时间算出来，第二篇看共置、分离、异步三种形态各在交换什么——读过算法地图后训练系列的读者到这里就知道"配方"在系统上花了多少钱。权重同步、异步修正、Agent 环境、源码与排障四篇在跑真实 RL 任务时读。

### 完整学习路径

```text
1 → 2 → 3 → 4 → 5 → 6 → 7 → 8
```

### 只想搞清楚"时间去哪了、该选哪种形态"

```text
1 → 2 → 8
```

前两篇是账与形态，第八篇是配置推导。读完能为一个任务选形态、定配比、解释利用率，但不涉及机制内部。

### 负责共置或同步形态的现有任务

```text
1 → 3 → 4 → 8
```

共置的显存切换与权重同步是这类任务的主要开销。

### 准备上异步或 Agent 训练

```text
1 → 2 → 5 → 6 → 8
```

第五、六篇是异步与环境两个新问题，第八篇的 checkpoint 与确定性在异步下尤其重要。

### 主要目标是读懂框架源码

```text
2 → 4 → 7
```

第七篇是主体；第二篇提供编程模型，第四篇覆盖四个框架里最"Infra"、也最容易被忽视的权重同步子系统。


## 本系列的边界

本系列只讨论在线 RL 后训练的**系统**：生成、打分、训练三段怎样共享 GPU、怎样同步、怎样调度。以下内容与它紧邻，但不在范围内：

- **RL 算法本身**：目标函数、PPO / GRPO 及其变体、奖励模型、可验证奖励、Agent RL 的公式与 mask。它们是算法地图 L5 的[《后训练：从 SFT 到可验证奖励》](/post-training-from-sft-to-verifiable-rewards.html)（八篇）；本系列只使用"一步里生成多少 token、要几个模型、样本怎样进 loss"这些结论，第五篇讨论 off-policy 修正时只讨论它对系统的要求，不评价它对效果的影响。
- **训练引擎内部**：并行策略、FSDP / Megatron 的实现、分布式 checkpoint 的格式、容错与弹性。它们在[《大规模训练工程》](/large-scale-training-from-parallelism-to-fault-tolerance.html)；本系列把训练器当作"一个能按给定并行配置跑一步、能导出分片权重"的黑盒。
- **推理引擎内部**：调度、KV cache 管理、连续批处理、PagedAttention、投机解码、PD 分离。它们在[《大模型推理系统揭秘》](/deep-dive-into-vllm.html)；本系列只使用推理引擎的三个接口——批量生成、让渡显存、加载权重——与它的吞吐特性。
- **集合通信的实现**：NCCL 的算法与调优、RDMA。本系列只用广播、all-gather、点对点的语义与带宽。
- **集群资源层**：Kubernetes / Ray 的调度、gang scheduling、容器运行时、沙箱平台的实现。本系列第六、八篇给出 RL 任务对它们的要求，不讨论它们自身。
- **SFT 与离线方法**：SFT、DPO 一族在系统上与预训练同构（读数据、训练），没有 rollout；蒸馏里的 on-policy 蒸馏与在线 RL 同构，第五篇会提及。
- **多模态与 VLA 的 RL**：图像输入让 rollout 多一个 vision encoder、环境多一类观测，系统形态不变；正文在涉及处标注，不单列。


## 前置要求与说明

### 前置要求

- 理解分布式训练的状态与并行：知道 FSDP / Megatron 把参数、梯度、优化器状态切在哪、每参数 16 字节从哪来、TP / PP / DP / EP 各是什么（[《大规模训练工程》](/large-scale-training-from-parallelism-to-fault-tolerance.html)前两篇的内容）；
- 理解 LLM 推理引擎的基本机制：prefill 与 decode 的形态差别、KV cache 的字节数、连续批处理、TP 部署（[《大模型推理系统揭秘》](/deep-dive-into-vllm.html)前五篇的内容）；
- 知道 GRPO / PPO 一步在做什么：采样、打分、更新，参考模型与 KL，重要性比与 clip（[《后训练》](/post-training-from-sft-to-verifiable-rewards.html)第三篇的内容；本系列第一篇会用一节复述所需的最小集）；
- 会用 Ray 启动多进程任务、读 Python 源码、用 profiler 看时间线；
- 至少一台 8 卡 GPU 机器（Ampere 或更新）用于实践；几百卡的内容以计算外推与公开数据为主。

不要求：

- 用过 verl；
- 了解 RL 算法的推导；
- 了解 NCCL 内部或 CUDA 编程。

### 框架与版本基线

- **verl v0.9.0**（2026-08-14）为全系列的源码对象：统一的 v1 trainer（`sync` / `colocate_async` / `separate_async`）、model engine（FSDP / Megatron / VeOmni）、rollout server 模式、checkpoint engine（含 `delta_sharded`）、agent loop 与 uni-agent；它 pin 的 vLLM 与 SGLang 版本随文标注；
- 对照框架只在第七篇末节与个别机制处出现：**slime v0.3.0**（2026-05-31）；**AReaL** 以其论文（Fu 等 2025，arXiv 2505.24298）与当前文档的异步机制为准；OpenRLHF 只提脉络，不引源码；
- 推理引擎以 **vLLM v0.27.1** 为主（与《大模型推理系统揭秘》一致），SGLang 作为对照；训练器以 **PyTorch 2.13.0** 的 FSDP2 与 **Megatron Core 0.18.0** 为准（与《大规模训练工程》一致）；
- 硬件以 **H100 SXM**（80 GB HBM3，BF16 dense 约 989 TFLOPS）为默认分析对象，节点内 8 卡 NVLink、节点间 InfiniBand；模型以 **Llama-3-8B / Qwen3-32B** 规格的 dense 模型与 **DeepSeek-V3** 规格的 MoE 为主要算例；给出的公开数字均注明来源（HybridFlow、AReaL、DeepSeek-R1、Kimi k1.5、各框架的 release note），实测会因集群与版本而异。

这个领域的框架仍在快速迭代，正文以**机制**为主：显存让渡、布局映射、传输方式、staleness 控制、环境服务化——这些在四个框架里的写法已经趋同，比任何一个框架的配置项稳定。源码引用只到目录与关键函数，不引用行号。

### 关于"几百卡"

正文中的"几百卡"指 64–512 张 GPU 量级，这是当前开源框架公开配方（7B–70B dense、几百 B 的 MoE）的常见规模；千卡以上的 RL 任务在方法上相同，差别在权重同步的带宽压力与沙箱集群的规模再高一个量级，正文会在相关位置标注。


## 章节目录

1. [负载画像：一步 RL 里发生什么——生成、打分、训练的算力、显存与时间账](/rl-step-anatomy-rollout-reward-train.html)
2. [系统形态：共置、分离与异步——三种拓扑的利用率、气泡与正确性代价](/rl-system-topologies-colocate-disaggregate-async.html)
3. [共置：训练器与推理引擎在同一组 GPU 上共存——显存归属切换与它的代价](/colocated-trainer-and-rollout-engine-memory-handoff.html)
4. [权重同步：从训练分片到推理分片——布局映射、传输方式、量化与增量同步](/weight-sync-from-training-shards-to-inference-shards.html)
5. [异步与 off-policy：staleness、部分 rollout、训推不一致与样本缓冲](/async-rl-staleness-partial-rollout-and-off-policy-correction.html)
6. [Agentic rollout：多轮、工具、沙箱集群与环境服务](/agentic-rollout-multi-turn-tools-sandboxes-and-environment-services.html)
7. [verl 源码导读：从一个 GRPO 配置追到每个 worker（附 slime 与 AReaL 的对照）](/verl-source-walkthrough-from-a-grpo-config-to-every-worker.html)
8. [配置、可观测与排障：GPU 配比、全步 MFU、RL 状态的 checkpoint 与常见故障](/rl-post-training-configuration-observability-and-troubleshooting.html)


## 最终目标

读完这套系列之后，面对任何一个 RL 后训练任务——无论是自己配的、别人交接的、还是利用率上不去要接手排查的——读者应该能够回答：

```text
一步里生成、打分、训练各花多少 GPU·秒？上限是多少？    → 第一篇：三段的账
这个任务该共置、分离还是异步？GPU 怎么分？             → 第二篇：三种形态的交换
每步的显存切换搬多少字节、要几秒？                     → 第三篇：让渡接口与切换代价
新权重怎么进推理引擎？传多少、走哪条链路、几秒？        → 第四篇：布局映射与传输
异步之后 reward 曲线变了，是 staleness 还是训推不一致？  → 第五篇：off-policy 的信号
Agent 训练的沙箱要多少并发？GPU 在等什么？             → 第六篇：环境账与长尾调度
verl 把权重同步 / 样本通路实现在哪？                   → 第七篇：源码导读
利用率差在哪一项？坏了从哪查？                         → 第八篇：全步 MFU 与排障
```

最终目标是三种能力：

1. **算账能力**：给定模型、任务形态（对话 / 推理 / Agent）与 GPU 数，算出三段的时间与显存、推荐系统形态与配比，并解释利用率的理论上限；
2. **实现能力**：读懂并修改 verl（或同类框架）里的显存让渡、权重同步、异步控制与环境接入，或不依赖框架把这几段自己写出来；
3. **运维能力**：为一个持续数天到数周的 RL 任务设计 checkpoint、确定性、监控与告警方案，让它在推理引擎、训练器、沙箱集群三方都可能出问题的前提下把全步利用率维持在可解释的水平。

这是 AI-Infra 引擎层里训练与推理两条主线**会合**的那一块：训练引擎与推理引擎不再各管一段，而是在同一个循环里、同一组 GPU 上轮流工作。
