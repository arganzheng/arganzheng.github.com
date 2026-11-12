---
layout: post
series: deep-dive-into-vllm
title: 大模型推理系统揭秘（03）：鸟瞰 vLLM：一个请求如何穿过整个推理系统？
tags: [AI, AI-Infra, Inference]
catalog: true
updated: 2026-10-18
---

> **NOTE** 本文基于 vLLM v0.27.1（tag `6e448d0`，commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`，2026-08-11）源码剖析。文中文件路径、类名和函数名均以该版本为准；vLLM 迭代很快，阅读时请以你手上的版本对照。文中的 **V1** 指引擎架构及其源码包 `vllm/v1/`，不是一个可选的版本：旧的 V0 引擎在 v0.11.0（2025-10-02）中被整体移除，release notes 原话是 “V1 is the only engine in the codebase now”，所以 v0.27.1 里只有这一套引擎，没有 “V2 引擎”。名字里带 V2 的只是两个组件——模型执行层的新 ModelRunner 和 Ray 后端的 `RayExecutorV2`，见[第二章第 4 节](#4-executorworker-与-modelrunner模型执行的设备抽象)。
>
> **版本提示 @2026-11-12**：v0.29.0（2026-09-09）起，Model Runner V2（`vllm/v1/worker/gpu/`）成为所有模型的默认实现，V1 的 `GPUModelRunner`（`vllm/v1/worker/gpu_model_runner.py`）被宣布弃用，只在 V2 尚不支持的配置下作为回退。本文第二章第 4 节里 ModelRunner 的部分讲的是 V1 路径；API Server → AsyncLLM → EngineCore → Executor → Worker 的分层与进程间通信不变。截至 v0.32.0 的其他变化见[总纲的版本与硬件基线](/deep-dive-into-vllm.html#版本与硬件基线)。


前两篇分别定义了问题（LLM Serving 为什么难）和尺子（用什么指标衡量）。在深入调度、KV Cache、GPU 执行等单个战场之前，需要先有一张全景图：一个请求进入 vLLM 之后，会经过哪些模块，以什么形式被传递，最后如何变成 token 返回给用户。

本篇的核心问题是：

> **一个请求如何穿过 vLLM 的整个推理系统——哪个模块负责决策、哪个模块负责执行，它们之间传递的到底是什么？[^q0]**

## 一、总览：用 4+1 视图看同一个系统

### 1. Kruchten 4+1：五个视角看同一个系统

借用软件架构里的 Kruchten 4+1 视图模型来看 vLLM。**逻辑视图**说明模块分工（第二章的静态拓扑）；**开发视图**把模块放回源码目录，说明去哪读代码（第三章）；**进程视图**说明模块在哪个进程、怎样交互（第四章）；**部署/物理视图**说明进程怎么落到 GPU 和节点上（第五章）。最后的 **+1 场景视图**沿时间轴追踪一次请求（第六章，含请求流与数据流两个子场景）——它把前四个视图串成一条线。

这样可以分清谁保存请求状态、谁决定这一步跑什么，以及 token 和调度结果怎样跨进程移动。

### 2. 本文的章节安排

| 章 | 视图 | 内容 |
|---|---|---|
| 二 | 逻辑视图 | 静态系统拓扑，自顶向下：API Server 与 AsyncLLM、EngineCore、Scheduler 与 KVCacheManager、Executor → Worker → ModelRunner |
| 三 | 开发视图 | 模块 ↔ 源码目录 ↔ 所在进程的对照表 |
| 四 | 进程视图 | API Server、EngineCoreProc、WorkerProc 的职责与 IPC |
| 五 | 部署/物理视图 | 进程到 GPU、节点的映射；会改变进程图的部署方式 |
| 六 | 场景视图（+1） | 两个子场景：1. 请求流（一次请求的完整生命周期，HTTP → SSE [DONE]）；2. 数据流（Token 如何穿过整个 Serving 栈） |
| 七 | 本文小结 | 模块分工，以及与第一篇“四问”的对应 |
| 八 | 自测 | 7 道题 |

Table: 本文的章节安排

## 二、逻辑视图：静态系统拓扑（自顶向下）

vLLM 把系统分成**控制面**和**数据面**两半：控制面在 CPU 上，只处理元数据——请求 id、token id、block id、每轮的调度决策；数据面在 GPU 上，只处理张量——模型权重、前向激活和 KV Cache。两半之间只有一个接口：EngineCore 把 `SchedulerOutput` 交给 Executor 下发，Worker 把 `ModelRunnerOutput` 交回来。

```mermaid
%% 图：vLLM 的逻辑视图：CPU 上的控制面只处理元数据，GPU 上的数据面只处理张量，两者只经 SchedulerOutput / ModelRunnerOutput 交互
flowchart TB
    Client["Client / OpenAI SDK"]

    subgraph CP["控制面 · CPU · 只处理元数据"]
        direction TB
        API["<b>API Server</b><br/>FastAPI + uvicorn"]
        AsyncLLM["<b>AsyncLLM</b><br/>tokenize / detokenize、流式输出"]
        EngineCore["<b>EngineCore</b><br/>step() 主循环"]
        Scheduler["<b>Scheduler</b><br/>请求状态、每轮给谁几个 token"]
        SOManager["<b>StructuredOutputManager</b><br/>语法约束 bitmask"]
        KVCacheManager["<b>KVCacheManager</b><br/>请求 → block id 列表"]
        BlockPool["<b>BlockPool</b><br/>空闲块队列、引用计数、前缀哈希表"]
    end

    EX["Executor（EngineCore 进程内的投递对象）<br/>共享内存 MessageQueue 广播 / 收回"]

    subgraph DP["数据面 · GPU · 只处理张量"]
        direction TB
        Worker["GPU Worker × N（每个 TP / PP rank 一个）<br/>GPUModelRunner：block id → 输入张量、前向、采样"]
        Attn["<b>Attention backend</b><br/>FlashAttention / FlashInfer"]
        Mem["<b>GPU 显存</b><br/>权重分片 · KV Cache · 激活"]
        NCCL["<b>NCCL</b><br/>rank 之间 all-reduce / P2P 传激活"]
    end

    Client -->|HTTP| API
    API --> AsyncLLM
    AsyncLLM -->|"EngineCoreRequest（ZMQ）"| EngineCore
    EngineCore --> Scheduler
    Scheduler --> SOManager
    Scheduler --> KVCacheManager
    KVCacheManager --> BlockPool
    BlockPool ~~~ EX
    EngineCore ==>|"↓ SchedulerOutput：请求 id、token 数、新 block id"| EX
    EX ==>|广播给每个 rank| Worker
    Worker ==>|"↑ ModelRunnerOutput：采样出的 token id"| EngineCore
    Worker --> Attn
    Attn --> Mem
    Worker -.- NCCL

    classDef cp fill:#eaf2fb,stroke:#5b8fd6,color:#1d3557
    classDef dp fill:#fdf0e6,stroke:#d9822b,color:#5a2e0e
    classDef edge fill:#f4f4f4,stroke:#888,color:#333
    class API,AsyncLLM,EngineCore,Scheduler,SOManager,KVCacheManager,BlockPool cp
    class Worker,Attn,Mem,NCCL dp
    class EX,Client edge
```

图中粗箭头就是控制面与数据面的边界。越过边界的东西如下：

1. **向下**：`SchedulerOutput`（`vllm/v1/core/sched/output.py`）。首次调度的请求以 `NewRequestData` 下发 prompt token ids、采样参数和分到的 block ids；已在 Worker 上的请求以 `CachedRequestData` 只下发增量——新 block ids、`num_computed_tokens`；外加每个请求这一轮的 token 数 `num_scheduled_tokens`。全是 Python 的 int 和 list。Worker 侧的 `InputBatch` 常驻一份 block table（`BlockTable`，`vllm/v1/worker/block_table.py`），收到增量后 `append_row`，`commit_block_table` 拷到 GPU，再由 `compute_slot_mapping` 的 Triton kernel 在 GPU 上把 block id 换算成每个 token 在 KV Cache 里的槽位。
2. **向上**：`ModelRunnerOutput`（`vllm/v1/outputs.py`），主体是 `req_ids` 和 `sampled_token_ids: list[list[int]]`，可选 logprobs。只有 `output_rank` 一个 Worker 回传。
3. **例外**：直接传嵌入的请求（`NewRequestData.prompt_embeds`）和多模态请求（`mm_features`）会随首次调度带一次输入张量；结构化输出的语法 bitmask 是一块 int32 数组，随 `sample_tokens()` 下发；pooling 模型的 `pooler_output` 是张量。这些都在 CPU 内存里，不涉及显存。

从不越过边界的是**权重、激活和 KV Cache**。KVCacheManager 管的"块"只是 `KVCacheBlock` 对象上的 `block_id`、`ref_cnt` 和一个可选哈希；真正的 KV 张量由 Worker 在初始化时按 `kv_cache_config` 一次性分配在显存里，block id 是它的下标。分配、释放、前缀复用、抢占在控制面都是整数运算，不发生任何显存拷贝。TP / PP 之间的激活由 Worker ranks 经 NCCL 直接交换，不经 EngineCore。

这样分开有三个直接收益：

1. **CPU 工作与 GPU 前向重叠。** `EngineCore.step()` 用 `execute_model(scheduler_output, non_block=True)` 发出前向后，不等结果就去算这一轮的语法 bitmask（`Scheduler.get_grammar_bitmask`），再 `sample_tokens()`；开启 async scheduling（v0.27.1 在没有不兼容选项时默认开启）后 `max_concurrent_batches = 2`，`step_with_batch_queue()` 在第 N 轮还在 GPU 上跑时就调度第 N+1 轮。
2. **边界消息的大小与模型大小无关。** 一轮的 `SchedulerOutput` 只和 batch 里的请求数、新 block 数有关，7B 和 670B 的模型一样，所以 `MultiprocExecutor` 用一块共享内存环形缓冲就能广播给所有 rank。
3. **调度状态只有一份。** Scheduler 和 KVCacheManager 只在 EngineCore 里存在，所有 rank 拿到同一份决定；换 Uni / Multiproc / Ray 部署，改的只是中间那个 Executor 对象。

### 1. API Server 与 AsyncLLM：异步并发处理的桥头堡

入口位于 `vllm/entrypoints/openai/api_server.py`，基于 FastAPI 构建的 HTTP 服务器，实现了 OpenAI 兼容的 `/v1/chat/completions`、`/v1/completions` 等 REST API。

`AsyncLLM`（`vllm/v1/engine/async_llm.py`）是面向外部的异步引擎接口。它自己不做重活，而是把工作交给三个同目录下的协作者：

- `InputProcessor`（`input_processor.py`）：把文本 / 多模态输入 tokenize、校验采样参数，产出 `EngineCoreRequest`；
- `EngineCoreClient`（`core_client.py`）：与 `EngineCore` 通信的客户端。默认部署下 `EngineCore` 运行在**独立进程**中，两者通过 ZMQ + msgspec 收发 `EngineCoreRequest` / `EngineCoreOutputs`（`AsyncMPClient`）；单进程调试时用 `InprocClient`。这条进程边界是 vLLM 让 HTTP 处理、tokenize / detokenize 与 GPU 调度循环互不阻塞的关键；
- `OutputProcessor`（`output_processor.py`）：接收 `EngineCoreOutputs`，做 detokenize、stop 条件检查、logprobs 整理，产出 `RequestOutput` 交给 API 层流式返回。

因此 `AsyncLLM` 的职责是：管理请求的异步生命周期、串起上述三者、支持 SSE 流式响应，以及数据并行（DP）场景下的多引擎协调。

### 2. EngineCore：推理系统的中央调度大脑

`EngineCore`（`vllm/v1/engine/core.py`）是引擎的核心，它的 `step()` 方法驱动整个推理循环：

```python title='EngineCore 的 step()：调度、执行、更新'
# vllm/v1/engine/core.py (简化)
class EngineCore:
    """Inner loop of vLLM's Engine."""

    def __init__(self, vllm_config, executor_class, ...):
        # 1. 初始化模型执行器
        self.model_executor = executor_class(vllm_config)

        # 2. 初始化 KV Cache
        kv_cache_config = self._initialize_kv_caches(vllm_config)

        # 3. 初始化调度器
        Scheduler = vllm_config.scheduler_config.get_scheduler_cls()
        self.scheduler = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            ...
        )
```

### 3. Scheduler 与 KVCacheManager：资源管理双核

`Scheduler`（`vllm/v1/core/sched/scheduler.py`）持有每个请求的权威状态：running / waiting 两个队列，以及每个 `Request` 已算到第几个 token（`num_computed_tokens`）。`KVCacheManager`（`vllm/v1/core/kv_cache_manager.py`）记录每个请求占了哪些 KV block，底下的块池是 `BlockPool`（`vllm/v1/core/block_pool.py`）。三者都在 EngineCore 进程里，处理的全是整数。

**每轮的两条上限。** `schedule()` 开头令 `token_budget = max_num_scheduled_tokens`（未单独设置时取 `max_num_batched_tokens`），同时在跑的请求数不超过 `max_num_running_reqs`（取自 `max_num_seqs`）。

**没有 prefill / decode 两个阶段。** `schedule()` 开头的注释说得很直接：每个请求只有 `num_computed_tokens` 和 `num_tokens_with_spec`（prompt + 已生成 + 投机 token）两个数，调度器每轮给请求分若干 token，让前者追上后者。新请求一次分到几千个 token，就是 prefill；running 请求分到 1 个，就是 decode；预算不够时只分一部分、下一轮接着算，就是 chunked prefill（`long_prefill_token_threshold` 可再给单个请求封顶）。

一轮 `schedule()` 的顺序：

1. **先排 running 队列。** 对每个请求算出 `num_new_tokens`，截到剩余预算，再调 `kv_cache_manager.allocate_slots(request, num_new_tokens, ...)` 给这些 token 要 block。
2. **块不够就抢占。** `allocate_slots` 返回 `None` 表示空闲块不够。默认 FCFS 策略弹出 running 队列尾部的请求（priority 策略选优先级最低的），`_preempt_request` 释放它的全部 block、把 `num_computed_tokens` 清零、放回 waiting 队首，然后重试分配。被抢占的请求下次调度时重新 prefill，前缀缓存里还没被驱逐的块可以直接复用。
3. **再排 waiting 队列。** 本轮只要发生过抢占，就不接纳新请求。否则逐个取 waiting 请求：running 数到上限就停；`num_computed_tokens == 0` 的请求先调 `kv_cache_manager.get_computed_blocks(request)` 查前缀缓存，命中的 token 不再计算；剩下的 token 走 `allocate_slots`，失败就停止接纳。
4. **打包 `SchedulerOutput`。** 首次调度的请求放进 `scheduled_new_reqs`（`NewRequestData`，带完整 prompt token ids、采样参数、block ids），已在 Worker 上的放进 `scheduled_cached_reqs`（`CachedRequestData`，只带新 block ids 等增量），再加上每个请求的 `num_scheduled_tokens`、总数 `total_num_scheduled_tokens`、`finished_req_ids` / `preempted_req_ids`，以及投机解码、多模态 encoder 输入等扩展字段。

模型跑完后，`update_from_output()` 把采样出的 token 追加到请求上、检查停止条件；结束的请求经 `_free_request` 调 `kv_cache_manager.free(request)` 归还 block。

`KVCacheManager` 自己不直接管块：它经 `KVCacheCoordinator`（`vllm/v1/core/kv_cache_coordinator.py`）为每个 KV cache group（例如全注意力层与滑动窗口层各一组）调对应的 `SingleTypeKVCacheManager`，这些 manager 共用一个 `BlockPool`。`BlockPool` 有三样东西：

1. **`blocks`**：全部 `KVCacheBlock`。每个块只有 `block_id`、`ref_cnt` 和一个可选的 block hash，是显存里 KV 张量的下标，不是显存本身。
2. **`free_block_queue`**（`FreeKVCacheBlockQueue`）：`ref_cnt` 为 0 的块，按驱逐顺序排列。`get_new_blocks` 从队头取；取出的块若还带哈希，先从哈希表删掉，这就是驱逐。`free_blocks` 让 `ref_cnt` 减 1，归零的块里没有哈希的放到队头（最先复用），有哈希的放到队尾（还留在前缀缓存里，最晚驱逐），整体是一个 LRU。
3. **`cached_block_hash_to_block`**：block hash → 块。`get_computed_blocks` 按请求 token 逐块算出的哈希在这里查最长命中；`allocate_slots` 落实命中时调 `touch`，`ref_cnt` 加 1，块若正躺在空闲队列里就先摘出来。多个请求共享同一段前缀，就是同一组块的 `ref_cnt` 大于 1。

```mermaid
%% 图：一次 schedule() 里 Scheduler、KVCacheManager 与 BlockPool 的交互：先排 running、块不够就抢占，再接纳 waiting
flowchart TB
    S["<b>Scheduler.schedule()</b><br/>token_budget = max_num_scheduled_tokens"]
    R["<b>取 running 队列下一个请求</b><br/>num_new_tokens 截到剩余预算"]
    A["KVCacheManager.allocate_slots()<br/>BlockPool.get_new_blocks()：空闲块够吗？"]
    P["<b>Scheduler._preempt_request()</b><br/>KVCacheManager.free() → BlockPool.free_blocks()<br/>ref_cnt − 1，请求放回 waiting 队首"]
    OK["<b>记入本轮</b><br/>token_budget −= num_new_tokens"]
    W["<b>取 waiting 队首请求</b><br/>running 数未到 max_num_running_reqs"]
    G["KVCacheManager.get_computed_blocks()<br/>查 BlockPool 的 hash → block 表，得到命中的块"]
    A2["KVCacheManager.allocate_slots()<br/>命中块 touch：ref_cnt + 1<br/>其余 token 要新块：够吗？"]
    OUT["<b>打包 SchedulerOutput</b><br/>交给 Executor"]

    S --> R
    R --> A
    A -->|够| OK
    A -->|不够| P
    P -->|重试| A
    OK -->|还有 running 请求且有预算| R
    OK -->|"running 排完，本轮无抢占"| W
    OK -->|"running 排完，本轮有抢占"| OUT
    W --> G
    G --> A2
    A2 -->|够，下一个 waiting 请求| W
    A2 -->|不够 / 预算用完| OUT

    classDef sched fill:#eaf2fb,stroke:#5b8fd6,color:#1d3557
    classDef kv fill:#fdf0e6,stroke:#d9822b,color:#5a2e0e
    class S,R,OK,W,OUT,P sched
    class A,G,A2 kv
```

图里蓝色是 Scheduler 自己的步骤，橙色是对 KVCacheManager 的调用，节点里写出的 `BlockPool.*` 是这些调用最终落到的块池操作。调度策略（FCFS 与 priority、公平性、抢占代价）在[《Scheduler：GPU 这一轮到底给谁用？》](/scheduler-batch-and-fairness.html)里展开；块的布局、前缀缓存哈希和驱逐在[《KV Cache：LLM Serving 的第一号内存问题》](/kv-cache-memory-core.html)里展开。

### 4. Executor、Worker 与 ModelRunner：模型执行的设备抽象

EngineCore 下面的执行层分三层，从外到内：**Executor** 决定一轮 batch 交给哪些 Worker、怎么交；**Worker** 管一个设备（一张 GPU、一个 TP / PP rank）；**ModelRunner** 在这个设备上真正把一轮 batch 跑完。

```mermaid
%% 图：Executor 与 Worker：一次 execute_model() 分发给多个 GPU Worker，各自的 ModelRunner 持有分片权重与 KV Cache
graph TB
    EX["<b>Executor</b><br/><i>执行抽象：单设备或多设备模型调用</i>"]
    EX --> W0 & W1 & WN
    subgraph W0["Worker 0 (GPU 0)"]
        direction TB
        M0["ModelRunner"] --- S0["<b>Model Weights（分片）</b><br/>KV Cache"]
    end
    subgraph W1["Worker 1 (GPU 1)"]
        direction TB
        M1["ModelRunner"] --- S1["<b>Model Weights（分片）</b><br/>KV Cache"]
    end
    subgraph WN["Worker N (GPU N)"]
        direction TB
        MN["ModelRunner"] --- SN["<b>Model Weights（分片）</b><br/>KV Cache"]
    end
    NC["NCCL：TP all-reduce / PP 激活收发"]
    W0 & W1 & WN <-.-> NC
```

常见 GPU 部署中，Executor 会把一次 `execute_model()` 调用分发给一个或多个 Worker；Worker 再调用本地的 `ModelRunner` 执行模型前向、采样和 KV Cache 读写。单卡、同机多进程、Ray 分布式和 external launcher 的进程/设备映射并不完全相同，因此不应把“一个 GPU 固定绑定一个 Worker 进程”写成绝对规则。

**Executor。** `Executor`（`vllm/v1/executor/abstract.py`）是模型执行抽象层，负责在一个设备或多个设备上执行模型，而不是 Scheduler 本身。v0.27.1 中可见的主要实现包括：
- `UniProcExecutor`：单进程单卡
- `MultiprocExecutor`：多进程多卡（同机）
- `RayDistributedExecutor`：Ray 分布式（直接继承 `Executor`）
- `RayExecutorV2`：Ray 分布式的新实现（`distributed_executor_backend="ray"` 时默认用它，`VLLM_USE_RAY_V2_EXECUTOR_BACKEND=0` 才退回 `RayDistributedExecutor`），注意它继承的是 `MultiprocExecutor` 而非 `Executor`——即复用同一套 worker 进程管理逻辑，只把进程的**拉起方式**换成 Ray（`RayWorkerProc(WorkerProc)`）
- `ExecutorWithExternalLauncher`：外部 launcher 场景（继承 `UniProcExecutor`）

这条继承链本身就说明了一件事：**"用不用 Ray"是部署方式的差异，不是执行模型的差异。** Executor 这层抽象的价值就在于把"进程怎么起、卡怎么分"和"一轮 batch 怎么执行"彻底分开，所以 `RayExecutorV2` 才能靠换掉进程拉起方式来复用同机多进程的全部逻辑。

**Worker。** `Worker`（`vllm/v1/worker/gpu_worker.py`）是一个设备的管家，通常负责：
- 初始化设备和本 rank 的分布式环境（TP / PP 通信组，`init_device()`）；
- 加载模型权重的分片（Tensor Parallel 下每卡持有一部分，`load_model()`）；
- 测显存余量（`determine_available_memory()`）、按 `kv_cache_config` 分配 KV Cache 物理显存（`initialize_from_config()`）；
- 持有一个 ModelRunner，把 Executor 发来的 `execute_model()` 转给它。

**ModelRunner。** `GPUModelRunner`（`vllm/v1/worker/gpu_model_runner.py`，本系列图中简写为 ModelRunner）是 Worker 里真正“跑一步”的对象。它常驻一份持久的 `InputBatch`（每个在跑请求一行：token ids、block table、采样参数），每轮做三件事：
1. **准备输入**（`_prepare_inputs()`）：按 `SchedulerOutput` 的增量更新 `InputBatch`，算出这一轮的 input ids、positions 和 slot mapping，组成 attention backend 要的元数据；
2. **前向**（`execute_model()`）：调模型和 attention backend，读写 KV Cache，能用 CUDA Graph 时走捕获好的图；
3. **采样**（`sample_tokens()`）：从 logits 采出每个请求的下一个 token，打包成 `ModelRunnerOutput` 交回。

Worker 管“这张卡上有什么”，ModelRunner 管“这一轮怎么算”，所以[《GPU 执行：如何让每个 Token 算得更快？》](/gpu-execution-kernels-and-graphs.html)展开的主要是 ModelRunner。v0.27.1 另有一份新实现 Model Runner V2（`vllm/v1/worker/gpu/model_runner.py`）：`VLLM_USE_V2_MODEL_RUNNER=1` 强制启用，未设置时只对少数架构（如 `DeepseekV2ForCausalLM`、`Qwen2MoeForCausalLM`）和 PCP、diffusion 等少数特性默认启用（`VllmConfig.use_v2_model_runner`，`vllm/config/vllm.py`）。文中以默认的 `GPUModelRunner` 为主线。

## 三、开发视图：模块在源码树的哪里

逻辑视图给了模块的名字，开发视图回答读代码时最直接的问题：每个模块在源码树的哪个目录、跑在哪个进程里。下表按 v0.27.1（tag `6e448d0`）核实，进程 A / B / C 沿用第四章的编号（A = API Server，B = EngineCoreProc，C = WorkerProc）。

| 源码位置 | 逻辑视图里的模块 | 所在进程 |
|---|---|---|
| `vllm/entrypoints/openai/` | API Server：FastAPI 路由、`ChatCompletionRequest` 等协议对象 | A |
| `vllm/v1/engine/async_llm.py` | `AsyncLLM`：异步请求管理、流式输出 | A |
| `vllm/v1/engine/input_processor.py` | `InputProcessor`：tokenize、构造 `EngineCoreRequest` | A |
| `vllm/v1/engine/output_processor.py` + `detokenizer.py` | `OutputProcessor` / `IncrementalDetokenizer`：增量 detokenize、`RequestOutput` | A |
| `vllm/v1/engine/core_client.py` | `AsyncMPClient` / `EngineCoreClient`：API 边界的 ZMQ 客户端 | A（client 侧） |
| `vllm/v1/engine/core.py` | `EngineCore` / `EngineCoreProc`：主循环、进程入口 | B |
| `vllm/v1/core/sched/` | `Scheduler`、`SchedulerOutput` / `NewRequestData` / `CachedRequestData` | B |
| `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager`：块分配与复用 | B |
| `vllm/v1/engine/coordinator.py` + `utils.py` | `DPCoordinator`、进程启动与握手 | B 侧 |
| `vllm/v1/executor/` | `Executor` 抽象与 `Multiproc` / `UniProc` / `Ray` 实现：拉起并驱动 Worker | B 内，拉起 C |
| `vllm/v1/worker/gpu_worker.py` | `Worker`：设备与 KV cache 显存 | C |
| `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner`：输入准备、前向、采样 | C |
| `vllm/model_executor/models/` | 各模型定义；TP 下每卡加载自己的权重分片 | C |
| `vllm/v1/attention/backends/` | FlashAttention、Triton 等 attention backend 实现 | C |

Table: 模块、源码位置与进程的映射（vLLM v0.27.1）

一个值得记住的边界：`vllm/v1/engine/` 下的文件分属两个进程——`async_llm` / `input_processor` / `output_processor` 在 A，`core` / `core_client` 的服务端在 B，目录按功能分不按进程分。

### 1. 子系统划分与包图

UML 的开发视图看的是子系统划分与包依赖。把上表按子系统归组，服务一条请求真正相关的包就六组，依赖单向向下：

![vLLM v0.27.1 的包图：entrypoints、v1.engine、v1.core、v1.executor、v1.worker、model_executor/distributed/csrc 六组子系统及依赖方向](/img/in-post/vllm-request-lifecycle-package-view.svg)

- **entrypoints**（`vllm/entrypoints/openai/`）：协议层。FastAPI 路由、`ChatCompletionRequest` 等 msgspec/pydantic 模型、`serve` CLI 都在这里，只依赖 `v1.engine` 的客户端侧。
- **v1.engine**（`vllm/v1/engine/`）：引擎壳。`async_llm` / `input_processor` / `output_processor` / `detokenizer` 是 A 侧；`core.py`（`EngineCore` / `EngineCoreProc` / `DPEngineCoreProc`）与 `core_client.py` 的服务端是 B 侧；`coordinator.py`、`utils.py` 管 DP 协调与进程启动。
- **v1.core**（`vllm/v1/core/`）：调度域。`sched/`（Scheduler、`SchedulerOutput`）、`kv_cache_manager`、`block_pool`、`encoder_cache_manager`，全部活在 B。
- **v1.executor**（`vllm/v1/executor/`）：执行抽象。`abstract.py` 的 `Executor` 基类派生 `UniProcExecutor` / `MultiprocExecutor` / `RayExecutorV2` / `RayDistributedExecutor`，是 B 内拉起并驱动 C 的唯一通道。
- **v1.worker**（`vllm/v1/worker/`）：设备域。`gpu_worker`、`gpu_model_runner`、`gpu_input_batch`，每个 C 进程一份。
- **执行依赖**：`vllm/model_executor/models/`（各模型定义）、`vllm/v1/attention/backends/`（FA/Triton 等 backend）、`vllm/distributed/`（`parallel_state`、`pynccl`、`shm_broadcast`）、`csrc/`（CUDA/C++ kernel）。

### 2. 构建与打包：Python 包 + 编译扩展

发布物是一个 wheel：`vllm` 是纯 Python 包，`csrc/` 下的 CUDA/C++ 源码由 `setup.py` 驱动 CMake 编译成 CPython 扩展 `vllm._C`（自定义 kernel：paged attention、量化、moe 等）与 `vllm._moe_C`（MoE 专用 kernel），随 wheel 分发；另有 `vllm/attention/_fa3` 等 FlashAttention/flashinfer 编译产物作可选 backend。模型权重、attention backend、kernel dispatch 全在 C 侧进程按需 dlopen/import 它们，A、B 两侧进程从不碰 `csrc` 产物。

## 四、进程视图：谁在什么进程里，如何交互

静态拓扑图画模块之间的调用；这张图把它们放回默认部署的进程和线程里。API Server 处理协议和文本，`EngineCoreProc` 管请求状态和调度，每个 `WorkerProc` rank 执行模型。

![vLLM 默认多进程部署：API Server、EngineCoreProc 与 WorkerProc 的线程和 IPC](/img/in-post/vllm-request-lifecycle-process-view.svg)

### 1. API Server：处理协议与文本

FastAPI 的 OpenAI route 调用同进程里的 `AsyncLLM`。`InputProcessor` 产出 `EngineCoreRequest`；`OutputProcessor` 把 `EngineCoreOutputs` 增量 detokenize 成 `RequestOutput`，输出循环由 asyncio task 驱动（`vllm/entrypoints/openai/api_server.py`、`vllm/v1/engine/async_llm.py`）。

### 2. EngineCoreProc：持有调度状态

`EngineCoreProc` 运行 `EngineCore`：输入、输出 socket I/O 各有线程，主线程循环调度与更新。`Scheduler` 持有权威 `Request` 状态和 `KVCacheManager` 元数据；默认 `MultiprocExecutor` 下，GPU forward 留在 Worker 侧（`vllm/v1/engine/core.py`、`vllm/v1/core/sched/scheduler.py`）。

注意 **Executor 不是一个进程**：它是 `EngineCore` 构造时创建的一个进程内对象（`vllm/v1/engine/core.py` 里 `self.model_executor = executor_class(vllm_config)`），活在进程 B 的主线程里。所以逻辑视图里的 Executor 在这张进程图里落在 B 内部——它不占进程，它的作用是封装“怎么把一轮调度交给 Worker”：`UniProc` 在本进程直接调，`Multiproc` 经共享内存 MessageQueue 广播给 `WorkerProc`，`Ray` 走 actor。换一个部署形态时 EngineCore 的代码一行不改，改的只是这个对象。

### 3. WorkerProc：一个模型并行 rank

`MultiprocExecutor` 为每个模型并行 rank 启动 `WorkerProc`。每个进程初始化 `Worker` 和 `GPUModelRunner`，加载本 rank 的权重分片，并在 GPU 上读写 KV cache；TP/PP 通信由 Worker ranks 直接完成，不经过 EngineCore（`vllm/v1/executor/multiproc_executor.py`、`vllm/v1/worker/gpu_worker.py`、`vllm/v1/worker/gpu_model_runner.py`、`vllm/distributed/communication_op.py`、`vllm/distributed/parallel_state.py`、`vllm/distributed/device_communicators/pynccl.py`）。

### 4. 两道边界上传什么

![API Server、EngineCoreProc 与 WorkerProc 之间的 IPC 边界和消息内容](/img/in-post/vllm-request-lifecycle-process-boundaries.svg)

API 边界传 `EngineCoreRequest` 与 `EngineCoreOutputs`；执行边界传 `SchedulerOutput` 与 `ModelRunnerOutput`。模型 forward 的 activations 不经 EngineCore；TP/PP activations 在 Worker ranks 之间传递（`vllm/v1/engine/core_client.py`、`vllm/v1/engine/core.py`、`vllm/v1/executor/multiproc_executor.py`、`vllm/distributed/device_communicators/shm_broadcast.py`）。

### 5. 启动顺序与就绪握手

单 API Server 时，`vllm serve` 在当前进程运行 `run_server()`；`AsyncLLM` 创建 `AsyncMPClient`，后者绑定 ZMQ sockets 并通过 `launch_core_engines()` 启动 EngineCore 进程。`--api-server-count > 1` 时，`run_multi_api_server()` 先启动 EngineCore，再由 `APIServerProcessManager` 启动 API Server 子进程。每个 EngineCore 在自己进程内构造 Executor（`EngineCore.__init__`），`MultiprocExecutor.__init__` 再为每个 rank fork 出 `WorkerProc`（`WorkerProc.make_worker_process`）。

EngineCore 先发 `HELLO` 注册，启动器返回地址和配置；初始化完成后，EngineCore 报告 `READY`。API client 还会等 EngineCore 在输入 socket 上发送 `EngineCoreReadyResponse`。Worker 初始化完成后通过 ready pipe 报告 `READY` 并交回 response queue 的 handle；Executor 随后等待各队列的 ready barrier（`vllm/entrypoints/cli/serve.py`、`vllm/entrypoints/openai/api_server.py`、`vllm/v1/engine/async_llm.py`、`vllm/v1/engine/core_client.py`、`vllm/v1/engine/utils.py`、`vllm/v1/engine/core.py`、`vllm/v1/executor/multiproc_executor.py`、`vllm/v1/utils.py`）。

### 6. 一个 decode step 怎样穿过三个进程

![vLLM 一个 decode step 在 API Server、EngineCoreProc 与 WorkerProc 间的时序](/img/in-post/vllm-request-lifecycle-decode-step.svg)

EngineCore 收到请求后，主线程调度出这一轮的 `SchedulerOutput`，交给进程内对象 `Executor`（默认 `MultiprocExecutor`）广播给所有 Worker——这条链路是 `EngineCore → Executor → WorkerProc`，进程视图那张图里 Executor 画在进程 B 内部，上面这张时序图里它是 B 侧的一小段激活。Worker 执行模型并由 `output_rank` 返回结果；EngineCore 更新请求，再把 `EngineCoreOutputs` 交给输出线程。API Server 收到 token ids 后做 detokenize，最后由 HTTP handler 写成 SSE。图里只展开一个 decode step；prefill 也走同一组进程边界。

这里的广播是一写多读：

- Executor 只写一次，写进 `rpc_broadcast_mq`（`MessageQueue`）。同机的 Worker 从共享内存环形缓冲里读，跨机的经 ZMQ 发布订阅收，每个 WorkerProc 读到的是同一份 `SchedulerOutput`。
- 要发给每个 Worker，是因为 TP / PP 下每个 Worker 只有一部分权重和自己那份 KV cache，大家要一起算同一个 batch：这一轮跑哪些请求、每个请求几个 token、KV 写进哪些 block，所有 rank 必须一致，前向里的 all-reduce 等 collective 才对得上。
- 调度器只在 EngineCore 里有一份，决定只做一次，所以是一发多收。
- 回程反过来，只收一份：只有 `output_rank`（最后一个 PP stage 的第一个 TP rank）回传 `ModelRunnerOutput`。最后一个 stage 的各 TP rank 采样结果相同，前面的 stage 没有 logits。

## 五、部署/物理视图：进程怎样落到 GPU 和节点上

进程视图说了有哪几类进程；这张视图把它们落到硬件上：API Server 和 EngineCoreProc 是纯 CPU 进程，执行侧每个模型并行 rank 占一张（或一份）GPU。几个典型配置：

| 部署 | 进程 | 落位 |
|---|---|---|
| 单卡 | 1 × API Server、1 × EngineCoreProc（`UniProcExecutor`，Worker 在 EngineCore 进程内，没有 WorkerProc） | 1 台节点，EngineCoreProc 占 1 张 GPU |
| 单机 TP=4 | 1 × API Server、1 × EngineCoreProc、4 × WorkerProc | 1 台节点，4 个 WorkerProc 各占 1 张 GPU，组内集合通信 |
| 多机 TP×PP | 1 × API Server、1 × EngineCoreProc、world_size 个 WorkerProc | 默认 `MultiprocExecutor` 配合 `--nnodes`，每台节点起自己的 WorkerProc；`--distributed-executor-backend ray` 时用 `RayExecutorV2` / `RayDistributedExecutor`（`RayWorkerProc`），rank 间 TP/PP collective 跨机 |
| DP=2 | 1 × API Server、2 × EngineCoreProc（rank 0 可能再带 `DPCoordinator`）、每 rank 各自一组 WorkerProc | 两个 EngineCore rank 各管自己的 Scheduler 与 Executor，共用一个 API 入口 |
| 多 API Server | `--api-server-count > 1`，N × API Server 子进程 | 只增加前端进程，模型侧 rank 数不变；P/D 分离部署则把 prefill 与 decode 拆到不同 EngineCore 实例、连 KV 传输层 |

Table: 部署方式到进程、GPU、节点的映射

> 单卡时没有 WorkerProc，Worker 直接跑在 EngineCoreProc 里，源码依据（v0.27.1）：`ParallelConfig.__post_init__` 里 `world_size == 1` 时把 `distributed_executor_backend` 置为 `"uni"`（`vllm/config/parallel.py`），`Executor.get_class` 再把 `"uni"` 映射到 `UniProcExecutor`（`vllm/v1/executor/abstract.py`）；它的 `_init_executor` 直接在 EngineCore 进程内 `WorkerWrapperBase.init_worker()`、`init_device()`、`load_model()`（`vllm/v1/executor/uniproc_executor.py`），全程没有 `make_worker_process`。单卡时连 A→B 的边界都可能取消（`InprocClient`），但那是另一层选择。

部署方式怎么改变进程图，对应关系如下：

![vLLM 的多 API Server 与 data parallel 部署变体](/img/in-post/vllm-request-lifecycle-deployment-variants.svg)

- `UniProcExecutor` 在 EngineCore 进程内直接初始化并调用 Worker，不创建 `WorkerProc` 子进程。`InprocClient` 是另一层选择：它把 EngineCore 本身也放在调用方进程里。二者改变的是不同的进程边界（`vllm/v1/executor/uniproc_executor.py`、`vllm/v1/engine/core_client.py`）。
- data parallel 会启动多个 EngineCore rank，各自有自己的 Scheduler 和 Executor。在线且配置需要协调时，rank 0 还会启动 `DPCoordinator`；它汇总队列统计、向 API frontends 发布负载信息，并在需要时协调 MoE request wave。它不是每个 token 的执行通道（`vllm/v1/engine/utils.py`、`vllm/v1/engine/coordinator.py`、`vllm/v1/engine/core.py`）。
- `--api-server-count > 1` 会启动多个 API Server 进程，并为它们分配各自的 ZMQ 地址；它增加的是前端进程数，不等于增加模型并行 rank。未显式指定时，参数默认值会根据 DP 配置决定（`vllm/entrypoints/cli/serve.py`、`vllm/entrypoints/openai/cli_args.py`）。
- Ray 改变 Worker 的启动和放置方式。`RayExecutorV2` 用 Ray actor 承载 `RayWorkerProc`，并复用 MessageQueue 控制面与 Worker 间的 collective；旧 `RayDistributedExecutor` 是另一套 Ray executor 实现（`vllm/v1/executor/ray_executor_v2.py`、`vllm/v1/executor/ray_executor.py`）。

把这张表翻成两条规则：**进程数由两类并行度分别决定**——data parallel size 决定 `EngineCoreProc` 的数量（`launch_core_engines` 对每个 dp rank 起一个 EngineCore 进程，`vllm/v1/engine/utils.py`、`vllm/v1/engine/core.py`）；`world_size = TP × PP`（×PCP，context parallel）决定每个 EngineCore 内 `MultiprocExecutor` 拉起的 `WorkerProc` 数（`vllm/v1/executor/multiproc_executor.py` 的断言 `world_size == tp_size * pp_size * pcp_size`）。expert parallel 不新增进程：EP 复用 DP×TP 的已有 rank（MoE 权重按 EP 组切分在已有 Worker 上）。DP rank 之间互不共享 KV state，每 rank 一套 Scheduler/Executor/Worker 组。

## 六、场景视图（+1）

Kruchten 4+1 的“+1”就是场景视图：逻辑、开发、进程、物理四个视图各看一个切面，场景把四个切面串起来——沿时间轴走一遍真实发生的交互，验证前四张图能对上同一件事。这里放两个子场景：先看一条请求的完整生命周期（谁经手、什么顺序），再只看 token 这个对象在四个进程之间怎么变形（数据流）。

### 1. 请求流：一次请求的完整生命周期


```mermaid
%% 图：一次请求的完整生命周期：从 POST /v1/chat/completions 经 AsyncLLM、EngineCore、Scheduler、Executor 到 GPU，再流式返回
sequenceDiagram
    participant C as Client
    participant API as API Server
    participant E as AsyncLLM
    participant EC as EngineCore
    participant S as Scheduler
    participant KV as KVCacheManager
    participant EX as Executor
    participant W as GPU Worker
    participant MR as ModelRunner
    participant GPU as GPU

    C->>API: POST /v1/chat/completions
    API->>API: 参数解析 & 校验
    API->>E: add_request(prompt, params)
    E->>E: InputProcessor: tokenize → EngineCoreRequest
    E->>EC: EngineCoreClient 经 ZMQ 发送（跨进程）
    EC->>S: add_request(Request)

    Note over S: Request.status = WAITING

    rect rgb(230, 245, 255)
        Note over EC,GPU: === 调度循环 step() ===
        S->>S: schedule() — 选择可运行请求
        S->>KV: get_computed_blocks() — 查 Prefix Cache
        KV-->>S: 缓存命中块 + 未命中数
        S->>KV: allocate_slots() — 分配新块
        KV-->>S: KVCacheBlocks
        Note over S: Request.status = RUNNING
        S-->>EC: SchedulerOutput
    end

    EC->>EX: execute_model(SchedulerOutput)
    EX->>W: forward pass

    rect rgb(255, 245, 230)
        Note over W,GPU: === Prefill 阶段 ===
        W->>MR: execute_model(SchedulerOutput)
        MR->>MR: _prepare_inputs(): input_ids / positions / slot_mapping
        MR->>GPU: 模型 Forward (所有 prompt tokens)
        GPU->>GPU: Attention 计算 + KV Cache 写入
        GPU->>GPU: MLP 计算
        GPU-->>MR: logits
        W->>MR: sample_tokens()
        MR->>MR: Sampling → 第 1 个 output token
    end

    MR-->>EC: ModelRunnerOutput
    EC->>S: update_from_output()
    EC-->>E: EngineCoreOutputs
    E-->>API: 第 1 个 token (TTFT)
    API-->>C: SSE: data: {"token": "Hello"}

    loop Decode 循环 (每步 1 token)
        rect rgb(245, 255, 230)
            S->>S: schedule()
            S->>KV: allocate_slots(1 new token)
            EC->>EX: execute_model()
            W->>MR: execute_model() → _prepare_inputs(1 token per request)
            MR->>GPU: Forward (读历史 KV Cache + 计算新 token)
            GPU-->>MR: logits
            MR->>MR: Sampling → next token
            MR-->>EC: ModelRunnerOutput
            EC->>S: update_from_output()
        end
        EC-->>E: EngineCoreOutputs
        E->>E: OutputProcessor: detokenize + stop 检查
        E-->>API: RequestOutput (stream)
        API-->>C: SSE: data: {"token": "..."}
    end

    Note over S: 遇到 stop token / max_tokens
    Note over S: Request.status = FINISHED_STOPPED
    S->>KV: free(request) — 释放所有块
    API-->>C: SSE: data: [DONE]
```

### 2. 数据流：Token 如何穿过整个 Serving 栈

#### 放大：一个 token 的回程（流式输出）

上面的时序图把 EngineCore 和 AsyncLLM 之间画成了同步的请求-应答，实际上它们跨进程、各自有独立的循环。下面把 decode 循环中**一个 token 的回程**放大，重点看两件事：detokenize 发生在哪个进程，以及引擎循环为什么不需要等它。

![一个 token 的回程：ModelRunner 采样 D2H 后 EngineCore 主线程立刻回到 schedule()；序列化在输出线程，detokenize 与 stop 检查在进程 A](/img/in-post/vllm-request-lifecycle-token-return.svg)

这张图解释了第四章的进程边界怎样影响流式输出：进程 B 的主线程把 `EngineCoreOutputs` 放进 `output_queue` 后立刻回到 `schedule()`，序列化由输出线程做，detokenize 和 stop 字符串检查由进程 A 的 `output_handler` 协程做，三者互不等待。代价是 stop **字符串**（而非 stop token id）的判定要晚一步——进程 A 检测到后反向发 abort，GPU 可能已经为这个请求多算了一步。另外，`RequestOutputCollector` 在 `DELTA` 模式下会把积压的输出合并，因此客户端收到的一个 SSE chunk 不一定恰好对应一个 decode 步。

去程把文本变成 token ids 送到 GPU，回程只带 token ids，到 API Server 才变回文本。四条泳道对应四类实体：

![一个 token 穿过 Serving 栈的泳道图：Client、API Server、EngineCoreProc、WorkerProc，标注每一跳传的是文本、token ids 还是调度元数据](/img/in-post/vllm-request-lifecycle-token-dataflow.svg)

注意 Detokenizer 不在 GPU 那一侧，而是在 API Server 进程的 `OutputProcessor` 里：EngineCore 回传的只有 token ids，文本是到这里才拼出来的。上图沿着数据走了一遍，但同一个请求在每一层其实是**不同的对象**——它们定义在不同文件、活在不同进程、寿命也不同。把这些对象排成一张表，就能看出"谁持有请求的真状态、谁只是一次性的快照"：

| 对象 | 定义位置 | 所在进程 | 生存期 | 它是什么 |
|---|---|---|---|---|
| HTTP JSON → `ChatCompletionRequest` | `vllm/entrypoints/openai/chat_completion/protocol.py` | A（API Server） | 一次 HTTP 连接 | 文本 prompt + 采样参数（用户视角） |
| `EngineCoreRequest` | `vllm/v1/engine/__init__.py` | A → B（msgspec 序列化过 ZMQ） | 只传一次 | 扁平的 IPC 载荷：`prompt_token_ids`、`sampling_params`、`arrival_time`… 已 tokenize，无状态 |
| `Request` | `vllm/v1/request.py`（`from_engine_core_request`） | B（Scheduler 内部） | 从入队到 `free()` | **请求状态的唯一权威副本**：`status` 状态机、`num_computed_tokens`、`output_token_ids`、`block_hashes` |
| `NewRequestData` / `CachedRequestData` | `vllm/v1/core/sched/output.py` | B → C（共享内存广播） | 一个 step | `SchedulerOutput` 里的两种"订单"：首次调度的请求带全量 `prompt_token_ids` + `block_ids`；已在 batch 里的只带增量 `new_token_ids` + `new_block_ids` |
| `SchedulerOutput` | `vllm/v1/core/sched/output.py` | B → C | 一个 step | 上面两者 + `num_scheduled_tokens{req_id: n}` + `finished_req_ids` 等：这一轮"谁跑、跑多少、块在哪" |
| `CachedRequestState` + `InputBatch` 行 | `vllm/v1/worker/gpu_input_batch.py` | C（Worker 常驻） | 从首次调度到 `finished_req_ids` | Worker 侧对 `Request` 的**镜像**，靠增量订单保持同步；`InputBatch` 是按行排列的持久 batch（`token_ids_cpu`、`block_table`） |
| `ModelRunnerOutput` | `vllm/v1/outputs.py` | C → B（`worker_response_mq`） | 一个 step | `req_ids` + `sampled_token_ids: list[list[int]]` + logprobs：只有采样结果，不含任何状态 |
| `EngineCoreOutput(s)` | `vllm/v1/engine/__init__.py` | B → A（ZMQ） | 一个 step | 每个请求的 `new_token_ids` + `finish_reason`；仍然只有 token ids |
| `RequestOutput` | `vllm/outputs.py` | A（`OutputProcessor` 产出） | 一个 SSE chunk | 第一次出现**文本**：detokenize 后的增量 `text` + `token_ids` |

Table: 一个请求在各层的对象

表里有两条规律。第一，**状态只在 B 和 C 各有一份**：`Request` 是权威，`CachedRequestState` 是靠每步增量同步的镜像；所有跨进程的载荷（`EngineCoreRequest`、`SchedulerOutput`、`ModelRunnerOutput`、`EngineCoreOutputs`）都是无状态的一次性消息，读完即弃，也因此可以随意序列化、走任何 IPC 通道。第二，**文本只在进程 A 出现**：从 `EngineCoreRequest` 到 `EngineCoreOutputs` 全程都是 token ids，进程 B、C 完全不需要 tokenizer。

## 七、本文小结

读 trace 时，先分清两种流量：请求、token ids 和 `SchedulerOutput` 属于控制流；TP/PP 的激活张量在 Worker 间走 collective，不经 EngineCore 转发。

把它和第一篇的四问对齐，就得到全系列的骨架：

| 四问 | 承担模块 |
|------|---------|
| 一、这一轮谁执行、执行多少 | `Scheduler`（+ `EngineCore` 驱动） |
| 二、状态放哪、怎么复用 | `KVCacheManager` / `BlockPool` |
| 三、怎么算得更快 | `ModelRunner` / Attention Backend / Kernel |
| 四、怎么扩出去 | `Executor` / `Worker` / 集合通信 |

Table: 四问与承担模块


<details markdown="1">
<summary><b>📂 本章源码导航</b></summary>

**入口与引擎循环**

| 想看什么 | 从哪开始 |
|---|---|
| HTTP 入口、OpenAI 兼容接口 | `vllm/entrypoints/openai/api_server.py`（路由）、`vllm/entrypoints/openai/chat_completion/`（chat 接口实现） |
| 异步请求生命周期、流式响应 | `vllm/v1/engine/async_llm.py` |
| tokenize 与请求构造 | `vllm/v1/engine/input_processor.py` → `InputProcessor` |
| 与 EngineCore 进程通信 | `vllm/v1/engine/core_client.py` → `EngineCoreClient` / `AsyncMPClient` |
| detokenize 与输出整理 | `vllm/v1/engine/output_processor.py` → `OutputProcessor` |
| **推理主循环（建议从这里入手）** | `vllm/v1/engine/core.py` → `EngineCore.step()` |
| 调度器给执行层的"订单" | `vllm/v1/core/sched/output.py` → `SchedulerOutput` / `NewRequestData` / `CachedRequestData` |
| 执行抽象与各种部署形态 | `vllm/v1/executor/abstract.py` |
| 一轮 batch 在 GPU 上怎么跑 | `vllm/v1/worker/gpu_model_runner.py` → `GPUModelRunner.execute_model()` / `sample_tokens()` |
| 进程启动、ZMQ 握手与 DP 协调 | `vllm/v1/engine/utils.py`、`vllm/v1/utils.py`、`vllm/v1/engine/coordinator.py` |

Table: 入口与引擎循环的源码导航

</details>

## 八、自测

1. vLLM 里 `AsyncLLM` 与 `EngineCore` 为什么分成两个进程？中间传什么？

   <details markdown="1"><summary>答案</summary>

   把 tokenize / detokenize、HTTP、流式响应这些 CPU 密集且抖动大的工作从引擎循环里摘出去，让 `EngineCore.step()` 稳定地驱动 GPU；中间经 ZMQ 传 `EngineCoreRequest`（token id + 采样参数）与 `EngineCoreOutputs`（新 token id），不传 tensor。

   </details>

2. `SchedulerOutput` 里有什么、没有什么？为什么 ModelRunner 能只凭它构造这一步的输入？

   <details markdown="1"><summary>答案</summary>

   有：每个被调度请求的 id、这一轮的新 token 数、已计算 token 数、block table、采样参数变化、被抢占 / 完成的请求；没有：任何 tensor。ModelRunner 维护持久的 `InputBatch`，凭这些元数据更新位置与 slot mapping、从 KV 块里读，不需要 Scheduler 传数据。

   </details>

3. 一个请求的状态机有哪几个主要状态？从 WAITING 到 RUNNING 要过哪些检查？

   <details markdown="1"><summary>答案</summary>

   WAITING → RUNNING → FINISHED（或被抢占回 WAITING / PREEMPTED；结构化输出有 WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR）；准入要 KV 块够（`KVCacheManager.allocate_slots`）、token budget 够、`max_num_seqs` 未满、（有 LoRA 时）`max_loras` 未满。

   </details>

4. TP=8 时 Executor 把一步分发给 8 个 Worker，谁做采样？logits 在哪？

   <details markdown="1"><summary>答案</summary>

   8 个 Worker 各算自己那 1/8 的权重，激活经 all-reduce 汇合；lm_head 的输出（logits）经 all-gather 后在每个 rank 都完整，每个 rank 都各自采样（同一个随机种子保证结果一致），只有 `output_rank`（rank 0）把结果回传给 EngineCore，其他 rank 的结果不回传。

   </details>

5. 前端 detokenize 为什么放在 `AsyncLLM` 而不是 EngineCore？流式输出时“一个 token 一次返回”有什么坑？

   <details markdown="1"><summary>答案</summary>

   detokenize 是纯 CPU 且要维护每请求的字符串状态，放引擎里会拖慢 step；坑：BPE（Byte Pair Encoding，字节对编码）的 token 边界与 UTF-8 字符边界不对齐，一个多字节字符可能跨两个 token，`IncrementalDetokenizer` 要缓存到能安全输出的完整字符再发。

   </details>

6. 默认多进程路径里，`EngineCoreProc` 与 `WorkerProc` 各自做什么？`SchedulerOutput` 和 `ModelRunnerOutput` 分别通过哪种通道传？

   <details markdown="1"><summary>答案</summary>

   `EngineCoreProc` 持有 Scheduler 和请求状态，驱动调度、执行、更新；`WorkerProc` 持有本 rank 的 Worker、GPUModelRunner、权重分片和 GPU KV cache。单机时 EngineCore 通过共享内存 `MessageQueue` 广播 `SchedulerOutput`，`output_rank` 通过 response queue 回传 `ModelRunnerOutput`。

   </details>

7. TP/PP、DP、`--api-server-count > 1` 分别改变哪一部分？RayExecutor 又改变什么？

   <details markdown="1"><summary>答案</summary>

   TP/PP 增加 Worker rank，并让 Worker 之间做 collective；DP 增加独立的 EngineCore rank，在线配置需要时由 DPCoordinator 汇总负载或协调 wave；`--api-server-count` 增加 API Server 进程。RayExecutor 改变 Worker 的启动和放置方式；`RayExecutorV2` 仍复用 MessageQueue 控制面与 Worker 间 collective。

   </details>

[^q0]: 链路是固定的：**入口**（`api_server` 的 OpenAI 兼容路由）→ `AsyncLLM` → `InputProcessor`（tokenize、构造 `EngineCoreRequest`）→ 经 `EngineCoreClient` 跨进程送进 **EngineCore**。EngineCore 的 `step()` 是驱动循环：**Scheduler 决策**——这一轮哪些请求跑、每个推进多少 token，产出 `SchedulerOutput`；**Executor 分发**到一个或多个 Worker；**ModelRunner 执行**——准备 `InputBatch`、调 attention backend 与模型前向、采样，产出 `ModelRunnerOutput`；EngineCore 把结果交回 Scheduler 更新请求状态，并把 `EngineCoreOutputs` 送回前端 detokenize、流式返回。**传递的是什么**：Scheduler 与 ModelRunner 之间传的不是 tensor 而是元数据——每个请求这一轮的 token 数、block table、slot mapping、采样参数；Worker 之间传的是激活（TP 的 all-reduce、PP 的 P2P）；前后端之间传的是 token id。详见[第二](#二逻辑视图静态系统拓扑自顶向下)至[第六章](#六场景视图1)（数据流在 6.2）。
