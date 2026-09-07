---
layout: post
title: "大规模训练工程（03）：三个框架——Megatron-LM、DeepSpeed 与 torchtitan 的架构对比与源码导读"
subtitle: "Megatron-LM, DeepSpeed and torchtitan: Architecture and Source Guide"
tags: [Megatron, DeepSpeed, torchtitan, Distributed Training, AI, AI-Infra]
catalog: true
---

> 本文是[《大规模训练工程：从并行策略到容错恢复》](/large-scale-training-from-parallelism-to-fault-tolerance.html)系列的第 3 篇（共八篇）。上一篇：[并行策略全景：每种并行切的是哪种状态](/parallelism-strategies-which-state-to-shard.html)；下一篇：[千卡配置实战：并行搭配、micro-batch、激活重计算与 MFU 调优](/thousand-gpu-configuration-and-mfu-tuning.html)。

> **更新 @2026-09-06**：本文 torchtitan 部分基于 v0.3.0 刷新；其余源码引用仍以 PyTorch 2.13.0 / Megatron Core 0.18.0 / DeepSpeed 0.19.2 为准。

上一篇结束在一个数字上：Llama 3 405B 用 TP 8 / PP 16 / DP 128 训练时，每张卡常驻的参数、梯度、优化器状态只有 13 GB，而在途激活可以到 73 GB。这个数字是从公式里算出来的，公式假设"bf16 参数在每张 TP×PP 分片上完整、fp32 主参数被 ZeRO-1 切成 128 份"。但公式不会告诉你：那份 bf16 参数在显存里是一个 `nn.Parameter` 还是一段大 buffer 的切片？fp32 主参数是谁在什么时候分配的？梯度算完之后经过几次拷贝才到达优化器？这些问题的答案决定了显存表里每一行的实际值，也决定了 profiler 时间线上每一段通信出现的位置。

三个框架——Megatron-LM（Megatron Core）、DeepSpeed、torchtitan——实现的是上一篇同一张表，但它们对这些问题的回答完全不同。Megatron 把一个 DP 副本内的所有 bf16 参数拍进一个连续 buffer，梯度也拍进另一个 buffer，优化器只看自己那 $$1/N_d$$ 段；DeepSpeed 让每个参数在不用的时候只剩 $$1/N_d$$ 的碎片，用到时临时拼回来、用完立即打碎；torchtitan 根本不保留一份常驻的 bf16 参数——常驻的是 fp32 分片，bf16 只在一层前向、反向的几十毫秒里存在。三种回答对应三种显存曲线、三种通信时间线、三种 checkpoint 格式，也对应三个框架各自最擅长的场景。

读框架源码最容易迷失在目录里：Megatron Core 0.18.0 的 `megatron/core/` 下有二十多个子目录、几十个顶层文件，DeepSpeed 的 `runtime/zero/stage3.py` 一个文件三千行，torchtitan v0.3.0 又刚刚把整套配置系统从 TOML 换成了 Python。本篇不打算逐个目录介绍，而是用一条线把三个框架串起来读：

> **一个 bf16 参数在这三个框架里各自存在哪里、什么时候被 all-gather、什么时候被释放、它的 fp32 主副本在哪张卡上？把这条链追清楚，三个框架的架构差异就全部显现了。**

沿着这条线，每个框架要回答同样四件事：**进程组怎么组织**（谁和谁通信）、**状态放在哪**（参数、梯度、优化器状态的实际容器）、**训练循环怎么写**（一个 step 里这些容器按什么顺序被读写）、**取舍是什么**。最后把三份答案并排：同一个"前向前把分片参数 all-gather 回来"的动作在 DeepSpeed Stage 3、FSDP1、FSDP2 里的三种实现，同一个 1F1B 调度在 Megatron 与 `torch.distributed.pipelining` 里的两种写法。

依照系列惯例，通信原语只用语义和通信量；所有源码引用给出路径与函数名，Megatron Core 以 core_v0.18.0、DeepSpeed 以 v0.19.2、PyTorch 以 v2.13.0 为准，torchtitan 以 v0.3.0 为准（见文首更新声明）。


## 一、总览

### 1. 符号与前两篇的结论

本篇沿用第一篇的记账符号，用到的复述如下：

```text
N                参数量（个数）；N_local = N / (N_t · N_p) 是一张卡在 TP × PP 切分后持有的份额
N_d N_t N_p N_c N_e   数据 / 张量 / 流水 / 上下文 / 专家并行度
s  b  h  L       序列长、micro-batch 大小、隐藏维、层数；m 为每个 DP 副本每 step 的 micro-batch 数
```

第二篇的两个结论是本篇的起点。第一，混合精度 + Adam 下每参数常驻 16 字节（bf16 参数 2 + bf16 梯度 2 + fp32 主参数 4 + fp32 一阶矩 4 + 二阶矩 4），若梯度以 fp32 累加则 18 字节；ZeRO-1/2/3 分别把后 12、14、16 字节除以 $$N_d$$。第二，ZeRO-1/2 的通信量与 DP 相同（$$2N$$：reduce-scatter + all-gather），ZeRO-3 涨到 $$3N$$（前向 all-gather、反向 all-gather、reduce-scatter），前向后不释放参数则回到 $$2N$$。本篇要看的就是这几个字节数和这几次通信在代码里的落点。

### 2. 三种取向：从哪种状态出发切

三个框架的架构差异可以压缩成一句话：**它们各自从四种状态中的哪一种出发来组织代码**。

```text
                Megatron-LM                      DeepSpeed                        torchtitan
出发点          模型结构：TP 切矩阵、PP 切层         优化器状态：ZeRO 逐级切 O → G → P     参数的表示：DTensor 的 placement
对模型代码      模型必须用它的层写                  包裹任意 nn.Module，零侵入           模型用它的 Module 协议写，但层是普通 PyTorch
状态容器        连续 buffer（参数一段、梯度一段）    每参数一个 ds_tensor 碎片 + 扁平分区    每参数一个 Shard(0) 的 DTensor
默认 DP 策略    DDP + 分布式优化器（ZeRO-1）         ZeRO-1/2/3 由 JSON 选                FSDP2 fully_shard（ZeRO-3）
bf16 参数常驻？ 是，每张 DP 卡一份完整 N_local        Stage 3 否，只剩 1/N_d 碎片          否，常驻的是 fp32 分片
fp32 主参数     优化器内部的 1/N_d 切片拷贝           扁平 fp32 分区（可放 CPU）            就是那个 fp32 分片本身，不另存
配置方式        argparse（几百个 --flag）             JSON                                 Python（Trainer.Config）
```

表里最后三行是本篇主线的缩影。Megatron 的 bf16 参数是一等公民，fp32 主参数是优化器私有的副本；DeepSpeed 的 bf16 参数在 Stage 3 下是"临时物"，fp32 分区才是持久的；torchtitan 把两者合一——fp32 分片既是模型参数也是主参数，bf16 只是通信和计算时的临时投影。理解了这一行，三个框架的显存表、通信时间线、checkpoint 格式的差别都能推出来。

### 3. 共同底座：PyTorch 分布式

三个框架都建在 `torch.distributed` 上，但用到的层次不同：

```text
torch.distributed 层次                                            Megatron   DeepSpeed   torchtitan
────────────────────────────────────────────────────────────────  ─────────  ──────────  ──────────
ProcessGroup（new_group / all_reduce / all_gather_into_tensor …）    ✓ 直接用   ✓ 经 deepspeed.comm 封装   经 DeviceMesh
DeviceMesh（torch/distributed/device_mesh.py）                       FSDP 路径  autoTP 路径              ✓ 全部并行维度
DTensor（torch/distributed/tensor/）与 placement                    FSDP 路径  —                        ✓ 参数与激活
fully_shard / FSDP2（torch/distributed/fsdp/_fully_shard/）         可选后端   —                        ✓ 默认 DP
pipelining（torch/distributed/pipelining/{stage,schedules}.py）      —（自写）  —（自写）                ✓
```

Megatron 与 DeepSpeed 诞生于 DeviceMesh 和 DTensor 之前，各自手写了进程组管理、集合通信调用和流水线调度；torchtitan 诞生于之后，是这些原生 API 的参考用法。所以读 torchtitan 有一个副产品：它告诉你 PyTorch 官方认为多维并行"应该"怎么组合。本篇第五章会看到，这条路在 v0.3.0 又往前走了一步——参数的切分方式已经不再由 `ColwiseParallel` 这类 API 逐个 module 指定，而是由一份声明式的 `ShardingConfig` 描述。

### 4. 版本与目录地图

本篇涉及的目录如下，每个框架只列本篇会读到的部分（完整目录见各章）。

```text
Megatron Core 0.18.0  megatron/core/
  parallel_state.py                 进程组：initialize_model_parallel、RankGenerator、get_*_group
  process_groups_config.py          ProcessGroupCollection：把进程组打包传给各模块的新式接口
  tensor_parallel/{layers,mappings}.py   TP 的层与通信
  pipeline_parallel/{schedules,p2p_communication}.py   PP 调度与点对点
  distributed/{distributed_data_parallel,param_and_grad_buffer,finalize_model_grads}.py   DDP 与 buffer
  distributed/fsdp/                 Megatron-FSDP
  optimizer/{distrib_optimizer,optimizer}.py   分布式优化器
                      megatron/training/{training,arguments}.py   训练循环与参数

DeepSpeed 0.19.2      deepspeed/
  __init__.py                       deepspeed.initialize()
  runtime/engine.py                 DeepSpeedEngine
  runtime/zero/{stage_1_and_2,stage3,partition_parameters,parameter_offload,partitioned_param_coordinator}.py
  runtime/zero/config.py            zero_optimization 的全部键
  runtime/pipe/{engine,schedule,module,topology,p2p}.py   流水线引擎
  utils/groups.py                   进程组
  comm/comm.py                      对 torch.distributed 的封装

torchtitan v0.3.0     torchtitan/
  train.py, trainer.py              入口与 Trainer
  config/{configs,manager}.py       ParallelismConfig / TrainingConfig 等；Trainer.Config 是根
  distributed/{parallel_dims,fsdp,tensor_parallel,pipeline_parallel,utils}.py, distributed/context_parallel/
  protocols/{module,sharding}.py    Module 协议与 ShardingConfig（TP 的声明式描述）
  models/common/decoder_sharding.py 一个 decoder 的 TP/SP 切分方案
  models/llama3/{__init__,model,parallelize,config_registry}.py
  components/{checkpoint,dataloader,metrics}.py, components/{checkpointer,optimizer}/
  tools/profiler.py

PyTorch 2.13.0        torch/distributed/
  device_mesh.py                    DeviceMesh
  tensor/_api.py, tensor/parallel/  DTensor、distribute_tensor、ColwiseParallel 等
  fsdp/_fully_shard/                FSDP2；fsdp/_flat_param.py、_runtime_utils.py 是 FSDP1
  pipelining/{stage,schedules,_backward}.py
```

一处与 torchtitan 旧版用法不同的地方要先说明：v0.2.x 的 torchtitan 用 TOML 文件配置一次运行，v0.3.0 已经不是——`torchtitan/config/README.md` 明确说一次运行由一个返回 `Trainer.Config` 的 Python 函数描述，通过 `--module` 与 `--config` 选择，`--section.option` 命令行参数只为兼容保留。本篇第五章与练手项目都按 v0.3.0 的写法。

### 5. 本文的章节安排

```text
第二章  进程组          Megatron parallel_state 的 RankGenerator；DeepSpeed groups.py 的 mpu 委托；torchtitan ParallelDims 的 mesh 拆分；对照表
第三章  Megatron-LM     TP 层与 mappings；DDP 的 buffer 与 bucket；分布式优化器；bf16 参数的一生；训练循环；Megatron-FSDP
第四章  DeepSpeed       DeepSpeedEngine；Stage 1/2 的扁平分区；Stage 3 的 ds_tensor 与 hook；bf16 参数的一生；训练循环；pipe/ 与 JSON
第五章  torchtitan      Trainer.__init__ 的装配顺序；ShardingConfig 描述的 TP；fully_shard 的应用；PP 与 CP；bf16 参数的一生；train_step
第六章  对照阅读        三条时序并排；前向前 all-gather 的三种实现（Stage 3 / FSDP1 / FSDP2）；1F1B 的两种写法；取舍表
第七章  小结            要点、源码位置、train-ledger 的 runs/ 与 probe_memory.py
```


## 二、进程组：谁和谁通信

并行策略落到代码里的第一件事是进程组：每张卡属于哪一个 TP 组、哪一个 DP 组、哪一个 PP 组。三个框架用三种方式回答，但产出物相同——一组 `ProcessGroup`，每个集合通信调用时指定其一。

### 1. Megatron：parallel_state 的 RankGenerator

`megatron/core/parallel_state.py` 是一个由模块级全局变量组成的注册表：`_TENSOR_MODEL_PARALLEL_GROUP`、`_DATA_PARALLEL_GROUP`、`_DATA_PARALLEL_GROUP_WITH_CP` 等几十个变量，各配一个 `get_*_group()` 读取函数。填这些变量的是 `initialize_model_parallel()`，它的签名里有 TP、PP、VP、CP、EP 各维度的大小，以及决定 rank 排布的 `order`，默认 `"tp-cp-ep-dp-pp"`。

排布由 `RankGenerator` 类完成。它接收各维度大小与 `order` 字符串，`get_ranks(token)` 对一个形如 `"tp"`、`"dp-cp"`、`"tp-ep-pp"` 的 token 调用 `generate_masked_orthogonal_rank_groups()`：把世界大小按 `order` 拆成一个多维网格，token 里出现的维度是"变化的"、其余维度是"固定的"，枚举固定维度的每一种取值，就得到一个进程组的 rank 列表。所以 `"tp-cp-ep-dp-pp"` 的含义是 TP 变化最快（相邻 rank、同节点）、PP 变化最慢——第二篇解释过这来自"TP 不可重叠必须 NVLink、PP 通信最少可放最远"。

`initialize_model_parallel()` 里有两个 `RankGenerator`：`decoder_rank_generator`（tp/cp/dp/pp，ep 固定为 1）和 `expert_decoder_rank_generator`（tp/ep/dp/pp，cp 固定为 1）。这是因为 EP 与 CP 在 Megatron 里被视为互斥的维度（`RankGenerator.__init__` 有 `ep == 1 or cp == 1` 的断言）：专家层的 DP 组 `expt-dp` 大小是 $$N_d N_c / N_e$$，与非专家层的 `dp-cp` 组不同。两套生成器分别为 dense 部分和 MoE 部分建组，而 PP 组必须一致（函数里有对应断言）。

建组的顺序有讲究：`dp-cp` 组最先建，注释说明是为了 SHARP（`sharp_enabled_group="dp"` 时设置 `NCCL_COLLNET_ENABLE`，只有最先创建的通信器能用交换机内归约）。每个 NCCL 组还有一个 gloo 孪生（`create_gloo_process_groups=True` 时），供 CPU 侧的集合操作（如分布式优化器保存 checkpoint 时的 gather）使用。`num_distributed_optimizer_instances > 1` 时再切出 `intra_distributed_optimizer_instance` 与 `inter_distributed_optimizer_instance` 两层——这是 Megatron 版的 HSDP：优化器状态只在 intra 组内分片，inter 组之间做 all-reduce。

0.18.0 里还有一条新路径：`process_groups_config.py` 的 `ProcessGroupCollection` 把这些进程组打包成一个对象，`DistributedDataParallel`、`_ParamAndGradBuffer`、调度函数都接受可选的 `pg_collection` 参数，不传则退回 `parallel_state` 的全局变量；`hyper_comm_grid.py` 的 `HyperCommGrid` 则是一个不依赖全局状态的多维网格实现。这是在往"进程组是显式传递的对象而不是全局单例"的方向走——也就是 DeviceMesh 的方向。

### 2. DeepSpeed：groups.py 的委托与自建

`deepspeed/utils/groups.py` 的设计是**委托优先**：模块级变量 `mpu` 若被设置（用户通过 `deepspeed.initialize(mpu=...)` 传入一个 Megatron 风格的对象，需提供 `get_data_parallel_group()`、`get_model_parallel_group()` 等方法），所有 `_get_data_parallel_group()`、`_get_model_parallel_world_size()` 都转发给它；没有 `mpu` 时，DP 组就是 `_clone_world_group()`——整个世界的一个克隆。这反映了 DeepSpeed 的定位：它自己只负责 DP 系（ZeRO）的那一维，模型并行的进程组由外部框架（历史上是 Megatron-DeepSpeed）提供。

自建的组有三类。`_create_expert_and_data_parallel()` / `_create_expert_data_and_model_parallel()` 为 MoE 建 `ep` 与 `expt-dp` 组，按 `ep_size` 命名存进字典（`_get_expert_parallel_group(group_name)`）；`_create_zero_param_parallel_group()` 为 ZeRO++ 的 hpZ（`zero_hpz_partition_size`）建节点内的"二级参数分区组"，Stage 3 反向时的参数 all-gather 走它（`ds_secondary_tensor`）；`_init_tp_mesh_device()` 为 autoTP（`tensor_parallel` 配置）调用 `dist.initialize_mesh_device()`（`deepspeed/comm/comm.py`）建一个 `(data_parallel, tensor_parallel)` 的二维 DeviceMesh——这是 DeepSpeed 里少数用到 DeviceMesh 的地方。

流水线的进程组另有一套：`deepspeed/runtime/pipe/topology.py` 的 `ProcessTopology` 用一组 `axes` 与 `dims` 描述网格，`PipeModelDataParallelTopology(num_pp, num_mp, num_dp)` 是三维实例，`PipelineParallelGrid` 从中建出 PP、DP、MP 各组，`PipelineEngine` 用它而不是 `groups.py`。所以 DeepSpeed 的进程组事实上有三个来源：外部 `mpu`、`groups.py`、`pipe/topology.py`，`DeepSpeedEngine._configure_distributed_model()` 里 `self.data_parallel_group`、`self.seq_data_parallel_group`、`self.expert_data_parallel_group` 各自从不同来源取。

### 3. torchtitan：ParallelDims 与 DeviceMesh

`torchtitan/distributed/parallel_dims.py` 的 `ParallelDims` 是一个 dataclass：`dp_replicate`、`dp_shard`、`cp`、`tp`、`pp`、`ep`、`world_size`，由 `ParallelDims.from_config()` 从 `ParallelismConfig` 读出（`data_parallel_shard_degree` 为 -1 时自动补成 `world_size / (dp_replicate · cp · tp · pp)`）。`_validate()` 检查乘积等于 world size。

`build_mesh()` 只做一件事：先 `init_device_mesh(device_type, (world_size,), mesh_dim_names=("world",))` 建一个一维的世界 mesh，再用 `DeviceMesh._unflatten()` 把它按不同的维度组合"展开"成几张视图：

```text
dataloading_mesh   ("pp", "batch", "cp", "tp")                       batch = dp_replicate × dp_shard，数据加载用它决定读哪一份
loss_mesh          dataloading_mesh["batch", "cp"]._flatten()         loss 归约：所有切数据的维度
dense_mesh         ("pp", "dp_replicate", "dp_shard", "cp", "tp")     参数分片用；fully_shard 从中挑 dp_shard（与 cp）为 shard 维
sparse_mesh        ("pp", "dp_replicate", "efsdp", "ep")              MoE 专家用；efsdp = dp_shard × cp × tp / ep
```

大小为 1 的维度用 `backend_override[name] = "fake"`，不真正创建 NCCL 通信器（`_mesh_exist()` 决定哪些维即使为 1 也要保留，例如 `dp_shard`，因为 `fully_shard` 需要它来安装 `MixedPrecisionPolicy`）。之后 `get_mesh("tp")`、`get_optional_mesh("pp")`、`get_mesh(["dp_replicate", "dp_shard"])` 按名取子 mesh，各并行模块拿到 mesh 后自己 `mesh.get_group()`。

与 Megatron 对比，这里没有任何全局变量、没有手写的 rank 枚举：rank 排布由 `_unflatten` 的维度顺序决定（PP 最外、TP 最内，与 Megatron 的 `"tp-cp-ep-dp-pp"` 一致），"哪些卡在同一组"这个问题被 DeviceMesh 的索引语义回答。第二篇提到的 HSDP 在这里就是 `dp_replicate > 1`：`fully_shard` 收到的 `DataParallelMeshDims(shard="dp_shard", replicate="dp_replicate")`（`torch/distributed/fsdp/_fully_shard/_fsdp_api.py`）告诉它在哪一维分片、哪一维复制。

v0.3.0 多了一个 `spmd_backend` 字段（`"partial_dtensor" | "full_dtensor" | "spmd_types"`，默认 `"spmd_types"`），它决定 `build_mesh()` 额外建哪些视图、以及第五章要讲的 TP 走哪条路径。三条路径对进程组的组织没有影响，本篇按默认值读。

### 4. 对照表

```text
                    Megatron Core 0.18.0                 DeepSpeed 0.19.2                     torchtitan v0.3.0
────────────────    ──────────────────────────────────   ──────────────────────────────────   ─────────────────────────────────
数据结构            模块级全局变量 + get_*_group()          模块级 mpu 委托 + 字典                ParallelDims 持有的 DeviceMesh 视图
rank 排布           RankGenerator(order="tp-cp-ep-dp-pp")  外部 mpu 决定；pipe 用 ProcessTopology   _unflatten 的维度顺序 (pp, dp_replicate, dp_shard, cp, tp)
DP 组               dp / dp-cp / expt-dp / intra-inter 两层   world 克隆 或 mpu.get_data_parallel_group()   dp_shard / dp_replicate / loss（batch×cp）
TP 组               tp；expert-tp 可与 dense-tp 不同        mpu 或 autoTP 的 mesh                 tp
PP 组               pp + embedding / position-embedding 组   PipelineParallelGrid                  pp（stage 之间由 PipelineStage 自己 send/recv）
CP 组               cp；梯度在 dp-cp 上归约                  sequence-parallel 组（Ulysses）        cp；fully_shard 把 dp_shard×cp 合成 shard 维
EP 组               ep / tp-ep / tp-ep-pp / expt-dp          ep 与 expt-dp，按 ep_size 命名          ep / efsdp
ZeRO/FSDP 分片组     intra_distributed_optimizer_instance    dp（Stage 3 另有 hpZ 二级组）           dp_shard（+cp）
gloo 孪生            每组一个                                 —                                    —
显式传递             pg_collection 可选                       mpu 参数                              parallel_dims 到处传
```

三者都把 TP 放在最内、PP 放在最外，都把 CP 视为"切数据、复制参数"的维度并入梯度归约组（Megatron 的 `dp-cp`、torchtitan 的 `loss` 与 FSDP 的 `dp_shard×cp`）。差别是历史层次：Megatron 是全局单例正在向对象化过渡，DeepSpeed 把模型并行的进程组交给外部，torchtitan 从一开始就是 mesh。


## 三、Megatron-LM：按模型结构切

### 1. 目录地图

```text
megatron/core/
  parallel_state.py                    进程组（第二章）
  model_parallel_config.py             ModelParallelConfig：各并行度、sequence_parallel、overlap_p2p_comm、*_sync_func 回调
  tensor_parallel/
    layers.py                          ColumnParallelLinear、RowParallelLinear、VocabParallelEmbedding、LinearWithGradAccumulationAndAsyncCommunication
    mappings.py                        _CopyToModelParallelRegion 等 autograd.Function；all_to_all_sp2hp / hp2sp（Ulysses 式）
  pipeline_parallel/
    schedules.py                       get_forward_backward_func、forward_backward_no_pipelining / _without_interleaving / _with_interleaving
    p2p_communication.py               P2PCommunicator：recv_forward、send_forward_recv_backward …
  distributed/
    distributed_data_parallel.py       DistributedDataParallel：hook、start/finish_grad_sync、start_param_sync
    param_and_grad_buffer.py           _ParamAndGradBuffer、_ParamAndGradBucket、_ParamAndGradBucketGroup
    distributed_data_parallel_config.py  DistributedDataParallelConfig：overlap_grad_reduce、overlap_param_gather、bucket_size、grad_reduce_in_fp32 …
    finalize_model_grads.py            finalize_model_grads：embedding / 非 TP 参数的跨组归约
    fsdp/                              Megatron-FSDP（第 7 节）
  optimizer/
    optimizer.py                       MixedPrecisionOptimizer、Float16OptimizerWithFloat16Params、ChainedOptimizer
    distrib_optimizer.py               DistributedOptimizer
  transformer/                         TransformerLayer、attention、mlp、moe/ —— 模型本身
megatron/training/
  training.py                          pretrain、setup_model_and_optimizer、train、train_step、training_log
  arguments.py                         parse_args、validate_args；大量参数由 training/config/ 下的 dataclass 生成
```

### 2. TP 层与 mappings：通信是手写的

第二篇已经讲过 `tensor_parallel/layers.py` 的 `ColumnParallelLinear`（权重按输出维切、`gather_output` 控制是否 all-gather 输出）与 `RowParallelLinear`（权重按输入维切、`input_is_parallel` 表示输入已分布、输出 all-reduce 或 reduce-scatter），以及 `mappings.py` 里每种通信作为一个 `autograd.Function` 出现、前向与反向互为共轭。这里补两点与"参数的一生"有关的细节。

第一，TP 分片的参数是**普通的 `nn.Parameter`**，形状就是切开后的形状（`ColumnParallelLinear.weight` 是 `[out/N_t, in]`），只是多了几个属性：`set_tensor_model_parallel_attributes()` 在参数上标 `tensor_model_parallel=True`、`partition_dim`、`partition_stride`。checkpoint、优化器、梯度裁剪靠这些属性知道哪些参数是切开的（裁剪时只算一次）、哪些是 TP 组内复制的（LayerNorm、bias 等，`param_is_not_tensor_parallel_duplicate()` 判断）。这和 DTensor 用 placement 携带同样信息形成对照。

第二，`LinearWithGradAccumulationAndAsyncCommunication` 这个 `autograd.Function` 的名字点出了 Megatron 的两个优化：**梯度累积融合**（`gradient_accumulation_fusion`，权重梯度直接累加进一个外部 fp32 buffer `weight.main_grad`，而不是先生成 `weight.grad` 再加）和**异步通信**（反向里对输入的梯度 all-reduce 与对权重的梯度 GEMM 重叠）。`main_grad` 这个属性是下一节 DDP 与 buffer 的接口：它是 `_ParamAndGradBuffer` 里一段 fp32 内存的视图。

### 3. DDP 与 ParamAndGradBuffer：两段连续内存

Megatron 的 `distributed/distributed_data_parallel.py` 的 `DistributedDataParallel` 与 PyTorch 自带的 DDP 同名但不是同一个东西。它的构造函数做的事是：

1. 用 `group_params_for_buffers()` 把所有需要梯度的参数按 `(param_dtype, grad_dtype, is_expert_parallel)` 分组。`grad_dtype` 在 `grad_reduce_in_fp32=True` 时是 fp32——`--bf16` 下 `arguments.py` 默认把 `accumulate_allreduce_grads_in_fp32` 置为 True（除非 `--grad-reduce-in-bf16`），所以 bf16 训练的梯度默认是 fp32 累加、fp32 归约：这是第一篇"18 字节"那一档。
2. 每组建一个 `_ParamAndGradBuffer`（`param_and_grad_buffer.py`）：分配一段连续的 `param_data`（参数 dtype）和一段 `grad_data`（梯度 dtype），把组内每个参数的 `.data` 替换成 `param_data` 里对应位置的视图、给每个参数挂一个 `main_grad` 属性指向 `grad_data` 里对应位置的视图。参数在 buffer 里按**反向顺序**排列（后面的层在前面），这样反向传播时梯度从 buffer 头部开始依次就位。
3. 按 `bucket_size`（默认 `max(40000000, 1000000 × dp_size)` 个元素）把 buffer 切成若干 `_ParamAndGradBucket`，再把 bucket 组成 `_ParamAndGradBucketGroup`——通信以 bucket group 为单位发起。用分布式优化器时每个 bucket 会被 pad 到 `dp_size` 的整数倍，使 `shard_buffer()` 能把它均分给各 DP rank。
4. 给每个参数注册反向 hook（`_make_backward_post_hook()`）：hook 里 `param.main_grad.add_(param.grad)` 然后 `param.grad = None`（若梯度累积融合已经直接写进 `main_grad`，这一步跳过），再 `register_grad_ready()`——当一个 bucket group 里所有参数的梯度在本 step 的最后一个 micro-batch 都就位时，`start_grad_sync()` 发起通信：分布式优化器下是 reduce-scatter（`dist_reduce_scatter_func` 到 `shard_buffer(bucket.grad_data, dp_size)` 里本 rank 的那一段），否则是 all-reduce；`overlap_grad_reduce=True` 时异步发起、`finish_grad_sync()` 等待。

前几个 micro-batch 的反向不触发通信：`no_sync()` 上下文（由调度函数通过 `config.no_sync_func` 调用）把 `is_last_microbatch` 置 False，hook 只累加不发送。这就是第二篇说的"梯度 bucket 与反向计算重叠"的实现。

`param_data` 与 `grad_data` 是两段独立的内存。`_ParamAndGradBuffer.__init__` 里有一个例外：参数是 MXFP8 张量且用分布式优化器时，二者共享同一段 `shared_buffer`（fp32 梯度时 `param_data` 是它前半段的 bf16 视图）——优化器步骤后 all-gather 更新的参数时复用已经归约完、不再需要的梯度 buffer。bf16 训练走的是两段独立内存的常规路径。

### 4. DistributedOptimizer：ZeRO-1 的 Megatron 写法

`optimizer/distrib_optimizer.py` 的 `DistributedOptimizer` 继承 `MixedPrecisionOptimizer`，是 `--use-distributed-optimizer` 打开时的优化器。它的分片单位不是参数而是 **buffer 的字节区间**：

- `_build_gbuf_range_map()` 把每个 bucket 的 `[0, numel)` 均分成 `dp_size` 段，本 rank 负责第 `dp_rank` 段。这一段的边界不对齐参数边界——一个参数可能一半在本 rank、一半在邻居。`_build_model_gbuf_param_range_map()` 算出每个参数落在本 rank 段里的那一部分（`Range` 对象）。
- `_build_model_and_main_param_groups()` 据此建五组张量：`model_float16_groups`（原 bf16 参数）、`shard_float16_groups`（bf16 参数在本 rank 段内的视图 `model_param.detach().view(-1)[start:end]`）、`shard_fp32_from_float16_groups`（**fp32 主参数**：`model_param.float().view(-1)[start:end]` 得到的独立 fp32 拷贝，只有本 rank 负责的那一段）、以及 fp32 参数的对应两组。内部优化器（Adam）的 `param_groups` 被替换成 `shard_fp32_from_float16_groups`——它只看到 $$1/N_d$$ 的 fp32 分片，一阶矩、二阶矩自然也只有 $$1/N_d$$。

一个 step 的优化器部分：

```text
_copy_model_grads_to_main_grads()     grad_data 里本 rank 那一段（reduce-scatter 的结果，fp32）→ 主参数分片的 .grad
inner optimizer.step()                 Adam 更新 fp32 主参数分片与 m、v
_copy_main_params_to_model_params()    fp32 主参数分片 → param_data 里本 rank 那一段（转 bf16）
step_with_ready_grads()                → start_param_sync_for_bucket_group_subset() → 每个 bucket group 的 start_param_sync()
                                       all-gather bucket 的 param_data：各 rank 的 1/N_d 段拼成完整 bf16 参数
```

`start_param_sync()`（`_ParamAndGradBucketGroup`）在 `overlap_param_gather=True` 时异步发起：第一个 bucket group 的 all-gather 在 `optimizer.zero_grad()` 里发出，之后每个 bucket group 的 all-gather 在前一个完成时接力发出，而**等待**发生在前向的 pre-hook 里——`DistributedDataParallel._make_forward_pre_hook()` 在一个 module 前向前对它的每个参数调用 `finish_param_sync()`，只等这个参数所在 bucket 的 all-gather。于是下一 step 的前向可以边算边收：算第一层时后面几层的参数还在路上。

这里 fp32 主参数与 bf16 参数是**两份内存**：主参数分片是优化器私有的 fp32 拷贝，占 $$4N_{\text{local}}/N_d$$；bf16 参数是模型的、完整的、每张 DP 卡一份，占 $$2N_{\text{local}}$$。

### 5. 一个 bf16 参数的一生：Megatron

把上面三节串起来，以一个 `ColumnParallelLinear.weight` 为例（`--bf16 --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather`，TP 组内它是 $$1/N_t$$ 的分片，下面只看 DP 维）：

```text
时刻                    事件                                                      在哪里
─────────────────────   ──────────────────────────────────────────────────────    ────────────────────────────────────────────
初始化                  weight.data ← param_data[a:b] 视图（bf16，完整）              _ParamAndGradBuffer.__init__
                        weight.main_grad ← grad_data[a:b] 视图（fp32，完整，置零）
                        主参数分片 ← weight.float()[本 rank 段]（fp32，1/N_d）          DistributedOptimizer._build_model_and_main_param_groups
step i 前向 pre-hook    等 weight 所在 bucket 的 all-gather 完成（step i-1 发出的）     DDP._make_forward_pre_hook → finish_param_sync
前向 / 反向 × m 次      GEMM 读 weight（bf16，完整，全程不释放）
                        每次反向：dW 累加进 main_grad（fp32）；param.grad 立即丢弃        LinearWithGradAccumulationAndAsyncCommunication / _make_backward_post_hook
最后一个 micro-batch    bucket 内梯度全部就位 → reduce-scatter grad_data                _ParamAndGradBucketGroup.start_grad_sync（异步）
                        本 rank 得到 grad_data[本 rank 段] 的归约结果
finalize_model_grads    embedding 等跨 PP / 非 TP 参数的额外 all-reduce                 distributed/finalize_model_grads.py
optimizer.step          grad_data[本 rank 段] → 主参数分片.grad                        _copy_model_grads_to_main_grads
                        Adam 更新主参数分片、m、v（全部 fp32，1/N_d）
                        主参数分片 → param_data[本 rank 段]（转 bf16）                   _copy_main_params_to_model_params
                        发起 param_data 的 all-gather（异步，逐 bucket 接力）             step_with_ready_grads → start_param_sync
step i+1 前向 pre-hook  等 all-gather 完成 → weight 又是完整、更新后的 bf16
```

**常驻**：bf16 参数 $$2N_{\text{local}}$$（完整）、fp32 梯度 $$4N_{\text{local}}$$（完整）、fp32 主参数 + m + v $$12N_{\text{local}}/N_d$$。**通信**：每 step 一次 reduce-scatter（fp32 梯度，与反向重叠）+ 一次 all-gather（bf16 参数，与下一 step 前向重叠）——正是第二篇 ZeRO-1 的 $$2N$$。**从不释放**：bf16 参数与 fp32 梯度 buffer 在整个训练期间常驻，这是 Megatron 显存最"稳"（碎片最少）但也最"贵"的地方。

### 6. 训练循环：pretrain → train → train_step

`megatron/training/training.py` 是 Megatron 的训练入口。骨架：

```text
pretrain()
  initialize_megatron()                       解析参数、init_process_group、parallel_state.initialize_model_parallel()
  setup_model_and_optimizer()
    get_model(model_provider_func)            每个 virtual stage 一个 model chunk；wrap_model_chunks_with_ddp() 套 DistributedDataParallel
    get_megatron_optimizer()                  → DistributedOptimizer / Float16OptimizerWithFloat16Params，多个则 ChainedOptimizer
    get_optimizer_param_scheduler()
  build_train_valid_test_data_iterators()
  train()
    config.no_sync_func / grad_sync_func / param_sync_func ← model_chunk.no_sync / start_grad_sync / start_param_sync
    config.finalize_model_grads_func ← finalize_model_grads
    forward_backward_func = get_forward_backward_func()      按 PP、VP 大小选三个调度函数之一
    while iteration < train_iters:
      train_step()
        model_chunk.zero_grad_buffer(); optimizer.zero_grad()        清 grad_data；overlap_param_gather 时在此发出第一个 all-gather
        forward_backward_func(forward_step_func, data_iterator, model, num_microbatches, seq_length, micro_batch_size, ...)
        optimizer.step()                                             → update_successful, grad_norm, num_zeros_in_grad
        opt_param_scheduler.step(increment=global_batch)
      training_log()                                                 loss、lr、grad_norm、timers、mem-*（log_memory_to_tensorboard）
      post_training_step_callbacks() / checkpoint_and_decide_exit()
```

三点值得注意。第一，`train_step()` 被 `rerun_state_machine.should_run_forward_backward()` 的 while 循环包着——这是第六篇要讲的 SDC 检测机制的入口，正常情况下只跑一次。第二，调度函数不知道 DDP 的存在，它通过 `config` 上的四个回调（`no_sync_func`、`grad_sync_func`、`param_sync_func`、`finalize_model_grads_func`）与 DDP 交互，所以 `schedules.py` 能在没有 DDP 的场合（推理、测试）复用。第三，`training_log()` 在 `--log-memory-to-tensorboard` 时直接读 `torch.cuda.memory_stats()` 的 `reserved_bytes.all.current`、`allocated_bytes.all.current`、`allocated_bytes.all.peak` 写 TensorBoard——本篇练手项目的 `probe_memory.py` 读的是同一组键。

### 7. Megatron-FSDP：趋势

`megatron/core/distributed/fsdp/` 是 0.18.0 里 Megatron 自己的 FSDP 实现，`--use-megatron-fsdp` 打开。`fsdp/src/megatron_fsdp/fully_shard.py` 的 `ShardingStrategy` 枚举把四级写得很直白：`NO_SHARD`、`OPTIM`（ZeRO-1）、`OPTIM_GRADS`（ZeRO-2）、`OPTIM_GRADS_PARAMS`（ZeRO-3），由 `--data-parallel-sharding-strategy` 选；`megatron_fsdp.py` 的 `MegatronFSDP` 是模块包装类，`mcore_fsdp_adapter.py` 的 `FullyShardedDataParallel` 把它接进 Megatron 的 `_BaseDataParallel` 接口，与 `DistributedDataParallel` 平级。它保留了 Megatron 的 buffer 思路（`fsdp/src/megatron_fsdp/param_and_grad_buffer.py`）但把参数也切了，并且用 DTensor 表达分片（`uneven_dtensor.py`），checkpoint 格式 `--ckpt-format fsdp_dtensor`。`DistributedOptimizer.step_with_ready_grads()` 里有一个 `use_megatron_fsdp` 分支，调用 `model_chunk.start_param_sync()` 提前 all-gather 主参数。

另一条路径 `--use-torch-fsdp2`（`distributed/torch_fully_sharded_data_parallel.py`）直接用 PyTorch 的 `fully_shard`，但 `arguments.py` 里列了一串不兼容项：不支持 PP、EP、分布式优化器、梯度累积融合。两条路径合起来说明一件事：Megatron 长期"只做 ZeRO-1，其余靠 TP/PP"的立场在松动——当模型能用 FSDP 放下时，FSDP 比 PP 少一个气泡问题、比 TP 少一个 NVLink 约束，Megatron 也需要这个选项。这与 torchtitan 从 FSDP2 出发是同一个方向的两端。


## 四、DeepSpeed：按优化器状态切

### 1. 目录地图

```text
deepspeed/
  __init__.py                          initialize(model, optimizer, config, mpu, ...) → (engine, optimizer, dataloader, lr_scheduler)
  runtime/
    engine.py                          DeepSpeedEngine(Module)：forward / backward / step；_configure_distributed_model / _configure_optimizer / _configure_zero_optimizer
    config.py                          DeepSpeedConfig：解析 JSON
    constants.py                       JSON 键名常量：train_batch_size、train_micro_batch_size_per_gpu、gradient_accumulation_steps、bf16 …
    zero/
      config.py                        DeepSpeedZeroConfig：stage、overlap_comm、contiguous_gradients、reduce_bucket_size、stage3_* …
      stage_1_and_2.py                 DeepSpeedZeroOptimizer（Stage 1 / 2）
      stage3.py                        DeepSpeedZeroOptimizer_Stage3
      partition_parameters.py          Init 上下文；_convert_to_deepspeed_param 给参数挂 ds_* 属性与 all_gather / partition 方法
      parameter_offload.py             DeepSpeedZeRoOffload：注册前向 / 反向 hook
      partitioned_param_coordinator.py PartitionedParameterCoordinator：fetch / release / prefetch
    pipe/
      module.py                        PipelineModule、LayerSpec、TiedLayerSpec
      engine.py                        PipelineEngine(DeepSpeedEngine)：train_batch、_exec_* 指令实现
      schedule.py                      TrainSchedule 等 PipeSchedule；PipeInstruction 子类
      topology.py                      ProcessTopology、PipelineParallelGrid
      p2p.py                           send / recv
  utils/groups.py                      进程组（第二章）
  comm/comm.py                         deepspeed.comm：对 torch.distributed 的封装，可换后端
```

### 2. DeepSpeedEngine：包裹用户模型

DeepSpeed 的入口是 `deepspeed.initialize(model=..., optimizer=..., config=...)`，返回一个 `DeepSpeedEngine`（`runtime/engine.py`）。它继承 `nn.Module`，持有用户的 `self.module`，用户之后调用的是 `engine(batch)`、`engine.backward(loss)`、`engine.step()` 三个方法，模型代码一行不改。构造函数的顺序：

```text
DeepSpeedEngine.__init__
  _configure_distributed_model(model)        把 module 移到设备、转 dtype；确定 data_parallel_group / seq_data_parallel_group / expert_data_parallel_group
                                             非 ZeRO 时广播参数使各 DP rank 一致
  _configure_optimizer(client_optimizer, model_parameters)
    _configure_basic_optimizer()             按 JSON 的 optimizer 段建 Adam / AdamW / FusedAdam …
    → zero_optimization().stage > 0 ?        _configure_zero_optimizer(basic_optimizer)
        stage 1 / 2                          DeepSpeedZeroOptimizer(...)      （stage_1_and_2.py）
        stage 3                              DeepSpeedZeroOptimizer_Stage3(...)（stage3.py）
      : bf16 ?                               _configure_bf16_optimizer()
      : fp16 ?                               _configure_fp16_optimizer()
  _configure_lr_scheduler()
```

所以 DeepSpeed 的"ZeRO 优化器"是一个**优化器包装类**：它接管用户优化器的 `param_groups`，把里面的参数替换成分片；ZeRO 的一切——分片、hook、通信——都在这个包装类里，engine 只在 `backward()` 与 `step()` 里调用它。这与 Megatron（DDP 负责梯度通信、优化器负责参数分片与 all-gather，两个类）和 torchtitan（FSDP 负责一切通信、优化器是普通的）都不同。

### 3. Stage 1 / 2：扁平分区

`stage_1_and_2.py` 的 `DeepSpeedZeroOptimizer` 构造时，对每个 param group：

1. 把组内所有 bf16 参数**拍平拼接**成一个一维张量 `bit16_groups_flat[i]`（`flatten_dense_tensors_aligned()`，pad 到 `dp_size` 的倍数），然后把每个参数的 `.data` 重新指向这个扁平张量里的视图——与 Megatron 的 `param_data` buffer 同一个思路，只是以 param group 而不是 dtype 为单位。
2. `get_data_parallel_partitions()` 把扁平张量均分成 `dp_size` 段，得到 `parallel_partitioned_bit16_groups[i]`。
3. `single_partition_of_fp32_groups[i]` = 本 rank 那一段的 fp32 克隆（`.detach().clone().float()`）——**fp32 主参数**，$$1/N_d$$。用户优化器的 `param_groups[i]['params']` 被替换成这一个 fp32 张量。

梯度路径：`create_gradient_handling_hooks()` 给每个参数注册梯度 hook，`reduce_ready_partitions_and_remove_grads()` 把就位的梯度攒进 IPG（independent partition gradient）bucket（`IPGBucket`，大小 `reduce_bucket_size`，默认 5e8 元素），满了就 `reduce_ipg_grads()` → `average_tensor()`：Stage 2（`partition_grads=True`）时按分区归属把 bucket 里的梯度分别 reduce 到各自的 owner（`reduce_scatter=True` 时用 reduce-scatter），本 rank 只保留自己分区的梯度、其余立即释放；Stage 1 时 all-reduce、所有梯度保留。`overlap_comm=True` 时通信在单独的 stream 上与反向重叠；`contiguous_gradients=True` 时本 rank 分区的梯度拷进一段连续内存以减少碎片。

`step()`：检查溢出（fp16 时）→ 梯度裁剪 → 对每个 group `_optimizer_step(i)`：用户优化器在 fp32 分区上更新 → `bit16_partitions[partition_id].data.copy_(fp32_partition.data)`（fp32 → bf16，只写本 rank 那一段）→ `all_gather_dp_groups()` 把各 rank 的 bf16 段拼成完整的 `bit16_groups_flat`。因为参数的 `.data` 是扁平张量的视图，all-gather 完成的瞬间所有参数就是更新后的值。

与 Megatron 的 ZeRO-1 相比，机制几乎一样（扁平 buffer、按字节区间分片、fp32 分片拷贝、更新后 all-gather），差别在于粒度（DeepSpeed 以 param group 为 buffer、以 bucket 为 reduce 单位；Megatron 以 dtype 为 buffer、以 bucket group 为通信单位）和重叠方式（DeepSpeed 的 all-gather 在 `step()` 里同步完成，不与下一 step 前向重叠；Megatron 可以）。

### 4. Stage 3：ds_tensor 与 hook

Stage 3 的实现分散在四个文件里，按参数的生命周期读：

**分片（`partition_parameters.py`）**。`deepspeed.zero.Init` 是一个上下文管理器（类 `Init`，继承 `InsertPostInitMethodToModuleSubClasses`）：在它里面构造的每个 `nn.Module`，其 `__init__` 结束后会被 `_post_init_method()` 拦截，对每个参数调用 `_convert_to_deepspeed_param()`。这个函数给参数挂上一组 `ds_*` 属性——`ds_status`（`AVAILABLE` / `NOT_AVAILABLE` / `INFLIGHT`）、`ds_shape`、`ds_numel`、`ds_tensor`（**分片**，本 rank 持有的 $$1/N_d$$ 一维碎片）、`ds_process_group`、`ds_persist`（小于 `stage3_param_persistence_threshold` 的参数在前向、反向期间保持完整，只在 step 前后短暂分片）、`ds_secondary_tensor`（hpZ 的节点内二级副本）——以及一组方法：`all_gather()`、`all_gather_coalesced()`、`partition()`、`ds_summary()`。随后 `_partition()` 把参数的完整数据切成 `dp_size` 段、只留本 rank 的一段进 `ds_tensor`，然后 `free_param()` 把 `param.data` 换成一个空张量。**从这一刻起，模型里的 `nn.Parameter` 只是一个壳**，形状对但没有数据。若不用 `zero.Init`，`DeepSpeedZeRoOffload.__init__`（`parameter_offload.py`）里的 `_convert_to_zero_parameters()` 会对已构造好的模型做同样的转换，但那要求模型先能完整放进内存。

**取回与释放（`parameter_offload.py` + `partitioned_param_coordinator.py`）**。`DeepSpeedZeRoOffload.setup_zero_stage3_hooks()` 遍历模型的每个子 module，`_register_deepspeed_module()` 给它注册四个 hook：`_pre_forward_module_hook`、`_post_forward_module_hook`、`_pre_backward_module_hook`、`_post_backward_module_hook`。它们分别调用 `pre_sub_module_forward_function()` → `PartitionedParameterCoordinator.fetch_sub_module(sub_module, forward=True)`（把该 module 直接持有的参数 all-gather 回来，`ds_status` 变 `AVAILABLE`，`param.data` 指向拼好的完整张量）、`post_sub_module_forward_function()` → `release_sub_module()`（对不再被任何活跃 module 使用、且非 `ds_persist` 的参数调用 `__release_param()` → `param.partition()`，完整张量释放）；反向对称。

`PartitionedParameterCoordinator` 里有两个提高效率的机制。**trace**：第一个 step 记录 module 的执行顺序（`record_module()`），之后按这个顺序**预取**——`fetch_sub_module()` 在取回当前 module 的参数时，顺带发起后面几个 module 的 all-gather（总量由 `stage3_prefetch_bucket_size` 控制），`stage3_max_live_parameters` 限制同时活着的完整参数总数、`stage3_max_reuse_distance` 决定"很快又要用"的参数不释放。**coalesced all-gather**：`all_gather_coalesced()`（`partition_parameters.py` 里的 `_all_gather_coalesced()`）把一个 module 的多个参数拼成一次 `all_gather_into_tensor` 调用，返回 `AllGatherCoalescedHandle`，`wait()` 时再拆回各参数。

**梯度（`stage3.py`）**。`create_reduce_and_remove_grad_hooks()` 给每个参数注册 `reduce_partition_and_remove_grads` hook；梯度就位后进 IPG bucket（`IPGBucketZ3`），满了 `__reduce_and_partition_ipg_grads()` → `__avg_scatter_grads()`（reduce-scatter）→ `partition_grads()` 把本 rank 分区的梯度写进 `fp32_partitioned_groups_flat[sub_group].grad` 或 bf16 的梯度分区（取决于 `gradient_accumulation_dtype`），然后 `param.grad = None`。反向结束时该参数本身也已被 `_post_backward_module_hook` 释放。

**fp32 主参数与 step（`stage3.py`）**。参数按 `sub_group_size`（默认 1e9 元素）分成若干 sub-group，每个 sub-group 有一个扁平的 bf16 分区 `fp16_partitioned_groups_flat[i]`（各参数 `ds_tensor` 的拼接）和一个扁平的 fp32 分区 `fp32_partitioned_groups_flat[i]`（`_create_fp32_partitions()`，`offload_optimizer` 时放 CPU pinned memory）。`step()` 对每个 sub-group：`_prepare_fp32_grad_for_sub_group()` → `_optimizer_step(i)`（用户优化器只看到这个 fp32 分区）→ `_reassign_or_swap_out_partitioned_parameters()` 把 fp32 分区拷回 bf16 分区（`fp16_partitioned_groups_flat[i].data.copy_(fp32_partitioned_groups_flat[i].data)`）。`_post_step()` 只 all-gather `persistent_parameters`——其余参数**不在 step 里 all-gather**，等下一 step 前向的 hook 按需取回。这就是第二篇里"ZeRO-3 省掉优化器后那次 all-gather"的代码位置。

### 5. 一个 bf16 参数的一生：DeepSpeed Stage 3

以一个 Transformer 层的 `nn.Linear.weight` 为例（`bf16.enabled=true`，`zero_optimization.stage=3`，不 offload）：

```text
时刻                    事件                                                          在哪里
─────────────────────   ──────────────────────────────────────────────────────────    ─────────────────────────────────────────
zero.Init 内构造        weight 完整构造 → _convert_to_deepspeed_param → _partition       partition_parameters.py Init._post_init_method
                        weight.ds_tensor ← 本 rank 的 1/N_d 一维碎片（bf16）；weight.data ← 空张量
优化器构造              fp16_partitioned_groups_flat[g] ← sub-group 内各 ds_tensor 拼接      stage3.py _create_fp16_partitions_with_defragmentation
                        fp32_partitioned_groups_flat[g] ← 其 fp32 拷贝（1/N_d）              _create_fp32_partitions
step i，该层前向前       _pre_forward_module_hook → fetch_sub_module                      parameter_offload.py / partitioned_param_coordinator.py
                        all_gather_coalesced：N_d 个碎片 → weight.data（bf16，完整）；同时预取后面几层
该层前向                GEMM 读 weight（完整）
该层前向后              _post_forward_module_hook → release_sub_module → weight.partition()    weight.data 又是空张量
该层反向前              _pre_backward_module_hook → fetch_sub_module（再 all-gather 一次）
该层反向                dW 生成 → 梯度 hook → 进 IPG bucket → reduce-scatter                 stage3.py reduce_partition_and_remove_grads / __avg_scatter_grads
                        本 rank 得到 dW[本 rank 段] → 写进 fp32 分区的 .grad；weight.grad = None
该层反向后              _post_backward_module_hook → release → weight.partition()
engine.step             对每个 sub-group：Adam 更新 fp32 分区（1/N_d）                       stage3.py step → _optimizer_step
                        fp32 分区 → fp16_partitioned_groups_flat（即各 ds_tensor，bf16）        _reassign_or_swap_out_partitioned_parameters
                        不 all-gather（除 ds_persist 的小参数）                              _post_step
step i+1，该层前向前     再 fetch_sub_module → all-gather                                  回到上面
```

**常驻**：bf16 碎片 $$2N/N_d$$、fp32 主参数 + m + v $$12N/N_d$$、梯度分区 $$2N/N_d$$（bf16 累加）或 $$4N/N_d$$——合计 $$16N/N_d$$ 或 $$18N/N_d$$。**临时**：正在前向 / 反向的那几层的完整 bf16 参数（预取深度决定有几层）。**通信**：每层前向一次 all-gather、反向一次 all-gather、一次 reduce-scatter，合计 $$3N$$。

与 Megatron 的时序放在一起，最大的差别是 **`param.data` 的身份是流动的**：Megatron 的 `weight.data` 从头到尾是 buffer 里同一段视图；DeepSpeed Stage 3 的 `weight.data` 在一个 step 里被替换四次（空 → 完整 → 空 → 完整 → 空）。这带来 Stage 3 的两个著名代价：任何在 hook 之外访问参数的代码（自定义的权重初始化、`param.data` 的手工操作、把参数当普通张量传给别的函数）都会拿到空张量，需要用 `deepspeed.zero.GatheredParameters` 显式取回；以及调试困难——参数"不在"的时候栈里看不到原因。`stage3_module_granularity_threshold` 与 `leaf_module` 配置（`DeepSpeedZeroLeafModuleConfig`）让 hook 挂在更粗的 module 上，是对这个代价的折衷。

### 6. 训练循环：forward / backward / step

DeepSpeed 没有训练循环——用户写循环，engine 提供三个方法：

```text
for step in ...:
  for micro in range(gradient_accumulation_steps):
    loss = engine(batch)                     DeepSpeedEngine.forward：计时、flops profiler、Stage 3 的 root hook 起点
    engine.backward(loss)                    loss / gradient_accumulation_steps；ZeRO 时 optimizer.scale_if_loss；loss.backward()
                                             _backward_epilogue：ZeRO 时 optimizer.backward_epilogue()（清 IPG bucket、结束 reduce）
                                                                非 ZeRO 时 allreduce_gradients()（在梯度累积边界）
    engine.step()                            is_gradient_accumulation_boundary() 才真正 _take_model_step：
                                               optimizer.step()（ZeRO 的 step，含裁剪、更新、all-gather）
                                               lr_scheduler.step(); zero_grad
                                             global_steps += 1；micro_steps += 1
```

`gradient_accumulation_steps` 由 JSON 的 `train_batch_size / (train_micro_batch_size_per_gpu × dp_size)` 推出，engine 用 `micro_steps` 计数判断边界——用户不需要自己写 `no_sync`。这个设计让"零侵入"成立：现有的单卡训练脚本只要把 `model` 换成 `engine`、`loss.backward()` 换成 `engine.backward(loss)`、`optimizer.step()` 换成 `engine.step()` 就能用 ZeRO-3。代价是控制权在 engine 手里：什么时候 all-gather、什么时候 reduce、什么时候释放，都由 JSON 里的阈值决定，用户能调的是数字而不是顺序。

### 7. pipe/ 引擎与 JSON 配置的代价

流水线在 DeepSpeed 里是另一个 engine：`runtime/pipe/module.py` 的 `PipelineModule` 要求用户把模型写成一个 `LayerSpec` 列表（每层的类与构造参数），`_partition_layers()` 按 `partition_method`（`"uniform"`、`"parameters"`、`"type:regex"`）把层分给各 stage；`runtime/pipe/engine.py` 的 `PipelineEngine` 继承 `DeepSpeedEngine`，`train_batch()` 取代 forward / backward / step 三步：它构造一个 `TrainSchedule`（`runtime/pipe/schedule.py`），`_exec_schedule()` 逐步执行调度产出的指令。

`TrainSchedule.steps()` 是一个生成器，共 `2 × (micro_batches + stages - 1)` 步，每步 yield 一组 `PipeInstruction`——`RecvActivation`、`ForwardPass`、`SendActivation`、`RecvGrad`、`BackwardPass`、`SendGrad`、`LoadMicroBatch`、最后一步加 `ReduceTiedGrads`、`ReduceGrads`、`OptimizerStep`；`_exec_schedule()` 用 `_INSTRUCTION_MAP` 把每种指令映射到 `_exec_forward_pass()`、`_exec_send_activations()` 等方法。这是介于 Megatron 的过程式与 PyTorch pipelining 的动作表之间的第三种写法：调度是声明的（指令序列），执行是解释的。`num_pipe_buffers()` 返回 `min(stages - stage_id, micro_batches)`——就是 1F1B 在途 micro-batch 数 $$p - i$$。PP 与 ZeRO 的组合有限制：`_configure_zero_optimizer()` 在 `PipelineModule` 下强制 `overlap_comm=False`，Stage 3 与 PP 不兼容。

JSON 配置是 DeepSpeed 易用性的来源，也是它的代价所在。`runtime/zero/config.py` 的 `DeepSpeedZeroConfig` 有四十多个字段，其中 Stage 3 的六个阈值（`stage3_prefetch_bucket_size`、`stage3_param_persistence_threshold`、`stage3_max_live_parameters`、`stage3_max_reuse_distance`、`sub_group_size`、`reduce_bucket_size`）共同决定显存峰值与通信重叠程度，但它们之间的关系不在任何一个地方被算出来——用户要靠经验或 `see_memory_usage` 的日志调。而且配置里有大量隐式耦合：`train_batch_size`、`train_micro_batch_size_per_gpu`、`gradient_accumulation_steps` 三者给两个推第三个；`bf16.enabled` 与 `fp16.enabled` 互斥；`zero_optimization.offload_optimizer` 只在 Stage 1–3 有效但 Stage 1/2 与 3 的行为不同。JSON 无法表达这些约束，只能在 `DeepSpeedConfig` 的校验里报错。


## 五、torchtitan：用原生原语组合

### 1. 目录地图

```text
torchtitan/
  train.py                             main()：ConfigManager 解析 --module / --config → Trainer(config).train()
  trainer.py                           Trainer：__init__ 装配、train_step、train
  config/
    configs.py                         TrainingConfig、ParallelismConfig、CompileConfig、CommConfig、DebugConfig
    configurable.py / manager.py       Configurable 基类（每个组件有内嵌 Config，Config.build() 构造组件）；ConfigManager
  distributed/
    parallel_dims.py                   ParallelDims、MeshAxisName、SpmdLayout
    fsdp.py                            apply_fsdp_to_decoder、get_fsdp_reshard_after_forward_policy
    tensor_parallel.py                 NoParallel（TP 的主体不在这里，见 protocols/ 与 models/common/）
    pipeline_parallel.py               pipeline_llm、_build_pipeline_schedule、_pipeline_module_split
    context_parallel/api.py            apply_cp_to_forward、prepare_context_parallel_input、cp_shard
    activation_checkpoint.py           FullAC、SelectiveAC
    compile.py / cudagraph.py          torch.compile 与 CUDA graph 包装
    full_dtensor.py                    resolve_fsdp_mesh、validate_config
    spmd_types.py                      spmd_types 后端的辅助函数
    utils.py                           init_distributed、set_determinism、clip_grad_norm_、dist_sum / dist_max、set_pg_timeouts
  protocols/
    module.py                          Module(nn.Module, Configurable)：parallelize()、_distribute_states()、_redistribute_inputs / outputs
    sharding.py                        ShardingConfig、LocalMapConfig、resolve_placements
    model.py / model_spec.py           BaseModel、ModelSpec（parallelize_fn、pipelining_fn、state_dict_adapter）
  models/
    common/                            attention、moe、embedding、linear、decoder_sharding（TP/SP 方案）、moe_sharding
    llama3/                            __init__（model_registry 与 debugmodel / 1B / 3B / 8B / 70B / 405B 规格）、model、parallelize、sharding、config_registry
  components/
    checkpoint.py + checkpointer/      CheckpointManager；DCP 与 torch.save 两种后端
    optimizer/                         OptimizersContainer、default_adamw、LRSchedulersContainer
    dataloader.py                      BaseDataLoader；hf_datasets/ 下的 HuggingFaceTextDataLoader
    metrics.py                         MetricsProcessor、DeviceMemoryMonitor、TensorBoard / W&B logger
    loss.py                            CrossEntropyLoss、ChunkedLossWrapper
    validate.py                        Validator
  tools/profiler.py                    Profiler（torch.profiler 包装）、MemoryProfiler
  experiments/                         torchft、autoparallel 等
```

与总纲和 brief 的对照：v0.3.0 没有 `components/data/`，数据在 `components/dataloader.py` 与 `hf_datasets/`；checkpoint 在 `components/checkpoint.py` + `components/checkpointer/`；metrics 在 `components/metrics.py`，另有 `observability/` 目录。

### 2. Trainer.__init__：装配的顺序就是架构

torchtitan 的架构最好的说明书是 `trainer.py` 里 `Trainer.__init__` 的执行顺序：

```text
Trainer.__init__(config: Trainer.Config)
  init_distributed()                             dist_utils.init_distributed(config.comm) → ParallelDims.from_config(config.parallelism, world_size)
                                                 → parallel_dims.build_mesh()（第二章）
  set_determinism()
  model_config.build()  在 torch.device("meta") 下   模型只有形状，没有内存；dtype = training.dtype（默认 float32）
  metrics_processor = config.metrics.build()
  loss_fn = config.loss.build()
  gradient_accumulation_steps = global_batch_size / (local_batch_size × batch_degree)
  if pp_enabled:
    pp_schedule, model_parts, ... = model_spec.pipelining_fn(model, parallel_dims, ..., parallelize_fn=model_spec.parallelize_fn)
                                                 → pipeline_llm：切 module、每个 stage 建 PipelineStage、对每个 model part 调 parallelize_fn
  else:
    model = model_spec.parallelize_fn(model, parallel_dims, ...)      → parallelize_llama：TP/CP → AC → compile → FSDP
    model_parts = [model]
  for m in model_parts: m.to_empty(device); m.init_weights()          此时才分配显存、初始化——已经是分片后的形状
  optimizers = config.optimizer.build(model_parts)                    OptimizersContainer：每个 model part 一个 torch 优化器
  lr_schedulers = config.lr_scheduler.build(optimizers)
  tokenizer / dataloader = config.dataloader.build(dp_world_size=batch_degree, dp_rank=batch_rank, ...)
  checkpointer = config.checkpoint.build(dataloader, model_parts, optimizers, lr_schedulers, states={"train_state": self})
  train_context = dist_utils.get_spmd_context(parallel_dims)
```

两个顺序上的决定造就了 torchtitan 的两个特性。**meta 设备构造 → 并行化 → `to_empty` → `init_weights`**：模型在被切分之前不占任何显存，405B 也能在 8 卡上"构造"出来；切分后各卡只分配自己那份。这是 DeepSpeed 需要 `zero.Init` 上下文才能做到的事，torchtitan 用 PyTorch 的 meta 设备原生做到。**并行化在优化器之前**：优化器看到的参数已经是 DTensor 分片，所以 `OptimizersContainer` 只是普通的 `torch.optim.AdamW` 加上 `Stateful` 接口，不需要知道任何并行——分布式优化器"消失"了，因为分片参数上的逐元素更新本来就是局部的。

`Trainer.Config` 是整棵配置树的根：每个组件（metrics、loss、optimizer、dataloader、checkpoint、profiler、activation_checkpoint、validator）是一个 `Configurable`，有内嵌的 `Config` dataclass，`Config.build(...)` 构造实例。`torchtitan/config/configs.py` 只放没有归属组件的字段（`TrainingConfig`、`ParallelismConfig` 等）。一次运行就是一个返回 `Trainer.Config` 的函数，例如 `models/llama3/config_registry.py` 的 `llama3_debugmodel()`；`--module llama3 --config llama3_debugmodel` 选它。

### 3. TP：从 ParallelStyle 到 ShardingConfig

第二篇讲的 PyTorch 原生 TP 是 `torch/distributed/tensor/parallel/style.py` 的 `ColwiseParallel` / `RowwiseParallel` / `SequenceParallel` 加 `parallelize_module()`。torchtitan v0.3.0 在默认的 `spmd_types` 与 `full_dtensor` 后端下**不再直接调用它们**，而是走一条声明式的路：

- `torchtitan/protocols/sharding.py` 的 `ShardingConfig` 用四组字段描述一个 module 的切分：`state_shardings`（每个参数 / buffer 的 placement）、`in_src_shardings` / `in_dst_shardings`（输入进来时的 placement 与希望重分布到的 placement）、`out_src_shardings` / `out_dst_shardings`（输出同理）、`local_map`（可选，把 forward 包成 `local_map()` 在本地张量上算）。placement 用 `SpmdLayout`（`parallel_dims.py`）按 mesh 轴名给出，例如 `{DP: R, CP: R, TP: S(0)}`。
- `torchtitan/models/common/decoder_sharding.py` 用 `ShardingConfig` 写出一个 decoder 的完整 TP/SP 方案：`colwise_config()`（权重 `S(0)`、输出 `S(-1)`，docstring 直接写"ColwiseParallel"）、`rowwise_config(output_sp=...)`（权重 `S(1)`、输出从 `Partial` 重分布到 `Replicate`（all-reduce）或 `Shard(1)`（reduce-scatter，即 SP））、`norm_config(enable_sp=...)`、`set_gqa_attention_sharding()`、`set_dense_ffn_sharding()`、`set_decoder_sharding_config()`。`models/llama3/sharding.py` 的 `set_llama3_sharding_config()` 把它们挂到 Llama 3 的各个子 module 上。
- `torchtitan/protocols/module.py` 的 `Module.parallelize(parallel_dims)` 递归执行：对每个带 `_sharding_config` 的 module，`_distribute_states()` 用 `torch.distributed.tensor.distribute_tensor()` 按 placement 把参数变成 DTensor，然后把 `forward` 包成"重分布输入 → [local_map] forward → 重分布输出"——通信由 DTensor 的 `redistribute()` 从 src / dst placement 之差推导出来。

所以 TP 在 torchtitan 里的实现位置从 `distributed/tensor_parallel.py`（v0.3.0 里只剩一个 `NoParallel` 风格类）搬到了 `protocols/` 与 `models/common/`。语义没变：列切 + 行切配对、SP 把 all-reduce 拆成 all-gather + reduce-scatter，都在 `decoder_sharding.py` 的几个函数里能对上第二篇的图；变的是表达方式——切分方案与模型代码分离，同一份 `ShardingConfig` 在 `spmd_types` 后端下还可以做静态类型检查（`spmd_validate_redistributions()`，`debug.spmd_typechecking` 打开时生效）。`partial_dtensor` 后端保留了旧路径（`parallelize_llama()` 里的 else 分支）。

### 4. FSDP2：apply_fsdp_to_decoder

`parallelize_llama()`（`models/llama3/parallelize.py`）的最后一步总是 `apply_fsdp_to_decoder()`（`distributed/fsdp.py`），即使 `dp_shard = 1`——注释说明这是为了安装 `MixedPrecisionPolicy`。它做的事：

```text
mp_policy = MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32, cast_forward_inputs=False)
                                                       training.mixed_precision_param / mixed_precision_reduce
reshard_after_forward = "default" → not pp_enabled     get_fsdp_reshard_after_forward_policy；PP 下默认不 reshard，避免每个 micro-batch 都 all-gather
fully_shard(tok_embeddings, mesh, mp_policy, reshard_after_forward)
for block in layers:  fully_shard(block, ...)          每个 Transformer 层一个 FSDP 单元
fully_shard([norm, lm_head], ..., reshard_after_forward=False)    最后几层前向后不释放：反向马上要用
```

`fully_shard()` 来自 `torch/distributed/fsdp/_fully_shard/_fully_shard.py`。对一个 module 调用它：把 module 的类动态换成 `FSDP<原类名>`（多继承 `FSDPModule`）、建一个 `FSDPState`（`_fsdp_state.py`）和一个 `FSDPParamGroup`（`_fsdp_param_group.py`），组内每个参数一个 `FSDPParam`（`_fsdp_param.py`）。`FSDPParam._init_sharded_param()` 把参数沿第 0 维切成 `mesh` 大小份，本 rank 的一份包成 `DTensor(placements=(Shard(0),))`（HSDP 时 `(Replicate(), Shard(0))`），**dtype 保持原样**——torchtitan 默认 `training.dtype = float32`，所以分片是 fp32。如果参数已经是 TP 的 DTensor（`Shard(0)` 或 `Shard(1)` 在 tp 维），FSDP 在它外面再加一维，得到一个二维 mesh 上的 DTensor。

前向与反向由 `FSDPState` 注册的 hook 驱动（`_register_group_forward_hooks()`）：`_pre_forward` → `FSDPParamGroup.pre_forward()` → `unshard()`（`_fsdp_collectives.py` 的 `foreach_all_gather()`：组内所有参数的分片拷进一个连续 buffer、一次 `all_gather_into_tensor`，输入在此处按 `mp_policy.param_dtype` 转 bf16）→ `wait_for_unshard()` → `foreach_all_gather_copy_out()` → `FSDPParam.to_unsharded()`（bf16 的完整参数注册到 module 上）；`_post_forward` → `post_forward()` → `reshard()` → `to_sharded()`（释放 bf16，module 上又是 fp32 分片）；反向 `pre_backward()` 再 unshard（`_backward_prefetch()` 按前向的逆序预取），`post_backward()` → `foreach_reduce()`（梯度拷进连续 buffer、`reduce_dtype` 为 fp32 的 reduce-scatter、结果写到分片参数的 `.grad`——也是 `Shard(0)` 的 fp32 DTensor）。三条 stream（`FSDPCommContext` 的 all-gather / reduce-scatter / all-reduce stream）让通信与计算重叠。

### 5. PP 与 CP

**PP**（`distributed/pipeline_parallel.py`）。`pipeline_llm()` 是 `ModelSpec.pipelining_fn`：`_get_pipeline_metadata()` 算出每个 PP rank 负责哪些 stage（`_get_pp_rank_to_stage_indices_mapping()`，支持 looped / V 形分配）、`_generate_llm_fqn_per_model_part()` 算出每个 stage 包含哪些 module（按 `pipeline_parallel_layers_per_stage` 或 `module_fqns_per_model_part`，首末 stage 可少几层以平衡 embedding / lm_head），`_pipeline_module_split()` 对每个 stage `copy.deepcopy` 整个模型再 `_split_module()` 删掉不属于它的 module，包成 `torch.distributed.pipelining.PipelineStage`；然后对每个 model part 调 `parallelize_fn`（TP、FSDP 都在 stage 内部再做一次）；最后 `_build_pipeline_schedule()` 用 `get_schedule_class(parallelism.pipeline_parallel_schedule)` 取调度类（`"1F1B"`、`"Interleaved1F1B"`、`"ZBVZeroBubble"` 等），`n_microbatches = local_batch_size / pipeline_parallel_microbatch_size`，`scale_grads=False`（梯度缩放由 loss 里的 token 数处理）。`pipeline_parallel_schedule_csv` 非空时用 `_PipelineScheduleRuntime` 加载一张自定义动作表。训练时 `Trainer.pp_forward_backward_step()` 调 `pp_schedule.step(arg_mbs, kwarg_mbs, target_mbs, losses)`——第六章对照两种 1F1B 时展开。

**CP**（`distributed/context_parallel/api.py`）。`apply_cp_to_forward()` 在 `partial_dtensor` 后端下把每层的 `inner_attention` 替换成 PyTorch 实验 API `torch.distributed.tensor.experimental._context_parallel` 的 ring attention（第二篇讲过）；`prepare_context_parallel_input()` 在 `Trainer.post_dataloading_process()` 里按 `context_parallel_load_balancer`（默认 `"headtail"`）把输入序列切成本 rank 的份。默认后端下 CP 由 `decoder_sharding.py` 的 `set_gqa_inner_attention_local_map()` 用 `LocalMapConfig` 描述。CP 对参数没有影响——第二章说过 torchtitan 把 `cp` 并入 FSDP 的 shard 维，梯度归约自动覆盖。

### 6. 一个 bf16 参数的一生：torchtitan

以一个 Transformer 层里 `feed_forward.w1.weight` 为例（`dp_shard = 8`，`tp = 1`，`training.dtype = float32`，`mixed_precision_param = bfloat16`，`fsdp_reshard_after_forward = "default"`，无 PP）：

```text
时刻                    事件                                                          在哪里
─────────────────────   ──────────────────────────────────────────────────────────    ─────────────────────────────────────────
meta 构造               weight 是 meta 张量（fp32，完整形状，无内存）                        Trainer.__init__ 的 torch.device("meta")
parallelize_llama       fully_shard(block) → FSDPParam._init_sharded_param              torch/distributed/fsdp/_fully_shard/_fsdp_param.py
                        weight ← DTensor(Shard(0))，本地 1/8 行，fp32，仍是 meta
to_empty + init_weights weight 的本地分片分配显存并初始化（fp32，1/8）                       trainer.py
优化器构造              AdamW 的 param 就是这个 fp32 分片；m、v 按它的形状分配                  components/optimizer/optimizer.py OptimizersContainer
step i，该层前向前       FSDPState._pre_forward → FSDPParamGroup.unshard                   _fsdp_state.py / _fsdp_param_group.py
                        foreach_all_gather：8 个 fp32 分片各自转 bf16 → 一次 all-gather         _fsdp_collectives.py
                        wait_for_unshard → to_unsharded：block.feed_forward.w1.weight ← 完整 bf16
该层前向                GEMM 读 bf16 完整 weight
该层前向后              _post_forward → reshard → to_sharded：bf16 释放，weight 又是 fp32 分片
该层反向前              pre_backward → unshard（同上，再一次 all-gather；由 _backward_prefetch 提前发出）
该层反向                dW（bf16 完整）→ post_backward → foreach_reduce：reduce-scatter，fp32 归约
                        分片.grad ← DTensor(Shard(0))，fp32，1/8；完整 bf16 weight 与 dW 释放
optimizers.step         AdamW 在 fp32 分片上更新（局部逐元素，不需要通信）                    torch.optim.AdamW
step i+1，该层前向前     再 unshard：从更新后的 fp32 分片 all-gather 出新的 bf16
```

**常驻**：fp32 分片 $$4N/N_d$$、fp32 梯度分片 $$4N/N_d$$、m + v $$8N/N_d$$——合计 $$16N/N_d$$，**没有独立的 bf16 参数和 fp32 主参数两份**。**临时**：正在算的层（及被预取的下一层）的完整 bf16 参数与 bf16 梯度。**通信**：与 DeepSpeed Stage 3 相同的 $$3N$$（前向 all-gather、反向 all-gather、reduce-scatter），PP 下 `reshard_after_forward=False` 时 $$2N$$。

对比三份时序，torchtitan 的独特之处是**fp32 分片既是模型参数也是主参数**。这不是 FSDP2 的发明——FSDP1 用 `MixedPrecisionPolicy` 时也是如此——但 torchtitan 把它作为默认（`training.dtype = float32`）。代价是模型在"未并行化"时是 fp32 的，单卡调试要 4 字节每参数；收益是少一次 fp32 → bf16 的拷贝、少一份内存、checkpoint 里只有一份权重（fp32 分片），恢复时不需要重建主参数。

### 7. 训练循环：train_step

`Trainer.train()` 的循环很短：加载 checkpoint → `while step < steps:` `train_step(data_iterator)` → `checkpointer.save()` → 可选 validate → `profiler.step()`。第一步之后 `set_pg_timeouts()` 把进程组超时从 `comm.init_timeout_seconds`（默认 300 s，容忍初始化与编译）降到 `comm.train_timeout_seconds`（默认 100 s）——hang 检测的第一道闸，第八篇回来看。

`train_step()`：

```text
optimizers.zero_grad()
取 gradient_accumulation_steps × num_pipeline_parallel_microbatches 个 micro-batch，统计 local_valid_tokens
global_valid_tokens = dist_sum_tensor(local_valid_tokens, batch_mesh)          loss 按全局 token 数归一，梯度累积不再需要 /N
for 每个梯度累积步:
  forward_backward_step()
    非 PP：with train_context(): pred = model(inputs); loss = loss_fn(pred, labels, global_valid_tokens); loss.backward()
    PP：  pp_schedule.step(arg_mbs, kwarg_mbs, target_mbs, losses, loss_kwargs)
grad_norm = dist_utils.clip_grad_norm_(all params, max_norm, pp_mesh=..., ep_enabled=...)   DTensor 感知：分片范数 + 跨 PP 归约
checkpointer.maybe_wait_for_staging()                                                       异步 checkpoint 的 staging 完成后才能改参数
optimizers.step(); lr_schedulers.step()
metrics_processor.log(step, global_avg_loss, global_max_loss, grad_norm, ...)                loss 在 loss_mesh 上 dist_sum / dist_max
```

与 Megatron 的 `train_step()` 对照：没有 `zero_grad_buffer`（FSDP 自己管梯度 buffer）、没有 `no_sync`（FSDP2 用 `set_requires_gradient_sync()` 控制，torchtitan 默认每个梯度累积步都 reduce-scatter——因为 reduce-scatter 的结果就是分片梯度，累积直接在分片上做，不像 DDP 那样"最后一步才通信"能省什么）、没有单独的 `finalize_model_grads`（DTensor 的 `Partial` placement 让 TP 组内复制参数的梯度归约由 redistribute 自动完成）。整个循环里看不到任何一次显式的集合通信调用，除了 `dist_sum_tensor` 与 `clip_grad_norm_` 里的范数归约——所有参数相关的通信都藏在 FSDP 的 hook 和 DTensor 的 redistribute 里。这是"可读"的来源，也是"看不见"的来源：要知道通信发生在哪里，得读 `_fully_shard/`，而不是 torchtitan。


## 六、对照阅读

### 1. 三条时序并排

把三、四、五章的三张时序压成一张表，只看一个 DP 组内的一个参数：

```text
                          Megatron（DDP + 分布式优化器）      DeepSpeed Stage 3                    torchtitan（FSDP2）
────────────────────────  ────────────────────────────────  ──────────────────────────────────  ──────────────────────────────────
bf16 参数常驻形态          完整；param_data buffer 的视图        1/N_d 碎片 ds_tensor；param.data 为空     无；常驻的是 fp32 Shard(0) DTensor
bf16 参数何时完整          始终                                该层前向 / 反向期间（hook 取回）         该层前向 / 反向期间（unshard）
何时释放                  从不                                该层前向后 / 反向后（partition）          该层前向后 / 反向后（reshard）
fp32 主参数在哪            优化器私有的 1/N_d 拷贝               扁平 fp32 分区（可 CPU）                 就是那个 fp32 分片
梯度形态                  fp32 完整 main_grad（buffer 视图）     bucket 后 reduce-scatter → 1/N_d 分区     bf16 临时 → reduce-scatter → fp32 1/N_d 分片
优化器何时碰它             step：拷 grad → 更新 → 拷回 bf16 段    step：每个 sub-group 更新 → 拷回 ds_tensor   step：直接更新 fp32 分片
更新后 all-gather          step 内发起，下一 step 前向 pre-hook 等   不做（persist 参数除外）；下一 step 前向取回   不做；下一 step 前向 unshard
每 step 通信              RS(N) + AG(N) = 2N                 AG + AG + RS = 3N                    AG + AG + RS = 3N（不 reshard 则 2N）
每参数常驻字节            2 + 4 + 12/N_d                      (2 + 2 或 4 + 12)/N_d                16/N_d
param.data 身份           固定                                一个 step 内换四次                       一个 step 内换四次（但 DTensor 自描述）
```

最后两行解释了三个框架的"手感"。Megatron 的显存曲线是一条直线（常驻大、临时少），profile 里通信是两大块（reduce-scatter 跟着反向、all-gather 跟着下一步前向）；DeepSpeed 与 torchtitan 的显存曲线是锯齿（常驻小、每层一个峰），通信是每层三小块。Megatron 的 `param.data` 可以随便读——checkpoint、日志、调试都直接看；DeepSpeed 的要用 `GatheredParameters`；torchtitan 的是 DTensor，`full_tensor()` 或 `to_local()` 二选一，但它知道自己是什么。

### 2. 前向前 all-gather 分片参数：Stage 3 vs FSDP1 vs FSDP2

第二篇提出的对照题：同一个"前向前把分片参数拼回来"，三种实现。现在可以逐项回答。

```text
                     DeepSpeed Stage 3                         FSDP1（FlatParameter）                       FSDP2（fully_shard）
───────────────────  ────────────────────────────────────────  ──────────────────────────────────────────  ──────────────────────────────────────────
分片单位              每个参数各自切成 1/N_d（ds_tensor）           一个 wrap 单元的所有参数拍平成一个 FlatParameter 再切   每个参数各自沿第 0 维切（FSDPParam），组成 FSDPParamGroup
触发点                每个子 module 的 forward pre-hook            wrap 单元的 forward pre-hook                  fully_shard 单元的 forward pre-hook
                     parameter_offload.py _pre_forward_module_hook  _runtime_utils.py _pre_forward → _pre_forward_unshard  _fsdp_state.py _pre_forward → FSDPParamGroup.pre_forward
all-gather 调用       fetch_sub_module → all_gather_coalesced      _unshard → FlatParamHandle.unshard → _all_gather_flat_param   unshard → foreach_all_gather（拷进连续 buffer、一次 all_gather_into_tensor）
                     一个 module 的多个参数拼成一次通信              一次通信（本来就是一个张量）                   一次通信
完整参数如何出现       param.data 指向拼好的张量                     _use_unsharded_flat_param → _use_unsharded_views：原参数变成 FlatParameter 的视图   foreach_all_gather_copy_out → FSDPParam.to_unsharded：各参数恢复原形状
dtype 转换            all-gather 前 get_allgather_dtype             MixedPrecision：拷进低精度 FlatParameter 再 gather   all_gather_inputs 按 mp_policy.param_dtype 转
预取                  trace 驱动（stage3_prefetch_bucket_size）      forward_prefetch / backward_prefetch（_prefetch_handle）   set_modules_to_forward_prefetch / _backward_prefetch（默认按前向逆序）
前向后释放             release_sub_module → partition               _post_forward_reshard → FlatParamHandle.reshard      post_forward → reshard → to_sharded
                     除 ds_persist                                 除 reshard_after_forward=False                  除 reshard_after_forward=False；可重分片到更小组（SHARDED_POST_FORWARD）
分片的类型             普通 Tensor + ds_* 属性                        FlatParameter（nn.Parameter 子类）+ 元数据        DTensor(Shard(0))
参数身份               保留（每参数一个 ds_tensor）                    抹掉（同一 dtype、同时冻结、checkpoint 是一维段）    保留（每参数一个 DTensor）
与 TP 组合            要靠外部框架                                  需专门适配                                     天然（二维 DTensor）
```

三者的通信量相同（都是每单元一次 all-gather），差别在**分片的表示**决定了什么能做、什么不能做。Stage 3 的 `ds_tensor` 是最灵活的（每参数独立，可以 offload、可以量化、可以 hpZ 二级分片），但它是 DeepSpeed 私有的约定，PyTorch 的其他组件（DCP、TP、`torch.compile`）都不认识 `ds_*` 属性。FSDP1 的 `FlatParameter` 通信效率最高、实现最简单，但抹掉了参数身份，是它被 FSDP2 取代的原因。FSDP2 的 `DTensor` 是三者中唯一"自描述"的：分片知道自己在哪个 mesh 的哪一维、全局形状多大，所以 DCP 能按参数名重分片（第五篇）、TP 能在另一维再切、`clip_grad_norm_` 能正确算范数——第二篇说的"可组合性质变"就在这里。

### 3. 1F1B 的两种写法

同一个 1F1B 调度（第二篇第五章），Megatron 与 PyTorch 各写了一遍。

**Megatron：过程式**。`megatron/core/pipeline_parallel/schedules.py` 的 `forward_backward_pipelining_without_interleaving()` 是一个几百行的函数，三段循环直接对应 1F1B 的三个阶段：

```text
num_warmup_microbatches = min(total_stages - current_stage - 1, num_microbatches)     stage i 预热 p - i - 1 个
num_microbatches_remaining = num_microbatches - num_warmup_microbatches

warmup:   for i in range(num_warmup_microbatches):
            input  = p2p_communicator.recv_forward(shapes)
            output = forward_step(...)
            p2p_communicator.send_forward(output)
            input_tensors.append(input); output_tensors.append(output); deallocate_output_tensor(output)
steady:   input = recv_forward()                                            先收一个
          for i in range(num_microbatches_remaining):
            output = forward_step(input)
            output_grad = p2p_communicator.send_forward_recv_backward(output)     发前向、收反向合成一次通信
            input, output = input_tensors.pop(0), output_tensors.pop(0)
            input_grad = backward_step(input, output, output_grad)
            input = p2p_communicator.send_backward_recv_forward(input_grad)        发反向、收前向合成一次通信
cooldown: for i in range(num_warmup_microbatches):
            output_grad = recv_backward(); input_grad = backward_step(...); send_backward(input_grad)
```

点对点在 `p2p_communication.py` 的 `P2PCommunicator` 类：`_communicate()` 把"发给 next / 发给 prev / 从 next 收 / 从 prev 收"四种需求打包成一次 `batch_isend_irecv`（`_batched_p2p_ops()`，或 `_p2p_ops()` 逐个发），所以 `send_forward_recv_backward()` 是**一次**通信调用而不是两次——这是 1F1B 稳态里每步只有一次同步点的关键。两个辅助函数值得记住：`deallocate_output_tensor()` 在 output 发出后把它的 storage 释放（只留一个空壳给 autograd 图），因为下游 stage 已经拿到数据、本地只需要在反向时接收梯度；相应地 `custom_backward()` 绕过 `torch.autograd.backward` 对 output 形状的检查直接调 `Variable._execution_engine.run_backward`。梯度归约的时机由 `no_sync_func` 与 `enable_grad_sync()` 控制：非首 stage 在 cooldown 的最后一个反向才打开梯度同步，让 DP 的 reduce-scatter 落在气泡里。

**PyTorch：先是过程式，再是动作表**。`torch/distributed/pipelining/schedules.py` 的 `Schedule1F1B._step_microbatches()` 结构与 Megatron 几乎逐行对应：`warmup_chunks = min(n_microbatches, num_stages - stage_index)`，warmup 循环 `get_fwd_recv_ops()` → `forward_one_chunk()` → `get_fwd_send_ops()`，稳态循环把上一轮的 `fwd_sends` 与本轮的 `bwd_recvs` 合成一次 `_batch_p2p()`（注释写的是"1B1F"——先反向再前向，与 Megatron 的"1F1B"只是切入点不同）。差别在两处：通信操作是 `dist.P2POp` 列表由 `PipelineStage`（`stage.py`）生成、调度器只负责 batch 与 wait，接收 buffer 的形状由 `PipelineStage._prepare_forward_infra()` 在第一个 micro-batch 时推断（`_forward_metadata_inference()`，`_utils.py` 的 `InferenceMode` 决定静态还是动态推断）而不是像 Megatron 那样由 `get_tensor_shapes()` 从配置算出；`backward_one_chunk()` 走 `_backward.py` 的 `stage_backward()`，它还有 `stage_backward_input()` / `stage_backward_weight()` 两个拆开的版本——这是 zero-bubble 调度的前提。

`Schedule1F1B` 同时实现了 `_get_pipeline_order()`：把上面的过程翻译成一张按时间步排列的 `_Action` 表（`_ComputationType` 枚举：`FORWARD`、`FULL_BACKWARD`、`BACKWARD_INPUT`、`BACKWARD_WEIGHT`、`SEND_F` / `RECV_F` / `SEND_B` / `RECV_B`、`UNSHARD` / `RESHARD`、`REDUCE_GRAD`）。`_PipelineScheduleRuntime` 就是执行这种表的解释器：`_add_send_recv()` 给计算动作插入配对的通信动作、`_add_unshard_reshard()` 插入 FSDP 的 unshard / reshard（PP 与 FSDP 组合时让参数 all-gather 提前）、`_merge_bw()` 把相邻的 `BACKWARD_INPUT` 与 `BACKWARD_WEIGHT` 合并。`ScheduleInterleaved1F1B`、`ScheduleInterleavedZeroBubble`、`ScheduleZBVZeroBubble`、`ScheduleDualPipeV` 都是 `_PipelineScheduleRuntime` 的子类，只重写 `_calculate_single_rank_operations()` 生成各自的表；`_load_csv()` 甚至允许从 CSV 加载一张手写的表——torchtitan 的 `pipeline_parallel_schedule_csv` 就接在这里。

两种写法的取舍：Megatron 的过程式**快**——每一步做什么在代码里写死，没有解释开销，与 Megatron 自己的 DDP、分布式优化器、interleaved 调度的 `overlap_p2p_comm` 深度耦合（`forward_backward_pipelining_with_interleaving()` 有一千行）；PyTorch 的动作表**通用**——新调度只是一张新表，PP 与 FSDP、与 zero-bubble 的组合是表变换，代价是每步的解释与 `_batch_p2p` 的开销，以及形状推断带来的第一步延迟。DeepSpeed 的 `TrainSchedule` 在两者之间：指令序列是声明的，但只有一种调度。

### 4. 取舍表

```text
                  Megatron Core 0.18.0                          DeepSpeed 0.19.2                             torchtitan v0.3.0
────────────────  ────────────────────────────────────────────  ───────────────────────────────────────────  ────────────────────────────────────────────
强项              极致性能：TE / FP8、手写重叠、MoE 全套            对模型零侵入；ZeRO-3 + offload 让小集群训大模型   可读、可组合；PyTorch 原生功能的参考实现
模型代码约束       必须用 megatron/core/transformer 的层写；         任意 nn.Module；PP 要改写成 LayerSpec 列表        用它的 Module 协议 + ShardingConfig；层是普通 PyTorch
                  外部模型需转换
DP 系覆盖         ZeRO-1（分布式优化器）+ Megatron-FSDP（ZeRO-1/2/3）  ZeRO-1/2/3、offload、hpZ、MiCS                FSDP2（ZeRO-3）、HSDP；无 ZeRO-1/2
模型并行覆盖       TP+SP、PP（1F1B / interleaved）、CP、EP 全套      依赖外部 mpu 的 TP；自带 PP；Ulysses CP；EP       TP+SP、PP（pipelining 全部调度）、CP（实验 API）、EP
配置              几百个 argparse flag，validate_args 做约束检查      JSON，约束隐式                               Python 函数返回 Trainer.Config，类型检查
调试              param.data 随时可读；buffer 布局要懂               参数常为空张量；GatheredParameters；日志靠 see_memory_usage   DTensor 自描述；但通信藏在 hook 里
checkpoint        dist_checkpointing（ShardedTensor）+ torch_dist / fsdp_dtensor 格式   ZeRO checkpoint 按 rank；universal checkpoint 转换   DCP（DTensor 原生重分片）
可观测            timers、log_memory_to_tensorboard、straggler 检测   wall_clock_breakdown、flops_profiler           MetricsProcessor、DeviceMemoryMonitor、Profiler、Flight Recorder 配置
代码量与耦合       大；DDP ↔ 优化器 ↔ 调度通过回调耦合                 大；engine ↔ ZeRO 优化器强耦合                 小；耦合在 PyTorch 侧
适合              千卡 dense / MoE 预训练，追求 MFU                 已有训练脚本要放大；显存不够；小集群              新项目、研究、需要改并行策略的场景
```

三者对同一格子的选择差异，多数可以从"从哪种状态出发"解释。Megatron 从模型结构出发，所以要求模型用它的层写、并行的每一步都手写、能做到最紧的重叠；DeepSpeed 从优化器状态出发，所以只需要接管优化器和 hook、对模型零侵入、但"参数不在"成为常态；torchtitan 从参数的表示出发，所以一切并行都是 placement 的变换、代码最少、但依赖 PyTorch 侧的实现是否到位。Megatron 引入 FSDP、DeepSpeed 引入 DeviceMesh、torchtitan 引入 `ShardingConfig`——三者都在向"分片是数据的属性而不是框架的秘密"靠拢。


## 七、本文小结

### 1. 要点回顾

```text
主线          一个 bf16 参数：Megatron 常驻完整（buffer 视图）、DeepSpeed Stage 3 常驻 1/N_d 碎片（hook 取回）、torchtitan 不常驻（fp32 分片 unshard 出 bf16）
fp32 主参数   Megatron 优化器私有的 1/N_d 拷贝；DeepSpeed 扁平 fp32 分区；torchtitan 就是 fp32 分片本身
每参数字节    Megatron 2 + 4 + 12/N_d（默认 fp32 梯度）；DeepSpeed (16 或 18)/N_d + 临时层；torchtitan 16/N_d + 临时层
通信          Megatron RS + AG = 2N（AG 与下一步前向重叠）；DeepSpeed / torchtitan AG + AG + RS = 3N（不 reshard 则 2N）
进程组        Megatron RankGenerator(order) 填全局变量；DeepSpeed 委托 mpu 或 world 克隆；torchtitan DeviceMesh _unflatten 成几张视图
Megatron      _ParamAndGradBuffer 两段连续内存 + bucket group；DDP hook 累加 main_grad、bucket 满则 RS；DistributedOptimizer 按字节区间分片；调度经 config 回调与 DDP 交互
DeepSpeed     engine 包裹 module；ZeRO 优化器包裹用户优化器；Stage 1/2 扁平分区 + IPG bucket；Stage 3 ds_tensor + 四个 module hook + 参数协调器（trace 预取）
torchtitan    meta 构造 → parallelize → to_empty；ShardingConfig 声明 TP，distribute_tensor + redistribute 实现；fully_shard 每层一个单元，MixedPrecisionPolicy 决定 bf16
all-gather 三种  Stage 3 每参数 ds_tensor + coalesced；FSDP1 FlatParameter 一次；FSDP2 每参数 DTensor 拷进连续 buffer 一次——通信量同，表示决定可组合性
1F1B 两种     Megatron 过程式三段循环 + P2PCommunicator 合并收发 + deallocate_output_tensor；PyTorch Schedule1F1B 同构但 P2POp 由 PipelineStage 生成，再翻译成 _Action 表由 _PipelineScheduleRuntime 解释
趋势          Megatron-FSDP（ShardingStrategy 四级）、DeepSpeed 的 DeviceMesh、torchtitan 的 ShardingConfig：分片正在成为数据的属性
```

### 2. 本篇涉及的源码位置

| 路径 | 内容 |
|---|---|
| Megatron Core 0.18.0 `megatron/core/parallel_state.py` | `initialize_model_parallel()`（`order="tp-cp-ep-dp-pp"`、`num_distributed_optimizer_instances`、`create_gloo_process_groups`）、`RankGenerator`、`generate_masked_orthogonal_rank_groups()`、`get_tensor_model_parallel_group()` / `get_data_parallel_group(with_context_parallel=)` / `get_expert_data_parallel_group()` / `get_intra_distributed_optimizer_instance_group()` 等 |
| Megatron Core 0.18.0 `megatron/core/process_groups_config.py`、`hyper_comm_grid.py` | `ProcessGroupCollection`、`HyperCommGrid`：显式传递的进程组 |
| Megatron Core 0.18.0 `megatron/core/tensor_parallel/layers.py`、`mappings.py` | `ColumnParallelLinear`、`RowParallelLinear`、`VocabParallelEmbedding`、`LinearWithGradAccumulationAndAsyncCommunication`、`set_tensor_model_parallel_attributes()`、`param_is_not_tensor_parallel_duplicate()`；`_CopyToModelParallelRegion` 等 |
| Megatron Core 0.18.0 `megatron/core/distributed/distributed_data_parallel.py`、`distributed_data_parallel_config.py` | `DistributedDataParallel`：`_make_forward_pre_hook()`、`_make_backward_post_hook()`、`no_sync()`、`start_grad_sync()` / `finish_grad_sync()`、`start_param_sync()`、`zero_grad_buffer()`；`DistributedDataParallelConfig`（`overlap_grad_reduce`、`overlap_param_gather`、`bucket_size`、`grad_reduce_in_fp32`、`use_megatron_fsdp`、`data_parallel_sharding_strategy`） |
| Megatron Core 0.18.0 `megatron/core/distributed/param_and_grad_buffer.py`、`finalize_model_grads.py` | `_ParamAndGradBuffer`、`_ParamAndGradBucket`、`_ParamAndGradBucketGroup`（`start_param_sync()`、`finish_param_sync()`、`start_grad_sync()`、`register_grad_ready()`）、`shard_buffer()`、`group_params_for_buffers()`；`finalize_model_grads()` |
| Megatron Core 0.18.0 `megatron/core/optimizer/distrib_optimizer.py`、`optimizer.py` | `DistributedOptimizer`：`_build_gbuf_range_map()`、`_build_model_gbuf_param_range_map()`、`_build_model_and_main_param_groups()`（`shard_fp32_from_float16_groups`）、`_copy_model_grads_to_main_grads()`、`_copy_main_params_to_model_params()`、`step_with_ready_grads()`；`MixedPrecisionOptimizer`、`Float16OptimizerWithFloat16Params`、`ChainedOptimizer` |
| Megatron Core 0.18.0 `megatron/core/pipeline_parallel/schedules.py`、`p2p_communication.py` | `get_forward_backward_func()`、`forward_backward_no_pipelining()`、`forward_backward_pipelining_without_interleaving()`、`forward_backward_pipelining_with_interleaving()`、`forward_step()`、`backward_step()`、`deallocate_output_tensor()`、`custom_backward()`、`get_tensor_shapes()`；`P2PCommunicator`（`recv_forward()`、`send_forward()`、`send_forward_recv_backward()`、`send_backward_recv_forward()`、`_communicate()`）、`_batched_p2p_ops()`、`_p2p_ops()` |
| Megatron Core 0.18.0 `megatron/core/distributed/fsdp/` | `mcore_fsdp_adapter.py` 的 `FullyShardedDataParallel`；`src/megatron_fsdp/fully_shard.py` 的 `ShardingStrategy`（`NO_SHARD` / `OPTIM` / `OPTIM_GRADS` / `OPTIM_GRADS_PARAMS`）、`fully_shard_model()`、`fully_shard_optimizer()`；`megatron_fsdp.py` 的 `MegatronFSDP` |
| Megatron Core 0.18.0 `megatron/training/training.py`、`arguments.py`、`config/common_config.py` | `pretrain()`、`setup_model_and_optimizer()`、`get_model()`、`wrap_model_chunks_with_ddp()`、`train()`、`train_step()`、`training_log()`；`validate_args()`（`accumulate_allreduce_grads_in_fp32` 默认）；`use_pytorch_profiler`、`profile_step_start` / `profile_step_end`、`profile_ranks`、`record_memory_history`、`memory_snapshot_path` |
| DeepSpeed 0.19.2 `deepspeed/__init__.py`、`runtime/engine.py` | `initialize()`；`DeepSpeedEngine`：`_configure_distributed_model()`、`_configure_optimizer()`、`_configure_zero_optimizer()`、`forward()`、`backward()`、`_backward_epilogue()`、`allreduce_gradients()`、`step()`、`_take_model_step()`、`is_gradient_accumulation_boundary()`、`get_data_types()` |
| DeepSpeed 0.19.2 `deepspeed/runtime/zero/stage_1_and_2.py` | `DeepSpeedZeroOptimizer`：`bit16_groups_flat`、`parallel_partitioned_bit16_groups`、`single_partition_of_fp32_groups`、`create_gradient_handling_hooks()`、`reduce_ready_partitions_and_remove_grads()`、`reduce_ipg_grads()`、`average_tensor()`、`step()`、`_optimizer_step()`、`update_lp_params()`；`IPGBucket` |
| DeepSpeed 0.19.2 `deepspeed/runtime/zero/stage3.py` | `DeepSpeedZeroOptimizer_Stage3`：`_create_fp16_partitions_with_defragmentation()`、`_create_fp32_partitions()`、`create_reduce_and_remove_grad_hooks()`、`reduce_independent_p_g_buckets_and_remove_grads()`、`partition_grads()`、`step()`、`_optimizer_step()`、`_reassign_or_swap_out_partitioned_parameters()`、`_post_step()`、`_partition_all_parameters()`；`sub_group_size` |
| DeepSpeed 0.19.2 `deepspeed/runtime/zero/partition_parameters.py`、`parameter_offload.py`、`partitioned_param_coordinator.py` | `Init`（`_post_init_method()`、`_convert_to_deepspeed_param()`、`_partition()`）、`ZeroParamStatus`、`free_param()`、`AllGatherCoalescedHandle`；`DeepSpeedZeRoOffload`（`setup_zero_stage3_hooks()`、`_register_deepspeed_module()`、`pre_sub_module_forward_function()` 等四个）；`PartitionedParameterCoordinator`（`fetch_sub_module()`、`release_sub_module()`、`record_module()`、`trace_prologue()`） |
| DeepSpeed 0.19.2 `deepspeed/runtime/zero/config.py`、`runtime/constants.py`、`profiling/constants.py` | `DeepSpeedZeroConfig`（`stage`、`overlap_comm`、`contiguous_gradients`、`reduce_bucket_size`、`allgather_bucket_size`、`prefetch_bucket_size` 等 `stage3_*` 别名、`sub_group_size`、`zero_hpz_partition_size`）；JSON 键常量 |
| DeepSpeed 0.19.2 `deepspeed/runtime/pipe/{module,engine,schedule,topology,p2p}.py` | `PipelineModule`、`LayerSpec`、`TiedLayerSpec`、`_partition_layers()`；`PipelineEngine`（`train_batch()`、`_exec_schedule()`、`_exec_forward_pass()` 等）；`TrainSchedule`（`steps()`、`num_pipe_buffers()`）、`PipeInstruction` 子类；`ProcessTopology`、`PipeModelDataParallelTopology`、`PipelineParallelGrid` |
| DeepSpeed 0.19.2 `deepspeed/utils/groups.py`、`comm/comm.py` | `mpu` 委托、`_get_data_parallel_group()`、`_clone_world_group()`、`_create_expert_and_data_parallel()`、`_create_zero_param_parallel_group()`、`_init_tp_mesh_device()`；`initialize_mesh_device()`、`init_distributed()` |
| torchtitan v0.3.0 `torchtitan/train.py`、`trainer.py` | `main()`；`Trainer`（`Config`、`__init__`、`init_distributed()`、`forward_backward_step()`、`_forward_backward_body()`、`pp_forward_backward_step()`、`train_step()`、`train()`） |
| torchtitan v0.3.0 `torchtitan/config/configs.py`、`README.md` | `TrainingConfig`（`dtype`、`mixed_precision_param`、`mixed_precision_reduce`、`local_batch_size`、`global_batch_size`、`seq_len`）、`ParallelismConfig`（`data_parallel_replicate_degree`、`data_parallel_shard_degree`、`tensor_parallel_degree`、`enable_sequence_parallel`、`pipeline_parallel_degree`、`pipeline_parallel_schedule`、`pipeline_parallel_microbatch_size`、`context_parallel_degree`、`fsdp_reshard_after_forward`、`spmd_backend`）、`CommConfig`、`DebugConfig`；配置即 Python 函数 |
| torchtitan v0.3.0 `torchtitan/distributed/parallel_dims.py`、`full_dtensor.py` | `ParallelDims`（`from_config()`、`build_mesh()`、`get_mesh()`、`get_optional_mesh()`）、`MeshAxisName`、`SpmdLayout`；`resolve_fsdp_mesh()` |
| torchtitan v0.3.0 `torchtitan/distributed/fsdp.py`、`pipeline_parallel.py`、`context_parallel/api.py`、`utils.py` | `apply_fsdp_to_decoder()`、`get_fsdp_reshard_after_forward_policy()`；`pipeline_llm()`、`_build_pipeline_schedule()`、`_pipeline_module_split()`；`apply_cp_to_forward()`、`prepare_context_parallel_input()`；`init_distributed()`、`clip_grad_norm_()`、`set_pg_timeouts()`、`dist_sum_tensor()` |
| torchtitan v0.3.0 `torchtitan/protocols/module.py`、`sharding.py`；`models/common/decoder_sharding.py`；`models/llama3/{__init__,parallelize,sharding,config_registry}.py` | `Module.parallelize()`、`_distribute_states()`；`ShardingConfig`、`LocalMapConfig`、`resolve_placements()`；`colwise_config()`、`rowwise_config()`、`norm_config()`、`set_decoder_sharding_config()`；`model_registry()`、`parallelize_llama()`、`set_llama3_sharding_config()`、`llama3_debugmodel()` |
| torchtitan v0.3.0 `torchtitan/components/{metrics,checkpoint,dataloader,loss}.py`、`components/optimizer/`、`tools/profiler.py` | `MetricsProcessor`、`DeviceMemoryMonitor.get_peak_stats()`；`CheckpointManager`；`BaseDataLoader`；`CrossEntropyLoss`、`ChunkedLossWrapper`；`OptimizersContainer`、`default_adamw()`、`LRSchedulersContainer`；`Profiler.Config`（`enable_profiling`、`profile_freq`、`profiler_active`、`profiler_warmup`） |
| PyTorch 2.13.0 `torch/distributed/fsdp/_fully_shard/` | `_fully_shard.py` 的 `fully_shard()`、`FSDPModule`（`unshard()`、`reshard()`、`set_reshard_after_forward()`、`set_modules_to_forward_prefetch()`、`set_requires_gradient_sync()`）；`_fsdp_state.py` 的 `FSDPState`（`_pre_forward()`、`_post_forward()`、`_pre_backward()`）；`_fsdp_param_group.py` 的 `FSDPParamGroup`（`unshard()`、`wait_for_unshard()`、`reshard()`、`pre_forward()`、`post_forward()`、`pre_backward()`、`post_backward()`、`_backward_prefetch()`）、`FSDPCommContext`；`_fsdp_param.py` 的 `FSDPParam`（`_init_sharded_param()`、`to_unsharded()`、`to_sharded()`、`all_gather_inputs()`）、`ShardedState`；`_fsdp_collectives.py` 的 `foreach_all_gather()`、`foreach_all_gather_copy_out()`、`foreach_reduce()`；`_fsdp_api.py` 的 `MixedPrecisionPolicy`、`DataParallelMeshDims`、`CPUOffloadPolicy` |
| PyTorch 2.13.0 `torch/distributed/fsdp/_flat_param.py`、`_runtime_utils.py` | FSDP1：`FlatParameter`、`FlatParamHandle`（`shard()`、`unshard()`、`_all_gather_flat_param()`、`_use_unsharded_flat_param()`、`_use_unsharded_views()`、`reshard()`）；`_pre_forward()`、`_pre_forward_unshard()`、`_post_forward_reshard()`、`_prefetch_handle()` |
| PyTorch 2.13.0 `torch/distributed/pipelining/schedules.py`、`stage.py`、`_backward.py` | `Schedule1F1B`（`_step_microbatches()`、`_get_pipeline_order()`）、`_Action`、`_ComputationType`、`_PipelineScheduleRuntime`（`_load_csv()`）、`_add_send_recv()`、`_add_unshard_reshard()`、`_merge_bw()`、`_batch_p2p()`、`get_schedule_class()`；`PipelineStage`（`forward_one_chunk()`、`backward_one_chunk()`、`backward_weight_one_chunk()`、`get_fwd_recv_ops()` 等）；`stage_backward()`、`stage_backward_input()`、`stage_backward_weight()` |
| PyTorch 2.13.0 `torch/distributed/device_mesh.py`、`tensor/_api.py`、`tensor/parallel/style.py` | `DeviceMesh`（`__getitem__`、`get_group()`、`_flatten()`、`_unflatten()`）、`init_device_mesh()`；`DTensor`（`redistribute()`、`to_local()`、`full_tensor()`）、`distribute_tensor()`；`ColwiseParallel`、`RowwiseParallel`、`SequenceParallel` |
| train-ledger `runs/{megatron,deepspeed,titan}/`、`probe_memory.py` | 本篇增量，见下 |

### 3. train-ledger 本篇增量：runs/ 与 probe_memory.py

**做什么**：在 8 张卡上用三个框架各跑一次同一个 Llama 风格的小模型（取 Llama 3.2 1B 的形状：16 层、$$h = 2048$$、32 头 / 8 个 K/V 头、FFN 8192、词表 128256、$$s = 8192$$；三个框架都不共享 embedding 与输出层，$$N \approx 1.5 \times 10^9$$），每个框架用它最自然的 8 卡配置，然后用 `probe_memory.py` 读每卡的 `torch.cuda.memory_stats()`、用 `torch.profiler` 数每 step 各类集合通信的次数与时间，和第二篇 `ledger.parallel.place()` 的预测对账。目录：

```text
train-ledger/
  runs/
    megatron/run.sh              TP 2 × PP 2 × DP 2，分布式优化器（ZeRO-1），SP
    deepspeed/ds_config.json     ZeRO-3，DP 8
    deepspeed/train_ds.py        约 40 行的循环：zero.Init → deepspeed.initialize → engine(batch) / backward / step
    titan/ledger_llama3_1b.py    FSDP 4 × TP 2；一个返回 Trainer.Config 的函数
  probe_memory.py                读 memory_stats 与 profiler，写 JSON，与 ledger 对账
```

**预期（先算再跑）**。用第二篇的 `ledger.parallel.report()` 对三个配置各算一遍，常驻状态（不含激活与临时 all-gather）大致是：

```text
配置                         N_local    参数（常驻）    梯度（常驻）    优化器状态             常驻合计   每 step DP 系通信（每卡发出）                    其余
Megatron TP2 PP2 DP2 ZeRO-1   375M     bf16 0.75 GB   fp32 1.5 GB    12 × 375M / 2 = 2.25 GB  ≈ 4.5 GB  RS(fp32 梯度 1.5 GB) + AG(bf16 参数 0.75 GB)，     TP ≈ 8.6 GB NVLink
                                                                                                        N_d = 2 各发一半 → ≈ 1.1 GB                    PP ≈ 0.27 GB
DeepSpeed ZeRO-3 DP8          1.5B     bf16 0.38 GB   bf16 0.38 GB   12 × 1.5B / 8 = 2.25 GB  ≈ 3.0 GB  AG(3 GB) + AG(3 GB) + RS(3 GB)，7/8 → ≈ 7.9 GB    —
torchtitan FSDP4 TP2          750M     fp32 0.75 GB   fp32 0.75 GB    8 × 750M / 4 = 1.5 GB   ≈ 3.0 GB  AG(bf16 1.5 GB) × 2 + RS(fp32 3 GB)，3/4 → ≈ 4.5 GB  TP ≈ 4.3 GB NVLink
```

（Megatron 的 TP 通信按 $$m = 8$$、每 stage 8 层、SP 下每层 all-gather + reduce-scatter 各两次算；torchtitan 无 PP，一次前向就是 `local_batch_size = 2` 的整批，TP 通信按 16 层算。）跑出来的 `allocated_bytes.all.peak` 会比常驻合计大很多——差额是激活、临时 all-gather 的完整层、NCCL buffer 与 allocator 碎片，本篇只要求**常驻部分对得上**（用 step 之间的 `allocated_bytes.all.current` 读，此时激活已释放），激活与峰值的对账留给第四篇。

**Megatron 配置**（`runs/megatron/run.sh`，参数名以 0.18.0 `arguments.py` 与 `training/config/` 为准）：

```bash
#!/bin/bash
# train-ledger / runs/megatron/run.sh -- Llama-3.2-1B shape, 8 GPUs: TP2 x PP2 x DP2, ZeRO-1 (distributed optimizer)
torchrun --nproc_per_node 8 pretrain_gpt.py \
  --use-mcore-models --transformer-impl transformer_engine \
  --num-layers 16 --hidden-size 2048 --ffn-hidden-size 8192 \
  --num-attention-heads 32 --group-query-attention --num-query-groups 8 \
  --seq-length 8192 --max-position-embeddings 8192 --position-embedding-type rope --rotary-base 500000 \
  --swiglu --normalization RMSNorm --disable-bias-linear --untie-embeddings-and-output-weights \
  --attention-dropout 0.0 --hidden-dropout 0.0 \
  --tensor-model-parallel-size 2 --pipeline-model-parallel-size 2 --sequence-parallel \
  --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
  --micro-batch-size 1 --global-batch-size 16 \
  --bf16 --lr 3e-4 --min-lr 3e-5 --lr-decay-style cosine --lr-warmup-iters 10 --train-iters 50 \
  --clip-grad 1.0 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 \
  --mock-data --tokenizer-type NullTokenizer --vocab-size 128256 --split 99,1,0 \
  --log-interval 1 --log-memory-to-tensorboard --tensorboard-dir "$PWD/tb/megatron" \
  --use-pytorch-profiler --profile-step-start 20 --profile-step-end 22 --profile-ranks 0 1 2 3 4 5 6 7 \
  --eval-iters 0 --save-interval 1000000 \
  --distributed-timeout-minutes 30
```

`--global-batch-size 16` 与 `--micro-batch-size 1`、DP 2 意味着每个 DP 副本每 step 8 个 micro-batch（$$m = 8$$，PP 2 的气泡 $$1/(8+1) = 11\%$$）。`--log-memory-to-tensorboard` 让 `training_log()` 每步写 `mem-reserved-bytes`、`mem-allocated-bytes`、`mem-max-allocated-bytes`；`--use-pytorch-profiler` 让 `train()` 在第 20–22 步用 `torch.profiler.profile` 记录，trace 写到 tensorboard 目录。要用 `--use-megatron-fsdp --data-parallel-sharding-strategy optim_grads_params` 换成 ZeRO-3 对比，只需去掉 PP 并改 `--ckpt-format fsdp_dtensor`。

**DeepSpeed 配置**（`runs/deepspeed/ds_config.json`，键名以 0.19.2 `runtime/constants.py` 与 `zero/config.py` 为准）：

```json
{
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 2,
  "gradient_clipping": 1.0,
  "steps_per_print": 1,
  "bf16": { "enabled": true },
  "optimizer": {
    "type": "AdamW",
    "params": { "lr": 3e-4, "betas": [0.9, 0.95], "weight_decay": 0.1 }
  },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e7,
    "stage3_prefetch_bucket_size": 5e7,
    "stage3_param_persistence_threshold": 1e5,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "sub_group_size": 1e9
  },
  "wall_clock_breakdown": true,
  "memory_breakdown": false,
  "flops_profiler": { "enabled": true, "profile_step": 20, "module_depth": -1, "top_modules": 3, "detailed": false }
}
```

`train_batch_size` 省略，由 `1 × 2 × 8 = 16` 推出。`train_ds.py` 的骨架：

```python
"""train-ledger / runs/deepspeed/train_ds.py -- ZeRO-3 on any plain-PyTorch Llama implementation."""
import deepspeed, torch
from probe_memory import MemoryProbe

def main():
    deepspeed.init_distributed()
    cfg = "runs/deepspeed/ds_config.json"
    # zero.Init: parameters are partitioned (ds_tensor) as soon as each module's __init__ returns,
    # so the full 1B model is never materialised on one GPU.
    with deepspeed.zero.Init(config_dict_or_path=cfg, dtype=torch.bfloat16):
        model = build_llama_1b()            # any plain nn.Module Llama (16 layers, h=2048, 32/8 heads, ffn 8192)
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=cfg)
    probe = MemoryProbe("deepspeed", profile_steps=range(20, 22))
    for step, batch in enumerate(mock_batches(seq_len=8192, vocab=128256, device=engine.device)):
        with probe.step(step):
            for micro in range(engine.gradient_accumulation_steps()):
                loss = engine(**batch[micro])
                engine.backward(loss)
                engine.step()                # only the accumulation boundary takes a real optimizer step
        if step == 50:
            break
    probe.dump("runs/deepspeed/probe.json")
```

模型实现不限（任何一份纯 PyTorch 的 Llama 都可以），这正是 DeepSpeed 的卖点；`build_llama_1b()` 与 `mock_batches()` 不在本篇给出。

**torchtitan 配置**（`runs/titan/ledger_llama3_1b.py`，v0.3.0 的配置是 Python 函数）：

```python
"""train-ledger / runs/titan/ledger_llama3_1b.py -- Llama 3.2 1B shape, 8 GPUs: FSDP 4 x TP 2.

Run from the torchtitan repo root:
    NGPU=8 MODULE=runs.titan.ledger_llama3_1b CONFIG=llama3_1b_fsdp4_tp2 ./run_train.sh
"""
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.llama3 import model_registry
from torchtitan.tools.profiler import Profiler
from torchtitan.trainer import Trainer


def llama3_1b_fsdp4_tp2() -> Trainer.Config:
    model_spec = model_registry("1B")            # 16 layers, dim 2048, 32/8 heads, ffn 8192, vocab 128256
    return Trainer.Config(
        model_spec=model_spec,
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=decoder_vocab_size(model_spec)),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        optimizer=default_adamw(lr=3e-4, weight_decay=0.1),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=10, decay_type="cosine", min_lr_factor=0.1),
        training=TrainingConfig(
            local_batch_size=2,                  # x FSDP 4 = global batch 8 sequences of 8192 tokens
            seq_len=8192,
            steps=50,
            dtype="float32",                     # sharded params stay fp32 = master copy
            mixed_precision_param="bfloat16",    # all-gather / compute in bf16
            mixed_precision_reduce="float32",    # reduce-scatter in fp32
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=4,
            tensor_parallel_degree=2,
            enable_sequence_parallel=True,
            fsdp_reshard_after_forward="default",   # no PP -> reshard after forward -> 3N
        ),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        activation_checkpoint=SelectiveAC.Config(),
        metrics=MetricsProcessor.Config(log_freq=1, enable_tensorboard=True, save_tb_folder="tb/titan"),
        profiler=Profiler.Config(enable_profiling=True, profile_freq=20, profiler_warmup=1, profiler_active=2),
    )
```

`MetricsProcessor` 每次 `log()` 都调用 `DeviceMemoryMonitor.get_peak_stats()` 输出 `max_active_gib`、`max_reserved_gib` 与 `num_alloc_retries`；`Profiler` 在第 20 步附近写 chrome trace。要对比 HSDP，加 `data_parallel_replicate_degree=2, data_parallel_shard_degree=2`。

**probe_memory.py**：三个框架都能挂的探针。Megatron 与 torchtitan 自带内存日志与 profiler，探针在那里只作为独立的对照；DeepSpeed 的脚本里它是主要的观测手段。

```python
"""train-ledger / probe_memory.py -- per-rank memory + collective-communication probe.

Reads torch.cuda.memory_stats() at step boundaries and, for a few steps, runs
torch.profiler to count NCCL kernels by collective type, so the numbers can be
compared with ledger.parallel.place() from article 2.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

# NCCL device kernels are named like ncclDevKernel_AllGather_RING_LL, ncclDevKernel_ReduceScatter_...,
# ncclDevKernel_AllReduce_..., ncclDevKernel_SendRecv, ncclDevKernel_Broadcast_...
_COLL_RE = re.compile(r"nccl(?:Dev)?Kernel_(AllGather|ReduceScatter|AllReduce|SendRecv|Broadcast|AllToAll)", re.I)
_KEYS = ("allocated_bytes.all.current", "allocated_bytes.all.peak",
         "reserved_bytes.all.current", "reserved_bytes.all.peak",
         "active_bytes.all.peak", "num_alloc_retries", "num_ooms")


class MemoryProbe:
    def __init__(self, tag: str, profile_steps=range(0)):
        self.tag = tag
        self.rank = int(os.environ.get("RANK", "0"))
        self.profile_steps = set(profile_steps)
        self.records: list[dict] = []
        self.comm: dict[int, dict[str, dict]] = {}

    @contextlib.contextmanager
    def step(self, step: int):
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()
        torch.cuda.reset_peak_memory_stats()
        if step in self.profile_steps:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                yield
            torch.cuda.synchronize()
            self.comm[step] = self._summarise(prof)
        else:
            yield
            torch.cuda.synchronize()
        after = torch.cuda.memory_stats()
        rec = {"step": step, "rank": self.rank}
        for k in _KEYS:
            rec[k] = after.get(k, -1)
        # "resident" = allocated at the step boundary, when activations are gone but
        # params / grads / optimizer states are still there
        rec["resident_before"] = before.get("allocated_bytes.all.current", -1)
        self.records.append(rec)

    @staticmethod
    def _summarise(prof) -> dict[str, dict]:
        out: dict[str, dict] = defaultdict(lambda: {"count": 0, "cuda_us": 0.0})
        for evt in prof.key_averages():
            m = _COLL_RE.search(evt.key)
            if m and evt.device_time_total > 0:
                kind = m.group(1).lower()
                out[kind]["count"] += evt.count
                out[kind]["cuda_us"] += float(evt.device_time_total)
        return dict(out)

    def dump(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path.replace(".json", f".rank{self.rank}.json"), "w") as f:
            json.dump({"tag": self.tag, "records": self.records, "comm": self.comm}, f, indent=1)


def compare_with_ledger(probe_json: str, model, cfg, precision="bf16", optimizer="adam") -> None:
    """Print ledger prediction next to measured resident bytes and collective counts."""
    from ledger.memory import state_bytes
    from ledger.parallel import fmt, place

    data = json.load(open(probe_json))
    st = state_bytes(model, precision, optimizer)
    pl = place(model, st, cfg)
    g = pl.per_gpu
    predicted = g.params_bytes + g.grads_bytes + g.optim_bytes
    steady = [r for r in data["records"] if r["step"] >= 5]
    measured = min(r["resident_before"] for r in steady) if steady else -1
    print(f"[{data['tag']}] resident: ledger {fmt(predicted)}   measured {fmt(measured)}   "
          f"peak allocated {fmt(max(r['allocated_bytes.all.peak'] for r in steady))}")
    for step, comm in data["comm"].items():
        print(f"  step {step} collectives:", {k: v["count"] for k, v in comm.items()})
    print("  ledger per-step comm by kind:", {k: fmt(v) for k, v in pl.by_kind().items()})
```

对账的读法。**常驻**：取稳态各步 `resident_before` 的最小值（step 边界、激活已释放）与 ledger 的 `per_gpu` 三项之和比较。Megatron 应当接近 4.5 GB 加上 fp32 梯度 buffer 未清零的部分（`zero_grad_buffer()` 只置零不释放，所以它一直在）；DeepSpeed 与 torchtitan 应当接近 3 GB，若 DeepSpeed 明显更高，看 `stage3_param_persistence_threshold` 下有多少参数被标成 `ds_persist`。**通信次数**：ledger 不给次数只给字节，但次数能验证结构——Megatron 一个 step 应看到 AllGather 与 ReduceScatter 各约等于 bucket group 数（$$N_{\text{local}}$$ 除以 bucket 大小）、加上 TP 的每层 4 次 AllGather / ReduceScatter（SP 下）乘以 $$L_s \cdot m$$、加上 PP 的 SendRecv；DeepSpeed Stage 3 应看到每层前向一次、反向一次 AllGather 与一次 ReduceScatter（合并后按 module 计）；torchtitan 应看到每个 `fully_shard` 单元（16 层 + embedding + norm/lm_head）前向、反向各一次 AllGather 与一次 ReduceScatter，加上 TP 的通信。**通信时间**：`cuda_us` 之和除以 step 时间给出通信占比的上界（未重叠部分要看时间线，第四篇），三个框架里 torchtitan 与 Megatron 的 AllGather 应大部分被计算掩盖，DeepSpeed 的取决于 `stage3_prefetch_bucket_size` 是否足够。

三份 probe 放在一起，第二篇那张五元组表的每一行都有了一个可以指认的 kernel 名和一个可以对照的字节数。下一篇把这些数字变成时间：给定模型与集群，如何推出 TP / PP / DP / micro-batch 的取值，以及跑出来的 MFU 缺的那几个点去了哪里。

> **一个 70B dense 模型，1024 张 H100，序列长 8192，global batch 4M token。TP、PP、DP 各多少？micro-batch 多大？要不要重计算？预期 MFU 多少？跑出来只有 32%，缺的 10 个点去了哪里？**


## 下一篇

[千卡配置实战：并行搭配、micro-batch、激活重计算与 MFU 调优](/thousand-gpu-configuration-and-mfu-tuning.html)
