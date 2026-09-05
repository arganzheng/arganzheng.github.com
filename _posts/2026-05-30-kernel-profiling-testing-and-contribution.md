---
layout: post
title: "GPU Kernel 工程（10）：剖析、测试与贡献——把 kernel 做成产品"
subtitle: "Profiling, Testing and Contributing: Turning a Kernel into a Product"
tags: [CUDA, Triton, GPU, AI, AI-Infra]
catalog: true
---

> 本文是[《GPU Kernel 工程：从 CUDA 执行模型到 FlashAttention》](/gpu-kernel-engineering.html)系列的第 10 篇（共十篇）。上一篇：[量化与融合 kernel](/quantization-and-fused-kernels.html)

前九篇结束时，手上有一个用自己写的 kernel 跑通的 decoder layer 前向：RMSNorm、RoPE、BF16 Tensor Core GEMM、FlashAttention 前向、SiLU-mul、fused residual+RMSNorm、INT4 weight-only GEMM。它们能跑、结果和 PyTorch eager 对得上、每一个都在自己的 benchmark 里比 naive 版本快很多。

但"能跑"和"能合入"之间还有一整段工程。一个 kernel 要成为别人敢用的东西，需要回答四个问题：它到底卡在哪里（剖析）；它在所有会遇到的输入上都对（测试）；它确实比原来快、而且以后不会悄悄变慢（benchmark）；它在别人的 GPU 上、在 `torch.compile` 下、在框架的调度路径里也能工作（多架构与接入）。这一篇讲的就是这四件事，最后用一个 kernel PR 的完整流程把它们串起来。

总纲给这一篇的核心问题是：

> **Nsight Compute 报告 achieved occupancy 25%、long scoreboard stall 60%。这个 kernel 应该改什么？**

这个问题的答案不是一个动作，而是一个判断顺序。要建立这个顺序，得先知道每个指标是什么、由什么决定、和 Roofline 是什么关系。所以本文的前半是"读报告"，后半是"做产品"。

全文延续系列的方法论：**先算理论上应该多快，再测，再解释差距，再缩小差距**。剖析工具不是用来"找热点"的，而是用来解释"为什么实测离理论下界差这么多"的。没有理论下界，profiler 的每个百分比都无从判断好坏。


## 一、先算理论：剖析之前要有一个参照数

### 1. 为什么 profiler 的数字本身不说明问题

Nsight Compute 会告诉你一个 kernel 的 DRAM 吞吐是 1.6 TB/s。这是好还是坏？如果这个 kernel 是 RMSNorm，它读一遍写一遍、没有任何复用的余地，1.6 TB/s 在 A100（HBM2e 标称约 2.0 TB/s）上就是 80% 的带宽利用率，已经接近工程上能做到的上限（通常 85–92%）。如果这个 kernel 是 4096×4096 的 BF16 GEMM，1.6 TB/s 的 DRAM 吞吐说明它在疯狂地反复读同一份数据——一个 tiling 正确的 GEMM 的 DRAM 流量应该接近三个矩阵各读写一次的 96 MiB，在 0.44 ms 的理论时间里只需要 0.23 TB/s。

所以同一个数字，在两个 kernel 上一个是"做到头了"，一个是"完全错了"。区分的依据只能是**理论下界**。

本系列的 Roofline 模型（第一篇有完整讨论）用两个数描述硬件：峰值算力 $$P_{peak}$$ 与峰值带宽 $$BW$$；用一个数描述 kernel：算术强度 $$I = \text{FLOPs} / \text{bytes}$$。可达性能是

$$
P_{attainable} = \min(P_{peak},\ I \cdot BW)
$$

两条线交于 ridge point $$I^* = P_{peak} / BW$$。A100 BF16 Tensor Core 的 ridge point 是 $$312 / 2.0 \approx 156$$ FLOP/byte，H100 是 $$989 / 3.35 \approx 295$$。$$I$$ 远小于 $$I^*$$ 的 kernel 是 memory-bound，它的理论时间是 $$\text{bytes} / BW$$；$$I$$ 远大于 $$I^*$$ 的是 compute-bound，理论时间是 $$\text{FLOPs} / P_{peak}$$。

### 2. decoder layer 各 kernel 的理论下界

把第九篇组装的 decoder layer 拆开，每个 kernel 先算一遍。取 Llama-3-8B 的形状（$$d = 4096$$、$$d_{ff} = 14336$$、32 个 query 头、8 个 KV 头、$$d_{head} = 128$$），一次 prefill 处理 $$M = 8192$$ 个 token，BF16：

```text
kernel                 主要字节数（读+写）              FLOPs             算术强度      类型             A100 理论时间
RMSNorm ×2             2 × (8192×4096×2 B ×2) = 256 MiB  ~0.27 G          ~1            memory-bound     ~134 µs
QKV GEMM               A 64 MiB + W 48 MiB + C 96 MiB    2·8192·4096·6144 ≈ 412 G   ~1900   compute-bound   ~1.32 ms
RoPE（Q、K）            读写 Q 64 MiB + K 16 MiB = 160 MiB  ~0.3 G           ~2            memory-bound     ~84 µs
attention（causal）     Q/K/V/O 各 ≤ 64 MiB               ≈ 2·2·8192²·128·32/2 ≈ 550 G  高      compute-bound   ~1.76 ms
O-proj GEMM            64 + 32 + 64 MiB                  2·8192·4096² ≈ 275 G   ~1700    compute-bound   ~0.88 ms
fused add+RMSNorm      读 x、residual 写两者 = 256 MiB     ~0.3 G           ~1            memory-bound     ~134 µs
gate/up GEMM           64 + 224 + 448 MiB                2·8192·4096·28672 ≈ 1.92 T  ~2500  compute-bound   ~6.2 ms
SiLU-mul               读 448 MiB 写 224 MiB = 672 MiB     ~0.7 G           ~1            memory-bound     ~352 µs
down GEMM              224 + 112 + 64 MiB                2·8192·14336·4096 ≈ 962 G   ~2300  compute-bound   ~3.1 ms
```

GEMM 的理论时间按 312 TFLOPS 算，memory-bound kernel 按 2.0 TB/s 算，attention 的 FLOPs 已经乘了因果掩码的 1/2。这张表有两个用途：一是给后面每一个 profiler 数字一个参照——RMSNorm 实测 180 µs 就是 74% 的带宽利用率，可以接受；实测 400 µs 就一定有问题；二是告诉我们 profiler 应该先看谁：如果 GEMM 与 attention 各自达到峰值的 70% 以上，整层的时间就由它们主导，memory-bound kernel 加起来不到 1 ms，融合它们的收益上限就是这 1 ms 的一部分。

decode 阶段（$$M = 1$$ 或几十）这张表会完全翻转：所有 GEMM 的算术强度都变成 $$M$$ 量级，全部 memory-bound，时间由读权重决定，整层理论时间约为权重字节 / 带宽。那时 profiler 要回答的问题变成"GEMV 类 kernel 的带宽利用率是多少"以及"kernel 之间的空隙有多大"。同一个 layer、两种形状、两套完全不同的瓶颈，这是 Nsight Systems 与 Nsight Compute 分工的起点。


## 二、Nsight Systems：先看 kernel 之间

### 1. 分工：nsys 看时间线，ncu 看内部

两个工具回答不同的问题。**Nsight Systems（nsys）** 采样整个进程的时间线：CPU 线程在做什么、每次 CUDA API 调用何时发出、每个 kernel 何时在 GPU 上开始和结束、多个 stream 之间是否重叠、内存拷贝占多少。它的开销很低（通常几个百分点），可以跑完整的模型前向。它回答的是"时间花在哪个 kernel 上、kernel 之间有没有空隙、CPU 是不是在拖后腿"。

**Nsight Compute（ncu）** 只看单个 kernel 的内部：它把这个 kernel 重放（replay）几十次，每次采集一组硬件计数器，最后拼成一份报告。开销极大（一个 kernel 可能被重放 40 次以上），所以必须用过滤器只选一两个 kernel。它回答的是"这个 kernel 为什么慢"。

顺序永远是先 nsys 再 ncu：先确认时间确实花在 kernel 内部而不是 kernel 之间，再确认是哪个 kernel，最后才打开 ncu。用 ncu 去优化一个只占 2% 时间的 kernel，或者去优化一个其实被 CPU launch 开销卡住的 decode 循环，都是浪费。

### 2. 用 NVTX 标记 + nsys 找出最耗时的 kernel

对第九篇的 decoder layer，先在 Python 侧给每个阶段打 NVTX 标记，再只在 warmup 之后打开采集：

```python
# run_layer.py
import torch

layer = build_decoder_layer()          # 第九篇组装的 layer
x = torch.randn(8192, 4096, dtype=torch.bfloat16, device="cuda")

for _ in range(5):                     # warmup：编译、cuBLAS 句柄、allocator 预热
    layer(x)
torch.cuda.synchronize()

torch.cuda.profiler.start()            # 与 nsys --capture-range=cudaProfilerApi 配合
with torch.cuda.nvtx.range("decoder_layer"):
    with torch.cuda.nvtx.range("attn_norm"):
        h = layer.attn_norm(x)
    with torch.cuda.nvtx.range("qkv_proj"):
        qkv = layer.qkv_proj(h)
    with torch.cuda.nvtx.range("rope"):
        q, k, v = layer.apply_rope(qkv)
    with torch.cuda.nvtx.range("attention"):
        o = layer.attention(q, k, v)
    with torch.cuda.nvtx.range("o_proj_residual_norm"):
        h2, resid = layer.o_proj_add_norm(o, x)
    with torch.cuda.nvtx.range("mlp"):
        y = layer.mlp(h2)
torch.cuda.synchronize()
torch.cuda.profiler.stop()
```

采集命令：

```bash
nsys profile -t cuda,nvtx,osrt \
     --capture-range=cudaProfilerApi --capture-range-end=stop \
     -o layer --force-overwrite true \
     python run_layer.py

# 按 kernel 汇总（时间、次数、均值），不开 GUI 就能看
nsys stats --report cuda_gpu_kern_sum layer.nsys-rep
# 按 NVTX 区间汇总每个阶段内 kernel 的总时间
nsys stats --report nvtx_kern_sum layer.nsys-rep
```

`cuda_gpu_kern_sum` 输出一张按总时间排序的表，每行是一个 kernel 名（模板参数很长，用 `--format csv` 再处理更方便）。读法有三条：

- **前几行加起来占多少**。如果 GEMM 与 attention 加起来超过 85%，融合小 kernel 的收益上限就在剩下的 15% 里，先优化 GEMM 与 attention。
- **每个 kernel 的均值与理论下界的比值**。用第一节的表逐个除，比值最大（离 Roofline 最远）的就是下一步 ncu 的对象。
- **kernel 之间的空隙**。在 `nsys-ui` 里打开 `.nsys-rep`，把时间线放大到一个 NVTX 区间内，看相邻 kernel 之间是否有空白。空白通常来自：CPU 侧 Python 开销（每个算子几十微秒的 dispatch，在 decode 这种每个 kernel 只有几微秒的场景下会让 GPU 大部分时间空转）、显式或隐式的同步（`.item()`、`.cpu()`、`print(tensor)`）、分配器在 cache miss 时调用 `cudaMalloc`。这些都不是 kernel 的问题，不该用 ncu 去修，该用 CUDA graph、减少同步或者融合算子去修。

多 stream 场景（通信与计算重叠、prefill 与 decode 混排）在 nsys 时间线上表现为多行 kernel 并行；如果期望重叠却看到串行，通常是某个 API 隐式同步了默认 stream。这类问题也只有 nsys 看得到。

### 3. torch.profiler 与 nsys 的关系

`torch.profiler.profile(activities=[CPU, CUDA])` 底层用的是同一套 CUPTI 接口，它能给出每个 kernel 的时间与调用它的 Python 栈（`with_stack=True`），导出的 Chrome trace 可以在 Perfetto 里看时间线。它的优点是不需要额外工具、能把 kernel 和 PyTorch 算子对应起来；缺点是采样 CPU 侧的开销比 nsys 大、看不到 CUDA API 之外的系统事件（线程调度、页错误、NCCL 内部）。工作流上：日常用 `torch.profiler` 看"哪个算子慢"，怀疑 CPU 或系统层问题时换 nsys，确认是某个 kernel 内部的问题后换 ncu。三者的粒度从粗到细，开销从小到大。


## 三、Nsight Compute：看 kernel 内部

### 1. 命令行

假设 nsys 指出 fused add+RMSNorm 离理论下界最远（比如实测 300 µs 对比理论 134 µs）。ncu 的基本用法：

```bash
# 只采集名字匹配 rms_norm 的第一个 kernel，采全部 section
ncu --set full -k regex:rms_norm -c 1 -o rms_full python bench_rmsnorm.py

# 跳过前 10 次 launch（warmup），采第 11 次
ncu --set full -k regex:rms_norm -s 10 -c 1 -o rms_full python bench_rmsnorm.py

# 只采几个 section，快很多
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis \
    --section WarpStateStats --section Occupancy --section LaunchStats \
    -k regex:rms_norm -c 1 python bench_rmsnorm.py

# 采集源码级计数器：编译时加 -lineinfo，采集时 --import-source yes
ncu --set full --import-source yes -k regex:rms_norm -c 1 -o rms_src python bench_rmsnorm.py

# 用 NVTX 区间过滤
ncu --nvtx --nvtx-include "attn_norm/" -c 1 -o rms python run_layer.py

# 打开报告
ncu-ui rms_full.ncu-rep
```

几个细节：`--set full` 包含所有 section，一个 kernel 会被重放 40 次左右，用 `-c` 限制数量；`-k regex:` 匹配的是 demangled 之后的函数名；`-lineinfo` 只加行号映射不影响优化（与 `-G` 不同，`-G` 会关闭优化，绝对不能用来剖析性能）。`torch.utils.cpp_extension.load` 编译时通过 `extra_cuda_cflags=["-lineinfo"]` 传入。还有一个容易忽略的点：ncu 默认把 GPU 时钟锁在基频（`--clock-control base`），所以 ncu 报告里的 duration 通常比 benchmark 测的慢一些，两者不要直接比；要看真实时钟下的表现加 `--clock-control none`。

### 2. GPU Speed of Light：与 Roofline 的对应

报告第一节是 **GPU Speed Of Light Throughput**，给出两个百分比：**SOL Memory%**（`gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`，取 DRAM、L2、L1 各级中最高的那个利用率）与 **SOL Compute%**（`sm__throughput.avg.pct_of_peak_sustained_elapsed`，取 SM 内各 pipe 中最高的利用率）。这两个数就是 Roofline 图上的位置：

- **Memory 高（>70–80%）、Compute 低**：memory-bound，已接近带宽屋顶。能做的只有减少字节数——融合、重算、低精度存储。
- **Compute 高、Memory 低**：compute-bound。接下来看 Compute Workload Analysis 里到底是哪个 pipe 饱和——如果是 FMA/ALU 而不是 Tensor pipe，说明 GEMM 没有用上 Tensor Core 或者 Tensor Core 之外的工作（fragment 转换、softmax、地址计算）太多。
- **两者都低（<40–50%）**：latency-bound。硬件两种资源都没喂饱，问题在延迟没有被隐藏——这是最常见的情况，也是 Warp State Statistics 与 Occupancy 两节要回答的。

`--set full` 里还包含一张 **Roofline chart**（`SpeedOfLight_RooflineChart` section），横轴算术强度、纵轴实测 FLOP/s，把这个 kernel 画成一个点。它用的是硬件计数器测出的 FLOPs 与 DRAM 字节数，因此点的横坐标是"实际算术强度"（包含所有重复读取），与我们按最小流量算出的理论强度之差，本身就是一个诊断信息：GEMM 实际强度远低于理论强度，说明 tile 复用做得不够。

要注意 SOL 的一个陷阱：它是"峰值可持续吞吐"的百分比，分子是整个 kernel 运行期间的平均值。一个前 80% 时间满带宽、最后 20% 只有几个 block 在跑的 kernel，SOL Memory 会显示 80% 左右，但问题其实在 tail。这要结合 Launch Statistics 的 waves 一起看。

### 3. Memory Workload Analysis：访存模式对不对

这一节回答的是系列第三、四篇的问题——合并、向量化、bank conflict——在真实硬件上的答案。要看的指标：

**DRAM 吞吐（Memory Throughput，GB/s）**：直接与标称带宽比。同时看 DRAM 读写的总字节数，与理论最小字节数比：RMSNorm 一行 4096 BF16 读 8 KiB 写 8 KiB，8192 行是 128 MiB，如果 ncu 报出 DRAM 读了 200 MiB，说明有重复读——比如两遍 kernel 先算方差再归一化、而两遍之间行数据被从 L2 逐出。

**L1/L2 hit rate**：memory-bound 的流式 kernel 的 L2 hit rate 本来就该低（每个字节只用一次），高反而说明有重复读被 L2 挡住了。GEMM 的 L2 hit rate 应该很高（多个 block 共享同一行/列的 tile）。

**Sectors per request**：全局内存以 32 字节 sector 为单位传输，一个 warp 的一次加载指令是一个 request。理想 sector 数 = 一个 warp 请求的总字节数 / 32：每线程加载 4 字节（`float`）时 $$32 \times 4 / 32 = 4$$；每线程 16 字节（`float4` 或 8 个 BF16）时 16；每线程 2 字节（标量 BF16）时 2。实际值高于理想值就是未合并：8 表示线程间有 stride、32 表示每个线程访问一个独立 sector（完全散开，比如按列访问行主序矩阵）。ncu 在 "Memory Workload Analysis" 的表格里给出 `Sectors/Req`，也有单独的 **excessive sectors** 提示。对应的原始指标是 `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` 除以 `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`。

**Shared memory bank conflicts**：`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`（ld）与 `_st.sum`。shared memory 有 32 个 4 字节宽的 bank，一个 warp 的一次访问如果多个线程落到同一个 bank 的不同地址，就要串行成多次 wavefront。理想值是 0；GEMM 的 tile 加载如果没有 padding 或 swizzle，这个数字会与加载次数同量级。这一节还会给出 "shared memory wavefronts vs ideal" 的比值，直接告诉你差了几倍。

**Local memory 流量**：如果出现非零的 local load/store，说明寄存器溢出（spill）——数组被放到了 local memory，或者 `-maxrregcount` 太小。这是 GEMM 寄存器分块过大时的典型症状。

### 4. Warp State Statistics：warp 在等什么

这一节是 latency-bound kernel 的核心。每个 SM 有 4 个 warp 调度器，每个周期各挑一个"就绪"的 warp 发一条指令。一个 warp 不就绪时处于某种 **stall** 状态；ncu 统计每个 warp 平均每发一条指令要 stall 多少周期，并按原因分解。原因的读法：

```text
stall 原因               含义                                            通常对应的问题
long scoreboard         等全局/局部内存（L1TEX 路径）的数据返回             访存延迟未隐藏：ILP 不够、occupancy 不够、未向量化
short scoreboard        等 shared memory 或 SFU（超越函数）的结果          shared 访问太多、bank conflict、exp/rsqrt 密集
MIO throttle            shared/特殊指令的发射队列满                       shared 访存指令过密（GEMM 未做寄存器分块）
LG throttle             全局访存指令的发射队列满                          访存指令过多且过细（未向量化）
barrier                 在 __syncthreads() 等其他 warp                    同步太频繁、block 内负载不均
math pipe throttle      算术 pipe 满                                     compute-bound，好事（或说明用错了 pipe）
wait                    等固定延迟的依赖（上一条算术指令的结果）             指令级依赖链太长，缺 ILP
not selected            就绪但调度器这一拍选了别的 warp                    好事：说明有足够的并行
selected                正在发射                                          —
dispatch stall          调度器选中但发射失败（寄存器 bank 冲突等）         少见
no instruction          指令 cache miss 或分支后取指                       kernel 体太大、展开过度
sleeping / membar       nanosleep / 内存屏障                              同步原语
```

读这张表的原则是：**不要看绝对周期数，看比例与 SOL 的组合**。一个 SOL Memory 85% 的 kernel 里 long scoreboard 占 70% 完全正常——它就是在等内存，而且内存已经满了；同样的 stall 分布出现在 SOL Memory 30% 的 kernel 上就是问题：warp 在等内存，但内存系统并不忙，说明"在飞"的请求太少，需要更多 warp 或每个 warp 更多独立请求。

"Warp Cycles Per Issued Instruction" 是这一节的汇总数：一个 warp 平均每发一条指令要等多少周期。理想情况下 4 个调度器、每个有多个就绪 warp，这个数在 4 × 每调度器 warp 数附近；memory-bound kernel 常见几十到一百多。

### 5. Occupancy：有多少 warp 在飞

**Theoretical occupancy** 是按资源算出来的每 SM 最大驻留 warp 数除以 64（A100/H100 每 SM 最多 2048 线程 = 64 warp）。限制因素三个，取最小：

$$
\text{blocks/SM} = \min\left( \left\lfloor \frac{65536}{R_{alloc} \cdot T} \right\rfloor,\ \left\lfloor \frac{S_{max}}{S_{block}} \right\rfloor,\ 32,\ \left\lfloor \frac{2048}{T} \right\rfloor \right)
$$

其中 $$T$$ 是 block 线程数、$$R_{alloc}$$ 是按 warp 粒度向上取整到 256 的倍数后的每线程寄存器数、$$S_{block}$$ 是每 block 的 shared memory（静态加动态，还要加上每 block 1 KB 的系统保留）。ncu 在 Occupancy 一节直接列出 "Block Limit Registers / Shared Mem / Warps / SM"，最小的那个就是限制因素。`Registers Per Thread` 在 Launch Statistics 里；超过 32 就开始压缩 occupancy（$$65536 / (32 \times 2048) = 1$$，即每线程 32 个寄存器刚好允许满 occupancy）。GEMM 用 128–255 个寄存器是常态，那时 theoretical occupancy 只有 12.5–25%，这是设计取舍而不是缺陷——它靠 ILP 而不是 TLP 隐藏延迟。

**Achieved occupancy**（`sm__warps_active.avg.pct_of_peak_sustained_active`）是运行期间实际的平均驻留 warp 数。它低于 theoretical 的原因主要是 **tail effect**：grid 不够大或 block 执行时间不均，kernel 后期只有零星 block 在跑，平均值被拉低。另一个原因是 block 之间调度不均衡（`__syncthreads` 让整个 block 一起等）。achieved 与 theoretical 差距大，看 Launch Statistics 的 waves。

### 6. Launch Statistics、Compute Workload Analysis 与 Source Counters

**Launch Statistics** 列出 grid/block 尺寸、每线程寄存器、静态/动态 shared、以及 **Waves Per SM**：grid 中的 block 数除以（SM 数 × 每 SM 最大驻留 block 数）。0.4 wave 说明 GPU 一大半 SM 是空的（grid 太小，decode 阶段的 GEMM、小 batch 的 attention 常见）；1.2 wave 说明第一波满、第二波只有 20%，尾巴占了将近一半时间——这时把 tile 减小或 split-K 让 wave 数变成整数附近或远大于 1，往往比任何 kernel 内部的优化更有效。

**Compute Workload Analysis** 给每个 pipe 的利用率：FMA（浮点乘加）、ALU（整数与逻辑）、**Tensor**（mma/wgmma）、LSU（访存指令发射）、XU（超越函数、类型转换）、FP16 等。对 GEMM 与 attention，`sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` 是唯一重要的数——它就是 Tensor Core 利用率，cuBLAS 大形状 GEMM 通常在 70–90%。如果 SOL Compute 高但 Tensor pipe 低而 ALU/FMA 高，说明时间花在 Tensor Core 之外：地址计算、fragment 布局转换、softmax 的 exp、类型转换（BF16 到 FP32 再回来）。attention 里 XU pipe 高是典型：每个 $$QK^T$$ 元素一个 `exp`，在 H100 上 MUFU 吞吐（每 SM 每周期 16 次）相对 Tensor Core 已经是瓶颈之一，FlashAttention-3（Shah 等 2024）把 softmax 与 GEMM 在 warpgroup 间交错正是为了掩盖它。

**Source Counters** 需要 `-lineinfo` + `--import-source yes`。它把 stall 采样、访存 sector 数、分支发散等按 SASS 指令（并映射回 CUDA 源码行）列出来。在 `ncu-ui` 的 Source 页面按 "Warp Stall Sampling" 排序，前几行就是延迟集中的位置——通常是某个全局加载之后第一次使用它的那条指令（stall 记在使用者而不是加载者上）。这一页也会给出每行的 "L2 Theoretical Sectors Global Excessive"，直接指出哪一条加载没有合并。

### 7. 一份典型 memory-bound kernel 的报告长什么样

以 fused add+RMSNorm（8192 行 × 4096，BF16，一个 block 处理一行、256 线程、每线程 16 字节向量化加载）为例，一个已经做对了的版本在 A100 上报告的**典型形态**——不是实测，而是根据字节数、硬件参数与经验得出的、读者跑出来应该大致落在的区间：

```text
GPU Speed Of Light Throughput
  Duration                         ~150–175 µs（理论下界 134 µs；ncu 锁基频时会更长）
  Memory Throughput [%]            75–90         ← 已接近带宽屋顶
  DRAM Throughput [%]              75–90
  Compute (SM) Throughput [%]      10–25         ← 算力几乎空闲：memory-bound 的标志

Memory Workload Analysis
  DRAM 读 + 写                     ≈ 256 MiB（x、residual 各读一次；两者各写一次）
  L2 Hit Rate [%]                  30–55（流式，低是正常的；写分配会贡献一部分命中）
  Sectors/Req（global load）        16（每线程 16 B × 32 = 512 B = 16 sector，即理想值）
  Shared bank conflicts            0 或接近 0（只有 warp 归约的几十个字节走 shared）

Warp State Statistics
  Warp Cycles Per Issued Instr.    30–60
  long scoreboard                  55–75%        ← 在等内存，且内存确实忙：正常
  barrier                          5–15%         ← 两次 __syncthreads（归约前后）
  wait / not selected / 其他        剩余

Occupancy
  Theoretical                      100%（256 线程、~30 寄存器、~1 KB shared → 8 block/SM）
  Achieved                         70–90%        ← 8192 block / (108 SM × 8) ≈ 9.5 wave，tail 小

Launch Statistics
  Grid 8192 × Block 256，Registers/Thread ~28–40，Waves Per SM ≈ 9.5
```

这份报告说"没什么可改的"：带宽利用率已经在 80% 上下，stall 集中在 long scoreboard 但 DRAM 已经忙，occupancy 高。剩下 10–20% 的差距来自 DRAM 读写切换、行尾的归约同步与 kernel 启动/收尾，是工程上接受的水平。

对比一个**有问题**的版本的典型形态：同样的 kernel，如果每线程只做 2 字节标量加载、且一个 block 只有 128 线程、每 SM 驻留 block 数被 shared memory 限制在 3——报告会变成 SOL Memory 35–50%、SOL Compute 10–15%（两者都低：latency-bound）、Sectors/Req 2（合并了但每个请求只搬 64 B，LSU 指令数是向量化版本的 8 倍，LG throttle 上升）、theoretical occupancy 19%、achieved 15%、long scoreboard 70% 以上。这两份报告的 stall 分布几乎一样，结论完全相反——判断依据是 SOL 与 occupancy，而不是 stall 本身。这就是第四章决策树的起点。


## 四、从指标到优化方向

### 1. 决策树

```text
                          读 GPU Speed of Light
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
  Memory > 80%             Compute > 80%           两者都 < 40–50%
  memory-bound             compute-bound           latency-bound
        │                        │                        │
  已到带宽极限，            看 Compute Workload：       看 Occupancy 与 Warp State
  只能减少字节数：           Tensor pipe 利用率？              │
   · 融合相邻 kernel              │                 ┌────────┴─────────┐
   · 重算代替存储           高 → 已到算力极限，       occupancy 低         occupancy 够
   · 低精度存储/传输          换算法或换精度               │                    │
   · 减少重复读（tile 复用）  低 → Tensor Core 没用上   限制因素？          看主导 stall：
                             或 fragment 转换/         · 寄存器 → 减每线程    · long scoreboard 高
                             softmax/地址计算太多：      工作量、__launch_       → 访存延迟未隐藏：
                             · 换 mma/wgmma 形状        bounds__、-maxrreg-     更多 ILP（多个独立
                             · ldmatrix / TMA           count（注意 spill）      加载在飞）、向量化、
                             · 交错 softmax 与 GEMM    · shared → 减 tile、      cp.async 多级流水、
                             · 减少类型转换              swizzle 代替 padding    提高 occupancy
                                                      · grid 太小（waves<1）  · MIO / short scoreboard
                                                        → split-K、更小 tile、  → shared 访问过多或
                                                        更多 block              bank conflict：寄存器
                                                      · tail（waves 非整数）    分块、swizzle、ldmatrix
                                                        → 调 tile 让 wave 满   · barrier 高 → 同步太多：
                                                                                双缓冲减同步、warp 级
                                                                                独立工作、减小 block
                                                                              · LG throttle → 访存指令
                                                                                太碎：向量化
                                                                              · math pipe throttle
                                                                                → 其实 compute-bound
                                                                              · no instruction → 展开
                                                                                过度，减小代码体积
```

这棵树的第一层用 SOL 把 kernel 分成三类，因为三类的优化手段互斥：memory-bound 的 kernel 提高 occupancy 没有用（带宽已满）；latency-bound 的 kernel 做融合没有用（问题不是字节数）。第二层才看 stall——stall 原因只在 latency-bound 分支里有诊断价值。

### 2. 回答核心问题：occupancy 25%、long scoreboard 60%

回到总纲的问题。achieved occupancy 25% 意味着每 SM 平均只有 16 个 warp 驻留，每个调度器 4 个；long scoreboard 60% 意味着大部分时间 warp 在等全局内存返回。这两个数放在一起的直观解读是"warp 太少，等内存的时候没有别的 warp 可以切换"。但它们本身不足以决定动作，判断顺序是：

**第一步：看 SOL Memory。** 如果它已经在 80% 以上，那 25% 的 occupancy 和 60% 的 long scoreboard 都是结果而不是原因——带宽已满，warp 当然在等。这时唯一的方向是减少字节数，不要碰 occupancy。如果 SOL Memory 只有 30–40%，继续。

**第二步：找 occupancy 低的限制因素。** 看 Occupancy 一节的 "Block Limit" 三个数与 Launch Statistics 的 waves：

- **寄存器**（Registers Per Thread 高，比如 128 以上）：这是 GEMM 类 kernel 的常态。选项是减小每线程的分块（TM×TN 从 8×8 到 4×8，寄存器减半，但 shared memory 读取量增加 1.33 倍），或者用 `__launch_bounds__(threads, minBlocksPerSM)` 让编译器把寄存器压到目标值，或者 `-maxrregcount`。后两者的风险是 spill——压到 local memory 反而更慢，改完必须回到 Memory Workload Analysis 确认 local 流量为零。
- **shared memory**：减 tile、或者把 padding 换成 swizzle（padding 会让 128×32 的 BF16 tile 多占 1/16 到 1/8 的空间）。
- **grid 太小**（waves < 1 或 achieved 远低于 theoretical）：这时问题不在 kernel 体内而在并行度。decode GEMM 的 $$M$$ 只有几十行、$$N/BN$$ 只有几十个 block 是典型情况。手段是 split-K（把 K 维切成多段各自算部分和再规约，block 数乘以段数）、更小的 tile、或者 stream-K 式的持久 kernel。
- **theoretical 本身就是 25% 且是有意的**（大 tile GEMM）：那 occupancy 不是问题，跳到第三步。

**第三步：在给定 occupancy 下增加 ILP。** 不论 occupancy 能否提高，让每个 warp 自己有更多"在飞"的独立内存请求都是正确的：

- 把循环里"加载→用→加载→用"改成"先发出 4 个独立加载、再依次使用"（编译器在没有别名与循环携带依赖时会自动做，`#pragma unroll` 加 `__restrict__` 能帮它）；
- 向量化：每线程一次 16 字节，同样多的数据用 1/4 甚至 1/8 的指令数，每条指令带回更多字节；
- `cp.async` 多级流水（Ampere）或 TMA（Hopper）：让下一个 tile 的加载与当前 tile 的计算重叠，把"等内存"的时间换成"算上一块"的时间。

延迟隐藏的账是这样的：DRAM 延迟约 500–800 周期，一个 SM 要把带宽跑满需要"在飞"的字节数 ≈ 延迟 × 每 SM 带宽份额。A100 每 SM 每周期约 2.0 TB/s ÷ 108 ÷ 1.41 GHz ≈ 13 字节，乘 600 周期约 8 KB 在飞。16 个 warp、每 warp 一次 512 字节的请求只有 8 KB——刚刚够，任何一点不均衡就掉下来；每 warp 同时发 4 个请求就是 32 KB，余量充足。这就是为什么低 occupancy 下 ILP 能救回来，也是 cuBLAS/CUTLASS 在 12.5% occupancy 下仍能满带宽的原因。

**第四步：改完回到第一步。** 目标不是把 long scoreboard 压到零（memory-bound kernel 永远在等内存），而是让 SOL Memory 或 SOL Compute 之一升到 80% 以上。

所以这个问题的答案是：**先查 SOL 排除"已经到头"的情况；再查 occupancy 的限制因素，按寄存器 / shared / grid 分别处理；同时不论哪种情况都加 ILP；改一轮再测。** 单独回答"提高 occupancy"是错的——它可能撞上寄存器 spill，也可能在 SOL 已满时什么都改不了。


## 五、正确性测试

### 1. 参考实现与 tolerance

kernel 的正确性只能相对某个参考实现来定义。参考实现的原则是**慢但显然正确**：用 PyTorch eager 的组合算子（`x.float().pow(2).mean(-1)` 这种），在 FP32 下算，最后再 cast 到目标 dtype。不要用另一个 kernel（比如 vLLM 的 CUDA 版）当参考，除非它已经被充分验证并且你要测的是"与它行为一致"。

tolerance 按 dtype 定：

```text
dtype        rtol        atol           依据
FP32         1e-5 量级   1e-6 至 1e-5   FP32 尾数 24 位，累加顺序不同带来 1e-6 级差异
BF16         1.6e-2      1e-5（默认）    BF16 尾数 8 位，1 ulp = 2^-8 ≈ 3.9e-3，容 4 ulp
FP16         1e-3        1e-5           尾数 11 位
FP8 / INT4   与量化后的参考比            比"反量化后的 FP32 参考"，而不是原始 FP32
```

`torch.testing.assert_close` 的 BF16 默认值就是 rtol=1.6e-2、atol=1e-5。归约类 kernel（norm、softmax、attention）通常要把 atol 调大到 1e-2 左右——归一化后的输出量级在 1 附近，1e-5 的 atol 在 rtol 面前几乎不起作用，但对接近 0 的输出元素（相对误差无意义）需要绝对容差兜底。vLLM 的 `test_layernorm.py` 对 RMSNorm 用 `atol=1e-2, rtol=1e-2` 并注释了原因："LayerNorm operators typically have larger numerical errors than other operators because they involve reductions"。量化 kernel 的对照对象是"量化过程本身的参考实现"：把权重按同样的 scale 反量化成 FP32 再做 GEMM，与 kernel 输出比；不要与原始 FP32 权重的结果比，那个差异是量化误差而不是 kernel 错误。

tolerance 调大之前要问一句：是我的 kernel 累加顺序不同带来的合理误差，还是有 bug？一个经验判断：合理误差在所有元素上是随机分布的、量级与 ulp 相当；bug 通常集中在某些行（边界行）、某些列（最后一个不完整的 tile）或者是系统性的（全部偏小一个因子）。`(y - ref).abs().max()` 的位置比大小更有信息量。

### 2. 边界 shape、非连续输入与其他维度

一个 kernel 最容易错的地方是它"没想到会有"的输入。测试矩阵至少覆盖：

- **0 元素**：`rows = 0`。grid 为 0 的 launch 是非法的（`cudaErrorInvalidConfiguration`），host 侧必须提前返回；
- **1 行、1 列**：归约 kernel 的 warp 归约在只有一个元素时是否正确；
- **非 8/16 对齐的宽度**：769、5125 这种奇数宽度会打穿所有向量化路径——每线程 16 字节加载要求 $$d$$ 是 8 的倍数且指针 16 字节对齐，不满足时必须走标量 fallback 或者报错，不能读越界；
- **非 2 的幂**：树形归约、`blockDim` 假设为 2 的幂的代码在 $$d = 1000$$ 时出错；
- **超过 INT32 的元素总数**：$$2^{31}$$ 个元素以上，用 `int` 算偏移会溢出。BF16 下 $$2^{31}$$ 个元素是 4 GiB，测试需要大显存，通常用 `pytest.mark.skipif` 按可用显存跳过，但必须有；
- **所有支持的 dtype**：FP32、FP16、BF16 各自参数化，不支持的 dtype 要明确报错而不是静默算错；
- **非连续输入**：`x.t()`（stride 反转）、`x[:, :d]`（行 stride 大于 d 的切片）、`x[::2]`（行方向有间隔）。kernel 要么正确处理 stride，要么在 host 侧 `.contiguous()`（多一次拷贝，但正确），要么明确拒绝——三者都可以，静默算错不可以；
- **固定随机种子**：`torch.manual_seed(seed)`，否则失败无法复现；
- **多 GPU 架构**：同一份测试在 A100（sm_80）与 H100（sm_90）上都要跑，Hopper-only 路径要有 sm_80 上的 fallback 测试；
- **原地修改的语义**：如果 kernel 原地写 `residual`，测试要先跑参考实现再跑 kernel（vLLM 的测试里专门注释了这一点），并检查被修改的 tensor 也与参考一致。

### 3. vLLM tests/kernels 的组织

vLLM 的 kernel 测试在 `tests/kernels/` 下按功能分目录（`core/`、`attention/`、`quantization/`、`moe/`、`mamba/` 等），每个 `.cu` 文件对应一到几个 `test_*.py`。看 `test_layernorm.py`（v0.20.0）的参数化风格：

```python
# vllm v0.20.0: tests/kernels/core/test_layernorm.py（节选）
DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 4096]  # Arbitrary values for testing
HIDDEN_SIZES = [8, 768, 769, 5120, 5125, 8192]  # Arbitrary values for testing
ADD_RESIDUAL = [False, True] if not on_mi250 else [True]
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.accelerator.device_count() == 1 else 2)
]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("strided_input", [False, True])
@torch.inference_mode()
def test_rms_norm(default_vllm_config, num_tokens, hidden_size, add_residual,
                  dtype, seed, device, strided_input) -> None:
    set_random_seed(seed)
    torch.set_default_device(device)
    layer = RMSNorm(hidden_size).to(dtype=dtype)
    layer.weight.data.normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    last_dim = 2 * hidden_size if strided_input else hidden_size
    x = torch.randn(num_tokens, last_dim, dtype=dtype)
    x = x[..., :hidden_size]
    assert x.is_contiguous() != strided_input
    x *= scale
    residual = torch.randn_like(x) * scale if add_residual else None

    # NOTE(woosuk): The reference implementation should be executed first
    # because the custom kernel is in-place.
    ref_out = layer.forward_native(x, residual)
    out = layer(x, residual)
    ...
        torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)
    ...
        opcheck(
            torch.ops._C.rms_norm, (out, x, layer.weight.data, layer.variance_epsilon)
        )
```

值得学的几点：宽度列表刻意混入 769 与 5125 这种非对齐值；`strided_input` 用切片制造非连续输入并断言它确实非连续；`forward_native` 是 PyTorch 组合算子的参考实现；最后一行的 `opcheck` 是 `tests/kernels/utils.py` 里对 `torch.library.opcheck` 的一层薄封装——把 `torch.allclose` patch 成支持 FP8 的版本，默认跑 `test_schema`、`test_autograd_registration`、`test_faketensor`、`test_aot_dispatch_dynamic` 四项。第八章会解释这四项检查的是什么。

### 4. 完整的 pytest 测试文件

下面是给本文 RMSNorm 算子（第八章注册为 `torch.ops.my_ops.rms_norm`）的完整测试文件。它假设 `my_ops.py` 已经完成编译加载与 fake 注册（第八章给出）：

```python
# test_rms_norm.py
import pytest
import torch

import my_ops  # noqa: F401  编译加载 my_ops.cu 并注册 fake kernel

ROWS = [1, 7, 83, 4096]
DIMS = [8, 768, 769, 4096, 5125]
LAYOUTS = ["contiguous", "sliced", "transposed", "row_strided"]
SEEDS = [0]
BF16 = torch.bfloat16


def ref_rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """FP32 参考实现：慢但显然正确，最后再 cast 回输入 dtype。"""
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps) * w.float()).to(x.dtype)


def make_input(rows: int, d: int, layout: str) -> torch.Tensor:
    if layout == "contiguous":
        x = torch.randn(rows, d, dtype=BF16, device="cuda")
    elif layout == "sliced":                      # 行 stride = 2d，列 stride = 1
        x = torch.randn(rows, 2 * d, dtype=BF16, device="cuda")[:, :d]
    elif layout == "transposed":                  # 列 stride = rows，行 stride = 1
        x = torch.randn(d, rows, dtype=BF16, device="cuda").t()
    elif layout == "row_strided":                 # 隔行取
        x = torch.randn(2 * rows, d, dtype=BF16, device="cuda")[::2]
    else:
        raise ValueError(layout)
    assert x.shape == (rows, d)
    if layout != "contiguous" and rows > 1 and d > 1:
        assert not x.is_contiguous()
    return x


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("d", DIMS)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_rms_norm_matches_reference(rows, d, layout, seed):
    torch.manual_seed(seed)
    x = make_input(rows, d, layout)
    w = torch.randn(d, dtype=BF16, device="cuda") * 0.1 + 1.0
    eps = 1e-6

    ref = ref_rms_norm(x, w, eps)
    out = torch.ops.my_ops.rms_norm(x, w, eps)

    assert out.shape == ref.shape and out.dtype == BF16 and out.is_contiguous()
    # 归约类 kernel：atol 从默认 1e-5 调大到 1e-2，rtol 保持 BF16 默认 1.6e-2
    torch.testing.assert_close(out, ref, rtol=1.6e-2, atol=1e-2)


@pytest.mark.parametrize("rows", [0, 3])
@torch.inference_mode()
def test_rms_norm_zero_rows_or_zero_dim(rows):
    d = 64 if rows == 0 else 0
    x = torch.empty(rows, d, dtype=BF16, device="cuda")
    w = torch.ones(d, dtype=BF16, device="cuda")
    out = torch.ops.my_ops.rms_norm(x, w, 1e-6)
    assert out.shape == (rows, d)


def test_rms_norm_rejects_wrong_dtype():
    x = torch.randn(4, 64, dtype=torch.float32, device="cuda")
    w = torch.ones(64, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError):
        torch.ops.my_ops.rms_norm(x, w, 1e-6)


@pytest.mark.skipif(
    torch.cuda.get_device_properties(0).total_memory < 16 * 2**30,
    reason="needs >= 16 GiB to exercise > INT32 element counts",
)
@torch.inference_mode()
def test_rms_norm_more_than_int32_elements():
    rows, d = 2**20 + 1, 2048            # 2.1e9 元素，每个 tensor 约 4.3 GB
    x = torch.randn(rows, d, dtype=BF16, device="cuda")
    w = torch.ones(d, dtype=BF16, device="cuda")
    out = torch.ops.my_ops.rms_norm(x, w, 1e-6)
    # 只抽查最后几行，避免再物化一份 FP32 参考
    ref_tail = ref_rms_norm(x[-4:], w, 1e-6)
    torch.testing.assert_close(out[-4:], ref_tail, rtol=1.6e-2, atol=1e-2)
    del x, out
    torch.cuda.empty_cache()


@pytest.mark.parametrize("rows,d", [(7, 769), (83, 4096)])
@pytest.mark.parametrize("layout", ["contiguous", "sliced"])
def test_rms_norm_opcheck(rows, d, layout):
    torch.manual_seed(0)
    x = make_input(rows, d, layout)
    w = torch.ones(d, dtype=BF16, device="cuda")
    # 检查 schema、autograd 注册、fake kernel 与 torch.compile 下的一致性
    torch.library.opcheck(torch.ops.my_ops.rms_norm, (x, w, 1e-6))
```

运行 `pytest -v test_rms_norm.py`。参数化后第一个测试有 $$4 \times 5 \times 4 = 80$$ 个用例，每个几毫秒；加上边界、dtype 拒绝、大元素数与 opcheck，一分钟以内。这份文件覆盖了第 2 小节清单里除"多架构"之外的所有项——多架构靠在不同机器上跑同一份文件。


## 六、benchmark 方法

### 1. 测什么、怎么测

benchmark 的目标是给出一个**可复现、可比较**的数字。四个来源的噪声必须处理：

**warmup**：第一次调用包含 JIT 编译（Triton、`torch.compile`）、cuBLAS 句柄创建、CUDA context 初始化、分配器冷启动。至少跑 10–25 次再开始计时。

**L2 flush**：A100 的 L2 是 40 MB，H100 是 50 MB。一个 128 MiB 的 RMSNorm 输入不会被 L2 装下，但一个 decode 阶段 16 MiB 的权重可以——连续测 100 次，后 99 次都从 L2 读，测出的"带宽"会超过 HBM 峰值。本系列第二篇的脚手架 `bench(fn, warmup=10, iters=100, flush_l2=True)` 在两次计时之间对一个 128 MB 缓冲区做 `cudaMemsetAsync` 把 L2 冲掉。Python 侧 `triton.testing.do_bench` 默认做同样的事（它在每次调用前写一个 256 MB 的缓冲区）。测"热 L2"还是"冷 L2"取决于生产环境里这个 kernel 前面跑的是什么，但**要说明测的是哪种**。

**计时方式**：用 `torch.cuda.Event(enable_timing=True)` 在 GPU 时间线上打点，而不是 host 侧 `time.perf_counter()` 加 `synchronize()`——后者把 launch 开销与同步延迟算进去，对几微秒的 kernel误差是 100% 量级。vLLM 的 `benchmarks/kernels/benchmark_layernorm.py` 用的是 host 计时加 `synchronize`，配合 `--profile` 开关用 `cudaProfilerStart/Stop` 框定采集范围；它把多次迭代取平均，对几十微秒以上的 kernel 是够的，但对本文的目的我们用更细的工具。

**统计量**：报告**中位数**（p50），必要时加 p90；不要报平均值——GPU 的时钟调整、其他进程的干扰会产生长尾，平均值被它们拉偏，中位数不会。`triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")` 直接给中位数；`quantiles=[0.5, 0.9]` 返回 p50 与 p90。`torch.utils.benchmark.Timer(stmt, globals).blocked_autorange(min_run_time=1.0)` 是 PyTorch 自带的版本，返回 `Measurement` 对象，`.median` 是中位数、`.iqr` 是四分位距，它会自动调整每块的迭代数使总时间达到 `min_run_time`。

**时钟**：GPU 在轻载时会降频、重载时 boost，同一 kernel 在不同时刻测出的时间可以差 10–20%。有 root 权限时用 `nvidia-smi -lgc <MHz>` 把 SM 时钟锁在一个固定值（比如基频），测完 `nvidia-smi -rgc` 解锁；没有权限时至少保证每次比较在同一台机器、同一时段、相同的 warmup 之后进行。

### 2. 与 baseline 比、回归阈值与 sweep

**baseline** 有两种：一是"原来的实现"（PyTorch eager、`torch.compile` 生成的 kernel、或者 vLLM 现有的 CUDA kernel），用来证明新 kernel 有收益；二是"上一个版本的自己"，用来防止回归。前者放在 PR 描述里，后者放进 CI。

回归阈值一般取 **3–5%**：低于这个值分不清是回归还是噪声。单次运行的中位数本身也有 1–2% 的抖动，所以一个可靠的回归判定要**多次运行取中位数的中位数**，或者要求连续两次运行都超过阈值。CI 里的性能测试通常只对少数关键 shape 做，并把阈值放宽到 5–10%，因为 CI 机器的噪声更大。

**shape sweep**：一个 kernel 在一个 shape 上快不说明什么。至少扫三个维度：行数（1、16、256、4096、8192——覆盖 decode 与 prefill）、宽度（模型常见的 $$d$$：2048、4096、5120、8192，加一个非对齐的）、dtype。结果做成一张表，每行给出时间、带宽利用率（memory-bound kernel）或 TFLOPS（compute-bound kernel）、与 baseline 的倍数。带宽利用率那一列尤其重要——它把"快多少倍"换成了"离硬件极限还有多远"，前者会随 baseline 好坏变化，后者不会。

### 3. 完整的 benchmark 脚本

```python
# bench_rmsnorm.py
import argparse
import json
import sys

import torch
import torch.nn.functional as F
import triton.testing

import my_ops  # noqa: F401

BF16 = torch.bfloat16
PEAK_BW_TBPS = {"A100": 2.0, "H100": 3.35}   # 标称值


def theoretical_bytes(rows: int, d: int) -> int:
    # 读 x、读 w、写 y；w 只有 d 个元素，几乎可忽略
    return rows * d * 2 * 2 + d * 2


def bench_one(fn, warmup=25, rep=100):
    # do_bench 自带 L2 flush（每次调用前写 256 MB 缓冲区），返回毫秒
    p50, p90 = triton.testing.do_bench(fn, warmup=warmup, rep=rep, quantiles=[0.5, 0.9])
    return p50 * 1e3, p90 * 1e3   # 转成微秒


def run(shapes, peak_tbps, compiled_baseline):
    rows_list = []
    for rows, d in shapes:
        x = torch.randn(rows, d, dtype=BF16, device="cuda")
        w = torch.randn(d, dtype=BF16, device="cuda") * 0.1 + 1.0
        eps = 1e-6

        def eager():
            return F.rms_norm(x, (d,), w, eps)

        def ours():
            return torch.ops.my_ops.rms_norm(x, w, eps)

        baseline = eager
        if compiled_baseline:
            baseline = torch.compile(eager, fullgraph=True)
            baseline()   # 触发编译，不计入计时

        b50, b90 = bench_one(baseline)
        o50, o90 = bench_one(ours)
        nbytes = theoretical_bytes(rows, d)
        lower_bound_us = nbytes / (peak_tbps * 1e12) * 1e6
        util = lower_bound_us / o50 * 100.0
        rows_list.append(dict(rows=rows, d=d, base_p50=b50, base_p90=b90,
                              ours_p50=o50, ours_p90=o90, speedup=b50 / o50,
                              lower_us=lower_bound_us, bw_util=util))
    return rows_list


def print_table(results, baseline_name):
    hdr = (f"{'rows':>6} {'d':>6} | {baseline_name + ' p50':>12} {'p90':>8} | "
           f"{'ours p50':>9} {'p90':>8} | {'x':>5} | {'bound':>8} {'BW%':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['rows']:>6} {r['d']:>6} | {r['base_p50']:>12.1f} {r['base_p90']:>8.1f} | "
              f"{r['ours_p50']:>9.1f} {r['ours_p90']:>8.1f} | {r['speedup']:>5.2f} | "
              f"{r['lower_us']:>8.1f} {r['bw_util']:>5.1f}%")
    print("(times in µs; bound = bytes / peak HBM bandwidth; BW% = bound / ours_p50)")


def compare_with_saved(results, path, threshold_pct):
    with open(path) as f:
        saved = {(r["rows"], r["d"]): r for r in json.load(f)}
    regressed = []
    for r in results:
        key = (r["rows"], r["d"])
        if key not in saved:
            continue
        delta = (r["ours_p50"] / saved[key]["ours_p50"] - 1.0) * 100.0
        flag = "  <-- REGRESSION" if delta > threshold_pct else ""
        print(f"{key}: {saved[key]['ours_p50']:.1f} -> {r['ours_p50']:.1f} µs ({delta:+.1f}%){flag}")
        if delta > threshold_pct:
            regressed.append(key)
    return regressed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", choices=list(PEAK_BW_TBPS), default="A100")
    ap.add_argument("--compiled-baseline", action="store_true",
                    help="use torch.compile(F.rms_norm) instead of eager as baseline")
    ap.add_argument("--save", type=str, help="write results to this JSON")
    ap.add_argument("--compare", type=str, help="compare against a saved JSON")
    ap.add_argument("--threshold", type=float, default=3.0,
                    help="regression threshold in percent")
    args = ap.parse_args()

    torch.manual_seed(0)
    shapes = [(r, d) for r in (1, 16, 256, 4096, 8192) for d in (2048, 4096, 5125, 8192)]
    results = run(shapes, PEAK_BW_TBPS[args.gpu], args.compiled_baseline)
    print_table(results, "compiled" if args.compiled_baseline else "eager")

    if args.save:
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)
    if args.compare:
        bad = compare_with_saved(results, args.compare, args.threshold)
        if bad:
            sys.exit(1)
```

用法：`python bench_rmsnorm.py --save baseline.json` 记录当前版本；改 kernel 之后 `python bench_rmsnorm.py --compare baseline.json`，任何 shape 变慢超过 3% 就以非零退出码结束，可以直接接进 CI。表格里 `BW%` 这一列对 8192×4096 应该落在 75–90%；对 1×4096（decode 单行，只有 16 KiB 数据、一个 block）会只有百分之几——那不是 kernel 的问题，是 launch 开销（约 3–5 µs）主导了 16 KiB 数据 8 ns 的理论时间，正是 decode 需要 CUDA graph 与算子融合的原因。

顺带一句 `torch.utils.benchmark.Timer` 的用法，作为 `do_bench` 之外的选择：

```python
from torch.utils.benchmark import Timer
m = Timer(stmt="torch.ops.my_ops.rms_norm(x, w, 1e-6)",
          globals={"x": x, "w": w}).blocked_autorange(min_run_time=1.0)
print(m.median * 1e6, "us; iqr", m.iqr * 1e6)
```

它不做 L2 flush，适合测"热"路径；`Compare` 类可以把多个 `Measurement` 排成表。


## 七、多架构

### 1. 编译期：`__CUDA_ARCH__` 与 fatbin

同一份 `.cu` 要在 sm_80（A100）、sm_86/89（消费卡与 L4）、sm_90（H100）上工作。编译期的工具是 `__CUDA_ARCH__` 宏与多目标编译：

```cpp
__device__ __forceinline__ void load_tile_async(/* ... */) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  // Hopper：TMA / wgmma 路径
  hopper_tma_load(/* ... */);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  // Ampere：cp.async + mma.sync 路径
  ampere_cp_async_load(/* ... */);
#else
  // 更老的架构：同步加载
  generic_load(/* ... */);
#endif
}
```

`__CUDA_ARCH__` 只在 device 编译阶段有定义，host 阶段没有——所以 host 代码里不能用它判断"当前 GPU 是什么"，只能用运行时查询。nvcc 用 `-gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90` 为每个目标各编译一份 SASS 打进同一个 fatbin，最后一项额外嵌入 PTX 供更新架构 JIT。代价是**编译时间与二进制体积随目标数线性增长**：一个几千行的 CUTLASS 模板实例化在三个架构上编三遍，vLLM 的完整编译在多架构下要几十分钟以上，wheel 体积也直接受影响。PyTorch 扩展通过 `TORCH_CUDA_ARCH_LIST="8.0;9.0"` 环境变量控制，vLLM 用 `CUDA_ARCHS` CMake 变量，并用 `cuda_archs_loose_intersection` 函数对每组 kernel 取"该 kernel 支持的架构"与"用户要求的架构"的交集——比如 Marlin 只在 `8.0+PTX` 以上编译、FP8 Marlin 只对 `8.9;12.0;12.1` 编译，这样不支持某个特性的架构就不会为它付编译时间。

### 2. 运行期：按 compute capability 选择

host 侧按设备能力分派：

```cpp
#include <ATen/cuda/CUDAContext.h>

at::Tensor my_gemm(const at::Tensor& a, const at::Tensor& b) {
  const auto* props = at::cuda::getCurrentDeviceProperties();
  const int cc = props->major * 10 + props->minor;
  if (cc >= 90) {
    return my_gemm_sm90(a, b);       // wgmma + TMA
  } else if (cc >= 80) {
    return my_gemm_sm80(a, b);       // mma.sync + cp.async
  }
  TORCH_CHECK(false, "my_gemm requires compute capability >= 8.0, got ", cc);
}
```

纯 CUDA 侧对应 `cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev)`。Python 侧是 `torch.cuda.get_device_capability()` 返回 `(major, minor)`，vLLM 封装成 `current_platform.has_device_capability(80)`。两个原则：**Hopper-only 路径必须有 fallback**——要么回到 sm_80 实现，要么明确报错并在 Python 侧提前选择别的后端，不能让用户在 A100 上看到一个 `no kernel image is available` 的运行时错误；**运行时分派的粒度放在 host 函数一级**，不要在 kernel 内部用 `if (cc >= 90)` 分支——kernel 内部用 `__CUDA_ARCH__` 在编译期决定，两个 SASS 各自最优。


## 八、接入 PyTorch：TORCH_LIBRARY、fake kernel 与 opcheck

### 1. 为什么要注册成算子

第二到九篇用 `torch.utils.cpp_extension.load_inline` 把 kernel 暴露成一个普通的 Python 函数，测试与 benchmark 够用了。但一个普通函数对 PyTorch 是黑盒：`torch.compile` 遇到它会 graph break（Dynamo 不知道它对 tensor 做了什么，只能把图切开、把这个调用留给 eager）；`torch.export` 无法序列化它；autograd 不知道它有没有原地修改输入。把 kernel 注册成 **算子（operator）** 就是给 PyTorch 一份"它做了什么"的声明：一个 schema 字符串描述输入输出与可变性，一个 fake kernel 描述输出的 shape/dtype/device 如何由输入决定。有了这两样，编译器就能把它当成一个不透明但形状已知的节点放进图里。

本文只讲 kernel 开发者需要的最小集，不展开 Dispatcher 的原理。

### 2. C++ 侧：TORCH_LIBRARY + TORCH_LIBRARY_IMPL

下面是一个完整、可编译的最小示例。kernel 本身是第四篇的 RMSNorm：一个 block 处理一行，256 线程，warp shuffle 归约，BF16 输入、float 累加。为了让例子聚焦在注册机制上，加载是标量的（每线程 2 字节），第三章说过它的向量化版本长什么样。

```cpp
// my_ops.cu
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <cuda_bf16.h>

namespace {

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

template <int kThreads>
__global__ void rms_norm_kernel(const __nv_bfloat16* __restrict__ x,
                                const __nv_bfloat16* __restrict__ w,
                                __nv_bfloat16* __restrict__ y,
                                int d, long long x_row_stride, float eps) {
  static_assert(kThreads % 32 == 0, "block must be whole warps");
  constexpr int kWarps = kThreads / 32;
  __shared__ float red[kWarps];

  const long long row = blockIdx.x;
  const __nv_bfloat16* xr = x + row * x_row_stride;
  __nv_bfloat16* yr = y + row * static_cast<long long>(d);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;

  float ss = 0.f;
  for (int i = threadIdx.x; i < d; i += kThreads) {
    const float v = __bfloat162float(xr[i]);
    ss += v * v;
  }
  ss = warp_sum(ss);
  if (lane == 0) red[warp] = ss;
  __syncthreads();
  if (warp == 0) {
    float v = (lane < kWarps) ? red[lane] : 0.f;
    v = warp_sum(v);
    if (lane == 0) red[0] = v;
  }
  __syncthreads();
  const float inv = rsqrtf(red[0] / static_cast<float>(d) + eps);

  for (int i = threadIdx.x; i < d; i += kThreads) {
    const float v = __bfloat162float(xr[i]) * inv * __bfloat162float(w[i]);
    yr[i] = __float2bfloat16(v);
  }
}

// host 包装：与 schema "rms_norm(Tensor x, Tensor w, float eps) -> Tensor" 对应。
// schema 里的 float 对应 C++ 的 double，int 对应 int64_t。
at::Tensor rms_norm_cuda(const at::Tensor& x, const at::Tensor& w, double eps) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda(), "rms_norm: expected CUDA tensors");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16,
              "rms_norm: only bfloat16 is supported");
  TORCH_CHECK(x.dim() == 2, "rms_norm: x must be 2-D [rows, d]");
  TORCH_CHECK(w.dim() == 1 && w.size(0) == x.size(1), "rms_norm: w must be [d]");

  const c10::cuda::CUDAGuard guard(x.device());
  // 内层 stride 必须为 1 才能按行读；否则拷一份（正确优先于零拷贝）
  const at::Tensor xc = (x.size(1) == 0 || x.stride(1) == 1) ? x : x.contiguous();
  const at::Tensor wc = w.contiguous();
  at::Tensor y = at::empty({x.size(0), x.size(1)}, x.options());   // 总是连续

  const long long rows = x.size(0);
  const int d = static_cast<int>(x.size(1));
  if (rows == 0 || d == 0) return y;                               // grid 不能为 0

  constexpr int kThreads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  rms_norm_kernel<kThreads><<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(xc.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(wc.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
      d, static_cast<long long>(xc.stride(0)), static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

}  // namespace

// 1) 声明 schema：库名 my_ops，算子名 rms_norm，函数式（不修改任何输入）
TORCH_LIBRARY(my_ops, m) {
  m.def("rms_norm(Tensor x, Tensor w, float eps) -> Tensor");
}

// 2) 给 CUDA 这个 dispatch key 注册实现
TORCH_LIBRARY_IMPL(my_ops, CUDA, m) {
  m.impl("rms_norm", &rms_norm_cuda);
}
```

schema 字符串的写法与 ATen 的 `native_functions.yaml` 一致：`Tensor` 是只读输入；`Tensor!` 表示这个参数会被**原地修改**（vLLM 的 `rms_norm(Tensor! result, Tensor input, Tensor weight, float epsilon) -> ()` 就是 out 参数风格，输出写进 `result`、返回 `()`）；`Tensor?` 是可选；`Tensor!?` 是可选且可变；`int`/`float`/`bool`/`str` 对应 `int64_t`/`double`/`bool`/`std::string`；`int[]` 是 `IntArrayRef`；默认值写成 `int rope_dim_offset=0`。可变性标注不是文档，它决定 autograd 与 functionalization 如何处理这个算子——写错了 `opcheck` 会抓出来。

`x.data_ptr<at::BFloat16>()` 已经包含了 `storage_offset`，所以切片输入 `x[:, :d]`（stride(0) = 2d、stride(1) = 1）不需要拷贝，直接以 `x.stride(0)` 作为行步长传进去；转置输入 stride(1) 不为 1，会走 `.contiguous()`。

### 3. Python 侧：编译加载、fake kernel、opcheck

```python
# my_ops.py
import os
import torch
from torch.utils.cpp_extension import load

_this_dir = os.path.dirname(os.path.abspath(__file__))

# is_python_module=False：把 .so 作为普通共享库加载，TORCH_LIBRARY 的静态初始化
# 会把 my_ops::rms_norm 注册进 torch.ops；不生成 Python 模块。
load(
    name="my_ops",
    sources=[os.path.join(_this_dir, "my_ops.cu")],
    extra_cuda_cflags=["-O3", "-lineinfo", "--use_fast_math"],
    is_python_module=False,
    verbose=False,
)


@torch.library.register_fake("my_ops::rms_norm")
def _rms_norm_fake(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    # fake kernel 只描述输出的 metadata，不能读数据。
    # 真实实现总是返回连续输出，所以这里用 new_empty(shape)（连续），
    # 而不是 empty_like(x)——后者会保留 x 的 stride，遇到转置输入时与真实输出不一致。
    torch._check(x.dim() == 2, lambda: "rms_norm: x must be 2-D")
    torch._check(w.dim() == 1 and w.shape[0] == x.shape[1], lambda: "rms_norm: w must be [d]")
    return x.new_empty(x.shape)


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Python 侧的薄包装，供模型代码调用。"""
    return torch.ops.my_ops.rms_norm(x, w, eps)


if __name__ == "__main__":
    x = torch.randn(83, 769, dtype=torch.bfloat16, device="cuda")
    w = torch.ones(769, dtype=torch.bfloat16, device="cuda")

    # 1. opcheck：schema / autograd 注册 / fake kernel / torch.compile 一致性
    torch.library.opcheck(torch.ops.my_ops.rms_norm, (x, w, 1e-6))
    torch.library.opcheck(torch.ops.my_ops.rms_norm, (x.t().contiguous().t(), w, 1e-6))
    print("opcheck passed")

    # 2. 验证 torch.compile 下没有 graph break
    def f(x, w):
        return torch.ops.my_ops.rms_norm(x, w, 1e-6) * 2.0

    explanation = torch._dynamo.explain(f)(x, w)
    assert explanation.graph_break_count == 0, explanation.break_reasons
    cf = torch.compile(f, fullgraph=True)      # fullgraph=True：任何 graph break 直接报错
    torch.testing.assert_close(cf(x, w), f(x, w))
    print("torch.compile: no graph break")
```

`torch.library.register_fake`（旧名 `impl_abstract`，vLLM 的 `_custom_ops.py` 开头有一段兼容代码）注册的函数在 FakeTensor 与 meta tensor 上运行：输入没有数据，只有 shape/stride/dtype/device，函数用 PyTorch 操作构造出同样只有 metadata 的输出。规则是**不能访问任何数据**（`.item()`、`.tolist()`、`data_ptr()` 都不行）、输出的 metadata 必须与真实 kernel 完全一致（包括 stride——这是上面用 `new_empty` 而不是 `empty_like` 的原因）。输出形状依赖数据的算子（比如 `nonzero`）需要 `torch.library.get_ctx().new_dynamic_size()` 造一个符号尺寸，本文不涉及。

`torch.library.opcheck(op, args)` 做四项检查（v2.10.0 `torch/library.py` 的 docstring 原文概括）：

- `test_schema`：schema 与实现是否一致——声明为可变的参数确实被修改了、声明返回新 tensor 的确实返回了新 tensor 而不是输入的 view；
- `test_autograd_registration`：如果算子支持训练，autograd 公式是否通过 `register_autograd` 或 Autograd dispatch key 正确注册；本文的算子没有注册反向，在 `requires_grad=False` 的输入上这项通过；
- `test_faketensor`：fake kernel 是否存在、其输出的 metadata 是否与真实运行一致；
- `test_aot_dispatch_dynamic`：在 `torch.compile` 的 AOTAutograd 路径下（含 functionalization、动态 shape）算子的输出是否与 eager 相同。

docstring 建议"用一组有代表性的输入多次调用 opcheck"——不同 shape、不同 stride、每个支持的设备。这就是第五章测试文件里 `test_rms_norm_opcheck` 参数化 layout 的原因。

验证 `torch.compile` 不 graph break 有三种方法，示例里用了两种：`torch._dynamo.explain(f)(*args)` 返回 `graph_break_count` 与 `break_reasons`；`torch.compile(f, fullgraph=True)` 在有任何 break 时直接抛异常；第三种是环境变量 `TORCH_LOGS="graph_breaks"` 运行，日志里列出每个 break 的位置与原因。如果 fake kernel 没注册，Dynamo 会报 "missing fake kernel" 类的错误并 break——这是最常见的接入失败原因。


## 九、接入 vLLM

### 1. csrc 的组织与注册

vLLM 的 CUDA 代码在 `csrc/` 下按功能组织：顶层是 `layernorm_kernels.cu`、`activation_kernels.cu`、`pos_encoding_kernels.cu`、`cache_kernels.cu` 这些通用 kernel；`attention/` 放 PagedAttention 与 merge_attn_states；`quantization/` 按量化方法分子目录（`gptq_marlin/`、`awq/`、`fp8/`、`cutlass_w8a8/`、`machete/`）；`moe/` 放 MoE 的 align/permute 与 grouped GEMM；`cutlass_extensions/` 放对 CUTLASS 的扩展。所有对外的 host 函数在 `csrc/ops.h` 里声明：

```cpp
// vllm v0.20.0: csrc/ops.h（节选）
void rms_norm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight,
              double epsilon);

void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual,
                        torch::Tensor& weight, double epsilon);
```

注册集中在 `csrc/torch_bindings.cpp`。它用 `TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops)`——`core/registration.h` 里定义的一层宏，作用是让库名可以是一个宏（`TORCH_EXTENSION_NAME` 由构建系统定义为 `_C`）而不必是字面 token：

```cpp
// vllm v0.20.0: csrc/torch_bindings.cpp（节选）
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // ...
  // Layernorm
  // Apply Root Mean Square (RMS) Normalization to the input tensor.
  ops.def(
      "rms_norm(Tensor! result, Tensor input, Tensor weight, float epsilon) -> "
      "()");
  ops.impl("rms_norm", torch::kCUDA, &rms_norm);

  // In-place fused Add and RMS Normalization.
  ops.def(
      "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, "
      "float epsilon) -> ()");
  ops.impl("fused_add_rms_norm", torch::kCUDA, &fused_add_rms_norm);
  // ...
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
```

与第八章的两段式（`TORCH_LIBRARY` + `TORCH_LIBRARY_IMPL`）相比，这里 `ops.def` 与 `ops.impl(name, torch::kCUDA, &fn)` 写在同一个块里，效果相同。注意 vLLM 的算子几乎都是 **out 参数风格**：输出 tensor 由 Python 侧分配好传进来（`Tensor! result`），算子返回 `()`。这样做的好处是 Python 侧控制内存（可以复用 buffer、配合 CUDA graph 固定地址），且返回 `()` 的算子不需要 fake kernel——输出的 metadata 就是传入 tensor 的 metadata，`torch.compile` 自然能追踪。只有返回新 tensor 的算子（如 `awq_gemm`、`gptq_gemm`）才需要 `register_fake`。文件末尾的 `REGISTER_EXTENSION` 生成 `PyInit__C`，让 `.so` 能被 `import` 语句加载。

`vllm/_custom_ops.py` 是 Python 侧包装层。每个算子一个薄函数，加上必要的 fake 注册：

```python
# vllm v0.20.0: vllm/_custom_ops.py（节选）
# layer norm ops
def rms_norm(
    out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> None:
    torch.ops._C.rms_norm(out, input, weight, epsilon)


def fused_add_rms_norm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> None:
    # Note: this func is batch invariant
    torch.ops._C.fused_add_rms_norm(input, residual, weight, epsilon)

# ...
if hasattr(torch.ops._C, "awq_gemm"):

    @register_fake("_C::awq_gemm")
    def _awq_gemm_fake(
        input: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        split_k_iters: torch.SymInt,
    ) -> torch.Tensor:
        num_in_feats = input.size(0)
        return torch.empty(
            (split_k_iters, num_in_feats, qweight.size(1) * 8),
            dtype=input.dtype,
            device=input.device,
        ).sum(0)
```

`hasattr(torch.ops._C, ...)` 的保护是因为同一份 Python 代码要在 CUDA、ROCm、CPU 构建上运行，某些算子可能没编进去。`_awq_gemm_fake` 展示了 fake kernel 的典型写法：只用 shape 算术构造输出，`split_k_iters` 标注为 `torch.SymInt` 以支持动态 shape。

构建侧在 `CMakeLists.txt`：把新的 `.cu` 加进 `VLLM_EXT_SRC` 列表（`csrc/layernorm_kernels.cu` 就在那里），`set_gencode_flags_for_srcs` 以 `CUDA_ARCHS` 变量为它设置架构；只在部分架构可用的 kernel 单独开一个列表，用 `cuda_archs_loose_intersection(MY_ARCHS "9.0a" ...)` 对 `CUDA_ARCHS` 取交集后再决定是否加入源文件与定义宏。

### 2. Python 侧的后端选择

kernel 注册进 `torch.ops._C` 只是让它可调用；决定"什么时候调它"的是 Python 侧的选择逻辑，分散在几个层次：

- **层级别**：`vllm/model_executor/layers/layernorm.py` 的 `RMSNorm` 继承 `CustomOp`，有 `forward_native`（纯 PyTorch 参考）与 `forward_cuda`（调 `ops.rms_norm` / `ops.fused_add_rms_norm`，还包含一段按 stride 与架构决定是否走另一个 fast path 的条件）。`CustomOp` 按平台与配置分派到 `forward_cuda`、`forward_hip`、`forward_cpu` 或回退 `forward_native`；`vllm/platforms/cuda.py` 里的 `has_device_capability(80)` 之类的判断在这一层大量出现。
- **量化方法级别**：`vllm/model_executor/layers/quantization/fp8.py` 的 `Fp8LinearMethod` 在构造时根据 `cutlass_fp8_supported()`、设备能力与配置选择底层的 GEMM kernel 类（CUTLASS、Marlin 还是 PyTorch `_scaled_mm`），`apply()` 里再按 `use_marlin` 等标志调用。新的量化 GEMM kernel 接入点就在这里：实现一个 kernel 类、在选择逻辑里加一个条件。
- **attention 后端级别**：`vllm/v1/attention/backends/` 下每个文件是一个后端（`flash_attn.py`、`flashinfer.py`、`triton_attn.py`、`flex_attention.py`、`mla/`），`registry.py` 注册它们，`vllm/platforms/cuda.py` 的 `get_attn_backend_cls` 按设备能力、dtype、head size、KV cache dtype 与用户配置选择。

接入一个新 kernel 的完整路径是：`csrc/xxx.cu` 实现 → `csrc/ops.h` 声明 → `torch_bindings.cpp` 注册 → `CMakeLists.txt` 加源文件与架构 → `_custom_ops.py` 包装（返回新 tensor 的加 fake）→ 对应层或方法类里加选择分支 → `tests/kernels/` 加测试 → `benchmarks/kernels/` 加 benchmark。


## 十、一个 kernel PR 的完整流程

### 1. 先讨论，再写

在 vLLM 或 PyTorch 这种规模的项目里，一个没有事先讨论的几千行 kernel PR 很难被合入，原因不是代码质量，而是 reviewer 无法判断"这是不是项目想要的方向"。所以第一步是开 issue 或 RFC：说明动机（哪个模型、哪个 shape、当前用的是什么 kernel、离 Roofline 差多少）、方案（新 kernel 还是改现有的、支持哪些架构与 dtype）、**benchmark 计划**（对比哪个 baseline、哪些 shape、哪些 GPU）。得到 maintainer 的正面回应之后再动手，能避免最常见的浪费：写完才发现已经有一个类似 kernel、或者项目正准备用别的方案替换整个路径。

### 2. PR 里要有什么

vLLM 的 PR 模板（`.github/PULL_REQUEST_TEMPLATE.md`）只有三段：**Purpose**、**Test Plan**、**Test Result**，加一个 checklist。对 kernel PR 这三段的内容是：

- **Purpose**：链接 issue；一句话说明改了什么、为什么；
- **Test Plan**：能复制粘贴的测试命令（`pytest -v -s tests/kernels/core/test_xxx.py`）与 benchmark 命令；
- **Test Result**：测试通过的输出；**before/after 表**——多个 shape、多个 GPU（至少 A100 与 H100 各一），每行时间与加速比，最好加带宽/算力利用率列；如果 kernel 改变了数值行为（量化、不同累加顺序），还要有**模型评测**（vLLM 用 `lm_eval` 跑 GSM8K 之类的小基准，`.buildkite/lm-eval-harness/` 里有配置），证明精度没有退化。

vLLM 的贡献指南还有两条硬要求：commit 必须带 `Signed-off-by:`（DCO，`git commit -s`）；**AI 辅助声明**——如果 AI 工具提供了非平凡的帮助，PR 描述里必须声明，commit 加 `Co-authored-by:` 之类的 trailer，且提交者要对每一行负责、不能提交"纯 agent"PR。

提交前跑 `pre-commit install && pre-commit run --all-files`：vLLM 用 ruff 做 Python 格式与 lint、clang-format 做 C++/CUDA 格式、还有 mypy、markdownlint 等。PyTorch 用 `lintrunner`。CI 的 lint 失败是最不值得的一种失败。

### 3. review 关注什么

从 reviewer 的角度看一个 kernel PR，顺序大致是：

1. **该不该有这个 kernel**：能不能用 `torch.compile` 生成、能不能复用 CUTLASS/FlashInfer 已有的实现、是不是与现有 kernel 重复（vLLM 里已经有十几个 RMSNorm 变体，新加一个要有充分理由）；
2. **正确性边界**：测试是否覆盖了非对齐宽度、0 行、非连续输入、所有 dtype；原地修改的语义是否在 schema 里标了 `Tensor!`；
3. **数值**：tolerance 是否合理、有没有为了让测试通过把 atol 调到掩盖 bug 的程度；量化 kernel 的精度评测；
4. **多架构**：sm_80 上能不能编、Hopper-only 路径有没有 fallback、`cuda_archs_loose_intersection` 有没有配对；
5. **编译时间与体积**：模板实例化了多少个、能不能砍掉不常用的组合、是否应该拆成独立的 `.cu` 让并行编译生效；
6. **代码复用**：归约、向量化加载、dtype 分发这些工具函数 `csrc/` 里都有（`reduction_utils.cuh`、`dispatch_utils.h`），不要重新发明；
7. **benchmark 的可信度**：有没有 warmup、L2 flush 是否说明、是中位数还是平均值、baseline 是否公平（同 dtype、同 shape、同时钟）。

### 4. CI 的硬件矩阵

vLLM 的 CI 跑在 Buildkite 上，`.buildkite/test_areas/kernels.yaml` 把 `tests/kernels/` 拆成若干 step，每个 step 声明 `source_file_dependencies`（只有相关文件改动时才触发）与可选的 `device:`（默认队列跑在较小的 GPU 上，需要特定架构的 step 指定 `h100`、`b200` 等）；改动 `csrc/` 或 `CMakeLists.txt` 会触发全量测试（`ci_config.yaml` 的 `run_all_patterns`）。PyTorch 的 CI 用 GitHub Actions，PR 默认只跑一小部分，通过打 `ciflow/trunk`、`ciflow/inductor`、`ciflow/h100` 这类 label 触发更多矩阵。两个项目的共同点是：**多架构测试是 CI 的一部分而不是贡献者的自觉**——但 CI 的 GPU 时间昂贵，PR 描述里先给出自己在多架构上测过的证据，能显著加快 review。


## 十一、小结

这一篇把"能跑"的 kernel 变成"能合入"的 kernel 需要的工程逐项过了一遍：

- **剖析的前提是理论下界**：先算字节数与 FLOPs，profiler 的每个百分比才有参照；
- **nsys 看 kernel 之间，ncu 看 kernel 内部**；先确认瓶颈在 kernel 内部再打开 ncu；
- **SOL 决定分类**：Memory 高是 memory-bound（只能减字节）、Compute 高是 compute-bound（查 Tensor pipe）、两者都低是 latency-bound（查 occupancy 与 stall）；stall 原因只在第三类里有诊断价值；
- **occupancy 25% + long scoreboard 60%** 的判断顺序：SOL 是否已满 → occupancy 的限制因素（寄存器 / shared / grid）→ 不论如何加 ILP → 改完重测；
- **正确性**靠 FP32 参考实现、按 dtype 的 tolerance、边界 shape 与非连续输入的参数化；
- **benchmark** 靠 warmup、L2 flush、事件计时、中位数与分布、带宽利用率列、与 baseline 比的回归阈值；
- **多架构**靠 `__CUDA_ARCH__` 编译期分支、fatbin、运行时按 compute capability 分派与 Hopper-only 路径的 fallback；
- **接入 PyTorch** 靠 `TORCH_LIBRARY` schema、`register_fake` 与 `opcheck`；**接入 vLLM** 靠 `csrc/` → `ops.h` → `torch_bindings.cpp` → `CMakeLists.txt` → `_custom_ops.py` → 层/方法类里的选择逻辑；
- **PR** 靠先讨论、before/after 表、多 GPU、测试命令、精度评测、AI 辅助声明与 pre-commit。

ncu 各 section 与指标的速查表：

```text
section                      关键指标                                   含义 / 判断
GPU Speed of Light           SOL Memory%、SOL Compute%                 Roofline 位置：>80% 到顶；两者 <40–50% 为 latency-bound
                             Roofline chart                            实测强度 vs 理论强度：差距 = 重复读取
Memory Workload Analysis     DRAM Throughput、DRAM 总字节               与标称带宽、与理论最小字节数比
                             L1/L2 Hit Rate                            流式 kernel 低正常；GEMM 应高
                             Sectors/Req                               理想 = 每线程字节数 × 32 / 32；8、32 为未合并
                             Shared bank conflicts、wavefronts/ideal   理想 0 / 1.0
                             Local load/store                          非零 = 寄存器 spill
Warp State Statistics        Warp Cycles Per Issued Instruction        越大越"闲"
                             long scoreboard                           等全局内存；SOL Memory 低时才是问题
                             short scoreboard / MIO throttle           shared 太多或 bank conflict
                             barrier                                   同步太多
                             LG throttle                               访存指令太碎，向量化
                             math pipe throttle                        算力饱和
                             not selected                              并行充足（好事）
Occupancy                    Theoretical / Achieved                    限制因素：Block Limit Registers / Shared / Warps
                             Registers Per Thread                      >32 开始压 occupancy；128+ 为 GEMM 常态
Launch Statistics            Grid、Block、Waves Per SM                 <1 grid 太小；非整数 tail 大
Compute Workload Analysis    Tensor / FMA / ALU / LSU / XU pipe        Tensor pipe 是 GEMM 的核心指标；XU 高 = exp 瓶颈
Source Counters              按 SASS 行的 stall 采样、excessive sectors  定位到源码行（需 -lineinfo + --import-source yes）
```

正确性测试清单：

```text
[ ] FP32 参考实现（组合算子，慢但显然正确）
[ ] tolerance 按 dtype：FP32 ~1e-5；BF16 rtol 1.6e-2（归约类 atol 放到 1e-2）；量化与量化参考比
[ ] 0 行、1 行、非 8/16 对齐宽度（769、5125）、非 2 的幂、> INT32 元素数
[ ] 所有支持的 dtype 参数化；不支持的 dtype 明确报错
[ ] 非连续输入：切片、转置、隔行；kernel 处理 / .contiguous() / 明确拒绝，三选一
[ ] 固定随机种子；原地算子先跑参考再跑 kernel
[ ] opcheck 在多组 shape/stride 上通过；torch.compile fullgraph=True 不 break
[ ] 多架构（至少 sm_80 与 sm_90）
```

kernel PR 清单：

```text
[ ] issue / RFC 先讨论动机、方案与 benchmark 计划
[ ] 实现 + 测试（tests/kernels/）+ benchmark（benchmarks/kernels/）
[ ] before/after 表：多 shape、多 GPU、中位数、带宽/算力利用率列
[ ] 数值有变化时附模型评测（lm_eval）
[ ] PR 描述：Purpose / Test Plan（可复制的命令）/ Test Result
[ ] DCO sign-off；AI 辅助声明与 Co-authored-by
[ ] pre-commit 全部通过；CMakeLists 架构列表与 cuda_archs_loose_intersection 配对
[ ] 编译时间与二进制体积可接受；没有重复已有 kernel
```


## 全系列总结

十篇文章各自建立了一项能力：

```text
第一篇    硬件结构与 Roofline        知道 GPU 有什么、每样资源多快；把任何 kernel 放到 Roofline 图上
第二篇    CUDA 编程模型             把线程/block/grid 映射到 SM/warp；写第一个 kernel 与 benchmark 脚手架
第三篇    访存合并与 elementwise     32 字节 sector、128 字节 cache line、向量化；memory-bound kernel 达到 80–90% 带宽
第四篇    shared memory 与 reduction warp shuffle、bank、__syncthreads 的代价；RMSNorm、softmax、online softmax
第五篇    GEMM 从 naive 到分块      tiling 如何把算术强度从 <1 提到 >100；寄存器分块；到 cuBLAS 的 70–80%
第六篇    Tensor Core、CUTLASS、CuTe mma.sync 与 wgmma、fragment 布局、cp.async/TMA 流水；BF16 GEMM 到 cuBLAS 的 80%+
第七篇    Triton                     块级抽象让编译器接管合并、shared、流水；知道它能做到哪一层、不能做到哪一层
第八篇    FlashAttention 与 Paged    online softmax 融合进 GEMM 消掉 N² 物化；分页 KV 的间接寻址
第九篇    量化与融合                 INT4/FP8 的反量化位置、fused norm/RoPE/SiLU-mul；组装成 decoder layer
第十篇    剖析、测试、贡献           ncu 决策树；tolerance 与边界；do_bench；TORCH_LIBRARY + opcheck；PR 流程
```

贯穿它们的是一条方法论：**先算理论上应该多快（字节数、FLOPs、Roofline），再测，再解释差距，再缩小差距。** 每一篇的每个 kernel 都从一个理论下界开始——RMSNorm 一行 16 KiB、GEMM 4096³ 是 0.44 ms、attention 的 HBM 流量是 $$O(N^2 d^2 / M)$$——然后才讨论实现。这个顺序不是写作上的偏好，而是工程上唯一可靠的路径：没有下界，"快了 3 倍"无法判断是做完了还是刚开始；有了下界，"离下界还差 40%"就是一个能用 profiler 分解、能用决策树处理的问题。

对照总纲"最终目标"列出的问题，现在每一条都有了对应的工具：

- **它读写多少字节、做多少 FLOP？** —— 按第一篇的方法手算：elementwise 每元素几个字节、GEMM $$2MNK$$、attention $$4N^2 d$$；算出算术强度，与 ridge point（A100 BF16 156、H100 295）比，定下 memory-bound 还是 compute-bound。
- **它理论上最快多少？实际多少？** —— 下界 = 字节数 / 带宽 或 FLOPs / 算力；实际用第二篇的 `bench` 或 `do_bench` 测中位数；两者相除就是带宽/算力利用率，memory-bound 的好 kernel 在 80–90%，GEMM 在 70–90%。
- **差距来自哪里？** —— 本篇的 ncu：SOL 分类、Memory Workload 看访存模式、Warp State 看 stall、Occupancy 看驻留、Source Counters 定位到行。
- **访存模式对不对？** —— 第三篇的合并与向量化、第四篇的 bank conflict，在 ncu 里对应 Sectors/Req 与 bank conflict 计数。
- **线程协作方式对不对？** —— 第四篇的 shuffle 与 shared 归约、第五篇的 tile 加载分工，在 ncu 里对应 barrier 与 MIO/short scoreboard stall。
- **用上 Tensor Core 了吗？用对了吗？** —— 第六篇的 mma/wgmma、fragment 布局与 ldmatrix，在 ncu 里对应 Tensor pipe 利用率与 FMA/ALU/XU 的占比。
- **用 Triton 写会怎样？** —— 第七篇：编译器接管合并、shared、软件流水与大部分 Tensor Core 指令选择；控制不了的是 fragment 级布局、warp specialization 与新硬件特性的时间差。
- **它在别的架构上会怎样？** —— 本篇第七章：`__CUDA_ARCH__`、fatbin、运行时分派与 fallback。
- **怎么证明它是对的、没变慢？** —— 本篇第五、六章：参考实现与 tolerance、边界与非连续、opcheck；warmup、L2 flush、中位数、回归阈值。

最后说明边界。本系列自始至终只讨论**单个 kernel 内部**：它如何映射到硬件、如何访存、如何计算、如何测量、如何交付。紧挨着它的几层不在范围内：框架运行时（Dispatcher 如何选到这个 kernel、Autograd 如何调用反向、Caching Allocator 如何给它分显存、Inductor 如何决定融合哪些算子）在《PyTorch 深度实践》系列；推理引擎的调度与内存管理（continuous batching、KV cache 分页、prefix caching、PD 分离、CUDA graph 的使用）属于引擎层的系列；多卡通信（NCCL、集合通信与计算的重叠、通信 kernel 本身）属于分布式的系列。这些层决定了 kernel 之外的时间花在哪里，nsys 的时间线是它们与本系列的接口：当时间线显示瓶颈在 kernel 之间而不是之内时，读者要去的是那些系列；当瓶颈确认在某个 kernel 之内时，这十篇给出了从理论下界到合入 PR 的完整路径。系列总纲与章节目录见[《GPU Kernel 工程：从 CUDA 执行模型到 FlashAttention》](/gpu-kernel-engineering.html)。
