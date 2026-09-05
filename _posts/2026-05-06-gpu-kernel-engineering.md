---
layout: post
title: GPU Kernel 工程：从 CUDA 执行模型到 FlashAttention（总纲）
subtitle: GPU Kernel Engineering, from the CUDA Execution Model to FlashAttention
tags: [CUDA, Triton, GPU, AI, AI-Infra]
catalog: true
---


## 内容简介

《GPU Kernel 工程：从 CUDA 执行模型到 FlashAttention》是一组共十篇的系列文章，面向已经理解 PyTorch 运行时、准备向下进入 GPU 执行层的工程师，系统讲解如何读懂、写出和优化运行在 GPU 上的 kernel。

它回答的问题是：

> **一个 kernel 为什么快、为什么慢，以及如何把它写到接近硬件极限？**

站在框架和推理系统的层面看，kernel 始终是一个黑盒：Profiler 告诉你"这个算子是 memory-bound 的"，论文告诉你"FlashAttention 把 HBM 流量压下去了"，推理引擎的文档告诉你"用了 PagedAttention 所以显存碎片少了"。这些结论是对的，但它们都建立在一个没有展开的前提上——**kernel 内部发生了什么**。

这个系列把黑盒打开。它会从 GPU 的硬件结构出发，建立一套用数字说话的分析方法，然后沿着 AI 负载里最重要的几类 kernel 逐个写过去：

```text
elementwise    → 访存合并与向量化，memory-bound 的极限是什么
reduction      → 共享内存与 warp 协作，softmax 与 LayerNorm 怎么写
GEMM           → 分块、寄存器、双缓冲，compute-bound 的极限是什么
Tensor Core    → mma 指令与 CUTLASS，硬件矩阵单元怎么用
Triton         → 同样的 kernel 用块级编程重写，编译器替你做了哪一层
Attention      → FlashAttention 与 PagedAttention 的推导和实现
量化与融合      → 低精度 GEMM、RoPE、SiLU-mul、MoE 的 kernel 层含义
```

每一个 kernel 都遵循同一套方法：**先算它理论上应该多快，再测它实际多快，再用 profiler 解释差距，再动手缩小差距**。

系列使用两种写法：CUDA C++ 和 Triton。前者是 PyTorch ATen、vLLM `csrc/`、FlashAttention、CUTLASS 的语言；后者是 `torch.compile` 生成代码和 vLLM 中大量融合算子的语言。两者会在同一组 kernel 上并行推进、互相对照。


## 为什么写这个系列？

### AI-Infra 项目里最有价值的贡献落在 kernel 层

看一下 vLLM、SGLang、FlashInfer、PyTorch 的 changelog，性能相关的重大改进大多是某个 kernel 的新实现或优化：一个新的 attention 后端、一个融合的 MoE kernel、一个 FP8 GEMM、一个更快的采样 kernel。这些改动的门槛不在框架知识，而在 GPU 编程本身。

Python 层的调度、内存管理、API 设计当然重要，但贡献者多、竞争密。能写 kernel 的人少，而需求一直在增长——每一代新硬件、每一种新量化格式、每一个新模型结构都需要新的 kernel。

### "GPU 利用率 90%"什么都不说明

`nvidia-smi` 的利用率只说明 GPU 上有 kernel 在跑，不说明它跑得好不好。一个访存模式糟糕的 kernel 可以把 SM 占满、把利用率打到 100%，同时只用到带宽的 10%。要判断一个 kernel 好不好，必须知道：

- 它读写了多少字节，理论上需要多长时间；
- 它做了多少浮点运算，理论上需要多长时间；
- 两者哪个是上界，实际时间离上界差多远；
- 差距来自访存模式、占用率、同步、还是指令发射。

这些问题的答案都在 kernel 内部，需要 Roofline 模型和 Nsight Compute 来回答。这也是本系列贯穿始终的分析框架。

### Triton 降低了门槛，但没有消灭底层

Triton 让写一个融合 kernel 的成本从几百行 CUDA 变成几十行 Python，`torch.compile` 更是完全自动。这会让人觉得手写 CUDA 已经不必要。但事实是：

- Triton 的编程模型本身就是 GPU 执行模型的抽象——不理解 warp、shared memory、coalescing，写不出快的 Triton；
- Triton 的性能上限低于精心手写的 CUDA/CUTLASS，FlashAttention-3、DeepGEMM、vLLM 的量化 GEMM 都不是 Triton 写的；
- Triton 出问题时（性能不如预期、编译失败、数值不对），排查手段是看它生成的 PTX——需要读懂底层。

所以本系列的顺序是：**先用 CUDA 把执行模型建立在直觉上，再用 Triton 看编译器把哪些事情自动化了，最后在 Attention 上两条路线对照**。读者读完后可以按场景选择工具，而不是只会其中一种。

### 现有材料的断层

- **CUDA 官方文档**（Programming Guide、Best Practices）是权威参考，但不是学习路径，也不讲 AI 负载；
- **经典 GPU 编程课程**（PMPP 教材、Udacity CS344）建立了正确的基础，但停留在 2015 年前的硬件，没有 Tensor Core、TMA、Triton；
- **FlashAttention、CUTLASS 的论文和源码**是当前实践的前沿，但假设读者已经是 GPU 编程专家；
- **博客与教程**大多讲单个 kernel 的优化技巧，缺少从硬件到 Attention 的连续路径。

本系列想填补的是从"会写 vector add"到"能读 FlashAttention 源码并写出自己的 attention kernel"之间的那段路。


## 适合哪些读者？

### 理解 PyTorch 运行时、想继续向下的工程师

你已经理解 Tensor 的 stride、Dispatcher 如何找到 CUDA kernel、Profiler 如何显示 kernel 时间线，也许还写过一个简单的 CUDA 扩展。你想知道的是：**那个 kernel 内部为什么是那个速度，如何让它更快**。

### 准备给 vLLM / SGLang / FlashInfer / PyTorch 贡献 kernel 的开发者

这些项目的 `csrc/` 目录和 Triton kernel 是本系列的源码阅读对象。读完后你应该能读懂它们的 attention、量化、MoE、norm kernel，并按同样的标准写新的。

### 做推理性能优化的工程师

你需要判断一个模型在特定硬件上的瓶颈是哪个 kernel、这个 kernel 离硬件极限有多远、值不值得为它写一个定制实现。本系列的 Roofline 方法和 Nsight Compute 读法是直接工具。

### 做硬件适配的工程师

新硬件（新一代 NVIDIA GPU、AMD、国产加速器）的适配工作，很大一部分是把已有 kernel 移植和重新调优。理解 kernel 为什么这样写，才知道换硬件后哪些假设失效。


## 系列的整体主线

十篇文章按"从硬件到应用"的顺序推进，同时也是 kernel 复杂度递增的顺序：

```text
第一篇：GPU 硬件与 Roofline —— 建立分析框架
        ↓
第二篇：CUDA 编程模型 —— 第一个 kernel 与它的测量
        ↓
第三篇：访存合并与 elementwise —— memory-bound 的极限
        ↓
第四篇：共享内存与 reduction —— softmax、LayerNorm、online softmax
        ↓
第五篇：GEMM 从 naive 到分块 —— compute-bound 的极限
        ↓
第六篇：Tensor Core 与 CUTLASS —— 硬件矩阵单元
        ↓
第七篇：Triton —— 块级编程与编译器的边界
        ↓
第八篇：Attention Kernel —— FlashAttention 与 PagedAttention
        ↓
第九篇：量化与融合 kernel —— 推理系统的其余部分
        ↓
第十篇：剖析、测试与贡献 —— Nsight Compute、正确性、接入框架
```

三条交织的线索：

```text
硬件线：SM 与 warp → 内存层次 → Tensor Core → Hopper 新特性
方法线：Roofline → 带宽测量 → 占用率 → Nsight Compute 指标 → 决策树
应用线：elementwise → norm → GEMM → attention → 量化/MoE → 一个完整的 decoder layer
```

前六篇的所有 kernel 用 CUDA 写；第七篇用 Triton 把第三到五篇重写一遍；第八、九篇两种写法并行；第十篇的方法对两者通用。


## 章节结构与分章导读

### 1. GPU 为什么这样设计：硬件结构与 Roofline

第一篇建立整个系列的分析框架。它不写任何 kernel，只回答一个问题：**在一块给定的 GPU 上，一段计算理论上最快能多快？**

这一篇会讨论：

- CPU 与 GPU 的设计目标差异：延迟优化与吞吐优化；
- SM、warp、SIMT 执行模型；一块 A100 或 H100 有多少 SM、每个 SM 有多少 warp 调度器、能同时驻留多少线程；
- 内存层次：寄存器、共享内存/L1、L2、HBM，各层的容量、带宽和延迟的数量级；
- Tensor Core 的位置和它与 CUDA Core 的关系；
- Roofline 模型：算术强度、带宽屋顶、算力屋顶、拐点；
- 用 Roofline 判断一个算子是 memory-bound 还是 compute-bound，以及它离屋顶有多远；
- 硬件代际的关键差异：Volta 引入 Tensor Core、Ampere 的异步拷贝与稀疏、Hopper 的 TMA/wgmma/线程块集群、Blackwell 的第五代 Tensor Core 与 FP4；本系列以 Ampere 为基线、标注 Hopper 特性；
- 工具链地图：nvcc、PTX、SASS、`cuobjdump`、Nsight Systems、Nsight Compute，各自看什么。

核心问题是：

> **把 A100 的 312 TFLOPS 和 2 TB/s 放在一起，得到拐点约 156 FLOP/byte。一个 BF16 的 elementwise 加法算术强度是多少？一个 4096×4096 的 GEMM 呢？这决定了后面每一篇的优化目标是什么。**

这一篇也会划清本系列的边界：在**系统**层面，一个训练或推理程序的瓶颈可能在 Python 开销、kernel launch、同步等待，也可能在 kernel 本身；本系列只关注 kernel 内部，那里只剩 memory 和 compute 两类瓶颈，但要把它们量化到字节和 FLOP。

### 2. CUDA 编程模型与第一个 kernel

第二篇讲 CUDA 编程模型本身，写出第一个 kernel，并建立**测量**的习惯。

这一篇会覆盖：

- host 与 device；`__global__`、`__device__`、`__host__`；
- kernel、grid、block、thread；`blockIdx`、`blockDim`、`threadIdx`；索引计算与边界检查；
- block 和 grid 的维度怎么选；一个 block 在硬件上如何被分成 warp；
- 设备内存：`cudaMalloc`、`cudaMemcpy`、`cudaFree`，以及为什么 PyTorch 不直接用它们（Caching Allocator 的存在理由）；
- stream 与 event；kernel launch 的异步语义；`cudaDeviceSynchronize` 的代价；
- 错误处理：`cudaGetLastError`、异步错误为什么在后面才报出来；
- 编译：nvcc 做了什么，`-arch=sm_80`、`-gencode`、fatbin 与 JIT，PTX 前向兼容；
- warp 的执行方式：分支发散、`__syncwarp`、warp 内的 lockstep 直觉与 Volta 之后独立线程调度的差别；
- 第一个 kernel：vector add；测它的带宽，与第一篇的理论带宽比，解释差距。

核心问题是：

> **vector add 的 kernel 只有五行，它跑出了理论带宽的多少？没跑满的部分去了哪里？**

这一篇结束时读者会有一个 benchmark 脚手架：warmup、多次计时、CUDA event 计时、L2 flush 的必要性，这套脚手架会用到第十篇。

### 3. 访存合并与 elementwise kernel

第三篇专注于 memory-bound kernel 的极限。AI 负载里大部分非 GEMM 算子——激活函数、残差加、dtype 转换、RoPE、掩码——都是 elementwise，把它们写到带宽极限是最基本的功课。

这一篇会覆盖：

- 内存事务：一个 warp 的 32 个线程访问内存时硬件如何合并成事务；对齐、连续、跨步访问的代价差异；
- 向量化访存：`float4`、`half2`、`__nv_bfloat162`，每线程处理多个元素；
- grid-stride loop：为什么 grid 大小不必等于元素数；
- 非连续 Tensor：stride 参数怎么传进 kernel，broadcast 怎么处理；ATen 的 TensorIterator 在 host 侧为 kernel 准备了什么；
- 读 ATen 的 elementwise 实现：`aten/src/ATen/native/cuda/CUDALoops.cuh` 里的 `vectorized_elementwise_kernel` 和 `elementwise_kernel`，它们如何在运行时选择向量化路径；
- 多 dtype 的模板化：CUDA kernel 的模板参数与 `AT_DISPATCH` 宏的配合——运行期 dtype 如何变成编译期类型；
- 融合的动机：三个 elementwise kernel 融合成一个，带宽需求怎么变；
- 占用率对 memory-bound kernel 的意义：为什么要有足够多的 warp 在飞才能隐藏访存延迟。

核心问题是：

> **一个 elementwise kernel 跑出了 90% 带宽，还有什么可优化的？答案是"没有了，要么融合，要么少做"。这是一条重要的边界。**

实践：从 naive 到向量化到 grid-stride，把 elementwise kernel 推到带宽的 90% 以上，并支持 stride 和 broadcast。

### 4. 共享内存与 reduction：softmax、LayerNorm 与 online softmax

第四篇进入需要线程协作的 kernel。reduction 是所有 norm、softmax、loss、统计类算子的核心，也是 FlashAttention 的关键部件之一。

这一篇会覆盖：

- 共享内存：声明、容量、生命周期、与 L1 的关系；
- `__syncthreads()` 与 block 内同步；
- bank conflict：什么是 bank、什么访问模式冲突、padding 与 swizzle；
- reduction 的逐步优化：从 naive 的交错寻址，到顺序寻址、循环展开、warp shuffle（`__shfl_down_sync`）、每线程处理多元素；
- warp 级原语：`__shfl_*_sync`、`__ballot_sync`、`__reduce_add_sync`（Ampere+）；cooperative groups 简介；
- 原子操作：`atomicAdd` 的代价、什么时候用、什么时候避免；
- 一行一个 block 还是一行一个 warp：按行 reduction 的 block 形状选择；
- softmax：三遍还是两遍？online softmax 的推导——一遍扫描同时维护 max 和 sum，这是 FlashAttention 的数学基础；
- LayerNorm 与 RMSNorm：Welford 算法、数值稳定性、fused residual + norm；
- 读源码：ATen 的 `Reduce.cuh` 与 `SoftMax.cu`，vLLM 的 `csrc/layernorm_kernels.cu`。

核心问题是：

> **一个 4096 维的 RMSNorm，读一次写一次，理论上是 memory-bound 的。为什么 naive 实现能慢 10 倍？shared memory 和 warp shuffle 各解决了哪部分？**

实践：写出 RMSNorm 和 softmax 两个 kernel，推到带宽极限；实现 online softmax 并验证与两遍实现的数值一致性。

### 5. GEMM：从 naive 到分块

第五篇是全系列的转折：从 memory-bound 走向 compute-bound。GEMM 是 Transformer 中计算量的绝对主体，也是唯一能真正把 Tensor Core 用起来的算子。这一篇不用 Tensor Core，用 CUDA Core 把分块的原理讲透。

这一篇会覆盖：

- GEMM 的算术强度随分块大小的变化：为什么 naive 实现是 memory-bound，分块后变成 compute-bound；
- 第一版：每线程算一个输出元素——它的访存量是多少；
- 共享内存分块：把 A 和 B 的 tile 加载到 shared memory，复用；
- 寄存器分块：每线程算一个 `TM×TN` 的小块，寄存器成为最内层缓存；
- 全局内存到共享内存的加载模式：向量化、转置、避免 bank conflict；
- 双缓冲与软件流水：加载下一个 tile 的同时计算当前 tile；Ampere 的 `cp.async` 异步拷贝；
- 边界处理：M、N、K 不是 tile 整数倍时怎么办；
- tile 大小的选择：寄存器压力、共享内存容量、占用率之间的三角关系；
- 到这里能达到 cuBLAS 的多少：CUDA Core 上大约能到 70–80%，剩下的差距来自 Tensor Core（下一篇）；
- split-K 与 stream-K：K 维很大、M×N 很小时的并行策略，为 decode 阶段的 GEMV 铺垫。

核心问题是：

> **一个 4096³ 的 FP32 GEMM，naive 实现要读多少字节？128×128 分块后要读多少？寄存器分块再压多少？每一步的算术强度是多少，对应 Roofline 上的哪个位置？**

实践：六个版本的 SGEMM，从 naive 到双缓冲，每一版给出带宽/算力利用率和 Roofline 上的位置。

### 6. Tensor Core、CUTLASS 与 CuTe

第六篇把上一篇的分块结构接到硬件矩阵单元上。Tensor Core 是 GPU 之所以能跑大模型的原因，但它的编程接口和 CUDA Core 完全不同。

这一篇会覆盖：

- Tensor Core 做什么：一条指令完成一个小矩阵的乘加（如 `16×8×16`），输入 FP16/BF16/TF32/FP8/INT8，累加 FP32；
- 三代编程接口：`wmma` API（易用但受限）、`mma.sync` PTX 指令（Ampere 主力）、`wgmma`（Hopper，warp group 级别，异步）；
- fragment 布局：一个 warp 的 32 个线程各持有矩阵的哪几个元素，为什么这个布局这么奇怪；
- `ldmatrix`：从共享内存按 fragment 布局加载；
- Hopper 的 TMA：用一条指令把多维 tile 从全局内存搬到共享内存，解放线程去做计算；
- warp specialization：生产者 warp 负责搬数据，消费者 warp 负责计算；
- CUTLASS 3.x 的分层：`device` → `kernel` → `collective` → `tiled MMA/copy` → CuTe；
- CuTe 的 `Layout` 与 `Tensor`：用代数方式描述分块和线程映射，为什么它是 FlashAttention-3 和大量新 kernel 的基础；
- 读一个 CUTLASS GEMM 实例：把它的模板参数与第五篇的 tile 概念一一对应；
- PyTorch 与 vLLM 如何使用：ATen 的 `matmul` 走 cuBLAS/cuBLASLt，vLLM 的 `csrc/quantization/cutlass_w8a8/` 用 CUTLASS 写 FP8/INT8 GEMM。

核心问题是：

> **同样是 128×128 的分块，用 CUDA Core 和用 Tensor Core 写出来的 kernel 结构差在哪里？为什么 Tensor Core 版本必须关心 fragment 布局和 `ldmatrix`？**

实践：用 `mma.sync` 写一个 BF16 GEMM，达到 cuBLAS 的 80% 以上；读 CUTLASS 的一个 GEMM 配置并用 CuTe 打印它的 layout。Hopper 部分（wgmma、TMA）以源码阅读为主，标注为需要 sm_90 硬件。

### 7. Triton：块级编程与编译器的边界

第七篇换一种写法。Triton 把"线程"从编程模型中拿掉，程序员以 block 为单位思考，编译器负责线程映射、共享内存分配、向量化和流水。这一篇用 Triton 重写第三到五篇的 kernel，看它自动化了什么、没有自动化什么。

这一篇会覆盖：

- Triton 的编程模型：`program_id`、`tl.arange`、`tl.load`/`tl.store` 与 mask、块级张量操作；
- `BLOCK_SIZE` 作为 `tl.constexpr`：与 CUDA 模板参数的对应；
- `num_warps`、`num_stages`：Triton 暴露给用户的两个硬件旋钮；
- `@triton.autotune`：配置搜索、缓存、key；
- 三个 kernel 的 Triton 版本：elementwise、softmax、matmul——逐个与 CUDA 版本对照代码量和性能；
- Triton 的 matmul 为什么能接近 cuBLAS：编译器自动做了 shared memory 分块、`mma` 指令选择、软件流水；
- 编译流水线：Python AST → Triton IR → TritonGPU IR → LLVM IR → PTX → cubin；每一层在做什么优化；
- 读 Triton 生成的 PTX 和 TTGIR：怎么看它选了什么 layout、用了几个 stage；
- `torch.compile`（Inductor）生成的 Triton kernel 长什么样，怎么把它 dump 出来读；
- vLLM 里的 Triton kernel：`fused_moe` 的 grouped GEMM、prefix-prefill attention；
- Triton 做不了或做不好的：细粒度 warp 控制、特殊指令、跨 block 通信、部分 Hopper 特性的支持进度。

核心问题是：

> **Triton 的 matmul 比手写 CUDA 少 80% 的代码，性能只差 10%。那 10% 在哪里？什么场景下这 10% 值得手写？**

实践：Triton 重写三个 kernel，建立 CUDA 与 Triton 的性能对照表；用 `TRITON_DEBUG` 环境变量看生成的中间表示。

### 8. Attention Kernel：FlashAttention 与 PagedAttention

第八篇是前七篇的汇合点。Attention 同时包含 GEMM（QK^T、PV）、reduction（softmax）和特殊的内存访问模式（KV cache），是 Transformer 推理中最重要、也最难写好的 kernel。

这一篇会覆盖：

- 标准 attention 的访存问题：`S = QK^T` 是 `N×N` 的，写出再读回是 O(N²) 的 HBM 流量；
- FlashAttention 的推导：分块 + online softmax（第四篇）+ 不物化 S；每个 tile 的计算与访存，为什么它是 IO-aware 的；
- FlashAttention-2 的改进：调整并行维度、减少非矩阵运算、warp 间分工；
- FlashAttention-3（Hopper）：wgmma、TMA、warp specialization、FP8——把第六篇的 Hopper 特性用在 attention 上；
- 反向传播的 recompute：为什么训练时不保存 S 而是重算；
- 推理的两种形态：prefill（长 Q，GEMM 主导）与 decode（Q 长度为 1，memory-bound）；
- PagedAttention：KV cache 分页后 kernel 如何通过 block table 间接寻址；vLLM `csrc/attention/` 的实现结构；
- Flash-decoding / split-KV：decode 时序列长、batch 小，怎么让足够多的 SM 有活干；
- GQA/MQA 在 kernel 层的含义：多个 Q head 共享 KV，怎么利用它减少 KV 的读取；
- 变长 batch：`cu_seqlens`、padding-free 的 attention；
- 因果掩码与 sliding window 在分块中的处理；
- Triton 版 FlashAttention 走读：与 CUDA 版本的结构对照；
- 后端生态：FlashAttention、FlashInfer、xFormers、cuDNN attention、Triton 实现，vLLM 如何选择。

核心问题是：

> **一个序列长度 4k、head dim 128 的 attention，标准实现和 FlashAttention 分别读写多少 HBM？decode 阶段每生成一个 token 要读多少 KV cache，这决定了什么？**

本篇不讨论 KV cache 的分页**管理**（block 的分配、换入换出、prefix 共享属于推理引擎的调度层），只讨论分页之后 kernel 如何**访问**它。

实践：从头写一个 FlashAttention 的前向 kernel（Triton 版完整、CUDA 版覆盖核心循环），支持因果掩码和 GQA，与 PyTorch 的 `scaled_dot_product_attention` 对照正确性和性能。

### 9. 量化与融合 kernel：推理系统的其余部分

第九篇覆盖 LLM 推理里除 attention 和标准 GEMM 之外的 kernel。它们单个不复杂，但数量多、变化快，是 vLLM 和 SGLang 里贡献最活跃的区域。

这一篇会覆盖：

- 低精度格式在 kernel 层的含义：FP16/BF16/FP8（E4M3、E5M2）/INT8/INT4 的位布局、动态范围、累加精度；
- weight-only 量化 GEMM：INT4 权重在 kernel 内反量化为 BF16 再进 Tensor Core；打包格式、scale 与 zero point 的处理；为什么 decode 阶段它是赢的（memory-bound）而 prefill 阶段可能不是；
- Marlin、AWQ、GPTQ kernel 的结构：vLLM `csrc/quantization/` 目录导读；
- FP8 GEMM：per-tensor 与 per-token/per-channel scale，CUTLASS 的 FP8 支持，DeepGEMM 的思路；
- 量化本身的 kernel：动态量化的 per-token absmax reduction + cast 融合；
- 融合 kernel 的常见模式：
  - bias + activation；
  - residual add + RMSNorm；
  - SiLU-and-mul（`silu(x[:, :d]) * x[:, d:]`）；
  - RoPE：旋转位置编码的 in-place 更新，与 KV cache 写入的融合；
  - `reshape_and_cache`：把新 token 的 KV 写进分页 cache；
- MoE 的 kernel：token 按 expert 排序（permutation）、grouped GEMM、结果 unpermute 与加权求和；vLLM 的 `fused_moe` Triton kernel 与 `csrc/moe/`；
- 采样 kernel：top-k/top-p、拒绝采样（投机解码），为什么它们也会成为瓶颈；
- 数值验证：与参考实现对比时 tolerance 怎么定，低精度下什么差异是正常的。

核心问题是：

> **一个 INT4 weight-only GEMM，decode 时比 BF16 快 3 倍，prefill 时反而慢。用 Roofline 解释这个现象。**

量化算法本身（如何校准、如何选 scale、精度损失多少）不在本篇范围；本篇只讲给定格式下 kernel 怎么写、为什么它的收益只在某个区间成立。

实践：写 fused residual+RMSNorm、SiLU-mul、RoPE 三个融合 kernel，和一个简化的 INT4 weight-only GEMM；把它们与前几篇的 attention、GEMM 组装成一个完整的 decoder layer 前向。

### 10. 剖析、测试与贡献：把 kernel 做成产品

最后一篇讲把一个"能跑"的 kernel 变成"能合入"的 kernel 需要的全部工程：读 profiler、写测试、做 benchmark、处理多架构、接入框架。

这一篇会覆盖：

- Nsight Compute 的读法：GPU Speed of Light、Memory Workload Analysis、Warp State Statistics、Occupancy、Source Counters 各节看什么；
- 关键指标的含义：achieved vs theoretical occupancy、memory throughput、L1/L2 hit rate、shared memory bank conflicts、warp stall reasons（long scoreboard、MIO throttle、barrier、not selected）；
- 一棵决策树：从指标到优化方向——占用率低→寄存器/共享内存压力；stall 在 long scoreboard→访存延迟未隐藏；带宽饱和→只能融合；算力饱和→检查 Tensor Core 利用率；
- Nsight Systems 与 Nsight Compute 的分工：前者看 kernel 之间，后者看 kernel 内部；
- 正确性测试：参考实现、tolerance 选择、边界 shape（0、1、非对齐、超大）、所有 dtype、非连续输入；
- benchmark 方法：warmup、L2 flush、`triton.testing.do_bench`、`torch.utils.benchmark`、报告分布而不是平均值；性能回归的阈值；
- 多架构：`__CUDA_ARCH__` 条件编译、fatbin、运行时按 compute capability 选择实现、Hopper-only 路径的 fallback；
- 接入 PyTorch：用 `TORCH_LIBRARY` 注册为算子、补 Meta kernel、用 `torch.library.opcheck` 验证——只讲 kernel 开发者需要的最小集，不展开 Dispatcher 的原理；
- 接入 vLLM：`csrc/` 的组织、`torch_bindings.cpp`、`_custom_ops.py`、Python 侧的后端选择逻辑；
- 一个 kernel PR 的完整流程：issue 讨论、benchmark 数据、测试覆盖、review 关注点、CI 的硬件矩阵。

核心问题是：

> **Nsight Compute 报告 achieved occupancy 25%、long scoreboard stall 60%。这个 kernel 应该改什么？**

实践：用 Nsight Compute 剖析第九篇组装的 decoder layer 里的每个 kernel，找出离 Roofline 最远的一个并优化；给它写完整的测试和 benchmark；用 `TORCH_LIBRARY` 注册并通过 `opcheck`。本篇最后给出全系列总结。


## 贯穿全系列的实践线

系列的练手项目是**一个 Transformer decoder layer 的 kernel 全集**。选它是因为它覆盖了 LLM 推理里所有主要的 kernel 类型，而且体量足够小，每个 kernel 都能单独讨论：

```text
第二篇    benchmark 脚手架            事件计时 · warmup · L2 flush
第三篇    elementwise                 residual add · 激活 · dtype cast
第四篇    RMSNorm · softmax           reduction · online softmax
第五篇    SGEMM                       六个版本，到 cuBLAS 的 70–80%
第六篇    BF16 Tensor Core GEMM       mma.sync，到 cuBLAS 的 80%+
第七篇    以上三类的 Triton 版本        性能对照表
第八篇    FlashAttention 前向          因果掩码 · GQA · Triton 完整版 + CUDA 核心循环
第九篇    RoPE · SiLU-mul · fused norm · INT4 GEMM    组装成完整 layer
第十篇    剖析 · 测试 · 注册            Nsight Compute · opcheck · TORCH_LIBRARY
```

到第九篇结束，读者手上有一个用自己写的 kernel 跑通的 decoder layer 前向，可以和 PyTorch eager 对照正确性、和 `torch.compile` 对照性能。它不是一个可用的推理引擎，但每一个 kernel 都能拿出来单独测、单独优化、单独讨论离 Roofline 有多远。

与它平行的源码阅读线：

```text
第三篇    ATen  native/cuda/CUDALoops.cuh · Loops.cuh · MemoryAccess.cuh
第四篇    ATen  native/cuda/Reduce.cuh · SoftMax.cu；vLLM  csrc/layernorm_kernels.cu
第五篇    CUTLASS  examples 的 SGEMM；PyTorch  cuBLAS 调用路径 native/cuda/Blas.cpp
第六篇    CUTLASS  include/cutlass/gemm/{device,kernel,collective}；vLLM  csrc/quantization/cutlass_w8a8/
第七篇    Triton  python/tutorials；vLLM  model_executor/layers/fused_moe/fused_moe.py；Inductor 生成的 kernel
第八篇    flash-attention  csrc/flash_attn/src；vLLM  csrc/attention/；FlashInfer  include/flashinfer/attention/
第九篇    vLLM  csrc/quantization/{gptq_marlin,awq,fp8}/ · csrc/activation_kernels.cu · csrc/pos_encoding_kernels.cu · csrc/moe/
第十篇    vLLM  csrc/torch_bindings.cpp · vllm/_custom_ops.py · tests/kernels/
```


## 阅读路径建议

### 完整学习路径

```text
1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10
```

### 只关心推理、想尽快到 attention

```text
1 → 2 → 3 → 4 → 8 → 10
```

跳过 GEMM 的三篇。attention 里的 GEMM 部分会假设读者接受 Tensor Core 是黑盒。

### 主要用 Triton，不打算手写 CUDA

```text
1 → 2 → 3 → 4 → 7 → 8 → 10
```

第二到四篇仍然必要：Triton 的性能直觉全部来自执行模型。

### 做 GEMM 与量化

```text
1 → 2 → 5 → 6 → 9 → 10
```

### 做硬件适配

```text
1 → 2 → 3 → 6 → 10
```

重点是执行模型、内存层次、Tensor Core 接口和多架构处理。


## 本系列的边界

本系列只讨论单个 kernel 内部：它如何映射到硬件、如何访存、如何计算、如何测量。以下内容与它紧邻，但不在范围内：

- **框架层的运行时机制**：Dispatcher 如何选择 kernel、Autograd 如何调用反向 kernel、Caching Allocator 如何管理显存、Inductor 如何决定融合哪些算子。本系列在需要时说明"框架在 host 侧准备了什么"，但不展开这些机制的原理。
- **系统层的性能问题**：Python 开销、kernel launch 开销、CPU-GPU 同步、数据加载、多卡通信。这些决定了 kernel 之外的时间花在哪里；本系列假设读者已经确认瓶颈在某个 kernel 内部。
- **推理引擎的调度与内存管理**：continuous batching、KV cache 的分页管理、prefix caching、PD 分离。本系列第八篇只讨论分页之后 kernel 如何访问 KV cache。
- **模型与算法**：注意力机制的设计动机、量化算法的校准方法、MoE 的路由策略。本系列把它们当作给定的数学定义，只讨论如何高效地算出来。
- **C++ 语言本身**：模板、RAII、lambda 等在 kernel 的 host 侧代码中大量出现，本系列假设读者已经掌握。


## 前置要求与说明

### 前置要求

- 理解 PyTorch 运行时的基本结构：Tensor 的 shape/stride/dtype、算子如何分发到 CUDA 实现、Profiler 显示的 kernel 时间线是什么；最好写过一个简单的 CUDA 扩展；
- 知道 memory-bound 与 compute-bound 的区别，用过 `torch.profiler` 或 Nsight Systems 看过 kernel 时间线；
- 能写 C++：模板、RAII、lambda；
- 线性代数基础：矩阵乘法、softmax 的定义；
- 一块可用的 NVIDIA GPU：**Ampere（sm_80，如 A100、RTX 30 系）或更新**；Hopper 专属内容（wgmma、TMA、FlashAttention-3）以源码阅读为主，明确标注。

不要求：

- 写过 CUDA；
- 了解 CUTLASS 或 Triton；
- 了解 FlashAttention 的原理。

### 硬件与版本基线

- 硬件：以 **A100** 为默认分析对象（80 GB HBM2e，约 2 TB/s，BF16 Tensor Core 约 312 TFLOPS dense），Hopper 特性随文标注 **H100** 数字；文中给出的带宽和算力数字为公开标称值或数量级示意，实测会因具体型号、频率和功耗设置有差异；
- 软件：CUDA 12.x、PyTorch 2.x（2.4 及之后）、Triton 3.x、CUTLASS 3.x；FlashAttention 与 vLLM 以主线为准；
- 版本敏感处随文标注：Triton 对 Hopper 特性的支持进度、`AT_DISPATCH_V2`、vLLM attention 后端的默认选择、FlashAttention-3 的接口。

### 关于 AMD 与其他硬件

本系列以 NVIDIA CUDA 为主线。ROCm/HIP 的编程模型与 CUDA 高度对应（wavefront 64 vs warp 32、LDS vs shared memory、MFMA vs mma），Triton 对 AMD 的支持也在推进；正文会在相关位置提及差异，但不展开。国产加速器不在范围内。


## 章节目录

1. [GPU 为什么这样设计：硬件结构与 Roofline](/gpu-architecture-and-roofline.html)
2. [CUDA 编程模型与第一个 kernel](/cuda-programming-model-and-first-kernel.html)
3. [访存合并与 elementwise kernel](/memory-coalescing-and-elementwise-kernels.html)
4. [共享内存与 reduction：softmax、LayerNorm 与 online softmax](/shared-memory-reduction-and-softmax.html)
5. [GEMM：从 naive 到分块](/gemm-from-naive-to-tiled.html)
6. [Tensor Core、CUTLASS 与 CuTe](/tensor-cores-cutlass-and-cute.html)
7. [Triton：块级编程与编译器的边界](/triton-block-level-programming.html)
8. [Attention Kernel：FlashAttention 与 PagedAttention](/attention-kernels-flashattention-and-pagedattention.html)
9. [量化与融合 kernel：推理系统的其余部分](/quantization-and-fused-kernels.html)
10. [剖析、测试与贡献：把 kernel 做成产品](/kernel-profiling-testing-and-contribution.html)


## 最终目标

读完这套系列之后，面对任何一个 GPU kernel——无论是自己写的、PyTorch 里的、还是 vLLM PR 里的——读者应该能够回答：

```text
它读写多少字节、做多少 FLOP？               → Roofline 上的位置
它理论上最快多少？实际多少？                 → 带宽/算力利用率
差距来自哪里？                              → Nsight Compute 的指标
访存模式对不对？                            → 合并、向量化、bank conflict
线程协作方式对不对？                        → shared memory、shuffle、同步
用上 Tensor Core 了吗？用对了吗？            → mma 指令、fragment 布局、流水
用 Triton 写会怎样？                        → 编译器能自动化到哪一层
它在别的架构上会怎样？                      → 多架构与 fallback
怎么证明它是对的、没变慢？                   → 测试、tolerance、benchmark
```

最终目标是三种能力：

1. **阅读能力**：读懂 ATen、vLLM、FlashAttention、CUTLASS 中的 kernel 源码，理解每个设计决定的原因；
2. **诊断能力**：面对一个慢的 kernel，用数字而不是猜测定位它的瓶颈；
3. **实现能力**：为一个新的算子、新的量化格式或新的硬件写出接近硬件极限的 kernel，并把它以可合入的质量交付。

这是 AI-Infra 执行平面的最底层，也是贡献者最稀缺的一层。
