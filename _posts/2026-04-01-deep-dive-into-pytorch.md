---
layout: post
title: PyTorch 深度实践：从 Tensor 到深度学习运行时（总纲）
subtitle: Deep Dive into PyTorch, from Tensor to Deep Learning Runtime
tags: [PyTorch, AI, AI-Infra]
catalog: true
---


## 内容简介

《PyTorch 深度实践：从 Tensor 到深度学习运行时》是一组共十篇的系列文章，面向有后端工程经验、尤其是 Java 背景、准备进入 AI-Infra 方向的工程师，系统梳理 PyTorch 从 Python API 到 C++、CUDA、编译器、性能分析和分布式运行时的主要机制。

这个系列不是 PyTorch API 速查表，也不是机器学习算法教程，而是试图回答一个问题：

> **PyTorch 如何把张量计算表达成可求导、可扩展、可优化、可分布式执行的深度学习系统？**

一个训练或推理调用，看起来可能只是：

```python
output = model(inputs)
loss = criterion(output, targets)
loss.backward()
optimizer.step()
```

但这几行代码背后涉及一整套运行时机制：

- Tensor 如何描述数据、布局、类型和设备；
- Autograd 如何构建和遍历动态计算图；
- `nn.Module` 如何管理参数、子模块和状态；
- Dispatcher 如何根据设备和运行时上下文选择算子实现；
- Python 如何连接到 C++、CUDA 和底层数学库；
- 编译器如何捕获动态图并生成优化后的执行代码；
- GPU 程序为什么会受到访存、同步、Kernel Launch 和数据搬运的影响；
- 多个进程如何通过通信协同完成一次训练；
- 一个新算子如何经过测试、Benchmark、构建和兼容性治理。

PyTorch 在 AI 系统中的位置可以概括为：

```text
模型与训练代码
        ↓
PyTorch 编程模型
        ↓
Tensor / Autograd / Module
        ↓
Dispatcher / ATen / Kernel
        ↓
Compiler / Runtime / Distributed
        ↓
CPU / GPU / 加速器
```

这个系列的重点不是记住更多 API，而是建立从上层模型代码一直追踪到硬件执行的能力。


## 为什么写这个系列？

### 会调用 PyTorch，不等于理解 PyTorch

很多教程可以让读者很快完成下面的代码：

```python
model = MyModel().cuda()
output = model(x.cuda())
loss = loss_fn(output, target.cuda())
loss.backward()
optimizer.step()
```

但真正进入训练框架、推理框架或算子开发之后，问题会变成：

- 为什么一个 Tensor 需要 `shape`、`stride`、`dtype` 和 `device`？
- 为什么 `view()` 有时不需要复制数据，有时却直接失败？
- `loss.backward()` 究竟是如何得到梯度的？
- 为什么 `model(x)` 不等于直接调用 `model.forward(x)`？
- `nn.Parameter` 为什么会被 Optimizer 自动发现？
- 同一个算子为什么能够同时运行在 CPU 和 CUDA 上？
- Python 调用是如何进入 C++ 和 CUDA 的？
- `torch.compile()` 为什么有时加速，有时反而变慢？
- GPU 利用率不高时，问题究竟在模型、数据、Kernel 还是同步？
- DDP 与 FSDP 分别解决什么问题？
- 一个自定义算子怎样才能真正具备生产质量？

如果只停留在 API 层，这些问题往往只能依靠搜索和经验解决。这个系列希望把这些现象还原成可以理解和验证的机制。

### PyTorch 不是一个单纯的 Python 库

PyTorch 的用户入口主要是 Python，但它的执行能力来自多个层次的协作：

```text
Python API
    ↓
Python Binding
    ↓
Operator Schema
    ↓
Dispatcher
    ↓
ATen
    ↓
CPU / CUDA / Meta Kernel
    ↓
底层数学库与硬件
```

因此，理解 PyTorch 不能只看 `torch.nn` 的 Python 代码，也不能直接跳到 CUDA Kernel。需要沿着一条连续的学习路径逐层深入：

```text
编程模型 → 数据抽象 → 自动求导 → 算子系统 → 扩展机制
    → 编译执行 → 性能工程 → 分布式运行时 → 工程治理
```

### Java 的知识可以成为脚手架，但不能成为边界

Java 背景会帮助理解很多工程概念：

- `nn.Module` 可以类比为具有层级管理能力的对象容器；
- `state_dict` 可以类比为结构化的状态快照；
- DataLoader 的 worker 可以类比为生产者—消费者系统；
- Process Group 可以类比分布式通信域；
- Dispatcher 可以帮助联想到运行时分发；
- PyTorch 的测试、构建和兼容性问题与大型 Java 框架同样重要。

但也必须明确类比的边界：

- Tensor 不是 `List<List<Float>>`；
- Autograd 不是普通事件回调；
- DDP 不是 RPC 或负载均衡；
- `torch.compile()` 不是简单的 `javac`；
- CUDA Stream 不是 Java Executor；
- checkpoint 不是普通的 Java 对象序列化。

本系列会使用 Java 作为参照系，但最终目标是建立 PyTorch 自己的运行时模型。


## 适合哪些读者？

### 从后端转向 AI-Infra 的工程师

这是本系列的主要读者。适合有 Java、Go、C++ 等后端工程经验，已经掌握基本 Python，准备参与以下工作的工程师：

- 模型训练平台；
- 推理服务；
- GPU 资源管理；
- 深度学习框架；
- 自定义算子与硬件适配；
- 分布式训练；
- AI 平台和中间件。

如果你希望理解下面这些问题，本系列会比较适合：

- Tensor 的内存布局如何影响算子性能？
- 为什么 GPU 计算经常需要显式同步？
- 为什么 DataLoader 会让 GPU“吃不饱”？
- PyTorch 如何在不同设备上找到正确的 Kernel？
- `torch.compile()` 的 graph break 是什么？
- DDP 如何同步梯度？FSDP 为什么能够节省显存？
- 一个 C++/CUDA 扩展如何安全地接入 PyTorch？
- 如何为算子设计跨设备、跨 dtype 的测试？

### 想读懂 PyTorch 和训练框架源码的开发者

适合准备阅读或贡献 PyTorch、Transformers、DeepSpeed、TorchTitan、Triton 等项目的开发者。

这些项目中的很多复杂性并不来自 Python 语法，而来自不同层次的协作：

- Tensor metadata 决定数据如何解释；
- Autograd 决定梯度如何传播；
- Dispatcher 决定算子如何分发；
- C++ 和 CUDA 决定底层执行；
- 编译器决定计算图如何变换；
- 分布式运行时决定参数和梯度如何通信。

### 已经会训练模型、但希望提升工程能力的开发者

适合已经能够使用 PyTorch 训练模型，但遇到过以下问题的读者：

- 训练速度不稳定；
- GPU 利用率很低；
- 显存逐步增长；
- 换一个 batch size 就 OOM；
- 多卡训练速度没有线性提升；
- 模型在 eager 下正常，compile 后结果异常；
- 自定义算子无法处理非连续 Tensor；
- 代码只能在当前机器和当前 CUDA 环境运行。

### 希望进入深度学习框架和系统方向的开发者

如果你希望从“使用模型”进一步走向“理解和开发模型执行系统”，这套系列可以作为从 Python 工程到深度学习运行时之间的桥梁。


## 系列的整体主线

十篇文章按照 PyTorch 的内部抽象逐步展开：

```text
第一篇：建立全局地图
        ↓
第二篇：理解 Tensor 如何表示数据
        ↓
第三篇：理解 Autograd 如何表示梯度计算
        ↓
第四篇：理解 Module 如何组织模型和训练
        ↓
第五篇：理解 Dispatcher 如何选择算子实现
        ↓
第六篇：理解 Python 如何扩展到 C++ 和 CUDA
        ↓
第七篇：理解动态图如何进入编译执行
        ↓
第八篇：理解程序为什么快或慢
        ↓
第九篇：理解多卡如何通过通信协同
        ↓
第十篇：理解一次改动如何经过测试、CI 和发布安全地到达用户
```

可以把这条主线进一步归纳为三条相互交织的线索：

```text
抽象线：Tensor → Autograd → Module → Operator → Compiler

执行线：Python → C++ → CUDA → Kernel → Hardware

工程线：Training → Profiling → Distributed → Testing → Build
```

需要说明的是，这条主线按**职责层**推进，而不是按源码目录。PyTorch 的源码按库分层是 `torch/`（Python）→ `torch/csrc/`（绑定、Autograd 引擎、c10d）→ `aten/src/ATen/`（Dispatcher、算子）→ `c10/`（TensorImpl、Device、Allocator），两种分层并不重合：第二篇讲的 Tensor 元数据在源码上住在最底层的 c10，却是用户最先接触、其余一切所依赖的抽象，所以放在最前面。第一篇会给出一张"组件 → 源码位置 → 职责层 → 展开篇"的对照表；第二到九篇的小结各附一张"本篇涉及的源码位置"表，把该篇讨论的机制落到具体文件；第十篇的仓库地图则给出完整目录。读者可以随时在职责层和源码层两个坐标系之间切换。

这不是严格的单向依赖：

- 第五篇会复用第二篇关于 stride、dtype 和 device 的知识；
- 第六篇会复用第五篇关于 Operator Schema 和 Dispatcher 的知识；
- 第七篇会同时复用 Tensor metadata 和算子系统；
- 第八篇会帮助判断第六篇的自定义 Kernel 是否真的有效；
- 第九篇会复用第八篇的性能分析方法，并把第八篇的案例模型扩到多卡；
- 第十篇会把第六篇的自定义算子重新拿出来，让它走完一次完整的工程生命周期。


## 章节结构与分章导读

### 1. PyTorch 整体介绍

第一篇从全局开始，介绍一段模型代码如何穿过 Python、C++、CUDA、Dispatcher、Compiler 和硬件执行层，帮助读者建立后续文章所需的整体地图。

这一篇会讨论：

- 深度学习框架解决什么问题；
- PyTorch 与 TensorFlow、JAX 等框架的设计差异；
- PyTorch 从动态图走向编译执行的架构演进；
- Eager Mode 的基本定位；
- Python、C++、CUDA 三层之间的关系；
- Tensor、Autograd、Dispatcher、ATen、Kernel 的关系；
- `torch.compile`、分布式和扩展机制在整体架构中的位置。

核心问题是：

> **执行 `torch.add(x, y)` 时，PyTorch 内部发生了什么？**

本篇会给出一条用于建立心智模型的典型调用路径：

```text
Python API
    ↓
Python Binding
    ↓
Operator Schema
    ↓
Dispatcher
    ↓
ATen Operator
    ↓
CPU / CUDA Kernel
    ↓
Hardware
```

这是一条概念路径，不意味着所有算子、所有版本都经过完全相同的源码调用栈。后续文章会分别展开其中的关键层次。

Eager Mode 在全系列中承担一条贯穿主线：第一篇介绍它是什么，第三篇解释它如何构建动态计算图，第七篇说明它如何进入编译执行，第八篇分析它的性能特征和代价。

### 2. Tensor 与内存布局

第二篇讨论 PyTorch 最核心的数据抽象。重点不是罗列 `torch.zeros()`、`torch.ones()` 等 API，而是理解 Tensor 如何解释一段设备上的数据。

这一篇会覆盖：

- Tensor、Storage 和数据指针；
- shape、dtype、device；
- stride；
- storage offset；
- contiguous 与 non-contiguous；
- `view`、`reshape`、`transpose`、`permute`；
- 广播与零拷贝视图；
- CPU Tensor 与 CUDA Tensor；
- dtype promotion；
- in-place 操作；
- 数据拷贝与设备迁移。

可以把一个简化版 Tensor 表示为：

```text
Tensor
├── storage
├── sizes
├── strides
├── storage_offset
├── dtype
└── device
```

这一篇会配套实现一个只支持 CPU 的简化版 Tensor，用来理解：

- shape 如何描述逻辑形状；
- stride 如何解释底层存储；
- transpose 为什么可以只改变 metadata；
- view 为什么有时不需要拷贝；
- contiguous 为什么可能触发真实数据复制。

Tensor 不只是一个多维数组，它是**数据、布局、类型和设备位置的组合**。后面讲算子、Kernel、显存和性能时，都会回到这一抽象。

### 3. 自动求导与动态计算图

第三篇从链式法则出发，解释 PyTorch 如何把数学上的梯度计算转化为可执行的反向图遍历。

这一篇会讨论：

- 链式法则；
- 标量、向量和张量求导；
- Eager Mode 下的动态计算图；
- leaf Tensor；
- `requires_grad`；
- `grad_fn`；
- Autograd Function；
- `saved_tensors`；
- 梯度累积；
- `backward()`；
- `detach()`；
- `no_grad()`；
- `inference_mode()`；
- 自定义 `autograd.Function`。

本篇会实现一个 Mini-Autograd：

```text
加法和乘法
    ↓
保存父节点
    ↓
保存局部导数
    ↓
拓扑排序
    ↓
反向传播
    ↓
梯度累积
```

核心问题不是“如何调用 `loss.backward()`”，而是：

> **PyTorch 如何把链式法则变成一次沿计算图执行的反向传播？**

同时，本篇会解释每次 forward 为什么可以创建一张新的图，以及 Python 控制流为什么能够直接参与 Eager Mode 下的模型计算。

### 4. `nn.Module` 与训练系统

第四篇从模型对象开始，讨论 PyTorch 如何组织参数、子模块、状态、数据和训练循环。

这一篇会覆盖：

- `nn.Module`；
- `nn.Parameter`；
- 参数注册；
- 子模块注册；
- `register_buffer()`；
- `state_dict()`；
- `load_state_dict()`；
- `train()` 与 `eval()`；
- hooks；
- Optimizer；
- param groups；
- 梯度清零与梯度累积；
- Dataset、Sampler、DataLoader；
- pinned memory；
- CPU-GPU 数据传输；
- 混合精度训练的基本概念。

内部会分成几个部分：

```text
4.1 模型对象与模块树
4.2 参数、Buffer 与 state_dict
4.3 Optimizer 与训练循环
4.4 Dataset、Sampler、DataLoader 与数据搬运
4.5 混合精度训练入门
```

这一篇重点回答：

- 为什么 `Parameter` 会被 Optimizer 自动发现？
- 为什么普通 Tensor 不会自动进入 `state_dict`？
- `Module.__setattr__()` 如何管理参数和子模块？
- 为什么 DataLoader 可能成为训练瓶颈？
- 为什么 `model.eval()` 不等于关闭梯度？

Java 对照会帮助理解 Module 的层级管理、state_dict 的状态快照、DataLoader 的生产者—消费者模型，以及 Optimizer 对参数状态的批量更新。

### 5. Dispatcher 与算子系统

第五篇是从 PyTorch 使用者走向 PyTorch 源码阅读者的关键一步。

这一篇会以简单的 `add` 算子为入口，分析：

```text
add
add_
add.out
```

并讨论：

- Operator Schema；
- functional、in-place 和 out variant；
- Dispatcher；
- Dispatch Key；
- CPU、CUDA、Autograd、Meta 后端；
- ATen；
- Native Functions；
- TensorIterator；
- Composite Kernel；
- Meta Kernel；
- 算子代码生成。

核心问题是：

- 同一个算子为什么可以支持 CPU 和 CUDA？
- Dispatcher 根据什么选择实现？
- Autograd Kernel 与设备 Kernel 是什么关系？
- TensorIterator 如何处理广播和不同 stride？
- Meta Kernel 为什么可以在没有真实数据的情况下推断结果？
- PyTorch 为什么需要算子代码生成？

这一篇不从复杂的 Attention 算子开始，而是先通过简单算子建立 Dispatcher、ATen 和 Kernel 之间的关系，再把这套方法迁移到更复杂的算子。

### 6. C++ 扩展与自定义算子

第六篇从“阅读框架”走向“扩展框架”，讨论 Python 如何连接到 C++ 和 CUDA。

这一篇会覆盖：

- PyTorch C++ Extension；
- pybind11；
- ATen C++ API；
- `at::Tensor`；
- Tensor metadata；
- `data_ptr()`；
- CMake 与 Ninja；
- CPU 自定义算子；
- CUDA 自定义算子；
- 自定义 Autograd；
- 算子注册；
- ABI 和编译问题；
- Python 与 C++ 之间的数据传递。

实践项目分成四个阶段：

```text
阶段 1：用 Python 实现一个算子
阶段 2：用 C++ 实现 CPU 算子
阶段 3：用 CUDA 实现 GPU 算子
阶段 4：补充 Autograd、测试和 Benchmark
```

这一篇的重点不是记住一套编译命令，而是理解：

```text
Python API
    ↓
pybind11 / Python Binding
    ↓
C++ Tensor
    ↓
Tensor metadata
    ↓
CPU / CUDA Kernel
    ↓
Autograd / Dispatcher
```

一个可用的自定义算子不能只在连续的 `float32` Tensor 上运行，还需要认真处理 dtype、device、stride、空 Tensor、生命周期、错误检查和 backward。

### 7. 编译执行与图优化

第七篇讨论 PyTorch 如何在保留 Eager Mode 灵活性的同时，通过编译器获得更高的执行效率。

这一篇会覆盖：

- FX Graph；
- Symbolic Tracing；
- Graph、Node、GraphModule；
- Graph Rewrite；
- TorchDynamo；
- Graph Break；
- Guard；
- Dynamic Shape；
- AOTAutograd；
- TorchInductor；
- Triton；
- 算子融合；
- 内存规划；
- 编译缓存。

贯穿全文的典型流程是：

```text
Python Model
    ↓
TorchDynamo
    ↓
FX Graph
    ↓
AOTAutograd
    ↓
TorchInductor
    ↓
Triton / C++ / Library Call
    ↓
CPU / GPU
```

本篇会用一个刻意简单的函数贯穿全文：

```python
def f(x, weight, bias):
    y = x @ weight + bias
    if x.shape[0] > 64:
        return torch.relu(y)
    return torch.tanh(y)
```

一段直线的矩阵计算让编译器的前端、中端、后端各有事可做；一个依赖输入 shape 的 Python 分支则暴露了"从 Python 程序中捕获图"的全部难点。本篇会跟踪这个函数的四次调用——冷编译、热路径、shape 变化、走到另一条分支——分别发生了什么。

重点区分：

- Eager Mode 中立即执行的算子；
- FX 对程序的图表示；
- 编译器内部的中间表示；
- 最终执行的 CPU/CUDA Kernel。

还会重点解释 graph break、guard、动态 shape 和编译缓存，因为 `torch.compile()` 并不是把任意 Python 代码直接转换成 CUDA，而是尝试捕获其中可分析的计算部分。

### 8. 性能优化与调试

第八篇围绕一个核心问题展开：

> **如何判断一个 PyTorch 程序慢，以及如何定位它为什么慢？**

这一篇的重心是**性能模型**和**测量方法**，优化手段是模型推导出来的结论。它从一个事实出发——CPU 和 GPU 是两条通过队列连接的异步时间线——把性能问题分成两个维度：

```text
时间维度：五类瓶颈
  CPU 侧      Python-bound、Launch-bound
  GPU 侧      Memory-bound、Compute-bound
  两侧之间    Sync-bound

空间维度：显存
  真的不够 · 碎片 · 泄漏 · 峰值
```

内部结构为：

```text
8.1 度量与工具地图：异步执行模型、Benchmark 方法、Profiler 与 Nsight
8.2 时间维度：五类瓶颈及各自的判断依据与处方
8.3 空间维度：显存的构成、Caching Allocator、碎片、峰值、泄漏与时空互换
8.4 完整案例：一个 Transformer block 的训练 step
```

这一篇会覆盖：

- CPU 与 CUDA 异步执行、CUDA Stream 与 CUDA Event；
- `torch.cuda.synchronize()` 与隐式同步点；
- `torch.utils.benchmark`、PyTorch Profiler、Nsight Systems、Nsight Compute；
- Kernel Launch 开销、CUDA Graphs；
- Memory Bandwidth、Compute Throughput、FLOPs、Arithmetic Intensity、Occupancy；
- 低精度（FP16、BF16、TF32）作为处方及其代价；
- CUDA Caching Allocator、`allocated` 与 `reserved`、显存碎片；
- 训练显存的构成与峰值；
- Activation Checkpointing 等时间与空间的互换。

所有性能结论都遵循同一套流程：

```text
建立基线
    ↓
设计正确性测试
    ↓
采集性能数据
    ↓
定位瓶颈
    ↓
修改实现
    ↓
重新 Benchmark
    ↓
确认没有性能回归
```

优化时不仅要说明“快了多少”，还要说明：

- 节省的是计算、访存、同步还是 Kernel Launch；
- 是否增加了显存；
- 是否改变了数值精度；
- 是否只对特定输入 shape 有效；
- 是否把成本从运行时转移到了编译或初始化阶段。

### 9. 分布式 PyTorch

第九篇回答单卡放不下模型或跑不完数据之后的问题。它不按 API 组织（DDP 一章、FSDP 一章、张量并行一章），而是用一条主线把所有并行策略放进同一张表：

> **每种并行策略，都是对训练中的五类状态——数据、参数、梯度、优化器状态、激活值——各自做一个决定：复制还是分片。每个决定对应一种集合通信原语和一个通信时机；所有决定加起来，决定了显存占用和通信量。**

全文分三层：

```text
运行时与工程        torchrun 启动 · 数据切分 · 分布式 Checkpoint · 多机拓扑 · 通信性能分析 · hang 的排查
        ↑
并行策略            DDP · ZeRO / FSDP · TP · PP · CP · EP · 多维组合
        ↑
通信底座            进程 / Rank / 进程组 · 集合通信原语 · NCCL · α + β 成本模型
```

这一篇会覆盖：

- SPMD 执行模型、Process、Rank、World Size、Process Group、`DeviceMesh`；
- NCCL 与 Gloo；
- Broadcast、Reduce、AllReduce、AllGather、ReduceScatter、AllToAll、P2P；
- 通信成本模型：α + β、Ring 与 Tree、`algbw` 与 `busbw`；
- 通信的异步语义与通信/计算重叠；
- DDP：Reducer、Gradient Bucket、通信 hook；
- ZeRO 三阶段与 FSDP2：`fully_shard`、DTensor、分片单元、预取、HSDP；
- 张量并行（TP）：列/行切分、Megatron 式共轭算子、词表并行、Sequence Parallel；
- 流水线并行（PP）：GPipe、1F1B、交错调度、Zero Bubble、`torch.distributed.pipelining`；
- 上下文并行（CP）：Ring Attention、负载均衡；
- 专家并行（EP）：dispatch/combine、容量因子、负载均衡损失；
- 多维混合并行的组合原则，以及训练与推理的并行选择差异；
- 分布式 Checkpoint、多机拓扑、通信性能分析、hang 的排查与扩展效率。

完整案例把第八篇的 Transformer block 训练 step 从 8 卡扩到 4 机 32 卡，逐步说明每一次扩展为什么要换并行策略，以及每一步的显存与通信账。

这一篇的重点不是罗列分布式 API，而是回答：

- 为什么需要通信？通信发生在 step 的哪个时刻？
- 每种策略的显存和通信量怎么自己算出来？
- 为什么 FSDP 比 DDP 多 50% 通信量？为什么 TP 只能在节点内做？
- 什么时候该从数据并行换到模型并行，多种并行怎么组合？
- 为什么增加 GPU 数量不一定带来线性加速？

### 10. PyTorch 的工程体系：一次改动如何安全地到达用户

前九篇描述的都是一个已经存在、并且正确运行的系统。第十篇换一个问题：**它是怎么做到一直正确、一直可用的？**

> **一个框架的工程体系，就是一次改动从写下到进入用户生产环境所经过的全部关卡，以及每个关卡守住什么。**

全文沿着一次改动的生命周期走一遍：

```text
写下改动
   ↓
① 本地构建能跑        仓库地图 · 构建流程与 Codegen · Debug 构建 · 调试到 C++
   ↓
② 结果正确            五种 oracle · OpInfo · 设备与 dtype 泛化 · gradcheck 内部 · 确定性
   ↓
③ 没有变慢            微基准与指令数 · TorchBench 与编译器看板 · 噪声与阈值
   ↓
④ 审查与 CI 合入       CI 分层 · flaky 的流程化处理 · 审批、MergeBot 与回滚
   ↓
⑤ 随版本发布          节奏与 release 分支 · wheel 矩阵与 ABI · 平台支持窗口
   ↓
⑥ 用户升级不坏        Python API 弃用周期 · 算子 Schema 的 BC/FC · C++ 稳定子集 · state_dict 版本
   ↓
⑦ 使用者跟随演进       版本策略 · 升级 playbook · 兼容矩阵 · nightly 与 RC
```

前六关是框架维护者的视角，第七关是使用者的视角。七个关卡共同守住三件事：**正确性**（主要由 ②）、**性能**（主要由 ③）、**兼容性**（主要由 ⑥）；构建是一切验证的前提，合入与发布把验证变成强制的、自动的流程。

框架与应用的区别在于**组合爆炸**：两千多个算子 × 十几种 dtype × 多个后端 × 各种 shape 与 layout × 多种执行模式。这个事实塑造了数据驱动的 OpInfo 测试、分层的 CI，以及按稳定程度分级的接口面。

实践终点是把第六篇的 `scale_shift` 算子（`myops` 项目）重新拿出来，让它走完这七关：复用 PyTorch 的测试基础设施、建立 Benchmark 基线与回归阈值、搭 CI 矩阵、守住 Schema 契约并完成一次向后兼容的演进、发布制品，最后站到使用者一侧看如何跟随 PyTorch 升级。

本篇最后给出全系列总结：回到总纲的那段代码和那串追问，逐一作答，并标出每个答案来自哪一篇。


## 贯穿全系列的实践线

为了避免每一篇都停留在孤立的 `Foo`、`Bar` 和玩具代码上，系列用两条实践线把机制落到代码上。它们不追求重新实现 PyTorch，而是通过受控的简化实验和一个持续演进的例子，理解 PyTorch 的关键机制。

### 第一条：用简化实现理解机制

前半段用两个自己动手写的最小实现，把"看懂"变成"能复现"：

```text
第二篇    简化版 Tensor      用 shape、stride、offset 实现索引、transpose 和 contiguous copy
第三篇    Mini-Autograd      用加法和乘法节点实现保存父节点、拓扑排序和反向传播
```

### 第二条：贯穿后半段的两个例子

后半段用两个持续演进的例子代替零散的代码片段。

一个是刻意简单的自定义算子：

```text
scale_shift(x, alpha, beta) = alpha * x + beta
```

```text
第六篇    实现它          Python 契约 → C++ CPU → CUDA → Autograd 与 Meta，接入 Dispatcher
第十篇    发布它          把它所在的 myops 项目走完构建、正确性、性能、CI、发布、兼容七关
```

另一个是一段真实的训练负载：

```text
第七篇    一个带 shape 分支的小函数  跟踪它的四次调用，看编译器前端、中端、后端和运行时各做了什么
第八篇    一个 Transformer block     以它的训练 step 为对象，走完基线 → 采集 → 归类 → 处方 → 优化报告
第九篇    同一个 Transformer block   从 8 卡扩到 4 机 32 卡，逐步换并行策略，算清每一步的显存与通信账
```

第四篇和第五篇没有独立的实践项目：第四篇以一个完整训练程序作为落点，第五篇则以原生算子 `add` 的完整路径作为源码阅读的样例。

各篇实践的重心不同：简化实现关注机制是否复现，算子项目关注正确性测试与 Benchmark，案例章节关注测量方法与优化报告。共同的要求只有一条：**每个结论都能在代码或数据上验证，而不是停留在示意图。**

最终目标不是造出一个新的深度学习框架，而是能够从用户代码一路追踪到运行时和硬件，并能解释不同设计的收益与代价。


## 阅读路径建议

### 完整学习路径

推荐按照以下顺序阅读：

```text
1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10
```

这是从用户编程模型走向框架内部和工程体系的完整路径。

### 想先理解模型训练

```text
1 → 2 → 3 → 4
```

先建立 Tensor、Autograd、Module 和训练循环的基本认知。

### 想尽快读懂 PyTorch 算子源码

```text
1 → 2 → 5 → 6
```

重点理解 Tensor metadata、Operator Schema、Dispatcher 和 C++/CUDA 扩展。

### 想进入 GPU 性能优化

```text
1 → 2 → 5 → 6 → 7 → 8
```

先理解 Tensor 和算子，再进入 Kernel、编译器和性能分析。

### 想进入分布式训练基础设施

```text
1 → 4 → 8 → 9 → 10
```

重点关注训练状态、显存、通信、性能和工程治理。

### 想进入推理系统和 LLM Serving

```text
1 → 2 → 4 → 8 → 9
```

推理引擎建立在 PyTorch 的模型计算、GPU 执行、显存管理和多卡通信之上。这条路径覆盖它们依赖的基础，Serving 系统自身的请求调度、KV Cache 和集群问题不在本系列范围。


## 前置要求与说明

### 前置要求

建议读者具备：

- 一门静态类型语言的工程经验，Java、C++、Go 均可；
- 基本 Python 语法和面向对象基础；
- 基本的 Linux 命令行使用能力；
- 向量、矩阵、导数和链式法则的基础概念；
- 能够阅读简单的 C++ 代码。

不要求一开始就掌握：

- CUDA Kernel 编程；
- 分布式训练；
- 编译器原理；
- 深度学习框架源码；
- 某个具体模型的完整实现。

### 数学内容的安排

本系列不会先单独展开一整套机器学习数学课程，而是将必要数学放在对应机制旁边：

| PyTorch 主题 | 同步补充的数学 |
|---|---|
| Tensor | 向量、矩阵和维度 |
| Broadcasting | 维度对齐 |
| Linear | 矩阵乘法 |
| Loss | 函数与误差 |
| Autograd | 导数与链式法则 |
| Backward | 梯度和 Jacobian |
| Attention | 点积和 softmax |
| Distributed | 矩阵切分与通信 |

数学在这里的目标不是完成考试，而是回答：

- 这个 Tensor 的 shape 为什么是这样？
- 这个算子的计算量是多少？
- 梯度从哪里来？
- 为什么 batch size 会影响显存和吞吐？
- 为什么某种并行策略会减少显存，却增加通信？

### 版本基线

正文以 **PyTorch 2.x（2.4 及之后）** 为基线。系列的重点是机制而不是可复现的基准数据，所以不逐篇给出实验环境；正文中涉及某个版本才引入或行为发生变化的 API（例如 `torch.library.custom_op`、FSDP2 的 `fully_shard`、`torch.load` 的 `weights_only` 默认值）时，会在使用处随文标注版本。文中出现的延迟、带宽、利用率等数字除明确标注硬件规格外均为示意，用于说明数量级与比例关系。

PyTorch 的以下部分变化较快：

- AMP API；
- `torch.compile` 内部实现；
- `torch.export`；
- Triton 集成；
- 分布式训练 API；
- C++/CUDA 扩展构建方式；
- CUDA 与硬件后端支持。

因此，文章会区分两类内容：

- **长期稳定的机制**：Tensor metadata、Autograd、算子分发、通信原语等；
- **版本敏感的实现**：编译器内部路径、具体 API、构建命令和配置参数。

阅读时应以对应版本的官方文档和源码为准。机制和分析方法通常比某个版本的命令更持久。

### 关于 AI-Infra 本身

本系列聚焦 PyTorch 的编程模型、执行引擎、算子系统、编译器、性能和分布式运行时，不试图覆盖所有 AI-Infra 领域。以下内容与本系列紧邻，但不在范围内：

- **Python 语言本身**：CPython 的对象模型、GIL、异步、类型系统。本系列假设读者已经掌握，只在 Python 与 C++ 边界处涉及；
- **C++ 语言本身**：模板、RAII、静态注册等在源码阅读中反复出现的特性。本系列解释它们在 PyTorch 中的用途，不讲语言机制；
- **单个 CUDA Kernel 的内部优化**：访存合并、共享内存、Tensor Core 编程。第六篇给出读懂一个 Kernel 所需的最小概念，第八篇在系统层面判断瓶颈类型，Kernel 内部如何写到硬件极限不展开；
- **LLM 推理系统**：请求生命周期、Continuous Batching、KV Cache 管理、Prefill/Decode 分离、集群路由。本系列覆盖它们所依赖的模型计算、GPU 执行、显存和通信基础，不讨论 Serving 系统自身的设计。


## 章节目录

1. [PyTorch 整体介绍](/pytorch-overall-introduction.html)
2. [Tensor 与内存布局](/pytorch-tensor-and-memory-layout.html)
3. [自动求导与动态计算图](/pytorch-autograd-and-dynamic-computation-graph.html)
4. [`nn.Module` 与训练系统](/pytorch-module-and-training-system.html)
5. [Dispatcher 与算子系统](/pytorch-dispatcher-and-operator-system.html)
6. [C++ 扩展与自定义算子](/pytorch-cpp-extension-and-custom-operators.html)
7. [编译执行与图优化](/pytorch-compilation-and-graph-optimization.html)
8. [性能优化与调试](/pytorch-performance-optimization-and-debugging.html)
9. [分布式 PyTorch](/pytorch-distributed-training.html)
10. [PyTorch 的工程体系：一次改动如何安全地到达用户](/pytorch-engineering-system.html)


## 最终目标

读完这套系列之后，读者应该能够从以下代码出发：

```python
output = model(inputs)
loss = criterion(output, targets)
loss.backward()
```

一路追问并回答：

```text
Tensor 如何表示输入？
    ↓
Module 如何组织模型？
    ↓
Autograd 如何构建计算图？
    ↓
Dispatcher 如何选择算子？
    ↓
Kernel 在 CPU 或 GPU 上如何执行？
    ↓
Compiler 如何对计算进行变换？
    ↓
Profiler 如何告诉我们瓶颈在哪里？
    ↓
Distributed Runtime 如何让多卡协同？
    ↓
Tests、Build 和 CI 如何保证系统可持续演进？
```

最终的目标不是“会用 PyTorch”，而是具备下面三种能力：

1. **阅读能力**：能够读懂 PyTorch、训练框架和推理框架中的关键源码；
2. **诊断能力**：能够解释错误、显存增长、性能下降和分布式故障的原因；
3. **扩展能力**：能够设计、实现、测试和优化一个新的算子或运行时组件。

这正是从后端工程师走向 AI-Infra 工程师时，PyTorch 最值得深入学习的部分。
