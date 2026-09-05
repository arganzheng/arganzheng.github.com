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
第十篇：理解大型框架如何测试、构建和演进
```

可以把这条主线进一步归纳为三条相互交织的线索：

```text
抽象线：Tensor → Autograd → Module → Operator → Compiler

执行线：Python → C++ → CUDA → Kernel → Hardware

工程线：Training → Profiling → Distributed → Testing → Build
```

这不是严格的单向依赖：

- 第五篇会复用第二篇关于 stride、dtype 和 device 的知识；
- 第六篇会复用第五篇关于 Operator Schema 和 Dispatcher 的知识；
- 第七篇会同时复用 Tensor metadata 和算子系统；
- 第八篇会帮助判断第六篇的自定义 Kernel 是否真的有效；
- 第九篇会复用第八篇的性能分析方法；
- 第十篇的测试和 Benchmark 方法应该贯穿前九篇。


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

本篇会用一个简单的矩阵计算模型演示：

```python
def f(x, weight, bias):
    return torch.relu(x @ weight + bias)
```

重点区分：

- Eager Mode 中立即执行的算子；
- FX 对程序的图表示；
- 编译器内部的中间表示；
- 最终执行的 CPU/CUDA Kernel。

还会重点解释 graph break、guard、动态 shape 和编译缓存，因为 `torch.compile()` 并不是把任意 Python 代码直接转换成 CUDA，而是尝试捕获其中可分析的计算部分。

### 8. 性能优化与调试

第八篇围绕一个核心问题展开：

> **如何判断一个 PyTorch 程序慢，以及如何定位它为什么慢？**

内部结构为：

```text
8.1 性能模型
8.2 Benchmark 方法
8.3 Profiler 与 Nsight
8.4 CUDA 执行与 Kernel 性能
8.5 显存分配与内存问题
8.6 混合精度与数值稳定性
8.7 一个完整优化案例
```

这一篇会覆盖：

- CPU 与 CUDA 异步执行；
- CUDA Event；
- `torch.cuda.synchronize()`；
- PyTorch Profiler；
- `torch.utils.benchmark`；
- Nsight Systems；
- Nsight Compute；
- Kernel Launch Overhead；
- Memory Bandwidth；
- Compute Throughput；
- FLOPs；
- Arithmetic Intensity；
- Occupancy；
- CUDA Stream；
- CUDA Caching Allocator；
- `allocated` 与 `reserved`；
- 显存碎片；
- Activation Checkpointing；
- FP16、BF16、TF32；
- 数值稳定性。

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

第九篇从通信原语开始，逐步进入 DDP 和 FSDP。

这一篇会覆盖：

- Process、Rank、World Size；
- Process Group；
- NCCL、Gloo；
- Broadcast；
- Reduce；
- AllReduce；
- AllGather；
- ReduceScatter；
- DistributedDataParallel；
- Gradient Bucket；
- 通信与计算重叠；
- FSDP；
- 参数、梯度和优化器状态分片；
- Checkpoint；
- 多机多卡拓扑；
- 通信性能和故障排查。

DDP 的核心流程可以概括为：

```text
每个进程持有一份模型副本
    ↓
各自处理不同数据
    ↓
各自执行前向和反向
    ↓
反向过程中同步梯度
    ↓
各进程独立更新参数
```

FSDP 则进一步讨论：

```text
参数、梯度、optimizer state 如何分片？
执行某个层时，参数如何临时聚合？
计算完成后，状态如何重新分片？
通信成本如何影响整体性能？
```

这一篇的重点不是罗列分布式 API，而是回答：

- 为什么需要通信？
- 通信发生在什么时候？
- 通信成本如何影响性能？
- 参数复制和参数分片如何影响显存？
- 为什么增加 GPU 数量不一定带来线性加速？

### 10. PyTorch 的工程体系：一次改动如何安全地到达用户

第十篇作为全系列收束，讨论一个复杂深度学习框架如何保证正确性、性能、可构建性和长期演进。

这一篇会覆盖：

- 算子测试；
- Autograd 测试；
- `gradcheck`；
- `gradgradcheck`；
- Reference Test；
- OpInfo；
- dtype、shape、device 测试；
- contiguous 与 non-contiguous 测试；
- CPU/CUDA 双后端测试；
- Benchmark 和性能回归；
- 从源码构建 PyTorch；
- CMake、Ninja、Debug/Release；
- PyTorch 工程目录；
- ABI；
- Python API、C++ API 和算子 Schema 的兼容性；
- 序列化与 checkpoint 兼容性；
- 后端兼容性和弃用机制。

如果为 PyTorch 添加一个自定义算子，至少需要覆盖：

```text
正确性
├── 多种 shape
├── 多种 dtype
├── CPU/CUDA
├── contiguous/non-contiguous
├── 空 Tensor
├── 标量 Tensor
├── 广播
├── requires_grad
├── 极端数值
├── NaN / Inf
└── 非法输入

工程质量
├── 单元测试
├── gradcheck
├── Benchmark
├── 文档
├── CI
└── 性能回归
```

最终实践项目是：

> **实现一个支持 CPU、CUDA、Autograd、Meta，并具有完整测试和 Benchmark 的自定义算子。**

它会把前九篇的知识串成一条完整链路：Tensor metadata、Autograd、Dispatcher、C++/CUDA、编译、性能分析、分布式兼容性和工程测试。


## 一个贯穿全系列的实践项目

为了避免每一篇都停留在孤立的 `Foo`、`Bar` 和玩具代码上，系列会使用一个逐步演进的实践项目：

### Mini PyTorch Runtime

项目不追求重新实现 PyTorch，而是通过一组受控的简化实验，理解 PyTorch 的关键机制。

```text
阶段 1：实现简化版 Tensor
阶段 2：实现 Mini-Autograd
阶段 3：构建简单的 nn.Module 和训练循环
阶段 4：实现一个 Python 算子
阶段 5：实现一个 C++ CPU 算子
阶段 6：实现一个 CUDA 算子
阶段 7：尝试 FX Graph 和图重写
阶段 8：使用 Profiler 和 Benchmark 定位瓶颈
阶段 9：使用 DDP 运行多进程训练
阶段 10：补充完整测试、构建和性能回归
```

每个阶段都应同时具备：

- 最小可运行代码；
- 正确性测试；
- 与 PyTorch 结果的对照；
- 性能基线；
- 失败案例；
- Java 工程师容易产生的误解。

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

### 想继续阅读 vLLM 和 LLM Serving

```text
PyTorch 1 → 2 → 4 → 8 → 9
        ↓
《大模型推理系统揭秘：从 vLLM 看 LLM Serving Infra 核心技术》
```

PyTorch 系列负责解释模型计算、GPU 执行、分布式和性能基础；vLLM 系列负责解释请求调度、KV Cache、Continuous Batching、模型适配、PD 分离和 Serving 集群。两者互为补充，而不是重复讲解。


## 与已有系列的关系

目前的学习路线可以形成三组相互衔接的内容。

### 第一组：Python 控制平面

《Python 在 AI-Infra：从语言机制到生产交付》解决的是：

> **如何使用 Python 编写 AI-Infra 系统中的组织、调度、扩展、观测和交付代码？**

它覆盖 Python 运行时、类型系统、并发、动态机制、内存、测试和工程化。

### 第二组：PyTorch 计算运行时

本系列解决的是：

> **如何表达、执行、扩展、优化和分布式运行 Tensor 计算？**

它处于 Python 控制平面和底层硬件执行之间。

### 第三组：LLM Serving 系统

《大模型推理系统揭秘：从 vLLM 看 LLM Serving Infra 核心技术》解决的是：

> **如何围绕请求、Token、KV Cache、GPU 和网络构建高性能 LLM Serving 系统？**

三个系列可以连接为：

```text
Python 语言与工程
        ↓
PyTorch 计算与运行时
        ↓
模型训练与推理执行
        ↓
LLM Serving 与 AI-Infra 系统
```


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

正文以 **PyTorch 2.x** 为主要基线，具体小版本、CUDA、驱动和编译器版本会在每篇文章的实验环境中明确记录。

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

本系列聚焦 PyTorch 的编程模型、执行引擎、算子系统、编译器、性能和分布式运行时，不试图覆盖所有 AI-Infra 领域。

如果你希望进一步理解：

- LLM 请求生命周期；
- Continuous Batching；
- KV Cache；
- Prefill/Decode；
- 多卡 Serving；
- PD 分离；
- 集群路由和状态管理；

可以继续阅读[《大模型推理系统揭秘：从 vLLM 看 LLM Serving Infra 核心技术》](/deep-dive-into-vllm.html)。

如果你希望补齐 Python 控制平面能力，可以阅读[《Python 在 AI-Infra：从语言机制到生产交付》](/python-for-ai-infra.html)。


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
