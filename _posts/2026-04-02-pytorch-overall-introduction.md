---
layout: post
title: "PyTorch 深度实践（01）：PyTorch 整体介绍"
subtitle: "PyTorch Overall Introduction" 
tags: [PyTorch, AI, AI-Infra]
catalog: true
---

> 本文是[《PyTorch 深度实践：从 Tensor 到深度学习运行时》](/deep-dive-into-pytorch.html)系列的第一篇（共十篇）。下一篇：[Tensor 与内存布局](/pytorch-tensor-and-memory-layout.html)

PyTorch 经常被介绍成一个“深度学习框架”，也经常被使用成一个 Python 库：导入 `torch`，创建 Tensor，定义 `nn.Module`，然后训练模型。

这种理解对于开始使用 PyTorch 已经足够，但对于训练平台、推理引擎、算子开发和 AI-Infra 来说还不够。真正需要理解的是：

> **PyTorch 如何把 Python 中表达的张量计算，转化为可以自动求导、跨设备执行、编译优化和分布式协作的运行时系统？**

一行看起来很普通的代码：

```python
z = torch.add(x, y)
```

背后可能涉及：

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
CPU / CUDA / Meta Kernel
    ↓
底层数学库与硬件
```

如果这条链路只停留在“PyTorch 会自动处理”，那么遇到下面的问题时就只能依赖试错：

- 为什么同一个算子既能运行在 CPU，也能运行在 CUDA 上？
- 为什么某些 Tensor 操作会产生拷贝，另一些操作只是创建 view？
- 为什么 `model(x)` 不等于简单调用 `model.forward(x)`？
- 为什么模型在 eager mode 下运行正常，`torch.compile()` 后却出现 graph break？
- 为什么 GPU 利用率很低，却找不到明显的 Python 瓶颈？
- 为什么增加 GPU 数量后，训练速度没有线性提升？
- 为什么一个看似简单的 C++ 扩展会遇到 ABI、stride、dtype 或生命周期问题？

本文是《PyTorch 深度实践：从 Tensor 到深度学习运行时》的第一篇。它只负责建立全局地图，不深入某一个模块的全部实现。后续文章会沿着这张地图，逐步展开 Tensor、Autograd、Module、Dispatcher、编译器、性能和分布式运行时。


## 一、PyTorch 到底是什么？

### 1. PyTorch 是什么？

PyTorch 是一个面向张量计算和深度学习的开源计算平台。它以 Python 作为主要用户接口，以 C++ 运行时和算子系统作为执行核心，并通过 CPU、CUDA 以及其他硬件后端完成实际计算。

更完整地说，PyTorch 提供了一套从模型表达一直到设备执行的连续抽象：

```text
Tensor 与模型
    ↓
自动求导与训练
    ↓
算子分发与设备抽象
    ↓
CPU / GPU 执行
    ↓
编译优化与分布式扩展
```

因此，PyTorch 既不是只有 Python API 的工具包，也不是单独的 GPU 算子集合，而是连接以下几个层次的深度学习运行时：

```text
用户模型代码
    ↓
PyTorch 编程模型
    ↓
算子运行时
    ↓
设备后端
    ↓
Kernel 与硬件
```

从使用者角度看，PyTorch 提供了大量 Python API：

```python
import torch
from torch import nn

x = torch.randn(32, 128, device="cuda")
layer = nn.Linear(128, 256, device="cuda")
y = layer(x)
```

但 Python API 只是 PyTorch 的入口，不是 PyTorch 的全部实现。

PyTorch 至少包含以下几类能力：

| 能力 | 解决的问题 |
|---|---|
| Tensor | 如何表示和操作多维数据 |
| Autograd | 如何自动计算梯度 |
| `nn.Module` | 如何组织模型、参数和状态 |
| Optimizer | 如何根据梯度更新参数 |
| Dispatcher | 如何选择具体的算子实现 |
| ATen | 如何提供统一的 Tensor 和算子抽象 |
| CPU/CUDA Kernel | 如何在不同设备上执行计算 |
| Compiler | 如何捕获、变换和优化计算图 |
| Distributed | 如何让多个进程和设备协同工作 |
| Extension | 如何接入 C++、CUDA 和自定义硬件 |

### 2. PyTorch 不是什么？

#### 2.1 PyTorch 不只是 Python API

```python
import torch

x = torch.randn(2, 3, device="cuda")
y = torch.randn(2, 3, device="cuda")
z = x + y
```

上面的 Python 代码只是用户入口。真正完成计算的部分还包括 Python Binding、Operator Schema、Dispatcher、ATen、设备后端以及 CPU/CUDA Kernel。

因此，阅读 PyTorch 源码时，不能只看 `torch` 和 `torch.nn` 目录；排查性能问题时，也不能只看 Python 函数是否高效。

#### 2.2 PyTorch 不等于 CUDA

CUDA 是 NVIDIA GPU 的编程平台和软件生态，PyTorch 是构建在 CUDA 等后端之上的深度学习计算平台。

```text
PyTorch
    ├── CPU 后端
    ├── CUDA 后端
    ├── ROCm 后端
    ├── Meta 后端
    └── 其他设备后端
```

PyTorch 可以调用 CUDA Kernel、cuBLAS 和 cuDNN，但这些只是它所使用的一个后端和若干底层库。即使使用 CUDA，模型、Autograd、Module、Dispatcher 和训练系统仍然属于 PyTorch 的职责范围。

#### 2.3 PyTorch 不只是神经网络层的集合

`nn.Linear`、`nn.Conv2d` 和 Transformer 模块是 PyTorch 的重要组成部分，但 PyTorch 的核心抽象是 Tensor 计算和围绕 Tensor 建立的运行时。

```text
Tensor 计算
    ↓
Autograd
    ↓
算子分发
    ↓
设备执行
```

`nn.Module` 负责组织模型和状态，它本身不是 Kernel；一个 Module 可能展开成许多 Tensor 操作，也可能在编译后被融合成不同的 Kernel 组合。

#### 2.4 几组需要分开的概念

| 概念 | 它是什么 | 它不是什么 |
|---|---|---|
| Tensor | 数据、布局、类型和设备位置的运行时对象 | 不是模型，也不是 Kernel |
| `nn.Module` | 组织层次、参数和状态的模型对象 | 不是一段固定的 GPU 指令 |
| Autograd Graph | 描述梯度传播关系的运行时结构 | 不是最终的硬件执行图 |
| FX Graph | 用于程序分析和重写的图表示 | 不等于最终 CUDA Kernel |
| Operator | 具有统一 Schema 和语义的计算操作 | 不等于某个后端的具体实现 |
| Kernel | 在 CPU、GPU 或其他设备上执行的实现 | 不等于完整的 PyTorch 模型 |
| CUDA | NVIDIA 的设备编程平台和后端生态 | 不等于 PyTorch 本身 |

这些边界会在后面的静态分层图和动态执行路径中逐一展开。


## 二、PyTorch 与其他深度学习框架

框架比较不能简单归结为“谁更好”。更有意义的比较是：它们如何表达计算、如何执行程序，以及如何把程序交给编译器和硬件。

| 框架 | 编程模型 | 典型执行方式 | 突出特点 |
|---|---|---|---|
| PyTorch | Python-first 的 Tensor 和 Module | Eager 为主，编译为辅 | 灵活、易调试、生态丰富 |
| TensorFlow | Tensor、Layer 和计算图 | Eager 与 Graph 并存 | 工业化工具链和部署生态完整 |
| JAX | 函数组合与程序变换 | 以变换和编译为核心 | `jit`、自动向量化、自动并行 |
| MXNet | 符号图与动态图 | 混合执行 | 历史上强调灵活性和分布式 |
| OneFlow | Tensor 与分布式训练抽象 | 图与运行时优化 | 面向大规模训练场景 |

PyTorch 的核心编程特色，是 Eager-first，同时逐步具备 Compiler-ready 能力。它的核心取舍可以概括为：

```text
优先提供灵活的 Python 编程体验
        ↓
允许程序在运行时立即执行
        ↓
通过 Autograd 动态构建计算图
        ↓
再使用 Compiler 对稳定部分进行优化
```

这种设计带来几个优势：

- Python 控制流可以直接参与模型计算；
- 出错时可以使用普通 Python 调试工具；
- 模型结构可以快速修改；
- 研究代码和工程代码之间的迁移成本较低；
- 可以逐步把性能敏感的部分下沉到 C++、CUDA 或编译器。

它也带来相应成本：

- Python 调度本身可能成为开销；
- 每次执行的动态性可能限制编译器优化；
- Kernel Launch 和同步可能放大细粒度操作的成本；
- 运行时行为比静态图更难提前分析；
- 不同设备、dtype 和 shape 的组合增加测试矩阵。

因此，PyTorch 的设计方向不是“只使用动态图”或“只使用静态图”，而是试图同时保留两者的价值：

```text
Eager Mode       → 灵活、可调试、适合探索
Compiled Mode    → 可分析、可融合、适合稳定执行
```


## 三、PyTorch 的架构演进

这里的“阶段”是为了帮助理解架构演进而做的归纳，不是 PyTorch 官方发布的固定分期。版本号和日期采用官方发布节点作为参照；不同能力往往跨越多个版本逐步成熟，不能简单归因于某一个版本。

| 阶段 | 代表版本与时间 | 主要变化 | 对今天架构的影响 |
|---|---|---|---|
| 动态图与研究友好 | 0.1.x，2016 年 9 月起公开 alpha，2017 年初持续迭代 | 以 Python 为中心的 Tensor、动态图和自动求导体验 | 奠定 Eager-first 的编程模型 |
| API 稳定与生产化 | 0.4，2018 年；1.0，2018 年 12 月 | Tensor/Variable 接口整合，API 稳定，TorchScript 和生产能力逐步引入 | 从研究工具走向通用深度学习平台 |
| C++ 运行时与分布式成熟 | 1.x，2019—2022 年 | ATen、C++ Tensor API、Dispatcher、分布式训练和自定义扩展持续完善 | Python API 之下形成完整运行时 |
| 图表示与编译基础设施 | 1.8—1.13，2021—2022 年 | FX、functorch、TorchDynamo、AOTAutograd、TorchInductor 等组件逐步发展 | 为 Eager 程序提供图捕获和优化路径 |
| 编译执行与大模型运行时 | 2.0，2023 年 3 月 15 日及之后 | `torch.compile` 成为主要编译入口，动态 Shape、分布式和 Transformer 优化持续增强 | 保留 Eager 体验，同时获得编译优化能力 |

### 1. 阶段一：动态图与研究友好

PyTorch 0.1.x 于 2016 年 9 月起以 alpha 版本公开发布，并在 2017 年初持续迭代。早期 PyTorch 的核心体验可以概括为：

```python
x = torch.randn(10, requires_grad=True)
y = x * 2
z = y.relu()
loss = z.sum()
loss.backward()
```

代码执行到哪一行，计算就发生到哪一行；Python 的 `if`、`for` 和函数调用可以直接参与模型逻辑。这个 Eager-first 的设计，成为 PyTorch 后续架构一直保留的用户体验基础。

### 2. 阶段二：API 稳定与生产化

PyTorch 0.4 在 2018 年带来了重要的 Tensor/Variable 接口整合。随后 PyTorch 1.0 于 2018 年 12 月发布，API 稳定性、生产使用和图执行能力成为重要方向。

这一阶段的关键不是“动态图被静态图取代”，而是开始提供从 Eager 模型走向更受约束执行环境的路径，例如 TorchScript。PyTorch 由此同时面对两类需求：

```text
研究与开发 → 灵活、即时、容易调试
生产与部署 → 可保存、可分析、可优化
```

### 3. 阶段三：C++ 运行时、算子系统与分布式成熟

在 1.x 系列中，PyTorch 持续强化：

- ATen；
- C++ Tensor API；
- Dispatcher；
- CPU/CUDA Kernel；
- C++ 前端；
- 分布式通信；
- 自定义算子和扩展机制。

这带来了一个重要变化：

> PyTorch 不再只是一个 Python 深度学习库，而是逐渐成为具有完整运行时和算子系统的深度学习平台。

Python 仍然是主要入口，但大量真正影响性能和设备行为的逻辑已经进入 C++、CUDA 和底层库。

### 4. 阶段四：从动态图走向编译执行

Eager Mode 灵活，但也存在明显成本：

- Python 代码需要参与调度；
- 大量细粒度操作会产生很多 Kernel Launch；
- 编译器难以看到跨算子的全局关系；
- 动态控制流和数据依赖会限制静态优化。

因此，1.8—1.13 期间逐步形成了多种图表示和编译基础设施：

- FX；
- functorch；
- TorchDynamo；
- AOTAutograd；
- TorchInductor；
- Triton。

### 5. 阶段五：编译执行与大模型运行时

PyTorch 2.0 于 2023 年 3 月 15 日发布，其主要方向是在保持 Eager Mode 开发体验的同时，通过 `torch.compile` 将稳定的计算部分交给编译器。

现代 PyTorch 还需要继续处理：

- 多 GPU 训练；
- 参数、梯度和优化器状态分片；
- 混合精度；
- 动态 Shape；
- Meta Tensor 和 Fake Tensor；
- CPU、CUDA、ROCm 及其他后端；
- 大模型检查点；
- 量化和推理优化。

这些能力看起来分散，实际都在回答同一个问题：

> **如何让同一套模型和算子抽象，在不同设备、不同规模和不同执行模式下保持可组合？**

## 四、三张地图

下面三张地图从三个视角组织全文：第一张按**职责**说明系统由哪些层组成，第二张说明一次算子调用如何**动态**穿过这些层，第三张按**源码**说明这些东西在仓库里住在哪个目录、哪个库。前两张是本系列后续篇章的推进顺序，第三张是读源码时的坐标。

### 1. 静态视角：PyTorch 的逻辑分层

这一节我们先从**静态视角**出发，看看“PyTorch 由哪些职责层组成、每一层负责什么”。

从上到下，可以把 PyTorch 粗略分为以下六层：

```mermaid
flowchart TB
    A[① 用户模型与训练代码]
    B[② 编程模型]
    C[③ 图与编译]
    D[④ 算子运行时]
    E[⑤ 设备与通信]
    F[⑥ Kernel 与硬件]

    A --> B
    B -- "Eager：逐算子" --> D
    B -- "torch.compile" --> C
    C -- "生成 Kernel" --> F
    D --> E --> F
```

| 层 | 核心组件 | 回答的问题 |
|---|---|---|
| ① 用户模型与训练代码 | `nn.Module` 子类、损失函数、训练循环、评估与推理逻辑 | 模型长什么样、怎么训练 |
| ② 编程模型 | Tensor、Autograd、`nn.Module` / Parameter / Buffer、Optimizer、Dataset / DataLoader、`state_dict` | 用户用什么抽象表达计算和状态 |
| ③ 图与编译 | TorchDynamo、FX Graph、AOTAutograd、TorchInductor、Guard / Graph Break / 编译缓存 | 动态 Python 程序如何变成可优化的图 |
| ④ 算子运行时 | Operator Schema、Dispatcher / DispatchKey、ATen 算子实现（CPU / CUDA / Composite / Meta）、TensorIterator | 一个算子在当前上下文该调用哪个实现 |
| ⑤ 设备与通信 | CUDA Runtime（Stream / Event / 同步）、Caching Allocator、H2D / D2H 数据迁移、进程组与集合通信（NCCL / Gloo） | 计算和数据如何到达设备、多设备如何协同 |
| ⑥ Kernel 与硬件 | C++ CPU Kernel、CUDA Kernel、cuBLAS / cuDNN 等厂商库、Triton Kernel、CPU / GPU / 显存 / 互连 | 计算最终消耗多少算力、带宽和时间 |

Eager 模式下每个算子从编程模型直接进入算子运行时；`torch.compile` 则先经过图与编译层，生成的 Kernel 直接落到最底层。这不是 PyTorch 源码目录的直接映射（源码分层见第 3 节），而是一张用于分析问题的逻辑地图。

#### 1.1 第一层：用户模型与训练代码

这是最接近业务和算法的部分：

```python
class Classifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
```

用户在这一层表达：

- 模型结构；
- 前向计算；
- 损失函数；
- 优化器；
- 训练循环；
- 推理逻辑。

这一层主要使用 Python，但它产生的每个 Tensor 操作最终都需要进入下面的运行时。

#### 1.2 第二层：编程模型

这一层包括：

- Tensor；
- Autograd；
- `nn.Module`；
- Dataset 和 DataLoader；
- Optimizer；
- Checkpoint。

它提供的是深度学习开发者使用的抽象。例如：

```python
output = model(inputs)
loss = criterion(output, targets)
loss.backward()
optimizer.step()
```

这段代码隐藏了大量细节，但隐藏不等于不存在：

- `inputs` 和 `targets` 有 dtype、shape、device；
- `model` 可能是一个带参数和 Buffer 的模块树；
- forward 可能动态构建 Autograd 图；
- `backward()` 会沿图传播梯度；
- `optimizer.step()` 会读取参数和梯度状态。

第二篇到第四篇会集中讨论这一层。

#### 1.3 第三层：图与编译

这一层负责把 Python 程序或 Tensor 操作转换为可以分析的图表示：

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
Triton / C++ / Vendor Library
```

这里需要区分几种概念：

- Eager 计算图：为了 Autograd 记录的运行时图；
- FX Graph：用于程序分析和重写的 Python 层图表示；
- 编译器中间表示：面向代码生成和优化的内部表示；
- Kernel：最终在 CPU 或 GPU 上执行的实现。

第七篇会详细讨论这几个概念的关系。

#### 1.4 第四层：算子运行时

这一层负责回答：

> 这个算子是什么？在当前运行时上下文中，应该调用哪个实现？

它包括：

- Operator Schema；
- Dispatcher；
- Dispatch Key；
- ATen；
- TensorIterator；
- Native Functions；
- Composite Kernel；
- Meta Kernel。

例如，同样是加法操作：

```python
z = x + y
```

如果 `x` 和 `y` 是 CPU Tensor，就需要 CPU 实现；如果它们是 CUDA Tensor，就需要 CUDA 实现；如果当前操作正在构建 Meta Tensor 的形状推断，又需要 Meta 实现。

第五篇会以 `add`、`add_` 和 `add.out` 为例展开这一层。

#### 1.5 第五层：设备与通信

这一层连接 PyTorch 运行时和具体设备或进程：

- CPU；
- CUDA；
- ROCm；
- Meta；
- NCCL；
- Gloo；
- 其他硬件后端。

它处理的不只是“把代码放到 GPU 上”，还包括：

- 内存分配；
- 数据迁移；
- Kernel Launch；
- stream；
- event；
- 进程间通信；
- 集合通信；
- 设备同步。

第八篇和第九篇会分别讨论性能执行与分布式通信。

#### 1.6 第六层：Kernel 与硬件

最底层是实际完成计算的代码和硬件：

- C++ CPU Kernel；
- CUDA Kernel；
- Triton Kernel；
- cuBLAS；
- cuDNN；
- 设备厂商库；
- CPU 指令集；
- GPU Streaming Multiprocessor。

这一层决定了计算最终消耗多少：

- 算力；
- 显存带宽；
- Kernel Launch；
- 寄存器和共享内存；
- 线程块调度；
- 设备间通信。

但性能问题不一定发生在最底层。上层的 Python 调度、Tensor 布局、数据搬运和同步，都可能成为瓶颈。


### 2. 动态视角：一次算子调用发生了什么？

上面的静态地图回答了“系统由什么组成”，但还没有回答“代码如何在系统中流动”。下面我们切换到**动态视角**：以一次 `torch.add(x, y)` 为例，追踪一个算子从 Python 入口经过绑定、Schema、Dispatcher 和 ATen，最终进入具体设备 Kernel 的过程。

以简单的加法为例：

```python
z = torch.add(x, y)
```

可以沿着下面的路径理解：

```mermaid
flowchart LR
    A[Python API<br/>torch.add / x + y]
    B[Python Binding]
    C[Operator Schema]
    D[Dispatcher<br/>Dispatch Key Set]
    E[ATen Operator]
    F{运行时上下文}
    G[CPU Kernel]
    H[CUDA Kernel]
    I[Meta Kernel]
    J[底层数学库与硬件]

    A --> B --> C --> D --> E --> F
    F -->|CPU Tensor| G
    F -->|CUDA Tensor| H
    F -->|Meta Tensor| I
    G --> J
    H --> J
```

这张图表达的是典型执行路径：统一的算子契约和 Dispatcher 位于上层，具体设备 Kernel 位于下层。实际路径会因算子实现、Autograd、编译模式和 PyTorch 版本而有所变化。

#### 2.1 第一步：Python API

用户调用的是 Python 暴露出来的函数：

```python
torch.add(x, y)
```

也可以使用运算符形式：

```python
z = x + y
```

或者 Tensor method：

```python
z = x.add(y)
```

这几个入口在用户层语义相近，但可能对应不同的生成绑定和调用形式。不要只根据 Python 表面语法判断内部实现路径。

#### 2.2 第二步：Python Binding

PyTorch 需要把 Python 对象转换为 C++ 运行时能够理解的对象：

```text
Python Tensor object
    ↓
C++ Tensor handle
    ↓
算子参数检查与转换
```

这一层涉及 Python/C++ 边界、引用管理和参数解析。

它不是把所有 Tensor 数据复制到 C++，而通常是让 C++ 侧获得对 Tensor 对象和底层存储的可管理引用。

#### 2.3 第三步：Operator Schema

算子需要有明确的签名和语义。例如可以抽象表示为：

```text
add(Tensor self, Tensor other, Scalar alpha=1) -> Tensor
```

Schema 描述：

- 参数类型；
- 返回类型；
- 默认参数；
- mutable 参数；
- aliasing 和 out variant 等语义。

Schema 是算子系统的重要契约。它让不同语言绑定、后端实现、Autograd 和编译器能够围绕同一个算子定义协作。

#### 2.4 第四步：Dispatcher

Dispatcher 根据运行时信息选择实现。影响选择的因素可能包括：

- Tensor 的 device；
- dtype；
- 是否需要 Autograd；
- 是否处于 tracing 或 compiling；
- 是否是 Meta Tensor；
- 是否有自定义后端；
- 是否使用特殊布局。

可以简化成：

```text
算子名 + Tensor 元数据 + 运行时上下文
                    ↓
             Dispatch Key Set
                    ↓
              具体 Kernel
```

这不是 Java 方法重载的简单等价物。Java 重载通常依据编译期静态类型选择方法，而 PyTorch 的分发还会受到设备、Autograd、Tracing 和运行时上下文影响。

#### 2.5 第五步：ATen Operator

ATen 是 PyTorch 的核心 Tensor 和算子库，提供跨设备的统一抽象。

从用户角度看：

```python
z = x + y
```

从运行时角度看，它需要保证：

- shape 规则一致；
- 广播规则一致；
- dtype 规则一致；
- 错误行为一致；
- 不同后端具有相同的算子语义。

ATen 并不意味着所有计算都由一份代码完成。它提供统一接口，具体执行仍可能落到不同后端。

#### 2.6 第六步：CPU、CUDA 或其他 Kernel

最终，Dispatcher 会让操作进入某个具体实现：

```text
CPU Tensor  → CPU Kernel
CUDA Tensor → CUDA Kernel
Meta Tensor → Meta Kernel
```

某些算子会调用底层库：

```text
矩阵乘法 → cuBLAS / cuBLASLt
卷积     → cuDNN 或专用 Kernel
通用逐元素操作 → Native CUDA Kernel
```

返回的结果仍然要重新包装成 PyTorch Tensor，并保留正确的：

- shape；
- stride；
- dtype；
- device；
- Autograd 信息；
- storage 生命周期。

#### 2.7 这是一条概念路径

上面的流程适合建立架构认知，但不是所有算子在所有 PyTorch 版本中的固定源码调用栈。

实际路径可能因为以下因素而变化：

- Python function、Tensor method 或运算符入口不同；
- 算子是否有 Composite 实现；
- 是否正在进行 Autograd；
- 是否处于编译或 Fake Tensor 模式；
- 后端是否覆盖了特定 Dispatch Key；
- PyTorch 版本的代码生成和绑定方式变化。

这些层次是稳定的职责边界；具体函数调用栈则可能随版本和算子实现变化。

### 3. 代码视角：源码目录与库的分层

前两张地图按职责划分，不对应源码目录。真正打开 `pytorch/` 仓库，看到的是另一种分层——按**库**划分，自上而下四层，每一层只依赖它下面的层：

```mermaid
flowchart TB
    L1["torch/<br/><br/>Python 层"]
    L2["torch/csrc/<br/><br/>C++ 绑定与运行时引擎"]
    L3["aten/src/ATen/<br/><br/>Tensor 算子库"]
    L4["c10/<br/><br/>核心基础库"]
    L5["系统库与硬件<br/><br/>CUDA Runtime · cuBLAS / cuDNN / oneDNN · NCCL"]

    L1 -- "torch._C 扩展模块（pybind11 / Python C API）" --> L2
    L2 -- "ATen C++ API：at::add(...) 进入 Dispatcher" --> L3
    L3 -- "基础类型：TensorImpl · DispatchKeySet · Allocator" --> L4
    L4 --> L5
    L2 -. "Autograd Kernel 注册进 Operator Table" .-> L3
```

| 层 | 源码位置 | 核心组件 | 向上提供 |
|---|---|---|---|
| Python 层 | `torch/` | `nn`、`optim`、`utils.data`、`distributed`（Python 侧）、`fx`、`_dynamo`、`_functorch`、`_inductor`、`torch.library`、`torch.testing` | 用户 API |
| C++ 绑定与运行时引擎 | `torch/csrc/` | Python 绑定、Autograd 引擎（`autograd/`）、c10d（`distributed/`）、Profiler、TorchScript（`jit/`，维护模式）、Dynamo 帧求值 hook、AOTInductor 运行时 | `torch._C` 扩展模块 |
| Tensor 算子库 | `aten/src/ATen/` | Dispatcher 与 Operator Table（`core/dispatch/`）、`native_functions.yaml`、CPU / CUDA Kernel（`native/`）、TensorIterator；粘合代码由 `torchgen/` 生成 | `at::Tensor` 与 ATen C++ API |
| 核心基础库 | `c10/` | TensorImpl、StorageImpl、Device / DeviceGuard / Stream、Allocator / CUDACachingAllocator、DispatchKey / DispatchKeySet、ScalarType / Layout | 所有层共用的基础类型 |
| 系统库与硬件 | — | CUDA Runtime、cuBLAS / cuDNN / CUTLASS（被 ATen Kernel 调用）、oneDNN / OpenMP、NCCL / Gloo（被 c10d 调用）、Triton（编译器生成的 Kernel） | 设备能力 |

实线是一次算子调用自上而下的依赖方向，箭头上是层与层之间的接口。虚线是一条反向关系：Autograd 引擎位于 `torch/csrc`，却把自己的 Kernel 注册进 ATen 的 Operator Table，所以 3.5 的调用路径会在这两层之间往返一次。厂商库和 NCCL 分别被 ATen 的 Kernel 和 c10d 直接调用，编译栈则绕过 ATen 的 Kernel 自己生成 Triton 代码（第七篇）。

编译之后，这四层对应几个动态库：`libc10`（c10）、`libtorch_cpu` 与 `libtorch_cuda`（ATen 和 `torch/csrc` 中与 Python 无关的部分：Autograd 引擎、c10d 等）、`libtorch_python`（Python 绑定）。`import torch` 时它们被依次加载；第六篇的 C++ 扩展链接的正是这些库，ABI 问题也由此而来。

下面自上而下逐层说明每一层放什么、向上提供什么。

#### 3.1 `torch/`：Python 层

用户直接接触的一切都在这里：`torch.nn`、`torch.optim`、`torch.utils.data`、`torch.distributed` 的 Python 侧，以及 `torch.Tensor` 的大部分 Python 方法。值得注意的是**编译栈的主体也在这一层**：Dynamo（`torch/_dynamo`）、AOTAutograd（`torch/_functorch`）、Inductor（`torch/_inductor`）和 FX（`torch/fx`）都是 Python 代码——编译器分析的是 Python 程序，产出的是 Triton / C++ 源码，它自己不必是 C++。

这一层不做 Tensor 计算。所有真正的计算都通过 `torch._C` 这个扩展模块进入下一层。

#### 3.2 `torch/csrc/`：C++ 绑定与运行时引擎

`csrc` 是 "C source" 的缩写，这里是 Python 与 C++ 的边界，也是几个大型运行时引擎的所在：

- **Python 绑定**：把 C++ 的 Tensor 和函数暴露成 Python 对象。`torch.add` 在 Python 侧是一个由 Codegen 生成的绑定函数，负责解析参数、把 Python 对象转成 `at::Tensor`，再调用 ATen；
- **Autograd 引擎**（`autograd/`）：反向传播的执行器，以及为每个算子生成的反向节点（第三篇）；
- **c10d**（`distributed/`）：进程组、NCCL / Gloo 后端、DDP 的 Reducer（第九篇）；
- **Profiler**（`profiler/`）：与 Kineto 集成，采集 CPU 和 CUDA 时间线（第八篇）；
- **TorchScript**（`jit/`）：上一代编译路径，C++ 实现，2.x 已进入维护模式，本系列不讨论；
- **Dynamo 与 Inductor 的少量 C++ 部分**：帧求值 hook（`dynamo/`）和 AOTInductor 运行时（`inductor/`）。

这一层向下依赖 ATen 的 C++ API：一切 Tensor 运算都写成 `at::add(x, y)` 之类的调用。**Dispatcher 不在这一层**——它在 ATen。

#### 3.3 `aten/src/ATen/`：Tensor 算子库

ATen（"A Tensor library"）是算子的家，包括两部分：

- **分发**（`core/dispatch/`）：Dispatcher 与 Operator Table。每个算子在这里有一个条目，条目里按 DispatchKey 挂着它的各个实现——CPU、CUDA、Autograd、Meta……`at::add(x, y)` 被调用时，Dispatcher 根据输入 Tensor 的 DispatchKeySet 选一个实现（第五篇）；
- **实现**（`native/`）：`native_functions.yaml` 声明每个算子的 Schema 和它在各后端的实现函数名；`native/cpu/` 和 `native/cuda/` 是 Kernel 本身。Kernel 处理 stride 和广播靠 `TensorIterator`，处理矩阵乘、卷积则调用 cuBLAS / cuDNN / oneDNN 等厂商库。

Autograd 的实现也是挂在 Operator Table 上的一个 Key：Autograd Kernel 位于 `torch/csrc/autograd/generated/`，但它被注册进 ATen 的表里，由 Dispatcher 先于设备 Kernel 调用。所以 3.2 与 3.3 之间是双向的：`torch/csrc` 调用 ATen 的 API，同时把 Autograd 实现注册进 ATen 的表。

#### 3.4 `c10/`：核心基础库

c10（读作 "C-ten"，名字是 Caffe2 与 ATen 的双关）是所有层共同依赖的基础类型，自身不依赖任何上层，也不包含任何算子：

- `TensorImpl`、`StorageImpl`：Tensor 的元数据（sizes、strides、storage_offset、dtype、device）和底层存储——**第二篇讨论的全部内容，源码上住在这里**；
- `Device`、`DeviceGuard`、`Stream`、`Event`：设备抽象，CUDA 的具体实现在 `c10/cuda/`；
- `Allocator` 与 `CUDACachingAllocator`：内存分配器，第八篇显存分析的对象；
- `DispatchKey`、`DispatchKeySet`：Dispatcher 用来选实现的键，键的定义在这里，查表的逻辑在 ATen；
- `ScalarType`、`Scalar`、`Layout`：类型系统。

一个常见的误解是 c10 只是"设备抽象和分配器"。实际上它是 Tensor 的定义所在：用户最先接触的抽象，恰恰是源码里最底层的东西。这也是本系列把 Tensor 放在第二篇、而不是按源码自下而上从 c10 讲起的原因——它是一切的依赖，但理解它不需要先理解其他任何层。

#### 3.5 用 `torch.add` 把四层串一遍

第 2 节的动态路径可以精确落到目录上：

```text
torch/               Python 调用 torch.add(x, y)，进入 torch._C 中生成的绑定函数
        ↓
torch/csrc/          绑定函数解析参数，调用 at::add(x, y)
        ↓
aten/src/ATen/       Dispatcher 查 aten::add 的条目，按 x、y 的 DispatchKeySet 选 Key
        ↓
torch/csrc/          Autograd Key 命中：记录 AddBackward0 节点（torch/csrc/autograd/generated/），再次分发
        ↓
aten/src/ATen/       CPU 或 CUDA Key 命中：native/ 下的 add 实现，用 TensorIterator 遍历元素
        ↓
c10/                 结果 Tensor 的 TensorImpl 与 StorageImpl 在此构造，内存由 Allocator 分配
```

路径在 `torch/csrc` 与 ATen 之间往返一次，正是因为 Autograd 作为一个 Key 挂在 ATen 的表上。第五篇会把这条路径的每一步展开。

#### 3.6 组件、源码位置与系列篇章的对照

| 组件 | 源码位置 | 职责层 | 展开篇 |
|---|---|---|---|
| TensorImpl、StorageImpl、stride、dtype、device | `c10/core/` | 编程模型 | 第二篇 |
| Autograd 引擎、`grad_fn`、saved tensors | `torch/csrc/autograd/` | 编程模型 | 第三篇 |
| `nn.Module`、Optimizer、DataLoader、序列化 | `torch/nn/` `torch/optim/` `torch/utils/data/` | 用户代码与编程模型 | 第四篇 |
| Operator Schema、Dispatcher、native 算子、Codegen | `aten/src/ATen/` `torchgen/` | 算子运行时 | 第五篇 |
| pybind11 绑定、`TORCH_LIBRARY`、C++/CUDA 扩展 | `torch/csrc/` `torch/utils/cpp_extension.py` | Python 与 C++ 边界 | 第六篇 |
| Dynamo、AOTAutograd、Inductor、FX | `torch/_dynamo/` `torch/_functorch/` `torch/_inductor/` `torch/fx/` | 图与编译 | 第七篇 |
| Profiler、Caching Allocator、Stream、CUDA Graphs | `torch/profiler/` `c10/cuda/` `torch/cuda/` | 设备与通信 | 第八篇 |
| c10d、ProcessGroup、DDP、FSDP、DTensor | `torch/csrc/distributed/` `torch/distributed/` | 设备与通信 | 第九篇 |
| 测试基础设施、构建、CI、发布 | `test/` `torch/testing/` `tools/` `.github/` | 横切 | 第十篇 |

**本系列有意不覆盖的部分**：TorchScript / `torch.jit`（维护模式）；量化、稀疏 Tensor、复数等专门的 Tensor 子系统；`torch.func`（`vmap`、函数式变换）；MPS、XPU 等非 CUDA 后端的实现细节；`torch.export` 与 AOTInductor 只在第七篇作为编译栈的另一个出口简要提及。模型 Serving、请求调度和 KV Cache 属于推理系统，由《大模型推理系统揭秘：从 vLLM 看 LLM Serving Infra 核心技术》系列讨论。


## 五、PyTorch 工程中最重要的几个边界

### 1. Python 与 C++ 的边界

Python 适合：

- 表达模型结构；
- 组织训练流程；
- 处理配置和生命周期；
- 编写实验逻辑。

C++ 更适合：

- 实现运行时；
- 管理高性能数据结构；
- 连接设备后端；
- 实现低层算子；
- 处理对性能敏感的路径。

这不是“Python 慢、C++ 快”这么简单，而是不同层次的职责不同：

```text
Python：表达和组织
C++：运行时和抽象
CUDA：设备执行
```

### 2. 通用抽象与后端实现的边界

PyTorch 希望用户使用统一 Tensor API，但不同设备不可能完全没有差异：

- 某些算子只在部分后端支持；
- 不同设备的 dtype 能力不同；
- Kernel 的性能特征不同；
- 内存和通信模型不同；
- 编译器后端能力不同。

因此，真正健康的抽象不是假装所有设备完全相同，而是：

> 在统一语义之下，允许后端保留必要的实现差异。

### 3. 灵活性与可分析性的边界

Eager Mode 鼓励动态 Python，但编译器更喜欢稳定、可推断的程序。

```text
更多动态性 → 更好的表达能力
更多静态性 → 更好的分析和优化机会
```

`torch.compile()` 的工程价值就在于尝试在两者之间建立桥梁。但这座桥不是无条件成立的，graph break、动态 shape 和运行时 guard 都是需要理解的边界。

### 4. 可移植性与性能特化的边界

统一代码可以跨设备运行，但高性能通常需要特化：

```text
通用实现 → 易移植、易维护
设备特化 → 更高性能、更高维护成本
```

一个 AI-Infra 工程师需要能够判断：

- 哪些逻辑应该留在通用层；
- 哪些路径值得写专用 Kernel；
- 哪些优化只适合特定 shape；
- 哪些设备差异应该通过 Dispatcher 隔离。


## 六、本文小结

### 1. 一个定位

PyTorch 不是一个单纯的 Python 库，而是连接模型代码、Tensor 编程模型、算子运行时、设备后端、Kernel 和硬件的一套计算平台。Python 是它的表达层和控制层，C++ 是运行时和抽象层，CUDA 是设备执行层。

### 2. 三张地图

职责地图回答"谁负责什么"：

```text
用户模型与训练代码
    ↓
编程模型：Tensor / Autograd / nn.Module / Optimizer
    ↓
图与编译：FX / Dynamo / AOTAutograd / Inductor
    ↓
算子运行时：Operator Schema / Dispatcher / ATen
    ↓
设备与通信：CPU / CUDA / Meta 后端 · 内存分配 · stream · NCCL / Gloo
    ↓
Kernel 与硬件
```

动态地图回答"一次调用怎么走"：

```text
Python API → Python Binding → Operator Schema → Dispatcher → ATen Operator → Kernel → Hardware
```

代码地图回答"东西在哪个目录、哪个库"：

```text
torch/（Python）→ torch/csrc/（绑定、Autograd 引擎、c10d）→ aten/src/ATen/（Dispatcher、算子）→ c10/（TensorImpl、Device、Allocator）
```

本系列按前两张地图的顺序推进；第四章 §3 的对照表标出了每一篇在代码地图上的位置。后续每一篇都是这些地图上某一格的放大：第二篇放大 Tensor，第三篇放大 Autograd，第四篇放大 Module 与训练系统，第五、六篇放大算子运行时，第七篇放大图与编译，第八篇沿动态地图测量时间去了哪里，第九篇放大设备与通信层，第十篇讨论整张图如何被持续维护。

### 3. 四个边界

- Python 与 C++：表达和组织 vs 运行时和抽象；
- 通用抽象与后端实现：统一语义之下允许必要的实现差异；
- 灵活性与可分析性：动态 Python 与编译器之间的桥不是无条件成立的；
- 可移植性与性能特化：通用实现易维护，设备特化更快但更贵。

读后面各篇时遇到的大多数设计取舍，都可以归到这四个边界之一。


## 下一篇

[Tensor 与内存布局](/pytorch-tensor-and-memory-layout.html)
