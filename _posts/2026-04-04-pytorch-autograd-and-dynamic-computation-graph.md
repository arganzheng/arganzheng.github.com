---
layout: post
title: "PyTorch 深度实践（03）：自动求导与动态计算图"
subtitle: "Autograd and Dynamic Computation Graphs in PyTorch"
tags: [PyTorch, AI, AI-Infra]
catalog: true
---

> 本文是[《PyTorch 深度实践：从 Tensor 到深度学习运行时》](/deep-dive-into-pytorch.html)系列的第三篇（共十篇）。上一篇：[Tensor 与内存布局](/pytorch-tensor-and-memory-layout.html)　下一篇：[`nn.Module` 与训练系统](/pytorch-module-and-training-system.html)

上一篇介绍了 Tensor 的核心模型：它不是一组孤立的数字，而是由 Storage、Shape、Stride、Storage Offset、dtype、device 和 layout 共同描述的一种数据抽象。

但 Tensor 只有数据和布局，还不能完成模型训练。训练还需要回答一个问题：

> **模型输出发生变化时，参数应该沿着什么方向、以多大的幅度变化？**

这需要计算梯度。PyTorch 通过 Autograd 把数学上的求导过程变成了一个可以执行的运行时系统：

```text
Tensor 运算
    ↓
动态记录运算关系
    ↓
形成计算图
    ↓
从结果反向遍历
    ↓
计算并累积梯度
    ↓
Optimizer 更新参数
```

初学者通常只需要记住：

```python
loss.backward()
optimizer.step()
```

但 AI-Infra 工程师还需要理解：

- `requires_grad` 决定了什么？
- leaf Tensor 和 non-leaf Tensor 有什么区别？
- `grad_fn` 指向什么？
- 梯度为什么会累积？
- 计算图什么时候创建，什么时候释放？
- 为什么保存一个 Tensor 可能让整张图无法回收？
- `detach()`、`no_grad()` 和 `inference_mode()` 有什么区别？
- 自定义算子如何向 Autograd 提供 backward？
- 为什么某些 in-place 操作会破坏反向传播？

本文会先从链式法则和向量—雅可比积开始，再逐层解释动态计算图、Autograd Node、梯度累积、图的生命周期和自定义 `autograd.Function`。最后会实现一个 Mini-Autograd，用一个很小的系统复现 PyTorch 反向传播的核心思想。


## 一、从数学求导到自动求导

### 1. 梯度解决什么问题？

假设有一个非常简单的函数：

```text
y = x²
```

当 `x = 3` 时，`y = 9`。如果希望调整 `x` 让 `y` 变小，就需要知道：

```text
y 对 x 的变化有多敏感？
```

这个敏感程度就是导数：

```text
dy/dx = 2x
```

当 `x = 3` 时：

```text
dy/dx = 6
```

如果目标是最小化 `y`，就可以沿着负梯度方向调整 `x`：

```text
x_new = x - learning_rate × gradient
```

神经网络中的参数通常有数百万甚至数十亿个，手工为每个参数推导和实现梯度是不现实的。Autograd 的任务就是自动完成这个过程。

### 2. 链式法则

对于复合函数：

```text
z = f(y)
y = g(x)
```

有：

```text
dz/dx = dz/dy × dy/dx
```

例如：

```text
y = x²
z = 3y + 1
```

可以拆成：

```text
dz/dy = 3
dy/dx = 2x

dz/dx = 3 × 2x = 6x
```

这正是反向传播的基本结构：从结果开始，沿着计算关系反向应用局部导数。

### 3. 从标量到向量

实际模型很少只有一个标量输入和一个标量输出。更常见的情况是：

```text
y = f(x)
```

其中 `x` 和 `y` 都是向量或 Tensor。

严格来说，这时的导数由 Jacobian 描述：

```text
J[i, j] = ∂y[i] / ∂x[j]
```

如果直接保存完整 Jacobian，内存和计算成本可能非常高。神经网络训练中通常不需要显式构造完整 Jacobian，而是计算向量—雅可比积：

```text
vᵀJ
```

其中 `v` 是从后续计算传回来的梯度。PyTorch 的 `backward()` 正是沿着计算图计算这种反向量积。

### 4. 为什么 backward 通常从标量 Loss 开始？

```python
loss.backward()
```

这里的 `loss` 通常是一个标量。标量的反向传播可以理解为：

```text
∂loss / ∂每个参数
```

如果对一个非标量 Tensor 直接调用 `backward()`：

```python
x = torch.randn(3, requires_grad=True)
y = x * 2
y.backward()
```

通常会报错，因为 PyTorch 不知道应该从哪个方向将 `y` 的梯度传回来。

可以显式提供外部梯度：

```python
y.backward(torch.ones_like(y))
```

这相当于计算：

```text
sum(y) 对 x 的梯度
```

所以 `backward()` 不只是“计算这个 Tensor 的梯度”，更准确地说是：

> 从当前结果出发，给定一个上游梯度，沿图计算各个输入的向量—雅可比积。


## 二、动态计算图：每次执行都记录一条新路径

### 1. 什么是计算图？

考虑下面的计算：

```python
x = torch.tensor(2.0, requires_grad=True)
y = x * x
z = y + 3
loss = z.sum()
```

可以表示为：

```text
x ─────┐
       * → y ───┐
x ─────┘        + → z → sum → loss
                3 ───┘
```

图中的节点可以表示：

- 输入 Tensor；
- 中间 Tensor；
- 算子；
- 输出 Tensor；
- 反向传播需要的局部信息。

### 2. Eager Mode 下的图是动态创建的

上一篇介绍过，PyTorch 默认采用 Eager-first 的编程体验。在 Eager Mode 下，代码执行到哪一行，对应的 Tensor 操作就可以立即执行；如果需要梯度，Autograd 同时记录必要的反向关系。

```python
x = torch.randn(3, requires_grad=True)
y = x * 2
z = y.relu()
loss = z.sum()
```

大致过程是：

```text
执行 x * 2
    ↓
创建乘法结果 y
    ↓
记录 y 如何由 x 得到

执行 relu
    ↓
创建 z
    ↓
记录 z 如何由 y 得到

执行 sum
    ↓
创建 loss
    ↓
记录 loss 如何由 z 得到
```

计算图不是预先写死的一张全局静态图，而是当前这次 forward 执行产生的运行时结构。

### 3. 每次 forward 通常都会创建新图

```python
for inputs, targets in loader:
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
```

通常每个训练 step 都会经历：

```text
本次 forward 创建图
    ↓
本次 backward 遍历图
    ↓
图中的中间信息被释放
    ↓
下一次 forward 创建新图
```

这种设计让 Python 控制流可以直接影响计算图：

```python
def f(x: torch.Tensor) -> torch.Tensor:
    if x.sum() > 0:
        return x * x
    return x + x
```

不同输入可能走不同分支，因此不同执行过程可能产生不同的计算关系。

### 4. 动态图的优点和成本

优点：

- Python 控制流自然可用；
- 中间结果容易检查；
- 模型结构容易修改；
- 失败位置通常比较直观；
- 适合研究和快速迭代。

成本：

- 每次执行都可能构建运行时图；
- Python 调度和对象创建有额外开销；
- 保存中间 Tensor 可能延长图生命周期；
- 编译器需要额外捕获和分析；
- 动态控制流可能导致 graph break。

`torch.compile()` 的一个重要目标，就是在保留这种编程体验的同时，捕获其中适合优化的计算部分。它并不改变 Autograd 的基本数学含义，但可能改变计算图被捕获和执行的方式。


## 三、`requires_grad`、Leaf Tensor 与 `grad_fn`

### 1. `requires_grad`

```python
x = torch.tensor(2.0, requires_grad=True)
```

这表示：只要后续计算在梯度记录开启的上下文中进行，并且结果参与反向传播，Autograd 就需要追踪与 `x` 有关的计算关系。

```python
x = torch.tensor(2.0, requires_grad=True)
y = x * x
z = y + 1
z.backward()

print(x.grad)  # tensor(4.)
```

这里：

```text
y = x²
z = y + 1
∂z/∂x = 2x = 4
```

`requires_grad=True` 并不意味着 Tensor 现在已经有梯度，也不意味着每个后续操作都一定会被记录。是否记录还会受到：

- 输入是否需要梯度；
- 是否处于 `no_grad()` 或 `inference_mode()`；
- 运算是否可微；
- Tensor 是否参与当前计算结果；

等条件影响。

### 2. Leaf Tensor

一个常见的 leaf Tensor 是用户直接创建、并设置了 `requires_grad=True` 的 Tensor：

```python
x = torch.tensor(2.0, requires_grad=True)
print(x.is_leaf)  # True
```

由运算产生的结果通常不是 leaf：

```python
y = x * 2
print(y.is_leaf)  # False
```

模型参数通常也是 leaf Tensor：

```python
from torch import nn

layer = nn.Linear(4, 2)
for parameter in layer.parameters():
    print(parameter.is_leaf)
```

Autograd 默认会把最终梯度保存在 leaf Tensor 的 `.grad` 中。对于 non-leaf Tensor，虽然它参与了计算图，但默认不一定保留 `.grad`。

### 3. 为什么 non-leaf 的 `.grad` 可能是 `None`？

```python
x = torch.tensor(2.0, requires_grad=True)
y = x * 2
z = y * 3
z.backward()

print(x.grad)  # tensor(6.)
print(y.grad)  # 通常为 None
```

如果确实需要查看 non-leaf Tensor 的梯度，可以在 backward 前调用：

```python
y.retain_grad()
z.backward()

print(y.grad)
```

这会要求 Autograd 额外保留该梯度，因此调试时可以使用，生产热路径中不要无差别对大量中间 Tensor 调用 `retain_grad()`。

### 4. `grad_fn`

```python
x = torch.tensor(2.0, requires_grad=True)
y = x * 2

print(x.grad_fn)  # None
print(y.grad_fn)  # 类似 <MulBackward0 ...>
```

leaf Tensor 通常没有由其他 Tensor 运算产生的 `grad_fn`；non-leaf Tensor 通常会带有指向反向节点的 `grad_fn`。

可以通过 `next_functions` 观察部分反向关系：

```python
print(y.grad_fn.next_functions)
```

这些对象属于 Autograd 的运行时实现细节，不应该依赖具体类名编写业务逻辑。它们的价值主要在于调试和理解计算图。

### 5. requires_grad 与 Parameter

`nn.Parameter` 是 Tensor 的一个特殊封装，目的是告诉 `nn.Module`：

```text
这个 Tensor 是模型参数
```

```python
from torch import nn

weight = nn.Parameter(torch.randn(4, 2))
print(weight.requires_grad)  # True
```

当 Parameter 被赋值为 Module 的属性时，Module 会将它注册到参数集合中。Optimizer 再从 Module 的参数中取得它们并更新。

这说明两个机制是分开的：

```text
requires_grad
    → 是否参与梯度计算

Parameter 注册
    → 是否被 Module 和 Optimizer 发现
```

一个普通 Tensor 可以参与梯度计算，但不会因为 `requires_grad=True` 就自动成为模型参数。


## 四、`backward()`：反向传播与梯度累积

### 1. 一次 backward 的过程

```python
x = torch.tensor(2.0, requires_grad=True)
y = x * x
loss = y + 1
loss.backward()
```

可以抽象为：

```text
loss
  ↓ 反向
加法节点
  ↓ 反向
乘法节点
  ↓ 反向
x
```

每个节点需要做两件事：

1. 接收从后继节点传来的上游梯度；
2. 使用自己的局部导数，计算传给前驱节点的梯度。

对于：

```text
 y = x * x
```

局部导数是：

```text
∂y/∂x = 2x
```

### 2. 梯度默认会累积

```python
x = torch.tensor(2.0, requires_grad=True)
y = x * 3

y.backward()
print(x.grad)  # tensor(3.)

y = x * 4
y.backward()
print(x.grad)  # tensor(7.)
```

第二次 backward 没有覆盖第一次梯度，而是累加了新的梯度：

```text
3 + 4 = 7
```

这也是为什么训练循环通常需要清空梯度：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

或者：

```python
optimizer.zero_grad(set_to_none=True)
```

### 3. 为什么梯度累积是有用的？

梯度累积并不只是一个容易忘记清零的陷阱，它也可以用于模拟更大的 batch：

```python
optimizer.zero_grad(set_to_none=True)

for micro_batch in micro_batches:
    loss = model_loss(micro_batch)
    loss.backward()

optimizer.step()
```

如果每个 micro-batch 的 loss 已经按需要缩放，那么多个 micro-batch 的梯度可以累积后再更新一次参数。

这在显存不足以容纳大 batch 时很有用，但需要同时考虑：

- loss 是否需要除以累积步数；
- BatchNorm 等状态是否仍按 micro-batch 更新；
- 梯度裁剪应该发生在累积之后还是每个 micro-batch；
- AMP 的 scaler 如何配合；
- 分布式训练中的梯度同步成本。

### 4. `optimizer.zero_grad(set_to_none=True)`

把梯度置零和把梯度设为 `None` 不是完全相同的操作：

```python
optimizer.zero_grad(set_to_none=True)
```

设为 `None` 可以避免不必要的填零，并让后续 backward 在需要时重新分配梯度。但业务代码不能无条件假设：

```python
parameter.grad is always a Tensor
```

正确性和性能都需要根据训练循环验证。

### 5. 多次 backward 与计算图释放

```python
x = torch.tensor(2.0, requires_grad=True)
y = x * x

y.backward()
y.backward()
```

第二次调用通常会失败，因为第一次 backward 后，为了节省内存，图中保存的中间信息已经被释放。

如果确实需要对同一张图多次反向传播，可以使用：

```python
y.backward(retain_graph=True)
y.backward()
```

但 `retain_graph=True` 会延长计算图生命周期、增加内存占用。它应该是有明确理由的选择，而不是遇到错误时盲目添加。


## 五、计算图中的保存值与生命周期

### 1. backward 为什么需要保存中间值？

有些算子的局部导数依赖 forward 阶段的输入或输出。

例如：

```text
y = x²
```

反向时需要：

```text
∂y/∂x = 2x
```

因此 Autograd 需要保留 `x`，或者保留足以计算梯度的信息。

对于 ReLU：

```text
 y = max(0, x)
```

反向时需要知道 forward 阶段哪些位置大于 0。不同算子会保存不同的中间信息。

这些被保存的中间值统称**激活值**（activation）。它们从 forward 保存到 backward 用完为止，是训练时显存占用中随 batch 和序列长度线性增长的那部分，通常比参数本身大得多。第八篇分析显存构成时，激活值是主要对象；那里的 Activation Checkpointing 做的事就是不保存、反向时重算。

### 2. 保存输出可能保存整张图

```python
losses = []

for inputs, targets in loader:
    loss = criterion(model(inputs), targets)
    losses.append(loss)
```

如果这些 `loss` 仍然连接着 Autograd 图，那么列表可能间接持有每个 step 的计算图和中间 Tensor，导致显存持续增长。

如果只需要记录数值，应转换为不再连接图的标量：

```python
losses.append(loss.detach().item())
```

如果需要保留 Tensor 但不需要梯度：

```python
losses.append(loss.detach().cpu())
```

### 3. `saved_tensors_hooks`

在某些内存受限场景，保存的 Tensor 可以通过 hooks 进行自定义处理，例如：

```python
from torch.autograd.graph import saved_tensors_hooks
```

概念上可以实现：

```text
保存到 GPU 的中间 Tensor
    ↓ pack
压缩或搬到 CPU
    ↓ unpack
backward 时恢复
```

这类机制可以换取显存空间，但会引入：

- 额外的数据搬运；
- 压缩和解压成本；
- 更复杂的生命周期；
- 可能的数值变化。

它适合明确的内存优化场景，不应作为默认编程方式。

### 4. 计算图与 Tensor 生命周期

需要同时区分两类生命周期：

```text
Tensor 对象生命周期
    由 Python 引用和 Storage 关系影响

Autograd 图生命周期
    由输出、grad_fn、保存值和 backward 过程影响
```

一个 Tensor 可能：

- 与其他 Tensor 共享 Storage；
- 通过 `grad_fn` 连接到图；
- 被某个 Python 容器长期持有；
- 让 Autograd 保存的中间值继续存活。

因此，内存问题不能只看变量名是否被删除，还要看 Tensor 是否仍然连接着 Storage 或计算图。


## 六、`detach()`、`no_grad()` 与 `inference_mode()`

### 1. `detach()`：切断一个 Tensor 的 Autograd 关系

```python
x = torch.tensor(2.0, requires_grad=True)
y = x * 2
z = y.detach()

print(y.requires_grad)  # True
print(z.requires_grad)  # False
```

`detach()` 返回一个不再沿原计算图传播梯度的 Tensor。它通常与原 Tensor 共享底层数据，因此：

```text
detach
    → 改变 Autograd 关系
    → 不等于 clone
    → 不等于数据复制
```

如果需要既切断梯度又创建独立数据，可以使用：

```python
z = y.detach().clone()
```

### 2. `torch.no_grad()`：临时关闭梯度记录

```python
with torch.no_grad():
    output = model(inputs)
```

在这个上下文中，通常不会为 Tensor 运算构建 Autograd 图，适合不需要训练的计算。

```python
model.eval()
with torch.no_grad():
    output = model(inputs)
```

这里的两个操作职责不同：

```text
model.eval()
    → 改变 Dropout、BatchNorm 等 Module 行为

no_grad()
    → 关闭梯度记录
```

不能用其中一个代替另一个。

### 3. `torch.inference_mode()`：更强的推理上下文

```python
with torch.inference_mode():
    output = model(inputs)
```

`inference_mode()` 面向纯推理场景，除了不记录梯度，还会进一步减少 Autograd 相关的元数据和版本计数开销。

它的约束也更强：在该模式中创建的 Tensor 不应被当作普通的、可以随时参与梯度计算的 Tensor 使用。适合明确隔离的推理路径，不适合包住一段之后还需要训练的混合逻辑。

### 4. 三者的区别

| 机制 | 是否创建新 Tensor | 是否复制数据 | 是否连接原计算图 | 典型用途 |
|---|---:|---:|---:|---|
| `detach()` | 是一个新的 Tensor 视图 | 通常否 | 否 | 将结果交给日志、缓存或非梯度逻辑 |
| `no_grad()` | 由内部操作决定 | 由内部操作决定 | 不记录新图 | 验证、评估、临时关闭梯度 |
| `inference_mode()` | 由内部操作决定 | 由内部操作决定 | 不记录新图且约束更强 | 独立的高性能推理路径 |
| `detach().clone()` | 是 | 是 | 否 | 需要独立数据和独立梯度关系 |

### 5. 与 `model.eval()` 的完整组合

推理代码通常写成：

```python
model.eval()

with torch.inference_mode():
    predictions = model(inputs)
```

训练代码通常写成：

```python
model.train()

with torch.enable_grad():
    predictions = model(inputs)
    loss = criterion(predictions, targets)
    loss.backward()
```

`train()`、`eval()`、`enable_grad()`、`no_grad()` 和 `inference_mode()` 分别控制不同的状态和上下文，不应该把它们当成同一类 API。


## 七、自定义 `autograd.Function`

### 1. 为什么需要自定义 Autograd？

如果使用已有的 PyTorch 算子，Autograd 通常已经知道它们的反向规则：

```python
y = torch.sin(x)
z = y * y
z.sum().backward()
```

但当你实现了一个新的算子，或者希望用特殊方式计算 backward，就需要告诉 Autograd：

```text
forward 如何计算结果？
backward 如何根据上游梯度计算输入梯度？
```

### 2. 一个平方算子

```python
class Square(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input * input

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output * 2 * input
```

调用：

```python
x = torch.tensor(3.0, requires_grad=True)
y = Square.apply(x)
y.backward()

print(x.grad)  # tensor(6.)
```

这里：

```text
forward：y = x²
局部导数：dy/dx = 2x
上游梯度：grad_output = dLoss/dy
输入梯度：dLoss/dx = grad_output × 2x
```

### 3. `ctx.save_for_backward()`

forward 阶段可以把 backward 所需的 Tensor 保存到 `ctx`：

```python
ctx.save_for_backward(input)
```

保存的 Tensor 会影响计算图和内存生命周期，因此不能把所有中间结果都无条件保存。

如果 backward 只需要一个标量配置，也可以保存普通属性：

```python
ctx.alpha = alpha
```

Tensor 和非 Tensor 状态的保存方式不同，应根据 backward 的需要选择。

### 4. 多输入和不可导输入

一个函数可能有多个输入：

```python
class Scale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.alpha = alpha
        return input * alpha

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.alpha, None
```

这里 `alpha` 如果是普通数值，就不需要返回梯度；backward 返回值的位置必须与 forward 输入对应。

### 5. 自定义 Autograd 的工程边界

自定义 backward 至少需要验证：

- forward 数值是否正确；
- backward 梯度是否正确；
- 多种 shape；
- 多种 dtype；
- CPU 和 CUDA；
- contiguous 和 non-contiguous；
- 空 Tensor 和边界输入；
- 一阶梯度；
- 二阶梯度是否支持；
- 异常输入；
- 内存是否正确释放。

可以使用：

```python
torch.autograd.gradcheck
```

和：

```python
torch.autograd.gradgradcheck
```

进行数值验证。


## 八、实现一个 Mini-Autograd

### 1. 实践目标

PyTorch 的 Autograd 内部包含复杂的 C++ 和 Python 组件，但可以用一个很小的系统复现核心思想：

```text
每个值知道它的父节点
每个运算保存局部梯度函数
从结果开始反向拓扑遍历
把梯度累加到父节点
```

为了聚焦计算图，本实践使用 Python 标量，不实现 Tensor、dtype、device 和广播。

### 2. Value 节点

```python
class Value:
    def __init__(self, data, parents=(), op=""):
        self.data = data
        self.grad = 0.0
        self.parents = set(parents)
        self.op = op
        self.backward_fn = lambda: None
```

每个 Value 包含：

```text
data
    当前数值

grad
    当前累积梯度

parents
    当前节点依赖的父节点

backward_fn
    当前节点如何把梯度传给父节点
```

### 3. 实现加法

```python
def add(left, right):
    result = Value(left.data + right.data, (left, right), "+")

    def backward():
        left.grad += result.grad
        right.grad += result.grad

    result.backward_fn = backward
    return result
```

因为：

```text
d(left + right)/dleft  = 1
d(left + right)/dright = 1
```

所以两个父节点都接收相同的上游梯度。

### 4. 实现乘法

```python
def multiply(left, right):
    result = Value(left.data * right.data, (left, right), "*")

    def backward():
        left.grad += right.data * result.grad
        right.grad += left.data * result.grad

    result.backward_fn = backward
    return result
```

因为：

```text
d(left × right)/dleft  = right
d(left × right)/dright = left
```

### 5. 拓扑排序

反向传播需要从结果开始，沿依赖关系逆序访问节点：

```python
def build_topological_order(root):
    visited = set()
    order = []

    def visit(node):
        if node in visited:
            return
        visited.add(node)
        for parent in node.parents:
            visit(parent)
        order.append(node)

    visit(root)
    return order
```

如果计算是：

```text
a → multiply → b → add → c
```

拓扑序是：

```text
a, b, c
```

反向遍历时则使用：

```text
c, b, a
```

### 6. 实现 backward

```python
def backward(root):
    for node in build_topological_order(root):
        node.grad = 0.0

    root.grad = 1.0

    for node in reversed(build_topological_order(root)):
        node.backward_fn()
```

一个更完整的实现通常会避免重复构建拓扑序，这里为了突出概念保持简单。

### 7. 运行一个例子

```python
a = Value(2.0)
b = Value(3.0)
c = multiply(a, b)
d = add(c, a)

backward(d)

print(d.data)  # 8.0
print(a.grad)  # 4.0
print(b.grad)  # 2.0
```

数学上：

```text
d = a × b + a

∂d/∂a = b + 1 = 4
∂d/∂b = a = 2
```

Mini-Autograd 只有很少的代码，却已经包含了 Autograd 的核心结构：

```text
Value
    ↓
父节点关系
    ↓
局部导数
    ↓
拓扑排序
    ↓
逆序传播
    ↓
梯度累积
```

### 8. Mini-Autograd 与 PyTorch Autograd 的差异

Mini-Autograd 没有实现：

- 多维 Tensor；
- Storage 和 stride；
- dtype 和 device；
- 广播；
- 原地操作检查；
- 高阶梯度；
- 多线程和异步设备执行；
- C++ Autograd Node；
- saved Tensor 生命周期管理。

它不是 PyTorch 的替代品，而是一个帮助理解反向传播的数据结构实验。


## 九、Autograd 常见问题与排查方法

### 1. `element 0 of tensors does not require grad`

常见原因：

- 输入没有设置 `requires_grad`；
- 模型参数被错误地冻结；
- forward 在 `no_grad()` 中执行；
- 中间结果被 `detach()`；
- 使用了不可导操作；
- 取出了 Python 标量后继续计算。

排查时可以沿路径打印：

```python
print(inputs.requires_grad)
print(outputs.requires_grad)
print(outputs.grad_fn)
print(loss.requires_grad)
print(loss.grad_fn)
```

### 2. `grad is None`

需要先区分：

```text
这是正常的 None
还是梯度链路断了？
```

可能原因：

- Tensor 是 non-leaf，默认不保留 `.grad`；
- Tensor 没有参与最终 loss；
- `requires_grad=False`；
- 中间调用了 `detach()`；
- backward 尚未执行；
- 梯度被设为 `None`。

对模型参数，可以检查：

```python
for name, parameter in model.named_parameters():
    print(name, parameter.requires_grad, parameter.grad is None)
```

### 3. 计算图被意外保留

典型风险包括：

```python
history.append(loss)
outputs_cache.append(outputs)
metrics[step] = hidden_state
```

如果这些对象还连接着图，可能造成内存不断增长。

根据需求选择：

```python
loss.item()
loss.detach()
loss.detach().cpu()
```

不要无条件使用 `detach()`，因为它会切断之后可能需要的梯度关系；也不要无条件使用 `clone()`，因为它会产生数据复制。

### 4. In-place 操作导致 backward 失败

```python
x = torch.randn(3, requires_grad=True)
y = x * x
x.add_(1)
```

如果 backward 需要 `x` 的旧值，而 `x` 已经被修改，Autograd 可能检测到版本不一致并报错。

遇到 in-place 相关错误时，先移除 in-place 操作验证正确性，再判断是否有必要通过更安全的方式优化内存。

### 5. 梯度异常和数值稳定性

Autograd 只负责按照定义计算梯度，不保证梯度一定数值稳定。出现 NaN 或 Inf 时，还要检查：

- 输入是否包含 NaN/Inf；
- loss 是否溢出；
- dtype 是否过低；
- 学习率是否过大；
- 指数、对数和除法是否处在危险范围；
- 混合精度和 GradScaler 是否配置正确。

可以使用：

```python
torch.autograd.set_detect_anomaly(True)
```

定位异常 backward，但它会增加大量开销，只适合调试阶段使用。


## 十、Java 工程师如何理解 Autograd

### 1. Autograd 不是普通事件回调

事件回调通常回答：

```text
某个事件发生后，调用哪些函数？
```

Autograd 回答的是：

```text
当前输出由哪些 Tensor 运算得到？
给定上游梯度，如何按照局部导数把梯度传回去？
```

它是一个带数学语义的计算图系统，不是任意业务事件的监听器。

### 2. Autograd 更接近运行时构建的反向程序

可以把一次 forward 和 backward 粗略理解为：

```text
forward
    产生数值结果
    同时记录反向所需的结构和数据

backward
    沿记录的结构逆序执行
    将上游梯度转换为下游梯度
```

这和 Java 编译器在编译期生成字节码不同，也和普通运行时调用栈不同。Autograd 图只描述当前计算中与梯度有关的部分。

### 3. 梯度累积类似显式状态管理

Java 工程师通常习惯把局部变量和对象状态区分开。在 PyTorch 中，`.grad` 是与参数生命周期相关的显式状态：

```text
forward 计算
    ↓
backward 写入 parameter.grad
    ↓
optimizer 读取 grad
    ↓
zero_grad 清理状态
```

如果不理解这个状态流，就容易出现：

- 梯度重复累积；
- 参数没有梯度；
- 梯度清零时机错误；
- 梯度裁剪作用在错误的阶段。

### 4. Tensor 布局和 Autograd 是两个正交维度

上一篇讨论了 Tensor 的：

```text
Storage / Shape / Stride / Offset / Dtype / Device
```

本文讨论了：

```text
requires_grad / grad_fn / Graph / Gradient
```

一个 Tensor 可以：

- 共享底层 Storage，但不共享 Autograd 关系；
- 连接同一张计算图，但拥有不同的 view 布局；
- 发生 dtype 或 device 转换，同时改变数据存储和梯度路径。

分析问题时，需要明确自己正在观察的是：

```text
数据布局问题
还是
梯度关系问题
```


## 十一、本文小结

Autograd 的核心任务，是把数学上的链式法则变成一次沿动态计算图执行的反向遍历。

### 1. 计算图的基本结构

```text
Tensor 运算
    ↓
记录父节点和局部导数
    ↓
形成动态计算图
    ↓
从结果节点逆序遍历
    ↓
计算和累积梯度
```

### 2. 关键对象

```text
requires_grad
    → 是否需要追踪梯度

leaf Tensor
    → 通常是用户或模型直接持有的参数节点

grad_fn
    → 当前 Tensor 对应的反向关系

.grad
    → 保存到 Tensor 或 Parameter 上的梯度状态
```

### 3. 三种常用的梯度控制方式

```text
detach()
    → 切断一个 Tensor 与原计算图的连接

no_grad()
    → 临时关闭新的梯度记录

inference_mode()
    → 面向纯推理的更强优化上下文
```

它们都不等价于 `model.eval()`。

### 4. 分析 backward 问题的顺序

```text
输入是否 requires_grad？
    ↓
中间结果是否仍有 grad_fn？
    ↓
loss 是否参与了目标参数的计算？
    ↓
是否被 detach 或 no_grad 截断？
    ↓
是否发生了 in-place 修改？
    ↓
是否正确处理梯度累积和清零？
```

### 5. Mini-Autograd 的核心

```text
父节点
    + 局部导数
    + 拓扑排序
    + 逆序遍历
    + 梯度累积
    = 自动求导的核心骨架
```

### 6. 本篇涉及的源码位置

本篇讨论的机制在源码中的位置（对应第一篇第四章 §3 的代码地图）：

| 路径 | 内容 |
|---|---|
| `torch/csrc/autograd/engine.cpp` | 反向执行引擎：依赖计数、ready queue、按设备的工作线程 |
| `torch/csrc/autograd/function.h` | `Node`：`grad_fn` 在 C++ 中的基类，`next_edges` 构成计算图 |
| `torch/csrc/autograd/variable.h` | `AutogradMeta`：`requires_grad`、`grad_`、`grad_fn_` 存放在这里，挂在 TensorImpl 上 |
| `torch/csrc/autograd/saved_variable.h` | `SavedVariable`：保存中间值与 version counter 检查（in-place 报错的来源） |
| `tools/autograd/derivatives.yaml` | 每个算子的导数公式 |
| `torch/csrc/autograd/generated/`（构建后才存在） | 由上者生成：`Functions.cpp`（反向节点）与 `VariableType*.cpp`（注册到 Autograd Key 的包装 Kernel） |
| `torch/autograd/function.py`、`torch/csrc/autograd/custom_function.h` | Python 与 C++ 的自定义 `autograd.Function` |
| `torch/autograd/graph.py`、`torch/autograd/gradcheck.py` | `saved_tensors_hooks`；`gradcheck` 的有限差分实现 |

下一篇将进入 Tensor 和 Autograd 之上的模型组织层：

> **`nn.Module` 如何管理模型层次、Parameter、Buffer、state_dict，并把这些对象连接到训练循环？**


## 下一篇

[`nn.Module` 与训练系统](/pytorch-module-and-training-system.html)
