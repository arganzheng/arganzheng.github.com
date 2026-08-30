---
layout: post
title: Python 内存管理与优化
subtitle: Python Memory Management and Optimization
tags: [Python]
catalog: true
---


在 AI-Infra 系统中，Python 通常不是执行密集计算的主体。模型推理、张量运算和部分数据处理，往往由 C、C++、CUDA 或其他原生运行时完成。

Python 更多承担以下职责：

- 组织请求、任务和配置对象；
- 管理数据结构与生命周期；
- 调用 NumPy、PyTorch 等原生库；
- 连接 CPU 内存、GPU 显存和外部服务；
- 协调模型、缓存、队列及资源句柄。

因此，Python 性能问题不一定表现为某段 Python 代码“计算得很慢”。很多时候，真正的问题来自：

- 创建了过多 Python 对象；
- 临时对象存活时间过长；
- 容器或缓存无限增长；
- 不小心进行了复制；
- Python 对象与原生数据之间发生了隐式转换；
- CPU 内存、原生内存和 GPU 显存的所有权边界不清晰。

本章聚焦 Python 工程师在 AI-Infra 场景下必须掌握的内存知识，包括对象模型、内存分配、复制语义、垃圾回收、缓存生命周期，以及 Python 与原生运行时之间的内存边界。

本章不重点讨论线程、进程、异步调度、动态批处理或 CPU 计算并行化。这些内容属于并发模型和任务协作的范畴。

## 一、为什么 AI-Infra 需要理解 Python 内存

一个典型的推理请求可能经历如下过程：

```text
网络请求
  → Python 请求对象
  → 输入校验与预处理
  → NumPy 数组或 PyTorch 张量
  → 原生推理运行时
  → 输出张量
  → Python 结果对象
  → JSON 或其他协议响应
```

在这条链路中，Python 可能参与了多次对象创建、格式转换和生命周期管理。

例如，下面的代码看起来只是进行一次简单的数据转换：

```python
payload = request.json()
values = payload["values"]

array = np.array(values, dtype=np.float32)
tensor = torch.from_numpy(array)
result = tensor.tolist()
```

但它背后可能发生了以下事情：

1. JSON 解析创建大量 Python `dict`、`list`、`float` 对象；
2. `np.array()` 将 Python 对象转换为连续的 NumPy 内存；
3. `torch.from_numpy()` 创建张量描述对象，并可能共享 NumPy 的底层存储；
4. 推理过程可能将数据复制到 GPU；
5. `tolist()` 又将底层数值转换回大量 Python 标量和列表对象；
6. JSON 序列化再次遍历这些对象。

如果请求数据规模较大，或者并发请求较多，内存峰值可能远高于输入数据本身。

因此，内存优化的第一步不是立即修改代码，而是回答以下问题：

- 当前对象由谁创建？
- 对象的底层数据存在哪里？
- 是否发生了复制？
- 对象会存活多久？
- 内存由 Python 管理，还是由原生库管理？
- 释放 Python 引用后，底层内存是否真的被归还？

## 二、Python 对象模型与内存开销

### 1. 名称、引用与对象

Python 变量本质上不是一个固定大小的内存槽位，而是指向对象的名称绑定。

```python
a = [1, 2, 3]
b = a

b.append(4)

print(a)
# [1, 2, 3, 4]
```

这里并没有创建两个列表。`a` 和 `b` 引用了同一个列表对象。

可以使用 `id()` 验证对象身份：

```python
print(id(a) == id(b))
# True
```

这意味着，理解 Python 内存时不能只看变量名，还要关注：

- 有多少个引用指向同一个对象；
- 对象是否可变；
- 引用是否被容器、闭包、任务或缓存长期持有。

### 2. Python 标量不是裸数据

在底层语言中，一个整数通常可以直接表示为固定大小的机器数据。但在 Python 中，整数是完整的对象：

```python
value = 42
```

这个对象除了保存数值本身，还需要保存类型信息以及解释器管理所需的元数据。

因此，大量 Python 标量组成的列表会产生较高的额外开销：

```python
values = [float(i) for i in range(1_000_000)]
```

这个列表包含：

- 列表自身的指针数组；
- 大量独立的 Python `float` 对象；
- 每个对象的类型和引用计数信息。

如果数据本质上是规则的数值集合，使用 NumPy 数组通常更适合：

```python
import numpy as np

values = np.arange(1_000_000, dtype=np.float32)
```

NumPy 数组通常将数据存储在连续的原生内存中，避免为每个数值创建独立的 Python 对象。

这并不意味着“所有列表都应该改成 NumPy 数组”。如果数据包含复杂的异构结构，例如请求元数据、配置项或插件描述，Python 容器仍然是合理选择。关键在于区分：

- 适合 Python 对象表达的数据；
- 适合连续数值内存表达的数据。

### 3. 容器自身也有额外开销

Python 容器通常保存的是对象引用，而不是对象内联数据。

```python
items = [1, 2, 3]
```

列表内部保存的是指向三个整数对象的引用。

对于字典而言，除了键和值对象本身，还需要维护哈希表结构：

```python
metadata = {
    "request_id": "abc",
    "model": "embedding",
    "priority": 1,
}
```

在请求量较大、对象生命周期较短的系统中，大量创建小型字典和列表会带来：

- 更高的内存占用；
- 更多的分配与释放操作；
- 更大的垃圾回收压力；
- 更高的对象遍历成本。

如果对象结构固定，可以考虑使用更紧凑的表示。

### 4. `__slots__` 减少实例字典

普通类实例通常拥有一个 `__dict__`，用于保存实例属性：

```python
class RequestContext:
    def __init__(self, request_id, model_name, deadline):
        self.request_id = request_id
        self.model_name = model_name
        self.deadline = deadline
```

当系统创建大量此类对象时，每个实例的属性字典都会产生额外开销。

可以使用 `__slots__`：

```python
class RequestContext:
    __slots__ = ("request_id", "model_name", "deadline")

    def __init__(self, request_id, model_name, deadline):
        self.request_id = request_id
        self.model_name = model_name
        self.deadline = deadline
```

`__slots__` 的主要作用是：

- 限定允许使用的属性；
- 通常减少实例内存占用；
- 避免为每个实例创建独立的 `__dict__`。

但它并不是无条件的优化。使用时需要注意：

- 不能随意添加未声明的属性；
- 某些依赖 `__dict__` 的工具可能无法正常工作；
- 继承关系复杂时需要谨慎设计；
- 如果对象数量很少，节省的内存可能没有实际意义。

对于大量创建、结构稳定、生命周期较短的请求元数据或任务节点，`__slots__` 比较适合。

### 5. 使用数据类表达结构化对象

对于结构化数据，可以使用 `dataclass` 提高可读性：

```python
from dataclasses import dataclass

@dataclass(slots=True)
class RequestContext:
    request_id: str
    model_name: str
    deadline: float
```

`slots=True` 可以让数据类使用槽位存储属性。

需要注意的是，数据类主要解决的是结构表达和代码维护问题。是否能够明显降低内存占用，仍然取决于对象数量、字段类型和实际生命周期。

## 三、复制、视图与对象共享

### 1. 赋值不是复制

```python
config = {"timeout": 1.0}
backup = config

backup["timeout"] = 2.0

print(config["timeout"])
# 2.0
```

`backup = config` 只增加了一个引用，并没有创建新的字典。

如果确实需要复制，可以使用浅拷贝：

```python
backup = config.copy()
```

此时修改顶层键值不会影响原字典：

```python
backup["timeout"] = 2.0
print(config["timeout"])
# 1.0
```

但浅拷贝只复制最外层容器：

```python
config = {
    "limits": {
        "max_tokens": 1024,
    }
}

backup = config.copy()
backup["limits"]["max_tokens"] = 2048

print(config["limits"]["max_tokens"])
# 2048
```

嵌套字典仍然是共享的。

### 2. 深拷贝可能代价很高

```python
import copy

backup = copy.deepcopy(config)
```

深拷贝会递归复制嵌套对象，可能导致：

- 更高的 CPU 开销；
- 更高的瞬时内存峰值；
- 复制不必要的数据；
- 对包含锁、文件句柄或原生资源的对象产生问题。

在请求路径中，不应为了“安全”而盲目使用 `deepcopy()`。更好的方式通常是：

- 明确哪些字段需要复制；
- 只复制会被修改的部分；
- 使用不可变对象；
- 构造新的轻量对象；
- 通过类型和接口限制修改范围。

例如：

```python
new_limits = {
    **config["limits"],
    "max_tokens": 2048,
}

new_config = {
    **config,
    "limits": new_limits,
}
```

这种方式虽然也会创建对象，但复制范围更加明确。

### 3. 切片可能创建副本

对于列表，切片会创建新的列表：

```python
items = list(range(1_000_000))
part = items[:500_000]
```

`part` 是一个新列表，包含新的引用数组。虽然列表中的元素对象可能仍然共享，但顶层容器已经被复制。

如果只需要遍历一段数据，可以考虑使用迭代器：

```python
from itertools import islice

part = islice(items, 500_000)
```

`islice()` 不会立即创建一个新的大列表，而是按需产生元素。

但需要注意，迭代器也可能持有原始对象的引用。如果原始列表很大，迭代器的生命周期同样不应过长。

### 4. 视图不等于副本

对于 NumPy 数组，某些切片操作会产生视图：

```python
import numpy as np

array = np.arange(10)
view = array[2:6]

view[:] = 100

print(array)
# [  0   1 100 100 100 100   6   7   8   9]
```

`view` 与 `array` 共享底层内存。

如果需要独立副本，需要显式调用：

```python
copy_array = array[2:6].copy()
```

视图能够减少内存复制，但也带来生命周期问题：一个很小的视图可能继续持有一个很大的底层数组。

```python
large_array = np.zeros(100_000_000, dtype=np.float32)
small_view = large_array[:10]
```

即使 `small_view` 只使用很少的数据，它仍然可能让 `large_array` 的底层内存保持存活。

如果只需要保留少量数据，可以显式复制：

```python
small_copy = large_array[:10].copy()
```

因此，视图优化需要同时考虑：

- 是否避免了不必要的复制；
- 视图是否会延长大对象生命周期；
- 共享数据是否可能被意外修改。

## 四、缓冲区协议与底层内存共享

### 1. `bytes`、`bytearray` 与 `memoryview`

处理网络数据、文件内容或二进制输入时，常见对象包括：

- `bytes`：不可变字节序列；
- `bytearray`：可变字节序列；
- `memoryview`：对已有缓冲区的视图。

```python
data = bytearray(b"abcdef")
view = memoryview(data)

view[0] = ord("X")

print(data)
# bytearray(b'Xbcdef')
```

`memoryview` 本身不需要复制整个底层缓冲区，可以让函数访问已有内存的一部分。

例如：

```python
def parse_header(buffer: memoryview) -> memoryview:
    return buffer[:16]

data = bytearray(1024)
header = parse_header(memoryview(data))
```

这类方式适合高频处理二进制数据的场景，例如：

- 网络协议解析；
- 文件分片；
- 二进制序列化；
- 音视频或图像缓冲区处理；
- 推理输入的预处理。

### 2. 零拷贝不是绝对概念

“零拷贝”通常意味着多个组件共享同一块底层内存，但它必须满足具体条件：

- 数据格式兼容；
- dtype 兼容；
- 内存布局兼容；
- 生命周期足够长；
- 接收方支持外部缓冲区；
- 不需要跨设备复制。

以下代码可能共享底层内存：

```python
import numpy as np
import torch

array = np.zeros(8, dtype=np.float32)
tensor = torch.from_numpy(array)

tensor[0] = 1.0

print(array[0])
# 1.0
```

但如果数据类型或接口不兼容，转换可能发生复制：

```python
array = np.zeros(8, dtype=np.float64)
tensor = torch.tensor(array, dtype=torch.float32)
```

这里通常需要创建新的 `float32` 存储，因为原数组是 `float64`。

因此，不能仅根据代码表面判断是否发生了复制。应当结合以下因素验证：

- 数据指针是否相同；
- dtype 是否一致；
- shape 和 stride 是否兼容；
- 数组或张量是否拥有独立存储；
- 转换接口的具体语义。

## 五、数据布局与隐式复制

### 1. 连续内存与非连续内存

底层数值库通常更喜欢连续内存。对数组转置或跨步切片后，得到的对象可能是非连续的：

```python
array = np.arange(12, dtype=np.float32).reshape(3, 4)
transposed = array.T

print(array.flags["C_CONTIGUOUS"])
# True

print(transposed.flags["C_CONTIGUOUS"])
# False
```

非连续数组不一定有问题。很多操作可以直接处理 stride 信息。但某些底层接口要求连续内存，此时可能自动创建一个连续副本。

可以显式检查：

```python
if not transposed.flags["C_CONTIGUOUS"]:
    transposed = np.ascontiguousarray(transposed)
```

显式转换的价值不只是“可能更快”，还在于：

- 复制发生的位置更明确；
- 更容易统计复制成本；
- 避免在不可控的底层调用中发生隐式复制；
- 便于在性能关键路径外完成准备工作。

### 2. dtype 转换也可能产生新内存

```python
array = np.zeros(1024, dtype=np.float64)
float32_array = array.astype(np.float32)
```

`astype()` 通常会创建新的数组，因为每个元素都需要重新编码。

如果 dtype 已经符合要求，可以使用：

```python
float32_array = array.astype(np.float32, copy=False)
```

但 `copy=False` 只是表达“尽量避免复制”，并不保证永远不复制。如果类型转换确实无法通过视图完成，底层仍可能创建副本。

类似地，`np.asarray()` 通常倾向于避免不必要的复制：

```python
array = np.asarray(source, dtype=np.float32)
```

但当输入类型或 dtype 不兼容时，仍然可能发生转换。

### 3. Python 列表与数组之间的边界

以下代码会将 Python 列表转换为 NumPy 数组：

```python
values = [1.0, 2.0, 3.0]
array = np.asarray(values, dtype=np.float32)
```

这个转换是必要的，因为列表中的 Python 对象并不是连续的 `float32` 内存。

相反，以下操作会将连续数组重新转换成大量 Python 对象：

```python
values = array.tolist()
```

在推理服务中，`tolist()` 很容易成为内存和延迟问题的来源，尤其是输出规模较大时。

如果协议或下游接口允许，应优先使用：

- NumPy 数组；
- PyTorch 张量；
- 二进制序列化格式；
- 支持数组的结构化协议。

如果最终必须返回 JSON，就应明确接受这一步转换的成本，并避免在链路中重复转换。

## 六、Python 与原生运行时的内存边界

### 1. Python 内存不等于进程内存

一个进程的内存通常来自多个来源：

- Python 对象；
- Python 解释器和标准库；
- NumPy、PyTorch 等扩展；
- C/C++ 分配的原生内存；
- 内存映射文件；
- 线程栈；
- GPU 驱动和设备缓存。

因此，`tracemalloc` 只能观察 Python 层的一部分分配，不能解释所有 RSS 增长。

例如：

```python
import tracemalloc

tracemalloc.start()

# 执行一段业务逻辑

current, peak = tracemalloc.get_traced_memory()
print(f"current={current}, peak={peak}")
```

如果 RSS 持续增长，但 `tracemalloc` 中没有对应增长，问题可能发生在：

- C/C++ 扩展；
- 深度学习框架；
- GPU 分配器；
- 内存映射；
- 系统级缓存。

### 2. CPU 内存与 GPU 显存是不同的资源

在深度学习系统中，至少需要区分：

1. Python 对象占用的内存；
2. 进程的 CPU 内存；
3. 原生库管理的 CPU 缓冲区；
4. GPU 显存；
5. GPU 框架的缓存分配器保留空间。

释放一个 Python 引用：

```python
del tensor
```

只表示当前 Python 作用域不再持有这个引用。它不一定意味着：

- 底层内存立即归还操作系统；
- GPU 缓存立即下降；
- 其他对象不再共享该存储；
- 异步设备操作已经完成。

如果还有其他引用存在，底层数据自然不会释放：

```python
outputs = model(inputs)
saved = outputs
del outputs
```

此时 `saved` 仍然持有输出对象。

### 3. 谨慎使用 `.cpu()`、`.numpy()` 和 `.tolist()`

这些操作可能改变数据所在的位置、表示形式或所有权关系：

```python
cpu_tensor = gpu_tensor.cpu()
array = cpu_tensor.numpy()
values = array.tolist()
```

这条链路可能包含：

- GPU 到 CPU 的数据传输；
- 张量与 NumPy 之间的共享或复制；
- NumPy 数值到 Python 标量的对象转换。

在调试代码中，下面这种写法尤其需要注意：

```python
print(gpu_tensor.cpu().tolist())
```

它可能为了打印少量信息而触发完整的数据传输和对象构造。

更稳妥的方式是：

```python
print(gpu_tensor.shape)
print(gpu_tensor.device)
print(gpu_tensor.dtype)
```

如果只需要查看少量值，应限制范围：

```python
print(gpu_tensor.flatten()[:8].cpu().tolist())
```

## 七、缓存、分配与对象生命周期

### 1. 无界缓存是常见的内存问题

下面的代码会让缓存持续增长：

```python
cache = {}

def remember(key, value):
    cache[key] = value
```

如果 `key` 持续变化，且没有淘汰策略，进程内存最终会不断上升。

更合理的方式是设置边界：

```python
from functools import lru_cache

@lru_cache(maxsize=1024)
def load_config(model_name: str):
    return build_config(model_name)
```

但 `lru_cache` 也不是自动解决方案：

- `maxsize` 仍然需要合理设置；
- 参数必须可哈希；
- 缓存值会被强引用；
- 大对象缓存可能显著增加内存峰值；
- 配置变化后需要考虑失效机制。

缓存设计必须同时考虑：

- 命中率；
- 单项大小；
- 总容量；
- 生命周期；
- 失效条件；
- 并发访问；
- 更新策略。

### 2. 闭包和回调可能延长对象生命周期

```python
def create_handler(large_model):
    def handler(request):
        return large_model.predict(request)

    return handler
```

`handler` 闭包持有 `large_model` 的引用。只要 `handler` 存在，模型对象就不会被释放。

这通常是有意的，但类似机制也可能造成意外保留：

- 任务回调捕获完整请求对象；
- 日志函数闭包捕获大数据；
- 异常对象保留局部变量；
- 全局列表保存历史任务；
- 调试代码保存完整输出。

在高并发服务中，应避免让短生命周期回调捕获不必要的大对象。

### 3. 任务对象和异常对象也可能持有引用

异步任务、Future、回调和异常对象可能间接持有：

- 请求上下文；
- 输入张量；
- 输出结果；
- 局部变量；
- traceback。

因此，任务完成后应及时清理不再需要的引用，避免将任务对象无限保存。

这并不意味着要到处手动调用 `gc.collect()`。频繁强制垃圾回收可能增加延迟，甚至降低吞吐。更重要的是：

- 缩短对象生命周期；
- 避免无界引用集合；
- 清理完成任务；
- 限制缓存；
- 解除不必要的闭包和回调引用。

### 4. 弱引用适合非所有权关系

当一个容器只需要观察对象，而不应该延长对象生命周期时，可以考虑 `weakref`：

```python
import weakref

class Model:
    pass

model = Model()
registry = weakref.WeakValueDictionary()
registry["default"] = model

del model
```

当对象没有其他强引用时，弱引用容器中的条目也会自动消失。

弱引用适合：

- 非所有权注册表；
- 对象索引；
- 调试辅助结构；
- 避免缓存反向持有对象。

但如果业务逻辑要求对象必须保持存活，就不应使用弱引用。

## 八、常见内存问题的排查方法

### 1. 先区分三类增长

看到进程内存上升时，首先要区分：

1. **仍然被业务对象引用**；
2. **分配器保留了内存，但对象已经释放**；
3. **原生库或设备运行时持有内存**。

它们的处理方法不同。

- 第一类需要查找引用链和生命周期；
- 第二类需要理解分配器行为；
- 第三类需要使用对应原生库的诊断工具。

不能仅凭 RSS 上升就判断发生了内存泄漏。

### 2. 使用 `sys.getsizeof()` 时要注意边界

```python
import sys

value = [1, 2, 3]
print(sys.getsizeof(value))
```

`sys.getsizeof()` 通常只返回对象自身的浅层大小，不会递归计算嵌套对象：

```python
items = [[1, 2, 3], [4, 5, 6]]
print(sys.getsizeof(items))
```

这个结果不包含嵌套列表和其中元素的完整大小。

因此，它适合做局部观察，不适合直接作为复杂对象的总内存统计工具。

### 3. 使用 `tracemalloc` 定位 Python 分配

```python
import tracemalloc

tracemalloc.start()

snapshot_before = tracemalloc.take_snapshot()

# 执行可能产生内存增长的操作

snapshot_after = tracemalloc.take_snapshot()

for statistic in snapshot_after.compare_to(
    snapshot_before,
    "lineno",
)[:10]:
    print(statistic)
```

它适合定位：

- 哪些 Python 代码行产生了较多分配；
- 哪些分配在操作前后持续增长；
- 哪些容器或对象可能被长期保留。

但它不能完整反映：

- NumPy 底层缓冲区；
- PyTorch 原生分配；
- GPU 显存；
- 所有 C 扩展分配。

### 4. 检查对象是否仍被引用

可以使用垃圾回收模块观察对象：

```python
import gc

unreachable = gc.collect()
print("unreachable objects:", unreachable)
```

也可以通过代码设计减少不必要的引用：

- 不保存完整请求；
- 只保存请求 ID；
- 不把输出张量挂在长期存活的任务对象上；
- 完成后清理临时缓存；
- 避免全局列表记录全部历史数据。

垃圾回收工具适合辅助诊断，但不应替代生命周期设计。

### 5. 观察进程级内存

进程 RSS 能够反映操作系统视角下的常驻内存，但不能直接告诉你增长来自哪里。

实际排查通常需要结合：

- Python 分配追踪；
- 进程 RSS；
- 原生库统计；
- GPU 内存统计；
- 请求数量和缓存大小；
- 长时间压力测试。

只有将这些指标放在同一时间线上，才能判断问题属于 Python 对象、原生内存还是设备缓存。

## 九、工程实践中的设计原则

### 1. 让数据表示匹配数据性质

- 结构化元数据使用 Python 对象；
- 大规模连续数值使用 NumPy 或张量；
- 二进制数据尽量使用缓冲区接口；
- 避免在 Python 列表和连续数组之间来回转换。

### 2. 明确复制边界

对以下操作保持敏感：

- `list(...)`；
- `dict(...)`；
- 列表切片；
- `copy.copy()`；
- `copy.deepcopy()`；
- `np.array()`；
- `astype()`；
- `.clone()`；
- `.cpu()`；
- `.numpy()`；
- `.tolist()`；
- 序列化与反序列化。

每次操作都应思考：

> 这里是共享数据、创建视图，还是产生了完整副本？

### 3. 控制对象生命周期

- 请求对象只存活在请求范围内；
- 不把大对象挂到全局状态；
- 完成任务后移除任务引用；
- 限制缓存容量；
- 避免闭包捕获无关对象；
- 不为了日志保存完整输入和输出。

### 4. 不要把所有内存问题归因于 Python

当内存增长时，应分别检查：

- Python 对象；
- C/C++ 扩展；
- NumPy 缓冲区；
- 深度学习框架缓存；
- GPU 显存；
- 操作系统分配器。

不同层的内存需要使用不同的工具和指标诊断。

### 5. 优先降低峰值，而不只是降低平均占用

AI-Infra 服务经常受到峰值内存限制。即使平均内存占用正常，以下情况仍可能导致 OOM：

- 多个请求同时创建大对象；
- 输入和输出短时间同时存活；
- 转换过程中同时存在原始数据和副本；
- 模型初始化期间存在临时权重；
- 缓存更新时产生新的大对象。

因此，应重点关注：

- 峰值内存；
- 并发请求下的最坏情况；
- 转换过程中的临时副本；
- 失败和超时路径上的资源释放。

## 十、一个简单的内存审查案例

假设服务中存在如下代码：

```python
def predict(payload):
    values = payload["values"]

    array = np.array(values, dtype=np.float32)
    tensor = torch.tensor(array, device="cuda")

    output = model(tensor)
    return output.cpu().numpy().tolist()
```

这段代码可能存在以下问题：

1. `payload` 中的 Python 列表本身占用较多对象内存；
2. `np.array()` 创建 NumPy 数组；
3. `torch.tensor()` 通常创建新的张量存储；
4. 张量被复制到 GPU；
5. `output.cpu()` 将结果复制回 CPU；
6. `.numpy()` 创建或暴露 NumPy 表示；
7. `.tolist()` 创建大量 Python 标量和列表对象。

这并不意味着每一步都可以直接删除，因为接口边界可能确实需要这些转换。但可以进行以下审查：

- 是否可以直接从协议读取更适合的数值格式；
- 是否可以使用支持共享内存的转换方式；
- 是否可以避免不必要的 dtype 转换；
- 是否必须把完整输出转成 Python 列表；
- 是否可以限制返回结果规模；
- 是否可以让下游接口直接处理数组或二进制数据；
- 是否需要同时保留输入和输出的多个表示。

优化后的代码可能类似：

```python
def predict(array: np.ndarray):
    if array.dtype != np.float32:
        array = array.astype(np.float32, copy=False)

    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array)

    tensor = torch.from_numpy(array).to(device="cuda")
    output = model(tensor)

    return output
```

这个版本并不适用于所有 API，但它展示了一个重要原则：

> 尽量在明确的边界完成必要转换，并避免在链路中反复改变数据表示。

## 十一、内存优化检查清单

在优化一条 Python AI-Infra 请求路径时，可以依次检查：

### 对象层面

- 是否创建了大量不必要的 Python 小对象？
- 是否可以使用数组或张量表示连续数值？
- 是否需要使用 `__slots__`？
- 是否存在大量临时字典、列表或元组？

### 复制层面

- 赋值是否被误认为复制？
- 是否发生了浅拷贝或深拷贝？
- 切片是否产生了新容器？
- NumPy 或张量操作是否创建了新存储？
- 是否存在重复的序列化和反序列化？

### 数据层面

- dtype 是否发生转换？
- 数组是否连续？
- 是否可以使用视图？
- 视图是否意外延长了大对象生命周期？
- 是否可以复用已有缓冲区？

### 生命周期层面

- 请求结束后，大对象是否仍被引用？
- 任务、Future 或回调是否保存了请求上下文？
- 缓存是否有容量和失效策略？
- 全局容器是否持续增长？
- 异常和日志对象是否保留了大对象？

### 运行时层面

- Python 内存是否与原生内存混淆？
- CPU 内存是否与 GPU 显存混淆？
- 释放引用后，底层分配器是否仍保留内存？
- 是否使用了正确层级的监控和诊断工具？

## 十二、总结

Python 内存管理的核心，不只是调用 `del` 或手动触发垃圾回收，而是理解以下几个层次：

```text
Python 名称与对象
  → 引用关系
  → 容器与实例开销
  → 复制、视图与共享
  → 内存分配与垃圾回收
  → 缓存和生命周期
  → 原生库与设备内存边界
```

在 AI-Infra 中，Python 工程师不一定需要自己实现底层内存分配器，但必须能够判断：

- 哪些对象由 Python 管理；
- 哪些数据存储在原生缓冲区中；
- 哪些操作会创建副本；
- 哪些引用会延长对象生命周期；
- 哪些内存由框架缓存；
- 哪些问题属于 Python，哪些问题属于原生运行时或设备运行时。

本章的核心结论是：

> **Python 内存优化的关键，不是简单地释放变量，而是减少不必要的对象创建和数据复制，明确对象生命周期，并正确理解 Python、原生库、进程和设备之间的内存所有权边界。**

对于 AI-Infra 系统而言，可以将这套方法概括为：

```text
用合适的数据结构表达数据
用引用关系理解对象生命周期
用显式边界识别复制行为
用视图和缓冲区减少不必要的分配
用有界缓存控制长期占用
用分层工具定位内存来源
```

当 Python 专注于组织和管理，而连续数据、底层缓冲区和设备资源由合适的原生运行时负责时，系统才能在保持工程灵活性的同时，避免不必要的内存开销。