---
layout: post
title: Python 核心机制与工程基础
tags: [Python]
catalog: true
---


Python 常被认为是一门“简单易学”的语言，但在 AI-Infra、深度学习框架、推理服务和分布式系统中，真正需要掌握的并不只是语法。

很多看似简单的代码，背后都依赖 Python 的运行时机制：

- `import` 可能触发注册、插件发现或 CUDA 扩展加载；
- `model(x)` 实际上可能调用了对象的 `__call__`；
- `with torch.inference_mode()` 背后是上下文管理协议；
- `for batch in loader` 依赖迭代器和生成器；
- `super()` 并不简单等于“调用父类方法”；
- 一个装饰器可能改变函数的执行逻辑、类型签名和异常栈；
- 一个异常是否被捕获，可能决定整个 Worker 是否退出；
- 一个模块是否已经存在于 `sys.modules`，可能影响注册逻辑是否执行。

因此，理解 Python 的核心机制，是阅读和设计 AI-Infra 代码的基础。

本文主要介绍以下内容：

1. 模块、包与导入机制；
2. 类、继承与对象模型；
3. 数据模型与对象协议；
4. 装饰器与函数对象；
5. 生成器与惰性计算；
6. 上下文管理器与资源生命周期；
7. 异常处理与资源安全。

最后通过综合示例、工程实践建议和总结，将这些机制串联起来。


## 一、模块、包与导入机制

### 1.1 模块和包是什么

在 Python 中，一个 `.py` 文件通常就是一个模块。

例如：

```text
project/
├── main.py
└── utils.py
```

其中 `utils.py` 是一个模块，可以在 `main.py` 中导入：

```python
import utils

utils.some_function()
```

包则是用于组织多个模块的目录：

```text
project/
└── mypackage/
    ├── __init__.py
    ├── model.py
    └── utils.py
```

这里：

- `mypackage` 是包；
- `mypackage.model` 是模块；
- `mypackage.utils` 是模块；
- `mypackage.__init__` 是包初始化模块。

现代 Python 支持没有 `__init__.py` 的命名空间包，但在大多数工程中，`__init__.py` 仍然具有重要作用。


### 1.2 `__init__.py` 的作用

`__init__.py` 常见作用包括：

#### 标识普通 Python 包

```text
mypackage/
└── __init__.py
```

#### 暴露公共接口

```python
# mypackage/__init__.py
from .model import Model
from .utils import load_config

__all__ = ["Model", "load_config"]
```

这样用户可以直接写：

```python
from mypackage import Model
```

而不需要：

```python
from mypackage.model import Model
```

#### 执行包初始化逻辑

```python
# mypackage/__init__.py
print("mypackage initialized")
```

但不建议在 `__init__.py` 中放置过重的逻辑，因为导入包时就会执行这些代码，可能带来：

- 导入速度变慢；
- 循环导入；
- 隐式副作用；
- CUDA 或系统环境检查提前执行；
- 不必要的依赖加载。

#### 声明公共 API：`__all__`

`__all__` 不仅影响 `from module import *` 的行为，更重要的作用是显式声明模块的公共接口：

```python
# mypackage/__init__.py
__all__ = ["Model", "load_config"]
```

它告诉使用者：

- 哪些对象是稳定接口；
- 哪些对象属于内部实现；
- 哪些名称不建议外部依赖。

实际工程中不建议使用通配符导入 `from module import *`，更推荐显式导入 `from module import Model, Runner`。



### 1.3 `import` 到底做了什么

执行：

```python
import mypackage.model
```

大致会经历以下过程：

1. 查找模块；
2. 创建模块对象；
3. 将模块放入 `sys.modules`；
4. 执行模块代码；
5. 将模块对象绑定到当前命名空间。

可以通过 `sys.modules` 查看已经加载的模块：

```python
import sys

print("math" in sys.modules)
```

需要注意，导入模块时，模块顶层代码会被执行：

```python
# config.py
print("loading config")

VALUE = 42
```

```python
import config
```

输出：

```text
loading config
```

这意味着，模块导入并不是一个纯粹的“声明依赖”操作，它可能产生副作用。

在 AI-Infra 中，导入模块可能触发：

- 算子注册；
- 后端注册；
- 插件发现；
- CUDA 扩展加载；
- 环境变量检查；
- 设备初始化；
- 日志系统配置；
- 全局缓存创建。

因此，阅读代码时不能只看函数调用，也要关注导入语句。



### 1.4 模块只执行一次

同一个模块在一个 Python 进程中通常只会执行一次。

```python
# example.py
print("module loaded")
```

```python
import example
import example
```

通常只会输出一次：

```text
module loaded
```

原因是第一次导入后，模块对象会被缓存到：

```python
sys.modules
```

再次导入时，Python 会优先从缓存中获取模块，而不会重新执行整个文件。

可以手动查看：

```python
import sys
import example

print(sys.modules["example"])
```

这条规则对注册机制非常重要。例如：

```python
# backend.py
BACKENDS = {}

BACKENDS["cpu"] = CPUBackend()
```

如果模块被重复执行，可能导致：

- 注册重复；
- 全局状态被覆盖；
- 单例失效；
- 资源被重复初始化。



### 1.5 `import module` 与 `from module import name`

两者有不同的命名空间行为。

```python
import math

math.sqrt(4)
```

这里导入的是模块对象，名称 `math` 被绑定到当前命名空间。

```python
from math import sqrt

sqrt(4)
```

这里直接将 `sqrt` 绑定到当前命名空间。

通常更推荐：

```python
import package
```

或者：

```python
import package.submodule
```

原因包括：

- 来源更清晰；
- 不容易发生名称冲突；
- 更容易通过模块路径理解代码；
- 避免大量名称污染当前命名空间。

当然，对于一些常用对象，可以使用：

```python
from pathlib import Path
from typing import Iterable
```



### 1.6 绝对导入与相对导入

绝对导入从顶层包开始：

```python
from mypackage.model import Model
```

相对导入从当前包位置开始：

```python
from .model import Model
from ..common import logger
```

其中：

- `.` 表示当前包；
- `..` 表示上一级包。

相对导入适合包内部模块之间的引用：

```python
# mypackage/runner.py
from .model import Model
```

绝对导入更适合公共代码或跨包依赖：

```python
from project.models import Model
```

工程中应尽量保持风格统一。混乱的导入方式容易导致：

- 循环导入；
- 直接运行脚本时报错；
- 包内路径不一致；
- 测试环境和生产环境行为不同。



### 1.7 `if __name__ == "__main__"`

每个 Python 模块都有一个特殊变量：

```python
__name__
```

当文件被直接执行时：

```bash
python train.py
```

该文件中的：

```python
__name__
```

通常等于：

```python
"__main__"
```

当文件被其他模块导入时：

```python
import train
```

此时：

```python
train.__name__ == "train"
```

因此可以写：

```python
def main():
    print("start training")


if __name__ == "__main__":
    main()
```

这样：

```bash
python train.py
```

会执行 `main()`，但：

```python
import train
```

不会自动执行训练逻辑。

这对于以下场景尤其重要：

- 命令行脚本；
- 多进程启动；
- 单元测试；
- 模块复用；
- 防止导入时执行副作用。



### 1.8 循环导入

循环导入是指模块之间相互依赖：

```python
# a.py
from b import func_b

def func_a():
    pass
```

```python
# b.py
from a import func_a

def func_b():
    pass
```

当导入 `a` 时：

1. Python 开始加载 `a`；
2. `a` 导入 `b`；
3. `b` 又尝试导入 `a`；
4. 此时 `a` 还没有执行完；
5. 可能出现部分初始化模块错误。

常见错误类似：

```text
ImportError: cannot import name ...
```

解决方式包括：

#### 调整模块依赖方向

将公共内容提取到第三个模块：

```text
a.py ──┐
       ├── common.py
b.py ──┘
```

#### 延迟导入

```python
def create_model():
    from .model import Model
    return Model()
```

#### 使用类型检查专用导入

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Model
```

这样运行时不会真正导入 `Model`，但类型检查器可以识别它。



### 1.9 动态导入与 `importlib`

有时模块名称只有在运行时才能确定：

```python
module_name = "mypackage.backends.cuda"
```

可以使用：

```python
import importlib

module = importlib.import_module(module_name)
```

AI-Infra 中常见用途：

- 按配置加载后端；
- 按设备类型加载实现；
- 自动发现插件；
- 延迟加载重量级依赖；
- 根据环境判断是否加载 CUDA 扩展。

例如：

```python
def load_backend(name):
    module = importlib.import_module(f"mypackage.backends.{name}")
    return module.create_backend()
```

动态导入增强了扩展性，但也带来一些问题：

- 依赖关系不容易被静态工具发现；
- 导入错误可能延迟到运行时；
- 调试和代码跳转更困难；
- 动态字符串来源不可信时会有安全风险。




## 二、类、继承与对象模型

### 2.1 类和实例

类可以理解为对象的模板，但 Python 中的类本身也是对象。

```python
class Runner:
    def run(self, batch):
        return batch
```

创建实例：

```python
runner = Runner()
```

此时：

- `Runner` 是类对象；
- `runner` 是 `Runner` 的实例；
- `runner.run` 是绑定到实例的方法。

可以检查：

```python
print(type(runner))
print(isinstance(runner, Runner))
```



### 2.2 实例属性和类属性

```python
class Counter:
    total = 0

    def __init__(self, name):
        self.name = name
```

其中：

- `total` 是类属性；
- `name` 是实例属性。

```python
a = Counter("a")
b = Counter("b")

print(a.name)
print(b.name)
print(a.total)
```

实例属性通常存储对象自己的状态，类属性通常用于：

- 常量；
- 默认配置；
- 类级别统计；
- 共享属性；
- 注册表。

需要警惕可变类属性：

```python
class Bad:
    items = []
```

所有实例都会共享同一个列表：

```python
a = Bad()
b = Bad()

a.items.append(1)

print(b.items)  # [1]
```

如果每个实例都需要独立列表，应放到 `__init__` 中：

```python
class Good:
    def __init__(self):
        self.items = []
```



### 2.3 方法绑定

```python
class Runner:
    def run(self, batch):
        return batch
```

调用：

```python
runner = Runner()
runner.run(data)
```

实际上近似于：

```python
Runner.run(runner, data)
```

`runner.run` 是一个绑定方法，Python 会自动将实例作为第一个参数传入。

这就是为什么实例方法通常需要写：

```python
def run(self, batch):
    ...
```

而不是：

```python
def run(batch):
    ...
```



### 2.4 `classmethod` 与 `staticmethod`

#### 实例方法

```python
class Model:
    def predict(self, x):
        ...
```

第一个参数是实例。

#### 类方法

```python
class Model:
    @classmethod
    def from_config(cls, config):
        return cls(config)
```

第一个参数是类对象，通常命名为 `cls`。

类方法适合：

- 工厂方法；
- 从配置创建对象；
- 从文件加载对象；
- 提供类级别操作。

#### 静态方法

```python
class MathUtils:
    @staticmethod
    def add(x, y):
        return x + y
```

静态方法不会自动接收实例或类。

它适合放置逻辑上属于某个类、但不依赖实例状态的方法。



### 2.5 `property`

`property` 可以把方法伪装成属性：

```python
class Model:
    def __init__(self, parameters):
        self.parameters = parameters

    @property
    def parameter_count(self):
        return len(self.parameters)
```

使用时：

```python
model.parameter_count
```

而不是：

```python
model.parameter_count()
```

也可以定义 setter：

```python
class Config:
    def __init__(self):
        self._device = "cpu"

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        if value not in {"cpu", "cuda"}:
            raise ValueError("invalid device")
        self._device = value
```

`property` 的价值在于：

- 隐藏内部实现；
- 在访问时进行计算；
- 对赋值进行校验；
- 保持类似字段的调用形式。

`property` 的底层实现依赖描述符协议，详见 3.8 节。



### 2.6 `__new__` 与 `__init__`

创建对象时，通常会经历两个阶段：

```python
obj = Class(...)
```

大致过程是：

1. `__new__` 创建实例；
2. `__init__` 初始化实例。

```python
class Example:
    def __new__(cls, *args, **kwargs):
        print("allocate object")
        return super().__new__(cls)

    def __init__(self, value):
        print("initialize object")
        self.value = value
```

`__new__` 更接近“创建对象”，常见于：

- 不可变类型的子类化；
- 单例；
- 对象缓存；
- 自定义实例创建；
- 元类或框架底层逻辑。

`__init__` 负责初始化已经创建好的实例。

一般业务代码只需要实现 `__init__`，不要轻易重写 `__new__`。



### 2.7 `__repr__`

`__repr__` 用于提供对象的开发者表示：

```python
class Device:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Device(name={self.name!r})"
```

```python
device = Device("cuda:0")
print(device)
```

输出：

```text
Device(name='cuda:0')
```

良好的 `__repr__` 对调试、日志和错误排查非常有帮助。

需要注意 `__repr__` 和 `__str__` 的区别：

- `__repr__` 面向开发者，用于调试和日志，交互式解释器和 `repr()` 调用它；
- `__str__` 面向用户，`print()` 和 `str()` 优先调用它；
- 如果只实现一个，应该实现 `__repr__`，因为 `__str__` 的默认实现会回退到 `__repr__`。

AI-Infra 中经常需要通过对象表示观察：

- 当前设备；
- Batch 大小；
- 模型配置；
- Worker 状态；
- 缓存状态；
- 请求 ID。



### 2.8 继承、组合与 Mixin

继承表示“是一种”关系：

```python
class BaseRunner:
    def run(self, batch):
        raise NotImplementedError


class CUDARunner(BaseRunner):
    def run(self, batch):
        return batch
```

组合表示“拥有”关系：

```python
class Runner:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
```

在工程代码中，组合通常比深层继承更容易维护。

Mixin 是一种用于复用局部能力的类：

```python
class LoggingMixin:
    def log(self, message):
        print(message)


class CacheMixin:
    def clear_cache(self):
        print("clear cache")


class Runner(LoggingMixin, CacheMixin):
    pass
```

Mixin 通常不代表一个完整的业务对象，而是提供某种能力。



### 2.9 MRO 与 `super()`

考虑以下代码：

```python
class Base:
    def run(self):
        print("Base")


class LoggingMixin:
    def run(self):
        print("Logging")
        super().run()


class Runner(LoggingMixin, Base):
    def run(self):
        print("Runner")
        super().run()
```

调用：

```python
Runner().run()
```

输出：

```text
Runner
Logging
Base
```

查看 MRO：

```python
print(Runner.__mro__)
```

大致结果：

```text
(Runner, LoggingMixin, Base, object)
```

`super()` 的含义不是“调用父类”，而是：

> 从当前类在 MRO 中的位置开始，寻找下一个符合条件的实现。

因此，在多继承和 Mixin 结构中，所有类都遵循协作式调用：

```python
super().run()
```

如果某个类绕过 `super()`，可能导致后续类的逻辑被跳过。



## 三、数据模型与对象协议

Python 中很多看起来像语法的行为，实际上是由特殊方法实现的。

### 3.1 常见语法和特殊方法

| Python 表达式 | 主要对应的方法 |
| --- | --- |
| `len(x)` | `x.__len__()` |
| `x[key]` | `x.__getitem__(key)` |
| `x[key] = value` | `x.__setitem__(key, value)` |
| `x + y` | `x.__add__(y)` |
| `x == y` | `x.__eq__(y)` |
| `hash(x)` | `x.__hash__()` |
| `x(...)` | `x.__call__(...)` |
| `with x` | `__enter__`、`__exit__` |
| `for item in x` | `__iter__`、`__next__` |
| `if x` | `__bool__` 或 `__len__` |

这些方法共同构成 Python 的数据模型。



### 3.2 可调用对象

任何实现了 `__call__` 的对象，都可以像函数一样调用：

```python
class Runner:
    def __init__(self, model):
        self.model = model

    def __call__(self, batch):
        return self.run(batch)

    def run(self, batch):
        return self.model(batch)
```

于是：

```python
runner(batch)
```

等价于：

```python
runner.__call__(batch)
```

在 AI-Infra 中，这种设计非常常见：

- 模型对象；
- 推理器；
- 数据预处理器；
- 后处理器；
- 调度器；
- Hook；
- Callback；
- Loss 对象。

使用可调用对象，可以同时保留：

- 调用接口；
- 配置状态；
- 缓存；
- 依赖对象；
- 生命周期管理。



### 3.3 迭代器与可迭代对象

可迭代对象可以被 `for` 遍历：

```python
for item in data:
    ...
```

迭代器通常需要实现：

```python
__iter__()
__next__()
```

示例：

```python
class CountDown:
    def __init__(self, start):
        self.current = start

    def __iter__(self):
        return self

    def __next__(self):
        if self.current <= 0:
            raise StopIteration

        value = self.current
        self.current -= 1
        return value
```

```python
for value in CountDown(3):
    print(value)
```

输出：

```text
3
2
1
```

`for` 循环会不断调用 `next()`，当收到 `StopIteration` 时结束。

AI-Infra 中大量组件都依赖迭代协议：

- Dataset；
- DataLoader；
- Batch 生成器；
- Token 流；
- 日志流；
- 请求队列；
- 分片数据读取器。



### 3.4 `__getitem__` 与容器协议

实现 `__getitem__` 后，对象可以支持下标访问：

```python
class Batch:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        return self.values[index]
```

```python
batch = Batch([1, 2, 3])
print(batch[0])
```

还可以支持切片：

```python
print(batch[1:])
```

AI-Infra 中，Batch、Tensor 包装器、缓存对象和配置对象都可能实现类似协议。



### 3.5 对象真值判断

以下表达式会触发对象的真值判断：

```python
if obj:
    ...
```

Python 会优先尝试调用：

```python
obj.__bool__()
```

如果没有，则可能使用：

```python
obj.__len__()
```

例如：

```python
class Queue:
    def __init__(self, items):
        self.items = items

    def __bool__(self):
        return bool(self.items)
```

需要注意，某些 Tensor 框架会禁止直接对多元素 Tensor 做布尔判断，因为这可能产生歧义：

```python
if tensor:
    ...
```

因此，在 AI-Infra 中，不要默认所有对象都可以安全地放入 `if`。



### 3.6 `__eq__` 与 `__hash__`

对象相等和对象身份是两个不同概念：

```python
a is b
a == b
```

- `is` 判断是否是同一个对象；
- `==` 调用相等性逻辑。

如果重写了 `__eq__`，通常需要谨慎处理 `__hash__`。

```python
class User:
    def __init__(self, user_id):
        self.user_id = user_id

    def __eq__(self, other):
        return isinstance(other, User) and self.user_id == other.user_id
```

如果对象是可变的，通常不应该让它作为字典键或集合元素，因为其哈希值不能在生命周期中变化。

需要特别注意：如果定义了 `__eq__` 但没有定义 `__hash__`，Python 会自动将 `__hash__` 设为 `None`，使对象变为不可哈希：

```python
class User:
    def __init__(self, user_id):
        self.user_id = user_id

    def __eq__(self, other):
        return isinstance(other, User) and self.user_id == other.user_id


user = User(1)
{user}  # TypeError: unhashable type: 'User'
```

如果对象确实需要放入集合或作为字典键，必须同时定义 `__eq__` 和 `__hash__`，并确保相等的对象具有相同的哈希值：

```python
class User:
    def __init__(self, user_id):
        self.user_id = user_id

    def __eq__(self, other):
        return isinstance(other, User) and self.user_id == other.user_id

    def __hash__(self):
        return hash(self.user_id)
```



### 3.7 属性访问：`__getattribute__` 和 `__getattr__`

`__getattribute__` 会拦截几乎所有属性访问：

```python
class DebugObject:
    def __getattribute__(self, name):
        print("access:", name)
        return object.__getattribute__(self, name)
```

实现时必须小心递归调用，通常使用：

```python
object.__getattribute__(self, name)
```

而不是：

```python
self.__dict__
```

`__getattr__` 只在正常属性查找失败后调用：

```python
class Config:
    def __getattr__(self, name):
        return None
```

这类机制常用于：

- 延迟加载；
- 兼容旧字段；
- 动态代理；
- 配置访问；
- 设备属性转发。

但过度使用会降低代码的可读性和静态分析能力。



### 3.8 描述符协议

描述符是实现了 `__get__`、`__set__` 或 `__delete__` 的对象。当描述符被作为类属性时，Python 会在属性访问时自动调用这些方法：

```python
class Typed:
    def __init__(self, expected_type):
        self.expected_type = expected_type
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name)

    def __set__(self, obj, value):
        if not isinstance(value, self.expected_type):
            raise TypeError(
                f"{self.name} expects {self.expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
        obj.__dict__[self.name] = value


class Config:
    batch_size = Typed(int)
    device = Typed(str)

    def __init__(self, batch_size, device):
        self.batch_size = batch_size
        self.device = device
```

```python
config = Config(32, "cuda")
config.batch_size = "big"  # TypeError: batch_size expects int, got str
```

描述符是 Python 数据模型中最底层的属性控制机制。前面介绍的 `property`、`classmethod`、`staticmethod` 以及方法绑定，底层都是通过描述符协议实现的。理解这一点有助于在阅读框架代码时理解各种"魔法"行为的来源。



## 四、装饰器与函数对象

### 4.1 函数也是对象

函数可以：

- 赋值给变量；
- 作为参数传递；
- 作为返回值；
- 存储在列表或字典中；
- 动态添加属性。

```python
def add(x, y):
    return x + y

operation = add
print(operation(1, 2))
```

这为装饰器、回调、注册表和高阶函数提供了基础。



### 4.2 闭包

当一个内部函数引用了外部函数的变量，并且外部函数已经返回时，这个内部函数就是一个闭包：

```python
def make_multiplier(factor):
    def multiply(x):
        return x * factor
    return multiply


double = make_multiplier(2)
print(double(5))  # 10
```

`double` 持有对 `factor` 的引用，即使 `make_multiplier` 已经返回。

闭包有一个常见陷阱——延迟绑定（late binding）：

```python
functions = []
for i in range(3):
    functions.append(lambda: i)

print([f() for f in functions])  # [2, 2, 2]，不是 [0, 1, 2]
```

`lambda` 中的 `i` 不是在定义时求值，而是在调用时查找。此时循环已结束，`i` 的值是 `2`。修复方式是用默认参数捕获当前值：

```python
functions = []
for i in range(3):
    functions.append(lambda i=i: i)

print([f() for f in functions])  # [0, 1, 2]
```

闭包是装饰器的基础——装饰器本质上就是"接受函数、返回闭包"的高阶函数。



### 4.3 基本装饰器

```python
from functools import wraps


def log_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("calling", func.__name__)
        return func(*args, **kwargs)

    return wrapper
```

使用：

```python
@log_call
def predict(x):
    return x * 2
```

等价于：

```python
def predict(x):
    return x * 2


predict = log_call(predict)
```

这说明装饰器是在函数定义阶段应用的，而不是每次调用时重新应用。



### 4.4 为什么要使用 `functools.wraps`

如果不使用 `wraps`：

```python
def decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper
```

被装饰函数的元信息可能丢失：

- 函数名变成 `wrapper`；
- 文档字符串丢失；
- 类型检查更困难；
- 调试和错误追踪信息变差；
- 反射工具获取的签名不准确。

推荐始终写：

```python
from functools import wraps
```



### 4.5 带参数的装饰器

带参数的装饰器实际上有两层函数：

```python
def retry(times):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(times):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if attempt == times - 1:
                        raise

        return wrapper

    return decorator
```

使用：

```python
@retry(times=3)
def request():
    ...
```

执行顺序可以理解为：

```python
request = retry(times=3)(request)
```

AI-Infra 中常见装饰器用途包括：

- 日志；
- 计时；
- 重试；
- 权限控制；
- 注册函数；
- 缓存；
- 自动同步/异步适配；
- 切换推理模式；
- 关闭梯度计算；
- 添加监控指标。

需要注意，装饰器会改变实际执行路径，因此阅读函数时必须同时查看它上方的装饰器。



## 五、生成器与惰性计算

### 5.1 `return` 和 `yield` 的区别

```python
def make_list():
    return [1, 2, 3]
```

调用后，列表会立即创建：

```python
values = make_list()
```

生成器函数则使用 `yield`：

```python
def make_generator():
    yield 1
    yield 2
    yield 3
```

调用时不会立即执行函数体：

```python
generator = make_generator()
```

只有在迭代时才开始执行：

```python
for value in generator:
    print(value)
```

生成器的核心特征是：

> 每次只产生一个结果，并保留上一次执行的位置。



### 5.2 惰性计算

生成器适合处理不能一次性全部加载到内存中的数据：

```python
def read_lines(path):
    with open(path) as file:
        for line in file:
            yield line
```

调用者可以逐行处理：

```python
for line in read_lines("large.txt"):
    process(line)
```

这比：

```python
lines = open("large.txt").readlines()
```

更节省内存。

AI-Infra 中常见场景包括：

- 流式 Token 输出；
- 大规模数据集读取；
- Batch 生成；
- 日志消费；
- 大文件处理；
- 请求流；
- 内存受限环境。



### 5.3 `yield from`

`yield from` 可以将一个可迭代对象的内容逐项转发：

```python
def combined():
    yield from [1, 2, 3]
    yield from [4, 5]
```

它也可以用于组合多个生成器：

```python
def read_dataset(dataset):
    for shard in dataset.shards:
        yield from read_shard(shard)
```



### 5.4 生成器生命周期

生成器支持：

```python
send()
throw()
close()
```

这些机制可以实现协程式控制流，但业务代码中不应为了“高级”而使用它们。

最常见的使用方式仍然是：

```python
for item in generator:
    ...
```

需要注意生成器的资源清理：

```python
def read_file(path):
    file = open(path)
    try:
        for line in file:
            yield line
    finally:
        file.close()
```

更推荐直接使用上下文管理器：

```python
def read_file(path):
    with open(path) as file:
        for line in file:
            yield line
```



## 六、上下文管理器与资源生命周期

### 6.1 `with` 的基本机制

以下代码：

```python
with resource() as value:
    use(value)
```

大致等价于：

```python
manager = resource()
value = manager.__enter__()

try:
    use(value)
except BaseException as exc:
    should_suppress = manager.__exit__(
        type(exc),
        exc,
        exc.__traceback__,
    )
    if not should_suppress:
        raise
else:
    manager.__exit__(None, None, None)
```

上下文管理器的核心价值是：

> 无论代码正常结束还是异常退出，都能执行清理逻辑。



### 6.2 自定义上下文管理器

```python
class Resource:
    def __enter__(self):
        print("acquire")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("release")
        return False
```

```python
with Resource() as resource:
    print("working")
```

输出：

```text
acquire
working
release
```

`__exit__` 返回：

- `False` 或 `None`：异常继续传播；
- `True`：异常被抑制。

一般不建议随意返回 `True`，否则可能吞掉重要错误。



### 6.3 `contextlib.contextmanager`

使用生成器可以更简洁地实现上下文管理器：

```python
from contextlib import contextmanager


@contextmanager
def resource():
    handle = acquire()
    try:
        yield handle
    finally:
        release(handle)
```

`yield` 前是进入逻辑，`yield` 后是退出逻辑。

适合管理：

- 文件；
- 锁；
- 数据库连接；
- 临时配置；
- GPU 资源；
- profiler；
- 通信环境；
- 推理模式。



### 6.4 AI-Infra 中的上下文管理器

例如：

```python
with torch.inference_mode():
    output = model(x)
```

这种写法通常意味着：

- 进入某种临时运行模式；
- 执行一段代码；
- 离开代码块后恢复原状态。

类似场景还包括：

```python
with autocast():
    output = model(x)
```

```python
with lock:
    update_shared_state()
```

```python
with profiler.profile():
    run_step()
```

阅读这类代码时，需要关注：

1. 上下文进入时修改了什么；
2. 退出时是否恢复；
3. 异常时是否仍然清理；
4. 是否存在嵌套上下文；
5. 是否支持异步版本。



### 6.5 异步上下文管理器

异步代码中使用：

```python
async with connection:
    await connection.send(data)
```

对应的方法是：

```python
__aenter__()
__aexit__()
```

它们通常是异步函数：

```python
class AsyncResource:
    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()
```



## 七、异常处理与资源安全

### 7.1 异常处理结构

Python 的异常处理结构包括：

```python
try:
    result = run()
except ValueError as exc:
    handle_value_error(exc)
else:
    process_success(result)
finally:
    cleanup()
```

执行规则：

- `try` 中发生异常时，寻找匹配的 `except`；
- 没有异常时执行 `else`；
- 无论是否异常，通常都会执行 `finally`。



### 7.2 不要随意捕获 `BaseException`

Python 的异常层级中：

```text
BaseException
├── SystemExit
├── KeyboardInterrupt
├── GeneratorExit
└── Exception
```

业务代码通常应该捕获：

```python
except Exception:
    ...
```

而不是：

```python
except BaseException:
    ...
```

因为捕获 `BaseException` 可能吞掉：

- 用户按下 `Ctrl+C`；
- 程序退出信号；
- 生成器关闭信号。



### 7.3 自定义异常

可以根据业务边界定义异常：

```python
class BackendError(Exception):
    pass


class BackendUnavailableError(BackendError):
    pass


class BackendExecutionError(BackendError):
    pass
```

这样调用者可以根据异常类型采取不同策略：

```python
try:
    run_backend()
except BackendUnavailableError:
    fallback_to_cpu()
except BackendExecutionError:
    retry()
```

好的异常设计应该表达：

- 发生了什么；
- 哪个组件失败；
- 是否可以重试；
- 是否应该切换后端；
- 是否需要终止任务。



### 7.4 异常链

如果在处理一个异常时抛出另一个异常，可以保留原始原因：

```python
try:
    load_config()
except OSError as exc:
    raise RuntimeError("failed to load configuration") from exc
```

这样错误信息会显示异常链，有助于定位根因。

相比之下：

```python
try:
    load_config()
except OSError:
    raise RuntimeError("failed to load configuration")
```

虽然也能抛出新异常，但原始异常上下文表达得不够明确。

如果需要显式断开异常链（例如封装内部实现细节，不想暴露底层异常），可以使用 `from None`：

```python
try:
    internal_operation()
except InternalError:
    raise PublicError("operation failed") from None
```

`from None` 会抑制原始异常的显示，只暴露新异常。



### 7.5 记录异常并重新抛出

常见工程写法：

```python
try:
    worker.run()
except Exception:
    logger.exception("worker failed")
    raise
```

这里的 `raise` 会重新抛出当前异常，并保留原始 traceback。

不建议这样写：

```python
except Exception as exc:
    logger.error(str(exc))
```

然后什么都不做，因为这会导致：

- 错误被吞掉；
- 上层误以为任务成功；
- Worker 继续运行在不一致状态；
- 分布式任务出现更难排查的问题。



### 7.6 异常和重试

不是所有异常都适合重试。

通常可以区分：

#### 可能适合重试

- 临时网络错误；
- 服务暂时不可用；
- 超时；
- 短暂资源不足；
- 临时连接断开。

#### 通常不适合重试

- 参数错误；
- 模型结构不匹配；
- 数据格式错误；
- 权限错误；
- CUDA 内核逻辑错误；
- 确定性的业务错误。

重试机制应当明确：

- 重试哪些异常；
- 最多重试多少次；
- 是否使用退避；
- 是否记录每次失败；
- 是否保证操作幂等；
- 最终失败后如何通知上层。



### 7.7 进程、线程和分布式任务中的异常

在普通函数中，异常通常沿调用栈向上传播。

但在并发环境中，异常传播会变得复杂：

- 子线程异常可能不会直接终止主线程；
- `Future.result()` 时才重新抛出任务异常；
- 子进程异常可能需要通过进程状态或队列传递；
- 分布式环境中一个 Rank 失败，其他 Rank 可能继续等待；
- 异步 Task 的异常如果没有被消费，可能只产生警告。

例如：

```python
future = executor.submit(run_task)

try:
    result = future.result()
except Exception:
    logger.exception("task failed")
```

AI-Infra 阅读并发代码时，应重点追踪：

1. 异常在哪里产生；
2. 异常在哪里被捕获；
3. 是否会重新抛出；
4. 谁负责取消其他任务；
5. 是否会触发资源清理；
6. 是否会导致其他 Worker 一直等待。



## 八、这些机制如何组合在一起

真实的 AI-Infra 代码通常不会只使用一种机制，而是将多种 Python 特性组合起来。

例如，一个简化的推理组件可能是：

```python
from contextlib import nullcontext
from functools import wraps


REGISTRY = {}


def registered(name):
    def decorator(cls):
        REGISTRY[name] = cls
        return cls

    return decorator


@registered("runner")
class Runner:
    def __init__(self, model, inference=True):
        self.model = model
        self.inference = inference

    def __call__(self, batch):
        context = inference_context() if self.inference else nullcontext()

        with context:
            return self.model(batch)

    def stream(self, batch):
        for output in self.model.generate(batch):
            yield output
```

这里同时使用了：

- 装饰器；
- 注册表；
- 类；
- `__call__`；
- 上下文管理器；
- `nullcontext`；
- 生成器；
- 延迟输出；
- 运行时组件发现。

阅读类似代码时，可以从几个问题入手：

### 代码是如何被加载的？

查看：

- 模块导入；
- 包初始化；
- 动态导入；
- 注册逻辑；
- 导入副作用。

### 对象如何被调用？

查看：

- `__call__`；
- 方法绑定；
- 代理对象；
- 装饰器；
- 继承关系。

### 数据如何流动？

查看：

- `__iter__`；
- 生成器；
- `yield`；
- Batch 协议；
- 流式输出。

### 资源如何管理？

查看：

- `with`；
- `try/finally`；
- 锁；
- GPU 上下文；
- 文件和连接；
- 进程退出逻辑。

### 失败后会发生什么？

查看：

- 异常类型；
- 捕获边界；
- 是否重试；
- 是否重新抛出；
- Worker 是否退出；
- 其他任务是否取消。



## 九、工程实践建议

### 9.1 减少导入副作用

避免在模块顶层执行过重逻辑：

```python
# 不推荐
model = load_large_model()
connect_to_database()
initialize_cuda()
```

更推荐显式初始化：

```python
def create_model():
    return load_large_model()
```

这样可以让初始化时机更加明确。



### 9.2 控制模块依赖方向

尽量保持依赖关系单向：

```text
基础模块 → 核心模块 → 服务模块
```

避免：

```text
a → b → c → a
```

如果两个模块互相需要对方的类型或工具，可以考虑：

- 抽取公共模块；
- 延迟导入；
- 使用 `TYPE_CHECKING`；
- 调整对象职责。



### 9.3 谨慎使用隐式魔法

以下机制虽然强大，但不应滥用：

- `__getattr__`；
- `__getattribute__`；
- 动态导入；
- 元类；
- 复杂装饰器；
- 运行时修改类；
- 全局注册表。

它们适合框架底层，但业务代码应优先选择：

- 显式函数调用；
- 清晰的依赖注入；
- 明确的接口；
- 可追踪的初始化流程。



### 9.4 保证资源清理

凡是涉及以下资源，都要考虑异常路径：

- 文件；
- Socket；
- 数据库连接；
- 锁；
- GPU 上下文；
- 临时文件；
- 进程；
- 线程；
- 分布式通信组。

优先使用：

```python
with ...
```

或者：

```python
try:
    ...
finally:
    cleanup()
```



### 9.5 让对象表示有助于调试

为重要对象实现清晰的 `__repr__`：

```python
class Request:
    def __repr__(self):
        return (
            f"Request(id={self.request_id!r}, "
            f"batch_size={self.batch_size}, "
            f"device={self.device!r})"
        )
```

日志中看到完整对象状态，往往比单独打印多个字段更容易排查问题。



## 十、总结

Python 核心机制可以概括为五个问题：

### 1. 代码如何组织和加载？

需要理解：

- 模块；
- 包；
- 导入；
- `sys.modules`；
- 循环导入；
- 动态导入；
- 导入副作用。

### 2. 对象如何创建和协作？

需要理解：

- 类；
- 实例；
- 属性；
- 方法绑定；
- 继承；
- 组合；
- MRO；
- `super()`。

### 3. 语法背后发生了什么？

需要理解：

- 特殊方法；
- 迭代协议；
- 容器协议；
- 可调用协议；
- 上下文管理协议；
- 属性访问协议。

### 4. 任务如何延迟执行和组合？

需要理解：

- 函数对象；
- 闭包；
- 装饰器；
- 生成器；
- `yield`；
- 惰性计算。

### 5. 资源和错误如何管理？

需要理解：

- `with`；
- `try/except/finally`；
- 异常链；
- 重试；
- 取消；
- 超时；
- 并发任务中的异常传播。

掌握这些内容后，阅读 AI-Infra 代码时就不再只是“逐行翻译语法”，而是能够理解：

- 一个组件为什么会被自动加载；
- 一个对象为什么可以像函数一样调用；
- 一个 Batch 为什么可以被遍历和切片；
- 一个上下文管理器修改了什么状态；
- 一个异常会不会导致 Worker 退出；
- 一个注册表是如何在运行时建立起来的。

这正是从“会写 Python”走向“能够理解 Python 工程和 AI-Infra 框架”的关键一步。