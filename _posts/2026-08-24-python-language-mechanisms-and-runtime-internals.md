---
layout: post
title: Python 在 AI-Infra（01）：语言机制与运行时原理
subtitle: Python Language Mechanisms and Runtime Internals
tags: [Python]
catalog: true
---

Python 常被认为是一门“简单易学”的语言。但在 AI-Infra、深度学习框架、推理服务和分布式系统中，真正需要掌握的并不只是语法，而是语法背后的运行时模型。

很多看似简单的代码，实际都依赖 Python 的底层机制：

- `import` 可能触发算子注册、插件发现或 CUDA 扩展加载；
- `model(x)` 可能经过对象的 `__call__` 协议；
- `for batch in loader` 依赖可迭代对象和迭代器协议；
- `with torch.inference_mode()` 背后是上下文管理协议；
- 一个装饰器可能改变函数的实际调用路径；
- 一个生成器可能暂停执行并长期持有文件、连接或 Batch；
- 一个异常是否被重新抛出，可能决定 Worker 是否退出；
- 一个模块是否已经存在于 `sys.modules`，可能决定注册逻辑是否再次执行。

因此，理解 Python 的语言机制和运行时原理，是阅读、设计和排查 AI-Infra 系统的基础。

本文不试图完整介绍 Python 的所有特性，而是围绕一个核心问题展开：

> 一段 AI-Infra 代码从加载、创建对象、执行任务到释放资源，Python 运行时究竟做了什么？

全文主要讨论：

1. Python 的执行模型；
2. 名称、对象、函数和执行帧；
3. 模块、包与导入系统；
4. 类、属性查找与对象协议；
5. 闭包、装饰器与注册机制；
6. 生成器、惰性执行与资源生命周期；
7. 上下文管理器和异常传播；
8. 这些机制如何组合成 AI-Infra 组件。


## 一、先建立 Python 的运行时模型

### 1.1 Python 代码并不是“逐行直接执行”

很多初学者会把 Python 理解为：

> 解释器读取一行代码，然后立即执行这一行。

这是一种便于入门的近似说法，但不够准确。

以函数为例：

```python
def add(x, y):
    return x + y
```

Python 通常会经历以下过程：

1. 读取源代码；
2. 将源代码解析为语法树；
3. 编译为代码对象（code object）；
4. 创建函数对象；
5. 调用函数时创建执行帧（frame）；
6. 解释器执行代码对象中的字节码；
7. 返回结果并销毁或暂时保留执行帧。

可以使用 `dis` 模块查看函数对应的字节码：

```python
import dis


def add(x, y):
    return x + y


dis.dis(add)
```

不同 Python 版本输出的具体指令可能不同，但通常可以看到加载参数、执行运算和返回结果等操作。

这里需要区分几个概念：

- **源代码**：开发者编写的 `.py` 文件；
- **代码对象**：编译后的执行描述，包含字节码、常量、变量名等信息；
- **函数对象**：对代码对象、默认参数、注解和全局命名空间等内容的封装；
- **执行帧**：一次函数调用过程中的运行时状态；
- **解释器**：负责执行字节码并管理调用栈、异常和对象操作。

在 CPython 中，字节码最终由解释器执行。具体的字节码格式、执行循环和优化策略属于 CPython 实现细节，并不完全等同于 Python 语言规范。

### 1.2 函数也是运行时对象

函数不仅是一段“可执行代码”，它本身也是一个对象：

```python
def predict(x):
    return x * 2


print(type(predict))
print(predict.__code__)
print(predict.__defaults__)
print(predict.__annotations__)
print(predict.__dict__)
```

函数对象可以：

- 赋值给变量；
- 作为参数传递；
- 作为返回值；
- 存储在列表或字典中；
- 被装饰器替换；
- 动态添加属性。

例如：

```python
operation = predict
print(operation(3))
```

`operation` 和 `predict` 只是两个名称，它们都绑定到同一个函数对象。

这也是回调、装饰器、注册表和高阶函数的基础。

### 1.3 名称不是变量盒子

Python 中的赋值通常表示“名称绑定”，而不是把值复制到一个变量盒子中。

```python
x = []
y = x

y.append(1)

print(x)  # [1]
print(y)  # [1]
print(x is y)  # True
```

执行过程可以理解为：

1. 创建一个列表对象；
2. 名称 `x` 绑定到该对象；
3. 名称 `y` 也绑定到同一个对象；
4. 通过 `y` 修改对象；
5. 通过 `x` 观察到同一修改。

因此需要区分：

- 名称；
- 对象；
- 对象身份；
- 对象类型；
- 对象可变性。

```python
a = 1
b = 1

print(a == b)  # True
print(a is b)  # 具体身份不应作为数值相等的判断依据
```

`==` 通常表示值相等，`is` 判断是否为同一个对象。工程代码中应使用 `is None` 判断空值，而不要使用 `== None`。

在 AI-Infra 中，名称绑定和对象别名会影响：

- 配置对象是否被意外修改；
- Batch 是否被多个组件共享；
- 缓存是否引用了可变对象；
- 数据预处理是否改变了调用方持有的数据；
- 多个 Worker 是否误用同一个进程内状态。

### 1.4 函数调用与执行帧

考虑下面的代码：

```python
def worker(batch):
    result = preprocess(batch)
    return model(result)
```

调用 `worker(batch)` 时，运行时需要处理：

- 参数绑定；
- 局部变量创建；
- 函数全局变量查找；
- 可能的闭包变量查找；
- 嵌套函数调用；
- 返回值；
- 异常和 traceback。

一次函数调用对应一个执行帧。执行帧中包含当前调用的运行状态，例如：

- 局部变量；
- 当前指令位置；
- 全局命名空间；
- 内置命名空间；
- 调用链信息。

异常 traceback 能够显示函数调用链，本质上就是在展示相关执行帧的信息。

可以使用 `inspect` 观察当前帧：

```python
import inspect


def show_frame():
    frame = inspect.currentframe()
    print(frame.f_code.co_name)
    print(frame.f_locals)


show_frame()
```

生产代码不应频繁依赖 `inspect.currentframe()` 或手动修改帧对象，但理解执行帧有助于解释：

- 为什么 traceback 能定位调用路径；
- 为什么局部变量、全局变量和闭包变量的查找方式不同；
- 为什么装饰器可能改变错误栈；
- 为什么递归和深层调用会消耗调用栈资源。

### 1.5 变量查找：局部、闭包、全局和内置

Python 常用 LEGB 规则查找名称：

1. Local：当前局部作用域；
2. Enclosing：外层函数作用域；
3. Global：模块全局作用域；
4. Builtins：内置命名空间。

```python
value = "global"


def outer():
    value = "enclosing"

    def inner():
        value = "local"
        return value

    return inner()


print(outer())  # local
```

如果内部函数没有定义同名变量，就会继续查找外层作用域：

```python
def outer():
    value = "enclosing"

    def inner():
        return value

    return inner()


print(outer())  # enclosing
```

AI-Infra 中，闭包变量常用于保存：

- 后端名称；
- 重试次数；
- 配置参数；
- 指标标签；
- 模型实例；
- 缓存对象。

但隐式捕获变量也可能让依赖关系变得不明显，因此需要谨慎使用。


## 二、模块、包与导入系统

### 2.1 模块和包是什么

在 Python 中，一个 `.py` 文件通常就是一个模块：

```text
project/
├── main.py
└── utils.py
```

`utils.py` 可以在 `main.py` 中被导入：

```python
import utils

utils.some_function()
```

包用于组织多个模块：

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

现代 Python 也支持没有 `__init__.py` 的命名空间包，但大多数工程仍然使用 `__init__.py` 来组织公共接口和包初始化行为。

### 2.2 `__init__.py` 的作用

`__init__.py` 常见作用包括：

#### 暴露公共接口

```python
# mypackage/__init__.py
from .model import Model
from .utils import load_config

__all__ = ["Model", "load_config"]
```

于是用户可以写：

```python
from mypackage import Model
```

而不必了解 `Model` 实际位于哪个内部模块。

#### 执行包初始化逻辑

```python
# mypackage/__init__.py
print("mypackage initialized")
```

但不建议在 `__init__.py` 中放置过重逻辑。导入包时，这些顶层代码就会执行，可能导致：

- 导入速度变慢；
- 循环导入；
- CUDA 或系统环境检查提前执行；
- 大量依赖被提前加载；
- 全局资源被隐式创建。

#### 声明公共名称

```python
__all__ = ["Model", "load_config"]
```

`__all__` 可以表达模块希望公开的名称集合，主要影响：

- `from module import *`；
- 文档工具；
- IDE 和静态分析工具；
- API 可见性约定。

但 `__all__` 不是访问控制机制，也不能阻止调用者显式导入其他名称。

### 2.3 `import` 到底做了什么

执行：

```python
import mypackage.model
```

可以简化理解为以下过程：

1. 检查 `sys.modules` 中是否已有对应模块；
2. 如果没有，通过导入器协议查找模块；
3. 创建 `ModuleSpec`；
4. 根据 loader 创建模块对象；
5. 将模块对象放入 `sys.modules`；
6. 执行模块顶层代码；
7. 将结果绑定到当前命名空间。

可以查看已经加载的模块：

```python
import sys

print("math" in sys.modules)
```

模块顶层代码会在导入时执行：

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

因此，导入并不是纯粹的“声明依赖”操作，它可能产生副作用。

AI-Infra 中，导入可能触发：

- 算子注册；
- 后端注册；
- 插件发现；
- CUDA 扩展加载；
- 环境变量检查；
- 设备初始化；
- 日志系统配置；
- 全局缓存创建。

### 2.4 `sys.modules` 与导入缓存

同一个模块在同一个 Python 进程中，使用相同模块名并通过正常导入流程加载时，通常只会执行一次：

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

可以验证：

```python
import sys
import example

first = sys.modules["example"]

import example

second = sys.modules["example"]

print(first is second)  # True
```

但“只执行一次”不是绝对规则。以下情况可能导致模块代码再次执行：

```python
import importlib

import example
importlib.reload(example)
```

此外：

- 不同进程拥有各自的 `sys.modules`；
- 使用不同模块名加载同一文件，可能造成重复加载；
- 手动删除 `sys.modules` 中的条目后再次导入，可能重新执行；
- 测试环境和生产环境可能使用不同的导入路径。

这对注册机制尤其重要：

```python
# backend.py
BACKENDS = {}

BACKENDS["cpu"] = CPUBackend()
```

如果模块被重复执行，可能出现：

- 重复注册；
- 全局状态被覆盖；
- 单例失效；
- 资源被重复初始化。

### 2.5 导入器：finder、loader 和 `ModuleSpec`

Python 的导入系统并不只是“在目录里寻找 `.py` 文件”。

导入过程通常涉及：

- **finder**：寻找模块，并返回模块规格；
- **loader**：根据模块规格创建并执行模块；
- **ModuleSpec**：描述模块名称、来源、loader 和包信息；
- **import hook**：允许框架或工具扩展导入过程。

可以观察模块的规格：

```python
import json

print(json.__spec__)
print(json.__loader__)
print(json.__package__)
print(json.__file__)
```

这套机制使 Python 能够导入多种来源的模块，例如：

- 普通 `.py` 文件；
- 编译扩展；
- zip 包中的模块；
- 命名空间包；
- 动态生成的模块；
- 自定义导入器提供的模块。

这也是插件系统、模型后端发现和某些框架自动注册机制的基础。

### 2.6 `import module`、`import module as name` 与 `from module import name`

Python 中常见的导入方式有三种：

```python
import math
import numpy as np
from math import sqrt
```

它们的区别主要在于：**导入后，什么名称会被绑定到当前模块的命名空间中**。

#### 2.6.1 `import module`

```python
import math

math.sqrt(4)
```

当前命名空间绑定的是模块名 `math`。这种写法能够明确标识成员来源，通常适合模块级依赖：

```python
import torch

torch.cuda.is_available()
```

#### 2.6.2 `import module as name`

```python
import numpy as np

array = np.array([1, 2, 3])
```

`as` 只为模块在当前命名空间创建一个别名，不会创建新的模块对象，也不会改变模块的真实名称：

```python
import numpy as np
import numpy

print(np is numpy)  # True
```

常见用途包括：

- 使用生态中约定俗成的缩写，如 `numpy as np`；
- 避免模块名称冲突；
- 为不同实现提供统一名称：

```python
try:
    import ujson as json
except ImportError:
    import json
```

#### 2.6.3 `from module import name`

```python
from math import sqrt

sqrt(4)
```

这里直接将 `sqrt` 绑定到当前命名空间。调用更简洁，但名称来源不够明显，也更容易发生冲突：

```python
from package_a import create
from package_b import create  # 覆盖前一个 create
```

如需重命名，可以写成：

```python
from package_a import create as create_a
```

#### 2.6.4 工程建议

模块级依赖通常优先使用：

```python
import package
import package.submodule
```

对于生态约定的缩写，可以使用：

```python
import numpy as np
import pandas as pd
```

对于少量、明确且稳定的公共对象，可以使用：

```python
from pathlib import Path
from contextlib import contextmanager
```

一般不建议使用：

```python
from module import *
```

因为它会污染当前命名空间，并且降低代码可读性和静态分析能力。

#### 2.6.5 与 Java `import` 的对比


| 特性 | Python | Java |
| :--- | :--- | :--- |
| 导入对象 | 模块、类、函数、变量等 | 类、接口及其成员 |
| 别名机制 | 支持 `as` | 没有通用的导入别名语法 |
| 直接导入成员 | `from module import name` | `import static Class.member` |
| 通配符导入 | `from module import *` | `import package.*` |
| 导入时执行代码 | 通常会执行模块顶层代码 | 主要用于编译期名称解析，类初始化在实际使用时发生 |
| 缓存机制 | 通过 `sys.modules` 缓存模块 | 由类加载器和 JVM 管理类的加载与初始化 |

对于Java程序员，有一个核心关键区别要特别注意：
- Python 的 `import` 更接近**运行时模块加载与名称绑定**；
- Java 的 `import` 更接近**编译期类型名称简化**；

在 AI-Infra 中，这一区别尤其重要：Python 导入模块可能触发注册、插件发现、CUDA 扩展加载或其他初始化副作用；而Java 的 `import` 本身通常不承担这类运行时初始化职责。

### 2.7 绝对导入与相对导入

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

包内部可以使用相对导入：

```python
# mypackage/runner.py
from .model import Model
```

公共代码和跨包依赖通常更适合使用绝对导入：

```python
from project.models import Model
```

混乱的导入方式容易造成：

- 循环导入；
- 直接运行脚本时报错；
- 包内路径不一致；
- 测试环境和生产环境行为不同。


### 2.8 循环导入

循环导入发生在模块依赖形成环时：

```text
module_a -> module_b -> module_a
```

例如：

```python
# a.py
from b import B

class A:
    def use(self):
        return B()


# b.py
from a import A

class B:
    def use(self):
        return A()
```

Python 导入模块时会先创建模块对象并放入 `sys.modules`，然后执行模块顶层代码。如果此时另一个模块反向导入它，可能出现“部分初始化的模块”，最终触发 `ImportError` 或 `AttributeError`。

循环导入的首选解决方案不是调整导入语句的位置，而是**调整依赖方向**：

1. 重新划分模块职责，避免两个模块互相依赖；
2. 将共享的数据结构、接口或协议提取到独立模块；
3. 必要时使用依赖倒置，让高层和底层共同依赖抽象接口。

例如：

```text
原结构：

runner -> backend -> runner

调整后：

runner  -> protocols <- backend
```

其中，`protocols` 只定义接口和数据契约，不依赖具体实现。

局部导入、`TYPE_CHECKING` 和延迟导入只能作为特定场景下的辅助技术：

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runner import Runner
```

或者：

```python
def create_runner():
    from .runner import Runner
    return Runner()
```

这些方式适用于类型标注、可选依赖、插件系统或确实需要延迟加载的场景，但它们通常只是改变导入时机，并没有消除模块之间的耦合。若代码需要频繁依赖局部导入来“修复”循环导入，通常说明模块边界或依赖方向仍需要重新设计。

因此，工程上应遵循：**先调整依赖结构，再考虑延迟导入；不要用导入技巧掩盖模块设计问题。**

### 2.9 动态导入与 `importlib`

当模块名称在运行时才能确定时，可以使用：

```python
import importlib

module_name = "mypackage.backends.cuda"
module = importlib.import_module(module_name)
```

AI-Infra 中常见用途包括：

- 按配置加载后端；
- 按设备类型加载实现；
- 自动发现插件；
- 延迟加载重量级依赖；
- 根据环境判断是否加载 CUDA 扩展。

例如：

```python
def load_backend(name):
    module = importlib.import_module(
        f"mypackage.backends.{name}"
    )
    return module.create_backend()
```

动态导入增强了扩展性，但也带来一些问题：

- 依赖关系不容易被静态工具发现；
- 导入错误可能延迟到运行时；
- 调试和代码跳转更困难；
- 动态字符串来源不可信时可能产生安全风险。

### 2.10 `sys.path`、项目布局与启动方式

导入查找依赖 `sys.path`：

```python
import sys

for path in sys.path:
    print(path)
```

其内容通常会受到以下因素影响：

- 启动脚本的位置；
- 当前工作目录；
- `PYTHONPATH`；
- 标准库目录；
- 当前环境的 `site-packages`；
- 虚拟环境；
- `.pth` 文件；
- 解释器启动参数。

下面两种启动方式的导入行为可能不同：

```bash
python src/myproject/cli.py
```

```bash
python -m myproject.cli
```

使用 `python script.py` 时，脚本所在目录通常会影响 `sys.path`；使用 `python -m package.module` 时，当前项目环境和包路径通常会参与模块解析。

因此，同一份代码可能出现：

- `python script.py` 可以导入；
- `python -m package.module` 不能导入；
- 测试环境可以导入；
- 安装后的环境不能导入。

对于可安装项目，通常推荐使用 `src` 布局：

```text
myproject/
├── pyproject.toml
├── src/
│   └── myproject/
│       ├── __init__.py
│       └── core.py
└── tests/
```

相比包直接位于仓库根目录的 flat 布局，src 布局可以减少测试时意外导入源码目录的问题，使本地测试更接近用户安装后的真实环境。

开发时可以使用 editable 安装：

```bash
pip install -e .
```

这样导入的代码仍然来自源码目录，修改代码后通常不需要重新复制源码。但如果修改了依赖、命令行入口或其他项目元数据，仍然需要重新执行安装命令。


## 三、类、对象与属性查找

### 3.1 类本身也是对象

在 Python 中，类不是编译期的静态模板，而是运行时对象。

```python
class Runner:
    def run(self, batch):
        return batch
```

执行类定义语句时，Python 会：

1. 创建类命名空间；
2. 执行类体；
3. 收集方法和类属性；
4. 调用元类创建类对象；
5. 将名称 `Runner` 绑定到类对象。

创建实例：

```python
runner = Runner()

print(type(runner))
print(isinstance(runner, Runner))
print(type(Runner))
```

其中：

- `Runner` 是类对象；
- `runner` 是 `Runner` 的实例；
- `type(Runner)` 通常是 `type`；
- 类对象本身也可以拥有属性和方法。

因此：**类是用于创建实例并定义实例行为的对象；类本身通常由元类创建。**

### 3.2 实例属性和类属性

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

需要警惕可变类属性：

```python
class Bad:
    items = []


a = Bad()
b = Bad()

a.items.append(1)

print(b.items)  # [1]
```

如果每个实例都需要独立列表，应在 `__init__` 中创建：

```python
class Good:
    def __init__(self):
        self.items = []
```

### 3.3 方法绑定

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

可以近似理解为：

```python
Runner.run(runner, data)
```

但更准确地说，类中的函数通常是非数据描述符。当通过实例访问时，它会被绑定为方法。

```python
bound = runner.run

print(bound.__self__ is runner)
print(bound.__func__ is Runner.run)
```

输出通常为：

```text
True
True
```

因此：

- `Runner.run` 是类上的函数对象；
- `runner.run` 是绑定方法；
- 绑定方法保存了实例和原始函数；
- 调用绑定方法时，实例会自动作为第一个参数传入。


### 3.4 `classmethod`、`staticmethod` 与 `property`

这三个装饰器都用于改变方法的绑定方式，但用途不同：

- `classmethod`：绑定到类对象；
- `staticmethod`：不自动绑定类或实例；
- `property`：将方法包装成属性访问形式。

#### 3.4.1 `classmethod`：面向类对象的方法

```python
class Runner:
    default_device = "cpu"

    @classmethod
    def create_default(cls):
        return cls(device=cls.default_device)

    def __init__(self, device):
        self.device = device
```

调用时：

```python
runner = Runner.create_default()
```

定义为 `classmethod` 后，Python 会自动将调用它的类传入第一个参数 `cls`：

```python
Runner.create_default()
# 等价于大致意义上的：
# Runner.create_default(Runner)
```

`cls` 与实例方法中的 `self` 类似，但它代表的是**类对象**，而不是某个实例。

因此，`classmethod` 的重点不只是“可以通过类调用”，而是：**方法需要访问或构造类本身，并且应当随着继承关系使用实际的子类。**

例如：

```python
class GPU_Runner(Runner):
    default_device = "cuda"

runner = GPU_Runner.create_default()
print(runner.device)  # cuda
```

这里 `cls` 实际上是 `GPU_Runner`，因此 `classmethod` 创建的是子类实例，而不是固定的 `Runner` 实例。这使它非常适合实现：

- 替代构造方法；
- 从配置、字典或文件创建对象；
- 不同后端的统一创建入口；
- 需要支持子类继承的工厂方法。

常见写法包括：

```python
class Config:
    def __init__(self, host, port):
        self.host = host
        self.port = port

    @classmethod
    def from_dict(cls, data):
        return cls(
            host=data["host"],
            port=data.get("port", 80),
        )
```

调用：

```python
config = Config.from_dict({"host": "localhost"})
```

这里 `from_dict` 不是普通的实例方法，因为对象尚未创建，无法通过 `self` 调用；它也不适合使用 `staticmethod`，因为工厂方法通常需要通过 `cls(...)` 创建当前类或子类对象。

#### 3.4.2 `staticmethod`：不需要实例或类上下文的方法

```python
class Runner:
    @staticmethod
    def validate_config(config):
        return "device" in config
```

调用时：

```python
Runner.validate_config(config)
```

`staticmethod` 不会自动接收 `self` 或 `cls`。它只是放在类命名空间中的普通函数，适合表达逻辑上属于某个类、但不依赖实例状态和类状态的操作。

如果一个函数既不需要访问实例，也不需要访问类，通常也可以考虑将它放在模块级，而不是强行定义成静态方法。

#### 3.4.3 `property`：将方法表现为属性

```python
class Runner:
    def __init__(self, workers):
        self.workers = workers

    @property
    def worker_count(self):
        return len(self.workers)
```

调用时：

```python
runner.worker_count
```

而不是：

```python
runner.worker_count()
```

`property` 适合表示根据对象状态计算得到的属性，或者在保持属性访问语法的同时加入校验、延迟计算等逻辑。

```python
class Runner:
    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        if value not in {"cpu", "cuda"}:
            raise ValueError("unsupported device")
        self._device = value
```

#### 3.4.4 选择建议

可以按以下规则选择：

- 需要实例状态：使用普通实例方法；
- 需要当前类或创建子类实例：使用 `classmethod`；
- 只依赖参数，不依赖实例和类：使用 `staticmethod` 或模块级函数；
- 需要通过属性语法访问计算结果或封装字段：使用 `property`。

在 AI-Infra 中，`classmethod` 尤其适合实现统一的构造和配置入口，例如：

```python
class Runner:
    @classmethod
    def from_config(cls, config):
        return cls(
            model=config["model"],
            device=config.get("device", "cpu"),
        )
```

它可以将“如何从外部配置创建对象”的逻辑集中在类内部，同时保留子类扩展和多后端实现的能力。


#### 3.4.5 与 Java 的对比

| Python | Java 中较接近的形式 | 是否自动获得当前类 |
| :--- | :--- | :--- |
| 实例方法 | 普通实例方法 | 通过 `self` 访问实例 |
| `classmethod` | 没有完全对应物，接近可继承的静态工厂方法 | 是，通过 `cls` |
| `staticmethod` | `static` 方法 | 否 |
| `property` | getter/setter 方法或属性访问器 | 通过属性语法访问 |


其中最大的区别还是`classmethod`，Java没有直接对应的概念，唯一类似的是静态工厂方法：

```java
class Runner {
    public static Runner createDefault() {
        return new Runner();
    }
}
```

这个方法类似 Python 的：

```python
@staticmethod
def create_default():
    return Runner()
```

但它固定创建 `Runner`，不具备 Python `classmethod` 自动适配子类的能力。下面Python代码接近一种“可继承的类级工厂方法”。如果子类继承该方法，`cls` 会自动变成子类：

```python
@classmethod
def create_default(cls):
    return cls()
```

### 3.5 属性查找的大致顺序

当执行：

```python
obj.attr
```

Python 会进行较复杂的属性查找。简化后可以理解为：

1. 查找对象类型及其基类中的数据描述符；
2. 查找实例字典；
3. 查找对象类型及其基类中的普通属性或非数据描述符；
4. 如果仍未找到，再尝试 `__getattr__`。

这里的优先级解释了一个重要现象：

- 数据描述符通常可以覆盖实例字典中的同名属性；
- 普通类属性可能被实例属性覆盖；
- 方法绑定依赖非数据描述符；
- `property` 可以控制属性访问和赋值。

可以通过下面的例子观察普通类属性和实例属性：

```python
class Demo:
    value = 10


obj = Demo()
obj.value = 20

print(obj.__dict__)       # {'value': 20}
print(Demo.__dict__["value"])  # 10
print(obj.value)          # 20
```

### 3.6 `__getattribute__` 和 `__getattr__`

`__getattribute__` 会拦截几乎所有属性访问：

```python
class DebugObject:
    def __getattribute__(self, name):
        print("access:", name)
        return object.__getattribute__(self, name)
```

实现时必须避免递归调用：

```python
# 不推荐
self.__dict__

# 推荐
object.__getattribute__(self, "__dict__")
```

`__getattr__` 只在正常属性查找失败后调用：

```python
class Config:
    def __getattr__(self, name):
        return None
```

它们可以用于：

- 延迟加载；
- 兼容旧字段；
- 动态代理；
- 配置访问；
- 设备属性转发。

但过度使用会降低代码可读性和静态分析能力。

### 3.7 `__new__` 与 `__init__`

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

一般业务代码只需要实现 `__init__`，不要轻易重写 `__new__`。

### 3.8 继承 与 MRO

继承是 Python 中实现类型复用和协议扩展的主要机制。它不仅可以减少重复代码，更重要的是让子类获得并遵守父类定义的接口、生命周期和对象协议。

#### 3.8.1 继承：复用类型协议

继承通常表达“是一种”关系。例如，PyTorch 中的模型继承 `torch.nn.Module`：

```python
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.layers(x)
```

`MLP` 继承 `nn.Module` 后，获得的不只是若干方法，还包括一整套模型协议和生命周期能力：

- 参数和子模块注册；
- `state_dict()`；
- `.to(device)`；
- `.train()` 和 `.eval()`；
- hooks；
- 序列化与设备迁移。

因此，继承适合表达稳定的类型关系和抽象协议，例如：

- `MLP` 是一种 `nn.Module`；
- 自定义 DataLoader 遵守某种数据加载协议；
- 推理器实现统一的 Runner 接口；
- 插件实现框架规定的 Backend 协议。

继承不适合被当作通用的组件拼装机制。对于 `model`、`scheduler`、`tokenizer` 等业务依赖，通常应优先考虑组合。

单继承时，`super()` 通常表现为调用父类实现，例如调用父类的初始化方法。但在多继承中，`super()` 的含义不再是“调用某个固定父类”，而是与 MRO 共同决定下一个实现。

#### 3.8.2 多继承：MRO 与协作式 `super()`

Python 支持一个类继承多个基类：

```python
class A:
    def run(self):
        return ["A"]


class B:
    def run(self):
        return ["B"]


class C(A, B):
    pass
```

```python
C.__mro__
# (C, A, B, object)

C().run()
# ["A"]
```

##### ① MRO 决定方法查找顺序

MRO 是 Method Resolution Order 的缩写，即方法解析顺序。它决定 Python 在多继承结构中按照什么顺序查找属性和方法。

上例中的查找路径是：

```text
C → A → B → object
```

Python 在 `A` 中找到 `run()` 后，普通方法查找就结束，因此不会自动调用 `B.run()`。

可以通过以下方式查看实际的解析顺序：

```python
C.__mro__
C.mro()
```

需要特别注意的是：**MRO 只决定方法的查找顺序，不会自动调用所有父类中的同名方法**。如果需要，那么需要显示的调用super()。

##### ② `super()` 沿 MRO 查找下一个实现

`super()` 不应简单理解为“调用父类”。更准确地说，它会从当前类在 MRO 中的位置之后，继续查找下一个实现。

```python
class Base:
    def run(self):
        return ["base"]


class Logging:
    def run(self):
        result = super().run()
        result.append("logging")
        return result


class Metrics:
    def run(self):
        result = super().run()
        result.append("metrics")
        return result


class Runner(Logging, Metrics, Base):
    pass
```

```python
Runner.__mro__
# (Runner, Logging, Metrics, Base, object)

Runner().run()
# ["base", "metrics", "logging"]
```

实际调用链为：

```text
Logging.run()
    → Metrics.run()
        → Base.run()
        ← Metrics.run() 返回
    ← Logging.run() 返回
```

这里，`Logging.run()` 中的 `super()` 并不是固定调用 `Base.run()`，而是按照 `Runner` 的 MRO，继续查找 `Logging` 后面的下一个实现，即 `Metrics.run()`。

一句话概括：

> MRO 解决“下一个是谁”，`super()` 解决“是否继续调用下一个”。

##### ③ 协作式多继承

只有当每一层实现都遵守协作式约定时，多继承才能形成完整的调用链：

```python
class LoggingMixin:
    def __init__(self, *args, **kwargs):
        self.logger = create_logger()
        super().__init__(*args, **kwargs)
```

主要约定包括：

1. 每一层都调用 `super()`；
2. 同名方法的签名保持兼容；
3. 返回值能够继续向调用链传递；
4. `__init__()` 也要调用 `super().__init__()`；
5. 不要假设 `super()` 固定指向某个父类；
6. 异常处理不能意外截断必要的调用链。

如果某一层不调用 `super()`，后续实现可能完全不会执行：

```python
class Broken:
    def run(self):
        return ["broken"]
```

同样，如果某个基类的 `__init__()` 没有调用 `super().__init__()`，后续基类的初始化也可能被跳过。

多继承的主要风险包括：

- 多个基类定义同名方法；
- 基类声明顺序改变 MRO；
- 初始化顺序不符合预期；
- 方法签名或返回值不兼容；
- 某一层遗漏 `super()`，导致调用链中断。

因此，使用多继承时，应通过 `__mro__`、`mro()` 或 `inspect` 检查实际查找顺序，并用测试验证调用链、返回值和异常行为。

#### 3.8.3 Mixin：多继承的一种实践模式

Mixin 是建立在多继承、MRO 和协作式 `super()` 之上的一种设计模式，用于向已有类型注入横向能力。

Mixin 通常不代表完整的业务实体，而是提供一组职责相对独立的方法、属性或元数据。例如：

```python
class LoggingMixin:
    def log(self, message):
        print(f"[LOG] {message}")


class MetricsMixin:
    def record_metric(self, name, value):
        print(f"{name}={value}")


class Runner(LoggingMixin, MetricsMixin):
    def run(self):
        self.log("start")
        self.record_metric("requests", 1)
```

常见的 Mixin 能力包括：

- 日志；
- 权限检查；
- 指标采集；
- 序列化；
- 缓存；
- 参数校验。

Django 的 `LoginRequiredMixin` 也是类似模式：

```python
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import TemplateView


class DashboardView(LoginRequiredMixin, TemplateView):
    template_name = "dashboard.html"
```

这里，`TemplateView` 是核心类型，`LoginRequiredMixin` 则为其增加登录检查能力。

Mixin 并没有绕开多继承的复杂性，而是将其限制在较明确的能力复用场景中。因此，Mixin 应满足以下原则：

- 职责单一，能力边界清晰；
- 不代表完整的业务对象；
- 不持有复杂的业务依赖；
- 对宿主类的要求应尽量明确；
- 多个 Mixin 的同名方法必须签名兼容；
- `__init__()` 必须遵守协作式 `super()` 约定；
- 通过测试固定实际的 MRO 和调用链。

如果只是给单个函数增加日志、重试或 tracing，装饰器通常比 Mixin 更直接；如果需要组合 `model`、`scheduler`、`tokenizer` 等业务组件，则应优先使用组合。


### 3.9 组合：继承之外的主要复用方式

组合表达“拥有”关系，即一个对象持有并协调其他对象：

```python
class InferenceRunner:
    def __init__(self, model, scheduler, tokenizer):
        self.model = model
        self.scheduler = scheduler
        self.tokenizer = tokenizer

    def run(self, request):
        inputs = self.tokenizer(request)
        batch = self.scheduler.schedule(inputs)
        return self.model(batch)
```

`InferenceRunner` 拥有 `model`、`scheduler` 和 `tokenizer`，但它不是这些对象的子类。

相比继承，组合具有以下优势：

- 依赖关系显式；
- 更容易注入 Mock、Fake 等测试替身；
- 可以在运行时替换协作者；
- 不依赖复杂的 MRO；
- 不需要隐式的 `super()` 协议；
- 更适合表达业务流程和组件编排。

在 AI-Infra 和高性能服务中，可以按照以下原则选择复用方式：

| 关系或需求 | 推荐方式 |
| :--- | :--- |
| 表达稳定的“是一种”关系 | 继承 |
| 遵守已有框架协议，如 `nn.Module` | 继承 |
| 复用独立的横向能力 | Mixin，谨慎使用 |
| 为单个调用增加日志、重试或 tracing | 装饰器 |
| 动态组织多个请求处理阶段 | 中间件或显式包装 |
| 管理 `model`、`scheduler` 等业务依赖 | 组合 |

在工程实践中，应遵循：

> 组合优于继承，多继承必须协作，Mixin 需要约束，复杂继承结构必须通过测试验证。


## 四、数据模型与对象协议

Python 中很多看起来像语法的行为，实际上依赖对象协议和特殊方法。

### 4.1 常见语法与协议

| Python 表达式 | 相关协议或特殊方法 |
| :--- | :--- |
| `len(x)` | `__len__` |
| `x[key]` | `__getitem__` |
| `x[key] = value` | `__setitem__` |
| `x + y` | `__add__` |
| `x == y` | `__eq__` |
| `hash(x)` | `__hash__` |
| `x(...)` | 可调用协议，通常由 `__call__` 实现 |
| `with x` | `__enter__`、`__exit__` |
| `for item in x` | `__iter__`、`__next__` |
| `if x` | `__bool__` 或 `__len__` |

这是一种帮助理解的概念映射。特殊方法并不总是通过普通的 `obj.method(...)` 属性查找来调用，解释器可能会在类型层面寻找对应实现。

### 4.2 可调用对象

实现 `__call__` 后，对象就可以像函数一样调用：

```python
class Runner:
    def __init__(self, model):
        self.model = model

    def __call__(self, batch):
        return self.model(batch)
```

于是：

```python
runner(batch)
```

会触发可调用协议。

在 AI-Infra 中，可调用对象常见于：

- 模型；
- 推理器；
- 数据预处理器；
- 后处理器；
- 调度器；
- Hook；
- Callback；
- Loss 对象。

相比普通函数，可调用对象可以同时保存：

- 调用接口；
- 配置状态；
- 缓存；
- 依赖对象；
- 生命周期状态。

### 4.3 迭代器与可迭代对象

可迭代对象可以被 `for` 遍历：

```python
for item in data:
    ...
```

一个简单的迭代器可以实现：

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

`for` 循环会不断取得下一个值，直到收到 `StopIteration`。

AI-Infra 中大量组件依赖迭代协议：

- Dataset；
- DataLoader；
- Batch 生成器；
- Token 流；
- 日志流；
- 请求队列；
- 分片数据读取器。

### 4.4 `__getitem__` 与容器协议

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
print(batch[1:])
```

通过实现 `__getitem__`，对象可以支持下标和切片操作。

Batch、Tensor 包装器、缓存对象和配置对象都可能实现类似协议。

### 4.5 对象真值判断

以下代码会触发对象的真值判断：

```python
if obj:
    ...
```

Python 会优先尝试：

```python
obj.__bool__()
```

如果没有，则可能使用：

```python
obj.__len__()
```

```python
class Queue:
    def __init__(self, items):
        self.items = items

    def __bool__(self):
        return bool(self.items)
```

需要注意，某些 Tensor 框架会禁止直接对多元素 Tensor 做布尔判断，因为这种判断存在歧义：

```python
if tensor:
    ...
```

因此在 AI-Infra 中，不要默认所有对象都可以安全放入 `if`。

### 4.6 `__eq__` 与 `__hash__`

对象身份和对象相等是不同概念：

```python
a is b
a == b
```

- `is` 判断是否为同一个对象；
- `==` 调用相等性逻辑。

如果定义了 `__eq__` 但没有定义 `__hash__`，Python 通常会将对象设为不可哈希：

```python
class User:
    def __init__(self, user_id):
        self.user_id = user_id

    def __eq__(self, other):
        return (
            isinstance(other, User)
            and self.user_id == other.user_id
        )


user = User(1)
# {user}  # TypeError: unhashable type: 'User'
```

如果对象确实要作为字典键或集合元素，就必须保证：

- 相等对象具有相同哈希值；
- 参与哈希的字段在生命周期中保持不变。

```python
class User:
    def __init__(self, user_id):
        self.user_id = user_id

    def __eq__(self, other):
        return (
            isinstance(other, User)
            and self.user_id == other.user_id
        )

    def __hash__(self):
        return hash(self.user_id)
```

如果 `user_id` 可变，就不适合在对象作为字典键期间修改它。

### 4.7 描述符协议

描述符是实现了 `__get__`、`__set__` 或 `__delete__` 的对象。当描述符被放置为类属性时，Python 会在属性访问时自动调用这些方法。

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
                f"{self.name} expects "
                f"{self.expected_type.__name__}, "
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
config.batch_size = "big"
```

最后一行会抛出类型错误。

`property`、`classmethod`、`staticmethod` 和方法绑定都与描述符协议有关。许多框架中的字段校验、依赖注入、模型参数声明和配置代理，也建立在类似机制上。


## 五、函数、闭包与装饰器

本节讨论 Python 中函数对象、闭包和装饰器的关系。

函数是一等对象，使函数能够被赋值、传递和返回；闭包提供了保存外部作用域状态的能力；装饰器则利用可调用对象包装机制，在不直接修改原函数主体的情况下增强其行为。

三者经常一起出现，但它们解决的问题并不相同：

- **函数对象**：函数可以像普通对象一样被操作；
- **闭包**：内部函数可以捕获并保存外部作用域中的变量；
- **装饰器**：接收一个可调用对象，并返回一个包装后的可调用对象。

需要特别注意的是：装饰器经常使用闭包实现，但装饰器不等于闭包；闭包也可以用于装饰器以外的场景。

### 5.1 函数是一等对象

在 Python 中，函数是“一等对象”（First-class Object）。这意味着函数可以：

- 赋值给变量；
- 作为参数传递；
- 作为其他函数的返回值；
- 存储在列表、字典等容器中；
- 在运行时动态创建；
- 通过属性保存额外信息。

例如：

```python
def add(x, y):
    return x + y


operation = add

print(operation(1, 2))  # 3
```

这里的 `operation` 和 `add` 指向同一个函数对象。

函数也可以作为参数传递：

```python
def apply(operation, value):
    return operation(value)


def double(value):
    return value * 2


print(apply(double, 3))  # 6
```

还可以作为返回值：

```python
def make_multiplier(factor):
    def multiply(value):
        return value * factor

    return multiply


double = make_multiplier(2)
print(double(5))  # 10
```

函数的一等对象特性是以下机制的基础：

- 高阶函数；
- 回调函数；
- 闭包；
- 装饰器；
- 策略函数；
- Hook；
- 任务处理器；
- 异步任务封装。

在 AI-Infra 中，可以将不同的处理逻辑作为函数传入组件，从而避免为每一种行为创建单独的子类：

```python
def process_batches(loader, process):
    for batch in loader:
        yield process(batch)
```

这里，`process` 可以是预处理、推理、后处理或日志记录函数。

### 5.2 闭包与词法作用域

当一个内部函数引用外部函数作用域中的变量，并且内部函数在外部函数返回后仍然可以访问这些变量时，就形成了闭包（Closure）。

```python
def make_multiplier(factor):
    def multiply(value):
        return value * factor

    return multiply


double = make_multiplier(2)
triple = make_multiplier(3)

print(double(5))  # 10
print(triple(5))  # 15
```

`make_multiplier()` 执行结束后，局部变量 `factor` 通常已经不再位于原来的执行帧中。但返回的 `multiply` 仍然可以访问它，因为函数对象保存了对外部变量的引用。

闭包的核心是**词法作用域**：

> 函数中名称的解析，主要依据函数定义时所在的代码结构，而不是调用它的位置。

闭包常用于：

- 封装状态；
- 生成带有预置配置的函数；
- 延迟计算；
- 回调函数；
- 创建策略函数；
- 实现参数化装饰器。

#### 5.2.1 在闭包中修改外部变量

如果只是读取外部变量，可以直接访问：

```python
def make_greeting(prefix):
    def greet(name):
        return f"{prefix}, {name}"

    return greet
```

如果需要重新绑定外部作用域中的不可变变量，需要使用 `nonlocal`：

```python
def make_counter(start=0):
    count = start

    def increment():
        nonlocal count
        count += 1
        return count

    return increment


counter = make_counter()
print(counter())  # 1
print(counter())  # 2
```

这里的 `count` 属于 `make_counter()` 的局部变量。`increment()` 通过 `nonlocal` 声明，表示修改外层函数中的 `count`，而不是创建一个新的局部变量。

也可以通过可变对象保存状态：

```python
def make_counter(start=0):
    state = [start]

    def increment():
        state[0] += 1
        return state[0]

    return increment
```

不过，如果状态较多或逻辑复杂，使用类通常比复杂闭包更清晰：

```python
class Counter:
    def __init__(self, start=0):
        self.count = start

    def increment(self):
        self.count += 1
        return self.count
```

因此，闭包适合封装少量状态和简单行为；当状态、生命周期和接口逐渐复杂时，应考虑使用显式对象。

#### 5.2.2 闭包中的延迟绑定

闭包有一个常见陷阱：循环变量可能会发生延迟绑定。

```python
functions = []

for i in range(3):
    functions.append(lambda: i)

print([function() for function in functions])
```

输出为：

```text
[2, 2, 2]
```

原因是 `lambda` 中的 `i` 并不会在函数创建时立即求值，而是在函数调用时查找。此时循环已经结束，`i` 的值为 `2`。

可以使用默认参数捕获当前值：

```python
functions = []

for i in range(3):
    functions.append(lambda i=i: i)

print([function() for function in functions])
```

输出为：

```text
[0, 1, 2]
```

在构造批处理函数、异步回调或并发任务时，需要特别注意这一问题。

### 5.3 装饰器

### 5.3.1 装饰器的基本机制

装饰器（Decorator）是一种可调用对象包装机制：

> 装饰器接收一个函数或类，并返回一个替代它的可调用对象。

例如：

```python
def log_call(func):
    def wrapper(*args, **kwargs):
        print("calling", func.__name__)
        return func(*args, **kwargs)

    return wrapper
```

使用装饰器：

```python
@log_call
def predict(x):
    return x * 2
```

其效果等价于：

```python
def predict(x):
    return x * 2


predict = log_call(predict)
```

因此，装饰器是在函数定义完成后应用的。每次调用 `predict()` 时，实际调用的是包装函数 `wrapper()`，而不是最初定义的函数对象。

对于多个装饰器：

```python
@outer
@inner
def run():
    ...
```

其绑定顺序等价于：

```python
run = outer(inner(run))
```

调用时，`outer` 返回的包装器会先接收到调用，然后再进入 `inner` 的包装器。

一个装饰器通常包含以下逻辑：

1. 接收原始函数；
2. 定义包装函数；
3. 在包装函数中增加额外逻辑；
4. 调用原始函数；
5. 返回结果；
6. 返回包装函数作为替代对象。

#### 5.3.2 装饰器与闭包的关系

上面的 `log_call()` 使用了闭包：

```python
def log_call(func):
    def wrapper(*args, **kwargs):
        print("calling", func.__name__)
        return func(*args, **kwargs)

    return wrapper
```

其中：

- `log_call` 是装饰器；
- `wrapper` 是包装函数；
- `wrapper` 捕获了外层作用域中的 `func`；
- `func` 即使在 `log_call()` 返回后，仍然可以被 `wrapper` 使用。

因此，可以说：

> 闭包是实现函数式装饰器的常见技术，但不是装饰器的必要条件。

装饰器也可以通过可调用对象实现：

```python
class LogCall:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        print("calling", self.func.__name__)
        return self.func(*args, **kwargs)
```

使用方式：

```python
@LogCall
def predict(x):
    return x * 2
```

这里的 `LogCall` 类本身作为装饰器接收函数，并创建一个可调用实例。调用 `predict()` 时，实际执行的是实例的 `__call__()` 方法。

两种实现方式各有适用场景：

- 闭包实现简洁，适合逻辑较少的包装；
- 可调用对象适合保存较多状态，或需要提供额外方法和生命周期控制的场景。

#### 5.3.3 使用 `functools.wraps` 保留元信息

直接返回 `wrapper` 会导致原函数的部分元信息丢失：

```python
def log_call(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper
```

例如：

```python
@log_call
def predict(x):
    """Run prediction."""
    return x * 2


print(predict.__name__)  # wrapper
print(predict.__doc__)   # None
```

这会影响：

- 调试；
- 日志；
- 文档生成；
- 反射；
- 类型分析；
- 测试框架；
- 错误信息；
- 性能分析工具。

推荐使用 `functools.wraps`：

```python
from functools import wraps


def log_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("calling", func.__name__)
        return func(*args, **kwargs)

    return wrapper
```

此时，包装函数会尽可能保留原函数的名称、文档字符串和模块信息：

```python
@log_call
def predict(x):
    """Run prediction."""
    return x * 2


print(predict.__name__)  # predict
print(predict.__doc__)   # Run prediction.
```

`wraps` 并不会让包装函数真正恢复原函数的全部行为。包装器仍然可能改变：

- 参数检查；
- 返回值；
- 异常类型；
- 执行时机；
- 上下文；
- 类型签名。

因此，`wraps` 是必要的元信息维护工具，但不是行为透明性的保证。

#### 5.3.4 带参数的装饰器

如果装饰器本身需要配置参数，就需要额外增加一层函数：

```python
from functools import wraps


def retry(max_attempts):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except TimeoutError:
                    if attempt == max_attempts - 1:
                        raise

        return wrapper

    return decorator
```

使用：

```python
@retry(max_attempts=3)
def request():
    ...
```

它等价于：

```python
request = retry(max_attempts=3)(request)
```

三层结构分别是：

```text
retry(max_attempts)
    → decorator(func)
        → wrapper(*args, **kwargs)
```

其中：

- 最外层接收装饰器配置；
- 中间层接收被装饰函数；
- 最内层执行实际包装逻辑。

在 AI-Infra 中，参数化装饰器可以用于：

- 配置重试次数；
- 指定超时时间；
- 设置采样率；
- 添加指标名称；
- 控制日志级别；
- 标记执行阶段；
- 配置缓存策略。

但重试装饰器不应简单捕获所有异常：

```python
except Exception:
    ...
```

更合理的方式是只处理明确适合重试的异常，例如超时或临时网络错误，并在达到最大次数后保留原始异常：

```python
from functools import wraps


def retry_on_timeout(max_attempts):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except TimeoutError:
                    if attempt == max_attempts - 1:
                        raise

        return wrapper

    return decorator
```

参数化装饰器会增加调用路径的层次，因此应保持实现简单，并通过测试确认重试次数、异常传播和返回值行为符合预期。

#### 5.3.5 同步与异步装饰器

同步函数和异步函数需要使用不同的包装方式。

同步装饰器：

```python
from functools import wraps


def log_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("start")
        result = func(*args, **kwargs)
        print("end")
        return result

    return wrapper
```

异步函数应使用 `async def` 包装器，并通过 `await` 调用原函数：

```python
from functools import wraps


def async_log_call(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        print("start")
        result = await func(*args, **kwargs)
        print("end")
        return result

    return wrapper
```

如果错误地使用同步包装器：

```python
def wrong_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper
```

虽然可能仍然返回协程对象，但装饰器无法正确处理异步执行过程中的：

- 开始和结束时机；
- 异常传播；
- `await`；
- 超时；
- 取消；
- 上下文变量。

在异步服务和推理系统中，应明确区分同步函数与异步函数。必要时可以使用 `inspect.iscoroutinefunction()` 判断目标对象是否为异步函数。


## 六、生成器、惰性执行与资源生命周期

### 6.1 `return` 与 `yield`

普通函数执行时会直接返回结果：

```python
def make_list():
    return [1, 2, 3]
```

生成器函数使用 `yield`：

```python
def make_generator():
    yield 1
    yield 2
    yield 3
```

调用生成器函数时，函数体不会立即执行：

```python
generator = make_generator()
print("created")
```

只有迭代时才会执行：

```python
for value in generator:
    print(value)
```

生成器每次产生一个值，并保留上一次暂停的位置。

### 6.2 观察生成器的暂停与恢复

```python
def stream():
    print("start")
    yield 1
    print("resume")
    yield 2
    print("end")


g = stream()

print("created")
print(next(g))
print("between")
print(next(g))
```

输出大致为：

```text
created
start
1
between
resume
2
```

这说明：

- 调用生成器函数只创建生成器对象；
- 第一次 `next()` 才开始执行；
- 执行到 `yield` 时暂停；
- 下一次 `next()` 从上次暂停位置继续。

### 6.3 惰性计算

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

AI-Infra 中常见场景包括：

- 流式 Token 输出；
- 大规模数据集读取；
- Batch 生成；
- 日志消费；
- 大文件处理；
- 请求流；
- 内存受限环境。

需要注意，惰性执行并不等于“没有成本”。生成器可能长期持有：

- 文件；
- 网络连接；
- 数据库游标；
- GPU 张量；
- 大型缓存；
- 进程间通信对象。

因此必须关注生成器何时结束、何时关闭以及异常路径是否执行清理。

### 6.4 `yield from`

`yield from` 可以转发另一个可迭代对象：

```python
def combined():
    yield from [1, 2, 3]
    yield from [4, 5]
```

也可以组合多个数据分片：

```python
def read_dataset(dataset):
    for shard in dataset.shards:
        yield from read_shard(shard)
```

它不仅能减少嵌套循环，还能转发子生成器的返回值和控制信号。

### 6.5 生成器关闭和资源清理

生成器支持：

```python
send()
throw()
close()
```

常见业务代码仍然主要使用：

```python
for item in generator:
    ...
```

如果生成器持有资源，应确保异常、提前退出和显式关闭时都能清理：

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

当生成器被关闭或因异常退出时，`finally` 和上下文管理器的退出逻辑可以帮助释放资源。


## 七、上下文管理器与资源生命周期

### 7.1 `with` 的基本机制

以下代码：

```python
with resource() as value:
    use(value)
```

可以近似理解为：

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

需要注意：

- 如果 `__enter__()` 抛出异常，`__exit__()` 不会被调用；
- 只有进入阶段成功后，退出阶段才会负责清理；
- `__exit__()` 返回真值时，异常会被抑制；
- `__exit__()` 返回 `False` 或 `None` 时，异常继续传播。

### 7.2 自定义上下文管理器

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

上下文管理器适合管理：

- 文件；
- 锁；
- 数据库连接；
- 临时配置；
- GPU 资源；
- profiler；
- 通信环境；
- 推理模式。

### 7.3 `contextlib.contextmanager`

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

使用时：

```python
with resource() as handle:
    process(handle)
```

### 7.4 AI-Infra 中的上下文管理器

例如：

```python
with torch.inference_mode():
    output = model(x)
```

这种写法通常意味着：

1. 进入某种临时运行模式；
2. 执行一段代码；
3. 退出代码块；
4. 恢复之前的运行状态。

类似场景包括：

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
5. 是否支持异步版本；
6. 上下文对象是否持有重量级资源。

### 7.5 异步上下文管理器

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

通常它们是异步函数：

```python
class AsyncResource:
    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()
```

异步上下文管理器适合数据库连接、网络连接、异步锁和异步资源池。


## 八、异常处理与失败传播

### 8.1 异常处理结构

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

执行规则是：

- `try` 中发生异常时，寻找匹配的 `except`；
- 没有异常时执行 `else`；
- 无论是否异常，进入成功执行阶段后通常都会执行 `finally`；
- 未被处理的异常继续沿调用栈传播。

### 8.2 不要随意捕获 `BaseException`

Python 的异常层级大致为：

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

### 8.3 自定义异常与错误边界

可以根据业务边界定义异常：

```python
class BackendError(Exception):
    pass


class BackendUnavailableError(BackendError):
    pass


class BackendExecutionError(BackendError):
    pass
```

调用方可以根据异常类型采取不同策略：

```python
try:
    run_backend()
except BackendUnavailableError:
    fallback_to_cpu()
except BackendExecutionError:
    retry()
```

好的异常类型应该帮助调用方判断：

- 发生了什么；
- 哪个组件失败；
- 是否可以重试；
- 是否可以切换后端；
- 是否需要终止任务。

### 8.4 异常链

如果在处理一个异常时抛出另一个异常，可以保留原始原因：

```python
try:
    load_config()
except OSError as exc:
    raise RuntimeError(
        "failed to load configuration"
    ) from exc
```

这样错误信息会显示异常链，有助于定位根因。

如果需要显式隐藏底层异常，可以使用：

```python
try:
    internal_operation()
except InternalError:
    raise PublicError("operation failed") from None
```

但应该谨慎使用 `from None`，因为它可能让排查根因变得困难。

### 8.5 记录异常并重新抛出

常见工程写法：

```python
try:
    worker.run()
except Exception:
    logger.exception("worker failed")
    raise
```

这里的 `raise` 会重新抛出当前异常，并保留原始 traceback。

不建议只记录错误而不抛出：

```python
try:
    worker.run()
except Exception as exc:
    logger.error(str(exc))
```

这可能导致：

- 错误被吞掉；
- 上层误以为任务成功；
- Worker 继续运行在不一致状态；
- 分布式任务出现更难排查的问题。


## 九、一个推理组件的完整运行时追踪

下面用一个简化的推理组件，将前面的机制串联起来。

```python
from contextlib import nullcontext


REGISTRY = {}


def registered(name):
    def decorator(cls):
        if name in REGISTRY:
            raise ValueError(
                f"duplicate registration: {name}"
            )

        REGISTRY[name] = cls
        return cls

    return decorator


class InferenceContext:
    def __enter__(self):
        print("enter inference mode")
        return self

    def __exit__(self, exc_type, exc, tb):
        print("exit inference mode")
        return False


@registered("runner")
class Runner:
    def __init__(self, model, inference=True):
        self.model = model
        self.inference = inference

    def __call__(self, batch):
        context = (
            InferenceContext()
            if self.inference
            else nullcontext()
        )

        with context:
            return self.model(batch)

    def stream(self, batch):
        yield from self.model.generate(batch)
```

阅读这段代码时，可以按照运行时顺序追踪。

### 9.1 模块导入阶段

导入该模块时：

1. 创建模块对象；
2. 将模块放入 `sys.modules`；
3. 执行顶层代码；
4. 创建 `REGISTRY`；
5. 创建 `registered` 函数；
6. 创建 `InferenceContext` 类；
7. 执行 `Runner` 类定义；
8. 调用 `registered("runner")` 返回装饰器；
9. 使用装饰器处理 `Runner`；
10. 将 `Runner` 注册到 `REGISTRY`；
11. 将名称 `Runner` 绑定到类对象。

如果该模块没有被导入，注册逻辑就不会执行。

### 9.2 创建对象阶段

执行：

```python
runner = Runner(model)
```

运行时会：

1. 调用类对象；
2. 通过 `__new__` 创建实例；
3. 通过 `__init__` 初始化实例；
4. 设置 `model` 和 `inference` 属性；
5. 返回 `Runner` 实例。

### 9.3 调用对象阶段

执行：

```python
output = runner(batch)
```

运行时触发可调用协议：

1. 找到 `Runner` 类型上的 `__call__`；
2. 将 `runner` 作为实例传入；
3. 根据 `inference` 选择上下文管理器；
4. 进入上下文；
5. 调用 `self.model(batch)`；
6. 离开上下文；
7. 返回结果。

### 9.4 流式执行阶段

执行：

```python
for output in runner.stream(batch):
    consume(output)
```

过程是：

1. 调用 `stream`，创建生成器对象；
2. 第一次迭代时进入函数体；
3. 调用 `self.model.generate(batch)`；
4. `yield from` 转发模型生成的结果；
5. 每次 `yield` 后暂停；
6. 下一次迭代时继续执行；
7. 生成结束后抛出 `StopIteration`；
8. `for` 循环结束。

### 9.5 异常发生时

如果模型调用抛出异常：

```python
try:
    output = runner(batch)
except Exception:
    logger.exception("inference failed")
    raise
```

则：

1. 模型异常向上冒泡；
2. `with` 退出；
3. `InferenceContext.__exit__()` 执行；
4. `__exit__()` 返回 `False`；
5. 原始异常继续传播；
6. 上层记录 traceback；
7. 根据异常类型决定重试、降级或终止 Worker。

这就是 Python 语言机制在 AI-Infra 组件中的完整组合：

- 导入系统负责加载和注册；
- 类机制负责创建组件；
- 描述符和属性查找负责组织对象行为；
- `__call__` 负责统一调用接口；
- 生成器负责惰性和流式输出；
- 上下文管理器负责资源和状态恢复；
- 异常机制负责失败传播和故障边界。


## 十、工程实践建议

### 10.1 减少导入副作用

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

这样可以让初始化时机更加明确，也更容易测试。

### 10.2 控制模块依赖方向

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
- 调整对象职责；
- 使用依赖注入替代直接导入。

### 10.3 谨慎使用隐式机制

以下机制很强大，但不应滥用：

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

### 10.4 保证资源清理

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
with resource:
    ...
```

或者：

```python
try:
    ...
finally:
    cleanup()
```

### 10.5 让对象表示有助于调试

为重要对象实现清晰的 `__repr__`：

```python
class Request:
    def __repr__(self):
        return (
            f"Request("
            f"id={self.request_id!r}, "
            f"batch_size={self.batch_size}, "
            f"device={self.device!r})"
        )
```

日志中看到完整对象状态，通常比单独打印多个字段更容易排查问题。

### 10.6 用小实验验证运行时假设

遇到不确定的 Python 行为时，不要只依赖记忆，可以使用最小实验验证：

```python
import dis
import inspect
import sys
```

适合验证的问题包括：

- 某个函数是否被装饰器替换；
- 某个模块是否已经进入 `sys.modules`；
- 某个属性是否来自描述符；
- 某个对象是否真正实现了迭代协议；
- 生成器何时开始执行；
- 异常在哪一层被捕获；
- `super()` 按什么顺序查找实现。

运行时实验不是替代文档和源码阅读，而是帮助建立可靠心智模型的工具。


## 十一、总结

理解 Python 在 AI-Infra 中的行为，可以从五个问题开始。

### 1. 代码如何被编译和执行？

需要理解：

- 源代码；
- code object；
- 字节码；
- 函数对象；
- 执行帧；
- 调用栈。

### 2. 名称如何绑定到对象？

需要理解：

- 名称与对象；
- 对象身份；
- 可变性；
- 局部作用域；
- 闭包作用域；
- 全局命名空间。

### 3. 代码如何被组织和加载？

需要理解：

- 模块；
- 包；
- `sys.modules`；
- `ModuleSpec`；
- finder 和 loader；
- 循环导入；
- 动态导入；
- `sys.path`；
- 导入副作用。

### 4. 对象如何创建、查找和协作？

需要理解：

- 类；
- 实例；
- 属性查找；
- 描述符；
- 方法绑定；
- 继承；
- 组合；
- MRO；
- `super()`；
- 特殊方法协议。

### 5. 任务、资源和错误如何流动？

需要理解：

- 函数对象；
- 闭包；
- 装饰器；
- 生成器；
- 惰性执行；
- 上下文管理器；
- 异常链；
- 重试；
- 取消；
- 并发任务中的异常传播。

掌握这些机制后，阅读 AI-Infra 代码时就不再只是逐行翻译语法，而是能够理解：

- 一个组件为什么会被自动加载；
- 一个后端为什么会在导入时注册；
- 一个对象为什么可以像函数一样调用；
- 一个 Batch 为什么可以被遍历和切片；
- 一个属性访问为什么会触发校验或动态加载；
- 一个生成器为什么没有在创建时立即执行；
- 一个上下文管理器修改了什么状态；
- 一个异常会不会导致 Worker 退出；
- 一个注册表为什么在不同进程中并不共享。

这正是从“会写 Python”走向“能够理解 Python 工程和 AI-Infra 框架”的关键一步。

后续文章将在此基础上，继续讨论 Python 的类型系统、并发与异步、动态机制、内存管理、调试实践以及生产工程化。