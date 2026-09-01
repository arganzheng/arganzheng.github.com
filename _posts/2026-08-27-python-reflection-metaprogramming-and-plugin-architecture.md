---
layout: post
title: Python 在 AI-Infra（04）：Python的动态机制及工程实践
subtitle: Python Dynamic Mechanisms and Practice
tags: [Python]
catalog: true
---

在传统业务系统中，Python 的动态特性常常被视为一种“方便开发”的语言能力：可以通过字符串获取属性，可以在运行时导入模块，也可以用装饰器包装函数。

但在 AI Infra 中，动态机制的价值远不止于“少写几行代码”。

推理引擎、模型加载器、调度器、缓存系统、存储后端和监控组件，往往需要在不同环境中自由组合。系统既要能够快速接入新组件，又不能让核心逻辑充满 `if-else`。这使得 Python 的反射、元编程和动态加载能力，逐渐从语言技巧演变为架构工具。

本文将从三个问题出发：

1. Python 如何在运行时观察代码？
2. Python 如何在运行时改造代码？
3. 这些能力如何应用到 AI Infra 的插件系统与路由分发系统中？


## 一、AI-Infra 为什么需要动态机制

在讨论"用什么机制"之前，先说清楚"为什么需要"。AI-Infra 对动态能力的需求来自三个方向：组件复杂性带来的解耦压力、静态代码与运行时环境之间的连接鸿沟，以及灵活性与热路径性能之间的权衡。

### 1.1 组件复杂性：系统需要解耦

一个典型的 AI 推理平台，可能包含以下组件：

- 模型格式解析器：PyTorch、ONNX、TensorRT、Safetensors 等；
- 推理后端：CUDA、CPU、ROCm、各类专用加速器；
- 调度器：批处理调度、优先级调度、流式调度；
- 缓存系统：本地缓存、Redis、对象存储；
- 监控组件：日志、Tracing、指标上报；
- 服务接口：HTTP、gRPC、消息队列。

如果所有能力都直接写在核心逻辑中，代码很快会变成：

```python
if backend == "torch":
    ...
elif backend == "tensorrt":
    ...
elif backend == "onnx":
    ...
elif backend == "custom":
    ...
```

随着后端增加，核心模块需要不断修改。这样的系统有几个问题：

1. 核心逻辑与具体实现强耦合；
2. 新增后端需要修改已有代码；
3. 测试范围不断扩大；
4. 第三方扩展难以接入；
5. 配置中的字符串与实际实现缺少可靠关联。

每加入一种新后端，就必须修改核心逻辑。随着分支增加，核心模块会越来越难以测试、扩展和维护。

动态机制提供了另一种思路：

> 核心系统只定义接口与生命周期，具体实现通过注册、发现和加载机制接入。

核心逻辑不需要知道所有实现，只需要知道如何找到并调用它们。

### 1.2 连接鸿沟：静态代码与运行时之间的映射

Python 源码里的类和函数是静态存在的，但 AI-Infra 的实际行为几乎全由运行时环境决定：

- 当前机器有没有 CUDA？驱动版本是多少？
- 配置文件指定了哪种模型格式？
- 这条请求要路由到哪个模型版本？
- 某个可选插件装了没有？
- 当前部署启用了哪种缓存后端？

这些问题的答案在写代码时都不知道。所以系统必须建立一层从**名字**到**实现**的映射：

```text
"onnx"                       -> ONNXModelLoader
"tensorrt"                   -> TensorRTModelLoader
"redis"                      -> RedisCache
"model-v2"                   -> handle_model_v2
("POST", "/v1/chat/completions") -> chat_completion
```

这层映射就是后面两个应用场景的共同骨架——插件化和路由分发的区别，只在于 key 的空间不同：一个是插件名，一个是 `(method, path)`。

而围绕这层映射有四个必须回答的问题，它们构成了第三、四章的全部内容：

```text
名字从哪来？      → 配置 / 入口点 / 类声明
名字映射到什么？  → 注册表（值是类还是工厂？重名怎么办？）
实现何时进内存？  → 动态导入（启动全量？懒加载？失败算致命吗？）
映射到的可信吗？  → 契约（静态 Protocol + 加载期校验）
```

这四个问题语言不会替你回答，必须由系统设计者显式决定。

### 1.3 性能权衡：动态不能无限侵入热路径

动态机制都有开销：

- 属性查找比直接调用更间接；
- 签名反射需要构造 `Signature` 对象；
- 动态导入有一次性的初始化成本；
- 插件发现可能要扫描包或读取包元数据；
- 动态分发多出一次字典查找。

这些开销放在启动阶段完全可以忽略，放进每请求、每 token 的循环里就不行了（具体量级见 6.1 节的实测数据）。

所以动态机制的合理位置是：

```text
✓ 服务启动    ✓ 配置解析    ✓ 插件加载    ✓ 路由构建    ✓ 依赖装配
✗ 每请求分发  ✗ 每 token 回调  ✗ 张量计算循环  ✗ 高频属性访问
```

**动态决策尽量提前完成，并缓存为可以直接调用的对象。**


## 二、Python 动态机制工具箱

Python 的动态能力可以分成三类，正好对应三个动作：

| 类别 | 动作 | 核心 API | 回答的问题 |
|------|------|---------|-----------|
| 反射 | 观察（Observe） | `getattr`、`inspect`、`__dict__` | 这个对象 / 函数 / 类长什么样？ |
| 元编程 | 改造（Transform） | 装饰器、描述符、`__init_subclass__`、元类 | 怎么在定义时注入行为和约束？ |
| 动态加载 | 导入（Load） | `importlib`、入口点 | 怎么把还没 `import` 的实现拿进来？ |

三者构成一条完整链路：**先能观察，才能校验；先能改造，才能注册；先能导入，才能扩展。** 后面两章的插件系统和路由系统，用的就是这三类能力的组合。

### 2.1 反射：运行时观察

反射并不是某一个 API，而是一类能力：

> 程序在运行时检查自身结构，并根据检查结果动态执行操作。

Python 的反射能力主要来自：

- `type()`；
- `isinstance()` 和 `issubclass()`；
- `getattr()`、`setattr()`、`hasattr()`；
- `dir()`；
- `inspect` 模块；
- `__dict__`；
- `__annotations__`；
- `importlib`；
- `callable()`。

#### 动态访问属性

最简单的反射形式是根据字符串访问属性：

```python
class ModelConfig:
    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device


config = ModelConfig("llama", "cuda")

field_name = "device"
value = getattr(config, field_name)

print(value)  # cuda
```

如果属性可能不存在，可以提供默认值：

```python
batch_size = getattr(config, "batch_size", 1)
```

在配置系统、序列化框架和指标采集中，这种能力非常常见。

例如，自动读取对象中的可观测字段：

```python
def collect_public_attributes(obj: object) -> dict[str, object]:
    result: dict[str, object] = {}

    for name in dir(obj):
        if name.startswith("_"):
            continue

        value = getattr(obj, name)

        if not callable(value):
            result[name] = value

    return result
```

但这段代码也暴露了反射的第一个风险：`getattr()` 可能触发自定义属性访问逻辑，甚至抛出异常或执行代价很高的操作。

因此，反射代码不能默认认为“读取属性是无副作用的”。

#### `inspect`：检查函数和调用约定

`inspect` 适合构建调试工具、依赖注入系统和接口校验工具。

```python
import inspect


def load_model(
    model_name: str,
    device: str = "cuda",
    *,
    batch_size: int = 1,
) -> object:
    ...


signature = inspect.signature(load_model)

for name, parameter in signature.parameters.items():
    print(name, parameter.annotation, parameter.default)
```

可以利用它检查一个实现是否满足调用约定：

```python
def validate_loader(loader: object) -> None:
    if not callable(loader):
        raise TypeError("loader must be callable")

    signature = inspect.signature(loader)
    parameters = signature.parameters

    if "model_name" not in parameters:
        raise TypeError("loader must accept model_name")
```

需要注意的是，运行时反射校验不能取代静态类型检查。它们解决的是不同问题：

- 静态类型检查：在开发阶段发现接口不匹配；
- 运行时反射：在组件动态加载后确认实际对象结构。

成熟系统通常会同时使用二者。

#### `__dict__` 与类的内部结构

Python 对象通常会暴露自己的属性字典：

```python
class Worker:
    max_batch_size = 16

    def __init__(self) -> None:
        self.device = "cuda"
        self.running = True


worker = Worker()

print(worker.__dict__) # {'device': 'cuda', 'running': True}
print(Worker.__dict__) # {'__module__': '__main__', 'max_batch_size': 16, '__init__': <function Worker.__init__ at 0x...>, '__dict__': <attribute '__dict__' of 'Worker' objects>, '__weakref__': <attribute '__weakref__' of 'Worker' objects>, '__doc__': None}
```

实例的 `__dict__` 通常保存实例属性，类的 `__dict__` 则包含方法、类属性和描述符。

不过，不能假设所有对象都有 `__dict__`：

- 使用 `__slots__` 的类可能没有；
- 部分 C 扩展对象不提供；
- 代理对象可能自定义属性访问行为。

更稳妥的代码应使用公开接口或 `getattr()`，而不是把 `__dict__` 当作稳定协议。

### 2.2 元编程：运行时改造

如果反射是"查询已有的结构"，元编程就是"参与结构的构造"。

Python 提供了一组侵入性递增的钩子，让代码可以在**类和函数被定义的时刻**介入：

```text
装饰器            → 包装已定义好的函数 / 类
描述符            → 接管某个属性的读写
__init_subclass__ → 在子类创建后执行逻辑
元类              → 控制类本身的创建过程
```

这四者能力递增、代价也递增。下面按从轻到重的顺序展开，选型依据留到 2.4 节统一给出。而它们能够存在的共同前提，是 Python 里一个容易被忽略的事实：类本身也是对象。

#### 类也是对象：理解 `type`

Python 中，实例是对象，类本身也是对象。

```python
class Model:
    pass


model = Model()

print(type(model))  # Model
print(type(Model))  # type
```

`type` 不只是用来查询类型，也可以动态创建类：

```python
Model = type(
    "Model",
    (),
    {
        "name": "demo",
        "describe": lambda self: f"model={self.name}",
    },
)

model = Model()
print(model.describe())
```

动态创建类在框架中有一定用途，例如：

- 根据协议生成适配器；
- 根据配置构造代理类型；
- 动态生成数据模型；
- 创建带有特定元数据的组件。

但它也会降低代码的可读性。除非确实需要动态生成类型，否则优先使用普通类、工厂函数或组合模式。

#### 装饰器：注入横切逻辑

> 装饰器的基础用法在[《Python 语言机制与运行时原理》](/python-language-mechanisms-and-runtime-internals.html)中已经覆盖，这里聚焦类型安全的装饰器写法和工程边界问题。

装饰器本质上是一个接收可调用对象并返回新可调用对象的函数。

```python
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def traced(
    func: Callable[P, R],
) -> Callable[P, R]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        print(f"calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"completed {func.__name__}")
        return result

    return wrapper
```

装饰器可以用于实现横切能力：

- 日志；
- 指标；
- tracing；
- 重试；
- 限流；
- 缓存；
- 权限检查；
- 资源生命周期管理。

例如，为推理函数增加耗时统计：

```python
import time
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def measure_async(
    func: Callable[P, Awaitable[R]],
) -> Callable[P, Awaitable[R]]:
    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()

        try:
            return await func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            print(f"{func.__name__}: {elapsed:.4f}s")

    return wrapper
```

这里的 `ParamSpec` 和 `TypeVar` 能够保留装饰器的类型信息，避免把被装饰函数退化成普通的 `Callable[..., Any]`。

##### 装饰器的工程边界

装饰器适合增强行为，但不适合隐藏核心业务流程。

不推荐：

```python
@retry
@cache
@trace
@limit
@fallback
async def infer(...):
    ...
```

当装饰器层数过多时，很难判断：

- 哪一层负责捕获异常；
- 重试发生在缓存之前还是之后；
- 指标统计是否包含排队时间；
- 取消异常是否被吞掉；
- 函数签名是否被正确保留。

更好的方式是：

- 控制装饰器数量；
- 给装饰器明确命名；
- 对异常和取消行为写测试；
- 将复杂策略显式组合为对象或中间件链。

#### 描述符：属性访问背后的机制

> 描述符的基本原理在[《Python 语言机制与运行时原理》](/python-language-mechanisms-and-runtime-internals.html)中已经介绍，这里侧重它在元编程和插件系统中的应用。

描述符是 Python 属性系统的重要基础。实现了以下任意方法的对象，都可以参与属性访问控制：

- `__get__`；
- `__set__`；
- `__delete__`。

`property`、方法、类方法和静态方法，都建立在描述符机制之上。

一个简单的验证描述符如下：

```python
class PositiveInteger:
    # __set_name__ 在类创建阶段由元类自动调用，
    # 将属性名注入描述符——这本身就是一种元编程钩子。
    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(
        self,
        instance: object | None,
        owner: type | None = None,
    ) -> int | "PositiveInteger":
        if instance is None:
            return self
        return instance.__dict__.get(self.name, 0)

    def __set__(self, instance: object, value: int) -> None:
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{self.name} must be a positive integer")

        instance.__dict__[self.name] = value


class BatchConfig:
    batch_size = PositiveInteger()

    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size
```

使用时：

```python
config = BatchConfig(16)
print(config.batch_size)

config.batch_size = -1
# ValueError
```

在 AI-Infra 中，描述符可以用于：

- 配置字段校验；
- 延迟初始化；
- 资源句柄管理；
- 自动生成指标；
- 将字段访问映射到远程状态；
- 实现缓存属性。

但描述符会改变普通属性访问的语义。使用时应明确其生命周期、线程安全和异常行为。

#### `__init_subclass__`：更轻量的自动注册

`__init_subclass__` 可以在子类创建后执行逻辑，通常比元类更简单。

```python
class Backend:
    registry: dict[str, type["Backend"]] = {}
    backend_name: str | None = None

    def __init_subclass__(
        cls,
        *,
        name: str | None = None,
        **kwargs,
    ) -> None:
        super().__init_subclass__(**kwargs)

        if name is not None:
            if name in cls.registry:
                raise ValueError(
                    f"backend already registered: {name}"
                )

            cls.registry[name] = cls
            cls.backend_name = name
```

定义插件：

```python
class TorchBackend(Backend, name="torch"):
    pass


class TensorRTBackend(Backend, name="tensorrt"):
    pass
```

查询：

```python
backend_cls = Backend.registry["torch"]
```

这种方式适用于：

- 自动登记子类；
- 检查必需的类属性；
- 构建简单的策略注册系统；
- 实现声明式组件定义。

不过，它同样依赖模块导入。没有导入的子类不会自动注册。

#### 元类：控制类的创建过程

元类是“创建类的类”。默认情况下，Python 中类的元类是 `type`。

元类可以介入类的创建过程：

```python
class RegistryMeta(type):
    # 所有使用此元类的类共享同一个 registry——
    # 这是元类的常见模式，因为元类实例本身就是类。
    registry: dict[str, type] = {}

    def __new__(
        mcls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict,
    ):
        cls = super().__new__(mcls, name, bases, namespace)

        if name != "BaseBackend":
            backend_name = namespace.get("backend_name")
            if backend_name:
                mcls.registry[backend_name] = cls

        return cls
```

使用：

```python
class BaseBackend(metaclass=RegistryMeta):
    backend_name: str | None = None


class TorchBackend(BaseBackend):
    backend_name = "torch"
```

此时：

```python
print(RegistryMeta.registry["torch"])
```

##### 元类的优点

- 子类定义时自动完成注册；
- 可以统一检查类结构；
- 可以生成或修改类属性；
- 适合构建 ORM、序列化框架和声明式 DSL。

##### 元类的缺点

- 语义隐式；
- 调试成本高；
- 多重继承时容易发生元类冲突；
- 注册时机与模块导入强相关；
- 继承层次复杂后不易维护。

在现代 Python 工程中，很多原本需要元类的场景，都可以使用以下方式替代：

- `__init_subclass__`；
- 类装饰器；
- 显式注册函数；
- 普通工厂；
- 组合而不是继承。

### 2.3 动态加载：运行时导入

Python 的模块导入本身就是一种运行时机制。

```python
import importlib

module = importlib.import_module("my_package.backends.torch")
backend_class = getattr(module, "TorchBackend")
```

可以根据配置动态加载模块：

```python
def load_object(path: str) -> object:
    module_name, object_name = path.rsplit(":", 1)

    module = importlib.import_module(module_name)
    return getattr(module, object_name)
```

配置：

```text
my_package.backends.torch:TorchBackend
```

加载：

```python
backend_cls = load_object(
    "my_package.backends.torch:TorchBackend"
)
```

这种机制很灵活，但也带来两个问题。

#### 风险一：错误延迟到运行时

拼写错误、模块不存在、对象名称错误，都只能在加载时发现。

#### 风险二：任意导入

如果模块路径直接来自不可信输入，动态导入可能带来严重安全问题。生产系统应：

- 只允许预先声明的插件；
- 对模块名建立白名单；
- 不允许用户直接提交任意 Python 路径；
- 将第三方插件放在隔离环境中；
- 对加载失败提供清晰错误信息。

动态加载是扩展机制，不应被当作任意代码执行接口。

### 2.4 关键决策点：如何避免滥用动态机制

工具箱到这里就齐了。但"能用"和"该用"是两件事——动态机制用错位置，扩展性的收益会被可维护性的损失吃掉。

这一节给两条决策线：**横向**选哪个元编程机制，**纵向**决定动态代码该出现在系统的哪个阶段。

#### 元编程机制选型：从轻到重

前面几种机制的侵入性是递增的。同一个"子类自动注册"的需求，四种写法都能实现，但代价并不相同：

| 机制 | 侵入性 | 适用场景 | 主要代价 |
|------|--------|---------|---------|
| 显式注册函数 | 最低 | 插件数量少、来源可控 | 需要手工维护一份注册表 |
| 装饰器 / 类装饰器 | 低 | 注册、横切能力、能力标记 | 依赖模块被导入才生效 |
| 描述符 | 中 | 字段级校验、延迟加载、属性映射 | 改变了普通属性访问的语义 |
| `__init_subclass__` | 中 | 子类自动登记、类属性校验 | 只对子类生效，同样依赖导入 |
| 元类 | 最高 | 声明式 DSL、ORM、需要改写类命名空间 | 语义隐式、多继承冲突、调试成本高 |

决策顺序建议**自上而下**，只有当上一层确实做不到时才下沉：

1. 能用显式注册函数就别用装饰器——显式代码永远最好读，也最好测；
2. 需要"定义即注册"的声明式体验，用**类装饰器**；
3. 需要作用于**所有子类**（包括第三方后来定义的那些），用 `__init_subclass__`；
4. 需要在**类创建之前**改写命名空间，或需要自定义类级别的 `__getattr__`、`isinstance` 行为，才用元类。

一条经验：如果写元类的目的只是"在类定义时做点什么"，那么 `__init_subclass__` 或类装饰器几乎总是更好的选择。元类真正不可替代的场景，是需要控制类**本身**的行为，而不是实例的行为。

> `type[C]` 作为类型标注如何表达"类对象本身"，以及它与元类的关系，见[《Python 类型系统与数据契约设计》](/python-type-system-and-data-contract-design.html)。

#### 动态边界：初始化可以动态，热路径必须静态

第二条决策线更重要，因为它决定的是系统结构而不是写法：

> 初始化阶段可以动态，热路径必须静态；系统边界可以动态，核心算法必须显式。

具体划分如下。

**初始化阶段（允许动态）**

- 动态导入与包扫描；
- `inspect` 签名分析；
- 插件接口校验；
- 路由注册与预编译；
- 依赖装配；
- 配置到对象的转换。

**运行阶段（尽量静态）**

- 已缓存的字典映射；
- 已绑定的方法引用；
- 预先构建好的参数绑定器；
- 明确的数据结构与直接的方法调用。

这条边界会在后面以三种形式反复出现：第三章的插件启动期校验、第四章的路由预编译、6.1 节的热路径实测开销。它是本文最核心的一条工程原则。

## 三、应用一：组件注册与插件化管理

第一个应用场景：让系统在不修改核心代码的前提下，接入新的模型格式、推理后端、缓存实现和监控组件。

这一章按真实的构建顺序展开：先有注册表（名字到实现的映射），再有发现机制（让注册代码真正被执行），然后是契约校验（确认找到的东西能用），最后把这些收敛成一个有边界、有生命周期、能演进的插件系统。

### 3.1 核心需求：不修改核心逻辑支持新后端

假设一个模型服务需要支持多种模型格式。最初的实现通常是这样：

```python
def create_loader(format_name: str):
    if format_name == "onnx":
        return ONNXLoader()
    elif format_name == "tensorrt":
        return TensorRTLoader()
    else:
        raise ValueError(f"不支持的格式: {format_name}")
```

实现只有两三种时这样写没问题。但随着后端增加会出现五个问题：

1. 每加一种格式都要改核心模块；
2. 分支越来越长，工厂函数变成新的"上帝函数"；
3. 所有后端的依赖被迫集中在同一个模块——导入 `create_loader` 就等于导入 TensorRT；
4. 测试矩阵按后端数量线性膨胀；
5. 第三方无法在不改你代码的前提下接入新格式。

第三点在 AI-Infra 里尤其致命：`import tensorrt` 在没装 TensorRT 的机器上直接失败，于是整个模型服务在 CPU 机器上都起不来。

解决方向就是把这个 `if/elif` 换成一层间接：注册表。

### 3.2 从 `if-else` 到注册表

注册表是 AI-Infra 中最常见的插件化基础设施。

#### 一个简单的注册表

```python
from collections.abc import Callable
from typing import Any


class BackendRegistry:
    def __init__(self) -> None:
        self._factories: dict[
            str,
            Callable[..., Any],
        ] = {}

    def register(
        self,
        name: str,
        factory: Callable[..., Any],
    ) -> None:
        if name in self._factories:
            raise ValueError(f"backend already registered: {name}")

        self._factories[name] = factory

    def create(self, name: str, **kwargs: Any) -> Any:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._factories))
            raise ValueError(
                f"unknown backend {name!r}; "
                f"available: {available}"
            ) from exc

        return factory(**kwargs)
```

定义统一接口：

```python
from typing import Protocol


class InferenceBackend(Protocol):
    async def infer(self, request: dict) -> dict:
        ...
```

注册具体实现：

```python
class TorchBackend:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    async def infer(self, request: dict) -> dict:
        return {"backend": "torch", "result": request}


registry = BackendRegistry()
registry.register("torch", TorchBackend)

backend = registry.create(
    "torch",
    model_path="/models/demo",
)
```

核心代码只依赖注册表和协议，而不依赖具体后端。

#### 用装饰器完成注册

```python
registry = BackendRegistry()


def register_backend(name: str):
    def decorator(factory):
        registry.register(name, factory)
        return factory

    return decorator


@register_backend("torch")
class TorchBackend:
    ...
```

这种写法简洁，但需要理解一个重要事实：

> 装饰器只有在定义它的模块被导入时才会执行。

如果插件模块没有被导入，注册动作就不会发生。

这也是许多“明明写了注册装饰器，但运行时找不到组件”问题的根源。

#### 装饰器注册与 `__init_subclass__` 注册的取舍

装饰器注册和 `__init_subclass__` 注册（2.2 节）都能做到"定义即注册"，选择标准是**插件之间有没有共同基类**。

**装饰器注册**

```python
@LOADERS.register("onnx")
class ONNXLoader:
    ...
```

- 可以注册类、函数和任意对象；
- 不要求插件继承任何基类，第三方实现更自由；
- 同一个类可以注册进多个注册表（既是 loader 又是 exporter）；
- 注册表与继承体系完全解耦。

**`__init_subclass__` 注册**

```python
class ONNXLoader(ModelLoader, format_name="onnx"):
    ...
```

- 注册逻辑集中在基类，插件侧只需一行声明；
- 可以在子类创建时顺带做结构校验（必需的类属性、必需的方法）；
- 天然表达"这一族组件"的归属关系；
- 但插件必须继承你的基类——这对第三方是一种侵入。

| | 装饰器 | `__init_subclass__` |
|---|---|---|
| 可注册对象 | 类、函数、实例 | 只能是子类 |
| 对插件的侵入 | 无（加一行装饰） | 必须继承基类 |
| 结构校验时机 | 需要额外写 | 类创建时天然可做 |
| 注册进多个注册表 | 容易 | 困难 |
| 适合 | 开放式扩展、异构组件 | 强约束的同族组件 |

经验判断：**面向第三方的扩展点用装饰器，内部同族组件用 `__init_subclass__`。**

但两者都依赖同一件事——模块必须被导入过，这就是下一节的主题。

### 3.3 插件发现机制

注册表有一个前提，前面已经反复提到，这里正式说清楚：

> 注册代码必须被执行。而在 Python 里，"被执行"等价于"所属模块被导入过"。

装饰器在模块导入时才运行，`__init_subclass__` 在子类定义时才触发。如果插件模块从未被导入，注册表就是空的——这是"明明写了注册装饰器，运行时却找不到组件"这类问题的唯一根源。

所以插件系统必须显式回答一个问题：**谁负责导入插件模块？** 有三种答案。

#### 显式导入

最简单的做法是在一个集中的位置手工导入：

```python
# my_project/plugins/__init__.py
def load_builtin_plugins() -> None:
    from my_project.plugins import onnx_loader      # noqa: F401
    from my_project.plugins import tensorrt_loader  # noqa: F401
```

优点是完全可控：导入顺序确定、失败点明确、断点好打、`grep` 就能查到调用关系。缺点是每加一个插件都要改这份列表，而且第三方插件无法通过这种方式接入。

一个在 AI-Infra 里很实用的变体是让导入失败可容忍，因为不同机器的硬件依赖不同：

```python
import importlib
import logging

logger = logging.getLogger(__name__)

OPTIONAL_PLUGINS = (
    "my_project.plugins.onnx_loader",
    "my_project.plugins.tensorrt_loader",
)


def load_builtin_plugins() -> None:
    for module_name in OPTIONAL_PLUGINS:
        try:
            importlib.import_module(module_name)
        except ImportError as exc:
            # 没装 TensorRT 的 CPU 机器不该因此起不来
            logger.warning("plugin %s unavailable: %s", module_name, exc)
```

注意这里只放过 `ImportError`。插件模块里真正的逻辑错误（`TypeError`、重名注册）应该继续抛出——插件加载的失败分级在 6.5 节展开。

#### 包扫描：`pkgutil.iter_modules`

如果插件都在同一个包下，可以让系统自己扫描：

```python
import importlib
import pkgutil
from types import ModuleType


def load_plugins(package: ModuleType) -> list[str]:
    loaded: list[str] = []

    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name.startswith("_"):
            continue

        module_name = f"{package.__name__}.{module_info.name}"
        importlib.import_module(module_name)
        loaded.append(module_name)

    return loaded
```

`pkgutil.iter_modules()` 只列出包的**直接**子模块。需要递归子包时用 `pkgutil.walk_packages()`，但要注意它必须**导入中间包**才能继续往下走，副作用比前者大得多。

包扫描省掉了维护导入列表的负担，代价是引入几个新问题：

- **导入顺序不确定**：`iter_modules()` 的顺序取决于文件系统，插件之间有依赖时会踩坑；
- **失败面变大**：任何一个子模块导入出错都会中断整个扫描，需要逐个 `try`；
- **副作用不可控**：你不再清楚启动时到底执行了哪些顶层代码；
- **对打包不友好**：`__path__` 在 zip 包、PyInstaller 单文件、某些 namespace package 布局下行为不一致。

所以包扫描适合"插件数量多、都在自己仓库里、且彼此独立"的场景，不适合作为对外的扩展点。

#### 入口点：面向发行包的插件发现

如果插件需要独立发布，可以使用 Python 包的入口点机制。

插件包声明入口点：

```toml
[project.entry-points."my_ai.backends"]
torch = "my_package.torch_backend:TorchBackend"
tensorrt = "my_package.trt_backend:TensorRTBackend"
```

主程序发现插件：

```python
from importlib.metadata import entry_points


def discover_backends() -> dict[str, object]:
    discovered: dict[str, object] = {}

    for entry_point in entry_points(
        group="my_ai.backends"
    ):
        discovered[entry_point.name] = entry_point.load()

    return discovered
```

入口点适合：

- 插件独立打包；
- 插件由不同团队维护；
- 运行时根据已安装包自动发现扩展；
- 构建可插拔的 CLI、存储后端和模型后端。

> 同一套入口点机制还用于声明命令行入口（`[project.scripts]`）。`pyproject.toml` 的完整配置与包的构建发布流程，见[《Python 项目工程化与生产交付》](/python-engineering-and-production-delivery.html)。

但使用入口点时要注意：

- 依赖包必须安装在当前环境；
- 插件加载可能执行导入级别代码；
- 插件版本需要兼容核心接口；
- 不能只验证“名称存在”，还要验证实际对象符合协议；
- 启动阶段加载过多插件会增加服务启动时间。

对于大型服务，可以采用延迟加载：

```python
class LazyPlugin:
    def __init__(self, entry_point) -> None:
        self.entry_point = entry_point
        self._loaded = None

    def load(self):
        if self._loaded is None:
            self._loaded = self.entry_point.load()
        return self._loaded
```

#### 三种发现方式的对比

三种方式并不互斥。成熟系统通常同时用两种：内置插件走显式导入（可控），第三方插件走入口点（免改代码）。

| | 显式导入 | 包扫描 | 入口点 |
|---|---|---|---|
| 谁能加插件 | 只有你 | 只有你 | 任何人，装个包即可 |
| 是否要改核心代码 | 要 | 不要 | 不要 |
| 插件是否需独立打包 | 不需要 | 不需要 | **需要** |
| 导入顺序 | 完全可控 | 不确定 | 不确定 |
| 启动成本 | 最低 | 与模块数成正比 | 读包元数据，可懒加载 |
| 失败定位 | 最容易 | 较难 | 较难（跨包） |
| 适合 | 内置组件 | 自有仓库内的大量插件 | 对外开放的扩展点 |

选择顺序建议：**默认用显式导入；插件多到列表难维护时引入包扫描；需要让别人扩展时才上入口点。** 不要一上来就用最灵活的那个。

### 3.4 契约保障：启动阶段验证插件接口

契约是插件系统里最容易被跳过、却最不该跳过的一环。注册表和发现机制解决的是"能不能找到实现"，契约解决的是"找到的东西能不能用"——尤其当实现来自动态导入、核心代码从未见过它的时候。

它同时需要静态和运行时两种手段：`Protocol` 在开发阶段约束实现，`inspect` 在插件加载阶段确认实际对象结构。

> `Protocol`、类型标注和运行时校验的详细机制，参见[《Python 类型系统与数据契约设计》](/python-type-system-and-data-contract-design.html)。

反射提供运行时灵活性，类型系统提供开发阶段约束。二者并不是互相替代，而是互相补充。

例如，插件加载后可以先进行运行时检查：

```python
from typing import cast


def load_backend(obj: object) -> InferenceBackend:
    if not hasattr(obj, "infer"):
        raise TypeError("backend must provide infer()")

    return cast(InferenceBackend, obj)
```

但仅使用 `hasattr()` 还不够。更完整的校验可以检查：

```python
import inspect


def validate_backend(backend_cls: type) -> None:
    infer = getattr(backend_cls, "infer", None)

    if infer is None or not callable(infer):
        raise TypeError("backend must define infer()")

    signature = inspect.signature(infer)

    if "request" not in signature.parameters:
        raise TypeError(
            "infer() must accept request parameter"
        )
```

类型检查工具则可以在开发阶段发现：

- 方法参数不匹配；
- 返回值类型不匹配；
- 异步接口被错误实现为同步接口；
- 插件没有实现协议所要求的方法。

推荐的组合方式是：

```text
Protocol / 类型标注
    ↓
开发阶段检查

反射 / 运行时校验
    ↓
插件加载时检查

集成测试
    ↓
真实行为检查
```

### 3.5 声明式收集与能力声明

声明式收集是契约的补充：契约规定"必须有哪些方法"，声明式收集则读取"这个实现声明了哪些能力和元数据"。

很多 AI-Infra 框架采用声明式写法：

```python
class ModelSpec:
    inputs = ["input_ids", "attention_mask"]
    outputs = ["logits"]
    supports_streaming = True
```

框架在启动时通过反射读取这些声明：

```python
def inspect_model_spec(spec_cls: type) -> dict[str, object]:
    return {
        "inputs": getattr(spec_cls, "inputs", []),
        "outputs": getattr(spec_cls, "outputs", []),
        "supports_streaming": getattr(
            spec_cls,
            "supports_streaming",
            False,
        ),
    }
```

这种方式能够减少重复代码，但要注意“声明”和“实际行为”可能不一致。

例如，插件声明支持流式输出，却在运行时返回完整结果。为避免配置漂移，应该：

- 在启动阶段验证声明；
- 在测试中验证能力；
- 在运行时对关键行为增加断言；
- 使用协议和类型标注表达接口；
- 将能力声明纳入版本兼容检查。

声明式设计的原则是：

> 静态声明用于发现和调度，运行时验证用于保证正确性。

### 3.6 插件系统的正式边界

一个可靠的插件系统至少需要定义以下内容：

#### 插件身份

插件使用稳定名称或唯一标识：

```text
backend.torch
scheduler.priority
storage.s3
```

名称应避免与内部模块路径强绑定。

#### 插件接口

接口可以使用 `Protocol` 表达：

```python
from typing import Protocol


class SchedulerPlugin(Protocol):
    def submit(self, task: dict) -> str:
        ...

    def cancel(self, task_id: str) -> None:
        ...

    def status(self, task_id: str) -> str:
        ...
```

#### 插件配置

配置应该明确区分：

- 插件名称；
- 插件版本；
- 插件参数；
- 运行时资源；
- 可选能力。

例如：

```python
config = {
    "name": "priority",
    "options": {
        "max_pending": 1000,
        "default_priority": 10,
    },
}
```

#### 生命周期

插件不仅有创建，还有启动、停止和销毁：

```python
from typing import Protocol


class LifecyclePlugin(Protocol):
    async def start(self) -> None:
        ...

    async def stop(self) -> None:
        ...
```

#### 能力声明

不同后端支持的能力可能不同：

```python
class BackendCapabilities:
    supports_streaming: bool
    supports_batching: bool
    supports_cancellation: bool
```

调度器不应该通过 `hasattr()` 到处猜测插件能力，而应使用显式能力声明。

### 3.7 插件版本与兼容性

插件系统最容易被忽略的问题之一，是接口演进。

假设核心系统最初定义：

```python
class Backend(Protocol):
    async def infer(self, request: dict) -> dict:
        ...
```

后来增加了：

```python
async def health_check(self) -> bool:
    ...
```

如果直接把新方法变成必需接口，旧插件可能全部无法加载。

一种方式是定义接口版本：

```python
from dataclasses import dataclass


@dataclass
class PluginMetadata:
    api_version: str = "2.0"
    plugin_version: str = "1.4.0"
```

加载时检查：

```python
def check_compatibility(metadata: PluginMetadata) -> None:
    if metadata.api_version.split(".")[0] != "2":
        raise RuntimeError(
            f"unsupported plugin api: {metadata.api_version}"
        )
```

另一种方式是使用能力协商：

```python
if plugin.capabilities.supports_streaming:
    await plugin.stream(request)
else:
    result = await plugin.infer(request)
```

实际工程中应区分三种版本：

- **插件版本**：插件自身发布版本；
- **核心框架版本**：主系统版本；
- **插件 API 版本**：双方约定的接口版本。

插件版本升级，不一定意味着 API 版本升级；核心框架升级，也不一定破坏插件 API。

### 3.8 热加载与动态更新

> 插件的异步生命周期管理（`start` / `stop` / 优雅关闭）涉及的并发机制，详见[《Python 并发、异步与任务协作》](/python-concurrency-asynchrony-and-task-collaboration.html)。

某些系统希望在不重启主服务的情况下加载新插件或更新策略。

热加载可能包括：

- 动态加载新的模型后端；
- 更新调度规则；
- 替换路由策略；
- 加载新的监控组件。

但热加载比“重新导入模块”复杂得多，因为必须处理：

- 已有请求是否继续使用旧实例；
- 新实例是否已经完成预热；
- 模型权重如何加载；
- 旧实例何时释放显存；
- 插件线程和后台任务如何停止；
- 旧版本连接是否还在使用；
- 更新失败后如何回滚。

一个更可靠的更新流程通常是：

```text
加载新插件
  → 校验接口和配置
  → 初始化资源
  → 执行健康检查
  → 切换路由
  → 等待旧请求完成
  → 销毁旧实例
```

这实际上类似于蓝绿发布，而不是简单的 `importlib.reload()`。

`importlib.reload()` 只会重新执行模块代码，并不能自动处理：

- 已创建对象；
- 旧类实例；
- 后台线程；
- 网络连接；
- GPU 资源；
- 其他模块持有的引用。

因此，生产系统应优先设计显式的插件生命周期，而不是依赖模块重载。

### 3.9 一个可维护的插件架构示例

下面给出一个简化的完整结构：

```text
ai_infra/
├── core/
│   ├── protocol.py
│   ├── registry.py
│   └── lifecycle.py
├── backends/
│   ├── torch_backend.py
│   └── tensorrt_backend.py
├── discovery/
│   └── entrypoints.py
└── service/
    └── server.py
```

接口定义：

```python
# core/protocol.py
from typing import Protocol


class InferenceBackend(Protocol):
    name: str

    async def start(self) -> None:
        ...

    async def infer(self, request: dict) -> dict:
        ...

    async def stop(self) -> None:
        ...
```

注册表：

```python
# core/registry.py
from collections.abc import Callable
from typing import Any


class Registry:
    def __init__(self) -> None:
        self._items: dict[str, Callable[..., Any]] = {}

    def register(
        self,
        name: str,
        factory: Callable[..., Any],
    ) -> None:
        if name in self._items:
            raise ValueError(f"duplicate plugin: {name}")

        self._items[name] = factory

    def get(self, name: str) -> Callable[..., Any]:
        try:
            return self._items[name]
        except KeyError as exc:
            raise LookupError(
                f"plugin not found: {name}"
            ) from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))
```

具体插件：

```python
# backends/torch_backend.py
class TorchBackend:
    name = "torch"

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    async def start(self) -> None:
        # 加载模型和初始化资源
        pass

    async def infer(self, request: dict) -> dict:
        return {
            "backend": self.name,
            "request": request,
        }

    async def stop(self) -> None:
        # 释放模型和设备资源
        pass
```

服务启动流程：

```python
async def start_backend(
    registry: Registry,
    name: str,
    config: dict,
) -> InferenceBackend:
    backend_factory = registry.get(name)
    backend = backend_factory(**config)

    await backend.start()
    return backend
```

这个结构将几个概念分开：

- `Protocol` 描述能力；
- `Registry` 管理发现；
- 插件实现具体功能；
- 生命周期由服务统一管理；
- 核心服务不依赖具体后端细节。

## 四、应用二：路由与请求分发

第二个应用场景，用来说明一件事：注册表这层间接不是插件系统的专利。

路由分发和插件化解决的是同一个问题——**名字到实现的映射**——只是 key 的空间从"插件名"换成了 `(method, path)`。把它单独拿出来讲，是因为它比插件系统多了一个约束：**它在热路径上**。插件加载一辈子只发生一次，路由分发每条请求都要走一遍。

所以这一章真正的主题不是"怎么写一个路由表"，而是 2.4 节那条边界原则在高频场景下的具体落地：**动态机制只负责在启动时"选择"，不负责在运行时"执行"。**

### 4.1 场景需求

一个模型服务通常要根据请求内容分发到不同处理逻辑：

- `GET /v1/models` 查询可用模型；
- `POST /v1/chat/completions` 对话生成；
- `POST /v1/embeddings` 向量化；
- `POST /v2/generate` 新版本接口；
- 同一路径下再根据模型名选择不同后端或不同权重版本。

如果分发逻辑全写成条件分支：

```python
def dispatch(path: str, request: dict) -> dict:
    if path == "/v1/chat/completions":
        return chat_completion(request)
    elif path == "/v1/embeddings":
        return embeddings(request)
    elif path == "/v1/models":
        return list_models(request)
    ...
```

问题和 3.1 节的 `create_loader` 一模一样：接口一多，分发层就变成新的"上帝模块"。而且还多了一个坏处——每个 handler 都被迫接收一个无类型的 `dict`，参数校验散落在各个函数内部。

### 4.2 基于注册表的动态路由

换成注册表，写法和 3.2 节的插件注册表几乎一样，只是 key 变成了二元组：

```python
from collections.abc import Callable
from typing import Any

Handler = Callable[..., Any]

ROUTES: dict[tuple[str, str], Handler] = {}


def route(path: str, method: str = "GET") -> Callable[[Handler], Handler]:
    def decorator(handler: Handler) -> Handler:
        key = (method.upper(), path)

        if key in ROUTES:
            existing = ROUTES[key]
            raise ValueError(
                f"duplicate route {method.upper()} {path}: "
                f"{existing.__module__}.{existing.__qualname__} "
                f"vs {handler.__module__}.{handler.__qualname__}"
            )

        ROUTES[key] = handler
        return handler

    return decorator
```

注意报错信息里带上了两个冲突 handler 的完整位置。路由重名是最难查的一类问题——两个模块各注册了同一个路径，谁先被导入谁生效——报错时不给出位置，排查成本会高一个数量级。

处理函数用普通的 Python 参数声明，而不是接收 `dict`：

```python
@route("/v1/models", method="GET")
def list_models() -> dict:
    return {"data": ["model-a", "model-b"]}


@route("/v1/chat/completions", method="POST")
def chat_completion(
    model: str,
    messages: list[dict],
    max_tokens: int = 128,
    temperature: float = 0.7,
    stream: bool = False,
) -> dict:
    ...
```

这一步是关键：**handler 的签名本身就是这个接口的参数契约**。下一节要做的，就是让框架去读取这份契约。

### 4.3 利用签名自动绑定请求参数

有了签名，就可以自动把请求体绑定到参数上，不用在每个 handler 里反复 `request.get(...)`。

一个正确的绑定器必须处理**参数种类**，这是最容易写错的地方：

```python
import inspect
from collections.abc import Callable
from typing import Any


def build_binder(handler: Callable[..., Any]):
    """启动阶段调用一次，返回一个只做字典查找的绑定函数。"""
    signature = inspect.signature(handler)

    required: list[str] = []
    optional: list[str] = []
    accepts_extra = False

    for name, parameter in signature.parameters.items():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            accepts_extra = True   # **kwargs：额外字段原样透传
            continue
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            continue               # *args：路由 handler 不支持，忽略
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
        else:
            optional.append(name)

    known = frozenset(required) | frozenset(optional)

    def bind(request: dict) -> dict:
        missing = [name for name in required if name not in request]
        if missing:
            raise ValueError(f"缺少必要参数: {', '.join(missing)}")

        arguments = {name: request[name] for name in required}
        arguments.update(
            (name, request[name]) for name in optional if name in request
        )

        if accepts_extra:
            arguments.update(
                (k, v) for k, v in request.items() if k not in known
            )

        return arguments

    return bind
```

三个容易踩的坑：

1. **`**kwargs` 的 `default` 也是 `empty`**。如果只用 `default is empty` 判断必填，任何带 `**kwargs` 的 handler 都会被误报"缺少参数 kwargs"。必须先看 `parameter.kind`。
2. **缺失参数要一次报全**。逐个 `raise` 会让客户端来回试错，一次返回全部缺失字段的体验完全不同。
3. **`inspect.signature()` 只能调一次**。它是微秒级操作（6.1 节有实测数据），放在请求路径上是纯浪费。

把绑定器和 handler 打包成一条"编译好"的路由：

```python
class CompiledRoute:
    __slots__ = ("handler", "bind")

    def __init__(self, handler: Callable[..., Any]) -> None:
        self.handler = handler
        self.bind = build_binder(handler)   # 签名分析只发生在这里

    def invoke(self, request: dict) -> Any:
        return self.handler(**self.bind(request))


# 启动阶段一次性编译
COMPILED_ROUTES: dict[tuple[str, str], CompiledRoute] = {
    key: CompiledRoute(handler) for key, handler in ROUTES.items()
}


def dispatch(method: str, path: str, request: dict) -> Any:
    route = COMPILED_ROUTES.get((method.upper(), path))

    if route is None:
        raise LookupError(f"未找到路由: {method} {path}")

    return route.invoke(request)   # 一次字典查找 + 一次字典构造
```

请求路径上剩下的动态操作只有一次 `dict.get()`，所有反射都发生在启动阶段。

> 这不是教学简化——**FastAPI 就是这么做的**。它在路由注册时（也就是装饰器执行时）用 `inspect` 分析 handler 签名，把每个参数解析成来源（path / query / body / 依赖项），构造出一个 `Dependant` 对象存进 `APIRoute`；请求到来时只是消费这份已经算好的计划。Pydantic 的校验器同样是在模型类定义时就编译好的。FastAPI 性能能打，很大程度来自这层"启动期编译"。

如果还需要类型转换和校验，不要自己写——把绑定结果交给 Pydantic：

```python
from pydantic import TypeAdapter

adapter = TypeAdapter(ChatRequest)   # 启动阶段构造，内部会编译校验器
payload = adapter.validate_python(request)
```

> 用类型标注驱动运行时校验的完整做法，见[《Python 类型系统与数据契约设计》](/python-type-system-and-data-contract-design.html)。

### 4.4 `getattr` 与映射表的选择

另一种常见的路由写法是靠命名约定，用 `getattr` 拼出方法名：

```python
class APIHandler:
    def handle_models(self, request: dict) -> dict:
        ...

    def handle_chat(self, request: dict) -> dict:
        ...


def dispatch(action: str, request: dict):
    handler = APIHandler()
    method = getattr(handler, f"handle_{action}")   # 危险
    return method(request)
```

这种写法在处理器结构稳定、命名规则严格时能用，但有一个致命问题：**`action` 来自请求，等于把方法名的一部分交给了外部输入。**

```python
dispatch("chat", req)             # 正常
dispatch("models", req)           # 正常
dispatch("chat.__globals__", req) # 这次 getattr 会走到哪里？
```

即使 `handle_` 前缀挡住了大部分情况，这仍然是一个由用户输入拼接出来的属性名。正确做法是显式映射表：

```python
HANDLERS: dict[str, Handler] = {
    "models": handle_models,
    "chat": handle_chat,
}


def dispatch(action: str, request: dict):
    try:
        handler = HANDLERS[action]
    except KeyError as exc:
        raise LookupError(
            f"unknown action {action!r}; "
            f"available: {', '.join(sorted(HANDLERS))}"
        ) from exc

    return handler(request)
```

映射表相比动态属性名的优势：

- **安全**：外部输入只能命中白名单里的 key；
- **可读**：所有路由一目了然，不需要在脑子里做字符串拼接；
- **可静态分析**：类型检查器知道 `HANDLERS` 的值类型，改名重构能被追踪；
- **可审计**：能直接 dump 出全部路由（6.6 节）；
- **报错友好**：能列出可用项。

一条通用规则，对插件系统同样适用：**用户输入可以决定"选哪个名字"，但不能决定"名字长什么样"。** 前者是查表，后者是代码注入的入口。这一点在 6.4 节会从安全角度再讲一次。

### 4.5 热路径优化：路由解析的扁平化

路由是全系统最热的一段代码，值得单独列出几条优化。它们本质上是同一件事：把动态操作往启动阶段推。

**1. 启动时完成路由构建**

不要在请求阶段扫描模块、检查装饰器或构造 handler 实例。上面的 `COMPILED_ROUTES` 就是为此存在。

**2. 缓存签名与绑定器**

`inspect.signature()` 一次约 3.7 微秒（6.1 节实测），是直接属性访问的数百倍。它只应该出现在 `CompiledRoute.__init__` 里。

**3. 启动时绑定依赖**

如果 handler 需要模型、tokenizer 或缓存客户端，用闭包在启动时装配好，而不是每次请求去容器里查：

```python
def build_generate_handler(model, tokenizer):
    def handler(prompt: str, max_tokens: int = 128) -> dict:
        inputs = tokenizer(prompt)
        return model.generate(inputs, max_tokens=max_tokens)

    return handler


ROUTES[("POST", "/v1/generate")] = build_generate_handler(model, tokenizer)
```

**4. 版本路由提前展开**

不要在请求里解析版本规则，把版本变成 key 的一部分：

```python
VERSIONED_ROUTES = {
    ("v1", "chat"): chat_v1,
    ("v2", "chat"): chat_v2,
}
```

**5. 用 `__slots__` 压缩热路径对象**

`CompiledRoute`、请求上下文这类每请求都要创建或频繁访问的小对象，`__slots__` 能同时省内存和属性查找时间。

> 属性访问的底层代价与 `__slots__` 的收益量级，见[《Python 内存管理与优化》](/python-memory-management-and-optimization.html)。

**6. 动态机制只负责"选择"，不负责"执行"**

这是前五条的总纲。动态路由的目标是**尽快找到那个最终函数**，而不是让整个请求处理过程都待在动态状态里。一旦拿到 handler，后面就应该是完全普通、可被类型检查、可被 profiler 看清的 Python 代码。

## 五、综合应用案例

前两章讲了插件化和路由分发。同一套机制还支撑着很多别的东西，列在这里说明它的通用性：

| 应用领域 | 核心机制 | 作用 | 真实例子 |
|---|---|---|---|
| 模型 / 后端加载 | 注册表 + 动态导入 | 按格式名选择加载后端 | transformers 的 `CONFIG_MAPPING`、vLLM 的 `_ModelRegistry` |
| API 路由 | 路由注册 + 签名绑定 | 请求映射到处理函数 | FastAPI 的 `APIRoute` / `Dependant` |
| 依赖注入 | 签名反射 + 类型标注 | 自动分析构造参数并装配 | FastAPI 的 `Depends`、Spring 的 `@Autowired` |
| 测试发现 | 动态导入 + 命名约定 | 自动收集并执行用例 | pytest 的 collection、`pluggy` 钩子 |
| 序列化派发 | 注册表 + 类型分发 | 按类型选择编解码器 | `functools.singledispatch`、`json.JSONEncoder.default` |
| 日志配置 | 字符串路径解析 | 配置文件里直接写类名 | `logging.config.dictConfig` 的 `"class"` 字段 |
| 异步任务 | 装饰器 + 队列注册 | 注册任务处理器并跟踪状态 | Celery 的 `@app.task` |
| 指标埋点 | 装饰器 + 描述符 | 自动记录耗时和调用次数 | Prometheus client 的装饰器 |
| 配置系统 | 描述符 + 类型校验 | 字段校验与延迟加载 | Pydantic Settings |
| 事件回调 | 注册表 + 回调查找 | 按事件名触发处理逻辑 | PyTorch 的 hook 机制 |

这些场景表面差别很大，骨架却是同一个：

> 建立一层"名字 → 实现"的可控映射，在启动阶段把它构造好并校验完，在运行阶段只做查表和调用。

值得注意的是最后一列——**这些都不是玩具**。`singledispatch` 和 `dictConfig` 就在标准库里，FastAPI 和 pytest 是各自领域的事实标准。动态机制不是"高级技巧"，它是 Python 生态基础设施的通用构造方式。

## 六、工程决策指南：权衡灵活性与可维护性

前面讲的是怎么用，这一章讲代价和边界。

动态机制的账单分三类：**性能**（热路径开销）、**可维护性**（调试变难、静态分析失效）和**安全**（不受限的动态执行）。下面八节按"先量化代价、再给出约束"的顺序展开。

### 6.1 边界清晰化：反射用于初始化，而非热路径

单次反射调用的开销以几十纳秒计，通常可以忽略。问题出在**它被放进了每请求、每 token 或每算子都要走一遍的路径**。

先看量级（CPython 3.9 / Apple Silicon，`timeit` 实测，仅作数量级参考）：

| 操作 | 耗时 | 相对直接访问 |
|------|------|-------------|
| `obj.x`（直接属性访问） | ≈ 10 ns | 1× |
| `obj.method`（绑定方法查找） | ≈ 15 ns | 1.5× |
| `getattr(obj, "x")` | ≈ 23 ns | 2.3× |
| `hasattr(obj, "method")` | ≈ 30 ns | 3× |
| `inspect.signature(f)` | ≈ 3700 ns | 约 370× |

`getattr()` 和 `hasattr()` 只是常数倍开销，真正危险的是 `inspect.signature()`——它比一次属性访问慢两个数量级以上，还会构造一批临时对象。`importlib.import_module()` 首次导入更是毫秒级。

三处典型的热路径反射：

```python
# 1. 每次请求都重新解析路径并导入
async def handle(request):
    backend = load_object(config["backend_path"])   # importlib + getattr
    return await backend().infer(request)

# 2. 每个 token 都做一次能力探测
async for token in stream:
    if hasattr(plugin, "on_token"):                # 每 token 一次 hasattr
        plugin.on_token(token)

# 3. 每次请求都重新校验签名
def dispatch(fn, payload):
    sig = inspect.signature(fn)                    # 微秒级，且可预先计算
    return fn(**{k: payload[k] for k in sig.parameters})
```

核心原则是：**把动态查找上移到初始化阶段，让热路径只做直接调用。**

```python
class Handler:
    def __init__(self, plugin: object) -> None:
        # 启动时一次性解析，反射结果被固化成普通引用
        self._backend = load_object(config["backend_path"])()
        self._on_token = getattr(plugin, "on_token", None)
        self._param_names = tuple(
            inspect.signature(self._backend.infer).parameters
        )

    async def handle(self, request: dict) -> dict:
        # 热路径上没有任何反射
        return await self._backend.infer(request)

    def emit(self, token: str) -> None:
        if self._on_token is not None:  # 只剩一次 None 判断
            self._on_token(token)
```

配套的三条实践：

- **缓存绑定方法**：`self._on_token = plugin.on_token` 把每次的属性查找降为一次局部变量读取；
- **用 `functools.lru_cache` 缓存解析结果**：适用于 `load_object()` 这类纯函数式的路径解析；
- **用 `__slots__` 替代 `__dict__`**：高频创建的小对象（如请求上下文）能同时节省内存和属性访问时间。

需要强调的是，这一节讨论的是**热路径**。插件发现、注册和校验发生在启动阶段，那里的反射开销无论多大都不值得优化——恰恰相反，应该把尽可能多的校验挪到启动阶段去做。

> 属性访问的底层代价、`__slots__` 的收益量级，以及热路径的 profiling 方法，见[《Python 内存管理与优化》](/python-memory-management-and-optimization.html)。

### 6.2 调试：栈追踪变得晦涩

静态代码里，异常栈是一条可读的调用链。动态代码里，这条链会被装饰器、代理和动态派发切碎。

看一个三层装饰器包裹的推理函数：

```python
@traced
@retry(times=3)
@measure_async
async def infer(request: dict) -> dict:
    raise ValueError("bad input")
```

抛出的栈里会出现三个都叫 `wrapper` 的帧，来自三个不同模块，而真正的业务帧只有最后一行。如果中间某一层写了 `except Exception: pass`，业务异常会彻底消失——连这条晦涩的栈都看不到。

三个能显著改善的做法。

**1. 永远使用 `functools.wraps`**

它会把 `__name__`、`__qualname__`、`__module__`、`__doc__` 和 `__wrapped__` 复制到 wrapper 上。其中 `__wrapped__` 尤其重要——`inspect.signature()` 会自动跟随它拿到原始签名：

```python
import inspect
from functools import wraps


def traced(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@traced
def infer(request: dict, *, timeout: float = 1.0) -> dict:
    ...


print(infer.__name__)            # infer，而不是 wrapper
print(inspect.signature(infer))  # (request: dict, *, timeout: float = 1.0) -> dict
print(infer.__wrapped__)         # <function infer at 0x...>
```

去掉 `@wraps`，这三行会分别变成 `wrapper`、`(*args, **kwargs)` 和 `AttributeError`。对于依赖签名做校验或依赖注入的框架，后两者是致命的。

**2. 用 `raise ... from` 保留因果链**

动态加载失败时，原始异常往往才是有用的那个：

```python
try:
    module = importlib.import_module(module_name)
except ImportError as exc:
    raise PluginLoadError(
        f"cannot load plugin {name!r} from {module_name!r}"
    ) from exc
```

`from exc` 会在栈里保留 "The above exception was the direct cause of..."，丢掉它等于丢掉根因——而动态导入失败的根因（依赖缺失、循环导入、C 扩展版本不匹配）几乎全在被丢掉的那一半里。

**3. 让注册表本身可查询**

动态系统最高频的故障是"组件没找到"。注册表报错时应该直接给出候选集合，而不是抛一个裸 `KeyError`——前面 `BackendRegistry.create()` 中的 `available: ...` 就是为此设计的。此外，启动阶段把已发现插件的完整清单打印一次，能省掉大量事后排查。

> 动态代码的调试手法（`pdb` 断点、`inspect` 现场取证、装饰器层的日志埋点）在[《Python 单元测试、问题定位与调试实践》](/python-unit-testing-troubleshooting-and-debugging.html)中有系统展开。

### 6.3 静态分析与 IDE 失效

这是动态性最容易被低估的代价：**类型检查器看不见运行时才生成的东西**。

三种典型的失效场景：

```python
# 1. 注册表返回 Any，下游全部丢失类型信息
backend = registry.create("torch")   # -> Any
backend.inferr(request)              # 拼错方法名，pyright 不报错

# 2. setattr 注入的属性，检查器不认识
for name, value in config.items():
    setattr(self, name, value)
self.batch_size                      # error: 属性不存在

# 3. 元类生成的类属性，补全为空
class User(Model):                   # 元类注入了 objects / fields
    name = CharField()

User.objects.filter(...)             # 检查器不知道 objects 存在
```

四种补救手段。

**1. 让注册表泛型化**

把协议作为类型参数，类型信息就能从注册表的另一端带出来：

```python
from typing import Protocol


class InferenceBackend(Protocol):
    async def infer(self, request: dict) -> dict:
        ...


class Registry[T]:  # Python 3.12+ 泛型语法；3.11 及以前用 Generic[T] + TypeVar
    def __init__(self) -> None:
        self._items: dict[str, type[T]] = {}

    def register(self, name: str, factory: type[T]) -> None:
        self._items[name] = factory

    def create(self, name: str, **kwargs: object) -> T:
        return self._items[name](**kwargs)


backends: Registry[InferenceBackend] = Registry()
```

此时 `backends.create("torch")` 的静态类型是 `InferenceBackend` 而不是 `Any`，前面那个 `inferr` 拼写错误会被当场抓住。

**2. 用 `Protocol` 而不是鸭子类型**

结构类型让检查器能验证插件实现，即使插件是运行时加载的。这正是 3.4 节强调契约的原因——契约不只是运行时的护栏，也是静态分析赖以工作的锚点。

**3. 给动态属性写 `.pyi` 存根**

如果某个类的属性确实由元类或 `setattr` 注入，唯一能让检查器和 IDE 理解它的办法是手写存根：

```python
# config.pyi
class ServiceConfig:
    batch_size: int
    device: str
    max_tokens: int
```

**4. 接受 `cast()`，但收窄它的作用范围**

`cast()` 是对检查器的单方面承诺：编译期无成本、运行时无校验。它应当紧跟在一次真实的运行时校验之后，而不是散落各处：

```python
validate_backend(obj)                    # 先在运行时确认结构
backend = cast(InferenceBackend, obj)    # 之后才向检查器做出承诺
```

> 泛型、`Protocol`、`.pyi` 存根与 `cast()` 的完整机制，见[《Python 类型系统与数据契约设计》](/python-type-system-and-data-contract-design.html)。

### 6.4 安全边界：动态能力不能突破信任边界

反射和插件机制本身不是安全问题，但“不受限制的动态执行”会变成安全问题。

需要重点关注：

#### 任意导入

不要允许外部用户直接提交：

```text
some_package.some_module:SomeClass
```

并由服务端直接导入执行。

#### 任意属性访问

不要把用户提供的字符串直接交给：

```python
getattr(obj, user_input)
```

如果必须支持动态字段，应建立白名单。

#### 任意表达式执行

避免使用：

```python
eval(expression)
exec(source_code)
```

即使对输入做简单过滤，也很难建立可靠的安全边界。

#### 插件权限

第三方插件可能访问：

- 文件系统；
- 网络；
- 环境变量；
- GPU；
- 进程和线程；
- 敏感凭据。

插件系统应明确其信任模型。必要时，可以通过独立进程、容器或更严格的运行时隔离降低风险。

### 6.5 故障隔离：插件错误不能拖垮整个服务

插件加载失败在 AI-Infra 里是常态而不是异常：

- 依赖包没装（`import tensorrt` 在纯 CPU 机器上）；
- CUDA 驱动版本不匹配；
- 插件顶层代码抛异常；
- 接口实现不完整；
- 与核心 API 版本不兼容。

关键问题是：**一个插件挂了，服务应该起不来，还是应该降级运行？**

答案取决于插件的等级，所以插件系统必须把这件事显式声明出来：

```python
import enum
import importlib
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class Criticality(enum.Enum):
    REQUIRED = "required"   # 加载失败 -> 拒绝启动
    OPTIONAL = "optional"   # 加载失败 -> 记录并降级


@dataclass
class PluginLoadResult:
    name: str
    criticality: Criticality
    loaded: bool
    error: str | None = None


@dataclass
class PluginLoader:
    results: list[PluginLoadResult] = field(default_factory=list)

    def load(
        self, name: str, module: str, criticality: Criticality
    ) -> None:
        try:
            importlib.import_module(module)
        except Exception as exc:
            if criticality is Criticality.REQUIRED:
                # 核心插件：立刻失败，并保留原始异常链
                raise RuntimeError(
                    f"required plugin {name!r} failed to load"
                ) from exc

            logger.warning("optional plugin %s unavailable: %r", name, exc)
            self.results.append(
                PluginLoadResult(name, criticality, False, repr(exc))
            )
            return

        self.results.append(PluginLoadResult(name, criticality, True))

    def degraded(self) -> list[str]:
        return [r.name for r in self.results if not r.loaded]
```

几个容易做错的地方：

1. **不要用裸 `except: pass`**。可选插件失败必须留下日志和结构化记录，否则"为什么这个功能没生效"会变成无从下手的问题。
2. **核心插件失败要 `raise ... from exc`**。丢掉原始异常链，等于丢掉"到底是依赖缺失还是驱动不匹配"这个唯一有用的信息。
3. **降级状态要能被外部看到**。把 `degraded()` 暴露到健康检查接口里——否则一个副本静默降级，流量照样打进来。
4. **降级不等于静默**。缓存插件加载失败可以继续跑，但应该同时上报一个指标，让告警系统知道当前处于降级状态。

一条边界：**故障隔离只处理"加载失败"，不处理"插件运行时行为异常"。** 后者需要的是超时、熔断和资源限额；如果插件来自不完全可信的第三方，隔离手段应该是独立进程或容器，而不是 `try/except`（见 6.4 节）。

### 6.6 可观测性：动态系统必须能够被审计

至少记录：

- 发现了哪些插件；
- 加载了哪个版本；
- 初始化耗时；
- 初始化是否成功；
- 当前使用的实现；
- 插件切换事件；
- 插件关闭是否完成。

没有可观测性的动态系统，出了问题很难定位。

这些信息应该能被主动查询，而不是只存在于启动日志里：

```python
import inspect


def dump_registry(registry: dict[str, type]) -> list[dict[str, str]]:
    return [
        {
            "name": name,
            "implementation": f"{obj.__module__}.{obj.__qualname__}",
            "source": getattr(
                inspect.getmodule(obj), "__file__", "<unknown>"
            ),
        }
        for name, obj in sorted(registry.items())
    ]
```

三个建议：

- **启动时打印一次完整清单**，包括插件名、实现类的完整路径、来源文件和版本。这一条日志能省掉的排查时间，远超它占的篇幅。
- **暴露一个管理端点**（如 `/admin/plugins`、`/admin/routes`），返回上面的结构化结果加降级状态。线上排查"这个副本到底装了什么"时，这比翻日志快得多。
- **把发现数量做成指标**。插件数或路由数突然变化，通常意味着某个依赖的安装状态变了——这类问题不做成指标几乎不可能被及时发现。

### 6.7 常见错误与改进方式

#### 注册逻辑依赖隐式导入

插件类虽然定义了注册装饰器，但模块从未被导入。

改进方式：

- 显式导入内置插件；
- 使用入口点自动发现；
- 启动阶段打印已发现插件列表；
- 对插件加载建立测试。

#### 只根据名称判断插件是否可用

插件名称存在，并不代表插件接口兼容。

改进方式：

- 加载时进行接口校验；
- 检查版本和能力；
- 执行启动健康检查。

#### 用 `hasattr()` 代替完整接口设计

```python
if hasattr(plugin, "stream"):
    ...
```

这会让调用方分散地猜测插件能力。

改进方式：使用显式能力声明或拆分协议：

```python
from collections.abc import AsyncIterator


class StreamingBackend(InferenceBackend, Protocol):
    async def stream(self, request: dict) -> AsyncIterator[dict]:
        ...
```

#### 元类承担过多逻辑

如果元类中同时负责注册、配置解析、日志、依赖注入和资源初始化，系统会很难理解。

改进方式：

- 元类只做必要的类级校验；
- 将注册逻辑独立出来；
- 将资源初始化放在生命周期方法中；
- 用组合替代深层继承。

#### 插件加载时执行重量级操作

不推荐在模块导入阶段直接加载模型：

```python
# import 时就加载数 GB 模型
model = load_large_model()
```

这样会导致：

- 导入变慢；
- 测试变慢；
- 资源初始化时机不可控；
- 导入失败影响整个服务。

更好的方式是延迟初始化：

```python
class Backend:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model = None

    async def start(self) -> None:
        self.model = await load_model(self.model_path)
```

#### 动态配置没有边界

配置中的字符串如果可以任意映射到模块、类和方法，系统实际上暴露了一个动态执行入口。

改进方式：

- 使用逻辑名称而不是任意 Python 路径；
- 建立插件白名单；
- 对配置做严格校验；
- 区分可信配置和用户输入。

### 6.8 设计原则

反射、元编程和插件化不是越多越好。可以遵循以下原则。

#### 稳定接口，动态实现

核心系统依赖稳定协议，具体实现可以动态发现和替换。

#### 显式边界，有限动态

动态能力应该集中在：

- 插件发现；
- 工厂创建；
- 配置映射；
- 能力协商；
- 生命周期管理。

不要让整个系统都依赖隐式反射。

#### 启动时失败，而不是请求时失败

如果插件名称错误、接口不兼容或配置无效，应该在服务启动阶段发现，而不是等到第一条线上请求才失败。

#### 动态加载，静态验证

即使插件是运行时加载的，也应该使用：

- `Protocol`；
- 类型标注；
- 静态类型检查；
- 运行时接口校验；
- 集成测试。

#### 生命周期必须显式

插件需要有明确的：

```text
创建 → 启动 → 使用 → 停止 → 销毁
```

特别是涉及模型、GPU、线程和网络连接时，不能把资源管理隐藏在导入副作用中。

## 附：Java 与 Python 动态机制对照

下表把本文涉及的动态能力与 Java 对应物并列。`层次`一列区分它属于语言机制还是架构决策。

| 层次 | 能力 | Java | Python |
|------|------|------|--------|
| 机制 | 运行时类型查询 | `instanceof`、`Class<?>` | `isinstance()`、`type()` |
| 机制 | 动态属性访问 | `Field.get()` / `Method.invoke()` | `getattr()` / `setattr()` |
| 机制 | 函数签名检查 | `java.lang.reflect.Parameter` | `inspect.signature()` |
| 机制 | 动态创建类 | `java.lang.reflect.Proxy`、字节码生成 | `type()` 三参数调用 |
| 机制 | 属性拦截 | 无原生机制（需 AOP / 动态代理） | 描述符协议（`__get__` / `__set__`） |
| 机制 | 横切关注点 | 注解 + AOP（Spring / AspectJ） | 装饰器 |
| 机制 | 自动注册子类 | 注解处理器（APT）/ SPI | `__init_subclass__` |
| 机制 | 类创建控制 | 无直接等价（类加载器可部分替代） | 元类（`__new__` / `__init__`） |
| 机制 | 按名加载实现 | `Class.forName()` + 反射实例化 | `importlib.import_module()` + `getattr()` |
| 架构 | 注册与发现 | `ServiceLoader`（SPI） | 注册表 + `entry_points` |
| 架构 | 接口契约 | `interface`（名义类型） | `Protocol`（结构类型） |
| 架构 | 请求路由 | Servlet 映射、Spring `@RequestMapping` | 路由注册表 + 签名绑定 |
| 架构 | 依赖装配 | Spring `@Autowired` / Guice binding | 签名反射 + 闭包预绑定 |
| 架构 | 插件生命周期 | OSGi Bundle 生命周期 / Spring Bean 回调 | 自定义 `start()` / `stop()` 协议 |
| 架构 | 插件热加载 | OSGi / 自定义 ClassLoader | `importlib.reload()`（有限） |
| 代价 | 安全沙箱 | `SecurityManager`（已废弃） | 无内建沙箱，依赖进程 / 容器隔离 |

有两点值得注意。

第一，**机制层 Python 明显更直接**：不需要反射 API 就能完成属性访问和类型操作，`getattr()` 就是个普通函数，描述符和元类更是没有 Java 等价物。但也因此更需要开发者自行建立边界，避免"什么都能做"变成"什么都看不懂"。

第二，**架构层两边高度相似**。`ServiceLoader` 和 `entry_points` 解决的是同一个问题，Spring 的 `BeanDefinition` 持有 `Class<?>` 与 Python 注册表持有类对象是同一个模式，`@RequestMapping` 扫描注解建路由表和第四章的 `@route` 装饰器也是同一件事。这恰好印证了本文的核心区分：**架构模式与语言无关**，真正因语言而异的只有底下那层机制。Java 用注解 + APT + SPI 拼出来的东西，Python 用装饰器 + `__init_subclass__` + `entry_points` 拼出来，形状是一样的。

## 七、结语：从语言技巧到架构能力

回到开头那个概念区分。

**反射和元编程是语言机制。** `getattr`、`inspect`、装饰器、描述符、`__init_subclass__`、元类，它们由 Python 提供，本身不构成任何架构——只是让代码在运行时能观察自己、改造自己。

**插件化和路由分发是架构模式。** 它们不引入任何新的语言特性，全部是工程决策：名字空间怎么定、发现机制选哪种、契约怎么校验、生命周期谁管、版本怎么协商、失败算不算致命。

把这两件事分开，才能看清一个规律：本文两个应用场景的骨架完全一样——

```text
定义契约 → 建立注册表 → 保证注册代码被执行 → 启动期校验 → 预编译 / 缓存 → 运行期只查表
```

插件化和路由的区别只在 key 的空间（插件名 vs `(method, path)`）和热度（一辈子一次 vs 每请求一次）。第五章那张表里的十个场景，走的都是这同一条链路。这也解释了为什么"多学几个动态特性"帮助有限，而"理解这条链路"能直接迁移到任何一个新场景。

至于取舍，全文可以压缩成一句：

> 初始化阶段动态，运行阶段稳定；系统边界灵活，核心路径显式；系统可扩展，同时可审计、可测试、可回滚。

展开就是 6.8 节那六条原则：稳定接口动态实现、显式边界有限动态、启动时失败而非请求时失败、动态加载静态验证、生命周期必须显式、可观测性必须覆盖动态系统。

当反射和元编程不再被当作 Python 的"高级技巧"，而是放进插件化、路由、依赖装配这些具体的架构语境里，它们才真正从语言能力变成 AI-Infra 中可复用、可演进的工程能力。
