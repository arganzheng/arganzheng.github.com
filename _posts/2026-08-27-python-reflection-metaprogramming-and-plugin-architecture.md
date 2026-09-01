---
layout: post
title: Python 在 AI-Infra（04）：反射、元编程与插件化机制
subtitle: Python Reflection, Metaprogramming, and Plugin Architecture in AI Infrastructure
tags: [Python]
catalog: true
---


在 AI-Infra 系统中，很多核心能力并不是通过一组固定的 `if/elif` 分支实现的。

模型类型可以动态注册，后端可以按名称加载，算子可以通过配置发现，调度器可以根据任务类型选择执行器，监控系统可以自动收集组件暴露的指标。

这些能力背后，通常依赖三类 Python 机制：

- **反射（Reflection）**：运行时检查和操作对象；
- **元编程（Metaprogramming）**：编写能够生成、修改或控制代码结构的代码；
- **插件化（Plugin Architecture）**：让系统在不修改核心代码的情况下发现和加载扩展组件。

它们共同构成了 Python 框架“动态性”的基础。

但动态性也意味着风险。过度依赖字符串、隐式注册和运行时修改，可能让系统变得难以调试、难以重构，甚至产生安全问题。

因此，本文不把重点放在“如何炫技”，而是讨论一个更实际的问题：

> 如何利用 Python 的动态机制构建可扩展的 AI-Infra 系统，同时保持类型约束、错误边界和运行时可控？

## 一、为什么 AI-Infra 需要动态机制？

一个模型服务系统可能需要支持：

- 多种模型架构；
- 多种推理后端；
- CPU、GPU 和其他加速设备；
- 不同版本的模型协议；
- 多种输入输出格式；
- 动态加载的调度策略；
- 可选的监控、缓存和限流组件。

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

更好的结构是：

```text
核心调度器
    ↓
统一接口
    ↓
注册表 / 插件发现机制
    ↓
具体后端实现
```

核心调度器只关心接口：

```python
backend = registry.create("tensorrt", config)
result = await backend.infer(request)
```

它不需要知道具体实现位于哪个模块，也不需要在代码中维护大量分支。

这就是反射、元编程和插件化的工程价值：**把“变化的实现”从“稳定的核心流程”中分离出来。**

## 二、反射：运行时理解和操作对象

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

### 1. 动态访问属性

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

### 2. `inspect`：检查函数和调用约定

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

### 3. `__dict__` 与类的内部结构

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

## 三、类也是对象：理解 `type`

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

## 四、描述符：属性访问背后的机制

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

## 五、装饰器：最实用的元编程入口

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

### 装饰器的工程边界

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

## 六、注册表：从条件分支到组件发现

注册表是 AI-Infra 中最常见的插件化基础设施。

### 1. 一个简单的注册表

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

### 2. 用装饰器完成注册

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

## 七、导入机制与自动发现

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

### 1. 错误延迟到运行时

拼写错误、模块不存在、对象名称错误，都只能在加载时发现。

### 2. 任意导入风险

如果模块路径直接来自不可信输入，动态导入可能带来严重安全问题。生产系统应：

- 只允许预先声明的插件；
- 对模块名建立白名单；
- 不允许用户直接提交任意 Python 路径；
- 将第三方插件放在隔离环境中；
- 对加载失败提供清晰错误信息。

动态加载是扩展机制，不应被当作任意代码执行接口。

## 八、插件系统的正式边界

一个可靠的插件系统至少需要定义以下内容：

### 1. 插件身份

插件使用稳定名称或唯一标识：

```text
backend.torch
scheduler.priority
storage.s3
```

名称应避免与内部模块路径强绑定。

### 2. 插件接口

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

### 3. 插件配置

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

### 4. 生命周期

插件不仅有创建，还有启动、停止和销毁：

```python
from typing import Protocol


class LifecyclePlugin(Protocol):
    async def start(self) -> None:
        ...

    async def stop(self) -> None:
        ...
```

### 5. 能力声明

不同后端支持的能力可能不同：

```python
class BackendCapabilities:
    supports_streaming: bool
    supports_batching: bool
    supports_cancellation: bool
```

调度器不应该通过 `hasattr()` 到处猜测插件能力，而应使用显式能力声明。

## 九、入口点：面向发行包的插件发现

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

## 十、元类：控制类的创建过程

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

### 元类的优点

- 子类定义时自动完成注册；
- 可以统一检查类结构；
- 可以生成或修改类属性；
- 适合构建 ORM、序列化框架和声明式 DSL。

### 元类的缺点

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

## 十一、`__init_subclass__`：更轻量的自动注册

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

## 十二、声明式框架与自动收集

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

## 十三、反射与类型系统如何协作

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

## 十四、插件版本与兼容性

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

## 十五、热加载与动态更新

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

## 十六、安全边界：动态能力不能突破信任边界

反射和插件机制本身不是安全问题，但“不受限制的动态执行”会变成安全问题。

需要重点关注：

### 1. 任意导入

不要允许外部用户直接提交：

```text
some_package.some_module:SomeClass
```

并由服务端直接导入执行。

### 2. 任意属性访问

不要把用户提供的字符串直接交给：

```python
getattr(obj, user_input)
```

如果必须支持动态字段，应建立白名单。

### 3. 任意表达式执行

避免使用：

```python
eval(expression)
exec(source_code)
```

即使对输入做简单过滤，也很难建立可靠的安全边界。

### 4. 插件权限

第三方插件可能访问：

- 文件系统；
- 网络；
- 环境变量；
- GPU；
- 进程和线程；
- 敏感凭据。

插件系统应明确其信任模型。必要时，可以通过独立进程、容器或更严格的运行时隔离降低风险。

## 十七、一个可维护的插件架构示例

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

## 十八、常见错误与改进方式

### 错误一：注册逻辑依赖隐式导入

插件类虽然定义了注册装饰器，但模块从未被导入。

改进方式：

- 显式导入内置插件；
- 使用入口点自动发现；
- 启动阶段打印已发现插件列表；
- 对插件加载建立测试。

### 错误二：只根据名称判断插件是否可用

插件名称存在，并不代表插件接口兼容。

改进方式：

- 加载时进行接口校验；
- 检查版本和能力；
- 执行启动健康检查。

### 错误三：用 `hasattr()` 代替完整接口设计

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

### 错误四：元类承担过多逻辑

如果元类中同时负责注册、配置解析、日志、依赖注入和资源初始化，系统会很难理解。

改进方式：

- 元类只做必要的类级校验；
- 将注册逻辑独立出来；
- 将资源初始化放在生命周期方法中；
- 用组合替代深层继承。

### 错误五：插件加载时执行重量级操作

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

### 错误六：动态配置没有边界

配置中的字符串如果可以任意映射到模块、类和方法，系统实际上暴露了一个动态执行入口。

改进方式：

- 使用逻辑名称而不是任意 Python 路径；
- 建立插件白名单；
- 对配置做严格校验；
- 区分可信配置和用户输入。

## 十九、设计原则：动态性应该服务于稳定性

反射、元编程和插件化不是越多越好。可以遵循以下原则。

### 1. 稳定接口，动态实现

核心系统依赖稳定协议，具体实现可以动态发现和替换。

### 2. 显式边界，有限动态

动态能力应该集中在：

- 插件发现；
- 工厂创建；
- 配置映射；
- 能力协商；
- 生命周期管理。

不要让整个系统都依赖隐式反射。

### 3. 启动时失败，而不是请求时失败

如果插件名称错误、接口不兼容或配置无效，应该在服务启动阶段发现，而不是等到第一条线上请求才失败。

### 4. 动态加载，静态验证

即使插件是运行时加载的，也应该使用：

- `Protocol`；
- 类型标注；
- 静态类型检查；
- 运行时接口校验；
- 集成测试。

### 5. 生命周期必须显式

插件需要有明确的：

```text
创建 → 启动 → 使用 → 停止 → 销毁
```

特别是涉及模型、GPU、线程和网络连接时，不能把资源管理隐藏在导入副作用中。

### 6. 可观测性必须覆盖动态系统

至少记录：

- 发现了哪些插件；
- 加载了哪个版本；
- 初始化耗时；
- 初始化是否成功；
- 当前使用的实现；
- 插件切换事件；
- 插件关闭是否完成。

没有可观测性的动态系统，出了问题很难定位。

## 附：Java 与 Python 动态机制对照

| 能力 | Java | Python |
|------|------|--------|
| 运行时类型查询 | `instanceof`、`Class<?>` | `isinstance()`、`type()` |
| 动态属性访问 | `Field.get()` / `Method.invoke()` | `getattr()` / `setattr()` |
| 函数签名检查 | `java.lang.reflect.Parameter` | `inspect.signature()` |
| 动态创建类 | `java.lang.reflect.Proxy`、字节码生成 | `type()` 三参数调用 |
| 属性拦截 | 无原生机制（需 AOP / 动态代理） | 描述符协议（`__get__` / `__set__`） |
| 横切关注点 | 注解 + AOP（Spring / AspectJ） | 装饰器 |
| 注册与发现 | `ServiceLoader`（SPI） | 注册表 + `entry_points` |
| 接口契约 | `interface`（名义类型） | `Protocol`（结构类型） |
| 自动注册子类 | 注解处理器（APT）/ SPI | `__init_subclass__` |
| 类创建控制 | 无直接等价（类加载器可部分替代） | 元类（`__new__` / `__init__`） |
| 插件热加载 | OSGi / 自定义 ClassLoader | `importlib.reload()`（有限） |
| 安全沙箱 | `SecurityManager`（已废弃） | 无内建沙箱，依赖进程 / 容器隔离 |

Python 的动态能力比 Java 更直接——不需要反射 API 就能完成属性访问和类型操作。但也因此更需要开发者自行建立边界，避免"什么都能做"变成"什么都看不懂"。

## 结语：让系统可扩展，但不要让系统不可理解

反射让 Python 能够在运行时理解对象。

元编程让 Python 能够定义和控制代码结构。

插件化则进一步把这些能力组织成可扩展的系统架构。

在 AI-Infra 中，它们可以支撑：

- 模型后端注册；
- 推理引擎发现；
- 调度策略扩展；
- 存储和通信组件替换；
- 动态能力协商；
- 统一的生命周期管理；
- 面向第三方团队的扩展接口。

但动态机制的价值并不在于“代码可以自动完成更多事情”，而在于：

> 让稳定的核心流程与变化的具体实现解耦，同时让系统仍然能够被验证、观测和维护。

一个成熟的插件系统通常具备以下特征：

```text
稳定协议
  → 明确注册
  → 受控发现
  → 启动校验
  → 显式生命周期
  → 能力协商
  → 版本兼容
  → 可观测运行
```

当这些边界设计清楚之后，Python 的动态性就不再只是语言层面的灵活，而会转化为 AI-Infra 中真正可复用、可演进的工程能力。