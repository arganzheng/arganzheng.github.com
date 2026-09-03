---
layout: post
title: Python 在 AI-Infra（02）：类型系统与数据契约设计
subtitle: Python Type System and Data Contract Design
tags: [Python]
catalog: true
---



## 导言：类型系统的两层架构与数据契约

Python 是动态类型语言，但这不意味着"无类型"。自 Python 3.5 引入 `typing` 模块以来，类型注解已经从"可选装饰"演变为大型项目的工程标配。PyTorch、vLLM、FastAPI 等 AI Infra 项目大量依赖类型系统的高级特性。

与 Java 把类型声明、编译检查、`.class` 文件携带类型信息、运行时反射合为一体不同，Python 的类型系统由两个协作层构成——**类型信息提供层**和**类型信息消费层**。而在这两层之上，还有一层工程落地：用它们构建**数据契约**。

```
                   类型信息提供层
┌──────────────────────────────────────────────┐
│ 类型表达                                      │
│ ├── Python 内建注解语法                       │
│ ├── 内建泛型：list[str]、dict[str, int]       │
│ ├── typing                                    │
│ └── typing_extensions                         │
│                                               │
│ 类型载体与分发                                 │
│ ├── inline types：.py                         │
│ ├── stub：.pyi                                │
│ ├── typeshed                                  │
│ ├── types-*                                   │
│ └── py.typed / PEP 561                        │
└──────────────────────────────────────────────┘
                       │
                       │ 读取、解释、推理、执行
                       ▼
                   类型信息消费层
┌──────────────────────────────────────────────┐
│ 静态消费                                      │
│ ├── mypy                                      │
│ ├── Pyright / Pylance                         │
│ ├── IDE、CI                                   │
│ └── 类型推断、收窄、兼容性检查                 │
│                                               │
│ 动态消费                                      │
│ ├── isinstance / issubclass                   │
│ ├── get_type_hints / __annotations__          │
│ ├── @dataclass（读注解生成代码）               │
│ ├── Pydantic（元类 + pydantic-core）           │
│ ├── beartype                                  │
│ └── Annotated 元数据解析                      │
└──────────────────────────────────────────────┘
                       │
                       │ 用消费层的能力构建
                       ▼
                 工程落地：数据契约
┌──────────────────────────────────────────────┐
│ 数据建模                                      │
│ ├── @dataclass：内部数据传递                  │
│ ├── Pydantic BaseModel：边界校验              │
│ └── TypedDict：字典形状约束                   │
│                                               │
│ 契约的输入输出                                 │
│ ├── 序列化 / 反序列化                         │
│ ├── JSON Schema / OpenAPI 生成                │
│ └── BaseSettings：配置即契约                  │
└──────────────────────────────────────────────┘
```

- **提供层**解决"类型信息从哪里来"——包括如何用语法表达类型意图（`typing` 模块的各种工具），以及如何把类型信息分发给消费方（`.pyi` 存根、`py.typed` 标记等）。
- **消费层**解决"类型信息被谁使用"——静态分析工具（mypy、pyright）在开发时检查类型正确性，动态工具（Pydantic、beartype）在运行时读取注解并据此生成代码或执行校验。
- **数据契约**解决"用这些能力构建什么"——把类型注解落到具体的数据结构上：请求体、配置项、模型元数据。这是 AI Infra 中类型系统最主要的落地形式。

Java 把前两件事合为一体：类型写在源码里，编译器既是提供者也是消费者，`.class` 文件既是载体也是运行时反射的依据。Python 则把提供和消费拆开，各层可以单独使用，也可以组合使用。

需要说明的是，前两层和第三层的**组织轴并不相同**：前两层沿着"类型信息如何流动"展开，第三章则切换到"程序的数据结构如何设计"。数据建模**使用**类型系统，但它不是类型信息消费的一个子类——所以本文把它单列一章，而不是塞进消费层。

本文将按这个顺序展开，每个特性都会说明：**它解决什么问题、怎么用、Java 中对应什么、在 AI Infra 真实项目中长什么样**。


## 一、类型信息提供层

这一层解决"类型信息从哪里来"。它包括两部分：**类型表达**——用什么语法和工具把类型意图写出来；**类型载体与分发**——如何让没有源码注解的库也能提供类型信息给消费方。

### 1. 类型表达

这一部分覆盖所有"把类型意图表达出来"的语法和工具——从 Python 内建的类型注解语法，到 `typing` 模块提供的高级类型构造，再到 `typing_extensions` 对旧版本的兼容。

> **关于 `typing_extensions`**：Python 类型系统演进很快，每个小版本都有新特性。但很多项目需要支持旧版本（AI Infra 项目线上常见 3.8 或 3.10）。`typing_extensions` 把新版本的特性向后移植，是 PyTorch、Pydantic、vLLM、FastAPI 的必装依赖。本文在介绍各特性时会标注版本要求；如果你的项目需要兼容旧版本，从 `typing_extensions` 导入即可。

#### 1.1 基础注解：变量、函数与容器

##### 变量和函数注解

```python
# 变量注解
name: str = "Alice"
age: int = 18
scores: list[float] = [95.5, 87.0]

# 函数注解
def greet(name: str, *, excited: bool = False) -> str:
    suffix = "!" if excited else "."
    return f"Hello, {name}{suffix}"
```

对应 Java：

```java
String name = "Alice";
int age = 18;
List<Double> scores = List.of(95.5, 87.0);

String greet(String name, boolean excited) {
    String suffix = excited ? "!" : ".";
    return "Hello, " + name + suffix;
}
```

核心区别：Java 的类型声明是语法强制的，编译器检查；Python 的类型注解**默认不影响运行时**，需要 mypy/pyright 做静态检查（见消费层"静态分析与推理"部分）。

##### 内置容器类型注解的演进

Python 的容器类型注解经历了三个阶段：

```python
# 阶段一：Python 3.5-3.8，必须从 typing 导入
from typing import List, Dict, Set, Tuple, FrozenSet
def process(items: List[str]) -> Dict[str, int]:
    ...

# 阶段二：Python 3.9+，内置类型直接支持下标
def process(items: list[str]) -> dict[str, int]:
    ...

# 阶段三：Python 3.12+，泛型类/函数的新语法（后面详述）
def first[T](items: list[T]) -> T:
    return items[0]
```

**推荐**：如果项目的最低 Python 版本 >= 3.9，直接用小写 `list`、`dict`、`set`、`tuple`，不需要从 `typing` 导入。

##### 真实项目中的基础注解

```python
# FastAPI: fastapi/applications.py
class FastAPI(Starlette):
    def add_api_route(
        self,
        path: str,
        endpoint: Callable[..., Any],
        *,
        response_model: type[Any] | None = None,
        status_code: int | None = None,
        tags: list[str | Enum] | None = None,
        summary: str | None = None,
    ) -> None: ...

# PyTorch: torch/_C/_VariableFunctions.pyi
def matmul(input: Tensor, other: Tensor, *, out: Tensor | None = None) -> Tensor: ...
```

完整的内置容器对照表：

| typing 旧写法 | 3.9+ 新写法 | Java 对应 | 说明 |
|---|---|---|---|
| `List[str]` | `list[str]` | `List<String>` | 有序、可变 |
| `Dict[str, int]` | `dict[str, int]` | `Map<String, Integer>` | 键值映射 |
| `Set[int]` | `set[int]` | `Set<Integer>` | 无序、唯一 |
| `Tuple[int, str]` | `tuple[int, str]` | 无直接对应 | 固定长度、异构 |
| `Tuple[int, ...]` | `tuple[int, ...]` | 无直接对应 | 不定长度、同构 |
| `FrozenSet[str]` | `frozenset[str]` | `Set.of(...)` | 不可变集合 |
| `Sequence[int]` | `collections.abc.Sequence[int]` | `List<Integer>` | 只读序列 |
| `Mapping[str, int]` | `collections.abc.Mapping[str, int]` | `Map<String, Integer>` | 只读映射 |


#### 1.2 Union、Optional 与 None：表达"可能性"

##### Union 类型

```python
# 旧写法：Python 3.5+
from typing import Union

def parse_id(raw: Union[str, int]) -> int:
    if isinstance(raw, str):
        return int(raw)
    return raw
```

```python
# 新写法：Python 3.10+
def parse_id(raw: str | int) -> int:
    if isinstance(raw, str):
        return int(raw)
    return raw
```

对应 Java：Java 没有直接的 Union 类型。Java 21 的 sealed interface + pattern matching 可以实现类似效果：

```java
sealed interface RawId permits StringId, IntId {}
record StringId(String value) implements RawId {}
record IntId(int value) implements RawId {}

int parseId(RawId raw) {
    return switch (raw) {
        case StringId s -> Integer.parseInt(s.value());
        case IntId i -> i.value();
    };
}
```

##### Optional：可空类型

```python
# 三种等价写法
from typing import Optional

# 写法一：旧式
def find_user(user_id: int) -> Optional[User]:
    ...

# 写法二：Union 写法
def find_user(user_id: int) -> Union[User, None]:
    ...

# 写法三：3.10+ 推荐写法
def find_user(user_id: int) -> User | None:
    ...
```

`Optional[X]` 就是 `Union[X, None]` 的语法糖，仅此而已。

对应 Java：

```java
// Java: Optional<User> 是一个包装类型
Optional<User> findUser(long userId) {
    ...
}
```

关键区别：Java 的 `Optional` 是一个运行时包装对象，有 `map`、`orElse` 等方法；Python 的 `X | None` 纯粹是类型注解，运行时就是 `X` 的实例或 `None`，没有额外包装。

对于 3.10+ 的项目更推荐 `X | None`，因为它：

- **更简洁**：不需要从 `typing` 导入 `Optional` 或 `Union`；
- **更易读**：`|` 直观地表达了"或者（OR）"的概念。

三者在类型检查时完全等价，`Optional[User]`、`Union[User, None]`、`User | None` 对 mypy/pyright 而言是同一个类型。

> **延伸：`|` 在 Python 中身兼数职**
>
> 这一段与类型系统无关，但有助于理解为什么 Python 选了 `|` 来表示联合类型。
>
> Python 官方设计团队非常喜欢复用 `|`，因为它的直观语义就是"合并 / 或者"。这使得 `|` 在不同上下文中扮演完全不同的角色。
>
> 很早的版本中，`|` 就用作集合的并集（Set Union）：
>
> ```python
> set_a = {1, 2, 3}
> set_b = {3, 4, 5}
>
> # 合并生成新集合
> union_set = set_a | set_b
> print(union_set)  # {1, 2, 3, 4, 5}
>
> # |= 就地更新（求并集并赋给自身）
> set_a |= set_b
> print(set_a)      # set_a 本身已被改变：{1, 2, 3, 4, 5}
> ```
>
> Python 3.9+ 又把它泛化到字典合并与更新。在 3.9 之前合并字典需要 `**` 解包或 `.update()`，比较冗长：
>
> - `|`（合并）：返回新字典。键冲突时右边的值覆盖左边。
> - `|=`（就地更新）：类似 `+=`，直接修改左边的字典。
>
> ```python
> defaults = {"host": "localhost", "port": 8080, "debug": True}
> overrides = {"port": 9000, "debug": False}
>
> merged = defaults | overrides
> print(merged)    # {'host': 'localhost', 'port': 9000, 'debug': False}
>
> defaults |= overrides
> print(defaults)  # defaults 本身已被改变
> ```
>
> 所以 Python 3.10 用 `X | Y` 表示联合类型，是这个"合并"语义的自然延续——只不过合并的对象从值变成了类型。

##### 真实项目中的用法

```python
# vLLM: sampling_params.py
@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int | None = None  # None 表示不限制
    stop: list[str] = field(default_factory=list)
```


#### 1.3 Any、Never 与 NoReturn：类型系统的边界

##### Any：逃逸舱

```python
from typing import Any

def process(data: Any) -> Any:
    # Any 与所有类型兼容，类型检查器不会报错
    return data.whatever()
```

`Any` 类似 Java 的裸类型（raw type）：`List` 而不是 `List<String>`。它告诉类型检查器"不要管这个"。

**使用场景**：和无类型注解的第三方库交互、快速原型阶段。**不要**把它当作"我不知道该写什么类型"的默认选择。

##### 真实项目中的 Any

```python
# PyTorch: torch/nn/modules/module.py
# __setattr__ 用 Any 接收灵活的子模块注册——value 可能是 Parameter、Module、Tensor 或普通属性
class Module:
    def __setattr__(self, name: str, value: Any) -> None: ...

# Pydantic: pydantic/main.py
# model_validate 接受任意数据源——JSON dict、ORM 对象、甚至原始字符串都行
class BaseModel:
    @classmethod
    def model_validate(cls, obj: Any, *, strict: bool | None = None) -> Self: ...

# FastAPI: fastapi/params.py
# Depends 的 dependency 参数接受任意可调用对象
class Depends:
    def __init__(self, dependency: Callable[..., Any] | None = None, *, use_cache: bool = True): ...
```

`Any` 在成熟项目中通常出现在两种位置：

1. **对外接口的入口**——接受用户传入的任意数据（如 Pydantic 的 `model_validate`、FastAPI 的 `Depends`）
2. **动态分发的边界**——对象在运行时才确定具体类型（如 PyTorch 的 `Module.__setattr__`）

核心内部逻辑尽量避免使用 `Any`，用它意味着你主动放弃了类型检查的保护。

##### Never 与 NoReturn

```python
from typing import Never, NoReturn

# NoReturn: 函数永远不会正常返回（抛异常或无限循环）
def fail(message: str) -> NoReturn:
    raise RuntimeError(message)

# Never (3.11+): 不可能存在的类型
# 在大多数场景下 Never 和 NoReturn 可以互换
# Never 更语义化：表示"这个类型不可能被实例化"
def assert_never(value: Never) -> Never:
    raise AssertionError(f"Unexpected value: {value}")
```

对应 Java：Java 没有直接对应的类型；最接近的是 Kotlin 的 `Nothing`。

**实际用途**：`Never` 配合穷尽检查非常有用：

```python
from enum import Enum

class Status(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"

def handle(status: Status) -> str:
    match status:
        case Status.ACTIVE:
            return "ok"
        case Status.INACTIVE:
            return "disabled"
        case _ as unreachable:
            assert_never(unreachable)  # 如果遗漏了某个枚举值，mypy 会报错
```

##### 真实项目中的 NoReturn / Never

```python
# click（命令行框架）: click/exceptions.py
# 所有 Abort/UsageError 最终调用的退出函数
class ClickException(Exception):
    def show(self, file: t.IO[t.Any] | None = None) -> None: ...
    def format_message(self) -> str:
        return self.message

# sys.exit 本身的类型声明（typeshed）
def exit(code: int = ...) -> NoReturn: ...

# 穷尽检查实战：click + Literal 组合
# 来源：https://purarue.xyz/x/blog/click-choice-type-narrowing/
from typing import Literal, assert_never, get_args
import click

OutputFormat = Literal["text", "json"]

@click.command()
@click.option("-o", "--output", type=click.Choice(get_args(OutputFormat)), default="text")
def main(output: OutputFormat) -> None:
    match output:
        case "text":
            print(data)
        case "json":
            print({"data": data})
        case _:
            assert_never(output)  # 新增 Literal 值时，mypy 立刻报错提醒你处理
```

`assert_never` 是 Python 3.11 加入 `typing` 模块的内置函数，底层就是 `def assert_never(arg: Never) -> Never`。

##### AI-Infra 中的穷尽检查

穷尽检查在 AI Infra 代码中极为重要——后端选型、硬件架构、量化方法等枚举分支**必须全部处理**，遗漏一个就可能导致运行时静默失败：

```python
from enum import Enum
from typing import assert_never

# vLLM 风格：推理后端选型
class Backend(Enum):
    CUDA = "cuda"
    ROCM = "rocm"
    CPU = "cpu"
    TPU = "tpu"

def get_attention_impl(backend: Backend) -> type:
    match backend:
        case Backend.CUDA:
            return FlashAttention
        case Backend.ROCM:
            return ROCmFlashAttention
        case Backend.CPU:
            return PagedAttention
        case Backend.TPU:
            return TPUAttention
        case _ as unreachable:
            assert_never(unreachable)
    # 如果后续新增 Backend.XPU 但忘记处理，mypy 立刻报错：
    # error: Argument 1 to "assert_never" has incompatible type "Literal[Backend.XPU]"

# DeepSpeed 风格：硬件架构分支
class DeviceArch(Enum):
    AMPERE = "ampere"       # A100
    HOPPER = "hopper"       # H100
    ADA = "ada"             # L40S/RTX 4090

def select_kernel(arch: DeviceArch, dtype: str) -> str:
    match arch:
        case DeviceArch.AMPERE:
            return "flash_attn_v2"
        case DeviceArch.HOPPER:
            return "flash_attn_v3" if dtype == "fp8" else "flash_attn_v2"
        case DeviceArch.ADA:
            return "flash_attn_v2"
        case _ as unreachable:
            assert_never(unreachable)
```

这种模式的核心价值：**把运行时的"找不到匹配分支"错误，提前到开发期的 mypy 检查阶段暴露**。在 GPU 硬件快速迭代的 AI Infra 领域，新增硬件/后端是家常便饭，穷尽检查能确保每次新增枚举值时，所有相关的分支逻辑都被更新。


#### 1.4 Literal：字面量类型

`Literal` 将类型限制为特定的字面值，类似 Java 中枚举的部分功能，但更轻量。

```python
from typing import Literal

# 只允许这三个字符串值
Mode = Literal["train", "eval", "export"]

def set_mode(mode: Mode) -> None:
    print(f"Setting mode to {mode}")

set_mode("train")    # OK
set_mode("debug")    # mypy 报错：不在允许的值中
```

对应 Java：

```java
// Java: 通常用枚举实现
enum Mode { TRAIN, EVAL, EXPORT }
void setMode(Mode mode) { ... }
```

区别：`Literal` 是纯静态的，运行时不做检查；Java 枚举是运行时的真实类型。

##### 真实项目中的 Literal

```python
# vLLM: vllm/config.py — 量化方法限定为几个固定字符串
QuantMethod = Literal["awq", "gptq", "squeezellm", "marlin"]

# Pydantic: pydantic/fields.py — 字段的 JSON Schema 模式
JsonSchemaMode = Literal["validation", "serialization"]

# httpx: httpx/_types.py — HTTP 方法限定
HttpMethod = Literal["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
```

`Literal` 非常适合替代那些"只接受几个固定字符串"的场景——不值得定义一个完整 Enum 类，但又想让类型检查器帮你约束。


#### 1.5 TypeVar 与泛型

##### TypeVar 基础

`TypeVar` 对应 Java 的类型参数 `<T>`，用于表达"输入和输出之间的类型关系"。

```python
from typing import TypeVar

T = TypeVar("T")

def first(items: list[T]) -> T:
    return items[0]

# 类型检查器知道：
result = first([1, 2, 3])     # result: int
result = first(["a", "b"])    # result: str
```

对应 Java：

```java
<T> T first(List<T> items) {
    return items.get(0);
}
```

##### bound：类型上界

```python
from typing import TypeVar

# T 必须是 SupportsFloat 的子类型（即支持 float() 转换的类型）
T = TypeVar("T", bound="SupportsFloat")

def normalize(value: T) -> float:
    return float(value) / 100.0

# 更常见的实际用法：bound 到自定义基类
from torch.nn import Module
M = TypeVar("M", bound=Module)

def freeze(model: M) -> M:
    """冻结模型参数，返回值类型与传入类型一致"""
    for p in model.parameters():
        p.requires_grad_(False)
    return model
```

对应 Java 的 `<T extends Number>` 或 `<M extends Module>`。

##### 约束到特定类型

```python
# T 只能是 str 或 bytes，不能是其他类型
StrOrBytes = TypeVar("StrOrBytes", str, bytes)

def concat(a: StrOrBytes, b: StrOrBytes) -> StrOrBytes:
    return a + b
```

这比 `Union[str, bytes]` 更严格：它要求 `a` 和 `b` 必须是**同一个类型**。

##### 自定义泛型类

```python
from typing import TypeVar, Generic

T = TypeVar("T")

class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        return self._items.pop()

stack = Stack[int]()
stack.push(1)       # OK
stack.push("a")     # mypy 报错
```

对应 Java：

```java
public class Stack<T> {
    private final List<T> items = new ArrayList<>();

    public void push(T item) { items.add(item); }
    public T pop() { return items.remove(items.size() - 1); }
}
```

##### 协变与逆变

Java 程序员熟悉 `? extends T`（协变）和 `? super T`（逆变）。Python 通过 TypeVar 的参数来表达：

```python
from typing import TypeVar, Generic

T_co = TypeVar("T_co", covariant=True)      # 协变：只读场景
T_contra = TypeVar("T_contra", contravariant=True)  # 逆变：只写场景
```

什么时候需要关心？看一个具体例子：

```python
from typing import TypeVar, Generic, Iterator

T_co = TypeVar("T_co", covariant=True)

class ReadOnlyList(Generic[T_co]):
    """只读容器——协变是安全的：ReadOnlyList[Cat] 可以赋值给 ReadOnlyList[Animal]"""
    def __getitem__(self, index: int) -> T_co: ...
    def __iter__(self) -> Iterator[T_co]: ...

# 类比 Java: List<? extends Animal>
```

| | 协变 (covariant) | 逆变 (contravariant) | 不变 (invariant) |
|---|---|---|---|
| Java | `? extends T` | `? super T` | `T` (默认) |
| Python 旧语法 | `TypeVar(..., covariant=True)` | `TypeVar(..., contravariant=True)` | `TypeVar(...)` (默认) |
| Python 3.12+ | 类型参数后加 `+` | 类型参数后加 `-` | 无标记 |
| 适用场景 | 只读/生产者 | 只写/消费者 | 可读可写 |

在实践中，很少需要手动声明协变/逆变——Protocol 中类型检查器会自动推断。主要在定义泛型容器/接口类时才需要关心。

##### Python 3.12+ 的新语法

Python 3.12 引入了更简洁的泛型语法，不再需要手动创建 `TypeVar`。上面的所有写法都有对应的新形式：

```python
# === 旧写法 ===
from typing import TypeVar, Generic

T = TypeVar("T")

class Stack(Generic[T]):
    def push(self, item: T) -> None: ...
    def pop(self) -> T: ...

def first(items: list[T]) -> T:
    return items[0]

# === 新写法：Python 3.12+ ===
class Stack[T]:
    def push(self, item: T) -> None: ...
    def pop(self) -> T: ...

def first[T](items: list[T]) -> T:
    return items[0]

# 带 bound 约束
def freeze[M: Module](model: M) -> M:
    for p in model.parameters():
        p.requires_grad_(False)
    return model

# 协变/逆变
class ReadOnlyList[+T]:     # + 表示协变（对应旧的 covariant=True）
    def __getitem__(self, index: int) -> T: ...

class WriteOnlyList[-T]:    # - 表示逆变（对应旧的 contravariant=True）
    def append(self, item: T) -> None: ...
```

新语法更接近 Java 和 TypeScript 的泛型声明方式。如果项目目标版本 >= 3.12，推荐使用。不过截至目前，大多数主流项目（PyTorch、vLLM、Pydantic）仍使用旧语法以兼容 3.10/3.11。

##### 真实项目中的 TypeVar 与泛型

```python
# SQLAlchemy: sqlalchemy/orm/session.py
# Session.get() 使用 TypeVar + bound 确保返回值类型安全
_O = TypeVar("_O", bound=object)

class Session:
    def get(self, entity: type[_O], ident: Any) -> _O | None:
        # 返回类型和传入的 entity 类型一致
        ...

# 用法：类型检查器能推断 user 的类型
user = session.get(User, 42)  # user: User | None

# vLLM: vllm/v1/utils.py — 泛型工具类
T = TypeVar("T")

class CpuGpuBuffer(Generic[T]):
    """在 CPU 和 GPU 之间同步的缓冲区"""
    def __init__(self, *size: int, dtype: torch.dtype, device: torch.device): ...
    def get_cpu_value(self) -> T: ...
    def get_gpu_value(self) -> T: ...

# 调用方指定具体类型后，类型检查器自动推断返回值：
buf = CpuGpuBuffer[torch.Tensor](1024, dtype=torch.float32, device="cuda:0")
cpu_val = buf.get_cpu_value()   # cpu_val: torch.Tensor（自动推断）
gpu_val = buf.get_gpu_value()   # gpu_val: torch.Tensor

# typing 模块标准库自身就是 TypeVar 协变/逆变的最大用户
# typing.py
T_co = TypeVar("T_co", covariant=True)

class Iterator(Iterable[T_co]):
    """Iterator 是协变的：Iterator[Cat] 可以赋值给 Iterator[Animal]"""
    @abstractmethod
    def __next__(self) -> T_co: ...
```

#### 1.6 Callable：函数类型

##### 基本用法

```python
from typing import Callable

# 接受两个 int 参数，返回 int 的函数
def apply(fn: Callable[[int, int], int], a: int, b: int) -> int:
    return fn(a, b)

apply(lambda x, y: x + y, 1, 2)  # OK
```

对应 Java：

```java
// Java: 函数式接口
int apply(BiFunction<Integer, Integer, Integer> fn, int a, int b) {
    return fn.apply(a, b);
}
```

##### 更灵活的可调用类型

`Callable[[int, int], int]` 无法表达 keyword-only 参数、默认值等复杂签名。如果需要精确描述，使用 Protocol：

```python
from typing import Protocol

class Comparator(Protocol):
    def __call__(self, a: str, b: str, *, reverse: bool = False) -> int: ...

def sort_with(items: list[str], cmp: Comparator) -> list[str]:
    ...
```

##### 任意参数的 Callable

```python
from typing import Callable

# 接受任意参数的函数
handler: Callable[..., None]  # ... 表示"任意参数"
```

##### 真实项目中的 Callable

```python
# PyTorch: torch/optim/optimizer.py
# 优化器的 step() 接受一个 closure 参数（用于重新计算 loss）
class Optimizer:
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        ...

# vLLM: vllm/entrypoints/llm.py
# use_tqdm 参数既接受 bool，也接受自定义的 tqdm 工厂函数
class LLM:
    def generate(
        self,
        prompts: PromptType | Sequence[PromptType],
        sampling_params: SamplingParams | None = None,
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,  # Callable[..., tqdm]
    ) -> list[RequestOutput]: ...

# FastAPI: fastapi/params.py
# Depends 接受一个 Callable 作为依赖注入的工厂函数
class Depends:
    def __init__(
        self,
        dependency: Callable[..., Any] | None = None,
        *,
        use_cache: bool = True,
    ): ...
```

#### 1.7 `type[C]`：类对象本身

上一节的 `Callable` 描述"可以被调用的东西"。而在 Python 里，**类本身就是一个可以被调用的对象**——调用它会返回实例。这引出一个容易被忽略但极其重要的注解：`type[C]`。

##### 实例 vs 类对象

这是 Python 类型注解里最需要分清的一组区别：

```python
class Model: ...

def run(m: Model) -> None: ...        # 参数是"一个 Model 实例"
def build(c: type[Model]) -> Model:   # 参数是"Model 这个类本身"
    return c()                        # 检查器知道调用它得到 Model 实例

run(Model())      # OK
run(Model)        # 错误：期望实例，给了类
build(Model)      # OK
build(Model())    # 错误：期望类，给了实例
```

`type[C]` 也接受 `C` 的**任意子类**（协变），这正符合直觉：

```python
class LlamaModel(Model): ...

build(LlamaModel)   # OK，type[LlamaModel] 是 type[Model] 的子类型
```

不带参数的裸 `type` 表示"任意类"，等价于 `type[Any]`——和裸 `list` 一样，属于放弃了类型信息的写法，尽量避免。

对应 Java：

```java
// Java: Class<T> 就是 type[C] 的对应物
Model build(Class<? extends Model> cls) throws Exception {
    return cls.getDeclaredConstructor().newInstance();
}
```

| Python | Java | 说明 |
| :--- | :--- | :--- |
| `Model` | `Model` | 实例类型 |
| `type[Model]` | `Class<? extends Model>` | 类对象，含子类 |
| `type` / `type[Any]` | `Class<?>` | 任意类 |
| `c()` | `cls.getDeclaredConstructor().newInstance()` | Python 里类天生可调用 |
| `type(x)` | `x.getClass()` | 取运行时类型 |
| `issubclass(a, b)` | `b.isAssignableFrom(a)` | 子类判定 |
| `isinstance(x, C)` | `C.isInstance(x)` / `instanceof` | 实例判定 |

两个实质差异：

1. **构造是一等操作。** Java 拿到 `Class<T>` 后要经过反射才能 `newInstance()`，还要处理一堆受检异常；Python 里 `c()` 就是普通调用，类型检查器甚至会**校验构造参数**——传错参数是静态错误，不是运行时才炸的 `NoSuchMethodException`。
2. **没有类型擦除。** Java 的 `Class<T>` 之所以常被用作"运行时类型令牌"（比如 `Gson.fromJson(json, Foo.class)`），正是因为泛型被擦除了，运行时拿不到 `T`。Python 没有擦除问题——注解本身在运行时就可以读取（见第二章）——所以 `type[C]` 的用途更纯粹：它就是"我需要一个类，而不是一个实例"。

##### 典型用途：工厂与注册表

`type[C]` 最常见的场景是把类当作值来传递和存储：

```python
from typing import TypeVar

T = TypeVar("T", bound=Model)

# 泛型工厂：输入什么类，就返回什么类的实例
def create(cls: type[T], **kwargs) -> T:
    return cls(**kwargs)

model = create(LlamaModel, hidden_size=4096)   # 推断为 LlamaModel，不是 Model
```

这里 `type[T]` 和 `TypeVar` 的配合是关键。如果写成 `def create(cls: type[Model]) -> Model`，返回值就退化成基类，调用方拿不到子类特有的方法。**这与 Java 里 `<T> T create(Class<T> cls)` 而不是 `Model create(Class<?> cls)` 是完全相同的动机。**

注册表则是插件化架构的基础形态：

```python
_REGISTRY: dict[str, type[Model]] = {}

def register(name: str) -> Callable[[type[T]], type[T]]:
    def decorator(cls: type[T]) -> type[T]:
        _REGISTRY[name] = cls
        return cls                    # 装饰器必须原样返回类
    return decorator

@register("llama")
class LlamaModel(Model): ...

def load(name: str) -> Model:
    return _REGISTRY[name]()
```

注意装饰器的签名 `Callable[[type[T]], type[T]]`——**接收类、返回类**。写成 `Callable[[type], type]` 会丢失具体类型，被装饰的类在下游就变成了 `type[Any]`。

##### `type[C]` vs `Callable[..., C]`

两者都能表达"能造出 C 的东西"，但语义不同：

```python
def a(factory: type[Model]) -> Model: ...      # 必须是类
def b(factory: Callable[..., Model]) -> Model: # 类或函数都行
```

- 需要访问**类属性、classmethod 或做 `issubclass` 判断**时，必须用 `type[C]`；
- 只关心"能调用出实例"，允许传入 `functools.partial`、lambda 或工厂函数时，用 `Callable[..., C]` 更宽松。

一个常见坑：**抽象类不满足 `type[C]` 的可实例化预期**。mypy 会对下面这段报 `Only concrete class can be given where "type[AbstractModel]" is expected`：

```python
from abc import ABC, abstractmethod

class AbstractModel(ABC):
    @abstractmethod
    def forward(self) -> None: ...

def build(cls: type[AbstractModel]) -> AbstractModel:
    return cls()

build(AbstractModel)     # mypy 报错：抽象类不能实例化
```

这其实是件好事——静态检查器帮你拦住了 `TypeError: Can't instantiate abstract class`。如果确实只想传递类而不实例化（比如存进注册表），把返回类型改成 `type[AbstractModel]` 即可。

##### classmethod 与 `Self`

classmethod 的第一个参数 `cls` 隐式就是 `type[Self]`，所以通常不需要显式标注：

```python
from typing import Self

class Config:
    @classmethod
    def from_dict(cls, data: dict) -> Self:    # cls 隐式是 type[Self]
        return cls(**data)


class TrainConfig(Config): ...

cfg = TrainConfig.from_dict({})   # 推断为 TrainConfig，不是 Config
```

用 `Self` 而不是 `Config` 作返回类型，子类才能拿到正确的推断结果（`Self` 见 1.14 节）。这解决的正是 Java 里"自限定泛型" `class Config<T extends Config<T>>` 那套笨重写法要解决的问题。

##### 与元类的关系

`type` 除了作为注解，它本身还是 **Python 中所有类的类**——这就是元类的起点：

```python
class Model: ...

type(Model())      # <class 'Model'>       实例的类型是类
type(Model)        # <class 'type'>        类的类型是 type
type(type)         # <class 'type'>        type 是自己的实例，递归终点
```

所以自定义元类都写成 `class Meta(type)`：元类就是"类的类"，继承 `type` 才能拦截类的创建过程。Pydantic 的 `ModelMetaclass` 正是这么来的（见第二章 2.2 节）。

两个用法之间的桥梁是：**如果一个类的元类是 `Meta`，那么这个类对象的类型就是 `Meta`**，可以直接用来注解：

```python
class Meta(type): ...
class Base(metaclass=Meta): ...

def configure(cls: Meta) -> None:     # 只接受元类为 Meta 的类
    ...
```

> **注意区分两个 `type`**：内置的 `type`（本节讨论的类对象），和 Python 3.12 引入的软关键字 `type`（用于声明类型别名，如 `type Vector = list[float]`，见 1.14 节）。两者拼写相同但毫无关系，靠语法位置区分。

##### 版本说明

`typing.Type[C]` 从 Python 3.9 起被内置的 `type[C]` 取代（PEP 585），新代码一律用小写。这和 `List` → `list`、`Dict` → `dict` 是同一次演进。

##### 真实项目中的 `type[C]`

```python
# transformers: transformers/models/auto/configuration_auto.py
# AutoConfig 的核心是一张 "模型类型字符串 -> 配置类" 的注册表，
# from_pretrained 读到 config.json 里的 model_type 后据此查表并实例化
CONFIG_MAPPING_NAMES: OrderedDict[str, str] = ...   # 惰性加载，值是类名
# 实际查表后得到的就是 type[PretrainedConfig]

# vLLM: vllm/model_executor/models/registry.py
# 架构名（来自 HF config 的 architectures 字段）-> 模型实现类
class _ModelRegistry:
    models: dict[str, _BaseRegisteredModel]

    def register_model(self, model_arch: str, model_cls: type[nn.Module]) -> None:
        ...

    def resolve_model_cls(self, architectures: str | list[str]) -> tuple[type[nn.Module], str]:
        ...

# Starlette: starlette/applications.py
# 异常处理器注册：键可以是异常类本身，也可以是 HTTP 状态码
def add_exception_handler(
    self,
    exc_class_or_status_code: type[Exception] | int,
    handler: Callable[[Request, Exception], Response],
) -> None: ...

# Pydantic: pydantic/main.py
# model_validate 是 classmethod，返回 Self 而非 BaseModel，
# 这样 MyModel.model_validate(...) 才能推断成 MyModel
class BaseModel:
    @classmethod
    def model_validate(cls, obj: Any, *, strict: bool | None = None) -> Self: ...
```

这几个例子体现了同一个模式：**框架把"用户提供的类"当作数据存起来，在运行时按需实例化**。这正是 Python 插件化架构的骨架——注册表的值类型永远是 `type[SomeBase]`，而不是 `SomeBase`。Java 里对应的是 Spring 的 `BeanDefinition` 持有 `Class<?>`、或 SPI 的 `ServiceLoader<S>`。

#### 1.8 ABC 与 Protocol：接口的两种方式

Python 定义"接口"有两种机制：**ABC（抽象基类）**和 **Protocol（协议）**。它们的定位不同，适用场景不同，理解两者的区别是读懂 AI Infra 源码的关键。

##### ABC：抽象基类（名义类型）

ABC（Abstract Base Class）来自标准库的 `abc` 模块，对应 Java 的 `abstract class` + `interface`。

```python
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def speak(self) -> str:
        """子类必须实现此方法"""
        ...

    def breathe(self) -> str:
        """可以提供默认实现"""
        return "breathing..."

class Dog(Animal):
    def speak(self) -> str:
        return "Woof!"

# 如果忘记实现抽象方法，实例化时立刻报 TypeError
class BadAnimal(Animal):
    pass

BadAnimal()  # TypeError: Can't instantiate abstract class BadAnimal
             # with abstract method speak
```

##### 真实项目中的 ABC

ABC 在主流框架中大量使用，特别是作为**框架基类**：

```python
# PyTorch: torch/nn/modules/module.py
# nn.Module 继承了 ABC——这是 PyTorch 整个模型体系的根基
class Module:
    # 虽然 Module 没有直接写 (ABC)，但它的 forward 方法
    # 通过 raise NotImplementedError 达到了类似抽象方法的效果
    def forward(self, *input: Any) -> Any:
        raise NotImplementedError(
            f"Module [{type(self).__name__}] is missing the required 'forward' function"
        )

# 用户必须继承并实现 forward
class MyModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

# collections.abc: Python 标准库中最重要的 ABC 集合
# 这些 ABC 定义了 Python 核心数据结构的"接口契约"
from collections.abc import Iterable, Iterator, Sequence, Mapping, MutableMapping

# 任何实现了 __iter__ 的类都可以注册为 Iterable
# 任何实现了 __getitem__ + __len__ 的类都可以注册为 Sequence

# SQLAlchemy: sqlalchemy/engine/interfaces.py
# 数据库方言的抽象接口
class Dialect(ABC):
    @abstractmethod
    def connect(self, *cargs: Any, **cparams: Any) -> DBAPIConnection: ...

    @abstractmethod
    def create_connect_args(self, url: URL) -> ConnectArgsType: ...
```

##### collections.abc 速查

| ABC | 需要实现的方法 | Java 对应 | 说明 |
|---|---|---|---|
| `Iterable` | `__iter__` | `Iterable<T>` | 可迭代 |
| `Iterator` | `__iter__`, `__next__` | `Iterator<T>` | 迭代器 |
| `Sequence` | `__getitem__`, `__len__` | `List<T>`（只读） | 有序序列 |
| `MutableSequence` | + `__setitem__`, `__delitem__`, `insert` | `List<T>` | 可变序列 |
| `Mapping` | `__getitem__`, `__len__`, `__iter__` | `Map<K,V>`（只读） | 映射 |
| `MutableMapping` | + `__setitem__`, `__delitem__` | `Map<K,V>` | 可变映射 |
| `Set` | `__contains__`, `__iter__`, `__len__` | `Set<T>` | 集合 |
| `Callable` | `__call__` | `Function<T,R>` | 可调用 |
| `Hashable` | `__hash__` | 重写 `hashCode()` | 可哈希 |
| `Sized` | `__len__` | 无直接对应 | 有长度 |

在类型注解中，当你希望参数是"只读"的时候，用 `Sequence` 而不是 `list`，用 `Mapping` 而不是 `dict`——这和 Java 中用 `List<T>` 接口而不是 `ArrayList<T>` 作为参数类型是同一个道理。

##### Protocol：结构化子类型（静态鸭子类型）

ABC 要求显式继承，但 Python 是鸭子类型语言——"如果它走路像鸭子、叫声像鸭子，那它就是鸭子"。`Protocol`（Python 3.8+）将这种理念形式化为类型系统的一部分：**不需要显式继承，只要方法签名匹配就算实现了该协议**。

```python
from typing import Protocol

class Closeable(Protocol):
    def close(self) -> None: ...

def cleanup(resource: Closeable) -> None:
    resource.close()

# 任何有 close() 方法的对象都可以传入，不需要继承 Closeable
class DatabaseConnection:
    def close(self) -> None:
        print("disconnected")

class FileHandle:
    def close(self) -> None:
        print("file closed")

cleanup(DatabaseConnection())  # OK —— 没有继承 Closeable，但有 close() 方法就行
cleanup(FileHandle())          # OK
```

对应 Java：

```java
// Java: 必须显式 implements
interface Closeable {
    void close();
}

class DatabaseConnection implements Closeable {  // 必须写 implements
    public void close() { ... }
}
```

##### ABC vs Protocol：选择指南

| 特性 | Java interface | Python ABC | Python Protocol |
|---|---|---|---|
| 显式继承 | 必须 `implements` | 必须继承 | **不需要** |
| 默认实现 | default method | 普通方法 | 不支持 |
| 运行时检查 | `instanceof` | `isinstance` | 需要 `@runtime_checkable` |
| 检查时机 | 编译期 | 实例化时 | 静态分析时 |
| 核心理念 | 名义类型 | 名义类型 | **结构化类型** |

**何时选 ABC：**

- 你在写**框架基类**，需要强制子类实现某些方法（如 PyTorch `nn.Module`）
- 需要在实例化时立刻报错（而不是等到调用时）
- 需要提供**默认实现**（抽象方法 + 普通方法混合）
- 需要 `isinstance` 运行时检查

**何时选 Protocol：**

- 你不控制第三方类的代码（无法让它继承你的基类）
- 只想约束"这个对象需要有某些方法"，不关心它的继承关系
- 跨库/跨团队的接口约定
- 更灵活，更 Pythonic

简单记忆：**框架作者用 ABC，框架使用者用 Protocol**。

##### 带 `@runtime_checkable` 的 Protocol

默认情况下 Protocol 只在静态检查时有效。加上 `@runtime_checkable` 后可以用 `isinstance` 做运行时检查：

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Sized(Protocol):
    def __len__(self) -> int: ...

print(isinstance([1, 2, 3], Sized))  # True
print(isinstance(42, Sized))          # False
```

注意：`@runtime_checkable` 只检查方法**是否存在**，不检查签名是否匹配。

##### 泛型 Protocol

```python
from typing import Protocol, TypeVar

T_co = TypeVar("T_co", covariant=True)

class Reader(Protocol[T_co]):
    def read(self) -> T_co: ...

def process(reader: Reader[str]) -> str:
    return reader.read().upper()
```

##### 真实项目中的 Protocol

```python
# PyTorch 风格：任何实现了 forward 和 __call__ 的对象
class ForwardModule(Protocol):
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...

# vLLM 风格：可替换的 executor 接口
class ExecutorBase(Protocol):
    def initialize(self, num_gpu_blocks: int) -> None: ...
    def execute_model(self, seq_group_metadata: list) -> list: ...
```

#### 1.9 TypedDict：字典的类型约束

Python 中大量使用 `dict` 传递数据。`TypedDict` 让你能对字典的"形状"（哪些 key、每个 key 的值类型）进行静态约束。

##### 基本用法

```python
from typing import TypedDict

class MovieRecord(TypedDict):
    title: str
    year: int
    rating: float

movie: MovieRecord = {
    "title": "Inception",
    "year": 2010,
    "rating": 8.8,
}

# mypy 会报错：
bad: MovieRecord = {"title": "X"}  # 缺少 year 和 rating
```

对应 Java：Java 通常用 DTO / record 代替，很少直接使用 `Map<String, Object>`。

##### 可选字段

```python
from typing import TypedDict, Required, NotRequired

# 方式一：total=False 让所有字段都可选
class Config(TypedDict, total=False):
    host: str
    port: int
    debug: bool

# 方式二：精确控制（3.11+）
class Config(TypedDict):
    host: Required[str]
    port: NotRequired[int]
    debug: NotRequired[bool]
```

##### TypedDict 与其他数据定义方式的关系

Python 有三种主流的"结构化数据"定义方式：`TypedDict`（约束字典形状，运行时仍是普通 `dict`）、`dataclass`（标准库数据类，类似 Java Record）、Pydantic `BaseModel`（带运行时校验，类似 Java Bean Validation）。

`TypedDict` 与后两者的根本区别在于：**它不创建新的对象类型**。`MovieRecord` 在运行时就是一个普通 `dict`，类型约束只对静态检查器生效。所以如果数据本身已经是 dict（JSON API 返回值、配置文件解析结果），用 `TypedDict` 约束形状最自然；如果需要创建新的结构化对象，则用 `dataclass` 或 Pydantic。

> 三者的完整对比、选型决策树以及"边界校验、内部传递"的工程模式，见第三章「工程落地：数据契约设计」的选型指南一节。

##### 用 TypedDict 约束 `**kwargs`（3.12+）

```python
from typing import Unpack, TypedDict

class Options(TypedDict, total=False):
    timeout: float
    retries: int
    verbose: bool

def request(url: str, **kwargs: Unpack[Options]) -> str:
    ...

# 类型检查器知道 kwargs 只能包含 timeout、retries、verbose
request("https://api.example.com", timeout=5.0)        # OK
request("https://api.example.com", unknown_key=True)    # mypy 报错
```

##### 真实项目中的 TypedDict

```python
# PyTorch: torch/optim/optimizer.py
# 优化器状态用 TypedDict 描述每个参数组的结构
class _RequiredParameter(TypedDict):
    params: list[Tensor]

class _AdamState(TypedDict):
    step: int
    exp_avg: Tensor
    exp_avg_sq: Tensor

# Pydantic-AI: examples/pydantic_ai_examples/chat_app.py
# 聊天消息用 TypedDict 描述传给前端的 JSON 形状
from typing_extensions import TypedDict
class ChatMessage(TypedDict):
    role: str
    timestamp: str
    content: str

# SQLAlchemy: 查询选项
class ExecuteOptions(TypedDict, total=False):
    stream_results: bool
    max_row_buffer: int
    yield_per: int
```

#### 1.10 Annotated：给类型附加元数据

`Annotated` 允许在类型上附加额外的元数据，类型检查器本身忽略这些元数据，但框架（如 FastAPI、Pydantic）可以读取并使用。

```python
from typing import Annotated

# 基本语法：Annotated[类型, 元数据1, 元数据2, ...]
UserId = Annotated[int, "must be positive"]
```

##### Pydantic 中的 Annotated

```python
from typing import Annotated
from pydantic import BaseModel, Field

class User(BaseModel):
    name: Annotated[str, Field(min_length=2, max_length=50)]
    age: Annotated[int, Field(ge=0, le=150)]
```

##### FastAPI 中的 Annotated

```python
from typing import Annotated
from fastapi import Depends, Header, Query

# 把依赖注入、校验规则等信息附加到类型上
async def list_items(
    q: Annotated[str | None, Query(max_length=50)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
    token: Annotated[str, Header()],
    db: Annotated[Session, Depends(get_db)],
) -> list[Item]:
    ...
```

对应 Java：最接近的概念是**注解（Annotation）**。Java 的 `@NotNull`、`@Size(max=50)` 作用在参数或字段上；Python 的 `Annotated` 把元数据嵌入到类型本身。

```java
// Java 的方式
void createUser(@NotNull @Size(min=2, max=50) String name,
                @Min(0) @Max(150) int age) { ... }
```


#### 1.11 ParamSpec 与 Concatenate：保留装饰器的类型信息

##### 问题：装饰器吃掉了类型信息

```python
from functools import wraps

def logged(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        print(f"Calling {fn.__name__}")
        return fn(*args, **kwargs)
    return wrapper

@logged
def add(a: int, b: int) -> int:
    return a + b

# 问题：类型检查器认为 add 的签名变成了 (*args, **kwargs) -> Any
# 原始的 (a: int, b: int) -> int 信息丢失了
```

##### ParamSpec 解决方案

```python
from typing import ParamSpec, TypeVar, Callable
from functools import wraps

P = ParamSpec("P")
R = TypeVar("R")

def logged(fn: Callable[P, R]) -> Callable[P, R]:
    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        print(f"Calling {fn.__name__}")
        return fn(*args, **kwargs)
    return wrapper

@logged
def add(a: int, b: int) -> int:
    return a + b

# 现在类型检查器知道 add 仍然是 (a: int, b: int) -> int
```

`ParamSpec` 捕获了被装饰函数的**完整参数签名**并透传出来。

##### Concatenate：装饰器添加参数

如果装饰器需要在原始函数前面添加参数：

```python
from typing import Callable, Concatenate, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

def with_request(
    fn: Callable[Concatenate[Request, P], R]
) -> Callable[P, R]:
    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        request = get_current_request()
        return fn(request, *args, **kwargs)
    return wrapper

@with_request
def handle(request: Request, user_id: int) -> Response:
    ...

# 装饰后：handle(user_id: int) -> Response
# request 参数被装饰器自动注入了
```

Java 没有对应概念——Java 的注解处理器不会改变方法签名。

##### 真实项目中的 ParamSpec

```python
# Tenacity（重试库）: tenacity/__init__.py
# retry 装饰器用 ParamSpec 保留原始函数签名
from typing import ParamSpec, TypeVar, Callable
P = ParamSpec("P")
R = TypeVar("R")

class Retrying:
    def __call__(self, fn: Callable[P, R]) -> Callable[P, R]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            ...  # 重试逻辑
            return fn(*args, **kwargs)
        return wrapper

# Celery（任务队列）: celery/app/task.py 风格
# 异步任务装饰器保留函数签名
def task(fn: Callable[P, R]) -> Task[P, R]:
    ...

# Pydantic-AI: 用 ParamSpec 在 pydantic_ai 的 agent 装饰器中保留工具函数签名
from typing_extensions import ParamSpec
P = ParamSpec("P")
```

`ParamSpec` 是装饰器密集型项目的"救星"——Python 生态有大量装饰器（retry、cache、trace、auth），没有 `ParamSpec` 之前类型信息全部丢失。

##### AI-Infra 中的 ParamSpec 与 Concatenate

在 AI Infra 中，装饰器模式无处不在：训练循环的 hook、性能分析、分布式通信包装、自动混合精度等等。`ParamSpec` 和 `Concatenate` 让这些装饰器不再是类型信息的黑洞。

```python
from typing import ParamSpec, TypeVar, Callable, Concatenate
from functools import wraps

P = ParamSpec("P")
R = TypeVar("R")

# 场景一：PyTorch 风格的性能分析装饰器
# 包装任意函数，记录 CUDA 事件耗时，但不改变函数签名
def cuda_timer(fn: Callable[P, R]) -> Callable[P, R]:
    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        print(f"{fn.__name__}: {start.elapsed_time(end):.2f}ms")
        return result
    return wrapper

@cuda_timer
def forward_pass(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return model(x)

# 类型检查器知道 forward_pass 仍然是 (nn.Module, torch.Tensor) -> torch.Tensor

# 场景二：Concatenate——自动注入分布式 rank 参数
def with_rank(
    fn: Callable[Concatenate[int, P], R]
) -> Callable[P, R]:
    """自动在第一个参数注入当前进程的 rank"""
    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        rank = torch.distributed.get_rank()
        return fn(rank, *args, **kwargs)
    return wrapper

@with_rank
def log_metrics(rank: int, loss: float, step: int) -> None:
    if rank == 0:  # 只在主进程打印
        print(f"Step {step}: loss={loss:.4f}")

# 装饰后签名变为：log_metrics(loss: float, step: int) -> None
# rank 被自动注入，调用方不需要传
log_metrics(loss=0.5, step=100)
```

这在分布式训练框架中特别有用——很多函数需要 `rank`/`world_size`/`device` 等上下文参数，通过 `Concatenate` 装饰器自动注入后，调用方的代码更干净，类型检查也不会丢失。

#### 1.12 TypeGuard 与 TypeIs：类型收窄

##### TypeGuard（3.10+）

类型收窄函数：返回 `True` 时，告诉类型检查器参数是特定类型。

```python
from typing import TypeGuard

def is_str_list(val: list[object]) -> TypeGuard[list[str]]:
    return all(isinstance(x, str) for x in val)

def process(data: list[object]) -> None:
    if is_str_list(data):
        # 这里 mypy 知道 data 是 list[str]
        print(data[0].upper())
```

对应 Java 的 `instanceof` pattern matching：

```java
if (data instanceof List<?> list && isStringList(list)) {
    // 但 Java 的类型擦除让这种检查比较有限
}
```

##### TypeIs（3.12+）

`TypeIs` 是 `TypeGuard` 的改进版，行为更直观：

```python
from typing import TypeIs

def is_string(val: object) -> TypeIs[str]:
    return isinstance(val, str)

def process(val: str | int) -> None:
    if is_string(val):
        print(val.upper())   # val: str
    else:
        print(val + 1)       # val: int（TypeIs 能正确收窄 else 分支）
```

`TypeGuard` 和 `TypeIs` 的区别：`TypeIs` 在 `else` 分支也会收窄类型，`TypeGuard` 不会。如果你的项目目标版本 >= 3.12，优先用 `TypeIs`。

##### 真实项目中的 TypeGuard

```python
# Pydantic: pydantic/_internal/_utils.py
# 判断一个值是否是 Pydantic 模型实例
from typing import TypeGuard

def is_model_instance(value: Any) -> TypeGuard[BaseModel]:
    return isinstance(value, BaseModel)

# typeshed（Python 官方类型存根库）: builtins.pyi
# callable() 内置函数本身的类型声明就用了 TypeGuard
def callable(obj: object) -> TypeGuard[Callable[..., object]]: ...

# pandas-stubs: 判断 DataFrame 的列类型
def is_numeric_dtype(arr_or_dtype: Any) -> TypeGuard[np.number]: ...
```

`TypeGuard` 在大型项目中使用相对低频——因为大多数场景 `isinstance` 已经能自动收窄。它主要出现在需要自定义复杂检查逻辑的地方（如容器内元素类型检查），以及类型存根（`.pyi`）文件中。

##### AI-Infra 中的 TypeGuard / TypeIs

在 AI Infra 代码中，TypeGuard/TypeIs 最典型的场景是**根据模型/张量的运行时属性做类型分支**——这些属性无法通过简单的 `isinstance` 判断：

```python
from typing import TypeGuard, TypeIs

# 场景一：判断张量是否在 CUDA 上
# isinstance 无法区分 CPU Tensor 和 CUDA Tensor（它们是同一个类）
# 但我们可以用 TypeGuard 让类型检查器理解分支逻辑

class CUDATensor(torch.Tensor):
    """标记类型：表示已经在 GPU 上的张量"""
    device: torch.device  # device.type == "cuda"

def is_cuda_tensor(t: torch.Tensor) -> TypeGuard[CUDATensor]:
    return t.is_cuda

def process(t: torch.Tensor) -> torch.Tensor:
    if is_cuda_tensor(t):
        # 类型检查器知道这里 t 是 CUDATensor
        return torch.ops.custom_cuda_kernel(t)
    else:
        return t.to("cuda")

# 场景二：判断模型是否已量化
class QuantizedModel:
    """标记类型：已经过量化处理的模型"""
    quantization_config: dict

def is_quantized(model: nn.Module) -> TypeIs[QuantizedModel]:
    return hasattr(model, "quantization_config") and model.quantization_config is not None

def optimize(model: nn.Module) -> nn.Module:
    if is_quantized(model):
        # TypeIs: 这里 model 被收窄为 QuantizedModel
        print(f"Already quantized: {model.quantization_config}")
        return model
    else:
        # TypeIs: else 分支也能收窄（TypeGuard 做不到）
        return quantize(model)
```

这种模式在 vLLM 的模型加载器、PyTorch 的 quantization 模块中都有类似逻辑——虽然不一定用了 `TypeGuard` 注解（很多是运行时 `if` 检查），但理解 TypeGuard 的思路有助于写出更清晰的分支代码。

#### 1.13 overload：多签名声明

`@overload` 不是运行时重载（Python 没有函数重载），而是给类型检查器提供多个调用签名的描述。

```python
from typing import overload

@overload
def fetch(url: str, as_json: Literal[True]) -> dict: ...
@overload
def fetch(url: str, as_json: Literal[False]) -> str: ...
@overload
def fetch(url: str) -> str: ...

# 实际实现（运行时只有这一个）
def fetch(url: str, as_json: bool = False) -> dict | str:
    response = requests.get(url)
    if as_json:
        return response.json()
    return response.text
```

对应 Java：Java 的方法重载是编译器真正支持的多个方法；Python 的 `@overload` 只是类型检查层面的声明，运行时只有最后一个实现生效。

##### 真实项目中的 overload

```python
# PyTorch: torch/_C/_VariableFunctions.pyi (类型存根文件)
# zeros 支持两种调用方式
@overload
def zeros(size: Sequence[int], *, dtype: torch.dtype = ...) -> Tensor: ...
@overload
def zeros(*size: int, dtype: torch.dtype = ...) -> Tensor: ...

# vLLM: vllm/entrypoints/llm.py
# generate() 同时支持新旧两种 API 风格
class LLM:
    @overload
    def generate(
        self,
        prompts: PromptType | Sequence[PromptType],
        /,
        sampling_params: SamplingParams | None = None,
    ) -> list[RequestOutput]: ...

    @overload  # LEGACY: 旧式参数
    @deprecated("'prompt_token_ids' will become part of 'prompts'")
    def generate(
        self,
        prompts: str,
        sampling_params: SamplingParams | None = None,
        prompt_token_ids: list[int] | None = None,
    ) -> list[RequestOutput]: ...

# httpx: httpx/_client.py
# Client.request() 根据 stream 参数返回不同类型
class Client:
    @overload
    def request(self, method: str, url: URL, *, stream: Literal[True]) -> Response: ...
    @overload
    def request(self, method: str, url: URL, *, stream: Literal[False] = ...) -> Response: ...
```

`@overload` 在需要向后兼容旧 API 的项目中尤其常见（如 vLLM 的 generate 方法同时支持新旧调用方式）。

#### 1.14 其他实用工具

##### Final 和 ClassVar

```python
from typing import Final, ClassVar

class Config:
    MAX_RETRIES: Final = 3                # 常量，不允许重新赋值
    instances: ClassVar[int] = 0          # 类变量，不是实例变量
    name: str = "default"                 # 普通实例变量
```

- `Final` 对应 Java 的 `final`
- `ClassVar` 对应 Java 的 `static` 字段（在 dataclass 中特别有用，防止被当作构造参数）

```python
# vLLM: vllm/entrypoints/llm.py — ClassVar 在真实项目中的用法
class LLM:
    DEPRECATE_LEGACY: ClassVar[bool] = False  # 类级别开关，不是实例属性

# Pydantic: BaseModel 的 model_config 就是 ClassVar
class User(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(strict=True)
    name: str  # 这是实例字段

# PyTorch: torch/nn/modules/module.py — Final 标记不可重写的方法
class Module:
    dump_patches: bool = False
    _version: int = 1

    # training 是 Final，子类不应该重写
    training: bool
```

##### Self（3.11+）

```python
from typing import Self

class Builder:
    def set_name(self, name: str) -> Self:
        self.name = name
        return self  # 返回 Self 让子类继承后链式调用仍然正确

    def set_value(self, value: int) -> Self:
        self.value = value
        return self
```

对应 Java 中 Builder 模式返回 `this` 的场景。在 `Self` 出现之前，Python 中实现这个需要复杂的 TypeVar bound。

```python
# httpx: httpx/_client.py — 上下文管理器返回 Self
class Client:
    def __enter__(self) -> Self:
        return self
    def __exit__(self, *args: Any) -> None:
        self.close()

class AsyncClient(Client):
    async def __aenter__(self) -> Self:  # 子类仍然返回正确的类型
        return self

# Pydantic: pydantic/main.py — model_validate 返回 Self
class BaseModel:
    @classmethod
    def model_validate(cls, obj: Any, *, strict: bool | None = None) -> Self: ...

# 子类继承后类型仍然正确：
class UserModel(BaseModel):
    name: str
user = UserModel.model_validate(data)  # user: UserModel（不是 BaseModel）

# SQLAlchemy: Query 的链式调用
class Query(Generic[_T]):
    def filter(self, *criterion: Any) -> Self: ...
    def order_by(self, *clauses: Any) -> Self: ...
    def limit(self, limit: int) -> Self: ...
```

`Self` 在返回 `self` 的链式调用和 `@classmethod` 工厂方法中特别有价值——PEP 673 统计发现它在 typeshed 中的使用频率是 `Callable` 的 40%，非常常见。

##### TypeAlias（3.10+）与 `type` 语句（3.12+）

```python
# 3.10+: 显式声明类型别名
from typing import TypeAlias

Vector: TypeAlias = list[float]
Matrix: TypeAlias = list[Vector]

# 3.12+: 新语法
type Vector = list[float]
type Matrix = list[Vector]

# 支持延迟求值——可以引用尚未定义的类型
type Tree[T] = T | list[Tree[T]]  # 递归类型别名
```

对应 Java 的 `typedef`——哦等等，Java 没有 typedef。这是 Python 类型系统比 Java 灵活的一个方面。

```python
# PyTorch: torch/types.py — 大量使用 TypeAlias 简化复杂类型
from typing import TypeAlias
Device: TypeAlias = str | torch.device | int
Number: TypeAlias = int | float

# vLLM: vllm/inputs/data.py — 输入类型的别名
PromptType: TypeAlias = str | TextPrompt | TokensPrompt

# Pydantic: pydantic/fields.py
JsonValue: TypeAlias = int | float | str | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
```

TypeAlias 在大型项目中极为常见——它让复杂的联合类型和嵌套泛型变得可读。

##### cast：类型断言

```python
from typing import cast

# 告诉类型检查器："相信我，这个值就是这个类型"
raw = get_value()                        # 返回 object
value = cast(int, raw)                   # 类型检查器认为 value: int
```

对应 Java 的强制类型转换 `(int) raw`。关键区别：Python 的 `cast` **运行时什么都不做**，只是给类型检查器的提示。

```python
# vLLM: vllm/entrypoints/llm.py — 用 cast 在类型检查器无法推断时提供帮助
from typing import cast
outputs = cast(list[RequestOutput], req_outputs)

# PyTorch: torch/jit/_script.py — 从动态注册表中取回已知类型
fn = cast(ScriptFunction, _get_function(qualified_name))

# SQLAlchemy: sqlalchemy/engine/result.py — 窄化 row 类型
row = cast(tuple[str, int], result.fetchone())
```

`cast` 的使用频率在成熟项目中相当高。它的典型场景：1) 从 `dict`/`list` 中取值后类型检查器无法推断；2) 经过动态注册/反射后丢失了类型信息。

##### TYPE_CHECKING：避免循环导入

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # 这个 import 只在类型检查时执行，运行时不执行
    from heavy_module import HeavyClass

class Light:
    def process(self, obj: "HeavyClass") -> None:
        ...
```

这是解决循环导入的标准做法：把只用于类型注解的 import 放在 `if TYPE_CHECKING:` 块中。

##### 真实项目中的 TYPE_CHECKING

```python
# vLLM: vllm/entrypoints/openai/generate/api_router.py
# 经典用法：避免在运行时导入重型引擎模块
from typing import TYPE_CHECKING

from fastapi import FastAPI

if TYPE_CHECKING:
    from argparse import Namespace
    from vllm.engine.protocol import EngineClient
    from vllm.entrypoints.logger import RequestLogger
    from vllm.tasks import SupportedTask
else:
    RequestLogger = object  # 运行时用 object 占位

async def init_generate_state(
    engine_client: "EngineClient",       # 引号内的前向引用
    args: "Namespace",
    request_logger: RequestLogger | None,
    supported_tasks: tuple["SupportedTask", ...],
): ...

# SQLAlchemy: sqlalchemy/orm/relationships.py
# ORM 关系定义中解决 Model 之间的循环引用
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .mapper import Mapper

class RelationshipProperty:
    mapper: "Mapper"  # 只在类型检查时需要 Mapper 的导入
```

`TYPE_CHECKING` 在 vLLM 源码中出现超过 200 次，在 PyTorch 中出现超过 500 次。它是大型 Python 项目管理模块依赖的标准手段。


### 2. 类型载体与分发

类型注解要对库的使用者生效，不仅需要写在源码中，还需要以类型检查器能够发现和读取的形式随库分发。

对于普通 Python 模块，类型信息通常直接写在 `.py` 文件中。对于 C/C++ 扩展、动态生成的 API，或者缺少源码注解的第三方库，类型信息则可以通过 `.pyi` 存根文件、独立的类型存根包、`typeshed`、`py.typed`/PEP 561，以及类型检查器插件等机制提供。

以类似 PyTorch `torch._C` 的底层扩展模块为例，类型检查器通常无法直接分析 `.so`、`.pyd` 等二进制文件内部的 C/C++ 实现。因此，库作者需要通过其他类型载体描述其对外暴露的 Python API，例如：

```python
# _native.pyi

def matmul(a: Tensor, b: Tensor) -> Tensor: ...
```

这里的 `.pyi` 文件不包含具体实现，只向类型检查器声明模块的接口、参数类型和返回值类型。类型检查器在分析下游代码时，可以读取这些声明，而不需要理解底层 C/C++ 实现。

因此，类型载体与分发机制解决的核心问题不是“如何给没有 Python 源码的库添加注解”，而是：

> **如何将库的公开 API 以类型检查器可发现、可读取和可传播的形式提供给下游代码。**

常见机制的分工如下：

```text
.py       类型信息与 Python 实现写在同一个文件中
.pyi      单独描述模块的接口和类型信息
typeshed  为标准库及部分第三方库提供外部存根
types-*   以独立软件包形式分发第三方库的类型存根
py.typed  声明包内的类型信息可以提供给下游类型检查器
PEP 561   规定 Python 包分发类型信息的相关机制
```

需要注意的是，C/C++ 扩展并不一定没有类型信息；它们通常只是无法通过二进制实现本身被类型检查器直接推导。类型信息仍然可以由 `.pyi` 文件、Python 包装层、外部存根包或类型检查器插件提供。下面我们就展开介绍这种外部存根包的类型信息提供机制。

#### 2.1 .pyi 存根文件与 typeshed

##### .pyi 文件

`.pyi`（Python Interface）文件只包含签名，不包含实现。它告诉类型检查器一个模块里有什么函数、什么类型：

```python
# torch/_C/__init__.pyi — PyTorch 的 C++ 扩展模块的类型存根

def _get_tracing_state() -> bool: ...
def _set_grad_enabled(enabled: bool) -> None: ...

class TensorBase:
    def dim(self) -> int: ...
    def size(self, dim: int | None = None) -> Size: ...
    def to(self, device: Device, dtype: dtype | None = None) -> Tensor: ...
```

`.pyi` 文件的语法和普通 Python 完全一样，只是函数体都用 `...`（Ellipsis）代替。

##### typeshed

[typeshed](https://github.com/python/typeshed) 是 Python 官方维护的类型存根仓库，覆盖：

- **Python 标准库**的所有模块（`os`、`sys`、`json`、`asyncio` 等）
- **部分知名第三方库**（`requests`、`docutils` 等）

mypy 和 pyright 都**内置了 typeshed**，所以你用标准库时不需要手动安装任何存根。

##### types-* 独立存根包

对于 typeshed 不覆盖的第三方库，社区通过 PyPI 发布独立的存根包，命名规则为 `types-<package>`：

```bash
pip install types-requests     # requests 的类型存根
pip install types-PyYAML       # PyYAML 的类型存根
pip install types-redis        # redis 的类型存根
pip install types-Pillow       # Pillow 的类型存根
```

mypy 安装后会自动发现并使用这些存根。

##### 存根文件的查找优先级

类型检查器按以下顺序查找类型信息（以 mypy 为例）：

1. **包内 inline 类型**：库自带的类型注解（源码中直接写的）
2. **包内存根**：库自带的 `.pyi` 文件（如 PyTorch 的 `torch/_C/__init__.pyi`）
3. **独立存根包**：通过 pip 安装的 `types-*` 包
4. **typeshed**：内置的标准库和知名库存根
5. **自定义存根**：项目内 `mypy.ini` 或 `pyproject.toml` 指定的路径

对应 Java：Java 没有存根文件的概念——类型信息编译进 `.class` 文件。最接近的是 `.jar` 中不含实现的接口定义。


#### 2.2 py.typed、PEP 561 与类型信息发布

##### PEP 561：类型信息的发布标准

[PEP 561](https://peps.python.org/pep-0561/) 定义了 Python 库如何声明"我提供了类型信息"。这个标准让类型检查器知道哪些库是"类型安全"的。

核心机制非常简单——在包的根目录放一个空文件 `py.typed`：

```
mypackage/
├── __init__.py
├── py.typed          ← 标记文件，可以是空文件
├── core.py           ← 源码中直接写类型注解
└── _internal.pyi     ← 或者提供 .pyi 存根
```

有了 `py.typed`，类型检查器就知道这个包的类型信息是**官方提供**的（不是第三方猜的），可以放心使用。

##### 三种类型信息发布方式

| 方式 | 说明 | 例子 |
|---|---|---|
| **Inline types** | 源码中直接写注解 + `py.typed` | FastAPI, Pydantic, httpx |
| **包内存根** | `.pyi` 文件随包发布 + `py.typed` | PyTorch (`torch/_C/*.pyi`) |
| **独立存根包** | 单独的 `types-*` 包 | `types-requests`, `types-PyYAML` |

主流 AI Infra 项目的选择：

- **FastAPI / Pydantic / httpx**：inline types——源码本身就有完整注解，加 `py.typed` 标记
- **PyTorch**：混合——Python 代码用 inline types，C++ 扩展用 `.pyi` 存根
- **NumPy**：从 1.20 开始提供 inline types + `py.typed`
- **requests**：本身无注解，依赖社区的 `types-requests` 独立存根包


#### 2.3 inline types vs stub：如何选择

如果你在**开发一个库**，需要决定如何提供类型信息：

| 考量 | Inline types（推荐） | .pyi 存根 |
|---|---|---|
| 维护成本 | 低——类型和代码一起改 | 高——改代码后要同步改存根 |
| 适用场景 | 纯 Python 库 | C 扩展、需要对外隐藏实现 |
| 对调用方的体验 | 跳转到源码能看到完整实现 | 跳转到 `.pyi` 只能看签名 |
| 运行时开销 | 极小（注解是惰性求值的） | 零 |

**推荐**：如果你的库是纯 Python，直接在源码中写类型注解 + 加 `py.typed` 标记。只有 C 扩展模块才需要 `.pyi` 存根。

```bash
# 发布一个带类型信息的包
mypackage/
├── __init__.py
├── py.typed              # 空文件即可
├── core.py               # def process(data: list[str]) -> dict[str, int]: ...
└── _c_extension.pyi      # C 扩展的存根
```

在 `pyproject.toml` 中确保 `py.typed` 被打包：

```toml
[tool.setuptools.package-data]
mypackage = ["py.typed", "*.pyi"]
```

这一步很容易漏——`py.typed` 在源码目录里存在，但如果没配 `package-data`，构建 wheel 时不会被打进去，下游依然看不到类型信息。

> `pyproject.toml` 的完整配置、wheel 与 sdist 的区别、带 C/CUDA 扩展的包如何构建与发布，见[《Python 项目工程化与生产交付》](/python-engineering-and-production-delivery.html)的"打包与分发"一章。



## 二、类型信息消费层

类型信息写好了、分发好了，接下来就是"谁来用"。消费方分为两种：**静态分析工具**在开发时检查类型正确性，**动态工具**在运行时利用类型信息做校验和解析。两者互补，不是替代关系。

### 1. 静态分析与推理

类型注解写在源码中，但 Python 解释器**完全忽略**它们——不会做任何检查。真正让类型注解产生价值的是**静态类型检查器**。这一部分关于"谁来检查、怎么检查、检查到什么程度"。

#### 1.1 mypy 与 pyright

##### mypy

mypy 是 Python 官方的类型检查器，由 Guido van Rossum 本人发起，也是历史最久、社区最广的选择。

```bash
pip install mypy

mypy src/                 # 基本检查
mypy --strict src/        # 严格模式（推荐新项目）
```

##### pyright

pyright 由 Microsoft 开发，用 TypeScript 编写，是 VS Code 插件 Pylance 的后端。速度是它的最大优势。

```bash
pip install pyright

pyright src/
```

##### 对比

| 特性 | mypy | pyright |
|---|---|---|
| 语言 | Python | TypeScript (Node.js) |
| 速度 | 较慢（大项目可达分钟级） | **快很多**（通常秒级） |
| IDE 集成 | 需要插件 | VS Code Pylance **内置** |
| 严格程度 | 可配置，默认较宽松 | 默认更严格 |
| 维护方 | Python 官方 + 社区 | Microsoft |
| 增量检查 | 支持（`--incremental`，默认开启） | 支持（文件级缓存） |
| CI 常见度 | **更常见**（老牌标准） | 在增长 |

两者都广泛使用。如果用 VS Code 开发，pyright 通过 Pylance 自动工作；CI 中 mypy 更常见。很多项目**同时**配置两者——本地开发用 pyright 获得即时反馈，CI 用 mypy 做门禁。

##### 类型推断

和 Java 的 `var` 一样，类型检查器也能自动推断类型，不需要处处手写注解：

```python
x = 42              # mypy/pyright 推断 x: int
items = [1, 2, 3]   # 推断 items: list[int]
d = {"a": 1}        # 推断 d: dict[str, int]

def double(n: int) -> int:
    return n * 2

result = double(5)  # 推断 result: int
```

但在以下场景推断会失败，需要显式注解：

```python
# 空容器——无法推断元素类型
items: list[str] = []

# 复杂的工厂/注册表模式
registry: dict[str, Callable[..., Module]] = {}

# 函数参数——必须注解（mypy --strict 要求）
def process(data):    # mypy --strict: error: Function is missing a type annotation
    ...
```


#### 1.2 配置实践与渐进式引入

##### pyproject.toml 配置

```toml
# mypy 配置
[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true

# 对没有类型存根的第三方库禁用检查
[[tool.mypy.overrides]]
module = "torch.*"
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = "transformers.*"
ignore_missing_imports = true
```

```toml
# pyright 配置
[tool.pyright]
pythonVersion = "3.11"
typeCheckingMode = "standard"  # off / basic / standard / strict
reportMissingImports = true
reportMissingTypeStubs = false
```

##### 渐进式引入策略

对于已有大型项目，不可能一步到位开启 `--strict`。推荐的渐进策略：

**第一步：仅检查新代码**

```toml
[tool.mypy]
# 不开 strict，只检查有注解的代码
check_untyped_defs = true
```

**第二步：对关键模块开启严格检查**

```toml
[[tool.mypy.overrides]]
module = "myproject.api.*"
strict = true

[[tool.mypy.overrides]]
module = "myproject.models.*"
strict = true
```

**第三步：CI 中逐步收紧**

```bash
# 只检查本次 PR 修改的文件
git diff --name-only origin/main | grep '\.py$' | xargs mypy
```

**第四步：全面严格模式**

```toml
[tool.mypy]
strict = true
```

vLLM、FastAPI 等项目都是逐步引入类型检查的——早期代码有大量 `Any` 和 `# type: ignore`，新代码则要求严格注解。


#### 1.3 静态检查的能力边界

类型检查器不是万能的。理解它做不到什么，才能在"加注解"和"写 `# type: ignore`"之间做正确选择。

##### 做不到的事情

**1. 运行时动态行为**

```python
# 动态属性——类型检查器无法追踪
class Config:
    pass

config = Config()
config.debug = True     # mypy: error: "Config" has no attribute "debug"
                        # 但运行时完全合法

# __getattr__ 让对象可以响应任意属性
class Flexible:
    def __getattr__(self, name: str) -> Any:
        return 42

f = Flexible()
f.anything   # 运行时返回 42，但 mypy 无法推断
```

**2. 复杂的元编程**

```python
# dataclass 的 __init__ 由装饰器在运行时生成
# mypy 有专门的插件支持 dataclass，但自定义元类可能无法推断

# SQLAlchemy 的声明式 ORM：Column 定义到属性类型的映射需要 mypy 插件
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)  # mypy 不知道 self.id 是 int
    name = Column(String)                   # 需要 sqlalchemy-stubs 或 mypy 插件
```

**3. 跨进程/跨语言边界**

```python
# C 扩展模块——需要 .pyi 存根文件提供类型信息（见"类型载体与分发"部分）
import numpy as np
arr = np.array([1, 2, 3])  # 没有存根就是 Any

# RPC / 序列化结果——类型信息在传输中丢失
result = pickle.loads(data)  # result: Any
```

##### 什么时候用 `# type: ignore`

合理的使用场景：

```python
# 1. 你确认代码正确，但类型检查器的能力有限
value = getattr(obj, attr_name)  # type: ignore[attr-defined]

# 2. 和没有类型存根的 C 扩展交互
import some_c_extension  # type: ignore[import-untyped]

# 3. 临时绕过，附上 TODO
result = complex_dynamic_call()  # type: ignore[no-any-return]  # TODO: add proper types
```

不合理的使用：**用 `# type: ignore` 压制所有错误来让 CI 通过**——这意味着类型注解形同虚设。

##### 不同检查器对同一代码的判断可能不同

```python
from typing import TypeVar

T = TypeVar("T")

def identity(x: T) -> T:
    return x

reveal_type(identity(42))
# mypy:    Revealed type is "builtins.int"
# pyright: Type of identity(42) is "int"
# 结果一样

# 但在某些边界情况下，两者的推断会有差异
# 特别是涉及 overload 解析、TypeVar 的多重约束时
```

如果项目同时使用 mypy 和 pyright，偶尔需要同时满足两者的要求。遇到冲突时，优先修正代码而不是加 `# type: ignore`。


### 2. 动态消费：运行时如何读取类型注解

类型注解在运行时**默认被忽略**——Python 解释器不会因为 `x: int = "hello"` 而报错。但注解本身是保留在对象上的，任何代码都可以在运行时把它读出来加以利用：有的用来生成代码（`@dataclass`），有的用来生成校验器（Pydantic），有的用来做即时检查（beartype）。

这一节关注的是**机制**：运行时工具通过什么途径拿到注解、在什么时机拿、拿到之后做了什么。至于用这些工具**怎么设计数据结构**（建模、校验、序列化、配置），是第三章「工程落地：数据契约设计」的主题。

#### 2.1 isinstance、`__annotations__` 与 get_type_hints()：原生能力

##### isinstance：最基本的运行时类型检查

```python
def process(value: int | str) -> str:
    if isinstance(value, int):
        return str(value * 2)
    elif isinstance(value, str):
        return value.upper()
```

`isinstance` 是 Python 内置的运行时类型检查，不依赖 `typing` 模块。它的限制：

- **不支持泛型**：`isinstance(x, list[int])` 会报错（`list[int]` 不是运行时类型）
- **不支持 Union**：`isinstance(x, int | str)` 从 3.10 开始才支持
- **不支持 Protocol**：除非加了 `@runtime_checkable`

##### `__annotations__`：注解存放在哪里

在 Python 中，当你在类内部写下 `id: int` 但不赋值时，解释器并不会把它当作普通类变量，而是把这个映射关系存入类的 `__annotations__` 字典：

```python
class RawUser:
    id: int
    name: str

print(RawUser.__annotations__)
# {'id': <class 'int'>, 'name': <class 'str'>}
```

普通的类对这个字典视而不见——它就静静躺在那里，不影响任何运行时行为。但 `@dataclass` 装饰器和 Pydantic 的元类正是通过读取它，拿到了模型所需的字段名和目标类型。**这是所有"运行时消费类型注解"的框架的共同起点。**

函数也一样：

```python
def greet(name: str, age: int = 18) -> str:
    return f"{name} is {age}"

print(greet.__annotations__)
# {'name': <class 'str'>, 'age': <class 'int'>, 'return': <class 'str'>}
```

##### get_type_hints()：更可靠的注解读取

```python
from typing import get_type_hints

class User:
    name: str
    age: int
    email: str | None = None

hints = get_type_hints(User)
# {'name': <class 'str'>, 'age': <class 'int'>, 'email': str | None}
```

`get_type_hints()` 返回一个类或函数的类型注解字典。它比直接读 `__annotations__` 更可靠，差别有三点：

| | `__annotations__` | `get_type_hints()` |
|---|---|---|
| 字符串注解 | 原样返回字符串 | 求值为真正的类型对象 |
| 继承来的字段 | 只有当前类自己的 | 合并整条 MRO 上的注解 |
| `Optional` 补全 | 不处理 | 带 `None` 默认值的参数自动补成 `X \| None` |

第一点尤其重要。开启 `from __future__ import annotations` 后（或使用前向引用），所有注解都会以字符串形式保存：

```python
from __future__ import annotations

class Node:
    value: int
    next: Node | None       # 此时 Node 还没定义完

print(Node.__annotations__)
# {'value': 'int', 'next': 'Node | None'}   ← 是字符串，不是类型

from typing import get_type_hints
print(get_type_hints(Node))
# {'value': <class 'int'>, 'next': Node | None}   ← 已求值
```

所以框架读注解时基本都用 `get_type_hints()` 而不是裸的 `__annotations__`——这是 Pydantic、beartype 等框架的底层基础：它们在运行时读取注解，然后根据注解生成校验逻辑。

对应 Java：`get_type_hints()` 类似 Java 的反射 API `Field.getGenericType()`，但由于 Java 有类型擦除，运行时拿不到完整的泛型信息。Python 反而更好——注解信息在运行时完整保留。

```python
# 框架底层原理示意
def validate(cls, data: dict) -> object:
    hints = get_type_hints(cls)
    instance = cls.__new__(cls)
    for field_name, field_type in hints.items():
        value = data[field_name]
        if not isinstance(value, field_type):   # 简化版——实际框架处理更复杂
            raise TypeError(f"{field_name} expects {field_type}, got {type(value)}")
        setattr(instance, field_name, value)
    return instance
```


#### 2.2 框架如何消费注解：代码生成与元类

上一节的 `validate()` 是一个玩具示例。真实框架读到注解之后做什么？主要有两条技术路线：**装饰器 + 代码生成**（`@dataclass`）和**元类 + 验证树**（Pydantic）。理解这两条路线，就理解了 Python 数据类框架的全部"魔法"。

##### `@dataclass`：读注解，生成代码，但不校验

标准库的 `@dataclass` 是一个装饰器。当它包裹一个类时：

1. 读取该类的 `__annotations__` 字典，得到字段名和字段顺序；
2. 在内存中拼接一段形如 `def __init__(self, id, name): self.id = id ...` 的**源码字符串**；
3. 用内置的 `exec()` 把这段字符串编译成真正的函数对象，绑定到类上。

本质是在类定义完成之后，由外挂的装饰器往类里塞方法。可以直接把生成的结果打印出来：

```python
from dataclasses import dataclass
import inspect

@dataclass
class User:
    id: int
    name: str

print(inspect.signature(User.__init__))
# (self, id: int, name: str) -> None      ← 这个 __init__ 是 exec 生成的
```

这里有一个**必须说清楚的点**：`@dataclass` 只用注解做了"字段声明"这一件事，它**完全不做类型校验**。注解在这里是触发器，不是检查依据：

```python
user = User(id="not an int", name=123)   # 不报错！
print(user.id)                            # 'not an int'
```

`dataclasses` 甚至不关心注解的内容是不是一个合法类型——写 `id: "随便什么字符串"` 它也照样生成 `__init__`。所以在类型信息的消费谱系里，`@dataclass` 属于**最轻度**的消费者：它只关心"有哪些字段"，不关心"字段是什么类型"。

##### Pydantic：元类拦截 + 构建验证树

Pydantic 走的是更底层的**元类**机制。当模型继承 `BaseModel` 时，Python 在**类创建阶段**（不是实例化阶段，更不是调用阶段）就会触发 Pydantic 的自定义元类。

Pydantic v2 中元类的核心流程：

1. **类创建期的拦截与组装**：元类拦截类的创建过程，遍历 `__annotations__` 的每个字段，检查是否附带 `EmailStr`、`Field()`、`Annotated[...]` 等高级定义；
2. **构建验证树**：为该模型构建一套验证树结构。v2 为了性能，把这部分核心校验逻辑编译后交给 Rust 编写的引擎 `pydantic-core` 驱动；
3. **重写 `__init__`**：生成一个特殊的 `__init__`。调用 `User(id="123")` 时它不直接赋值，而是把入参丢进验证树——先清洗与转换（`"123"` → `123`），再执行复杂校验（正则、范围），通过后写入实例；失败则收集**所有**错误路径，一次性抛出结构化的 `ValidationError`。

关键在于**时机**：验证树是在类定义时一次性构建好的，实例化时只是执行它。这也是 Pydantic v2 比 v1 快一个数量级的原因之一——把工作从"每次实例化"挪到了"仅一次的类创建"。

```python
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str

# 类定义完成的那一刻，验证器就已经生成好了
print(type(User))               # <class 'pydantic._internal._model_construction.ModelMetaclass'>
print(User.__pydantic_core_schema__ is not None)   # True
```

上面 `type(User)` 打印出 `ModelMetaclass` 而不是 `type`，正是 1.7 节"类的类型是 `type`，自定义元类则是 `type` 的子类"那条规则的直接体现。换句话说，`User` 这个**类对象**的类型是 `ModelMetaclass`，所以任何接受 `type[BaseModel]` 的函数都能拿到它。

> 元类本身的机制（`type` 的三参数形式、`__new__` 的拦截时机、与 `__init_subclass__` 的取舍）在[《Python 动态机制及 AI-Infra 实践》](/python-reflection-metaprogramming-and-plugin-architecture.html)的"元类：控制类的创建过程"一节有完整展开，这里只关注它作为注解消费者的角色。

##### 两条路线的对比

| | `@dataclass` | Pydantic `BaseModel` |
|---|---|---|
| 介入方式 | 装饰器，类创建**之后** | 元类，类创建**过程中** |
| 读注解的手段 | `__annotations__` | `get_type_hints()`（处理前向引用） |
| 拿注解干什么 | 只取字段名和顺序，生成 `__init__` | 解析类型语义，构建验证树 |
| 注解内容是否被理解 | **否**，只当占位符 | **是**，驱动转换与校验 |
| 运行时校验 | 无 | 有（Rust 实现的 `pydantic-core`） |
| 实现技术 | `exec()` 动态代码生成 | 元类 + Rust 扩展 |

对应 Java：`@dataclass` 类似 Lombok——编译期往类里塞方法，注解只是生成指令；Pydantic 类似 Hibernate Validator——真正解析注解的语义并在运行时执行校验。差别是 Lombok 在编译期改 AST，`@dataclass` 在运行时 `exec` 字符串。

> **接下来**：这一节讲的是"框架怎么读注解"。至于**用**这些框架怎么设计数据结构——什么时候该用 `dataclass`、什么时候该上 Pydantic、如何做序列化和配置管理——见第三章「工程落地：数据契约设计」。


#### 2.3 beartype 与其他运行时检查工具

##### beartype：零配置的运行时类型检查

如果你不需要 Pydantic 的完整数据建模能力，只想在运行时检查函数参数类型，[beartype](https://github.com/beartype/beartype) 是一个轻量级选择：

```python
from beartype import beartype

@beartype
def add(a: int, b: int) -> int:
    return a + b

add(1, 2)       # OK
add(1, "2")     # beartype.roar.BeartypeCallHintParamViolation:
                # @beartyped add() parameter b="2" violates hint <class 'int'>
```

beartype 的特点：

| 特性 | beartype | Pydantic | 纯 isinstance |
|---|---|---|---|
| 使用方式 | `@beartype` 装饰器 | 继承 `BaseModel` | 手动写 `if isinstance` |
| 校验时机 | 函数调用时 | 对象实例化时 | 你调用时 |
| 支持泛型 | 支持（`list[int]`） | 支持 | 不支持 |
| 性能开销 | 极低（O(1) 抽样检查） | 中等（完整校验） | 最低 |
| 数据转换 | 不做 | 自动转换 | 不做 |
| 适用场景 | 防御式编程、调试期 | API 边界、数据建模 | 简单分支判断 |

##### typeguard：另一个运行时检查库

```python
from typeguard import typechecked

@typechecked
def process(data: list[str]) -> dict[str, int]:
    return {s: len(s) for s in data}
```

[typeguard](https://github.com/agronholm/typeguard) 功能类似 beartype，但做**完整**检查（不是抽样），性能开销更大。适合测试环境。

##### 何时需要运行时类型检查

**需要的场景**：
- API 入参校验（用 Pydantic）
- 外部数据解析（JSON、配置文件、用户输入）
- 调试期防御（用 beartype，发版时可关闭）

**不需要的场景**：
- 内部函数调用——静态检查已经足够
- 性能关键路径——任何运行时检查都有开销
- 类型检查器已经保证正确的代码


#### 2.4 静态与运行时的协作边界

Python 类型系统的一个核心设计原则是：**静态检查和运行时检查是互补的，不是替代关系**。

##### 分工

```
                    静态检查（mypy/pyright）          运行时检查（Pydantic/beartype）
                    ─────────────────────           ─────────────────────────────
覆盖范围           你写的代码 + 有存根的库            所有实际运行的数据
检查时机           开发时 / CI                       运行时
性能开销           零（不影响运行）                   有（校验成本）
能做到             推断、收窄、穷尽检查               精确值校验（范围、格式、正则）
做不到             校验外部输入的具体值               类型推断、代码可读性提升
```

##### 推荐实践

**1. 信任边界（Trust Boundary）模式**：在系统的**入口处**做运行时校验，内部用静态检查。

```python
# 入口：运行时校验——外部数据不可信
@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):  # Pydantic 校验
    # 经过 Pydantic 校验后，内部可以信任类型正确
    result = engine.generate(request.prompt, request.params)
    return result

# 内部：静态检查——数据已经可信
def generate(prompt: str, params: SamplingParams) -> GenerateOutput:
    tokens = self.tokenizer.encode(prompt)  # mypy 确保 prompt 是 str
    ...
```

**2. 保持一致**：静态注解和运行时校验的类型要匹配。

```python
# 好：Pydantic 模型的字段注解 == mypy 看到的类型
class Config(BaseModel):
    batch_size: int = 32

# 坏：运行时校验和类型注解不一致
class Config(BaseModel):
    batch_size: Any = 32  # 运行时不校验，mypy 也不检查——两头都放弃了
```

**3. 不要在热路径上做运行时检查**

```python
# 坏：每次 forward 都做运行时类型检查
@beartype
def forward(self, x: torch.Tensor) -> torch.Tensor:  # GPU 推理瓶颈
    ...

# 好：只在初始化时检查
@beartype
def __init__(self, config: ModelConfig) -> None:  # 只调用一次
    ...
```


## 三、工程落地：数据契约设计

前两章沿着"类型信息如何流动"展开：怎么表达、怎么分发、谁来消费。这一章**组织轴切换**——不再讨论类型信息本身，而是讨论用这些能力去构建什么：**数据契约**。

所谓数据契约，就是对"一组数据长什么样"的正式约定。在 AI-Infra 系统里，它无处不在：

- **请求体**：`/v1/chat/completions` 收到的 JSON 应该有哪些字段、什么类型、什么取值范围；
- **配置项**：从 `.env`、YAML 读进来的一堆字符串，怎么变成强类型的配置对象；
- **模型元数据**：模型名、量化方式、并行度、KV cache 配置在进程间传递时的形状；
- **内部数据结构**：一次推理请求在引擎内部流转时携带的上下文。

在 Java 里这件事由多个技术栈拼起来：Lombok / `record` 消除模板代码，Bean Validation 做校验，Jackson 做序列化，`@ConfigurationProperties` 做配置绑定。Python 则高度收敛——`@dataclass` 和 Pydantic 两个工具覆盖了绝大部分场景。看一眼 Pydantic 模型的定义：

```python
from pydantic import BaseModel, Field
from typing import Annotated, Literal

class InferenceConfig(BaseModel):
    model_name: str
    backend: Literal["cuda", "rocm", "cpu"]
    max_tokens: Annotated[int, Field(ge=1, le=32768)] = 2048
    temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 1.0
    top_p: float = 1.0
```

这一个类定义同时做了三件事：

1. **静态类型检查**：mypy/pyright 能检查代码中对 `InferenceConfig` 字段的使用；
2. **运行时校验**：Pydantic 确保传入的数据满足约束（`ge=1`、`le=32768`）；
3. **文档 / Schema 生成**：FastAPI 自动从这个模型生成 OpenAPI 文档。

这就是"数据契约"的价值——一处声明，三处受益。而它能成立，靠的正是第二章讲的机制。

> **前置阅读**：如果你想先搞清楚 `@dataclass` 和 Pydantic **怎么**读到类型注解、`exec` 代码生成和元类分别在什么时机介入，见第二章「类型信息消费层」的"框架如何消费注解"一节。本章只讲用法与取舍。

### 1. dataclass：标准库的数据类

#### 1.1 从原生 `__init__` 到 `@dataclass`

最传统的写法是显式定义构造函数：

```python
class User:
    def __init__(self, id: int, name: str, email: str):
        self.id = id
        self.name = name
        self.email = email
```

问题很明显：**大量 `self.x = x` 的模板代码**。字段一多就难以维护，而且还要手写 `__repr__`、`__eq__` 才能方便调试和比较。

Python 3.7 引入的数据类消除了这些样板。它利用类变量类型标注（PEP 526）语法：

```python
from dataclasses import dataclass

@dataclass
class User:
    id: int
    name: str
    email: str
```

实例化方式不变，但 `__init__`、`__repr__`、`__eq__` 都自动有了：

```python
user = User(id=1, name="Alice", email="alice@example.com")
print(user)          # User(id=1, name='Alice', email='alice@example.com')
print(user == User(1, "Alice", "alice@example.com"))   # True
```

对应 Java：

```java
// Java 14 之前：Lombok
import lombok.Data;

@Data
public class User {
    private Long id;
    private String name;
    private String email;
}

// Java 14+：官方 record，定位与 @dataclass 极其相似
public record User(Long id, String name, String email) {}
```

#### 1.2 `__post_init__` 手工校验及其局限

`@dataclass` **不做运行时校验**——这一点在第二章已经说明原因：它只把注解当字段清单，不理解注解的语义。所以下面这行不会报错：

```python
user = User(id="abc", name=123, email=None)   # 静默通过
```

如果需要校验，得借助 `__post_init__` 钩子（实例化后自动触发）：

```python
from dataclasses import dataclass
import re

@dataclass
class User:
    id: int
    name: str
    email: str

    def __post_init__(self):
        # 1. 手动校验类型
        if not isinstance(self.id, int):
            raise TypeError("id 必须是 int 类型")

        # 2. 手动用正则校验邮箱
        if not re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", self.email):
            raise ValueError("邮箱格式不正确")
```

这条路能走通，但代价很快显现：

- **逐字段手写**：每个字段的类型、范围、格式都要自己 `if`；
- **不做类型转换**：外部传进来的 `"123"` 不会变成 `123`，只能自己转；
- **错误不聚合**：第一个 `raise` 就中断了，用户看不到全部问题；
- **与注解脱节**：注解写 `int`，校验逻辑另写一遍，两边可能不一致。

Java 生态在这一点上遇到了完全相同的问题——`record` 和 Lombok 都只解决"数据容器"，不具备校验能力，所以才需要引入 Hibernate Validator。Python 的答案则是 Pydantic（下一节）。

#### 1.3 `frozen` 与 `slots`

两个常用参数：

```python
from dataclasses import dataclass

# frozen=True：不可变，实例创建后不能改字段，且自动获得 __hash__
@dataclass(frozen=True)
class ModelKey:
    model_name: str
    revision: str

key = ModelKey("llama-3", "main")
# key.revision = "dev"     # 报错：FrozenInstanceError
cache: dict[ModelKey, object] = {key: ...}   # 可以做 dict 的 key

# slots=True（3.10+）：用槽位存储属性，省去 __dict__
@dataclass(slots=True)
class RequestContext:
    request_id: str
    model_name: str
    deadline: float
```

`frozen=True` 对应 Java 的 `record`（天然不可变）；`slots=True` 没有 Java 对应物，它解决的是 Python 特有的每实例 `__dict__` 开销问题。

> `__slots__` 的内存收益取决于对象数量和字段类型，详见[《Python 内存管理与优化》](/python-memory-management-and-optimization.html)。

### 2. Pydantic：带校验的数据模型

Pydantic 是 Python 生态中最流行的运行时数据校验框架，也是 FastAPI 的核心基石。它把数据建模、类型转换和深度校验融合在一起——相当于 Java 的 `record` + Bean Validation + Jackson 三者合一。

#### 2.1 BaseModel 基础与类型强制转换

```python
from pydantic import BaseModel, EmailStr

class User(BaseModel):
    id: int
    name: str
    email: EmailStr
```

与 `@dataclass` 最大的差别是它**真的会校验，并且会转换**：

```python
# 自动校验 + 类型转换（coercion）
user = User(id="1", name="Alice", email="alice@example.com")
print(user.id, type(user.id))    # 1 <class 'int'>   ← 字符串 "1" 被转成了 int

# 校验失败时抛出详细错误
User(id="abc", name="Bob", email="not-an-email")
# pydantic_core.ValidationError: 2 validation errors for User
# id
#   Input should be a valid integer, unable to parse string as an integer
# email
#   value is not a valid email address
```

注意两点：一是 `EmailStr` 这类语义类型开箱即用；二是**两个错误一次性全部报出来**，而不是遇到第一个就中断——这正是第二章提到的"收集所有错误路径"。

#### 2.2 Field：默认值与约束

`Field()` 用来表达注解本身表达不了的约束（范围、长度、正则）：

```python
from pydantic import BaseModel, EmailStr, Field

class User(BaseModel):
    id: int
    email: EmailStr

    # 1. 基础默认值：不传时默认为 "user"
    role: str = "user"

    # 2. 完全可选字段：允许为 None，不传时默认就是 None
    bio: str | None = None

    # 3. 业务边界约束
    name: str = Field(default="Anonymous", min_length=2, max_length=20)
    age: int = Field(default=18, ge=0, le=120)
```

约束也可以写在 `Annotated` 里，这是 Pydantic v2 更推荐的形式，因为它让类型和元数据分离得更干净（见第一章「类型信息提供层」的 `Annotated` 一节）：

```python
from typing import Annotated

class User(BaseModel):
    name: Annotated[str, Field(min_length=2, max_length=20)] = "Anonymous"
    age: Annotated[int, Field(ge=0, le=120)] = 18
```

对应 Java：需要 Hibernate Validator 配合注解，且必须在调用处用 `@Valid` 开启切面校验，否则注解形同虚设：

```java
import jakarta.validation.constraints.*;

public record User(
    @NotNull Long id,
    @Size(min = 2, max = 20) String name,
    @Email @NotBlank String email,
    @Min(0) @Max(120) Integer age
) {}
```

关键差别：Java 的校验**默认不发生**，要靠 `@Valid` 触发；Pydantic 的校验**默认发生**，是 `__init__` 的一部分，无法绕过。

#### 2.3 ValidationError 与错误聚合

Pydantic 抛出的 `ValidationError` 是结构化的，可以直接转成 API 响应：

```python
from pydantic import ValidationError

try:
    User(id="abc", email="bad", age=200)
except ValidationError as e:
    print(e.error_count())   # 3
    for err in e.errors():
        print(err["loc"], err["type"], err["msg"])
    # ('id',)    int_parsing        Input should be a valid integer...
    # ('email',) value_error        value is not a valid email address
    # ('age',)   less_than_equal    Input should be less than or equal to 120
```

`loc` 是字段路径，嵌套模型时会是 `("items", 0, "name")` 这样的元组，能精确定位到出错位置。FastAPI 正是拿这个结构直接生成 422 响应体的。

#### 2.4 AI-Infra 实例

```python
# vLLM: vllm/entrypoints/openai/protocol.py
class ChatCompletionRequest(OpenAIBaseModel):
    model: str
    messages: list[ChatCompletionMessageParam]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool | None = False

# FastAPI 路由直接使用 Pydantic 模型
@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    # 进入函数体时 request 已经过校验，类型安全
    ...
```

相当于 Spring Boot 的 `@RequestBody` + `@Valid` + Swagger，但零配置。

### 3. 序列化、反序列化与 Schema 生成

数据契约不只是"在内存里长什么样"，还包括**怎么进来、怎么出去、怎么被外部理解**。这是 Pydantic 相比 `@dataclass` 的另一个主要优势。

#### 3.1 model_dump 与 model_dump_json

```python
config = InferenceConfig(model_name="llama-3", backend="cuda")

config.model_dump()
# {'model_name': 'llama-3', 'backend': 'cuda', 'max_tokens': 2048, ...}

config.model_dump_json()
# '{"model_name":"llama-3","backend":"cuda","max_tokens":2048,...}'

# 常用选项
config.model_dump(exclude={"top_p"})          # 排除字段
config.model_dump(exclude_defaults=True)      # 只输出被显式设置过的字段
config.model_dump(mode="json")                # 把 datetime/UUID 等转成 JSON 可序列化的形式
```

`@dataclass` 也能序列化，但要自己动手，且不处理嵌套的非 JSON 原生类型：

```python
from dataclasses import asdict
import json

json.dumps(asdict(some_dataclass))    # datetime 字段会直接抛 TypeError
```

对应 Java：Jackson 的 `ObjectMapper.writeValueAsString()`，配合 `@JsonProperty`、`@JsonIgnore` 控制字段。

#### 3.2 model_json_schema 与 OpenAPI

```python
InferenceConfig.model_json_schema()
# {
#   'type': 'object',
#   'properties': {
#     'model_name': {'type': 'string', 'title': 'Model Name'},
#     'backend': {'enum': ['cuda', 'rocm', 'cpu'], 'type': 'string'},
#     'max_tokens': {'type': 'integer', 'maximum': 32768, 'minimum': 1, 'default': 2048},
#     ...
#   },
#   'required': ['model_name', 'backend']
# }
```

注意 `Literal["cuda", "rocm", "cpu"]` 被翻译成了 JSON Schema 的 `enum`，`Field(ge=1, le=32768)` 变成了 `minimum`/`maximum`——**类型注解和约束被完整地传递到了外部契约**。FastAPI 的 `/docs` 页面就是拿这个 schema 渲染的。

对应 Java：需要额外引入 Swagger / springdoc 注解，且与 Bean Validation 的注解是两套体系。

#### 3.3 解析 YAML / JSON 配置文件

AI-Infra 项目大量使用 YAML 配置（vLLM 的引擎参数、DeepSpeed 的并行策略、训练任务的超参）。裸读 YAML 得到的是一个 `dict[str, Any]`，类型信息全丢——这正是数据契约要解决的问题。

`model_validate()` 把任意 dict 转成校验过的模型：

```python
import yaml
from pydantic import BaseModel, Field
from typing import Literal

class ParallelConfig(BaseModel):
    tensor_parallel_size: int = Field(default=1, ge=1)
    pipeline_parallel_size: int = Field(default=1, ge=1)

class ServingConfig(BaseModel):
    model_name: str
    dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"
    max_model_len: int = Field(default=4096, ge=1)
    parallel: ParallelConfig = ParallelConfig()      # 嵌套模型

with open("serving.yaml") as f:
    raw = yaml.safe_load(f)          # dict[str, Any]，无类型保障

config = ServingConfig.model_validate(raw)   # 校验 + 转换 + 嵌套构建
```

对应的 `serving.yaml`：

```yaml
model_name: meta-llama/Llama-3-8B
dtype: bfloat16
max_model_len: 8192
parallel:
  tensor_parallel_size: 4
```

这样做的收益：

- **嵌套自动构建**：`parallel` 那一段 dict 自动变成 `ParallelConfig` 实例，不用手工递归；
- **拼写错误立刻暴露**：YAML 里写成 `dtype: bf16` 会在启动时报错，而不是等到加载权重时才崩；
- **启动即失败**：配置错误在进程启动的一瞬间抛出，不会浪费几分钟加载模型后才失败——这对动辄几十 GB 权重的推理服务尤其重要；
- **IDE 补全**：后续代码写 `config.parallel.tensor_parallel_size` 有完整提示。

如果 YAML 中有多余字段，默认会被忽略；想让它报错（防止配置项拼错被静默吞掉），加上 `extra="forbid"`：

```python
from pydantic import ConfigDict

class ServingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ...
```

反过来，把模型写回 YAML：

```python
yaml.safe_dump(config.model_dump(mode="json"))
```

### 4. BaseSettings：配置即契约

配置是数据契约的一个特例：数据源是环境变量和 `.env` 文件，内容全是字符串，需要解析成强类型对象。`pydantic-settings` 提供的 `BaseSettings` 专门做这件事。

#### 4.1 基本用法与 .env

```python
from pydantic import PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # 1. 强类型声明
    APP_NAME: str = "Awesome App"  # 环境变量没配时用默认值
    DEBUG: bool = False            # 自动把 "True"、"true"、"1" 解析为 True
    PORT: int = 8000               # 自动把 "8000" 解析为数字 8000

    # 还可以使用 Pydantic 的高级类型，自动校验 URL 格式
    DATABASE_URL: PostgresDsn

    # 2. 配置读取行为
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# 实例化时自动读取环境变量和 .env 文件
settings = Settings()

print(settings.APP_NAME)
print(settings.PORT)       # int，不是 str
```

对应的 `.env`：

```text
DEBUG=True
PORT=9000
DATABASE_URL=postgresql://user:pass@localhost:5432/dbname
```

#### 4.2 优先级、前缀与 Fail-Fast

- **大小写不敏感**（默认）：类里定义 `PORT`，环境变量写成 `port=1234` 也能识别。
- **优先级**（由高到低）：
  1. 实例化时显式传入的值（`Settings(PORT=5000)`）
  2. 操作系统环境变量（`export PORT=...`）
  3. `.env` 文件中的值
  4. 类中定义的默认值
- **前缀支持**：项目复杂时为防止环境变量冲突可加前缀，在 `SettingsConfigDict` 中设 `env_prefix="APP_"`，则 `APP_PORT` 映射到 `PORT`。
- **Fail-Fast**：执行 `settings = Settings()` 的那一瞬间，任何必填配置缺失或类型错误（比如 `PORT` 被配成 `"hello"`）都会立刻抛异常并阻止程序启动。

最后这一条是配置契约最重要的性质：**把配置错误从运行期挪到启动期**。

#### 4.3 多环境配置

指定多个 `.env` 文件，右边的覆盖左边的：

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DB_HOST: str
    API_KEY: str

    model_config = SettingsConfigDict(
        env_file=(".env.base", ".env.production"),
        env_file_encoding="utf-8",
    )
```

或者根据环境变量动态选择配置文件（类似 Spring Profile）：

```python
import os
from pydantic_settings import BaseSettings, SettingsConfigDict

# 先从系统环境获取当前运行环境，默认 development
run_env = os.getenv("ENV", "development")

class Settings(BaseSettings):
    DEBUG: bool
    DATABASE_URL: str

    # 动态加载 .env.development 或 .env.production
    model_config = SettingsConfigDict(
        env_file=f".env.{run_env}",
        env_file_encoding="utf-8",
    )

settings = Settings()
```

#### 4.4 对比 Spring `@ConfigurationProperties`

两者目的相同——把松散的配置字符串映射为强类型对象——但实现差异明显：

| 特性 | Pydantic BaseSettings | Spring `@ConfigurationProperties` |
|---|---|---|
| 底层核心技术 | 类型注解运行时解析 | 反射、Setter 或构造器注入 |
| 默认支持格式 | `.env`、系统环境变量、JSON / YAML（需插件） | `.properties`、`.yml` / `.yaml` |
| 框架耦合度 | 完全独立，任何脚本都能直接实例化 | 深度绑定 Spring 容器，需为 Bean |
| 前缀映射机制 | 扁平化为主，`env_prefix="APP_"` 匹配 `APP_PORT` | 天然层级嵌套，`prefix = "app"` 匹配 `app.database.url` |
| 校验触发时机 | 实例化时立即校验 | 容器启动阶段，需配合 `@Validated` |
| 宽松绑定 | 较严格，主要靠大小写不敏感 | 极宽松，`server.port`、`server_port`、`SERVER_PORT` 都能映射 |

### 5. 选型指南：dataclass vs Pydantic vs TypedDict

#### 5.1 三者对比

| 特性 | TypedDict | dataclass | Pydantic |
|---|---|---|---|
| 运行时类型 | 普通 `dict` | 自定义类实例 | 自定义类实例 |
| 运行时校验 | 无 | 无（除非手写 `__post_init__`） | **自动校验** |
| 类型转换 | 无 | 无 | **自动转换**（`"1"` → `1`） |
| 类型检查 | 静态 | 静态 | 静态 + 运行时 |
| 性能开销 | 零（就是 dict） | 极低 | 有（校验成本） |
| 序列化 | 天然是 dict，直接 `json.dumps` | 需 `asdict()`，不处理特殊类型 | 内置 `model_dump_json()` |
| Schema 生成 | 无 | 无 | `model_json_schema()` |
| 适用场景 | 已经是 dict 的数据 | 内部数据传递 | 系统边界、外部输入 |

#### 5.2 决策树

```text
外部数据（HTTP / JSON / YAML / 环境变量）
    └─ 需要解析和校验 ──→ Pydantic BaseModel
    └─ 来自环境变量 / .env ──→ Pydantic BaseSettings

内部数据传递
    └─ 需要不可变（可做 dict key / 跨线程共享）──→ @dataclass(frozen=True) / NamedTuple
    └─ 需要可变状态 ──→ @dataclass
    └─ 实例数量极大、内存敏感 ──→ @dataclass(slots=True)

数据本身就是 dict（第三方 API 返回值、已解析的 JSON）
    └─ 只想约束形状，不想改变运行时行为 ──→ TypedDict
```

#### 5.3 核心原则：边界校验一次，内部自由传递

**在系统边界用 Pydantic 校验一次，内部传递 `@dataclass` 对象。**

这正是第二章「静态与运行时的协作边界」中信任边界（Trust Boundary）模式在数据建模上的体现：外部数据不可信，进门时付一次校验成本；进门之后数据已经可信，用零开销的 `@dataclass` 传递，靠 mypy 做静态保障。

vLLM 的源码就是这个模式：API 层（`entrypoints/openai/protocol.py`）用 Pydantic 定义请求体，引擎内部（`SamplingParams`、`SchedulerConfig`）一律用 `@dataclass`。热路径上不做重复校验。

反过来的两种常见错误：

- **内部到处用 Pydantic**：每次构造对象都跑一遍校验，在每 token 都要构造对象的推理热路径上会成为可观的开销；
- **边界用 dataclass**：外部脏数据长驱直入，错误在很深的调用栈里才暴露，排查成本极高。

> 顺带一提，`attrs` 是比 `@dataclass` 更早、功能更全的第三方库，但在 AI-Infra 生态中已基本被"标准库 `@dataclass` + Pydantic"的组合取代，新项目一般不需要引入。


## 附录


### Java 与 Python 类型系统对照

| 维度 | Java 泛型 | Python 泛型 |
|---|---|---|
| 引入版本 | Java 5 (2004) | Python 3.5 (2015) |
| 实现方式 | 类型擦除（编译后泛型信息消失） | **纯注解**（运行时完全没有泛型检查） |
| 运行时获取泛型信息 | 困难（需要反射技巧） | `get_type_hints()` 可以获取 |
| 通配符 | `? extends T` / `? super T` | `TypeVar(covariant/contravariant)` |
| 上界约束 | `<T extends Number>` | `TypeVar("T", bound=Number)` |
| 多重约束 | `<T extends A & B>` | 不直接支持（可用 Protocol 组合） |
| 协议/接口 | 名义类型（必须 implements） | Protocol 结构化类型（不需要继承） |
| 检查时机 | 编译期 | 开发时（mypy/pyright）或运行时（Pydantic） |
| 新语法 | 无变化 | 3.12+ `class Stack[T]:` |


### Java 与 Python 数据契约对照

| 场景 | Java | Python |
|---|---|---|
| 不可变数据载体 | `record` | `@dataclass(frozen=True)` |
| 减模板代码 | Lombok `@Data` / `@Value` / `@Builder` | `@dataclass` |
| 运行时字段校验 | Bean Validation + `@Valid`（需显式触发） | Pydantic `BaseModel`（默认触发） |
| JSON 序列化 / 反序列化 | Jackson `@JsonProperty` | Pydantic `model_dump()` / `model_validate()` |
| API Schema 生成 | Swagger / springdoc 注解 | Pydantic `model_json_schema()`（自动） |
| 配置绑定 | Spring `@ConfigurationProperties` | Pydantic `BaseSettings` |
| 字典类型约束 | `Map<K,V>` + DTO | `TypedDict` |
| 轻量返回值 | `record` / 匿名类 | `NamedTuple` |
| 可替换接口 | `interface` | `Protocol` |
| 枚举 | `enum` | `enum.Enum` / `Literal` |
| 实现机制 | 编译期改 AST（Lombok）/ 运行时反射（Validator） | 运行时 `exec` 代码生成（dataclass）/ 元类 + Rust（Pydantic） |

核心差异：Java 用**多个独立框架**拼出完整的数据契约能力，每个框架各管一段；Python 用 **Pydantic 一个库**覆盖了校验、转换、序列化、Schema 生成、配置绑定的全部环节。


### 类型工具选择决策树

```
需要定义数据结构？
├── 数据来自外部（API/JSON/YAML/用户输入）？ → Pydantic BaseModel
├── 数据来自环境变量 / .env？ → Pydantic BaseSettings
├── 数据本身就是 dict，只想约束形状？ → TypedDict
└── 内部传递？
    ├── 需要不可变 → @dataclass(frozen=True) / NamedTuple
    ├── 实例数量极大 → @dataclass(slots=True)
    └── 其他 → @dataclass
│
需要定义接口？
├── 你控制实现类的代码？
│   ├── 是 → ABC（抽象基类）
│   └── 否 → Protocol
│
需要约束参数值？
├── 几个固定字符串？ → Literal
├── 枚举类型 + 穷尽检查？ → Enum + assert_never
└── 数值范围/格式？ → Annotated + Pydantic Field
│
需要泛型？
├── 简单的"输入输出类型一致"？ → TypeVar
├── 自定义泛型容器？ → Generic[T]
└── 装饰器保留签名？ → ParamSpec
│
函数有多种调用方式？ → @overload
│
需要运行时类型检查？
├── 数据建模 + 校验 → Pydantic
├── 函数级防御 → beartype
└── 简单分支判断 → isinstance
```


### typing 功能速查表

| 工具 | 版本 | 一句话说明 | Java 对应 | 频次 |
|---|---|---|---|---|
| `list[str]` | 3.9 | 内置容器泛型 | `List<String>` | ★★★★★ |
| `X \| Y` | 3.10 | 联合类型 | sealed interface | ★★★★★ |
| `Optional[X]` | 3.5 | `X \| None` 的语法糖 | `Optional<X>` | ★★★★★ |
| `Any` | 3.5 | 逃逸舱，跳过检查 | 裸类型 `List` | ★★★★☆ |
| `Literal` | 3.8 | 字面量类型 | `enum` | ★★★★☆ |
| `TypeVar` | 3.5 | 泛型类型变量 | `<T>` | ★★★★☆ |
| `Generic[T]` | 3.5 | 泛型基类 | `class Foo<T>` | ★★★★☆ |
| `ABC` | 2.6 | 抽象基类 | `abstract class` | ★★★★★ |
| `Protocol` | 3.8 | 结构化子类型 | 无直接对应 | ★★★☆☆ |
| `TypedDict` | 3.8 | 字典形状约束 | DTO / record | ★★★☆☆ |
| `Annotated` | 3.9 | 附加元数据 | `@Annotation` | ★★★★☆ |
| `Callable` | 3.5 | 函数类型 | `Function<T,R>` | ★★★★☆ |
| `ParamSpec` | 3.10 | 保留函数签名 | 无直接对应 | ★★☆☆☆ |
| `Concatenate` | 3.10 | 装饰器添加参数 | 无直接对应 | ★☆☆☆☆ |
| `TypeGuard` | 3.10 | 类型收窄函数 | `instanceof` | ★★☆☆☆ |
| `TypeIs` | 3.12 | 改进的类型收窄 | `instanceof` | ★☆☆☆☆ |
| `overload` | 3.5 | 多签名声明 | 方法重载 | ★★★★☆ |
| `Final` | 3.8 | 常量标记 | `final` | ★★★☆☆ |
| `ClassVar` | 3.5.3 | 类变量（非实例） | `static` | ★★★☆☆ |
| `Self` | 3.11 | 返回自身类型 | 返回 `this` | ★★★☆☆ |
| `Never` | 3.11 | 不可能的类型 | Kotlin `Nothing` | ★★☆☆☆ |
| `cast` | 3.5 | 类型断言 | `(Type) obj` | ★★★★☆ |
| `TYPE_CHECKING` | 3.5.2 | 避免循环导入 | 无直接对应 | ★★★★★ |
| `TypeAlias` | 3.10 | 类型别名 | 无 | ★★★★☆ |
| `get_type_hints()` | 3.5 | 运行时获取注解 | `Field.getGenericType()` | ★★★☆☆ |

> 频次说明：基于 PyTorch、vLLM、FastAPI、Pydantic、httpx、SQLAlchemy 等主流项目源码中的实际出现情况估算。★★★★★ 表示几乎每个模块都会用到，★☆☆☆☆ 表示仅在特定场景出现。


## 结语

回头看这三章，Python 的类型系统其实是一条链路：**注解把类型意图写下来，存根和 `py.typed` 把它分发出去，mypy 和 Pydantic 在两端各自消费它，最后落到数据契约上变成可执行的约束。**

与 Java 的最大差异不在于语法，而在于这条链路是**拆开的**。Java 把声明、检查、载体、反射合为一体，你没得选；Python 把每一环都做成可插拔的组件，你可以只写注解不做检查，也可以只在边界上做运行时校验而内部完全不管。这种自由度是代价也是优势——代价是需要自己决定在哪里投入，优势是可以按项目实际情况精确控制。

无论是 Java 靠注解切面实现的 Bean Validation，还是 Python 用元类与 Rust 引擎构建的 Pydantic，本质都是把开发者从繁琐的"防错代码"中解放出来。理解了 `__annotations__` 与元类之后会发现，Pydantic 并不是什么不可知的魔法——它只是充分利用了 Python 的动态性，把校验逻辑下沉到了语言机制层面。

对 AI-Infra 工程来说，实践上最值得记住的就三条：

1. **注解要写**，哪怕暂时不上 mypy——它首先是给人读的；
2. **边界要校验**，用 Pydantic 挡住所有外部输入，让错误在启动时或入口处暴露；
3. **热路径要干净**，内部传递用 `@dataclass`，不要在每 token 的循环里反复做运行时检查。

再遇到 AI Infra 源码中的类型注解，就不会觉得是天书了。关键不是一次记住所有工具，而是理解每个工具解决的问题——在真实代码中遇到时能查到、能读懂、能用对。
