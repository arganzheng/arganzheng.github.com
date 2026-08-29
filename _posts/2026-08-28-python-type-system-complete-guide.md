---
layout: post
title: Python 类型系统完全指南
tags: [Python]
catalog: true
---


Python 是动态类型语言，但这不意味着"无类型"。自 Python 3.5 引入 `typing` 模块以来，类型注解已经从"可选装饰"演变为大型项目的工程标配。PyTorch、vLLM、FastAPI 等 AI Infra 项目大量使用类型系统的高级特性。

Python 的类型系统由多个部分组成：

- **`typing` 模块**：核心类型注解工具箱（`TypeVar`、`Protocol`、`Literal`、`ParamSpec` 等）
- **`abc` 模块**：抽象基类，定义接口契约（`ABC`、`abstractmethod`）
- **`collections.abc`**：标准容器的抽象接口（`Iterable`、`Sequence`、`Mapping` 等）
- **类型检查器**：mypy、pyright——静态分析工具，真正执行类型检查
- **`.pyi` 存根文件**：为无类型注解的库提供类型信息
- **`typing_extensions`**：新特性的向后移植包

对于 Java 程序员来说，这套体系既熟悉又陌生：熟悉的是泛型、接口这些概念都有对应；陌生的是它完全不影响运行时行为，而且有 `Protocol`、`TypeGuard`、`ParamSpec` 这些 Java 中没有直接对应的工具。

本文将系统梳理 Python 类型系统的重要功能，每个特性都会说明：**它解决什么问题、怎么用、Java 中对应什么、在 AI Infra 真实项目中长什么样**。


## 一、类型注解基础：从"注释"到"契约"

### 1. 变量和函数注解

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

核心区别：Java 的类型声明是语法强制的，编译器检查；Python 的类型注解**默认不影响运行时**，需要 mypy/pyright 做静态检查。

### 2. 内置容器类型注解的演进

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

### 真实项目中的基础注解

```python
# FastAPI: 路由函数的参数和返回值都有完整注解
# fastapi/applications.py
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

# PyTorch: Tensor 操作的类型注解
# torch/_C/_VariableFunctions.pyi
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


## 二、Union、Optional 与 None：表达"可能性"

### 1. Union 类型

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

### 2. Optional：可空类型

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

### 3. 真实项目中的用法

```python
# vLLM: sampling_params.py
@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int | None = None  # None 表示不限制
    stop: list[str] = field(default_factory=list)
```


## 三、Any、Never 与 NoReturn：类型系统的边界

### 1. Any：逃逸舱

```python
from typing import Any

def process(data: Any) -> Any:
    # Any 与所有类型兼容，类型检查器不会报错
    return data.whatever()
```

`Any` 类似 Java 的裸类型（raw type）：`List` 而不是 `List<String>`。它告诉类型检查器"不要管这个"。

**使用场景**：和无类型注解的第三方库交互、快速原型阶段。**不要**把它当作"我不知道该写什么类型"的默认选择。

### 真实项目中的 Any

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

# httpx: httpx/_types.py
# 请求数据可以是多种形式，用 Any 作为最外层约束
RequestData = Mapping[str, Any]
```

`Any` 在成熟项目中通常出现在两种位置：

1. **对外接口的入口**——接受用户传入的任意数据（如 Pydantic 的 `model_validate`、FastAPI 的 `Depends`）
2. **动态分发的边界**——对象在运行时才确定具体类型（如 PyTorch 的 `Module.__setattr__`）

核心内部逻辑尽量避免使用 `Any`，用它意味着你主动放弃了类型检查的保护。

### 2. Never 与 NoReturn

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

### 真实项目中的 NoReturn / Never

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

### AI-Infra 中的穷尽检查

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


## 四、Literal：字面量类型

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

### 真实项目中的 Literal

```python
# vLLM: vllm/config.py — 量化方法限定为几个固定字符串
QuantMethod = Literal["awq", "gptq", "squeezellm", "marlin"]

# Pydantic: pydantic/fields.py — 字段的 JSON Schema 模式
JsonSchemaMode = Literal["validation", "serialization"]

# httpx: httpx/_types.py — HTTP 方法限定
HttpMethod = Literal["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]

# Rich（终端美化库）: rich/console.py — 对齐方式
JustifyMethod = Literal["default", "left", "center", "right", "full"]
```

`Literal` 非常适合替代那些"只接受几个固定字符串"的场景——不值得定义一个完整 Enum 类，但又想让类型检查器帮你约束。在实际项目中，`Literal` 大量用于配置选项和 API 参数的约束。


## 五、TypeVar 与泛型：复用类型关系

### 1. TypeVar 基础

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

### 2. bound：类型上界

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

### 3. 约束到特定类型

```python
# T 只能是 str 或 bytes，不能是其他类型
StrOrBytes = TypeVar("StrOrBytes", str, bytes)

def concat(a: StrOrBytes, b: StrOrBytes) -> StrOrBytes:
    return a + b
```

这比 `Union[str, bytes]` 更严格：它要求 `a` 和 `b` 必须是**同一个类型**。

### 4. 自定义泛型类

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

### 5. 协变与逆变

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

### 6. Python 3.12+ 的新语法

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

### 真实项目中的 TypeVar 与泛型

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

# Pydantic: pydantic/main.py
# BaseModel.model_validate 用 Self（本质是 bound TypeVar）
# 确保子类调用后返回的仍然是子类类型
class BaseModel:
    @classmethod
    def model_validate(cls, obj: Any) -> Self: ...

class UserModel(BaseModel):
    name: str

user = UserModel.model_validate({"name": "Alice"})  # user: UserModel，不是 BaseModel

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


## 六、Callable：函数类型

### 1. 基本用法

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

### 2. 更灵活的可调用类型

`Callable[[int, int], int]` 无法表达 keyword-only 参数、默认值等复杂签名。如果需要精确描述，使用 Protocol：

```python
from typing import Protocol

class Comparator(Protocol):
    def __call__(self, a: str, b: str, *, reverse: bool = False) -> int: ...

def sort_with(items: list[str], cmp: Comparator) -> list[str]:
    ...
```

### 3. 任意参数的 Callable

```python
from typing import Callable

# 接受任意参数的函数
handler: Callable[..., None]  # ... 表示"任意参数"
```

### 真实项目中的 Callable

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


## 七、ABC 与 Protocol：定义接口的两种方式

Python 定义"接口"有两种机制：**ABC（抽象基类）**和 **Protocol（协议）**。它们的定位不同，适用场景不同，理解两者的区别是读懂 AI Infra 源码的关键。

### 1. ABC：抽象基类（名义类型）

ABC（Abstract Base Class）来自标准库的 `abc` 模块，对应 Java 的 `abstract class` + `interface`。它是 Python 中历史最悠久的接口定义方式。

```python
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def speak(self) -> str:
        """子类必须实现此方法——对应 Java 的抽象方法"""
        ...

    def breathe(self) -> str:
        """可以提供默认实现——对应 Java abstract class 的普通方法"""
        return "breathing..."

# 使用 ABC 时，子类必须显式继承
class Dog(Animal):
    def speak(self) -> str:
        return "Woof!"

# 如果忘记实现抽象方法，实例化时立刻报 TypeError（类似 Java 编译错误）
class BadAnimal(Animal):
    pass

BadAnimal()  # TypeError: Can't instantiate abstract class BadAnimal
             # with abstract method speak
```

对应 Java：

```java
abstract class Animal {
    abstract String speak();        // 子类必须实现
    String breathe() {              // 可以有默认实现
        return "breathing...";
    }
}

class Dog extends Animal {
    String speak() { return "Woof!"; }
}
```

#### 真实项目中的 ABC

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

#### collections.abc 速查

`collections.abc` 提供了 Python 最常用的抽象基类，理解它们对读懂类型注解至关重要：

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

### 2. Protocol：结构化子类型（静态鸭子类型）

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

### 3. ABC vs Protocol：选择指南

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

### 4. 带 `@runtime_checkable` 的 Protocol

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

### 5. 泛型 Protocol

```python
from typing import Protocol, TypeVar

T_co = TypeVar("T_co", covariant=True)

class Reader(Protocol[T_co]):
    def read(self) -> T_co: ...

def process(reader: Reader[str]) -> str:
    return reader.read().upper()
```

### 6. 真实项目中的 Protocol

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


## 八、TypedDict：字典的类型约束

Python 中大量使用 `dict` 传递数据。`TypedDict` 让你能对字典的"形状"（哪些 key、每个 key 的值类型）进行静态约束。

### 1. 基本用法

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

### 2. 可选字段

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

### 3. TypedDict vs dataclass vs Pydantic

Python 有三种主流的"结构化数据"定义方式。简单说：`dataclass` 是标准库提供的数据类（类似 Java Record），Pydantic 是带运行时校验的增强版（类似 Java Bean Validation）。关于这两者的详细对比，参见我的另一篇文章《[Python中如何定义POJO](/python-dataclass-definition-and-validation.html)》。

这里聚焦 TypedDict 和它们的区别：

| 特性 | TypedDict | dataclass | Pydantic |
|---|---|---|---|
| 运行时类型 | 普通 `dict` | 自定义类实例 | 自定义类实例 |
| 运行时校验 | 无 | 无（除非手动） | **自动校验** |
| 适用场景 | JSON 数据、字典形状约束 | 内部数据传递 | API 边界、外部输入 |
| 类型检查 | 静态 | 静态 | 静态 + 运行时 |
| 性能开销 | 零（就是 dict） | 极低 | 有（校验成本） |
| 序列化/反序列化 | 天然是 dict，直接 json.dumps | 需要 `asdict()` | 内置 `.model_dump_json()` |

**选择原则**：如果数据已经是 dict（如 JSON API 返回值、配置文件解析结果），用 TypedDict 约束形状最自然；如果需要创建新的结构化对象，用 dataclass 或 Pydantic。

### 4. 用 TypedDict 约束 `**kwargs`（3.12+）

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

### 真实项目中的 TypedDict

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


## 九、Annotated：给类型附加元数据

`Annotated` 允许在类型上附加额外的元数据，类型检查器本身忽略这些元数据，但框架（如 FastAPI、Pydantic）可以读取并使用。

```python
from typing import Annotated

# 基本语法：Annotated[类型, 元数据1, 元数据2, ...]
UserId = Annotated[int, "must be positive"]
```

### 1. Pydantic 中的 Annotated

```python
from typing import Annotated
from pydantic import BaseModel, Field

class User(BaseModel):
    name: Annotated[str, Field(min_length=2, max_length=50)]
    age: Annotated[int, Field(ge=0, le=150)]
```

### 2. FastAPI 中的 Annotated

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


## 十、ParamSpec 与 Concatenate：保留装饰器的类型信息

### 1. 问题：装饰器吃掉了类型信息

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

### 2. ParamSpec 解决方案

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

### 3. Concatenate：装饰器添加参数

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

### 真实项目中的 ParamSpec

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

### AI-Infra 中的 ParamSpec 与 Concatenate

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


## 十一、TypeGuard 与 TypeIs：类型收窄

### 1. TypeGuard（3.10+）

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

### 2. TypeIs（3.12+）

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

### 真实项目中的 TypeGuard

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

### AI-Infra 中的 TypeGuard / TypeIs

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


## 十二、overload：多签名声明

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

### 真实项目中的 overload

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


## 十三、其他实用工具

### 1. Final 和 ClassVar

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

### 2. Self（3.11+）

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

### 3. TypeAlias（3.10+）与 `type` 语句（3.12+）

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

### 4. cast：类型断言

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

### 5. TYPE_CHECKING：避免循环导入

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

### 真实项目中的 TYPE_CHECKING

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


## 十四、类型检查工具：mypy 与 pyright

类型注解本身不会被 Python 解释器检查，需要外部工具。

### 1. mypy

```bash
# 安装
pip install mypy

# 基本检查
mypy src/

# 严格模式（推荐用于新项目）
mypy --strict src/
```

常用配置（`pyproject.toml`）：

```toml
[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true

# 对第三方库禁用检查
[[tool.mypy.overrides]]
module = "torch.*"
ignore_missing_imports = true
```

### 2. pyright

```bash
# 安装（通常通过 VS Code 的 Pylance 扩展自动获得）
pip install pyright

# 检查
pyright src/
```

### 3. mypy vs pyright

| 特性 | mypy | pyright |
|---|---|---|
| 语言 | Python | TypeScript (Node.js) |
| 速度 | 较慢 | **快很多** |
| IDE 集成 | 插件 | VS Code Pylance 内置 |
| 严格程度 | 可配置 | 默认更严格 |
| 社区 | 官方维护 | Microsoft 维护 |

两者都广泛使用。如果用 VS Code 开发，pyright 通过 Pylance 自动工作；CI 中 mypy 更常见。

### 4. `.pyi` 存根文件

当一个库本身没有类型注解时（比如用 C 写的扩展模块），可以通过 `.pyi` 存根文件为其提供类型信息。`.pyi` 文件只包含签名，不包含实现：

```python
# torch/_C/__init__.pyi — PyTorch 的 C++ 扩展模块的类型存根
# 这个文件告诉 mypy/pyright：torch._C 里有哪些函数、什么签名

def _get_tracing_state() -> bool: ...
def _set_grad_enabled(enabled: bool) -> None: ...

class TensorBase:
    def dim(self) -> int: ...
    def size(self, dim: int | None = None) -> Size: ...
    def to(self, device: Device, dtype: dtype | None = None) -> Tensor: ...
```

存根文件的查找优先级：

1. **包内存根**：库自带的 `.pyi` 文件（如 PyTorch 的 `torch/_C/__init__.pyi`）
2. **typeshed**：Python 官方维护的标准库和知名第三方库的存根仓库，mypy/pyright 内置
3. **独立存根包**：通过 pip 安装，命名规则为 `types-<package>`（如 `types-requests`、`types-PyYAML`）
4. **自定义存根**：项目内自己写的 `.pyi` 文件

```bash
# 安装第三方库的存根
pip install types-requests types-PyYAML types-redis

# mypy 会自动发现并使用这些存根
```

对应 Java：Java 没有存根文件的概念——类型信息编译进 `.class` 文件。最接近的是 `.jar` 中不含实现的接口定义。

### 5. `typing_extensions`：新特性的向后移植

Python 类型系统发展很快，每个小版本都会加入新特性。但很多项目需要支持旧版本 Python（特别是 AI Infra 项目，线上环境经常是 3.8 或 3.10）。`typing_extensions` 包解决了这个问题——它把新版本的 typing 特性向后移植到旧版本：

```python
import sys

if sys.version_info >= (3, 11):
    from typing import Self, assert_never
else:
    from typing_extensions import Self, assert_never

if sys.version_info >= (3, 12):
    from typing import TypeIs
else:
    from typing_extensions import TypeIs
```

在实际项目中，更常见的做法是**直接从 `typing_extensions` 导入**，让包自动处理版本兼容：

```python
# Pydantic: pydantic/_internal/_generics.py
# 不管 Python 版本是多少，统一从 typing_extensions 导入
from typing_extensions import Self, TypeAlias, TypeGuard, get_args, get_origin

# vLLM: 很多模块的导入区域
from typing_extensions import ParamSpec, TypeIs

# httpx: httpx/_types.py
from typing_extensions import TypeAlias
```

`typing_extensions` 在 AI Infra 项目中几乎是必装依赖——PyTorch、Pydantic、vLLM、FastAPI 的 `requirements.txt` 里都有它。如果你写需要兼容多版本 Python 的库代码，优先从 `typing_extensions` 导入类型工具。


## 十五、横向对比：Java 与 Python 类型系统

| 维度 | Java 泛型 | Python 泛型 |
|---|---|---|
| 引入版本 | Java 5 (2004) | Python 3.5 (2015) |
| 实现方式 | 类型擦除（编译后泛型信息消失） | **纯注解**（运行时完全没有泛型检查） |
| 运行时获取泛型信息 | 困难（需要反射技巧） | `get_type_hints()` 可以获取 |
| 通配符 | `? extends T` / `? super T` | `TypeVar(covariant/contravariant)` |
| 上界约束 | `<T extends Number>` | `TypeVar("T", bound=Number)` |
| 多重约束 | `<T extends A & B>` | 不直接支持（可用 Protocol 组合） |
| 协议/接口 | 名义类型（必须 implements） | Protocol 结构化类型（不需要继承） |
| 新语法 | 无变化 | 3.12+ `class Stack[T]:` |


## 十六、总结：typing 功能速查表

| 工具 | 版本 | 一句话说明 | Java 对应 | 开源使用频次 |
|---|---|---|---|---|
| `list[str]` | 3.9 | 内置容器泛型 | `List<String>` | ★★★★★ 几乎每个文件 |
| `X | Y` | 3.10 | 联合类型 | sealed interface | ★★★★★ 极高 |
| `Optional[X]` | 3.5 | `X \| None` 的语法糖 | `Optional<X>` | ★★★★★ 极高（旧项目） |
| `Any` | 3.5 | 逃逸舱，跳过检查 | 裸类型 `List` | ★★★★☆ 高（边界/兼容层） |
| `Literal` | 3.8 | 字面量类型 | `enum` | ★★★★☆ 高（配置/选项） |
| `TypeVar` | 3.5 | 泛型类型变量 | `<T>` | ★★★★☆ 高 |
| `Generic[T]` | 3.5 | 泛型基类 | `class Foo<T>` | ★★★★☆ 高 |
| `ABC` | 2.6 | 抽象基类（名义类型） | `abstract class` | ★★★★★ 极高（框架基类） |
| `Protocol` | 3.8 | 结构化子类型（鸭子类型） | 无直接对应 | ★★★☆☆ 中（接口约定） |
| `TypedDict` | 3.8 | 字典形状约束 | DTO / record | ★★★☆☆ 中 |
| `Annotated` | 3.9 | 附加元数据 | `@Annotation` | ★★★★☆ 高（FastAPI/Pydantic） |
| `ParamSpec` | 3.10 | 保留函数签名 | 无直接对应 | ★★☆☆☆ 低（装饰器库） |
| `Concatenate` | 3.10 | 装饰器添加参数 | 无直接对应 | ★☆☆☆☆ 很低 |
| `TypeGuard` | 3.10 | 类型收窄函数 | `instanceof` | ★★☆☆☆ 低（类型存根） |
| `TypeIs` | 3.12 | 改进的类型收窄 | `instanceof` | ★☆☆☆☆ 新特性，正在采用 |
| `overload` | 3.5 | 多签名声明 | 方法重载 | ★★★★☆ 高（.pyi 存根） |
| `Final` | 3.8 | 常量标记 | `final` | ★★★☆☆ 中 |
| `ClassVar` | 3.5.3 | 类变量（非实例） | `static` | ★★★☆☆ 中（dataclass 必备） |
| `Self` | 3.11 | 返回自身类型 | 返回 `this` | ★★★☆☆ 中（链式/工厂） |
| `Never` | 3.11 | 不可能的类型 | 无（Kotlin `Nothing`） | ★★☆☆☆ 低（穷尽检查） |
| `Unpack` | 3.11 | TypedDict 解包 | 无直接对应 | ★☆☆☆☆ 新特性，正在采用 |
| `cast` | 3.5 | 类型断言 | `(Type) obj` | ★★★★☆ 高 |
| `TYPE_CHECKING` | 3.5.2 | 避免循环导入 | 无直接对应 | ★★★★★ 极高（大型项目标配） |
| `TypeAlias` | 3.10 | 类型别名 | 无（`typedef`） | ★★★★☆ 高 |
| `type X = ...` | 3.12 | 类型别名语句 | 无 | ★☆☆☆☆ 新特性，正在采用 |

> 频次说明：基于 PyTorch、vLLM、FastAPI、Pydantic、httpx、SQLAlchemy 等主流项目源码中的实际出现情况估算。★★★★★ 表示几乎每个模块都会用到，★☆☆☆☆ 表示仅在特定场景出现或属于较新特性尚未广泛采用。

掌握这张表，再遇到 AI Infra 源码中的类型注解，就不会觉得是天书了。关键不是一次记住所有工具，而是理解每个工具解决的问题——在真实代码中遇到时能查到、能读懂、能用对。优先掌握 ★★★★ 以上的高频工具，★★ 以下的低频工具遇到时再回来查阅即可。
