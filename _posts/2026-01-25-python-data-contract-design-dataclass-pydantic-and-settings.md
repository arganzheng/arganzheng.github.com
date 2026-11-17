---
layout: post
series: python-for-ai-infra
title: "Python 在 AI-Infra（02 下）：类型系统——数据契约设计"
subtitle: "Type System III: Data Contract Design"
tags: [Python]
catalog: true
date: 2026-01-25 20:00:00 +0800
updated: 2026-11-16
---

前两篇沿着"类型信息如何流动"展开：[上篇](/python-type-expression-and-the-typing-toolbox.html)讲怎么写（`typing` 工具箱），[中篇](/python-type-information-distribution-and-consumption.html)讲怎么分发、谁消费（存根、mypy、运行时读注解）。本篇的组织轴切换——不再讨论类型信息本身，而是讨论**用这些能力去构建什么**：数据契约。

所谓数据契约，就是对"一组数据长什么样"的正式约定。在 AI Infra 里它无处不在：推理服务的请求体与响应体、训练任务的配置文件、模型仓库里的元数据、Worker 之间传递的消息。Java 工程师熟悉的 Bean Validation、Jackson、`@ConfigurationProperties` 在这里对应的是 `@dataclass`、Pydantic `BaseModel`、`BaseSettings`——但它们不是一一对应的翻译，而是同一份类型注解的三种消费方式，各有适用位置。

本篇围绕的核心问题是：

> **什么时候用 `@dataclass`、什么时候上 Pydantic、什么时候 `TypedDict` 就够？校验该发生在哪里，序列化与配置怎么跟着契约走？[^q0]**

答案的骨架是中篇末尾那句话——**边界要校验，内部要干净**：外部数据进门时用 Pydantic 解析和校验，内部可以用不自动校验的 `@dataclass` 传递，由 mypy 检查代码中的类型使用。内部对象若仍会被修改，业务不变量还需要由接口和测试维护，注解不会替程序兜底。

## 一、总览

本篇沿着一份推理配置的生命周期展开：启动代码读取配置，Pydantic 解析并校验，调度器消费内部对象，API 或运维工具再把它导出。先看数据在哪一步跨过边界，再选承载它的类型。

### 1. 在三篇地图上的位置

![类型信息的三层地图：提供层（类型表达、类型载体与分发）经“读取、解释、推理、执行”到消费层（静态消费、动态消费），再“用消费层的能力构建”工程落地的数据契约（数据建模、契约的输入输出）](/img/in-post/python-data-contract-design-type-info-layers.svg)

本篇讲图里最下面那格。它**使用**前两格的能力，但组织轴不同：前两格沿着"类型信息如何流动"，这一格沿着"程序的数据结构如何设计"。

### 2. 一份推理配置从输入到消费

后面的例子用 `model_name="llama-3"`、`max_tokens="32"` 这组配置：文件或 HTTP 请求带来的 `"32"` 还是字符串，调度器需要的是受范围约束的整数；服务端口则由 `Settings()` 在启动时从环境变量等来源取得。

```mermaid
%% 图：推理配置在启动入口、请求入口与内部调度之间的数据流
flowchart TB
    FILE["配置文件 dict / HTTP JSON<br/>max_tokens 是字符串 32"] --> LOAD["启动代码 / FastAPI<br/>调用 model_validate 或请求校验"]
    ENV["环境变量 / .env / 默认值<br/>PORT 等启动参数"] --> SET["启动时调用 Settings()<br/>先选择来源，再解析并校验"]
    LOAD --> VALID{"Pydantic 校验通过？"}
    SET --> VALID
    VALID -- "否" --> FAIL["构造失败或请求校验错误<br/>启动代码 / API 处理异常"]
    VALID -- "是" --> MODEL["内存中的配置模型<br/>max_tokens / PORT 为整数"]
    MODEL -- "显式构建内部数据" --> DC["dataclass / 引擎参数对象<br/>调度器读取已检查的字段"]
    MODEL -- "model_dump / model_dump_json" --> OUTPUT["dict / JSON<br/>交给 API、日志或配置文件写入代码"]
```

- **读取不等于校验**：JSON / YAML 解析器只负责读出数据；`BaseModel` 才按字段约束检查它。
- **模型不自动落盘**：对象存在进程内存里，`model_dump_json()` 返回字符串；写文件或发响应仍由调用方完成。
- **转换内部结构是一次设计选择**：需要方法、内部状态或大量轻量实例时用 dataclass；没有性能或职责上的收益时，也可以继续传递已经校验的模型。

### 3. 本文的章节安排

| 章 | 主题 | 内容 |
|---|---|---|
| 二 | 工程落地：数据契约设计 | dataclass、Pydantic、序列化与 Schema、BaseSettings、选型指南 |
| 三 | 附录 | 类型工具选择决策树；Java 与 Python 数据契约对照 |
| 四 | 本文小结 |  |
| 五 | 自测 | 3 道题 |

Table: 本文的章节安排

## 二、工程落地：数据契约设计

同样是数据契约，输入来源和消费方不同，校验位置也不同：

- **请求体**：`/v1/chat/completions` 收到的 JSON 应该有哪些字段、什么类型、什么取值范围；
- **配置项**：从 `.env`、YAML 读进来的一堆字符串，怎么变成强类型的配置对象；
- **模型元数据**：模型名、量化方式、并行度、KV cache 配置在进程间传递时的形状；
- **内部数据结构**：一次推理请求在引擎内部流转时携带的上下文。

在 Java 里这件事由多个技术栈拼起来：Lombok / `record` 消除模板代码，Bean Validation 做校验，Jackson 做序列化，`@ConfigurationProperties` 做配置绑定。Python 则高度收敛——`@dataclass` 和 Pydantic 两个工具覆盖了绝大部分场景。看一眼 Pydantic 模型的定义：

```python title='一个 Pydantic 模型 InferenceConfig：Literal、Annotated 与 Field 约束'
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
2. **运行时校验**：通过普通构造或 `model_validate()` 进入模型的数据，会按约束解析、转换并校验（`ge=1`、`le=32768`）；
3. **文档 / Schema 生成**：把模型用于 FastAPI 路由后，框架将其 Schema 纳入 OpenAPI 文档。

这就是"数据契约"的价值——一处声明，三处受益。而它能成立，靠的正是[中篇第四章](/python-type-information-distribution-and-consumption.html#四类型信息消费层下动态消费运行时如何读取类型注解)讲的机制。

```python title='同一份 InferenceConfig：输入字符串变成内部整数'
config = InferenceConfig.model_validate(
    {"model_name": "llama-3", "backend": "cuda", "max_tokens": "32"}
)
print(config.max_tokens, type(config.max_tokens).__name__)  # 32 int
```

这里使用 Python 3.10–3.12 的常见写法，Pydantic API 均按 v2；`BaseSettings` 来自独立的 `pydantic-settings` 包。

> **前置阅读**：如果你想先搞清楚 `@dataclass` 和 Pydantic **怎么**读到类型注解、`exec` 代码生成和元类分别在什么时机介入，见[中篇第四章](/python-type-information-distribution-and-consumption.html#四类型信息消费层下动态消费运行时如何读取类型注解)「类型信息消费层（下）：动态消费」的"框架如何消费注解"一节。本章只讲用法与取舍。

### 1. dataclass：标准库的数据类

**从原生 `__init__` 到 `@dataclass`**

最传统的写法是显式定义构造函数：

```python title='手写 __init__ 的 User：满是 self.x = x'
class User:
    def __init__(self, id: int, name: str, email: str):
        self.id = id
        self.name = name
        self.email = email
```

问题很明显：**大量 `self.x = x` 的模板代码**。字段一多就难以维护，而且还要手写 `__repr__`、`__eq__` 才能方便调试和比较。

Python 3.7 引入的数据类消除了这些样板。它利用类变量类型标注（PEP 526）语法：

```python title='用 @dataclass 定义 User'
from dataclasses import dataclass

@dataclass
class User:
    id: int
    name: str
    email: str
```

实例化方式不变，但 `__init__`、`__repr__`、`__eq__` 都自动有了：

```python title='dataclass 自动生成的 __init__、__repr__、__eq__'
user = User(id=1, name="Alice", email="alice@example.com")
print(user)          # User(id=1, name='Alice', email='alice@example.com')
print(user == User(1, "Alice", "alice@example.com"))   # True
```

> [!NOTE] 与 Java 对照：数据载体与 record
> Lombok `@Data` 为普通类生成访问器、比较等方法；Java 16 正式引入的 `record` 则是另一种数据载体（Java 14、15 为预览）。下面是可单独保存为 `User.java` 的 Java 16+ 定义：
>
> ```java title='Java record：简洁的数据载体'
> public record User(Long id, String name, String email) {}
> ```
>
> 使用 Lombok 的项目也可以保留普通类（下面是另一份 `User.java`，需要 Lombok 依赖和注解处理器）：
>
> ```java title='Lombok @Data：为普通类生成模板方法'
> import lombok.Data;
>
> @Data
> public class User {
>     private Long id;
>     private String name;
>     private String email;
> }
> ```
>
> 默认的 Python dataclass 可以修改字段，不等于 record；两者也都不会仅凭字段类型声明完成外部 JSON 的解析和业务校验。

**`__post_init__` 手工校验及其局限**

`@dataclass` **不按字段注解做运行时类型校验**——它主要把注解当作生成方法的字段清单，也会特殊识别 `ClassVar`、`InitVar` 等标记。所以下面这行不会报错：

```python title='@dataclass 不做运行时校验：错误类型静默通过'
user = User(id="abc", name=123, email=None)   # 静默通过
```

如果需要校验，可以借助 `__post_init__` 钩子（自动生成的 `__init__` 完成字段赋值后调用）：

```python title='用 __post_init__ 手工校验字段'
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

> [!NOTE] 与 Java 对照：载体声明不自动校验
> Java record 可以在构造器里手写校验，但仅声明 record 或使用 Lombok `@Data` 不会自动校验外部输入。Bean Validation 与 Python 的 Pydantic 都是在载体之外安排约束和执行时机，触发方式有所不同。

**`frozen` 与 `slots`**

两个常用参数：

```python title='frozen=True 与 slots=True 两个常用参数'
from dataclasses import dataclass

# frozen=True：阻止普通字段赋值；默认 eq=True 时还会生成 __hash__
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

`frozen=True` 限制的是字段重新赋值，**不是深度不可变**。字段若是列表，仍能 `append()`；默认生成的哈希若遇到列表字段，也会抛 `TypeError`。上面的 `ModelKey` 全部使用字符串，所以适合作为字典键。`slots=True` 在本例没有基类实例字典时省去 `__dict__`；有基类提供字典时不能照搬这个结论。

> [!NOTE] 与 Java 对照：浅层不可变与实例存储
> Java record 的组件引用不能重新赋值，但引用指向的列表仍然能变；这一点与 `frozen=True` 的浅层限制相近。两者都不自动保证跨线程共享安全。`slots=True` 则用于控制 Python 实例的属性存储，没有对应的 Java 声明开关。

> `__slots__` 的内存收益取决于对象数量和字段类型，详见[《Python 内存管理与优化》](/python-memory-management-and-optimization.html)。

### 2. Pydantic：带校验的数据模型

Pydantic 是 FastAPI 常用的运行时数据校验库，把数据建模、类型转换、校验和导出放在同一模型上。它覆盖了 Java record、Bean Validation、Jackson 的部分职责，具体触发条件仍需分别理解。

**BaseModel 基础与类型强制转换**

```python title='一个 Pydantic BaseModel：User 带 EmailStr'
from pydantic import BaseModel, EmailStr

class User(BaseModel):
    id: int
    name: str
    email: EmailStr
```

与 `@dataclass` 最大的差别是它**真的会校验，并且会转换**：

```python title='Pydantic 真的校验并转换："1" 变成 int，错误一次报全'
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

注意两点：一是 `EmailStr` 需要安装可选依赖 `email-validator`（可用 `pip install 'pydantic[email]'`）；二是普通字段校验能**聚合多个错误**，而不是遇到第一个就中断——这正是[中篇第四章](/python-type-information-distribution-and-consumption.html#四类型信息消费层下动态消费运行时如何读取类型注解)提到的"收集所有错误路径"。

**Field：默认值与约束**

`Field()` 用来表达注解本身表达不了的约束（范围、长度、正则）：

```python title='Field()：默认值、范围、长度与正则约束'
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

约束也可以写在 `Annotated` 里，这是 Pydantic v2 更推荐的形式，因为它让类型和元数据分离得更干净（见[上篇第二章](/python-type-expression-and-the-typing-toolbox.html#二类型表达从基础注解到-typing-工具箱)「类型信息提供层（上）：类型表达」的 `Annotated` 一节）：

```python title='约束写进 Annotated：Pydantic v2 推荐形式'
from typing import Annotated
from pydantic import BaseModel, Field

class UserProfile(BaseModel):
    name: Annotated[str, Field(min_length=2, max_length=20)] = "Anonymous"
    age: Annotated[int, Field(ge=0, le=120)] = 18
```

> [!NOTE] 与 Java 对照：谁触发 Bean Validation
> 下列 Java 16+ record 需要 Jakarta Validation API；真正执行校验还需要 Hibernate Validator 等实现：
>
> ```java title='Bean Validation：声明字段约束'
> import jakarta.validation.constraints.*;
>
> public record User(
>     @NotNull Long id,
>     @Size(min = 2, max = 20) String name,
>     @Email @NotBlank String email,
>     @Min(0) @Max(120) Integer age
> ) {}
> ```
>
> `new User(...)` 本身不触发 Bean Validation。可以显式调用 `validator.validate(user)`，也可以交给 Spring MVC 等集成在请求入口触发；`@Valid` 还用于级联校验，不是通用的“开启切面”开关。

Pydantic 的普通构造路径默认校验，但**可以绕过**：`model_construct()` 用于已可信数据；`model_copy(update=...)` 的更新值不重新校验；字段赋值默认也不重校验，需要时开启 `ConfigDict(validate_assignment=True)`。默认还允许部分类型转换，严格边界可使用 `model_validate(raw, strict=True)`。这些都是库行为，不是 Python 注解的语言保证。

**ValidationError 与错误聚合**

Pydantic 抛出的 `ValidationError` 是结构化的，可以直接转成 API 响应：

```python title='结构化的 ValidationError：errors() 里的 loc / type / msg'
from pydantic import BaseModel, EmailStr, Field, ValidationError

class User(BaseModel):
    id: int
    email: EmailStr
    age: int = Field(ge=0, le=120)

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

`loc` 是字段路径，嵌套模型时会是 `("items", 0, "name")` 这样的元组，能精确定位到出错位置。FastAPI 会把请求校验错误包装成响应；模型其他位置抛出的 `ValidationError` 不会都自动变成 HTTP 422。

**AI-Infra 实例**

```python title='AI-Infra 实例：vLLM 的 ChatCompletionRequest'
# 结构摘录：省略 vLLM 的导入、基类定义与路由依赖，不是独立脚本
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
    # FastAPI 已按请求模型完成输入校验
    ...
```

> [!NOTE] 与 Java 对照：请求体、校验与文档
> Spring MVC 常用 `@RequestBody` 解析 JSON、`@Valid` 触发校验，再由 springdoc 生成文档。FastAPI 集中读取函数签名和 Pydantic 模型来组织这些步骤，框架仍需注册路由，并维护模型未表达的业务约束。

### 3. 序列化、反序列化与 Schema 生成

数据契约不只是"在内存里长什么样"，还包括**怎么进来、怎么出去、怎么被外部理解**。这是 Pydantic 相比 `@dataclass` 的另一个主要优势。

**model_dump 与 model_dump_json**

```python title='model_dump 与 model_dump_json'
config = InferenceConfig(model_name="llama-3", backend="cuda")

config.model_dump()
# {'model_name': 'llama-3', 'backend': 'cuda', 'max_tokens': 2048, ...}

config.model_dump_json()
# '{"model_name":"llama-3","backend":"cuda","max_tokens":2048,...}'

# 常用选项
config.model_dump(exclude={"top_p"})          # 排除字段
config.model_dump(exclude_defaults=True)      # 排除值等于默认值的字段
config.model_dump(exclude_unset=True)         # 排除未显式设置的字段
config.model_dump(mode="json")                # 把 datetime/UUID 等转成 JSON 可序列化的形式
```

`@dataclass` 也能序列化，但要自己动手，且不处理嵌套的非 JSON 原生类型：

```python title='dataclass 的 asdict + json.dumps：datetime 需要额外编码'
from dataclasses import asdict, dataclass
from datetime import datetime
import json

@dataclass
class Event:
    created_at: datetime

try:
    json.dumps(asdict(Event(datetime(2026, 1, 25))))
except TypeError as exc:
    print(type(exc).__name__)     # TypeError
```

> [!NOTE] 与 Java 对照：对象转 JSON
> `model_dump_json()` 的用途接近 Jackson 的 `ObjectMapper.writeValueAsString()`；字段选择可以用模型配置或导出参数表达。`model_dump()` 默认返回 Python 对象，可能仍含 `datetime`，不等于已经得到 JSON。

**model_json_schema 与 OpenAPI**

```python title='model_json_schema 生成的 JSON Schema'
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

注意 `Literal["cuda", "rocm", "cpu"]` 被翻译成了 JSON Schema 的 `enum`，`Field(ge=1, le=32768)` 变成了 `minimum`/`maximum`。FastAPI 把这些 Schema 组合进 OpenAPI，再提供 `/docs` 页面；自定义 Python 校验逻辑未必都能表达成 JSON Schema，不能认为文档包含了全部运行时规则。

> [!NOTE] 与 Java 对照：Schema 生成
> springdoc 可以读取 Java 类型和部分 Bean Validation 约束生成 Schema，需要额外描述时再加 Swagger 注解。两边都应检查生成结果，自定义校验规则不会自动完整翻译成跨语言契约。

**解析 YAML / JSON 配置文件**

AI-Infra 项目大量使用 YAML 配置（vLLM 的引擎参数、DeepSpeed 的并行策略、训练任务的超参）。裸读 YAML 得到的是一个 `dict[str, Any]`，类型信息全丢——这正是数据契约要解决的问题。

`model_validate()` 尝试把读出的数据解析为模型；字段不满足约束时抛出 `ValidationError`：

```python title='用 model_validate 把 YAML 读成校验过的 ServingConfig'
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
    parallel: ParallelConfig = Field(default_factory=ParallelConfig)

with open("serving.yaml") as f:
    raw = yaml.safe_load(f)          # dict[str, Any]，无类型保障

config = ServingConfig.model_validate(raw)   # 校验 + 转换 + 嵌套构建
```

对应的 `serving.yaml`：

```yaml title='对应的 serving.yaml'
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

如果 YAML 中有多余字段，默认会被忽略；想让它报错（防止配置项拼错被静默吞掉），在对应模型上加 `extra="forbid"`。下面是配置片段，字段沿用上面的定义；嵌套字段也要拒绝额外键时，`ParallelConfig` 也需要同样的配置：

```python title='extra="forbid"：多余字段直接报错'
from pydantic import ConfigDict

class ServingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ...
```

反过来，把模型写回 YAML：

```python title='把模型写回 YAML'
yaml.safe_dump(config.model_dump(mode="json"))
```

### 4. BaseSettings：配置即契约

配置是数据契约的一个特例：数据源是环境变量和 `.env` 文件，内容全是字符串，需要解析成强类型对象。`pydantic-settings` 提供的 `BaseSettings` 专门做这件事。

**基本用法与 .env**

```python title='BaseSettings：从环境变量与 .env 解析成强类型 Settings'
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

```text title='对应的 .env 文件'
DEBUG=True
PORT=9000
DATABASE_URL=postgresql://user:pass@localhost:5432/dbname
```

**优先级、前缀与 Fail-Fast**

- **大小写不敏感**（默认）：类里定义 `PORT`，环境变量写成 `port=1234` 也能识别。
- **优先级**（由高到低）：
  1. 实例化时显式传入的值（`Settings(PORT=5000)`）
  2. 操作系统环境变量（`export PORT=...`）
  3. `.env` 文件中的值
  4. secrets 目录中的文件值（配置 `secrets_dir` 后）
  5. 类中定义的默认值
- **前缀支持**：项目复杂时为防止环境变量冲突可加前缀，在 `SettingsConfigDict` 中设 `env_prefix="APP_"`，则 `APP_PORT` 映射到 `PORT`。
- **Fail-Fast**：在启动入口执行 `settings = Settings()`，必填配置缺失或无法解析的值（比如 `PORT="hello"`）会抛异常；不吞掉异常，就能在连接外部服务前停止启动。

最后这一条是配置契约最重要的性质：**把配置错误从运行期挪到启动期**。

用上面的 `Settings` 和 `.env` 走一遍：假设启动命令是 `export PORT=7000; python app.py`，代码里写的是 `Settings()`（没有显式传值）。每个字段独立地从高到低找第一个有值的来源，然后把字符串转换成声明的类型：

| 字段 | 显式传入 | 环境变量 | `.env` | 类默认值 | 最终取值 |
|---|---|---|---|---|---|
| `APP_NAME` | — | — | — | `"Awesome App"` | `"Awesome App"`（默认值） |
| `DEBUG` | — | — | `True` | `False` | `True`（`.env` 里的字符串 `"True"` 转成 `bool`） |
| `PORT` | — | `7000` | `9000` | `8000` | `7000`（环境变量盖过 `.env`，字符串转成 `int`） |
| `DATABASE_URL` | — | — | `postgresql://…` | 无（必填） | 按 `PostgresDsn` 校验通过后的 URL |

Table: 一次 Settings() 里每个字段的取值来源

本例没有配置 secrets 目录。若删掉 `.env` 里的 `DATABASE_URL`，其他来源也没有这个值，`Settings()` 就抛 `ValidationError`（`Field required`）；只要启动代码不吞掉它，就能在连接数据库前失败。以上为未启用 CLI、未自定义来源的默认顺序，来源可以通过 `settings_customise_sources()` 调整。

来源选择和类型校验是两步。把例子中的环境变量改成 `PORT=hello`，即使 `.env` 仍有合法的 `9000`，也不会退回去重试：

```mermaid
%% 图：Settings 的来源覆盖与校验失败路径
flowchart TB
    INIT["显式传值<br/>本例没有 PORT"] --> PICK["按优先级合并来源<br/>高优先级覆盖同名字段"]
    ENV["环境变量<br/>PORT = hello"] --> PICK
    DOT[".env<br/>PORT = 9000"] --> PICK
    DEFAULT["类默认值<br/>PORT = 8000"] --> PICK
    PICK --> RAW["选中的 PORT 是 hello<br/>其余候选值已被覆盖"]
    RAW --> CHECK{"能解析为 int？"}
    CHECK -- "否" --> ERROR["抛 ValidationError<br/>不回退到 .env 或默认值"]
    CHECK -- "是" --> SETTINGS["构建 Settings 对象<br/>交给服务启动代码"]
```

**多环境配置**

指定多个 `.env` 文件，右边的覆盖左边的：

```python title='多个 .env 文件：右边覆盖左边'
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

```python title='按 ENV 环境变量动态选择配置文件'
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

**对比 Spring `@ConfigurationProperties`**

两者目的相同——把松散的配置字符串映射为强类型对象——但实现差异明显：

| 特性 | Pydantic BaseSettings | Spring `@ConfigurationProperties` |
|---|---|---|
| 底层核心技术 | 类型注解运行时解析 | 反射、Setter 或构造器注入 |
| 配置来源 | 显式传值、环境变量、默认值；`.env` / secrets 需配置启用，JSON / YAML 需自定义来源 | Spring Boot 常用 `.properties`、`.yml` / `.yaml` 与环境变量 |
| 框架耦合度 | 完全独立，任何脚本都能直接实例化 | 深度绑定 Spring 容器，需为 Bean |
| 前缀映射机制 | 扁平化为主，`env_prefix="APP_"` 匹配 `APP_PORT` | 天然层级嵌套，`prefix = "app"` 匹配 `app.database.url` |
| 校验触发时机 | 实例化时立即校验 | 容器启动阶段，需配合 `@Validated` |
| 宽松绑定 | 较严格，主要靠大小写不敏感 | 极宽松，`server.port`、`server_port`、`SERVER_PORT` 都能映射 |

Table: Pydantic BaseSettings 与 Spring @ConfigurationProperties 的对比

### 5. 选型指南：dataclass vs Pydantic vs TypedDict

**三者对比**

| 特性 | TypedDict | dataclass | Pydantic |
|---|---|---|---|
| 运行时类型 | 普通 `dict` | 自定义类实例 | 自定义类实例 |
| 运行时校验 | 无 | 无（除非手写 `__post_init__`） | 普通构造默认校验；绕过路径见上文 |
| 类型转换 | 无 | 无 | 默认允许部分转换（`"1"` → `1`），可开启严格模式 |
| 类型检查 | 静态 | 静态 | 静态 + 运行时 |
| 性能开销 | 零（就是 dict） | 极低 | 有（校验成本） |
| 序列化 | 本身是 dict；值仍须能被 JSON 编码 | 需 `asdict()`，特殊类型另处理 | 内置 `model_dump_json()` |
| Schema 生成 | 无 | 无 | `model_json_schema()` |
| 适用场景 | 已经是 dict 的数据 | 内部数据传递 | 系统边界、外部输入 |

Table: TypedDict、dataclass、Pydantic 三者对比

**决策树**

```mermaid
%% 图：数据契约选型决策树：外部数据用 Pydantic，内部传递用 dataclass，只约束形状用 TypedDict
flowchart TB
    Q0{"这组数据从哪来？"}
    Q0 -->|"外部：HTTP / JSON / YAML / 环境变量"| Q1{"来自环境变量或 .env？"}
    Q1 -->|"是"| S["Pydantic BaseSettings"]
    Q1 -->|"否"| M["<b>Pydantic BaseModel</b><br/>进门时解析并校验一次"]
    Q0 -->|"内部：模块之间传递"| Q2{"需要什么性质？"}
    Q2 -->|"限制字段赋值：仍需检查字段可变性"| F["@dataclass(frozen=True) / NamedTuple"]
    Q2 -->|"可变状态"| D["@dataclass"]
    Q2 -->|"实例数量极大、内存敏感"| SL["@dataclass(slots=True)"]
    Q0 -->|"已有 dict：输入已可信或已校验"| Q3{"只想约束形状、不改运行时行为？"}
    Q3 -->|"是"| TD["TypedDict"]
    Q3 -->|"否，要校验值"| M
    classDef q fill:#fff7e0,stroke:#c98a00,color:#222
    classDef a fill:#eef4fb,stroke:#5b8dc9,color:#222
    class Q0,Q1,Q2,Q3 q
    class S,M,F,D,SL,TD a
```

**核心原则：边界校验一次，内部自由传递**

**在系统边界用 Pydantic 校验一次，内部传递 `@dataclass` 对象。**

这正是[中篇第四章](/python-type-information-distribution-and-consumption.html#四类型信息消费层下动态消费运行时如何读取类型注解)「静态与运行时的协作边界」中信任边界（Trust Boundary）模式在数据建模上的体现，对应第一章的输入分支。校验结论只覆盖当时检查的值和约束；后续修改、跨进程消息或另一套外部接口，可能形成新的边界。

例如把前面的 `InferenceConfig` 转成调度器只读的参数：

```python title='从边界模型显式提取内部所需字段'
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class SamplingOptions:
    max_tokens: int
    temperature: float

config = InferenceConfig.model_validate(
    {"model_name": "llama-3", "backend": "cuda", "max_tokens": "32"}
)
options = SamplingOptions(config.max_tokens, config.temperature)
print(options.max_tokens, options.temperature)  # 32 1.0
```

API 层用 Pydantic、引擎内部选轻量结构是常见做法，但不能据此断言某个引擎的所有配置都是 dataclass 或完全没有校验；实现还可能使用 `attrs`、`msgspec.Struct` 和显式业务检查。

反过来的两种常见错误：

- **内部到处用 Pydantic**：每次构造对象都跑一遍校验，在每 token 都要构造对象的推理热路径上会成为可观的开销；
- **边界用 dataclass**：外部脏数据长驱直入，错误在很深的调用栈里才暴露，排查成本极高。

`attrs` 同样是成熟选择，支持转换器、校验器等能力；已有项目依赖它时，不必为了统一工具而迁移。选型要看现有模型能力和热路径的实际成本。

## 三、附录：选型决策树与 Java 数据契约对照
### 1. 类型工具选择决策树

| 问题 | 条件 | 选用 |
|---|---|---|
| 需要定义数据结构？ | 数据来自外部（API/JSON/YAML/用户输入） | Pydantic `BaseModel` |
| ^^ | 数据来自环境变量 / `.env` | Pydantic `BaseSettings` |
| ^^ | 数据本身就是 dict，只想约束形状 | `TypedDict` |
| ^^ | 内部传递，需要限制字段重新赋值 | `@dataclass(frozen=True)` / `NamedTuple`（浅层限制） |
| ^^ | 内部传递，实例数量极大 | `@dataclass(slots=True)` |
| ^^ | 内部传递，其他 | `@dataclass` |
| 需要定义接口？ | 你控制实现类的代码 | ABC（抽象基类） |
| ^^ | 你不控制实现类的代码 | `Protocol` |
| 需要约束参数值？ | 几个固定字符串 | `Literal` |
| ^^ | 枚举类型 + 穷尽检查 | `Enum` + `assert_never` |
| ^^ | 数值范围 / 格式 | `Annotated` + Pydantic `Field` |
| 需要泛型？ | 简单的"输入输出类型一致" | `TypeVar` |
| ^^ | 自定义泛型容器 | `Generic[T]` |
| ^^ | 装饰器保留签名 | `ParamSpec` |
| 函数有多种调用方式？ | — | `@overload` |
| 需要运行时类型检查？ | 数据建模 + 校验 | Pydantic |
| ^^ | 函数级防御 | beartype |
| ^^ | 简单分支判断 | `isinstance` |

Table: 类型工具选择决策树

### 2. Java 与 Python 数据契约对照

| 场景 | Java | Python |
|---|---|---|
| 限制字段重新赋值的载体 | `record` | `@dataclass(frozen=True)`（两者都非深度不可变） |
| 减模板代码 | Lombok `@Data` / `@Value` / `@Builder` | `@dataclass` |
| 运行时字段校验 | Bean Validation + `@Valid`（需显式触发） | Pydantic `BaseModel`（默认触发） |
| JSON 序列化 / 反序列化 | Jackson `ObjectMapper` | Pydantic `model_dump_json()` / `model_validate_json()` |
| API Schema 生成 | springdoc 读取类型与约束 | `model_json_schema()`；FastAPI 集成到 OpenAPI |
| 配置绑定 | Spring `@ConfigurationProperties` | Pydantic `BaseSettings` |
| 字典类型约束 | `Map<K,V>` + DTO | `TypedDict` |
| 轻量返回值 | `record` / 匿名类 | `NamedTuple` |
| 可替换接口 | `interface` | `Protocol` |
| 枚举 | `enum` | `enum.Enum` / `Literal` |
| 实现机制 | 编译期改 AST（Lombok）/ 运行时反射（Validator） | 运行时 `exec` 代码生成（dataclass）/ 元类 + Rust（Pydantic） |

Table: Java 与 Python 数据契约对照

这张表是跨主题速查。配置绑定使用独立包 `pydantic-settings`；接口、枚举等也不是 Pydantic 的专属能力，各工具的触发条件仍按正文对应小节理解。

## 四、本文小结

回头看这三篇，Python 的类型系统其实是一条链路：**注解把类型意图写下来，存根和 `py.typed` 把它分发出去，mypy 和 Pydantic 在两端各自消费它，最后落到数据契约上变成可执行的约束。**

Java 的编译期类型检查由语言工具链完成，JSON 解析和 Bean Validation 仍需各自的运行时框架；Python 的注解本身不强制检查，由静态工具和 Pydantic 等库分别消费。Pydantic v2 构建校验 Schema 并交给 `pydantic-core` 执行，这仍然是库提供的约束，不是把注解变成了 Python 语言的强制类型系统。

对 AI-Infra 工程来说，实践上最值得记住的就三条：

1. **注解要写**，哪怕暂时不上 mypy——它首先是给人读的；
2. **边界要校验**，用 Pydantic 挡住所有外部输入，让错误在启动时或入口处暴露；
3. **热路径要干净**，内部传递用 `@dataclass`，不要在每 token 的循环里反复做运行时检查。

再遇到 AI Infra 源码中的类型注解，就不会觉得是天书了。关键不是一次记住所有工具，而是理解每个工具解决的问题——在真实代码中遇到时能查到、能读懂、能用对。

## 五、自测

1. `@dataclass` 与 Pydantic `BaseModel` 各该用在什么位置？为什么？

   <details markdown="1"><summary>答案</summary>

   `dataclass` 适合内部轻量数据传递，默认不做字段类型校验，但对象构造和访问仍有成本；Pydantic 适合系统边界（配置、请求、外部输入），提供解析、校验和结构化错误。是否转换内部结构应看职责和测量结果，避免在每 token 的循环里无谓地重复校验。

   </details>

2. 为什么“注解要写，哪怕暂时不上 mypy”？

   <details markdown="1"><summary>答案</summary>

   注解首先是给人读的接口文档，其次是让 IDE 能补全与跳转，再次是让将来上 mypy 时不必回头补；而边界上的 Pydantic 校验直接依赖注解生成。不写注解的代价在读别人代码时才显现。

   </details>

3. `Settings` 里声明 `PORT: int = 8000`，`.env` 写着 `PORT=9000`，启动前又 `export PORT=hello`。`Settings()` 的结果是什么？

   <details markdown="1"><summary>答案</summary>

   抛 `ValidationError`；启动代码未捕获时进程退出。环境变量的优先级高于 `.env`，所以 `PORT` 取到的是字符串 `"hello"`；它无法转换成 `int`，校验失败。`.env` 里的 `9000` 和默认值 `8000` 不会被当作"后备"再试一次——取值和校验是两步，先定来源，再校验。

   </details>

[^q0]: 原则是**边界要校验，内部要干净**：外部数据进门（HTTP 请求、配置文件、环境变量、消息）用 Pydantic 解析并校验，内部按需改用 `@dataclass`（或 `attrs`）传递，靠 mypy 检查类型使用。`TypedDict` 适合给已有且可信的 `dict` 加静态形状约束，不执行运行时校验。校验结论依赖对象之后未被非法修改；新的外部入口仍要重新建立信任。序列化由契约类的 `model_dump` / `asdict` 派生，配置由 `BaseSettings` 读取并校验，避免另写一套字段清单。
