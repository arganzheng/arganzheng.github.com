---
layout: post
series: python-for-ai-infra
title: "Python 在 AI-Infra（02 下）：类型系统——数据契约设计"
subtitle: "Type System III: Data Contract Design"
tags: [Python]
catalog: true
date: 2026-01-25 20:00:00 +0800
updated: 2026-09-21
---

前两篇沿着"类型信息如何流动"展开：[上篇](/python-type-expression-and-the-typing-toolbox.html)讲怎么写（`typing` 工具箱），[中篇](/python-type-information-distribution-and-consumption.html)讲怎么分发、谁消费（存根、mypy、运行时读注解）。本篇的组织轴切换——不再讨论类型信息本身，而是讨论**用这些能力去构建什么**：数据契约。

所谓数据契约，就是对"一组数据长什么样"的正式约定。在 AI Infra 里它无处不在：推理服务的请求体与响应体、训练任务的配置文件、模型仓库里的元数据、Worker 之间传递的消息。Java 工程师熟悉的 Bean Validation、Jackson、`@ConfigurationProperties` 在这里对应的是 `@dataclass`、Pydantic `BaseModel`、`BaseSettings`——但它们不是一一对应的翻译，而是同一份类型注解的三种消费方式，各有适用位置。

本篇围绕的核心问题是：

> **什么时候用 `@dataclass`、什么时候上 Pydantic、什么时候 `TypedDict` 就够？校验该发生在哪里，序列化与配置怎么跟着契约走？**

答案的骨架是中篇末尾那句话——**边界要校验，内部要干净**：外部数据进门时用 Pydantic 付一次校验成本，进门之后用零开销的 `@dataclass` 传递，靠 mypy 做静态保证。

## 一、总览

### 1. 在三篇地图上的位置

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

本篇讲图里最下面那格。它**使用**前两格的能力，但组织轴不同：前两格沿着"类型信息如何流动"，这一格沿着"程序的数据结构如何设计"。

### 2. 本文的章节安排

| 章 | 主题 | 内容 |
|---|---|---|
| 二 | 工程落地：数据契约设计 | dataclass、Pydantic、序列化与 Schema、BaseSettings、选型指南 |
| 三 | 附录 | 类型工具选择决策树；Java 与 Python 数据契约对照 |
| 四 | 本文小结 |  |
| 五 | 自测 | 2 道题 |

Table: 本文的章节安排

## 二、工程落地：数据契约设计

前两篇沿着"类型信息如何流动"展开：怎么表达（上篇）、怎么分发、谁来消费（中篇）。这一篇**组织轴切换**——不再讨论类型信息本身，而是讨论用这些能力去构建什么：**数据契约**。

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

这就是"数据契约"的价值——一处声明，三处受益。而它能成立，靠的正是[中篇第四章](/python-type-information-distribution-and-consumption.html#四类型信息消费层下动态消费运行时如何读取类型注解)讲的机制。

> **前置阅读**：如果你想先搞清楚 `@dataclass` 和 Pydantic **怎么**读到类型注解、`exec` 代码生成和元类分别在什么时机介入，见[中篇第四章](/python-type-information-distribution-and-consumption.html#四类型信息消费层下动态消费运行时如何读取类型注解)「类型信息消费层（下）：动态消费」的"框架如何消费注解"一节。本章只讲用法与取舍。

### 1. dataclass：标准库的数据类

**从原生 `__init__` 到 `@dataclass`**

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

**`__post_init__` 手工校验及其局限**

`@dataclass` **不做运行时校验**——这一点在[中篇第四章](/python-type-information-distribution-and-consumption.html#四类型信息消费层下动态消费运行时如何读取类型注解)已经说明原因：它只把注解当字段清单，不理解注解的语义。所以下面这行不会报错：

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

**`frozen` 与 `slots`**

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

**BaseModel 基础与类型强制转换**

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

注意两点：一是 `EmailStr` 这类语义类型开箱即用；二是**两个错误一次性全部报出来**，而不是遇到第一个就中断——这正是[中篇第四章](/python-type-information-distribution-and-consumption.html#四类型信息消费层下动态消费运行时如何读取类型注解)提到的"收集所有错误路径"。

**Field：默认值与约束**

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

约束也可以写在 `Annotated` 里，这是 Pydantic v2 更推荐的形式，因为它让类型和元数据分离得更干净（见[上篇第二章](/python-type-expression-and-the-typing-toolbox.html#二类型表达从基础注解到-typing-工具箱)「类型信息提供层（上）：类型表达」的 `Annotated` 一节）：

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

**ValidationError 与错误聚合**

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

**AI-Infra 实例**

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

**model_dump 与 model_dump_json**

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

**model_json_schema 与 OpenAPI**

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

**解析 YAML / JSON 配置文件**

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

**基本用法与 .env**

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

**优先级、前缀与 Fail-Fast**

- **大小写不敏感**（默认）：类里定义 `PORT`，环境变量写成 `port=1234` 也能识别。
- **优先级**（由高到低）：
  1. 实例化时显式传入的值（`Settings(PORT=5000)`）
  2. 操作系统环境变量（`export PORT=...`）
  3. `.env` 文件中的值
  4. 类中定义的默认值
- **前缀支持**：项目复杂时为防止环境变量冲突可加前缀，在 `SettingsConfigDict` 中设 `env_prefix="APP_"`，则 `APP_PORT` 映射到 `PORT`。
- **Fail-Fast**：执行 `settings = Settings()` 的那一瞬间，任何必填配置缺失或类型错误（比如 `PORT` 被配成 `"hello"`）都会立刻抛异常并阻止程序启动。

最后这一条是配置契约最重要的性质：**把配置错误从运行期挪到启动期**。

**多环境配置**

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

**对比 Spring `@ConfigurationProperties`**

两者目的相同——把松散的配置字符串映射为强类型对象——但实现差异明显：

| 特性 | Pydantic BaseSettings | Spring `@ConfigurationProperties` |
|---|---|---|
| 底层核心技术 | 类型注解运行时解析 | 反射、Setter 或构造器注入 |
| 默认支持格式 | `.env`、系统环境变量、JSON / YAML（需插件） | `.properties`、`.yml` / `.yaml` |
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
| 运行时校验 | 无 | 无（除非手写 `__post_init__`） | **自动校验** |
| 类型转换 | 无 | 无 | **自动转换**（`"1"` → `1`） |
| 类型检查 | 静态 | 静态 | 静态 + 运行时 |
| 性能开销 | 零（就是 dict） | 极低 | 有（校验成本） |
| 序列化 | 天然是 dict，直接 `json.dumps` | 需 `asdict()`，不处理特殊类型 | 内置 `model_dump_json()` |
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
    Q1 -->|"否"| M["Pydantic BaseModel<br/>进门时解析并校验一次"]
    Q0 -->|"内部：模块之间传递"| Q2{"需要什么性质？"}
    Q2 -->|"不可变：可做 dict key、跨线程共享"| F["@dataclass(frozen=True) / NamedTuple"]
    Q2 -->|"可变状态"| D["@dataclass"]
    Q2 -->|"实例数量极大、内存敏感"| SL["@dataclass(slots=True)"]
    Q0 -->|"数据本身就是 dict：第三方 API 返回值、已解析的 JSON"| Q3{"只想约束形状、不改运行时行为？"}
    Q3 -->|"是"| TD["TypedDict"]
    Q3 -->|"否，要校验值"| M
    classDef q fill:#fff7e0,stroke:#c98a00,color:#222
    classDef a fill:#eef4fb,stroke:#5b8dc9,color:#222
    class Q0,Q1,Q2,Q3 q
    class S,M,F,D,SL,TD a
```

**核心原则：边界校验一次，内部自由传递**

**在系统边界用 Pydantic 校验一次，内部传递 `@dataclass` 对象。**

这正是[中篇第四章](/python-type-information-distribution-and-consumption.html#四类型信息消费层下动态消费运行时如何读取类型注解)「静态与运行时的协作边界」中信任边界（Trust Boundary）模式在数据建模上的体现：外部数据不可信，进门时付一次校验成本；进门之后数据已经可信，用零开销的 `@dataclass` 传递，靠 mypy 做静态保障。

vLLM 的源码就是这个模式：API 层（`entrypoints/openai/protocol.py`）用 Pydantic 定义请求体，引擎内部（`SamplingParams`、`SchedulerConfig`）一律用 `@dataclass`。热路径上不做重复校验。

反过来的两种常见错误：

- **内部到处用 Pydantic**：每次构造对象都跑一遍校验，在每 token 都要构造对象的推理热路径上会成为可观的开销；
- **边界用 dataclass**：外部脏数据长驱直入，错误在很深的调用栈里才暴露，排查成本极高。

> 顺带一提，`attrs` 是比 `@dataclass` 更早、功能更全的第三方库，但在 AI-Infra 生态中已基本被"标准库 `@dataclass` + Pydantic"的组合取代，新项目一般不需要引入。

## 三、附录：选型决策树与 Java 数据契约对照
### 1. 类型工具选择决策树

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

### 2. Java 与 Python 数据契约对照

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

Table: Java 与 Python 数据契约对照

核心差异：Java 用**多个独立框架**拼出完整的数据契约能力，每个框架各管一段；Python 用 **Pydantic 一个库**覆盖了校验、转换、序列化、Schema 生成、配置绑定的全部环节。

## 四、本文小结

回头看这三篇，Python 的类型系统其实是一条链路：**注解把类型意图写下来，存根和 `py.typed` 把它分发出去，mypy 和 Pydantic 在两端各自消费它，最后落到数据契约上变成可执行的约束。**

与 Java 的最大差异不在于语法，而在于这条链路是**拆开的**。Java 把声明、检查、载体、反射合为一体，你没得选；Python 把每一环都做成可插拔的组件，你可以只写注解不做检查，也可以只在边界上做运行时校验而内部完全不管。这种自由度是代价也是优势——代价是需要自己决定在哪里投入，优势是可以按项目实际情况精确控制。

无论是 Java 靠注解切面实现的 Bean Validation，还是 Python 用元类与 Rust 引擎构建的 Pydantic，本质都是把开发者从繁琐的"防错代码"中解放出来。理解了 `__annotations__` 与元类之后会发现，Pydantic 并不是什么不可知的魔法——它只是充分利用了 Python 的动态性，把校验逻辑下沉到了语言机制层面。

对 AI-Infra 工程来说，实践上最值得记住的就三条：

1. **注解要写**，哪怕暂时不上 mypy——它首先是给人读的；
2. **边界要校验**，用 Pydantic 挡住所有外部输入，让错误在启动时或入口处暴露；
3. **热路径要干净**，内部传递用 `@dataclass`，不要在每 token 的循环里反复做运行时检查。

再遇到 AI Infra 源码中的类型注解，就不会觉得是天书了。关键不是一次记住所有工具，而是理解每个工具解决的问题——在真实代码中遇到时能查到、能读懂、能用对。

## 五、自测

1. `@dataclass` 与 Pydantic `BaseModel` 各该用在什么位置？为什么？

   <details markdown="1"><summary>答案</summary>

   `dataclass` 用于内部热路径：零运行时校验、开销只是属性访问；Pydantic 用于系统边界（配置、请求、外部输入）：解析 + 校验 + 错误信息，让错误在启动时或入口处暴露。每 token 的循环里做 Pydantic 校验是常见的性能事故。

   </details>

2. 为什么“注解要写，哪怕暂时不上 mypy”？

   <details markdown="1"><summary>答案</summary>

   注解首先是给人读的接口文档，其次是让 IDE 能补全与跳转，再次是让将来上 mypy 时不必回头补；而边界上的 Pydantic 校验直接依赖注解生成。不写注解的代价在读别人代码时才显现。

   </details>

## 下一篇

[并发、异步与任务协作](/python-concurrency-asynchrony-and-task-collaboration.html)

