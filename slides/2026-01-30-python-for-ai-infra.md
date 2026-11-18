---
layout: slides
title: "Python 在 AI-Infra：从语言机制到生产交付"
subtitle: "系列精华 · 七篇正文每篇一页，按 ↓ 看机制、数字与误区"
permalink: /slides/python-for-ai-infra.html
series: python-for-ai-infra
date: 2026-01-30 23:30:00 +0800
author: arganzheng
description: "《Python 在 AI-Infra》系列的分享用幻灯片：import 是运行时动作、属性查找算法、类型注解由谁消费、GIL 与线程 / 进程 / asyncio 的选择、插件系统的侵入性阶梯、三块内存的边界、按症状选调试工具、锁文件与 GPU 服务的 worker 数。"
theme: white
transition: slide
---

## 这个系列的一句话主张

> Python 的**语法**很容易，Python **工程**不容易——难的不是怎么写，而是语法背后**哪些机制在起作用、代价是什么、什么时候不该用**。

| 链 | 篇 |
|---|---|
| 代码怎么运转 | 01–06 语言机制与运行时 |
| 怎么写得健壮 | 07–09 类型系统与数据契约 |
| 怎么协作 | 10 并发、异步与任务协作 |
| 怎么扩展 | 11 动态机制与插件 |
| 内存怎么优化 | 12 内存管理 |
| 怎么确认没问题 | 13 测试、定位与调试 |
| 怎么交付 | 14 工程化与生产交付 |

参照系是 Java；例子取自 PyTorch、vLLM、FastAPI、Pydantic。

<aside class="notes" markdown="1">
总纲：/python-for-ai-infra.html。链有依赖：不理解 import 的副作用就看不懂插件注册（一 → 四）；不理解注解如何被消费就说不清 Pydantic（二 → 七）；不理解 GIL 就定不了 worker 数（三 → 七）。
</aside>

---

## 七个部分怎么连起来

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 190}}}%%
flowchart TB
    P1["01–06 语言机制"] --> P2["07–09 类型系统"]
    P1 --> P3["10 并发与异步"]
    P1 --> P4["11 动态机制"]
    P1 --> P5["12 内存管理"]
    P3 & P4 & P5 --> P6["13 测试与调试"]
    P2 --> P7["14 工程化交付"]
    P3 --> P7
    P6 --> P7
```

---

## 01 · 执行模型：先整模块编译，再在帧里执行

**没被调用的函数里有语法错误，整个模块也导不进来**

| 形态 | 什么时候产生 | 里面有什么 |
|---|---|---|
| code object | 编译时：脚本启动或第一次导入，整个模块一起 | 字节码、常量表、名字表 |
| 函数对象 | 执行到 `def` 时 | code + `__globals__` + 默认值 + cell |
| 帧 | 每次调用时 | 局部变量、指令位置、`f_back` |

<aside class="notes" markdown="1">
原文 /python-execution-model-code-objects-functions-and-frames.html。
结论原文：源码在脚本启动或第一次导入时，以整个模块为单位编译成 code object；执行到 `def` 时创建函数对象；每次调用创建一个帧。
补充要点：`import` 的模块写 `.pyc`，直接运行的脚本不写；函数里的全局名称在定义它的模块里查找（`__globals__`）；traceback 就是沿 `f_back` 串起来的帧链；名称是绑定到对象的标签，`is` 只用于单例。
</aside>

---

## 02 · 作用域与闭包：名称去哪找，编译期就定了

**闭包 = 函数对象 + cell；cell 是共享的，调用时才读**

| 写法 | 结果 | 原因 |
|---|---|---|
| 函数里 `x = x + 1`，`x` 是全局变量 | `UnboundLocalError` | 赋值让 `x` 在整个函数里都是局部变量 |
| 循环里 `lambda: i` | 全返回最后一个值 | 几个函数共享同一个 cell |
| `lambda i=i: i` | 0、1、2 | 默认值在创建函数时求值 |
| `outer()` 返回后调用 `inner` | 还能读外层变量 | cell 被 `__closure__` 引用着 |

<aside class="notes" markdown="1">
原文 /python-scopes-and-closures.html。
结论原文：名称属于哪个作用域在编译期按「本函数里有没有绑定」决定，运行时只是按位置取值；闭包 = 函数对象 + cell。
补充要点：要改外层的名称用 `global` / `nonlocal`；每调用一次外层函数就新建一个 cell，两次得到的闭包互不影响。
</aside>

---

## 03 · 模块与导入：`import` 是运行时动作

**先登记进 `sys.modules`，再执行顶层代码**

| 步骤 | 做什么 |
|---|---|
| 1. 查缓存 | `sys.modules` 里有就直接用 |
| 2. 查找 | 依次问 `sys.meta_path` 里的 finder，得到 `ModuleSpec` |
| 3. 登记 | 模块对象先放进 `sys.modules` |
| 4. 执行 | loader 跑顶层代码：注册、CUDA 初始化都在这一步 |

<aside class="notes" markdown="1">
原文 /python-modules-and-the-import-system.html。
结论原文：`import` 是**运行时动作**——执行模块顶层代码，注册与 CUDA 初始化都在这一步。
补充要点：模块在顶层代码执行完之前就已在 `sys.modules` 里（循环导入拿到半成品就在这个窗口）；顶层代码抛异常时模块会被删掉，下次 `import` 重新执行。`.py`、`.so`、zip 里的文件只是不同的 loader，`torch._C` 由 `ExtensionFileLoader` 载入。
</aside>

<!-- v -->

## 导入的四条规则

| 对象 | 规则 |
|---|---|
| `sys.path[0]` | 由启动方式决定：`python script.py` 是脚本目录，`python -m` 是当前目录 |
| `sys.modules` | 每进程一份——spawn 的子进程要重新导入 |
| 同一文件两个名字 | 执行两次，注册表重复、`isinstance` 为 `False` |
| editable 安装 | 不复制文件，只改查找路径 |

---

## 04 · 异常：沿帧链向上，途经每个 `finally`

**日志不是处理：记录后用裸 `raise` 重抛**

| 要做的决定 | 做法 |
|---|---|
| 捕获什么 | `Exception`；`BaseException` 会吞掉 Ctrl+C 和 `sys.exit()` |
| 在哪一层翻译 | 模块或服务的公共接口处 |
| 怎么保留根因 | `raise X from exc`，记在 `__cause__` |
| 记录后怎么抛 | 裸 `raise`；`raise exc` 多一层 traceback |

<aside class="notes" markdown="1">
原文 /python-exceptions-and-failure-propagation.html。
结论原文：异常沿帧链 `f_back` 向上传播，每离开一帧先执行这一帧里活动的 `finally` 和 `__exit__`；只记录不重抛是反模式。
补充要点：`__exit__` 返回真值会抑制异常；只记录不重抛时 Worker 带着不一致的状态继续跑，分布式训练里表现为某个 rank 悄悄掉队。
</aside>

---

## 05 · 对象模型与协议：`obj.attr` 是一个固定算法

**直接调 `forward` 绕过全部 hook**

| 机制 | 规则 |
|---|---|
| 属性查找 | 数据描述符 → 实例字典 → 非数据描述符 → `__getattr__` |
| 特殊方法 | 在**类型**上查找，跳过实例字典 |
| `super()` | MRO 里的**下一个**，不是父类；漏一层之后全被跳过 |
| `nn.Module` | `__setattr__` 把子模块写进 `_modules`，`__getattr__` 取回 |

<aside class="notes" markdown="1">
原文 /python-object-model-and-protocols.html。
结论原文：`obj.attr` 是一个固定算法；`model(x)` 查类型上的 `__call__`——**直接调 `forward` 绕过全部 hook**。
补充要点：方法、`classmethod`、`staticmethod`、`property` 只是描述符 `__get__` 返回值不同；只定义 `__eq__` 会让 `__hash__` 变成 `None`。
</aside>

---

## 06 · 装饰器、生成器、`with`：前五篇机制的组合

**装饰器在定义时执行一次；生成器是挂起的帧**

| 写法 | 什么时候执行 | 容易错的地方 |
|---|---|---|
| 装饰器 | 定义时一次，等于 `f = d(f)` | 模块没被导入，注册就不发生 |
| 生成器 | `next()` 时才跑函数体 | `break` 不等于 `close()` |
| `with` | 进入、退出各一次，异常时也执行 | `__exit__` 返回真值会吞掉异常 |

<aside class="notes" markdown="1">
原文 /python-decorators-generators-and-context-managers.html。
结论原文：装饰器在定义时执行一次、返回替代对象，靠闭包记参数、靠描述符协议对方法透明；生成器是被挂起而非销毁的帧；`with` 展开为 `__enter__` / `__exit__`。
补充要点：用类实现的装饰器用在方法上要补 `__get__`；需要确定性清理的生成器用 `finally` 里 `close()` 或 `contextlib.closing`。原文最后一章用一个推理组件把导入、创建、调用、流式、异常五个阶段完整追踪了一遍。
</aside>

---

## 07–09 · 类型系统：注解由谁消费

**注解只是 `__annotations__`；Pydantic 在类创建时构建验证树**

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 210}}}%%
flowchart LR
    A["def f(x: int) -> str<br/>class User: id: int"] --> ANN["__annotations__<br/>解释器只存、不看"]
    ANN --> S["静态消费：mypy / pyright<br/>不运行代码"]
    ANN --> D["@dataclass<br/>只读字段名与顺序，exec 生成 __init__"]
    ANN --> P["Pydantic / FastAPI / beartype<br/>get_type_hints() 读出来，运行时校验"]
    classDef hot fill:#fde68a,stroke:#b45309
    class P hot
```

<aside class="notes" markdown="1">
原文三篇：/python-type-expression-and-the-typing-toolbox.html、/python-type-information-distribution-and-consumption.html、/python-data-contract-design-dataclass-pydantic-and-settings.html。
结论原文：**解释器只把注解存进 `__annotations__`**，谁读、怎么用由消费者决定——`@dataclass` 只当字段清单，**Pydantic 在类创建时构建验证树**。
</aside>

<!-- v -->

## 要点

**`get_type_hints()` 比 `__annotations__` 多做三件事**

| 要点 | 细节 |
|---|---|
| 版本：`Protocol` 3.8、`X &#124; Y` 3.10、`ParamSpec` 3.10、`Self` 3.11 | `get_type_hints()` 比 `__annotations__` 多做三件事（解析字符串、合并 MRO、去 Optional） |
| `@runtime_checkable` 只查方法存在，不查签名 | |

<!-- v -->


### 常见误区

- 「过了 mypy 就是被检查过了」——默认不检查没标注的函数体；`check_untyped_defs` 或 `strict`

---

## 10 · 并发：按瓶颈选模型，不按 API 流行度

**线程管 I/O、进程绕 GIL、asyncio 管大量协作**

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 200}}}%%
flowchart TB
    Q{"瓶颈？"} -- "等 I/O" --> IO{"并发数？"}
    IO -- "几十个" --> TH["<b>线程</b>"]
    IO -- "几千连接" --> AS["<b>asyncio</b>"]
    Q -- "纯 Python 计算" --> PR["<b>进程</b>"]
    Q -- "矩阵 / 卷积" --> NAT["不用并发<br/>原生库已多线程"]
    TH & AS & PR --> C["组合：asyncio 收 → 线程跑阻塞 → 进程预处理"]
```

<aside class="notes" markdown="1">
原文 /python-concurrency-asynchrony-and-task-collaboration.html。
线程：GIL 等待时释放；asyncio：事件循环；进程：每进程一把 GIL。
结论原文：**线程管阻塞 I/O**，**进程绕开 GIL**，**asyncio 管大量 I/O 协作**；背压、超时、取消、批处理是过载时的稳定手段。
</aside>

<!-- v -->

### 并发 vs 并行

![并发与并行的时间线示意](/img/in-post/python-concurrency-asynchrony-and-task-collaboration-concurrency-vs-parallelism.svg){: style="max-height: 380px"}

**并发是交错推进、并行是同时执行；GIL 让 Python 的并行只在多进程**

<!-- v -->

### 数字与陷阱

| 量 | 数 |
|---|---|
| 线程池默认大小 | $$\min(32, \text{cpu} + 4)$$ |
| OS 线程栈 vs 协程 | 约 8 MB vs 几 KB |
| `fork` | 只复制调用线程；3.14 起 Linux 默认 `forkserver` |
| `/dev/shm` | 默认 64 MB——DataLoader worker 常撞 |
| FastAPI `def` 端点 | 线程池 40 |

<aside class="notes" markdown="1">
补充要点：队列默认无界；`asyncio.Lock` 不可重入；`CancelledError` 是 `BaseException`。
</aside>

<!-- v -->

### 常见误区

- 「`Future.cancel()` 能停掉运行中的线程」——线程不能取消、进程只能 kill、协程在下一个 `await` 点协作式取消
- 「在协程里调 `requests.get()` 没问题」——事件循环整个停摆；用异步客户端或 `asyncio.to_thread()`

---

## 11 · 动态机制与插件：只在启动时「选择」，热路径必须静态

**插件化 = 注册表 + 发现 + 契约 + 边界**

| 操作 | 耗时 | 相对 |
|---|---|---|
| `obj.attr` | ≈ 10 ns | 1× |
| `getattr(obj, name)` | ≈ 23 ns | 2.3× |
| `inspect.signature` | ≈ 3700 ns | **约 370×** |

<aside class="notes" markdown="1">
原文 /python-reflection-metaprogramming-and-plugin-architecture.html。
结论原文：反射观察、元编程改造、动态加载导入；**插件化 = 注册表 + 发现 + 契约 + 边界**；侵入性递增：显式注册 → 装饰器 → 描述符 → `__init_subclass__` → 元类——**元类几乎总是最后的选择**。
</aside>

<!-- v -->

## 要点

**三种发现机制：越灵活失败定位越难**

| 要点 | 细节 |
|---|---|
| 注册表为空 ⇔ 模块未被导入 | 《模块与导入系统》的 import 副作用在这里兑现 |
| 用户输入能**选**名字、不能**造**名字（`getattr(module, user_input)` 是漏洞） | |

<aside class="notes" markdown="1">
补充要点：三种发现机制——显式导入（默认）→ 包扫描（多到难维护）→ 入口点（要让别人扩展）。
</aside>
---

## 12 · 内存管理：多数不是「泄漏」，是「被意外长期持有」

**对象释放 ≠ RSS 下降——arena 要等整个 pool 空**

| 机制 | 数 |
|---|---|
| 引用计数 | 归零立即释放；循环靠分代 GC，阈值 (700, 10, 10) |
| pymalloc | 管 ≤ 512 字节；arena 256 KB / pool 4 KB |
| 小整数缓存 | −5 到 256 |
| torch | `memory_reserved` ≥ `memory_allocated`；差值是缓存分配器 |

<aside class="notes" markdown="1">
原文 /python-memory-management-and-optimization.html。
结论原文：三块内存——Python 对象、原生缓冲区（NumPy / torch CPU）、设备显存——边界不同、归还方式不同；**对象释放不等于 RSS 下降**（pymalloc 的 arena 只在全部 pool 释放后才归还 OS）。
</aside>

<!-- v -->

## 要点

**先分三类再找源——释放 ≠ RSS 下降**

| 要点 | 细节 |
|---|---|
| 定位：**先分三类**（仍被引用 / 分配器保留 / 原生或设备持有）再找源 | `tracemalloc`、`gc.get_referrers`、`torch.cuda.memory_summary` |

<!-- v -->


### 常见误区

- 「用 `__del__` 释放 GPU 句柄」——时机不可控、可能在解释器关闭时才调；用 `with` 或 `weakref.finalize`

---

## 13 · 测试、定位与调试：按症状选工具

**难在按症状选工具；`raise from` 与上下文日志是前提**

| 症状 | 工具 |
|---|---|
| 协程没被等待 | `pytest -W error::RuntimeWarning`；`Mock` 不能 `await`，用 `AsyncMock` |
| Mock 没生效 | 替换「被测模块里实际用的名字」，不是定义处 |
| 进程卡死 / 段错误 | `python -X faulthandler`、`py-spy dump` |
| 内存增长 | `PYTHONTRACEMALLOC=25` 记录分配栈 |
| 动态调用看不到 | `inspect`、`sys.settrace`、结构化日志 |

<!-- v -->

### 常见误区

- 「在库里 `logging.basicConfig()`」——它配置 root logger，篡改了应用的全局配置；库只 `getLogger(__name__)`，最多加 `NullHandler`

<aside class="notes" markdown="1">
原文 /python-unit-testing-troubleshooting-and-debugging.html。
结论原文：工具不难，难在**按症状选工具**；`raise ... from` 与带上下文的日志是所有工具的前提；日志分调试期与生产期，logger 与 handler 两道级别关卡。
</aside>

---

## 14 · 工程化与生产交付：Java 里框架强制的事交还给你

**锁文件进 CI；GPU 服务每 GPU 一个 worker**

| 量 | 数 |
|---|---|
| `pip install torch` | 2 GB+；环境 5–8 GB；镜像 8–12 GB |
| 装 torch | `--index-url`，**不用** `--extra-index-url`（pip 在多索引里选最高版本，PyPI 一有更新的上游版本就装成 PyPI 默认构建） |
| CUDA 三层 | 驱动 / runtime / 库；同大版本向前兼容 |
| 4 个 worker × 14 GB 模型 | 56 GB 显存——「多开 worker 提吞吐」的代价 |
| tag | 不能是 `latest` |

<aside class="notes" markdown="1">
原文 /python-engineering-and-production-delivery.html。
结论原文：**锁文件进 CI**（`uv sync --locked`）、**torch 交给固定 tag 的基础镜像**、静态检查是编译器的替代品；**GPU 服务每 GPU 一个 worker**，进程内靠 asyncio + 批处理。
补充要点：镜像分层：基础镜像（CUDA + torch）→ 依赖层 → 代码层，改代码不重装 torch
</aside>

<!-- v -->


### 常见误区

- 「`requirements.txt` 全写 `==` 就可复现」——传递依赖仍浮动、无 hash；锁文件带 hash，`--frozen` 不做一致性检查

---

## 一条贯穿线：import → 注册表 → GIL → worker 数

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 220}}}%%
flowchart TB
    A["import 是运行时动作（03）"] --> B["顶层副作用：注册 · CUDA 初始化（03）"]
    B --> C["注册表为空 ⇔ 模块未导入（11）"]
    C --> D["发现：显式导入 / 包扫描 / 入口点（11）"]
    A --> H["spawn 重导主模块 → __main__（10）"]
    E["GIL：同一刻一个线程跑字节码（10）"] --> F["多进程是唯一绕开（10）"]
    F --> H
    F --> G["每进程各自 sys.modules（10、12）"]
    G --> I["每 GPU 一个 worker（14）"]
```

---

## 常见误区

- 「`import` 只是声明依赖」——它执行顶层代码
- 「`model.forward(x)` 与 `model(x)` 一样」——直接调 forward 绕过全部 hook
- 「`super()` 调父类」——调的是 MRO 里的下一个
- 「类型注解在运行时生效」——只有消费者主动读取时才生效
- 「`@dataclass` 会按注解校验」——`User(id="abc")` 静默通过
- 「多线程能加速 CPU 计算」——GIL；纯 Python CPU 用进程
- 「子类自动注册应该用元类」——`__init_subclass__` 几乎总是更好
- 「RSS 不降就是泄漏」——先分三类
- 「装 torch 用 `--extra-index-url`」——用 `--index-url`
- 「GPU 服务多开几个 worker 提吞吐」——每个 worker 一份模型进显存
{: .fragments}

---

## 七个出口

| 篇 | 一个判据 / 一个数 |
|---|---|
| 01–06 | 属性查找四步；`super()` = MRO 下一个；`sys.modules` 每进程一份 |
| 07–09 | 解释器只存 `__annotations__`；边界校验一次、内部零开销 |
| 10 | 线程池 min(32, cpu + 4)；线程 8 MB / 协程几 KB；`/dev/shm` 64 MB |
| 11 | `inspect.signature` 370×；能选名字不能造名字 |
| 12 | GC 阈值 (700, 10, 10)；pymalloc ≤ 512 B、arena 256 KB |
| 13 | `AsyncMock`；`-X faulthandler`；`PYTHONTRACEMALLOC=25` |
| 14 | `uv sync --locked`；`--index-url`；每 GPU 一个 worker |

---

## 下一步

- **往下**：《C++ 在 AI-Infra》——Python 之下的那一层；《PyTorch 深度实践》——这些机制在 PyTorch 里的形态（`__setattr__` 注册、Dispatcher、DataLoader 进程）
- **往上**：《vLLM 源码》——asyncio 引擎循环、多进程 worker、插件注册表的真实案例；《AI 平台工程》——镜像与交付
- 原文总纲：`/python-for-ai-infra.html`；通关自测在系列总结
