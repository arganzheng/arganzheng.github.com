---
layout: post
title: Python 并发、异步与任务协作
subtitle: Python Concurrency, Asynchrony, and Task Collaboration in AI Systems
tags: [Python]
catalog: true
---

在 AI-Infra 系统中，Python 往往并不直接承担最重的数值计算。真正消耗算力的部分，通常由 CUDA、C++、通信库或专用推理引擎完成。

但这并不意味着 Python 不需要并发。

恰恰相反，模型服务、任务调度、数据预处理、资源监控和流水线编排，通常都由 Python 负责。系统的吞吐量、尾延迟和资源利用率，很大程度上取决于 Python 是否能够正确组织并发任务。

常见场景包括：

- 同时处理大量模型推理请求；
- 并发访问对象存储、数据库和远程 RPC 服务；
- 在多个模型节点之间转发流式数据；
- 调度训练任务并持续监控任务状态；
- 让 CPU 预处理、GPU 推理和结果后处理形成流水线；
- 在请求高峰期通过队列和背压保护系统资源。

因此，理解 Python 并发，不能停留在"如何创建线程"或"如何使用 `async`"的层面。更重要的是回答三个工程问题：

1. 当前瓶颈是 CPU、GPU、网络还是外部服务？
2. 任务之间应该如何协作？
3. 当下游处理速度跟不上上游输入速度时，系统如何保持稳定？

## 一、先理解并发：并发不等于并行

### 1. 并发与并行

**并发（Concurrency）**描述的是多个任务在同一时间段内交替推进。

**并行（Parallelism）**描述的是多个任务在同一时刻真正同时执行。

例如，一个事件循环可以在多个网络请求之间快速切换。虽然某个时刻只有一段 Python 代码在运行，但多个请求都能向前推进，这属于并发。

而多个进程分别运行在不同 CPU 核心上，则属于并行。

可以简单地表示为：

```text
并发：任务 A ──等待──继续
             任务 B ──等待──继续

并行：任务 A ─────────────
      任务 B ─────────────
```

对于 AI-Infra，二者经常组合使用：

```text
Python asyncio
    ├── 管理大量网络请求
    ├── 管理任务生命周期
    └── 调度后台协作任务

线程或进程
    ├── 执行阻塞操作
    └── 执行 CPU 密集型预处理

GPU / C++ Runtime
    └── 执行真正的模型计算
```

### 2. 并发模型的选择

Python 中常见的并发方式可以分为三类：

| 模型 | 适合场景 | 主要成本 | 典型问题 |
| :--- | :--- | :--- | :--- |
| 线程 | 阻塞 I/O、兼容同步库 | 线程栈、上下文切换 | 共享状态、线程安全 |
| 进程 | CPU 密集型计算 | 进程创建、序列化、内存 | 数据复制、进程通信 |
| asyncio | 大量 I/O、任务编排 | 需要异步调用链 | 事件循环被阻塞 |

没有一种模型适合所有场景。工程上的第一原则是：

> 不要根据 API 的流行程度选择并发模型，而要根据瓶颈类型选择并发模型。

## 二、GIL：Python 并发的底层约束

GIL（Global Interpreter Lock）是 CPython 解释器中的一把全局锁。它保证在任意时刻，只有一个线程可以执行 Python 字节码。

这意味着即使创建了多个线程，纯 Python 的 CPU 密集型代码在同一时刻仍然只有一个线程在运行。GIL 的存在是为了保护 CPython 内部的引用计数和内存管理机制，使得对象的创建和销毁不需要细粒度锁。

### 1. GIL 何时释放

GIL 并非始终被持有。以下操作通常会释放 GIL：

- 网络 I/O（socket read/write）；
- 文件 I/O（磁盘读写）；
- `time.sleep()`；
- C 扩展调用（NumPy、PyTorch 的底层计算）；
- 数据库驱动的等待操作。

这就是为什么线程仍然能有效加速 I/O 密集型任务：当一个线程在等待网络响应时，GIL 被释放，其他线程可以继续执行 Python 代码。

### 2. 对并发模型选择的影响

| 任务类型 | 线程能否加速 | 原因 |
| :--- | :--- | :--- |
| 网络 I/O | 能 | I/O 等待期间 GIL 释放 |
| 文件 I/O | 能 | 同上 |
| C 扩展计算 | 能 | 扩展代码可以主动释放 GIL |
| 纯 Python CPU 计算 | 不能 | 字节码执行期间 GIL 不释放 |

对于纯 Python 的 CPU 密集型任务，需要使用多进程（每个进程有独立的解释器和 GIL）或者使用 C 扩展库将计算下沉到 native 层。

### 3. Python 3.13+ free-threaded 模式

Python 3.13 引入了实验性的 free-threaded 模式（PEP 703），允许多个线程真正并行执行 Python 字节码：

```bash
# 需要专门编译的 free-threaded 版本
python3.13t script.py
```

截至目前，这一特性仍处于实验阶段：

- 许多第三方 C 扩展尚未适配；
- 部分场景下单线程性能可能略有下降；
- 生产环境建议继续使用默认的 GIL 模式。

但这是 Python 并发模型的重要演进方向。对于 Java 开发者来说，可以把 free-threaded Python 理解为 Python 正在向 Java 的真正多线程并行能力靠拢。

## 三、线程、进程与异步：如何做工程决策

### 1. 线程：适合阻塞 I/O

线程适合这样的任务：

- 调用只提供同步接口的 HTTP 客户端；
- 访问传统数据库驱动；
- 读取文件或设备；
- 调用同步 SDK；
- 等待外部服务返回。

示例：

```python
from concurrent.futures import ThreadPoolExecutor


def invoke_sync_model(request_id: str) -> str:
    # 假设这里调用的是同步模型服务客户端
    return f"result-{request_id}"


def run_batch(request_ids: list[str]) -> list[str]:
    with ThreadPoolExecutor(max_workers=16) as executor:
        return list(executor.map(invoke_sync_model, request_ids))
```

线程不会让 Python CPU 代码突破 GIL（参见第二章）。对于纯 Python 的 CPU 密集型计算，增加线程通常不能获得理想的线性加速。

但网络请求、磁盘访问和部分 C 扩展操作会释放 GIL，因此线程依然适合大量阻塞 I/O。

### 2. 进程：适合 CPU 密集型任务

如果任务主要执行 Python 层面的 CPU 计算，例如：

- 文本切分；
- 图像解码；
- 特征预处理；
- 数据清洗；
- 大量 JSON 或协议解析；

可以考虑使用进程池。

```python
from concurrent.futures import ProcessPoolExecutor


def preprocess(item: bytes) -> list[int]:
    # 示例：模拟 CPU 密集型预处理
    return [value * value for value in item]


def preprocess_batch(items: list[bytes]) -> list[list[int]]:
    with ProcessPoolExecutor() as executor:
        return list(executor.map(preprocess, items))
```

进程通过独立解释器绕开了单个解释器中的 GIL 限制，但代价是数据通常需要在进程之间序列化和传输。

对于大模型场景，尤其要注意：

- 不要无意中复制巨大的模型对象；
- 不要通过进程间消息传输大型 Tensor；
- 不要在每个子进程中重复初始化昂贵的运行时；
- 使用共享内存时，需要明确生命周期和所有权。

进程并不是"更快的线程"，而是更强隔离、也更高成本的执行单元。

### 3. asyncio：适合高并发 I/O

当系统需要同时维护大量网络连接时，异步 I/O 通常更加合适：

```python
import asyncio
import httpx


async def fetch(client: httpx.AsyncClient, url: str) -> str:
    response = await client.get(url)
    response.raise_for_status()
    return response.text


async def fetch_all(urls: list[str]) -> list[str]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = [fetch(client, url) for url in urls]
        return await asyncio.gather(*tasks)
```

这里并不是创建了与请求数量相同的线程，而是由事件循环统一管理网络连接。当某个请求等待网络响应时，事件循环可以继续运行其他任务。

但异步并不意味着所有代码都自动变快。下面的代码会阻塞整个事件循环：

```python
import asyncio
import time


async def bad_task() -> None:
    time.sleep(5)  # 错误：阻塞事件循环
    print("done")
```

应该改为：

```python
async def good_task() -> None:
    await asyncio.sleep(5)
    print("done")
```

`asyncio.sleep()` 会主动交出执行权，而 `time.sleep()` 会占用当前线程，使同一事件循环中的其他任务全部停顿。

## 四、事件循环：调度核心与阻塞陷阱

很多人把 `await` 理解为"等待结果"。更准确地说，`await` 是一次协作式调度机会。

当协程执行到：

```python
result = await client.get(url)
```

它通常会经历以下过程：

1. 发起网络请求；
2. 注册对 socket 的关注；
3. 当前协程暂时挂起；
4. 事件循环处理其他就绪任务；
5. socket 可读时恢复当前协程；
6. 协程继续执行后续代码。

事件循环底层通常依赖操作系统提供的 I/O 多路复用机制，例如：

- Linux 上的 `epoll`；
- macOS 上的 `kqueue`；
- Windows 上的 IOCP 等机制。

抽象后的流程可以表示为：

```text
事件循环
  ├── 检查已完成的定时器
  ├── 检查已完成的 Future
  ├── 轮询网络 I/O
  ├── 恢复可继续执行的协程
  └── 执行协程直到下一次 await
```

因此，异步代码有一个关键约束：

> 协程必须频繁地、主动地交出执行权。

如果某个协程长时间执行同步代码，整个事件循环都会受到影响。下面是三种常见的阻塞场景和对应的解决方式。

### 1. 把同步函数放入线程池

当必须调用同步库时，可以使用 `asyncio.to_thread()`：

```python
import asyncio


def call_blocking_sdk(payload: dict) -> dict:
    # 第三方 SDK 只有同步接口
    return {"ok": True, "payload": payload}


async def call_sdk(payload: dict) -> dict:
    return await asyncio.to_thread(call_blocking_sdk, payload)
```

在较早版本的 Python 中，也可以使用：

```python
loop = asyncio.get_running_loop()
result = await loop.run_in_executor(
    None,
    call_blocking_sdk,
    payload,
)
```

这是一种典型的 Sync/Async Bridge：异步系统负责任务协作，线程池负责承载阻塞调用。

### 2. 不要把 CPU 密集型代码放进协程

下面这种写法仍然可能阻塞事件循环所在的执行环境：

```python
async def bad_preprocess(data: bytes) -> list[int]:
    return expensive_python_computation(data)
```

更合适的方式是交给进程池：

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor


def expensive_preprocess(data: bytes) -> list[int]:
    return [x * x for x in data]


async def preprocess_async(
    data: bytes,
    pool: ProcessPoolExecutor,
) -> list[int]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        pool,
        expensive_preprocess,
        data,
    )
```

实际系统中还需要评估序列化成本。如果输入数据很大，进程池可能因为数据复制而抵消计算收益。

### 3. 识别事件循环阻塞

可以从几个方面监控事件循环：

- 统计协程执行时间；
- 记录事件循环延迟；
- 检查请求的 P99 和 P999 延迟；
- 使用调试模式检测慢回调；
- 对同步 I/O 和 CPU 计算建立代码审查规则。

开发环境可以启用：

```python
import asyncio

asyncio.run(main(), debug=True)
```

生产系统则应结合日志、指标和 tracing，定位具体是哪一个任务长时间没有让出执行权。

## 五、任务：协程、Task 与 Future

在 `asyncio` 中，需要区分几个概念。

### 协程对象

调用异步函数时，得到的是协程对象：

```python
coro = fetch_data()
```

此时函数体还没有开始执行。协程对象必须被 `await` 或提交给 `create_task()` 后才会运行。

### Task

Task 是事件循环中被调度执行的协程：

```python
task = asyncio.create_task(fetch_data())
```

创建 Task 后，事件循环可以在适当时机执行它。

### Future

Future 表示一个未来会完成的结果。它通常是底层异步操作和任务调度之间的桥梁。

大多数业务代码不需要直接创建 Future，但理解它有助于理解 `await`、Task 和底层 I/O 的关系。

### 使用 `TaskGroup` 管理任务

在需要启动多个相互关联的任务时，推荐使用结构化并发：

```python
import asyncio


async def run_pipeline(request_id: str) -> str:
    async with asyncio.TaskGroup() as group:
        encode_task = group.create_task(encode(request_id))
        metadata_task = group.create_task(load_metadata(request_id))

    encoded = encode_task.result()
    metadata = metadata_task.result()
    return await infer(encoded, metadata)
```

结构化并发强调：

- 子任务属于明确的父任务；
- 父任务结束前，需要处理子任务；
- 子任务失败时，错误不会被悄悄丢弃；
- 任务生命周期更加容易管理。

相比之下，随意创建后台任务容易造成任务泄漏：

```python
asyncio.create_task(background_job())
```

如果没有保存引用、处理异常和设计退出流程，任务可能在服务关闭时被强行终止，异常也可能无法被业务感知。

## 六、超时、取消与异常传播

在 AI 服务中，超时不是异常情况，而是正常的控制手段。

```python
import asyncio


async def infer_with_timeout(payload: dict) -> dict:
    try:
        async with asyncio.timeout(2.0):
            return await infer(payload)
    except TimeoutError:
        return {"error": "inference timeout"}
```

> **版本说明：** Python 3.11+ 中 `asyncio.timeout()` 抛出内置 `TimeoutError`。Python 3.10 及以前通常使用 `asyncio.wait_for()`，它抛出的是 `asyncio.TimeoutError`（内置 `TimeoutError` 的子类）。

超时通常意味着取消底层任务。协程应该正确响应取消：

```python
async def worker() -> None:
    try:
        while True:
            item = await queue.get()
            await process(item)
            queue.task_done()
    except asyncio.CancelledError:
        # 释放连接、关闭文件、清理临时资源
        await cleanup()
        raise
```

这里重新抛出 `CancelledError` 很重要。吞掉取消异常，可能导致任务无法正常停止，进而影响服务关闭和资源回收。

### 不要无限等待下游

错误示例：

```python
result = await remote_call()
```

在生产服务中，远程调用应具备：

- 连接超时；
- 读取超时；
- 总体超时；
- 重试上限；
- 重试退避；
- 熔断或限流机制。

异步只解决"等待期间如何调度其他任务"，并不解决远程服务本身的不可靠性。

## 七、ContextVars：在异步任务中传递上下文

AI-Infra 服务通常需要在日志、指标和链路追踪中传递：

- Trace ID；
- Request ID；
- 用户或租户 ID；
- 模型名称；
- 调度队列；
- 权限上下文。

全局变量不适合异步并发，因为多个请求会互相覆盖。

可以使用 `contextvars`：

```python
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar(
    "request_id",
    default="-",
)


async def handle_request(request_id: str) -> None:
    token = request_id_var.set(request_id)
    try:
        await call_model()
        print(f"request_id={request_id_var.get()}")
    finally:
        request_id_var.reset(token)
```

在异步任务中，ContextVar 会随着任务上下文传播，比手动给每个函数增加大量参数更加适合日志和 tracing。

但它也不应该被滥用。业务上真正重要的数据，仍然应该通过明确的函数参数传递。ContextVar 更适合横切关注点，例如：

- 日志字段；
- tracing；
- 请求范围的鉴权信息；
- 事务或会话上下文。

## 八、生产者—消费者与背压

AI-Infra 中一个非常常见的结构是：

```text
请求输入 → 预处理 → 排队 → 批处理 → GPU 推理 → 结果返回
```

如果输入速度高于推理速度，系统就会产生积压。

没有背压的实现可能类似这样：

```python
tasks = []

for request in incoming_requests:
    tasks.append(asyncio.create_task(process(request)))
```

请求越多，Task 越多，内存最终会被耗尽。

### 使用有界队列

```python
import asyncio


queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=1000)
```

生产者：

```python
async def producer(request: dict) -> None:
    await queue.put(request)
```

当队列达到上限时，`put()` 会等待。这就是最基本的背压。

消费者：

```python
async def consumer() -> None:
    while True:
        request = await queue.get()
        try:
            await process(request)
        finally:
            queue.task_done()
```

启动多个消费者：

```python
async def start_workers(worker_count: int) -> None:
    workers = [
        asyncio.create_task(consumer())
        for _ in range(worker_count)
    ]

    await asyncio.gather(*workers)
```

### 背压策略不只有等待

在在线推理服务中，队列满时可以采用不同策略：

1. **阻塞等待**：适合允许排队的离线任务；
2. **立即拒绝**：返回 429 或服务繁忙；
3. **丢弃低优先级任务**：保护关键请求；
4. **降级处理**：使用更小模型或较低分辨率；
5. **动态扩容**：增加推理实例；
6. **批处理合并**：提高 GPU 利用率。

背压的目标不是让所有请求都成功，而是防止系统在压力下失控。

## 九、异步同步原语：Semaphore、Lock 与 Event

`asyncio` 提供了一组与 `threading` 模块对应的同步原语，但它们是协作式的——不会阻塞线程，而是让协程在事件循环中等待。

### 1. Semaphore：限制并发度

Semaphore 是最常用的限流工具。当系统需要限制同时执行的任务数量时：

```python
import asyncio


semaphore = asyncio.Semaphore(10)


async def limited_fetch(client: httpx.AsyncClient, url: str) -> str:
    async with semaphore:
        response = await client.get(url)
        return response.text


async def fetch_all(urls: list[str]) -> list[str]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = [limited_fetch(client, url) for url in urls]
        return await asyncio.gather(*tasks)
```

典型应用场景：

- 限制并发连接数；
- 限制同时使用的 GPU slot；
- 控制对外部 API 的调用速率。

注意 Semaphore 只限制并发度，不提供队列、超时和拒绝语义。如果需要完整的流量控制，通常要与有界队列配合使用。

### 2. Lock：保护共享状态

当多个协程需要互斥地访问共享资源时：

```python
import asyncio


class ModelManager:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._model: Model | None = None

    async def reload(self, model_path: str) -> None:
        async with self._lock:
            self._model = await load_model(model_path)

    async def predict(self, inputs: dict) -> dict:
        async with self._lock:
            if self._model is None:
                raise RuntimeError("model not loaded")
            return await self._model.infer(inputs)
```

`Lock` 保证 `reload` 和 `predict` 不会同时执行，避免在模型热加载过程中接受推理请求。

需要注意：`asyncio.Lock` 不可重入——同一个协程在持有锁时再次 `acquire` 会死锁。如果需要可重入语义，需要自行封装。

### 3. Event：协程间的信号通知

`Event` 适合"等待某个条件就绪"的场景：

```python
import asyncio


model_ready = asyncio.Event()


async def handle_request(request: dict) -> dict:
    await model_ready.wait()  # 阻塞直到模型加载完成
    return await predict(request)


async def load_and_serve(model_path: str) -> None:
    await load_model(model_path)
    model_ready.set()  # 通知所有等待的协程
```

典型应用：

- 模型加载完成前暂停推理请求；
- 等待初始化流程结束；
- 协调多个工作阶段的启停。

`Event.set()` 会唤醒所有正在 `wait()` 的协程。如果需要反复触发，可以调用 `clear()` 重置状态。

## 十、异步批处理：连接并发与 GPU 利用率

GPU 推理通常不是单个请求越快越好，而是需要在延迟和批量之间做权衡。

一个简单的动态批处理器可以这样实现：

```python
import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

# Python 3.12+ 可以使用 class BatchProcessor[T, R]: 语法
T = TypeVar("T")
R = TypeVar("R")


class BatchProcessor:
    def __init__(
        self,
        infer_batch: Callable[[list[T]], Awaitable[list[R]]],
        max_batch_size: int = 16,
        max_wait_seconds: float = 0.01,
    ) -> None:
        self.queue: asyncio.Queue[
            tuple[T, asyncio.Future[R]]
        ] = asyncio.Queue()
        self.infer_batch = infer_batch
        self.max_batch_size = max_batch_size
        self.max_wait_seconds = max_wait_seconds

    async def submit(self, item: T) -> R:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[R] = loop.create_future()

        await self.queue.put((item, future))
        return await future

    async def run(self) -> None:
        while True:
            first_item, first_future = await self.queue.get()
            batch = [(first_item, first_future)]

            deadline = asyncio.get_running_loop().time() + (
                self.max_wait_seconds
            )

            while len(batch) < self.max_batch_size:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break

                try:
                    item = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=remaining,
                    )
                    batch.append(item)
                except TimeoutError:
                    break

            items = [item for item, _ in batch]

            try:
                results = await self.infer_batch(items)

                for (_, future), result in zip(batch, results):
                    if not future.cancelled():
                        future.set_result(result)

            except Exception as exc:
                for _, future in batch:
                    if not future.cancelled():
                        future.set_exception(exc)

            finally:
                for _ in batch:
                    self.queue.task_done()
```

这个例子体现了几个重要思想：

- 请求通过队列进入推理阶段；
- 推理端按照最大批量收集请求；
- 如果批量未满，也只等待有限时间；
- 每个请求通过自己的 Future 获得结果；
- 推理失败时，错误传播给对应请求。

真实系统还需要增加：

- 请求级超时；
- 取消后的 Future 清理；
- 按模型、租户或优先级分队列；
- GPU 显存水位控制；
- 批量大小动态调整；
- 推理实例熔断和重试。

## 十一、异步流式处理

对于文本生成、音频处理和视频推理，结果可能不是一次性返回，而是持续产生。

Python 可以使用异步生成器表达流式结果：

```python
from collections.abc import AsyncIterator


async def stream_tokens(prompt: str) -> AsyncIterator[str]:
    async for token in engine.generate_stream(prompt):
        yield token
```

调用方可以边生成边发送：

```python
async def handle_request(prompt: str) -> None:
    async for token in stream_tokens(prompt):
        await send_to_client(token)
```

流式处理有三个关键问题。

### 1. 客户端断开连接

客户端关闭连接后，应取消后续推理或生成任务，否则 GPU 仍然可能继续消耗资源。

### 2. 流速不匹配

如果服务端生成速度高于客户端读取速度，发送缓冲区会逐渐增大。因此，发送操作也必须是可等待的，并且要设置合理的缓冲上限。

### 3. 资源清理

流式响应中需要保证：

- 连接被关闭；
- 推理上下文被释放；
- 临时文件被删除；
- 统计指标被上报；
- 取消状态被正确传递。

可以使用异步上下文管理器统一处理生命周期：

```python
from contextlib import asynccontextmanager


@asynccontextmanager
async def inference_session(request_id: str):
    session = await create_session(request_id)
    try:
        yield session
    finally:
        await session.close()
```

## 十二、混合并发：Python 异步与底层 GPU 运行时

很多 AI 系统并不是纯 Python 系统：

```text
异步 HTTP / gRPC 服务
        ↓
Python 编排层
        ↓
PyTorch / TensorRT / CUDA Runtime
        ↓
GPU
```

这类系统需要明确区分两种"异步"：

1. **Python 异步**：事件循环层面的协作式调度；
2. **GPU 异步**：CPU 提交 CUDA 操作后，GPU 在设备端继续执行。

二者并不等价。

例如，某个 GPU API 返回后，并不一定代表 GPU 计算已经完成。某些同步操作可能触发等待，导致 Python 线程停顿。另一方面，如果多个请求共享模型、CUDA Stream 或缓存，也需要明确线程安全和并发语义。

一个常见的做法是将推理放到独立线程中，避免阻塞事件循环：

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

import torch

# 推理线程池，与事件循环线程隔离
_infer_pool = ThreadPoolExecutor(max_workers=2)


def _run_inference(
    model: torch.nn.Module,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """在独立线程中执行推理，避免阻塞事件循环。"""
    with torch.no_grad():
        return model(inputs.to("cuda")).cpu()


async def infer_async(
    model: torch.nn.Module,
    inputs: torch.Tensor,
) -> torch.Tensor:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _infer_pool,
        _run_inference,
        model,
        inputs,
    )
```

这种模式下，事件循环负责接收请求和编排任务，推理线程负责与 GPU 交互。需要注意的是，如果模型或推理引擎不是线程安全的，`max_workers` 应设为 1，或者为每个 worker 维护独立的模型实例。

工程上应注意：

- 不要在事件循环中执行长时间的同步 GPU 等待；
- 不要假设 Python Task 并发就等于 GPU 计算并行；
- 根据推理引擎的线程安全要求设计 worker；
- 明确 CUDA Stream、显存缓存和请求上下文的所有权；
- 通过批处理和设备级调度提高 GPU 利用率；
- 使用指标区分排队时间、CPU 预处理时间、GPU 执行时间和结果传输时间。

一个请求的总延迟可以拆解为：

```text
总延迟 =
    排队时间
  + 预处理时间
  + Host 到 Device 传输时间
  + GPU 执行时间
  + Device 到 Host 传输时间
  + 后处理时间
  + 网络发送时间
```

如果只观察接口总耗时，就很难判断瓶颈到底在哪里。

## 十三、异步资源监控

资源监控任务通常不应该阻塞主调度循环。

```python
import asyncio


async def report_gpu_metrics() -> None:
    while True:
        try:
            metrics = await collect_gpu_metrics()
            await publish_metrics(metrics)
        except Exception:
            # 监控失败不能影响主服务
            log_exception()
        finally:
            await asyncio.sleep(5)
```

启动监控任务：

```python
async def serve() -> None:
    monitor_task = asyncio.create_task(report_gpu_metrics())

    try:
        await run_server()
    finally:
        monitor_task.cancel()
        await asyncio.gather(
            monitor_task,
            return_exceptions=True,
        )
```

后台任务必须具备明确的关闭流程：

1. 停止接收新请求；
2. 等待关键请求完成；
3. 取消监控和刷新任务；
4. 关闭网络连接；
5. 释放线程池、进程池和 GPU 资源；
6. 刷新日志和指标。

如果只调用 `cancel()` 而不等待任务结束，清理逻辑可能没有机会执行。

## 十四、常见错误与改进方式

### 错误一：把所有任务都改成异步

异步适合 I/O，不适合自动解决 CPU 瓶颈。

```python
async def compute() -> int:
    return heavy_python_computation()
```

改进方式：根据计算类型选择线程、进程或专用计算运行时。

### 错误二：在协程中调用同步网络库

```python
async def handler() -> None:
    response = requests.get(url)
```

改进方式：

- 使用异步客户端；
- 或通过 `asyncio.to_thread()` 调用同步客户端。

### 错误三：无限制创建 Task

```python
for item in items:
    asyncio.create_task(process(item))
```

改进方式：使用有界队列（第八章）、Semaphore（第九章）或固定数量的 worker。

```python
semaphore = asyncio.Semaphore(100)


async def limited_process(item: dict) -> None:
    async with semaphore:
        await process(item)
```

但需要注意，Semaphore 只能限制同时执行的任务数，不能替代完整的队列、超时和拒绝策略。

### 错误四：没有设置超时

任何跨网络边界的调用都可能永远等待。生产环境中的 RPC、数据库、对象存储和模型调用都应设置超时。

### 错误五：忽略取消

服务关闭、客户端断开、超时和上游取消都会触发取消传播。协程必须释放资源，并且通常应重新抛出 `CancelledError`。

### 错误六：把异步当成低延迟保证

异步可以减少等待期间的资源浪费，但不能消除：

- 网络延迟；
- GPU 排队；
- 模型执行时间；
- 序列化成本；
- 锁竞争；
- 数据拷贝；
- 下游服务拥塞。

异步改善的是并发组织方式，不是物理执行时间。

## 十五、一个实用的并发决策树

可以按照以下顺序做选择：

### 第一步：确定主要瓶颈

- 网络、数据库、RPC、磁盘等待：优先考虑 `asyncio` 或线程；
- 纯 Python CPU 计算：优先考虑进程或专用计算库；
- GPU 推理：重点优化批处理、设备调度和数据移动；
- 外部任务等待：使用异步任务、队列和状态轮询。

### 第二步：确认依赖库的接口类型

- 原生异步库：直接接入事件循环；
- 只有同步接口：放入线程池；
- 不适合进程间复制的大对象：避免直接交给进程池；
- 线程不安全的客户端：为每个 worker 管理独立实例。

### 第三步：设计容量边界

至少明确：

- 最大并发请求数；
- 队列最大长度；
- 单请求超时时间；
- worker 数量；
- 批处理最大大小；
- 内存和显存上限；
- 队列满时的处理策略。

### 第四步：设计故障传播

明确以下问题：

- 一个子任务失败，是否取消同组任务？
- 请求超时，是否取消 GPU 推理？
- 客户端断开，是否停止结果生成？
- 服务退出时，哪些任务必须完成？
- 重试是否可能造成重复执行？

### 附：Java 与 Python 并发概念对照

| 概念 | Java | Python |
| :--- | :--- | :--- |
| 线程 | `Thread` / `ExecutorService` | `threading` / `ThreadPoolExecutor` |
| 线程池 | `Executors.newFixedThreadPool()` | `concurrent.futures.ThreadPoolExecutor` |
| 异步编排 | `CompletableFuture` | `asyncio` / `await` |
| 轻量级线程 | Project Loom 虚拟线程 | 协程（`async def`） |
| 结构化并发 | `StructuredTaskScope`（Loom） | `asyncio.TaskGroup` |
| 背压队列 | `BlockingQueue` / Reactive Streams | `asyncio.Queue(maxsize=N)` |
| 信号量 | `java.util.concurrent.Semaphore` | `asyncio.Semaphore` |
| 互斥锁 | `synchronized` / `ReentrantLock` | `asyncio.Lock` |
| 上下文传递 | `ThreadLocal` / `ScopedValue` | `contextvars.ContextVar` |
| CPU 并行 | `parallelStream()` / `ForkJoinPool` | `ProcessPoolExecutor` |
| GIL 限制 | 无 | 限制 Python 字节码的多线程并行 |

## 结语：并发的核心是控制复杂性

Python 并发编程的难点，并不在于记住 `async def`、`await` 或线程池 API，而在于建立正确的系统模型：

- 线程解决部分阻塞 I/O 问题；
- 进程解决部分 CPU 并行问题；
- asyncio 解决大量 I/O 任务的协作问题；
- 队列解决任务解耦和流量缓冲问题；
- 背压解决生产速度超过消费能力的问题；
- 超时和取消解决任务生命周期问题；
- 结构化并发解决任务归属和异常传播问题；
- 批处理解决请求并发与 GPU 利用率之间的平衡问题。

在 AI-Infra 中，一个健壮的并发系统通常不是"同时运行尽可能多的任务"，而是：

> 在明确的容量边界内，让不同类型的任务以合适的并发模型协作，并且在过载、失败和关闭时保持可控。

当 Python 负责的是模型服务和基础设施编排时，真正需要优化的往往不是某一行代码，而是整个任务流：

```text
请求接入
  → 限流
  → 排队
  → 批处理
  → 调度
  → 推理
  → 流式返回
  → 指标上报
  → 资源回收
```

只有把这条链路中的等待、计算、资源和失败边界都设计清楚，异步和并发才能真正转化为 AI 系统的吞吐量、稳定性与可观测性。
