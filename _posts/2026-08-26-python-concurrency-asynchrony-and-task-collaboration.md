---
layout: post
title: Python 在 AI-Infra（03）：并发、异步与任务协作
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

本文每一节都会给出**对应 Java** 的说明，基准是 **Java 25 LTS**（2025-09 发布）。这个基准很重要：Java 21 之后的虚拟线程、结构化并发和 ScopedValue，让 Java 的并发模型和 Python 的 asyncio 在概念上前所未有地接近，但底层机制和失败模式仍有本质差异。凡是 preview 特性都会显式标注。

## 一、先理解并发：并发不等于并行

### 1. 并发与并行

**并发（Concurrency）** 同时发生。描述的是多个任务在同一时间段内交替推进。

**并行（Parallelism）** 同时进行。描述的是多个任务在同一时刻真正同时执行。

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

| 模型 | 适合场景 | 主要成本 | 典型问题 | Java 对应 |
| :--- | :--- | :--- | :--- | :--- |
| 线程 | 阻塞 I/O、兼容同步库 | 线程栈、上下文切换 | 共享状态、线程安全 | 平台线程 `ExecutorService` |
| 进程 | CPU 密集型计算 | 进程创建、序列化、内存 | 数据复制、进程通信 | 无需对应（无 GIL） |
| asyncio | 大量 I/O、任务编排 | 需要异步调用链 | 事件循环被阻塞 | 虚拟线程 / `CompletableFuture` / Reactor |

这张表里最需要 Java 程序员注意的是第二行和第三行。

**第二行**：Java 几乎从不为了绕开语言限制而开多进程，因为 JVM 里多线程就能吃满多核。Python 的 `ProcessPoolExecutor` 在 Java 里没有等价物——它存在的唯一理由就是 GIL。

**第三行**：Java 21 之后，"高并发 I/O"的答案从 `CompletableFuture`/Reactor 变成了虚拟线程。虚拟线程让你**用同步代码写法拿到异步的伸缩性**，而 Python 至今仍必须显式写 `async`/`await`。这是两个生态最大的分野，后面第三章会展开。

没有一种模型适合所有场景。工程上的第一原则是：

> 不要根据 API 的流行程度选择并发模型，而要根据瓶颈类型选择并发模型。

## 二、GIL：Python 并发的底层约束

GIL（Global Interpreter Lock）是 CPython 解释器中的一把全局锁。它保证在任意时刻，只有一个线程可以执行 Python 字节码。

这意味着即使创建了多个线程，纯 Python 的 CPU 密集型代码在同一时刻仍然只有一个线程在运行。GIL 的存在是为了保护 CPython 内部的引用计数和内存管理机制，使得对象的创建和销毁不需要细粒度锁。

对应 Java：**JVM 没有任何等价物**。这是两个语言并发模型最根本的差异，根源在内存管理策略：

| | CPython | JVM |
| :--- | :--- | :--- |
| 内存回收 | 引用计数为主 + 循环 GC 兜底 | 追踪式 GC（G1 / ZGC / Shenandoah） |
| 对象访问的并发保护 | 一把全局锁（GIL） | 无全局锁，靠 JMM + 逃逸分析 + 偏向/轻量级锁 |
| 多线程 CPU 并行 | 不可以（默认构建） | 可以 |
| 内存可见性模型 | 无正式规范（GIL 隐式提供了顺序） | JMM（JLS 第 17 章）、`volatile`、`happens-before` |

有意思的是，GIL 让 Python 程序员**几乎不需要考虑内存可见性**：因为同一时刻只有一个线程在跑字节码，你不会遇到 Java 里那种"没加 `volatile` 导致另一个线程永远看不到 flag 变化"的问题。Python 没有 `volatile` 关键字，也不需要。

但这份"便利"代价极高：你失去了多核。Java 程序员迁移过来时最常见的误判就是——把在 Java 里跑得好好的 `ExecutorService` + CPU 计算直接翻译成 `ThreadPoolExecutor`，然后发现 16 线程和 1 线程一样慢。

### 1. GIL 何时释放

GIL 并非始终被持有。以下操作通常会释放 GIL：

- 网络 I/O（socket read/write）；
- 文件 I/O（磁盘读写）；
- `time.sleep()`；
- C 扩展调用（NumPy、PyTorch 的底层计算）；
- 数据库驱动的等待操作。

这就是为什么线程仍然能有效加速 I/O 密集型任务：当一个线程在等待网络响应时，GIL 被释放，其他线程可以继续执行 Python 代码。

对应 Java：这个"释放 GIL"的动作，在 Java 21+ 的虚拟线程里有一个形似而神不同的对应物——**unmount（卸载）**。虚拟线程执行阻塞 I/O 时会从载体平台线程上卸载，让出载体给别的虚拟线程。区别在于：

- Python 释放 GIL 是**让出解释器的执行许可**，线程本身还在 OS 里阻塞着，OS 线程数不变；
- Java 卸载虚拟线程是**归还 OS 线程**，所以能开几十万个虚拟线程而只占几十个 OS 线程。

换句话说，Python 的线程模型永远是 1:1 映射到 OS 线程的（相当于 Java 的平台线程），Python 里 M:N 调度的那一层是 `asyncio` 的协程，而不是 `threading`。

顺带一提，Java 里曾有一个和"GIL 未释放"高度类似的坑：虚拟线程在 `synchronized` 块内阻塞会 **pin**（钉住）载体线程，导致载体无法复用。**JEP 491 在 JDK 24 已经消除了这个问题**，`synchronized` 内阻塞也能正常卸载了。JDK 21～23 上仍需把热点路径的 `synchronized` 换成 `ReentrantLock`。

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

对应 Java：free-threaded CPython 面临的挑战，正是 JVM 在 20 多年前就解决过的那些——细粒度锁、无锁数据结构、内存序、伪共享。有一点差异值得留意：JVM 从第一天起就有 JMM 规范，任何库作者都知道自己写的代码要在多线程下正确。而 Python 生态里绝大多数 C 扩展是在"有 GIL 兜底"的假设下写的，所以 free-threaded 模式的真正阻力不在 CPython 本身，而在 NumPy、PyTorch 这些扩展需要逐一审计和适配。这类似于"如果 Java 的整个生态都是在单线程假设下写出来的，然后某天你打开了多线程"。

## 三、线程、进程与异步：如何做工程决策

### 1. 线程：适合阻塞 I/O

线程适合这样的任务：

- 调用只提供同步接口的 HTTP 客户端；
- 访问传统数据库驱动；
- 读取文件或设备；
- 调用同步 SDK；
- 等待外部服务返回。

一个真实例子是 `huggingface_hub` 下载模型权重。`snapshot_download` 内部用线程池并发拉取多个 shard 文件，因为 HTTP 下载和磁盘写入都会释放 GIL：

```python
from huggingface_hub import snapshot_download

# max_workers 控制并发下载的文件数，默认 8
snapshot_download(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    max_workers=16,
)
```

自己写的话是大概是这个样子：

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

对应 Java：

```java
// 平台线程池：与 ThreadPoolExecutor(max_workers=16) 语义最接近
try (var pool = Executors.newFixedThreadPool(16)) {   // Java 19+ 实现了 AutoCloseable
    List<Future<String>> futures = pool.invokeAll(
        requestIds.stream().map(id -> (Callable<String>) () -> invokeSyncModel(id)).toList()
    );
}

// Java 21+ 更推荐：虚拟线程，不需要调 max_workers
try (var pool = Executors.newVirtualThreadPerTaskExecutor()) {
    List<Future<String>> futures = pool.invokeAll(tasks);
}
```

三个关键差异：

1. **`with` 块的语义相同**。`ThreadPoolExecutor.__exit__` 等价于 `shutdown()` + `awaitTermination()`；Java 19 起 `ExecutorService` 实现了 `AutoCloseable`，`close()` 也是这个语义。之前必须手写 `try/finally { pool.shutdown(); }`。
2. **`executor.map()` ≈ `invokeAll()`**，都是提交一批并等全部完成。但 `map` 返回的迭代器是**惰性且按顺序**的：取第 2 个结果时会阻塞，即使第 3 个早就完成了。要按完成顺序拿结果得用 `as_completed()`，对应 Java 的 `ExecutorCompletionService` 或 Java 21+ 的 `StructuredTaskScope`。
3. **线程池大小的选择逻辑不同**。Java 21+ 用虚拟线程时你根本不需要选 `max_workers`——I/O 任务开多少个都行。Python 里 `max_workers` 是硬约束，每个 worker 都是一个真实的 OS 线程（默认栈 8MB 虚拟内存），开到几千个就会有明显的调度开销。所以 Python 里大规模并发 I/O 的正确答案是 asyncio 而不是线程池。

**注意**：`ThreadPoolExecutor` 默认 `max_workers = min(32, os.cpu_count() + 4)`，不像 Java 的 `newCachedThreadPool()` 那样无上限增长。这个默认值对 I/O 任务通常偏小，需要显式指定。

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

这个模式在 AI 生态里最典型的落地就是 `torch.utils.data.DataLoader`：

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,        # 8 个子进程做数据加载和预处理
    prefetch_factor=2,    # 每个 worker 预取 2 个 batch
    pin_memory=True,      # 结果放入 pinned memory，加速 H2D 拷贝
    persistent_workers=True,  # 不要每个 epoch 都重建进程
)
```

`num_workers>0` 时 DataLoader 会 fork 出子进程，每个子进程独立跑图像解码、augmentation 这些纯 Python/CPU 的活，再通过**共享内存**把 Tensor 传回主进程（而不是 pickle 整个数组）。`num_workers=0` 就退化成主进程串行加载，GPU 大概率会饿死。

DataLoader 也精确演示了多进程的两个经典代价：

- `RuntimeError: DataLoader worker (pid X) is killed by signal: Bus error` —— 容器里 `/dev/shm` 默认只有 64MB，共享内存不够。要么 `docker run --shm-size=8g`，要么调小 `num_workers`。
- 每个 worker 都是主进程的完整拷贝。如果 `Dataset.__init__` 里加载了一个大字典，8 个 worker 就是 8 份（copy-on-write 会因为引用计数写入而失效）。

对于大模型场景，尤其要注意：

- 不要无意中复制巨大的模型对象；
- 不要通过进程间消息传输大型 Tensor；
- 不要在每个子进程中重复初始化昂贵的运行时；
- 使用共享内存时，需要明确生命周期和所有权。

进程并不是"更快的线程"，而是更强隔离、也更高成本的执行单元。

对应Java：

Java 只有在需要**故障隔离**或**独立 JVM 参数**时才拆进程（`ProcessBuilder`），而 Python 拆进程首要目的是绕开 GIL。这个动机差异带来一串连锁反应：

| | Java 多线程 | Python 多进程 |
| :--- | :--- | :--- |
| 数据传递 | 共享堆上的对象引用，零拷贝 | pickle 序列化 + 管道，或显式共享内存 |
| 传大对象 | 免费 | 昂贵，可能抵消并行收益 |
| 全局状态 | 天然共享 | 每个进程一份，需要 `initializer` 重建 |
| 启动成本 | 微秒级 | 毫秒到秒级（`spawn` 要重新 import 整个模块树） |
| 崩溃影响 | 整个 JVM | 只死一个 worker |

其中"启动成本"这条最容易踩坑：macOS 和 Windows 上 Python 默认用 `spawn` 而不是 `fork`，子进程会重新执行模块顶层代码，所以必须写 `if __name__ == "__main__":` 保护——这在 Java 里完全没有对应概念。**Python 3.14 起 Linux 上的默认启动方式也从 `fork` 改成了 `forkserver`**，因为 `fork` 与多线程（尤其是 CUDA 已初始化的进程）混用会死锁。

如果确实需要"共享堆"的效果，Python 侧最接近的是 `multiprocessing.shared_memory`，但生命周期要自己管，不像 JVM 有 GC 兜底。

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

对应 Java：这一节是两个生态**分歧最大**的地方。同样的"并发抓一批 URL"，Java 有三种写法，演进脉络很清楚：

```java
// 1. Java 8 起：CompletableFuture，回调式，可读性差
List<CompletableFuture<String>> futures = urls.stream()
        .map(url -> client.sendAsync(request(url), BodyHandlers.ofString())
                          .thenApply(HttpResponse::body))
        .toList();
CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).join();

// 2. Java 21+：虚拟线程，同步写法 + 异步伸缩性
try (var pool = Executors.newVirtualThreadPerTaskExecutor()) {
    List<Future<String>> results = pool.invokeAll(
        urls.stream().map(url -> (Callable<String>) () ->
            client.send(request(url), BodyHandlers.ofString()).body()
        ).toList()
    );
}
```

对比 Python 的写法，核心区别是**函数染色（function coloring）**：

| | Python asyncio | Java 虚拟线程 |
| :--- | :--- | :--- |
| 调用阻塞操作 | 必须 `await`，函数必须是 `async def` | 直接调用，普通方法即可 |
| 传染性 | 有：一个 `async` 会沿调用链向上传染 | 无 |
| 同步库能否直接用 | 不能，会阻塞事件循环 | 能，JDK 已改写内部阻塞点 |
| 库生态 | 需要 `requests` / `httpx.AsyncClient` 两套 | 一套 `HttpClient` 通吃 |
| 让出执行权 | 只在 `await` 点，协作式 | 由 JDK 在阻塞点自动卸载 |

`requests.get()` 在虚拟线程里跑就是正确的高并发代码；`requests.get()` 在协程里跑就是把整个服务卡死的事故。这是 Java 程序员写 Python 异步代码时最高频的线上事故来源。

**一个补偿性的好处**：正因为 Python 只在 `await` 处让出，两个 `await` 之间的代码是原子的（相对于同一事件循环内的其他协程），所以协程之间共享可变状态时，很多在 Java 里必须加锁的场景在 Python 里不需要。代价是你必须能一眼看出哪些行是 `await` 点——这也是为什么 Python 保留显式 `async` 语法而不学 Java 走隐式路线的主要论据之一。

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

对应 Java：这套模型 Java 程序员其实很熟悉，它就是 **Netty 的 EventLoop**。`uvicorn`（Python 最主流的 ASGI 服务器）默认用 `uvloop`——基于 libuv 的事件循环实现，和 Netty 基于 epoll/kqueue 是同一层抽象。

对照关系：

| Python asyncio | Netty / Reactor | 说明 |
| :--- | :--- | :--- |
| 事件循环 | `EventLoop` / `EventLoopGroup` | 都是单线程轮询 + 回调 |
| `uvloop` | Netty native transport（epoll/kqueue） | 都是 C 实现的加速版 |
| 协程被 `await` 挂起 | `ChannelFuture` + listener | Python 用语法糖藏起了回调 |
| 阻塞事件循环 | 在 EventLoop 线程里做阻塞调用 | 完全相同的事故模式 |
| `asyncio.to_thread()` | `Schedulers.boundedElastic()` / 业务 `EventExecutorGroup` | 把阻塞活外包出去 |

Netty 那条铁律"**永远不要在 EventLoop 线程里执行阻塞操作**"，逐字适用于 asyncio。Spring WebFlux 里 `.block()` 引发的线上事故，和 Python 里 `time.sleep()` 引发的是同一类问题。

区别在于**默认线程数**：Netty 默认起 `2 * CPU` 个 EventLoop，一个被阻塞还有其他的顶着；`asyncio.run()` 只有**一个**事件循环、跑在一个线程里，阻塞一次就是全局停摆。多核部署靠的是起多个 worker 进程（`uvicorn --workers 4` 或 gunicorn + UvicornWorker），本质上是"多进程 + 每进程一个事件循环"的 SO_REUSEPORT 模型。

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

这个桥在 FastAPI 里是自动的，而且是它最重要的设计决策之一。Starlette 的实现只有几行：

```python
# starlette/concurrency.py
async def run_in_threadpool(func, *args, **kwargs):
    if kwargs:
        func = functools.partial(func, **kwargs)
    return await anyio.to_thread.run_sync(func, *args)
```

FastAPI 据此分流：**`async def` 端点在事件循环上跑，普通 `def` 端点自动丢进线程池**。

```python
@app.post("/embed")
async def embed_async(req: Request):
    # 在事件循环上执行——这里面绝不能有阻塞调用
    return await client.post(...)


@app.post("/predict")
def predict_sync(req: Request):
    # 自动被 run_in_threadpool 包装，可以放心写阻塞代码
    return model.predict(req.data)
```

所以 FastAPI 里最反直觉的一条经验是：**如果你的处理函数是阻塞的，把 `async def` 改回 `def` 反而更安全**。线程池默认 40 个线程（AnyIO 的 default limiter），这也是它的容量上限。

对应 Java：

```java
// Reactor / WebFlux：把阻塞调用外包给弹性线程池
Mono.fromCallable(() -> blockingSdk.call(payload))
    .subscribeOn(Schedulers.boundedElastic());
```

`Schedulers.boundedElastic()` 就是 `asyncio.to_thread()` 的等价物，连"有界"这个设计取向都一样（默认上限 `10 * CPU` 个线程）。

但 Java 21+ 之后，这座桥基本可以拆了——虚拟线程上直接调阻塞 SDK 就行，不需要外包给另一个线程池。Spring 6.1 的 `spring.threads.virtual.enabled=true` 就是在做这件事。**Python 目前没有等价的"拆桥"方案**，Sync/Async Bridge 是长期存在的结构，而不是过渡方案。

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

debug 模式会对执行超过 100ms 的回调打 `Executing <Handle...> took 0.5 seconds` 警告，阈值可以用 `loop.slow_callback_duration` 调整。

对应 Java：同一类问题有三种成熟工具，Python 侧目前都没有等价物：

| 目的 | Java | Python |
| :--- | :--- | :--- |
| 开发期检测阻塞调用 | **BlockHound**（字节码插桩，检测到阻塞直接抛异常） | `asyncio` debug 模式（只事后告警，不拦截） |
| 生产期定位 pinning | **JFR `jdk.VirtualThreadPinned` 事件** | 无内置对应；靠 `asyncio.all_tasks()` + 自建指标 |
| 线程/协程转储 | `jstack`、`Thread.dump_to_file`（JDK 21+ 支持虚拟线程） | `asyncio.all_tasks()` + `task.get_stack()` |

BlockHound 的缺失是实打实的短板：Python 没有办法在运行时可靠地识别"这次调用会阻塞"。工程上只能靠约定和 code review——比如在 CI 里用 `flake8-async`（旧名 `flake8-trio`）静态扫描协程里的 `time.sleep`、`requests.`、`open()` 这类调用。

生产系统则应结合日志、指标和 tracing，定位具体是哪一个任务长时间没有让出执行权。一个便宜且有效的做法是常驻一个"心跳协程"来测量事件循环延迟，等价于监控 Netty EventLoop 的 pending task 积压：

```python
async def monitor_loop_lag(interval: float = 0.5) -> None:
    loop = asyncio.get_running_loop()
    while True:
        start = loop.time()
        await asyncio.sleep(interval)
        lag = loop.time() - start - interval
        if lag > 0.1:
            logger.warning("event loop lag: %.3fs", lag)
```

## 五、任务：协程、Task 与 Future

在 `asyncio` 中，需要区分几个概念。

### 协程对象

调用异步函数时，得到的是协程对象：

```python
coro = fetch_data()
```

此时函数体还没有开始执行。协程对象必须被 `await` 或提交给 `create_task()` 后才会运行。

对应 Java：**没有对应物，而且这是最容易出 bug 的差异**。Java 里没有"惰性的异步计算"这个概念：

```java
// Java: 调用的瞬间就已经提交给 ForkJoinPool 开始跑了
CompletableFuture<String> f = CompletableFuture.supplyAsync(this::fetchData);
```

```python
# Python: 什么都没发生，函数体一行都没执行
coro = fetch_data()
```

如果你按 Java 的直觉写下面这段，两个请求是**串行**的，总耗时是两者之和：

```python
a = await fetch_user()      # 等它完成
b = await fetch_orders()    # 再开始下一个
```

要并发必须显式转成 Task：

```python
task_a = asyncio.create_task(fetch_user())    # 现在才开始调度
task_b = asyncio.create_task(fetch_orders())
a, b = await task_a, await task_b
```

配套的报错也是 Python 特有的：忘记 `await` 时，你得到的不是编译错误（Java 里 `CompletableFuture<String>` 赋给 `String` 编译不过），而是运行时一句 `RuntimeWarning: coroutine 'fetch_data' was never awaited`，业务逻辑静默地没有执行。mypy/pyright 能查出大部分这类问题，务必开启。

### Task

Task 是事件循环中被调度执行的协程：

```python
task = asyncio.create_task(fetch_data())
```

创建 Task 后，事件循环可以在适当时机执行它。

对应 Java：`asyncio.Task` ≈ `CompletableFuture` + `Future`（承担了两者的职责）。API 对照：

| Python | Java | 备注 |
| :--- | :--- | :--- |
| `asyncio.create_task(coro)` | `CompletableFuture.supplyAsync(sup)` | Java 侧立即开始执行 |
| `await task` | `f.get()` / `f.join()` | Java 阻塞线程，Python 挂起协程 |
| `task.done()` | `f.isDone()` | 相同 |
| `task.result()` | `f.getNow(null)` | 未完成时 Python 抛 `InvalidStateError` |
| `task.cancel()` | `f.cancel(true)` | 见第六章，取消语义差别很大 |
| `asyncio.gather(*ts)` | `CompletableFuture.allOf(...)` | `gather` 直接返回结果列表，`allOf` 返回 `Void` |
| `asyncio.wait(ts, return_when=FIRST_COMPLETED)` | `CompletableFuture.anyOf(...)` | 相同语义 |
| `task.add_done_callback(fn)` | `f.thenAccept(fn)` / `whenComplete` | Python 没有链式组合子 |

最后一行值得展开：Python **没有** `thenApply` / `thenCompose` / `thenCombine` 这一整套组合子，因为 `await` 已经让顺序组合变成了普通的顺序代码。`thenCompose(this::next)` 在 Python 里就是 `next(await first())`。这是 async/await 语法相对 CompletableFuture 链式 API 的主要优势。

还有一个 Python 独有的陷阱：**`create_task()` 返回的 Task 只被事件循环弱引用**。如果不保存引用，任务可能在执行到一半时被 GC 回收：

```python
# 错误：task 随时可能被 GC
asyncio.create_task(background_job())

# 正确：持有强引用直到完成
_background_tasks: set[asyncio.Task] = set()

task = asyncio.create_task(background_job())
_background_tasks.add(task)
task.add_done_callback(_background_tasks.discard)
```

Java 里没有这个问题——线程是 GC root，跑起来的任务不会被回收。

### Future

Future 表示一个未来会完成的结果。它通常是底层异步操作和任务调度之间的桥梁。

大多数业务代码不需要直接创建 Future，但理解它有助于理解 `await`、Task 和底层 I/O 的关系。

对应 Java：`asyncio.Future` 就是**可以手动完成的 `CompletableFuture`**，而不是 `java.util.concurrent.Future`（后者只读，对应 Python 里没有单独类型）。

| Python | Java |
| :--- | :--- |
| `loop.create_future()` | `new CompletableFuture<>()` |
| `future.set_result(v)` | `f.complete(v)` |
| `future.set_exception(e)` | `f.completeExceptionally(e)` |

用途也完全一致：当结果由**事件循环之外**的东西产生时（回调式 C 库、另一个线程、批处理器的调度端），用它把回调世界桥接回 `await` 世界。第十章的批处理器就是这个模式的完整例子。

跨线程完成 Future 时有一个 Python 特有的约束：**`asyncio.Future` 不是线程安全的**，必须用 `loop.call_soon_threadsafe(future.set_result, value)`。Java 的 `CompletableFuture.complete()` 本身就是线程安全的，可以从任意线程直接调。这也是把回调式 SDK 接入 asyncio 时最常见的错误来源。

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

对应 Java：`asyncio.TaskGroup`（Python 3.11+）对应 **`StructuredTaskScope`**，来自 Project Loom。

> **版本说明：** `StructuredTaskScope` 到 **JDK 25 仍是 preview**（JEP 505，第五次预览；JDK 26 的 JEP 525 是第六次），编译运行需要 `--enable-preview`。JDK 25 对 API 做了较大改动：不再用构造器，改为静态工厂 `open()`，并引入 `Joiner` 表达完成策略。网上 JDK 21 时代的 `new StructuredTaskScope.ShutdownOnFailure()` 示例已经过时。

```java
// JDK 25 (preview)
Response handle() throws InterruptedException {
    try (var scope = StructuredTaskScope.open()) {     // 默认：任一失败则全体取消
        Subtask<Encoded>  encoded  = scope.fork(() -> encode(requestId));
        Subtask<Metadata> metadata = scope.fork(() -> loadMetadata(requestId));

        scope.join();                                  // 汇合，失败则抛出

        return infer(encoded.get(), metadata.get());
    }
}
```

两边的设计意图高度一致，逐条对照：

| 语义 | Python `TaskGroup` | Java `StructuredTaskScope`（JDK 25） |
| :--- | :--- | :--- |
| 作用域 | `async with` | `try`-with-resources |
| 启动子任务 | `group.create_task(coro)` | `scope.fork(callable)` |
| 汇合 | 退出 `async with` 块时隐式 | 显式 `scope.join()`，且必须调用 |
| 取结果 | `task.result()` | `subtask.get()`（`join()` 之后才安全） |
| 执行载体 | 事件循环上的协程 | 每个 fork 一个虚拟线程 |
| 默认失败策略 | 任一子任务失败 → 取消其余 → 抛出 | 同左（零参 `open()`） |
| "取第一个成功的" | 无内置，需 `asyncio.wait(FIRST_COMPLETED)` + 手动取消 | `Joiner.anySuccessfulResultOrThrow()` |
| "全部完成不管成败" | `asyncio.gather(..., return_exceptions=True)` | `Joiner.allSuccessfulOrThrow()` / 自定义 Joiner |
| 多个子任务同时失败 | `ExceptionGroup`，配 `except*` 语法 | 首个异常 + `addSuppressed()` |

三个实质差异：

**1. 汇合是隐式还是显式。** Python 在 `async with` 退出时自动等待，你无法"忘记 join"；Java 必须显式 `join()`，漏掉会在 `close()` 时抛异常。Python 的写法更难写错，但也意味着**块内不能提前拿到子任务结果**——`task.result()` 必须写在 `async with` 块之外。

**2. 多重失败的表达。** 这是 Python 更完整的一处。两个子任务同时失败时：

```python
try:
    async with asyncio.TaskGroup() as group:
        group.create_task(encode(request_id))
        group.create_task(load_metadata(request_id))
except* TimeoutError as eg:          # 3.11+ 的 except* 语法
    logger.warning("timeouts: %d", len(eg.exceptions))
except* ValueError as eg:
    logger.error("bad input: %s", eg.exceptions)
```

`ExceptionGroup` + `except*` 让你能按类型分别处理**一组**并发异常。Java 侧只有 `Throwable.addSuppressed()`，遍历 suppressed 数组要手写 `instanceof` 分派，没有语言级支持。

**3. 成熟度相反。** Python 的 `TaskGroup` 自 3.11 起就是**稳定 API**，vLLM、FastAPI 生态已经在生产里大量使用；Java 的 `StructuredTaskScope` 预览了五轮仍未转正，API 每个版本都在变。所以从 Java 迁过来的人反而会发现 Python 这块更能直接用。

相比之下，随意创建后台任务容易造成任务泄漏：

```python
asyncio.create_task(background_job())
```

如果没有保存引用、处理异常和设计退出流程，任务可能在服务关闭时被强行终止，异常也可能无法被业务感知。这正是 `TaskGroup` 和 `StructuredTaskScope` 共同要消灭的东西——Loom 团队称之为"**逃逸的并发**"，即子任务的生命周期超出了创建它的语法块。

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

对应 Java：

```java
// CompletableFuture：超时后 future 以 TimeoutException 完成
CompletableFuture<Response> f = inferAsync(payload)
        .orTimeout(2, TimeUnit.SECONDS)
        .exceptionally(e -> Response.error("inference timeout"));

// 或者给一个降级值
inferAsync(payload).completeOnTimeout(FALLBACK, 2, TimeUnit.SECONDS);

// 虚拟线程 / 阻塞风格
future.get(2, TimeUnit.SECONDS);   // 抛 TimeoutException
```

一个关键的语义差异：**`orTimeout()` 和 `future.get(timeout)` 都不会真正停止上游计算**，它们只是让下游不再等。上游任务仍在后台跑，继续占着连接和 CPU。而 Python 的 `asyncio.timeout()` 是一个**取消作用域**——超时时它会向作用域内的协程投递 `CancelledError`，任务真的会停下来。

这个差异在 AI 服务里代价很大：客户端超时后如果 GPU 推理还在跑，就是纯粹的算力浪费。Java 侧要拿到同等效果，得显式 `future.cancel(true)` 并保证任务体响应中断，或者用 `StructuredTaskScope`（其 `close()` 会取消所有未完成子任务）。

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

对应 Java：`asyncio.CancelledError` 的对应物是 **`InterruptedException`**，而且连"不要吞掉"这条铁律都一模一样。Java 侧的标准写法：

```java
try {
    while (true) {
        Item item = queue.take();
        process(item);
    }
} catch (InterruptedException e) {
    cleanup();
    Thread.currentThread().interrupt();   // 恢复中断标志，等价于 Python 的 raise
}
```

`Thread.currentThread().interrupt()` 这行的作用，和 Python 里那句 `raise` 完全一致：把"我被要求停止"这个信号继续往上传，否则调用栈上层永远不知道发生过取消。两个语言里最经典的并发代码坏味道，都是空的 catch 块：

```java
catch (InterruptedException e) { }        // Java 反模式
```

```python
except asyncio.CancelledError: pass       # Python 反模式，效果完全相同
```

对照表：

| 语义 | Python | Java |
| :--- | :--- | :--- |
| 取消信号 | `task.cancel()` | `thread.interrupt()` / `future.cancel(true)` |
| 收到信号的表现 | 下一个 `await` 点抛 `CancelledError` | 下一个阻塞点抛 `InterruptedException` |
| 协作式？ | 是，纯计算循环里不会被打断 | 是，纯计算循环里不会被打断 |
| 状态标志 | `task.cancelling()` / `uncancel()`（3.11+） | `Thread.isInterrupted()` |
| 继承自 | `BaseException`（不被 `except Exception` 捕获） | `Exception`（**会**被 `catch (Exception e)` 捕获） |
| 屏蔽取消 | `asyncio.shield(coro)` | 无内置对应，需手动管理中断标志 |

倒数第二行是 Python 设计得更好的一处。Python 3.8 起把 `CancelledError` 挪到了 `BaseException` 之下，所以下面这种到处都是的代码**不会**误吞取消信号：

```python
try:
    await do_work()
except Exception:            # CancelledError 不会被这里捕获
    logger.exception("failed")
```

Java 里对应的 `catch (Exception e) { log.error(...); }` **会**把 `InterruptedException` 一起吞掉，这是 JVM 生态里长期存在的坑（也是 `IOException` 包装 `InterruptedException` 之类问题的源头）。从 Java 迁过来的人可以在这里放松一点，但反过来写 Java 时要格外小心。

最后一行的 `asyncio.shield()` 是 Python 独有的实用工具：保护一段"启动了就必须做完"的操作不被外层超时取消，比如写入计费记录、提交事务。Java 侧只能靠手动保存和恢复中断标志。

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

对应 Java：这里有三个候选，选错了就是 bug。

**`ThreadLocal` —— 不是对应物。** `ThreadLocal` 绑定的是**物理线程**。在 asyncio 下，成千上万个协程跑在同一个线程里，用 `threading.local()` 意味着所有请求共享同一份数据，Trace ID 会互相覆盖。这正是 `contextvars`（PEP 567，Python 3.7 引入）存在的理由：

```python
import threading
from contextvars import ContextVar

wrong = threading.local()          # 整个事件循环共用一份，请求间会串
right: ContextVar[str] = ContextVar("request_id")   # 每个 Task 一份
```

**`InheritableThreadLocal` —— 更接近，但传播时机不同。** 它在**创建**子线程时复制父线程的值，之后各自独立。`asyncio.create_task()` 的行为几乎一样：Task 创建时通过 `contextvars.copy_context()` 拷贝一份当前上下文的**快照**，子任务里 `set()` 的值不会回流给父任务。

```python
request_id_var.set("req-1")
task = asyncio.create_task(handler())   # handler 内看到 "req-1"
# handler 里再 set，不影响这里
```

需要注意的是，Java 21+ 的虚拟线程**不推荐**用 `InheritableThreadLocal`：每个请求一个虚拟线程时，几十万份线程本地副本的内存开销是实打实的问题。这也是 `ScopedValue` 诞生的直接动机。

**`ScopedValue` —— 语义上最接近的对应物**，Java 21 预览、**JDK 25 正式转正**（JEP 506）：

```java
private static final ScopedValue<String> REQUEST_ID = ScopedValue.newInstance();

// 绑定只在这个作用域内有效，退出即自动解绑
ScopedValue.where(REQUEST_ID, requestId)
           .run(() -> handleRequest());

String id = REQUEST_ID.get();   // 未绑定时抛 NoSuchElementException
```

对照：

| | `contextvars.ContextVar` | `ThreadLocal` | `ScopedValue`（JDK 25 final） |
| :--- | :--- | :--- | :--- |
| 绑定粒度 | 协程 / Context | 物理线程 | 动态作用域（调用栈） |
| 可变性 | 可随时 `set()` | 可随时 `set()` | **不可变**，只能重新绑定 |
| 解绑方式 | `reset(token)`，需自己保证 | `remove()`，忘了就内存泄漏 | 作用域退出自动解绑 |
| 传给子任务 | `create_task()` 自动拷贝快照 | 不传播（`Inheritable` 版才传） | 传给 `StructuredTaskScope.fork()` 的子任务 |
| 未设置时 | 返回 `default` 或抛 `LookupError` | 返回 `null` 或 `initialValue()` | 抛 `NoSuchElementException` |
| 内存开销 | 每个 Task 一份引用（写时复制） | 每线程一份 | 共享，仅栈上一个绑定帧 |

两个实践层面的结论：

1. **`ScopedValue` 比 `ContextVar` 更严格**（不可变 + 强制作用域），Python 的 `set()`/`reset(token)` 更灵活但更容易忘记 `reset`。所以 Python 侧务必用 `try/finally` 包住，就像上面的例子那样。
2. **真实项目里通常不直接用它们。** OpenTelemetry 在两个生态里都做了封装：Python 的 `opentelemetry.context` 底层就是 `ContextVar`，Java 的 `io.opentelemetry.context.Context` 配 `try (Scope s = context.makeCurrent())`。日志侧，Python 用 `structlog.contextvars.bind_contextvars()`，Java 用 SLF4J 的 `MDC`（其实现正是 `ThreadLocal`，所以在 WebFlux/虚拟线程下同样有传播问题，需要 `MDCAdapter` 或 Reactor 的 `contextWrite`）。

```python
# 真实项目里更常见的形态（structlog）
import structlog

structlog.contextvars.bind_contextvars(request_id=request_id, model="llama-3.1-8b")
logger.info("inference started")   # 自动带上这两个字段
```

但它也不应该被滥用。业务上真正重要的数据，仍然应该通过明确的函数参数传递。ContextVar 更适合横切关注点，例如：

- 日志字段；
- tracing；
- 请求范围的鉴权信息；
- 事务或会话上下文。

这条建议和 Loom 团队对 `ScopedValue` 的定位完全一致：它是"隐式的方法参数"，只应该承载那些显式传递会污染每一层签名的横切数据。

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

对应 Java：`asyncio.Queue(maxsize=N)` 对应 `new ArrayBlockingQueue<>(N)` 或 `new LinkedBlockingQueue<>(N)`，方法一一对应，只是阻塞对象不同（线程 vs 协程）：

| 语义 | Python | Java |
| :--- | :--- | :--- |
| 满则等待 | `await queue.put(x)` | `queue.put(x)` |
| 满则立即失败 | `queue.put_nowait(x)` → `QueueFull` | `queue.offer(x)` → `false` |
| 满则限时等待 | `asyncio.wait_for(queue.put(x), t)` | `queue.offer(x, t, SECONDS)` |
| 空则等待 | `await queue.get()` | `queue.take()` |
| 空则限时等待 | `asyncio.wait_for(queue.get(), t)` | `queue.poll(t, SECONDS)` |
| 等待队列排空 | `await queue.join()` + `task_done()` | 无内置，需 `CountDownLatch` / `Phaser` |
| 无界队列 | `asyncio.Queue()`（默认无界） | `new LinkedBlockingQueue<>()`（默认 `Integer.MAX_VALUE`） |

两边的默认值都是**无界**，这也是两边同一个经典事故：`Executors.newFixedThreadPool()` 内部用的就是无界 `LinkedBlockingQueue`，任务堆积到 OOM 时线程池看起来一切正常；Python 里 `asyncio.Queue()` 不写 `maxsize` 是同样的结局。**两个语言里，写下队列容量都应该是肌肉记忆。**

`queue.join()` / `task_done()` 这一对是 Python 多出来的，用于"等所有已入队任务处理完"，Java 侧要自己用 `Phaser` 或 `CountDownLatch` 拼。

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

对应 Java：这份策略清单在 Java 里是**标准库内置的**——`ThreadPoolExecutor` 的 `RejectedExecutionHandler`：

| 背压策略 | Java 内置实现 | Python |
| :--- | :--- | :--- |
| 阻塞等待 | `CallerRunsPolicy`（由提交方自己跑，天然减速） | `await queue.put()` |
| 立即拒绝 | `AbortPolicy`（默认，抛 `RejectedExecutionException`） | `queue.put_nowait()` + 捕获 `QueueFull` |
| 静默丢弃 | `DiscardPolicy` | 自己写 |
| 丢弃最旧的 | `DiscardOldestPolicy` | 自己写 |
| 需求驱动背压 | Reactive Streams `Subscription.request(n)` | 异步生成器天然按需拉取 |

Python 没有这层抽象，所有策略都要在业务代码里手写。一个可用的模板：

```python
async def submit(request: dict) -> None:
    try:
        queue.put_nowait(request)          # 对应 AbortPolicy
    except asyncio.QueueFull:
        raise HTTPException(status_code=429, detail="server busy")
```

`CallerRunsPolicy` 特别值得一提：它是 Java 生态里被低估的背压神器——队列满时让提交线程自己执行任务，提交方自动被拖慢，压力沿调用链向上游传导。Python 里没有等价物，因为"提交方"是协程，让它同步执行任务反而会阻塞事件循环。

最后一行的 Reactive Streams 是概念上最完整的背压：消费者用 `request(n)` 声明自己能吃多少，需求信号逐级向上游传播。有意思的是，**Python 的异步生成器天然具备这个性质**——`async for` 每次迭代才驱动生成器产出一个元素，消费者不拉就不生产，不需要额外协议。这是第十一章的重点。

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

对应 Java：`java.util.concurrent.Semaphore`，语义几乎逐字对应：

```java
private final Semaphore semaphore = new Semaphore(10);

String limitedFetch(String url) throws Exception {
    semaphore.acquire();
    try {
        return client.send(request(url), BodyHandlers.ofString()).body();
    } finally {
        semaphore.release();        // Python 的 async with 帮你做了这件事
    }
}
```

| | Python | Java |
| :--- | :--- | :--- |
| 获取 | `await sem.acquire()` / `async with sem` | `sem.acquire()` |
| 非阻塞获取 | `sem.locked()` 只能查询，无 `try_acquire` | `sem.tryAcquire()` |
| 限时获取 | `asyncio.wait_for(sem.acquire(), t)` | `sem.tryAcquire(t, SECONDS)` |
| 释放 | `sem.release()` / 退出 `async with` | `sem.release()`（必须在 `finally`） |
| 公平性 | 先到先得（内部 FIFO 等待队列） | 默认**非公平**，`new Semaphore(n, true)` 才公平 |
| 可释放超额许可 | 可以（`BoundedSemaphore` 才会检查） | 可以（`Semaphore` 不校验） |

两点差异值得注意：

1. **Python 缺 `tryAcquire()`**。想实现"拿不到就立刻返回 429"，得用 `asyncio.wait_for(sem.acquire(), timeout=0)` 或者干脆用有界队列。
2. **公平性默认相反**。Java 的 `Semaphore` 默认非公平（吞吐更高但可能饿死），Python 的 `asyncio.Semaphore` 内部是 FIFO 等待队列，天然公平。做 GPU slot 分配这类场景时，Python 的默认行为通常正是你想要的。

顺带一提，真实项目里限流常常不需要自己写 Semaphore。`httpx` 内建了连接池限制：

```python
limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
async with httpx.AsyncClient(limits=limits, timeout=10.0) as client:
    ...
```

这对应 Java 里 `HttpClient` 的连接池配置，或 Resilience4j 的 `Bulkhead`。能用库自带的就别自己造。

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

对应 Java：**这里是最容易踩坑的一处，因为 Java 的两种锁都是可重入的。**

```java
synchronized void a() { b(); }     // 可以，同一线程重入
synchronized void b() { }

private final ReentrantLock lock = new ReentrantLock();   // 名字就写着 Reentrant
```

`asyncio.Lock` 不可重入，意味着这段在 Java 里天经地义的代码，在 Python 里是永久死锁：

```python
class ModelManager:
    async def predict(self, inputs: dict) -> dict:
        async with self._lock:
            return await self._infer(inputs)

    async def predict_batch(self, batch: list[dict]) -> list[dict]:
        async with self._lock:              # 已持有锁
            return [await self.predict(x) for x in batch]   # 死锁
```

正确做法是把临界区逻辑抽成一个**不加锁的私有方法**，由公开方法负责加锁——这在 Java 里也是好实践，只是 Python 里它是硬性要求。

完整对照：

| | Python | Java |
| :--- | :--- | :--- |
| 互斥锁 | `asyncio.Lock`（**不可重入**） | `synchronized` / `ReentrantLock`（可重入） |
| 线程版互斥锁 | `threading.Lock`（不可重入） | 同上 |
| 可重入版 | `threading.RLock`（**仅线程，无 asyncio 版**） | `ReentrantLock` |
| 读写锁 | 无内置 | `ReentrantReadWriteLock` / `StampedLock` |
| 条件变量 | `asyncio.Condition` | `Condition` / `Object.wait()` |
| 限时获取 | `asyncio.wait_for(lock.acquire(), t)` | `lock.tryLock(t, SECONDS)` |

注意第三行：`threading` 模块有 `RLock`，`asyncio` **没有**。以及第四行——模型热加载这类"多读少写"的场景在 Java 里会用 `ReentrantReadWriteLock`，Python 里得自己用 `Condition` 或计数器拼一个。

不过在 asyncio 下，你需要锁的场合比 Java 少得多：协程只在 `await` 处让出，所以任何**不含 `await` 的代码段天然是原子的**。上面 `ModelManager` 需要加锁，只是因为 `load_model` 和 `infer` 里有 `await`。纯粹的 `self.counter += 1` 在协程里不需要锁，而在 Java 多线程里必须用 `AtomicInteger`。

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

对应 Java：没有单一对应物，`asyncio.Event` 覆盖了 Java 里三个类的职责：

| 场景 | Python | Java |
| :--- | :--- | :--- |
| 一次性"就绪"信号 | `Event` + `set()`（不再 `clear()`） | `CountDownLatch(1)` 或 `CompletableFuture<Void>` |
| 可重复触发的开关 | `Event` + `set()` / `clear()` | 无内置；`Condition` 或 `Phaser` 手写 |
| 等待 N 件事完成 | `asyncio.gather()` / `queue.join()` | `CountDownLatch(N)` |
| 多方反复汇合 | 无内置 | `CyclicBarrier` / `Phaser` |

上面模型加载的例子，Java 里就是经典的 `CountDownLatch`：

```java
private final CountDownLatch modelReady = new CountDownLatch(1);

Response handleRequest(Request req) throws InterruptedException {
    modelReady.await();          // 等价于 await model_ready.wait()
    return predict(req);
}

void loadAndServe(String path) {
    loadModel(path);
    modelReady.countDown();      // 等价于 model_ready.set()
}
```

关键差异是 **`Event` 可以 `clear()` 而 `CountDownLatch` 不能重置**。这让 Python 的 `Event` 能直接表达"暂停/恢复"这类反复切换的开关（比如模型热更新期间挂起推理、恢复后放行），Java 里得用 `Phaser` 或自己拿 `Condition` 拼。

两边共有的坑：`set()` 之后再来的 `wait()` **立即返回**（因为 Event 是有状态的），这和 `Condition.signal()` / `notify()` 的"错过就永远错过"完全不同。想要"错过就等下一次"的语义，用 `asyncio.Condition`。

## 十、异步批处理：连接并发与 GPU 利用率

GPU 推理通常不是单个请求越快越好，而是需要在延迟和批量之间做权衡。

这是 AI-Infra 里最有价值的一个并发模式，主流框架都内置了。先看真实实现长什么样——Ray Serve 的 `@serve.batch`：

```python
from ray import serve


@serve.deployment
class Model:
    @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.1)
    async def __call__(self, samples: list[int]) -> list[int]:
        # 调用方传单个对象，这里收到的是一批
        return (np.array(samples) * 2).tolist()
```

调用方看到的仍然是"一次请求一个结果"，攒批完全透明。两个参数就是延迟/吞吐权衡的全部：`max_batch_size` 决定批量上限，`batch_wait_timeout_s` 决定第一个请求最多等多久。Triton Inference Server 的配置里是同一组概念，只是换了名字（`max_batch_size` 和 `max_queue_delay_microseconds`）。

vLLM 更进一步，用的是 **continuous batching**（也叫 in-flight batching）：不等整批推理完成才换批，而是每个 decode step 都重新组批，已完成的序列立即离开、新请求立即加入。这消除了传统静态批处理里"整批等最慢的那个"的浪费。

理解内部机制仍然重要——下面手写一个最小可用的动态批处理器：

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

对应 Java：同一个模式在 Java 里是"攒批线程 + `BlockingQueue` + `CompletableFuture`"，`drainTo` 让攒批代码比 Python 短得多：

```java
record Job<T, R>(T item, CompletableFuture<R> future) { }

private final BlockingQueue<Job<T, R>> queue = new ArrayBlockingQueue<>(1000);

void runBatchLoop() throws InterruptedException {
    while (true) {
        Job<T, R> first = queue.take();                    // 阻塞等第一个
        List<Job<T, R>> batch = new ArrayList<>(List.of(first));

        // 等待窗口内尽量多攒，攒不到就算了
        long deadline = System.nanoTime() + Duration.ofMillis(10).toNanos();
        while (batch.size() < maxBatchSize) {
            long remain = deadline - System.nanoTime();
            if (remain <= 0) break;
            Job<T, R> next = queue.poll(remain, TimeUnit.NANOSECONDS);
            if (next == null) break;
            batch.add(next);
        }

        try {
            List<R> results = inferBatch(batch.stream().map(Job::item).toList());
            for (int i = 0; i < batch.size(); i++) {
                batch.get(i).future().complete(results.get(i));   // 逐个回填
            }
        } catch (Exception e) {
            batch.forEach(j -> j.future().completeExceptionally(e));
        }
    }
}
```

结构逐行同构：

| 角色 | Python | Java |
| :--- | :--- | :--- |
| 入队等待 | `await queue.get()` | `queue.take()` |
| 限时补批 | `asyncio.wait_for(queue.get(), remaining)` | `queue.poll(remaining, NANOSECONDS)` |
| 批量抽干 | 无对应，只能循环 `get()` | **`queue.drainTo(batch, maxSize)`** |
| 单请求的结果句柄 | `loop.create_future()` | `new CompletableFuture<>()` |
| 回填结果 | `future.set_result(r)` | `future.complete(r)` |
| 回填异常 | `future.set_exception(e)` | `future.completeExceptionally(e)` |
| 调用方等待 | `await future` | `future.get()` / 虚拟线程里直接 `join()` |
| 攒批循环的载体 | 一个常驻 Task | 一个专用线程 |

`drainTo(collection, maxElements)` 是 Java 独有的便利：一次性把队列里现有元素全捞出来，不需要逐个 `get()`。Python 侧要模拟只能写 `while not queue.empty(): queue.get_nowait()`。

另一个差异是**跨线程回填的安全性**（第五章提过）：Java 的 `CompletableFuture.complete()` 线程安全，推理线程可以直接调；Python 里如果推理跑在别的线程，必须 `loop.call_soon_threadsafe(future.set_result, r)`。上面的 Python 实现之所以能直接 `set_result`，是因为整个循环都在事件循环线程里。

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

这正是 vLLM 暴露的接口形态——`AsyncLLMEngine.generate()` 返回一个异步生成器，逐步产出 `RequestOutput`：

```python
async def stream(prompt: str, request_id: str):
    results = engine.generate(prompt, SamplingParams(max_tokens=512), request_id)
    async for output in results:
        yield output.outputs[0].text
```

配 FastAPI 的 `StreamingResponse` 就是一个完整的 SSE 接口：

```python
from fastapi.responses import StreamingResponse


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    async def event_stream():
        async for chunk in stream(req.prompt, uuid4().hex):
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

对应 Java：`AsyncIterator` 对应 **Reactive Streams 的 `Publisher`**，实践中是 Reactor 的 `Flux` 或 RxJava 的 `Flowable`。同一个 SSE 接口在 WebFlux 里：

```java
@PostMapping(value = "/v1/completions", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
public Flux<String> completions(@RequestBody CompletionRequest req) {
    return engine.generate(req.prompt())
                 .map(chunk -> "{\"text\":\"" + chunk + "\"}");
}
```

对照：

| | Python | Java（Reactor） | Java 21+（虚拟线程） |
| :--- | :--- | :--- | :--- |
| 流类型 | `AsyncIterator[T]` | `Flux<T>` | 阻塞 `Iterator<T>` / `Stream<T>` |
| 生产 | `yield` | `Flux.create` / `generate` | 直接 return |
| 消费 | `async for` | `.subscribe()` / `.map()` | `for (T x : it)` |
| 背压 | **拉取式，天然背压** | `request(n)`，显式协议 | 阻塞即背压 |
| 组合算子 | 无（用 `async for` + 普通代码） | 极其丰富（`flatMap`、`window`、`buffer`…） | Stream API |
| 清理钩子 | `finally` / `aclosing()` | `doFinally` / `onCancel` | `try`-with-resources |

**背压那一行是最重要的差异**。Reactive Streams 需要一整套 `request(n)` 协议来表达"消费者还能吃多少"，而 Python 的异步生成器是**拉取式**的：`async for` 不迭代下一轮，生成器就停在 `yield` 那一行不动。消费者慢，生产者自动跟着慢，不需要任何协议。

这也是为什么 Python 生态里几乎没有出现 Reactive Streams 那样的库——async/await + 异步生成器已经在语言层面免费提供了背压。Java 到了虚拟线程时代其实也在往回走：能用阻塞 `Iterator` 就不用 `Flux`，因为阻塞本身就是最朴素的背压。

代价是 Python **没有 Reactor 那套算子**。`Flux.buffer(100)`、`window(Duration.ofSeconds(1))`、`retryWhen(...)` 这些在 Python 里都要手写循环。做流式聚合、限速、窗口化时会明显感到缺失。

流式处理有三个关键问题。

### 1. 客户端断开连接

客户端关闭连接后，应取消后续推理或生成任务，否则 GPU 仍然可能继续消耗资源。

这在 LLM 服务里是实打实的成本问题：用户关掉浏览器，后端还在为一个没人要的回答烧 GPU。vLLM 的做法是显式中止：

```python
async def event_stream(request: Request, request_id: str):
    try:
        async for chunk in stream(prompt, request_id):
            if await request.is_disconnected():
                await engine.abort(request_id)      # 通知调度器丢弃这个序列
                return
            yield chunk
    except asyncio.CancelledError:
        await engine.abort(request_id)
        raise
```

对应 Java：Reactor 里是 `doOnCancel(() -> engine.abort(id))`——下游 `Subscription.cancel()` 会沿链路向上游传播，语义和 `CancelledError` 沿 `await` 链传播完全一致。虚拟线程 + `HttpServletResponse` 写法下则表现为写操作抛 `IOException`。

### 2. 流速不匹配

如果服务端生成速度高于客户端读取速度，发送缓冲区会逐渐增大。因此，发送操作也必须是可等待的，并且要设置合理的缓冲上限。

对应 Java：这就是 Reactive Streams 存在的全部理由。Reactor 里可以显式选策略——`onBackpressureBuffer(1000)`、`onBackpressureDrop()`、`onBackpressureLatest()`。Python 侧因为是拉取式的，只要**全链路都是 `async for` + `await send()`**，背压自动成立；一旦中间插了一个无界 `asyncio.Queue` 做解耦，背压就断了。这是 Python 流式代码里最隐蔽的一类 bug：链路看起来是异步的，但某个环节的队列悄悄吸收了所有压力。

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

上面代码里 `.to("cuda")` 和 `.cpu()` 都是**同步阻塞**的拷贝。真实项目会用 pinned memory + 异步拷贝把传输和计算重叠起来：

```python
# 输入张量在 pinned memory 中（DataLoader 的 pin_memory=True 会做这件事）
inputs = inputs.pin_memory()

with torch.no_grad():
    # non_blocking=True：拷贝提交到 CUDA stream 后立即返回，不等完成
    gpu_inputs = inputs.to("cuda", non_blocking=True)
    outputs = model(gpu_inputs)
    result = outputs.to("cpu", non_blocking=True)

torch.cuda.synchronize()   # 显式等待 stream 上所有操作完成
```

`non_blocking=True` 只在源张量位于 pinned memory 时才真正异步；普通可分页内存上它会静默退化为同步拷贝。

对应 Java：**这一整章在 Java 生态里没有主流对应场景**，但底层机制有直接类比。

Python 调 PyTorch，本质上是通过 C 扩展（pybind11）进入 native 层；Java 调 native 的路径是 JNI，或者 Java 22 起转正的 **Panama FFM API**（`java.lang.foreign`，JEP 454）。两边共有的约束是一致的：

| 关注点 | Python + PyTorch | Java + JNI / Panama |
| :--- | :--- | :--- |
| 进入 native | pybind11，调用期间释放 GIL | JNI / `Linker.downcallHandle()` |
| 阻塞式 native 调用 | 阻塞整个事件循环 | 阻塞该线程；**虚拟线程会被 pin** |
| 内存所有权 | Tensor 的 refcount + CUDA caching allocator | `Arena` 管理 `MemorySegment` 生命周期 |
| 零拷贝传数据 | `__cuda_array_interface__` / DLPack | `MemorySegment` / `ByteBuffer.allocateDirect` |

注意第二行：Java 虚拟线程在执行 native 帧时**仍然会被 pin 到载体线程**（JEP 491 只解决了 `synchronized`，没有也无法解决 native 帧）。所以"长时间的 native 调用必须放到专用线程池，不要占着调度线程"这条规则，在两个生态里字面相同——Python 是别占事件循环，Java 是别占虚拟线程的载体。这也是为什么上面那段 `run_in_executor` 的写法，在 Java 里的对应物是 `Executors.newFixedThreadPool(2)` 而不是虚拟线程池。

最后一条差异：Java 侧完全没有"GPU 异步"这层。CUDA stream 的语义（提交即返回、需要显式 `synchronize`）在 Java 生态里没有等价的心智负担，因为几乎没人在 JVM 上直接跑 GPU 推理。

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

在 FastAPI 里，这套生命周期由 `lifespan` 上下文管理器承载：

```python
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    monitor = asyncio.create_task(report_gpu_metrics())
    yield                                    # 服务运行期
    monitor.cancel()
    await asyncio.gather(monitor, return_exceptions=True)


app = FastAPI(lifespan=lifespan)
```

对应 Java：

| 语义 | Python | Java |
| :--- | :--- | :--- |
| 后台常驻任务 | `asyncio.create_task()` | `ScheduledExecutorService` / `@Scheduled` |
| 周期执行 | `while True: ... await sleep(5)` | `scheduleAtFixedRate(task, 0, 5, SECONDS)` |
| 生命周期钩子 | FastAPI `lifespan` | `@PostConstruct` / `@PreDestroy`、`SmartLifecycle` |
| 请求取消 | `task.cancel()` | `executor.shutdownNow()` |
| 等待收尾 | `await gather(t, return_exceptions=True)` | `executor.awaitTermination(30, SECONDS)` |
| 优雅停机 | `uvicorn` 捕获 SIGTERM 后跑 lifespan 收尾 | Spring Boot `server.shutdown=graceful` |

"`shutdownNow()` 之后必须 `awaitTermination()`"和"`cancel()` 之后必须 `await`"是同一条规则的两种写法。

一个 Python 特有的差异：`scheduleAtFixedRate` 是**固定速率**（上一轮超时会连续补跑），而手写的 `while True: work(); await sleep(5)` 是**固定延迟**（对应 `scheduleWithFixedDelay`）。想要固定速率语义得自己算下一次的绝对时刻。上面 `report_gpu_metrics` 用 `finally: await asyncio.sleep(5)` 是固定延迟，对监控场景来说这通常反而是更安全的选择——采集变慢时自动降频，不会雪崩式堆积。

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

**这条对 Java 程序员格外重要。** 在 Java 21+ 的虚拟线程里，直接调用阻塞的 `HttpClient.send()` 是**完全正确**的高并发写法，JDK 会自动卸载线程。同样的直觉搬到 Python 协程里，就是把整个服务卡死的线上事故。迁移时要建立的第一条肌肉记忆是：**Python 的运行时不会替你把阻塞调用变成非阻塞的。**

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

### 附：Java 与 Python 并发概念速查表

基准为 **Java 25 LTS** 与 **Python 3.13**。标注"preview"的 Java 特性需要 `--enable-preview`。

**执行单元**

| 概念 | Java 25 | Python 3.13 | 关键差异 |
| :--- | :--- | :--- | :--- |
| OS 线程 | `Thread`（平台线程） | `threading.Thread` | 语义相同 |
| 轻量执行单元 | 虚拟线程（Java 21 转正） | 协程 `async def` | Java 隐式让出，Python 只在 `await` 让出 |
| 线程池 | `Executors.newFixedThreadPool(n)` | `ThreadPoolExecutor(max_workers=n)` | Python 默认 `min(32, cpu+4)` |
| 无限线程池 | `newVirtualThreadPerTaskExecutor()` | 无对应 | Python 用 asyncio 顶替 |
| CPU 并行 | `parallelStream()` / `ForkJoinPool` | `ProcessPoolExecutor` | Python 需跨进程序列化 |
| 事件循环 | Netty `EventLoop` / Reactor | `asyncio` 事件循环 / `uvloop` | Python 单循环单线程 |
| 全局锁 | 无 | GIL | 决定了上面所有行的取舍 |

**任务与编排**

| 概念 | Java 25 | Python 3.13 | 关键差异 |
| :--- | :--- | :--- | :--- |
| 异步结果句柄 | `CompletableFuture<T>` | `asyncio.Task` / `Future` | Java 线程安全完成，Python 需 `call_soon_threadsafe` |
| 惰性 vs 急性 | 创建即执行 | 协程对象惰性，`create_task` 才调度 | Python 忘记 `await` 静默无操作 |
| 等待全部 | `CompletableFuture.allOf()` | `asyncio.gather()` | `gather` 直接返回结果列表 |
| 等待任一 | `anyOf()` | `asyncio.wait(FIRST_COMPLETED)` | 相同 |
| 按完成顺序 | `ExecutorCompletionService` | `asyncio.as_completed()` | 相同 |
| 顺序组合 | `thenApply` / `thenCompose` | 直接写 `await`，无需组合子 | Python 语法优势 |
| 结构化并发 | `StructuredTaskScope`（**preview**，JEP 505） | `asyncio.TaskGroup`（3.11 起**稳定**） | 成熟度 Python 领先 |
| 多重异常 | `addSuppressed()` | `ExceptionGroup` + `except*` | Python 有语言级支持 |

**生命周期与上下文**

| 概念 | Java 25 | Python 3.13 | 关键差异 |
| :--- | :--- | :--- | :--- |
| 取消信号 | `interrupt()` / `cancel(true)` | `task.cancel()` | 都是协作式 |
| 取消异常 | `InterruptedException`（`Exception` 子类） | `CancelledError`（**`BaseException`** 子类） | Python 不会被 `except Exception` 误吞 |
| 恢复取消信号 | `Thread.currentThread().interrupt()` | `raise` | 同一条铁律 |
| 屏蔽取消 | 无内置 | `asyncio.shield()` | Python 独有 |
| 超时 | `orTimeout()` / `get(t, unit)` | `asyncio.timeout()` | **Java 不取消上游，Python 会** |
| 上下文传递 | `ScopedValue`（**JDK 25 转正**，JEP 506） | `contextvars.ContextVar` | ScopedValue 不可变且强制作用域 |
| 线程本地 | `ThreadLocal` | `threading.local()` | asyncio 下**不能用**，会串请求 |
| 日志上下文 | SLF4J `MDC` | `structlog.contextvars` | MDC 底层是 ThreadLocal，有传播问题 |

**同步与流控**

| 概念 | Java 25 | Python 3.13 | 关键差异 |
| :--- | :--- | :--- | :--- |
| 互斥锁 | `ReentrantLock` / `synchronized` | `asyncio.Lock` | **Java 可重入，asyncio 不可重入** |
| 读写锁 | `ReentrantReadWriteLock` | 无内置 | Python 需手写 |
| 信号量 | `Semaphore`（默认非公平） | `asyncio.Semaphore`（FIFO 公平） | 公平性默认相反 |
| 非阻塞获取 | `tryAcquire()` / `tryLock()` | 无对应 | Python 需 `wait_for(..., 0)` |
| 一次性信号 | `CountDownLatch(1)` | `asyncio.Event` | Event 可 `clear()` 重置 |
| 等 N 件完成 | `CountDownLatch(N)` | `queue.join()` / `gather()` | 相同 |
| 有界队列 | `ArrayBlockingQueue(n)` | `asyncio.Queue(maxsize=n)` | 都**默认无界**，都是 OOM 常见来源 |
| 批量出队 | `drainTo(list, max)` | 无对应 | Java 便利 |
| 拒绝策略 | `RejectedExecutionHandler`（4 种内置） | 需手写 | Java 有 `CallerRunsPolicy` |
| 流式背压 | Reactive Streams `request(n)` | 异步生成器天然拉取式 | Python 不需要额外协议 |
| 流类型 | `Flux<T>` / `Publisher<T>` | `AsyncIterator[T]` | Reactor 算子远比 Python 丰富 |

**诊断工具**

| 目的 | Java 25 | Python 3.13 |
| :--- | :--- | :--- |
| 阻塞调用检测 | BlockHound（运行时拦截） | `asyncio` debug 模式（事后告警） |
| 静态扫描 | ArchUnit 等 | `flake8-async` |
| 线程/任务转储 | `jstack` / `Thread.dump_to_file` | `asyncio.all_tasks()` + `get_stack()` |
| 调度延迟 | JFR `jdk.VirtualThreadPinned` | 自建心跳协程测 loop lag |

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

最后给从 Java 迁移过来的读者三句话的总结：

1. **不要指望运行时替你兜底。** Java 21 之后，虚拟线程让"用同步代码写高并发"重新成立；Python 没有这条捷径，阻塞就是阻塞，`async` 的传染性是你必须接受的成本。
2. **概念对得上，默认值经常对不上。** 队列默认无界、锁不可重入、信号量默认公平、`CancelledError` 是 `BaseException`——每一条单独看都是小事，叠在一起就是事故。
3. **Python 在任务协作层反而更成熟。** `TaskGroup`、`ExceptionGroup`/`except*`、`asyncio.timeout()` 的取消作用域、异步生成器的天然背压，这几样在 Java 侧要么还在 preview，要么需要引入 Reactor 才有。别因为 GIL 就低估 asyncio 的表达能力。
