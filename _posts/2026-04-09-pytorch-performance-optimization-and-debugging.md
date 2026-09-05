---
layout: post
title: "PyTorch 深度实践（08）：性能优化与调试"
subtitle: "Performance Optimization and Debugging in PyTorch"
tags: [PyTorch, AI, AI-Infra]
catalog: true
---

> 本文是[《PyTorch 深度实践：从 Tensor 到深度学习运行时》](/deep-dive-into-pytorch.html)系列的第八篇（共十篇）。上一篇：[编译执行与图优化](/pytorch-compilation-and-graph-optimization.html)　下一篇：[分布式 PyTorch](/pytorch-distributed-training.html)

前面七篇建立了 PyTorch 的执行模型：Tensor 如何存储（第二篇），Autograd 如何记录反向（第三篇），算子如何分发到 Kernel（第五篇），编译器如何把多个算子融合成更少的 Kernel（第七篇）。每一篇都在某处留下一句"性能问题第八篇讨论"。

这一篇兑现这些承诺。它围绕一个问题展开：

> **如何判断一个 PyTorch 程序慢，以及如何定位它为什么慢？**

这个问题比"如何让它变快"更基础。AI-Infra 工作中大量的性能优化失败，不是因为不知道优化手段，而是因为**没有正确地测量**，或者**对着错误的瓶颈用力**：GPU 利用率只有 30% 时去优化 Kernel，显存碎片导致 OOM 时去减 batch，Python 开销占主导时去换更快的 cuBLAS 版本。

所以本文的重心是**性能模型**和**测量方法**，优化手段是模型推导出来的结论。

> **性能优化的第一步不是修改代码，而是回答"时间花在哪一层"。答错这个问题，之后所有努力都在错误的地方。**

本文的数字（延迟、带宽、利用率）除明确标注的硬件规格外均为示意，用于说明数量级和比例关系；具体值随硬件、驱动、PyTorch 版本变化。测量方法和分析框架比数字本身持久。


## 一、总览：一次 step 的时间去哪了

### 1. 两条时间线

理解 PyTorch 性能的起点是一个事实：**CPU 和 GPU 是两个独立运行的处理器，通过一条队列连接**。

```text
CPU 时间线   Python 解释 → Dispatcher → launch Kernel A → launch Kernel B → launch Kernel C → ...
                                              │                │                │
                                              ▼ 入队           ▼ 入队           ▼ 入队
GPU 时间线                                    ──[Kernel A]──[Kernel B]──────[Kernel C]──...
```

CPU 执行 Python 代码，走完第五篇的入口和分发，把 Kernel **提交**到 GPU 的队列（CUDA Stream）后**立即返回**，继续执行下一行 Python。GPU 从队列中按序取出 Kernel 执行。两者异步。

这意味着一次训练 step 的耗时由两条时间线中**较长的那条**决定，而且两条线上各有不同的瓶颈：

| 时间线 | 可能的瓶颈 | 典型表现 |
|---|---|---|
| CPU 侧 | Python 解释开销、Dispatcher 开销、Kernel launch 开销、数据加载 | GPU 空闲等待 CPU 提交，利用率低 |
| GPU 侧 | Kernel 计算受限、Kernel 访存受限 | GPU 忙，CPU 等待 GPU 完成 |
| 两者之间 | 同步点：CPU 必须等 GPU 结果才能继续 | 两条线都有空闲 |

### 2. 时间维度：五类瓶颈

把上表展开，一个 PyTorch 程序慢，几乎总是以下五类之一或其组合：

| 位置 | 类别 | 时间花在 | 判断依据 | 主要处方 |
|---|---|---|---|---|
| CPU 侧 | **Python-bound** | Python 解释、框架逻辑 | GPU 利用率低，CPU 100%，Profiler 中 Python 函数占主导 | 减少 Python 层操作次数、`torch.compile`、向量化 |
| CPU 侧 | **Launch-bound** | 提交大量小 Kernel | GPU 利用率低，Kernel 平均耗时几微秒，数量多 | 增大 batch、算子融合、CUDA Graphs |
| GPU 侧 | **Memory-bound** | 搬运数据 | Kernel 达到显存带宽上限，算力利用低 | 融合减少中间结果、降低精度、改变数据布局 |
| GPU 侧 | **Compute-bound** | 做算术 | Kernel 达到算力上限 | 用 Tensor Core（低精度）、更好的算法、更少的计算量 |
| 两侧之间 | **Sync-bound** | CPU 等 GPU 或 GPU 等 CPU | 时间线上两侧都有空洞，同步调用频繁 | 消除隐式同步点、异步数据传输、重叠通信与计算 |

同一个程序在不同条件下会落入不同类别：batch 很小时 launch-bound，batch 大了变 memory-bound，换成低精度后可能变 compute-bound。**优化就是不断把瓶颈从一类推到另一类，直到达到硬件上限或成本上限。**

### 3. 空间维度：显存

时间不是唯一的维度。显存决定了 batch 能开多大、模型能放多大——而 batch 大小反过来决定了上面五类中的哪一类是瓶颈。所以显存分析是性能分析的一部分，不是独立话题。

显存问题有自己的分类：

```text
真的不够      参数 + 梯度 + 优化器状态 + 激活值 超过物理显存
碎片          allocated 远小于 reserved，有空间但不连续
泄漏          allocated 随 step 单调增长
峰值          平均占用不高，某一时刻（如反向开始时）尖峰 OOM
```

两个维度之间可以交换：Activation Checkpointing 用时间换空间；混合精度同时省时间和空间，代价是数值精度。

### 4. 工具地图

每种工具只能观察模型中的某几层。用错工具，看到的是噪声。

| 工具 | 观察什么 | 看不到什么 |
|---|---|---|
| `time.time()` + `synchronize` | 端到端墙钟时间 | 时间花在哪一层 |
| `torch.utils.benchmark` | 单个操作的稳定耗时 | 整体程序的行为 |
| `torch.profiler` | CPU 算子 / GPU Kernel 时间线、调用栈、显存事件 | Kernel 内部为什么慢 |
| Nsight Systems | 系统级时间线：CPU 线程、CUDA API、Kernel、内存拷贝、NVTX 标记 | Kernel 内部为什么慢 |
| Nsight Compute | 单个 Kernel 的硬件指标：占用率、带宽利用、指令吞吐 | 程序整体 |
| `torch.cuda.memory_*` / memory snapshot | 显存分配的时间线与调用栈 | 时间性能 |

顺序通常是：先用 Profiler 或 Nsight Systems 确定**哪一类**瓶颈，再决定是否需要 Nsight Compute 深入**某个 Kernel**。绝大多数问题在第一步就能定位。

### 5. 诊断流程

所有性能工作遵循同一条流程，第五章的案例会完整走一遍：

```text
建立基线          可复现的测量脚本，固定输入、固定环境，记录数字
    ↓
设计正确性测试    改之前就要有：与参考实现比对，容差明确
    ↓
采集性能数据      Profiler / Nsight，看时间线而不是猜
    ↓
定位瓶颈          归入五类之一，确认是主要矛盾
    ↓
修改实现          一次只改一件事
    ↓
重新 Benchmark    同一脚本、同一环境
    ↓
确认没有回归      正确性测试通过；显存、精度、编译时间是否变差
```

### 6. 本文的章节安排

```text
二    度量与工具地图：归类之前必须先测对
      1  异步执行模型：正确计时的前提
      2  Benchmark 方法：怎么得到可信的数字
      3  Profiler 与 Nsight：怎么看出瓶颈属于哪一类

三    时间维度：五类瓶颈
      1  CPU 侧：Python-bound 与 Launch-bound
      2  GPU 侧：Memory-bound 与 Compute-bound（含处方：融合、低精度、数据布局）
      3  两侧之间：Sync-bound（含数据加载）

四    空间维度：显存

五    完整案例：一个 Transformer block 的训练 step
六    Java 对照
七    小结
```


## 二、度量与工具地图：归类之前必须先测对

### 1. 异步执行模型：正确计时的前提

#### 1.1 为什么 `time.time()` 会骗你

```python
import time
start = time.time()
y = model(x)                    # 提交几百个 Kernel 到队列，立即返回
elapsed = time.time() - start   # 测到的是"提交"耗时，不是"执行"耗时
```

如果 GPU 侧的工作比 CPU 侧多，`elapsed` 会远小于真实耗时；如果之后某处发生了同步，那一行代码会显得异常慢——但慢的不是它，是它在等前面所有 Kernel 完成。

这是性能分析中最常见的误读来源。修正方法是在计时前后显式同步：

```python
torch.cuda.synchronize()
start = time.time()
y = model(x)
torch.cuda.synchronize()        # 等待队列中所有 Kernel 完成
elapsed = time.time() - start
```

或者用 CUDA Event 在 GPU 时间线上打点，避免 CPU 参与：

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
y = model(x)
end.record()
torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end)
```

Event 记录的是 GPU 执行到该位置的时刻，因此测量的是 GPU 侧的真实区间，不受 CPU 提交速度影响。

#### 1.2 Stream：队列本身

CUDA Stream 是 GPU 的一条命令队列。同一 Stream 内的 Kernel 按提交顺序串行执行；不同 Stream 之间可以并发。PyTorch 默认所有操作提交到当前设备的默认 Stream，这保证了 `y = f(x); z = g(y)` 的顺序正确——`g` 的 Kernel 排在 `f` 的后面。

多 Stream 是第三章 §3 重叠传输与计算的基础。跨 Stream 的依赖需要用 Event 或 `wait_stream` 显式表达，否则会读到未完成的数据——这是多 Stream 最常见的 bug。

#### 1.3 同步是测量的边界

任何需要 CPU 拿到 GPU 数据的操作都会阻塞 CPU 直到队列排空——这就是同步。对测量而言，同步是**必要的**：没有同步就测不到 GPU 时间。对性能而言，同步是**有代价的**：它让两条时间线互相等待。哪些操作会隐式同步、代价何时显现，是第三章 §3 Sync-bound 的内容。这一章只需记住：**测量必须同步，且只在测量边界同步**。


### 2. Benchmark 方法：怎么得到可信的数字

#### 2.1 用 `torch.utils.benchmark`

手写计时循环需要处理 warmup、同步、多次采样、统计。`torch.utils.benchmark` 把这些封装好了：

```python
import torch.utils.benchmark as benchmark

x = torch.randn(4096, 4096, device="cuda")
w = torch.randn(4096, 4096, device="cuda")

t = benchmark.Timer(
    stmt="x @ w",
    globals={"x": x, "w": w},
    label="matmul 4096",
    description="fp32",
)
m = t.blocked_autorange(min_run_time=1.0)
print(m)
```

```text
matmul 4096: fp32
  Median: 8.21 ms
  IQR:    0.03 ms (8.20 to 8.23)
  122 measurements, 1 runs per measurement, 1 thread
```

它自动在每次测量前后 `synchronize`，自动决定运行次数直到累计时间达到 `min_run_time`，报告中位数和四分位距（IQR）而不是均值——性能数据通常是右偏分布（偶发的慢），中位数更稳健。

`Compare` 可以把多个 `Measurement` 排成表格对比不同实现、不同 dtype、不同 shape。

#### 2.2 warmup 为什么必须

第一次运行总是慢的，原因来自多层：

```text
CUDA 上下文初始化              首次使用 GPU 时创建，数百毫秒
cuBLAS / cuDNN 句柄与算法选择   首次调用时初始化，cuDNN 可能对多个算法测速（benchmark 模式）
Caching Allocator 预热          首次分配要向驱动申请，之后复用（第四章）
torch.compile 冷编译            第七篇：秒级到分钟级
Python 层的惰性初始化           模块属性缓存、参数展平等
```

Benchmark 必须先跑几次不计时的迭代把这些排除，否则测到的是初始化成本而不是稳态性能。`torch.utils.benchmark` 会做 warmup；手写循环要自己做。

#### 2.3 测什么：口径决定结论

同一个模型，可以报告很多种数字，结论可能相反：

| 口径 | 含义 | 陷阱 |
|---|---|---|
| 单次前向延迟 | batch 固定，一次 `model(x)` 耗时 | batch=1 时几乎总是 launch-bound，不反映训练性能 |
| 吞吐（samples/s） | 单位时间处理的样本数 | 大 batch 吞吐高，但延迟和显存也高 |
| 训练 step 时间 | 前向 + 反向 + 优化器更新 | 反向通常是前向的 2 倍，优化器对大模型不可忽略 |
| 端到端 epoch 时间 | 含数据加载、日志、checkpoint | 数据加载可能是真正的瓶颈 |
| MFU（模型 FLOPs 利用率） | 实际 FLOPs / 硬件峰值 FLOPs | 需要正确计算模型 FLOPs |

报告 Benchmark 时必须写明口径、batch、序列长度、dtype、硬件、PyTorch 版本。缺任何一项，数字不可比。

#### 2.4 正确性测试先于性能

优化必然改变实现。改之前就要有测试证明改之后结果仍然正确：

```python
def check(fn_new, fn_ref, *inputs, rtol=1e-3, atol=1e-3):
    out_new = fn_new(*inputs)
    out_ref = fn_ref(*inputs)
    torch.testing.assert_close(out_new, out_ref, rtol=rtol, atol=atol)
```

容差要有依据：fp32 实现之间的差异通常在 `1e-5` 量级；涉及 bf16 时 `1e-2` 是常见起点（第三章 §2.6 解释为什么）；融合改变了浮点结合顺序，`1e-6` 的容差几乎必然失败。训练场景还应比对若干 step 后的 loss 曲线，而不只是单次前向。

没有正确性测试的"加速"不是优化，是待发现的 bug。

#### 2.5 控制变量

- 固定随机种子和输入数据；
- 关闭其他 GPU 进程（`nvidia-smi` 确认）；
- 注意 GPU 时钟：连续高负载会触发降频，长 Benchmark 的后半段可能比前半段慢；
- `torch.backends.cudnn.benchmark = True` 会让 cuDNN 为每个新 shape 试跑多个算法——对固定 shape 有益，对动态 shape 有害，且首次调用极慢；
- 一次只改一个变量。


### 3. Profiler 与 Nsight：怎么看出瓶颈属于哪一类

Benchmark 告诉你"多慢"。要知道"为什么慢"，必须看时间线。

#### 3.1 `torch.profiler`

```python
from torch.profiler import profile, ProfilerActivity, schedule, record_function

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=2, active=3),      # 跳过 1 步，预热 2 步，记录 3 步
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
) as prof:
    for step, batch in enumerate(loader):
        with record_function("forward"):
            loss = model(batch).sum()
        with record_function("backward"):
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        prof.step()
        if step >= 6:
            break

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
```

它同时记录 CPU 侧的算子调用（第五篇的入口、分发）和 GPU 侧的 Kernel 执行，并把两者关联起来。`record_function` 在时间线上打自定义标记，`schedule` 避免记录 warmup。

#### 3.2 表格怎么读

`key_averages().table()` 输出类似：

```text
Name                          Self CPU %  Self CPU   CPU total  Self CUDA   CUDA total  # of Calls
aten::mm                           2.1%    1.2 ms     1.8 ms     42.3 ms     42.3 ms        96
aten::add_                         1.5%    0.9 ms     0.9 ms      8.1 ms      8.1 ms       384
aten::layer_norm                   0.8%    0.5 ms     2.1 ms      6.4 ms      6.4 ms        48
aten::gelu                         0.4%    0.2 ms     0.2 ms      4.9 ms      4.9 ms        24
cudaLaunchKernel                  38.2%   22.6 ms    22.6 ms      0.0 ms      0.0 ms      1840
...
Self CPU time total: 59.1 ms
Self CUDA time total: 71.5 ms
```

几个关键读法：

- **`Self CUDA time total` vs 墙钟时间**：如果 3 个 step 墙钟 210 ms、CUDA 总时间 71.5 ms，GPU 只忙了三分之一——瓶颈在 CPU 侧或同步。
- **`cudaLaunchKernel` 的 Self CPU**：1840 次 launch 花了 22.6 ms，平均 12 µs 一次，占 CPU 时间 38%——典型的 launch-bound 信号。
- **`# of Calls` 大、`Self CUDA` 小的算子**：`aten::add_` 384 次共 8.1 ms，每次 21 µs——小 Kernel，融合的候选。
- **`CPU total` 远大于 `Self CPU`**：说明这个算子内部调用了别的算子（第五篇的 Composite 路径），例如 `layer_norm`。

#### 3.3 时间线怎么读

导出的 trace 用 Chrome 的 `chrome://tracing` 或 Perfetto 打开，能看到两条（或更多）泳道：

```text
CPU 线程    ▓▓ aten::mm ▓▓ aten::add_ ▓ aten::gelu ▓▓ aten::mm ▓▓ ...    ← 密集，几乎无空隙
GPU Stream  ░░░[mm]░░░░░░[add]░░░░[gelu]░░░░░░░░[mm]░░░░░░░ ...          ← 稀疏，Kernel 之间大量空白
```

看一眼时间线的**形态**，比读任何数字都更快地判断瓶颈类型。

#### 3.4 Nsight Systems：系统级时间线

`torch.profiler` 看的是 PyTorch 眼中的世界。Nsight Systems（`nsys`）看的是操作系统眼中的世界：所有线程、CUDA API 调用、Kernel、内存拷贝、甚至 NCCL 通信（第九篇），以及 PyTorch 之外的进程。

```bash
nsys profile -t cuda,nvtx,osrt -o report python train.py
```

PyTorch 用 NVTX 在时间线上标记算子，也可以手动标记：

```python
torch.cuda.nvtx.range_push("attention")
...
torch.cuda.nvtx.range_pop()
```

用 Nsight Systems 而非 `torch.profiler` 的场景：怀疑瓶颈在 PyTorch 之外（数据预处理进程、文件 I/O、其他进程抢 GPU）；多 GPU 训练看通信与计算的重叠；需要看 CUDA 驱动层的行为（内存拷贝方向、Stream 依赖）。

#### 3.5 Nsight Compute：单个 Kernel 的内部

前两个工具回答"哪个 Kernel 慢"。Nsight Compute（`ncu`）回答"这个 Kernel **为什么**慢"：它重放单个 Kernel，采集硬件计数器。

```bash
ncu --set full --kernel-name regex:triton_poi_fused_add_relu -c 1 python demo.py
```

关键指标：

| 指标 | 含义 | 判断 |
|---|---|---|
| DRAM Throughput | 显存带宽利用率 | 接近 100% → memory-bound |
| SM Throughput / Compute Throughput | 算力利用率 | 接近 100% → compute-bound |
| Achieved Occupancy | 实际驻留的 warp 数 / 最大值 | 过低 → 无法隐藏延迟 |
| Registers / Shared Memory per Block | 资源占用 | 是限制 occupancy 的原因 |
| Memory Coalescing / L2 Hit Rate | 访存效率 | 低 → 访存模式差（如非连续 stride） |

这是最深的一层，只在已经确认某个 Kernel 是瓶颈、且要动手写或改 Kernel（第六篇）时才需要。理解它的指标需要第三章 §2 的性能模型。

#### 3.6 从形态到归类

本章到此结束。把测量结果对应到五类瓶颈，是进入第三章的入口：

| 时间线形态 | Profiler 表格特征 | 归类 | 章节 |
|---|---|---|---|
| CPU 泳道密集无空隙，GPU 泳道稀疏、Kernel 短 | `cudaLaunchKernel` 占 CPU 大头；Kernel 数多、平均几微秒 | Launch-bound | 三 §1 |
| CPU 泳道密集，但大部分时间不在算子上 | Python 函数、`DataLoader`、框架逻辑占 CPU 大头 | Python-bound | 三 §1 |
| GPU 泳道密集首尾相接，逐元素算子占 CUDA 时间大头 | `add`、`mul`、`layer_norm`、`softmax` 等累计占比高 | Memory-bound | 三 §2 |
| GPU 泳道密集，矩阵乘占 CUDA 时间大头 | `mm`、`bmm`、`conv` 占比高 | Compute-bound | 三 §2 |
| 两条泳道周期性交替空洞 | `cudaStreamSynchronize`、`cudaMemcpy` 频繁出现 | Sync-bound | 三 §3 |
| GPU 泳道大段空白，CPU 停在数据加载 | `DataLoader.__next__` 耗时长 | Sync-bound 的特例：数据加载 | 三 §3 |


## 三、时间维度：五类瓶颈

### 1. CPU 侧：Python-bound 与 Launch-bound

CPU 侧的两类瓶颈有同一个症状——**GPU 在等 CPU**——但原因不同：Python-bound 是 CPU 在做与 Kernel 提交无关的事；Launch-bound 是 CPU 在提交 Kernel，但每个 Kernel 太小，提交的固定成本超过了 GPU 执行它的时间。

#### 1.1 每个算子在 CPU 上的固定成本

回到第五篇：一次 `torch.add(x, y)` 在 CPU 上要经过 Python Binding 解析参数、Dispatcher 合并 DispatchKeySet 并查表、Autograd 包装记录反向节点、TensorIterator 构造、最后 `cudaLaunchKernel`。这一串在现代 CPU 上大约需要**10～30 µs**，与 Tensor 大小无关。

对比 GPU 侧：对 `[128, 64]` 的 fp32 Tensor 做一次 `add`，Kernel 执行时间约 3～5 µs。CPU 提交它的时间是执行时间的几倍。此时 GPU 大部分时间空转，等下一个 Kernel 到达。

```text
CPU    |--提交 add 20µs--|--提交 relu 20µs--|--提交 mm 25µs--|
GPU         |add 4µs|          |relu 3µs|         |mm 8µs|         ← 利用率 ~25%
```

**只要单个 Kernel 的 GPU 时间低于 CPU 提交它的时间，就是 launch-bound**。这个阈值大约在 10～20 µs。

#### 1.2 小 Kernel 从哪来

```text
小 batch / 小 shape           每个算子处理的数据少；推理 batch=1 几乎必然 launch-bound
逐元素算子链                   LayerNorm → 残差加 → GeLU → Dropout → cast，每个一到几个 Kernel
优化器更新                     每个参数 Tensor 一组 Kernel；100 个参数 Tensor × Adam 的 ~10 个算子 = 1000 次 launch
Python 循环中的 Tensor 操作     for i in range(seq_len): 每次迭代 launch 几个 Kernel
autocast 的类型转换            每次进出低精度区域插入 cast Kernel
```

#### 1.3 Python-bound：不在提交上的 CPU 时间

Launch-bound 的 CPU 至少在提交 Kernel。Python-bound 的 CPU 在做别的：

```text
Python 解释                    纯 Python 的循环、条件、字典操作、对象构造
框架逻辑                        nn.Module 的 __call__ 钩子、参数遍历、shape 检查
数据预处理在主进程              tokenize、augmentation、collate 在训练循环里同步执行
日志与监控                      每步 .item()、格式化字符串、写 TensorBoard
```

Profiler 里的信号是：CPU 泳道密集，但顶部的耗时项不是 `aten::*` 算子，而是 Python 函数或 `DataLoader`。`with_stack=True` 能看到具体是哪行代码。

#### 1.4 处方一：增大 batch，让 Kernel 变大

Launch-bound 的本质是 Kernel 的 GPU 时间低于 CPU 提交时间。最直接的解法是让每个 Kernel 处理更多数据：batch 从 8 到 64，`add` Kernel 的执行时间从 4 µs 变成 30 µs，超过了提交它的 20 µs，GPU 不再空转。

```text
CPU    |--提交 add 20µs--|--提交 relu 20µs--|
GPU         |------ add 30µs ------|------ relu 25µs ------|     ← GPU 成为长边，CPU 提交被隐藏
```

这一步不减少 Kernel 数量，只是让 GPU 时间线变长直到超过 CPU 时间线。前提是显存允许（第四部分）。推理场景 batch 由请求决定，对应的手段是**动态 batching**——把时间窗口内到达的多个请求合并成一个 batch 执行。

它也是最应该先做的一步：batch 太小时，后面所有 GPU 侧优化的效果都测不出来，因为 GPU 本来就在等。

#### 1.5 处方二：融合，减少 Kernel 数量

第七篇讨论的算子融合在这里的价值是 CPU 侧的：N 个逐元素 Kernel 融合成 1 个，提交成本从 N × 20 µs 变成 20 µs。

```python
compiled_model = torch.compile(model)      # Inductor 把 LayerNorm → 残差 → GeLU → cast 链融合
```

对第五章案例里的 Transformer block，`torch.compile` 把每 step 的 Kernel 数从 2100 降到 640。融合同时也是 GPU 侧 memory-bound 的处方（§2.4），所以它在两侧都有收益——这是 `torch.compile` 对小模型也常常有效的原因，即使小模型的 GPU 侧根本不是瓶颈。

#### 1.6 处方三：fused 优化器，合并参数更新

优化器是 launch-bound 的高发区，且常被忽略。一个有 100 个参数 Tensor 的模型，朴素的 Adam 实现每个参数要做约 10 个逐元素操作：

```text
100 个参数 × 10 个操作 = 1000 次 launch，每次处理一个参数 Tensor（可能只有几千个元素）
```

PyTorch 提供两级合并：

```python
torch.optim.AdamW(params, foreach=True)    # 默认：把同一操作对所有参数的调用合并成一次多 Tensor Kernel
torch.optim.AdamW(params, fused=True)      # 更进一步：整个 Adam 更新一个 Kernel，只支持 CUDA 上的浮点参数
```

`foreach` 把 1000 次 launch 降到约 10 次（每个操作一次，内部遍历所有参数）；`fused` 降到个位数。对参数 Tensor 多而小的模型，优化器从 step 时间的 20% 降到 2% 是常见的。

#### 1.7 处方四：向量化，消除 Python 循环

Python 循环里的 Tensor 操作是 launch-bound 和 Python-bound 的叠加：每次迭代既有解释开销，又有几次 launch。

```python
# 慢：seq_len 次迭代，每次 3 个 Kernel，seq_len=512 时 1500 次 launch
out = []
for t in range(seq_len):
    h = torch.tanh(x[:, t] @ W + h @ U)
    out.append(h)

# 快：能并行的部分一次算完
xW = x @ W                                  # [B, T, H]，一个大 mm 代替 T 个小 mm
for t in range(seq_len):
    h = torch.tanh(xW[:, t] + h @ U)        # 循环里只剩真正有时序依赖的部分
```

真正的循环依赖（RNN 的 `h @ U`）无法消除，但循环体外的部分应全部提前批量计算。更彻底的做法是换成没有时序依赖的结构（Transformer 取代 RNN 的动机之一就是这个），或用 `torch.compile` 让 Inductor 把循环体融合成一个 Kernel。

Python 侧的其他细节同理：`x.shape[0]` 比 `x.size(0)` 慢一点、`torch.tensor(...)` 在循环里反复构造、用 Python 标量参与 Tensor 运算触发隐式转换——单个微不足道，在每 step 几千次的循环里累积起来可观。`with_stack=True` 的 Profiler 能定位到行。

#### 1.8 处方五：CUDA Graphs，消除提交本身

前四条处方减少 Kernel 数量或增大 Kernel。CUDA Graphs 走另一条路：**不减少 Kernel，但让提交它们几乎免费**。

原理是把一段固定的 Kernel 序列（含参数、依赖关系）录制成一张图，之后每次只需一次 `cudaGraphLaunch` 就能整体重放，CPU 侧成本从"每 Kernel 十几微秒"降到"每图几微秒"。GPU 侧也受益：图内 Kernel 之间的调度间隙更小。

```python
compiled_model = torch.compile(model, mode="reduce-overhead")     # Inductor 自动录制并重放
```

手动用 `torch.cuda.CUDAGraph` 也可以，但需要自己处理静态输入 buffer 和 warmup。

限制来自"录制"这个前提：图内的 shape、控制流、内存地址都必须固定。动态 shape 会触发重新录制；图内不能有 CPU 同步（第三节）；每张图占用一份固定的显存池。它最适合**推理**——shape 固定、无反向、launch 占比高——是 LLM 推理引擎的标配。

#### 1.9 处方六：把非计算工作移出训练循环

Python-bound 的处方与 launch 无关，是把 CPU 在做的"别的事"挪走：

```text
数据预处理     交给 DataLoader 的 worker 进程（第三节 §4），主进程只负责取现成的 batch
日志与监控     每 N 步一次；.item() 批量取回（第三节 §3）；TensorBoard 写入放到后台线程
框架开销       推理用 torch.inference_mode() 而非 no_grad()，跳过更多 Autograd 簿记；
              torch.compile 消掉 nn.Module 的 __call__ 钩子、参数遍历、shape 检查
Python 对象    避免在循环里构造 Tensor、dict、字符串；预先分配、复用
```

一个判断标准：训练循环的每次迭代里，除了 `model(x)`、`loss.backward()`、`optimizer.step()` 之外的所有代码，都应该问一句"它必须在这里、每步都做吗"。

#### 1.10 处方的顺序

```text
增大 batch          → 先做，否则后面的效果测不出来
fused 优化器        → 一行改动，零风险
torch.compile 融合   → 收益大，代价是编译时间和 shape 约束
向量化              → 需要改代码结构
CUDA Graphs         → 约束最多，收益在前几条做完后才明显
移出非计算工作      → 与上面并行进行
```

#### 1.11 何时停止

当 Profiler 显示 GPU 利用率超过 90%、`cudaLaunchKernel` 占 CPU 时间不到 10%，CPU 侧不再是瓶颈。此时继续融合、继续减 launch 已无收益，瓶颈转移到 GPU 侧——进入下一节。


### 2. GPU 侧：Memory-bound 与 Compute-bound

GPU 已经满负荷，问题变成：它在忙什么？GPU 执行一个 Kernel 只做两件事——搬数据、做算术。哪一件先到上限，Kernel 就受哪一件限制。

#### 2.1 一个 Kernel 的时间下界

从显存读入数据、写出结果是**访存**；对数据做算术是**计算**。两者在硬件上由不同单元执行，可以重叠。因此一个 Kernel 的时间下界是：

```text
T ≥ max( 数据量 / 显存带宽 ,  运算量 / 峰值算力 )
```

哪一项更大，Kernel 就受哪一项限制。这就是 memory-bound 与 compute-bound 的定义。

#### 2.2 Arithmetic Intensity 与 Roofline

把两项的比值定义为 **Arithmetic Intensity**（算术强度）：

```text
AI = 运算量 (FLOPs) / 数据量 (Bytes)
```

硬件也有一个对应的比值：峰值算力 / 显存带宽，称为 **ridge point**。以 A100 为例（取整）：

```text
FP32 算力       19.5 TFLOPs
FP16/BF16 算力  312 TFLOPs（Tensor Core）
显存带宽        2 TB/s
ridge point     FP32:  19.5e12 / 2e12 ≈ 10 FLOP/Byte
                FP16:  312e12 / 2e12 ≈ 156 FLOP/Byte
```

Kernel 的 AI 低于 ridge point → memory-bound，能达到的算力 = AI × 带宽；高于 → compute-bound，能达到的算力 = 峰值。画成图就是 Roofline：横轴 AI，纵轴可达算力，一条斜线接一条水平线。

#### 2.3 把第七篇的三个算子归类

第七篇的 `f` 里有 `mm`、`add`、`relu`。用 fp32、`x: [128, 32]`、`weight: [32, 64]`：

| 算子 | 运算量 | 数据量 | AI | 类别 |
|---|---|---|---|---|
| `mm` | 2 × 128 × 32 × 64 = 524k FLOPs | (128×32 + 32×64 + 128×64) × 4 B ≈ 57 KB | ≈ 9 | 接近 ridge point，但规模太小，实际为 launch-bound（§1） |
| `add` | 128 × 64 = 8k FLOPs | 3 × 128 × 64 × 4 B ≈ 98 KB | ≈ 0.08 | 严重 memory-bound |
| `relu` | 8k 次比较 | 2 × 128 × 64 × 4 B ≈ 65 KB | ≈ 0.12 | 严重 memory-bound |

换成大模型尺度，`mm` 为 `[4096, 4096] × [4096, 4096]` fp16：

```text
运算量  2 × 4096³ ≈ 137 GFLOPs
数据量  3 × 4096² × 2 B ≈ 100 MB
AI      ≈ 1370  ≫ 156   → compute-bound
```

结论：

- **逐元素算子永远是 memory-bound**。它们的 AI 在 0.1 量级，与硬件 ridge point 差两到三个数量级。让它们变快的唯一途径是**减少访存**。
- **大矩阵乘是 compute-bound**。让它变快的途径是用更高算力的单元，或减少计算量。
- **小矩阵乘两头不靠**，规模不足以填满 GPU，实际受 launch 开销和低 occupancy 限制。回到 §1。

归约类算子（`sum`、`softmax`、`layer_norm`）介于两者之间但更接近 memory-bound：每个元素读一次，做常数次运算。Attention 的 `softmax(q @ k.T) @ v` 手写时 score 矩阵要写出再读回，是 memory-bound 的典型；融合实现（FlashAttention）不物化它，把整个 attention 变成 compute-bound。

#### 2.4 处方一：融合，减少访存

对 memory-bound 算子，融合的收益是精确可算的。对 `N` 个元素的 `add` + `relu`：

```text
分开    add: 读 2N 写 N    relu: 读 N 写 N      合计 5N 次访存
融合    读 N + bias，写 N                       合计约 2N 次访存
```

时间几乎按访存量线性下降。第七篇讨论了编译器如何自动融合逐元素链；本章要补充的是**算法级融合**——编译器发现不了、需要人知道存在的融合算子：

```text
F.scaled_dot_product_attention     不物化 attention score 矩阵
融合的 LayerNorm / RMSNorm          归约与归一化一个 Kernel
融合的优化器（fused=True）          所有参数的 Adam 更新一个 Kernel
```

这类算子的收益往往比编译器融合大一个量级，因为它们改变的是算法的访存模式，不只是消掉中间结果。

#### 2.5 处方二：低精度，同时提升算力上限和降低数据量

对照 Roofline，降低精度同时移动两条线：

```text
Compute-bound 的矩阵乘   Tensor Core 的 bf16 算力是 fp32 的 16 倍，水平线上移
Memory-bound 的逐元素    bf16 数据量减半，斜线上的位置右移，时间大致减半
Launch-bound             没有作用，Kernel 数量不变；autocast 插入的 cast Kernel 反而增加 launch
```

所以低精度对 launch-bound 的小模型几乎无效甚至变慢。**先解决 §1 的问题，再上低精度**。

三种低精度格式：

| 格式 | 指数位 | 尾数位 | 范围 | 精度 | 用途 |
|---|---|---|---|---|---|
| FP32 | 8 | 23 | ~1e±38 | ~7 位十进制 | 基线；主参数、优化器状态、归约 |
| TF32 | 8 | 10 | 与 FP32 相同 | ~3 位 | Tensor Core 内部格式，fp32 矩阵乘的免费加速 |
| FP16 | 5 | 10 | ±65504，最小正规数 6e-5 | ~3 位 | 需 GradScaler 防下溢 |
| BF16 | 8 | 7 | 与 FP32 相同 | ~2 位 | 大模型训练默认 |

- **TF32** 不是存储格式：输入输出仍是 fp32 Tensor，矩阵乘内部把尾数截到 10 位。`torch.set_float32_matmul_precision("high")` 开启。对用户几乎透明，代价是矩阵乘精度降到 fp16 水平。
- **FP16** 指数位少，**容易溢出和下溢**：梯度小于 6e-5 归零，激活值大于 65504 变 inf。第四篇的 `GradScaler` 为它存在：把 loss 放大后反向，让小梯度不下溢。
- **BF16** 范围与 fp32 相同，不需要 GradScaler，但尾数只有 7 位——相近的数相减会损失大部分有效位。它是当前大模型训练的默认，因为范围问题比精度问题更难处理。

第四篇讲过 `autocast` 的用法。它不是把所有 Tensor 变成 bf16，而是按算子分类：

```text
转到低精度      matmul、conv、linear、bmm        ← compute-bound，受益最大，对精度最不敏感
保持 fp32       softmax、log_softmax、layer_norm、loss、sum、exp、pow   ← 涉及大范围或累加
跟随输入        add、mul、relu 等逐元素算子        ← 输入是什么就算什么
```

因此在 autocast 下，Profiler 里会出现大量 `aten::to` / `aten::_to_copy`——精度转换 Kernel。它们是逐元素 memory-bound 算子，`torch.compile` 可以把它们融合进相邻算子。

#### 2.6 处方二的代价：数值稳定性

低精度不是免费的。它放大了三类数值问题，这些问题在 fp32 下往往被掩盖：

**溢出**。Attention score `q @ k.T / sqrt(d)` 在长序列、大 hidden 下容易超过 fp16 范围。softmax 前减去最大值是标准做法，PyTorch 的 `softmax` 内部已经这样做，但手写的 `exp(x) / exp(x).sum()` 没有。

**下溢与精度丢失**。fp16 梯度小于 6e-5 归零，深层网络的早期层梯度容易落入此区间。bf16 下 `x + delta` 当 `delta < x / 128` 时直接等于 `x`——优化器更新 `lr × grad` 相对参数值极小，用 bf16 存参数会导致更新丢失，所以主参数必须是 fp32。

**累加误差**。对 N 个 bf16 数求和，如果累加器也是 bf16，误差随 N 增长。PyTorch 的 CUDA 归约 Kernel 内部用 fp32 累加，矩阵乘的 Tensor Core 同样以 fp32 累加，但用户手写的循环累加不会。

几个规则：

```text
loss 计算保持 fp32；用 F.cross_entropy(logits) 而不是 log(softmax(logits))
归约（sum、mean、norm）在 fp32 中做，或确认算子内部已用 fp32 累加
优化器状态和主参数保持 fp32
对比 bf16 与 fp32 的 loss 曲线，而不只是单步输出
出现 NaN 时用 torch.autograd.detect_anomaly() 定位第一个 NaN 的算子
```

这也解释了第二章 §2.4 的容差依据：bf16 有效精度约 2～3 位十进制，实现之间的差异在 `1e-2` 量级是正常的，`1e-5` 的容差对 bf16 没有意义。

什么时候不该用低精度：模型很小、launch-bound（没有收益，有 cast 开销）；数值敏感的任务；推理时精度要求高于速度；还没有 fp32 基线和正确性测试。

#### 2.7 处方三：数据布局

第二篇讲过 stride 和连续性。对 memory-bound 的 Kernel，访存模式直接决定实际带宽：连续访问能合并成宽事务，跨 stride 的访问浪费带宽。

```python
x = torch.randn(4096, 4096, device="cuda")
y = x.t()                    # view，非连续
z = y + 1                    # TensorIterator 按 stride 遍历，访存不合并，比连续情形慢数倍
z = y.contiguous() + 1       # 先复制成连续（一次额外访存），再快速逐元素
```

哪种更快取决于后续用几次：用一次时直接遍历，用多次时先 `contiguous()`。`channels_last` 内存格式对卷积网络的加速也是同一原理——让 cuDNN 的访存模式与数据布局匹配。

#### 2.8 Occupancy：为什么达不到 Roofline

GPU 靠大量并发线程隐藏访存延迟：一个 warp 等数据时，调度器切换到另一个 warp。能同时驻留多少 warp 称为 **Occupancy**，受每个线程的寄存器数、每个 Block 的共享内存、Block 大小限制。

Occupancy 低的 Kernel 即使 AI 合适也达不到 Roofline，因为访存延迟暴露出来了。这是第二章 §3 Nsight Compute 里 "Achieved Occupancy" 指标的意义。它只在自己写 Kernel 时（第六篇）需要直接关心；用 PyTorch 原生算子和 Inductor 生成的 Kernel 时，这一层通常已经处理好。

#### 2.9 何时停止

当矩阵乘 Kernel 的 Tensor Core 利用率接近峰值、逐元素算子已融合到不能再融合、Nsight Compute 显示主要 Kernel 的 DRAM 或 SM Throughput 接近 100%——GPU 侧到了硬件上限。剩下的路只有减少计算量本身（模型结构、序列长度、稀疏化），或增加硬件（第九篇）。


### 3. 两侧之间：Sync-bound

前两章各自假设另一侧不是问题。Sync-bound 是两侧**互相等待**：CPU 等 GPU 排空，然后 GPU 等 CPU 重新填队列。时间线上的形态是两条泳道**交替出现空洞**。

#### 3.1 隐式同步点

显式的 `synchronize()` 容易发现。危险的是**隐式**同步：任何需要 CPU 拿到 GPU 数据的操作都会阻塞 CPU，直到队列排空。

| 操作 | 为什么同步 |
|---|---|
| `tensor.item()`、`float(tensor)`、`int(tensor)` | 把一个值传回 CPU |
| `tensor.cpu()`、`tensor.numpy()`、`tensor.tolist()` | 把数据传回 CPU |
| `print(tensor)` | 内部调用 `.cpu()` |
| `if tensor > 0:` | 需要 `bool`，即 `.item()` |
| `tensor[mask]`（布尔索引）、`torch.nonzero` | 输出 shape 依赖数据，CPU 必须知道 shape 才能分配 |
| `torch.cuda.empty_cache()` | 等待所有使用中的 Block 释放 |
| 非 pinned memory 的 `.to("cuda")` | 同步拷贝 |

训练循环里一行看似无害的 `loss.item()` 累加日志，每个 step 都会把 CPU 拖住等 GPU 排空，然后 GPU 再等 CPU 重新提交。

#### 3.2 同步的代价何时显现

同步点本身不消耗 GPU 时间，它只是让 CPU 等待。代价取决于同步时**队列的深度**：

```text
GPU-bound 场景   CPU 领先 GPU 很多，队列深；同步让 CPU 等一会儿，GPU 一直有活干  → 代价小
CPU-bound 场景   队列本来就浅；同步后 GPU 立即空闲，等 CPU 重新提交              → 代价大
高频同步         每步一次，队列永远填不深，两侧交替空闲                          → 最坏
```

所以 sync-bound 的信号不是"有同步"，而是**同步频繁且发生时队列很浅**。它常与 §1 的 launch-bound 叠加：小 Kernel 让队列浅，频繁 `.item()` 让队列反复排空。

#### 3.3 处方一：批量取回，降低同步频率

同步的代价与频率成正比。把每步的 `loss` 留在 GPU 上累加，每 N 步 `.item()` 一次：

```python
running = torch.zeros((), device="cuda")
for step, batch in enumerate(loader):
    loss = train_step(batch)
    running += loss.detach()                  # GPU 上累加，不同步
    if step % 100 == 0:
        log(running.item() / 100)             # 每 100 步同步一次
        running.zero_()
```

同步次数从每步一次降到每百步一次，队列在绝大多数 step 里保持深度。同样的思路适用于梯度范数、准确率等所有监控指标：在 GPU 上累积，定期取回。

#### 3.4 处方二：pinned memory 与异步拷贝

Host 到 Device 的传输默认是同步的：CPU 发起拷贝后等它完成。让它异步需要两个条件：源内存是 **pinned memory**（页锁定，不会被操作系统换出，GPU 的 DMA 引擎可以直接读），且调用时指定 `non_blocking=True`：

```python
loader = DataLoader(dataset, batch_size=64, pin_memory=True, num_workers=4)
for batch in loader:
    batch = batch.to("cuda", non_blocking=True)     # 拷贝在传输引擎上进行，CPU 不等
    ...
```

缺任一条件，`non_blocking` 静默退化为同步拷贝——不报错，只是不异步。`pin_memory=True` 让 `DataLoader` 的 worker 把 batch 放进 pinned memory，主进程拿到的已经是可以异步传输的数据。反方向（Device 到 Host）同理：目标是 pinned Tensor 且 `non_blocking=True`，之后需要显式同步再读值。

#### 3.5 处方三：多 Stream 重叠传输与计算

`non_blocking` 让 CPU 不等拷贝，但拷贝仍在默认 Stream 上排队，与计算 Kernel 串行。要让传输与计算真正并发，需要把它们放到不同的 Stream：

```python
copy_stream = torch.cuda.Stream()

next_batch = None
for batch in loader:
    with torch.cuda.stream(copy_stream):
        next_batch = batch.to("cuda", non_blocking=True)     # 在另一条队列上传输
    if current is not None:
        y = model(current)                                    # 默认队列上计算，与传输并发
    torch.cuda.current_stream().wait_stream(copy_stream)      # 计算队列等传输完成
    current = next_batch
    current.record_stream(torch.cuda.current_stream())        # 告知 allocator 跨 Stream 使用
```

`wait_stream` 表达跨 Stream 依赖，`record_stream` 防止 allocator 在拷贝完成前回收内存。这两行漏掉任何一行都会产生难以复现的数据错误——这是多 Stream 的主要风险。`DataLoader` 加 `pin_memory` 加 `non_blocking` 在大多数情况下已经足够，显式多 Stream 用于传输时间与计算时间同量级的场景。

#### 3.6 处方四：消除数据依赖的 shape

`tensor[mask]`、`torch.nonzero`、`torch.unique` 的同步来自输出 shape 未知：CPU 必须等 GPU 算出有多少个元素，才能分配输出。这类同步无法用异步手段消除，只能改写算法让 shape 固定：

```python
# 同步：输出长度取决于 mask 中 True 的个数
selected = x[mask]
loss = selected.pow(2).mean()

# 不同步：输出 shape 与 x 相同，用 0 填充未选中的位置
loss = torch.where(mask, x.pow(2), 0.0).sum() / mask.sum()
```

变长序列同理：用固定长度加 padding 加 attention mask，代替按实际长度切分。这也是第七篇 Dynamic Shape 里"数据依赖的 shape 会导致 graph break"的同一个问题——在 Eager 下表现为同步，在编译下表现为无法捕获。

#### 3.7 处方五：`set_sync_debug_mode`，找出所有同步点

前四条处方针对已知的同步点。未知的同步点用调试模式找：

```python
torch.cuda.set_sync_debug_mode("warn")     # 每次隐式同步打印警告和 Python 调用栈
# 或 "error"：直接抛异常，用于 CI 中确保热路径无同步
```

在训练循环上跑几步，警告列表就是完整的同步点清单。逐条判断：是必要的（如每百步的日志），还是意外的（如某个工具函数里的 `.item()`）。

#### 3.8 特例：数据加载

数据加载是 sync-bound 的一种常见形态：GPU 泳道大段空白，CPU 停在 `DataLoader.__next__`。GPU 不是在等同步，而是在等数据到达。

```text
症状        GPU 利用率周期性掉到零，周期等于一个 batch 的加载时间
原因        num_workers 不足；预处理太重；磁盘或网络 I/O 慢；collate 在主进程做了大量工作
处方        增加 num_workers；预处理离线化（提前 tokenize、缓存为二进制格式）；
            prefetch_factor 增大预取深度；pin_memory=True；把 augmentation 移到 GPU 上做
```

判断方法：单独 Benchmark `DataLoader` 的迭代速度（不带模型），与训练 step 时间对比。如果加载一个 batch 的时间接近或超过训练一个 batch 的时间，数据加载就是瓶颈——此时优化模型毫无意义。


## 四、空间维度：显存

时间维度上的每一类处方几乎都与显存有关：增大 batch 消耗显存，融合减少中间 Tensor 节省显存，低精度减半激活值。所以显存分析不是独立话题，它决定了时间维度上有多少操作空间。

### 1. 训练时显存的构成

```text
参数                 P × 每参数字节数
梯度                 与参数同形状、同 dtype
优化器状态           Adam：两份 fp32 状态（m、v）；SGD with momentum：一份
激活值               前向保存供反向使用的中间结果，∝ batch × seq × hidden × layers
临时工作区           cuBLAS / cuDNN workspace，Attention 的 score 矩阵
Caching Allocator 的保留量   已向驱动申请但当前未分配给 Tensor 的部分
```

前三项是**静态**的，与 batch 无关；激活值随 batch 线性增长，是单卡训练中可调节的主要部分。

### 2. 混合精度对显存的两面

第三章 §2 把低精度作为时间维度的处方。它对空间的影响不是单向的：

```text
激活值        bf16 减半                                    ← 主要收益，∝ batch
参数与梯度    bf16 减半
主参数        必须保留 fp32 副本（第三章 §2.6：bf16 存不住小的更新量）    ← 额外开销
优化器状态    保持 fp32
```

以 Adam 为例，每个参数的静态占用：

```text
纯 fp32       fp32 参数 4 + fp32 梯度 4 + m 4 + v 4 = 16 B/参数
混合精度      bf16 参数 2 + bf16 梯度 2 + fp32 主参数 4 + m 4 + v 4 = 16 B/参数
```

静态部分**没有减少**，收益全在激活值上。7B 参数的模型仅静态部分就是 112 GB，单卡放不下——这是第九篇分布式的起点。

### 3. Caching Allocator：`allocated` 与 `reserved`

`cudaMalloc` / `cudaFree` 很慢（微秒到毫秒级，且 `cudaFree` 隐式同步）。PyTorch 的 CUDA Caching Allocator 在中间加了一层：向驱动申请大块（Segment），切成 Block 分配给 Tensor；Tensor 释放时 Block 回到缓存池，不还给驱动。

```python
torch.cuda.memory_allocated()       # 当前 Tensor 实际占用
torch.cuda.memory_reserved()        # 已向驱动申请的总量（allocated + 缓存中的空闲 Block）
torch.cuda.max_memory_allocated()   # 峰值，OOM 分析的关键数字
torch.cuda.reset_peak_memory_stats()
```

`nvidia-smi` 显示的是 `reserved`（加上 CUDA 上下文本身），不是模型真正在用的量。**`reserved` 高而 `allocated` 低不是泄漏，是缓存**。

### 4. 碎片

缓存池中的空闲 Block 可能总量足够但没有一块连续的足够大：

```text
Segment: [  已用 1MB  ][ 空闲 3MB ][  已用 1MB  ][ 空闲 3MB ]
请求 5MB → 空闲总量 6MB，但无连续 5MB → 向驱动申请新 Segment → 可能 OOM
```

OOM 报错信息里能直接看到这个状态：

```text
CUDA out of memory. Tried to allocate 5.00 GiB. GPU 0 has a total capacity of 79.15 GiB
of which 2.31 GiB is free. Process has 76.84 GiB memory in use. Of the allocated memory
68.12 GiB is allocated by PyTorch, and 7.91 GiB is reserved by PyTorch but unallocated.
```

"reserved but unallocated" 接近 8 GB 就是碎片。缓解手段：

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`：让 Segment 可以扩展而不是新申请，显著减少碎片（2.x 引入）；
- 让 shape 稳定：动态 shape 是碎片的主要来源，因为每种 shape 的 Block 大小不同；
- `torch.cuda.empty_cache()` 归还缓存给驱动——**它不解决碎片**，只在多进程共享 GPU 时有用，且会引入同步（第三章 §3）。

### 5. 峰值在哪里

平均占用不高但 OOM，说明某一时刻有尖峰。典型位置：

```text
反向开始时       所有激活值都还在，第一个反向 Kernel 又要分配梯度
优化器 step 时    foreach 优化器可能一次性为所有参数分配临时 Tensor
Attention        score 矩阵 batch × heads × seq × seq，seq=8k 时单层就是 GB 级
evaluation       忘了 torch.no_grad()，保存了不需要的激活值
```

用 memory snapshot 精确定位：

```python
torch.cuda.memory._record_memory_history(max_entries=100000)
train_step()
torch.cuda.memory._dump_snapshot("snapshot.pickle")
```

把 `snapshot.pickle` 拖进 `pytorch.org/memory_viz`，得到每个分配的时间线、大小和**分配时的 Python 调用栈**。这是 OOM 排查最有效的工具：直接看到峰值时刻哪几个 Tensor 占了显存、是哪行代码分配的。

### 6. 泄漏

`allocated` 随 step 单调增长，常见原因都是**无意中持有了计算图**：

```python
total_loss += loss            # loss 带 grad_fn，整张图被引用，激活值无法释放
total_loss += loss.item()     # 正确，但是同步点（第三章 §3）
total_loss += loss.detach()   # 正确且不同步
```

其他来源：把 Tensor 存进跨 step 的 list 或 dict；hook 中保存了中间结果；异常路径没有走到 `zero_grad`。

### 7. 用时间换空间：Activation Checkpointing

激活值占用 ∝ 层数。Activation Checkpointing 只保存部分层的输入，反向时重新计算这段前向：

```python
from torch.utils.checkpoint import checkpoint

y = checkpoint(transformer_block, x, use_reentrant=False)
```

代价是被 checkpoint 的段前向算两遍，通常增加约 30% 计算时间，换来激活显存从 O(层数) 降到 O(√层数) 或更低。第七篇提过，AOTAutograd 的切分器在编译时自动做局部版本的这个权衡。

它是否值得取决于：省下的显存能否换来更大的 batch，更大的 batch 是否让 Kernel 从 launch-bound 变成 compute-bound（第三章 §1）——如果能，总吞吐反而上升；如果 GPU 已经饱和，就是纯损失。第五章的案例会给出一个"不值得"的实例。

### 8. 用空间换时间

反方向的交换同样常见：

```text
KV Cache            推理时缓存已计算的 key/value，避免重算，显存 ∝ 序列长度
cuDNN workspace     给 cuDNN 更多工作区，它能选择更快的算法
更大的 batch        本身就是用显存换 GPU 利用率
预分配              提前分配固定大小的 buffer，避免运行时分配与碎片
```

### 9. 内存：CPU 侧的空间什么时候重要

时间维度上 CPU 与 GPU 对称，空间维度上不对称：CPU 内存很少成为"时间去哪了"意义上的瓶颈，但在几个场景下是硬约束。

```text
DataLoader 多 worker     每个 worker 是一个进程；用 Python list 持有的数据集会因引用计数触发 copy-on-write，
                        N 个 worker 复制 N 份。用 numpy 数组、内存映射文件或 Arrow 格式避免
pinned memory           页锁定内存不能换出，总量受操作系统限制；pin_memory=True 加大 prefetch 时可能耗尽
CPU offload             把优化器状态或部分参数放到内存（第九篇 ZeRO-Offload），用 PCIe 带宽换显存容量；
                        此时 CPU 内存容量和 PCIe 带宽同时成为约束
数据集缓存              把预处理结果全部放进内存加速 epoch，前提是放得下
```

诊断手段是操作系统级的（`free`、`/proc/<pid>/status`、`psutil`），PyTorch 没有对应 `memory_allocated` 的 API——因为 CPU Tensor 用的是系统分配器，没有 Caching Allocator 那一层。


## 五、一个完整案例：Transformer block 的训练 step

把前面的工具和模型串起来，走一遍第一章 §5 的流程。数字为示意，比例关系反映真实规律。

### 1. 对象

一个标准的 Transformer block（Pre-LN）：

```python
class Block(nn.Module):
    def __init__(self, d=512, heads=8):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)
        self.heads = heads

    def attention(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q = q.view(B, T, self.heads, -1).transpose(1, 2)
        k = k.view(B, T, self.heads, -1).transpose(1, 2)
        v = v.view(B, T, self.heads, -1).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        att = att.softmax(dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(y)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x

model = nn.Sequential(*[Block() for _ in range(12)]).cuda()
```

训练 step：前向 → `loss.backward()` → `optimizer.step()`（AdamW）→ `zero_grad()`。序列长度 512。

### 2. 基线与正确性测试

```python
def step(model, x):
    loss = model(x).float().pow(2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss

# 正确性：保存 fp32、batch=8、固定种子下 20 步的 loss 序列作为参考
ref_losses = run_steps(model_ref, 20)

# 基线：Benchmark 一个 step
t = benchmark.Timer(stmt="step(model, x)", globals={...})
print(t.blocked_autorange(min_run_time=5))
```

```text
基线  batch=8   fp32   eager        step: 48.2 ms      吞吐: 166 samples/s     峰值显存: 3.1 GB
```

### 3. 第一轮：采集与归类

```text
Profiler 表格
  Self CUDA total: 21 ms / step          ← GPU 只忙了 44%
  cudaLaunchKernel: 2100 次 / step       ← 平均 12 µs CPU 成本，共 25 ms
  aten::mm: 平均 18 µs                   ← 小 Kernel

时间线形态：CPU 泳道密集，GPU 泳道稀疏
```

对照第二章 §3.6 的表：**Launch-bound**（第三章 §1）。batch=8、d=512 下每个 Kernel 太小，GPU 等 CPU。此时优化任何 Kernel 都无效。

### 4. 修改一：增大 batch（第三章 §1 处方一）

最直接的处方。显存有余量（3.1 GB），把 batch 提到 64：

```text
batch=64  fp32   eager        step: 118 ms       吞吐: 542 samples/s     峰值显存: 19.6 GB
```

吞吐提升 3.3 倍。再看 Profiler：

```text
  Self CUDA total: 112 ms / step         ← GPU 忙 95%
  aten::mm: 68 ms（61%）                 ← 矩阵乘主导
  aten::softmax + aten::_softmax_backward_data: 14 ms
  aten::layer_norm / gelu / add / mul 等逐元素与归约: 22 ms
```

瓶颈从 launch-bound 变成了 GPU-bound，其中矩阵乘占六成——**Compute-bound**（第三章 §2）。这才是可以谈 Kernel 性能的起点。

### 5. 修改二：低精度（第三章 §2 处方二）

矩阵乘 compute-bound，fp32 只能用 19.5 TFLOPs 的算力。切到 bf16 autocast：

```text
batch=64  bf16   eager        step: 51 ms        吞吐: 1255 samples/s    峰值显存: 12.8 GB
```

```text
  aten::mm: 21 ms（Tensor Core，约 3 倍）
  逐元素与归约: 13 ms（数据量减半）
  aten::_to_copy: 4 ms                   ← 新出现：autocast 的 cast Kernel
```

正确性：bf16 的 20 步 loss 与 fp32 参考的相对差异在 1% 以内，曲线趋势一致。通过。显存：激活值减半，静态部分不变（第四章 §2）。

### 6. 修改三：`torch.compile`（第三章 §2 处方一，兼 §1 处方二）

剩余时间里逐元素算子链（LayerNorm → 残差 → GeLU → cast）仍是一串独立的 memory-bound Kernel。交给 Inductor 融合：

```text
batch=64  bf16   compile      step: 38 ms        吞吐: 1684 samples/s    峰值显存: 11.9 GB
                              首次调用: 47 s（冷编译）
```

```text
  Kernel 数: 2100 → 640 / step
  aten::_to_copy: 消失（融合进相邻 Kernel）
  triton_*_fused_*: 9 ms（替代了原来 17 ms 的逐元素与归约）
  aten::mm: 21 ms（不变，Extern Kernel）
```

代价：47 秒冷编译；输入 shape 变化会触发重编译（第七篇）。训练场景 shape 固定，可接受。

### 7. 修改四：算法级融合（第三章 §2 处方一）

Profiler 里 `softmax` 及其反向仍占 8 ms，且 memory snapshot 显示峰值处最大的 Tensor 是 `att`：`64 × 8 × 512 × 512 × 2 B = 268 MB` 每层，12 层前向保存 3.2 GB。手写的 attention 把 score 矩阵完整写到显存再读回来做 softmax，是典型的 memory-bound 模式。

换成融合的 `scaled_dot_product_attention`：

```python
def attention(self, x):
    ...
    y = F.scaled_dot_product_attention(q, k, v)      # 内部选择 FlashAttention 等融合实现
    ...
```

```text
batch=64  bf16   compile+sdpa step: 29 ms        吞吐: 2207 samples/s    峰值显存: 8.4 GB
```

score 矩阵不再物化，显存降 3.5 GB，时间降 9 ms。这是编译器发现不了的融合。

### 8. 第二轮：时间换空间是否值得（第四章 §7）

峰值显存降到 8.4 GB 后，可以考虑用 checkpointing 换更大的 batch：

```text
batch=64  + checkpoint 每个 Block   step: 36 ms    峰值显存: 4.1 GB
batch=128 + checkpoint 每个 Block   step: 68 ms    吞吐: 1882 samples/s    峰值显存: 7.6 GB
```

checkpoint 后加大 batch 的吞吐（1882）反而低于不 checkpoint 的 batch=64（2207）——GPU 在 batch=64 时已经饱和（第三章 §2.9），再加 batch 没有 launch 收益，而重算增加了 30% 计算。**这个案例里不值得**。如果基线是 batch=8 时显存就满了，结论会反过来。

### 9. 优化报告

按大纲的要求，每项改动不只说"快了多少"：

| 改动 | step 时间 | 节省的是 | 显存 | 精度 | shape 依赖 | 转移的成本 |
|---|---|---|---|---|---|---|
| batch 8→64 | 48 → 118 ms（吞吐 ×3.3） | Launch 开销 | +16.5 GB | 无 | — | 无 |
| bf16 autocast | 118 → 51 ms | 计算（Tensor Core）+ 访存（减半） | −6.8 GB | loss 相对差 <1% | — | 无 |
| `torch.compile` | 51 → 38 ms | 访存（融合）+ Launch | −0.9 GB | 浮点结合顺序 | shape 变化触发重编译 | 47 s 冷编译 |
| SDPA | 38 → 29 ms | 访存（不物化 score） | −3.5 GB | 与手写实现容差内一致 | 要求 head_dim 等满足 Kernel 约束 | 无 |
| checkpoint | 29 → 36 ms | — | −4.3 GB | 无 | — | +30% 计算，**未采用** |

从 166 到 2207 samples/s，13 倍。其中没有一项是"优化某个 Kernel"，全部是**改变瓶颈类别**：先消灭 launch-bound，再对 compute-bound 用 Tensor Core，再对 memory-bound 做融合。

### 10. 这条路径的通用性

案例的顺序不是偶然：

```text
1. 先看 GPU 利用率        低 → CPU 侧或同步（第三章 §1、§3），先解决它，否则其他优化无法测量
2. 再看 Kernel 时间分布    矩阵乘主导 → 低精度；逐元素主导 → 融合（第三章 §2）
3. 再看显存峰值组成        大 Tensor 能否不物化；激活值能否重算；是否值得（第四章）
4. 每步回到正确性测试和 Benchmark
```

不同模型的具体数字不同，但顺序几乎总是这样。跳过第一步直接做第二步，是性能优化中最常见的浪费。


## 六、Java 工程师如何理解 PyTorch 性能分析

### 1. 异步与测量：`CompletableFuture`

```java
long start = System.nanoTime();
CompletableFuture<Result> f = executor.submit(task);   // 立即返回
long elapsed = System.nanoTime() - start;              // 测的是提交，不是执行
```

Java 工程师一眼能看出这段代码测错了。PyTorch 的 `time.time()` 围住 `model(x)` 犯的是同一个错——`model(x)` 就是 `submit`，`synchronize()` 就是 `f.get()`。CUDA Event 相当于在任务内部打时间戳。

### 2. Benchmark：JMH

JMH 解决的问题与 `torch.utils.benchmark` 完全对应：

| JMH | `torch.utils.benchmark` |
|---|---|
| `@Warmup` 排除 JIT 编译、类加载 | warmup 排除 CUDA 初始化、cuDNN 算法选择、编译 |
| `@Fork` 隔离 JVM 状态 | 固定种子、清理 GPU 进程 |
| `Blackhole` 防止死代码消除 | GPU 没有此问题，但 `torch.compile` 可能消除无用计算 |
| 报告分位数而非均值 | 报告中位数与 IQR |
| `Mode.Throughput` vs `Mode.AverageTime` | 吞吐 vs 延迟口径 |

用过 JMH 的人知道"微基准测试很容易测错"，这个直觉在 GPU 上同样成立，且多了异步这一层陷阱。

### 3. Profiler：JFR 与 async-profiler

`torch.profiler` 的角色是 JFR 加 async-profiler：采样调用栈、记录事件、导出时间线。区别是它要同时观察两个处理器并对齐时间轴。Nsight Systems 更像 `perf` 加 Linux 系统级追踪——看操作系统眼中的所有线程和设备。Nsight Compute 没有 Java 世界的直接对应物，最接近的是用 `perf stat` 看单个热点函数的硬件计数器（cache miss、IPC）。

### 4. 五类瓶颈的 Java 对应

| PyTorch | Java 世界的相似物 |
|---|---|
| Launch-bound | 每次调用开销远大于工作量：细粒度 RPC、逐条 JDBC 而非 batch insert |
| Python-bound | 业务逻辑在 GC 或反射、序列化上花的时间超过真正的计算 |
| Memory-bound | cache miss 主导：遍历链表 vs 遍历数组、伪共享 |
| Compute-bound | 真正的 CPU 密集，`perf` 显示 IPC 高、热点在算术 |
| Sync-bound | 锁竞争、`Future.get()` 在错误位置、同步 I/O 阻塞线程池 |

### 5. 内存：GC 与 Caching Allocator

| JVM | PyTorch |
|---|---|
| `-Xmx` 堆上限 | 物理显存 |
| committed vs used | `reserved` vs `allocated` |
| GC 回收后堆不立即归还 OS | 释放的 Block 留在缓存池，不 `cudaFree` |
| 老年代碎片，G1 的 humongous 分配失败 | 大 Block 碎片，"reserved but unallocated" |
| `System.gc()` 通常无益 | `torch.cuda.empty_cache()` 通常无益 |
| 堆 dump + MAT 找泄漏 | memory snapshot + memory_viz 找泄漏 |
| 持有对象引用导致无法回收 | 持有带 `grad_fn` 的 Tensor 导致整张图无法释放 |

一个关键差别：JVM 有 GC，对象生命周期由可达性决定；PyTorch 靠引用计数（第二篇），Tensor 引用归零立即释放。所以 PyTorch 的"泄漏"几乎总是**有一个明确的引用还活着**，比 JVM 的泄漏更容易定位——只要找到那个引用。

### 6. 精度：`float` vs `double`

Java 工程师默认用 `double`，把 `float` 当作节省内存的特例。PyTorch 反过来：fp32 是基线，还要继续往下降到 bf16。原因是 GPU 的算力和带宽对精度极其敏感——Tensor Core 在 bf16 上的算力是 fp32 的 16 倍。这不是"够用就行"的取舍，而是 Roofline 直接推出的结论。

但 Java 工程师熟悉的浮点陷阱（`0.1 + 0.2 != 0.3`、大数吃小数、累加误差）在 bf16 下会被放大百倍。第三章 §2.6 的规则——归约用高精度、主参数用 fp32、避免相近数相减——都是这些陷阱的对策。


## 七、本文小结

### 1. 两个维度

```text
时间    两条时间线：CPU 提交 / GPU 执行，异步，由较长者决定 step 时间
        CPU 侧      Python-bound / Launch-bound        ← 第三章 §1
        GPU 侧      Memory-bound / Compute-bound        ← 第三章 §2，T ≥ max(数据量/带宽, 运算量/算力)
        两侧之间    Sync-bound                          ← 第三章 §3
空间    显存 = 参数 + 梯度 + 优化器状态 + 激活值 + 工作区 + 缓存    ← 第四章
交换    时间 ↔ 空间：checkpointing、batch；低精度同时省两者，代价是数值精度
```

### 2. 症状 → 归类 → 工具 → 处方

| 症状 | 归类 | 观察工具 | 处方 | 位置 |
|---|---|---|---|---|
| GPU 利用率低，CPU 100%，Kernel 多而小 | Launch-bound | Profiler 表格（`cudaLaunchKernel` 占比）、时间线形态 | 增大 batch、融合、CUDA Graphs、fused 优化器 | 三 §1 |
| GPU 利用率低，CPU 时间不在算子上 | Python-bound | Profiler `with_stack` | 向量化、`torch.compile`、预处理移出循环 | 三 §1 |
| GPU 忙，逐元素算子占比高 | Memory-bound | Profiler 按 CUDA 时间排序、ncu DRAM Throughput | 融合、算法级融合、bf16、数据布局 | 三 §2 |
| GPU 忙，矩阵乘占比高 | Compute-bound | Profiler、ncu SM Throughput | Tensor Core（bf16 / TF32）、减少计算量 | 三 §2 |
| 时间线两侧交替空洞 | Sync-bound | `set_sync_debug_mode`、时间线 | 批量取回、pinned memory + 异步拷贝、避免数据依赖 shape | 三 §3 |
| GPU 大段空白，CPU 停在 DataLoader | 数据加载 | 单独 Benchmark DataLoader | `num_workers`、离线预处理、prefetch | 三 §3 |
| `reserved` ≫ `allocated`，OOM | 碎片 | OOM 信息、`memory_reserved` | `expandable_segments`、稳定 shape | 四 |
| `allocated` 单调增长 | 泄漏 | memory snapshot | 找到持有 `grad_fn` 的引用 | 四 |
| 平均显存不高但 OOM | 峰值 | memory snapshot 时间线 | 不物化大 Tensor、checkpoint、`no_grad` | 四 |

### 3. 测量的纪律

```text
计时必须同步，或用 CUDA Event
warmup 后再测，报告中位数与分布
写明口径：batch / seq / dtype / 硬件 / 版本
正确性测试先于性能改动，容差有依据
一次只改一件事，每次回到同一个 Benchmark
```

### 4. 优化报告的五个问题

```text
节省的是计算、访存、同步还是 launch？
是否增加了显存？
是否改变了数值精度？
是否只对特定输入 shape 有效？
是否把成本转移到了编译或初始化阶段？
```

### 5. 与前几篇的连接

```text
第二篇 stride / contiguous     → 第三章 §2.7：访存模式决定 memory-bound Kernel 的实际带宽
第三篇 grad_fn 持有计算图      → 第四章 §6：泄漏几乎总是持有了带 grad_fn 的 Tensor
第四篇 autocast / GradScaler   → 第三章 §2.5-2.6：为什么快、快在哪类瓶颈、代价的边界
第五篇 入口 / 分发 / 执行      → 第三章 §1.1：每个算子在 CPU 上的固定成本由此而来
第七篇 融合与内存规划          → 第三章 §2.4：融合是 memory-bound 算子的处方
```

到这里，单卡上的 PyTorch 已经讲完：Tensor、Autograd、Module、算子、扩展、编译、性能。第四章算过，7B 模型仅静态显存就要 112 GB，单卡放不下。下一篇进入多卡：

> **当一张卡放不下模型或跑不完数据时，PyTorch 如何把计算和状态切分到多个设备，并让通信与计算重叠？**


## 下一篇

[分布式 PyTorch](/pytorch-distributed-training.html)
