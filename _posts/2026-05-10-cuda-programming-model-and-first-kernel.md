---
layout: post
title: "GPU Kernel 工程（02）：CUDA 编程模型与第一个 kernel"
subtitle: "The CUDA Programming Model and Your First Kernel, Measured"
tags: [CUDA, Triton, GPU, AI, AI-Infra]
catalog: true
---

> 本文是[《GPU Kernel 工程：从 CUDA 执行模型到 FlashAttention》](/gpu-kernel-engineering.html)系列的第 2 篇（共十篇）。上一篇：[GPU 为什么这样设计：硬件结构与 Roofline](/gpu-architecture-and-roofline.html)　下一篇：[访存合并与 elementwise kernel](/memory-coalescing-and-elementwise-kernels.html)

上一篇建立了本系列的分析框架，用到的结论可以压缩成三个数字。GPU 的基本执行单位是 **warp**：32 个线程共用一个指令流，一条指令同时作用在 32 个数据上。以 A100 SXM 80GB 为默认分析对象（标称值）：HBM2e 带宽约 **2.0 TB/s**，BF16 Tensor Core 算力 312 TFLOPS，两者相除得到 Roofline 的拐点（ridge point）：

$$
\text{ridge} = \frac{312 \times 10^{12}\ \text{FLOP/s}}{2.0 \times 10^{12}\ \text{byte/s}} \approx 156\ \text{FLOP/byte}
$$

算术强度低于 156 FLOP/byte 的计算，时间下界由字节数决定：$$T \ge \text{bytes} / (2.0\ \text{TB/s})$$；高于它的，由 FLOPs 决定。一个 BF16 的 elementwise 加法 $$y = x + b$$，每元素读 4 字节、写 2 字节、做 1 次 FLOP，算术强度 $$1/6 \approx 0.17$$，距拐点差三个数量级，是彻底的 memory-bound——它的时间只由要搬多少字节决定。

这一篇写这个 kernel。但写它之前需要先把 CUDA 编程模型本身讲清楚：代码在哪里运行、线程如何组织、内存在哪里、什么时候真正开始执行、错误什么时候报出来、编译器把源码变成了什么。这些是后面八篇每一段代码都依赖的基础。然后回答总纲提出的核心问题：

> **vector add 的 kernel 只有五行，它跑出了理论带宽的多少？没跑满的部分去了哪里？**

回答这个问题需要一套可靠的测量方法。本篇结束时会留下一个 benchmark 脚手架——事件计时、warmup、多次迭代取中位数、L2 flush——它会一直用到第十篇。

依照系列惯例，本文的所有性能数字要么是可推导的理论下界，要么用"通常能达到"的区间给出；本文没有 GPU 实测值，读者自己跑出的数字应当落在给出的区间里。


## 一、host、device 与 kernel

### 1. 两个处理器、两个地址空间

一个 CUDA 程序同时运行在两个处理器上。**host** 是 CPU 和它的内存；**device** 是 GPU 和它的显存（HBM）。两边有各自独立的地址空间：host 上的指针不能在 GPU 上解引用，反之亦然。数据要从一边到另一边，必须显式拷贝（或者使用统一内存/pinned 内存这类机制，本系列不展开）。

控制流始终在 host 上。CPU 负责分配显存、拷贝数据、发起 GPU 上的计算、等待结果。GPU 上运行的每一段代码都是被 CPU"发射"（launch）出去的一个函数，这个函数叫 **kernel**。

### 2. 三个函数限定符

CUDA C++ 用三个限定符标记一个函数在哪里运行、从哪里调用：

```text
限定符          运行在    只能从哪里调用          备注
__global__     device   host（或 device 端动态并行）  这就是 kernel；返回类型必须是 void
__device__     device   device                  kernel 内部调用的辅助函数
__host__       host     host                    默认值，通常省略
```

`__host__ __device__` 可以同时加在一个函数上，让它在两边各编译一份——`cuda_bf16.h` 里的 `__float2bfloat16` 就是这样，所以它既能在 kernel 里用，也能在 host 侧做数据准备。

一个 kernel 长这样：

```cpp
__global__ void vector_add_f32(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ c, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
```

它没有循环。kernel 描述的是**一个线程**做什么；host 在发射时说明要启动多少个线程，每个线程通过内建变量算出自己负责哪个元素。这是 CUDA 编程模型与 CPU 代码最根本的区别：把 `for` 循环的循环体拿出来，循环变量换成线程编号。

`__restrict__` 告诉编译器几个指针互不别名，允许它更自由地重排和缓存访存；对 elementwise kernel 这是习惯性写法。

### 3. 发射一个 kernel

host 用三尖括号语法发射：

```cpp
const int block = 256;
const unsigned grid = (unsigned)((n + block - 1) / block);
vector_add_f32<<<grid, block>>>(d_a, d_b, d_c, n);
```

`<<<grid, block, sharedMemBytes, stream>>>` 四个参数里后两个可省略。`grid` 是要启动多少个 block，`block` 是每个 block 多少线程。这两个数怎么定，是下一节的内容。


## 二、线程层级：grid、block、thread

### 1. 三层结构与内建变量

一次 kernel launch 启动一个 **grid**；grid 由若干 **block** 组成；block 由若干 **thread** 组成。grid 和 block 都可以是一维、二维或三维的，用 `dim3` 表示。每个线程可以读到四个内建变量：

```text
threadIdx   本线程在 block 内的坐标      (x, y, z)
blockDim    block 的尺寸                (x, y, z)
blockIdx    本 block 在 grid 内的坐标    (x, y, z)
gridDim     grid 的尺寸                 (x, y, z)
```

对一维问题，全局索引就是"跳过前面所有 block 的线程数，再加上自己在 block 里的位置"：

$$
i = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}
$$

对二维问题（比如一个 $$M \times N$$ 的矩阵、每线程一个元素）：

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;   // 沿 N
int row = blockIdx.y * blockDim.y + threadIdx.y;   // 沿 M
if (row < M && col < N) out[row * N + col] = ...;
```

注意 `x` 对应最内层、连续变化的维度。这不是随意约定：下一节会看到同一 warp 里的线程是沿 `threadIdx.x` 连续编号的，而下一篇会说明让相邻线程访问相邻地址是 memory-bound kernel 的第一要求。

上面 `vector_add_f32` 里把 `blockDim.x` 转成 `size_t` 再乘，是因为 `blockIdx.x * blockDim.x` 是两个 `unsigned int` 相乘，元素数超过 $$2^{32}$$ 时会溢出。本篇 $$n = 2^{28}$$ 不会触发，但养成习惯比事后排查便宜。

### 2. 边界检查为什么不可省

grid 大小是向上取整算出来的，所以最后一个 block 里通常有一部分线程对应的索引 $$i \ge n$$。没有 `if (i < n)`，这些线程会越界读写，后果从静默地写坏相邻缓冲区到 `cudaErrorIllegalAddress` 都有可能，而且——后面第五章会讲——这个错误不会在 launch 时报出来。

唯一可以省掉边界检查的情形是 host 侧已经保证 $$n$$ 是 block 大小的整数倍，且这个保证写成了 `assert` 或 `TORCH_CHECK`。

### 3. block 大小怎么选

几条经验规则：

- **必须是 32 的倍数。** block 在硬件上被切成 warp，每 32 个线程一个；block 大小 100 会得到 4 个 warp，其中最后一个只有 4 个线程有效，剩下 28 个 lane 的执行槽位白白浪费。
- **128–256 是最常见的选择。** 上限是 1024。太小（比如 32）会让每个 SM 上驻留的 block 数触到"每 SM 最多 32 个 block"的限制，从而无法填满每 SM 2048 线程的容量；太大则让 block 成为粗粒度的调度单位，寄存器和 shared memory 的分配也更难与 SM 的容量匹配。
- **与 occupancy 的关系一句话：** 一个 SM 能同时驻留多少个 block，由寄存器（65536 个 / SM）、shared memory、最大 32 block/SM、最大 2048 线程/SM 四个上限中最紧的那个决定；block 大小是这个计算的输入之一，具体展开留给下一篇讨论延迟隐藏时。

对一维 elementwise kernel，256 是一个几乎不会错的起点。

### 4. grid 大小

最直接的策略是"每线程一个元素"：

$$
\text{grid} = \left\lceil \frac{n}{\text{block}} \right\rceil = \frac{n + \text{block} - 1}{\text{block}}
$$

$$n = 2^{28}$$、block = 256 时 grid = 1,048,576 个 block。grid 的 x 维上限是 $$2^{31} - 1$$，y、z 维上限 65535，一维问题不必担心。

要注意 grid 与 SM 数量的关系。A100 有 108 个 SM，如果 grid 只有几十个 block，大部分 SM 是空的；这是第八章"为什么跑不满"的原因之一。下一篇会引入 grid-stride loop，让 grid 大小与元素数解耦。

### 5. block 在硬件上如何切成 warp

程序员看到的是 block 和 thread；硬件调度的是 warp。一个 block 被分配到一个 SM 之后，它的线程被**线性化**再按 32 个一组切分：

$$
\text{linear} = \text{threadIdx.x} + \text{threadIdx.y} \cdot \text{blockDim.x} + \text{threadIdx.z} \cdot \text{blockDim.x} \cdot \text{blockDim.y}, \qquad \text{warp} = \lfloor \text{linear} / 32 \rfloor
$$

`x` 是变化最快的维度。几个例子：

```text
block (256)        warp k = 线程 32k … 32k+31，共 8 个 warp

block (32, 8)      linear = x + 32y → 每一行 y 恰好是一个 warp，共 8 个 warp
                   warp 0 = (0..31, 0)，warp 1 = (0..31, 1)，…

block (16, 16)     linear = x + 16y → 每两行合成一个 warp，共 8 个 warp
                   warp 0 = (0..15, 0) ∪ (0..15, 1)
                   warp 1 = (0..15, 2) ∪ (0..15, 3)，…

block (8, 8)       linear = x + 8y → 64 线程 = 2 个 warp，每个 warp 覆盖 4 行
```

这个划分决定了"哪 32 个线程会同时发出访存请求"，进而决定访存能否合并、shared memory 会不会 bank conflict——第三、四篇的核心。二维 block 里如果把行方向放在 `y`、列方向放在 `x`，那么 (16, 16) 的 block 里一个 warp 会跨两行；对行主序矩阵，这意味着一个 warp 的 32 个访问落在两段各 64 字节的连续地址上，而不是一段 128 字节。是好是坏取决于具体 kernel，但必须先知道这件事在发生。

同一个 block 内的线程可以通过 shared memory 通信、用 `__syncthreads()` 同步；不同 block 之间没有这两种手段（除了原子操作与 Hopper 的 cluster）。block 一旦开始执行就不会迁移到别的 SM，且 SM 会等到 block 内所有 warp 结束才释放它占的资源。


## 三、设备内存与数据搬运

### 1. `cudaMalloc`、`cudaMemcpy`、`cudaFree`

最基本的三个调用：

```cpp
float* d_a = nullptr;
CUDA_CHECK(cudaMalloc(&d_a, bytes));                                   // 分配显存
CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));       // host → device
CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));       // device → host
CUDA_CHECK(cudaFree(d_a));
```

`cudaMalloc` 返回的是 device 地址空间里的指针，只能传给 kernel 或 CUDA API，不能在 host 上解引用。`cudaMemcpy` 是同步的：它等待之前所有 GPU 工作完成，做完拷贝再返回。异步版本 `cudaMemcpyAsync` 需要配合 pinned host 内存才真正异步。

还有 `cudaMemset` / `cudaMemsetAsync`，把一段显存填成某个字节值；本篇会用它来做 L2 flush。

### 2. 为什么 PyTorch 不直接用它们

PyTorch 的每个 CUDA Tensor 的 Storage 背后并不是一次 `cudaMalloc`。原因一句话：**`cudaMalloc` 和 `cudaFree` 慢（微秒到毫秒级），并且 `cudaFree` 隐含一次设备同步**——它要等 GPU 上所有正在运行的工作结束，才能安全地回收这块内存。一个训练步里有成千次 Tensor 的创建与销毁，每次都同步会把 CPU-GPU 流水彻底打断。

所以 PyTorch 用 **Caching Allocator**（`c10/cuda/CUDACachingAllocator.cpp`，v2.10.0）：向驱动申请大块显存后自己切分和复用，Tensor 释放时只是把块还回缓存池而不调 `cudaFree`。这也是为什么 `del` 一个 Tensor 后 `nvidia-smi` 显示的显存占用不会下降。本系列不展开分配器的机制；kernel 开发者需要知道的只是：从 PyTorch 拿到的 `data_ptr()` 是分配器切出来的一段地址，它的对齐通常是 512 字节（分配器的最小粒度），可以放心用于向量化访存。


## 四、stream、event 与异步语义

### 1. kernel launch 是异步的

`vector_add_f32<<<grid, block>>>(...)` 这一行在 CPU 上做的事情只是把一个"启动请求"放进 GPU 的命令队列，然后**立刻返回**——通常只需几微秒。GPU 何时真正执行、何时执行完，CPU 并不知道，除非显式等待。

这意味着下面的计时是错的：

```cpp
auto t0 = std::chrono::steady_clock::now();
vector_add_f32<<<grid, block>>>(d_a, d_b, d_c, n);
auto t1 = std::chrono::steady_clock::now();   // 只测到了 launch 的开销，kernel 可能还没开始
```

正确的方法有两种：在 `t1` 之前加 `cudaDeviceSynchronize()`（粗糙），或者用 CUDA event 让 GPU 自己给时间戳（推荐，见第 3 小节）。

异步是 GPU 编程性能模型的核心。CPU 可以连续发射几十个 kernel 而不等待，GPU 按顺序消化；只要 CPU 发射得比 GPU 执行得快，GPU 就不会空转。反过来，任何让 CPU 停下来等 GPU 的调用，都是流水线上的一个气泡。

### 2. stream

**stream** 是 GPU 上的一条有序工作队列。同一 stream 里的操作（kernel、memcpy、memset）严格按提交顺序执行；不同 stream 之间没有顺序保证，可以并发。

不指定 stream 时，工作进入**默认 stream**（legacy default stream，编号 0）。它有一个特殊性质：与所有其他阻塞式 stream 互相同步——默认 stream 上的操作要等其他 stream 上已提交的工作完成，反之亦然。这让它安全但不适合并发。

创建自己的 stream：

```cpp
cudaStream_t stream;
CUDA_CHECK(cudaStreamCreate(&stream));
vector_add_f32<<<grid, block, 0, stream>>>(d_a, d_b, d_c, n);
CUDA_CHECK(cudaStreamSynchronize(stream));   // 只等这一条 stream
CUDA_CHECK(cudaStreamDestroy(stream));
```

在 PyTorch 扩展里，永远用 `at::cuda::getCurrentCUDAStream()` 拿当前 stream 再传给 launch，否则你的 kernel 会跑在默认 stream 上，与 PyTorch 自己的 stream 之间失去顺序保证（或者因默认 stream 的全局同步语义拖慢一切）。

### 3. event：GPU 侧的时间戳与依赖

**event** 是插进 stream 里的一个标记。GPU 执行到它时记录当前时间，CPU 可以等待它，也可以查两个 event 之间的时间差：

```cpp
cudaEvent_t start, stop;
CUDA_CHECK(cudaEventCreate(&start));
CUDA_CHECK(cudaEventCreate(&stop));

CUDA_CHECK(cudaEventRecord(start, stream));
vector_add_f32<<<grid, block, 0, stream>>>(d_a, d_b, d_c, n);
CUDA_CHECK(cudaEventRecord(stop, stream));

CUDA_CHECK(cudaEventSynchronize(stop));       // CPU 等到 stop 被 GPU 执行到
float ms = 0.f;
CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));   // 毫秒，分辨率约 0.5 µs
```

`cudaEventElapsedTime` 测的是 GPU 上两个标记之间的时间，不含 CPU 侧 launch 的开销，也不受 CPU 何时调用 `cudaEventSynchronize` 影响。这是 kernel 计时的标准做法；PyTorch 的 `torch.cuda.Event` 是它的封装。

event 还可以用于 stream 之间建立依赖（`cudaStreamWaitEvent`），本系列用到的地方不多。

### 4. `cudaDeviceSynchronize` 的代价

`cudaDeviceSynchronize()` 让 CPU 阻塞直到 GPU 上**所有** stream 的所有工作完成。它简单、可靠，是调试时定位错误的最直接手段。但在生产代码里它是最昂贵的一类调用：

- CPU 停止发射新工作，GPU 队列被排空；
- GPU 做完最后一个 kernel 后必须等 CPU 醒来、再发射下一个 kernel、再等 launch 延迟——中间几微秒到几十微秒 GPU 是空的；
- 一个 decode step 里有几百个小 kernel，每个只跑十几微秒，任何一次同步造成的气泡都与 kernel 本身同量级。

PyTorch 里 `.item()`、`.cpu()`、打印一个 CUDA Tensor 都隐含这种同步；这也是 `torch.cuda.set_sync_debug_mode` 存在的原因。kernel 开发者的原则：**只在 benchmark 与调试代码里同步，正式路径里让 stream 自己排序。**


## 五、错误处理

### 1. 同步错误与异步错误

CUDA API 的错误分两类。

**同步错误**在调用返回时就知道：`cudaMalloc` 显存不足、launch 配置非法（block 超过 1024 线程、shared memory 超限）、传了错的指针给 `cudaMemcpy`。这类错误通过 API 的返回值报出。

**异步错误**发生在 GPU 执行期间：越界访问、非法指令、断言失败。因为 launch 是异步的，CPU 在错误发生时早已往下走了；错误会被记录在上下文里，在**下一次任何与 GPU 同步的调用**时才报出来——可能是几十行之后的一个 `cudaMemcpy`，报出的是一个与它自身毫无关系的 `cudaErrorIllegalAddress`。而且这类错误是**粘性**的：上下文进入不可恢复状态，之后所有 CUDA 调用都返回同一个错误，只能重启进程。

三尖括号 launch 本身没有返回值，所以要用两个 API 查询它：

- `cudaGetLastError()`：返回上一次错误并**清除**它；
- `cudaPeekAtLastError()`：返回但**不清除**。

launch 后紧跟一个 `cudaGetLastError()` 可以捕获同步的配置错误；要捕获 kernel 执行中的异步错误，必须在它后面同步一次再查。调试时可以设环境变量 `CUDA_LAUNCH_BLOCKING=1`，让每次 launch 变成同步的，错误就会精确地停在出错的那一行；PyTorch 的报错信息里推荐它也是这个原因。

### 2. 一个 `CUDA_CHECK` 宏

每个 CUDA 调用都要检查返回值，写一个宏：

```cpp
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_),  \
              __FILE__, __LINE__, cudaGetErrorString(err_));                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
```

launch 之后的习惯写法是 `CUDA_CHECK(cudaGetLastError());`。PyTorch 里对应的是 `C10_CUDA_KERNEL_LAUNCH_CHECK()`（`c10/cuda/CUDAException.h`，v2.10.0），它展开为 `C10_CUDA_CHECK(cudaGetLastError())`，抛出带文件行号的 C++ 异常而不是 `exit`。

### 3. `compute-sanitizer`

一句话：`compute-sanitizer ./vector_add` 会在 kernel 里每次越界访问、未初始化读、竞争条件（`--tool racecheck`）处精确报出线程坐标与源码行（需要 `-lineinfo`）。它比"等到下一次同步点看到一个 illegal address"快一个数量级，写新 kernel 时先过一遍是划算的。


## 六、编译：nvcc 做了什么

### 1. host 与 device 分离编译

nvcc 不是一个编译器，而是一个驱动程序。它把一个 `.cu` 文件拆成两部分：

```text
.cu ──┬── host 代码   ──→ 交给宿主 C++ 编译器（gcc/clang/MSVC）→ 目标文件
      │                    三尖括号被改写成 cudaLaunchKernel 调用
      │
      └── device 代码 ──→ 前端（基于 LLVM）→ PTX（虚拟 ISA，文本）
                              → ptxas → SASS（真实机器码，cubin）
                              → 打包成 fatbin，嵌入 host 目标文件
```

**PTX** 是一种与具体 GPU 无关的虚拟指令集，类似汇编语言的文本；**SASS** 是某一代 GPU 的真实机器码。同一段 PTX 可以被汇编成不同代 GPU 的 SASS。运行时，driver 从可执行文件里嵌入的 fatbin 中挑出与当前 GPU 匹配的那份 SASS 加载；如果没有匹配的 SASS 但有 PTX，就即时编译（JIT）。

### 2. `-arch`、`-gencode` 与 fatbin

`-arch=sm_80` 是最常用的写法，它是下面这条的缩写：

```bash
nvcc -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80 ...
```

也就是：生成 sm_80 的 SASS，**同时**把 compute_80 的 PTX 也嵌进去。前者让 A100 直接运行，后者让更新的 GPU 可以 JIT。

如果只想要 SASS、不带 PTX（二进制更小、无 JIT 可能）：

```bash
nvcc -gencode arch=compute_80,code=sm_80 ...
```

要同时支持多代硬件，就写多个 `-gencode`，得到一个包含多份 SASS 的 fatbin：

```bash
nvcc -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_86,code=sm_86 \
     -gencode arch=compute_90,code=sm_90 \
     -gencode arch=compute_90,code=compute_90 \
     -o kernel kernel.cu
```

PyTorch 的 wheel 就是这样构建的（`TORCH_CUDA_ARCH_LIST` 环境变量控制列表），所以一个 wheel 能在从 Volta 到 Hopper 的 GPU 上运行。代价是编译时间和二进制体积随架构数线性增长——一个 CUTLASS 重的项目多加一个架构，编译时间可能多几十分钟。

### 3. `compute_XX` 与 `sm_XX` 的区别，以及前向兼容

这是最容易混淆的一对概念：

- `compute_XX` 是**虚拟架构**，决定生成的 PTX 可以用哪些指令特性（比如 `compute_80` 才有 `cp.async` 和 BF16 的 `mma`）；
- `sm_XX` 是**真实架构**，决定 SASS 给哪一代 GPU 用。

`arch=compute_80,code=sm_80` 的意思是"按 compute_80 的特性集生成 PTX，再把它汇编成 sm_80 的机器码"。`code=compute_80` 则是"把 PTX 本身也放进 fatbin"。

兼容规则：

- **SASS 在同一大版本内向前兼容**：sm_80 的 cubin 可以在 sm_86（RTX 30 系）、sm_89（RTX 40 系 / L40）上运行，但不能在 sm_90（H100）上运行；
- **PTX 向前兼容**：compute_80 的 PTX 可以被更新的 driver JIT 成任何 ≥ sm_80 的 SASS，包括 H100 甚至更新的架构；
- 反过来都不行：compute_90 的 PTX 不能跑在 A100 上。

JIT 的代价是第一次加载时几秒到几十秒的编译（结果缓存在 `~/.nv/ComputeCache`，默认上限可用 `CUDA_CACHE_MAXSIZE` 调），以及 JIT 出的代码可能不如 ptxas 针对该架构离线编译的优——它不知道新架构的调度细节，也无法用新架构独有的指令。所以生产环境的做法是：为所有目标架构提供 SASS，再附一份最高版本的 PTX 兜底。

一个常见的错误现象值得记住：在 H100 上运行一个只有 `-gencode arch=compute_80,code=sm_80`（没有 PTX）编出来的程序，会得到 `cudaErrorNoKernelImageForDevice`——"no kernel image is available for execution on the device"。

### 4. 两个必备的编译选项

`-lineinfo` 让 SASS 带上源码行号映射，Nsight Compute 的 Source 视图与 `compute-sanitizer` 的报错都靠它。它**不**影响优化（不同于 `-G`，后者关闭所有优化、只用于单步调试），所以 benchmark 时也可以开着。

`--ptxas-options=-v`（或 `-Xptxas -v`）让 ptxas 打印每个 kernel 的资源用量：

```text
ptxas info    : Compiling entry function '_Z14vector_add_f32PKfS0_Pfm' for 'sm_80'
ptxas info    : Function properties for _Z14vector_add_f32PKfS0_Pfm
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 16 registers, 380 bytes cmem[0]
```

（具体寄存器数随 CUDA 版本略有不同，vector add 这样的 kernel 通常在十几个。）四个数字要看：

- **registers**：每线程用了多少寄存器，直接决定 occupancy 上限。每 SM 65536 个寄存器，一个用 R 个寄存器、T 个线程的 block 最多能驻留 $$\lfloor 65536 / (R \cdot T) \rfloor$$ 个（还受其他限制）。R = 16、T = 256 时是 16，早已超过其他上限；R = 128 时只剩 2 个 block、512 线程，占 SM 容量的 25%。
- **spill stores / loads**：寄存器不够时编译器把变量溢出到 local memory（实际在 L1/L2/HBM）。非零就要警惕，GEMM 篇会反复看这个数字。
- **stack frame**：非零通常意味着递归、大数组或未内联的函数调用。
- **smem**：静态 shared memory 用量，本篇的 kernel 没有，从第四篇开始有。

本系列所有 CUDA 示例的默认编译命令：

```bash
nvcc -O3 -arch=sm_80 -lineinfo --ptxas-options=-v -o vector_add vector_add.cu
```


## 七、warp 的执行方式

### 1. 一条指令、32 个线程、一个 mask

warp 是硬件调度和执行的单位。一个 warp 调度器每个周期选出一个就绪的 warp，发射它的下一条指令，这条指令同时作用于 32 个线程的数据（SIMT，单指令多线程）。每条指令带一个 32 位的 **active mask**，哪些 lane 真正执行这条指令由 mask 决定。

程序员写的是"每个线程做什么"，硬件执行的是"每个 warp 做什么"。两者一致的前提是 warp 内 32 个线程走的是同一条路径。当它们不一致时，就出现了**分支发散**（branch divergence）。

### 2. 分支发散的代价

```cpp
if (threadIdx.x % 2 == 0) {
  x = f(x);       // 偶数 lane 执行，奇数 lane 被 mask 掉，但仍占用发射槽位
} else {
  x = g(x);       // 奇数 lane 执行，偶数 lane 被 mask 掉
}
```

warp 不能一半执行 `f`、一半执行 `g`；它会**两路都走**：先以 mask = 偶数 lane 执行 `f` 的所有指令，再以 mask = 奇数 lane 执行 `g` 的所有指令，最后在 `if` 之后重新会合（reconverge）。执行时间是两路之和，而每一路只有一半的 lane 在做有用功——等效吞吐减半。嵌套 `if` 或 `switch` 的分支数越多，损失越大；极端情况是 32 路各不相同，吞吐降为 1/32。

关键是**发散只发生在 warp 内部**。按 warp 边界对齐的分支没有代价：

```cpp
if (threadIdx.x < 32) { ... }        // warp 0 全走 if，其他 warp 全跳过：无发散
if ((threadIdx.x / 32) % 2 == 0) { ... }   // 偶数 warp 走、奇数 warp 不走：无发散
if (i < n) c[i] = a[i] + b[i];       // 只有最后一个 warp 可能发散，可忽略
```

所以 vector add 的边界检查几乎是免费的：$$2^{28}$$ 个元素里，只有 grid 末尾的一个 warp 可能有一部分 lane 被 mask。真正要警惕的是数据依赖的分支——按元素值决定走哪条路径——那是 warp 内每个 lane 各不相同的情形。第四篇的 reduction 和第八篇 attention 的因果掩码都会回到这个问题。

### 3. 独立线程调度：Volta 之后不能再假定 lockstep

Volta（sm_70）之前，一个 warp 只有一个程序计数器（PC），32 个线程严格 lockstep：发散的两路是串行执行的，`if` 之后必定会合。很多早期代码依赖这个隐式保证做"warp 同步编程"——比如 shared memory reduction 的最后几步不加 `__syncthreads()`，因为"反正同一个 warp 里的线程是同步的"。

从 Volta 开始，硬件给**每个线程**维护独立的 PC 和调用栈——这叫**独立线程调度**（Independent Thread Scheduling）。目的是让发散路径之间可以交错执行、允许在发散分支里做线程间的生产者-消费者同步而不死锁。代价是：**编译器和硬件不再保证发散后的线程何时、是否会合。** 上面那种不加同步的 reduction 从"恰好正确"变成"可能读到没写完的值"。

因此 CUDA 9 起所有 warp 级原语都换成了带 `_sync` 后缀、显式接收 mask 的版本：

```cpp
// mask 指明哪些 lane 参与；这些 lane 必须全部执行到这条指令，硬件会在此处把它们会合
v = __shfl_down_sync(0xffffffffu, v, 16);
unsigned ballot = __ballot_sync(0xffffffffu, pred);
__syncwarp();                     // 等价于 __syncwarp(0xffffffff)：warp 内的内存屏障 + 会合点
```

不带 `_sync` 的旧版本（`__shfl`、`__ballot`、`__any`）已被标记废弃并在新架构上移除。规则很简单：**凡是假定"warp 内其他线程已经执行到某处"的地方，都要有一个 `_sync` 原语或 `__syncwarp()` 把这个假定写出来。**

`__syncwarp(mask)` 的语义是：mask 中的所有 lane 到达这里之前谁也不能往下走，并且它们在此之前对 shared/global memory 的写入对彼此可见。它比 `__syncthreads()` 便宜得多——后者同步整个 block。

### 4. active mask 与部分 warp

mask 参数通常填 `0xffffffff`（全 warp），但要保证 32 个 lane 确实都会执行到那条指令。在边界处这不成立——最后一个 warp 可能只有一部分 lane 满足 `i < n`：

```cpp
// 错误：i >= n 的 lane 不会执行到 __shfl_down_sync，但 mask 说它们参与 → 未定义行为
if (i < n) { v = __shfl_down_sync(0xffffffffu, v, 1); }

// 正确：先用 ballot 算出真正会进入分支的 lane 集合，再作为 mask
unsigned mask = __ballot_sync(0xffffffffu, i < n);
if (i < n) { v = __shfl_down_sync(mask, v, 1); }
```

`__activemask()` 返回"此刻哪些 lane 恰好活跃"，看起来像是 mask 的现成答案，但它**不是**同步点：独立线程调度下，两个本应一起到达的 lane 可能一先一后，`__activemask()` 只报告先到的那些。用 `__ballot_sync` 在一个明确的会合点算 mask 才是正确做法。

对本篇的 vector add 这一切都用不上——它没有线程间通信。但第四篇的 warp shuffle reduction 会立刻用到。


## 八、第一个 kernel 的测量

### 1. 先算理论：3 GiB 要多久

取 $$n = 2^{28}$$ 个 float。每个数组 $$2^{28} \times 4 = 2^{30}$$ 字节 = 1 GiB；两读一写，总流量 3 GiB $$\approx 3.22 \times 10^9$$ 字节。A100 标称 2.0 TB/s：

$$
T_{\min} = \frac{3 \times 2^{30}}{2.0 \times 10^{12}} \approx 1.61\ \text{ms}
$$

按 3 GB 粗算是 1.5 ms，按精确的 3 GiB 是 1.6 ms；本文后面以"约 1.5–1.6 ms"称之。这是任何实现都不可能突破的下界，因为这 3 GiB 每个字节都必须经过 HBM 一次。

BF16 版本每个数组 512 MiB、总流量 1.5 GiB，下界约 0.8 ms。FLOPs 是 $$2^{28}$$ 次加法 = 0.27 GFLOP，在 19.5 TFLOPS 的 FP32 CUDA Core 上只需 14 µs，与访存时间相差两个数量级——Roofline 的判断再一次得到确认：算力无关紧要，只看字节。

### 2. L2 flush 为什么必要

A100 有 40 MB L2。如果 benchmark 的工作集小于 40 MB（比如 $$n = 2^{20}$$ 的 float 数组，三个共 12 MB），第一次运行把数据从 HBM 搬进 L2，**第二次及以后的运行直接命中 L2**——L2 带宽是 HBM 的数倍，测出的"带宽"会远超 2.0 TB/s，是虚高的。而在真实工作负载里，一个 elementwise kernel 的输入通常是上一个 kernel 刚写出来的、可能在 L2 里也可能不在，取决于中间隔了多少其他 kernel——最保守、最可复现的假设是**不在**。

解决方法：在每次计时迭代之前，对一块**至少 2 倍 L2 大小**的缓冲区做一次 `cudaMemsetAsync`，把 L2 里的旧数据全部逐出。A100 取 128 MB（H100 L2 是 50 MB，128 MB 同样够用）。memset 在同一 stream 上排在 kernel 之前，`start` event 记录在 memset 之后，所以它的时间不会被算进去。

本篇 $$n = 2^{28}$$ 的工作集是 3 GiB，远大于 L2，flush 与否差别不大；但脚手架要通用到第十篇的所有 kernel，其中很多（RMSNorm 一行 16 KiB、decode 阶段的小 GEMM）工作集都在 L2 以内，flush 是必需的。`triton.testing.do_bench` 默认也做同样的事。

### 3. 完整的 C++ 程序与 `bench` 脚手架

下面是完整可编译的程序，包含 FP32 与 BF16 两个 kernel、`CUDA_CHECK`、事件计时、warmup、多次迭代取中位数、L2 flush。函数名 `bench(fn, warmup, iters, flush_l2)` 会被后面所有篇复用。

```cpp
// vector_add.cu
// nvcc -O3 -arch=sm_80 -lineinfo --ptxas-options=-v -o vector_add vector_add.cu
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_),  \
              __FILE__, __LINE__, cudaGetErrorString(err_));                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------- kernels
__global__ void vector_add_f32(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ c, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

__global__ void vector_add_bf16(const __nv_bfloat16* __restrict__ a,
                                const __nv_bfloat16* __restrict__ b,
                                __nv_bfloat16* __restrict__ c, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) c[i] = __hadd(a[i], b[i]);
}

__global__ void f32_to_bf16(const float* __restrict__ in,
                            __nv_bfloat16* __restrict__ out, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2bfloat16(in[i]);
}

// ---------------------------------------------------------------- bench
// 对一块 128 MB（≥ 2 × A100 L2）的缓冲区做 memset，把 L2 中的数据全部逐出。
static void flush_l2(cudaStream_t stream) {
  static void* buf = nullptr;
  static const size_t bytes = 128u << 20;
  if (buf == nullptr) CUDA_CHECK(cudaMalloc(&buf, bytes));
  CUDA_CHECK(cudaMemsetAsync(buf, 0, bytes, stream));
}

// fn(stream) 负责在给定 stream 上发射待测 kernel。返回中位数毫秒。
template <typename Fn>
float bench(Fn&& fn, int warmup = 10, int iters = 100, bool flush = true,
            cudaStream_t stream = 0) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (int i = 0; i < warmup; ++i) fn(stream);
  CUDA_CHECK(cudaGetLastError());                 // 捕获 launch 配置错误
  CUDA_CHECK(cudaStreamSynchronize(stream));      // 捕获 warmup 中的异步错误

  std::vector<float> times(iters);
  for (int i = 0; i < iters; ++i) {
    if (flush) flush_l2(stream);
    CUDA_CHECK(cudaEventRecord(start, stream));
    fn(stream);
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
  }
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  std::sort(times.begin(), times.end());
  return times[iters / 2];
}

// ---------------------------------------------------------------- main
int main() {
  const size_t n = size_t(1) << 28;                        // 2^28 个元素
  const size_t bytes_f32 = n * sizeof(float);              // 1 GiB
  const size_t bytes_bf16 = n * sizeof(__nv_bfloat16);     // 512 MiB

  // host 侧初始化
  std::vector<float> h_a(n), h_b(n), h_c(n);
  for (size_t i = 0; i < n; ++i) {
    h_a[i] = (float)(i % 1024) * 0.001f;
    h_b[i] = 1.0f;
  }

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, bytes_f32));
  CUDA_CHECK(cudaMalloc(&d_b, bytes_f32));
  CUDA_CHECK(cudaMalloc(&d_c, bytes_f32));
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes_f32, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes_f32, cudaMemcpyHostToDevice));

  const int block = 256;
  const unsigned grid = (unsigned)((n + block - 1) / block);   // 1,048,576 个 block

  // ---- FP32
  float ms_f32 = bench([&](cudaStream_t s) {
    vector_add_f32<<<grid, block, 0, s>>>(d_a, d_b, d_c, n);
  });
  double traffic_f32 = 3.0 * (double)bytes_f32;                // 两读一写
  printf("f32  n=2^28: %.3f ms  %.0f GB/s  (理论下界 %.3f ms @ 2.0 TB/s)\n",
         ms_f32, traffic_f32 / (ms_f32 * 1e-3) / 1e9, traffic_f32 / 2.0e12 * 1e3);

  // 正确性抽查
  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes_f32, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < n; i += n / 64) {
    if (h_c[i] != h_a[i] + h_b[i]) {
      fprintf(stderr, "mismatch at %zu: %f vs %f\n", i, h_c[i], h_a[i] + h_b[i]);
      return EXIT_FAILURE;
    }
  }

  // ---- BF16：在 device 上把 FP32 输入转成 BF16
  __nv_bfloat16 *d_a16 = nullptr, *d_b16 = nullptr, *d_c16 = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a16, bytes_bf16));
  CUDA_CHECK(cudaMalloc(&d_b16, bytes_bf16));
  CUDA_CHECK(cudaMalloc(&d_c16, bytes_bf16));
  f32_to_bf16<<<grid, block>>>(d_a, d_a16, n);
  f32_to_bf16<<<grid, block>>>(d_b, d_b16, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float ms_bf16 = bench([&](cudaStream_t s) {
    vector_add_bf16<<<grid, block, 0, s>>>(d_a16, d_b16, d_c16, n);
  });
  double traffic_bf16 = 3.0 * (double)bytes_bf16;
  printf("bf16 n=2^28: %.3f ms  %.0f GB/s  (理论下界 %.3f ms @ 2.0 TB/s)\n",
         ms_bf16, traffic_bf16 / (ms_bf16 * 1e-3) / 1e9, traffic_bf16 / 2.0e12 * 1e3);

  CUDA_CHECK(cudaFree(d_a));   CUDA_CHECK(cudaFree(d_b));   CUDA_CHECK(cudaFree(d_c));
  CUDA_CHECK(cudaFree(d_a16)); CUDA_CHECK(cudaFree(d_b16)); CUDA_CHECK(cudaFree(d_c16));
  return 0;
}
```

几处设计说明：

- **warmup**：第一次 launch 包含上下文初始化、模块加载（可能触发 JIT）、GPU 从低功耗状态升频等一次性开销，可能比稳态慢几倍到几十倍。10 次 warmup 足以进入稳态。
- **中位数而不是平均值**：GPU 计时的噪声是单边的——偶尔会有某次迭代因为时钟调整、其他进程抢占等原因慢很多，几乎不会有异常快的。平均值被这些离群点拉高，中位数不受影响。报告分布（最小值、中位数、P90）比一个数更好，第十篇会回到这个话题。
- **event 记录在 flush 之后**：确保 memset 的时间不计入。
- **只在 warmup 后同步一次**：`cudaEventSynchronize(stop)` 每次迭代都会等 GPU，本身就是同步点，能捕获异步错误；额外的 `cudaStreamSynchronize` 只是让 warmup 阶段的错误更早暴露。
- **BF16 版本用 `__hadd`**：`cuda_bf16.h` 提供的 BF16 加法内建函数；在 sm_80 及以上是原生指令，在更老的架构上会展开成 FP32 转换再加。PyTorch 和 vLLM 的 BF16 elementwise 通常先转 float 计算再转回去，精度更好，下一篇会讨论两者的选择。

### 4. Python/PyTorch 侧的等价脚手架

同一个 kernel 用 `torch.utils.cpp_extension.load_inline` 接进 PyTorch，用 `torch.cuda.Event` 计时。这是本系列后面测试正确性（`torch.testing.assert_close`）与对照 PyTorch 自带算子的标准方式。

```python
# bench_vector_add.py
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

__global__ void vector_add_bf16(const __nv_bfloat16* __restrict__ a,
                                const __nv_bfloat16* __restrict__ b,
                                __nv_bfloat16* __restrict__ c, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) c[i] = __hadd(a[i], b[i]);
}

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(a.scalar_type() == torch::kBFloat16 && b.scalar_type() == torch::kBFloat16,
              "inputs must be bf16");
  TORCH_CHECK(a.sizes() == b.sizes(), "shape mismatch");
  auto c = torch::empty_like(a);
  size_t n = a.numel();
  if (n == 0) return c;
  const int block = 256;
  const unsigned grid = (unsigned)((n + block - 1) / block);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  vector_add_bf16<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(c.data_ptr<at::BFloat16>()), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}
"""
cpp_src = "torch::Tensor vector_add(torch::Tensor a, torch::Tensor b);"

# 目标架构由 PyTorch 根据当前 GPU（或 TORCH_CUDA_ARCH_LIST）自动加 -gencode，不要再手动传 -arch
mod = load_inline(
    name="vector_add_ext",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["vector_add"],
    extra_cuda_cflags=["-O3", "-lineinfo"],
    verbose=False,
)


def bench(fn, warmup=10, iters=100, flush_l2=True):
    """返回中位数毫秒。与 C++ 版 bench 语义一致。"""
    flush_buf = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        if flush_l2:
            flush_buf.zero_()          # 在当前 stream 上排队，起 L2 flush 作用
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


if __name__ == "__main__":
    n = 1 << 28
    a = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, device="cuda", dtype=torch.bfloat16)

    c = mod.vector_add(a, b)
    torch.testing.assert_close(c, a + b)      # bf16 默认 rtol=1.6e-2, atol=1e-5

    traffic = 3 * n * a.element_size()        # 两读一写，字节
    for name, fn in [("ours", lambda: mod.vector_add(a, b)),
                     ("torch.add", lambda: torch.add(a, b))]:
        ms = bench(fn)
        print(f"{name:10s} {ms:.3f} ms  {traffic / (ms * 1e-3) / 1e9:.0f} GB/s"
              f"  (lower bound {traffic / 2.0e12 * 1e3:.3f} ms @ 2.0 TB/s)")
```

`torch.testing.assert_close` 对 BF16 的默认容差是 rtol = 1.6e-2、atol = 1e-5。这里两边都是单次 BF16 加法（`torch.add` 内部先转 float 再舍回 BF16，`__hadd` 在 sm_80 上直接做 BF16 加法并舍入），结果按 IEEE 舍入到最近的 BF16，通常逐位相同；即使因为舍入路径差异有 1 ulp 的不同，也远在容差内。

也可以直接用 `triton.testing.do_bench(fn, warmup=..., rep=...)`，它内部做的事和上面的 `bench` 一样（默认也用 256 MB 的缓冲区 flush L2），并支持返回分位数。

### 5. 结果应该落在哪里，为什么跑不满

理论下界 1.6 ms、带宽 2.0 TB/s。读者在 A100 上跑上面的程序，FP32 版本的中位数通常会落在 **1.7–2.0 ms**，等效带宽 **1.6–1.9 TB/s**，即标称值的 **80–92%**；BF16 版本在 0.85–1.0 ms 附近。`torch.add` 会稍好一点，因为 ATen 的 elementwise kernel 做了向量化访存（下一篇的主题）。差距去了哪里，可以分成四个部分：

**（1）可达带宽本身低于标称值。** 2.0 TB/s 是 HBM 接口的峰值数据率。DRAM 需要周期性刷新（refresh），行切换有时序约束，读写方向切换（read-write turnaround）有惩罚，每个事务还带协议开销。这些让**任何 kernel** 都只能达到标称值的约 **85–92%**——这是第一篇提到过的"可达带宽"（achievable bandwidth）。用 `nvbandwidth` 或简单的 `cudaMemcpyDeviceToDevice` 测出的设备内拷贝带宽就是这个数字，它才是 elementwise kernel 应当对照的实际屋顶。以 1.75–1.85 TB/s 计，3 GiB 的实际下界是 1.74–1.84 ms。**所以一个跑到 1.8 ms 的 vector add 已经几乎贴着实际屋顶了**——这是本篇最重要的一个校准。

**（2）launch 开销与尾部效应。** 一次 kernel launch 从 GPU 收到命令到第一个 warp 开始执行有几微秒的延迟；kernel 末尾，最后一波 block 结束时 SM 逐渐空闲，硬件不能瞬间填满——这叫尾部效应（tail effect）。两者合计在几微秒到十几微秒量级，对一个 1.6 ms 的 kernel 只占 1% 以下；但对一个 10 µs 的小 kernel（decode 阶段的典型情形）就是 50%。这是为什么小 kernel 永远跑不出高带宽、也是为什么融合（第三、九篇）重要。

**（3）grid 太小时 SM 未填满。** 本篇 grid 有一百万个 block，不存在这个问题。但如果 $$n$$ 只有 $$10^5$$，grid 只有 400 个 block，每个 SM 摊到不到 4 个 block、不到 1024 线程——占 SM 容量的一半，在飞的访存请求数不够，带宽自然上不去。第一篇讨论过的"GPU 靠并发隐藏延迟"在这里变成了具体的数字：一个 SM 要维持接近满带宽，需要足够多的字节在飞（bytes in flight ≈ 带宽 × 延迟；2 TB/s × 约 600 ns 的 HBM 延迟 ≈ 1.2 MB，摊到 108 个 SM 每个约 11 KB），每线程只发一个 4 字节的读请求时，需要几千个线程同时驻留才能凑够。

**（4）每线程一个元素，warp 数不足以隐藏延迟。** 这与（3）相关但不同。即使 grid 足够大，每个线程的工作也只是"读 8 字节、写 4 字节、结束"。一个 warp 发出加载请求后要等约几百个周期才能拿到数据，期间调度器必须切到别的 warp；而每个 warp 从开始到结束只发三条访存指令，warp 的生命周期太短，调度器频繁地在 block 退出和新 block 进驻之间切换，SM 的实际并发访存量达不到上限。解决方法是让每个线程做更多事：用 `float4` 一次读 16 字节（向量化访存），用 grid-stride loop 让一个线程处理多个元素。这两个手段是**下一篇**的全部内容，它们能把 elementwise kernel 从 80% 推到可达带宽的 95% 以上。

回到核心问题：**vector add 的五行 kernel 跑出了理论带宽的 80–90%，没跑满的部分主要是 DRAM 本身的物理开销（不可消除）、加上每线程一个元素造成的访存并发不足（下一篇消除）。** 一个只有五行的 kernel 能到这个水平，是因为它的访存模式恰好是理想的：warp 内 32 个线程读 32 个相邻的 float，正好是 128 字节一条 cache line。下一篇会说明这个"恰好"背后的规则，以及打破它的代价。


## 九、小结

### 1. 要点回顾

- CUDA 程序跨两个处理器与两个地址空间。`__global__` 函数是 kernel，描述单个线程的工作；host 用 `<<<grid, block, shmem, stream>>>` 发射，launch 立即返回。
- 线程按 grid → block → thread 三层组织；全局索引 $$i = \text{blockIdx.x} \cdot \text{blockDim.x} + \text{threadIdx.x}$$，边界检查不可省。block 取 32 的倍数、常用 128–256；grid = ceil(n / block)。
- block 被线性化后每 32 个线程切成一个 warp，`x` 是变化最快的维度；二维 block (16, 16) 的一个 warp 跨两行。
- `cudaMalloc/cudaFree` 慢且 `cudaFree` 隐含同步，所以 PyTorch 用 Caching Allocator。
- stream 是有序队列，event 是 GPU 侧时间戳；用 event 计时，不要用 CPU 时钟；`cudaDeviceSynchronize` 打断 CPU-GPU 流水，只在调试与 benchmark 里用。
- 异步错误在下一次同步点才报出、且粘性不可恢复；每个 CUDA 调用包 `CUDA_CHECK`，launch 后查 `cudaGetLastError`，调试用 `CUDA_LAUNCH_BLOCKING=1` 与 `compute-sanitizer`。
- nvcc 分离编译 host 与 device；`-arch=sm_80` = SASS + PTX；SASS 同大版本内向前兼容，PTX 靠 JIT 跨大版本；`compute_XX` 是虚拟架构、`sm_XX` 是真实架构；`-lineinfo` 与 `--ptxas-options=-v` 常开。
- warp 内分支发散时两路都走、以 mask 屏蔽；Volta 之后线程有独立 PC，不能假定 lockstep，warp 内通信必须用 `_sync` 原语或 `__syncwarp()`。
- vector add：先算下界（3 GiB / 2.0 TB/s ≈ 1.6 ms），再测（通常 1.7–2.0 ms），差距来自 DRAM 物理开销（可达带宽约 85–92%）、launch 与尾部、SM 填充不足、每线程工作量太小。

### 2. 速查表

```text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
launch 配置速查
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
全局索引（1D）        i = blockIdx.x * blockDim.x + threadIdx.x      （乘法用 size_t）
全局索引（2D）        row = blockIdx.y*blockDim.y + threadIdx.y；col 用 x
block 大小            32 的倍数；常用 128–256；上限 1024
grid 大小             (n + block - 1) / block；x 维上限 2^31 - 1
warp 划分             linear = x + y*bdx + z*bdx*bdy；warp = linear / 32
每 SM 上限（A100）    2048 线程 / 64 warp / 32 block / 65536 寄存器 / 164 KB shared
block 数 ≤ SM 数时    大部分 SM 空闲 → 带宽上不去

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
编译选项速查
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
-arch=sm_80                            = compute_80 PTX + sm_80 SASS
-gencode arch=compute_80,code=sm_80    只有 SASS，不能 JIT 到更新架构
-gencode arch=compute_90,code=compute_90   只有 PTX，靠 JIT
多个 -gencode                          fatbin，多份 SASS + 一份最高 PTX 兜底
-lineinfo                              源码行号映射，不影响优化，常开
-G                                     关闭优化，仅单步调试
--ptxas-options=-v                     打印 registers / spill / smem / stack
-O3                                    host 与 device 侧优化
SASS 兼容                              同大版本向前（sm_80 → sm_86/89），跨大版本不行
PTX 兼容                               向前 JIT 到任意更新架构

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
benchmark 脚手架要点          bench(fn, warmup=10, iters=100, flush_l2=True) → 中位数 ms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
计时                  cudaEventRecord / ElapsedTime；Python 用 torch.cuda.Event
warmup                ≥ 10 次：上下文、模块加载、JIT、升频
统计                  中位数（噪声单边）；报告分布更好
L2 flush              每次迭代前 cudaMemsetAsync 一块 128 MB 缓冲（≥ 2 × 40 MB L2）
错误                  warmup 后 cudaGetLastError + 同步；每次 EventSynchronize 也是同步点
stream                PyTorch 扩展里用 at::cuda::getCurrentCUDAStream()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
vector add 数字（A100，n = 2^28）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FP32  流量 3 GiB      理论下界 ≈ 1.6 ms @ 2.0 TB/s     常见实测区间 1.7–2.0 ms
BF16  流量 1.5 GiB    理论下界 ≈ 0.8 ms                常见实测区间 0.85–1.0 ms
FLOPs                 0.27 GFLOP → 14 µs @ 19.5 TFLOPS   与访存差两个数量级
可达带宽              标称的 85–92%（DRAM refresh、turnaround、协议开销）
差距来源              可达带宽 > 每线程工作量太小 > launch/尾部 > SM 填充
```


## 下一篇

[访存合并与 elementwise kernel](/memory-coalescing-and-elementwise-kernels.html)
