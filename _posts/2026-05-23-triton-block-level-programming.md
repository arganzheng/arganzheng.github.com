---
layout: post
title: "GPU Kernel 工程（07）：Triton——块级编程与编译器的边界"
subtitle: "Triton: Block-Level Programming and Where the Compiler Stops"
tags: [CUDA, Triton, GPU, AI, AI-Infra]
catalog: true
---

> 本文是[《GPU Kernel 工程：从 CUDA 执行模型到 FlashAttention》](/gpu-kernel-engineering.html)系列的第 7 篇（共十篇）。上一篇：[Tensor Core、CUTLASS 与 CuTe](/tensor-cores-cutlass-and-cute.html)　下一篇：[Attention Kernel：FlashAttention 与 PagedAttention](/attention-kernels-flashattention-and-pagedattention.html)

前六篇一直在 CUDA 的世界里：每个线程算什么、warp 怎么合并访存、shared memory 怎么分块、`mma.sync` 怎么喂 fragment。到了第六篇，一个能跑到 cuBLAS 七八成性能的 BF16 GEMM 已经是两三百行代码，而且每一行都有"为什么这样写"的理由——tile 尺寸、bank conflict 的 padding、`cp.async` 的 stage 数、寄存器分块的形状。

这一篇换一种写法。Triton 把"线程"从编程模型中拿掉：程序员以 **block** 为单位思考，写的是"这个 block 加载哪一块数据、做什么张量运算、存到哪里"，而线程到元素的映射、shared memory 的分配、向量化、软件流水、`mma` 指令的选择，全部交给编译器。代价是失去对这些东西的控制权。

总纲对这一篇提出的核心问题是：

> **Triton 的 matmul 比手写 CUDA 少 80% 的代码，性能只差 10%。那 10% 在哪里？什么场景下这 10% 值得手写？**

回答这个问题需要三步：先弄清 Triton 的编程模型到底把什么交给了编译器（第一章）；用 Triton 重写前几篇的三个 kernel，对照代码量和性能（第二章）；然后打开编译器，看它在每一层做了什么优化、留下了什么做不到的（第三、五章）。中间穿插两类"生产中的 Triton"：`torch.compile` 生成的 kernel，和 vLLM 里的 `fused_moe_kernel` 与 prefix-prefill attention（第四章）。

全文延续系列的方法论：**每个 kernel 先算理论上应该多快，再看实现，再解释差距**。硬件基线取 A100 SXM 80GB 的公开标称值：HBM2e 约 2.0 TB/s，BF16 Tensor Core 312 TFLOPS dense，108 个 SM，L2 40 MB。本文没有 GPU 实测，所有性能数字要么是可推导的理论下界，要么用"通常能达到"的区间给出。软件基线为 Triton 3.x、PyTorch v2.10.0、vLLM v0.20.0。


## 一、把"线程"拿掉之后：Triton 的编程模型

### 1. 一个 program 对应一个 CUDA block

CUDA 的执行模型有三层：grid、block、thread。程序员写的是**一个线程**的代码，用 `blockIdx`、`threadIdx` 算出自己负责哪个元素。Triton 只保留了前两层：

- **grid** 仍然是 grid，可以是一维到三维；
- **program** 对应 CUDA 的一个 block，用 `tl.program_id(axis)` 拿到自己的坐标（对应 `blockIdx.x/y/z`）；
- **没有 thread**。一个 program 的代码操作的是**块级张量**（block-level tensor）——形状固定、在编译期已知的一小块数据，比如 `[BLOCK_SIZE]` 或 `[BLOCK_M, BLOCK_N]`。

一个 program 底层仍然由 `num_warps × 32` 个线程执行（`num_warps` 默认 4，即 128 线程），但这些线程如何分摊张量里的元素，程序员看不到、也不能指定。这就是 Triton 与 CUDA 最本质的区别：**CUDA 是 SIMT（每个线程一份标量代码），Triton 是 SPMD 之上的块级张量语言（每个 block 一份张量代码）**。

用最小的 elementwise 例子对照两者：

```cpp
// CUDA：一个线程处理一个元素
__global__ void add_kernel(const float* x, const float* y, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = x[idx] + y[idx];
}
```

```python
# Triton：一个 program 处理 BLOCK_SIZE 个元素
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # [BLOCK_SIZE] 的整数张量
    mask = offs < n                                        # [BLOCK_SIZE] 的布尔张量
    x = tl.load(x_ptr + offs, mask=mask)                   # [BLOCK_SIZE]
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)
```

CUDA 版的 `idx` 是一个标量，Triton 版的 `offs` 是一个长度为 `BLOCK_SIZE` 的张量；CUDA 版的 `if (idx < n)` 是一个分支，Triton 版的 `mask` 是一个逐元素的谓词。CUDA 里"128 个线程各自算一个 `idx`"这件事，在 Triton 里被 `tl.arange` 一次性表达为一个张量，由编译器决定这 `BLOCK_SIZE` 个元素怎么分给 128 个线程。

### 2. 四个原语：program_id、arange、load/store、mask

Triton 的核心原语只有几个，全部围绕"块级张量"：

**`tl.program_id(axis)`**：当前 program 在 grid 的第 `axis` 维上的坐标，标量。`tl.num_programs(axis)` 给出该维的 grid 大小。

**`tl.arange(start, end)`**：生成 `[start, end)` 的整数张量。**`end - start` 必须是 2 的幂**，且在编译期已知——这是 Triton 的硬性限制，来源是编译器需要把元素规整地分配给 warp 与线程（layout，见下节），非 2 的幂会破坏这个分配。处理长度不是 2 的幂的数据，方法是取 `next_power_of_2` 再用 mask 屏蔽多余部分。

**`tl.load(ptr + offs, mask=..., other=...)`**：`ptr + offs` 是一个**指针张量**（每个元素是一个地址），`tl.load` 按这些地址读出同形状的数据张量。`mask` 是同形状的布尔张量，为 False 的位置不发起访存、结果填 `other`（默认 0）。`tl.store(ptr + offs, value, mask=...)` 对称。**Triton 没有 `if (idx < n)` 这种线程级分支来做边界处理，一切边界都靠 mask**。

**块级张量运算**：加减乘除、比较、`tl.where`、类型转换 `.to(tl.float32)` 都是逐元素的，支持 NumPy 风格的广播（`x[:, None]`、`y[None, :]`）；`tl.sum(x, axis)`、`tl.max(x, axis)` 沿某一维归约；`tl.exp`、`tl.log`、`tl.sqrt` 等数学函数；`tl.dot(a, b, acc)` 做 `[M, K] × [K, N]` 的矩阵乘并累加到 `acc`——这是唯一会被编译到 Tensor Core 的运算，要求输入是 FP16/BF16/FP8/INT8（或 TF32 的 FP32）、M/N/K 均不小于 16 且为 2 的幂。

**没有 `__syncthreads()`、没有 `__shared__`**。程序员不能声明 shared memory，也不需要同步：编译器会在需要跨线程交换数据的地方（归约、layout 转换、`tl.dot` 的操作数准备）自动分配 shared memory 并插入 barrier。

### 3. layout：编译器决定线程到元素的映射

把 `tl.arange(0, 1024)` 交给一个 `num_warps=4` 的 program，128 个线程各拿 8 个元素。但**哪 8 个**？连续的 8 个（线程 0 拿 0–7，线程 1 拿 8–15）还是跨步的（线程 0 拿 0, 128, 256, …）？

这就是 **layout** ——Triton 编译器为每个块级张量选定的"线程到元素的映射"。它不在源代码里，而是编译器中间表示（TTGIR）的一部分，第三章会读它的具体编码。这里先给出最重要的直觉：

- 对于要合并访存的 `tl.load`，编译器会选"每个线程拿连续几个元素、相邻线程拿相邻段"的 layout，使一个 warp 的 32 个线程覆盖连续的 32 × 16 字节 = 512 字节，并把每个线程的 8 个 BF16（16 字节）合成一条 128 bit 的向量化 load——这正是第三篇手写 `__nv_bfloat162`/`uint4` 向量化访存所做的事，Triton 自动做了；
- 对于 `tl.sum(x, axis=0)` 这类归约，编译器先在线程内累加自己持有的元素，再用 warp shuffle 在 warp 内归约，最后（如果跨 warp）经 shared memory 归约——第四篇的三段式 reduction，Triton 自动做了；
- 对于 `tl.dot`，操作数会被转换成 `mma` 指令要求的 fragment 布局（第六篇手写的 `ldmatrix` + fragment 排布），结果落在 `mma` 输出的布局上。

**layout 是 Triton 把 CUDA 的线程级细节"藏起来"的具体机制**。它能被藏起来，是因为绝大多数 kernel 里"哪个线程拿哪个元素"只有几种合理的选择，编译器按访存模式和运算类型就能选对；它藏不好的场景，就是第五章要讲的边界。

### 4. `tl.constexpr`：与 CUDA 模板参数的对应

`BLOCK_SIZE: tl.constexpr` 这个类型标注，把参数变成**编译期常量**。它精确对应 CUDA 的模板参数：

```cpp
template <int BLOCK_SIZE>
__global__ void add_kernel(...);      // BLOCK_SIZE 是编译期常量，每个取值一份代码
```

两者的语义完全一致：

- **每一组 `constexpr` 取值生成一份独立的机器码**。`add_kernel[grid](..., BLOCK_SIZE=1024)` 与 `BLOCK_SIZE=2048` 会各自编译一份 cubin。Triton 会把编译结果缓存在 `~/.triton/cache/`（可用环境变量 `TRITON_CACHE_DIR` 改），以源码 hash、constexpr 取值、参数类型（以及指针的 16 字节对齐、整数是否为 16 的倍数等**特化属性**）为 key；命中缓存时不重新编译；
- **`constexpr` 参数可以参与形状**：`tl.arange(0, BLOCK_SIZE)`、`tl.zeros((BLOCK_M, BLOCK_N), ...)` 的形状必须是 constexpr，因为 layout 必须在编译期确定；
- **`constexpr` 上的 `if` 是编译期分支**：`if USE_BIAS:` 里 `USE_BIAS: tl.constexpr` 为 False 时整段代码被删除，不产生运行时开销。vLLM 的 `fused_moe_kernel` 用十几个 constexpr 开关（`use_fp8_w8a8`、`HAS_BIAS`、`MUL_ROUTED_WEIGHT` 等）让一份源码生成几十种量化/非量化的变体，就是这个用法（第四章会看）。

一个容易踩的坑：**普通整数参数也会触发特化**。Triton 默认会对等于 1 的整数参数和能被 16 整除的整数参数做特化（生成不同的 cubin），目的是让编译器知道 stride 为 1（可向量化）或大小是 16 的倍数（可省 mask）。这意味着同一个 kernel 用 `n=1024` 与 `n=1000` 调用可能是两份不同的编译产物——首次遇到新的特化组合时会有一次编译延迟。可以用 `do_not_specialize` 参数关掉。

### 5. `num_warps` 与 `num_stages`：暴露给用户的两个硬件旋钮

Triton 藏起了线程，但留下两个与硬件直接相关的旋钮，都在 launch 时作为关键字参数传入（`kernel[grid](..., num_warps=8, num_stages=3)`），也可以写进 `triton.Config`：

**`num_warps`**：一个 program 用多少个 warp，即 CUDA 的 `blockDim.x / 32`。默认 4（128 线程）。它决定：

- 每个线程持有多少元素——一个 `[128, 128]` 的 FP32 累加器在 `num_warps=4` 时每线程 128 个寄存器，`num_warps=8` 时 64 个。累加器太大会导致寄存器溢出（spill 到 local memory），太小则每线程工作量不足、指令级并行受限；
- occupancy——第二篇的公式在这里照样适用：每 SM 寄存器上限 65536，一个 block 用 `R × num_warps × 32` 个寄存器，`R` 由编译器分配，可以在 `kernel.n_regs` 里读到；`kernel.n_spills` 给出溢出数，不为 0 通常意味着该减 tile 或加 `num_warps`；
- 一般经验：elementwise 用 4；处理 4096 以上长度的 softmax 行用 8 或 16；`[128, 256]` 的 matmul tile 用 8。

**`num_stages`**：软件流水的深度，即编译器把 `for` 循环体（典型是 GEMM 的 K 循环）转换成多级流水时，同时在飞的迭代数。Ampere 上默认 3。它精确对应第五、六篇手写的**多 stage `cp.async` 流水**：`num_stages=3` 意味着 shared memory 里同时保有 3 个 K-tile 的缓冲区，一个在被 `mma` 消费，另两个正在从全局内存异步加载。它的效果与代价：

- 更深的流水能掩盖更长的全局访存延迟，但 shared memory 占用线性增长：一个 `BLOCK_M=128, BLOCK_N=128, BLOCK_K=32` 的 BF16 tile，A 与 B 各 128 × 32 × 2 B = 8 KiB，每 stage 16 KiB，`num_stages=4` 共 64 KiB——超过默认 48 KB 上限，Triton 会自动申请 opt-in 的动态 shared memory，但同一 SM 能驻留的 block 数随之下降；
- Hopper 上，`num_stages` 同样控制 TMA 加载的流水深度；
- 对没有循环的 kernel（elementwise、单遍 softmax），`num_stages` 无意义。

**只有这两个旋钮**是有意的设计：tile 形状（`BLOCK_M/N/K`）通过 constexpr 暴露，线程数与流水深度通过这两个参数暴露，其余一切（layout、shared memory 布局、向量化宽度、指令选择）交给编译器。

### 6. `@triton.autotune` 与 `@triton.heuristics`

既然 tile 形状、`num_warps`、`num_stages` 都是可选的，就需要一个机制选它们。`@triton.autotune` 是 Triton 内建的配置搜索器：

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
                      num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
                      num_warps=4, num_stages=4),
        # ...
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, ...):
    ...
```

它的工作方式：

- **`configs`** 列出候选：每个 `triton.Config` 包含一组 constexpr 取值和 `num_warps`/`num_stages`；
- **`key`** 列出哪些运行时参数的取值变化会触发重新搜索。`key=["M", "N", "K"]` 意味着每一组新的 (M, N, K) 都会跑一次搜索，结果缓存在进程内的字典里（**不持久化到磁盘**，进程重启后重新搜索；这一点常被误解）；
- **搜索代价**：首次遇到新 key 时，Triton 会**逐个编译并 benchmark 每个 config**（每个跑若干次取中位数），选最快的。5 个 config 的搜索通常要几秒；几十个 config 可能要几十秒到几分钟，全在首次调用时发生。生产系统里这就是"第一个请求慢"的来源之一，vLLM 因此把 MoE kernel 的最优配置离线搜好存成 JSON 文件（`fused_moe/configs/` 目录下按 GPU 型号与形状命名），运行时直接查表，不用 autotune；
- **`prune_configs_by`**：一个字典，可以给 `early_config_prune`（一个函数，接收全部 configs 和参数，返回值得 benchmark 的子集，比如按 shared memory 用量或 M 的大小剪掉明显不合适的）和 `perf_model`（一个估算函数，只 benchmark 估算最好的 `top_k` 个），用来压缩搜索时间；
- `TRITON_PRINT_AUTOTUNING=1` 会在搜索结束时打印每个 key 选中的 config 与耗时，是查看"它到底选了什么"的最快办法。

`@triton.heuristics` 是 autotune 的确定性版本：不搜索，而是用一个函数从运行时参数直接算出 constexpr：

```python
@triton.heuristics({"BLOCK_N": lambda args: triton.next_power_of_2(args["N"])})
@triton.jit
def softmax_kernel(x_ptr, out_ptr, stride, N, BLOCK_N: tl.constexpr): ...
```

两者可以叠加：先 heuristics 定死某些 constexpr（如 `EVEN_K = K % BLOCK_K == 0`，用来在编译期去掉 K 方向的 mask），再 autotune 搜索其余的。


## 二、三个 kernel 的 Triton 版本

现在用 Triton 重写第三、四、五篇的三个 kernel。每个都先给理论下界，再给代码，再与 CUDA 版对照代码量和通常能达到的性能。

### 1. BF16 elementwise add

**理论下界**。`y = x + b`，BF16，每元素读 2 × 2 B、写 2 B，共 6 字节，1 FLOP，算术强度 $$1/6 \approx 0.17$$ FLOP/byte，与 A100 BF16 的 ridge point 156 相差三个数量级——彻头彻尾的 memory-bound。取 $$n = 2^{28}$$ 个元素（每个张量 512 MiB），总流量 $$6 \times 2^{28} \approx 1.61 \text{ GB}$$，A100 上的理论时间：

$$
t_{\min} = \frac{1.61 \times 10^{9}\ \text{B}}{2.0 \times 10^{12}\ \text{B/s}} \approx 0.81\ \text{ms}
$$

任何实现都只能逼近这个数字。第三篇的 CUDA 版（每线程 8 个 BF16 的 `uint4` 向量化 + grid-stride）通常能达到标称带宽的 85–92%。

**Triton 版**：

```python
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.is_contiguous() and x.shape == y.shape
    out = torch.empty_like(x)
    n = out.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
    return out
```

几个细节：

- **grid 是一个 lambda**，接收 `meta`（本次 launch 的全部 constexpr 与 `num_warps` 等），返回 grid 元组。这样 grid 的计算可以依赖 `BLOCK_SIZE`，在 autotune 的场景下尤其必要——不同 config 的 `BLOCK_SIZE` 不同，grid 也不同；
- **`n_elements` 不是 constexpr**，它是普通的运行时整数（但会被 16 的倍数特化，见第一章 §4）；
- **mask 的形状**是 `[BLOCK_SIZE]`，与 `offs`、`x`、`y` 一致。最后一个 program 处理尾部时 `mask` 有 False 项，对应位置不读不写；
- BF16 的加法：Triton 会把 `x + y` 编译成 BF16 → FP32 → 加 → BF16 的序列（Ampere 没有 BF16 的标量加法指令，与 CUDA 里 `__hadd` 的实现一致），结果与 PyTorch 的 `x + y` 逐位相同。

编译器为它做的：`BLOCK_SIZE=1024`、`num_warps=4` → 128 线程各 8 个连续元素 → 每线程一条 128 bit 的 `ld.global.v4.b32`，一个 warp 一次读 512 字节连续内存，完全合并。这正是第三篇手写的向量化方案。**代码约 12 行 vs CUDA 版（含向量化、尾部处理、launch）约 50–80 行；性能通常与手写 CUDA 相当（都是带宽的 85–92%）**，因为 memory-bound kernel 只要访存模式对了就到顶了，没有留给手工优化的空间。

一个不同点：Triton 版没有做 grid-stride loop，`n=2^28` 时 grid 有 262144 个 program，靠硬件调度器轮转。对于纯 elementwise 这没有可测量的差别；如果想复用 program，可以在 kernel 里写 `for` 循环。

### 2. 按行 softmax

**理论下界**。$$M = 8192$$ 行，每行 $$N = 4096$$ 个 BF16：读 64 MiB、写 64 MiB，共 128 MiB ≈ 134 MB，A100 上：

$$
t_{\min} = \frac{134 \times 10^{6}}{2.0 \times 10^{12}} \approx 67\ \mu\text{s}
$$

这个下界假定每个元素只读一次。第四篇讨论过：朴素三遍实现（max、sum、normalize 各读一遍）读三次，要 3 倍时间；正确的做法是一行由一个 block 负责、把整行读进寄存器（或 shared memory）后在片上完成 max 与 sum，只读一遍全局内存。

**Triton 版**：一行一个 program，`BLOCK_N = next_power_of_2(N)`，整行作为一个 `[BLOCK_N]` 张量加载进寄存器。

```python
@triton.jit
def softmax_kernel(x_ptr, out_ptr, stride_xm, stride_om, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=float("-inf"))
    x = x.to(tl.float32)
    row_max = tl.max(x, axis=0)             # 第一遍：寄存器内归约
    e = tl.exp(x - row_max)                 # 被 mask 的位置：exp(-inf) = 0
    row_sum = tl.sum(e, axis=0)             # 第二遍：寄存器内归约
    y = e / row_sum
    tl.store(out_ptr + row * stride_om + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2 and x.is_cuda
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(N)
    num_warps = 4
    if BLOCK_N >= 2048:
        num_warps = 8
    if BLOCK_N >= 4096:
        num_warps = 16
    softmax_kernel[(M,)](x, out, x.stride(0), out.stride(0), N,
                         BLOCK_N=BLOCK_N, num_warps=num_warps)
    return out
```

这里的"两遍"（一遍 max、一遍 exp/sum）**全部在寄存器里**：`x` 加载一次后就是一个驻留在 128–512 个线程寄存器中的 `[BLOCK_N]` 张量，`tl.max` 和 `tl.sum` 是编译器生成的线程内累加 + warp shuffle + 跨 warp 的 shared memory 归约，与第四篇手写的 warp/block 版本在结构上等价。全局内存只读一次，所以理论上能贴近 67 µs 的下界；实际通常在带宽的 80–90%。

`other=float("-inf")` 是关键：被 mask 掉的列填 $$-\infty$$，于是不影响 `max`，`exp(-inf - m) = 0` 也不影响 `sum`。如果 `N` 本身就是 2 的幂，mask 全为 True，编译器会在特化后把它省掉。

`num_warps` 随 `BLOCK_N` 增大：4096 个 FP32 在 4 个 warp（128 线程）上是每线程 32 个寄存器，在 16 个 warp（512 线程）上是每线程 8 个。前者寄存器压力更大但 warp 少，后者 occupancy 更好。这个经验规则来自 Triton 官方教程 `02-fused-softmax.py`。需要说明的是，**官方教程的当前版本比这里的写法更复杂**：它是一个持久 kernel（persistent kernel），用 `triton.runtime.driver.active.utils.get_device_properties` 查询 SM 数与寄存器数、按 occupancy 算出应该 launch 多少个 program，每个 program 用 `for row_idx in tl.range(row_start, n_rows, row_step, num_stages=...)` 循环处理多行，让多行的加载与计算软件流水。对 $$M = 8192$$ 这样的行数，两种写法的性能差别不大；行数很少、每行很长时持久版本更好。本文用简单版是为了把编程模型看清楚。

**代码约 25 行 vs CUDA 版（warp reduce + block reduce + shared memory + 尾部处理）约 100–150 行；性能通常相当**。同 elementwise 一样，这是 memory-bound kernel，编译器只要做对合并访存与归约就到顶了。

### 3. matmul：`tl.dot` 与 L2 swizzle

**理论下界**。$$4096 \times 4096 \times 4096$$ 的 BF16 GEMM：FLOPs $$= 2 \cdot 4096^3 \approx 137.4$$ GFLOP，最小访存（A、B 各读一次、C 写一次）$$= 3 \times 4096^2 \times 2\ \text{B} = 96\ \text{MiB} \approx 100.7$$ MB，算术强度 $$\approx 1365$$ FLOP/byte，远超 ridge point 156，compute-bound。A100 BF16 Tensor Core 的理论时间：

$$
t_{\min} = \frac{137.4 \times 10^{9}}{312 \times 10^{12}} \approx 0.44\ \text{ms}
$$

cuBLAS 在这个形状上通常能达到标称峰值的 70–80%，即 0.55–0.63 ms。第六篇手写的多 stage `cp.async` + `mma.sync` 版本通常在 cuBLAS 的 70–85%。

**Triton 版**：

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
                      num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
                      num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
                      num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
                      num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
                      num_warps=4, num_stages=5),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # ---- 1. program id -> (pid_m, pid_n)，按 GROUP_SIZE_M 分组重排以提高 L2 命中 ----
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ---- 2. 第一个 K-tile 的指针张量：a_ptrs [BLOCK_M, BLOCK_K]，b_ptrs [BLOCK_K, BLOCK_N] ----
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # ---- 3. K 循环：FP32 累加器，tl.dot 走 Tensor Core ----
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # ---- 4. epilogue：转 BF16，带 mask 写回 ----
    c = acc.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0]
    assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c
```

逐段解释。

**K 循环与 `tl.dot`**。`a` 是 `[BLOCK_M, BLOCK_K]` 的 BF16 张量，`b` 是 `[BLOCK_K, BLOCK_N]`，`tl.dot(a, b, acc)` 计算 `acc += a @ b`，累加器是 FP32 的 `[BLOCK_M, BLOCK_N]`。编译器把它翻译成 Ampere 上的 `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`——一个 `128 × 128 × 32` 的 tile 需要 $$(128/16) \times (128/8) \times (32/16) = 256$$ 条 `mma.sync`，分给 4 个 warp 各 64 条。这对应第六篇手写的 fragment 循环，但程序员不需要知道 `m16n8k16` 是什么。Hopper 上同一行 `tl.dot` 会被编译成 `wgmma.mma_async`。

**mask 与 `other=0.0`**。K 方向的 mask 只在 `K % BLOCK_K != 0` 时的最后一个迭代有作用，越界的位置填 0，对点积没有贡献。M/N 方向没有 mask，而是用了 `% M`、`% N` 取模：越界的行会回卷到矩阵开头，读到的是合法地址上的无用数据，算出来的 `acc` 行在 epilogue 被 `c_mask` 丢弃。这样做是为了让 `tl.load` 在 K 循环内不带 M/N 方向的 mask——mask 会阻碍编译器生成 `cp.async` 的多 stage 流水（带 mask 的加载需要额外的谓词处理）。这是 Triton 官方教程 `03-matrix-multiplication.py` 的写法。

**`GROUP_SIZE_M` 的 L2 swizzle**。这是 matmul kernel 里最不直观、也最值得推导的一段。grid 是一维的，共 $$P = \lceil M/B_M \rceil \times \lceil N/B_N \rceil$$ 个 program。最简单的映射是行主序：`pid_m = pid // num_pid_n`，`pid_n = pid % num_pid_n`。用 $$M = N = K = 4096$$、$$B_M = B_N = 128$$ 代入，$$\lceil 4096/128 \rceil = 32$$，共 1024 个 program。A100 的 108 个 SM 同时驻留的 program 数取决于 occupancy，设一"波"约 108 个（每 SM 一个）：

- **行主序**：前 108 个 program 的 `pid_m` 取 0–3（108/32 ≈ 3.4），`pid_n` 跑遍 0–31。它们需要的数据：A 的 4 个行 panel（每个 $$128 \times 4096 \times 2\ \text{B} = 1$$ MiB），加 B 的**全部** 32 个列 panel（每个 $$4096 \times 128 \times 2\ \text{B} = 1$$ MiB），共约 36 MiB；
- **分组（`GROUP_SIZE_M = 8`）**：把 program 按 8 个 `pid_m` 为一组重排，组内按列主序走：同一组的 $$8 \times 32 = 256$$ 个 program 中，前 108 个覆盖 `pid_m` 0–7 的全部 8 行 panel，`pid_n` 只到 0–13（108/8 = 13.5）。需要的数据：A 的 8 个 panel + B 的 14 个 panel ≈ 22 MiB。

更一般地，一波 $$W$$ 个 program、分组大小 $$G$$ 时需要的 panel 数约为 $$G + W/G$$，在 $$G \approx \sqrt{W}$$ 时最小。这个工作集能否放进 L2 直接决定 A、B 的每个 tile 是从 L2 还是从 HBM 读进 SM。A100 的 L2 是 40 MB：22 MiB 能放下，36 MiB 放不下（且 L2 还要放 C 的写回和其他数据）。第五篇推导过分块 GEMM 的全局读取量是 $$M N K (1/B_N + 1/B_M)$$ 个元素——4096³、128 tile 时是 $$2 \times 4096^3 / 128 \times 2\ \text{B} = 2$$ GiB，是 96 MiB 最小流量的 21 倍。这 2 GiB 中有多少落在 L2、多少真去 HBM，就由 swizzle 决定。**L2 带宽约是 HBM 的数倍**，命中率的差异在 compute-bound 的 GEMM 上通常体现为 5–15% 的性能差别。

映射公式本身：`group_id = pid // (G × num_pid_n)` 是第几组；`first_pid_m = group_id × G` 是组的起始行；组内偏移 `pid % num_pid_in_group` 按列主序拆成 `pid_m = first_pid_m + 偏移 % group_size_m`、`pid_n = 偏移 // group_size_m`。`group_size_m = min(num_pid_m - first_pid_m, G)` 处理最后一组不满 $$G$$ 行的情况。这段代码在 vLLM 的 `fused_moe_kernel` 里几乎逐字出现（第四章）。

**编译器为它做的**（细节见第三章）：为 `a`、`b` 的加载分配 shared memory 缓冲区并选择无 bank conflict 的 swizzled 布局；把 K 循环转成 `num_stages` 级的 `cp.async` 流水；把 `tl.dot` 翻译成 `mma.sync`，并为操作数生成 `ldmatrix`；在 epilogue 把 `mma` 输出布局转成适合合并写回的布局。

**代码约 60 行（不含 autotune 配置）vs CUDA 版约 200–300 行；性能通常达 cuBLAS 的 80–95%，视 shape**——大而规整的形状（4096³）接近上限，小 M（decode 阶段的 $$M = 16$$）或奇怪的 K 会落到下限甚至更低。


## 三、编译器做了什么：从 Python 到 cubin

Triton matmul 能接近 cuBLAS，是因为编译器自动做了第五、六篇手工做的事。要理解"那 10% 在哪里"，先要看清编译器做了哪些、在哪一层做的。

### 1. 编译流水线的六层

```text
Python 源码
   │  @triton.jit：JITFunction 解析 Python AST，按 constexpr 特化
   ▼
TTIR（Triton IR）            与硬件无关的块级张量 IR：tt.load / tt.store / tt.dot / tt.reduce
   │  转换 + 加 layout 编码
   ▼
TTGIR（TritonGPU IR）        每个张量带 layout：#blocked / #shared / #mma / #dot_op
   │  coalescing、pipeline、prefetch、layout 转换消除、去 barrier …
   ▼
LLVM IR（NVPTX 后端）        线程级标量代码，layout 已经被"展开"成线程索引运算
   │  LLVM 优化 + NVPTX codegen
   ▼
PTX                         mma.sync / cp.async / ld.global.v4 …
   │  ptxas
   ▼
cubin（SASS）                寄存器分配、指令调度，缓存到 ~/.triton/cache/
```

每一层做什么：

| 层 | 输入 | 主要工作 | 对应 CUDA 里的手工操作 |
|---|---|---|---|
| Python AST → TTIR | `@triton.jit` 函数源码 | `JITFunction` 用 `ast` 模块解析源码，`CodeGenerator` 把每个节点翻译成 TTIR op；constexpr 取值代入并折叠常量、删除编译期分支；类型推断（BF16、指针、int32/int64） | 模板实例化、`if constexpr` |
| TTIR | `tt.load`、`tt.dot`、`tt.reduce`、`scf.for` | 与硬件无关的优化：公共子表达式消除、循环不变量外提、广播/reshape 的规范化、死代码消除 | 编译器通用优化 |
| TTIR → TTGIR | 加上 layout | 为每个张量选定初始 `#blocked` layout（按 `num_warps` 与元素数）；`tt.dot` 的操作数与结果标为 `#dot_op` / `#mma` | 决定每线程持有哪些元素、fragment 布局 |
| TTGIR passes | 带 layout 的 IR | **Coalesce**：根据指针的连续性分析（`tl.multiple_of`/`tl.max_contiguous` 提示或推断），重选 load/store 的 layout 使每线程持有连续元素 → 128 bit 向量化；**Pipeline**：把 `scf.for` 内的 load 提前 `num_stages - 1` 个迭代，插入 `cp.async` 与 shared memory 环形缓冲；**Prefetch**：把 `mma` 操作数的 shared → 寄存器搬运提前一个子迭代；**RemoveLayoutConversions**：消除冗余的 layout 转换（每次转换意味着一次 shared memory 往返）；**ReorderInstructions**、**OptimizeDotOperands**（把转置折进 `ldmatrix.trans`）；为 `#shared` 选 swizzle 参数避免 bank conflict；插入 barrier | 向量化访存、多 stage `cp.async` 流水、`ldmatrix`、shared memory padding/swizzle、`__syncthreads()` 位置 |
| TTGIR → LLVM IR | | 把 layout "展开"：每个块级 op 变成每线程对自己持有的元素的标量/向量运算，`tt.reduce` 变成线程内循环 + `shfl.sync` + shared memory；`tt.dot` 变成 `mma.sync` 内联 PTX（Hopper 上 `wgmma`）；地址计算、mask 变成谓词 | 手写线程索引、warp shuffle、inline PTX |
| LLVM → PTX → cubin | | LLVM 的标量优化与 NVPTX codegen；`ptxas` 做寄存器分配、SASS 指令调度 | nvcc 的后端，与 CUDA 相同 |

重点是 TTGIR 那一层：**第五、六篇手工做的几乎所有优化——向量化、分块、流水、`ldmatrix`、swizzle、同步——都是 TTGIR 上的 pass**。程序员写的 `tl.load` + `tl.dot` + `for` 只是"意图"，性能来自这些 pass 的质量。这也解释了 Triton 的两个特性：为什么它对"规整"的代码（连续访存、标准 GEMM 循环）效果好——pass 的模式匹配到了；为什么对"不规整"的代码（间接寻址、带 mask 的 K 循环、数据依赖的循环边界）效果差——pass 匹配不上，退回保守的代码。

### 2. 读 TTGIR：layout 的编码

把中间表示 dump 出来有几种方法（第六章有完整代码）：

- 环境变量 `TRITON_KERNEL_DUMP=1`（配合 `TRITON_DUMP_DIR=/path` 指定目录），Triton 会把每个 kernel 的 `.ttir`、`.ttgir`、`.llir`、`.ptx`、`.cubin` 写到目录下（默认在 `~/.triton/dump/`）；
- `kernel[grid](...)` 的返回值是 `CompiledKernel`，它的 `.asm` 字典包含各层：`compiled.asm["ttir"]`、`["ttgir"]`、`["llir"]`、`["ptx"]`、`["cubin"]`；
- `~/.triton/cache/<hash>/` 下也有同样的产物，以及记录编译参数的 JSON。

TTGIR 里的每个张量类型都带一个 layout 属性。以 `add_kernel`（`BLOCK_SIZE=1024`、`num_warps=4`、BF16）为例，会看到形如：

```text
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
...
%x = tt.load %ptrs, %mask : tensor<1024x!tt.ptr<bf16>, #blocked>
```

读法：`sizePerThread = [8]` 每个线程持有连续 8 个元素；`threadsPerWarp = [32]` 一个 warp 的 32 个线程沿这一维排开；`warpsPerCTA = [4]` 4 个 warp 沿这一维排开。于是一个线程持有 8 × 2 B = 16 字节连续数据（一条 128 bit load），一个 warp 覆盖 512 字节连续内存，4 个 warp 覆盖 2 KiB = 1024 个 BF16，恰好是整个 tile。**看到 `sizePerThread` 里有 8（BF16）或 4（FP32），就意味着编译器做了 128 bit 向量化；看到 1，就是没做**——通常是因为它推断不出指针的连续性，需要检查 stride 是否为 constexpr 或加 `tl.multiple_of` 提示。

二维的例子，matmul 里 A 的 `[128, 32]` BF16 tile 在 `num_warps=4` 下可能是：

```text
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
```

`order = [1, 0]` 表示第 1 维（列，K 方向）是最快变化的维；`sizePerThread = [1, 8]` 每线程持有一行中的 8 个连续列（16 字节，一条 128 bit load）；`threadsPerWarp = [4, 8]` 一个 warp 覆盖 4 行 × (8 线程 × 8 元素 = 64 列)——但 tile 只有 32 列，所以 warp 内 8 个线程中实际是 4 个覆盖 32 列，layout 会"回绕"（wrap）复制；`warpsPerCTA = [4, 1]` 4 个 warp 沿行方向排开。这个 layout 只是全局内存 → 寄存器的加载布局，随后会被写入 `#shared` 布局的 shared memory：

```text
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
```

`vec = 8, perPhase = 2, maxPhase = 4` 就是编译器选的 XOR swizzle 参数，作用是让 `ldmatrix` 读 8 × 8 的子块时不发生 bank conflict——第六篇手工用 padding 或 XOR 解决的问题。再从 shared memory 读成 `#dot_op` 布局喂给 `mma`，结果落在：

```text
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
```

`versionMajor = 2` 是 Ampere 的 `mma.sync`（Hopper `wgmma` 是 3），`instrShape = [16, 8]` 是 `m16n8`，`warpsPerCTA = [2, 2]` 4 个 warp 排成 2 × 2 各负责 64 × 64 的子块。epilogue 的 `tl.store` 需要的 layout 与 `#mma` 不同，中间会有一个 `ttg.convert_layout` ——这一次 shared memory 往返就是 epilogue 的固有开销之一。

### 3. 读 PTX：mma、cp.async 与 stage 数

`compiled.asm["ptx"]` 是文本，直接 `grep`：

- `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` 出现 → `tl.dot` 用上了 Tensor Core。数一数它在循环体里出现的次数，应等于 $$(B_M/16)(B_N/8)(B_K/16)/\text{num\_warps}$$；如果看到的是 `fma.rn.f32`，说明 `tl.dot` 的输入 dtype 或形状不满足要求，退化成了 CUDA Core；
- `cp.async.cg.shared.global` / `cp.async.commit_group` / `cp.async.wait_group N` 出现 → 多 stage 流水生效。**`wait_group` 后面的数字就是允许在飞的 group 数，等于 `num_stages - 2`**（`num_stages=3` 时是 `wait_group 1`）——这是从 PTX 反推 stage 数最可靠的办法；
- `ldmatrix.sync.aligned.m8n8.x4.shared.b16` 出现 → 操作数从 shared memory 到 fragment 的搬运用了 `ldmatrix`（带 `.trans` 后缀说明折进了转置）；
- `ld.global.v4.b32` / `st.global.v4.b32` → 128 bit 向量化访存；如果是 `ld.global.b16`，向量化失败；
- `bar.sync 0` 的数量与位置对应 `__syncthreads()`；
- 文件头的 `.maxntid 128, 1, 1` 是 `num_warps × 32`，`.reqntid` 类似；`ptxas` 的信息（寄存器数、spill）可通过 `compiled.n_regs`、`compiled.n_spills` 读到，或加 `--verbose` 让 ptxas 打印。

Hopper 上的对应物：`wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16`、`cp.async.bulk.tensor`（TMA）、`mbarrier` 系列指令。


## 四、生产中的 Triton kernel

### 1. `torch.compile` 生成的 Triton kernel

PyTorch 的 Inductor 后端是目前最大的 Triton 代码生成器。任何 `torch.compile` 的模型，其 pointwise、reduction 融合 kernel 都是 Inductor 生成的 Triton 源码。把它 dump 出来：

```bash
TORCH_LOGS="output_code" python model.py          # 生成的代码打印到 stderr
TORCH_COMPILE_DEBUG=1 python model.py             # 写到 ./torch_compile_debug/run_<时间戳>/torchinductor/model__*/output_code.py
```

一个 `y = (x + b) * 2.0`（BF16，融合成一个 kernel）生成的代码大致长这样（略去部分元数据）：

```python
@triton_heuristics.pointwise(
    size_hints={'x': 1048576},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16',
                               'out_ptr0': '*bf16', 'xnumel': 'i32'},
                 'device': DeviceProperties(type='cuda', index=0, ...),
                 'constants': {}, 'configs': [...]},
    inductor_meta={'kernel_name': 'triton_poi_fused_add_mul_0', ...},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = 2.0
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x0), tmp4, xmask)
```

对照第二章 §1 的手写 `add_kernel`，它的结构完全一样，只是命名规范化了：`xnumel` 是元素总数，`XBLOCK` 是 `BLOCK_SIZE`，`xindex`/`xmask` 是 `offs`/`mask`，`tmp*` 是融合进来的每一个 elementwise op。`@triton_heuristics.pointwise` 是 Inductor 自己的装饰器（在 `torch/_inductor/runtime/triton_heuristics.py`），它包装了 `triton.autotune`：按 `size_hints` 生成若干 `XBLOCK`/`num_warps` 候选，默认只选一个启发式配置，`torch.compile(mode="max-autotune")` 时才真正 benchmark。`_poi_` 表示 pointwise，`_red_` 表示 reduction（会多出 `rnumel`、`RBLOCK`、`tl.sum(..., 1)`），`_per_` 是 persistent reduction。

一个重要的事实：**Inductor 的 matmul 默认不用 Triton**。`torch.mm`/`addmm`/`bmm` 在生成的代码里是 `extern_kernels.mm(...)`，即直接调 cuBLAS。只有 `mode="max-autotune"`（或 `torch._inductor.config.max_autotune_gemm=True`）时，Inductor 才会拿它的 Triton matmul 模板（`torch/_inductor/kernel/mm.py`，与第二章 §3 的写法同源，多了 epilogue 融合）与 cuBLAS 同台 benchmark，选快的那个。这本身就是对本文核心问题的一个工业界回答：**Triton 的 GEMM 在多数形状上不如 cuBLAS，但在能把后续 pointwise 融进 epilogue 的场景下可能反超，所以值得比一比**。

### 2. vLLM 的 `fused_moe_kernel`：grouped GEMM

MoE 层的计算是：每个 token 被路由到 top-k 个 expert，每个 expert 是一个独立的权重矩阵 $$W_e \in \mathbb{R}^{K \times N}$$，token $$t$$ 对 expert $$e$$ 的输出是 $$x_t W_e$$。朴素做法是对每个 expert 挑出它的 token 单独做一次 GEMM——$$E$$ 次 launch，每次的 $$M$$ 很小、很不均匀。vLLM 的做法是一个 kernel 做完所有 expert 的 GEMM（grouped GEMM），关键是**两个索引数组**：

- `sorted_token_ids`：把 $$M \times \text{top\_k}$$ 个 (token, expert) 对按 expert 排序后的 token 索引，并在每个 expert 的段尾填充到 `BLOCK_SIZE_M` 的整数倍（填充位置的值 $$\geq$$ `num_valid_tokens`，用来做 mask）；
- `expert_ids`：长度为 `EM / BLOCK_SIZE_M`，第 $$i$$ 项告诉第 $$i$$ 个 M 方向的 tile 属于哪个 expert。

排序和填充由 `moe_align_block_size`（`vllm/model_executor/layers/fused_moe/moe_align_block_size.py`）在 kernel 之前完成。有了这两个数组，grouped GEMM 就变成了一个普通 GEMM：M 方向的 tile 编号 `pid_m` 通过 `expert_ids[pid_m]` 找到 expert，通过 `sorted_token_ids[pid_m * BLOCK_SIZE_M + arange]` 找到该 tile 的 token 行。核心片段（`vllm/model_executor/layers/fused_moe/fused_moe.py`，v0.20.0，`fused_moe_kernel`，略去量化分支）：

```python
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    if not naive_block_assignment:
        offs_token_id = pid_m * BLOCK_SIZE_M + offs
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    ...
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        write_zeros_to_output(...)
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )
```

与第二章 §3 的 matmul 对照，差别只有三处：

1. **A 的行是间接寻址的**：`offs_token[:, None] // top_k * stride_am`——`sorted_token_ids` 里存的是 (token, expert) 对的展平索引，除以 `top_k` 得到 token 行号，同一个 token 被路由到 top-k 个 expert 就会出现在 top-k 个 tile 里。这是一次 gather，编译器不能再假定行是连续的，但 K 方向（`offs_k`）仍然连续，向量化仍然可用；
2. **B 的 expert 由 `off_experts * stride_be` 选定**：`b_ptr` 加上 expert 偏移后，其余与普通 GEMM 相同。`off_experts == -1` 是 expert 并行时"这个 expert 不在本 rank"的标记，直接写零返回；
3. **无效（填充）token 用 `token_mask` 屏蔽**：K 循环里 `a` 的 mask 是 `token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K)`，形状 `[BLOCK_SIZE_M, BLOCK_SIZE_K]`，填充行读到 0；epilogue 的 `c_mask = token_mask[:, None] & (offs_cn[None, :] < N)` 不写填充行。

K 循环与写回（同文件）：

```python
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        ...
        else:
            accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    ...
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]
    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
```

`MUL_ROUTED_WEIGHT` 是 epilogue 融合：路由权重直接乘在 FP32 累加器上再转 BF16，省掉一次单独的 elementwise kernel。grid 的计算在 `invoke_fused_moe_kernel` 里：`triton.cdiv(EM, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N)`，与普通 matmul 的一维 grid 一致。

这个 kernel 是 Triton 适用范围的一个好例子：它是 GEMM 的一个"不规整"变体，cuBLAS 做不了（每个 tile 的 A 行是 gather 出来的，B 按 expert 切换），手写 CUDA 要几百行而且要为每种量化格式各写一份；Triton 用 constexpr 开关把 FP8/INT8/INT4/BF16 的变体收在一份 600 行的源码里，性能对多数 MoE 形状足够好。它也暴露了 Triton 的局限：gather 出来的 A 行不连续，编译器的 pipeline pass 对带 `token_mask` 的加载生成的流水不如纯 GEMM 紧；对 decode 阶段极小的 $$M$$，tile 利用率低——vLLM 在这些场景下会切到 CUTLASS 或 DeepGEMM 的 grouped GEMM 后端。

### 3. vLLM 的 prefix-prefill attention kernel

`vllm/v1/attention/ops/prefix_prefill.py` 里的 `_fwd_kernel` 是 vLLM 早期（V0 时代）用 Triton 实现的 chunked-prefill attention，V1 里仍作为某些配置的后端保留。它是下一篇 FlashAttention 的一个 Triton 预览。整体结构（v0.20.0，略去 FP8、sliding window、sink 等分支）：

```python
@triton.jit
def _fwd_kernel(Q, K, V, K_cache, V_cache, B_Loc, sm_scale, ..., Out, ...,
                BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
                BLOCK_DMODEL_PADDED: tl.constexpr, BLOCK_N: tl.constexpr, ...):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)
    ...
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q = tl.load(Q + off_q, mask=dim_mask[None, :] & (offs_m[:, None] < cur_batch_query_len),
                other=0.0)                                            # [M, D]
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)        # running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                      # running sum
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_PADDED], dtype=tl.float32)  # [M, D]

    # 第一段循环：query 对已缓存的 context（paged KV cache，无 causal mask）
    for start_n in tl.range(0, cur_batch_ctx_len, BLOCK_SIZE, loop_unroll_factor=num_unroll_cache):
        bn = tl.load(B_Loc + cur_batch * stride_b_loc_b + bn_logical_indices * stride_b_loc_s)
        k = tl.load(K_cache + off_k)                                   # 按 block table 间接寻址
        qk = sm_scale * tl.dot(q, k, input_precision=IN_PRECISION)     # [M, N]
        qk = tl.where((start_n + offs_bs_n[None, :]) < cur_batch_ctx_len, qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        v = tl.load(V_cache + off_v)
        acc = tl.dot(p.to(v.dtype), v, acc=acc, input_precision=IN_PRECISION)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    # 第二段循环：query 对本次新输入的 token（causal），结构相同，K/V 直接来自 Q/K/V 张量
    ...
    acc = acc / (l_i[:, None] + 1e-10)
    tl.store(Out + off_o, acc.to(Out.dtype.element_ty), mask=...)
```

三层 grid `(batch, head, query 方向的 tile)`，每个 program 负责一个 head 的 `BLOCK_M` 行 query，在 K/V 方向上循环，用 `m_i`、`l_i`、`acc` 三个寄存器驻留的张量做 **online softmax**：$$m_{\text{new}} = \max(m, \max_j s_j)$$，$$l_{\text{new}} = l \cdot e^{m - m_{\text{new}}} + \sum_j e^{s_j - m_{\text{new}}}$$，`acc` 同步按 $$\alpha = e^{m - m_{\text{new}}}$$ 缩放。两次 `tl.dot`（QKᵀ 与 PV）都走 Tensor Core，$$S$$ 与 $$P$$ 从不落到全局内存——这就是 FlashAttention 的算法骨架，下一篇会从头推导。这里值得注意的两点：

- **paged KV cache 的间接寻址**：`B_Loc` 是 block table，`bn` 是物理块号，`off_k` 由 `bn` 与块内偏移拼出。与 `fused_moe_kernel` 一样，这是 Triton 能写、cuBLAS/标准库不能写的"不规整"访存；
- **`num_stages=1`**：launch 处显式传了 `num_stages=1`（文件顶部的注释也记录了原因）。attention 的内循环有数据依赖（`m_i`、`l_i` 跨迭代传递）、有间接寻址、有两个 `tl.dot`，Triton 的 pipeline pass 对它的收益有限甚至为负，所以关掉。这与 FlashAttention-2/3 手工设计的流水（K 与 V 的加载交错、softmax 与 `mma` 重叠）形成鲜明对比——那正是第五章要讨论的"10%"。


## 五、编译器的边界：那 10% 在哪里

### 1. Triton 做不了或做不好的

把第一章"编译器决定"的每一项反过来看，就是 Triton 的边界：

**细粒度 warp 级控制**。Triton 没有 `threadIdx`，也没有"warp 0 做加载、warp 1–7 做计算"这种角色划分。FlashAttention-3 的 **warp specialization**（producer warpgroup 用 TMA 搬数据，consumer warpgroup 做 `wgmma` 与 softmax，二者用 `mbarrier` 同步）和 **ping-pong 调度**（两个 consumer warpgroup 交错执行 GEMM 与 softmax，让 Tensor Core 与 SFU 同时忙）在 Triton 里无法用源码表达。Triton 3.x 的编译器在 Hopper 上有自动 warp specialization 的 pass（把加载与计算自动分给不同 warp），但它是编译器的决定，不是程序员的，且能处理的循环模式有限。

**任意 PTX 指令**。`tl.inline_asm_elementwise` 允许嵌一段 PTX，但限于**逐元素**的运算（输入输出都是同形状张量，每个线程对自己持有的元素执行）：可以用它做 FP8 转换、特殊数学函数、`prmt` 字节重排，不能用它发 `mma`、`cp.async`、`ldmatrix`、`mbarrier` 这些涉及 warp 协作或 shared memory 的指令。

**跨 block 通信与持久 kernel 的手工调度**。Triton 有 `tl.atomic_add/cas/xchg` 等原子操作，可以做 split-K 的归约或简单的 tile 计数器，但没有 cooperative groups 的 grid 同步、没有 Hopper 的 thread block cluster 与 distributed shared memory（DSMEM）。持久 kernel 可以写（`for tile in range(pid, num_tiles, num_programs)`），但 stream-K 那种"把 K 维切成不等长的段分给不同 block 再归约"的精细负载均衡，写起来笨重且性能不稳。

**寄存器级布局控制**。累加器在 `#mma` 布局上、写回要 `#blocked` 布局，中间的 `convert_layout` 是一次 shared memory 往返。手写 CUTLASS 可以让 epilogue 直接在 `mma` 输出的 fragment 上做 pointwise 并用 `stmatrix` 写回，避开这次转换；也可以把 $$P$$ 矩阵从 QKᵀ 的累加器布局直接"重解释"为下一个 `mma` 的 A 操作数布局（FlashAttention-2 的一个关键技巧，因为 `m16n8k16` 的 C 布局和 A 布局恰好兼容）。Triton 的编译器有时能识别这个兼容性并省掉转换，有时不能，程序员无法强制。

**Hopper 特性的支持随版本演进**。`tl.dot` 在 sm_90 上自动选 `wgmma`；TMA 有 `tl.make_tensor_descriptor` / `tl.load_tensor_descriptor`（早期版本是 `_experimental_descriptor_load`）；`num_stages` 控制 TMA 流水；warp specialization 有自动 pass。但 FlashAttention-3 级别的手工调度——TMA multicast 到 cluster、`mbarrier` 的精细 arrive/wait 编排、`wgmma` 的异步 commit/wait 与 softmax 的手工重叠、FP8 的寄存器内转置——做不到。Triton 3.x 版本间 Hopper 后端的变化很大，写文档时的 API 与读者手上的版本可能不同，用前查所用版本的 `triton.language` 文档。

**非 2 的幂 block**。`tl.arange` 与所有 tile 形状必须是 2 的幂。对 $$N = 4096$$ 无所谓，对 head_dim = 96 或 hidden = 3072 × 1.5 这样的形状要 pad 到 128 或 8192 并 mask，浪费的算力和寄存器在 20–33% 之间。`prefix_prefill.py` 里 `BLOCK_DMODEL_PADDED` 与 `dim_mask` 就是在处理这个。

**pipeline pass 的适用范围**。带 mask 的加载、数据依赖的循环边界、间接寻址、循环内的 `if`，都可能让编译器放弃流水或生成保守的流水。`fused_moe_kernel` 的 `token_mask` 和 `prefix_prefill` 的 `num_stages=1` 都是这个问题的实例。

### 2. 那 10% 在哪里

把边界与 GEMM/attention 的性能构成对上，"通常差 5–20%"的来源可以列成四项：

**流水与 warp specialization 的精细控制**。cuBLAS/CUTLASS 在 Hopper 上用 TMA + producer/consumer warp specialization + `wgmma` 异步流水，能把 Tensor Core 利用率推到 80% 以上；Triton 自动生成的流水在同一形状上通常低 5–10 个百分点。Ampere 上差距小些，因为 `cp.async` + `mma.sync` 的流水模式更规整、编译器更容易匹配。

**epilogue 融合与布局转换的开销**。`#mma → #blocked` 的 `convert_layout` 是一次 shared memory 写 + 读 + 两次 barrier，对 $$128 \times 128$$ 的 FP32 累加器是 64 KiB 的流量。对 4096³ 这种 K 很长的 GEMM 这是 $$1/128$$ 量级的开销可以忽略；对 $$K = 128$$ 的小 GEMM 或 attention 里每个 K/V tile 都要做一次的 $$P$$ 矩阵转换，就成了主要开销。

**小 shape 的 tile 选择**。decode 阶段 $$M = 16$$ 的 GEMM，$$B_M = 64$$ 的 tile 有 3/4 是浪费；cuBLAS 有为小 M 特制的 split-K、stream-K 与非方形 tile 的 kernel 库，Triton 只有 autotune 列表里的那几个 config。`tl.dot` 要求 $$M \geq 16$$，$$M = 1$$ 的 GEMV 在 Triton 里只能 pad 到 16 或改用 `tl.sum(a[:, None] * b, axis=0)` 的 CUDA Core 路径。

**指令级调度**。`ptxas` 对 Triton 生成的 PTX 与对 CUTLASS 生成的 PTX 做同样的调度，但 CUTLASS 的源码结构（显式的 fragment 循环、手工的 `ldmatrix` 与 `mma` 交错）给了 `ptxas` 更好的起点；Triton 从 layout 展开出来的代码有时会有多余的地址计算与谓词处理，占用发射槽位。

这四项加起来，就是"4096³ 达 cuBLAS 的 90–95%、小形状 70–85%、Hopper 上差距略大于 Ampere"这个经验区间的来源。

### 3. 什么时候值得手写

反过来就是答案：

- **生产中的热点 GEMM 或 attention**：如果一个 kernel 占端到端时间的 30% 以上、形状固定、要在成千上万张卡上跑几个月，10% 就是 3% 的总成本，值得几周的 CUTLASS/CuTe 工程。FlashAttention、DeepGEMM、vLLM 的 CUTLASS MoE 后端都是这个逻辑；
- **需要特殊指令**：FP8 的 per-block scaling 要在 `wgmma` 的累加器上做精细的 scale（DeepGEMM 的核心技巧）、用 `stmatrix` 做 epilogue、用 `cp.reduce.async.bulk` 做 TMA 归约、用 `mbarrier` 做跨 warp 的生产者-消费者——这些指令 Triton 没有暴露；
- **需要压榨 Hopper**：warp specialization、TMA multicast、cluster、ping-pong 调度是 H100 上从 60% 到 75%+ Tensor Core 利用率的手段，只能手写；
- **非 2 的幂形状且无法 pad**、**需要跨 block 协作的算法**（比如某些 all-reduce 融合）。

其余情况——elementwise、normalization、softmax、非热点的 GEMM 变体（grouped、带奇怪 epilogue 的）、需要快速迭代的研究性 kernel、要同时支持 NVIDIA 与 AMD 的 kernel——Triton 是更好的选择：30–60 行代码、性能通常在手写的 90% 以上、不用管 fragment 布局。**判断标准不是"Triton 能不能写"，而是"这 10% 值多少钱、需要什么指令"**。


## 六、实践：正确性、性能表与中间表示

### 1. 正确性对照

三个 kernel 都与 PyTorch 参考用 `torch.testing.assert_close` 比对。BF16 默认容差 `rtol=1.6e-2, atol=1e-5`。

```python
import torch
import triton
import triton.language as tl

# 假定 add_kernel / triton_add、softmax_kernel / triton_softmax、
# matmul_kernel / triton_matmul 已按第二章定义

torch.manual_seed(0)
dev = "cuda"

# (1) elementwise add
n = 2 ** 28
x = torch.randn(n, device=dev, dtype=torch.bfloat16)
y = torch.randn(n, device=dev, dtype=torch.bfloat16)
torch.testing.assert_close(triton_add(x, y), x + y)          # 逐位相等，默认容差足够

# (2) row softmax：N 取非 2 的幂以覆盖 mask 路径
M, N = 8192, 4000
xs = torch.randn(M, N, device=dev, dtype=torch.bfloat16)
ref = torch.softmax(xs.float(), dim=-1).to(torch.bfloat16)
torch.testing.assert_close(triton_softmax(xs), ref)

# (3) matmul：M/N/K 各取非 tile 整数倍以覆盖 K mask 与 % M 回卷
Mm, Km, Nn = 4096 + 8, 4096 + 24, 4096 + 16
a = torch.randn(Mm, Km, device=dev, dtype=torch.bfloat16)
b = torch.randn(Km, Nn, device=dev, dtype=torch.bfloat16)
torch.testing.assert_close(triton_matmul(a, b), a @ b, rtol=1.6e-2, atol=1e-2)
print("all close")
```

matmul 一项把 `atol` 从 1e-5 放宽到 1e-2：两边都是 FP32 累加、最后转 BF16，但累加顺序不同（cuBLAS 的 K 分块、split-K 与 Triton 的 `BLOCK_K` 顺序不同），在 $$K = 4120$$ 次累加后可能有最后一位 BF16 的差异；对 $$\mathcal{N}(0, 1)$$ 的输入，输出量级约 $$\sqrt{K} \approx 64$$，一个 BF16 ulp 约 0.25–0.5，`rtol=1.6e-2` 能覆盖，`atol` 放宽是为接近 0 的输出留余量。

### 2. 性能对照表模板

用 `triton.testing.do_bench` 计时，它默认在每次迭代前刷 L2（写一个 256 MB 的缓冲区）、返回若干次运行的中位数毫秒，与第二篇定义的 `bench(fn, flush_l2=True)` 语义一致，Python 侧直接用它。CUDA 版通过第二篇的 `load_inline` 扩展接进来（此处记为 `cuda_add`、`cuda_softmax`、`cuda_matmul`）。

```python
from triton.testing import do_bench

def gbps(bytes_moved, ms):
    return bytes_moved / ms / 1e6          # GB/s

def tflops(flops, ms):
    return flops / ms / 1e9                # TFLOPS

rows = []

# elementwise：6 B/元素
ms_t = do_bench(lambda: triton_add(x, y))
ms_c = do_bench(lambda: cuda_add(x, y))
ms_p = do_bench(lambda: x + y)
rows.append(("bf16 add, n=2^28", 6 * n / 2.0e12 * 1e3, ms_c, ms_t, ms_p))

# softmax：读 + 写各 M*N*2 B
xs = torch.randn(8192, 4096, device=dev, dtype=torch.bfloat16)
by = 2 * xs.numel() * 2
ms_t = do_bench(lambda: triton_softmax(xs))
ms_c = do_bench(lambda: cuda_softmax(xs))
ms_p = do_bench(lambda: torch.softmax(xs, dim=-1))
rows.append(("bf16 softmax, 8192x4096", by / 2.0e12 * 1e3, ms_c, ms_t, ms_p))

# matmul：2 M N K FLOP
a = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
b = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
fl = 2 * 4096 ** 3
ms_t = do_bench(lambda: triton_matmul(a, b))
ms_c = do_bench(lambda: cuda_matmul(a, b))
ms_p = do_bench(lambda: a @ b)              # cuBLAS
rows.append(("bf16 matmul, 4096^3", fl / 312e12 * 1e3, ms_c, ms_t, ms_p))

print(f"{'kernel':28s} {'theory ms':>10s} {'CUDA ms':>9s} {'Triton ms':>10s} {'PyTorch ms':>11s}")
for name, th, c, t, p in rows:
    print(f"{name:28s} {th:10.3f} {c:9.3f} {t:10.3f} {p:11.3f}")
```

读者在 A100 上跑出的数字大致应落在这个区间（理论列是可推导的下界，其余三列是经验区间，不是实测）：

```text
kernel                    理论下界     CUDA（第3–6篇）   Triton           PyTorch/cuBLAS
bf16 add, n=2^28          0.81 ms     0.88–0.95 ms     0.88–0.95 ms     0.88–0.95 ms
bf16 softmax, 8192×4096   0.067 ms    0.075–0.085 ms   0.075–0.085 ms   0.075–0.085 ms
bf16 matmul, 4096³        0.44 ms     0.65–0.80 ms     0.58–0.70 ms     0.55–0.63 ms
```

前两行三者相当——memory-bound kernel 到了带宽上限就没有区别。第三行 Triton 落在 cuBLAS 的 85–95%，手写的 `mma.sync` 版本（第六篇）反而可能不如 Triton——除非投入 CUTLASS 级别的工程量，否则编译器自动做的流水与 swizzle 比多数人手写的更好。

### 3. 打印 TTGIR 与 PTX

```python
import os
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"     # 打印 autotune 选中的 config（需在 import triton 前设置）

# kernel[grid](...) 的返回值是 CompiledKernel（autotune 包装后同样返回）
compiled = matmul_kernel[
    lambda meta: (triton.cdiv(4096, meta["BLOCK_M"]) * triton.cdiv(4096, meta["BLOCK_N"]),)
](a, b, torch.empty(4096, 4096, device=dev, dtype=torch.bfloat16),
  4096, 4096, 4096, a.stride(0), a.stride(1), b.stride(0), b.stride(1), 4096, 1)

print("stages of IR available:", list(compiled.asm.keys()))    # ['ttir', 'ttgir', 'llir', 'ptx', 'cubin']
print("registers/thread:", compiled.n_regs, " spills:", compiled.n_spills)

ttgir = compiled.asm["ttgir"]
for line in ttgir.splitlines():
    if line.startswith("#"):                # layout 定义都在文件头：#blocked / #shared / #mma
        print(line)

ptx = compiled.asm["ptx"]
def count(s):
    return sum(1 for l in ptx.splitlines() if s in l)
print("mma.sync      :", count("mma.sync"))
print("wgmma         :", count("wgmma"))
print("cp.async      :", count("cp.async.cg") + count("cp.async.ca"))
print("cp.async.wait :", [l.strip() for l in ptx.splitlines() if "cp.async.wait_group" in l][:3])
print("ldmatrix      :", count("ldmatrix"))
print("ld.global.v4  :", count("ld.global.v4"))
print("bar.sync      :", count("bar.sync"))
```

或者不改代码，用环境变量把所有产物落盘：

```bash
TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=./triton_dump python bench.py
ls triton_dump/*/            # matmul_kernel.ttir  .ttgir  .llir  .ptx  .cubin  .json
grep -c "mma.sync" triton_dump/*/matmul_kernel.ptx
grep "cp.async.wait_group" triton_dump/*/matmul_kernel.ptx | sort | uniq -c
```

期望看到的：`#blocked` 里 `sizePerThread` 含 8（BF16 的 128 bit 向量化）；`#shared` 带 swizzle 参数；`#mma`（或 `#nvidia_mma`）`versionMajor = 2`（Ampere）；PTX 里 `mma.sync.m16n8k16` 与 `cp.async.cg.shared.global` 大量出现，`cp.async.wait_group` 的数字为 `num_stages - 2`；`n_spills` 为 0。如果 `sizePerThread` 是 1 或 `mma.sync` 计数为 0，回到第三章 §3 逐项排查。


## 七、小结

这一篇把前六篇手工做的事交给了编译器，然后打开编译器看它做了什么、没做什么。

**编程模型**：Triton 去掉了线程，程序员以 program（= CUDA block）为单位写块级张量代码；`tl.program_id`、`tl.arange`（2 的幂）、`tl.load/tl.store` + mask、`tl.dot` 是核心原语；线程到元素的映射（layout）、shared memory、同步全部由编译器决定；`tl.constexpr` 对应模板参数，`num_warps` 与 `num_stages` 是仅有的两个硬件旋钮；`@triton.autotune` 按 key 缓存搜索结果，首次运行有搜索代价。

**三个 kernel**：elementwise 12 行、softmax 25 行、matmul 60 行，分别对应 CUDA 的 50–80、100–150、200–300 行；memory-bound 的前两个性能与手写相当，matmul 通常达 cuBLAS 的 80–95%。

**编译器做了什么**：TTGIR 层的 coalescing、pipeline、prefetch、layout 转换消除、swizzle、`mma` 选择——正是第三到六篇手工做的向量化、多 stage `cp.async`、`ldmatrix`、bank conflict 消除与 Tensor Core 编程。读 TTGIR 的 `#blocked`/`#shared`/`#mma` 与 PTX 的 `mma.sync`/`cp.async.wait_group`，能直接看到这些决定。

**那 10%**：流水与 warp specialization 的精细控制、epilogue 的布局转换、小 shape 的 tile 选择、指令级调度。值得手写的场景：生产热点 GEMM/attention、需要特殊指令、需要压榨 Hopper、跨 block 协作。

```text
CUDA 与 Triton 概念对照
  CUDA                              Triton
  blockIdx.x / y / z                tl.program_id(0 / 1 / 2)
  threadIdx.x                       无——编译器决定 layout
  blockDim.x                        num_warps × 32（launch 参数，默认 4 warp）
  template <int BLOCK>              BLOCK: tl.constexpr（每组取值一个 cubin，缓存于 ~/.triton/cache）
  if (idx < n)                      mask=（tl.load / tl.store 的逐元素谓词）
  float4 / uint4 向量化             编译器由连续性分析自动做（tl.multiple_of / tl.max_contiguous 提示）
  __shared__ + __syncthreads()      无显式声明——layout 转换、归约、tl.dot 操作数自动分配 + 自动 barrier
  warp shuffle + block reduce       tl.sum / tl.max(axis=)
  mma.sync / wgmma + ldmatrix       tl.dot(a, b, acc)
  多 stage cp.async 流水            num_stages（Ampere 默认 3；PTX 中 cp.async.wait_group 的数字 = num_stages − 2）
  kernel<<<grid, block>>>(...)      kernel[grid](..., num_warps=, num_stages=)，grid 可为 lambda meta: (...)
  nvcc 离线编译                     @triton.jit 首次调用 JIT，按 constexpr 与参数特化属性缓存

编译流水线
  层            关键内容                                   主要优化
  Python AST    JITFunction / CodeGenerator                constexpr 代入、编译期分支删除
  TTIR          tt.load / tt.dot / tt.reduce / scf.for     硬件无关：CSE、LICM、DCE
  TTGIR         #blocked / #shared / #mma / #dot_op        coalesce、pipeline、prefetch、layout 转换消除、swizzle、barrier
  LLVM IR       线程级标量/向量代码                         layout 展开、shfl、mma 内联 PTX
  PTX           mma.sync / cp.async / ldmatrix / ld.v4     NVPTX codegen
  cubin         SASS                                       ptxas 寄存器分配与调度

Triton 的边界
  做不了：指定某个 warp 做什么（warp specialization 手工编排）、mma/cp.async/mbarrier 级 inline PTX、
          cluster / DSMEM / grid 同步、寄存器级布局重解释、非 2 的幂 tile
  做不好：带 mask / 间接寻址 / 数据依赖边界的循环流水、小 M 的 tile 利用、epilogue 布局转换、
          Hopper 的 TMA multicast 与 ping-pong 调度（随 3.x 版本演进，FA-3 级别做不到）
  性能区间（A100，经验值）：elementwise / softmax 与手写 CUDA 相当；matmul 为 cuBLAS 的 80–95%（大形状高、小形状低）

本篇推导用到的数字
  bf16 add, n=2^28：1.61 GB / 2.0 TB/s ≈ 0.81 ms
  bf16 softmax 8192×4096：134 MB / 2.0 TB/s ≈ 67 µs
  bf16 matmul 4096³：137.4 GFLOP / 312 TFLOPS ≈ 0.44 ms；cuBLAS 通常 0.55–0.63 ms
  128×128×32 BF16 tile：每 stage 16 KiB shared；256 条 m16n8k16 mma / tile / K-step
  GROUP_SIZE_M swizzle：一波 108 program 的工作集，行主序 ≈ 36 MiB，G=8 分组 ≈ 22 MiB（A100 L2 40 MB）
```

下一篇进入 attention：FlashAttention 为什么把 $$O(N^2)$$ 的 HBM 流量降到 $$O(N^2 d^2 / M)$$，FlashAttention-2 与 3 在 warp 分工和 Hopper 特性上做了什么，PagedAttention 的 block table 如何改变 decode 的访存模式——以及第四章 §3 的 Triton 版为什么会在这些地方输给手写版本。


## 下一篇

[Attention Kernel：FlashAttention 与 PagedAttention](/attention-kernels-flashattention-and-pagedattention.html)
