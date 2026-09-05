---
layout: post
title: "GPU Kernel 工程（03）：访存合并与 elementwise kernel"
subtitle: "Memory Coalescing and Elementwise Kernels: Hitting the Bandwidth Ceiling"
tags: [CUDA, Triton, GPU, AI, AI-Infra]
catalog: true
---

> 本文是[《GPU Kernel 工程：从 CUDA 执行模型到 FlashAttention》](/gpu-kernel-engineering.html)系列的第 3 篇（共十篇）。上一篇：[CUDA 编程模型与第一个 kernel](/cuda-programming-model-and-first-kernel.html)　下一篇：[共享内存与 reduction](/shared-memory-reduction-and-softmax.html)

上一篇写出了第一个 kernel：一个 BF16 的 `y = x + b`，每个线程处理一个元素。它能跑、结果正确，但没有回答"它跑得够快吗"。这一篇就回答这个问题，并把答案推到极限。

先把理论下界放在最前面，后面所有讨论都对着它算。BF16 的 `y = x + b` 对每个元素读 2 个 BF16、写 1 个 BF16，共 6 字节，做 1 次加法（1 FLOP）：

$$
\text{算术强度} = \frac{1\ \text{FLOP}}{6\ \text{B}} \approx 0.17\ \text{FLOP/B}
$$

A100 SXM 80GB 的 HBM2e 带宽约 2.0 TB/s，BF16 Tensor Core 约 312 TFLOPS（均为公开标称值），ridge point 是 $$312 / 2.0 \approx 156$$ FLOP/B。0.17 与 156 差了三个数量级：这个 kernel 无论怎么写都是 memory-bound，它的时间下界只由字节数决定：

$$
t_{\min} = \frac{\text{字节数}}{\text{带宽}} = \frac{3 \cdot n \cdot 2\ \text{B}}{2.0\ \text{TB/s}}
$$

$$n = 2^{28}$$（每个 tensor 512 MiB，三个共 1.5 GiB ≈ 1.61 GB）时，$$t_{\min} \approx 0.81$$ ms。任何实现都不可能比这个数快；工程的全部目标，就是逼近它。

AI 负载里除 GEMM 与 attention 之外的绝大多数算子——激活函数、残差加、dtype 转换、RoPE、掩码、dropout、量化/反量化——都属于这一类。把它们写到带宽极限是最基本的功课，也是理解后面所有 kernel 的起点：GEMM 和 attention 的 tile 加载，本质上仍是这一篇讨论的访存模式。

总纲给这一篇的核心问题是：

> **一个 elementwise kernel 跑出了 90% 带宽，还有什么可优化的？**

答案是"没有了，要么融合，要么少做"。这条边界很重要——它决定了工程师应该把时间花在哪里。本文先把"如何到 90%"讲清楚，再解释为什么 90% 是一条墙。


## 一、一个 warp 的内存请求发生了什么

### 1. 32 字节 sector 与 128 字节 cache line

GPU 不是按线程访问内存的，而是按 warp。一个 warp 的 32 个线程同时执行一条加载指令时，硬件收集 32 个地址，按它们落在哪些**内存事务**里进行**合并**（coalescing），然后向 L1/L2 发出请求。

合并的粒度有两级：

- **sector**：32 字节，是 L2 与 HBM 之间、以及 L1 与 L2 之间传输的最小单位；
- **cache line**：128 字节，由 4 个连续的 sector 组成，是 L1 的行大小。

一次 warp 级加载最终被拆成若干个 sector 请求。**决定效率的不是线程数，而是这 32 个地址一共触碰了多少个 sector**。有效字节数（warp 真正需要的）除以实际搬运的字节数（sector 数 × 32 B），就是访存效率。

举最简单的例子：32 个线程每人读一个 `float`（4 字节），地址连续且起点按 128 字节对齐。总共需要 128 字节，恰好落在 4 个 sector 里，效率 100%。这是所有 elementwise kernel 应该追求的形态：

```text
线程   0   1   2   3  ...  31
地址   0   4   8  12  ... 124        (相对 128 B 对齐的起点)
sector |---- 0 ----|---- 1 ----|---- 2 ----|---- 3 ----|
       每个 sector 32 B，全部有效  →  4 sector，128 B，效率 100%
```

### 2. 访问模式与效率表

把访问模式改一改，sector 数会迅速膨胀。以下每种情况都是 32 个线程各读 4 字节（有效字节固定为 128 B）：

- **跨步 2**（线程 i 读地址 $$8i$$）：地址范围 0–252，跨 8 个 sector，搬运 256 B，只用一半，效率 50%。
- **跨步 8**（线程 i 读地址 $$32i$$）：每个线程独占一个 sector，32 个 sector、1024 B，效率 $$1/8 = 12.5\%$$。
- **跨步 ≥ 32 字节**：不管跨多大，每线程至少一个 sector，效率不会比 1/8 更好；再大只会让 L2/TLB 行为更差。
- **未对齐起始**（起点偏移 4 B）：128 B 数据从 sector 0 的第 4 字节开始，尾巴落入第 5 个 sector，搬运 160 B，效率 80%。
- **随机地址**：最坏 32 个 sector，效率 12.5%，且没有任何局部性可利用。

汇总成表：

```text
访问模式（32 线程 × 4 B，有效 128 B）        sector 数   搬运字节   效率
连续、128 B 对齐                             4          128 B     100%
连续、起点偏移 4 B（未对齐）                  5          160 B      80%
跨步 2（每线程间隔 8 B）                      8          256 B      50%
跨步 4（间隔 16 B）                          16          512 B      25%
跨步 8 及以上（间隔 ≥ 32 B）                 32         1024 B     12.5%
随机                                         ≤32        ≤1024 B    ≥12.5%
```

BF16 是 2 字节，32 个线程连续读只有 64 B，占 2 个 sector——效率仍是 100%，但每条加载指令只带回 64 B，要靠更多指令才能填满带宽。这一点是下一章向量化的动机。

需要注意：这张表算的是 L1/L2 层面的**请求效率**，HBM 侧因为 L2 会缓存被"浪费"的 sector，跨步访问在某些情况下的实际 DRAM 流量比表中略好；但 L2 命中率依赖数据能否留在 40 MB（A100）的 L2 里，对 GB 级的 tensor 不能指望。工程上直接按这张表估算即可。

### 3. AoS 与 SoA

跨步访问最常见的来源不是显式的 `stride`，而是数据结构。考虑存 $$n$$ 个三维点：

```cpp
struct Point { float x, y, z; };   // Array of Structures (AoS)
Point* pts;                         // 线程 i 读 pts[i].x
```

线程 i 读 `pts[i].x` 时地址间隔 12 B，32 个线程跨 384 B、12 个 sector，有效 128 B，效率 33%。如果一个 kernel 只需要 x 分量，另两个分量是纯浪费。改成 **SoA**（Structure of Arrays）：

```cpp
struct Points { float* x; float* y; float* z; };   // 三个独立数组
```

`x[i]` 连续，效率 100%。深度学习框架里的 tensor 天然是 SoA：一个 tensor 一块连续内存，dtype 单一。这也是为什么 tensor 抽象对 GPU 友好——当你在 kernel 里定义结构体数组时，要意识到你在往 AoS 的方向走。

一个反例是刻意的 AoS：把 RoPE 的 cos/sin 交错存放成 `(cos, sin)` 对，这样一个线程一次 `float2` 加载就同时拿到两者，反而比两个数组各读一次少一条指令。规则不是"永远 SoA"，而是"让一个线程一次访问的字节在内存里连续，让相邻线程访问的字节也连续"。

### 4. 写入与只读路径

以上讨论的是加载；存储的合并规则相同——warp 的 32 个写地址按 sector 合并，部分写入的 sector 需要 L2 做字节掩码合并，效率同样按触碰的 sector 数计算。区别在于写不需要等待返回，warp 发出存储指令后可以立即继续，因此写延迟对 kernel 的影响远小于读延迟；但写流量在 HBM 侧同样占带宽，2 读 1 写的 `add` 中写占了三分之一。

对只读数据，从 Volta 起 L1 与纹理缓存已经合并为同一块存储，用 `const T* __restrict__` 修饰的指针，编译器会生成 `ld.global.nc`（non-coherent，等价于 `__ldg`）加载，允许数据在 L1 中缓存且不必与其他 SM 的写入保持一致性。对 elementwise kernel 来说数据只读一次、L1 命中率为零，这条路径的收益不在缓存，而在于编译器得到"没有别名"的保证后可以自由重排加载与存储——把三条加载都提前发出、再统一计算和写回，这正是第三章要讨论的 ILP。


## 二、向量化访存

### 1. 为什么每线程 4 字节不够

上一节的结论是连续访问效率 100%，看起来问题已经解决。但 A100 的 2.0 TB/s 是一个很高的速率：每个 SM 每秒要吞掉 $$2.0 \times 10^{12} / 108 \approx 18.5$$ GB/s，按 1.41 GHz 折算约 13 字节/周期。一条 32 线程 × 4 B 的加载指令带回 128 B，也就是每个 SM 每 10 个周期就得发出一条加载指令并让它命中——这还没算地址计算、边界判断、类型转换和存储指令。

对 BF16 更糟：每线程 2 B，一条指令只带回 64 B。指令发射能力（每个 SM 4 个 warp 调度器，每周期各发一条）在 memory-bound kernel 里通常够用，但 LSU（load/store unit）的请求队列和 L1 的每周期事务数是有限资源。经验上，**每线程只搬 2–4 字节的 kernel 很难超过 70–80% 的带宽**。

解法是让每条加载指令搬更多字节：CUDA 支持 8、16 字节的向量化加载（`ld.global.v2.b32`、`ld.global.v4.b32`），一个线程一次拿 16 字节，一个 warp 一条指令 512 B、4 条完整的 cache line。指令数减到 1/4（相对 `float`）或 1/8（相对 BF16），每请求字节数增大，LSU 压力同比下降。

### 2. `float4`、`__nv_bfloat162` 与 `int4`

CUDA 内建向量类型中，`float4`、`int4`、`uint4`、`double2` 都是 16 字节且按 16 字节对齐；`float2`、`__nv_bfloat162`、`half2` 是 4 或 8 字节。对 BF16 数据，一个 16 字节的加载对应 8 个元素，常用的做法是**用 `int4` 搬运，用 `__nv_bfloat162` 计算**：

```cpp
#include <cuda_bf16.h>

// 对 16 字节（8 个 BF16）做 x + b，累加用 float
__device__ __forceinline__ int4 add_bf16x8(int4 xa, int4 ba) {
  const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xa);
  const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&ba);
  int4 ya;
  __nv_bfloat162* y2 = reinterpret_cast<__nv_bfloat162*>(&ya);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    float2 xf = __bfloat1622float2(x2[k]);
    float2 bf = __bfloat1622float2(b2[k]);
    y2[k] = __floats2bfloat162_rn(xf.x + bf.x, xf.y + bf.y);
  }
  return ya;
}
```

`__bfloat1622float2` 把一对 BF16 转成 `float2`，`__floats2bfloat162_rn` 反向舍入打包。这里的 `reinterpret_cast` 作用在寄存器里的局部变量上，编译器会把它优化成纯寄存器操作，不产生额外访存。

为什么不用 `__hadd2` 直接在 BF16 上加？BF16 只有 8 位尾数，直接相加的舍入误差比先转 float 再舍入更大，而转换和 FP32 加法在 memory-bound kernel 里是免费的——本文所有 kernel 都遵循"BF16 存储、float 计算"的约定，与 ATen 的 `opmath_type` 做法一致。

### 3. 对齐要求、`reinterpret_cast` 与尾部处理

向量化加载有一个硬性要求：**地址必须按向量宽度对齐**。用 `int4` 读一个不是 16 字节倍数的地址，会触发 misaligned address 错误并让 kernel 崩溃。这带来三个工程细节。

第一，**起始地址检查**。`cudaMalloc` 返回的指针至少 256 字节对齐，PyTorch 的 caching allocator 也保证 512 字节对齐，所以完整 tensor 的 `data_ptr()` 天然满足条件；但 `x[:, 1:]` 这样的切片、`storage_offset` 非零的 view 就不一定。host 侧要检查：

```cpp
bool aligned16 = (reinterpret_cast<uintptr_t>(ptr) % 16) == 0;
```

不满足时退回标量路径。ATen 的 `can_vectorize_up_to` 做的就是这件事，第五章会读它。

第二，**`reinterpret_cast` 的语义**。`reinterpret_cast<const int4*>(x)[i]` 表示"把 `x` 看成 `int4` 数组，取第 i 个"，也就是从 `x` 起第 $$16i$$ 字节处读 16 字节。它要求 `x` 本身 16 字节对齐，而不只是 `x + 16i`。这一点很容易被"我只在 i 为 8 的倍数处访问"的直觉误导。

第三，**尾部**。$$n$$ 不是 8 的倍数时，最后 $$n \bmod 8$$ 个元素不能用 `int4` 读（会越界读取，甚至越界写入）。常见做法是：主循环只处理前 $$\lfloor n / 8 \rfloor \times 8$$ 个元素，剩余的由某个 block 的前几个线程用标量方式补齐。第七章的代码会给出完整写法。

`float4` 对 FP32 是 4 个元素/线程；`__nv_bfloat162` × 4 是 8 个元素/线程；INT8 用 `int4` 是 16 个元素/线程。**每线程处理 4–8 个元素**是 elementwise kernel 最常见的配置，ATen 的默认也在这个范围。


## 三、grid-stride loop 与占用率

### 1. grid 不必等于元素数

上一篇的 kernel 用 `grid = ceil(n / blockDim)` 让每个线程恰好处理一个元素。这没有错，硬件的 block 调度器会依次把 block 派发到 SM 上，几十万个 block 也能跑完。但还有另一种写法：

```cpp
for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
     i += (int64_t)gridDim.x * blockDim.x) {
  // 处理第 i 个元素（或第 i 个向量）
}
```

这就是 **grid-stride loop**：grid 大小与 $$n$$ 解耦，每个线程处理 $$\lceil n / (\text{grid} \times \text{block}) \rceil$$ 个元素，步长是整个 grid 的线程总数。相邻线程仍然访问相邻地址，合并性质不变。

grid 该多大？一个常见选择是**每 SM 可驻留 block 数 × SM 数**，再乘一个小倍数（2–4）：

$$
\text{grid} = k \cdot N_{SM} \cdot \text{blocks\_per\_SM}, \quad k \in [1, 4]
$$

`blocks_per_SM` 可以用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 查询。这样恰好填满 GPU 一到几轮，每个线程循环多次。

好处有三：

- **任意 $$n$$**：$$n$$ 超过 $$2^{31}$$ 个 block 的上限、或极小时都能用同一个 kernel，不需要在 host 侧算 grid；
- **复用**：每个线程处理多个元素时，`blockIdx`/`threadIdx` 的地址计算、边界检查等固定开销被摊薄，且一个线程连续发出多个独立加载（不同迭代之间没有依赖），为下一节的延迟隐藏提供 ILP（指令级并行）；
- **占用率可控**：grid 大小成为一个显式参数，可以调；某些 kernel（如需要跨 block 归约的）也依赖"所有 block 同时驻留"这个性质。

grid-stride 不是免费的：循环控制和 64 位地址算术要占几条指令，尾部迭代会有一部分线程空转。它在 elementwise kernel 上的收益通常是几个百分点，不是决定性的；决定性的是向量化。ATen 的 elementwise kernel 就**没有**用 grid-stride，而是"每 block 处理固定 1024 个元素、grid = ceil(N / 1024)"，第五章会看到。两种写法都能到 90%。

### 2. 占用率与 Little's law：memory-bound 为什么也需要多 warp

一个常见误解是"memory-bound 的 kernel 不需要高占用率，反正瓶颈在带宽"。恰恰相反：**要把带宽用满，必须有足够多的字节在飞**。

HBM 的访问延迟约 500–800 ns（A100 上典型值取 ~600 ns）。带宽 2.0 TB/s。Little's law 说，稳态下系统中的在飞请求量 = 吞吐 × 延迟：

$$
\text{在飞字节数} = 2.0\ \text{TB/s} \times 600\ \text{ns} \approx 1.2\ \text{MB}
$$

分到 108 个 SM 上，每个 SM 要保持约 11 KB 的加载请求在路上。一个 warp 一条 128-bit 加载（32 线程 × 16 B）是 512 B，所以每个 SM 至少要有

$$
\frac{11\ \text{KB}}{512\ \text{B}} \approx 22
$$

个 warp 各有一条 128-bit 加载在飞，才能把带宽填满。如果每线程只用 32-bit 加载（128 B/warp），需要约 88 个 warp——超过了每 SM 64 个 warp 的硬件上限，**单凭占用率填不满带宽**。

有两条路：一是提高占用率，让更多 warp 驻留；二是让每个线程连续发出多个独立加载（例如 grid-stride 循环展开两次、或每线程处理两个 `int4`），一个 warp 就有 2–4 条加载在飞，需要的 warp 数按比例减少。后一条路就是 ILP，它对寄存器的要求更高（每个在飞的加载都要一个目标寄存器），但对占用率的要求更低。真实 kernel 通常两者兼用：每线程 2 个 `int4`、每 SM 32 个以上的 warp。

### 3. 占用率的约束

每 SM 最多 2048 个线程（64 个 warp）、32 个 block、65536 个 32-bit 寄存器、164 KB shared memory（A100）。elementwise kernel 不用 shared memory，占用率只受寄存器和 block 数限制。一个线程用 $$R$$ 个寄存器、block 有 $$T$$ 个线程时：

$$
\text{blocks\_per\_SM} = \min\left(\left\lfloor \frac{65536}{R \cdot T} \right\rfloor,\ \left\lfloor \frac{2048}{T} \right\rfloor,\ 32\right)
$$

（寄存器分配按 warp 级、以 256 个为单位向上取整，实际值略小。）向量化 add kernel 每线程用 20–30 个寄存器，$$T = 256$$ 时 $$65536 / (32 \times 256) = 8$$ 个 block、2048 线程，占用率 100%。elementwise kernel 几乎不会遇到寄存器压力；用 `__launch_bounds__(256)` 告诉编译器 block 大小，可以防止它为了 ILP 过度分配寄存器。

block 大小本身对 elementwise kernel 影响不大，128 到 512 都常见。太小（如 32 或 64）会撞上每 SM 最多 32 个 block 的限制——32 个 block × 64 线程 = 2048 线程刚好够，但 32 × 32 = 1024 线程只有一半占用率；太大（1024）则一个 block 占满整个 SM，block 之间切换时的空档无法被填补，且尾部 block 的浪费更多。ATen 取 128、本文取 256，都是让每 SM 驻留 8–16 个 block 的选择，粒度足够细，调度器有余地。

还有一个与占用率无关但常被忽视的因素：**每 SM 的 L1/LSU 事务数上限**。一条 warp 级加载指令覆盖 4 条 cache line 时，L1 需要 4 个周期（每周期处理一条 128 B 的 line）才能把它处理完——这不是坏事，恰恰说明 128-bit 加载让 LSU 的每条指令都在做满载的工作；反过来，2 字节的标量加载一条指令只占半条 line，L1 每周期能处理的有效字节数只有向量化时的 1/8。这就是第二章"naive 很难超过 70–80%"的微架构解释。

至此，把 elementwise kernel 写到带宽极限的三件事已经齐了：**合并（连续对齐）、向量化（16 B/线程）、足够的在飞请求（占用率 + ILP）**。第七章把它们落成代码，先解决另一个绕不开的问题——tensor 不连续怎么办。


## 四、非连续 Tensor：stride 与 broadcast

### 1. 把线性 index 变成多维 offset

上面的 kernel 都假设 `x`、`b`、`y` 是同形状、连续的一维数组。实际调用 `x + b` 时，`x` 可能是转置后的 view（stride 不连续），`b` 可能是形状 `[1, d]` 的 bias 要广播到 `[m, d]`。kernel 需要 stride 信息。

一个 tensor 的元素 $$(i_0, i_1, \ldots, i_{k-1})$$ 在存储中的位置是：

$$
\text{offset} = \sum_{d} i_d \cdot \text{stride}_d
$$

kernel 里每个线程拿到的是一个线性 index $$i \in [0, n)$$，需要先按输出形状拆成多维坐标（从最内维开始连续做除法与取模），再用每个输入各自的 stride 算出各自的 offset：

```cpp
// 二维情形：输出形状 [size0, size1]，行主序
int64_t i0 = i / size1;
int64_t i1 = i - i0 * size1;
int64_t off_x = i0 * xs0 + i1 * xs1;
int64_t off_b = i0 * bs0 + i1 * bs1;
```

**broadcast 就是 stride 为 0**：`b` 的形状 `[1, d]` 扩展到 `[m, d]`，第 0 维的 stride 设为 0，所有行读同一段内存。不需要物化任何数据。

两个性能提示。第一，只要最内维 stride 为 1，相邻线程仍访问相邻地址，合并不受影响；但如果最内维 stride 不是 1（如转置后的 `x.t()`，最内维 stride 是 4096 个元素 = 8 KiB），每个线程独占一个 sector，BF16 的效率只剩 $$2 / 32 = 6.25\%$$。这种情况下应该先 `contiguous()`（一次 transpose kernel 的代价远小于低效访问），或者让 kernel 用 shared memory 做 tile 转置——那是第四篇的话题。第二，64 位整数除法在 GPU 上很慢（几十条指令），ATen 的做法是把除数预处理成"魔数 + 移位"（`IntDivider`），用乘法代替除法，且用 32 位 index。

### 2. TensorIterator 在 host 侧做了什么

PyTorch 的 elementwise 算子并不直接把 sizes/strides 传给 kernel，而是先经过 `TensorIterator`。它在 host 侧完成：形状广播、dtype 推断与类型提升、把可以合并的维度合并（例如 `[m, d]` 两维连续就当作一维 `[m·d]`）、按 stride 重排维度让最内维是访问最密的、判断所有操作数是否连续并检查 32 位索引是否够用——然后把一个"已经整理好的迭代空间"交给 CUDA 端。这样 kernel 只需要处理"连续一维"和"带 OffsetCalculator 的一般情况"两种形态。本文不展开它，只需要知道下一章读到的 `iter.is_contiguous()`、`iter.strides(i)` 这些信息就来自这里。


## 五、读 ATen 的 elementwise 实现

有了以上概念，读 PyTorch 的实现就很直接了。源码版本为 v2.10.0，路径在 `aten/src/ATen/native/cuda/` 下，主要是三个头文件：`Loops.cuh`（入口 `gpu_kernel`）、`CUDALoops.cuh`（kernel 与 launch）、`MemoryAccess.cuh`（向量化加载与 policy）。

### 1. 入口与路径选择

算子实现（如 `ActivationSiluKernel.cu`）用 `gpu_kernel(iter, lambda)` 描述"对每个元素做什么"，其余全部交给框架。`gpu_kernel` 经过 32 位索引检查后到 `gpu_kernel_impl_nocast`（`CUDALoops.cuh`）：

```cpp
// aten/src/ATen/native/cuda/CUDALoops.cuh（节选）
template <typename func_t>
void gpu_kernel_impl_nocast(TensorIteratorBase& iter, const func_t& f) {
  using traits = function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  constexpr int ntensors = traits::arity + 1;
  // ...
  std::array<char*, ntensors> data;
  for (int i = 0; i < ntensors; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }
  int64_t numel = iter.numel();
  bool contiguous = iter.is_contiguous();

  if (contiguous) {
    return launch_vectorized_kernel(numel, f, data);
  }
  auto offset_calc = ::make_offset_calculator<traits::arity + 1>(iter);
#ifndef USE_ROCM
  constexpr int unroll_factor = sizeof(arg0_t) >= 4 ? 2 : 4;
  launch_legacy_kernel<128, unroll_factor>(numel, [=] GPU_LAMBDA(int idx) {
    auto offsets = offset_calc.get(idx);
    arg0_t* out = (arg0_t*)(data[0] + offsets[0]);
    *out = invoke(f, &data[1], &offsets[1], 1);
  });
#endif
}
```

路径选择只有一个分支：**全部操作数连续 → 向量化路径；否则 → 带 `OffsetCalculator` 的通用路径**。`data` 是一个 `char*` 数组，`data[0]` 是输出、其后是输入；lambda `f` 的参数类型（通过 `function_traits` 提取）决定了每个操作数的元素类型。

### 2. `launch_vectorized_kernel` 与 `can_vectorize_up_to`

连续路径先决定向量宽度（CUDA 分支，去掉 ROCm 部分）：

```cpp
// aten/src/ATen/native/cuda/CUDALoops.cuh（节选，CUDA 分支）
template <typename func_t, typename array_t>
static inline void launch_vectorized_kernel(int64_t N, const func_t& f, array_t data) {
  using traits = function_traits<func_t>;
  constexpr auto io_size = calc_io_size<func_t>();
  auto stream = at::cuda::getCurrentCUDAStream();
  using cpp_type = typename function_traits<func_t>::result_type;
  const uint16_t max_vec_size = memory::can_vectorize_up_to<func_t>(data);
  uint16_t vec_size = 16 / static_cast<uint16_t>(sizeof(cpp_type));
  vec_size = std::min<uint16_t>(vec_size, max_vec_size);
  cudaDeviceProp* p = at::cuda::getDeviceProperties(stream.device().index());
  const int computeCapability = p->major * 10 + p->minor;
  if (computeCapability != 90 && computeCapability != 100) {
    vec_size = std::min<uint16_t>(vec_size, 4);
  }
  if constexpr (sizeof(cpp_type) < 2) {
    vec_size = std::min<uint16_t>(vec_size, 4);
  }
  int tws = elems_per_thread<io_size>();
  int bws = tws * num_threads();
  int64_t grid = (N + bws - 1) / bws;
  switch (vec_size) {
    case 8: vectorized_elementwise_kernel<8, func_t, array_t>
                <<<grid, num_threads(), 0, stream>>>(N, f, data); break;
    case 4: vectorized_elementwise_kernel<4, func_t, array_t>
                <<<grid, num_threads(), 0, stream>>>(N, f, data); break;
    case 2: vectorized_elementwise_kernel<2, func_t, array_t>
                <<<grid, num_threads(), 0, stream>>>(N, f, data); break;
    case 1: { /* 退回 unrolled_elementwise_kernel，见下节 */ }
  }
}
```

逐行对应前面的概念：

- `vec_size = 16 / sizeof(cpp_type)`：目标是 16 字节一次加载。BF16 得 8，FP32 得 4。
- `can_vectorize_up_to<func_t>(data)`：对输出和每个输入指针检查对齐，取最小值（`MemoryAccess.cuh`）：

```cpp
// aten/src/ATen/native/cuda/MemoryAccess.cuh（节选，CUDA 分支）
template<typename scalar_t>
inline C10_HOST_DEVICE int can_vectorize_up_to(const char *pointer) {
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  constexpr int vec2_alignment = std::alignment_of_v<aligned_vector<scalar_t, 2>>;
  constexpr int vec4_alignment = std::alignment_of_v<aligned_vector<scalar_t, 4>>;
  constexpr int vec8_alignment = std::alignment_of_v<aligned_vector<scalar_t, 8>>;
  if (address % vec8_alignment == 0) {
   return 8;
  } else if (address % vec4_alignment == 0) {
    return 4;
  } else if (address % vec2_alignment == 0) {
    return 2;
  }
  return 1;
}
```

  `aligned_vector<scalar_t, N>` 是一个 `alignas(sizeof(scalar_t) * N)` 的结构体（`scalar_t val[N]`），它的对齐要求就是向量的字节宽度。这正是第二章说的"host 侧对齐检查"。
- `computeCapability != 90 && != 100` 时把 `vec_size` 压到 4：在 2.10 中，8 元素向量只在 Hopper/Blackwell 上启用——所以 **A100 上 BF16 的 `add` 实际用的是 4 × 2 B = 8 字节（64-bit）加载**，H100 上是 16 字节。源码注释说明这是为了规避一个 NVCC 数值问题，并控制二进制体积（vec8 实例只为 sm_90/sm_100 编译）。
- `elems_per_thread<io_size>()`：`io_size` 是所有输入与输出元素大小之和（BF16 二元 op 为 6）；它等于 1 时每线程 16 个元素，否则 8 个。`num_threads()` 是 128（`thread_constants.h` 中定义为 `C10_WARP_SIZE * 4`）。所以一个 block 处理 $$128 \times 8 = 1024$$ 个元素，grid = $$\lceil N / 1024 \rceil$$——**不是 grid-stride**。

### 3. `vectorized_elementwise_kernel` 的结构

```cpp
// aten/src/ATen/native/cuda/CUDALoops.cuh（节选，vec_size != 8 分支）
template <int vec_size, typename func_t, typename array_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void vectorized_elementwise_kernel(int N, func_t f, array_t data) {
  using traits = function_traits<func_t>;
  constexpr auto io_size = calc_io_size<func_t>();
  int remaining = N - io_block_work_size<io_size>() * blockIdx.x;

  if (remaining < io_block_work_size<io_size>()) {
    // 最后一个不满的 block：退回带边界检查的 unroll policy
    auto input_calc = TrivialOffsetCalculator<traits::arity>();
    auto output_calc = TrivialOffsetCalculator<1>();
    auto loader = memory::LoadWithoutCast();
    auto storer = memory::StoreWithoutCast();
    auto policy = memory::policies::unroll<array_t, decltype(input_calc),
        decltype(output_calc), memory::LoadWithoutCast,
        memory::StoreWithoutCast, elems_per_thread<io_size>()>(
        data, remaining, input_calc, output_calc, loader, storer);
    elementwise_kernel_helper(f, policy);
  } else {
    // 满 block：向量化加载，不做任何边界检查
    elementwise_kernel_helper(
        f, memory::policies::vectorized<vec_size, array_t,
                                        elems_per_thread<io_size>()>(data));
  }
}
```

结构非常清楚：**尾部处理是按 block 而不是按元素做的**——只有最后一个 block 走带边界检查的慢路径，其余 block 完全不做 `if (i < n)`。`elementwise_kernel_helper`（`Loops.cuh`）是通用骨架：`policy.load(args, blockIdx.x)` 把本 block 的 1024 个元素装进每线程 8 个寄存器槽，循环调用 `f`，再 `policy.store`。

`vectorized` policy 的 `load_single_arg` 是向量化加载的核心（`MemoryAccess.cuh`）：

```cpp
// aten/src/ATen/native/cuda/MemoryAccess.cuh（节选）
template<typename accessor_t, typename scalar_t>
__device__ inline void load_single_arg(accessor_t to, scalar_t *from) {
  int thread_idx = threadIdx.x;
  #pragma unroll
  for (int i = 0; i < loop_size; i++) {          // loop_size = elems_per_thread / vec_size
    int index = thread_idx + i * num_threads();
    auto v = load_vector<vec_size>(from, index);  // 一次 aligned_vector 加载
    #pragma unroll
    for (int j = 0; j < vec_size; j++) {
      to(vec_size * i + j) = v.val[j];
    }
  }
}
```

注意 `index = thread_idx + i * num_threads()`：第 i 次迭代时，线程 t 读第 $$t + 128 i$$ 个向量——相邻线程读相邻向量，每次迭代 warp 覆盖连续的 $$32 \times \text{vec\_size} \times \text{sizeof}$$ 字节。BF16、vec 4 时一个 warp 一条指令 256 B、2 条 cache line；每线程 2 次迭代（8 元素 / 4）。这是第一章"连续对齐"与第二章"向量化"在源码里的直接体现。

### 4. 非向量化路径：`unrolled_elementwise_kernel` 与 `elementwise_kernel`

`vec_size == 1`（指针对齐不够）时走：

```cpp
// aten/src/ATen/native/cuda/CUDALoops.cuh（节选）
template <typename func_t, typename array_t, int elems_per_thread,
          typename inp_calc_t, typename out_calc_t,
          typename loader_t, typename storer_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel(int N, func_t f, array_t data,
    inp_calc_t ic, out_calc_t oc, loader_t l, storer_t s) {
  int remaining = N - elems_per_thread * num_threads() * blockIdx.x;
  auto policy = memory::policies::unroll<array_t, inp_calc_t, out_calc_t,
      loader_t, storer_t, elems_per_thread>(data, remaining, ic, oc, l, s);
  elementwise_kernel_helper(f, policy);
}
```

这里 `elems_per_thread` 传入 `elementwise_thread_work_size()` = 4（源码注释说 8 在 CUDA 上反而退化），block 处理 512 个元素。`unroll` policy 对每个元素做标量加载、用 `remaining` 做边界检查，并通过 `inp_calc_t` 计算 offset——连续时是 `TrivialOffsetCalculator`（offset = index），非连续时是 `OffsetCalculator<N>`。

而非连续 tensor 的通用路径（第 1 节 `launch_legacy_kernel<128, unroll_factor>`）用的是更朴素的 `elementwise_kernel`：

```cpp
// aten/src/ATen/native/cuda/CUDALoops.cuh（节选）
template <int nt, int vt, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel(int N, func_t f) {
  int tid = threadIdx.x;
  int nv = nt * vt;
  int idx = nv * blockIdx.x + tid;
#pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      f(idx);
      idx += nt;
    }
  }
}
```

`nt = 128` 线程、每线程 `vt` 个元素（4 字节以上类型取 2，更小的取 4）。`idx += nt` 保持相邻线程访问相邻 index，展开 `vt` 次给出 ILP。传入的 lambda 用 `OffsetCalculator::get(idx)` 把线性 index 拆成各操作数的字节 offset——它内部就是第四章描述的"逐维 divmod × stride"，只是用 `IntDivider` 的魔数除法代替了真除法，最多支持 25 维（`MAX_DIMS`）。

三个 kernel 对比一下：

```text
kernel                          block  每线程元素   加载方式       边界检查        适用
vectorized_elementwise_kernel   128    8 (io=1:16)  aligned_vector  只在最后 block  连续且对齐
unrolled_elementwise_kernel     128    4            标量            每元素          连续但未对齐
elementwise_kernel (legacy)     128    2 或 4       标量+OffsetCalc 每元素          非连续/广播
```

### 5. `AT_DISPATCH`：运行期 dtype 到编译期类型

`gpu_kernel` 的 lambda 是模板化的——`scalar_t` 必须在编译期确定。而 `iter.dtype()` 是运行期的枚举。桥梁是 `AT_DISPATCH_*` 宏族（`aten/src/ATen/Dispatch.h`）。以 SiLU 的反向为例（`ActivationSiluKernel.cu`）：

```cpp
AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half, at::ScalarType::BFloat16,
    iter.dtype(), "silu_backward_cuda", [&]() {
      gpu_kernel(iter, [] GPU_LAMBDA(scalar_t dy, scalar_t x) -> scalar_t {
        using opmath_t = at::opmath_type<scalar_t>;
        const opmath_t dy_acc = static_cast<opmath_t>(dy);
        const opmath_t x_acc = static_cast<opmath_t>(x);
        const opmath_t s_acc =
            opmath_t(1) / (opmath_t(1) + c10::cuda::compat::exp(-x_acc));
        return dy_acc * s_acc * (opmath_t(1) + x_acc * (opmath_t(1) - s_acc));
      });
    });
```

宏展开的骨架（`Dispatch.h` 与 `torch/headeronly/core/Dispatch.h`）：

```cpp
#define AT_DISPATCH_FLOATING_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME,                                                   \
      AT_DISPATCH_CASE_FLOATING_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, __VA_ARGS__))

#define AT_DISPATCH_CASE_FLOATING_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, ...) \
  AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__)                     \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)                      \
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                                \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)

// 每个 case 展开为（THO_PRIVATE_CASE_TYPE_USING_HINT_TMPL）：
//   case enum_type: {
//     using scalar_t = ScalarTypeToCPPTypeT<enum_type>;
//     return __VA_ARGS__();
//   }
// AT_DISPATCH_SWITCH 展开为：
//   [&] { switch (::detail::scalar_type(TYPE)) { <cases> default: TORCH_CHECK_NOT_IMPLEMENTED(...) } }()
```

所以这段代码最终是一个 `switch (dtype)`，每个 `case` 里 `using scalar_t = double / float / c10::Half / c10::BFloat16;`，然后调用传入的 lambda。lambda 体在每个 `case` 里被实例化一次，`gpu_kernel` 里的 `GPU_LAMBDA` 随之实例化出 4 份不同 `scalar_t` 的 `vectorized_elementwise_kernel`——**运行期的一个枚举值，选中编译期已经生成好的一份模板实例**。代价是二进制体积：每多支持一种 dtype、每多一种 vec_size，就多一份 kernel。这也是为什么 `launch_vectorized_kernel` 要费心限制 vec8 只为 sm_90/sm_100 生成。

`opmath_type<scalar_t>` 对 Half/BFloat16 给出 `float`，对 float/double 给出自身——这正是"低精度存储、float 计算"约定的框架级实现。


## 六、融合：90% 之后

### 1. 三个 kernel 与一个 kernel

回到核心问题。假设 `add` 已经跑到 90% 带宽，要做的运算是一个典型的 MLP 尾部：

$$
\text{out} = \text{silu}(x + b) \odot y
$$

分成三个 elementwise 算子执行：

```text
kernel 1:  t1 = x + b        读 x, b        写 t1
kernel 2:  t2 = silu(t1)     读 t1          写 t2
kernel 3:  out = t2 * y      读 t2, y       写 out
```

每个元素共 5 次读、3 次写，BF16 下 $$8 \times 2 = 16$$ 字节。三个 kernel 各自都可以做到 90% 带宽，但**总字节数是 16 B/元素**。

融合成一个 kernel：读 x、b、y，算完写 out。3 次读、1 次写，$$4 \times 2 = 8$$ 字节。总时间减半，而三个中间量 `t1`、`t2` 根本不需要存在。

```text
                         读     写     字节/元素(BF16)   相对时间
三个独立 kernel          5      3      16               1.0
融合 kernel              3      1       8               0.5
```

如果中间 tensor 小于 L2（A100 40 MB），`t1` 的写和随后的读可能命中 L2，不走 HBM；但 LLM 激活值随便就是几十到几百 MB，不能指望。此外每个 kernel 还有 launch 开销（几微秒）和启动/收尾阶段带宽利用不足的时间，小 tensor 时这部分比例更高。

### 2. 这就是 Inductor 融合的收益来源

`torch.compile` 的 Inductor 后端对 pointwise 算子做的最主要优化，就是把这一串融合成一个 Triton kernel。它的收益不来自任何单个算子"更快"——`add` 的 Triton 版本和 ATen 版本都是 90% 带宽——而来自**字节数减少**。用 Roofline 的语言：memory-bound 区域里，kernel 已经贴在带宽斜线上，往上走的唯一办法是把点向右移（提高算术强度），融合就是把多个 1 FLOP/6 B 的点合并成一个 3 FLOP/8 B 的点。

从这个角度重新审视 elementwise 优化的边界：

- 单个 elementwise kernel 的上限是带宽，可优化空间就是从 60% 到 90% 这段，手写与框架实现差距通常不超过 1.5 倍；
- 一旦到了 90%，**唯一有意义的优化就是让它消失**：融合进邻居（Inductor、手写 fused kernel），或融合进 GEMM 的 epilogue（CUTLASS 的 epilogue fusion，第六篇）、attention 的输出阶段（第八篇）；
- "少做"还包括：不物化 mask 而在 kernel 里按坐标判断、不物化 broadcast、用 in-place 减少一次分配、把 dtype 转换合并进相邻算子。

关于 in-place 需要澄清一点：`x.add_(b)` 与 `y = x + b` 的字节数是一样的（都是 2 读 1 写），in-place 省的是显存分配与 caching allocator 的开销，不是带宽。真正减少字节的是"不要把中间结果写出去再读回来"。类似地，`x.to(torch.float32)` 后再做一次 FP32 的 elementwise，比直接在 kernel 里"读 BF16、float 计算、写 BF16"多了一读一写 FP32（8 B/元素），是常见的隐性浪费——本文所有 kernel 内部都用 float 计算，正是为了不需要这一步。

融合并非没有代价。融合后的 kernel 寄存器更多、模板实例更多，Inductor 需要为每一种算子组合生成并编译一个新 kernel；对手写 kernel 而言，每个融合模式都是一份要维护、要测试的代码。所以融合的优先级应该由 profile 决定：先看时间线上哪些 elementwise kernel 相邻且合计占比高，再决定融合哪一段。第十篇会回到这个方法论。

这条边界也是本系列后面篇章的组织逻辑：reduction、GEMM、attention 之所以值得单独写 kernel，是因为它们的上限不再是"读一遍写一遍"，而有更多结构可以利用。


## 七、实践：把 BF16 add 推到 90%

### 1. 理论下界

统一测试规模 $$n = 2^{28}$$（每个 tensor 512 MiB）：

$$
\text{字节数} = 3 \times 2^{28} \times 2\ \text{B} = 1.5\ \text{GiB} \approx 1.61\ \text{GB},\qquad t_{\min} = \frac{1.61\ \text{GB}}{2.0\ \text{TB/s}} \approx 0.81\ \text{ms}
$$

任何一个版本的实测时间除以这个数，就是"达到了理论带宽的百分之几"。选这么大的规模是为了让 L2（40 MB）的影响可以忽略，同时让 launch 开销（微秒级）小于 1%。

### 2. 版本一：naive，每线程一个元素

```cpp
#include <cuda_bf16.h>
#include <cstdint>

__global__ void add_bf16_naive(const __nv_bfloat16* __restrict__ x,
                               const __nv_bfloat16* __restrict__ b,
                               __nv_bfloat16* __restrict__ y,
                               int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float xv = __bfloat162float(x[i]);
    float bv = __bfloat162float(b[i]);
    y[i] = __float2bfloat16(xv + bv);
  }
}
// launch: block = 256, grid = (n + 255) / 256
```

访存是合并的（相邻线程相邻地址，2 B × 32 = 64 B/warp，2 个 sector 全部有效），但每条加载只带回 64 B，每个线程 3 条访存指令只搬 6 字节。`__restrict__` 告诉编译器三个指针不重叠，允许它把加载提前、并对只读数据用 `ld.global.nc`（只读缓存路径）。

### 3. 版本二：向量化，每线程 8 个元素

```cpp
__device__ __forceinline__ int4 add_bf16x8(int4 xa, int4 ba) {
  const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xa);
  const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&ba);
  int4 ya;
  __nv_bfloat162* y2 = reinterpret_cast<__nv_bfloat162*>(&ya);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    float2 xf = __bfloat1622float2(x2[k]);
    float2 bf = __bfloat1622float2(b2[k]);
    y2[k] = __floats2bfloat162_rn(xf.x + bf.x, xf.y + bf.y);
  }
  return ya;
}

// 要求 x、b、y 均 16 字节对齐（host 侧检查）
__global__ void add_bf16_vec8(const __nv_bfloat16* __restrict__ x,
                              const __nv_bfloat16* __restrict__ b,
                              __nv_bfloat16* __restrict__ y,
                              int64_t n) {
  const int64_t n_vec = n / 8;            // 完整的 16 字节块数
  const int4* x4 = reinterpret_cast<const int4*>(x);
  const int4* b4 = reinterpret_cast<const int4*>(b);
  int4* y4 = reinterpret_cast<int4*>(y);

  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_vec) {
    y4[i] = add_bf16x8(x4[i], b4[i]);
  }
  // 尾部 n % 8 个元素：由 block 0 的前 (n % 8) 个线程标量处理
  const int tail = (int)(n & 7);
  if (blockIdx.x == 0 && (int)threadIdx.x < tail) {
    int64_t j = n_vec * 8 + threadIdx.x;
    y[j] = __float2bfloat16(__bfloat162float(x[j]) + __bfloat162float(b[j]));
  }
}
// launch: block = 256, grid = max(1, (n_vec + 255) / 256)
```

检查几个容易出错的地方：`n_vec * 8 + tail == n`，尾部索引 `j` 严格小于 `n`；`tail < 8 ≤ 256`，block 0 一定有足够的线程；`n < 8` 时 `n_vec = 0`，主体一个线程都不进，只有尾部起作用，grid 用 `max(1, …)` 保证至少有 block 0。三个数组各一次 `int4` 访问，一个 warp 一条指令 512 B、4 条完整 cache line。

### 4. 版本三：向量化 + grid-stride

```cpp
__global__ void __launch_bounds__(256)
add_bf16_vec8_gs(const __nv_bfloat16* __restrict__ x,
                 const __nv_bfloat16* __restrict__ b,
                 __nv_bfloat16* __restrict__ y,
                 int64_t n) {
  const int64_t n_vec = n / 8;
  const int4* x4 = reinterpret_cast<const int4*>(x);
  const int4* b4 = reinterpret_cast<const int4*>(b);
  int4* y4 = reinterpret_cast<int4*>(y);

  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       i < n_vec; i += stride) {
    y4[i] = add_bf16x8(x4[i], b4[i]);
  }
  const int tail = (int)(n & 7);
  if (blockIdx.x == 0 && (int)threadIdx.x < tail) {
    int64_t j = n_vec * 8 + threadIdx.x;
    y[j] = __float2bfloat16(__bfloat162float(x[j]) + __bfloat162float(b[j]));
  }
}
// launch: block = 256, grid = num_SMs * blocks_per_SM * 2（下面的 host 代码计算）
```

与版本二的差别只有循环。想再多要一点 ILP，可以把循环体展开两次：先发出两组 `int4` 加载再做两组计算与存储，让每个线程同时有 4 条加载在飞（2 个输入 × 2 组）；编译器在 `#pragma unroll 2` 下通常会自动做这个调度。

### 5. 通用版本：2D stride 与 broadcast

输出 `y` 为连续的 `[size0, size1]`，输入 `x`、`b` 可以是任意 stride（含 0，即 broadcast）：

```cpp
__global__ void __launch_bounds__(256)
add_bf16_strided2d(const __nv_bfloat16* __restrict__ x,
                   const __nv_bfloat16* __restrict__ b,
                   __nv_bfloat16* __restrict__ y,
                   int64_t size0, int64_t size1,
                   int64_t xs0, int64_t xs1,     // x 的 stride（元素单位）
                   int64_t bs0, int64_t bs1) {   // b 的 stride，broadcast 维为 0
  const int64_t n = size0 * size1;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += stride) {
    const int64_t i0 = i / size1;
    const int64_t i1 = i - i0 * size1;
    const float xv = __bfloat162float(x[i0 * xs0 + i1 * xs1]);
    const float bv = __bfloat162float(b[i0 * bs0 + i1 * bs1]);
    y[i] = __float2bfloat16(xv + bv);
  }
}
```

这个版本没有向量化：一般 stride 下无法保证 8 个相邻元素在内存里连续。它的性能取决于 `xs1`、`bs1`：等于 1 时访存合并，能接近版本一；`x` 是转置 view（`xs1 = size0`）时效率掉到 6.25%，慢一个数量级——这时正确的做法是先 `contiguous()`。`i / size1` 是 64 位除法，每次约几十条指令，在 memory-bound 下通常被访存延迟掩盖；追求极致时改用 ATen 的 `IntDivider` 思路预计算魔数。ATen 的 `TensorIterator` 还会把 `size1` 连续的情况直接合并为一维、走向量化路径，我们这里为了展示 stride 机制没有做。

### 6. host 侧 launch 与 `load_inline` 测试

把四个 kernel 放进一个 CUDA 源码字符串，用 `torch.utils.cpp_extension.load_inline` 编译成 Python 可调用的函数：

```python
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <algorithm>

// ---- 此处粘贴第 2–5 节的四个 kernel 与 add_bf16x8 ----

static inline bool aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) % 16) == 0;
}
static inline const __nv_bfloat16* bf(const at::Tensor& t) {
  return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr<at::BFloat16>());
}
static inline __nv_bfloat16* bf_mut(at::Tensor& t) {
  return reinterpret_cast<__nv_bfloat16*>(t.data_ptr<at::BFloat16>());
}

at::Tensor add_naive(at::Tensor x, at::Tensor b) {
  TORCH_CHECK(x.is_contiguous() && b.is_contiguous() && x.sizes() == b.sizes());
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16);
  auto y = at::empty_like(x);
  const int64_t n = x.numel();
  if (n == 0) return y;
  const int block = 256;
  const int64_t grid = (n + block - 1) / block;
  add_bf16_naive<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      bf(x), bf(b), bf_mut(y), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor add_vec8(at::Tensor x, at::Tensor b) {
  TORCH_CHECK(x.is_contiguous() && b.is_contiguous() && x.sizes() == b.sizes());
  auto y = at::empty_like(x);
  TORCH_CHECK(aligned16(x.data_ptr()) && aligned16(b.data_ptr()) && aligned16(y.data_ptr()),
              "add_vec8 requires 16-byte aligned pointers");
  const int64_t n = x.numel();
  if (n == 0) return y;
  const int block = 256;
  const int64_t grid = std::max<int64_t>(1, (n / 8 + block - 1) / block);
  add_bf16_vec8<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      bf(x), bf(b), bf_mut(y), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor add_vec8_gs(at::Tensor x, at::Tensor b) {
  TORCH_CHECK(x.is_contiguous() && b.is_contiguous() && x.sizes() == b.sizes());
  auto y = at::empty_like(x);
  TORCH_CHECK(aligned16(x.data_ptr()) && aligned16(b.data_ptr()) && aligned16(y.data_ptr()));
  const int64_t n = x.numel();
  if (n == 0) return y;
  const int block = 256;
  int num_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  int blocks_per_sm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, add_bf16_vec8_gs, block, 0);
  int64_t grid = (int64_t)num_sms * std::max(blocks_per_sm, 1) * 2;
  grid = std::max<int64_t>(1, std::min<int64_t>(grid, (n / 8 + block - 1) / block));
  add_bf16_vec8_gs<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      bf(x), bf(b), bf_mut(y), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// x: [size0, size1] 任意 stride；b: 可 broadcast 到同形状（用 expand 得到 stride 0）
at::Tensor add_strided2d(at::Tensor x, at::Tensor b) {
  TORCH_CHECK(x.dim() == 2 && b.dim() == 2 && x.sizes() == b.sizes());
  auto y = at::empty(x.sizes(), x.options());   // 连续输出
  const int64_t n = x.numel();
  if (n == 0) return y;
  const int block = 256;
  int num_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  int64_t grid = std::max<int64_t>(1, std::min<int64_t>((int64_t)num_sms * 16,
                                                        (n + block - 1) / block));
  add_bf16_strided2d<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      bf(x), bf(b), bf_mut(y), x.size(0), x.size(1),
      x.stride(0), x.stride(1), b.stride(0), b.stride(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
"""

cpp_src = """
at::Tensor add_naive(at::Tensor x, at::Tensor b);
at::Tensor add_vec8(at::Tensor x, at::Tensor b);
at::Tensor add_vec8_gs(at::Tensor x, at::Tensor b);
at::Tensor add_strided2d(at::Tensor x, at::Tensor b);
"""

ext = load_inline(
    name="ew_add_bf16",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["add_naive", "add_vec8", "add_vec8_gs", "add_strided2d"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)
```

`data_ptr<at::BFloat16>()` 同时做了 dtype 检查；`at::BFloat16` 与 `__nv_bfloat16` 都是 2 字节、位模式一致，`reinterpret_cast` 是安全的。`load_inline` 默认按当前 GPU 的架构编译（可用环境变量 `TORCH_CUDA_ARCH_LIST=8.0` 固定）。

计时脚手架与第二篇一致：`bench(fn, warmup=10, iters=100, flush_l2=True)`，`cudaEvent` 计时，两次计时之间用一个 128 MB（≥ 2 × A100 L2）的缓冲区 `memset` 冲掉 L2，取中位数毫秒：

```python
_flush_buf = None

def bench(fn, warmup=10, iters=100, flush_l2=True):
    global _flush_buf
    if flush_l2 and _flush_buf is None:
        _flush_buf = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        if flush_l2:
            _flush_buf.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


HBM_TBPS = 2.0   # A100 标称值；H100 改为 3.35

def report(name, ms, nbytes):
    tbps = nbytes / (ms * 1e-3) / 1e12
    print(f"{name:14s} {ms:8.3f} ms  {tbps:5.2f} TB/s  {100 * tbps / HBM_TBPS:5.1f}% of peak")


n = 1 << 28
x = torch.randn(n, device="cuda").to(torch.bfloat16)
b = torch.randn(n, device="cuda").to(torch.bfloat16)
ref = x + b
nbytes = 3 * n * 2

for name, fn in [("torch.add", lambda: x + b),
                 ("naive", lambda: ext.add_naive(x, b)),
                 ("vec8", lambda: ext.add_vec8(x, b)),
                 ("vec8+gs", lambda: ext.add_vec8_gs(x, b))]:
    torch.testing.assert_close(fn(), ref)          # BF16 默认 rtol=1.6e-2, atol=1e-5
    report(name, bench(fn), nbytes)

# 尾部与小规模的正确性
for m in [1, 7, 8, 9, 1000, 4097]:
    xs, bs = x[:m].clone(), b[:m].clone()
    torch.testing.assert_close(ext.add_vec8_gs(xs, bs), xs + bs)

# stride 与 broadcast
x2 = torch.randn(4096, 4096, device="cuda").to(torch.bfloat16)
bias = torch.randn(1, 4096, device="cuda").to(torch.bfloat16).expand(4096, 4096)  # stride (0, 1)
torch.testing.assert_close(ext.add_strided2d(x2, bias), x2 + bias)
xt = x2.t()                                                                        # stride (1, 4096)
torch.testing.assert_close(ext.add_strided2d(xt, bias), xt + bias)
report("strided (x)", bench(lambda: ext.add_strided2d(x2, bias)), 3 * x2.numel() * 2)
report("strided (x.t)", bench(lambda: ext.add_strided2d(xt, bias)), 3 * x2.numel() * 2)
```

结果与 `x + b` 逐位相同是可以期待的：两边都是"转 float、加、舍入回 BF16"，`assert_close` 的默认容差只是保险。

### 7. 读者应看到的量级，以及为什么最后 8–10% 拿不到

没有实测数字可以照抄——不同型号、频率、功耗墙下带宽都不同——但按文献与经验，A100 上 $$n = 2^{28}$$ 的结果大致落在这些区间：

```text
版本                 有效带宽占标称峰值       特征
naive（2 B/线程）     约 60–80%               合并但每指令 64 B，LSU 与指令发射成瓶颈
vec8（16 B/线程）     约 85–92%               每 warp 一条指令 512 B
vec8 + grid-stride    约 85–92%               与 vec8 相当或略好 1–3 个百分点
torch.add (ATen)      约 85–92%               A100 上实际为 vec4（8 B/线程）+ 每线程 8 元素
strided, xs1 = 1      接近 naive
strided, x.t()        远低于 10%              最内维 stride 4096，每元素独占一个 sector
```

如果 naive 版本就跑到了 85% 以上，大概率是 L2 没冲干净或 tensor 太小；如果 vec8 版本低于 80%，先检查 `nvcc` 有没有真的生成 128-bit 加载（`cuobjdump -sass` 里找 `LDG.E.128`），再看是否触发了对齐回退。

那最后 8–10% 去了哪里？

- **标称带宽是接口峰值，不是可达带宽**。HBM 有刷新周期、bank 冲突、读写方向切换（bus turnaround）的开销，读写混合的负载通常只能做到标称的 90–93%，纯读能高一点。`add` 是 2 读 1 写，天生有方向切换。
- **启动与收尾**。kernel 开始时 warp 逐渐被调度、请求逐渐填满管道；结束时最后一批 block 收尾，SM 逐渐空闲。这两段带宽利用不足，对 0.8 ms 的 kernel 大约占 1–2%。
- **L2 与 DRAM 页局部性**。写回策略、L2 分片之间的交叉带宽、DRAM 页打开/关闭，都不是 kernel 能控制的。
- **功耗与频率**。带宽压满时 HBM 与 SM 的功耗都高，GPU 可能降频，标称值对应的是理想条件。

这些因素合起来就是那道 90% 的墙。在它面前，继续调 block 大小、展开因子、grid 倍数，收益都在噪声范围内。此时应该做的事在第六章已经说过：融合，或者少做。


## 八、小结

这一篇围绕一个理论下界（BF16 `add`：6 B/元素，$$n = 2^{28}$$ 时 0.81 ms）讨论了 elementwise kernel 的全部工程要点：

- **合并**：warp 的 32 个地址落在多少个 32 B sector 里决定效率；连续对齐 100%，跨步 2 只剩 50%，跨步 ≥ 32 B 只剩 12.5%，未对齐多付一个 sector；AoS 是隐藏的跨步。
- **向量化**：16 字节/线程的 `int4`/`float4` 加载让每条指令搬 512 B，指令数减到 1/8（BF16），要求 16 字节对齐，尾部用标量补。
- **在飞请求**：Little's law 给出 A100 需要 ~1.2 MB 在飞、每 SM ~11 KB、128-bit 加载下至少 ~22 个 warp；memory-bound 同样需要占用率与 ILP。
- **grid-stride**：解耦 grid 与 $$n$$，摊薄固定开销，提供 ILP；ATen 没有用它，也能到 90%。
- **stride 与 broadcast**：线性 index 逐维 divmod 乘 stride；broadcast 即 stride 0；最内维 stride 不为 1 时应先 `contiguous()`。
- **ATen 实现**：`gpu_kernel` → 连续则 `launch_vectorized_kernel`（`can_vectorize_up_to` 查对齐，`16 / sizeof` 定宽度，A100 上 BF16 压到 vec4）→ `vectorized_elementwise_kernel`（128 线程、每线程 8 元素、只有最后 block 做边界检查）；非连续走 `elementwise_kernel` + `OffsetCalculator`；`AT_DISPATCH_*` 把运行期 dtype 展开成 `switch`，每个 `case` 里 `using scalar_t = ...` 实例化一份模板。
- **融合**：三个 elementwise 分开 16 B/元素，融合后 8 B/元素，这是 Inductor 融合的全部收益来源；90% 带宽之后，唯一的优化是让 kernel 消失。

数字汇总：

```text
访存模式（32 线程 × 4 B）          sector    效率
连续、128 B 对齐                   4         100%
连续、起点偏移 4 B                 5          80%
跨步 2                             8          50%
跨步 4                            16          25%
跨步 ≥ 8（≥ 32 B）                32        12.5%
AoS 12 B 结构体只读一个字段        12          33%

Little's law（A100）
在飞字节 = 2.0 TB/s × 600 ns ≈ 1.2 MB；每 SM ≈ 11 KB
128-bit 加载：512 B/warp → ≥ 22 warp/SM；32-bit 加载：128 B/warp → ≈ 88 warp/SM（超上限 64）

三版 BF16 add kernel（n = 2^28，理论下界 0.81 ms @ 2.0 TB/s）
版本        每线程元素   每 warp 每指令字节   grid            常见带宽占比
naive       1            64 B                 n/256           60–80%
vec8        8            512 B                n/8/256         85–92%
vec8 + gs   8 × k        512 B                SM × blk × 2    85–92%

融合（BF16，silu(x + b) * y）
三个 kernel：5 读 3 写 = 16 B/元素；融合：3 读 1 写 = 8 B/元素 → 时间减半

ATen（v2.10.0）elementwise 配置
num_threads = 128；vectorized 每线程 8 元素（io_size = 1 时 16）；vec_size = min(16/sizeof, 对齐)，非 sm_90/100 上 ≤ 4
unrolled 每线程 4 元素；legacy elementwise_kernel<128, 2 或 4> + OffsetCalculator（MAX_DIMS = 25）
```

下一篇进入需要线程之间协作的 kernel。softmax、LayerNorm、RMSNorm 都要对一行做归约，而归约的结果要被同一行的所有元素使用——这需要 shared memory、warp shuffle 与 `__syncthreads()`，也需要 online softmax 把三遍读变成一遍。它们的理论下界仍然是"读一遍写一遍"，但实现的自由度和陷阱都比 elementwise 多得多。


## 下一篇

[共享内存与 reduction：softmax、LayerNorm 与 online softmax](/shared-memory-reduction-and-softmax.html)
