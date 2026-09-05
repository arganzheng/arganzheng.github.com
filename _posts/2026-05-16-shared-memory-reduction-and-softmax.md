---
layout: post
title: "GPU Kernel 工程（04）：共享内存与 reduction——softmax、LayerNorm 与 online softmax"
subtitle: "Shared Memory and Reductions: Softmax, LayerNorm and Online Softmax"
tags: [CUDA, Triton, GPU, AI, AI-Infra]
catalog: true
---

> 本文是[《GPU Kernel 工程：从 CUDA 执行模型到 FlashAttention》](/gpu-kernel-engineering.html)系列的第 4 篇（共十篇）。上一篇：[访存合并与 elementwise kernel](/memory-coalescing-and-elementwise-kernels.html)　下一篇：[GEMM：从 naive 到分块](/gemm-from-naive-to-tiled.html)

上一篇的 elementwise kernel 有一个共同特征：每个线程只管自己的元素，线程之间不需要说话。把访存合并、向量化、grid-stride 做对，带宽就能推到 90% 以上。

这一篇进入需要线程**协作**的 kernel。RMSNorm 要先知道整行的均方才能缩放每个元素；softmax 要先知道整行的最大值和指数和才能归一化每个元素。"先算出一个整行的标量，再用它处理每个元素"——这个模式叫 reduction，它是所有 norm、softmax、loss、统计类算子的核心，也是 FlashAttention 内循环的一半。

总纲给这一篇的核心问题是：

> **一个 4096 维的 RMSNorm，读一次写一次，理论上是 memory-bound 的。为什么 naive 实现能慢 10 倍？shared memory 和 warp shuffle 各解决了哪部分？**

回答它需要三块知识：shared memory（block 内线程交换数据的地方）、`__syncthreads()`（交换数据时的栅栏）、warp shuffle（不经过 shared memory 的 32 线程内交换）。本文先算理论下界，再把这三块讲清楚，然后用六个版本把一个 reduction 从 naive 推到接近极限，最后落到 softmax、LayerNorm/RMSNorm 两类真实 kernel，以及 online softmax 的推导——它是第八篇 FlashAttention 的数学基础。

硬件基线仍取 A100 SXM 80GB 的公开标称值：HBM 约 2.0 TB/s，108 个 SM，每 SM L1/shared 合计 192 KB（shared 最大可配 164 KB，单 block 最大 163 KB 需 opt-in，默认静态上限 48 KB），shared memory 32 个 bank、每 bank 4 字节宽、每 SM 每周期 128 字节。


## 一、先算理论下界

### 1. RMSNorm：读一次写一次

RMSNorm 对一行 $$x \in \mathbb{R}^d$$ 的定义是：

$$
y_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \epsilon}} \cdot \gamma_i
$$

它必须读完整行才能得到分母，然后再对每个元素做一次乘法。最少的访存是：读一次 $$x$$（$$2d$$ 字节，BF16），写一次 $$y$$（$$2d$$ 字节），再读一次 $$\gamma$$（$$2d$$ 字节，但它对所有行共享，被 L2 缓存后只算一次，可忽略）。

取 $$d = 4096$$：每行读 8 KiB、写 8 KiB，共 16 KiB。取 batch·seq = 8192 行（比如 batch 4、序列 2048）：

$$
8192 \times 16\ \text{KiB} = 128\ \text{MiB} \approx 134.2\ \text{MB}
$$

$$
t_{\min} = \frac{134.2\ \text{MB}}{2.0\ \text{TB/s}} \approx 67\ \mu s
$$

FLOPs 方面：每个元素一次乘方累加（2 FLOP）、一次乘 rstd、一次乘 $$\gamma$$（2 FLOP），每行再加一次 rsqrt，合计约 $$4d$$ FLOP，每字节 $$4d / 4d = 1$$ FLOP，远低于 A100 FP32 CUDA Core 的 ridge point $$19.5 / 2.0 \approx 10$$ FLOP/byte。**RMSNorm 是彻底的 memory-bound kernel**，67 µs 是它在 A100 上的物理下界，任何实现只能逼近，不能突破。

### 2. softmax：三遍还是一遍，差的是读几次

softmax 一行 $$x \in \mathbb{R}^N$$：

$$
y_i = \frac{e^{x_i - m}}{\sum_{j} e^{x_j - m}}, \qquad m = \max_j x_j
$$

减去 $$m$$ 是为了数值安全（后面第六章解释），它带来的代价是：需要先知道 $$m$$，才能算 $$\sum e^{x_j - m}$$，再才能写 $$y$$。最朴素的实现要**三遍扫描**：第一遍求 max，第二遍求指数和，第三遍归一化并写出。如果一行放不进寄存器或 shared memory，每遍都要从 HBM/L2 重新读，总流量是**读 3 次、写 1 次**。

如果一行能放进寄存器（比如 $$N \le 4096$$，一个 warp 处理一行，每线程 128 个 float），三遍扫描只发生在寄存器里，HBM 流量仍是读 1 次写 1 次。这时"三遍还是一遍"影响的是指令数，不是带宽。

但有两种情况一行放不进寄存器：一是词表 logits（Llama-3 的 $$V = 128256$$，一行 BF16 就是 250 KiB，超过 shared memory 上限）；二是 attention 的 $$QK^T$$ 分数矩阵，它根本不应该被物化。这时 online softmax 的价值就出来了：**一遍读完同时得到 max 和归一化后的指数和**，把读 3 次降到读 2 次（一次求统计量，一次写出），在 FlashAttention 里进一步降到读 1 次（因为输出可以在寄存器里被重新缩放）。

### 3. naive 慢 10 倍的四个来源

有了下界，就能问：为什么很多人第一次写的 RMSNorm 跑出来是 500–700 µs 而不是 67 µs？常见原因有四个，后文会逐一解决：

- **一个线程串行求和**。让线程 0 循环 4096 次把平方加起来，其他 255 个线程等着。一个线程的串行加法是 latency-bound 的，4096 次依赖链上的 FMA 要几万个周期，而整个 block 读 8 KiB 只需要几百个周期。
- **每线程 `atomicAdd` 到同一个地址**。256 个线程同时对 shared 或 global 的同一个 float 做原子加。原子操作在同一地址上是串行化的，256 次争用比串行加法更慢，而且 float 加法不满足结合律，结果每次运行还不一样。
- **多次 kernel launch**。先 launch 一个 kernel 求平方和写回 global，再 launch 一个 kernel 读回来缩放。每次 launch 有几微秒的固定开销，中间结果还要往返 HBM 一次。
- **shared memory bank conflict 或 `__syncthreads()` 太多**。用了 shared memory 但访问模式让 32 个线程撞到同一个 bank，或者 reduction 树的每一级都做一次 block 级同步——每次同步都让 block 里所有 warp 停下来等最慢的那个。

这四个问题的解法分别是：让所有线程并行累加、用树形归约替代原子加、把两个阶段融合进一个 kernel、用 warp shuffle 替代 shared memory 归约的最后五级。下面从 shared memory 开始。


## 二、shared memory：block 内的公共草稿纸

### 1. 声明：静态、动态、超过 48 KB

shared memory 是每个 SM 上的一块片上 SRAM，由该 SM 上正在运行的 block 分享。一个 block 内的所有线程看到同一块 shared memory，不同 block 之间互不可见。它的延迟约为 20–30 个周期，是 global memory（几百个周期）的十分之一左右，带宽是每 SM 每周期 128 字节。

有两种声明方式。**静态**：大小在编译期确定。

```cpp
__global__ void k_static(const float* in, float* out) {
  __shared__ float tile[256];          // 1 KiB，编译期确定
  tile[threadIdx.x] = in[threadIdx.x];
  __syncthreads();
  out[threadIdx.x] = tile[255 - threadIdx.x];
}
```

**动态**：大小在 launch 时通过 `<<<grid, block, smem_bytes>>>` 的第三个参数给出，kernel 内用 `extern __shared__` 声明一个不定长数组：

```cpp
__global__ void k_dynamic(const float* in, float* out, int n) {
  extern __shared__ float buf[];       // 大小由 launch 第三参数决定
  for (int i = threadIdx.x; i < n; i += blockDim.x) buf[i] = in[i];
  __syncthreads();
  for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = buf[n - 1 - i];
}

// launch：n 个 float 的动态 shared memory
k_dynamic<<<1, 256, n * sizeof(float)>>>(in, out, n);
```

一个 kernel 里只能有一个 `extern __shared__` 数组；需要多种类型时，声明成 `char` 或 `float` 再自己切分、对齐。

默认情况下，一个 block 最多申请 48 KB shared memory（静态加动态）。要用更多，必须在 launch 之前对该 kernel opt-in：

```cpp
constexpr int kSmem = 96 * 1024;     // 96 KB，超过 48 KB 默认上限
cudaFuncSetAttribute(k_dynamic,
                     cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
k_dynamic<<<grid, block, kSmem>>>(in, out, n);
```

A100 上单 block 上限是 163 KB，H100 是 227 KB。GEMM 和 FlashAttention 的大 tile 都依赖这个 opt-in，第五、六、八篇会反复用到。

### 2. 容量、与 L1 的关系、生命周期

A100 每个 SM 有 192 KB 的片上存储，由 shared memory 和 L1 data cache **共享**：shared memory 可以配置为 0、8、16、32、64、100、132、164 KB 中的一档，剩下的归 L1。CUDA 运行时会根据 kernel 申请的 shared 大小自动选档（也可以用 `cudaFuncAttributePreferredSharedMemoryCarveout` 给提示）。这意味着 shared memory 用得越多，L1 就越小——对 reduction 这类只需要几百字节 shared 的 kernel无所谓，对 GEMM 则要权衡。

生命周期等于 block：block 被调度到 SM 上时分配，block 的所有线程退出时释放。内容不会跨 block 保留，也不会被初始化——读之前不写，得到的是上一个 block 留下的垃圾。

一个 SM 上能同时驻留多少个 block，受 shared memory 总量限制：如果每个 block 申请 32 KB，164 KB 的配置下最多驻留 5 个 block；如果每个 block 只申请 128 字节（比如一次 block reduction 的 32 个 float），shared memory 就不再是 occupancy 的瓶颈，寄存器与线程数上限（每 SM 2048 线程、32 个 block）才是。

### 3. `__syncthreads()`：语义与陷阱

`__syncthreads()` 是 block 级栅栏：block 内所有线程都到达这一点之后，任何线程才能继续；同时它保证在它之前的 shared memory 与 global memory 写入对 block 内所有线程可见。典型用法是"写 shared → sync → 读 shared"：

```cpp
tile[threadIdx.x] = in[i];   // 每个线程写自己的位置
__syncthreads();             // 等所有人写完
float v = tile[(threadIdx.x + 1) % blockDim.x];   // 读别人写的位置
```

没有这个 sync，线程 5 可能在线程 6 写入之前就去读 `tile[6]`。

它有两条铁律：

**第一，所有线程都必须执行到同一个 `__syncthreads()`**。放在条件分支里、而分支条件在 block 内不一致，就是死锁或未定义行为：

```cpp
// 错误：部分线程永远到不了这个栅栏
if (threadIdx.x < 128) {
  tile[threadIdx.x] = in[threadIdx.x];
  __syncthreads();            // 后 128 个线程不会来，前 128 个永远等
}
// 正确：把 sync 提到分支外
if (threadIdx.x < 128) tile[threadIdx.x] = in[threadIdx.x];
__syncthreads();
```

同理，循环里的 sync 要求所有线程的循环次数一致；一个线程提前 `return` 之后，block 内后续的 `__syncthreads()` 对它而言就"缺席"了——所以 reduction 类 kernel 里通常用"越界线程贡献 0"而不是"越界线程 return"。

**第二，sync 是有代价的**。每次 `__syncthreads()` 都让 block 内先到的 warp 空转，等最慢的 warp。一个 1024 线程的 block 有 32 个 warp，某个 warp 因为一次 cache miss 晚到 500 个周期，其他 31 个 warp 就一起浪费 500 个周期。reduction 树每一级都 sync 一次，10 级就是 10 次全 block 等待。后面第三章的优化有一大半是在减少 sync 的次数。

### 4. bank conflict：32 个 bank，每个 4 字节

shared memory 在物理上被切成 **32 个 bank**，每个 bank 每周期能服务一个 4 字节的访问。连续的 4 字节字轮流分配给 32 个 bank：

$$
\text{bank}(addr) = \left\lfloor \frac{addr}{4} \right\rfloor \bmod 32
$$

也就是字地址 0 在 bank 0，字地址 1 在 bank 1，……，字地址 32 又回到 bank 0。一个 warp 的 32 个线程同时发起一次 shared 访问时：

- 32 个线程访问 32 个不同 bank → 一个周期完成（无冲突）；
- 多个线程访问**同一个 bank 的不同地址** → 硬件把它们串行化，$$k$$ 个线程撞同一个 bank 就是 $$k$$-way conflict，耗时 $$k$$ 个周期；
- 多个线程访问**同一个地址**（同一个字）→ 广播，一个周期完成，**不算冲突**。

64 位访问（double、`float2`）每线程占两个 bank，硬件分两个半 warp 处理，规则相应调整；128 位访问（`float4`）分四个 quarter warp。本文以 4 字节访问为主。

最经典的冲突场景是**二维 tile 的按列访问**。设 `__shared__ float tile[32][32]`，线程 `threadIdx.x` 访问 `tile[threadIdx.x][k]`（固定列 `k`，不同行）：

```text
线程 t 访问的字地址 = t * 32 + k
bank = (t * 32 + k) % 32 = k          ← 所有 32 个线程落在同一个 bank
```

这是 32-way conflict，一次访问要 32 个周期，等于把 shared memory 带宽砍成 1/32。按行访问 `tile[k][threadIdx.x]` 则地址连续，bank 各不相同，无冲突。所以规则是：**warp 内相邻线程访问的地址在 4 字节粒度上应该连续，或者至少 stride 与 32 互质**。stride 为 2 → 2-way，stride 为 4 → 4-way，stride 为 32 → 32-way；stride 为奇数则无冲突。

### 5. padding 与 swizzle

按列访问在 GEMM 的转置加载、矩阵转置 kernel 里躲不开。两种标准解法：

**padding**：把行宽从 32 改成 33。

```cpp
__shared__ float tile[32][33];   // 每行多一个字
// 线程 t 访问 tile[t][k]：字地址 = t * 33 + k
// bank = (t * 33 + k) % 32 = (t + k) % 32   ← 32 个线程落在 32 个不同 bank
```

代价是浪费 1/33 的 shared memory，以及行宽不再是 2 的幂、地址计算多一次乘法。对 32×32 的 tile 这个代价可以忽略。

**swizzle**：不改变存储大小，而是把逻辑列索引和行索引做 XOR，让同一列在不同行上落到不同 bank：

```cpp
__shared__ float tile[32][32];
__device__ __forceinline__ int swz(int r, int c) { return c ^ (r & 31); }
// 写：tile[r][swz(r, c)] = v;   读：v = tile[r][swz(r, c)];
// 线程 t 读第 k 列：物理列 = k ^ t，32 个 t 给出 32 个不同的物理列 → 32 个不同 bank
```

XOR 是一个置换：对固定的 `r`，`c → c ^ r` 把 0..31 一一映射到 0..31，所以每行的 32 个元素仍然占满 32 个位置，不浪费空间；对固定的 `c`，不同的 `r` 给出不同的物理列，所以按列访问不再冲突。swizzle 是 CUTLASS/CuTe 里 shared memory layout 的标准做法（第六篇会看到 `Swizzle<3,3,3>` 这样的类型），因为 `ldmatrix` 和 Tensor Core 的 fragment 加载要求 16 字节对齐、padding 不好用。

reduction 本身很少碰到 bank conflict，因为它的访问模式是连续的。但 Harris 那篇经典幻灯片里的"交错寻址"版本正好是一个 stride 为 2 的幂的例子，下一章会看到。


## 三、reduction 的六个版本

任务：一个 block 把 $$n$$ 个 float 加成一个数。这是 RMSNorm 的平方和、softmax 的指数和、LayerNorm 的均值的共同内核。下面六个版本对应 Harris（2007，"Optimizing Parallel Reduction in CUDA"）的思路，但用现代原语（`__shfl_*_sync`）重写。每一版都说清楚它去掉了什么。

假设 block 有 `BLOCK = 1024` 个线程，`sdata` 是 1024 个 float 的 shared 数组，每个线程已经把自己的一个元素放进 `sdata[tid]` 并 sync 过。

### v1：交错寻址 + 取模分支

```cpp
// v1: interleaved addressing, divergent branch
for (int s = 1; s < blockDim.x; s *= 2) {
  if (tid % (2 * s) == 0) sdata[tid] += sdata[tid + s];
  __syncthreads();
}
// 结果在 sdata[0]
```

第一轮 `s=1`：偶数线程加相邻的奇数线程。第二轮 `s=2`：tid 为 4 的倍数的线程加 `tid+2`。……10 轮之后 `sdata[0]` 是总和。

问题有三个。**分支发散**：`tid % (2*s) == 0` 在一个 warp 内一半为真一半为假（第一轮），之后是 1/4、1/8……但 warp 是整体调度的，只要 warp 里有一个线程活跃，整个 warp 就要走一遍，活跃线程越少浪费越大。**取模**：整数取模在 GPU 上不是单条指令。**bank conflict**：第 $$k$$ 轮里活跃线程 tid 是 $$2^k$$ 的倍数，它们访问 `sdata[tid]` 的地址 stride 是 $$2^k$$ 个字，从第 2 轮开始就有 2-way、4-way……的冲突。

Harris 的 v2 把 `if (tid % (2*s) == 0)` 换成 `int i = 2*s*tid; if (i < blockDim.x)`，前一半线程干活、后一半空闲——消除了发散和取模，但地址 stride 还是 $$2s$$，bank conflict 变得更严重。所以直接跳到下一步。

### v2：顺序寻址

```cpp
// v2: sequential addressing — no divergence within active warps, no bank conflict
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
  if (tid < s) sdata[tid] += sdata[tid + s];
  __syncthreads();
}
```

第一轮 `s=512`：前 512 个线程分别加后 512 个。活跃线程永远是连续的前 $$s$$ 个，所以：warp 要么全活跃要么全不活跃（$$s \ge 32$$ 时），没有发散；`sdata[tid]` 和 `sdata[tid+s]` 对连续的 tid 都是连续地址，没有 bank conflict；没有取模。

减少了：发散、取模、bank conflict。没变的是：10 次 `__syncthreads()`，以及一半的线程从第一轮起就闲着。

### v3：加载时先加一次

```cpp
// v3: first add during global load — each block handles 2 * blockDim.x elements
unsigned i = blockIdx.x * (blockDim.x * 2) + tid;
float v = 0.f;
if (i < n)              v += in[i];
if (i + blockDim.x < n) v += in[i + blockDim.x];
sdata[tid] = v;
__syncthreads();
// 然后接 v2 的循环
```

v2 的第一轮里 512 个线程各做一次加法，等价于"每个线程加载时就把两个元素加起来"，后者不需要 shared memory 也不需要 sync。于是让一个 block 处理 $$2 \times$$ `BLOCK` 个元素，第一级归约在寄存器里完成。

减少了：一次 sync、一半的 block 数（同样的数据量 grid 减半），shared memory 的一轮读写。这一步的思想推到极致就是 v5。

### v4：最后一个 warp 用 shuffle 展开

v2 的循环里当 $$s \le 16$$ 时，只有一个 warp 的一部分线程活跃，却还要全 block sync 5 次（s = 16, 8, 4, 2, 1）。这 5 级可以完全在 warp 内做，用 warp shuffle 指令直接读同 warp 其他 lane 的寄存器：

```cpp
// v4: shared-memory tree down to 32, then warp shuffle
for (int s = blockDim.x / 2; s > 32; s >>= 1) {
  if (tid < s) sdata[tid] += sdata[tid + s];
  __syncthreads();
}
if (tid < 32) {
  float v = sdata[tid] + sdata[tid + 32];          // s = 32 这一级
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_down_sync(0xffffffffu, v, off);   // s = 16..1
  if (tid == 0) sdata[0] = v;
}
```

`__shfl_down_sync(mask, v, off)` 返回 lane `lane + off` 的 `v`（越界时返回自己的 `v`），mask 为全 1 表示 32 个 lane 全部参与。5 轮之后 lane 0 持有 32 个值的和。shuffle 不经过 shared memory，没有 bank 问题，也不需要 `__syncthreads()`——它本身就是 warp 内的同步点。

减少了：5 次 `__syncthreads()`（10 次 → 5 次）、最后 5 级的 shared 读写。

### v5：每线程处理多个元素（grid-stride 加载）

v3 让每线程加载时加 2 个元素。为什么不加更多？把 grid 定为固定大小（比如 SM 数 × 每 SM 可驻留 block 数），每个线程用 grid-stride 循环把属于它的所有元素先在寄存器里累加：

```cpp
// v5: grid-stride load, many elements per thread, then v4 tree
float v = 0.f;
for (unsigned i = blockIdx.x * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x)
  v += in[i];
sdata[tid] = v;
__syncthreads();
// 然后接 v4
```

这时 shared memory 树只做一次，处理的是每线程已经累加过 $$n / (\text{grid} \times \text{BLOCK})$$ 个元素的部分和。算法复杂度上这是 Brent 定理的应用：$$O(n / p)$$ 的串行累加加 $$O(\log p)$$ 的树，总工作量 $$O(n)$$ 而不是 $$O(n \log n)$$。而且串行累加部分是纯 load + FMA，可以被 ILP 和多 warp 完美地重叠——这正是 elementwise kernel 的访存模式，能推到带宽极限。

减少了：sync 和 shared 访问按处理元素数摊薄到接近零；每个 block 的固定开销（launch、树归约）被更多的数据分摊。

### v6：两级 shuffle——现代写法

v4 还留着一棵 shared memory 树（1024 → 32 需要 5 级、5 次 sync）。反过来想：先让**每个 warp**用 shuffle 把自己的 32 个值归约成 1 个，32 个 warp 得到 32 个部分和，写进 shared 的 32 个 float，sync 一次，再由第一个 warp 用 shuffle 把这 32 个值归约成 1 个：

```cpp
// v6: two-level warp shuffle — 1 (or 2) __syncthreads() for the whole block
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;                                   // 32 个 lane 都持有总和
}

__device__ float block_reduce_sum_v6(float v, float* shared /* >= 32 floats */) {
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const int nwarps = (blockDim.x + 31) >> 5;
  v = warp_reduce_sum(v);                     // 级 1：warp 内 5 步 shuffle
  if (lane == 0) shared[wid] = v;             // 每个 warp 写一个部分和
  __syncthreads();                            // 唯一必需的 block 级同步
  v = (threadIdx.x < nwarps) ? shared[lane] : 0.f;
  if (wid == 0) v = warp_reduce_sum(v);       // 级 2：warp 0 再 5 步 shuffle
  return v;                                   // 只有 warp 0 的结果有效
}
```

这里用 `__shfl_xor_sync` 而不是 `__shfl_down_sync`：XOR 版本做的是蝶形交换，5 轮之后**所有 32 个 lane** 都持有总和，而不只是 lane 0。这在需要把结果广播给整个 warp 时省掉一次额外的 shuffle。

整个 block 归约只用了 1 次 `__syncthreads()`、32 个 float 的 shared memory（128 字节）、10 步 shuffle。如果 kernel 里要连续调两次 block reduce（比如 LayerNorm 先求均值再求方差），第二次调用的 `shared[wid] = v` 可能在有线程还没读完上一次 `shared[lane]` 时发生，所以在写之前要再加一次 `__syncthreads()`——PyTorch 的 `BlockReduceSum` 就是这样做的（后面读源码会看到）。

### 六版对比

```text
版本  寻址方式            sync 次数   shared 读写      主要消除的问题
v1    交错 + 取模          10          10 级树           —
v2    顺序                 10          10 级树           发散、取模、bank conflict
v3    v2 + 加载时先加      9           9 级树            一半 block、一级 shared
v4    v2 树到 32 + shfl    5           5 级树            最后 5 级 sync 与 shared
v5    grid-stride + v4     5           5 级树（摊薄）     每 block 固定开销
v6    warp shfl × 2        1 (或 2)    32 float          几乎全部 shared 流量与 sync
```

回到总纲的问题：**shared memory 解决的是"block 内 32 个 warp 的部分和怎么汇总"**——它是唯一能让不同 warp 交换数据的地方；**warp shuffle 消灭的是最后 5 级同步**——32 个 lane 之内的交换不需要栅栏也不需要 shared memory。两者组合成 v6，就是今天所有生产 kernel（PyTorch、vLLM、CUB）里 block reduction 的形状。


## 四、warp 级原语与原子操作

### 1. `__shfl_*_sync` 家族

上一章用了两个 shuffle 指令，这里把整个家族列出来。所有 `*_sync` 变体的第一个参数是 32 位 mask，指定哪些 lane 参与；mask 中的每个 lane 都必须执行到这条指令，否则行为未定义。绝大多数情况下用 `0xffffffffu`（全 warp）。

- `__shfl_sync(mask, v, srcLane)`：所有 lane 读取 `srcLane` 的 `v`。用于**广播**——lane 0 算出一个标量，其他 31 个 lane 一条指令拿到。
- `__shfl_up_sync(mask, v, delta)` / `__shfl_down_sync(mask, v, delta)`：读取 `lane − delta` / `lane + delta` 的 `v`，越界返回自身。down 用于归约树（结果落在 lane 0），up 用于前缀和（scan）。
- `__shfl_xor_sync(mask, v, laneMask)`：读取 `lane ^ laneMask` 的 `v`。蝶形模式，5 轮 `laneMask = 16, 8, 4, 2, 1` 之后所有 lane 都持有归约结果。

支持的类型是 32 位和 64 位标量（int、unsigned、long long、float、double，以及 `__half`/`__nv_bfloat16` 通过重载）。归约 `(m, l)` 这样的二元组要分别 shuffle 两个分量。

### 2. 投票与硬件归约

- `__ballot_sync(mask, pred)`：返回一个 32 位整数，第 $$i$$ 位是 lane $$i$$ 的 `pred`。配合 `__popc` 可以一条指令数出 warp 内有多少个线程满足条件，配合 `__ffs` 找到第一个满足的 lane。stream compaction、top-k 的分桶、稀疏化都靠它。
- `__any_sync(mask, pred)` / `__all_sync(mask, pred)`：warp 内是否有任一 / 全部线程满足条件。常用于"这个 warp 是否需要走慢路径"的判断，避免发散。
- `__reduce_add_sync(mask, v)`、`__reduce_min_sync`、`__reduce_max_sync`、`__reduce_and/or/xor_sync`（**sm_80+**）：一条指令完成 warp 内归约，但**只支持 32 位整数**（`int` / `unsigned`）。浮点归约仍要走 5 步 shuffle。整数版本在直方图、计数、掩码合并里能省 4 条指令。

### 3. cooperative groups

CUDA 9 引入的 cooperative groups（`#include <cooperative_groups.h>`）把"一组线程"变成一等对象，让归约代码不用手写 lane 索引：

```cpp
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__device__ float warp_sum_cg(float v) {
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
  return cg::reduce(warp, v, cg::plus<float>());   // 所有 lane 得到结果
}
```

`tiled_partition<32>` 把 block 切成 32 线程的 tile（就是 warp），`cg::reduce` 在 tile 上做归约，编译器会对整数类型自动选用 `__reduce_add_sync`，对浮点类型展开成 shuffle 树。`tiled_partition<16>` 或 `<8>` 可以在半个或四分之一 warp 上归约，对"一行只有 16 个元素"这样的小 reduction 很方便。`block.sync()` 等价于 `__syncthreads()`。生产代码（PyTorch、vLLM）多数仍直接用 intrinsic，但 cooperative groups 是官方推荐的可读写法。

### 4. 原子操作：什么时候用，什么时候避免

`atomicAdd(float* addr, float v)` 在 global 或 shared 地址上做原子加。它解决的是跨线程、跨 block 向同一个地址累加的问题，代价有两个：

**争用串行化**。同一个地址上的原子操作必须一个接一个执行。在 L2 上（global atomic）每次大约几十到上百个周期；256 个线程打同一个地址就是 256 次串行。shared memory 上的原子操作（sm_60+ 有原生实现）快一些，但同地址争用同样串行。

**浮点非确定性**。float 加法不满足结合律：$$(a + b) + c \ne a + (b + c)$$。原子加的执行顺序由硬件调度决定，每次运行不同，结果的最后几位也就不同。对训练里要求 bitwise 可复现的场景（vLLM 里甚至有一个 `batch_invariant` 模式专门为此关掉某些优化路径），这是不可接受的。

因此规则是：**block 内用树形归约，不用原子；跨 block 汇总少量值（比如每个 block 一个部分和、总共几百个 block）可以用原子，争用低、非确定性可控；热点地址（成千上万个线程加同一个位置）坚决避免**。替代方案是两阶段：每 block 把部分和写到 `partial[blockIdx.x]`，再 launch 一个小 kernel 归约这几百个值；或者用"最后到达的 block 负责收尾"的 semaphore 模式——PyTorch `Reduce.cuh` 的 `global_reduce` 就是这样，第八章会看到。


## 五、一行一个 block，还是一行一个 warp

RMSNorm、softmax、LayerNorm 都是**按行归约**：输入是 $$[R, d]$$，每行独立算一个（或两个）标量再作用回该行。这时有两种基本的线程组织：

**一行一个 block**（block-per-row）：block 内 $$T$$ 个线程分担 $$d$$ 个元素，每线程 $$d / T$$ 个，用第三章的 `block_reduce_sum` 汇总。grid 大小 = 行数 $$R$$。需要 shared memory（32 个 float）和至少一次 `__syncthreads()`。

**一行一个 warp**（warp-per-row）：一个 warp 的 32 个 lane 分担一行，每 lane $$d / 32$$ 个元素放在寄存器里，用 5 步 shuffle 汇总。一个 block 装若干个 warp（比如 4 或 8），grid 大小 = $$R / \text{warps\_per\_block}$$。不需要 shared memory，不需要 `__syncthreads()`。

选哪个由 $$d$$ 决定：

- $$d \le 1024$$（BF16 一行 ≤ 2 KiB）：一个 warp 处理一行，每 lane 最多 32 个元素、32 个寄存器，寄存器驻留、零 shared、零 block sync，三遍扫描全在寄存器里。PyTorch 的 `dispatch_softmax_forward` 走的就是这条路（阈值是 `dim_size <= 2048 && dim_size * sizeof(scalar_t) <= 8192`，即 BF16/FP16/FP32 都到 2048、FP64 到 1024，后面读源码会看到）。
- $$1024 < d \le 4096$$：warp-per-row 还能做，但每 lane 要 128 个寄存器持有数据，加上其他寄存器很可能超过 255 的上限或把 occupancy 压到很低；block-per-row 更稳。
- $$d > 4096$$：一个 block（256–1024 线程）处理一行，每线程 $$d / T$$ 个元素；如果 $$d$$ 大到寄存器也放不下（词表 logits），要么把行暂存进 shared memory（PyTorch 的 `cunn_SoftMaxForwardSmem`），要么接受第二遍重读 global（有 L2 兜底）。

grid 大小则由行数决定：$$R = \text{batch} \times \text{seq}$$。prefill 时 $$R$$ 是几千到几万，足以填满 108 个 SM；decode 时 $$R = \text{batch}$$，可能只有几十行——这时 block-per-row 只有几十个 block，GPU 大部分 SM 空闲，kernel 是 latency-bound 的，vLLM 的 `rms_norm` 在 `num_tokens < 256` 时把 block 放大到 1024 线程就是为了在行少时让每行的加载并行度更高。

一个补充：**warp-per-row 时行的边界处理更简单**。行是按 warp 分配的，`row >= R` 的判断对整个 warp 一致，可以直接 `return`，不会破坏后面的 shuffle（shuffle 的 mask 是整 warp）。block-per-row 时 block 内不会有"越界行"，越界的是元素，用"贡献 0"处理。


## 六、softmax：safe、三遍、两遍、online

### 1. 为什么要减 max

$$e^x$$ 增长得很快。FP16 的最大值是 65504，$$\ln 65504 \approx 11.09$$，也就是 **$$x > 11.09$$ 时 $$e^x$$ 在 FP16 里就是 inf**。BF16 和 FP32 有相同的 8 位指数，最大值约 $$3.4 \times 10^{38}$$，$$\ln(3.4 \times 10^{38}) \approx 88.7$$，即 $$x > 88.7$$ 时溢出。attention 分数 $$q \cdot k / \sqrt{d}$$ 和 lm_head logits 在训练不稳定时超过 88 并不罕见，超过 11 更是家常便饭。

减去行最大值 $$m$$ 之后，$$x_i - m \le 0$$，$$e^{x_i - m} \in (0, 1]$$，分子永远不溢出；分母至少有一项等于 1（最大值那一项），永远不为 0 也不下溢成 0。数学上 $$\frac{e^{x_i - m}}{\sum e^{x_j - m}} = \frac{e^{x_i}}{\sum e^{x_j}}$$，结果不变。这就是 safe softmax，所有生产实现都这样做，代价是必须先知道 $$m$$。

### 2. 三遍与两遍

**三遍**（朴素）：

```text
pass 1:  m = max_j x_j                    读 x
pass 2:  l = sum_j exp(x_j - m)           读 x
pass 3:  y_i = exp(x_i - m) / l           读 x，写 y
```

读 3 次、写 1 次。当一行在寄存器里时这三遍就是三个寄存器循环，无所谓；不在寄存器里时是 3 次 HBM/L2 读。

**两遍**的常见想法是：pass 1 求 max，pass 2 同时算 $$e^{x_i - m}$$ 写到 $$y$$ 并累加 $$l$$，最后再乘一个 $$1/l$$。但"最后再乘"本身又是一遍读写 $$y$$，总流量变成读 2 次、写 2 次、再读写 1 次，反而更多。所以真正把遍数降下来的，不是重排这三步，而是**让 max 和 sum 在同一遍里出来**。

### 3. online softmax 的推导

Milakov 与 Gimelshein（2018，"Online normalizer calculation for softmax"）的观察是：$$\sum_j e^{x_j - m}$$ 依赖 $$m$$，但如果 $$m$$ 中途变了，已经累加的和可以**用一个指数因子修正**。维护一对状态 $$(m, l)$$：$$m$$ 是目前见过的最大值，$$l$$ 是目前为止的 $$\sum e^{x_j - m}$$。来了一个新元素 $$x_i$$：

$$
m_{\text{new}} = \max(m, x_i)
$$

$$
l_{\text{new}} = l \cdot e^{m - m_{\text{new}}} + e^{x_i - m_{\text{new}}}
$$

第一项把旧的和从"相对旧 max"换算成"相对新 max"——因为 $$\sum_{j<i} e^{x_j - m} \cdot e^{m - m_{\text{new}}} = \sum_{j<i} e^{x_j - m_{\text{new}}}$$；第二项是新元素相对新 max 的贡献。如果 $$m_{\text{new}} = m$$（新元素没有刷新最大值），修正因子 $$e^0 = 1$$，退化成普通累加。指数的参数永远 $$\le 0$$，不会溢出。

一遍扫完，$$(m, l)$$ 就是最终的 max 和 sum。归一化仍需要再过一遍写出 $$y_i = e^{x_i - m} / l$$，所以 HBM 意义上的遍数是**读 2 次、写 1 次**，比三遍少一次读。

更重要的是它**可以并行**。两个部分状态 $$(m_a, l_a)$$（来自元素子集 $$A$$）和 $$(m_b, l_b)$$（来自子集 $$B$$）可以合并成 $$A \cup B$$ 的状态：

$$
m = \max(m_a, m_b), \qquad l = l_a \cdot e^{m_a - m} + l_b \cdot e^{m_b - m}
$$

这个合并运算满足结合律和交换律，所以 $$(m, l)$$ 可以像普通求和一样做树形归约：每个线程先串行处理自己的元素得到局部 $$(m, l)$$，然后用 shuffle 两两合并，5 步得到 warp 的 $$(m, l)$$，再经 shared 合并 32 个 warp。这就是 online softmax 与第三章 reduction 框架的接口——归约的对象从一个 float 变成一对 float，合并算子从 `+` 变成上面的公式。

### 4. 通向 FlashAttention

在 attention 里，softmax 的输出 $$P = \text{softmax}(QK^T / \sqrt{d})$$ 不是终点，而是要接着乘 $$V$$。FlashAttention（Dao 等 2022）把 online softmax 再推一步：不仅 $$l$$ 可以在 max 变化时用 $$e^{m - m_{\text{new}}}$$ 修正，累加中的输出 $$O = \sum_j e^{x_j - m} v_j$$ 也可以用同一个因子修正。于是 $$K$$、$$V$$ 按块流过 shared memory，每个块更新 $$(m, l, O)$$ 三元组，$$S$$ 和 $$P$$ 永远不需要完整地写出来——这是第八篇的主题，那里会给出完整推导；本篇只需要记住：**FlashAttention 的数学核心就是上面那两行 $$(m, l)$$ 的递推与合并公式**。


## 七、LayerNorm 与 RMSNorm

### 1. RMSNorm：均方在 FP32 累加

$$
y = \frac{x}{\sqrt{\frac{1}{d}\sum x_j^2 + \epsilon}} \cdot \gamma
$$

只需要一次归约（平方和），比 LayerNorm 少一个均值，这是它在 Llama 系模型里取代 LayerNorm 的工程理由之一。实现上要注意：

- **累加必须用 FP32**。BF16 只有 8 位尾数（约 3 位十进制有效数字），4096 个平方项在 BF16 里累加，误差会累计到百分之几；在 FP32 里累加，再转回 BF16 输出，误差只来自最后一次舍入。输入和 $$\gamma$$ 从 BF16 转 FP32 是一条 `cvt` 指令（或者 `__bfloat1622float2` 一次转两个），代价可以忽略。
- **$$\epsilon$$ 加在均方上再开方**，不是加在开方之后。Llama 用 $$10^{-5}$$ 或 $$10^{-6}$$，`rsqrtf` 是硬件近似指令，误差在 FP32 的 2 ulp 之内，对 BF16 输出而言完全够。
- **第二遍要么重读 global，要么把行留在寄存器**。d = 4096、256 线程时，每线程 16 个 BF16（两个 16 字节向量），留在寄存器毫无压力；vLLM 选择重读——8 KiB 的行刚被读过，一定在 L2 里（大概率也在 L1 里），代价是一次 L2 hit 的延迟而不是一次 HBM 访问。两种做法在带宽利用上差异很小。

### 2. LayerNorm：Welford 与相消

LayerNorm 需要均值和方差：

$$
y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta, \qquad \mu = \frac{1}{d}\sum x_j, \quad \sigma^2 = \frac{1}{d}\sum (x_j - \mu)^2
$$

朴素做法是一遍归约同时累加 $$\sum x$$ 和 $$\sum x^2$$，然后 $$\sigma^2 = E[x^2] - E[x]^2$$。这在数学上对，在浮点上有**相消**问题：当 $$\lvert \mu \rvert$$ 远大于 $$\sigma$$ 时（比如 $$x$$ 在 1000 附近、波动只有 1），$$E[x^2] \approx 10^6$$，$$E[x]^2 \approx 10^6$$，两个大数相减得到一个约等于 1 的小数，FP32 的 7 位有效数字里前 6 位都被相消掉，剩下 1 位精度，甚至可能得到负数。Transformer 的激活在残差流里确实会出现均值远大于方差的情况。

Welford（1962）的在线算法避免了这个问题：维护 $$(n, \mu, M_2)$$，$$M_2 = \sum (x_j - \mu)^2$$，每来一个元素：

$$
n \leftarrow n + 1, \quad \delta = x - \mu, \quad \mu \leftarrow \mu + \frac{\delta}{n}, \quad M_2 \leftarrow M_2 + \delta \cdot (x - \mu)
$$

每一步只做"减去当前均值"这样量级相当的运算，没有大数相消。和 online softmax 一样，它有一个**合并公式**（Chan 等 1979），让两个部分状态 $$(n_a, \mu_a, M_{2,a})$$、$$(n_b, \mu_b, M_{2,b})$$ 可以并行归约：

$$
n = n_a + n_b, \quad \delta = \mu_b - \mu_a, \quad \mu = \mu_a + \delta \cdot \frac{n_b}{n}, \quad M_2 = M_{2,a} + M_{2,b} + \delta^2 \cdot \frac{n_a n_b}{n}
$$

PyTorch 的 `WelfordOps::combine`（`aten/src/ATen/native/SharedReduceOps.h`）就是这四行的直译，`var`、`std`、`layer_norm` 的 CUDA 实现都走它。代价是每个元素多一次除法（或预先算好 $$1/n$$ 的乘法）和归约对象从 1 个 float 变成 3 个；对 memory-bound 的 LayerNorm 这点算力是免费的。

RMSNorm 没有这个问题——它不减均值，$$\sum x^2$$ 直接就是要的量，不存在相消。

### 3. fused residual + RMSNorm

Transformer 每一层里 norm 前面总有一个残差加：

```text
h = h + attn_out          # residual add（elementwise）
x = rmsnorm(h) * gamma    # 下一层的输入
```

分开实现是两个 kernel：residual add 读 2 写 1（读 h、attn_out，写 h），RMSNorm 读 1 写 1（读 h，写 x），共**读 3 写 2**，每元素 10 字节。融合成一个 kernel：读 h 和 attn_out，相加后就地写回 h（residual 流需要保留更新后的值供下一层用），同时累加平方和，第二遍用 rstd 缩放写出 x——**读 2 写 2**，每元素 8 字节，省 20%，还少一次 launch。vLLM 的 `fused_add_rms_norm` 就是这个 kernel，接口上 `input` 被就地改写为 norm 输出、`residual` 被就地改写为相加结果，下面读它的源码。


## 八、读源码

### 1. vLLM `rms_norm_kernel`：block-per-row + CUB 归约

`csrc/layernorm_kernels.cu`（v0.20.0）的 `rms_norm_kernel` 是教科书式的 block-per-row 结构。核心部分：

```cpp
// vllm csrc/layernorm_kernels.cu（v0.20.0）, rms_norm_kernel，节选
template <typename scalar_t, int VEC_SIZE, int NUM_DIMS>
__global__ void rms_norm_kernel(scalar_t* __restrict__ out,
                                const scalar_t* __restrict__ input, /* strides... */
                                const scalar_t* __restrict__ weight,
                                const float epsilon, const int num_tokens,
                                const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;
  // ... 根据 NUM_DIMS 计算 input_row ...
  auto vec_op = [&variance](const vec_n_t<scalar_t, VEC_SIZE>& vec) {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      float x = static_cast<float>(vec.val[i]);
      variance += x * x;
    }
  };
  // scalar_op 同理；对齐部分走 vec_op，头尾走 scalar_op
  vllm::vectorize_read_with_alignment<VEC_SIZE>(
      input_row, hidden_size, threadIdx.x, blockDim.x, vec_op, scalar_op);

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();
  // 第二遍：重新读 input_row 与 weight（向量化），乘 s_variance 写 out
  // ...
}
```

对照第三、五、七章逐点看：

- **一行一个 block**：`grid(num_tokens)`，`blockIdx.x` 就是行号。block 大小由 host 侧决定：`max_block_size = (num_tokens < 256) ? 1024 : 256`，行少时用大 block 增加每行的并行度，行多时用小 block 让更多 block 同时驻留。
- **向量化累加**：`VEC_SIZE = gcd(16 / sizeof(scalar_t), hidden_size)`，BF16 时是 8（16 字节一次加载）。累加变量 `variance` 是 `float`，每个 BF16 先 `static_cast<float>` 再平方——FP32 累加。
- **block 归约用 CUB**：`cub::BlockReduce<float, 1024>` 是 CUB 库的 block 级归约，内部就是 v6 的两级 shuffle（CUB 会根据架构选 warp shuffle 或 shared 树）。`TempStorage` 是它需要的 shared memory，模板参数 1024 是最大线程数，运行时用 `blockDim.x` 告诉它实际线程数。`CubAddOp` 在 `csrc/cub_helpers.h` 里定义，CUB 2.8 之后是 `cuda::std::plus<>`，之前是 `cub::Sum`。
- **归约结果只在线程 0 有效**——所以线程 0 算出 `rsqrtf` 写进 `s_variance`，`__syncthreads()` 之后全 block 读取。这就是"归约结果广播"的标准两步。
- **第二遍重读 input**：没有把第一遍读的向量留在寄存器，而是从 global 再读一次。8 KiB 的行刚被这个 block 读过，在 L2（很可能也在 L1）里，代价很小。

`fused_add_rms_norm_kernel` 的 FP16/BF16 特化版本用了自定义的 `_f16Vec<scalar_t, width>`（定义在 `csrc/type_convert.cuh`）：

```cpp
// vllm csrc/layernorm_kernels.cu（v0.20.0）, fused_add_rms_norm_kernel 向量化版，节选
for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
  int id = blockIdx.x * vec_hidden_size + idx;
  int64_t strided_id = blockIdx.x * vec_input_stride + idx;
  _f16Vec<scalar_t, width> temp = input_v[strided_id];
  temp += residual_v[id];              // 打包的 bf162 加法
  variance += temp.sum_squares();      // FP32 累加
  residual_v[id] = temp;               // residual 就地更新
}
// ... BlockReduce → s_variance → __syncthreads() ...
for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
  int id = blockIdx.x * vec_hidden_size + idx;
  int64_t strided_id = blockIdx.x * vec_input_stride + idx;
  _f16Vec<scalar_t, width> res = residual_v[id];    // 重读更新后的 residual
  _f16Vec<scalar_t, width> w = weight_v[idx];
  // 逐元素 x * s_variance * w，写回 input_v[strided_id]
}
```

`_f16Vec` 是 `alignas(16)` 的 POD 结构，`width = 8` 时正好 16 字节，`operator+=` 用 `__nv_bfloat162` 的打包加法一次处理两个元素。host 侧检查三个指针都 16 字节对齐、`hidden_size % 8 == 0`，才走 `width = 8` 的特化，否则回退到 `width = 0` 的标量版本。这正是第七章说的读 2 写 2：第一遍读 input 和 residual、写 residual；第二遍读 residual（L2 hit）和 weight、写 input。

### 2. PyTorch `SoftMax.cu`：warp softmax 与 block softmax 的分派

`aten/src/ATen/native/cuda/SoftMax.cu`（v2.10.0）的 `host_softmax` 按 `dim_size` 选实现：

```cpp
// pytorch aten/src/ATen/native/cuda/SoftMax.cu（v2.10.0）, host_softmax，节选
if (dim_size <= 2048 && dim_size*sizeof(scalar_t) <= 8192) {
  int64_t remaining = outer_size;
  int64_t chunk_size = (1L << 30L) / dim_size;
  while(remaining > 0) {
    dispatch_softmax_forward<scalar_t, scalar_t, accscalar_t, is_log_softmax, false>(
      output_ptr, input_ptr, dim_size, dim_size,
      std::min<int64_t>(remaining, chunk_size), nullptr/* not masked */);
    // ... 推进指针 ...
  }
} else {
  constexpr int ILP = sizeof(float4) / sizeof(scalar_t);
  dim3 block = SoftMaxForward_getBlockSize(dim_size);
  size_t smem_reduction_sz = block.x / at::cuda::warp_size() * sizeof(accscalar_t);
  // 三条路：寄存器驻留（潜在寄存器数 < 10）、整行进 shared、两遍读 global
  if (potential_reg_cnt < 10) { /* cunn_SoftMaxForwardReg<..., N> */ }
  else if (can_use_smem)      { /* cunn_SoftMaxForwardSmem，smem = dim_size*sizeof + reduction */ }
  else                        { /* cunn_SoftMaxForward，只用 reduction 的 shared */ }
}
```

**小行走 warp softmax**。`dispatch_softmax_forward`（`PersistentSoftmax.cuh`）在 `dim_size <= 2048` 且一行不超过 8 KiB 时被选中。它把 `dim_size` 向上取到 2 的幂 `next_power_of_two`，用 `log2_elements` 作为模板参数实例化 `softmax_warp_forward`：每个 warp 处理 `WARP_BATCH` 行（≤128 元素时 2 行，否则 1 行），每 lane 持有 `WARP_ITERATIONS = next_power_of_two / 32` 个元素在寄存器数组 `elements[WARP_BATCH][WARP_ITERATIONS]` 里；block 固定 128 线程 = 4 个 warp；max 和 sum 各用一次 `warp_reduce`：

```cpp
// pytorch aten/src/ATen/native/cuda/PersistentSoftmax.cuh（v2.10.0）
template <typename acc_t, int WARP_BATCH, int WARP_SIZE, template<typename> class ReduceOp>
__device__ __forceinline__ void warp_reduce(acc_t* sum) {
    ReduceOp<acc_t> r;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0;  i < WARP_BATCH;  ++i) {
            acc_t b = WARP_SHFL_XOR(sum[i], offset, WARP_SIZE);
            sum[i] = r(sum[i], b);
        }
    }
}
```

这就是第五章的 warp-per-row：零 shared memory、零 `__syncthreads()`，三遍（max、sum、write）都在寄存器上。注意它不是 online 版本——一行在寄存器里，三遍的代价只是指令，没必要用 online 递推。

**大行走 block softmax**。`cunn_SoftMaxForward` 是 block-per-row：`ilpReduce` 让每线程向量化（ILP = 8 个 BF16）读取并局部归约，然后 `blockReduceWarp` 做 block 归约。后者调用的 `cuda_utils::BlockReduce`（`block_reduce.cuh`）就是 v6：

```cpp
// pytorch aten/src/ATen/native/cuda/block_reduce.cuh（v2.10.0）, BlockReduceSum
template <typename T, typename B = Block1D>
__inline__ __device__ T BlockReduceSum(T val, T* shared) {
  const int tid = B::Tid();
  const int lid = tid % C10_WARP_SIZE;
  const int wid = tid / C10_WARP_SIZE;
  val = WarpReduceSum(val);
  __syncthreads(); // prevent races when BlockReduces are called in a row.
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (tid < B::Warps()) ? shared[lid] : T(0);
  if (wid == 0) {
    val = WarpReduceSum(val);
  }
  return val;
}
```

注意第一个 `__syncthreads()` 的注释："prevent races when BlockReduces are called in a row"——softmax 连续调两次（max 再 sum），第二次写 `shared[wid]` 之前必须确保第一次的 `shared[lid]` 已被读完。这正是第三章 v6 末尾提到的那一次额外 sync。同一文件里还保留着一个老的 `blockReduce`（无 warp shuffle、纯 shared 树，第一个 warp 串行读 32 个值），用于对照可以看到两种写法的差别：老版本 4 次 sync，新版本 2 次。

### 3. PyTorch `Reduce.cuh`：通用归约的三级结构

`torch.sum`、`torch.mean`、`torch.var` 等所有通用归约走 `aten/src/ATen/native/cuda/Reduce.cuh` 的 `ReduceOp`。它要处理任意维度、任意 stride 的 reduction，所以结构比 softmax 复杂，但归约本身仍是三级：

- **thread reduce**：每线程串行累加自己负责的元素。当归约维是最内层且连续（`reduction_on_fastest_striding_dimension && dim0 >= 128`）时启用 `vectorize_input`，用 `aligned_vector` 一次读 16 字节，且用 `input_vec_size` 个独立累加器（`value_list[i]`）打破相邻 FMA 之间的依赖链——这是第五章 v5 的 grid-stride 加载加上 ILP。
- **block_x_reduce / block_y_reduce**：block 内沿 x 或 y 方向归约。`block_x_reduce` 先用 shared 树把 `blockDim.x` 压到 32（每级一次 `__syncthreads()`），再用 `warp_shfl_down` 做最后 5 级——这是第三章的 v4 形状。`block_y_reduce` 沿 y 方向纯 shared 树。选 x 还是 y 取决于归约维是否是最内层：是则沿 x（连续线程读连续地址），否则沿 y（让 x 方向的线程各自对应不同的输出以保持合并访存）。
- **global_reduce**：一行太长、一个 block 不够时，`ctas_per_output` 个 block 分担一个输出，每个 block 把部分和写进 global 的 staging buffer，`__threadfence()` 之后 `atomicAdd(&semaphores[blockIdx.x], 1)`——**最后一个到达的 block**（`prev_blocks_finished == gridDim.y - 1`）负责读回所有部分和做最终归约。这就是第四章说的"跨 block 汇总少量值用原子、但不用原子累加浮点"的模式：原子操作只用来计数，浮点归约仍是确定性的树。

```cpp
// pytorch aten/src/ATen/native/cuda/Reduce.cuh（v2.10.0）, mark_block_finished，节选
C10_DEVICE bool mark_block_finished() const {
  __shared__ bool is_last_block_done_shared;
  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    int prev_blocks_finished = atomicAdd(&semaphores[blockIdx.x], 1);
    is_last_block_done_shared = (prev_blocks_finished == gridDim.y - 1);
  }
  // ... __syncthreads(); return is_last_block_done_shared;
}
```

读这三处源码可以看到同一个模式的三种规模：softmax 的 warp 版本是"一级"（只有 shuffle），block 版本和 vLLM 的 RMSNorm 是"两级"（shuffle + shared），`Reduce.cuh` 是"三级"（shuffle + shared + global semaphore）。


## 九、实践：RMSNorm 与 online softmax

下面是完整可编译的实现（`nvcc -arch=sm_80`）。所有代码累加用 float，输入输出用 `__nv_bfloat16`。

### 1. `warp_reduce_sum` / `block_reduce_sum` 模板

```cpp
// reduce_utils.cuh
#pragma once
#include <cuda_runtime.h>

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;   // 所有 32 个 lane 持有结果
}

template <typename T>
__device__ __forceinline__ T warp_reduce_max(T v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v = max(v, __shfl_xor_sync(0xffffffffu, v, off));
  return v;
}

// 要求：blockDim.x 是 32 的倍数且 <= 1024；shared 至少 32 个 T。
// 返回值对 block 内所有线程有效（已广播）。可以连续调用。
template <typename T>
__device__ __forceinline__ T block_reduce_sum(T v, T* shared) {
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  const int nwarps = blockDim.x >> 5;
  v = warp_reduce_sum(v);
  __syncthreads();                       // 保护上一次调用的 shared 读
  if (lane == 0) shared[wid] = v;
  __syncthreads();
  v = (threadIdx.x < nwarps) ? shared[lane] : T(0);
  if (wid == 0) v = warp_reduce_sum(v);  // warp 0 的所有 lane 得到总和
  if (threadIdx.x == 0) shared[0] = v;
  __syncthreads();
  return shared[0];
}
```

三次 `__syncthreads()`：第一次防止与上一次调用竞争，第二次让 32 个部分和可见，第三次广播最终结果。如果只需要线程 0 拿到结果，可以去掉最后一次 sync 和写回，退化成 PyTorch `BlockReduceSum` 的形状。

### 2. RMSNorm kernel：一行一个 block，8 个 BF16/线程，寄存器驻留

```cpp
// rmsnorm.cu
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "reduce_utils.cuh"

// 一行一个 block。每线程每次加载 8 个 BF16（16 字节），共 ITERS 次；
// 要求 d == BLOCK * 8 * ITERS，且 in/out/gamma 16 字节对齐。
template <int BLOCK, int ITERS>
__global__ void __launch_bounds__(BLOCK)
rmsnorm_bf16_kernel(__nv_bfloat16* __restrict__ out,
                    const __nv_bfloat16* __restrict__ in,
                    const __nv_bfloat16* __restrict__ gamma,
                    int d, float eps) {
  static_assert(BLOCK % 32 == 0 && BLOCK <= 1024, "bad BLOCK");
  constexpr int VEC = 8;
  __shared__ float red[32];

  const size_t row = blockIdx.x;
  const uint4* in_v  = reinterpret_cast<const uint4*>(in + row * d);
  const uint4* g_v   = reinterpret_cast<const uint4*>(gamma);
  uint4* out_v       = reinterpret_cast<uint4*>(out + row * d);

  // 第一遍：加载到寄存器，FP32 累加平方和
  uint4 buf[ITERS];
  float ss = 0.f;
#pragma unroll
  for (int it = 0; it < ITERS; ++it) {
    buf[it] = in_v[threadIdx.x + it * BLOCK];      // 合并：相邻线程读相邻 16 字节
    const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(&buf[it]);
#pragma unroll
    for (int j = 0; j < VEC / 2; ++j) {
      float2 f = __bfloat1622float2(p2[j]);
      ss += f.x * f.x + f.y * f.y;
    }
  }
  ss = block_reduce_sum(ss, red);                  // 所有线程得到整行平方和
  const float rstd = rsqrtf(ss / (float)d + eps);

  // 第二遍：寄存器中的值乘 rstd 与 gamma，写出
#pragma unroll
  for (int it = 0; it < ITERS; ++it) {
    const int idx = threadIdx.x + it * BLOCK;
    uint4 gk = g_v[idx];
    uint4 ok;
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&buf[it]);
    const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&gk);
    __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&ok);
#pragma unroll
    for (int j = 0; j < VEC / 2; ++j) {
      float2 x = __bfloat1622float2(x2[j]);
      float2 w = __bfloat1622float2(w2[j]);
      o2[j] = __floats2bfloat162_rn(x.x * rstd * w.x, x.y * rstd * w.y);
    }
    out_v[idx] = ok;
  }
}

// host 侧：按 d 选模板实例。d 必须是 BLOCK*8 的倍数。
void rmsnorm_bf16(__nv_bfloat16* out, const __nv_bfloat16* in,
                  const __nv_bfloat16* gamma, int rows, int d, float eps,
                  cudaStream_t stream) {
  constexpr int BLOCK = 256;                       // 256 线程 × 8 = 2048 元素/次
  dim3 grid(rows), block(BLOCK);
  switch (d) {
    case 2048: rmsnorm_bf16_kernel<BLOCK, 1><<<grid, block, 0, stream>>>(out, in, gamma, d, eps); break;
    case 4096: rmsnorm_bf16_kernel<BLOCK, 2><<<grid, block, 0, stream>>>(out, in, gamma, d, eps); break;
    case 8192: rmsnorm_bf16_kernel<BLOCK, 4><<<grid, block, 0, stream>>>(out, in, gamma, d, eps); break;
    default: /* 其他 d：回退到通用的 grid-stride + 重读版本，此处省略 */ break;
  }
}
```

逐行核对：`in_v[threadIdx.x + it * BLOCK]` 让 warp 内 32 个线程读连续的 512 字节（4 条 128 字节 cache line，完全合并）；`buf[ITERS]` 在 d = 4096 时是 2 个 `uint4` = 8 个 32 位寄存器，整个 kernel 大约 30–40 个寄存器，occupancy 不受限；`red[32]` 是 128 字节 shared；`block_reduce_sum` 内部的 sync 都在所有线程都会执行的路径上（没有提前 return）。每行流量：读 8 KiB、写 8 KiB、读 `gamma` 8 KiB（L2 hit）。

### 3. softmax kernel：一行一个 warp，online 版本

```cpp
// softmax_online.cu
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>

// 一行一个 warp；每 lane 持有 ITEMS 个元素在寄存器（ITEMS * 32 >= d）。
// 要求 32 <= d <= ITEMS * 32；一个 block 含 blockDim.x / 32 个 warp，即处理这么多行。
template <int ITEMS>
__global__ void softmax_warp_online_kernel(__nv_bfloat16* __restrict__ out,
                                           const __nv_bfloat16* __restrict__ in,
                                           int rows, int d) {
  const int warps_per_block = blockDim.x >> 5;
  const int row = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= rows) return;                 // 判断对整个 warp 一致，安全

  const __nv_bfloat16* x = in + (size_t)row * d;
  __nv_bfloat16* y = out + (size_t)row * d;

  // 第一遍（寄存器 + 加载）：每 lane 维护 (m, l)
  float v[ITEMS];
  float m = -INFINITY, l = 0.f;
#pragma unroll
  for (int i = 0; i < ITEMS; ++i) {
    const int idx = lane + i * 32;         // 合并：warp 一次读 64 字节连续
    if (idx < d) {
      v[i] = __bfloat162float(x[idx]);
      const float m_new = fmaxf(m, v[i]);
      l = l * __expf(m - m_new) + __expf(v[i] - m_new);
      m = m_new;
    } else {
      v[i] = -INFINITY;                    // 越界元素不参与
    }
  }

  // warp 内合并 32 个 (m, l)：m = max(ma, mb), l = la*e^(ma-m) + lb*e^(mb-m)
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const float m_o = __shfl_xor_sync(0xffffffffu, m, off);
    const float l_o = __shfl_xor_sync(0xffffffffu, l, off);
    const float m_new = fmaxf(m, m_o);
    l = l * __expf(m - m_new) + l_o * __expf(m_o - m_new);
    m = m_new;
  }
  const float inv_l = 1.f / l;

  // 第二遍（寄存器）：写出
#pragma unroll
  for (int i = 0; i < ITEMS; ++i) {
    const int idx = lane + i * 32;
    if (idx < d) y[idx] = __float2bfloat16(__expf(v[i] - m) * inv_l);
  }
}

void softmax_bf16(__nv_bfloat16* out, const __nv_bfloat16* in, int rows, int d,
                  cudaStream_t stream) {
  constexpr int THREADS = 128;                     // 4 个 warp = 4 行 / block
  const int grid = (rows + THREADS / 32 - 1) / (THREADS / 32);
  if      (d <= 1024) softmax_warp_online_kernel<32 ><<<grid, THREADS, 0, stream>>>(out, in, rows, d);
  else if (d <= 2048) softmax_warp_online_kernel<64 ><<<grid, THREADS, 0, stream>>>(out, in, rows, d);
  else if (d <= 4096) softmax_warp_online_kernel<128><<<grid, THREADS, 0, stream>>>(out, in, rows, d);
  // d > 4096：改用 block-per-row 的版本
}
```

几处细节：

- **初值**。$$m = -\infty, l = 0$$。第一个元素到来时 $$m_{\text{new}} = x_0$$，$$l = 0 \cdot e^{-\infty} + e^0 = 1$$，正确。`__expf(-INFINITY)` 返回 0，不是 NaN。
- **越界 lane**。d ≥ 32 保证每个 lane 至少有一个真实元素，所以合并时不会出现两个 $$m = -\infty$$ 相减产生 NaN。如果要支持 $$d < 32$$ 或输入本身含 $$-\infty$$（掩码），合并处要加一个 `m_new == -INFINITY ? 0 : ...` 的守卫。
- **寄存器**。d = 4096 时 `v[128]` 占 128 个寄存器，加上其他约 20 个，接近 150；每 SM 65536 个寄存器，128 线程的 block 最多驻留 $$\lfloor 65536 / (150 \times 128) \rfloor = 3$$ 个 block、12 个 warp——occupancy 只有 19%。这是 warp-per-row 在 d = 4096 时的代价，也是 PyTorch 把 warp softmax 的上限定在 2048 的原因。d ≤ 1024 时 `v[32]` 很轻，occupancy 接近满。
- **访存**。`x[lane + i*32]` 每次 warp 读 64 字节连续，是半条 cache line；下一次迭代读接下来的 64 字节，同一条 line 在 L1 里被用完，HBM 流量仍是每字节一次。改成 `__nv_bfloat162` 一次读两个可以把每次 warp 访问凑成 128 字节，是一个可选的小优化。
- **`__expf` vs `expf`**。`__expf` 是硬件近似指令（SFU 的 `ex2` 加一次乘法），误差约 2 ulp，对 BF16 输出绰绰有余；`expf` 精度更高但要十几条指令。softmax 是 memory-bound 的，两者在带宽上看不出差别，但 d 小、行多时指令数会开始有影响。

### 4. 对照测试与预期量级

用 `torch.utils.cpp_extension.load_inline` 把两个 kernel 接到 Python，然后与 PyTorch 参考实现对照：

```python
import torch

rows, d = 8192, 4096
x = torch.randn(rows, d, device="cuda", dtype=torch.bfloat16) * 3   # 放大方差，让 max 有意义
gamma = torch.randn(d, device="cuda", dtype=torch.bfloat16)

# RMSNorm：参考实现在 FP32 里算，再转回 BF16
ref = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5)
       * gamma.float()).to(torch.bfloat16)
out = my_ext.rmsnorm(x, gamma, 1e-5)
torch.testing.assert_close(out, ref, rtol=1.6e-2, atol=1e-5)

# softmax：online 版本 vs 两遍（PyTorch 的 warp/block softmax 就是三遍寄存器版本）
ref_sm = torch.softmax(x.float(), dim=-1).to(torch.bfloat16)
out_sm = my_ext.softmax(x)
torch.testing.assert_close(out_sm, ref_sm, rtol=1.6e-2, atol=1e-5)

# 在 FP32 上也比一次，确认 online 递推本身没有引入超过 1e-6 量级的误差
x32 = x.float()
torch.testing.assert_close(my_ext.softmax_fp32(x32), torch.softmax(x32, -1),
                           rtol=1e-5, atol=1e-6)
```

BF16 的默认容差 rtol = 1.6e-2 对应约 2 个 BF16 ulp（BF16 尾数 8 位，1 ulp 约 $$2^{-8} \approx 3.9 \times 10^{-3}$$ 的相对误差）。softmax 的输出值很小（平均 $$1/d \approx 2.4 \times 10^{-4}$$），atol = 1e-5 在这个量级上仍是相对容差在起作用，不必调大。online 版本与三遍版本在数学上完全等价，差别只在 FP32 舍入的顺序，FP32 对照能验证这一点。

性能上，先算下界：rows = 8192、d = 4096 的 RMSNorm 读写 128 MiB，A100 上 ≈ 67 µs；softmax 同样形状也是读 8 KiB 写 8 KiB 每行，下界相同。用第二篇的 `bench(fn, warmup=10, iters=100, flush_l2=True)` 计时，**读者跑出的数字大致应该落在：RMSNorm 75–85 µs（带宽的 80–90%），warp softmax 在 d ≤ 2048 时同样 80–90%，d = 4096 时因为 occupancy 下降通常掉到 60–75%**。如果 RMSNorm 落在 150 µs 以上，先查是否向量化（用 Nsight Compute 看 `ld.global.v4` 是否出现）、block 是否太大导致行少时并行度不足；落在 500 µs 以上，几乎一定是第一章列的四个 naive 问题之一。

作为参照，`torch.nn.functional.rms_norm` 和 vLLM 的 `rms_norm` 在同样形状上也处于同一区间——这个 kernel 没有太多花样，做对访存和归约之后，所有人写出来的都差不多。


## 十、小结

这一篇从 memory-bound 的 elementwise 走到了需要线程协作的 reduction。回顾要点：

- **理论下界先行**。RMSNorm d = 4096 BF16、8192 行：读写 128 MiB，A100 ≈ 67 µs。算术强度约 1 FLOP/byte，远低于 ridge point，任何实现都是 memory-bound。
- **shared memory** 是 block 内线程交换数据的唯一场所：静态 `__shared__`、动态 `extern __shared__` + launch 第三参数、超过 48 KB 要 `cudaFuncSetAttribute` opt-in；A100 每 SM 与 L1 共享 192 KB、shared 最多 164 KB；生命周期 = block。
- **`__syncthreads()`** 是 block 级栅栏，必须在所有线程都执行的路径上；每次 sync 都让快的 warp 等慢的 warp。
- **bank conflict**：32 个 bank × 4 字节，bank = (addr/4) mod 32；warp 内不同地址落同一 bank 就串行化，同地址广播不算冲突；按列访问 `tile[t][k]` 是 32-way 冲突，padding 到 33 列或 XOR swizzle 解决。
- **reduction 六版**：从交错寻址到两级 warp shuffle，sync 从 10 次降到 1 次，shared 流量从 10 级树降到 32 个 float。shared memory 解决 block 内 32 个 warp 的汇总，shuffle 消灭最后 5 级同步。
- **warp 原语**：`__shfl_{,up,down,xor}_sync`、`__ballot_sync`、`__any/all_sync`、`__reduce_add_sync`（sm_80+，整数）、cooperative groups 的 `tiled_partition<32>` + `cg::reduce`。
- **原子操作**：同地址串行化 + float 非确定性；block 内用树，跨 block 计数用原子、累加不用原子。
- **一行一个 warp 还是一个 block**：d ≤ 1024 用 warp（寄存器驻留、零 shared、零 sync），d 大用 block；grid 由行数决定。
- **softmax**：减 max 是因为 FP16 在 x > 11.09、BF16/FP32 在 x > 88.7 时 $$e^x$$ 溢出；online softmax 维护 (m, l) 一遍得到统计量，合并公式满足结合律可并行归约，是 FlashAttention 的数学基础。
- **LayerNorm/RMSNorm**：均方与方差在 FP32 累加；LayerNorm 用 Welford 避免 $$E[x^2] - E[x]^2$$ 的相消；fused residual + RMSNorm 把读 3 写 2 变成读 2 写 2。
- **源码**：vLLM `rms_norm_kernel` 是 block-per-row + `cub::BlockReduce` + 16 字节向量化；PyTorch `SoftMax.cu` 按 d ≤ 2048 分派 warp softmax（零 shared）与 block softmax（`BlockReduceSum` 两级 shuffle）；`Reduce.cuh` 加上 global semaphore 构成三级归约。

```text
reduction 六版对比（1024 线程 block）
版本  寻址              sync   shared 流量     消除的问题
v1    交错 + 取模        10     10 级树         —
v2    顺序               10     10 级树         发散 · 取模 · bank conflict
v3    加载时先加         9      9 级树          一半 block · 一级 shared
v4    树到 32 + shfl     5      5 级树          最后 5 级 sync
v5    grid-stride        5      摊薄            每 block 固定开销
v6    shfl + 32 float    1–2    128 字节        几乎全部

bank conflict 速查
bank(addr) = (addr / 4) mod 32；warp 内 stride s 个字 → gcd(s, 32)-way 冲突
stride 1 / 奇数 → 无冲突；stride 2 → 2-way；stride 32 → 32-way；同地址 → 广播
修复：行宽 +1（padding）或 col ^ (row & 31)（swizzle）

online softmax
单步：m' = max(m, x)；l' = l·e^(m − m') + e^(x − m')
合并：m = max(ma, mb)；l = la·e^(ma − m) + lb·e^(mb − m)
safe softmax 阈值：FP16 e^x 溢出于 x > 11.09；BF16/FP32 于 x > 88.7

理论下界（A100，2.0 TB/s）
RMSNorm / softmax，8192 行 × 4096 BF16：128 MiB → ≈ 67 µs
常见可达：RMSNorm 80–90% 带宽；warp softmax d ≤ 2048 时 80–90%
```

reduction 是"先算一个整行的标量，再作用回每个元素"。下一篇的 GEMM 是另一种协作：每个输出元素需要一整行和一整列，shared memory 的角色从"汇总 32 个部分和"变成"暂存被 128 个线程重复读取的 tile"，bank conflict 和 padding/swizzle 会从本篇的一段话变成决定性能的主角。

> **一个 4096×4096 的 BF16 GEMM 理论上只需要 0.44 ms（Tensor Core）或 7 ms（CUDA Core）。naive 实现为什么慢 50 倍？分块把访存量减少了多少？**


## 下一篇

[GEMM：从 naive 到分块](/gemm-from-naive-to-tiled.html)
