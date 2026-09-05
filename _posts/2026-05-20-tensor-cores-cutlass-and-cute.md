---
layout: post
title: "GPU Kernel 工程（06）：Tensor Core、CUTLASS 与 CuTe"
subtitle: "Tensor Cores, CUTLASS and CuTe: Programming the Matrix Units"
tags: [CUDA, Triton, GPU, AI, AI-Infra]
catalog: true
---

> 本文是[《GPU Kernel 工程：从 CUDA 执行模型到 FlashAttention》](/gpu-kernel-engineering.html)系列的第 6 篇（共十篇）。上一篇：[GEMM：从 naive 到分块](/gemm-from-naive-to-tiled.html)　下一篇：[Triton：块级编程与编译器的边界](/triton-block-level-programming.html)

上一篇用 CUDA Core 把 GEMM 的分块结构讲透了。回顾一下那个结构，因为本篇要做的事情就是把它"接"到另一种计算单元上：

- 一个 thread block 负责输出矩阵 $$C$$ 的一个 $$BM \times BN$$ 的 tile，沿 $$K$$ 维以 $$BK$$ 为步长循环；
- 每一步把 $$A$$ 的 $$BM \times BK$$ 子块和 $$B$$ 的 $$BK \times BN$$ 子块搬进 shared memory，全局读取量从 naive 的 $$2MNK$$ 个元素降为 $$MNK \cdot (1/BM + 1/BN)$$，$$BM = BN = 128$$ 时减少 64 倍；
- 每个线程再从 shared memory 把自己需要的一小段 $$A$$、$$B$$ 读进寄存器，算一个 $$TM \times TN$$（如 $$8 \times 8$$）的累加器块，对 shared memory 的读取量再减少 $$TM \cdot TN / (TM + TN) = 4$$ 倍；
- 用 `cp.async` 做多 stage 流水，让下一块 tile 的搬运与当前 tile 的计算重叠。

数据流是三段：**global → shared → 寄存器 → FMA**。这一版在 A100 上通常能达到 cuBLAS FP32 的 70–80%。

但这个"cuBLAS FP32"本身就是个天花板很低的参照。A100 的 FP32 CUDA Core 标称 19.5 TFLOPS，而 BF16 Tensor Core 标称 312 TFLOPS（dense），相差 16 倍。一个 $$4096^3$$ 的 BF16 GEMM 是 $$2 \cdot 4096^3 \approx 137.4$$ GFLOP，在 Tensor Core 上理论时间 $$137.4 / 312 \approx 0.44$$ ms，在 FP32 CUDA Core 上是 $$137.4 / 19.5 \approx 7.0$$ ms。大模型训练和推理的全部矩阵乘法都跑在前者上，所以第五篇的 kernel 无论多精致，离生产环境中的 GEMM 还差一个数量级——差的就是 Tensor Core。

这一篇要回答总纲的核心问题：

> **同样是 128×128 的分块，用 CUDA Core 和用 Tensor Core 写出来的 kernel 结构差在哪里？为什么 Tensor Core 版本必须关心 fragment 布局和 `ldmatrix`？**

答案先说在前面，后文逐步展开：

- **计算单元变了**：从"每线程一个 $$TM \times TN$$ 累加器"变成"每 warp 若干个 mma fragment"。累加器仍在寄存器里，但它属于 warp 而不是线程——每个线程持有的是一个 $$16 \times 8$$ 结果矩阵里由硬件规定的 4 个位置。
- **数据流变了**：从"shared → 寄存器（任意布局）"变成"shared → `ldmatrix` → fragment（固定布局）→ `mma`"。因为 `mma` 指令对 32 个线程各自寄存器里放的是矩阵的哪几个元素有硬性规定，所以你必须关心布局；`ldmatrix` 是硬件提供的"按这个规定从 shared memory 装载"的指令。
- **Hopper 上再进一步**：数据流变成"TMA → shared（swizzled）→ `wgmma` 直接读 shared"，寄存器里只剩累加器，搬运工作从线程手里拿走交给硬件。

文中所有数字为公开标称值或可推导的理论值，实测区间用"通常能达到"的措辞给出。本篇实践部分的 kernel 需要 sm_80（A100 / RTX 30 系及更新），Hopper 部分（`wgmma`、TMA）需要 sm_90，以源码结构阅读为主，明确标注。


## 一、Tensor Core 做什么

### 1. 一条指令，一个小矩阵乘加

CUDA Core 的基本操作是标量 FMA：$$d = a \cdot b + c$$，一条指令、一个线程、一次乘加。Tensor Core 的基本操作是矩阵 FMA：

$$
D_{M \times N} = A_{M \times K} \cdot B_{K \times N} + C_{M \times N}
$$

一条指令、一个 warp（32 个线程协作）、$$M \cdot N \cdot K$$ 次乘加。Ampere 上 BF16 的主力形状是 `m16n8k16`，指令写全是：

```text
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32  D, A, B, C
```

逐段读：`mma.sync` 是 warp 级同步矩阵乘加；`aligned` 要求 warp 内所有线程执行同一条指令；`m16n8k16` 是形状 $$M=16, N=8, K=16$$；`row.col` 表示 $$A$$ 按行主序、$$B$$ 按列主序给出（这两个修饰符对 BF16 是固定的，只有这一种组合）；`.f32.bf16.bf16.f32` 依次是 $$D$$、$$A$$、$$B$$、$$C$$ 的类型——输入 BF16，累加 FP32。

一条 `m16n8k16` 做 $$16 \times 8 \times 16 = 2048$$ 次乘加，即 **4096 FLOP**。相比之下，一个 warp 执行一条 FFMA 是 32 次乘加、64 FLOP。指令数差 64 倍，这就是 Tensor Core 提升算力密度的方式：用更少的指令发射、更少的寄存器读写，换更多的算术。

Ampere 支持的输入类型和对应形状（Tensor Core 路径）：

```text
输入类型      累加类型   mma.sync 形状（Ampere 主力）   备注
FP16          FP32/FP16 m16n8k16                     
BF16          FP32      m16n8k16                     sm_80 起
TF32          FP32      m16n8k8                      FP32 输入自动截断到 19 位，K 减半
INT8          INT32     m16n8k32                     K 翻倍，每字节吞吐翻倍
FP8 (e4m3)    FP32      m16n8k32                     sm_89（Ada）起；Hopper 走 wgmma
```

K 维长度与元素位宽的乘积基本不变（BF16 是 $$16 \times 16 = 256$$ bit，INT8 是 $$32 \times 8 = 256$$ bit）：Tensor Core 每条指令"吃"的字节数恒定，位宽越窄，一条指令做的乘加越多。这就是 INT8 624 TOPS 是 BF16 312 TFLOPS 两倍的硬件来源。

### 2. 每 SM 每周期 1024 次乘加

A100 每个 SM 有 4 个 Tensor Core（每个 warp 调度器一个），每个 Tensor Core 每周期完成 256 次 dense FP16/BF16 乘加，所以每 SM 每周期 1024 次乘加。验算一下标称算力：

$$
108 \text{ SM} \times 1024 \text{ FMA/clk} \times 2 \text{ FLOP/FMA} \times 1.41 \text{ GHz} \approx 312 \text{ TFLOPS}
$$

这个数字对写 kernel 有一个直接的推论：一条 `m16n8k16` 是 2048 次乘加，一个 Tensor Core 每周期 256 次，所以**一条 mma 在 Tensor Core 上占 8 个周期**。要让 Tensor Core 饱和，每个调度器上必须每 8 个周期发出一条 mma。中间那 8 个周期，同一个调度器只能发出大约 8 条其他指令——装载 fragment、算地址、同步、循环控制，全部要挤在这个预算里。CUDA Core 版本的 GEMM 里，每个线程"从 shared memory 读 $$TM + TN$$ 个数、做 $$TM \cdot TN$$ 次 FMA"这种比例，在 Tensor Core 上完全不够看：**装载数据的指令必须极度高效**。这就是 `ldmatrix`（一条指令装满一个 fragment）、`cp.async`（不占寄存器的搬运）以及 Hopper 上 TMA（一条指令搬整个 tile）存在的理由。

### 3. 三代编程接口

从 Volta 到 Hopper，NVIDIA 给 Tensor Core 提供了三层接口：

**`nvcuda::wmma`（CUDA C++ API）**。以 $$16 \times 16 \times 16$$ 的 `fragment` 为单位：

```cpp
#include <mma.h>
#include <cuda_bf16.h>
using namespace nvcuda;

__device__ void wmma_tile(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C,
                          int lda, int ldb, int ldc) {
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
  wmma::fill_fragment(c, 0.0f);
  wmma::load_matrix_sync(a, A, lda);      // warp 内 32 个线程协作装载
  wmma::load_matrix_sync(b, B, ldb);
  wmma::mma_sync(c, a, b, c);             // 编译成一条或多条 mma.sync
  wmma::store_matrix_sync(C, c, ldc, wmma::mem_row_major);
}
```

优点是简单：不需要知道每个线程手里是哪几个元素。缺点也来自这里：fragment 的内部布局是**不透明**的（`fragment::x[i]` 的含义未文档化、随架构变），你不能对累加器做逐元素的自定义操作（比如 FlashAttention 里对 $$S$$ 做 online softmax），也不能用 `ldmatrix` 优化装载。`load_matrix_sync` 从 shared memory 读取的方式由编译器决定，通常会有 bank conflict，性能一般在 cuBLAS 的 50–70%。适合教学和对性能要求不高的地方。

**`mma.sync` PTX 指令（Ampere 主力）**。通过内联 PTX 直接发 `mma.sync.aligned.m16n8k16...`，操作数是普通的 32 位寄存器，每个线程持有哪些元素由 PTX ISA 文档明确规定（下一章详述）。这意味着你可以自己决定怎么把数据装进这些寄存器（`ldmatrix`、普通 `ld.shared`、甚至直接从上一条 mma 的结果转换），也可以对累加器逐元素操作。CUTLASS 2.x、FlashAttention-2 的 Ampere 路径都基于它。代价是需要手工管理 fragment 布局。

**`wgmma.mma_async`（Hopper，sm_90）**。三个根本变化：

- 粒度从 warp 变成 **warpgroup**（4 个连续的 warp，128 个线程）；形状是 `m64nNk16`，$$N$$ 为 8 的倍数、最大 256，一条指令最多 $$64 \times 256 \times 16 = 262144$$ 次乘加，是 `m16n8k16` 的 128 倍；
- 操作数 $$B$$ **必须在 shared memory**（$$A$$ 可在寄存器或 shared memory），通过一个 64 位的 **matrix descriptor** 描述其起始地址、leading dimension byte offset、stride byte offset 和 swizzle 模式，硬件直接从 shared memory 读，不再经过线程寄存器；
- **异步**：`wgmma.mma_async` 发出后立刻返回，用 `wgmma.fence`（保证之前对累加器寄存器 / shared memory 的写对 wgmma 可见）、`wgmma.commit_group`（把之前发出的 wgmma 打成一组）、`wgmma.wait_group N`（等到未完成组数不超过 $$N$$）来管理完成。

伪代码形态如下（需 sm_90，仅展示结构）：

```text
wgmma.fence.sync.aligned;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16
    {d0, d1, ..., d63},        // 每线程 64 个 f32 累加器（64*128 / 128 线程）
    desc_a, desc_b,            // 两个 64 位 descriptor，指向 shared memory
    scale_d, imm_scale_a, imm_scale_b, imm_trans_a, imm_trans_b;
wgmma.commit_group.sync.aligned;
wgmma.wait_group.sync.aligned 0;
```

三代接口的对照放在文末小结表中。本篇实践用 `mma.sync`，因为它是 Ampere 上唯一能拿到接近峰值性能的手段，也是理解 fragment 布局这个核心概念的最佳入口。


## 二、fragment 布局与 ldmatrix

### 1. m16n8k16 的三个 fragment

`mma.sync` 的操作数 A、B、C/D 分别是几个 32 位寄存器，每个线程各持有一份。PTX ISA 规定了 warp 内第 `lane` 个线程的寄存器里放的是矩阵的哪几个元素。先定义两个量：

$$
\text{groupID} = \lfloor \text{lane} / 4 \rfloor \in [0, 8), \qquad t = \text{lane} \bmod 4 \in [0, 4)
$$

也就是 32 个线程被分成 8 组，每组 4 个线程（一个 **quad**）。

**A fragment（$$16 \times 16$$ BF16）**：每线程 4 个 32 位寄存器 `a0..a3`，每个装 2 个相邻的 BF16，共 8 个元素：

```text
寄存器   行            列
a0       groupID       t*2, t*2+1
a1       groupID + 8   t*2, t*2+1
a2       groupID       t*2+8, t*2+9
a3       groupID + 8   t*2+8, t*2+9
```

画成图（每格标出持有该元素的 lane 和寄存器；一格是两个相邻 BF16）：

```text
A (16 x 16 BF16)          列 0-1  2-3  4-5  6-7  | 8-9  10-11 12-13 14-15
行 0  (groupID 0)          T0.a0 T1.a0 T2.a0 T3.a0 | T0.a2 T1.a2 T2.a2 T3.a2
行 1  (groupID 1)          T4.a0 T5.a0 T6.a0 T7.a0 | T4.a2 T5.a2 T6.a2 T7.a2
...
行 7  (groupID 7)         T28.a0 ...              | T28.a2 ...
行 8  (groupID 0)          T0.a1 T1.a1 T2.a1 T3.a1 | T0.a3 T1.a3 T2.a3 T3.a3
...
行 15 (groupID 7)         T28.a1 ...              | T28.a3 ...
```

**B fragment（$$16 \times 8$$，$$K \times N$$，BF16）**：每线程 2 个寄存器：

```text
寄存器   行 (k)            列 (n)
b0       t*2, t*2+1        groupID
b1       t*2+8, t*2+9      groupID
```

```text
B (16 x 8 BF16, K x N)     列 n=0    n=1    ...  n=7
行 k=0-1                    T0.b0    T4.b0  ...  T28.b0
行 k=2-3                    T1.b0    T5.b0  ...  T29.b0
行 k=4-5                    T2.b0    T6.b0  ...  T30.b0
行 k=6-7                    T3.b0    T7.b0  ...  T31.b0
行 k=8-9                    T0.b1    T4.b1  ...  T28.b1
...
行 k=14-15                  T3.b1    T7.b1  ...  T31.b1
```

注意 B 的一个寄存器里是**同一列、相邻两行**的元素——这就是 `.col`（列主序）的含义：B 在寄存器里是按列连续的。

**C/D fragment（$$16 \times 8$$ FP32）**：每线程 4 个 float：

```text
寄存器   行            列
c0       groupID       t*2
c1       groupID       t*2+1
c2       groupID + 8   t*2
c3       groupID + 8   t*2+1
```

```text
C/D (16 x 8 FP32)          列 0     1     2     3     4     5     6     7
行 0  (groupID 0)          T0.c0 T0.c1 T1.c0 T1.c1 T2.c0 T2.c1 T3.c0 T3.c1
行 1  (groupID 1)          T4.c0 T4.c1 T5.c0 T5.c1 ...
...
行 7  (groupID 7)         T28.c0 T28.c1 ...
行 8  (groupID 0)          T0.c2 T0.c3 T1.c2 T1.c3 ...
...
行 15 (groupID 7)         T28.c2 T28.c3 ...
```

一个 warp 64 个寄存器（32 线程 × 2）装下 $$16 \times 16$$ 个 BF16 的 A，正好不多不少；C 是 32 × 4 = 128 个 float，正好 $$16 \times 8$$。fragment 布局的本质是一个**双射**：`(lane, reg) ↔ (row, col)`。

### 2. 为什么布局这么"奇怪"

初看这个布局不像任何自然的"行主序"或"列主序"。它由三个约束决定：

**匹配硬件数据路径**。Tensor Core 从寄存器文件读操作数时，是按 quad（4 个线程）为单位取一行 A（或一列 B）的 16 个元素：4 个线程 × 2 个寄存器 × 2 个 BF16 = 16。`groupID` 选行、`t` 选行内的哪一段，正好让"取 A 的第 $$i$$ 行"变成"读第 $$i$$ 个 quad 的寄存器"。这个布局不是为程序员设计的，是硬件内部数据通路的直接暴露。

**quad 共享一行**。同一 quad 的 4 个线程持有 A 的同一行（以及 C 的同一行）。这对 attention 类 kernel 至关重要：对 $$S = QK^T$$ 的每一行做 softmax 需要行内 max 和 sum，而一行的 8 列（一个 n8 tile）分布在同一 quad 的 4 个线程里，跨多个 n8 tile 的列仍然在这 4 个线程里。行内归约只需要 quad 内两次 `__shfl_xor_sync`（mask 1 和 2），不需要跨 warp、更不需要 shared memory。第八篇的 FlashAttention 实现直接依赖这一点。

**D 的布局能直接作为下一次 mma 的 A**。比较 C/D 与 A 的布局：C 中线程持有 `(groupID, t*2..t*2+1)` 与 `(groupID+8, t*2..t*2+1)`，A 中线程持有 `(groupID, t*2..t*2+1)`、`(groupID+8, ...)`、以及列 +8 的两组。所以**两个相邻 n8 tile 的 C fragment 拼起来（$$16 \times 16$$），逐对把 FP32 转成 BF16x2，就得到一个合法的 A fragment**：

```text
tile0.{c0,c1} → a0      tile0.{c2,c3} → a1
tile1.{c0,c1} → a2      tile1.{c2,c3} → a3
```

FlashAttention 里 $$O = P \cdot V$$ 的 $$P = \text{softmax}(S)$$ 就是上一步 mma 的输出。有了这个性质，$$P$$ 可以留在寄存器里直接作为下一次 mma 的 A 操作数，不需要写回 shared memory 再读出来。这一点是 FlashAttention-2 在 Ampere 上能做到高 Tensor Core 利用率的关键之一。

### 3. ldmatrix：按 fragment 布局从 shared memory 装载

有了布局，问题变成：怎么高效地把 shared memory 里的一个 $$16 \times 16$$ 子块装进 32 个线程的 `a0..a3`？用普通的 `ld.shared.b32` 需要每线程 4 条指令，而且每条指令 32 个线程访问的地址模式（每个 quad 一行、行间跨 BK 个元素）几乎必然 bank conflict。

`ldmatrix` 是专门为此设计的指令：

```text
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {r0, r1, r2, r3}, [addr];
```

语义：一个 warp 协作从 shared memory 载入 **4 个 $$8 \times 8$$ 的 16 位矩阵**（`.x4`；也有 `.x1`、`.x2`）。每个 $$8 \times 8$$ 矩阵占 8 行 × 16 字节。**每个线程提供一行的起始地址**：lane 0–7 提供第 0 个矩阵的 8 行，lane 8–15 提供第 1 个矩阵的 8 行，lane 16–23 第 2 个，lane 24–31 第 3 个。载入后，对第 $$i$$ 个矩阵，线程 `lane` 的 `r_i` 里放的是该矩阵第 `lane/4` 行、第 `(lane%4)*2` 和 `+1` 列的两个元素——**这正好是 mma fragment 里一个寄存器的布局**。

于是 A fragment 的 4 个寄存器对应 $$16 \times 16$$ 子块的四个 $$8 \times 8$$ 象限：`a0` = (行 0–7, 列 0–7)，`a1` = (行 8–15, 列 0–7)，`a2` = (行 0–7, 列 8–15)，`a3` = (行 8–15, 列 8–15)。让 lane 0–7 提供行 0–7、列 0 的地址，lane 8–15 提供行 8–15、列 0，lane 16–23 提供行 0–7、列 8，lane 24–31 提供行 8–15、列 8，一条 `ldmatrix.x4` 就把整个 A fragment 装好了。用公式写，lane 提供的地址是：

$$
\text{row} = \text{lane} \bmod 16, \qquad \text{col} = 8 \cdot \lfloor \text{lane} / 16 \rfloor
$$

B fragment 稍有不同。`mma` 要求 B 是 $$K \times N$$ 列主序，即寄存器里是同一列 $$n$$ 的相邻两个 $$k$$。如果 B 在 shared memory 里按 **$$N \times K$$ 行主序**存放（每行是一个 $$n$$、沿 $$k$$ 连续），那么把它看成"行 = $$n$$、列 = $$k$$"的矩阵，`ldmatrix`（非转置）载入的 $$8 \times 8$$ 块给线程 `lane` 的就是 $$n = \text{lane}/4$$、$$k = (\text{lane}\%4) \cdot 2, +1$$——正是 `b0` 的布局。一条 `.x4` 可以装两个 n8 tile 的 `b0, b1`：lane 0–7 提供 $$n$$ 0–7、$$k$$ 0；lane 8–15 提供 $$n$$ 0–7、$$k$$ 8；lane 16–23 提供 $$n$$ 8–15、$$k$$ 0；lane 24–31 提供 $$n$$ 8–15、$$k$$ 8。

这个"B 按 $$N \times K$$ 行主序存放"的要求看起来别扭，但它正是 PyTorch `nn.Linear` 的权重布局：`weight` 的形状是 `[out_features, in_features]`，即 $$N \times K$$ 行主序，$$y = x W^T$$ 正好是这种 GEMM。CUTLASS 和 cuBLAS 把它叫做 "TN" GEMM（A 行主序、B 列主序），是 Tensor Core 上最自然的一种。

### 4. `.trans` 变体

如果 B 在 shared memory 里是 $$K \times N$$ 行主序（每行是一个 $$k$$、沿 $$n$$ 连续），`ldmatrix` 非转置载入给线程的就是"同一 $$k$$、相邻两个 $$n$$"，与 `b0` 需要的"同一 $$n$$、相邻两个 $$k$$"正好差一个转置。`ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16` 在载入时对每个 $$8 \times 8$$ 块做转置：线程 `lane` 得到的是第 `(lane%4)*2` 和 `+1` 行、第 `lane/4` 列的元素。这样 $$K \times N$$ 行主序的 B 也能一条指令装成 fragment。

第八篇的 attention 就要用到它：$$O = P \cdot V$$ 中 $$V$$ 的存储是 `[seq, d]`，即 $$K \times N$$ 行主序，B fragment 必须用 `.trans` 装载。而 $$S = Q K^T$$ 中 $$K$$ 的存储 `[seq, d]` 恰好是 $$N \times K$$ 行主序，用非转置版本。同一个 kernel 里两种都出现，这是理解 FlashAttention 源码时的一个常见障碍。

### 5. 为什么 ldmatrix 下 shared memory 必须 swizzle

`ldmatrix` 每个阶段（phase）服务 8 个线程，每个线程读 16 字节，共 128 字节——正好是 32 个 bank × 4 字节。要无冲突，这 8 个 16 字节段必须落在互不相同的 8 个"bank 组"（每组 4 个 bank）上。

现在看实际访问模式。以 $$128 \times 32$$ 的 BF16 tile 为例，一行 32 个元素 = 64 字节 = 4 个 16 字节的 chunk。一个 phase 中的 8 个线程读 **8 个连续行的同一个 chunk**：地址是 $$r \cdot 64 + c \cdot 16$$，$$r = 0..7$$。bank 组编号是地址除以 16 再模 8，即 $$(4r + c) \bmod 8$$——$$r$$ 为偶数时都等于 $$c$$，奇数时都等于 $$c + 4$$。8 个访问只落在 2 个 bank 组上，**4 路 conflict**，`ldmatrix` 的吞吐直接掉到四分之一。如果 $$BK = 64$$（一行 128 字节），8 行同一 chunk 全落在同一个 bank 组，8 路 conflict。

CUDA Core 版本的 GEMM 也有这个问题（第五篇通过 padding 或转置存储解决），但在 Tensor Core 版本里它更致命：前面算过，每 8 个周期就要发一条 mma，`ldmatrix` 慢 4 倍意味着 Tensor Core 大部分时间在等数据。

解决办法是 **swizzle**：不改变数据，只改变"逻辑 (row, chunk) → 物理 chunk"的映射，让 8 个连续行的同一逻辑 chunk 落在 8 个不同的物理 bank 组。对 64 字节的行，一个常用的映射是：

$$
\text{chunk}_{\text{phys}} = \text{chunk}_{\text{logic}} \oplus \left( \lfloor r / 2 \rfloor \bmod 4 \right)
$$

验证：物理 bank 组 = $$(4r + \text{chunk}_{\text{phys}}) \bmod 8 = 4 (r \bmod 2) + (c \oplus \lfloor r/2 \rfloor \bmod 4)$$。$$r = 0..7$$ 时，$$r \bmod 2$$ 取 2 个值、$$\lfloor r/2 \rfloor \bmod 4$$ 取 4 个值，组合出 8 个互不相同的 bank 组。无冲突。

因为 `cp.async`（16 字节一次）和 `ldmatrix`（16 字节一行）都以 16 字节 chunk 为粒度访问 shared memory，swizzle 只需要在 chunk 级别做 XOR，chunk 内的 8 个元素保持连续。这也是为什么 padding 这种 CUDA Core 时代的办法在这里不好用：padding 会破坏 16 字节对齐或者浪费 shared memory；XOR swizzle 两者都不影响。

Hopper 的 TMA 硬件支持 32B / 64B / 128B 三种 swizzle 模式，`wgmma` 的 descriptor 也带 swizzle 字段——硬件做地址置换，程序员只需要在两边声明同一种模式。Ampere 上没有这个便利，swizzle 要在 `cp.async` 目标地址和 `ldmatrix` 源地址两处手工算出来，但公式是同一个。


## 三、实践：Ampere BF16 GEMM

### 1. 先算理论

问题：$$C_{M \times N} = A_{M \times K} \cdot B^T$$，A 是 $$M \times K$$ 行主序，B 存为 $$N \times K$$ 行主序（TN GEMM），BF16 输入、FP32 累加、BF16 输出。$$M = N = K = 4096$$。

- FLOPs：$$2 \cdot 4096^3 \approx 137.4$$ GFLOP。
- 最小 HBM 流量：$$3 \times 4096^2 \times 2$$ B = 96 MiB ≈ 100.7 MB。
- 算术强度 $$137.4 \text{G} / 100.7 \text{M} \approx 1365$$ FLOP/byte，远超 A100 BF16 ridge point 156，compute-bound。
- 理论下界：$$137.4 / 312 \approx 0.44$$ ms。

block tile $$128 \times 128 \times 32$$。每个 k-tile 从全局搬入 $$(128 + 128) \times 32 \times 2$$ B = 16 KiB，做 $$2 \times 128 \times 128 \times 32 = 1{,}048{,}576$$ FLOP，即每字节 32 次乘加（64 FLOP）；A100 上一个 SM 每周期 2048 FLOP，需要 32 B/clk 的 L2 → shared 带宽，L2 可以提供。

再算 **shared memory 带宽**，这在 Tensor Core 版本里是新的瓶颈。8 个 warp 排成 $$2 \times 4$$，每个 warp 负责 $$64 \times 32$$ 的输出，即 $$4 \times 4 = 16$$ 个 `m16n8` tile，每个 k16 步做 16 条 mma（65536 FLOP）。每步要装载 A 的 $$64 \times 16$$（2 KiB）和 B 的 $$32 \times 16$$（1 KiB），共 3 KiB。算术强度 $$65536 / 3072 \approx 21.3$$ FLOP/byte。SM 要跑满 2048 FLOP/clk 需要 $$2048 / 21.3 \approx 96$$ B/clk 的 shared memory 带宽，而 A100 每 SM 每周期 shared 带宽是 128 字节，占用 75%。够，但不宽裕——这解释了为什么 cuBLAS/CUTLASS 常用 $$64 \times 64$$ 的 warp tile（4 KiB 装载做 131072 FLOP，32 FLOP/byte，只需 64 B/clk），代价是每线程 128 个累加器寄存器。本篇选 $$64 \times 32$$ 是为了每线程 64 个累加器，寄存器压力小、代码更易读。

### 2. 设计

```text
grid    = (N/128, M/128)，每 block 256 线程 = 8 warps
warp 排布 2 (M) x 4 (N)，warp tile 64 x 32
每 warp 每 k16 步：4 条 ldmatrix.x4 (A) + 2 条 ldmatrix.x4 (B) + 16 条 mma
shared memory：3 stage x (A 128x32 + B 128x32) BF16 = 3 x 16 KiB = 48 KiB（静态上限）
swizzle：chunk ^= (row >> 1) & 3
流水：cp.async 3 stage，prologue 预取 2 个 tile；每轮 wait_group 1 + __syncthreads 后预取第 kt+2 个 tile
寄存器：64 累加器 + 16 (A frag) + 8 (B frag) + 地址 ≈ 100–128/线程
```

为聚焦 Tensor Core 部分，本版假设 $$M, N$$ 是 128 的倍数、$$K$$ 是 32 的倍数（边界处理与第五篇相同：用 `cp.async` 的 src-size 操作数做零填充，或对残余 tile 走标量路径）。

### 3. 代码

```cpp
// bf16_gemm_mma.cu   nvcc -arch=sm_80 -O3
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int BM = 128, BN = 128, BK = 32;
constexpr int STAGES  = 3;
constexpr int THREADS = 256;                 // 8 warps
constexpr int WARPS_M = 2, WARPS_N = 4;
constexpr int WM = BM / WARPS_M;             // 64
constexpr int WN = BN / WARPS_N;             // 32
constexpr int MT = WM / 16;                  // 4 个 m16 tile
constexpr int NT = WN / 8;                   // 4 个 n8 tile
constexpr int TILE_ELEMS = BM * BK;          // 128 x 32 BF16 = 8 KiB
static_assert(BM == BN, "A/B tile share one smem layout");
static_assert(NT % 2 == 0, "B ldmatrix.x4 loads two n8 tiles at once");

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// tile 为 [128 行][32 BF16]，一行 64 B = 4 个 16 B chunk。
// 逻辑 (row, col) -> 物理元素偏移，chunk 索引与 (row>>1)&3 异或。
__device__ __forceinline__ int swz(int row, int col) {
  int chunk = (col >> 3) ^ ((row >> 1) & 3);
  return row * BK + (chunk << 3) + (col & 7);
}

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :: "r"(smem_u32(smem)), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1,
                                            uint32_t& r2, uint32_t& r3,
                                            const void* smem) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 { %0, %1, %2, %3 }, [%4];\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(smem_u32(smem)));
}

// D = A * B + D，A: 4 x b32 (16x16 BF16)，B: 2 x b32 (16x8 BF16)，D: 4 x f32
__device__ __forceinline__ void mma_bf16_16816(float* d, const uint32_t* a,
                                               const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{ %0, %1, %2, %3 }, { %4, %5, %6, %7 }, { %8, %9 }, { %0, %1, %2, %3 };\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// 把全局内存中一个 128 x 32 的 BF16 子块（行主序，leading dim = ld）
// 以 cp.async 装入 swizzled shared memory。512 个 16 B chunk，256 线程各 2 个。
__device__ __forceinline__ void load_tile(__nv_bfloat16* smem,
                                          const __nv_bfloat16* g, int ld, int tid) {
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int c   = tid + i * THREADS;     // 0..511
    int row = c >> 2;                // 每行 4 个 chunk
    int col = (c & 3) << 3;
    cp_async_16(smem + swz(row, col), g + (size_t)row * ld + col);
  }
}

}  // namespace

// C[M][N] = A[M][K] * B[N][K]^T，全部行主序（"TN" GEMM，对应 nn.Linear 的 x @ W^T）
// 要求 M % 128 == 0, N % 128 == 0, K % 32 == 0
__global__ void __launch_bounds__(THREADS)
bf16_gemm_tn_kernel(const __nv_bfloat16* __restrict__ A,
                    const __nv_bfloat16* __restrict__ B,
                    __nv_bfloat16* __restrict__ C, int M, int N, int K) {
  __shared__ __align__(128) __nv_bfloat16 smem[STAGES][2][TILE_ELEMS];  // 48 KiB

  const int tid    = threadIdx.x;
  const int lane   = tid & 31;
  const int warp   = tid >> 5;
  const int warp_m = warp % WARPS_M;   // 0..1
  const int warp_n = warp / WARPS_M;   // 0..3

  const int bm = blockIdx.y * BM;
  const int bn = blockIdx.x * BN;
  const __nv_bfloat16* A_blk = A + (size_t)bm * K;
  const __nv_bfloat16* B_blk = B + (size_t)bn * K;

  float acc[MT][NT][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.f;

  const int KT = K / BK;

  // prologue：预取前 STAGES-1 个 k-tile，每个 tile 一个 commit group
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < KT) {
      load_tile(smem[s][0], A_blk + s * BK, K, tid);
      load_tile(smem[s][1], B_blk + s * BK, K, tid);
    }
    cp_async_commit();
  }

  // 每个 lane 在 ldmatrix.x4 中提供的行/列（相对 16x16 子块）
  const int a_row = lane & 15;                        // A: 行 0..15
  const int a_col = (lane >> 4) << 3;                 // A: 列 0 或 8
  const int b_row = (lane & 7) + ((lane >> 4) << 3);  // B: n 0..7 或 8..15
  const int b_col = ((lane >> 3) & 1) << 3;           // B: k 0 或 8

  for (int kt = 0; kt < KT; ++kt) {
    cp_async_wait<STAGES - 2>();   // 第 kt 个 tile 已到达（本线程的部分）
    __syncthreads();               // 所有线程的部分都到达；且 kt-1 的 stage 已被读完

    {  // 预取第 kt+STAGES-1 个 tile，写入 kt-1 刚用完的 stage
      int nk = kt + STAGES - 1;
      if (nk < KT) {
        int s = nk % STAGES;
        load_tile(smem[s][0], A_blk + nk * BK, K, tid);
        load_tile(smem[s][1], B_blk + nk * BK, K, tid);
      }
      cp_async_commit();           // 空 group 也 commit，保持计数一致
    }

    const __nv_bfloat16* sA = smem[kt % STAGES][0];
    const __nv_bfloat16* sB = smem[kt % STAGES][1];

#pragma unroll
    for (int ks = 0; ks < BK; ks += 16) {
      uint32_t af[MT][4];
      uint32_t bf[NT][2];
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        int row = warp_m * WM + i * 16 + a_row;
        ldmatrix_x4(af[i][0], af[i][1], af[i][2], af[i][3],
                    sA + swz(row, ks + a_col));
      }
#pragma unroll
      for (int j = 0; j < NT; j += 2) {
        int row = warp_n * WN + j * 8 + b_row;
        ldmatrix_x4(bf[j][0], bf[j][1], bf[j + 1][0], bf[j + 1][1],
                    sB + swz(row, ks + b_col));
      }
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
          mma_bf16_16816(acc[i][j], af[i], bf[j]);
    }
  }

  // epilogue：按 C fragment 布局写回，每线程每 tile 两个 bf16x2（4 B）store
  const int g = lane >> 2;
  const int t = lane & 3;
#pragma unroll
  for (int i = 0; i < MT; ++i) {
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      int row = bm + warp_m * WM + i * 16 + g;
      int col = bn + warp_n * WN + j * 8 + t * 2;
      __nv_bfloat162 v0 = __floats2bfloat162_rn(acc[i][j][0], acc[i][j][1]);
      __nv_bfloat162 v1 = __floats2bfloat162_rn(acc[i][j][2], acc[i][j][3]);
      *reinterpret_cast<__nv_bfloat162*>(C + (size_t)row * N + col)       = v0;
      *reinterpret_cast<__nv_bfloat162*>(C + (size_t)(row + 8) * N + col) = v1;
    }
  }
}

void bf16_gemm_tn(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
                  int M, int N, int K, cudaStream_t stream) {
  dim3 grid(N / BN, M / BM);
  bf16_gemm_tn_kernel<<<grid, THREADS, 0, stream>>>(A, B, C, M, N, K);
}
```

几处需要逐行核对的地方：

- **内联 PTX 的约束**：`ldmatrix` 输出 4 个 `"=r"`（32 位无符号），地址是 `"r"`（shared memory 的 32 位地址，由 `__cvta_generic_to_shared` 得到）；`mma` 的累加器用 `"+f"`（既读又写的 float），A 4 个、B 2 个 `"r"`，共 10 个操作数，`%0`–`%9` 与 PTX 中 `D, A, B, C` 的顺序一致，C 与 D 复用同一组寄存器。`cp.async` 的全局地址用 `"l"`（64 位）。
- **`ldmatrix` 的 lane → 地址映射**：A 用 `a_row = lane & 15`、`a_col = (lane >> 4) * 8`，得到 lane 0–7 → (行 0–7, 列 0)、8–15 → (行 8–15, 列 0)、16–23 → (行 0–7, 列 8)、24–31 → (行 8–15, 列 8)，输出 `r0..r3` 依次是 `a0..a3`。B 用 `b_row = (lane & 7) + (lane >> 4) * 8`、`b_col = ((lane >> 3) & 1) * 8`，lane 0–7 → ($$n$$ 0–7, $$k$$ 0) 给第 $$j$$ 个 n8 tile 的 `b0`，8–15 → ($$n$$ 0–7, $$k$$ 8) 给它的 `b1`，16–23 与 24–31 给第 $$j+1$$ 个 tile。
- **对齐**：`ldmatrix` 每行地址必须 16 字节对齐。`swz()` 的输出在 `col` 是 8 的倍数时是 8 个元素（16 字节）的倍数，`smem` 数组以 128 字节对齐，每个 stage/操作数子数组是 16 KiB 的倍数。
- **流水的正确性**：进入第 `kt` 轮之前已经 commit 了 $$(STAGES-1) + kt$$ 个 group；`wait_group STAGES-2` 保证最多 1 个 group 未完成，即前 $$kt+1$$ 个 group（tile 0..kt）都已完成。随后的 `__syncthreads()` 同时保证（a）所有线程的 `cp.async` 都完成、tile `kt` 对全 block 可见；（b）所有线程都已离开第 `kt-1` 轮的计算，stage `(kt-1) % STAGES` 可以被覆盖——而 `nk % STAGES = (kt + 2) % 3 = (kt - 1) % 3` 正是这个 stage。
- **shared memory**：$$3 \times 2 \times 128 \times 32 \times 2$$ B = 49152 B，恰好是不 opt-in 的 48 KiB 静态上限。想用 4 stage 需要改成 `extern __shared__` 加 `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, 65536)`。
- **占用率**：每线程约 128 个寄存器 × 256 线程 = 32768，每 SM 可驻留 2 个 block（寄存器）；shared memory 48 KiB × 2 = 96 KiB，也允许 2 个。2 个 block = 16 个 warp，每个调度器 4 个 warp，足以隐藏 mma 与 ldmatrix 的延迟。

### 4. 测试

用 `load_inline` 接到 PyTorch，与 `torch.matmul` 对照（B 传入 `W` 本身即 $$N \times K$$，参考值是 `x @ W.T`）：

```python
import torch
from torch.utils.cpp_extension import load_inline

cpp_src = "void bf16_gemm_tn(const at::Tensor&, const at::Tensor&, at::Tensor&);"
cuda_src = open("bf16_gemm_mma.cu").read() + r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
void bf16_gemm_tn(const at::Tensor& A, const at::Tensor& B, at::Tensor& C) {
  int M = A.size(0), K = A.size(1), N = B.size(0);
  TORCH_CHECK(M % 128 == 0 && N % 128 == 0 && K % 32 == 0);
  bf16_gemm_tn(reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
               reinterpret_cast<const __nv_bfloat16*>(B.data_ptr()),
               reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
               M, N, K, at::cuda::getCurrentCUDAStream());
}
"""
mod = load_inline("bf16_gemm_mma", cpp_sources=cpp_src, cuda_sources=cuda_src,
                  functions=["bf16_gemm_tn"], extra_cuda_cflags=["-arch=sm_80", "-O3"])

M = N = K = 4096
A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
mod.bf16_gemm_tn(A, W, C)
ref = A @ W.T
torch.testing.assert_close(C, ref, rtol=1.6e-2, atol=1e-2)

ms = bench(lambda: mod.bf16_gemm_tn(A, W, C))        # 第二篇的 bench()
ms_ref = bench(lambda: torch.matmul(A, W.T))
flops = 2 * M * N * K
print(f"ours {flops / ms / 1e9:.0f} GFLOPS, cuBLAS {flops / ms_ref / 1e9:.0f} GFLOPS")
```

`atol` 从默认的 1e-5 调大到 1e-2：$$K = 4096$$ 项 FP32 累加、再舍入到 BF16（8 位尾数），我们的累加顺序与 cuBLAS 不同，绝对误差在 $$10^{-2}$$ 量级是正常的；`rtol` 保持 1.6e-2。

### 5. 预期与差距

按前面的分析，这个 kernel 在 A100 上的性能读者跑出来大致应落在 cuBLAS BF16 的 **80–90%** 区间，即 200–260 TFLOPS 量级（cuBLAS 本身在 $$4096^3$$ 上通常能达到标称 312 的 85–95%）。差距来自：

- **warp tile 偏小**：$$64 \times 32$$ 的 shared memory 算术强度 21.3 FLOP/byte，占用 75% 的 shared 带宽；cuBLAS 用 $$64 \times 64$$ 或更大。
- **只有 3 stage**：48 KiB 的静态上限限制了流水深度；opt-in 到 4–5 stage 能更好地覆盖 L2 miss 的延迟。
- **没有 tile 调度优化**：`grid = (32, 32)` 的 1024 个 block 按 `blockIdx.x` 顺序分派，L2 中 A/B tile 的复用不如 swizzled 或 persistent 的 tile 顺序（cuBLAS 与 CUTLASS 的 `ThreadblockSwizzle` 做的正是这个），第十篇会再回到这个问题。
- **epilogue 未向量化**：每个线程写 4 字节，一个 quad 写 16 字节连续，整个 warp 的 store 覆盖 8 行——按 32 字节 sector 算效率只有 50%。cuBLAS 通常经 shared memory 转置后以 128 位 store 写出。

如果读者想再往上推，优先级依次是：换 $$64 \times 64$$ warp tile（block tile 128×256 或 256×128）、开 4–5 stage 的动态 shared memory、把 `ldmatrix` 与 `mma` 做软件流水（在做第 $$ks$$ 步 mma 的同时发出第 $$ks+1$$ 步的 ldmatrix，即"寄存器双缓冲"）。这三项加起来通常能把差距压到 5–10% 以内，也就是 CUTLASS 2.x 的水平。


## 四、Hopper：TMA、wgmma 与 warp specialization

以下内容需要 sm_90（H100），本机无法运行，以指令语义和 kernel 结构为主。H100 BF16 Tensor Core 标称约 989 TFLOPS，是 A100 的 3.2 倍，但 shared memory 带宽（每 SM 每周期 128 字节）没有同比增长，SM 数只从 108 增至 132。这意味着 Ampere 那套"每个线程发 `cp.async`、每个 warp 发 `ldmatrix` 再发 `mma`"的结构在 Hopper 上无法喂饱 Tensor Core：**指令发射带宽和 shared memory 带宽都不够**。Hopper 的三项新机制都是针对这一点。

### 1. TMA：一条指令搬一个多维 tile

Ampere 的 `cp.async` 是每线程 16 字节；搬一个 $$128 \times 64$$ 的 BF16 tile（16 KiB）要 1024 条指令、每线程算 4 次地址。Hopper 的 **Tensor Memory Accelerator（TMA）** 把这件事变成一条指令：

```text
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
    [smem_dst], [tensor_map, {x, y}], [mbarrier];
```

- `tensor_map` 是一个 128 字节的 `CUtensorMap` 对象，在 host 端用 driver API `cuTensorMapEncodeTiled` 创建：给出全局张量的基址、每维大小与 stride、**box 大小**（一次搬多大的 tile，如 $$64 \times 128$$）、元素类型、**swizzle 模式**（`CU_TENSOR_MAP_SWIZZLE_128B` 等）、越界填充方式。它以 `const __grid_constant__ CUtensorMap` 形参传给 kernel；
- `{x, y}` 是 tile 在全局张量中的多维坐标（元素单位），硬件负责算地址、处理边界（越界部分填零）、做 swizzle；
- 完成通知通过 **mbarrier**：发起前用 `mbarrier.arrive.expect_tx` 告知 barrier "期待 N 个字节到达"，TMA 每写完一段就把 `tx-count` 减掉相应字节数，消费者用 `mbarrier.try_wait.parity` 等到 barrier 翻转；
- **单线程发起**：整个 block 只需一个线程发出这条指令，其他线程完全空闲。

对比 Ampere：256 个线程各发 4 条 `cp.async` + 算地址，变成 1 个线程发 2 条指令。释放出来的不只是指令槽，还有寄存器（不需要每线程保存全局指针和偏移）。

### 2. wgmma：从 shared memory 直接读操作数

有了 TMA 把 tile 以 swizzled 布局放进 shared memory，`wgmma.mma_async` 直接用 descriptor 读它：descriptor 编码了 shared memory 起始地址（14 位，16 字节单位）、leading dimension byte offset、stride byte offset、以及与 TMA 一致的 swizzle 模式。$$B$$ 必须来自 shared memory，$$A$$ 也可以。于是 Hopper 上主循环的数据流变成：

```text
Ampere:  global --cp.async--> shared --ldmatrix--> 寄存器 fragment --mma.sync--> 寄存器累加器
Hopper:  global ----TMA-----> shared (swizzled) ---------------wgmma----------> 寄存器累加器
```

寄存器里只剩累加器。一条 `m64n256k16` 是 262144 次乘加；H100 每 SM 每周期约 2048 次 dense BF16 乘加（4 个第四代 Tensor Core 各 512），一条指令占约 128 个周期。异步语义在这里是必需的：发出 wgmma 后，warpgroup 可以立刻去处理下一个 tile 的 barrier 等待、或者做 epilogue，而不是原地等 128 个周期。

### 3. warp specialization：生产者与消费者

Ampere kernel 中每个 warp 既搬数据又算矩阵，两种工作交织在同一条指令流里。Hopper 上 CUTLASS 3.x 采用 **warp specialization**：把 block 内的 warp 按角色分组。

- **Producer warpgroup**（4 个 warp，实际只有 1 个线程干活）：循环地等待某个 stage 变空、发出 TMA 装载 A 和 B 到该 stage、`arrive.expect_tx`；
- **Consumer warpgroup(s)**（1 或 2 个，各 4 个 warp）：循环地等待某个 stage 装满、对它发 `wgmma`、算完后通知该 stage 已空。

同步用两组 mbarrier，每个 stage 一对：`full_barrier[s]`（producer 到 consumer："数据到了"）和 `empty_barrier[s]`（consumer 到 producer："我读完了，可以覆盖"）。伪代码：

```text
// 共享：STAGES 个 stage，每个有 full[s]、empty[s] 两个 mbarrier
if (warpgroup == producer) {
  setmaxnreg.dec 40;                       // 让出寄存器给 consumer
  if (elect_one()) {
    for (kt = 0; kt < KT; ++kt) {
      s = kt % STAGES;
      mbarrier.wait(empty[s], phase);      // 等 consumer 释放
      mbarrier.arrive.expect_tx(full[s], bytes_A + bytes_B);
      tma_load(smem_A[s], tmap_A, {kt*BK, bm}, full[s]);
      tma_load(smem_B[s], tmap_B, {kt*BK, bn}, full[s]);
    }
  }
} else {                                   // consumer warpgroup(s)
  setmaxnreg.inc 232;
  for (kt = 0; kt < KT; ++kt) {
    s = kt % STAGES;
    mbarrier.wait(full[s], phase);         // 等 TMA 写完
    wgmma.fence;
    for (k16 in stage s) wgmma.mma_async(acc, desc_A[s][k16], desc_B[s][k16]);
    wgmma.commit_group;
    wgmma.wait_group 1;                    // 保留一组在飞，形成 wgmma 级流水
    mbarrier.arrive(empty[(kt - 1) % STAGES]);   // 上一 stage 的 wgmma 已完成，释放
  }
  wgmma.wait_group 0;
  epilogue(acc);
}
```

`setmaxnreg` 是这套结构里容易被忽略但很关键的一条指令：一个 block 的寄存器预算按线程数平均分配（384 线程时每线程最多 168 个），但 producer 只需要几十个寄存器，consumer 却要装 $$64 \times 256$$ 累加器的一部分（每线程 128 个 float）加地址。`setmaxnreg.dec`/`.inc` 允许 warpgroup 在运行时把寄存器"让"给同一 block 的其他 warpgroup——上面的 40 / 232 / 232 是 CUTLASS 中 1 producer + 2 consumer 配置的典型值，$$(40 + 232 + 232) \times 128 = 64512 \le 65536$$。

两个 consumer warpgroup 有两种调度：

- **Cooperative**：两个 consumer 合作算同一个 output tile（如 $$256 \times 128$$，各算 128 行），同时进入主循环、同时做 epilogue。tile 大、shared memory 算术强度高，适合大 M、N。
- **Ping-pong**：两个 consumer 各算一个不同的 $$128 \times N$$ tile，通过一个额外的 ordered barrier 错开：一个在跑 mainloop 时另一个在做 epilogue，然后交换。这样 Tensor Core 在 epilogue 期间也不空闲。适合 epilogue 相对较重（如带量化 scale 的输出）的场景。

vLLM 的 FP8 GEMM 配置里两种都出现，下一章会看到。


## 五、CUTLASS 与 CuTe

手写 `mma.sync` kernel 之后再看 CUTLASS，会发现它做的事情与上一章完全对应，只是把每个决定（tile 形状、warp 排布、fragment 布局、swizzle、流水深度）变成了模板参数。

### 1. 2.x 与 3.x

CUTLASS 2.x（Volta 到 Ampere）的核心抽象是 `ThreadblockShape / WarpShape / InstructionShape` 三层 tile 加一个 `Stages`，对应上一章的 block tile 128×128×32、warp tile 64×32、mma 形状 16×8×16、3 stage。它的线程 ↔ 数据映射由一堆 `ThreadMap`、`TileIterator` 类手工实现，每种布局、每种数据类型一套。

CUTLASS 3.x（2023 年起）以 **CuTe** 和 **Hopper** 为中心重写：所有"哪个线程持有哪个元素"的问题统一用 CuTe 的 `Layout` 代数表达，kernel 结构围绕 TMA + warp specialization 设计，Ampere 通过同一套接口向下兼容。理解 3.x 的关键是先理解 CuTe，再理解分层。

### 2. 3.x 的分层

从上到下：

```text
device::GemmUniversalAdapter<Kernel>      host 侧入口：参数检查、workspace、launch
  └─ kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue, TileScheduler>
       __global__ 函数体：取 tile 坐标、调用 mainloop 与 epilogue；warp specialization 在这一层
       ├─ collective::CollectiveMma<DispatchPolicy, TileShape, ...>     一个 block 的主循环
       │    （由 CollectiveBuilder 根据 arch/schedule 自动选择实现）
       │    ├─ TiledMma        把 MMA_Atom（一条 mma.sync 或 wgmma）平铺成 warp/warpgroup tile
       │    └─ TiledCopy       把 Copy_Atom（cp.async / ldmatrix / TMA）平铺成 block 级拷贝
       └─ collective::CollectiveEpilogue<...>                             写回 + 融合算子（EVT）
            └─ TiledCopy / fusion::Sm90EVT
CuTe：Layout / Tensor / Shape / Stride / local_tile / local_partition —— 所有层共用的坐标代数
```

对应到上一章的手写 kernel：`bf16_gemm_tn()` 是 device 层；`bf16_gemm_tn_kernel` 的 `blockIdx` 解析和 epilogue 是 kernel 层；k 循环加流水是 `CollectiveMma`；`MT × NT` 个 `mma_bf16_16816` 加 6 条 `ldmatrix_x4` 是 `TiledMma` 与 `TiledCopy`；`swz()` 和 `a_row/b_row` 这些索引公式就是 CuTe 要替我们表达的东西。

### 3. CuTe：Layout 是从坐标到偏移的函数

CuTe 的核心只有一个概念：

$$
\text{Layout} = (\text{Shape}, \text{Stride}), \qquad \text{offset}(c) = \sum_i c_i \cdot s_i
$$

`Shape` 给出每一维的大小，`Stride` 给出每一维走一步跨多少个元素，Layout 把一个多维坐标映射成一个一维偏移。这和 PyTorch 的 `size/stride` 是同一个概念，但有两点关键扩展：

**层次化 shape**。shape 的每一维本身可以是一个 shape。`Shape<Shape<_2,_4>,_8>` 是一个"逻辑上二维、其中第一维内部又分成 2×4"的形状，配合 `Stride<Stride<_1,_2>,_8>`，坐标 $$((i, j), k)$$ 映射到 $$i + 2j + 8k$$。层次化让"一个 128×128 的 tile 被分成 2×4 个 warp tile、每个 warp tile 又被分成 32 个线程的份"这种嵌套结构可以写成一个 Layout。

**编译期整数**。`_128` 是 `Int<128>` 的别名，是类型而非值，所以 Layout 的绝大部分运算在编译期完成，生成的索引代码没有运行时开销。

基本操作（都在 `cute/` 头文件中，大致位于 CUTLASS 仓库的 `include/cute/` 目录）：

- `make_layout(shape, stride)`：构造；只给 shape 时默认列主序（第一维 stride 为 1）。`Layout<Shape<_128,_128>, Stride<_1,_128>>` 是 128×128 的列主序，`Stride<_128,_1>` 是行主序。
- `make_tensor(ptr, layout)`：Layout 加一个指针就是 Tensor。
- `local_tile(tensor, tile_shape, coord)`：把 tensor 按 `tile_shape` 切块，取第 `coord` 块。这就是"block tile"——上一章的 `A_blk = A + bm * K` 加上 128×32 的形状。
- `local_partition(tensor, thread_layout, thread_idx)`：把 tensor 按 `thread_layout` **交错**划分给线程，取第 `thread_idx` 个线程的那一份。与 `local_tile` 的区别：tile 是把相邻元素分给同一个人（分块），partition 是把相隔 stride 的元素分给同一个人（交错）。
- `partition_S(tensor)` / `partition_D(tensor)`（`ThrCopy` 上的方法）：按一个 `TiledCopy` 的源/目标布局划分给当前线程；`partition_A/B/C`（`ThrMMA` 上）按一个 `TiledMma` 的 fragment 布局划分。
- `print(layout)` / `print_latex(layout)`：打印；后者对二维 layout 输出一张 LaTeX 表，每格标注持有它的 (thread, value)，是理解复杂布局的最好工具。

fragment 布局在 CuTe 里就是一个 Layout。`m16n8k16` BF16 的 MMA_Atom 是 `SM80_16x8x16_F32BF16BF16F32_TN`，它的 A 操作数布局（`mma_traits_sm80.hpp` 中）写成从 `(thread, value)` 到 $$16 \times 16$$ 矩阵（列主序，偏移 $$m + 16k$$）的映射：

```text
ALayout = Layout<Shape <Shape <_4, _8>, Shape <_2, _2, _2>>,
                 Stride<Stride<_32, _1>, Stride<_16, _8, _128>>>
```

读法：thread 坐标 $$(t, g)$$，$$t = \text{lane} \bmod 4$$ 的 stride 是 32 = $$16 \times 2$$，即 $$k = 2t$$；$$g = \text{lane}/4$$ 的 stride 是 1，即 $$m = g$$。value 坐标 $$(v_0, v_1, v_2)$$：$$v_0$$ stride 16 是 $$k + 1$$（同一寄存器里的第二个 BF16），$$v_1$$ stride 8 是 $$m + 8$$（`a1`），$$v_2$$ stride 128 是 $$k + 8$$（`a2`）。这正是第二章那张 A fragment 图，只是变成了一个可以被 `print_latex` 打印、可以和其他 Layout 组合的代数对象。

### 4. 一个 CuTe 小程序

下面用 CuTe 复现上一章的划分：128×128 的 tile → 8 个 warp（2×4，每个 64×32）→ 每个 warp 内 32 个线程。只用到 CuTe 头文件，可以在 host 上编译运行（`nvcc -std=c++17 -I<cutlass>/include cute_demo.cu`）：

```cpp
// cute_demo.cu
#include <cstdio>
#include <vector>
#include <cute/tensor.hpp>
using namespace cute;

int main() {
  // 128 x 128 的 tile，列主序：(m, n) -> m + 128 n
  Layout<Shape<_128, _128>, Stride<_1, _128>> tile_layout{};
  print("tile   : "); print(tile_layout); print("\n");
  // 输出 (_128,_128):(_1,_128)

  // 用 8 个 warp 做 2 x 4 的分块，每个 warp 拿 64 x 32
  auto warp_tiler = make_shape(Int<64>{}, Int<32>{});
  auto tiled = zipped_divide(tile_layout, warp_tiler);
  print("divided: "); print(tiled); print("\n");
  // 输出 ((_64,_32),(_2,_4)):((_1,_128),(_64,_4096))
  //   第一部分是 warp tile 内部坐标，第二部分是 "第几个 warp tile"

  std::vector<float> buf(128 * 128);
  Tensor T = make_tensor(buf.data(), tile_layout);

  int warp_id = 5;                                        // 与 kernel 中 warp_m/warp_n 一致
  auto warp_coord = make_coord(warp_id % 2, warp_id / 2); // (warp_m, warp_n) = (1, 2)
  Tensor wT = local_tile(T, warp_tiler, warp_coord);
  print("warp 5 : "); print(wT.layout()); print("\n");
  // 输出 (_64,_32):(_1,_128)，起始偏移 = 1*64 + 2*32*128 = 8256

  // warp tile 再按 8 x 4 的线程网格交错划分给 32 个 lane
  auto thr_layout = make_layout(make_shape(Int<8>{}, Int<4>{}));  // 列主序 8x4
  Tensor tT = local_partition(wT, thr_layout, /*lane=*/3);
  print("lane 3 : "); print(tT.layout()); print("\n");
  // 输出 (_8,_8):(_8,_512)：lane 3 持有 64x32 里每隔 8 行、每隔 4 列的 8x8 个元素

  print_latex(thr_layout);   // 打印 8x4 线程网格的 LaTeX 表格
  return 0;
}
```

`zipped_divide(tile_layout, warp_tiler)` 输出 `((_64,_32),(_2,_4)):((_1,_128),(_64,_4096))`——它的意思是：这个 128×128 的 layout 被重新表达为"warp tile 内坐标 × warp 坐标"两级，第二级的 stride `(64, 4096)` 说明沿 M 移到下一个 warp tile 跨 64 个元素、沿 N 跨 $$32 \times 128 = 4096$$ 个元素。上一章 kernel 里 `warp_m * WM + i * 16 + a_row` 这类手算的索引，在这里被 Layout 代数替代，而且是可组合的：把 `tiled` 与一个 `TiledMma` 的 fragment layout 再 `compose`，就得到"warp 5 的 lane 3 的第 k 个 mma 的 `a2` 寄存器对应全局矩阵的哪个元素"。

这就是 CuTe 成为 FlashAttention-3 和大量新 kernel 基础的原因：attention kernel 里 Q、K、V、S、P、O 六个张量，每个都有自己的 tile 形状、shared memory swizzle 和 fragment 布局，其中 P 还要从 C fragment 转成 A fragment，V 要用 `.trans` 装载。手算每一处索引既容易错也无法复用；用 Layout 代数写，改一个 tile 形状或换一种 mma 指令只需改一个类型参数。

### 5. 读一个 CUTLASS 3.x GEMM 实例

vLLM 的 FP8/INT8 scaled GEMM 用 CUTLASS 3.x 写在 `csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/`（v0.20.0；早期版本在 `csrc/quantization/cutlass_w8a8/`）。`scaled_mm.cuh` 中的 `cutlass_3x_gemm` 模板是一个标准的 3.x 组装过程（节选，去掉注释）：

```cpp
// vllm v0.20.0: csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm.cuh
template <typename ElementAB_, typename ElementD_,
          template <typename, typename, typename> typename Epilogue_,
          typename TileShape, typename ClusterShape, typename KernelSchedule,
          typename EpilogueSchedule>
struct cutlass_3x_gemm {
  using ElementAB  = ElementAB_;
  using ElementAcc = typename std::conditional<std::is_same_v<ElementAB, int8_t>,
                                               int32_t, float>::type;
  using Epilogue   = Epilogue_<ElementAcc, ElementD_, TileShape>;
  using EVTCompute = typename Epilogue::EVTCompute;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape,
          ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAcc, float, ElementC, StrideC, AlignmentCD, ElementD, StrideD,
          AlignmentCD, EpilogueSchedule, EVTCompute>::CollectiveOp;

  using Stages = typename cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
          ElementAB, cutlass::layout::RowMajor,    AlignmentAB,   // A
          ElementAB, cutlass::layout::ColumnMajor, AlignmentAB,   // B
          ElementAcc, TileShape, ClusterShape, Stages,
          KernelSchedule>::CollectiveOp;

  using KernelType = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
};
```

四个模板参数在 `scaled_mm_sm90_fp8_dispatch.cuh` 里按问题形状选择，例如默认配置（$$M > 128$$）与大形状配置：

```cpp
// vllm v0.20.0: csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm90_fp8_dispatch.cuh
struct sm90_fp8_config_default {          // M in (128, inf)
  using KernelSchedule   = cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape        = Shape<_128, _128, _128>;
  using ClusterShape     = Shape<_2, _1, _1>;
};
struct sm90_fp8_config_M8192_K6144 {      // M >= 8192, K >= 6144
  using KernelSchedule   = cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
  using TileShape        = Shape<_256, _128, _128>;
  using ClusterShape     = Shape<_2, _1, _1>;
};
```

与第五篇和本篇的概念一一对应：

- **`TileShape = Shape<_128,_128,_128>`** 就是 $$BM \times BN \times BK$$。FP8 一个元素 1 字节，$$BK = 128$$ 与 BF16 的 $$BK = 64$$ 搬同样多的字节。大形状用 $$256 \times 128$$，与上文"更大 tile → 更高 shared memory 算术强度"的分析一致。
- **`ClusterShape = Shape<_2,_1,_1>`** 是 Hopper 新增的一层：2 个 block 组成一个 thread block cluster，沿 M 相邻的两个 block 共享同一个 B tile——TMA 的 multicast 可以把一次全局读取同时写进两个 SM 的 shared memory，把 B 的 L2 流量减半。
- **`KernelSchedule`** 选主循环结构：`KernelTmaWarpSpecialized` 是 1 producer + 1 consumer；`...Pingpong` 与 `...Cooperative` 是上一章的两种双 consumer 调度；`FP8FastAccum` 表示用 wgmma 的快速累加模式（不做每条指令后的 FP32 归一化，精度略降、吞吐更高）。默认形状用 Pingpong 隐藏 epilogue，大形状用 Cooperative 拿大 tile。
- **`EpilogueSchedule = TmaWarpSpecialized`**：epilogue 也用 TMA 写回（先把累加器写进 shared memory，再一条 TMA store 到全局），并与 mainloop 的 warp 角色配合。
- **`StageCountAutoCarveout`**：从 shared memory 上限（H100 每 block 227 KB）减去 epilogue 需要的部分，剩下的全部用作 mainloop 的流水 stage——第五篇里"stage 数由 shared memory 容量决定"的自动版本。
- **`PersistentScheduler`**：每个 SM 只启动固定数量的 block，循环领取 tile，替代"一个 block 一个 tile"的 grid——第十篇讨论 tile 顺序时会回来。
- **`EVTCompute`（Epilogue Visitor Tree）**：融合算子以类型树表达。vLLM 的 `ScaledEpilogue`（`csrc/cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp`）是 `Sm90EVT<multiplies, ScaleA, Sm90EVT<multiplies, ScaleB, Accum>>`：先把 INT32/FP32 累加器逐列乘 `b_scale`、再逐行乘 `a_scale`，然后转成 BF16 写出——量化 GEMM 的反量化被融合进 epilogue，不需要额外 kernel。第九篇展开。

### 6. Ampere 的 2.x 风格实例

同一目录的 `scaled_mm_c2x.cuh` 与 `scaled_mm_c2x_sm80_dispatch.cuh` 是给 sm_80 的 CUTLASS 2.x 版本：

```cpp
// vllm v0.20.0: csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_c2x_sm80_dispatch.cuh
struct sm80_config_default {              // M in (128, inf)
  using TileShape        = typename cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape        = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  using Cutlass2xGemm =
      cutlass_2x_gemm<cutlass::arch::Sm80, enable_sm80_to_sm89, InType, OutType,
                      Epilogue, TileShape, WarpShape, InstructionShape, 5>;
};
```

`cutlass_2x_gemm` 内部用 `cutlass::gemm::kernel::DefaultGemmWithVisitor<...>::GemmKernel` 组装，再包进 `device::GemmUniversalAdapter`。三层 tile 直接对应本篇 kernel：`TileShape<128,128,64>` 是 block tile（INT8 的 $$BK = 64$$ 与 BF16 的 32 字节数相同）；`WarpShape<64,64,64>` 是 warp tile，$$2 \times 2$$ 排布 4 个 warp（128 线程）；`InstructionShape<16,8,32>` 是 INT8 的 `mma.sync.m16n8k32`，BF16 就是 `<16,8,16>`；最后的 `5` 是 stage 数——$$5 \times (128 + 128) \times 64 \times 1$$ B = 80 KiB 的 shared memory，与源码注释的 81920 字节一致。不用 CUTLASS 直接写 Ampere kernel 的读者，可以把 `DefaultGemmConfiguration<OpClassTensorOp, Sm80, ...>` 中的默认值当作调参起点。

vLLM 在 `scaled_mm_entry.cu` 中按 `get_sm_version_num()` 分派：sm_90 走 `cutlass_scaled_mm_sm90`（c3x），sm_89 走 `cutlass_scaled_mm_sm89`（c2x，Ada 有 FP8 mma），sm_80 走 `cutlass_scaled_mm_sm80`（c2x，仅 INT8），sm_75 走 Turing 版本；sm_100/sm_120 各有自己的 c3x 版本。它还做了本篇一直强调的布局检查：

```cpp
// vllm v0.20.0: csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_entry.cu
STD_TORCH_CHECK(a.stride(1) == 1 && c.stride(1) == 1);  // Row-major
STD_TORCH_CHECK(b.stride(0) == 1);                      // Column-major
STD_TORCH_CHECK(c.stride(0) % 16 == 0 && b.stride(1) % 16 == 0);  // 16 Byte Alignment
```

A 行主序、B 列主序（即 $$N \times K$$ 行主序）、16 字节对齐——与本篇 kernel 的 "TN" 约定和 `cp.async`/`ldmatrix` 的 16 字节要求完全相同。


## 六、PyTorch 与 vLLM 如何使用 Tensor Core

### 1. ATen `matmul` → cuBLAS / cuBLASLt

PyTorch 自己不写 GEMM kernel。`torch.matmul`、`torch.mm`、`nn.Linear` 在 CUDA 上最终进入 `aten/src/ATen/native/cuda/Blas.cpp` 的 `addmm_out_cuda_impl`，它有两条路径（v2.10.0）：

```cpp
// pytorch v2.10.0: aten/src/ATen/native/cuda/Blas.cpp（节选）
Tensor& addmm_out_cuda_impl(Tensor& result, const Tensor& self, const Tensor& mat1,
                            const Tensor& mat2, const Scalar& beta, const Scalar& alpha,
                            Activation activation, bool disable_addmm_cuda_lt_override) {
  // ...
  cublasCommonArgs args(mat1, mat2, result);
  if (!disable_addmm_cuda_lt) {           // The Lt path
    lt_success = launchGemmAndBiasCublasLt<scalar_t>(args, use_bias_ptr_lt ? ... , alpha, activation);
    if (!lt_success)
      return addmm_out_cuda_impl(result, self, mat1, mat2, beta, alpha, activation, true);
  } else {                                 // No Lt, we use a GEMM instead
    launchGemmCublas<scalar_t>(args, alpha, beta);
    switch (activation) {                  // epilogue 退化为单独的 kernel
      case Activation::RELU: at::relu_(*args.result); break;
      case Activation::GELU: at::gelu_(*args.result, "tanh"); break;
      default: break;
    }
  }
}
```

- **cuBLASLt 路径**（`gemm_and_bias`，在 `aten/src/ATen/cuda/CUDABlas.cpp` 中调用 `cublasLtMatmul`）：当 `self` 是可以广播成一维的 bias 时走这条路，通过 `CUBLASLT_MATMUL_DESC_EPILOGUE` 设置 `CUBLASLT_EPILOGUE_BIAS` / `RELU_BIAS` / `GELU_BIAS`，把 bias 加法和激活融合进 GEMM 的 epilogue——`nn.Linear` 后接 `GELU` 时，`torch._addmm_activation` 就是这条路，一个 kernel 完成。cuBLASLt 内部按问题形状、布局、对齐做启发式选择，选出的 kernel 在结构上就是本篇讲的那种（Ampere 上 `mma.sync` + `ldmatrix` + `cp.async` 多 stage，Hopper 上 TMA + wgmma + warp specialization）。
- **cuBLAS 路径**（`at::cuda::blas::gemm`，调用 `cublasGemmEx` / `cublasGemmStridedBatchedEx`）：一般的 $$\alpha AB + \beta C$$，没有 epilogue 融合；激活单独再跑一个 elementwise kernel。

所以对 PyTorch 用户，"Tensor Core 有没有用上"由 dtype 与对齐决定：BF16/FP16 输入默认走 Tensor Core；FP32 输入默认**不**走 TF32 Tensor Core，需要 `torch.backends.cuda.matmul.allow_tf32 = True`（或 `set_float32_matmul_precision("high")`）；维度不是 8（BF16）/ 16（INT8、FP8）的倍数时 cuBLAS 可能退到效率较低的 kernel，这也是为什么 vocab size、hidden size 通常对齐到 64 或 128。

### 2. vLLM：用 CUTLASS 写 cuBLAS 不提供的 GEMM

cuBLAS 覆盖了标准 dtype 的标准 GEMM，但推理系统需要的很多 GEMM 它不提供或不够快：per-token / per-channel scale 的 FP8/INT8 GEMM（反量化融合进 epilogue）、blockwise FP8（DeepSeek-V3 风格的 128×128 块 scale）、MoE 的 grouped GEMM（一次 launch 算多个专家、每个专家不同 M）、W4A8 混合精度。vLLM 用 CUTLASS 自己组装这些 kernel：上一章看到的 `scaled_mm_c3x_sm90.cu`、`scaled_mm_blockwise_sm90_fp8.cu`、`moe/grouped_mm_c3x_sm90.cu`、`cutlass_w4a8/` 都在同一目录树下。它们的共同结构是：**mainloop 直接复用 CUTLASS 的 `CollectiveBuilder`，只写 epilogue（EVT）和按形状选配置的 dispatch 表**——这正是 CUTLASS 分层的意义：Tensor Core 编程的困难部分（fragment 布局、swizzle、TMA descriptor、warp specialization）被封装在 collective 层，上层应用只做组合。

第九篇讨论量化时会回到这些 kernel 的 epilogue 细节；第七篇会看到 Triton 版本的 fused MoE GEMM 如何用完全不同的方式（编译器自动选择 mma 布局与流水）达到相近的效果。


## 七、小结

回到核心问题：同样是 $$128 \times 128$$ 的分块，Tensor Core 版本与 CUDA Core 版本的结构差异在于：

1. **计算单元**：从"每线程一个 $$TM \times TN$$ 累加器、逐元素 FMA"变成"每 warp 若干 mma fragment、一条指令 4096 FLOP"。累加器仍在寄存器，但按硬件规定的布局分散在 32 个线程里。
2. **数据流**：从"shared → 寄存器（任意布局）→ FMA"变成"shared（swizzled）→ `ldmatrix` → fragment（固定布局）→ `mma.sync`"。因为 mma 对操作数布局有硬性要求，装载必须用 `ldmatrix` 这类"按 fragment 布局装载"的指令，shared memory 也必须 swizzle 以配合 `ldmatrix` 的 8 行 × 16 字节访问模式。
3. **指令预算**：一条 mma 占 Tensor Core 8 个周期，其间只能发约 8 条其他指令，所以装载和地址计算必须极度精简——这是 `cp.async`（Ampere）和 TMA（Hopper）存在的理由。
4. **Hopper**：数据流再简化为"TMA → shared → wgmma 直接读 shared"，寄存器只剩累加器；warp 按 producer / consumer 分工，用 mbarrier 的 full/empty 两组屏障流水。
5. **CUTLASS/CuTe**：把上述每一个决定变成模板参数，把线程 ↔ 数据映射写成 Layout 代数。读懂了手写版本，CUTLASS 的每一层都能对号入座。

三代接口对照：

```text
                 wmma (nvcuda::wmma)      mma.sync (PTX)              wgmma.mma_async (PTX)
架构             Volta+                   Volta+，Ampere 主力          Hopper (sm_90)
粒度             warp                     warp                        warpgroup (4 warps, 128 线程)
BF16 形状        16x16x16                 m16n8k16                    m64nNk16, N = 8..256
每指令乘加       4096                     2048                        最大 262144
操作数来源       寄存器 (load_matrix_sync) 寄存器 (自行装载, ldmatrix)   A: 寄存器或 shared；B: 必须 shared，经 descriptor
fragment 布局    不透明                    文档规定，可逐元素操作         累加器布局文档规定；输入不经寄存器
同步             同步                     同步                        异步：fence / commit_group / wait_group
典型性能         cuBLAS 50–70%            80–95%（CUTLASS 2.x 水平）   接近峰值（CUTLASS 3.x / FA3）
```

m16n8k16 BF16 fragment 速查（$$g = \text{lane}/4$$，$$t = \text{lane} \bmod 4$$）：

```text
操作数   形状        每线程寄存器   寄存器 -> (行, 列)
A        16x16 BF16  4 x b32       a0:(g, 2t..2t+1)  a1:(g+8, 2t..2t+1)  a2:(g, 2t+8..2t+9)  a3:(g+8, 2t+8..2t+9)
B        16x8  BF16  2 x b32       b0:(k=2t..2t+1, n=g)  b1:(k=2t+8..2t+9, n=g)
C/D      16x8  FP32  4 x f32       c0:(g, 2t)  c1:(g, 2t+1)  c2:(g+8, 2t)  c3:(g+8, 2t+1)
ldmatrix.x4 (A)      lane 提供地址  row = lane % 16,  col = 8 * (lane / 16)
ldmatrix.x4 (B, NxK) lane 提供地址  n = lane % 8 + 8 * (lane / 16),  k = 8 * ((lane / 8) % 2)
C -> 下一次 A        两个相邻 n8 tile: {c0,c1}->a0  {c2,c3}->a1  {c0',c1'}->a2  {c2',c3'}->a3
swizzle (64 B 行)    chunk_phys = chunk ^ ((row >> 1) & 3)
```

CUTLASS 3.x 层次与本篇概念的对应：

```text
CUTLASS 层                              本篇手写 kernel 中的对应              关键模板参数
device::GemmUniversalAdapter            bf16_gemm_tn() host 函数              —
kernel::GemmUniversal                   __global__ 函数体、blockIdx 解析       TileScheduler (Persistent)
collective::CollectiveMma               k 循环 + cp.async 流水                TileShape, ClusterShape, Stages, KernelSchedule
collective::CollectiveEpilogue          acc -> bf16x2 store                  EpilogueSchedule, EVT
TiledMma / MMA_Atom                     16 条 mma_bf16_16816 / 一条 mma.sync  SM80_16x8x16_F32BF16BF16F32_TN
TiledCopy / Copy_Atom                   load_tile + ldmatrix_x4               SM80_CP_ASYNC_CACHEGLOBAL, SM75_U32x4_LDSM_N
CuTe Layout / Tensor                    swz(), a_row/b_row, warp_m/warp_n     Shape / Stride / local_tile / local_partition
2.x 对应                                ThreadblockShape / WarpShape / InstructionShape / Stages
```

本篇数字汇总：

```text
A100 BF16 Tensor Core（标称）                312 TFLOPS = 108 SM x 1024 FMA/clk x 2 x 1.41 GHz
A100 FP32 CUDA Core（标称）                  19.5 TFLOPS，相差 16 倍
一条 mma.sync.m16n8k16                      2048 FMA = 4096 FLOP，占一个 Tensor Core 8 个周期
一条 wgmma.m64n256k16                       262144 FMA，Hopper 每 SM 每周期约 2048 FMA
4096^3 BF16 GEMM                            137.4 GFLOP，最小流量 100.7 MB，AI 1365 FLOP/B，理论 0.44 ms
128x128x32 block tile                       每 k-tile 16 KiB / 1 MFLOP，每字节 32 次乘加
64x32 warp tile 的 shared 算术强度           21.3 FLOP/B，需 96 B/clk（上限 128）
64x64 warp tile 的 shared 算术强度           32 FLOP/B，需 64 B/clk
本篇 kernel 预期区间                         cuBLAS 的 80–90%（约 200–260 TFLOPS）
未 swizzle 的 ldmatrix（64 B 行）             4 路 bank conflict；128 B 行 8 路
Hopper setmaxnreg 典型分配                   producer 40 / consumer 232 x 2，(40+232+232) x 128 = 64512 <= 65536
```

到这里，GEMM 这条线已经从 naive 走到了 Tensor Core 的手写实现和 CUTLASS 的组装方式。下一篇换一种完全不同的写法：Triton 把线程、fragment、ldmatrix、swizzle 全部藏进编译器，程序员只写 block 级的张量运算。它藏掉了什么、藏不掉什么，是下一篇的问题：

> **Triton 的 matmul 比手写 CUDA 少 80% 的代码，性能只差 10%。那 10% 在哪里？什么场景下这 10% 值得手写？**


## 下一篇

[Triton：块级编程与编译器的边界](/triton-block-level-programming.html)
