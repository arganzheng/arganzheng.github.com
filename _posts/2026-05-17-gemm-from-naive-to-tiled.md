---
layout: post
title: "GPU Kernel 工程（05）：GEMM——从 naive 到分块"
subtitle: "GEMM from Naive to Tiled: Reaching the Compute Ceiling on CUDA Cores"
tags: [CUDA, Triton, GPU, AI, AI-Infra]
catalog: true
---

> 本文是[《GPU Kernel 工程：从 CUDA 执行模型到 FlashAttention》](/gpu-kernel-engineering.html)系列的第 5 篇（共十篇）。上一篇：[共享内存与 reduction](/shared-memory-reduction-and-softmax.html)　下一篇：[Tensor Core、CUTLASS 与 CuTe](/tensor-cores-cutlass-and-cute.html)

前四篇讨论的 kernel——elementwise、reduction、softmax、LayerNorm——有一个共同点：它们都是 memory-bound 的。每个元素读进来、算一两次、写回去，算术强度远低于 ridge point，优化的全部目标是"把 HBM 带宽用满"。做到了带宽的 80–90%，这类 kernel 就到头了。

GEMM 是这个系列的转折点。一个 4096×4096×4096 的矩阵乘法有 1374 亿次浮点运算，但三个矩阵合计只有 192 MiB（FP32）。它的算术强度是 elementwise 的几千倍，理论上应该被算力而不是带宽限制。但一个不加思考写出来的 GEMM kernel，跑出来的却是 FP32 峰值的百分之一二——它被访存拖住了，只是拖住它的不是 HBM，而是 L2、shared memory 和指令发射。

这一篇要回答总纲提出的问题：

> **一个 4096³ 的 FP32 GEMM，naive 实现要读多少字节？128×128 分块后要读多少？寄存器分块再压多少？每一步的算术强度是多少，对应 Roofline 上的哪个位置？**

全文只用 CUDA Core 做 FP32 SGEMM，不碰 Tensor Core。这是刻意的：分块、寄存器复用、软件流水、bank conflict 这些原理在 FP32 上最容易看清楚，而下一篇的 Tensor Core、CUTLASS、CuTe 只是把同一套结构换成了更宽的指令。方法论与前几篇一致：**每一版先算它的访存量与算术强度，把它标在 Roofline 上，再写代码，再解释差距。**

约定：$$C = A \cdot B$$，$$A$$ 为 $$M \times K$$，$$B$$ 为 $$K \times N$$，$$C$$ 为 $$M \times N$$，全部行主序（row-major）、FP32、$$M = N = K = 4096$$。硬件以 A100 SXM 80GB 为基准：FP32 CUDA Core 19.5 TFLOPS、HBM 约 2.0 TB/s、108 个 SM、每 SM 每周期 shared memory 带宽 128 字节、32 个 4 字节宽的 bank，均为公开标称值。


## 一、理论：4096³ SGEMM 应该多快

### 1. FLOPs 与最小访存

每个输出元素是 $$K$$ 个乘积的和，按 $$K$$ 次乘加、每次乘加 2 FLOP 计：

$$
\text{FLOPs} = 2 \cdot M \cdot N \cdot K = 2 \cdot 4096^3 \approx 1.374 \times 10^{11} = 137.4\ \text{GFLOP}
$$

最少要做的访存是把 $$A$$、$$B$$ 各读一遍、$$C$$ 写一遍。FP32 每个矩阵 $$4096^2 \times 4\ \text{B} = 64\ \text{MiB}$$：

$$
\text{Bytes}_{\min} = 3 \times 64\ \text{MiB} = 192\ \text{MiB} \approx 201\ \text{MB}
$$

算术强度（arithmetic intensity）是两者之比：

$$
I_{\text{GEMM}} = \frac{2MNK}{4(MK + KN + MN)} = \frac{2 \cdot 4096^3}{4 \cdot 3 \cdot 4096^2} = \frac{4096}{6} \approx 683\ \text{FLOP/byte}
$$

对比第一篇的 Roofline：A100 FP32 CUDA Core 的 ridge point 是 $$19.5 / 2.0 \approx 10$$ FLOP/byte。683 比 10 高了近 70 倍，这个 GEMM 毫无疑问是 compute-bound 的。理论时间由算力决定：

$$
t_{\text{FP32}} = \frac{137.4\ \text{GFLOP}}{19.5\ \text{TFLOPS}} \approx 7.0\ \text{ms}
$$

而带宽下界只有 $$201\ \text{MB} / 2.0\ \text{TB/s} \approx 0.1\ \text{ms}$$，是算力下界的 1/70。也就是说，一个理想的 SGEMM kernel 在 A100 上跑 7.0 ms，其中 HBM 只忙 0.1 ms，剩下 6.9 ms 都在等 FMA 单元。换成 BF16 走 Tensor Core（312 TFLOPS），理论时间是 0.44 ms——这是下一篇的事。

### 2. 目标：cuBLAS 在哪里

本篇所有版本的比较对象是 cuBLAS 的 `cublasSgemm`。在 A100 这个形状上，cuBLAS SGEMM 通常能达到 FP32 峰值的 85–95%，即大致落在 7.4–8.2 ms 之间（这是文献与常见实测的经验区间，不是本文的实测值——本系列没有 GPU 可供实测，所有性能数字要么是可推导的理论下界，要么用区间给出）。

`torch.matmul` 对两个 FP32 CUDA 张量最终就是调它。`aten/src/ATen/native/cuda/Blas.cpp`（PyTorch v2.10.0）里的 `addmm_out_cuda_impl` 先尝试 cuBLASLt 路径（能把 bias 和激活融合进 epilogue），失败则退回普通 cuBLAS：

```cpp
// aten/src/ATen/native/cuda/Blas.cpp (v2.10.0), addmm_out_cuda_impl 节选
  cublasCommonArgs args(mat1, mat2, result);
  // The Lt path
  if (!disable_addmm_cuda_lt) {
    bool lt_success = false;
    ...
      AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, scalar_type, "addmm_cuda_lt",
        [&] {
          lt_success = launchGemmAndBiasCublasLt<scalar_t>(args, use_bias_ptr_lt ? std::make_optional(self) : std::nullopt, alpha, activation);
        });
    if (!lt_success) {
      // lt path failed; recurse but disable lt path
      return addmm_out_cuda_impl(result, self, mat1, mat2, beta, alpha, activation, true);
    }
  } else {
    // No Lt, we use a GEMM instead
    ...
          launchGemmCublas<scalar_t>(args, alpha, beta);   // -> at::cuda::blas::gemm<scalar_t>
  }
```

`at::cuda::blas::gemm<float>` 在 `aten/src/ATen/cuda/CUDABlas.cpp` 里落到：

```cpp
// aten/src/ATen/cuda/CUDABlas.cpp (v2.10.0)
template <>
void gemm_internal_cublas<float>(CUDABLAS_GEMM_ARGTYPES(float)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  GEMM_CHECK_ARGVALUES(float);
  TORCH_CUDABLAS_CHECK(cublasSgemm(
      handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
}
```

有两个细节值得留意。第一，cuBLAS 是列主序接口，而 PyTorch 张量默认行主序，`cublasCommonArgs`（`native/cuda/cuBlasCommonArgs.h`）的注释写得很直白："we run the gemm as B.T @ A.T"——行主序的 $$C = AB$$ 等价于列主序的 $$C^T = B^T A^T$$，交换两个操作数即可，不需要真的转置任何数据。第二，同一文件里 FP32 的 `computeType` 默认是 `CUBLAS_COMPUTE_32F`，只有当 `torch.backends.cuda.matmul.allow_tf32 = True`（或 `torch.set_float32_matmul_precision("high")`）时才切到 `CUBLAS_COMPUTE_32F_FAST_TF32`，把 FP32 输入送进 TF32 Tensor Core（156 TFLOPS，8 倍算力，但尾数只有 10 位）。后面做正确性对照时要确认这个开关是关的，否则参照值本身就带着 TF32 的误差。

有了 7.0 ms 的理论时间和 cuBLAS 的位置，下面六版 kernel 每一版都可以问同一个问题：它离 7.0 ms 差多少，差在哪。


## 二、v1：naive——每线程一个输出

### 1. 代码

最直接的写法：把 $$M \times N$$ 个输出元素铺成二维 grid，每个线程负责一个 $$C_{ij}$$，沿 $$K$$ 循环累加。

```cpp
#include <cuda_runtime.h>

// v1: one thread per output element, all operands read from global memory.
__global__ void sgemm_v1_naive(int M, int N, int K,
                               const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
      acc += A[(size_t)row * K + k] * B[(size_t)k * N + col];
    }
    C[(size_t)row * N + col] = acc;
  }
}

// launch: block (32, 8) -> threadIdx.x walks along N so a warp reads 32 consecutive B elements
//   dim3 block(32, 8);
//   dim3 grid((N + 31) / 32, (M + 7) / 8);
//   sgemm_v1_naive<<<grid, block>>>(M, N, K, A, B, C);
```

`threadIdx.x` 映射到列而不是行是有意的：一个 warp 的 32 个线程在同一个 $$k$$ 上读 $$B[k][\text{col} \ldots \text{col}+31]$$，是连续的 128 字节，一次合并访存；而它们读的 $$A[\text{row}][k]$$ 是同一个地址，硬件广播。如果反过来把 `threadIdx.x` 映射到行，warp 内 32 个线程读 $$A$$ 的 32 个不同行，跨度 $$K \times 4 = 16$$ KB，每个线程一个 sector，带宽利用率 1/8——第三篇讲过的非合并访问。

### 2. 访存量与算术强度

每个输出元素要读 $$A$$ 的一行（$$K$$ 个数）和 $$B$$ 的一列（$$K$$ 个数），做 $$K$$ 次 FMA。不考虑任何缓存，从"程序发出的读请求"数：

$$
\text{Reads}_{\text{v1}} = 2 \cdot M \cdot N \cdot K \cdot 4\ \text{B} = 2 \times 4096^3 \times 4\ \text{B} = 512\ \text{GiB}
$$

对全局内存的算术强度：

$$
I_{\text{v1}} = \frac{2K\ \text{FLOP}}{2K \times 4\ \text{B}} = 0.25\ \text{FLOP/byte}
$$

512 GiB 是最小访存 192 MiB 的 2700 多倍；0.25 FLOP/byte 比 ridge point 10 低 40 倍。如果这 512 GiB 真的走 HBM，时间是 $$550\ \text{GB} / 2.0\ \text{TB/s} \approx 275$$ ms，是理论时间的 40 倍。这就是"naive GEMM 是 memory-bound"的字面含义：每一次 FMA 都配了两次全局读。

### 3. cache 挽救了 naive，但只能到几个百分点

真实情况没有这么糟，因为 L1 与 L2 会吸收大部分请求。在 warp 层面，上面已经说过 $$A$$ 是广播、$$B$$ 是合并；在 block 层面，一个 32×8 的 block 里 8 个 warp 在同一个 $$k$$ 读同样的 $$B[k][\text{col}\ldots]$$，会命中 L1；相邻 block 之间还共享 $$A$$ 的行与 $$B$$ 的列，会命中 L2（A100 的 L2 是 40 MB，能装下 $$B$$ 的相当一部分）。真正到 HBM 的流量远小于 512 GiB。

但把请求量按 warp 归并之后再算一遍，就知道 cache 也救不了它。一个 warp 每前进一个 $$k$$：一条 FFMA 指令（32 次 FMA、64 FLOP），配 1 个 sector 的 $$A$$（广播）加 4 个 sector 的 $$B$$（128 字节），合计 160 字节的 L1 请求。整个 GEMM 有 $$MNK/32$$ 个这样的 warp-step：

$$
\text{L1 requests} = \frac{4096^3}{32} \times 160\ \text{B} \approx 343\ \text{GB}
$$

L1 的命中率并不高（$$B$$ 是 64 MiB 的流式访问，一个 SM 上的 block 之间没有同步，读同一个 $$k$$ 行的时机对不上），大量请求会落到 L2。A100 的 L2 带宽约为 HBM 的两三倍（数量级），就算全部命中 L2，343 GB 也要几十毫秒，已经是理论时间的 5–10 倍。再叠加两个因素：每条 FFMA 前面有两条依赖它的 load，每个线程只有一条累加链、没有指令级并行，延迟只能靠 warp 数量掩盖；而指令流里 FFMA 只占大约四分之一，其余是 load 和地址计算。三个因素叠在一起，v1 通常只能达到 FP32 峰值的 1–3%，即 200–700 ms 量级。

这一版的教训不是"cache 没用"，而是：**cache 只能减少 HBM 流量，减不掉指令数和 L1/L2 的请求数**。要让 FMA 单元忙起来，必须让每个操作数被读进来之后**在寄存器或 shared memory 里被复用很多次**。这就是分块。


## 三、v2：shared memory 分块

### 1. 分块把算术强度变成一个可设计的参数

把 $$C$$ 切成 $$BM \times BN$$ 的 tile，每个 block 负责一个 tile。计算这个 tile 需要 $$A$$ 的 $$BM \times K$$ 条带和 $$B$$ 的 $$K \times BN$$ 条带；沿 $$K$$ 再切成长度 $$BK$$ 的段，每一段把 $$A$$ 的 $$BM \times BK$$ 子块与 $$B$$ 的 $$BK \times BN$$ 子块载入 shared memory，block 内所有线程从 shared 取数做 $$BM \cdot BN \cdot BK$$ 次 FMA。

数一下全局读取量。每个 tile 沿 $$K$$ 读取 $$(BM + BN) \cdot K$$ 个元素，共 $$\frac{M}{BM} \cdot \frac{N}{BN}$$ 个 tile：

$$
\text{Reads}_{\text{tiled}} = \frac{MN}{BM \cdot BN} \cdot (BM + BN) \cdot K = MNK \left( \frac{1}{BN} + \frac{1}{BM} \right)
$$

对比 naive 的 $$2MNK$$：减少倍数是 $$\frac{2}{1/BM + 1/BN}$$，即 $$BM = BN$$ 时减少 $$BM$$ 倍。每个 tile 一段 $$BK$$ 的算术强度：

$$
I_{\text{tile}} = \frac{2 \cdot BM \cdot BN \cdot BK}{(BM + BN) \cdot BK \cdot 4\ \text{B}} = \frac{BM \cdot BN}{2 (BM + BN)}\ \text{FLOP/byte}
$$

$$BK$$ 消掉了——它不影响对全局内存的算术强度，只影响一次载入的粒度和同步次数。代入几个尺寸：

```text
BM = BN     全局读取（4096³ FP32）   比 naive 减少   I_tile（FLOP/byte）   相对 FP32 ridge 10
  16        32 GiB                 16 倍          4                    低于
  32        16 GiB                 32 倍          8                    接近
  64         8 GiB                 64 倍         16                   跨过
 128         4 GiB                128 倍         32                   跨过（3 倍余量）
```

关键的一行是 128×128：全局读取从 512 GiB 降到 4 GiB，算术强度 32 FLOP/byte，**跨过了 FP32 的 ridge point 10 并留下 3 倍余量**。这意味着只要 tile 足够大，GEMM 对全局内存就是 compute-bound 的；4 GiB 走 HBM 只要 2 ms，比 7.0 ms 的算力时间短。32×32 的 tile 是一个有意思的临界点：16 GiB、8 FLOP/byte，刚好卡在 ridge 下面；64×64 以 16 FLOP/byte 勉强跨过，但余量只有 1.6 倍，任何一点访存效率损失就会把它拖回 memory-bound——这也是为什么 FP32 GEMM 的默认 tile 通常是 128 宽而不是 64 宽。

### 2. 代码：一线程一输出的分块

v2 保持"每线程一个输出"的结构，只是操作数换成从 shared memory 取：

```cpp
// v2: BM x BN tile of C per block, one thread per output; A/B tiles staged in shared memory.
// Assumes M % BM == 0, N % BN == 0, K % BK == 0 (boundary handling is discussed in section VI).
template <int BM, int BN, int BK>
__global__ void sgemm_v2_smem(int M, int N, int K,
                              const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C) {
  static_assert(BM == BN && BM == BK, "v2 loads one element per thread per tile; keep tiles square");
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  const int tx = threadIdx.x;            // 0..BN-1, along N
  const int ty = threadIdx.y;            // 0..BM-1, along M
  const int row = blockIdx.y * BM + ty;
  const int col = blockIdx.x * BN + tx;

  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += BK) {
    As[ty][tx] = A[(size_t)row * K + k0 + tx];          // BM x BK tile, coalesced along K
    Bs[ty][tx] = B[(size_t)(k0 + ty) * N + col];        // BK x BN tile, coalesced along N
    __syncthreads();                                    // tile visible to all threads
#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      acc += As[ty][kk] * Bs[kk][tx];
    }
    __syncthreads();                                    // everyone done before the tile is overwritten
  }
  C[(size_t)row * N + col] = acc;
}

// launch (BM = BN = BK = 32):
//   dim3 block(32, 32);
//   dim3 grid(N / 32, M / 32);
//   sgemm_v2_smem<32, 32, 32><<<grid, block>>>(M, N, K, A, B, C);
```

两次 `__syncthreads()` 都不能省：第一次保证所有线程写完 shared 再开始读；第二次保证所有线程读完这一段再让下一轮的写覆盖它。shared memory 的读取模式是好的：`As[ty][kk]` 在一个 warp 内（`ty` 固定）是同一地址、广播；`Bs[kk][tx]` 32 个线程读连续 32 个 float，落在 32 个不同 bank，无 conflict。

### 3. 为什么 v2 到不了 128×128，以及 shared 带宽的天花板

这一版有一个结构性的限制：一线程一输出意味着 block 有 $$BM \times BN$$ 个线程，而 block 最多 1024 个线程，所以 tile 最大只能是 32×32。32×32 的算术强度是 8 FLOP/byte，**仍然低于 FP32 的 ridge point**。上面表里 128×128 那行的 32 FLOP/byte 要到 v3 才能兑现。

即使不看全局内存，v2 还有第二个瓶颈——shared memory 带宽。每次 FMA 要从 shared 读两个数（一条 `LDS` 读 $$A$$，一条读 $$B$$），对 shared 的算术强度是：

$$
I_{\text{shared, v2}} = \frac{2\ \text{FLOP}}{2 \times 4\ \text{B}} = 0.25\ \text{FLOP/byte}
$$

A100 每个 SM 每周期能从 shared memory 取 128 字节 = 32 个 float，也就是每周期服务一个 warp 宽度的一次访问（一个"wavefront"）。一个 SM 有 64 个 FP32 单元，峰值每周期 64 次 FMA，即每周期 2 条 warp 级 FFMA 指令。按 v2 的取数模式，一条 FFMA 前面有 2 条 LDS，各占一个 wavefront（$$A$$ 的广播也是一个 wavefront，只是数据少），所以每周期 2 条 FFMA 需要 4 个 wavefront，而 shared 只能给 1 个：

$$
\text{shared 带宽上限} = \frac{1}{4} \times \text{FP32 峰值} = 25\%
$$

再加上指令发射（每 3 条指令只有 1 条是 FFMA，每个 warp 调度器每周期只发一条）和两次同步的开销，v2 通常落在 FP32 峰值的 10–20%。它把 HBM 流量从 512 GiB 砍到 16 GiB，性能却只提高了几倍——瓶颈已经从 L2 挪到了 shared memory 和指令发射，Roofline 的"内存"轴要换成 shared memory 带宽重新画一次。

结论是清楚的：**必须让每次从 shared 读进来的数在寄存器里被复用多次**。这又是同一个原理——把"最内层缓存"从 shared memory 再下移一层到寄存器。


## 四、v3：寄存器分块（含 v4 的加载优化）

### 1. 每线程算 TM×TN 个输出

让每个线程负责 $$C$$ tile 里一个 $$TM \times TN$$ 的小块。每前进一个 $$k$$，线程从 shared 读 $$A$$ 的 $$TM$$ 个数（同一列 $$k$$ 上的 $$TM$$ 行）和 $$B$$ 的 $$TN$$ 个数（同一行 $$k$$ 上的 $$TN$$ 列），做外积，$$TM \times TN$$ 次 FMA 累加到寄存器里。$$TM = TN = 8$$：

$$
I_{\text{shared, v3}} = \frac{2 \cdot TM \cdot TN}{(TM + TN) \times 4\ \text{B}} = \frac{128}{64} = 2\ \text{FLOP/byte}
$$

对 shared 的算术强度从 0.25 升到 2，读取量按 LDS 指令数算减少 $$2 TM \cdot TN / (TM + TN) = 8$$ 倍；如果只按需要独立传输的数据算（v2 里 $$A$$ 那次读在 warp 内是广播，只传一份），减少 $$TM \cdot TN / (TM + TN) = 4$$ 倍。两种算法给出的结论相同：shared 不再是瓶颈。用上一节的 wavefront 语言重算：一个 warp 每前进一个 $$k$$ 发 64 条 FFMA、4 条 128 位的 LDS（$$A$$、$$B$$ 各两条 `float4`），64 条 FFMA 按峰值要 32 个周期（每周期 2 条 warp 级 FFMA），4 条 LDS 中 $$A$$ 的两条是广播、各 1 个 wavefront，$$B$$ 的两条各 2–4 个 wavefront，合计不超过 10 个周期，shared 端有 3 倍以上的余量。指令流里 FFMA 占到 90% 以上。

同时 block 的线程数从 $$BM \cdot BN$$ 降到 $$\frac{BM}{TM} \cdot \frac{BN}{TN}$$：128×128 的 tile、8×8 的线程块，只要 256 个线程。这就解锁了 32 FLOP/byte 的全局算术强度。三层复用一起看：

```text
层次            复用单位                 每次载入被用几次
HBM → shared    BM×BK 与 BK×BN 的 tile    每个 A 元素被 BN 个输出用到，每个 B 元素被 BM 个输出用到
shared → 寄存器  TM 个 a 与 TN 个 b        每个 a 被 TN 次 FMA 用到，每个 b 被 TM 次 FMA 用到
寄存器           TM×TN 个累加器            沿整个 K 累加，K 次写回一次
```

### 2. 寄存器压力与占用率

代价是寄存器。每线程 64 个累加器、8 + 8 个操作数片段、再加上地址、循环变量和编译器为了流水化做的复制，通常在 100–128 个之间。A100 每 SM 有 65536 个寄存器，一个 256 线程的 block 用 $$256 \times 128 = 32768$$ 个：

$$
\text{blocks/SM} = \left\lfloor \frac{65536}{256 \times 128} \right\rfloor = 2 \quad \Rightarrow \quad 16\ \text{warps} = 25\%\ \text{occupancy}
$$

25% 的占用率在前几篇 memory-bound 的 kernel 里是不可接受的，那里靠大量 warp 掩盖几百周期的 HBM 延迟。但 v3 已经是 compute-bound：每个 warp 手里有 64 条互不依赖的 FFMA，指令级并行足以填满 FMA 流水线，不需要靠 warp 之间切换。**占用率是掩盖延迟的手段之一，不是目标**；GEMM 用寄存器换占用率，是划得来的交易。用 `__launch_bounds__(256, 2)` 告诉编译器目标是每 SM 2 个 block，它会把寄存器控制在 128 以内，超出的部分溢出到 local memory——溢出是要避免的，用 `-Xptxas -v` 看 `spill stores/loads` 是否为 0。

### 3. 从全局到 shared 的加载：向量化、转置、bank conflict

这是总纲里的 v4，本文把它并进 v3 的代码里，因为它们几乎总是一起出现。

**向量化**。256 个线程要搬 $$128 \times 8 = 1024$$ 个 $$A$$ 元素和 1024 个 $$B$$ 元素，每人 4 + 4 个，正好一个 `float4`。128 位加载（`LDG.128`）把加载指令数降到 1/4，对 L1 的请求数也更少。前提是地址 16 字节对齐：$$K$$、$$N$$ 是 4 的倍数，基址来自 `cudaMalloc`（256 字节对齐），就满足。

**转置**。计算阶段线程要读"同一列 $$k$$ 上连续 $$TM$$ 行"的 $$A$$。如果 shared 里 $$A$$ 按原样存 `As[BM][BK]`，这 8 个数的地址间隔是 $$BK$$ 个 float，不连续，只能用 8 条 32 位 `LDS`；而且在 warp 内如果有 8 个线程的 `tm` 不同、读同一个 `kk`，它们的地址间隔 $$8 \times BK \times 4$$ 字节，$$BK = 8$$ 时正好是 256 字节，全部落在同一个 bank——8 路 bank conflict。两个经典解法：一是 `As[BM][BK + 1]` 的 padding，行跨度变成 9 个 float，同一列的元素错开一个 bank，conflict 消失，但 8 个数仍然不连续、不能用 `float4` 读；二是**转置存入** `As[BK][BM]`，同一 $$k$$ 的 $$BM$$ 个 $$A$$ 元素在 shared 里连续，读取阶段一条 `LDS.128` 取 4 个，两条取完 $$TM = 8$$，且 16 个共享同一个 `tm` 的线程读同一地址、是广播。转置的代价发生在写入端：`float4` 从全局读进来的 4 个 $$k$$ 连续元素要拆成 4 次标量写到 shared 的 4 个不同行。每个 tile 只写一次、读 $$BK$$ 次，把开销放在写这边是对的。

$$B$$ 不需要转置：计算阶段读"同一行 $$k$$ 上连续 $$TN$$ 列"，正好是 `Bs[BK][BN]` 的一行里连续的 8 个，`float4` 直接读。

### 4. 代码

```cpp
#include <cuda_runtime.h>

// v3 (+v4): BM x BN tile per block, TM x TN outputs per thread, float4 global loads,
// A tile stored transposed (As[k][m]) so per-k fragments are contiguous float4 reads.
// Assumes M % BM == 0, N % BN == 0, K % BK == 0 and 16-byte aligned A/B/C.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN), 2)
sgemm_v3_regtile(int M, int N, int K,
                 const float* __restrict__ A,
                 const float* __restrict__ B,
                 float* __restrict__ C) {
  constexpr int THREADS_N = BN / TN;
  constexpr int THREADS_M = BM / TM;
  constexpr int NTHREADS  = THREADS_M * THREADS_N;
  constexpr int A_LOADS   = BM * BK / (4 * NTHREADS);   // float4 chunks of A per thread
  constexpr int B_LOADS   = BK * BN / (4 * NTHREADS);   // float4 chunks of B per thread
  static_assert(BM * BK % (4 * NTHREADS) == 0, "A tile must split evenly into float4 per thread");
  static_assert(BK * BN % (4 * NTHREADS) == 0, "B tile must split evenly into float4 per thread");
  static_assert(TM % 4 == 0 && TN % 4 == 0, "fragments are read as float4");
  static_assert(BK % 4 == 0 && BN % 4 == 0, "tile rows must be float4 multiples");

  __shared__ __align__(16) float As[BK][BM];   // transposed: As[k][m]
  __shared__ __align__(16) float Bs[BK][BN];   // Bs[k][n]

  const int tid  = threadIdx.x;
  const int tm   = tid / THREADS_N;            // thread's row index inside the tile grid
  const int tn   = tid % THREADS_N;            // thread's column index
  const int row0 = blockIdx.y * BM;
  const int col0 = blockIdx.x * BN;

  const float* A_blk = A + (size_t)row0 * K;   // top-left of this block's A strip
  const float* B_blk = B + col0;               // top-left of this block's B strip

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;
  float a_frag[TM];
  float b_frag[TN];

  for (int k0 = 0; k0 < K; k0 += BK) {
    // ---- global -> shared: A tile (BM x BK), transposed on the way in ----
#pragma unroll
    for (int i = 0; i < A_LOADS; ++i) {
      const int idx = tid + i * NTHREADS;          // float4 index in [0, BM*BK/4)
      const int r   = idx / (BK / 4);              // row (m) inside the tile
      const int c   = (idx % (BK / 4)) * 4;        // k offset inside the tile
      const float4 v = *reinterpret_cast<const float4*>(&A_blk[(size_t)r * K + k0 + c]);
      As[c + 0][r] = v.x;
      As[c + 1][r] = v.y;
      As[c + 2][r] = v.z;
      As[c + 3][r] = v.w;
    }
    // ---- global -> shared: B tile (BK x BN), same layout ----
#pragma unroll
    for (int i = 0; i < B_LOADS; ++i) {
      const int idx = tid + i * NTHREADS;
      const int r   = idx / (BN / 4);              // k row inside the tile
      const int c   = (idx % (BN / 4)) * 4;        // n offset inside the tile
      *reinterpret_cast<float4*>(&Bs[r][c]) =
          *reinterpret_cast<const float4*>(&B_blk[(size_t)(k0 + r) * N + c]);
    }
    __syncthreads();

    // ---- compute: TM x TN outer products per k ----
#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
#pragma unroll
      for (int i = 0; i < TM; i += 4) {
        const float4 v = *reinterpret_cast<const float4*>(&As[kk][tm * TM + i]);
        a_frag[i] = v.x; a_frag[i + 1] = v.y; a_frag[i + 2] = v.z; a_frag[i + 3] = v.w;
      }
#pragma unroll
      for (int j = 0; j < TN; j += 4) {
        const float4 v = *reinterpret_cast<const float4*>(&Bs[kk][tn * TN + j]);
        b_frag[j] = v.x; b_frag[j + 1] = v.y; b_frag[j + 2] = v.z; b_frag[j + 3] = v.w;
      }
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] += a_frag[i] * b_frag[j];
    }
    __syncthreads();
  }

  // ---- epilogue: write TM x TN block of C with float4 stores ----
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    float* c_row = C + (size_t)(row0 + tm * TM + i) * N + col0 + tn * TN;
#pragma unroll
    for (int j = 0; j < TN; j += 4) {
      *reinterpret_cast<float4*>(c_row + j) =
          make_float4(acc[i][j], acc[i][j + 1], acc[i][j + 2], acc[i][j + 3]);
    }
  }
}

// launch (BM = BN = 128, BK = 8, TM = TN = 8 -> 256 threads/block):
//   dim3 block(256);
//   dim3 grid(N / 128, M / 128);
//   sgemm_v3_regtile<128, 128, 8, 8, 8><<<grid, block>>>(M, N, K, A, B, C);
```

几处索引值得对一遍。$$A$$ 的加载：`A_LOADS = 128 × 8 / 1024 = 1`，`idx` 取遍 0–255，`r = idx / 2` 覆盖 128 行，`c ∈ {0, 4}` 覆盖 8 个 $$k$$，每个元素恰好被搬一次。$$B$$ 的加载：`r = idx / 32` 覆盖 8 行，`c = (idx % 32) × 4` 覆盖 128 列。转置写入 `As[c + 0][r]` 在一个 warp 内：`r` 取 16 个连续值、`c` 取两个值，地址 `c × 128 + r`，$$c = 4$$ 的那一组偏移 512 个 float，是 32 的整倍数，与 $$c = 0$$ 落在相同的 bank——2 路 conflict。它每个 tile 只发生 4 次，对比 $$8 \times 64$$ 条 FFMA，可以忽略；要消除它，把 `As` 改成 `As[BK][BM + 4]`（行跨度 132 个 float，`c = 4` 那组错开 16 个 bank，且 528 字节的行跨度仍是 16 字节对齐）即可。

读取阶段：`As[kk][tm * TM + i]` 一个 warp 里只有两个不同的 `tm`（`tid / 16`），是两个地址的广播，一个 wavefront。`Bs[kk][tn * TN + j]` 16 个不同的 `tn` 各读一个 `float4`，地址间隔 32 字节；32 字节的间隔意味着 `tn` 与 `tn + 4` 落在相同的 4 个 bank，16 个地址分成 4 组，需要 4 个 wavefront，而理想是 2 个（256 字节 / 128 字节）。这是本文实现里**有意保留的一处残余 bank conflict**：修法是把每个线程的 8 列拆成两段不相邻的 4 列（`tn * 4` 和 `BN/2 + tn * 4`），让相邻 `tn` 读相邻的 16 字节，代价是 epilogue 的写地址也要相应拆开。第八节讨论"离 cuBLAS 还差什么"时会回到它。

### 5. v3 在 Roofline 上的位置

对 HBM：32 FLOP/byte，跨过 ridge，compute-bound。对 shared：2 FLOP/byte，shared 带宽有 4 倍以上余量。指令流：FFMA 占 90% 以上。占用率 25%，靠 ILP 而不是 TLP 掩盖延迟。剩下最明显的浪费是：每个 tile 的加载与计算是串行的——加载时 FMA 单元空转，计算时加载单元空转，中间还有两次 `__syncthreads()`。这类实现通常能到 FP32 峰值的 40–60%（没有 float4 与转置时）到 55–70%（有了本节的加载优化）。把加载藏到计算后面，是 v5。


## 五、v5：双缓冲与软件流水

### 1. Ampere 之前的写法：寄存器预取

思路是让 tile $$k+1$$ 的加载与 tile $$k$$ 的计算重叠。在 Ampere 之前，全局到 shared 的数据必须经过寄存器，所以流水线是三段式的：在计算 tile $$k$$ 之前，先把 tile $$k+1$$ 从全局 `LDG` 到一组临时寄存器（这些 load 是异步的，发出去就可以继续算）；算完 tile $$k$$ 之后把临时寄存器写进 shared 的**另一套** buffer；一次 `__syncthreads()`，交换 buffer 指针，进入下一轮。两套 shared buffer 就是"双缓冲"。代价是那组临时寄存器（本文配置下 8 个）和更复杂的代码；收益是加载延迟被计算掩盖，每个 tile 的同步从两次降到一次——因为写入的是另一套 buffer，不需要"读完再写"的那次同步。

### 2. Ampere 的 cp.async：绕过寄存器

Ampere（sm_80）引入了 `cp.async` 指令：直接把全局内存的 4/8/16 字节拷到 shared memory，不经过寄存器，也不阻塞发出它的线程。`.cg` 变体（cache global）绕过 L1、只在 L2 缓存，适合这种"读一次就进 shared"的数据。配套两条指令管理完成状态：`cp.async.commit_group` 把此前发出的所有 `cp.async` 打包成一个 group；`cp.async.wait_group N` 阻塞到"未完成的 group 不超过 N 个"。

有了它，流水线可以做成任意深度的多 stage：shared 里放 $$S$$ 套 buffer，永远保持 $$S - 1$$ 个 tile 在途。第 $$t$$ 轮的循环体是：

```text
wait_group(S - 2)          # 最多允许 S-2 个 group 未完成 → tile t 一定已落地（对本线程而言）
__syncthreads()            # 对所有线程而言 tile t 已落地；且所有线程都算完了 tile t-1，它的 buffer 可以复用
issue cp.async for tile t + S - 1 → buffer (t + S - 1) mod S   ( == (t - 1) mod S，刚释放的那套 )
commit_group               # 即使没有实际 copy 也 commit 一个空 group，保持 group 计数一致
compute tile t from buffer t mod S
```

每个 tile 只有一次 `__syncthreads()`。shared 用量：

$$
\text{smem} = S \cdot (BM + BN) \cdot BK \cdot 4\ \text{B}
$$

$$BM = BN = 128$$、$$BK = 8$$、$$S = 3$$ 时是 24 KB，在 48 KB 的静态上限之内；换 $$BK = 32$$ 就是 96 KB，必须走动态 shared memory 并用 `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)` 显式 opt-in（A100 每 block 最多 163 KB）。更大的 $$BK$$ 意味着每个 tile 更多的 FMA、更少的同步，但 shared 用量按 stage 数成倍增长，会挤占每 SM 能驻留的 block 数。

### 3. cp.async 与 A 的转置

`cp.async` 一次搬 16 字节到 shared 的连续 16 字节，**无法在途中转置**。上一节把 $$A$$ 转置存成 `As[k][m]` 的技巧在这里用不了了，有三种选择：一是 shared 里 $$A$$ 按原样存 `As[m][k]`，读取阶段用 $$TM$$ 条 32 位 `LDS`——在本文的线程映射下一个 warp 只有两个不同的 `tm`，是广播，没有 bank conflict，只是指令数多了（每个 $$k$$ 8 + 2 = 10 条 LDS 对 64 条 FFMA，FFMA 仍占 85% 以上）；二是 $$A$$ 走寄存器预取并转置、$$B$$ 走 `cp.async`，两种流水线混用；三是要求 $$A$$ 本身按 $$K$$ 主序存放（即传入 $$A^T$$）。第三种在实际系统里并不罕见：`nn.Linear` 算的是 $$X W^T$$，$$W$$ 以 $$[N, K]$$ 行主序存放，所以是 $$B$$ 而不是 $$A$$ 需要"转置"——cuBLAS 与 CUTLASS 为 NN、NT、TN、TT 四种布局组合分别准备了 kernel，就是因为最优的 shared 布局与线程映射依赖于操作数的主序。本文的 v5 取第一种，代码最短，也足以说明流水线本身。

### 4. 代码

```cpp
#include <cuda_runtime.h>
#include <cstdint>

// 16-byte asynchronous global -> shared copy (Ampere+), bypassing L1 (.cg) and registers.
__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* gmem_ptr) {
  const unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// v5: STAGES-deep cp.async pipeline; one __syncthreads() per BK tile.
// Shared layout per stage: As[BM][BK] (not transposed), Bs[BK][BN]. Dynamic shared memory:
//   STAGES * (BM + BN) * BK * sizeof(float) bytes.
// Assumes M % BM == 0, N % BN == 0, K % BK == 0 and 16-byte aligned A/B/C.
template <int BM, int BN, int BK, int TM, int TN, int STAGES>
__global__ void __launch_bounds__((BM / TM) * (BN / TN), 2)
sgemm_v5_cp_async(int M, int N, int K,
                  const float* __restrict__ A,
                  const float* __restrict__ B,
                  float* __restrict__ C) {
  constexpr int THREADS_N = BN / TN;
  constexpr int THREADS_M = BM / TM;
  constexpr int NTHREADS  = THREADS_M * THREADS_N;
  constexpr int A_LOADS   = BM * BK / (4 * NTHREADS);
  constexpr int B_LOADS   = BK * BN / (4 * NTHREADS);
  constexpr int A_STAGE   = BM * BK;          // floats per stage
  constexpr int B_STAGE   = BK * BN;
  static_assert(STAGES >= 2, "need at least double buffering");
  static_assert(BM * BK % (4 * NTHREADS) == 0 && BK * BN % (4 * NTHREADS) == 0,
                "tiles must split evenly into 16-byte chunks per thread");
  static_assert(BK % 4 == 0 && BN % 4 == 0 && TN % 4 == 0, "16-byte alignment");

  extern __shared__ __align__(16) float smem[];
  float* As = smem;                           // STAGES * A_STAGE
  float* Bs = smem + STAGES * A_STAGE;        // STAGES * B_STAGE

  const int tid  = threadIdx.x;
  const int tm   = tid / THREADS_N;
  const int tn   = tid % THREADS_N;
  const int row0 = blockIdx.y * BM;
  const int col0 = blockIdx.x * BN;
  const float* A_blk = A + (size_t)row0 * K;
  const float* B_blk = B + col0;

  // Issue the asynchronous copies for one BK tile into one stage buffer.
  auto load_stage = [&](int stage, int k0) {
    float* as = As + stage * A_STAGE;
    float* bs = Bs + stage * B_STAGE;
#pragma unroll
    for (int i = 0; i < A_LOADS; ++i) {
      const int idx = tid + i * NTHREADS;
      const int r   = idx / (BK / 4);
      const int c   = (idx % (BK / 4)) * 4;
      cp_async_16(&as[r * BK + c], &A_blk[(size_t)r * K + k0 + c]);
    }
#pragma unroll
    for (int i = 0; i < B_LOADS; ++i) {
      const int idx = tid + i * NTHREADS;
      const int r   = idx / (BN / 4);
      const int c   = (idx % (BN / 4)) * 4;
      cp_async_16(&bs[r * BN + c], &B_blk[(size_t)(k0 + r) * N + c]);
    }
  };

  const int num_tiles = K / BK;

  // Prologue: put STAGES-1 tiles in flight (one commit group each, possibly empty).
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < num_tiles) load_stage(s, s * BK);
    cp_async_commit();
  }

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;
  float a_frag[TM];
  float b_frag[TN];

  for (int t = 0; t < num_tiles; ++t) {
    cp_async_wait<STAGES - 2>();   // this thread's copies for tile t have landed
    __syncthreads();               // ...for every thread; and everyone finished tile t-1

    // Refill the buffer freed by tile t-1 with tile t+STAGES-1.
    const int nt = t + STAGES - 1;
    if (nt < num_tiles) load_stage(nt % STAGES, nt * BK);
    cp_async_commit();

    const float* as = As + (t % STAGES) * A_STAGE;
    const float* bs = Bs + (t % STAGES) * B_STAGE;
#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        a_frag[i] = as[(tm * TM + i) * BK + kk];        // scalar LDS, warp-broadcast
      }
#pragma unroll
      for (int j = 0; j < TN; j += 4) {
        const float4 v = *reinterpret_cast<const float4*>(&bs[kk * BN + tn * TN + j]);
        b_frag[j] = v.x; b_frag[j + 1] = v.y; b_frag[j + 2] = v.z; b_frag[j + 3] = v.w;
      }
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] += a_frag[i] * b_frag[j];
    }
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    float* c_row = C + (size_t)(row0 + tm * TM + i) * N + col0 + tn * TN;
#pragma unroll
    for (int j = 0; j < TN; j += 4) {
      *reinterpret_cast<float4*>(c_row + j) =
          make_float4(acc[i][j], acc[i][j + 1], acc[i][j + 2], acc[i][j + 3]);
    }
  }
}

// launch (128, 128, 8, 8, 8, 3 stages -> 24 KB dynamic shared memory):
//   constexpr size_t SMEM = 3 * (128 + 128) * 8 * sizeof(float);
//   cudaFuncSetAttribute(sgemm_v5_cp_async<128,128,8,8,8,3>,
//                        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
//   sgemm_v5_cp_async<128,128,8,8,8,3><<<dim3(N / 128, M / 128), dim3(256), SMEM>>>(M, N, K, A, B, C);
```

对 group 计数做一次验算。进入第 $$t$$ 轮时已经 commit 了 $$(S - 1) + t$$ 个 group，对应 tile $$0 \ldots t + S - 2$$；`wait_group(S - 2)` 允许最近的 $$S - 2$$ 个（tile $$t + 1 \ldots t + S - 2$$）仍在途，所以 tile $$t$$ 必然完成。$$S = 2$$ 时退化为 `wait_group 0`，即经典双缓冲：每轮把下一个 tile 发出去、算当前 tile、下一轮开头等它。本轮写入的 buffer 是 $$(t + S - 1) \bmod S = (t - 1) \bmod S$$，装的是 tile $$t - 1$$，它在第 $$t - 1$$ 轮被计算，本轮开头的 `__syncthreads()` 保证了所有线程都算完了它。`cp.async` 要求源与目标都 16 字节对齐：`as[r * BK + c]` 的字节偏移是 $$(8r + c) \times 4$$，$$c$$ 是 4 的倍数；stage 之间的偏移 $$BM \cdot BK \cdot 4 = 4096$$ 字节；全局侧依赖 $$K$$、$$N$$ 是 4 的倍数与 `cudaMalloc` 的对齐。

### 5. v5 在 Roofline 上的位置

v5 相对 v3 没有改变任何算术强度——对 HBM 仍是 32 FLOP/byte，对 shared 仍是 2 FLOP/byte。它改变的是**时间轴上的重叠**：加载不再占用 FMA 单元的空档，同步从每 tile 两次降到一次。这类实现通常能到 FP32 峰值的 70–80%，也就是 cuBLAS SGEMM 的 80–90%。再往上，就要动更精细的东西了（第八节）。


## 六、v6：边界处理

前面五版都假设 $$M$$、$$N$$、$$K$$ 是 tile 的整数倍。真实形状不是这样：Llama-3-8B 的 $$d_{ff} = 14336 = 112 \times 128$$ 还好，但 batch·seq 那一维几乎从来不是 128 的倍数，$$K$$ 也未必是 $$BK$$ 的倍数。三个维度分别处理：

**M、N 方向**：grid 取上整 $$\lceil M / BM \rceil \times \lceil N / BN \rceil$$，边界 block 里超出的行/列在加载时用 predicated load（越界的元素补 0）、在 epilogue 时跳过越界的写。补 0 是安全的：多出来的行列只会算出无人读取的 $$C$$ 元素，而不会污染有效元素。

**K 方向**：最后一段 $$BK$$ 不满时，越界的 $$k$$ 补 0 同样正确——$$0 \times b = 0$$ 不改变累加。但要注意 `float4` 加载与 `cp.async` 的 16 字节粒度：如果 $$K$$ 不是 4 的倍数，最后一个 `float4` 会跨越行尾读到下一行的数据（甚至越过分配末尾），必须退化成标量加载或用 `cp.async` 的 `src-size` 变体（`cp.async.cg.shared.global [dst], [src], 16, src_size;`，不足 16 字节的部分自动零填充）。

**host 侧 padding** 是另一种选择：把矩阵 pad 到 tile 的整数倍，kernel 不带任何边界分支。它多出一次拷贝和额外显存，但 kernel 更快也更简单，量化 GEMM 里常见（权重可以离线 pad 一次）。

边界处理对性能的影响主要是两点：一是边界 block 里的谓词分支让 warp 部分空转，对 4096 这种大形状影响很小（只有最后一行/列的 block 受影响），对小形状影响大；二是模板化——CUTLASS 的做法是为"整除"和"不整除"分别实例化 kernel，前者完全没有边界检查。本文的四个 kernel 为了篇幅只给整除版本，第十篇会在测试框架里覆盖非整除形状。


## 七、tile 大小的三角关系与 wave quantization

### 1. 三角关系

$$(BM, BN, BK, TM, TN)$$ 五个参数受四个约束互相牵制：

- **算术强度**（对 HBM）：$$I = \frac{BM \cdot BN}{2(BM + BN)}$$，tile 越大越高；对 shared：$$\frac{TM \cdot TN}{2(TM + TN)}$$，线程块越大越高。
- **寄存器**：每线程 $$\approx TM \cdot TN + TM + TN + \text{地址与临时} \approx TM \cdot TN + 40$$；每 SM 65536 个，决定能驻留多少 block。
- **shared memory**：$$S \cdot (BM + BN) \cdot BK \cdot 4$$ 字节，每 SM 最多 164 KB（A100）。
- **占用率**：$$\min(\text{寄存器限制}, \text{shared 限制}, \text{线程限制}, 32\ \text{block/SM})$$。

```text
(BM, BN, BK, TM, TN)   线程数  累加器  ≈寄存器  smem/stage  I_HBM   3-stage smem  blocks/SM（寄存器限）
(64,  64,  8,  4, 4)    256     16     ~64      4 KB        16      12 KB         4（ridge 10 之上余量小，不推荐）
(128, 64,  8,  8, 4)    256     32     ~80      6 KB        21.3    18 KB         3
(128, 128, 8,  8, 8)    256     64     ~128     8 KB        32      24 KB         2  ← 本文默认
(128, 128, 16, 8, 8)    256     64     ~128     16 KB       32      48 KB         2（同步次数减半）
(128, 256, 8,  8, 16)   256     128    ~200+    12 KB       42.7    36 KB         1（寄存器接近 255 上限）
(256, 128, 8,  16, 8)   256     128    ~200+    12 KB       42.7    36 KB         1
```

"blocks/SM"那一列按 $$\lfloor 65536 / (256 \times R) \rfloor$$ 估算。可以看到 128×128 是一个自然的平衡点：算术强度已经跨过 ridge 一大截，寄存器正好允许 2 个 block、16 个 warp。256 宽的 tile 算术强度更高，但只能驻留 1 个 block，8 个 warp 要在 `__syncthreads()` 处一起停下，加载与计算的重叠余地更小；它们在 Tensor Core GEMM 里更常见，因为那里每个 warp 的算力更大、更需要减少访存。

### 2. wave quantization

还有一个与 tile 无关于算术强度、却实实在在影响几个百分点的因素。128×128 的 tile 在 4096² 的 $$C$$ 上切出：

$$
\frac{4096}{128} \times \frac{4096}{128} = 1024\ \text{blocks}
$$

108 个 SM、每 SM 2 个 block，同时能跑 216 个：

$$
\frac{1024}{216} \approx 4.74\ \text{waves}
$$

前 4 波满载，第 5 波只有 $$0.74 \times 216 = 160$$ 个 block，SM 有 26% 在空转。如果每个 block 的时间相同，整体效率是 $$4.74 / 5 = 94.8\%$$，损失 5%。这叫 wave quantization（波次量化）。缓解办法有：换一个让 block 数接近 216 整数倍的 tile 尺寸（比如 128×256 → 512 个 block，每 SM 1 个 → 4.74 波，没有改善；256×128 同理；64×128 → 2048 个 block，每 SM 3 个 → 324 并发 → 6.3 波，第 7 波 32%，效率 90%——更差），或者用第九节的 stream-K 把最后一波的工作按 $$K$$ 拆碎分给所有 SM。cuBLAS 的启发式选择 kernel 时就在权衡这些，这也是为什么同一个 GEMM 换一组形状，cuBLAS 的效率会在 85% 与 95% 之间跳动。


## 八、Roofline 汇总与 CUDA Core 的极限

### 1. 六版在 Roofline 上的位置

把前面每一版的数字放在一张表里。"占峰值"一列是文献与常见实现的经验区间，具体数字取决于参数与形状，读者在 A100 上跑出的数字大致应该落在这些区间内：

```text
版本   对 HBM 的 I         对 shared 的 I    主要瓶颈                        占 FP32 峰值（经验区间）
v1     0.25（逻辑读 512 GiB） —              L1/L2 请求数、load 延迟、指令发射     ~1–3%
v2     8（32×32 tile）      0.25            shared 带宽（上限 25%）、指令发射     ~10–20%
v3     32（128×128）        2               加载与计算串行、每 tile 2 次同步       ~40–60%
v4     32                   2               同上，但 LDG.128 / LDS.128 减少指令   ~55–70%
v5     32                   2               残余 bank conflict、LDS/FFMA 配比    ~70–80%
cuBLAS 16–21（多种 tile）    ≥2              —                                  ~85–95%
理论    683                 —               算力                                100%（7.0 ms）
```

从 Roofline 的视角复述一遍这条路径：v1 在图的最左边，x 轴 0.25、被斜线压死；v2 向右走到 8，仍在斜线下面，而且换成 shared memory 那条 Roofline 之后更明显；v3 跨过 ridge 到 32，进入水平线下方，从此以后的优化都不再改变 x 坐标，只是让点沿着垂直方向向水平线靠近——v4 减少指令数、v5 重叠加载与计算，都是在"已经 compute-bound"的前提下压缩非计算的开销。**跨过 ridge 只需要分块；贴近峰值需要流水线。**

### 2. CUDA Core 的极限：到 cuBLAS 的 70–80% 以后

v5 与 cuBLAS 之间还有 10–20 个百分点，分散在几个地方：

- **LDS 与 FFMA 的配比**。每个 warp 调度器每周期只发一条指令。v5 每个 $$k$$ 发 10 条 LDS 对 64 条 FFMA，FFMA 占 86%；cuBLAS/CUTLASS 的 SIMT kernel 用更大的 warp tile（比如每 warp 64×32 或 32×64）与更精细的 lane 映射，把 LDS 压到 4 条 LDS.128 对 64 条 FFMA，再配合双缓冲的寄存器片段（读下一个 $$k$$ 的片段时算当前 $$k$$），让 FFMA 占到 95% 以上。
- **残余 bank conflict**。第四节指出的 $$B$$ 片段 4 个 wavefront 对 2 个理想值，以及 v5 里 $$A$$ 的标量读。修法是调整 lane 到片段的映射（把 8 列拆成两段 4 列）或对 shared 地址做 XOR swizzle，让一个 warp 的 32 次访问恰好覆盖 32 个 bank。下一篇的 CuTe 会把这类布局变换变成可组合的代数。
- **epilogue**。$$C$$ 的写回是 64 MiB，理论 33 µs，占 7 ms 的 0.5%；但 v5 的 epilogue 是每线程 8 行各一个 `float4`，一个 warp 一次写 2 行各 256 字节，合并度尚可但不完美。cuBLAS 会先把累加器经 shared memory 重排，再以完整 128 字节 cache line 写回，并在这里融合 $$\alpha$$、$$\beta$$、bias、激活。
- **wave quantization** 与 tile 启发式，上一节已述。

值得强调的是：这些优化每一项只值 2–5 个百分点，而且互相牵制——更大的 warp tile 要更多寄存器，swizzle 增加地址计算。**在 CUDA Core 上把 SGEMM 从 80% 推到 95% 的工程量，比从 3% 推到 80% 还大**，这也是为什么绝大多数场景直接用 cuBLAS 或 CUTLASS 而不是手写。

而真正的量级差距不在这里。同一块 A100，BF16 Tensor Core 的峰值是 312 TFLOPS，是 FP32 CUDA Core 的 16 倍；一条 `mma.sync.m16n8k16` 让一个 warp 一条指令做 2048 次乘加，而 FFMA 一条只做 32 次。本文建立的每一层结构——block tile、warp tile、shared 布局、流水线——原样保留，只是最内层的 $$TM \times TN$$ 外积换成了 Tensor Core 指令，并且因为算力高了 16 倍，ridge point 也从 10 跳到 156 FLOP/byte，128×128 的 tile 在 BF16 下只有 32 FLOP/byte，**又回到了斜线下面**。这就是下一篇要解决的问题。


## 九、split-K、stream-K 与 GEMV

### 1. 小 M×N、大 K：block 不够用

前面的分析都建立在 1024 个 block 能填满 108 个 SM 的前提上。LLM 推理的 decode 阶段不是这样：每步只处理 batch 个 token，GEMM 的 $$M = \text{batch}$$，可能只有 8 或 16，而 $$N = K = 4096$$ 或更大。用 128×128 的 tile：

$$
\left\lceil \frac{8}{128} \right\rceil \times \frac{4096}{128} = 1 \times 32 = 32\ \text{blocks}
$$

108 个 SM 只有 32 个有活干，其余 76 个空转——不论 kernel 写得多好，上限是 30%。而且每个 block 要独自沿 $$K = 4096$$ 走完 512 个 tile，串行时间很长。

**split-K** 的解法是把 $$K$$ 切成 $$S$$ 段，每段一个 block 独立算部分和，然后归约。$$S = 4$$ 时 block 数变成 128，能覆盖 108 个 SM；归约有两种做法：各 block 用 `atomicAdd` 直接加到 $$C$$ 上（简单，但 FP32 原子加的顺序不确定，结果不可复现，且要先把 $$C$$ 清零），或者写到一个 $$S \times M \times N$$ 的 workspace，再跑一个小 kernel 求和（确定性，多一次 launch 与一次 $$S \times M \times N$$ 的读写）。cuBLAS 与 CUTLASS 都提供 split-K，CUTLASS 的 `GemmSplitKParallel` 就是后一种。

**stream-K**（Osama 等 2023）走得更远：不按 tile 分配 block，而是把总 MAC 数（所有 tile 的所有 $$K$$ 段）均分给固定数量的持久 block（通常等于 SM 数的整数倍），每个 block 拿到的工作可以跨越 tile 边界——比如从 tile 7 的第 300 个 $$k$$ 段一直算到 tile 8 的第 100 个 $$k$$ 段。跨 tile 的部分和通过 workspace 与 block 间的标志位拼回去。它同时解决了 split-K 的"$$S$$ 该取多少"和第七节的 wave quantization：没有最后一波了，因为工作是按量而不是按 tile 切的。CUTLASS 3.x 在 Hopper 的 kernel 里把 stream-K 做成了一种 tile scheduler，大致位于 `include/cutlass/gemm/kernel/` 目录下。

### 2. GEMV：GEMM 的 memory-bound 极限

$$M = 1$$ 时，GEMM 退化为 GEMV。FLOPs 是 $$2NK$$，但必须把整个 $$K \times N$$ 的 $$B$$ 读一遍：

$$
I_{\text{GEMV}} = \frac{2NK}{4NK + 4K + 4N} \approx \frac{2}{4} = 0.5\ \text{FLOP/byte（FP32）}，\quad 1\ \text{FLOP/byte（BF16）}
$$

这比 FP32 的 ridge 10 低 20 倍、比 BF16 的 ridge 156 低 150 倍，是彻头彻尾的 memory-bound。分块、寄存器复用在这里都无济于事——每个权重只用一次，没有可复用的东西；能做的只是把 $$B$$ 以满带宽读一遍：$$N = K = 4096$$ BF16 时 32 MiB，A100 上 17 µs。这是 decode 阶段每一层每一个 Linear 的下界，也是为什么 decode 的延迟由权重字节数决定、与 FLOPs 几乎无关。当 $$M = \text{batch}$$ 从 1 涨到 $$b$$，权重仍只读一次，FLOPs 却涨 $$b$$ 倍，算术强度线性增长，直到 $$b$$ 达到 ridge point（BF16 约 156）才重新变成 compute-bound。第九篇的量化 GEMM 就是在这个前提下工作的：INT4 权重把 $$B$$ 的字节数压到 1/4，直接把 GEMV 的下界降到 1/4。


## 十、实践：接到 PyTorch，验证与测算

### 1. 用 load_inline 编译四个 kernel

把上面的 kernel 放进一个字符串，加上四个 host 端 wrapper，用 `torch.utils.cpp_extension.load_inline` 编译。wrapper 只做形状检查、分配输出、算 grid、拿当前 stream 发射：

```cpp
// ---- host wrappers appended after the four kernels above ----
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

static void check_inputs(const torch::Tensor& A, const torch::Tensor& B) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
  TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32, "FP32 only");
  TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "row-major contiguous only");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && A.size(1) == B.size(0), "shape mismatch");
}

torch::Tensor sgemm_v1(torch::Tensor A, torch::Tensor B) {
  check_inputs(A, B);
  const int M = A.size(0), K = A.size(1), N = B.size(1);
  auto C = torch::empty({M, N}, A.options());
  dim3 block(32, 8);
  dim3 grid((N + 31) / 32, (M + 7) / 8);
  sgemm_v1_naive<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      M, N, K, A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>());
  return C;
}

torch::Tensor sgemm_v2(torch::Tensor A, torch::Tensor B) {
  check_inputs(A, B);
  const int M = A.size(0), K = A.size(1), N = B.size(1);
  TORCH_CHECK(M % 32 == 0 && N % 32 == 0 && K % 32 == 0, "v2 needs multiples of 32");
  auto C = torch::empty({M, N}, A.options());
  sgemm_v2_smem<32, 32, 32><<<dim3(N / 32, M / 32), dim3(32, 32), 0, at::cuda::getCurrentCUDAStream()>>>(
      M, N, K, A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>());
  return C;
}

torch::Tensor sgemm_v3(torch::Tensor A, torch::Tensor B) {
  check_inputs(A, B);
  const int M = A.size(0), K = A.size(1), N = B.size(1);
  TORCH_CHECK(M % 128 == 0 && N % 128 == 0 && K % 8 == 0, "v3 needs M,N % 128 == 0 and K % 8 == 0");
  auto C = torch::empty({M, N}, A.options());
  sgemm_v3_regtile<128, 128, 8, 8, 8><<<dim3(N / 128, M / 128), dim3(256), 0, at::cuda::getCurrentCUDAStream()>>>(
      M, N, K, A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>());
  return C;
}

torch::Tensor sgemm_v5(torch::Tensor A, torch::Tensor B) {
  check_inputs(A, B);
  const int M = A.size(0), K = A.size(1), N = B.size(1);
  TORCH_CHECK(M % 128 == 0 && N % 128 == 0 && K % 8 == 0, "v5 needs M,N % 128 == 0 and K % 8 == 0");
  auto C = torch::empty({M, N}, A.options());
  constexpr int STAGES = 3;
  constexpr size_t SMEM = (size_t)STAGES * (128 + 128) * 8 * sizeof(float);   // 24 KB
  auto kernel = sgemm_v5_cp_async<128, 128, 8, 8, 8, STAGES>;
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM);
    attr_set = true;
  }
  kernel<<<dim3(N / 128, M / 128), dim3(256), SMEM, at::cuda::getCurrentCUDAStream()>>>(
      M, N, K, A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>());
  return C;
}
```

Python 侧：

```python
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = open("sgemm_kernels.cu").read()   # the four kernels + wrappers above
cpp_src = """
torch::Tensor sgemm_v1(torch::Tensor A, torch::Tensor B);
torch::Tensor sgemm_v2(torch::Tensor A, torch::Tensor B);
torch::Tensor sgemm_v3(torch::Tensor A, torch::Tensor B);
torch::Tensor sgemm_v5(torch::Tensor A, torch::Tensor B);
"""
ext = load_inline(
    name="sgemm_ext",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["sgemm_v1", "sgemm_v2", "sgemm_v3", "sgemm_v5"],
    extra_cuda_cflags=["-O3", "-arch=sm_80", "-Xptxas", "-v"],   # -v prints registers/spills per kernel
    verbose=True,
)
```

`-Xptxas -v` 的输出要看两行：`Used N registers` 应在 128 以内（v3/v5），`spill stores`/`spill loads` 应为 0。

### 2. 正确性：与 torch.matmul 对照

```python
torch.backends.cuda.matmul.allow_tf32 = False   # make cuBLAS use true FP32, not TF32
torch.manual_seed(0)
M = N = K = 4096
A = torch.randn(M, K, device="cuda", dtype=torch.float32)
B = torch.randn(K, N, device="cuda", dtype=torch.float32)
ref = torch.matmul(A, B)

for name in ["sgemm_v1", "sgemm_v2", "sgemm_v3", "sgemm_v5"]:
    out = getattr(ext, name)(A, B)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-3)
    print(name, "ok")
```

关于容差：本文的 kernel 与 cuBLAS 都用 FP32 累加，但累加顺序不同（cuBLAS 的 tile 尺寸和 split 方式与本文不同），$$K = 4096$$ 项的和在 FP32 下会有 $$10^{-5}$$ 到 $$10^{-4}$$ 量级的相对差异，所以取 rtol 1e-4；$$C$$ 的元素量级约为 $$\sqrt{K} = 64$$，atol 1e-3 相对它只是 $$1.6 \times 10^{-5}$$，实际由 rtol 主导。若忘记关 TF32，参照值本身有 $$10^{-3}$$ 量级的误差，这个断言会在 v1 上就失败——这正是一个检验测试是否可信的好现象。

### 3. 性能：TFLOPS 与占峰值百分比

第二篇定义的 `bench` 脚手架（`cudaEvent` 计时、warmup、每次迭代前用 `cudaMemsetAsync` 刷一个 128 MB 的缓冲区把 L2 清掉、取中位数）在 Python 侧的等价写法：

```python
def bench(fn, warmup=10, iters=100, flush_l2=True):
    cache = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")  # 128 MB > 2 x 40 MB L2
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        if flush_l2:
            cache.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]   # median, in ms

FLOPS = 2.0 * M * N * K            # 137.4 GFLOP
PEAK_FP32_TFLOPS = 19.5            # A100 SXM, nominal

def report(name, fn):
    ms = bench(fn)
    tflops = FLOPS / (ms * 1e-3) / 1e12
    print(f"{name:12s} {ms:8.2f} ms  {tflops:6.2f} TFLOPS  {100 * tflops / PEAK_FP32_TFLOPS:5.1f}% of FP32 peak")

report("cublas", lambda: torch.matmul(A, B))
report("v1", lambda: ext.sgemm_v1(A, B))
report("v2", lambda: ext.sgemm_v2(A, B))
report("v3", lambda: ext.sgemm_v3(A, B))
report("v5", lambda: ext.sgemm_v5(A, B))
```

读者跑出来的数字大致应该落在第八节表里的区间：v1 在 200–700 ms（1–3%），v2 在 35–70 ms，v3 在 10–13 ms，v5 在 8.8–10 ms，cuBLAS 在 7.4–8.2 ms。如果 v3 或 v5 明显低于区间下限，先看 `-Xptxas -v` 有没有 spill，再看 Nsight Compute 里 `smsp__sass_average_data_bytes_per_wavefront_mem_shared`（bank conflict 指标）和 `sm__inst_executed_pipe_fma` 占比——第十篇会系统讲这些指标。

对 L2 flush 多说一句：4096³ 的 GEMM 输入 128 MiB 超过 40 MB 的 L2，flush 与否差别不大；但形状小的时候（比如 1024³，输入 8 MiB）不 flush 会让 $$A$$、$$B$$ 全程驻留 L2，测出来的是一个偏乐观的数字。


## 小结

这一篇把 GEMM 从 memory-bound 一路推到 compute-bound，每一步都先算数、再写码：

1. **理论**：4096³ FP32 GEMM 有 137.4 GFLOP、最少 192 MiB 访存、算术强度 683 FLOP/byte，远超 A100 FP32 的 ridge point 10；理论时间 7.0 ms 由算力决定，HBM 只忙 0.1 ms。cuBLAS 通常到峰值的 85–95%。
2. **v1 naive**：每次 FMA 配两次全局读，逻辑读取 512 GiB、算术强度 0.25 FLOP/byte。cache 挽救了 HBM 流量，但 343 GB 的 L1/L2 请求、依赖链延迟与 3:1 的 load/FFMA 配比把它压在峰值的 1–3%。
3. **v2 shared memory 分块**：全局读取降到 $$MNK(1/BM + 1/BN)$$，128×128 时 32 FLOP/byte 跨过 ridge；但一线程一输出把 tile 限制在 32×32（8 FLOP/byte），每次 FMA 两条 LDS 让 shared 带宽（每 SM 每周期 128 字节）成为 25% 的上限。
4. **v3 寄存器分块**：每线程 8×8 外积，对 shared 的算术强度从 0.25 升到 2 FLOP/byte，LDS 指令减少 8 倍，256 线程算 128×128 的 tile；约 128 个寄存器、每 SM 2 个 block、25% 占用率——compute-bound 的 kernel 靠 ILP 而不是 TLP。
5. **v4 加载优化**：`float4` 向量化、$$A$$ 转置存入 `As[BK][BM]` 让片段读成为连续的 `LDS.128` 且无 bank conflict；`As[BM][BK+1]` padding 是不能向量化的替代方案。
6. **v5 软件流水**：Ampere 前用寄存器预取 + 双 shared buffer；Ampere 用 `cp.async.cg.shared.global` 直接全局到 shared，`commit_group / wait_group` 管理 $$S$$ 级流水，每 tile 一次 `__syncthreads()`，shared 用量 $$S(BM + BN) \cdot BK \cdot 4$$ 字节。
7. **v6 边界**：predicated load 补 0、越界不写；`float4` 与 `cp.async` 的 16 字节粒度要求 $$K$$、$$N$$ 是 4 的倍数，否则退化或用 `src-size` 零填充；或 host 侧 pad。
8. **tile 三角关系**：算术强度、寄存器、shared、占用率互相牵制，128×128×8、8×8 是 CUDA Core 上的自然平衡点；wave quantization（1024 个 block / 216 并发 = 4.74 波）损失约 5%。
9. **CUDA Core 极限**：v5 到 FP32 峰值的 70–80%、cuBLAS 的 80–90%；剩余差距在 LDS/FFMA 配比、残余 bank conflict、epilogue、tile 启发式。Tensor Core 的峰值高 16 倍，ridge 从 10 跳到 156，128×128 的 tile 在 BF16 下又回到斜线下面——下一篇的起点。
10. **split-K / stream-K / GEMV**：decode 阶段 $$M$$ 很小、block 数填不满 108 个 SM，split-K 沿 $$K$$ 并行再归约，stream-K 按 MAC 总量给持久 block 分工；$$M = 1$$ 的 GEMV 算术强度 0.5（FP32）/ 1（BF16）FLOP/byte，只能靠减少权重字节数（量化）加速。

```text
版本   核心改动                 对 HBM 的 I    对 shared 的 I   每 tile 同步   占 FP32 峰值（经验区间）
v1     一线程一输出，全局读       0.25          —               —             ~1–3%
v2     shared 分块 32×32×32      8             0.25            2             ~10–20%
v3     寄存器分块 128×128×8/8×8   32            2               2             ~40–60%
v4     float4 加载 + A 转置       32            2               2             ~55–70%
v5     cp.async 3-stage 流水     32            2               1             ~70–80%
v6     边界处理（predicated）     同上          同上             同上          大形状几乎无影响
cuBLAS                          16–21         ≥2              —             ~85–95%

理论下界（A100 FP32 19.5 TFLOPS）：137.4 GFLOP / 19.5 TFLOPS = 7.0 ms；HBM 下界 0.1 ms

tile 参数（本文默认）：BM=BN=128, BK=8, TM=TN=8, 256 线程, 3 stage
  寄存器 ≈128/线程 → 2 block/SM → 16 warp（25%）
  shared = 3 × 256 × 8 × 4 B = 24 KB/block
  grid = 32 × 32 = 1024 block → 4.74 wave（约 5% 量化损失）
```


## 下一篇

[Tensor Core、CUTLASS 与 CuTe](/tensor-cores-cutlass-and-cute.html)
