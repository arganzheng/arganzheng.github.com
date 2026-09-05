---
layout: post
title: "GPU Kernel 工程（09）：量化与融合 kernel——推理系统的其余部分"
subtitle: "Quantized and Fused Kernels: The Rest of the Inference Stack"
tags: [CUDA, Triton, GPU, AI, AI-Infra]
catalog: true
---

> 本文是[《GPU Kernel 工程：从 CUDA 执行模型到 FlashAttention》](/gpu-kernel-engineering.html)系列的第 9 篇（共十篇）。上一篇：[Attention Kernel：FlashAttention 与 PagedAttention](/attention-kernels-flashattention-and-pagedattention.html)　下一篇：[剖析、测试与贡献](/kernel-profiling-testing-and-contribution.html)

上一篇把 attention 讨论完了。一个 decoder layer 里除了 attention 和标准 GEMM，剩下的是一堆"小 kernel"：RMSNorm、RoPE、SiLU-mul、把 KV 写进分页 cache、把权重从 INT4 解开、把激活压成 FP8、MoE 的 token 重排、采样。它们单个都不复杂，但数量多、变化快，加起来占掉推理时间的一个可观比例——而且是 vLLM、SGLang 这些项目里 PR 最活跃的区域。

这一篇把它们放在同一个方法论下过一遍：**先算这个 kernel 理论上要搬多少字节、做多少 FLOPs，再看实现，再解释差距。**核心问题是总纲提出的那个：

> **一个 INT4 weight-only GEMM，decode 时比 BF16 快 3 倍，prefill 时反而慢。用 Roofline 解释这个现象。**

答案其实只有一句话——decode 在 Roofline 的 memory-bound 一侧，prefill 在 compute-bound 一侧，而 INT4 只减少字节、不减少 FLOPs、还额外增加反量化指令——但要把这句话说清楚，需要先知道 INT4 在寄存器里长什么样、怎么解开、解开之后怎么喂给 Tensor Core。所以本篇从格式开始。

量化算法本身（怎么校准、怎么选 scale、掉多少精度）不在范围内；本篇只讨论给定格式下 kernel 怎么写、它的收益在什么区间成立。硬件基线仍是 A100 SXM 80GB（HBM2e 约 2.0 TB/s、BF16 Tensor Core 约 312 TFLOPS，均为标称值），Hopper 数字随文标注 H100（约 3.35 TB/s、BF16 约 989 TFLOPS、FP8 约 1979 TFLOPS）。vLLM 源码以 v0.20.0 为准。


## 一、低精度格式在 kernel 层的含义

### 1. 位布局与动态范围

对写 kernel 的人来说，一个格式只有三件事要紧：多少位、多少位给指数（决定动态范围）、多少位给尾数（决定相对精度）。把常见格式排在一起：

```text
格式        位  符号/指数/尾数   最大正值          最小正规范值      相对精度 eps (2^-mantissa)
FP32        32  1 / 8 / 23      3.4e38            1.2e-38           1.2e-7
FP16        16  1 / 5 / 10      65504             6.1e-5            9.8e-4
BF16        16  1 / 8 / 7       3.4e38            1.2e-38           7.8e-3
FP8 E4M3    8   1 / 4 / 3       448（无 inf）      1.6e-2（2^-6）     0.125
FP8 E5M2    8   1 / 5 / 2       57344             6.1e-5（2^-14）    0.25
INT8        8   有符号整数        127               —（均匀间隔 1）    绝对误差 0.5 LSB
INT4        4   有符号整数        7（或 0..15 + zp） —                16 个电平
```

几个要点：

- **BF16 与 FP16 的取舍**：BF16 拿 FP32 的高 16 位，指数范围与 FP32 相同，所以不会像 FP16 那样在 65504 处溢出；代价是尾数只有 7 位，相对精度 $$2^{-7} \approx 0.8\%$$。LLM 推理里激活的 outlier 可以到几百甚至上千，所以 BF16 成为默认。
- **E4M3 与 E5M2**：E4M3 的最大值只有 448，并且 **没有 inf**——指数尾数全 1 的编码被拿去表示 NaN，换来多一个可表示的数（448 而不是 240）。这是 NVIDIA 的 `e4m3fn` 变体（fn = finite + NaN），PyTorch 的 `torch.float8_e4m3fn` 与 CUDA 的 `__nv_fp8_e4m3` 都是它。E5M2 就是 FP16 砍掉 8 位尾数，有 inf/NaN，范围到 57344，但相对精度只有 25%。工程上 E4M3 用于权重与前向激活，E5M2 主要出现在训练梯度里。
- **INT8 / INT4** 本身没有动态范围概念，动态范围全部由 scale 承担：真实值 $$x \approx s \cdot (q - z)$$，$$s$$ 是浮点 scale，$$z$$ 是零点。INT4 只有 16 个电平，所以 scale 的粒度必须很细（per-group，通常 128 个元素一组），否则一个 outlier 会把整组的分辨率吃掉。

### 2. 累加精度

低精度格式说的是**输入**的精度。乘累加的中间结果用什么精度，是另一件事：

- Ampere/Hopper 的 Tensor Core 做 FP16/BF16 输入的 `mma` 时，累加器可以是 FP32（这是默认，也是 cuBLAS 的做法）；FP16 输入还可以选 FP16 累加，快一点但对长 K 的 reduction 不安全。BF16 输入没有 BF16 累加的选项。
- Hopper 的 FP8 `wgmma` 累加器是 FP32 寄存器，但硬件内部的累加精度**有限**——DeepSeek-V3 技术报告（2024）指出 Hopper FP8 Tensor Core 的累加大约保留 14 位尾数精度，对 K=4096 这种长 reduction 会引入可见误差。这是后面"per-block 128×128 scale 为什么要 promote 到 FP32 累加"的原因。
- INT8 Tensor Core 用 INT32 累加，没有精度损失（$$127 \times 127 \times 4096 \approx 6.6 \times 10^7$$，远小于 $$2^{31}$$）。

一句话：**输入可以低精度，累加尽量 FP32。**这也是本系列所有 CUDA 示例"输入 BF16、累加 float"约定的来源。

### 3. CUDA 类型与转换指令

`cuda_bf16.h` 提供 `__nv_bfloat16` 与它的双元素打包 `__nv_bfloat162`；`cuda_fp16.h` 提供 `__half` 与 `__half2`；`cuda_fp8.h`（CUDA 11.8+）提供 `__nv_fp8_e4m3`、`__nv_fp8_e5m2` 及打包 `__nv_fp8x2_e4m3`、`__nv_fp8x4_e4m3`，底层存储是 `__nv_fp8_storage_t`（一个 `uint8_t`）。

常用转换 intrinsic：

```cpp
float f = __bfloat162float(b);                 // BF16 -> FP32：只是移位，几乎免费
__nv_bfloat16 b = __float2bfloat16(f);         // FP32 -> BF16：round-to-nearest-even
float2 f2 = __bfloat1622float2(b2);            // 一对
__nv_bfloat162 b2 = __float22bfloat162_rn(f2); // 一对（sm_80+ 有单条 cvt.rn.bf16x2.f32）
// FP8：一次转两个
__nv_fp8x2_storage_t p = __nv_cvt_float2_to_fp8x2(f2, __NV_SATFINITE, __NV_E4M3);
__half2_raw h2 = __nv_cvt_fp8x2_to_halfraw2(p, __NV_E4M3);
```

`__NV_SATFINITE` 表示超出 448 的值饱和到 448 而不是变 NaN——量化 kernel 一律用它。对应的 PTX 是 `cvt.rn.satfinite.e4m3x2.f32 d, a, b;`（sm_89+，一条指令转两个 FP32 到一对 FP8）和 `cvt.rn.f16x2.e4m3x2`（反向）。Ampere 没有 FP8 硬件转换，`__nv_cvt_*` 在 sm_80 上会展开成一串整数位运算，慢一个数量级——这是"Ampere 上 FP8 只能当存储格式用"的第一个原因。

vLLM 的 FP8 量化落到 `csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh` 里的 `__nv_cvt_float_to_fp8(a, __NV_SATFINITE, fp8_type)`，饱和之前先用 `fmaxf/fminf` 把值夹到 $$\pm 448$$（`csrc/quantization/w8a8/fp8/common.cuh` 的 `scaled_fp8_conversion`）。

### 4. INT4 打包与"magic number"反量化

INT4 没有 CUDA 类型。8 个 INT4 装进一个 `uint32_t`，第 $$j$$ 个元素占 bit $$4j$$ 到 $$4j+3$$。最朴素的解包是 8 次移位 + 掩码 + 8 次 `cvt` 整数转浮点 + 8 次减零点 + 8 次乘 scale——每个元素约 4 条指令，在 compute-bound 的 tile 里完全不可接受。

Marlin 和 FasterTransformer 用的技巧是**不做整数到浮点的 `cvt`，而是直接把 4 位塞进一个 FP16 的尾数**。FP16 的 1024（十进制）编码是 `0x6400`：指数 $$2^{10}$$，尾数全 0。若把一个 4 位整数 $$q$$（0 到 15）放进这个编码的尾数低 4 位，得到 `0x6400 | q`，它表示的 FP16 值恰好是 $$1024 + q$$（因为尾数最低位的权重是 $$2^{10} \times 2^{-10} = 1$$）。于是 $$q = \text{fp16}(0x6400 \mid q) - 1024$$，两个 FP16 打包在一个 32 位寄存器里，一次处理两个元素：

```cpp
// csrc/quantization/marlin/dequant.h（v0.20.0）：把 int32 里的 8 个 INT4 解成 4 个 half2
// 这里的 lop3 是 PTX 的三输入逻辑运算：一条指令完成 (q & MASK) | EX
template <>
__device__ inline void dequant<half2, vllm::kU4B8.id(), false>(int q,
                                                               half2* frag_b) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  // We want signed int4 outputs, hence we fuse the `-8` symmetric zero point
  // directly into `SUB` and `ADD`.
  const int SUB = 0x64086408;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd480d480;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&SUB));
  frag_b[1] = __hfma2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&MUL),
                      *reinterpret_cast<const half2*>(&ADD));
}
```

逐行看：`lo` 取每个 16 位半字的低 4 位（第 0、4 个元素），或上 `0x6400` 得到 $$1024 + q$$；`SUB = 0x6408` 是 FP16 的 $$1032 = 1024 + 8$$，一次 `__hsub2` 同时完成"减 1024"和"减零点 8"。`hi` 取 bit 4 到 7（第 1、5 个元素），它们在尾数里的位置高了 4 位，表示的值是 $$1024 + 16q$$；`MUL = 0x2c00` 是 FP16 的 $$1/16$$，`ADD = 0xd480` 是 $$-72 = -(1024/16 + 8)$$，一次 `__hfma2` 完成 $$(1024 + 16q)/16 - 72 = q - 8$$。**四个元素、四条指令**（两条 `lop3`、一条 `hsub2`、一条 `hfma2`，均摊每元素一条），再乘 scale 是每两个元素一条 `hmul2`。BF16 版本同理，只是"magic"常数换成 `0x4300`（BF16 的 128，尾数 7 位，最低位权重 1）。这一招只用整数逻辑与 FP16 算术，比逐元素的 `cvt` + 减法 + 乘法（每元素 3 到 4 条）快得多，且不占用 Tensor Core。

注意它假设权重的位排列已被预先打乱成"第 0、4 个元素在低半字，第 1、5 个在高半字"这种交错顺序——这就是 Marlin 需要 **repack** 步骤的原因之一。


## 二、Weight-only 量化 GEMM（W4A16）

### 1. 结构：INT4 tile → 寄存器解包 → mma fragment

W4A16 的含义是：权重（W）4 位，激活（A）16 位，Tensor Core 做的仍然是 BF16/FP16 的 `mma`。所以 kernel 的主循环比第六篇的 BF16 GEMM 多了一段：

```text
BF16 GEMM 主循环                  W4A16 GEMM 主循环
cp.async A tile -> smem            cp.async A tile -> smem
cp.async B tile (BF16) -> smem     cp.async B tile (INT4, 字节数 /4) -> smem
ldmatrix A, B -> 寄存器 fragment   ldmatrix A -> fragment; ld.shared B (uint32 打包)
                                   寄存器内 dequant：lop3 + hsub2/hfma2 -> BF16 fragment
                                   乘 group scale（每 128 个 k 换一次）
mma.sync                           mma.sync
```

权重的元数据：per-group scale 是 `[K/g, N]` 的 FP16/BF16 矩阵（$$g = 128$$ 时每 128 个权重多 2 字节，即 0.125 bit/权重），可选的 per-group zero point 是 `[K/g, N/8]` 的 INT4 打包（再加 0.03 bit/权重）。所以 W4 g128 实际约 **4.15 bit/权重**，是 BF16 的 26%。

### 2. Roofline 两侧：核心问题的答案

回到第一篇的 Roofline：一个 kernel 的时间是

$$
t = \max\left( \frac{\text{FLOPs}}{P_{peak}},\ \frac{\text{Bytes}}{BW} \right)
$$

一次 $$[M, K] \times [K, N]$$ GEMM 的 FLOPs 是 $$2MKN$$，与 $$M$$ 线性；字节数是权重 $$KN \cdot b_w$$ 加激活 $$2MK + 2MN$$（$$b_w$$ 为每个权重的字节数），其中权重项与 $$M$$ 无关。算术强度：

$$
I(M) = \frac{2MKN}{KN \cdot b_w + 2M(K + N)} \approx \frac{2M}{b_w} \quad (M \ll K, N)
$$

BF16 时 $$b_w = 2$$，$$I \approx M$$ FLOP/byte；INT4 时 $$b_w \approx 0.5$$，$$I \approx 4M$$。A100 的 BF16 ridge point 是 $$312 / 2.0 \approx 156$$ FLOP/byte。

**decode（M = batch，通常 1 到几十）**：$$I \ll 156$$，两者都在 memory-bound 一侧，时间由权重字节决定。Llama-3-8B 的 GEMM 权重约 7.5B 参数：BF16 是 15 GB（加 embedding 与 lm_head 共约 16 GB），INT4 g128 若全部量化约 4.1 GB。A100 上读一遍的下界：

$$
t_{BF16} \approx \frac{16\ \text{GB}}{2.0\ \text{TB/s}} = 8\ \text{ms}, \qquad t_{INT4} \approx \frac{4.1\ \text{GB}}{2.0\ \text{TB/s}} \approx 2\ \text{ms}
$$

H100 上对应 4.8 ms 与 1.2 ms。理论上界是 4 倍（更准确是 $$16 / 4.15 \approx 3.9$$），实际因为 lm_head 通常不量化、KV cache 读取不变、小 M 下 kernel 效率不到峰值带宽，落到 **2.5 到 3.5 倍**——这就是"快 3 倍"。

**prefill（M = 数千）**：$$I \approx M \gg 156$$，两者都在 compute-bound 一侧。此时时间由 $$2MKN / P_{peak}$$ 决定——**INT4 不减少 FLOPs**，Tensor Core 做的仍是 BF16 `mma`，权重字节少读的那 24 MiB（对 4096×4096）在 0.44 ms 的计算时间面前只值 12 µs。而反量化是纯额外指令：每个 BF16 权重元素进入 `mma` 之前，都要在寄存器里被 `lop3` + `hfma2` + `hmul2` 处理一次，而且是**每被一个 block 加载一次就处理一次**——权重 tile $$BN \times BK$$ 被 $$M / BM$$ 个 block 各加载一遍，所以反量化的总指令数约 $$KN \cdot (M / BM) \cdot c$$（$$c \approx 1$$ 条/元素）。$$M = 4096$$、$$BM = 64$$ 时是 $$4096^2 \times 64 \approx 1.1$$ G 条整数/FP16 指令，与 `ldmatrix`、`mma` 抢同一个 warp 调度器的 issue slot。加上 INT4 kernel 的 tile 配置是为小 M 调的（Marlin 的 `thread_m_blocks` 最大 4，即 $$BM \le 64$$），大 M 下比 cuBLAS 为大 tile 优化的 BF16 kernel 慢 10% 到 30% 是常见的。

**交叉点**在哪里？INT4 GEMM 从 memory-bound 变 compute-bound 的 M 满足 $$4M \approx 156$$，即 $$M \approx 40$$；BF16 的交叉点是 $$M \approx 156$$。在 $$M < 40$$ 时 INT4 的收益随 M 减小而增大（趋近 4 倍）；$$40 < M < 156$$ 时 INT4 已经 compute-bound 而 BF16 还在 memory-bound，两者接近；$$M > 156$$ 时两者都 compute-bound，INT4 因反量化开销略慢。H100 上 ridge 是 295，交叉点约 $$M \approx 74$$。

所以答案是：**INT4 weight-only 量化是一个"把字节换指令"的交易，只在 memory-bound 一侧划得来。**这也解释了为什么推理引擎会按 M 选不同的 kernel（vLLM 的 AWQ 非 Marlin 路径在 token 数达到 256 时改为先把权重整体反量化成 FP16、再走 cuBLAS），以及为什么 FP8 W8A8 这种"FLOPs 也减半"的方案在 Hopper 上更受欢迎。

### 3. Marlin 的结构

Marlin（Frantar 等 2024，"Mixed Auto-Regressive Linear kernel"）是专门为 $$M \le 64$$ 的小 batch 设计的 W4A16 kernel，目标是在 M 从 1 增长到 16 甚至 32 时**仍然保持接近 4 倍的加速**——朴素的反量化 GEMV 在 M = 1 时能接近带宽上限，但 M 一到 8 就掉下来。它的做法可以概括为六点：

1. **权重预先重排（repack）**成 `mma` fragment 友好的布局：一个 `int4`（16 字节）从 shared memory 读出来直接就是某个 warp 某条 `mma.m16n8k16` 需要的 B fragment 的打包形式，不需要 `ldmatrix` 转置，也不需要在寄存器间 shuffle。`csrc/quantization/marlin/gptq_marlin_repack.cu` 与 `awq_marlin_repack.cu` 做这件事，一次性离线完成。
2. **`cp.async` 多 stage 流水**（默认 4 stage）：全局到 shared 的拷贝异步进行，同时 shared 到寄存器（`ldsm` 即 `ldmatrix`，用于 A）和 `mma` 并行；`cp_async_wait<stages - 2>` 控制流水深度。
3. **反量化在寄存器中**完成，就是 §1.4 那段 `lop3` + `hfma2`，紧挨着 `mma` 发射。
4. **striped partitioning**：把 B 矩阵按"条"分给 block，每个 block 处理一条覆盖多个列 slice 的斜带，保证 108 个 SM 在各种 N、K 形状下都有活干，同时尽量少做跨 block 的归约。
5. **global reduce**：当一个列 slice 被多个 block 分摊 K 维时，用一个全局 `locks` 数组做 barrier，后到的 block 把部分和累加到先到者的输出上（有 FP16 与 FP32 两个版本）。
6. **小 M 特化**：M ≤ 8 时用 `m_block_size_8` 走 `mma_trans` 路径，让一条 `m16n8k16` 的 16 行不至于浪费一半。

kernel 的签名与作者对 striped partitioning 的注释：

```cpp
// csrc/quantization/marlin/marlin_template.h（v0.20.0），摘要
template <const vllm::ScalarTypeId a_type_id, const vllm::ScalarTypeId b_type_id,
          const vllm::ScalarTypeId c_type_id, const vllm::ScalarTypeId s_type_id,
          const int threads,          // number of threads in a threadblock
          const int thread_m_blocks,  // number of 16x16 blocks in the m dimension
          const int thread_n_blocks,  // same for n dimension (output)
          const int thread_k_blocks,  // same for k dimension (reduction)
          const bool m_block_size_8,  // whether m_block_size == 8
          const int stages,           // number of stages for the async global->shared
          const int group_blocks,     // number of consecutive 16x16 blocks
                                      // with a separate quantization scale
          const bool is_zp_float>
__global__ void Marlin(
    const int4* __restrict__ A0,  // fp16 input matrix of shape mxk
    const int4* __restrict__ B,   // 4bit quantized weight matrix of shape kxn
    int4* __restrict__ C0,        // fp16 output buffer of shape mxn
    int4* __restrict__ C_tmp,     // fp32 tmp output buffer (for reduce)
    /* ... bias, a_scales, scales_ptr, zp_ptr, g_idx ... */
    int prob_m, int prob_n, int prob_k, int lda,
    int* locks,      // extra global storage for barrier synchronization
    bool has_bias, bool use_atomic_add, bool use_fp32_reduce, int max_shared_mem) {
  // Each threadblock processes one "stripe" of the B matrix with (roughly) the
  // same size, which might involve multiple column "slices" (of width 16 *
  // `thread_n_blocks`). Stripes are defined as shown in the 3x3 matrix 5 SM
  // example:
  //   0 1 3
  //   0 2 3
  //   1 2 4
  // While this kind of partitioning makes things somewhat more complicated, it
  // ensures good utilization of all SMs for many kinds of shape and GPU
  // configurations, while requiring as few slow global cross-threadblock
  // reductions as possible.
```

主循环是一个双层展开的流水：外层 `pipe` 遍历 stage，内层 `k` 遍历一个 stage 内的 `mma` 步；每步先把下一步的 B 打包数据、scale、zero point 从 shared 取到寄存器（`fetch_to_registers`），在倒数第二步发起下一个 stage 的 `cp.async`（`fetch_to_shared`），然后调用 `matmul(k, pipe)`——在那里面完成 `dequant_data` → `scale` → `mma`。同一份文件 `marlin.cu` 的 host 侧为小 batch 与大 batch 各准备了一组 `(thread_k, thread_n, num_threads)` 配置（如 `{128, 128, 256}` 与 `{64, 256, 256}`），并把 M 按 64 行一段切开重复调用。

### 4. GPTQ、AWQ 与 Marlin 的关系；Machete

GPTQ 与 AWQ 是两种**量化算法**，各自带着一个参考 kernel：`csrc/quantization/gptq/q_gemm.cu`（源自 exllama，`qdq_4.cuh` 里有自己的 4 位解包）与 `csrc/quantization/awq/gemm_kernels.cu`（源自 llm-awq，`dequantize.cuh` 同样用 `lop3` + magic number）。它们的权重打包格式与 Marlin 不同，但数学上都是 per-group scale（AWQ 还有 per-group zero point）的 INT4。vLLM 的做法是：**加载时把 GPTQ/AWQ 格式的权重 repack 成 Marlin 布局，然后统一走 Marlin kernel**（`gptq_marlin_repack.cu`、`awq_marlin_repack.cu`），只有 Marlin 不支持的组合（如 Turing 之前的架构、特殊的 act-order 情况）才回落到原始 kernel。

Machete 是 Hopper 上的 Marlin 继任者：基于 CUTLASS 3.x 的 mixed-input GEMM，用 `wgmma` + TMA，同样要求权重 prepack（`csrc/quantization/machete/`，`machete_prepacked_layout.cuh` 定义布局，`generate.py` 生成各类型组合的实例）。它的 Readme 一句话概括了做的事：`out = (w_q.to(scale_type) * w_s - w_z.to(scale_type)) @ a`。

### 5. `csrc/quantization/` 目录导读

```text
目录 / 文件                                   内容
marlin/                                     Marlin W4A16 / W8A16（marlin_template.h 主体，dequant.h 反量化，
                                            gptq_marlin_repack.cu / awq_marlin_repack.cu 权重重排，
                                            marlin_int4_fp8_preprocess.cu W4A8-FP8 预处理）
gptq/                                       exllama 系 GPTQ 参考 kernel（q_gemm.cu，qdq_{2,3,4,8}.cuh）
awq/                                        llm-awq 参考 kernel（gemm_kernels.cu，dequantize.cuh）
gptq_allspark/                              W8A16（allspark_qgemm_w8a16.cu）
machete/                                    Hopper CUTLASS 3.x mixed-input GEMM（prepack + mainloop + 生成脚本）
gguf/                                       llama.cpp 的 GGUF 量化格式（Q4_0/Q4_K/... mmvq.cuh 为 GEMV，mmq.cuh 为 GEMM）
w8a8/fp8/                                   FP8 量化 kernel：common.cu（static / dynamic per-token scaled_fp8_quant），
                                            nvidia/quant_utils.cuh（__nv_cvt 封装）
w8a8/int8/                                  INT8 量化：scaled_quant.cu（含非对称 azp）
w8a8/cutlass/                               v0.20.0 里只剩 Epilogues.md；CUTLASS scaled_mm 实现移到
                                            csrc/libtorch_stable/quantization/w8a8/cutlass/{c2x, c3x, moe}/
fused_kernels/                              fused_layernorm_dynamic_per_token_quant.cu（RMSNorm + 动态量化），
                                            fused_silu_mul_block_quant.cu（SiLU-mul + 分块量化）
hadamard/                                   hadacore：Hadamard 变换（旋转量化方案 QuaRot / SpinQuant 用）
activation_kernels.cu                       量化版激活（如 silu_and_mul 直接输出 FP8/NVFP4）
utils.cuh                                   quant_type_max_v（E4M3 为 448、INT8 为 127）、min_scaling_factor
```

（`csrc/cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp` 是 FP8/INT8 GEMM 的 epilogue 定义，下一节用到。）


## 三、FP8 GEMM（W8A8）与动态量化

### 1. FP8 Tensor Core：字节与 FLOPs 同时减半

W8A8 与 W4A16 是不同的交易。权重与激活都是 FP8，Tensor Core 直接吃 FP8 输入（Hopper `wgmma` 支持 `e4m3` × `e4m3` → FP32 累加），标称算力 1979 TFLOPS 是 BF16 的两倍。于是：

- decode：权重字节减半（对比 INT4 的减到 1/4），时间下界减半；
- **prefill：FLOPs 时间也减半**，这是 W4A16 做不到的。

代价是激活也要量化——它是运行时才知道的，所以需要一个**动态量化 kernel**（§5），并且 FP8 E4M3 只有 3 位尾数，对激活里的 outlier 敏感，所以 scale 的粒度成为关键。

### 2. scale 的三种粒度与 epilogue

量化后的 GEMM 是

$$
D_{ij} = s_a(i) \cdot s_b(j) \cdot \sum_k A^q_{ik} B^q_{kj}
$$

只要 scale 只依赖 $$i$$ 或只依赖 $$j$$（不依赖 $$k$$），它就可以从求和号里提出来，**在 epilogue 里乘回去**——主循环仍然是纯 FP8 `mma`，累加器是 FP32，最后每个输出元素乘两个数再转 BF16。三种粒度：

- **per-tensor**：$$s_a$$、$$s_b$$ 各一个标量；
- **per-token**（激活按行）：$$s_a$$ 是长度 $$M$$ 的向量，每个 token 一个 scale——这是动态量化的自然粒度，因为一个 token 的激活是一次 reduction 就能拿到 absmax 的单位；
- **per-channel**（权重按列）：$$s_b$$ 是长度 $$N$$ 的向量，每个输出通道一个 scale，离线算好。

CUTLASS 3.x 用 Epilogue Visitor Tree（EVT）表达这件事。vLLM 的定义：

```cpp
// csrc/cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp（v0.20.0）
// D = (a_scales * A) (b_scales * B)，scale 可为 per-tensor 或 per-row/col，numpy 广播语义
template <typename ElementAcc, typename ElementD, typename TileShape>
struct ScaledEpilogue
    : private ScaledEpilogueBase<ElementAcc, ElementD, TileShape> {
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoad<float>;  // 按行广播（每个 i 一个）
  using ScaleB = typename SUPER::template RowOrScalarLoad<float>;  // 按列广播（每个 j 一个）

  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTCompute0 = cutlass::epilogue::fusion::Sm90EVT<Compute0, ScaleB, Accum>;

  using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, ElementD, float, cutlass::FloatRoundStyle::round_to_nearest>;
 public:
  using EVTCompute = cutlass::epilogue::fusion::Sm90EVT<Compute1, ScaleA, EVTCompute0>;
};
```

树的叶子是 `Accum`（累加器 tile）、`ScaleB`（一行 float，按列广播）、`ScaleA`（一列 float，按行广播），两层 `multiplies` 节点依次乘上去，最外层转成 `ElementD`。同一文件里 `ScaledEpilogueBias`、`ScaledEpilogueBiasAzp` 在此基础上加 bias 与 INT8 非对称零点修正。`Sm90ColBroadcast` / `Sm90RowBroadcast` 负责把那一列/一行 scale 从全局内存搬进来——这一段流量是 $$4(M + N)$$ 字节，与 $$2MN$$ 的输出相比可以忽略。

### 3. per-block 128×128 scale：为什么不能放 epilogue

DeepSeek-V3 用的是更细的粒度：激活 per-token 每 128 个元素一组（$$1 \times 128$$），权重 $$128 \times 128$$ 一块一个 scale。此时 $$s_b$$ 依赖 $$k$$ 所在的块：

$$
D_{ij} = \sum_{b} s_a(i, b) \cdot s_b(b, j) \cdot \sum_{k \in \text{block } b} A^q_{ik} B^q_{kj}
$$

内层求和只在一个 128 宽的 K 块内做，**每个块的部分和要先乘上该块的 scale 再累加到总和**——这一步必须在主循环里、每 128 个 K 做一次，不能推到 epilogue。它带来两个后果：一是 K 循环里多了一次"累加器 × scale 再加到另一组累加器"的 FP32 运算，需要两套累加寄存器；二是这恰好给了一个机会把 `wgmma` 的有限精度累加"promote"到真正的 FP32——每 128 个 K，把 Tensor Core 累加器的值乘 scale 后加进 CUDA Core 维护的 FP32 累加器，然后清零 Tensor Core 累加器重新开始。DeepGEMM（DeepSeek 2025）用 JIT 生成针对具体形状的 kernel、TMA 搬运、`wgmma` 异步发射、warp specialization（producer warp 做 TMA，consumer warpgroup 做 `wgmma` + promote）来实现它；vLLM 在 `csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm90_fp8.cu` 里用 CUTLASS 的 blockwise scaling mainloop 做了同样的事。

### 4. Ampere 的 fallback

A100 没有 FP8 Tensor Core。FP8 权重在 Ampere 上只能是**存储格式**：加载时用位运算转成 BF16（`__nv_cvt_fp8_to_halfraw` 的软件展开，或用与 INT4 类似的 magic-number 技巧——Marlin 的 `dequant<..., kFE4M3fn>` 就是这样做的，把 FP8 的指数尾数移进 FP16 的位域再乘一个 $$2^{\Delta e}$$ 的修正），然后走 BF16 `mma`。这时它是一个 W8A16 kernel，收益只有 decode 侧的字节减半，没有算力侧的收益。vLLM 的 `scaled_mm_c2x_sm89_fp8_dispatch.cuh` 面向 Ada（sm_89，有 FP8 Tensor Core 但没有 `wgmma`），sm_80 的 FP8 走 Marlin 的 W8A16 路径。

### 5. 动态量化 kernel：absmax reduction + cast，以及与 RMSNorm 的融合

per-token 动态量化就是第四篇 reduction 的一个变体：一行 $$d$$ 个元素，先归约出 $$\max_k \lvert x_k \rvert$$，算 $$s = \max / 448$$（或 $$/127$$），再把每个元素除以 $$s$$、饱和、转 FP8 写出。vLLM 的实现：

```cpp
// csrc/quantization/w8a8/fp8/common.cu（v0.20.0）
template <typename scalar_t, typename fp8_type>
__global__ void dynamic_per_token_scaled_fp8_quant_kernel_strided(
    fp8_type* __restrict__ out, float* __restrict__ scale,
    const scalar_t* __restrict__ input, const float* __restrict__ scale_ub,
    int hidden_size, int64_t in_row_stride, int64_t out_row_stride) {
  const int64_t token_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const scalar_t* token_in = input + token_idx * in_row_stride;
  fp8_type* token_out = out + token_idx * out_row_stride;

  // 1) per-token absmax
  float absmax_val = 0.f;
  vectorize_read_with_alignment<16>(
      token_in, hidden_size, tid, blockDim.x, [&] __device__(scalar_t v) {
        absmax_val = fmaxf(absmax_val, fabsf(static_cast<float>(v)));
      });
  using BlockReduce = cub::BlockReduce<float, 256>;
  __shared__ typename BlockReduce::TempStorage tmp;
  const float block_max =
      BlockReduce(tmp).Reduce(absmax_val, CubMaxOp{}, blockDim.x);

  __shared__ float token_scale;
  if (tid == 0) {
    token_scale = scale_ub ? fminf(block_max, *scale_ub) : block_max;
    token_scale = fmaxf(token_scale / quant_type_max_v<fp8_type>,
                        min_scaling_factor<fp8_type>::val());
    scale[token_idx] = token_scale;
  }
  __syncthreads();

  // 2) quantize
  vectorize_with_alignment<16>(
      token_in, token_out, hidden_size, tid, blockDim.x,
      [=] __device__(fp8_type & dst, const scalar_t& src) {
        dst = scaled_fp8_conversion<false, fp8_type>(static_cast<float>(src),
                                                     token_scale);
      });
}
```

一行一个 block，两遍读（第二遍在 L1/L2 里）、一遍写。`scale_ub` 是激活 scale 的上限（防止某个 token 的 outlier 把 scale 拉得过大），`min_scaling_factor` 防止全零行除零。

**算字节**。在 decoder layer 里，动态量化紧跟在 RMSNorm 之后（attention 与 FFN 的输入都是 norm 的输出）。设一行 $$d$$ 个 BF16 元素、$$T$$ 行：

```text
                       读                          写                       合计 / 元素
分开：RMSNorm           x (2B)                      xn (2B)                  4 B
      + 动态量化         xn (2B)                     x_fp8 (1B) + scale        3 B
                                                                            7 B
融合：norm + quant      x (2B)                      x_fp8 (1B) + scale        3 B
```

融合节省 4/7 的流量，并且不需要物化 BF16 的中间张量。vLLM 的 `csrc/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu` 就是这个融合，结构是三段：`compute_rms`（sum of squares 归约）→ `compute_dynamic_per_token_scales`（对**归一化后**的值 $$\text{bf16}(x \cdot \text{rms}) \cdot w$$ 做 absmax 归约，注意先转一次 BF16 再乘 $$w$$，让 scale 与非融合路径逐位一致）→ `norm_and_quant`（第三遍读 $$x$$，归一化、除 scale、`cvt`、写 FP8）。三遍读的后两遍在 L1/L2 里，HBM 流量还是上表的 3 B/元素。它还带 `has_residual` 模板参数，把 residual add 也吸进来（见下一章）。


## 四、融合 kernel 的常见模式

### 1. 融合为什么赢：只有字节

这一章的所有 kernel 都是 memory-bound 的 elementwise 或 row-wise 操作，算术强度在 1 FLOP/byte 以下（回顾：BF16 的 $$y = x + b$$ 每元素 6 字节、1 FLOP，$$1/6$$ FLOP/byte，与 A100 的 ridge 156 差三个数量级）。它们的理论时间就是字节数除以带宽，融合唯一的目的是**减少往返 HBM 的字节数**——中间结果留在寄存器里，不写出再读回。所以每种模式先列字节表。

### 2. bias + activation 与 SiLU-and-mul

最简单的模式：GEMM 输出 $$y$$，接 $$\text{act}(y + b)$$。分开做是 GEMM 写 $$y$$（2 B）、bias kernel 读写（4 B）、act kernel 读写（4 B），融合进 GEMM 的 epilogue 之后只有 GEMM 写一次（2 B）。这在 CUTLASS 里是 epilogue functor 的事，属于第六篇。

Llama 系模型的 FFN 用 SwiGLU：`gate_up` GEMM 一次输出 $$[T, 2d_{ff}]$$（前一半是 gate，后一半是 up），然后

$$
h = \text{silu}(x[:, :d]) \odot x[:, d:], \qquad \text{silu}(z) = \frac{z}{1 + e^{-z}}
$$

输入 $$[T, 2d]$$、输出 $$[T, d]$$，每个输出元素读 2 个写 1 个，**6 B/元素、约 5 FLOP/元素**（一个 exp、一个除、一个乘、几个加），memory-bound。Llama-3-8B 的 $$d_{ff} = 14336$$，$$T = 8192$$ 时读 448 MiB 写 224 MiB，A100 下界约 350 µs。vLLM 的实现是一个模板 `act_and_mul_kernel<scalar_t, packed_t, ACT_FN, PACKED_ACT_FN, act_first, use_vec, HAS_CLAMP, use_256b>`，一个 token 一个 block，向量化时每线程一次读 16 字节（8 个 BF16）的 gate 与 up，用 `packed_t`（`__nv_bfloat162`）两两计算：

```cpp
// csrc/activation_kernels.cu（v0.20.0），摘要
template <typename scalar_t, typename packed_t,
          scalar_t (*ACT_FN)(const scalar_t&),
          packed_t (*PACKED_ACT_FN)(const packed_t&), bool act_first,
          bool use_vec, bool HAS_CLAMP, bool use_256b = false>
__global__ void act_and_mul_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    const int d, const float limit) {
  const scalar_t* x_ptr = input + blockIdx.x * 2 * d;
  const scalar_t* y_ptr = x_ptr + d;
  scalar_t* out_ptr = out + blockIdx.x * d;
  if constexpr (use_vec) {
    using pvec_t = PackedVec<cuda_t, use_256b>;
    const int num_vecs = d / 2 / pvec_t::NUM_ELTS;
    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
      pvec_t x, y;
      ld128(x, &x_vec[i]); ld128(y, &y_vec[i]);
#pragma unroll
      for (int j = 0; j < pvec_t::NUM_ELTS; j++)
        x.elts[j] = packed_compute<packed_t, PACKED_ACT_FN, act_first, HAS_CLAMP>(
            x.elts[j], y.elts[j], limit);
      st128(x, &out_vec[i]);
    }
  } else { /* 标量回退：d 不对齐时 */ }
}

template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
  return (T)(((float)x) / (1.0f + expf((float)-x)));   // x * sigmoid(x)
}
```

`act_first` 区分 `silu(gate) * up` 与 `gate * silu(up)`（有些模型顺序相反），`HAS_CLAMP` 是 gpt-oss 需要的预裁剪，同一模板实例化出 `silu_and_mul`、`gelu_and_mul`、`gelu_tanh_and_mul` 等。host 侧 `silu_and_mul()` 就一行：`LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel, vllm::packed_silu_kernel, true, false, 0.0f)`。

### 3. residual add + RMSNorm

Pre-norm Transformer 的每个子层是 $$h \leftarrow h + f(\text{norm}(h))$$，两个子层之间的模式是"上一个子层的输出加到残差流、再 norm 给下一个子层"。分开与融合的字节数（$$d$$ 个 BF16 一行）：

```text
                    读                      写                       合计 / 元素
分开：add           x, residual (4B)        residual (2B)            6 B
      RMSNorm       residual (2B)           out (2B)                 4 B
                                                                     10 B
融合                x, residual (4B)        residual, out (4B)       8 B
```

融合只省 20%，但更重要的是省了一次 kernel launch 和一次 $$T \times d$$ 的中间张量物化。vLLM 的 `fused_add_rms_norm_kernel`（`csrc/layernorm_kernels.cu`）签名是 `(input, residual, weight, epsilon)`，**就地**：`residual = input + residual`，`input = norm(residual) * weight`——两个输出都写回输入缓冲区，所以调用者不需要分配任何新张量。宽度 8 的 BF16/FP16 特化版每线程一次搬 16 字节：

```cpp
// csrc/layernorm_kernels.cu（v0.20.0），width > 0 的特化
  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    int64_t strided_id = blockIdx.x * vec_input_stride + idx;
    _f16Vec<scalar_t, width> temp = input_v[strided_id];
    temp += residual_v[id];
    variance += temp.sum_squares();
    residual_v[id] = temp;
  }
  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    int64_t strided_id = blockIdx.x * vec_input_stride + idx;
    _f16Vec<scalar_t, width> res = residual_v[id];
    _f16Vec<scalar_t, width> w = weight_v[idx];
    _f16Vec<scalar_t, width> out;
    for (int j = 0; j < width; ++j) {
      float x = Converter::convert(res.data[j]);
      float wf = Converter::convert(w.data[j]);
      out.data[j] = Converter::convert(x * s_variance * wf);
    }
    input_v[strided_id] = out;
  }
```

注意两个细节：方差用的是**写回 BF16 之后**的 `temp`（`sum_squares` 在 `_f16Vec` 上算，先转 float），第二遍从 `residual_v` 重新读——这一读命中 L1/L2，不增加 HBM 流量，但省下了把一行 4096 个值留在寄存器里的压力。第八章的实现沿用这个结构。

### 4. RoPE：两种布局，就地旋转

旋转位置编码把每个 head 的 $$d_{head}$$ 维向量看成 $$d_{head}/2$$ 个二维平面，第 $$j$$ 个平面按位置 $$p$$ 旋转角度 $$p \cdot \theta_j$$：

$$
\begin{pmatrix} x'_j \\ y'_j \end{pmatrix}
=
\begin{pmatrix} \cos p\theta_j & -\sin p\theta_j \\ \sin p\theta_j & \cos p\theta_j \end{pmatrix}
\begin{pmatrix} x_j \\ y_j \end{pmatrix}
$$

$$\cos$$ 与 $$\sin$$ 与输入无关，预先算成 `cos_sin_cache[max_position, rot_dim]`（前一半 cos、后一半 sin，FP32）。kernel 对每个 token 查 `positions[token]` 拿到那一行，对 q 的全部 head 与 k 的全部 KV head 就地旋转。字节数：q 与 k 各读写一次，每元素 4 B，Llama-3-8B 一个 token 的 q 是 $$32 \times 128 \times 2 = 8$$ KiB、k 是 2 KiB（GQA，8 个 KV head），$$T = 8192$$ 时共 160 MiB 读写，A100 下界约 80 µs——比 GEMM 小两个数量级，但它是每层都有的一次 launch。

两种布局的差别在"哪两个分量组成一个平面"：

- **NEOX 风格**（GPT-NeoX、Llama、Qwen 等）：第 $$j$$ 个平面是 $$(x[j], x[j + d/2])$$——前半段与后半段配对，就是 HuggingFace 代码里的 `rotate_half`；
- **GPT-J 风格**（GPT-J、ChatGLM 等，也叫 interleaved）：第 $$j$$ 个平面是 $$(x[2j], x[2j+1])$$——相邻两个元素配对。

vLLM 用一个 `IS_NEOX` 模板参数区分，差别只在索引：

```cpp
// csrc/pos_encoding_kernels.cu（v0.20.0）
template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_token_rotary_embedding(
    scalar_t* __restrict__ arr, const float* __restrict__ cos_ptr,
    const float* __restrict__ sin_ptr, int rot_offset, int embed_dim,
    const bool inverse) {
  int x_index, y_index;
  float cos_f, sin_f;
  if (IS_NEOX) {
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos_f = VLLM_LDG(cos_ptr + x_index);
    sin_f = VLLM_LDG(sin_ptr + x_index);
  } else {
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos_f = VLLM_LDG(cos_ptr + x_index / 2);
    sin_f = VLLM_LDG(sin_ptr + x_index / 2);
  }
  if (inverse) {
    sin_f = -sin_f;
  }
  const float x_f = static_cast<float>(arr[x_index]);
  const float y_f = static_cast<float>(arr[y_index]);
  arr[x_index] = static_cast<scalar_t>(x_f * cos_f - y_f * sin_f);
  arr[y_index] = static_cast<scalar_t>(y_f * cos_f + x_f * sin_f);
}
```

外层 `rotary_embedding_kernel` 一个 token 一个 block，线程先遍历 `num_heads * embed_dim` 个 q 平面，再遍历 `num_kv_heads * embed_dim` 个 k 平面（GQA 下 k 的 head 数少，循环次数少）；`rot_dim` 可以小于 `head_size`（部分旋转），`rope_dim_offset` 支持只旋转 head 的一段。访存模式上 NEOX 风格每个线程读 `arr[j]` 与 `arr[j + d/2]`，一个 warp 的 32 个线程读连续的 32 个 BF16（64 字节）两段，coalescing 良好；GPT-J 风格每个线程读相邻的一对，一个 warp 读连续 128 字节，同样良好。

更进一步的融合是 **QK-norm + RoPE**：Qwen3 等模型在 RoPE 之前对每个 head 的 q、k 做一次 RMSNorm，两者都是按 head 的 row-wise 操作，可以在一个 warp 里完成。vLLM v0.20.0 的 `csrc/fused_qknorm_rope_kernel.cu`（`fusedQKNormRopeKernel`，改自 TensorRT-LLM）就是这个融合：一个 warp 处理一个 (token, head)，从合并的 QKV 张量里读 128 个元素，算 RMS、乘 norm 权重、查 cos/sin、旋转、写回——每元素 4 B 读写，比"norm kernel + RoPE kernel"的 8 B 少一半。更激进的是把 RoPE 与下一节的 KV cache 写入合并（`csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu` 这种名字说明了它做什么），把 k 旋转之后直接写进分页 cache，不再经过 `[T, n_kv, d]` 的中间张量。

### 5. `reshape_and_cache`：把新 token 的 KV 写进分页 cache

上一篇讨论的 PagedAttention 从分页 cache 里读 KV；这一节是它的写入端。每个新 token 的 k、v（形状 `[num_tokens, num_kv_heads, head_size]`）要写到 cache 里由调度器指定的位置：`slot_mapping[token]` 给出一个全局 slot 编号，`block_idx = slot / block_size`、`block_offset = slot % block_size`。字节数：每 token 读 k、v 各 $$n_{kv} \cdot d_{head} \cdot 2$$ 字节、写同样多——Llama-3-8B 一层一个 token 是 4 KiB 读 4 KiB 写，纯拷贝。

kernel 的全部复杂性来自 cache 的**布局**。v0.20.0 有两个 kernel：

```cpp
// csrc/cache_kernels.cu（v0.20.0）：v1 PagedAttention 的布局
template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void reshape_and_cache_kernel(
    const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
    cache_t* __restrict__ key_cache,     // [num_blocks, num_heads, head_size/x, block_size, x]
    cache_t* __restrict__ value_cache,   // [num_blocks, num_heads, head_size, block_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int key_stride, const int value_stride, const int num_heads,
    const int head_size, const int block_size, const int x,
    const float* k_scale, const float* v_scale) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    return;                        // padding token
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int h_block_count = head_size / x;
  const int h_block_idx = threadIdx.x;
  if (h_block_idx >= num_heads * h_block_count) return;
  const int head_idx = h_block_idx / h_block_count;
  const int h_block = h_block_idx % h_block_count;
  cache_t* __restrict__ key_dst =
      key_cache + block_idx * num_heads * h_block_count * block_size * x +
      head_idx * h_block_count * block_size * x + h_block * block_size * x +
      block_offset * x;
  const int64_t tgt_value_start =
      block_idx * num_heads * h_block_count * x * block_size +
      head_idx * h_block_count * x * block_size + h_block * x * block_size +
      block_offset;
  /* ... vectorize_with_alignment<VEC_SIZE>(key_src, key_dst, x, 0, 1, k_op);
     for (i < x) v_op(value_dst[i * block_size], value_src[i]); */
}
```

K cache 的 `[num_blocks, num_heads, head_size/x, block_size, x]` 布局里，$$x = 16 / \text{sizeof(cache\_t)}$$（BF16 时 8）：把 head_size 维切成若干段，每段 $$x$$ 个元素（16 字节）连续存放，同一段内 `block_size` 个 token 相邻。这是为 v1 PagedAttention kernel 设计的——那里一个 thread group 一次读 16 字节的 K，正好是一个 token 在一段上的 $$x$$ 个元素，`block_size` 个 token 的同一段连续，一个 warp 一次读的就是连续 `block_size × 16` 字节。V cache 的 `[num_blocks, num_heads, head_size, block_size]` 则是 head_size 维在外、token 在内，对应 v1 kernel 里 V 按列（head_size 维）做点积、沿 token 维合并读取。写入端的代价是 V 的每个元素要跨 `block_size` 的 stride 写（上面 `value_dst[i * block_size]`）——写是分散的，但每个 token 只有 4 KiB，可以接受。

`reshape_and_cache_flash_kernel` 服务于 FlashAttention / FlashInfer 后端，布局是 `[num_blocks, block_size, num_heads, head_size]`（NHD：一个 token 的所有 head 连续，就是普通的 `[tokens, heads, head_size]` 每 `block_size` 行切一页）或 `[num_blocks, num_heads, block_size, head_size]`（HND）。NHD 下写入是一个 token 整段连续，`vectorize_with_alignment<VEC_SIZE>` 一次 16 字节。两种 kernel 的 `Fp8KVCacheDataType kv_dt` 模板参数与 `k_scale / v_scale` 指针处理 **FP8 KV cache**：`CopyWithScaleOp` 在 `kv_dt != kAuto` 时调用 `fp8::scaled_convert`，把 BF16 除以 scale、饱和、转 E4M3 后写入，读取端（attention kernel）再乘回来。KV cache 用 FP8 之后每 token 从 128 KiB 变成 64 KiB，decode 时读 KV 的时间减半——这与权重量化的逻辑完全一样，都是 memory-bound 侧的字节交易。


## 五、MoE 的 kernel 流水线

### 1. 流水线

MoE 层把 FFN 换成 $$E$$ 个 expert，每个 token 只走 $$k$$ 个（DeepSeek-V3：$$E = 256$$、$$k = 8$$；Mixtral：$$E = 8$$、$$k = 2$$）。数学上每个 token 的输出是

$$
y_t = \sum_{e \in \text{top}_k(t)} w_{t,e} \cdot \text{FFN}_e(x_t)
$$

GPU 上不能一个 token 一个 token 地算——要把走同一个 expert 的 token 收集到一起做 GEMM。于是 vLLM 的 fused MoE 流水线（`vllm/model_executor/layers/fused_moe/fused_moe.py` 编排，kernel 在 `csrc/moe/`）是：

```text
router logits [T, E]
  -> topk_softmax                 每 token 选 top-k：topk_weights [T, k]、topk_ids [T, k]（可 renormalize）
  -> moe_align_block_size         按 expert 排序 token；每个 expert 的 token 数 padding 到 BLOCK_M 倍数；
                                  产出 sorted_token_ids、expert_ids（每个 M-tile 属于哪个 expert）、
                                  num_tokens_post_padded
  -> grouped GEMM #1              Triton fused_moe_kernel：每个 M-tile 查 expert_ids 选 W1[e]，
                                  输出 [T*k, 2*d_ff]（gate 与 up）
  -> SiLU-and-mul                 [T*k, d_ff]
  -> grouped GEMM #2              同一 kernel，W2[e]，输出 [T*k, d]，epilogue 里可乘 topk_weights
  -> moe_sum / unpermute          按 token 把 k 个 expert 的输出加权求和 -> [T, d]
```

### 2. `topk_softmax`：一个 warp 的一部分处理一行

router logits 一行只有 $$E$$ 个数（8 到 256），一个 block 一行太浪费。`csrc/moe/topk_softmax_kernels.cu`（源自 FasterTransformer / TensorRT-LLM）的 `topkGating<VPT, NUM_EXPERTS, WARPS_PER_CTA, BYTES_PER_LDG, ...>` 让 `THREADS_PER_ROW = NUM_EXPERTS / VPT` 个线程共同处理一行（$$E = 8$$、`VPT = 4` 时 2 个线程一行，一个 warp 处理 16 行），每个线程把自己的 `VPT` 个 logits 读进寄存器数组 `row_chunk[VPT]`，用 `__shfl_xor_sync` 在这几个线程之间做 max 与 sum 归约得到 softmax，然后循环 $$k$$ 次：每个线程本地 argmax → butterfly shuffle 达成一致 → 胜出的线程把自己的那个值置 $$-\infty$$ → 下一轮。整个过程没有 shared memory，也不排序。它还支持 `bias`（DeepSeek-V3 的 correction bias：用 logits + bias 选 expert，但权重用不加 bias 的值）与 `renormalize`（把选出的 $$k$$ 个权重归一化到和为 1）。这个 kernel 的字节数是 $$T \times E \times 4$$ 读、$$T \times k \times 8$$ 写，微不足道；它存在的意义是避免 `torch.softmax` + `torch.topk` 两次 launch 与中间张量。

### 3. `moe_align_block_size`：排序与 padding

这是 MoE 流水线里最"CPU 味"的 kernel：输入 `topk_ids [T, k]`，输出三个数组：

- `sorted_token_ids[max_num_tokens_padded]`：按 expert 排好的 (token, k-slot) 扁平索引 $$i = t \cdot k + j$$，每个 expert 的一段 padding 到 `block_size`（Triton GEMM 的 `BLOCK_M`）的倍数，padding 位填 `numel`（越界哨兵，GEMM 里对它 mask 掉）；
- `expert_ids[max_num_m_blocks]`：第 $$m$$ 个 M-tile 属于哪个 expert；
- `num_tokens_post_padded`：padding 后的总行数。

```cpp
// csrc/moe/moe_align_sum_kernels.cu（v0.20.0），_moe_align_block_size 核心
  // 1) 每个 expert 的 token 计数（shared memory 上原子加）
  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];
    if (expert_id >= num_experts) continue;
    /* expert_map / token_mask 处理略 */
    atomicAdd(&shared_counts[warp_idx * experts_per_warp + expert_offset], mask);
  }
  __syncthreads();
  // 2) 每个 expert 的计数向上取整到 block_size 倍数，做 exclusive prefix sum
  int expert_count = 0;
  int expert_id = threadIdx.x;
  if (expert_id < num_experts) {
    expert_count = shared_counts[warp_idx * experts_per_warp + expert_offset];
    expert_count = CEILDIV(expert_count, block_size) * block_size;
  }
  int cumsum_val;
  BlockScan(temp_storage).ExclusiveSum(expert_count, cumsum_val);
  if (expert_id <= num_experts) cumsum[cumsum_offset + expert_id] = cumsum_val;
  if (expert_id == num_experts) total_tokens_post_pad[model_offset] = cumsum_val;
  __syncthreads();
  // 3) 每个 M-tile 标上 expert 编号
  if (threadIdx.x < num_experts) {
    for (int i = cumsum[cumsum_offset + threadIdx.x];
         i < cumsum[cumsum_offset + threadIdx.x + 1]; i += block_size) {
      expert_ids[expert_ids_offset + i / block_size] = threadIdx.x;
    }
  }
```

第四步 `_count_and_sort_expert_tokens`（另一个 kernel，多 block 并行）再扫一遍 `topk_ids`，对每个 $$i$$ 用 `atomicAdd(&cumsum[expert_id], 1)` 拿到它在该 expert 段内的位置，写 `sorted_token_ids[pos] = i`。这是一个计数排序（counting sort），两遍扫描，$$O(Tk + E)$$；同一 expert 内的 token 顺序由原子操作的到达顺序决定，不稳定，但 GEMM 不在乎。`moe_permute_unpermute_op.cu` 提供另一条路径：用 CUB 的 radix sort 对 `topk_ids` 排序得到 `permuted_idx`，然后 `expandInputRowsKernel` 真的把 $$x$$ 的行按排序结果**物化**成 `permuted_input [T*k, d]`（gather，多写一份 $$Tk \cdot d \cdot 2$$ 字节），供 CUTLASS grouped GEMM 这类需要每个 expert 的输入连续的实现使用；`finalizeMoeRoutingKernel` 是它的逆——按 `inv_permuted_idx` gather 回来并乘 `topk_weights` 求和。Triton 的 `fused_moe_kernel` 则不物化，直接用 `sorted_token_ids` 做间接寻址：

```python
# vllm/model_executor/layers/fused_moe/fused_moe.py（v0.20.0）fused_moe_kernel 摘要
num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
    return                                        # 这个 M-tile 全是 padding
offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)   # 本 tile 的 T*k 扁平索引
token_mask = offs_token < num_valid_tokens
off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)  # 本 tile 属于哪个 expert
if off_experts == -1:
    return
a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
```

A 的行地址是 `offs_token // top_k`（同一个 token 被 $$k$$ 个 expert 各读一次），B 的基址加 `off_experts * stride_be` 选中 expert 的权重——**每个 tile 有自己的 B 矩阵**，这就是 grouped GEMM 与普通 GEMM 的唯一区别。

### 4. 为什么 grouped GEMM 的效率低

每个 expert 的 GEMM 是 $$[M_e, K] \times [K, N]$$，$$M_e$$ 是分到它的 token 数。平均 $$\bar{M}_e = Tk / E$$：DeepSeek-V3 decode batch 128 时 $$128 \times 8 / 256 = 4$$，prefill 4096 token 时也只有 128。用 §2.2 的公式，每个 expert GEMM 的算术强度约为 $$\bar{M}_e$$（BF16）——**每个 expert 的权重只被 $$Tk/E$$ 行摊薄**，而 dense FFN 的权重被全部 $$T$$ 行摊薄。所以同样 $$T$$ 下 MoE 的 FFN 部分比 dense 更靠 Roofline 的 memory-bound 一侧，交叉点要 $$Tk/E > 156$$，即 $$T > 156 E / k \approx 5000$$（DeepSeek-V3）。再加上 padding 浪费（每个 expert 平均浪费 `BLOCK_M/2` 行，$$E = 256$$、`BLOCK_M = 64` 时是 8192 行的空 `mma`）与负载不均（热门 expert 的 tile 多，冷 expert 只有一个 tile），grouped GEMM 达到的 MFU 通常显著低于同规模 dense GEMM。这也是 FP8 与 W4 权重对 MoE 模型收益更大的原因——它们几乎总在 memory-bound 一侧。

### 5. `csrc/moe/` 目录导读

```text
文件                                内容
topk_softmax_kernels.cu             topkGating：softmax + top-k 选择（FasterTransformer 系）
grouped_topk_kernels.cu             DeepSeek-V3 的分组 top-k（先选 group 再选 expert）
topk_softplus_sqrt_kernels.cu       其他 scoring 函数的变体
moe_align_sum_kernels.cu            moe_align_block_size（计数排序 + padding）、moe_sum、
                                    batched / small-batch / LoRA 变体
moe_permute_unpermute_op.cu +       CUB radix sort 排序、expandInputRows（物化 permute）、
  permute_unpermute_kernels/        finalizeMoeRouting（unpermute + 加权求和）
moe_wna16.cu                        W4A16 / W8A16 的 MoE GEMM（非 Marlin 路径）
marlin_moe_wna16/                   Marlin 的 MoE 版本（每个 expert 一组 Marlin stripe）
mxfp8_moe/                          MXFP8 量化 MoE
router_gemm.cu、dsv3_router_gemm_*  router 的小 GEMM 特化（[T, d] x [d, E]，E=256 时 N 极小）
torch_bindings.cpp                  以上算子的 TORCH_LIBRARY 注册（_moe_C）
```


## 六、采样 kernel

### 1. top-k / top-p：对 128K 的词表做选择

生成阶段每个 token 都要从 `logits [B, V]` 里采样。$$V = 128256$$（Llama-3）时一行 512 KB（FP32），$$B = 256$$ 时 128 MiB——只读一遍是 64 µs，与一层 decode 的时间同量级。问题在于 top-k 与 top-p（nucleus）都需要**排序或选择**：top-p 要按概率降序累加到阈值，朴素实现是 `torch.sort` 一整行（128K 元素的 radix sort，多次 pass，每 pass 读写全行），再 cumsum、再 mask、再 `multinomial`——每 token 十几次 kernel launch 与几 MB 的中间张量，batch 大时是时间线上清晰可见的一段。

kernel 层的改进方向是**不排序**：用 radix select（按位分桶、只保留包含第 $$k$$ 大元素的那个桶继续）或"猜阈值 + 验证"的方式找到 top-k 的阈值，两三遍扫描完成；top-p 类似地用二分查找阈值使超过它的概率和恰好越过 $$p$$。vLLM v0.20.0 的 `csrc/sampler.cu` 有 `topKPerRowDecode` / `topKPerRowPrefill`（每行一个 block 做 top-k），`csrc/persistent_topk.cuh` 与 `topk.cu` 是更通用的实现；FlashInfer 提供了一组 `sampling` kernel（top-k、top-p、top-k+top-p 联合、min-p），用 rejection-based 的思路把采样与截断合成一个 kernel，不物化排序结果——这是目前 SGLang 与 vLLM 都会调用的实现。

### 2. 拒绝采样：投机解码的验证

投机解码里 draft 模型给出 $$\gamma$$ 个候选 token，目标模型一次前向算出每个位置的分布 $$p$$，与 draft 的分布 $$q$$ 逐位置比较：以概率 $$\min(1, p(x)/q(x))$$ 接受候选 $$x$$，第一个被拒绝的位置从修正分布 $$\text{norm}(\max(0, p - q))$$ 重采样，全部接受则额外从最后一个位置的 $$p$$ 采一个 **bonus token**。kernel 层它是一个 $$[B, \gamma, V]$$ 的 elementwise 比较加一次每行的归一化重采样，本身不重；它成为瓶颈的方式与 top-p 一样——朴素实现是十几个小 launch 加同步（"接受了几个"要回到 CPU 才能决定下一步）。FlashInfer 的 `chain_speculative_sampling` 把整条链的验证做成一个 kernel，输出接受长度与采样结果，不需要中间同步。


## 七、数值验证：tolerance 怎么定

写完这些 kernel，第一件事是与参考实现对比。三个原则：

**1. BF16 GEMM 的误差随 $$\sqrt{K}$$ 增长。**$$K$$ 个乘积在 FP32 里累加、最后转 BF16，误差有两个来源：输出的一次 BF16 舍入（相对 $$2^{-8} \approx 0.4\%$$），以及每个输入元素的 BF16 舍入误差（各约 $$\varepsilon = 2^{-8}$$ 相对）经 $$K$$ 项累加后的随机游走——绝对误差按 $$\varepsilon \sqrt{K}$$ 倍的"典型乘积大小"增长。对随机符号的数据，输出本身也是 $$\sqrt{K}$$ 倍的典型乘积大小，相对误差仍在 $$\varepsilon$$ 量级；但一旦输出因为正负抵消而偏小（这在真实激活里很常见），相对误差就会放大到 $$10^{-2}$$ 甚至更高。所以对 BF16 GEMM 用 `torch.testing.assert_close(out, ref, rtol=1.6e-2, atol=1e-5)` 是合理的起点（这是 PyTorch 对 BF16 的默认值），$$K$$ 越大 atol 要相应放宽；参考应当用 FP32 输入的 `torch.matmul`，而不是 BF16 的 cuBLAS——后者自己也有同量级误差，两个误差叠加会让你误判。

**2. 量化 kernel 与"参考的量化实现"比，不与 FP16 比。**INT4 反量化 GEMM 的输出与 FP16 GEMM 的差异由**量化本身**决定（per-group 128、4 位时权重相对误差约 $$1/(2 \times 7) \approx 7\%$$ 量级），这是算法的事、不是 kernel 的事。验证 kernel 时，先用 Python 把打包权重按定义反量化成 FP32 矩阵 $$\hat{W} = s \cdot (q - z)$$，再算 $$x \hat{W}^T$$ 作为参考——这样差异只剩下 kernel 的累加顺序与舍入，应当在 BF16 GEMM 的 tolerance 内。FP8 同理：参考是 $$\text{dequant}(A_q) \cdot \text{dequant}(B_q)$$，不是原始的 $$A \cdot B$$。

**3. FP8 per-tensor 的 outlier 效应要单独测。**一行激活里一个 300 的 outlier 会让 per-tensor scale 变成 $$300/448 \approx 0.67$$，此时绝对值小于 0.67 的元素全部量化为 0 或 $$\pm 0.67$$——这不是 bug，是格式的性质。测试数据要用有 outlier 的分布（如少数元素乘 100 的正态分布）而不是纯 `randn`，并检查 per-token 与 per-tensor 两种粒度下误差的差别是否符合预期。

tolerance 的经验表：

```text
kernel                    参考                              rtol        atol       备注
BF16 elementwise / norm   FP32 计算再转 BF16                 1.6e-2      1e-5       主要是最后一次舍入
BF16 GEMM (K=4096)        FP32 matmul                       1.6e-2      1e-3       atol 随输出量级调
FP8 GEMM                  dequant 后的 FP32 matmul           2e-2        1e-2       E4M3 尾数 3 位
INT4 W4A16 GEMM           dequant 后的 FP32 matmul           1.6e-2      1e-3       与 BF16 GEMM 同
动态量化 (absmax)          Python 逐行 absmax / 448 再 cast   scale 精确；量化值允许 1 LSB 差异
top-k 采样                 排序后的集合相等                    —           —          比较集合而不是顺序
```


## 八、实践：四个 kernel，组装成 decoder layer

以下代码默认 `-arch=sm_80`，输入 BF16、累加 float。用第二篇的 `bench` 与 `torch.utils.cpp_extension.load_inline` 接到 Python；扩展机制不展开。

### 1. fused residual + RMSNorm

一行一个 block，两遍：第一遍 $$z = x + r$$ 写回 $$r$$、累加 $$z^2$$；block 归约得 $$\text{rstd} = 1/\sqrt{\text{mean}(z^2) + \epsilon}$$；第二遍从 $$r$$（L1/L2 命中）读 $$z$$，写 $$\text{out} = z \cdot \text{rstd} \cdot w$$。理论字节：8 B/元素。

```cpp
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// out[row] = RMSNorm(x[row] + residual[row]) * w；residual 就地更新为 x + residual
// 要求 d 为偶数，x/residual/out 16 字节对齐
template <int BLOCK>
__global__ void fused_add_rms_norm_kernel(__nv_bfloat16* __restrict__ out,
                                         const __nv_bfloat16* __restrict__ x,
                                         __nv_bfloat16* __restrict__ residual,
                                         const __nv_bfloat16* __restrict__ w,
                                         int d, float eps) {
  static_assert(BLOCK % 32 == 0, "BLOCK must be a multiple of 32");
  constexpr int NUM_WARPS = BLOCK / 32;
  __shared__ float s_partial[NUM_WARPS];
  __shared__ float s_rstd;

  const size_t row_off = static_cast<size_t>(blockIdx.x) * d;
  const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x + row_off);
  __nv_bfloat162* r2 = reinterpret_cast<__nv_bfloat162*>(residual + row_off);
  __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(out + row_off);
  const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(w);
  const int d2 = d / 2;

  // pass 1: z = x + residual -> residual; sum of squares (of the BF16-rounded z)
  float ss = 0.f;
  for (int i = threadIdx.x; i < d2; i += BLOCK) {
    float2 xv = __bfloat1622float2(x2[i]);
    float2 rv = __bfloat1622float2(r2[i]);
    __nv_bfloat162 zb = __float22bfloat162_rn(make_float2(xv.x + rv.x, xv.y + rv.y));
    r2[i] = zb;
    float2 z = __bfloat1622float2(zb);
    ss += z.x * z.x + z.y * z.y;
  }
  for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) s_partial[warp] = ss;
  __syncthreads();
  if (warp == 0) {
    float v = (lane < NUM_WARPS) ? s_partial[lane] : 0.f;
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    if (lane == 0) s_rstd = rsqrtf(v / static_cast<float>(d) + eps);
  }
  __syncthreads();
  const float rstd = s_rstd;

  // pass 2: out = z * rstd * w（z 从 residual 重读，命中 L1/L2）
  for (int i = threadIdx.x; i < d2; i += BLOCK) {
    float2 z = __bfloat1622float2(r2[i]);
    float2 wv = __bfloat1622float2(w2[i]);
    o2[i] = __float22bfloat162_rn(make_float2(z.x * rstd * wv.x, z.y * rstd * wv.y));
  }
}

// launch：rows 个 block，每 block 256 线程
// fused_add_rms_norm_kernel<256><<<rows, 256, 0, stream>>>(out, x, residual, w, d, 1e-5f);
```

与 vLLM 的差别只在向量宽度（这里 2 个元素、vLLM 8 个）与归约方式（这里手写 warp shuffle、vLLM 用 `cub::BlockReduce`）。$$d = 4096$$、256 线程时每线程处理 8 个 `__nv_bfloat162`，第一遍的 load 是 4 字节粒度——改成 `uint4`（8 个 BF16）一次读 16 字节能再提高一点带宽利用率，留作练习。

### 2. SiLU-and-mul

一个 token 一个 block，`__nv_bfloat162` 向量化，每次算两个元素。理论字节：6 B/输出元素。

```cpp
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// out[token, :d] = silu(in[token, :d]) * in[token, d:2d]；要求 d 为偶数
__global__ void silu_and_mul_kernel(__nv_bfloat16* __restrict__ out,       // [tokens, d]
                                    const __nv_bfloat16* __restrict__ in,  // [tokens, 2*d]
                                    int d) {
  const size_t in_off = static_cast<size_t>(blockIdx.x) * 2 * d;
  const __nv_bfloat162* gate2 = reinterpret_cast<const __nv_bfloat162*>(in + in_off);
  const __nv_bfloat162* up2 = reinterpret_cast<const __nv_bfloat162*>(in + in_off + d);
  __nv_bfloat162* out2 =
      reinterpret_cast<__nv_bfloat162*>(out + static_cast<size_t>(blockIdx.x) * d);
  const int d2 = d / 2;
  for (int i = threadIdx.x; i < d2; i += blockDim.x) {
    float2 g = __bfloat1622float2(gate2[i]);
    float2 u = __bfloat1622float2(up2[i]);
    float2 r;
    r.x = g.x / (1.f + __expf(-g.x)) * u.x;
    r.y = g.y / (1.f + __expf(-g.y)) * u.y;
    out2[i] = __float22bfloat162_rn(r);
  }
}

// launch：tokens 个 block；block 取 min(d/2, 1024) 向上取到 32 的倍数，d=14336 时用 1024
// silu_and_mul_kernel<<<tokens, 1024, 0, stream>>>(out, in, d);
```

`__expf` 是快速近似指数（SFU 指令），相对误差约 $$2^{-21}$$ 量级，远小于 BF16 的 $$2^{-8}$$，在这里安全。

### 3. RoPE（NEOX 风格，q/k 就地）

一个 token 一个 block；线程编号 $$i$$ 先覆盖 q 的 `num_heads × half` 个平面，再覆盖 k 的 `num_kv_heads × half` 个平面；每个线程读写一对 $$(x_j, x_{j + d/2})$$，互不重叠，所以就地更新没有竞争。

```cpp
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// q: [tokens, num_heads, head_size]; k: [tokens, num_kv_heads, head_size]（就地）
// cos_sin_cache: [max_position, rot_dim] fp32，前 rot_dim/2 为 cos，后 rot_dim/2 为 sin
// NEOX 布局：第 j 个平面是 (x[j], x[j + rot_dim/2])；rot_dim <= head_size
__global__ void rope_neox_kernel(const int64_t* __restrict__ positions,   // [tokens]
                                 __nv_bfloat16* __restrict__ q,
                                 __nv_bfloat16* __restrict__ k,
                                 const float* __restrict__ cos_sin_cache,
                                 int num_heads, int num_kv_heads,
                                 int head_size, int rot_dim) {
  const int token = blockIdx.x;
  const int64_t pos = positions[token];
  const int half = rot_dim / 2;
  const float* cos_ptr = cos_sin_cache + pos * rot_dim;
  const float* sin_ptr = cos_ptr + half;

  __nv_bfloat16* q_tok = q + static_cast<size_t>(token) * num_heads * head_size;
  __nv_bfloat16* k_tok = k + static_cast<size_t>(token) * num_kv_heads * head_size;
  const int nq = num_heads * half;
  const int nk = num_kv_heads * half;

  for (int i = threadIdx.x; i < nq + nk; i += blockDim.x) {
    __nv_bfloat16* base;
    int j;
    if (i < nq) {
      base = q_tok + (i / half) * head_size;
      j = i % half;
    } else {
      const int t = i - nq;
      base = k_tok + (t / half) * head_size;
      j = t % half;
    }
    const float c = cos_ptr[j];
    const float s = sin_ptr[j];
    const float x = __bfloat162float(base[j]);
    const float y = __bfloat162float(base[j + half]);
    base[j] = __float2bfloat16(x * c - y * s);
    base[j + half] = __float2bfloat16(y * c + x * s);
  }
}

// launch：tokens 个 block，256 线程（Llama-3-8B：nq + nk = (32 + 8) * 64 = 2560 个平面）
// rope_neox_kernel<<<tokens, 256, 0, stream>>>(positions, q, k, cos_sin_cache, 32, 8, 128, 128);
```

`cos_sin_cache` 在 Python 侧构造：`inv_freq = 1 / base^(arange(0, rot_dim, 2) / rot_dim)`，`freqs = outer(arange(max_pos), inv_freq)`，`cache = cat([cos(freqs), sin(freqs)], dim=-1)`，形状 `[max_pos, rot_dim]`。与 HuggingFace `apply_rotary_pos_emb` 的 `rotate_half` 实现逐元素对照即可。

### 4. 教学版 INT4 weight-only GEMV

这是 Marlin 的"教学版"：不用 Tensor Core、不做 repack、不做 `cp.async` 流水，只保留最核心的结构——**每个 warp 负责输出的一列 $$n$$，32 个 lane 沿 K 维交错读打包权重，寄存器里解包、乘 scale、与 BF16 激活做 FMA，FP32 累加，最后 warp 归约**。它的目标是在 $$M = 1$$ 时接近权重字节的带宽下界。

权重格式（Python 侧准备）：`W` 逻辑形状 `[N, K]`（`nn.Linear` 的 `weight`），per-group 对称量化 $$g = 128$$，$$q \in [0, 15]$$，零点 8，真实值 $$\hat{w} = s \cdot (q - 8)$$；8 个连续 $$k$$ 的 $$q$$ 装进一个 32 位字，第 $$j$$ 个占 bit $$4j$$ 到 $$4j + 3$$：

```python
import torch

def quantize_w4_g128(W: torch.Tensor, G: int = 128):
    """W: [N, K] bf16/fp32 -> packed int32 [N, K//8], scales bf16 [N, K//G], W_hat fp32 [N, K]"""
    N, K = W.shape
    assert K % G == 0 and G % 8 == 0
    Wg = W.float().view(N, K // G, G)
    s = (Wg.abs().amax(dim=-1, keepdim=True) / 7.0).clamp(min=1e-8)      # [N, K/G, 1]
    q = torch.clamp(torch.round(Wg / s) + 8, 0, 15).to(torch.int64)        # 0..15，零点 8
    W_hat = ((q.float() - 8.0) * s).view(N, K)                            # 参考用的反量化权重
    q = q.view(N, K // 8, 8)
    packed = torch.zeros(N, K // 8, dtype=torch.int64, device=W.device)
    for j in range(8):
        packed |= q[:, :, j] << (4 * j)
    packed = torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32)  # 按位转 int32
    return packed.contiguous(), s.view(N, K // G).to(torch.bfloat16).contiguous(), W_hat
```

kernel：

```cpp
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// y[M, N] = x[M, K] @ W_hat[N, K]^T，W_hat = s * (q - 8)
// w_packed: [N, K/8] uint32，每字 8 个连续 k 的 INT4（第 j 个在 bit 4j..4j+3）
// scales:   [N, K/G] bf16；x: [M, K] bf16（16 字节对齐）；y: [M, N] bf16
// 每个 warp 处理一列 n；M <= MAX_M
template <int MAX_M, int G>
__global__ void w4a16_gemv_kernel(__nv_bfloat16* __restrict__ y,
                                  const __nv_bfloat16* __restrict__ x,
                                  const uint32_t* __restrict__ w_packed,
                                  const __nv_bfloat16* __restrict__ scales,
                                  int M, int N, int K) {
  static_assert(G % 8 == 0, "group size must be a multiple of 8");
  const int warps_per_block = blockDim.x >> 5;
  const int n = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (n >= N) return;

  const int kw = K / 8;                     // 每列的 32 位字数
  constexpr int WORDS_PER_GROUP = G / 8;
  const uint32_t* wcol = w_packed + static_cast<size_t>(n) * kw;
  const __nv_bfloat16* scol = scales + static_cast<size_t>(n) * (K / G);

  float acc[MAX_M];
#pragma unroll
  for (int m = 0; m < MAX_M; ++m) acc[m] = 0.f;

  // 32 个 lane 交错读同一列的连续字：一个 warp 一次读 128 字节，coalesced
  for (int w = lane; w < kw; w += 32) {
    const uint32_t packed = wcol[w];
    const float s = __bfloat162float(scol[w / WORDS_PER_GROUP]);
    float wq[8];
#pragma unroll
    for (int j = 0; j < 8; ++j)
      wq[j] = static_cast<float>(static_cast<int>((packed >> (4 * j)) & 0xFu)) - 8.f;

    const int k0 = w * 8;
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
      if (m < M) {
        // 一次读 8 个 BF16 激活（16 字节）；x 的 M 行反复被所有列读到，驻留 L2
        const uint4 xv = *reinterpret_cast<const uint4*>(x + static_cast<size_t>(m) * K + k0);
        const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xv);
        float partial = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 xf = __bfloat1622float2(xp[j]);
          partial = fmaf(wq[2 * j], xf.x, partial);
          partial = fmaf(wq[2 * j + 1], xf.y, partial);
        }
        acc[m] = fmaf(s, partial, acc[m]);   // 同一字的 8 个 k 属于同一 group，scale 提到外面
      }
    }
  }

#pragma unroll
  for (int m = 0; m < MAX_M; ++m) {
    float v = acc[m];
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    acc[m] = v;
  }
  if (lane == 0) {
    for (int m = 0; m < M && m < MAX_M; ++m)
      y[static_cast<size_t>(m) * N + n] = __float2bfloat16(acc[m]);
  }
}

// launch：128 线程 = 4 个 warp = 4 列一个 block；grid = ceil(N / 4)
// w4a16_gemv_kernel<8, 128><<<(N + 3) / 4, 128, 0, stream>>>(
//     y, x, reinterpret_cast<const uint32_t*>(packed.data_ptr<int32_t>()), scales, M, N, K);
```

逐行核对几处容易错的地方：`(packed >> (4 * j)) & 0xF` 先转 `int` 再转 `float` 再减 8，得到 $$-8$$ 到 $$7$$；`scol[w / WORDS_PER_GROUP]` 正确，因为字 $$w$$ 覆盖 $$k = 8w$$ 到 $$8w + 7$$，group 编号是 $$8w / G = w / (G/8)$$，且 $$G$$ 是 8 的倍数保证一个字不跨 group；`x + m * K + k0` 的字节偏移是 $$2(mK + 8w)$$，$$K$$ 为 8 的倍数时是 16 的倍数，`uint4` 读对齐；`acc[m]` 的归约用 `__shfl_xor_sync` 全 mask，因为 warp 里所有 lane 都活着（`n >= N` 的 return 是整 warp 一起的）。

它与 Marlin 的差距：（1）不用 Tensor Core，$$M$$ 一大就 compute-bound 在 CUDA Core 的 19.5 TFLOPS 上——$$M = 8$$、$$N = K = 4096$$ 时 FLOPs 是 0.27 G，CUDA Core 需要 14 µs，已接近权重 8 MiB 的 4 µs 带宽时间的三倍，所以它只在 $$M \le 2$$ 时接近带宽下界；（2）解包用整数移位 + `cvt`（每元素约 3 条指令），Marlin 用 `lop3` + magic number（每元素约一条）；（3）没有 `cp.async` 流水，靠 warp 数量掩盖延迟——$$N = 4096$$ 时 1024 个 block、4096 个 warp，108 个 SM 每个约 38 个 warp，勉强够。

对照：`ref = x.float() @ W_hat.T`，`assert_close(y.float(), ref, rtol=1.6e-2, atol=1e-3)`。按第七章的原则，这里比的是 `W_hat` 而不是原始 `W`。

### 5. 组装 decoder layer 前向

把本篇的四个 kernel 与前几篇的 GEMM（第六篇 BF16 `mma.sync` GEMM）、attention（第八篇 FlashAttention 前向 / paged decode）按下面的顺序接起来，就是一个 Llama 风格 decoder layer。以 Llama-3-8B（$$d = 4096$$、32 个 q head、8 个 KV head、$$d_{head} = 128$$、$$d_{ff} = 14336$$）、$$T$$ 个 token 为例：

```text
步  kernel                       输入 shape                       输出 shape            来源
1   fused_add_rms_norm           x [T,4096], residual [T,4096]    xn [T,4096]           本篇 §8.1（第 0 层第一次用普通 RMSNorm）
2   QKV GEMM                     xn [T,4096] x Wqkv [6144,4096]^T qkv [T,6144]          第六篇 BF16 GEMM，或 §8.4 INT4 GEMV（T 小）
    split                        qkv -> q [T,32,128], k [T,8,128], v [T,8,128]（view，不拷贝）
3   rope_neox                    positions [T], q, k（就地）        q, k                   本篇 §8.3
4   reshape_and_cache            k, v, slot_mapping [T]            paged K/V cache        第八篇（写入端本篇 §4.5）
5   attention                    q, K cache, V cache, block_table  o [T,32,128]          第八篇（prefill: flash；decode: paged）
6   O GEMM                       o.view(T,4096) x Wo [4096,4096]^T attn_out [T,4096]     第六篇 / §8.4
7   fused_add_rms_norm           attn_out, residual（就地）          xn2 [T,4096]          本篇 §8.1；residual <- residual + attn_out
8   gate_up GEMM                 xn2 x Wgu [28672,4096]^T          gu [T,28672]          第六篇 / §8.4
9   silu_and_mul                 gu [T,28672]                      h [T,14336]           本篇 §8.2
10  down GEMM                    h x Wd [4096,14336]^T             ffn_out [T,4096]      第六篇 / §8.4
    -> 下一层的步 1 以 (x = ffn_out, residual) 进入
```

PyTorch eager 对照实现（省略 KV cache，用 `F.scaled_dot_product_attention` 做 attention）：

```python
import torch
import torch.nn.functional as F

def rms_norm_ref(x, w, eps=1e-5):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)

def rope_ref(x, pos, cos_sin):              # x: [T, H, D], NEOX
    D = x.shape[-1]
    cs = cos_sin[pos]                        # [T, D]
    cos, sin = cs[:, :D // 2].unsqueeze(1), cs[:, D // 2:].unsqueeze(1)
    x1, x2 = x[..., :D // 2].float(), x[..., D // 2:].float()
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).to(x.dtype)

def decoder_layer_ref(h, residual, p, pos, cos_sin, n_h=32, n_kv=8, d_head=128):
    T = h.shape[0]
    residual = residual + h
    xn = rms_norm_ref(residual, p["norm1"])
    qkv = xn @ p["wqkv"].T
    q, k, v = qkv.split([n_h * d_head, n_kv * d_head, n_kv * d_head], -1)
    q = rope_ref(q.view(T, n_h, d_head), pos, cos_sin)
    k = rope_ref(k.view(T, n_kv, d_head), pos, cos_sin)
    v = v.view(T, n_kv, d_head)
    k = k.repeat_interleave(n_h // n_kv, dim=1)   # GQA 展开
    v = v.repeat_interleave(n_h // n_kv, dim=1)
    o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                                       v.transpose(0, 1)[None], is_causal=True)
    o = o[0].transpose(0, 1).reshape(T, n_h * d_head)
    attn_out = o @ p["wo"].T
    residual = residual + attn_out
    xn2 = rms_norm_ref(residual, p["norm2"])
    gu = xn2 @ p["wgu"].T
    d_ff = gu.shape[-1] // 2
    hh = F.silu(gu[:, :d_ff]) * gu[:, d_ff:]
    return hh @ p["wd"].T, residual
```

自己的 kernel 版本逐步替换：先只换 norm、SiLU、RoPE 三个 elementwise kernel（GEMM 与 attention 仍用 PyTorch），确认 `assert_close` 通过；再换 GEMM；再换 attention。每一步单独对照，误差定位才可能。用 INT4 GEMV 替换 GEMM 时，参考侧的权重要换成 `W_hat`。

### 6. 读者应看到的量级

没有 GPU 可实测，以下是理论下界与文献、经验中常见的区间：

- **fused_add_rms_norm** $$T = 8192$$、$$d = 4096$$：8 B/元素 × 33.5 M 元素 = 256 MiB，A100 下界约 134 µs。一行一个 block、向量化读写的实现通常能达到 **80% 到 90% 的带宽**，即 150 到 170 µs；`__nv_bfloat162` 版比 `uint4` 版差几个百分点。
- **silu_and_mul** $$T = 8192$$、$$d_{ff} = 14336$$：6 B × 117 M = 672 MiB，下界约 350 µs，预期 **85% 到 90%** 带宽（它没有归约，比 norm 更容易跑满）。
- **rope_neox** $$T = 8192$$：q + k 共 $$8192 \times 40 \times 128 \times 2 \times 2 = 160$$ MiB 读写，下界约 84 µs；每线程只读写 4 个 BF16，访存粒度小，通常 **60% 到 80%**。
- **w4a16_gemv** $$M = 1$$、$$N = K = 4096$$：权重 8 MiB + scale 256 KiB，下界约 4.4 µs；BF16 GEMV 读 32 MiB 下界 16.8 µs。教学版通常能到带宽的 50% 到 70%（约 6 到 9 µs），对比 cuBLAS 的 BF16 GEMV（约 18 到 20 µs），**2 到 3 倍加速**；Marlin 在同样形状下接近 4 倍。$$M = 8$$ 时教学版因 CUDA Core 算力见顶，加速收窄到 1.5 倍以内，而 Marlin 仍保持接近 4 倍——这就是 Tensor Core 路径存在的理由。
- **整层** decode（$$T = 1$$）：Llama-3-8B 一层权重 BF16 约 436 MB，读一遍 218 µs（A100），INT4 约 113 MB、57 µs；加上 KV cache 读取（上下文 $$s$$ 时 $$128$$ KiB $$\times s / 32$$ 每层）。elementwise kernel 在 $$T = 1$$ 时各只有几微秒，此时 kernel launch 开销（每个约 3 到 5 µs）与它们的执行时间同量级——这是第十篇讨论 CUDA Graph 与 launch 融合的动机。


## 九、小结

回到核心问题。INT4 W4A16 GEMM 在 decode 快 3 倍、prefill 反而慢，是因为它是一个"用指令换字节"的交易：字节除以 4，FLOPs 不变，还多了每元素约一条的反量化指令。decode 的 $$M$$ 小、算术强度 $$\approx 4M$$ 远低于 ridge 156，时间由字节决定，交易划得来；prefill 的 $$M$$ 大、两者都 compute-bound，时间由 FLOPs 决定，多出的指令与为小 M 调的 tile 配置让它比 cuBLAS 慢。交叉点在 $$M \approx \text{ridge}/4 \approx 40$$（A100）。FP8 W8A8 是不同的交易——字节与 FLOPs 同时减半——所以在 Hopper 上两侧都受益，代价是要动态量化激活并处理 scale 粒度。

本篇其余部分的要点：

- 格式只看三件事：位数、指数位（范围）、尾数位（精度）；E4M3 max 448 无 inf，E5M2 max 57344；累加尽量 FP32，Hopper FP8 累加精度有限是 per-block scale 要 promote 的原因。
- INT4 反量化用 magic number：`0x6400 | q` 是 FP16 的 $$1024 + q$$，`lop3` + `hsub2`/`hfma2` 每元素约一条指令；权重需 repack 成交错顺序。
- Marlin：repack、`cp.async` 流水、寄存器内反量化、striped partitioning、global reduce，为 $$M \le 64$$ 设计；vLLM 把 GPTQ/AWQ 权重 repack 后统一走 Marlin；Machete 是 Hopper 上的 CUTLASS 3.x 版本。
- FP8 GEMM 的 scale：per-tensor / per-token / per-channel 都不依赖 $$k$$，放 epilogue（CUTLASS EVT 两层 `multiplies`）；per-block 128×128 依赖 $$k$$，必须在 K 循环里按块乘并累加到 FP32。
- 动态量化 = absmax 归约 + 除 scale + `cvt`，与 RMSNorm 融合从 7 B/元素降到 3 B/元素。
- 融合 kernel 只赢字节：residual + RMSNorm 10 B → 8 B；SiLU-mul 6 B；RoPE 4 B（NEOX 前后半配对、GPT-J 相邻配对，只差索引）；`reshape_and_cache` 是纯拷贝，复杂性全在 cache 布局（v1 的 `[blocks, heads, d/x, block_size, x]` 与 flash 的 `[blocks, block_size, heads, d]`）。
- MoE：`topk_softmax` → `moe_align_block_size`（计数排序 + padding）→ grouped GEMM（每 tile 查 `expert_ids`）→ SiLU-mul → grouped GEMM → `moe_sum`；每个 expert 的有效 $$M$$ 只有 $$Tk/E$$，所以 MoE 的 FFN 比 dense 更靠 memory-bound 一侧。
- 采样：top-k/top-p 对 128K 词表的 sort/select 在大 batch 下可见；拒绝采样的瓶颈是 launch 与同步，FlashInfer 把它们做成单 kernel。
- 验证：BF16 GEMM 用 FP32 参考、rtol 1.6e-2；量化 kernel 与反量化后的参考比，不与 FP16 比；FP8 per-tensor 要用带 outlier 的数据测。

数字汇总：

```text
格式                     位布局 (S/E/M)   最大值       eps          用途
FP16                     1/5/10           65504        9.8e-4       激活（旧）、Marlin 反量化目标
BF16                     1/8/7            3.4e38       7.8e-3       默认激活/权重
FP8 E4M3 (fn)            1/4/3            448（无 inf） 0.125        W8A8 权重与激活、FP8 KV cache
FP8 E5M2                 1/5/2            57344        0.25         训练梯度
INT8                     整数              127          —            W8A8 (Ampere)、KV cache
INT4                     整数              7 / 0..15    —            W4A16 权重（g=128，约 4.15 bit/权重）

Roofline（A100，N=K=4096）        算术强度 I(M)      交叉点 M       decode 下界 (Llama-3-8B)   prefill
BF16 GEMM                         约 M               约 156         16 GB / 2 TB/s = 8 ms      0.44 ms/GEMM
INT4 W4A16                        约 4M              约 40          约 4.1 GB -> 约 2 ms        略慢于 BF16
FP8 W8A8 (H100, ridge 590)        约 2M              约 295         8 GB / 3.35 TB/s = 2.4 ms  FLOPs 时间减半

融合模式                          分开 (B/元素)      融合 (B/元素)   理论时间 (T=8192, A100)
residual add + RMSNorm (d=4096)  10                 8              134 us
RMSNorm + 动态量化 FP8            7                  3              —
SiLU-and-mul (d_ff=14336)        —                  6              350 us
RoPE q+k (32+8 heads x 128)      8 (norm+rope)      4              84 us
reshape_and_cache (每 token/层)   —                  4 KiB 读 + 4 KiB 写

vLLM v0.20.0 csrc 导读
csrc/quantization/{marlin, gptq, awq, machete, gguf, w8a8/{fp8,int8}, fused_kernels, hadamard}/
csrc/libtorch_stable/quantization/w8a8/cutlass/{c2x, c3x, moe}/   CUTLASS scaled_mm（FP8/INT8，blockwise）
csrc/cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp      ScaledEpilogue（EVT）
csrc/activation_kernels.cu · layernorm_kernels.cu · pos_encoding_kernels.cu · fused_qknorm_rope_kernel.cu
csrc/cache_kernels.cu（reshape_and_cache / _flash）· csrc/moe/ · csrc/sampler.cu · topk.cu
```

到这里，decoder layer 的 kernel 全集已经凑齐：第三篇的 elementwise、第四篇的 norm 与 softmax、第五六篇的 GEMM、第八篇的 attention、本篇的 RoPE、SiLU-mul、fused norm、INT4 GEMV 与 KV 写入。它们能跑通一个 layer 的前向、能与 PyTorch eager 对照正确性。但"能跑"与"能进 vLLM 主线"之间还有一段距离：怎么用 Nsight Compute 确认离 Roofline 还有多远、怎么写 `opcheck` 通得过的测试、怎么用 `TORCH_LIBRARY` 注册成算子并让 `torch.compile` 认识它、怎么处理多架构编译。这是最后一篇的内容。


## 下一篇

[剖析、测试与贡献](/kernel-profiling-testing-and-contribution.html)
