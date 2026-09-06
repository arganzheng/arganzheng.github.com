---
layout: post
title: "通信与互联（07）：推理侧的通信——custom all-reduce 与 KV 传输"
subtitle: "Communication on the Inference Side: Custom All-Reduce and KV Cache Transfer"
tags: [NCCL, RDMA, GPU, AI, AI-Infra]
catalog: true
---

> 本文是[《通信与互联：从 NCCL 到 RDMA》](/communication-and-interconnect-for-ai-infra.html)系列的第 7 篇（共七篇）。上一篇：[nccl-tests、调优与排障：从带宽曲线到 hang](/nccl-tests-tuning-and-debugging-hangs.html)

前六篇建立了一条完整的路径：第一篇的 α-β 模型给出任何一次通信的理论下界，第二、三篇给 α 和 β 填上 NVLink、PCIe、InfiniBand 与 RDMA 的真实数字，第四篇讲 NCCL 如何把这些硬件能力组织成一次 `ncclAllReduce`，第五篇讲 PyTorch 如何在 stream 上使用它，第六篇把这一切变成 nccl-tests 的曲线和一棵排障决策树。这些内容的默认场景是训练：消息几十 MB 到几 GB、参与者固定、通信可以和反向计算重叠。

推理把这套方法推到两个极端。第一个极端是**张量并行的 decode**：batch 很小、hidden 维度固定，每一层两次 all_reduce，每次只有几十到几百 KB，一个 decode step 要做上百次；按第一篇的模型这是纯延迟主导，带宽多少完全无关，NCCL 为大消息设计的一切——多 channel、ring 的 $$2(n-1)$$ 步、协议握手——在这里都是负担。第二个极端是**PD 分离的 KV 传输**：一个请求 prefill 结束后，几百 MB 到几 GB 的 KV cache 要从 prefill 实例点对点搬到 decode 实例，目标是动态的、不需要归约、必须不打扰两边正在跑的计算；这是纯带宽主导，而且 NCCL 的"固定 communicator + kernel 占 SM + 与 stream 强耦合"三个特点恰好都不合适。

vLLM 对这两个场景给出了两套完全不同的答案。对前者，它绕开 NCCL 自己写了一个 all_reduce kernel（custom all-reduce），并用 ctypes 直接调用 `libnccl.so` 作为兜底（PyNccl）；对后者，它把 KV 传输抽象为 `KVConnector`，底层交给 NIXL、UCX 或 Mooncake Transfer Engine 这些以单边 RDMA 为核心的传输库。理解这两套答案，就是把前六篇的代价模型、拓扑、RDMA、NCCL 内部机制和 stream 语义一次性用上。

本篇要回答总纲提出的核心问题：

> **8 卡 TP 的 decode，每层一次 128 KB 的 all_reduce，NCCL 要 30 微秒，custom all-reduce 要 10 微秒。这 20 微秒省在哪里？为什么这个方法不能用在训练的梯度同步上？**

以及第二个同等重要的问题：KV cache 该用什么传、能传多快、为什么不是 NCCL。

本文的源码以 **vLLM v0.23.0（2026-06-14 发布）** 为准，NCCL 以 2.28.9 为准，PyTorch 以 2.12 为准，nccl-tests 以 2.18.3 为准。vLLM 的通信后端迭代很快，all_reduce 后端的选择顺序、阈值表、NIXL connector 的接口在几个月内都可能变化，正文提到的文件与函数名请以你手上的版本核对。文中所有性能数字都是理论下界、公开标称值或 NCCL 自己调优表里的模型常数，明确标注"非实测"；总纲核心问题里的 30 µs 与 10 µs 是量级示意，不是某台机器的测量结果。


## 一、总览：两种相反的通信需求

### 1. 推理里的两条通信路径

```text
                     decode step（每层两次，几十～几百 KB，延迟主导）
   ┌──────────────────────────────────────────────────────────────────────┐
   │ GPU0 ──┐                                                             │
   │ GPU1 ──┤  TP all_reduce                                              │
   │  ...   ├──► CudaCommunicator.all_reduce                              │
   │ GPU7 ──┘      ├─ NCCL symmetric memory（可选，需 VLLM_USE_NCCL_SYMM_MEM）│
   │               ├─ quick all-reduce（仅 ROCm）                          │
   │               ├─ FlashInfer all-reduce（可选）                        │
   │               ├─ custom all-reduce  ← 本篇主角：IPC + flag + 一个 kernel │
   │               ├─ torch symmetric memory（multimem / two-shot）        │
   │               └─ PyNccl（ctypes 直调 libnccl.so）                     │
   └──────────────────────────────────────────────────────────────────────┘

                     PD 分离（每请求一次，几百 MB～GB，带宽主导）
   ┌──────────────────────────────────────────────────────────────────────┐
   │  Prefill 实例（TP=Np）                    Decode 实例（TP=Nd）          │
   │  KV blocks（显存，已 register_memory）      KV blocks（显存，已注册）    │
   │        ▲                                        │                    │
   │        │  单边 RDMA READ（NixlConnector）         │                    │
   │        └────── NIXL → UCX → verbs / GDR ◄────────┘                    │
   │                    或 Mooncake Transfer Engine（WRITE）                │
   │  scheduler 侧：决定传哪些 block、何时可释放     worker 侧：发起与轮询传输 │
   └──────────────────────────────────────────────────────────────────────┘
```

上半部分是节点内、GPU 之间、每层都发生的集合通信；下半部分是跨节点、实例之间、每个请求发生一次的点对点搬运。两者共享的只有硬件：NVLink 与 IB/RoCE 网卡。软件栈上，前者最终落到一个 CUDA kernel 直接读写对端显存，后者最终落到网卡的 DMA 引擎直接读写显存，两边都没有 NCCL——这是本篇要解释的第一件事。

### 2. 两本账在推理侧的落点

用第一篇的记号，一次 all_reduce 的时间 $$T = \alpha_{\text{eff}} + S/\beta_{\text{eff}}$$。训练的梯度 bucket S 在 25 MB 量级，$$S/\beta$$ 项几十到几百微秒，α 项十几微秒，看的是带宽的账；decode 的 TP all_reduce S 在 128 KB 量级，NVLink 上 $$S/\beta$$ 不到 1 µs，α 项十几微秒，看的是延迟的账。同一个 API，两本账的答案相反：训练希望多 channel、大 chunk、Simple 协议跑满带宽；推理希望一次同步、一个 kernel、最少的步数。

KV 传输回到带宽的账，但和梯度 all_reduce 又不一样：它不是集合通信，是点对点；它的对端在运行时才知道；它不能占用正在做 decode 的 GPU 的 SM；它要跑满的是每 GPU 一张的 400 Gb/s 网卡，而不是 NVLink。这四个约束合起来，把它推向第三篇讲的单边 RDMA。

### 3. 本文的章节安排

```text
第二章  算一算：decode 阶段 TP all_reduce 每层多少字节、为什么是纯延迟、NCCL 在这个区间的固定开销有哪些、
        30 µs 与 10 µs 的差在哪里
第三章  看一看：custom all-reduce 的实现——CUDA IPC 共享缓冲区、Signal 与 flag 同步、one-shot 与 two-shot
        kernel、启用条件、为什么不能用于训练
第四章  后端选择链：GroupCoordinator 与 device communicator、CudaCommunicator.all_reduce 的精确顺序、
        PyNccl 为什么绕开 ProcessGroupNCCL、对称内存与 NVLS 的适用区间
第五章  CUDA Graph 与通信：all_reduce 被捕获需要满足什么、register_graph_buffers 在做什么
第六章  算一算：PD 分离的 KV 传输——多少字节、什么时候传、传给谁、为什么不用 NCCL
第七章  看一看：KVConnector 的 scheduler / worker 分工、NixlConnector 的注册、握手、READ 与通知、
        NIXL 与 UCX、Mooncake Transfer Engine
第八章  测一测与比一比：TP all_reduce 延迟异常与 KV 传输慢的排障检查项
第九章  本文小结与 comm-probe 增量：tp_ar_bench.py 与 kv_xfer/
系列总结与系列目录
```


## 二、算一算：decode 阶段 TP all_reduce 的账

### 1. 每层多少字节

张量并行把每个 Transformer 层切成两段需要归约的地方：attention 的输出投影之后、MLP 的 down projection 之后，各一次 all_reduce，归约的张量形状是 `[batch_tokens, hidden]`。decode 阶段每个请求一步只产生一个 token，所以 `batch_tokens` 就是并发请求数：

$$
S = \text{batch\_tokens} \times \text{hidden} \times \text{dtype\_bytes}
$$

以 Llama-3-70B 为例，hidden = 8192，BF16：

```text
batch   1    →   8192 × 2 =  16 KB
batch   8    →   8 × 8192 × 2 = 128 KB        ← 总纲核心问题里的数字
batch  32    →  512 KB
batch 128    →    2 MB
```

80 层 × 2 次 = 每个 decode step 160 次 all_reduce。batch 8 时每步通信总量 20 MB，但这 20 MB 是 160 次 128 KB，不是一次 20 MB——这是全部问题的来源。

### 2. α-β 模型下这是纯延迟

8 卡 H100 NVSwitch，NVLink 每 GPU 双向合计 900 GB/s，单向 450 GB/s（标称）。用 ring all_reduce，每个 rank 收发 $$\frac{2(n-1)}{n} S = 1.75 \times 128\ \text{KB} = 224\ \text{KB}$$，带宽项：

$$
\frac{224\ \text{KB}}{450\ \text{GB/s}} \approx 0.5\ \mu\text{s}
$$

用 one-shot（每张卡直接读所有人的数据），每个 rank 读 7 份远端的 128 KB 共 896 KB，带宽项约 2 µs。两个数字都远小于任何一次同步的固定开销。第一篇给的判据是：当 $$S/\beta \ll \alpha$$ 时时间由 α 决定；这里 α 是十几微秒量级，128 KB 的消息离拐点差两个数量级。结论是：**decode TP all_reduce 的优化目标只有一个，压低每次调用的固定开销，带宽怎样都无关**。

把它放回 decode step 的总时间里看意义。70B 模型 TP8 每张卡放 17.5 GB 权重，每步至少要把权重从 HBM 读一遍，H100 HBM3 约 3.35 TB/s（标称），下界约 5.2 ms。如果 160 次 all_reduce 每次 30 µs，通信 4.8 ms，与读权重的下界同量级；每次 10 µs，通信 1.6 ms。这就是 vLLM 愿意为此维护一个自己的 kernel 的原因。

### 3. NCCL 在这个区间的固定开销

NCCL 自己的调优表给出了它对小消息延迟的估计。`src/graph/tuning.cc` 里 `ncclTunerConstantsDefaults` 的 `baseLatencies` 和 `hwLatencies`（单位 µs，NCCL 2.28.9 的默认常数，用于算法选择，非实测）：

```text
baseLatencies（一次 collective 的基础开销）     LL     LL128   Simple
   Tree                                        6.8    14.0    8.4
   Ring                                        6.6    14.0    8.4
hwLatencies[NVLINK]（每一步的链路开销）        LL     LL128   Simple
   Tree                                        0.6    1.25    4.0
   Ring                                        0.6    1.9     3.4
   NVLS                                        -      -       25
```

同一文件里对单节点 ring 的延迟公式是 `baseLatencies + nsteps × intraLat`，all_reduce 的 `nsteps = 2(n-1)`。8 卡：

```text
Ring  + LL      6.6 + 14 × 0.6  ≈ 15.0 µs
Ring  + LL128  14.0 + 14 × 1.9  ≈ 40.6 µs
Ring  + Simple  8.4 + 14 × 3.4  ≈ 56.0 µs
Tree  + LL      6.8 + 2 × 7 × 0.6 ≈ 15.2 µs
NVLS  + Simple                    25   µs
```

这些数字是 NCCL 的模型，不是测量，但它告诉我们 NCCL 自己认为 8 卡 128 KB 的 all_reduce 在 NVLink 上最好也要 15 µs 左右，而且这 15 µs 由两部分组成：约 6～8 µs 的基础开销，加上 14 步每步 0.6 µs 的串行同步。把这个 15 µs 拆开，NCCL 在这个区间要付的固定成本有：

- **host 侧入队与 kernel 启动。** `src/enqueue.cc` 的 `ncclLaunchPrepare` 构建 kernel plan、`ncclLaunchKernel` 用 `cuLaunchKernelEx` 启动 `ncclDevKernel_AllReduce_*`；在 ProcessGroupNCCL 之上还有 Work 对象、event 记录、stream 依赖的建立（第五篇）。这是几微秒的 CPU 时间，并且在 CPU 侧串行，和 GPU 上前一个 kernel 的尾部不一定重叠。
- **算法的步数。** ring 的 $$2(n-1) = 14$$ 步、tree 的 $$2 \log_2 n$$ 级，每一步都是一次"等对端 flag 到达→读数据→归约→写给下一个→写 flag"，NVLink 上每步 0.6 µs 是 LL 协议下的最优值。
- **协议开销。** LL 协议为了免掉内存屏障，8 字节数据搭 8 字节 flag，带宽效率 50%；对 128 KB 无所谓，但它把每个 chunk 的传输拆成大量 16 字节原子写，每一步的 0.6 µs 就是这样来的。LL128 和 Simple 带宽高但每步更慢。
- **proxy 线程。** 节点内 NVLink P2P 传输由 GPU kernel 直接完成，proxy 线程只参与连接建立（`src/transport/p2p.cc` 的 `ncclProxyConnect` / `ncclProxyCallBlocking` 都在 setup 阶段），数据路径不经过 CPU；proxy 的固定开销属于跨节点场景，8 卡节点内的 TP 不付这笔账。但一旦 TP 跨节点，每步都要经过 proxy 提交 RDMA 请求并轮询完成，延迟跳到几十微秒。

### 4. 30 µs 与 10 µs：20 µs 省在哪里

把总纲核心问题里的两个数字当作量级：NCCL 路径约 30 µs，custom all-reduce 约 10 µs。差的 20 µs 分布在四处：

```text
                              NCCL（Ring + LL，经 ProcessGroupNCCL）    custom all-reduce
kernel 启动与入队              每次 collective 一次完整 launch 路径       一个普通 CUDA kernel，
                              + Work / event（数 µs CPU 侧）             与模型其他算子同一 stream 顺序启动
同步步数                       2(n-1) = 14 步串行，每步一次 flag 握手     one-shot：2 次 barrier（起始、结束）
                                                                        two-shot：2 次 barrier + 一轮中间缓冲区读写
数据搬运方式                   每步经 channel buffer 中转，LL 拆成 16 B   直接 ld.global 读对端显存（IPC 映射），
                              数据 + flag                               16 B 向量化 load，fp32 累加后一次写回
CUDA Graph                    可捕获，但 launch 路径与 comm 状态有约束     天然可捕获（普通 kernel + 固定地址）
```

第一项是 host 侧的启动成本；第二项是最大的一块，14 步串行同步换成 2 次 barrier；第三项是访存模式，custom all-reduce 没有中间 buffer，每个线程读 8 个地址各 16 字节就完成了归约。第四项在 vLLM 里是决定性的：decode 阶段几乎全部在 CUDA Graph 里回放，所有 host 侧启动成本都在捕获时付掉，回放时只剩 GPU 上的步数，这时 NCCL 的 14 步和 custom AR 的 2 步之差就是全部差距。

这 20 µs 里没有"带宽"这一项。这是核心问题的第一半答案；第二半——为什么不能用到训练上——放到第三章看完实现之后回答。


## 三、看一看：custom all-reduce 的实现

vLLM 的 custom all-reduce 分三层：CUDA kernel、host 类与 `Signal` / `RankData` / barrier 原语都在同一个头文件 `csrc/custom_all_reduce.cuh`，torch 算子绑定在 `csrc/libtorch_stable/custom_all_reduce.cu`，Python 侧的启用判断与缓冲区管理在 `vllm/distributed/device_communicators/custom_all_reduce.py` 的 `CustomAllreduce` 类。

### 1. IPC 与共享缓冲区：让一张卡读到别人的显存

第二篇讲过 GPUDirect P2P：同一节点内有 NVLink 或 PCIe P2P 能力的两张卡，一张卡的 kernel 可以直接解引用另一张卡显存的地址。跨进程使用它需要 CUDA IPC：拥有者用 `cudaIpcGetMemHandle` 把一段 `cudaMalloc` 的显存导出为句柄，其他进程用 `cudaIpcOpenMemHandle` 打开得到本进程地址空间里的指针。vLLM 的每个 TP rank 是一个进程，所以全部走这条路。

`csrc/libtorch_stable/custom_all_reduce.cu` 的 `allocate_shared_buffer_and_handle` 分配一块显存并导出句柄；Python 侧 `CustomAllreduce.create_shared_buffer` 用 `dist.all_gather_object` 在 CPU group（gloo）上交换句柄，再对每个远端句柄调用 `open_mem_handle`（内部是 `cudaIpcOpenMemHandle(..., cudaIpcMemLazyEnablePeerAccess)`），得到一个长度为 world_size 的指针列表——第 i 项是 rank i 那块缓冲区在本进程里的地址。

`CustomAllreduce.__init__` 创建两组这样的共享缓冲区：

```python
# vllm/distributed/device_communicators/custom_all_reduce.py，CustomAllreduce.__init__，有删节
self.meta_ptrs = self.create_shared_buffer(
    ops.meta_size() + max_size, group=group, uncached=True
)
# ...
self.buffer_ptrs = self.create_shared_buffer(max_size, group=group)
# ...
self.rank_data = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=self.device)
self._ptr = ops.init_custom_ar(self.meta_ptrs, self.rank_data, rank, self.fully_connected)
ops.register_buffer(self._ptr, self.buffer_ptrs)
```

- `meta_ptrs`：每个 rank 一块，前面 `sizeof(Signal)` 字节是同步用的 flag（`meta_size()` 返回 `sizeof(vllm::Signal)`），后面 `max_size` 字节是 two-shot 用的中间结果区。C++ 侧的 `CustomAllreduce` 类注释把它画成 `| -- sizeof(Signal) -- | ------ a few MB ----- |`。
- `buffer_ptrs`：每个 rank 一块预注册的输入缓冲区。eager 模式下输入张量的地址每次不同、没有 IPC 句柄，所以先 `cudaMemcpyAsync` 到这块已注册的缓冲区再做归约（`all_reduce` 绑定函数里 `if (reg_buffer) cudaMemcpyAsync(...)`）。CUDA Graph 模式下不需要这一步，见第五章。
- `rank_data`：8 MB 的设备端数组，存放 `RankData` 结构——每个已注册缓冲区在所有 rank 上的地址列表。kernel 拿到一个 `RankData*` 就知道 8 张卡各自的数据在哪里。

```cpp
// csrc/custom_all_reduce.cuh
struct __align__(16) RankData { const void* ptrs[8]; };
struct __align__(16) RankSignals { Signal* signals[8]; };
```

### 2. Signal 与 flag 同步：barrier_at_start 与 barrier_at_end

没有 NCCL，跨卡同步靠 flag。`Signal` 结构（同一文件 `csrc/custom_all_reduce.cuh`，8 是硬编码的最大 rank 数）：

```cpp
constexpr int kMaxBlocks = 36;
using FlagType = uint32_t;
struct Signal {
  alignas(128) FlagType start[kMaxBlocks][8];
  alignas(128) FlagType end[kMaxBlocks][8];
  alignas(128) FlagType _flag[kMaxBlocks];  // incremental flags for each rank
};
```

每个 block、每个对端 rank 一个计数器，`start` 与 `end` 两组。源码注释解释为什么要两组：一个 rank 的 block 可能已经到达第二个同步点，而另一个 rank 的 block 还在等第一个同步点，如果共用一个计数器，前者的 `+1` 会破坏后者的等待条件。`_flag` 是本地单调递增的期望值，每次 barrier 自增 1，`uint32_t` 溢出是良定义的。

`barrier_at_start` 的逻辑（同文件，CUDA 分支）：

```cpp
template <int ngpus>
DINLINE void barrier_at_start(const RankSignals& sg, Signal* self_sg, int rank) {
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    auto peer_counter_ptr = &sg.signals[threadIdx.x]->start[blockIdx.x][rank];
    auto self_counter_ptr = &self_sg->start[blockIdx.x][threadIdx.x];
    st_flag_volatile(peer_counter_ptr, flag);            // 写到对端显存：我到了
    while (ld_flag_volatile(self_counter_ptr) != flag);  // 自旋等对端写到我这里
  }
  __syncthreads();
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}
```

block 里前 `ngpus` 个线程各负责一个对端：线程 i 把 `flag` 写到 rank i 的 `Signal` 里属于本 rank 的槽位（一次经 NVLink 的远端 store），然后自旋读本地槽位直到 rank i 也写过来。这是一次 all-to-all 的 flag 交换，延迟约等于一次 NVLink 往返。`barrier_at_end` 结构相同但用 `end` 数组，并且在非最终同步时用 `st.release.sys` / `ld.acquire.sys`（`st_flag_release` / `ld_flag_acquire`）保证前面的数据写入对对端可见；最终同步（`final_sync = true`）不需要可见性保证，用 volatile 版本更快。ROCm 分支用 `__scoped_atomic_store_n` / `__scoped_atomic_load_n` 实现同样语义。

第四篇讲过 NCCL 的 LL 协议也是用 flag 免屏障。区别在于：NCCL 的 flag 跟着每 8 字节数据走，是数据通道的一部分；custom all-reduce 的 flag 只在 kernel 开头和结尾各交换一次，数据通道是裸的 `ld.global`。

### 3. one-shot 与 two-shot kernel

`csrc/custom_all_reduce.cuh` 里两个 kernel。**one-shot**（`cross_device_reduce_1stage`）：

```cpp
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_1stage(RankData* _dp, RankSignals sg, Signal* self_sg,
                               T* __restrict__ result, int rank, int size) {
  using P = typename packed_t<T>::P;   // 16 字节向量，BF16 时 8 个元素
  using A = typename packed_t<T>::A;   // 对应的 float 累加器
  auto dp = *_dp;
  barrier_at_start<ngpus>(sg, self_sg, rank);
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    ((P*)result)[idx] = packed_reduce<P, ngpus, A>((const P**)&dp.ptrs[0], idx);
  }
  barrier_at_end<ngpus, true>(sg, self_sg, rank);
}
```

每个线程处理 16 字节：从 8 个 rank 的输入各读一个 16 字节向量（7 次经 NVLink 的远端 load），在 fp32 累加，转回 BF16 写到本地输出。全 kernel 两次 barrier。源码注释特别说明不重排地址，让所有 rank 以相同顺序累加，结果 bit 级一致——训练里 `NCCL_ALGO` 不同导致数值不同的问题（第六篇）在这里不存在。

**two-shot**（`cross_device_reduce_2stage`）：先 reduce_scatter 再 all_gather。每个 rank 负责 `size / ngpus` 那一段：读 8 个 rank 该段的数据归约，写到自己 `Signal` 后面的临时区（`get_tmp_buf`）；`barrier_at_end` 同步（非最终，带 release/acquire）；然后每个线程从 8 个 rank 的临时区各读回一段拼成完整结果。源码注释强调两个阶段必须用相同的 `tid` 处理相同的下标，因为跨设备可见性只在相同 tid 的线程之间有保证。

两者的选择在 `CustomAllreduce::allreduce`（同文件）的 `REDUCE_CASE` 宏里，可用环境变量 `VLLM_CUSTOM_ALLREDUCE_ALGO`（取值 `1stage` / `oneshot` / `2stage` / `twoshot`）强制：

```cpp
if (world_size_ == 2) {
  KL(ngpus, cross_device_reduce_1stage);
} else if (fully_connected_) {
  if ((world_size_ <= 4 && bytes < 512 * 1024) ||
      (world_size_ <= 8 && bytes < 256 * 1024)) {
    KL(ngpus, cross_device_reduce_1stage);
  } else {
    KL(ngpus, cross_device_reduce_2stage);
  }
}
```

用两本账解释这个阈值。one-shot 每个 rank 读 $$(n-1) S$$ 字节远端数据，two-shot 读 $$\frac{n-1}{n} S$$ 两次共 $$\frac{2(n-1)}{n} S$$；n = 8 时前者是后者的 4 倍。但 one-shot 的两次 barrier 都是便宜的 volatile 版本，数据只经过一次"读—归约—写"；two-shot 的第二次 barrier 带 release/acquire 语义（要等 reduce_scatter 的结果对所有对端可见），数据还要多经过一轮中间缓冲区的写与远端读。S 小时同步占主导选 one-shot，S 大时读量占主导选 two-shot；8 卡的切换点 256 KB 对应"多读 $$\frac{7 \cdot 256 - 1.75 \cdot 256}{450} \approx 3\ \mu\text{s}$$ 字节的时间开始超过一次 barrier"。这和第一篇 ring 与 tree 的取舍是同一个道理：步数换带宽。

kernel 只用 36 个 block × 512 线程（`kMaxBlocks = 36`，`defaultBlockLimit = 36`，ROCm 为 16）。源码注释说这是在 A100 / A10 / A30 / T4 / V100 上网格搜索的结果，猜测是太多 SM 同时发 NVLink 请求会造成争用。这也意味着 all_reduce 期间绝大多数 SM 空闲——对 decode 这种 memory-bound 的场景没有损失。

### 4. 启用条件

`CustomAllreduce.__init__` 与 `should_custom_ar` 一起决定它是否被使用。逐条列出，因为排障时"custom AR 为什么没生效"就是对着这个清单查：

```text
条件                                        源码位置 / 日志
custom op 库可用（ops.meta_size() 不抛异常）    custom_all_reduce.py 模块级 try；否则 "Custom allreduce is disabled
                                            because of missing custom allreduce library"
group 是非 NCCL 的 CPU group                assert dist.get_backend(group) != dist.Backend.NCCL
group 不跨节点                              in_the_same_node_as(group, source_rank=0)；否则 warning "spans across nodes"
world_size ∈ _SUPPORTED_WORLD_SIZES         [2, 4, 6, 8]；否则 warning "unsupported world size"
                                            （kernel 侧 REDUCE_CASE 同样只有 2/4/6/8）
同一节点内 NVLink 全互联                      is_fully_connected：pynvml.nvmlDeviceGetP2PStatus(..., NVML_P2P_CAPS_INDEX_NVLINK)
                                            对每一对 GPU 都返回 NVML_P2P_STATUS_OK（vllm/platforms/cuda.py）
                                            world_size > 2 且非全互联 → "not supported on more than two PCIe-only GPUs"
                                            should_custom_ar 里再查一次：world_size == 2 或 fully_connected 才返回 True
P2P 实际可用                                 _can_p2p → gpu_p2p_access_check（all_reduce_utils.py）：真的开两个进程
                                            用 IPC 写一块显存再读回来验证；结果缓存在 VLLM_CACHE_ROOT 下
                                            gpu_p2p_access_cache_for_*.json；VLLM_SKIP_P2P_CHECK 运行时默认按 1 解析
                                            （envs.py 里声明的默认值是 False，但 getenv 兜底是 "1"），此时
                                            只信 torch.cuda.can_device_access_peer；源码注释建议排障时设为 0 做真实探测
输入字节数 < max_size                        默认 8 MB；启用 torch symm mem 时按 CUSTOM_ALL_REDUCE_MAX_SIZES
                                            收紧（H100 8 卡 256 KB、4 卡 32 MB；见第四章）
输入字节数是 16 的倍数、weak contiguous        should_custom_ar
ParallelConfig.disable_custom_all_reduce     vllm/config/parallel.py；CLI --disable-custom-all-reduce；
为 False                                    VLLM_BATCH_INVARIANT=1 会强制置 True；平台 use_custom_allreduce() 为 False 或
                                            nnodes > 1 也会
```

`is_fully_connected` 要求每一对 GPU 之间都有 NVLink 直连（NVSwitch 机器满足），因为 one-shot 的每个线程要同时读 7 个对端，任何一对走 PCIe 都会拖慢全部。源码注释说明了为什么两卡例外：两张 PCIe 直连的卡上 custom AR 仍然比 NCCL 好，四张以上 PCIe 卡则"提升很小"。

### 5. 为什么不能用在训练的梯度同步上

现在可以回答核心问题的第二半。custom all-reduce 的每一个设计选择都以"小消息、节点内、固定地址、延迟优先"为前提，梯度同步逐条违反：

- **消息大小。** 梯度 bucket 25 MB 量级，超过 `max_size`（默认 8 MB）；即使放宽，one-shot 每 rank 要读 $$(n-1)S = 175\ \text{MB}$$，two-shot 也没有 pipeline，36 个 block 跑不满 NVLink 带宽。NCCL 的多 channel、chunk 流水、LL128/Simple 协议就是为这个区间设计的，第四篇算过它能逼近链路上限。
- **只能节点内。** 数据通道是 `ld.global` 解引用 IPC 映射的对端地址，这要求所有参与者在同一个 PCIe/NVLink 地址域里。跨节点没有这个能力（`CustomAllreduce.__init__` 用 `in_the_same_node_as` 直接拒绝跨节点的 group），而训练几乎总是跨节点。
- **内存注册与 IPC 约束。** 输入必须是 IPC 注册过的地址：eager 模式先拷到 `buffer_ptrs`，多一次 `cudaMemcpyAsync`（源码注释估计 ≤ 1% 延迟，对 128 KB 成立，对 25 MB 不成立）；CUDA Graph 模式要求捕获期间的地址固定并在捕获后集体交换句柄。训练的梯度 buffer 由 Caching Allocator 管理，地址随迭代变化，第五篇讲过这正是 `recordStream` 问题的来源。
- **缓冲区上限。** two-shot 的中间结果放在 `Signal` 后面那块 `max_size` 大小的区域里，`rank_data` 一共 8 MB，按每组 8 × 8 字节算最多存 131072 组地址（Python 侧源码注释，并说最大的模型也只用到不到 10000 组）。这些都是为"一个模型几千次 all_reduce、每次几百 KB"设计的容量。
- **没有重叠、没有归约算子选择、没有容错。** kernel 只做 sum，没有 `ReduceOp` 选择；自旋等待的 barrier 没有 timeout，一张卡挂了其他卡永远自旋——第六篇讲的 watchdog 与 Flight Recorder 在这里都不存在。推理引擎可以接受这些（一张卡挂了整个实例都要重启），训练不行。

一句话：custom all-reduce 是把 NCCL 在 P2P transport + LL 协议 + 单 channel 下的行为，去掉所有为大消息、多节点、通用性付出的固定开销之后剩下的最小实现。它赢在减法，减掉的恰好是训练需要的。


## 四、后端选择链与 PyNccl

### 1. GroupCoordinator 与 device communicator

`vllm/distributed/parallel_state.py` 的 `GroupCoordinator` 是 vLLM 对一组进程的封装：它同时持有一个 `cpu_group`（gloo 后端的 `ProcessGroup`，用于交换 IPC 句柄、广播对象等控制面通信）、一个 `device_group`（NCCL 后端，作为兜底），以及一个 `device_communicator`。TP、PP、DP、EP 各是一个 `GroupCoordinator`，由 `initialize_model_parallel` 创建，`get_tp_group()` / `get_pp_group()` / `get_dp_group()` / `get_ep_group()` 返回。

`device_communicator` 在 `GroupCoordinator.__init__` 里由 `current_platform.get_device_communicator_cls()` 解析出类名后实例化，CUDA 平台上是 `CudaCommunicator`，构造参数带 `unique_name`——这个名字决定了哪些后端可用：

```python
# vllm/distributed/device_communicators/cuda_communicator.py，CudaCommunicator.__init__，有删节
if "tp" not in unique_name:
    # custom allreduce or torch symm mem can be used only by tp
    use_custom_allreduce = False
    use_torch_symm_mem = False
    use_flashinfer_allreduce = False
else:
    from vllm.distributed.parallel_state import _ENABLE_CUSTOM_ALL_REDUCE
    use_custom_allreduce = _ENABLE_CUSTOM_ALL_REDUCE
    use_torch_symm_mem = envs.VLLM_ALLREDUCE_USE_SYMM_MEM
    use_flashinfer_allreduce = envs.VLLM_ALLREDUCE_USE_FLASHINFER
    # ...
```

只有 TP group 才会创建 custom all-reduce 等快速后端；PP 的 send/recv、DP 的同步都只走 PyNccl。`_ENABLE_CUSTOM_ALL_REDUCE` 是 `parallel_state.py` 的模块级开关，`gpu_worker.py` 在初始化时用 `set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)` 设置。

模型代码调用 `tensor_model_parallel_all_reduce(x)`（`vllm/distributed/communication_op.py`）→ `get_tp_group().all_reduce(x)`。`GroupCoordinator.all_reduce` 里有一个分叉：`use_custom_op_call` 为真时走 `torch.ops.vllm.all_reduce(input_, group_name=...)` 这个注册的自定义算子，再由 `parallel_state.all_reduce` 按 `group_name` 查回 coordinator 调用 `_all_reduce_out_place`。源码注释说明了原因：Dynamo 不能把任意 Python 对象传进自定义算子，只能传字符串；自定义算子也不支持原地修改，所以 vLLM 的 all_reduce 全部是 out-of-place。这是 `torch.compile` 与 CUDA Graph 对通信层的第一个约束。

### 2. CudaCommunicator.all_reduce 的调度顺序

`CudaCommunicator.all_reduce` 是一条 if 链，每个后端有自己的 `should_*` 判断，不满足就落到下一个。**以 v0.23.0 源码为准的精确顺序**：

```text
 1. NCCL symmetric memory     should_nccl_symm_mem_allreduce(world_size, input_)
                              → torch.ops.vllm.all_reduce_symmetric_with_copy
                              需 VLLM_USE_NCCL_SYMM_MEM=1（默认 0）且 world_size ≥ 4
 2. quick all-reduce          qr_comm.should_quick_allreduce → quick_all_reduce   仅 ROCm MI300 系列
 3. FlashInfer all-reduce     fi_ar_comm.should_use_fi_ar → all_reduce            需 VLLM_ALLREDUCE_USE_FLASHINFER=1（默认 0）
 4. vLLM custom all-reduce    ca_comm.should_custom_ar → custom_all_reduce        本篇第三章
 5. torch symmetric memory    symm_mem_comm.should_use_symm_mem → all_reduce      VLLM_ALLREDUCE_USE_SYMM_MEM（默认 1）
 6. PyNccl                    pynccl_comm.all_reduce
 7. torch.distributed         input_.clone() + torch.distributed.all_reduce(group=device_group)  测试兜底
```

启动时 `_log_all_reduce_backend_selection` 会打一行日志，把这个 group 上实际启用的后端按调度顺序列出：

```text
Using ['CUSTOM', 'SYMM_MEM', 'PYNCCL'] all-reduce backends (in dispatch order) for group 'tp' out of potential backends: ['NCCL_SYMM_MEM', 'QUICK_REDUCE', 'FLASHINFER', 'CUSTOM', 'SYMM_MEM', 'PYNCCL'].
```

这是排障时第一个要看的东西：列表里没有 `CUSTOM`，就去查第三章第 4 节的清单。注意这个列表是"可能被选中"的集合，每一次调用具体走哪个还取决于输入大小——比如 `CUSTOM` 在列表里，但输入超过 `max_size` 时这一次仍会落到 `SYMM_MEM` 或 `PYNCCL`。

`vllm/distributed/device_communicators/all_reduce_utils.py` 里的注释记录了 NCCL symmetric memory 与 custom AR 的实测对比（H100 与 GB200，vLLM 团队的 benchmark，非本文实测）：8 卡时 2K–16K 与 128K 以上 NCCL symm mem 更快，32K–64K custom AR 更快；4 卡时 32K–256K custom AR 更快。对应的 `NCCL_SYMM_MEM_ALL_REDUCE_CONFIG` 定义了 custom AR 的"优势区间"：8 卡 (16 KB, 128 KB)，4 卡 (16 KB, 512 KB)，区间内不走 NCCL symm mem。这个表值得记住——它说明 custom all-reduce 不是在所有小消息上都赢，最小的消息（几 KB）上 NCCL 的对称内存路径反而更快，因为 custom AR 的 36 个 block 启动和两次 all-to-all flag 交换也是固定成本。

### 3. PyNccl：ctypes 直调 libnccl.so

`vllm/distributed/device_communicators/pynccl_wrapper.py` 的文件头注释是理解 PyNccl 的最好材料。它列出了 vLLM 尝试过并放弃的方案：cupy（初始化 communicator 时经常卡住）、`torch.distributed`（`all_reduce` 内部包含许多 CUDA Graph 捕获期间不允许的 CUDA API）、C/C++ 绑定（NCCL 版本切换要重新编译）。最终方案是纯 Python 的 ctypes 封装：

```python
# vllm/distributed/device_communicators/pynccl_wrapper.py，NCCLLibrary.__init__，有删节
so_file = so_file or find_nccl_library()       # 读 VLLM_NCCL_SO_PATH，否则 "libnccl.so.2"（ROCm 为 librccl.so.1）
lib = ctypes.CDLL(so_file)
```

`find_nccl_library`（`vllm/utils/nccl.py`）优先取环境变量 `VLLM_NCCL_SO_PATH`，这让运行时切换 NCCL 版本变成改一个变量。`NCCLLibrary` 用 `ctypes` 声明 `ncclGetUniqueId`、`ncclCommInitRank`、`ncclAllReduce`、`ncclAllGather`、`ncclReduceScatter`、`ncclSend` / `ncclRecv`、`ncclBroadcast`、`ncclGroupStart` / `ncclGroupEnd`、`ncclCommWindowRegister` / `ncclCommWindowDeregister`（NCCL 2.27+ 的对称内存窗口注册）等函数的签名。

`PyNcclCommunicator`（`pynccl.py`）的 `all_reduce`：

```python
def all_reduce(self, in_tensor, out_tensor=None, op=ReduceOp.SUM, stream=None):
    if self.disabled:
        return None
    # ...
    if out_tensor is None:
        out_tensor = torch.empty_like(in_tensor)
    if stream is None:
        stream = current_stream()
    self.nccl.ncclAllReduce(
        buffer_type(in_tensor.data_ptr()), buffer_type(out_tensor.data_ptr()),
        in_tensor.numel(), ncclDataTypeEnum.from_torch(in_tensor.dtype),
        ncclRedOpTypeEnum.from_torch(op), self.comm, cudaStream_t(stream.cuda_stream),
    )
    return out_tensor
```

和第五篇的 ProcessGroupNCCL 对照，区别有三点，都对应推理引擎的需求：

- **stream 由调用者决定。** `ncclAllReduce` 的 stream 参数直接传当前 stream（`current_stream()`），NCCL kernel 和模型 kernel 排在同一条队列里，不存在 ProcessGroupNCCL 那条内部 stream、不需要 event 建立依赖、不需要 `recordStream`。推理的前向是严格串行的，没有"通信与反向重叠"的需求，多一条 stream 只增加复杂度。
- **CUDA Graph 友好。** 纯 ctypes 调用只发起一次 `ncclAllReduce`，捕获期间不会触发 ProcessGroupNCCL 那些不可捕获的 CUDA API（event 查询、错误检查、Work 对象的 `cudaEventRecord` 等）。第五章展开。
- **没有 watchdog 与 heartbeat 线程。** 没有后台线程轮询 Work、没有 timeout 检测、没有 `ncclCommAbort` 的自动触发。推理引擎的失败模式是整个实例重启，不需要这套机制；同时也少了 watchdog 线程与主线程争 GIL 和 CPU 的干扰。`destroy` 里的 `ncclCommAbort` 放在一个带 timeout 的 daemon 线程里执行——源码注释说明 `ncclCommAbort` 会阻塞到所有捕获了该 comm 的 CUDA Graph 被销毁为止，直接在主线程调用会自锁。

`PyNcclCommunicator.__init__` 断言 group 不是 NCCL 后端：`ncclUniqueId` 通过 gloo 的 `cpu_group` 广播，NCCL communicator 由 vLLM 自己用 `ncclCommInitRank` 建立，与 PyTorch 的 `device_group` 是两个独立的 communicator。`VLLM_DISABLE_PYNCCL=1` 可以关掉它（此时兜底到 `torch.distributed`）。

### 4. 对称内存与 NVLS：两条路径

第五篇结尾提到 PyTorch 的对称内存（`torch/csrc/distributed/c10d/symm_mem/`）。它用 `cuMemCreate` + `cuMemImportFromShareableHandle` 让每个 rank 拿到所有 rank 缓冲区的映射（和 custom AR 的 IPC 做同一件事，但用的是 CUDA 12 的 VMM API），并在 Hopper 及之后用 `cuMulticastCreate` 建立**多播地址**：对多播地址的一次 `multimem.ld_reduce` 让 NVSwitch 在交换机内完成归约并把结果返回，一次 `multimem.st` 让 NVSwitch 把数据广播给所有 rank。这就是第四篇讲的 NVLS 算法的底层能力，NCCL 的 NVLS 也用它。

vLLM 有两条使用对称内存的路径：

**torch symmetric memory**（`symm_mem.py` 的 `SymmMemCommunicator`，默认开启 `VLLM_ALLREDUCE_USE_SYMM_MEM=1`）：用 `torch.distributed._symmetric_memory.empty` 分配一块 `max_size` 的 BF16 缓冲区并 `rendezvous`，`handle.multicast_ptr == 0` 则禁用（没有 NVSwitch 多播）。`all_reduce` 先把输入拷进缓冲区，然后按 world size 选算法：

```python
# vllm/distributed/device_communicators/symm_mem.py，有删节
_WORLD_SIZES_MULTIMEM = {"9.0": [4, 6, 8], "10.0": [6, 8], "10.3": [6, 8], ...}
# ...
if use_multimem:
    torch.ops.symm_mem.multimem_all_reduce_(self.buffer[: inp.numel()], "sum", self.group.group_name)
else:
    torch.ops.symm_mem.two_shot_all_reduce_(self.buffer[: inp.numel()], "sum", self.group.group_name)
```

H100（9.0）上 4/6/8 卡用 `multimem_all_reduce_`（NVSwitch 归约），2 卡用 `two_shot_all_reduce_`（P2P 读写的 reduce_scatter + all_gather，与 custom AR 的 two-shot 同构）。它的 `max_size` 来自 `SYMM_MEM_ALL_REDUCE_MAX_SIZES`：H100 8 卡 64 MB，4 卡 32 MB。只接受 BF16。

它和 custom AR 的分工由 `CUSTOM_ALL_REDUCE_MAX_SIZES` 决定：`CustomAllreduce.__init__` 在 `symm_mem_enabled` 为真时把自己的 `max_size` 收紧到这张表的值——H100 8 卡 256 KB、6 卡 512 KB、4 卡 32 MB、2 卡 64 MB。也就是说在 8 卡 H100 上：≤ 256 KB 走 custom AR 的 one-shot，256 KB～64 MB 走对称内存的 NVSwitch 多播归约，更大走 PyNccl。上一小节的 NCCL symm mem 表加上这一张，就是 vLLM 目前对"哪个区间用哪个后端"的全部经验。

**NCCL symmetric memory**（`pynccl_allocator.py` + `pynccl.py` 的 `register_nccl_symmetric_ops`，需 `VLLM_USE_NCCL_SYMM_MEM=1`）：用 NCCL 2.27+ 的 `ncclMemAlloc` 分配、`ncclCommWindowRegister` 注册窗口，让 NCCL 自己的 kernel 直接用对端地址（省掉 channel buffer 中转）并启用 NVLS。它是"让 NCCL 少付固定开销"的路线，而不是绕开 NCCL；`all_reduce_symmetric_with_copy` 里先 `copy_` 到对称内存张量再调 `pynccl_comm.all_reduce`。

### 5. 一张表：按消息大小与拓扑看各后端的适用区间

综合第三章的阈值、`all_reduce_utils.py` 的两张表和 NCCL 的调优常数，8 卡 H100 NVSwitch 节点内 TP all_reduce 的默认落点。这是一张**简化视图**：假定 v0.23.0 的默认配置——`VLLM_ALLREDUCE_USE_SYMM_MEM=1`、`VLLM_USE_NCCL_SYMM_MEM=0`、未启用 FlashInfer / QuickReduce（后两者只在显式开启或 ROCm 上才排在前面），完整的判定顺序以第四章 `cuda_communicator.py` 的 `all_reduce` 为准；若关掉 `VLLM_ALLREDUCE_USE_SYMM_MEM`，custom AR 的上限回到 8 MB，中段由它接管：

| 消息大小 | 默认走的后端 | 机制 | 两本账 |
|---|---|---|---|
| < 16 KB | custom AR one-shot（若开 NCCL symm mem 则 NCCL symm mem 更快） | 2 次 barrier + 直接读 | 纯 α；barrier 往返是全部成本 |
| 16 KB ～ 256 KB | custom AR one-shot | 同上 | 纯 α；decode batch 1～16 的典型区间 |
| 256 KB ～ 64 MB | torch symm mem `multimem_all_reduce_` | NVSwitch 多播归约 | α 与 β 各占一部分；NVLS 一步完成 |
| > 64 MB | PyNccl（NCCL 自选算法，NVSwitch 上通常 NVLS） | 多 channel 流水 | 纯 β；prefill 大 batch 的区间 |

拓扑的影响：

| 拓扑 | custom AR | torch symm mem | PyNccl |
|---|---|---|---|
| 8 卡 NVSwitch（默认节点） | 可用，≤ 256 KB | 可用（多播），256 KB ～ 64 MB | 兜底 |
| 8 卡 PCIe 直连（无 NVLink） | 禁用（>2 卡非全互联） | 禁用（无多播） | 全部 |
| 2 卡 PCIe | 可用（例外） | two-shot 版本，若 rendezvous 成功 | 兜底 |
| TP 跨节点 | 禁用（`nnodes > 1` 时 `ParallelConfig` 置 `disable_custom_all_reduce`，`CustomAllreduce.__init__` 也按 `in_the_same_node_as` 拒绝） | 禁用 | 全部，走 IB + proxy，每步几十 µs |
| 多机 NVLink（NVL72 类机型） | 同上，v0.23.0 的 custom AR 没有跨节点路径 | 取决于 symm mem 的 rendezvous 与多播能否建立，本文不展开 | 兜底 |

TP 跨节点那一行值得单独说：一旦 TP 组跨了节点，每层两次 all_reduce 都要经过 NIC 与 proxy 线程，延迟从十几微秒跳到几十微秒，160 次就是几毫秒到十几毫秒——这是"TP 不要跨节点、跨节点用 PP"这条经验的通信层根据。


## 五、CUDA Graph 与通信

### 1. 捕获 all_reduce 的条件

vLLM 的 decode 阶段用 CUDA Graph 回放整个前向。一次 all_reduce 要被捕获进 graph，需要满足三个条件：

- **捕获期间只发生允许的 CUDA 调用。** `cudaMalloc`、`cudaStreamSynchronize`、`cudaEventQuery` 等在捕获期间是非法的。这就是 pynccl_wrapper.py 头注释里放弃 `torch.distributed` 的原因，也是 custom AR 在 `allocate_shared_buffer_and_handle` 里用 `cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureModeRelaxed)` 包住 `cudaMalloc` 的原因。
- **kernel 参数在回放时仍然有效。** graph 记录的是 kernel 与它的参数（指针）。输入张量的地址必须在每次回放时相同——vLLM 用固定的输入缓冲区与 Caching Allocator 的私有内存池保证这一点——而 custom AR 还多一个问题：它的 kernel 参数里有*对端*的地址，捕获时并不知道。
- **通信对端也在同一位置捕获同一操作。** 所有 rank 的 graph 必须包含相同序列的集合通信，回放时才能配对。这与第六篇讲的"调用顺序不一致导致 hang"是同一个约束，只是搬到了 graph 里。

### 2. register_graph_buffers 在做什么

`csrc/custom_all_reduce.cuh` 的 `CustomAllreduce` 类注释把流程写得很清楚。捕获期间 `allreduce` 检测到 `cudaStreamIsCapturing` 为 `cudaStreamCaptureStatusActive`，不去查 `buffers_` 表，而是：

```cpp
if (status == cudaStreamCaptureStatusActive) {
  ptrs = d_rank_data_base_ + graph_unreg_buffers_.size();   // 预留一个 RankData 槽位
  graph_unreg_buffers_.push_back(input);                    // 记下本 rank 的输入地址
} else {
  auto it = buffers_.find(input);                           // eager：必须已注册
  // ...
}
```

kernel 参数里的 `RankData*` 指向 `rank_data` 数组中一个**尚未填内容**的槽位——地址固定，内容以后填。捕获结束后 Python 侧 `CustomAllreduce.capture()` 上下文管理器退出时调用 `register_graph_buffers`：

1. `get_graph_buffer_ipc_meta`（C++）对 `graph_unreg_buffers_` 里每个地址，用 `cuPointerGetAttribute(CU_POINTER_ATTRIBUTE_RANGE_START_ADDR)` 找到所属分配块的基址（IPC 句柄只能对整块分配导出），`cudaIpcGetMemHandle` 导出句柄并记录偏移；
2. Python 侧用 `dist.broadcast_object_list` 在 gloo group 上逐 rank 广播（源码注释说 `all_gather_object` 与 inference mode 下的 gloo 不兼容），每个 rank 拿到所有 rank 的句柄与偏移列表；
3. `register_graph_buffers`（C++）对每个远端句柄 `open_ipc_handle`（有 `ipc_handles_` 缓存去重）加偏移得到对端地址，填成 `RankData` 一次 `cudaMemcpy` 写进预留的槽位。

回放时 kernel 读到的 `RankData` 已经是完整的 8 个地址。这个设计的前提是每次回放输入地址不变，源码注释也提到故意不对地址去重，以防不同 rank 的分配模式不同。这也是第三章第 5 节"为什么不能用在训练上"里"地址固定"那一条的来源。

### 3. graph_capture 上下文

`parallel_state.py` 的 `GroupCoordinator.graph_capture` 把这些串起来：创建（或接收）一条捕获用的 stream，取出 `device_communicator.ca_comm`，进入 `ca_comm.capture()`，让调用者在 `torch.cuda.graph(...)` 里跑前向；退出时触发上面的注册。`CustomAllreduce.custom_all_reduce` 在 `_IS_CAPTURING` 为真且 `torch.cuda.is_current_stream_capturing()` 为真时走 `all_reduce(input, registered=True)`（不拷贝、直接用输入地址）；`_IS_CAPTURING` 为真但当前 stream 没在捕获（warmup 阶段）时只返回 `torch.empty_like(input)` 模拟分配模式，让 Caching Allocator 的地址序列与正式捕获一致。

PyNccl 的 `ncclAllReduce` 被捕获时，NCCL 内部的启动路径分成 `ncclLaunchKernelBefore_NoUncapturedCuda` / `ncclLaunchKernel` / `ncclLaunchKernelAfter_NoCuda` 几段（`src/enqueue.cc`，函数名本身就标出了哪些段不能有未被捕获的 CUDA 调用），kernel 与它的参数进入 graph，回放时 NCCL 的 host 侧不再参与——这时 NCCL 剩下的成本就只有 GPU 上的步数，第二章的对比表里"CUDA Graph 消除 launch 成本、不消除步数"就是这个意思。


## 六、算一算：PD 分离的 KV 传输

### 1. 每 token 每层多少字节

标准多头 / GQA attention 的 KV cache，每个 token 每层存 K 和 V 各 `num_kv_heads × head_dim` 个元素：

$$
B_{\text{token,layer}} = 2 \times \text{num\_kv\_heads} \times \text{head\_dim} \times \text{dtype\_bytes}
$$

vLLM 的 `AttentionSpec.real_page_size_bytes`（`vllm/v1/kv_cache_interface.py`）就是这个量乘上 `block_size`：`2 × block_size × num_kv_heads × head_size × dtype_size`（NVFP4 KV cache 走另一条带 block scale 的分支；`page_size_bytes` 在此之上再加 per-token-head scale 与 padding）。

Llama-3-70B：80 层，8 个 KV head（GQA），head_dim 128，BF16：

```text
每 token 每层     2 × 8 × 128 × 2 B  = 4 KB
每 token 全部层   4 KB × 80          = 320 KB
4096 token 的请求  320 KB × 4096     = 1.25 GiB ≈ 1.34 GB
8192 token 的请求                    = 2.5 GiB
TP8 时每个 rank    1 个 KV head      → 每 token 每层 512 B，4096 token 全部层 160 MiB
```

对照：MLA 架构（DeepSeek-V3 类）每 token 每层只存一个压缩的 latent（512 维）加 rope 部分（64 维），BF16 下 1152 字节，61 层共约 70 KB/token，比 GQA 少 4～5 倍，但 TP 各 rank 存的是同一份（tp_mapping.py 里 `is_mla` 分支对此有专门处理）。

传输时间的下界。默认节点每 GPU 一张 400 Gb/s 网卡（单向 50 GB/s）：

```text
TP8 → TP8，每 rank 传 160 MiB       160 MiB / 50 GB/s ≈ 3.4 ms（理论）；按 RDMA 通常 90% 效率约 3.7 ms
                                    8 个 rank 并行走 8 张网卡，请求整体也是 ≈ 3.4 ms
只有一张网卡承担全部 1.25 GiB          ≈ 27 ms
```

对比 prefill 本身：4096 token 的 70B 前向约 $$2 \times 70 \times 10^9 \times 4096 \approx 5.7 \times 10^{14}$$ FLOP，8 卡 H100 BF16 稠密算力 989 TFLOPS/卡（标称），按 50% 利用率约 145 ms。3.4 ms 的传输不到它的 3%；但如果传输被挤到一张网卡、或者没走 GPUDirect 经过了主机内存、或者 GPU 与 NIC 跨了 root complex（第二、三篇），27 ms 甚至更多就会显著吃掉 PD 分离本来要争取的 TTFT 收益。**KV 传输的账是带宽的账，而且是"每 GPU 一张网卡是否都被用上"的账**。

### 2. 什么时候传：一次性还是按层流水

两种时机。**prefill 结束后一次性传**：实现简单，传输与 prefill 完全串行，请求的首 token 延迟里多出完整的传输时间。**按层流水**：第 l 层的 attention 算完就开始传第 l 层的 KV，传输与后面各层的计算重叠，理想情况下只暴露最后一层的传输时间。后者显然更好，但对传输层的要求高得多：每层一次小传输（4096 token × 512 B = 2 MB/rank/层）、80 次提交、每次都要通知对端。

`KVConnectorBase_V1` 为两种时机都留了接口：`start_load_kv` / `wait_for_layer_load`（接收侧）、`save_kv_layer` / `wait_for_save`（发送侧），文档字符串写明 `wait_for_layer_load` "will be useful for layer-by-layer pipelining"。但 **vLLM 的 NixlConnector 在请求粒度传输，按层的钩子是空实现**（`vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py`）：

```python
def wait_for_layer_load(self, layer_name: str) -> None:
    """NixlConnector does not do layerwise saving."""
    pass

def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None:
    """NixlConnector does not save explicitly."""
    pass
```

MooncakeConnector 的两个方法同样是 `pass`。原因是传输方式决定的：NixlConnector 是 decode 侧发起的单边 READ，prefill 侧根本不"发送"，它只是在 prefill 完成后把 block 列表告诉 decode 侧，然后等 decode 侧读完发通知再释放 block；这个模式下"按层"没有发起者。传输与计算的重叠靠另一种方式实现：READ 由网卡 DMA 完成，不占 SM，decode 实例在读 KV 的同时可以继续跑别的请求的 decode；prefill 实例在被读的同时可以跑下一个请求的 prefill。重叠的粒度是请求之间而不是层之间。

### 3. 传给谁：异构 TP 映射

prefill 与 decode 实例的 TP 度可以不同——prefill 算力密集适合更大的 TP，decode 访存密集、要放更多 batch。KV head 在 TP 各 rank 之间是按 head 切的，所以 TP 度不同时 rank 之间的对应关系是 head 的对应关系。`tp_mapping.py` 的 `compute_tp_mapping` 分两种情况（以 decode 为本地、prefill 为远端）：

```python
# vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py，compute_tp_mapping，有删节
if transfer_topology.is_mla or tp_size >= remote_tp_size:
    # D (local TP) > P (remote TP): multiple local ranks read different chunks from
    # *one* remote rank, corresponding to different kv heads.
    attn_ranks = [tp_rank * remote_tp_size // tp_size]
else:
    # P (remote TP) > D (local TP): one local rank reads from multiple remote ranks.
    # GQA dedup: when K < remote_tp_size, several remote ranks hold the same KV head.
    abs_tp = remote_tp_size // tp_size
    start = tp_rank * abs_tp
    heads = np.arange(start, start + abs_tp) * total_num_kv_heads // remote_tp_size
    _, unique_idx = np.unique(heads, return_index=True)
    attn_ranks = (start + np.sort(unique_idx)).tolist()
```

- **decode TP ≥ prefill TP**（如 P 用 TP2、D 用 TP8）：每个 decode rank 只从一个 prefill rank 读，但只读那个 rank 的 KV 里属于自己 head 的那一段；`rank_offset_factor` 记录偏移，`add_remote_agent` 展开远端描述符时（`_build_fa_remote`）用它乘上远端 block 长度得到远端地址内的 head 偏移。源码注释提到 NVIDIA 的说法：这种"多个 D rank 读一个 P rank 的不同片段"正是为了让多张网卡同时工作、打满 IB。
- **prefill TP > decode TP**（如 P 用 TP8、D 用 TP2）：每个 decode rank 从多个 prefill rank 读，拼成自己的 head 集合。GQA 下 KV head 数可能小于 prefill 的 TP 度，多个 prefill rank 持有同一个 head 的副本，`np.unique` 去重只读一份。
- **MLA**：所有 rank 的 KV 相同，每个 decode rank 只需读一个 prefill rank，按 `tp_rank * remote_tp_size // tp_size` 分散到不同的远端 rank 上，让读负载均匀。

`_nixl_handshake` 里 `transfer_topo.handshake_target_ranks(remote_tp_size)` 用同一套逻辑决定要和远端哪几个 rank 握手——只和自己会读的那些 rank 建立连接。

### 4. 为什么不用 NCCL

到这里可以系统地回答。NCCL 是集合通信库，KV 传输的四个特征逐条与它的设计冲突：

- **点对点、对端动态。** NCCL communicator 是在固定的一组 rank 上初始化的（`ncclCommInitRank` 需要所有成员参与，第四篇），每对 prefill/decode 实例之间建一个 communicator 意味着实例增减时所有人重新初始化；而 PD 分离的路由是每个请求动态决定的。NIXL 的 agent 之间是按需握手的两两连接，加一个实例只影响与它有关的连接。
- **不需要归约。** NCCL 的 send/recv 是它最不擅长的操作：必须成对放在 group 里、kernel 要占 SM、经 channel buffer 中转。KV 传输需要的只是"把这 20480 个 8 KB 的块从那边的这些地址搬到这边的这些地址"，这正是第三篇讲的单边 RDMA READ/WRITE：一次 verbs 调用带一个 scatter-gather 列表，网卡 DMA 引擎直接读写两边的显存，对端的 CPU 与 GPU 都不参与。
- **必须与计算解耦。** NCCL 的通信是 stream 上的 kernel：它占 SM（第五篇讲过 `NCCL_NTHREADS` 与重叠的 SM 竞争），它的完成顺序与同一 stream 的计算 kernel 耦合，它需要两边在同一时刻都调用。KV 传输时 decode 实例正在跑 CUDA Graph 回放的 decode step，任何插入 stream 的 kernel 都会打断它；RDMA READ 由网卡完成，GPU 完全不知道有传输发生，完成状态由 CPU 线程轮询 completion。
- **失败隔离。** 一个 NCCL communicator 里任何一个 rank 出错，整个 communicator 要 abort；PD 部署里一个 prefill 实例挂掉不应该影响 decode 实例正在服务的其他请求。单边 RDMA 的 QP 是两两独立的，一条连接的失败只影响相关的请求，NixlConnector 里有 `_handle_failed_transfer`、`kv_load_failure_policy`（`recompute` 或 `fail`）这类按请求处理失败的逻辑。

所以 KV 传输层的"自然原语"是：内存注册（一次性，把 KV block 池注册成 MR）、单边 READ/WRITE（带地址列表的批量提交）、完成通知（notification，让对端知道可以释放 block）。NIXL、UCX、Mooncake Transfer Engine 提供的都是这三样。


## 七、看一看：KVConnector 与传输层

### 1. KVConnectorBase_V1：scheduler 侧与 worker 侧

`vllm/distributed/kv_transfer/kv_connector/v1/base.py` 的 `KVConnectorBase_V1` 是所有 KV connector 的基类，构造时带一个 `KVConnectorRole`（`SCHEDULER` 或 `WORKER`）。同一个 connector 类会被实例化两次：一次在 scheduler 进程，一次在每个 worker 进程，两边通过 `KVConnectorMetadata` 单向传递决策。文件头的文档字符串按角色列出了方法：

```text
Scheduler 侧（决定"传什么"）                        Worker 侧（执行"怎么传"）
  get_num_new_matched_tokens()                       register_kv_caches()         注册 KV block 池
  update_state_after_alloc()                         start_load_kv()              发起本步需要的加载
  build_connector_meta()      → KVConnectorMetadata → bind_connector_metadata()
  update_connector_output()   ← KVConnectorOutput  ← get_finished()              轮询完成的传输
  request_finished()                                 wait_for_layer_load() / save_kv_layer() / wait_for_save()
  take_events()                                      handle_preemptions()
```

本系列的边界（总纲）明确把"何时搬、搬谁"排除在外，所以 scheduler 侧的方法这里只需要知道一件事：它们的产出是每个请求的 `(本地 block id 列表, 远端 block id 列表, 远端 engine / host / port / tp 信息)`，打包进 `KVConnectorMetadata` 随 `SchedulerOutput` 发到 worker。worker 侧的全部工作是把这些 block id 变成地址、发起传输、轮询完成、报告回去。传输层只存在于 worker 侧。

`KVTransferConfig`（`vllm/config/kv_transfer.py`）的关键字段：`kv_connector`（类名，如 `NixlConnector`、`MooncakeConnector`，由 `factory.py` 的 `KVConnectorFactory.register_connector` 注册）、`kv_role`（`kv_producer` / `kv_consumer` / `kv_both`，NixlConnector 已把 `kv_both` 标为 deprecated）、`kv_buffer_device`（`cuda` / `cpu` / `xpu`，默认当前平台设备类型）、`kv_connector_extra_config`（connector 私有配置，NIXL 的 `backends`、Mooncake 的 `mooncake_protocol` 都从这里读）。

### 2. NixlConnector：注册、握手、READ、通知

`vllm/distributed/kv_transfer/kv_connector/v1/nixl/` 目录在 v0.23.0 里是一个包：`connector.py` 的 `NixlConnector` 是一层薄门面（文件头注释："thin facade that delegates to scheduler / worker"），按 `KVConnectorRole` 把调用转给 `scheduler.py` 的 `NixlConnectorScheduler` 或 `worker.py` 的 `NixlConnectorWorker`；`metadata.py` 放 `NixlAgentMetadata` / `NixlHandshakePayload` / `NixlConnectorMetadata` 这些消息结构，`tp_mapping.py` 放异构 TP 映射，`stats.py` 放统计与 Prometheus 指标，`utils.py` 放 ZMQ 上下文等小工具。传输方向只有一种：decode 侧发起 RDMA READ。worker 侧四个阶段：

**初始化 agent。** `NixlConnectorWorker.__init__` 从 `kv_connector_extra_config` 读 `backends`（默认 `["UCX"]`），只用 UCX 时以 `nixl_agent_config(num_threads=..., capture_telemetry=True)` 创建 NIXL agent，配置了非 UCX 后端时改用 `nixl_agent_config(backends=..., capture_telemetry=True)`（`NixlWrapper` 是 `nixl._api.nixl_agent` 的延迟导入别名，见 `vllm/distributed/nixl_utils.py`）。源码注释解释了为什么要限制 `num_threads`（默认 4）：每个 UCX 线程通过 DevX 分配 UAR（doorbell 页），过多会耗尽 Mellanox 网卡的 UAR 空间，导致同机的 NVSHMEM（DeepEP）初始化失败——这是第三篇 verbs 资源管理在生产里的具体表现。

**注册 KV block 池。** `register_kv_caches` 对每层的 KV cache 张量（或 `kv_buffer_device="cpu"` 时的 host 中转缓冲区）构造 `(base_addr, size, device_id, ...)` 描述，用 `get_reg_descs(caches_data, "VRAM" 或 "DRAM")` 生成注册描述符，然后：

```python
self.nixl_wrapper.register_memory(descs, backends=self.nixl_backends)
```

这一步对应第三篇的 `ibv_reg_mr`：pin 住整块 KV cache、建立 NIC 的地址翻译、（显存时）走 GPUDirect RDMA 的 `nvidia-peermem` 或 DMA-BUF 路径。KV cache 是引擎启动时一次性分配的大块显存，所以注册只做一次、在启动时付掉全部成本，之后每个 block 的传输都命中已注册区域——第三篇讲的 MR cache 问题在这里被设计绕开了。注意源码注释：FlashAttention 布局下 K 与 V 分别注册为两个 region（好处是天然支持 MLA 和 K/V 不连续的情况，代价是握手元数据大一些，约 8 KB 对 5 KB）；FlashInfer 布局下 K 与 V 在同一个 region 里，为了异构 TP 能分别索引 K/V，再把 `num_regions` 逻辑上翻倍（`virtually_split_kv_in_blocks`）。无论哪种，一个 block 都是一个传输单位。

之后 `register_local_xfer_handler` 用 `get_xfer_descs` + `prep_xfer_dlist("NIXL_INIT_AGENT", descs)` 把本地所有 (layer, block) 的地址预先展开成传输描述符列表并交给 NIXL 预处理，以后每次传输只传下标。

**握手。** `register_kv_caches` 末尾把 agent 元数据（`get_agent_metadata()`，包含 UCX 地址等）、每层 KV 基址、block 数与长度、layout、attention 后端名等打包成 `NixlAgentMetadata`，经 `NixlHandshakePayload` 带上兼容性 hash 保存。这份 payload 会被送到 scheduler 侧，由 `NixlConnectorScheduler` 的 `_nixl_handshake_listener` 线程在一个 ZMQ ROUTER socket 上对外提供（监听地址来自 `VLLM_NIXL_SIDE_CHANNEL_HOST` / `VLLM_NIXL_SIDE_CHANNEL_PORT`，默认 `localhost:5600`——这是 vLLM 自己的旁路控制通道，与数据面无关）。decode 侧第一次遇到某个 prefill engine 时，worker 的 `_nixl_handshake` 在后台线程里用 ZMQ REQ socket 发 `(GET_META_MSG, remote_rank)` 索取对端某个 TP rank 的元数据，先比对 `compatibility_hash`，再交给 `add_remote_agent`：计算 `TPMapping`（`compute_tp_mapping`），`_validate_remote_agent_handshake` 校验 block 长度与 layout 是否兼容，然后 `nixl_wrapper.add_remote_agent(agent_metadata)` 建立到对端的连接，并把对端每个 (layer, block) 的地址展开成远端描述符列表。源码注释提醒握手线程必须先 `set_device`：UCX 在没有 CUDA context 的线程里初始化会禁用 CUDA IPC，从而禁用节点内 NVLink 传输。

**READ 与通知。** 每个 decode step，`start_load_kv` 对元数据里新到的请求调用 `_read_blocks_for_req` → `_read_blocks`（`worker.py`）：

```python
# vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py，_read_blocks，有删节
notif_id = f"{remote_request_id}:{self.world_size}".encode()
# ...
remote_block_descs_ids = self._compute_desc_ids(block_ids=remote_block_ids, ...)
local_block_descs_ids = self._compute_desc_ids(block_ids=local_block_ids, ...)
handle = self.nixl_wrapper.make_prepped_xfer(
    "READ",
    local_xfer_side_handle, local_block_descs_ids,
    remote_xfer_side_handle, remote_block_descs_ids,
    notif_msg=notif_id,
)
self.nixl_wrapper.transfer(handle)            # 异步发起
self._recving_transfers[request_id].append(handle)
```

一次 `make_prepped_xfer("READ", ...)` 带上本地与远端各几千到几万个描述符下标——4096 token、block_size 16、80 层就是 256 × 80 = 20480 对 (local, remote) 8 KB 的块——NIXL 把它们变成 UCX 的批量 RDMA READ。`notif_msg` 是传输完成后自动发给对端的通知：内容是 `请求 id:本地 world_size`，让 prefill 侧知道有多少个 decode rank 会来读、全部读完才能释放 block。`_read_blocks` 里对"完全 prefix cache 命中、不需要读"的情况直接 `send_notif`，语义相同。

之后每一步 `get_finished` → `_pop_done_transfers` 用 `check_xfer_state(handle)` 轮询：`DONE` 则取 `get_xfer_telemetry` 记录字节数与耗时进 `NixlKVConnectorStats`，`PROC` 则继续等，其他状态按失败处理。prefill 侧 `_get_new_notifs` 收通知，计数达到 world_size 后把请求标为 `done_sending`，scheduler 侧才释放 block。整条路径上 GPU 没有执行任何与传输相关的指令。

### 3. NIXL 与 UCX

**NIXL**（NVIDIA Inference Xfer Library）是 NVIDIA 为推理数据搬运设计的库，出自 Dynamo 项目。它的抽象是：agent（一个进程一个）、内存注册（`register_memory`，支持 VRAM / DRAM / 文件等多种 memory type）、描述符列表、传输请求（READ / WRITE，带可选通知）、后端插件。后端里 UCX 负责网络与节点内 GPU 之间的传输，另有 GPUDirect Storage、POSIX 文件、对象存储等后端用于 KV 卸载到存储；vLLM 的 `backends` 配置默认只有 `UCX`。NIXL 的价值在于统一了"注册—描述—传输—通知"的接口，让 connector 不必直接面对 verbs 或 UCX 的 API，也让 KV 从显存传到远端显存、远端主机内存、本地 NVMe 用同一套代码。

**UCX**（Unified Communication X）是 HPC 领域的统一通信框架，Open MPI 等的默认传输层。它在一组 transport（`rc_verbs` / `rc_mlx5` / `dc_mlx5` 等 RDMA 传输、`tcp`、`shm` / `posix` / `sysv` 共享内存、`cuda_ipc`、`cuda_copy`、`gdr_copy`）之上做**自动选路**：两个 endpoint 之间根据可达性和性能估计选出最优 transport，同一节点内 GPU 之间选 `cuda_ipc`（走 NVLink，和第三章 custom AR 用的是同一种能力），跨节点显存之间选 RDMA 传输 + GPUDirect（`nvidia-peermem` 可用时），退化时用 `cuda_copy` 经主机内存中转，再退化用 `tcp`。UCX 的选路是它最强也最容易出问题的地方：走了 `tcp` 或 `cuda_copy` 时功能完全正常，只是带宽差一个数量级。控制它用 UCX 的标准环境变量：`UCX_TLS` 限定允许的 transport（如 `UCX_TLS=rc,cuda_copy,cuda_ipc`），`UCX_NET_DEVICES` 限定网卡（vLLM 的 NIXL 集成测试脚本里设了 `UCX_NET_DEVICES=all`），`UCX_LOG_LEVEL=info` 打印实际选择——这些是 UCX 的变量而不是 vLLM 的，以你安装的 UCX 版本文档为准。

UCX 与 NCCL 的 NET 传输解决的是同一层的问题——如何在 verbs 之上组织 RDMA 请求——但取向不同：NCCL 的 net_ib（第三、四篇）为集合通信优化，用 RDMA WRITE 推数据、proxy 线程配合 GPU kernel 的 flag 协议；UCX 为通用消息传递与单边操作优化，提供 tag matching、active message、`ucp_put` / `ucp_get`，GPU 只是内存的一种。

### 4. Mooncake Transfer Engine

`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py` 的 `MooncakeConnector` 用 Mooncake（Moonshot AI 的 KV 缓存与传输项目）的 Transfer Engine。worker 侧 `MooncakeConnectorWorker.__init__`：

```python
self.engine = TransferEngine()
protocol = kv_transfer_config.kv_connector_extra_config.get("mooncake_protocol", "rdma")
ret_value = self.engine.initialize(self.hostname, "P2PHANDSHAKE", protocol, "")   # 网卡由 Transfer Engine 自选
```

`register_kv_caches` 用 `engine.batch_register_memory(kv_data_ptrs, kv_data_lens)` 注册 KV 张量（与 NIXL 的 `register_memory` 对应）。与 NixlConnector 相反，Mooncake 路径是**prefill 侧推送**：`_send_blocks` 调 `engine.batch_transfer_sync_write(remote_session, src_ptrs, dst_ptrs, lengths)`，一次同步的批量 RDMA WRITE，在一个有 `num_workers`（默认 10）个线程的池里并发执行。Transfer Engine 自己处理多网卡的选择与分片、拓扑感知（选离 GPU 最近的网卡）、失败重试，协议可选 `rdma` / `tcp`。

NIXL 的 READ 和 Mooncake 的 WRITE 是单边 RDMA 的两个方向，第三篇讲过它们在 verbs 层的差别：WRITE 是发起方知道数据在哪、目标在哪，一个 WR 就走；READ 是发起方去拉，需要先知道对端地址与 rkey，完成时数据已在本地。选 READ 的好处是 decode 侧掌握节奏——它分配好 block 才去读，不需要 prefill 侧等 decode 侧准备好；选 WRITE 的好处是 prefill 完成即可推送、不必等对端来取。两者都不需要对端 CPU 与 GPU 参与数据搬运，这是与 NCCL 的本质差别。

### 5. kv_buffer_device 与 host buffer 路径

`kv_buffer_device="cuda"`（默认）时 NIXL 直接注册显存为 `VRAM`，传输走 GPUDirect RDMA（跨节点）或 CUDA IPC（节点内）。`kv_buffer_device="cpu"` 时 `use_host_buffer` 为真：`initialize_host_xfer_buffer` 分配与 KV cache 同形状的 pinned host 缓冲区注册为 `DRAM`，prefill 侧 `wait_for_save` → `save_kv_to_host` 用 `copy_blocks` 把 KV 从显存拷到 host 缓冲区，decode 侧读到 host 缓冲区后 `sync_recved_kv_to_device` 拷回显存。多两次 PCIe 拷贝（第二篇：x16 PCIe 5.0 单向 64 GB/s 标称），但不依赖 `nvidia-peermem`；在 GPUDirect RDMA 不可用（跨 root complex、驱动限制、云环境）的机器上这是可用的退路。排障时"KV 传输慢"的第一个检查项就是确认自己在哪条路径上。


## 八、测一测与比一比：推理侧的排障

### 1. TP all_reduce：先看走了哪个后端

TP all_reduce 延迟异常——decode 每 token 延迟比预期高、profiler 里通信 kernel 时间长——的排查顺序：

```text
1. 启动日志里找 "Using [...] all-reduce backends (in dispatch order) for group 'tp'"
   期望 8 卡 NVSwitch：['CUSTOM', 'SYMM_MEM', 'PYNCCL']（默认配置）
   只有 ['PYNCCL']  → custom AR 与 symm mem 都被禁用，查第 2 步
2. 往上翻 "Custom allreduce is disabled because ..." 的 warning，对照第三章第 4 节的清单：
   - "unsupported world size"          → TP 不在 {2,4,6,8}
   - "spans across nodes"              → TP group 跨了节点（另见第 6 步）
   - "not supported on more than two PCIe-only GPUs" → NVLink 全互联检测失败：nvidia-smi topo -m 看是否全是 NV#
   - "lacks GPU P2P capability or P2P test failed" → 删掉 VLLM_CACHE_ROOT 下 gpu_p2p_access_cache_for_*.json 重测；
                                                     容器里检查 IPC namespace（--ipc=host）与 /dev/shm
   - "missing custom allreduce library" → 安装的 vLLM 没带 custom op（CPU 版或编译不全）
   "SymmMemCommunicator: ... multicast operations are not supported" → 无 NVSwitch 多播，symm mem 禁用，正常
3. 确认 --disable-custom-all-reduce 没有被设置；VLLM_BATCH_INVARIANT=1 会隐式关闭它
4. profiler（torch profiler，vLLM 的 profiler_config 或 LLM.start_profile / stop_profile）里看 kernel 名：
   cross_device_reduce_1stage / 2stage → custom AR；multimem_all_reduce_kernel → torch symm mem；
   ncclDevKernel_AllReduce_* → PyNccl
5. 消息大小是否落在预期区间：batch 大时 128 KB 会变成几 MB，自然切到 symm mem 或 PyNccl，这不是异常
6. TP 是否跨了节点（多机 TP）：每层两次 all_reduce 走 IB，延迟几十 µs 是正常的，此时该改 PP 而不是调通信
7. 与 nccl-tests 对照：同一台机器 all_reduce_perf -b 16K -e 4M -g 8 的小消息端延迟（第六篇的读法），
   PyNccl 的每次 all_reduce 不应明显高于它；custom AR 应明显低于它
```

### 2. KV 传输慢：从路径到亲和

KV 传输的症状是 decode 侧 TTFT 高、`vllm:nixl_xfer_time_seconds` 直方图偏大、或日志里 `NixlKVConnectorStats` 的 `Throughput (MB/s)` 远低于网卡带宽。排查顺序：

```text
1. 算理论值：每请求字节数（第六章公式）/ 每 rank 网卡单向带宽；Throughput (MB/s) 应达到它的 70–90%
2. 走的是哪条路径？
   - kv_buffer_device 是 cuda 还是 cpu（cpu 多两次 PCIe 拷贝，且 host 缓冲区受 PCIe 带宽限制）
   - UCX_LOG_LEVEL=info 看 UCX 选了哪个 transport：跨节点期望 rc_mlx5 / dc_mlx5 + GPUDirect；
     看到 tcp 或 cuda_copy 说明退化了
   - GPUDirect RDMA 是否可用：lsmod | grep nvidia_peermem（或 DMA-BUF 支持）；ib_write_bw --use_cuda 能否跑通
     （第三篇的检查）
3. GPU 与 NIC 亲和：nvidia-smi topo -mp 看每个 GPU 到其网卡是 PIX/PXB 还是 SYS（第二篇）；
   UCX_NET_DEVICES 限定每个 rank 用离自己最近的网卡，否则 8 个 rank 可能挤到同一张卡上
4. 内存注册：register_kv_caches 只在启动时做一次，看启动日志 "Registering KV_Caches" 后是否有注册失败或
   回退；注册显存失败通常意味着 peermem 不可用，UCX 会静默退到 cuda_copy
5. 描述符数量：block_size 小、层数多时一次传输有数万个 8 KB 描述符，NIC 的每描述符开销开始显现；
   HND layout 与 enable_cross_layers_blocks 让每个 block 的 K/V 或多层连续，减少描述符数
6. 与 ib_write_bw 对照：两台机器之间 ib_write_bw -d <hca> --use_cuda=<gpu> -s 8192 -q 4 的带宽是网卡 + GDR
   这条路径的上限；NIXL 的 Throughput 与它的差距就是软件栈（描述符、线程、握手）的开销
7. 异构 TP：D TP > P TP 时多个 D rank 读同一个 P rank，P 侧那张网卡的出向带宽被共享；看 P 侧网卡计数器
```

### 3. 两条路径共用的检查项

- **容器与 IPC。** custom AR 的 `cudaIpcOpenMemHandle` 与 UCX 的 `cuda_ipc` 都要求进程在同一个 IPC namespace；容器化部署要 `--ipc=host` 或共享 IPC namespace，否则 P2P 检测失败、UCX 节点内退化到 `cuda_copy`。
- **`CUDA_VISIBLE_DEVICES` 与物理 id。** `CustomAllreduce.__init__` 先按 `CUDA_VISIBLE_DEVICES` 把本 rank 的设备号换算成物理 id 并在 group 内 `all_gather`，再把物理 id 列表交给 `is_fully_connected` 查 NVML；P2P 缓存文件名带 `CUDA_VISIBLE_DEVICES`。不同实例用不同的可见设备集合时缓存互不干扰，但改了设备映射要删缓存。
- **多张网卡与 NCCL 的关系。** 同一台机器上 TP 的 PyNccl 兜底（跨节点时）与 KV 传输的 UCX 共用网卡，`NCCL_IB_HCA` 与 `UCX_NET_DEVICES` 要一致地按 GPU 亲和分配。


## 九、本文小结

### 1. 要点回顾

```text
decode TP all_reduce      S = batch_tokens × hidden × 2 B，batch 8 / hidden 8192 → 128 KB；每步 2 × 层数 次
                          NVLink 上 S/β < 1 µs，全部时间是 α；优化目标只有一个：压低每次调用的固定开销
NCCL 的固定开销            调优表模型：Ring+LL 8 卡 ≈ 6.6 + 14 × 0.6 = 15 µs；launch、2(n-1) 步串行握手、LL 协议；
                          节点内 P2P 数据路径不经 proxy，跨节点每步经 proxy
custom all-reduce         IPC 映射对端显存 + Signal flag 同步 + 一个 36 block 的 kernel；one-shot 2 次 barrier；
                          two-shot 2 次 barrier + 中间缓冲区一轮读写；8 卡 < 256 KB 用 one-shot；bit 级一致的累加顺序
20 µs 省在哪里             launch 路径、14 步 → 2 步、无 channel buffer 中转、天然可捕获进 CUDA Graph；没有"带宽"这一项
为什么不能用于训练          消息上限 8 MB、只能节点内、地址必须 IPC 注册且固定、中间缓冲区容量有限、无重叠 / 无容错
后端选择链（v0.23.0）      NCCL symm mem（可选）→ quick reduce（ROCm）→ FlashInfer（可选）→ custom AR
                          → torch symm mem → PyNccl → torch.distributed；日志 "Using [...] all-reduce backends"
PyNccl                    ctypes 加载 libnccl.so（VLLM_NCCL_SO_PATH）；stream 由调用者传、无 watchdog、可捕获
对称内存                   torch symm mem：H100 8 卡 256 KB ～ 64 MB 用 multimem（NVSwitch 多播归约）；
                          NCCL symm mem：ncclMemAlloc + ncclCommWindowRegister，让 NCCL 自己少付固定开销
CUDA Graph                捕获期间 RankData 槽位预留、register_graph_buffers 事后交换 IPC 句柄填对端地址
KV 传输的账                每 token 每层 2 × num_kv_heads × head_dim × dtype；Llama-3-70B 320 KB/token，4096 token
                          1.25 GiB；TP8 每 rank 160 MiB，400 Gb/s 网卡 ≈ 3.4 ms（理论）
什么时候传 / 传给谁         vLLM 按请求粒度、decode 侧 READ；按层钩子是空实现；异构 TP 按 KV head 映射（tp_mapping.py）
为什么不用 NCCL            点对点 + 动态对端 + 无归约 + 与计算解耦 + 按请求隔离失败 → 单边 RDMA READ/WRITE
KVConnector               scheduler 侧决定 block 列表，worker 侧注册 / 握手 / 传输 / 轮询；传输层只在 worker 侧
NIXL / UCX / Mooncake     NIXL：register_memory、make_prepped_xfer("READ")、notif；UCX 自动选路（cuda_ipc / rc / tcp）；
                          Mooncake：batch_transfer_sync_write 推送
```

### 2. 排障检查项 / 决策要点

```text
TP all_reduce 慢     → 启动日志后端列表 → custom AR 禁用原因 warning → profiler kernel 名 → 消息大小区间 → TP 是否跨节点
                       → 与 all_reduce_perf 小消息端对照
KV 传输慢            → 理论值（字节 / 每 rank 网卡带宽）→ kv_buffer_device → UCX transport（UCX_LOG_LEVEL）→ GDR 可用性
                       → GPU–NIC 亲和（topo -mp、UCX_NET_DEVICES）→ 注册日志 → 描述符数量 → 与 ib_write_bw --use_cuda 对照
决策：TP 放哪里       → 节点内、NVLink 全互联；跨节点用 PP
决策：KV 用什么传     → 默认 NixlConnector（UCX），显存直传；GDR 不可用时 kv_buffer_device=cpu；已有 Mooncake 生态用 Mooncake
决策：何时自己写原语   → 消息 < 几百 KB、节点内、地址固定、每步上百次、失败可整体重启——四条都满足才值得
```

### 3. 本篇涉及的源码与工具位置

| 内容 | 位置 |
|---|---|
| custom AR kernel | `csrc/custom_all_reduce.cuh`：`cross_device_reduce_1stage`、`cross_device_reduce_2stage`、`CustomAllreduce::allreduce` / `register_buffer` / `register_graph_buffers` / `get_graph_buffer_ipc_meta` |
| flag 同步与共用结构 | 同一文件 `csrc/custom_all_reduce.cuh`：`Signal`、`RankData`、`RankSignals`、`barrier_at_start`、`barrier_at_end`、`st_flag_release` / `ld_flag_acquire`、`packed_reduce`、`kMaxBlocks` / `defaultBlockLimit` |
| torch 绑定 | `csrc/libtorch_stable/custom_all_reduce.cu`：`init_custom_ar`、`all_reduce`、`meta_size`、`allocate_shared_buffer_and_handle`、`open_mem_handle` |
| Python 侧 custom AR | `vllm/distributed/device_communicators/custom_all_reduce.py`：`CustomAllreduce.__init__` / `should_custom_ar` / `custom_all_reduce` / `capture` / `register_graph_buffers` / `create_shared_buffer`、`_can_p2p` |
| 阈值表与 P2P 检测 | `vllm/distributed/device_communicators/all_reduce_utils.py`：`CUSTOM_ALL_REDUCE_MAX_SIZES`、`SYMM_MEM_ALL_REDUCE_MAX_SIZES`、`NCCL_SYMM_MEM_ALL_REDUCE_CONFIG`、`should_nccl_symm_mem_allreduce`、`gpu_p2p_access_check`、`can_actually_p2p` |
| 后端选择链 | `vllm/distributed/device_communicators/cuda_communicator.py`：`CudaCommunicator.__init__` / `all_reduce` / `_log_all_reduce_backend_selection` |
| PyNccl | `vllm/distributed/device_communicators/pynccl.py`：`PyNcclCommunicator`、`register_nccl_symmetric_ops`；`pynccl_wrapper.py`：`NCCLLibrary`；`vllm/utils/nccl.py`：`find_nccl_library`；`pynccl_allocator.py`：`is_symmetric_memory_enabled` |
| 对称内存 | `vllm/distributed/device_communicators/symm_mem.py`：`SymmMemCommunicator`；PyTorch `torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryOps.cu`：`multimem_all_reduce_kernel` |
| FlashInfer / quick reduce | `flashinfer_all_reduce.py`：`FlashInferAllReduce.should_use_fi_ar`；`quick_all_reduce.py`：`QuickAllReduce`（ROCm） |
| group 管理 | `vllm/distributed/parallel_state.py`：`GroupCoordinator`（`all_reduce`、`_all_reduce_out_place`、`graph_capture`、`use_custom_op_call`）、`get_tp_group` / `get_pp_group` / `get_dp_group`、`set_custom_all_reduce`、`initialize_model_parallel`、`in_the_same_node_as` |
| NVLink 全互联检测 | `vllm/platforms/cuda.py`：`is_fully_connected`（`nvmlDeviceGetP2PStatus`） |
| 配置 | `vllm/config/parallel.py`：`ParallelConfig.disable_custom_all_reduce`；`vllm/config/kv_transfer.py`：`KVTransferConfig`（`kv_connector`、`kv_role`、`kv_buffer_device`、`kv_connector_extra_config`） |
| 环境变量 | `vllm/envs.py`：`VLLM_NCCL_SO_PATH`、`VLLM_ALLREDUCE_USE_SYMM_MEM`、`VLLM_ALLREDUCE_USE_FLASHINFER`、`VLLM_USE_NCCL_SYMM_MEM`、`VLLM_DISABLE_PYNCCL`、`VLLM_SKIP_P2P_CHECK`、`VLLM_BATCH_INVARIANT`、`VLLM_NIXL_SIDE_CHANNEL_HOST` / `_PORT`、`VLLM_KV_CACHE_LAYOUT`；`csrc` 里的 `VLLM_CUSTOM_ALLREDUCE_ALGO` |
| KVConnector 基类 | `vllm/distributed/kv_transfer/kv_connector/v1/base.py`：`KVConnectorBase_V1`、`KVConnectorRole`；`factory.py`：`KVConnectorFactory.register_connector` |
| NIXL connector | `vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py`（`NixlConnector` 门面，空的 `wait_for_layer_load` / `save_kv_layer`）、`scheduler.py`（`NixlConnectorScheduler`、`_nixl_handshake_listener`）、`worker.py`（`NixlConnectorWorker.register_kv_caches` / `register_local_xfer_handler` / `_nixl_handshake` / `add_remote_agent` / `_read_blocks` / `_pop_done_transfers` / `_get_new_notifs` / `_handle_failed_transfer`）、`metadata.py`（`NixlAgentMetadata`、`NixlHandshakePayload`、`NixlConnectorMetadata`）、`tp_mapping.py`（`compute_tp_mapping`、`TPMapping`）、`stats.py`（`NixlKVConnectorStats`、`NixlPromMetrics`） |
| Mooncake connector | `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py`：`MooncakeConnectorWorker`（`TransferEngine.initialize`、`batch_register_memory`、`batch_transfer_sync_write`） |
| KV 页大小 | `vllm/v1/kv_cache_interface.py`：`AttentionSpec.real_page_size_bytes` / `page_size_bytes` |
| NCCL 调优常数 | NCCL 2.28.9 `src/graph/tuning.cc`：`ncclTunerConstantsDefaults`（`baseLatencies`、`hwLatencies`）与 ring 延迟公式；`src/enqueue.cc`：`ncclLaunchPrepare`、`ncclLaunchKernel` |
| 工具 | `nvidia-smi topo -m` / `-mp`、`lsmod`（查 `nvidia_peermem`）、`ib_write_bw --use_cuda`、nccl-tests `all_reduce_perf`、torch profiler、UCX 的 `UCX_TLS` / `UCX_NET_DEVICES` / `UCX_LOG_LEVEL` |

### 4. comm-probe 本篇增量

本篇给 comm-probe 加两件东西：`tp_ar_bench.py` 测 vLLM 各 all_reduce 后端的每次调用延迟，`kv_xfer/` 搭一对 PD 实例测 KV 传输带宽。

**`tp_ar_bench.py`**：用 `torchrun` 启动 N 个进程，直接初始化 vLLM 的分布式环境与 TP group（与 `tests/distributed/test_custom_all_reduce.py` 相同的方式），对一组消息大小分别在 eager 与 CUDA Graph 下测 `get_tp_group().all_reduce` 的延迟，并与 PyNccl 单独调用对照。开关后端不需要改源码：`set_custom_all_reduce(False)` 关掉 custom AR（等价于 `--disable-custom-all-reduce`），环境变量 `VLLM_ALLREDUCE_USE_SYMM_MEM=0` 关掉 torch symm mem，两者都关就只剩 PyNccl。

```python
# comm-probe/tp_ar_bench.py —— torchrun --nproc_per_node 8 tp_ar_bench.py [--no-custom-ar]
# 对照：VLLM_ALLREDUCE_USE_SYMM_MEM=0 torchrun ... ；nccl-tests: all_reduce_perf -b 16K -e 4M -f 2 -g 8
import argparse, os, torch, torch.distributed as dist
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (init_distributed_environment, ensure_model_parallel_initialized,
                              get_tp_group, graph_capture)
from vllm.distributed.parallel_state import set_custom_all_reduce

def bench(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters          # µs / call

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-custom-ar", action="store_true")
    ap.add_argument("--hidden", type=int, default=8192); args = ap.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    set_custom_all_reduce(not args.no_custom_ar)     # == ParallelConfig.disable_custom_all_reduce
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(world, rank, "env://", rank)
        ensure_model_parallel_initialized(tensor_model_parallel_size=world, pipeline_model_parallel_size=1)
    tp = get_tp_group(); dc = tp.device_communicator
    if rank == 0:
        print("backends:", "CUSTOM" if dc.ca_comm and not dc.ca_comm.disabled else "-",
              "SYMM_MEM" if dc.symm_mem_comm and not dc.symm_mem_comm.disabled else "-", "PYNCCL")
        print(f"{'tokens':>6} {'bytes':>9} {'eager_us':>9} {'graph_us':>9} {'pynccl_us':>10}")
    for tokens in (1, 8, 32, 128, 512):
        x = torch.randn(tokens, args.hidden, dtype=torch.bfloat16, device="cuda")
        eager = bench(lambda: tp.all_reduce(x))
        with graph_capture(device=x.device) as ctx:                 # 触发 ca_comm.capture()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=ctx.stream):
                y = tp.all_reduce(x)
        graph = bench(g.replay)
        pyn = bench(lambda: dc.pynccl_comm.all_reduce(x)) if dc.pynccl_comm else float("nan")
        dist.barrier()
        if rank == 0:
            print(f"{tokens:>6} {x.numel() * 2:>9} {eager:>9.1f} {graph:>9.1f} {pyn:>10.1f}")

if __name__ == "__main__":
    main()
```

读法：`eager_us` 一列在 custom AR 开启时应明显低于 `pynccl_us`，且随字节数几乎不变（纯 α）；`graph_us` 进一步去掉 launch 成本，是 decode 真实看到的数字；`tokens` 增大到 custom AR 的 `max_size` 之上（默认 8 MB，开 symm mem 时 H100 8 卡为 256 KB，即 tokens = 16）后 `eager_us` 会跳到另一个后端的水平——这个跳变点就是第四章第 5 节那张表的实测版。`pynccl_us` 应与同机 `all_reduce_perf` 相同字节数的 `time(us)` 列接近；两者差得远，先查 nccl-tests 那边的环境（第六篇）。

**`kv_xfer/`**：一个 `run_pd.sh` 启动两个 vLLM 实例（prefill 用 `kv_role=kv_producer`、decode 用 `kv_consumer`，都用 `NixlConnector`），一个 `probe.py` 发一批固定长度的请求，然后从 decode 实例的日志或 `/metrics` 里取 `vllm:nixl_xfer_time_seconds` 与 `vllm:nixl_bytes_transferred` 算出实际带宽，与第三篇 `ib_write_bw --use_cuda` 的结果和第六章的理论值并排打印。关键的启动参数：

```bash
# comm-probe/kv_xfer/run_pd.sh —— 单机两实例示意；跨机时把 kv_ip / side channel host 换成对端地址
MODEL=${MODEL:-meta-llama/Llama-3.1-70B-Instruct}
KV_P='{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_buffer_device":"cuda"}'
KV_D='{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_buffer_device":"cuda"}'
# UCX 标准变量：限定 transport 与网卡，按 nvidia-smi topo -mp 的亲和为每个实例选网卡
export UCX_TLS=${UCX_TLS:-rc,cuda_copy,cuda_ipc}
export UCX_LOG_LEVEL=${UCX_LOG_LEVEL:-info}
CUDA_VISIBLE_DEVICES=0,1,2,3 UCX_NET_DEVICES=${P_NICS:-all} VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
  vllm serve $MODEL --port 8100 --tensor-parallel-size 4 --kv-transfer-config "$KV_P" &
CUDA_VISIBLE_DEVICES=4,5,6,7 UCX_NET_DEVICES=${D_NICS:-all} VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
  vllm serve $MODEL --port 8200 --tensor-parallel-size 4 --kv-transfer-config "$KV_D" &
wait
```

{% raw %}
```python
# comm-probe/kv_xfer/probe.py —— 取 decode 侧 /metrics 里的 NIXL 直方图算带宽，与理论值、ib_write_bw 对照
import re, sys, urllib.request
def hist_sum(text, name):                 # prometheus 直方图的 _sum
    m = re.search(rf'^{name}_sum(?:{{[^}}]*}})? ([0-9.e+-]+)', text, re.M)
    return float(m.group(1)) if m else 0.0
metrics = urllib.request.urlopen(f"http://{sys.argv[1]}/metrics").read().decode()
secs, byts = hist_sum(metrics, "vllm:nixl_xfer_time_seconds"), hist_sum(metrics, "vllm:nixl_bytes_transferred")
nic_gbps, ib_write_bw_gbs = float(sys.argv[2]), float(sys.argv[3])   # 例：400  45.2（ib_write_bw --use_cuda 实测 GB/s）
print(f"NIXL:        {byts/1e9:.2f} GB in {secs:.3f} s → {byts/secs/1e9:.1f} GB/s per rank (avg over transfers)")
print(f"theory:      {nic_gbps/8:.1f} GB/s per NIC (unidirectional)   ib_write_bw: {ib_write_bw_gbs:.1f} GB/s")
print(f"efficiency:  {byts/secs/1e9/(nic_gbps/8)*100:.0f}% of NIC, {byts/secs/1e9/ib_write_bw_gbs*100:.0f}% of ib_write_bw")
```
{% endraw %}

注意 `probe.py` 里的每 rank 带宽是"传输进行期间"的平均值——`nixl_xfer_time_seconds` 只计时传输本身，不含握手与排队，所以它衡量的是数据面效率，正好用来和 `ib_write_bw` 比。差距在 10–20% 内是正常的软件开销；差一倍以上去查第八章第 2 节的清单——多数情况是 UCX 选了 `cuda_copy` 或 `tcp`，或者 8 个 rank 的流量挤在一两张网卡上。

到这里，七篇文章各自给出的示例代码合起来就是 comm-probe 的全部（它们随文给出、由读者自行保存成对应文件，不是一个已发布的软件包）：`cost_model.py` 算理论值，`topo_map.py` 画拓扑并记录链路实测，`rdma_write.c` 验证 GDR 路径，`nccl_log_reader.py` 读 NCCL 的决策，`overlap_bench.py` 检查框架侧的重叠，`sweep.sh` 与 `hang_lab/` 跑 nccl-tests 曲线与 hang 剧本，`tp_ar_bench.py` 与 `kv_xfer/` 验收推理侧的两个后端。


## 系列总结

七篇文章走了一条从抽象到物理再回到软件的路：

```text
第一篇  代价模型        T = α + S/β；ring 的 2(n-1)α + (2(n-1)/n)·S/β；algbw 与 busbw；消息大小的谱
第二篇  硬件互联        PCIe 各代单向带宽、NVLink 每 GPU 双向合计 600 / 900 GB/s / 1.8 TB/s、IB HDR / NDR；
                        nvidia-smi topo 的六个等级；NUMA 与亲和；每 GPU 一张网卡的理由
第三篇  RDMA 与 GDR     verbs：PD / MR / QP / CQ / WR；单边 WRITE / READ 与双边 SEND / RECV；GPUDirect RDMA 的
                        PCIe 拓扑条件；注册的代价；ib_write_bw --use_cuda
第四篇  NCCL 架构       bootstrap → 拓扑探测 → 图搜索 → transport → 调优表 → enqueue → kernel → proxy；
                        Ring / Tree / NVLS；Simple / LL / LL128；channel
第五篇  PyTorch 通信栈   ProcessGroupNCCL 的 stream 与 event 语义、Work 与 wait 的含义、recordStream、重叠的条件、
                        watchdog 与 timeout、对称内存
第六篇  测量与排障       nccl-tests 曲线的读法、调优参数的层次、hang 的分类、Flight Recorder、决策树
第七篇  推理侧          decode TP all_reduce 的纯延迟账与 custom all-reduce；后端选择链与 PyNccl；CUDA Graph；
                        KV 传输的带宽账与 NIXL / UCX / Mooncake 的单边 RDMA
```

贯穿始终的是**两种账**。带宽的账看链路速率、算法的带宽效率（ring 的 $$\frac{2(n-1)}{n}$$、tree 的一半、NVLS 的一步）、协议开销（LL 的 50%、LL128 的 94%）；延迟的账看步数、握手次数、kernel 启动、proxy 的响应。每一处取舍两本账的答案都相反：更多 channel 提高带宽却增加小消息延迟，Tree 降低延迟却在某些拓扑上损失带宽，custom all-reduce 把步数从 14 压到 2 却放弃了大消息的带宽与跨节点的能力，KV 传输用单边 RDMA 跑满网卡却放弃了 NCCL 的集合语义。分清一次通信在算哪本账，是判断"该换算法、该换硬件、还是什么都不用换"的前提；本系列的每一篇都在各自的层上把两本账各算了一遍。

方法是同一个四段法：**算一算**用代价模型给出理论上限，**看一看**读源码、日志、拓扑文件弄清这一层怎么做决定，**测一测**用 nvbandwidth、ib_write_bw、nccl-tests、profiler 测出实际数字，**比一比**解释差距并落成检查清单。它的产物是 comm-probe：

```text
cost_model.py        α-β 模型 · ring / tree 预测 · algbw 与 busbw 换算                       第一篇
topo_map.py          解析 nvidia-smi topo / lspci / NUMA · 画拓扑图 · 记录 nvbandwidth 与 ib_write_bw   第二篇
rdma_write.c         libibverbs 最小 RDMA WRITE · 显存版（GPUDirect RDMA）· 与 ib_write_bw 对照      第三篇
nccl_log_reader.py   解析 NCCL_DEBUG=INFO 日志 · 提取 ring / tree / channel / 算法协议决策 · 与预测比对  第四篇
overlap_bench.py     计算与通信重叠的 micro-benchmark · profiler trace 检查 · 重叠失效的复现集        第五篇
sweep.sh · hang_lab/ nccl-tests 扫描与画图 · 三种 hang 的复现与定位剧本 · 排障决策树                第六篇
tp_ar_bench.py       vLLM 各 all_reduce 后端延迟对照 · eager / CUDA Graph / PyNccl · 与 nccl-tests 对照  第七篇
kv_xfer/             两实例 PD 分离 · NixlConnector · 从 /metrics 取 NIXL 带宽 · 与 ib_write_bw 对照      第七篇
```

拿到一台新机器，按顺序跑一遍：先画拓扑，再测每段链路，再跑 nccl-tests 与理论对照，再检查框架侧的重叠与后端选择。之后每一次"通信慢了"或"通信卡了"，都能用它把问题定位到某一层。

总纲承诺的三种能力，现在可以逐条对照：

1. **阅读能力。** NCCL 的主路径（`init.cc` → `graph/` → `transport/` → `enqueue.cc` → `device/`）、PyTorch c10d 的 `ProcessGroupNCCL`、vLLM 的 `CudaCommunicator` 后端链与 `KVConnector` 的 scheduler / worker 分工，每个设计决定背后都能指出硬件原因：为什么 LL 协议要 8 字节搭 8 字节、为什么 ProcessGroupNCCL 要有内部 stream 与 watchdog、为什么 custom all-reduce 只用 36 个 block 且只能在 NVLink 全互联的节点内用、为什么 KV 传输要用单边 RDMA 而不是 NCCL。
2. **诊断能力。** 面对一次慢的或卡住的通信：先算理论值（第一篇），再看数据走了哪条链路（第二、三篇），再看 NCCL 或 vLLM 选了什么（第四、七篇的日志），再看框架侧有没有浪费（第五篇），再用 nccl-tests 与 Flight Recorder 把差距或 hang 定位到具体的一层（第六篇）——而不是靠试环境变量。
3. **决策能力。** 为一个任务判断通信的理论上限、选择算法与传输路径、给平台提出拓扑与亲和的要求（每 GPU 一张网卡、GPU 与 NIC 同一 PCIe switch、TP 不跨节点、容器共享 IPC namespace），并知道什么时候该自己写一个通信原语：消息小、节点内、地址固定、每步上百次、失败可整体重启——四条都满足才值得，否则用 NCCL。

通信层是单卡之外一切系统的底座，也是训练与推理两条路径唯一共享的一层。这个系列把它从 `dist.all_reduce(t)` 一行代码展开到 PCIe、NVLink、InfiniBand 上的每一段路，再收回到 vLLM 的两个 kernel 和一次 RDMA READ。展开是为了看清代价，收回是为了在正确的层上做决定。


## 系列目录

1. [集合通信原语与代价模型：α-β 模型与 ring all-reduce](/collective-communication-primitives-and-cost-model.html)
2. [硬件互联：PCIe、NVLink、NVSwitch 与网络拓扑](/hardware-interconnect-pcie-nvlink-and-topology.html)
3. [RDMA 与 GPUDirect：绕过 CPU 和主机内存的数据通路](/rdma-and-gpudirect.html)
4. [NCCL 架构：拓扑探测、channel、算法与协议](/nccl-architecture-topology-channels-algorithms-and-protocols.html)
5. [PyTorch 的通信栈：ProcessGroupNCCL、stream 语义与计算通信重叠](/pytorch-communication-stack-processgroupnccl-and-streams.html)
6. [nccl-tests、调优与排障：从带宽曲线到 hang](/nccl-tests-tuning-and-debugging-hangs.html)
7. [推理侧的通信：custom all-reduce 与 KV 传输](/inference-communication-custom-all-reduce-and-kv-transfer.html)
