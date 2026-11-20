---
layout: slides
title: "通信与互联：从 NCCL 到 RDMA"
subtitle: "系列精华 · 八篇正文每篇一页，每页算两本账：带宽的账与延迟的账"
permalink: /slides/communication-and-interconnect.html
series: communication-and-interconnect
date: 2026-06-27 23:30:00 +0800
author: arganzheng
description: "《通信与互联：从 NCCL 到 RDMA》系列的分享用幻灯片：α-β 模型与 ring all-reduce、PCIe / NVLink / IB 的真实带宽、RDMA 与 GPUDirect 的三条路径、NCCL 的拓扑探测与调优表、ProcessGroupNCCL 的 stream 语义、nccl-tests 曲线与六类 hang、custom all-reduce 与 KV 传输、MoE 的 all-to-all 与 DeepEP。"
theme: white
transition: slide
---

## 这个系列的一句话主张

> 每一次通信都在算**两本账——带宽的账与延迟的账**——分清在算哪本账，才知道该换算法、该换硬件、还是什么都不用换。

$$
T = \underbrace{\text{步数} \times \alpha}_{\text{延迟的账}} + \underbrace{\text{每 rank 字节数} / \beta}_{\text{带宽的账}},\qquad
\text{拐点}\ S^* = n\,\alpha\,\beta
$$

| 账 | 看什么 |
|---|---|
| 带宽 $$\beta$$ | 链路速率、算法的带宽效率、协议的有效载荷比例 |
| 延迟 $$\alpha$$ | 步数、握手次数、kernel 启动、proxy 线程响应；MoE decode 再加**发起速率** |

八篇同一个四段法：算一算 → 看一看 → 测一测 → 比一比；留下诊断工具集 comm-probe。

<aside class="notes" markdown="1">
总纲：/communication-and-interconnect-for-ai-infra.html。
</aside>

---

## 八篇怎么连起来：先造尺子，再填数

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 180}}}%%
flowchart TB
    N1["01 α-β 模型与 ring all-reduce"]
    N1 --> N2["02 硬件互联"]
    N1 --> N3["03 RDMA 与 GPUDirect"]
    N2 & N3 --> N4["04 NCCL 架构"]
    N4 --> N5["05 PyTorch 的通信栈"]
    N5 --> N6["06 nccl-tests 与排障"]
    N4 & N5 --> N7["07 推理侧的通信"]
    N4 & N5 --> N8["08 MoE 的通信"]
```

<!-- v -->

### 一次 all_reduce 的全路径：两本账在末端分叉

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 140}}}%%
flowchart LR
    PY["<b>dist.all_reduce</b><br/>c10d、Work、stream（05）"]
    H["<b>NCCL host 侧</b><br/>查调优表选算法与协议（04）"]
    D["<b>NCCL kernel</b><br/>每 channel 一个 block（04）"]
    NVL["<b>节点内 NVLink</b><br/>单向 450 GB/s（02）"]
    NET["<b>proxy 上网卡</b><br/>RDMA + GDR，50 GB/s（02、03）"]
    ACC{"落在 α-β 拐点哪一侧（01）"}
    BW["<b>带宽的账</b><br/>换链路、换算法、合并消息"]
    LAT["<b>延迟的账</b><br/>减步数，逼出 custom AR（07）与 IBGDA（08）"]
    PY --> H --> D
    D --> NVL --> ACC
    D --> NET --> ACC
    ACC -->|"大"| BW
    ACC -->|"小"| LAT
```

<aside class="notes" markdown="1">
总纲「一次 all_reduce 的数据路径」那张全篇机制图的精简版：每一篇负责这条路径上的一段，第六篇的测量与排障贯穿全程。
</aside>

---

## 01 · α-β 模型：1 GB 与 64 KB 是两本不同的账

**ring 带宽项与 n 无关——1 GB ≈ 70 ms、64 KB ≈ 145 µs**

![ring all_reduce 的 T(S) 双对数曲线：延迟项水平渐近线与带宽项斜线在 S* = nαβ 处相交；n 变大整条曲线左端抬高、拐点右移，链路变快只压低右半段](/img/in-post/collective-communication-primitives-and-cost-model-alpha-beta-regimes.svg){: style="max-height: 360px"}

<aside class="notes" markdown="1">
原文 /collective-communication-primitives-and-cost-model.html。`busbw` = `algbw` × $$2(n-1)/n$$；8 卡 IB 拐点约 2 MB；tree 2⌈log₂n⌉ 步。「通信慢就换更快的网卡」——延迟主导的小消息对 β 不敏感。
结论原文：$$T_{ring} = 2(n-1)\alpha + \frac{2(n-1)}{n}\frac{S}{\beta}$$——带宽项与 n 无关、延迟项随 n 线性增长；8 卡 IB 上 **1 GB 约 70 ms（延迟占 0.2%）、64 KB 约 145 µs（带宽占 3%）**。
各篇关键词：01 造一把尺子 T = α + S/β；02 给 β 填真实数字 PCIe / NVLink / IB；03 给 α 填真实数字——绕过 CPU；04 在探测到的拓扑上选 ring / tree、channel、协议；05 ProcessGroupNCCL、stream、重叠；07 custom all-reduce、KV 传输；08 all-to-all、DeepEP、GPU 发起。
</aside>

<!-- v -->

### 从并行策略到理论时间：一条推算链

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 180}}}%%
flowchart LR
    W["<b>并行策略</b><br/>DP 梯度桶、TP 激活、MoE token、decode 小张量——定下 n 与 S"]
    A["<b>算法</b><br/>ring 走 2(n−1) 步、每步 S/n 字节"]
    M["<b>α-β 两项相加</b><br/>步数 × α ＋ 每 rank 字节数 / β"]
    B["<b>换算成带宽</b><br/>algbw = S / T，busbw = algbw × 原语系数"]
    W --> A --> M --> B
```

每一环都能单独核对：步数错了查算法，β 错了查链路，busbw 与硬件标称对不上才是真异常。

<aside class="notes" markdown="1">
原文第三章到第七章就是这条链的四节。原语给的是"每 rank 至少要收多少字节"的下界；两级算法把节点内、节点间分开各算一遍。
</aside>

---

## 02 · 硬件互联：节点内三个量级

**厂商标的 NVLink 带宽是双向合计，β 要用单向**

| 链路 | 单向带宽 | 备注 |
|---|---|---|
| NVLink A100 / H100 / Blackwell | 300 / **450** / 900 GB/s | 厂商 600 / 900 / 1800 是双向合计 |
| PCIe x16 3.0 / 4.0 / 5.0 | ≈ 16 / 32 / 64 GB/s | 网卡速率约为 PCIe 链路的 78% |
| IB HDR / NDR | 25 / 50 GB/s | 200 / 400 Gb/s |
| 整机 NVLink 容量 | 网卡总带宽的 **9 倍** | 跨机是瓶颈的根源 |

<aside class="notes" markdown="1">
原文 /hardware-interconnect-pcie-nvlink-and-topology.html。
结论原文：NVLink 几百 GB/s、PCIe 与网卡几十、跨 socket 个位数到几十；`nvidia-smi topo -m` 的六个等级就是 NCCL 决策的输入；**厂商标的 NVLink 带宽是双向合计，β 要用单向**。
</aside>

<!-- v -->


### 要点

- 选网卡的规则：带宽最大、其次路径最近——GPU0 配 NIC0 不配 NIC4

<!-- v -->


### 常见误区

- 「拓扑是 `PIX` 所以 P2P 一定快」——ACS / IOMMU 可能把 switch 内 P2P 重定向到 root complex；`p2pBandwidthLatencyTest` 对比、查 `ACSCtl`

---

## 03 · RDMA 与 GPUDirect：显存到对端显存的三条路

**TCP 2 次拷贝、RDMA 1 次、RDMA + GDR 0 次——上限 min(NIC, PCIe)**

![显存到对端显存的三条路径：TCP、RDMA、RDMA + GDR](/img/in-post/rdma-and-gpudirect-three-paths.svg){: style="max-height: 420px"}

<aside class="notes" markdown="1">
原文 /rdma-and-gpudirect.html。
</aside>

<!-- v -->

### 零拷贝的前提：一段内存先要被注册

![内存注册做了什么：一段 MR 在应用、内核与网卡里的三份表示](/img/in-post/rdma-and-gpudirect-mr-translation.svg){: style="max-height: 400px"}

<aside class="notes" markdown="1">
`ibv_reg_mr` 做两件事：把页锁住不让换出 / 迁移，把虚拟地址到物理地址的映射灌进网卡的 MTT，换回一对 lkey / rkey。网卡此后用 rkey 直接写对端物理页，不经对端 CPU——这就是单边操作的底座，代价是注册本身是慢路径，所以 NCCL 要缓存 MR。
</aside>

<!-- v -->

### 数字

| 量 | 数 |
|---|---|
| 上限 min(NIC, PCIe) | H100 / PCIe 5.0 / NDR = 50 GB/s；A100 / PCIe 4.0 / HDR = 25；**A100 配 NDR 被卡在 32** |
| 延迟 | IB 1–2 µs、RoCE 2–4 µs、TCP 15–50 µs |
| `NCCL_NET_GDR_LEVEL` | 默认 `PXB`（同一 switch 下才开 GDR） |
| `NCCL_IB_TIMEOUT=20` | 4.3 s × 7 次重传 ≈ 30 s 后 status=12 |

<!-- v -->


### 常见误区

- 「RDMA 不开 GDR 也差不了多少」——主机内存成了汇聚点：A100 上实际到不了 HDR 的 25 GB/s，8 卡 H100 的 staging 流量合计 800 GB/s 顶到两 socket 内存带宽
- 「`GPU Direct RDMA Disabled` 就设 `GDR_LEVEL=SYS`」——跨 root complex 的 P2P 可能只有几 GB/s；先修亲和，再用 `ib_write_bw --use_cuda` 证明路径能跑

---

## 04 · NCCL 架构：决策全在 init 里做完

**init 建调优表、调用只查表——同一 all_reduce 换台机器换算法**

| 机器 | 选择 | 为什么 |
|---|---|---|
| NVSwitch（H100 ×8） | NVLS + Simple | 交换机做归约，带宽项最优 |
| NVLink 直连（A100 ×8） | Ring + LL128 | LL128 有效载荷约 94%，只在 NVLink 路径启用 |
| PCIe 机 | Simple / LL | LL 50% 载荷，小消息延迟低 |
| 32 节点 | Tree | ring 510 步 vs tree 2 × (7 + 5) 步 |

<aside class="notes" markdown="1">
原文 /nccl-architecture-topology-channels-algorithms-and-protocols.html。T = lat × latCount + S / (1000 × bw)。
结论原文：`ncclCommInitRank` 里拓扑 → 路径 → 图搜索 → channel → **调优表**；`ncclAllReduce` 只查表；调优表就是 α-β 模型按算法 × 协议 × 拓扑算出的 lat 与 bw——所以同一次 8 卡 all_reduce 在不同机器选不同算法。
</aside>

<!-- v -->

### 初始化期做完决策，执行期只查表

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 160}}}%%
flowchart TB
    subgraph INIT["初始化期 ncclCommInitRank：与消息无关，只做一次"]
        direction LR
        B["<b>bootstrap</b><br/>uniqueId 带 rank 0 的地址，TCP 环上两次 AllGather"]
        X["<b>拓扑探测与路径计算</b><br/>PCIe 树、NVLink、网卡；每对设备一个类型与带宽"]
        G["<b>图搜索与 channel</b><br/>nChannels 条 ring / tree，拼成全局环与双二叉树"]
        T["<b>调优表</b><br/>每个算法 × 协议一对 lat 与 bw"]
        B --> X --> G --> T
    end
    subgraph RUN["执行期 ncclAllReduce：每次调用"]
        direction LR
        E["<b>enqueue</b><br/>查表取 T 最小的组合，按消息大小缩 channel 数"]
        K["<b>一个 kernel，nChannels 个 block</b><br/>节点内直写对端显存，跨机推 tail 让 proxy 上网卡"]
        E --> K
    end
    INIT --> RUN
```

<aside class="notes" markdown="1">
所以 `NCCL_DEBUG=INFO` 里那几行 TUNING / channel 日志是 init 时打的，执行期一行都不打——排障要看的决策全在启动那几百毫秒里。
</aside>

<!-- v -->


### 要点

- 强行 Ring：NVSwitch 1 GB 慢 1.9 倍（带宽的账）、32 节点中小消息慢 2–9 倍（延迟的账）

<!-- v -->


### 常见误区

- 「设 `NCCL_ALGO` / `NCCL_PROTO` 是常规调优」——常是三年前留下的 env；先看 TUNING 日志与 nccl-tests 扫描，再写 tuner 配置

---

## 05 · PyTorch 的通信栈：三个「不一定」

**`wait()` 不是等通信完成，是当前 stream 等 end event；CPU 全程不停**

![async_op=True 时两条 stream、两个 event 与一个 Work](/img/in-post/pytorch-communication-stack-processgroupnccl-and-streams-async-streams-events.svg){: style="max-height: 420px"}

<aside class="notes" markdown="1">
原文 /pytorch-communication-stack-processgroupnccl-and-streams.html。重叠的来源与失效的来源是同一套编排。
</aside>

<!-- v -->

### 重叠成立，和它两种失效的样子

![重叠成立与两种失效的 GPU 时间线](/img/in-post/pytorch-communication-stack-processgroupnccl-and-streams-overlap-timelines.svg){: style="max-height: 420px"}

<aside class="notes" markdown="1">
三段的 kernel 长度完全一样，变的只是怎么排。第一段没人等；第二段是**当前 stream 等 NCCL stream**（`wait()` 放早了，CPU 侧看不出任何异常）；第三段是**CPU 等 GPU**（`.item()` 的同步拷贝，或显存碎片触发的 `cudaFree`），空洞更长，因为新 kernel 也发不出去。
</aside>

<!-- v -->

### 数字

| 量 | 数 |
|---|---|
| watchdog | 每 100 ms 轮询；`opTimeout_` 默认 10 分钟；heartbeat monitor 每 480 s 读一次心跳，watchdog 卡死后 8–16 分钟判定、再等最多 8 分钟强杀 |
| DDP bucket 25 MiB | 节点内带宽项约 97 µs、跨机约 875 µs |
| 1000 个 25 KB 各做一次 vs 合并一次 | 延迟项差 1000 倍——bucket 存在的理由 |

<!-- v -->


### 常见误区

- 「`work.wait()` 阻塞 CPU 到通信完成」——CPU 只在 `TORCH_NCCL_BLOCKING_WAIT`、barrier、显式 synchronize 时阻塞
- 「trace 里 NCCL kernel 很长 = 带宽不够」——本 rank 先到、在自旋等别人；**kernel 最短的是最晚到的 straggler**

---

## 06 · nccl-tests、调优与排障：曲线怎么读，hang 怎么找

**左端 α 右端 β、拐点 $$n\alpha\beta$$；Flight Recorder 按 seq_id 指 culprit**

![ring 模型下的 busbw 曲线：延迟主导区、拐点区、带宽主导区，以及 α 翻倍、β 减半、中段凹陷三种偏差的形状](/img/in-post/nccl-tests-tuning-and-debugging-hangs-busbw-curve.svg){: style="max-height: 360px"}

<aside class="notes" markdown="1">
原文 /nccl-tests-tuning-and-debugging-hangs.html。8×H100 节点内拐点约 10 MB、64 卡跨机约 30 MB；参考线 8×H100 350–480 GB/s、2 节点 NDR 每 GPU 40–48。
结论原文：曲线**左端看 α、右端看 β**、拐点 $$S_{knee} = n\alpha\beta$$，到 90% 平台约 9 倍拐点；hang 分六类，前四类各 rank 最后一次操作不一致、后两类一致；**Flight Recorder 按 `collective_seq_id` 对齐给出 culprit**。
</aside>

<!-- v -->

### 参考线与参数优先级

| 参考线 | 值 |
|---|---|
| 8×H100 节点内 all_reduce busbw | 350–480 GB/s |
| 2 节点 NDR，每 GPU | 40–48 GB/s |
| 拐点与平台 | 8×H100 节点内约 10 MB、64 卡跨机约 30 MB；到 90% 平台约 9 倍拐点 |
| 参数优先级 | env > `NCCL_CONF_FILE` > `~/.nccl.conf` > `/etc/nccl.conf` > 代码默认值 |

<!-- v -->

### hang 的六类

| hang 类型 | 各 rank 最后一次操作 |
|---|---|
| A 形状或 dtype 不一致<br/>B 调用顺序 / 次数不一致<br/>C send / recv 不配对<br/>D 两个 communicator 交叉等待 | **不一致**——Flight Recorder 对齐 `collective_seq_id` 找出谁 |
| E 某 rank 崩了或卡在非通信代码<br/>F 网络或硬件真的断了 | 一致——看进程数、dmesg 与 py-spy |

<!-- v -->

### 判别树：先数进程，再比 Flight Recorder

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 170}}}%%
flowchart LR
    S["<b>watchdog 报 timeout</b><br/>先收齐进程数、py-spy、FR dump、NCCL WARN、dmesg"]
    S --> Q1{"进程数够吗"}
    Q1 -->|"少了"| E1["<b>类 E：崩溃</b>"]
    Q1 -->|"都在"| Q2{"各 rank 的 FR 记录一致吗"}
    Q2 -->|"同序号、同为 started"| F1["<b>类 F：网络或硬件</b>"]
    Q2 -->|"报 MISMATCH"| AB["<b>类 A / B：形状或次数不一致</b>"]
    Q2 -->|"少一条、末条 completed"| E2["<b>类 E：卡在通信之外</b>"]
    Q2 -->|"p2p 找不到配对"| C2["<b>类 C：send / recv 不配对</b>"]
    Q2 -->|"停在不同 PG"| D2["<b>类 D：多 communicator 交叉</b>"]
```

<aside class="notes" markdown="1">
第一层就是"所有 rank 都在等同一个不存在的东西，还是有一个 rank 在等别的东西"：前者查基础设施，后者查代码。</aside>

<!-- v -->


### 常见误区

- 「所有 rank 停在 all_reduce 是 NCCL 的 bug」——kernel 自旋无超时，唯一计时器是 c10d 的 watchdog；任何 rank 掉队都表现为全体 timeout

---

## 07 · 推理侧的通信：decode TP 是纯 $$\alpha$$ 的账

**custom all-reduce 30 → 10 µs：14 步 → 2 步 + CUDA Graph**

| 量 | 数 |
|---|---|
| 128 KB | 8 × 8192 × 2 B（TP8、d = 8192、BF16） |
| NCCL Ring + LL 模型 | 6.6 + 14 × 0.6 ≈ 15 µs |
| custom AR | 36 个 block；8 卡 < 256 KB one-shot；上限 8 MB；只能节点内 |
| Llama-3-70B KV | 每 token 320 KB；4096 token 1.25 GiB；TP8 每 rank 160 MiB，400 Gb/s 约 3.4 ms |

<aside class="notes" markdown="1">
原文 /inference-communication-custom-all-reduce-and-kv-transfer.html。
结论原文：8 卡 TP decode 每层 128 KB 的 all_reduce，NCCL 30 µs、custom all-reduce 10 µs——省在 launch 路径、**14 步 → 2 步**、无 channel buffer 中转、可捕获进 CUDA Graph；**KV 传输是纯 β 的点对点账**，单边 RDMA 比 NCCL 更自然。
</aside>

<!-- v -->

### 一次 all_reduce 要问六个后端：谁接

`tensor_model_parallel_all_reduce(x)` 进 `CudaCommunicator`，六个后端按固定顺序各自回答「这一次我接不接」，第一个接的就执行：

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 110}}}%%
flowchart LR
    Q1{"NCCL<br/>对称内存"}
    Q2{"quick AR<br/>与 FlashInfer"}
    Q3{"custom<br/>all-reduce"}
    Q4{"torch<br/>对称内存"}
    B["<b>PyNccl</b><br/>兜底"]
    Q1 -->|"不接"| Q2 -->|"不接"| Q3 -->|"不接"| Q4 -->|"不接"| B
```

8 卡 H100 默认：≤ 256 KB 走 custom AR 的 one-shot，256 KB～64 MB 走对称内存的 NVSwitch 多播归约，更大交给 PyNccl。

<aside class="notes" markdown="1">
六个后端按固定顺序各自回答"这一次我接不接"，`CUSTOM_ALL_REDUCE_MAX_SIZES` 这张表（H100 8 卡 256 KB、6 卡 512 KB、4 卡 32 MB、2 卡 64 MB）决定交接点；`NCCL_SYMM_MEM_ALL_REDUCE_CONFIG` 定义 custom AR 的优势区间：8 卡 (16 KB, 128 KB)、4 卡 (16 KB, 512 KB)。
</aside>

<!-- v -->


### 要点

- 不能用在梯度同步上：一致性模型只保证 one-shot 内，且 25 MB bucket 在带宽账上

<!-- v -->


### 常见误区

- 「custom all-reduce 在所有小消息上都赢」——36 个 block 的启动与两次 flag 交换也是固定成本；8 卡优势区间 (16 KB, 128 KB)，更小的消息 NCCL 对称内存更快

---

## 08 · MoE 的通信：prefill 是网卡带宽的账，decode 是发起速率的账

**decode = 1024 条 7.4 KB 消息——IBGDA 让 warp 自己发**

| 量 | 数 |
|---|---|
| 每 token | FP8 dispatch 59 KB、BF16 combine 115 KB |
| 跨节点比例 | 1 − 1/N |
| 去重份数 $$N(1-(1-1/N)^k)(1-1/N)$$ | 7 → 4.6 → 3.2 |
| prefill 一层 | 5.6–12.5 ms |
| decode 一层 | 理论 429 µs、DeepEP README 487 µs；3/4 是字节、40–60 µs 是 α |

<aside class="notes" markdown="1">
原文 /moe-communication-all-to-all-deepep-and-gpu-initiated.html。
结论原文：通信矩阵由路由决定、每步不同、最慢的 rank 决定时间；prefill **按节点去重**把网卡上的份数从 7 压到 3.2；decode 是 1024 条 7.4 KB 的消息，**CPU proxy 给不了发起速率，IBGDA 让 warp 自己写 WQE 与 doorbell**。
</aside>

<!-- v -->

### 一个 token 的来回：网卡上一个节点只走一份

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 150}}}%%
flowchart LR
    A["<b>router + 写发送 buffer</b><br/>每 token 选 k 个专家<br/>FP8 转换、按目标分段"]
    B1["<b>同节点</b><br/>远程写对端<br/>channel 队列"]
    B2["<b>跨节点</b><br/>一次 RDMA WRITE<br/>最窄的一段"]
    F["<b>对端同号 GPU</b><br/>再经 NVLink<br/>转发给目标卡"]
    E["<b>recv_x → GEMM</b><br/>最慢的 rank<br/>决定整层"]
    C["<b>combine 回来</b><br/>转置的矩阵<br/>BF16"]
    A --> B1 --> E
    A --> B2 --> F --> E
    E --> C
```

<aside class="notes" markdown="1">
「网卡上一个节点只走一份」就是 7 → 3.2 那笔账的机制：先 RDMA 到对端节点的同号 GPU，再由它 NVLink 分发，剩下两跳落在快 9 倍的链路上。

细节：发起网卡那一跳的是 CPU proxy（NCCL / normal kernel）或 warp 自己（IBGDA / low-latency kernel）；跨节点只传一份、由对端同号 GPU 按 SourceMeta 转发，是「网卡上一个节点只走一份」的来源；各 rank 收到的 token 数不同，combine 复用 dispatch 的 handle。
</aside>

<!-- v -->


### 要点

- NCCL 的 all_to_all 在 decode 不够用：CPU proxy 每条消息一次响应

<!-- v -->


### 常见误区

- 「MoE 的 decode 通信是纯延迟问题」——EP=64 时 6.6 MB 过 50 GB/s 网卡本身就要 132 µs；瓶颈是发起速率

---

## 两本账在八篇里

| 篇 | 带宽的账 β | 延迟的账 α |
|---|---|---|
| 01 | ring 带宽项与 n 无关 | 延迟项随 n 线性；拐点 nαβ |
| 02 | 单向 450 GB/s、PCIe 78%、整机 9 倍 | 路径等级 PIX / PXB / SYS |
| 03 | 上限 min(NIC, PCIe) | IB 1–2 µs、TCP 15–50 µs；0 拷贝 |
| 04 | LL 50% / LL128 94% / Simple 100% 载荷 | ring 510 步 vs tree 24 步 |
| 05 | bucket 25 MiB | 1000 次 vs 1 次差 1000 倍 |
| 06 | 曲线右端、参考线 | 曲线左端、六类 hang |
| 07 | KV 传输 3.4 ms | custom AR 14 → 2 步 |
| 08 | prefill 去重 7 → 3.2 | decode 发起速率、IBGDA |

---

## 常见误区

- 「通信慢就换更快的网卡」——先判断消息在拐点哪一侧
- 「algbw 能和硬件带宽比」——用 busbw
- 「H100 的 NVLink 是 900 GB/s」——单向 450
- 「拓扑是 PIX 所以 P2P 一定快」——ACS / IOMMU 可能重定向
- 「RDMA 不开 GDR 也差不了多少」——主机内存带宽顶住
- 「Ring 是 all_reduce 的最优算法」——只在带宽账上
- 「`work.wait()` 阻塞 CPU」——只是 stream 等 event
- 「NCCL kernel 很长 = 带宽不够」——是在等 straggler
- 「所有 rank 停在 all_reduce 是 NCCL 的 bug」——找 culprit
- 「MoE decode 通信是纯延迟问题」——3/4 是字节，瓶颈是发起速率
{: .fragments}

---

## 八个出口

| 篇 | 一个公式 / 一个数 |
|---|---|
| 01 | $$T_{ring} = 2(n-1)\alpha + \frac{2(n-1)}{n}\frac{S}{\beta}$$；$$S^* = n\alpha\beta$$ |
| 02 | NVLink 单向 300 / 450 / 900；PCIe 5.0 x16 64 GB/s；NDR 50 |
| 03 | 拷贝 2 / 1 / 0 次；IB 1–2 µs；上限 min(NIC, PCIe) |
| 04 | 决策在 init；LL 50% / LL128 94%；tree 2⌈log₂n⌉ |
| 05 | NCCL stream 等 event、wait() 等 end event；watchdog 10 分钟 |
| 06 | 左 α 右 β；90% 平台 ≈ 9 S_knee；Flight Recorder |
| 07 | 128 KB：30 → 10 µs；14 → 2 步；KV 320 KB / token |
| 08 | 去重 7 → 3.2；decode 487 µs，3/4 是字节 |

---

## 下一步

- **往上**：《大规模训练工程》——这些原语怎么组成 DP / TP / PP / EP；《大模型推理系统揭秘》——TP decode 与 PD 分离里的 custom AR 与 KV 传输
- **往下**：《GPU Kernel 工程》——通信 kernel 也是 kernel
- **实践**：comm-probe 在任何新机器上跑一遍
- 原文总纲：`/communication-and-interconnect-for-ai-infra.html`；通关自测在系列总结
