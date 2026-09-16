---
layout: post
title: 扩散模型推理基础设施：图像与视频生成的 serving（总纲）
subtitle: "Diffusion Model Inference Infrastructure: Serving Image and Video Generation"
tags: [Diffusion, DiT, Video Generation, Inference, SGLang, vLLM, xDiT, AI, AI-Infra]
catalog: true
---


## 内容简介

《扩散模型推理基础设施：从一次去噪到一个生成服务》是一组共九篇的系列文章，面向已经理解 LLM 推理引擎（vLLM 一类：请求、调度、KV cache、批处理、多卡）的工程师，系统讲解**图像与视频生成模型**——Stable Diffusion 3、FLUX、Qwen-Image、Wan、HunyuanVideo 一类——的推理是怎样一种负载，以及围绕它建立起来的一套与 LLM serving 几乎不重叠的基础设施：一次生成的三段（文本编码、几十步去噪、VAE 解码）各花多少算力、显存与时间；单卡上 attention 后端、编译、量化、offload 各能换回多少；相邻去噪步之间的冗余怎样被缓存与跳步利用；视频的十万级 token 序列让 attention 占到算力的大半之后，稀疏化怎样做；多卡为什么用序列并行、CFG 并行与 patch 级流水线而不是张量并行；步数蒸馏与自回归视频把"几十步"变成"四步"或"一段流"之后，前面的优化哪些失效、哪些回归（KV cache 又出现了）；一个生成服务的请求形态、批处理、三段分离、LoRA 与异步任务 API 该怎样设计；以及 SGLang Diffusion、vLLM-Omni、xDiT 三个引擎各把这些机制放在哪里。

它回答的问题是：

> **一个 12B 的图像模型生成一张 1024² 的图要 2 PFLOPs，是一个 7B LLM 回答一千 token 的 150 倍，时间却差不多；一个 14B 的视频模型生成 5 秒 720p 要 650 PFLOPs，单卡二十几分钟。这种负载的推理系统该长什么样？为什么 vLLM 的那一套——KV cache、连续批处理、PD 分离——在它身上大半用不上？**

答案的起点是一个结构性的差别：LLM 的生成是**一次前向产出一个 token、串行几百次、每次读一遍全部权重**——memory-bound，所有的系统设计（KV cache、批处理、投机解码、PD 分离）都在对付"每步读权重"这件事。扩散模型的生成是**一次前向处理整张 latent 的几千到十几万个 token、重复几十步、每步同样读一遍权重**——单请求就是 compute-bound，算术强度是 LLM decode 的几千倍，batch 对吞吐几乎无益，没有跨步的缓存（除了文本编码器的输出），请求的时长在收到它的那一刻就完全确定。这一个差别决定了两套系统在每个层面上的分歧：它的瓶颈在算力而不是带宽，所以优化的是 FLOPs（少算：跨步缓存、稀疏 attention、步数蒸馏）与 MFU（算得快：编译、量化、FA3），而不是字节；它的多卡是为了把一个请求的延迟切短而不是把权重装下，所以用序列并行而不是张量并行；它的调度面对的是可预测的批任务而不是不可预测的流，所以是队列与 SLO 而不是连续批处理。

系列的组织原则来自这本账：**先算清一次生成的三本账（FLOPs、字节、秒），再看每一类优化在账上的哪一项做交换**。单卡优化换 MFU；跨步缓存与稀疏 attention 换 FLOPs 的系数；多卡并行用通信换墙钟；步数蒸馏直接改步数这个乘数；serving 层在这些之上决定请求怎样排队、哪一段放在哪张卡。九篇的每一篇都把新引入的机制记回第一篇的账。

系列**不设单一的源码阅读对象**。与《大模型推理系统揭秘》以 vLLM 为主线、《RL 后训练基础设施》以 verl 为主线不同，扩散推理的三个主要引擎——**SGLang Diffusion**（SGLang 在 2025 年 11 月并入的图像 / 视频生成框架，继承 SGLang 的 scheduler 与 kernel 栈）、**vLLM-Omni**（vLLM 面向全模态与非自回归模型的扩展，以 stage 分离的流水线为特点）、**xDiT**（并行方法的源头：USP、PipeFusion、CFG 并行、Parallel VAE 都从这里出来）——各自代表一种取向，没有哪一个像 vLLM 之于 LLM serving 那样成为事实标准；而 **diffusers** 是三者共同的底座（模型定义、调度器、缓存 hook 的接口）。所以正文以机制为主，每篇末尾用一张表对照同一个机制在三个引擎里的位置；第八篇专门用一个请求的路径把三个引擎完整走一遍，看它们在哪一段做了不同的选择、为什么。


## 为什么写这个系列？

### 它是一类真实存在、正在变大的 AI-Infra 负载

2024 年以前，图像生成的推理是"一张 A10 跑 SD 1.5、几秒一张"的规模，服务形态是 Web UI 加一个队列，不需要基础设施。2024 年之后两件事同时发生：模型从 1B 的 U-Net 变成 12B–20B 的 DiT（FLUX、Qwen-Image），一张 1024² 的图的算力涨了两个量级；视频生成从实验变成产品（Sora、Kling、Veo、Wan、HunyuanVideo），单请求的算力比图像再高两个量级、单卡几十分钟。到 2025 年底，SGLang 与 vLLM 两个 LLM serving 的主流项目都把扩散模型纳入了自己的框架，这是最清楚的信号：图像与视频生成已经是与 LLM 并列的一类 serving 负载，需要同样等级的基础设施。

### 08 系列建立的方法在它身上大半用不上

这是把它单列一个系列而不是附在 08 后面的理由。《大模型推理系统揭秘》十四篇建立的分析方法——请求的 prefill / decode 两阶段、KV cache 的字节数与分页、连续批处理、投机解码、PD 分离——每一项的前提都是"decode 是 memory-bound 的串行过程"。扩散模型没有 decode：每一步是一次对整个序列的完整前向，几千个 token 一起过一遍权重，MFU 在 batch 1 就能到 50% 以上。于是：

```text
KV cache        没有：每步的输入是整张 latent，没有跨步可复用的中间状态（文本编码器输出除外）
prefill/decode  没有：每步的形态相同，都是"prefill"
连续批处理      收益小：compute-bound 下 batch 2 几乎就是 2 倍时间；视频 batch = 1
投机解码        没有：没有串行的 token 链可以猜
PD 分离         不适用；但有另一种分离——文本编码器 / DiT / VAE 三段各自的资源需求不同
张量并行        通信 ∝ token 数 × 层数，序列长时不划算；且权重单卡放得下——多卡是为了延迟
请求时长        完全可预测：分辨率 × 步数 × CFG 在收到请求时就决定了秒数
```

同时它有一套 LLM serving 没有的东西：相邻步之间的**时间冗余**（TeaCache 一族）、几十步的**步数**本身作为可蒸馏的乘数、把 latent 切成 patch 的**空间并行**、以及视频里让 attention 占到七成算力的**十万级序列**。这些需要另一套分析方法，本系列就是这套方法。

### 为什么进主线

地图最初把它列为选修，理由是"不做生成的 Infra 工程师可能永远碰不到它"。这个理由到 2026 年不再成立：SGLang 与 vLLM 两个 LLM serving 主项目都把扩散 / 全模态并入了自己的框架，图像与视频生成已经是与 LLM 并列的一类 serving 负载；而只读 08 的人会以为 KV cache、连续批处理、张量并行就是"推理系统"——本系列是推理主线的另一半，与 09 当年从选修升入主线是同一类判断。它也是当前少有的一类"系统尚未定型"的 Infra 方向：三个引擎都在快速迭代，机制层却已经稳定（并行方法、缓存方法、量化方法在 2024–2025 年基本收敛），正是适合写机制、少写实现的时候。

### 现有材料的断层

- **论文**（xDiT / PipeFusion、DistriFusion、TeaCache、SVDQuant、Sparse VideoGen、Radial Attention、CausVid、Self-Forcing）各讲一个机制与它的加速比，不讲这些机制怎样叠加、在一张账上各占多少、什么时候互斥；
- **框架文档**告诉你每个 flag 是什么（`--ulysses-degree`、`--enable-cfg-parallel`、`--enable-teacache`），不告诉你给定模型、分辨率与 GPU 该选哪个、为什么；
- **算法侧的材料**（本站算法地图 L7 多模态系列第五至七篇）讲扩散的数学、模型结构与训练配方，成本只算到"一张图多少 FLOPs"；
- **一线经验**散在三个引擎的 benchmark 页、issue 与博客里，用的模型、分辨率、GPU 各不相同，无法直接比较。

本系列想填补的是从"能用 diffusers 跑出一张图"到"能为一个生成服务选引擎、定并行度、开哪些加速、解释它的每张图为什么花这么多秒、这么多钱"之间的那段路。


## 适合哪些读者？

### 负责图像 / 视频生成服务的系统工程师

你的团队用 FLUX、Qwen-Image 或 Wan 对外提供生成服务，能跑，但一张 1024² 的图要十几秒、一段 5 秒视频要十几分钟、GPU 账单比同等流量的 LLM 服务高一个量级，而你不知道哪些优化是无损的、哪些会改变图片、叠加起来能到多少。本系列是为这个阶段准备的。

### 读完 08 系列、想知道"另一半 serving"长什么样的 Infra 工程师

你理解 vLLM 的每个机制，想知道当负载从 memory-bound 变成 compute-bound 时，这些机制哪些还成立、哪些要换成什么。本系列第一、五、七篇是最直接的对照。

### 做生成模型、想知道系统在限制什么的算法工程师

你在选模型结构与蒸馏方案，需要知道每个选择对推理成本的影响：text token 进不进联合 attention、latent 通道数与 patch 大小、要不要 CFG、蒸馏到几步、视频的帧数与分辨率——每一项在系统上各花多少。第一、三、四、六篇是这些问题的系统侧答案；模型本身（扩散的数学、DiT 与 MMDiT、文生图配方、步数蒸馏的方法）在算法地图的[《多模态：从视觉编码器到扩散模型》](/multimodal-from-vision-encoders-to-diffusion.html)第五至七篇。

### 想读懂 SGLang Diffusion / vLLM-Omni / xDiT 源码的开发者

第八篇按一个请求的路径把三个引擎各走一遍，每个机制给出文件与类名；读完后你应该能在任何一个里定位并行组的建立、attention 后端的选择、缓存 hook、调度器与 API 层，并知道三者的分歧点在哪。


## 系列的整体主线

九篇按"先算账、再优化单卡、再利用两种冗余、再多卡、再改步数、最后是服务与引擎"的顺序推进：

```text
第一篇：负载画像 —— 一次生成的三段：FLOPs、显存、时间账
        ↓  token 数公式 · 线性项与 attention 项 · 为什么单请求就 compute-bound · 与 LLM 的对照 · 四个放大器
第二篇：单卡执行 —— 换 MFU：attention 后端、编译、量化、offload
        ↓  FA3 / SageAttention · torch.compile 与 CUDA graph · FP8 / SVDQuant INT4 · 三段的 offload · VAE 分块
第三篇：跨步冗余 —— 换 FLOPs 的系数：缓存与跳步
        ↓  相邻步为什么相似 · TeaCache / FBCache / Cache-DiT 一族 · 命中率 → 加速比 · 与 CFG / SP / 少步的交互
第四篇：视频 —— attention 占七成之后：长序列的账与稀疏化
        ↓  3D VAE 与时空 patch · 十万 token 的 4·L·N²·d · Sparse VideoGen / Radial / STA · Amdahl
第五篇：多卡并行 —— 用通信换墙钟：为什么不是张量并行
        ↓  Ulysses / Ring / USP · CFG 并行 · PipeFusion 与 DistriFusion · Parallel VAE · NVLink vs PCIe
第六篇：少步与自回归 —— 改步数这个乘数：蒸馏之后与流式之后
        ↓  4 步 / 1 步的账 · 哪些优化失效 · StreamDiffusion · CausVid / Self-Forcing 与 KV cache 的回归
第七篇：serving 形态 —— 请求、批、三段分离、附件、异步任务与成本
        ↓  batch 为什么几乎不提吞吐 · 时长可预测的调度 · 文本编码器 / DiT / VAE 分离 · LoRA / ControlNet · /v1/videos
第八篇：三个引擎的对照导读 —— 同一张图的请求在 SGLang Diffusion、vLLM-Omni、xDiT 里各走过什么
        ↓  进程模型 · pipeline 抽象 · 并行组 · attention 后端 · 缓存 hook · 调度器 · API 层 · 分歧点
第九篇：配置、评测与排障 —— 从一张卡的推导到一条伪影的排查
           配置推导顺序 · 性能与质量的评测方法 · 确定性 · 常见故障
```

三条交织的线索：

```text
账本线：FLOPs · 字节 · 秒 —— 每篇都把新引入的机制记进第一篇的那张账
形态线：图像 · 视频 · 实时 —— 同一个机制在 4K token、100K token、4 步流式三种形态下各是什么样
框架线：SGLang Diffusion · vLLM-Omni · xDiT（diffusers 底座）—— 每篇末尾一张对照表，第八篇完整走一遍
```


## 章节结构与分章导读

### 1. 负载画像：一次生成在 GPU 上发生什么

第一篇建立全系列的记账框架。它不讨论任何优化，只回答一个问题：**一张图 / 一段视频从 prompt 到像素，三段各做多少 FLOP、占多少显存、在什么 MFU 下花多少秒？**

这一篇会讨论：

- 三段流水线：文本编码器（T5-XXL / CLIP / LLM，一次前向、几百 token、几十毫秒）→ 去噪网络（DiT，对整张 latent 一次完整前向 × 步数 × CFG 分支）→ VAE 解码（卷积，一次、算力小但显存峰值大）；
- token 数公式：$$N = \frac{H}{f\,p} \cdot \frac{W}{f\,p} \cdot \frac{F}{f_t\,p_t}$$，FLUX 的 1024² 是 4096 个图像 token 加 512 个进联合 attention 的文本 token；Wan 的 720p 81 帧是 75,600 个；
- 每步 FLOPs = 线性项 $$2 P_\text{tok} N$$ + attention 项 $$4 L N^2 d$$（$$P_\text{tok}$$ 是一个 token 真正经过的参数量，FLUX 的 6.45B 而不是 11.9B）：FLUX 上 attention 占 20%，Wan 上占 72%——**图像模型是 GEMM 负载，视频模型是 attention 负载**；
- 显存：权重（FLUX 12B bf16 22 GiB + T5 9 GiB）+ 激活（FlashAttention 下随 $$N$$ 线性，几百 MiB 到十几 GiB）+ VAE 解码的峰值（720p 81 帧不分块要上百 GiB）——**没有 KV cache**；
- roofline：一次 DiT 前向的算术强度是几千到几十万 FLOP/字节，H100 的拐点是 295——单请求就在算力屋顶上，与 LLM decode 的每 token 读一遍权重（强度 ≈ 2）相反；
- 时间模型：每步 = FLOPs / (峰值 × MFU)，FLUX 28 步在 H100 上 eager 6.7 s、编译后 4.3 s（xDiT 实测，MFU 0.31 / 0.49），Wan 14B 50 步单卡二十几分钟；
- 与 LLM 的对照表：FLOPs 高 2–5 个量级、时间相近或更长、瓶颈在算力、batch 无益；
- 四个放大器：分辨率（$$N \propto$$ 像素，attention $$\propto N^2$$）、帧数、步数、CFG。

核心问题是：

> **FLUX.1-dev 生成一张 1024² 的图、28 步，在一张 H100 上：每步多少 FLOPs、attention 占几成、MFU 多少时是几秒、三段各占多少？换成 Wan2.1-14B 的 5 秒 720p，哪一项变了几个量级？**

实践：`diffusion_ledger.py`——输入模型规格、分辨率 / 帧数、步数、CFG、硬件与 MFU，输出三段的 FLOPs、显存、时间与 LLM 对照。它是全系列唯一的配套脚本，后面每一篇在它的账上加一项交换。

### 2. 单卡执行：attention 后端、编译、FP8 / INT4 与 offload

第二篇讨论不改变模型输出（或改变得可控）的单卡优化：**同一次前向，怎样让 MFU 从 40% 到 75%、显存从 32 GiB 到 12 GiB。**

这一篇会覆盖：

- 三段的显存不必同时在卡上：文本编码器算完就可以让出显存（CPU offload / 顺序加载），VAE 只在最后用；DiT 的 layerwise offload（只有两层在卡上、H2D 与计算重叠——视频模型上几乎免费，图像模型上会慢）；一张 24 GB 的 4090 跑 12B FLUX 的路径；
- attention 后端：FlashAttention-2 / 3、SDPA、xformers；**SageAttention** 把 Q·K 量化到 INT8 / FP8——为什么扩散模型能容忍 8-bit attention（误差被后续步的去噪吸收）而 LLM 的 decode 不行；
- `torch.compile` 与 CUDA graph：扩散是最适合编译的负载——形状固定、步数固定、每步相同；xDiT 实测 FLUX 单卡 6.71 → 4.30 s；代价是编译时间、动态分辨率下的重编译，以及 SGLang 的 breakable CUDA graph（固定分辨率的 DiT 段捕获、attention 与集合通信留在 eager）；
- 量化：Hopper 上 FP8 W8A8 的线性层 1.3–1.6×（ModelOpt / 在线 FP8）；**SVDQuant** 的 W4A4——低秩分支吸收权重与激活的离群值、Nunchaku 把低秩 kernel 融进 4-bit kernel、FLUX 显存 3.6× 缩减、4090 上 3× 加速；Blackwell 的 NVFP4；图像对 W4 比 LLM 更敏感，评测用 PSNR / LPIPS 而不是困惑度；
- VAE：解码器在全分辨率上的 fp32 特征图是显存峰值的来源，tiled / sliced 解码、视频 3D VAE 的时间分块与接缝；
- 融合：adaLN 的 scale / shift / gate、QK-norm + RoPE、GELU epilogue——SGLang 与 vLLM-Omni 各自的 fast path；
- 叠加顺序与收益表：bf16 eager → FA3 → compile → FP8 → INT4，每一步在 FLUX 1024² 上换回多少。

核心问题是：

> **一张 24 GB 的 RTX 4090 上，12B 的 FLUX.1-dev 放不放得下？放下之后 28 步几秒？把 FA3、compile、FP8、SVDQuant 依次加上，每步降到多少，哪一步开始图片可见地变了？**

实践建议：用 diffusers 在一张卡上对同一个 prompt 与 seed 依次打开每项优化，记录每步耗时、峰值显存与对基线图的 PSNR / LPIPS，画出"加速比—质量"曲线。

### 3. 跨步冗余：TeaCache、First-Block Cache 一族的缓存与跳步

第三篇进入扩散特有的第一类冗余：**相邻两个去噪步的网络输出很相似，能不能有的步不算？**

这一篇会覆盖：

- 为什么相似：采样是沿一条平滑的概率流 ODE 轨迹走，相邻步的输入 $$x_t$$ 与输出差别小；差别随 $$t$$ 不均匀——开头几步（决定构图）与结尾几步（决定细节）变化大、中间平缓；
- 谱系：DeepCache（U-Net 时代，复用跳连的高层特征）→ FORA / Δ-DiT（均匀间隔跳步）→ **TeaCache**（用时间步 embedding 调制后的输入的相对 L1 差预测输出差，多项式重标、累积、过阈值才全算）→ **First-Block Cache**（算第一个 block、看它的残差变了多少，没变就复用整个上一步的输出）→ **Cache-DiT** 的 DBCache（前 $$F_n$$ 个 block 全算、后面的按残差差决定）与 TaylorSeer（用泰勒展开外推而不是复用）→ MagCache（幅度比）→ AdaCache（逐层）；
- 命中率 → 加速比：$$\text{speedup} = T / (T_\text{full} + T_\text{hit} \cdot \epsilon)$$，FLUX 28 步阈值 0.4 命中约一半、端到端 1.5–2×；视频上到 4.4×（TeaCache 在 Open-Sora-Plan）；
- 质量代价与评测：VBench 掉 0.07%、对原图的 PSNR 30 dB 上下；伪影的形态（闪烁、细节丢失）与阈值扫描曲线；
- 交互：与 CFG（条件 / 无条件分支各自的缓存状态）、与序列并行（各卡的跳步决策必须一致——用同一个标量）、与 layerwise offload（跳过的层不用搬）、与**少步蒸馏互斥**（4 步之间没有冗余可利用）；首末步总是全算；
- 实现：全部是 hook——diffusers 的 `CacheMixin` / `HookRegistry`，vLLM-Omni 的 `CachedTransformer` 与 TeaCache / MagCache / Cache-DiT 后端，SGLang 的 `runtime/cache/` 与 Cache-DiT 集成，xDiT 的 `cache_manager`。

核心问题是：

> **FLUX 28 步，TeaCache 阈值取 0.4：命中多少步、端到端加速多少、对原图的 PSNR 掉到多少？同样的方法为什么在 FLUX.1-schnell 的 4 步上一步都省不下来？**

实践建议：固定 prompt / seed / 分辨率，扫阈值 0.1–0.8，记录命中步的位置分布、总时间与 PSNR / LPIPS，观察命中集中在哪一段步、伪影从哪个阈值开始可见。

### 4. 视频：长序列 attention 的账与稀疏化

第四篇讨论把"图像"换成"视频"后账的质变：**当 $$N$$ 从 4K 到 100K，$$4 L N^2 d$$ 超过 $$2 P N$$，负载从 GEMM 变成 attention，怎么办？**

这一篇会覆盖：

- 视频 token 账：3D VAE 时间 4× 空间 8×、patch 1×2×2，Wan 720p 81 帧 75,600 token、HunyuanVideo 129 帧 119K；attention 占比 38%（17 帧）→ 72%（81 帧）→ 80%（129 帧）；单卡 50 步二十几分钟；
- 全 3D attention vs 时空分解：为什么 2024 年后主流回到全 attention（时间一致性），代价就是这个 $$N^2$$；
- 稀疏化：**Sparse VideoGen**（在线 profiling 把 head 分成 spatial / temporal 两类、各用一种稀疏掩码，2.3×）→ SVG2（语义置换 + 动态 kernel）→ **Radial Attention**（静态 $$O(n \log n)$$ 掩码：注意力随时空距离能量衰减，窗口随时间距离减半；1.9×，配 LoRA 微调可把长度扩到 4×）→ **Sliding Tile Attention**（FastVideo，按 tile 滑窗）；与 block-sparse FlashAttention kernel 的关系——稀疏只有落到 kernel 的 block 粒度上才换回时间；
- 8-bit attention（SageAttention）在视频上的收益比图像大——因为 attention 占比大；
- 训练无关（SVG、STA）与需微调（Radial 的 LoRA）的分界；长度外推；
- 显存：激活随 $$N$$ 线性到十几 GiB、3D VAE 解码的峰值上百 GiB 必须分块；
- 叠加表：Wan 14B 720p 81 帧 50 步，在"稀疏 attention + 跨步缓存 + 多卡"各自与叠加下的时间（Amdahl：attention 稀疏 80% 时端到端最多 2.2×）。

核心问题是：

> **Wan2.1-14B 生成 5 秒 720p，一步的 attention 是多少 PFLOPs、占几成？把 attention 稀疏掉 80%，端到端加速多少？帧数加到 4 倍，账上哪一项变了 16 倍？**

实践建议：用 `diffusion_ledger.py --model wan --sweep` 看帧数与分辨率对 attention 占比的影响；有卡的话在 Wan 1.3B 上对比 dense / SageAttention / STA 的每步时间与 VBench 分项。

### 5. 多卡并行：序列并行、CFG 并行与 PipeFusion——为什么不是张量并行

第五篇讨论多卡：**权重单卡放得下、单请求已经 compute-bound，多卡的目标是把一个请求的墙钟切短——该切什么？**

这一篇会覆盖：

- 为什么不是 TP：TP 每层两次 all-reduce、通信量 $$\propto N \cdot d$$，序列长时通信超过计算；且它解决的是"权重放不下"，扩散模型多数放得下；
- 三类并行：**图间**——data parallel（多请求）与 **CFG 并行**（条件 / 无条件两个分支放两组卡，恒为 2，通信只有每步末的一次小张量交换）；**图内**——序列并行：Ulysses（all-to-all 切 head）、Ring（P2P 切序列、与 FlashAttention 融合）、**USP**（两者组合，节点内 Ulysses、跨节点 Ring）；**层间**——**PipeFusion**（把 latent 切成 $$M$$ 个 patch、网络切成 $$N$$ 段流水线，用上一步的 stale activation 做 KV 让流水线不等——利用的正是第三篇的时间冗余；通信量最小，适合 PCIe / 以太网）与 DistriFusion（patch 并行 + 异步 all-gather 的 stale activation）；
- Parallel VAE：把解码器的输入 latent 切 patch 分卡、卷积边界用 halo 交换——解决 VAE 解码的显存峰值；
- 通信量公式与硬件：NVLink 节点内 USP 最优（xDiT：FLUX 4×H100 compile 后 1.63 s，2.6×）；PCIe / 以太网上 PipeFusion（两台 8×L40 用 ulysses 4 × pipefusion 4）；
- 混合并行的乘积 = GPU 数；CFG 2 × USP 4 vs USP 8 该怎么比（vLLM-Omni 的 8 卡候选矩阵）；
- 视频里 SP 是必需：不是为了快，是 14 GiB 的激活与十几分钟的单卡时间；
- 与训练侧的对照：Wan 的训练同样用 FSDP + SP，序列并行是 DiT 训练与推理共用的并行。

核心问题是：

> **8 张 H100 生成一张 FLUX 1024²，CFG 2 × Ulysses 4 与 TP 8 各通信多少字节、几秒？两台以太网互联的 8×L40 该选什么组合，为什么 PipeFusion 在这里赢？**

实践建议：用 xDiT 或 SGLang Diffusion 在 2 / 4 / 8 卡上扫 Ulysses / Ring / CFG 并行的组合，对每种记录每步时间与 NCCL 时间占比，与通信量公式对账。

### 6. 少步与自回归：把步数变成系统参数

第六篇讨论账上最大的那个乘数——步数——被算法侧改掉之后系统的变化：**当 28 步变成 4 步、当"一次生成整段视频"变成"一段一段流式生成"，前面五篇的哪些结论失效、哪些回归？**

这一篇会覆盖：

- 步数蒸馏与 guidance 蒸馏的系统含义（不推导方法，方法在算法地图 L7 第六篇）：FLUX.1-schnell 4 步无 CFG = dev 的 1/7 FLOPs，单卡 1 s；SD3-Turbo、LCM、DMD2 的 1–4 步；
- 失效的：跨步缓存（4 步之间没有冗余，首末步还要全算）、PipeFusion（依赖 stale activation 的相似性）、CFG 并行（没有 CFG 了）；仍有效的：attention 后端、编译、量化、序列并行；
- 形态变了：单卡亚秒 → 多卡的意义从延迟变为吞吐 → batch 开始有意义（小模型、少步下 GEMM 不饱和）→ 调度像 LLM 的 prefill 池；
- 实时交互：StreamDiffusion 的 stream batch——把处于不同去噪步的连续帧拼成一个 batch 一次前向；
- **自回归视频**：CausVid（把双向 DiT 改成因果、DMD 蒸馏到 4 步、按 chunk 生成、单卡 9.4 fps）→ **Self-Forcing**（训练时按推理方式 rollout、rolling KV cache、单张 4090 实时流式）→ Causal Forcing；
- **KV cache 的回归**：因果视频模型对已生成的 chunk 做 KV cache，每 chunk 的 KV 字节数、滑动窗口的长度、长视频的误差累积；服务形态从批任务变为**会话**——08 系列的前缀缓存、会话状态、流式输出在这里重新出现；SGLang 的 `realtime/` 与 `layers/kvcache/`、vLLM-Omni 的 `diffusion_kv/`；
- 世界模型 / 交互式生成（LingBot World、Cosmos 一类）作为这条线的延伸。

核心问题是：

> **FLUX.1-schnell 4 步 vs dev 28 步：每张 FLOPs、单卡 QPS、哪些优化还有用？自回归视频模型每个 chunk 的 KV cache 多大、滑动窗口留几帧、为什么一个"没有 KV cache"的负载又需要 KV cache 了？**

实践建议：对比 schnell 与 dev 在单卡上开 / 关 TeaCache 与 compile 的每张图时间；用 Self-Forcing 的开源实现看一次 chunk 生成的 KV 占用与逐 chunk 延迟。

### 7. serving 形态：请求、批、三段分离、附件、异步任务与成本

第七篇把前六篇的机制放进一个服务：**一个对外的生成 API 该怎样接请求、排队、分卡、算钱？**

这一篇会覆盖：

- 请求形态：T2I、I2I / 编辑（多一次 VAE 编码与参考图条件）、T2V、I2V、加 ControlNet / IP-Adapter / LoRA 的附件；参数 size / steps / cfg / seed **决定时长**——收到请求就知道它要几秒；
- 与 LLM serving 的对照表：无 KV、无 decode、时长确定、静态 batch 可行、抢占只在步边界有意义；
- 批处理的反直觉：compute-bound 下 batch 2 ≈ 2× 时间，吞吐几乎不变（视频永远 batch 1）；只有小模型 / 少步 / 小分辨率下 GEMM 不饱和时 batch 有收益；SGLang 的动态批处理（合并形状与参数兼容的请求）与 vLLM-Omni 的 step 级调度器；
- 调度：时长可预测 → 最短作业优先、按分辨率 / 步数分池、SLO 排队；抢占的价值与代价；
- **三段分离**：文本编码器（小、一次、可以 CPU 或独立）、DiT（主体）、VAE 解码（显存峰值、可以独立 stage 或 Parallel VAE）——vLLM-Omni 的 stage-based 部署与 OmniConnector、SGLang 的 disaggregation；SwiftDiffusion 把 ControlNet 做成独立服务、LoRA 用 bounded async loading（前 $$k$$ 步不加 LoRA、边算边加载）；
- 多 LoRA 服务：merge 与 unmerged 的代价、按请求切换、Nunchaku 的 4-bit + LoRA；模型级联（DiffServe：先小模型、判别器不过关再上大模型）；
- 异步任务 API：`/v1/images/generations` 同步返回 vs `/v1/videos` 创建 job + 轮询 + 对象存储，进度与中间预览；
- 成本：每张图 GPU·秒 → 价格（FLUX 1024² 在 H100 上 4–6 s ≈ 几美分；视频几分钟）；扩缩容（队列长度、冷启动 = 加载几十 GiB 权重）；对 11 平台的要求。

核心问题是：

> **一个 100 QPS 的 FLUX 1024² 服务需要多少张 H100？batch 有没有用？p99 怎样保证？为什么视频服务必须做成异步 job？**

实践建议：用 SGLang Diffusion 或 vLLM-Omni 起一个服务，用不同并发打 `/v1/images/generations`，记录吞吐与 p50 / p99 随并发的曲线，验证"batch 不提吞吐"。

### 8. 三个引擎的对照导读

第八篇进入源码：**同一张 FLUX 图的请求，在 SGLang Diffusion、vLLM-Omni、xDiT 里各经过哪些进程、哪些类、哪些函数？**

这一篇会覆盖：

- 进程模型：xDiT 是 torchrun 起的 SPMD 进程、每个进程跑同一段 pipeline；SGLang Diffusion 是 HTTP server + scheduler + GPU worker（继承 SGLang 的结构）；vLLM-Omni 是 stage 进程（orchestrator / API 与 worker 可以分卡、分机）；
- pipeline 抽象：diffusers 的 pipeline 类是底座；SGLang 的原生 pipeline（`runtime/pipelines/`，stage 组合）与 `--backend diffusers` 回退；vLLM-Omni 的 `diffusion/models/` 适配器与 diffusers loader；xDiT 的 `xFuserPipelineBaseWrapper` 包装 diffusers pipeline；
- 并行组的建立：三者的 `parallel_state` 与 `GroupCoordinator`（都从 vLLM / Megatron 的写法演化而来）；USP 在 attention 层的接入点；CFG 并行在哪一层切；
- attention 后端的选择器；缓存 hook 的挂载点；量化与 LoRA 的加载路径；
- 调度器与 API：SGLang 的 scheduler 与动态批处理、vLLM-Omni 的 `StepScheduler` / `RequestScheduler` 与 OpenAI 兼容层、xDiT 的 Ray 部署；
- 分歧点：SGLang 把扩散塞进 LLM serving 的结构里（scheduler、kernel、warmup、CUDA graph 都复用）；vLLM-Omni 为全模态模型（LLM + DiT + TTS 串起来）设计 stage 流水线，扩散是其中一种 stage；xDiT 只做并行、不做 serving，是另两者并行方法的来源。

核心问题是：

> **一个 `/v1/images/generations` 请求从进 HTTP 到返回 base64，在三个引擎里各经过哪些函数？它们在进程模型、pipeline 抽象、并行组、调度上各做了什么不同的选择，为什么？**

实践建议：在三个引擎里各跑一次同样的 FLUX 请求，打开 profiler，把每步的时间线对到本篇讲的函数上；对比三者在同一硬件、同一并行度下的每步时间。

### 9. 配置、评测与排障

最后一篇把前八篇变成决策与运维：给定模型、分辨率、步数、GPU 与 SLO，推出配置；跑起来后怎样评测、坏了怎么查。

这一篇会覆盖：

- 配置推导顺序：模型 / 分辨率 / 步数 → 单卡够不够（显存与延迟 SLO）→ 单卡优化叠加（无损优先）→ 跨步缓存（有损、看质量预算）→ 并行度（延迟不够再切）→ batch / 实例数 / 分池；
- 性能评测方法：warmup 与编译排除在外、步级计时、三段分开、按分辨率 bucket；imgs/s/GPU 与延迟分开报；ABBA 交替测量；
- **加速的质量评测**：FID 不够用（对单图无意义、对模式坍缩不敏感）；对基线图的 PSNR / SSIM / LPIPS；ImageReward / HPSv2 / GenEval；视频用 VBench 与时间一致性；人工 A/B；阈值扫描曲线；
- 确定性：seed、确定性算子、并行度是否改变结果（SP 的 reduce 顺序）、编译 / CUDA graph 的漂移（SSIM 0.98 而非 bit-exact）；
- 常见故障：VAE 解码 OOM、FP8 的 NaN 与色偏、缓存阈值的伪影与闪烁、动态分辨率的重编译风暴、SP 度数不整除 token 数、LoRA 未生效或 scale 错、T5 长 prompt 截断、3D VAE 时间分块的接缝；每类的信号与排查路径；
- 可观测：每步耗时、缓存命中率、队列深度、GPU 利用率高不等于有效（eager 的小 kernel 也能把利用率打满）；

核心问题是：

> **一个 FLUX 服务上线后 p99 抬升 / 图片出现伪影 / 半夜 OOM，各先查什么？开训前该采集哪些信号才能十分钟内定位？**

实践建议：为练手服务搭一个面板（每步耗时、三段时间、缓存命中率、队列、显存峰值），人为制造三类故障（换一个更激进的缓存阈值、发一批非常规分辨率、关掉 VAE tiling）并用面板定位。

### 10. 系列总结与通关自测

最后一篇不讲新内容：把九篇正文压成一张「问题 → 结论 → 必记数字」的表并逐篇回顾，拎出贯穿全系列的几条线与常见误区，然后给一套三段式通关自测——十道判断与计算、五道跨篇综合、若干道面试题，答案各自折叠，附「读过 / 掌握 / 能教人」的判据。各篇末尾的自测检验的是一篇读懂了没有，这一篇检验的是九篇能不能连起来用；读完正文再做。

## 贯穿全系列的实践线

系列只有第一篇有配套脚本：`ai-learning-labs` 的 `diffusion-inference-infra/diffusion_ledger.py`，一张能对任意模型、形状、步数、硬件给出三段 FLOPs / 显存 / 时间的账。后面各篇把新引入的机制用公式与表格记回这张账，不再单独给脚本——它们要验证的东西（每步毫秒、缓存命中率、通信时间、图片质量）都要在真实 GPU 上量，纸面模型给出的是量之前该期待的数字。各篇末尾的"实践建议"是给有卡读者的动手方向。

```text
第一篇    账本脚本                    token 数 · 每步 FLOPs 拆分 · 三段显存与时间 · LLM 对照（配套脚本）
第二篇    单卡叠加实验                FA3 / compile / FP8 / INT4 逐项打开 · 每步 ms · 峰值显存 · PSNR / LPIPS
第三篇    阈值扫描                    TeaCache 阈值 0.1–0.8 · 命中位置 · 时间 · 质量曲线
第四篇    帧数与稀疏对比              ledger 的 --sweep · Wan 1.3B 上 dense / Sage / STA
第五篇    并行组合扫描                2 / 4 / 8 卡 Ulysses / Ring / CFG · 每步 ms · NCCL 占比
第六篇    少步与流式                  schnell vs dev · Self-Forcing 的 chunk KV 与延迟
第七篇    并发曲线                    并发 1–16 打 /v1/images/generations · 吞吐与 p99
第八篇    三引擎同一请求的时间线      profiler 对函数
第九篇    面板与故障注入              三类故障定位 · 配置推导记录
```

源码阅读线（第八篇的主体，其余各篇末尾的对照表指向这里）：

```text
diffusers v0.40.0     src/diffusers/pipelines/ · models/transformers/ · hooks/（CacheMixin、group offload）
SGLang v0.5.19        python/sglang/multimodal_gen/：runtime/{pipelines,pipelines_core,distributed,layers/attention,cache,managers,entrypoints,realtime}
vLLM-Omni v0.28.0     vllm_omni/diffusion/：{diffusion_engine.py,sched,worker,distributed,attention,cache,offloader,diffusion_kv,lora}
xDiT（2026-09-02 主线）  xfuser/：core/{distributed,long_ctx_attention,cache_manager} · model_executor/{pipelines,layers,cache} · parallel.py
```


## 阅读路径建议

### 第一遍怎么读（全栈 / 新手读者）

```text
1 → 7
```

约 2 小时。第一篇把一次生成的三段账算出来、与 LLM 对照，第七篇看这种负载的服务长什么样——读过 08 系列前两篇的读者到这里就知道"另一半 serving"与 LLM 的差别在哪。中间五篇的优化机制与第八篇的源码在真的要做生成服务时读。

### 完整学习路径

```text
1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9
```

### 只做图像生成服务

```text
1 → 2 → 3 → 7 → 9
```

图像模型的 $$N$$ 只有几千，attention 占比小、单卡放得下，优化以单卡（第二篇）与跨步缓存（第三篇）为主，多卡只在延迟 SLO 逼迫时用。

### 做视频生成

```text
1 → 4 → 5 → 6 → 9
```

视频的账由 attention 与序列长度主导，第四篇的稀疏化、第五篇的序列并行、第六篇的自回归流式是三条主要出路。

### 平台 / 调度团队

```text
1 → 7 → 9
```

第一篇知道一个请求要几秒几 GiB，第七篇知道服务的形态与对资源层的要求，第九篇知道该看什么指标。

### 主要目标是读懂引擎源码

```text
1 → 5 → 8
```

第八篇是主体；第一篇给出每个函数在处理的量级，第五篇的并行组是三个引擎里最"Infra"、也最相似的一段。


## 本系列的边界

本系列只讨论扩散模型（含 flow matching 模型，系统上无区别）**推理**的系统。以下内容与它紧邻，但不在范围内：

- **扩散模型的数学、结构与训练**：DDPM / score matching / flow matching、DiT 与 MMDiT、VAE 的设计、文生图与视频的配方、步数蒸馏的方法。它们是算法地图 L7 的[《多模态：从视觉编码器到扩散模型》](/multimodal-from-vision-encoders-to-diffusion.html)第五至七篇；本系列只使用"一步是一次对 $$N$$ 个 token 的前向、有没有 CFG、蒸馏到几步"这些结论。
- **LLM 推理系统**：KV cache、连续批处理、PagedAttention、投机解码、PD 分离。它们在[《大模型推理系统揭秘》](/deep-dive-into-vllm.html)；本系列在每个对应位置说明"扩散为什么不同"，不重讲 LLM 侧。
- **多模态理解模型**（把图片送进 LLM）的推理：vision encoder 的调度、image token 的 KV、请求形态。它们在[《Transformer 与 LLM》](/transformer-and-llm-for-infra-engineers.html)第八篇与 08 系列第十一篇；生成模型与它们除了"都有一个 vision 部件"之外没有共同的系统问题。
- **kernel 的实现**：FlashAttention、SageAttention、block-sparse attention、量化 GEMM 的内部。它们在[《GPU Kernel 工程》](/gpu-kernel-engineering.html)；本系列只用它们的接口与加速比。
- **集合通信的实现**：all-to-all、P2P、all-gather 的算法与调优。本系列只用它们的语义与带宽。
- **扩散模型的后训练与 RL**：Diffusion-DPO、奖励微调、Flow-GRPO 的系统。它们的 rollout 就是本系列讲的推理，训练侧属于 09 系列的形态。
- **训练侧的并行**：DiT 的 FSDP + SP 训练。第五篇只在对照处提及。


## 前置要求与说明

### 前置要求

- 理解 LLM 推理引擎的基本机制：prefill 与 decode 的形态差别、KV cache 的字节数、连续批处理、TP 部署、roofline 与 MFU（[《大模型推理系统揭秘》](/deep-dive-into-vllm.html)前五篇的内容；本系列每处对照都会先复述所需的最小集）；
- 知道 Transformer 一层的 FLOPs 从哪来（$$2 P N$$ 与 $$4 N^2 d$$，[《Transformer 与 LLM》](/transformer-and-llm-for-infra-engineers.html)第二篇）；
- 知道扩散模型在做什么：从噪声 latent 出发、网络预测噪声或速度、几十步去噪、VAE 解码、CFG 是两次前向（算法地图 L7 第五、六篇的内容；本系列第一篇会用一节复述所需的最小集，不涉及数学）；
- 会用 diffusers 跑一个文生图 pipeline、读 Python 源码、用 profiler 看时间线；
- 一张 24 GB 以上的 GPU 用于实践；多卡与视频的内容以计算外推与公开数据为主。

不要求：

- 用过 SGLang Diffusion、vLLM-Omni 或 xDiT；
- 了解扩散模型的数学推导；
- 了解 CUDA 编程或 NCCL 内部。

### 框架与版本基线

- **diffusers v0.40.0**（2026-08-20）：模型定义、pipeline、调度器与 hook 接口的底座；
- **SGLang v0.5.19**（2026-09-03）的 `sglang.multimodal_gen`（SGLang Diffusion）：原生 pipeline、USP / CFG 并行、Cache-DiT / TeaCache、动态批处理、breakable CUDA graph、disaggregation、realtime 会话；
- **vLLM-Omni v0.28.0**（2026-08-31）的 `vllm_omni.diffusion`：stage 分离的部署、TP / USP / Ring / CFG 并行、HSDP、VAE patch 并行、TeaCache / MagCache / Cache-DiT 后端、diffusion KV；
- **xDiT**（`xfuser`，无版本 tag，以 2026-09-02 的主线 commit `07572e7` 为准）：USP、PipeFusion、CFG 并行、Parallel VAE 的参考实现；
- 论文以其 arXiv 版本为准：xDiT / PipeFusion / USP、DistriFusion、TeaCache、SVDQuant、Sparse VideoGen 1 / 2、Radial Attention、CausVid、Self-Forcing、StreamDiffusion、SwiftDiffusion、DiffServe；
- 硬件以 **H100 SXM**（80 GB HBM3，BF16 dense 约 989 TFLOPS，3.35 TB/s）为默认分析对象，**RTX 4090**（24 GB，165 TFLOPS）作单卡 / 实时算例；模型以 **FLUX.1-dev**（12B）、**SD3-medium**（2B）、**Qwen-Image**（20B）为图像算例，**Wan2.1-14B**、**HunyuanVideo**（13B）为视频算例；给出的实测数字均注明来源（xDiT 与 SGLang Diffusion 的 benchmark 页、各论文），会因版本与集群而异。

三个引擎都在快速迭代，正文以**机制**为主：账、并行方法、缓存方法、量化方法、分离方式——这些在三个引擎里的写法已经趋同，比任何一个的 flag 稳定。源码引用只到目录与关键类 / 函数，不引用行号。


## 章节目录

1. [负载画像：一次生成在 GPU 上发生什么——三段流水线的 FLOPs、显存与时间账](/diffusion-inference-workload-anatomy-and-cost-ledger.html)
2. [单卡执行：attention 后端、编译、FP8 / INT4 与 offload](/single-gpu-diffusion-execution-attention-compile-quantization-offload.html)
3. [跨步冗余：TeaCache、First-Block Cache 一族的缓存与跳步](/timestep-redundancy-caching-and-step-skipping.html)
4. [视频：长序列 attention 的账与稀疏化](/video-diffusion-long-sequence-attention-and-sparsity.html)
5. [多卡并行：序列并行、CFG 并行与 PipeFusion——为什么不是张量并行](/multi-gpu-diffusion-parallelism-usp-cfg-pipefusion.html)
6. [少步与自回归：把步数变成系统参数——蒸馏后哪些优化失效、KV cache 的回归、实时流式](/few-step-and-autoregressive-video-generation-systems.html)
7. [serving 形态：请求形态、批处理、三段分离、LoRA / ControlNet、异步任务 API 与成本](/diffusion-serving-shapes-batching-disaggregation-and-cost.html)
8. [三个引擎的对照导读：同一张图的请求在 SGLang Diffusion、vLLM-Omni 与 xDiT 里各走过什么](/diffusion-engines-compared-sglang-diffusion-vllm-omni-xdit.html)
9. [配置、评测与排障：从一张卡的推导到一条伪影的排查](/diffusion-inference-configuration-evaluation-and-troubleshooting.html)
10. [系列总结与通关自测](/diffusion-inference-infra-series-recap-and-self-test.html)


## 最终目标

读完这套系列之后，面对任何一个图像或视频生成的推理任务——无论是要上线一个新模型、给现有服务降本、还是接手一个慢得莫名其妙的部署——读者应该能够回答：

```text
这次生成要多少 FLOPs、多少显存、几秒？瓶颈在算力还是别处？      → 第一篇：三段的账
单卡还能快多少？哪些优化不改图、哪些会改？                      → 第二篇：换 MFU
几十步里有多少步可以不算？代价是什么？                          → 第三篇：跨步冗余
视频为什么是 attention 负载？稀疏化能换回多少？                  → 第四篇：长序列
该切序列、切 CFG 还是切流水线？NVLink 和以太网上答案为什么不同？  → 第五篇：多卡
蒸馏到 4 步之后系统怎么变？自回归视频为什么又要 KV cache？        → 第六篇：步数
服务该怎样排队、分卡、算钱？视频为什么是异步 job？               → 第七篇：serving
三个引擎各把这些放在哪？该选哪个？                              → 第八篇：源码
配置怎么推？质量怎么测？坏了从哪查？                            → 第九篇：运维
```

最终目标是三种能力：

1. **算账能力**：给定模型、形状、步数、硬件，算出三段的 FLOPs / 显存 / 时间，判断瓶颈，并预估每一类优化能换回多少；
2. **选型与配置能力**：为一个生成服务选引擎、定并行度、决定开哪些加速与它们的质量预算，并解释每个选择在账上的依据；
3. **运维能力**：为一个持续对外的生成服务设计评测、确定性、监控与告警方案，在图片质量、延迟与成本三方都可能出问题的前提下把每张图的成本维持在可解释的水平。

这是 AI-Infra 引擎层里推理这条主线的**另一半**：同样是"把一个模型的前向跑得快、稳、省"，但因为负载从 memory-bound 换成了 compute-bound，几乎每一个答案都换了。
