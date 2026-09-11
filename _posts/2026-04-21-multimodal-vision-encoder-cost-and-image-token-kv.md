---
layout: post
series: transformer-and-llm
title: "Transformer 与 LLM（08）：多模态：vision encoder 的算量与 image token 的 KV 代价"
subtitle: "Multimodal LLMs: The Cost of Vision Encoders and Image Tokens"
tags: [Transformer, LLM, AI, AI-Infra]
catalog: true
---

前七篇讨论的模型只有一种输入：token id。它查一张 embedding 表得到向量，然后进入 decoder。这个前提决定了前面所有的账——参数量、FLOPs、KV cache——都只与 token 数有关，而 token 数由 tokenizer 决定。

多模态模型打破了这个前提。一张图片不经过 tokenizer，它先经过一个**独立的神经网络**（vision encoder，通常是 ViT），被切成几百到几千个向量，再由一个**连接层**（connector）变成 decoder 认得的 token embedding，插进 prompt 的对应位置。于是出现了三笔新账：encoder 自己的算量、connector 决定的 token 数、以及这些 token 进入 decoder 之后与文本 token 完全相同的 prefill FLOPs 和 KV cache。

本篇要回答的核心问题是：

> **一张 1024×1024 的图片，在 Qwen2-VL 里等于多少个 token？它的代价花在 encoder、connector 还是 decoder 的 KV 上？为什么"encoder 输出只有 10 MB"而"这张图占的显存有 400 MB"两句话可以同时成立？**


## 一、总览：三笔账与一个结论

### 1. 先说答案

一张图进入多模态 LLM，要经过三段，每段一笔账：

| 段 | 做什么 | 账 | 性质 |
|---|---|---|---|
| vision encoder | 把像素切成 patch，过一个 ViT | encoder 参数量 × patch 数 × 2 的 FLOPs；一次性 | compute-bound，与文本无关 |
| connector | 把 encoder 输出变成 decoder 的 embedding | 决定 **image token 数**；输出 $$n_{img} \times d_{model} \times 2$$ 字节 | 便宜，但它的压缩比决定后面两笔账 |
| decoder | image token 与文本 token 一起做 prefill 和 decode | prefill $$2N \cdot n_{img}$$ FLOPs；KV cache $$n_{img} \times$$ 每 token KV，**活到请求结束** | 与文本 token 完全同价 |

结论提前给出：**图片贵的不是 encoder，而是它变成的那几百上千个 token 在 decoder 里占的 KV**。以 Llama-3-70B 规格的 decoder（每 token KV 320 KiB）为例，1369 个 image token 的 encoder 输出是 21 MiB，它们的 KV 是 418 MiB，是前者的 20 倍；而且 encoder 输出用完即弃，KV 要陪伴整个请求。

### 2. 本文的路线

沿用系列的方法：写出公式，代入真实模型的 `config.json`，算出数字，解释数字对系统的意义。四个模型贯穿全篇：

| 模型 | 发布 | encoder | connector | decoder | 代表的路线 |
|---|---|---|---|---|---|
| LLaVA-1.5-7B | 2023-10 | CLIP ViT-L/14，336 px，24 层 d=1024 | 2 层 MLP，不压缩 | Vicuna-7B（MHA） | 固定分辨率、最简单的 decoder-only 注入 |
| Qwen2-VL-7B | 2024-08 | 自训 ViT，32 层 d=1280，原生动态分辨率 | 2×2 merge + MLP，÷4 | Qwen2-7B（GQA 4 KV 头） | 动态分辨率 + M-RoPE |
| InternVL2-8B | 2024-07 | InternViT-300M，448 px，24 层 d=1024 | pixel-shuffle + MLP，÷4 | InternLM2.5-7B | 固定 tile + 动态 tile 数 |
| Llama-3.2-11B-Vision | 2024-09 | ViT-H/14，560 px tile，32 局部 + 8 全局层 | 线性投影 | Llama-3.1-8B + 8 层 cross-attention | cross-attention 注入 |

数字都是理论值，硬件基线仍是 H100 SXM（80 GB，3.35 TB/s，BF16 约 989 TFLOPS）。

### 3. 本文的章节安排

| 章 | 主题 | 内容 |
|---|---|---|
| 二 | 从像素到 patch | patchify、ViT 的参数量与 FLOPs 公式、四个 encoder 的数字 |
| 三 | connector：谁决定 token 数 | MLP / 2×2 merge 与 pixel-shuffle / resampler 三类，各自的 token 数公式 |
| 四 | image token 在 decoder 里 | prefill FLOPs、KV cache、encoder 输出字节，三者的数量级对比 |
| 五 | 另一条路线：cross-attention | Llama 3.2 Vision 的结构，它的 KV 为什么不随文本增长 |
| 六 | 位置编码：从一维到三维 | M-RoPE 把 `head_dim` 分给 (t, h, w)，ViT 内部的 2D RoPE |
| 七 | 视频与音频 | 帧 × 每帧 token；Whisper encoder 的 30 秒 → 1500 个位置 |
| 八 | 训练侧 | 冻结 encoder 省什么、不省什么；图片解码是 CPU 的活 |
| 九 | 实践 | `llm_cost.py` 的多模态支持与最终成本表 |
| 十 | 本文小结与系列总结 |  |


## 二、从像素到 patch：vision encoder 的账

### 1. patchify：图片怎么变成"序列"

ViT（Dosovitskiy 等 2020）处理图片的方法与 Transformer 处理文本的方法在形式上完全一样：把输入切成一串向量，然后堆 Transformer 层。区别只在第一步——文本查表，图片切块：

- 把 $$H \times W$$ 的图片切成 $$p \times p$$ 的小方块（patch），主流取 $$p = 14$$；
- 每个 patch 有 $$3 p^2$$ 个像素值（三通道），用一个线性层（等价于 stride = p 的卷积）映射到 $$d_{vit}$$ 维；
- 加上位置编码，得到 $$n_p = (H/p)(W/p)$$ 个 $$d_{vit}$$ 维向量，进入 ViT 的 Transformer 层。

$$n_p = \frac{H}{p} \cdot \frac{W}{p}$$

代入：336 px 的图，$$336 / 14 = 24$$，$$n_p = 576$$；448 px，$$32^2 = 1024$$；560 px，$$40^2 = 1600$$；1024 px（Qwen2-VL 先把边长凑成 28 的倍数，1022 或 1036 px），约 $$73^2 = 5329$$ 到 $$74^2 = 5476$$。

一个 patch 就是 ViT 的一个"token"。ViT 内部的 attention 是**双向**的（不是 causal），每个 patch 看得到全图的所有 patch。

### 2. ViT 的参数量与 FLOPs

ViT 的一层与第一篇的 decoder 层结构相同：attention 四个矩阵加 FFN 两个矩阵（ViT 用 GELU 的经典 FFN，不是 SwiGLU，所以是两个矩阵；FFN 宽度通常是 $$4 d_{vit}$$）。忽略 bias 与 norm：

$$N_{layer} = 4 d_{vit}^2 + 2 \cdot 4 d_{vit}^2 = 12\, d_{vit}^2, \qquad N_{vit} \approx L_{vit} \cdot 12\, d_{vit}^2 + 3p^2 d_{vit}$$

FLOPs 的公式与第二篇相同：权重部分每个 patch 每参数 2 FLOPs，attention 部分每层 $$4 n_p^2 d_{vit}$$（$$QK^\top$$ 与 $$PV$$ 两个 GEMM）：

$$\text{FLOPs}_{vit} = 2 N_{vit} \cdot n_p + 4 L_{vit}\, n_p^2\, d_{vit}$$

代入四个 encoder：

| encoder | $$L_{vit}$$ | $$d_{vit}$$ | 参数量 | 输入 | $$n_p$$ | 权重 FLOPs | attention FLOPs | 合计 | H100 峰值下界 |
|---|---|---|---|---|---|---|---|---|---|
| CLIP ViT-L/14-336（LLaVA-1.5） | 24 | 1024 | 302 M | 336² | 577（含 CLS） | 0.35 T | 0.03 T | 0.38 T | 0.4 ms |
| InternViT-300M（InternVL2） | 24 | 1024 | 302 M | 448² × 1 tile | 1025 | 0.62 T | 0.10 T | 0.72 T | 0.7 ms |
| Qwen2-VL ViT | 32 | 1280 | 629 M | 1024² | 5476 | 6.89 T | 4.91 T | 11.8 T | 11.9 ms |
| Qwen2.5-VL ViT（28 层 window + 4 层 full） | 32 | 1280 | 629 M | 1024² | 5476 | 6.89 T | 0.66 T | 7.55 T | 7.6 ms |
| Llama-3.2 Vision ViT-H/14 | 32 + 8 | 1280 | 786 M | 560² × 4 tile | 6404 | 10.1 T | 8.4 T | 18.5 T | 18.7 ms |

三个观察：

1. **encoder 参数量在 0.3–0.8 B**，是 decoder 的 4–10%。它的权重字节（BF16 0.6–1.6 GB）在显存账里不是主角。
2. **attention 的二次项在高分辨率下会追上权重项**。576 个 patch 时 attention 只占 8%；5476 个 patch 时占 42%；Llama 3.2 的 4 tile 拼成 6404 个 patch 时占 45%。这就是 Qwen2.5-VL 改用 window attention 的原因：32 层里 28 层只在 8×8 patch 的窗口（112 px）内做 attention，attention FLOPs 从 4.9 T 降到 0.66 T，总量降 36%。
3. **encoder 是 compute-bound 的**。一张图的 5476 个 patch 是一个 5476 行的 GEMM，算术强度远超 H100 的 ridge（约 295，见第二篇）；不像 decode 那样受带宽限制。一张 1024² 的图在 Qwen2-VL 上的 12 ms 是算力的时间，用更大的 batch 摊不掉。

第三条对系统的含义：encoder 的时间**与文本无关、与 batch 无关、一次性**。推理引擎可以把它当作独立于 decoder 的一段计算单独调度、单独预算——这是它在系统里被拆成"encoder 预算"的原因，本篇只给出数字，机制不展开。

### 3. 分辨率的策略

四个模型对"图片多大"的处理不同，直接决定了 $$n_p$$：

| 策略 | 代表 | 做法 | $$n_p$$ 的范围 |
|---|---|---|---|
| 固定分辨率 | LLaVA-1.5 | 任何图缩放（或 pad）到 336² | 恒为 576 |
| 固定 tile，动态 tile 数 | InternVL2、Llama 3.2 Vision | 图按长宽比切成 1–12（InternVL）或 1–4（Llama）个 448² / 560² 的 tile，各自过 encoder | 1024 × tile 数 / 1601 × tile 数 |
| 原生动态分辨率 | Qwen2-VL / 2.5-VL | 边长凑到 28 的倍数后整图一次过 encoder，`min_pixels` / `max_pixels` 限制范围 | 从 4 个 patch 到默认上限 16384 × 4 个 patch |

固定分辨率简单但浪费：一张 4K 截图缩到 336² 什么字都看不清。tile 方案让 encoder 每次只处理固定形状（对 kernel 和 batch 友好），代价是 tile 之间没有 attention。原生动态分辨率最灵活，但 $$n_p$$ 可以相差三个数量级，encoder 的 FLOPs 与后面 decoder 的账都随之剧烈变化——系统必须按图片尺寸而不是"图片张数"来预算。


## 三、connector：谁决定 image token 数

### 1. 三类 connector

encoder 输出 $$n_p$$ 个 $$d_{vit}$$ 维向量，decoder 需要 $$n_{img}$$ 个 $$d_{model}$$ 维向量。connector 做两件事：换维度、（可能）减数量。按是否减数量分三类：

| 类型 | 代表 | 做法 | $$n_{img}$$ | 参数量 |
|---|---|---|---|---|
| MLP projector | LLaVA-1.5 | 每个 patch 向量独立过 2 层 MLP（1024 → 4096 → 4096） | $$= n_p$$，576 | $$d_{vit} d + d^2 \approx 21$$ M |
| 空间合并 | Qwen2-VL（2×2 merge）、InternVL（pixel-shuffle） | 相邻 2×2 个 patch 的向量拼接成 $$4 d_{vit}$$ 维，再过 MLP | $$= n_p / 4$$ | Qwen2-VL：5120 → 5120 → 3584，约 45 M |
| 重采样 | Flamingo 的 Perceiver Resampler、BLIP-2 的 Q-Former、MiniCPM-V | 固定数量的可学习 query 对 patch 向量做 cross-attention | 固定 64 / 96 / 256 | 数十到数百 M |

第一类把 token 数的决定权完全交给 encoder；第二类给一个固定的压缩比；第三类把 token 数固定下来，与图片分辨率脱钩。

### 2. 2×2 merge 的精确布局

第二类是当前的主流，值得把布局画清楚。下图以一个 112×112 像素的区域（8×8 个 patch，正好是 Qwen2.5-VL 的一个 attention window）为例：

![patchify → 2×2 merge → decoder 序列](/img/in-post/multimodal-vision-encoder-cost-patch-merge.svg)

三点：

- merge 不是池化，是**拼接**：4 个 $$d_{vit}$$ 维向量拼成一个 $$4 d_{vit}$$ 维向量，信息没有丢，只是让 MLP 决定怎么压。MLP 的输入维度因此是 5120 而不是 1280。
- 每个 image token 对应 28×28 像素。Qwen2-VL 的 token 数公式是 $$n_{img} = \lceil H / 28 \rceil \cdot \lceil W / 28 \rceil$$（先把边长凑到 28 的倍数）：336² → 144，1024² → 37 × 37 = 1369，1920 × 1080 → 69 × 39 = 2691。
- 展平是行优先的：token $$t_k$$ 的二维坐标是 $$(k \,/\, W_{tok},\; k \bmod W_{tok})$$。这个坐标在第六章会被 M-RoPE 用到。

InternVL 的 pixel-shuffle 在数学上与 2×2 merge 相同（把 $$2 \times 2 \times d$$ 重排成 $$1 \times 1 \times 4d$$），448² 的 1024 个 patch 变成 256 个 token。

### 3. token 数的一张表

把三种 connector 与三种分辨率策略合起来，"一张图等于多少 token"的答案是：

| 模型 | 一张 336² | 一张 1024² | 上限 |
|---|---|---|---|
| LLaVA-1.5 | 576 | 576（缩放到 336） | 576 |
| InternVL2 | 256（1 tile） | 256 × tile 数，1024² 约 4 tile + 缩略图 = 1280 | 13 × 256 = 3328 |
| Qwen2-VL | 144 | 1369 | 默认 `max_pixels` 下 16384 |
| Llama-3.2 Vision | 1601（1 tile） | 4 × 1601 = 6404 | 6404 |

同一张 1024² 的图，从 576 到 6404，差 11 倍。这个数决定了下一章的全部内容。


## 四、image token 在 decoder 里：真正的账

### 1. 三个字节数

image token 进入 decoder 之后与文本 token 没有任何区别：一样参与 prefill 的 GEMM，一样在每一层留下 K 和 V。第二、三篇的公式直接可用：

$$\text{FLOPs}_{prefill} = 2 N_{dec} \cdot n_{img}, \qquad \text{KV}_{img} = n_{img} \cdot 2 L\, n_{kv}\, d_{head} \cdot 2\ \text{B}, \qquad \text{Enc}_{out} = n_{img} \cdot d_{model} \cdot 2\ \text{B}$$

代入两种 decoder 规格（Llama-3-8B：32 层、8 个 KV 头，每 token KV 128 KiB；Llama-3-70B：80 层、8 个 KV 头，320 KiB）：

| $$n_{img}$$ | 来源 | 8B：prefill FLOPs | 8B：KV | 8B：encoder 输出 | 70B：prefill FLOPs | 70B：KV | 70B：encoder 输出 |
|---|---|---|---|---|---|---|---|
| 576 | LLaVA 336² | 9.3 T | 72 MiB | 4.5 MiB | 81 T | 180 MiB | 9.0 MiB |
| 1369 | Qwen2-VL 1024² | 22 T | 171 MiB | 10.7 MiB | 193 T | 428 MiB | 21.4 MiB |
| 3328 | InternVL2 13 tile | 53 T | 416 MiB | 26 MiB | 470 T | 1040 MiB | 52 MiB |
| 6404 | Llama-3.2 4 tile | 103 T | 800 MiB | 50 MiB | 904 T | 2001 MiB | 100 MiB |

（Qwen2-VL-7B 自己的 decoder 是 28 层、4 个 KV 头，每 token KV 只有 56 KiB，1369 个 token 的 KV 是 73 MiB；LLaVA-1.5 的 Vicuna-7B 是 MHA，每 token 512 KiB，576 个 token 是 288 MiB——decoder 的 attention 变体对图片的代价影响是 4–9 倍，这正是第三篇 GQA 的价值在多模态上的放大。）

### 2. 三个数量级的对比

以 70B 规格、一张 1024² 的 Qwen2-VL 风格图片（1369 token）为例，把三段的字节数放在一起：

| 项 | 大小 | 生命周期 |
|---|---|---|
| encoder 权重（BF16） | 1.26 GB | 常驻，所有请求共享 |
| encoder 输出 | 21 MiB | prefill 期间用一次；prefill 完成即可释放 |
| image token 的 KV | 428 MiB | 从 prefill 到请求结束（含全部 decode 步） |
| prefill FLOPs | 193 TFLOP ≈ 195 ms（H100 峰值） | 一次性 |

**KV 是 encoder 输出的 20 倍**。原因在公式里：encoder 输出每 token 是 $$d_{model} \times 2$$ 字节（16 KiB），KV 每 token 是 $$2 L n_{kv} d_{head} \times 2$$ 字节（320 KiB），比值是 $$2 L \cdot n_{kv} d_{head} / d_{model} = 2 \times 80 \times 1024 / 8192 = 20$$——K 与 V 两份，乘层数，乘 KV 头总维度与模型维度之比。这个比值对 Llama-3-8B 是 16，对 Qwen2-VL-7B（28 层、4 个 KV 头）是 8。

再加上生命周期的差别：encoder 输出用完即弃，KV 要陪伴请求走完。一个"看一张图、写 300 字说明"的请求，image token 的 KV 会在显存里停留几秒到几十秒，占据的空间等价于 1369 个文本 token。**在 KV cache 的账上，一张 1024² 的图就是一段 1369 token 的 system prompt**——只是这段"prompt"没法被 tokenizer 压短。

### 3. prefill 的另一面

image token 让 prefill 变长，而 prefill 是 compute-bound 的（第二篇）：1369 个 image token 在 70B 上要 193 TFLOP，与 1369 个文本 token 完全相同；再加上 encoder 自己的 11.8 TFLOP，一张图让这个请求的首 token 延迟多了约 200 ms（H100 峰值下界，实际 1.5–2 倍）。

这里有一个常见的误判：encoder 12 ms、decoder prefill 195 ms，看起来 encoder 不重要。但 encoder 的时间是**串行前置**的——decoder 的 prefill 必须等 encoder 输出就绪才能开始（image token 的 embedding 来自它）。系统层面能做的是把 encoder 与其他请求的 decoder 计算重叠，而不是缩短它。

### 4. 几张图与一张图

多图请求（或多轮对话里累积的图）按 token 数线性叠加：Qwen2-VL 里 4 张 1024² 的图是 5476 个 image token，KV 在 70B 规格下 1.7 GiB。这比大多数文本对话的上下文都长。对推理系统而言，多模态请求的 KV 需求方差远大于纯文本请求——纯文本的 prompt 长度分布相对集中，而一个 VLM 服务会同时收到 144 token 的缩略图和 16384 token 的 4K 截图。


## 五、另一条路线：cross-attention 注入

### 1. 图片特征不进 decoder 序列

前面四个模型里有三个把 image token 直接塞进 decoder 的输入序列（decoder-only 注入，也叫 early fusion 或 LLaVA 式）。Llama 3.2 Vision 走了 Flamingo（Alayrac 等 2022）的路线：图片特征**不进序列**，decoder 的某几层增加一个 cross-attention 子层，让文本 token 作为 query 去读图片特征作为 key / value。

Llama-3.2-11B-Vision 的 `config.json` 给出的结构：

- decoder 共 40 层，其中 8 层是 cross-attention 层（第 3、8、13、…、38 层），32 层是原 Llama-3.1-8B 的自注意力层；
- vision encoder 输出 6404 × 7680 维（把 5 个中间层与最后一层的 1280 维拼接），线性投影到 4096 维；
- cross-attention 的 K、V 由 6404 个图片向量算出，Q 由文本 token 算出；K、V 头数与自注意力相同（8 个 KV 头 × 128）。

### 2. 两种注入的成本对照

| | decoder-only 注入 | cross-attention 注入 |
|---|---|---|
| 图片进入哪里 | decoder 输入序列，与文本 token 并列 | 8 个 cross-attention 层的 K / V |
| 序列长度 | 文本 + $$n_{img}$$ | 只有文本 |
| prefill FLOPs（图片部分） | $$2 N_{dec} \cdot n_{img}$$：6404 token × 8B = 103 T | 只有投影：8 层 K、V 各 $$2 d\, d_{kv} n_{img}$$ 共 0.9 T，加 7680 → 4096 的视觉投影 0.4 T，约 1.3 T |
| 图片 KV | 全部 32 层自注意力：6404 × 128 KiB = 800 MiB | 8 层 cross-attention：6404 × 2 × 8 × 128 × 2 B × 8 = 200 MiB |
| 每步 decode 多读的字节 | 多读 6404 token 的 KV（800 MiB） | 多读 cross-attention 的 K / V（200 MiB） |
| 图片 KV 随文本增长？ | 否，但文本 KV 与图片 KV 在同一份 cache 里 | 否，且与文本 KV 分离，形状固定 |
| 参数量 | 不变 | 每个 cross-attention 层 $$\approx 4 d^2$$，8 层约 0.5 B（11B = 8B + 0.9B ViT + cross-attn + 投影） |
| 结构改动 | decoder 完全不变，任何 LLM 都能接 | decoder 加层，权重要重训 |

cross-attention 用**参数**换**序列长度**：多了 0.5 B 参数，换来 decoder 序列不被图片撑长、图片 KV 减到四分之一。代价是 decoder 不再是"标准的 Llama"——推理引擎要为它单独实现 cross-attention 的 KV 管理（图片 KV 的形状与文本 KV 不同，不能放进同一套分页），训练框架也要处理两种 attention 的并行切分。这是它在开源社区里不如 decoder-only 注入流行的工程原因；Llama 4 已经改回 early fusion。


## 六、位置编码：从一维到三维

### 1. 问题

RoPE（第四篇）给每个 token 一个一维位置 $$m$$，把 $$d_{head}$$ 维的 q、k 分成 $$d_{head}/2$$ 对，每对以不同频率旋转 $$m \theta_i$$。image token 排成一行之后当然可以沿用一维位置：LLaVA 就是这么做的，576 个 token 从左到右编号。但这丢掉了二维信息——第 24 个 token 与第 25 个 token 在图上是"这一行的末尾"与"下一行的开头"，一维位置只差 1；而第 1 个与第 25 个在图上是上下相邻，一维位置差 24。

### 2. M-RoPE：把 head_dim 分给三个轴

Qwen2-VL 的 M-RoPE（Multimodal RoPE）把 $$d_{head} = 128$$ 的 64 对旋转维度分成三段（`mrope_section: [16, 24, 24]`）：16 对用时间位置 $$t$$ 旋转，24 对用高度位置 $$h$$，24 对用宽度位置 $$w$$。每个 token 的位置不再是一个数，而是三元组 $$(t, h, w)$$：

| token 类型 | $$(t, h, w)$$ |
|---|---|
| 文本 token（第 $$m$$ 个） | $$(m, m, m)$$——三个分量相同，退化为普通一维 RoPE |
| 图片第 $$(i, j)$$ 个 token（图片起始位置 $$s$$） | $$(s, s + i, s + j)$$ |
| 视频第 $$f$$ 帧第 $$(i, j)$$ 个 token | $$(s + f, s + i, s + j)$$ |
| 图片之后的文本 | 从 $$s + \max(H_{tok}, W_{tok})$$ 继续，而不是 $$s + n_{img}$$ |

最后一行是 M-RoPE 对长上下文的一个副作用：一张 37 × 37 = 1369 个 token 的图，只让位置编号前进 37 而不是 1369。图片在位置空间里占的"长度"是它的边长，不是它的面积。这对第四篇讨论的 RoPE 外推范围是个好消息，但**不改变 KV cache 的账**——KV 仍然是 1369 份。位置编码决定 attention 怎么"看"，不决定要"存"多少。

对 kernel 的含义：M-RoPE 的实现是三组不同的 $$\cos / \sin$$ 表按维度段拼接后做一次普通 RoPE 旋转，计算量与一维 RoPE 相同；但 position id 从一个 `[seq]` 向量变成 `[3, seq]`，推理引擎的 position 管理要随之改动。

### 3. ViT 内部的 2D RoPE

encoder 自己也要位置编码。CLIP ViT 用的是可学习的绝对位置 embedding，固定 577 个位置，这也是它只能处理固定分辨率的原因之一（换分辨率要插值位置表）。Qwen2-VL 的 ViT 改用 2D RoPE：$$d_{head}$$ 的一半用 patch 的行号旋转、一半用列号旋转，位置与分辨率无关，这是原生动态分辨率的前提。


## 七、视频与音频

### 1. 视频：帧数乘上每帧 token

视频是一串图片。Qwen2-VL 把相邻两帧在时间维上合并（`temporal_patch_size: 2`），所以 token 数是"每两帧的图片 token 数 × 帧对数"：

$$n_{video} = \frac{F}{2} \cdot \frac{H}{28} \cdot \frac{W}{28}$$

一段 60 秒、1 fps 采样、720p（1280 × 720 → 46 × 26 = 1196 token / 帧对）的视频：30 个帧对 × 1196 = **35880 个 token**。在 70B 规格下 KV 是 11 GiB，prefill 5 PFLOP（约 5 秒）。一分钟视频比 128K 上下文的四分之一还长。

所以视频模型的全部工程都围绕"减 token"：降采样帧率（Qwen2-VL 默认 2 fps 但有 `max_pixels` 限制单帧）、把每帧压到很低的分辨率、在时间维做更激进的合并。它们都是在改公式里的 $$F$$ 和 $$H \cdot W / 28^2$$，账的形式不变。

### 2. 音频：Whisper encoder 的 1500 个位置

音频 LLM（Qwen2-Audio 等）用 Whisper 的 encoder 做前端。它的输入是 30 秒音频的 log-mel 谱：16 kHz 采样、10 ms 一帧、3000 帧；两层卷积把时间轴减半到 **1500 个位置**，然后过 32 层 $$d = 1280$$ 的 Transformer（large 版，encoder 约 0.63 B 参数）。

$$n_{audio} = \frac{T_{sec}}{30} \times 1500 = 50\ \text{token / 秒}$$

一段 30 秒音频经 encoder 是 2.3 TFLOP、输出 1500 × 1280 × 2 B = 3.7 MiB；如果 connector 不压缩，就是 1500 个 token 进 decoder。实际模型通常再做 2–4 倍的时间维合并（Qwen2-Audio 池化到 25 token / 秒）。音频每秒 12–50 token，介于文本（每秒说话约 3–4 token）与视频（每秒数百到上千）之间。

### 3. 三种模态的一张表

| 模态 | 单位 | encoder | 每单位 token 数 | 主要调节手段 |
|---|---|---|---|---|
| 文本 | 1000 汉字 | 无（查表） | 约 600–1000 | tokenizer 词表 |
| 图片 | 1024² | ViT，7–12 TFLOP | 576（固定）/ 1369（动态 ÷4）/ 6404（4 tile） | 分辨率、merge 比、tile 数 |
| 视频 | 1 分钟 720p | ViT × 帧对数 | 35880（1 fps）| 帧率、单帧分辨率、时间合并 |
| 音频 | 1 分钟 | Whisper encoder，4.5 TFLOP | 3000（不压缩）/ 1500（Qwen2-Audio） | 时间维池化 |

所有模态最终都归结为同一个数——进入 decoder 的 token 数。**decoder 不知道也不关心 token 从哪里来**，它的 prefill FLOPs 和 KV 只看这个数。encoder 的差异只影响前置的一次性计算。


## 八、训练侧的账

### 1. 冻结 encoder 省的是状态，不是激活

多模态模型的训练通常分阶段：先冻结 encoder 与 LLM、只训 connector（对齐），再解冻 LLM（指令微调），encoder 是否解冻各家不同（LLaVA-1.5 冻结，Qwen2-VL 在前两个阶段训练 ViT、第三阶段冻结）。用第六篇的训练状态公式看冻结省了什么：

| 组件 | 参数 | 冻结时的状态 | 解冻时的状态（16 B/参数） |
|---|---|---|---|
| ViT（0.63 B） | 1.26 GB BF16 | 1.26 GB（只有权重） | 10 GB |
| connector（45 M） | 90 MB | — | 0.7 GB |
| LLM（7.6 B） | 15.2 GB | 15.2 GB | 122 GB |

冻结 ViT 省 9 GB 状态，相对 LLM 的 122 GB 只有 7%——状态不是冻结的主要收益。真正省的是**激活值**：encoder 位于计算图的最前端，它的参数不需要梯度、它前面也没有需要梯度的层，所以反向传播到 connector 就停了，ViT 前向的中间张量一个都不必保存（可以在 `torch.no_grad()` 下跑）。一张 1024² 图片的 5476 个 patch × 32 层，每层十几个 `seq × d` 的张量，激活值在 10 GB 量级，比它的训练状态还大；解冻 ViT 意味着这些全部要留到反向。

### 2. 序列长度的方差

纯文本预训练可以把样本打包成定长序列（第七篇的 `cu_seqlens`），每个 micro-batch 的 token 数恒定。多模态样本的 token 数由图片分辨率决定，同一 batch 里 144 与 5476 并存；打包算法要按 token 数而非样本数装箱，否则 GPU 在小图样本上空转。这是 Qwen2-VL 一类原生动态分辨率模型在训练效率上付出的代价，也是 tile 方案（每个 tile 的 token 数固定）在训练时的优势。

### 3. 图片解码是 CPU 的活

一张 1024² 的 JPEG 解码加 resize 加归一化，在一个 CPU 核上是毫秒级；encoder 在 H100 上处理它也是十毫秒级。文本预训练里数据加载几乎不占 CPU，多模态训练里每张卡每秒要喂几十到几百张图，8 卡机器的 CPU 很容易先于 GPU 饱和。数据管线的形态从"读 token id"变成"解码图片"，这是训练基础设施在多模态上遇到的第一个实际瓶颈，解法（预处理离线化、GPU 解码 nvJPEG、DALI）都是在把这一步搬离 CPU。


## 九、实践：llm_cost.py 的多模态支持

### 1. 新增的函数

延续全系列的 `llm_cost.py`，本篇新增 vision encoder 的参数与 FLOPs、image token 数、image token 在 decoder 中的三个字节数。`ModelConfig`、`param_count`、`kv_bytes_per_token` 沿用第七篇的定义。

```python
from dataclasses import dataclass
from math import ceil

@dataclass
class VisionConfig:
    name: str
    layers: int
    hidden: int
    patch: int = 14
    mlp_ratio: int = 4
    merge: int = 1          # 空间合并的边长：Qwen2-VL / InternVL 为 2，LLaVA 为 1
    tile: int = 0           # 固定 tile 边长（px）；0 表示原生动态分辨率
    max_tiles: int = 1
    cls_token: int = 0

CLIP_L_336 = VisionConfig("CLIP ViT-L/14-336", 24, 1024, tile=336, cls_token=1)
INTERN_VIT_300M = VisionConfig("InternViT-300M", 24, 1024, merge=2, tile=448, max_tiles=13)
QWEN2_VL_VIT = VisionConfig("Qwen2-VL ViT", 32, 1280, merge=2)
LLAMA32_VIT = VisionConfig("Llama-3.2 ViT-H/14", 40, 1280, tile=560, max_tiles=4, cls_token=1)

def vit_params(v):
    return v.layers * 12 * v.hidden ** 2 + 3 * v.patch ** 2 * v.hidden

def patches_per_tile(v):
    return (v.tile // v.patch) ** 2 + v.cls_token

def image_patches(v, h, w, tiles=1):
    """一张 h×w 图片进入 encoder 的 patch 数。"""
    if v.tile:
        return tiles * patches_per_tile(v)
    f = v.patch * v.merge
    return ceil(h / f) * ceil(w / f) * v.merge ** 2

def image_tokens(v, h, w, tiles=1):
    """connector 之后进入 decoder 的 token 数（tile 方案里 CLS 不进 decoder）。"""
    if v.tile:
        return tiles * ((v.tile // v.patch) ** 2 // v.merge ** 2 + v.cls_token)
    return image_patches(v, h, w) // v.merge ** 2

def vit_flops(v, n_patches, window=0, full_layers=0):
    """一张图的 encoder FLOPs；window>0 时按窗口 attention 计，full_layers 层做全图 attention。"""
    weight = 2 * vit_params(v) * n_patches
    if window:
        attn = 4 * n_patches * window * v.hidden * (v.layers - full_layers) \
             + 4 * n_patches ** 2 * v.hidden * full_layers
    else:
        attn = 4 * n_patches ** 2 * v.hidden * v.layers
    return weight, attn

def image_cost_in_decoder(cfg, n_img, dtype_bytes=2):
    """image token 在 decoder 里的三个数：prefill FLOPs、KV 字节、encoder 输出字节。"""
    gemm_params = param_count(cfg)["total"] - cfg.vocab * cfg.hidden
    return {
        "prefill_flops": 2 * gemm_params * n_img,
        "kv_bytes": n_img * kv_bytes_per_token(cfg, dtype_bytes),
        "encoder_out_bytes": n_img * cfg.hidden * dtype_bytes,
    }

if __name__ == "__main__":
    MiB = 2 ** 20
    for v, (h, w, tiles) in [(CLIP_L_336, (336, 336, 1)), (QWEN2_VL_VIT, (1024, 1024, 1)),
                             (INTERN_VIT_300M, (1024, 1024, 5)), (LLAMA32_VIT, (1024, 1024, 4))]:
        n_p, n_t = image_patches(v, h, w, tiles), image_tokens(v, h, w, tiles)
        wf, af = vit_flops(v, n_p)
        print(f"{v.name:22s} patches {n_p:5d} tokens {n_t:5d} "
              f"encoder {(wf + af) / 1e12:5.2f} TFLOP (attn {af / (wf + af):.0%})")
        for cfg in (LLAMA3_8B, LLAMA3_70B):
            c = image_cost_in_decoder(cfg, n_t)
            print(f"    {cfg.name:12s} prefill {c['prefill_flops'] / 1e12:6.1f} TFLOP  "
                  f"KV {c['kv_bytes'] / MiB:7.1f} MiB  enc-out {c['encoder_out_bytes'] / MiB:5.1f} MiB")
```

运行输出（节选）：

```text
CLIP ViT-L/14-336      patches   577 tokens   576 encoder  0.38 TFLOP (attn 8%)
    Llama-3-8B   prefill    9.3 TFLOP  KV    72.0 MiB  enc-out   4.5 MiB
    Llama-3-70B  prefill   81.3 TFLOP  KV   180.0 MiB  enc-out   9.0 MiB
Qwen2-VL ViT           patches  5476 tokens  1369 encoder 11.80 TFLOP (attn 42%)
    Llama-3-8B   prefill   22.0 TFLOP  KV   171.1 MiB  enc-out  10.7 MiB
    Llama-3-70B  prefill  193.3 TFLOP  KV   427.8 MiB  enc-out  21.4 MiB
Llama-3.2 ViT-H/14     patches  6404 tokens  6404 encoder 18.47 TFLOP (attn 45%)
    Llama-3-8B   prefill  102.9 TFLOP  KV   800.5 MiB  enc-out  50.0 MiB
    Llama-3-70B  prefill  904.2 TFLOP  KV  2001.2 MiB  enc-out 100.1 MiB
```

### 2. 成本表新增的一列

全系列的成本表在第七篇完成了三个文本模型的对照。本篇给它加上"一张 1024² 图片"这一行，按三种注入方式放到 Llama-3-8B 规格的 decoder 上：

| 一张 1024² 图片 | LLaVA 式（576） | Qwen2-VL 式（1369） | Llama-3.2 式（6404，cross-attn） |
|---|---|---|---|
| encoder FLOPs | 0.38 T | 11.8 T | 18.5 T |
| encoder 时间下界（H100） | 0.4 ms | 12 ms | 19 ms |
| decoder prefill FLOPs | 9.3 T | 22 T | 1.3 T（仅投影） |
| prefill 时间下界 | 9 ms | 22 ms | 1.3 ms |
| image KV（8B 规格） | 72 MiB | 171 MiB | 200 MiB（8 层 cross-attn，与 decoder 层数无关） |
| encoder 输出 | 4.5 MiB | 10.7 MiB | 50 MiB（7680 维拼接前） |
| 等价于多长的文本 prompt | 576 token | 1369 token | KV 上约 1600 token；序列长度上 0 |

### 3. 实验设计

有 GPU 时可以验证两件事：

1. **encoder 时间与 batch 无关**：用 `transformers` 加载 Qwen2-VL-7B，只跑 `visual` 子模块，输入 1 张与 8 张 1024² 图片，测时间。预期接近线性（compute-bound），与 decode 那种"8 个请求几乎不比 1 个慢"形成对照。
2. **image token 就是 token**：用同一模型对比"1369 个文本 token 的 prompt"与"一张 1024² 图片 + 几个字"的首 token 延迟与 `torch.cuda.max_memory_allocated()` 的增量。预期后者比前者多出的只有 encoder 的 12 ms 与 encoder 输出的 10 MiB；KV 增量相同。


## 十、本文小结与系列总结

### 1. 本文小结

三笔账与它们的量级（Qwen2-VL 风格、1024² 图片、70B 规格 decoder）：

| 项 | 公式 | 数字 | 性质 |
|---|---|---|---|
| encoder FLOPs | $$2 N_{vit} n_p + 4 L_{vit} n_p^2 d_{vit}$$ | 11.8 TFLOP | 一次性、compute-bound、与 batch 无关 |
| image token 数 | $$(H/28)(W/28)$$ | 1369 | 由分辨率与 connector 压缩比决定 |
| encoder 输出 | $$n_{img} \cdot d_{model} \cdot 2$$ B | 21 MiB | prefill 后即可释放 |
| prefill FLOPs | $$2 N_{dec} \cdot n_{img}$$ | 193 TFLOP | 与同样长度的文本相同 |
| image KV | $$n_{img} \cdot 2 L n_{kv} d_{head} \cdot 2$$ B | 428 MiB | 活到请求结束；是 encoder 输出的 $$2 L n_{kv} d_{head} / d_{model} = 20$$ 倍 |

核心问题的答案：一张 1024² 的图在 Qwen2-VL 里是 1369 个 token；encoder 的 12 ms 和 21 MiB 输出是前置的一次性开销，而 1369 个 token 在 decoder 里的 prefill FLOPs 和 428 MiB 的 KV 与 1369 个文本 token 完全相同，且 KV 要活到请求结束。**"encoder 输出 21 MB"与"这张图占 400 MB 显存"同时成立，因为前者是 connector 的输出、后者是它在每一层留下的 K 和 V；两者之比是层数乘以 KV 头维度与模型维度之比。** cross-attention 注入用 0.5 B 参数把图片 KV 压到四分之一并让它不进序列，代价是 decoder 不再是标准结构。位置编码（M-RoPE）改变图片在位置空间里占的长度（边长而非面积），但不改变 KV 的账。

### 2. 全系列总结

八篇文章，每篇留下几个公式和几个数字：

```text
第一篇  参数量        每层 attention d(d_q + 2d_kv + d_q)、FFN 3·d·d_ff；Llama-3-8B 218.1M/层 × 32 + 1.05B = 8.03B；
                      70B 70.55B；DeepSeek-V3 671B（每 token 激活 37B）；Mixtral 46.7B（激活 12.9B）
第二篇  FLOPs·字节    GEMM 2mkn；每参数每 token 2 FLOPs；8B 每 token 15 GFLOPs；prefill 8K 约 158 TFLOP；
                      decode 算术强度 ≈ B；H100 ridge 295；decode 下界 16.06 GB / 3.35 TB/s = 4.8 ms
第三篇  KV cache      2·L·n_kv·d_head·bytes；8B GQA 128 KiB/token（MHA 512 KiB）；70B 320 KiB；
                      MLA (512+64)×2×61 = 68.6 KiB，压缩 57×；一张 H100 放 8B 后约 50 万 token 的 KV
第四篇  长上下文      RoPE 波长 2π·base^(2i/d)；base 500000 最低频 ~250 万；attention 二次项 4ds/层；
                      8B 128K prefill 权重 2.0 PFLOP + attention 4.5 PFLOP；128K KV 16 GiB
第五篇  MoE           期望激活专家 E·[1−(1−k/E)^B]：DeepSeek-V3 B=32 → 163，B=128 → 252；
                      dispatch 7 KiB + combine 14 KiB 每 token 每专家；grouped GEMM 每专家 128 行
第六篇  数值          BF16 1/8/7 ε=2^-7；FP16 max 65504，softmax 溢出 x > 11.09；E4M3 max 448；
                      混合精度 + Adam 16 B/参数，8B 训练状态 128 GB；FP8 每 128 元素提升 FP32 累加
第七篇  量化·投机·LoRA INT4 g128 4.25 bit，70B 37.5 GB 单卡；W4A16 decode 4.8 → 1.27 ms，转折 B ≈ ridge/4；
                      投机 E = (1−α^(γ+1))/(1−α) = 3.36，加速 2.4×，转折 B ≈ ridge/(γ+1)；
                      LoRA r=16 41.9M（0.52%），训练状态 128 GB → 16.7 GB
第八篇  多模态        ViT 12·L·d²，0.3–0.8 B；image token = (H/28)²（÷4 merge）；1024² → 1369；
                      encoder 11.8 TFLOP 一次性；image KV = 文本 KV，70B 规格 428 MiB，是 encoder 输出的 20 倍
```

贯穿这些数字的是**四组变量**的成本模型：

$$
\text{参数量 } N \ \to\ \text{FLOPs/token} \approx 2N,\ \text{权重字节} = N \cdot \text{bytes/param},\ \text{KV cache/token} = 2 L n_{kv} d_{head} \cdot \text{bytes/elem}
$$

再加一张卡的两个上限（算力 $$F$$、带宽 $$BW$$）和一个比值（ridge $$= F / BW$$）。结构（GQA、MLA、MoE、RoPE）决定前三组变量的值；精度（BF16、FP8、INT4）决定 bytes；工作点（batch、序列长度、prefill 还是 decode）决定落在 Roofline 的哪一侧；多模态不引入新变量，只是让 token 数由分辨率而非 tokenizer 决定，并在前面加一段一次性的 encoder 计算。每一篇都是在这个模型里填一格。

回到总纲的"最终目标"——拿到一个 `config.json` 和一张 GPU 的规格表，现在能回答：

```text
它有多少参数，分布在哪里？                 → 逐矩阵公式代入 config；8B 里 87% 在 FFN+attention、13% 在 embedding/lm_head
一张卡放得下吗？剩多少显存？               → N × bytes/param；BF16 8B 占 16 GB，70B 需 INT4 才能单卡；剩余给 KV cache
每 token 多少 FLOPs？各阶段瓶颈？          → 2N；prefill compute-bound，decode memory-bound，分界是 ridge
batch 开到多大才能用满算力？               → B ≈ ridge ≈ 295（H100 BF16），FP8 下 591；W4A16 后是 ridge/4
支持多长上下文？代价在哪？                 → KV cache 线性项 + attention 二次项；RoPE 的波长决定外推
attention 变体让 kernel 长什么样？          → GQA 的 4/8 个 query 头共享一个 KV 头；MLA 的吸收让 KV 变成 576 维
MoE 多卡要传多少数据？                     → 每 token 每专家 7 + 14 KiB，乘期望激活专家数
用什么精度？哪一步会出数值问题？            → BF16 前向、FP32 累加与主权重；softmax 与 RMSNorm 的溢出/下溢点
量化能快多少？在哪个阶段？                 → 字节数之比，只在 decode 且 B ≲ ridge/k 时兑现；W8A8 才对 prefill 有效
投机解码值得开吗？上界多少？               → (1−α^(γ+1))/(1−α) 除以 (γc+1)，只在 B ≲ ridge/(γ+1) 时成立
微调需要多少显存？                         → 全量 16 B/参数；LoRA 为 2 B/参数 + 可忽略；激活值另算，随序列长度线性
一张图等于多少 token？贵在哪？             → (H/28)² 或 576 或 1601×tile；encoder 一次性，KV 与同长文本相同且活到请求结束
```

这三种能力——不看 benchmark 先算出理论值、用理论值判断优化的有效区间、用同一张表与算法、kernel、平台工程师对话——是本系列试图建立的全部内容。

本系列的边界也在这里：它只把模型当作一个**计算对象**，算它的参数、算量、字节数与通信量。FlashAttention 与量化 GEMM 的 kernel 怎么写、continuous batching 与 PagedAttention 怎么调度、encoder 在推理引擎里怎么单独预算与缓存、TP / PP / EP 怎么切分与同步、训练配方怎么定——这些都建立在本系列给出的数字之上，但各自是另一个系列的内容。回到总纲：[《Transformer 与 LLM：结构、算量与数值》](/transformer-and-llm-for-infra-engineers.html)。

配套代码：[`transformer-and-llm/llm_cost_08_multimodal.py`](https://github.com/arganzheng/ai-learning-labs/blob/main/transformer-and-llm/llm_cost_08_multimodal.py)（复用第七版的 `ModelConfig`）；本文各表的理论数字由 [`vlm_cost_numbers.py`](https://github.com/arganzheng/ai-learning-labs/blob/main/transformer-and-llm/vlm_cost_numbers.py) 算出。全系列八版脚本与运行输出在 [ai-learning-labs/transformer-and-llm](https://github.com/arganzheng/ai-learning-labs/tree/main/transformer-and-llm)。
