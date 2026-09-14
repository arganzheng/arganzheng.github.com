---
layout: post
series: multimodal
title: "多模态（02）：VLM 的结构：connector、注入方式与动态分辨率"
subtitle: "VLM Architecture: Connectors, Injection Methods and Dynamic Resolution"
tags: [AI, Multimodal, VLM, Architecture]
catalog: true
---

编码器输出了几百个向量，它们要进入 LLM。这一步有三个设计决定：**connector**——用什么把编码器的 $$d_v$$ 维特征映射到 LLM 的 $$d$$ 维输入空间，顺便要不要压缩 token 数；**注入方式**——图片 token 是像文本一样进入 LLM 的输入序列（decoder-only 注入），还是通过额外的 cross-attention 层被 LLM "看"（cross-attention 注入）；**分辨率策略**——固定尺寸、切 tile、还是让编码器接受原生分辨率。三个决定合起来回答一个问题：一张图在 LLM 里占多少 token、保留了多少信息、花了多少算力。

[04 系列第八篇](/multimodal-vision-encoder-cost-and-image-token-kv.html)已经算过这三个决定的**成本**：三类 connector 的 token 数表、两种注入的 FLOPs 与 KV 对照、M-RoPE 的三维位置、视频与音频的 token 数。这一篇讲成本背后的**动机与效果**：为什么 2024 年后主流从 Q-Former 回到 MLP、cross-attention 注入为什么被 Llama 3.2 选中又被多数人放弃、原生分辨率解决了 tile 的什么问题。每个选择都是一次"信息 vs token"的交换，这一篇把交换的两边都说清楚。

本篇要回答的核心问题是：

> **LLaVA 的一个 MLP 与 BLIP-2 的 Q-Former 相差什么？为什么 Qwen2-VL 要让 ViT 接受原生分辨率？**


## 一、总览：三个决定、一张地图

### 1. 主流 VLM 的结构选择

| 模型 | 编码器 | connector | 压缩 | 注入 | 分辨率 | 一张图的 token |
|---|---|---|---|---|---|---|
| Flamingo（2022） | NFNet | Perceiver resampler | 固定 64 | cross-attention（gated） | 固定 | 64（不进序列） |
| BLIP-2（2023） | EVA ViT-g | Q-Former（32 query） | 固定 32 | decoder 序列 | 224 | 32 |
| LLaVA-1.5（2023） | CLIP-L/14-336 | 2 层 MLP | 无 | decoder 序列 | 336 | 576 |
| Qwen-VL（2023） | ViT-bigG | cross-attention resampler（256 query） | 固定 256 | decoder 序列 | 448 | 256 |
| LLaVA-NeXT（2024） | CLIP-L/14-336 | MLP | 无 | decoder 序列 | AnyRes ≤ 4 tile + 全图 | 576 × 5 = 2880 |
| InternVL 1.5 / 2（2024） | InternViT-6B / 300M | pixel shuffle + MLP | 4× | decoder 序列 | 448 tile ≤ 40 + 缩略图 | 256 / tile |
| Qwen2-VL（2024） | 675M ViT，2D RoPE | 2×2 merge + MLP | 4× | decoder 序列 | 原生动态 | $$HW / (28^2)$$，可设上下限 |
| Llama 3.2 Vision（2024） | ViT-H/14 | 投影 | 无 | **cross-attention**（每 4 层插 1 层） | 560，≤ 4 tile | 不进序列 |
| Idefics3（2024） | SigLIP-SO400M | pixel shuffle | 4× | decoder 序列 | tile | 169 / tile |
| Molmo（2024） | CLIP-L/14-336 | 2×2 pooling + MLP | 4× | decoder 序列 | tile（≤ 12） | 144 / tile |
| Gemma 3（2025） | SigLIP-SO400M-896 | 平均池化 | 16× | decoder 序列 | 896 固定 + pan & scan | 256 |
| Qwen2.5-VL（2025） | 675M ViT，窗口 attention | 2×2 merge + MLP | 4× | decoder 序列 | 原生动态 | 同 Qwen2-VL |

三个趋势读得出来：（1）注入方式几乎全部收敛到 decoder 序列；（2）connector 从"固定 query 的 resampler"（Flamingo、BLIP-2、Qwen-VL）转向"MLP + 空间压缩"（2×2 merge / pixel shuffle / pooling）；（3）分辨率从固定转向 tile 或原生动态。

### 2. 先说答案

LLaVA 的 MLP 与 BLIP-2 的 Q-Former 差在**信息瓶颈的位置**。MLP 对每个 patch 特征独立做一个映射，576 个 patch 进去 576 个 token 出来，空间结构与全部信息保留，代价是 token 多；Q-Former 用 32 个可学习的 query 对全部 patch 做 cross-attention，把任意数量的 patch 压成 32 个 token，token 少，但 32 个向量装不下一张图的细节——尤其是 OCR、小物体、精确位置——且 query 是**与内容无关**的（同一组 query 用于所有图），它学到的是"平均而言什么重要"，对每张图的具体细节没有自适应。实证上 LLaVA-1.5 用 MLP 在 12 个 benchmark 上超过了用 Q-Former 的 BLIP-2 与 InstructBLIP，之后 Q-Former 基本退出。2024 年的折中是 MLP + **空间压缩**（2×2 merge）：压缩 4 倍但保持空间结构，每个输出 token 对应一个确定的 2×2 patch 区域。

Qwen2-VL 让 ViT 接受原生分辨率，是为了解决 tile 的三个问题：**边界**（tile 之间没有 attention，一个跨 tile 的物体或一行文字被切断）、**失真**（把任意宽高比的图 resize 或 pad 到正方形 tile 会拉伸或浪费）、**token 效率**（小图也要占满一个 tile 的 576 个 token，大图的 tile 网格与内容的实际密度不匹配）。原生分辨率让 token 数**与图片的像素数成比例**（每 $$28 \times 28$$ 像素一个 token），一张 $$224 \times 224$$ 的图 64 个 token、一张 $$1344 \times 896$$ 的文档 1536 个 token，全图在一个 attention 里。代价是 ViT 的 attention 是 $$O(N^2)$$——Qwen2.5-VL 用窗口 attention 缓解——以及位置编码要换成 2D RoPE。第五章展开。

### 3. 本文的章节安排

| 章 | 主题 | 内容 |
|---|---|---|
| 二 | connector | MLP、空间压缩、resampler 三类的机制、参数量与信息保留；为什么 resampler 退出；压缩率的极限 |
| 三 | 注入方式 | decoder 序列 vs cross-attention 的结构、参数、文本能力保持、上下文占用；Llama 3.2 的选择与代价 |
| 四 | 固定分辨率与 tile | AnyRes 的网格选择；缩略图的作用；tile 数与 token 预算；边界问题 |
| 五 | 原生动态分辨率 | Qwen2-VL 的做法：2D RoPE、可变 patch 数、2×2 merge、上下限；窗口 attention；NaFlex |
| 六 | 视频与多图 | 帧采样与时间合并；M-RoPE 的三个轴；token 预算在帧间的分配；交错图文 |
| 七 | 信息 vs token 的交换 | 一张"任务 × 分辨率 × 压缩"的效果矩阵；文档、自然图、图表各需要什么 |
| 八 | 成本 | 回指 04-08 的账；三个决定各改了什么；训练侧的影响 |
| 九 | 动手（建议） | 分辨率—token—精度的三角 |
| 十 | 本文小结 | |


## 二、connector：从编码器空间到 LLM 空间

### 1. MLP projector

LLaVA（Liu 等 2023）用一个线性层，LLaVA-1.5 改成两层 MLP（GELU）：$$z_i = W_2\, \text{GELU}(W_1 h_i)$$，对每个 patch 特征 $$h_i \in \mathbb{R}^{d_v}$$ 独立作用，输出 $$z_i \in \mathbb{R}^d$$。参数量 $$d_v d + d^2$$：CLIP-L（1024）到 Vicuna-7B（4096）是 $$1024 \times 4096 + 4096^2 \approx 21M$$，可以忽略。token 数不变。

它的性质：**逐 token、无信息损失**（映射是可逆的当 $$d \ge d_v$$）、**保留空间结构**（第 $$i$$ 个输出对应第 $$i$$ 个 patch，位置由 LLM 的位置编码给出）、**不做任何选择**（所有 patch 一律进 LLM，让 LLM 的 attention 决定看哪）。LLaVA-1.5 的实验证明，就凭这个 MLP 加 558K 对齐数据 + 665K 指令数据，在 12 个 benchmark 上超过用 Q-Former 的 InstructBLIP 与用 129M 数据的 BLIP-2——**LLM 的 attention 比一个小的 resampler 更会挑信息**。

### 2. 空间压缩：2×2 merge、pixel shuffle、pooling

576 个 token 对 $$336^2$$ 可以接受，但高分辨率下 token 数爆炸（tile 5 个 = 2880，原生 $$1344^2$$ = 9216）。空间压缩把相邻的 $$k \times k$$ 个 patch 特征合成一个 token：

- **2×2 merge / concat**（Qwen2-VL）：把 $$2 \times 2$$ 的四个 $$d_v$$ 维特征**拼接**成 $$4 d_v$$ 维，再过 MLP 到 $$d$$。信息全部保留（拼接不丢），只是让 LLM 用一个 token 处理原来四个的内容；MLP 的参数量变为 $$4 d_v d + d^2$$。
- **pixel shuffle**（InternVL）：同一件事的另一个名字——来自超分辨率里的 space-to-depth 操作，把 $$H \times W \times C$$ 重排成 $$H/2 \times W/2 \times 4C$$。InternVL 把 $$448^2$$ 的 1024 个 patch 压成 256 个。
- **平均 / 注意力池化**（Gemma 3、Molmo）：$$k \times k$$ 个特征**平均**（或加权平均）成一个 $$d_v$$ 维向量再投影。有信息损失（平均丢掉了四个 patch 之间的差别），但更便宜。Gemma 3 用 $$4 \times 4$$ 的平均池化把 $$896^2$$ 的 4096 个 patch 压到 256——16 倍。

压缩率的极限在哪？经验上 **4×（2×2）几乎无损**——一个 $$28 \times 28$$ 像素的区域用一个 LLM token 表示，对文字（一个字约 $$20 \times 20$$ 像素）够用；**16×（4×4）在 OCR 与细节任务上开始掉**，Gemma 3 用 pan & scan（对大图或非方形图再切块）补偿；更高的压缩（64×）只在缩略图或视频帧上用。[04-08](/multimodal-vision-encoder-cost-and-image-token-kv.html)第三章给了 2×2 merge 的精确布局。

### 3. resampler：Q-Former 与 Perceiver

**Perceiver resampler**（Flamingo，Alayrac 等 2022）：$$K$$ 个可学习的 query 向量（$$K = 64$$）对编码器的全部 patch 特征做若干层 cross-attention（query 是 learned latents，key / value 是 patch 特征），输出 $$K$$ 个向量。任意数量的 patch → 固定 $$K$$ 个 token。

**Q-Former**（BLIP-2，Li 等 2023）：结构类似（32 个 query、12 层 BERT 式的 cross-attention + self-attention），但训练分两阶段——先用图文对比 / 匹配 / 生成三个目标训 Q-Former 让 query 学会提取"与文本相关"的特征，再接 LLM 训第二阶段。

**Qwen-VL 的 resampler**：单层 cross-attention，256 个 query，加了 2D 绝对位置编码在 key 上以保留位置信息。

resampler 的问题：（1）**固定预算**——32 或 64 个 token 装不下高分辨率图的信息，OCR 与细节任务上明显落后；（2）**query 与内容无关**——同一组 query 对所有图，它学到的是"通用的重要性"，不能对每张图自适应地决定保留什么；（3）**空间结构丢失**——输出的 $$K$$ 个 token 没有明确的空间对应，LLM 难以做定位与 grounding；（4）**多了一个要训的模块**——Q-Former 的第一阶段训练需要 129M 图文对，且它的表示与 LLM 的表示之间又有一个 gap。

LLaVA-1.5 之后的实证（Cambrian-1 的系统对照、MM1 的消融）一致：在同样的数据与 LLM 下，MLP（+ 空间压缩）优于 resampler，尤其在需要细节的任务上。resampler 在 2024 年后只剩两种用法：**视频**（帧太多，需要激进压缩到固定预算）与 **cross-attention 注入**（下一章，Flamingo 式结构本身就是 resampler + cross-attention）。

### 4. connector 需要多大

MM1（McKinzie 等 2024）的消融：connector 的类型（MLP / 池化 / C-Abstractor）对最终效果的影响**远小于**分辨率与 token 数的影响。connector 只要不丢信息、不引入瓶颈就够了；真正决定效果的是进 LLM 的 token 承载了多少信息（分辨率）与 LLM 有多少 token 可以看（token 数）。这是本篇后半的主题。


## 三、注入方式

### 1. decoder 序列注入

图片 token 与文本 token 拼成一个序列送进 LLM，图片 token 占据序列位置、参与 causal attention、有自己的位置编码、产生 KV。LLM 结构不改，只是输入里有一段"看不懂但能 attend"的 token。LLaVA 起的所有主流 VLM 都是这个。

性质：**简单**（LLM 零改动）、**复用一切**（LLM 的后训练、推理引擎、量化、投机解码全部直接用）、**多图与交错自然**（图片 token 在文本里出现在哪就是哪）。代价：**图片 token 占上下文**——一张高分辨率图 1–3K token，十张图就是 10–30K，与文本竞争窗口；**每层都要为图片 token 算 attention 与 FFN**——04-08 算过一张 576 token 的图在 7B LLM 里的 prefill 约 8 TFLOPs、KV 约 72 MB（Llama 结构）。

### 2. cross-attention 注入

Flamingo 与 Llama 3.2 Vision：图片特征**不进**序列。LLM 的每 $$k$$ 层（Flamingo 每层、Llama 3.2 每 4 层）之间插入一个新的 **cross-attention 层**——文本 token 作为 query，图片特征作为 key / value——再加一个 gate（初始为零的 tanh 门，让训练开始时 LLM 行为不变）。图片特征只作为被 attend 的对象，不产生 KV、不占位置。

性质：**不占上下文**（图片再多也不消耗文本窗口）；**LLM 的文本能力严格不变**（原来的层一个参数没动，gate 关掉就是原 LLM——Llama 3.2 Vision 在纯文本任务上与 Llama 3.1 完全相同）；**图片特征可以更多**（不进序列，几千个特征的成本只在 cross-attention 层）。代价：**新增参数**（Llama 3.2 90B 的 cross-attention 层约 20B 参数——每 4 层一个 cross-attention 层，每个约 $$4d^2$$）；**需要单独训**（这些层从零开始，需要大量图文数据）；**多图与交错的处理复杂**（哪张图对哪段文本可见需要额外的 mask 逻辑）；**推理引擎要特殊支持**（vLLM 为 Llama 3.2 Vision 单独实现了 encoder-decoder 式的 attention 路径）。

### 3. 为什么主流是序列注入

Llama 3.2 Vision 是 2024 年唯一的主流 cross-attention 模型，它的选择有明确的动机：Meta 要保证视觉版本的文本能力**与 Llama 3.1 完全一致**（产品承诺），且愿意为此付出 20B 参数与单独的训练。多数团队没有这个约束，序列注入的简单性与生态兼容性胜出。NVLM（NVIDIA 2024）做了直接对照：同数据下 decoder 序列注入（NVLM-D）在 OCR 与推理任务上优于 cross-attention（NVLM-X），后者训练更快；他们的折中是混合结构（NVLM-H：缩略图进序列、高分辨率 tile 走 cross-attention）。

一个中间形态值得知道：**序列注入 + 图片 token 不产生 loss、用双向 attention**。PaliGemma 让图片 token 之间用双向 attention（prefix 部分非因果），文本部分因果——图片内部的 patch 本来就没有先后顺序，双向让每个 patch token 看到全图。这不改变注入方式，只改 mask。


## 四、固定分辨率与 tile

### 1. 固定分辨率

LLaVA-1.5 把任何图 resize（保持宽高比、pad 到正方形）到 $$336^2$$，576 个 token。简单、成本固定，但对文档（一页 A4 缩到 336 像素宽——每个字 3 像素）与宽图（pad 浪费一半）不可用。TextVQA / DocVQA 上的表现由此受限。

### 2. AnyRes：切 tile

LLaVA-NeXT（2024）：把图按预设的网格（$$1 \times 2$$、$$2 \times 2$$、$$1 \times 3$$……从一组候选里选与原图宽高比最接近、且总 tile 数不超上限的）切成若干 $$336^2$$ 的 tile，每个 tile 独立过编码器得 576 个 token；再加一个全图 resize 到 $$336^2$$ 的**缩略图**（global view）提供全局上下文。$$2 \times 2$$ + 缩略图 = 2880 个 token。InternVL 1.5 把上限推到 40 个 tile（$$448^2$$，压缩后每 tile 256 token → 最多 10K token），Molmo 12 个。

缩略图的作用被反复证实：没有它，模型看到的是四块互不相连的局部，对"这张图整体是什么"的判断变差；有它，全局与局部都有。tile 之间在 LLM 里通过 attention 交互，但**编码器内部**tile 是独立的——一个跨 tile 边界的物体，两半各自被编码，编码器不知道它们是同一个东西。对文字行尤其明显：一行字被切成两段。

### 3. 网格选择与 token 预算

AnyRes 的网格候选是超参数：LLaVA-NeXT 用 $$\{1 \times 1, 1 \times 2, 2 \times 1, 2 \times 2, 1 \times 3, 3 \times 1\}$$（≤ 4 tile）；InternVL 用 1 到 40 个 tile 的所有宽高比组合。选择规则：找与原图宽高比最近的网格，把图 resize 到该网格的总尺寸（可能有轻微拉伸）。token 预算随 tile 数线性增长，一张 $$4000 \times 3000$$ 的文档照片在 InternVL 里可能用掉 25 个 tile = 6400 token。

tile 方案的优点是编码器**完全不变**（每个 tile 是标准的 $$336^2$$ 或 $$448^2$$ 输入，位置编码不需要插值），可以直接用任何现成的 CLIP / SigLIP。这是它在 2024 年流行的工程原因。


## 五、原生动态分辨率

### 1. Qwen2-VL 的做法

Qwen2-VL（Wang 等 2024）让 ViT 直接处理任意大小的图：

1. **可变 patch 数**：图按原尺寸（resize 到 28 的倍数，保持宽高比，像素总数限制在 $$[\text{min}, \text{max}]$$ 之间——默认 $$256 \times 28^2$$ 到 $$1280 \times 28^2$$）切成 $$14 \times 14$$ 的 patch，patch 数 $$N = HW / 14^2$$ 随图变化。
2. **2D RoPE**：ViT 的位置编码从可学习的绝对位置换成 2D RoPE（[04-08](/multimodal-vision-encoder-cost-and-image-token-kv.html)第六章），head_dim 的一半编码行、一半编码列，任意 $$H \times W$$ 无需插值。
3. **2×2 merge**：ViT 输出后相邻 $$2 \times 2$$ 的 patch 特征拼接过 MLP，token 数变为 $$N / 4 = HW / 28^2$$。
4. **M-RoPE**：进入 LLM 后，图片 token 的位置用三维（时间、高、宽）编码，文本 token 三维相同——让 LLM 知道每个图片 token 的二维位置。

一张 $$224 \times 224$$ 的图：64 个 token；$$448 \times 448$$：256；$$1344 \times 896$$（一页文档）：1536；上限 1280 个 token（约 $$1000 \times 1000$$）。**token 数与像素数成正比**，小图不浪费、大图不截断（在上限内）。

### 2. 它解决了什么

对照 tile 的三个问题：**边界**——全图在一个 ViT 的 attention 里，跨区域的物体与文字行完整；**失真**——不 pad 不拉伸（只 resize 到 28 的倍数，误差 < 14 像素）；**效率**——一张小图标 64 个 token 而不是 576，一张宽的横幅按实际比例给 token。Qwen2-VL 在 DocVQA（94.5）、InfoVQA、OCRBench 等文档任务上相比 tile 方案有明显优势，且 token 用得更少。

### 3. 代价与 Qwen2.5-VL 的窗口 attention

ViT 的 attention 是 $$O(N^2 d)$$。$$N = 5120$$ 个 patch（$$1280 \times 4$$，merge 之前）时 attention 的 FLOPs 是 $$2 \times 5120^2 \times 1280 \approx 67$$ GFLOPs 每层，32 层 2.1 TFLOPs，与 patch 的线性部分（$$2 \times 675M \times 5120 \approx 6.9$$ TFLOPs）同量级——不可忽略，且激活内存 $$N^2$$ 增长。Qwen2.5-VL 把 ViT 的大部分层改成**窗口 attention**（$$112 \times 112$$ 像素的窗口，即 $$8 \times 8$$ 个 patch，只有 4 层保留全局 attention），把 attention 成本变成线性。这与 Swin Transformer 的思路一致，效果上几乎无损。

另一个代价是**工程**：可变 token 数的 batch 需要 padding 或 packing（Qwen2-VL 用 packing——多张图的 patch 拼成一个序列，用 attention mask 隔开，与 [L5 第一篇](/sft-data-chat-template-loss-mask-and-peft.html)讲的 SFT packing 同一个技术）。

### 4. NaFlex 与趋势

SigLIP 2 的 NaFlex 变体让预训练的编码器本身支持原生宽高比与可变分辨率（训练时就用多种分辨率与宽高比，位置编码按实际网格插值）——让"原生动态"不再需要自己从头训 ViT。2025 年新发布的 VLM（Kimi-VL 的 MoonViT、InternVL 3 的部分配置）越来越多走原生动态路线；tile 方案因为工程简单仍广泛存在。


## 六、视频与多图

### 1. 视频：帧的采样与合并

视频是图的序列，token 数 = 帧数 × 每帧 token。1 分钟视频以 1 fps 采样 60 帧，每帧 256 token = 15K token；以 2 fps 就是 30K。三个控制手段：

- **帧率**：Qwen2-VL 默认 2 fps 采样；长视频降到 0.5 fps 或按总 token 预算自适应。
- **时间合并**：相邻两帧的 patch 在时间维上合并（Qwen2-VL 的 3D patch：$$2 \times 14 \times 14$$，两帧合成一组 patch），token 数减半；视频编码器（如 InternVideo）可以在更长的时间窗上合并。
- **每帧分辨率**：视频帧用比单图更低的分辨率与更高的压缩（Qwen2-VL 视频帧的 token 上限单独设）。

Qwen2.5-VL 用**绝对时间**的 M-RoPE（时间维的位置 id 与真实秒数对齐，而不是帧序号），让模型知道两帧之间隔了多久——对"第 30 秒发生了什么"一类的定位任务重要。

### 2. M-RoPE 的三个轴

[04-08](/multimodal-vision-encoder-cost-and-image-token-kv.html)第六章讲了 M-RoPE 的结构：head_dim 分成三段，分别编码时间、高、宽。文本 token 三个 id 相同（退化为 1D RoPE）；图片 token 时间 id 固定、高宽 id 是二维网格位置；视频 token 时间 id 随帧递增。它让 LLM 用同一套位置编码处理三种模态，且图片内部的相对位置（"左边的物体"）在 RoPE 的相对性下有意义。后续 token 的位置 id 从图片占据的最大 id + 1 开始，所以一张 $$32 \times 32$$ token 的图只"消耗"32 个位置而不是 1024 个——对长上下文外推有利。

### 3. 多图与交错

decoder 序列注入下多图是自然的：每张图的 token 出现在它在文本里的位置，`<image>` 占位符被替换。挑战在**token 预算的分配**——十张图各 1000 token 就是 10K；InternVL、Qwen2-VL 对多图场景自动降低每张图的分辨率上限。交错图文（网页、论文、漫画）的训练数据（MMC4、OBELICS）让模型学会图与它附近文本的对应。


## 七、信息 vs token 的交换

### 1. 一张效果矩阵

把公开报告里的数字放在一起（不同模型的 LLM 不同，只看趋势）：

| 任务 | 需要的信息 | 固定 336 / 576 token | tile ≤ 5 / ~2.9K token | 原生动态 / 可变 | 16× 压缩 / 256 token |
|---|---|---|---|---|---|
| 自然图片理解（VQAv2、MMBench） | 全局语义 + 主要物体 | 足够（LLaVA-1.5 ~80 VQAv2） | 小幅提升 | 小幅提升 | 足够（Gemma 3） |
| 文档 OCR（DocVQA） | 小字可读 | 不够（~60） | 好（~85–90） | 最好（Qwen2-VL 94.5） | 需 pan & scan 补偿 |
| 图表（ChartQA） | 数字、线条、图例 | 中（~60） | 好（~80） | 好（~83） | 中 |
| 场景文字（TextVQA） | 中等大小文字 | 中（~60） | 好（~75） | 好（~85） | 中 |
| 计数 / 定位 | 空间结构完整 | 中 | 边界问题 | 好 | 差 |
| 数学图形（MathVista） | 结构 + 推理 | 依赖 LLM | 小幅 | 小幅 | 小幅 |

结论：**自然图片任务对分辨率与 token 数不敏感**——LLM 的推理能力是瓶颈；**文档、图表、文字任务对分辨率极度敏感**——是 tile 与原生动态方案的主要受益者；**空间 / 计数任务受益于原生动态**（边界完整）。一个 VLM 的设计要看它的目标负载：通用助手可以用 Gemma 3 式的固定 256 token（便宜），文档理解必须高分辨率。

### 2. 每 token 的信息密度

另一个看法：一个 LLM token 该对应多少像素？LLaVA-1.5：$$336^2 / 576 = 196$$ 像素/token（$$14 \times 14$$）；Qwen2-VL：$$28 \times 28 = 784$$；Gemma 3：$$896^2 / 256 = 3136$$（$$56 \times 56$$）。经验上 $$28 \times 28$$（一个汉字或两三个英文字母的大小）是文字任务的甜点；$$56 \times 56$$ 对自然图片够、对文字勉强；$$14 \times 14$$ 浪费——相邻 patch 的信息高度冗余，2×2 merge 几乎无损正是因此。


## 八、成本

### 1. 三个决定各改了什么

回指 [04-08](/multimodal-vision-encoder-cost-and-image-token-kv.html) 的账：

- **connector** 决定 $$N_{img}$$（进 LLM 的 token 数）。LLM 侧的成本——prefill FLOPs $$2 P N_{img}$$、KV $$N_{img} \times$$ 每 token KV——全部随 $$N_{img}$$ 线性。2×2 merge 把这部分除以 4。connector 自身的 FLOPs 可忽略。
- **注入方式** 决定图片 token 是否产生 KV、是否占用位置。cross-attention 注入下图片特征不进 KV，每层的成本是 $$2 \times N_{text} \times N_{img} \times d$$ 的 cross-attention（只在插入的层），比序列注入的每层 $$2 P N_{img}$$（FFN 也要算）便宜；但新增的 cross-attention 层参数在 decode 时也要读。
- **分辨率策略** 决定编码器的成本与 $$N_{img}$$ 的范围。tile：编码器成本 $$\times$$ tile 数；原生：编码器 attention $$O(N^2)$$ 需要窗口化。

### 2. 训练侧

VLM 训练的显存主要由 LLM 决定（与文本 SFT 相同），图片 token 增加序列长度——2880 token 的图 + 500 token 的文本 = 3.4K 的序列，激活内存是纯文本 SFT 的 6–7 倍。编码器是否解冻决定它的优化器状态是否存在（675M 参数 × 16 字节 = 10.8 GB）。数据加载是另一个瓶颈：图片解码与 resize 是 CPU 密集的（04-08 第八章），高分辨率下每个样本几十毫秒，需要足够的 dataloader worker 或预处理。


## 九、动手（建议）

一张 24 GB 的卡，两组对照：

- **分辨率**：Qwen2.5-VL-3B-Instruct，调 `min_pixels` / `max_pixels`（比如 $$256 \times 28^2$$、$$1024 \times 28^2$$、$$4096 \times 28^2$$）在 DocVQA 验证集的 500 题子集与 MMBench 的 500 题子集上各测准确率，并记录每题的平均图片 token 数与 prefill 时间。
- **connector 与 tile**：LLaVA-1.5-7B（固定 336）与 LLaVA-NeXT-7B（AnyRes）在同样的 DocVQA 子集上比；用 `lmms-eval` 统一协议。

该看的：DocVQA 随 token 预算的曲线是否陡而 MMBench 是否平；token 数与 prefill 时间的线性关系；LLaVA-NeXT 相比 LLaVA-1.5 在 DocVQA 上的提升是否远大于在 MMBench 上的。再挑几张跨 tile 边界有文字行的图，比较 AnyRes 与原生动态的输出。不引用任何未跑过的数字。


## 十、本文小结

| 项 | 规则 | 备注 |
|---|---|---|
| MLP projector | 逐 patch 映射，无损，保留空间结构 | LLaVA-1.5 凭它超过 BLIP-2；参数可忽略 |
| 空间压缩 | 2×2 merge / pixel shuffle（拼接，无损）；池化（有损） | 4× 几乎无损；16× 需补偿；$$28 \times 28$$ 像素/token 是文字甜点 |
| resampler | 固定 $$K$$ 个 query 的 cross-attention | 固定预算、内容无关、丢空间结构；退出主流，仅视频与 cross-attn 注入 |
| connector 大小 | 类型影响远小于分辨率与 token 数（MM1） | 不丢信息即可 |
| 序列注入 | 图片 token 进序列，LLM 零改动，复用生态 | 占上下文；每层全算 |
| cross-attn 注入 | 每 $$k$$ 层插 gated cross-attention，图片不进序列 | 文本能力严格不变；+20B 参数；引擎需特殊支持；Llama 3.2 唯一主流 |
| tile | 切 $$336^2$$ / $$448^2$$ 块 + 缩略图；编码器不变 | 边界切断、pad 失真、小图浪费；InternVL ≤ 40 tile |
| 原生动态 | 2D RoPE、$$N = HW/14^2$$、2×2 merge、M-RoPE、上下限 | token ∝ 像素；文档任务最好；ViT attention $$O(N^2)$$ → 窗口化 |
| 视频 | 帧率 × 每帧 token；时间合并 ×2；绝对时间 M-RoPE | 1 分钟 2 fps 256/帧 = 30K |
| 任务敏感性 | 自然图不敏感；文档 / 图表 / 文字极敏感；空间受益原生 | 设计看目标负载 |

核心问题的答案：LLaVA 的 MLP 对每个 patch 独立映射、不丢信息、保留空间结构、把"看哪里"交给 LLM 的 attention；BLIP-2 的 Q-Former 用 32 个与内容无关的 query 把整张图压成 32 个 token，装不下细节、丢了空间结构、且多了一个要单独训的模块——LLaVA-1.5 的实证让主流转向 MLP，2024 年的折中是 MLP + 2×2 merge，压缩 4 倍而无损。Qwen2-VL 让 ViT 接受原生分辨率，是因为 tile 方案切断跨块的物体与文字行、pad 与拉伸造成失真、小图也要占满一个 tile 的 token；原生分辨率让 token 数与像素数成正比、全图在一个 attention 里，用 2D RoPE 取代需要插值的绝对位置编码，代价是 ViT 的 $$O(N^2)$$ attention（Qwen2.5-VL 用窗口 attention 解决）与可变长度的 batch 工程。三个决定合起来是一次"信息 vs token"的交换，交换的合理位置取决于任务：自然图片对 token 数不敏感，文档与文字任务要每 $$28 \times 28$$ 像素一个 token。下一篇讲这个结构怎么训：阶段、数据、评测与幻觉。


## 下一篇

[VLM 的训练：数据、阶段与评测](/vlm-training-recipe-data-stages-and-evaluation.html)
