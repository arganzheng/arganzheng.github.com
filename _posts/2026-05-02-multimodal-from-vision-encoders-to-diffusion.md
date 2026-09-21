---
layout: post
title: 多模态：从视觉编码器到扩散模型（总纲）
subtitle: "Multimodal Models: From Vision Encoders to Diffusion"
tags: [AI, LLM, Multimodal, Diffusion]
catalog: true
---


## 内容简介

《多模态：从视觉编码器到扩散模型》是一组共九篇正文加一篇总结的系列文章，对应[《AI 算法工程师学习地图》](/ai-algorithm-engineer-learning-roadmap.html)的第 L7 层。它面向已经理解 Transformer 与 LLM（L4）、做过后训练（L5）的读者，回答两个问题：**图片、视频、语音怎么进入一个语言模型**，以及**图像与视频的生成为什么是另一套数学**。

"多模态"下面有两条几乎独立的线。**理解线**把其他模态编码成 token 送进 LLM：一个视觉编码器（ViT）把图片变成几百个向量，一个 connector 把它们对齐到 LLM 的输入空间，LLM 像处理文本一样处理它们——这条线是 LLM 的扩展，用的是 L4、L5 的全部方法，新增的是编码器、connector 与对齐训练。**生成线**从噪声出发逐步去噪得到图片：扩散模型有自己的目标函数（去噪 / score matching / flow matching）、自己的结构（U-Net → DiT）、自己的采样过程（几十步迭代）与成本结构（无 KV cache、compute-bound）——这条线与 LLM 共享 Transformer 与 scaling 的经验，但数学是新的。两条线在 2025 年开始交汇：自回归图像生成用 LLM 的方式生成图像 token；统一模型让一个 Transformer 既理解又生成。

| 篇 | 主题 | 线 | 回答的问题 |
|---|---|---|---|
| 一 | 视觉编码器：CLIP、SigLIP 与自监督 ViT | 理解 | 一张图变成的几百个向量里有什么？对比学习为什么能学出"语义" |
| 二 | VLM 的结构：connector、注入方式与动态分辨率 | 理解 | 图片 token 怎么进入 LLM？三类 connector 与两种注入各自的取舍；分辨率怎么处理 |
| 三 | VLM 的训练：数据、阶段与评测 | 理解 | 先训什么后训什么、每阶段冻结谁；数据从哪来；多模态幻觉从哪来 |
| 四 | 语音（上）：从波形到 token——mel 谱、Whisper 与神经 codec | 理解 → 生成 | 一秒声音在模型眼里是什么？声音怎么变成离散 token（RVQ） |
| 五 | 语音（下）：语音理解、语音生成与全双工 | 理解 → 生成 | 怎么听、怎么说；LLM 直接说为什么伤文本能力；全双工的时延 |
| 六 | 扩散模型（上）：DDPM——加噪、去噪与「预测噪声」 | 生成 | 去噪为什么等于学会生成？变分下界怎么变成一行 MSE；DDIM 为什么能跳步 |
| 七 | 扩散模型（下）：score matching、flow matching 与 CFG | 生成 | 三种视角为什么是同一件事；轨迹为什么弯、怎么拉直；guidance 在做什么 |
| 八 | Latent diffusion、DiT 与文生图配方 | 生成 | 为什么在 latent 空间做；U-Net 到 DiT；SD / FLUX 的配方；采样加速；视频 |
| 九 | 自回归图像生成与统一模型 | 交汇 | 图像怎么 token 化；AR 生成 vs 扩散；理解与生成能不能用一个模型 |

每一篇都按"小白能看懂"的标准写：每个机制先给一个能在纸上算的小例子、一张图、再给公式，核心代码贴在文中并附真实输出——CLIP 的对比学习、VLM 的 connector、mel 谱、RVQ、DDPM 的加噪去噪、flow matching、CFG、latent diffusion、VQ 与自回归生成，全部在 CPU 上用几十行代码的 toy 跑通过一遍（配套代码见"实践线"）。

读完这个系列，读者应该能够：读懂一个 VLM 的技术报告（编码器选什么、connector 怎么设计、分辨率策略、训练阶段、数据配比、评测），并判断它的每个选择在成本与效果上的取舍；读懂一个文生图模型的技术报告（噪声调度、预测目标、结构、guidance、采样步数），并理解它与 LLM 在训练与推理上的根本不同；知道两条线在哪里交汇、统一模型当前的三种路线各是什么。


## 为什么写这个系列？

### 多模态已经是默认配置

2025 年发布的主要模型几乎都是多模态的：GPT-4o、Gemini 2.5、Claude 的视觉能力、Qwen2.5-VL / Qwen2.5-Omni、Llama 3.2 Vision、Gemma 3、InternVL 3、Kimi-VL。"纯文本 LLM"正在变成一个历史阶段。一个算法工程师如果只懂文本，读不懂当前一半的技术报告。

### 理解线是 L4 + L5 的直接延伸，但有自己的坑

VLM 的 LLM 部分与文本模型完全一样，后训练方法（SFT、DPO、RL）也一样。新的东西集中在三处：编码器与 connector 的设计（一张图占多少 token、信息保留多少）、分辨率的处理（固定 vs 动态、tile vs 原生）、训练阶段的安排（先对齐 connector 还是一起训）。这三处决定了 VLM 的成本与上限，且每一处的选择在各家的报告里差别很大——LLaVA 的一个 MLP 与 BLIP-2 的 Q-Former（早期 Qwen-VL 用的是单层 cross-attention resampler，同一类）相差几倍的 token 数；InternVL 的 tile 与 Qwen2-VL 的原生分辨率是两种哲学。系列的前三篇把这些选择放在同一把尺子上。

多模态还引入了文本模型没有的失效模式：**多模态幻觉**——描述图片里不存在的物体，是语言先验压过视觉证据的结果——以及视觉编码器的"盲点"（CLIP 类编码器对计数、空间关系、文字的弱点）。理解它们的来源才能在数据与训练上对症。

### 生成线的数学是新的

扩散模型不是 LLM 的变体。它的目标函数从变分下界或 score matching 推出，采样是解一个微分方程，结构从 U-Net 演化到 DiT，guidance 是一个对条件分布的重加权。这些在 L0–L5 里没有出现过。但它与 LLM 共享很多经验——Transformer 的 scaling、数据质量的决定性、蒸馏加速采样——理解了这一部分，就能读懂从 Stable Diffusion 到 FLUX、从 Sora 到 Wan 的整条线。

### 两条线在交汇

自回归图像生成（把图像 token 化后用 LLM 的方式生成）与统一模型（一个 Transformer 既理解图片又生成图片）是 2024–2025 年的活跃方向：Chameleon、Emu3、Janus、Transfusion、BAGEL、以及 GPT-4o 的原生图像生成。它们要回答的问题是"理解与生成能不能共享同一套表示"，而这个问题的答案决定了多模态模型的下一个形态。系列的第九篇把两条线合起来讨论。

### 现有材料的断层

VLM 的材料多是各家的技术报告（各说各的选择，没有横向比较）；扩散模型的材料要么是数学推导（Song、Lilian Weng 的博客）要么是使用教程（diffusers 的文档），中间缺"为什么这样设计、每个选择值多少"的层次。这个系列按算法地图的一贯写法——推导 → 算账 → 公开配方对照——填这一层。


## 适合哪些读者？

### 读完 L4、L5，准备做多模态的算法学习者

主要读者。按顺序读；理解线（一到五）与生成线（六到八）可以只走一条，第九篇需要两条都读过。

### 已经在做 VLM、但对编码器与训练阶段的选择缺乏系统认识的工程师

第一到三篇是为你写的：编码器的对照、connector 与分辨率的取舍、训练阶段与数据的公开配方比较、评测与幻觉。

### 做文生图 / 视频生成的工程师

第六到八篇：从 DDPM 到 flow matching 的推导链、CFG 的含义、DiT 与 latent 空间、SD3 / FLUX 的配方、采样加速、视频的时空 patch。第九篇的自回归生成作为对照。

### Infra 工程师

[04 系列第八篇](/multimodal-vision-encoder-cost-and-image-token-kv.html)已经算过多模态的成本；本系列第二篇讲这些成本背后的设计动机，第八篇讲扩散模型完全不同的成本结构（无自回归 KV、compute-bound、多步）——它决定了扩散模型的服务系统与 LLM 的服务系统为什么长得不一样。


## 系列的整体主线

系列的主线是**两条线、一个交汇点**：

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 240}}}%%
flowchart TB
    subgraph U["理解线：把模态送进 LLM"]
        direction TB
        ENC["`**一 · 视觉编码器**
ViT · CLIP / SigLIP · DINOv2
一张图 → 几百个向量`"]
        ARCH["`**二 · VLM 结构**
connector · 注入方式
动态分辨率 · 视频`"]
        TRAIN["`**三 · VLM 训练**
阶段 · 数据 · 评测 · 幻觉`"]
        AUDIO["`**四 · 语音（上）**
mel 谱 · Whisper
codec · RVQ`"]
        AUDIO2["`**五 · 语音（下）**
语音理解 · TTS
全模态 · 全双工`"]
        ENC --> ARCH --> TRAIN --> AUDIO --> AUDIO2
    end
    subgraph G["生成线：从噪声到图片"]
        direction TB
        DIFF["`**六 · 扩散（上）**
DDPM · ELBO → MSE · DDIM`"]
        DIFF2["`**七 · 扩散（下）**
score · flow matching · CFG`"]
        LDM["`**八 · Latent diffusion 与 DiT**
VAE · DiT · 文生图配方
采样加速 · 视频`"]
        DIFF --> DIFF2 --> LDM
    end
    UNI["`**九 · 自回归生成与统一模型**
VQ · AR 图像生成
理解 + 生成的三种路线`"]
    AUDIO2 --> UNI
    LDM --> UNI

    classDef und fill:#e8f4ff,stroke:#2e6da4,stroke-width:2px,color:#222
    classDef gen fill:#fff7e0,stroke:#c98a00,stroke-width:2px,color:#222
    classDef uni fill:#eaf7ea,stroke:#1e8449,stroke-width:2px,color:#222
    class ENC,ARCH,TRAIN,AUDIO,AUDIO2 und
    class DIFF,DIFF2,LDM gen
    class UNI uni
```

贯穿九篇的三条线索：

- **推导线**：InfoNCE 与对比学习为什么学出语义（第一篇）；connector 的信息瓶颈（第二篇）；codec 的 RVQ（第四篇）；ELBO → 去噪目标（第六篇）；score matching 与 flow matching 的统一、CFG 的贝叶斯推导（第七篇）；VQ-VAE 的离散瓶颈（第九篇）。
- **成本线**：编码器的 FLOPs 与 token 数（回指 04-08）；训练阶段各自的算力；语音 token 与全双工的持续成本（第五篇）；扩散模型每张图的 FLOPs 与 LLM 的对比（第八篇）；AR 图像生成的 token 数与 KV。
- **配方线**：LLaVA → Qwen2.5-VL → InternVL 3 的演化；Whisper → Qwen2.5-Omni；SD 1.x → SDXL → SD3 → FLUX；Chameleon → Janus → BAGEL。每一处设计选择在公开报告里的对照。


## 章节结构与分章导读

### 1. 视觉编码器：CLIP、SigLIP 与自监督 ViT

VLM 的第一步是把图片变成向量序列，做这件事的几乎总是一个预训练好的 ViT。它是怎么训出来的，决定了它"看到"什么。

**核心内容**：ViT 回顾（回指 [L3 第五篇](/cnn-from-lenet-to-resnet-and-vit.html)与 [04-08](/multimodal-vision-encoder-cost-and-image-token-kv.html)）；对比学习——InfoNCE 的推导、它为什么等价于估计互信息的下界、为什么需要大 batch（负样本数）、温度的作用；CLIP 的 4 亿图文对与训练算力账；SigLIP 用 sigmoid 损失把每对的判定解耦、为什么它对 batch 大小不敏感且更省；自监督（DINOv2）学到的与对比学习不同的东西（局部、几何、密集特征），以及为什么 VLM 里 CLIP 类编码器仍是主流但开始混用；编码器的"盲点"——计数、空间关系、文字、细粒度差别——它们的来源（对比学习的目标只要求区分图文对，不要求精细结构）；分辨率与 patch 大小对 token 数与信息量的影响；编码器选型对照（CLIP ViT-L/14、SigLIP-SO400M、InternViT-6B、DINOv2）。

**要回答的问题**：为什么几乎所有 VLM 都用 CLIP / SigLIP 而不用 ImageNet 分类预训练的 ViT？编码器看不到什么？

### 2. VLM 的结构：connector、注入方式与动态分辨率

编码器输出的几百个向量怎么进入 LLM。04-08 算了每种选择的 token 数与 FLOPs；这一篇讲**为什么这样选、效果差在哪**。

**核心内容**：三类 connector——MLP projector（LLaVA：信息全保留、token 数 = patch 数）、pixel-shuffle / 2×2 merge（Qwen2-VL、InternVL：空间压缩 4 倍、信息略损）、Q-Former / Perceiver resampler（BLIP-2、Flamingo、早期 Qwen-VL：固定 token 数、信息瓶颈）——各自的参数量、压缩率与信息损失，以及为什么 2024 年后主流从 Q-Former 回到 MLP + 空间压缩；decoder-only 注入（图片 token 进序列）vs cross-attention 注入（Flamingo、Llama 3.2 Vision：LLM 层间插 cross-attention 层，图片特征不进序列）的取舍——前者简单、复用一切、图片 token 占上下文；后者不占上下文、LLM 文本能力不受影响、但要新增参数与训练；分辨率的三种做法——固定（224 / 336）、AnyRes tile（LLaVA-NeXT、InternVL：切成若干 tile 各编码再拼）、原生动态（Qwen2-VL：ViT 接受任意分辨率，2D RoPE，token 数随图片大小变）——各自的 token 预算与对 OCR / 细节任务的影响；视频——帧采样率、时间维的合并（Qwen2-VL 的 2 帧合并）、M-RoPE 的三维位置、token 预算的分配；多图与交错。

**要回答的问题**：LLaVA 的一个 MLP 与 BLIP-2 的 Q-Former 相差什么？为什么 Qwen2-VL 要让 ViT 接受原生分辨率？

### 3. VLM 的训练：数据、阶段与评测

VLM 的训练不是一步到位的：先让 connector 学会对齐、再让 LLM 学会看图、再用指令数据教它回答问题、最后对齐偏好。每一阶段冻结谁、用什么数据、多少数据，各家不同。

**核心内容**：训练阶段的标准形态（connector 预对齐 → 多模态预训练 / 继续预训练 → 多模态 SFT → 偏好对齐 / RL）与各阶段的冻结策略、数据量级、lr；数据的类型——caption（LAION、CC12M、合成 recaption）、交错图文（MMC4、OBELICS）、OCR 与文档（PDF、图表、网页截图）、grounding（框、点）、视频字幕、指令数据（LLaVA-Instruct 的 GPT-4 合成、ShareGPT4V）——各自教会模型什么；对照 LLaVA-1.5 / LLaVA-NeXT / Qwen2.5-VL / InternVL 2.5 / Molmo / Llama 3.2 Vision / Idefics 的公开配方（阶段数、每阶段数据量、冻结策略、分辨率）；文本能力的保持——多模态训练会不会伤 LLM 的文本能力、怎么用文本数据混合防止；评测——MMMU、MMBench、MMStar、DocVQA、ChartQA、TextVQA、MathVista、Video-MME 各测什么；多模态幻觉——来源（语言先验、编码器盲点、训练数据里的共现偏差）、度量（POPE、HallusionBench）与缓解（数据、RLHF-V 一类的偏好对齐、解码侧的对比）。

**要回答的问题**：为什么先冻结 LLM 只训 connector？多模态幻觉从哪来、怎么减少？

### 4. 语音（上）：从波形到 token——mel 谱、Whisper 与神经 codec

声音进入 LLM 有两条路：连续的音频特征（像图片一样经编码器与 connector）或离散的语音 token（像文本一样）。这一篇讲表示：声音在计算机里是什么、怎么变成模型能读的形式、怎么变成离散 token。

**核心内容**：波形（16 kHz 一秒 16000 个数）；分帧、FFT、mel 滤波器——用 40 行 NumPy 从一段合成语音算出 log-mel 谱并画出来；Whisper 的 encoder-decoder（回指 04-08 的 1500 个位置）；CTC 与 attention 解码的对比；自监督（HuBERT、w2v-BERT）与语义 token；神经 codec——SoundStream / EnCodec 的结构、向量量化就是 K-Means、残差向量量化（RVQ）的推导与一个 2 维手算例子、2000 个向量上 8 级 RVQ 每级误差约减半的实验、训练 RVQ 的四个技巧、语义 token 与声学 token 的分层。

**要回答的问题**：一秒钟的声音在模型眼里是什么？语音为什么比图片更需要离散 token，RVQ 怎么用几个小码本表示高保真音频？

### 5. 语音（下）：语音理解、语音生成与全双工

有了两种表示，这一篇讲怎么用：听、说、边听边说。

**核心内容**：语音理解——音频编码器 + connector + LLM（Qwen2-Audio，VLM 的翻版）与离散 token 路线，音频的 token 预算；语音生成——TTS 的三条路（VALL-E 的 AR 第一码本 + NAR 其余，直接来自 RVQ 的层次；语义→声学两级；F5-TTS 的流匹配）；LLM 直接说与**模态竞争**（Moshi 的内心独白、Qwen2.5-Omni 的 Thinker-Talker，一张三种做法的结构图）；全模态（TMRoPE 对齐音视频时间轴）；全双工——半双工与全双工的时间线对比图、Moshi 的多流与 RQ-Transformer、时延的四段账；评测与成本。

**要回答的问题**：让 LLM 直接生成语音 token 会伤它的文本能力，怎么办？全双工的时延由什么决定？

### 6. 扩散模型（上）：DDPM——加噪、去噪与「预测噪声」

生成线的数学核心，从最容易从零看懂的 DDPM 视角开始。全篇用一个二维的 toy（两个月牙形的点云）把每一步跑出来看。

**核心内容**：前向加噪的一步定义、两步合并的推导（两个数验算）与闭式解 $$x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon$$；toy 上 $$t = 0 \ldots 999$$ 的点云与"1000 步 vs 一步"的数值验算；反向过程——为什么真实的反向算不出、给定 $$x_0$$ 的算得出，后验均值代一组数字（每步只挪 0.6%）；变分下界是什么、逐项读、两个高斯的 KL、化简成噪声预测、$$L_{simple}$$；训练算法四行代码与 toy 的损失曲线（按 $$t$$ 分桶看哪里学到了数据）；采样公式与代码，toy 上从噪声到两个月牙的六帧；三种预测目标；DDIM——非马尔可夫前向、确定性更新、跳步（toy 上 1000 / 50 / 20 / 10 / 5 步）。

**要回答的问题**："扩散模型学的是去噪"——去掉噪声为什么等于学会了生成？训练目标从一个复杂的变分下界怎么变成了一行 MSE？DDIM 为什么能 1000 步跳到 50 步？

### 7. 扩散模型（下）：score matching、flow matching 与 classifier-free guidance

另外两种看同一件事的方式，以及所有文生图模型都依赖的两个技术。

**核心内容**：分数——把上篇训好的模型画成箭头图；Tweedie 公式（toy 验算）；去噪分数匹配等价于噪声预测；SDE 与概率流 ODE；flow matching——直线路径、速度场、toy 上从零训一个；与 DDPM 的换算；轨迹为什么弯（随机配对，toy 直线度 0.49）、reflow 怎么拉直（1.00，一步采样）；1 / 2 / 5 / 20 步下 DDIM vs flow matching vs reflow 的对比图；噪声调度（linear、cosine、零终端 SNR、logit-normal、分辨率平移）；classifier-free guidance——条件 dropout、从贝叶斯到 $$\tilde\epsilon = \epsilon_\emptyset + w(\epsilon_c - \epsilon_\emptyset)$$、toy 上 $$w = 0 / 1 / 2 / 4 / 8$$ 的样本图、它在采样什么分布、副作用与修正。

**要回答的问题**：DDPM、score matching、flow matching 为什么是同一件事？CFG 的 $$w = 7.5$$ 在数学上意味着什么？

### 8. Latent diffusion、DiT 与文生图配方

从数学到一个能用的文生图模型：在哪个空间做扩散、用什么网络、文本怎么注入、怎么采样快。

**核心内容**：像素空间扩散的成本与 latent diffusion 的解法（一张账：像素 vs latent 的数的个数与 DiT 序列长度）；用 PCA 当"VAE"、在 16 维 latent 里跑上篇的 DDPM、生成手写数字（toy）；VAE 把 $$1024^2 \times 3$$ 压到 $$128^2 \times 4$$（或 16 通道），扩散在 latent 上做，4 通道 48 倍、16 通道 12 倍的压缩；VAE 的训练（重建 + KL + 感知 + 对抗）与它的瓶颈（细节、文字）；U-Net（SD 1.x / SDXL）→ DiT（Peebles & Xie 2023：把 latent patch 化送进 Transformer，adaLN 注入时间步与条件）→ MMDiT（SD3：文本与图像 token 在同一个 Transformer 里双流交互）；文本编码器的选择（CLIP 文本塔、T5-XXL、LLM）与它对 prompt 理解的影响；配方对照——SD 1.5 / SDXL / SD3 / FLUX.1 / Imagen / DALL-E 3 的参数量、latent 通道数、预测目标、调度、文本编码器、训练数据与 recaption；扩散模型的成本结构——一张 $$1024^2$$ 图 = 一个 4096 token 的序列前向 × 步数、无 KV cache、compute-bound——与 LLM 的对比，以及它对服务系统的含义；采样加速——步数蒸馏（progressive distillation）、一致性模型（consistency models / LCM）、对抗蒸馏（SDXL-Turbo / ADD）、rectified flow 的直线优势——从 50 步到 1–4 步；视频生成——时空 patch（Sora 的"patch 是视频的 token"）、3D VAE、DiT 的时空 attention、Wan / HunyuanVideo / CogVideoX 的配方；扩散的后训练一瞥（DPO for diffusion、奖励微调）。

**要回答的问题**：为什么在 latent 空间做？DiT 相比 U-Net 赢在哪？一张图的生成成本与一次 LLM 推理怎么比？

### 9. 自回归图像生成与统一模型

两条线的交汇。图片能不能像文本一样被 token 化、然后自回归地生成？理解与生成能不能用一个模型？

**核心内容**：用 K-Means 码本把手写数字变成 16 个 token、再用一个计数版的 next-token 模型生成数字（toy）；VQ-VAE / VQGAN——把图片编码成离散 token 网格的推导（最近邻码本、commitment loss、STE 传梯度）、码本坍缩与对策（EMA 更新、码本重置、FSQ / LFQ 的无码本量化）、重建质量与 token 数的权衡；AR 图像生成——Parti、LlamaGen 用 LLM 结构逐 token 生成图像 token 的效果与成本（一张 $$256^2$$ 图 256–1024 个 token，与扩散的对比），栅格顺序的问题；VAR（next-scale prediction）——从粗到细逐尺度生成，把顺序问题变成尺度问题，效果与速度都超过栅格 AR；MaskGIT 一类的并行解码；统一模型的三条路线——纯 token（Chameleon、Emu3：一个词表同时有文本与图像 token，一个 Transformer 全做）、双编码器（Janus：理解用 SigLIP 特征、生成用 VQ token，共享 LLM）、AR + 扩散混合（Transfusion、BAGEL：文本用 AR、图像用扩散 loss，同一个 Transformer）——各自的取舍、公开结果与 GPT-4o 原生图像生成的启示；两条线的成本对照；系列总结。

**要回答的问题**：AR 生成图像与扩散各赢在哪？理解与生成的表示能不能共享？

### 10. 系列总结与通关自测

最后一篇不讲新内容：把九篇正文压成一张「问题 → 结论 → 必记数字」的表并逐篇回顾，拎出贯穿全系列的几条线与常见误区，然后给一套三段式通关自测——十道判断与计算、五道跨篇综合、若干道面试题，答案各自折叠，附「读过 / 掌握 / 能教人」的判据。各篇末尾的自测检验的是一篇读懂了没有，这一篇检验的是九篇能不能连起来用；读完正文再做。

## 贯穿全系列的实践线

每篇正文都有一组**能在 CPU 上几秒到一分钟跑完的 toy 实验**，文中的核心代码、数字与图全部由它们产生，放在 [`ai-learning-labs/multimodal/`](https://github.com/arganzheng/ai-learning-labs/tree/main/multimodal)（NumPy / scikit-learn / PyTorch CPU，不下载任何模型）：

| 篇 | 脚本 | 跑通什么 |
|---|---|---|
| 一 | `01_vision_encoders_and_contrastive.py` | 8×8 图切 16 个 patch；3×3 相似度矩阵手算 InfoNCE；30 行 PyTorch 训一个 toy CLIP；温度；sigmoid 损失 |
| 二 | `02_connectors_and_resolution.py` | MLP / 2×2 merge / 池化 / resampler 四种 connector 的形状与信息损失；三种分辨率策略的 token 数 |
| 三 | `03_vlm_training_toys.py` | 冻结 LLM vs 一起训 vs 两阶段（文本能力保住了没）；共现偏差 → 幻觉的两特征模型 |
| 四 | `04_audio_mel_and_rvq.py` | 波形 → log-mel 谱（手写 STFT 与 mel 滤波器）；VQ 与 8 级 RVQ |
| 五 | `05_duplex_timeline.py` | 半双工 vs 全双工的时间线示意图 |
| 六 | `06_ddpm_toy.py` | 二维两个月牙上从零训 DDPM：加噪、闭式验算、训练、1000 步采样、DDIM 跳步 |
| 七 | `07_flow_score_cfg_toy.py` | 分数场箭头图；flow matching 与 reflow；1 / 2 / 5 / 20 步对比；CFG 扫 $$w$$ |
| 八 | `08_latent_diffusion_toy.py` | PCA 当 VAE，在 16 维 latent 里跑 DDPM 生成手写数字 |
| 九 | `09_vq_tokenizer_and_ar_toy.py` | K-Means 码本把数字变成 16 个 token；FSQ；计数版 next-token 模型生成数字 |

真实模型上的复现（需要一张 24 GB 的 GPU）在每篇的"动手（建议）"一节：用现成工具（`transformers`、`open_clip`、`diffusers`、`lmms-eval`、`encodec`）复现该篇核心现象的骨架与该看的指标，不引用未跑过的数字。

## 前置要求与说明

### 前置要求

- [L3 第五篇](/cnn-from-lenet-to-resnet-and-vit.html)：ViT 的结构与 patch embedding。
- [04 系列](/transformer-and-llm-for-infra-engineers.html)第一、三、八篇：Transformer 结构、KV cache、多模态成本。
- [L5](/post-training-from-sft-to-verifiable-rewards.html)第一、二、四篇：SFT、偏好数据、DPO——第三篇的多模态对齐直接用它们。
- [L2 经典机器学习](/classical-machine-learning-in-the-llm-era.html)：K-Means（第四、九篇的码本就是它）、PCA（第八篇的 toy VAE）、逻辑回归与 softmax（第一篇的对比损失）。
- 概率的基本概念（高斯分布、条件概率、期望）：第六、七篇会在用到的地方原地解释，[L0 数学系列](/math-for-ai-algorithm-engineers.html)是更系统的补充。

### 版本与基线

- 理解线的对照模型：LLaVA-1.5 / NeXT、Qwen2-VL / Qwen2.5-VL、InternVL 2.5 / 3、Llama 3.2 Vision、Molmo、Gemma 3、Kimi-VL 的技术报告。
- 语音：Whisper、EnCodec、VALL-E、Qwen2-Audio、Qwen2.5-Omni、Moshi 的论文与报告。
- 生成线：DDPM、DDIM、Score SDE、Flow Matching / Rectified Flow、DiT、SD 1.x / SDXL / SD3、FLUX.1、Imagen、Sora 技术报告、Wan 2.1、HunyuanVideo、CogVideoX。
- 交汇：VQGAN、MaskGIT、Parti、LlamaGen、VAR、Chameleon、Emu3、Janus / Janus-Pro、Transfusion、BAGEL。
- 工具版本变化快，"动手"节只给接口形状。


## 章节目录

1. [视觉编码器：CLIP、SigLIP 与自监督 ViT](/vision-encoders-clip-siglip-and-self-supervised-vit.html)
2. [VLM 的结构：connector、注入方式与动态分辨率](/vlm-architecture-connectors-injection-and-dynamic-resolution.html)
3. [VLM 的训练：数据、阶段与评测](/vlm-training-recipe-data-stages-and-evaluation.html)
4. [语音（上）：从波形到 token——mel 谱、Whisper 与神经 codec](/speech-and-omni-models-audio-encoders-codecs-and-duplex.html)
5. [语音（下）：语音理解、语音生成与全双工](/speech-understanding-generation-and-full-duplex.html)
6. [扩散模型（上）：DDPM——加噪、去噪与「预测噪声」](/diffusion-models-ddpm-score-matching-and-flow-matching.html)
7. [扩散模型（下）：score matching、flow matching 与 classifier-free guidance](/score-matching-flow-matching-and-classifier-free-guidance.html)
8. [Latent diffusion、DiT 与文生图配方](/latent-diffusion-dit-and-text-to-image-recipes.html)
9. [自回归图像生成与统一模型](/autoregressive-image-generation-and-unified-models.html)
10. [系列总结与通关自测](/multimodal-series-recap-and-self-test.html)


## 最终目标

读完这个系列，面对一个多模态的需求，读者应该能够回答：

| 追问 | 答案来自 |
|---|---|
| 编码器选 CLIP、SigLIP 还是 InternViT？它看不到什么？ | 第一篇 |
| 一张图占多少 token 是合适的？用 MLP 还是压缩？固定分辨率还是动态？ | 第二篇 · 04-08 |
| 训练分几个阶段、每阶段冻结谁、用什么数据？怎么防止文本能力退化？ | 第三篇 |
| 模型描述了图里没有的东西，问题在数据、编码器还是解码？ | 第三篇 |
| 语音要不要离散化？codec 的比特率与码本层数怎么选？ | 第四篇 |
| 语音助手要不要让 LLM 直接说？全双工的时延瓶颈在哪？ | 第五篇 |
| 这个扩散模型的调度、预测目标、采样步数是什么意思？ | 第六篇 |
| 它用的是 $$\epsilon$$ 预测还是 flow matching？guidance 该设多少？ | 第七篇 |
| 生成一张图的成本与一次 LLM 推理怎么比？步数怎么从 50 降到 4？ | 第八篇 |
| 要一个既能看图又能画图的模型，选纯 token、双编码器还是 AR + 扩散？ | 第九篇 |

多模态的模型每年都在换结构。不变的是两条线的数学、"一张图值多少 token"这个账、以及"模型看到了什么、没看到什么"这个问题。
