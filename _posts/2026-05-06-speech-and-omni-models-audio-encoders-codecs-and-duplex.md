---
layout: post
series: multimodal
title: "多模态（04）：语音与全模态：音频编码器、codec 与全双工"
subtitle: "Speech and Omni Models: Audio Encoders, Neural Codecs and Full-Duplex Dialogue"
tags: [AI, Multimodal, Speech, Audio]
catalog: true
---

声音进入 LLM 的路与图片相似又不同。相似的是范式——一个编码器把音频变成向量序列，一个 connector 对齐到 LLM，然后像文本一样处理；Whisper 的 encoder 之于语音，就像 CLIP ViT 之于图片。不同的是**语音要双向**：VLM 几乎只做理解（看图说话），语音模型从一开始就要既听又说——语音助手要用语音回答。生成侧要求把 LLM 的输出变回波形，而波形是每秒 16000–48000 个采样点的连续信号，不能像文本一样逐 token 生成。

这就是语音比图片更需要**离散 token** 的原因：神经 codec 把一秒音频压成几十到几百个离散 token，让 LLM 可以用生成文本的方式生成语音。有了离散 token，理解与生成可以统一在一个自回归模型里；没有它，生成要靠一个独立的 TTS。2024–2025 年的全模态模型（GPT-4o、Qwen2.5-Omni、Moshi）都建立在这个基础上，并进一步要求**实时**——边听边说的全双工，时延几百毫秒。

这一篇也是理解线到生成线的过渡：离散 token 与自回归生成，正是第七篇自回归图像生成的思路。

本篇要回答的核心问题是：

> **语音为什么比图片更需要离散 token？全双工的时延由什么决定？**


## 一、总览：两条路、三个层次

### 1. 声音进入 LLM 的两条路

| 路 | 表示 | 理解 | 生成 | 代表 |
|---|---|---|---|---|
| **连续特征** | 音频编码器（Whisper encoder 一类）的隐状态，经 connector 进 LLM | 好——保留全部声学细节 | 不能直接生成；需外接 TTS | Qwen-Audio / Qwen2-Audio、SALMONN、LLaMA-Omni（理解侧） |
| **离散 token** | 神经 codec（EnCodec 一类）的量化码，或语义 token（HuBERT 聚类） | 可以，但有信息损失 | 可以——LLM 自回归生成 token，codec 解码成波形 | AudioLM、VALL-E、SpeechGPT、Moshi、GPT-4o（推测） |
| **混合** | 理解用连续特征，生成用离散 token | 好 | 好 | Qwen2.5-Omni、Mini-Omni、GLM-4-Voice |

VLM 几乎全在第一行（理解用连续特征），因为它们不生成图片。语音模型要生成，所以必须有离散 token（第二、三行）——这是核心问题前半的答案，第三章展开。

### 2. 三个层次的 token

音频 token 分三个层次，对应压缩的三个粒度：

| 层次 | 来源 | 每秒 token 数 | 保留什么 | 丢什么 |
|---|---|---|---|---|
| 声学 token（acoustic） | codec 的 RVQ 码（EnCodec：75 帧/秒 × 8 码本） | 几百 | 音色、韵律、噪声——可以高保真重建 | 无（有损压缩） |
| 语义 token（semantic） | 自监督模型（HuBERT / w2v-BERT）的隐状态聚类 | 25–50 | 音素级内容、部分韵律 | 音色、细节——不能重建高保真波形 |
| 文本 token | ASR | 2–4（词） | 内容 | 一切非语言信息 |

一段一分钟的语音：声学 token 约 3.6 万（EnCodec 8 码本）、语义 token 约 1500–3000、文本 200 词。语音 LLM 要在这三层之间选择或组合：AudioLM 的做法是先生成语义 token 再生成声学 token（内容与声音分开）；Moshi 的做法是把语义与声学 token 一起以多流的方式生成。

### 3. 先说答案

全双工的时延由**四段**相加决定：（1）音频分帧的粒度——codec 帧 80 ms（Moshi 12.5 Hz）或 Whisper 的 20–40 ms，模型至少要等一帧到齐；（2）编码器与 LLM 的**首 token 延迟**——听到的音频要过编码器、进 LLM、生成第一个语音 token；（3）codec **解码**的延迟——语音 token 变波形，流式 codec 可以逐帧解码；（4）**语义决策**的延迟——模型判断"该说了"需要的上下文——这一项不是计算，是模型的能力。Moshi 报告的理论时延 160 ms（两帧 80 ms）、实测约 200 ms；GPT-4o 报告平均 320 ms；人类对话的换手间隔约 200 ms。达到这个数字要求模型**持续地同时处理输入流与输出流**——不是"听完再说"，而是每一帧都在决定要不要说、说什么——这是 Moshi 的多流结构与 Qwen2.5-Omni 的 Thinker-Talker 分离要解决的问题。第六章展开。

### 4. 本文的章节安排

| 章 | 主题 | 内容 |
|---|---|---|
| 二 | 音频的表示与编码器 | 波形、mel 谱；Whisper 的 encoder-decoder；CTC 与 attention；自监督（HuBERT、w2v-BERT） |
| 三 | 神经 codec 与 RVQ | SoundStream / EnCodec 的结构；RVQ 的推导；为什么几个码本能高保真；语义 token vs 声学 token |
| 四 | 语音理解 | 编码器 + connector + LLM（Qwen2-Audio）；离散 token 路线；音频的 token 数与成本 |
| 五 | 语音生成 | TTS 的三条路：AR 声学 token（VALL-E）、AR + NAR、流匹配（F5-TTS、CosyVoice）；LLM 直接说 |
| 六 | 全模态与全双工 | Qwen2.5-Omni 的 Thinker-Talker 与 TMRoPE；Moshi 的多流与内心独白；时延的四段账 |
| 七 | 评测 | ASR 的 WER、TTS 的 MOS / WER / 说话人相似度、语音对话的评测 |
| 八 | 成本 | 音频 token 的账；全双工的算力 |
| 九 | 动手（建议） | EnCodec 的码本层数 |
| 十 | 本文小结 | |


## 二、音频的表示与编码器

### 1. 从波形到 mel 谱

语音波形是 16 kHz 采样的一维信号，一秒 16000 个数。直接在波形上建模太长；标准做法是**短时傅里叶变换**：25 ms 的窗（400 点）、10 ms 的步长，每帧做 FFT 得到频谱，取幅度、按 mel 刻度（模仿人耳的频率感知，低频密高频疏）合并成 80 或 128 个频带，取对数——得到 $$100 \times 80$$ 每秒的 log-mel 谱图。一秒语音 = 100 帧 × 80 维。它是所有语音模型的输入（除了直接在波形上工作的 codec 编码器）。

### 2. Whisper

Whisper（Radford 等 2022）是语音的"CLIP"——一个在 68 万小时（v3 用 500 万小时含伪标签）多语言、多任务数据上训的 encoder-decoder Transformer，任务是转写与翻译。encoder 输入 30 秒的 log-mel（3000 帧 × 80），过两层步长 2 的卷积（→ 1500 帧，20 ms 一帧）加正弦位置编码，32 层 Transformer（large：1.55B）；decoder 是标准的自回归文本 decoder，输出转写。

它的 encoder 输出（1500 × 1280）是语音理解任务最常用的特征——就像 CLIP ViT 的 patch 特征。[04-08](/multimodal-vision-encoder-cost-and-image-token-kv.html)第七章算过它的成本：30 秒固定 1500 个位置（不足 30 秒 pad），是 VLM 里 576 个 patch 的 2.6 倍。Qwen2-Audio 用 Whisper-large-v3 的 encoder，再 pool 到 25 Hz（每秒 25 个特征，30 秒 750 个），进 LLM。

### 3. CTC 与 attention 解码

ASR 的两种解码：**CTC**（Connectionist Temporal Classification）让 encoder 的每一帧独立输出一个字符或"空白"，去重去空白得到转写——不需要 decoder、并行、流式友好，但假设帧间条件独立，语言模型能力弱；**attention decoder**（Whisper）自回归地生成文本，每步 attend 到全部 encoder 输出——准确但非流式（要等 encoder 看完一整段）。工业 ASR 常用 RNN-T（Transducer）作为两者的折中——流式且有隐式语言模型。语音 LLM 里 Whisper encoder 的 attention 特征被 LLM 消费，LLM 本身就是 decoder。

### 4. 自监督编码器：HuBERT 与 w2v-BERT

对应视觉的 DINOv2：不用标签，在海量无标注语音上学表示。wav2vec 2.0 用对比目标（预测被 mask 的帧的量化表示）；HuBERT 用 k-means 聚类得到的伪标签做 masked prediction，迭代地用自己的隐状态重新聚类；w2v-BERT 结合两者。它们的隐状态（尤其中间层）编码**音素级内容**——对隐状态做 k-means（500–2000 类）得到的离散码就是**语义 token**，AudioLM、SpeechGPT 用它做内容表示。它们不编码音色（说话人信息在 k-means 中被丢掉）——这是"语义"的含义。


## 三、神经 codec 与 RVQ

### 1. 为什么需要 codec

要让 LLM **生成**语音，需要一个离散的、序列不太长的、能重建波形的表示。mel 谱是连续的且每秒 100 帧；波形每秒 16000 个点；语义 token 离散但不能重建音色。神经 codec 填这个空：把波形压成每秒几十帧、每帧几个离散码，且能高保真解码回波形。

### 2. SoundStream / EnCodec 的结构

SoundStream（Zeghidour 等 2021）与 EnCodec（Défossez 等 2022）的结构相同：

- **编码器**：一维卷积网络，对波形逐级下采样（EnCodec 24 kHz：步长 2, 4, 5, 8 共 320 倍 → 75 帧/秒），每帧一个 128 维的连续向量。
- **量化器**：残差向量量化（RVQ，下一节），把每帧的 128 维向量变成 $$N_q$$ 个离散码（$$N_q = 8$$，每个码本 1024 项 = 10 bit）。
- **解码器**：卷积网络镜像上采样，从量化后的向量重建波形。
- **训练**：重建损失（波形 L1 + 多尺度 mel 谱损失）+ 对抗损失（多个判别器判断真假）+ 感知特征匹配 + RVQ 的 commitment 损失。

比特率：75 帧/秒 × 8 码本 × 10 bit = 6 kbps——比 MP3 的 128 kbps 低 20 倍，主观质量接近。

### 3. RVQ 的推导

向量量化（VQ）：一个码本 $$\{e_1, \ldots, e_K\} \subset \mathbb{R}^d$$，把向量 $$z$$ 映射到最近的码字 $$\hat z = e_{k^*}$$，$$k^* = \arg\min_k \lVert z - e_k \rVert$$。$$K = 1024$$ 时每帧 10 bit，但一个 128 维的向量用 1024 个码字表示误差太大——要高保真需要 $$K$$ 巨大（$$2^{80}$$ 一类），不可能。

**残差**向量量化把量化分成多级：

$$
r_0 = z, \quad k_i = \arg\min_k \lVert r_{i-1} - e^{(i)}_k \rVert, \quad r_i = r_{i-1} - e^{(i)}_{k_i}, \quad i = 1, \ldots, N_q
$$

$$
\hat z = \sum_{i=1}^{N_q} e^{(i)}_{k_i}
$$

第一个码本量化 $$z$$ 本身，第二个码本量化第一级的**残差**，第三个量化第二级的残差……每级一个独立的码本（$$K = 1024$$）。$$N_q = 8$$ 级的总表示能力是 $$1024^8 = 2^{80}$$ 个不同的 $$\hat z$$——用 8 个小码本得到一个巨大的等效码本，且每级的搜索只是 1024 次比较。

为什么它有效：每一级的残差 $$r_i$$ 的范数递减（每级去掉了一部分），后面的码本在越来越小的尺度上精细化。这天然形成**由粗到细的层次**：第 1 个码本编码最主要的信息（内容与大致音色），后面的码本编码细节（高频、微弱的噪声）。只用前 $$n$$ 个码本解码，得到的是一个低比特率、质量较差但内容完整的重建——EnCodec 用同一个模型支持 1.5 / 3 / 6 / 12 kbps（2 / 4 / 8 / 16 码本）就是靠训练时随机丢弃后面的码本（quantizer dropout）。

训练 RVQ 时的问题与 VQ-VAE 相同（第七篇详述）：$$\arg\min$$ 不可微，用 STE（[L6 第四篇](/quantization-aware-training-low-bit-and-evaluating-quantized-models.html)）把梯度直通到编码器；码本用 EMA 更新（而不是梯度）；加 commitment 损失 $$\lVert z - \text{sg}(\hat z) \rVert^2$$ 让编码器输出靠近码字；用 k-means 初始化码本并对长期未用的码字随机重置，防止码本坍缩（只有少数码字被用到）。

### 4. 声学 token 的层次结构对生成的意义

RVQ 的层次结构决定了语音生成的形态：第 1 个码本的 token 序列（75/秒）承载内容，可以由一个自回归模型精确地生成；后面 7 个码本是"细化"，彼此依赖弱、可以**并行**生成——这正是 VALL-E 的 AR + NAR 设计（第五章）。以及 AudioLM 的三阶段：语义 token → 粗声学 token（前几个码本）→ 细声学 token。

### 5. 语义 token 与声学 token 的分工

声学 token 保留一切但序列长（75 × 8 = 600/秒）且"内容"埋在细节里，LLM 直接在它上面建模语言内容效率低。语义 token 短（25–50/秒）、内容清晰，但不能重建。所以很多系统**两者都用**：LLM 在语义 token（或文本 + 语义 token）上做内容建模，再用一个较小的模型从语义 token 生成声学 token 并解码。近年的 codec（SpeechTokenizer、Mimi）试图把两者合一：让 RVQ 的**第一个码本**被蒸馏成语义 token（用 WavLM / HuBERT 的表示做目标），后面的码本承载声学细节——一个 codec 同时给出语义与声学，Moshi 用的 Mimi 就是这样（12.5 Hz、8 码本、1.1 kbps，第一码本语义蒸馏）。


## 四、语音理解

### 1. 编码器 + connector + LLM

Qwen2-Audio（2024）的结构就是 VLM 的翻版：Whisper-large-v3 encoder → 平均池化到 25 Hz → 线性投影 → Qwen-7B。训练三阶段（预训练对齐、多任务 SFT、DPO），数据覆盖语音识别、翻译、音频描述、声音事件、音乐、语音对话；能力包括 ASR、翻译、音频问答、语音指令跟随（用户说话、模型文本回答）。SALMONN 用双编码器（Whisper 管语音 + BEATs 管一般音频）。这条路的优点与 VLM 相同：LLM 零改动、理解质量高（连续特征保留全部信息）。缺点：**不能说**——输出是文本，要说话需外接 TTS，时延叠加。

### 2. 离散 token 路线

把语音 token 化后当文本处理：SpeechGPT 用 HuBERT 语义 token 扩展 LLM 词表，语音与文本在同一个自回归模型里，能听能说（说的是语义 token，再用一个 vocoder 变波形——音色由 vocoder 决定，不可控）。这条路让理解与生成统一，但理解质量受 token 信息损失的限制（语义 token 丢了副语言信息——语气、情绪、说话人），且离散 token 序列长（一分钟 3000 个）。

### 3. 音频的 token 预算

对比图片：一张图 256–1000 个 token；一分钟语音在 Qwen2-Audio 里 25 Hz × 60 = 1500 个 token，在 Whisper 原生 50 Hz 下 3000，在 EnCodec 声学 token 下 3.6 万（8 码本）。**语音的 token 数量级比图片大**，且随时长线性增长——一小时的会议录音是 9 万个连续特征 token，超过多数 LLM 的上下文。这是语音模型比 VLM 更需要压缩（更低的帧率：Mimi 的 12.5 Hz、Qwen2.5-Omni 的 25 Hz→ 进一步合并）与流式处理（不把整段放进上下文）的原因。


## 五、语音生成

### 1. TTS 的三条路

| 路线 | 做法 | 代表 | 特点 |
|---|---|---|---|
| AR 声学 token | 文本 → 自回归生成 codec 第一码本 token → NAR 生成其余码本 → codec 解码 | VALL-E（2023）、VALL-E 2 | 零样本音色克隆（3 秒 prompt）；AR 的不稳定（重复、跳词） |
| 语义 → 声学 两级 | 文本 → 语义 token（AR）→ 声学 token（NAR 或流匹配）→ 波形 | CosyVoice、SpeechGPT-Gen、AudioLM 式 | 内容与音色解耦；语义 token 的 AR 更稳 |
| 非自回归 / 流匹配 | 文本 + 参考音频 → 直接用扩散 / flow matching 生成 mel 谱或 codec latent → vocoder | F5-TTS、E2 TTS、Voicebox、NaturalSpeech 3 | 快（几十步并行）、稳；需要文本—音频对齐（或 filler token 技巧） |

**VALL-E** 的设计直接来自 RVQ 的层次：第一个码本的 token 用自回归 Transformer 生成（以文本音素与 3 秒 prompt 的声学 token 为条件），这一步决定内容与大致韵律；其余 7 个码本用一个非自回归 Transformer 一次性生成（每个码本一层，以前面所有码本为条件）——细节的并行填充。它把 TTS 变成了"语言建模"，用 6 万小时数据训出的零样本克隆能力震动了领域。它的问题也是语言模型的问题：AR 采样的不稳定（重复、丢字、幻觉式的多说），VALL-E 2 用重复感知采样与分组建模缓解。

**流匹配 TTS**（F5-TTS 2024）走第六篇要讲的扩散 / 流路线：把文本（字符序列 pad 到目标长度）与参考音频的 mel 拼起来，用 DiT 在噪声与目标 mel 之间学一个速度场，32 步 ODE 采样得到 mel，再用 vocoder（Vocos）变波形。不需要音素、不需要对齐器、不需要时长模型；非自回归所以没有 AR 的不稳定；速度快（RTF 0.15）。2024–2025 年开源 TTS 的主流转向这条路或两级路（CosyVoice 2 用 AR 语义 token + 流匹配声学）。

### 2. LLM 直接说

语音 LLM 的"说"有两种实现：（a）LLM 生成文本，外接 TTS——简单，但时延 = LLM 首 token + TTS 首帧，且 TTS 拿不到 LLM 的"语气意图"；（b）LLM 直接生成语音 token（声学或语义）——Moshi、GLM-4-Voice、Mini-Omni 的做法——时延低、语气可控，但 LLM 要同时学两种输出，且语音 token 序列长导致文本推理能力下降（**模态间的能力竞争**：Moshi 报告纯语音训练的模型在文本知识任务上明显弱于其文本底座，靠"内心独白"——先生成文本 token 再生成对应的语音 token——缓解）。

Qwen2.5-Omni 的 **Thinker-Talker** 是 (a) 与 (b) 的折中：Thinker 是一个完整的多模态 LLM（听、看、想、输出文本），Talker 是一个较小的双轨自回归模型，接收 Thinker 的**隐状态与文本 token**流式地生成语音 codec token；两者端到端联合训练。Talker 拿到的不只是文本还有 Thinker 的隐状态（语气与语义意图），且流式（Thinker 每出几个 token Talker 就开始说），而 Thinker 的文本能力不受语音 token 的干扰。


## 六、全模态与全双工

### 1. 全模态：一个模型听、看、说

Qwen2.5-Omni（2025）：输入文本、图片、音频、视频（带音轨），输出文本与语音。结构上是 Qwen2.5-VL 的视觉路径 + Qwen2-Audio 的音频路径 + Thinker-Talker 的语音输出。一个新问题是**音视频的时间对齐**：视频帧的 token 与音频的 token 要按真实时间交错，模型才知道"这句话是在这一帧说的"。TMRoPE（Time-aligned Multimodal RoPE）扩展 M-RoPE：时间轴的位置 id 以 40 ms 为单位与绝对时间对齐，音频帧与视频帧按时间戳插入同一序列（每 2 秒一块，块内先视频后音频），三个模态共享时间轴。

GPT-4o（2024）是这条线的先行者——一个端到端模型处理文本、音频、图像的输入与输出，平均语音时延 320 ms——但没有公开结构；从行为推测它在音频上用了离散 token 与端到端的生成。

### 2. 全双工：边听边说

半双工（走对讲机）：用户说完 → 模型听完 → 模型回答 → 用户等回答完再说。全双工（打电话）：双方随时可以说，模型能被打断、能插话（"嗯""对"）、能在用户停顿时接话。

Moshi（Kyutai 2024）是第一个开源的全双工语音模型，它的结构回答了"怎么同时听与说"：

- **多流**：模型的每一步同时处理三条流——用户的音频 token（Mimi 8 码本）、模型自己的音频 token（8 码本）、模型的文本 token（内心独白）。每个时间步（80 ms）三条流各前进一帧。用户说话时模型的输出流是"静音"token，模型说话时用户流是背景噪声或静音——但两条流**始终都在**，模型每一帧都在决定自己这一帧输出什么。
- **RQ-Transformer**：时间维用一个大的 Transformer（7B，Helium）建模帧序列；每帧内的 8 + 8 + 1 个 token 用一个小的 depth Transformer 自回归生成——把"每帧 17 个 token"的深度维从主模型里拆出来，主模型每 80 ms 只跑一步。
- **内心独白**：在生成音频 token 之前先生成对应的文本 token（时间对齐、略微提前），文本作为语音内容的"脚本"，让语音的语言质量接近文本模型；反过来延后文本也可以做流式 ASR。
- **时延**：帧 80 ms，理论最小时延两帧 160 ms（一帧输入到齐 + 一帧输出），实测约 200 ms。

### 3. 时延的四段账

回到核心问题后半。一次"用户说完 → 模型开始出声"的时延：

| 段 | 内容 | 量级 | 由什么决定 |
|---|---|---|---|
| 分帧 | 等待当前帧的音频到齐 | codec 帧长：Mimi 80 ms、EnCodec 13 ms、Whisper 特征 20 ms（但 Whisper 要 30 s 窗——非流式） | codec 的帧率；流式编码器的窗 |
| 首 token | 编码器 + LLM 前向 → 第一个输出 token | LLM 一步 decode 20–50 ms（7B）；Thinker-Talker 里 Thinker 出几个文本 token 后 Talker 才开始 | 模型大小、硬件、结构 |
| 解码 | 语音 token → 波形 | 流式 codec 解码器逐帧几 ms；非流式 vocoder 要等一段 | codec 解码器是否因果 / 流式 |
| 语义决策 | 模型判断"用户说完了 / 该我说了" | 人类约 200 ms；模型依赖对停顿与语义完成度的判断 | 训练数据里的对话节奏；这不是计算延迟 |

Moshi 的 160–200 ms 里：80 ms 分帧 + 一步 7B decode（约 40 ms，帧内 depth Transformer 另加几 ms）+ Mimi 流式解码（几 ms）+ 决策（模型每帧都在决策，没有额外等待）。半双工系统（ASR → LLM → TTS 串联）的典型时延是 1–3 秒：ASR 要等静音检测（VAD）判断说完（300–700 ms）+ LLM 首 token（几百 ms，含 prefill）+ TTS 首帧（几百 ms）。**全双工把三段串联变成一个模型的一步**，且去掉了 VAD 的等待——这是 10 倍时延差的来源。

代价：全双工模型要**持续运行**——用户不说话时它也每 80 ms 跑一步（输出静音 token），一小时对话是 4.5 万步 decode；半双工只在有输入时算。且全双工的训练数据（真实的、有打断与重叠的双流对话）极少，Moshi 用合成对话（两个 TTS 声音按脚本对话）与真实数据混合。


## 七、评测

### 1. 理解侧

- **ASR**：WER（词错误率）在 LibriSpeech（clean / other）、Common Voice、多语言的 FLEURS 上。Whisper-large-v3 在 LibriSpeech clean 约 2%。
- **音频理解**：AIR-Bench（音频指令跟随）、MMAU（音频多选推理）、声音事件分类（AudioSet）、音乐（MusicCaps）。
- **语音对话**：VoiceBench——把文本指令 benchmark（AlpacaEval、IFEval、常识问答）用 TTS 转成语音输入，评模型的文本或语音回答；它暴露了语音输入下的能力退化（同一模型语音输入比文本输入低 5–15 个点是常态）。

### 2. 生成侧

- **可懂度**：把生成的语音用 ASR 转写，与目标文本比 WER。
- **自然度**：MOS（人工 1–5 分）；自动代理 UTMOS / DNSMOS。
- **说话人相似度**：用说话人验证模型（WavLM-TDNN）算生成音频与参考音频的 embedding 余弦相似度。
- **韵律 / 情感**：较难自动评，多靠 MOS 的细分。

### 3. 全双工

- **时延**：从用户结束到模型开始出声的分布（不是平均——尾部决定体验）。
- **打断处理**：用户插话时模型是否停下、多快停下。
- **对话流畅度**：合成的双流测试集上的换手成功率、重叠率、不当沉默。

标准化程度低，是 2025 年评测方法论的空白之一。


## 八、成本

### 1. 音频 token 的账

以 Qwen2-Audio 的 25 Hz 连续特征为例，一分钟音频 1500 个 token 进 7B LLM：prefill $$2 \times 7B \times 1500 = 21$$ TFLOPs，KV 1500 × 128 KB（Llama-7B 结构）= 188 MB。一小时录音 9 万 token——超过多数上下文；实际系统按段处理（30 秒一段）或用更低帧率。

离散声学 token 的 LLM 建模：Moshi 每帧 17 个 token（8 + 8 + 1），主模型每 80 ms 一步（12.5 步/秒），一小时 4.5 万步——但每步是 7B 的一次 decode（约 40 ms），持续占一张卡的一部分。深度 Transformer 每帧 16 步（小模型，几 ms）。

### 2. 全双工的持续成本

半双工系统在无输入时空闲；全双工模型每 80 ms 跑一步不管有没有人说话。一路对话持续占用约 1/2 张 H100（7B 模型 40 ms/步 in 80 ms）——比半双工贵一个量级，且不能像文本服务那样靠 batch 摊薄（每路对话的节拍是实时的，batch 只能在多路之间做）。这是全双工产品定价高的算力基础。


## 九、动手（建议）

- **RVQ 的层次**：用 `encodec`（或 `transformers` 的 EncodecModel）对一段 10 秒语音编码，分别用 2 / 4 / 8 个码本解码（对应 1.5 / 3 / 6 kbps），听三段重建，并用一个 ASR（Whisper-small）转写测 WER、用 UTMOS 打分。看第一个码本承载了多少内容、后面的码本改善了什么。
- **语义 vs 声学**：用 HuBERT-large 的第 9 层特征做 k-means（500 类）得到语义 token，用一个开源的 unit vocoder 解码，与 EnCodec 重建对比——语义 token 的重建应该内容对但音色变了。
- **理解退化**：用 VoiceBench 的一个子集（比如 IFEval 的 100 条），文本输入与 TTS 语音输入分别喂 Qwen2-Audio-7B-Instruct，比准确率。
- **时延**：本地跑 Moshi（开源，需要 24 GB 卡）测端到端时延分布；与一个 Whisper → LLM → F5-TTS 的串联管线比。

该看的：2 码本的 WER 是否已接近 8 码本（内容在第一码本）而 MOS 差很多；语义 token 重建的说话人相似度是否低；语音输入比文本输入掉几个点；串联管线的时延是否在秒级而 Moshi 在 200–300 ms。不引用任何未跑过的数字。


## 十、本文小结

| 项 | 规则 | 备注 |
|---|---|---|
| 表示 | 16 kHz 波形 → log-mel（100 帧/秒 × 80）→ 编码器 | Whisper encoder 1500 位置 / 30 秒 |
| 两条路 | 连续特征（理解好、不能说）；离散 token（能说、理解有损）；混合 | 语音要双向 → 需要离散 token |
| 三层 token | 声学（几百/秒，可重建）；语义（25–50/秒，内容）；文本（2–4/秒） | 一分钟：3.6 万 / 3000 / 200 |
| RVQ | $$r_i = r_{i-1} - e^{(i)}_{k_i}$$，$$N_q$$ 个 1024 码本 ≈ $$2^{10 N_q}$$ 等效码本 | 由粗到细；quantizer dropout 支持多比特率；STE + EMA + 重置防坍缩 |
| 语义蒸馏 | 第一码本蒸馏 HuBERT / WavLM（SpeechTokenizer、Mimi） | 一个 codec 同时给语义与声学 |
| 理解 | Whisper encoder → pool 25 Hz → 投影 → LLM（Qwen2-Audio） | 一分钟 1500 token；语音输入比文本掉 5–15 点 |
| TTS | AR 声学（VALL-E：AR 第一码本 + NAR 其余）；语义→声学两级；流匹配（F5-TTS，32 步 DiT） | 主流转向流匹配与两级 |
| 直接说 | LLM 生成语音 token；模态竞争伤文本能力 → 内心独白 / Thinker-Talker | Qwen2.5-Omni：Thinker 隐状态 + 文本流式给 Talker |
| 全双工 | Moshi：三流每帧 80 ms 同步、RQ-Transformer（时间 7B + 深度小模型）、内心独白 | 160 ms 理论 / 200 ms 实测；GPT-4o 320 ms；人类 200 ms |
| 时延四段 | 分帧 + 首 token + 解码 + 语义决策 | 半双工串联 1–3 s（VAD + ASR + LLM + TTS）；全双工一步 |
| 成本 | 全双工持续 decode，每路约半张 H100，不能 batch 摊薄 | 半双工空闲时零成本 |

核心问题的答案：语音比图片更需要离散 token，因为语音模型要**生成**——VLM 只需理解，连续特征进 LLM 就够；而把 LLM 的输出变回每秒上万个采样点的波形，唯一可行的方式是让 LLM 生成一个短的离散序列（codec token，每秒几十帧、每帧几个码），再用 codec 解码器还原波形。RVQ 用几个 1024 项的小码本得到 $$2^{80}$$ 的等效表示能力，且自然分层（第一码本内容、后面细节），这个层次直接决定了 VALL-E 的 AR + NAR 结构与 Moshi 的多流建模。全双工的时延由四段相加：codec 帧长（Mimi 80 ms）、模型一步的首 token 时间（7B 约 40 ms）、流式 codec 解码（几 ms）、以及模型判断"该说了"的语义决策——后者不是计算而是能力，全双工模型每一帧都在决策所以没有额外等待。半双工把 VAD、ASR、LLM、TTS 串联起来是 1–3 秒，全双工把它们合成一个模型的一步是 200 ms，代价是模型要持续运行、每路对话占半张卡。下一篇进入生成线的数学核心：扩散模型。


## 下一篇

[扩散模型：DDPM、score matching 与 flow matching](/diffusion-models-ddpm-score-matching-and-flow-matching.html)
