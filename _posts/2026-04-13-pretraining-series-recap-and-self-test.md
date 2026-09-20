---
layout: post
series: pretraining
title: "预训练（05）：系列总结与通关自测"
subtitle: "Pretraining: Series Recap and Final Self-Test"
tags: [Transformer, LLM, AI, Pretraining]
catalog: true
date: 2026-04-13 20:00:00
---

四篇正文回答了一个问题：**一个基座模型是怎么训出来的，每个训练决定花多少**。第一篇算 tokenizer 与词表，第二篇算算力怎么分给参数与数据，第三篇算 15T token 从哪来、丢掉的是什么，第四篇算超参表里每个数字的来历与训练为什么会崩。四篇合起来，是[《Transformer 与 LLM》](/transformer-and-llm-for-infra-engineers.html)那张成本表的训练侧。

本文不讲新内容，做三件事：把四篇压成一张表与四段回顾，把贯穿四篇的几条线拎出来，然后给一套三段式的通关自测——判断与计算、跨篇综合、面试题。各篇末尾的自测检验的是"这一篇读懂了没有"，这里检验的是"四篇能不能连起来用"。

> **读完这四篇，你应该能回答哪些问题？[^q0] 哪些数字与结论必须能脱口而出？[^q1] 怎么判断自己是"读过"还是"掌握"了？[^q2]**

## 一、总览：系列回答的问题与主线

系列的一句话主张是：**预训练的每个决定都能算账，算不出来的部分靠小模型消融外推**。词表大小换压缩率、参数换数据、过滤的严格程度换 token 量、学习率与 batch 换稳定性——四篇各算一笔账，用的是同一套方法（推导 → 代入真实模型 → 解释数字）、同两个对象（Llama 3 与 DeepSeek-V3）。

| 篇 | 回答的问题 | 一句话结论 | 必记的数字 / 公式 |
|---|---|---|---|
| [第一篇：分词与词表](/tokenizer-vocabulary-and-token-efficiency.html) | 词表从 32K 扩到 128K，每 token 贵了 5.6%，为什么反而省钱？ | 成本要按字符而不是按 token 算：更大的词表让每 token 贵一点、让每段文字的 token 少很多，后者赢；前提是词表针对目标语言训练 | 词表参数 $$2Vd$$（Llama-3-8B 1.05B，13.1%）；lm_head 每 token $$2Vd$$ FLOPs（7.0%，0.5B 模型 38%）；英文 3.17 → 3.94 字符/token，每字符 FLOPs 低 15%、KV 低 20%；中文在两个 128K 量级词表下差 2.1 倍 |
| [第二篇：Scaling law](/scaling-laws-and-compute-optimal-training.html) | Llama-3 8B 训 15T 是 Chinchilla 最优的 10 倍数据、loss 高 0.05，为什么是正确的？ | Chinchilla 只最小化训练算力下的 loss；把推理算进去，最优点移向小模型、多数据 | $$L = E + A/N^\alpha + B/D^\beta$$；$$C = 6ND$$；$$D/N \approx 20$$；固定 $$C$$ 缩小 10 倍：loss +0.053、推理 1/10；服务 100T token 时最优 24B / 13.8T 而非 81B / 1.5T；4 epoch 值 93% |
| [第三篇：数据工程](/pretraining-data-pipeline-dedup-filtering-and-mixture.html) | Common Crawl 有 240T token，为什么只用 15T？丢掉的 94% 是什么？ | 一条漏斗（启发式过滤 → 四粒度去重 → 模型打分），每一级的刻度都被消融验证过；配比的百分比本质是 epoch 数 | 240T → 15T（6%）→ 1.3–5.4T；MinHash 14 × 8 阈值 0.72；跨快照全局去重反而更差；25% 数学推理 ≈ 7.5 epoch；一次消融 2700 H100 小时；抽取 ≫ 去重 ≈ tokenize |
| [第四篇：配方与稳定性](/pretraining-recipe-and-training-stability.html) | 405B 的 lr 8e-5、V3 的 2.2e-4，batch 16M 与 63M——怎么定的？V3 靠什么没有一次不可恢复的 spike？ | 超参表的每个数字都有来历（$$\mu$$P、梯度噪声尺度、$$1/(\eta\lambda)$$）；不稳定拆成三个可单独度量、单独修的机制 | lr 3e-4 → 1.5e-4 → 8e-5 随宽度降；batch 4M → 16M / 12.6M → 63M ramp；warmup 0.4–0.9%；wd 时间尺度 $$1/(\eta\lambda)$$ ≈ 7–13% 训练；QK-norm：logit 12592 → 22；spike 一次约 1 万 GPU 小时；$$T_{opt} = \sqrt{2\delta \cdot \text{MTBF}}$$ |

### 1. 本文的章节安排

| 章 | 内容 |
|---|---|
| 二 | 逐篇回顾：核心问题、结论、必记、常见误解 |
| 三 | 贯穿四篇的四条线：按字符算账、算力与数据的分配、重复与 epoch、消融外推 |
| 四 | 常见误区表 |
| 五 | 通关自测：A 判断与计算 10 题、B 跨篇综合 5 题、C 面试题 7 题、D 掌握判据 |
| 六 | 下一步 |

## 二、逐篇回顾

### 1. 第一篇：分词与词表：BPE、词表大小与 token 效率

**核心问题**：Llama 3 把词表从 32K 扩到 128K，每个 token 贵了 5.6%，为什么反而是省钱的？同一句中文在两个 128K 量级的词表下 token 数差 2.1 倍，差在哪？

**结论**：tokenizer 同时决定成本表的两端。词表大小 $$V$$ 进参数量（embedding 与 lm_head 各 $$V \times d$$）、进 lm_head 的 FLOPs（每 token $$2Vd$$）、进 decode 的字节（每步读一遍 lm_head）、进训练显存（logits 占 $$\text{tokens} \times V \times 4$$ 字节）；压缩率决定一段文字要付多少个 token。两条路方向相反，所以成本要按**每个字符**算：$$\text{FLOPs/字符} = \text{FLOPs/token} \div \text{字符/token}$$。Llama-3-8B 的骨架配 32K 词表是 4.49 GFLOPs/字符，配 128K 是 3.81——每字符便宜 15%，每字符的 KV 少 20%。BPE 的合并顺序就是词表；byte-level 初始词表保证任何字节串可切分；预分词正则决定数字与空格怎么切。词表翻倍时压缩率近似对数增长，与 lm_head 的线性成本相交于"最优词表"，当前的 128K–256K 是这条曲线上的一段。

**必记**：

- 词表参数 $$2Vd$$（tied 为 $$Vd$$）：Llama-3-8B 1.05B，占 13.1%；训练状态按 16 字节/参数是 16.8 GB。
- lm_head 每 token $$2Vd$$ FLOPs：8B 占 7.0%；Qwen2.5-0.5B（$$d = 896$$、$$V = 152$$K）占 38%——小模型的词表开销失控。
- 训练 logits：8K 序列 × 128K × 4 B = 3.9 GiB，每个序列；softmax 前后两份，必须 vocab-parallel 或分块融合。
- 英文压缩率 3.17 → 3.94 字符/token（+24%）；每 token +5.6%，每字符 $$1.056 / 1.243 = 0.85$$。
- 中文：cl100k 每字 1.46 token，DeepSeek-V3 0.69——差 2.1 倍，差在训练语料不在 $$V$$。
- 跨 tokenizer 比 loss 要换算 $$\text{bits/byte} = \frac{L}{\ln 2} \cdot \frac{T}{B}$$；同等 bits/byte 下 Llama 3 每 token loss 应比 Llama 2 高 24%。

**常见误解**："1 token ≈ 0.75 个英文词"是通用规则——只对英文成立，中文视词表 0.4–1.5 字符/token，多语言 API 的 token 预算、上下文"有多长"、计费公平性都要按语言分别估。另一个：两个 tokenizer 不同的模型可以直接比 per-token loss——不能，token 更细的那个 per-token loss 天然更低。

### 2. 第二篇：Scaling law：从 Chinchilla 到"过训练"，算力怎么分给参数与数据

**核心问题**：Llama-3 8B 用 15T token，是 Chinchilla 最优数据量的 10 倍，loss 高 0.05 nats。为什么放弃这 0.05 反而是正确的？"最优"在 2022 和 2024 各指什么？

**结论**：loss 是 $$N$$ 与 $$D$$ 的幂律 $$L = E + A/N^\alpha + B/D^\beta$$，在 $$C = 6ND$$ 约束下用拉格朗日乘子得 $$\alpha A/N^\alpha = \beta B/D^\beta$$，代入重拟合常数得 $$N_{opt} \propto C^{0.5}$$、$$D/N \approx 20$$。Kaplan 2020 的 $$N \propto C^{0.73}$$（模型优先）与它的差异来自固定长度的学习率调度、不数 embedding、规模小。但 Chinchilla 的"最优"只最小化**训练算力**下的 loss；模型训完要服务，推理成本 $$2ND_{inf}$$ 不在 $$6ND$$ 里。固定算力把 $$N$$ 缩 $$k$$ 倍，$$\Delta L \propto (\ln k)^2$$——$$k = 10$$ 时只 +0.053 nats，推理便宜 10 倍。把推理算进去，最优条件多一项 $$(1 + D_{inf}/3D)$$，服务 100T token 的模型最优点从 81B / 1.5T 移到 24B / 13.8T。这就是 2024 年后 $$D/N$$ 到 200–2000 的"过训练"。scaling law 同时是**实验方法**：固定 $$D$$ 扫 $$N$$ 或 IsoFLOP，用万分之一的算力定 405B 的配置；最常见的失败是学习率调度不匹配、tokenizer 不一致、超参没随尺寸调、外推太远。

**必记**：

- $$E = 1.82$$、$$\alpha = 0.35$$、$$\beta = 0.37$$（Besiroglu 等 2024 重拟合）。
- Llama-3 8B：$$C = 6 \times 8 \times 10^9 \times 1.5 \times 10^{13} = 7.2 \times 10^{23}$$；405B：$$3.8 \times 10^{25}$$ = 2670 万 H100 小时 @ 40% MFU（实际 3084 万）。
- 最优 loss 随算力：$$L_{opt} - E \propto C^{-\alpha\beta/(\alpha+\beta)}$$，指数 0.178——算力 ×10 可约 loss ×0.66，减半要 ×49。
- 数据重复：$$D' = U + UR^*(1 - e^{-R/R^*})$$，4 epoch 值 93%，16 epoch 值 66%，上限约 16 倍。
- 超参随算力：$$\eta_{opt} \propto C^{-0.125}$$、$$B_{opt} \propto C^{0.33}$$——算力 ×10，lr −25%，batch ×2.1。
- 长上下文的算力用 $$M = 72Ld^2 + 12Lds$$ 而不是 $$6N$$。

**常见误解**："Chinchilla 说 $$D/N = 20$$ 最优，所以 Llama-3 8B 训 15T 是浪费"——它最小化的是训练算力，不是全生命周期成本。另一个：cosine 调度中途的 loss 可以拿来拟合——不能，这正是 Kaplan 与 Chinchilla 分歧的根源之一。

### 3. 第三篇：预训练数据工程：从 Common Crawl 到 15T token，去重、过滤与配比的账

**核心问题**：Common Crawl 有 240T token 的文本，为什么 Llama 3 只用了 15T？被丢掉的 94% 是什么、怎么判定的？15T 里 25% 的"数学与推理"从哪来？

**结论**：数据管线是一条漏斗，公开数据集给了每一级刻度：抽出正文 240T → 过滤去重后的通用网页 15T（6.2%）→ 模型打分后 1.3–5.4T（0.5–2.2%）。过滤分两层：启发式规则（Gopher 的 7 条文档规则与重复度、C4 的行级规则）零成本但只清明显垃圾；模型打分用大模型标几十万篇、小分类器跑全量（FineWeb-Edu ≥ 3 分留 8.7%，DCLM 取前 10%）。去重有四个粒度（URL、文档、行、子串），文档级用 MinHash：$$P[\min h(A) = \min h(B)] = J$$，LSH 的 $$P(\text{候选}) = 1 - (1 - J^r)^b$$，FineWeb 用 $$b = 14$$、$$r = 8$$ 得阈值 0.72；跨快照全局去重反而更差——重复是要控制分布，不是消灭。配比换算成 epoch 是 $$w_i D / U_i$$：25% × 15T ÷ 约 0.5T 的独立数学数据 ≈ 7.5 epoch，"15T"不是 15T 条不同的文本。管线成本按文档数而非 token 数增长，贵在最上游的抽取；训练时读带宽只有 9–46 MB/s，加载器的难点是掩码、确定性与配比切换而非吞吐。

**必记**：

- 漏斗刻度：240T → 15T（6%）→ 5.4T（FineWeb-Edu ≥ 2）/ 3.8T（DCLM 前 10%）/ 1.3T（≥ 3）。
- MinHash 112 个哈希估 Jaccard 到 ±0.04（方差 $$J(1-J)/k$$）；LSH 14 × 8：$$J = 0.6$$ 时 21% 成候选，$$0.8$$ 时 92%。
- 模型打分的账：用 8B 给 15T 打分 $$2.4 \times 10^{23}$$ FLOPs 是训练的 1/3，所以两级——标注只花约 $$7 \times 10^{19}$$。
- 领域数据靠分类器迭代召回：DeepSeekMath 四轮 14.7B → 120B。
- 一次配比消融 1.8B × 350B token ≈ 2700 H100 小时，差距常达 3–5 分；预算里预留 1–3% 算力做消融。
- 抽取约 70 万核·小时 ≫ 去重 ≈ tokenize；存储 60 TB；训练读带宽 9–46 MB/s。

**常见误解**："去重越彻底越好"——FineWeb 发现跨快照全局去重把被多次转载的高质量内容删成长尾，质量反而下降。另一个："配比表里的 25% 是 25% 的不同数据"——它是重复次数，25% 数学推理意味着有限语料跑 7 个多 epoch。

### 4. 第四篇：训练配方与稳定性：学习率、batch、调度与 loss spike

**核心问题**：Llama 3 405B 的峰值学习率是 8e-5，DeepSeek-V3 是 2.2e-4，GPT-3 是 6e-5；batch 分别是 16M、63M、3.2M token。这些数字怎么定的？DeepSeek-V3 在 FP8 下训了 14.8T token 没有一次不可恢复的 loss spike——它开了哪些开关，每个开关在防什么？

**结论**：配方分三组决定——目标（交叉熵；MTP 权重 0.3 → 0.1；代码 FIM 50%；Llama 3 掩跨文档 attention、DeepSeek 不掩）、优化（AdamW $$\beta = (0.9, 0.95)$$、wd 0.1、clip 1.0；峰值 lr 随宽度减小；batch 由临界 batch 定并随训练 ramp；warmup 按步数 0.4–0.9%；cosine → 10% 或 WSD）、稳定（QK-norm、z-loss、初始化、wd 排除项、Adam $$\epsilon$$）。峰值 lr 随宽度减小是 $$\mu$$P 的 $$\propto 1/d$$ 与 DeepSeek 经验律 $$0.31 C^{-0.125}$$ 的共同结论，7B 3e-4 → 70B 1.5e-4 → 405B 8e-5，GPT-3 的 6e-5 在同一条线上；V3 的 2.2e-4 高是因为 MoE 激活参数只有 37B 且 batch 更大。batch 由梯度噪声尺度 $$\text{tr}(\Sigma) / \lVert G \rVert^2$$ 决定、随训练增大所以 ramp（405B 4M → 8M → 16M，V3 12.6M → 63M），硬件下界是副本数 × 序列长。不稳定拆成三个机制：attention logit 增长（QK-norm 把 12592 压到 22）、输出 logit 漂移（z-loss $$10^{-4}$$）、单步过大（裁剪、warmup、$$\beta_2 = 0.95$$）；低精度放大每个开关的必要性。spike 的处理是回退 100 步、跳 200–500 个 batch，405B 一次 8 千–1.6 万 GPU 小时；硬件故障每 3 小时一次，checkpoint 间隔按 $$T_{opt} = \sqrt{2\delta \cdot \text{MTBF}}$$，异步保存让间隔短到 4–5 分钟、有效时间 > 90%。

**必记**：

- lr：$$\mu$$P $$\propto 1/d$$；$$0.31 C^{-0.125}$$；算力涨 1000 倍 lr 只降到 40%。
- batch：405B 4M → 8M → 16M；V3 12.6M → 63M；16M / 8192 = 2048 条序列，512 路数据并行每副本 4 条。
- weight decay 时间尺度 $$\tau = 1/(\eta\lambda)$$：7B 33K 步（7%）、405B 125K 步（13%）。
- 调度实验：WSD 1.464 < 常数 1.536 < cosine 1.578。
- QK-norm：无时 logit 12592、loss 2.23 → 2.75；有时 22、2.19 → 2.45。
- 405B checkpoint 5.7 TB；长上下文阶段 800B token、六步到 128K，attention 占比 4% → 40%。

**常见误解**："lr 是调出来的经验值"——它随宽度的下降有 $$\mu$$P 与经验律两条依据，公开配方全部落在同一条线上。另一个："loss spike 是数据的问题"——三个机制里两个（attention logit、$$\log Z$$ 漂移）是模型内部的数值问题，各有自己的开关与监控曲线。

## 三、贯穿全系列的几条线

### 1. 按字符算账，不按 token

第一篇建立的度量贯穿其后三篇。tokenizer 决定了一段文字要付多少 token，所以第二篇的 $$D$$、第三篇的 15T、第四篇的 batch 16M——每一个"token 数"背后都隐含一个词表。Llama 2 → 3 的 3.17 → 3.94 字符/token 意味着同样 15T token 的训练集，Llama 3 读了多 24% 的字符；跨 tokenizer 比 loss 必须换算成 bits/byte，这也是第二篇 scaling law 拟合"tokenizer 不一致"这条常见错误的来源。中文在不同词表下差 2.1 倍的 token 数，直接换成 2.1 倍的 KV、prefill FLOPs 与计费。

### 2. 算力怎么分：参数、数据与推理

第二篇的核心是 $$6ND$$ 的分配问题，但它的三个变量分别被另外三篇约束。$$N$$ 里有第一篇的 $$2Vd$$ 词表参数（8B 的 13%）；$$D$$ 的上限是第三篇的漏斗——过训练要求 $$D/N$$ 上千，唯一 token 至少是目标的四分之一，所以数据管线必须供应得上；$$D$$ 与 $$N$$ 定下后，第四篇从 batch 与序列长算出总步数、每步时间、checkpoint 字节与写带宽。推理成本 $$2ND_{inf}$$ 是让最优点从 81B / 1.5T 移到 24B / 13.8T 的那一项，它在第二篇出现，在成本表系列的推理侧被逐项展开。

### 3. 重复、epoch 与有效数据

同一个公式 $$D' = U + UR^*(1 - e^{-R/R^*})$$ 在第二篇（数据不够怎么办：4 epoch 值 93%、16 epoch 值 66%）与第三篇（配比换算成 epoch：25% × 15T ÷ 0.5T ≈ 7.5 epoch，8 epoch 有效约 80%）各出现一次，两处说的是同一件事的两面：第二篇从"总量不够"出发，第三篇从"某一类不够"出发。第三篇"跨快照全局去重反而更差"是它的反面——重复要控制分布而不是消灭；第四篇退火阶段换高质量数据与用退火评估一份新数据，是把 epoch 这个变量用在训练末段。

### 4. 消融外推：算不出来的靠小模型

四篇里所有无法从公式推出的决定——词表多大、常数是多少、哪个过滤阈值、哪种配比、哪组超参——都靠同一种方法：用小模型的消融外推。第二篇给出方法本身（两种扫法、Llama 3 用万分之一算力定 405B、从 loss 到 benchmark 的两步法、常见错误）；第三篇给出它的价格（1.8B × 350B token = 2700 H100 小时，差距 3–5 分，预算留 1–3%）；第四篇给出超参也要 scaling（$$\eta_{opt} \propto C^{-0.125}$$、$$B_{opt} \propto C^{0.33}$$）与 $$\mu$$P 让超参跨宽度迁移。这条线是"预训练的每个决定都能算账"这一主张的另一半。

| 概念 | 出现的篇 | 关系 |
|---|---|---|
| 字符/token、bits/byte | 一、二 | 一定义度量；二用它解释"tokenizer 不一致"为什么让拟合失效 |
| $$6ND$$、$$D/N$$ | 二、三、四 | 二推导最优分配；三供应 $$D$$；四把 $$N$$、$$D$$ 换成步数与时间 |
| 有效数据 $$D'$$、epoch | 二、三 | 二从总量不够出发；三从配比出发；同一公式 |
| $$2Vd$$、lm_head | 一、二 | 一算它的成本；二决定 $$N$$ 数不数 embedding（Kaplan 与 Chinchilla 分歧之一） |
| 学习率调度 | 二、四 | 二说 cosine 中途不可比、拟合要匹配调度；四比较 cosine / WSD / 四段 |
| 小模型消融 | 二、三、四 | 二给方法；三给价格；四给超参迁移（$$\mu$$P） |
| checkpoint 与故障 | 二、四 | 二给 GPU 小时；四给 5.7 TB、每 3 小时一次故障、$$T_{opt}$$ |

## 四、常见误区

| 误区 | 为什么错 | 正确的说法 | 出处 |
|---|---|---|---|
| 词表越大每 token 越贵，所以小词表省钱 | 只看了 token 一端；同一段文字的 token 数随词表变大而减少 | 按每字符算：128K 词表比 32K 每字符便宜 15% | [第一篇](/tokenizer-vocabulary-and-token-efficiency.html) |
| 两个模型的 per-token loss 可以直接比 | tokenizer 不同时 token 粒度不同 | 换算成 bits/byte 再比 | [第一篇](/tokenizer-vocabulary-and-token-efficiency.html) |
| 词表大小相近，效率就相近 | 效率取决于词表用什么语料训出来 | 中文在 cl100k 与 DeepSeek-V3 下差 2.1 倍 | [第一篇](/tokenizer-vocabulary-and-token-efficiency.html) |
| Chinchilla 说 $$D/N = 20$$ 最优，过训练是浪费 | Chinchilla 只最小化训练算力下的 loss | 把推理算进去，最优点移向小模型、多数据 | [第二篇](/scaling-laws-and-compute-optimal-training.html) |
| 算力翻 10 倍 loss 就明显下降 | 可约 loss 只 ×0.66，减半要 ×49 | 收益是幂律的，指数 0.178 | [第二篇](/scaling-laws-and-compute-optimal-training.html) |
| 数据不够就重复，效果一样 | 有效数据随 epoch 递减 | 4 epoch 值 93%、16 epoch 只值 66% | [第二篇](/scaling-laws-and-compute-optimal-training.html) |
| 去重越彻底越好 | 跨快照全局去重把被多次转载的好内容删成长尾 | 按快照内去重；重复是控制分布 | [第三篇](/pretraining-data-pipeline-dedup-filtering-and-mixture.html) |
| 用大模型给全量数据打分 | 8B 给 15T 打分是训练算力的 1/3 | 大模型标几十万篇，小分类器跑全量 | [第三篇](/pretraining-data-pipeline-dedup-filtering-and-mixture.html) |
| 数据管线的瓶颈是训练时的读吞吐 | 训练读带宽只需 9–46 MB/s | 贵在最上游的抽取；加载器难在掩码、确定性、配比切换 | [第三篇](/pretraining-data-pipeline-dedup-filtering-and-mixture.html) |
| 学习率是调出来的经验值 | 公开配方全部落在随宽度下降的同一条线上 | $$\mu$$P $$\propto 1/d$$、$$0.31 C^{-0.125}$$ | [第四篇](/pretraining-recipe-and-training-stability.html) |
| batch 越大越好，反正有卡 | 临界 batch 随训练增大，早期大 batch 浪费样本 | ramp：4M → 8M → 16M；硬件下界是副本数 × 序列长 | [第四篇](/pretraining-recipe-and-training-stability.html) |
| loss spike 是脏数据造成的 | 三个机制里两个是模型内部数值问题 | attention logit（QK-norm）、$$\log Z$$（z-loss）、单步过大（裁剪） | [第四篇](/pretraining-recipe-and-training-stability.html) |

## 五、通关自测

### A. 判断与计算（10 题）

1. 一个 $$d = 2048$$、$$V = 256$$K 的 untied 模型，词表参数多少？若骨架参数 2B，词表占比多少？

   <details markdown="1"><summary>答案</summary>

   $$2Vd = 2 \times 262144 \times 2048 = 1.07$$B；占 $$1.07 / (2 + 1.07) \approx 35\%$$——比 Llama-3-8B 的 13.1% 高得多，小模型配大词表时词表开销失控。

   </details>

2. 训练时序列长 32K、$$V = 128$$K、FP32 logits，一个序列的 logits 占多少显存？

   <details markdown="1"><summary>答案</summary>

   $$32768 \times 128256 \times 4 \approx 15.6$$ GiB——是 8K 序列 3.9 GiB 的 4 倍，且 softmax 前后要两份；必须 vocab-parallel 或分块融合交叉熵。

   </details>

3. 某 tokenizer 英文 4.2 字符/token，另一个 3.0；同一模型骨架下每字符 FLOPs 相差多少（忽略词表大小对每 token FLOPs 的影响）？

   <details markdown="1"><summary>答案</summary>

   $$3.0 / 4.2 = 0.71$$，前者每字符便宜 29%；同一段文字 token 数少 29%，KV 也少 29%。

   </details>

4. 训练算力 $$10^{24}$$ FLOPs，按 Chinchilla（$$D/N \approx 20$$）模型多大、数据多少？

   <details markdown="1"><summary>答案</summary>

   $$6N \cdot 20N = 10^{24}$$，$$N = \sqrt{10^{24}/120} \approx 91$$B，$$D \approx 1.8$$T——与第二篇表中 $$5.76 \times 10^{23}$$ → 72B / 1.33T 同一条 $$N \propto C^{0.5}$$ 线。

   </details>

5. 可约 loss 要降到原来的一半，训练算力要乘多少？只降 10% 呢？

   <details markdown="1"><summary>答案</summary>

   指数 0.178：$$0.5^{-1/0.178} \approx 49$$ 倍；$$0.9^{-1/0.178} \approx 1.8$$ 倍。

   </details>

6. 独立数据只有 2T token，训一个 $$D = 8$$T 的配置，有效数据约多少？改成 $$D = 32$$T 呢？

   <details markdown="1"><summary>答案</summary>

   4 epoch 有效约 93%，$$D' \approx 7.4$$T；16 epoch 只值 66%，$$D' \approx 21$$T——多训的 24T 只换来 13.6T 的有效数据，收益递减。

   </details>

7. MinHash-LSH 用 $$b = 20$$、$$r = 5$$，候选阈值约多少？与 FineWeb 的 14 × 8 比，哪个更"宽"？

   <details markdown="1"><summary>答案</summary>

   $$(1/20)^{1/5} \approx 0.55$$，比 0.72 低——更多相似度较低的文档对成为候选，去重更激进。

   </details>

8. 配比里代码占 17%、总量 15T、独立代码数据约 1T：代码训了几个 epoch？

   <details markdown="1"><summary>答案</summary>

   $$0.17 \times 15 / 1 \approx 2.6$$ epoch——比数学推理的 7.5 温和得多，4 epoch 以内几乎无损。

   </details>

9. 峰值 lr $$1.5 \times 10^{-4}$$、wd 0.1、总步数 60 万：weight decay 的时间尺度多少步、占训练多少？

   <details markdown="1"><summary>答案</summary>

   $$\tau = 1/(1.5 \times 10^{-4} \times 0.1) = 67$$K 步，占 11%——参数只记住最近约一成的梯度历史，落在 7–13% 的区间。

   </details>

10. batch 63M token、序列长 4096、数据并行 1024 路：每副本每步几条序列？batch 能否降到 2M？

    <details markdown="1"><summary>答案</summary>

    $$63\text{M} / 4096 \approx 15.4$$K 条，每副本 15 条；硬件下界 = 1024 × 4096 ≈ 4.2M，2M 会让一半副本空转，不行。

    </details>

### B. 跨篇综合（5 题）

1. Llama 2 → Llama 3 词表 32K → 128K，如果两者都训 15T token，Llama 3 实际读了多少字符的数据？这对第二篇的 $$D$$ 意味着什么？

   <details markdown="1"><summary>答案</summary>

   第一篇：压缩率 3.17 → 3.94，多 24% 的字符；第二篇：scaling law 里的 $$D$$ 是 token 数，跨 tokenizer 比较时同一 $$D$$ 不是同样多的信息，拟合必须固定 tokenizer——这正是"tokenizer 不一致"这条常见错误。

   </details>

2. 一个 7B 模型要服务 100T token，按第二篇应训多少数据？这些数据从第三篇的漏斗里够不够拿？

   <details markdown="1"><summary>答案</summary>

   第二篇：推理感知最优点在 24B / 13.8T 一侧，7B 应把 $$D$$ 推到十几 T、$$D/N$$ 上千；第三篇：通用网页过滤去重后只有 15T（模型打分后 1.3–5.4T），高质量子集要跑多个 epoch——用第二篇的 $$D'$$ 公式，4 epoch 以内几乎无损，配比表要按 epoch 数写。

   </details>

3. 数学推理占 25%、独立数据 0.5T，第三篇算出 7.5 epoch。这个 epoch 数在第四篇的调度里会怎么用？

   <details markdown="1"><summary>答案</summary>

   第四篇：多阶段与退火——高质量数据在退火阶段加大权重、且退火可以用来评估一份新数据；DeepSeek-V3 的四段调度与 Llama 3 的退火都在末段换数据。7.5 epoch 意味着这部分数据在每个阶段都被反复看到，退火阶段的权重要与前面阶段合计算总 epoch。

   </details>

4. 405B 训练算力 $$3.8 \times 10^{25}$$ FLOPs、batch 16M、序列长 8192、每步时间已知：从这些数字推出总步数与一次 spike 的代价。

   <details markdown="1"><summary>答案</summary>

   第二篇：$$D = C / 6N = 3.8 \times 10^{25} / (6 \times 4.05 \times 10^{11}) \approx 15.6$$T；第四篇：步数 = 15.6T / 16M ≈ 97 万步（ramp 阶段 batch 更小，实际更多）；spike 回退 100 步 + 跳 200–500 batch，按 2670 万 H100 小时 / 97 万步 ≈ 27 H100 小时/步，一次约 8 千–1.6 万 GPU 小时。

   </details>

5. 用 scaling law 定一个新模型的配置时，第三篇与第四篇各贡献了哪一条"常见错误"的解药？

   <details markdown="1"><summary>答案</summary>

   第三篇：数据消融要用足够大的规模（1.8B × 350B，2700 H100 小时），太小看不出质量差别，且 tokenizer 与数据配比在扫的各个尺寸间必须一致；第四篇：超参数也要 scaling（$$\eta_{opt} \propto C^{-0.125}$$、$$B_{opt} \propto C^{0.33}$$）或用 $$\mu$$P 迁移，否则小模型的最优超参在大模型上不成立、外推失效；学习率调度要跑完整（cosine 中途不可比）。

   </details>

### C. 面试题（7 题）

1. 我们要为一个中文为主的产品训基座模型，tokenizer 应该怎么选？给出判断依据与代价。

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 词表要用目标语言语料训——同为 128K 量级，中文每字 1.46 对 0.69 token，差 2.1 倍，直接换成 KV、prefill、计费；(2) 词表大小按每字符成本选：lm_head $$2Vd$$ 随 $$V$$ 线性涨、压缩率对数涨，小模型（0.5B 级 lm_head 占 38%）不宜配大词表；(3) 预分词正则决定数字怎么切（逐位 vs 1–3 位），影响算术；(4) 换 tokenizer 的代价是从头预训练或一段扩词表的继续预训练，欠训练 token 会漏到模型行为里。
   **追问方向**：训练时 logits 显存（8K 序列 3.9 GiB）怎么办；如何跨 tokenizer 比 loss（bits/byte）。
   **好答案与一般答案的区别**：一般答案只说"用中文语料训、词表大一点"；好答案把 $$V$$ 的两端成本与目标语言压缩率放到一张每字符账上。

   </details>

2. 有 $$10^{24}$$ FLOPs 的训练预算，模型该多大？把你的推理与"业界都在过训练"调和起来。

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 先给 Chinchilla 答案 $$N \propto C^{0.5}$$、$$D/N \approx 20$$ → 约 90B / 1.8T；(2) 指出它只最小化训练算力下的 loss；(3) 引入推理成本 $$2ND_{inf}$$：固定 $$C$$ 缩 10 倍 loss 只 +0.053、推理 1/10；(4) 用预期服务量定最优点（服务 100T：24B / 13.8T 而非 81B / 1.5T）；(5) 数据供应是否跟得上（$$D'$$ 公式、4 epoch 93%）。
   **追问方向**：MoE 的 $$N$$ 用哪个；长上下文 $$6ND$$ 差多少（$$M = 72Ld^2 + 12Lds$$）；Kaplan 与 Chinchilla 为什么不同。
   **好答案与一般答案的区别**：一般答案背 $$D/N = 20$$；好答案说出"最优"是相对哪个约束的，并给出推理感知后的移动方向与数量级。

   </details>

3. 描述一条从 Common Crawl 到训练集的数据管线，每一级大约留下多少，哪一步最贵？

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 漏斗刻度 240T → 15T（6%）→ 1.3–5.4T；(2) 顺序：URL 过滤 → 正文抽取（WARC + trafilatura 远好于 WET）→ 语言识别 → 启发式规则（Gopher / C4）→ 四粒度去重（MinHash 14 × 8 阈值 0.72）→ 模型打分（两级）；(3) 最贵的是抽取（约 70 万核·小时），去重贵在 shuffle 不在哈希；(4) 每一级的取舍靠消融验证（2700 H100 小时一次）；(5) 训练读带宽只有几十 MB/s，加载器难在掩码、确定性、配比切换。
   **追问方向**：为什么全局去重更差；模型打分为什么两级；污染检测（8-gram）。
   **好答案与一般答案的区别**：一般答案列步骤；好答案每步给刻度与成本，并说出哪些结论反直觉（全局去重更差、重复是控制分布）。

   </details>

4. 配比表写着"数学与推理 25%"，你会怎么核查它是否合理？

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 换算成 epoch：$$w_i D / U_i$$，25% × 15T ÷ 约 0.5T ≈ 7.5 epoch；(2) 用有效数据公式判断递减（8 epoch 有效约 80%，16 epoch 66%）；(3) 问这 3.75T 从哪来——分类器迭代召回（DeepSeekMath 四轮 14.7B → 120B）、合成与改写；(4) 用小模型消融验证配比（DoReMi / RegMix 用代理模型定权重）；(5) 退火阶段是否再加权。
   **追问方向**：消融的规模与价格；如何检测评测集污染。
   **好答案与一般答案的区别**：一般答案讨论"数学重要不重要"；好答案把百分比翻译成 epoch 与有效数据，再问数据来源与验证方法。

   </details>

5. 训练 loss 突然上升，你排查的顺序是什么？各对应哪个开关？

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 先看三条曲线定位机制：attention logit 最大值（增长 → QK-norm / soft-cap / QK-Clip）、输出 $$\log Z$$（漂移 → z-loss $$10^{-4}$$）、梯度范数（单步过大 → 裁剪 1.0、warmup、$$\beta_2 = 0.95$$）；(2) 低精度（FP8 / BF16）放大每个开关的必要性——V3 用分块量化 + 每 128 项提升到 FP32 累加；(3) 处理流程：回退 100 步、跳 200–500 个 batch；(4) 代价：405B 一次约 1 万 GPU 小时，所以 checkpoint 间隔按 $$T_{opt} = \sqrt{2\delta \cdot \text{MTBF}}$$ 缩到几分钟；(5) 数据只是嫌疑之一，不是默认答案。
   **追问方向**：QK-norm 实验里的数字（logit 12592 → 22）；weight decay 排除项；MoE 的专家坍缩。
   **好答案与一般答案的区别**：一般答案说"降 lr、跳过坏数据"；好答案先把不稳定拆成三个可单独度量的机制，再对症。

   </details>

6. 为什么公开配方里大模型的学习率都比小模型小？给出两条独立的依据与一组数字。

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) $$\mu$$P：隐层学习率 $$\propto 1/d$$，让激活尺度不随宽度变；(2) DeepSeek 经验律 $$\eta = 0.31 C^{-0.125}$$，算力涨 1000 倍 lr 只降到 40%；(3) 数字：7B 3e-4 → 70B 1.5e-4 → 405B 8e-5，GPT-3 175B 6e-5 在同一条线上；(4) 例外要能解释：V3 的 2.2e-4 因为 MoE 激活参数 37B 且 batch 63M 更大；(5) batch 同样有律：$$B_{opt} \propto C^{0.33}$$、临界 batch 随训练增大所以 ramp。
   **追问方向**：warmup 占多少步、为什么按步数不按比例；WSD 与 cosine 谁好（1.464 vs 1.578）。
   **好答案与一般答案的区别**：一般答案说"大模型更不稳定所以 lr 小"；好答案给理论（$$\mu$$P）与经验律两条依据并解释例外。

   </details>

7. 让你规划一次 70B 级预训练的容量与 I/O，你会算哪几笔账？

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) GPU 小时：$$C = 6ND$$ ÷（峰值 FLOPs × MFU），405B 是 2670 万 H100 小时 @ 40%（实际 3084 万）；(2) 数据侧 CPU：抽取 ≫ 去重 ≈ tokenize，约 70 万核·小时量级，存储 60 TB；(3) 训练读带宽只需 9–46 MB/s；(4) checkpoint：405B 5.7 TB，故障每 3 小时一次，间隔按 $$T_{opt} = \sqrt{2\delta \cdot \text{MTBF}}$$，异步保存到几分钟，写带宽 GB/s 级；(5) spike 回滚与消融预算各留几个百分点；(6) 长上下文阶段 attention 占比 4% → 40%，算力要单独算。
   **追问方向**：MFU 取多少、为什么；数据并行副本数决定 batch 下界；异步 checkpoint 怎么做到 $$\delta$$ 很小。
   **好答案与一般答案的区别**：一般答案只算 GPU 小时；好答案把 CPU、存储、带宽、checkpoint 与失败预算一起放上表。

   </details>

### D. 掌握判据

| 水平 | 表现 |
|---|---|
| 读过 | 能说出四篇各讲什么；知道 $$6ND$$、$$D/N \approx 20$$、MinHash、QK-norm 这些名词 |
| 掌握 | A 组能不翻书算出 8 题以上；B 组能说出每题用了哪两篇的什么；拿到一份技术报告能指出它的 $$D/N$$ 落在哪个时代、超参与同行差在哪 |
| 能教人 | C 组每题能给出全部要点并预判追问；能解释四篇里每个反直觉结论（过训练是对的、全局去重更差、lr 随宽度降）为什么成立 |

通关标准：A 组至少 8 题、B 组至少 4 题、C 组每题能说出一半以上要点。没过的部分回到第二章对应篇的"必记"，再回该篇正文。

## 六、下一步

四篇算的是"一个基座模型怎么训出来"的账，四个方向紧邻但不在范围内：

- **模型作为计算对象的成本**（参数量、FLOPs、字节、KV cache、通信量的推导）是本系列的前提，在[《Transformer 与 LLM：结构、算量与数值》](/transformer-and-llm-for-infra-engineers.html)。
- **后训练**（SFT、RLHF / DPO、蒸馏、评测）在[《后训练：从 SFT 到可验证奖励》](/post-training-from-sft-to-verifiable-rewards.html)。
- **深度学习基础的推导**（反向传播、初始化与归一化、优化器）在[《深度学习基础：从反向传播到残差》](/deep-learning-foundations.html)——第四篇直接用了它们的结论。
- **分布式训练的实现**（TP / PP / EP 怎么切、checkpoint 怎么写、故障怎么恢复）在[《大规模训练工程：从并行策略到容错恢复》](/large-scale-training-from-parallelism-to-fault-tolerance.html)——本系列只算它们的量。

回到总纲：[《预训练：从 tokenizer 到训练配方》](/pretraining-from-tokenizer-to-training-recipe.html)。

## 七、延伸阅读

本系列只讨论"一个基座模型**怎么训出来**"的账。以下内容与它紧邻，但不在范围内：

- **模型作为计算对象的成本**：参数量、FLOPs、字节数、KV cache、通信量的推导。它们是本系列的前提，在[《Transformer 与 LLM：结构、算量与数值》](/transformer-and-llm-for-infra-engineers.html)。
- **后训练**：SFT、RLHF / DPO、蒸馏、评测。把一个基座模型变成对话模型的方法在[《后训练：从 SFT 到可验证奖励》](/post-training-from-sft-to-verifiable-rewards.html)。
- **深度学习基础的推导**：反向传播、初始化与归一化、优化器、正则化的公式。第四篇直接使用它们的结论，推导在[《深度学习基础》](/deep-learning-foundations.html)。
- **分布式训练的实现**：TP / PP / EP / 序列并行如何切分与同步、checkpoint 如何写、故障如何恢复。本系列只算它们的**量**（GPU 小时、写带宽、回滚代价），实现在[《大规模训练工程：从并行策略到容错恢复》](/large-scale-training-from-parallelism-to-fault-tolerance.html)。
- **数据管线的工程实现**：本系列算 CPU 小时与带宽，不讲 Spark / Ray / datatrove 的用法。


[^q0]: 四个：换一个 tokenizer 会怎样（$$2Vd$$ 与每字符成本）；给定算力模型多大、数据多少、训完要服务多少（Chinchilla 与推理感知的最优点）；15T token 从哪来、丢掉的是什么、够不够（漏斗、MinHash、配比 → epoch）；超参表里的每个数字从哪来、训练为什么会崩（$$\mu$$P、梯度噪声尺度、三个机制与六个开关）。详见[第二章](#二逐篇回顾)。
[^q1]: 词表参数 $$2Vd$$、Llama-3-8B 的 13.1% 与 7.0%；英文 3.17 → 3.94 字符/token、每字符低 15%；$$L = E + A/N^\alpha + B/D^\beta$$ 与 $$C = 6ND$$、$$D/N \approx 20$$；固定 $$C$$ 缩 10 倍 loss +0.053；服务 100T token 时 24B / 13.8T；4 epoch 值 93%；漏斗 240T → 15T → 1.3–5.4T；MinHash 14 × 8 阈值 0.72；25% ≈ 7.5 epoch；lr 3e-4 → 8e-5 随宽度、batch 4M → 16M ramp；QK-norm 12592 → 22；spike 一次约 1 万 GPU 小时。详见[第一章](#一总览系列回答的问题与主线)、[第三章](#三贯穿全系列的几条线)。
[^q2]: 用第五章的三段自测：A 组 10 题判断与计算（至少 8 题）、B 组 5 题跨篇综合（至少 4 题）、C 组 7 道面试题（每题说出一半以上要点）；D 组的表给出"读过 / 掌握 / 能教人"三级的表现。详见[第五章](#五通关自测)。
