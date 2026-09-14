---
layout: post
title: 预训练：从 tokenizer 到训练配方（总纲）
subtitle: "Pretraining: Tokenizers, Scaling Laws, Data Pipelines and Training Recipes"
tags: [Transformer, LLM, AI, Pretraining]
catalog: true
---


## 内容简介

《预训练：从 tokenizer 到训练配方》是一组共四篇的系列文章，面向要做或要读懂一次预训练的算法工程师，以及要为一次预训练做容量与 I/O 规划的训练基础设施工程师。它是[《Transformer 与 LLM：结构、算量与数值》](/transformer-and-llm-for-infra-engineers.html)那张成本表的**训练侧**：那八篇把 token 数、参数量、数据量当作给定的输入，算出模型每一步算多少、读多少、存多少；本系列讲这几个输入各自是怎么定下来的——tokenizer 决定 token 数，scaling law 决定参数量与数据量的分配，数据管线决定有多少 token 可用、怎么配，训练配方决定用什么超参把它训出来、怎么不崩。

它回答的问题是：

> **打开一份预训练技术报告——词表 128K、15T token、峰值 lr 8e-5、batch 16M——这些数字是怎么定的？每一个花了多少钱？换一个会怎样？**

方法与成本表那八篇相同：**写出公式，代入真实模型的超参数，算出数字，解释数字对系统意味着什么**。词表大小换压缩率、参数换数据、过滤的严格程度换 token 量、学习率与 batch 换稳定性——每个预训练决定都算一笔账。算不出来的那部分（哪个阈值、哪种配比、哪组超参更好）都靠同一种方法：用小模型的消融外推，这就是 scaling law 作为方法论的全部内容。

系列覆盖的范围可以用一行概括——成本表的第六组变量：

```text
训练变量    词表大小与压缩率 · N 与 D 的分配 · 数据的过滤与配比 · lr、batch、调度与稳定性开关   → 第一到四篇
```

读完之后，读者应该能把任何一份预训练技术报告放进这四组变量里，看出它站在哪个时代、与同行差在哪、每个决定花了多少。


## 为什么写这个系列？

### 成本表的自变量不是天上掉下来的

$$6ND$$ 里的 $$N$$ 与 $$D$$、每个公式里的 token 数 $$s$$、`config.json` 里的 `vocab_size`——在成本表那八篇里全是输入。但 Llama 3 为什么用 128K 词表而不是 32K？8B 模型为什么训 15T token 而不是 Chinchilla 说的 160B？15T 从哪来？lr 为什么是 8e-5？这些决定各自有一套可以算的账，而且相互牵连：词表决定 $$D$$ 与"数据量"之间的汇率，scaling law 决定要多少 $$D$$，数据管线决定有没有这么多 $$D$$，配方决定能不能把它训完。

### 这四件事在现有材料里是四个领域

tokenizer 在 NLP 教材里、scaling law 在几篇论文里、数据工程在 FineWeb / DCLM 的技术报告里、训练配方散落在各家的模型报告与几篇稳定性论文里。它们很少被放在同一张表上讨论，但在一次真实的预训练里它们是同一个预算表的四行。本系列把它们放回同一张成本表。

### 算法与 Infra 在这里再次交汇

预训练是算法工程师的领域，但它的每一步都有 Infra 的形态：lm_head 是最贵的单个矩阵、数据管线的 CPU 小时与 I/O 带宽、checkpoint 的写带宽换训练的有效时间、loss spike 一次一万 GPU 小时。算法工程师从本系列知道每个决定花多少，Infra 工程师从中知道 $$N$$、$$D$$ 从哪来、数据侧与 checkpoint 侧要准备什么。


## 适合哪些读者？

### 要做或要读懂一次预训练的算法工程师

你在为一个新模型定词表、定规模、配数据、写超参表，或者在读别人的技术报告想知道它每个数字的依据。本系列给的是每个决定背后的公式与公开配方的对照表。

### 要为一次预训练做容量与 I/O 规划的训练基础设施工程师

你要回答"这个项目要多少卡多少天、数据管线要多少 CPU、存储要多大的读写带宽、checkpoint 多久写一次"。$$6ND$$ 与 MFU 给 GPU 小时，数据管线给 CPU 小时与几十 MB/s 的读带宽，checkpoint 与故障率给 GB/s 的写带宽——都在本系列里。

### 读完成本表、想知道"训练侧"怎么算的读者

成本表那八篇建立的是推理与一次前向的账；本系列用同样的方法把账算到训练的整个生命周期。


## 系列的整体主线

四篇按"token 从哪来 → 算力怎么分 → 数据从哪来 → 怎么训出来"的顺序推进：

```text
第一篇：分词与词表 —— BPE、2Vd、压缩率与每字符成本
        ↓
第二篇：Scaling law —— Chinchilla 的 N/D 分配、推理成本纳入后的"过训练"
        ↓
第三篇：预训练数据工程 —— 从 240T 到 15T 的漏斗、MinHash、配比 → epoch
        ↓
第四篇：训练配方与稳定性 —— lr、batch、调度、loss spike 的三个机制与六个开关
```

贯穿四篇的是同一条训练线：**Llama 3** 与 **DeepSeek-V3** 的技术报告——tokenizer 128K 与 129K、15T 与 14.8T token、lr 8e-5 与 2.2e-4；数据侧以 **FineWeb** 的公开管线作为参照。每一篇都用同样的方法：写出公式，代入真实模型的数字，算出结果，解释结果对系统意味着什么。


## 章节结构与分章导读

### 1. 分词与词表：BPE、词表大小与 token 效率

第一篇从成本表里最后一个外生变量——token 数——从哪来讲起。tokenizer 同时决定成本表的两端：词表大小 $$V$$ 进参数量与 lm_head 的 FLOPs，压缩率决定一段文字要付多少个 token。

这一篇会覆盖：

- 词级与字符级两端各失败在哪；子词；n-gram 语言模型与困惑度、词向量到上下文相关表示的一页史，以及困惑度为什么依赖 tokenizer（跨 tokenizer 要换算到 bits/byte）；
- BPE 算法（合并顺序就是词表）与经典玩具例子；byte-level 初始词表；预分词正则如何决定数字与空格的切法（GPT-2、cl100k 的 1–3 位数字、Qwen 的逐位）；WordPiece 与 Unigram；
- 词表大小的账：$$2Vd$$ 参数（Llama-3-8B 1.05B、13.1%），lm_head 每 token $$2Vd$$ FLOPs（7.0%；Qwen2.5-0.5B 38%），decode 每步读 1.05 GB，训练时 logits $$\text{tokens} \times V \times 4$$ 字节（8K 序列 3.9 GiB，必须分块或融合）；
- token 效率的账：五个真实 tokenizer 在英文 / 中文 / 代码 / 数字上的字符/token；Llama 2 → 3 的 3.17 → 3.94 让每字符 FLOPs 低 15%、KV 低 20%；词表翻倍压缩率近似对数增长，与 lm_head 的线性成本相交于"最优词表"；中文在 cl100k 与 DeepSeek-V3 下每字 1.46 对 0.69 个 token；
- tokenizer 对模型行为的副作用：欠训练 token、数字切分与算术、多语言的价格差、特殊 token 与 chat template。

核心问题是：

> **Llama 3 把词表从 32K 扩到 128K，每个 token 贵了 5.6%，为什么反而是省钱的？同一句中文在两个 128K 量级的词表下 token 数差 2.1 倍，差在哪？**

实践：从零实现 byte-level BPE 并扫词表大小；用 `tiktoken` / `tokenizers` 对比五个真实 tokenizer；`llm_cost.py` 加上词表这一列与"每字符成本"。

### 2. Scaling law：从 Chinchilla 到"过训练"，算力怎么分给参数与数据

第二篇回答 $$C = 6ND$$ 没有说的事：同样的算力怎么分给 $$N$$ 与 $$D$$。分法在 2020、2022、2024 各改了一次。

这一篇会覆盖：

- Kaplan 等 2020 的三条幂律与 $$N \propto C^{0.73}$$；Chinchilla 的参数化 $$L = E + A/N^\alpha + B/D^\beta$$ 与三种拟合方法；两者为什么不同（固定长度的 lr 调度、不数 embedding、规模）；常数的可靠性（Besiroglu 等 2024 的重拟合）；
- 拉格朗日推导 $$N_{opt} \propto C^{0.5}$$ 与 $$D/N \approx 20$$；$$10^{21}$$ 到 $$10^{26}$$ FLOPs 的最优点表与 GPU 小时；十几个真实模型的 $$D/N$$（从 GPT-3 的 2 到 Qwen2.5-7B 的 2368）与它们离最优点的 loss 差；
- Chinchilla 之后：推理成本 $$2N D_{inf}$$ 不在 $$6ND$$ 里；固定算力缩小模型 10 倍 loss 只高 0.053 而推理便宜 10 倍；推理感知的最优点随预期服务量移动（服务 100T token 时 24B / 13.8T 而非 81B / 1.5T）；数据重复的有效 token（4 epoch 值 93%）；MoE 的 $$N$$ 用哪个；
- scaling law 作为实验方法：固定 $$D$$ 扫 $$N$$ 与 IsoFLOP 两种扫法、Llama 3 用万分之一算力定 405B、从 loss 到 benchmark 的两步法、常见错误。

核心问题是：

> **Llama-3 8B 用 15T token，是 Chinchilla 最优数据量的 10 倍，loss 高 0.05 nats。为什么放弃这 0.05 反而是正确的？"最优"在 2022 和 2024 各指什么？**

实践：在 CPU 上训 7 个字符级小模型，拟合 $$L(N)$$ 并外推最大的那个（外推 1.317，实测 1.342）；`llm_cost.py` 加上 Chinchilla 计算器、推理感知最优点与有效 token。

### 3. 预训练数据工程：从 Common Crawl 到 15T token，去重、过滤与配比的账

第三篇讲预训练里唯一不在 GPU 上跑的大工程：从 240T token 的网页正文到 15T 训练集之间的几十个步骤，每步留下多少、花多少、为什么。

这一篇会覆盖：

- 原料：Common Crawl 的规模、正文抽取（WARC + trafilatura 远好于 WET）、语言识别；漏斗刻度 240T → 15T（6%）→ 1.3–5.4T（模型打分后）；
- 过滤的两层：Gopher 的文档级与重复度规则、C4 的行级规则（零成本，只清明显垃圾）；FineWeb-Edu 与 DCLM 的模型打分（大模型标几十万篇 → 小分类器跑全量）及其偏差；
- 去重的四个粒度：URL、文档（MinHash 的 $$P[\min h(A) = \min h(B)] = J$$，LSH 的 $$1 - (1 - J^r)^b$$ 与阈值 0.72，灵敏度随文档长度变化）、行、子串；FineWeb"跨快照全局去重反而更差"的发现；
- 配比换算成 epoch：$$w_i D / U_i$$——Llama 3 的 25% 数学推理意味着有限语料跑 7 个多 epoch；代理模型定权重（DoReMi、RegMix）；退火阶段换高质量数据，以及用退火评估一份新数据；合成数据的算术动机；
- 污染检测（8-gram）；管线的 CPU 账（抽取 ≫ 去重 ≈ tokenize）；存储 60 TB 与训练时只有几十 MB/s 的读带宽。

核心问题是：

> **Common Crawl 有 240T token 的文本，为什么 Llama 3 只用了 15T？被丢掉的 94% 是什么、怎么判定的？15T 里 25% 的"数学与推理"从哪来？**

实践：从零实现 MinHash + LSH 并验证 S 曲线；实现 Gopher / C4 规则并对典型网页判定；`llm_cost.py` 加上漏斗、CPU 小时、配比 → epoch。

### 4. 训练配方与稳定性：学习率、batch、调度与 loss spike

第四篇讲那张十几行的超参表：每个数字从哪来、改了会怎样、训练为什么会崩以及怎么让它不崩。它不重推深度学习基础的推导（方差传播、Adam、warmup），只把公开配方放到同一张表上比较，并把"不稳定"拆成三个可度量、可单独修的机制。

这一篇会覆盖：

- 目标函数：交叉熵的单位（nats、PPL、bits/byte）；MTP 的收益与它的 lm_head 成本；文档打包时掩不掩跨文档 attention（Llama 3 掩、DeepSeek 不掩）；
- 优化器与超参：AdamW 的 $$\beta_2 = 0.95$$、解耦 weight decay 与它的排除项、$$\epsilon$$ 随规模；batch 由梯度噪声尺度决定并随训练增大（405B：4M → 8M → 16M）；峰值 lr 随宽度减小（7B 3e-4 → 405B 8e-5）、$$\mu$$P 与 DeepSeek 的经验律；warmup 占步数不到 1%；
- 调度：cosine、WSD、DeepSeek-V3 的四段；为什么 cosine 中途的 loss 不可比（第二篇分歧的根源）；退火与换数据；
- 稳定性：attention logit 增长（QK-norm、soft-cap、QK-Clip）、输出 logit 漂移（z-loss）、单步过大（裁剪、warmup、$$\beta_2$$）；六个开关与 2024–25 年的默认配置（OLMo 2、Gemma 3、Qwen3、Kimi K2）；spike 的处理流程与代价（405B 一次约 1 万 GPU 小时）；低精度如何放大每个开关的必要性；
- 长上下文继续预训练（Llama 3 六步到 128K、DeepSeek-V3 两步）；该监控的几条曲线。

核心问题是：

> **Llama 3 405B 的峰值 lr 是 8e-5，DeepSeek-V3 是 2.2e-4；batch 分别是 16M 与 63M token。这些数字怎么定的？DeepSeek-V3 在 FP8 下训 14.8T token 没有一次不可恢复的 loss spike——它开了哪些开关，每个在防什么？**

实践：CPU 上复现三种调度的对比、batch 与最优 lr 的关系、attention logit 随 lr 从 36 涨到 12592 与 QK-norm 把它压到 22、z-loss 对 $$\log Z$$ 的抑制；`llm_cost.py` 加上超参表、checkpoint 字节数与写带宽、spike 回滚的代价。本篇最后给出系列总结。


## 贯穿全系列的实践线

本系列接着成本表那八篇的 `llm_cost.py` 往下长：脚本在第八版结束时可以为任何一个给出 `config.json` 的模型、任何一组硬件参数输出推理侧的成本表，本系列再加上训练侧的四列：

```text
第一篇    词表                   2Vd 与 lm_head 占比；logits 显存；每字符成本
第二篇    scaling law            Chinchilla 最优 N/D 与 GPU 小时；推理感知最优点；有效 token
第三篇    数据                   漏斗刻度；抽取 / 去重 / tokenize 的 CPU 小时；配比 → epoch
第四篇    配方                   超参表 → 步数与每步时间；checkpoint 字节与写带宽；spike 回滚代价
```

与它平行的独立实验全部在 CPU 上可跑：从零实现的 BPE 与 MinHash + LSH、7 个字符级小模型的 scaling law 拟合、三种 lr 调度与 QK-norm / z-loss 的对照。

源码与资料阅读线：

```text
第一篇    Sennrich 等 2016（BPE）· Radford 等 2019（GPT-2 的 byte-level BPE）· Kudo 2018（Unigram）· Tao 等 2024（词表的 scaling law）· Llama 3 论文的 tokenizer 一节
第二篇    Kaplan 等 2020 · Hoffmann 等 2022（Chinchilla）· Besiroglu 等 2024（重拟合）· Sardana & Frankle 2023（推理感知）· Muennighoff 等 2023（数据受限）· Llama 3 论文的 scaling law 一节
第三篇    Penedo 等 2024（FineWeb）· Li 等 2024（DCLM）· Rae 等 2021（Gopher 的过滤规则）· Lee 等 2021（去重）· Broder 1997（MinHash）· Llama 3 与 DeepSeek-V3 的数据章节
第四篇    McCandlish 等 2018（梯度噪声尺度）· Yang 等 2022（μP）· Wortsman 等 2023（小规模复现不稳定）· Chowdhery 等 2022（PaLM 的 z-loss 与 spike 处理）· OLMo 2 · Llama 3 / DeepSeek-V3 / Kimi K2 报告的训练配方
```

四篇的脚本（`llm_cost_09` 到 `llm_cost_12`，各自独立可运行）与独立实验保存在 [ai-learning-labs/transformer-and-llm](https://github.com/arganzheng/ai-learning-labs/tree/main/transformer-and-llm)，与成本表八篇的脚本同一目录，附每个脚本的完整输出。


## 阅读路径建议

### 完整学习路径

```text
1 → 2 → 3 → 4
```

先读[《Transformer 与 LLM》](/transformer-and-llm-for-infra-engineers.html)的第一、二篇（参数量与 $$6ND$$），再进本系列，四篇顺序读。

### 算法工程师，要做或要读懂一次预训练

四篇全读。读技术报告时，第二篇的表告诉你它的 $$D/N$$ 落在哪个时代，第四篇的表告诉你它的超参与同行差在哪。

### 做训练基础设施，要为一次预训练做容量与 I/O 规划

```text
2 → 3 → 4
```

$$6ND$$ 与 MFU 给 GPU 小时（第二篇的表），数据侧的 CPU 小时与几十 MB/s 的读带宽（第三篇），checkpoint 的 GB/s 写带宽与 spike 回滚的代价（第四篇）。

### 排查训练不稳定

```text
4
```

loss spike 的三个机制、六个开关、该监控的曲线都在第四篇；数值格式本身的问题回到《Transformer 与 LLM》第六篇。


## 本系列的边界

本系列只讨论"一个基座模型**怎么训出来**"的账。以下内容与它紧邻，但不在范围内：

- **模型作为计算对象的成本**：参数量、FLOPs、字节数、KV cache、通信量的推导。它们是本系列的前提，在[《Transformer 与 LLM：结构、算量与数值》](/transformer-and-llm-for-infra-engineers.html)。
- **后训练**：SFT、RLHF / DPO、蒸馏、评测。把一个基座模型变成对话模型的方法在[《后训练：从 SFT 到可验证奖励》](/post-training-from-sft-to-verifiable-rewards.html)。
- **深度学习基础的推导**：反向传播、初始化与归一化、优化器、正则化的公式。第四篇直接使用它们的结论，推导在[《深度学习基础》](/deep-learning-foundations.html)。
- **分布式训练的实现**：TP / PP / EP / 序列并行如何切分与同步、checkpoint 如何写、故障如何恢复。本系列只算它们的**量**（GPU 小时、写带宽、回滚代价），实现在[《大规模训练工程：从并行策略到容错恢复》](/large-scale-training-from-parallelism-to-fault-tolerance.html)。
- **数据管线的工程实现**：本系列算 CPU 小时与带宽，不讲 Spark / Ray / datatrove 的用法。


## 前置要求与说明

### 前置要求

- 读过[《Transformer 与 LLM》](/transformer-and-llm-for-infra-engineers.html)的第一、二篇，知道参数量公式与 $$6ND$$；
- 会读 Python 与 PyTorch 代码；
- 知道交叉熵、Adam、学习率 warmup 是什么（第四篇会用到它们的结论，不重推）。

不要求：

- 训练过大模型；
- 有 GPU。全部实验在 CPU 上可以完成。

### 版本与模型基线

- 以 **Llama 3**（405B：15.6T token、lr 8e-5、batch 4M → 16M；8B / 70B：15T token）与 **DeepSeek-V3**（14.8T token、lr 2.2e-4、batch 63M、FP8）的技术报告为主要对象；
- 数据侧以 **FineWeb**（96 个 Common Crawl 快照、15T token）与 **DCLM** 的公开管线为参照；
- tokenizer 对比用可公开下载的 GPT-2、cl100k_base、o200k_base、Qwen2.5、DeepSeek-V3；
- Chinchilla 的常数用 Besiroglu 等 2024 的重拟合值；
- 文中所有 FLOPs、GPU 小时与带宽都是**理论值**，用于建立数量级判断与相互比较；论文引用以第一作者与年份标注。


## 章节目录

1. [分词与词表：BPE、词表大小与 token 效率](/tokenizer-vocabulary-and-token-efficiency.html)
2. [Scaling law：从 Chinchilla 到"过训练"，算力怎么分给参数与数据](/scaling-laws-and-compute-optimal-training.html)
3. [预训练数据工程：从 Common Crawl 到 15T token，去重、过滤与配比的账](/pretraining-data-pipeline-dedup-filtering-and-mixture.html)
4. [训练配方与稳定性：学习率、batch、调度与 loss spike](/pretraining-recipe-and-training-stability.html)


## 最终目标

读完这套系列之后，拿到任何一份预训练技术报告，读者应该能够回答：

```text
换一个 tokenizer 会怎样？                          → 第一篇：2Vd 与每字符成本
给定算力，模型多大、数据多少？训完要服务多少？        → 第二篇：Chinchilla 与推理感知的最优点
15T token 从哪来、丢掉的是什么、够不够？             → 第三篇：漏斗、MinHash、配比 → epoch
超参表里的每个数字从哪来？训练为什么会崩？           → 第四篇：μP、梯度噪声尺度、三个机制与六个开关
```

最终目标是一种能力：**读报告的能力**——打开一份预训练技术报告，能把它的 tokenizer、$$D/N$$、数据配比与超参表放到本系列的表里，看出它站在哪个时代、与同行差在哪、每个决定花了多少；反过来，在自己定这些数字时，知道每个数字背后的账。
