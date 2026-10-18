---
layout: post
title: 深度学习基础：从反向传播到残差（总纲）
subtitle: "Deep Learning Foundations: From Backpropagation to Residual Connections"
tags: [AI, Deep Learning, LLM]
catalog: true
---


## 内容简介

《深度学习基础：从反向传播到残差》是一组十篇正文加一篇总结的系列文章，对应[《AI 算法工程师学习地图》](/ai-algorithm-engineer-learning-roadmap.html)的第 L3 层。它面向已经有 [L0 数学](/math-for-ai-algorithm-engineers.html)、[L1 工具](/tooling-for-ai-algorithm-engineers.html)、[L2 经典机器学习](/classical-machine-learning-in-the-llm-era.html)基础、准备进入 Transformer 与 LLM 的读者，讲的是**训练一个深度神经网络时会发生什么、为什么、以及怎么算出来**。

它回答的问题是：

> **一个几十层甚至上百层的网络，为什么能训、什么时候不能训、训不动的时候该看哪个数字？[^q0]**

深度学习教材通常按"模型"组织：感知机、MLP、CNN、RNN、Transformer。这个系列先用第一篇沿着感知机到 Transformer 回看深度模型的演进与动机，再用六篇讲训练动力学——梯度、激活、深层稳定性、优化与泛化，最后用三篇回看卷积、表示学习与循环结构如何把组件带到 Transformer。每一篇都用**推导**、**算账**与 CPU 上可复现的**实验**，最后指出它在 Transformer 与 LLM 里的形态。

举一个例子说明这个系列的取法。"残差连接让深网络能训"是每本教材都有的一句话；本系列要把它变成三件可验证的事：

1. **推导**：没有残差时第 $$l$$ 层的梯度是 $$l$$ 个 [Jacobian](# "tip: 雅可比矩阵：一个向量函数对它的向量输入的导数，第 (i, j) 格是第 i 个输出对第 j 个输入的偏导，形状 [输出维, 输入维]。一层网络的 Jacobian 描述这一层把上游梯度放大或缩小多少；《[算法工程师的数学（07）：导数、梯度与链式法则——softmax 的梯度与策略梯度](/derivatives-gradients-chain-rule-and-policy-gradient.html)》「三、链式法则与 Jacobian」有一个 2×2 的手算例子") 的乘积，任何一个的谱范数系统性地偏离 1 就指数级放大或缩小；有残差时每层的 Jacobian 变成 $$I + J_l$$，乘积里多出一条恒等通路。用大白话说：梯度从顶层传到底层要连乘 $$l$$ 个系数，每个系数只要稍微小于 1，乘一百多次就归零，稍微大于 1 就爆炸；残差给它加了一条"不用乘、直接过"的通路。
2. **算账**：一个 128 层、每层 Jacobian 谱范数 0.9 的网络，最底层梯度是顶层的 $$0.9^{128} \approx 1.4 \times 10^{-6}$$；加残差后不再有这个因子。
3. **实验**：用 NumPy 搭一个 64 层 MLP，画出有无残差时各层梯度范数的分布，两条曲线差六个数量级。

Transformer 的每一层都靠这条恒等通路，Pre-Norm 与 Post-Norm 的全部争论也在这条通路上。

系列先沿年代回看模型演进，再覆盖六个训练动力学问题，最后用三篇聚焦视觉、表示与序列结构：第一篇从感知机讲到 Transformer；第二至第七篇从损失、梯度、激活与初始化讲到优化和泛化；第八至第十篇分别讲 CNN、Embedding 与 RNN / attention。

| | 问题 | 内容 | 篇 |
|---|---|---|---|
| 演进史 | 深度模型怎样走到 Transformer | 感知机、ADALINE、XOR、两次寒冬、逐层预训练与结构演进 | 第一篇 |
| 动力学一 | 「好」怎样写成一个可求导的数 | 损失函数：MSE / 交叉熵、focal、label smoothing、InfoNCE、DPO | 第二篇 |
| 动力学二 | 梯度怎么流 | 反向传播、链式法则、激活存储、反向为什么是前向的两倍 | 第三篇 |
| 动力学三 | 深层网络为什么需要非线性 | sigmoid / tanh 的饱和、ReLU 的稀疏与死亡、GELU / SiLU、GLU 门控 | 第四篇 |
| 动力学四 | 为什么深了就难训 | 方差在层间的传播、初始化、归一化、残差、Pre/Post-Norm | 第五篇 |
| 动力学五 | 优化器在做什么 | SGD、Momentum、Adam / AdamW、学习率调度、梯度裁剪 | 第六篇 |
| 动力学六 | 什么情况会过拟合？怎么办？ | 正则化、dropout、weight decay、double descent、epoch 数 | 第七篇 |
| 结构史一 | 卷积解决了什么 | 参数共享、感受野、LeNet → ResNet、ViT 把图切成 token | 第八篇 |
| 结构史二 | 离散符号如何变成连续表示 | one-hot、Embedding 查表、Word2Vec、静态到上下文表示 | 第九篇 |
| 结构史三 | 循环解决了什么、败在哪 | RNN、时间反传、LSTM 门控、seq2seq、attention 的诞生 | 第十篇 |

Table: 系列覆盖的模型演进史、六个训练动力学问题与三类结构


## 术语与缩写

{% include glossary-table.html series="deep-learning-foundations" %}

## 一个深度网络由哪些部件组成

把十篇正文压到一张图上，就是一条前向链加一条训练回路：

![部件地图：输入 → Embedding → 层（线性/卷积/循环/attention + 激活 + 归一化 + 残差）→ 输出头 → 损失；损失 ↓ 反向传播 → 优化器 ↑ 回层里更新参数；初始化从肘形进层、正则化把惩罚项 / 平滑目标注进损失](/img/in-post/dl-component-map.svg)

每个部件对应一个它要回答的问题，以及本系列回答它的那一篇：

| 部件 | 回答的问题 | 在哪一篇 |
|---|---|---|
| 输入 / Embedding | 离散 token 或像素怎样变成向量 | 《[深度学习基础（09）：Embedding 与表示学习——从查表到句向量](/embedding-and-representation-learning.html)》 |
| 层：线性 / 卷积 / 循环 / attention | 结构的先验从哪来 | 《[深度学习基础（08）：CNN——从 LeNet 到 ResNet，再到 ViT](/cnn-from-lenet-to-resnet-and-vit.html)》《[深度学习基础（10）：RNN——从 LSTM 到 attention 的诞生](/rnn-lstm-and-the-birth-of-attention.html)》 |
| 激活函数 | 非线性怎样改变梯度 | 《[深度学习基础（04）：激活函数——从 sigmoid 到 SwiGLU](/activation-functions-from-sigmoid-to-swiglu.html)》 |
| 归一化 + 残差 | 深层的尺度怎么稳住 | 《[深度学习基础（05）：训练为什么不稳定——初始化、归一化与残差](/initialization-normalization-and-residual.html)》 |
| 损失 | 「好」怎样写成一个可求导的数 | 《[深度学习基础（02）：损失函数——怎样把「好」写成一个可求导的数](/loss-functions-from-mse-to-dpo.html)》 |
| 反向传播 | 梯度怎样算出来 | 《[深度学习基础（03）：反向传播——手推一个两层网络](/backpropagation-by-hand.html)》 |
| 优化器 | 参数怎样更新、步长多大 | 《[深度学习基础（06）：优化器——从 SGD 到 AdamW 与学习率调度](/optimizers-from-sgd-to-adamw.html)》 |
| 初始化 | 训练从哪开始 | 《[深度学习基础（05）：训练为什么不稳定——初始化、归一化与残差](/initialization-normalization-and-residual.html)》 |
| 正则化 | 会不会泛化 | 《[深度学习基础（07）：正则化与泛化——为什么参数比样本多却不过拟合](/regularization-and-generalization.html)》 |

Table: 部件 → 问题 → 篇

十篇正文于是分成两组：**动力学篇**（第二至第七篇）是让网络训得动、训得好的部件——损失、反向传播、激活、初始化 / 归一化 / 残差、优化器、正则化；**结构史篇**（第八至第十篇）是层的三种结构先验加输入表示——CNN、Embedding、RNN / attention；第一篇演进史与第十一篇总结分别给出年代线与回收。下文的章节安排按这个分组展开。


## 为什么写这个系列？

### LLM 时代的深度学习基础没有变少，只是换了名字

Transformer 之后，"深度学习基础"常被当成历史课跳过。但打开任何一份预训练技术报告，遇到的仍是这一层的词：loss spike 与梯度裁剪、warmup 步数、AdamW 的 $$\beta_2$$、weight decay 系数、RMSNorm 与 Pre-Norm、初始化的标准差、$$\mu$$P、z-loss、QK-norm。每一个都是本系列某一篇的主题在大模型上的形态。不懂这层基础，报告里的三句话就读不动：

- 为什么 Llama 的初始化标准差是 0.02、GPT-2 的残差分支要额外除以 $$\sqrt{2L}$$？[^q1]
- Adam 的二阶矩与 warmup 是什么关系，为什么 warmup 必不可少？[^q2]
- dropout 的期望等价是什么，为什么 LLM 预训练几乎不用它？[^q3]

上面这一串词每一个在本系列里都有出处，总结篇会逐个回收；先给对照表，读完可以回来核对：

| 技术报告里的词 | 它是哪个基础概念在大模型上的形态 | 篇 |
|---|---|---|
| sigmoid、GELU、SiLU、SwiGLU | 导数怎样改变梯度流<br/>门控 FFN 如何用激活选择信息 | 第四篇 |
| RMSNorm、Pre-Norm | 方差在层间怎么传播，归一化放在残差分支前还是后 | 第五篇 |
| 初始化标准差 0.02、GPT-2 残差分支除以 $$\sqrt{2L}$$ | 残差流的方差随层数线性增长 | 第五篇 |
| $$\mu$$P、z-loss、QK-norm | 规模化之后的三个稳定性补丁：学习率随宽度漂移、lm_head logits 漂移、attention logits 增长 | 第五篇第七章 |
| loss spike、梯度裁剪、warmup | 梯度范数的量级与坏 batch<br/>Adam 初期步长偏大<br/>裁剪把一步总范数压回阈值 | 第六篇 |
| AdamW 的 $$\beta_2$$、weight decay 系数 | 二阶矩的记忆长度<br/>解耦的 L2 正则 | 第五、六篇 |
| 预训练不用 dropout、一个 epoch | dropout 的期望等价与数据充足时的正则化需求 | 第七篇 |
| token embedding、Word2Vec、weight tying | 离散 token 如何查表成向量，词向量如何从上下文学习 | 第九篇 |

Table: 技术报告里的词与它对应的基础概念

### 从"会调用"到"会诊断"

L1 工具箱系列教的是二十行训练循环。写出它不难，难的是它跑起来之后 loss 不降、突然爆成 NaN、降到一半开始震荡——每种情况该看哪个数字、改哪个开关。这需要知道每个开关背后的公式——"开关"指训练循环里那些能拧的量：学习率、batch size、warmup 步数、梯度裁剪阈值、weight decay、初始化标准差、归一化的位置。每拧一个都对应一条公式：梯度范数曲线说明什么、学习率与 batch 的 scaling 规则从哪来、梯度裁剪剪的是什么。这个系列想给的就是这套诊断能力，而不是又一份 API 说明。

### 现有材料的断层

- **教材**（Goodfellow 等《Deep Learning》、Bishop《Deep Learning》）完整但厚，且成书时 Transformer 尚未成为主线，很多现象没有对到 LLM 上；
- **课程**（CS231n、fast.ai）讲直觉与代码，很少把一个现象推到能算出数字；
- **论文**每篇只讲一个方法（Kaiming 初始化、LayerNorm、AdamW），假设读者能自己把它们放进同一张图；
- **博客**大多止于"残差让深网络能训"这一句，不推、不算、不测。

本系列取的是中间那段路：每个现象都推到公式、算到数字、测到曲线，篇幅控制在教材一章的三分之一。


## 适合哪些读者？

### 准备进入 L4 Transformer / LLM 的算法学习者

你已经能写训练循环，接下来要读 Llama、DeepSeek 的技术报告与 [《Transformer 原理与实现》](/transformer-and-llm-structure-implementation-and-evolution.html)系列。本系列是那之前的最后一层基础：读完之后，报告里关于初始化、归一化、优化器、稳定性的每一段都有对应的公式可查。

### 做后训练但基础是"跳过来"的工程师

你在用 `trl` 做 SFT / DPO，能跑通，但 loss 曲线怪的时候不知道往哪看。第二至第七篇是为此准备的：梯度、非线性、稳定性、参数更新与泛化，各自的诊断信号与开关。

### 需要回顾 CNN / RNN 的多模态或序列建模学习者

L7 多模态的 vision encoder 来自 ViT，ViT 来自 CNN 的 patch 化；表示学习把离散 token 映射为向量；语音模型里仍有卷积前端与循环结构。第八至十篇分别讲视觉、表示与序列结构；第一篇则按年代说明它们如何出现，不再将这些分支集中成一个脱离历史顺序的章节。

### Infra 工程师，想知道自己在优化的训练过程里发生了什么

你在做训练框架、checkpoint、容错，看到梯度裁剪、混合精度、optimizer state 这些词，想知道它们在算法上的含义。第三篇（反向为什么是前向两倍、激活为什么要存）与第六篇（每参数 8 字节优化器状态从哪来）直接回答。


## 系列的整体主线

本系列的终点是读懂 Transformer / LLM 的组件从哪里来。顺序不是照年代排：先用演进史说明模型为什么出现，再看各种网络都会遇到的训练动力学，最后追到 Transformer 直接继承的结构线。

| 篇 | 主题 | 内容 |
|---|---|---|
| 第一篇 | [深度模型演进史：从感知机到 Transformer](/evolution-of-deep-models-from-perceptron-to-transformer.html) | 感知机到 Transformer 的演进、XOR 与训练动机 |
| 第二篇 | [损失函数：怎样把「好」写成一个可求导的数](/loss-functions-from-mse-to-dpo.html) | MSE / 交叉熵 / focal / label smoothing / InfoNCE / DPO<br/>梯度形状、初始 loss 诊断、z-loss |
| 第三篇 | [反向传播：手推一个两层网络](/backpropagation-by-hand.html) | 梯度的形状规则，反向 = 2 × 前向，激活为什么要存 |
| 第四篇 | [激活函数：从 sigmoid 到 SwiGLU](/activation-functions-from-sigmoid-to-swiglu.html) | 从 sigmoid、tanh 到 ReLU、GELU、SiLU 与 SwiGLU<br/>比较梯度、门控与计算代价 |
| 第五篇 | [初始化、归一化与残差](/initialization-normalization-and-residual.html) | 方差怎么在层间传播，深网络为什么难训，残差与 Pre-Norm 怎么修 |
| 第六篇 | [优化器：从 SGD 到 AdamW](/optimizers-from-sgd-to-adamw.html) | 学习率调度、梯度裁剪、梯度累积，batch 与学习率的 scaling |
| 第七篇 | [正则化与泛化](/regularization-and-generalization.html) | dropout、weight decay、早停<br/>参数远多于样本时什么情况会过拟合、怎么办 |
| 第八篇 | [CNN：从 LeNet 到 ResNet，再到 ViT](/cnn-from-lenet-to-resnet-and-vit.html) | 卷积作为参数共享，感受野，ViT 把图切成 token |
| 第九篇 | [Embedding 与表示学习](/embedding-and-representation-learning.html) | 从 one-hot 查表、Word2Vec 到上下文表示<br/>token embedding 如何学出可用的几何结构 |
| 第十篇 | [RNN：从 LSTM 到 attention 的诞生](/rnn-lstm-and-the-birth-of-attention.html) | 时间上的反传与梯度消失，LSTM 的门，seq2seq 的瓶颈，attention 的诞生 |

Table: 十篇正文的主题与内容

第二至第七篇覆盖六个跨结构的训练问题：目标怎么写、梯度怎样计算、非线性怎样影响梯度、信号怎样穿过深层、参数如何更新、模型怎样泛化。它不是完整的训练工程课程；混合精度、分布式与显存、数据工程也不在本系列范围内：

| 训练中的问题 | 本系列覆盖 | 边界与延伸 |
|---|---|---|
| 梯度怎么算？ | [反向传播：手推一个两层网络](/backpropagation-by-hand.html) | — |
| 非线性选什么？梯度会不会消失？ | [激活函数：从 sigmoid 到 SwiGLU](/activation-functions-from-sigmoid-to-swiglu.html) | — |
| 信号能否保持深层尺度？ | [初始化、归一化与残差](/initialization-normalization-and-residual.html) | — |
| 参数怎样更新？学习率与 batch 怎么选？ | [优化器：从 SGD 到 AdamW](/optimizers-from-sgd-to-adamw.html) | — |
| 训练好后为什么测试表现变差？ | [正则化与泛化](/regularization-and-generalization.html) | 讨论正则化、验证集表现与泛化边界 |
| 损失函数怎样选择？ | [损失函数：怎样把「好」写成一个可求导的数](/loss-functions-from-mse-to-dpo.html) | MLE 推导另见[《算法工程师的数学（05）：从最大似然到交叉熵——第一个要会推的 loss》](/from-maximum-likelihood-to-cross-entropy.html) |
| 混合精度与浮点格式怎样取舍？ | 本系列不讲 | 见[《算法工程师的工具箱（04）：PyTorch 使用层（下）——混合精度、显存的账与多卡启用》](/pytorch-in-use-mixed-precision-memory-ledger-and-multi-gpu.html)与[《现代 LLM 结构（10）：浮点格式、数值稳定性与混合精度》](/floating-point-formats-and-mixed-precision.html) |
| 分布式训练和显存怎样安排？ | 本系列不讲 | 见[《大规模训练工程（01）：训练任务的状态解剖——显存账与 MFU》](/training-state-anatomy-memory-and-mfu.html)及[《大规模训练工程（02）：并行策略全景——每种并行切的是哪种状态》](/parallelism-strategies-which-state-to-shard.html) |
| 数据从哪里来、怎样清洗配比？ | 本系列不讲 | [《预训练（04）：预训练数据工程：从 Common Crawl 到 15T token，去重、过滤与配比的账》](/pretraining-data-pipeline-dedup-filtering-and-mixture.html) |

Table: 训练一个网络要回答的问题

结构篇不是把所有神经网络都列一遍，而是挑出三条直接通向 Transformer 的线索：

| Transformer 组件 | 来自哪条结构线 | 本系列对应内容 |
|---|---|---|
| FFN 与激活 | MLP | [演进史](/evolution-of-deep-models-from-perceptron-to-transformer.html)与[激活函数篇](/activation-functions-from-sigmoid-to-swiglu.html) |
| 残差 | CNN 线的 ResNet | [CNN 篇](/cnn-from-lenet-to-resnet-and-vit.html)与[初始化、归一化与残差篇](/initialization-normalization-and-residual.html) |
| 归一化 | CNN 中成熟的 BatchNorm<br/>Transformer 使用 LayerNorm / RMSNorm | [CNN 篇](/cnn-from-lenet-to-resnet-and-vit.html)与[初始化、归一化与残差篇](/initialization-normalization-and-residual.html) |
| token embedding | 表示学习 | [Embedding 与表示学习篇](/embedding-and-representation-learning.html) |
| attention | RNN 的 seq2seq 与 encoder-decoder attention | [RNN 篇](/rnn-lstm-and-the-birth-of-attention.html) |
| patch embedding / ViT | CNN 的图像结构 | [CNN 篇](/cnn-from-lenet-to-resnet-and-vit.html) |

Table: Transformer 的组件从哪里来

Hopfield 网络、Boltzmann machine / RBM、自编码器、GAN、VAE 与 GNN 有各自的历史位置，但不构成本系列的三条结构主线；它们只在演进史的年代叙事中定位。站内的[多模态扩散模型篇](/diffusion-models-ddpm-score-matching-and-flow-matching.html)与[Latent Diffusion 篇](/latent-diffusion-dit-and-text-to-image-recipes.html)延伸生成模型，尚无专门的 GAN 或 VAE 文章。

阅读顺序按依赖关系安排，不按年份：CNN 放在前面，因为残差与归一化在这条线上成熟，后面会复用；Embedding 在 RNN 前，因为循环网络接收的输入已经是 token 向量；RNN 放在最后，以 2014 年 attention 收尾，正好接到 Transformer。RNN / LSTM 的年代早于 Word2Vec；年代顺序见[《深度模型演进史：从感知机到 Transformer》](/evolution-of-deep-models-from-perceptron-to-transformer.html)。

三条交织的线索：

| 线索 | 从第一篇到第十篇 |
|---|---|
| 推导线 | 链式法则 → 激活导数改变梯度 → Jacobian 乘积与谱范数 → 更新量的量级 → 期望等价与先验 → 卷积线性算子 → embedding 查表与分布式词向量 → 时间反传 |
| 数字线 | 感知机线性边界：AND 第 6 轮零误分、XOR 最少 3 次；权重状态自第 2 轮起周期 1（脚本输出 `(2, 1)`）→ 反向 FLOPs = 2 × 前向 → sigmoid 首层梯度 $$3.94\times10^{-10}$$、20 层 SiLU 测试准确率 92.88% → $$0.9^{128}$$ → 每参数 8 字节 → 一个 epoch 就够 → ResNet-50 的 4.1 GFLOPs → XOR：ADALINE MSE 0.25、MLP H=4 为 20/20 → text8 17,005,207 词、71,290 词表、100 维 → RNN 不能并行 |
| LLM 线 | 6ND · 激活重算 → SwiGLU · GELU → RMSNorm · Pre-Norm · 0.02 → AdamW · warmup · 裁剪到 1.0 → 预训练不用 dropout → ViT / patch → GPT-2 embedding 参数占 31.0% · 上下文表示 → attention → Transformer |

Table: 贯穿十篇的三条线索

每一篇都用同样的方法：**推导公式，代入真实网络算出数字，用几十行代码复现现象，指出它在 LLM 里的形态**。


## 章节结构与分章导读

### 1. 深度模型演进史——从感知机到 Transformer

第一篇把本系列放回历史：从线性阈值单元与感知机，到 XOR 暴露的表达边界、反向传播与逐层预训练，再到 CNN、Embedding、循环网络、attention 与 Transformer。重点不是重复后文推导，而是交代每种结构解决了什么、留下什么问题。

案例用同一份 XOR 数据比较感知机、ADALINE 与两层 MLP：线性模型的限制、平方误差的最小二乘解、以及 H=2 与 H=4 在不同初始化下的训练结果，给出“能表示 ≠ 能学会”的可复现例子。

### 2. 反向传播：手推一个两层网络

第三篇建立整个系列的对象：梯度。它把反向传播从"框架自动做的事"变成可以手推、手算、手写的东西。

这一篇会覆盖：

- 计算图与链式法则：标量对向量、向量对向量（Jacobian）、标量对矩阵的导数；反向传播就是从 loss 出发沿计算图反向逐节点乘 Jacobian；
- 矩阵求导的形状规则：$$Y = XW$$ 时 $$\partial L / \partial W = X^T (\partial L / \partial Y)$$、$$\partial L / \partial X = (\partial L / \partial Y) W^T$$——记住"梯度与被求导的量形状相同"，两条公式可以直接推出来；
- 逐层手推一个两层 MLP（Linear → ReLU → Linear → softmax → 交叉熵）的前向与反向，每一步写出形状；softmax + 交叉熵的梯度 $$p - y$$（L0 数学系列第八篇推过，这里放进完整网络）；
- 反向为什么是前向的两倍：每个 Linear 层反向要做两个矩阵乘法（对输入的梯度、对权重的梯度），前向只做一个；由此得到训练 FLOPs $$\approx 6ND$$——现代 LLM 结构（02）引用的这个数字在这里推出来；
- 激活为什么要存：反向计算 $$\partial L / \partial W$$ 需要前向时的输入 $$X$$，所以前向的中间结果必须保留到反向；激活显存与 batch、序列长度、层数成正比，与参数量无关；激活重算（gradient checkpointing）用一次额外前向换掉这份存储；
- 用有限差分验证手推的梯度：梯度检查的方法与精度标准；
- Autograd 做了什么：动态图记录、每个算子的 backward 函数、叶子节点的 `.grad` 累加——只到理解框架行为的程度，实现在 Infra 地图 03 系列第四篇。

核心问题是：

> **不用框架，能不能手推并手写一个两层网络的反向传播，用有限差分验证到 $$10^{-6}$$？能不能由此说出为什么训练 FLOPs 是 $$6ND$$、为什么激活要存？[^q4]**

实验：纯 NumPy 实现两层 MLP 的前向、反向与梯度检查；在 MNIST 规模的数据上训到 97% 以上；统计前向与反向的浮点运算次数，验证 2 倍关系。这份代码是后面各篇实验的基座。

### 3. 激活函数：从 sigmoid 到 SwiGLU

第四篇[《深度学习基础（04）：激活函数——从 sigmoid 到 SwiGLU》](/activation-functions-from-sigmoid-to-swiglu.html)从“没有非线性时，深层 Linear 仍能折叠成一层”出发，沿 sigmoid / tanh、ReLU 及其修补、GELU / SiLU，走到 GLU 门控家族。

这一篇会覆盖：

- 饱和如何让 sigmoid 与 tanh 的梯度变小，ReLU 为什么梯度直通又可能死亡；
- Leaky ReLU、PReLU、ELU、GELU、SiLU 的公式、导数与曲线；
- GLU、ReGLU、GeGLU、SwiGLU 的门控结构，以及它对 FFN 矩阵数与宽度的影响；
- 逐元素激活的访存代价、kernel fusion，以及同一深层 MNIST MLP 上的梯度和准确率对照。

核心问题是：

> **深层网络靠什么打破线性？从 sigmoid 到 SwiGLU，激活函数改变了梯度与 FFN 的哪些部分？[^q5]**

实验：固定 Kaiming-normal 初始化和 Adam，在同一 20 层 MLP 上比较六种激活函数的初始梯度、250 步后的测试准确率与 ReLU 死亡单元。

### 4. 训练为什么不稳定：初始化、归一化与残差

第五篇讲深网络最核心的问题：信号与梯度经过很多层之后为什么会消失或爆炸，以及三十年里发展出的三种修法。

这一篇会覆盖：

- 方差的传播：一层 $$y = Wx$$ 的输出方差是输入方差的 $$n_{in} \cdot \text{Var}(w)$$ 倍；要让方差在层间不变，$$\text{Var}(w) = 1/n_{in}$$——这是 Xavier 初始化的推导；ReLU 砍掉一半，所以 Kaiming 初始化是 $$2/n_{in}$$；反向传播里同一个推导给出 $$1/n_{out}$$；
- 梯度作为 Jacobian 的乘积：$$L$$ 层网络最底层的梯度是 $$L$$ 个 Jacobian 的乘积，其谱范数的几何平均偏离 1 就指数级放大或缩小；$$0.9^{128} \approx 1.4 \times 10^{-6}$$、$$1.1^{128} \approx 2 \times 10^{5}$$；
- 归一化：BatchNorm（沿 batch 归一）、LayerNorm（沿特征归一）、RMSNorm（去掉均值中心化）；三者的公式、参数量、计算量；BatchNorm 为什么在序列模型里不好用（batch 内统计量依赖 padding 与序列长度、推理时要用 running statistics）；RMSNorm 比 LayerNorm 省了什么；
- 残差连接：每层的 Jacobian 从 $$J_l$$ 变成 $$I + J_l$$，乘积展开后有一条恒等通路，梯度不再随深度指数衰减；残差流的方差随层数线性增长，所以 GPT-2 把残差分支的初始化再除以 $$\sqrt{2L}$$，深层模型要把这个因子算进去；
- Pre-Norm 与 Post-Norm：归一化放在残差分支之前还是之后，对梯度路径的影响；Post-Norm 效果略好但需要 warmup 且深了难训，Pre-Norm 稳定、当前 LLM 几乎全用它；
- 大模型上的稳定性工具：loss spike 的常见来源、QK-norm、z-loss、$$\mu$$P 的动机（让超参数在不同宽度下可迁移）——只到概念，配方细节在 L4；
- 诊断：看每层激活的方差、每层梯度的范数分布，判断是初始化、归一化还是学习率的问题。

核心问题是：

> **一个 64 层的 MLP 不加任何技巧为什么训不动？初始化、归一化、残差三样东西各修了哪一段，缺了哪一样会怎样？[^q6]**

实验：用第三篇的 NumPy 基座搭一个 64 层 MLP，画出四种配置（随机初始化 / Kaiming / Kaiming + LayerNorm / Kaiming + LayerNorm + 残差）下各层激活方差与梯度范数随深度的曲线；观察 Pre-Norm 与 Post-Norm 在同一网络上的差别。

### 5. 优化器：从 SGD 到 AdamW 与学习率调度

第六篇讲有了梯度之后怎么更新参数。它把优化器的每一项拆开，解释各自在解决什么问题、带来什么状态与代价。

这一篇会覆盖：

- SGD 与它的噪声：mini-batch 梯度是全量梯度的无偏估计，方差与 $$1/B$$ 成正比；学习率与 batch 的 scaling 规则（线性 scaling、平方根 scaling）的来源与适用范围；
- Momentum：梯度的指数移动平均，在一致的方向上加速、在震荡的方向上抵消；Nesterov 的修正；
- Adam：一阶矩 $$m$$ 与二阶矩 $$v$$，更新量 $$m / \sqrt{v}$$ 让每个参数有自己的有效学习率；偏差修正为什么必须；$$\beta_1 = 0.9$$、$$\beta_2 = 0.95$$（LLM 常用，而不是默认的 0.999）各自意味着多长的记忆窗；$$\epsilon$$ 的作用；
- AdamW 与 $$L_2$$ 正则的区别：在 loss 里加 $$\frac{\lambda}{2}\|w\|^2$$ 会被 $$1/\sqrt{v}$$ 缩放，而 decoupled weight decay 直接对参数衰减；两者在 Adam 下不等价的推导；LLM 常用 $$\lambda = 0.1$$；
- 优化器状态的账：Adam 每参数两个 fp32 状态 8 字节，加 fp32 主权重 4 字节，是 L1 工具箱系列第四篇"16 字节 / 参数"里的 12；8-bit Adam、Adafactor 等减状态的方法各省多少；
- 学习率调度：warmup 为什么必不可少（$$v$$ 的估计在早期不可靠、更新量过大）、cosine 衰减、WSD（warmup-stable-decay）、衰减到峰值的 10% 还是 0；典型的峰值学习率量级（$$3 \times 10^{-4}$$ 级别的预训练、$$10^{-5}$$ 级别的 SFT）与它们和模型宽度的关系；
- 梯度裁剪：按全局范数裁剪到 1.0 在做什么，为什么它是 loss spike 的第一道防线；梯度范数曲线怎么读；
- 二阶方法与新优化器：Shampoo、Muon、SOPHIA 一类的动机（用更多曲率信息换更少步数），到知道它们存在、知道它们与 Adam 的差别在哪即可。

核心问题是：

> **Adam 的两个矩各在做什么？为什么 AdamW 与在 loss 里加 $$L_2$$ 不一样？warmup 为什么在 Adam 下几乎不能省？[^q7]**

实验：在同一个小网络上实现 SGD、Momentum、Adam、AdamW，画训练曲线对比；关掉 warmup 与偏差修正观察前几百步；用不同 batch 验证学习率 scaling 规则的适用范围。

### 6. 正则化与泛化：为什么参数比样本多却不过拟合

第七篇讲深度学习里最反直觉的现象——参数量远大于样本量的网络照样泛化——以及什么时候这个好运会用完。

这一篇会覆盖：

- 经典视角回顾（L2 经典机器学习系列）：容量、偏差 - 方差、过拟合；它在深网络上失效的地方——参数量早已超过样本量，按经典理论应该严重过拟合；
- double descent：测试误差随模型容量先降、再升、越过插值阈值后再降；隐式正则化——SGD 倾向找到平坦、低范数的解；
- 显式正则化：weight decay（作为高斯先验、作为有效学习率的调节）；dropout（训练时随机置零、推理时按期望缩放，等价于集成大量子网络）；早停（等价于对训练轨迹的约束）；数据增强；label smoothing；
- 为什么 LLM 预训练几乎不用 dropout：数据量远大于模型能记住的量、每个样本只见一次（一个 epoch），过拟合不是主要风险；weight decay 仍保留，主要作用变成控制参数范数与稍稍改善优化；
- 什么时候过拟合回来：SFT 几千条数据训多个 epoch 后的逐字记忆；奖励模型的过拟合与 reward hacking（L2 经典机器学习系列）；多 epoch 预训练在数据受限时的收益递减（Muennighoff 等 2023：重复到 4 个 epoch 以内几乎无损，之后收益迅速下降）；
- 泛化的诊断：训练 loss 与验证 loss 的差、验证 loss 何时开始回升、记忆检测（模型能否逐字复述训练样本）。

核心问题是：

> **一个参数量是样本量一千倍的网络为什么不过拟合？同一个网络在什么数据规模、多少个 epoch 之后会开始过拟合？[^q8]**

实验：在小数据集上扫模型宽度，画出 double descent 曲线；在同一网络上对比无正则化 / weight decay / dropout 的训练 - 验证 loss 差；用一个小语言模型对比 1 epoch 与 10 epoch 的记忆程度。

### 7. CNN：从 LeNet 到 ResNet，再到 ViT

第八篇讲卷积网络——它解决了什么问题、演化到 ResNet 留下了什么、以及 ViT 怎么把它的输入方式带进了 Transformer。

这一篇会覆盖：

- 卷积作为带约束的线性层：局部连接与参数共享，一个 $$3 \times 3 \times C_{in} \times C_{out}$$ 的卷积核相当于一个被强约束的全连接矩阵；参数量与 FLOPs 的公式，代入 ResNet-50 得到 25.6M 参数、约 4.1 GFLOPs（224×224，单次前向的乘加计）；
- 感受野与层数：感受野随层数线性增长，要看到整张图需要很多层，这是"深"的最初动机；池化与 stride；
- 五个里程碑各解决了什么：LeNet（卷积 + 池化的范式）、AlexNet（ReLU、dropout、GPU 训练）、VGG（小核堆深）、GoogLeNet（多分支与 1×1 卷积）、ResNet（残差让 152 层能训）；每一步的参数量与深度；
- ResNet 留给 Transformer 的三样东西：残差连接、归一化（BatchNorm → LayerNorm）、"堆同样的块"这个设计原则；
- 卷积的归纳偏置——平移等变性、局部性——在数据足够多时不再是优势；ViT 把图切成 16×16 的 patch、每个 patch 线性投影成一个 token、加位置编码后送进标准 Transformer；224×224 的图是 196 个 token；ViT 在小数据上不如 CNN、在大数据上反超，这是"归纳偏置 vs 数据量"的一个干净实验；
- 卷积在多模态里的残余：ViT 的 patch embedding 本身就是一个 stride 等于 kernel 的卷积；语音模型的卷积前端；扩散模型的 U-Net。

核心问题是：

> **一个 3×3 卷积核相当于多大的全连接矩阵？ResNet 的残差与 Transformer 的残差是同一个东西吗？ViT 为什么可以不用卷积？[^q9]**

实验：用 NumPy 实现二维卷积并验证它等价于一个稀疏的全连接矩阵；用 PyTorch 在 CIFAR-10 上对比一个 plain 深网络与加残差的同深度网络；实现 patch embedding，验证它与 stride 卷积输出一致。

### 8. Embedding 与表示学习：从查表到句向量

第九篇[《深度学习基础（09）：Embedding 与表示学习——从查表到句向量》](/embedding-and-representation-learning.html)从离散 token 编号出发，解释模型怎样查表得到稠密向量，以及训练目标怎样把有用的几何结构写进向量空间。

这一篇会覆盖：

- one-hot 与 embedding 矩阵乘法为何等价于按行查表；`nn.Embedding` 的梯度只更新被查到的行；
- GPT-2 small 的输入 embedding 表有 38,597,376 个参数，占模型参数约 31.0%；输入与输出词表矩阵是否共享；
- skip-gram、CBOW、full softmax 与 negative sampling；用二维例子手算正负样本更新；
- SGNS 与 shifted PMI 的关系、静态词向量的能力边界，以及表示从预测、重建到对比学习的发展；
- 完整 text8 上的 CPU Word2Vec 案例：比较随机初始化与训练后近邻、排除原词后的类比，并用 PCA 投影向量。

核心问题是：

> **一个整数 token 怎样变成可学习的向量？Word2Vec 的邻居与类比从何而来，为什么现代 Transformer 还需要上下文表示？[^q10]**

实验：在完整 text8 的 17,005,207 个词上训练 100 维 skip-gram negative sampling；`min_count=5` 后词表 71,290，频率抽样后保留 5,597,324 个词。窗口半径 1–5、5 个负样本、3 个 epoch 共 6,153 步，CPU 耗时 310.0 秒；对照随机初始化近邻与训练后近邻，并把选定词向量投影到二维 PCA 平面。

### 9. RNN：从 LSTM 到 attention 的诞生

第十篇讲循环网络这条线。它是 attention 的直接来源：理解 RNN 的失败，才理解 Transformer 赢在哪。

这一篇会覆盖：

- 序列建模的问题设定：变长输入、共享参数、依赖历史；RNN 的状态更新 $$h_t = f(W h_{t-1} + U x_t)$$；
- 时间上的反向传播（BPTT）：梯度沿时间回传是同一个矩阵 $$W$$ 的连乘，第五篇的 Jacobian 乘积在时间维上的版本；梯度消失让 RNN 记不住 20 步之外的东西；梯度爆炸靠裁剪（梯度裁剪最初就是为 RNN 发明的）；
- LSTM 与 GRU：用门控让状态可以"加法式"地穿过时间，$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t$$ 里的 $$f_t \odot c_{t-1}$$ 是残差在时间上的形态；门的参数量；
- seq2seq 与它的瓶颈：encoder 把整个源序列压进一个固定长度的向量，decoder 从它出发生成；长句子上质量下降，因为一个向量装不下；
- attention 的诞生（Bahdanau 等 2014）：decoder 每一步对 encoder 的所有隐状态算一个权重、加权求和，绕过固定向量瓶颈；这就是 $$\text{softmax}(q^T k) v$$ 的最初形式，只是 $$q, k, v$$ 还没有分开命名；
- RNN 的两个致命缺点与 Transformer 的回答：无法并行（$$h_t$$ 依赖 $$h_{t-1}$$，序列长度决定串行步数）与长依赖衰减（路径长度 $$O(n)$$）；self-attention 把任意两个位置的路径长度变成 $$O(1)$$、把序列维完全并行，代价是 $$O(n^2)$$ 的算量与 KV——现代 LLM 结构系列的成本线从这里开始；
- RNN 的回声：状态空间模型（Mamba 一类）与线性 attention 试图找回 RNN 的 $$O(n)$$ 推理成本，知道它们在权衡什么即可。

核心问题是：

> **RNN 为什么记不住 20 步之外的东西？LSTM 的遗忘门与残差连接是什么关系？attention 最初是为了解决什么问题被发明的，它又为什么最终取代了发明它的 RNN？[^q11]**

实验：用 NumPy 实现一个 RNN 与 LSTM，在"记住第一个 token 并在 $$T$$ 步后输出"的任务上扫 $$T$$，画出两者能记住的最长距离；实现 Bahdanau attention 并可视化对齐矩阵；测量 RNN 与 self-attention 在同一序列长度下的前向时间随长度的变化。

### 10. 系列总结与通关自测

最后一篇不讲新内容：把十篇正文压成一张「问题 → 结论 → 必记数字」的表并逐篇回顾，拎出贯穿全系列的几条线与常见误区，然后给一套通关自测——十道判断与计算、五道跨篇综合、若干道面试题，以及激活函数和 Embedding 的四道加试题，答案各自折叠，附「读过 / 掌握 / 能教人」的判据。各篇末尾的自测检验的是一篇读懂了没有，这一篇检验的是十篇能不能连起来用；读完正文再做。

## 贯穿全系列的实践线

本系列的贯穿物是**一份几百行的 NumPy 小框架和一组建立在它之上的实验**。实验从第一篇的 XOR 案例开始；NumPy 框架从第三篇的两层 MLP 开始，逐篇增加：

| 篇 | 框架里加的东西 |
|---|---|
| 第一篇 | 感知机、ADALINE 与 MLP 的 XOR 决策边界及训练曲线 |
| 第三篇 | Linear · ReLU · softmax-CE 的前向与反向<br/>梯度检查<br/>FLOPs 计数 |
| 第四篇 | 六种激活函数<br/>逐层梯度与准确率统计<br/>ReLU 死亡单元演示 |
| 第五篇 | 深层堆叠<br/>Kaiming 初始化<br/>LayerNorm / RMSNorm<br/>残差块<br/>逐层方差与梯度范数统计 |
| 第六篇 | SGD · Momentum · Adam · AdamW<br/>学习率调度<br/>梯度裁剪与梯度累积 |
| 第七篇 | dropout<br/>weight decay<br/>训练 - 验证曲线<br/>宽度扫描 |
| 第八篇 | Conv2d 与它的全连接等价形式<br/>patch embedding（PyTorch 对照 CIFAR-10） |
| 第九篇 | one-hot 查表<br/>skip-gram negative sampling<br/>text8 近邻、类比与 PCA |
| 第十篇 | RNN 单元 · LSTM 单元 · BPTT<br/>Bahdanau attention |
| 案例脚本 | 原有 `case_01` … `case_06` 案例脚本<br/>新增 `case_activation_functions.py`、`case_embedding_word2vec.py`<br/>`tools/paper_figures.py` 从原论文 PDF 裁结构图 |

Table: 各篇在 NumPy 小框架里加的东西

每篇另有一节**来龙去脉**（这一样东西是谁、为解决什么问题发明的，前一代败在哪）、一章**案例**（把本篇的实验放回一个完整的问题里：问题 → 思路 → 代码 → 效果 → 落地还差什么，曲线画出来）和**原论文的结构图**（LeNet-5、AlexNet、VGG、ResNet 残差块、ViT、LSTM 细胞、seq2seq、Bahdanau attention、Pre-LN vs Post-LN、Adam 的算法框、dropout、double descent——直接引用原图并注明来源）：

| 篇 | 案例 | 数字 |
|---|---|---|
| 第一篇 | XOR：感知机、ADALINE 与两层 MLP | AND 第 6 轮零误分<br/>XOR 感知机最少 3 次误分<br/>ADALINE MSE 0.25<br/>H=4 为 20/20
| 第三篇 | MNIST 手写数字，120 行 NumPy 从零 | 15 个 epoch 97.6%<br/>与 KNN 2.95% / SVM 1.43% 同一份数据 |
| 第四篇 | 20 层 MLP 对比六种激活函数，三个随机种子 | sigmoid 准确率 10.64%<br/>tanh 90.48%<br/>ReLU 91.88%<br/>SiLU 92.88%<br/>ReLU 死亡单元 12.26% |
| 第五篇 | 把 MLP 加深到 64 层，七种接法 | 只有 Pre-Norm 在两个学习率下都能训<br/>逐层激活 std / 梯度范数曲线 |
| 第六篇 | 同一个 MLP，四种优化器 × 五个学习率<br/>64 层网络的 warmup | 最优学习率差 300 倍、成绩差不到半个点<br/>lr 1e-2 无 warmup loss 冲到 4.59 |
| 第七篇 | 1,000 张图的过拟合与三种解药<br/>宽度扫描 | 训练 loss 归零、测试 loss 掉头<br/>double descent 的尖峰在宽度 8–16 |
| 第八篇 | 复现 LeNet-5 | 61,706 个参数、5 个 epoch、0.82%——MLP 参数的 1/3、错误率的 1/3<br/>第一层学出边缘检测器 |
| 第九篇 | 完整 text8 的 skip-gram negative sampling | 17,005,207 个词<br/>`min_count=5` 后词表 71,290，subsampling 后 5,597,324 个词<br/>100 维、3 epochs、CPU 310 秒 |
| 第十篇 | 字符级 LSTM 写莎士比亚，与 nanoGPT 同预算 | val 1.71 vs 1.66<br/>倒序任务的对齐矩阵 |

Table: 各篇的案例

大多数实验在 CPU 上几分钟内跑完（第八、第九、第十篇的 PyTorch 案例在 Apple 芯片 / GPU 上更快），没有 GPU 不影响。框架的目的不是替代 PyTorch，而是让每一个训练现象都能在自己写的、每一行都懂的代码里复现一次；之后回到 PyTorch，`loss.backward()` 与 `optimizer.step()` 就不再是黑盒。

与它平行的源码与资料阅读线：

| 篇 | 论文与资料 |
|---|---|
| 第一篇 | McCulloch & Pitts 1943 [神经活动的逻辑演算](https://doi.org/10.1007/BF02478259) · Rosenblatt 1958 [感知机](https://doi.org/10.1037/h0042519) · Widrow & Hoff 1960 [ADALINE / LMS](https://doi.org/10.1109/IREWES.1960.4503255) · Minsky & Papert 1969 [*Perceptrons*](https://mitpress.mit.edu/9780262630221/perceptrons/) · Hinton 等 2006 [DBN](https://www.cs.toronto.edu/~hinton/absps/ncfast.pdf) · Vaswani 等 2017 [Transformer](https://arxiv.org/abs/1706.03762) |
| 第三篇 | Rumelhart 等 1986 [*Learning representations by back-propagating errors*](https://www.nature.com/articles/323533a0) · Karpathy [micrograd](https://github.com/karpathy/micrograd) · PyTorch autograd 文档 [*How autograd encodes the history*](https://pytorch.org/docs/stable/notes/autograd.html) |
| 第四篇 | Nair & Hinton 2010 [ReLU](https://icml.cc/Conferences/2010/papers/432.pdf) · He 等 2015 [PReLU](https://arxiv.org/abs/1502.01852) · Hendrycks & Gimpel 2016 [GELU](https://arxiv.org/abs/1606.08415) · Shazeer 2020 [GLU variants](https://arxiv.org/abs/2002.05202) |
| 第五篇 | Glorot & Bengio 2010 [Xavier](https://proceedings.mlr.press/v9/glorot10a.html) · He 等 2015 [Kaiming](https://arxiv.org/abs/1502.01852) · Ioffe & Szegedy 2015 [BatchNorm](https://arxiv.org/abs/1502.03167) · Ba 等 2016 [LayerNorm](https://arxiv.org/abs/1607.06450) · Zhang & Sennrich 2019 [RMSNorm](https://arxiv.org/abs/1910.07467) · Xiong 等 2020 [Pre-LN](https://arxiv.org/abs/2002.04745) |
| 第六篇 | Kingma & Ba 2014 [Adam](https://arxiv.org/abs/1412.6980) · Loshchilov & Hutter 2017 [AdamW](https://arxiv.org/abs/1711.05101) · Goyal 等 2017 [线性 scaling](https://arxiv.org/abs/1706.02677) · Hu 等 2024 [MiniCPM，WSD](https://arxiv.org/abs/2404.06395) |
| 第七篇 | Srivastava 等 2014 [dropout](https://jmlr.org/papers/v15/srivastava14a.html) · Zhang 等 2017 [*Understanding deep learning requires rethinking generalization*](https://arxiv.org/abs/1611.03530) · Nakkiran 等 2019 [double descent](https://arxiv.org/abs/1912.02292) · Muennighoff 等 2023 [数据受限的 scaling](https://arxiv.org/abs/2305.16264) |
| 第八篇 | LeCun 等 1998 [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) · Krizhevsky 等 2012 [AlexNet](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) · He 等 2015 [ResNet](https://arxiv.org/abs/1512.03385) · Dosovitskiy 等 2020 [ViT](https://arxiv.org/abs/2010.11929) |
| 第九篇 | Mikolov 等 2013 [Word2Vec](https://arxiv.org/abs/1310.4546) · Levy & Goldberg 2014 [SGNS 与 shifted PMI](https://arxiv.org/abs/1402.3722) · Radford 等 2019 [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) |
| 第十篇 | Hochreiter & Schmidhuber 1997 [LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf) · Sutskever 等 2014 [seq2seq](https://arxiv.org/abs/1409.3215) · Bahdanau 等 2014 [attention](https://arxiv.org/abs/1409.0473) · Vaswani 等 2017 [Transformer](https://arxiv.org/abs/1706.03762) 第 1–2 节 |

Table: 各篇平行的论文与资料阅读线


## 前置要求与说明

### 前置要求

- [L0 数学系列](/math-for-ai-algorithm-engineers.html)的四个分支：链式法则、Jacobian、期望与方差、范数与谱范数的概念；
- [L1 工具箱系列](/tooling-for-ai-algorithm-engineers.html)的 NumPy 形状与广播，会写二十行 PyTorch 训练循环；
- [L2 经典机器学习系列](/classical-machine-learning-in-the-llm-era.html)的过拟合、偏差 - 方差、正则化的概念。

不要求：了解任何具体的初始化、归一化、优化器方法；有 GPU。

### 版本与基线

- 实验基线：第二至第七篇用 MNIST（几万样本、几十万到几百万参数的 MLP 与小卷积网络），第六篇的语言模型用 Python 标准库源码做字符级语料，第九篇用官方完整 text8 语料，第十篇用合成的序列任务；所有数字在普通笔记本 CPU 上几分钟到十几分钟内可复现；
- 配套代码：每篇一个脚本，连同系列贯穿的迷你框架 `dlf/` 与作者机器上的运行输出，在 [ai-learning-labs/deep-learning-foundations](https://github.com/arganzheng/ai-learning-labs/tree/main/deep-learning-foundations)；
- 算账时引用的真实网络：ResNet-50（25.6M 参数）、GPT-2（124M 到 1.5B，$$\sqrt{2L}$$ 初始化缩放）、Llama-3-8B（32 层、初始化标准差 0.02、AdamW $$\beta_2 = 0.95$$、weight decay 0.1、裁剪 1.0），超参数取自公开论文与技术报告；
- 论文引用以第一作者与年份标注。


## 章节目录

1. [深度模型演进史——从感知机到 Transformer](/evolution-of-deep-models-from-perceptron-to-transformer.html)
2. [反向传播：手推一个两层网络](/backpropagation-by-hand.html)
3. [激活函数：从 sigmoid 到 SwiGLU](/activation-functions-from-sigmoid-to-swiglu.html)
4. [训练为什么不稳定：初始化、归一化与残差](/initialization-normalization-and-residual.html)
5. [优化器：从 SGD 到 AdamW 与学习率调度](/optimizers-from-sgd-to-adamw.html)
6. [正则化与泛化：为什么参数比样本多却不过拟合](/regularization-and-generalization.html)
7. [CNN：从 LeNet 到 ResNet，再到 ViT](/cnn-from-lenet-to-resnet-and-vit.html)
8. [Embedding 与表示学习：从查表到句向量](/embedding-and-representation-learning.html)
9. [RNN：从 LSTM 到 attention 的诞生](/rnn-lstm-and-the-birth-of-attention.html)
10. [系列总结与通关自测](/deep-learning-foundations-series-recap-and-self-test.html)


## 最终目标

读完这套系列之后，面对一个训练中的网络，读者应该能够回答：

| 问题 | 答案在 |
|---|---|
| loss 不降，是梯度没传到还是学习率不对？[^q12] | 第三篇：梯度检查<br/>第五篇：逐层梯度与初始化<br/>第六篇：学习率与 warmup |
| sigmoid 深层梯度为什么消失？ReLU 有什么不同？[^q13] | 第四篇：激活函数导数与死亡单元 |
| loss 爆成 NaN，先看哪个数字？[^q14] | 第五篇：逐层激活方差、梯度范数与裁剪 |
| 这个网络为什么要 Pre-Norm？初始化为什么是 0.02？[^q15] | 第五篇：方差传播与残差流 |
| AdamW 的每个超参数在控制什么？状态占多少显存？[^q16] | 第六篇 |
| SFT 训了 5 个 epoch，模型开始背答案，怎么判断、怎么修？[^q17] | 第七篇 |
| 预训练为什么一个 epoch、不用 dropout？[^q18] | 第七篇 |
| ViT 的 patch embedding 与卷积是什么关系？[^q19] | 第八篇 |
| token id 如何变成向量，静态与上下文表示有什么不同？[^q20] | 第九篇 |
| attention 为什么能取代 RNN？代价是什么？[^q21] | 第十篇 |

Table: 读完深度学习基础系列后应能回答的问题

最终目标是三种能力：

1. **推导能力**：面对一个训练现象，能写出它背后的公式（Jacobian 乘积、更新量的量级、期望等价）；
2. **诊断能力**：面对一条不对的曲线，知道看哪个数字、动哪个开关、预期改善多少；
3. **衔接能力**：读 Transformer 与 LLM 的技术报告时，每一个关于初始化、归一化、优化器、稳定性的段落都能对到本系列的某一节。

这一层是 L4 之前的最后一层基础。它不新，但没有它，后面每一层的"为什么"都只能靠记。

[^q0]: 能训是因为几件事让信号与梯度在逐层传递中保持尺度：激活函数提供非线性并改变 Jacobian 因子，初始化（Xavier / Kaiming）控制每层方差，归一化（BatchNorm / LayerNorm / RMSNorm）把激活钉回单位尺度，残差（每层 Jacobian 变成 $$I+J_l$$）给梯度一条恒等通路——128 层、每层谱范数 0.9 的无残差网络底层梯度只有顶层的 $$1.4\times10^{-6}$$。不能训的时候通常是其中一项失效——激活饱和、初始化尺度错、没有 warmup 让 Adam 在前几百步方差估计不准、Post-Norm 深层不稳、学习率过大。该看的数字：**每层梯度范数**随深度的分布、**激活的均值 / 方差**逐层变化、**全局梯度范数**与 **loss 曲线**（spike 前梯度范数先涨）、attention logit 的最大值；第四篇对照激活导数与训练梯度，第五篇检查初始化 / 归一化 / 残差，第六篇讨论 warmup、学习率与裁剪。
[^q1]: 这是方差传播问题：残差流的方差随层数线性增长，第 $$l$$ 个残差分支要按 $$1/\sqrt{2L}$$ 缩放才不抢主干道；初始化标准差按宽度量级取（768 维约 0.05，GPT-2 取 0.02），推导与数字在[第五篇「一、方差怎样在层间传播」与「七、放到 Transformer 上」](/initialization-normalization-and-residual.html)。
[^q2]: Adam 的二阶矩 $$v$$ 在前几十步只由极少样本估计，偏差修正后仍噪声极大，$$1/\sqrt v$$ 会把步长放大几个数量级，训练开头就被打坏；warmup 让 $$v$$ 先积累样本再放开步长，几乎不能省。见[第六篇「六、学习率调度 · 1. Warmup 为什么必不可少」](/optimizers-from-sgd-to-adamw.html)。
[^q3]: dropout 训练时按 $$1/(1-p)$$ 放大保留单元，期望与推理等价——它用缩小有效容量换泛化；预训练在数据主导区间（$$N/D \ll 1$$）、一个 epoch 数据尚且看不完，这个交换不划算。见[第七篇「五、显式正则化」](/regularization-and-generalization.html)。
[^q4]: 能，第三篇就是这件事：两层 MLP 的前向 $$h=\sigma(W_1x+b_1)$$、$$\hat y=W_2h+b_2$$，反向按链式法则从 loss 往回：$$\partial L/\partial W_2 = \delta_2 h^\top$$、$$\delta_1 = (W_2^\top\delta_2)\odot\sigma'(z_1)$$、$$\partial L/\partial W_1=\delta_1 x^\top$$，用中心差分 $$(L(\theta+\epsilon)-L(\theta-\epsilon))/2\epsilon$$ 逐分量对拍，相对误差到 $$10^{-6}$$ 量级（float64）。由此可见：前向每个参数做一次乘加（$$2N$$ FLOPs / token），反向要算两样——对输入的梯度 $$W^\top\delta$$ 与对权重的梯度 $$\delta x^\top$$，各 $$2N$$，所以反向 = 2 × 前向，训练一个 token 共 $$6N$$，$$D$$ 个 token 就是 $$6ND$$；而 $$\partial L/\partial W = \delta\, x^\top$$ 需要前向的输入 $$x$$（激活），所以激活必须存到反向用完为止，这就是激活显存与 $$B\times T$$ 成正比、以及激活重算（checkpointing）用算力换显存的根源。
[^q5]: 没有非线性时 $$L$$ 层线性堆叠等价于一层，深度失去意义；非线性决定梯度形状：sigmoid 饱和使 $$\sigma' \le 0.25$$ 连乘消失，ReLU 正半轴导数为 1 但会死单元，GELU / SiLU 平滑门控，SwiGLU 把门控搬进 FFN。见[第四篇](/activation-functions-from-sigmoid-to-swiglu.html)。
[^q6]: 前向方差与反向梯度都是逐层连乘，因子偏离 1 六十四次后趋零或爆炸；初始化只修 $$t=0$$，归一化把前向方差拉回但修不了反向，残差给梯度一条 $$I+J$$ 恒等通路——三样一起用，Pre-Norm 是当前摆法。见[第五篇](/initialization-normalization-and-residual.html)。
[^q7]: 一阶矩 $$m$$ 是梯度的指数移动平均（动量，平滑方向）；二阶矩 $$v$$ 是梯度平方的指数移动平均，用 $$1/\sqrt{v}$$ 给每个参数单独缩放步长——梯度常年很小的参数步子大、常年很大的步子小，所以 Adam 对特征尺度与学习率不敏感。AdamW 与在 loss 里加 $$L_2$$ 不一样，因为 $$L_2$$ 项的梯度 $$\lambda w$$ 会进入 $$m$$ 与 $$v$$、被 $$1/\sqrt v$$ 一起缩放——梯度大的参数几乎不衰减；AdamW 把 $$-\eta\lambda w$$ 直接加在参数更新上、绕开自适应缩放，衰减对所有参数一致，这才是真正的 weight decay（在 SGD 下两者才等价）。warmup 在 Adam 下几乎不能省，因为前几十步 $$v$$ 只由极少样本估计、偏差修正后仍噪声极大，$$1/\sqrt v$$ 会把某些参数的步长放大几个数量级，训练在开头就被打坏（RAdam 的分析即此）；线性 warmup 让 $$v$$ 先积累足够样本。
[^q8]: 因为过参数化网络在梯度下降下有**隐式正则化**：在能拟合训练集的无数个解里，SGD 从小初始化出发会找到范数小、「简单」的那个（线性模型里就是最小范数解），泛化误差随宽度增加先升后降——double descent：在参数量 ≈ 样本量的插值阈值附近最差，继续加宽反而变好。会开始过拟合的条件是**数据规模相对任务太小、或重复 epoch 太多**：第五篇的实验里，小语言模型训 1 epoch 的训练 / 验证 loss 几乎重合，训到 10 epoch 时验证 loss 回升、模型开始逐字记忆训练文本；预训练的实践是数据过一遍（≤ 4 epoch 以内收益接近新数据，之后急剧下降），weight decay 与 dropout 把训练 - 验证差距压小但不改变这个趋势。
[^q9]: 3×3 卷积相当于只含 9 个非零权重、在全图共享的稀疏全连接；两种残差是同一条 $$I+J$$ 恒等通路；ViT 把图切成 patch 当 token，数据足够时卷积先验可以从数据里学出来。见[第八篇](/cnn-from-lenet-to-resnet-and-vit.html)。
[^q10]: token id 查 embedding 表的一行就是向量；Word2Vec 用「预测邻居」的训练目标把共现结构压进几何（近义靠近、类比成差）；静态向量不分上下文，Transformer 用整句算出随语境变化的表示。见[第九篇](/embedding-and-representation-learning.html)。
[^q11]: 时间反传是同一个 $$W$$ 的连乘，20 步外梯度量级约千分之四，信号传不到；遗忘门让细胞状态乘上接近 1 的系数，是时间上的残差通路（$$b_f$$ 要初始化为 1）；attention 为绕过 seq2seq 的固定向量瓶颈而发明，后来发现可以直接交换任意两个位置的信息、且能并行，于是循环本身被去掉。见[第十篇](/rnn-lstm-and-the-birth-of-attention.html)。
[^q12]: 先做梯度检查（相对误差 $$<10^{-6}$$）与逐层梯度范数：逐层范数断说明梯度没传到（第三篇方法、第五篇修结构）；范数正常但不降是学习率 / warmup 问题（第六篇）。
[^q13]: sigmoid 导数最大 0.25，$$L$$ 层连乘 $$0.25^L$$ 指数消失；ReLU 正半轴导数为 1 不衰减，但负半轴恒零会死单元。见[第四篇](/activation-functions-from-sigmoid-to-swiglu.html)。
[^q14]: 先看全局梯度范数是否 spike、各层激活 RMS 在哪一层开始爆，再看 warmup 是否缺失；对应开关是初始化 / 归一化 / 残差（第五篇）与梯度裁剪、warmup（第六篇）。
[^q15]: Pre-Norm 把归一化放残差分支内、主干道保持恒等，顶层与底层梯度均衡；0.02 是宽度量级上的初始化标准差。见[第五篇「七、放到 Transformer 上」](/initialization-normalization-and-residual.html)。
[^q16]: $$\beta_1$$ 控制动量记忆、$$\beta_2$$ 控制二阶矩记忆窗、$$\lambda$$ 是解耦的 weight decay、$$\eta$$ 是峰值步长；AdamW 每参数存 $$m$$、$$v$$ 两个状态共 8 字节。见[第六篇](/optimizers-from-sgd-to-adamw.html)。
[^q17]: 用 held-out loss 拐点与记忆探针（训练集 prompt 续写与原答案逐字比对）判断；修法是降 epoch、降学习率、按拐点早停或补数据。见[第七篇「七、案例」](/regularization-and-generalization.html)。
[^q18]: 预训练 $$N/D \ll 1$$ 在数据主导区间，模型背不下数据，过拟合不是主要风险；一个 epoch 都没看完的数据重复利用价值低，dropout 只缩小有效容量。见[第七篇](/regularization-and-generalization.html)。
[^q19]: patch embedding 是步长等于核大小的卷积特例（Conv2d，k=s=patch），一张 224² 图切成 196 个 token。见[第八篇](/cnn-from-lenet-to-resnet-and-vit.html)。
[^q20]: token id 直接查表得向量；静态向量每个词固定一行，上下文表示由整句经 self-attention 算出、随语境变化。见[第九篇](/embedding-and-representation-learning.html)。
[^q21]: RNN 的时间反传使远处梯度消失且逐步串行难并行；attention 直接对任意两个位置做交互、整步可并行，代价是 $$O(T^2d)$$。见[第十篇](/rnn-lstm-and-the-birth-of-attention.html)。
