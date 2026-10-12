---
layout: post
series: deep-learning-foundations
title: "深度学习基础（01）：深度模型演进史——从感知机到 Transformer"
subtitle: "A History of Deep Models: From the Perceptron to the Transformer"
date: 2026-03-23 20:00:00
tags: [AI, Deep Learning, LLM]
catalog: true
---

Transformer 不是突然出现的。它接过了感知机、反向传播、卷积、Embedding、循环网络与 attention 留下的问题。本篇沿时间线看这些结构为什么出现；公式推导和可复现实验放在后续文章。

> **为什么单层感知机会遇到 XOR？多层网络何时变得可训练？Transformer 又从哪些旧结构中继承了组件？**

## 一、总览：三起两落

![1943 年到 Transformer 的模型演进时间线：三次兴起、两次寒冬及 2020 年后的大语言模型](/img/in-post/dl-evolution-timeline.svg)

神经网络的发展不是平滑上升。感知机带来第一次热潮，能力边界、资金收缩与英国的研究评估共同构成第一次寒冬；反向传播带来第二次兴起，但算力、数据与训练稳定性仍有限；2006 年逐层预训练重燃深层网络研究，随后 GPU、ImageNet 与更可靠的训练方法让深度学习进入第三次兴起。

三次兴起、两次寒冬是对研究热度的概括，不是五个边界精确的年份。Lighthill 的报告发表于 1973 年前后的英国政策讨论中；报告原稿标注为 1972 年 7 月，不能把寒冬归因于单一本书或一份报告。[Lighthill 原报告](https://www.chilton-computing.org.uk/inf/literature/reports/lighthill_report/p001.htm)

本文讲历史与动机；感知机、ADALINE 和 XOR 先用小例子建立直觉，反向传播、激活、初始化、优化器和各类结构的推导留给系列后续篇章。

| 章 | 主题 | 要回答的问题 |
|---|---|---|
| 一 | 总览：三起两落 | 从 1943 到 2017 的演进时间线、三次兴起与两次寒冬；本文与系列各篇如何分工？ |
| 二 | 人工神经元与感知机 | 固定阈值单元如何变成可学习的分类器？ |
| 三 | ADALINE 与 delta 规则 | 为什么误差要在阈值前计算？ |
| 四 | XOR 与第一次寒冬 | 单层线性边界能表示什么？ |
| 五 | 多层网络回归 | 隐藏层与反向传播带来了什么？ |
| 六 | 第二次寒冬 | 为什么 SVM 更受青睐？ |
| 七 | 2006 年逐层预训练 | 深网络如何重新变得可训练？ |
| 八 | 结构线 | 卷积、Embedding、循环和 attention 各留下什么？ |
| 九 | 对照表 | 不同结构交换了哪些假设？ |
| 十 | 案例 | 感知机、ADALINE 与 MLP 在 XOR 上表现如何？ |
| 十一 | 本文小结 | 哪些历史结论值得带走？ |
| 十二 | 自测 | 能否解释模型演进的关键转折？ |

## 二、人工神经元与感知机

1943 年，McCulloch 与 Pitts 把神经元抽象成二值阈值单元：输入加权求和，超过阈值就输出 1，否则输出 0。连接权重由模型外部指定；这个单元本身不学习。[原论文](https://doi.org/10.1007/BF02478259)

1950 年，图灵提出以模仿游戏讨论机器能否表现出类似人类的智能；1956 年的 Dartmouth 暑期研讨会则把“人工智能”作为研究议程提出。[图灵原文](https://doi.org/10.1093/mind/LIX.236.433)；[Dartmouth 提案](http://jmc.stanford.edu/articles/dartmouth/dartmouth.pdf)

Rosenblatt 的感知机把权重变成可学习参数。1957 年的技术报告描述了早期系统，1958 年的论文给出概率模型。对标签 $$y\in\{0,1\}$$，预测为 $$\hat y$$，输入为 $$x$$ 时，犯错后沿误差方向更新：

$$
w \leftarrow w + \eta (y-\hat y)x,\qquad b\leftarrow b+\eta(y-\hat y)
$$

[1957 年感知机报告](https://archive.org/details/bitsavers_cornellrepPerceptron1957_2026182)；[1958 年论文](https://doi.org/10.1037/h0042519)

Novikoff 1962 年的收敛定理给出了条件与上界：若数据可被间隔 $$\gamma$$ 的超平面线性分开，且每个输入满足 $$\|x\|\le R$$，感知机至多犯 $$ (R/\gamma)^2 $$ 次错。[原论文](https://cs.uwaterloo.ca/~y328yu/classics/novikoff.pdf)

直觉是：间隔越大、样本尺度越小，分类边界越容易被一次次错误更新推到正确位置；这个结论不保证 XOR 这样的不可线性分数据能收敛。

## 三、ADALINE 与 delta 规则

Widrow 与 Hoff 在 1960 年提出 ADALINE（adaptive linear element）。它先计算线性输出 $$z=w^\top x+b$$，再按需要阈值化；训练误差却在阈值前的 $$z$$ 上计算。平方误差的梯度下降就是 LMS（least mean squares）规则。[原论文](https://isl.stanford.edu/~widrow/papers/c1960adaptiveswitching.pdf)

感知机只在分类错误时更新离散预测；ADALINE 每步都根据线性输出与目标的差更新。前者是 mistake-driven 规则，后者明确沿着可微损失的梯度走。这条“定义 loss → 求梯度 → 更新参数”的路径，正是 `loss.backward() + optimizer.step()` 的早期形式。

## 四、XOR 与第一次寒冬

设一个阈值单元在 $$w_1x_1+w_2x_2+b>0$$ 时输出 1。XOR 的四个标签要求：

$$
b\le 0,\quad w_1+b>0,\quad w_2+b>0,\quad w_1+w_2+b\le 0.
$$

中间两个不等式相加，得 $$w_1+w_2+2b>0$$。因为 $$b\le0$$，这又要求 $$w_1+w_2+b>-b\ge0$$，与最后一个不等式矛盾。一个线性阈值单元无法同时满足四个标签。

Minsky 与 Papert 在 1969 年的《Perceptrons》系统分析了感知机的能力边界，特别是当时模型对某些全局结构性质的限制。[MIT Press 书目](https://mitpress.mit.edu/9780262630221/perceptrons/)

第一次寒冬有多种背景：早期系统的能力与宣传预期有差距，资金与研究优先级变化；英国的 Lighthill 评估及 1973 年公开讨论也影响了当地人工智能研究资助。不能把寒冬写成《Perceptrons》单独造成的结果。[Lighthill 原报告与讨论记录](https://www.chilton-computing.org.uk/inf/literature/reports/lighthill_report/p001.htm)

## 五、多层网络回归

两层隐藏单元已经能表示 XOR。令阈值函数 $$H(z)=1$$（当 $$z\ge0$$），构造隐藏层：

$$
h_1=H(x_1+x_2-0.5)\quad(\mathrm{OR}),\qquad
h_2=H(-x_1-x_2+1.5)\quad(\mathrm{NAND}),
$$

输出层为 $$\hat y=H(h_1+h_2-1.5)$$，也就是对 OR 与 NAND 的结果再做 AND。四个输入逐一代入，输出正好是 XOR。这个例子证明“有隐藏层可以表示”，并不说明梯度下降一定找到这组权重。

Rumelhart、Hinton 与 Williams 在 1986 年把反向传播用于多层网络学习内部表示。[Nature 原文](https://doi.org/10.1038/323533a0) 推导和手算放在《[深度学习基础（02）：反向传播——手推一个两层网络](/backpropagation-by-hand.html)》。

Cybenko（1989）证明，具有 sigmoid 激活的单隐藏层网络在适当条件下可一致逼近紧集上的连续函数；Hornik、Stinchcombe 与 White（1989）给出更一般的逼近结果。[Cybenko 原文](https://doi.org/10.1007/BF02551274)；[Hornik 等原文](https://doi.org/10.1016/0893-6080(89)90020-8)

万能逼近定理说的是存在性：某些函数可以由足够宽的网络表示。它没有告诉我们需要多宽、训练算法是否找到这些参数、有限数据上是否泛化。**能表示 ≠ 能学会。**

## 六、第二次寒冬

1990 年代到 2000 年代初，支持向量机（SVM）成为许多小样本任务的有力选择。凸优化目标能给出全局最优解；核技巧把特征映射隐式放进内积；最大间隔也带来清楚的泛化分析。小数据下不必训练庞大的深层参数模型。[Cortes 与 Vapnik 原文](https://doi.org/10.1007/BF00994018)；细节见[《支持向量机与核方法》](/svm-and-kernel-methods.html)。

当时深层网络也很难优化：sigmoid 饱和导致反向梯度逐层变小，初始化不当会破坏层间尺度；标注数据少、CPU 计算慢，训练更深模型成本高。激活饱和见《[深度学习基础（03）：激活函数——从 sigmoid 到 SwiGLU](/activation-functions-from-sigmoid-to-swiglu.html)》；方差传播与初始化见《[深度学习基础（04）：训练为什么不稳定——初始化、归一化与残差](/initialization-normalization-and-residual.html)》。

## 七、2006 年逐层预训练

Hinton、Osindero 与 Teh 在 2006 年提出深度信念网络（DBN）的贪心逐层训练：逐层训练受限玻尔兹曼机（RBM），再微调整个网络。[论文](https://doi.org/10.1162/neco.2006.18.7.1527) RBM 是两个可见层、隐藏层之间对称连接的能量模型；对比散度（contrastive divergence）是常用的近似训练算法，这里不展开推导。[对比散度原文](https://doi.org/10.1162/089976602760128018)

同年，Hinton 与 Salakhutdinov 在 Science 展示了用深层自编码器逐层预训练降维；Bengio 等人比较了逐层无监督预训练与深网络训练。[Science 原文](https://doi.org/10.1126/science.1127647)；[Bengio 等论文](https://proceedings.neurips.cc/paper/2006/file/5da713a690c067105aeb2fae32403405-Paper.pdf)

这条路线后来不再是标准的训练前置步骤。ReLU 减轻了饱和问题，Glorot / He 初始化改善了层间尺度，dropout 提供了正则化；GPU 与 ImageNet 让大规模端到端监督训练成为现实。AlexNet 在 ILSVRC-2012 的 top-5 错误率为 15.3%，第二名为 26.2%；论文报告以随机初始化的卷积网络监督训练，并未采用逐层 RBM 预训练。[Nair 与 Hinton（2010）](https://proceedings.mlr.press/v9/nair10a.html)；[Glorot 与 Bengio（2010）](https://proceedings.mlr.press/v9/glorot10a.html)；[He 等（2015）](https://arxiv.org/abs/1502.01852)；[dropout 原文](https://jmlr.org/papers/v15/srivastava14a.html)；[AlexNet 原文](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

今天 LLM 的“预训练”保留了先学通用表示、再微调的思路，但通常是整个模型做自监督目标训练；它不是先逐层训练 RBM 再堆叠的那套层级预训练。

## 八、结构线

### 1. 从 Neocognitron 到 ResNet

Neocognitron 将局部感受野与层级特征组合起来；LeNet 把卷积用于手写字符识别；AlexNet 借助 GPU 与大规模图像数据证明深卷积网络的效果；ResNet 用恒等捷径让更深的网络可训练。[Neocognitron](https://doi.org/10.1007/BF00344251)；[LeNet-5](https://doi.org/10.1109/5.726791)；[AlexNet](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)；[ResNet](https://arxiv.org/abs/1512.03385)

它们逐渐形成的先验是局部性、平移等变与空间参数共享；代价是结构固定，性能提升依赖更深网络、更多数据和算力。见《[深度学习基础（07）：CNN——从 LeNet 到 ResNet，再到 ViT](/cnn-from-lenet-to-resnet-and-vit.html)》。

### 2. Embedding：从编号到几何

Embedding 把离散 token 映射为稠密向量；Word2Vec 展示了如何从上下文预测学习词向量。[Word2Vec 原文](https://arxiv.org/abs/1301.3781) 查表让输入适配神经网络，但静态词向量对同一个词只存一份表示，不能直接表达上下文变化。见《[深度学习基础（08）：Embedding 与表示学习——从 one-hot 到上下文表示](/embedding-and-representation-learning.html)》。

### 3. 循环网络、seq2seq 与 attention

RNN 以循环状态处理变长序列并共享时间步参数；LSTM 用门控缓解长距离依赖的学习困难。[LSTM 原文](https://doi.org/10.1162/neco.1997.9.8.1735) seq2seq 曾把输入压缩进固定长度向量；Bahdanau 等人的 attention 让解码器在每一步重新对齐编码器状态。[seq2seq](https://arxiv.org/abs/1409.3215)；[Bahdanau attention](https://arxiv.org/abs/1409.0473)

循环带来时间维串行计算，固定向量也形成信息瓶颈；attention 减轻了后者，却仍依附循环。见《[深度学习基础（09）：RNN——从 LSTM 到 attention 的诞生](/rnn-lstm-and-the-birth-of-attention.html)》。

### 4. Transformer：去掉循环

Transformer 以 self-attention 建立位置间的内容相关连接，训练时可并行处理序列位置，避免 RNN 的逐步递归。[原论文](https://arxiv.org/abs/1706.03762) 代价是注意力对序列长度有二次计算与内存项，推理还需维护 KV cache。后续细节见《[Transformer 原理与实现：从论文到手搓 GPT-2](/transformer-and-llm-structure-implementation-and-evolution.html)》系列总览。

## 九、对照表

| 模型 | 年份 | 结构假设 | 参数共享 | 并行度（训练时） | 典型任务 |
|---|---:|---|---|---|---|
| 感知机 | 1957–1958 | 单个线性阈值边界 | 否 | 高 | 二分类 |
| ADALINE | 1960 | 线性输出、平方误差 | 否 | 高 | 自适应滤波、分类 |
| MLP / 反向传播 | 1986 | 全连接层与非线性组合 | 否 | 高 | 分类、回归 |
| CNN | 1980–2015 | 局部感受野、平移等变 | 空间共享 | 高 | 图像识别 |
| Embedding | 2013 | 离散符号查表为向量 | 同一 token 共享一行 | 高 | 语言表示 |
| RNN / LSTM | 1997 | 状态递归、时间平移共享 | 时间共享 | 低，按步串行 | 序列建模 |
| seq2seq + attention | 2014 | 对齐查询输入状态 | 编码器与解码器分开 | 部分并行 | 机器翻译 |
| Transformer | 2017 | 内容相关的全局注意力 | 位置间共享投影参数 | 序列位置并行 | 翻译、语言建模 |

Table: 不同模型用结构假设换取计算效率或数据效率

## 十、案例：感知机、ADALINE 与 MLP 跑 XOR

实验用 NumPy 固定数据顺序 `(0,0),(0,1),(1,0),(1,1)`。感知机从零权重和偏置开始、学习率 1，AND 在第 6 轮首次达到零误分；XOR 的每轮误分最少为 3，100 轮内没有零误分，权重状态从第 2 轮起以周期 1 循环。

ADALINE 用全批量梯度下降、学习率 0.1 训练 500 轮，最终权重为 `(0.0000000207, 0.0000000207)`，偏置 `0.4999999754`，MSE 为 `0.2500000000`。最小二乘解为 `w=(0,0), b=0.5`，四个样本的预测全是 `0.5`。

两层 MLP 使用 tanh 隐藏层、sigmoid 输出、BCE、学习率 0.5，训练 5000 步。H=2 时 20 个种子有 13 个达到 100% 准确率，最终 BCE 中位数为 `0.00130205`、最大值为 `0.34750613`；H=4 时 20 个种子全部达到 100%，中位数为 `0.00084360`、最大值为 `0.00103286`。图中 H=4 使用 seed 0。

两层结构能表达 XOR，不意味着每个初始化都能学到它；H=2 的结果直接显示了表示能力与优化结果之间的差别。

![XOR 案例：感知机、ADALINE 与成功的 H=4 MLP 决策区域，以及训练误分与 loss 曲线](/img/in-post/dl-evolution-xor.webp)

Table: 感知机、ADALINE 与 MLP 的 XOR 数值来自固定脚本运行

完整输出见 [XOR 实验脚本](https://github.com/arganzheng/ai-learning-labs/blob/main/deep-learning-foundations/case_evolution_xor.py)。

## 十一、本文小结

- 感知机从固定阈值单元变成可学习分类器，但线性边界无法解 XOR。
- ADALINE 把训练写成可微的平方误差优化；多层网络提供表达能力，反向传播提供有效训练梯度。
- 结构历史不断在归纳偏置、参数共享、并行性与数据需求之间取舍；Transformer 汇合了残差、Embedding 与 attention 等多条线索。
- 定理证明能表示不等于优化能学会；数据、初始化、损失与计算资源同样重要。

## 十二、自测

1. 感知机收敛界 $$ (R/\gamma)^2 $$ 中，增大间隔或缩小输入范数分别意味着什么？
2. 为什么单个线性阈值单元无法分类 XOR？
3. ADALINE 和感知机分别在哪个量上计算误差、何时更新？
4. 万能逼近定理没有保证哪些训练或泛化性质？
5. SVM 在第二次寒冬为何有吸引力？逐层预训练后来为何不再是常规做法？
6. Transformer 相比 RNN 提高了什么并行性，又保留了什么长度代价？

<details markdown="1"><summary>答案</summary>

   1. 输入尺度上界越小、线性分隔间隔越大，定理允许的犯错次数越少。
   2. 四个 XOR 标签要求的四条线性不等式互相矛盾，单条直线无法把两组对角点分开。
   3. 感知机看阈值后的类别，犯错才更新；ADALINE 看阈值前的线性输出与目标的平方误差，沿梯度更新。
   4. 它不保证最小宽度、优化器能找到参数、样本效率或测试集泛化。
   5. SVM 有凸目标、核技巧与间隔理论；ReLU、较好初始化、正则化、GPU 与大规模数据让端到端训练更可行。
   6. Transformer 可并行计算序列位置，但标准全注意力仍有 $$O(n^2)$$ 的序列长度计算 / 内存代价。

</details>

## 下一篇

下一篇是《[深度学习基础（02）：反向传播——手推一个两层网络](/backpropagation-by-hand.html)》。
