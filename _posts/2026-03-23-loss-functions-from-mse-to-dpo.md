---
layout: post
series: deep-learning-foundations
title: "深度学习基础（02）：损失函数——怎样把「好」写成一个可求导的数"
subtitle: "Loss Functions: How to Write 'Good' as a Differentiable Number"
tags: [AI, Deep Learning, LLM]
catalog: true
date: 2026-03-23 21:00:00
---

训练代码里 `loss.backward()` 的前提是 loss 存在——是一个标量、可求导、而且"更小"恰好意味着"模型更好"。把任务翻译成这样一个数，这件事本身有将近两百年的历史：最小二乘、最大似然、交叉熵、Huber、focal、InfoNCE、DPO，每一个损失函数背后都是同一个问题的回答——**「好」在这件事上到底怎么量化**。

上一篇《[深度学习基础（01）：深度模型演进史——从感知机到 Transformer](/evolution-of-deep-models-from-perceptron-to-transformer.html)》把网络结构从感知机追到 Transformer；这一篇回到所有结构共有的那一端：输出之后、反向传播之前，那个决定"往哪调"的数是怎么写出来的。

分工先说清楚，避免重复：最大似然 → 交叉熵的推导在《[算法工程师的数学（05）：从最大似然到交叉熵——第一个要会推的 loss](/from-maximum-likelihood-to-cross-entropy.html)》；DPO 的 KL 视角在《[算法工程师的数学（06）：熵、交叉熵与 KL——从困惑度到 DPO](/entropy-cross-entropy-and-kl-to-dpo.html)》；各损失的代码手写在《[面试手撕代码（17）：手撕损失函数与训练算法](/coding-interview-losses-and-training-algorithms.html)》。本篇管的是**为什么选它**、**梯度长什么样**、以及**曲线不对时往哪查**。全篇的核心问题是：

> **一个训练任务，怎样变成一个可求导的数？不同任务为什么选不同的 loss？loss 曲线不对时，往哪看？[^q0]**

## 一、总览：训练只能优化一个可求导的标量

梯度下降只认一样东西：一个标量 $$L(\theta)$$，对它求 $$\partial L / \partial \theta$$。但"任务"五花八门——拟合一条直线、从十个类里挑一个、让相似的样本靠近、让模型偏好更好的回答——损失函数就是把任务翻译成标量的那一步。翻译的依据几乎都是同一条：**假设数据的生成分布，把"概率最大"取负对数**。

| 任务 | 常用 loss | 背后的概率假设 |
|---|---|---|
| 回归 | MSE（最小二乘） | 噪声是高斯分布 |
| 回归（含离群点） | MAE / Huber | 噪声是拉普拉斯 / 截断分布 |
| 分类 | 交叉熵 | 标签是类别分布（softmax 是它的 link） |
| 不均衡分类 | focal loss | 简单样本的梯度要按 $$(1-p_t)^\gamma$$ 降权 |
| 语言模型 | next-token 交叉熵 | 每 token 是词表上的分类 |
| 表示学习 | InfoNCE / 对比损失 | 正例对的相似度要在 softmax 里赢过负例 |
| 偏好对齐 | Bradley–Terry / DPO | 人类偏好服从成对比较模型 |

Table: 任务 → loss → 概率假设

### 1. 来龙去脉：从最小二乘到 DPO

| 年 | 谁 | 当时的问题 | 怎么解决 | 效果 | 引入或遗留问题 |
|---|---|---|---|---|---|
| 1805 / 1809 | Legendre / Gauss | 天文与大地测量里"最贴合数据的直线"怎么定 | 最小二乘：误差平方和最小，等价于高斯噪声下的最大似然 | 成为回归的标准做法，至今是默认基线 | 平方放大离群点的影响，一个坏点带歪整条线 |
| 1912 / 1922 | Fisher | 估计量有没有统一原则 | 最大似然：选让观测数据概率最大的参数 | MSE 与 CE 都成了它的特例 | 需要写出分布；小样本下可以过拟合 |
| 1948 | Shannon | 通信里"信息量"怎么度量 | 熵与交叉熵；编码按 $$-\log p$$ 分配码长 | 交叉熵后来成为分类与语言模型的标准 loss | 目标是最优编码而非决策边界，校准好但边界不直接最优 |
| 1964 | Huber | 最小二乘怕离群点，MAE 在 0 点不可导 | 误差小用平方、误差大用线性，$$\delta$$ 处光滑衔接 | 回归对离群点稳健，梯度有界 | 多了一个超参 $$\delta$$ |
| 2006 | Hadsell 等（对比损失）；2015 Schroff 等（FaceNet triplet） | 没有标签的表示怎么学 | 让同类样本的嵌入靠近、不同类拉开一个 margin | 人脸识别打开新局面 | 三元组难挖、margin 要调；度量学习训练不稳定 |
| 2016 | Szegedy 等 | one-hot 逼着 logit 差走向无穷，模型过度自信 | label smoothing：正确类目标 $$1-\epsilon$$，其余类均分 | 校准变好，Inception 上 top-1 微升 | 置信度被压低，对蒸馏与奖励模型下游要小心 |
| 2017 | Lin 等（RetinaNet） | 检测里背景框成千上万，简单负例淹没梯度 | focal loss：$$-(1-p_t)^\gamma \log p_t$$ | one-stage 检测器精度追上 two-stage | $$\gamma$$ 要调；对本来就均衡的任务增益不大 |
| 2018 | Oord 等（CPC） | 对比学习每个正例对一次只比一条 | InfoNCE：batch 内所有负例进同一个 softmax | 统一了对比学习的形式，成为 CLIP、SimCLR 的 loss | 依赖大 batch 或 memory bank；温度 $$\tau$$ 敏感 |
| 2021–2022 | ST-MoE / PaLM | 大模型 logits 无界漂移，训到一半 loss spike | z-loss：$$\log^2 Z$$ 项惩罚 logsumexp 过大 | PaLM、ST-MoE 用它稳住训练 | 又多一个要调的系数；小模型上不必要 |
| 2023 | Rafailov 等 | RLHF 的 PPO  pipeline 复杂难调 | DPO：偏好学习直接在 $$\log$$ 比值上做，不需要单独的奖励模型 rollout | 成为偏好对齐的主流 baseline | 离线偏好不能在线纠错；对分布外样本置信度差 |

Table: 损失函数的来龙去脉

### 2. 本文的章节安排

| 章 | 内容 |
|---|---|
| 二 | 回归：MSE、MAE、Huber 各假设了什么噪声，梯度差在哪 |
| 三 | 分类：交叉熵与 MLE 的关系，softmax 的 $$p - y$$ 梯度，为什么不选 MSE |
| 四 | 不均衡与过度自信：weighted CE、focal、label smoothing |
| 五 | 语言模型：next-token CE、困惑度、初始 loss、z-loss、MTP |
| 六 | 对比与度量学习：triplet、InfoNCE、温度 $$\tau$$ |
| 七 | 偏好：Bradley–Terry 与 DPO |
| 八 | 诊断：初始 loss、NaN、loss 降而指标不降 |
| 九 | 案例：MSE vs CE、focal vs CE、label smoothing 的校准，MNIST 复现 |
| 十 | 小结 |
| 十一 | 自测 |

## 二、回归：MSE、MAE 与 Huber

**问题**：回归要一个"预测离真值多远"的数。$$e = \hat y - y$$，取它的平方还是绝对值？这不是数学洁癖——选哪个等于假设噪声服从什么分布。

**怎么解决**：

- **MSE**：$$L = \frac{1}{2}e^2$$，等价于假设噪声是高斯分布时的最大似然。梯度 $$\partial L/\partial e = e$$——误差多大梯度多大，离群点拿最大步长。
- **MAE**：$$L = \lvert e \rvert$$，对应拉普拉斯噪声。梯度恒为 $$\pm 1$$——离群点和普通点一样大的一步，抗离群点；代价是 0 点不可导、好样本也一直在震荡。
- **Huber**：$$\lvert e \rvert \le \delta$$ 内用平方、外面用线性，梯度是 $$\operatorname{clip}(e, \pm\delta)$$——小误差光滑、大误差有界。

![回归三兄弟：左图是损失值，右图是对误差的梯度——MSE 线性无界、MAE 恒为 ±1、Huber 在两段之间](/img/in-post/dl-loss-regression.svg)

**效果**：一个算例就看清差别。真实值 $$y=0$$，某个样本预测 $$\hat y = 5$$，$$e=5$$：MSE 给梯度 5，Huber（$$\delta=1$$）给 1，MAE 给 1——离群点对 MSE 的拉力是另外两者的五倍。

**引入或遗留问题**：$$\delta$$ 要调；MAE 在极小值附近不收细。实践中回归默认 MSE，数据脏时换 Huber。

## 三、分类：交叉熵为什么不选 MSE

**问题**：分类输出是一个类别分布。把 one-hot 目标当成回归目标用 MSE 行不行？

**怎么解决**：交叉熵就是最大似然取负对数——完整推导见《[算法工程师的数学（05）：从最大似然到交叉熵——第一个要会推的 loss](/from-maximum-likelihood-to-cross-entropy.html)》，本篇只引用结论：$$L = -\log p_y$$。softmax 与交叉熵组合后对 logits 的梯度恰好是 $$p - y$$——预测概率减目标，推法见《[深度学习基础（03）：反向传播——手推一个两层网络](/backpropagation-by-hand.html#3-softmax-与交叉熵)》「三、…3. softmax 与交叉熵」。

分类不用 MSE 的真正原因在**梯度**，不在"直觉上不合适"。设 $$y=1$$、logit $$z$$、$$p = \sigma(z)$$：

- CE：$$\partial L/\partial z = p - 1$$，错了多少梯度就是多少，永远有力气往对的方向拉；
- MSE：$$\partial L/\partial z = \sigma'(z)\,(p - 1)$$，多乘了一个 $$\sigma'(z) \le 0.25$$——在最需要梯度的饱和区（$$z$$ 很负时）$$\sigma'$$ 近似为 0，梯度被扼杀。

![左图：y=1 时 MSE 与 CE 的损失曲线；右图：对 logit 的梯度——MSE 在饱和区只剩 CE 的 σ' 倍](/img/in-post/dl-loss-classification-gradient.svg)

**效果**：$$z = -6$$ 时 MSE 的梯度是 CE 的 $$1/403$$（$$\sigma'(-6)\approx0.0025$$）。第九章的 MNIST 对照里，CE 一个 epoch 到 94.5%，MSE 只到 91.4%，前 50 步平均梯度范数差 3 倍。

**引入或遗留问题**：CE 对"对的置信度"无止境追求——$$p_y \to 1$$ 才满足，这会推着 logit 差走向无穷，是过度自信与校准差的来源（第四章、正则化篇的 label smoothing 都在治它）。

## 四、不均衡与过度自信

**问题**：交叉熵有两个结构性偏差——所有样本的梯度权重相同（类别不均衡时多数类淹没少数类），所有预测都被推向 $$p_y \to 1$$（过度自信）。

**怎么解决**：

- **weighted CE**：给类别 $$k$$ 一个权重 $$w_k$$（如频率倒数），直接放大少数类梯度。
- **focal loss**（Lin 等 2017）：$$L = -(1-p_t)^\gamma \log p_t$$。不区分类别，区分**难易**：$$p_t$$ 已经很高的样本，loss 被 $$(1-p_t)^\gamma$$ 压下去。$$\gamma=0$$ 就是 CE。
- **label smoothing**：把 one-hot 换成"正确类 $$1-\epsilon$$、其余类均分 $$\epsilon$$"，给 logit 差一个有限的最优点。机制与推导见《[深度学习基础（07）：正则化与泛化——为什么参数比样本多却不过拟合](/regularization-and-generalization.html#4-label-smoothing-与数据增强)》「五、4. Label smoothing 与数据增强」，本篇不重复。

![focal loss：γ 越大，高置信样本的 loss 被压得越狠](/img/in-post/dl-loss-focal.svg)

**效果**：$$\gamma=2$$ 时，$$p_t = 0.9$$ 的样本对 $$p_t = 0.5$$ 的样本的相对 loss 权重从 1 降到 0.04——简单样本几乎不再贡献梯度。第九章的类不均衡实验里，focal 把只占 0.2% 的少数类召回从 7.4% 拉到 21.6%。

**引入或遗留问题**：focal 的 $$\gamma$$ 与 weighted CE 的权重都要调；label smoothing 压低的置信度对蒸馏、奖励模型这类"下游要吃掉概率"的场景要小心。

## 五、语言模型的 loss

**问题**：语言模型预测下一个 token，词表几万到十几万——本质上是每步一次 $$V$$ 分类。

**怎么解决**：next-token 交叉熵，对序列求平均：

$$L = -\frac{1}{T}\sum_t \log p_\theta(x_t \mid x_{<t})$$

配套的几个数（都来自第九章或论文）：

- **困惑度** $$PPL = e^{L}$$：loss 1.8 nat 对应 PPL ≈ 6.05——模型平均每步在约 6 个等可能的候选里犹豫。
- **初始 loss ≈ ln V**：随机初始化的模型输出接近均匀，$$-\log(1/V) = \ln V$$。GPT-2 的 $$V = 50257$$，$$\ln 50257 \approx 10.82$$；Llama-3 的 128256 给 11.8。训练第一屏 loss 离这个数差得远就是初始化或算错——第八章的诊断靠它。
- **z-loss**（ST-MoE 2021 / PaLM 2022）：给 loss 加一项 $$\lambda \log^2 Z$$，惩罚 logsumexp 过大——logits 无界漂移是大模型训练 loss spike 的一个来源。
- **MTP 辅助 loss**：DeepSeek-V3 等多预测几步，每一步一个 CE，主 loss 加辅助项；推理时丢掉辅助头或用于投机解码。

**效果与遗留**：next-token CE 简单、好求导、和 MLE 一致，代价是**它只管下一个 token**——长程一致性、事实性、风格都不在这个数里，那是后训练（RLHF / DPO）要补的事，见第七章。

## 六、对比与度量学习

**问题**：没有"正确类别"，只有"谁和谁像"——embedding、检索、自监督表示都是这种问题。

**怎么解决**：

- **triplet loss**（FaceNet 2015）：锚点 $$a$$、正例 $$p$$、负例 $$n$$，$$L = \max(d(a,p) - d(a,n) + m, 0)$$——让正例距离比负例至少小一个 margin $$m$$。
- **InfoNCE**（CPC 2018，CLIP / SimCLR 沿用）：把一个锚点的所有正例与 batch 内全部负例放进同一个 softmax，$$L = -\log \frac{e^{\text{sim}(a,p)/\tau}}{\sum_j e^{\text{sim}(a,j)/\tau}}$$——温度 $$\tau$$ 控 softmax 的锐度，batch 里的所有其他样本都是负例。

![in-batch 相似度矩阵 ×1/τ：红字是正例对；InfoNCE 对每一行做一次 softmax，要求正例格的 logit 相对整行最大](/img/in-post/dl-loss-infonce.svg)

**效果**：一个 8 样本 2 类的具体例子（脚本输出）：$$\tau = 0.1$$，$$z_0$$ 一行的 logsumexp = 10.06，三个正例的相似度 logit 之和 12.30，InfoNCE loss = 5.96。负例越多分母越大，正例必须赢过整行。

**引入或遗留问题**：triplet 需要挖难负例，容易塌缩；InfoNCE 的负例就是 batch——batch 小负例就少，需要大 batch 或 memory bank；$$\tau$$ 对结果非常敏感。

## 七、偏好 loss：Bradley–Terry 与 DPO

**问题**：RLHF / 对齐里要优化的不是"答对"而是"更好"——标注员给的是成对比较，不是分数。

**怎么解决**：

- **Bradley–Terry 模型**把偏好写成概率：$$P(y_w \succ y_l) = \frac{e^{r_w}}{e^{r_w} + e^{r_l}} = \sigma(r_w - r_l)$$——对奖励分数求 sigmoid 差，训练奖励模型就是最大化这个似然，等价于在 $$r_w - r_l$$ 上做二分类 CE。
- **DPO**（2023）发现不需要单独训一个奖励模型：把策略的 log 比值直接塞进 Bradley–Terry 的形式，

$$L_{DPO} = -\log \sigma\!\left(\beta\big[\log \tfrac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \tfrac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\big]\right)$$

梯度直觉：对 $$r_\theta = \beta \log(\pi_\theta / \pi_{ref})$$ 做 logistic 回归，赢的样本概率升、输的降，同时被 $$\beta$$ 拉住不偏离参考模型太远。推导见《[算法工程师的数学（06）：熵、交叉熵与 KL——从困惑度到 DPO](/entropy-cross-entropy-and-kl-to-dpo.html)》；实现在《[面试手撕代码（17）：手撕损失函数与训练算法](/coding-interview-losses-and-training-algorithms.html)》，RLHF 全貌在《[后训练（04）：offline RL——DPO 与它的家族](/offline-rl-dpo-and-its-family.html)》。

**效果与遗留**：DPO 把 RLHF 的三段式（RM → PPO → KL 约束）压成一个 loss，成为最普及的偏好 baseline；代价是离线偏好数据不能在线纠错，且对分布外回答的置信度外推很差——这些是 PPO / GRPO 一派继续存在的理由，见《[后训练（03）：online RL——PPO、GRPO 与 RLHF 三件套](/online-rl-ppo-grpo-and-the-rlhf-trio.html)》。

## 八、诊断：loss 曲线告诉你的第一件事

**问题**：loss 是第一个能看的数，也是最常被误读的数。

**怎么解决**——四个动作，都能立刻做：

1. **对初始 loss**：随机初始化的分类任务 loss 应该接近 $$\ln C$$（$$C$$ 是类别数）。MNIST 是 $$\ln 10 \approx 2.30$$，GPT-2 是 10.82。差得远——高很多是初始化错了，低很多是标签泄漏。
2. **查 NaN / log(0)**：softmax 先减 max；分母加 eps；logsumexp 用稳定形式。出现 NaN 先怀疑 logits 溢出或 log 了 0。
3. **loss 降而指标不降**：训练 loss 在降但准确率不动，常见的是"越学越自信地错"——分布内样本更自信、错误的也更自信，过度自信本身就是 CE 的副作用；去看校准与逐类召回，别只看均值。
4. **loss spike 与 z-loss**：大模型训练里 logits 无界漂移会把 loss 顶飞，z-loss 把这个漂移写进目标里。

## 九、案例：三个 MNIST 对照实验

脚本 `deep-learning-foundations/case_02_loss_functions.py`，784-256-10 两层 MLP、SGD lr 0.1、一个 epoch（469 步 × 128）、固定种子。

### 1. MSE vs CE：收敛差半个量级

同一个网络、同一个学习率、同一个初始化，只换 loss：

| loss | 初始 loss | 末 100 步均值 | 前 50 步平均梯度范数 | test acc |
|---|---|---|---|---|
| softmax + CE | 2.939 | 0.185 | 1.66 | 94.50% |
| sigmoid + MSE | 1.500 | 0.092 | 0.53 | 91.40% |

MSE 的"loss 更小"是假象——它的梯度从第一步就小 3 倍，学习率相同就走得更慢。

![左：训练 loss；右：梯度范数——MSE 一开始就低一个量级](/img/in-post/dl-loss-mse-vs-ce-training.svg)

### 2. focal vs CE：类不均衡下的逐类召回

把类 9 砍到原来的 2%（118 张、占训练集 0.2%），其余类不动：

| loss | 总体 acc | 类 9 召回 | 备注 |
|---|---|---|---|
| CE | 86.75% | 7.4% | 少数类几乎被多数类淹没 |
| focal γ=2 | 87.71% | 21.6% | 难样本梯度权重上来了 |

![逐类召回率：CE 与 focal 在其余 9 类上接近，少数类 9 上 focal 明显更高](/img/in-post/dl-loss-focal-imbalance.svg)

### 3. label smoothing 的校准

同一个网络，one-hot vs ε=0.1 平滑目标：

| 目标 | test acc | 平均置信度 | ECE（10 bin） |
|---|---|---|---|
| one-hot | 94.42% | 93.6% | 0.96% |
| 平滑 ε=0.1 | 94.43% | 81.4% | 13.00% |

![可靠性图：越贴对角线越准；平滑把置信度压到了对角线下方——这个实验里它从"略自信"变成了"不自信"](/img/in-post/dl-loss-calibration.svg)

注意这里的读法：one-hot 在 MNIST 上已经相当校准（ECE 0.96%），label smoothing 把平均置信度从 93.6% 压到 81.4%，反而**低于**实际准确率——它对过度自信的模型是药，对本来校准好的模型是另一种偏差。这就是正则化篇「引入或遗留问题」里那句"对蒸馏与奖励模型要小心"的具体形态。

## 十、小结

- 训练只能优化一个可求导的标量；损失函数是把任务翻译成这个数的那一步，依据几乎都是"假设分布 → 最大似然 → 取负对数"。
- 回归默认 MSE，怕离群点用 Huber；分类用交叉熵——不是"更合理"，是 MSE 的梯度在饱和区被 $$\sigma'$$ 杀死。
- 不均衡用 weighted CE 或 focal（简单样本降权）；过度自信用 label smoothing（给 logit 差一个有限最优点）；大模型的 logit 漂移用 z-loss。
- 语言模型的 loss 是 next-token CE，初始值 ≈ $$\ln V$$；对比学习是 batch 内 softmax 的 InfoNCE；偏好学习是 Bradley–Terry 形式的 DPO。
- 诊断先对初始 loss（≈ $$\ln C$$），再查 NaN 与梯度尺度，最后别把"loss 降"当成"模型好"——看校准与逐类指标。

## 十一、自测

1. 为什么分类不用 MSE？对 $$y=1$$、$$z=-6$$ 的具体梯度差多少倍？
2. label smoothing $$\epsilon=0.1$$、10 分类，错误类的目标概率是多少？它把 logit 差推向无穷还是拉回有限值？
3. focal loss 的 $$\gamma$$ 从 0 调到 5，$$p_t = 0.9$$ 与 $$p_t = 0.5$$ 两个样本的相对权重怎么变？
4. 一个新的 200K 词表语言模型，第一步 loss 应该大约是多少？loss 是 4 说明什么？
5. InfoNCE 的 batch 为什么越大越好？$$\tau$$ 变大时 softmax 是更锐还是更平？
6. DPO 相比"训奖励模型 + PPO"省掉了什么，又放弃了什么？

<details><summary>参考答案</summary>

1. MSE 对 logit 的梯度多乘一个 $$\sigma'(z) \le 0.25$$，$$z=-6$$ 时只有 CE 的 $$1/403$$（$$\sigma'(-6)\approx0.0025$$），饱和区基本不动。
2. 错误类各 $$\epsilon/(C-1) = 0.1/9 \approx 0.0111$$；logit 差的最优点有限（$$g^* = \ln 81 \approx 4.394$$），不再无穷。
3. $$(1-p_t)^\gamma$$ 项把 $$p_t=0.9$$ 相对 $$p_t=0.5$$ 的权重从 1 压到 $$0.1^\gamma/0.5^\gamma$$——$$\gamma=2$$ 时 0.04、$$\gamma=5$$ 时约 0.00032。
4. $$\ln 200000 \approx 12.21$$；4 远低于它，说明标签泄漏、初始化就是某个训练过的权重，或算错了。
5. InfoNCE 的分母是 batch 内全部样本，batch 大负例才多、才逼近真实分布；$$\tau$$ 大 softmax 更平、梯度更散。
6. 省掉了奖励模型 rollout 与 PPO 的在线采样；放弃了在线纠错能力与分布外置信度——离线偏好学到什么就是什么。

</details>

[^q0]: 分工：怎么推导（数学篇）、怎么写代码（面试篇）、为什么选它（本篇）。三篇不重复——推导见《[算法工程师的数学（05）：从最大似然到交叉熵](/from-maximum-likelihood-to-cross-entropy.html)》，实现见《[面试手撕代码（17）：手撕损失函数与训练算法](/coding-interview-losses-and-training-algorithms.html)》。
