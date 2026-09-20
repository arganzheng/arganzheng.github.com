---
layout: post
series: multimodal
title: "多模态（07）：扩散模型（下）：score matching、flow matching 与 classifier-free guidance"
subtitle: "Diffusion II: Score Matching, Flow Matching, Noise Schedules, and Classifier-Free Guidance"
tags: [AI, Multimodal, Diffusion, Generative Models]
catalog: true
date: 2026-05-07 20:00:00 +0800
---

[上篇](/diffusion-models-ddpm-score-matching-and-flow-matching.html)用 DDPM 的语言把扩散模型讲了一遍：加噪、猜噪声、逐步去噪、DDIM 跳步。这一篇讲另外两种看同一件事的方式——**score matching**（网络学的其实是"概率密度上升最快的方向"）与 **flow matching**（网络学的是"从噪声到数据的速度场"，SD3 与 FLUX 用的就是它）——它们的推导与记号完全不同，读起来像三个东西，实际上训练的是同一个网络、只是**参数化不同**；看清它们在哪一步汇合，读任何一篇扩散论文都不会再迷路。然后讲两个所有文生图模型都依赖的技术：噪声调度（哪些噪声水平该多学）与 **classifier-free guidance**（$$w = 7.5$$ 到底在做什么）。

全篇继续用上篇那个二维的两个月牙 toy：分数场可以画成箭头图、flow matching 的轨迹可以画成线、CFG 的效果可以画成点——每个概念都有一张能看的图和一组能验算的数。

本篇要回答的核心问题是：

> **DDPM、score matching、flow matching 为什么是同一件事？[^q0] CFG 的 $$w = 7.5$$ 在数学上意味着什么？[^q1]**

## 一、总览：三种视角、一个网络

### 1. 一张对照表

| 视角 | 提出 | 前向过程 | 网络学什么 | 训练损失 | 采样 |
|---|---|---|---|---|---|
| DDPM（去噪，上篇） | Sohl-Dickstein 2015；Ho 等 2020 | 离散 $$T$$ 步高斯加噪的马尔可夫链 | 噪声 $$\epsilon_\theta(x_t, t)$$ | $$\lVert \epsilon - \epsilon_\theta \rVert^2$$（ELBO 的简化） | $$T$$ 步逐步去噪（随机）；DDIM 确定性、可跳步 |
| Score SDE（分数） | Song 等 2021 | 连续时间 SDE $$dx = f\,dt + g\,dw$$ | 分数 $$s_\theta(x_t, t) \approx \nabla_x \log p_t(x)$$ | 去噪分数匹配 $$\lVert s_\theta - \nabla \log p_t(x_t \mid x_0) \rVert^2$$ | 反向 SDE 或概率流 ODE，任意数值求解器 |
| Flow matching（流） | Lipman 等 2023；Liu 等 2023（rectified flow） | 直线插值 $$x_t = (1-t) x_0 + t\, \epsilon$$ | 速度 $$v_\theta(x_t, t)$$ | $$\lVert (\epsilon - x_0) - v_\theta \rVert^2$$ | ODE $$dx/dt = v_\theta$$，Euler 几十步 |

三行的网络输入都是（带噪样本，时间），输出都是一个与样本同形状的向量场。噪声 $$\epsilon$$、分数 $$s$$、速度 $$v$$ 之间是**线性变换**（第三章给出公式）——训练其中一个就能算出另外两个。差别在：训练时对不同时间步的**加权**不同、采样时的**路径**不同（曲线 vs 直线）。

### 2. 先说答案

三者是同一件事，因为它们都在学同一个对象——每个噪声水平 $$t$$ 下带噪数据分布 $$p_t(x)$$ 的**分数** $$\nabla_x \log p_t(x)$$——的不同参数化。DDPM 学的噪声 $$\epsilon$$ 与分数的关系是 $$s = -\epsilon / \sigma_t$$（Tweedie 公式的推论，第二章）；flow matching 学的速度 $$v$$ 是 $$\epsilon$$ 与 $$x_0$$ 的差，也可以用分数表达。三种损失在换元后只差一个与 $$t$$ 有关的权重。它们的采样都是解同一个概率流 ODE（或对应的 SDE），只是 flow matching 选择的前向过程让 ODE 的轨迹更容易拉直，因而少步数就够。

CFG 的 $$w = 7.5$$ 意味着采样的分布不是 $$p(x \mid c)$$，而是 $$p(x \mid c)^{w} / p(x)^{w-1}$$ 的归一化——把条件分布"锐化"了 $$w$$ 倍：条件 $$c$$ 下比无条件更可能的区域被放大 $$w$$ 次幂，其他区域被压掉。$$w = 1$$ 是原条件分布；$$w = 7.5$$ 让样本强烈偏向"最典型地符合 $$c$$"的模式——保真度与文本一致性上升、多样性下降、颜色饱和度过高（over-saturation）。它是一个**有意的分布改变**，与 [L6 第一篇](/decoding-strategies-sampling-and-constrained-generation.html)的低温采样是同类操作。第五章展开。

### 3. 本文的章节安排

本文按"换一种语言 → 再换一种语言 → 两个工程技术"组织：先把上篇的噪声预测器翻译成"分数"（第二章），再翻译成"速度场"（第三章），每次翻译都在 toy 上验证是同一个东西；然后讲调度（第四章）与 CFG（第五章）。

| 章 | 主题 | 内容 |
|---|---|---|
| 二 | score matching | 分数是什么（把上篇的模型画成箭头图）；Tweedie 公式（toy 验算）；去噪分数匹配 = 噪声预测；SDE 与概率流 ODE |
| 三 | flow matching | 直线路径与速度场；toy 上从零训一个（代码）；与 DDPM 的换算；轨迹为什么弯、reflow 怎么拉直（图）；1 / 2 / 5 / 20 步对比（图） |
| 四 | 噪声调度与时间步采样 | linear / cosine / 零终端 SNR；logit-normal；分辨率与调度的耦合 |
| 五 | classifier-free guidance | 条件 dropout；从贝叶斯到 $$\tilde\epsilon$$；toy 上 w = 0 / 1 / 2 / 4 / 8（图）；它在采样什么分布；副作用与修正 |
| 六 | 成本 | 训练与采样的 FLOPs；与 LLM 的对比 |
| 七 | 动手（建议） | CIFAR-10 上 DDPM vs flow matching；配套代码 |
| 八 | 本文小结 | |
| 九 | 自测 | 5 道题 |

## 二、score matching：分数的视角

### 1. 分数

分布 $$p(x)$$ 的**分数**（score）是 $$\nabla_x \log p(x)$$——对数密度的梯度（L2 第二篇：梯度指向函数上升最快的方向）：在每一点，它是一个箭头，指向"概率密度增加最快"的方向，箭头长度是增加的速率。在数据密集的地方附近，箭头都指向数据。上篇训好的 DDPM 模型可以直接画成这样的箭头图（第 2 节会说明为什么它就是分数）：

![三张箭头图，灰点是缩放后的数据：t = 600（σ = 0.99）箭头几乎全部指向中心，均匀而短；t = 300（σ = 0.78）箭头开始朝两个月牙的位置弯；t = 80（σ = 0.26）箭头密集地指向两个月牙的轮廓，离数据越远箭头越长](/img/in-post/multimodal-07-score-field.svg)

怎么读：每个网格点画的是网络在那里算出的分数向量。噪声大（$$t = 600$$）时，带噪分布接近一个大高斯，分数就是"指向原点"；噪声小（$$t = 80$$）时，分数精确地指向月牙。**采样就是顺着箭头走**——知道分数就能采样：Langevin 动力学 $$x \leftarrow x + \frac{\delta}{2} \nabla_x \log p(x) + \sqrt{\delta}\, z$$ 反复迭代收敛到 $$p$$。分数**不需要归一化常数**（$$\nabla \log(p / Z) = \nabla \log p$$），这是它相对于直接建模密度的优势。

问题：数据分布的分数在数据稀疏的区域估计不准，且低维流形上的数据让分数无定义。解法（Song & Ermon 2019）：对数据加不同尺度的高斯噪声，学**每个噪声尺度下**带噪分布 $$p_\sigma(x) = \int p(x_0) \mathcal{N}(x; x_0, \sigma^2 I) dx_0$$ 的分数，从大噪声（分数处处有定义、指向数据的大致方向）到小噪声逐步 Langevin——这就是"扩散"。

### 2. 去噪分数匹配

怎么学 $$\nabla_x \log p_\sigma(x)$$？Vincent 2011 的去噪分数匹配：

$$
\mathbb{E}_{x_0, \tilde x \sim \mathcal{N}(x_0, \sigma^2 I)} \big\lVert s_\theta(\tilde x, \sigma) - \nabla_{\tilde x} \log \mathcal{N}(\tilde x; x_0, \sigma^2 I) \big\rVert^2
$$

即让网络匹配**条件**分布 $$p(\tilde x \mid x_0)$$ 的分数（这个是显式的高斯，分数 $$= -(\tilde x - x_0) / \sigma^2 = -\epsilon / \sigma$$），可以证明它的最优解等于**边缘**分布 $$p_\sigma$$ 的分数（对 $$x_0$$ 积分后一致）。于是

$$
s_\theta(\tilde x, \sigma) \approx -\frac{\epsilon}{\sigma} \quad \Longleftrightarrow \quad \epsilon_\theta = -\sigma\, s_\theta
$$

**去噪分数匹配就是噪声预测**，差一个 $$-\sigma$$ 的因子。DDPM 的 $$\epsilon_\theta$$ 学的正是 $$-\sqrt{1 - \bar\alpha_t}\, \nabla_{x_t} \log p_t(x_t)$$。

### 3. Tweedie 公式

同一件事的另一种说法。Tweedie（Efron 2011）：对高斯噪声污染的观测 $$\tilde x = x_0 + \sigma \epsilon$$，

$$
\mathbb{E}[x_0 \mid \tilde x] = \tilde x + \sigma^2 \nabla_{\tilde x} \log p_\sigma(\tilde x)
$$

后验均值等于观测加 $$\sigma^2$$ 倍的分数——"最好的去噪结果 = 带噪点沿分数方向走 $$\sigma^2$$ 那么远"。用 toy 验算（$$t = 300$$，DDPM 的记号里 $$x_t = \sqrt{\bar\alpha_t} x_0 + \sigma_t \epsilon$$，所以走完还要除以 $$\sqrt{\bar\alpha_t}$$）：

```text
真实 x_0 = [-0.45 -0.17]，带噪 x_t = [ 0.81 -1.31]，x_t + σ²·s 再除以 √ᾱ = [ 0.53 -0.85]
```

从 $$(0.81, -1.31)$$ 出发沿分数走一步到 $$(0.53, -0.85)$$——朝 $$x_0$$ 的方向去了，但没有到 $$x_0$$：Tweedie 给的是**后验均值**，即"所有可能产生这个 $$x_t$$ 的 $$x_0$$ 的平均"，一个带噪点可能来自月牙上的一段而不是一个点，平均落在它们中间。这正是上篇 $$t = 50$$ 的 MSE 降不到 0 的原因。它把三样东西连起来：**去噪**（估 $$\mathbb{E}[x_0 \mid \tilde x]$$）、**分数**、**噪声预测**（$$\mathbb{E}[x_0 \mid \tilde x] = \tilde x - \sigma \mathbb{E}[\epsilon \mid \tilde x]$$）。一个最优的去噪器自动给出分数，反之亦然。这是"为什么去噪等于学分布"的最短回答。

### 4. SDE 与概率流 ODE

Song 等 2021 把离散的加噪链推广为连续时间 SDE：

$$
dx = f(x, t)\, dt + g(t)\, dw
$$

DDPM 对应 VP（variance preserving）SDE：$$f = -\frac{1}{2}\beta(t) x$$，$$g = \sqrt{\beta(t)}$$；Song & Ermon 的 NCSN 对应 VE（variance exploding）SDE。Anderson 1982 的结果：这个 SDE 有一个时间反向的 SDE

$$
dx = \big[f(x, t) - g(t)^2 \nabla_x \log p_t(x)\big] dt + g(t)\, d\bar w
$$

从 $$x_T \sim p_T$$ 反向解它，得到 $$x_0 \sim p_0$$。它只需要分数——正是网络学的东西。更进一步，存在一个**确定性**的 ODE 与这个 SDE 有相同的边缘分布 $$p_t$$：

$$
\frac{dx}{dt} = f(x, t) - \frac{1}{2} g(t)^2 \nabla_x \log p_t(x)
$$

这是**概率流 ODE**。DDIM 的 $$\eta = 0$$ 正是它的一种离散化。有了 ODE，采样就是数值积分——可以用任何高阶求解器（Heun、DPM-Solver 利用了 ODE 的半线性结构做指数积分，10–20 步达到 DDIM 50 步的质量）。

至此三种视角汇合了两种：DDPM 的噪声预测 = 分数的负 $$\sigma$$ 倍；DDPM 的采样 = 反向 SDE 的离散化；DDIM = 概率流 ODE 的离散化。

## 三、flow matching：直线的视角

### 1. 从 ODE 直接出发

既然采样最终是解一个 ODE $$dx/dt = v(x, t)$$，为什么不**直接学速度场 $$v$$**？这是 flow matching（Lipman 等 2023）与 rectified flow（Liu 等 2023）的出发点。定义一条从数据 $$x_0$$ 到噪声 $$x_1 = \epsilon$$ 的路径（注意这里 $$t = 0$$ 是数据、$$t = 1$$ 是噪声，与 DDPM 的记号相反——这是 SD3 / FLUX 的约定），最简单的路径是**直线**：

$$
x_t = (1 - t)\, x_0 + t\, \epsilon, \qquad \frac{dx_t}{dt} = \epsilon - x_0
$$

给定 $$(x_0, \epsilon)$$ 这一对，路径上每一点的速度是常数 $$\epsilon - x_0$$。**条件流匹配**的损失：

$$
\mathcal{L}_{CFM} = \mathbb{E}_{t \sim U[0,1],\ x_0,\ \epsilon} \big\lVert v_\theta(x_t, t) - (\epsilon - x_0) \big\rVert^2
$$

读法：随机抽一个数据点 $$x_0$$、一个噪声点 $$\epsilon$$、一个时刻 $$t$$，在两点连线的 $$t$$ 处取 $$x_t$$，让网络在 $$(x_t, t)$$ 输出这条线的方向 $$\epsilon - x_0$$。代码与上篇的 `ddpm_loss` 几乎一样，只是 $$x_t$$ 的算法与目标换了：

```python
def fm_loss(model, x0):
    t = torch.rand(len(x0))                                                    # ① t ~ U[0, 1]（0 = 数据，1 = 噪声）
    eps = torch.randn_like(x0)
    xt = (1 - t)[:, None] * x0 + t[:, None] * eps                              # ② 直线插值
    return ((model(xt, t) - (eps - x0)) ** 2).mean()                           # ③ 让网络猜速度 ε − x_0

@torch.no_grad()
def euler_sample(model, n, steps):
    x = torch.randn(n, 2)
    for i in range(steps):
        t = 1 - i / steps
        x = x - model(x, torch.full((n,), t)) / steps                          # ④ 沿速度反方向走一小步：x_{t−Δ} = x_t − v·Δ
    return x
```

Lipman 等证明：这个只用"配对的条件速度"的损失，与匹配**边缘**速度场（把所有经过 $$x_t$$ 的配对平均）的损失有相同的梯度——所以 $$v_\theta$$ 学到的是边缘速度场，解 $$dx/dt = v_\theta$$ 从 $$\epsilon$$ 到 $$t = 0$$ 就得到数据分布的样本。推导的结构与去噪分数匹配（条件分数的期望 = 边缘分数）完全平行。toy 上训 12000 步（loss 稳定在 1.17——这个数不会到 0：给定 $$x_t$$，$$\epsilon - x_0$$ 本身就有随机性，网络只能给出平均），Euler 20 步采样得到的点离真实数据 0.017——与上篇 DDPM 1000 步的 0.021 同量级。

### 2. 与 DDPM 的换算

直线路径 $$x_t = (1 - t) x_0 + t \epsilon$$ 与 DDPM 的 $$x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon$$ 是同一族（$$x_t = a_t x_0 + b_t \epsilon$$）中的两个成员：DDPM 是**方差保持**（$$a_t^2 + b_t^2 = 1$$，圆弧），flow matching 是 $$a_t + b_t = 1$$（直线）。在任何这样的路径上，$$\epsilon$$、$$x_0$$、$$v = \epsilon - x_0$$、分数 $$s$$ 都由 $$x_t$$ 与其中一个线性决定：

$$
\epsilon = x_t + (1 - t)\, v, \qquad x_0 = x_t - t\, v, \qquad s = -\frac{\epsilon}{t} = -\frac{x_t + (1-t) v}{t}
$$

（直线路径下 $$\sigma_t = t$$。）所以 flow matching 的 $$v$$-预测与 DDPM 的 $$\epsilon$$-预测学的是同一个信息，训练损失只差一个与 $$t$$ 有关的权重（$$\lVert v - v_\theta \rVert^2 = \lVert \epsilon - \epsilon_\theta \rVert^2 / (1-t)^2$$ 一类）。Kingma & Gao 2023 把所有这些损失统一成"加权的 ELBO"——不同方法的差别**全部**归结为对不同噪声水平的加权。这是核心问题前半的完整答案。

### 3. 为什么直线让步数少

采样是解 ODE。ODE 的轨迹越直，Euler 法一步走得越远、误差越小——一条直线一步就能走完，一条弯路必须分很多小段。flow matching 的**条件**路径是直线，但**边缘**速度场（多条直线的平均）不一定直——训练时 $$x_0$$ 与 $$\epsilon$$ 是随机配对的，两条相交的直线在交点处方向不同，网络只能输出它们的平均，轨迹就在那里拐弯。toy 上把 12 个噪声起点的轨迹画出来，用"弦长 / 路径长"量直线度（1 = 完全直）：

```text
同一批 12 个噪声起点：DDIM（DDPM 模型）轨迹的直线度 0.74，flow matching 轨迹 0.49（1 = 完全直）
```

flow matching 的轨迹**比 DDIM 的还弯**——随机配对的代价。Rectified flow 的 **reflow** 操作修的就是它：用训好的模型从噪声生成样本，得到新的 $$(\epsilon, x_0)$$ 配对——它们由 ODE 轨迹连接、彼此不相交——在这些固定的配对上重训一轮：

```text
reflow 一轮后同一批起点的轨迹直线度 1.00
```

![三张图，灰点是数据，每张 12 条从灰色起点到彩色终点的轨迹：DDIM 的轨迹（橙）中等弯曲；flow matching 的轨迹（蓝）有几条先绕一段再拐向月牙、末端打卷；reflow 后的轨迹（绿）几乎全是直线段](/img/in-post/multimodal-07-trajectories.svg)

直线度直接换成采样步数。三个模型（DDPM 用 DDIM 采样、flow matching 用 Euler、reflow 后用 Euler）在 1 / 2 / 5 / 20 步下：

| 步数 | DDIM | flow matching | flow matching + reflow |
|---:|---:|---:|---:|
| 1 | 3.715（散成一片） | 0.290（缩成一个点） | **0.030** |
| 2 | 0.168 | 0.112 | 0.025 |
| 5 | 0.099 | 0.041 | 0.023 |
| 20 | 0.033 | 0.017 | 0.020 |

![3 行 4 列的散点图：第一行 DDIM——1 步是满屏乱点，2 步是一个斜条，5 步一团，20 步才是月牙；第二行 flow matching——1 步缩成一个点，2 步一个椒盐团，5 步月牙初现，20 步清晰；第三行 reflow——1 步就是两个清晰的月牙，之后几乎不变](/img/in-post/multimodal-07-few-steps.svg)

reflow 后**一步**就生成出两个月牙（0.030），而 DDIM 一步是灾难、flow matching 一步全缩到均值——这就是"轨迹直 = 步数少"的全部含义。每轮 reflow 让边缘轨迹更直，几轮后可以 1–2 步采样。SD3 与 FLUX 没有做 reflow，但直线参数化本身已经让它们在 20–30 步达到 DDPM 50 步的质量（上表 5 步一列：0.041 vs 0.099），且更容易做后续的步数蒸馏（下一篇）。

另一个实用的好处：直线路径下 $$t$$ 的语义直接——$$t = 0.5$$ 就是"一半数据一半噪声"——调度设计更直观（第四章）。

## 四、噪声调度与时间步采样

### 1. 调度决定"什么噪声水平被学得多"

调度是 $$t \mapsto (a_t, b_t)$$（$$x_t = a_t x_0 + b_t \epsilon$$ 里信号与噪声的系数）或等价地 $$t \mapsto \text{SNR}(t) = a_t^2 / b_t^2$$（[信噪比](# "tip: signal-to-noise ratio：信号功率 / 噪声功率。上篇 t = 300 处 a² = 0.394、b² = 0.606，SNR = 0.65；t = 100 处 SNR = 8.5。SNR 从无穷降到 0 就是一条加噪过程")）。它与训练时 $$t$$ 的采样分布一起决定每个噪声水平在训练里出现的频率——即网络在哪些噪声水平上学得好。

- **DDPM linear**（$$\beta_t$$ 线性）：SNR 在中间下降太快，很多步浪费在几乎纯噪声的区域。
- **cosine**（Nichol & Dhariwal 2021）：$$\bar\alpha_t = \cos^2(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2})$$，SNR 下降更均匀，在小数据集上提升明显。
- **零终端 SNR**（Lin 等 2024）：DDPM linear 在 $$t = T$$ 处 $$\bar\alpha_T \approx 0.0047 \ne 0$$——训练时最噪的样本仍含一点信号，但采样从纯噪声开始，训练与采样不一致；表现是生成的图片平均亮度总是中等（不能生成很暗或很亮的图）。修正：把调度缩放到 $$\bar\alpha_T = 0$$，用 $$v$$-prediction（此时 $$\epsilon$$-prediction 无定义），采样从真正的纯噪声开始。
- **EDM**（Karras 等 2022）：在 $$\sigma$$ 空间用对数正态分布采样噪声水平，配合网络输入输出的预处理（preconditioning），是很多后续工作的调度基线。

### 2. 时间步采样：logit-normal

flow matching 的 $$t \sim U[0, 1]$$ 让所有噪声水平等权。SD3 发现**中间的 $$t$$** 更难也更重要（两端要么几乎是数据要么几乎是噪声，预测容易），于是用 **logit-normal** 采样：$$t = \sigma(u)$$，$$u \sim \mathcal{N}(0, 1)$$，让 $$t$$ 集中在 0.5 附近。消融显示它优于均匀采样与几种其他方案。

### 3. 分辨率与调度的耦合

同一个噪声水平在不同分辨率下的"破坏程度"不同：$$1024^2$$ 的图加 $$\sigma = 1$$ 的噪声，相邻像素平均后噪声被抵消一部分，图的低频结构仍清晰；$$256^2$$ 的图加同样的噪声几乎看不出内容。所以高分辨率需要**更多噪声**才能达到同样的破坏。SD3 的做法：把时间步按分辨率**平移**（shift），$$t_{new} = \frac{\alpha t}{1 + (\alpha - 1) t}$$，$$\alpha = \sqrt{m / n}$$（$$m$$、$$n$$ 是两个分辨率的 token 数），$$1024^2$$ 相比 $$256^2$$ 平移 $$\alpha = 3$$。FLUX 沿用（并在采样时也做动态平移）。Simple Diffusion（Hoogeboom 等 2023）更早提出了同样的调度缩放。这是从 $$256^2$$ 到 $$1024^2$$ 直接训练（不用超分级联）成为可能的原因之一。

## 五、classifier-free guidance

### 1. 从 classifier guidance 到 classifier-free

条件生成要采样 $$p(x \mid c)$$（$$c$$ 是文本）。贝叶斯：$$\nabla_x \log p(x \mid c) = \nabla_x \log p(x) + \nabla_x \log p(c \mid x)$$——条件分数 = 无条件分数 + 分类器的梯度。Dhariwal & Nichol 2021 的 **classifier guidance**：训一个在带噪图片上的分类器 $$p_\phi(c \mid x_t)$$，采样时把它的梯度**放大 $$w$$ 倍**加到分数上：$$\tilde s = s + w \nabla \log p_\phi(c \mid x_t)$$。$$w > 1$$ 让样本更"像 $$c$$"，FID 显著改善。缺点是要额外训一个能看噪声图的分类器。

**classifier-free guidance**（Ho & Salimans 2022）：不训分类器，而是让同一个扩散模型**同时学条件与无条件**——训练时以 10–20% 的概率把 $$c$$ 替换为空条件 $$\emptyset$$（条件 dropout）。采样时两次前向：

$$
\tilde\epsilon_\theta(x_t, c) = \epsilon_\theta(x_t, \emptyset) + w \big(\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset)\big)
$$

推导：由贝叶斯 $$\nabla \log p(c \mid x) = \nabla \log p(x \mid c) - \nabla \log p(x)$$，用两个分数的差替代分类器梯度，代入 classifier guidance 的公式：$$\tilde s = s_\emptyset + w(s_c - s_\emptyset)$$，再用 $$\epsilon = -\sigma s$$ 换成噪声。$$w = 1$$ 是纯条件采样；$$w = 0$$ 是无条件；$$w > 1$$ 是**外推**——沿"从无条件指向有条件"的方向走过头。

在 toy 上做一遍：条件 $$c$$ 是"上面的月牙 / 下面的月牙"（两类），训练时 15% 的样本把 $$c$$ 换成 $$\emptyset$$，采样时每步两次前向：

```python
c = torch.where(torch.rand(len(y)) < 0.15, torch.full_like(y, 2), y)          # 训练：15% 的概率把条件换成 ∅（编号 2）

v_c, v_0 = model(x, t, cc), model(x, t, nul)                                  # 采样：有条件、无条件各算一次
v = v_0 + w * (v_c - v_0)                                                      # CFG：沿「有条件 − 无条件」的方向外推 w 倍
```

指定"类 0"，扫 $$w$$：

```text
w = 0：条件「类 0」的 1500 个样本里落在类 0 一侧的 59.9%，样本坐标的标准差 0.827
w = 1：95.1%，标准差 0.602
w = 2：99.6%，标准差 0.459
w = 4：100.0%，标准差 0.354
w = 8：100.0%，标准差 0.417
```

![五张散点图，灰色是全部数据、红色是指定「类 0」生成的样本：w = 0 红点铺满两个月牙；w = 1 红点基本在上面的月牙上、少数落到下面；w = 2 全在上月牙、覆盖整条弧；w = 4 集中在上月牙中段；w = 8 红点被推到月牙外侧、末端甩出一撮离群点](/img/in-post/multimodal-07-cfg.svg)

$$w = 0$$ 无条件，两个月牙各一半；$$w = 1$$ 是纯条件采样，95% 在对的月牙上但有 5% 漏到另一边；$$w = 2$$ 几乎全对且仍覆盖整条弧；$$w = 4$$ 全对，但样本挤到月牙中段——**多样性开始下降**（标准差 0.60 → 0.35）；$$w = 8$$ 样本被推到数据之外、甩出离群点——这就是文生图里"过饱和"的二维版本：外推走过了头，落到了训练分布没有的地方。

### 2. 它在采样什么分布

把 $$\tilde s = (1 - w) s_\emptyset + w\, s_c$$ 看成某个分布的分数：

$$
\tilde p(x \mid c) \propto p(x)^{1 - w}\, p(x \mid c)^{w} = p(x) \left(\frac{p(x \mid c)}{p(x)}\right)^{w} \propto p(x)\, p(c \mid x)^{w}
$$

即 $$p(x \mid c)^w / p(x)^{w-1}$$ 归一化。$$w = 7.5$$（SD 1.x 的默认）意味着**分类器项 $$p(c \mid x)$$ 被升到 7.5 次幂**：一个区域在条件下比无条件下可能 2 倍，被放大到 $$2^{7.5} \approx 180$$ 倍；可能 0.5 倍的区域被压到 $$1/180$$。分布被极度锐化到"最典型地符合 $$c$$"的模式上。这就是核心问题后半的答案——它与 [L6 第一篇](/decoding-strategies-sampling-and-constrained-generation.html)的低温采样是同一类操作（温度 $$1/w$$ 作用在 $$p(c \mid x)$$ 上），带来同样的权衡：**保真度与一致性上升、多样性下降**。

（严格地说，两个分数的线性组合不一定是任何归一化分布的分数——上面的 $$\tilde p$$ 是在每个 $$t$$ 的带噪分布上的近似解释，且 $$\epsilon_\theta$$ 不是精确的分数。但作为理解 $$w$$ 的含义，这个解释是标准的。）

### 3. 副作用与修正

- **过饱和与过曝**：$$w$$ 大时 $$\tilde\epsilon$$ 的范数超出训练分布，像素值被推出 $$[-1, 1]$$，颜色饱和、对比过强。修正：**动态阈值**（Imagen：每步把预测的 $$\hat x_0$$ 按分位数裁剪再缩放）；**CFG rescale**（Lin 等 2024：把 $$\tilde\epsilon$$ 的标准差缩回 $$\epsilon_c$$ 的标准差）。
- **多样性坍缩**：同一 prompt 的样本趋同。修正：**区间 guidance**（Kynkäänniemi 等 2024：只在中间噪声水平用 guidance，两端 $$w = 1$$——大噪声时 guidance 破坏多样性、小噪声时无益）；**自适应 $$w$$**。
- **两倍成本**：每步两次前向（条件与无条件）。修正：**CFG 蒸馏**（Meng 等 2023：训一个直接输出 $$\tilde\epsilon$$ 的学生，$$w$$ 作为输入）；FLUX.1-dev 与 schnell 就是 guidance 蒸馏过的模型，只需一次前向、$$w$$ 作为条件输入。
- **负 prompt**：把 $$\emptyset$$ 换成一段"不想要的描述"$$c^-$$，$$\tilde\epsilon = \epsilon(c^-) + w(\epsilon(c) - \epsilon(c^-))$$——沿"远离 $$c^-$$、靠近 $$c$$"的方向外推。这是社区里"negative prompt"的数学。

### 4. guidance scale 的典型值

SD 1.x / 2.x：7.5；SDXL：5–7；SD3：3.5–7（rectified flow 下需要的 $$w$$ 更小）；FLUX.1-dev（蒸馏）：3.5；Imagen：动态阈值下可到 10+。$$w$$ 与模型、调度、是否蒸馏耦合，不可跨模型比较——与 L6 第一篇的温度同理。

## 六、成本

### 1. 训练

每个训练样本一次前向 + 反向，在一个随机的 $$t$$ 上。网络是 U-Net 或 DiT（下一篇），一次前向的 FLOPs 由输入的空间尺寸决定：SD 1.5 的 U-Net（860M）在 $$64^2 \times 4$$ 的 latent 上约 0.8 TFLOPs 前向；DiT-XL/2（675M）在 $$32^2 \times 4$$（$$256^2$$ 图）上约 0.12 TFLOPs，在 $$64^2$$（$$512^2$$）上约 0.5。训练 SD 1.5 的公开数字：约 15 万 A100 小时（LAION-2B 子集，几个 epoch）；DiT-XL/2 $$256^2$$ 7M 步 × batch 256 = 18 亿样本。扩散模型每个样本只在**一个** $$t$$ 上算——比语言模型"每个 token 都算 loss"的样本效率低，需要更多 epoch。

### 2. 采样

一张图 = 步数 × 每步前向（CFG 下 ×2）。SD 1.5：50 步 × 2 × 0.8 = 80 TFLOPs，A100 上约 2–3 秒；SDXL（2.6B U-Net，$$128^2$$ latent）：约 5 倍。与 LLM 对比：生成 1000 个 token 的 7B LLM 是 $$2 \times 7B \times 1000 = 14$$ TFLOPs 的算力但 memory-bound、约 20–30 秒；一张 SD 图是它的 5 倍 FLOPs 却只要 3 秒——扩散模型的每步是一次大 batch 的 GEMM（整张图的所有 patch 并行），**compute-bound、没有 KV cache、没有自回归的串行**。这是扩散模型服务与 LLM 服务形态不同的根源，下一篇展开。

## 七、动手（建议）

CIFAR-10（$$32^2$$）上从零训两个小模型（同一个 U-Net，约 35M 参数，`diffusers` 的 `UNet2DModel` 或自写），一张 24 GB 卡各几小时：

- **DDPM**（上篇动手节）与 **flow matching**：直线路径，$$v$$-prediction，$$t \sim U[0,1]$$ 与 logit-normal 两种；采样用 Euler 50 / 20 / 10 / 5 步。
- 各训 20 万步 batch 128；用 `torch-fidelity` 或 `clean-fid` 算 FID（对 10K 样本）。
- 加一个条件版本（类别标签，条件 dropout 10%），扫 $$w \in \{1, 2, 4, 8\}$$，看 FID 与 Inception Score（或按类的多样性）的权衡；对 $$w = 8$$ 看像素值的分布是否溢出。
- 对 flow matching 模型做一轮 reflow（用它生成 5 万对 (噪声, 样本)，在固定配对上再训 5 万步），比 1 / 2 / 4 步的 FID。

该看的：flow matching 10 步与 DDIM 10 步的 FID 差；reflow 后 2 步的 FID 是否接近原模型 20 步；logit-normal 是否优于均匀；$$w$$ 增大时 FID 先降后升、多样性单调降。不引用任何未跑过的数字。

配套代码：本文全部 toy 数字与四张图由 [`multimodal/07_flow_score_cfg_toy.py`](https://github.com/arganzheng/ai-learning-labs/blob/main/multimodal/07_flow_score_cfg_toy.py) 产生（`score` / `flow` / `steps` / `cfg` 四个子实验；需要先跑上篇的 `06_ddpm_toy.py train` 得到 DDPM 模型），CPU 一两分钟。

## 八、本文小结

| 项 | 公式 | 备注 |
|---|---|---|
| 分数 | $$s = \nabla \log p_t$$，指向密度上升最快的方向；去噪分数匹配 = 噪声预测，$$\epsilon = -\sigma s$$ | toy：DDPM 模型画成箭头图，箭头指向月牙 |
| Tweedie | $$\mathbb{E}[x_0 \mid x_t] = x_t + \sigma^2 s$$ | 去噪 = 沿分数走一步；给的是后验均值 |
| SDE / ODE | 反向 SDE $$dx = [f - g^2 s] dt + g\, d\bar w$$；概率流 ODE $$dx = [f - \frac{1}{2} g^2 s] dt$$ | 任意求解器；DPM-Solver 10–20 步 |
| flow matching | $$x_t = (1-t) x_0 + t \epsilon$$，$$\mathcal{L} = \lVert v_\theta - (\epsilon - x_0) \rVert^2$$ | 条件速度的期望 = 边缘速度；与 $$\epsilon$$ 线性换算 |
| 统一 | 所有损失 = 加权 ELBO，差别只在噪声水平的权重与路径形状 | Kingma & Gao 2023 |
| 直线 | 随机配对 → 边缘轨迹弯（toy 直线度 0.49）；reflow 在 ODE 配对上重训 → 1.00，一步采样 0.030 | SD3 / FLUX 20–30 步；5 步时 FM 0.041 vs DDIM 0.099 |
| 调度 | cosine；零终端 SNR；logit-normal 采 $$t$$；分辨率平移 $$\alpha = \sqrt{m/n}$$ | 高分辨率需更多噪声 |
| CFG | $$\tilde\epsilon = \epsilon_\emptyset + w(\epsilon_c - \epsilon_\emptyset)$$；采样 $$\propto p(x) p(c \mid x)^w$$ | toy：$$w$$ 1 → 4 命中 95% → 100%、标准差 0.60 → 0.35；$$w = 8$$ 甩出分布外 |
| 修正 | 过饱和 → 动态阈值 / rescale；多样性 → 区间 guidance；两倍成本 → CFG 蒸馏 | FLUX-dev 是蒸馏过的 |
| 成本 | 训练每样本一个 $$t$$；采样步数 × 2（CFG）× 前向；compute-bound、无 KV | SD 1.5 一张图 80 TFLOPs、3 秒 |

## 九、自测

1. 噪声预测 $$\epsilon_\theta$$、分数 $$s = \nabla \log p_t$$、flow matching 的速度 $$v$$ 三者怎么互相换算？

   <details markdown="1"><summary>答案</summary>

   $$\epsilon = -\sigma_t s$$（去噪分数匹配 = 噪声预测）；线性插值路径 $$x_t = (1 - t) x_0 + t\epsilon$$ 下 $$v = \epsilon - x_0$$，与 $$\epsilon$$、$$x_0$$ 线性相关。三个视角训的是同一个网络、不同的参数化。

   </details>

2. flow matching 的每条训练路径都是直线，为什么学出来的采样轨迹会弯？reflow 做了什么？

   <details markdown="1"><summary>答案</summary>

   训练时 $$x_0$$ 与 $$\epsilon$$ 随机配对，不同配对的直线互相交叉，网络在交点只能输出它们的平均速度，轨迹就拐弯（toy 直线度 0.49）。reflow 用训好的模型生成 (噪声, 样本) 配对——它们由 ODE 轨迹连接、不相交——在这些固定配对上重训，轨迹变直（1.00），一步就能采样。

   </details>

3. CFG $$w = 7.5$$ 在数学上做了什么？为什么不是 $$w = 1$$？

   <details markdown="1"><summary>答案</summary>

   $$\hat\epsilon = \epsilon(\varnothing) + w[\epsilon(c) - \epsilon(\varnothing)]$$，等价于从 $$p(x)\, p(c \mid x)^w$$ 采样——把条件似然的幂放大 $$w$$ 倍，让样本更"像 prompt"；$$w = 1$$ 是原始条件分布，多样但贴合度低（toy 上 5% 落错月牙）；7.5 是贴合与多样、饱和之间的经验点。代价是每步两次前向。

   </details>

4. toy 上 $$w = 8$$ 的样本被甩到数据之外，对应文生图里的什么现象？怎么修？

   <details markdown="1"><summary>答案</summary>

   过饱和 / 过曝：$$\tilde\epsilon$$ 的范数超出训练分布，像素被推出 $$[-1, 1]$$。修法：动态阈值（按分位数裁剪 $$\hat x_0$$ 再缩放）、CFG rescale（把 $$\tilde\epsilon$$ 的标准差缩回 $$\epsilon_c$$ 的）、区间 guidance（只在中间噪声水平用 $$w > 1$$）。

   </details>

5. Tweedie 公式说最优去噪器给出 $$\mathbb{E}[x_0 \mid x_t]$$。为什么一步去噪不能直接得到干净的图、要走几十步？

   <details markdown="1"><summary>答案</summary>

   后验均值是"所有可能产生 $$x_t$$ 的 $$x_0$$ 的平均"，噪声大时可能的 $$x_0$$ 很多，平均是一张模糊的图（toy 上落在月牙的中间）。每一步只朝均值走一小段、再让下一步在更低的噪声水平上重新估计，逐步细化才得到清晰的样本——直线度高的模型（reflow 后）例外，因为它的一步就是整条路。

   </details>

## 下一篇

[Latent diffusion、DiT 与文生图配方](/latent-diffusion-dit-and-text-to-image-recipes.html)

[^q0]: 三者学的是同一个对象——每个噪声水平下带噪数据分布的分数 $$\nabla_x \log p_t(x)$$——的三种线性参数化：DDPM 的噪声 $$\epsilon = -\sigma s$$（Tweedie 公式），flow matching 的速度 $$v = \epsilon - x_0$$ 也由 $$x_t$$ 与 $$\epsilon$$ 线性决定；三种训练损失换元后只差一个与噪声水平有关的权重，全部是加权的 ELBO；三者的采样都是解同一个概率流 ODE（或反向 SDE），DDIM 是它的一种离散化，flow matching 的直线参数化让轨迹容易拉直（reflow）、Euler 法少步就够。详见[第二章](#二score-matching分数的视角)、[第三章](#三flow-matching直线的视角)。
[^q1]: 采样分布不是 $$p(x \mid c)$$ 而是 $$\propto p(x)\, p(c \mid x)^{7.5}$$——把「这张图有多符合文本」这一项升到 7.5 次幂，分布被锐化到最典型地符合文本的模式上：一致性与保真度上升、多样性下降（toy：$$w$$ 从 1 到 4，标准差 0.60 → 0.35）、外推过头落到分布之外导致过饱和（toy 的 $$w = 8$$），需要动态阈值或 rescale 修正，且每步要两次前向（除非蒸馏掉）。详见[第五章](#五classifier-free-guidance)。
