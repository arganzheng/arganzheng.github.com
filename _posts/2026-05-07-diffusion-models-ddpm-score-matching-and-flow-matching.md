---
layout: post
series: multimodal
title: "多模态（05）：扩散模型：DDPM、score matching 与 flow matching"
subtitle: "Diffusion Models: DDPM, Score Matching and Flow Matching Are One Thing"
tags: [AI, Multimodal, Diffusion, Generative Models]
catalog: true
updated: 2026-09-14
---

生成线的数学从这里开始。语言模型的生成是"下一个 token 的分类"——目标函数是交叉熵，采样是逐个 token；图像生成走了另一条路：从纯噪声出发，一步步去噪，几十步后得到一张图。这条路在 2020 年由 DDPM 确立、2021 年被 score-based SDE 统一、2023 年被 flow matching 简化，三种视角各有一套推导与记号，读起来像三个不同的东西——实际上它们训练的是同一个网络、只是**参数化不同**，而 SD3 与 FLUX 用的 flow matching 是其中最简洁的一种。

这一篇的目标是把三种视角的推导链完整地走一遍，让读者看清楚它们在哪一步汇合；然后讲两个所有文生图模型都依赖的技术——噪声调度（时间步怎么采）与 classifier-free guidance（$$w = 7.5$$ 到底在做什么）。下一篇再讲怎么把这套数学变成 SD / FLUX 这样的模型。

本篇要回答的核心问题是：

> **DDPM、score matching、flow matching 为什么是同一件事？CFG 的 $$w = 7.5$$ 在数学上意味着什么？**


## 一、总览：三种视角、一个网络

### 1. 一张对照表

| 视角 | 提出 | 前向过程 | 网络学什么 | 训练损失 | 采样 |
|---|---|---|---|---|---|
| DDPM（去噪） | Sohl-Dickstein 2015；Ho 等 2020 | 离散 $$T$$ 步高斯加噪的马尔可夫链 | 噪声 $$\epsilon_\theta(x_t, t)$$ | $$\lVert \epsilon - \epsilon_\theta \rVert^2$$（ELBO 的简化） | $$T$$ 步逐步去噪（随机）；DDIM 确定性、可跳步 |
| Score SDE（分数） | Song 等 2021 | 连续时间 SDE $$dx = f\,dt + g\,dw$$ | 分数 $$s_\theta(x_t, t) \approx \nabla_x \log p_t(x)$$ | 去噪分数匹配 $$\lVert s_\theta - \nabla \log p_t(x_t \mid x_0) \rVert^2$$ | 反向 SDE 或概率流 ODE，任意数值求解器 |
| Flow matching（流） | Lipman 等 2023；Liu 等 2023（rectified flow） | 直线插值 $$x_t = (1-t) x_0 + t\, \epsilon$$ | 速度 $$v_\theta(x_t, t)$$ | $$\lVert (\epsilon - x_0) - v_\theta \rVert^2$$ | ODE $$dx/dt = v_\theta$$，Euler 几十步 |

三行的网络输入都是（带噪样本，时间），输出都是一个与样本同形状的向量场。噪声 $$\epsilon$$、分数 $$s$$、速度 $$v$$ 之间是**线性变换**（第五章给出公式）——训练其中一个就能算出另外两个。差别在：训练时对不同时间步的**加权**不同、采样时的**路径**不同（曲线 vs 直线）。

### 2. 先说答案

三者是同一件事，因为它们都在学同一个对象——每个噪声水平 $$t$$ 下带噪数据分布 $$p_t(x)$$ 的**分数** $$\nabla_x \log p_t(x)$$——的不同参数化。DDPM 学的噪声 $$\epsilon$$ 与分数的关系是 $$s = -\epsilon / \sigma_t$$（Tweedie 公式的推论，第四章）；flow matching 学的速度 $$v$$ 是 $$\epsilon$$ 与 $$x_0$$ 的差，也可以用分数表达。三种损失在换元后只差一个与 $$t$$ 有关的权重。它们的采样都是解同一个概率流 ODE（或对应的 SDE），只是 flow matching 选择的前向过程让 ODE 的轨迹接近直线，因而少步数就够。

CFG 的 $$w = 7.5$$ 意味着采样的分布不是 $$p(x \mid c)$$，而是 $$p(x \mid c)^{w} / p(x)^{w-1}$$ 的归一化——把条件分布"锐化"了 $$w$$ 倍：条件 $$c$$ 下比无条件更可能的区域被放大 $$w$$ 次幂，其他区域被压掉。$$w = 1$$ 是原条件分布；$$w = 7.5$$ 让样本强烈偏向"最典型地符合 $$c$$"的模式——保真度与文本一致性上升、多样性下降、颜色饱和度过高（over-saturation）。它是一个**有意的分布改变**，与 [L6 第一篇](/decoding-strategies-sampling-and-constrained-generation.html)的低温采样是同类操作。第七章展开。

### 3. 本文的章节安排

| 章 | 主题 | 内容 |
|---|---|---|
| 二 | DDPM | 前向加噪的闭式解；反向过程；ELBO 的逐项推导；化简为噪声预测的 MSE；三种预测目标 |
| 三 | DDIM 与确定性采样 | 非马尔可夫的推广；$$\eta = 0$$ 的 ODE；跳步 |
| 四 | score matching | 分数；Tweedie 公式；去噪分数匹配等价于噪声预测；SDE 与概率流 ODE |
| 五 | flow matching | 条件流；直线路径的速度场；与前两者的换算；为什么直线让步数少 |
| 六 | 噪声调度与时间步采样 | linear / cosine / 方差保持；SNR 视角；logit-normal 采样；分辨率对调度的影响 |
| 七 | classifier-free guidance | 从贝叶斯到 $$\tilde\epsilon$$；它在采样什么分布；副作用与修正（动态阈值、区间 guidance、CFG 蒸馏） |
| 八 | 成本 | 训练与采样的 FLOPs；与 LLM 的对比 |
| 九 | 动手（建议） | CIFAR-10 上的 DDPM vs flow matching |
| 十 | 本文小结 | |


## 二、DDPM：去噪扩散概率模型

### 1. 前向过程

两条链方向相反：前向是固定的、没有参数的加噪，反向是学出来的去噪，每一步都由同一个网络 $$\epsilon_\theta(x_t, t)$$ 完成：

```text
前向 q（固定，无参数）：逐步加高斯噪声，T 步后变成纯噪声
   x_0 ──q(x_1|x_0)──► x_1 ──► x_2 ──► ... ──► x_{t−1} ──q(x_t|x_{t−1})──► x_t ──► ... ──► x_T ≈ N(0, I)
  数据                                                                                        纯噪声
   ▲                                                                                            │
   │  闭式 q(x_t | x_0)：训练时任取一个 t，一步采出 x_t，不用跑整条链                                 │
   │                                                                                            │
   x_0 ◄── p_θ(x_0|x_1) ◄── x_1 ◄── ... ◄── x_{t−1} ◄──p_θ(x_{t−1}|x_t)──── x_t ◄── ... ◄────── x_T
反向 p_θ（学习）：每一步用同一个网络 ε_θ(x_t, t) 预测噪声，减掉一点，T 步（或 DDIM 的几十步）回到数据
```

给数据 $$x_0 \sim q(x_0)$$，定义 $$T$$ 步的加噪链，每步加一点高斯噪声：

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\big(x_t;\ \sqrt{1 - \beta_t}\, x_{t-1},\ \beta_t I\big), \qquad t = 1, \ldots, T
$$

$$\beta_t$$ 是预设的噪声调度（DDPM：$$T = 1000$$，$$\beta_t$$ 从 $$10^{-4}$$ 线性到 $$0.02$$）。记 $$\alpha_t = 1 - \beta_t$$，$$\bar\alpha_t = \prod_{s \le t} \alpha_s$$。高斯的组合仍是高斯，反复代入得到从 $$x_0$$ 直接到 $$x_t$$ 的**闭式**：

$$
q(x_t \mid x_0) = \mathcal{N}\big(x_t;\ \sqrt{\bar\alpha_t}\, x_0,\ (1 - \bar\alpha_t) I\big), \qquad
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon, \ \ \epsilon \sim \mathcal{N}(0, I)
$$

推导：$$x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \epsilon_{t-1} = \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{\alpha_t (1 - \alpha_{t-1})} \epsilon_{t-2} + \sqrt{1 - \alpha_t} \epsilon_{t-1}$$，两个独立高斯之和的方差相加：$$\alpha_t(1 - \alpha_{t-1}) + (1 - \alpha_t) = 1 - \alpha_t \alpha_{t-1}$$，归纳得到。$$\bar\alpha_T \approx 0$$ 时 $$x_T$$ 接近纯噪声——前向过程把任何数据分布变成标准高斯。这个闭式让训练可以**任取一个 $$t$$ 直接采 $$x_t$$**，不用跑整条链。

### 2. 反向过程

生成是反过来：从 $$x_T \sim \mathcal{N}(0, I)$$ 出发，逐步 $$x_T \to x_{T-1} \to \cdots \to x_0$$。真实的反向条件分布 $$q(x_{t-1} \mid x_t)$$ 依赖整个数据分布、不可得；但**给定 $$x_0$$ 的**反向条件 $$q(x_{t-1} \mid x_t, x_0)$$ 是可算的高斯（贝叶斯公式 + 两个高斯相乘）：

$$
q(x_{t-1} \mid x_t, x_0) = \mathcal{N}\big(x_{t-1};\ \tilde\mu_t(x_t, x_0),\ \tilde\beta_t I\big)
$$

$$
\tilde\mu_t = \frac{\sqrt{\bar\alpha_{t-1}}\, \beta_t}{1 - \bar\alpha_t}\, x_0 + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t}\, x_t, \qquad
\tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\, \beta_t
$$

模型用一个参数化的高斯 $$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$ 去逼近它，$$\sigma_t^2$$ 固定为 $$\beta_t$$ 或 $$\tilde\beta_t$$（两者效果相近；后来的改进让它可学习）。

### 3. ELBO

最大化 $$\log p_\theta(x_0)$$ 不可直接算，用变分下界（与 VAE 相同的推导，把前向过程 $$q(x_{1:T} \mid x_0)$$ 当作变分后验）：

$$
-\log p_\theta(x_0) \le \mathbb{E}_q\Big[ \underbrace{D_{KL}(q(x_T \mid x_0) \,\|\, p(x_T))}_{L_T} + \sum_{t=2}^{T} \underbrace{D_{KL}\big(q(x_{t-1} \mid x_t, x_0) \,\|\, p_\theta(x_{t-1} \mid x_t)\big)}_{L_{t-1}} \underbrace{- \log p_\theta(x_0 \mid x_1)}_{L_0} \Big]
$$

$$L_T$$ 与 $$\theta$$ 无关（前向过程固定，$$x_T$$ 接近标准高斯）；$$L_0$$ 是最后一步的重建项。中间的 $$L_{t-1}$$ 是两个高斯的 KL——方差都固定，KL 只剩均值的差：

$$
L_{t-1} = \frac{1}{2\sigma_t^2} \big\lVert \tilde\mu_t(x_t, x_0) - \mu_\theta(x_t, t) \big\rVert^2 + C
$$

### 4. 化简为噪声预测

把 $$x_0 = \frac{1}{\sqrt{\bar\alpha_t}}(x_t - \sqrt{1 - \bar\alpha_t}\, \epsilon)$$ 代入 $$\tilde\mu_t$$，整理后

$$
\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}} \Big( x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\, \epsilon \Big)
$$

于是自然地把 $$\mu_\theta$$ 也参数化成同样的形式、让网络预测 $$\epsilon$$：

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \Big( x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\, \epsilon_\theta(x_t, t) \Big)
$$

代回 $$L_{t-1}$$：

$$
L_{t-1} = \frac{\beta_t^2}{2 \sigma_t^2 \alpha_t (1 - \bar\alpha_t)} \big\lVert \epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon, t) \big\rVert^2
$$

Ho 等的关键经验发现：**去掉前面那个与 $$t$$ 有关的权重**，用

$$
L_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \big\lVert \epsilon - \epsilon_\theta(x_t, t) \big\rVert^2
$$

效果更好。这个权重在 $$t$$ 小（噪声少）时很大——ELBO 要求精确重建细节——而实际生成质量更依赖 $$t$$ 大时（决定全局结构）的准确性；去掉权重等于对大 $$t$$ 加权。**训练算法**：采一张图 $$x_0$$、一个 $$t \sim \text{Uniform}\{1..T\}$$、一个 $$\epsilon$$，算 $$x_t$$，让网络从 $$(x_t, t)$$ 预测 $$\epsilon$$，MSE。就这么简单。**采样**：从 $$x_T$$ 开始，每步 $$x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z$$，$$z \sim \mathcal{N}(0, I)$$，1000 步。

### 5. 三种预测目标

网络可以预测 $$\epsilon$$、预测 $$x_0$$、或预测 $$v$$（velocity，Salimans & Ho 2022：$$v = \sqrt{\bar\alpha_t}\, \epsilon - \sqrt{1 - \bar\alpha_t}\, x_0$$）。三者线性相关，知道其一与 $$x_t$$ 就能算另外两个：$$x_0 = (x_t - \sqrt{1 - \bar\alpha_t}\, \epsilon) / \sqrt{\bar\alpha_t}$$。差别在数值行为：预测 $$\epsilon$$ 在 $$t \to T$$（几乎纯噪声）时容易——$$x_t \approx \epsilon$$——但在 $$t \to 0$$ 时从 $$\epsilon$$ 恢复 $$x_0$$ 要除以接近零的 $$\sqrt{1 - \bar\alpha_t}$$，误差放大；预测 $$x_0$$ 相反。$$v$$-prediction 在两端都稳定，是 SDXL 精炼模型、Imagen Video 等的选择，且在**零终端 SNR** 的调度下（第六章）是必需的——那时 $$t = T$$ 处 $$x_T$$ 完全是噪声，预测 $$\epsilon$$ 就是输出输入本身、网络学不到东西。


## 三、DDIM 与确定性采样

### 1. 非马尔可夫的前向过程

DDPM 的 1000 步采样太慢。DDIM（Song 等 2020）的观察：DDPM 的训练目标只依赖边缘分布 $$q(x_t \mid x_0)$$（每个 $$t$$ 独立地采），不依赖前向链是否马尔可夫。可以定义一族**非马尔可夫**的前向过程，边缘分布相同、但 $$q(x_{t-1} \mid x_t, x_0)$$ 不同——同一个训好的 $$\epsilon_\theta$$ 适用于这一族里的任何一个。这一族由参数 $$\eta$$ 索引，反向更新：

$$
x_{t-1} = \sqrt{\bar\alpha_{t-1}}\, \underbrace{\frac{x_t - \sqrt{1 - \bar\alpha_t}\, \epsilon_\theta}{\sqrt{\bar\alpha_t}}}_{\hat x_0} + \sqrt{1 - \bar\alpha_{t-1} - \sigma_t^2}\, \epsilon_\theta + \sigma_t z
$$

$$\sigma_t = \eta \sqrt{\tilde\beta_t}$$。$$\eta = 1$$ 回到 DDPM（随机）；$$\eta = 0$$ **没有随机项**——给定 $$x_T$$，整条轨迹确定。直觉：先用 $$\epsilon_\theta$$ 估出 $$\hat x_0$$，再按 $$t - 1$$ 的噪声水平重新"加噪"到 $$x_{t-1}$$——但用的是**预测的**噪声方向而不是新采的。

### 2. 跳步

因为不依赖马尔可夫性，可以在时间步的子序列 $$\{\tau_1, \ldots, \tau_S\} \subset \{1, \ldots, T\}$$ 上做同样的更新（把 $$t - 1$$ 换成 $$\tau_{i-1}$$）。$$S = 50$$ 步的 DDIM 与 1000 步的 DDPM 质量接近；$$S = 20$$ 也可用。20–50 倍的加速，不需要重训。这是 Stable Diffusion 默认 20–50 步的来源。

### 3. 确定性的意义

$$\eta = 0$$ 的 DDIM 定义了一个从噪声到数据的**确定性映射**——它是一个 ODE 的离散化（第四章第四节）。确定性带来：可以对 $$x_T$$ 做插值得到语义上平滑的图像插值；可以把一张图**编码**回噪声（反向跑 ODE）再编辑；以及采样的步数可以用更好的 ODE 求解器（DPM-Solver、UniPC）进一步减少到 10–15 步。


## 四、score matching：分数的视角

### 1. 分数

分布 $$p(x)$$ 的**分数**是 $$\nabla_x \log p(x)$$——在每一点指向概率密度上升最快的方向。知道分数就能采样：Langevin 动力学 $$x \leftarrow x + \frac{\delta}{2} \nabla_x \log p(x) + \sqrt{\delta}\, z$$ 反复迭代收敛到 $$p$$。分数**不需要归一化常数**（$$\nabla \log(p / Z) = \nabla \log p$$），这是它相对于直接建模密度的优势。

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

后验均值等于观测加 $$\sigma^2$$ 倍的分数。它把三样东西连起来：**去噪**（估 $$\mathbb{E}[x_0 \mid \tilde x]$$）、**分数**、**噪声预测**（$$\mathbb{E}[x_0 \mid \tilde x] = \tilde x - \sigma \mathbb{E}[\epsilon \mid \tilde x]$$）。一个最优的去噪器自动给出分数，反之亦然。这是"为什么去噪等于学分布"的最短回答。

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


## 五、flow matching：直线的视角

### 1. 从 ODE 直接出发

既然采样最终是解一个 ODE $$dx/dt = v(x, t)$$，为什么不**直接学速度场 $$v$$**？这是 flow matching（Lipman 等 2023）与 rectified flow（Liu 等 2023）的出发点。定义一条从数据 $$x_0$$ 到噪声 $$x_1 = \epsilon$$ 的路径（注意这里 $$t = 0$$ 是数据、$$t = 1$$ 是噪声，与 DDPM 的记号相反——这是 SD3 / FLUX 的约定），最简单的路径是**直线**：

$$
x_t = (1 - t)\, x_0 + t\, \epsilon, \qquad \frac{dx_t}{dt} = \epsilon - x_0
$$

给定 $$(x_0, \epsilon)$$ 这一对，路径上每一点的速度是常数 $$\epsilon - x_0$$。**条件流匹配**的损失：

$$
\mathcal{L}_{CFM} = \mathbb{E}_{t \sim U[0,1],\ x_0,\ \epsilon} \big\lVert v_\theta(x_t, t) - (\epsilon - x_0) \big\rVert^2
$$

Lipman 等证明：这个只用"配对的条件速度"的损失，与匹配**边缘**速度场（把所有经过 $$x_t$$ 的配对平均）的损失有相同的梯度——所以 $$v_\theta$$ 学到的是边缘速度场，解 $$dx/dt = v_\theta$$ 从 $$\epsilon$$ 到 $$t = 0$$ 就得到数据分布的样本。推导的结构与去噪分数匹配（条件分数的期望 = 边缘分数）完全平行。

### 2. 与 DDPM 的换算

直线路径 $$x_t = (1 - t) x_0 + t \epsilon$$ 与 DDPM 的 $$x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon$$ 是同一族（$$x_t = a_t x_0 + b_t \epsilon$$）中的两个成员：DDPM 是**方差保持**（$$a_t^2 + b_t^2 = 1$$，圆弧），flow matching 是 $$a_t + b_t = 1$$（直线）。在任何这样的路径上，$$\epsilon$$、$$x_0$$、$$v = \epsilon - x_0$$、分数 $$s$$ 都由 $$x_t$$ 与其中一个线性决定：

$$
\epsilon = x_t + (1 - t)\, v, \qquad x_0 = x_t - t\, v, \qquad s = -\frac{\epsilon}{t} = -\frac{x_t + (1-t) v}{t}
$$

（直线路径下 $$\sigma_t = t$$。）所以 flow matching 的 $$v$$-预测与 DDPM 的 $$\epsilon$$-预测学的是同一个信息，训练损失只差一个与 $$t$$ 有关的权重（$$\lVert v - v_\theta \rVert^2 = \lVert \epsilon - \epsilon_\theta \rVert^2 / (1-t)^2$$ 一类）。Kingma & Gao 2023 把所有这些损失统一成"加权的 ELBO"——不同方法的差别**全部**归结为对不同噪声水平的加权。这是核心问题前半的完整答案。

### 3. 为什么直线让步数少

采样是解 ODE。ODE 的轨迹越直，Euler 法一步走得越远、误差越小。DDPM 的 VP 路径是圆弧，且边缘速度场在中间时刻弯曲；flow matching 的**条件**路径是直线，但**边缘**速度场（多条直线的平均）不一定直——两条相交的直线平均后在交点附近弯。Rectified flow 的 **reflow** 操作：用训好的模型从噪声生成样本得到新的 $$(x_0, \epsilon)$$ 配对（它们由 ODE 轨迹连接、不相交），在新配对上重训——每轮 reflow 让边缘轨迹更直，几轮后可以 1–2 步采样。SD3 与 FLUX 没有做 reflow，但直线路径本身已经让它们在 20–30 步达到 DDPM 50 步的质量，且更容易做后续的步数蒸馏（下一篇）。

另一个实用的好处：直线路径下 $$t$$ 的语义直接——$$t = 0.5$$ 就是"一半数据一半噪声"——调度设计更直观（第六章）。


## 六、噪声调度与时间步采样

### 1. 调度决定"什么噪声水平被学得多"

调度是 $$t \mapsto (a_t, b_t)$$ 或等价地 $$t \mapsto \text{SNR}(t) = a_t^2 / b_t^2$$。它与训练时 $$t$$ 的采样分布一起决定每个噪声水平在训练里出现的频率——即网络在哪些噪声水平上学得好。

- **DDPM linear**（$$\beta_t$$ 线性）：SNR 在中间下降太快，很多步浪费在几乎纯噪声的区域。
- **cosine**（Nichol & Dhariwal 2021）：$$\bar\alpha_t = \cos^2(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2})$$，SNR 下降更均匀，在小数据集上提升明显。
- **零终端 SNR**（Lin 等 2024）：DDPM linear 在 $$t = T$$ 处 $$\bar\alpha_T \approx 0.0047 \ne 0$$——训练时最噪的样本仍含一点信号，但采样从纯噪声开始，训练与采样不一致；表现是生成的图片平均亮度总是中等（不能生成很暗或很亮的图）。修正：把调度缩放到 $$\bar\alpha_T = 0$$，用 $$v$$-prediction（此时 $$\epsilon$$-prediction 无定义），采样从真正的纯噪声开始。
- **EDM**（Karras 等 2022）：在 $$\sigma$$ 空间用对数正态分布采样噪声水平，配合网络输入输出的预处理（preconditioning），是很多后续工作的调度基线。

### 2. 时间步采样：logit-normal

flow matching 的 $$t \sim U[0, 1]$$ 让所有噪声水平等权。SD3 发现**中间的 $$t$$** 更难也更重要（两端要么几乎是数据要么几乎是噪声，预测容易），于是用 **logit-normal** 采样：$$t = \sigma(u)$$，$$u \sim \mathcal{N}(0, 1)$$，让 $$t$$ 集中在 0.5 附近。消融显示它优于均匀采样与几种其他方案。

### 3. 分辨率与调度的耦合

同一个噪声水平在不同分辨率下的"破坏程度"不同：$$1024^2$$ 的图加 $$\sigma = 1$$ 的噪声，相邻像素平均后噪声被抵消一部分，图的低频结构仍清晰；$$256^2$$ 的图加同样的噪声几乎看不出内容。所以高分辨率需要**更多噪声**才能达到同样的破坏。SD3 的做法：把时间步按分辨率**平移**（shift），$$t_{new} = \frac{\alpha t}{1 + (\alpha - 1) t}$$，$$\alpha = \sqrt{m / n}$$（$$m$$、$$n$$ 是两个分辨率的 token 数），$$1024^2$$ 相比 $$256^2$$ 平移 $$\alpha = 3$$。FLUX 沿用（并在采样时也做动态平移）。Simple Diffusion（Hoogeboom 等 2023）更早提出了同样的调度缩放。这是从 $$256^2$$ 到 $$1024^2$$ 直接训练（不用超分级联）成为可能的原因之一。


## 七、classifier-free guidance

### 1. 从 classifier guidance 到 classifier-free

条件生成要采样 $$p(x \mid c)$$（$$c$$ 是文本）。贝叶斯：$$\nabla_x \log p(x \mid c) = \nabla_x \log p(x) + \nabla_x \log p(c \mid x)$$——条件分数 = 无条件分数 + 分类器的梯度。Dhariwal & Nichol 2021 的 **classifier guidance**：训一个在带噪图片上的分类器 $$p_\phi(c \mid x_t)$$，采样时把它的梯度**放大 $$w$$ 倍**加到分数上：$$\tilde s = s + w \nabla \log p_\phi(c \mid x_t)$$。$$w > 1$$ 让样本更"像 $$c$$"，FID 显著改善。缺点是要额外训一个能看噪声图的分类器。

**classifier-free guidance**（Ho & Salimans 2022）：不训分类器，而是让同一个扩散模型**同时学条件与无条件**——训练时以 10–20% 的概率把 $$c$$ 替换为空条件 $$\emptyset$$（条件 dropout）。采样时两次前向：

$$
\tilde\epsilon_\theta(x_t, c) = \epsilon_\theta(x_t, \emptyset) + w \big(\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset)\big)
$$

推导：由贝叶斯 $$\nabla \log p(c \mid x) = \nabla \log p(x \mid c) - \nabla \log p(x)$$，用两个分数的差替代分类器梯度，代入 classifier guidance 的公式：$$\tilde s = s_\emptyset + w(s_c - s_\emptyset)$$，再用 $$\epsilon = -\sigma s$$ 换成噪声。$$w = 1$$ 是纯条件采样；$$w = 0$$ 是无条件；$$w > 1$$ 是外推。

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


## 八、成本

### 1. 训练

每个训练样本一次前向 + 反向，在一个随机的 $$t$$ 上。网络是 U-Net 或 DiT（下一篇），一次前向的 FLOPs 由输入的空间尺寸决定：SD 1.5 的 U-Net（860M）在 $$64^2 \times 4$$ 的 latent 上约 0.8 TFLOPs 前向；DiT-XL/2（675M）在 $$32^2 \times 4$$（$$256^2$$ 图）上约 0.12 TFLOPs，在 $$64^2$$（$$512^2$$）上约 0.5。训练 SD 1.5 的公开数字：约 15 万 A100 小时（LAION-2B 子集，几个 epoch）；DiT-XL/2 $$256^2$$ 7M 步 × batch 256 = 18 亿样本。扩散模型每个样本只在**一个** $$t$$ 上算——比语言模型"每个 token 都算 loss"的样本效率低，需要更多 epoch。

### 2. 采样

一张图 = 步数 × 每步前向（CFG 下 ×2）。SD 1.5：50 步 × 2 × 0.8 = 80 TFLOPs，A100 上约 2–3 秒；SDXL（2.6B U-Net，$$128^2$$ latent）：约 5 倍。与 LLM 对比：生成 1000 个 token 的 7B LLM 是 $$2 \times 7B \times 1000 = 14$$ TFLOPs 的算力但 memory-bound、约 20–30 秒；一张 SD 图是它的 5 倍 FLOPs 却只要 3 秒——扩散模型的每步是一次大 batch 的 GEMM（整张图的所有 patch 并行），**compute-bound、没有 KV cache、没有自回归的串行**。这是扩散模型服务与 LLM 服务形态不同的根源，下一篇展开。


## 九、动手（建议）

CIFAR-10（$$32^2$$）上从零训两个小模型（同一个 U-Net，约 35M 参数，`diffusers` 的 `UNet2DModel` 或自写），一张 24 GB 卡各几小时：

- **DDPM**：linear 调度，$$T = 1000$$，$$\epsilon$$-prediction，$$L_{simple}$$；采样用 DDPM 1000 步与 DDIM 50 / 20 / 10 步。
- **flow matching**：直线路径，$$v$$-prediction，$$t \sim U[0,1]$$ 与 logit-normal 两种；采样用 Euler 50 / 20 / 10 / 5 步。
- 各训 20 万步 batch 128；用 `torch-fidelity` 或 `clean-fid` 算 FID（对 10K 样本）。
- 加一个条件版本（类别标签，条件 dropout 10%），扫 $$w \in \{1, 2, 4, 8\}$$，看 FID 与 Inception Score（或按类的多样性）的权衡；对 $$w = 8$$ 看像素值的分布是否溢出。

该看的：DDIM 50 步是否接近 DDPM 1000 步；flow matching 10 步与 DDIM 10 步的 FID 差（预期 flow matching 更好——直线）；logit-normal 是否优于均匀；$$w$$ 增大时 FID 先降后升、多样性单调降。不引用任何未跑过的数字。


## 十、本文小结

| 项 | 公式 | 备注 |
|---|---|---|
| 前向 | $$x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon$$ | 闭式；任取 $$t$$ 训练 |
| 反向后验 | $$q(x_{t-1} \mid x_t, x_0)$$ 是高斯，均值 $$\tilde\mu_t$$ | 贝叶斯 + 高斯相乘 |
| ELBO → MSE | $$L_{t-1} \propto \lVert \epsilon - \epsilon_\theta \rVert^2$$；去掉权重得 $$L_{simple}$$ | 去权重 = 对大噪声加权 |
| 预测目标 | $$\epsilon$$ / $$x_0$$ / $$v$$ 线性相关 | $$v$$ 两端稳定；零终端 SNR 需 $$v$$ |
| DDIM | $$x_{t-1} = \sqrt{\bar\alpha_{t-1}} \hat x_0 + \sqrt{1 - \bar\alpha_{t-1} - \sigma_t^2}\, \epsilon_\theta + \sigma_t z$$，$$\eta = 0$$ 确定性 | 跳步 50 步；概率流 ODE 的离散化 |
| 分数 | $$s = \nabla \log p_t$$；去噪分数匹配 = 噪声预测，$$\epsilon = -\sigma s$$ | Tweedie：$$\mathbb{E}[x_0 \mid x_t] = x_t + \sigma^2 s$$ |
| SDE / ODE | 反向 SDE $$dx = [f - g^2 s] dt + g\, d\bar w$$；概率流 ODE $$dx = [f - \frac{1}{2} g^2 s] dt$$ | 任意求解器；DPM-Solver 10–20 步 |
| flow matching | $$x_t = (1-t) x_0 + t \epsilon$$，$$\mathcal{L} = \lVert v_\theta - (\epsilon - x_0) \rVert^2$$ | 条件速度的期望 = 边缘速度；与 $$\epsilon$$ 线性换算 |
| 统一 | 所有损失 = 加权 ELBO，差别只在噪声水平的权重与路径形状 | Kingma & Gao 2023 |
| 直线 | 边缘轨迹更直 → Euler 少步；reflow 进一步拉直 | SD3 / FLUX 20–30 步 |
| 调度 | cosine；零终端 SNR；logit-normal 采 $$t$$；分辨率平移 $$\alpha = \sqrt{m/n}$$ | 高分辨率需更多噪声 |
| CFG | $$\tilde\epsilon = \epsilon_\emptyset + w(\epsilon_c - \epsilon_\emptyset)$$；采样 $$\propto p(x) p(c \mid x)^w$$ | 锐化 $$w$$ 次幂；过饱和 → 动态阈值 / rescale；区间 guidance；蒸馏去掉两倍成本 |
| 成本 | 训练每样本一个 $$t$$；采样步数 × 2（CFG）× 前向；compute-bound、无 KV | SD 1.5 一张图 80 TFLOPs、3 秒 |

核心问题的答案：DDPM、score matching、flow matching 学的是同一个对象——每个噪声水平下带噪数据分布的分数 $$\nabla_x \log p_t(x)$$——的三种线性参数化：DDPM 的噪声 $$\epsilon = -\sigma s$$（Tweedie 公式），flow matching 的速度 $$v = \epsilon - x_0$$ 也由 $$x_t$$ 与 $$\epsilon$$ 线性决定；三种训练损失换元后只差一个与噪声水平有关的权重，全部是加权的 ELBO；三者的采样都是解同一个概率流 ODE（或反向 SDE），DDIM 是它的一种离散化，flow matching 的直线路径让 ODE 轨迹更直、Euler 法少步就够。CFG 的 $$w = 7.5$$ 意味着采样分布不是 $$p(x \mid c)$$ 而是 $$\propto p(x)\, p(c \mid x)^{7.5}$$——把"这张图有多符合文本"这一项升到 7.5 次幂，分布被锐化到最典型地符合文本的模式上：一致性与保真度上升、多样性下降、像素溢出导致过饱和，需要动态阈值或 rescale 修正，且每步要两次前向（除非蒸馏掉）。下一篇讲怎么把这套数学变成 SD 与 FLUX：latent 空间、DiT、文本编码器与采样加速。


## 下一篇

[Latent diffusion、DiT 与文生图配方](/latent-diffusion-dit-and-text-to-image-recipes.html)
