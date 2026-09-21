---
layout: post
series: math-for-ai
title: "算法工程师的数学（06）：熵、交叉熵与 KL——从困惑度到 DPO"
subtitle: "Entropy, Cross-Entropy and KL Divergence: From Perplexity to the DPO Loss"
tags: [AI, LLM, Math]
catalog: true
updated: 2026-09-17
---

信息论在这张地图上看起来最"理论"，却是后训练的主语言：SFT 的 loss 是交叉熵，RLHF 与 DPO 的约束项是 KL，蒸馏的目标是 KL，投机解码的接受率是两个分布的总变差。它们全部建立在三个量上——**熵、交叉熵、KL 散度**——以及一条把三者连起来的等式。这一篇从定义讲起，讲清 KL 的两个方向为什么行为完全不同，然后走完一条完整的推导链：从"最大化奖励且不偏离参考模型"的目标，推出它的闭式最优解，再推出 DPO 的 loss。

全篇的核心问题是：

> **能不能分清熵、交叉熵、KL 各是什么？[^q0] 能不能从 KL 约束的最优策略推出 DPO 的 loss？[^q1]**

## 一、总览

### 1. 三个量一条等式

| 量 | 定义 | 含义 | 在 LLM 里 |
|---|---|---|---|
| 熵 | $$H(p) = -\sum p \log p$$ | $$p$$ 自己的不确定程度 | 训练 loss 的下限（数据的熵） |
| 交叉熵 | $$H(p, q) = -\sum p \log q$$ | 用 $$q$$ 编码 $$p$$ 的平均码长 | 训练 loss（$$p$$ = one-hot 时 $$= -\log q(x_t)$$） |
| KL | $$D(p \,\Vert\, q) = \sum p \log (p/q)$$ | 用 $$q$$ 代替 $$p$$ 多付的码长 | RLHF / DPO 的约束项、蒸馏的目标 |

$$
H(p, q) = H(p) + D(p \,\Vert\, q) \qquad \text{交叉熵 = 熵 + KL}
$$

### 2. 本文的章节安排

| 章 | 主题 | 内容 |
|---|---|---|
| 二 | 熵与困惑度 | 熵的定义与直觉、nat 与 bit、PPL = $$e^{\text{loss}}$$ |
| 三 | 交叉熵 = 熵 + KL | 三者的定义与等式、训练时的 $$p$$ 是 one-hot、蒸馏时是软分布、loss 的下限 |
| 四 | KL 的方向 | forward 与 reverse、mode-covering 与 mode-seeking、RLHF 用哪个 |
| 五 | Bradley-Terry | 偏好如何变成概率；奖励模型的 loss 是逻辑回归 |
| 六 | 从 KL 约束的最优策略到 DPO | 闭式解、反解奖励、代入、$$Z(x)$$ 抵消 |
| 七 | 其他两处 | 总变差与投机解码的接受率；互信息与对比学习 |
| 八 | 本文小结 | |
| 九 | 自测 | 五道题 |

## 二、熵与困惑度

### 1. 熵

分布 $$p$$ 的**熵**（entropy）是

$$
H(p) = -\sum_x p(x) \log p(x) = \mathbb{E}_{x \sim p}\big[-\log p(x)\big]
$$

——"$$-\log p(x)$$"这个量（上一篇说它是"看到 $$x$$ 时的惩罚"，信息论里叫它 $$x$$ 的**信息量**：越不可能的事发生了信息越大）在 $$p$$ 下的期望。它度量分布的**不确定程度**：

| 分布 | 熵 | |
|---|---|---|
| 确定性：$$p = (1, 0, 0, 0)$$ | $$-1 \cdot \log 1 = 0$$ | 完全确定，没有信息 |
| 均匀：$$p = (1/4, 1/4, 1/4, 1/4)$$ | $$-4 \cdot \frac{1}{4} \log \frac{1}{4} = \log 4$$ | 最大——$$V$$ 个取值的均匀分布熵是 $$\log V$$ |
| 偏斜：$$p = (0.7, 0.1, 0.1, 0.1)$$ | $$-(0.7 \log 0.7 + 3 \cdot 0.1 \log 0.1) \approx 0.94$$ nat | 介于两者之间 |

熵的单位与对数的底一致：自然对数是 nat，以 2 为底是 bit（1 nat ≈ 1.44 bit）。"熵是 $$\log V$$"对应上一篇"初始 loss 是 $$\ln V$$"——随机初始化的模型输出接近均匀分布，交叉熵等于均匀分布的熵。

### 2. 困惑度

训练日志里的 loss（每 token 交叉熵，nat）有一种更直觉的读法：**困惑度**（perplexity）

$$
\text{PPL} = e^{\text{loss}}
$$

它的含义是"模型每一步平均在几个 token 里犹豫"：若模型在每个位置都对 $$k$$ 个 token 给均匀概率 $$1/k$$，loss $$= \ln k$$，PPL $$= k$$。所以 loss 1.8 nat 对应 PPL $$e^{1.8} = 6.05$$——平均每步在 6 个候选里选。同一个 loss 换成 bit 是 $$1.8 \times 1.44 = 2.6$$ bit/token。

| loss (nat) | 2.5 | 2.0 | 1.8 | 1.5 | 1.0 |
|---|---:|---:|---:|---:|---:|
| PPL | 12.2 | 7.39 | 6.05 | 4.48 | 2.72 |
| bit/token | 3.61 | 2.89 | 2.60 | 2.16 | 1.44 |

这两个换算要熟到不用计算器：看到 PPL 6 知道 loss 1.8，看到 loss 2.0 知道 PPL 7.4。

注意困惑度依赖 tokenizer：同一段文本切成更多 token，每个 token 更好猜，PPL 更低但不代表模型更好。比较不同 tokenizer 的模型要用 bits per byte（按字节而不是按 token 归一），L4 预训练系列的第一篇讨论这个陷阱。

## 三、交叉熵 = 熵 + KL

### 1. 定义

用分布 $$q$$ 去描述（编码）实际来自 $$p$$ 的数据，平均要付的码长是**交叉熵**：

$$
H(p, q) = -\sum_x p(x) \log q(x) = \mathbb{E}_{x \sim p}\big[-\log q(x)\big]
$$

与熵的差别只在 $$\log$$ 里面是 $$q$$ 不是 $$p$$：数据按 $$p$$ 出现，但我们用 $$q$$ 给它们编码。用错了分布会多付码长，多付的那部分就是 **KL 散度**（Kullback–Leibler divergence，相对熵）：

$$
D_{\mathrm{KL}}(p \,\Vert\, q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = H(p, q) - H(p)
$$

移项就是本篇的中心等式：

$$
H(p, q) = H(p) + D_{\mathrm{KL}}(p \,\Vert\, q)
$$

KL 的两条性质：**非负**（$$D_{\mathrm{KL}} \ge 0$$，用错分布只会多付不会少付；等于零当且仅当 $$p = q$$），**不对称**（$$D_{\mathrm{KL}}(p \Vert q) \ne D_{\mathrm{KL}}(q \Vert p)$$，下一章）。因为不对称，它不是数学意义上的"距离"，但日常仍说"两个分布的 KL 距离"。

一个手算的例子：$$p = (0.5, 0.5)$$，$$q = (0.9, 0.1)$$。$$H(p) = \ln 2 = 0.693$$；$$H(p, q) = -0.5\ln 0.9 - 0.5 \ln 0.1 = 0.053 + 1.151 = 1.204$$；$$D_{\mathrm{KL}}(p \Vert q) = 1.204 - 0.693 = 0.511$$。反过来 $$D_{\mathrm{KL}}(q \Vert p) = 0.9 \ln(0.9/0.5) + 0.1 \ln(0.1/0.5) = 0.529 - 0.161 = 0.368$$——两个方向的值不同。

### 2. 训练时：$$p$$ 是 one-hot

训练语言模型时，"真实分布" $$p$$ 在每个位置是 one-hot：真实的下一个 token 概率 1，其余 0。此时 $$H(p) = 0$$（完全确定），交叉熵只剩一项 $$-\log q(x_t)$$——就是上一篇推出的负对数似然。**"交叉熵 loss"与"最大似然"是同一个东西**，上一篇说过，这里看到了原因。

### 3. 蒸馏时：$$p$$ 是软分布

**知识蒸馏**让小模型（student）学大模型（teacher）：目标不是 one-hot，而是 teacher 在每个位置输出的完整分布 $$p_{\text{teacher}}$$。此时 $$H(p) \ne 0$$，但它不依赖 student 的参数——所以最小化交叉熵 $$H(p_{\text{teacher}}, q_{\text{student}})$$ 与最小化 KL $$D_{\mathrm{KL}}(p_{\text{teacher}} \Vert q_{\text{student}})$$ 只差一个常数，得到同一个 student。软分布比 one-hot 多出的信息是 teacher 分布里"第二名、第三名是谁"——这正是蒸馏能超过直接训 one-hot 的原因。L5 后训练系列的蒸馏篇展开。

### 4. loss 的下限

等式 $$H(p, q) = H(p) + D_{\mathrm{KL}}$$ 还说明一件事：**训练 loss 不可能低于数据本身的熵 $$H(p)$$**。自然语言有不可约的随机性（"今天天气"后面可以接很多词），再大的模型、再多的数据也降不下去。scaling law 里的常数项 $$E$$（Chinchilla 拟合为 1.69 nat，第八篇）就是对这个下限的估计——loss 曲线趋向的那条渐近线。

## 四、KL 的方向

### 1. 两个方向

固定一个目标分布 $$p$$，用 $$q$$ 去逼近它，可以最小化 $$D_{\mathrm{KL}}(p \Vert q)$$ 或 $$D_{\mathrm{KL}}(q \Vert p)$$，两者的行为完全不同：

| | $$D_{\mathrm{KL}}(p \,\Vert\, q) = \sum p \log \frac{p}{q}$$ | $$D_{\mathrm{KL}}(q \,\Vert\, p) = \sum q \log \frac{q}{p}$$ |
|---|---|---|
| 名字 | forward KL | reverse KL |
| 期望在谁上 | 在 $$p$$ 上采样 | 在 $$q$$ 上采样 |
| 什么时候惩罚大 | $$p$$ 有概率而 $$q$$ 接近零的地方（$$\log(p/q) \to \infty$$） | $$q$$ 有概率而 $$p$$ 接近零的地方 |
| 行为 | $$p$$ 有概率的地方 $$q$$ 都必须覆盖——**mode-covering**，$$q$$ 变宽 | $$q$$ 只能待在 $$p$$ 有概率的地方——**mode-seeking**，$$q$$ 收窄到 $$p$$ 的某个峰 |
| 出现在 | MLE / 交叉熵训练（$$p$$ 是数据）；标准蒸馏 | RLHF 的 KL 惩罚；on-policy 蒸馏 |

一个直觉图——注意它的前提：$$p$$ 是双峰的，而 $$q$$ **被限制**只能是单峰的（比如一个高斯）。两种 KL 的"覆盖 / 收窄"之别只在 $$q$$ 的表达能力不够、够不到 $$p$$ 时才出现；$$q$$ 能任意取时，两种 KL 的最优解都是 $$q = p$$，没有差别。

```text
forward KL：q 必须盖住两个峰            reverse KL：q 挑一个峰待着
   p:   ▲       ▲                        p:   ▲       ▲
   q:  ▁▁▁▁▂▃▅▆▅▃▂▁▁▁▁  （宽、平）         q:  ▁▂▅█▅▂▁▁▁▁▁▁▁▁▁  （窄、尖）
```

### 2. RLHF 用的是 reverse

先说这个目标从哪来。预训练完的模型只会"接着往下写"，再经过 **SFT**（supervised fine-tuning，用人写的问答对做上一篇那种交叉熵训练）它学会了按问答的格式回话，但回答好不好——有没有帮助、是否胡编、是否安全——SFT 数据覆盖不到的地方它不知道。**RLHF**（reinforcement learning from human feedback）的做法是：先训一个**奖励模型** $$r(x, y)$$（给"问题 $$x$$ 的回答 $$y$$"打一个分数，从人的偏好比较里学出来，第五章讲怎么学），再调整模型让它的回答拿到更高的分。但只追分数会出事：奖励模型是从有限数据学出来的，模型很快会找到它的漏洞——写得更长、更谄媚、或者干脆说一些奖励模型没见过而误判为好的胡话（**reward hacking**）。所以要加一根"拉绳"：不许离出发点（**参考模型** $$\pi_{\text{ref}}$$，通常就是 SFT 之后的那个模型）太远。用什么量"离得多远"？就是 KL。于是 RLHF 的目标写成"最大化奖励，同时不偏离参考模型"：

$$
\max_\pi\; \mathbb{E}_{y \sim \pi(\cdot \mid x)}\big[r(x, y)\big] - \beta\, D_{\mathrm{KL}}\big(\pi(\cdot \mid x) \,\Vert\, \pi_{\text{ref}}(\cdot \mid x)\big)
$$

$$\pi$$ 是正在训的策略（第四篇说过，就是条件分布 $$p(y \mid x)$$），$$\pi_{\text{ref}}$$ 是参考模型，$$\beta$$ 是拉力系数。KL 的第一个参数是 $$\pi$$——期望在 $$\pi$$ 上取，是 **reverse** 方向。这个方向的含义：策略可以**放弃**参考模型的一部分模式（那些地方 $$\pi$$ 小、$$\pi_{\text{ref}}$$ 大，惩罚不大），但**不能去**参考模型认为不可能的地方（$$\pi$$ 大、$$\pi_{\text{ref}}$$ 接近零，惩罚巨大）。

于是策略倾向于在参考模型的高概率区域里挑奖励高的那部分——这是"对齐常常降低多样性"的机制之一。但要说准：它**不是** reverse KL 的定义行为，多样性降不降取决于奖励长什么样。一个两值的反例：$$\pi_{\text{ref}} = (0.9, 0.1)$$，奖励 $$r = (0, \ln 9)$$，$$\beta = 1$$，下一章的闭式解给 $$\pi^* = (0.5, 0.5)$$——熵从 0.325 **升到** 0.693，策略比参考更多样。真实 RLHF 里多样性下降，是因为奖励模型偏好某一类回答（长、稳、有礼），是奖励的形状加上 mode-seeking 的倾向，两者缺一不可。$$\beta$$ 越大越保守（更贴近参考模型），越小越激进（更追奖励、更容易 reward hacking）。

为什么用 reverse 而不是 forward？工程原因：reverse KL 的期望在 $$\pi$$ 上，样本从正在训的策略里抽就行；forward KL 要从 $$\pi_{\text{ref}}$$ 里抽样并算 $$\pi$$ 在那些样本上的概率，多一份推理。两者的行为差别（收窄 vs 覆盖）则各有取舍。

## 五、Bradley-Terry：偏好如何变成概率

### 1. 模型

奖励模型要从"标注员觉得回答 $$y_w$$ 比 $$y_l$$ 好"（$$w$$ = win，$$l$$ = lose）学出一个标量分 $$r(x, y)$$。怎么把"比较"变成"概率"？**Bradley-Terry 模型**假设：

$$
P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}
$$

$$\sigma$$ 是 **sigmoid** 函数：把任意实数压到 $$(0, 1)$$，$$\sigma(0) = 0.5$$，$$\sigma(2) = 0.88$$，$$\sigma(-2) = 0.12$$，$$\sigma(z) + \sigma(-z) = 1$$。含义：两个回答的**分差**过一个 sigmoid 就是"前者胜"的概率——分差 0 时五五开，分差 2 时 88%。这与国际象棋的 Elo 评分是同一个模型。

### 2. 奖励模型的 loss

套用上一篇的模板——把观察到的偏好的概率放进 $$-\log$$：

$$
\mathcal{L}_{\text{RM}} = -\log \sigma\big(r(x, y_w) - r(x, y_l)\big)
$$

它就是**逻辑回归**的 loss（L2 系列会从分类的角度再讲一遍逻辑回归），只是"特征"换成了两个回答的分差。L2 说"逻辑回归是奖励模型的数学骨架"，指的就是这一行。

## 六、从 KL 约束的最优策略到 DPO

### 1. 闭式解

第四章那个目标——最大化奖励减 $$\beta$$ 倍 KL——有闭式解。对每个 $$x$$，最优策略是

$$
\pi^*(y \mid x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y \mid x)\, \exp\!\left(\frac{r(x, y)}{\beta}\right)
$$

其中 $$Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp(r(x, y)/\beta)$$ 是让概率和为 1 的归一化常数。读它：**最优策略 = 参考模型 × 按奖励指数加权**。奖励高的回答被放大，$$\beta$$ 小时放大得猛，$$\beta$$ 大时几乎不动。

推导只用拉格朗日乘子（第七篇第七章会讲这个工具）与 $$\log$$ 的性质，几行可以完成，建议自己推一次：把目标写成 $$\sum_y \pi(y)[r(y) - \beta\log\pi(y) + \beta\log\pi_{\text{ref}}(y)]$$，加约束 $$\sum_y \pi(y) = 1$$，对 $$\pi(y)$$ 求导令为零。

### 2. 反解奖励

把闭式解两边取 $$\log$$，解出 $$r$$：

$$
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
$$

这句话说：**奖励可以用"最优策略相对参考模型的对数比"表示**，再加一个只依赖 $$x$$ 的常数。

### 3. 代入 Bradley-Terry

把这个 $$r$$ 代进第五章的 $$P(y_w \succ y_l) = \sigma(r_w - r_l)$$。两个回答共享同一个 $$x$$，所以 $$\beta\log Z(x)$$ 在分差里**抵消**：

$$
r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}
$$

用正在训的策略 $$\pi_\theta$$ 代替 $$\pi^*$$，套上一篇的模板放进 $$-\log\sigma(\cdot)$$：

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma\!\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right)
$$

这就是 **DPO**（Direct Preference Optimization）：不训奖励模型、不做 RL，直接用偏好对优化策略——把奖励模型的 loss 里的 $$r$$ 换成了策略自己的对数比。

### 4. 读这个 loss

```text
                    ┌── 隐含奖励 r̂(y_w) ──┐   ┌── 隐含奖励 r̂(y_l) ──┐
−log σ(  β · log [π_θ(y_w) / π_ref(y_w)]  −  β · log [π_θ(y_l) / π_ref(y_l)]  )

每一项 log[π_θ / π_ref] 是"策略比参考模型多喜欢这个回答多少"
loss 要求：对好回答的偏好增量 > 对坏回答的偏好增量，差得越多 loss 越小
β 控制"偏好增量"的尺度——β 大时同样的 loss 需要更小的对数比变化，即更贴近参考模型
```

整条推导链——**MLE 模板 → Bradley-Terry → KL 约束的最优策略 → 反解奖励 → 代入消去 $$Z(x)$$ → DPO**——用到的全是本系列列出的概念，没有一步超出最小集。能独立走完它，L5 后训练系列里"DPO 一族"（IPO、KTO、SimPO 等）就只是在改其中某一步的假设：换掉 Bradley-Terry、去掉参考模型、改 $$\sigma$$ 的形状。

## 七、其他两处

### 1. 总变差与投机解码

**总变差距离**（total variation distance）是比 KL 更朴素的分布距离：

$$
\mathrm{TV}(p, q) = \frac{1}{2}\sum_x \lvert p(x) - q(x) \rvert
$$

它等于"两个分布对同一个事件给出的概率之差的最大值"，取值 $$[0, 1]$$，对称、有界——KL 不对称、可以无穷大。

**投机解码**用小模型的分布 $$q$$ 起草几个 token、大模型的分布 $$p$$ 一次性验证。一个草稿 token 被接受的概率是

$$
\alpha = \sum_x \min\big(p(x), q(x)\big) = 1 - \mathrm{TV}(p, q)
$$

（两式相等：$$\sum \min(p, q) = \sum p - \sum (p - q)_+ = 1 - \frac{1}{2}\sum \lvert p - q \rvert$$。）两个分布越近接受率越高；而拒绝采样保证最终输出**严格服从** $$p$$——小模型只是加速，不改变大模型的分布。L6 高效推理系列第二篇从这一行推出加速比。

### 2. 互信息与对比学习

**互信息** $$I(X; Y) = D_{\mathrm{KL}}\big(p(x, y) \,\Vert\, p(x)\,p(y)\big)$$ 度量两个变量的相关程度：联合分布离"独立时的分布"有多远，独立时为 0。CLIP 的对比学习 loss（InfoNCE）给出互信息的一个下界 $$I \ge \log N - \mathcal{L}_{\text{InfoNCE}}$$——**最小化** loss 就是抬高这个下界，让配对的图文比不配对的更"相关"。L7 多模态系列讲；这里知道定义即可。

## 八、本文小结

- **熵** $$H(p) = -\sum p\log p$$ 是不确定程度，均匀分布最大（$$\log V$$）、确定性分布为 0；**困惑度** $$= e^{\text{loss}}$$，loss 1.8 ↔ PPL 6.05 ↔ 2.6 bit/token。
- **交叉熵 = 熵 + KL**：$$H(p, q) = H(p) + D_{\mathrm{KL}}(p \Vert q)$$；KL 非负、不对称。训练时 $$p$$ 是 one-hot 所以交叉熵 $$= -\log q(x_t)$$（与 MLE 同一件事）；蒸馏时 $$p$$ 是软分布，最小化交叉熵与最小化 KL 等价；loss 降不到数据的熵以下（Chinchilla 的 $$E = 1.69$$）。
- **KL 的方向**：在 $$q$$ 表达能力不够时，forward（期望在 $$p$$ 上）mode-covering、$$q$$ 变宽；reverse（期望在 $$q$$ 上）mode-seeking、$$q$$ 收窄。RLHF 的 $$\beta D_{\mathrm{KL}}(\pi \Vert \pi_{\text{ref}})$$ 是 reverse，但"对齐降低多样性"要靠奖励的形状一起解释（反例：ref (0.9, 0.1)、$$r = (0, \ln 9)$$ 时最优策略是 (0.5, 0.5)，熵升高）；$$\beta$$ 越大越保守。
- **Bradley-Terry**：$$P(y_w \succ y_l) = \sigma(r_w - r_l)$$，分差过 sigmoid；奖励模型的 loss $$-\log\sigma(r_w - r_l)$$ 是逻辑回归。
- **DPO 推导链**：KL 约束目标的闭式解 $$\pi^* \propto \pi_{\text{ref}}\, e^{r/\beta}$$ → 反解 $$r = \beta\log(\pi^*/\pi_{\text{ref}}) + \beta\log Z$$ → 代入 Bradley-Terry，$$Z(x)$$ 抵消 → $$-\log\sigma(\beta\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta\log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)})$$。没有一步超出最小集。
- **总变差** $$\frac{1}{2}\sum\lvert p - q \rvert$$ 对称有界；投机解码的接受率 $$= 1 - \mathrm{TV}$$。**互信息**是联合分布与独立分布的 KL，对比学习通过最小化 InfoNCE 抬高它的下界 $$\log N - \mathcal{L}$$。

## 九、自测

1. $$p = (0.8, 0.2)$$ 的熵是多少 nat？换成 bit？

   <details markdown="1"><summary>答案</summary>

   $$-(0.8\ln 0.8 + 0.2\ln 0.2) = 0.179 + 0.322 = 0.500$$ nat $$= 0.72$$ bit。

   </details>

2. 训练 loss 从 2.3 降到 2.0，PPL 从多少降到多少？

   <details markdown="1"><summary>答案</summary>

   $$e^{2.3} = 9.97 \to e^{2.0} = 7.39$$。

   </details>

3. $$p = (1, 0)$$（one-hot），$$q = (0.7, 0.3)$$：$$H(p)$$、$$H(p, q)$$、$$D_{\mathrm{KL}}(p \Vert q)$$ 各是多少？$$D_{\mathrm{KL}}(q \Vert p)$$ 呢？

   <details markdown="1"><summary>答案</summary>

   $$0$$、$$-\ln 0.7 = 0.357$$、$$0.357$$；$$D(q \Vert p) = 0.7\ln(0.7/1) + 0.3\ln(0.3/0) = \infty$$——$$q$$ 在 $$p$$ 为零的地方有概率，reverse 方向惩罚无穷。

   </details>

4. RLHF 里把 $$\beta$$ 从 0.1 改到 0.01，策略会更贴近还是更远离参考模型？多样性会怎么变？

   <details markdown="1"><summary>答案</summary>

   更远离（KL 拉力变弱）；策略更追奖励，reward hacking 风险更高；多样性通常下降更多（奖励模型偏好某类回答时），但不是定律——奖励若偏向参考模型的低概率区，多样性反而上升。

   </details>

5. DPO 的 loss 里为什么会出现 $$\pi_{\text{ref}}$$？如果把它去掉（令 $$\pi_{\text{ref}}$$ 为均匀分布），loss 变成什么？

   <details markdown="1"><summary>答案</summary>

   它来自 KL 约束的闭式解 $$\pi^* \propto \pi_{\text{ref}} e^{r/\beta}$$；去掉后 $$\log(\pi_\theta / \pi_{\text{ref}})$$ 变成 $$\log \pi_\theta$$ 加常数，loss 变成 $$-\log\sigma(\beta[\log\pi_\theta(y_w) - \log\pi_\theta(y_l)])$$——只比较策略自己给两个回答的对数概率，没有了"不偏离"的约束。

   </details>

下一篇进入微积分与优化：有了目标怎么求导（链式法则、softmax 的梯度 $$p - y$$）、目标是期望时怎么求导（策略梯度）、以及用梯度更新参数的最简单方法。

[^q0]: 熵 $$H(p)$$ 是分布自身的不确定度；交叉熵 $$H(p, q)$$ 是用 $$q$$ 编码 $$p$$ 的平均代价；KL 是两者之差 $$H(p, q) - H(p)$$，非负、不对称。训练最小化交叉熵，因为熵是数据的常数——loss 降不到它以下。详见[第二](#二熵与困惑度)至[四章](#四kl-的方向)。
[^q1]: 能，四步：KL 约束下的最优策略有闭式解 $$\pi^* \propto \pi_{\text{ref}}\, e^{r/\beta}$$（拉格朗日乘子）；反解出 $$r = \beta\log(\pi^*/\pi_{\text{ref}}) + \beta\log Z$$；代入 Bradley-Terry 的 $$\sigma(r_w - r_l)$$；同一 prompt 的 $$\log Z$$ 抵消，得到只含策略与参考的 loss。没有一步超出本系列的最小集。详见[第五章](#五bradley-terry偏好如何变成概率)、[第六章](#六从-kl-约束的最优策略到-dpo)。
