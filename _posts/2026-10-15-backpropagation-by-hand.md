---
layout: post
series: deep-learning-foundations
title: "深度学习基础（01）：反向传播——手推一个两层网络"
subtitle: "Backpropagation by Hand: Shapes, the 2x Rule and Why Activations Must Be Saved"
tags: [AI, Deep Learning, LLM]
catalog: true
---

`loss.backward()` 是训练代码里最短的一行，也是被理解得最少的一行。它做的事在 1986 年就已经写清楚了：沿着计算图反向应用链式法则。但只有自己推过一遍、写过一遍、用有限差分验证过一遍，才会真的知道三件后面每一篇都要用的事——**梯度的形状与被求导的量相同**、**反向的计算量是前向的两倍**、**前向的中间结果必须保留到反向**。第一件决定了怎么读任何一个梯度公式，第二件是训练 FLOPs 等于 $$6ND$$ 的来源，第三件是激活显存与激活重算的全部原因。

本篇用一个两层 MLP（Linear → ReLU → Linear → softmax → 交叉熵）把这三件事推到底。选它是因为它足够小——每一步的形状能写在一行里——又足够完整：Transformer 里除了 attention 之外的每个部件（Linear、激活函数、norm、loss）的反向都是同一套规则。全篇的核心问题是：

> **不用框架，能不能手推并手写一个两层网络的反向传播，用有限差分验证到 $$10^{-6}$$ 以内？能不能由此说出为什么训练 FLOPs 是 $$6ND$$、为什么激活要存？**


## 一、总览：从链式法则到三条结论

### 1. 本文的对象

一个把 784 维输入分成 10 类的两层网络：

```text
X [m, 784]  →  Linear(784, 256)  →  H [m, 256]  →  ReLU  →  A [m, 256]  →  Linear(256, 10)  →  Z [m, 10]  →  softmax + CE  →  L
```

$$m$$ 是 batch 大小。参数是 $$W_1 \in \mathbb{R}^{784 \times 256}, b_1 \in \mathbb{R}^{256}, W_2 \in \mathbb{R}^{256 \times 10}, b_2 \in \mathbb{R}^{10}$$，共 203,530 个。全文要做的是：给出 $$L$$ 对这四个参数的梯度、说出每一步的计算量与需要保存的量、用数值方法验证公式没错、在 MNIST 上训到 97%。

### 2. 本文的章节安排

```text
第二章   链式法则与计算图          标量对向量与矩阵的导数、Jacobian、为什么反向传播只算 VJP 从不构造 Jacobian
第三章   矩阵求导的形状规则        Y = XW + b 的三条梯度公式与它们的推导；ReLU 的反向
第四章   两层网络逐层推导          从 loss 到 W_1 的每一步，每一步的形状
第五章   反向为什么是前向的两倍    每个 Linear 反向做两个 GEMM；6ND 的推导；激活重算的 8ND
第六章   激活为什么要存            dW 需要 X；激活显存与 batch、序列长度成正比；checkpointing 换掉它
第七章   梯度检查                  有限差分、相对误差、为什么要 float64；两个训练前的 sanity check
第八章   Autograd 做了什么         录带、每个算子的 backward、saved tensors、.grad 累加
第九章   实验                      120 行 NumPy：梯度检查 1e-7、FLOPs 比 2.00、与 PyTorch 对齐、MNIST 97%
第十章   本文小结
```


## 二、链式法则与计算图

### 1. 三种导数的形状

深度学习里只出现三种求导，各自的结果形状要先记住：

| 被求导的函数 | 对什么求导 | 结果 | 形状 |
|---|---|---|---|
| 标量 $$L$$ | 向量 $$x \in \mathbb{R}^n$$ | 梯度 $$\nabla_x L$$ | $$\mathbb{R}^n$$，与 $$x$$ 相同 |
| 标量 $$L$$ | 矩阵 $$W \in \mathbb{R}^{k \times n}$$ | 梯度 $$\partial L / \partial W$$ | $$\mathbb{R}^{k \times n}$$，与 $$W$$ 相同 |
| 向量 $$y \in \mathbb{R}^n$$ | 向量 $$x \in \mathbb{R}^k$$ | Jacobian $$J = \partial y / \partial x$$ | $$\mathbb{R}^{n \times k}$$，$$J_{ij} = \partial y_i / \partial x_j$$ |

前两种是训练最终要的东西——loss 对每个参数的梯度，形状与参数一样，所以可以直接 `W -= lr * dW`。第三种是中间量：每一层把输入变成输出，层的局部导数是一个 Jacobian。

### 2. 链式法则的向量形式

复合函数 $$L = \ell(y), \; y = f(x)$$，$$L$$ 对 $$x$$ 的梯度是

$$
\nabla_x L = J^T \, \nabla_y L, \qquad J = \frac{\partial y}{\partial x} \in \mathbb{R}^{n \times k}
$$

也就是"上游传来的梯度，左乘本层 Jacobian 的转置"。把网络看成一串函数 $$x_0 \to x_1 \to \dots \to x_l \to L$$，从 $$L$$ 出发反向逐层套用这一行，就是反向传播的全部：

$$
\nabla_{x_{i}} L = J_i^T \, \nabla_{x_{i+1}} L, \qquad J_i = \frac{\partial x_{i+1}}{\partial x_i}
$$

每一层只需要知道两件事：自己的 Jacobian 是什么、上游传来的梯度是什么。它不需要知道网络其他部分长什么样——这就是为什么框架可以把任意算子拼起来自动求导。

### 3. 为什么从不构造 Jacobian

上面的公式里 Jacobian 只以"乘一个向量"的形式出现：$$J^T v$$，叫 **vector-Jacobian product（VJP）**。反向传播每一步算的是 VJP，从不把 $$J$$ 本身写出来。原因是尺寸：本文第一层 Linear 在 batch 128 下，输入 $$X \in \mathbb{R}^{128 \times 784}$$、输出 $$H \in \mathbb{R}^{128 \times 256}$$，把它们拉直后 Jacobian 是 $$(128 \times 256) \times (128 \times 784) = 32768 \times 100352$$，约 $$3.3 \times 10^9$$ 个元素，fp32 下 **13 GB**——一个 20 万参数的小网络的一层。而 VJP $$J^T v$$ 的结果只有 $$X$$ 那么大（100352 个数，392 KiB），并且对 Linear 这类结构化的层，它可以直接写成一个矩阵乘法，下一章推。

所以"反向传播"这个词精确的含义是：**按计算图的反向拓扑序，对每个算子算一次 VJP**。每个算子的 VJP 有自己的闭式公式，框架里每个 `Function` 的 `backward` 就是那条公式。


## 三、矩阵求导的形状规则

### 1. Linear 层的三条公式

$$Y = XW + b$$，$$X \in \mathbb{R}^{m \times k}$$（$$m$$ 个样本、每个 $$k$$ 维），$$W \in \mathbb{R}^{k \times n}$$，$$b \in \mathbb{R}^n$$ 广播到每一行，$$Y \in \mathbb{R}^{m \times n}$$。记上游梯度 $$G = \partial L / \partial Y \in \mathbb{R}^{m \times n}$$。三条公式：

$$
\frac{\partial L}{\partial W} = X^T G \in \mathbb{R}^{k \times n}, \qquad
\frac{\partial L}{\partial X} = G\, W^T \in \mathbb{R}^{m \times k}, \qquad
\frac{\partial L}{\partial b} = \sum_{i=1}^{m} G_{i,:} \in \mathbb{R}^{n}
$$

**推导。** 按元素写：$$Y_{ij} = \sum_r X_{ir} W_{rj} + b_j$$。$$L$$ 对 $$W_{rj}$$ 的导数，按链式法则对所有依赖 $$W_{rj}$$ 的 $$Y_{ij}$$ 求和：

$$
\frac{\partial L}{\partial W_{rj}} = \sum_i \frac{\partial L}{\partial Y_{ij}} \frac{\partial Y_{ij}}{\partial W_{rj}} = \sum_i G_{ij} X_{ir} = (X^T G)_{rj}
$$

对 $$X_{ir}$$ 同理：$$\partial L / \partial X_{ir} = \sum_j G_{ij} W_{rj} = (G W^T)_{ir}$$。对 $$b_j$$：每一行都加了 $$b_j$$，所以 $$\partial L / \partial b_j = \sum_i G_{ij}$$。

**形状记忆法。** 三条公式不必背，用形状就能重建：$$\partial L / \partial W$$ 必须是 $$k \times n$$，手头有 $$X$$（$$m \times k$$）和 $$G$$（$$m \times n$$），唯一能凑出 $$k \times n$$ 的乘法是 $$X^T G$$；$$\partial L / \partial X$$ 必须是 $$m \times k$$，手头有 $$G$$（$$m \times n$$）和 $$W$$（$$k \times n$$），唯一的凑法是 $$G W^T$$。**梯度的形状与被求导的量相同**这一条约束，加上"只能用手头的量做矩阵乘"，几乎总能唯一确定公式。读论文里任何一个梯度表达式时先做这个形状检查，能抓住大部分笔误。

### 2. 逐元素算子：ReLU

$$A = \text{ReLU}(H) = \max(H, 0)$$，逐元素。它的 Jacobian 是对角阵，对角线上是 $$\mathbb{1}[H > 0]$$；VJP 就是逐元素乘一个 0/1 掩码：

$$
\frac{\partial L}{\partial H} = \frac{\partial L}{\partial A} \odot \mathbb{1}[H > 0]
$$

所有逐元素激活函数（GELU、SiLU、sigmoid、tanh）的反向都是"上游梯度逐元素乘导数"，区别只在导数的表达式。$$H = 0$$ 处 ReLU 不可导，实践中取 0 或 1 都行——概率为零的事件。

### 3. softmax 与交叉熵

$$L = -\frac{1}{m}\sum_i \log p_{i, y_i}, \; p_i = \text{softmax}(z_i)$$。[L0 导读第五章](/math-for-ai-algorithm-engineers.html)推过对单个样本 $$\partial L_i / \partial z_i = p_i - e_{y_i}$$（$$e_{y_i}$$ 是 one-hot）。对 batch 取平均后：

$$
\frac{\partial L}{\partial Z} = \frac{1}{m}(P - Y_{\text{onehot}}) \in \mathbb{R}^{m \times 10}
$$

这里 softmax 与交叉熵**合在一起**求导，中间不经过 $$\partial L / \partial P$$。分开算在数学上等价，但数值上更差（$$1/p$$ 在 $$p \to 0$$ 时溢出），计算上也多一步。框架里 `F.cross_entropy` 是一个融合算子，正是为此。


## 四、两层网络逐层推导

把三章的规则串起来。前向五步、反向五步，每一步标出形状（$$m = 128$$）：

```text
前向（带 * 的量要保存到反向）

  X*         XW₁+b₁    H*        ReLU     A*        AW₂+b₂    Z        softmax·CE   L
  [128,784] ────────▶ [128,256] ──────▶ [128,256] ────────▶ [128,10] ──────────▶ 标量
     │ 保存              │ 保存符号          │ 保存
     ▼                   ▼                 ▼
反向（每一步一个 VJP）

  ∂L/∂X       · W₁ᵀ    ∂L/∂H     ⊙𝟙[H>0]  ∂L/∂A       · W₂ᵀ    ∂L/∂Z = (P−Y)/m
  [128,784] ◀──────── [128,256] ◀─────── [128,256] ◀──────── [128,10]
  （不需要）              │ Xᵀ ·                                  │ Aᵀ ·
                         ▼                                      ▼
                      ∂L/∂W₁ [784,256]                       ∂L/∂W₂ [256,10]
                      ∂L/∂b₁ [256] = Σ行                      ∂L/∂b₂ [10] = Σ行
```

写成公式：

| 步 | 计算 | 形状 | 用到前向保存的 |
|---|---|---|---|
| 反向 1 | $$G_Z = (P - Y) / m$$ | $$[128, 10]$$ | $$P$$（由 $$Z$$ 算出） |
| 反向 2 | $$\partial L / \partial W_2 = A^T G_Z$$；$$\partial L / \partial b_2 = \sum_i G_Z$$ | $$[256, 10]$$；$$[10]$$ | $$A$$ |
| 反向 3 | $$G_A = G_Z W_2^T$$ | $$[128, 256]$$ | — |
| 反向 4 | $$G_H = G_A \odot \mathbb{1}[H > 0]$$ | $$[128, 256]$$ | $$H$$ 的符号（掩码） |
| 反向 5 | $$\partial L / \partial W_1 = X^T G_H$$；$$\partial L / \partial b_1 = \sum_i G_H$$ | $$[784, 256]$$；$$[256]$$ | $$X$$ |

$$\partial L / \partial X = G_H W_1^T$$ 在数学上存在，但 $$X$$ 是数据不是参数，不需要它的梯度，框架会跳过这一步。任何深度的 MLP 都是把反向 3–5 重复若干次；Transformer 的 FFN 子层就是一个两层 MLP，attention 子层多几个矩阵乘和一个 softmax，规则不变。


## 五、反向为什么是前向的两倍

### 1. 每个 Linear 反向做两个 GEMM

前向一个 Linear 做一次矩阵乘 $$XW$$：$$[m, k] \times [k, n]$$，$$2mkn$$ FLOPs。反向做两次：$$X^T G$$（$$[k, m] \times [m, n]$$，$$2mkn$$）和 $$G W^T$$（$$[m, n] \times [n, k]$$，$$2mkn$$）。**反向 = 2 × 前向**，对每一个 Linear 层都严格成立。逐元素算子（ReLU、norm、softmax）的计算量与 GEMM 相比可以忽略，所以整个网络也近似成立。

代入本文的网络，batch 128：

| | 第一层 $$784 \to 256$$ | 第二层 $$256 \to 10$$ | 合计 |
|---|---|---|---|
| 前向 | $$2 \times 128 \times 784 \times 256 = 51.4$$M | $$0.66$$M | **52.0 MFLOPs** |
| 反向 | $$102.8$$M | $$1.31$$M | **104.1 MFLOPs** |
| 比值 | | | **2.00** |

第九章的实验里用计数器验证了这两个数字。实践中第一层的 $$\partial L / \partial X$$ 可以省，比值略小于 2；层数多时这一项可忽略。

### 2. 训练 FLOPs 等于 6ND

对一个参数量为 $$N$$ 的网络，前向每个 token 约 $$2N$$ FLOPs（每个参数参与一次乘加——L0 导读第二章），反向 $$4N$$，一步训练合计 $$6N$$ FLOPs / token。训练 $$D$$ 个 token 就是 $$6ND$$——scaling law 论文与 [04 系列第二篇](/transformer-flops-bytes-and-roofline.html)用的这个数字，来源就是本章的"反向做两个 GEMM"。Llama-3-8B 训 15T token：$$6 \times 8 \times 10^9 \times 15 \times 10^{12} = 7.2 \times 10^{23}$$ FLOPs。

两点补充。第一，$$2N$$ 忽略了 attention 里 $$QK^T$$ 与 $$PV$$ 这两个与参数无关、与序列长度成正比的项，短序列下可忽略，长序列下不能（04 系列第二篇算了）。第二，如果用了激活重算（下一章），反向前要再做一次前向，总量变成 $$8N$$ / token——训练报告里"MFU 按 $$6ND$$ 算、HFU 按 $$8ND$$ 算"的区别就在这里。


## 六、激活为什么要存

### 1. 参数梯度需要本层的输入

第四章的表最后一列是关键：反向算 $$\partial L / \partial W_1 = X^T G_H$$ 需要 $$X$$，算 $$\partial L / \partial W_2 = A^T G_Z$$ 需要 $$A$$，算 ReLU 的反向需要 $$H$$ 的符号。这些都是**前向的中间结果**，反向时要用，所以前向算完不能丢，必须一直留到反向走到那一层。这就是"激活显存"。

它的大小与什么成正比？每一层保存的是该层的输入，形状 $$[m, \text{宽度}]$$——与 **batch 大小** $$m$$ 成正比，与**层数**成正比，与**宽度**成正比，**与参数量无关**。本文的网络在 batch 128 下：

| 保存的量 | 形状 | 字节 |
|---|---|---|
| $$X$$ | $$[128, 784]$$ fp32 | 392 KiB |
| $$H$$ 的掩码 | $$[128, 256]$$ bool | 32 KiB |
| $$A$$ | $$[128, 256]$$ fp32 | 128 KiB |
| 合计 | | **552 KiB**（权重 795 KiB） |

batch 换成 4096，激活变成 17.3 MiB，权重不变。序列模型里 $$m$$ 是 batch × 序列长度，所以长上下文训练的激活显存会远超权重——[L1 导读](/tooling-for-ai-algorithm-engineers.html)给过 Llama-3-8B 在 4096 长度下仅残差流一份就是 1 GiB / 序列的锚点，精确公式在 Infra 地图 07 系列第一篇。

### 2. 激活重算：用计算换存储

既然激活是前向算出来的，可以不存、反向时重算。**gradient checkpointing** 的做法是只保存每个块（比如 Transformer 的一层）的输入，反向走到这一块时先用保存的输入重新前向一次得到块内所有中间量，再做反向。代价是多一次前向，即上一章说的 $$6N \to 8N$$，约 33% 的额外计算；收益是激活显存从"所有中间量"降到"每块一个输入"。这是训练长序列或大 batch 时的标准开关，`model.gradient_checkpointing_enable()` 一行。

FlashAttention 做的是同一件事的算子级版本：不保存 $$[\text{seq}, \text{seq}]$$ 的 attention 矩阵，反向时分块重算——04 系列第三篇讲它的 IO 复杂度。


## 七、梯度检查

### 1. 有限差分

手推的公式对不对，用数值导数验证。对参数 $$\theta$$ 的某一个元素 $$\theta_j$$，中心差分

$$
\frac{\partial L}{\partial \theta_j} \approx \frac{L(\theta + \epsilon e_j) - L(\theta - \epsilon e_j)}{2\epsilon}
$$

的截断误差是 $$O(\epsilon^2)$$（单侧差分是 $$O(\epsilon)$$，所以用中心）。把它与解析梯度比，用**相对误差** $$|g_{\text{num}} - g_{\text{ana}}| / (|g_{\text{num}}| + |g_{\text{ana}}|)$$——绝对误差在梯度本身很小的元素上没有意义。

### 2. 为什么必须 float64

$$\epsilon$$ 有两头约束：太大截断误差大，太小舍入误差大——$$L(\theta + \epsilon) - L(\theta - \epsilon)$$ 是两个相近数相减，误差约为 $$L$$ 的机器精度除以 $$2\epsilon$$。fp32 机器精度约 $$10^{-7}$$，取 $$\epsilon = 10^{-6}$$ 时舍入误差约 $$10^{-7} / 10^{-6} = 0.1$$，检查毫无意义；fp64 机器精度约 $$10^{-16}$$，同样的 $$\epsilon$$ 舍入误差 $$10^{-10}$$，截断误差 $$10^{-12}$$，可以把相对误差压到 $$10^{-7}$$ 以下。所以梯度检查一定在 float64 上做、用小网络、随机抽几十个元素（全部检查要做 $$2 \times 203530$$ 次前向）。第九章的结果：最大相对误差 $$1.1 \times 10^{-7}$$。

判断标准：相对误差 $$< 10^{-6}$$ 通过；$$10^{-4}$$ 量级要怀疑（常见于 ReLU 在 0 附近被 $$\epsilon$$ 翻转符号，属于正常）；$$> 10^{-2}$$ 一定有错。

### 3. 两个训练前的 sanity check

梯度对了，训练前还有两个几乎免费的检查。**初始 loss 应接近 $$\ln C$$**：10 类是 $$\ln 10 = 2.30$$，本文实验初始 2.46（Kaiming 初始化让 logits 方差略大于 1，比均匀分布稍差，正常）；远大于它说明初始化太大，远小于它说明数据泄漏或 loss 算错。LLM 上对应 $$\ln V \approx 11.8$$（L0 导读）。**能过拟合一个小 batch**：拿 16 个样本反复训，loss 应能降到接近 0；降不下去说明梯度没传到某处，或学习率不对。这两个检查在框架里同样适用。


## 八、Autograd 做了什么

本文手写的东西，框架用四个机制自动化了：

| 机制 | 本文对应 | PyTorch 里 |
|---|---|---|
| 录带 | 前向时按顺序记住每个算子 | `requires_grad=True` 的张量参与的每个算子被记进动态图，输出张量的 `grad_fn` 指向它 |
| 每个算子的 backward | 第三章的三类 VJP 公式 | 每个 `Function` 有 `forward` 与 `backward`，后者接收上游梯度、返回对每个输入的 VJP |
| saved tensors | 第六章要保存的 $$X$$、掩码、$$A$$ | `ctx.save_for_backward(...)`；`backward()` 后释放，所以同一个图不能反向两次（除非 `retain_graph=True`） |
| 梯度累加 | `dW[...] = X.T @ dY` | 叶子张量的 `.grad` 是**累加**而不是覆盖，所以每步要 `zero_grad()`；梯度累积正是利用这一点 |

`torch.no_grad()` 关掉录带（推理与评测时省激活显存）；`.detach()` 把一个张量从图上摘下来（RL 里对 old logprobs 常用）；`torch.utils.checkpoint` 就是第六章的激活重算。第九章的实验把手写梯度与 PyTorch autograd 的结果对了一遍，差在 $$10^{-8}$$ 量级——两边算的是同一组公式，差异只是浮点求和顺序。

Autograd 引擎的实现——图怎么存、多线程怎么调度、hook 在哪里——属于 Infra 地图 [03 系列第三篇](/pytorch-autograd-and-dynamic-computation-graph.html)。算法工程师到这一层就够：知道它记了什么、存了什么、什么时候释放。


## 九、实验

### 1. 代码

约 120 行 NumPy，是整个系列的基座，后面每篇往上加。核心的三个类：

```python
class Linear:
    def forward(self, X):
        self.X = X                                  # 保存激活
        return X @ self.W + self.b
    def backward(self, dY):
        self.dW[...] = self.X.T @ dY                # X^T G      2mkn
        self.db[...] = dY.sum(0)
        return dY @ self.W.T                        # G W^T      2mkn

class ReLU:
    def forward(self, X): self.mask = X > 0; return X * self.mask
    def backward(self, dY): return dY * self.mask

def softmax_ce(logits, y):
    z = logits - logits.max(1, keepdims=True)       # 数值稳定
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    m = logits.shape[0]
    loss = -np.log(p[np.arange(m), y]).mean()
    d = p.copy(); d[np.arange(m), y] -= 1.0         # p - y
    return loss, d / m, p
```

网络是 `MLP([784, 256, 10])`，训练循环就是 `forward → softmax_ce → backward → 对每个参数 P -= lr * dP`。梯度检查在一个 `[20, 16, 5]` 的 float64 小网络上做，每个参数随机抽 30 个元素、$$\epsilon = 10^{-6}$$、中心差分。FLOPs 用一个全局计数器在每次矩阵乘处累加。

### 2. 结果

```text
grad check worst rel err: 1.11e-07
params: 203530
batch 128: fwd 52.0 MFLOPs  bwd 104.1 MFLOPs  ratio 2.00
  2*N*tokens = 52.1 MFLOPs (fwd approx ignoring bias)
  saved activations: 552 KiB  vs weights 795 KiB
```

与 PyTorch autograd 对齐（同一份权重、同一个 batch）：

```text
loss numpy 2.460904 torch 2.460904
dW1 max abs diff 5.59e-09
dW2 max abs diff 7.45e-09
```

MNIST 上 SGD、学习率 0.1、batch 128：

```text
epoch 1   train loss 0.4477  test acc 92.52%
epoch 5   train loss 0.1416  test acc 96.01%
epoch 10  train loss 0.0842  test acc 97.26%
epoch 15  train loss 0.0585  test acc 97.61%
```

每个 epoch 在笔记本 CPU 上约 0.1 秒。四个数字各验证了本文的一个结论：$$10^{-7}$$ 的相对误差说明第三、四章的公式没错；2.00 的比值是第五章；552 KiB 与 795 KiB 的对比是第六章——这个小网络的激活已经与权重同量级，batch 再大 32 倍就是权重的 22 倍；97% 说明这一套确实能训。

### 3. 三个值得自己动手的扩展

- 把 ReLU 换成 GELU 或 SiLU，只改 `backward` 里的导数表达式，梯度检查应仍通过；
- 加一层，看 FLOPs 比值仍是 2、激活线性增长；
- 把梯度检查改在 float32 上做，观察相对误差跳到 $$10^{-2}$$ 量级——第七章的论证。


## 十、本文小结

- 反向传播 = 按计算图反向拓扑序对每个算子算一次 **VJP** $$J^T v$$；从不构造 Jacobian（本文第一层的 Jacobian 有 13 GB）。
- **梯度的形状与被求导的量相同。** Linear 的三条公式 $$\partial L / \partial W = X^T G$$、$$\partial L / \partial X = G W^T$$、$$\partial L / \partial b = \sum_i G_{i,:}$$ 可以从形状唯一重建；逐元素算子的反向是上游梯度乘导数；softmax 与交叉熵合并求导得 $$(P - Y) / m$$。
- **反向 = 2 × 前向**：每个 Linear 反向做两个 GEMM。由此训练 FLOPs $$= 6N$$ / token，激活重算下 $$8N$$；本文实测比值 2.00。
- **激活必须保存到反向**，因为 $$\partial L / \partial W$$ 需要该层的输入。激活显存与 batch × 序列长度 × 层数 × 宽度成正比、与参数量无关；gradient checkpointing 用一次额外前向换掉它。
- 梯度检查用 **float64、中心差分、相对误差**，$$< 10^{-6}$$ 通过；训练前看初始 loss 是否接近 $$\ln C$$、能否过拟合一个小 batch。
- Autograd 做的是录带、每个算子的 backward、saved tensors、`.grad` 累加；手写结果与它差 $$10^{-8}$$。
- 下一篇把这个网络加深到 64 层，看梯度在层间传播时会发生什么。

配套代码：[`deep-learning-foundations/01_backprop.py`](https://github.com/arganzheng/ai-learning-labs/blob/main/deep-learning-foundations/01_backprop.py)——本文的全部数字由它跑出（梯度检查、FLOPs、与 autograd 对齐、15 个 epoch 的训练），`--quick` 一分钟内跑完；第九章的三个扩展可以直接在上面改。


## 下一篇

[训练为什么不稳定：初始化、归一化与残差](/initialization-normalization-and-residual.html)
