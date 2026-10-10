---
layout: post
series: transformer-and-llm
title: "Transformer 与 LLM（07）：位置编码与外推"
subtitle: "Positional Encoding and Extrapolation: RoPE Wavelengths and Context Extension"
tags: [Transformer, LLM, AI, AI-Infra]
catalog: true
updated: 2026-09-14
date: 2026-04-03 14:00:00
---

> **本篇在系列中的位置。** 第二段的第三篇，也是第一个结构专项。第 05 篇只讲到 RoPE 为什么替代位置表，本篇讲位置编码本身：RoPE 的推导与波长、外推（在比训练时更长的序列上推理）为什么失败、各种长度扩展方法改了什么。下一篇的 MLA 要把 RoPE 从低秩压缩里解耦出来，需要这里的结论；上下文拉长之后的成本（KV cache、二次项 attention）与 sliding window 等结构手段在第 09 篇。完整地图见[总纲](/transformer-and-llm-structure-implementation-and-evolution.html)。

[《Transformer 与 LLM（01）：Transformer 长什么样——从一句话到下一个 token》](/transformer-architecture-from-a-sentence-to-the-next-token.html)的换序实验说明 attention 是集合运算——把输入 token 打乱，输出只是跟着换位置——所以位置必须显式喂给模型；GPT-2 的做法是查一张位置表，表的行数就是它的上下文长度 1024，上限写死在参数里，第 1025 个位置没有训练过的向量。Llama 换成了 RoPE，没有表，也就没有写死的行数，但这不等于它能处理任意长度：模型凭什么知道一个 token 在第几个位置？超过训练长度之后它还认不认得？

这两个问题都由位置编码回答。它在参数量表里几乎不占位置（RoPE 一个参数都没有），在算量表里也可以忽略（一次逐元素乘加），却决定了模型在多长的序列上还"认得"位置。

本篇要回答的核心问题是：

> **一个用 8K 上下文训练的 RoPE 模型，为什么不能直接推理 32K？[^q0] 把 base 从 10000 改到 500000 解决了什么，没解决什么？[^q1]**

## 一、总览：位置编码与上下文长度是什么关系

### 1. 先把因果说清楚

位置编码和长上下文常被放在一起讲，但两者不是"一个为另一个而生"的关系：

- **位置编码解决的是顺序问题，不是长度问题。** 不加位置与掩码的 attention 对置换等变（第二章），位置编码是为了让"猫追狗"和"狗追猫"得到不同的表示；就算模型只处理 100 个 token，它也必须有。
- **它约束的是位置能否外推，但不是长上下文能力的全部。** 位置方案决定模型在训练长度之外还认不认得位置：可学习位置表的行数就是硬上限（GPT-2 的 1024）；RoPE 没有表，没有硬上限，但超过训练长度后可能遇到未覆盖的相位与距离分布，效果不能保证（第四章）。Position Interpolation、NTK-aware、YaRN、Llama 3.1 的分段缩放都是在改 RoPE 的频率，让"见过的相位"覆盖更长的序列（第五章）。
- **另一组问题是"用得起吗"——这些成本并不因换位置编码而消失。** 上下文拉长后，KV cache 线性增长、prefill 的 attention 算量二次增长、$$s \times s$$ 的 logits 不能物化；sliding window、全局/局部交错、attention sink、稀疏 attention 改的是 attention "看哪些 token"，与位置怎么编码无关。这组问题是下一篇[《Transformer 与 LLM（08）：长上下文的成本与结构手段》](/long-context-cost-and-structural-remedies.html)的内容。

所以本篇只回答位置这一部分：位置怎么编码、为什么 RoPE 能处理相对位置、它的外推为什么失败、各种扩展方法各改了什么。

### 2. 数字基线

数字基线与前几篇相同：Llama-3-8B（$$d = 4096$$，32 层，32 个 query 头、8 个 KV 头，$$d_{head} = 128$$）、Llama-3-70B（$$d = 8192$$，80 层，64 个 query 头、8 个 KV 头）、DeepSeek-V3（$$d = 7168$$，61 层，128 头，MLA 的 $$d_c = 512$$、$$d_h^R = 64$$）。

### 3. 本文的章节安排

| 章 | 主题 | 内容 |
|---|---|---|
| 二 | 为什么需要位置编码 | 无 mask 的 attention 是置换等变的；绝对（正弦/可学习）与相对位置编码 |
| 三 | RoPE 的推导 | `d_head/2` 个复数、相对性的完整形式、`rotate_half` 实现、与 KV cache 的关系 |
| 四 | 波长：RoPE 的频谱 | 每个维度对的波长、8K 训练时哪些对"没转完一圈"、外推为什么失败、base 10000 → 500000 |
| 五 | 长上下文扩展方法 | Position Interpolation、NTK-aware、YaRN、Llama 3.1 分段缩放、DeepSeek-V3/Qwen 的配置、ALiBi |
| 六 | 实践 | NumPy 实现 RoPE 并验证相对性、波长表与三种缩放 |
| 七 | 本文小结 |  |
| 八 | 自测 | 3 道题 |

Table: 本文的章节安排

## 二、为什么需要位置编码

### 1. attention 是置换等变的（在没有 mask 时）

单个 attention 头的计算是：

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_{head}}}\right) V
$$

其中 $$Q = XW_Q$$，$$K = XW_K$$，$$V = XW_V$$，$$X \in \mathbb{R}^{s \times d}$$ 是 $$s$$ 个 token 的输入。把输入的行做任意置换 $$P$$（$$X' = PX$$），则 $$Q' = PQ$$、$$K' = PK$$、$$V' = PV$$，于是：

$$
\text{softmax}\!\left(\frac{PQK^\top P^\top}{\sqrt{d_{head}}}\right) PV = P\,\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_{head}}}\right) V
$$

输出只是原输出的同样置换——这叫**置换等变**（输出跟着输入一起换位；"不变"是输出完全不动，那是对集合做 pooling 才有的性质）。换句话说，attention 把输入看作一个**集合**而不是序列：第 $$m$$ 个 token 的输出只取决于"其他 token 是什么"，与"它们在哪里"无关。FFN 是逐 token 的，RMSNorm 也是逐 token 的，所以整个 Transformer block 都是置换等变的。

上面的推导有一个前提：**没有 mask**。加上因果掩码之后等变性就不再成立——固定的下三角 mask 在行置换下不再是下三角，第 $$m$$ 个位置能看到多少个 token 本身就泄露了它的序号（用 NumPy 验证：无 mask 时置换输入、输出最大差 $$10^{-7}$$；加 causal mask 后差 0.73）。所以严格说，causal 模型即便不加任何位置编码也能从"能看见几个"里学到一些顺序信息（确实有 NoPE 的实验），只是这种信号很弱、不显式、外推性差。

要让模型稳定地读懂"猫追狗"和"狗追猫"的区别，位置信息还是要显式注入。注入的方式分为两大类。

### 2. 绝对位置编码：正弦与可学习

原始 Transformer（Vaswani 等 2017）在 embedding 上直接加一个与位置有关的向量：

$$
PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \qquad PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)
$$

每一对维度 $$(2i, 2i+1)$$ 是一个频率为 $$10000^{-2i/d}$$ 的正弦对，低维高频、高维低频。这个频率表在后面的 RoPE 里会原样出现——区别只在于它是**加**到 embedding 上，还是**乘**到 q、k 上。

加法的问题在于：$$q_m^\top k_n = (x_m + p_m)^\top W_Q^\top W_K (x_n + p_n)$$ 展开后有四项，其中 $$p_m^\top W_Q^\top W_K p_n$$ 依赖两个绝对位置 $$m$$ 和 $$n$$，而不只是它们的差。模型可以学到"相对距离"这个概念，但没有结构上的保证；训练时没见过的绝对位置 $$m > L$$，对应的 $$p_m$$ 虽然可以计算出来，模型却不知道怎么用它。

GPT-2、BERT 一类用**可学习**的位置 embedding：一张 $$L_{max} \times d$$ 的表，每个位置一行。GPT-2 的 $$L_{max} = 1024$$，这张表就是 $$1024 \times 768$$。它的上限是硬的：位置 1025 没有对应的行，模型物理上无法处理更长的输入。

### 3. 相对位置编码

相对位置编码（Shaw 等 2018；T5 的 relative bias；Transformer-XL）把"位置"从 embedding 里拿出来，直接在 attention 分数上加一个只依赖 $$m - n$$ 的项：

$$
\text{score}(m, n) = \frac{q_m^\top k_n}{\sqrt{d_{head}}} + b_{m - n}
$$

T5 把 $$m - n$$ 分桶（距离越远桶越粗），每个桶每个 head 学一个标量。这类方法天然只依赖相对距离，但有两个工程上的代价：一是 bias 项 $$b_{m-n}$$ 是一个 $$s \times s$$ 的矩阵，要么显式物化，要么在 kernel 里逐元素查表，与 FlashAttention 一类的 fused kernel 配合并不顺手；二是它只能给 attention 分数加偏置，不能让 q、k 的内容与位置发生交互（"第 3 个位置上的名词"这样的联合特征无法表达）。

RoPE 的目标是同时拿到两边的好处：**以绝对位置的形式实现**（每个 token 独立处理自己的 q、k，不需要 $$s \times s$$ 的额外矩阵），**得到相对位置的性质**（$$q_m^\top k_n$$ 只依赖 $$m - n$$）。

## 三、RoPE 的推导

### 1. 把 d_head 维向量看成 d_head/2 个复数

RoPE（Rotary Position Embedding，Su 等 2021）的核心想法是：找一个函数 $$f(x, m)$$，把向量 $$x$$ 和位置 $$m$$ 映射成一个新向量，使得对任意 $$q$$、$$k$$：

$$
\langle f(q, m), f(k, n) \rangle = g(q, k, m - n)
$$

即内积只依赖相对位置。先看二维的情形。把一个二维向量 $$x = (x_1, x_2)$$ 看成复数 $$x_1 + \mathrm{i} x_2$$，取

$$
f(x, m) = x \cdot e^{\mathrm{i} m\theta}
$$

也就是把这个复数在复平面上逆时针旋转 $$m\theta$$ 角。两个复数的实内积等于 $$\text{Re}[q \bar{k}]$$（$$\bar{k}$$ 是共轭），所以：

$$
\langle f(q, m), f(k, n) \rangle = \text{Re}\!\left[ q e^{\mathrm{i} m\theta} \cdot \overline{k e^{\mathrm{i} n\theta}} \right] = \text{Re}\!\left[ q \bar{k}\, e^{\mathrm{i}(m - n)\theta} \right]
$$

$$m$$ 和 $$n$$ 只以 $$m - n$$ 的形式出现。这就是相对性——旋转是等距变换，两个向量各转 $$m\theta$$ 和 $$n\theta$$ 之后的夹角，只取决于 $$(m - n)\theta$$。

推广到 $$d_{head}$$ 维：把向量切成 $$d_{head}/2$$ 对，第 $$i$$ 对（$$i = 0, 1, \ldots, d_{head}/2 - 1$$）当作一个复数，以各自的角频率 $$\theta_i$$ 旋转：

$$
\theta_i = \text{base}^{-2i/d_{head}}
$$

原论文取 $$\text{base} = 10000$$，与正弦位置编码的频率表完全相同。第 0 对 $$\theta_0 = 1$$，最后一对 $$\theta_{63} = 10000^{-126/128} \approx 1.15 \times 10^{-4}$$（$$d_{head} = 128$$）。

### 2. 相对性的完整形式

对整个向量，位置 $$m$$ 的 query 与位置 $$n$$ 的 key 的内积是各对内积之和：

$$
q_m^\top k_n = \text{Re}\!\left[ \sum_{i=0}^{d_{head}/2 - 1} q_i \bar{k}_i\, e^{\mathrm{i}(m - n)\theta_i} \right]
$$

其中 $$q_i$$、$$k_i$$ 是第 $$i$$ 对对应的复数（旋转之前的内容）。这个式子里 $$m$$、$$n$$ 只以差的形式出现，所以 RoPE 给出的 attention 分数严格只依赖相对位置。同时它是对 q、k **逐 token 独立**施加的：算 $$q_m$$ 时只需要知道 $$m$$，不需要知道任何其他 token 的位置。这就是"以绝对位置的实现得到相对位置的性质"。

用实数矩阵写，位置 $$m$$ 的旋转是一个分块对角矩阵 $$R_m$$，每块是二维旋转：

$$
R_m^{(i)} = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix}
$$

旋转矩阵满足 $$R_m^\top R_n = R_{n - m}$$，所以 $$(R_m q)^\top (R_n k) = q^\top R_{n - m} k$$，与复数形式一致。

再看一个直接的推论：与正弦编码不同，RoPE 对 $$e^{\mathrm{i}(m-n)\theta_i}$$ 的依赖是**乘性**的，相对距离通过 $$\cos((m-n)\theta_i)$$ 和 $$\sin((m-n)\theta_i)$$ 调制每一对的内积。高频对（$$\theta_i$$ 大）在距离变化几个 token 时就转过很多圈，对局部顺序敏感；低频对（$$\theta_i$$ 小）在几千个 token 内只转一小段弧，提供"大致离多远"的信息。这个直觉是第四章的基础。

### 3. 实现：rotate_half 与 cos/sin 缓存

实际实现不会真的去做复数乘法。展开二维旋转：

$$
\begin{pmatrix} x_1' \\ x_2' \end{pmatrix} = \begin{pmatrix} x_1 \cos m\theta - x_2 \sin m\theta \\ x_2 \cos m\theta + x_1 \sin m\theta \end{pmatrix}
$$

写成向量形式就是 $$x' = x \odot \cos + \text{rotate}(x) \odot \sin$$，其中 $$\text{rotate}(x)$$ 把每一对的两个分量交换并给第一个取负。原论文按相邻维度 $$(2i, 2i+1)$$ 配对；HuggingFace transformers 的 Llama 实现（以及大多数推理引擎）按 $$(i, i + d_{head}/2)$$ 配对，即前一半与后一半对应位置配对，于是：

$$
\text{rotate\_half}(x) = \left[ -x_{[d/2:]},\; x_{[:d/2]} \right]
$$

两种配对的维度布局对照如下（$$d_{head} = 128$$，第 $$i$$ 对共用同一个 $$\theta_i$$）：

```text title="RoPE 两种维度配对的布局对照"
原论文（相邻配对）  第 i 对 = (x_2i, x_2i+1)
  x0 x1 | x2 x3 | x4 x5 | ... | x126 x127
  └─┬─┘   └─┬─┘   └─┬─┘         └───┬────┘
   θ_0     θ_1     θ_2              θ_63

HF rotate_half（前后半配对）  第 i 对 = (x_i, x_i+64)
  x0  x1  x2  ...  x63 | x64  x65  x66  ...  x127
  │   │   │         │     │    │    │          │
  θ_0 θ_1 θ_2      θ_63   θ_0  θ_1  θ_2       θ_63
  └───┴───┴─────────┴──── 对应位置两两配对 ────┘
```

两种配对方式在数学上等价（只是维度的一个固定置换），但 checkpoint 的 $$W_Q$$、$$W_K$$ 行顺序要与配对方式一致——Llama 官方权重转成 HF 格式时对 q/k 投影做的那次 permute 就是为此。这是位置编码唯一会出现在"权重转换"环节的地方，也是一个常见的精度对不上的来源。

计算上，$$\cos(m\theta_i)$$ 和 $$\sin(m\theta_i)$$ 与输入无关，可以预先算好一张 $$[L_{max}, d_{head}]$$ 的表（cos/sin 各一张，$$L_{max} = 131072$$、$$d_{head} = 128$$ 时 BF16 各 32 MiB；FP32 各 64 MiB），推理时按位置索引取行、与 q 和 k 做一次逐元素乘加。每 token 每层的额外算量约 $$6 \cdot n_h \cdot d_{head}$$ 次乘加（q 和 k 各三个逐元素操作），对 Llama-3-8B 是每层约 $$6 \times 4096 \times 2 \approx 49$$ KFLOPs，相比每层 436 MFLOPs 的权重 GEMM 是万分之一，可以忽略。它真正的开销在访存：这是一个逐元素的 memory-bound 操作，所以推理引擎通常把它融进 QKV 投影之后的 kernel 或 attention kernel 里（vLLM 的 `rotary_embedding` kernel 会在写入 KV cache 之前原位完成旋转）。

### 4. RoPE 与 KV cache 的关系

第八篇讲 KV cache 时默认存的是投影后的 $$K$$、$$V$$。有了 RoPE 之后，存的是**旋转后**的 $$k_n = R_n W_K x_n$$。因为 $$R_n$$ 只依赖 $$n$$，每个 token 的 key 只需要在它进入时旋转一次，之后所有 query 都可以直接用；这是 RoPE 与 KV cache 天然兼容的原因。相对位置 bias 一类方法（T5、Transformer-XL）其实也能缓存 K、V——bias 加在 logits 上、不进 K，每个新 query 只需算自己那一行 $$b_{m-n}$$（$$s$$ 个标量或一次查表），ALiBi 就是它的特例；区别在于 bias 是 attention kernel 里的一个额外项，而 RoPE 在 kernel 之前逐元素做完、kernel 本身不用知道位置。

MLA（DeepSeek-V2/V3）把 K、V 压成一个 512 维的 latent $$c$$，decode 时把 $$W_{UK}$$ 吸收进 query 一侧。问题是旋转矩阵 $$R_n$$ 夹在 $$W_{UK}$$ 与 $$c_n$$ 之间，无法与 $$W_{UK}$$ 交换次序，所以吸收后 $$c_n$$ 上没法再补 RoPE。DeepSeek 的解法是把位置信息分离到一个独立的 64 维 "decoupled RoPE" key 上（$$d_h^R = 64$$），与 latent 一起缓存——每层每 token $$(512 + 64) \times 2 = 1152$$ 字节，61 层 68.6 KiB，这是第八篇 8.6 GiB（128K 上下文）的来源。位置编码的形式直接决定了 KV cache 的结构。

## 四、波长：RoPE 的频谱

### 1. 每个维度对的波长

第 $$i$$ 对以角频率 $$\theta_i$$ 旋转，位置每前进 1 转过 $$\theta_i$$ 弧度，转满一圈（$$2\pi$$）需要的 token 数就是它的波长：

$$
\lambda_i = \frac{2\pi}{\theta_i} = 2\pi \cdot \text{base}^{2i/d_{head}}
$$

代入 $$d_{head} = 128$$（Llama、Mistral、Qwen、DeepSeek 的 RoPE 头都是 128 或 64 维，前者更普遍），base 分别取 10000（原论文、Llama 2、Mistral 7B）和 500000（Llama 3）：

```text title="各维度对的波长：base 10000 与 500000"
                      base = 10000                     base = 500000
 i     theta_i        wavelength lambda_i      theta_i        wavelength lambda_i
 0     1.000e+00           6.28                1.000e+00           6.28
16     1.000e-01          62.8                 3.761e-02         167.1
32     1.000e-02         628                   1.414e-03        4443
48     1.000e-03        6283                   5.318e-05      118143
63     1.155e-04       54410  (约 5.4 万)      2.455e-06     2559196  (约 250 万)
```

base 10000 时，波长从 6 个 token 到 5.4 万个 token 跨越四个数量级，是一个几何级数（每 16 对乘 10）。base 提到 500000 后，$$i = 0$$ 不变（$$\theta_0 = 1$$ 与 base 无关），其余各对波长都被拉长，$$i = 63$$ 的波长从 5.4 万拉到 256 万。

### 2. 训练长度 8K 时，哪些维度对"没转完一圈"

假设训练上下文 $$L = 8192$$。训练中模型见过的相对距离 $$m - n$$ 最多是 8191。第 $$i$$ 对在这个范围内转过的角度最多是 $$8191 \cdot \theta_i$$；如果 $$\lambda_i > 8192$$，这一对在整个训练过程中**没有转完一圈**，模型只见过 $$[0, 2\pi \cdot 8192/\lambda_i)$$ 这一段相位。

$$\lambda_i > 8192$$ 等价于 $$\text{base}^{2i/d_{head}} > 8192/2\pi \approx 1304$$。对 base 10000：

$$
\frac{2i}{128} \cdot \log_{10} 10000 > \log_{10} 1304 \;\Rightarrow\; i > 49.8
$$

即 $$i = 50, \ldots, 63$$ 共 14 对没转完一圈（$$\lambda_{50} \approx 8379$$，$$\lambda_{49} \approx 7256$$）。前 50 对（$$\lambda_i < 8192$$）在训练中都转过至少一整圈，模型见过它们所有的相位。

对 base 500000，同样的条件给出 $$i > 34.98$$，即 $$i = 35, \ldots, 63$$ 共 29 对没转完一圈。base 越大，8K 训练下"没转完一圈"的维度**越多**，不是越少——这一点常被误解，后面第 4 小节会回到它。

### 3. 为什么外推失败

现在把这个 8K 训练的模型直接用在 32K 的输入上。相对距离 $$m - n$$ 最大变成 32767。分三种维度对讨论：

- **高频对**（$$\lambda_i \ll 8192$$，如 $$i \le 40$$，波长不到 2000）：训练中已经转过很多圈，所有相位都见过。距离 32767 对它们只是"又转了几十圈"，$$\cos((m-n)\theta_i)$$ 的取值分布与训练时一致。这些维度不受影响。
- **低频对**（$$\lambda_i > 8192$$，$$i \ge 50$$）：训练时只见过 $$[0, 2\pi \cdot 8192/\lambda_i)$$ 这段弧。以 $$i = 63$$ 为例，$$\lambda_{63} \approx 54410$$，训练中最多转过 $$8192/54410 \approx 0.15$$ 圈，即 $$54°$$。推 32K 时要转到 $$0.6$$ 圈（$$217°$$）——$$\cos$$ 从训练中见过的 $$[0.59, 1]$$ 区间跑到了 $$-0.8$$。对这些维度而言，$$q_i \bar{k}_i e^{\mathrm{i}(m-n)\theta_i}$$ 落在了一个模型**从未见过的相位**上，它对这些维度的 $$W_Q$$、$$W_K$$ 学到的任何模式都建立在"这一对的相位不会超过 $$54°$$"的前提下。
- **中间对**（$$\lambda_i$$ 在几千量级）：部分相位见过，部分没见过。

把几个代表性维度对在训练（8K）与推理（32K）下扫过的相位范围画在同一把 0°–360° 的尺子上（base 10000，每格 15°；满格表示至少转完一圈、全部相位见过）：

```text title="训练 8K 与推理 32K 各维度对扫过的相位"
维度对          场景       0°  转过的相位（每格 15°）  360°
i=48  λ=6283    训练 8K    [########################]  1.30 圈，全相位见过
                推理 32K   [########################]  5.2 圈，没有新相位
i=52  λ=11174   训练 8K    [##################......]  264°
                推理 32K   [########################]  2.9 圈 → 96° 从未见过
i=56  λ=19869   训练 8K    [##########..............]  148°
                推理 32K   [########################]  1.65 圈 → 212° 从未见过
i=63  λ=54410   训练 8K    [####....................]  54°
                推理 32K   [##############..........]  217° → 163° 从未见过
```

结果是 attention 分数在长距离上出现训练时没有的取值，而 softmax 对分数是指数敏感的：某几个错误的高分就会把注意力吸走，perplexity 在超过训练长度后迅速发散。这不是"模型不够聪明"，而是低频维度上的输入分布发生了偏移。

一个常被忽略的细节：attention 分数是 64 对的**和**。即使只有 14 对（约 22%）的相位出界，只要这 14 对上 $$\lvert q_i \rvert \cdot \lvert k_i \rvert$$ 不小，总分就会被带偏。经验上模型恰恰倾向于在低频维度上放较大的范数——因为训练中低频维度几乎是单调的（相位没转完一圈时 $$\cos$$ 是单调的），是模型判断"离多远"最好用的特征。

### 4. base 10000 → 500000 解决了什么，没解决什么

Llama 3 把 base 提到 500000，并在 8K 上下文上预训练。从波长表看，它**解决**的是：

- 低频维度的波长被大幅拉长（$$i = 63$$ 从 5.4 万到 256 万，$$i = 48$$ 从 6283 到 11.8 万），在后续扩展到 128K 时，$$\lambda_i > 131072$$ 的维度对有 15 对（$$i \ge 49$$），它们在 128K 内仍然是"单调"的，能提供不重复的长距离位置信号。base 10000 下没有任何一对的波长超过 131072——128K 范围内每一对都至少转完了两圈，"相对距离 500 与 55000"在低频对上几乎不可区分（相位相差刚好接近一圈）。
- 更大的 base 意味着同样的位置范围对应更小的相位变化，模型在 8K 训练后把上下文扩到 128K 时，需要"填补"的相位缺口相对更规则（YaRN 一类方法的分段处理正是利用这个几何结构，见第五章）。

它**没有解决**的是：

- **仍然需要在长序列上训练**。base 500000 在 8K 训练后，$$i \ge 35$$ 的 29 对都没转完一圈；直接推 32K 一样会遇到没见过的相位，只是出界的角度更小（$$i = 63$$ 从 $$1.15°$$ 到 $$4.6°$$，几乎不出界；但 $$i = 40$$ 波长约 2.2 万，出界很明显）。Llama 3.1 之所以能到 128K，是在 8K 预训练后分阶段用长序列继续训练了 800B token，再配合位置缩放，而不是换 base 就完事。
- **高频维度不变**。$$\theta_0 = 1$$ 永远是 1，前十几对的波长几乎不受 base 影响（$$i = 16$$ 从 62.8 到 167）。这些维度负责局部顺序，本来就不是外推的瓶颈，但也说明改 base 只是对频谱的低端做了重新分配。
- **attention 熵随长度增长**。把 softmax 的分母从 8192 项变成 131072 项，即便分数分布不变，注意力也会被摊薄——均匀分布的熵从 $$\ln 8192 \approx 9.0$$ 涨到 $$\ln 131072 \approx 11.8$$。这是所有位置编码都无法处理的问题，YaRN 的温度修正就是为它准备的。

至此可以回答本篇的核心问题：**8K 训练的 RoPE 模型推不了 32K，是因为低频维度对在训练中没转完一圈，32K 上出现了从未见过的相位；改 base 是在频谱低端腾出更长的波长，让长距离位置在数学上可区分，但"见过"这件事只能靠训练。**

## 五、长上下文扩展方法

所有扩展方法面对的是同一个问题：训练长度 $$L$$，目标长度 $$L' = s \cdot L$$（$$s$$ 是扩展倍数，下面用 factor 表示以免与序列长度混淆），如何让 $$[0, L')$$ 内的位置在每个维度对上都落在模型见过的相位范围内。

### 1. Position Interpolation：把位置压回训练范围

Position Interpolation（PI，Chen 等 2023）的做法最直接：把位置除以 factor。

$$
m' = \frac{m}{\text{factor}}, \qquad \text{等价于} \quad \theta_i' = \frac{\theta_i}{\text{factor}}
$$

$$L' = 32768$$、factor 4 时，位置 32767 被映射到 8191.75，每一对的相位范围与训练时完全一致——不存在没见过的相位。代价是所有维度的分辨率都降低了 4 倍：原本相邻 token 在第 0 对上相差 $$1$$ 弧度（$$57°$$），现在只差 $$0.25$$ 弧度。高频维度负责局部顺序，被压缩后相邻 token 变得"挤在一起"，模型需要微调才能重新分辨；PI 论文报告用约 1000 步微调可以把 Llama 扩到 32K。它的短板是在 factor 较大时（如 16、32）高频维度损伤太大，微调后仍有明显的短文本性能下降。

### 2. NTK-aware 插值：改 base 而不是改位置

NTK-aware 插值（最早由 bloc97 在 2023 年以社区帖子形式提出，后被 YaRN 论文正式化）的观察是：高频维度不需要插值（它们早就转完了所有相位），低频维度才需要。PI 对所有维度一视同仁地除以 factor 是浪费。

它的做法是改 base：

$$
\text{base}' = \text{base} \cdot \text{factor}^{d_{head}/(d_{head} - 2)}
$$

看看这个指数从哪来。新的角频率是 $$\theta_i' = (\text{base}')^{-2i/d_{head}}$$。在最低频的一对 $$i = d_{head}/2 - 1$$ 上：

$$
\theta'_{d/2-1} = \text{base}^{-(d-2)/d} \cdot \text{factor}^{-\frac{d}{d-2} \cdot \frac{d-2}{d}} = \frac{\theta_{d/2-1}}{\text{factor}}
$$

即最低频一对被精确地插值了 factor 倍（与 PI 相同），而 $$i = 0$$ 的一对 $$\theta_0' = 1$$ 完全不动；中间各对按几何级数平滑过渡。$$d_{head} = 128$$、base 10000、factor 4 时 $$\text{base}' = 10000 \times 4^{128/126} \approx 40890$$。这就是"改 base"与"扩上下文"之间的定量关系：Llama 3 直接用 base 500000 预训练，效果上相当于在 base 10000 的频谱基础上把低频端预先拉长了 $$50^{126/128} \approx 47$$ 倍。

NTK-aware 的问题是它对高频维度**完全**不动，而某些中高频维度的波长其实略大于 $$L$$ 的一个分数，它们外推时也会轻微出界。它也没有处理熵的问题。

还有一个与"改哪个量"无关、但工程上很重要的变体：**Dynamic NTK**（HF 的 `rope_scaling.type = "dynamic"`）。上面的 factor 是固定的——即使当前序列只有 2K，频谱也已经按 factor 4 拉长，短文本的分辨率白白受损。Dynamic 版把 factor 变成当前序列长度的函数：$$s \le L$$ 时 factor = 1，原频谱一点不动；$$s > L$$ 时实时重算 base' 与 cos/sin 表。论文里的定义是 $$\text{factor} = s/L$$；transformers 的实现（`modeling_rope_utils.py` 的 dynamic 分支，5.x）用的是 $$\text{base}' = \text{base} \cdot [\text{factor} \cdot s/L - (\text{factor} - 1)]^{d/(d-2)}$$，多了一个配置里的 `factor` 参与——读实现时要按具体版本对。代价在推理侧：base 随长度变，cos/sin 表不能预计算一次用到底，每当序列跨过 $$L$$ 后每步都要更新；更麻烦的是 KV cache 里已存的 k 是用**旧** base 旋转的，与新 base 下的 q 不一致——严格实现要么重算已缓存的 k（违背 KV cache 的初衷），要么接受这个不一致（HF 的实现选了后者）。这就是 vLLM 这类推理框架对 dynamic 支持有限、生产上多用静态 YaRN 的原因。

### 3. YaRN：按波长分三段，再修正温度

YaRN（Yet another RoPE extensioN，Peng 等 2023）把 NTK-aware 的"按频率区分对待"做成了显式的分段规则。定义第 $$i$$ 对在训练长度内转过的圈数：

$$
r_i = \frac{L}{\lambda_i}
$$

$$r_i$$ 大，说明这一对在训练中转过很多圈，所有相位都见过，不该动；$$r_i$$ 小，说明连一圈都没转完，需要完全插值。YaRN 用两个阈值 $$\alpha$$、$$\beta$$（Llama 系列推荐 $$\alpha = 1$$、$$\beta = 32$$）分三段：

$$
\gamma_i = \begin{cases} 0, & r_i < \alpha \quad \text{（低频：完全插值）} \\ 1, & r_i > \beta \quad \text{（高频：不动）} \\ \dfrac{r_i - \alpha}{\beta - \alpha}, & \text{其他（线性混合）} \end{cases}
$$

$$
\theta_i' = (1 - \gamma_i) \cdot \frac{\theta_i}{\text{factor}} + \gamma_i \cdot \theta_i
$$

对 base 10000、$$d_{head} = 128$$、$$L = 8192$$：$$r_i > 32$$ 对应 $$\lambda_i < 256$$，即 $$i \le 25$$ 的 26 对不动；$$r_i < 1$$ 对应 $$\lambda_i > 8192$$，即 $$i \ge 50$$ 的 14 对完全插值；中间 24 对线性混合。这与第四章"没转完一圈"的分析完全对应：完全插值的恰好就是那 14 对。

把三种方法对每个维度对的缩放比 $$\theta_i / \theta_i'$$ 画在同一张图上（base 10000、$$d_{head} = 128$$、$$L = 8192$$、factor 4），三条曲线的形状就是三种方法的全部区别：

![PI、NTK-aware、YaRN 三种方法对 64 个维度对的缩放比：PI 是水平线 4，NTK-aware 从 1 指数过渡到 4，YaRN 在 λ<256 不动、λ>8192 完全插值、中间线性混合](/img/in-post/positional-encoding-and-long-context-rope-scaling.svg)

PI 对所有维度一刀切；NTK-aware 用一条指数曲线让高频端少动、低频端多动，但 $$i = 40$$ 附近（波长约 2000、训练中转过 4 圈）仍被缩了 2.4 倍；YaRN 把这条曲线"拉直"成三段，$$i \le 25$$ 严格不动。

YaRN 的第二个部分是**attention 温度**。为了对抗长上下文下 softmax 被摊薄，它在 logits 上除以一个 $$t < 1$$：

$$
\text{softmax}\!\left(\frac{q_m^\top k_n}{t \sqrt{d_{head}}}\right), \qquad \sqrt{1/t} = 0.1 \ln(\text{factor}) + 1
$$

factor 4 时 $$\sqrt{1/t} \approx 1.139$$、$$1/t \approx 1.30$$；factor 16（8K → 128K）时 $$\sqrt{1/t} \approx 1.277$$、$$1/t \approx 1.63$$。实现上不改 attention kernel，而是把 cos/sin 表整体乘以 $$\sqrt{1/t}$$——q 和 k 各被放大 $$\sqrt{1/t}$$，内积放大 $$1/t$$。这个技巧使 YaRN 对任何现成的 attention kernel 都是透明的。

YaRN 在 Llama 2 上以 factor 16、约 400 步微调扩到 64K，比 PI 需要的数据少一个数量级。

### 4. Llama 3.1 的分段缩放

Llama 3.1 的 `config.json` 里 `rope_scaling` 是：

```json title="Llama 3.1 的 rope_scaling 配置"
{
  "rope_type": "llama3",
  "factor": 8.0,
  "low_freq_factor": 1.0,
  "high_freq_factor": 4.0,
  "original_max_position_embeddings": 8192
}
```

它的规则用波长写最清楚。令 $$L_0 = 8192$$，两个阈值波长：

$$
\lambda_{low} = \frac{L_0}{\text{low\_freq\_factor}} = 8192, \qquad \lambda_{high} = \frac{L_0}{\text{high\_freq\_factor}} = 2048
$$

对每一对：

$$
\theta_i' = \begin{cases} \theta_i, & \lambda_i < 2048 \quad \text{（高频，训练中转过 4 圈以上）} \\ \theta_i / 8, & \lambda_i > 8192 \quad \text{（低频，没转完一圈）} \\ (1 - \gamma_i)\,\theta_i/8 + \gamma_i\,\theta_i, & \text{其他}, \;\; \gamma_i = \dfrac{L_0/\lambda_i - 1}{4 - 1} \end{cases}
$$

与 YaRN 对照：$$L_0/\lambda_i$$ 就是 $$r_i$$；`low_freq_factor` 就是 $$\alpha = 1$$，`high_freq_factor` 就是 $$\beta = 4$$；中间段是同样的线性混合。Llama 3.1 的缩放**就是 YaRN 的分段规则**，只是把 $$\beta$$ 从 32 收紧到 4（更多维度被判定为"高频不动"），并且**没有**温度修正——Llama 3.1 用长序列继续训练来解决熵的问题，而不是靠温度。

代入 base 500000：$$\lambda_i < 2048$$ 的是 $$i \le 28$$ 共 29 对（$$\lambda_{28} \approx 1957$$），完全不动；$$\lambda_i > 8192$$ 的是 $$i \ge 35$$ 共 29 对（$$\lambda_{35} \approx 8219$$），全部除以 8；中间 $$i = 29, \ldots, 34$$ 共 6 对混合（$$i = 32$$ 的 $$\theta$$ 缩小约 2.7 倍）。缩放后最低频一对的波长约 2047 万——128K 上下文在它上面只转过 0.6%。

### 5. DeepSeek-V3 与 Qwen 的 YaRN 配置

DeepSeek-V3 的 `config.json`：

```json title="DeepSeek-V3 的 YaRN 配置"
{
  "rope_scaling": {
    "type": "yarn",
    "factor": 40,
    "original_max_position_embeddings": 4096,
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 1.0,
    "mscale_all_dim": 1.0
  },
  "rope_theta": 10000,
  "max_position_embeddings": 163840
}
```

各字段的含义：`type: yarn` 选择 YaRN 的三段规则；`original_max_position_embeddings: 4096` 是分段时用的 $$L$$；`factor: 40` 是扩展倍数（$$4096 \times 40 = 163840$$，即 `max_position_embeddings`）；`beta_fast: 32`、`beta_slow: 1` 分别是 $$\beta$$ 与 $$\alpha$$（YaRN 论文里的记法是 $$\beta$$ 对应快、$$\alpha$$ 对应慢）；`mscale` 与 `mscale_all_dim` 控制温度项 $$0.1 \cdot \text{mscale} \cdot \ln(\text{factor}) + 1$$ 如何施加——DeepSeek 只对 64 维的 decoupled RoPE 部分做旋转缩放，温度则乘到整个 attention 分数上。注意这里的 `rope_theta` 是 10000，DeepSeek 选择了"小 base + 大 factor 的 YaRN"路线，与 Llama 3 的"大 base + 小 factor"是两种到达同一目标的路径。

Qwen2.5 的做法类似：预训练与默认配置是 32K，官方说明中给出的 128K 配置是在 `rope_scaling` 里填 `type: yarn`、`factor: 4.0`、`original_max_position_embeddings: 32768`。因为 HF 的 YaRN 实现是静态的（对所有长度都按 factor 缩放），Qwen 建议只在确实需要处理超过 32K 的输入时才启用它，否则短文本的性能会轻微下降——这正是前面说的"插值损伤高频维度分辨率"的体现。vLLM 与 SGLang 读的就是这几个字段。

把第 1–5 节的方法按"改了哪个量、按什么规则改、还需要什么"放在一起对照：

| 方法 | 改的量 | 高频对 / 低频对 / 中间 | 温度修正 | 训练代价与采用者 |
|---|---|---|---|---|
| Position Interpolation | 位置 $$m \to m/\text{factor}$$，等价于所有 $$\theta_i$$ ÷ factor | 全部 ÷ factor，一刀切；高频分辨率受损 | 无 | 约 1000 步微调（Llama → 32K）；早期社区扩展 |
| NTK-aware | base $$\to \text{base} \cdot \text{factor}^{d/(d-2)}$$ | $$i = 0$$ 不动 / 最低频恰好 ÷ factor / 几何级数指数过渡 | 无 | 可不微调但效果有限；Llama 3 的 base 500000 在效果上等价于此 |
| Dynamic NTK | 同 NTK-aware，但 factor $$= \max(1, s/L)$$ 随当前长度变 | 同上；$$s \le L$$ 时完全不动 | 无 | 无需训练；短文本零损伤，但 cos/sin 表要随长度重算、KV cache 中旧 k 与新 base 不一致，推理框架支持有限 |
| YaRN | 按 $$r_i = L/\lambda_i$$ 分三段 | $$r_i > \beta$$ 不动 / $$r_i < \alpha$$ ÷ factor / 线性混合 | $$\sqrt{1/t} = 0.1\ln(\text{factor}) + 1$$，乘进 cos/sin 表 | 约 400 步微调（Llama 2 → 64K）；DeepSeek-V2/V3（factor 40）、Qwen2.5（factor 4） |
| Llama 3.1 `llama3` | 同 YaRN 分段，$$\alpha = 1$$、$$\beta = 4$$ | $$\lambda_i < 2048$$ 不动（29 对）/ $$\lambda_i > 8192$$ ÷ 8（29 对）/ 线性混合（6 对） | 无，靠长序列训练解决熵 | 8K 预训练后分阶段长序列训练 800B token；Llama 3.1 |

Table: RoPE 外推方法对照：改的量、规则与代价

### 6. ALiBi：不旋转，直接加线性惩罚

ALiBi（Attention with Linear Biases，Press 等 2021）走了完全不同的路：不给 q、k 加任何位置信息，直接在 attention 分数上减去一个与距离成正比的惩罚：

$$
\text{score}(m, n) = \frac{q_m^\top k_n}{\sqrt{d_{head}}} - \mu_h \cdot (m - n), \qquad m \ge n
$$

斜率 $$\mu_h$$ 每个 head 不同，不学习，按几何级数固定：

$$
\mu_h = 2^{-8h/n_h}, \qquad h = 1, \ldots, n_h
$$

$$n_h = 8$$ 时斜率是 $$1/2, 1/4, \ldots, 1/256$$；$$n_h = 32$$ 时从 $$2^{-0.25}$$ 到 $$2^{-8}$$。斜率大的 head 只看得见很近的 token（距离 100 处已经被扣了 50 分），斜率小的 head 能看到几千个 token。

ALiBi 的外推能力很好：训练 1K、推理 2K 几乎不掉 perplexity，因为线性惩罚在任何距离上的"形状"都一样，没有"没见过的相位"这种事。它在 BLOOM、MPT 上被采用。但在长上下文竞争中它被 RoPE 取代，原因有三：

- 它本质上是一个**局部性先验**：所有 head 对远处 token 都有惩罚，模型很难在 10 万 token 之外精确取回一个具体的信息（needle-in-a-haystack 一类的任务表现差）。外推时 perplexity 不涨，很大程度上是因为模型根本没去看远处。
- 它无法表达内容与位置的交互——惩罚只依赖距离，与 q、k 的内容无关。
- 工程上，bias 项要在 attention kernel 里逐元素加，FlashAttention 2 支持 ALiBi 但需要额外的分支；而 RoPE 只在进 kernel 之前对 q、k 做一次逐元素操作，kernel 本身完全不需要知道位置编码的存在。

RoPE 加上第 1–5 节的缩放方法，成了 2023 年之后长上下文模型的事实标准。把第二章与本章出现过的五种位置编码放在一起，从 Infra 关心的几个维度对照：

| 方案 | 注入位置 | 分数依赖 | 位置参数 | 超出训练长度 | KV cache 兼容 | 对 attention kernel 的要求 |
|---|---|---|---|---|---|---|
| 正弦绝对编码（原始 Transformer） | embedding 上**加** $$p_m$$ | 展开含 $$p_m^\top W p_n$$，依赖绝对位置 | 0 | 可计算，模型不会用 | 兼容（位置已在 K 里） | 无 |
| 可学习绝对编码（GPT-2、BERT） | embedding 上加查表行 | 依赖绝对位置 | $$L_{max} \times d$$（GPT-2：1024 × 768） | 物理上不可能（没有那一行） | 兼容 | 无 |
| 相对 bias（T5、Transformer-XL） | logits 上加 $$b_{m-n}$$ | 只依赖 $$m - n$$（T5 是纯标量；Transformer-XL 还有内容–位置交互项） | 每 head 每桶一个标量 | 远距离落入最粗的桶，可用 | 兼容：K、V 照常缓存，新 query 只算自己那一行 bias | 需要 kernel 内加 bias（物化 $$s \times s$$ 或查表） |
| RoPE | q、k 上**乘**旋转 $$R_m$$ | 只依赖 $$m - n$$，且与 q、k 内容交互 | 0 | 低频对出现未见相位，失败；需缩放 + 训练 | 天然兼容：存旋转后的 k | 无（kernel 之前逐元素完成） |
| ALiBi（BLOOM、MPT） | logits 上减 $$\mu_h (m - n)$$ | 只依赖 $$m - n$$，与内容无关 | 0（斜率固定） | 好：惩罚形状不随距离变 | 兼容 | 需要 kernel 内逐元素加 bias（FA2 有分支支持） |

Table: 五种位置编码从 Infra 维度的对照

## 六、实践


### 1. 用 NumPy 实现 RoPE 并验证相对性

按 HF 的 rotate_half 布局实现，并检查 $$q_m \cdot k_n$$ 与 $$q_{m+t} \cdot k_{n+t}$$ 相等：

```python title="NumPy 实现 RoPE 并验证相对性"
import numpy as np

def rope_inv_freq(head_dim, base=10000.0):
    i = np.arange(0, head_dim // 2)
    return base ** (-2.0 * i / head_dim)            # theta_i, [head_dim/2]

def rope_cos_sin(positions, head_dim, base=10000.0, inv_freq=None):
    if inv_freq is None:
        inv_freq = rope_inv_freq(head_dim, base)
    angles = np.outer(positions, inv_freq)           # [T, head_dim/2]
    emb = np.concatenate([angles, angles], axis=-1)  # [T, head_dim]，前后半各一份
    return np.cos(emb), np.sin(emb)

def rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)

def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin

rng = np.random.default_rng(0)
d_head = 128
q = rng.standard_normal(d_head)
k = rng.standard_normal(d_head)

def score(m, n, base=10000.0):
    cos, sin = rope_cos_sin(np.array([m, n]), d_head, base)
    return apply_rope(q, cos[0], sin[0]) @ apply_rope(k, cos[1], sin[1])

s1 = score(100, 37)
s2 = score(100 + 5000, 37 + 5000)     # 同样的相对距离 63，整体平移 5000
print(f"q_100  . k_37   = {s1:+.6f}")
print(f"q_5100 . k_5037 = {s2:+.6f}   diff = {abs(s1 - s2):.1e}")
print(f"q_100  . k_38   = {score(100, 38):+.6f}   (相对距离 62，应当不同)")

# 与复数闭式 Re[sum q_i conj(k_i) e^{i (m-n) theta_i}] 对照
inv_freq = rope_inv_freq(d_head)
qc = q[:64] + 1j * q[64:]
kc = k[:64] + 1j * k[64:]
closed = np.real(np.sum(qc * np.conj(kc) * np.exp(1j * (100 - 37) * inv_freq)))
print(f"closed form     = {closed:+.6f}")
```

输出：

```text title="RoPE 相对性验证的输出"
q_100  . k_37   = +2.210451
q_5100 . k_5037 = +2.210451   diff = 1.2e-12
q_100  . k_38   = +2.006575   (相对距离 62，应当不同)
closed form     = +2.210451
```

平移 5000 个位置后内积在浮点误差内不变，而相对距离变 1 就变了；复数闭式与 rotate_half 实现给出同一个数——两种配对方式（相邻配对与前后半配对）在这里对应于 `qc` 的构造方式，与 `rotate_half` 一致即可。

### 2. 波长表与三种缩放方法扰动后的频率

```python title="三种缩放方法扰动后的频率"
d_head, L, factor = 128, 8192, 4
base = 10000.0
theta = rope_inv_freq(d_head, base)
lam = 2 * np.pi / theta

# Position Interpolation：所有频率除以 factor
pi_theta = theta / factor

# NTK-aware：改 base
ntk_base = base * factor ** (d_head / (d_head - 2))
ntk_theta = rope_inv_freq(d_head, ntk_base)

# YaRN：按 r = L / lambda 分三段（alpha=1, beta=32）
alpha, beta = 1.0, 32.0
r = L / lam
ramp = np.clip((r - alpha) / (beta - alpha), 0.0, 1.0)    # 0: 全插值, 1: 不动
yarn_theta = (1 - ramp) * theta / factor + ramp * theta
yarn_attn_factor = 0.1 * np.log(factor) + 1                # sqrt(1/t)，乘到 cos/sin 上

print(f"NTK base' = {ntk_base:.0f}, YaRN sqrt(1/t) = {yarn_attn_factor:.4f}")
print(f"{'i':>3} {'lambda':>9} {'r=L/lam':>9} {'theta':>10} {'PI':>10} {'NTK':>10} {'YaRN':>10}")
for i in (0, 8, 16, 24, 32, 40, 48, 56, 63):
    print(f"{i:3d} {lam[i]:9.1f} {r[i]:9.2f} {theta[i]:10.3e} "
          f"{pi_theta[i]:10.3e} {ntk_theta[i]:10.3e} {yarn_theta[i]:10.3e}")
print("YaRN 不动 / 混合 / 全插值 的对数:",
      int((ramp == 1).sum()), int(((ramp > 0) & (ramp < 1)).sum()), int((ramp == 0).sum()))
```

输出：

```text title="波长表与 PI / NTK / YaRN 频率输出"
NTK base' = 40890, YaRN sqrt(1/t) = 1.1386
  i    lambda   r=L/lam      theta         PI        NTK       YaRN
  0       6.3   1303.80  1.000e+00  2.500e-01  1.000e+00  1.000e+00
  8      19.9    412.30  3.162e-01  7.906e-02  2.652e-01  3.162e-01
 16      62.8    130.38  1.000e-01  2.500e-02  7.032e-02  1.000e-01
 24     198.7     41.23  3.162e-02  7.906e-03  1.865e-02  3.162e-02
 32     628.3     13.04  1.000e-02  2.500e-03  4.945e-03  5.412e-03
 40    1986.9      4.12  3.162e-03  7.906e-04  1.311e-03  1.029e-03
 48    6283.2      1.30  1.000e-03  2.500e-04  3.478e-04  2.573e-04
 56   19869.2      0.41  3.162e-04  7.906e-05  9.222e-05  7.906e-05
 63   54410.1      0.15  1.155e-04  2.887e-05  2.887e-05  2.887e-05
YaRN 不动 / 混合 / 全插值 的对数: 26 24 14
```

三列的形状对应第五章的分析：PI 一刀切除以 4；NTK-aware 在 $$i = 0$$ 不动、在 $$i = 63$$ 恰好除以 4（$$2.887 \times 10^{-5} = 1.155 \times 10^{-4} / 4$$），中间几何过渡；YaRN 在 $$i \le 25$$ 完全不动，$$i \ge 50$$ 与 PI 相同，中间线性混合。想画图的话，对 `i` 画 `theta / new_theta`（缩放比）的三条曲线即可：PI 是水平线 4，NTK 是从 1 到 4 的指数曲线，YaRN 是从 1 到 4 的分段折线，用 `matplotlib.pyplot.semilogy` 把 $$\lambda_i$$ 画在同一横轴上能直接看到 8192 这条线落在 $$i \approx 50$$。

把 base 换成 500000、按 Llama 3.1 的 `factor 8 / low 1 / high 4 / 8192` 规则跑同一段逻辑（阈值改成 $$\lambda < 2048$$ 不动、$$\lambda > 8192$$ 除以 8），得到不动 29 对、混合 6 对、全插值 29 对，与第五章第 4 节一致。

## 七、本文小结

位置编码在参数量和算量表里几乎不占位置，却决定了模型在多长的序列上还"认得"位置。本篇的结论：

1. 无 mask 的 attention 是置换等变的，causal mask 只给弱的顺序信号，位置要显式注入。正弦编码是加性的，$$q^\top k$$ 展开后依赖绝对位置；可学习编码有硬上限；相对 bias 需要 $$s \times s$$ 的额外项。RoPE 把 $$d_{head}$$ 维向量看成 $$d_{head}/2$$ 个复数、第 $$i$$ 对以 $$\theta_i = \text{base}^{-2i/d_{head}}$$ 旋转，$$q_m^\top k_n = \text{Re}[\sum_i q_i \bar{k}_i e^{\mathrm{i}(m-n)\theta_i}]$$ 只依赖 $$m - n$$——以绝对位置的实现得到相对位置的性质，且与 KV cache 天然兼容。
2. 每一对的波长 $$\lambda_i = 2\pi \cdot \text{base}^{2i/d_{head}}$$ 是理解一切的钥匙。base 10000、$$d_{head} = 128$$ 时从 6.28 到 5.4 万；训练长度 8K 时 $$i \ge 50$$ 的 14 对没转完一圈，推 32K 时这些维度出现从未见过的相位，是外推失败的根源。base 500000 把最低频波长拉到 256 万，让 128K 内的长距离在数学上可区分，但"见过"只能靠在长序列上训练；高频维度不变，attention 熵随长度增长的问题也不归它管。
3. PI 把所有 $$\theta_i$$ 除以 factor；NTK-aware 用 $$\text{base}' = \text{base} \cdot \text{factor}^{d/(d-2)}$$ 使最低频恰好插值 factor 倍、最高频不动；YaRN 按 $$r_i = L/\lambda_i$$ 分三段（$$r > \beta$$ 不动、$$r < \alpha$$ 全插值、中间线性），再用 $$\sqrt{1/t} = 0.1 \ln(\text{factor}) + 1$$ 修正温度；Llama 3.1 的 `factor 8 / low 1 / high 4 / 8192` 就是 $$\alpha = 1$$、$$\beta = 4$$ 的 YaRN 分段规则、不带温度；DeepSeek-V3 与 Qwen2.5 直接用 YaRN 字段。ALiBi 用 $$2^{-8h/n_h}$$ 的线性惩罚，外推好但局部性先验太强、无法表达内容与位置交互，被 RoPE 取代。
位置方案只回答位置能否外推，不能保证模型在长文中检索与推理的质量；"用得起吗"——KV cache、prefill 的二次项和 sliding window 等结构手段——见第九篇[《长上下文的成本与结构手段》](/long-context-cost-and-structural-remedies.html)。在那之前先看 attention 自己：[下一篇《Transformer 与 LLM（08）：Attention 变体与 KV cache》](/attention-variants-and-kv-cache.html)讲 MHA → GQA → MQA → MLA 各把每 token 的 KV 压到多少，其中 MLA 要把 RoPE 从低秩压缩里解耦出来单独缓存——用的正是本篇 $$R_m^\top R_n = R_{n-m}$$ 这条性质。

配套代码：RoPE 的 NumPy 实现与三种缩放的波长表在 [`rope_numpy.py`](https://github.com/arganzheng/ai-learning-labs/blob/main/transformer-and-llm/rope_numpy.py)。

## 八、自测


1. RoPE base 10000、$$d_{head} = 128$$：第 0 对与第 63 对的波长各是多少？训练长度 8192 时有多少对没转完一圈？

   <details markdown="1"><summary>答案</summary>

   $$\lambda_0 = 2\pi \approx 6.28$$；$$\lambda_{63} = 2\pi \times 10000^{126/128} \approx 5.4$$ 万；波长大于 8192 的对：$$10000^{2i/128} > 1304$$，$$i \ge 50$$，共 14 对。

   </details>

2. 为什么 RoPE 用绝对位置实现却得到相对位置的性质？写出关键的一步。

   <details markdown="1"><summary>答案</summary>

   $$q_m^T k_n = \text{Re}[\sum_i q_i \bar k_i e^{\mathrm{i}(m - n)\theta_i}]$$：两个旋转相乘，角度相减，只剩 $$m - n$$。所以 K 可以带着位置缓存，与 KV cache 天然兼容。

   </details>

3. PI（位置插值）把 8K 模型拉到 32K 做了什么？它的代价是什么？

   <details markdown="1"><summary>答案</summary>

   把所有 $$\theta_i$$ 除以 factor 4，等价于把位置压缩 4 倍塞回训练范围；代价是高频维度也被压了 4 倍，相邻 token 的区分度下降，短文本能力受损——NTK-aware 与 YaRN 只插值低频、不动高频就是为了修它。

   </details>


[^q0]: RoPE 把 $$d_{head} = 128$$ 维拆成 64 对，第 $$i$$ 对以 $$\theta_i = \text{base}^{-2i/d_{head}}$$ 旋转，波长从 6.28 到 5.4 万（base 10000）；训练长度 8K 时 $$i \ge 50$$ 的 14 对低频维度还没转完一圈，推到 32K 这些维度出现训练时从未见过的相位，attention 分布崩掉——不是装不下，是没见过。PI / NTK / YaRN 是在不重训的前提下把「没见过的相位」映射回见过的范围。详见[第四章](#四波长rope-的频谱)、[第五章](#五长上下文扩展方法)。
[^q1]: **解决了**：最低频波长拉到 256 万，128K 内的任何两个位置在数学上可区分，为长序列训练提供了可用的位置表示（Llama 3 的做法）。**没解决**：「见过」只能靠在长序列上真的训练，改 base 不省这笔钱；高频维度不受影响；attention 熵随长度增长、注意力被稀释的问题不归它管；成本也不归它管——128K 时 Llama-3-8B 每 token attention 68.7 GFLOPs 是权重项的 4.6 倍，prefill 约 11 秒，KV cache 16 GiB。详见[第五章](#五长上下文扩展方法)；成本见[《Transformer 与 LLM（08）：长上下文的成本与结构手段》](/long-context-cost-and-structural-remedies.html#二长上下文的成本)。
