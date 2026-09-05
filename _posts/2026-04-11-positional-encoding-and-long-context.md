---
layout: post
title: "Transformer 与 LLM（04）：位置编码与长上下文"
subtitle: "Positional Encoding and Long Context: RoPE Wavelengths, Extrapolation and Cost"
tags: [Transformer, LLM, AI, AI-Infra]
catalog: true
---

> 本文是[《Transformer 与 LLM：结构、算量与数值》](/transformer-and-llm-for-infra-engineers.html)系列的第四篇（共七篇）。上一篇：[Attention 变体与 KV cache](/attention-variants-and-kv-cache.html)　下一篇：[MoE：路由、激活参数量与通信形态](/moe-compute-and-communication.html)

前三篇把一个 Transformer 拆成了参数量、算量、访存量和 KV cache 四个数字。这些数字里有一个变量一直被当作常数处理：上下文长度 $$s$$。第二篇算 prefill 时取 $$s = 8192$$，第三篇算 KV cache 时取 $$s = 131072$$，但都没有回答两个问题：模型凭什么知道一个 token 在第几个位置？以及，一个模型能处理的上下文长度到底由什么决定？

这两个问题在结构上由同一个部件回答——位置编码。它在参数量表里几乎不占位置（RoPE 一个参数都没有），在算量表里也可以忽略（一次逐元素乘加），却决定了"上下文长度"这个对 Infra 成本最敏感的维度的上限。上下文长度同时进入 KV cache 的一次项和 attention 算量的二次项：Llama-3-70B 在 128K 上下文下，每个 token 花在 attention 上的算量（344 GFLOPs）已经超过了花在全部权重上的算量（141 GFLOPs）。

本篇要回答的核心问题是：

> **一个用 8K 上下文训练的 RoPE 模型，为什么不能直接推理 32K？把 base 从 10000 改到 500000 解决了什么，没解决什么？**

回答它需要把 RoPE 的每个维度对当作一个有波长的旋转来看。本篇的顺序是：先说明 attention 为什么需要位置信息，再完整推导 RoPE，然后算出它每个维度对的波长，用波长解释外推为什么失败、Position Interpolation / NTK-aware / YaRN / Llama 3.1 各自改了什么；最后回到成本——长上下文的算量与字节数，以及 sliding window、全局/局部交错、attention sink 这些结构手段各自把成本改成了什么函数。

数字基线与前几篇相同：Llama-3-8B（$$d = 4096$$，32 层，32 个 query 头、8 个 KV 头，$$d_{head} = 128$$）、Llama-3-70B（$$d = 8192$$，80 层，64 个 query 头、8 个 KV 头）、DeepSeek-V3（$$d = 7168$$，61 层，128 头，MLA 的 $$d_c = 512$$、$$d_h^R = 64$$）；硬件以 H100 SXM 为准（BF16 dense 989 TFLOPS，3.35 TB/s，80 GB）。所有 FLOPs 与字节数都是理论下界，不是实测。


## 一、为什么需要位置编码

### 1. attention 是置换不变的

单个 attention 头的计算是：

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_{head}}}\right) V
$$

其中 $$Q = XW_Q$$，$$K = XW_K$$，$$V = XW_V$$，$$X \in \mathbb{R}^{s \times d}$$ 是 $$s$$ 个 token 的输入。把输入的行做任意置换 $$P$$（$$X' = PX$$），则 $$Q' = PQ$$、$$K' = PK$$、$$V' = PV$$，于是：

$$
\text{softmax}\!\left(\frac{PQK^\top P^\top}{\sqrt{d_{head}}}\right) PV = P\,\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_{head}}}\right) V
$$

输出只是原输出的同样置换。换句话说，attention 把输入看作一个**集合**而不是序列：第 $$m$$ 个 token 的输出只取决于"其他 token 是什么"，与"它们在哪里"无关。FFN 是逐 token 的，RMSNorm 也是逐 token 的，所以整个 Transformer block 都是置换等变的。因果掩码给了模型一点方向感（只能看前面），但仍然无法区分"前面第 1 个"和"前面第 100 个"。

要让模型读懂"猫追狗"和"狗追猫"的区别，位置信息必须显式注入。注入的方式分为两大类。

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


## 二、RoPE 的推导

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

再看一个直接的推论：与正弦编码不同，RoPE 对 $$e^{\mathrm{i}(m-n)\theta_i}$$ 的依赖是**乘性**的，相对距离通过 $$\cos((m-n)\theta_i)$$ 和 $$\sin((m-n)\theta_i)$$ 调制每一对的内积。高频对（$$\theta_i$$ 大）在距离变化几个 token 时就转过很多圈，对局部顺序敏感；低频对（$$\theta_i$$ 小）在几千个 token 内只转一小段弧，提供"大致离多远"的信息。这个直觉是第三章的基础。

### 3. 实现：rotate_half 与 cos/sin 缓存

实际实现不会真的去做复数乘法。展开二维旋转：

$$
\begin{pmatrix} x_1' \\ x_2' \end{pmatrix} = \begin{pmatrix} x_1 \cos m\theta - x_2 \sin m\theta \\ x_2 \cos m\theta + x_1 \sin m\theta \end{pmatrix}
$$

写成向量形式就是 $$x' = x \odot \cos + \text{rotate}(x) \odot \sin$$，其中 $$\text{rotate}(x)$$ 把每一对的两个分量交换并给第一个取负。原论文按相邻维度 $$(2i, 2i+1)$$ 配对；HuggingFace transformers 的 Llama 实现（以及大多数推理引擎）按 $$(i, i + d_{head}/2)$$ 配对，即前一半与后一半对应位置配对，于是：

$$
\text{rotate\_half}(x) = \left[ -x_{[d/2:]},\; x_{[:d/2]} \right]
$$

两种配对方式在数学上等价（只是维度的一个固定置换），但 checkpoint 的 $$W_Q$$、$$W_K$$ 行顺序要与配对方式一致——Llama 官方权重转成 HF 格式时对 q/k 投影做的那次 permute 就是为此。这是位置编码唯一会出现在"权重转换"环节的地方，也是一个常见的精度对不上的来源。

计算上，$$\cos(m\theta_i)$$ 和 $$\sin(m\theta_i)$$ 与输入无关，可以预先算好一张 $$[L_{max}, d_{head}]$$ 的表（cos/sin 各一张，$$L_{max} = 131072$$、$$d_{head} = 128$$ 时 BF16 各 32 MiB；FP32 各 64 MiB），推理时按位置索引取行、与 q 和 k 做一次逐元素乘加。每 token 每层的额外算量约 $$6 \cdot n_h \cdot d_{head}$$ 次乘加（q 和 k 各三个逐元素操作），对 Llama-3-8B 是每层约 $$6 \times 4096 \times 2 \approx 49$$ KFLOPs，相比每层 436 MFLOPs 的权重 GEMM 是万分之一，可以忽略。它真正的开销在访存：这是一个逐元素的 memory-bound 操作，所以推理引擎通常把它融进 QKV 投影之后的 kernel 或 attention kernel 里（vLLM 的 `rotary_embedding` kernel 会在写入 KV cache 之前原位完成旋转）。

### 4. RoPE 与 KV cache 的关系

第三篇讲 KV cache 时默认存的是投影后的 $$K$$、$$V$$。有了 RoPE 之后，存的是**旋转后**的 $$k_n = R_n W_K x_n$$。因为 $$R_n$$ 只依赖 $$n$$，每个 token 的 key 只需要在它进入时旋转一次，之后所有 query 都可以直接用；这是 RoPE 与 KV cache 天然兼容的原因，也是相对位置 bias 一类方法做不到的（bias 依赖 $$m - n$$，每个新 query 都要重算）。

MLA（DeepSeek-V2/V3）把 K、V 压成一个 512 维的 latent $$c$$，decode 时把 $$W_{UK}$$ 吸收进 query 一侧。问题是旋转矩阵 $$R_n$$ 夹在 $$W_{UK}$$ 与 $$c_n$$ 之间，无法与 $$W_{UK}$$ 交换次序，所以吸收后 $$c_n$$ 上没法再补 RoPE。DeepSeek 的解法是把位置信息分离到一个独立的 64 维 "decoupled RoPE" key 上（$$d_h^R = 64$$），与 latent 一起缓存——每层每 token $$(512 + 64) \times 2 = 1152$$ 字节，61 层 68.6 KiB，这是第三篇 8.6 GiB（128K 上下文）的来源。位置编码的形式直接决定了 KV cache 的结构。


## 三、波长：RoPE 的频谱

### 1. 每个维度对的波长

第 $$i$$ 对以角频率 $$\theta_i$$ 旋转，位置每前进 1 转过 $$\theta_i$$ 弧度，转满一圈（$$2\pi$$）需要的 token 数就是它的波长：

$$
\lambda_i = \frac{2\pi}{\theta_i} = 2\pi \cdot \text{base}^{2i/d_{head}}
$$

代入 $$d_{head} = 128$$（Llama、Mistral、Qwen、DeepSeek 的 RoPE 头都是 128 或 64 维，前者更普遍），base 分别取 10000（原论文、Llama 2、Mistral 7B）和 500000（Llama 3）：

```text
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

结果是 attention 分数在长距离上出现训练时没有的取值，而 softmax 对分数是指数敏感的：某几个错误的高分就会把注意力吸走，perplexity 在超过训练长度后迅速发散。这不是"模型不够聪明"，而是低频维度上的输入分布发生了偏移。

一个常被忽略的细节：attention 分数是 64 对的**和**。即使只有 14 对（约 22%）的相位出界，只要这 14 对上 $$\lvert q_i \rvert \cdot \lvert k_i \rvert$$ 不小，总分就会被带偏。经验上模型恰恰倾向于在低频维度上放较大的范数——因为训练中低频维度几乎是单调的（相位没转完一圈时 $$\cos$$ 是单调的），是模型判断"离多远"最好用的特征。

### 4. base 10000 → 500000 解决了什么，没解决什么

Llama 3 把 base 提到 500000，并在 8K 上下文上预训练。从波长表看，它**解决**的是：

- 低频维度的波长被大幅拉长（$$i = 63$$ 从 5.4 万到 256 万，$$i = 48$$ 从 6283 到 11.8 万），在后续扩展到 128K 时，$$\lambda_i > 131072$$ 的维度对有 15 对（$$i \ge 49$$），它们在 128K 内仍然是"单调"的，能提供不重复的长距离位置信号。base 10000 下没有任何一对的波长超过 131072——128K 范围内每一对都至少转完了两圈，"相对距离 500 与 55000"在低频对上几乎不可区分（相位相差刚好接近一圈）。
- 更大的 base 意味着同样的位置范围对应更小的相位变化，模型在 8K 训练后把上下文扩到 128K 时，需要"填补"的相位缺口相对更规则（YaRN 一类方法的分段处理正是利用这个几何结构，见第四章）。

它**没有解决**的是：

- **仍然需要在长序列上训练**。base 500000 在 8K 训练后，$$i \ge 35$$ 的 29 对都没转完一圈；直接推 32K 一样会遇到没见过的相位，只是出界的角度更小（$$i = 63$$ 从 $$1.15°$$ 到 $$4.6°$$，几乎不出界；但 $$i = 40$$ 波长约 2.2 万，出界很明显）。Llama 3.1 之所以能到 128K，是在 8K 预训练后分阶段用长序列继续训练了 800B token，再配合位置缩放，而不是换 base 就完事。
- **高频维度不变**。$$\theta_0 = 1$$ 永远是 1，前十几对的波长几乎不受 base 影响（$$i = 16$$ 从 62.8 到 167）。这些维度负责局部顺序，本来就不是外推的瓶颈，但也说明改 base 只是对频谱的低端做了重新分配。
- **attention 熵随长度增长**。把 softmax 的分母从 8192 项变成 131072 项，即便分数分布不变，注意力也会被摊薄——均匀分布的熵从 $$\ln 8192 \approx 9.0$$ 涨到 $$\ln 131072 \approx 11.8$$。这是所有位置编码都无法处理的问题，YaRN 的温度修正就是为它准备的。

至此可以回答本篇的核心问题：**8K 训练的 RoPE 模型推不了 32K，是因为低频维度对在训练中没转完一圈，32K 上出现了从未见过的相位；改 base 是在频谱低端腾出更长的波长，让长距离位置在数学上可区分，但"见过"这件事只能靠训练。**


## 四、长上下文扩展方法

所有扩展方法面对的是同一个问题：训练长度 $$L$$，目标长度 $$L' = s \cdot L$$（$$s$$ 是扩展倍数，下面用 factor 表示以免与序列长度混淆），如何让 $$[0, L')$$ 内的位置在每个维度对上都落在模型见过的相位范围内。

### 1. Position Interpolation：把位置压回训练范围

Position Interpolation（PI，Chen 等 2023）的做法最直接：把位置除以 factor。

$$
m' = \frac{m}{\text{factor}}, \qquad \text{等价于} \quad \theta_i' = \frac{\theta_i}{\text{factor}}
$$

$$L' = 32768$$、factor 4 时，位置 32767 被映射到 8191.75，每一对的相位范围与训练时完全一致——不存在没见过的相位。代价是所有维度的分辩率都降低了 4 倍：原本相邻 token 在第 0 对上相差 $$1$$ 弧度（$$57°$$），现在只差 $$0.25$$ 弧度。高频维度负责局部顺序，被压缩后相邻 token 变得"挤在一起"，模型需要微调才能重新分辨；PI 论文报告用约 1000 步微调可以把 Llama 扩到 32K。它的短板是在 factor 较大时（如 16、32）高频维度损伤太大，微调后仍有明显的短文本性能下降。

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

对 base 10000、$$d_{head} = 128$$、$$L = 8192$$：$$r_i > 32$$ 对应 $$\lambda_i < 256$$，即 $$i \le 25$$ 的 26 对不动；$$r_i < 1$$ 对应 $$\lambda_i > 8192$$，即 $$i \ge 50$$ 的 14 对完全插值；中间 24 对线性混合。这与第三章"没转完一圈"的分析完全对应：完全插值的恰好就是那 14 对。

YaRN 的第二个部分是**attention 温度**。为了对抗长上下文下 softmax 被摊薄，它在 logits 上除以一个 $$t < 1$$：

$$
\text{softmax}\!\left(\frac{q_m^\top k_n}{t \sqrt{d_{head}}}\right), \qquad \sqrt{1/t} = 0.1 \ln(\text{factor}) + 1
$$

factor 4 时 $$\sqrt{1/t} \approx 1.139$$、$$1/t \approx 1.30$$；factor 16（8K → 128K）时 $$\sqrt{1/t} \approx 1.277$$、$$1/t \approx 1.63$$。实现上不改 attention kernel，而是把 cos/sin 表整体乘以 $$\sqrt{1/t}$$——q 和 k 各被放大 $$\sqrt{1/t}$$，内积放大 $$1/t$$。这个技巧使 YaRN 对任何现成的 attention kernel 都是透明的。

YaRN 在 Llama 2 上以 factor 16、约 400 步微调扩到 64K，比 PI 需要的数据少一个数量级。

### 4. Llama 3.1 的分段缩放

Llama 3.1 的 `config.json` 里 `rope_scaling` 是：

```json
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

```json
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

RoPE 加上第 1–5 节的缩放方法，成了 2023 年之后长上下文模型的事实标准。


## 五、长上下文的成本

位置编码决定了模型**能不能**处理长上下文；这一章算它**要花多少**。第二篇的两个基本公式重新写在这里：矩阵乘 $$[m, k] \times [k, n]$$ 是 $$2mkn$$ FLOPs，因此每参数每 token 2 FLOPs；attention 对上下文 $$s$$ 的部分，每层每 token $$QK^\top$$ 与 $$PV$$ 各 $$2 \cdot n_h \cdot d_{head} \cdot s = 2ds$$，合计 $$4ds$$。

### 1. 每 token 的 attention 算量与权重算量的交叉点

Llama-3-8B 的权重 GEMM 部分每 token $$2 \times (8.03 - 0.53)\text{B} \approx 15.0$$ GFLOPs（embedding 查表不算 GEMM，lm_head 算）。attention 部分每层 $$4 \times 4096 \times s = 16384\,s$$，32 层 $$0.524 \times 10^6 \cdot s$$ FLOPs：

```text
                      Llama-3-8B                       Llama-3-70B
上下文 s      attention/token   占比           attention/token   占比
  8192          4.3 GFLOPs      22.2%            21.5 GFLOPs     13.2%
 32768         17.2 GFLOPs      53.4%            85.9 GFLOPs     37.9%
131072         68.7 GFLOPs      82.1%           343.6 GFLOPs     70.9%
权重部分       15.0 GFLOPs                       141 GFLOPs
```

Llama-3-70B 在 128K 下：$$4 \times 8192 \times 131072 \times 80 \approx 344$$ GFLOPs，是权重 141 GFLOPs 的 2.4 倍。交叉点（attention 等于权重）在 8B 约 28.6K、70B 约 53.8K——超过这个长度，模型每生成一个 token 的主要算量就不再是"跑一遍权重"，而是"看一遍上下文"。

这个交叉点对 decode 的 Roofline 判断有直接影响。第二篇的结论是 decode 时权重 GEMM 的算术强度约等于 batch 大小 $$B$$（BF16），$$B = 1$$ 时距 H100 的 ridge point 295 差两个数量级，是 memory-bound。attention 对 KV cache 的读取也是 memory-bound 的，而且它**不随 batch 摊薄**——每个请求有自己的 KV cache，$$B$$ 个请求就读 $$B$$ 份。上下文 8K、batch 64 时 8B 模型每步要读 $$128\,\text{KiB} \times 8192 \times 64 = 64$$ GiB 的 KV cache，是权重（16 GB）的四倍。长上下文下 decode 的瓶颈从"读权重"变成"读 KV cache"。

### 2. prefill 的二次项：一个 128K 请求的 11 秒

prefill 要对 $$s$$ 个 token 各算一遍。权重项与 $$s$$ 成正比，attention 项与 $$s^2$$ 成正比（利用因果掩码，只算下三角，取 $$s^2/2$$）：

$$
\text{FLOPs}_{prefill}(s) = 2N_{gemm} \cdot s + 4dL \cdot \frac{s^2}{2}
$$

Llama-3-8B、$$s = 131072$$：

$$
\underbrace{15.0 \times 10^9 \times 131072}_{\approx 2.0\ \text{PFLOP}} + \underbrace{0.524 \times 10^6 \times \frac{131072^2}{2}}_{\approx 4.5\ \text{PFLOP}} \approx 6.5\ \text{PFLOP}
$$

H100 BF16 989 TFLOPS，按 60% MFU 算 593 TFLOPS，$$6.5 \times 10^{15} / 593 \times 10^{12} \approx 11$$ s。同一个模型 8K 的 prefill 约 0.14 PFLOP、0.24 s；128K 是 8K 的 16 倍长度、46 倍算量、46 倍时间。二次项已经占了 70%。

对 70B，128K prefill 约 41 PFLOP（权重 18.5 + attention 22.5），单卡 60% MFU 要 69 s；即便 8 卡 TP 完美线性，也接近 9 s。这就是 TTFT（time to first token）在长上下文下的物理下界：不是调度问题，是算量问题。

### 3. KV cache 的线性项

第三篇的公式：每 token 的 KV cache 字节数为 $$2 \cdot L \cdot n_{kv} \cdot d_{head} \cdot \text{bytes}$$。

```text
                    bytes/token      8K          32K         128K
Llama-3-8B          128 KiB          1.0 GiB     4.0 GiB     16 GiB
Llama-3-70B         320 KiB          2.5 GiB    10.0 GiB     40 GiB
DeepSeek-V3 (MLA)  68.6 KiB          0.54 GiB    2.1 GiB     8.6 GiB
```

Llama-3-8B 一个 128K 请求的 KV cache 是它权重（16.06 GB）的大小；70B 一个 128K 请求 40 GiB，是单张 H100 一半的显存。对 Infra 来说，长上下文的 KV cache 意味着**并发数**的上限：H100 放下 8B 权重后剩约 64 GB，8K 上下文可以放 64 个请求的 KV，128K 只能放 4 个。

### 4. 中间量：s² 的 logits 矩阵

还有一项不在 FLOPs 和 KV cache 里，但在实现上更致命：attention 分数矩阵 $$QK^\top$$ 本身是 $$s \times s$$。128K 时单个 head 的 logits 就是 $$131072^2 \times 2\,\text{B} = 32$$ GiB（BF16），32 个 head 就是 1 TiB。它不可能物化。FlashAttention（Dao 等 2022）把 softmax 拆成分块的 online softmax，logits 只在 SRAM 里存一个块，从来不写回 HBM——长上下文可行的前提不是显存大，而是 attention kernel 从不物化 $$s \times s$$。同样，训练时 attention 的中间激活如果物化，反向传播也需要它，这是 FlashAttention 对长上下文训练的意义。


## 六、缩短成本的结构手段

上面的成本函数是 full attention 的：每个 token 看所有前面的 token。改变"看哪些"就改变了函数形式。

### 1. sliding window attention

Mistral 7B（Jiang 等 2023）让每个 token 只看前面 $$W = 4096$$ 个 token。attention 算量从 $$4dLs$$ 变成 $$4dL \cdot \min(s, W)$$，KV cache 从 $$s$$ 个 token 变成 $$\min(s, W)$$ 个——两者都从 $$O(s)$$ 变成 $$O(W)$$，即常数。Mistral 7B 的每 token KV cache 与 Llama-3-8B 相同（32 层、8 个 KV 头、$$d_{head} = 128$$，128 KiB），但上限是 $$4096 \times 128\,\text{KiB} = 512$$ MiB，无论输入多长；attention 算量上限 $$4 \times 4096 \times 4096 \times 32 \approx 2.1$$ GFLOPs/token，不到权重的 15%。

实现上 KV cache 变成一个环形缓冲区（rolling buffer），位置 $$n$$ 的 K、V 写到槽 $$n \bmod W$$，第 $$W + 1$$ 个 token 覆盖第 1 个。vLLM 对 sliding window 模型的 block 分配就是按这个做的。

代价是信息只能通过层间传递向远处流动：第 $$\ell$$ 层的 token 能间接看到 $$\ell \cdot W$$ 之内的信息（每层扩一个窗口），32 层理论上是 131072，但每一跳都有损。事实上 Mistral 后续的模型（Mistral Large、Mixtral 的部分版本）取消了滑窗，回到 full attention，说明纯滑窗在长程精确检索任务上有代价。

### 2. 全局层与局部层交错

Gemma 2（2024）把两种层交错排列：奇数层用 $$W = 4096$$ 的局部 attention，偶数层用 full attention，1:1 交错。这是一个折中：一半的层保留了远距离精确取回的能力，另一半的 KV cache 与算量被封顶。

KV cache 变成：

$$
\text{KV}(s) = \frac{L}{2} \cdot b \cdot \min(s, W) + \frac{L}{2} \cdot b \cdot s
$$

其中 $$b$$ 是每层每 token 的 KV 字节数。$$s \gg W$$ 时约为 full attention 的一半。attention 算量同理，$$s \gg W$$ 时约减半。它没有改变函数的阶（仍然 $$O(s)$$ 的 KV 与 $$O(s^2)$$ 的 prefill），只是把系数减半；如果全局层占 $$1/k$$，系数就变成 $$1/k$$。这类设计在 2024–2025 年成为长上下文模型的常见选择，全局层比例从 1:1 到 1:5 不等。

### 3. attention sink 与 StreamingLLM

StreamingLLM（Xiao 等 2023）观察到一个现象：在 full attention 训练的模型里，大量的 attention 分数会集中到序列最开头的几个 token 上，无论那几个 token 是什么内容——它们是 softmax 的"泄洪口"（attention sink）：当一个 query 与所有 key 都不相关时，softmax 仍然要把概率分配出去，模型学会了把这些概率倒进开头几个 token。

这解释了为什么朴素的滑窗（丢掉最早的 token）会让 full attention 训练的模型崩溃：sink 被丢了，softmax 的概率没地方去。StreamingLLM 的做法是永远保留开头 4 个 token 的 K、V，再加一个滑动窗口。KV cache 是 $$b \cdot L \cdot (4 + \min(s, W))$$，与滑窗同阶。它使一个 full attention 训练的模型可以在不微调的情况下处理无限长的流式输入——但代价与滑窗一样，窗口之外的信息丢失了，它是"流式稳定"而非"长上下文理解"。

对位置编码有一个细节：保留 sink 并滑动窗口后，位置用的是**cache 内的相对位置**（sink 是 0–3，窗口内从 4 开始连续编号），而不是原始文本中的位置；否则 RoPE 的相对距离会超过训练长度，回到第三章的问题。

### 4. 稀疏 attention 的形态

更一般的做法是让每个 query 只看上下文中的一个子集，子集的选法有多种：

- **块稀疏**（block-sparse）：把 K、V 按块（例如 64 个 token）分组，每个 query 块只看部分 key 块——固定模式（局部块 + 若干全局块，如 Longformer、BigBird）或按块的粗粒度分数动态选（top-k 块）。稀疏度 $$\rho$$（保留的块占比）下算量约为 $$4dLs \cdot \rho$$；KV cache 通常不减（需要保留所有块以便选择），除非配合淘汰策略。
- **DeepSeek 的 NSA**（Native Sparse Attention，Yuan 等 2025）一类：把上下文压成粗粒度的块表示、按块选 top-k 精细 attention、再加一个滑窗，三路结果加权，并在训练时就使用这种结构，让 kernel 形态对齐硬件的块粒度。这一类方法的算量是 $$O(s \cdot k_{blocks} \cdot B_{size})$$ 加压缩部分，接近线性。

这里只提形态，不展开。关键在于任何稀疏方法的实现都要处理**索引与不规则访存**——被选中的 K、V 块散落在 HBM 中，kernel 要 gather，这与 dense attention "顺序读一整段"是完全不同的访存模式，也是稀疏 attention 的理论算量节省常常兑现不到实测的原因。

### 5. 对比：每种手段把成本改成了什么函数

以每层每 token 的 KV 字节 $$b$$、模型层数 $$L$$、上下文 $$s$$、窗口 $$W$$、全局层占比 $$1/k$$、稀疏度 $$\rho$$ 表示：

```text
手段                    KV cache（每请求）                   attention 算量（每 token）
full attention          b · L · s                            4 d L · s
sliding window (W)      b · L · min(s, W)                    4 d L · min(s, W)
全局/局部交错 (1/k)     b · L · [ s/k + (1 − 1/k) min(s, W) ] 4 d L · [ s/k + (1 − 1/k) min(s, W) ]
sink + window           b · L · (4 + min(s, W))              4 d L · (4 + min(s, W))
块稀疏（保留 ρ）        b · L · s（一般不减）                4 d L · s · ρ  + 选择开销
MLA（DeepSeek-V3）      (d_c + d_h^R) · L · s（系数减 57 倍） 与 full attention 同阶（吸收后 FLOPs 更高）
```

值得注意的是最后一行：MLA 减的是 KV cache 的**系数**（从 3.81 MiB 到 68.6 KiB），不改变它对 $$s$$ 的线性依赖，也不减少 attention 算量；而滑窗改的是**阶**（从 $$s$$ 到常数），但丢信息。两者正交，可以叠加。


## 七、对 Infra 的影响汇总

把前两章的成本落到系统上，长上下文带来的影响集中在五处。

**线性项：KV cache 决定并发数。** 每请求 KV cache $$= b \cdot L \cdot s$$，Llama-3-70B 在 128K 是 40 GiB。给定显存预算，最大并发数与上下文长度成反比；这也是 PagedAttention（vLLM）按 block 而不是按最大长度预分配 KV 的原因——大多数请求用不满 128K，但只要**允许** 128K，静态分配就得按 128K 留。GQA（8B/70B 的 8 个 KV 头）与 MLA（DeepSeek-V3 的 68.6 KiB/token）是从模型结构上减这一项的系数。

**二次项：prefill 算量。** 128K 的 prefill 在 8B 上是 6.5 PFLOP、11 s，其中 attention 占 70%。这一项无法靠 batch 摊薄，因为它本身就是 compute-bound 的（算术强度远高于 ridge point）。减少它只能靠减少算量本身：滑窗、交错、稀疏，或者 prefix caching（同一个 system prompt 的 KV 只算一次）。

**TTFT：单请求的物理下界。** 用户感知的首 token 延迟至少等于 prefill 时间。一个 128K 请求在单卡 8B 上的 TTFT 下界约 11 s（60% MFU），要压到 1 s 以内需要至少 11 张卡并行处理同一个请求——这是序列并行的动机之一。

**chunked prefill 的必要性。** 如果调度器让一个 128K 请求一次性 prefill，它会独占 GPU 约 11 s，期间所有正在 decode 的请求全部停顿——它们的 token 间延迟从几十毫秒跳到 11 s。chunked prefill（Sarathi-Serve，Agrawal 等 2023；vLLM 与 SGLang 默认启用）把长 prefill 切成若干个 chunk（例如每次 2K–8K token），每个调度步里让一个 prefill chunk 与若干 decode 请求拼成一个 batch。decode 请求的 KV 读取是 memory-bound、prefill chunk 是 compute-bound，两者拼在一起恰好能同时用满带宽与算力。代价是长请求自己的 TTFT 略微变长，换来其他请求的延迟稳定。

**序列并行 / context parallel 的动机。** 当单个请求的 KV cache（70B 的 40 GiB）或激活（128K 时每层的 hidden state 就是 $$131072 \times 8192 \times 2\,\text{B} = 2$$ GiB）放不进一张卡、或 TTFT 要求单请求必须由多卡并行时，就需要把**序列维度**切到多张卡上。TP 切的是 head 维度（第三篇），每张卡仍要处理全部 $$s$$ 个 token；序列并行切的是 token 维度，每张卡处理 $$s/P$$ 个 token，但 attention 需要所有 token 的 K、V——Ring Attention（Liu 等 2023）让 K、V 块在卡之间环形传递，每张卡对每个到达的 K、V 块做一次局部 attention 并用 online softmax 合并。它引入了新的通信项（每层传一遍全部 K、V），是长上下文训练与超长请求推理的标准手段。

最后一个跨章节的提醒：位置编码的选择会限制以上所有手段。滑窗与 sink 依赖"cache 内相对位置"的重新编号；YaRN 的温度要乘进 cos/sin 表；Llama 3.1 的分段缩放要在 kernel 之前的 inv_freq 计算里实现。推理引擎里 `rope_scaling` 字段解析错误是长上下文精度问题的常见根源之一——数学上只差一个分段规则，效果上是 32K 之后 perplexity 是否发散。


## 八、实践

### 1. 用 NumPy 实现 RoPE 并验证相对性

按 HF 的 rotate_half 布局实现，并检查 $$q_m \cdot k_n$$ 与 $$q_{m+t} \cdot k_{n+t}$$ 相等：

```python
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

```text
q_100  . k_37   = +2.210451
q_5100 . k_5037 = +2.210451   diff = 1.2e-12
q_100  . k_38   = +2.006575   (相对距离 62，应当不同)
closed form     = +2.210451
```

平移 5000 个位置后内积在浮点误差内不变，而相对距离变 1 就变了；复数闭式与 rotate_half 实现给出同一个数——两种配对方式（相邻配对与前后半配对）在这里对应于 `qc` 的构造方式，与 `rotate_half` 一致即可。

### 2. 波长表与三种缩放方法扰动后的频率

```python
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

```text
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

三列的形状对应第四章的分析：PI 一刀切除以 4；NTK-aware 在 $$i = 0$$ 不动、在 $$i = 63$$ 恰好除以 4（$$2.887 \times 10^{-5} = 1.155 \times 10^{-4} / 4$$），中间几何过渡；YaRN 在 $$i \le 25$$ 完全不动，$$i \ge 50$$ 与 PI 相同，中间线性混合。想画图的话，对 `i` 画 `theta / new_theta`（缩放比）的三条曲线即可：PI 是水平线 4，NTK 是从 1 到 4 的指数曲线，YaRN 是从 1 到 4 的分段折线，用 `matplotlib.pyplot.semilogy` 把 $$\lambda_i$$ 画在同一横轴上能直接看到 8192 这条线落在 $$i \approx 50$$。

把 base 换成 500000、按 Llama 3.1 的 `factor 8 / low 1 / high 4 / 8192` 规则跑同一段逻辑（阈值改成 $$\lambda < 2048$$ 不动、$$\lambda > 8192$$ 除以 8），得到不动 29 对、混合 6 对、全插值 29 对，与第四章第 4 节一致。

### 3. llm_cost.py：上下文长度扫描

在前三篇的骨架上新增 `context_scan(cfg, gpu, ctxs)`，输出"上下文长度 → KV cache、prefill FLOPs、attention 占比"：

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    name: str
    hidden: int
    layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    vocab: int
    tie_embeddings: bool = False
    mla_rank: Optional[int] = None   # MLA 的 d_c；None 表示 MHA/GQA
    rope_dim: Optional[int] = None   # MLA 解耦 RoPE 的 d_h^R

LLAMA3_8B = ModelConfig("Llama-3-8B", 4096, 32, 32, 8, 128, 14336, 128256)
LLAMA3_70B = ModelConfig("Llama-3-70B", 8192, 80, 64, 8, 128, 28672, 128256)

@dataclass
class GPU:
    name: str
    hbm_bytes: float
    bandwidth: float   # bytes/s
    bf16_flops: float  # FLOP/s

H100 = GPU("H100 SXM", 80e9, 3.35e12, 989e12)
A100 = GPU("A100 80GB", 80e9, 2.0e12, 312e12)

GiB = 2 ** 30

def param_count(cfg: ModelConfig) -> dict:
    """dense 模型参数量（第一篇的公式，重给以便独立运行）。"""
    d, dh = cfg.hidden, cfg.head_dim
    attn = d * cfg.n_heads * dh + 2 * d * cfg.n_kv_heads * dh + cfg.n_heads * dh * d
    ffn = 3 * d * cfg.d_ff
    per_layer = attn + ffn + 2 * d
    emb = cfg.vocab * d
    head = 0 if cfg.tie_embeddings else cfg.vocab * d
    total = cfg.layers * per_layer + d + emb + head
    return {"per_layer": per_layer, "embedding": emb, "lm_head": head, "total": total}

def kv_bytes_per_token(cfg: ModelConfig, dtype_bytes: int = 2) -> int:
    """每 token 的 KV cache 字节数（第三篇），支持 MLA。"""
    if cfg.mla_rank is not None:
        return cfg.layers * (cfg.mla_rank + cfg.rope_dim) * dtype_bytes
    return 2 * cfg.layers * cfg.n_kv_heads * cfg.head_dim * dtype_bytes

def weight_flops_per_token(cfg: ModelConfig) -> float:
    """权重 GEMM 部分：每参数每 token 2 FLOPs，embedding 查表不计。"""
    p = param_count(cfg)
    return 2.0 * (p["total"] - p["embedding"])

def attn_flops_per_token(cfg: ModelConfig, ctx: int) -> float:
    """对上下文 ctx 做一次 attention（QK^T 与 PV）：每层 4·d·ctx。"""
    return 4.0 * cfg.n_heads * cfg.head_dim * ctx * cfg.layers

def forward_flops_per_token(cfg: ModelConfig, ctx: int) -> float:
    """decode 一个 token、上下文 ctx 时的前向 FLOPs（第二篇）。"""
    return weight_flops_per_token(cfg) + attn_flops_per_token(cfg, ctx)

def prefill_flops(cfg: ModelConfig, ctx: int, causal: bool = True) -> tuple:
    """一次 ctx 长度 prefill 的总 FLOPs，返回 (权重项, attention 项)。"""
    w = weight_flops_per_token(cfg) * ctx
    a = attn_flops_per_token(cfg, ctx) * ctx
    if causal:
        a /= 2
    return w, a

def context_scan(cfg: ModelConfig, gpu: GPU, ctxs, mfu: float = 0.6) -> None:
    """上下文长度 → KV cache、prefill FLOPs、attention 占比。"""
    kv = kv_bytes_per_token(cfg)
    w_tok = weight_flops_per_token(cfg)
    print(f"{cfg.name}: KV {kv/1024:.1f} KiB/token, weights {w_tok/1e9:.1f} GFLOPs/token")
    print(f"{'ctx':>8} {'KV cache':>10} {'prefill':>12} {'prefill@'+str(int(mfu*100))+'%':>12} "
          f"{'attn/token':>12} {'attn share':>10}")
    for s in ctxs:
        w, a = prefill_flops(cfg, s)
        t = (w + a) / (gpu.bf16_flops * mfu)
        a_tok = attn_flops_per_token(cfg, s)
        share = a_tok / (w_tok + a_tok)
        print(f"{s:>8d} {kv*s/GiB:>8.1f} G {(w+a)/1e15:>9.2f} PF {t:>10.2f} s "
              f"{a_tok/1e9:>9.1f} GF {share*100:>9.1f}%")

if __name__ == "__main__":
    for cfg in (LLAMA3_8B, LLAMA3_70B):
        context_scan(cfg, H100, [8192, 32768, 131072])
        print()
```

输出：

```text
Llama-3-8B: KV 128.0 KiB/token, weights 15.0 GFLOPs/token
     ctx   KV cache      prefill  prefill@60%   attn/token attn share
    8192      1.0 G      0.14 PF       0.24 s       4.3 GF      22.2%
   32768      4.0 G      0.77 PF       1.30 s      17.2 GF      53.4%
  131072     16.0 G      6.47 PF      10.90 s      68.7 GF      82.1%

Llama-3-70B: KV 320.0 KiB/token, weights 139.0 GFLOPs/token
     ctx   KV cache      prefill  prefill@60%   attn/token attn share
    8192      2.5 G      1.23 PF       2.07 s      21.5 GF      13.4%
   32768     10.0 G      5.96 PF      10.05 s      85.9 GF      38.2%
  131072     40.0 G     40.74 PF      68.65 s     343.6 GF      71.2%
```

8B 的三列与第五章一致：128K 时 16 GiB、6.5 PFLOP、11 s、attention 占 82%。70B 的权重项脚本给出 139 GFLOPs（$$2 \times (70.55 - 1.05)$$B），正文沿用总纲取整的 141，差异 1.5%，不影响任何结论；70B 的 prefill 时间是"单卡等效"，实际至少要 2 张 H100 才放得下权重。要加 DeepSeek-V3，传入 `mla_rank=512, rope_dim=64` 即可得到 68.6 KiB/token 与 128K 的 8.6 GiB；它的权重 FLOPs 项需要第五篇的 MoE 字段（激活 37B → 74 GFLOPs），attention 项按 128 头、q/k 192 维、v 128 维手算是每层 $$2 \times 128 \times (192 + 128) \cdot s = 81920\,s$$，61 层约 $$5.0 \times 10^6 \cdot s$$（未吸收的朴素形式）。


## 九、小结

位置编码在参数量和算量表里几乎不占位置，却决定了上下文长度这个维度的上限与代价。本篇的结论：

1. attention 是置换不变的，位置必须显式注入。正弦编码是加性的，$$q^\top k$$ 展开后依赖绝对位置；可学习编码有硬上限；相对 bias 需要 $$s \times s$$ 的额外项。RoPE 把 $$d_{head}$$ 维向量看成 $$d_{head}/2$$ 个复数、第 $$i$$ 对以 $$\theta_i = \text{base}^{-2i/d_{head}}$$ 旋转，$$q_m^\top k_n = \text{Re}[\sum_i q_i \bar{k}_i e^{\mathrm{i}(m-n)\theta_i}]$$ 只依赖 $$m - n$$——以绝对位置的实现得到相对位置的性质，且与 KV cache 天然兼容。
2. 每一对的波长 $$\lambda_i = 2\pi \cdot \text{base}^{2i/d_{head}}$$ 是理解一切的钥匙。base 10000、$$d_{head} = 128$$ 时从 6.28 到 5.4 万；训练长度 8K 时 $$i \ge 50$$ 的 14 对没转完一圈，推 32K 时这些维度出现从未见过的相位，是外推失败的根源。base 500000 把最低频波长拉到 256 万，让 128K 内的长距离在数学上可区分，但"见过"只能靠在长序列上训练；高频维度不变，attention 熵随长度增长的问题也不归它管。
3. PI 把所有 $$\theta_i$$ 除以 factor；NTK-aware 用 $$\text{base}' = \text{base} \cdot \text{factor}^{d/(d-2)}$$ 使最低频恰好插值 factor 倍、最高频不动；YaRN 按 $$r_i = L/\lambda_i$$ 分三段（$$r > \beta$$ 不动、$$r < \alpha$$ 全插值、中间线性），再用 $$\sqrt{1/t} = 0.1 \ln(\text{factor}) + 1$$ 修正温度；Llama 3.1 的 `factor 8 / low 1 / high 4 / 8192` 就是 $$\alpha = 1$$、$$\beta = 4$$ 的 YaRN 分段规则、不带温度；DeepSeek-V3 与 Qwen2.5 直接用 YaRN 字段。ALiBi 用 $$2^{-8h/n_h}$$ 的线性惩罚，外推好但局部性先验太强、无法表达内容与位置交互，被 RoPE 取代。
4. 长上下文的成本有一个线性项（KV cache）、一个二次项（prefill 的 attention）和一个不能物化的中间量（$$s \times s$$ logits）。Llama-3-70B 128K 每 token attention 344 GFLOPs 超过权重 141 GFLOPs；Llama-3-8B 128K prefill 6.5 PFLOP、60% MFU 约 11 s。滑窗把两项都变成 $$O(W)$$ 但丢信息；全局/局部交错把系数变成 $$1/k$$；sink + 滑窗让 full attention 模型能流式运行；MLA 减 KV 的系数不改阶。
5. 对 Infra：KV cache 决定并发数，prefill 二次项决定 TTFT 下界，chunked prefill 是为了不让一个 128K 请求独占 GPU 11 s，序列并行是为了把单请求的 KV、激活与 TTFT 分到多卡。

本篇算出的数字汇总（theoretical，BF16，H100 60% MFU）：

```text
                              Llama-3-8B        Llama-3-70B       DeepSeek-V3
RoPE base                     500000            500000            10000 (+YaRN ×40)
d_head (RoPE 维度)            128               128               64 (decoupled)
最低频波长 λ_63 (base 500000) 256 万            256 万            —
权重 FLOPs/token              15.0 GFLOPs       141 GFLOPs        74 GFLOPs
KV bytes/token                128 KiB           320 KiB           68.6 KiB
KV cache   8K / 32K / 128K    1.0 / 4.0 / 16 GiB  2.5 / 10 / 40 GiB  0.54 / 2.1 / 8.6 GiB
attention FLOPs/token  8K     4.3 GFLOPs        21.5 GFLOPs       41 GFLOPs
                       32K    17.2 GFLOPs       85.9 GFLOPs       164 GFLOPs
                       128K   68.7 GFLOPs       344 GFLOPs        655 GFLOPs
attention 占比  8K / 32K / 128K  22% / 53% / 82%  13% / 38% / 71%   36% / 69% / 90%
prefill FLOPs   8K / 32K / 128K  0.14 / 0.77 / 6.5 PF  1.24 / 6.0 / 41 PF   —
prefill 时间（单卡等效）128K   约 11 s           约 69 s           —
attention = 权重 的交叉点      约 28.6K          约 53.8K          —
```

DeepSeek-V3 的 attention FLOPs 按未吸收的朴素形式（128 头、q/k 192 维、v 128 维）计算，吸收后的形式访存更少但 FLOPs 更高，第三篇有讨论；它的 prefill 总量需要第五篇 MoE 的激活参数量才能完整给出。


## 下一篇

[MoE：路由、激活参数量与通信形态](/moe-compute-and-communication.html)
