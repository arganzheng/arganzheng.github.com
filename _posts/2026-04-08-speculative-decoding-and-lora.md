---
layout: post
series: transformer-and-llm
title: "Transformer 与 LLM（15）：投机解码与 LoRA"
subtitle: "Speculative Decoding and LoRA: Two Algorithmic Levers That Leave the Architecture Alone"
tags: [Transformer, LLM, AI, AI-Infra]
catalog: true
date: 2026-04-08 16:00:00
---

> **本篇在系列中的位置。** 第三段的最后一篇。上一篇的量化改的是每步要搬的字节数；本篇的两种方法不改结构、也不改数值格式，改的是算法：投机解码改变每次前向产出几个 token，LoRA 改变训练时有多少参数要更新。完整地图见[总纲](/transformer-and-llm-for-infra-engineers.html)。

[《Transformer 与 LLM（14）：量化》](/quantization-speculative-decoding-and-lora.html)用一个时间模型 $$T(m) = \max(W_{bytes}/BW,\ 2Nm/F)$$ 说明 decode 是 memory-bound 的，算力在空转，量化用省下的字节换时间。本篇的两种方法攻击另外两个成本项：

- **投机解码**用一次前向验证多个 token，攻击的是 decode 每步只产出一个 token 的串行形态；
- **LoRA** 把可训练参数从 $$N$$ 降到 $$N$$ 的千分之几，攻击的是训练状态每参数 16 字节的显存。

两者的收益同样不是无条件的。本篇要回答的核心问题是：

> **同一套投机解码，为什么 batch 1 时加速 2 倍，batch 64 时没有收益？[^q0] LoRA 把可训练参数降到 0.5%，为什么训练显存只降到约八分之一，激活值一点没少？[^q1]**

## 一、总览：同一条 Roofline，另外两个变量

### 1. 本文的路线

投机解码直接复用上一篇第二章的时间模型：它增大每步的 $$m$$，在 memory-bound 区间几乎免费（第二章）。LoRA 在训练侧，主要减少要保存优化器状态的参数个数；训练 kernel 仍受 Roofline 约束，省显存不等于按比例省时间（第三章）。最后把量化、投机解码与 LoRA 三组函数加进 `llm_cost.py`，合成全系列文本模型的成本表（第四章）。

### 2. 本文的章节安排

| 章 | 主题 | 内容 |
|---|---|---|
| 二 | 投机解码：改变 m | 分布等式、期望接受数与加速比、验证 γ+1 个 token 何时免费、草稿从哪里来 |
| 三 | LoRA：改变训练时的 N | 形式与参数量、训练状态 128 GB → 16.7 GB、额外 FLOPs 与 kernel 数、QLoRA |
| 四 | 实践 | 脚本新增的三组函数、文本模型的成本表 |
| 五 | 本文小结 |  |
| 六 | 自测 | 3 道题 |

Table: 本文的章节安排

## 二、投机解码：改变 m

### 1. 问题：一次前向只产出一个 token

decode 每步读 16 GB 权重、产出 $$B$$ 个 token。$$B = 1$$ 时，4.8 ms 里 Tensor Core 只做了 15 GFLOPs，利用率约 0.3%。Roofline 告诉我们：在斜线上，多算几行几乎不多花时间——$$T(m)$$ 在 $$m < 316$$ 时是常数。如果能一次前向验证多个候选 token，就把串行的产出变成了并行的验证。

问题是候选从哪里来，以及怎么保证结果与原模型**一致**。

### 2. 算法与分布等式

记目标模型的分布为 $$p(\cdot \mid \text{prefix})$$，一个便宜的草稿模型的分布为 $$q(\cdot \mid \text{prefix})$$。投机解码（Leviathan 等 2023；Chen 等 2023）一轮做四件事：

1. 草稿模型自回归采样 $$\gamma$$ 个 token $$x_1, \ldots, x_\gamma$$，$$x_i \sim q(\cdot \mid \text{prefix}, x_{<i})$$；
2. 目标模型对 $$\text{prefix}, x_1, \ldots, x_\gamma$$ 做**一次**前向，得到 $$\gamma + 1$$ 个位置的分布 $$p_1, \ldots, p_{\gamma+1}$$；
3. 从 $$i = 1$$ 起逐个判定：以概率 $$\min(1, p_i(x_i) / q_i(x_i))$$ 接受 $$x_i$$；一旦拒绝，从修正分布 $$\text{norm}(\max(0, p_i - q_i))$$ 采样一个 token 替代 $$x_i$$，本轮结束；
4. 若 $$\gamma$$ 个全部接受，再从 $$p_{\gamma+1}$$ 采样一个 token。

每轮至少产出 1 个 token（拒绝时的重采样或全接受时的额外采样），最多 $$\gamma + 1$$ 个。一轮的分支与回退如下：

```mermaid
%% 图：投机解码的一轮：草稿模型自回归 γ 步，目标模型一次前向验证，逐位置接受或拒绝后重采样
flowchart TB
    dr["草稿模型自回归 γ 步<br/>x_1 … x_γ ~ q，成本 γ · c · T(B)"] --> vf["目标模型一次前向 prefix, x_1 … x_γ<br/>得到 p_1 … p_γ+1，成本 T(B(γ+1))"]
    vf --> i1["i = 1"]
    i1 --> acc{"以概率 min(1, p_i(x_i) / q_i(x_i))<br/>接受 x_i？"}
    acc -->|"接受"| more{"i = γ？"}
    more -->|"否，i ← i + 1"| acc
    more -->|"是，γ 个全接受"| bonus["从 p_γ+1 再采样 1 个<br/>本轮产出 γ + 1 个"]
    acc -->|"拒绝"| rs["从 norm(max(0, p_i − q_i)) 重采样替代 x_i<br/>本轮产出 i 个"]
    rs --> rb["回退：丢弃位置 i 及之后的草稿 token 与其 KV；<br/>位置 i 换成重采样的 token，<br/>它的 KV 下一轮才算"]
    rb --> nx["下一轮"]
    bonus --> nx
    nx --> dr
    classDef draft fill:#fdebd0,stroke:#b9770e;
    classDef target fill:#d6eaf8,stroke:#2e6da4;
    classDef ok fill:#d5f5e3,stroke:#1e8449;
    classDef bad fill:#fadbd8,stroke:#c0392b;
    class dr draft;
    class vf target;
    class bonus ok;
    class rs,rb bad;
```

关键性质：**输出分布严格等于 $$p$$**。看单步。在某一位置，草稿提出 $$x$$ 的概率是 $$q(x)$$，被接受的概率是 $$\min(1, p(x)/q(x))$$，所以"接受且输出 $$x$$"的概率是

$$
q(x) \min\left(1, \frac{p(x)}{q(x)}\right) = \min(q(x), p(x))
$$

总接受概率

$$
\beta = \sum_x \min(p(x), q(x))
$$

拒绝的概率是 $$1 - \beta$$，拒绝后从 $$\text{norm}(\max(0, p - q))$$ 采到 $$x$$ 的概率是 $$\max(0, p(x) - q(x)) / Z$$，归一化常数

$$
Z = \sum_y \max(0, p(y) - q(y)) = \sum_y \left( p(y) - \min(p(y), q(y)) \right) = 1 - \beta
$$

于是输出 $$x$$ 的总概率

$$
P(x) = \min(p(x), q(x)) + (1 - \beta) \cdot \frac{\max(0, p(x) - q(x))}{1 - \beta} = \min(p(x), q(x)) + \max(0, p(x) - q(x)) = p(x)
$$

每个位置都从 $$p$$ 采样，且被接受的 token 之后的位置以它为条件——与目标模型自回归采样的联合分布逐位相同。greedy 解码是 $$p$$ 退化为 one-hot 的特例：接受当且仅当草稿与目标 argmax 一致。这个证明不依赖 $$q$$ 是什么——$$q$$ 只影响**效率**，不影响**正确性**。

两点补充。第一，拒绝后的重采样分布 $$\text{norm}(\max(0, p - q))$$ 有直观含义：它只在 $$p(x) > q(x)$$ 的 token 上有质量，即"目标模型认为比草稿更可能"的那些 token——草稿高估的 token 已经被接受步骤按 $$p/q$$ 的比例采纳过了，剩下的概率质量正好是目标模型比草稿多出来的部分。第二，验证时目标模型输出的 $$\gamma + 1$$ 个分布只需要一次前向，是因为因果掩码下每个位置的输出只依赖它之前的 token，草稿序列的每个前缀恰好对应一个位置——这与 prefill 一次算出整个 prompt 所有位置的 KV 是同一件事，投机解码的验证本质上是一次长度为 $$\gamma + 1$$ 的小 prefill。KV cache 的回退要精确到位置：被拒绝的位置 $$i$$ **本身**的 KV 是按草稿 token $$x_i$$ 算的，也要丢掉（或覆盖），保留到 $$i - 1$$；重采样出的新 token 占据位置 $$i$$，它的 KV 在下一轮验证前向里才会算出来；全接受时的 bonus token 同理。这是引擎实现中需要处理的细节。

### 3. 期望接受数与加速比

记单个位置的接受率 $$\alpha = \mathbb{E}[\beta]$$（它等于 $$1 - \text{TV}(p, q)$$，$$p$$ 与 $$q$$ 的总变差距离的补）。假设各位置独立且接受率相同，一轮产出的 token 数是"连续接受的个数 + 1"，期望

$$
\mathbb{E}[\text{tokens}] = 1 + \alpha + \alpha^2 + \cdots + \alpha^\gamma = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}
$$

$$\alpha = 0.8$$、$$\gamma = 4$$：$$(1 - 0.8^5)/0.2 = (1 - 0.328)/0.2 = 3.36$$。

一轮的成本：$$\gamma$$ 次草稿前向加一次目标前向。记草稿一次前向的时间是目标前向的 $$c$$ 倍，并且——这是关键假设——目标模型验证 $$\gamma + 1$$ 个 token 的时间与验证 1 个相同。那么

$$
\text{speedup} = \frac{\mathbb{E}[\text{tokens}]}{\gamma c + 1}
$$

$$c = 0.1$$：$$3.36 / 1.4 = 2.4$$。若草稿几乎免费（$$c = 0.02$$，多头或 n-gram 方案），$$3.36 / 1.08 \approx 3.1$$。

几个变体的数字：

| alpha | gamma | E[tokens] | speedup(c=0.1) | speedup(c=0.02) |
|---|---|---|---|---|
| 0.6 | 4 | 2.31 | 1.65 | 2.13 |
| 0.8 | 2 | 2.44 | 2.03 | 2.35 |
| 0.8 | 4 | 3.36 | 2.40 | 3.11 |
| 0.8 | 8 | 4.33 | 2.40 | 3.73 |
| 0.9 | 4 | 4.10 | 2.93 | 3.79 |

Table: 投机解码几个 α、γ 组合的期望 token 数与加速比

$$\gamma$$ 越大，期望接受数增长越慢（$$\alpha^\gamma$$ 衰减），而草稿成本线性增长；$$c = 0.1$$ 时 $$\gamma = 4$$ 与 $$\gamma = 8$$ 的加速比相同，最优 $$\gamma$$ 由 $$\alpha$$ 与 $$c$$ 共同决定。

### 4. 为什么验证 γ+1 个 token 几乎免费——以及何时不再免费

"验证 $$\gamma + 1$$ 个与验证 1 个同样贵"是 Roofline 的直接推论。目标模型的 GEMM 从 $$[B, k] \times [k, n]$$ 变成 $$[B(\gamma + 1), k] \times [k, n]$$：FLOPs 乘 $$\gamma + 1$$，**权重读取不变**。只要 $$B(\gamma + 1)$$ 仍在 ridge 之下，

$$
T(B(\gamma + 1)) = T(B) = \frac{W_{\text{bytes}}}{BW}
$$

多出的 FLOPs 填的是本来空转的 Tensor Core。KV cache 读取也一样：验证 5 个 token 的 attention 读同一份 KV cache。

条件是 $$B(\gamma + 1) \lesssim \text{ridge}$$，即

$$
B \lesssim \frac{\text{ridge}}{\gamma + 1} \approx \frac{300}{5} = 60
$$

超过这个 batch，验证前向进入 compute-bound，$$T(B(\gamma+1))$$ 开始以 $$(\gamma + 1)$$ 倍于 $$T(B)$$ 的斜率增长。极限情况（完全 compute-bound）加速比变为

$$
\frac{\mathbb{E}[\text{tokens}]}{\gamma c + (\gamma + 1)} = \frac{3.36}{0.4 + 5} \approx 0.62
$$

**低于 1**。原因很朴素：投机解码是用 FLOPs 换延迟——每产出 3.36 个 token 要为 5 个位置做完整前向，FLOPs 效率是 $$3.36/5 = 67\%$$，被拒绝的 token 的计算是白做的。当 FLOPs 是瓶颈时，这笔交易亏本。

用上一篇第二章的时间模型算 Llama-3-8B 在 H100 上各 batch 的加速比（$$\alpha = 0.8$$、$$\gamma = 4$$、$$c = 0.1$$，草稿成本按 $$c \cdot T(B)$$ 计）：

$$
\text{speedup}(B) = \frac{\mathbb{E}[\text{tokens}] \cdot T(B)}{\gamma c \cdot T(B) + T(B(\gamma + 1))}
$$

| batch B | T(B) ms | T(5B) ms | 加速比（峰值算力） | 加速比（60% MFU） |
|---|---|---|---|---|
| 1 | 4.79 | 4.79 | 2.40 | 2.40 |
| 8 | 4.79 | 4.79 | 2.40 | 2.40 |
| 32 | 4.79 | 4.79 | 2.40 | 2.40 |
| 64 | 4.79 | 4.85 | 2.38 | 1.61 |
| 128 | 4.79 | 9.71 | 1.39 | 0.89 |
| 256 | 4.79 | 19.4 | 0.76 | 0.62 |

Table: 投机解码加速比随 batch 的变化

第二列按 60% MFU 折算实际可达算力（593 TFLOPS，有效 ridge 约 177，转折 batch 约 35）：$$B = 64$$ 时验证前向已进入计算区，加速比掉到 1.6；$$B = 128$$ 时低于 1。再算上真实系统里草稿模型在大 batch 下的开销、每轮调度与采样的固定成本、以及 $$\alpha$$ 在不同位置并不独立同分布，**"batch 64 时没有收益"** 是这条曲线的工程表述——精确的转折位置随模型、硬件、$$\gamma$$ 移动，但它的量级由 $$\text{ridge}/(\gamma + 1)$$ 决定。

把上一篇第三章（量化）与本章的数字画在同一条 $$T(m)$$ 曲线上（对数坐标）：量化把 memory-bound 的平台**向下**移，投机解码把工作点**向右**推，两者的收益都止于平台与斜线的交点：

![Llama-3-8B 在 H100 上的 T(m) 曲线：BF16 与 W4A16 两条平台、共同的 compute 斜线，量化下移平台、投机右移工作点](/img/in-post/quantization-speculative-decoding-and-lora-time-model.svg)

于是核心问题的两半合上了：INT4 在 $$B < \text{ridge}/4$$ 时兑现字节收益，投机解码在 $$B < \text{ridge}/(\gamma+1)$$ 时兑现并行验证的收益——**两者都只在 Roofline 的斜线上有效，越过 ridge 就消失甚至反转**。它们优化的是同一个量：memory-bound 区间里被浪费的算力。这也意味着两者可以叠加：W4A16 的目标模型验证 5 个 token 同样几乎免费，只是转折 batch 变成 $$\text{ridge}/(4 \times 5) \approx 15$$。

### 5. 草稿从哪里来

草稿方案决定 $$\alpha$$ 与 $$c$$。以下区间是各论文与工程报告中**通常报告**的量级，不是本文实测：

| 方案 | 草稿形态 | c（相对目标一次前向） | alpha（通常报告） |
|---|---|---|---|
| 独立小模型 | 同 tokenizer 的小模型自回归 γ 步 | 参数量之比，~0.05–0.15 | 0.6–0.8（取决于配对） |
| Medusa（Cai 等 2024） | 目标模型顶层加 K 个头并行预测 t+2.. | ≈0（一次前向内） | 第 1 头 ~0.6–0.7，逐头下降 |
| EAGLE（Li 等 2024） | 一层 Transformer 在特征级自回归起草 | ~0.02–0.05 | ~0.75–0.85（论文报告） |
| n-gram / prompt lookup | 在上下文中查找 n-gram 复制后续 token | ≈0 | 任务依赖：改写/摘要/RAG 高，自由生成低 |
| DeepSeek-V3 MTP | 训练时联合训练的一个额外 block | 1/61 层的量级 | 第二 token 85–90%（技术报告） |

Table: 常见草稿方案的形态、代价 c 与接受率 α（文献通常报告的量级）

- **独立小模型**要求与目标共享 tokenizer（Llama-3-8B 给 70B 起草），$$c$$ 约等于参数量之比，在 memory-bound 区间也等于字节数之比。$$\alpha$$ 取决于两者分布的接近程度，同系列同数据训练的模型配对最好。
- **Medusa** 在目标模型最后一层 hidden state 上接 $$K$$ 个轻量头，第 $$k$$ 个头预测第 $$t + k + 1$$ 个 token，一次前向同时出所有草稿；用 tree attention 一次验证多条候选路径。$$c \approx 0$$，但各头独立预测（没有以前一个草稿为条件），$$\alpha$$ 随头序号下降。论文报告 Medusa-1 约 2.2×、Medusa-2 约 2.3–3.6×。
- **EAGLE** 的观察是：在特征（倒数第二层的 hidden state）而非 token 层面做自回归，不确定性更低；草稿模块只有一层 decoder，输入是目标模型的特征与已采样 token 的 embedding。论文报告 LLaMA2-Chat 70B 上约 2.7–3.5×，EAGLE-2 用动态草稿树进一步提高。它的 $$\alpha$$ 通常高于 Medusa，$$c$$ 是一层对全模型的比例。
- **n-gram / prompt lookup**：把上下文里最近出现的 n-gram 后面接的 token 当草稿，零成本、零训练，在有大量复制的任务（改写、摘要、代码编辑、RAG）上 $$\alpha$$ 很高，在自由生成上接近 0——加速比完全依赖任务。
- **DeepSeek-V3 的 MTP**：训练时就带一个预测下一下个 token 的额外模块，推理时可当草稿用（$$\gamma = 1$$）。技术报告称第二个 token 的接受率在 85–90% 之间，对应 $$\mathbb{E}[\text{tokens}] = 1 + \alpha \approx 1.85 \sim 1.9$$，报告的解码吞吐提升约 1.8×，与 $$c$$ 很小时 $$1.9 / (c + 1)$$ 的估算一致。

所有这些方案共享同一条约束：它们提升的是 $$\alpha$$ 或降低 $$c$$，但都改不了 $$B \lesssim \text{ridge}/(\gamma + 1)$$ 这个收益区间。

## 三、LoRA：改变训练时的 N

> 本节只算 LoRA 的计算形态。它的原理、选参与上线工程见[《LoRA 专题》](/lora-for-sft-from-low-rank-hypothesis-to-serving.html)（三篇）。

### 1. 形式与参数量

全量微调的代价不在 FLOPs 而在**状态**。BF16 混合精度 + Adam 每参数 16 字节（BF16 权重 2 + BF16 梯度 2 + FP32 主权重 4 + Adam 一阶、二阶矩各 4），Llama-3-8B 的训练状态 $$8.03 \times 16 = 128$$ GB，一张 80 GB 的卡放不下，还没算激活值。

LoRA（Hu 等 2021）冻结 $$W \in \mathbb{R}^{d_{out} \times d_{in}}$$，只训练一个低秩增量：

$$
W' = W + \frac{\alpha}{r} B A, \qquad A \in \mathbb{R}^{r \times d_{in}},\ B \in \mathbb{R}^{d_{out} \times r}
$$

$$A$$ 高斯随机初始化，$$B$$ 初始化为零——于是训练开始时 $$BA = 0$$，模型与原模型完全一致，梯度从 $$B$$ 开始流动。$$\alpha / r$$ 是一个缩放常数，让换 $$r$$ 时不必重调学习率。可训练参数 $$r(d_{in} + d_{out})$$，对 $$r \ll \min(d_{in}, d_{out})$$ 远小于 $$d_{in} d_{out}$$。

代入 Llama-3-8B，$$r = 16$$，只加在 attention 四个矩阵上（$$d_{in} = 4096$$；$$W_Q$$、$$W_O$$ 的 $$d_{out} = 4096$$，$$W_K$$、$$W_V$$ 的 $$d_{out} = 8 \times 128 = 1024$$）：

$$
16 \times (8192 + 5120 + 5120 + 8192) = 425{,}984 \ \text{每层}, \qquad \times 32 = 13.63\text{M} \ (0.17\%)
$$

再加 FFN 的 gate/up（$$4096 \to 14336$$）与 down（$$14336 \to 4096$$）：

$$
3 \times 16 \times (4096 + 14336) = 884{,}736 \ \text{每层}, \qquad \times 32 = 28.3\text{M}
$$

合计 41.9M，占 8.03B 的 **0.52%**。Llama-3-70B 同样配置：attention 65.5M（0.093%），全部七个矩阵 207M（0.29%）——模型越大比例越小，因为 LoRA 参数随 $$d$$ 线性增长而 $$W$$ 随 $$d^2$$。

$$r$$ 与作用范围是两个独立的旋钮，参数量对两者都是线性的：$$r = 64$$ 只加 attention 是 54.5M，$$r = 16$$ 加全部七个矩阵是 41.9M，两者相近。LoRA 原论文的实验与后续经验都倾向于后者——**以小 $$r$$ 覆盖更多矩阵，比以大 $$r$$ 只覆盖 attention 更有效**，因为增量的"秩"很低这一假设对每个矩阵都成立，而覆盖 FFN 让适配能触及模型三分之二以上的参数所在。就成本而言两种选择没有区别，都是 0.5% 量级。

### 2. 训练状态：从 128 GB 到 16.7 GB，但激活值不变

LoRA 训练的显存：

```text title="全量微调与 LoRA 的训练显存对照"
                        全量微调                 LoRA r=16（全部七个矩阵）
冻结 / 可训练权重        8.03B × 16 B = 128 GB    BF16 冻结权重 8.03B × 2 B = 16.06 GB
                                                 + LoRA 状态 41.9M × 16 B ≈ 0.67 GB
激活值                   与序列长度成正比           基本相同
```

冻结权重只需要 BF16 一份，没有梯度、没有主权重、没有 Adam 状态；LoRA 参数的 16 字节/参数只作用在 41.9M 上。**权重侧从 128 GB 降到 16.7 GB**，这是 LoRA 能在单卡上微调 8B、在 8 卡上微调 70B 的原因。

但激活值不变。反向传播要经过每一层算 $$\partial L / \partial x$$，这需要每层保存的中间量（RMSNorm 输入、attention 的 softmax 统计、FFN 的 SiLU 输入等），与权重是否冻结无关；$$A$$ 的梯度 $$\partial L / \partial A = B^\top (\partial L / \partial y) x^\top$$ 同样需要保留输入 $$x$$。按 Korthikanti 等 2022 的估算，用 FlashAttention 后每层每 token 约 $$34 \cdot d$$ 字节，Llama-3-8B 每 token 32 层约 4.5 MB，4096 token 的一条序列约 18 GB，8192 token 约 37 GB。所以 LoRA 微调长序列仍然需要激活重算（gradient checkpointing），换来的是额外约一次前向的 FLOPs。

### 3. 计算形态：额外 FLOPs 不到 1%，但 kernel 数翻倍

前向多了两个小矩阵乘：$$x \to xA^\top \to (xA^\top) B^\top$$，FLOPs 是 $$2r(d_{in} + d_{out})$$ 对比原来的 $$2 d_{in} d_{out}$$，比例

$$
\frac{r(d_{in} + d_{out})}{d_{in} d_{out}}
$$

$$W_Q$$：$$16 \times 8192 / 16.78\text{M} \approx 0.78\%$$；$$W_K$$、$$W_V$$：$$16 \times 5120 / 4.19\text{M} \approx 1.95\%$$；FFN 三个矩阵约 0.50%。训练的 FLOPs 几乎全在冻结权重的前向与反向上，**LoRA 省的是状态不是算量**——反向仍要算 $$\partial L / \partial x$$ 穿过每一层，只省掉了 $$\partial L / \partial W$$ 那一项（约占反向的一半），所以 LoRA 训练每 token 约 $$4N$$ 而非 $$6N$$ FLOPs。

一层线性层上的前向与反向，实线是前向、虚线是反向，标出哪些梯度仍要算、哪一项被省掉：

```mermaid
%% 图：LoRA 一层线性层的前向与反向：∂L/∂x 仍要算穿过每一层，∂L/∂W 被省掉
flowchart TB
    x["输入 x<br/>（仍需保存供反向，激活值不省）"] --> W["冻结 W  d_out × d_in<br/>BF16 一份：无梯度、无主权重、无 Adam"]
    x --> A["A  r × d_in<br/>可训练，高斯初始化"]
    A --> B["B  d_out × r<br/>可训练，初始为 0"]
    W --> y["y = Wx + (α/r) BAx<br/>额外 FLOPs ≈ r(d_in + d_out) / (d_in d_out) 不到 1%"]
    B --> y
    y --> gy["∂L/∂y（来自上一层的反向）"]
    gy -.->|"∂L/∂x = Wᵀ ∂L/∂y  仍要算，穿过每一层"| x
    gy -.->|"∂L/∂B = ∂L/∂y (Ax)ᵀ"| B
    gy -.->|"∂L/∂A = Bᵀ ∂L/∂y xᵀ"| A
    gy -.->|"∂L/∂W  不算：省掉反向约一半、省掉 14 B/参数状态"| W
    classDef frozen fill:#eeeeee,stroke:#888;
    classDef train fill:#d5f5e3,stroke:#1e8449;
    classDef act fill:#fdebd0,stroke:#b9770e;
    class W frozen;
    class A,B train;
    class x,y,gy act;
```

推理时有两条路：

- **合并**：$$W' = W + (\alpha/r) BA$$ 算一次存下来，推理与原模型零差别、零开销。单租户部署的默认选择。
- **不合并**：为了让一个底座同时服务多个 LoRA（多租户），$$W$$ 只存一份，每个请求带自己的 $$A_i$$、$$B_i$$。decode 时 $$B = 1$$ 每步除了原来的 7 个 GEMV，多了 14 个极小的 GEMV（$$A$$、$$B$$ 各 $$r \times d$$，$$W_Q$$ 上 $$16 \times 4096 \times 2$$ B = 128 KB），字节数可忽略，但 32 层 × 14 = 448 次额外 kernel 启动，每次几微秒，加起来与 4.8 ms 的下界同量级——这是 decode memory-bound 的又一面：小 kernel 的固定开销比它的 FLOPs 和字节都贵。

多 LoRA 服务把一个 batch 里属于不同 adapter 的行分组：

$$
Y = X W^\top + \begin{bmatrix} X_1 A_1^\top B_1^\top \\ X_2 A_2^\top B_2^\top \\ \vdots \end{bmatrix}
$$

（按上文 $$W \in \mathbb{R}^{d_{out} \times d_{in}}$$、$$B A$$ 与 $$W$$ 同形的约定，行向量 $$X$$ 要右乘转置；$$\alpha / r$$ 略去。）

前一项是所有请求共享的一次 GEMM，后一项是"每段 $$X_i$$ 乘各自的小矩阵"——Punica（Chen 等 2023）称之为 SGMV（Segmented Gather Matrix-Vector），一个 kernel 内按段 gather 不同的 $$A_i$$、$$B_i$$ 完成全部请求；S-LoRA（Sheng 等 2023）在此之上把 adapter 权重与 KV cache 统一分页管理，支持上千个 adapter 常驻。vLLM 的 multi-LoRA 支持基于这类 kernel。它们的成本模型与本篇的 Roofline 一致：adapter 字节数小，瓶颈在 kernel 组织，不在带宽。

### 4. QLoRA：把底座也量化

LoRA 的 16.7 GB 里 16.06 GB 是冻结的 BF16 底座。QLoRA（Dettmers 等 2023）把这一项也量化，训练时反量化到 BF16 参与前向与反向，梯度只流向 BF16 的 LoRA 参数：

- **NF4（NormalFloat 4）**：不是均匀量化，16 个级别取标准正态分布的等概率分位点——预训练权重近似正态分布，这样每个级别被使用的概率相等，信息论上最优。block size 64，每块一个 FP32 absmax scale，元数据 $$32/64 = 0.5$$ bit/权重；
- **双重量化**：把 FP32 的 scale 再按每 256 个一组量化到 FP8，元数据降到 $$8/64 + 32/(64 \times 256) \approx 0.127$$ bit/权重，总计约 4.13 bit；
- **paged optimizer**：用 CUDA 统一内存把优化器状态在显存尖峰时换页到 CPU，避免长序列梯度检查点时的 OOM。

8B 底座：$$8.03\text{B} \times 4.13 / 8 \approx 4.1$$ GB，保留部分层高精度后**约 4.5 GB**；加 LoRA 状态 0.67 GB，权重侧不到 5.2 GB，一张 24 GB 的消费级卡可以微调 8B 模型（激活值决定能开多长的序列）。代价是每次前向和反向都要反量化整份权重，每步时间明显长于 BF16 LoRA——又是第三章的结论：量化省字节，反量化加算量，训练是 compute-bound 的，所以 QLoRA 是**用时间换显存**。

## 四、实践：完成文本模型的成本表

### 1. 脚本新增的三组函数

延续贯穿全系列的 `llm_cost.py`，本篇新增量化字节数、投机解码加速比、LoRA 参数三组函数。为了独立运行，下面同时给出前几篇中本篇用到的 `param_count`、`forward_flops_per_token`、`kv_bytes_per_token` 的 dense 版本（MoE 与 MLA 的版本在第六、九篇）。

```python title="llm_cost.py：量化、投机解码与 LoRA 三组函数"
from dataclasses import dataclass

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

# ---- 前几篇的函数（dense 版本）----
def embedding_params(cfg):
    return cfg.vocab * cfg.hidden * (1 if cfg.tie_embeddings else 2)

def param_count(cfg):
    """第五篇的参数量（重给以便独立运行），返回 dict。"""
    d = cfg.hidden
    q, kv = cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim
    attn = d * q + d * kv + d * kv + q * d
    ffn = 3 * d * cfg.d_ff
    per_layer = attn + ffn + 2 * d
    embed = cfg.vocab * d
    lm_head = 0 if cfg.tie_embeddings else cfg.vocab * d
    total = per_layer * cfg.layers + embed + lm_head + d
    return {"per_layer": per_layer, "embedding": embed, "lm_head": lm_head, "total": total}

def forward_flops_per_token(cfg, ctx=0):
    # embedding 查表不算 GEMM；lm_head 算
    # 输入 embedding 是查表不算 GEMM；tied 模型那张表兼作 lm_head，仍要算
    gemm_params = param_count(cfg)["total"] - (0 if cfg.tie_embeddings else cfg.vocab * cfg.hidden)
    return 2 * gemm_params + 4 * cfg.hidden * ctx * cfg.layers

def kv_bytes_per_token(cfg, dtype_bytes=2):
    return 2 * cfg.layers * cfg.n_kv_heads * cfg.head_dim * dtype_bytes

# ---- 第十四、十五篇共用 ----
def quantized_weight_bytes(cfg, bits=4, group_size=128, scale_bits=16,
                           zero_bits=16, keep_embed_bf16=False):
    """weight-only 量化后的权重字节数；返回 (bytes, 等效 bit/权重)。"""
    eff_bits = bits + (scale_bits + zero_bits) / group_size
    n = param_count(cfg)["total"]
    if keep_embed_bf16:
        e = embedding_params(cfg)
        return (n - e) * eff_bits / 8 + e * 2, eff_bits
    return n * eff_bits / 8, eff_bits

def roofline_step_time(cfg, gpu, rows, weight_bytes, mfu=1.0):
    """一次前向处理 rows 个 token 行的时间下界 max(访存, 计算)，只计权重部分。"""
    t_mem = weight_bytes / gpu.bandwidth
    t_cmp = forward_flops_per_token(cfg) * rows / (gpu.bf16_flops * mfu)
    return max(t_mem, t_cmp)

def speculative_speedup(alpha, gamma, c, batch, cfg, gpu, mfu=1.0):
    """投机解码相对普通 decode 的加速比；返回 (speedup, E[tokens])。"""
    assert 0.0 <= alpha <= 1.0
    # alpha == 1 时几何级数的闭式是 0/0，极限是 gamma + 1（全部接受 + bonus）
    exp_tokens = gamma + 1 if alpha == 1.0 else (1 - alpha ** (gamma + 1)) / (1 - alpha)
    w = param_count(cfg)["total"] * 2              # BF16 目标模型
    t_base = roofline_step_time(cfg, gpu, batch, w, mfu)
    t_verify = roofline_step_time(cfg, gpu, batch * (gamma + 1), w, mfu)
    t_round = gamma * c * t_base + t_verify
    return exp_tokens * t_base / t_round, exp_tokens

LORA_TARGETS_ATTN = ("q", "k", "v", "o")
LORA_TARGETS_ALL = ("q", "k", "v", "o", "gate", "up", "down")

def lora_params(cfg, rank=16, targets=LORA_TARGETS_ALL):
    d = cfg.hidden
    q, kv = cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim
    shapes = {"q": (d, q), "k": (d, kv), "v": (d, kv), "o": (q, d),
              "gate": (d, cfg.d_ff), "up": (d, cfg.d_ff), "down": (cfg.d_ff, d)}
    per_layer = sum(rank * (din + dout)
                    for t in targets for din, dout in [shapes[t]])
    return per_layer * cfg.layers

if __name__ == "__main__":
    for cfg in (LLAMA3_8B, LLAMA3_70B):
        n = param_count(cfg)["total"]
        b4, eb = quantized_weight_bytes(cfg)
        print(f"{cfg.name}: {n/1e9:.2f}B  BF16 {n*2/1e9:.1f} GB  "
              f"INT4(g128) {eb:.2f} bit -> {b4/1e9:.2f} GB")
        print(f"  decode 下界 BF16 {n*2/H100.bandwidth*1e3:.2f} ms  "
              f"W4A16 {b4/H100.bandwidth*1e3:.2f} ms")
        print(f"  LoRA r=16 attn {lora_params(cfg, 16, LORA_TARGETS_ATTN)/1e6:.2f}M  "
              f"all {lora_params(cfg, 16)/1e6:.2f}M ({lora_params(cfg, 16)/n*100:.2f}%)")
    print("投机解码 alpha=0.8 gamma=4 c=0.1 (Llama-3-8B, H100):")
    for B in (1, 8, 32, 64, 128, 256):
        s, e = speculative_speedup(0.8, 4, 0.1, B, LLAMA3_8B, H100)
        s60, _ = speculative_speedup(0.8, 4, 0.1, B, LLAMA3_8B, H100, mfu=0.6)
        print(f"  B={B:4d}  peak {s:.2f}  60%MFU {s60:.2f}  E[tokens]={e:.2f}")
```

输出：

```text title="量化、投机解码与 LoRA 的输出"
Llama-3-8B: 8.03B  BF16 16.1 GB  INT4(g128) 4.25 bit -> 4.27 GB
  decode 下界 BF16 4.79 ms  W4A16 1.27 ms
  LoRA r=16 attn 13.63M  all 41.94M (0.52%)
Llama-3-70B: 70.55B  BF16 141.1 GB  INT4(g128) 4.25 bit -> 37.48 GB
  decode 下界 BF16 42.12 ms  W4A16 11.19 ms
  LoRA r=16 attn 65.54M  all 207.09M (0.29%)
投机解码 alpha=0.8 gamma=4 c=0.1 (Llama-3-8B, H100):
  B=   1  peak 2.40  60%MFU 2.40  E[tokens]=3.36
  B=   8  peak 2.40  60%MFU 2.40  E[tokens]=3.36
  B=  32  peak 2.40  60%MFU 2.40  E[tokens]=3.36
  B=  64  peak 2.38  60%MFU 1.61  E[tokens]=3.36
  B= 128  peak 1.39  60%MFU 0.89  E[tokens]=3.36
  B= 256  peak 0.76  60%MFU 0.62  E[tokens]=3.36
```

70B 的 BF16 decode 下界 42 ms 是"假设能放进一张卡"的数值，实际放不进；INT4 的 11.2 ms 是真的单卡数字。换 `keep_embed_bf16=True` 得到 40.6 GB / 12.1 ms；换 `zero_bits=4` 得到 4.16 bit。

### 2. 文本模型的成本表

前面各篇的数字合到一张表（H100 SXM，理论值；DeepSeek-V3 列用第六、九篇的 MLA 与 MoE 版本函数）：

|  | Llama-3-8B | Llama-3-70B | DeepSeek-V3 |
|---|---|---|---|
| 参数量 | 8.03B | 70.55B | 671B（每 token 激活 37B） |
| 权重字节 BF16 | 16.06 GB | 141 GB | 1342 GB（FP8 671 GB） |
| 权重字节 INT4（4.25 bit） | 4.27 GB | 37.5 GB | 356 GB |
| 每 token 权重 FLOPs | 15.0 GFLOPs | ~141 GFLOPs | ~74 GFLOPs |
| KV cache / token（BF16） | 128 KiB | 320 KiB | 68.6 KiB（MLA） |
| KV cache / token（FP8） | 64 KiB | 160 KiB | 34.3 KiB |
| 128K 上下文 KV cache（BF16） | 16 GiB | 40 GiB | 8.6 GiB |
| decode 下界 B=1，BF16 | 4.8 ms | 不能单卡 | 不能单卡 |
| decode 下界 B=1，W4A16 | 1.2–1.3 ms | 11.2 ms（单卡可放） | 8 卡可放 |
| 投机解码 α=0.8 γ=4 c=0.1 | 2.4×（B ≲ 60） | 2.4×（B ≲ 60） | MTP α≈0.85–0.9 γ=1 → ~1.8× |
| LoRA r=16 q/k/v/o | 13.63M（0.17%） | 65.5M（0.093%） | — |
| LoRA r=16 全部七个矩阵 | 41.9M（0.52%） | 207M（0.29%） | — |
| LoRA 训练权重侧显存 | 16.06 + 0.67 GB | 141 + 3.3 GB | — |

Table: 前面各篇数字的汇总：三个模型的权重、KV cache、decode 下界、投机解码与 LoRA

DeepSeek-V3 的投机一行按其技术报告的 MTP 接受率转述；LoRA 一行留空是因为细粒度 MoE 的专家矩阵通常不做 LoRA，attention 侧可以用同一函数算。

## 五、本文小结

两种方法各改一个变量：

| 方法 | 改的量 | 机制 | 收益与边界 |
|---|---|---|---|
| 投机解码 | 每步验证的 token 数 | 一次前向验证多个候选 | 小 batch 的 memory-bound 区间；加速取决于接受率和草稿开销 |
| LoRA | 可训练参数量 | 冻结 W，训练低秩 BA | 8B 状态约 128 GB → 16.7 GB；激活仍需保存 |
| QLoRA | 冻结底座字节数 | NF4 底座 + BF16 LoRA | 权重侧约 5 GB；反量化增加开销 |

Table: 投机解码、LoRA 与 QLoRA 的收益及边界

本篇的数字：

|  | Llama-3-8B | Llama-3-70B | DeepSeek-V3 |
|---|---|---|---|
| 投机 E[tokens]（α=0.8, γ=4） | 3.36 | 3.36 | 1.85–1.9（MTP） |
| 投机加速比（c=0.1，B ≲ 60） | 2.4× | 2.4× | ~1.8× |
| 投机转折 batch（ridge/(γ+1)） | ~60 | ~60 | — |
| LoRA r=16 参数（七个矩阵） | 41.9M（0.52%） | 207M（0.29%） | — |
| LoRA 训练权重侧显存 | 16.7 GB | 144 GB | — |
| LoRA 额外 FLOPs（W_Q） | 0.78% | 0.39% | — |

Table: 本篇的数字：投机解码与 LoRA 在三个模型上的账

到这里，文本 LLM 的成本模型已经完整：结构决定参数量、KV 与通信量，精度决定字节数，量化、投机解码与 LoRA 在不改结构的前提下改变计算形态。本篇只算了它们的账；每种方法在最小化什么、输出分布改变了多少、草稿怎么训、KV 怎么压、剪枝怎么恢复，在算法地图的 L6 系列[《高效推理与压缩（算法侧）》](/efficient-inference-and-compression-for-llms.html)里展开。第十一篇已经把图片 token 的那一行加进了同一张表；至此三段十三篇全部讲完，[《Transformer 与 LLM：系列总结与通关自测》](/transformer-and-llm-series-recap-and-self-test.html)把它们压成一张「问题 → 结论 → 必记数字」的表，并给一套通关自测。

配套代码：[`transformer-and-llm/llm_cost_07_quant_specdec_lora.py`](https://github.com/arganzheng/ai-learning-labs/blob/main/transformer-and-llm/llm_cost_07_quant_specdec_lora.py)。

## 六、自测

1. 投机解码 $$\alpha = 0.8$$、$$\gamma = 4$$、$$c = 0.1$$：一轮期望产出几个 token？加速比多少？batch 多大时失效？

   <details markdown="1"><summary>答案</summary>

   $$E = (1 - 0.8^5)/(1 - 0.8) = 3.36$$；加速 $$3.36 / (4 \times 0.1 + 1) = 2.4\times$$；验证一次前向要算 $$\gamma + 1 = 5$$ 倍的 token，等效 batch 过 ridge 的点是 $$295 / 5 \approx 60$$，之后验证不再免费、加速开始随 $$B$$ 下降，但不是立刻低于 1：要到验证时间变成基线的 3.36 倍以上才亏（第四章的表里 $$B = 64$$ 仍有 2.38 倍、$$B = 128$$ 是 1.39 倍）。

   </details>

2. 量化与投机解码为什么都是"同一条 Roofline"上的事？LoRA 为什么不是？

   <details markdown="1"><summary>答案</summary>

   两者都在兑现 memory-bound 区间里空转的算力：量化用省下的字节换时间，投机用多算的 FLOPs 换 token；工作点过了 ridge 算力不再空转，收益同时消失。LoRA 在训练侧，省的是每参数 16 字节的状态，不是算力也不是带宽。

   </details>

3. Llama-3-70B 加 $$r = 16$$ 的 LoRA（七个矩阵）：可训练参数多少、占比多少？训练权重侧显存多少？

   <details markdown="1"><summary>答案</summary>

   每层 $$16 \times \sum(\text{in} + \text{out})$$，80 层合计约 207M，占 0.29%；冻结 BF16 底座 141 GB + LoRA 状态 $$207\text{M} \times 16 = 3.3$$ GB，约 144 GB——仍要两张 80 GB 卡，QLoRA 把底座压到约 35 GB。

   </details>


[^q0]: 投机解码一次前向验证 $$\gamma + 1$$ 个 token，等于把每步的 $$m$$ 放大 $$\gamma + 1$$ 倍；batch 1 时 decode 是 memory-bound 的，多算的 FLOPs 落在空转的算力上，几乎免费，$$\alpha = 0.8$$、$$\gamma = 4$$ 时一轮期望产出 3.36 个 token、加速约 2.4 倍。batch 增大到约 $$295/(\gamma+1) \approx 60$$ 时验证本身就过了 ridge，多算的 FLOPs 开始花真时间，加速比随 batch 下降。详见[第二章第 4 节](#4-为什么验证-γ1-个-token-几乎免费以及何时不再免费)。
[^q1]: 全量训练每参数要存权重、梯度、两份 Adam 状态与 FP32 master weights，约 16 字节；LoRA 冻结底座，只有约 0.5% 的 LoRA 参数带这 16 字节，底座仍以 BF16 存 2 字节/参数，所以 8B 模型从 128 GB 降到 16.7 GB，下限就是底座本身。激活值由前向的张量形状决定，LoRA 不改前向的形状，反向仍要它们，所以一点没少。详见[第三章第 2 节](#2-训练状态从-128-gb-到-167-gb但激活值不变)。
