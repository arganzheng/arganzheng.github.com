---
layout: post
title: "Transformer 与 LLM（01）：Transformer 解剖与参数量"
subtitle: "Transformer Anatomy and Parameter Count: From config.json to 8.03B"
tags: [Transformer, LLM, AI, AI-Infra]
catalog: true
---

> 本文是[《Transformer 与 LLM：结构、算量与数值》](/transformer-and-llm-for-infra-engineers.html)系列的第 1 篇（共七篇）。下一篇：[前向的算量与访存量](/transformer-flops-bytes-and-roofline.html)

做推理系统、训练基础设施或 kernel 的工程师，迟早会被问到这样的问题：这个模型有多少参数？一张 80 GB 的卡放得下吗？某个 GEMM 的 $$m, k, n$$ 是多少？为什么 Llama 的 FFN 中间维度是 14336 这样一个看起来不整的数？这些问题的答案全部藏在一个几十行的 `config.json` 里，不需要下载权重，也不需要运行代码。

本篇要建立的是整个系列的分析对象：一个 decoder-only Transformer 里到底有哪些矩阵、每个矩阵的形状由哪个超参数决定、把它们加起来等于多少。它不讲 attention 为什么有效，也不讲训练方法，只把结构拆到能数出每一个参数的粒度。全篇的核心问题是：

> **给你任意一个模型的 `config.json`，不运行代码，能不能在五分钟内算出它的参数量，并说出这些参数在 attention、FFN、embedding 之间怎么分配？误差要在 1% 以内。**

贯穿本系列的三个模型是 **Llama-3-8B**、**Llama-3-70B** 和 **DeepSeek-V3**。本篇把前两个 dense 模型算到最后一位，对 DeepSeek-V3 只列出 config 里与前两者不同的字段，它的 attention（MLA）在第三篇、FFN（MoE）在第五篇展开。后续每一篇的 FLOPs、字节数、KV cache、通信量，都建立在本篇的形状表之上。


## 一、decoder-only Transformer 的整体结构

### 1. 四段式：embedding、L 个 layer、final norm、lm_head

一个现代 decoder-only 语言模型（GPT、Llama、Mistral、DeepSeek 的 dense 部分都属于这一类）从输入 token id 到输出 logits，只有四段：

```text
token ids  [batch, seq]
    │
    ▼
embedding  E ∈ R^{V × d}            查表：每个 id 换成一个 d 维向量
    │
    ▼  h_0 ∈ R^{batch × seq × d}
┌──────────────────────────────┐
│  layer 1                      │
│    h  = h + Attn(RMSNorm(h))  │
│    h  = h + FFN(RMSNorm(h))   │
├──────────────────────────────┤
│  layer 2  （结构完全相同）      │
├──────────────────────────────┤
│  ...                          │
├──────────────────────────────┤
│  layer L                      │
└──────────────────────────────┘
    │
    ▼  h_L ∈ R^{batch × seq × d}
final RMSNorm
    │
    ▼
lm_head    W_out ∈ R^{d × V}        投影回词表：logits ∈ R^{batch × seq × V}
```

三个关键观察：

第一，**L 个 layer 结构完全相同**，每一层的参数量一样。这意味着算参数量只要算清一层，乘以 $$L$$，再加上首尾两端。对 Llama-3-8B，$$L = 32$$；70B，$$L = 80$$；DeepSeek-V3，$$L = 61$$。

第二，**从头到尾贯穿的是一个 $$d$$ 维向量**。`hidden_size` 这个超参数（Llama-3-8B 是 4096，70B 是 8192，DeepSeek-V3 是 7168）决定了每个 token 在层与层之间传递的向量长度，所有子层都从它出发、回到它。这个不变量是后面推导所有矩阵形状的锚点。

第三，**embedding 与 lm_head 是两个形状互为转置的大矩阵**。$$E \in \mathbb{R}^{V \times d}$$ 把词表映射到 $$d$$ 维，$$W_{out} \in \mathbb{R}^{d \times V}$$ 把 $$d$$ 维映回词表。它们的大小由 `vocab_size` 决定，与层数无关，这一点会让它们在小模型里占比很高、在大模型里几乎可以忽略。

顺带说明"decoder-only"这个限定词。原始 Transformer（Vaswani 等 2017）是 encoder-decoder：encoder 用双向 attention 读源序列，decoder 除了自身的因果 attention 之外还有一个 cross-attention 子层去读 encoder 的输出，所以 decoder 每层有三个子层、三组投影矩阵。GPT 系列去掉了 encoder 和 cross-attention，只保留因果 attention 与 FFN，这就是 decoder-only。它的每层只有两个子层，结构更单一，也更容易数参数。当前所有主流生成式 LLM（Llama、Mistral、Qwen、DeepSeek、GPT）都是这一类，本系列只讨论它。"因果"（causal）指第 $$t$$ 个 token 只能看到位置 $$\le t$$ 的 token，这个约束不改变任何矩阵形状，但决定了推理可以分成 prefill 与 decode 两个阶段、以及 KV cache 为什么可以增量追加——这两点在第二、三篇展开。

### 2. 残差流：为什么所有子层的输出维度都必须是 d

每一层内部有两个子层：attention 和 FFN。它们都以同样的形式接入：

$$
h \leftarrow h + \text{SubLayer}(\text{Norm}(h))
$$

这个加法就是残差连接。它的存在决定了一个硬约束：**子层的输出维度必须等于输入维度 $$d$$**，否则加不起来。这就是为什么：

- attention 内部可以把 $$d$$ 拆成 $$n_h$$ 个 head、每个 head 只有 $$d_{head}$$ 维，但最后必须有一个 $$W_O$$ 把 $$n_h \cdot d_{head}$$ 维重新投影回 $$d$$；
- FFN 内部可以把向量放大到 $$d_{ff} = 14336$$ 维，但必须有一个 down 矩阵把它压回 $$d = 4096$$。

从系统视角看，残差流（residual stream）是一条固定宽度为 $$d$$ 的"总线"，attention 和 FFN 是挂在总线上的两种"设备"，各自从总线读一份 $$d$$ 维数据、处理、再把 $$d$$ 维结果加回总线。每一层的激活值 `[batch, seq, d]` 在 32 层之间形状不变，这也是第二篇算激活值显存时可以直接用 $$\text{batch} \times \text{seq} \times d$$ 的原因。

### 3. pre-norm 与 post-norm

Norm 放在哪里，有两种写法。原始 Transformer（Vaswani 等 2017）是 post-norm：

$$
h \leftarrow \text{Norm}(h + \text{SubLayer}(h))
$$

GPT-2 以后几乎所有大模型改用 pre-norm：

$$
h \leftarrow h + \text{SubLayer}(\text{Norm}(h))
$$

区别在于残差主干上有没有 Norm。pre-norm 的主干是一条纯加法链 $$h_L = h_0 + \sum_i \text{SubLayer}_i(\cdot)$$，梯度可以沿这条链无衰减地传到底层，训练深层网络时稳定得多；代价是主干上的数值会随层数单调增长（每层都往上加），所以最后需要一个 final norm 把它归一化再送进 lm_head。post-norm 每层都归一化一次，主干幅度受控，但深层训练需要 warmup 等技巧。

对本篇的参数量计算而言，两者没有区别：每层都是两个 Norm。对第六篇的数值分析而言，区别很大：pre-norm 的残差流在 BF16 下随深度累积的幅度增长，是低精度训练需要关心的地方。Llama、Mistral、DeepSeek 全部使用 pre-norm，本系列后续默认 pre-norm。


## 二、attention 子层的四个矩阵

### 1. 从 Q、K、V 的定义到矩阵形状

attention 子层的输入是归一化后的 $$x \in \mathbb{R}^{s \times d}$$（$$s$$ 个 token，每个 $$d$$ 维；先不管 batch）。它先做三个线性投影：

$$
Q = x W_Q, \quad K = x W_K, \quad V = x W_V
$$

然后按 head 切开，每个 head 独立算 $$\text{softmax}(Q_i K_i^\top / \sqrt{d_{head}}) V_i$$，把 $$n_h$$ 个 head 的结果拼起来，再过一个输出投影 $$W_O$$ 回到 $$d$$ 维。

这里有三个超参数：`num_attention_heads`（$$n_h$$）、`num_key_value_heads`（$$n_{kv}$$）、`head_dim`（$$d_{head}$$）。它们的关系是：

- Q 有 $$n_h$$ 个 head，每个 $$d_{head}$$ 维，所以 $$W_Q \in \mathbb{R}^{d \times n_h d_{head}}$$；
- K 和 V 有 $$n_{kv}$$ 个 head，所以 $$W_K, W_V \in \mathbb{R}^{d \times n_{kv} d_{head}}$$；
- $$W_O$$ 把 $$n_h$$ 个 head 的输出拼接后（$$n_h d_{head}$$ 维）映回 $$d$$，所以 $$W_O \in \mathbb{R}^{n_h d_{head} \times d}$$。

大多数模型满足 $$n_h \cdot d_{head} = d$$，即 head 是把 $$d$$ 均分。Llama-3-8B：$$32 \times 128 = 4096 = d$$；70B：$$64 \times 128 = 8192 = d$$。但这不是必然的，DeepSeek-V3 的 $$n_h \cdot d_{head} = 128 \times 128 = 16384 \ne 7168$$，所以正式的公式要用 $$n_h d_{head}$$ 而不是 $$d$$。

`config.json` 里通常没有 `head_dim` 字段，`transformers` 的默认是 $$d_{head} = d / n_h$$。近期模型有的显式写出 `head_dim`，读 config 时要检查。

为什么 $$d_{head}$$ 几乎总是 128（Llama 2/3、Mistral、Qwen2.5、DeepSeek-V3 的 V 均如此，少数模型用 64 或 256）？从模型侧看，$$d_{head}$$ 是每个 head 做点积的维度，太小则单个 head 的表达能力不够，太大则 head 数太少。从系统侧看，attention kernel 把一个 head 的 Q、K、V 分块装进 SM 的共享内存，$$d_{head}$$ 决定了每个 tile 的宽度：128 个 BF16 恰好是 256 字节，与 Tensor Core 的 MMA 指令形状和共享内存的 bank 宽度对齐得很好；FlashAttention 一类 kernel 对 64 和 128 做了专门调优，$$d_{head} = 256$$ 时共享内存压力翻倍、可用的分块策略变少。这是模型超参数与 kernel 实现互相迁就的一个典型例子：一旦主流 kernel 围绕 128 优化，新模型就倾向于沿用 128。

### 2. MHA、GQA、MQA 只是 n_kv 的取值不同

$$n_{kv}$$ 与 $$n_h$$ 的关系定义了三种 attention：

```text
MHA  (multi-head)          n_kv = n_h          每个 Q head 有自己的 K/V head
GQA  (grouped-query)       1 < n_kv < n_h      每 g = n_h / n_kv 个 Q head 共用一组 K/V
MQA  (multi-query)         n_kv = 1            所有 Q head 共用一组 K/V
```

Llama-3-8B 是 GQA，$$n_h = 32$$、$$n_{kv} = 8$$，每 $$g = 4$$ 个 Q head 共用一组 K/V；70B 是 $$n_h = 64$$、$$n_{kv} = 8$$，$$g = 8$$。记

$$
d_{kv} = n_{kv} \cdot d_{head}
$$

Llama-3-8B 和 70B 的 $$d_{kv}$$ 都是 $$8 \times 128 = 1024$$。

GQA 的动机不是省参数（后面会看到 $$W_K, W_V$$ 本来就不大），而是省 KV cache：推理时每个 token 要缓存的 K、V 向量长度从 $$2 \cdot n_h d_{head}$$ 缩到 $$2 \cdot n_{kv} d_{head}$$，Llama-3-8B 缩了 4 倍。第三篇会把 KV cache 的字节数公式 $$2 \cdot L \cdot n_{kv} \cdot d_{head} \cdot \text{bytes}$$ 代入得到 128 KiB/token；本篇只需要知道 $$n_{kv}$$ 决定了 $$W_K$$、$$W_V$$ 的列数。

### 3. 代入 Llama-3-8B

四个矩阵的形状与参数量：

```text
矩阵     形状 [in, out]           参数量
W_Q      [4096, 32×128=4096]      16,777,216   = 16.78M
W_K      [4096,  8×128=1024]       4,194,304   =  4.19M
W_V      [4096,  8×128=1024]       4,194,304   =  4.19M
W_O      [4096, 4096]             16,777,216   = 16.78M
────────────────────────────────────────────────────────
attention 每层                     41,943,040   = 41.94M
```

用公式写：

$$
P_{attn} = d \cdot n_h d_{head} + 2 \cdot d \cdot d_{kv} + n_h d_{head} \cdot d
$$

当 $$n_h d_{head} = d$$ 时简化为 $$P_{attn} = d(2d + 2d_{kv}) = 2d^2 + 2 d \cdot d_{kv}$$。代入 $$d = 4096$$、$$d_{kv} = 1024$$：$$2 \times 16.78\text{M} + 2 \times 4.19\text{M} = 41.94\text{M}$$。

如果是 MHA（$$n_{kv} = 32$$），attention 每层会是 $$4 d^2 = 67.1\text{M}$$。GQA 省了 25.2M/层、32 层 805M，约占总参数的 10%。但这不是它的主要收益，主要收益在 KV cache。

Llama-3-70B：$$d = 8192$$，$$d_{kv} = 1024$$：

$$
P_{attn} = 2 \times 8192^2 + 2 \times 8192 \times 1024 = 134.2\text{M} + 16.8\text{M} = 151.0\text{M}
$$

注意 $$W_Q$$ 和 $$W_O$$ 随 $$d^2$$ 增长，而 $$W_K$$、$$W_V$$ 只随 $$d \cdot d_{kv}$$ 增长；70B 把 $$d$$ 翻倍、$$n_{kv}$$ 不变，K/V 投影在 attention 里的占比从 20% 降到 11%。

### 4. attention 里没有参数的部分

$$QK^\top$$、softmax、$$PV$$ 这几步没有任何可学习参数，它们的算量与上下文长度 $$s$$ 成正比（每层每 token $$4 d s$$ FLOPs，第二篇推导），但对本篇的参数量没有贡献。RoPE 位置编码也没有参数，它只是对 Q、K 做一个由位置决定的旋转（第四篇）。这提醒我们：**参数量只衡量权重，不衡量 attention 对上下文的那部分计算**，两者在长上下文下会严重分离。


## 三、FFN 子层：从两矩阵到 SwiGLU 三矩阵

### 1. 传统 FFN

原始 Transformer 的 FFN 是两个矩阵夹一个非线性：

$$
\text{FFN}(x) = W_2 \, \sigma(W_1 x), \quad W_1 \in \mathbb{R}^{d \times d_{ff}}, \; W_2 \in \mathbb{R}^{d_{ff} \times d}
$$

$$\sigma$$ 是 ReLU 或 GELU，$$d_{ff} = 4d$$ 是从 Vaswani 等 2017 一直沿用到 GPT-3 的惯例。参数量 $$2 d \cdot d_{ff} = 8 d^2$$。

### 2. SwiGLU：gate、up、down

Shazeer 2020 提出用门控线性单元（GLU）替换 FFN 的第一层，其中 SiLU 门控的版本称为 SwiGLU，被 PaLM、Llama 系列以及之后几乎所有开源模型采用：

$$
\text{FFN}(x) = W_{down} \left[ \text{SiLU}(W_{gate} x) \odot (W_{up} x) \right]
$$

三个矩阵：$$W_{gate}, W_{up} \in \mathbb{R}^{d \times d_{ff}}$$，$$W_{down} \in \mathbb{R}^{d_{ff} \times d}$$。$$\odot$$ 是逐元素乘。参数量：

$$
P_{ffn} = 3 \cdot d \cdot d_{ff}
$$

`transformers` 里对应 `gate_proj`、`up_proj`、`down_proj` 三个 `nn.Linear`，`hidden_act` 字段为 `silu`。

### 3. 14336 是怎么来的

Llama-3-8B 的 `intermediate_size` 是 14336，不是 $$4d = 16384$$。这个数字的来历分三步。

**第一步：保持参数量不变。** SwiGLU 有三个矩阵，传统 FFN 有两个。若 $$d_{ff}$$ 仍取 $$4d$$，参数量会从 $$8d^2$$ 涨到 $$12d^2$$。Shazeer 2020 为了公平比较，把 $$d_{ff}$$ 缩到 $$\frac{2}{3}$$，使 $$3 \cdot d \cdot \frac{2}{3} \cdot 4d = 8d^2$$ 与原来相同：

$$
d_{ff} = \frac{2}{3} \cdot 4d = \frac{8}{3} d
$$

代入 $$d = 4096$$：$$\frac{8}{3} \times 4096 = 10922.67$$，取整为 10922（Llama 官方代码用 `int(2 * 4 * dim / 3)`，向下取整；手算常写作 10923，差一个不影响后续结果）。

**第二步：对齐到 `multiple_of` 的倍数。** 10922 不是一个 GPU 友好的数字，Llama 的代码把它向上取到 `multiple_of` 的整数倍。Llama 2 的 `multiple_of = 256`，于是 $$\lceil 10922 / 256 \rceil \times 256 = 43 \times 256 = 11008$$，这正是 Llama-2-7B 的 `intermediate_size`。

**第三步：Llama 3 额外乘一个 `ffn_dim_multiplier = 1.3`。** Llama 2 的 70B 和 Llama 3 全系在第二步之前多乘了一个系数 1.3，把 FFN 加宽以吸收更多参数：

$$
10922 \times 1.3 = 14198.6 \to 14198
$$

再向上对齐。Llama 3 的 `params.json` 里 `multiple_of = 1024`：

$$
\lceil 14198 / 1024 \rceil \times 1024 = 14 \times 1024 = 14336
$$

如果用 256 对齐，$$\lceil 14198 / 256 \rceil \times 256 = 56 \times 256 = 14336$$，恰好相同。所以完整的公式是：

$$
d_{ff} = \text{multiple\_of} \cdot \left\lceil \frac{\text{ffn\_dim\_multiplier} \cdot \lfloor \frac{2}{3} \cdot 4d \rfloor}{\text{multiple\_of}} \right\rceil
$$

对 70B 验证一遍：$$\lfloor \frac{2}{3} \times 32768 \rfloor = 21845$$，$$\times 1.3 = 28398$$，对齐到 1024：$$\lceil 28398 / 1024 \rceil \times 1024 = 28 \times 1024 = 28672$$。与 config 一致（注意此处若用 256 对齐会得到 28416，说明 Llama 3 确实用的是 1024）。

为什么要对齐到 256 或 1024 的倍数？Tensor Core 的 GEMM 以 8、16、64、128 的 tile 分块，$$n$$ 或 $$k$$ 不是 tile 尺寸的倍数时会有边角块浪费算力；张量并行（TP）把 $$d_{ff}$$ 切成 8 份时，每份也需要仍是 tile 尺寸的倍数。$$14336 / 8 = 1792 = 14 \times 128$$，切 8 路 TP 后每张卡上的 FFN 中间维度仍是 128 的倍数。

### 4. 代入两个模型

```text
              d       d_ff     3·d·d_ff              占每层参数
Llama-3-8B    4096    14336    176,160,768 = 176.16M    80.8%
Llama-3-70B   8192    28672    704,643,072 = 704.64M    82.3%
```

$$d_{ff} / d = 3.5$$，对两个模型都成立。因此 SwiGLU FFN 的参数量可以记成 $$3 \times 3.5 \, d^2 = 10.5 \, d^2$$，比 attention 的 $$2d^2 + 2 d \cdot d_{kv} \approx 2.5 d^2$$ 大 4 倍多。**dense 模型每一层约 80% 的参数在 FFN 里**，这是 MoE 选择把 FFN 而不是 attention 换成专家的直接原因：参数大头在这里，把它"稀疏化"收益最大（第五篇）。


## 四、Norm、bias 与 embedding

### 1. RMSNorm 与 LayerNorm

LayerNorm（Ba 等 2016）对每个 token 的 $$d$$ 维向量减均值、除标准差，再乘 $$\gamma$$ 加 $$\beta$$：

$$
\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta, \quad \gamma, \beta \in \mathbb{R}^d
$$

参数量 $$2d$$。RMSNorm（Zhang 与 Sennrich 2019）去掉了减均值和 $$\beta$$，只除以均方根：

$$
\text{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d} \sum_i x_i^2 + \epsilon}}, \quad \gamma \in \mathbb{R}^d
$$

参数量 $$d$$。Llama-3-8B 每层两个 RMSNorm 共 $$2 \times 4096 = 8192$$ 个参数，加上最后的 final norm 4096 个，全模型 $$32 \times 8192 + 4096 = 266{,}240$$ 个，占总参数的 0.003%。**参数量上 Norm 可以忽略**。

计算量上两者都是每 token $$O(d)$$ 次乘加：算一次平方和（$$d$$ 次乘加）、一次 rsqrt、$$d$$ 次缩放。与同一层里 attention 和 FFN 的 GEMM 每 token $$2 \times 218\text{M} \approx 436$$ MFLOPs 相比，Norm 每 token 只有约 $$3d \approx 12$$ KFLOPs，差五个数量级。但 Norm 是 memory-bound 的逐元素操作，它读写一遍 `[batch, seq, d]` 的激活值，在 decode 阶段的 kernel 数量和 launch 开销里占一席之地，推理引擎通常把它与相邻的残差加法融合成一个 kernel（vLLM 的 `fused_add_rms_norm` 即是）。RMSNorm 比 LayerNorm 少一次对均值的规约，在融合 kernel 里少一趟同步。

`rms_norm_eps` 字段（Llama 3 为 $$10^{-5}$$）是分母里的 $$\epsilon$$，防止除零。这个值与第六篇的数值稳定性有关：BF16 下 $$x_i^2$$ 的求和是否先转 FP32，直接影响 RMSNorm 的精度。

### 2. bias 为什么消失了

原始 Transformer 与 GPT-2 的每个 `nn.Linear` 都带 bias。Llama-3-8B 的 `config.json` 里 `attention_bias: false`、`mlp_bias: false`，所有投影矩阵都没有 bias；embedding 和 lm_head 也没有。

参数量上 bias 从来不重要：一个 $$[4096, 4096]$$ 的矩阵有 16.78M 个权重，它的 bias 只有 4096 个，占 0.02%。去掉它的原因主要有三：

- pre-norm 结构里，每个子层的输入都刚被 RMSNorm 归一化过，输入的均值信息已经被移除（或者说由 $$\gamma$$ 承载），线性层的 bias 学不到有用的偏移；
- PaLM（Chowdhery 等 2022）的报告指出去掉 bias 提升了大模型训练的稳定性；
- 对系统而言，无 bias 的 GEMM 是纯 $$Y = XW$$，少一次 broadcast add 的 epilogue，量化时也少一个需要处理的浮点向量（第七篇 INT4 权重量化只需要处理 $$W$$）。

Qwen2 系列是一个例外，它在 Q、K、V 投影上保留 bias（`attention_bias: true`），据其报告是为了改善 RoPE 的长度外推。DeepSeek-V3 与 Llama 一样无 bias。本系列的参数量公式忽略 bias。

### 3. embedding、lm_head 与 tie

embedding 矩阵 $$E \in \mathbb{R}^{V \times d}$$，lm_head $$W_{out} \in \mathbb{R}^{d \times V}$$，各 $$V \cdot d$$ 个参数。是否共享（weight tying，`tie_word_embeddings: true`）由模型决定：

- GPT-2、Gemma、Qwen2.5 的小尺寸（0.5B、1.5B、3B）共享，只算一份 $$V \cdot d$$；
- Llama 3 全系、DeepSeek-V3、Mixtral、Qwen2.5 的 7B 以上不共享，算两份。

Llama-3-8B：$$V = 128256$$，$$d = 4096$$：

$$
V \cdot d = 128256 \times 4096 = 525{,}336{,}576 = 525.3\text{M}
$$

两份合计 1.05B，占 8.03B 的 13.1%。Llama-3-70B：$$128256 \times 8192 = 1.05\text{B}$$，两份 2.10B，占 70.55B 的 3.0%。

词表大小对系统的影响有两面：

**参数量与显存。** Llama 2 的词表是 32000，Llama 3 扩到 128256（4 倍）。同样 $$d = 4096$$，embedding + lm_head 从 262M 涨到 1.05B，多出的 789M 参数在 BF16 下是 1.58 GB 显存。Llama-2-7B 到 Llama-3-8B 的"多出来的 1B"，几乎全部来自词表（另有 GQA 省下的部分抵消了一些）。

**lm_head 的算量。** lm_head 是一个 $$[m, d] \times [d, V]$$ 的 GEMM，每 token $$2 d V = 2 \times 4096 \times 128256 \approx 1.05$$ GFLOPs，占 Llama-3-8B 每 token 总 FLOPs（约 15 GFLOPs）的 7%。而 embedding 是查表，不是 GEMM，每 token 只读一行 $$d$$ 个数，FLOPs 为零。这就是为什么第二篇算每 token FLOPs 时用 $$2 \times (8.03 - 0.53)\text{B} \approx 15.0$$ GFLOPs：总参数减去 embedding 那 525M，因为它不参与乘加。

**词表越大，每个 token 编码的文本越多。** 128K 词表的 tokenizer 平均每个 token 对应的字符数比 32K 词表多约 15%（Llama 3 报告的数字），同一段文本的 token 数减少，prefill 和 decode 的总步数随之减少。对以 token 计价的推理服务而言，这是一个"隐形"的效率提升。

训练时 lm_head 输出的 logits 是 `[batch, seq, V]` 的 FP32 张量，$$V = 128256$$ 时每个 token 512 KB，8K 序列、batch 1 就是 4 GB——这是训练显存里经常被忽视的一块，也是很多框架把 lm_head 与 cross-entropy 融合、分块计算的原因。


## 五、参数量公式与三个模型

### 1. 公式

把前面几节合起来。每层：

$$
P_{layer} = \underbrace{d \cdot n_h d_{head} + 2 \cdot d \cdot d_{kv} + n_h d_{head} \cdot d}_{\text{attention}} + \underbrace{3 \cdot d \cdot d_{ff}}_{\text{FFN}} + \underbrace{2d}_{\text{RMSNorm}}
$$

当 $$n_h d_{head} = d$$ 时：

$$
P_{layer} = d(2d + 2 d_{kv}) + 3 \cdot d \cdot d_{ff} + 2d
$$

全模型：

$$
N = L \cdot P_{layer} + (1 + [\text{untied}]) \cdot V \cdot d + d
$$

其中 $$[\text{untied}]$$ 在不共享 embedding 时为 1，共享时为 0；最后的 $$d$$ 是 final norm。总纲里给的简化形式 $$N \approx L \cdot [d(d + 2d_{kv} + d) + 3 d \cdot d_{ff}] + 2 V d$$ 是省去 Norm 之后的同一个式子。

### 2. 逐项代入 Llama-3-8B

超参数：$$d = 4096$$，$$L = 32$$，$$n_h = 32$$，$$n_{kv} = 8$$，$$d_{head} = 128$$，$$d_{ff} = 14336$$，$$V = 128256$$，不共享。

```text
attention 每层
  W_Q   4096 × 4096              =    16,777,216
  W_K   4096 × 1024              =     4,194,304
  W_V   4096 × 1024              =     4,194,304
  W_O   4096 × 4096              =    16,777,216
                                 ────────────────
                                      41,943,040   (41.94M)

FFN 每层
  gate  4096 × 14336             =    58,720,256
  up    4096 × 14336             =    58,720,256
  down  14336 × 4096             =    58,720,256
                                 ────────────────
                                     176,160,768   (176.16M)

RMSNorm 每层  2 × 4096           =         8,192

每层合计                              218,112,000   (218.11M)
× 32 层                             6,979,584,000   (6.98B)

embedding   128256 × 4096        =   525,336,576   (525.3M)
lm_head     128256 × 4096        =   525,336,576   (525.3M)
final norm                       =         4,096
                                 ────────────────
总计                                8,030,261,248   (8.03B)
```

Meta 公布的 Llama-3-8B 参数量是 8.03B，与我们算出的 8,030,261,248 一致到小数点后两位。这个数字是精确的，不是估算：dense Transformer 的每一个参数都在上面的表里。

### 3. 逐项代入 Llama-3-70B

超参数：$$d = 8192$$，$$L = 80$$，$$n_h = 64$$，$$n_{kv} = 8$$，$$d_{head} = 128$$，$$d_{ff} = 28672$$，$$V = 128256$$，不共享。

```text
attention 每层   2 × 8192² + 2 × 8192 × 1024   =   150,994,944   (151.0M)
FFN 每层         3 × 8192 × 28672              =   704,643,072   (704.6M)
RMSNorm 每层     2 × 8192                      =        16,384
每层合计                                            855,654,400   (855.7M)
× 80 层                                          68,452,352,000   (68.45B)

embedding + lm_head   2 × 128256 × 8192        =  2,101,346,304   (2.10B)
final norm                                     =          8,192
总计                                             70,553,706,496   (70.55B)
```

公布值 70.6B。注意从 8B 到 70B 的放大方式：$$d$$ 翻倍（每层参数约 4 倍），$$L$$ 从 32 到 80（2.5 倍），$$n_h$$ 翻倍但 $$n_{kv}$$ 不变（GQA 组从 4 变 8），$$d_{ff}/d$$ 保持 3.5。每层 855.7M 与 218.1M 之比是 3.92，接近 4，其中差的 0.08 来自 K/V 投影没有随 $$d^2$$ 增长。

### 4. 第三次验证：Llama-3.1-405B

同一个公式再往上代一次，作为它对超大 dense 模型是否仍然精确的检验。Llama-3.1-405B 的 config：$$d = 16384$$，$$L = 126$$，$$n_h = 128$$，$$n_{kv} = 8$$，$$d_{head} = 128$$，$$d_{ff} = 53248$$，$$V = 128256$$，不共享。

```text
attention 每层   2 × 16384² + 2 × 16384 × 1024  =    570,425,344   (570.4M)
FFN 每层         3 × 16384 × 53248              =  2,617,245,696   (2.617B)
RMSNorm 每层     2 × 16384                      =         32,768
每层合计                                            3,187,703,808   (3.188B)
× 126 层                                        401,650,679,808   (401.65B)

embedding + lm_head   2 × 128256 × 16384        =  4,202,692,608   (4.20B)
final norm                                     =         16,384
总计                                            405,853,388,800   (405.85B)
```

公布值 405B。三个尺寸的模型都对上了，说明 dense Transformer 的参数量确实没有任何"隐藏"的部分——RoPE 没有参数，attention 计算没有参数，softmax 没有参数，全部权重就是这张表里的矩阵。

顺便检查 405B 的 $$d_{ff}$$：$$\lfloor \frac{2}{3} \times 65536 \rfloor = 43690$$，$$\times 1.3 = 56797$$，但 config 里是 53248 $$= 3.25 d$$，不是 3.5d。这说明 405B 没有沿用 1.3 的 multiplier（对应约 1.219），Meta 在这个尺寸上重新选了 FFN 宽度。读 config 时以实际字段为准，推导公式只是帮助理解数字从哪里来，不能代替它。

### 5. 参数分布

```text
                      Llama-3-8B                 Llama-3-70B
attention（全部层）    1.342B    16.7%            12.08B    17.1%
FFN（全部层）          5.637B    70.2%            56.37B    79.9%
Norm                  0.0003B    0.0%            0.0013B    0.0%
embedding             0.525B     6.5%            1.051B     1.5%
lm_head               0.525B     6.5%            1.051B     1.5%
────────────────────────────────────────────────────────────────
合计                  8.030B                     70.55B

层内：attention / FFN   19.2% / 80.8%             17.6% / 82.3%
```

两个规律：

**层内 FFN 占约 80%。** 这个比例由 $$d_{ff}/d = 3.5$$ 和 GQA 决定，对所有采用 SwiGLU + GQA 的 dense 模型大致相同。它意味着 dense 模型每 token 的 GEMM FLOPs 也有约 80% 花在 FFN 上（每参数 2 FLOPs），attention 的投影只占 20%——但这不包括 attention 对上下文的 $$QK^\top$$ 与 $$PV$$，那部分随 $$s$$ 增长，长上下文下会反过来成为主导。

**embedding 占比随模型变大而消失。** $$V \cdot d$$ 随 $$d$$ 线性增长，而层参数随 $$L \cdot d^2$$ 增长。8B 时 embedding + lm_head 占 13%，70B 时 3%，405B（$$d = 16384$$、$$L = 126$$）时约 1%。所以谈"小模型"的参数量时必须说清楚是否含 embedding：Qwen2.5-0.5B 的 embedding（$$151936 \times 896 = 136\text{M}$$，tie）占了总参数的 27%，"0.5B"里只有 0.36B 是层参数。

从系统视角，这张分布表直接对应显存的分布：BF16 下 Llama-3-8B 的 16.06 GB 权重里，FFN 11.3 GB、attention 2.7 GB、embedding 与 lm_head 各 1.05 GB。做张量并行时，FFN 和 attention 的权重按列/行切到各卡，embedding 通常按词表切（vocab parallel），lm_head 同样按词表切并在 cross-entropy 处做规约——切法不同是因为它们的形状不同。

这张表也决定了优化精力应该花在哪里。权重量化（第七篇）如果只量化 FFN 的三个矩阵而保留 attention 为 BF16，就已经覆盖了 70% 的字节；反过来，attention 投影的量化收益有限，很多量化方案对 `o_proj` 或 `down_proj` 单独保留更高精度，付出的显存代价不到 10%。LoRA（第七篇）默认只挂在 Q、K、V、O 四个矩阵上，覆盖的是那 17% 的参数；要覆盖 FFN 就得再挂 gate、up、down，可训练参数会翻倍。embedding 与 lm_head 在 8B 上占 13%，是 INT4 量化通常跳过的部分——跳过它们意味着 8B 模型量化后的字节数是 $$6.98\text{B} \times 0.5 + 1.05\text{B} \times 2 \approx 5.6$$ GB，相对 BF16 的压缩比不是 4 倍而是不到 3 倍，这个差异在容量规划时不能忽略。

### 6. 几种常见的算错方式

参数量公式简单，但在真实 config 上套用时容易在几个地方出错，每一个都会造成 5% 到 30% 的偏差：

**把 K/V 投影当成 MHA 算。** 忽略 `num_key_value_heads`，attention 每层从 41.9M 变成 67.1M，Llama-3-8B 总数会多出 805M（10%）。反过来，读到 `num_key_value_heads: 8` 却按 $$n_{kv} = 8$$ 去算 Q，会少算 W_Q 的四分之三。

**漏掉不共享的 lm_head。** 只算一份 $$V \cdot d$$，Llama-3-8B 会少 525M（6.5%）。判断依据是 `tie_word_embeddings` 字段，缺省时 `transformers` 视为 true，但 Llama 3 的 config 显式写了 false。反过来 Gemma 与小尺寸的 Qwen2.5 是 true，多算一份会高估 20% 以上。

**用 $$4d$$ 当 $$d_{ff}$$。** 传统 FFN 的直觉。对 Llama-3-8B，$$3 \times 4096 \times 16384 = 201\text{M}$$ 而不是 176M，每层多 14%。永远以 `intermediate_size` 为准。

**忘了 SwiGLU 是三个矩阵。** 按两矩阵算 FFN，每层少 58.7M（33%），总数少 1.9B。看 `hidden_act` 是否为 `silu`/`swiglu` 类，或者直接看 `modeling_*.py` 里 MLP 有几个 `nn.Linear`。

**把 embedding 算进 FLOPs。** 参数量上 embedding 与 lm_head 对称，都是 $$V \cdot d$$；但算量上 embedding 是查表、FLOPs 为零，lm_head 是 GEMM、每 token $$2Vd$$。把两者都乘 2 会高估每 token FLOPs 约 7%。这不是参数量的错误，而是从参数量推 FLOPs 时最常见的错误。

**默认 $$n_h \cdot d_{head} = d$$。** 对 Llama 成立，对 DeepSeek-V3（$$128 \times 128 \ne 7168$$）和一些显式给出 `head_dim` 的模型不成立。公式里用 $$n_h d_{head}$$ 而不是 $$d$$ 作为 $$W_Q$$ 的列数、$$W_O$$ 的行数，就不会错。

这些错误都可以用第九章的脚本避免：它按 config 字段逐项算，不依赖任何"通常等于"的假设。


## 六、DeepSeek-V3 的 config：形状不同在哪里

DeepSeek-V3 是本系列的第三个贯穿模型，它的 attention 和 FFN 都不是上面的形状，本篇只列出 config 里的关键字段并说明差异在哪里，推导留给第三篇（MLA）和第五篇（MoE）。

```json
{
  "hidden_size": 7168,
  "num_hidden_layers": 61,
  "num_attention_heads": 128,
  "num_key_value_heads": 128,
  "q_lora_rank": 1536,
  "kv_lora_rank": 512,
  "qk_nope_head_dim": 128,
  "qk_rope_head_dim": 64,
  "v_head_dim": 128,
  "intermediate_size": 18432,
  "moe_intermediate_size": 2048,
  "n_routed_experts": 256,
  "n_shared_experts": 1,
  "num_experts_per_tok": 8,
  "first_k_dense_replace": 3,
  "vocab_size": 129280,
  "tie_word_embeddings": false
}
```

**attention 不再是四个矩阵。** `num_key_value_heads` 等于 `num_attention_heads`，看起来像 MHA，但 MLA（Multi-head Latent Attention）用低秩分解代替了直接的 $$W_K$$、$$W_V$$：K 和 V 先被压成一个 $$d_c = 512$$ 维（`kv_lora_rank`）的潜向量，再由两个上投影矩阵展开成 128 个 head；Q 同样经过 $$1536$$ 维（`q_lora_rank`）的低秩瓶颈。每个 head 的 Q/K 维度是 $$128 + 64 = 192$$（`qk_nope_head_dim` 不带 RoPE 的部分加 `qk_rope_head_dim` 带 RoPE 的部分），V 是 128。这样每层 attention 的参数约 187M（六个矩阵：$$7168 \times 1536$$、$$1536 \times 24576$$、$$7168 \times 576$$、$$512 \times 16384$$、$$512 \times 16384$$、$$16384 \times 7168$$），61 层约 11.4B——比同等 $$d$$ 下的 MHA 少，但更重要的是 KV cache 只需存那个 512 + 64 维的潜向量。这是第三篇的主题。

**FFN 不再是一组三矩阵。** 前 3 层（`first_k_dense_replace`）是 dense SwiGLU，$$d_{ff} = 18432$$，每层 $$3 \times 7168 \times 18432 = 396\text{M}$$。第 4 到 61 层是 MoE：每层 256 个路由专家加 1 个共享专家，每个专家是一个 $$d_{ff} = 2048$$ 的小 SwiGLU（$$3 \times 7168 \times 2048 = 44.04\text{M}$$），每层 257 个专家共 11.32B，58 层共 656.5B。每个 token 只经过 top-8 路由专家加 1 个共享专家，所以**总参数 671B，每 token 激活约 37B**——参数量与算量在 MoE 里第一次分离，这是第五篇的主题。

**embedding 与 lm_head。** $$129280 \times 7168 = 926.7\text{M}$$，两份 1.85B，占 671B 的 0.3%。在 MoE 模型里 embedding 更加可以忽略。

对本篇的意义在于：**参数量公式的骨架不变**——仍然是"每层参数 × 层数 + 首尾"，只是 attention 和 FFN 那两项要换成各自的形状。第五篇会把 `llm_cost.py` 扩展到能算 MoE 的总参数与激活参数。

把三个模型放在一起看，能看到两条不同的放大路线。Llama 从 8B 到 70B 到 405B 是同一个形状按比例放大：$$d$$、$$L$$、$$n_h$$ 一起增长，$$d_{ff}/d$$ 和 $$n_{kv}$$ 基本不变，每 token 的算量与参数量同步增长。DeepSeek-V3 则是把参数量放大到 671B，但通过路由让每 token 只用其中 37B，算量停留在一个 40B 级 dense 模型的水平；代价是全部 671B 参数都必须常驻显存（FP8 下 671 GB，至少 9 张 H100 只放权重），以及专家之间的 all-to-all 通信。"参数量"这个词在 MoE 出现之后就不再单独对应成本，必须同时报总参数（决定显存）和激活参数（决定算量）——这是本系列反复强调"参数量只是成本的一个维度"的第一个具体例子。


## 七、对照 transformers 的 modeling_llama.py

`transformers` 库里 `models/llama/modeling_llama.py` 是上面所有形状的代码形式。读它的时候只需要盯住每个 `nn.Linear(in_features, out_features, bias)` 的两个维度，就能与公式一一对应。以下按类结构描述，不引用具体行号（不同版本行号会变，类结构多年稳定）。

### 1. LlamaRMSNorm

```python
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))   # gamma, d 个参数
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)       # 平方和用 FP32
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
```

只有一个 `weight`，形状 `[hidden_size]`，对应 $$\gamma \in \mathbb{R}^d$$。没有 `bias`。注意 forward 里先转 FP32 再算平方和——这是第六篇会回来讨论的数值细节。

### 2. LlamaAttention

```python
class LlamaAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.head_dim = getattr(config, "head_dim",
                                config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = (config.num_attention_heads
                                     // config.num_key_value_heads)

        self.q_proj = nn.Linear(config.hidden_size,
                                config.num_attention_heads * self.head_dim,
                                bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size,
                                config.num_key_value_heads * self.head_dim,
                                bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size,
                                config.num_key_value_heads * self.head_dim,
                                bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim,
                                config.hidden_size,
                                bias=config.attention_bias)
```

四个 `nn.Linear` 与 $$W_Q, W_K, W_V, W_O$$ 对应：

```text
q_proj  in=hidden_size (d)              out=num_attention_heads × head_dim (n_h·d_head)
k_proj  in=hidden_size (d)              out=num_key_value_heads × head_dim (n_kv·d_head)
v_proj  in=hidden_size (d)              out=num_key_value_heads × head_dim (n_kv·d_head)
o_proj  in=num_attention_heads × head_dim  out=hidden_size (d)
```

`num_key_value_groups` 就是 GQA 的 $$g = n_h / n_{kv}$$。forward 里用 `repeat_kv` 把 K、V 沿 head 维复制 $$g$$ 次以匹配 Q 的 head 数（或者交给支持 GQA 的 attention kernel 直接处理，不做物理复制）。

一个需要注意的细节：`nn.Linear` 的 `weight` 张量形状是 `[out_features, in_features]`，即 `q_proj.weight.shape == [4096, 4096]`，`k_proj.weight.shape == [1024, 4096]`。数学上写 $$x W_K$$、$$W_K \in \mathbb{R}^{d \times d_{kv}}$$，PyTorch 存的是它的转置。参数量不受影响，但读权重文件（safetensors 的 shape 字段）时要记得这一点。

### 3. LlamaMLP

```python
class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size,
                                   bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size,
                                 bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size,
                                   bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]      # "silu"

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

forward 的一行就是 $$W_{down}[\text{SiLU}(W_{gate} x) \odot (W_{up} x)]$$。推理引擎通常把 `gate_proj` 与 `up_proj` 合并成一个 `[d, 2 d_{ff}]` 的矩阵做一次 GEMM（vLLM 的 `MergedColumnParallelLinear`），再用一个融合 kernel 做 SiLU 与逐元素乘——参数量不变，GEMM 次数从 3 减到 2。同理 `q_proj`、`k_proj`、`v_proj` 也常合并成一个 `[d, (n_h + 2 n_{kv}) d_{head}]` 的 QKV 矩阵。

### 4. LlamaDecoderLayer 与 LlamaModel、LlamaForCausalLM

```python
class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = LlamaAttention(config, layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, ...):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, ...)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
```

这是第一章的 pre-norm 结构逐字翻译：两个 RMSNorm（`input_layernorm`、`post_attention_layernorm`，名字里的 "layernorm" 是历史遗留，实际是 RMSNorm），两条残差。

```python
class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config):
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size,
                                         config.pad_token_id)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)   # 无参数

class LlamaForCausalLM(LlamaPreTrainedModel):
    def __init__(self, config):
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
```

`embed_tokens.weight` 形状 `[vocab_size, hidden_size]`，`lm_head.weight` 形状 `[vocab_size, hidden_size]`（同样是 `[out, in]`）。`tie_word_embeddings` 为 true 时两者指向同一个张量。`rotary_emb` 没有可学习参数，它在初始化时算好一张 $$\cos / \sin$$ 表作为 buffer。

用 `transformers` 验证参数量只需要：

```python
from transformers import AutoConfig, AutoModelForCausalLM
cfg = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-8B")
with torch.device("meta"):                       # 只建图不分配内存
    model = AutoModelForCausalLM.from_config(cfg)
print(sum(p.numel() for p in model.parameters()))   # 8030261248
```

在 `meta` 设备上构造模型不占显存，几秒钟就能验证任意 config 的参数总量。


## 八、shape 追踪：[batch, seq, hidden] 到 GEMM 的 m、k、n

参数量决定显存，但决定算量和 kernel 行为的是每个矩阵乘的具体形状。这一节追踪一个 `[batch, seq, hidden]` 的激活张量在一层里经过的每一个 GEMM。

### 1. GEMM 的记法

$$[m, k] \times [k, n] \to [m, n]$$，FLOPs 为 $$2mkn$$（每个输出元素 $$k$$ 次乘加）。在 Transformer 里：

- $$m$$ 永远是 **token 数**：`batch × seq` 展平后的行数；
- $$k$$ 是权重的输入维度，$$n$$ 是权重的输出维度；
- 权重 $$[k, n]$$ 与 $$m$$ 无关，这就是"每参数每 token 2 FLOPs"的来源：$$2mkn / (kn) / m = 2$$。

激活张量 `[B, S, d]` 进 `nn.Linear` 前会被视作 `[B·S, d]`，$$m = B \cdot S$$。

### 2. 一层里的全部 GEMM（Llama-3-8B）

设 $$T = B \cdot S$$ 为本次前向的 token 总数。

```text
                    m      k        n         权重参数     FLOPs / token
q_proj              T      4096     4096      16.78M      33.6M
k_proj              T      4096     1024       4.19M       8.4M
v_proj              T      4096     1024       4.19M       8.4M
  (QKV 合并)        T      4096     6144      25.17M      50.3M
attention 计算      ——  按 head 的 batched matmul，与 S 有关，见下
o_proj              T      4096     4096      16.78M      33.6M
gate_proj           T      4096    14336      58.72M     117.4M
up_proj             T      4096    14336      58.72M     117.4M
  (gate/up 合并)    T      4096    28672     117.44M     234.9M
down_proj           T     14336     4096      58.72M     117.4M
────────────────────────────────────────────────────────────────
每层权重 GEMM 合计                             218.1M     436.2M

lm_head（全模型一次）T     4096   128256     525.34M    1050.7M
```

"FLOPs / token" 一列正是"参数量 × 2"。32 层加 lm_head：$$32 \times 436.2\text{M} + 1050.7\text{M} \approx 15.0$$ GFLOPs/token，与第二篇的数字一致。

attention 计算那一行不是权重 GEMM。以 Q 为 `[B, n_h, S, d_head]`、K 为 `[B, n_kv, S, d_head]`（经 GQA 广播到 $$n_h$$）为例：

```text
QK^T   每个 (batch, head)：[S, 128] × [128, S] → [S, S]     2·S²·128 FLOPs
PV     每个 (batch, head)：[S, S] × [S, 128] → [S, 128]     2·S²·128 FLOPs
```

32 个 head 合计每层 $$4 \cdot S^2 \cdot d$$，平均到每个 token 是 $$4 d S$$——随上下文长度线性增长，与权重无关。这是参数量公式看不见的那一项，第二篇会算它在 8K 和 128K 上下文下分别占多少。

### 3. prefill 与 decode 的 m 不同

同一个权重矩阵，在推理的两个阶段面对的 $$m$$ 差别巨大：

```text
阶段        输入                  m = B·S                 例（B=1, S=8192）
prefill     [B, S, d]             B × S（成千上万）        m = 8192
decode      [B, 1, d]             B（batch 大小）          m = 1
训练        [B, S, d]             B × S                    m = 8 × 8192 = 65536
```

训练的 $$m$$ 最大，而且每个前向 GEMM 在反向传播中对应两个同样大小的 GEMM：一个算对输入的梯度 $$\partial X = \partial Y \, W^\top$$（形状 $$[m, n] \times [n, k]$$），一个算对权重的梯度 $$\partial W = X^\top \partial Y$$（形状 $$[k, m] \times [m, n]$$）。三个 GEMM 的 FLOPs 相同，所以训练每 token 的算量是前向的 3 倍，即每参数 6 FLOPs，这就是训练总算量 $$6ND$$（$$D$$ 为训练 token 数）里的 6 的来源，第二篇会完整推导。注意 $$\partial W$$ 那个 GEMM 的规约维度是 $$m$$ 而不是 $$k$$——它把所有 token 的贡献加起来，这是数据并行中梯度可以按 token 切分再 all-reduce 的数学基础，也是 $$m$$ 很大时累加误差问题（第六篇）最先出现的地方。

prefill 的 $$m$$ 是整个 prompt 的 token 数，GEMM 是"胖"的，$$m$$ 与 $$k, n$$ 同一量级，Tensor Core 能吃满。decode 每步每个请求只有一个新 token，$$m = B$$，batch 1 时 GEMM 退化成 GEMV（矩阵乘向量）：为了做 $$2 \times 4096 \times 4096 \approx 33.6$$M 次 FLOPs，要把 16.78M 个 BF16 权重（33.6 MB）从 HBM 搬进片上，每读一个字节只做 1 次 FLOP。H100 的算力与带宽之比是 $$989 / 3.35 \approx 295$$ FLOP/byte，decode 在 $$B = 1$$ 时距这个 ridge point 两个数量级——这是 decode memory-bound 的根源，也是 batching、量化、投机解码三条优化路线的共同出发点（第二、七篇）。

### 4. 张量并行如何切这些形状

上表的 $$k, n$$ 也是张量并行切分的依据。以 8 路 TP 为例，Megatron 的切法是：

- `q/k/v_proj` 按 $$n$$（列）切，每卡 $$n = 6144 / 8 = 768$$，即每卡 4 个 Q head、1 个 KV head——注意 $$n_{kv} = 8$$ 恰好允许 8 路 TP 每卡一个 KV head，超过 8 路就必须复制 KV head；
- `o_proj` 按 $$k$$（行）切，每卡 $$k = 512$$，输出后 all-reduce；
- `gate/up_proj` 按 $$n$$ 切，每卡 $$n = 28672 / 8 = 3584$$；
- `down_proj` 按 $$k$$ 切，每卡 $$k = 1792$$，输出后 all-reduce。

第三章说 $$d_{ff}$$ 对齐到 1024 的倍数，在这里体现为切 8 路后 $$1792 = 14 \times 128$$ 仍是 Tensor Core tile 的倍数。Llama-3-70B 的 $$n_{kv} = 8$$ 同样是为 8 卡 TP 准备的。


## 九、实践：llm_cost.py 第一版

本系列的贯穿脚本 `llm_cost.py` 从本篇开始，每篇增加几个函数。第一版只做一件事：从超参数算出逐组件参数量并打印表格。完整可运行代码如下。

```python
"""llm_cost.py -- 第一版：从超参数算出参数量。

用法：
    python llm_cost.py                # 打印内置模型的参数表
    python llm_cost.py config.json    # 读 transformers 风格的 config.json
"""
import json
import sys
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


def param_count(cfg: ModelConfig) -> dict:
    """返回逐组件参数量（单位：个）。键的顺序即打印顺序。"""
    d, L = cfg.hidden, cfg.layers
    d_q = cfg.n_heads * cfg.head_dim          # W_Q 的输出维度，通常等于 d
    d_kv = cfg.n_kv_heads * cfg.head_dim      # W_K / W_V 的输出维度

    w_q = d * d_q
    w_k = d * d_kv
    w_v = d * d_kv
    w_o = d_q * d
    attn = w_q + w_k + w_v + w_o

    ffn = 3 * d * cfg.d_ff                    # gate + up + down
    norms = 2 * d                             # 两个 RMSNorm 的 gamma
    per_layer = attn + ffn + norms

    embed = cfg.vocab * d
    lm_head = 0 if cfg.tie_embeddings else cfg.vocab * d
    final_norm = d

    total = L * per_layer + embed + lm_head + final_norm
    return {
        "W_Q": w_q, "W_K": w_k, "W_V": w_v, "W_O": w_o,
        "attention/layer": attn,
        "FFN/layer": ffn,
        "norms/layer": norms,
        "per_layer": per_layer,
        "all_layers": L * per_layer,
        "embedding": embed,
        "lm_head": lm_head,
        "final_norm": final_norm,
        "total": total,
    }


def fmt(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.3f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    return f"{n:,}"


def print_table(cfg: ModelConfig) -> None:
    p = param_count(cfg)
    total = p["total"]
    print(f"== {cfg.name}: d={cfg.hidden} L={cfg.layers} "
          f"n_h={cfg.n_heads} n_kv={cfg.n_kv_heads} d_head={cfg.head_dim} "
          f"d_ff={cfg.d_ff} V={cfg.vocab}")
    print(f"{'component':<18}{'params':>14}{'exact':>18}{'share':>9}")
    for k, v in p.items():
        share = "" if k == "total" else f"{100 * v / total:6.2f}%"
        if k in ("W_Q", "W_K", "W_V", "W_O"):
            share = ""  # 单个投影矩阵不算全局占比，避免表格噪音
        print(f"{k:<18}{fmt(v):>14}{v:>18,}{share:>9}")
    print()


def from_config_json(path: str) -> ModelConfig:
    with open(path) as f:
        c = json.load(f)
    n_heads = c["num_attention_heads"]
    return ModelConfig(
        name=path,
        hidden=c["hidden_size"],
        layers=c["num_hidden_layers"],
        n_heads=n_heads,
        n_kv_heads=c.get("num_key_value_heads", n_heads),
        head_dim=c.get("head_dim", c["hidden_size"] // n_heads),
        d_ff=c["intermediate_size"],
        vocab=c["vocab_size"],
        tie_embeddings=c.get("tie_word_embeddings", False),
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print_table(from_config_json(sys.argv[1]))
    else:
        for cfg in (LLAMA3_8B, LLAMA3_70B):
            print_table(cfg)
```

几点设计说明：

- `ModelConfig` 的字段与 `config.json` 一一对应，`from_config_json` 负责翻译字段名并处理缺省（没有 `num_key_value_heads` 视为 MHA，没有 `head_dim` 用 $$d / n_h$$）。后面几篇会给它加 `mla_rank`、`n_experts` 等字段，dense 模型这些字段保持默认值即可；
- `param_count` 返回字典而不是单个数字，因为第二篇算 FLOPs、第三篇算 KV cache、第七篇算 LoRA 参数都需要按组件取值；
- `GPU` 结构本篇用不到，先按系列约定放进来，第二篇的 `decode_step_time(cfg, gpu, batch, ctx)` 会用。

运行 `python llm_cost.py` 的输出：

```text
== Llama-3-8B: d=4096 L=32 n_h=32 n_kv=8 d_head=128 d_ff=14336 V=128256
component                 params             exact    share
W_Q                       16.78M        16,777,216
W_K                        4.19M         4,194,304
W_V                        4.19M         4,194,304
W_O                       16.78M        16,777,216
attention/layer           41.94M        41,943,040    0.52%
FFN/layer                176.16M       176,160,768    2.19%
norms/layer                8,192             8,192    0.00%
per_layer                218.11M       218,112,000    2.72%
all_layers                6.980B     6,979,584,000   86.92%
embedding                525.34M       525,336,576    6.54%
lm_head                  525.34M       525,336,576    6.54%
final_norm                 4,096             4,096    0.00%
total                     8.030B     8,030,261,248

== Llama-3-70B: d=8192 L=80 n_h=64 n_kv=8 d_head=128 d_ff=28672 V=128256
component                 params             exact    share
W_Q                       67.11M        67,108,864
W_K                        8.39M         8,388,608
W_V                        8.39M         8,388,608
W_O                       67.11M        67,108,864
attention/layer          150.99M       150,994,944    0.21%
FFN/layer                704.64M       704,643,072    1.00%
norms/layer               16,384            16,384    0.00%
per_layer                855.65M       855,654,400    1.21%
all_layers               68.452B    68,452,352,000   97.02%
embedding                 1.051B     1,050,673,152    1.49%
lm_head                   1.051B     1,050,673,152    1.49%
final_norm                 8,192             8,192    0.00%
total                    70.554B    70,553,706,496
```

两个总数与 Meta 公布的 8.03B、70.6B 一致。用 Llama-3-8B 真实的 `config.json` 跑一遍也是同样的结果，下面是它的关键字段（省略了 token id、dtype 等与结构无关的项）：

```json
{
  "architectures": ["LlamaForCausalLM"],
  "attention_bias": false,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "intermediate_size": 14336,
  "max_position_embeddings": 8192,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-05,
  "rope_theta": 500000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "vocab_size": 128256
}
```

八个字段决定了全部 8,030,261,248 个参数：`hidden_size`、`intermediate_size`、`num_hidden_layers`、`num_attention_heads`、`num_key_value_heads`、`vocab_size`、`tie_word_embeddings`，以及隐含的 `head_dim = 4096 / 32`。`rope_theta = 500000` 是第四篇的主角，`max_position_embeddings = 8192` 是它的训练上下文长度，`torch_dtype` 告诉我们权重以 BF16 存储、每参数 2 字节。

可以试着把其他模型的 `config.json` 喂给脚本：Mistral-7B（$$d = 4096$$、$$L = 32$$、$$n_{kv} = 8$$、$$d_{ff} = 14336$$、$$V = 32000$$）会得到 7.24B，与 Llama-3-8B 的差恰好是词表从 32000 到 128256 多出的 $$2 \times 96256 \times 4096 = 789\text{M}$$；Qwen2.5-7B（$$d = 3584$$、$$L = 28$$、$$n_h = 28$$、$$n_{kv} = 4$$、$$d_{ff} = 18944$$、$$V = 152064$$）会得到 7.6B 左右，与公布的 7.61B 一致（它的 Q/K/V 有 bias，差的几十万个参数在脚本的忽略范围内）。DeepSeek-V3 的 config 喂进去会得到错误的结果，因为它的 attention 与 FFN 不是这个形状——那是第三篇和第五篇要扩展的。


## 十、小结

本篇把一个 decoder-only Transformer 拆到了每一个矩阵：

- 整体是 embedding → $$L$$ 个相同的 layer → final RMSNorm → lm_head，每层两个子层（attention、FFN）以 pre-norm 残差接入，残差流固定宽度 $$d$$ 决定了所有子层的输出维度；
- attention 四个矩阵 $$W_Q \in \mathbb{R}^{d \times n_h d_{head}}$$、$$W_K, W_V \in \mathbb{R}^{d \times n_{kv} d_{head}}$$、$$W_O \in \mathbb{R}^{n_h d_{head} \times d}$$，GQA 通过 $$n_{kv} < n_h$$ 缩小 K/V 投影和 KV cache；
- SwiGLU FFN 三个矩阵 $$3 \cdot d \cdot d_{ff}$$，$$d_{ff} = 14336$$ 来自 $$\frac{2}{3} \cdot 4d \times 1.3$$ 向上对齐到 1024 的倍数；
- RMSNorm 每个 $$d$$ 个参数，bias 已消失，两者对参数量都可忽略；embedding 与 lm_head 各 $$V \cdot d$$，Llama 3 不共享；
- 参数量公式 $$N = L[d(2d + 2d_{kv}) + 3 d \cdot d_{ff} + 2d] + 2Vd + d$$，代入得 Llama-3-8B 精确到 8,030,261,248，Llama-3-70B 到 70,553,706,496；
- 层内约 80% 参数在 FFN，embedding 在 8B 占 13%、70B 占 3%；
- 每个 `nn.Linear` 在前向中是一个 $$m = B \cdot S$$ 的 GEMM，prefill 时 $$m$$ 是 prompt 长度，decode 时 $$m$$ 是 batch 大小。

本篇算出的数字：

```text
                          Llama-3-8B        Llama-3-70B       DeepSeek-V3
hidden d                  4096              8192              7168
layers L                  32                80                61（3 dense + 58 MoE）
n_h / n_kv / d_head       32 / 8 / 128      64 / 8 / 128      128 / MLA / 192（v 128）
d_ff                      14336             28672             18432 dense / 2048 专家
vocab                     128256            128256            129280

attention 每层            41.94M            151.0M            约 187M（MLA，第三篇）
FFN 每层                  176.16M           704.6M            396M dense / 11.32B MoE（第五篇）
每层合计                  218.1M            855.7M            —
所有层                    6.98B             68.45B            约 668B
embedding + lm_head       1.05B（13.1%）    2.10B（3.0%）     1.85B（0.3%）
总参数                    8.03B             70.55B            约 671B
每 token 激活参数         8.03B             70.55B            约 37B（第五篇推导）
层内 FFN 占比             80.8%             82.3%             —

BF16 权重字节数           16.06 GB          141.1 GB          1342 GB（FP8 671 GB）
每 token 权重 GEMM FLOPs  约 15.0 G         约 141 G          约 74 G（第二篇推导）
```

最后一行用到的关系是"每参数每 token 2 FLOPs，embedding 查表不计"，即 $$2 \times (8.03 - 0.53)\text{B} \approx 15.0$$ GFLOPs。这是下一篇的起点：有了每个矩阵的形状，就能算每个 GEMM 的 FLOPs 和要搬多少字节，把 prefill 与 decode 放到 Roofline 上，回答"一张 H100 跑 Llama-3-8B，decode 一个 token 最快多少毫秒"。


## 下一篇

[前向的算量与访存量：prefill、decode 与 Roofline](/transformer-flops-bytes-and-roofline.html)
