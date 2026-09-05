---
layout: post
title: "Transformer 与 LLM（03）：Attention 变体与 KV cache"
subtitle: "Attention Variants and the KV Cache: Deriving MHA, GQA, MQA and MLA"
tags: [Transformer, LLM, AI, AI-Infra]
catalog: true
---

> 本文是[《Transformer 与 LLM：结构、算量与数值》](/transformer-and-llm-for-infra-engineers.html)系列的第三篇（共七篇）。上一篇：[前向的算量与访存量](/transformer-flops-bytes-and-roofline.html)　下一篇：[位置编码与长上下文](/positional-encoding-and-long-context.html)

上一篇把一次前向拆成了"权重项"和"上下文项"两部分：权重项每 token 每参数 2 FLOPs，与上下文长度无关；上下文项只来自 attention，随序列长度 $$s$$ 线性增长（decode）或平方增长（prefill）。这一篇专门讲 attention，因为它是 Transformer 里唯一成本随上下文增长的部分，也是过去几年结构改动最集中的地方。

MHA、MQA、GQA、MLA 四种结构，做的是同一件事的不同取舍：

> **减少每个 token 的 KV cache 字节数，同时尽量不损失质量。**

顺着这条线，本篇要回答总纲里提出的问题：

> **DeepSeek-V3 有 128 个 attention head、61 层，KV cache 却比 32 头 32 层的 Llama-3-8B 小。这是怎么做到的？代价是什么？**

本篇沿用全系列的三个模型：Llama-3-8B（$$d = 4096$$，32 层，32 头 / 8 个 KV 头，$$d_{head} = 128$$）、Llama-3-70B（$$d = 8192$$，80 层，64 头 / 8 个 KV 头）、DeepSeek-V3（$$d = 7168$$，61 层，128 头，MLA）。硬件以 H100 SXM 为默认：80 GB HBM3，3.35 TB/s，BF16 dense 989 TFLOPS，ridge point 约 295 FLOP/byte。所有数字都是理论下界或估算，不是实测。


## 一、为什么需要 KV cache

### 1. attention 公式与 decode 的形态

对一层里的一个 head，attention 是：

$$
\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_{head}}}\right) V
$$

其中 $$Q, K, V$$ 分别由输入 $$X \in \mathbb{R}^{s \times d}$$ 经 $$W_Q, W_K, W_V$$ 投影得到。$$K$$ 和 $$V$$ 的每一行只依赖对应位置那个 token 的输入，与后面的 token 无关——这个性质是 KV cache 存在的全部前提。

上一篇的结论用一句话重述：decode 阶段每一步只处理一个新 token，权重 GEMM 的形状是 $$[B, d] \times [d, n]$$，算术强度约等于 batch 大小 $$B$$ FLOP/byte（BF16 权重每读 2 字节做 $$2B$$ FLOPs），而 H100 的 ridge point 是 $$989 / 3.35 \approx 295$$ FLOP/byte。$$B = 1$$ 时距 ridge 两个数量级，decode 是彻底的 memory-bound：Llama-3-8B 的 16.06 GB BF16 权重从 HBM 读一遍至少 4.8 ms，单请求上限约 208 token/s。

这一篇要加的是 decode 时的第二项 HBM 流量：KV cache。

### 2. 不缓存的代价

自回归生成第 $$t$$ 个 token 时，它的 query 要与前面全部 $$t$$ 个位置的 key 做点积，再对 $$t$$ 个 value 加权求和。这 $$t$$ 个位置的 $$K$$、$$V$$ 在生成第 $$t-1$$ 个 token 时已经算过一遍，而且**数值完全不变**（causal 模型里，前面的 token 看不到后面的 token）。

如果不缓存，生成第 $$t$$ 个 token 时就得把前 $$t$$ 个 token 重新做一次完整前向：

- 权重项：$$t$$ 个 token × 每 token $$2N$$ FLOPs，Llama-3-8B 是 $$15.0 \text{ GFLOPs} \times t$$；
- attention 项：$$QK^\top$$ 与 $$PV$$ 各 $$2 \cdot n_h \cdot d_{head} \cdot t^2$$，即每层 $$4 d t^2$$，随 $$t$$ 平方增长。

也就是说不缓存时**每一步都是一次 $$t$$ 个 token 的 prefill**，单步 $$O(t^2)$$，生成 $$s$$ 个 token 累计 $$O(s^3)$$。缓存之后，每一步只算当前 token 的投影（权重项 $$2N$$ FLOPs），attention 项退化为 $$4 d t$$ 每层——单步 $$O(t)$$，累计 $$O(s^2)$$。

用 Llama-3-8B 在 $$t = 4096$$ 处生成一个 token 感受一下差距。不缓存：权重项 $$15.0 \times 4096 \approx 61$$ TFLOP，attention 项 $$0.524 \text{ M} \times 4096^2 \approx 8.8$$ TFLOP，合计约 70 TFLOP，H100 上按 60% MFU 约 0.12 秒——**一个 token**。缓存：权重项 15 GFLOPs，attention 项 $$0.524 \text{ M} \times 4096 \approx 2.1$$ GFLOPs，合计 17 GFLOPs，且瓶颈在读权重与读 KV 的几毫秒上。两者差了四个数量级。所以"要不要 KV cache"从来不是一个选项，问题只是它要占多少显存、怎么让它占得更少。

代价是显存：$$K$$、$$V$$ 每层每 token 都要留在 HBM 里，并且每一步都要被读一遍。

### 3. KV cache 的大小公式

每 token 的 KV cache 字节数：

$$
\text{bytes/token} = 2 \cdot L \cdot n_{kv} \cdot d_{head} \cdot \text{bytes/elem}
$$

因子 2 是 K 与 V 两份；$$L$$ 是层数；$$n_{kv}$$ 是 KV head 数（MHA 下等于 $$n_h$$）；$$d_{head}$$ 是每个 head 的维度。代入三个模型（BF16，2 字节）：

- Llama-3-8B 若为 MHA（$$n_{kv} = 32$$）：$$2 \times 32 \times 32 \times 128 \times 2 = 524288$$ B = **512 KiB**；
- Llama-3-8B 实际是 GQA（$$n_{kv} = 8$$）：$$2 \times 32 \times 8 \times 128 \times 2 = 131072$$ B = **128 KiB**；
- Llama-3-70B（$$L = 80$$，$$n_{kv} = 8$$）：$$2 \times 80 \times 8 \times 128 \times 2 = 327680$$ B = **320 KiB**。

乘上上下文长度就是一条序列的 KV cache。128K（131072）上下文：

- Llama-3-8B：$$128 \text{ KiB} \times 131072 = 16 \text{ GiB}$$；
- Llama-3-70B：$$320 \text{ KiB} \times 131072 = 40 \text{ GiB}$$。

一条 128K 的序列，就要吃掉 8B 模型权重（16 GB）同等的显存。

反过来看容量：一张 80 GB 的 H100 放下 Llama-3-8B 的 16.06 GB 权重后剩约 64 GB，全部给 KV cache：

$$
\frac{64 \text{ GiB}}{128 \text{ KiB}} = \frac{2^{36}}{2^{17}} = 2^{19} = 524288
$$

约 52 万个 token。这个数字可以理解为"64 条 8K 序列"或"4 条 128K 序列"（不考虑激活和碎片）。推理系统里 KV cache 是显存的主要消耗者，不是权重。

### 4. KV cache 是 decode 的第二项流量

decode 每一步，每条序列都要把自己全部历史的 K、V 读一遍。上下文 8K、batch 64 的 Llama-3-8B：

$$
128 \text{ KiB} \times 8192 \times 64 = 64 \text{ GiB}
$$

每步读 64 GiB 的 KV，是权重 16 GB 的 4 倍。3.35 TB/s 下仅 KV 读取就要约 20 ms，此时权重读取的 4.8 ms 已经不是主项。这就是为什么 batch 放大以后 decode 的瓶颈从"读权重"转到"读 KV"，也是为什么 KV cache 的每 token 字节数直接决定了一台机器能服务多少并发、每步能跑多快。

后面所有变体，都是围着这个公式里的 $$n_{kv} \cdot d_{head}$$ 这一项做减法。


## 二、MQA 与 GQA：直接减少 KV head

### 1. MQA：所有 query head 共享一组 K、V

Multi-Query Attention（Shazeer 2019）最直接：保留 $$n_h$$ 个 query head，但 K 和 V 各只投影一份，所有 head 共用。$$n_{kv} = 1$$，KV cache 缩小 $$n_h$$ 倍——Llama-3-8B 若用 MQA，每 token 只需 $$2 \times 32 \times 1 \times 128 \times 2 = 16$$ KiB。

参数量也随之减少：$$W_K$$、$$W_V$$ 从 $$d \times n_h d_{head}$$ 变为 $$d \times d_{head}$$，Llama-3-8B 一层里 K、V 投影从 2 × 16.78M 缩到 2 × 0.52M。但代价是质量：所有 head 只能看同一组 key/value，表达能力有损，PaLM 与 Falcon 等模型采用过它，后来的开源模型基本都退了一步用 GQA。

### 2. GQA：分组共享

Grouped-Query Attention（Ainslie 等 2023）取中间值：把 $$n_h$$ 个 query head 分成 $$n_{kv}$$ 组，每组 $$g = n_h / n_{kv}$$ 个 query head 共享一组 K、V。

- Llama-3-8B：$$n_h = 32$$，$$n_{kv} = 8$$，$$g = 4$$；
- Llama-3-70B：$$n_h = 64$$，$$n_{kv} = 8$$，$$g = 8$$。

$$g = 1$$ 是 MHA，$$g = n_h$$ 是 MQA。Llama-3-8B 的 128 KiB/token、70B 的 320 KiB/token，就是 GQA 下的数字。参数量上 Llama-3-8B 每层 $$W_K$$、$$W_V$$ 各 $$4096 \times 8 \times 128 = 4.19$$M，与 $$W_Q$$、$$W_O$$ 的 16.78M 相比只剩四分之一——上一篇算出的每层 attention 41.94M，已经包含了这一节省。

Ainslie 等 2023 的另一个贡献是**从已有 MHA checkpoint 转换**：把同一组内 $$g$$ 个 head 的 $$W_K$$（以及 $$W_V$$）做均值池化，

$$
W_K^{(\text{group } j)} = \frac{1}{g} \sum_{h \in \text{group } j} W_K^{(h)}
$$

得到一组 K、V 投影，然后用原预训练算量的一小部分（论文用 5%）继续训练（uptraining）恢复质量。这意味着 GQA 不必从头训，也说明 K、V 投影在不同 head 之间的冗余度确实很高——否则均值池化不会是一个好的初始化。

### 3. GQA 对 decode attention 算术强度的影响

GQA 除了省显存，还有一个常被忽略的收益：**提高 decode 时 attention 对 KV cache 的算术强度**。

decode 时，一条序列一个 query token 对上下文长度 $$s$$ 做 attention。看 KV cache 中的**一个元素**——比如某个位置 $$j$$、某个 KV head、某一维 $$i$$ 上的 key 值 $$k_{j,i}$$——它被读入片上之后被哪些计算使用：

- 在 $$QK^\top$$ 里，与每一个共享它的 query head $$h$$ 的 $$q_{h,i}$$ 做一次乘加：2 FLOPs；
- 对应的 value 元素 $$v_{j,i}$$ 在 $$PV$$ 里，与每一个共享它的 head 的权重 $$p_{h,j}$$ 做一次乘加：2 FLOPs。

每个 KV 元素 BF16 占 2 字节，服务 $$g$$ 个 query head，每个 head 消耗它 2 FLOPs：

$$
\text{Intensity} = \frac{2 g \text{ FLOPs}}{2 \text{ bytes}} = g \text{ FLOP/byte}
$$

MHA 下 $$g = 1$$，约 1 FLOP/byte；Llama-3-8B 的 GQA 提到 4，70B 提到 8。注意 batch 对这个数字**没有帮助**——每条序列有自己的 KV cache，多一条序列就多读一份。所以 decode 时 attention 部分的算术强度只由结构决定，与 batch 无关；它比权重 GEMM 的 $$B$$ FLOP/byte 还要低（batch 64 时权重强度 64，KV 强度仍是 4），是 decode 里最"访存"的一段。

GQA 的 $$g$$ 把这个数字从 1 提到 4 或 8，距 ridge point 295 仍差得远，但 KV 读取的**总字节数**同时缩小了 $$g$$ 倍，两者叠加，decode 的 KV 部分时间直接缩小 $$g$$ 倍。这是 GQA 的第二个收益。

把这个推导记住：**每读一个 KV 元素，服务多少个 query head，算术强度就是多少**。MLA 会把这个数字推到 128。

### 4. GQA 与张量并行的边界

$$n_{kv}$$ 还悄悄决定了另一件事：tensor parallel 的切分粒度。attention 按 head 切分时，每张卡拿到 $$n_h / \text{TP}$$ 个 query head 和 $$n_{kv} / \text{TP}$$ 个 KV head。Llama-3-70B 的 8 个 KV 头意味着 TP = 8 时每卡恰好一个 KV 头、8 个 query head，KV cache 也恰好按卡均分——每卡每 token $$320 / 8 = 40$$ KiB。

TP 一旦超过 $$n_{kv}$$（比如 70B 用 TP = 16），KV 头就不够分了，只能**复制**：两张卡持有同一个 KV 头的副本，各算 4 个 query head。此时每卡的 KV cache 不再随 TP 缩小，总的 KV 显存变成 $$\text{TP} / n_{kv}$$ 倍。这是为什么 8 个 KV 头的模型在 8 卡以上的 TP 收益递减，也是 Megatron 与 vLLM 里 `num_kv_heads` 与 TP 度之间要满足整除或复制关系的原因。MLA 只有一个（latent）KV 头，任何 TP 度下都必须整份复制——DeepSeek 自己的推理方案因此在 attention 部分不用 TP 而用 DP（每卡处理不同的请求），这个选择直接来自本节的算术。


## 三、MLA：把 K、V 压成一个 latent

### 1. 低秩压缩的想法

GQA 的思路是"少几组 K、V"，每组仍是完整的 $$d_{head}$$ 维、K 与 V 各一份。Multi-head Latent Attention（DeepSeek-AI 2024，DeepSeek-V2 提出、V3 沿用）换了一个方向：**不缓存 K 和 V 本身，缓存一个能重建出 K 和 V 的低维向量**。

回顾 $$K = X W_K$$、$$V = X W_V$$：每个 token 的 $$k$$、$$v$$ 都是同一个 $$d$$ 维输入 $$h$$ 的线性变换。如果先把 $$h$$ 压到一个 $$d_c$$ 维的 $$c$$，再由 $$c$$ 线性升维得到全部 head 的 $$k$$ 和 $$v$$，那么只要 $$d_c \ll 2 n_h d_{head}$$，缓存 $$c$$ 就比缓存 $$k$$、$$v$$ 省得多。这在数学上等价于给 $$[W_K, W_V]$$ 施加了秩不超过 $$d_c$$ 的约束。

DeepSeek-V3 取 $$d_c = 512$$，而 128 个 head 的完整 K、V 是 $$2 \times 128 \times 128 = 32768$$ 维。压缩比 64 倍——如果没有后面要讲的 RoPE 问题。

### 2. 完整结构

记输入为 $$h_t \in \mathbb{R}^{7168}$$，DeepSeek-V3 的 MLA 每层由以下几组投影构成（下标 D 表示 down、U 表示 up、R 表示 RoPE）：

**KV 侧**：

$$
c^{KV}_t = W_{DKV} h_t \in \mathbb{R}^{512}, \qquad k^R_t = \text{RoPE}(W_{KR} h_t) \in \mathbb{R}^{64}
$$

$$
k^C_t = W_{UK} c^{KV}_t \in \mathbb{R}^{128 \times 128}, \qquad v^C_t = W_{UV} c^{KV}_t \in \mathbb{R}^{128 \times 128}
$$

实现上 $$W_{DKV}$$ 与 $$W_{KR}$$ 合成一个 $$7168 \times 576$$ 的矩阵一次算出，前 512 维经 RMSNorm 后作为 $$c^{KV}_t$$，后 64 维施加 RoPE 后作为 $$k^R_t$$。注意 $$k^R_t$$ **只有一份，所有 128 个 head 共享**——这是它能被塞进 KV cache 的原因。

**Query 侧**也做低秩（目的是省训练时的激活显存，不影响 KV cache）：

$$
c^Q_t = W_{DQ} h_t \in \mathbb{R}^{1536}, \qquad q^C_t = W_{UQ} c^Q_t \in \mathbb{R}^{128 \times 128}, \qquad q^R_t = \text{RoPE}(W_{QR} c^Q_t) \in \mathbb{R}^{128 \times 64}
$$

实现上 $$W_{UQ}$$ 与 $$W_{QR}$$ 合成 $$1536 \times 24576$$（$$24576 = 128 \times 192$$）。query 的 RoPE 部分每个 head 各有一份。

**拼接与 attention**：第 $$h$$ 个 head 的 query 与 key 都是 192 维：

$$
q_{t,h} = [q^C_{t,h};\ q^R_{t,h}] \in \mathbb{R}^{192}, \qquad k_{j,h} = [k^C_{j,h};\ k^R_j] \in \mathbb{R}^{192}
$$

$$
o_{t,h} = \sum_{j \le t} \text{softmax}_j\left(\frac{q_{t,h}^\top k_{j,h}}{\sqrt{192}}\right) v^C_{j,h}, \qquad u_t = W_O [o_{t,1}; \dots; o_{t,128}]
$$

$$W_O$$ 是 $$16384 \times 7168$$（$$16384 = 128 \times 128$$，value 的 head dim 是 128）。这就是总纲基线里 $$d_{head} = 192$$（q/k：128 nope + 64 rope）、v 为 128 的来源。

这些量在 DeepSeek-V3 的 `config.json` 里对应的字段是：

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
  "v_head_dim": 128
}
```

注意 `num_key_value_heads` 写的是 128——它描述的是升维之后 $$k^C$$、$$v^C$$ 的 head 数，**不能**拿去套 GQA 的 KV cache 公式，否则会算出 3.81 MiB/token 而不是 68.6 KiB。判断一个模型是不是 MLA，看的是 `kv_lora_rank` 字段是否存在；缓存量由 `kv_lora_rank + qk_rope_head_dim` 决定，与 `num_key_value_heads` 无关。这是本篇实践脚本里 `attn_type="mla"` 分支存在的理由。

### 3. 参数量与缓存量

参数按基线核一遍（每层）：$$W_{DQ}$$ $$7168 \times 1536 = 11.0$$M，$$W_{UQ}$$ $$1536 \times 24576 = 37.7$$M，$$W_{DKV}$$（含 $$W_{KR}$$）$$7168 \times 576 = 4.13$$M，$$W_{UK}$$ $$512 \times 16384 = 8.39$$M，$$W_{UV}$$ $$512 \times 16384 = 8.39$$M，$$W_O$$ $$16384 \times 7168 = 117.4$$M，合计约 187M，61 层约 11.4B。作为对比，如果 DeepSeek-V3 用 $$d = 7168$$、128 头、$$d_{head} = 128$$ 的标准 MHA，$$W_Q$$、$$W_K$$、$$W_V$$、$$W_O$$ 各 $$7168 \times 16384 = 117.4$$M，每层 470M——MLA 的 attention 参数反而少了 60%。

缓存量：每 token 每层只缓存 $$c^{KV}_t$$（512 个数）和 $$k^R_t$$（64 个数），共 **576 个数**，BF16 下 1152 字节。61 层：

$$
61 \times 576 \times 2 = 70272 \text{ B} \approx 68.6 \text{ KiB}
$$

如果它用 MHA（按 $$d_{head} = 128$$ 计）：

$$
2 \times 61 \times 128 \times 128 \times 2 = 3997696 \text{ B} \approx 3.81 \text{ MiB}
$$

压缩比 $$3997696 / 70272 \approx 57$$ 倍。128K 上下文：$$70272 \times 131072 \approx 9.21 \times 10^9$$ B ≈ **8.6 GiB**——比 Llama-3-8B 的 16 GiB 还小一半。

于是核心问题的前半已经有了答案：DeepSeek-V3 的 128 个 head 在 KV cache 里根本不存在，缓存的是 128 个 head 共享的一个 576 维向量；$$n_{kv} \cdot d_{head}$$ 这一项在公式里变成了 $$d_c + d_h^R$$。层数 61 比 32 多，但 $$576 \times 61 = 35136$$ 仍小于 Llama-3-8B 的 $$2 \times 8 \times 128 \times 32 = 65536$$。

### 4. 为什么 RoPE 与低秩压缩不兼容

上面那个 64 维的 $$k^R$$ 看起来是个补丁——为什么不直接把 RoPE 加在 $$k^C$$ 上，让 $$d_c = 512$$ 就够？

RoPE 对位置 $$m$$ 的向量施加一个只依赖 $$m$$ 的旋转矩阵 $$R_m$$（细节在下一篇），它的关键性质是 $$R_m^\top R_n = R_{n-m}$$：点积只依赖相对位置。如果把 RoPE 直接加在压缩路径上，位置 $$t$$ 的 query 与位置 $$j$$ 的 key 的点积是：

$$
(R_t W_{UQ} c^Q_t)^\top (R_j W_{UK} c^{KV}_j) = (c^Q_t)^\top W_{UQ}^\top R_{j-t} W_{UK}\, c^{KV}_j
$$

$$R_{j-t}$$ 夹在 $$W_{UQ}^\top$$ 与 $$W_{UK}$$ 之间，且随 $$j - t$$ 变化。下一节要做的"矩阵吸收"依赖把 $$W_{UQ}^\top W_{UK}$$ 预先乘成一个矩阵，与位置无关；中间插一个随位置变化的 $$R_{j-t}$$，这个乘积就无法预计算——要么对每个相对位置存一个矩阵（不可能），要么老老实实把 $$c^{KV}_j$$ 升维成 $$k^C_j$$ 再旋转（那就等于放弃了压缩：升维后的 K 是 $$128 \times 128$$，每个 cached token 都要做一次 $$512 \times 16384$$ 的乘法）。

所以 MLA 把 RoPE **解耦**出来：压缩路径 $$c^{KV}$$ 完全不带位置信息，位置信息由一个独立的、直接从 $$h_t$$ 算出的 64 维 $$k^R_t$$ 承载；它是已经旋转好的，缓存的就是旋转后的值，所有 head 共用。代价是每 token 每层多缓存 64 个数（576 而不是 512，多 12.5%），换来压缩路径可以做矩阵吸收。

### 5. 矩阵吸收

decode 时如果按第 2 节的公式字面执行，每一步要把上下文中每个位置的 $$c^{KV}_j$$ 用 $$W_{UK}$$、$$W_{UV}$$ 升维成 128 个 head 的 $$k^C_j$$、$$v^C_j$$——每个 cached token 每层 $$2 \times 512 \times 32768 \approx 33.5$$M FLOPs，$$s = 128$$K 时每层 4.4 TFLOP，比整个模型的权重 GEMM 还贵。显然不能这么做。

**Query 侧的吸收**。看 $$q^C$$ 与 $$k^C$$ 的点积（第 $$h$$ 个 head，$$W_{UK,h}$$ 是 $$W_{UK}$$ 中对应第 $$h$$ 个 head 的 $$128 \times 512$$ 子块）：

$$
(q^C_{t,h})^\top k^C_{j,h} = (q^C_{t,h})^\top W_{UK,h}\, c^{KV}_j = \left(W_{UK,h}^\top q^C_{t,h}\right)^\top c^{KV}_j
$$

把 $$W_{UK,h}^\top$$ 移到 query 一侧：先算 $$\tilde q_{t,h} = W_{UK,h}^\top q^C_{t,h} \in \mathbb{R}^{512}$$，这是**每步只做一次**的小计算（128 个 head 各一个 $$512 \times 128$$ 的矩阵向量乘），然后 $$\tilde q_{t,h}$$ 直接与 cache 里的 $$c^{KV}_j$$ 做 512 维点积。更进一步，$$W_{UK,h}^\top W_{UQ,h}$$ 可以离线乘成一个 $$512 \times 1536$$ 的矩阵，query 从 $$c^Q_t$$ 一步得到 $$\tilde q_{t,h}$$——这就是"把 $$W_{UK}$$ 吸进 $$W_{UQ}$$"。

再把 RoPE 部分拼上，完整的 score 是：

$$
\text{score}_{t,j,h} = \frac{\tilde q_{t,h}^\top c^{KV}_j + (q^R_{t,h})^\top k^R_j}{\sqrt{192}} = \frac{[\tilde q_{t,h};\ q^R_{t,h}]^\top [c^{KV}_j;\ k^R_j]}{\sqrt{192}}
$$

一个 576 维的点积，右侧恰好就是 cache 里的那 576 个数。注意缩放因子仍是原始 head dim 的 $$\sqrt{192}$$，不是 $$\sqrt{576}$$——吸收只是重排了乘法顺序，数值应与原式相同。

**输出侧的吸收**。第 $$h$$ 个 head 的输出：

$$
o_{t,h} = \sum_j p_{t,j,h}\, v^C_{j,h} = \sum_j p_{t,j,h}\, W_{UV,h}\, c^{KV}_j = W_{UV,h} \left(\sum_j p_{t,j,h}\, c^{KV}_j\right)
$$

先在 512 维的 latent 上做加权和 $$\tilde o_{t,h} = \sum_j p_{t,j,h} c^{KV}_j$$，再乘 $$W_{UV,h}$$；而 $$W_O$$ 紧跟其后，$$u_t = \sum_h W_{O,h} W_{UV,h} \tilde o_{t,h}$$，其中 $$W_{O,h} W_{UV,h}$$ 是 $$7168 \times 512$$，可以离线乘好——这就是"把 $$W_{UV}$$ 吸进 $$W_O$$"。

吸收之后，attention 核心（softmax 前后两次矩阵乘）看到的形状是：

- 128 个 query head，每个 query 576 维；
- **一个** KV head，key 576 维（512 nope + 64 rope），value 512 维（就是 key 的前 512 维——K 与 V 共享同一份存储）；
- 每个 head 的输出 512 维。

这在 kernel 层面就是一个 **head dim 为 576/512、128 个 query head 共享一个 KV head 的 MQA**。vLLM、SGLang 中的 MLA decode 路径以及 DeepSeek 开源的 FlashMLA，都是按这个形状写的。

### 6. 吸收后的算量与访存：decode 划算，prefill 不一定

套用第二章第 3 节的推导：每读一个 KV 元素服务 128 个 query head，$$g = 128$$。更精确地算每个 cached token 每层：

- 读取：576 个 BF16，1152 字节；
- FLOPs：128 个 head，每个 head 做 576 维点积（QK）与 512 维加权和（PV），各 2 FLOPs 每维：$$128 \times 2 \times (576 + 512) = 278528$$；
- 算术强度：$$278528 / 1152 \approx 242$$ FLOP/byte。

这已经接近 H100 的 ridge point 295。对比 Llama-3-8B 的 GQA：每 cached token 每层读 $$2 \times 8 \times 128 \times 2 = 4096$$ 字节，做 $$32 \times 2 \times 256 = 16384$$ FLOPs，强度 4。MLA 的 decode attention 从"极度访存受限"变成了"接近平衡"，这是 KV cache 缩小 57 倍之外的第二个收益——每一步的 KV 读取时间也随之缩小。128K 上下文时 DeepSeek-V3 每层每步读 $$576 \times 2 \times 131072 \approx 151$$ MB，61 层 8.6 GiB，3.35 TB/s 下约 2.7 ms；对应的 attention FLOPs 约 $$278528 \times 61 \times 131072 \approx 2.2$$ TFLOP，989 TFLOPS 下约 2.3 ms（假设全放一张卡上，仅为量级说明）。两项已经是同一个量级，无论哪个是瓶颈都不会比另一个差太多。

但**算量确实变大了**。未吸收时每个 head 的点积长度是 192（QK）+ 128（PV）= 320；吸收后是 576 + 512 = 1088，约 3.4 倍。decode 时这不要紧：attention 本来就在等 HBM，多做的 FLOPs 是"免费"的，而且省掉的是每 cached token 33.5M FLOPs 的升维。

prefill 就不同了。prefill 时全部 $$s$$ 个 token 同时在场：

- 升维 $$W_{UK}$$、$$W_{UV}$$ 是一次普通的权重 GEMM：$$[s, 512] \times [512, 32768]$$，$$s = 8192$$ 时 $$2 \times 8192 \times 512 \times 32768 \approx 275$$ GFLOPs 每层，算术强度高，跑在接近峰值；
- 升维后按 192/128 的 head dim 做标准（causal、FlashAttention 式）attention：$$s = 8192$$ 时每层 $$128 \times 2 \times 320 \times 8192^2 / 2 \approx 2.75$$ TFLOP，61 层约 168 TFLOP；
- 如果也走吸收路径：$$128 \times 2 \times 1088 \times 8192^2 / 2 \approx 9.35$$ TFLOP 每层，61 层约 570 TFLOP。

prefill 是 compute-bound，多 400 TFLOP 就是多 0.4 秒以上（按 989 TFLOPS 峰值）。$$s^2$$ 项随上下文平方增长，越长的 prefill 吸收路径亏得越多；而升维 GEMM 只随 $$s$$ 线性增长，且只做一次。所以 vLLM 与 FlashMLA 在 prefill（以及长 chunked prefill）阶段走**非吸收路径**：读 $$c^{KV}$$、升维、做普通 attention；只在 decode 走吸收路径。同一个模型、同一份 cache，两个阶段用两套数学上等价的算法——这是 MLA 带给推理引擎的额外复杂度。

### 7. 与 GQA 放在同一把尺子上

一个自然的问题：如果 DeepSeek-V3 不用 MLA，而用一个"激进的 GQA"，能不能达到同样的 KV cache？每层 576 个数、$$d_{head} = 128$$，相当于 $$n_{kv} = 576 / 256 = 2.25$$ 个 KV 头——取 2 个 KV 头，128 个 query head 每 64 个共享一组 K、V，每层 512 个数，比 MLA 还少一点。缓存量上 GQA 完全做得到。

差别在表达能力。GQA 的 2 个 KV 头意味着上下文里每个 token 只提供 2 个 128 维的 key 和 2 个 128 维的 value，128 个 query head 只能从这 2 组里选；MLA 的 128 个 head 各有自己的 $$W_{UK,h}$$、$$W_{UV,h}$$，从同一个 512 维 latent 里各取一个不同的 128 维投影——每个 head 看到的 key、value 都不同，只是它们共同被约束在一个 512 维子空间里。前者是"少几个头"，后者是"头很多但共享一个低秩底座"。DeepSeek-V2 论文里的消融是 MLA 优于同等 KV 预算的 GQA、甚至不弱于 MHA；这个结论的代价，就是下一节要列的那几条。

另一个角度是把 GQA 看成 MLA 的特例：GQA 相当于 $$W_{UK}$$、$$W_{UV}$$ 是固定的 0/1 选择矩阵（每个 query head 直接选中它所属组的 K、V），$$d_c = 2 n_{kv} d_{head}$$，且没有 RoPE 解耦的需要（因为"升维"是恒等的，RoPE 可以直接加在缓存的 K 上）。MLA 把这两个选择矩阵变成可学习的稠密矩阵，换来更小的 $$d_c$$，也换来了 RoPE 不兼容的问题。

### 8. 回答核心问题

**怎么做到的**：DeepSeek-V3 每层 128 个 head 的 K、V 不进 cache，进 cache 的是它们共同的 512 维低秩 latent 加一个 64 维、所有 head 共享的解耦 RoPE key，每 token 每层 576 个数；61 层 BF16 共 68.6 KiB，比 Llama-3-8B GQA 的 128 KiB 小 46%，比自己用 MHA 的 3.81 MiB 小 57 倍。decode 时通过矩阵吸收直接在 latent 上做 attention，等价于 128 个 query head 共享一个 576/512 维 KV head 的 MQA，算术强度约 242 FLOP/byte。

**代价是什么**：

1. **算量**：吸收后每个 head 的点积从 320 维变 1088 维，attention 核心 FLOPs 约 3.4 倍；decode 里被访存掩盖，prefill 里掩盖不了，因此要维护吸收/非吸收两条路径；
2. **结构约束**：K、V 被限制在秩 512 的子空间里，RoPE 只能加在额外的 64 维上，与标准 RoPE 模型的位置编码行为不完全相同；DeepSeek 报告 MLA 在他们的训练设置下质量不低于 MHA，但这是一个特定规模、特定超参下的结论，不是低秩约束"免费"的证明；
3. **工程复杂度**：head dim 576 不是标准 FlashAttention 的常规尺寸，需要专用 kernel（FlashMLA、FlashInfer 的 MLA 后端）；prefill 与 decode 的 cache 布局要能同时服务两条路径；KV 量化时 nope 与 rope 两段的数值范围不同，通常要分开处理；
4. **不能从 MHA checkpoint 直接转换**：GQA 可以从 MHA 均值池化 uptrain 得到，MLA 的投影结构不同，需要从头训（或专门的转换方法）。


## 四、attention 的中间结果与 FlashAttention 的 IO 复杂度

前三章讲的是 KV cache——decode 的问题。prefill 阶段 attention 还有另一个显存问题：$$S = QK^\top$$ 这个 $$s \times s$$ 的中间矩阵。

### 1. 4 GiB 的中间结果

按公式字面实现，每层每个 head 都要物化一个 $$s \times s$$ 的 score 矩阵，再做 softmax，再乘 $$V$$。$$s = 8192$$、32 个 head、BF16：

$$
32 \times 8192 \times 8192 \times 2 \text{ B} = 4 \text{ GiB}
$$

每层 4 GiB，$$s = 128$$K 时每层 1 TiB。这显然不能存，也不能读写：4 GiB 写出再读回是 8 GiB 的 HBM 流量，而这一层真正有用的输出只有 $$s \times d \times 2 = 64$$ MiB。attention 的 HBM 流量是 $$O(s^2)$$，而它的 FLOPs 是 $$O(s^2 d)$$——算术强度只有 $$O(d)$$，对 $$d_{head} = 128$$ 来说不到 ridge point 的一半，何况还有 softmax 那一趟额外读写。

### 2. FlashAttention 的 IO 复杂度

FlashAttention（Dao 等 2022）的做法是不物化 $$S$$：把 $$K$$、$$V$$ 切成大小为 $$B_c$$ 行的块，$$Q$$ 切成 $$B_r$$ 行的块，一个 $$Q$$ 块对一个 $$K$$、$$V$$ 块的 $$B_r \times B_c$$ 局部 score 在 SRAM 里算完就丢，用 online softmax 维护每行的运行最大值与归一化因子，逐块累积输出 $$O$$。这里只推导它的 HBM 流量，不讲 kernel。

设 SRAM 容量为 $$M$$ 个元素。块的大小受 SRAM 限制：一个 $$K$$ 块加一个 $$V$$ 块要放得下，$$B_c \cdot d = \Theta(M)$$，即 $$B_c = \Theta(M / d)$$。外层循环遍历 $$K$$、$$V$$ 块，共 $$s / B_c = \Theta(s d / M)$$ 次；每次外层迭代读入一个 $$K$$、$$V$$ 块（累计只读一遍 $$K$$、$$V$$，$$O(sd)$$），然后内层遍历全部 $$Q$$ 块，读入 $$Q$$ 块与对应的 $$O$$ 块、更新后写回 $$O$$——每次外层迭代把 $$Q$$ 和 $$O$$ 整体过一遍，$$O(sd)$$。合计：

$$
\text{HBM accesses} = O(sd) + O\!\left(\frac{sd}{M}\right) \cdot O(sd) = O\!\left(\frac{s^2 d^2}{M}\right)
$$

对比标准实现的 $$O(sd + s^2)$$。$$d = 128$$，H100 每个 SM 的 shared memory 约 228 KB，按 BF16 算 $$M \approx 10^5$$ 个元素，$$d^2 / M \approx 0.16$$——流量比物化 $$S$$ 少一个数量级左右，且不再需要 $$O(s^2)$$ 的中间显存。FLOPs 不变（还是 $$4 s^2 d$$ 每 head，外加 online softmax 的少量重缩放），变的只是访存，算术强度从 $$O(d)$$ 提到 $$O(M / d)$$，在 $$M / d \approx 800$$ 的量级上超过了 ridge point，attention 从 memory-bound 变成 compute-bound。这个"块大小由 SRAM 决定、流量与 $$1/M$$ 成正比"的结论，是后面所有 attention kernel（FlashAttention-2/3、FlashInfer、FlashMLA）共享的出发点。

这个推导也解释了为什么 MLA 吸收后 576 的 head dim 需要专门的 kernel：$$B_c = \Theta(M / d)$$，$$d$$ 从 128 变成 576，同样的 SRAM 只能放下不到四分之一的 K、V 行，块变小、外层循环次数变多、$$Q$$ 和 $$O$$ 被重读的次数增加；而流量公式里的 $$d^2$$ 项直接放大 20 倍。FlashMLA 的应对是利用 K 与 V 共享存储（V 是 K 的前 512 维），只装载一份 576 维的数据，再在寄存器与 shared memory 之间重新分配空间——这些属于 kernel 实现，本篇不展开，但它们要解决的问题就是这个公式里的 $$d$$。

对 decode 这个公式退化成另一个形态：$$s_q = 1$$，$$Q$$ 只有一行，外层循环 $$sd / M$$ 次每次重读 $$Q$$ 和 $$O$$ 的代价可以忽略，流量就是 $$O(sd)$$——读一遍 KV cache，即第一章的结论。此时问题不在流量而在**并行度**：一条序列一个 query 只能启动一个线程块，填不满 132 个 SM。FlashDecoding 一类实现把 KV 沿序列维切成若干段并行计算局部 softmax，最后再合并归一化因子——本质是把 online softmax 的"分块可合并"性质用在了另一个维度上。这就是为什么 decode 的 attention kernel 与 prefill 的不是同一个。

### 3. 因果掩码与 sliding window

因果掩码让位置 $$t$$ 只 attend 到 $$j \le t$$。物化实现只是把上三角填 $$-\infty$$，FLOPs 一分不少；分块实现则可以直接**跳过**全在上三角的块，attention 的 FLOPs 与流量近似减半：

$$
\sum_{t=1}^{s} t \approx \frac{s^2}{2}
$$

上一篇的 prefill 数字就用了这一点：Llama-3-8B 8K prefill 不利用掩码约 158 TFLOP，利用后约 140 TFLOP；128K 时 attention 项按 $$s^2/2$$ 计约 4.5 PFLOP，若不利用则是 9 PFLOP，比权重项的 2 PFLOP 多得多。128K 以上的 prefill，causal skip 不是优化，是必需。

sliding window attention（Mistral 7B 用 4096 的窗口）进一步只让位置 $$t$$ attend 到 $$[t - w, t]$$。attention 的 FLOPs 变成 $$O(s w)$$——对 $$s$$ 线性；KV cache 也不再随上下文增长，每层每序列最多 $$w$$ 个 token：Mistral 7B（结构与 Llama-3-8B 同为 32 层、8 个 KV 头、$$d_{head} = 128$$）的 KV cache 上限是 $$128 \text{ KiB} \times 4096 = 512$$ MiB，无论上下文多长。代价是超出窗口的信息只能通过多层堆叠间接传递（$$L$$ 层理论感受野 $$L \cdot w$$），长距离检索能力有损，所以后来的模型多是滑窗层与全局层交替（如 Gemma 2、Llama 4 的部分层）。


## 五、KV cache 的工程变量

公式 $$\text{bytes/token} = 2 L n_{kv} d_{head} \cdot \text{bytes/elem}$$ 给的是每 token 的下界。推理引擎实际占用与之的差别来自三件事。

### 1. 分页：碎片率

早期实现按每条序列的最大长度预分配连续 KV 显存，一条 8K 上限、实际只用 500 token 的请求浪费 94%。PagedAttention（Kwon 等 2023，vLLM）把每条序列的 KV 切成固定大小的 block（默认 16 个 token），按需分配，逻辑块到物理块的映射表由 kernel 在读 KV 时查。

它对上面公式的影响只有一项：**内部碎片率**。每条序列只有最后一个 block 不满，平均浪费半个 block；Llama-3-8B、block 16 时约 $$8 \times 128 \text{ KiB} = 1$$ MiB 每序列，相对于一条 8K 序列 1 GiB 的 KV 可以忽略。每 token 的字节数不变，变的是"能用上的比例"从几成提到接近 100%，第一章算的"约 52 万 token"这个容量才真的能被填满。

### 2. prefix 共享

同一个 system prompt、同一段 few-shot 示例、同一份文档被多条请求共用时，它们的 KV 在数值上完全相同（causal 模型的 KV 只依赖前缀）。分页之后，共享只是让多条序列的映射表指向同一批物理块，加引用计数即可。$$n$$ 条请求共享 $$p$$ 个 token 的前缀，省下 $$(n - 1) \cdot p \cdot \text{bytes/token}$$ 的显存和对应的 prefill 算量。对 Agent 类负载（几千 token 的工具描述 + 短 query），命中率通常很高，这是 KV cache 容量的一个乘数，而不是加数。

### 3. 量化：字节数减半

bytes/elem 从 BF16 的 2 降到 FP8（E4M3）或 INT8 的 1，每 token 字节数直接减半：Llama-3-8B 64 KiB，Llama-3-70B 160 KiB，DeepSeek-V3 约 34.3 KiB；128K 上下文分别是 8 GiB、20 GiB、4.3 GiB。缩放因子通常按 head 或按 token 一个，开销不到 1%。decode 时 KV 读取的字节数同样减半，第一章那个 8K × batch 64 的 64 GiB 变成 32 GiB。

数值上 K 比 V 更敏感（K 直接进 softmax 指数，某些通道有明显的离群值），INT4 KV 一般要对 K 做按通道量化、对 V 做按 token 量化（KIVI 一类方法）。FP8 KV 在 H100 上还有一个额外好处：attention kernel 可以直接用 FP8 的输入，不必先反量化。这些是第六、第七篇的内容，这里只需记住：**量化是公式里唯一一个不改变结构就能减半的因子**。

### 4. 三个因子怎么叠加

把这一章与前面合起来，一台机器"能放多少 token 的 KV"由四个乘子决定：

$$
\text{tokens} = \frac{\text{显存预算}}{2 L n_{kv} d_{head} \cdot \text{bytes/elem}} \times \frac{1}{1 + \text{碎片率}} \times \text{prefix 复用倍数}
$$

结构（GQA 的 $$n_{kv}$$、MLA 的 $$d_c + d_h^R$$）决定分母里的元素个数，量化决定 bytes/elem，分页决定碎片率，prefix 共享决定复用倍数。四个乘子彼此独立，可以同时用：一个 MLA + FP8 KV + 分页 + prefix 缓存的服务栈，相对 MHA + BF16 + 预分配的朴素实现，KV 容量的差距可以有两到三个数量级。这四个因子里只有第一个必须在训练前决定，其余三个都是推理侧的选择——这也是为什么 KV cache 是推理系统里优化空间最大的一块。


## 六、实践：给 llm_cost.py 加上 KV cache

本篇在贯穿脚本里新增 `attn_type` 字段（`"mha" | "gqa" | "mqa" | "mla"`）、MLA 用到的 `kv_lora_rank`（$$d_c$$）与 `qk_rope_head_dim`（$$d_h^R$$）字段，以及 `n_params`（总参数量，取第一篇与第五篇的结果，用来估权重显存）。新增三个函数：`kv_bytes_per_token`、`decode_attn_intensity`、`max_concurrency`。

```python
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
    attn_type: str = "gqa"        # "mha" | "gqa" | "mqa" | "mla"
    kv_lora_rank: int = 0         # MLA: d_c
    qk_rope_head_dim: int = 0     # MLA: d_h^R
    n_params: float = 0.0         # 总参数量（含 embedding / lm_head）

LLAMA3_8B = ModelConfig("Llama-3-8B", 4096, 32, 32, 8, 128, 14336, 128256,
                        attn_type="gqa", n_params=8.03e9)
LLAMA3_70B = ModelConfig("Llama-3-70B", 8192, 80, 64, 8, 128, 28672, 128256,
                         attn_type="gqa", n_params=70.55e9)
DEEPSEEK_V3 = ModelConfig("DeepSeek-V3", 7168, 61, 128, 128, 128, 18432, 129280,
                          attn_type="mla", kv_lora_rank=512, qk_rope_head_dim=64,
                          n_params=671e9)

@dataclass
class GPU:
    name: str
    hbm_bytes: float
    bandwidth: float   # bytes/s
    bf16_flops: float  # FLOP/s

H100 = GPU("H100 SXM", 80e9, 3.35e12, 989e12)
A100 = GPU("A100 80GB", 80e9, 2.0e12, 312e12)

KIB, MIB, GIB = 1024, 1024 ** 2, 1024 ** 3


def kv_elems_per_token_per_layer(cfg: ModelConfig) -> int:
    """每层每 token 需要缓存的元素个数（与 dtype 无关）。"""
    if cfg.attn_type == "mha":
        return 2 * cfg.n_heads * cfg.head_dim
    if cfg.attn_type == "gqa":
        return 2 * cfg.n_kv_heads * cfg.head_dim
    if cfg.attn_type == "mqa":
        return 2 * 1 * cfg.head_dim
    if cfg.attn_type == "mla":
        # 只缓存 c_KV（d_c 维）与解耦的 RoPE key k^R（d_h^R 维），所有 head 共享
        return cfg.kv_lora_rank + cfg.qk_rope_head_dim
    raise ValueError(cfg.attn_type)


def kv_bytes_per_token(cfg: ModelConfig, dtype_bytes: int = 2) -> int:
    return cfg.layers * kv_elems_per_token_per_layer(cfg) * dtype_bytes


def kv_bytes(cfg: ModelConfig, ctx: int, dtype_bytes: int = 2) -> int:
    return kv_bytes_per_token(cfg, dtype_bytes) * ctx


def decode_attn_intensity(cfg: ModelConfig, dtype_bytes: int = 2) -> float:
    """decode 时 attention 对 KV cache 的算术强度（FLOP/byte）。
    每个 cached token：每个 query head 对 K 做一次点积、对 V 做一次加权和，每维 2 FLOPs。"""
    if cfg.attn_type == "mla":
        k_dim = cfg.kv_lora_rank + cfg.qk_rope_head_dim   # 吸收后 K 维 576
        v_dim = cfg.kv_lora_rank                          # 吸收后 V 维 512
        flops = cfg.n_heads * 2 * (k_dim + v_dim)
    else:
        flops = cfg.n_heads * 2 * (cfg.head_dim + cfg.head_dim)
    return flops / (kv_elems_per_token_per_layer(cfg) * dtype_bytes)


def weight_bytes(cfg: ModelConfig, weight_dtype_bytes: int = 2) -> float:
    return cfg.n_params * weight_dtype_bytes


def max_concurrency(cfg: ModelConfig, gpu: GPU, ctx: int, dtype_bytes: int = 2,
                    n_gpus: int = 1, weight_dtype_bytes: int = 2,
                    reserve_frac: float = 0.0) -> int:
    """给定显存预算，能同时驻留多少条上下文为 ctx 的序列。
    只算权重 + KV cache，忽略激活、碎片与框架开销；多卡按权重与 KV 均匀切分估算。"""
    budget = gpu.hbm_bytes * n_gpus * (1 - reserve_frac) - weight_bytes(cfg, weight_dtype_bytes)
    if budget <= 0:
        return 0
    return int(budget // kv_bytes(cfg, ctx, dtype_bytes))


if __name__ == "__main__":
    print(f"{'model':<14}{'attn':<6}{'KV B/token':>12}{'KiB/token':>11}"
          f"{'128K ctx':>11}{'FLOP/byte':>11}")
    for cfg in (LLAMA3_8B, LLAMA3_70B, DEEPSEEK_V3):
        b = kv_bytes_per_token(cfg)
        print(f"{cfg.name:<14}{cfg.attn_type:<6}{b:>12}{b / KIB:>11.1f}"
              f"{kv_bytes(cfg, 131072) / GIB:>9.1f} G{decode_attn_intensity(cfg):>11.0f}")

    mha8b = ModelConfig(**{**LLAMA3_8B.__dict__, "attn_type": "mha"})
    mhav3 = ModelConfig(**{**DEEPSEEK_V3.__dict__, "attn_type": "mha"})
    print(f"\nLlama-3-8B as MHA : {kv_bytes_per_token(mha8b) / KIB:.0f} KiB/token")
    print(f"DeepSeek-V3 as MHA: {kv_bytes_per_token(mhav3) / MIB:.2f} MiB/token, "
          f"ratio = {kv_bytes_per_token(mhav3) / kv_bytes_per_token(DEEPSEEK_V3):.1f}x")

    print("\nmax concurrency on H100 (weights + KV only, BF16 KV):")
    plans = [(LLAMA3_8B, 1, 2), (LLAMA3_70B, 8, 2), (DEEPSEEK_V3, 16, 1)]
    ctxs = [4096, 8192, 32768, 131072]
    print(f"{'model':<14}{'GPUs':>5}{'weights':>9}" + "".join(f"{c // 1024:>7}K" for c in ctxs))
    for cfg, n, wb in plans:
        row = f"{cfg.name:<14}{n:>5}{weight_bytes(cfg, wb) / 1e9:>7.0f}GB"
        for c in ctxs:
            row += f"{max_concurrency(cfg, H100, c, 2, n, wb):>8}"
        print(row)

    print("\nsame, with FP8 KV cache:")
    for cfg, n, wb in plans:
        row = f"{cfg.name:<14}{n:>5}{weight_bytes(cfg, wb) / 1e9:>7.0f}GB"
        for c in ctxs:
            row += f"{max_concurrency(cfg, H100, c, 1, n, wb):>8}"
        print(row)
```

运行输出：

```text
model         attn    KV B/token  KiB/token   128K ctx  FLOP/byte
Llama-3-8B    gqa         131072      128.0     16.0 G          4
Llama-3-70B   gqa         327680      320.0     40.0 G          8
DeepSeek-V3   mla          70272       68.6      8.6 G        242

Llama-3-8B as MHA : 512 KiB/token
DeepSeek-V3 as MHA: 3.81 MiB/token, ratio = 56.9x

max concurrency on H100 (weights + KV only, BF16 KV):
model          GPUs  weights      4K      8K     32K    128K
Llama-3-8B        1     16GB     119      59      14       3
Llama-3-70B       8    141GB     371     185      46      11
DeepSeek-V3      16    671GB    2115    1057     264      66

same, with FP8 KV cache:
Llama-3-8B        1     16GB     238     119      29       7
Llama-3-70B       8    141GB     743     371      92      23
DeepSeek-V3      16    671GB    4231    2115     528     132
```

读这张表要注意几点：

- 并发数是**只算权重与 KV cache 的上界**。实际引擎还要留激活（prefill 一个 chunk 的中间张量）、CUDA graph、框架自身的显存，vLLM 默认 `gpu_memory_utilization=0.9` 一类的预留会再压掉 10% 左右，可以用 `reserve_frac` 模拟；
- Llama-3-8B 在一张 H100 上 128K 上下文只能放 3 条序列，这就是"长上下文 = 低并发"的定量版本；8K 上下文 59 条对应第一章"64 条 8K 序列"的估算（差别来自 16.06 GB 权重与 64 GiB 预算的取整）；
- DeepSeek-V3 的 FP8 权重 671 GB 放不进 8 张 H100（640 GB），表里用了 16 卡；即便如此每卡分到的 KV 预算只有约 38 GB，但因为每 token 只有 68.6 KiB，128K 上下文仍能放 66 条——这就是 MLA 对服务成本的意义；
- FP8 KV 让每一格翻倍，且不改变模型结构，是所有优化中性价比最高的一项——前提是质量可接受。

下一篇会给 `kv_bytes` 加上上下文长度扫描，把 RoPE 外推与 KV 显存放在同一张图里看。


## 七、小结

本篇从 KV cache 公式出发推了四种 attention 结构：

- KV cache 存在的理由是 causal 模型的 K、V 一旦算出就不再变化；不缓存则每步都是一次 $$O(t^2)$$ 的 prefill。缓存的代价是 $$2 L n_{kv} d_{head} \cdot \text{bytes/elem}$$ 每 token 的显存，以及 decode 每步把它全读一遍的 HBM 流量；
- MQA 与 GQA 直接减少 $$n_{kv}$$：GQA 的 $$g = n_h / n_{kv}$$ 同时缩小 KV 字节数 $$g$$ 倍、提高 decode attention 算术强度到 $$g$$ FLOP/byte；可以从 MHA checkpoint 均值池化后 uptrain 得到；
- MLA 把 K、V 压成 512 维 latent 加 64 维解耦 RoPE key，每 token 每层 576 个数；RoPE 必须解耦是因为位置相关的旋转矩阵不能被吸进与位置无关的升维矩阵；矩阵吸收后 decode 等价于 128 头共享一个 576/512 维 KV 头的 MQA，强度约 242 FLOP/byte，但 attention 核心 FLOPs 约 3.4 倍，所以 prefill 走非吸收路径；
- 标准 attention 物化的 $$S$$ 在 8K、32 头时每层 4 GiB；FlashAttention 分块后 HBM 流量为 $$O(s^2 d^2 / M)$$；因果掩码让 prefill attention 减半，sliding window 让它线性；
- 分页只影响碎片率，prefix 共享是容量的乘数，FP8/INT8 KV 让字节数减半。

本篇算出的数字：

```text
                              Llama-3-8B     Llama-3-70B    DeepSeek-V3
attention 结构                GQA (g=4)      GQA (g=8)      MLA
n_h / n_kv                    32 / 8         64 / 8         128 / latent
每层每 token 缓存元素数        2×8×128=2048   2048           512+64=576
KV bytes/token (BF16)         128 KiB        320 KiB        68.6 KiB
  若为 MHA                    512 KiB        2.5 MiB        3.81 MiB
  压缩比（vs MHA）             4×             8×             57×
KV @ 128K 上下文 (BF16)       16 GiB         40 GiB         8.6 GiB
KV @ 128K 上下文 (FP8)        8 GiB          20 GiB         4.3 GiB
decode attention 强度          4 FLOP/byte    8 FLOP/byte    242 FLOP/byte
8K×batch 64 每步 KV 读取      64 GiB         160 GiB        34 GiB
H100 放下权重后 KV 容量        ~52 万 token   —（需多卡）     —（需多卡）
最大并发 @8K（BF16 KV）        59 (1×H100)    185 (8×H100)   1057 (16×H100)
最大并发 @128K（BF16 KV）      3 (1×H100)     11 (8×H100)    66 (16×H100)
```

MLA 的 K、V 之所以能压成 576 个数，前提是 RoPE 被单独拿了出来；而 RoPE 本身——它的频率、波长、为什么 base 从 10000 涨到 500000、外推到 128K 要付什么代价——是下一篇的内容。

> **RoPE 的旋转频率如何决定模型"能看多远"？把 8K 训练的模型拉到 128K，哪些频率会失效，YaRN 与 Llama 3.1 的分段缩放各自修了什么？**


## 下一篇

[位置编码与长上下文](/positional-encoding-and-long-context.html)
