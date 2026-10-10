---
layout: slides
title: "Transformer 原理与实现：从论文到手搓 GPT-2"
subtitle: "系列精华 · 四篇正文，按 ↓ 看推导、代码与数字"
permalink: /slides/transformer-and-llm.html
series: transformer-and-llm
date: 2026-04-09 23:30:00 +0800
updated: 2026-10-10
author: arganzheng
description: "《Transformer 原理与实现：从论文到手搓 GPT-2》系列的分享用幻灯片：五种运算、训练与推理两种形态、nanoGPT 逐行、模型训练与验证。"
theme: white
transition: slide
---

## 这个系列的一句话主张

> Transformer 的结构可以从五种运算读懂；沿着一个 token 的旅程，再把同一套结构写成代码并训练起来。

| 篇 | 内容 |
|---|---|
| 01–04 | 静态结构 → token 旅程 → nanoGPT `model.py` → `train.py` 与实际训练 |

<aside class="notes" markdown="1">
总纲：/transformer-and-llm-structure-implementation-and-evolution.html。现代 LLM 结构系列从这里的 GPT-2 基线继续展开。
</aside>

---

## 四篇怎么连起来

```mermaid
flowchart LR
    P1["01 静态结构"] --> P2["02 token 的旅程"] --> P3["03 model.py"] --> P4["04 train.py"]
```

---

## 01 · Transformer 长什么样：只有五种运算

**结论**：embedding 查表 → **attention（唯一让 token 互相看的地方）** → FFN（逐 token 的非线性，知识在这，占一层 2/3）→ 残差 + LayerNorm → lm_head；attention 是集合运算不知道顺序，位置必须显式给。

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 330}}}%%
flowchart LR
    IN["token 编号<br/>[464, 3797, 3332, 319, 262]"] --> EMB["embedding 查表<br/>50257 × 768 → 5 个 768 维向量<br/>+ 位置表"]
    EMB --> B
    subgraph B["× 12 个 block"]
        direction TB
        A["LayerNorm → causal self-attention → 加回"]
        F["LayerNorm → FFN 768→3072→768 → 加回"]
        A --> F
    end
    B --> HEAD["LayerNorm → lm_head 768 × 50257<br/>softmax → p(mat) = 0.31"]
```

<aside class="notes" markdown="1">
原文 /transformer-architecture-from-a-sentence-to-the-next-token.html。GPT-2 small：12 × 7.09M + embedding 38.6M + 位置表 0.79M = 124,439,808；FFN 占一层 66.6%。原始 Transformer 是 encoder-decoder，GPT 去掉 encoder 与 cross-attention。
</aside>

<!-- v -->

### 一个 attention 头的六步手算（d = 4、T = 3）

![① x 乘三个矩阵得到 Q、K、V；② S = QKᵀ；③ 除 √d；④ 加 causal mask；⑤ softmax；⑥ 乘 V](/img/in-post/transformer-01-attention-by-hand.svg){: style="max-height: 400px"}

- $$\text{Attention}(Q,K,V) = \text{softmax}(QK^\top/\sqrt d + M)\,V$$；t2 的权重 (0.27, 0.27, 0.45)，与 PyTorch 对拍差 $$6 \times 10^{-8}$$
- 随机 $$q \cdot k$$ 的标准差是 $$\sqrt d$$：不除，d = 128 时 softmax 一个 token 独占

<!-- v -->

### GPT-2 真实的头在看什么

![GPT-2 small 处理 The cat sat on the mat because it was tired：第 4 层第 11 头看上一个词，第 3 头把 it 指回 cat（0.84）](/img/in-post/transformer-01-gpt2-attention-heads.svg){: style="max-height: 400px"}

- 多头不增加算量：h 个小乘法 = 一个大乘法，只多一个 $$W_O$$
- 换序实验：把输入打乱，attention 输出跟着换位——所以位置必须另给（GPT-2 查表 / Llama RoPE）

---

## 02 · 一个 token 的旅程：训练侧与推理侧

**结论**：causal mask + teacher forcing 让一句话 = T 个训练样本；推理时旧 token 的 K、V 不变，**算一次存下来就是 KV cache**——训练 / prefill / decode 是三种形态。

![训练的一步：五个位置一次前向同时得到五个分布，目标是输入右移一位，五个交叉熵取平均](/img/in-post/transformer-02-training-step-teacher-forcing.svg){: style="max-height: 380px"}

<aside class="notes" markdown="1">
原文 /transformer-token-journey-training-and-inference.html。随机初始化 loss ≈ ln V：16 词 2.78、GPT-2 词表 10.8、65 字符 4.17。反向要用前向中间量，激活与 B × T 成正比。
</aside>

<!-- v -->

### prefill 与 decode：为什么前面的 token 不用重算

![prefill 把五个 token 一次算完只取最后位置，K、V 存进 cache；decode 每步只算新 token 的 Q、K、V 并追加](/img/in-post/transformer-02-prefill-decode-kv-cache.svg){: style="max-height: 360px"}

| 形态 | 输入 | 有反向 | 存什么 | 瓶颈 |
|---|---|---|---|---|
| 训练 | [B, T] 全位置有用 | 是 | 激活值 | 算力 |
| prefill | [B, T] 只取最后 | 否 | K、V | 算力（TTFT） |
| decode | [B, 1] | 否 | 追加 K、V | **访存**（TPOT） |

<!-- v -->

### 数字

- 有 / 无 KV cache 输出逐 token **一致**；prompt 256 生成 256 快 **7.9 倍**
- 代价：Llama-3-8B **128 KiB / token** = 32 层 × 2 × 8 个 KV 头 × 128 × 2 B
- KV cache 不存 Q——旧 token 的 Q 用过即弃
- 训练长上下文比推理多一份与 B × T 成正比的激活值

---

## 03 · 手搓 GPT（上）：nanoGPT model.py 逐行

**结论**：330 行、6 个类，结构本身（LayerNorm、CausalSelfAttention、MLP、Block）**不到 90 行**；与 HF GPT-2 对拍相对差 $$9 \times 10^{-5}$$。

```python
class Block(nn.Module):
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # pre-norm：先 norm 再子层，残差直连
        x = x + self.mlp(self.ln_2(x))
        return x

# CausalSelfAttention.forward 的骨架
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)          # 一次 GEMM 算 QKV
k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # 头换到第 1 维
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # 融合 kernel
y = y.transpose(1, 2).contiguous().view(B, T, C)              # 拼回去
```

<aside class="notes" markdown="1">
原文 /nanogpt-model-py-line-by-line.html。wte.weight = lm_head.weight 共享省 31% 参数；所有权重 N(0, 0.02²)，写回残差流的 c_proj 再除 √(2L)；forward 传 targets 算全部位置，不传只算最后一个位置的 lm_head（省 99.9%）。
</aside>

<!-- v -->

### 每个类落在结构的哪个方框

| 类 | 行数 | 实现的是 |
|---|---|---|
| `LayerNorm` | 10 | 可选 bias（GPT-2 有、Llama 无） |
| `CausalSelfAttention` | 40 | `c_attn` 一次算 QKV · 拆头 · SDPA · `c_proj` |
| `MLP` | 12 | 768 → 3072 → GELU → 768 |
| `Block` | 10 | 上面两行 pre-norm |
| `GPT` | 200+ | 拼结构、共享权重、初始化、`generate`、`from_pretrained`、`configure_optimizers`、`estimate_mfu` |

- Llama 相对 GPT-2 只改五处：RMSNorm、RoPE、SwiGLU、GQA、去 bias（lm_head 不共享）
- 三个独立 Linear 与一个 `c_attn` 数学等价，合并只为一次 GEMM

---

## 04 · 手搓 GPT（下）：train.py 与训一个会续写的模型

**结论**：1.1 MB 莎士比亚、0.8M 参数、2000 步 7 分钟，loss 从 **ln 65 = 4.17 → 1.66**；层是串行的——参数与耗时线性、loss 收益递减。

![只改层数在 shakespeare_char 上 2000 步的 loss：2 / 4 / 8 层 val 1.82 / 1.66 / 1.59，每步耗时约 0.5× / 1× / 2×](/img/in-post/transformer-04-shakespeare-loss-by-depth.svg){: style="max-height: 400px"}

<aside class="notes" markdown="1">
原文 /nanogpt-train-py-and-training-a-model-that-writes.html。默认配置每次迭代 40 × 12 × 1024 = 491,520 token；8 卡时每卡累积 5 次。MFU 0.2%：小 batch 在 MPS 上 launch-bound。
</aside>

<!-- v -->

### 训练循环每一步在做什么

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 170}}}%%
flowchart LR
    LR["设 lr<br/>warmup + cosine"] --> EV["到点评估<br/>存 checkpoint 五样"] --> ACC["k 个 micro-batch<br/>loss ÷ k 累积"] --> CL["unscale<br/>+ 梯度裁剪"] --> ST["optimizer.step()<br/>zero_grad()"] --> LR
```

- `get_batch`：`memmap` + 随机窗口 + `stack`，`y` 是 `x` 右移一位，没有 epoch
- 不除累积步数 ≠ 只是尺度不同：等价于学习率放大 k 倍；DDP 只在最后一个 micro-batch 同步
- checkpoint 五样：model、optimizer、model_args、iter_num、config——续训不带优化器状态，前几百步抖

---
