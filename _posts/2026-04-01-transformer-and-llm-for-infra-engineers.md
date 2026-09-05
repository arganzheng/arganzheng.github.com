---
layout: post
title: Transformer 与 LLM：结构、算量与数值（总纲）
subtitle: "Transformers and LLMs for Infrastructure Engineers: Architecture, Arithmetic and Numerics"
tags: [Transformer, LLM, AI, AI-Infra]
catalog: true
---


## 内容简介

《Transformer 与 LLM：结构、算量与数值》是一组共七篇的系列文章，面向不训练模型、但要为模型搭建训练与推理系统的工程师，以及想知道自己的模型在硬件上"花多少钱"的算法工程师。它讲的是大语言模型的**成本结构**：每一层做多少次乘加、读多少字节、存多少状态，这些数字由哪些超参数决定，以及各种结构上和数值上的改动如何改变这些数字。

它回答的问题是：

> **Infra 工程师不训练模型，但必须知道自己在优化什么：这个模型的每一步算多少、读多少、存多少？**

Infra 工作的绝大多数决策——一张卡能不能放下这个模型、批要开到多大、KV cache 该留多少显存、量化到 INT4 能快多少、投机解码值不值得开、要不要为 MoE 上专家并行——答案都不在框架文档里，而在几行推导里。例如，Llama-3-8B 的 `config.json` 里有六个数字：

```text
hidden_size          4096
intermediate_size    14336
num_hidden_layers    32
num_attention_heads  32
num_key_value_heads  8
vocab_size           128256
```

从这六个数字出发，不需要运行任何代码，可以算出：

- 它有 8.03B 参数，BF16 权重 16.06 GB；
- 每生成一个 token 需要约 15 GFLOPs（不含 attention 对上下文的那部分），而这部分在上下文 8K 时再加 4.3 GFLOPs，128K 时加 68.7 GFLOPs——超过权重那部分；
- 每个 token 的 KV cache 占 128 KiB；如果它没有用 GQA，会是 512 KiB；
- 在一张 H100 上，batch 为 1 的 decode 每步至少要 4.8 ms，因为要把 16 GB 权重从 HBM 读一遍；要让 Tensor Core 忙起来，batch 得接近三百。

这些推导是本系列的全部内容。系列不由 API 驱动——不讲 `transformers` 库怎么用、不讲训练配方、不讲如何调 prompt；它由**推导**驱动：每一篇给出公式、代入公开模型的真实超参数、得到数字，再解释这个数字对系统设计意味着什么。

系列覆盖的范围可以用一句话概括——一个 LLM 的成本由四组变量决定，本系列逐一展开：

```text
结构变量    层数 · hidden · FFN 宽度 · head 数 · KV 头数 · 专家数与 top-k       → 第一、三、五篇
运行变量    batch · 上下文长度 · prefill 还是 decode                          → 第二、四篇
数值变量    每个数占几个字节 · 在哪一步累加 · 误差怎么积累                       → 第六篇
方法变量    量化格式 · 投机解码的草稿与接受率 · LoRA 的秩                      → 第七篇
```

读完之后，读者应该能把任何一个模型放进这四组变量里，算出它在任何一张 GPU 上的成本表。


## 为什么写这个系列？

### 引擎与 kernel 的一切优化都以模型的算量和访存量为目标

推理引擎的 continuous batching、PagedAttention、prefix caching、PD 分离，训练框架的张量并行、流水并行、专家并行、激活重算，kernel 层的 FlashAttention、融合算子、低精度 GEMM——每一项都是对模型某个成本项的回应。不知道 KV cache 是怎么算出来的，就不理解 PagedAttention 在管理什么；不知道 decode 为什么是 memory-bound 的，就不理解为什么 weight-only 量化对 decode 有效、对 prefill 无效；不知道 MoE 每层只激活 8/256 的专家，就不理解为什么它必须做 all-to-all 而不是 all-reduce。

反过来，掌握这些推导之后，面对一个新模型、新硬件、新的优化方法，可以在看任何 benchmark 之前先算出它**应该**多快，再用测量去解释差距。这是本系列想给读者的东西。

### "参数量"只是成本的一个维度

模型卡上通常只有一个数字：参数量。但一个 8B 的 dense 模型、一个 47B 总参数 13B 激活参数的 MoE 模型、一个 671B 总参数 37B 激活参数的 MoE 模型，它们的显存、算量、访存量、通信量之间的关系完全不同：

```text
              参数量决定    激活参数量决定    上下文长度决定    batch 决定
显存           权重          —               KV cache         KV cache · 激活值
算量（FLOPs）  —            每 token 的 GEMM   attention 项      总量
访存量         decode 的下界  —               KV 读取           摊薄权重读取
通信量         并行切分      MoE 的 all-to-all  序列并行          —
```

每一格都有自己的公式，公式里的变量各不相同。本系列把这张表的每一格填上。

### 数值不是算法工程师才需要关心的事

BF16 为什么能训练而 FP16 需要 loss scaling？混合精度里的 FP32 master weights 为什么不能省？FP8 训练为什么要分块量化和高精度累加？INT4 量化为什么不损失太多精度？这些问题一半是数学，一半是硬件：浮点格式的位布局决定了动态范围和精度，Tensor Core 的累加精度决定了哪些误差会积累。Infra 工程师经常是第一个发现 loss 变 NaN、量化后输出乱码、两种 kernel 数值对不上的人，而排查这些问题需要知道数值在哪一步、以什么方式丢失。

### 现有材料的断层

- **论文**给出了每个方法的定义和实验结果，但假设读者会自己算成本，而且每篇只讲一个方法，读者需要自己把它们放进同一个坐标系；
- **框架文档**（PyTorch、`transformers`、vLLM、Megatron）讲怎么用，把模型当作黑盒；
- **算法课程**讲 attention 的直觉和训练方法，很少讲一个矩阵乘法在 GPU 上花多少字节；
- **博客**里的"Transformer 数学"多数止步于参数量和 $$6ND$$，不覆盖 MLA、MoE、FP8、投机解码这些当前实践的重心。

本系列想填补的是从"知道 Transformer 长什么样"到"能为任何一个模型算出一张完整成本表"之间的那段路。它的取法是：**每一个方法都放回同一张成本表里讨论，用同样三个模型、同一张 GPU 的数字，让不同方法之间可以直接比较**。


## 适合哪些读者？

### 做推理系统与训练基础设施的工程师

你在配置或改造 vLLM、SGLang、Megatron、DeepSpeed 这类系统，需要判断：这个模型在这几张卡上该怎么切、batch 和上下文的上限在哪、开哪种量化、投机解码的收益上界是多少、MoE 该用 EP 还是 TP。本系列给的是你做这些判断时要用的公式和数字。

### 写 kernel 的工程师

你要优化的 attention、GEMM、MoE、量化 kernel 的输入 shape 和访存模式都来自模型结构。理解 GQA 的组数如何改变 decode attention 的算术强度、MLA 的矩阵吸收如何把 attention 变成一个大 head dim 的 MQA、MoE 的 grouped GEMM 的 M 维为什么这么小，才知道该往哪个方向优化。

### 想知道自己的模型在硬件上花多少钱的算法工程师

你在设计或调整模型结构：选 GQA 组数、选 MoE 的专家粒度、选 RoPE 的 base、决定是否用 MLA。本系列从系统角度告诉你每个选择在显存、算量、访存、通信上的代价，让结构设计和硬件效率在同一张表上讨论。

### 从后端转向 AI-Infra、需要一份"模型知识最小集"的工程师

你不打算成为算法工程师，但读引擎源码、看性能报告、参加技术讨论时，需要知道 head、layer、KV cache、prefill、decode、MoE、FP8 这些词背后的数量关系。本系列是为此准备的最小集：它不讲训练模型的方法，只讲模型作为一个计算对象的结构和成本。


## 系列的整体主线

七篇文章按"先建立成本模型，再看每一种结构和数值上的改动如何改变它"的顺序推进：

```text
第一篇：Transformer 解剖与参数量 —— 每个矩阵的形状，从 config.json 算出 8.03B
        ↓
第二篇：前向的算量与访存量 —— FLOPs、字节数、Roofline 视角下的 prefill 与 decode
        ↓
第三篇：Attention 变体与 KV cache —— MHA / GQA / MQA / MLA 的推导
        ↓
第四篇：位置编码与长上下文 —— RoPE 的波长、外推方法与长上下文的代价
        ↓
第五篇：MoE —— 路由、激活参数量与 all-to-all 的通信形态
        ↓
第六篇：浮点格式、数值稳定性与混合精度 —— 数值在哪里丢失，为什么还能工作
        ↓
第七篇：量化、投机解码与 LoRA —— 三种改变计算形态的方法及其数学
```

前两篇建立成本模型：一个 dense Transformer 在给定超参数下的参数量、FLOPs、字节数。第三到五篇是**结构**上的改动：分别改变 KV cache、上下文长度、参数与激活参数的比例。第六、七篇是**数值**上的改动：改变每个数占几个字节，以及绕过 decode 串行瓶颈的方法。

三条交织的线索：

```text
成本线：参数量 → FLOPs 与字节数 → KV cache → 长上下文的二次项 → 激活参数与通信量 → 字节/数 → 量化后的字节数
硬件线：Roofline 与 ridge point → decode 的 memory-bound → 多卡的通信 → Tensor Core 的累加精度 → 低精度 GEMM 的收益区间
模型线：Llama-3-8B / 70B（dense、GQA）→ DeepSeek-V3（MLA、MoE、FP8）→ Mixtral 8x7B（粗粒度 MoE）
```

每一篇都用同样的方法：**写出公式，代入真实模型的超参数，算出数字，解释数字对系统意味着什么**。


## 章节结构与分章导读

### 1. Transformer 解剖与参数量：从 config.json 算出 8.03B

第一篇建立整个系列的对象：一个 decoder-only Transformer 里到底有哪些矩阵、每个矩阵的形状由哪个超参数决定。它不讲 attention 的直觉和训练方法，只把结构拆到能数出每一个参数的粒度。

这一篇会覆盖：

- decoder-only Transformer 的整体结构：embedding → N 个相同的 layer → 最终 RMSNorm → lm_head；pre-norm 与 post-norm 的区别；
- 一个 layer 里的两个子层：attention 与 FFN，各自的输入输出形状；残差连接为什么让所有子层的输出维度都必须是 $$d$$；
- attention 的四个投影矩阵 $$W_Q$$、$$W_K$$、$$W_V$$、$$W_O$$ 的形状，`num_attention_heads`、`num_key_value_heads`、`head_dim` 三者的关系；
- FFN 的形状：传统两矩阵 FFN 与 SwiGLU 的三矩阵 FFN（gate、up、down），为什么 Llama 的 `intermediate_size` 是 14336 而不是 $$4d = 16384$$；
- RMSNorm 与 LayerNorm 的参数量与计算量；为什么 bias 项在近年的模型里几乎消失；
- embedding 与 lm_head 是否共享（tie），vocab 大小对参数量和 lm_head 计算量的影响；
- 参数量公式：

$$
N \approx L \cdot \left[ d \cdot (d + 2 d_{kv} + d) + 3 \cdot d \cdot d_{ff} \right] + 2 \cdot V \cdot d
$$

  其中 $$d_{kv} = n_{kv} \cdot d_{head}$$；逐项代入 Llama-3-8B：attention 每层 41.9M，FFN 每层 176.2M，32 层共 6.98B，embedding 与 lm_head 各 525M，合计 8.03B；再代入 Llama-3-70B（$$d = 8192$$，$$d_{ff} = 28672$$，80 层，64 头，8 个 KV 头）得到 70.6B；
- 参数分布：dense 模型里 FFN 占每层参数的约 80%，embedding 在小模型里占比很高（8B 的 13%）而在大模型里可以忽略；
- 读 `transformers` 的 `modeling_llama.py`：把每个 `nn.Linear` 的 `in_features`/`out_features` 与上面的公式一一对应；
- 训练与推理时的 shape：`[batch, seq, hidden]` 在每个矩阵乘法处变成什么 GEMM。

核心问题是：

> **给你任意一个模型的 `config.json`，不运行代码，能不能在五分钟内算出它的参数量，并说出这些参数在 attention、FFN、embedding 之间怎么分配？误差要在 1% 以内。**

实践：写一个读 `config.json` 输出逐层参数表的脚本，用 Llama-3-8B、Llama-3-70B 验证到与官方公布的参数量一致。这个脚本会在后面每一篇里长出新的列。

### 2. 前向的算量与访存量：prefill、decode 与 Roofline

第二篇把第一篇的参数表变成成本表。它回答：跑一次前向要做多少浮点运算、从 HBM 读多少字节，以及这两个数的比值如何决定一段计算是 compute-bound 还是 memory-bound。

这一篇会覆盖：

- 矩阵乘法的 FLOPs：$$[m, k] \times [k, n]$$ 需要 $$2mkn$$ 次浮点运算；为什么每个参数每个 token 贡献 2 FLOPs，得到前向的经典近似 $$2N$$ FLOPs/token；
- attention 对上下文的那部分：$$QK^\top$$ 与 $$PV$$ 每层每 token 各 $$2 \cdot n_h \cdot d_{head} \cdot s = 2ds$$ FLOPs，合计 $$4ds$$；对 Llama-3-8B 这是每层 $$16384 \cdot s$$，32 层共 $$0.52\,\text{MFLOPs} \times s$$；上下文 8K 时 4.3 GFLOPs，128K 时 68.7 GFLOPs，与权重部分的 15 GFLOPs 对比；
- 训练的 $$6ND$$：前向 $$2N$$，反向约 $$4N$$（对输入的梯度和对权重的梯度各一次），乘以 token 总数；激活重算再加一个前向；
- prefill 与 decode 的区别：prefill 一次处理 $$s$$ 个 token，GEMM 的 $$m$$ 维是 $$s$$；decode 每步处理 1 个 token，$$m$$ 维是 batch 大小；
- 访存量：权重每步必须读一遍——Llama-3-8B BF16 是 16.06 GB；KV cache 每步读一遍——每个 token 128 KiB 乘以上下文长度乘以 batch；激活值在 decode 时可以忽略；
- Roofline：算术强度 $$I = \text{FLOPs} / \text{bytes}$$；H100 SXM 的 ridge point 约 $$989 / 3.35 \approx 295$$ FLOP/byte（BF16 dense 算力 989 TFLOPS，HBM3 带宽 3.35 TB/s），A100 约 156；
- decode 的算术强度：batch 为 $$B$$ 时，权重 GEMM 的强度约为 $$B$$ FLOP/byte（每 2 字节权重做 $$2B$$ 次运算）；$$B = 1$$ 时距 ridge point 差两个数量级——这就是"decode 是 memory-bound 的"的全部含义；
- decode 每步的时间下界：Llama-3-8B 在 H100 上 $$16.06\,\text{GB} / 3.35\,\text{TB/s} \approx 4.8$$ ms，即单请求最多约 200 token/s；加上 KV cache：上下文 8K、batch 64 时 KV 读取 64 GiB，已远超权重；
- prefill 的时间下界：8K 个 token 约 $$8192 \times 19\,\text{GFLOPs} \approx 156$$ TFLOP，按 60% 的 MFU 约 0.26 s；这是 TTFT 的物理下限；
- 激活值显存：训练时每层每 token 的激活值随 $$d$$、$$s$$、head 数变化的估算式（Megatron 团队论文中的 $$s b h (34 + 5 a s / h)$$ 字节，不用 FlashAttention 时），以及为什么 $$s^2$$ 项让长序列训练必须重算或用 FlashAttention；
- MFU 与 HFU：如何从 token 吞吐反推硬件利用率，为什么 40–50% 的 MFU 已经算好。

核心问题是：

> **Llama-3-8B 在一张 H100 上，batch 多大时 decode 从 memory-bound 变成 compute-bound？考虑 KV cache 之后，这个 batch 还能达到吗？**

实践：脚本增加 FLOPs 与字节数两列，输入 batch、上下文长度和硬件参数，输出 prefill 和 decode 的理论时间下界；与 vLLM 或 `transformers` 实测对比，解释差距。

### 3. Attention 变体与 KV cache：MHA、GQA、MQA 与 MLA 的推导

第三篇专门讲 attention，因为它是 Transformer 里唯一成本随上下文长度增长的部分，也是过去几年结构改动最集中的地方。每一种变体都是在同一个目标下做取舍：**减少每个 token 的 KV cache 字节数，同时尽量不损失质量**。

这一篇会覆盖：

- 为什么需要 KV cache：自回归 decode 时每个新 token 要 attend 到全部历史 token 的 K 和 V，不缓存就要重算，缓存就要占显存；
- KV cache 大小公式：

$$
\text{bytes/token} = 2 \cdot L \cdot n_{kv} \cdot d_{head} \cdot \text{bytes/elem}
$$

  MHA（$$n_{kv} = n_h$$）：Llama-3-8B 如果是 MHA，每 token $$2 \times 32 \times 32 \times 128 \times 2 = 512$$ KiB；
- MQA（Shazeer 2019）：$$n_{kv} = 1$$，KV cache 缩小 $$n_h$$ 倍，但质量有损；
- GQA（Ainslie 等 2023）：$$n_{kv}$$ 取中间值，Llama-3-8B 的 8 个 KV 头把 KV cache 压到 128 KiB/token，Llama-3-70B 为 320 KiB/token；128K 上下文时两者分别是 16 GiB 和 40 GiB；一张 80 GB 的 H100 放下 8B 权重后剩余约 64 GB，能容纳约 50 万个 token 的 KV；
- GQA 对 decode attention 算术强度的影响：每读一个 KV 元素服务 $$g = n_h / n_{kv}$$ 个 query head，强度从 MHA 的约 1 FLOP/byte 提到约 $$g$$；这是 GQA 除了省显存之外的第二个收益；
- MLA（DeepSeek-V2 / V3）：把 K 和 V 联合压缩到一个 $$d_c = 512$$ 维的 latent，外加一个 $$d_h^R = 64$$ 维的解耦 RoPE key；每 token 每层只缓存 $$512 + 64 = 576$$ 个数；DeepSeek-V3 的 61 层 BF16 下每 token 约 68.6 KiB——尽管它有 128 个 head，KV cache 比 Llama-3-8B 还小；如果它用 MHA，每 token 是 3.8 MiB，压缩了约 57 倍；
- MLA 的矩阵吸收：推理时把 $$W_{UK}$$ 吸收进 $$W_Q$$、$$W_{UV}$$ 吸收进 $$W_O$$，attention 直接在 576 维的 latent 上做——它在 kernel 层等价于一个 head dim 为 576（K）/ 512（V）、128 个 query head 共享一个 KV 头的 MQA，算术强度极高，但每个 head 的点积长度从 192 变成 576，计算量上升；为什么这对 decode 是划算的，对 prefill 不一定；
- 为什么 RoPE 与低秩压缩不兼容，MLA 要把 RoPE 部分解耦出来单独缓存；
- 标准 attention 的中间结果：$$S = QK^\top$$ 对 $$s = 8K$$、32 个 head 的 BF16 是每层 4 GiB；FlashAttention 通过分块与 online softmax 不物化它，HBM 流量从 $$O(s^2)$$ 降到 $$O(s^2 d^2 / M)$$（$$M$$ 为 SRAM 大小）——本篇只推导 IO 复杂度，不讲 kernel 实现；
- 因果掩码让 prefill 的 attention 实际算量减半，sliding window 让它变成线性；
- KV cache 的分页、prefix 共享、量化（FP8 / INT8 KV）对上面公式的影响。

核心问题是：

> **DeepSeek-V3 有 128 个 attention head、61 层，KV cache 却比 32 头 32 层的 Llama-3-8B 小。这是怎么做到的？代价是什么？**

实践：脚本增加 KV cache 列，支持 MHA / GQA / MQA / MLA 四种模式；给定显存预算，输出各模型在不同上下文长度下的最大并发数。

### 4. 位置编码与长上下文：RoPE 的波长、外推与代价

第四篇讲位置编码，以及它与"上下文能有多长"的关系。上下文长度是当前模型能力竞争的重要维度，也是 Infra 成本最敏感的维度：它同时进入 KV cache 的一次项和 attention 算量的二次项。

这一篇会覆盖：

- 为什么 attention 本身是置换不变的，位置信息必须显式注入；绝对位置编码（正弦、可学习）与相对位置编码的区别；
- RoPE（Su 等 2021）的推导：把 $$d_{head}$$ 维向量看成 $$d_{head}/2$$ 个复数，第 $$i$$ 对以角频率 $$\theta_i = \text{base}^{-2i/d_{head}}$$ 旋转；$$q_m^\top k_n$$ 只依赖 $$m - n$$；为什么它能以绝对位置的实现得到相对位置的性质；
- 每个维度对的波长 $$\lambda_i = 2\pi \cdot \text{base}^{2i/d_{head}}$$：base 为 10000、$$d_{head} = 128$$ 时最低频维度的波长约 5.4 万，超过它的位置在训练中从未被完整"转过一圈"，这是 RoPE 外推失败的根源；Llama 3 把 base 提到 500000 之后，最低频波长约 250 万；
- 长上下文扩展方法的数学：Position Interpolation（把位置压缩到训练范围内）、NTK-aware 插值（改 base 而非改位置）、YaRN（按波长分段处理高频与低频维度，并修正 attention 温度）；DeepSeek-V3 与 Qwen 系列用 YaRN、Llama 3.1 用自己的分段缩放，各自的参数意义；
- ALiBi（Press 等 2021）：不旋转，直接给 attention 分数加线性惩罚；为什么它外推好但在长上下文上被 RoPE 取代；
- 长上下文的成本：Llama-3-70B 在 128K 上下文下，每 token 的 attention 算量 $$4 \times 8192 \times 131072 \times 80 \approx 344$$ GFLOPs，超过权重部分的 141 GFLOPs；一次 128K 的 prefill 在 Llama-3-8B 上约 $$2.0 + 4.5 = 6.5$$ PFLOPs（权重项 $$15\,\text{GFLOPs} \times 131072$$，attention 项按因果掩码取 $$s^2/2$$），H100 上以 60% MFU 约 11 s；
- 缩短这个成本的结构手段：sliding window attention（Mistral 7B 的 4096 窗口）、全局层与局部层交错（Gemma 2 一类的设计）、attention sink 与 StreamingLLM、稀疏 attention 的形态；每种手段把 KV cache 与 attention 算量各改成什么函数；
- 长上下文对 Infra 的全部影响汇总：KV cache 的线性项、prefill 的二次项、单请求的 TTFT、chunked prefill 的必要性、序列并行的动机。

核心问题是：

> **一个用 8K 上下文训练的 RoPE 模型，为什么不能直接推理 32K？把 base 从 10000 改到 500000 解决了什么，没解决什么？**

实践：用 NumPy 实现 RoPE，画出各维度对的波长；对 Position Interpolation、NTK-aware、YaRN 三种方法画出扰动后的频率分布；脚本增加"上下文长度 → KV cache、prefill FLOPs、attention 占比"三列。

### 5. MoE：路由、激活参数量与通信形态

第五篇讲混合专家模型。MoE 把"参数量"与"每 token 算量"解耦，是当前大模型扩展的主流路线；它也把一个新的成本项——**all-to-all 通信**——引入了模型前向。

这一篇会覆盖：

- MoE layer 的结构：router（一个 $$d \times E$$ 的线性层加 softmax 或 sigmoid）为每个 token 选 top-$$k$$ 个专家，专家是独立的 FFN，输出按路由权重加权求和；
- 两种粒度：Mixtral 8x7B 是 8 个 $$d_{ff} = 14336$$ 的专家取 top-2，DeepSeek-V3 是 256 个 $$d_{ff} = 2048$$ 的路由专家取 top-8、外加 1 个共享专家；细粒度专家为什么在同样激活参数下表达能力更强；
- 参数量与激活参数量：Mixtral 8x7B 总参数 46.7B（8 个专家每层 1.41B，32 层），每 token 激活 12.9B；DeepSeek-V3 每个专家 $$3 \times 7168 \times 2048 \approx 44$$M，每个 MoE 层 257 个专家共 11.3B，58 个 MoE 层加 3 个 dense 层与 attention、embedding 合计约 671B，每 token 激活 37B；
- 算量按激活参数算，显存按总参数算：DeepSeek-V3 每 token 约 74 GFLOPs（$$2 \times 37\text{B}$$），但 FP8 权重也要 671 GB，一台 8 卡 H100（640 GB）放不下；
- decode 时的访存形态：batch 为 $$B$$ 时期望被激活的专家数为 $$E \cdot [1 - (1 - k/E)^B]$$，DeepSeek-V3 在 $$B = 32$$ 时约 163 个、$$B = 128$$ 时约 252 个——中等 batch 就几乎要把所有专家的权重读一遍，"稀疏"在访存上不成立；这是它的推理部署要用大规模专家并行的原因；
- 专家并行（EP）的通信：每个 token 的 hidden state 要发到它的 $$k$$ 个专家所在的 GPU，再收回来——两次 all-to-all；DeepSeek-V3 dispatch 用 FP8（每 token 每专家 7 KiB）、combine 用 BF16（14 KiB），并限制每个 token 最多路由到 4 个节点以控制跨节点流量；
- EP 与 TP 的对比：TP 切每个专家的矩阵，通信是 all-reduce，量与 dense 相同；EP 按专家切，通信是 all-to-all，量与 $$k$$ 成正比；两者在什么规模下哪个划算；
- grouped GEMM 的形态：$$T$$ 个 token 分到 $$E$$ 个专家后，每个专家的 GEMM 平均只有 $$Tk/E$$ 行——prefill 4096 个 token 在 DeepSeek-V3 里每个专家平均 128 行，decode 时可能只有几行；为什么这让 MoE 的 GEMM 效率天然低于 dense；
- 负载均衡：辅助损失（Switch Transformer）、容量因子与 token 丢弃、DeepSeek-V3 的 aux-loss-free 偏置调节；负载不均对 EP 意味着什么（最慢的 GPU 决定这一层的时间）；
- 与 MoE 配套的其他结构：共享专家、多 token 预测（MTP）作为一种内置的投机解码草稿。

核心问题是：

> **DeepSeek-V3 每 token 只算 37B 参数，为什么部署它比部署一个 dense 70B 难得多？把"参数量"、"激活参数量"、"每步实际读取的参数量"三个数分开算。**

实践：脚本增加 MoE 支持——总参数、激活参数、给定 batch 下期望激活的专家数、EP 下每层的 all-to-all 字节数；用 Mixtral 8x7B 与 DeepSeek-V3 的公开超参验证。

### 6. 浮点格式、数值稳定性与混合精度

第六篇从"每个数占几个字节"进入"每个字节里存了什么"。前五篇的所有字节数都以 BF16 的 2 字节为默认；这一篇解释为什么是 BF16，以及把它换成 FP16、FP8、INT8 时数值上会发生什么。

这一篇会覆盖：

- 浮点格式的位布局：FP32（1/8/23）、TF32（1/8/10，19 位有效）、FP16（1/5/10）、BF16（1/8/7）、FP8 E4M3（1/4/3）与 E5M2（1/5/2）、INT8、INT4；每种格式的最大值、最小正规数、机器精度：FP16 最大 65504、BF16 与 FP32 同范围但相对精度只有 $$2^{-8}$$ 量级、E4M3 最大 448 且没有 inf；
- 指数位与尾数位的取舍：范围与精度不可兼得，BF16 用范围换精度，FP16 反之；为什么深度学习几乎总是选范围；
- 数值在哪些地方丢失：加法中大数吃小数、长求和的误差积累、softmax 的指数溢出（FP16 下 $$e^x$$ 在 $$x > 11.09$$ 时溢出，所以要先减最大值）、方差计算中的相消；
- 累加精度：一个 $$k = 4096$$ 的点积如果在 BF16 中累加会损失多少位；Tensor Core 用 FP32 累加器的原因；FP8 Tensor Core 的累加精度有限，DeepSeek-V3 每 128 个元素就把部分和提升到 FP32 的做法；
- 混合精度训练（Micikevicius 等 2017）为什么能工作：前向和反向用低精度、权重更新用 FP32 master weights；Adam 的单步更新量常在 $$10^{-4}$$ 到 $$10^{-3}$$ 的相对量级，低于 BF16 的机器精度 $$2^{-8} \approx 0.0039$$，直接用 BF16 累加会把更新吃掉；
- FP16 的 loss scaling：梯度的动态范围与 FP16 的最小正规数 $$6.1 \times 10^{-5}$$ 之间的矛盾，动态 loss scale 的机制；BF16 为什么不需要它；
- 训练状态的字节数：混合精度 + Adam 下每参数 16 字节（BF16 权重 2 + BF16 梯度 2 + FP32 master 4 + FP32 一阶矩 4 + FP32 二阶矩 4），Llama-3-8B 全量训练的状态就是 128 GB，这是 ZeRO 和 FSDP 存在的理由；
- FP8 训练：E4M3 用于前向与权重、E5M2 用于梯度的分工；per-tensor scaling 与 delayed scaling（Transformer Engine）；DeepSeek-V3 的分块量化（激活按 $$1 \times 128$$、权重按 $$128 \times 128$$）为什么能用 E4M3 训练 671B；
- 推理中的数值：attention logit 随训练增长、QK-norm 的作用；RMSNorm 的 $$\epsilon$$；不同 kernel 实现之间的数值差异应该有多大，怎么判断一个差异是 bug 还是正常的浮点噪声；
- 随机性与可复现：atomicAdd 的非确定性、`torch.use_deterministic_algorithms` 的代价。

核心问题是：

> **BF16 的相对精度只有 FP16 的 1/8，为什么它反而成了训练的默认格式？把它同时用在权重更新上会出什么问题？**

实践：用 NumPy / PyTorch 逐位构造各种格式的数，验证最大值、最小值和机器精度；模拟一个 BF16 权重更新被吃掉的过程；对同一个 GEMM 用 FP32 / BF16 / FP8 计算并度量误差随 $$k$$ 的增长。

### 7. 量化、投机解码与 LoRA：改变计算形态的三种方法

最后一篇讲三种在不改变模型结构的前提下改变其计算形态的方法。它们分别攻击前面算出的三个成本项：量化减少权重字节数，投机解码绕过 decode 的串行瓶颈，LoRA 把微调的状态从 16 字节/参数降到几乎为零。

这一篇会覆盖：

- 量化的基本形式：$$\hat{w} = s \cdot \text{round}(w / s) + z$$；per-tensor、per-channel、per-group（group size 128）的粒度与元数据开销——INT4 + group 128 的 FP16 scale 与 zero 约合 4.25 bit/权重，Llama-3-70B 约 40 GB，能放进一张 H100；
- weight-only 量化（W4A16）的收益区间：decode 时权重字节数降为 1/4，时间下界随之降为 1/4；prefill 时计算仍在 BF16 Tensor Core 上做，反量化是纯开销——同一个量化格式在两个阶段的收益符号相反，用第二篇的 Roofline 解释；
- GPTQ（Frantar 等 2022）：基于 Hessian 的逐列量化与误差补偿，为什么它需要校准数据，代价是什么；
- AWQ（Lin 等 2023）：按激活幅度找出约 1% 的显著权重通道，用逐通道缩放保护它们；
- SmoothQuant（Xiao 等 2022）：激活里的离群通道让 W8A8 难做，用 $$s_j = \max\lvert X_j\rvert ^\alpha / \max\lvert W_j\rvert ^{1-\alpha}$$ 把激活的难度迁移到权重上；LLM.int8() 对离群值的另一种处理；
- FP8 推理量化：per-tensor / per-token / per-block scale 的区别，为什么 FP8 比 INT8 对离群值更宽容；W8A8 与 W4A16 在 prefill 与 decode 上各自的位置；
- KV cache 量化：INT8 / FP8 KV 把第三篇的数字再减半；
- 投机解码（Leviathan 等 2023；Chen 等 2023）的数学：小模型起草 $$\gamma$$ 个 token，大模型一次前向验证；接受概率 $$\min(1, p(x)/q(x))$$、拒绝后从 $$\text{norm}(\max(0, p - q))$$ 重采样，为什么输出分布与大模型严格一致；期望每轮接受的 token 数

$$
\mathbb{E}[\text{tokens}] = \frac{1 - \alpha^{\gamma + 1}}{1 - \alpha}
$$

  $$\alpha = 0.8$$、$$\gamma = 4$$ 时为 3.36；在 memory-bound 的 decode 里验证 $$\gamma + 1$$ 个 token 的成本与验证 1 个几乎相同，所以加速比约为 $$\mathbb{E}[\text{tokens}] / (\gamma c + 1)$$，草稿成本 $$c = 0.1$$ 时约 2.4 倍；batch 变大、decode 逼近 compute-bound 之后这个"几乎免费"不再成立，加速比随之消失；
- 草稿从哪里来：独立小模型、Medusa 的多头、EAGLE 的特征级草稿、n-gram / prompt lookup、DeepSeek-V3 的 MTP 头；各自的 $$\alpha$$ 与 $$c$$ 大致在什么区间；
- LoRA（Hu 等 2021）：$$W + BA$$，$$A \in \mathbb{R}^{r \times d_{in}}$$、$$B \in \mathbb{R}^{d_{out} \times r}$$；Llama-3-8B 在 q/k/v/o 上用 $$r = 16$$ 的 LoRA 只有 13.6M 可训练参数（0.17%），加上 FFN 三个矩阵约 41.9M（0.52%）；训练状态从全量的 128 GB 降到 BF16 冻结权重的 16 GB 加可忽略的 LoRA 状态，但激活值显存不变；
- LoRA 的计算形态：额外 FLOPs 约为原矩阵的 $$r(d_{in} + d_{out}) / (d_{in} d_{out})$$，$$W_Q$$ 上不到 1%；推理时可以合并回 $$W$$ 零开销，也可以不合并以支持多租户——多 LoRA 服务的 batched GEMV（Punica、S-LoRA）形态；
- QLoRA：NF4 量化的冻结底座加 BF16 的 LoRA，8B 模型的底座压到约 4.5 GB。

核心问题是：

> **同一个 INT4 量化模型，decode 快 3 倍，prefill 反而慢；同一套投机解码，batch 1 时加速 2 倍，batch 64 时没有收益。这两个现象背后是同一条 Roofline。**

实践：脚本增加量化后的字节数、投机解码的期望加速、LoRA 的参数与状态三组输出，完成最终的成本表；用 vLLM 加载 BF16 与 INT4 两个版本的同一模型，在 batch 1 和 batch 64 下实测 decode 与 prefill 吞吐，与推导对照。本篇最后给出全系列总结。


## 贯穿全系列的实践线

本系列的贯穿物是**一张成本表和一组生成它的推导脚本**。脚本从第一篇的参数量开始，每篇增加几列，到第七篇结束时可以为任何一个给出 `config.json` 的模型、任何一组硬件参数输出：

```text
第一篇    参数量                 逐层、逐矩阵；attention / FFN / embedding 的分布
第二篇    FLOPs · 字节数          prefill 与 decode 的理论时间下界；Roofline 位置
第三篇    KV cache               MHA / GQA / MQA / MLA；给定显存的最大并发
第四篇    长上下文               上下文长度 → KV cache、prefill FLOPs、attention 占比
第五篇    MoE                    总参数 · 激活参数 · 期望激活专家数 · all-to-all 字节数
第六篇    精度                   各格式的字节数与训练状态；误差随累加长度的增长
第七篇    量化 · 投机 · LoRA      量化后字节数；期望加速比；LoRA 参数与状态
```

三个模型贯穿全部七篇：**Llama-3-8B** 与 **Llama-3-70B** 代表 dense + GQA 的主流结构，**DeepSeek-V3** 代表 MLA + 细粒度 MoE + FP8 的另一条路线；Mixtral 8x7B 在 MoE 一篇作为粗粒度专家的对照。每篇算出的数字都会填进同一张表，读者在第七篇结束时手上有一张三个模型在 H100 上的完整成本对照。表的骨架大致如下（BF16，H100 SXM，数字为理论值）：

```text
                        Llama-3-8B        Llama-3-70B       DeepSeek-V3
参数量                   8.03B             70.6B             671B（激活 37B）
权重字节数（BF16）        16.1 GB           141 GB            1342 GB（FP8 为 671 GB）
每 token 权重 FLOPs      ~15 GFLOPs        ~141 GFLOPs       ~74 GFLOPs
KV cache / token         128 KiB           320 KiB           68.6 KiB
128K 上下文的 KV cache    16 GiB            40 GiB            8.6 GiB
batch 1 decode 时间下界   4.8 ms（单卡）     不能单卡           不能单卡
```

脚本的价值不在这几个数字本身，而在换一个模型、换一张卡、换一种精度之后能立刻重算。

与它平行的源码与资料阅读线：

```text
第一篇    transformers  modeling_llama.py · Llama-3 的 config.json
第二篇    Kaplan 等 2020 与 Hoffmann 等 2022（scaling laws）的 FLOPs 估算；Korthikanti 等 2022（激活重算）
第三篇    Shazeer 2019（MQA）· Ainslie 等 2023（GQA）· DeepSeek-V2 论文的 MLA 章节 · FlashAttention 论文的 IO 复杂度分析
第四篇    Su 等 2021（RoPE）· Chen 等 2023（Position Interpolation）· Peng 等 2023（YaRN）· Press 等 2021（ALiBi）
第五篇    Fedus 等 2021（Switch Transformer）· Mixtral 与 DeepSeek-V3 的技术报告 · transformers 的 modeling_deepseek_v3.py
第六篇    Micikevicius 等 2017（混合精度）· Micikevicius 等 2022（FP8 格式）· DeepSeek-V3 技术报告的 FP8 训练章节
第七篇    Frantar 等 2022（GPTQ）· Lin 等 2023（AWQ）· Xiao 等 2022（SmoothQuant）· Leviathan 等 2023（投机解码）· Hu 等 2021（LoRA）
```


## 阅读路径建议

### 完整学习路径

```text
1 → 2 → 3 → 4 → 5 → 6 → 7
```

### 做推理系统，最关心显存与吞吐

```text
1 → 2 → 3 → 7
```

参数量、算量与访存量、KV cache、量化与投机解码，是推理系统容量规划的四块。第四篇在需要支持长上下文时补读。

### 做训练基础设施，最关心状态与通信

```text
1 → 2 → 5 → 6
```

第二篇的 $$6ND$$ 与激活值、第五篇的 MoE 通信形态、第六篇的混合精度与训练状态字节数是训练侧的核心。

### 写 kernel，需要知道输入 shape 从哪里来

```text
1 → 2 → 3 → 5 → 7
```

GEMM 的 $$m, k, n$$、attention 的 head 数与 head dim、MoE grouped GEMM 的行数、量化 GEMM 的收益区间，都在这几篇。

### 算法工程师，想知道结构选择的硬件代价

```text
2 → 3 → 4 → 5
```

先有 Roofline，再看 GQA 组数、MLA、RoPE base、专家粒度分别改变了成本表的哪一格。

### 排查数值问题

```text
6 → 7
```

loss 变 NaN、量化后输出异常、两个 kernel 的结果对不上——第六篇给出数值丢失的位置和判断标准，第七篇给出各量化方法各自的误差来源。前五篇按需回查。


## 本系列的边界

本系列只讨论模型作为一个**计算对象**的结构与成本。以下内容与它紧邻，但不在范围内：

- **训练方法与算法**：预训练配方、数据配比、SFT、RLHF / DPO、评测。本系列只算训练要多少 FLOPs 和多少字节状态，不讲怎么把 loss 降下去。
- **kernel 实现**：FlashAttention 的分块与 online softmax 如何写、量化 GEMM 如何反量化、MoE 的 permute 与 grouped GEMM 如何实现。本系列只推导它们的 IO 复杂度与收益区间，把实现当作黑盒。
- **推理引擎的调度与内存管理**：continuous batching、PagedAttention 的 block 管理、prefix caching、PD 分离。本系列给出这些机制所依据的数字，不讲机制本身。
- **分布式并行的实现**：TP / PP / EP / 序列并行如何切分与同步、集合通信的算法。本系列在 MoE 一篇讨论 EP 的通信**量**，不讨论通信**怎么做**。
- **框架 API**：`transformers`、PyTorch、vLLM 的使用方式。实践部分会调用它们做验证，但不解释它们。
- **非 Transformer 结构**：状态空间模型（Mamba 一类）、线性 attention、扩散模型。它们改变了成本结构的基本形态，值得单独讨论，不进入本系列。
- **多模态**：视觉编码器、音频前端。本系列只讨论文本 decoder。


## 前置要求与说明

### 前置要求

- 线性代数基础：矩阵乘法的定义与形状规则、向量的点积与范数；
- 会读 Python 与 PyTorch 代码：能看懂 `nn.Linear`、`torch.matmul`、`softmax` 的调用，能跑一个 `transformers` 模型的前向；
- 知道 GPU 有算力和带宽两个上限，见过"memory-bound / compute-bound"这两个词（第二篇会从头建立 Roofline，不假设读者用过）；
- 对 Transformer 有最基本的印象：知道它由 attention 和 FFN 堆叠而成，看过 attention 的公式 $$\text{softmax}(QK^\top / \sqrt{d}) V$$。

不要求：

- 训练过模型；
- 了解 GQA、MLA、MoE、RoPE、FP8、GPTQ 等任何一个具体方法；
- 写过 CUDA；
- 有 GPU。全部推导可以在纸上完成；实践部分的验证需要一张能放下 8B 模型的 GPU，没有也不影响阅读。

### 版本与模型基线

- 模型：以 **Llama-3-8B / 70B**（Llama 3 与 3.1 结构相同，$$d = 4096 / 8192$$，32 / 80 层，GQA 8 个 KV 头，$$d_{head} = 128$$，vocab 128256）和 **DeepSeek-V3**（$$d = 7168$$，61 层，128 头，MLA 的 $$d_c = 512$$、$$d_h^R = 64$$，256 个路由专家 + 1 个共享专家取 top-8，专家 $$d_{ff} = 2048$$）为主要分析对象，超参数取自各自公开的 `config.json` 与技术报告；Mixtral 8x7B 在 MoE 一篇作为对照；
- 硬件：以 **H100 SXM** 为默认（80 GB HBM3，3.35 TB/s，BF16 dense 约 989 TFLOPS，FP8 dense 约 1979 TFLOPS），必要处标注 **A100**（80 GB，约 2 TB/s，BF16 约 312 TFLOPS）；这些是公开标称值，实测会因型号、频率与功耗设置有差异；
- 文中所有 FLOPs 与字节数都是**理论下界**，用于建立数量级判断与相互比较，不是任何具体实现的实测值；实测与理论的差距本身就是本系列要教读者解释的东西；
- 论文引用以第一作者与年份标注；方法本身比它们在某个框架中的实现稳定，正文只在必要处提及 vLLM、Megatron、Transformer Engine 等项目中的对应实现。


## 章节目录

1. [Transformer 解剖与参数量：从 config.json 算出 8.03B](/transformer-anatomy-and-parameter-count.html)
2. [前向的算量与访存量：prefill、decode 与 Roofline](/transformer-flops-bytes-and-roofline.html)
3. [Attention 变体与 KV cache：MHA、GQA、MQA 与 MLA 的推导](/attention-variants-and-kv-cache.html)
4. [位置编码与长上下文：RoPE 的波长、外推与代价](/positional-encoding-and-long-context.html)
5. [MoE：路由、激活参数量与通信形态](/moe-compute-and-communication.html)
6. [浮点格式、数值稳定性与混合精度](/floating-point-formats-and-mixed-precision.html)
7. [量化、投机解码与 LoRA：改变计算形态的三种方法](/quantization-speculative-decoding-and-lora.html)


## 最终目标

读完这套系列之后，拿到任何一个模型的 `config.json` 和一张 GPU 的规格表，读者应该能够在动手之前回答：

```text
它有多少参数，分布在哪里？                         → 第一篇：参数量公式
一张卡放得下吗？放下之后还剩多少显存？               → 第一篇、第三篇：权重与 KV cache
每个 token 多少 FLOPs？prefill 和 decode 各是什么瓶颈？ → 第二篇：Roofline
batch 开到多大才能把算力用起来？                    → 第二篇：算术强度与 ridge point
支持多长的上下文？代价在哪一项？                     → 第三篇、第四篇：KV cache 与二次项
它的 attention 变体让 kernel 长什么样？              → 第三篇：GQA 的组、MLA 的吸收
如果是 MoE，多卡之间要传多少数据？                   → 第五篇：all-to-all 字节数
用什么精度？哪一步可能出数值问题？                   → 第六篇：格式与累加
量化能快多少？在哪个阶段快？                        → 第七篇：字节数与 Roofline
投机解码值得开吗？加速上界是多少？                   → 第七篇：期望接受长度
微调它需要多少显存？                               → 第六篇、第七篇：训练状态与 LoRA
```

最终目标是三种能力：

1. **推导能力**：面对一个新模型或新方法，不依赖 benchmark，先算出它的参数量、算量、访存量、显存和通信量的理论值；
2. **判断能力**：用这些数字判断一个优化在什么区间有效、一个部署方案的瓶颈在哪一项、一个实测结果离理论下界差多远；
3. **对话能力**：与算法工程师讨论结构选择、与 kernel 工程师讨论输入 shape、与平台工程师讨论资源需求时，用同一张成本表说话。

这一层知识在架构图上没有位置，却是 AI-Infra 每一层优化的共同目标。
