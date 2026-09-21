---
layout: post
series: efficient-inference
title: "高效推理与压缩（05）：KV cache 压缩：量化、驱逐与稀疏 attention"
subtitle: "KV Cache Compression: Quantization, Eviction and Sparse Attention"
tags: [AI, LLM, Inference, KV Cache, Long Context]
catalog: true
updated: 2026-09-14
---

[04 系列第三篇](/attention-variants-and-kv-cache.html)算过 KV cache 的账：Llama-3-70B 在 128K 上下文下每个请求的 KV 是 40 GiB（GQA 之后）——权重 141 GB 的近三成，一张 80 GB 卡的一半；decode 每步要把它全部读一遍，单请求 128K 时 KV 读取已是权重的 30%，batch 到 4 就与权重相当（每 token 320 KiB，$$141\text{ GB}/320\text{ KiB} \approx 43$$ 万 token 与权重打平），长上下文 + 并发下 KV 读取超过权重读取成为 decode 的主要流量。结构级的解法——GQA 把 KV 头数除以 8、MLA 把每 token 的 KV 压到 576 维——在训练时就决定了，训好之后不能改。

这一篇讲**训好之后**还能对 KV 做什么。三条路：**量化**（每个元素的字节从 2 降到 1 或 0.5 甚至 0.25）、**驱逐**（丢掉一部分 token 的 KV，只留"重要"的）、**稀疏 attention**（每步只读一部分 KV——如果模型训练时就这样，推理时可以精确地这样做）。三条路对输出分布的影响从小到大：量化是可控的噪声；驱逐是有损的、任务依赖的近似；训练时就稀疏的 attention 在推理时是精确的（但需要重新训练）。

KV 压缩是本系列里"退化最不均匀"的一类方法：摘要任务上驱逐 80% 的 KV 几乎无损，"大海捞针"上驱逐 20% 就可能失败。理解每种方法**丢掉的是什么信息**，是判断它在哪类任务上安全的唯一办法。

本篇要回答的核心问题是：

> **128K 上下文的 KV 从 40 GB 压到 10 GB，哪种办法在哪类任务上安全？[^q0]**

## 一、总览：三条路与它们丢掉的东西

### 1. 一张对照表

| 方法 | 压缩什么 | 压缩比 | 丢掉什么 | 安全的任务 | 危险的任务 |
|---|---|---|---|---|---|
| KV 量化（INT8 / FP8） | 每元素字节 2 → 1 | 2× | 几乎不丢 | 全部 | — |
| KV 量化（INT4 / INT2，KIVI 一类） | 每元素字节 2 → 0.5 / 0.25 | 4–8× | 数值精度；长序列上累积 | 多数；短到中等上下文 | 超长上下文的精确检索 |
| 驱逐（StreamingLLM） | 只留 sink + 最近窗口 | 任意 | 窗口外的全部信息 | 流式对话、只依赖近期 | 任何需要回看的任务 |
| 驱逐（H2O / SnapKV / PyramidKV） | 按注意力得分留 top-k | 2–5× | 当前"不重要"但以后可能重要的 token | 摘要、QA（问题已知） | 多跳、needle、问题在文档之后 |
| token 合并 | 相似 KV 合成一个 | 2–3× | 细粒度差别 | 冗余多的输入 | 精确检索 |
| 跨层共享（CLA / YOCO） | 多层共用一份 KV | 2–L× | 各层独立的表示（需训练） | 训练时决定 | — |
| 训练时稀疏（NSA / MoBA） | 每步只读部分块 | 5–10× 的读取 | 不丢（精确等于训练时的模型） | 全部（模型就是这样训的） | 需要重新训练 |
| prompt 压缩 | 输入 token 数 | 2–10× | 输入的一部分文字 | 冗余大的输入 | 细节依赖 |

### 2. 先说答案

40 GB → 10 GB 是 4 倍。安全性由高到低：

1. **FP8 / INT8 KV**（2 倍）几乎在所有任务上安全，是第一步，没有理由不做。剩下的 2 倍：
2. **INT4 KV（KIVI 式，key per-channel、value per-token）**再 2 倍，在 32K 以内的多数任务上退化在 1 个点以内；超过 64K 的精确检索任务要测。
3. 如果任务是**问题已知、文档在前**的 QA / 摘要（SnapKV 的场景），驱逐到 25% 通常安全——用 prompt 尾部（问题）的注意力选 KV。
4. 如果任务是**多跳、needle、或问题在文档之后**（Agent 的长对话历史），驱逐不安全——任何基于"当前注意力"的重要性打分都不知道未来的问题要什么。这时只有量化，或者换一个训练时就稀疏的模型。

所以对通用负载：直接 BF16 → INT4（KIVI 式）拿到约 3 倍（元数据后），或保守些只做 FP8 拿 2 倍——FP8 与 INT4 是同一份 KV 的两种精度选择，不能叠成"再压 2 倍"；不驱逐；对已知形态的 QA 负载：FP8 + SnapKV 驱逐拿到 4 倍以上。第七章的决策表。

### 3. 本文的章节安排

| 章 | 主题 | 内容 |
|---|---|---|
| 二 | KV 的数值结构 | key 的通道离群与 value 的均匀；massive activations 与 sink 的关系；误差怎么进入 attention |
| 三 | KV 量化 | FP8 / INT8 的免费午餐；KIVI 的非对称粒度推导；2 bit 的极限；与 PagedAttention 的配合 |
| 四 | 驱逐 | attention sink 的成因；StreamingLLM；H2O 的累计注意力；SnapKV 的观察窗；PyramidKV 的层分配；驱逐在 needle 上失败的原因 |
| 五 | 合并与共享 | token 合并；CLA / YOCO 的跨层共享（训练时） |
| 六 | 训练时稀疏 | 稀疏 attention 的三种模式；NSA 的三分支与可微选择；MoBA 的块路由；为什么它们是精确的 |
| 七 | prompt 压缩与决策 | LLMLingua 一类；一张按任务形态的决策表 |
| 八 | 成本 | 字节账；量化 / 驱逐的运行时开销；与投机解码、量化权重的叠加 |
| 九 | 动手（建议） | needle 准确率随 KV 预算的曲线 |
| 十 | 本文小结 | |
| 十一 | 自测 | 5 道题 |

## 二、KV 的数值结构

### 1. key 与 value 的分布不同

KV 量化的第一个事实（Liu 等 2024 KIVI 的分析；Hooper 等 2024 KVQuant 独立发现）：**key 有固定通道的离群值，value 没有**。

key 是 $$W_K$$ 作用在 hidden state 上的结果（再经过 RoPE）。hidden state 有通道级离群（上一篇讲的 LayerNorm γ 放大的维度），$$W_K$$ 是线性的，这些离群传到 key 的某些通道上——对所有 token 都在同样的通道，幅度是其他通道的几十倍。RoPE 是逐对维度的旋转，不改变每对维度的范数，所以离群保留在原来的维度对上（只是在对内旋转）。

value 是 $$W_V$$ 作用在同一个 hidden state 上，理论上也会继承离群——但实证上 value 的分布在通道间接近均匀，没有系统性离群通道。一个解释是训练动力学：value 直接乘 attention 权重后求和进入输出，离群的 value 通道会主导输出、被后面的层"惩罚"；而 key 只参与点积打分，离群通道在点积里被 query 的对应通道"选择"（query 的这些通道小或大决定关注与否），是 attention 实现"关注特定位置"的机制之一。

### 2. 误差怎么进入 attention

key 的误差 $$\delta k$$ 进入 logits：$$q^\top (k + \delta k) = q^\top k + q^\top \delta k$$。softmax 对 logits 的误差是**指数级**敏感的——logits 差 1 对应概率比差 $$e$$。如果 $$q^\top \delta k$$ 在不同位置上的量级是 0.5，那么 attention 分布被明显扰动；且 query 在离群通道上的值大（那是它选择位置的方式），离群通道的 key 误差被 query 放大。这就是 key 量化比 value 量化难的原因。

value 的误差 $$\delta v$$ 进入输出：$$\sum_i a_i (v_i + \delta v_i) = \sum_i a_i v_i + \sum_i a_i \delta v_i$$。误差被 attention 权重加权平均——$$a_i$$ 之和为 1，如果 $$\delta v_i$$ 独立同分布，平均后的误差方差是单个的 $$\sum a_i^2 \le 1$$ 倍，通常远小于 1（attention 分散在多个位置时）。**value 的量化误差被 attention 的平均抑制**，key 的被 softmax 放大。

### 3. 长序列上的累积

decode 时每个新 token 的输出依赖对所有历史 KV 的 attention。历史 KV 的量化误差是固定的（写入时决定），不随时间累积；但**生成的 token 本身**依赖这些误差，下一个 token 的 KV 是在略偏的隐状态上算的，再被量化——误差通过生成的文本间接累积。在几千 token 的生成里这通常不显著，在几万 token 的推理链（推理模型）里可能显著。这是推理模型对 KV 量化更敏感的机制之一。

## 三、KV 量化

### 1. FP8 / INT8：免费的 2 倍

KV 到 8 bit——FP8 E4M3 per-tensor 或 per-head 静态 scale，或 INT8 per-token 动态——对几乎所有模型与任务无损（KL 增量在 0.001 nat 量级）。原因是 8 bit 的 255 个级别足够精细，即使 key 有 20 倍的离群，其他通道仍有十几个级别。vLLM 的 `kv_cache_dtype="fp8"` 就是这个，attention kernel 在读 KV 时 dequant。它的收益是 KV 字节减半、decode 的 KV 读取流量减半——长上下文下直接体现为 decode 速度。**这是所有部署都应该开的选项。**

### 2. KIVI：非对称的量化粒度

到 4 bit 与 2 bit，粒度要匹配第二章的结构：

- **key：per-channel**。离群在固定通道，所以每个通道一个 scale——沿 token 维度分组（每 $$G$$ 个 token 一组，对每个通道在这 $$G$$ 个 token 上取范围）。这样离群通道有自己的大 scale，不影响其他通道。
- **value：per-token**。value 没有通道离群，每个 token 一个 scale（沿通道维度取范围）——这也是最自然的粒度，因为 value 是按 token 写入、按 token 被读的。

KIVI 论文（Liu 等 2024）的消融：Llama-2-13B 的 2 bit KV，key per-token + value per-token 崩掉（困惑度几十），key per-channel + value per-token 困惑度只升 0.1 左右——粒度的选择是决定性的。

per-channel 的 key 量化有一个实现上的问题：per-channel 的 scale 要在**一组 token 都到齐后**才能算（沿 token 维度取范围），而 decode 是一个 token 一个 token 地来。KIVI 的做法是保留最近 $$R$$ 个 token（$$R = 32$$ 或 128）的 key 为全精度"残差"，凑满一组再量化；attention 时对量化部分与残差部分分别计算再合并。这个残差窗口还有一个副作用：最近的 token 保持全精度，正是 attention 最集中的位置——精度损失最小。

### 3. 2 bit 的极限

INT2 只有 4 个级别，即使 per-channel，key 的量化误差也很大。KIVI 的 2 bit 靠残差窗口与 group（$$G = 32$$）勉强工作；KVQuant 加了两个技术——**pre-RoPE 量化**（在 RoPE 之前量化 key，因为 RoPE 的旋转让离群通道的值在 token 间变化、per-channel 的范围变大；量化 pre-RoPE 的 key 再在读取时施加 RoPE，需要 kernel 配合）与**非均匀格点**（按 key 分布的分位数放格点，类似 NF4）——把 3 bit 做到接近无损、2 bit 困惑度 +0.1–0.3。再往下没有实用的方法。

实践的位置：**4 bit KV 是舒适区**（8 倍于 FP8 的 2 倍再 2 倍 = 4 倍总压缩），2 bit 是研究前沿。

### 4. 与 PagedAttention 的配合

量化 KV 在系统侧要与分页存储配合：每个 block（16 个 token）存量化值加 scale（per-token 的 value scale 每 token 一个；per-channel 的 key scale 每 group 一个——group 大小要与 block 对齐）。attention kernel 在读 block 时 dequant。FlashInfer 与 vLLM 的 FP8 KV 路径已经成熟；INT4 KV 的 kernel 支持在 2025 年逐步进入主流引擎。

## 四、驱逐

### 1. attention sink 的成因

[04 系列第四篇](/positional-encoding-and-long-context.html)介绍了现象：LLM 对序列的**第一个 token** 分配了不成比例的注意力（常常 30–50%），无论它的内容是什么。StreamingLLM（Xiao 等 2023）发现只要保留这几个 sink token 的 KV 加一个最近窗口，模型就能在无限长的流上生成而不崩；丢掉 sink 则立即崩掉。

成因（Xiao 等的解释，后续 Sun 等 2024 的 massive activations 分析补充）：softmax 强制 attention 权重之和为 1，但很多 head 在很多位置**不需要关注任何东西**（当前 token 的信息足够，或这个 head 负责的模式没出现）。模型需要一个"垃圾桶"位置吸收多余的注意力——第一个 token 是最方便的选择，因为它对所有位置都可见（因果掩码下唯一对全序列可见的位置）。模型学会在第一个 token 的隐状态上产生 massive activation（上一篇讲的数千量级的值），让它的 key 与所有 query 的点积都大——成为 sink。sink 的 value 通常接近零向量，所以关注它等于"不加东西"。

这解释了两件事：为什么驱逐 sink 会崩（softmax 的多余质量没地方去，被分到不该关注的位置）；为什么 sink 不一定只在第一个 token（分隔符、换行也常成为次级 sink）。它还有一个训练侧的推论——在 softmax 里加一个可学习的"零位置"（off-by-one softmax，或 gpt-oss 使用的每 head 一个可学习的 sink 标量），模型就不需要用真实 token 当垃圾桶，量化与驱逐都更容易。

### 2. StreamingLLM：只留 sink 与窗口

保留前 4 个 token 加最近 $$W$$ 个 token（$$W = 2K$$–$$4K$$）。KV 是常数大小，可以无限生成。但**窗口外的信息全部丢失**——它是流式场景（长对话的实时生成、只依赖近期上下文）的解法，不是长上下文理解的解法。任何需要回看 $$W$$ 之前内容的任务都失败。

### 3. H2O：累计注意力的 top-k

H2O（Zhang 等 2023）的观察：attention 权重在 token 上高度集中——少数"heavy hitter"token 获得了绝大部分注意力，且这些 token 在生成过程中相对稳定。策略：为每个 token 维护它**累计**获得的注意力（所有后续 query 对它的 attention 权重之和），KV 预算满时驱逐累计最低的；同时保留最近窗口（最近的 token 累计少但可能马上重要）。预算 20% 时在摘要、QA 上接近无损。

它的问题：（1）累计注意力偏向**早出现**的 token（累计时间长），对新出现但重要的 token 不公平；（2）需要 attention 权重——FlashAttention 不输出它们（那正是 FlashAttention 省内存的方式），要么用另一个 kernel、要么近似；（3）**贪心不可逆**——驱逐的 KV 找不回来，如果后面的问题需要它，就失败了。

### 4. SnapKV：用 prompt 尾部选

SnapKV（Li 等 2024）针对**长 prompt、短生成**的场景（文档 QA、摘要）：prompt 的最后一段（"观察窗"，最后 16–64 个 token，通常是问题本身）对 prompt 的 attention 模式已经揭示了哪些位置重要——对每个 head，用观察窗的 query 对全部 prompt 的 attention 得分投票，保留 top-k（加一个池化让选中的位置有一定连续性），在 prefill 结束时一次性压缩。生成时不再驱逐。

它比 H2O 好在：选择发生在 prefill 后，用的是**已知的问题**——对"问题在文档后面"的形态是精确的针对。预算 25% 时 LongBench 几乎无损，needle 在 128K 上仍能通过（论文报告）。它不能处理的：多轮对话（第二个问题不在第一次的观察窗里）、生成很长的任务（生成过程中的新需求没被考虑）。

### 5. PyramidKV：按层分配

PyramidKV（Cai 等 2024）的观察：不同层的 attention 模式不同——浅层的 attention 分散（几乎均匀地看全部上下文），深层的集中（聚焦在少数 token 上）。所以浅层需要更多 KV、深层需要更少。给每层不同的预算（金字塔形：浅层多深层少），总预算相同时比均匀分配好。这是一个正交的改进，可以叠在 SnapKV 或 H2O 上。

### 6. 驱逐在 needle 上失败的原因

"大海捞针"（在长文档某处埋一句话，文档后面问它）是驱逐方法的试金石。失败的机制：needle 在被读到时是"不重要"的——它与周围文本无关、没有 query 关注它——所以按注意力打分它被驱逐了；等到问题来了，它已经不在了。SnapKV 用问题选 KV 能过（问题在观察窗里），H2O 过不了（问题来之前 needle 的累计注意力低）。多跳任务更难：第一跳的答案决定第二跳要找什么，任何在问题之前做的选择都不知道第二跳。

**一般原则**：基于注意力的驱逐假设"过去不重要的将来也不重要"，这个假设在信息密度高、问题未知的任务上不成立。它安全的条件是：问题已知（SnapKV）、或任务只依赖近期（StreamingLLM）、或输入冗余大（摘要——丢掉重复的段落无所谓）。

## 五、合并与共享

### 1. token 合并

与驱逐（丢掉）不同，合并把相似的 KV **平均**成一个（加权，按注意力或按相似度），保留部分信息。对 value 合并是直接的（平均）；对 key 合并会改变点积结构（两个 key 的平均与 query 的点积不等于两个点积的 softmax 加权），有近似误差。压缩 2–3 倍时比驱逐同等比例更稳，但实现复杂（要找相似对——聚类），收益上限低。实践中不如量化 + 驱逐的组合常见。

### 2. 跨层共享：CLA 与 YOCO

如果多层共用一份 KV，KV 总量除以共享的层数。Cross-Layer Attention（Brandon 等 2024）让相邻两层共享 K、V（第二层不算自己的 K、V，直接用第一层的），KV 减半，困惑度损失很小（在 GQA 之上再叠加）。YOCO（Sun 等 2024）更激进：模型分成两半，前一半用高效 attention（滑窗或线性）产生一份全局 KV，后一半的所有层**只用这一份** KV 做 cross-attention——KV 从 $$L$$ 层变成 1 层，prefill 也可以提前退出（后一半不需要为 prompt 算 KV）。

它们是**训练时的结构决定**，不是训好之后的压缩——放在这里是因为它们回答同一个问题（KV 太大怎么办），且提示了一个趋势：2025 年的新模型越来越多地把 KV 效率设计进结构（MLA、滑窗与全局交错、跨层共享），让推理时的压缩需求变小。

## 六、训练时稀疏

### 1. 三种稀疏模式

推理时的 KV 驱逐是**事后**的近似。另一条路是让模型在训练时就只看一部分 KV——推理时精确地做同样的事，没有近似误差。稀疏 attention 的模式（[04 系列第四篇](/positional-encoding-and-long-context.html)第七章列过形态）：

- **固定模式**：滑窗（每个 query 看最近 $$w$$ 个）、全局 token（少数位置全部可见）、扩张（隔 $$k$$ 个看一个）、块对角。Longformer / BigBird 时代的做法；现代模型里滑窗与全局层交错（Gemma 2/3、gpt-oss）是它的活形态。
- **内容路由**：每个 query 按内容选择看哪些 KV 块——需要一个便宜的"选块"机制。这是 NSA 与 MoBA 的路线。
- **压缩 + 选择**：先把 KV 压缩成粗粒度的摘要（每块一个向量），query 对摘要做 attention 决定关注哪些块，再对选中的块做细粒度 attention。

### 2. NSA：三分支与可微的选择

Native Sparse Attention（Yuan 等 2025，DeepSeek）在每个 query 位置上并行三条分支，输出加权求和（权重由一个门控 MLP 给出）：

1. **压缩分支**：把 KV 按块（32 个 token）压缩——每块经过一个小 MLP（带块内位置编码）变成一个 K 与一个 V——query 对所有压缩块做 attention。这给了粗粒度的全局视野，成本是 $$1/32$$。
2. **选择分支**：用压缩分支的 attention 得分（query 对每个压缩块的权重）作为块重要性，选 top-$$n$$ 个块（$$n = 16$$），对这些块内的**原始** KV 做细粒度 attention。得分在 GQA 组内的 head 之间共享（让同组 head 选同样的块，kernel 才能高效）。
3. **滑窗分支**：最近 512 个 token 的原始 KV，保证局部信息不丢。

**可微性**：top-$$n$$ 的选择本身不可微，但选择依赖的得分来自压缩分支（可微），且被选中的块内 attention 是可微的——梯度通过压缩分支流到"哪些块重要"的表示上。三个分支的门控让模型学会在不同情境下依赖不同分支。NSA 报告 27B 模型（MoE）在 64K 序列上训练，与全 attention 的困惑度与下游任务相当或略好，64K 下 decode 的 KV 读取减少约 11 倍、前向与反向的 attention 速度提升 6–9 倍。

### 3. MoBA：块路由

Mixture of Block Attention（Lu 等 2025，Moonshot）把 MoE 的路由思路用在 KV 块上：KV 按块（512 个 token）分组，每块的 key 取均值作为块的代表；query 与所有块代表做点积，选 top-$$k$$ 个块（$$k = 3$$），对选中块的全部 KV 做标准 attention（加上当前块必选，且保持因果）。没有压缩分支与门控，更简单；块的选择用的是 key 均值的点积，不需要额外参数。它的一个设计点是**可以与全 attention 切换**——训练时前期用全 attention、后期或部分层切到 MoBA，或者反过来，让同一个模型在两种模式间迁移。Kimi 的长上下文模型用了它。

### 4. 为什么它们是精确的

NSA / MoBA 训练时就按这个模式计算 attention，模型学到的一切都建立在"我只看这些块"的前提上；推理时执行同样的计算，得到的就是模型的输出，没有近似。与驱逐的区别是本质的：驱逐是在一个假设"看到全部"的模型上强行遮掉一部分；稀疏 attention 是一个从来就只看一部分的模型。

代价：（1）需要重新训练（或至少继续预训练几百 B token）；（2）需要专门的 kernel（块选择 + gather + attention，NSA 论文自己写了 Triton kernel）；（3）它是训练时的结构决定，属于 L4 的"结构选择"，对已经训好的模型无用——所以在本篇的位置是"如果驱逐不安全而你有训练能力，这是正确的方向"。

## 七、prompt 压缩与决策

### 1. prompt 压缩

另一条路不碰 KV，直接减少输入的 token：LLMLingua 一类用一个小模型对 prompt 的每个 token 打困惑度（或信息量）分，删掉低信息的 token（功能词、冗余），压缩 2–10 倍后送给大模型。它对**冗余大的输入**（多篇检索到的文档、有大量模板的日志）有效，对细节依赖的任务（代码、法律条文）有害——删掉的"低信息" token 可能正是关键的限定词。它是应用层的手段（属于应用地图的 RAG / 上下文工程），这里只作为对照列出。

### 2. 决策表

| 负载形态 | 推荐 | 预期压缩 | 不要做 |
|---|---|---|---|
| 通用对话，上下文 < 32K | FP8 KV | 2× | 驱逐（收益小、风险不值） |
| 长文档 QA / 摘要（问题在后） | INT4 KV（或 FP8）；或 FP8 + SnapKV 25% | 3–8× | H2O（不知道问题） |
| 多轮 Agent、长历史、问题未知 | INT4 KV（保守则 FP8） | 2–3× | 任何驱逐 |
| 流式生成、只依赖近期 | StreamingLLM（sink + 窗口） | 常数 | — |
| 推理模型、长输出 | FP8 KV；INT4 谨慎（生成误差累积）；不驱逐 | 2–4× | 驱逐（推理链的每一步都可能被回看） |
| 超长（> 128K）、精确检索 | FP8 KV；INT4 要测 needle | 2× 稳妥 | 2 bit、驱逐 |
| 有训练能力、长上下文是核心 | 训练时稀疏（NSA / MoBA）或结构（MLA、滑窗交错、CLA） | 5–10× 读取 | — |

## 八、成本

### 1. 字节账

Llama-3.1-70B（80 层、8 KV head、head_dim 128）每 token 的 KV：$$2 \times 80 \times 8 \times 128 \times 2 = 327{,}680$$ 字节 = 320 KiB（BF16）。128K = 131072 token 上下文：40 GiB（约 43 GB；下表沿用二进制单位，写作 GB 的地方都是 GiB）。

| 配置 | 每 token | 128K | 备注 |
|---|---|---|---|
| BF16 | 320 KB | 40 GB | 基线 |
| FP8 | 160 KB + scale（per-head 静态，忽略） | 20 GB | 免费 |
| INT4 KIVI（G = 32，R = 128 残差） | 80 KiB + scale/zero（每 32 token 每通道 FP16 ×2：$$2 \times 80 \times 8 \times 128 \times 4 / 32$$ = 20 KiB） | 12.5 GiB + 残差 40 MiB | 元数据 25%——group 小的代价 |
| INT2 KIVI（同 G = 32） | 40 KiB + 同样 20 KiB 元数据 | 7.5 GiB + 残差 | 元数据占一半，2 bit 的"4 倍"只兑现 2.7 倍 |
| INT4 + SnapKV 25% | 上述 × 0.25 | 3.1 GB | 仅 QA 形态 |
| MLA（DeepSeek-V3，训练时决定） | 576 × 2 = 1.15 KB（61 层：70 KB） | 8.8 GB | 结构级；不同模型 |

INT4 KV 的元数据比权重量化重（group 沿 token 维度只有 32），实际压缩 3.2 倍而不是 4 倍。

### 2. 运行时开销

FP8 KV 的 dequant 在 attention kernel 内，开销几个百分点，被 KV 读取减半的收益远远盖过。INT4 KV 的 dequant 更重（unpack + scale + zero），且残差窗口的全精度部分要单独算再合并——KIVI 的实现在长上下文下 decode 快 2–3 倍（KV 读取减少主导），短上下文下持平或略慢。驱逐的开销在打分（需要 attention 权重或近似）与 KV 的物理移动（PagedAttention 下是 block 的释放，便宜）；SnapKV 只在 prefill 结束做一次，几乎免费。

### 3. 与其他方法的叠加

KV 量化与权重量化叠加：decode 的两项流量（权重、KV）都减少，在长上下文下 KV 量化的边际收益更大（Llama-3-70B 每 token KV 320 KiB，与 141 GB 权重打平要约 43 万 token——单请求 430K，或 batch 4 × 108K、batch 8 × 54K；"约 40K 交叉"只在 batch ≈ 10 时成立，交叉点随并发移动）。KV 量化与投机解码叠加：验证 $$N_{tree}$$ 个 token 要读 KV，KV 字节减半让长上下文下投机的验证成本也减半（上一篇第八章的问题被缓解）。驱逐与投机叠加则要小心——树的多条路径需要一致的 KV 视图。

## 九、动手（建议）

一张 24 GB 的卡，Llama-3.1-8B-Instruct（128K 上下文，GQA 8 头，每 token KV 128 KB）：

- **KV 量化**：vLLM 的 `kv_cache_dtype` ∈ {auto, fp8}；INT4 用 KIVI 的开源实现或 HF `transformers` 的 `cache_implementation="quantized"`（HQQ / quanto 后端，4 bit 与 2 bit）。
- **驱逐**：SnapKV 与 H2O 的开源实现（或 `kvpress` 库，它统一了多种驱逐方法的接口），预算 ∈ {100%, 50%, 25%, 10%}。
- **任务**：RULER 的 needle 单针 / 多针 / 多跳子集，在 8K / 32K / 64K；LongBench 的摘要与 QA 子集；GSM8K（短上下文对照，看量化是否伤推理）。
- **指标**：准确率随 KV 预算的曲线（每种方法一条）；对 BF16 KV 的逐 token KL（量化方法）；decode 吞吐在 64K 上下文下的变化。

该看的：FP8 是否在所有任务上与 BF16 重合；INT4 在 64K needle 上是否开始掉；SnapKV 25% 在单针上是否仍通过、在多跳上是否掉；H2O 在 needle 上何时失败；2 bit 的 KL 与 4 bit 差多少。不引用任何未跑过的数字。

## 十、本文小结

| 项 | 规则 / 公式 | 备注 |
|---|---|---|
| 结构 | key 有固定通道离群（继承 hidden state），value 没有 | key 误差被 softmax 指数放大；value 误差被 attention 平均抑制 |
| 8 bit | FP8 / INT8 KV 几乎无损，字节减半 | 所有部署都应开 |
| KIVI | key per-channel（沿 token 分组 G = 32）+ value per-token；残差窗口 R 保持全精度 | 2 bit 靠粒度选择从崩掉到 +0.1；4 bit 是舒适区 |
| KVQuant | pre-RoPE 量化 + 非均匀格点 | 3 bit 接近无损 |
| sink | softmax 的多余质量需要垃圾桶；第一个 token 全序列可见；massive activation 让它的 key 与所有 query 点积大 | 驱逐 sink 即崩；可学习 sink 标量是训练侧解法 |
| StreamingLLM | sink + 最近窗口，常数 KV | 只依赖近期的流式场景 |
| H2O | 累计注意力 top-k + 窗口 | 偏向早 token；需 attention 权重；needle 失败 |
| SnapKV | prefill 后用观察窗（问题）选 top-k | 问题已知的 QA 25% 近无损；多轮 / 长生成不适用 |
| 失败原则 | "过去不重要 = 将来不重要"在问题未知、信息密度高时不成立 | 多跳、needle、Agent 历史不要驱逐 |
| 训练时稀疏 | NSA 三分支（压缩 / 选择 / 滑窗，门控）；MoBA 块路由（key 均值点积 top-k） | 精确（模型就这样训的）；需重训与 kernel；64K 下 KV 读取 ÷ 11 |
| 账 | 70B 128K：40 GB → FP8 20 → INT4 12.5（元数据 25%）→ +SnapKV 3.1 | INT4 KV 实际 3.2×；MLA 结构级 8.8 GB |

## 十一、自测

1. key 与 value 的数值结构差在哪？这决定了各自该怎么量化？

   <details markdown="1"><summary>答案</summary>

   key 有固定通道的离群值（继承 hidden state 的通道级离群），value 没有；key 的误差被 softmax 指数放大、value 的误差被 attention 权重平均抑制。所以 key 沿通道分组（per-channel）、value 沿 token 分组（per-token）——KIVI 的做法。

   </details>

2. 为什么驱逐第一个 token（sink）会让模型崩掉？

   <details markdown="1"><summary>答案</summary>

   softmax 必须把多余的注意力质量放到某个地方，第一个 token 全序列可见、且 massive activation 让它的 key 与所有 query 点积都大，成了“垃圾桶”；驱逐它后多余质量被迫分给内容 token，attention 分布全乱。StreamingLLM 永远保留 sink。

   </details>

3. H2O 与 SnapKV 各按什么选留哪些 KV？各在什么任务上失败？

   <details markdown="1"><summary>答案</summary>

   H2O 按累计注意力 top-k + 最近窗口，偏向早期 token（累计时间长），needle 类任务失败；SnapKV 在 prefill 后用观察窗（问题）的 attention 选 top-k，问题已知的 QA 上留 25% 近无损，但多轮或长生成里问题会变、选错。

   </details>

4. 128K 上下文、70B 模型的 KV 40 GB：FP8 KV、KIVI 2 bit、SnapKV 留 25% 各压到多少？哪个最安全？

   <details markdown="1"><summary>答案</summary>

   FP8 20 GiB（多数任务几乎无损，但要看 kernel / 后端是否支持、scale 怎么校准，不是"任何部署都该开"）；2 bit 约 7.5 GiB（40 KiB 载荷 + G = 32 下同样 20 KiB 的 scale/zero，元数据占一半，再加残差窗；靠分组粒度才从崩掉到 +0.1）；25% 驱逐 10 GiB（只在问题已知的 QA 上安全）。安全性：8 bit > 4 bit > 驱逐。

   </details>

5. 训练时稀疏 attention（NSA、MoBA）与推理时驱逐（H2O）在原理上差在哪？

   <details markdown="1"><summary>答案</summary>

   训练时稀疏让模型学会“该看哪些块”并在训练中适应，稀疏模式是模型的一部分；推理时驱逐是在一个全 attention 模型上事后删 KV，模型没学过缺失，靠启发式猜哪些不重要——前者可以稠密训练同质量，后者总有失效场景。

   </details>

## 下一篇

[剪枝、深度缩放与小模型配方](/pruning-depth-scaling-and-small-model-recipes.html)

[^q0]: 第一步永远是 FP8 KV（2 倍，全任务安全，[第三章](#三kv-量化)）；第二步看任务形态。**问题未知、需要回看**的任务（多轮 Agent、多跳、推理模型的长链、超长精确检索）只能靠量化再往下——INT4 KV（KIVI 式的 key per-channel / value per-token）在 32K 以内的多数任务上退化在 1 个点内，64K 以上的精确检索要测；不要驱逐，因为任何基于当前注意力的重要性打分都不知道未来的问题要什么，needle 在被读到时就是「不重要」的（[第四章](#四驱逐)）。**问题已知、文档在前**的 QA / 摘要可以用 SnapKV 在 prefill 后按问题选 25% 的 KV，通常无损，与 FP8 叠加拿到 8 倍。流式只依赖近期的场景用 StreamingLLM。如果长上下文是核心需求且有训练能力，正确的方向不是推理时压缩而是训练时稀疏（NSA / MoBA）或结构（MLA、滑窗交错、跨层共享）——它们是精确的，且把 KV 读取减少一个量级（[第五](#五合并与共享)至[七章](#七prompt-压缩与决策)）。
