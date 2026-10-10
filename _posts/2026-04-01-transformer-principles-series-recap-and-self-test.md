---
layout: post
series: transformer-and-llm
title: "Transformer 原理与实现：系列总结与通关自测"
subtitle: "Transformer Principles and Implementation: Series Recap and Self-Test"
tags: [Transformer, LLM, AI, AI-Infra]
catalog: true
date: 2026-04-01 23:00:00 +0800
---

四篇正文沿着一条线，从静态结构走到可训练的 GPT。这里逐篇回顾，再把代码主线与容易混淆的概念收在一起，最后用几道题检查是否能把原理和实现连起来。本文不增加新技术内容；读完后可继续进入现代 LLM 结构系列。

## 一、总览

## 二、逐篇回顾

### 1. 第一篇：Transformer 长什么样：从一句话到下一个 token

**核心问题**：原始 Transformer 的 encoder-decoder 如何变成 GPT-2 的 decoder-only？一个 token 的编号进入 GPT-2，到词表上的一个概率分布出来，中间经过了哪些运算？

**结论**：[原始 Transformer 论文](https://arxiv.org/html/1706.03762v4)提出的是为机器翻译设计的 encoder-decoder：encoder 双向读取源句，decoder 用 causal self-attention 生成目标句，并通过 cross-attention 读取 encoder 输出；GPT-2 属于 decoder-only 路线，去掉 encoder 与 cross-attention，把要参考的内容拼进输入。进入 GPT-2 后，token embedding 把编号变成向量；attention 是**唯一让 token 之间交流的地方**——query 与所有 key 打分、除 $$\sqrt d$$ 防饱和、causal mask 禁止看未来、softmax 得权重、加权求和 value；它本身不知道顺序，所以位置必须显式给（GPT-2 查表 / Llama RoPE）；FFN 是逐 token 的两层小网络，提供非线性、存知识、占一层参数的 2/3；残差流让多层修正训得动，LayerNorm 让每个子层看到同一尺度；lm_head 与 embedding 共享。encoder-only（BERT）、decoder-only（GPT）与 encoder-decoder（T5/BART）分别对应不同结构与目标。

**必记**：

- $$\text{Attention}(Q, K, V) = \text{softmax}(QK^T/\sqrt d + M)V$$；$$d = 4$$、$$T = 3$$ 的手算：t2 的权重 $$(0.27, 0.27, 0.45)$$，输出 $$(0.73, 0.73, 0.27, 0.27)$$，与 PyTorch 对拍差 $$6 \times 10^{-8}$$。
- 随机 $$q \cdot k$$ 的标准差 $$= \sqrt d$$；不除，$$d = 128$$ 时 softmax 一个 token 独占权重。
- GPT-2 small：12 层 × 7.09M + embedding 38.6M + 位置表 0.79M = 124,439,808；FFN 占一层 66.6%、embedding 占总量 31.6%。
- GPT-2 真实的头：第 4 层第 11 头看上一个词、第 3 头把 it 指回 cat（0.84）。

**常见误解**："参数主要在 attention 里"（FFN 是 2/3）；"mask 给了位置信息所以不用位置编码"（mask 只给左右不给距离）；"多头增加算量"（$$h$$ 个小乘法 = 一个大乘法，只多 $$W_O$$）。

### 2. 第二篇：一个 token 的旅程：训练侧与推理侧

**核心问题**：训练时一句话的 $$T$$ 个 token 进入模型，为什么一次前向就能得到 $$T$$ 个训练信号？推理时为什么前面的 token 不用重算——KV cache 里存的是什么、省了什么、花了什么？

**结论**：训练侧：$$T + 1$$ 个连续 token → 输入与右移一位的目标 → 一次前向 $$[B, T, V]$$ → $$T$$ 个交叉熵取平均 → 反向沿原路 → AdamW；**causal mask** 让位置 $$i$$ 只用前文（与推理一致），**teacher forcing** 用真实 token 当目标（位置间可并行），所以一句话 = $$T$$ 个样本。反向要用前向的中间量，激活值必须保存到反向结束、与 $$B \times T$$ 成正比。推理侧：prefill 同训练前向只取最后位置（TTFT），decode 每步一个 token（TPOT）；causal 结构下旧 token 的 K、V 不随新 token 改变，算一次存进 KV cache，每步只算新 token 的 Q、K、V 并追加。

**必记**：

- 随机初始化 loss $$\approx \ln V$$：16 词 2.78、GPT-2 词表 10.8、65 字符 4.17。
- 有 / 无 KV cache 输出逐 token 一致；prompt 256 生成 256，快 7.9 倍；代价 Llama-3-8B 128 KiB / token（32 层 × 2 × 8 头 × 128 × 2 B）。
- 三种形态：训练 $$[B, T]$$ 全位置有用、有反向、存激活值；prefill $$[B, T]$$ 只用最后位置、存 K/V；decode $$[B, 1]$$ 无 mask 矩阵、追加 K/V、访存瓶颈。
- `targets` 若是切片要 `.reshape(-1)`，`view` 会因不连续报错。

**常见误解**："推理就是训练的前向"（decode 完全是另一种形态）；"KV cache 也存 Q"（旧 token 的 Q 用过即弃）；"训练长上下文和推理长上下文一样难"（训练多一份与 $$B \times T$$ 成正比的激活值）。

### 3. 第三篇：手搓 GPT（上）：nanoGPT model.py 逐行解析

**核心问题**：一个能加载 GPT-2 权重、能训练、能生成的 Transformer，最少需要写哪些东西？nanoGPT 的每一行分别在实现前两篇的哪个方框、哪一步？

**结论**：330 行、6 个类，结构本身（`LayerNorm` + `CausalSelfAttention` + `MLP` + `Block`）不到 90 行。Q、K、V 合成一个 $$d \to 3d$$ 的 `c_attn` 一次算完再 `split`；`view` + `transpose` 把头换到第 1 维以便批量矩阵乘；有 SDPA 走融合 kernel，否则手写五行；拼回去要 `contiguous()`；mask 叫 `bias` 是为了和 checkpoint 键名一致。`GPT` 用 `ModuleDict` 让键名与 HF 一致，`wte.weight = lm_head.weight` 共享省 31% 参数，所有权重 $$\mathcal N(0, 0.02^2)$$、写回残差流的 `c_proj` 再除 $$\sqrt{2L}$$；`forward` 传 `targets` 算全部位置的交叉熵，不传只算最后一个位置的 lm_head；`generate` 无 KV cache（教学取舍）；`from_pretrained` 按键名拷 HF 权重、四个 `Conv1D` 矩阵转置；`configure_optimizers` 只对二维参数 decay；`estimate_mfu` 用 $$6N + 12LHQT$$。

**必记**：

- `x = x + self.attn(self.ln_1(x)); x = x + self.mlp(self.ln_2(x))`——pre-norm 两行。
- 与 HF GPT-2 对拍：最后位置 logits 相对差 $$9 \times 10^{-5}$$，argmax 一致。
- `get_num_params()` 默认扣 `wpe`：123.65M；加回 124.44M。
- Llama 相对 GPT-2 只改五处：RMSNorm、RoPE、SwiGLU、GQA、去 bias（lm_head 不共享）。

**常见误解**："三个独立的 Linear 与一个 `c_attn` 数学上不同"（等价，合并只为一次 GEMM）；"推理时 lm_head 也要算全部位置"（只算最后一个，省 99.9%）；"`c_proj` 的特殊初始化是玄学"（$$2L$$ 次相加后方差不涨）。

### 4. 第四篇：手搓 GPT（下）：nanoGPT train.py 与训一个会续写的模型

**核心问题**：从一个 1.1 MB 的文本文件到一个能续写它的模型，中间每一步的代码在哪、为什么那样写？把层数从 4 改到 2 或 8，loss 和速度各会怎样？

**结论**：`prepare.py` 字符级分词成 `uint16` 的 `train.bin` / `val.bin`（9 : 1）+ `meta.pkl`，词表大小由数据决定；`train.py` 的 72 个全局变量默认是 GPT-2 small 复现配方，`configurator.py` 用 `exec` 覆盖；`get_batch` 用 `memmap` + 随机窗口 + `stack`（连续）造 `x` 与右移一位的 `y`，没有 epoch；三种模型来源（从零 / 续训强制结构一致 / GPT-2 权重）+ GradScaler（仅 fp16）/ 优化器 / `compile` / DDP；循环每步：设 lr → 到点评估存 checkpoint 五样 → $$k$$ 个 micro-batch 累积（loss ÷ $$k$$、预取、DDP 只在最后一步同步）→ unscale + 裁剪 → step → zero_grad。实跑 0.8M 参数 2000 步 7 分钟，loss $$\ln 65 = 4.17 \to 1.66$$，输出有莎士比亚格式；train / val 从 1000 步起拉开。

**必记**：

- 默认配置每次迭代 $$40 \times 12 \times 1024 = 491{,}520$$ token；8 卡时每卡累积 5 次，总量不变。
- 2 / 4 / 8 层：参数 0.40 / 0.80 / 1.58M，val loss 1.82 / 1.66 / 1.59，每步耗时约 0.5× / 1× / 2×——参数与耗时线性、loss 收益递减。
- checkpoint 五样：model、optimizer、model_args、iter_num、config。
- MFU 0.2%：小 batch 在 MPS 上完全 launch-bound，与 A100 分母无关。

**常见误解**："不除累积步数只是 loss 尺度不同"（等价于学习率放大 $$k$$ 倍）；"续训不需要优化器状态"（动量从零重估，前几百步抖）；"更深总是更好"（同样时间内更深跑的步数更少）。


## 三、贯穿原理与实现系列的几条线

### 一份代码逐步长出来

第一篇用 `attention_by_hand.py` 对拍 attention；第二篇把它放进带 KV cache 的极小 GPT；第三篇读 nanoGPT 的 `model.py`；第四篇再用 `train.py` 把模型训到能续写。阅读顺序也是实现逐步完整的顺序。

### 几个容易混淆的概念

| 误区 | 正确理解 |
|---|---|
| 大模型的参数主要在 attention 里 | FFN 通常占 Transformer block 参数的大头 |
| causal mask 已经提供位置信息 | mask 只约束可见范围，不能说明左边第几个位置 |
| 推理就是训练时的前向 | 训练、prefill 与 decode 的输入形态和计算方式不同 |
| 三个独立的 Q/K/V Linear 与 `c_attn` 不等价 | 拼接矩阵切开后就是三个矩阵；合并是实现上的选择 |

## 四、通关自测

### 判断与计算

1. 用第一篇的玩具模型：某个 token 的 query 与三个 key 的内积是 $$(2, 6, 4)$$，$$d = 4$$，它是第 2 个位置（能看全部三个）。softmax 权重是多少？如果不除 $$\sqrt d$$ 呢？

   <details markdown="1"><summary>答案</summary>

   除 2 后 $$(1, 3, 2)$$：$$e^1, e^3, e^2 = 2.72, 20.1, 7.39$$，和 30.2，权重 $$(0.09, 0.67, 0.24)$$。不除：$$e^2, e^6, e^4 = 7.39, 403, 54.6$$，权重 $$(0.016, 0.867, 0.117)$$——第二个 token 更独占，$$d$$ 越大越极端。

   </details>



2. 一个 8 层、$$d = 256$$、4 头、词表 1000、上下文 512、GPT-2 结构（带 bias、位置表、权重共享）的模型有多少参数？

   <details markdown="1"><summary>答案</summary>

   每层：attention $$256 \times 768 + 768 + 256 \times 256 + 256 = 263{,}424$$，FFN $$256 \times 1024 + 1024 + 1024 \times 256 + 256 = 525{,}568$$，两个 LN 1024，合计 790,016；8 层 6.32M；`wte` $$1000 \times 256 = 256{,}000$$，`wpe` $$512 \times 256 = 131{,}072$$，`ln_f` 512；总计约 6.71M。lm_head 共享不另算。

   </details>



3. 训练时 $$B = 4$$、$$T = 2048$$、$$V = 128256$$、bf16。logits 张量多大？为什么第三篇的推理分支只算 `x[:, [-1], :]`？

   <details markdown="1"><summary>答案</summary>

   $$4 \times 2048 \times 128256 \times 2$$ B $$\approx 2.1$$ GB——训练时必须全算（每个位置都有目标）。推理只需要最后一个位置的分布，全算是浪费 $$T - 1$$ 倍的 lm_head 计算与这 2 GB 显存。

   </details>



4. 第四篇的配置下 decode 一个 500 token 的回答（prompt 500）：nanoGPT 的 `generate`（无 KV cache）总共前向了多少个 token 位置？第二篇的带 cache 版本呢？

   <details markdown="1"><summary>答案</summary>

   无 cache：第 $$k$$ 步算 $$500 + k$$ 个位置，$$\sum_{k=1}^{500}(500 + k) = 250{,}000 + 125{,}250 = 375{,}250$$。有 cache：prefill 500 + decode 500 = 1000。差 375 倍；实测倍数小于此（每步还要读 cache、小矩阵效率低）。

   </details>



### 掌握判据

读完后，能够画出 Transformer 主干、手算一个 attention 头、说明训练与推理的差别，并沿着 `model.py` 与 `train.py` 找到公式在代码中的位置。

## 五、下一步

接着读[《现代 LLM 结构：从 GPT-2 到今天的演进》](/llm-architecture-evolution-roadmap-from-gpt2.html)，看熟悉的主干后来发生了哪些变化。
