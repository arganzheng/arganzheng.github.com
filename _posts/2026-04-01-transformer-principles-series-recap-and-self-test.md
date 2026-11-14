---
layout: post
series: transformer-and-llm
title: "Transformer 原理与实现（05）：系列总结与通关自测"
subtitle: "Transformer Principles and Implementation: Series Recap and Self-Test"
tags: [Transformer, LLM, AI, AI-Infra]
catalog: true
date: 2026-04-01 23:00:00 +0800
updated: 2026-11-14
---

四篇正文沿着一条线，从静态结构走到一个训得出来的 GPT：第一篇把 GPT-2 small 的每个方框打开，用 $$d = 4$$、3 个 token 的例子手算 attention；第二篇让一个 token 在训练、prefill 与 decode 三种形态里流过这些方框，解释一次前向的 $$T$$ 个训练信号和 KV cache；第三篇把前两篇的图落到 nanoGPT `model.py` 的 330 行，并与 HuggingFace GPT-2 对拍；第四篇读 `train.py`，在一台 MacBook 上把 0.8M 参数的模型训到能续写莎士比亚，再改层数看后果。四篇合起来，是[《Transformer 原理与实现：从论文到手搓 GPT-2》](/transformer-and-llm-structure-implementation-and-evolution.html)总纲里那条「图 → 数据流 → 代码 → 训练」的路径。

本文不讲新内容，做三件事：把四篇压成一张表与四段回顾，把贯穿四篇的几条线拎出来，再给一套通关自测。各篇末尾的自测检验的是「这一篇读懂了没有」，这里检验的是「四篇能不能连起来用」：拿到一个 `config.json` 能不能数出参数、画出形状；拿到一段公式能不能指出它在 `model.py` 的哪一行；拿到一份训练日志能不能判断它正不正常。

> **读完这四篇，你应该能回答哪些问题？[^q0] 哪些数字与结论必须能脱口而出？[^q1] 怎么判断自己是「读过」还是「掌握」了？[^q2]**

## 一、总览：系列回答的问题与主线

系列的一句话主张是：**Transformer 的每个部件都能手算、都能在代码里找到那一行、都能训出来验证**。一个 decoder-only 模型只有六种部件，从头到尾每个方框的输入输出都是 $$[T, d]$$；只有 attention 让 token 之间交流。同一套结构在训练时一次前向算 $$T$$ 个位置、推理时先 prefill 再逐个 decode；nanoGPT 用不到 90 行写出结构本身，再用一个训练循环把它从 $$\ln V$$ 训下来。每一步都先用小例子手算，再用 PyTorch、HuggingFace 或一次真实训练核对。

| 篇 | 回答的问题 | 一句话结论 | 必记 |
|---|---|---|---|
| [01 Transformer 长什么样](/transformer-architecture-from-a-sentence-to-the-next-token.html) | 原始 Transformer 的 encoder-decoder 如何变成 GPT-2 的 decoder-only？编号到词表上的概率分布经过哪些运算？ | 六种部件，只有 attention 让 token 交流；GPT 去掉 encoder 与 cross-attention，参考内容拼进输入 | $$\text{softmax}(QK^\top/\sqrt d + M)V$$<br/>打分沿特征维求和、加权沿 token 维求和<br/>FFN 占一层 2/3（4.72M 对 2.36M）<br/>GPT-2 small 124,439,808 个参数 |
| [02 一个 token 的旅程](/transformer-token-journey-training-and-inference.html) | 为什么一次前向有 $$T$$ 个训练信号？推理时为什么前面的 token 不用重算？ | causal mask + teacher forcing；旧 token 的 K、V 不随新 token 变，算一次存进 KV cache | 随机初始化 loss ≈ $$\ln V$$<br/>decode 一步读权重 0.12 ms、计算 0.0008 ms（GPT-2 small、A100）<br/>Llama-3-8B 8K 训练激活约 41 GiB（不含 $$[h, T, T]$$ 表）<br/>KV cache 128 KiB / token |
| [03 手搓 GPT（上）](/nanogpt-model-py-line-by-line.html) | 能加载 GPT-2 权重、能训能生成的 Transformer 最少写哪些东西？ | 结构不到 90 行；`GPT` 类其余部分是周边；与 HF 对拍相对差 $$9 \times 10^{-5}$$ | `c_attn` 一次算 QKV<br/>`c_proj` 初始化除 $$\sqrt{2L}$$（$$L = 12$$ 时 0.0041）<br/>推理只算最后位置的 lm_head（省 99.9%）<br/>每 token 855M FLOPs（前向 + 反向） |
| [04 手搓 GPT（下）](/nanogpt-train-py-and-training-a-model-that-writes.html) | 1.1 MB 文本到会续写的模型，每一步代码在哪、为什么那样写？层数从 4 改到 2 或 8，loss 和速度怎样？ | 参数与每步耗时随层数线性增长，loss 收益递减 | 每次迭代 491,520 token<br/>val loss 4.17 → 1.66（0.8M 参数、2000 步、7 分钟）<br/>2 / 4 / 8 层 val 1.82 / 1.66 / 1.59 |

Table: 四篇正文的问题、结论与必记

### 1. 本文的章节安排

| 章 | 内容 |
|---|---|
| 二 | 逐篇回顾：核心问题、结论、必记、常见误解 |
| 三 | 贯穿四篇的几条线：代码线、形状线、两种形态、算账线 |
| 四 | 常见误区表 |
| 五 | 通关自测：A 判断与计算 10 题、B 跨篇综合 5 题、C 面试题 7 题、D 掌握判据 |
| 六 | 下一步 |
| 七 | 延伸阅读：本系列不讲什么、去哪里读 |

Table: 本文的章节安排

## 二、逐篇回顾

### 1. 第一篇：Transformer 长什么样：从一句话到下一个 token

**核心问题**：原始 Transformer 的 encoder-decoder 如何变成 GPT-2 的 decoder-only？一个 token 的编号进入 GPT-2，到词表上的概率分布出来，又经过哪些运算？

**结论**：[原始 Transformer](https://arxiv.org/html/1706.03762v4) 是为机器翻译设计的 encoder-decoder（base 配置各 6 层、$$d_{\text{model}} = 512$$、8 头、$$d_{\text{ff}} = 2048$$）：encoder 用不加 mask 的双向 self-attention 读源句，decoder 用 causal self-attention 生成目标句，并通过 cross-attention（Q 来自 decoder，K、V 来自 encoder 输出）读源句。GPT-2 去掉 encoder，cross-attention 随之消失，要参考的内容直接拼进输入，由 causal self-attention 读取；LayerNorm 也从子层之后挪到子层之前（pre-norm）。

进入 GPT-2 后经过六种部件：

1. token embedding 查表，把编号变成 768 维向量（等价于 one-hot 乘矩阵，不算矩阵乘的 FLOPs）。
2. 位置 embedding：attention 是集合运算，打乱输入顺序只会让输出跟着换位置，所以位置必须显式给——GPT-2 加一张 1024 行的位置表，Llama 用 RoPE 在打分前按位置旋转 $$q$$、$$k$$，分数只依赖位置差。
3. attention：query 与所有 key 打分、除 $$\sqrt{d}$$ 防 softmax 饱和、causal mask 禁止看未来、softmax 得权重、加权求和 value；它是**唯一让 token 之间交流的地方**。多头是把投影结果切成 $$h$$ 段、每段单独做 softmax，换成了一个不同的函数，但打分与加权的 FLOPs 不变，只多一个 $$W_O$$。
4. FFN：逐 token「放大到 $$4d$$ → GELU → 压回」，提供非线性，占一层参数的 2/3。
5. 残差 + LayerNorm：残差流让 24 次修正训得动，LayerNorm 让每个子层看到同一尺度的输入；pre-norm 的主干只有一连串「+」，代价是末尾多一次 `ln_f`。
6. lm_head：与 embedding 表共享权重，第 $$v$$ 个词的 logit 就是最后的向量与 embedding 表第 $$v$$ 行的内积。

encoder-only（BERT）、decoder-only（GPT）、encoder-decoder（T5 / BART）是三条路线；decoder-only 成为主流有几种常见解释（训练目标与生成方式一致、所有任务共用一个输入接口），不是定论。

**必记**：

- $$\text{Attention}(Q, K, V) = \text{softmax}(QK^\top/\sqrt d + M)V$$；打分 $$S_{ij} = \sum_c Q_{ic} K_{jc}$$ 沿特征维求和，加权 $$\text{out}_{ic} = \sum_j P_{ij} V_{jc}$$ 沿 token 维求和——token 之间的信息在第二次求和里传递。
- $$d = 4$$、$$T = 3$$ 的手算：t2 的权重 $$(0.27, 0.27, 0.45)$$，输出 $$(0.73, 0.73, 0.27, 0.27)$$，与 PyTorch 对拍差 $$6 \times 10^{-8}$$。切成两个 $$d_h = 2$$ 的头时，t2 在头 0 上的权重是 $$(0.25, 0.25, 0.50)$$、在头 1 上是 $$(1/3, 1/3, 1/3)$$，拼起来 $$(0.75, 0.75, 0.33, 0.33)$$，与单头结果不同。
- 随机 $$q \cdot k$$ 的标准差 $$= \sqrt d$$（$$d = 128$$ 时 11.3）；分数差 10 时 softmax 是 $$(0.99995, 0.00005)$$、梯度约 0.00005，除以 $$\sqrt{128}$$ 后是 $$(0.71, 0.29)$$、梯度 0.21。
- LayerNorm 手算：$$(2, 4, 4, 6) \to (-1.41, 0, 0, 1.41)$$，均值与方差沿 $$d$$ 那一维算，每个 token 各算各的。
- GPT-2 small：12 层 × 7.09M + embedding 38.6M + 位置表 0.79M + `ln_f` 1,536 = 124,439,808；FFN 占一层 66.6%、embedding 占总量 31.6%；每层 attention 四个矩阵合计 2,362,368。
- GPT-2 真实的头：第 4 层第 11 头看上一个词、第 3 头把 it 指回 cat（0.84）。

**常见误解**：「参数主要在 attention 里」（FFN 是 2/3，attention 的打分、softmax、加权求和本身没有参数）；「mask 给了位置信息所以不用位置编码」（mask 只说谁在左边，不说在左边第几个）；「多头增加算量」（$$h$$ 个小乘法 = 一个大乘法，只多 $$W_O$$）；「多头是把输入的 768 维分给 12 个头」（切的是投影之后的结果，每个头都从完整的 768 维读入）。

### 2. 第二篇：一个 token 的旅程：训练侧与推理侧

**核心问题**：训练时一句话的 $$T$$ 个 token 进入模型，为什么一次前向就能得到 $$T$$ 个训练信号？推理时为什么前面的 token 不用重算——KV cache 里存的是什么、省了什么、花了什么？

**结论**：训练侧：$$T + 1$$ 个连续 token → 输入与右移一位的目标 → 一次前向 $$[B, T, V]$$ → $$B \times T$$ 个交叉熵取平均 → 反向沿原路 → AdamW。**causal mask** 让位置 $$i$$ 只用前文（与推理时只有前文的情形一致），**teacher forcing** 用真实 token 当目标（位置之间不互相等待，可并行），所以一句话 = $$T$$ 个样本，成本只是一次前向；代价是模型训练时没见过自己的错误（exposure bias）。反向过 attention 时，梯度沿注意力权重回传到被关注的 token：正向被关注得多，反向拿到的梯度就多。反向要用前向的中间量，所以激活值必须保存到反向结束，大小与 $$B \times T$$ 成正比。

推理侧：prefill 同训练前向、只用最后位置（决定 TTFT），decode 每步一个 token（决定 TPOT）。decode 读一遍全部权重只算 1 个 token，所以是访存密集；prefill 读同样多的权重算 $$T$$ 个 token，是计算密集。causal 结构下旧 token 的 K、V 不随新 token 改变——追加一个 token 后，有 mask 时前三行输出一个数都不变，无 mask 时全变——所以算一次存进 KV cache，每步只算新 token 的 Q、K、V 并追加。省的是计算，花的是显存。

**必记**：

- 随机初始化 loss $$\approx \ln V$$：16 词 2.78（实测 2.7805）、GPT-2 词表 10.8、65 字符 4.17。
- 三种形态：训练 $$[B, T]$$ 全位置有用、有反向、存激活值；prefill $$[B, T]$$ 只用最后位置、存 K/V；decode $$[B, 1]$$ 无 mask 矩阵、追加 K/V、访存瓶颈。
- GPT-2 small 在 A100 上：decode 一个 token 算 249 MFLOPs 要 0.0008 ms、读 249 MB 权重要 0.12 ms（读是算的 150 倍）；prefill 1024 个 token 算 255 GFLOPs 要 0.82 ms，读仍是 0.12 ms。
- 有 / 无 KV cache 输出逐 token 一致；$$d = 64$$、4 层、prompt 256 生成 256，快 7.9 倍；代价 Llama-3-8B 128 KiB / token（32 层 × 2 × 8 头 × 128 × 2 B），8K 上下文 1 GiB。
- Llama-3-8B、$$B = 1$$、$$T = 8192$$ 训练时，每层要为反向保存约 1.28 GiB 随 $$T$$ 线性增长的激活值（32 层 41 GiB，FFN 占 28 GiB），朴素 attention 的 $$[h, T, T]$$ 权重表每层再加 4 GiB（32 层 128 GiB，随 $$T^2$$ 增长）；参数只有 16 GB。
- `targets` 若是 `idx[:, 1:]` 这样的切片要用 `.reshape(-1)`，`.view(-1)` 会因不连续报错。

**常见误解**：「推理就是训练的前向」（对 prefill 大致成立，decode 是 $$[B, 1]$$ 输入、无 mask 矩阵、访存瓶颈的另一种形态）；「KV cache 也存 Q」（旧 token 的 Q 用过即弃）；「训练长上下文和推理长上下文一样难」（训练多一份与 $$B \times T$$ 成正比的激活值，8K 时 41 GiB 对 KV cache 的 1 GiB）；「双向 attention 也能用 KV cache」（新 token 会改变旧 token 的输出，缓存失效）。

### 3. 第三篇：手搓 GPT（上）：nanoGPT model.py 逐行解析

**核心问题**：一个能加载 GPT-2 权重、能训练、能生成的 Transformer，最少需要写哪些东西？nanoGPT 的每一行分别在实现前两篇的哪个方框、哪一步？

**结论**：330 行、6 个类，结构本身（`LayerNorm` + `CausalSelfAttention` + `MLP` + `Block`）不到 90 行。

1. `CausalSelfAttention`：Q、K、V 合成一个 $$d \to 3d$$ 的 `c_attn` 一次算完再 `split`；`view` + `transpose` 把头换到第 1 维以便批量矩阵乘（`view` 只换读法，不动数据）；有 SDPA 走融合 kernel，否则手写五行（打分缩放、mask、softmax、dropout、加权）；拼回去要 `contiguous()`；mask 叫 `bias` 是为了和 checkpoint 键名一致。
2. `GPT.__init__`：`ModuleDict` 让键名与 HF 一致；`wte.weight = lm_head.weight` 共享省 31% 参数；所有权重 $$\mathcal N(0, 0.02^2)$$，写回残差流的 `c_proj` 再除 $$\sqrt{2L}$$，让 $$2L$$ 次相加后方差不涨。
3. `forward`：传 `targets` 算全部位置的 lm_head 与交叉熵（训练），不传只算最后一个位置（推理）。
4. `generate`：裁到 `block_size` → 前向 → 除温度 → top-k → 按概率抽 → 接上；每步重算整段，没有 KV cache（教学取舍）。
5. `from_pretrained` 按键名拷 HF 权重，四个来自 `Conv1D` 的矩阵要转置；`configure_optimizers` 只对二维参数 decay；`estimate_mfu` 用 $$6N + 12LHQT$$。

**必记**：

- `x = x + self.attn(self.ln_1(x))`、`x = x + self.mlp(self.ln_2(x))`——pre-norm 两行。
- 与 HF GPT-2 对拍：最后位置 logits 相对差 $$9 \times 10^{-5}$$，argmax 一致，前五名 `so / too / a / hungry / the`。
- `get_num_params()` 默认扣 `wpe`：123.65M；加回 124.44M。`vocab_size = 50304` 是 50257 向上取到 64 的倍数。
- $$L = 12$$ 时 `c_proj` 标准差 $$0.02 / \sqrt{24} = 0.0041$$；不缩放，24 份写回后残差流标准差是一份的 4.9 倍。
- lm_head 对 1024 个位置全算要 79 GFLOPs，推理只算最后一个位置省 99.9%。
- logits $$(2.0, 1.0, 0.5, 0.0, -1.0)$$：$$\tau = 1$$ 时最大者 0.56、$$\tau = 0.5$$ 时 0.83、$$\tau = 2$$ 时 0.38，top-k = 2 时 0.73。
- 每 token FLOPs：$$6N = 742$$M + attention 打分与加权 113M = 855M；默认配置一次迭代 $$4.2 \times 10^{14}$$ FLOPs，一张 A100 跑满要 1.35 s，实测 3.4 s 即 MFU 40%。
- Llama 相对 GPT-2 只改五处：RMSNorm、RoPE、SwiGLU、GQA、去 bias（lm_head 不共享）。

**常见误解**：「三个独立的 Linear 与一个 `c_attn` 数学上不同」（等价，合并只为一次 GEMM）；「推理时 lm_head 也要算全部位置」（只算最后一个）；「`c_proj` 的特殊初始化是玄学」（$$2L$$ 次相加后方差不涨）；「`transpose` 之后直接 `view` 就行」（不连续，要先 `contiguous()`）。

### 4. 第四篇：手搓 GPT（下）：nanoGPT train.py 与训一个会续写的模型

**核心问题**：从一个 1.1 MB 的文本文件到一个能续写它的模型，中间每一步的代码在哪、为什么那样写？把层数从 4 改到 2 或 8，loss 和速度各会怎样？

**结论**：

1. 数据：`prepare.py` 字符级分词成 `uint16` 的 `train.bin` / `val.bin`（9 : 1）+ `meta.pkl`，词表大小 65 由数据决定。
2. 配置：`train.py` 的 72 个全局变量默认是 GPT-2 small 的复现配方，`configurator.py` 用 `exec` 让每个变量都能 `--name=value` 覆盖。
3. 取数：`get_batch` 用 `memmap` + 随机窗口 + `stack`（连续）造 `x` 与右移一位的 `y`，没有 epoch。
4. 模型：三种来源（从零 / 续训强制结构一致 / GPT-2 权重），再包 GradScaler（仅 fp16）、优化器、`compile`、DDP 四层。
5. 循环每步：设 lr（warmup + cosine）→ 到点评估、存 checkpoint 五样 → $$k$$ 个 micro-batch 累积（loss ÷ $$k$$、预取下一批、DDP 只在最后一步同步）→ unscale + 裁剪 → step → zero_grad → 日志与 MFU。

实跑 0.8M 参数 2000 步 7 分钟，loss $$\ln 65 = 4.17 \to 1.66$$，输出有莎士比亚格式；train / val 从 1000 步起拉开。改层数：参数与每步耗时线性增长，loss 收益递减。

**必记**：

- 默认配置每次迭代 $$40 \times 12 \times 1024 = 491{,}520$$ token；8 卡时每卡累积 5 次，总量不变。
- `get_lr` 默认取值：`it = 0` 时 $$3.0 \times 10^{-7}$$、999 时 $$3.0 \times 10^{-4}$$、2000 时峰值 $$6.0 \times 10^{-4}$$、301000 时 $$3.3 \times 10^{-4}$$、600000 时 $$6.0 \times 10^{-5}$$。实跑只覆盖了 `max_iters` 和 `lr_decay_iters`，`warmup_iters` 仍是 2000，所以那 2000 步全部在 warmup 里。
- fp16 最小正数约 $$6 \times 10^{-8}$$；GradScaler 默认乘 $$2^{16}$$，$$10^{-8}$$ 的梯度变成 $$6.6 \times 10^{-4}$$ 才存得下。
- 2 / 4 / 8 层：参数 0.40 / 0.80 / 1.58M，val loss 1.82 / 1.66 / 1.59，每步耗时约 0.5× / 1× / 2×；每加一层多 0.20M。
- checkpoint 五样：model、optimizer、model_args、iter_num、config。
- MFU：实跑每次迭代 40 个 768 token 的 micro-batch，共 30,720 token，按公式 0.14%（日志平滑后 0.20%）；分母仍按 A100 的 312 TFLOP/s 算，主因是小 micro-batch 让 MPS 上 launch-bound。

**常见误解**：「不除累积步数只是 loss 尺度不同」（等价于学习率放大 $$k$$ 倍）；「续训不需要优化器状态」（动量从零重估，前几百步抖）；「更深总是更好」（同样 7 分钟，2 层能跑 4000 步、8 层只能跑 1000 步）；「梯度裁剪是每个张量各自裁到 1」（裁的是全部梯度合起来的范数，方向不变）。

## 三、贯穿全系列的几条线

### 1. 代码线：一份代码逐步长出来

四篇共用一条能运行的代码线，每一站都有一个对拍的参照物，后一站在前一站验证过的基础上加东西：

| 篇 | 代码 | 加了什么 | 用什么验证 |
|---|---|---|---|
| 01 | `attention_by_hand.py` | 一个 attention 头的六步 | `F.scaled_dot_product_attention`，差 $$6 \times 10^{-8}$$ |
| 02 | `token_journey.py` | 极小 GPT（16 词、8 维、2 层）+ KV cache | 有 / 无 cache 逐 token 一致；loss ≈ $$\ln 16$$ |
| 03 | nanoGPT `model.py` | 真实尺寸的结构、权重加载、生成 | HF GPT-2 logits，相对差 $$9 \times 10^{-5}$$ |
| 04 | nanoGPT `train.py` | 数据、训练循环、checkpoint、DDP | 一次真实训练：4.17 → 1.66，2 / 4 / 8 层对比 |

Table: 四篇的代码线：每一站加了什么、用什么验证

### 2. 形状线：任何方框，先问输入输出形状

第一篇说「每个方框的输入输出都是 $$[T, d]$$」，后三篇不断回到这条线索：

| 张量 | 形状 | 在哪里出现 |
|---|---|---|
| 残差流 | $$[B, T, d]$$，从 `wte` 到 `ln_f` 不变 | 01 结构图；02 每一站的形状打印；03 `Block.forward` |
| 临时变宽 | `c_attn` 后 $$[B, T, 3d]$$，FFN 中间 $$[B, T, 4d]$$ | 02 形状图；03 `split` 与 `MLP` |
| 多头 | $$[B, h, T, d_h]$$，分数表 $$[B, h, T, T]$$ | 01 两头手算；03 `view` + `transpose` 的形状表 |
| logits | 训练 $$[B, T, V]$$，推理 $$[B, 1, V]$$ | 02 teacher forcing；03 `forward` 的两个分支 |
| KV cache | 每层 $$[B, h_{kv}, T, d_h]$$ 两份 | 02 prefill / decode；03 `generate` 没有它 |
| 一个 batch | `x`、`y` 各 $$[B, T]$$，`y` 是 `x` 右移一位 | 02 目标右移；04 `get_batch` 的手算 |

Table: 贯穿四篇的形状

### 3. 两种形态：同一套结构，训练与推理各走一条路

| 概念 | 01 结构 | 02 数据流 | 03 代码 | 04 训练 |
|---|---|---|---|---|
| causal mask | 禁止看未来，不给距离 | 让 $$T$$ 个信号不偷看；KV cache 成立的前提 | `is_causal=True` / `bias` buffer | 训练与评估同一个 mask |
| 只看最后位置 | lm_head 把 $$[T, d]$$ 变成 $$[T, V]$$ | prefill 只用最后一个分布 | `x[:, [-1], :]` 省 99.9% | `sample.py` 续写 |
| 激活值与 KV cache | — | 训练存激活、推理存 K / V | `generate` 不缓存，每步重算 | 梯度累积用小 micro-batch 控制激活 |
| dropout | — | 训练开、推理关 | `attn_dropout` / `resid_dropout` | `estimate_loss` 切 `eval()` / `train()` |

Table: 训练与推理两条线在四篇里的落点

### 4. 算账线：每个数字都能从形状推出来

参数、算量、显存三本账都从同一组形状出发：

1. **参数**：每层 $$12d^2 + 13d$$（attention $$4d^2 + 4d$$、FFN $$8d^2 + 5d$$、两个 LayerNorm $$4d$$），加 `wte`、`wpe`、`ln_f`；GPT-2 small 124,439,808，莎士比亚模型每层 0.20M。
2. **算量**：每 token $$6N + 12LHQT$$，GPT-2 small 855M；乘每次迭代的 token 数、除以时间与峰值，就是 MFU。
3. **显存**：训练是激活值，与 $$B \times T$$ 成正比（Llama-3-8B 8K 41 GiB）；推理是 KV cache，与 token 数成正比（128 KiB / token）。

## 四、常见误区

| 误区 | 为什么错 | 正确的说法 | 出处 |
|---|---|---|---|
| 大模型的参数主要在 attention 里 | attention 只有四个 $$d \times d$$ 矩阵，FFN 是 $$8d^2$$ | FFN 占一层参数的 2/3（GPT-2 small 4.72M 对 2.36M） | [第一篇](/transformer-architecture-from-a-sentence-to-the-next-token.html) |
| causal mask 已经提供位置信息 | mask 只约束可见范围 | 位置编码仍不可少：GPT-2 查表，Llama 用 RoPE | [第一篇](/transformer-architecture-from-a-sentence-to-the-next-token.html) |
| 多头比单头多算很多 | $$h \times 2T^2 d_h = 2T^2 d$$ | 只多一个 $$W_O$$ 和 $$h$$ 倍的权重表 | [第一篇](/transformer-architecture-from-a-sentence-to-the-next-token.html) |
| 推理就是训练时的前向 | decode 的输入是 $$[B, 1]$$，靠 cache 而不是重算 | 训练、prefill、decode 是三种形态，decode 瓶颈在访存 | [第二篇](/transformer-token-journey-training-and-inference.html) |
| KV cache 也要存 Q | 旧 token 的 Q 之后没人再用 | 只存 K、V，每层每 token 一份 | [第二篇](/transformer-token-journey-training-and-inference.html) |
| 训练显存的大头是参数 | 反向要用的激活值与 $$B \times T$$ 成正比 | Llama-3-8B 8K：激活 41 GiB、朴素权重表再 128 GiB，参数 16 GB | [第二篇](/transformer-token-journey-training-and-inference.html) |
| 三个独立的 Q/K/V Linear 与 `c_attn` 不等价 | 宽矩阵横着切开就是三个矩阵 | 合并只为一次 GEMM 更快 | [第三篇](/nanogpt-model-py-line-by-line.html) |
| 推理时 lm_head 也要算全部位置 | 下一个 token 只要最后一个分布 | 只算 `x[:, [-1], :]`，1024 个位置时省 79 GFLOPs 的 99.9% | [第三篇](/nanogpt-model-py-line-by-line.html) |
| 梯度累积不除步数只是 loss 尺度不同 | `.grad` 在 $$k$$ 个 micro-batch 上累加 | 不除等价于学习率放大 $$k$$ 倍 | [第四篇](/nanogpt-train-py-and-training-a-model-that-writes.html) |
| 加深模型总是更好 | 层串行，每步耗时随层数线性增长 | 同样时间内深模型步数更少，loss 收益递减、更易过拟合小数据 | [第四篇](/nanogpt-train-py-and-training-a-model-that-writes.html) |

Table: 常见误区与正确说法

## 五、通关自测

### A. 判断与计算（10 题）

1. 用第一篇的玩具模型：某个 token 的 query 与三个 key 的内积是 $$(2, 6, 4)$$，$$d = 4$$，它是第 2 个位置（能看全部三个）。softmax 权重是多少？如果不除 $$\sqrt d$$ 呢？

   <details markdown="1"><summary>答案</summary>

   除 2 后 $$(1, 3, 2)$$：$$e^1, e^3, e^2 = 2.72, 20.1, 7.39$$，和 30.2，权重 $$(0.09, 0.67, 0.24)$$。不除：$$e^2, e^6, e^4 = 7.39, 403, 54.6$$，权重 $$(0.016, 0.867, 0.117)$$——第二个 token 更独占，$$d$$ 越大越极端。

   </details>

2. 一个 8 层、$$d = 256$$、4 头、词表 1000、上下文 512、GPT-2 结构（带 bias、位置表、权重共享）的模型有多少参数？

   <details markdown="1"><summary>答案</summary>

   每层：attention $$256 \times 768 + 768 + 256 \times 256 + 256 = 263{,}168$$，FFN $$256 \times 1024 + 1024 + 1024 \times 256 + 256 = 525{,}568$$，两个 LN 1024，合计 789,760（即 $$12d^2 + 13d$$）；8 层 6,318,080；`wte` $$1000 \times 256 = 256{,}000$$，`wpe` $$512 \times 256 = 131{,}072$$，`ln_f` 512；总计 6,705,664 ≈ 6.71M。lm_head 共享不另算，头数不影响参数量。

   </details>

3. 训练时 $$B = 4$$、$$T = 2048$$、$$V = 128256$$、bf16。logits 张量多大？为什么第三篇的推理分支只算 `x[:, [-1], :]`？

   <details markdown="1"><summary>答案</summary>

   $$4 \times 2048 \times 128256 \times 2$$ B $$\approx 2.1$$ GB——训练时必须全算（每个位置都有目标）。推理只需要最后一个位置的分布，全算是浪费 $$T - 1$$ 倍的 lm_head 计算与这 2 GB 显存。

   </details>

4. prompt 500 个 token、生成 500 个：nanoGPT 的 `generate`（无 KV cache）总共前向了多少个 token 位置？第二篇的带 cache 版本呢？

   <details markdown="1"><summary>答案</summary>

   无 cache：第 $$k$$ 步算 $$500 + k$$ 个位置，$$\sum_{k=1}^{500}(500 + k) = 250{,}000 + 125{,}250 = 375{,}250$$。有 cache：prefill 500 + decode 500 = 1000。差 375 倍；只比 decode 阶段则是 375,250 对 500。第二篇实测的倍数（7.9 倍）小于此：每步还要读 cache，小矩阵乘的效率也低。

   </details>

5. GPT-2 small（12 层、12 头 × 64 维、bf16）每个 token 的 KV cache 多大？1024 个 token 呢？和 Llama-3-8B 的 128 KiB / token 比，差在哪？

   <details markdown="1"><summary>答案</summary>

   $$12 \text{ 层} \times 2 \times 768 \times 2$$ B $$= 36{,}864$$ B = 36 KiB / token，1024 个 token 36 MiB。Llama-3-8B 层数多（32 对 12），但 GQA 让 K、V 只有 $$8 \times 128 = 1024$$ 维（GPT-2 的 K、V 与 Q 同宽 768），所以每 token 是 $$32 \times 2 \times 1024 \times 2$$ B = 128 KiB。

   </details>

6. 一个 6 层的 GPT-2 结构模型，`c_proj.weight` 初始化的标准差是多少？不做这一步缩放，残差流上所有写回加完，标准差是一份的几倍？

   <details markdown="1"><summary>答案</summary>

   $$2L = 12$$ 份写回，$$0.02 / \sqrt{12} \approx 0.0058$$。不缩放时 12 份独立量相加，方差 12 倍，标准差 $$\sqrt{12} \approx 3.5$$ 倍。

   </details>

7. 三个词的 logits 是 $$(3, 1, 0)$$。$$\tau = 1$$、top-k = 2 时第一个词被抽中的概率是多少？$$\tau = 0.5$$、不做 top-k 呢？

   <details markdown="1"><summary>答案</summary>

   top-k = 2 砍掉第三个：$$e^3 / (e^3 + e^1) = 20.1 / 22.8 = 0.88$$。$$\tau = 0.5$$ 时 logits 变成 $$(6, 2, 0)$$：$$e^6 / (e^6 + e^2 + e^0) = 403 / 412 = 0.98$$。

   </details>

8. 第四篇的默认学习率配置下，`it = 151500` 时学习率是多少？第四篇的莎士比亚实跑（`max_iters = lr_decay_iters = 2000`，其余默认）第 1000 步的学习率又是多少，为什么？

   <details markdown="1"><summary>答案</summary>

   `decay_ratio` $$= (151500 - 2000) / (600000 - 2000) = 0.25$$，`coeff` $$= 0.5 \times (1 + \cos(\pi/4)) = 0.854$$，学习率 $$= 6 \times 10^{-5} + 0.854 \times 5.4 \times 10^{-4} \approx 5.2 \times 10^{-4}$$。实跑的 `warmup_iters` 仍是默认 2000，2000 步全在 warmup：第 1000 步是 $$6 \times 10^{-4} \times 1001 / 2001 \approx 3.0 \times 10^{-4}$$，cosine 那段没有出现。

   </details>

9. GPT-2 small 的默认配置在一张 A100 上，一次迭代用了 2.7 s。MFU 是多少？

   <details markdown="1"><summary>答案</summary>

   每 token 855M FLOPs × 491,520 token ≈ $$4.2 \times 10^{14}$$ FLOPs；$$4.2 \times 10^{14} / 2.7 \approx 1.56 \times 10^{14}$$ FLOP/s，除以 312 TFLOP/s 约 50%。

   </details>

10. 判断对错并说一句理由：(a) GPT-2 small 的 lm_head 与 embedding 共享权重省下约 31% 的参数；(b) decode 每步也需要一个 $$T \times T$$ 的 mask 矩阵；(c) nanoGPT 的 `generate` 自带 KV cache；(d) Llama-3-8B 训练 8K 上下文时，FlashAttention 能把 41 GiB 的线性激活也省掉。

    <details markdown="1"><summary>答案</summary>

    (a) 对：$$50257 \times 768 = 38.6$$M，占 124.4M 的 31%。(b) 错：新 token 本来就只看历史，天然满足 causal，不需要 mask 矩阵。(c) 错：每步重算整段，是教学取舍。(d) 错：FlashAttention 省的是每层 4 GiB 的 $$[h, T, T]$$ 权重表（32 层 128 GiB），线性部分仍约 41 GiB，要靠激活重算等手段。

    </details>

### B. 跨篇综合（5 题）

1. 第一篇的六种部件，在第三篇的 `model.py` 里各是哪个属性？第四篇调用的 `configure_optimizers` 会对其中哪些做 weight decay？

   <details markdown="1"><summary>答案</summary>

   第一篇 → 第三篇：token embedding = `transformer.wte`；位置 = `transformer.wpe`；attention = `h[i].attn`（`c_attn` + `c_proj`）；FFN = `h[i].mlp`（`c_fc` + `c_proj`）；残差 + LayerNorm = `Block.forward` 的两个加号 + `ln_1` / `ln_2` / `ln_f`；lm_head = `lm_head`，与 `wte` 是同一个 `Parameter`。第三篇的 `configure_optimizers`：二维以上的做 decay（`wte` / `lm_head`、`wpe`、`c_attn`、`c_proj`、`c_fc` 的 weight），一维的不做（所有 bias、LayerNorm 的 $$\gamma$$、$$\beta$$）。

   </details>

2. 第二篇的玩具模型 step 0 loss 是 2.78，第四篇莎士比亚模型是 4.17。两个数各从哪来？如果把第三篇 `_init_weights` 的 0.02 改成 1.0，第四篇 step 0 的 loss 还会接近 4.17 吗？

   <details markdown="1"><summary>答案</summary>

   第二篇：随机模型在词表上均匀乱猜，交叉熵 $$= \ln V$$，$$\ln 16 = 2.77$$、$$\ln 65 = 4.17$$。第三篇：这个「均匀」靠 $$\mathcal N(0, 0.02^2)$$ 的小初始化——logits 都接近 0，softmax 接近均匀。标准差改成 1.0，logits 变得极端，分布不再均匀，正确答案常被分到很小的概率，step 0 的 loss 会明显高于 4.17。所以 step 0 ≈ $$\ln V$$ 是检查初始化与 loss 实现的第一道关。

   </details>

3. 把第三篇 attention 里的 `is_causal=True` 改成 `False`（手写路径去掉 mask），用第四篇的配置训练。train loss 会怎样？生成会怎样？第二篇的 KV cache 还能用吗？

   <details markdown="1"><summary>答案</summary>

   第一篇：位置 $$i$$ 能看到 token $$i+1$$，而第四篇 `get_batch` 的 `y` 恰是 `x` 右移一位，模型把答案抄过来就行，train / val loss 都会很快降得很低，但什么都没学到。第二篇：推理时没有未来 token 可抄，生成是胡话——训练与推理的可见信息不一致了。KV cache 也失效：没有 mask 时追加一个 token 会改变所有旧 token 的输出（玩具例子 t0 从 $$(0.92, 0.43, 0.57, 0.08)$$ 变成 $$(0.82, 0.38, 0.62, 0.18)$$），上层的 K、V 随之全变。

   </details>

4. 要给第三篇的 nanoGPT 加上第二篇的 KV cache，至少要改哪几处？

   <details markdown="1"><summary>答案</summary>

   第二篇的三行加在第三篇 `CausalSelfAttention.forward` 里：`split` / 拆头之后把 cache 里的 K、V 沿 token 维接在前面、存回，再用新 token 的 Q 去打分；decode 那一步 Q 只有 1 行，看全部历史天然满足 causal，不需要 mask 矩阵。`GPT.forward` 的位置要接着历史长度往后数（`pos` 从已缓存的 token 数开始，否则 `wpe` 给的位置是错的）。`generate` 改成先对整段 prompt 做一次 prefill，之后每步只喂新抽出的 1 个 token。改完的验证方法就是第二篇那条：有 / 无 cache 的生成逐 token 一致。

   </details>

5. 第四篇的默认配置每次迭代 491,520 个 token，为什么写成 12 × 1024 的 micro-batch 累积 40 次，而不是一次喂 480 条序列？两种写法的梯度一样吗？

   <details markdown="1"><summary>答案</summary>

   第二篇：激活值与 $$B \times T$$ 成正比，480 × 1024 的激活是 12 × 1024 的 40 倍，一张卡放不下；梯度累积每次只留一个 micro-batch 的激活。第四篇：每个 micro-batch 的 loss 除以 40 再 `backward`，`.grad` 累加后等于大 batch 的平均梯度，数学上一样（dropout 的随机性除外）。第三篇：算量也一样，$$855\text{M} \times 491{,}520 \approx 4.2 \times 10^{14}$$ FLOPs，只是分 40 次做。8 卡 DDP 时每卡累积 5 次、只在最后一次同步梯度，总 token 数不变。

   </details>

### C. 面试题（7 题）

1. 不看资料写出一个多头 causal self-attention 的 `forward`，并说出每一行之后的形状。

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) `c_attn(x)` 得 $$[B, T, 3d]$$，`split` 成三个 $$[B, T, d]$$；(2) `view(B, T, h, d_h).transpose(1, 2)` 得 $$[B, h, T, d_h]$$；(3) `q @ k.transpose(-2, -1) / sqrt(d_h)` 得 $$[B, h, T, T]$$，除的是每头维度；(4) 下三角外填 $$-\infty$$、沿最后一维 softmax；(5) `att @ v` 得 $$[B, h, T, d_h]$$；(6) `transpose(1, 2).contiguous().view(B, T, d)`；(7) `c_proj` 混合各头，输出 $$[B, T, d]$$ 加回残差流。
   **追问方向**：为什么除 $$\sqrt{d_h}$$（随机点积标准差 $$\sqrt{d_h}$$、softmax 饱和梯度消失）；为什么要 `contiguous()`；多头多了多少 FLOPs；SDPA / FlashAttention 省的是什么（不物化 $$[T, T]$$ 表）。
   **好答案与一般答案的区别**：一般答案写出公式；好答案每行都能报形状，说清 mask 的值为什么是 $$-\infty$$ 而不是 0，并能指出两次求和分别沿哪一维。

   </details>

2. 讲讲 KV cache：存什么、为什么成立、省了什么、花了什么。

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 存每层每个 token 的 K、V，形状 $$[B, h_{kv}, T, d_h]$$，不存 Q；(2) 成立的前提是 causal：旧 token 的 $$x$$ 只依赖前文，新 token 不改变它们的 K、V，一层层往上都成立；(3) 省的是 decode 每步重算整段，500 + 500 的例子里前向位置从 375,250 降到 1000，实测 256 + 256 快 7.9 倍；(4) 花的是显存，Llama-3-8B 128 KiB / token，8K 上下文 1 GiB；(5) 有 / 无 cache 输出必须逐 token 一致，这是验证实现的方法。
   **追问方向**：双向模型为什么不能用；GQA 怎样把 KV 头从 32 减到 8 省 4 倍；decode 为什么访存密集（读一遍权重只算一个 token）。
   **好答案与一般答案的区别**：一般答案说「缓存之前算过的结果」；好答案说清为什么只有 causal 模型能缓存、为什么不存 Q，并能按层数 × 2 × KV 头 × 维度 × 字节当场算出大小。

   </details>

3. 原始 Transformer 是 encoder-decoder，GPT 是 decoder-only。区别在哪？为什么今天的 LLM 多是 decoder-only？

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) encoder 双向 self-attention 读源句，decoder causal self-attention 生成，cross-attention 的 Q 来自 decoder、K / V 来自 encoder 输出；(2) GPT 去掉 encoder 与 cross-attention，要参考的内容拼进 prompt 由 causal self-attention 读；(3) 归一化从 post-norm 改为 pre-norm；(4) 主流的常见解释：next-token prediction 既是训练目标也是生成方式，所有任务共用一个输入接口——这是解释之一，不是定论；并行训练不是 decoder-only 独有的优势。
   **追问方向**：BERT / T5 / BART 的训练目标；pre-norm 为什么更好训；decoder-only 的 KV cache 与 encoder-decoder 的 cross-attention cache 有什么不同。
   **好答案与一般答案的区别**：一般答案说「GPT 只用了 decoder」；好答案说出 decoder 少的是哪个子层、它的功能在 GPT 里由什么代替，并对「为什么是主流」给出有保留的回答。

   </details>

4. 给你一个 GPT-2 结构的 `config.json`（`n_layer`、`n_embd`、`n_head`、`vocab_size`、`n_positions`），估算参数量，并说出参数主要在哪。

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 每层 $$12d^2 + 13d$$：attention $$4d^2$$、FFN $$8d^2$$，bias 与 LayerNorm 是零头；(2) 加 `wte` $$V \times d$$、`wpe` $$n_{\text{positions}} \times d$$、`ln_f` $$2d$$，lm_head 共享不另算；(3) GPT-2 small：$$12 \times 7{,}087{,}872 + 38{,}597{,}376 + 786{,}432 + 1{,}536 = 124{,}439{,}808$$；(4) FFN 占每层 2/3，embedding 在小模型里占 31.6%、模型越大占比越小；(5) 头数不影响参数量。
   **追问方向**：nanoGPT 为什么打印 123.65M（扣了 `wpe`）；Llama 的五处改动各让参数怎么变（SwiGLU 三个矩阵、GQA 让 K、V 变窄、lm_head 不共享）；`vocab_size` 为什么填成 50304。
   **好答案与一般答案的区别**：一般答案记得「124M」；好答案能从形状逐项写出来，知道每一项落在 `state_dict` 的哪个键上。

   </details>

5. 为什么 Transformer 能叠几十层而不难训？说说残差、LayerNorm、pre-norm 和 nanoGPT 的初始化各起什么作用。

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 残差流：每个子层读一份、算修正量、加回主干，加法对梯度原样传递；(2) LayerNorm：每个 token 自己归一化，保证子层看到同一尺度，不跨 token 也不跨 batch；(3) pre-norm：LayerNorm 只作用在送进子层的那一份上，主干从 embedding 到 `ln_f` 只有一连串「+」，post-norm 深了要 warmup 与小学习率；(4) 初始化：权重 $$\mathcal N(0, 0.02^2)$$，写回残差流的 `c_proj` 再除 $$\sqrt{2L}$$，否则 GPT-2 small 24 份写回后残差流标准差放大 4.9 倍；(5) 代价：pre-norm 末尾要多一次 `ln_f`。
   **追问方向**：RMSNorm 与 LayerNorm 差在哪；为什么 LayerNorm 适合变长序列而 BatchNorm 不适合；$$\gamma$$、$$\beta$$ 为什么不做 weight decay。
   **好答案与一般答案的区别**：一般答案说「残差缓解梯度消失」；好答案能画出两种放法、说出 $$\sqrt{2L}$$ 这个数从方差相加推出来。

   </details>

6. 你从零训一个小 GPT，看哪些数字判断训练是否正常？

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) step 0 loss ≈ $$\ln V$$（65 字符 4.17、GPT-2 词表 10.8），偏离说明初始化或 loss 实现有问题；(2) 学习率曲线符合 warmup + cosine——注意 `warmup_iters` 比 `max_iters` 还长时整段都在 warmup；(3) train / val 的差距：莎士比亚模型从 1000 步起拉开（1.50 / 1.66），是过拟合的信号；(4) 梯度累积时 loss 除以步数、梯度裁剪到总范数 1.0；(5) 每次迭代的 token 数与 MFU：GPT-2 small 在 A100 上 40% 正常，MPS 上小 micro-batch 只有 0.2% 是 launch-bound；(6) 生成样本：格式先对，词义后来。
   **追问方向**：val loss 1.66 对应多少候选（$$e^{1.66} \approx 5.3$$）；续训时为什么要读回优化器状态；loss spike 怎么处理。
   **好答案与一般答案的区别**：一般答案说「看 loss 降不降」；好答案给出每个数字的期望值，并能说出偏离时先查哪一行代码。

   </details>

7. 训练与推理的显存各花在哪？以 Llama-3-8B、8K 上下文为例。

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 训练：参数 + 梯度 + 优化器状态 + 激活值；激活值随 $$B \times T$$ 增长，$$B = 1$$、$$T = 8192$$ 时线性部分约 41 GiB（FFN 占 28 GiB），朴素 attention 的权重表再 128 GiB，参数本身只有 16 GB；(2) FlashAttention 不存权重表，激活重算再用时间换显存；(3) 推理：参数 + KV cache，8K 上下文一条请求 1 GiB，一张 80 GB 卡除掉 16 GB 权重只能同时服务几十条；(4) 推理没有反向，中间量用完即弃。
   **追问方向**：$$T$$ 翻倍各项怎么变（线性部分翻倍、权重表 4 倍、KV cache 翻倍）；logits 那一项多大（$$8192 \times 128256$$，fp32 3.9 GiB）；GQA 对两边各有什么影响。
   **好答案与一般答案的区别**：一般答案说「训练要存梯度」；好答案能按层、按每 token 多少个数把 41 GiB 和 128 KiB 当场算出来。

   </details>

### D. 掌握判据

| 水平 | 表现 |
|---|---|
| 读过 | 能说出四篇各讲什么<br/>知道六种部件、teacher forcing、KV cache、prefill / decode、`c_attn`、梯度累积这些名词 |
| 掌握 | A 组能不翻书算出 8 题以上<br/>B 组能说出每题用了哪几篇的什么<br/>拿到一个 `config.json` 能数出参数、报出每一站的形状；能在 `model.py` / `train.py` 里指出每个公式在哪一行 |
| 能教人 | C 组每题能给出全部要点并预判追问<br/>能解释几个反直觉结论为什么成立：参数主要在 FFN、mask 不给位置、多头不增加 FLOPs、KV cache 只对 causal 模型成立、训练显存的大头是激活、更深不一定更好 |

Table: 掌握程度的判据

通关标准：A 组至少 8 题、B 组至少 4 题、C 组每题能说出一半以上要点。没过的部分回到第二章对应篇的「必记」，再回该篇正文；形状说不清的先回第一篇，把六种部件的输入输出形状重画一遍——后三篇的每个数字都是从那张图推出来的。

## 六、下一步

- 主线的下一站是[《现代 LLM 结构：从 GPT-2 到今天的演进》](/llm-architecture-evolution-roadmap-from-gpt2.html)：把本系列的 GPT-2 看成一组可替换的槽位，逐个讲 Llama 之后改了什么、改动怎样改变算量与显存。
- 想知道第四篇那 1 MB 文本换成 15T token 时差什么：[《预训练：从 tokenizer 到训练配方》](/pretraining-from-tokenizer-to-training-recipe.html)。
- 本系列在两张地图里的位置：[《AI-Infra 工程师学习地图》](/ai-infra-learning-roadmap.html)、[《AI 算法工程师学习地图》](/ai-algorithm-engineer-learning-roadmap.html)。

回到总纲：[《Transformer 原理与实现：从论文到手搓 GPT-2》](/transformer-and-llm-structure-implementation-and-evolution.html)。

## 七、延伸阅读

本系列只讨论经典 GPT-2 结构的原理、数据流与最小实现。以下内容与它紧邻，但不在范围内：

- **今天的结构改动**：RMSNorm、RoPE 的推导与外推、SwiGLU、GQA / MLA、长上下文、MoE、MTP、投机解码、浮点格式。本系列只在第三篇末尾列出 Llama 相对 GPT-2 的五处改动；逐项展开与算量 / 访存 / Roofline 的账在[《现代 LLM 结构：从 GPT-2 到今天的演进》](/llm-architecture-evolution-roadmap-from-gpt2.html)。
- **真实预训练**：BPE 分词与词表大小、数据抓取与去重、scaling law、训练配方与 loss spike。本系列用字符级分词和 1 MB 文本走通流水线；规模化之后的问题在[《预训练：从 tokenizer 到训练配方》](/pretraining-from-tokenizer-to-training-recipe.html)。
- **前置的深度学习基础**：反向传播、激活函数、初始化与归一化、优化器、RNN。本系列直接使用它们的结论，推导在[《深度学习基础：从反向传播到残差》](/deep-learning-foundations.html)。
- **kernel 内部**：FlashAttention 怎样不写出 $$[T, T]$$ 的表、Tensor Core 为什么要 64 对齐。本系列只说它省了什么，实现在[《GPU Kernel 工程：从 CUDA 执行模型到 FlashAttention》](/gpu-kernel-engineering.html)。
- **多卡训练**：DDP 之外的张量 / 流水 / 序列 / 专家并行、ZeRO、分布式 checkpoint 与容错。第四篇只读到 DDP 与单文件 `ckpt.pt`；其余在[《大规模训练工程：从并行策略到容错恢复》](/large-scale-training-from-parallelism-to-fault-tolerance.html)。
- **推理服务**：KV cache 的内存管理（PagedAttention）、continuous batching、调度。第二篇只讲 KV cache 为什么成立和多大；服务端的工程在[《大模型推理系统揭秘：从 vLLM 看 LLM Serving Infra 核心技术》](/deep-dive-into-vllm.html)。
- **训练之后**：SFT、RLHF / RLVR，以及采样策略、约束解码。见[《后训练：从 SFT 到可验证奖励》](/post-training-from-sft-to-verifiable-rewards.html)与[《高效推理与压缩（算法侧）：解码、投机、量化与 KV》](/efficient-inference-and-compression-for-llms.html)。

[^q0]: 四篇各一个核心问题，即总纲分章导读里的四问：GPT-2 的 decoder-only 结构从原始 encoder-decoder 怎么来、一个编号经过哪六种部件变成词表上的分布（第一篇）；为什么一次前向有 $$T$$ 个训练信号、为什么推理时旧 token 不用重算（第二篇：causal mask + teacher forcing，KV cache）；能加载 GPT-2 权重、能训能生成的 Transformer 最少写哪些东西（第三篇：四个类不到 90 行，加 `GPT` 的周边）；从 1.1 MB 文本到会续写的模型每一步代码在哪、改层数会怎样（第四篇：参数与耗时线性、loss 收益递减）。详见[第二章](#二逐篇回顾)。

[^q1]: $$\text{softmax}(QK^\top/\sqrt d + M)V$$，打分沿特征维、加权沿 token 维求和；$$d = 4$$ 手算 t2 权重 $$(0.27, 0.27, 0.45)$$；FFN 占一层 2/3；GPT-2 small 124,439,808（nanoGPT 报 123.65M）；随机初始化 loss ≈ $$\ln V$$（2.78 / 10.8 / 4.17）；decode 一步读 0.12 ms、算 0.0008 ms；KV cache Llama-3-8B 128 KiB / token、快 7.9 倍；Llama-3-8B 8K 训练激活 41 GiB + 朴素权重表 128 GiB；HF 对拍 $$9 \times 10^{-5}$$；`c_proj` 除 $$\sqrt{2L}$$（0.0041）；每 token 855M FLOPs；每次迭代 491,520 token；val loss 4.17 → 1.66；2 / 4 / 8 层 1.82 / 1.66 / 1.59。详见[第一章](#一总览系列回答的问题与主线)、[第三章](#三贯穿全系列的几条线)。

[^q2]: 用第五章的自测：A 组 10 题判断与计算（至少 8 题）、B 组 5 题跨篇综合（至少 4 题）、C 组 7 道面试题（每题说出一半以上要点）；D 组的表给出「读过 / 掌握 / 能教人」三级的表现。详见[第五章](#五通关自测)。
