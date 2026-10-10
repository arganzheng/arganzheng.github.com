---
layout: post
series: transformer-and-llm
title: "Transformer 与 LLM（02）：一个 token 的旅程——训练侧与推理侧"
subtitle: "The Dynamic View: What Happens to a Token During Training and During Inference"
tags: [Transformer, LLM, AI, AI-Infra]
catalog: true
---

> **本篇在系列中的位置。** 第一段的第二篇。第 01 篇给了静态结构，本篇讲 token 怎么流过它：训练侧的 teacher forcing 与反向，推理侧的 prefill、decode 与 KV cache；第 03 篇把这两条动态线对应到 nanoGPT 的代码。完整地图见[总纲](/transformer-and-llm-structure-implementation-and-evolution.html)。

上一篇把 Transformer 的每个方框打开看了一遍，但那是一张**静止**的图。同一台机器在两种场合下的运转方式很不一样：**训练**时一句话的几千个 token 一起进去、几千个 loss 一起出来、梯度沿原路返回、几十亿参数各挪一小步；**推理**时先把用户的问题一次算完，然后一个字一个字往外吐，每吐一个字只算一个 token。不搞清这两条动态线，就解释不了几件天天碰到的事：为什么训练一次前向能同时算 $$T$$ 个位置的预测、为什么推理"第一个字慢、后面快"、KV cache 到底缓存了什么、为什么训练长上下文时"激活值"比参数本身还占显存。

这一篇用一个 2 层、8 维、16 个词的玩具 GPT 把两条线走通，每一步打印形状和数字；再用 512 个 token 的模型实测 KV cache 带来的 8 倍加速。上一篇[《Transformer 与 LLM（01）：Transformer 长什么样——从一句话到下一个 token》](/transformer-architecture-from-a-sentence-to-the-next-token.html)画的是静态结构，这一篇画的是动态线；下一篇[《Transformer 与 LLM（03）：手搓 GPT（上）——nanoGPT model.py 逐行解析》](/nanogpt-model-py-line-by-line.html)把两篇的图变成 nanoGPT 的代码。

本篇要回答的核心问题是：

> **训练时一句话的 $$T$$ 个 token 进入模型，为什么一次前向就能得到 $$T$$ 个训练信号？推理时为什么前面的 token 不用重算——KV cache 里存的是什么、省了什么、花了什么？[^q0]**

## 一、总览：同一台机器的两种运转方式

| | 训练（一步） | 推理（一次请求） |
|---|---|---|
| 输入 | $$B$$ 句话、每句 $$T$$ 个 token，全部已知 | 一个 prompt（$$T$$ 个 token），之后每次 1 个新 token |
| 前向次数 | 1 次 | 1 次 prefill + 每个生成的 token 1 次 decode |
| 一次前向算多少位置 | $$T$$ 个位置全算、全有用 | prefill 算 $$T$$ 个只用最后一个；decode 只算 1 个 |
| 有没有反向 | 有：loss → 梯度 → 更新参数 | 无：参数冻结 |
| 显存大头 | 参数 + 梯度 + 优化器状态 + **激活值** | 参数 + **KV cache** |
| 需要存的中间量 | 每层的输入、Q/K/V、attention 权重、FFN 中间量（为反向） | 每层每个 token 的 K、V（为下一步） |

Table: 训练与推理：同一个模型，两条不同的动态线

![同一个模型的两条动态线。左：训练是一个循环——取一个 batch，前向得到 [B, T, V] 的 logits 并沿路保存激活值，B × T 个交叉熵平均成 loss，反向给每个参数一份 .grad，AdamW 更新参数，再取下一个 batch。右：推理时参数冻结——prompt 先做一次 prefill 并把每层的 K、V 写进 cache，然后在「只取最后位置的分布采样一个 token」与「只喂这 1 个 token 做 decode、读写 KV cache」之间循环，直到 EOS](/img/in-post/transformer-02-two-dynamic-lines.svg)

两条线共用上一篇的全部结构，区别只在"进什么、算几次、留什么"。下面先走训练线（第二至四章），再走推理线（第五、六章），最后把两侧的差别汇总（第七章）。

本文的章节安排：

| 章 | 主题 | 内容 |
|---|---|---|
| 二 | 训练侧的前向 | 一个 batch 的形状从头到尾；teacher forcing：一次前向 $$T$$ 个预测 |
| 三 | loss 与反向 | 交叉熵<br/>反向沿哪条路走<br/>为反向保存的激活值有多大 |
| 四 | 更新 | AdamW 走一步；一步的完整清单 |
| 五 | 推理侧：prefill 与 decode | 逐 token 生成<br/>为什么第一个 token 慢<br/>采样 |
| 六 | KV cache | 为什么旧 token 不用重算<br/>实测有 / 无 cache 一致且快 8 倍<br/>代价 |
| 七 | 两侧的差别 | mask、dropout、形状、显存构成的对照表 |
| 八 | 本文小结 |  |
| 九 | 自测 | 六道题 |

Table: 本文的章节安排

玩具模型：$$V = 16$$ 个词、$$d = 8$$、2 层、每层 2 个头、上下文上限 8。结构与 nanoGPT 完全一致（`wte` / `wpe` / `Block` / `ln_f` / `lm_head`），只多了一个可选的 KV cache——配套脚本 `token_journey.py` 就是本文全部数字的来源。下文所有形状、loss、梯度和计时都是这个玩具模型的实际输出，不是运行 nanoGPT 得到的；nanoGPT 的 GPT-2 small 换成 $$V = 50257$$、$$d = 768$$、12 层、上下文 1024，每一站的形状规律完全相同，只是数字变大。

## 二、训练侧的前向：一个 batch 的旅程

### 1. 形状从头到尾

训练时取 $$B = 2$$ 句话、每句 $$T = 5$$ 个 token 作为输入，让 `token_journey.py` 把每一站的输出形状打印出来：

```text title='B=2、T=5 时每一站的输出形状'
输入 idx: (2, 5)  目标 targets: (2, 5)
  wte 查表                                 (2, 5, 8)      ← 每个 token 一个 8 维向量
  wpe 查表                                 (5, 8)         ← 5 个位置各一个向量，广播加到每句上
  block0.attn.c_attn（QKV 合在一起）        (2, 5, 24)     ← 3 × 8：Q、K、V 一次算出再切开
  block0.attn 输出                         (2, 5, 8)
  block0.mlp.c_fc（放大 4 倍）              (2, 5, 32)
  block0.mlp 输出                          (2, 5, 8)
  block0 输出（残差流）                     (2, 5, 8)
  block1 ……                                (2, 5, 8)
  ln_f                                     (2, 5, 8)
  lm_head → logits                         (2, 5, 16)     ← 每个位置 16 个分数
loss（标量）: 2.7805   ≈ ln(16) = 2.7726
```

把这份输出画成图，条的宽度代表最后一维的大小：

![训练前向每一站的输出形状（B = 2、T = 5、d = 8、V = 16，数字取自上面的输出）：输入 idx 是 (2, 5) 的整数编号；wte 查表得 (2, 5, 8)，加上 (5, 8) 的位置向量；每个 block 里 c_attn 临时变宽到 (2, 5, 24) 再切成 Q、K、V，attention 输出回到 (2, 5, 8)，FFN 的 c_fc 放大到 (2, 5, 32) 再压回 (2, 5, 8)；两层之后 ln_f 仍是 (2, 5, 8)，lm_head 得到 (2, 5, 16) 的 logits，最后平均成一个标量 loss](/img/in-post/transformer-02-shape-flow.svg)

三件事情值得关注：

- 从 `wte` 到 `ln_f`，形状一直是 $$[B, T, d] = [2, 5, 8]$$——上一篇说的"每个方框输入输出相同"在这里变成一列相同的数字；只有两处临时变宽（QKV 合并成 $$3d$$、FFN 放大到 $$4d$$）又缩回来。
- `lm_head` 之后是 $$[B, T, V] = [2, 5, 16]$$：**每个位置**都有一个词表分布，不只是最后一个。这是下一节的关键。
- 随机初始化的 loss 恰好约等于 $$\ln V = \ln 16 = 2.77$$：模型还什么都不会，每个位置在 16 个词里均匀乱猜，交叉熵就是 $$\ln V$$（L0 第五篇）。GPT-2 词表 50257，训练开始时 loss 约 10.8，同理。

### 2. Teacher forcing：一次前向，T 个训练信号

训练数据是一段连续文本切出来的 $$T + 1$$ 个 token；前 $$T$$ 个当输入，**后 $$T$$ 个当目标**——目标就是输入右移一位：

```text title='Teacher forcing：目标是输入右移一位，T 个位置各一个 loss'
第 0 句的输入 : [5, 3, 8, 10, 14]
第 0 句的目标 : [3, 8, 10, 14, 12] （就是输入右移一位）
  位置 0: 看到 [5]                → 预测第 1 个 token，正确答案  3，模型给它的概率 0.068，loss 2.693
  位置 1: 看到 [5, 3]             → 预测第 2 个 token，正确答案  8，模型给它的概率 0.065，loss 2.734
  位置 2: 看到 [5, 3, 8]          → 预测第 3 个 token，正确答案 10，模型给它的概率 0.062，loss 2.783
  位置 3: 看到 [5, 3, 8, 10]      → 预测第 4 个 token，正确答案 14，模型给它的概率 0.062，loss 2.778
  位置 4: 看到 [5, 3, 8, 10, 14]  → 预测第 5 个 token，正确答案 12，模型给它的概率 0.061，loss 2.803
T 个 loss 的平均 = 2.7583（两句话一起平均就是上面的 2.7805）
```

![训练的一步：输入 The cat sat on the 的五个位置经过一次前向同时得到五个词表分布，目标是输入右移一位 cat sat on the mat，每个位置一个 −log p，平均成这一步的 loss；反向传播沿同一条路走回去，每个参数得到 .grad](/img/in-post/transformer-02-training-step-teacher-forcing.svg)

为什么一次前向能同时得到 5 个训练信号，而不是像推理那样一个一个来？两个条件：

1. **causal mask**（上一篇第三章第 5 节）保证位置 $$i$$ 的输出只用了 token $$0..i$$ 的信息——所以位置 2 的预测"看到 [5, 3, 8] 预测下一个"，和推理时只有这三个 token 的情形**完全一样**，没有偷看答案。
2. **目标用真实的下一个 token，而不是模型自己刚预测的那个**。如果位置 3 的输入要等位置 2 预测出来才知道，就串行了；直接把真实文本喂进去，5 个位置互不依赖，可以并行算。这个做法叫 **teacher forcing**（老师强制喂正确答案）。

于是一句 $$T$$ 个 token 的话提供 $$T$$ 个训练样本，成本只是一次前向——这是 decoder-only 训练效率高的根本原因（上一篇第七章）。代价是训练时模型从没见过"自己刚犯的错"，推理时错一个字后面就可能越错越远（exposure bias），后训练系列的 RL 部分处理这个问题。

## 三、loss 与反向

### 1. 交叉熵

每个位置的 loss 是 $$-\log p_i(y_i)$$：模型分给正确答案的概率取对数再取负（L0 第五篇讲了它为什么是这个形状）。位置 0 给正确答案 3 的概率是 0.068，$$-\ln 0.068 = 2.69$$。$$B \times T$$ 个 loss 取平均，得到一个标量——训练日志里那个数。

代码上就是一行：

```python title='交叉熵一行：F.cross_entropy'
loss = F.cross_entropy(logits.view(-1, V), targets.reshape(-1))   # [B·T, V] 对 [B·T]
```

`logits` 从 $$[2, 5, 16]$$ 拍平成 $$[10, 16]$$，`targets` 拍平成 $$[10]$$，10 个位置一起算。（顺带一个真实的坑：这里 `targets` 是 `idx[:, 1:]` 切出来的、在内存里不连续，用 `.view(-1)` 会报错，要用 `.reshape(-1)`——Infra PyTorch 系列第二篇讲的 stride 问题在训练代码里长这样。）

### 2. 反向沿哪条路

`loss.backward()` 从这个标量出发，沿前向的每一步反着走一遍（L3 第一篇的链式法则、Infra PyTorch 第三篇的 Autograd）：lm_head → `ln_f` → block1 的 FFN → block1 的 attention → block0 …… → `wpe`、`wte`。每个参数拿到一份和自己同形状的梯度：

```text title='backward 后每个参数拿到同形状的梯度'
  wte.weight                   参数 (16, 8)      梯度范数 1.0940
  wpe.weight                   参数 (8, 8)       梯度范数 0.6659
  blocks.0.ln_1.weight         参数 (8,)         梯度范数 0.4008
  blocks.0.attn.c_attn.weight  参数 (24, 8)      梯度范数 1.4743
  ……共 28 个参数张量、1,952 个数，每个都有 .grad
```

反向过 attention 时，梯度会**沿着注意力权重流回被关注的 token**：位置 4 的 loss 会通过 attention 权重影响 token 0 的表示——这就是"远处的 token 也能得到训练信号"的机制，RNN 做不到这一点（L3 第六篇）。

### 3. 为反向保存的东西：激活值

链式法则要用到前向的中间结果——算 $$W_1$$ 的梯度要有 FFN 的输入、算 $$V$$ 投影的梯度要有 attention 权重。所以前向时这些中间量不能丢，要一直留到反向用完。它们统称**激活值**（activations），每层要留：LayerNorm 的输入、Q / K / V、attention 权重 $$[B, h, T, T]$$、FFN 的 $$4d$$ 维中间量、各处的残差流……

关键性质：激活值的大小与 $$B \times T$$ **成正比**，而参数与它无关。玩具模型看不出来，换成 Llama-3-8B、$$B = 1$$、$$T = 8192$$：参数 16 GB，为反向保存的激活值按朴素实现要几十 GB——**比参数本身大**。这就是为什么训练长上下文比推理长上下文难得多、为什么有"激活值重算"（activation checkpointing，用时间换显存）这类技术；第六篇算这笔账，Infra 大规模训练系列讲怎么切。

![前向与反向的配合：前向从 wte + wpe 经 block0、block1、ln_f、lm_head 走到 loss，每一站把反向要用的激活值存下来（LayerNorm 的输入、Q/K/V、attention 权重、FFN 的 4d 中间量、logits）；反向从 loss 出发按 lm_head → ln_f → block1 → block0 → wte / wpe 的顺序走回去，每一站用自己存下的激活值算出本站参数的 .grad，用完即可释放；激活值的大小与 B × T 成正比，参数量与 B × T 无关](/img/in-post/transformer-02-forward-backward-activations.svg)

## 四、更新：AdamW 走一步

有了每个参数的 `.grad`，优化器把参数沿梯度反方向挪一小步（L3 第三篇的 AdamW：不是直接减梯度，而是维护每个参数的一阶、二阶动量再决定步长）。同一个 batch 更新前后：

```text title='AdamW 走一步：同一 batch 更新前后的 loss'
  更新前 2.7805 → 更新后 2.6832（同一 batch；真实训练每步换一个新 batch）
```

一步就下降了 0.1——但这是在**同一个 batch** 上量的，只说明更新方向对；真正的进步要看新 batch 上的 loss（L2 第一篇：训练集 vs 验证集）。

到这里训练的一步走完了。清单如下——这就是工具箱第三篇那二十行的每一行在模型内部对应的事：

1. 取一个 batch：$$T + 1$$ 个连续 token → 输入 $$x$$ 与目标 $$y$$（右移一位）；
2. 前向：$$[B, T] \to [B, T, d] \to \cdots \to [B, T, V]$$，沿路保存激活值；
3. loss：$$B \times T$$ 个交叉熵取平均；
4. 反向：沿原路算出每个参数的 `.grad`，用完释放激活值；
5. 更新：AdamW 用 `.grad` 与动量改参数；`zero_grad`；
6. 换下一个 batch，回到 1。几十万步之后就是一个语言模型（下下篇实训一个）。

## 五、推理侧：prefill 与 decode

### 1. 生成是一个循环

推理时参数冻结，模型只做前向，而且是**一个 token 一个 token**地生成：给 prompt，算出下一个 token 的分布，从中抽一个，接到 prompt 后面，再算、再抽……直到抽到结束符（EOS）或达到长度上限。用玩具模型走 5 步（贪心：每步取概率最大的）：

```text title='玩具模型贪心生成 5 步：prefill 与 decode 的形状'
  prefill：输入 (1, 3)，每层 cache 里 K 的形状 (1, 2, 3, 4)  [B, h, 已有 token 数, d_h]
  decode 第 1 步：输入形状 (1, 1)，cache 里 K 变成 (1, 2, 4, 4)，新 token 14
  decode 第 2 步：输入形状 (1, 1)，cache 里 K 变成 (1, 2, 5, 4)，新 token 0
  decode 第 3 步：输入形状 (1, 1)，cache 里 K 变成 (1, 2, 6, 4)，新 token 0
  ……
  生成: [3, 7, 1, 14, 0, 0, 0, 0]
```

（未训练的模型生成的是无意义的 token，这里只看形状；下下篇训完的模型会续写出像莎士比亚的句子。）

这个循环天然分成两个阶段：

- **prefill**（预填充）：prompt 的 $$T$$ 个 token 一次前向——和训练的前向**一模一样**，$$T$$ 个位置同时算，但只有最后一个位置的分布有用（它预测第一个新 token）。这一步是矩阵乘法为主的"计算密集"阶段，决定了用户等第一个字的时间（TTFT，time to first token）。
- **decode**（解码）：之后每一步只喂**1 个**新 token（形状 $$[1, 1]$$），得到 1 个分布，抽 1 个 token。这一步计算量很小、但要把全部参数从显存读一遍，是"访存密集"阶段，决定了后面每个字之间的间隔（TPOT / ITL）。第六篇用 Roofline 把这两个阶段的时间算出来。

"第一个字慢、后面快"就是 prefill 与 decode 的区别在用户体验上的样子：

![一次请求的时间线（示意，不按比例）：先是一大段 prefill，把 T 个 prompt token 一次前向算完，结束时吐出第一个 token mat，这一段就是 TTFT；之后是一串很短的 decode，每段只算 1 个 token、吐出 because、it、was、tired……，相邻两个字的间隔就是 TPOT，每步都要把全部权重从显存读一遍](/img/in-post/transformer-02-prefill-decode-timeline.svg)

### 2. 从分布里抽一个 token

每步得到的是词表上的一个分布，怎么变成一个 token？最简单是取概率最大的（贪心），但那样会重复、呆板；实际用**采样**：按概率随机抽，再用温度调节分布的尖锐程度、用 top-k / top-p 砍掉长尾（L0 第五篇讲温度，高效推理系列第一篇讲采样策略）。nanoGPT 的 `generate` 就是"温度 + top-k + 按概率抽"三行（下一篇）。采样意味着同一个 prompt 每次生成不同——这是特性不是 bug，评测时要固定它。

## 六、KV cache：为什么旧 token 不用重算

### 1. 朴素做法在浪费什么

最朴素的 decode：每生成一个 token，把**整段序列**（prompt + 已生成的）重新前向一遍，取最后一个位置。生成第 100 个 token 时要算 $$T + 100$$ 个位置，其中 $$T + 99$$ 个上一步刚算过。总计算量随生成长度平方增长。

哪些能省？回头看上一篇 attention 的六步：位置 $$i$$ 的 **K 和 V** 只由 token $$i$$ 自己的向量算出（$$k_i = x_i W_K$$），而 causal mask 保证 $$x_i$$ 本身只依赖 token $$0..i$$——**后面来了新 token，前面所有位置的 K、V 一个数都不会变**（L0 第四篇用图讲过这条因果性）。既然不变，算一次存下来就行。至于 Q：只有新 token 需要 Q（只有它要做预测），旧 token 的 Q 用过就没用了。

于是：

![推理的 prefill 与 decode：prefill 把 The cat sat on the 五个 token 一次算完并把每层的 K、V 存进 cache，取最后一个位置的分布抽出 mat；之后每步 decode 只算新 token 的 Q、K、V，Q 去看 cache 里全部 K，新的 K、V 追加进 cache，cache 从 5 个长到 6、7、8 个](/img/in-post/transformer-02-prefill-decode-kv-cache.svg)

- prefill：$$T$$ 个 token 正常前向，**顺手把每层每个 token 的 K、V 存进 cache**——每层一对张量，形状 $$[B, h, T, d_h]$$（玩具模型：$$[1, 2, 3, 4]$$）。
- decode：新 token 只算自己的 Q、K、V；它的 Q 去和 cache 里全部 K 打分、加权 cache 里全部 V；它的 K、V **追加**到 cache 末尾（$$[1, 2, 3, 4] \to [1, 2, 4, 4] \to \cdots$$）。每步的计算量从"整段"降到"1 个 token 过一遍模型 + 看一遍 cache"。

代码上只是在上一篇的 attention 里加三行：

```python title='KV cache 只在 attention 里加三行'
if cache["k"] is not None:                               # decode：把旧 K、V 接在前面
    k = torch.cat([cache["k"], k], dim=2)                # [B, h, 旧长度 + 1, d_h]
    v = torch.cat([cache["v"], v], dim=2)
cache["k"], cache["v"] = k, v                            # 存回去，下一步接着用
att = (q @ k.transpose(-2, -1)) / math.sqrt(d_h)         # q 只有 1 行，k 有全部历史
```

### 2. 实测：一致，且快 8 倍

有 cache 和没 cache 必须给出**完全相同**的结果，否则就是实现错了：

```text title='有无 cache 的生成结果完全一致'
  无 cache 生成: [3, 7, 1, 14, 0, 0, 0, 0]
  有 cache 生成: [3, 7, 1, 14, 0, 0, 0, 0]
  两者一致: True
```

把模型放大到 $$d = 64$$、4 层、prompt 256 个 token、生成 256 个（CPU）：

```text title='d=64、4 层、256+256 token：有 cache 快 7.9 倍'
  无 cache（每步重算整段，长度 256→512）: 0.74 s
  有 cache（每步只算 1 个 token）       : 0.09 s   快 7.9 倍
  代价：cache 里存着 4 层 × K、V × 512 token × 64 维 × 4 B = 1024 KiB
```

加速比大致等于"平均序列长度"（这里 384）除以"每步实际算的 token 数"（1）再打个折——生成越长、prompt 越长，省得越多。真实服务里 decode 阶段的每一步几乎全靠 cache。

### 3. 代价：显存

省下的是计算，花掉的是显存：每层、每个 token 都要存一份 K 和一份 V。Llama-3-8B：32 层 × 2（K、V）× 8 个 KV 头 × 128 维 × 2 字节（bf16）= **128 KiB / token**；一个 8K 上下文的请求 1 GiB，一张 80 GB 的卡除掉 16 GB 权重只能同时服务几十个这样的请求。这就是为什么第六篇整篇在讲怎么把 K、V 做小（GQA 把 KV 头从 32 减到 8 已经省了 4 倍；MLA 再压），也是 Infra vLLM 系列的核心矛盾（PagedAttention 就是给 KV cache 做的内存管理）。

## 七、两侧的差别

把两条线并排：

| | 训练 | 推理 prefill | 推理 decode |
|---|---|---|---|
| 一次前向的输入形状 | $$[B, T]$$，$$T$$ 几千 | $$[B, T]$$，$$T$$ = prompt 长度 | $$[B, 1]$$ |
| causal mask | $$T \times T$$ 下三角 | 同左 | 新 token 看全部历史，天然满足，不需要 mask 矩阵 |
| dropout | 开（`model.train()`） | 关（`model.eval()`） | 关 |
| 有用的输出位置 | 全部 $$T$$ 个 | 只有最后 1 个 | 那 1 个 |
| 保存的中间量 | 激活值（为反向，用完释放） | 每层的 K、V（存进 cache） | 追加 K、V |
| 反向 / 更新 | 有 | 无 | 无 |
| 瓶颈 | 计算 + 激活值显存 | 计算（TTFT） | 访存：每步读全部权重 + cache（TPOT） |
| 采样 | 无：目标是真实 token | 从最后位置的分布抽 1 个 | 同左 |

Table: 训练、prefill、decode 三种运转方式的差别

一个常见的误解是"推理就是训练的前向"。对 prefill 大致成立；decode 完全是另一种形态——$$[B, 1]$$ 的输入、没有 mask 矩阵、靠 cache 而不是靠重算、瓶颈在访存而不在计算。推理引擎（vLLM、TensorRT-LLM）的大部分工程都在优化 decode 这一形态。

## 八、本文小结

- **训练侧**：$$T + 1$$ 个连续 token → 输入与右移一位的目标 → 一次前向得到 $$[B, T, V]$$ 的 logits → $$B \times T$$ 个交叉熵取平均 → 反向沿原路给每个参数一份梯度 → AdamW 挪一步。
- 一次前向能同时得到 $$T$$ 个训练信号，靠两件事：**causal mask** 让每个位置的预测只用前文（与推理一致），**teacher forcing** 让目标用真实文本而不等模型自己预测（位置间可并行）。
- 反向要用前向的中间量，所以**激活值**必须保存到反向结束；它与 $$B \times T$$ 成正比，长上下文训练时比参数还大。
- **推理侧**：prefill 一次算完 prompt（同训练前向，只取最后位置），之后每步 decode 只算 1 个 token；"第一个字慢、后面快"就是这两个阶段。
- **KV cache**：causal 结构下旧 token 的 K、V 不随新 token 改变，算一次存下来；实测有 / 无 cache 输出完全一致、生成 256 个 token 快 7.9 倍；代价是每层每 token 一份 K、V（Llama-3-8B 128 KiB / token）。
- 训练、prefill、decode 是三种不同的运转形态，差别在输入形状、mask、dropout、保存什么、瓶颈在哪。

配套脚本 `token_journey.py`（[ai-learning-labs/transformer-and-llm](https://github.com/arganzheng/ai-learning-labs/tree/main/transformer-and-llm)）实现了带 KV cache 的极小 GPT，打印本文全部形状、逐位置 loss、梯度、一步更新、有 / 无 cache 的一致性与计时；`tools/gen_token_journey_svg.py` 生成本文的六张图。

## 九、自测

1. 一句 $$T = 2048$$ 的训练样本，一次前向为模型提供了多少个"预测下一个 token"的训练信号？靠什么保证这些信号没有偷看答案？

   <details markdown="1"><summary>答案</summary>

   2048 个（每个位置一个；最后一个位置预测第 2049 个 token，所以取 $$T + 1$$ 个 token）。靠 causal mask：位置 $$i$$ 的输出只依赖 token $$0..i$$。见[第二章第 2 节](#2-teacher-forcing一次前向t-个训练信号)。
   </details>

2. 随机初始化的 GPT-2（词表 50257）训练开始时 loss 大约是多少？为什么？

   <details markdown="1"><summary>答案</summary>

   约 $$\ln 50257 = 10.8$$。随机模型在每个位置对词表均匀乱猜，正确答案的概率约 $$1/V$$，交叉熵 $$= -\ln(1/V) = \ln V$$。玩具模型 $$V = 16$$ 时是 2.78，见[第二章第 1 节](#1-形状从头到尾)。
   </details>

3. 训练 Llama-3-8B、$$T = 8192$$ 时，为什么显存里"激活值"比 16 GB 的参数还大？推理同样的上下文为什么没有这个问题？

   <details markdown="1"><summary>答案</summary>

   反向传播要用前向的中间量（每层的输入、Q/K/V、attention 权重、FFN 中间量），它们必须保存到反向结束，大小与 $$B \times T$$ 成正比。推理没有反向，中间量用完即弃，只需要留每层的 K、V（KV cache）。见[第三章第 3 节](#3-为反向保存的东西激活值)、[第六章第 3 节](#3-代价显存)。
   </details>

4. KV cache 为什么只存 K 和 V、不存 Q？

   <details markdown="1"><summary>答案</summary>

   Q 只在"做预测的那个位置"用一次：新 token 的 Q 去看所有 K。旧 token 的 Q 在它自己那一步用完就没用了——之后没有人再以它为 query。而旧 token 的 K、V 每一步都被新 token 看，所以要存。见[第六章第 1 节](#1-朴素做法在浪费什么)。
   </details>

5. 一个请求 prompt 500 个 token、生成 500 个 token。无 KV cache 时 decode 阶段总共前向了多少个 token 位置？有 cache 时呢？

   <details markdown="1"><summary>答案</summary>

   无 cache：第 $$k$$ 步算 $$500 + k$$ 个位置，$$\sum_{k=1}^{500}(500 + k) \approx 375{,}000$$ 个。有 cache：每步 1 个，共 500 个（prefill 另算 500 个）。差 750 倍——实测的 7.9 倍小于此，因为每步还要读 cache、且小矩阵乘的效率低；生成越长差距越大。见[第六章第 2 节](#2-实测一致且快-8-倍)。
   </details>

6. decode 每步的输入形状是 $$[B, 1]$$，为什么说它是"访存密集"而 prefill 是"计算密集"？

   <details markdown="1"><summary>答案</summary>

   每步不管算几个 token，全部参数都要从显存读一遍；prefill 一次读参数算 $$T$$ 个 token，平均每个 token 摊到的读取量小、计算多；decode 一次读参数只算 1 个 token，读的时间远大于算的时间。第六篇用 Roofline 把这两个数算出来。见[第五章第 1 节](#1-生成是一个循环)、[第七章](#七两侧的差别)。
   </details>

## 下一篇

到这里，两篇的图都画完了：[《Transformer 与 LLM（01）：Transformer 长什么样——从一句话到下一个 token》](/transformer-architecture-from-a-sentence-to-the-next-token.html)画的是静态结构（每个方框是什么、形状怎么变），这一篇画的是动态线（训练的「teacher forcing → T 个 loss → 反向 → 更新」，推理的「prefill → decode → KV cache」）。下一篇[《Transformer 与 LLM（03）：手搓 GPT（上）——nanoGPT model.py 逐行解析》](/nanogpt-model-py-line-by-line.html)把它们变成代码：330 行、6 个类，每一行对应到前两篇的哪个方框、哪一步。

[^q0]: **训练**：目标是输入右移一位，causal mask 保证位置 $$i$$ 只用前文，teacher forcing 用真实 token 当目标，于是 $$T$$ 个位置互不依赖、一次前向同时得到 $$T$$ 个交叉熵；反向沿原路给每个参数梯度，为此要保存与 $$B \times T$$ 成正比的激活值。**推理**：causal 结构下旧 token 的 K、V 不随新 token 改变，prefill 算一次存进 KV cache，decode 每步只算新 token 的 Q、K、V 并追加——省的是每步重算整段的计算（实测 7.9 倍），花的是每层每 token 一份 K、V 的显存（Llama-3-8B 128 KiB / token）。详见[第二](#二训练侧的前向一个-batch-的旅程)、[三](#三loss-与反向)、[六章](#六kv-cache为什么旧-token-不用重算)。
