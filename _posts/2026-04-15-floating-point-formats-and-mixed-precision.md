---
layout: post
title: "Transformer 与 LLM（06）：浮点格式、数值稳定性与混合精度"
subtitle: "Floating-Point Formats, Numerical Stability and Mixed Precision"
tags: [Transformer, LLM, AI, AI-Infra]
catalog: true
---

> 本文是[《Transformer 与 LLM：结构、算量与数值》](/transformer-and-llm-for-infra-engineers.html)系列的第六篇（共七篇）。上一篇：[MoE：路由、激活参数量与通信形态](/moe-compute-and-communication.html)　下一篇：[量化、投机解码与 LoRA](/quantization-speculative-decoding-and-lora.html)

前五篇算了大量的字节数：Llama-3-8B 的权重 16.06 GB、KV cache 每 token 128 KiB、decode 一步至少搬 16 GB。所有这些数字都默认"每个数占 2 字节"，也就是 BF16。这一篇把镜头再推近一层，从"每个数占几个字节"进入"这几个字节里到底存了什么"，回答一个在训练和推理系统里都绕不开的问题：

> **BF16 的相对精度只有 FP16 的 1/8，为什么它反而成了训练的默认格式？把它同时用在权重更新上会出什么问题？**

这个问题的答案不需要任何"经验"，只需要把浮点数的定义写出来、把几个真实的量级代进去。本篇的写法和前几篇一样：公式 → 代入真实数字 → 得到结论 → 这个结论对系统意味着什么。所有数字都是从格式定义推出来的理论值，不是任何硬件的实测。


## 一、浮点数的一般形式

### 1. 三段位与四类编码

任何 IEEE 754 风格的浮点格式都由三段位组成：1 位符号 $$s$$、$$E$$ 位指数 $$e$$、$$M$$ 位尾数 $$m$$。一个正规数（normal）的值是

$$
x = (-1)^s \cdot 1.m \cdot 2^{\,e - \text{bias}}, \qquad \text{bias} = 2^{E-1} - 1
$$

"$$1.m$$"里的 1 是隐含位，不占存储，所以 $$M$$ 位尾数实际提供 $$M+1$$ 位有效数字。指数字段的全 0 和全 1 被保留下来编码三类特殊值：

- 指数全 0、尾数非 0：次正规数（subnormal），$$x = (-1)^s \cdot 0.m \cdot 2^{1-\text{bias}}$$，没有隐含 1，用来填补最小正规数与 0 之间的空隙，但有效位数随数值减小而减少；
- 指数全 0、尾数全 0：$$\pm 0$$；
- 指数全 1、尾数全 0：$$\pm\infty$$；
- 指数全 1、尾数非 0：NaN。

由这三段位可以直接推出每种格式的四个关键量：

$$
\begin{aligned}
x_{\max} &= (2 - 2^{-M}) \cdot 2^{2^{E-1}-1} \\
x_{\min}^{\text{normal}} &= 2^{\,2 - 2^{E-1}} \\
x_{\min}^{\text{subnormal}} &= 2^{\,2 - 2^{E-1} - M} \\
\varepsilon &= 2^{-M}
\end{aligned}
$$

$$\varepsilon$$ 是机器精度：1 与比 1 大的下一个可表示数之间的距离。舍入到最近时，任何实数被表示后的相对误差不超过单位舍入 $$u = \varepsilon / 2$$。这两个数决定"精度"，$$E$$ 决定"范围"，两者在固定的总位数里此消彼长。

### 2. 八种格式的位布局

把 $$E$$ 和 $$M$$ 代进上面的公式，得到本篇会反复用到的一张表：

```text
格式     符号/指数/尾数   bias   最大值        最小正规数     最小次正规数   ε=2^-M      有效十进制位
FP32     1 / 8 / 23      127    3.40e38       1.18e-38       1.40e-45       1.19e-7     ~7
TF32     1 / 8 / 10      127    3.40e38       1.18e-38       1.15e-41       9.77e-4     ~3
FP16     1 / 5 / 10      15     65504         6.10e-5        5.96e-8        9.77e-4     ~3
BF16     1 / 8 / 7       127    3.39e38       1.18e-38       9.18e-41       7.81e-3     ~2
E4M3     1 / 4 / 3       7      448           1.56e-2        1.95e-3        0.125       ~1
E5M2     1 / 5 / 2       15     57344         6.10e-5        1.53e-5        0.25        <1
INT8     整数            —      127（-128）   1（步长）      —              —           —
INT4     整数            —      7（-8）       1（步长）      —              —           —
```

几处需要单独说明：

- **FP16** 的最大值 $$(2 - 2^{-10}) \cdot 2^{15} = 65504$$，最小正规数 $$2^{-14} \approx 6.1 \times 10^{-5}$$，最小次正规数 $$2^{-24} \approx 6 \times 10^{-8}$$。它的范围只有约 12 个十进制数量级。
- **BF16** 就是 FP32 砍掉低 16 位尾数：指数位与 FP32 完全相同，所以范围一样是 $$\sim 3.4 \times 10^{38}$$，最小正规数一样是 $$2^{-126}$$；代价是尾数只剩 7 位，$$\varepsilon = 2^{-7} \approx 0.0078$$，单位舍入 $$u = 2^{-8} \approx 0.0039$$。FP32 与 BF16 之间的转换只是截断/舍入低 16 位，硬件上几乎免费。
- **TF32** 不是一种内存格式。它是 Ampere 起 Tensor Core 接受 FP32 输入时在**乘法器入口**做的截断：把 23 位尾数截成 10 位（与 FP16 同精度），指数保留 8 位（与 FP32 同范围），乘积再用 FP32 累加。张量在显存里仍然是 32 位、4 字节，`torch.float32` 的 dtype 不变；开关是 `torch.backends.cuda.matmul.allow_tf32` 或 `torch.set_float32_matmul_precision("high")`。这意味着"FP32 GEMM"在默认开启 TF32 的框架里，输入精度其实只有 $$2^{-10}$$。
- **E4M3** 是 FP8 的"精度型"变体，但它偏离了 IEEE 惯例：指数全 1 不再保留给 inf，$$S.1111.110$$ 是合法的最大数 $$1.75 \times 2^{8} = 448$$，只有 $$S.1111.111$$ 一个编码（两个符号）留给 NaN。没有 inf 意味着溢出会被饱和（saturate）成 448 或直接变 NaN，取决于转换指令的模式。PyTorch 中的 dtype 名 `float8_e4m3fn` 里的 "fn" 就是 finite + NaN-only 的意思。
- **E5M2** 是 FP8 的"范围型"变体，与 FP16 共享指数结构（bias 15），可以看作 FP16 砍掉 8 位尾数，保留 IEEE 的 inf/NaN 约定：最大值 $$1.75 \times 2^{15} = 57344$$，最小正规数与 FP16 同为 $$6.1 \times 10^{-5}$$。它只有 3 位有效数字（含隐含位），$$\varepsilon = 0.25$$。
- **INT8 / INT4** 不是浮点，没有指数，所有可表示数等距分布。它们必须搭配一个（通常是 FP16/FP32 的）缩放因子 scale 才能表示实数，这个 scale 的粒度问题正是第六节和第七篇的主题。

### 3. 一个直观的比较

用"数轴上可表示数的分布"来看这些格式最清楚。浮点数在每个二进制数量级 $$[2^k, 2^{k+1})$$ 内均匀放 $$2^M$$ 个点，所以相邻可表示数的间距是 $$2^{k-M}$$，随数值大小成比例变化——这是"相对精度恒定"的来源。BF16 每个数量级放 128 个点，FP16 放 1024 个，E4M3 放 8 个，E5M2 放 4 个。而 INT8 在整个 $$[-128, 127]$$ 上只放 256 个等距点，对靠近 0 的小数没有任何额外照顾。

把它们画在同一根对数轴上：FP16 覆盖 $$[6 \times 10^{-8}, 6.5 \times 10^4]$$，BF16 覆盖 $$[9 \times 10^{-41}, 3.4 \times 10^{38}]$$，E4M3 覆盖 $$[2 \times 10^{-3}, 448]$$，E5M2 覆盖 $$[1.5 \times 10^{-5}, 5.7 \times 10^4]$$。下面几节要回答的就是：一个 LLM 的权重、激活、梯度、优化器状态分别落在这根轴的哪一段，以及每一段需要多少个点。


## 二、指数位与尾数位的取舍：为什么深度学习选范围

### 1. 固定 16 位的两种分法

FP16 和 BF16 都是 16 位，差别只是把 3 位从尾数挪到了指数。这 3 位的代价与收益分别是：

- 精度：FP16 有 11 位有效数字，BF16 有 8 位，相对精度差 $$2^3 = 8$$ 倍。这就是标题里"BF16 的相对精度只有 FP16 的 1/8"的来源。
- 范围：FP16 的指数范围是 $$[-14, 15]$$，共 30 个二进制数量级（次正规再加 10 个）；BF16 是 $$[-126, 127]$$，共 254 个。范围差 $$2^{224}$$ 倍——但这个数字本身没有意义，有意义的是训练里的数落不落在这个范围里。

### 2. 训练中的数跨多少个数量级

一次训练迭代中至少有四类张量：权重、激活、激活梯度、权重梯度。Micikevicius 等 2017 在提出混合精度训练时给出了一个典型网络的激活梯度直方图：大部分值分布在 $$2^{-40}$$ 到 $$2^{0}$$ 之间，其中相当一部分低于 FP16 的最小正规数 $$2^{-14}$$，甚至低于最小次正规数 $$2^{-24}$$。这不是特例：反向传播中梯度是链式乘积，经过几十层之后量级漂移几个十进制数量级是常态；而在同一个张量内部，不同通道的梯度差三四个数量级也是常态。

与此相对，深度学习对**相对精度**的需求非常低。权重初始化本身就是随机的，训练过程中每步更新带来的相对变化在 $$10^{-3}$$ 量级；SGD 本身是一个带噪声的优化过程，前向和反向计算中引入 $$0.4\%$$（$$2^{-8}$$）的相对舍入噪声，只是在已有的 mini-batch 噪声上再叠一层小得多的噪声。大量的经验（Kalamkar 等 2019 对 BF16 训练的系统研究）表明，前向与反向的矩阵乘用 8 位有效数字，模型的收敛曲线与 FP32 基本重合。

所以取舍很清楚：训练需要范围，不太需要精度。3 位尾数换 3 位指数，让梯度不需要任何缩放就能直接落进可表示区间，同时 BF16 与 FP32 的转换只是截断。这就是核心问题前半部分的答案：**BF16 不是"精度更差却被选中"，而是它把精度花在了训练不需要的地方省下来，换成了训练需要的范围**。

### 3. 答案的另一半

但"前向反向不需要精度"不等于"训练的每一步都不需要精度"。有一个环节对精度极其敏感，就是权重更新 $$w \leftarrow w + \Delta w$$。第五节会给出具体数字：$$\Delta w / w$$ 的量级低于 BF16 的单位舍入，直接在 BF16 中做这一步，更新会被舍回原值。这是核心问题后半部分的答案，也是"FP32 master weights"存在的全部理由。在此之前，先把浮点误差发生的几个典型位置讲清楚。


## 三、数值在哪些地方丢失

### 1. 大数吃小数

浮点加法要先把两个数对齐到同一个指数，再加尾数。当两个数相差超过 $$2^{M+1}$$ 倍时，小的那个在对齐后会完全移出尾数窗口。FP16 中：

$$
10^4 + 1 \;\to\; 10^4
$$

因为 $$10^4$$ 落在 $$[2^{13}, 2^{14})$$，这个区间内 FP16 相邻可表示数的间距是 $$2^{13-10} = 8$$，加 1 不足半个间距，舍回 10000。同一个例子在 BF16 中更糟：间距是 $$2^{13-7} = 64$$，10000 本身就不可表示（会被存成 9984 或 10048），加 1 到加 31 都没有任何效果。

这个现象在 LLM 里的直接体现就是残差流：$$h_{l+1} = h_l + f(h_l)$$，几十层累加后 $$h_l$$ 的量级可能是 $$f(h_l)$$ 的几十上百倍，如果残差流本身存成 BF16，后面层的贡献会被系统性地吃掉低位。这是很多训练框架把残差流的加法在 FP32 中完成的原因。

### 2. 长求和的误差积累

对 $$n$$ 个数依次相加，每次加法引入至多 $$u$$ 的相对误差。最坏情况下误差以线性累积：

$$
\left| \text{fl}\Big(\sum_{i=1}^{n} x_i\Big) - \sum_{i=1}^{n} x_i \right| \le (n-1)\, u \sum_{i=1}^{n} |x_i| + O(u^2)
$$

即朴素求和的误差界是 $$O(n u)$$。如果舍入误差是独立随机的，实际误差按随机游走增长，量级是 $$O(\sqrt{n}\, u)$$。两个改进方向：

- **pairwise（树形）求和**：把数组两两配对相加，再对结果两两配对，递归深度 $$\log_2 n$$，每个元素只经过 $$\log_2 n$$ 次加法，误差界是 $$O(u \log n)$$；
- **Kahan 补偿求和**：用一个额外变量记录每次加法丢掉的低位，下一次加回去，误差界是 $$O(u)$$，与 $$n$$ 无关，代价是每个元素 4 次浮点运算。

GPU 上的归约天然接近 pairwise：一个 block 内的 reduce 是 warp shuffle 树，block 之间再做一层树；一个 $$n = 4096$$ 的求和在硬件上的实际累加深度大约是 $$\log_2 4096 = 12$$ 而不是 4096。这是为什么同样用 FP32 累加，GPU 上的 `sum()` 结果往往比 CPU 上的顺序循环更接近真值，也是为什么两者的结果不会逐位相同。

### 3. softmax 的指数溢出

attention 的 softmax 是 $$\text{softmax}(x)_i = e^{x_i} / \sum_j e^{x_j}$$。$$e^x$$ 在 FP16 中溢出的条件是

$$
e^{x} > 65504 \iff x > \ln 65504 \approx 11.09
$$

11.09 是一个非常小的阈值：attention logit 是 $$q \cdot k / \sqrt{d_{head}}$$，训练后期超过 11 是家常便饭（第七节会讨论 logit 增长）。BF16 和 FP32 的阈值是 $$\ln(3.4 \times 10^{38}) \approx 88.7$$，宽松得多但也不是不可能。标准解法是利用 softmax 的平移不变性，先减去最大值：

$$
\text{softmax}(x)_i = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \qquad m = \max_j x_j
$$

减完之后指数的自变量全部 $$\le 0$$，分子在 $$(0, 1]$$ 内不会溢出；分母至少有一项是 1，不会下溢成 0。

FlashAttention 一类 kernel 使用的 online softmax（Milakov 与 Gimelshein 2018）把这个技巧变成了可以分块累加的形式：处理到第 $$t$$ 块时维护当前最大值 $$m_t$$ 与分母 $$\ell_t$$，遇到新块的最大值 $$m'$$ 时更新

$$
m_{t+1} = \max(m_t, m'), \qquad \ell_{t+1} = \ell_t \cdot e^{m_t - m_{t+1}} + \sum_{j \in \text{block}} e^{x_j - m_{t+1}}
$$

这个重缩放（rescale）常被当作"为了分块而付出的代价"来讲，但它首先是数值动机：如果不随时相对于当前最大值重缩放，分块内的 $$e^{x_j}$$ 就没有溢出保护。第三篇讨论过 FlashAttention 的访存收益，这里补上它的数值前提。

### 4. 方差计算中的相消

方差有两个数学上等价的公式：

$$
\text{Var}(x) = \mathbb{E}[x^2] - (\mathbb{E}[x])^2 = \mathbb{E}\big[(x - \mathbb{E}[x])^2\big]
$$

前者只需要一遍扫描，但在浮点中是灾难性相消的教科书案例：如果 $$x \approx 1000 \pm 1$$，则 $$\mathbb{E}[x^2] \approx 10^6$$，方差约 1，需要从两个 $$10^6$$ 量级的数相减得到一个 $$10^0$$ 量级的结果，损失 6 位有效数字——FP32 只有 7 位，结果基本是噪声；BF16 只有 2–3 位，结果直接是负数或 0。Welford 1962 的在线算法维护运行均值 $$\mu_t$$ 与偏差平方和 $$M_t$$：

$$
\mu_t = \mu_{t-1} + \frac{x_t - \mu_{t-1}}{t}, \qquad M_t = M_{t-1} + (x_t - \mu_{t-1})(x_t - \mu_t)
$$

每一步只做小量级的差，没有相消。LayerNorm 的 CUDA 实现（包括 apex 与 PyTorch 自带的）都用 Welford 或它的分块并行版本。

RMSNorm 是 Llama、DeepSeek、Qwen 一类模型的选择，它只算均方 $$\frac{1}{d}\sum x_i^2$$ 不减均值，没有相消问题，但仍然是一个 $$d = 4096$$ 到 $$8192$$ 项的求和，且后面要接 $$1/\sqrt{\cdot + \epsilon}$$。如果用 BF16 累加，按上一小节的估算，$$\sqrt{4096} \cdot 2^{-8} \approx 0.25$$ 的相对噪声会直接乘进整层输出。所以几乎所有实现都把 BF16 输入上转 FP32 计算均方与倒数平方根，再把归一化后的结果转回 BF16——这是一个访存量不变、计算量微不足道、但对数值至关重要的选择。


## 四、累加精度：为什么 Tensor Core 用 FP32 累加器

### 1. k = 4096 的点积在 BF16 中累加会怎样

一个 GEMM 输出元素是长度 $$k$$ 的点积 $$\sum_{i=1}^{k} a_i b_i$$。Llama-3-8B 的 $$W_Q$$ 输入维度 $$k = d = 4096$$，FFN 的 down 投影 $$k = d_{ff} = 14336$$。假设各项 $$a_i b_i$$ 量级相近为 $$p$$，考虑部分和的增长：

- 各项符号随机时，部分和量级增长到 $$\sqrt{k} \cdot p = 64p$$；
- 各项符号相同时（比如 ReLU/SwiGLU 后的激活乘正权重列），部分和增长到 $$k \cdot p = 4096 p$$。

在后一种情况下，累加到最后时，部分和比新加进来的项大 $$2^{12}$$ 倍。按第三节第 1 小节的分析，新项在对齐后要右移 12 位；BF16 的有效位只有 8 位，右移 12 位后一位都不剩，新项的贡献被完全舍掉。即便是随机符号的情形，$$2^6 = 64$$ 倍的量级差也让每个新项只剩 2 位有效数字。用 $$O(\sqrt{k} u)$$ 估算总相对误差：$$64 \times 2^{-8} = 0.25$$。**一个 BF16 累加的 4096 维点积，结果有 25% 量级的噪声，基本不可用。** 在 FP32 中累加，同样的估算给出 $$64 \times 2^{-24} \approx 4 \times 10^{-6}$$，远低于 BF16 输入本身的表示误差，可以忽略。

这就是为什么"BF16 GEMM"从来不是真正全程 BF16：Tensor Core 的 `mma` 指令接受 BF16/FP16 输入，乘积与累加在 FP32 累加器中进行，最后由 epilogue 把 FP32 结果舍成 BF16 写回。输入精度 8 位、累加精度 24 位，是这个组合让 BF16 GEMM 的输出误差停留在输入表示误差的量级（约 $$2^{-8}$$），而不随 $$k$$ 增长。第七节判断 kernel 数值差异时会用到这个结论。

### 2. FP8 Tensor Core 的累加精度

Hopper 的 FP8 Tensor Core（`wgmma` 指令）名义上也用 FP32 累加器，但 DeepSeek-V3 的技术报告（DeepSeek-AI 2024）指出，在 H800 上 FP8 GEMM 的累加实际上只保留约 14 位有效精度，而不是完整的 24 位。这个差别在 BF16 GEMM 里无关紧要，在 FP8 GEMM 里却很致命：把上面的估算重做一遍，$$k = 4096$$、随机符号，$$64 \times 2^{-14} \approx 0.4\%$$，已经与 BF16 输出的表示误差同量级；如果是同号累加，或者 $$k = 14336$$、甚至 MoE 的 $$k = 7168$$ 再乘专家数，误差就会超过 FP8 输入本身带来的误差，成为主导项。报告的原话是，对 $$k = 4096$$ 的随机数据，这个截断会带来接近 2% 的相对误差。

DeepSeek-V3 的解法是**分段提升（promotion）**：Tensor Core 每累加 $$N_C = 128$$ 个 $$k$$ 方向的元素（对 FP8 `wgmma` 的 $$k = 32$$ 来说就是每 4 条 WGMMA 指令），就把累加器里的部分和搬到 CUDA core 的 FP32 寄存器上加进最终结果，然后清零 Tensor Core 累加器重新开始。这样 14 位精度的累加深度被限制在 128 项以内（$$\sqrt{128} \times 2^{-14} \approx 7 \times 10^{-4}$$），跨段的累加在真正的 FP32 中完成。128 这个数字同时也是它分块量化的块大小（第六节），两者对齐后，每一段的反量化 scale 乘法也可以在搬到 CUDA core 的同时完成。代价是 Tensor Core 与 CUDA core 之间的数据搬运和 CUDA core 的额外 FMA，报告称通过与 WGMMA 的重叠把这部分开销控制在可接受范围。


## 五、混合精度训练

### 1. 三要素

Micikevicius 等 2017 提出的混合精度训练包含三个部件，后来所有 FP16/BF16 训练（Megatron、DeepSpeed、PyTorch AMP）都是这三件事的变体：

1. **FP32 master weights**：权重的权威副本用 FP32 保存，每步更新作用于它；前向反向使用的是它的低精度副本；
2. **低精度前向与反向**：GEMM 输入用 FP16/BF16，累加用 FP32（上一节），激活以低精度保存，梯度以低精度计算；
3. **loss scaling**：只对 FP16 需要，把 loss 乘一个大常数再反向传播，让梯度整体平移到 FP16 的可表示范围内，更新前再除回去。

第 2 点节省的是激活的显存与 GEMM 的时间——这是混合精度的全部收益来源。第 1 点和第 3 点都是为了让第 2 点不破坏收敛而付出的代价。

### 2. 为什么 master weights 不能省：一个具体数字

Adam 的更新量是 $$\Delta w = -\eta \cdot \hat m / (\sqrt{\hat v} + \epsilon)$$。经过偏置修正后 $$\hat m / \sqrt{\hat v}$$ 的量级接近 1（它本质上是梯度的符号加上一点幅度信息），所以 $$\lvert \Delta w\rvert \approx \eta$$。LLM 预训练的峰值学习率典型值是 $$10^{-4}$$ 到 $$3 \times 10^{-4}$$（Llama 3 的 8B 用 $$3 \times 10^{-4}$$，405B 用 $$8 \times 10^{-5}$$），衰减阶段更小；权重本身的量级是初始化标准差 $$\sigma \approx 0.02$$ 到 $$O(1)$$（norm 的 gain）。因此单步更新的**相对**量级 $$\lvert \Delta w / w\rvert$$ 大致在 $$10^{-4}$$ 到 $$10^{-3}$$。

BF16 的单位舍入是 $$u = 2^{-8} \approx 0.0039$$：任何相对变化小于 $$u$$ 的更新，在舍入到最近时都会被舍回原值。取 $$w = 1.0$$、$$\Delta w = 0.001$$：

$$
\text{fl}_{\text{BF16}}(1.0 + 0.001) = 1.0
$$

因为 BF16 在 $$[1, 2)$$ 内相邻可表示数的间距是 $$2^{-7} = 0.0078125$$，1.001 距 1.0 比距 1.0078 近得多，被舍回 1.0。**这一步更新彻底消失，不是变小，是变成零。** 而且会连续消失：只要每步 $$\Delta w$$ 都不到 0.0039，一万步之后 $$w$$ 仍然是 1.0，尽管 FP32 中它早已走到 11.0。对比 FP16：间距是 $$2^{-10} \approx 0.000977$$，$$1.001$$ 会被舍成 $$1.000977$$，虽然有 2.3% 的误差但更新保留了下来。这就是核心问题后半部分的答案：**BF16 用在权重更新上，会让 $$\eta \lesssim 4 \times 10^{-3}$$ 的所有更新在 $$O(1)$$ 量级的权重上归零；对 $$0.02$$ 量级的权重，阈值相应降到 $$8 \times 10^{-5}$$，训练后期的小学习率同样会撞上。** FP32 master weights 的单位舍入是 $$2^{-24} \approx 6 \times 10^{-8}$$，比任何合理的 $$\Delta w / w$$ 小四个数量级以上，更新可以精确积累。

一句话提一下替代方案：**随机舍入（stochastic rounding）**以 $$\Delta w / \text{spacing}$$ 的概率向上舍、否则向下舍，让舍入结果的期望等于真值，从而在多步平均意义上保留小更新（Gupta 等 2015）。它可以让纯 BF16 权重的训练大体收敛，但方差更大，主流框架没有采用，FP32 master 仍是默认。

### 3. FP16 的 loss scaling

FP16 的最小正规数是 $$2^{-14} \approx 6.1 \times 10^{-5}$$，最小次正规数 $$2^{-24} \approx 6 \times 10^{-8}$$。第二节提到的梯度直方图里，大量激活梯度落在 $$2^{-24}$$ 以下——它们在 FP16 中变成 0，反向传播经过这些位置的信息直接断掉。另一边，FP16 的上界 65504 又离前向激活的典型量级不远。梯度的分布与 FP16 的窗口错位：梯度太小，窗口太高。

loss scaling 的做法是把 loss 乘 $$S$$（如 $$2^{16}$$）再反向，链式法则让所有梯度线性放大 $$S$$ 倍，整体平移进窗口；在更新权重之前把梯度除以 $$S$$（在 FP32 中进行，因为 master weights 是 FP32）。$$S$$ 太小则下溢没有解决，太大则梯度溢出成 inf。**动态 loss scaling** 自动调节：每步检查梯度里有没有 inf/NaN，有则跳过本步更新并把 $$S$$ 减半；连续 $$N$$ 步（PyTorch `GradScaler` 默认 2000）没有溢出则把 $$S$$ 加倍。它让 $$S$$ 始终逼近"刚好不溢出"的最大值，把 FP16 的 30 个二进制数量级尽量对准梯度的分布。代价是每步一次全局的 inf 检查（一次 all-reduce 级别的同步）和偶尔的跳步。

### 4. BF16 为什么不需要

BF16 的最小正规数是 $$2^{-126}$$，任何 FP32 里不为零的梯度在 BF16 里都不为零；最大值 $$3.4 \times 10^{38}$$，任何 FP32 里不溢出的激活在 BF16 里都不溢出。梯度的分布无论怎样漂移都在窗口内，没有需要平移的理由，loss scaling 整套机制（缩放、检查、跳步、同步）全部省掉。这也是 A100 之后 BF16 迅速取代 FP16 的原因：它不是精度更好，而是少了一个需要调参、会引入跳步和同步的子系统。

### 5. 训练状态的字节数

混合精度 + Adam 下每个参数需要保存的东西，逐项列出：

```text
BF16 权重副本（前向反向用）    2 B
BF16 梯度                     2 B
FP32 master weights           4 B
FP32 Adam 一阶矩 m            4 B
FP32 Adam 二阶矩 v            4 B
---------------------------------
合计                         16 B / 参数
```

注意纯 FP32 训练（权重 4 + 梯度 4 + m 4 + v 4）也是 16 字节：混合精度**没有**减少训练状态的显存，它减少的是激活显存和 GEMM 时间。乘上三个模型的参数量：

$$
\begin{aligned}
\text{Llama-3-8B:}\quad & 8.03 \times 10^9 \times 16 = 128.5\ \text{GB} \\
\text{Llama-3-70B:}\quad & 70.55 \times 10^9 \times 16 = 1128.8\ \text{GB} \approx 1.13\ \text{TB} \\
\text{DeepSeek-V3:}\quad & 671 \times 10^9 \times 16 = 10736\ \text{GB} \approx 10.7\ \text{TB}
\end{aligned}
$$

这三个数字还没有算任何激活。一张 H100 是 80 GB：8B 模型的训练状态就已经超过一张卡，70B 至少需要 15 张卡只放状态，671B 需要 135 张。这就是 ZeRO（Rajbhandari 等 2020）与 FSDP 存在的理由——把这 16 字节中的优化器状态（8 字节）、梯度（2 字节）、权重（2+4 字节）分片到数据并行组的各个 rank 上。怎么切、切完通信量多少，不在本篇范围；本篇只负责算出"不切就放不下"。

顺带算一个变体：DeepSeek-V3 报告中用 FP8 保存 GEMM 输入用的权重副本（1 B）、BF16 保存梯度（2 B）、FP32 保存 master weights（4 B），并把 Adam 的 $$m$$、$$v$$ 改为 BF16（各 2 B），合计约 11 字节/参数，671B 约 7.4 TB。这种"优化器状态降精度"之所以可行，是因为 $$m$$、$$v$$ 只是梯度的滑动平均，对它们做 $$2^{-8}$$ 的相对舍入不会像对权重那样累积——它们每一步都被新梯度以 $$1 - \beta$$ 的权重覆盖一部分。


## 六、FP8 训练

### 1. E4M3 与 E5M2 的分工

FP8 有两个变体，恰好对应训练里两类性质不同的张量：

- **权重与激活**：分布相对集中（经过 norm 之后激活的 RMS 约为 1，权重约为 $$\sigma$$），动态范围不大，但直接决定前向输出的质量。用精度型的 **E4M3**：4 位有效数字（含隐含位），范围 $$[2^{-9}, 448]$$，配合 per-tensor 或分块 scale 把张量的最大值对齐到 448 附近。
- **梯度**：量级跨度大（第二节），而对精度的要求更低——梯度本身就是噪声估计。用范围型的 **E5M2**：范围与 FP16 相同 $$[2^{-16}, 57344]$$，只有 3 位有效数字。

前向 GEMM 的两个输入都是 E4M3；反向计算权重梯度 $$\nabla W = X^\top \nabla Y$$ 与激活梯度 $$\nabla X = \nabla Y W^\top$$ 时，一个输入是 E4M3 的激活或权重，另一个是 E5M2 的输出梯度。Hopper 的 FP8 Tensor Core 支持两种格式的任意组合作为两个输入。

### 2. per-tensor scaling 与 delayed scaling

E4M3 的范围只有 5 个十进制数量级，任何真实张量都要先乘一个 scale 才能放进去。最简单的是 **per-tensor scaling**：整个张量共享一个 $$s = 448 / \text{amax}(x)$$，量化 $$\tilde x = \text{fl}_{\text{E4M3}}(s \cdot x)$$，GEMM 之后再把两个输入的 scale 乘回去。问题是 amax 需要先扫一遍张量才能得到，这意味着每个 GEMM 之前多一次全张量的读取，而 FP8 训练的收益恰恰在于省访存。

Transformer Engine 使用 **delayed scaling**：不用当前张量的 amax，而是用这个张量在过去若干步（默认记录最近 1024 步的历史）的 amax 的最大值或最近值来确定本步的 scale，本步的 amax 在 GEMM 的 epilogue 里顺便算出来，更新历史供下一步用。这样量化与 GEMM 可以融合在一个 kernel 里，代价是当激活分布突变时，scale 落后一步，可能溢出（E4M3 无 inf，溢出饱和成 448 或变 NaN）。

### 3. DeepSeek-V3 的分块量化：scale 的粒度决定离群值的影响范围

per-tensor scaling 的根本问题是**离群值（outlier）**。LLM 的激活中存在少数通道的值比其余大两三个数量级（第七篇会再讨论），如果整个张量共享一个 scale，这个 scale 被离群值决定：离群值被对齐到 448，占据 E4M3 窗口的顶端。E4M3 从最小次正规数 $$2^{-9}$$ 到 448 只有 18 个二进制数量级，其中正规区 15 个；任何比离群值小 $$2^{15} \approx 3 \times 10^4$$ 倍以上的元素就落进次正规区开始丢有效位，小 $$2^{18}$$ 倍以上直接变 0。一个 $$100 \times$$（约 $$2^7$$）的离群值加上激活本身三四个十进制数量级的自然分布，尾部恰好被推进这个区域；更糟的是 delayed scaling 用历史 amax，离群值让 amax 剧烈波动，scale 在"太大溢出"与"太小下溢"之间摇摆。

DeepSeek-V3 的做法是缩小 scale 的作用范围：**激活按 $$1 \times 128$$ 分块**（每个 token 每 128 个通道一个 scale），**权重按 $$128 \times 128$$ 分块**。一个离群值现在只能拖累同一个块里的 127 个邻居，其余所有块的 scale 由各自的正常值决定，不受影响。用数字说：DeepSeek-V3 的 $$d = 7168$$ 激活向量有 56 个块，一个离群通道影响 $$1/56 \approx 1.8\%$$ 的元素；per-tensor 时影响 100%。块大小 128 与第四节的累加提升周期 $$N_C = 128$$ 对齐，每 128 个 $$k$$ 元素的部分和搬到 CUDA core 时正好乘上这一块的 $$s_a \cdot s_w$$，反量化没有额外的遍历。

这也解释了为什么分块量化必须配合修改 GEMM kernel：标准 FP8 GEMM 假设 per-tensor scale，在 epilogue 乘一次；分块 scale 在 $$k$$ 方向变化，必须在累加中途乘，这正是 promotion 到 CUDA core 那一步顺便做的事。scale 的粒度、累加的分段、kernel 的结构，三者是同一个设计决策。


## 七、推理中的数值

### 1. attention logit 的增长与 QK-norm

attention logit 是 $$q_i \cdot k_j / \sqrt{d_{head}}$$。训练过程中 $$W_Q$$、$$W_K$$ 的范数会持续增长，logit 的量级随之上升；Dehghani 等 2023 在训练 22B 参数的 ViT 时观察到 logit 增长到几百导致 attention 变成近乎 one-hot、训练发散，Wortsman 等 2023 在语言模型上系统研究了同一现象（称为 attention logit growth）。对 FP16 推理来说这是直接的溢出风险（阈值 11.09）；对 BF16 来说范围没问题，但 softmax 之后 $$e^{-100}$$ 量级的概率在 BF16 中不是问题、在 E4M3 中就是 0。

**QK-norm** 的做法是在计算 logit 之前对 $$q$$ 与 $$k$$ 各做一次 RMSNorm（每个头独立，通常带可学习 gain），于是 $$\Vert q\Vert  = \Vert k\Vert  = \sqrt{d_{head}} \cdot g$$，logit 的上界变成

$$
\frac{|q \cdot k|}{\sqrt{d_{head}}} \le \frac{\|q\| \|k\|}{\sqrt{d_{head}}} = \sqrt{d_{head}} \cdot g_q g_k
$$

$$d_{head} = 128$$ 时上界是 $$11.3 \cdot g_q g_k$$——与 FP16 的 11.09 阈值几乎重合，这是个巧合，但说明了 QK-norm 把 logit 限制在了一个"任何格式都安全"的量级。Gemma 3、Qwen3 一类模型采用了 QK-norm；对推理引擎来说它意味着 attention kernel 前多两个逐头的 norm（访存量与 RoPE 同级），以及 KV cache 中存的是 norm 之后的 $$k$$。

### 2. RMSNorm 的 ε

RMSNorm 的输出是 $$x / \sqrt{\frac{1}{d}\sum x_i^2 + \epsilon} \cdot \gamma$$。Llama 3 用 $$\epsilon = 10^{-5}$$，DeepSeek-V3 与 Qwen2.5 用 $$10^{-6}$$。它的意义分两种情形：

- 正常输入：均方在 $$O(1)$$，$$\epsilon$$ 的相对贡献是 $$10^{-5}$$ 或 $$10^{-6}$$，低于 BF16 甚至 FP16 的单位舍入，输出上完全看不出区别；
- 近零输入：均方趋近 0 时，输出被 $$1/\sqrt{\epsilon}$$ 限制，$$\epsilon = 10^{-5}$$ 允许最大放大 316 倍，$$10^{-6}$$ 允许 1000 倍。$$\epsilon$$ 决定了一行接近零的激活最多被放大多少——这是它对数值稳定性的全部作用。

一个实现层面的推论：$$\epsilon$$ 必须在 FP32 中加。$$\text{fl}_{\text{BF16}}(1 + 10^{-5}) = 1$$，在 BF16 里加 $$\epsilon$$ 等于没加，只在均方本身接近 0 时才有效果（那时 $$\epsilon$$ 与均方同量级，加法不丢位）。另一个推论：如果一个推理 kernel 硬编码了 $$10^{-6}$$ 而模型 checkpoint 是 $$10^{-5}$$，输出会有一个**系统性**（所有 token 同方向）的微小偏差——这类差异不是浮点噪声，是 bug，下一小节讨论如何区分。

### 3. 两个 kernel 的数值差异应该有多大

同一个 GEMM 用 cuBLAS、CUTLASS、Triton 或不同的 tile 配置算出来，逐位结果通常不同。判断差异是否正常的方法是：**用 FP32（最好 FP64）算一个参考值，分别度量两个实现相对参考值的误差，再和格式决定的理论量级比较。**

对 BF16 输入、FP32 累加、BF16 输出的 GEMM，误差来自三处：

- 输入 $$a$$、$$b$$ 各自舍入到 BF16：每个乘积 $$a_i b_i$$ 带约 $$2u = 2^{-7}$$ 的相对误差，各项独立，求和后绝对误差约 $$2u \cdot \sqrt{\sum (a_i b_i)^2} \approx 2u \sqrt{k}\, p$$；
- FP32 累加：$$\sqrt{k} \cdot 2^{-24}$$，可忽略；
- 输出舍到 BF16：$$u$$。

相对于输出本身的量级（随机符号时约 $$\sqrt{k}\, p$$），第一项给出约 $$2u \approx 0.8\%$$ 的相对误差，与 $$k$$ 基本无关；如果各项同号，输出量级是 $$k p$$，相对误差反而随 $$k$$ 缩小到 $$2u / \sqrt{k}$$。所以**BF16 GEMM 相对 FP32 参考的逐元素相对误差在 $$10^{-3}$$ 到 $$10^{-2}$$ 之间是正常的**；用 Frobenius 范数度量的整体相对误差 $$\Vert C - C_{\text{ref}}\Vert _F / \Vert C_{\text{ref}}\Vert _F$$ 通常在 $$10^{-3}$$ 量级。如果某个 kernel 在低精度中累加（没有 FP32 累加器，或 split-K 的部分和以 BF16 交换），误差会变成 $$u \sqrt{k}$$ 随 $$k$$ 增长：$$k = 4096$$ 时 25%，一眼可辨。

判断规则因此可以写成三条：

1. 两个实现相对参考值的误差都在 $$\varepsilon_{\text{fmt}} \cdot O(1)$$ 到 $$\varepsilon_{\text{fmt}} \cdot \sqrt{k}$$ 之间：正常噪声，逐位不同是预期行为；
2. 误差比理论量级大几个数量级：某处在低精度累加、丢了 scale、或数据类型转换出错；
3. 误差不大但**有系统性符号**（所有元素同方向偏、或与输入某个特征相关）：不是舍入，是算法差异——$$\epsilon$$ 不同、RoPE 频率计算精度不同、softmax 的 scale 位置不同一类。舍入噪声是零均值的，系统性偏差不是。

`torch.testing.assert_close` 的默认容差正是按这个思路为各 dtype 设定的：

```text
dtype       rtol        atol
float64     1e-7        1e-7
float32     1.3e-6      1e-5
float16     1e-3        1e-5
bfloat16    1.6e-2      1e-5
```

`rtol` 大致是格式 $$\varepsilon$$ 的一个小倍数（BF16 的 $$1.6 \times 10^{-2} \approx 2\varepsilon$$，允许输入和输出各舍一次），`atol` 处理靠近 0 的元素（此时相对误差没有意义）。对比一个自定义 kernel 与参考实现时，默认容差是合理的起点；需要放宽时，放宽的倍数应该能用上面的 $$\sqrt{k}$$ 或 $$\log k$$ 解释，解释不了就应该怀疑 kernel。

### 4. 随机性与可复现

浮点加法不满足结合律：$$(a + b) + c \ne a + (b + c)$$。任何让加法顺序不确定的机制都会让同一输入产生不同输出：

- **atomicAdd**：多个线程把值加到同一地址，先后顺序由硬件调度决定。PyTorch 里 `index_add_`、`scatter_add_`、`index_put_(accumulate=True)`、embedding 的 backward（多个 token 映射到同一词表行时梯度用 atomicAdd 累加）、`bincount` 一类都默认使用它；
- **cuBLAS split-K**：当 $$m$$、$$n$$ 小而 $$k$$ 大时（decode 阶段的 GEMM 正是这个形状），cuBLAS 把 $$k$$ 切成若干段并行计算再归约，归约顺序可能依赖 workspace 的分配情况；
- 多 GPU 的 all-reduce 的归约树、以及不同 batch size 下选中不同的 kernel。

`torch.use_deterministic_algorithms(True)` 强制所有算子使用确定性实现：有确定性版本的算子（如 embedding backward 换成先排序再分段归约）切换过去，没有的直接抛异常。cuBLAS 需要额外设置环境变量 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 或 `:16:8`，让它对每个 stream 使用固定大小的 workspace，从而每次选中相同的算法与归约顺序。代价是明显的：排序版 scatter 比 atomicAdd 慢数倍，固定 workspace 排除了某些更快的 split-K 配置。对训练来说，确定性通常只在调试时开启；对推理来说，同一请求在不同 batch 里落到不同 kernel 上本来就会产生 $$10^{-3}$$ 量级的差异，这是第 3 小节讨论的正常噪声，采样温度大于 0 时它完全被采样随机性掩盖，贪心解码时它偶尔会翻转一个 top-1/top-2 概率接近的 token——这是"同一 prompt 两次 greedy 结果不同"的通常原因。


## 八、实践

### 1. 逐位构造各格式并验证 max / min / ε

下面的代码直接用位模式构造每种格式的边界值，把第一节的表跑出来。FP32/FP16 用 NumPy，BF16/FP8 用 PyTorch 的位级 `view`。

```python
import struct
import numpy as np
import torch

def fp32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]

def fp16_from_bits(bits: int) -> float:
    return float(np.frombuffer(struct.pack("<H", bits), dtype=np.float16)[0])

def torch_from_bits(bits: int, storage, dtype) -> torch.Tensor:
    return torch.tensor([bits], dtype=storage).view(dtype)

# FP32: 1/8/23
print("FP32 max        ", fp32_from_bits(0x7F7FFFFF))   # 3.4028235e38
print("FP32 min normal ", fp32_from_bits(0x00800000))   # 1.1754944e-38
print("FP32 min subnorm", fp32_from_bits(0x00000001))   # 1.4e-45

# FP16: 1/5/10
print("FP16 max        ", fp16_from_bits(0x7BFF))       # 65504.0
print("FP16 min normal ", fp16_from_bits(0x0400))       # 6.104e-05
print("FP16 min subnorm", fp16_from_bits(0x0001))       # 5.96e-08
print("FP16 inf        ", fp16_from_bits(0x7C00))       # inf

# BF16: 1/8/7  (int16 位模式, 0x7F7F 为正数, 可直接用)
print("BF16 max        ", torch_from_bits(0x7F7F, torch.int16, torch.bfloat16).item())  # 3.3895e38
print("BF16 min normal ", torch_from_bits(0x0080, torch.int16, torch.bfloat16).item())  # 1.1755e-38
print("BF16 min subnorm", torch_from_bits(0x0001, torch.int16, torch.bfloat16).item())  # 9.18e-41

# FP8 E4M3 (fn): 1/4/3, 无 inf, 0x7F 为 NaN
e4m3 = torch.float8_e4m3fn
print("E4M3 max        ", torch_from_bits(0x7E, torch.uint8, e4m3).float().item())  # 448.0
print("E4M3 0x7F       ", torch_from_bits(0x7F, torch.uint8, e4m3).float().item())  # nan
print("E4M3 min normal ", torch_from_bits(0x08, torch.uint8, e4m3).float().item())  # 0.015625
print("E4M3 min subnorm", torch_from_bits(0x01, torch.uint8, e4m3).float().item())  # 0.001953125

# FP8 E5M2: 1/5/2, 有 inf
e5m2 = torch.float8_e5m2
print("E5M2 max        ", torch_from_bits(0x7B, torch.uint8, e5m2).float().item())  # 57344.0
print("E5M2 0x7C       ", torch_from_bits(0x7C, torch.uint8, e5m2).float().item())  # inf
print("E5M2 min normal ", torch_from_bits(0x04, torch.uint8, e5m2).float().item())  # 6.104e-05
print("E5M2 min subnorm", torch_from_bits(0x01, torch.uint8, e5m2).float().item())  # 1.526e-05

# 机器精度: torch.finfo 与 2^-M 对照
for dt, M in [(torch.float32, 23), (torch.float16, 10), (torch.bfloat16, 7), (e4m3, 3), (e5m2, 2)]:
    print(f"{str(dt):22s} eps={torch.finfo(dt).eps:.3e}  2^-M={2.0**-M:.3e}  "
          f"max={torch.finfo(dt).max:.3e}  tiny={torch.finfo(dt).tiny:.3e}")

# TF32 只是输入截断: 张量仍是 4 字节
x = torch.randn(1024, 1024)
print("float32 element_size:", x.element_size())  # 4, 打开 allow_tf32 后也不变
```

`torch.finfo(torch.float8_e4m3fn).max` 会给出 448，`torch.finfo(torch.float8_e5m2).max` 给出 57344，与逐位构造的结果一致。

### 2. 模拟 BF16 权重更新被吃掉

```python
import torch

w32 = torch.tensor(1.0, dtype=torch.float32)
w16 = torch.tensor(1.0, dtype=torch.bfloat16)
wh  = torch.tensor(1.0, dtype=torch.float16)
lr_step = 1e-3   # 单步 |Δw|, 相对 w=1.0 为 1e-3, 低于 BF16 的 u=2^-8

for step in range(1000):
    w32 = w32 + lr_step
    w16 = w16 + torch.tensor(lr_step, dtype=torch.bfloat16)
    wh  = wh  + torch.tensor(lr_step, dtype=torch.float16)

print("FP32 :", w32.item())   # 约 2.0 (1.0 + 1000 × 0.001, 含微小舍入)
print("FP16 :", wh.item())    # 约 1.98, 每步被舍成 0.000977, 有损但更新保留
print("BF16 :", w16.item())   # 1.0, 一千步更新全部被吃掉

# BF16 在 1.0 附近的间距
one = torch.tensor(1.0, dtype=torch.bfloat16)
print("BF16 spacing at 1.0:", (torch.tensor(1.0 + 2**-7, dtype=torch.bfloat16) - one).item())  # 0.0078125
print("fl_bf16(1.0 + 0.001) =", torch.tensor(1.001, dtype=torch.bfloat16).item())              # 1.0
print("fl_bf16(1.0 + 0.004) =", torch.tensor(1.004, dtype=torch.bfloat16).item())              # 1.0078125 (超过半间距, 进位)

# 正确做法: FP32 master + BF16 副本
master = torch.tensor(1.0, dtype=torch.float32)
for step in range(1000):
    master = master + lr_step
    w_bf16_copy = master.to(torch.bfloat16)   # 前向用这份
print("master:", master.item(), " bf16 copy:", w_bf16_copy.item())  # 2.0, 2.0
```

预期输出：FP32 走到约 2.0，FP16 走到约 1.98（每步舍成 $$2^{-10}$$ 的整数倍），BF16 停在 1.0。这就是第五节的数字例子在代码里的样子。

### 3. GEMM 误差随 k 的增长

下面的代码在 CPU 上可跑：对同一对随机矩阵，用 FP64 做参考，比较 (a) FP32 GEMM、(b) BF16 输入 + FP32 累加（PyTorch 的 CPU BF16 matmul 内部以 FP32 累加）、(c) 手工模拟的 BF16 输入 + BF16 累加，度量 Frobenius 相对误差随 $$k$$ 的变化。FP8 版本需要 Hopper 及以上的 GPU 与 `torch._scaled_mm`，附在最后，按需启用。

```python
import torch

torch.manual_seed(0)

def rel_err(c, ref):
    return ((c.double() - ref).norm() / ref.norm()).item()

def bf16_accumulate_matmul(a16, b16):
    """模拟没有 FP32 累加器的 BF16 GEMM: 每加一项都舍回 BF16 (慢, 仅演示)."""
    m, k = a16.shape
    n = b16.shape[1]
    acc = torch.zeros(m, n, dtype=torch.bfloat16)
    for i in range(k):
        acc = acc + (a16[:, i:i+1] * b16[i:i+1, :])   # 逐项加, 结果留在 BF16
    return acc

m = n = 64
print(f"{'k':>6} {'fp32':>10} {'bf16/fp32acc':>14} {'bf16/bf16acc':>14} {'u*sqrt(k)':>10}")
for k in [64, 256, 1024, 4096, 16384]:
    a = torch.randn(m, k, dtype=torch.float64)
    b = torch.randn(k, n, dtype=torch.float64)
    ref = a @ b
    e32 = rel_err(a.float() @ b.float(), ref)
    e16 = rel_err(a.bfloat16() @ b.bfloat16(), ref)
    eacc = rel_err(bf16_accumulate_matmul(a.bfloat16(), b.bfloat16()), ref) if k <= 4096 else float("nan")
    print(f"{k:>6} {e32:>10.2e} {e16:>14.2e} {eacc:>14.2e} {2**-8 * k**0.5:>10.2e}")

# FP8 (需要 Hopper / Ada 及以上, torch >= 2.1 且 CUDA 可用)
if torch.cuda.is_available() and hasattr(torch, "_scaled_mm"):
    dev = "cuda"
    for k in [256, 1024, 4096]:
        a = torch.randn(m, k, dtype=torch.float64, device=dev)
        b = torch.randn(k, n, dtype=torch.float64, device=dev)
        ref = a @ b
        sa = 448.0 / a.abs().max()
        sb = 448.0 / b.abs().max()
        a8 = (a * sa).to(torch.float8_e4m3fn)
        b8 = (b * sb).to(torch.float8_e4m3fn)
        # _scaled_mm 要求 b 为列主序 (k×n 转置后 contiguous), scale 为 FP32 标量张量
        c = torch._scaled_mm(a8, b8.t().contiguous().t(),
                             scale_a=(1 / sa).float().reshape(()),
                             scale_b=(1 / sb).float().reshape(()),
                             out_dtype=torch.float32)
        print(f"FP8 E4M3 per-tensor, k={k}: rel_err={rel_err(c, ref):.2e}")
```

预期趋势（理论量级，不是实测）：

- FP32 列：约 $$10^{-7}$$，几乎不随 $$k$$ 变化（如果在 GPU 上跑且 TF32 开启，会跳到 $$10^{-3}$$ 量级——这本身就是一个值得亲手验证的现象）；
- BF16 输入 + FP32 累加列：约 $$3 \times 10^{-3}$$ 到 $$6 \times 10^{-3}$$，**不随 $$k$$ 增长**，由输入舍入决定；
- BF16 累加列：随 $$k$$ 明显增长，$$k = 64$$ 时与上一列接近，$$k = 4096$$ 时上升到 $$10^{-1}$$ 量级，与 $$u\sqrt{k}$$ 列同步增长；
- FP8 E4M3 列：约 $$3 \times 10^{-2}$$ 到 $$6 \times 10^{-2}$$（$$\varepsilon = 0.125$$，$$u = 0.0625$$），对高斯随机数 per-tensor scale 已经足够，对带离群值的真实激活会明显变差——可以把 `a` 的某一列乘 100 再跑一次看变化。

拿到一个新 kernel 的误差数字之后，把它放到这张表里对应的列上，就知道它是正常噪声还是 bug。

### 4. llm_cost.py：dtype 字节表与训练状态

本篇给贯穿脚本加两样东西：`DTYPE_BYTES` 表和 `training_state_bytes()`。为保持可独立运行，这里附上第一篇 `param_count()` 的 dense 版本；第五篇的 MoE 版本对 dense 模型给出相同结果，DeepSeek-V3 用 `param_override` 直接填入 671B。

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
    param_override: int = 0      # MoE/MLA 模型直接给总参数量 (第五篇的 param_count 可算出)

LLAMA3_8B  = ModelConfig("Llama-3-8B",  4096, 32, 32, 8, 128, 14336, 128256)
LLAMA3_70B = ModelConfig("Llama-3-70B", 8192, 80, 64, 8, 128, 28672, 128256)
DEEPSEEK_V3 = ModelConfig("DeepSeek-V3", 7168, 61, 128, 128, 192, 18432, 129280,
                          param_override=671_000_000_000)

def param_count(cfg: ModelConfig) -> dict:
    """第一篇的 dense 参数量（重给以便独立运行）；MoE 模型用 param_override 直接给总量。"""
    d, dh = cfg.hidden, cfg.head_dim
    attn = d * cfg.n_heads * dh * 2 + d * cfg.n_kv_heads * dh * 2     # W_Q, W_O, W_K, W_V
    ffn = 3 * d * cfg.d_ff
    per_layer = attn + ffn + 2 * d
    embed = cfg.vocab * d
    lm_head = 0 if cfg.tie_embeddings else cfg.vocab * d
    total = cfg.param_override or (per_layer * cfg.layers + embed + lm_head + d)
    return {"per_layer": per_layer, "embedding": embed, "lm_head": lm_head, "total": total}

# ---- 第六篇新增 ----
DTYPE_BYTES = {
    "fp32": 4, "tf32": 4,          # TF32 在内存中仍是 32 位
    "fp16": 2, "bf16": 2,
    "fp8_e4m3": 1, "fp8_e5m2": 1,
    "int8": 1, "int4": 0.5,        # int4 不含 scale/zero-point 开销 (第七篇)
}

def training_state_bytes(cfg: ModelConfig, optimizer: str = "adam", mixed: bool = True,
                         weight_dtype: str = "bf16", grad_dtype: str = "bf16",
                         optim_dtype: str = "fp32") -> dict:
    """每参数训练状态字节数与总量. mixed=True: 低精度权重副本 + 低精度梯度 + FP32 master;
    mixed=False: 全 FP32 (权重即 master, 无副本)."""
    n = param_count(cfg)["total"]
    if mixed:
        per = {"weight_copy": DTYPE_BYTES[weight_dtype],
               "grad": DTYPE_BYTES[grad_dtype],
               "master": DTYPE_BYTES["fp32"]}
    else:
        per = {"weight": DTYPE_BYTES["fp32"], "grad": DTYPE_BYTES["fp32"]}
    n_moments = {"sgd": 0, "momentum": 1, "adam": 2, "adamw": 2}[optimizer]
    for i in range(n_moments):
        per[f"moment{i+1}"] = DTYPE_BYTES[optim_dtype]
    per_param = sum(per.values())
    return {"params": n, "per_param_bytes": per_param, "breakdown": per,
            "total_bytes": n * per_param}

if __name__ == "__main__":
    for cfg in (LLAMA3_8B, LLAMA3_70B, DEEPSEEK_V3):
        r = training_state_bytes(cfg)
        print(f"{cfg.name:12s} params={r['params']/1e9:7.2f}B  "
              f"{r['per_param_bytes']:>2} B/param  state={r['total_bytes']/1e9:9.1f} GB  "
              f"H100(80GB) >= {r['total_bytes']/80e9:6.1f} 张")
    # DeepSeek-V3 报告的变体: FP8 权重副本, BF16 梯度, FP32 master, BF16 m/v
    r = training_state_bytes(DEEPSEEK_V3, weight_dtype="fp8_e4m3", optim_dtype="bf16")
    print(f"{'V3-fp8-recipe':12s} {r['per_param_bytes']} B/param  state={r['total_bytes']/1e12:.2f} TB  {r['breakdown']}")
    # 全 FP32 对照: 同样 16 B/param
    r = training_state_bytes(LLAMA3_8B, mixed=False)
    print(f"{'8B-fp32':12s} {r['per_param_bytes']} B/param  state={r['total_bytes']/1e9:.1f} GB")
```

输出示例：

```text
Llama-3-8B   params=   8.03B  16 B/param  state=    128.5 GB  H100(80GB) >=    1.6 张
Llama-3-70B  params=  70.55B  16 B/param  state=   1128.9 GB  H100(80GB) >=   14.1 张
DeepSeek-V3  params= 671.00B  16 B/param  state=  10736.0 GB  H100(80GB) >=  134.2 张
V3-fp8-recipe 11 B/param  state=7.38 TB  {'weight_copy': 1, 'grad': 2, 'master': 4, 'moment1': 2, 'moment2': 2}
8B-fp32      16 B/param  state=128.5 GB
```

"张数"是把训练状态平均分摊后每张 H100 的 80 GB 至少要放多少，未计激活与通信 buffer，是纯下界。


## 九、小结

这一篇把前五篇默认的"2 字节"打开来看，主线是一条：**指数位决定范围、尾数位决定精度，深度学习的前向反向需要范围而不需要精度，权重更新需要精度而不需要范围**。由此推出的每一个结论：

- BF16 用 3 位尾数换 3 位指数，范围与 FP32 相同，前向反向不需要 loss scaling，代价是相对精度 $$2^{-8}$$；这是它取代 FP16 的原因；
- 权重更新 $$\Delta w / w \sim 10^{-4}$$ 到 $$10^{-3}$$ 低于 $$2^{-8}$$，在 BF16 中做更新会被舍回原值（$$1.0 + 0.001 \to 1.0$$），所以需要 FP32 master weights；
- 点积累加必须在比输入高得多的精度里进行：BF16 累加 $$k = 4096$$ 有 25% 噪声，FP32 累加可忽略；FP8 Tensor Core 约 14 位的累加精度迫使 DeepSeek-V3 每 128 项提升到 CUDA core 的 FP32；
- 数值丢失有四种典型位置：大数吃小数（残差流）、长求和（归约用树形）、指数溢出（softmax 减最大值、online softmax 的重缩放）、相消（方差用 Welford、RMSNorm 用 FP32 算均方）；
- FP8 训练用 E4M3 存权重/激活、E5M2 存梯度，scale 的粒度决定一个离群值能拖累多少邻居，分块 128 与累加提升周期 128 是同一个设计；
- 两个 kernel 的差异在 $$\varepsilon$$ 到 $$\varepsilon\sqrt{k}$$ 之间是噪声，大几个数量级或有系统性符号是 bug；非确定性来自 atomicAdd 与 split-K，确定性模式有明确的性能代价。

本篇的数字汇总：

```text
格式         位布局      最大值      最小正规     ε=2^-M     单位舍入 u     备注
FP32         1/8/23     3.40e38     1.18e-38     1.19e-7    6.0e-8
TF32         1/8/10     3.40e38     1.18e-38     9.77e-4    4.9e-4         仅 Tensor Core 输入, 内存仍 4 B
FP16         1/5/10     65504       6.10e-5      9.77e-4    4.9e-4         e^x 溢出于 x > 11.09
BF16         1/8/7      3.39e38     1.18e-38     7.81e-3    3.9e-3         fl(1.0 + 0.001) = 1.0
E4M3         1/4/3      448         1.56e-2      0.125      0.0625         无 inf, 一个 NaN 编码
E5M2         1/5/2      57344       6.10e-5      0.25       0.125          有 inf/NaN
INT8 / INT4  整数       127 / 7     步长 1       —          —              需外部 scale

                              Llama-3-8B     Llama-3-70B    DeepSeek-V3
参数量                        8.03B          70.55B         671B
BF16 权重                     16.06 GB       141 GB         1342 GB (FP8: 671 GB)
训练状态 (16 B/参数)          128.5 GB       1128.8 GB      10736 GB
  = BF16 权重副本 2           16.06 GB       141.1 GB       1342 GB
  + BF16 梯度 2               16.06 GB       141.1 GB       1342 GB
  + FP32 master 4             32.12 GB       282.2 GB       2684 GB
  + FP32 Adam m 4             32.12 GB       282.2 GB       2684 GB
  + FP32 Adam v 4             32.12 GB       282.2 GB       2684 GB
仅放状态需 H100 (80 GB)       >= 2 张        >= 15 张       >= 135 张
V3 报告配方 (11 B/参数)       —              —              7.38 TB
BF16 累加 k=4096 相对噪声     ~2^-8 · sqrt(4096) = 0.25 (任何模型, 不可用)
FP32 累加 k=4096 相对噪声     ~2^-24 · 64 = 4e-6 (可忽略)
```

下一篇进入本系列最后一站：把权重换成 INT4 之后字节数怎么算、投机解码如何用一个小模型改变 decode 的算术强度、LoRA 的额外参数与 FLOPs 各占多少——三种"改变计算形态"的方法。


## 下一篇

[量化、投机解码与 LoRA](/quantization-speculative-decoding-and-lora.html)
