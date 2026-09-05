---
layout: post
title: "Transformer 与 LLM（02）：前向的算量与访存量"
subtitle: "FLOPs, Bytes and Roofline: Prefill versus Decode"
tags: [Transformer, LLM, AI, AI-Infra]
catalog: true
---

> 本文是[《Transformer 与 LLM：结构、算量与数值》](/transformer-and-llm-for-infra-engineers.html)系列的第 2 篇（共七篇）。上一篇：[Transformer 解剖与参数量](/transformer-anatomy-and-parameter-count.html)　下一篇：[Attention 变体与 KV cache](/attention-variants-and-kv-cache.html)

上一篇把一个 decoder-only Transformer 拆到了能数出每一个参数的粒度。结论可以压缩成一个公式：

$$
N \approx L \cdot \left[ d \cdot (d + 2 d_{kv} + d) + 3 \cdot d \cdot d_{ff} \right] + 2 \cdot V \cdot d
$$

其中 $$d_{kv} = n_{kv} \cdot d_{head}$$。代入 Llama-3-8B（$$d = 4096$$，$$L = 32$$，$$n_{kv} = 8$$，$$d_{head} = 128$$，$$d_{ff} = 14336$$，$$V = 128256$$）：每层 attention 41.94M、FFN 176.16M，32 层共 6.98B，embedding 与 lm_head 各 525.3M，合计 8.03B。其中真正参与矩阵乘法的是 32 层的 6.98B 加 lm_head 的 0.53B，约 **7.5B**；embedding 那 0.53B 只做查表。

这一篇把这张参数表变成**成本表**。要回答的是三个问题：

- 跑一次前向，要做多少次浮点运算（FLOPs）？
- 要从 HBM 读多少字节？
- 这两个数的比值，如何决定一段计算是 compute-bound 还是 memory-bound？

以及总纲提出的核心问题：

> **Llama-3-8B 在一张 H100 上，batch 多大时 decode 从 memory-bound 变成 compute-bound？考虑 KV cache 之后，这个 batch 还能达到吗？**

全文所有数字都是**理论下界或估算**，用于建立数量级判断，不是任何实现的实测值。硬件基线取 H100 SXM（80 GB HBM3，3.35 TB/s，BF16 dense 989 TFLOPS）与 A100 80GB（2.0 TB/s，BF16 312 TFLOPS）的公开标称值。


## 一、算量：FLOPs 从哪里来

### 1. 一个矩阵乘法的 FLOPs：2mkn

一切从矩阵乘法开始。把 $$[m, k]$$ 的矩阵乘以 $$[k, n]$$ 的矩阵，输出 $$[m, n]$$，每个输出元素是 $$k$$ 个乘积的和，即 $$k$$ 次乘法与 $$k - 1$$ 次加法，工程上按 $$k$$ 次乘加（multiply-add）计，每次乘加算 2 FLOPs。于是：

$$
\text{FLOPs}([m,k] \times [k,n]) = 2 \cdot m \cdot k \cdot n
$$

这是本文唯一需要记住的原始公式，后面所有算量都是它在不同形状下的代入。

三个维度在 Transformer 里各有固定的角色：$$k$$ 和 $$n$$ 是权重矩阵的两条边，由 `config.json` 决定，与输入无关；$$m$$ 是"这次一起算多少行"，由 batch 与序列长度决定，是运行时才知道的量。FLOPs 对三个维度都是线性的，但**读权重的字节数只与 $$k \cdot n$$ 有关，与 $$m$$ 无关**——这个不对称是后面所有结论的根源，先记住它。

还要注意"FLOPs"与"FLOPS"的区别：前者是运算次数（一个量），后者是每秒运算次数（一个速率）。H100 的 989 TFLOPS 是速率；一次前向的 15 GFLOPs 是量；两者相除才是时间。本文严格区分这两个写法。

一个 `nn.Linear(k, n)` 作用在 $$m$$ 个 token 上，就是一次 $$[m, k] \times [k, n]$$ 的 GEMM。它的权重有 $$k \cdot n$$ 个参数，FLOPs 是 $$2 m k n$$。两者相除：

$$
\frac{\text{FLOPs}}{\text{参数数} \times \text{token 数}} = \frac{2mkn}{kn \cdot m} = 2
$$

**每个参数、每个 token，贡献 2 FLOPs。**这个比值不依赖矩阵形状，所以可以直接对整个模型求和。

### 2. 前向的经典近似：2N FLOPs/token

把模型里所有参与 GEMM 的参数加起来记为 $$N_{gemm}$$，每个 token 的前向 FLOPs 就约是：

$$
\text{FLOPs}_{\text{weight}} \approx 2 \cdot N_{gemm}
$$

对 Llama-3-8B，$$N_{gemm} \approx 7.5\text{B}$$：

$$
2 \times 7.5 \times 10^9 \approx 15.0\ \text{GFLOPs/token}
$$

同样的方法：Llama-3-70B 总参数 70.55B，GEMM 部分 69.5B，每 token 约 139–141 GFLOPs（本系列统一取 141，即 $$2 \times 70.55\text{B}$$，两者差别在 embedding 那一项，不影响任何结论）。DeepSeek-V3 总参数 671B，但每个 token 只经过被路由到的 8 个专家加 1 个共享专家，**激活参数**约 37B，所以每 token 约 $$2 \times 37\text{B} = 74$$ GFLOPs——MoE 的全部意义就是让 $$N$$ 与 $$N_{active}$$ 分离，第五篇会展开。

这个近似里被忽略的项：RMSNorm（每 token 每层约 $$4d$$ 次运算）、RoPE、softmax、SwiGLU 里的逐元素乘法与激活函数、残差加法。它们都是 $$O(d)$$ 或 $$O(d_{ff})$$ 每 token，与 GEMM 的 $$O(d^2)$$ 相比小两到三个数量级。它们在 FLOPs 上可以忽略，**但在时间上不一定能忽略**——这些算子是 memory-bound 的，这是第八节讨论实测差距时要回来的一点。

### 3. 为什么 embedding 查表不算，而 lm_head 要算

embedding 层在数学上是 one-hot 向量乘以 $$[V, d]$$ 矩阵，如果真这么算是 $$2Vd$$ FLOPs。但没有任何实现这么做：token id 是一个整数，embedding 是按下标取出第 $$i$$ 行，这是一次 gather，读 $$d \times 2$$ 字节，零次乘法。所以 embedding 的 525.3M 参数不贡献 FLOPs。

lm_head 恰好相反。它的输入是最后一层输出的 $$[m, d]$$ 隐状态，输出是 $$[m, V]$$ 的 logits，这是一次实打实的 $$[m, 4096] \times [4096, 128256]$$ GEMM，每 token $$2 \times 4096 \times 128256 \approx 1.05$$ GFLOPs——占 8B 模型每 token 算量的 7%。这也是为什么 128K 的大词表在小模型上并不"免费"：它同时抬高了参数量、lm_head 的算量，以及 lm_head 那 1.05 GB 权重的读取。

### 4. attention 的上下文项：每层每 token 4ds

上面的 2N 只计算了"token 与权重"之间的乘法。attention 里还有一部分乘法是"token 与 token"之间的，它的算量不依赖参数量，依赖上下文长度 $$s$$。

对一个 query token，每个 head 做两次矩阵乘：

- $$QK^\top$$：$$[1, d_{head}] \times [d_{head}, s]$$，$$2 \cdot d_{head} \cdot s$$ FLOPs；
- $$PV$$：$$[1, s] \times [s, d_{head}]$$，$$2 \cdot d_{head} \cdot s$$ FLOPs。

$$n_h$$ 个 head 合计：

$$
\text{FLOPs}_{\text{attn}}^{\text{per layer, per token}} = 2 \cdot 2 \cdot n_h \cdot d_{head} \cdot s = 4 d s
$$

注意这里用的是 $$n_h$$ 而不是 $$n_{kv}$$：GQA 只减少 K、V 的存储与投影，每个 query head 仍然要与全部 $$s$$ 个 key 做点积。

代入 Llama-3-8B：每层 $$4 \times 4096 \times s = 16384 \cdot s$$，32 层：

$$
32 \times 16384 \cdot s = 524288 \cdot s \approx 0.524\ \text{MFLOPs} \times s
$$

于是：

| 上下文 $$s$$ | attention 上下文项 | 与权重项 15.0 GFLOPs 的比 |
|---|---|---|
| 2048 | 1.07 GFLOPs | 7% |
| 8192 | 4.29 GFLOPs | 29% |
| 32768 | 17.2 GFLOPs | 114% |
| 131072 | 68.7 GFLOPs | 458% |

在 8K 上下文，attention 的上下文项是权重项的不到三分之一，"2N 近似"仍然好用；到 128K，它是权重项的 4.6 倍，模型每生成一个 token 的算量主要花在"看历史"而不是"过权重"上。对 Llama-3-70B（$$d = 8192$$，80 层）在 128K：$$4 \times 8192 \times 131072 \times 80 \approx 344$$ GFLOPs，是它 141 GFLOPs 权重项的 2.4 倍。这一项对系统的意义，第四篇讲长上下文时会算得更细。

### 5. 训练的 6ND 与激活重算的 8N

训练一个 token 的 FLOPs 是前向的三倍，这是"6ND"的来源：

- 前向：$$2N$$，如上；
- 反向：每个 GEMM 要算两个梯度。对输入的梯度 $$\partial L / \partial X = \partial L / \partial Y \cdot W^\top$$ 是一次 $$[m, n] \times [n, k]$$，$$2mkn$$；对权重的梯度 $$\partial L / \partial W = X^\top \cdot \partial L / \partial Y$$ 是一次 $$[k, m] \times [m, n]$$，也是 $$2mkn$$。合计 $$4N$$。

所以每 token $$6N$$，训练 $$D$$ 个 token 总计 $$6ND$$。若开了激活重算（activation checkpointing），反向前要把前向重做一遍，再加 $$2N$$，成为 $$8N$$。

代入 Llama-3-8B 在 15T token 上训练：

$$
6 \times 8.03 \times 10^9 \times 15 \times 10^{12} \approx 7.2 \times 10^{23}\ \text{FLOPs}
$$

一张 H100 的 BF16 峰值 989 TFLOPS，按 40% 的 MFU（第七节解释为什么这是一个合理甚至偏乐观的假设）有效算力约 396 TFLOPS：

$$
\frac{7.2 \times 10^{23}}{3.96 \times 10^{14} \times 3600} \approx 5.1 \times 10^5\ \text{GPU-小时}
$$

即约 **50 万 H100 GPU-小时**；如果全程激活重算（$$8N$$），约 68 万。同样的算法对 Llama-3-70B 得到约 450 万 GPU-小时。这些数字与公开模型卡上的量级一致（模型卡的数字更大，因为包含更低的 MFU、故障重启与实验）。反过来用它们也可以做一件事：给一个训练团队的 GPU 数量与时间，反推他们大概训练了多少 token。


## 二、prefill 与 decode：同一组矩阵，两种 GEMM 形状

自回归推理有两个阶段，它们跑的是**同一组权重**，但 GEMM 的形状完全不同。

### 1. prefill：m = s

prefill 把 prompt 的 $$s$$ 个 token 一次送入模型。每个 `nn.Linear` 看到的输入是 $$[s, k]$$（多请求时是 $$[\sum s_i, k]$$），GEMM 是 $$[s, k] \times [k, n]$$，$$m = s$$。以 Llama-3-8B、$$s = 8192$$ 为例，每层的几个 GEMM 形状是：

```text
W_Q      [8192, 4096] x [4096, 4096]
W_K, W_V [8192, 4096] x [4096, 1024]
W_O      [8192, 4096] x [4096, 4096]
gate, up [8192, 4096] x [4096, 14336]
down     [8192, 14336] x [14336, 4096]
QK^T     每 head [8192, 128] x [128, 8192]
PV       每 head [8192, 8192] x [8192, 128]
```

$$m$$ 维是几千，$$k$$、$$n$$ 也是几千，这是 Tensor Core 最喜欢的形状：每读一次权重可以复用 8192 次。prefill 的总 FLOPs 是每 token 算量乘以 $$s$$（attention 项本身还随 $$s$$ 增长，所以是 $$s^2$$）。

### 2. decode：m = B

decode 每步只处理每个请求最新的 1 个 token。batch 里有 $$B$$ 个请求，每个 `nn.Linear` 看到的输入是 $$[B, k]$$，GEMM 是 $$[B, k] \times [k, n]$$，$$m = B$$。$$B = 1$$ 时它退化为矩阵向量乘（GEMV）：

```text
W_Q      [B, 4096] x [4096, 4096]
gate, up [B, 4096] x [4096, 14336]
down     [B, 14336] x [14336, 4096]
QK^T     每 head 每请求 [1, 128] x [128, s_i]
PV       每 head 每请求 [1, s_i] x [s_i, 128]
```

而 attention 的形状里 $$s$$ 仍然存在——它出现在 K、V 一侧，也就是 KV cache。这一步的算量是 $$B \times (2N_{gemm} + 4dsL)$$，是 prefill 的 $$1/s$$，但读的权重字节数与 prefill 一样多。这个不对称是全文的核心。

### 3. 两个阶段的算量

对 Llama-3-8B、上下文 8K：

```text
                        prefill (s=8192)            decode 一步 (B=1, s=8192)
权重项                   8192 x 15.0 G = 123 TFLOP    15.0 GFLOPs
attention 上下文项        8192 x 4.29 G /2 = 17.6 T   4.29 GFLOPs   （因果掩码减半，见第五节）
合计                     约 140 TFLOP                 19.3 GFLOPs
比值                     约 7300 : 1
```

一次 8K 的 prefill 等于 7000 多步 decode 的算量，但两者都要把 16 GB 权重读一遍。

这个不对称直接解释了推理引擎里几个常见设计：

- **continuous batching**：既然一步 decode 无论 $$B$$ 是 1 还是 64 都要读一遍权重，把尽可能多的请求塞进同一步就是免费的吞吐。vLLM 一类引擎在每步之间动态加入新请求、移出结束的请求，就是为了让 $$m = B$$ 尽量大；
- **chunked prefill**：一个 8K 的 prefill 要占用 GPU 约四分之一秒，期间其他请求的 decode 全部停住。把 prefill 切成若干块与 decode 步交替执行，用 prefill 的高算术强度"填满" decode 步空闲的 Tensor Core；
- **prefill/decode 分离**：两个阶段的瓶颈不同（一个吃算力，一个吃带宽），把它们放到不同的 GPU 上各自调优，中间通过网络传 KV cache。

这些设计能否成立、收益多大，都可以用本篇的数字直接估算，而不需要先实现出来。


## 三、访存量：每一步要从 HBM 读什么

FLOPs 是成本的一半。另一半是每一步必须从 HBM 搬进 SM 的字节数。decode 一步要读三类数据。

### 1. 权重：每步读一遍，16.06 GB

GPU 的片上 SRAM（H100 每个 SM 256 KB，共 132 个 SM，加 50 MB L2）放不下任何一层的权重，所以每一步 decode，模型的每个权重矩阵都要从 HBM 完整读一遍。BF16 每参数 2 字节：

$$
8.03 \times 10^9 \times 2 = 16.06\ \text{GB}
$$

这个数字与 $$B$$ 无关：batch 里 1 个请求和 100 个请求，权重都只读一遍。Llama-3-70B 是 141 GB——已经超过一张 80 GB 的 H100，必须切到至少两张卡上。DeepSeek-V3 BF16 是 1342 GB，FP8 是 671 GB，一台 8 卡 H100 节点（640 GB）连 FP8 权重都放不下。

多卡时这个数怎么变？tensor parallel 把每个权重矩阵按列或按行切成 $$n$$ 份，每张卡只读自己那 $$1/n$$，每步读权重的时间也变成 $$1/n$$——这是 TP 在 decode 上真正的收益：不是算得更快，而是**每张卡读得更少**。代价是每层两次 all-reduce，通信量每 token 每层约 $$2 \times 2d$$ 字节，在 NVLink 上通常小于省下的 HBM 时间，但它引入了同步点，$$B$$ 小时这些同步的固定延迟会吃掉相当一部分收益。pipeline parallel 则不同：每张卡持有 $$L/n$$ 层的完整权重，每步仍然要把这些层读一遍，一步 decode 的总时间不变，只是每张卡各读自己的部分——PP 对 decode 延迟没有帮助，只解决容量。

### 2. KV cache：每 token 128 KiB，乘以上下文乘以 batch

每个请求的历史 token 的 K、V 都要在 attention 里读一遍。每层每 token 存 $$n_{kv}$$ 个 head 的 K 和 V，各 $$d_{head}$$ 个元素：

$$
\text{bytes/token} = 2 \cdot L \cdot n_{kv} \cdot d_{head} \cdot \text{bytes/elem}
$$

第一个 2 是 K 与 V 两份。代入 Llama-3-8B、BF16：

$$
2 \times 32 \times 8 \times 128 \times 2 = 131072\ \text{B} = 128\ \text{KiB}
$$

如果它用的是 MHA（$$n_{kv} = n_h = 32$$），这个数是 512 KiB——GQA 把 KV cache 压到四分之一，第三篇会详细推导各种 attention 变体下的这个数。Llama-3-70B 是 $$2 \times 80 \times 8 \times 128 \times 2 = 320$$ KiB。

decode 一步要读的 KV 字节数是：

$$
\text{KV bytes/step} = B \cdot s \cdot 128\ \text{KiB}
$$

$$B = 1$$、$$s = 8192$$ 时是 1 GiB，与 16 GB 的权重相比是小项；但它随 $$B$$ 和 $$s$$ 线性增长，而权重不随 $$B$$ 增长，所以总会有一个点 KV 的读取超过权重。$$B = 64$$、$$s = 8192$$ 时：

$$
64 \times 8192 \times 128\ \text{KiB} = 64\ \text{GiB}
$$

已经是权重的四倍。这个例子第五节会算完。

### 3. 激活值：decode 时可以忽略

decode 每步每层的激活是 $$[B, d]$$ 与 $$[B, d_{ff}]$$ 量级的张量，$$B = 64$$ 时是 $$64 \times 14336 \times 2 = 1.8$$ MB，与 GB 量级的权重和 KV 相比差三个数量级，而且大部分能留在 L2 里。prefill 时激活是 $$[s, d_{ff}]$$，$$s = 8192$$ 时 235 MB 每层，需要写回 HBM，但 prefill 是 compute-bound（下面会证明），这些字节数不决定时间。真正让激活值成为问题的是**训练**，第七节单独讨论。


## 四、Roofline：把 FLOPs 和字节数放到同一张图上

### 1. 两条上限

任何一段 GPU 上的计算，运行时间至少是两个数中较大的那个：

$$
T \ge \max\left( \frac{\text{FLOPs}}{P_{peak}},\ \frac{\text{Bytes}}{BW} \right)
$$

第一项是算力上限：H100 BF16 dense 989 TFLOPS。第二项是带宽上限：HBM3 3.35 TB/s。哪一项更大，这段计算就受哪一项限制——compute-bound 或 memory-bound。

### 2. 算术强度与 ridge point

把两项的比值提出来。定义算术强度：

$$
I = \frac{\text{FLOPs}}{\text{Bytes}}\quad [\text{FLOP/byte}]
$$

计算的实际算力是：

$$
P_{achieved} = \min\left( P_{peak},\ I \cdot BW \right)
$$

两条线的交点：

$$
I_{ridge} = \frac{P_{peak}}{BW}
$$

H100 SXM：$$989 / 3.35 \approx 295$$ FLOP/byte。A100：$$312 / 2.0 \approx 156$$。一段计算要吃满 H100 的 Tensor Core，每从 HBM 读一个字节，至少要做 295 次浮点运算；做不到，Tensor Core 就在等数据。

注意 ridge point 的量级在几代 GPU 上都在上升：A100 的 156 到 H100 的 295，算力增长快于带宽增长。这意味着**同一个模型、同一个 batch，在新一代 GPU 上更容易 memory-bound**。

### 3. 图

```text
  实际算力
  (FLOP/s, 对数轴)
     |
 989T|- - - - - - - - - - - - - - - -+--------------------- 算力上限 (compute roof)
     |                              /|
     |                            /  |
     |                          /    |
     |          带宽上限        /      |
     |      斜率 = 3.35 TB/s  /        |
     |                      /          |
     |                    /            |
     |                  /              |
   3T|                / x              |
     |              /   decode B=1     |
     |            /     I ~ 1          |
     +----------/--------|-------------|------------------------> 算术强度 I
             1         10           295          10^4      (FLOP/byte, 对数轴)
                                  ridge point       prefill s=8192, I ~ 8000
                    memory-bound  <--  |  -->  compute-bound
```

左侧斜线是带宽上限，右侧水平线是算力上限。一个工作负载由它的算术强度决定落在横轴哪一点，垂直向上碰到的第一条线就是它能达到的算力。$$B = 1$$ 的 decode 落在 $$I \approx 1$$，能用的算力只有 3.35 TFLOPS——峰值的 0.3%。

Roofline 是一个粗粒度模型，它假设访存与计算完美重叠、所有字节只从 HBM 读一次、算力可以被完全用满。真实 kernel 的点总是落在两条线之下。但它的用法不在于精确预测，而在于回答两个问题：这段计算**应该**受什么限制？它现在离那个限制还有多远？前者决定优化方向（减字节还是减 FLOPs），后者决定还值不值得优化。

同一张图也可以用来看单个算子。以 RMSNorm 为例：读 $$[m, d]$$ 的输入、写同形状的输出，每个元素约 4 次浮点运算（平方、求和、乘 rsqrt、乘 gain），读写各 2 字节，强度约 $$4 / 4 = 1$$ FLOP/byte。它永远在带宽线上，时间就是字节数除以 3.35 TB/s，与 GEMM 的时间无关、也不随 Tensor Core 升级而变快。这是为什么"FLOPs 上可忽略"的算子在时间上不可忽略：它们各自贡献一段带宽时间，算子融合（把 RMSNorm 融进前一个 GEMM 的 epilogue、把残差加融进 RMSNorm）是把这些字节读写次数合并的唯一办法。

### 4. decode 权重 GEMM 的算术强度 ≈ B

对权重部分，decode 一步的 FLOPs 是 $$2 N_{gemm} B$$，读的字节是 $$2 N_{gemm}$$（BF16）：

$$
I_{\text{weight}} = \frac{2 N_{gemm} \cdot B}{2 N_{gemm}} = B\ \text{FLOP/byte}
$$

这个结论干净得令人不安：**BF16 decode 的算术强度在数值上就等于 batch 大小。**每 2 字节的权重被读进来，对 $$B$$ 个 token 各做一次乘加，共 $$2B$$ FLOPs。

$$B = 1$$ 时 $$I = 1$$，距 ridge point 295 差两个多数量级。这就是"decode 是 memory-bound 的"这句话的全部含义：不是某个 kernel 写得不好，而是工作负载的算术强度天然比硬件的 ridge point 低两个数量级。任何 kernel 优化都不可能把 $$B = 1$$ 的 decode 变成 compute-bound；能做的只有提高 $$B$$（continuous batching）、减少每步读的字节（量化，第七篇）、或者一步产出多个 token（投机解码，第七篇）。

顺便得到 FP8 的情况：权重字节减半，$$I_{\text{weight}} = 2B$$；同时 H100 FP8 算力翻倍到 1979 TFLOPS，ridge point 变为 $$1979 / 3.35 \approx 590$$。距离没有变：仍然需要 $$B \approx 295$$。量化在 decode 上的收益来自字节数减少，而不是算力提高。

### 5. attention 读 KV cache 的算术强度 = g

attention 上下文项的算术强度也可以单独算。每层每 token 做 $$4 d s = 4 n_h d_{head} s$$ FLOPs，读 $$2 n_{kv} d_{head} s \times 2$$ 字节的 K、V：

$$
I_{\text{KV}} = \frac{4 n_h d_{head} s}{4 n_{kv} d_{head} s} = \frac{n_h}{n_{kv}} = g
$$

它等于 GQA 的组大小 $$g$$，与 $$s$$、$$B$$ 都无关——每个请求的 KV 只被自己读，batch 不带来复用。Llama-3-8B 的 $$g = 4$$，70B 的 $$g = 8$$，MHA 是 1。这说明 KV cache 的读取是比权重更"顽固"的 memory-bound 部分：权重的强度随 $$B$$ 线性上升，KV 的强度是个常数。第三篇讲 MLA 时会看到，把 K、V 压成一个低秩向量再"吸收"到权重里，本质上就是把这个常数抬高。


## 五、时间下界

### 1. decode：4.8 ms，208 token/s

$$B = 1$$、短上下文时 KV 可忽略，decode 一步的时间下界就是读一遍权重：

$$
T_{decode} \ge \frac{16.06\ \text{GB}}{3.35\ \text{TB/s}} \approx 4.8\ \text{ms}
$$

对应单个请求的生成速度上限：

$$
\frac{1}{4.8\ \text{ms}} \approx 208\ \text{token/s}
$$

作为对照，这一步的算力时间是 $$19.3\ \text{GFLOPs} / 989\ \text{TFLOPS} \approx 0.02$$ ms，是访存时间的 1/250——与 $$I / I_{ridge} = 1/295$$ 一致。任何在单张 H100 上宣称 Llama-3-8B BF16 单流超过 208 token/s 的数字，要么用了量化，要么用了投机解码，要么测的不是这个模型。同样的算法，Llama-3-70B 若能放进一张卡：$$141 / 3.35 \approx 42$$ ms，约 24 token/s；A100 上的 8B 是 $$16.06 / 2.0 \approx 8.0$$ ms，125 token/s。

这个 4.8 ms 值得多看一眼：它与模型的算力需求完全无关。把 Llama-3-8B 的 FFN 换成一半大小的 $$d_{ff}$$，FLOPs 减少 40%，但只要参数字节数不变，decode 时间下界就不变；反过来把权重量化到 INT4（每参数约 0.53 字节，第七篇会算精确的 4.25 bit），FLOPs 不变，下界降到约 1.3 ms。**对 decode 而言，"模型多大"的正确度量是字节，不是 FLOPs，也不是参数个数。**

这也解释了 70B 与 8B 在 decode 上的差距为什么是 8.8 倍而不是"参数多所以更慢"这种模糊的说法：141 GB 对 16 GB，字节数之比就是时间之比。用两张 H100 做 TP=2 跑 70B，每卡读 70 GB，下界 21 ms、约 48 token/s；用 8 卡 TP=8，每卡读 17.6 GB，下界 5.3 ms，接近单卡 8B 的速度——前提是 all-reduce 的时间被重叠掉。

### 2. 加上 KV cache：8K 上下文、batch 64

$$B = 64$$、$$s = 8192$$，每步读：

$$
\text{Bytes} = 16.06\ \text{GB} + 64\ \text{GiB} = 16.06 + 68.7 = 84.8\ \text{GB}
$$

$$
T_{decode} \ge \frac{84.8\ \text{GB}}{3.35\ \text{TB/s}} \approx 25.3\ \text{ms}
$$

这一步的 FLOPs 是 $$64 \times 19.3 = 1.24$$ TFLOP，算力时间 1.25 ms，仍然是 memory-bound。吞吐是 $$64 / 25.3\ \text{ms} \approx 2500$$ token/s——batch 从 1 涨到 64，吞吐只涨了 12 倍，因为每步的字节数涨了 5.3 倍。

用"有效 batch"来描述这个损失：这一步的算术强度是

$$
I = \frac{1.24\ \text{TFLOP}}{84.8\ \text{GB}} \approx 14.6\ \text{FLOP/byte}
$$

按第四节 $$I_{\text{weight}} = B$$ 的标度，它相当于一个**没有 KV cache 时 batch 约 15** 的 decode。名义 batch 64，Roofline 上的有效 batch 15，差距全是 KV cache 的读取。上下文越长，这个折损越大。

### 3. prefill：8K 约 0.24–0.27 s，128K 约 11 s

prefill 的 8192 个 token 一次进入模型，权重项 $$8192 \times 15.0\ \text{GFLOPs} = 123$$ TFLOP。attention 上下文项若按每个 token 都看全部 8192 个 key 算，是 $$8192 \times 4.29 = 35$$ TFLOP，合计约 158 TFLOP。但因果掩码下第 $$i$$ 个 token 只看前 $$i$$ 个，$$QK^\top$$ 与 $$PV$$ 里有一半是被 mask 掉的，FlashAttention 一类 kernel 会直接跳过这些块，attention 项减半为 17.6 TFLOP，合计约 140 TFLOP。

按 60% 的 MFU（prefill 是 compute-bound，大 GEMM 上这是可以达到的效率）：

$$
T_{prefill} \ge \frac{140 \sim 158\ \text{TFLOP}}{989\ \text{TFLOPS} \times 0.6} \approx 0.24 \sim 0.27\ \text{s}
$$

这是 8K prompt 下 TTFT（time to first token）的物理下限。算术强度上，这一步的 FLOPs 是 140 TFLOP、读权重 16 GB，$$I \approx 8700$$，远在 ridge point 右侧。

128K 的 prefill：权重项 $$131072 \times 15.0 = 2.0$$ PFLOP；attention 项按因果的 $$s^2 / 2$$：$$0.524\ \text{MFLOPs} \times 131072^2 / 2 \approx 4.5$$ PFLOP；合计 6.5 PFLOP，60% MFU 下约 11 s。attention 项在这里是权重项的 2.2 倍，而且随 $$s^2$$ 增长——这就是长上下文 prefill 要做 chunked prefill、要把 prefill 与 decode 分开调度的算量原因。

把 prefill 与 decode 的下界放在一起，就是一个请求端到端延迟的骨架。一个 8K prompt、生成 512 个 token 的请求，在单张 H100 上 $$B = 1$$ 时：prefill 约 0.25 s，decode 512 步各约 5 ms 共 2.6 s，合计约 2.8 s，其中 90% 在 decode，而 decode 期间 Tensor Core 的利用率不到 1%。这就是推理服务必须做 batching 的算量理由：不 batch，一张 H100 的绝大部分算力在等 HBM。

prefill 的字节数也值得算一次以确认它确实 compute-bound：读权重 16 GB，写 KV cache $$8192 \times 128\ \text{KiB} = 1$$ GiB，激活的读写按每层十几个 $$[8192, 4096]$$ 到 $$[8192, 14336]$$ 的张量估算约几 GB。总字节数在 20–30 GB 量级，带宽时间不到 10 ms，与 240 ms 的算力时间相比可以忽略——prefill 的算术强度在几千，Roofline 图上落在 ridge point 右侧很远。


## 六、核心问题：batch 多大 decode 才 compute-bound？

### 1. 只看权重：B ≈ 295

由第四节，BF16 decode 权重 GEMM 的算术强度 $$I = B$$，compute-bound 的条件是 $$I \ge I_{ridge}$$：

$$
B \ge \frac{P_{peak}}{BW} = \frac{989 \times 10^{12}}{3.35 \times 10^{12}} \approx 295
$$

换一个角度验证：$$B$$ 个请求的算力时间 $$B \times 15.0\ \text{GFLOPs} / 989\ \text{TFLOPS}$$ 等于访存时间 4.8 ms 时，$$B = 4.8\ \text{ms} \times 989\ \text{TFLOPS} / 15.0\ \text{GFLOPs} \approx 316$$；差异来自这里用的 $$N_{gemm} = 7.5\text{B}$$ 而字节数用的 $$N = 8.03\text{B}$$。量级是 300 左右。A100 上是 156。

所以答案的前半段是：**在 H100 上，Llama-3-8B 的 decode 要到 batch 约 300 才从 memory-bound 转为 compute-bound。**

### 2. 加上 KV cache：8K 上下文下单卡不可达

$$B = 295$$ 个请求同时在 decode，每个上下文 8K，KV cache 的总量是：

$$
295 \times 8192 \times 128\ \text{KiB} \approx 295\ \text{GiB}
$$

一张 H100 放下 16.06 GB 权重后剩下约 64 GB（下面按 64 GiB 粗算）。295 GiB 是它的 4.6 倍。**放不下，所以这个 batch 在单卡、8K 上下文下不可达。**

进一步，就算容量不是问题，8K 上下文下 KV 读取的算术强度也把总强度压住了。$$B = 295$$、$$s = 8192$$ 时每步读 $$16.06 + 317 = 333$$ GB，算 $$295 \times 19.3 = 5.7$$ TFLOP，$$I \approx 17$$——仍然是 memory-bound，且随 $$B$$ 增大趋近于一个常数：

$$
\lim_{B \to \infty} I = \frac{2N_{gemm} + 4dsL}{s \cdot \text{KV bytes/token}} = \frac{19.3\ \text{GFLOPs}}{8192 \times 128\ \text{KiB}} \approx 18
$$

在 8K 上下文下，BF16 decode **无论 batch 多大都不会 compute-bound**。只有上下文短到 KV 读取可忽略时，$$I \approx B$$ 的标度才成立。

### 3. 64 GB 的预算：B × s ≤ 52 万

单卡上 KV cache 能容纳的 token 总数：

$$
\frac{64\ \text{GiB}}{128\ \text{KiB}} = 524288 \approx 52\ \text{万 token}
$$

这是 batch 与上下文乘积的上限：$$B \times s \le 524288$$。几个点：

```text
上下文 s      最大 batch B      每步读 KV       每步下界       I (FLOP/byte)
   1024         512            64 GiB          25 ms          116
   2048         256            64 GiB          25 ms           58
   8192          64            64 GiB          25 ms           15
  32768          16            64 GiB          25 ms            3.7
 131072           4            64 GiB          25 ms            0.9
```

注意到当 KV cache 把显存填满时，每步读的 KV 字节数总是 64 GiB，与 $$s$$ 无关——总时间下界固定在约 25 ms，变的只是这 25 ms 里产出多少个 token。$$s = 1024$$ 时 $$B = 512$$，强度 116，是这张卡上离 ridge point 最近的配置，但仍差 2.5 倍。

这张表还说明了一件事：在显存被 KV cache 填满的前提下，**吞吐与上下文长度成反比**。同样 25 ms 一步，1K 上下文能产出 512 个 token，128K 只能产出 4 个；每 token 的成本差 128 倍。这是长上下文服务比短上下文贵得多的直接原因，也是为什么服务方按"输入 token + 输出 token"计费而不是按请求数计费——它们对应的是真实的 HBM 字节数。

答案的后半段：**考虑 KV cache 之后，B ≈ 295 在 8K 上下文下既放不下、也不会 compute-bound；单卡 Llama-3-8B 的 BF16 decode 在任何实际上下文长度下都是 memory-bound 的。**要改变这个结论，只能减字节：量化权重（第七篇）、压缩 KV cache（第三篇 GQA/MLA、第七篇 KV 量化），或者用多卡把权重读取分摊（tensor parallel 让每卡只读 $$1/n$$ 的权重，但也只提供 $$1/n$$ 的算力——ridge point 不变，只是每卡的 KV 显存变多了）。


## 七、训练侧：激活值显存与 MFU

推理时激活值可以忽略，训练时不能：反向传播需要每一层前向的中间结果，它们要在显存里从前向一直活到反向。

### 1. 每层激活值：sbh(34 + 5as/h) 字节

Korthikanti 等 2022（Megatron 团队，*Reducing Activation Recomputation in Large Transformer Models*）给了一个标准 Transformer 层在 16-bit 精度、不用 FlashAttention 时每层激活值的估算式，其中 $$s$$ 是序列长度、$$b$$ 是 micro-batch、$$h$$ 是隐层维度、$$a$$ 是 head 数：

$$
\text{Activation bytes per layer} = s b h \left( 34 + 5 \frac{a s}{h} \right)
$$

逐项来源（每个元素 2 字节，dropout mask 1 字节）：

- **attention 块 $$11 sbh + 5 a s^2 b$$**：Q、K、V 投影的输入 $$2sbh$$；$$QK^\top$$ 的两个输入 Q 和 K，$$4sbh$$；softmax 输出 $$[b, a, s, s]$$ 要保留给反向，$$2as^2b$$；softmax 后的 dropout mask $$as^2b$$，dropout 输出 $$2as^2b$$；$$PV$$ 的 V 输入 $$2sbh$$；输出投影 $$W_O$$ 的输入 $$2sbh$$；最后的 dropout mask $$sbh$$。与 $$s$$ 线性的项合计 $$11sbh$$，与 $$s^2$$ 相关的项合计 $$5as^2b$$。
- **FFN 块 $$19 sbh$$**：两个 linear 的输入 $$2sbh + 8sbh$$（中间维 $$4h$$），GeLU 的输入 $$8sbh$$，dropout mask $$sbh$$。
- **两个 LayerNorm $$4 sbh$$**：各保存输入 $$2sbh$$。

三部分相加：$$34 sbh + 5 a s^2 b = sbh(34 + 5as/h)$$。

代入 Llama-3-8B 的形状（$$h = 4096$$，$$a = 32$$，$$s = 8192$$，$$b = 1$$）：

$$
34 sbh = 34 \times 8192 \times 4096 = 1.14\ \text{GB} \approx 1.06\ \text{GiB}
$$

$$
5 a s^2 b = 5 \times 32 \times 8192^2 = 10.7\ \text{GB} = 10\ \text{GiB}
$$

每层约 11 GiB，32 层约 **354 GiB**——单是一个 8K 序列、micro-batch 为 1 的激活值就是四张 H100 的显存，而且其中 90% 是那个 $$s^2$$ 项。$$s^2$$ 项与线性项的比是 $$5as/(34h) = 5 \times 32 \times 8192 / (34 \times 4096) \approx 9.4$$；在 $$s = h \cdot 34 / (5a) = 870$$ 附近两者相等，超过这个长度 attention 矩阵就成为激活显存的主体。（Llama 用 SwiGLU 而非 GeLU、$$d_{ff} = 14336$$ 而非 $$4h$$、且没有 dropout，线性项的系数与 34 略有不同，但 $$s^2$$ 项的结论不变。）

### 2. FlashAttention 与重算如何去掉 s² 项

$$5as^2b$$ 全部来自把 $$[b, a, s, s]$$ 的 attention 矩阵物化到 HBM。FlashAttention（Dao 等 2022）不物化它：前向按块在 SRAM 里算 softmax 与 $$PV$$，只把输出 $$O$$（$$2sbh$$）和每行的 logsumexp（$$abs$$ 个数）写回；反向时按块重算 $$QK^\top$$ 与 softmax。代价是 attention 部分多了约一次 $$QK^\top$$ 的重算（每层 $$2ds$$ 每 token，远小于 $$2N$$），换来每层激活从 11 GiB 降到约 1 GiB。

Korthikanti 等 2022 提出的选择性重算（selective recomputation）是同一个思路的另一面：不保存 $$s^2$$ 项的几个张量，反向时重算它们，只用 attention 的算量（不到总量的几分之一）就消掉了 90% 的激活显存；而全量重算（每层只保存输入，整层重做前向）代价是整个 $$2N$$，即第一节说的 $$8N$$。今天的训练框架（Megatron-LM、以及基于 FlashAttention 的任何实现）默认已经没有这个 $$s^2$$ 项，激活值显存回到与 $$s$$ 线性的量级——这才使 8K 以上的序列训练成为可能。

### 3. MFU 与 HFU

MFU（model FLOPs utilization）定义为模型**理论上需要**的 FLOPs 吞吐与硬件峰值算力的比值：

$$
\text{MFU} = \frac{\text{tokens/s} \times 6N}{P_{peak} \times n_{GPU}}
$$

HFU（hardware FLOPs utilization）的分子改为硬件**实际执行**的 FLOPs，把重算也算进去（全量重算时用 $$8N$$）。HFU 总不低于 MFU，MFU 才是衡量"多少算力变成了有用的训练进度"的指标。

从 token 吞吐反推很直接。Llama-3-8B 在 $$n$$ 张 H100 上，若观测到每卡每秒 $$t$$ 个 token：

$$
\text{MFU} = \frac{t \times 6 \times 8.03 \times 10^9}{989 \times 10^{12}} = \frac{t}{20527}
$$

即每卡 8200 token/s 对应 40% MFU，10300 token/s 对应 50%。反过来，读到一个训练吞吐数字，除以两万就是 MFU。

MFU 与 HFU 的差别在有重算时才显现。全量重算下硬件每 token 做 $$8N$$，若测得 HFU 为 50%，MFU 只有 $$50\% \times 6/8 = 37.5\%$$。报告 HFU 会让数字好看，但用户关心的是每小时训练了多少 token，那是 MFU。读训练报告时要先看清用的是哪一个。

为什么 40–50% 已经算好：

- **非 GEMM 算子**：RMSNorm、RoPE、softmax、SwiGLU 的逐元素乘、残差加，FLOPs 可忽略，但每一个都要把 $$[s, d]$$ 或 $$[s, d_{ff}]$$ 的激活从 HBM 读出写回，算术强度约 1，跑在带宽上限上。它们占 FLOPs 的 1%，占时间的 10–20% 是常态；算子融合就是为了把这些访存合并；
- **通信**：tensor parallel 的 all-reduce、pipeline parallel 的气泡、data parallel 的梯度同步，能与计算重叠的部分有限；
- **GEMM 本身达不到峰值**：989 TFLOPS 是理想形状、理想频率下的数字，实际大 GEMM 在 70–80% 的峰值；功耗墙下频率会降；
- **micro-batch 与 pipeline 的尾效应**、优化器步骤、数据加载、checkpoint 保存。

把这些乘起来，$$0.8 \times 0.85 \times 0.8 \times 0.9 \approx 0.49$$——50% 左右是大规模训练在没有明显低效的情况下的自然上限。公开的大规模训练报告中 MFU 多在 35–45% 之间，与这个估算一致。


## 八、实践：llm_cost.py 增加 FLOPs、字节数与时间下界

在第一篇脚本（`ModelConfig`、`GPU`、`param_count`）的基础上，本篇新增五个函数和一个打印表格。为了独立可运行，下面把骨架也一并给出。

### 1. 新增函数

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


# ---- 第一篇：参数量 ------------------------------------------------------

def param_count(cfg):
    """第一篇的逐组件参数量（重给以便独立运行），返回 dict。"""
    d, d_kv = cfg.hidden, cfg.n_kv_heads * cfg.head_dim
    attn = d * d + 2 * d * d_kv + d * d          # W_Q, W_K, W_V, W_O
    ffn = 3 * d * cfg.d_ff                         # gate, up, down
    norms = 2 * d
    per_layer = attn + ffn + norms
    embed = cfg.vocab * d
    lm_head = 0 if cfg.tie_embeddings else cfg.vocab * d
    total = cfg.layers * per_layer + d + embed + lm_head   # + 最终 RMSNorm
    return {"per_layer": per_layer, "all_layers": cfg.layers * per_layer,
            "embedding": embed, "lm_head": lm_head, "total": total}


def gemm_params(cfg):
    """参与 GEMM 的参数量：embedding 只做查表，不算；lm_head 永远要算一次。"""
    p = param_count(cfg)
    return cfg.layers * (p["per_layer"] - 2 * cfg.hidden) + cfg.vocab * cfg.hidden


# ---- 第二篇：算量、字节数、Roofline --------------------------------------

def forward_flops_per_token(cfg, ctx):
    """每 token 前向 FLOPs = 权重项 2N_gemm + 上下文项 4·d·ctx·L。"""
    weight_flops = 2 * gemm_params(cfg)
    attn_flops = 4 * cfg.hidden * ctx * cfg.layers
    return weight_flops, attn_flops


def weight_bytes(cfg, dtype_bytes=2):
    return param_count(cfg)["total"] * dtype_bytes


def kv_bytes_per_token(cfg, dtype_bytes=2):
    """K 和 V 各一份：2 · L · n_kv · d_head · bytes/elem。第三篇扩展到 MLA。"""
    return 2 * cfg.layers * cfg.n_kv_heads * cfg.head_dim * dtype_bytes


def decode_step_time(cfg, gpu, batch, ctx, dtype_bytes=2):
    """decode 一步的理论下界：max(访存时间, 算力时间)。返回各分项便于打印。"""
    w_flops, a_flops = forward_flops_per_token(cfg, ctx)
    flops = batch * (w_flops + a_flops)
    w_bytes = weight_bytes(cfg, dtype_bytes)
    kv_bytes = batch * ctx * kv_bytes_per_token(cfg, dtype_bytes)
    total_bytes = w_bytes + kv_bytes
    t_mem = total_bytes / gpu.bandwidth
    t_cmp = flops / gpu.bf16_flops
    return {
        "flops": flops,
        "weight_bytes": w_bytes,
        "kv_bytes": kv_bytes,
        "intensity": flops / total_bytes,
        "t_mem": t_mem,
        "t_cmp": t_cmp,
        "t": max(t_mem, t_cmp),
        "bound": "memory" if t_mem >= t_cmp else "compute",
    }


def prefill_time(cfg, gpu, seq, mfu=0.6, causal=True):
    """prefill 的理论时间：总 FLOPs / (峰值算力 × MFU)。因果掩码可让上下文项减半。"""
    w_flops, a_flops = forward_flops_per_token(cfg, seq)
    if causal:
        a_flops /= 2
    total = seq * (w_flops + a_flops)
    return total, total / (gpu.bf16_flops * mfu)


def ridge_point(gpu):
    return gpu.bf16_flops / gpu.bandwidth


def fmt(x, unit=""):
    for div, suffix in ((1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= div:
            return f"{x / div:.2f} {suffix}{unit}"
    return f"{x:.2f} {unit}"


def report(cfg, gpu, ctx=8192, batches=(1, 8, 64, 295)):
    total, gemm = param_count(cfg)["total"], gemm_params(cfg)
    w_flops, a_flops = forward_flops_per_token(cfg, ctx)
    print(f"== {cfg.name} on {gpu.name} (ridge = {ridge_point(gpu):.0f} FLOP/byte) ==")
    print(f"params total {total / 1e9:.2f} B, gemm {gemm / 1e9:.2f} B")
    print(f"weight FLOPs/token   {fmt(w_flops, 'FLOPs')}")
    print(f"attn FLOPs/token@{ctx} {fmt(a_flops, 'FLOPs')}")
    print(f"weight bytes (BF16)  {fmt(weight_bytes(cfg), 'B')}")
    print(f"KV bytes/token       {kv_bytes_per_token(cfg) / 1024:.0f} KiB")
    print()
    print(f"{'batch':>6} {'ctx':>6} {'KV read':>10} {'I(FLOP/B)':>10} "
          f"{'t_mem(ms)':>10} {'t_cmp(ms)':>10} {'step(ms)':>9} {'tok/s':>8} bound")
    for b in batches:
        r = decode_step_time(cfg, gpu, b, ctx)
        print(f"{b:>6} {ctx:>6} {r['kv_bytes'] / 2**30:>8.1f}Gi "
              f"{r['intensity']:>10.1f} {r['t_mem'] * 1e3:>10.2f} "
              f"{r['t_cmp'] * 1e3:>10.2f} {r['t'] * 1e3:>9.2f} "
              f"{b / r['t']:>8.0f} {r['bound']}")
    print()
    for seq in (8192, 131072):
        for causal in (False, True):
            fl, t = prefill_time(cfg, gpu, seq, mfu=0.6, causal=causal)
            print(f"prefill {seq:>6} causal={str(causal):5}  {fmt(fl, 'FLOP'):>14}  "
                  f"@60% MFU {t:.3f} s")
    print()


if __name__ == "__main__":
    report(LLAMA3_8B, H100)
    report(LLAMA3_70B, H100, batches=(1, 8, 64))
```

### 2. 输出示例

```text
== Llama-3-8B on H100 SXM (ridge = 295 FLOP/byte) ==
params total 8.03 B, gemm 7.50 B
weight FLOPs/token   15.01 GFLOPs
attn FLOPs/token@8192 4.29 GFLOPs
weight bytes (BF16)  16.06 GB
KV bytes/token       128 KiB

 batch    ctx    KV read  I(FLOP/B)  t_mem(ms)  t_cmp(ms)  step(ms)    tok/s bound
     1   8192      1.0Gi        1.1       5.11       0.02      5.11      196 memory
     8   8192      8.0Gi        6.3       7.36       0.16      7.36     1087 memory
    64   8192     64.0Gi       14.6      25.31       1.25     25.31     2529 memory
   295   8192    295.0Gi       17.1      99.35       5.76     99.35     2969 memory

prefill   8192 causal=False    158.14 TFLOP  @60% MFU 0.266 s
prefill   8192 causal=True     140.55 TFLOP  @60% MFU 0.237 s
prefill 131072 causal=False     10.97 PFLOP  @60% MFU 18.494 s
prefill 131072 causal=True       6.47 PFLOP  @60% MFU 10.905 s

== Llama-3-70B on H100 SXM (ridge = 295 FLOP/byte) ==
params total 70.55 B, gemm 69.50 B
weight FLOPs/token   139.00 GFLOPs
attn FLOPs/token@8192 21.47 GFLOPs
weight bytes (BF16)  141.11 GB
KV bytes/token       320 KiB

 batch    ctx    KV read  I(FLOP/B)  t_mem(ms)  t_cmp(ms)  step(ms)    tok/s bound
     1   8192      2.5Gi        1.1      42.92       0.16     42.92       23 memory
     8   8192     20.0Gi        7.9      48.53       1.30     48.53      165 memory
    64   8192    160.0Gi       32.8      93.40      10.38     93.40      685 memory

prefill   8192 causal=False      1.31 PFLOP  @60% MFU 2.215 s
prefill   8192 causal=True       1.23 PFLOP  @60% MFU 2.067 s
prefill 131072 causal=False     63.26 PFLOP  @60% MFU 106.598 s
prefill 131072 causal=True      40.74 PFLOP  @60% MFU 68.651 s
```

几处对照：$$B = 1$$、8K 时一步 5.11 ms（4.8 ms 权重 + 0.3 ms 的 1 GiB KV），196 token/s；$$B = 64$$ 时 25.3 ms、有效强度 14.6；$$B = 295$$ 时 KV 读取 295 GiB，脚本没有检查容量，这一行在单卡上是不存在的配置——加一行 `assert weight_bytes + kv_bytes <= gpu.hbm_bytes` 就能让脚本自己拒绝它。70B 的表说明它在单卡上放不下（141 GB 权重），表里的时间是"假如放得下"的下界。

### 3. 与实测对照的方法与预期差距

验证的方法是用 vLLM 或 `transformers` 在一张 H100 上跑 Llama-3-8B BF16，固定 batch 与上下文长度，测 decode 每步时间（或 token 间延迟）与 prefill 时间，再与脚本的下界相除。这里不给出实测数字——它们随驱动、CUDA 版本、引擎版本、功耗设置而变，读者应当自己测——但可以预告差距的来源和量级：

- **decode**：好的引擎能做到下界的 1.2–1.5 倍。差距来自：非 GEMM 算子（RMSNorm、RoPE、残差、采样）各自是一次访存往返，虽然字节数小但 kernel launch 与同步有固定开销，$$B$$ 小时每层十几个 kernel 的 launch 时间就能与 GEMV 本身相当；GEMV 类 kernel 达不到 100% 的带宽利用（通常 80–90%）；paged KV cache 的间接寻址；调度器在两步之间做的 CPU 工作（vLLM 用 CUDA graph 与异步调度压缩这部分）。
- **prefill**：差距来自 GEMM 效率（大形状 GEMM 约 70–80% 峰值）、attention kernel 的效率（FlashAttention 在 H100 上通常低于 GEMM 的效率）、非 GEMM 算子的带宽时间——这些在 prefill 里读写 $$[s, d_{ff}]$$ 的激活，$$s = 8192$$ 时每个算子几百 MB，加起来不再可忽略。把 MFU 从 60% 调到 40% 再看，往往更接近。
- **通用**：脚本假设权重与 KV 各读一遍，没有算激活写回、没有算 lm_head 输出的 $$[B, V]$$ logits（每 token 256 KB，BF16）、没有算采样。

如果实测与下界差距超过 2 倍，通常不是"硬件就这样"，而是某处有可以修的低效：没开 CUDA graph、KV cache 碎片、batch 没有真正合并、或者某个算子回落到了非融合实现。Roofline 的价值就在于给出"应该多快"的参照，让"慢"变成一个可以定位的问题。


## 九、小结

这一篇建立了本系列的成本模型的第二半。核心链条是：

- 每个参数每 token 2 FLOPs，前向约 $$2N$$；embedding 查表不算，lm_head 要算；
- attention 上下文项每层每 token $$4ds$$，随 $$s$$ 线性、prefill 总量随 $$s^2$$，128K 时超过权重项数倍；
- 训练 $$6ND$$（前向 2N、反向 4N），重算 $$8N$$；Llama-3-8B 训 15T token 约 50 万 H100 GPU-小时（40% MFU）；
- decode 每步读一遍权重（16.06 GB）加全部 KV cache（$$B \cdot s \cdot 128$$ KiB），激活可忽略；
- Roofline：ridge point $$P_{peak} / BW$$，H100 295、A100 156；BF16 decode 权重 GEMM 的算术强度 $$= B$$，KV 读取的强度 $$= g$$；
- 单卡 Llama-3-8B：decode 下界 4.8 ms / 208 token/s；要 compute-bound 需 $$B \approx 295$$，但 8K 上下文下 KV cache 需要 295 GiB，且总强度趋于 18，因此不可达；64 GB 预算下 $$B \times s \le 52$$ 万 token；
- prefill 8K 约 140–158 TFLOP，60% MFU 下 0.24–0.27 s，是 TTFT 的下限；
- 训练激活 $$sbh(34 + 5as/h)$$，$$s = 8192$$ 时每层 11 GiB，其中 10 GiB 是 $$s^2$$ 项，FlashAttention 与选择性重算把它去掉；
- MFU $$= \text{tokens/s} \times 6N / P_{peak}$$，8B 模型每卡 8200 token/s 即 40%，40–50% 已是好成绩。

三个模型的数字汇总（BF16，H100 SXM 单卡，理论下界）：

```text
                           Llama-3-8B      Llama-3-70B     DeepSeek-V3
参数量 N                    8.03 B          70.55 B         671 B（激活 37 B）
权重 FLOPs/token (2N)       15.0 GFLOPs     141 GFLOPs      74 GFLOPs（按激活参数）
attention 项 @8K (4dsL)     4.29 GFLOPs     21.5 GFLOPs     约 14 GFLOPs（MLA 形态见第三篇）
attention 项 @128K          68.7 GFLOPs     344 GFLOPs      约 229 GFLOPs（同上）
权重字节                    16.06 GB        141 GB          1342 GB / FP8 671 GB
KV bytes/token              128 KiB         320 KiB         68.6 KiB（MLA）
decode 下界 (B=1, 读权重)   4.8 ms          42 ms（需 2+ 卡）  671 GB / (n 卡 x 3.35 TB/s)
单流上限                    208 token/s     24 token/s      —
compute-bound 所需 B        约 295          约 295          约 295（FP8 约 590 的 ridge, I=2B）
prefill 8K @60% MFU         0.24–0.27 s     2.1–2.2 s       —
训练 6ND @15T token         7.2e23 FLOP     6.3e24 FLOP     —
                            约 51 万 GPU-h  约 450 万 GPU-h
```

下一篇进入第一个**结构**上的改动。本篇把 KV cache 当作一个给定的数字（128 KiB/token）使用；它为什么是这个数、MHA、GQA、MQA、MLA 各自如何改变它、代价是什么，是下一篇的内容：

> **DeepSeek-V3 的 MLA 如何把每 token 的 KV cache 从 3.81 MiB 压到 68.6 KiB，而 attention 的算量与 GQA 相比又变成了什么？**


## 下一篇

[Attention 变体与 KV cache](/attention-variants-and-kv-cache.html)
