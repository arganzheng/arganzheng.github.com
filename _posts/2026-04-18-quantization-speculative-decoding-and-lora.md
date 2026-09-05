---
layout: post
title: "Transformer 与 LLM（07）：量化、投机解码与 LoRA"
subtitle: "Quantization, Speculative Decoding and LoRA: Three Ways to Reshape the Computation"
tags: [Transformer, LLM, AI, AI-Infra]
catalog: true
---

> 本文是[《Transformer 与 LLM：结构、算量与数值》](/transformer-and-llm-for-infra-engineers.html)系列的第七篇（共七篇）。上一篇：[浮点格式、数值稳定性与混合精度](/floating-point-formats-and-mixed-precision.html)

前六篇把一个 Transformer 拆成了四组变量：参数量 $$N$$、每 token 的 FLOPs、每步要搬的字节数、每 token 的 KV cache。这些变量由结构决定——层数、hidden、GQA 的组数、专家数——一旦 `config.json` 定下来，它们就定下来了。

最后一篇讲的是三种**不改结构、只改计算形态**的方法。它们分别攻击前面算出的三个成本项：

- **量化**减少权重（和 KV cache）的字节数，攻击的是 decode 每步要搬的 16 GB；
- **投机解码**用一次前向验证多个 token，攻击的是 decode 每步只产出一个 token 的串行形态；
- **LoRA** 把可训练参数从 $$N$$ 降到 $$N$$ 的千分之几，攻击的是训练状态每参数 16 字节的显存。

三者有一个共同点：收益都不是无条件的。同一个 INT4 模型，decode 快 3 倍，prefill 反而更慢；同一套投机解码，batch 1 时加速 2 倍，batch 64 时没有收益。本篇要回答的核心问题是：

> **这两个看似矛盾的现象，为什么背后是同一条 Roofline？**

先把这条 Roofline 重新写出来。


## 一、起点：decode 是 memory-bound 的

### 1. 算术强度与 ridge point

对一个 $$[m, k] \times [k, n]$$ 的矩阵乘，FLOPs 是 $$2mkn$$；如果权重 $$[k, n]$$ 以 BF16 存放，读一遍要 $$2kn$$ 字节。decode 阶段每步每个请求只有一个 token，$$m$$ 就是 batch 大小 $$B$$，于是权重 GEMM 的算术强度

$$
I = \frac{2Bkn}{2kn} = B \ \text{FLOP/byte}
$$

H100 SXM 的 BF16 dense 算力 989 TFLOPS，HBM 带宽 3.35 TB/s，ridge point

$$
\frac{989 \times 10^{12}}{3.35 \times 10^{12}} \approx 295 \ \text{FLOP/byte}
$$

也就是说 batch 不到大约 300，decode 的权重 GEMM 就在 Roofline 的斜线上——时间由字节数决定，与 FLOPs 无关。$$B = 1$$ 时距离 ridge 两个数量级。

### 2. 一个时间模型

本篇后面所有的估算都用同一个模型：一次前向处理 $$m$$ 个 token 行，时间下界是访存时间与计算时间的较大者

$$
T(m) = \max\left( \frac{W_{\text{bytes}}}{BW},\ \frac{2 N m}{F} \right)
$$

其中 $$W_{\text{bytes}}$$ 是要读的权重字节数，$$N$$ 是参与 GEMM 的参数量（embedding 是查表，不算），$$F$$ 是算力。代入 Llama-3-8B（8.03B 参数，BF16 权重 16.06 GB，每 token 权重 FLOPs 约 15.0 GFLOPs）：

$$
T_{\text{mem}} = \frac{16.06 \ \text{GB}}{3.35 \ \text{TB/s}} \approx 4.8 \ \text{ms}, \qquad
T_{\text{cmp}}(m) = m \times \frac{15.0 \ \text{GFLOPs}}{989 \ \text{TFLOPS}} \approx m \times 15.2 \ \mu\text{s}
$$

两者相等在 $$m \approx 316$$，与 ridge 295 同一量级（差别来自 16.06 GB 包含了 embedding 而 15.0 GFLOPs 不含）。这个模型忽略了 KV cache 读取、activations 与 kernel 效率，是**理论下界**，不是任何实现的实测。

这三个数字——**算术强度 ≈ B、ridge ≈ 295、8B 模型 decode 下界 4.8 ms（约 208 token/s 单请求上限）**——是本篇全部推导的基础。量化改的是 $$W_{\text{bytes}}$$，投机解码改的是 $$m$$，LoRA 改的是训练时的 $$N$$。


## 二、量化：改变 W_bytes

### 1. 基本形式与粒度

均匀量化把一个浮点权重 $$w$$ 映射到 $$b$$ 位整数再映射回来：

$$
q = \text{clamp}\left( \text{round}\left(\frac{w}{s}\right) + z,\ 0,\ 2^b - 1 \right), \qquad
\hat{w} = s \cdot (q - z)
$$

常写成 $$\hat{w} = s \cdot \text{round}(w/s) + z$$ 的形式。$$s$$ 是 scale，$$z$$ 是 zero point。**对称量化** $$z = 0$$，$$s = \max\lvert w\rvert  / (2^{b-1} - 1)$$，整数范围以 0 为中心；**非对称量化** $$z \neq 0$$，$$s = (\max w - \min w) / (2^b - 1)$$，把实际的 $$[\min, \max]$$ 区间完整映射到 $$[0, 2^b - 1]$$，对分布偏斜的权重多用掉一个 bit 的表达力。

$$s$$ 与 $$z$$ 按什么范围共享，决定了量化的**粒度**：

- **per-tensor**：整个矩阵一个 $$s$$。元数据可忽略，但一个离群值会把所有权重的步长拉大。
- **per-channel**：每个输出通道（矩阵的一列/一行）一个 $$s$$。$$W_Q$$ 有 4096 个 scale，元数据仍可忽略。这是 INT8 权重量化的常态。
- **per-group**：沿输入维度每 $$g$$ 个权重一个 $$s$$（和 $$z$$），$$g = 128$$ 是 4-bit 的事实标准。粒度越细，每个 group 的范围越紧，量化误差越小，但元数据线性增长。

per-group 的元数据开销直接算：每个权重 $$b$$ 位，每 $$g$$ 个权重摊一个 scale 和一个 zero，

$$
\text{bit/weight} = b + \frac{b_{\text{scale}} + b_{\text{zero}}}{g}
$$

INT4、$$g = 128$$、FP16 scale + 4-bit zero：$$4 + 20/128 = 4.16$$ bit；FP16 scale + FP16 zero：$$4 + 32/128 = 4.25$$ bit；$$g = 64$$ 时 $$4 + 32/64 = 4.5$$ bit。所以常说的"INT4 模型"实际是 **4.25–4.5 bit/权重**。

代入 Llama-3-70B（70.55B 参数）：

$$
70.55 \times 10^9 \times \frac{4.25}{8} \approx 37.5 \ \text{GB}
$$

如果 embedding 与 lm_head（合计 2.10B）保留 BF16，则 $$68.45 \times 0.531 + 2.10 \times 2 \approx 40.6$$ GB。**约 37–40 GB，能放进一张 80 GB 的 H100**，剩下约 40 GB 给 KV cache（70B 每 token 320 KiB，约 13 万 token）。BF16 的 141 GB 需要至少两张卡——这是 weight-only 量化最直接、也最不需要 Roofline 就能理解的收益：把一个模型从两卡变成一卡。

同样算 Llama-3-8B：$$8.03 \times 4.25 / 8 \approx 4.27$$ GB。

哪些部分通常不量化，也值得用参数量的分布来理解。embedding 是查表，量化它省的是显存而不是 decode 带宽（每步只读一行）；lm_head 在 decode 每步被完整读一遍（1.05 GB），量化它对时间有实际帮助，但它直接决定输出分布，对误差最敏感，多数方案保留 FP16；attention 与 FFN 的 6.98B 是量化的主体。评价量化质量的常用指标是校准集之外文本上的困惑度（perplexity）相对 FP16 的增量，以及下游任务的分数——两者不总是一致，INT4 上"困惑度只涨 0.1"与"某个推理任务掉 3 分"可以同时发生，这是选方案时要一起看的。

### 2. W4A16 的收益区间：decode 与 prefill 符号相反

W4A16 指权重 4 位、activations 保持 16 位（BF16/FP16）。计算时 kernel 从显存读 INT4 权重，在片上反量化成 BF16，再与 BF16 的 activations 在 BF16 Tensor Core 上做乘加。**FLOPs 一个都没少，少的只是权重字节数。**

回到时间模型。decode $$B = 1$$：

$$
T_{\text{mem}}^{\text{INT4}} = \frac{4.0 \sim 4.3 \ \text{GB}}{3.35 \ \text{TB/s}} \approx 1.2 \sim 1.3 \ \text{ms}
$$

从 4.8 ms 降到 1.2 ms（纯 4 bit）或 1.27 ms（4.25 bit）。这个下界成立的前提是：反量化开销可忽略（它在 SM 内部做，不碰 HBM）、activations 仍是 BF16（$$B = 1$$ 时 activations 只有几十 KB，确实可忽略）、KV cache 读取不计。

prefill 是另一回事。8K prompt 的 prefill 处理 $$m = 8192$$ 行，算术强度约 8192 FLOP/byte，远在 ridge 295 之上，时间由 FLOPs 决定：

$$
T_{\text{cmp}} = \frac{8192 \times 15.0 \ \text{GFLOPs}}{989 \ \text{TFLOPS}} \approx 0.12 \ \text{s} \ (\text{100\% MFU})
$$

FLOPs 没变，Tensor Core 还是 BF16 的，权重那 16 GB 本来在 0.12 s 里只占 4.8 ms，省掉 3/4 也只是省了 3.6 ms。而反量化是纯增加的工作：每个权重元素在被用于 8192 行的乘加之前，要先做一次移位、掩码、乘 scale、加 zero，且 W4A16 kernel 在大 $$m$$ 下通常达不到 cuBLAS BF16 GEMM 的效率。结果是 **prefill 可能比 BF16 更慢**。

把反量化的成本放进模型里看得更清楚。一个 W4A16 GEMM kernel 对每个权重元素做的额外工作是常数（几条整数与浮点指令），总量与 $$kn$$ 成正比、与 $$m$$ 无关；而有用的乘加与 $$mkn$$ 成正比。$$m = 1$$ 时两者同量级，但此时 kernel 在等 HBM，反量化藏在访存延迟后面；$$m = 8192$$ 时有用计算是反量化的 8192 倍，反量化本身可忽略，但它占用的寄存器与指令槽、以及为了容纳 INT4 布局而偏离 cuBLAS 最优 tile 形状的代价，让 kernel 的 MFU 低于纯 BF16 GEMM。所以"prefill 更慢"的幅度不是理论能算出来的，取决于 kernel 实现；理论能说的是它**不可能更快**。

同一个 4.25 bit 的权重文件，decode 快约 4 倍（理论），prefill 慢——不是两种现象，是 Roofline 上两个位置：一个在斜线上（字节数决定时间，字节减少直接兑现），一个在平台上（FLOPs 决定时间，字节减少不兑现，反量化的额外指令反而算进去）。

### 3. 从 batch 1 到 256：交叉点在 B ≈ ridge/4

把 batch 扫一遍，用同一个时间模型算 BF16 与 W4A16 各自的下界（Llama-3-8B，H100，纯 4 bit 4.0 GB，忽略反量化开销）：

```text
batch B    BF16 下界 (ms)    W4A16 下界 (ms)    BF16/W4A16
   1          4.79              1.20              4.0
   8          4.79              1.20              4.0
  32          4.79              1.20              4.0
  64          4.79              1.20              4.0
 128          4.79              1.94              2.5
 256          4.79              3.89              1.2
 512          7.77              7.77              1.0（实际 W4A16 更慢）
```

BF16 在 $$B < 316$$ 时一直被 4.8 ms 的访存时间钉住；W4A16 的访存时间是 1.2 ms，计算时间 $$B \times 15.2 \ \mu s$$ 追上它的位置是

$$
B^* = \frac{1.2 \ \text{ms}}{15.2 \ \mu\text{s}} \approx 79 \approx \frac{\text{ridge}}{4}
$$

一般地，权重压缩 $$k$$ 倍，W4A16 的"转折 batch"就是 $$\text{ridge}/k$$。超过 $$B^*$$ 之后 W4A16 转为 compute-bound，收益按 $$1/B$$ 衰减，到 $$B \approx \text{ridge}$$ 处两者相遇，再往上反量化开销让 W4A16 落到 BF16 下方。

这张表回答了"什么时候该用 W4A16"：**单请求或小 batch 的延迟敏感服务**（本地部署、交互式应用），以及**为了放进一张卡**。高吞吐、大 batch 的服务里，它对 decode 的帮助随 batch 变小，对 prefill 是负的。

实际测到的 decode 加速通常在 3 倍左右而非 4 倍，差在几处：lm_head 常保留 FP16（1.05 GB，占 4.27 GB 的四分之一）、KV cache 读取不随权重量化减少、group 元数据、kernel 效率。这些都能用上面的模型逐项归因。

### 4. GPTQ：用 Hessian 把误差补偿到未量化的列

上面只讲了格式，没讲怎么选 $$s$$ 与 $$\hat{w}$$ 使模型精度损失最小。最简单的 round-to-nearest（RTN）在 INT8 per-channel 下够用，到 INT4 就不够了。

GPTQ（Frantar 等 2022）继承 OBQ（Optimal Brain Quantization）的思路：**逐个量化权重，每量化一个，就调整剩下未量化的权重去补偿它带来的输出误差**。对一个线性层的一行权重 $$w \in \mathbb{R}^{d_{in}}$$，让量化后的输出尽量接近原输出：

$$
\min_{\hat{w}} \ \| w X - \hat{w} X \|_2^2 = (w - \hat{w}) H (w - \hat{w})^\top, \qquad H = 2 X X^\top
$$

$$H \in \mathbb{R}^{d_{in} \times d_{in}}$$ 是这个二次目标的 Hessian，由该层的输入 $$X$$ 决定，**对矩阵的所有行相同**。OBQ 的结论是：把第 $$q$$ 个权重量化为 $$\text{quant}(w_q)$$ 后，其余权重的最优更新是

$$
\delta = -\frac{w_q - \text{quant}(w_q)}{[H^{-1}]_{qq}} \cdot H^{-1}_{:, q}
$$

即量化误差按 $$H^{-1}$$ 第 $$q$$ 列的比例分摊到其他权重上，然后从 $$H^{-1}$$ 中删去第 $$q$$ 行列继续。GPTQ 做了三处工程改造使它能跑到百亿参数：

1. **固定列顺序**：OBQ 每步挑误差最小的权重，各行顺序不同；GPTQ 让所有行按同一列顺序量化，于是 $$H^{-1}$$ 的更新对所有行共享，一列一列推进，每列是一次矩阵向量操作；
2. **lazy batch**：每 128 列为一块，块内更新只作用在块内，块结束时再一次性更新块外的列，减少对 $$d_{out} \times d_{in}$$ 大矩阵的反复读写；
3. **Cholesky**：预先算 $$H^{-1}$$ 的 Cholesky 分解，逐列取用，避免反复求逆时的数值问题。

$$H$$ 从哪里来？从**校准数据**：通常约 128 条、每条 2048 token 的文本，跑一遍前向，在每层收集输入 $$X$$ 累加 $$X X^\top$$。$$W_Q$$ 的 $$H$$ 是 $$4096^2$$ 个 FP32，64 MB；down_proj 的 $$H$$ 是 $$14336^2$$，约 820 MB。整个过程逐层顺序进行，175B 模型在单张 A100 上约几个 GPU 小时。代价是需要校准数据、有过拟合校准集分布的风险、以及对某些层需要"按激活大小排序列顺序"（act-order）的启发式。GPTQ 产出的是标准的 INT4 per-group 权重，推理 kernel 与格式无关。

### 5. AWQ：保护 1% 的显著通道

AWQ（Lin 等 2023）从另一个观察出发：**权重不是同等重要的**。把对应于激活幅度最大的约 1% 输入通道的权重保留 FP16、其余 RTN 到 INT4，困惑度几乎恢复到 FP16 水平；而如果按权重自身幅度挑这 1%，效果远差。也就是说，重要的是 $$\lvert x_j\rvert $$ 大的那些输入通道 $$j$$ 所对应的权重列 $$W_{:, j}$$。

混合精度的权重矩阵对 kernel 不友好。AWQ 改为**逐输入通道缩放**：

$$
Y = W X = (W \cdot \text{diag}(s)) \cdot (\text{diag}(s)^{-1} X)
$$

数学上恒等；量化 $$W \cdot \text{diag}(s)$$ 而不是 $$W$$。对通道 $$j$$ 乘 $$s_j > 1$$，这一列的权重变大，在 group 内占据更多整数级别，量化相对误差约降为 $$1/s_j$$；只要 $$s_j$$ 不大到改变 group 的最大值（少数通道乘 2 通常不会），其他通道的误差不变。$$s_j$$ 按激活统计量搜索：

$$
s_j = \left( \text{mean}|X_j| \right)^{\alpha}, \qquad \alpha \in [0, 1]
$$

在校准集上网格搜索 $$\alpha$$ 使输出误差最小。$$\text{diag}(s)^{-1}$$ 一侧折进前一个算子（RMSNorm 的 $$\gamma$$，或前一个线性层的输出通道），运行时零开销。

与 GPTQ 相比，AWQ 只需要前向收集激活统计与做一次网格搜索，**不需要反向、不需要 Hessian**，校准数据更少也更不容易过拟合。两者产出格式相同，vLLM 中的 W4A16 GEMM kernel（如 Marlin）对两者通用。

### 6. SmoothQuant 与 LLM.int8()：激活的离群通道

到此为止量化的都是权重。要让 GEMM 本身在 INT8 或 FP8 Tensor Core 上跑（W8A8），activations 也得量化——这是完全不同难度的问题。

从大约 6.7B 参数起，LLM 的 activations 在**特定的 hidden 维度**上系统性地出现 20–100 倍于其他维度的值，而且这些维度在不同 token、不同输入上是固定的。per-tensor INT8：$$s = \max\lvert X\rvert  / 127$$ 被离群通道决定，其他通道的值落在 1–2 个整数级别上，信息几乎全丢。per-channel 按 hidden 维度给 activations 不同 scale 也不行——这个维度是 GEMM 的归约维 $$k$$，scale 不能从累加中提出来。per-token（按行）可以，但对离群通道无效，因为每一行都有它们。

SmoothQuant（Xiao 等 2022）把难度**从激活迁移到权重**：

$$
Y = X W = (X \cdot \text{diag}(s)^{-1}) (\text{diag}(s) \cdot W), \qquad
s_j = \frac{\max|X_j|^{\alpha}}{\max|W_j|^{1 - \alpha}}
$$

激活的通道 $$j$$ 除以 $$s_j$$，权重的对应行乘 $$s_j$$。$$\alpha = 0.5$$ 时 $$\max\lvert \hat{X}_j\rvert  = \max\lvert \hat{W}_j\rvert  = \sqrt{\max\lvert X_j\rvert  \cdot \max\lvert W_j\rvert }$$，两边的每通道范围被拉平到几何平均——激活的离群被压下去，权重的对应行被抬上来，两者都变得"可量化"。激活离群越严重的模型 $$\alpha$$ 越大（有的模型用 0.75）。$$\max\lvert X_j\rvert $$ 需要校准数据统计，是静态的；$$\text{diag}(s)^{-1}$$ 同样折进前面的 RMSNorm。

它的形式与 AWQ 惊人地相似——同一个 $$\text{diag}(s)$$ 分解——但方向相反：AWQ 把权重乘大保护权重，SmoothQuant 把激活除小保护激活。迁移的难度在于：权重被抬高后自身的量化也变难，$$\alpha$$ 是两边的折中；对离群极端的模型（Llama 系列某些层的 massive activations 可达数千），INT8 per-tensor 静态量化仍会掉点，实践中常退到 per-token 动态量化。

LLM.int8()（Dettmers 等 2022）选择不迁移而是**分离**：把 $$X$$ 中任一元素绝对值超过阈值（论文用 6.0）的列（hidden 维度）单独抽出来与对应权重行做 FP16 GEMM，其余部分做 INT8 按行/按列 vector-wise 量化 GEMM，最后相加。离群维度约占 0.1%，精度保持得很好，但两个 GEMM 加上 gather/scatter 使它在多数情况下比 FP16 慢，主要价值是省显存而不是提速。

### 7. FP8 推理量化：浮点的相对精度 vs 整数的绝对精度

H100 提供 FP8 Tensor Core（E4M3 与 E5M2，dense 1979 TFLOPS，BF16 的两倍），使 W8A8 有了比 INT8 更宽容的载体。原因在格式本身：

- INT8 的步长是**绝对**的：$$s = \max/127$$，所有值的量化误差都是 $$\pm s/2$$。一个 100 倍于典型值的离群值让 $$s$$ 变大 100 倍，典型值被压到 1 个级别附近。
- E4M3 有 3 位尾数，任何正规数的**相对**误差都约 $$2^{-4} = 6\%$$，与该值本身的大小无关；它的正规数范围 $$2^{-6}$$ 到 448，加上次正规数到 $$2^{-9}$$，动态范围约 $$2.3 \times 10^5$$（约 17.8 位），INT8 只有 127（7 位）。

同一个 per-tensor scale 下，离群值把典型值"挤没"的情况在 FP8 里不发生——典型值只是跌到更小的指数段，仍保留 3 位尾数。这就是 FP8 比 INT8 对离群值宽容的全部原因；代价是 3 位尾数的相对精度低于 INT8 在满量程附近的相对精度。

scale 的粒度在 FP8 里同样重要：

- **per-tensor**：一个标量，可静态（校准）或动态（每步算 max）；H100 的 FP8 GEMM 原生只接受这种标量 scale，在 epilogue 乘回去；
- **per-token**：activations 每一行一个 scale，动态计算，对付 token 间幅度差异；
- **per-block**：权重按 $$128 \times 128$$、activations 按 $$1 \times 128$$ 分块各自一个 scale（DeepSeek-V3 的做法），scale 沿归约维变化，不能在 epilogue 一次乘回，必须在累加过程中每 128 个元素乘一次——DeepSeek-V3 在 CUDA core 上做这个提升累加，Tensor Core 的 FP8 累加器只负责 128 元素内。

### 8. W8A8 与 W4A16 各自的位置

现在可以把两类量化放到 Roofline 上：

```text
                 权重字节     GEMM 精度        decode (memory-bound)   prefill (compute-bound)
BF16             1×          BF16 Tensor Core     4.8 ms                 基线
W8A8 (FP8/INT8)  1/2         FP8/INT8 TC 2× 算力  2.4 ms 下界            上限快 2×
W4A16            1/4         BF16 TC + 反量化     1.2 ms 下界            不变或更慢
W4A8             1/4         FP8/INT8 TC          1.2 ms 下界            上限快 2×（反量化到 INT8/FP8）
```

W8A8 让 GEMM 在 FP8 Tensor Core 上跑，ridge 变为 $$1979/3.35 \approx 591$$，prefill 的 FLOPs 上限翻倍；同时权重减半，decode 下界 2.4 ms。8K prefill 的 158 TFLOP 在 60% MFU 下从约 0.27 s 降到约 0.13 s。W4A16 对 prefill 无益，对 decode 的收益是 W8A8 的两倍。两者组合的 W4A8 试图兼得，代价是把 4 位权重反量化到 8 位整数/浮点的精度损失与 kernel 复杂度。

这就是"用哪种量化"的判断依据：**prefill 重（长 prompt、高吞吐）用 W8A8；decode 重（长生成、低延迟、小 batch）用 W4A16；显存装不下先用 W4。**

### 9. KV cache 量化

KV cache 每 token 的字节数是

$$
\text{bytes/token} = 2 \cdot L \cdot n_{kv} \cdot d_{head} \cdot \text{bytes/elem}
$$

Llama-3-8B：$$2 \times 32 \times 8 \times 128 \times 2 = 128$$ KiB。把 K、V 存成 FP8（E4M3，每个 head 或每 token-head 一个 scale）或 INT8，**128 KiB → 64 KiB**；128K 上下文从 16 GiB 变 8 GiB；70B 从 320 KiB 变 160 KiB；DeepSeek-V3 的 MLA 从 68.6 KiB 变 34.3 KiB。

它的意义要和权重一起看。上下文 8K、batch 64 时，Llama-3-8B 每步 decode 要读 $$128 \ \text{KiB} \times 8192 \times 64 = 64$$ GiB 的 KV cache，是权重 16 GB 的四倍。这个区间里，**KV cache 量化对 decode 时间的影响大于权重量化**：权重 INT4 省 12 GB，KV FP8 省 32 GiB。attention 部分的读取在 FP8 KV 下同样减半（attention 与权重 GEMM 不同，读的是 activations 而非权重，但 memory-bound 的性质相同）。vLLM 的 `kv_cache_dtype=fp8` 对应的就是这一项。


## 三、投机解码：改变 m

### 1. 问题：一次前向只产出一个 token

decode 每步读 16 GB 权重、产出 $$B$$ 个 token。$$B = 1$$ 时，4.8 ms 里 Tensor Core 只做了 15 GFLOPs，利用率约 0.3%。Roofline 告诉我们：在斜线上，多算几行几乎不多花时间——$$T(m)$$ 在 $$m < 316$$ 时是常数。如果能一次前向验证多个候选 token，就把串行的产出变成了并行的验证。

问题是候选从哪里来，以及怎么保证结果与原模型**一致**。

### 2. 算法与分布等式

记目标模型的分布为 $$p(\cdot \mid \text{prefix})$$，一个便宜的草稿模型的分布为 $$q(\cdot \mid \text{prefix})$$。投机解码（Leviathan 等 2023；Chen 等 2023）一轮做四件事：

1. 草稿模型自回归采样 $$\gamma$$ 个 token $$x_1, \ldots, x_\gamma$$，$$x_i \sim q(\cdot \mid \text{prefix}, x_{<i})$$；
2. 目标模型对 $$\text{prefix}, x_1, \ldots, x_\gamma$$ 做**一次**前向，得到 $$\gamma + 1$$ 个位置的分布 $$p_1, \ldots, p_{\gamma+1}$$；
3. 从 $$i = 1$$ 起逐个判定：以概率 $$\min(1, p_i(x_i) / q_i(x_i))$$ 接受 $$x_i$$；一旦拒绝，从修正分布 $$\text{norm}(\max(0, p_i - q_i))$$ 采样一个 token 替代 $$x_i$$，本轮结束；
4. 若 $$\gamma$$ 个全部接受，再从 $$p_{\gamma+1}$$ 采样一个 token。

每轮至少产出 1 个 token（拒绝时的重采样或全接受时的额外采样），最多 $$\gamma + 1$$ 个。

关键性质：**输出分布严格等于 $$p$$**。看单步。在某一位置，草稿提出 $$x$$ 的概率是 $$q(x)$$，被接受的概率是 $$\min(1, p(x)/q(x))$$，所以"接受且输出 $$x$$"的概率是

$$
q(x) \min\left(1, \frac{p(x)}{q(x)}\right) = \min(q(x), p(x))
$$

总接受概率

$$
\beta = \sum_x \min(p(x), q(x))
$$

拒绝的概率是 $$1 - \beta$$，拒绝后从 $$\text{norm}(\max(0, p - q))$$ 采到 $$x$$ 的概率是 $$\max(0, p(x) - q(x)) / Z$$，归一化常数

$$
Z = \sum_y \max(0, p(y) - q(y)) = \sum_y \left( p(y) - \min(p(y), q(y)) \right) = 1 - \beta
$$

于是输出 $$x$$ 的总概率

$$
P(x) = \min(p(x), q(x)) + (1 - \beta) \cdot \frac{\max(0, p(x) - q(x))}{1 - \beta} = \min(p(x), q(x)) + \max(0, p(x) - q(x)) = p(x)
$$

每个位置都从 $$p$$ 采样，且被接受的 token 之后的位置以它为条件——与目标模型自回归采样的联合分布逐位相同。greedy 解码是 $$p$$ 退化为 one-hot 的特例：接受当且仅当草稿与目标 argmax 一致。这个证明不依赖 $$q$$ 是什么——$$q$$ 只影响**效率**，不影响**正确性**。

两点补充。第一，拒绝后的重采样分布 $$\text{norm}(\max(0, p - q))$$ 有直观含义：它只在 $$p(x) > q(x)$$ 的 token 上有质量，即"目标模型认为比草稿更可能"的那些 token——草稿高估的 token 已经被接受步骤按 $$p/q$$ 的比例采纳过了，剩下的概率质量正好是目标模型比草稿多出来的部分。第二，验证时目标模型输出的 $$\gamma + 1$$ 个分布只需要一次前向，是因为因果掩码下每个位置的输出只依赖它之前的 token，草稿序列的每个前缀恰好对应一个位置——这与 prefill 一次算出整个 prompt 所有位置的 KV 是同一件事，投机解码的验证本质上是一次长度为 $$\gamma + 1$$ 的小 prefill。被拒绝位置之后的 KV cache 条目要回退丢弃，这是引擎实现中需要处理的细节。

### 3. 期望接受数与加速比

记单个位置的接受率 $$\alpha = \mathbb{E}[\beta]$$（它等于 $$1 - \text{TV}(p, q)$$，$$p$$ 与 $$q$$ 的总变差距离的补）。假设各位置独立且接受率相同，一轮产出的 token 数是"连续接受的个数 + 1"，期望

$$
\mathbb{E}[\text{tokens}] = 1 + \alpha + \alpha^2 + \cdots + \alpha^\gamma = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}
$$

$$\alpha = 0.8$$、$$\gamma = 4$$：$$(1 - 0.8^5)/0.2 = (1 - 0.328)/0.2 = 3.36$$。

一轮的成本：$$\gamma$$ 次草稿前向加一次目标前向。记草稿一次前向的时间是目标前向的 $$c$$ 倍，并且——这是关键假设——目标模型验证 $$\gamma + 1$$ 个 token 的时间与验证 1 个相同。那么

$$
\text{speedup} = \frac{\mathbb{E}[\text{tokens}]}{\gamma c + 1}
$$

$$c = 0.1$$：$$3.36 / 1.4 = 2.4$$。若草稿几乎免费（$$c = 0.02$$，多头或 n-gram 方案），$$3.36 / 1.08 \approx 3.1$$。

几个变体的数字：

```text
alpha   gamma   E[tokens]   speedup(c=0.1)   speedup(c=0.02)
0.6     4       2.31        1.65             2.13
0.8     2       2.44        2.03             2.35
0.8     4       3.36        2.40             3.11
0.8     8       4.33        2.40             3.73
0.9     4       4.10        2.93             3.79
```

$$\gamma$$ 越大，期望接受数增长越慢（$$\alpha^\gamma$$ 衰减），而草稿成本线性增长；$$c = 0.1$$ 时 $$\gamma = 4$$ 与 $$\gamma = 8$$ 的加速比相同，最优 $$\gamma$$ 由 $$\alpha$$ 与 $$c$$ 共同决定。

### 4. 为什么验证 γ+1 个 token 几乎免费——以及何时不再免费

"验证 $$\gamma + 1$$ 个与验证 1 个同样贵"是 Roofline 的直接推论。目标模型的 GEMM 从 $$[B, k] \times [k, n]$$ 变成 $$[B(\gamma + 1), k] \times [k, n]$$：FLOPs 乘 $$\gamma + 1$$，**权重读取不变**。只要 $$B(\gamma + 1)$$ 仍在 ridge 之下，

$$
T(B(\gamma + 1)) = T(B) = \frac{W_{\text{bytes}}}{BW}
$$

多出的 FLOPs 填的是本来空转的 Tensor Core。KV cache 读取也一样：验证 5 个 token 的 attention 读同一份 KV cache。

条件是 $$B(\gamma + 1) \lesssim \text{ridge}$$，即

$$
B \lesssim \frac{\text{ridge}}{\gamma + 1} \approx \frac{300}{5} = 60
$$

超过这个 batch，验证前向进入 compute-bound，$$T(B(\gamma+1))$$ 开始以 $$(\gamma + 1)$$ 倍于 $$T(B)$$ 的斜率增长。极限情况（完全 compute-bound）加速比变为

$$
\frac{\mathbb{E}[\text{tokens}]}{\gamma c + (\gamma + 1)} = \frac{3.36}{0.4 + 5} \approx 0.62
$$

**低于 1**。原因很朴素：投机解码是用 FLOPs 换延迟——每产出 3.36 个 token 要为 5 个位置做完整前向，FLOPs 效率是 $$3.36/5 = 67\%$$，被拒绝的 token 的计算是白做的。当 FLOPs 是瓶颈时，这笔交易亏本。

用第一章的时间模型算 Llama-3-8B 在 H100 上各 batch 的加速比（$$\alpha = 0.8$$、$$\gamma = 4$$、$$c = 0.1$$，草稿成本按 $$c \cdot T(B)$$ 计）：

$$
\text{speedup}(B) = \frac{\mathbb{E}[\text{tokens}] \cdot T(B)}{\gamma c \cdot T(B) + T(B(\gamma + 1))}
$$

```text
batch B    T(B) ms    T(5B) ms    加速比（峰值算力）   加速比（60% MFU）
   1        4.79       4.79          2.40                2.40
   8        4.79       4.79          2.40                2.40
  32        4.79       4.79          2.40                2.40
  64        4.79       4.85          2.38                1.61
 128        4.79       9.71          1.39                0.89
 256        4.79      19.4           0.76                0.62
```

第二列按 60% MFU 折算实际可达算力（593 TFLOPS，有效 ridge 约 177，转折 batch 约 35）：$$B = 64$$ 时验证前向已进入计算区，加速比掉到 1.6；$$B = 128$$ 时低于 1。再算上真实系统里草稿模型在大 batch 下的开销、每轮调度与采样的固定成本、以及 $$\alpha$$ 在不同位置并不独立同分布，**"batch 64 时没有收益"** 是这条曲线的工程表述——精确的转折位置随模型、硬件、$$\gamma$$ 移动，但它的量级由 $$\text{ridge}/(\gamma + 1)$$ 决定。

于是核心问题的两半合上了：INT4 在 $$B < \text{ridge}/4$$ 时兑现字节收益，投机解码在 $$B < \text{ridge}/(\gamma+1)$$ 时兑现并行验证的收益——**两者都只在 Roofline 的斜线上有效，越过 ridge 就消失甚至反转**。它们优化的是同一个量：memory-bound 区间里被浪费的算力。这也意味着两者可以叠加：W4A16 的目标模型验证 5 个 token 同样几乎免费，只是转折 batch 变成 $$\text{ridge}/(4 \times 5) \approx 15$$。

### 5. 草稿从哪里来

草稿方案决定 $$\alpha$$ 与 $$c$$。以下区间是各论文与工程报告中**通常报告**的量级，不是本文实测：

```text
方案                        草稿形态                          c（相对目标一次前向）     alpha（通常报告）
独立小模型                  同 tokenizer 的小模型自回归 γ 步     参数量之比，~0.05–0.15    0.6–0.8（取决于配对）
Medusa（Cai 等 2024）       目标模型顶层加 K 个头并行预测 t+2..    ≈0（一次前向内）         第 1 头 ~0.6–0.7，逐头下降
EAGLE（Li 等 2024）         一层 Transformer 在特征级自回归起草   ~0.02–0.05               ~0.75–0.85（论文报告）
n-gram / prompt lookup      在上下文中查找 n-gram 复制后续 token   ≈0                       任务依赖：改写/摘要/RAG 高，自由生成低
DeepSeek-V3 MTP             训练时联合训练的一个额外 block         1/61 层的量级            第二 token 85–90%（技术报告）
```

- **独立小模型**要求与目标共享 tokenizer（Llama-3-8B 给 70B 起草），$$c$$ 约等于参数量之比，在 memory-bound 区间也等于字节数之比。$$\alpha$$ 取决于两者分布的接近程度，同系列同数据训练的模型配对最好。
- **Medusa** 在目标模型最后一层 hidden state 上接 $$K$$ 个轻量头，第 $$k$$ 个头预测第 $$t + k + 1$$ 个 token，一次前向同时出所有草稿；用 tree attention 一次验证多条候选路径。$$c \approx 0$$，但各头独立预测（没有以前一个草稿为条件），$$\alpha$$ 随头序号下降。论文报告 Medusa-1 约 2.2×、Medusa-2 约 2.3–3.6×。
- **EAGLE** 的观察是：在特征（倒数第二层的 hidden state）而非 token 层面做自回归，不确定性更低；草稿模块只有一层 decoder，输入是目标模型的特征与已采样 token 的 embedding。论文报告 LLaMA2-Chat 70B 上约 2.7–3.5×，EAGLE-2 用动态草稿树进一步提高。它的 $$\alpha$$ 通常高于 Medusa，$$c$$ 是一层对全模型的比例。
- **n-gram / prompt lookup**：把上下文里最近出现的 n-gram 后面接的 token 当草稿，零成本、零训练，在有大量复制的任务（改写、摘要、代码编辑、RAG）上 $$\alpha$$ 很高，在自由生成上接近 0——加速比完全依赖任务。
- **DeepSeek-V3 的 MTP**：训练时就带一个预测下一下个 token 的额外模块，推理时可当草稿用（$$\gamma = 1$$）。技术报告称第二个 token 的接受率在 85–90% 之间，对应 $$\mathbb{E}[\text{tokens}] = 1 + \alpha \approx 1.85 \sim 1.9$$，报告的解码吞吐提升约 1.8×，与 $$c$$ 很小时 $$1.9 / (c + 1)$$ 的估算一致。

所有这些方案共享同一条约束：它们提升的是 $$\alpha$$ 或降低 $$c$$，但都改不了 $$B \lesssim \text{ridge}/(\gamma + 1)$$ 这个收益区间。


## 四、LoRA：改变训练时的 N

### 1. 形式与参数量

全量微调的代价不在 FLOPs 而在**状态**。BF16 混合精度 + Adam 每参数 16 字节（BF16 权重 2 + BF16 梯度 2 + FP32 主权重 4 + Adam 一阶、二阶矩各 4），Llama-3-8B 的训练状态 $$8.03 \times 16 = 128$$ GB，一张 80 GB 的卡放不下，还没算激活值。

LoRA（Hu 等 2021）冻结 $$W \in \mathbb{R}^{d_{out} \times d_{in}}$$，只训练一个低秩增量：

$$
W' = W + \frac{\alpha}{r} B A, \qquad A \in \mathbb{R}^{r \times d_{in}},\ B \in \mathbb{R}^{d_{out} \times r}
$$

$$A$$ 高斯随机初始化，$$B$$ 初始化为零——于是训练开始时 $$BA = 0$$，模型与原模型完全一致，梯度从 $$B$$ 开始流动。$$\alpha / r$$ 是一个缩放常数，让换 $$r$$ 时不必重调学习率。可训练参数 $$r(d_{in} + d_{out})$$，对 $$r \ll \min(d_{in}, d_{out})$$ 远小于 $$d_{in} d_{out}$$。

代入 Llama-3-8B，$$r = 16$$，只加在 attention 四个矩阵上（$$d_{in} = 4096$$；$$W_Q$$、$$W_O$$ 的 $$d_{out} = 4096$$，$$W_K$$、$$W_V$$ 的 $$d_{out} = 8 \times 128 = 1024$$）：

$$
16 \times (8192 + 5120 + 5120 + 8192) = 425{,}984 \ \text{每层}, \qquad \times 32 = 13.63\text{M} \ (0.17\%)
$$

再加 FFN 的 gate/up（$$4096 \to 14336$$）与 down（$$14336 \to 4096$$）：

$$
3 \times 16 \times (4096 + 14336) = 884{,}736 \ \text{每层}, \qquad \times 32 = 28.3\text{M}
$$

合计 41.9M，占 8.03B 的 **0.52%**。Llama-3-70B 同样配置：attention 65.5M（0.093%），全部七个矩阵 207M（0.29%）——模型越大比例越小，因为 LoRA 参数随 $$d$$ 线性增长而 $$W$$ 随 $$d^2$$。

$$r$$ 与作用范围是两个独立的旋钮，参数量对两者都是线性的：$$r = 64$$ 只加 attention 是 54.5M，$$r = 16$$ 加全部七个矩阵是 41.9M，两者相近。LoRA 原论文的实验与后续经验都倾向于后者——**以小 $$r$$ 覆盖更多矩阵，比以大 $$r$$ 只覆盖 attention 更有效**，因为增量的"秩"很低这一假设对每个矩阵都成立，而覆盖 FFN 让适配能触及模型三分之二以上的参数所在。就成本而言两种选择没有区别，都是 0.5% 量级。

### 2. 训练状态：从 128 GB 到 16.7 GB，但激活值不变

LoRA 训练的显存：

```text
                        全量微调                 LoRA r=16（全部七个矩阵）
冻结 / 可训练权重        8.03B × 16 B = 128 GB    BF16 冻结权重 8.03B × 2 B = 16.06 GB
                                                 + LoRA 状态 41.9M × 16 B ≈ 0.67 GB
激活值                   与序列长度成正比           基本相同
```

冻结权重只需要 BF16 一份，没有梯度、没有主权重、没有 Adam 状态；LoRA 参数的 16 字节/参数只作用在 41.9M 上。**权重侧从 128 GB 降到 16.7 GB**，这是 LoRA 能在单卡上微调 8B、在 8 卡上微调 70B 的原因。

但激活值不变。反向传播要经过每一层算 $$\partial L / \partial x$$，这需要每层保存的中间量（RMSNorm 输入、attention 的 softmax 统计、FFN 的 SiLU 输入等），与权重是否冻结无关；$$A$$ 的梯度 $$\partial L / \partial A = B^\top (\partial L / \partial y) x^\top$$ 同样需要保留输入 $$x$$。按 Korthikanti 等 2022 的估算，用 FlashAttention 后每层每 token 约 $$34 \cdot d$$ 字节，Llama-3-8B 每 token 32 层约 4.5 MB，4096 token 的一条序列约 18 GB，8192 token 约 37 GB。所以 LoRA 微调长序列仍然需要激活重算（gradient checkpointing），换来的是额外约一次前向的 FLOPs。

### 3. 计算形态：额外 FLOPs 不到 1%，但 kernel 数翻倍

前向多了两个小矩阵乘：$$x \to xA^\top \to (xA^\top) B^\top$$，FLOPs 是 $$2r(d_{in} + d_{out})$$ 对比原来的 $$2 d_{in} d_{out}$$，比例

$$
\frac{r(d_{in} + d_{out})}{d_{in} d_{out}}
$$

$$W_Q$$：$$16 \times 8192 / 16.78\text{M} \approx 0.78\%$$；$$W_K$$、$$W_V$$：$$16 \times 5120 / 4.19\text{M} \approx 1.95\%$$；FFN 三个矩阵约 0.50%。训练的 FLOPs 几乎全在冻结权重的前向与反向上，**LoRA 省的是状态不是算量**——反向仍要算 $$\partial L / \partial x$$ 穿过每一层，只省掉了 $$\partial L / \partial W$$ 那一项（约占反向的一半），所以 LoRA 训练每 token 约 $$4N$$ 而非 $$6N$$ FLOPs。

推理时有两条路：

- **合并**：$$W' = W + (\alpha/r) BA$$ 算一次存下来，推理与原模型零差别、零开销。单租户部署的默认选择。
- **不合并**：为了让一个底座同时服务多个 LoRA（多租户），$$W$$ 只存一份，每个请求带自己的 $$A_i$$、$$B_i$$。decode 时 $$B = 1$$ 每步除了原来的 7 个 GEMV，多了 14 个极小的 GEMV（$$A$$、$$B$$ 各 $$r \times d$$，$$W_Q$$ 上 $$16 \times 4096 \times 2$$ B = 128 KB），字节数可忽略，但 32 层 × 14 = 448 次额外 kernel 启动，每次几微秒，加起来与 4.8 ms 的下界同量级——这是 decode memory-bound 的又一面：小 kernel 的固定开销比它的 FLOPs 和字节都贵。

多 LoRA 服务把一个 batch 里属于不同 adapter 的行分组：

$$
Y = X W + \begin{bmatrix} X_1 B_1 A_1 \\ X_2 B_2 A_2 \\ \vdots \end{bmatrix}
$$

前一项是所有请求共享的一次 GEMM，后一项是"每段 $$X_i$$ 乘各自的小矩阵"——Punica（Chen 等 2023）称之为 SGMV（Segmented Gather Matrix-Vector），一个 kernel 内按段 gather 不同的 $$A_i$$、$$B_i$$ 完成全部请求；S-LoRA（Sheng 等 2023）在此之上把 adapter 权重与 KV cache 统一分页管理，支持上千个 adapter 常驻。vLLM 的 multi-LoRA 支持基于这类 kernel。它们的成本模型与本篇的 Roofline 一致：adapter 字节数小，瓶颈在 kernel 组织，不在带宽。

### 4. QLoRA：把底座也量化

LoRA 的 16.7 GB 里 16.06 GB 是冻结的 BF16 底座。QLoRA（Dettmers 等 2023）把这一项也量化，训练时反量化到 BF16 参与前向与反向，梯度只流向 BF16 的 LoRA 参数：

- **NF4（NormalFloat 4）**：不是均匀量化，16 个级别取标准正态分布的等概率分位点——预训练权重近似正态分布，这样每个级别被使用的概率相等，信息论上最优。block size 64，每块一个 FP32 absmax scale，元数据 $$32/64 = 0.5$$ bit/权重；
- **双重量化**：把 FP32 的 scale 再按每 256 个一组量化到 FP8，元数据降到 $$8/64 + 32/(64 \times 256) \approx 0.127$$ bit/权重，总计约 4.13 bit；
- **paged optimizer**：用 CUDA 统一内存把优化器状态在显存尖峰时换页到 CPU，避免长序列梯度检查点时的 OOM。

8B 底座：$$8.03\text{B} \times 4.13 / 8 \approx 4.1$$ GB，保留部分层高精度后**约 4.5 GB**；加 LoRA 状态 0.67 GB，权重侧不到 5.2 GB，一张 24 GB 的消费级卡可以微调 8B 模型（激活值决定能开多长的序列）。代价是每次前向和反向都要反量化整份权重，每步时间明显长于 BF16 LoRA——又是第二章的结论：量化省字节，反量化加算量，训练是 compute-bound 的，所以 QLoRA 是**用时间换显存**。


## 五、实践：完成最终成本表

### 1. 脚本新增的三组函数

延续贯穿全系列的 `llm_cost.py`，本篇新增量化字节数、投机解码加速比、LoRA 参数三组函数。为了独立运行，下面同时给出前几篇中本篇用到的 `param_count`、`forward_flops_per_token`、`kv_bytes_per_token` 的 dense 版本（MoE 与 MLA 的版本在第三、五篇）。

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

# ---- 前几篇的函数（dense 版本）----
def embedding_params(cfg):
    return cfg.vocab * cfg.hidden * (1 if cfg.tie_embeddings else 2)

def param_count(cfg):
    """第一篇的参数量（重给以便独立运行），返回 dict。"""
    d = cfg.hidden
    q, kv = cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim
    attn = d * q + d * kv + d * kv + q * d
    ffn = 3 * d * cfg.d_ff
    per_layer = attn + ffn + 2 * d
    embed = cfg.vocab * d
    lm_head = 0 if cfg.tie_embeddings else cfg.vocab * d
    total = per_layer * cfg.layers + embed + lm_head + d
    return {"per_layer": per_layer, "embedding": embed, "lm_head": lm_head, "total": total}

def forward_flops_per_token(cfg, ctx=0):
    # embedding 查表不算 GEMM；lm_head 算
    gemm_params = param_count(cfg)["total"] - cfg.vocab * cfg.hidden
    return 2 * gemm_params + 4 * cfg.hidden * ctx * cfg.layers

def kv_bytes_per_token(cfg, dtype_bytes=2):
    return 2 * cfg.layers * cfg.n_kv_heads * cfg.head_dim * dtype_bytes

# ---- 第七篇新增 ----
def quantized_weight_bytes(cfg, bits=4, group_size=128, scale_bits=16,
                           zero_bits=16, keep_embed_bf16=False):
    """weight-only 量化后的权重字节数；返回 (bytes, 等效 bit/权重)。"""
    eff_bits = bits + (scale_bits + zero_bits) / group_size
    n = param_count(cfg)["total"]
    if keep_embed_bf16:
        e = embedding_params(cfg)
        return (n - e) * eff_bits / 8 + e * 2, eff_bits
    return n * eff_bits / 8, eff_bits

def roofline_step_time(cfg, gpu, rows, weight_bytes, mfu=1.0):
    """一次前向处理 rows 个 token 行的时间下界 max(访存, 计算)，只计权重部分。"""
    t_mem = weight_bytes / gpu.bandwidth
    t_cmp = forward_flops_per_token(cfg) * rows / (gpu.bf16_flops * mfu)
    return max(t_mem, t_cmp)

def speculative_speedup(alpha, gamma, c, batch, cfg, gpu, mfu=1.0):
    """投机解码相对普通 decode 的加速比；返回 (speedup, E[tokens])。"""
    exp_tokens = (1 - alpha ** (gamma + 1)) / (1 - alpha)
    w = param_count(cfg)["total"] * 2              # BF16 目标模型
    t_base = roofline_step_time(cfg, gpu, batch, w, mfu)
    t_verify = roofline_step_time(cfg, gpu, batch * (gamma + 1), w, mfu)
    t_round = gamma * c * t_base + t_verify
    return exp_tokens * t_base / t_round, exp_tokens

LORA_TARGETS_ATTN = ("q", "k", "v", "o")
LORA_TARGETS_ALL = ("q", "k", "v", "o", "gate", "up", "down")

def lora_params(cfg, rank=16, targets=LORA_TARGETS_ALL):
    d = cfg.hidden
    q, kv = cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim
    shapes = {"q": (d, q), "k": (d, kv), "v": (d, kv), "o": (q, d),
              "gate": (d, cfg.d_ff), "up": (d, cfg.d_ff), "down": (cfg.d_ff, d)}
    per_layer = sum(rank * (din + dout)
                    for t in targets for din, dout in [shapes[t]])
    return per_layer * cfg.layers

if __name__ == "__main__":
    for cfg in (LLAMA3_8B, LLAMA3_70B):
        n = param_count(cfg)["total"]
        b4, eb = quantized_weight_bytes(cfg)
        print(f"{cfg.name}: {n/1e9:.2f}B  BF16 {n*2/1e9:.1f} GB  "
              f"INT4(g128) {eb:.2f} bit -> {b4/1e9:.2f} GB")
        print(f"  decode 下界 BF16 {n*2/H100.bandwidth*1e3:.2f} ms  "
              f"W4A16 {b4/H100.bandwidth*1e3:.2f} ms")
        print(f"  LoRA r=16 attn {lora_params(cfg, 16, LORA_TARGETS_ATTN)/1e6:.2f}M  "
              f"all {lora_params(cfg, 16)/1e6:.2f}M ({lora_params(cfg, 16)/n*100:.2f}%)")
    print("投机解码 alpha=0.8 gamma=4 c=0.1 (Llama-3-8B, H100):")
    for B in (1, 8, 32, 64, 128, 256):
        s, e = speculative_speedup(0.8, 4, 0.1, B, LLAMA3_8B, H100)
        s60, _ = speculative_speedup(0.8, 4, 0.1, B, LLAMA3_8B, H100, mfu=0.6)
        print(f"  B={B:4d}  peak {s:.2f}  60%MFU {s60:.2f}  E[tokens]={e:.2f}")
```

输出：

```text
Llama-3-8B: 8.03B  BF16 16.1 GB  INT4(g128) 4.25 bit -> 4.27 GB
  decode 下界 BF16 4.79 ms  W4A16 1.27 ms
  LoRA r=16 attn 13.63M  all 41.94M (0.52%)
Llama-3-70B: 70.55B  BF16 141.1 GB  INT4(g128) 4.25 bit -> 37.48 GB
  decode 下界 BF16 42.12 ms  W4A16 11.19 ms
  LoRA r=16 attn 65.54M  all 207.09M (0.29%)
投机解码 alpha=0.8 gamma=4 c=0.1 (Llama-3-8B, H100):
  B=   1  peak 2.40  60%MFU 2.40  E[tokens]=3.36
  B=   8  peak 2.40  60%MFU 2.40  E[tokens]=3.36
  B=  32  peak 2.40  60%MFU 2.40  E[tokens]=3.36
  B=  64  peak 2.38  60%MFU 1.61  E[tokens]=3.36
  B= 128  peak 1.39  60%MFU 0.89  E[tokens]=3.36
  B= 256  peak 0.76  60%MFU 0.62  E[tokens]=3.36
```

70B 的 BF16 decode 下界 42 ms 是"假设能放进一张卡"的数值，实际放不进；INT4 的 11.2 ms 是真的单卡数字。换 `keep_embed_bf16=True` 得到 40.6 GB / 12.1 ms；换 `zero_bits=4` 得到 4.16 bit。

### 2. 最终成本表

七篇的数字合到一张表（H100 SXM，理论值；DeepSeek-V3 列用第三、五篇的 MLA 与 MoE 版本函数）：

```text
                            Llama-3-8B          Llama-3-70B           DeepSeek-V3
参数量                       8.03B               70.55B                671B（每 token 激活 37B）
权重字节 BF16                16.06 GB            141 GB                1342 GB（FP8 671 GB）
权重字节 INT4（4.25 bit）     4.27 GB             37.5 GB               356 GB
每 token 权重 FLOPs          15.0 GFLOPs         ~141 GFLOPs           ~74 GFLOPs
KV cache / token（BF16）      128 KiB             320 KiB               68.6 KiB（MLA）
KV cache / token（FP8）       64 KiB              160 KiB               34.3 KiB
128K 上下文 KV cache（BF16）  16 GiB              40 GiB                8.6 GiB
decode 下界 B=1，BF16         4.8 ms              不能单卡              不能单卡
decode 下界 B=1，W4A16        1.2–1.3 ms          11.2 ms（单卡可放）    8 卡可放
投机解码 α=0.8 γ=4 c=0.1      2.4×（B ≲ 60）       2.4×（B ≲ 60）        MTP α≈0.85–0.9 γ=1 → ~1.8×
LoRA r=16 q/k/v/o            13.63M（0.17%）      65.5M（0.093%）       —
LoRA r=16 全部七个矩阵        41.9M（0.52%）       207M（0.29%）         —
LoRA 训练权重侧显存           16.06 + 0.67 GB     141 + 3.3 GB          —
```

DeepSeek-V3 的投机一行按其技术报告的 MTP 接受率转述；LoRA 一行留空是因为细粒度 MoE 的专家矩阵通常不做 LoRA，attention 侧可以用同一函数算。

### 3. 实验设计：BF16 与 INT4 在 batch 1 与 64 下的对照

本篇没有实测数字，只给设计与预期，读者可以在一张 H100 上复现：

1. **准备**：同一个 Llama-3-8B（或 3.1）的 BF16 权重与它的 AWQ / GPTQ INT4 版本（$$g = 128$$），用 vLLM 分别加载，关闭 prefix caching，固定 `max_num_seqs` 使 batch 恰好为 1 或 64；
2. **decode 测量**：prompt 固定为短序列（如 128 token），生成 512 token，记录 token 间延迟（ITL）与总吞吐；
3. **prefill 测量**：prompt 8192 token，生成 1 token，记录首 token 延迟（TTFT）；
4. **对照量**：decode 用 $$T_{\text{mem}}$$（4.8 ms 与 1.27 ms）；prefill 用 $$8192 \times 15.0 \ \text{GFLOPs} / 989 \ \text{TFLOPS}$$ 除以一个 MFU（0.5–0.7）。

预期现象：

- batch 1 decode：INT4 的 ITL 约为 BF16 的 1/3（不到理论的 1/4，差在 lm_head、KV 读取与 kernel 效率）；
- batch 64 decode：两者差距缩小——INT4 的算术强度已接近其转折 batch，BF16 仍在斜线上；如果上下文长，KV cache 读取成为共同的主项，差距进一步缩小；
- prefill 8K：INT4 与 BF16 的 TTFT 接近，INT4 通常略慢，与 W4A16 kernel 在大 $$m$$ 下的效率有关。

若再加投机解码（vLLM 的 `speculative_config`，用 n-gram 或一个小草稿模型），预期 batch 1 下 ITL 明显下降，batch 64 下不变或上升，与第三章表格一致。**任何实测与下界的差距都应该能归因到本文模型忽略的某一项**——这比数字本身重要。


## 六、小结

三种方法各改一个变量：

```text
                改的量            机制                                  收益区间
量化 W4A16      W_bytes ↓ 4×      权重 4 bit，片上反量化到 BF16 计算       decode，B ≲ ridge/4 ≈ 75；prefill 无益或更慢
量化 W8A8       W_bytes ↓ 2×，     FP8/INT8 Tensor Core 算力 2×            decode 2× 与 prefill 2× 上限都有
                F ↑ 2×
KV cache 量化   KV bytes ↓ 2×      FP8/INT8 存 K、V                        长上下文、大 batch 的 decode
投机解码        每步 m ↑ (γ+1)     一次前向验证 γ+1 个 token               B ≲ ridge/(γ+1) ≈ 60；compute-bound 后 < 1×
LoRA            训练 N ↓ 200×      冻结 W，训练 BA                          训练状态 128 GB → 16.7 GB；激活值不变
QLoRA           底座 bytes ↓ 4×    NF4 底座 + BF16 LoRA                    权重侧 ~5 GB；每步更慢
```

本篇的数字：

```text
                                  Llama-3-8B        Llama-3-70B       DeepSeek-V3
INT4 g128 等效位宽                  4.25 bit          4.25 bit          4.25 bit
INT4 权重字节                       4.27 GB           37.5 GB           356 GB
decode 下界 BF16 → W4A16            4.8 → 1.27 ms     — → 11.2 ms       —
W4A16 转折 batch（ridge/4）          ~79               ~79               —
KV cache/token BF16 → FP8           128 → 64 KiB      320 → 160 KiB     68.6 → 34.3 KiB
投机 E[tokens]（α=0.8, γ=4）         3.36              3.36              1.85–1.9（MTP）
投机加速比（c=0.1，B ≲ 60）           2.4×              2.4×              ~1.8×
投机转折 batch（ridge/(γ+1)）         ~60               ~60               —
LoRA r=16 参数（七个矩阵）             41.9M（0.52%）     207M（0.29%）      —
LoRA 训练权重侧显存                  16.7 GB           144 GB            —
LoRA 额外 FLOPs（W_Q）               0.78%             0.39%             —
```

核心问题的答案：INT4 模型 decode 快、prefill 慢，投机解码 batch 1 有效、batch 64 无效，是同一条 Roofline 上的同一件事——**两种方法都在兑现 memory-bound 区间里空转的算力，一个用省下的字节换时间，一个用多算的 FLOPs 换 token；一旦 batch（或 prompt 长度）把工作点推过 ridge，算力不再空转，两者的收益就同时消失。** 而 LoRA 站在训练这一侧，它省的不是算力也不是带宽，是每参数 16 字节的状态。


## 全系列总结

七篇文章，每篇留下几个公式和几个数字：

```text
第一篇  参数量        每层 attention d(d_q + 2d_kv + d_q)、FFN 3·d·d_ff；Llama-3-8B 218.1M/层 × 32 + 1.05B = 8.03B；
                      70B 70.55B；DeepSeek-V3 671B（每 token 激活 37B）；Mixtral 46.7B（激活 12.9B）
第二篇  FLOPs·字节    GEMM 2mkn；每参数每 token 2 FLOPs；8B 每 token 15 GFLOPs；prefill 8K 约 158 TFLOP；
                      decode 算术强度 ≈ B；H100 ridge 295；decode 下界 16.06 GB / 3.35 TB/s = 4.8 ms
第三篇  KV cache      2·L·n_kv·d_head·bytes；8B GQA 128 KiB/token（MHA 512 KiB）；70B 320 KiB；
                      MLA (512+64)×2×61 = 68.6 KiB，压缩 57×；一张 H100 放 8B 后约 50 万 token 的 KV
第四篇  长上下文      RoPE 波长 2π·base^(2i/d)；base 500000 最低频 ~250 万；attention 二次项 4ds/层；
                      8B 128K prefill 权重 2.0 PFLOP + attention 4.5 PFLOP；128K KV 16 GiB
第五篇  MoE           期望激活专家 E·[1−(1−k/E)^B]：DeepSeek-V3 B=32 → 163，B=128 → 252；
                      dispatch 7 KiB + combine 14 KiB 每 token 每专家；grouped GEMM 每专家 128 行
第六篇  数值          BF16 1/8/7 ε=2^-7；FP16 max 65504，softmax 溢出 x > 11.09；E4M3 max 448；
                      混合精度 + Adam 16 B/参数，8B 训练状态 128 GB；FP8 每 128 元素提升 FP32 累加
第七篇  量化·投机·LoRA INT4 g128 4.25 bit，70B 37.5 GB 单卡；W4A16 decode 4.8 → 1.27 ms，转折 B ≈ ridge/4；
                      投机 E = (1−α^(γ+1))/(1−α) = 3.36，加速 2.4×，转折 B ≈ ridge/(γ+1)；
                      LoRA r=16 41.9M（0.52%），训练状态 128 GB → 16.7 GB
```

贯穿这些数字的是**四组变量**的成本模型：

$$
\text{参数量 } N \ \to\ \text{FLOPs/token} \approx 2N,\ \text{权重字节} = N \cdot \text{bytes/param},\ \text{KV cache/token} = 2 L n_{kv} d_{head} \cdot \text{bytes/elem}
$$

再加一张卡的两个上限（算力 $$F$$、带宽 $$BW$$）和一个比值（ridge $$= F / BW$$）。结构（GQA、MLA、MoE、RoPE）决定前三组变量的值；精度（BF16、FP8、INT4）决定 bytes；工作点（batch、序列长度、prefill 还是 decode）决定落在 Roofline 的哪一侧。每一篇都是在这个模型里填一格。

回到总纲的"最终目标"——拿到一个 `config.json` 和一张 GPU 的规格表，现在能回答：

```text
它有多少参数，分布在哪里？                 → 逐矩阵公式代入 config；8B 里 87% 在 FFN+attention、13% 在 embedding/lm_head
一张卡放得下吗？剩多少显存？               → N × bytes/param；BF16 8B 占 16 GB，70B 需 INT4 才能单卡；剩余给 KV cache
每 token 多少 FLOPs？各阶段瓶颈？          → 2N；prefill compute-bound，decode memory-bound，分界是 ridge
batch 开到多大才能用满算力？               → B ≈ ridge ≈ 295（H100 BF16），FP8 下 591；W4A16 后是 ridge/4
支持多长上下文？代价在哪？                 → KV cache 线性项 + attention 二次项；RoPE 的波长决定外推
attention 变体让 kernel 长什么样？          → GQA 的 4/8 个 query 头共享一个 KV 头；MLA 的吸收让 KV 变成 576 维
MoE 多卡要传多少数据？                     → 每 token 每专家 7 + 14 KiB，乘期望激活专家数
用什么精度？哪一步会出数值问题？            → BF16 前向、FP32 累加与主权重；softmax 与 RMSNorm 的溢出/下溢点
量化能快多少？在哪个阶段？                 → 字节数之比，只在 decode 且 B ≲ ridge/k 时兑现；W8A8 才对 prefill 有效
投机解码值得开吗？上界多少？               → (1−α^(γ+1))/(1−α) 除以 (γc+1)，只在 B ≲ ridge/(γ+1) 时成立
微调需要多少显存？                         → 全量 16 B/参数；LoRA 为 2 B/参数 + 可忽略；激活值另算，随序列长度线性
```

这三种能力——不看 benchmark 先算出理论值、用理论值判断优化的有效区间、用同一张表与算法、kernel、平台工程师对话——是本系列试图建立的全部内容。

本系列的边界也在这里：它只把模型当作一个**计算对象**，算它的参数、算量、字节数与通信量。FlashAttention 与量化 GEMM 的 kernel 怎么写、continuous batching 与 PagedAttention 怎么调度、TP / PP / EP 怎么切分与同步、训练配方怎么定——这些都建立在本系列给出的数字之上，但各自是另一个系列的内容。回到总纲：[《Transformer 与 LLM：结构、算量与数值》](/transformer-and-llm-for-infra-engineers.html)。
