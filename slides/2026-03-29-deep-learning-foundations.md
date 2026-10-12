---
layout: slides
title: "深度学习基础：从反向传播到残差"
subtitle: "系列精华 · 九篇正文每篇一页，按 ↓ 看实验与原论文图"
permalink: /slides/deep-learning-foundations.html
series: deep-learning-foundations
date: 2026-03-29 23:30:00 +0800
author: arganzheng
description: "《深度学习基础》系列的分享用幻灯片：一条连乘链——反向传播写出它、激活改变梯度、初始化 / 归一化 / 残差修好它、优化器定步长、正则化看泛化，再在 CNN、Embedding 与 RNN 上看结构演进；每篇一个 CPU 实验。"
theme: white
transition: slide
---

## 这个系列回答一个问题

> 深网络训练里的每个现象，能不能**推到公式、代到数字、用几十行代码复现**，然后在 LLM 里认出它的形态？

- 主线是**一条连乘链**：前向的方差每层乘一个因子、反向的梯度是一串 Jacobian 的乘积——因子偏离 1 就指数放大或消失
- 01 回顾演进史 → 02 写出梯度乘积 → 03 激活改变梯度 → 04 稳定深层训练 → 05 步长怎么定 → 06 泛化 → 07–09 看卷积、表示与循环
- 九篇用同一份几百行的 NumPy 小框架和配套脚本，实验可在笔记本 CPU 上复现

<aside class="notes" markdown="1">
总纲：/deep-learning-foundations.html。ResNet 的退化问题与 RNN 的 BPTT 衰减是同一条链的两个历史形态，残差与 LSTM 的遗忘门是同一个修法。
</aside>

---

## 全景：一条连乘链

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 190}}}%%
flowchart TB
    H["01 深度模型演进史<br/>感知机 → Transformer"] --> D1["02 反向传播<br/>写出那串 Jacobian 的乘积"]
    D1 --> D2["03 激活函数<br/>非线性改变梯度与 FFN"]
    D2 --> D3["04 初始化 / 归一化 / 残差<br/>稳定深层训练"]
    D3 --> D4["05 优化器<br/>拿到稳定的梯度后步长怎么定"]
    D4 --> D5["06 正则化与泛化<br/>挑出的解为什么泛化、何时不再"]
    D3 --> D6["07 CNN<br/>ResNet 的退化问题 = 连乘的历史形态一"]
    D6 --> D7["08 Embedding<br/>离散 token → 可学习向量"]
    D7 --> D8["09 RNN<br/>BPTT 衰减；LSTM 遗忘门 ≈ 残差"]
    D8 -. "seq2seq 瓶颈 → attention 诞生" .-> T["→ L4 Transformer"]
    D6 -. "patch → token（ViT）" .-> T
```

---

## 01 · 深度模型演进史：从感知机到 Transformer

![从 McCulloch–Pitts 到 Transformer、GPT-3 与 ChatGPT 的深度模型时间线](/img/in-post/dl-evolution-timeline.svg){: style="max-height: 420px"}

从阈值单元、感知机与两次寒冬，到反向传播、预训练、CNN、Embedding、attention 与 Transformer；结构演进不断改变归纳偏置、可训练性与并行度。

<aside class="notes" markdown="1">
原文 /evolution-of-deep-models-from-perceptron-to-transformer.html。XOR 案例：AND 感知机第 6 轮零误分；XOR 感知机每轮最少 3 次误分，权重状态自第 2 轮起周期 1（输出 `(2, 1)`）；ADALINE 最小二乘 MSE 0.25；H=2 的 MLP 有 13/20 个种子达到 100%，H=4 为 20/20。
</aside>

<!-- v -->

### XOR：能表示，不等于能学会

![感知机、ADALINE 与 MLP 的 XOR 案例和训练曲线](/img/in-post/dl-evolution-xor.webp){: style="max-height: 430px"}

感知机的单一线性边界无法解 XOR；隐藏层提供表达能力，初始化与优化仍影响能否找到解。

---

## 02 · 反向传播：手推一个两层网络

**结论**：反向传播 = 沿计算图反向拓扑序对每个算子算一次 **VJP**，从不构造 Jacobian；每个 Linear 反向做两个 GEMM，所以反向 = 2 × 前向、训练 $$6ND$$；$$\partial L/\partial W = X^\top G$$ 需要本层输入，所以**激活要存**。

![120 行 NumPy 从零训 MNIST：第一个 epoch 从 ln 10 = 2.30 在 100 步内掉到 0.5；15 个 epoch 97.6%](/img/in-post/dl-case-01-training.svg){: style="max-height: 420px"}

<aside class="notes" markdown="1">
原文 /backpropagation-by-hand.html。实测反向 / 前向比值 2.00；梯度检查 float64 相对误差 < 1e-6。
</aside>

<!-- v -->

### 三个公式与两个数字

| 量 | 公式 | 形状检查 |
|---|---|---|
| 输出层梯度 | $$G = (P - Y)/m$$ | 与 logits 同形 |
| 权重梯度 | $$\partial L/\partial W = X^\top G$$ | $$[n_{in}, m] \times [m, n_{out}]$$——需要本层输入 $$X$$ |
| 传给下一层 | $$\partial L/\partial X = G\,W^\top$$ | $$[m, n_{out}] \times [n_{out}, n_{in}]$$ |

- 一个 20 万参数网络的一层 Jacobian 就有 13 GB——所以只算 $$J^\top v$$
- 激活 552 KiB 对权重 795 KiB，batch × 32 → 17.3 MiB：**激活显存与 batch × 序列 × 层数 × 宽度成正比，与参数量无关**；重算用 33% 算力换掉（$$8ND$$）

---

## 03 · 激活函数：从 sigmoid 到 SwiGLU

**结论**：没有非线性，多层 Linear 仍可折叠成一层；激活的导数决定梯度能否穿过深层，门控激活则增加一个按输入选择信息的分支。小规模 MNIST 实验里，sigmoid 梯度在 20 层后消失并停在机会水平，其他五种激活均能训练。

![sigmoid、tanh、ReLU、LeakyReLU、GELU、SiLU 六种激活函数与导数：每格一组，实线为函数、虚线为导数，共用坐标轴](/img/in-post/activation-functions.webp){: style="max-height: 400px"}

<aside class="notes" markdown="1">
原文 /activation-functions-from-sigmoid-to-swiglu.html。20 层 MLP、250 步、三个 seed：sigmoid 10.64% [10.28, 11.35]，tanh 90.48% [89.84, 91.20]，ReLU 91.88% [91.16, 92.49]，LeakyReLU 92.02% [91.89, 92.15]，GELU 92.22% [91.87, 92.48]，SiLU 92.88% [92.63, 93.02]。
</aside>

<!-- v -->

### ReLU 到 SwiGLU

- sigmoid：导数最大 0.25，重复相乘；$$0.25^{10}\approx 9.54\times10^{-7}$$
- ReLU：正半轴梯度为 1，但实验中约 12.26% 的隐藏单元对测试样本恒为 0
- GELU / SiLU：平滑门控；GPT-2 使用 `gelu_new`
- SwiGLU：三矩阵门控 FFN；Llama、PaLM 等采用

---

## 04 · 训练为什么不稳定：初始化、归一化与残差

**结论**：深网络的稳定性是连乘问题；初始化只管 $$t = 0$$，归一化切断前向的连乘但修不了反向，残差把 Jacobian 变成 $$I + J$$ 给梯度一条恒等通路；**三样必须同用，Pre-Norm 是当前摆法**。

![64 层 MLP 七种接法逐块的激活标准差：plain 掉到 1e-9、无归一化的残差冲到 1e9、其余贴着 1](/img/in-post/dl-case-02-init-stats.svg){: style="max-height: 420px"}

<aside class="notes" markdown="1">
原文 /initialization-normalization-and-residual.html。Var(y) = n_in σ_w² Var(x)，Kaiming 2/n_in；0.9^128 ≈ 1.4e-6。
</aside>

<!-- v -->

### 七种接法只有 Pre-Norm 两个学习率都能训

![lr 0.05 与 0.005 下的 loss 曲线：Pre-Norm 两边 50 步内掉到 0.5 以下](/img/in-post/dl-case-02-training-curves.svg){: style="max-height: 400px"}

- Kaiming 初始化的 64 层 plain 网络：梯度范数在层间 1.7 到 13，lr 0.05 三步 NaN——初始化只管 $$t = 0$$
- 加了 LN 不加残差：300 步 37%；Post-Norm 顶层梯度是底层 4 倍，lr 0.05 时 8.8% 对 Pre-Norm 91.6%

<!-- v -->

### Post-LN vs Pre-LN（Xiong 等 2020）

![Xiong 等 2020 图 1：Post-LN 把 LayerNorm 放在残差相加之后，Pre-LN 放在子层之前。图片版权归原作者，教学评述引用](/img/in-post/dl-paper-xiong-preln.webp){: style="max-height: 400px"}

- 残差流无归一化每层翻倍（$$2^{64}$$），Pre-Norm 下线性增长；GPT-2 把残差分支缩放 $$1/\sqrt{2L}$$；LLM 用 Pre-Norm 加 final norm

---

## 05 · 优化器：从 SGD 到 AdamW 与学习率调度

**结论**：Adam 让每个参数的步长 $$\approx \eta$$、与梯度大小无关——所以第一步是满步长 $$\eta \cdot \text{sign}(g)$$、**必须 warmup**；$$L_2$$ 被 $$1/\sqrt v$$ 缩放而 AdamW 不被；线性 scaling 只在临界 batch 内成立。

![四种优化器 × 五个学习率 1 个 epoch 后的测试准确率：最优 lr 差 300 倍、成绩差不到半个点](/img/in-post/dl-case-03-optimizers.svg){: style="max-height: 420px"}

<aside class="notes" markdown="1">
原文 /optimizers-from-sgd-to-adamw.html。Adam 状态 8 字节 / 参数，Llama-3-8B 64 GB——16 字节里的 12。
</aside>

<!-- v -->

### warmup 为什么不能省

![峰值 lr 0.01 不加 warmup 的 loss 冲到 4.59（高于随机），加 100 步 warmup 则 1.26](/img/in-post/dl-case-03-warmup.svg){: style="max-height: 400px"}

- 偏差修正后第一步是 $$\eta \cdot \text{sign}(g)$$，$$v$$ 还没学到尺度
- batch 翻倍 lr 翻倍：512 成立、2048 NaN——只在「$$k$$ 步内梯度基本不变」时成立；裁剪 1.0 限制步长上界

<!-- v -->

### Adam + L2 不是 AdamW（Kingma & Ba 2014 Algorithm 1）

![Adam 算法：一阶矩 m、二阶矩 v、偏差修正、按 m̂/(√v̂ + ε) 更新。图片版权归原作者，教学评述引用](/img/in-post/dl-paper-adam-alg1.webp){: style="max-height: 360px"}

- $$\lambda\theta$$ 混进 $$g$$ 被 $$1/\sqrt v$$ 缩放，梯度小的参数被过度衰减；同一 $$\lambda = 0.1$$：$$\lVert W_1\rVert$$ 2.33 对 23.59，准确率 86.2% 对 95.6%
- $$\beta_2 = 0.95$$ 记 20 步；Momentum 有效学习率 $$\times 1/(1-\beta)$$

---

## 06 · 正则化与泛化：为什么参数比样本多却不过拟合

**结论**：先算参数 / 数据比定体制；过参数化时优化器挑最小范数、平坦、先学简单的解——**隐式正则化**；double descent 越过插值阈值再变好；预训练在数据体制里不用 dropout、一个 epoch。

![4,000 张、20% 错标签的 MNIST，宽度 2 → 2048：测试错误率在宽度 8–16（训练错误率刚到 0）处出现尖峰，之后再降到 14.5%](/img/in-post/dl-case-04-double-descent.svg){: style="max-height: 420px"}

<aside class="notes" markdown="1">
原文 /regularization-and-generalization.html。N/D：预训练 0.0005、SFT 1600、RM 160、本实验 407。
</aside>

<!-- v -->

### 过拟合最常见的形态：loss 涨而准确率不动

![1,000 张图 300 个 epoch：无正则测试 loss 从第 10 个 epoch 起升到 0.55，wd 压到 0.42，dropout 0.62](/img/in-post/dl-case-04-regularizers.svg){: style="max-height: 380px"}

- 测试 loss 0.40 → 0.55 而准确率不动——是**过度自信**；dropout 0.5 到 300 epoch 反而最差（0.62）
- 重复数据 4 epoch 以内几乎无损；2 万字符第 16 epoch 拐点、记忆率 10%，200 万字符未到

---

## 07 · CNN：从 LeNet 到 ResNet，再到 ViT

**结论**：卷积 = 全连接 + **局部性 + 参数共享**两条约束；深度是为了感受野，深了就训不动——BN 修前向、残差修反向；数据够多时约束成负担，ViT 把图切成 token，只留下 patch embedding 这一个 stride 卷积。

![LeCun 等 1998 图 2：LeNet-5——32×32 输入 → C1 6@28×28 → S2 → C3 16@10×10 → S4 → C5 120 → F6 84 → 10。图片版权归原作者 / IEEE，教学评述引用](/img/in-post/dl-paper-lenet5-fig2.webp){: style="max-height: 300px"}

<aside class="notes" markdown="1">
原文 /cnn-from-lenet-to-resnet-and-vit.html。3×3 核相当于 16×36 的矩阵、144 个非零、9 个自由参数。
</aside>

<!-- v -->

### 残差块（He 等 2015）：与 Transformer 的残差是一回事

![He 等 2015 图 5：左 ResNet-34 的基本块（两个 3×3），右 ResNet-50 的 bottleneck（1×1 降维 → 3×3 → 1×1 升维）。图片版权归原作者，教学评述引用](/img/in-post/dl-paper-resnet-fig5.webp){: style="max-height: 340px"}

- plain 56 层 loss 0.72、首末层梯度比 2849——退化问题就是连乘；bottleneck 69K 参数对 1.18M
- ResNet-50 25.6M 参数、**8.2 GFLOPs**（4.1 数的是 MACs）、每参数用 320 次
- ViT：196 / 576 / 5476 个 token；patch embedding = kernel = stride = p 的卷积，两种写法差 $$10^{-6}$$、参数同为 590,592

<!-- v -->

### 案例：复现 LeNet-5

![一张手写 9 过 C1 后的 6 张特征图：不同的核分别把左边缘、右边缘、笔画内部点亮；下排是学出的 6 个 5×5 卷积核](/img/in-post/dl-case-05-lenet-filters.svg){: style="max-height: 360px"}

- 61,706 个参数、5 个 epoch、**0.82%**——MLP 的 1/3 参数、1/3 错误率；第一层学出边缘检测器

---

## 08 · Embedding 与表示学习：从 one-hot 到上下文表示

**结论**：one-hot 矩阵乘法等价于 embedding 表查一行；梯度只更新被访问的行。Word2Vec 从上下文预测目标或从目标预测上下文，使向量几何反映共现关系；Transformer 则让表示随上下文变化。

![完整 text8 训练后选定的 100 维词向量在二维 PCA 平面上的投影](/img/in-post/embedding-word2vec.webp){: style="max-height: 380px"}

<aside class="notes" markdown="1">
原文 /embedding-and-representation-learning.html。完整 text8 17,005,207 词，`min_count=5` 后词表 71,290；subsampling 后 5,597,324 词，100 维、动态窗口 1–5、5 个负样本、3 epochs、batch 16,384，共 6,153 步；CPU 310.0 秒。随机初始化近邻与训练后近邻并列，类比排除输入词，结果不宣称普遍语义规律。
</aside>

<!-- v -->

### 参数账与实验读数

- GPT-2 small 输入 embedding：50,257 × 768 = 38,597,376 个参数，占模型约 31.0%
- 训练后 `computer` 的近邻包括 `computers / machines / computing / hardware / software`；`king - man + woman` 排除输入词后没有命中 `queen`
- 完整语料训练仍不保证每个类比都命中预期词；PCA 展示投影，不代表高维距离被完整保留

---

## 09 · RNN：从 LSTM 到 attention 的诞生

**结论**：BPTT 是同一个 $$W$$ 的连乘，训练信号传不到远处；LSTM 的 $$c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t$$ 是**时间上的残差**，遗忘门偏置要初始化为 1；attention 为绕过 seq2seq 的固定向量瓶颈而生，然后人们发现循环可以不要。

![Graves 2013 图 2：LSTM 记忆细胞——输入门写入、遗忘门自环保持、输出门读出。图片版权归原作者，教学评述引用](/img/in-post/dl-paper-graves-fig2.webp){: style="max-height: 380px"}

<aside class="notes" markdown="1">
原文 /rnn-lstm-and-the-birth-of-attention.html。$$J_k = \text{diag}(1-h_k^2)W$$；20 步外梯度千分之四、60 步外 $$10^{-10}$$。
</aside>

<!-- v -->

### 从固定向量瓶颈到对齐（Bahdanau 等 2015）

![Bahdanau 等 2015 图 1：解码每一步对编码器所有状态算权重再加权求和——attention 的第一次出现。图片版权归原作者，教学评述引用](/img/in-post/dl-paper-bahdanau-fig1.webp){: style="max-height: 360px"}

- seq2seq 把整句压进一个向量（Sutskever 2014）；倒序 16 token 任务整句正确率 0% → **76%**
- attention 算力 4.5 倍、$$O(T^2 d)$$ 对 $$O(Td^2)$$——换来的是可并行与不衰减的通路

<!-- v -->

### 案例：字符级 LSTM 写莎士比亚，与 nanoGPT 同预算

![LSTM 前 500 步降得快得多，2000 步时 val 1.71 vs nanoGPT 1.66](/img/in-post/dl-case-06-char-lstm.svg){: style="max-height: 380px"}

- 这个规模看不出 Transformer 的优势——差别在顺序 vs 并行；RNN 记不住远处不是隐状态太小（64 维装得下 69 bit），是训练信号传不到

---

## Transformer 部件的来历：都从这条链上长出来

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 210}}}%%
flowchart TB
    CHAIN["梯度 = Jacobian 连乘（01、03）"] --> VAN["因子偏离 1 → 指数消失 / 爆炸（03、06、08）"]
    VAN --> INIT["初始化：t = 0 时因子期望为 1"]
    VAN --> NORM["归一化：前向方差拉回 1"]
    VAN --> RES["残差：I + J 的恒等通路"]
    RES --> GROW["残差流方差增长 → 归一化 + 1/√(2L)"]
    RES --> LSTM["LSTM 遗忘门 = 时间上的残差（08）"]
    RES --> PRE["Pre-Norm / ResNet-v2（03、06）"]
    NORM --> SCALE["尺度不变性 → 有效学习率由范数决定"]
    SCALE --> WD["AdamW weight decay = 范数平衡点（04、05）"]
    ADAM["Adam 步长 ≈ η（04）"] --> WARM["warmup、β₂ = 0.95、η ∝ 1/n_in"]
```

---

## 常见误区

- 反向传播要算出每层的 Jacobian——一层就 13 GB，只算 VJP
- 激活显存与参数量成正比——与 batch × 序列 × 层数 × 宽度成正比
- 初始化对了深网络就能训——初始化只管 $$t = 0$$，64 层 plain 三步 NaN
- Adam + L2 就是 AdamW——$$\lambda\theta$$ 被 $$1/\sqrt v$$ 缩放，86.2% 对 95.6%
- 参数比样本多就会过拟合——407 倍的网络测试 loss 最好；别停在插值阈值
- 验证准确率没掉就没过拟合——loss 涨而准确率不变是过度自信
- ViT 完全不用卷积——patch embedding 就是 stride 卷积
- Embedding 是 one-hot 的另一个名字——它是按 token id 查表得到的可学习向量
- RNN 记不住远处是隐状态太小——是训练信号传不到
{: .fragments}

---

## 下一步

- **原文**：总纲 [/deep-learning-foundations.html](/deep-learning-foundations.html) · 总结与通关自测 [/deep-learning-foundations-series-recap-and-self-test.html](/deep-learning-foundations-series-recap-and-self-test.html)
- **配套代码**：[ai-learning-labs/deep-learning-foundations](https://github.com/arganzheng/ai-learning-labs/tree/main/deep-learning-foundations)——NumPy 小框架、激活函数对照、64 层七种接法、优化器扫描、double descent、LeNet-5、text8 Word2Vec、字符级 LSTM
- **往后读**：L4 [Transformer 原理与实现](/transformer-and-llm-structure-implementation-and-evolution.html)与[现代 LLM 结构](/llm-architecture-evolution-roadmap-from-gpt2.html)——前者接 GPT-2 的结构与实现，后者接今天的结构改动与成本

<aside class="notes" markdown="1">
收尾。
</aside>
