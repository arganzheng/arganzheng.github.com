---
layout: post
series: deep-learning-foundations
title: "深度学习基础（03）：激活函数——从 sigmoid 到 SwiGLU"
subtitle: "Activation Functions: From Sigmoid to SwiGLU"
date: 2026-03-24 20:00:00
tags: [AI, Deep Learning, LLM]
catalog: true
---

两层 Linear 中间放一个 ReLU，和把两层直接乘起来，表达能力完全不同。差别不在层数，而在中间那一下非线性。本篇把常见激活函数按这条线串起来：它们怎样让网络学到弯曲的边界，又怎样改变梯度、显存和算子开销。

> **如果把 100 层 Linear 叠起来，为什么它仍只能表示一个线性 / 仿射变换？[^q0] sigmoid、ReLU、GELU 和 SwiGLU 分别解决了什么问题？[^q1]**

## 一、总览：为什么非要非线性

线性层负责混合输入维度；激活函数打破线性，才让层叠起来的网络表示复杂函数。若每层之间没有非线性，任意多层 Linear 都能折叠成一个 Linear（带 bias 时是仿射变换）。《[深度学习基础（02）：反向传播——手推一个两层网络](/backpropagation-by-hand.html)》的「两层网络前向」已经把这个矩阵乘法写出来了；这里沿用结论，不重做推导。

本篇从 sigmoid 与 tanh 的饱和讲起，经 ReLU 及其修补，走到平滑激活，再到现代 FFN 常用的门控变体。最后把曲线、梯度、参数形状和真实训练放在一起对照。

### 1. 本文的章节安排

| 章 | 主题 | 内容 |
|---|---|---|
| 二 | Sigmoid 与 tanh | 饱和、梯度消失、输出是否以 0 为中心 |
| 三 | ReLU | 稀疏、便宜与死亡单元；联系 Kaiming 初始化 |
| 四 | ReLU 的修补 | Leaky ReLU、PReLU、ELU |
| 五 | 平滑激活 | GELU、SiLU、非单调区间与 GPT-2 的近似 |
| 六 | GLU 家族 | GLU、ReGLU、GeGLU、SwiGLU；门控 FFN 的矩阵与宽度 |
| 七 | 成本与融合 | 逐元素读写、kernel fusion、反向所需的保存 |
| 八 | 对照表 | 公式、导数、值域、代价与代表模型 |
| 九 | 案例：深层 MNIST MLP | 初始梯度、固定步数准确率、ReLU 死亡单元 |
| 十 | 本文小结 |  |
| 十一 | 自测 | 6 道题 |

### 2. 来龙去脉：从饱和到门控

| 时间 | 工作 | 留下的东西 |
|---|---|---|
| 早期 | Sigmoid、tanh | 平滑、值域有界；深层连乘时容易饱和 |
| 2010–2012 | [Nair 与 Hinton（2010）](https://icml.cc/Conferences/2010/papers/432.pdf)、[Glorot、Bordes 与 Bengio（2011）](https://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf)、[AlexNet（2012）](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) | ReLU 从 RBM 与深层网络研究进入大规模视觉模型 |
| 2013–2015 | [Maas 等（2013）](https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf) 的 Leaky ReLU、[He 等（2015）](https://arxiv.org/abs/1502.01852) 的 PReLU、[Clevert 等（2015）](https://arxiv.org/abs/1511.07289) 的 ELU | 给 ReLU 的负半轴留一条梯度路径 |
| 2016 | [GELU（Hendrycks 与 Gimpel）](https://arxiv.org/abs/1606.08415)；[GLU（Dauphin 等）](https://arxiv.org/abs/1612.08083) | 平滑门控与逐元素门控两条线并行出现 |
| 2017 | [SiLU（Elfwing 等）](https://arxiv.org/abs/1702.03118)；[Swish（Ramachandran 等）](https://arxiv.org/abs/1710.05941) | 输入乘以 sigmoid 门，得到平滑、略带非单调性的曲线 |
| 2020 | [GLU 变体（Shazeer）](https://arxiv.org/abs/2002.05202) | 把 ReLU、GELU、SiLU 放进 GLU 门控结构，得到 ReGLU、GeGLU、SwiGLU |
| 2022–2024 | [PaLM（2022）](https://arxiv.org/abs/2204.02311)、[LLaMA（2023）](https://arxiv.org/abs/2302.13971)、[Gemma（2024）](https://arxiv.org/abs/2403.08295) | PaLM 与 LLaMA 使用 SwiGLU；Gemma 使用 GeGLU，门控激活进入大语言模型 |

## 二、Sigmoid 与 tanh：平滑，但会饱和

Sigmoid 把任意实数压进 $$(0,1)$$：

$$
\sigma(x) = \frac{1}{1 + e^{-x}}, \qquad
\sigma'(x) = \sigma(x)(1-\sigma(x))
$$

导数最大值出现在 $$x=0$$，是 $$0.25$$。输入绝对值很大时，输出靠近 0 或 1，曲线变平，导数也靠近 0；这叫**饱和**。反向传播把每层导数连乘，若连续十层都在最有利情况下乘上 $$0.25$$，梯度也只剩
$$
0.25^{10} \approx 9.54 \times 10^{-7}
$$
实际值还会乘上权重矩阵的导数因子。

若一个神经元的输入 $$x_i$$ 全为正，权重梯度 $$\partial L/\partial w_i=\delta x_i$$ 就都与 $$\delta$$ 同号。一次更新只能把这组权重整体上移或整体下移，不能在输入维度间灵活调整，优化路径容易 zig-zag；tanh 的零中心输出能缓和这个问题。

tanh 把输出移到 $$(-1,1)$$，以 0 为中心：

$$
\tanh'(x) = 1-\tanh^2(x)
$$

它在 0 点的最大导数为 1，但两端同样饱和。中心化改善了输入分布，却没有消除深层反向的连乘问题。

## 三、ReLU：简单、稀疏，也可能死亡

$$
\operatorname{ReLU}(x)=\max(0,x), \qquad
\operatorname{ReLU}'(x)=
\begin{cases}
0 & x<0 \\
1 & x>0
\end{cases}
$$

正半轴导数恰好为 1，不会像 sigmoid 那样把梯度再乘一个小于 1 的数。负半轴直接归零，让每个样本只有一部分神经元激活；计算只需比较和取最大值。

代价是负半轴的导数为零。如果某个单元对所有输入的预激活都变成负数，它输出永远为零，反向梯度也为零，参数无法靠梯度自己回来——这就是 **dead ReLU**。一次过大的更新可能把单元推到这个区域。

ReLU 让对称、零均值的信号有一半落到 0，初始化时需要补回这部分尺度。《[深度学习基础（04）：训练为什么不稳定——初始化、归一化与残差](/initialization-normalization-and-residual.html)》的「ReLU 砍掉一半」推导了 Kaiming 初始化里的因子 2；这里不重复推导。

## 四、ReLU 的修补：负半轴留一点路

- **Leaky ReLU**：负半轴改为 $$\alpha x$$，例如 $$\alpha=0.01$$。它保留一条小梯度，不需要学习额外参数。
- **PReLU**：把 $$\alpha$$ 设成可学习参数，让模型自己选负半轴斜率。
- **ELU**：负半轴用 $$\alpha(e^x-1)$$，输出下界为 $$-\alpha$$；它能产生负值，但需要指数运算。

三者都保留了 ReLU 的正半轴直通梯度，又避免负半轴完全没有梯度。现代 Transformer FFN 更常选 GELU 或 SiLU；门控形式见第六章。

## 五、平滑激活：GELU 与 SiLU

GELU 用标准正态分布的累积分布函数 $$\Phi$$ 给输入做软门控：

$$
\operatorname{GELU}(x)=x\Phi(x)
$$

可以把它读作“以概率 $$\Phi(x)$$ 保留输入 $$x$$”。它不像 ReLU 在 0 处突然折断；负输入大多被压小，正输入大多保留。导数为
$$
\operatorname{GELU}'(x)=\Phi(x)+x\phi(x)
$$
其中 $$\phi$$ 是标准正态密度。GELU 不是单调函数：数值计算的极小点约为 $$x=-0.752$$，函数值约 $$-0.170$$。

SiLU（也常叫 Swish，取参数 $$\beta=1$$）是

$$
\operatorname{SiLU}(x)=x\sigma(x), \qquad
\operatorname{SiLU}'(x)=\sigma(x)+x\sigma(x)(1-\sigma(x))
$$

它在 $$x=-1.278$$ 处达到约 $$-0.278$$，也会轻微落到 0 以下。两者都平滑地衔接负、正输入；与 ReLU 相比，它们在负半轴保留了连续变化。

GPT-2 配置里的 `activation_function` 是 [`gelu_new`](https://huggingface.co/openai-community/gpt2/blob/main/config.json)。这是 GELU 的 tanh 近似：

$$
\operatorname{GELU}_{\text{new}}(x)=\frac{x}{2}
\left(1+\tanh\left(\sqrt{\frac{2}{\pi}}\left(x+0.044715x^3\right)\right)\right)
$$

在 $$[-10,10]$$ 上取 2,000,001 个等距点，与精确的 $$x\Phi(x)$$ 比较，最大绝对误差约 $$4.73\times10^{-4}$$。

![六个小图分别展示 sigmoid、tanh、ReLU、Leaky ReLU、GELU 和 SiLU；每图实线为函数、虚线为导数，共用坐标轴](/img/in-post/activation-functions.webp)

## 六、GLU 家族：用一个分支控制另一个分支

普通两层 FFN 是“升维 → 激活 → 降维”。GLU 把升维分成值分支和门分支，再逐元素相乘：

$$
\operatorname{GLU}(x)=(xW+b)\odot\sigma(xV+c)
$$

门的输出介于 0 和 1，控制值分支各维保留多少。把门函数换掉，就得到：

$$
\begin{aligned}
\operatorname{ReGLU}(x)&=\operatorname{ReLU}(xW)\odot xV \\
\operatorname{GeGLU}(x)&=\operatorname{GELU}(xW)\odot xV \\
\operatorname{SwiGLU}(x)&=\operatorname{SiLU}(xW)\odot xV
\end{aligned}
$$

一个门控 FFN 有三张矩阵：门投影、值投影、输出投影。普通 FFN 的两张矩阵参数量约 $$2dd_{\mathrm{ff}}$$；门控 FFN 约 $$3dd_g$$。令门控宽度 $$d_g \approx \frac{2}{3}d_{\mathrm{ff}}$$，两者的主要矩阵参数量就接近。

模型例子能从论文和配置里交叉核对：[Llama 3 的官方实现](https://github.com/meta-llama/llama3/blob/main/llama/model.py)用 SiLU 门控，三张投影矩阵对应代码里的 `w1`、`w3`、`w2`；[Qwen2.5 技术报告](https://arxiv.org/abs/2412.15115)称其 dense 模型使用 SwiGLU，[Hugging Face 配置](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json)的 `hidden_act` 为 `silu`。

[PaLM 论文](https://jmlr.org/papers/v24/22-1144.html)明确采用 SwiGLU，并说明 MLP 因此需要三张矩阵。[Gemma 论文](https://arxiv.org/html/2403.08295)称其用 GeGLU。[T5 v1.1 配置](https://huggingface.co/google/t5-v1_1-base/blob/main/config.json)则把 `feed_forward_proj` 写作 `gated-gelu`。

这些结构的参数量与 FFN 宽度怎么算，见《现代 LLM 结构（01）：从 GPT-2 到 Llama——五处改动与参数量》的「[GELU 两矩阵 FFN → SwiGLU 三矩阵](/gpt2-to-llama-five-changes-and-parameter-count.html)」。

## 七、成本与融合：算得少，不代表花时少

逐元素激活常是**访存受限**：对一个 BF16 元素，读输入 2 字节、写输出 2 字节；若只做几次算术，计算强度大约只有 1.25 FLOPs/byte。以 $$[T,d_{\mathrm{ff}}]=[4096,11008]$$ 为例，一次读写约 $$4096\times11008\times4 \approx 180$$ MB，按每元素 5 次普通运算估算约 226 MFLOPs。

NVIDIA 的 [H100 SXM 数据表](https://resources.nvidia.com/en-us-gpu/h100-datasheet-24306)列出 BF16 Tensor Core 峰值 1,979 TFLOPS，星号注明该数值使用 sparsity；dense 峰值约为一半，即 989 TFLOPS。除以 3.35 TB/s HBM 带宽，ridge point 约 $$989/3.35=295$$ FLOPs/byte。逐元素激活约 1.25 FLOPs/byte，只有 ridge point 的 $$1/236$$，说明它更可能受带宽限制。Roofline 的两条上限与 ridge point 见《[现代 LLM 结构（02）：前向的算量与访存量](/transformer-flops-bytes-and-roofline.html)》「五、Roofline：把 FLOPs 和字节数放到同一张图上」的「算术强度与 ridge point」。

**融合**把相邻小算子合成一次 kernel，减少 launch 与中间张量读写。常见模式是 bias + GELU，以及 SwiGLU 里的 SiLU + multiply。细节见《GPU Kernel 工程（09）：量化与融合 kernel——推理系统的其余部分》的「[bias + activation 与 SiLU-and-mul](/quantization-and-fused-kernels.html)」。

训练时还要反向求梯度，权重梯度需要前向输入，因此激活不能随算完即弃。《[深度学习基础（02）：反向传播——手推一个两层网络](/backpropagation-by-hand.html)》的「激活为什么要存」解释了保存与 gradient checkpointing 的取舍。

## 八、常见激活函数对照

导数里的 $$\phi$$、$$\Phi$$ 分别是标准正态密度与累积分布；门控行的 $$g$$ 表示对应的 ReLU、GELU 或 SiLU。

| 函数 | 公式 | 导数 | 值域 | 相对开销 | 代表模型 / 场景 |
|---|---|---|---|---|---|
| Sigmoid | $$\sigma(x)$$ | $$\sigma(1-\sigma)$$ | $$(0,1)$$ | 较高：exp | 平滑非线性基线 |
| tanh | $$\tanh(x)$$ | $$1-\tanh^2(x)$$ | $$(-1,1)$$ | 较高：exp | 零中心输出对照 |
| ReLU | $$\max(0,x)$$ | $$\mathbf{1}_{x>0}$$ | $$[0,\infty)$$ | 最低 | AlexNet |
| Leaky / PReLU | $$x\;(x>0),\;\alpha x\;(x<0)$$ | $$1\;(x>0),\;\alpha\;(x<0)$$ | $$\mathbb{R}$$ | 低 | PReLU： [He 等（2015）的 ImageNet 模型](https://arxiv.org/abs/1502.01852)；Leaky ReLU：— |
| ELU | $$x\;(x>0),\;\alpha(e^x-1)\;(x\le0)$$ | $$1\;(x>0),\;\alpha e^x\;(x\le0)$$ | $$(-\alpha,\infty)$$ | 中：负侧 exp | — |
| GELU | $$x\Phi(x)$$ | $$\Phi(x)+x\phi(x)$$ | 约 $$[-0.170,\infty)$$ | 较高：CDF / tanh 近似 | GPT-2 |
| SiLU | $$x\sigma(x)$$ | $$\sigma+x\sigma(1-\sigma)$$ | 约 $$[-0.278,\infty)$$ | 中：sigmoid + 乘法 | SwiGLU 的门 |
| GLU | $$a\odot\sigma(b)$$ | $$a'\sigma(b)+ab'\sigma'(b)$$ | 取决于值分支 | 门 + 逐元素乘 | [Dauphin 等（2016）的门控卷积语言模型](https://arxiv.org/abs/1612.08083) |
| ReGLU / GeGLU / SwiGLU | $$g(xW)\odot xV$$ | $$J=W\,\operatorname{diag}(g'(xW)\odot xV)+V\,\operatorname{diag}(g(xW))$$ | 取决于门函数与输入 | 三投影 + 门控乘法 | GeGLU：T5 v1.1、Gemma；SwiGLU：PaLM、LLaMA、Llama 3、Qwen2.5 |

表里的“相对开销”只比较激活与逐元素门控；大模型 FFN 的总耗时还取决于矩阵乘法、融合和硬件。

## 九、案例：六种激活函数跑同一个深层 MLP

三个随机种子都在 MNIST 上比较同一个 20 层、宽度 256 的 MLP。每种激活使用相同的 Kaiming-normal 初始化、Adam 学习率 $$10^{-3}$$、batch 128 和 250 个更新步；训练子集 12,000 张，测试准确率在标准 MNIST 测试集 10,000 张上计算。

初始梯度范数列保留种子 2026 的结果，取第一与第二十个隐藏 Linear 的权重梯度，在同一批 128 张样本上测量。准确率与 ReLU 死亡率报告三个种子的均值和范围。死亡率定义为：测试集前 1,024 张图中，对所有输入都输出 0 的隐藏单元占比。

| 激活 | 初始第一层梯度范数 | 初始第二十层梯度范数 | 测试准确率：均值［最小–最大］ | ReLU 死亡率：均值［最小–最大］ |
|---|---:|---:|---:|---:|
| sigmoid | $$3.944\times10^{-10}$$ | $$1.010$$ | 10.64%［10.28–11.35］ | — |
| tanh | $$5.015$$ | $$1.303$$ | 90.48%［89.84–91.20］ | — |
| ReLU | $$2.999$$ | $$4.811$$ | 91.88%［91.16–92.49］ | 12.26%［11.41–13.22］ |
| LeakyReLU | $$3.182$$ | $$4.758$$ | 92.02%［91.89–92.15］ | — |
| GELU | $$2.722$$ | $$1.452$$ | 92.22%［91.87–92.48］ | — |
| SiLU | $$1.752$$ | $$0.9921$$ | 92.88%［92.63–93.02］ | — |

sigmoid 第一层的梯度只有 $$3.94\times10^{-10}$$，第二十层仍约为 $$1.01$$。这是第二章所说导数连乘在训练里的直接表现：梯度传回输入端时几乎消失，准确率均值 10.64%，仍在十分类的随机水平附近。

tanh 的导数在 0 附近约为 1，输出也以 0 为中心；梯度没有像 sigmoid 那样一路缩小，因此这个固定预算下能训练到约 90%。

ReLU 即使用普通学习率 $$10^{-3}$$，仍有平均 12.26% 的隐藏单元对这 1,024 张测试图都输出 0。这里的“死亡”是对整批输入都为零，不是某个样本上的稀疏激活。

ReLU、LeakyReLU 与 GELU 的准确率范围重叠，三次运行不足以区分它们。SiLU 的最低值 92.63% 比其他三者的最高值 92.49% 高 0.14 个百分点；这只说明在这组固定设置下超出了观察到的 seed spread，不足以推出普遍的激活函数排名。

一个单神经元例子能看到死亡如何发生：输入为 $$[1,2]$$，初始权重 $$w=2$$、偏置 $$b=0$$、目标均为 1。过大的 SGD 学习率把参数一步推到负侧；两个输入之后都落在 ReLU 的零梯度区，损失停在非零值。

```text
lr=0.6：w 2.00 → -2.20，b 0.00 → -2.40
输出 [2.00, 4.00] → [0.00, 0.00]；loss 5.00 → 1.00；下一步梯度 0
```

## 十、本文小结

- 没有非线性时，深层线性 / 仿射层仍可折叠成一层；激活函数让网络真正增加表达能力。
- Sigmoid 与 tanh 会饱和；Sigmoid 的最大导数是 0.25，ReLU 正半轴导数为 1，但可能产生死亡单元。
- Leaky ReLU、PReLU、ELU 给负半轴留下梯度；GELU、SiLU 用平滑门控处理输入。
- SwiGLU 等 GLU 变体用门控分支控制值分支，增加一张输入投影矩阵；约 2/3 宽度可维持主要参数预算。
- 激活是低算术强度的逐元素操作，融合可省中间张量流量；反向所需的前向激活仍要保存或重算。

配套代码：[`case_activation_functions.py`](https://github.com/arganzheng/ai-learning-labs/blob/main/deep-learning-foundations/case_activation_functions.py)——生成曲线并复现六种激活的初始化梯度、三种子固定步数训练与死亡 ReLU 示例。

## 十一、自测

1. 100 层 Linear 之间都没有激活函数时，为什么它们仍可折叠成一层？

   <details markdown="1"><summary>答案</summary>

   矩阵的乘积仍是一个矩阵；若有 bias，多层仿射变换的复合仍是仿射变换。层数增加了参数化方式，却没有增加非线性表达能力。

   </details>

2. Sigmoid 的导数上限是多少？连续十层都乘这个上限，梯度约剩多少？

   <details markdown="1"><summary>答案</summary>

   $$\sigma'(x)=\sigma(x)(1-\sigma(x))\le0.25$$；$$0.25^{10}=9.54\times10^{-7}$$，约百万分之一。

   </details>

3. ReLU 的稀疏激活与 dead ReLU 有什么区别？

   <details markdown="1"><summary>答案</summary>

   稀疏是某个样本上部分单元输出 0；死亡是某个单元对所有输入都输出 0，负半轴导数也为 0，因此不能从当前梯度中恢复。

   </details>

4. GELU 与 SiLU 的公式分别是什么？它们在负半轴有什么共同点？

   <details markdown="1"><summary>答案</summary>

   GELU 是 $$x\Phi(x)$$，SiLU 是 $$x\sigma(x)$$。两者都平滑，不像 ReLU 在零点折断；在一段负输入上函数值略低于 0。

   </details>

5. SwiGLU 为什么比普通 FFN 多一张投影矩阵？为什么常把隐藏宽度降到约 2/3？

   <details markdown="1"><summary>答案</summary>

   它有门分支、值分支、输出投影三张矩阵；普通 FFN 是两张。令 $$3d d_g \approx 2d d_{\mathrm{ff}}$$，得到 $$d_g\approx2d_{\mathrm{ff}}/3$$，主要矩阵参数量就接近。

   </details>

6. 为什么融合 bias + GELU 或 SiLU + multiply 可能比单独优化激活函数更重要？

   <details markdown="1"><summary>答案</summary>

   逐元素激活读写量相对运算数大，常受显存带宽与 kernel launch 限制。融合减少中间张量写回与再次读取，也减少一次 launch。

   </details>

## 下一篇

激活函数回答了层与层之间怎样引入非线性；接下来要让这些梯度在深网络里不消失、不爆炸。《[深度学习基础（04）：训练为什么不稳定——初始化、归一化与残差](/initialization-normalization-and-residual.html)》从方差传播与 ReLU 的因子 2 开始。

[^q0]: 多层线性变换的矩阵连乘仍是线性变换；加入 bias 后，仿射变换的复合仍然仿射。详见《[深度学习基础（02）：反向传播——手推一个两层网络](/backpropagation-by-hand.html)》的「两层网络前向」。
[^q1]: Sigmoid / tanh 平滑但会饱和；ReLU 让正半轴梯度直通；GELU 与 SiLU 使用平滑门控；GLU 家族再用一个分支控制另一个分支。本文分别讨论它们的公式、梯度与代价。
