---
layout: post
series: deep-learning-foundations
title: "深度学习基础（05）：CNN——从 LeNet 到 ResNet，再到 ViT"
subtitle: "CNN: Convolution as a Constrained Linear Layer, the ResNet Legacy and How ViT Turns Images into Tokens"
tags: [AI, Deep Learning, LLM]
catalog: true
---

前四篇讲训练动力学，用的网络全是 MLP。这一篇和下一篇回看 Transformer 之前的两条结构史——卷积与循环——不是为了怀旧，而是因为 Transformer 的每个部件都有来历：残差连接、归一化、"堆同样的块"来自卷积这条线；attention 来自循环那条线。理解一个部件当初解决了什么问题，才知道它今天还在解决什么、什么时候可以拿掉。

卷积网络的故事可以压缩成三句话：**卷积是一个被强约束的线性层**，约束带来的参数节省与归纳偏置让它在数据不多时远胜 MLP；**深度是为了感受野**，而深了就训不动，ResNet 用残差解决了它；**数据足够多时约束成了负担**，ViT 把图切成 patch、当成 token 送进标准 Transformer，只保留了卷积的一个影子——patch embedding 本身就是一个 stride 等于 kernel 的卷积。三句话各对应本篇的一个实验。全篇的核心问题是：

> **一个 3×3 卷积核相当于多大的全连接矩阵？ResNet 的残差与 Transformer 的残差是同一个东西吗？ViT 为什么可以不用卷积？**


## 一、总览：三个阶段

### 1. 卷积网络的三个阶段

| 阶段 | 年代 | 解决的问题 | 代表 | 留给 Transformer 的 |
|---|---|---|---|---|
| 卷积作为归纳偏置 | 1989–2014 | 图像的局部性与平移不变性；用参数共享把参数量压下来 | LeNet、AlexNet、VGG、GoogLeNet | 1×1 卷积 = 逐位置的线性层；小核堆深 |
| 深度与残差 | 2015–2016 | 深了训不动（退化问题）；BN 让前向稳定，残差让反向稳定 | ResNet、ResNet-v2 | **残差连接、归一化、堆同样的块、pre-activation（Pre-Norm 的前身）** |
| 去掉归纳偏置 | 2020– | 数据够多时局部性约束成了上限；把图切成 token 交给 attention | ViT、ConvNeXt | **patch embedding = stride 卷积；一张图等于多少 token** |

### 2. 本文的章节安排

| 章 | 主题 | 内容 |
|---|---|---|
| 二 | 卷积作为带约束的线性层 | 定义、参数量与 FLOPs 公式、等价的稀疏矩阵（实测 16×36 矩阵 144 个非零 9 个自由参数）、两个归纳偏置 |
| 三 | 感受野、stride 与深度 | 感受野公式 1+2L、下采样、为什么必须深 |
| 四 | 五个里程碑 | LeNet → AlexNet → VGG → GoogLeNet → ResNet；参数量与设计思想；ResNet-50 实测 25.6M / 8.2 GFLOPs；bottleneck 的算术 |
| 五 | ResNet 的实验与遗产 | plain vs residual 在 20 / 56 层的实测；BN + 深 plain 网络的梯度爆炸；四样遗产 |
| 六 | 从 CNN 到 ViT | 归纳偏置 vs 数据量；ViT 的结构；patch embedding == 卷积（实测差 1e-6）；一张图多少 token；卷积在多模态里的残余 |
| 七 | 实验 | 代码与结果 |
| 八 | 本文小结 |  |


## 二、卷积作为带约束的线性层

### 1. 定义与两条公式

二维卷积（深度学习里实际是互相关，不翻转核）：输入 $$x \in \mathbb{R}^{C_{in} \times H \times W}$$，核 $$k \in \mathbb{R}^{C_{out} \times C_{in} \times k_h \times k_w}$$，输出的每个位置是核与输入对应窗口的内积：

$$
y_{o, i, j} = \sum_{c=1}^{C_{in}} \sum_{a=0}^{k_h - 1} \sum_{b=0}^{k_w - 1} k_{o, c, a, b}\, x_{c,\, i + a,\, j + b} + \beta_o
$$

**参数量** $$= C_{out} \cdot C_{in} \cdot k_h \cdot k_w + C_{out}$$，与图像大小无关。**FLOPs** $$= 2 \cdot H_{out} \cdot W_{out} \cdot C_{out} \cdot C_{in} \cdot k_h \cdot k_w$$——每个输出位置做一次长度为 $$C_{in} k_h k_w$$ 的内积，与图像大小成正比。这是卷积与全连接层最大的算术区别：全连接层的参数量与 FLOPs 是同一个数（[第一篇](/backpropagation-by-hand.html)），卷积层的 FLOPs 是参数量乘以输出位置数。ResNet-50 有 2560 万参数、每张图 82 亿 FLOPs，比值 320——一个参数平均被用了 320 次。

### 2. 它就是一个稀疏矩阵

对每个输出位置，卷积是输入的一个线性函数；把输入拉直成向量，整个卷积就是一个矩阵乘法 $$y = Mx$$。第七章的实验构造了这个矩阵：一个 $$3 \times 3$$ 核作用在 $$6 \times 6$$ 单通道图像上，输出 $$4 \times 4$$，$$M$$ 是 $$16 \times 36$$：

```text
max |conv - M@x| = 8.9e-16
matrix shape (16, 36), entries 576, nonzero 144, distinct free params 9
row 0 of M (reshaped 6x6):          ← 输出位置 (0,0) 对输入的权重
[[-0.65 -0.13  0.78  0.    0.    0. ]
 [ 1.49 -1.26  1.51  0.    0.    0. ]
 [ 1.35  0.78  0.26  0.    0.    0. ]
 [ 0.    0.    0.    0.    0.    0. ]
 [ 0.    0.    0.    0.    0.    0. ]
 [ 0.    0.    0.    0.    0.    0. ]]
```

576 个矩阵元素里 144 个非零（每行 9 个）、且这 144 个只有 **9 个不同的值**——每一行是同一个核在不同位置。所以卷积 $$=$$ 全连接层 $$+$$ 两条约束：

- **局部性**：每个输出只看输入的一个 $$k \times k$$ 窗口，矩阵稀疏；
- **参数共享**（平移等变性）：所有位置用同一组权重，矩阵的各行是同一个模式的平移。

代一个真实尺寸：ResNet-50 第二阶段一个 $$3 \times 3$$、$$64 \to 64$$ 通道、$$56 \times 56$$ 特征图的卷积，参数 36,864；等价的全连接矩阵是 $$(56 \times 56 \times 64)^2 = 200704^2 \approx 4 \times 10^{10}$$ 个元素——160 GB 的 fp32。两条约束把参数压了六个量级，这是 CNN 在 2012–2016 年能训、MLP 不能的算术原因。

### 3. 约束就是先验

上一篇第五章说显式正则化是"把模型拉向某种简单解的先验"。卷积的两条约束是最强的一种：不是拉向、而是**只允许**满足局部性与平移等变性的解。在图像上这个先验几乎总是对的（一只猫在左上角和右下角是同一只猫），所以数据不多时 CNN 远胜 MLP。但它也是一个上限——长距离的关系（图像两端的两个物体）要靠堆很多层才能看到（第三章），而数据足够多时模型本可以自己学出比"局部 + 平移"更好的先验（第六章）。


## 三、感受野、stride 与深度

### 1. 感受野

第 $$L$$ 层的一个输出位置能"看到"输入的多大范围，叫感受野。stride 为 1 的 $$3 \times 3$$ 卷积每层把感受野扩大 2：$$\text{RF}_L = 1 + 2L$$。5 层是 11，10 层是 21，20 层是 41——要让一个输出看到整张 $$224 \times 224$$ 的图，stride 1 需要 112 层。这就是"深"的最初动机：**感受野随深度线性增长，而任务需要全局信息**。

### 2. Stride 与池化

stride 为 $$s$$ 的卷积（或池化）把特征图缩小 $$s$$ 倍，之后每层的感受野增量乘 $$s$$。ResNet-50 用一个 stride 2 的 $$7 \times 7$$ 卷积和一个 stride 2 的池化把 224 降到 56，之后每个阶段再降一半，最终特征图 $$7 \times 7$$、总下采样 32 倍——50 层的感受野覆盖整张图有余。下采样同时把 FLOPs 降下来（每次减 4 倍）、把通道数升上去（每次乘 2），保持每个阶段的计算量大致相同。这个"空间减半、通道翻倍"的模式从 VGG 沿用到 ResNet，是 CNN 设计的默认节奏。

### 3. 深了就训不动

感受野要求深，深了就遇到[第二篇](/initialization-normalization-and-residual.html)的全部问题：方差衰减或爆炸、梯度连乘。2014 年的 VGG 到 19 层就停了，不是不想更深，是更深训不动。BatchNorm（2015 年初）解决了前向的一半；同年底 ResNet 用残差解决了反向的一半，一口气到 152 层。第五章的实验在 56 层上复现这个转折。


## 四、五个里程碑

### 1. 各解决了什么

| 网络 | 年 | 层数 | 参数 | 引入的东西 | 今天还在用的 |
|---|---|---|---|---|---|
| LeNet-5 | 1998 | 5 | 60K | 卷积 + 池化 + 全连接的范式；反向传播训练 | 范式本身 |
| AlexNet | 2012 | 8 | 60M | ReLU、dropout、GPU 训练、数据增强；ImageNet 上把错误率从 26% 降到 16% | ReLU、GPU、增强 |
| VGG-16 | 2014 | 16 | 138M | 只用 $$3 \times 3$$ 核堆深；两个 $$3 \times 3$$ 的感受野等于一个 $$5 \times 5$$，参数 18 vs 25 且多一个非线性 | 小核堆深；"空间减半通道翻倍" |
| GoogLeNet | 2014 | 22 | 6.8M | $$1 \times 1$$ 卷积做通道降维（bottleneck）；多分支；去掉大全连接层，用全局平均池化 | $$1 \times 1$$ 卷积；全局平均池化 |
| ResNet-50 / 152 | 2015 | 50 / 152 | 25.6M / 60M | 残差连接；BN 到处用；bottleneck 残差块 | **残差、归一化、堆同样的块** |

两个趋势值得看：参数量从 AlexNet 的 60M 到 GoogLeNet 的 6.8M 再到 ResNet-50 的 25.6M——大全连接层被砍掉后（AlexNet 的 60M 里 58M 在最后三个全连接层），参数量不再是深度的函数；层数从 8 到 152，深度成了主要的扩展维度。

### 2. ResNet-50 的账

第七章用 `torchvision` 的 ResNet-50 逐层数了一遍：

```text
params 25.56M  (conv 23.45M, fc 2.05M, bn 0.05M)
FLOPs per image (224x224): conv 8.17 G, fc 0.004 G -> 8.18 GFLOPs (= 4.09 GMACs)
```

文献里常说的"ResNet-50 4.1 GFLOPs"数的是乘加次数（MACs），按本系列"一次乘加 = 2 FLOPs"的约定是 8.2 GFLOPs。参数 92% 在卷积，全连接只有最后 $$2048 \times 1000$$ 一层；BN 的参数可以忽略。

### 3. Bottleneck 块的算术

ResNet-50 的残差块是三层：$$1 \times 1$$ 降维 → $$3 \times 3$$ → $$1 \times 1$$ 升维。以 256 通道的输入为例：

| 设计 | 层 | 参数 |
|---|---|---|
| 两个 $$3 \times 3$$，$$256 \to 256 \to 256$$（ResNet-18/34 的块） | 2 | $$2 \times 256 \times 256 \times 9 = 1.18$$M |
| Bottleneck：$$1 \times 1$$ $$256 \to 64$$，$$3 \times 3$$ $$64 \to 64$$，$$1 \times 1$$ $$64 \to 256$$ | 3 | $$16\text{K} + 37\text{K} + 16\text{K} = 69$$K |

三层比两层少 17 倍参数，因为昂贵的 $$3 \times 3$$ 在 4 倍窄的通道上做。$$1 \times 1$$ 卷积没有空间感受野，它就是**对每个位置独立做一次线性变换**——与 Transformer 里对每个 token 独立做的 FFN 是同一种算子。Transformer 的 FFN 是反过来的 bottleneck（$$d \to 4d \to d$$，先升后降），但"用逐位置的线性层做通道混合、用另一种算子做位置混合"这个分工是共同的：CNN 用 $$3 \times 3$$ 混合空间位置，Transformer 用 attention 混合序列位置。


## 五、ResNet 的实验与遗产

### 1. 退化问题

He 等 2015 的出发点是一个实验事实：在 CIFAR-10 上，56 层的 plain 网络比 20 层的**训练误差更高**——不是过拟合（训练误差高），是优化失败。理论上深网络至少能表示浅网络（多出的层学成恒等），但 SGD 找不到那个解。残差把"学恒等"变成"学零"（$$f(x) = 0$$ 时块就是恒等），找到就容易了。

### 2. 实测：20 层与 56 层

第七章在 MNIST（stride 2 到 $$14 \times 14$$，16 通道，每块一个 $$3 \times 3$$ 卷积 + BN，plain 或残差）上复现：

| 深度 | 结构 | 初始梯度范数：第 1 块 / 第 $$L$$ 块 | 比值 | 训练 loss @ 1 epoch | @ 3 epoch | 测试准确率 |
|---|---|---|---|---|---|---|
| 20 | plain | 1.7 / 0.16 | 11 | 0.460 | 0.128 | 96.4% |
| 20 | residual | 1.3 / 0.37 | 3.4 | 0.224 | 0.093 | 96.8% |
| 56 | plain | **950 / 0.33** | **2849** | 1.676 | **0.724** | **70.9%** |
| 56 | residual | 2.2 / 0.76 | 2.9 | 0.268 | 0.129 | 96.1% |

两个观察。第一，**退化问题复现了**：plain 网络从 20 层到 56 层，训练 loss 从 0.13 恶化到 0.72；残差网络 0.09 与 0.13，深了没有变坏。第二，**梯度爆炸的方向**：plain-56 第 1 块的梯度范数是第 56 块的 2849 倍——是靠近输入的层梯度大，不是小。这与第二篇"没有归一化时梯度消失"相反：有了 BN 的深 plain 网络，梯度反向穿过每个 BN 时被输入方差归一化放大，越靠输入越大（Yang 等 2019 从平均场理论证明了 BN 在深 plain 网络里必然导致梯度爆炸）。BN 修了前向，把反向的问题换了一个方向。残差网络的比值在 3 左右——恒等通路把它压平了。

### 3. 四样遗产

ResNet 留给 Transformer 的不只是残差：

| 遗产 | ResNet 里 | Transformer 里 |
|---|---|---|
| 残差连接 | $$x + f(x)$$，每块 | $$x + \text{Attn}(\cdot)$$，$$x + \text{FFN}(\cdot)$$，每层两次 |
| 归一化到处用 | 每个卷积后一个 BN | 每个子层一个 LayerNorm / RMSNorm |
| Pre-activation | ResNet-v2（He 等 2016）把 BN 与 ReLU 移到卷积**之前**、残差相加之后不再有非线性，更深更稳 | **Pre-Norm**（第二篇第六章）——同一个想法在一年后被 Transformer 采用 |
| 堆同样的块 | 一个块的设计定好，重复 $$N$$ 次，只改深度与宽度 | 一层的设计定好，重复 $$L$$ 次；scaling 只动 $$L$$、$$d$$、$$d_{ff}$$ |

第三条最少被提到、也最有意思：Transformer 的 Pre-Norm 在 ResNet 这条线上已经被发现过一次。


## 六、从 CNN 到 ViT

### 1. 归纳偏置与数据量

第二章说卷积的两条约束是先验。先验的价值与数据量反相关：数据少时先验替你排除了大量错误假设，数据多时数据本身足以排除它们，先验只剩限制。Dosovitskiy 等 2020 的 ViT 论文把这个关系做成了一个干净的实验：只用 ImageNet-1k（130 万张）训练，ViT 不如同规模的 ResNet；用 JFT-300M（3 亿张）预训练再迁移，ViT 反超，且模型越大差距越大。CNN 的先验在 100 万张图上是资产，在 3 亿张图上是负债。

### 2. ViT 的结构

ViT 对 Transformer encoder **几乎没有改动**，改的只是输入：

```text
图像 [3, 224, 224]
  │  切成 16×16 的 patch，共 (224/16)² = 196 个
  ▼
patch 序列 [196, 3·16·16 = 768]
  │  线性投影到 d = 768（"patch embedding"）；前面拼一个 [CLS] token；加可学习的位置编码
  ▼
token 序列 [197, 768]
  │  标准 Transformer encoder × 12 层（Pre-Norm，MHA + FFN，与 BERT 相同）
  ▼
[CLS] 的输出 → 分类头
```

没有卷积、没有池化、没有"空间减半通道翻倍"。一张图就是 196 个 token，之后的每一步与处理 196 个词完全相同。图像的二维结构只通过位置编码告诉模型——模型要自己从数据里学出"相邻 patch 相关"这件事，这就是它需要更多数据的原因。

### 3. Patch embedding 就是一个卷积

"切 patch + 线性投影"这一步，等价于一个 kernel 与 stride 都等于 patch 大小的卷积。第七章用 PyTorch 验证：`nn.Conv2d(3, 768, kernel_size=16, stride=16)` 与"`unfold` 切 patch → 矩阵乘"的输出差 $$1.2 \times 10^{-6}$$（fp32 的求和顺序差异）。参数 $$3 \times 16 \times 16 \times 768 + 768 = 590{,}592$$，两种写法完全相同。所以 ViT 保留了卷积的一个影子：只在输入那一层、只做一次、窗口互不重叠。之后的 Transformer 层里，任意两个 patch 之间的路径长度是 1——CNN 要堆几十层才有的全局感受野，attention 第一层就有。

### 4. 一张图等于多少 token

token 数 $$= (H / p) \times (W / p)$$，由分辨率与 patch 大小决定，与模型大小无关：

| 配置 | 分辨率 | patch | token 数 |
|---|---|---|---|
| ViT-B/16、ViT-L/16 | 224 | 16 | 196 |
| ViT-L/14、ViT-H/14 | 224 | 14 | 256 |
| CLIP ViT-L/14-336（LLaVA-1.5 用） | 336 | 14 | 576 |
| 原生分辨率 ViT（Qwen2-VL 一类）在 1024² 上 | 1024 | 14 | 5476（merge 前） |

这张表是 L7 多模态的入口：VLM 把 ViT 的输出 token 送进 LLM，一张图占多少上下文、多少 KV cache，从这里开始算——[04 系列第八篇](/multimodal-vision-encoder-cost-and-image-token-kv.html)把这笔账算完了。attention 的算量随 token 数平方增长，所以高分辨率图像要么用更大的 patch、要么在 encoder 后合并 token（2×2 merge）、要么用窗口 attention——三种办法都在那一篇。

### 5. 卷积的残余

卷积没有消失，它退到了几个特定位置：

- **ViT 的 patch embedding**：如上；Xiao 等 2021 发现把这一个大卷积换成几个小卷积堆叠（"early convolutions"）能让 ViT 更好训、对超参数更稳；
- **语音的前端**：Whisper 的 encoder 在 Transformer 之前有两层一维卷积（第二层 stride 2），把 mel 频谱下采样一半再进 attention；
- **扩散模型的 U-Net**：Stable Diffusion 1.x / 2.x 的去噪网络是卷积 U-Net 加 attention 层；DiT（扩散 Transformer）之后才被 patch 化的 Transformer 取代；
- **ConvNeXt**（Liu 等 2022）：把 ResNet 按 Transformer 的设计（大核、更少的归一化、GELU、倒 bottleneck）现代化之后，纯卷积网络在 ImageNet 上追平了 Swin Transformer——说明 ViT 的优势有相当一部分来自训练配方而非结构本身。

对算法工程师，卷积今天要懂到的程度是：知道它是带约束的线性层、会算参数与 FLOPs、知道 patch embedding 是它、能读懂 encoder 前端的几层。设计新的 CNN 骨干不再是主流工作。


## 七、实验

### 1. 代码

四个实验，前两个纯 NumPy，后两个用 PyTorch（`torchvision` 只用来加载 ResNet-50 的结构）：

```python
# 1. 卷积 == 稀疏矩阵
def conv_as_matrix(k, H, W):            # 构造 (Ho*Wo) x (H*W) 的矩阵 M，使 conv(x,k).flatten() == M @ x.flatten()
    ...
# 2. ResNet-50 逐层数参数与 FLOPs：对每个 Conv2d / Linear 注册 forward hook，FLOPs = 2 * 输出元素数 * (C_in * kh * kw)
# 3. plain vs residual：stem(stride 2) + L × [3x3 conv, BN]，forward 里一行切换 h = relu(h + out) / relu(out)
# 4. patch embedding：nn.Conv2d(3, 768, 16, stride=16) 对比 x.unfold(...) @ W.T
```

### 2. 结果

实验 1、2、4 的输出已在第二、四、六章引用。实验 3 的完整输出（MNIST 前 2 万张，3 个 epoch，SGD momentum 0.9、$$\eta = 0.05$$、裁剪 1.0，测试前用 1 万张训练图重估 BN 统计量）：

```text
L=20 plain   : init grad norm block1 1.7e+00 vs block20 1.6e-01 (ratio 10.9)  | train loss @ep1 0.460 @ep3 0.128 | test acc 96.4%
L=20 residual: init grad norm block1 1.3e+00 vs block20 3.7e-01 (ratio 3.4)   | train loss @ep1 0.224 @ep3 0.093 | test acc 96.8%
L=56 plain   : init grad norm block1 9.5e+02 vs block56 3.3e-01 (ratio 2849)  | train loss @ep1 1.676 @ep3 0.724 | test acc 70.9%
L=56 residual: init grad norm block1 2.2e+00 vs block56 7.6e-01 (ratio 2.9)   | train loss @ep1 0.268 @ep3 0.129 | test acc 96.1%
```

四个实验里前三个在笔记本 CPU 上几秒钟，实验 3 的四个配置约 9 分钟（56 层的两个占了三分之二）。

### 3. 值得自己动手的扩展

- 把实验 3 的 BN 去掉、给残差分支零初始化（$$f(x) = 0$$，块在初始时刻是恒等），看没有 BN 的残差网络能否训——这是 Fixup / SkipInit 一类工作的起点；
- 用 `conv_as_matrix` 构造 stride 2 或 padding 的卷积矩阵，看稀疏模式怎么变；
- 把 ViT 的 patch 从 16 改到 8，token 数变 4 倍，用 04 系列第二篇的公式算 attention 的 FLOPs 变了多少倍。


## 八、本文小结

- **卷积是带两条约束的线性层**：局部性（矩阵稀疏）与参数共享（各行是同一核的平移）。$$3 \times 3$$ 核在 $$6 \times 6$$ 图上是一个 $$16 \times 36$$ 矩阵、144 个非零、9 个自由参数；ResNet-50 一个卷积层的等价全连接矩阵有 $$4 \times 10^{10}$$ 个元素。参数量与图像大小无关，FLOPs 与之成正比（ResNet-50：25.6M 参数、8.2 GFLOPs、每个参数用 320 次）。
- **深度是为了感受野**：$$1 + 2L$$，stride 与池化把它乘上去；深了就遇到第二篇的全部问题。VGG 止于 19 层，BN 修前向，残差修反向，ResNet 到 152 层。
- 五个里程碑各留下一样东西：ReLU 与 GPU（AlexNet）、小核堆深（VGG）、$$1 \times 1$$ 卷积与全局池化（GoogLeNet）、残差 + BN + 堆同样的块（ResNet）。Bottleneck 块用 $$1 \times 1$$ 降维把参数压 17 倍；$$1 \times 1$$ 卷积就是逐位置的线性层，与 Transformer 的 FFN 同类。
- **退化问题复现**：plain 网络 20 → 56 层训练 loss 从 0.13 恶化到 0.72，残差网络不变；有 BN 的深 plain 网络梯度**向输入方向爆炸**（第 1 块是第 56 块的 2849 倍），残差把比值压到 3。ResNet-v2 的 pre-activation 是 Pre-Norm 的前身。
- **ViT**：数据足够多时卷积的先验成为负担；把图切成 $$(H/p)(W/p)$$ 个 patch、线性投影成 token、送进标准 encoder。**Patch embedding 就是 kernel = stride = $$p$$ 的卷积**（实测差 $$10^{-6}$$，590,592 个参数）。一张 224 图是 196 个 token，CLIP-336 是 576，原生分辨率 1024² 是 5476——这是多模态成本的起点。
- 卷积退到了 patch embedding、语音前端、U-Net 与 ConvNeXt；懂到"带约束的线性层 + 会算账 + 认得出 patch embedding"即可。
- 下一篇：另一条线——循环网络怎么处理序列、为什么记不住远处、attention 如何从它的瓶颈里被发明出来。

配套代码：[`deep-learning-foundations/05_cnn.py`](https://github.com/arganzheng/ai-learning-labs/blob/main/deep-learning-foundations/05_cnn.py)——`matrix` / `resnet` / `patch` / `deep` 四个子实验，`deep`（L=20 / 56 的 plain 与 residual）在 CPU 上约 10 分钟。


## 下一篇

[RNN：从 LSTM 到 attention 的诞生](/rnn-lstm-and-the-birth-of-attention.html)
