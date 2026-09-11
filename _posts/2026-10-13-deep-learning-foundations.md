---
layout: post
title: 深度学习基础：从反向传播到残差（总纲）
subtitle: "Deep Learning Foundations: From Backpropagation to Residual Connections"
tags: [AI, Deep Learning, LLM]
catalog: true
---


## 内容简介

《深度学习基础：从反向传播到残差》是一组共六篇的系列文章，对应[《AI 算法工程师学习地图》](/ai-algorithm-engineer-learning-roadmap.html)的第 L3 层。它面向已经有 [L0 数学](/math-for-ai-algorithm-engineers.html)、[L1 工具](/tooling-for-ai-algorithm-engineers.html)、[L2 经典机器学习](/classical-machine-learning-in-the-llm-era.html)基础、准备进入 Transformer 与 LLM 的读者，讲的是**训练一个深度神经网络时会发生什么、为什么、以及怎么算出来**。

它回答的问题是：

> **一个几十层甚至上百层的网络，为什么能训、什么时候不能训、训不动的时候该看哪个数字？**

深度学习教材通常按"模型"组织：感知机、MLP、CNN、RNN、Transformer。这个系列按**训练现象**组织：梯度怎么流、为什么会消失或爆炸、优化器在做什么、为什么参数比样本多却不过拟合、卷积与循环各解决了什么又败在哪里。每一个现象都用三步处理——**推导**（公式从哪来）、**算账**（代到真实网络是多少）、**实验**（几十行 NumPy / PyTorch 在 CPU 上复现它）——最后指出它在 Transformer 与 LLM 里的形态。

举一个例子说明这个系列的取法。"残差连接让深网络能训"是每本教材都有的一句话；本系列要把它变成三件可验证的事：（一）推导：没有残差时第 $$l$$ 层的梯度是 $$l$$ 个 Jacobian 的乘积，任何一个的谱范数系统性地偏离 1 就指数级放大或缩小；有残差时每层的 Jacobian 变成 $$I + J_l$$，乘积里多出一条恒等通路。（二）算账：一个 128 层、每层 Jacobian 谱范数 0.9 的网络，最底层梯度是顶层的 $$0.9^{128} \approx 1.4 \times 10^{-6}$$；加残差后不再有这个因子。（三）实验：用 NumPy 搭一个 64 层 MLP，画出有无残差时各层梯度范数的分布，两条曲线差六个数量级。Transformer 的每一层都靠这条恒等通路，Pre-Norm 与 Post-Norm 的全部争论也在这条通路上。

系列覆盖的范围可以概括为五个训练现象与两段历史：

```text
现象一   梯度怎么流                反向传播、链式法则、激活存储、反向为什么是前向的两倍       → 第一篇
现象二   为什么深了就难训          方差在层间的传播、初始化、归一化、残差、Pre/Post-Norm     → 第二篇
现象三   优化器在做什么            SGD、Momentum、Adam / AdamW、学习率调度、梯度裁剪        → 第三篇
现象四   为什么不过拟合、何时会    正则化、dropout、weight decay、double descent、epoch 数    → 第四篇
历史一   卷积解决了什么             参数共享、感受野、LeNet → ResNet、ViT 把图切成 token        → 第五篇
历史二   循环解决了什么、败在哪    RNN、时间反传、LSTM 门控、seq2seq、attention 的诞生        → 第六篇
```


## 为什么写这个系列？

### LLM 时代的深度学习基础没有变少，只是换了名字

Transformer 之后，"深度学习基础"常被当成历史课跳过。但打开任何一份预训练技术报告，遇到的仍是这一层的词：loss spike 与梯度裁剪、warmup 步数、AdamW 的 $$\beta_2$$、weight decay 系数、RMSNorm 与 Pre-Norm、初始化的标准差、$$\mu$$P、z-loss、QK-norm。每一个都是本系列某一篇的主题在大模型上的形态。不懂方差传播，就不理解为什么 Llama 的初始化标准差是 0.02 而 GPT-2 的残差分支要额外除以 $$\sqrt{2L}$$；不懂 Adam 的二阶矩，就不理解 warmup 为什么必须；不懂 dropout 的期望等价，就不理解为什么 LLM 预训练几乎不用它。

### 从"会调用"到"会诊断"

L1 导读教的是二十行训练循环。写出它不难，难的是它跑起来之后 loss 不降、突然爆成 NaN、降到一半开始震荡——每种情况该看哪个数字、改哪个开关。这需要知道每个开关背后的公式：梯度范数曲线说明什么、学习率与 batch 的 scaling 规则从哪来、梯度裁剪剪的是什么。这个系列想给的就是这套诊断能力，而不是又一份 API 说明。

### 现有材料的断层

- **教材**（Goodfellow 等《Deep Learning》、Bishop《Deep Learning》）完整但厚，且成书时 Transformer 尚未成为主线，很多现象没有对到 LLM 上；
- **课程**（CS231n、fast.ai）讲直觉与代码，很少把一个现象推到能算出数字；
- **论文**每篇只讲一个方法（Kaiming 初始化、LayerNorm、AdamW），假设读者能自己把它们放进同一张图；
- **博客**大多止于"残差让深网络能训"这一句，不推、不算、不测。

本系列取的是中间那段路：每个现象都推到公式、算到数字、测到曲线，篇幅控制在教材一章的三分之一。


## 适合哪些读者？

### 准备进入 L4 Transformer / LLM 的算法学习者

你已经能写训练循环，接下来要读 Llama、DeepSeek 的技术报告与 [《Transformer 与 LLM》](/transformer-and-llm-for-infra-engineers.html)系列。本系列是那之前的最后一层基础：读完之后，报告里关于初始化、归一化、优化器、稳定性的每一段都有对应的公式可查。

### 做后训练但基础是"跳过来"的工程师

你在用 `trl` 做 SFT / DPO，能跑通，但 loss 曲线怪的时候不知道往哪看。第一至四篇是为此准备的：梯度、稳定性、优化器、正则化，各自的诊断信号与开关。

### 需要回顾 CNN / RNN 的多模态或序列建模学习者

L7 多模态的 vision encoder 来自 ViT，ViT 来自 CNN 的 patch 化；语音模型里仍有卷积前端与循环结构。第五、六篇把这两条线讲到"它们留给 Transformer 什么遗产"为止。

### Infra 工程师，想知道自己在优化的训练过程里发生了什么

你在做训练框架、checkpoint、容错，看到梯度裁剪、混合精度、optimizer state 这些词，想知道它们在算法上的含义。第一篇（反向为什么是前向两倍、激活为什么要存）与第三篇（每参数 8 字节优化器状态从哪来）直接回答。


## 系列的整体主线

六篇按"先建立梯度的流动，再看它为什么会坏、怎么修，再看用梯度更新参数的方法与它的副作用，最后回看两段历史"的顺序推进：

```text
第一篇：反向传播 —— 手推两层网络，梯度的形状规则，反向 = 2 × 前向，激活为什么要存
        ↓
第二篇：初始化、归一化与残差 —— 方差怎么在层间传播，深网络为什么难训，残差与 Pre-Norm 怎么修
        ↓
第三篇：优化器 —— 从 SGD 到 AdamW，学习率调度，梯度裁剪，batch 与学习率的 scaling
        ↓
第四篇：正则化与泛化 —— dropout、weight decay、早停，参数远多于样本为什么不过拟合，何时会
        ↓
第五篇：CNN —— 卷积作为参数共享，感受野，LeNet → ResNet 的遗产，ViT 把图切成 token
        ↓
第六篇：RNN —— 时间上的反传与梯度消失，LSTM 的门，seq2seq 的瓶颈，attention 的诞生
```

前四篇是**训练动力学**：一个网络从初始化到收敛，梯度经历了什么、参数怎么动、什么时候停。后两篇是**结构史**：Transformer 之前的两条主线各解决了什么、留下了什么——残差与归一化来自 CNN 这条线，attention 来自 RNN 这条线。读完第六篇，Transformer 的每一个组件都有了来历，L4 的 04 系列可以直接接上。

三条交织的线索：

```text
推导线：链式法则 → Jacobian 乘积与谱范数 → 更新量的量级 → 期望等价与先验 → 卷积的线性算子形式 → 时间反传
数字线：反向 FLOPs = 2 × 前向 → 0.9^128 → 每参数 8 字节 → 一个 epoch 就够 → ResNet-50 的 4.1 GFLOPs → RNN 不能并行
LLM 线：6ND · 激活重算 → RMSNorm · Pre-Norm · 0.02 → AdamW · warmup · 裁剪到 1.0 → 预训练不用 dropout → ViT / patch → attention → Transformer
```

每一篇都用同样的方法：**推导公式，代入真实网络算出数字，用几十行代码复现现象，指出它在 LLM 里的形态**。


## 章节结构与分章导读

### 1. 反向传播：手推一个两层网络

第一篇建立整个系列的对象：梯度。它把反向传播从"框架自动做的事"变成可以手推、手算、手写的东西。

这一篇会覆盖：

- 计算图与链式法则：标量对向量、向量对向量（Jacobian）、标量对矩阵的导数；反向传播就是从 loss 出发沿计算图反向逐节点乘 Jacobian；
- 矩阵求导的形状规则：$$Y = XW$$ 时 $$\partial L / \partial W = X^T (\partial L / \partial Y)$$、$$\partial L / \partial X = (\partial L / \partial Y) W^T$$——记住"梯度与被求导的量形状相同"，两条公式可以直接推出来；
- 逐层手推一个两层 MLP（Linear → ReLU → Linear → softmax → 交叉熵）的前向与反向，每一步写出形状；softmax + 交叉熵的梯度 $$p - y$$（L0 导读推过，这里放进完整网络）；
- 反向为什么是前向的两倍：每个 Linear 层反向要做两个矩阵乘法（对输入的梯度、对权重的梯度），前向只做一个；由此得到训练 FLOPs $$\approx 6ND$$——04 系列第二篇引用的这个数字在这里推出来；
- 激活为什么要存：反向计算 $$\partial L / \partial W$$ 需要前向时的输入 $$X$$，所以前向的中间结果必须保留到反向；激活显存与 batch、序列长度、层数成正比，与参数量无关；激活重算（gradient checkpointing）用一次额外前向换掉这份存储；
- 用有限差分验证手推的梯度：梯度检查的方法与精度标准；
- Autograd 做了什么：动态图记录、每个算子的 backward 函数、叶子节点的 `.grad` 累加——只到理解框架行为的程度，实现在 Infra 地图 03 系列第三篇。

核心问题是：

> **不用框架，能不能手推并手写一个两层网络的反向传播，用有限差分验证到 $$10^{-6}$$？能不能由此说出为什么训练 FLOPs 是 $$6ND$$、为什么激活要存？**

实验：纯 NumPy 实现两层 MLP 的前向、反向与梯度检查；在 MNIST 规模的数据上训到 97% 以上；统计前向与反向的浮点运算次数，验证 2 倍关系。这份代码是后面各篇实验的基座。

### 2. 训练为什么不稳定：初始化、归一化与残差

第二篇讲深网络最核心的问题：信号与梯度经过很多层之后为什么会消失或爆炸，以及三十年里发展出的三种修法。

这一篇会覆盖：

- 方差的传播：一层 $$y = Wx$$ 的输出方差是输入方差的 $$n_{in} \cdot \text{Var}(w)$$ 倍；要让方差在层间不变，$$\text{Var}(w) = 1/n_{in}$$——这是 Xavier 初始化的推导；ReLU 砍掉一半，所以 Kaiming 初始化是 $$2/n_{in}$$；反向传播里同一个推导给出 $$1/n_{out}$$；
- 梯度作为 Jacobian 的乘积：$$L$$ 层网络最底层的梯度是 $$L$$ 个 Jacobian 的乘积，其谱范数的几何平均偏离 1 就指数级放大或缩小；$$0.9^{128} \approx 1.4 \times 10^{-6}$$、$$1.1^{128} \approx 2 \times 10^{5}$$；
- 归一化：BatchNorm（沿 batch 归一）、LayerNorm（沿特征归一）、RMSNorm（去掉均值中心化）；三者的公式、参数量、计算量；BatchNorm 为什么在序列模型里不好用（batch 内统计量依赖 padding 与序列长度、推理时要用 running statistics）；RMSNorm 比 LayerNorm 省了什么；
- 残差连接：每层的 Jacobian 从 $$J_l$$ 变成 $$I + J_l$$，乘积展开后有一条恒等通路，梯度不再随深度指数衰减；残差流的方差随层数线性增长，所以 GPT-2 把残差分支的初始化再除以 $$\sqrt{2L}$$，深层模型要把这个因子算进去；
- Pre-Norm 与 Post-Norm：归一化放在残差分支之前还是之后，对梯度路径的影响；Post-Norm 效果略好但需要 warmup 且深了难训，Pre-Norm 稳定、当前 LLM 几乎全用它；
- 大模型上的稳定性工具：loss spike 的常见来源、QK-norm、z-loss、$$\mu$$P 的动机（让超参数在不同宽度下可迁移）——只到概念，配方细节在 L4；
- 诊断：看每层激活的方差、每层梯度的范数分布，判断是初始化、归一化还是学习率的问题。

核心问题是：

> **一个 64 层的 MLP 不加任何技巧为什么训不动？初始化、归一化、残差三样东西各修了哪一段，缺了哪一样会怎样？**

实验：用第一篇的 NumPy 基座搭一个 64 层 MLP，画出四种配置（随机初始化 / Kaiming / Kaiming + LayerNorm / Kaiming + LayerNorm + 残差）下各层激活方差与梯度范数随深度的曲线；观察 Pre-Norm 与 Post-Norm 在同一网络上的差别。

### 3. 优化器：从 SGD 到 AdamW 与学习率调度

第三篇讲有了梯度之后怎么更新参数。它把优化器的每一项拆开，解释各自在解决什么问题、带来什么状态与代价。

这一篇会覆盖：

- SGD 与它的噪声：mini-batch 梯度是全量梯度的无偏估计，方差与 $$1/B$$ 成正比；学习率与 batch 的 scaling 规则（线性 scaling、平方根 scaling）的来源与适用范围；
- Momentum：梯度的指数移动平均，在一致的方向上加速、在震荡的方向上抵消；Nesterov 的修正；
- Adam：一阶矩 $$m$$ 与二阶矩 $$v$$，更新量 $$m / \sqrt{v}$$ 让每个参数有自己的有效学习率；偏差修正为什么必须；$$\beta_1 = 0.9$$、$$\beta_2 = 0.95$$（LLM 常用，而不是默认的 0.999）各自意味着多长的记忆窗；$$\epsilon$$ 的作用；
- AdamW 与 $$L_2$$ 正则的区别：在 loss 里加 $$\frac{\lambda}{2}\|w\|^2$$ 会被 $$1/\sqrt{v}$$ 缩放，而 decoupled weight decay 直接对参数衰减；两者在 Adam 下不等价的推导；LLM 常用 $$\lambda = 0.1$$；
- 优化器状态的账：Adam 每参数两个 fp32 状态 8 字节，加 fp32 主权重 4 字节，是 L1 导读"16 字节 / 参数"里的 12；8-bit Adam、Adafactor 等减状态的方法各省多少；
- 学习率调度：warmup 为什么必须（$$v$$ 的估计在早期不可靠、更新量过大）、cosine 衰减、WSD（warmup-stable-decay）、衰减到峰值的 10% 还是 0；典型的峰值学习率量级（$$3 \times 10^{-4}$$ 级别的预训练、$$10^{-5}$$ 级别的 SFT）与它们和模型宽度的关系；
- 梯度裁剪：按全局范数裁剪到 1.0 在做什么，为什么它是 loss spike 的第一道防线；梯度范数曲线怎么读；
- 二阶方法与新优化器：Shampoo、Muon、SOPHIA 一类的动机（用更多曲率信息换更少步数），到知道它们存在、知道它们与 Adam 的差别在哪即可。

核心问题是：

> **Adam 的两个矩各在做什么？为什么 AdamW 与在 loss 里加 $$L_2$$ 不一样？warmup 为什么在 Adam 下几乎不能省？**

实验：在同一个小网络上实现 SGD、Momentum、Adam、AdamW，画训练曲线对比；关掉 warmup 与偏差修正观察前几百步；用不同 batch 验证学习率 scaling 规则的适用范围。

### 4. 正则化与泛化：为什么参数比样本多却不过拟合

第四篇讲深度学习里最反直觉的现象——参数量远大于样本量的网络照样泛化——以及什么时候这个好运会用完。

这一篇会覆盖：

- 经典视角回顾（L2 导读）：容量、偏差 - 方差、过拟合；它在深网络上失效的地方——参数量早已超过样本量，按经典理论应该严重过拟合；
- double descent：测试误差随模型容量先降、再升、越过插值阈值后再降；隐式正则化——SGD 倾向找到平坦、低范数的解；
- 显式正则化：weight decay（作为高斯先验、作为有效学习率的调节）；dropout（训练时随机置零、推理时按期望缩放，等价于集成大量子网络）；早停（等价于对训练轨迹的约束）；数据增强；label smoothing；
- 为什么 LLM 预训练几乎不用 dropout：数据量远大于模型能记住的量、每个样本只见一次（一个 epoch），过拟合不是主要风险；weight decay 仍保留，主要作用变成控制参数范数与稍稍改善优化；
- 什么时候过拟合回来：SFT 几千条数据训多个 epoch 后的逐字记忆；奖励模型的过拟合与 reward hacking（L2 导读）；多 epoch 预训练在数据受限时的收益递减（Muennighoff 等 2023：重复到 4 个 epoch 以内几乎无损，之后收益迅速下降）；
- 泛化的诊断：训练 loss 与验证 loss 的差、验证 loss 何时开始回升、记忆检测（模型能否逐字复述训练样本）。

核心问题是：

> **一个参数量是样本量一千倍的网络为什么不过拟合？同一个网络在什么数据规模、多少个 epoch 之后会开始过拟合？**

实验：在小数据集上扫模型宽度，画出 double descent 曲线；在同一网络上对比无正则化 / weight decay / dropout 的训练 - 验证 loss 差；用一个小语言模型对比 1 epoch 与 10 epoch 的记忆程度。

### 5. CNN：从 LeNet 到 ResNet，再到 ViT

第五篇讲卷积网络——它解决了什么问题、演化到 ResNet 留下了什么、以及 ViT 怎么把它的输入方式带进了 Transformer。

这一篇会覆盖：

- 卷积作为带约束的线性层：局部连接与参数共享，一个 $$3 \times 3 \times C_{in} \times C_{out}$$ 的卷积核相当于一个被强约束的全连接矩阵；参数量与 FLOPs 的公式，代入 ResNet-50 得到 25.6M 参数、约 4.1 GFLOPs（224×224，单次前向的乘加计）；
- 感受野与层数：感受野随层数线性增长，要看到整张图需要很多层，这是"深"的最初动机；池化与 stride；
- 五个里程碑各解决了什么：LeNet（卷积 + 池化的范式）、AlexNet（ReLU、dropout、GPU 训练）、VGG（小核堆深）、GoogLeNet（多分支与 1×1 卷积）、ResNet（残差让 152 层能训）；每一步的参数量与深度；
- ResNet 留给 Transformer 的三样东西：残差连接、归一化（BatchNorm → LayerNorm）、"堆同样的块"这个设计原则；
- 卷积的归纳偏置——平移等变性、局部性——在数据足够多时不再是优势；ViT 把图切成 16×16 的 patch、每个 patch 线性投影成一个 token、加位置编码后送进标准 Transformer；224×224 的图是 196 个 token；ViT 在小数据上不如 CNN、在大数据上反超，这是"归纳偏置 vs 数据量"的一个干净实验；
- 卷积在多模态里的残余：ViT 的 patch embedding 本身就是一个 stride 等于 kernel 的卷积；语音模型的卷积前端；扩散模型的 U-Net。

核心问题是：

> **一个 3×3 卷积核相当于多大的全连接矩阵？ResNet 的残差与 Transformer 的残差是同一个东西吗？ViT 为什么可以不用卷积？**

实验：用 NumPy 实现二维卷积并验证它等价于一个稀疏的全连接矩阵；用 PyTorch 在 CIFAR-10 上对比一个 plain 深网络与加残差的同深度网络；实现 patch embedding，验证它与 stride 卷积输出一致。

### 6. RNN：从 LSTM 到 attention 的诞生

第六篇讲循环网络这条线。它是 attention 的直接来源：理解 RNN 的失败，才理解 Transformer 赢在哪。

这一篇会覆盖：

- 序列建模的问题设定：变长输入、共享参数、依赖历史；RNN 的状态更新 $$h_t = f(W h_{t-1} + U x_t)$$；
- 时间上的反向传播（BPTT）：梯度沿时间回传是同一个矩阵 $$W$$ 的连乘，第二篇的 Jacobian 乘积在时间维上的版本；梯度消失让 RNN 记不住 20 步之外的东西；梯度爆炸靠裁剪（梯度裁剪最初就是为 RNN 发明的）；
- LSTM 与 GRU：用门控让状态可以"加法式"地穿过时间，$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t$$ 里的 $$f_t \odot c_{t-1}$$ 是残差在时间上的形态；门的参数量；
- seq2seq 与它的瓶颈：encoder 把整个源序列压进一个固定长度的向量，decoder 从它出发生成；长句子上质量下降，因为一个向量装不下；
- attention 的诞生（Bahdanau 等 2014）：decoder 每一步对 encoder 的所有隐状态算一个权重、加权求和，绕过固定向量瓶颈；这就是 $$\text{softmax}(q^T k) v$$ 的最初形式，只是 $$q, k, v$$ 还没有分开命名；
- RNN 的两个致命缺点与 Transformer 的回答：无法并行（$$h_t$$ 依赖 $$h_{t-1}$$，序列长度决定串行步数）与长依赖衰减（路径长度 $$O(n)$$）；self-attention 把任意两个位置的路径长度变成 $$O(1)$$、把序列维完全并行，代价是 $$O(n^2)$$ 的算量与 KV——04 系列的一切从这里开始；
- RNN 的回声：状态空间模型（Mamba 一类）与线性 attention 试图找回 RNN 的 $$O(n)$$ 推理成本，知道它们在权衡什么即可。

核心问题是：

> **RNN 为什么记不住 20 步之外的东西？LSTM 的遗忘门与残差连接是什么关系？attention 最初是为了解决什么问题被发明的，它又为什么最终取代了发明它的 RNN？**

实验：用 NumPy 实现一个 RNN 与 LSTM，在"记住第一个 token 并在 $$T$$ 步后输出"的任务上扫 $$T$$，画出两者能记住的最长距离；实现 Bahdanau attention 并可视化对齐矩阵；测量 RNN 与 self-attention 在同一序列长度下的前向时间随长度的变化。


## 贯穿全系列的实践线

本系列的贯穿物是**一份几百行的 NumPy 小框架和一组建立在它之上的实验**。框架从第一篇的两层 MLP 开始，每篇加一点：

```text
第一篇    Linear · ReLU · softmax-CE 的前向与反向；梯度检查；FLOPs 计数
第二篇    深层堆叠；Kaiming 初始化；LayerNorm / RMSNorm；残差块；逐层方差与梯度范数统计
第三篇    SGD · Momentum · Adam · AdamW；学习率调度；梯度裁剪
第四篇    dropout；weight decay；训练 - 验证曲线；宽度扫描
第五篇    Conv2d 与它的全连接等价形式；patch embedding（PyTorch 对照 CIFAR-10）
第六篇    RNN 单元 · LSTM 单元 · BPTT；Bahdanau attention
```

全部实验在 CPU 上几分钟内跑完，没有 GPU 不影响。框架的目的不是替代 PyTorch，而是让每一个训练现象都能在自己写的、每一行都懂的代码里复现一次；之后回到 PyTorch，`loss.backward()` 与 `optimizer.step()` 就不再是黑盒。

与它平行的源码与资料阅读线：

```text
第一篇    Rumelhart 等 1986（反向传播）· Karpathy micrograd · PyTorch autograd 文档的 "How autograd encodes the history"
第二篇    Glorot & Bengio 2010（Xavier）· He 等 2015（Kaiming）· Ioffe & Szegedy 2015（BatchNorm）· Ba 等 2016（LayerNorm）· Zhang & Sennrich 2019（RMSNorm）· Xiong 等 2020（Pre-LN）
第三篇    Kingma & Ba 2014（Adam）· Loshchilov & Hutter 2017（AdamW）· Goyal 等 2017（线性 scaling）· Hu 等 2024（MiniCPM，WSD）
第四篇    Srivastava 等 2014（dropout）· Zhang 等 2017（Understanding deep learning requires rethinking generalization）· Nakkiran 等 2019（double descent）· Muennighoff 等 2023（数据受限的 scaling）
第五篇    LeCun 等 1998（LeNet）· Krizhevsky 等 2012（AlexNet）· He 等 2015（ResNet）· Dosovitskiy 等 2020（ViT）
第六篇    Hochreiter & Schmidhuber 1997（LSTM）· Sutskever 等 2014（seq2seq）· Bahdanau 等 2014（attention）· Vaswani 等 2017（Transformer）第 1–2 节
```


## 阅读路径建议

### 完整学习路径

```text
1 → 2 → 3 → 4 → 5 → 6
```

### 只为进入 Transformer / LLM 做准备

```text
1 → 2 → 3 → 6
```

梯度、稳定性、优化器是读技术报告的必需；第六篇给出 attention 的来历。第四篇在做后训练时补读，第五篇在做多模态时补读。

### 做后训练，loss 曲线不对时不知道看哪

```text
1 → 3 → 4 → 2
```

先懂梯度与优化器（学习率、warmup、裁剪是最常动的开关），再看正则化（SFT 多 epoch 的过拟合），最后补稳定性。

### 做多模态，需要 CNN / ViT 的来历

```text
1 → 2 → 5
```

### Infra 工程师，只想知道训练过程里的算法含义

```text
1 → 3
```

第一篇解释反向为什么是前向两倍、激活为什么要存（激活重算的对象），第三篇解释优化器状态从哪来（ZeRO / FSDP 切分的对象）。


## 本系列的边界

- **Transformer 本身**：attention 的变体、位置编码、MoE、参数量与 FLOPs 的完整推导，在 [《Transformer 与 LLM》](/transformer-and-llm-for-infra-engineers.html)系列。本系列第六篇止于"attention 为什么被发明、为什么取代 RNN"。
- **预训练配方**：具体的学习率、batch、warmup 数值怎么随规模定，scaling law，数据配比。属于 L4。本系列只讲每个开关的原理与诊断。
- **框架内部**：Autograd 引擎、Dispatcher、分布式通信、混合精度的实现。属于 Infra 地图 03 系列。本系列只到"框架在做什么"。
- **数值格式**：bf16 / fp8 的位布局、混合精度为什么能工作。在 04 系列第六篇。
- **泛化理论**：VC 维、Rademacher 复杂度、PAC-Bayes。本系列只讲现象与实践中的正则化手段，不做理论。
- **具体的 CNN / RNN 应用**：目标检测、分割、语音识别的网络设计。本系列只讲两条结构史留给 Transformer 的遗产。


## 前置要求与说明

### 前置要求

- [L0 导读](/math-for-ai-algorithm-engineers.html)的四个分支：链式法则、Jacobian、期望与方差、范数与谱范数的概念；
- [L1 导读](/tooling-for-ai-algorithm-engineers.html)的 NumPy 形状与广播，会写二十行 PyTorch 训练循环；
- [L2 导读](/classical-machine-learning-in-the-llm-era.html)的过拟合、偏差 - 方差、正则化的概念。

不要求：了解任何具体的初始化、归一化、优化器方法；有 GPU。

### 版本与基线

- 实验基线：第一至四篇用 MNIST 与 Fashion-MNIST 规模的数据（几万样本、几十万到几百万参数的 MLP），第五篇用 CIFAR-10，第六篇用合成的序列任务；所有数字在普通笔记本 CPU 上几分钟内可复现；
- 算账时引用的真实网络：ResNet-50（25.6M 参数）、GPT-2（124M 到 1.5B，$$\sqrt{2L}$$ 初始化缩放）、Llama-3-8B（32 层、初始化标准差 0.02、AdamW $$\beta_2 = 0.95$$、weight decay 0.1、裁剪 1.0），超参数取自公开论文与技术报告；
- 论文引用以第一作者与年份标注。


## 章节目录

1. [反向传播：手推一个两层网络](/backpropagation-by-hand.html)
2. [训练为什么不稳定：初始化、归一化与残差](/initialization-normalization-and-residual.html)
3. [优化器：从 SGD 到 AdamW 与学习率调度](/optimizers-from-sgd-to-adamw.html)
4. [正则化与泛化：为什么参数比样本多却不过拟合](/regularization-and-generalization.html)
5. [CNN：从 LeNet 到 ResNet，再到 ViT](/cnn-from-lenet-to-resnet-and-vit.html)
6. [RNN：从 LSTM 到 attention 的诞生](/rnn-lstm-and-the-birth-of-attention.html)


## 最终目标

读完这套系列之后，面对一个训练中的网络，读者应该能够回答：

```text
loss 不降，是梯度没传到还是学习率不对？                 → 第一篇：梯度检查；第三篇：学习率与 warmup
loss 爆成 NaN，先看哪个数字？                          → 第二篇：逐层激活方差；第三篇：梯度范数与裁剪
这个网络为什么要 Pre-Norm？初始化为什么是 0.02？        → 第二篇：方差传播与残差流
AdamW 的每个超参数在控制什么？状态占多少显存？          → 第三篇
SFT 训了 5 个 epoch，模型开始背答案，怎么判断、怎么修？  → 第四篇
预训练为什么一个 epoch、不用 dropout？                 → 第四篇
ViT 的 patch embedding 与卷积是什么关系？              → 第五篇
attention 为什么能取代 RNN？代价是什么？               → 第六篇
```

最终目标是三种能力：

1. **推导能力**：面对一个训练现象，能写出它背后的公式（Jacobian 乘积、更新量的量级、期望等价）；
2. **诊断能力**：面对一条不对的曲线，知道看哪个数字、动哪个开关、预期改善多少；
3. **衔接能力**：读 Transformer 与 LLM 的技术报告时，每一个关于初始化、归一化、优化器、稳定性的段落都能对到本系列的某一节。

这一层是 L4 之前的最后一层基础。它不新，但没有它，后面每一层的"为什么"都只能靠记。
