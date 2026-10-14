---
layout: post
title: "深度学习基础（09）：Embedding 与表示学习——从查表到句向量"
subtitle: "Embeddings and Representation Learning: From Lookup Tables to Sentence Vectors"
date: 2026-03-28 20:00:00
tags: [AI, Deep Learning, LLM]
series: deep-learning-foundations
catalog: true
updated: 2026-10-14
---

词、token、用户 ID 都是离散编号；神经网络接收的却是向量。中间这一步查表，把一个编号变成什么？[^q0]

## 一、总览：从离散编号到可学习的表示

Embedding 把每个离散符号映射成一个可训练的稠密向量。模型再通过预测、重建或对比等目标，调整这些向量，让它们携带任务需要的信息。

Embedding 能嵌入的不只是词。句子、图像、位置、图里的节点、推荐系统里的用户 ID，都可以走同一条路：先有东西被嵌入，再有某种训练信号把向量摆到位。这篇覆盖这条路的全貌：

| 嵌入什么 | 代表方法 | 怎么得到向量 | 去处 |
|---|---|---|---|
| 词 | LSA、PPMI + SVD、GloVe、Word2Vec | 统计共现再分解，或预测上下文 | [第四章](#counting-embeddings)、[第五章](#word2vec) |
| 子词 / 词片 | fastText、BPE | 字符 n-gram 求和；按频率合并 | [第六章](#subword-embeddings) |
| 句子与文本 | Sentence-BERT、SimCSE、E5 / BGE | 对比学习或弱监督对 | [第八章](#sentence-embeddings) |
| 图像 | ViT、CLIP | patch 投影；图文对比 | [第九章](#beyond-words)、《[多模态（01）：视觉编码器：CLIP、SigLIP 与自监督 ViT](/vision-encoders-clip-siglip-and-self-supervised-vit.html)》 |
| 位置 | 正弦编码、RoPE | 编码 token 的先后位置 | [第九章](#beyond-words)、《[现代 LLM 结构（03）：位置编码与外推](/positional-encoding-and-long-context.html)》 |
| 图节点与实体 | DeepWalk、node2vec、TransE | 随机游走当句子跑 skip-gram；关系平移 | [第九章](#beyond-words) |
| ID 与类别特征 | 矩阵分解、双塔 | 从交互数据里学 | [第九章](#beyond-words) |

Table: 各类 embedding 对象、代表方法与本文去处

同一张图画出来：

![embedding 全景：左边是嵌入的对象（词、子词、句子、图像、位置、图节点、ID），中间是四类得到向量的信号（计数与分解、预测、上下文、对比），右边是它们共同的产物——一个 d 维向量空间](/img/in-post/dl-emb-landscape.svg)

这篇先从 one-hot 与查表出发，再按「怎么得到向量」分四块展开：计数式分解（第四章）、预测式词向量（第五章）、子词方法（第六章），然后是向量应用的两大延伸——随上下文变化的表示（第七章）和句子级表示（第八章），最后把对象铺开（第九章）、把训练信号收拢（第十章）。实验还是用完整 text8 训练 skip-gram 模型收尾。

### 1. 本文的章节安排

| 章 | 主题 | 要回答的问题 |
|---|---|---|
| 二 | one-hot 与 embedding 表 | 编号怎样查成向量，梯度更新哪一行？ |
| 三 | 分布式假设 | 为什么共享上下文的词会靠近？ |
| 四 | 计数式：LSA、PPMI + SVD 与 GloVe | 不训练网络，能不能直接算出词向量？ |
| 五 | Word2Vec | skip-gram、CBOW 与负采样如何训练词向量？ |
| 六 | 子词与开放词表 | 词表外的词怎么办？ |
| 七 | 从静态到上下文 | 同一个词为什么能有不同表示？ |
| 八 | 句子与文本 embedding | 一整句话怎么变成一个向量？ |
| 九 | 词以外的对象 | 位置、图像、图节点、ID 怎么嵌入？ |
| 十 | 表示学习的监督信号 | 五种目标分别在学什么？ |
| 十一 | 工程取舍 | 词表大小、稀疏梯度与相似度检索有什么代价？ |
| 十二 | text8 案例 | CPU 训练后的近邻和类比会怎样变化？ |
| 十三–十四 | 小结与自测 | 把整条链路收起来，再检查理解 |

Table: Embedding 系列各章的主题与问题

### 2. 来龙去脉

| 年 | 谁 | 当时的问题 | 怎么解决 | 效果 | 引入或遗留问题 |
|---|---|---|---|---|---|
| 1954–1957 | Harris<br/>Firth | 词的意义怎么形式化 | 分布假设：词的使用环境能提供关于词的信息 | 成为后来一切分布式表示的思想源头 | 只是语言学观点，没有给出可计算的向量 |
| 1990 | Deerwester 等 | 词-文档共现矩阵太大太稀疏 | **LSA**：对共现矩阵做 SVD，取前几维当词的表示 | 词与文档第一次有了低维语义坐标 | 计数稀疏、窗口全局混合，向量仍粗糙 |
| 2003 | Bengio 等 | 词表是离散编号，词与词之间没有几何 | 神经概率语言模型：把词的分布式向量与语言模型一起学习 | 词向量第一次作为模型参数被学出来 | 全词表 softmax 太贵，静态词向量还没普及 |
| 2013 | Mikolov 等 | 联合训练太慢，词向量训不起 | **Word2Vec**：skip-gram、CBOW 加负采样，把训练成本降下来 | 静态词向量在 CPU 上也能训出来，成为标配工具 | 向量不随上下文变化，多义词只有一行 |
| 2013 | Bordes 等 | 知识图谱里的实体和关系也要进向量空间 | **TransE**：让 $$h + r \approx t$$，用间距损失学 | 结构化知识第一次能用向量做链路预测 | 一对一关系好建，一对多、多对多关系失真 |
| 2014 | Pennington 等 | Word2Vec 只看局部上下文窗口 | **GloVe**：把全局词共现统计纳入向量学习 | 词向量的几何结构里出现了可解释的线性关系 | 仍是静态向量，上下文无关 |
| 2014 / 2016 | Perozzi 等<br/>Grover & Leskovec | 图节点没有文本，怎么嵌入 | **DeepWalk**：随机游走当句子，跑 skip-gram<br/>**node2vec**：有偏游走控制往宽还是往深走 | 图结构直接被翻译成词向量问题 | 只能给训练时见过的节点向量，新节点要重训 |
| 2017 | Bojanowski 等 | 罕见词与形态变化没有向量 | **fastText**：把字符 n-gram 纳入词表示 | 罕见词、拼写变体也能合成向量 | 仍是静态向量，上下文无关 |
| 2018–2019 | ELMo、GPT、BERT | 静态词向量不随上下文变化 | 表示由整句上下文决定，不再查词表里固定的一行 | 上下文表示取代静态词向量成为主流 | embedding 层退化为输入接口，表示工作交给后面的网络 |
| 2019 | Reimers & Gurevych | BERT 输出的句向量直接拿来比效果不好，逐对编码又太慢 | **Sentence-BERT**：siamese 结构把每个句子单独编码成一个向量 | 句向量检索从小时级降到毫秒级，SBERT 成为句向量的标准做法 | 需要标注的句对数据 |
| 2021 | Gao 等 | 句向量各向异性，没有标注对怎么办 | **SimCSE**：同一句子过两次 dropout 当正例，batch 内互作负例 | 无监督句向量效果逼近有监督 | 正例只来自 dropout 噪声，语义偏差有限 |
| 2021 | Radford 等 | 图像和文本能不能进同一个向量空间 | **CLIP**：4 亿图文对做对比学习，对齐两个模态的向量 | 零样本分类、图文检索都靠这个共享空间 | 跨模态空间对齐粒度粗，细节靠后接模型补 |
| 2022 | Wang 等 | 各家句向量各训各的，检索任务缺少通用品 | **E5**：大规模弱监督文本对 + `query:`/`passage:` 前缀区分两侧 | 一个模型覆盖检索、聚类、语义相似度 | 前缀约定成了新负担，换模型要记得换前缀 |

Table: 词分布思想与表示学习方法的演进

Harris 在 [《Distributional Structure》](https://doi.org/10.1080/00437956.1954.11659520) 中研究词项的分布环境；Firth 在 [《A synopsis of linguistic theory 1930–1955》](https://languagelog.ldc.upenn.edu/myl/Firth1957.pdf) 中写下常被引用的“知道一个词，要看它与哪些词为伴”。这是一条从统计共现到神经网络表示的线索，不表示后来所有 embedding 都采用同一种目标函数。

Bengio 等人的 [神经概率语言模型](https://jmlr.org/papers/v3/bengio03a.html) 同时学习词的分布式表示和词序列概率。之后，[Word2Vec](https://arxiv.org/abs/1301.3781)、[GloVe](https://aclanthology.org/D14-1162/) 与 [fastText](https://aclanthology.org/Q17-1010/) 分别发展出不同的训练信号或统计形式。

表示的另一条变化是：从“词表中每个词只有一行”，走到“每次出现的 token 都能根据上下文改变”。[ELMo](https://arxiv.org/abs/1802.05365) 用双向语言模型内部状态生成上下文表示；[GPT-2](https://arxiv.org/abs/1907.11692) 与 [BERT](https://arxiv.org/abs/1810.04805) 则把 Transformer 的隐藏状态用于上下文建模。

## 二、one-hot 与 embedding 表
{: #embedding-table }

假设词表只有“猫、狗、跑”三个词。one-hot 用一个与词表等长的向量表示词 ID：

| 词 | one-hot |
|---|---|
| 猫 | $$[1,0,0]$$ |
| 狗 | $$[0,1,0]$$ |
| 跑 | $$[0,0,1]$$ |

Table: 三词词表的 one-hot 编码示例

向量维度等于词表大小 $$V$$。任意两个不同词的内积都是 0，表示本身不提供“猫比跑更像狗”这样的相似度。

Embedding 表 $$E\in\mathbb{R}^{V\times d}$$ 为每个词保存一行。查词 ID $$i$$ 就取第 $$i$$ 行：

$$
\operatorname{Embedding}(i)=E_i.
$$

这也等价于 one-hot 行向量乘表：

$$
\operatorname{onehot}(i)^\top E=E_i.
$$

实现时不必真的构造长度为 $$V$$ 的 one-hot 向量；`nn.Embedding` 直接按 ID 查表。

反向传播时，只有被查到的行会收到非零梯度。下面用 `sparse=True` 让 PyTorch 把梯度实际存成稀疏张量；重复查到 ID 2 时，该行的梯度会累加两次。

```python
import torch
from torch import nn

table = nn.Embedding(5, 2, sparse=True)
table(torch.tensor([0, 2, 2])).sum().backward()
print(table.weight.grad.coalesce().indices()[0].tolist())
print(table.weight.grad.coalesce().values())
```

输出的索引只有第 0、2 行。默认 `sparse=False` 时梯度张量是稠密的，只是未查到的行数值为零；两种情况不要混为一谈。

### GPT-2 的词表矩阵有多大？

[GPT-2 的配置](https://huggingface.co/openai-community/gpt2/blob/main/config.json) 给出词表大小 $$V=50{,}257$$、隐藏维度 $$d=768$$。仅输入 embedding 表就有：

$$
50{,}257\times768=38{,}597{,}376
$$

个参数。GPT-2 small 总参数约 124M，因此这张表约占 31%。词表参数不是可以忽略的“索引开销”。

GPT-2 把输入词向量矩阵与输出词表投影共享。给定最后隐藏状态 $$h$$，词 $$w$$ 的输出 logit 是：

$$
\operatorname{logit}(w)=h^\top E_w.
$$

同一行既作为输入词的表示，也作为输出分类器对应词的方向；共享权重减少一张词表矩阵，并让输入与预测端共用统计。这个做法叫 weight tying，[Press 与 Wolf 2017](https://arxiv.org/abs/1608.05859) 最早在语言模型上系统验证了它。Hugging Face 的 GPT-2 模型配置与 [模型实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py) 采用 tying；nanoGPT 也在 [模型代码](https://github.com/karpathy/nanoGPT/blob/master/model.py) 中显式把 `lm_head.weight` 设为 `wte.weight`。

## 三、分布式假设：从共同上下文中学相似度

Word2Vec 的直觉是：如果两个词经常出现在相似的上下文里，它们的向量就应该相似。向量不再只是 ID 的编码，而是从共现数据里学出的分布式表示。

以“猫追着球跑”和“狗追着球跑”为例，猫与狗都出现在“追着球跑”附近。预测上下文的训练会把它们的向量都往能预测这些上下文的方向推。共同的邻居因此会在向量空间里形成共同方向。

这不是说两个词完全同义，也不是说语料里的偏见会自动消失。模型学到的是训练数据和目标函数所奖励的统计规律。

## 四、计数式：LSA、PPMI + SVD 与 GloVe
{: #counting-embeddings }

在神经网络之前，词向量已经可以“算”出来：先数共现，再把矩阵分解掉。这条线没有训练循环，只有统计量和线性代数。

### 1. 一个可以手算的 toy 语料

拿一个只有五句话的小语料：「猫 追 狗」「狗 追 猫」「狗 追 猫」「猫 吃 鱼」「猫 睡」。词表六个词。窗口取相邻的两个词，数出对称共现矩阵 $$X$$（$$X_{ij}$$ 是词 $$i$$ 与词 $$j$$ 相邻出现的次数）：

| | 猫 | 狗 | 鱼 | 吃 | 追 | 睡 |
|---|---:|---:|---:|---:|---:|---:|
| 猫 | 0 | 0 | 0 | 1 | 3 | 1 |
| 狗 | 0 | 0 | 0 | 0 | 3 | 0 |
| 鱼 | 0 | 0 | 0 | 1 | 0 | 0 |
| 吃 | 1 | 0 | 1 | 0 | 0 | 0 |
| 追 | 3 | 3 | 0 | 0 | 0 | 0 |
| 睡 | 1 | 0 | 0 | 0 | 0 | 0 |

Table: toy 语料的对称共现矩阵（labs 脚本 `case_09_concept_figures.py` 输出）

共现次数本身受词频支配：高频词跟谁都容易撞上。要看的是共现比独立假设下多出多少，这就是点互信息（PMI）：$$p_{ij}=X_{ij}/N$$ 是观测到的共现概率（$$N$$ 是矩阵里所有计数之和，本例 $$N=20$$），$$p_i=\sum_j X_{ij}/N$$ 是词 $$i$$ 单独出现的概率。

$$
\operatorname{PMI}(i,j)=\log_2\frac{p_{ij}}{p_i\,p_j},\qquad
\operatorname{PPMI}(i,j)=\max(0,\operatorname{PMI}(i,j)).
$$

PMI 大于 0 表示共现多于随机预期，小于 0 表示少于。负值在稀疏数据里噪声很大，实践里直接截成 0——PPMI 这个截断出自 Church & Hanks 1990 的 PMI，[Bullinaria & Levy 2007](https://aclanthology.org/P07-1044/) 系统比较过它在词向量上的表现。对上面的矩阵取 PPMI：

| | 猫 | 狗 | 鱼 | 吃 | 追 | 睡 |
|---|---:|---:|---:|---:|---:|---:|
| 猫 | 0 | 0 | 0 | 0.85 | 0.85 | 1.85 |
| 狗 | 0 | 0 | 0 | 0 | 1.58 | 0 |
| 鱼 | 0 | 0 | 0 | 3.17 | 0 | 0 |
| 吃 | 0.85 | 0 | 3.17 | 0 | 0 | 0 |
| 追 | 0.85 | 1.58 | 0 | 0 | 0 | 0 |
| 睡 | 1.85 | 0 | 0 | 0 | 0 | 0 |

Table: toy 语料的 PPMI 矩阵

现在每个词是一个六维向量。猫与狗的行很像（都指向「追」），鱼只指向「吃」。LSA（Deerwester 等 1990）的做法是：对这张表做截断 SVD，只留最大的几个奇异方向。SVD 的分解式与取前几维的含义见《[算法工程师的数学（03）：正交与旋转、特征值与 SVD——从 RoPE 到 LoRA](/orthogonal-rotation-svd-and-low-rank.html)》「四、秩与 SVD · 2. SVD：任何矩阵都能这样分解」。留两维，把 PPMI 矩阵分解为 $$P\approx U\Sigma V^\top$$，取 $$U$$ 的前两列乘 $$\sqrt{\Sigma}$$ 当坐标，得到：

![toy 语料六个词的 PPMI 行向量经 SVD 压到二维后的散点：猫与狗在左侧聚在一起，鱼离它们很远，吃、追、睡这些上下文词在另一侧](/img/in-post/dl-emb-ppmi-svd-toy.svg)

图里共享「追」这个上下文的猫和狗落在相近位置，只跟「吃」共现的鱼离它们很远——这就是“共现结构变成几何距离”的全部含义。这一整步没有梯度、没有训练，只有数数和分解；矩阵分解作为降维手段的来龙去脉见《[LLM 时代的经典机器学习（08）：降维——PCA、SVD、t-SNE / UMAP 与 embedding 的各向异性](/dimensionality-reduction-pca-svd-tsne-and-umap.html)》「四、三个名字一件事」。

### 2. GloVe：把计数塞进一个回归

GloVe（Pennington 等 2014）不再显式分解矩阵，而是直接拟合共现的对数。词 $$i$$ 与 $$j$$ 的向量 $$w_i$$、$$\tilde w_j$$ 满足：

$$
J=\sum_{i,j}f(X_{ij})\bigl(w_i^\top \tilde w_j+b_i+\tilde b_j-\log X_{ij}\bigr)^2.
$$

$$w_i$$ 是词作为中心词时的向量，$$\tilde w_j$$ 是它作为上下文词时的向量，$$b_i,\tilde b_j$$ 是对应的偏置；$$f(X_{ij})$$ 是加权函数，给中等频次的共现更大权重、给极稀有和极常见的共现压低权重，[原论文](https://aclanthology.org/D14-1162/)取的分段函数在 $$X_{ij}$$ 小于 100 时随频次上升、之后封顶为 1。目标的意思是：两个向量的点积应该预测它们共现次数的对数——计数没有被丢掉，只是从「分解的输入」变成了「回归的目标」。

## 五、Word2Vec：预测邻居，还是预测中心词
{: #word2vec }

上一章用分解共现矩阵直接算向量，Word2Vec 换了一条路：设一个预测任务，把向量当参数学出来。

它的两个常见目标只是在输入、预测方向上不同：

| 目标 | 输入 | 预测 |
|---|---|---|
| skip-gram | 中心词 | 窗口内的上下文词 |
| CBOW | 上下文词向量的平均 | 中心词 |

Table: Skip-gram 与 CBOW 的输入和预测目标

以句子“猫 追着 球 跑”为例，窗口半径为 1 时，skip-gram 会从“追着”预测“猫”和“球”；CBOW 则把“猫”和“球”的向量合起来，预测“追着”。[Word2Vec 原论文](https://arxiv.org/abs/1301.3781) 讨论了这两种架构。

### 1. 为什么不直接对整个词表做 softmax

给定中心词向量 $$v_c$$，模型要为词表中每个候选上下文词计算分数：

$$
s_w=v_w^\top v_c,\qquad
P(w\mid c)=\frac{\exp(s_w)}{\sum_{u\in V}\exp(s_u)}.
$$

词表每增加一个词，就多算一个长度为 $$d$$ 的点积。取 $$V=50{,}000$$、$$d=300$$，单个中心词一次全词表打分约需 $$2Vd=30$$ 百万 FLOPs；批量与 softmax 归一化还会增加工作量。

Mikolov 等人在 [《Distributed Representations of Words and Phrases and their Compositionality》](https://arxiv.org/abs/1310.4546) 中采用**负采样**：不再归一化全词表概率，而是让一个真实的中心词—上下文词对得分高，再让 $$k$$ 个噪声词对得分低。每个样本只算 $$k+1$$ 个点积。

对一个正样本 $$ (c,o) $$ 和负样本 $$n_1,\ldots,n_k$$，目标可写成：

$$
-\log\sigma(v_o^\top v_c)
-\sum_{j=1}^{k}\log\sigma(-v_{n_j}^\top v_c).
$$

它就是“真实共现”与“抽来的噪声”之间的二分类。若 $$d=300$$、$$k=5$$，每个正样本是 6 个点积，约 3,600 FLOPs，而不是 3,000 万量级。

### 2. 二维手算与类比

先看一组特意构造的二维向量：

$$
\text{man}=(1,0),\quad
\text{woman}=(1,1),\quad
\text{king}=(2,0),\quad
\text{queen}=(2,1).
$$

于是：

$$
\text{king}-\text{man}+\text{woman}
=(2,0)-(1,0)+(1,1)=(2,1)=\text{queen}.
$$

这只是为了看清向量加减如何表达关系的手算例子，不是任何真实模型跑出的结果。

真实向量里常用余弦相似度检索候选。做类比时必须把输入的原词排除，否则“king”或“woman”可能因本身与查询向量相似而占住结果。即使排除了它们，类比也只是某种检索行为，并不能证明关系在语言或社会意义上成立。[Nissim 等人](https://aclanthology.org/2020.cl-2.7/) 指出，经典词向量类比对偏见的诊断并不可靠，数据、实现与选择会影响看起来“惊艳”的例子。

### 3. 负采样与 PMI 的联系：两条路在同一个地方汇合

负采样训练出的点积还可以从矩阵角度理解。Levy 与 Goldberg 在 [2014 年的分析](https://arxiv.org/abs/1402.3722) 中证明，在特定假设下，skip-gram with negative sampling（SGNS）隐式分解一个 **shifted PMI** 矩阵：词与上下文共现越出乎独立分布的预期，它们的点积越大。

也就是说，第四章「计数 → PMI → 分解」和这一章「预测任务 + 负采样」学的是同一张统计表——一个显式分解它，一个隐式拟合它。GloVe 干脆把它当回归目标直接拟合。矩阵分解与低维投影的联系可见《[LLM 时代的经典机器学习（08）：降维——PCA、SVD、t-SNE / UMAP 与 embedding 的各向异性](/dimensionality-reduction-pca-svd-tsne-and-umap.html)》「四、三个名字一件事」。

## 六、子词与开放词表：fastText、BPE 与 hashing
{: #subword-embeddings }

到目前为止，表的一行对应词表里的一个词。这留下两个洞：词表外的词（OOV）查不到任何一行；词表越大，行数与参数越多。这一章是补这两个洞的三条路。

### 1. fastText：词是字符 n-gram 的和

[Bojanowski 等人 2017](https://aclanthology.org/Q17-1010/) 的做法是不给词一个原子行，而把词拆成字符 n-gram，词的向量是它所有 n-gram 向量之和。取 $$n$$ 从 3 到 6、词 `跑的`，拆出的 n-gram 包括 `<跑`、`跑的`、`跑的>`、`跑`、`的`、`的>`（`<`、`>` 是词首尾标记）。训练照常跑 skip-gram，只是查表前先查 n-gram。词表外的词只要由见过的 n-gram 组成，也能合成一个向量。代价：n-gram 表本身很大，通常再用 hashing 压进固定大小的桶。

### 2. BPE：把「词」变成「词片」

另一条路是根本不用整词当单位。BPE 反复合并语料里频率最高的相邻符号对，得到固定大小的子词词表；罕见词被拆成若干子词，OOV 在分词那层就消失了。embedding 层仍然只是一张表，只是行从「词」变成「词片」。合并过程与词表大小的权衡见《[预训练（02）：分词与词表：BPE、词表大小与 token 效率](/tokenizer-vocabulary-and-token-efficiency.html)》；LLM 里的 embedding 层查的就是这种子词 ID。

### 3. hashing trick：共享桶代替长表

再激进一点，可以不要词表。[Weinberger 等人 2009](https://arxiv.org/abs/0902.2206) 的 feature hashing 把任意特征（词、n-gram、类别取值）哈希进固定数量的桶，每个桶共享一行向量。优点是不需要词表、天然处理开放词表；代价是碰撞——不同的词被压进同一行、不可区分。hash embedding（Svenstrup 等 2017）给每个词配多个哈希函数、把命中向量加权求和，来缓解碰撞。推荐系统里的超大 ID 表常用这类结构，见第九章。

## 七、从静态词向量到上下文表示
{: #contextual-representation }

静态词向量为一个词类型保存一行。“银行”出现在“河岸边”与“银行存款”里，查到的仍是同一个向量；向量本身无法区分这两种上下文。

[ELMo](https://arxiv.org/abs/1802.05365) 把双向语言模型每层的状态组合成 token 表示。Transformer 语言模型则让 token 表示经过多层 attention 和 FFN 后，逐层带上上下文信息。

因此，词表里的输入 embedding 只是第 0 层表示；模型内部每个位置的 hidden state 也是一个上下文 embedding。Transformer 中输入 embedding 的具体路径见《[Transformer 原理与实现（01）：Transformer 长什么样——从一句话到下一个 token](/transformer-architecture-from-a-sentence-to-the-next-token.html)》「token embedding：把编号变成向量」。

第二章的 GPT-2 例子已经出现过 weight tying：输入表与输出投影共享同一组权重。同一个词在输入端查到的向量，同时是输出端对它打分的方向——这让「词的方向」这个词表几何在语言模型里承担了双重角色，也是[Press 与 Wolf 2017](https://arxiv.org/abs/1608.05859) 的结论： tying 不丢效果、省一张大表。

分工上，tokenizer 负责把字符串切成离散 token ID；embedding 表负责把 ID 映射到可训练向量。表示方式的对照见《[预训练（02）：分词与词表：BPE、词表大小与 token 效率](/tokenizer-vocabulary-and-token-efficiency.html)》「文本怎么变成数字：五种表示方式，以及 LLM 为什么选“子词 + embedding”」。

## 八、句子与文本 embedding
{: #sentence-embeddings }

词向量之后，自然的问题是：一整句话怎么变成一个向量，用来做检索、聚类、相似度判断？这一章是这个问题的答案史。

### 1. 为什么朴素平均和裸 CLS 不行

最简单的做法是把句子里各词的向量取平均。它能用，但很弱：高频词的向量把平均拉向公共方向，句子特有的信息被稀释。直接拿 BERT 的 `[CLS]` 或最后层 hidden state 平均也一样不好——BERT 为下一句判断训练出的 `[CLS]` 不是为相似度检索设计的，而且它的表示空间各向异性严重：所有句向量挤在锥形里，余弦相似度普遍偏高，区分度差。各向异性的诊断见《[LLM 时代的经典机器学习（08）：降维——PCA、SVD、t-SNE / UMAP 与 embedding 的各向异性](/dimensionality-reduction-pca-svd-tsne-and-umap.html)》「embedding 的各向异性」。

### 2. Sentence-BERT：为相似度而生的孪生结构

[Reimers & Gurevych 2019](https://arxiv.org/abs/1908.10084) 的 Sentence-BERT 用两个共享权重的 encoder（siamese / bi-encoder）分别编码两个句子，在句对任务（NLI、STS）上微调后，把每个句子的向量（通常取 mean pooling）用余弦直接比。它解决两件事：句向量第一次被「为相似度训练」而不是借用 CLS；每个句子单独编码，可以离线算好存起来，检索时只做一次点积。

### 3. SimCSE：dropout 就是数据增强

[Gao 等 2021](https://arxiv.org/abs/2104.08821) 的 SimCSE 证明不需要标注对：把同一句子用不同的 dropout mask 过两遍 encoder，两次输出互为正例，batch 里其他句子互为负例—— InfoNCE 目标即可。无监督版本的效果在 STS 上接近甚至超过一些有监督基线，把句向量训练的门槛降到一个 encoder + 一批句子。

### 4. E5 / BGE 一族：弱监督对 + 前缀约定

[Wang 等 2022 的 E5](https://arxiv.org/abs/2212.03533) 改用海量弱监督对（Reddit 问答、维基标题-正文、StackExchange 等）做对比预训练，再用小量标注微调。它引入的一个约定至今还在用：查询侧文本前加 `query:`、文档侧加 `passage:`，靠前缀区分两侧分布。BGE、GTE 等后续模型沿用同样的配方。

### 5. 两个实用细节

- **decoder 模型做 embedding**：GPT 类 decoder 是因果的，前面位置看不到后面，取句向量用最后一个 token 的 hidden state（last-token pooling）而不是平均。
- **Matryoshka 表示学习**（[Kusupati 等 2022](https://arxiv.org/abs/2205.13147)）：训练时同时优化向量多个前缀维度上的目标，使前 64 / 128 / 256 维都是可用的完整表示——部署时按需截断，省存储和检索时间。

选模型、看 MTEB 榜、以及 embedding 模型在检索链路里的位置，见《[检索与知识接入（04）：索引、混合检索与 rerank](/indexing-hybrid-search-and-reranking.html)》「二、embedding 模型」。

## 九、词以外的对象：位置、图像、图与 ID
{: #beyond-words }

Embedding 不只嵌入词。凡是「离散、数量大、需要被模型连续处理」的对象，都可以放一张表或一个映射函数。下面每块一个对象。

### 1. 位置

**问题**：Transformer 的 attention 对输入顺序不敏感，模型需要「第几个」这个信息。
**怎么做**：把位置 $$t$$ 也嵌入成向量加进 token 表示——可学习的表，或固定的正弦/旋转位置编码（RoPE）。
**效果与遗留**：RoPE 已成主流，外推与长上下文是它的战场。为什么需要位置编码、各路方案怎么选，见《[现代 LLM 结构（03）：位置编码与外推](/positional-encoding-and-long-context.html)》「二、为什么需要位置编码」。

### 2. 图像

**问题**：像素不是离散符号，但 Transformer 需要 token。
**怎么做**：把图切成 patch，每个 patch 线性投影成向量——patch embedding 等价于步长等于核大小的卷积；CLIP 再用图文对比把图像和文本放进同一个空间。
**效果与遗留**：切图的过程见《[深度学习基础（08）：CNN——从 LeNet 到 ResNet 与 ViT](/cnn-from-lenet-to-resnet-and-vit.html)》「六、从 CNN 到 ViT · 2. ViT 的结构」，CLIP 与 ViT 在多模态里的用法见《[多模态（01）：视觉编码器：CLIP、SigLIP 与自监督 ViT](/vision-encoders-clip-siglip-and-self-supervised-vit.html)》「二、ViT 回顾：编码器输出的是什么」。

### 3. 图节点与知识图谱实体

**问题**：图里的节点没有「词表」也没有序列顺序。
**怎么做**：DeepWalk（[Perozzi 等 2014](https://arxiv.org/abs/1403.6652)）在图上做随机游走，把游走序列当句子、节点当词，直接跑第五章的 skip-gram；node2vec（[Grover & Leskovec 2016](https://arxiv.org/abs/1607.00653)）再加有偏游走控制偏向 BFS（邻域同质）还是 DFS（结构角色）。知识图谱的实体和关系则用 TransE（[Bordes 等 2013](https://proceedings.neurips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html)）：要求 $$h + r \approx t$$（头实体向量 + 关系向量 ≈ 尾实体向量），用 margin 排序损失训练。
**效果与遗留**：图上每个新节点都要重新游走强；TransE 对一对一关系好用，一对多、多对多、对称关系会失真。

### 4. 推荐里的 ID 与类别特征

**问题**：推荐系统输入是一堆 ID（用户、商品、品类、城市），每个都可能是百万级词表。
**怎么做**：每个 ID 一张 embedding 表；也可以用矩阵分解（把用户-物品交互矩阵分解出两组向量）或双塔模型（用户塔、物品塔各自输出一个向量，线上做内积检索）。
**效果与遗留**：成本集中在表的大小上——一张 10 亿行、64 维、fp32 的表就是 $$10^9\times64\times4\text{ B}=256\text{ GB}$$，单机放不下，必须分片、量化或用第六章的 hashing 压缩。

## 十、表示学习的五种监督信号

“表示学习”不是一个固定网络结构，而是用某种训练目标，让向量保存对任务有用的信息。信号一共五种，每种都回答同一个问题——监督从哪里来。五种信号的输入与目标画在一排：

![五种监督信号的示意：预测用前文推下一个词或盖住中间猜回来；重建把 x 压进瓶颈再还原；对比让正例近、负例远；自蒸馏让同一输入的两个视图对齐，teacher 用 EMA；有监督用人工标签](/img/in-post/dl-emb-supervision-signals.svg)

**1. 预测。** 给一部分，预测缺的那部分。自回归语言模型（GPT）预测下一个 token：$$-\log P(x_t\mid x_{<t})$$；掩码模型（BERT、MAE）盖住中间一块，用两侧内容预测被盖位置。模型为了完成预测，会把有助于区分目标的特征写进向量里。Word2Vec 与全部语言模型都在这一类。

**2. 重建。** 自编码器先把输入压进较窄的 latent，再从 latent 重建输入：

$$
x\longrightarrow z=f_\theta(x)\longrightarrow \hat x=g_\phi(z),
\qquad \mathcal{L}_{\text{rec}}=\lVert x-\hat x\rVert^2.
$$

瓶颈迫使表示保留足以重建输入的信息。VAE 在此基础上约束 latent 分布，详见《[多模态（08）：Latent diffusion、DiT 与文生图配方](/latent-diffusion-dit-and-text-to-image-recipes.html#3-vae-的结构与训练)》「VAE 的结构与训练」。

**3. 对比。** 给定正样本对与一批负样本，让正样本相似度高于负样本。van den Oord 等人在 [CPC 原论文](https://arxiv.org/abs/1807.03748) 中使用 InfoNCE；一个常见形式是：

$$
\mathcal{L}_{\text{NCE}}
=-\log\frac{\exp(s(q,k^+)/\tau)}
{\exp(s(q,k^+)/\tau)+\sum_j\exp(s(q,k_j^-)/\tau)}.
$$

这里 $$q$$ 是查询，$$k^+$$ 是匹配对象，$$k_j^-$$ 是负样本，$$\tau$$ 控制 softmax 的尖锐度。SimCSE、CLIP 走的就是这条路；InfoNCE 的推导与温度细节见《[深度学习基础（02）：损失函数——从 MSE 到 DPO](/loss-functions-from-mse-to-dpo.html)》「六、对比与度量学习」，互信息与 InfoNCE 的关系见《[算法工程师的数学（06）：熵、交叉熵与 KL——从困惑度到 DPO](/entropy-cross-entropy-and-kl-to-dpo.html#2-互信息与对比学习)》「互信息与对比学习」，图文对比的 CLIP 例子见《[多模态（01）：视觉编码器：CLIP、SigLIP 与自监督 ViT](/vision-encoders-clip-siglip-and-self-supervised-vit.html#1-infonce)》「InfoNCE」。

**4. 无负样本的自蒸馏。** 对比的麻烦是需要大量负样本。BYOL（[Grill 等 2020](https://arxiv.org/abs/2006.07733)）与 DINO（[Caron 等 2021](https://arxiv.org/abs/2104.14294)）干脆不要负样本：同一输入做两个增广视图，student 网络预测 teacher 网络的输出。这本来会塌缩成常数解（所有输入输出同一个向量），两家用相似的手段避免：BYOL 给 student 加一个 predictor、并给 teacher 用 student 的指数滑动平均（EMA）+ stop-gradient；DINO 给 teacher 输出做中心化与锐化。

**5. 有监督。** 用人工标注当信号：ImageNet 分类预训练出的特征长期是视觉表示的默认起点；度量学习（triplet loss、有监督对比）让同类样本在空间里聚得更紧。

五种信号摆在一起：

| 信号 | 需要什么数据 | 学到什么 | 常见问题 |
|---|---|---|---|
| 预测 | 序列 / 被盖住的输入 | 让表示能推出邻居 | 学的是统计规律不是语义真理；mask 类目标与下游对齐要靠微调 |
| 重建 | 原始输入 | 保留足以还原输入的信息 | 容易被像素级细节占满，高维语义不突出 |
| 对比 | 正例对 + 一批负例 | 把相似样本拉到一起 | 依赖大 batch 或 memory bank；温度敏感 |
| 自蒸馏 | 同一输入的增广视图 | 不要负例也能拉开表示 | 塌缩风险，靠 EMA / predictor / 中心化撑着 |
| 有监督 | 人工标签 | 表示直接服务标注任务 | 受标注覆盖与偏见限制 |

Table: 五种监督信号的数据需求、学到的内容与常见问题

## 十一、工程取舍：词表、维度与优化器状态

词表大小 $$V$$ 与向量维度 $$d$$ 一起决定 embedding 参数量 $$Vd$$。词表越大，能直接表示的离散符号越多；维度越高，每行容量越大，但参数与优化器状态也随之增加。

| 模型 | 配置中的 $$V\times d$$ | 输入 embedding 参数 | 总参数约数 | 输入表占比 |
|---|---:|---:|---:|---:|
| GPT-2 small | $$50{,}257\times768$$ | 38.6M | 124M | 31.0% |
| Llama 3 8B | $$128{,}256\times4{,}096$$ | 525.3M | 8.03B | 6.5% |

Table: GPT-2 与 Llama 3 的 embedding 参数规模

GPT-2 的配置来自 [Hugging Face 模型配置](https://huggingface.co/openai-community/gpt2/blob/main/config.json)。Llama 3 8B 的模型设置见 [Meta 技术报告](https://arxiv.org/abs/2407.21783) 与 [官方 Hugging Face 配置](https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json)，架构定义见 [Meta 官方实现](https://github.com/meta-llama/llama3/blob/main/llama/model.py)。按该配置的层数、注意力与 FFN 矩阵形状合计约 8.03B 参数；官方实现将输入 embedding 与输出投影分别定义，因此输出词表矩阵还占约 6.5%。

词频差异也会影响更新频率：高频词的行每一步都可能更新，罕见词则很少被访问。PyTorch 的 `nn.Embedding(sparse=True)` 能产生稀疏梯度，但标准 AdamW 不接受这种稀疏梯度；`SparseAdam` 等稀疏优化器只处理出现的行。反过来，用普通稠密 AdamW 时，梯度以及一阶、二阶状态都按整张表分配，状态内存随 $$Vd$$ 增长。

向量空间还有各向异性问题：很多向量可能挤在少数方向附近，导致余弦相似度普遍偏高——第八章句向量的问题与它同源。诊断方法见《[LLM 时代的经典机器学习（08）：降维——PCA、SVD、t-SNE / UMAP 与 embedding 的各向异性](/dimensionality-reduction-pca-svd-tsne-and-umap.html)》「embedding 的各向异性」；余弦相似度本身见《[算法工程师的数学（02）：内积、范数与余弦相似度](/inner-product-norms-and-cosine-similarity.html)》「余弦相似度」。

## 十二、案例：在 text8 上训练 skip-gram

实验使用 Matt Mahoney 提供的[完整 text8 语料](http://mattmahoney.net/dc/text8.zip)，共 17,005,207 个词。先按 `min_count=5` 得到 71,290 词词表，再按 Mikolov 等人的高频词抽样公式降低常见词的出现率：词频为 $$f$$、阈值 $$t=10^{-5}$$ 时，若 $$f>t$$，保留概率为 $$P_{\text{keep}}=(\sqrt{f/t}+1)t/f$$。抽样后保留 5,597,324 个词。

训练使用 100 维稀疏 embedding、窗口半径随机取 1–5、每个正样本配 5 个按 $$P(w)\propto f(w)^{0.75}$$ 构造的 alias-table 负样本。批大小为 16,384，`SparseAdam` 学习率从 0.01 衰减到 0.002；对 33,594,601 个训练对完整遍历 3 个 epoch，共 6,153 次更新。脚本首次运行会下载并缓存语料；本次 CPU 运行耗时 310.0 秒。

| 查询词 | 随机初始化的最近邻（余弦） | 训练后的最近邻（余弦） |
|---|---|---|
| king | irn (.410)、blaue (.392)、na (.384) | fortinbras (.672)、weds (.660)、montferrat (.646) |
| queen | auvergne (.455)、specific (.394)、yucat (.370) | elizabeth (.736)、highness (.677)、lady (.650) |
| computer | ambo (.393)、staged (.390)、soybeans (.383) | computers (.801)、machines (.725)、computing (.710)、hardware (.696)、software (.680) |
| software | namibian (.471)、dysfunctional (.397)、niger (.389) | proprietary (.753)、macos (.751)、netbeans (.747) |
| hardware | canucks (.413)、orchestrating (.400)、parse (.362) | cpu (.779)、firmware (.772)、chipsets (.757) |
| language | tribune (.381)、trs (.377)、microcoded (.366) | languages (.810)、spoken (.717)、syntax (.714) |

Table: text8 词向量训练前后的近邻变化

类比 $$\text{king}-\text{man}+\text{woman}$$ 在随机初始化时返回 fairclough、mindstorms、bius；训练后返回 montferrat、widowed、matilda、daughter、eschiva。检索时排除了 king、man、woman，训练后也没有得到 queen。computer、software、hardware 的近邻已出现同领域词，但 king 的近邻仍受 text8 中的专名与叙事搭配影响。这展示了共现统计怎样改变检索排序，不是高质量词典，也不是类比关系成立的证据。

![完整 text8 训练后选定的 100 维词向量在二维 PCA 平面上的投影，标注 king、queen、man、woman、computer、software、hardware 等词](/img/in-post/embedding-word2vec.webp)

图中把训练后的 100 维词向量投影到二维 PCA 平面。PCA 只保留方差最大的两个方向，所以点之间的二维距离不能替代原始空间里的相似度。

配套脚本：[`case_embedding_word2vec.py`](https://github.com/arganzheng/ai-learning-labs/blob/main/deep-learning-foundations/case_embedding_word2vec.py)、[`case_09_concept_figures.py`](https://github.com/arganzheng/ai-learning-labs/blob/main/deep-learning-foundations/case_09_concept_figures.py)。

## 十三、本文小结

Embedding 表把离散 ID 映射为可训练向量；得到这些向量的路线则分四条：计数式分解（LSA、PPMI+SVD、GloVe）、预测式训练（Word2Vec、语言模型）、子词与哈希方法处理开放词表、以及句子与更广义对象上的对比和监督信号。

Word2Vec 用局部上下文训练静态词向量，负采样把全词表归一化改成少量正负样本判别——Levy & Goldberg 证明它和 PMI 分解学的是同一张统计表。现代语言模型把输入 embedding 继续变成受上下文影响的 hidden state，句子向量由专门的编码器给出，位置、图像、图节点和推荐 ID 走同一条「查表或映射 → 用信号摆向量」的路。工程上，词表乘维度决定参数量，稀疏访问与优化器状态决定了训练成本。

## 十四、自测

1. one-hot 乘 embedding 矩阵为什么等价于查表？`nn.Embedding(sparse=True)` 的梯度与默认设置有何不同？
   <details markdown="1"><summary>答案</summary>

   one-hot 只有第 $$i$$ 位为 1，因此乘 $$E$$ 后只取第 $$i$$ 行。`sparse=True` 把非零行用稀疏张量存储；默认设置仍是稠密梯度，只是没查到的行梯度为零。
   </details>

2. PPMI 里的 P 是怎么来的，为什么负数要截成 0？
   <details markdown="1"><summary>答案</summary>

   PMI 是共现概率与独立假设下概率之比的对数；稀疏共现的负值噪声很大，截成 0 只保留「共现多于随机」的部分，作为矩阵分解的输入更稳。
   </details>

3. Word2Vec、GloVe、PPMI+SVD 三条路学的是不是同一样东西？
   <details markdown="1"><summary>答案</summary>

   是。负采样在数学上等价于分解一个 shifted PMI 矩阵；GloVe 把共现对数当回归目标；PPMI+SVD 直接分解计数矩阵。三者都在编码同一张词-上下文统计表，只是走法不同。
   </details>

4. skip-gram 与 CBOW 的预测方向分别是什么？负采样节省了哪一部分计算？
   <details markdown="1"><summary>答案</summary>

   skip-gram 用中心词预测上下文；CBOW 用上下文预测中心词。负采样只比较一个真实词对和少量噪声词对，不再对整个词表计算并归一化 softmax。
   </details>

5. 为什么 skip-gram 类比检索要排除输入的原词？类比结果能证明什么？
   <details markdown="1"><summary>答案</summary>

   原词可能因与查询向量相似而占据近邻列表，需排除它们才能看其他候选。类比只展示特定向量空间里的检索行为，不能证明语言或社会关系；偏见分析也不能只靠经典类比题。
   </details>

6. 词表外的词（OOV）有哪几种处理办法，各自代价是什么？
   <details markdown="1"><summary>答案</summary>

   fastText 把词拆成字符 n-gram 求和，代价是 n-gram 表变大；BPE 把词拆成子词，OOV 在分词层消失；hashing trick 把任意特征哈希进固定桶，代价是不同对象可能撞进同一行。
   </details>

7. 静态词向量与上下文表示的关键差别是什么？
   <details markdown="1"><summary>答案</summary>

   静态表按词类型查固定一行；上下文表示由 token 所在句子和模型层共同决定，同一个词的不同出现可以有不同向量。
   </details>

8. 直接拿 BERT 的 `[CLS]` 当句向量为什么效果差？Sentence-BERT 和 SimCSE 各自怎么解决？
   <details markdown="1"><summary>答案</summary>

   BERT 的 CLS 不是为相似度检索训的，且空间各向异性严重。Sentence-BERT 用孪生 encoder 在句对上微调；SimCSE 用同一句子两遍 dropout 当正例、batch 内互作负例。
   </details>

9. GPT-2 small 与 Llama 3 8B 的输入 embedding 表各有多少参数？为什么前者占总参数的比例更高？
   <details markdown="1"><summary>答案</summary>

   GPT-2 small 是 $$50{,}257\times768=38.6\text{M}$$，约占 124M 的 31%；Llama 3 8B 是 $$128{,}256\times4{,}096=525.3\text{M}$$，约占 8.03B 的 6.5%。模型主体变大后，同一张词表矩阵占比下降。
   </details>

10. 预测、重建、对比、自蒸馏、有监督五种信号各需要什么数据？
   <details markdown="1"><summary>答案</summary>

   预测要序列或遮盖的输入；重建要原始输入；对比要正例对加一批负例；自蒸馏只要同一输入的增广视图；有监督要人工标签。
   </details>

## 下一篇

下一篇进入序列建模：[《深度学习基础（10）：RNN——从 LSTM 到 attention 的诞生》](/rnn-lstm-and-the-birth-of-attention.html)。

[^q0]: 离散 ID 本身没有几何结构。Embedding 把 ID 查成向量；预测、重建或对比等训练目标再把信息写进这些向量。
