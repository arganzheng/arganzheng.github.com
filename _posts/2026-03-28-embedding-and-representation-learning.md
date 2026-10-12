---
layout: post
title: "深度学习基础（08）：Embedding 与表示学习——从 one-hot 到上下文表示"
subtitle: "Embeddings and Representation Learning: From One-Hot Vectors to Contextual Representations"
date: 2026-03-28 20:00:00
tags: [AI, Deep Learning, LLM]
series: deep-learning-foundations
---

词、token、用户 ID 都是离散编号；神经网络接收的却是向量。中间这一步查表，把一个编号变成什么？[^q0]

## 一、总览：从离散编号到可学习的表示

Embedding 把每个离散符号映射成一个可训练的稠密向量。模型再通过预测、重建或对比等目标，调整这些向量，让它们携带任务需要的信息。

这篇先从 one-hot 与查表出发，再看 word2vec 怎样用上下文训练词向量，以及表示如何从“一个词一个向量”走向“一个 token 在当前句子里的向量”。最后用完整 text8 训练 skip-gram 模型。

### 1. 本文的章节安排

| 章 | 主题 | 要回答的问题 |
|---|---|---|
| 二 | one-hot 与 embedding 表 | 编号怎样查成向量，梯度更新哪一行？ |
| 三 | 分布式假设 | 为什么共享上下文的词会靠近？ |
| 四 | Word2Vec | skip-gram、CBOW 与负采样如何训练词向量？ |
| 五 | 从静态到上下文 | 同一个词为什么能有不同表示？ |
| 六 | 表示学习的监督信号 | 预测、重建、对比分别在学什么？ |
| 七 | 工程取舍 | 词表大小、稀疏梯度与相似度检索有什么代价？ |
| 八 | text8 案例 | CPU 训练后的近邻和类比会怎样变化？ |
| 九–十 | 小结与自测 | 把整条链路收起来，再检查理解 |

### 2. 来龙去脉

| 时间 | 想法 | 变化 |
|---|---|---|
| 1954–1957 | Harris 的分布结构、Firth 的上下文观点 | 词的使用环境能提供关于词的信息 |
| 2003 | Bengio 等人的神经概率语言模型 | 把词的分布式向量与语言模型一起学习 |
| 2013 | Mikolov 等人的 Word2Vec | 用 skip-gram、CBOW 与负采样高效训练静态词向量 |
| 2014 | GloVe | 把全局词共现统计纳入向量学习 |
| 2017 | fastText | 把字符 n-gram 纳入词表示，改善罕见词与形态变化 |
| 2018–2019 | ELMo、GPT 与 BERT | 表示开始由整句上下文决定，不再是词表里固定的一行 |

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

同一行既作为输入词的表示，也作为输出分类器对应词的方向；共享权重减少一张词表矩阵，并让输入与预测端共用统计。Hugging Face 的 GPT-2 模型配置与 [模型实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py) 采用 tying；nanoGPT 也在 [模型代码](https://github.com/karpathy/nanoGPT/blob/master/model.py) 中显式把 `lm_head.weight` 设为 `wte.weight`。

## 三、分布式假设：从共同上下文中学相似度

Word2Vec 的直觉是：如果两个词经常出现在相似的上下文里，它们的向量就应该相似。向量不再只是 ID 的编码，而是从共现数据里学出的分布式表示。

以“猫追着球跑”和“狗追着球跑”为例，猫与狗都出现在“追着球跑”附近。预测上下文的训练会把它们的向量都往能预测这些上下文的方向推。共同的邻居因此会在向量空间里形成共同方向。

这不是说两个词完全同义，也不是说语料里的偏见会自动消失。模型学到的是训练数据和目标函数所奖励的统计规律。

## 四、Word2Vec：预测邻居，还是预测中心词
{: #word2vec }
{: #word2vec }

Word2Vec 的两个常见目标只是在输入、预测方向上不同：

| 目标 | 输入 | 预测 |
|---|---|---|
| skip-gram | 中心词 | 窗口内的上下文词 |
| CBOW | 上下文词向量的平均 | 中心词 |

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

### 3. 负采样与 PMI 的联系

负采样训练出的点积还可以从矩阵角度理解。Levy 与 Goldberg 在 [2014 年的分析](https://arxiv.org/abs/1402.3722) 中证明，在特定假设下，skip-gram with negative sampling（SGNS）隐式分解一个 **shifted PMI** 矩阵：词与上下文共现越出乎独立分布的预期，它们的点积越大。

因此可以把它看成“用低维向量近似一张词—上下文统计表”。矩阵分解与低维投影的联系可见《[LLM 时代的经典机器学习（08）：降维——PCA、SVD、t-SNE / UMAP 与 embedding 的各向异性](/dimensionality-reduction-pca-svd-tsne-and-umap.html)》「四、三个名字一件事」。

## 五、从静态词向量到上下文表示
{: #contextual-representation }
{: #contextual-representation }

静态词向量为一个词类型保存一行。“银行”出现在“河岸边”与“银行存款”里，查到的仍是同一个向量；向量本身无法区分这两种上下文。

[ELMo](https://arxiv.org/abs/1802.05365) 把双向语言模型每层的状态组合成 token 表示。Transformer 语言模型则让 token 表示经过多层 attention 和 FFN 后，逐层带上上下文信息。

因此，词表里的输入 embedding 只是第 0 层表示；模型内部每个位置的 hidden state 也是一个上下文 embedding。Transformer 中输入 embedding 的具体路径见《[Transformer 原理与实现（01）：Transformer 长什么样——从一句话到下一个 token](/transformer-architecture-from-a-sentence-to-the-next-token.html)》「token embedding：把编号变成向量」。

分工上，tokenizer 负责把字符串切成离散 token ID；embedding 表负责把 ID 映射到可训练向量。表示方式的对照见《[预训练（02）：分词与词表：BPE、词表大小与 token 效率](/tokenizer-vocabulary-and-token-efficiency.html)》「文本怎么变成数字：五种表示方式，以及 LLM 为什么选“子词 + embedding”」。

## 六、表示学习的三种监督信号

“表示学习”不是一个固定网络结构，而是用某种训练目标，让向量保存对任务有用的信息。

**预测。** Word2Vec 预测上下文，语言模型预测下一个 token。模型为了完成预测，会把有助于区分目标的特征写进向量里。

**重建。** 自编码器先把输入压进较窄的 latent，再从 latent 重建输入：

$$
x\longrightarrow z=f_\theta(x)\longrightarrow \hat x=g_\phi(z),
\qquad \mathcal{L}_{\text{rec}}=\lVert x-\hat x\rVert^2.
$$

瓶颈迫使表示保留足以重建输入的信息。VAE 在此基础上约束 latent 分布，详见《[多模态（08）：Latent diffusion、DiT 与文生图配方](/latent-diffusion-dit-and-text-to-image-recipes.html#3-vae-的结构与训练)》「VAE 的结构与训练」。

**对比。** 给定正样本对与一批负样本，让正样本相似度高于负样本。van den Oord 等人在 [CPC 原论文](https://arxiv.org/abs/1807.03748) 中使用 InfoNCE；一个常见形式是：

$$
\mathcal{L}_{\text{NCE}}
=-\log\frac{\exp(s(q,k^+)/\tau)}
{\exp(s(q,k^+)/\tau)+\sum_j\exp(s(q,k_j^-)/\tau)}.
$$

这里 $$q$$ 是查询，$$k^+$$ 是匹配对象，$$k_j^-$$ 是负样本，$$\tau$$ 控制 softmax 的尖锐度。互信息与 InfoNCE 的关系见《[算法工程师的数学（06）：熵、交叉熵与 KL——从困惑度到 DPO](/entropy-cross-entropy-and-kl-to-dpo.html#2-互信息与对比学习)》「互信息与对比学习」；图文对比的 CLIP 例子见《[多模态（01）：视觉编码器：CLIP、SigLIP 与自监督 ViT](/vision-encoders-clip-siglip-and-self-supervised-vit.html#1-infonce)》「InfoNCE」。

## 七、工程取舍：词表、维度与优化器状态

词表大小 $$V$$ 与向量维度 $$d$$ 一起决定 embedding 参数量 $$Vd$$。词表越大，能直接表示的离散符号越多；维度越高，每行容量越大，但参数与优化器状态也随之增加。

| 模型 | 配置中的 $$V\times d$$ | 输入 embedding 参数 | 总参数约数 | 输入表占比 |
|---|---:|---:|---:|---:|
| GPT-2 small | $$50{,}257\times768$$ | 38.6M | 124M | 31.0% |
| Llama 3 8B | $$128{,}256\times4{,}096$$ | 525.3M | 8.03B | 6.5% |

GPT-2 的配置来自 [Hugging Face 模型配置](https://huggingface.co/openai-community/gpt2/blob/main/config.json)。Llama 3 8B 的模型设置见 [Meta 技术报告](https://arxiv.org/abs/2407.21783) 与 [官方 Hugging Face 配置](https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json)，架构定义见 [Meta 官方实现](https://github.com/meta-llama/llama3/blob/main/llama/model.py)。按该配置的层数、注意力与 FFN 矩阵形状合计约 8.03B 参数；官方实现将输入 embedding 与输出投影分别定义，因此输出词表矩阵还占约 6.5%。

词频差异也会影响更新频率：高频词的行每一步都可能更新，罕见词则很少被访问。PyTorch 的 `nn.Embedding(sparse=True)` 能产生稀疏梯度，但标准 AdamW 不接受这种稀疏梯度；`SparseAdam` 等稀疏优化器只处理出现的行。反过来，用普通稠密 AdamW 时，梯度以及一阶、二阶状态都按整张表分配，状态内存随 $$Vd$$ 增长。

向量空间还有各向异性问题：很多向量可能挤在少数方向附近，导致余弦相似度普遍偏高。诊断方法见《[LLM 时代的经典机器学习（08）：降维——PCA、SVD、t-SNE / UMAP 与 embedding 的各向异性](/dimensionality-reduction-pca-svd-tsne-and-umap.html)》「embedding 的各向异性」；余弦相似度本身见《[算法工程师的数学（02）：内积、范数与余弦相似度](/inner-product-norms-and-cosine-similarity.html)》「余弦相似度」。

## 八、案例：在 text8 上训练 skip-gram

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

类比 $$\text{king}-\text{man}+\text{woman}$$ 在随机初始化时返回 fairclough、mindstorms、bius；训练后返回 montferrat、widowed、matilda、daughter、eschiva。检索时排除了 king、man、woman，训练后也没有得到 queen。computer、software、hardware 的近邻已出现同领域词，但 king 的近邻仍受 text8 中的专名与叙事搭配影响。这展示了共现统计怎样改变检索排序，不是高质量词典，也不是类比关系成立的证据。

![完整 text8 训练后选定的 100 维词向量在二维 PCA 平面上的投影，标注 king、queen、man、woman、computer、software、hardware 等词](/img/in-post/embedding-word2vec.webp)

图中把训练后的 100 维词向量投影到二维 PCA 平面。PCA 只保留方差最大的两个方向，所以点之间的二维距离不能替代原始空间里的相似度。

配套脚本：[`case_embedding_word2vec.py`](https://github.com/arganzheng/ai-learning-labs/blob/main/deep-learning-foundations/case_embedding_word2vec.py)。

## 九、本文小结

Embedding 表把离散 ID 映射为可训练向量；目标函数再通过预测、重建或对比等监督信号，决定这些向量编码什么。

Word2Vec 用局部上下文训练静态词向量，负采样把全词表归一化改成少量正负样本判别。现代语言模型则把输入 embedding 继续变成受上下文影响的 hidden state。工程上，词表乘维度决定参数量，稀疏访问与优化器状态决定了训练成本。

## 十、自测

1. one-hot 乘 embedding 矩阵为什么等价于查表？`nn.Embedding(sparse=True)` 的梯度与默认设置有何不同？
   <details markdown="1"><summary>答案</summary>

   one-hot 只有第 $$i$$ 位为 1，因此乘 $$E$$ 后只取第 $$i$$ 行。`sparse=True` 把非零行用稀疏张量存储；默认设置仍是稠密梯度，只是没查到的行梯度为零。
   </details>

2. skip-gram 与 CBOW 的预测方向分别是什么？负采样节省了哪一部分计算？
   <details markdown="1"><summary>答案</summary>

   skip-gram 用中心词预测上下文；CBOW 用上下文预测中心词。负采样只比较一个真实词对和少量噪声词对，不再对整个词表计算并归一化 softmax。
   </details>

3. 为什么 skip-gram 类比检索要排除输入的原词？类比结果能证明什么？
   <details markdown="1"><summary>答案</summary>

   原词可能因与查询向量相似而占据近邻列表，需排除它们才能看其他候选。类比只展示特定向量空间里的检索行为，不能证明语言或社会关系；偏见分析也不能只靠经典类比题。
   </details>

4. 静态词向量与上下文表示的关键差别是什么？
   <details markdown="1"><summary>答案</summary>

   静态表按词类型查固定一行；上下文表示由 token 所在句子和模型层共同决定，同一个词的不同出现可以有不同向量。
   </details>

5. GPT-2 small 与 Llama 3 8B 的输入 embedding 表各有多少参数？为什么前者占总参数的比例更高？
   <details markdown="1"><summary>答案</summary>

   GPT-2 small 是 $$50{,}257\times768=38.6\text{M}$$，约占 124M 的 31%；Llama 3 8B 是 $$128{,}256\times4{,}096=525.3\text{M}$$，约占 8.03B 的 6.5%。模型主体变大后，同一张词表矩阵占比下降。
   </details>

6. 预测、重建、对比三种目标分别怎样提供表示学习信号？
   <details markdown="1"><summary>答案</summary>

   预测让表示有助于猜上下文或下一个 token；重建要求 latent 保留还原输入的信息；对比让匹配样本比分布中的负样本更相似。
   </details>

## 下一篇

下一篇进入序列建模：[《深度学习基础（09）：RNN——从 LSTM 到 attention 的诞生》](/rnn-lstm-and-the-birth-of-attention.html)。

[^q0]: 离散 ID 本身没有几何结构。Embedding 把 ID 查成向量；预测、重建或对比等训练目标再把信息写进这些向量。
