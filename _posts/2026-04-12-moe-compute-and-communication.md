---
layout: post
title: "Transformer 与 LLM（05）：MoE 的路由、激活参数量与通信形态"
subtitle: "Mixture of Experts: Routing, Active Parameters and Communication Patterns"
tags: [Transformer, LLM, AI, AI-Infra]
catalog: true
---

> 本文是[《Transformer 与 LLM：结构、算量与数值》](/transformer-and-llm-for-infra-engineers.html)系列的第五篇（共七篇）。上一篇：[位置编码与长上下文](/positional-encoding-and-long-context.html)　下一篇：[浮点格式、数值稳定性与混合精度](/floating-point-formats-and-mixed-precision.html)

前四篇讨论的都是 dense 模型：每个 token 经过每一层时，会用到这一层的全部权重。参数量、每 token 算量、每步 decode 的权重读取量，三者之间只差一个常数——参数量 $$N$$ 对应每 token $$2N$$ FLOPs，对应每步读 $$N \times \text{bytes/elem}$$ 字节。

混合专家（Mixture of Experts，MoE）把这三个数拆开了。DeepSeek-V3 的技术报告里写着"总参数 671B，每 token 激活 37B"，从算量看它比 Llama-3-70B 便宜一半，但实际部署时它需要几十张 GPU 组成的专家并行集群，而 70B 一台 8 卡机器就能跑得很好。本篇要回答的核心问题是：

> **DeepSeek-V3 每 token 只算 37B 参数，为什么部署它比部署一个 dense 70B 难得多？把"参数量"、"激活参数量"、"每步实际读取的参数量"三个数分开算。**

要回答这个问题，需要把 MoE 层的结构、参数量的推导、decode 时的访存形态、专家并行引入的 all-to-all 通信、grouped GEMM 的形状与负载均衡逐个算一遍。全篇沿用系列的三个基线模型：Llama-3-70B 作为 dense 对照，Mixtral 8x7B 与 DeepSeek-V3 作为两种粒度的 MoE。所有数字都是从超参数推出来的理论值，不是实测。


## 一、从 dense FFN 到 MoE 层

### 1. dense FFN 的参数与算量

现代 LLM 的 FFN 几乎都是 SwiGLU 形式（Shazeer 2020）：

$$\text{FFN}(x) = W_{down}\,\big(\text{SiLU}(W_{gate}\,x) \odot W_{up}\,x\big)$$

其中 $$W_{gate}, W_{up} \in \mathbb{R}^{d_{ff} \times d}$$，$$W_{down} \in \mathbb{R}^{d \times d_{ff}}$$，三个矩阵合计：

$$N_{FFN} = 3 \cdot d \cdot d_{ff}$$

每个参数在前向中参与一次乘加，即每 token 每参数 2 FLOPs，所以一层 FFN 对一个 token 的算量是 $$2 \cdot 3 d d_{ff} = 6 d d_{ff}$$。代入三个 dense 基线：

```text
                      d        d_ff      3·d·d_ff        每 token FLOPs
Llama-3-8B            4096     14336     176.16M         352.3 MFLOPs
Llama-3-70B           8192     28672     704.6M          1.409 GFLOPs
DeepSeek-V3 dense 层  7168     18432     396.4M          792.7 MFLOPs
```

Llama-3-70B 每层 attention 151.0M、FFN 704.6M，FFN 占每层参数的 82%；80 层合计 68.45B，加 embedding 与 lm_head 2.10B，总计 70.55B。dense 模型中 FFN 是参数的大头，也是 MoE 要改造的部分。

### 2. MoE 层的结构

MoE 把一层里的一个 FFN 换成 $$E$$ 个结构相同、参数独立的 FFN（称为专家），再加一个 router 决定每个 token 用哪几个。对输入 $$x \in \mathbb{R}^d$$：

第一步，router 是一个 $$d \times E$$ 的线性层，输出每个专家的亲和度分数：

$$s = \text{softmax}(W_r\, x) \quad \text{或} \quad s = \sigma(W_r\, x), \qquad W_r \in \mathbb{R}^{E \times d}$$

Mixtral 用 softmax，DeepSeek-V3 用 sigmoid。router 的参数量 $$d \cdot E$$ 相对专家可以忽略——DeepSeek-V3 每层 $$7168 \times 256 = 1.8$$M，58 层 106M，在 671B 里占 0.016%。

第二步，取分数最高的 $$k$$ 个专家（top-$$k$$），把它们的分数归一化为门控权重 $$g_i$$，其余专家的权重为 0：

$$g_i = \frac{s_i}{\sum_{j \in \text{TopK}(s)} s_j} \ \text{（} i \in \text{TopK}\text{）}, \qquad g_i = 0 \ \text{（其余）}$$

第三步，每个被选中的专家是一个独立的 SwiGLU FFN，输出按门控权重加权求和：

$$y = \sum_{i \in \text{TopK}(s)} g_i \cdot \text{FFN}_i(x) \ \big(+ \sum_{j=1}^{n_{shared}} \text{FFN}^{shared}_j(x)\big)$$

括号里的共享专家是 DeepSeek 系列的做法：有 $$n_{shared}$$ 个专家不经过路由、对所有 token 都激活，第七章再讨论它的意义。

对一个 token 而言，这一层只执行了 $$k$$（加共享）个 FFN 的计算，算量是 $$k \cdot 6 d d_{ff}$$，与 $$E$$ 无关。但这一层持有 $$E \cdot 3 d d_{ff}$$ 个参数，全部要放在显存里。参数量与算量的解耦就发生在这里。

这个想法本身不新：Shazeer 等 2017 在 LSTM 上做过稀疏门控的 MoE 层，GShard（Lepikhin 等 2020）与 Switch Transformer（Fedus 等 2021）把它搬到 Transformer 并解决了大规模训练的均衡与并行问题。Mixtral（Jiang 等 2024）与 DeepSeek 系列让它成为开源模型的主流结构。结构层面几年没有大变，变的是粒度与规模，而正是粒度与规模决定了系统成本。

还要注意 MoE 只替换 FFN，attention 部分与 dense 模型完全相同。因此一个 MoE 模型里 attention 的参数、KV cache、对上下文长度的二次方算量，都和同结构的 dense 模型一样——第三、四篇的所有结论直接适用。MoE 改变的只是 FFN 那一块的参数、算量与访存之间的比例关系。

### 3. 一段参考实现

下面是 MoE 层的循环版参考实现，只为说明数据流；生产实现（vLLM 的 fused MoE kernel、Megatron 的 grouped GEMM）会把所有专家的计算合并成一个 grouped GEMM，第五章解释为什么：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUExpert(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.w_gate = nn.Linear(d, d_ff, bias=False)
        self.w_up = nn.Linear(d, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MoELayer(nn.Module):
    def __init__(self, d, d_ff, n_experts, top_k, n_shared=0):
        super().__init__()
        self.router = nn.Linear(d, n_experts, bias=False)       # W_r: d x E
        self.experts = nn.ModuleList(
            [SwiGLUExpert(d, d_ff) for _ in range(n_experts)])
        self.shared = nn.ModuleList(
            [SwiGLUExpert(d, d_ff) for _ in range(n_shared)])
        self.top_k = top_k

    def forward(self, x):                                       # x: [T, d]
        scores = F.softmax(self.router(x), dim=-1)              # [T, E]
        w, idx = scores.topk(self.top_k, dim=-1)                # [T, k]
        w = w / w.sum(dim=-1, keepdim=True)                     # 选中的 k 个重新归一化
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):               # 循环版：逐专家
            tok, slot = (idx == e).nonzero(as_tuple=True)       # 选了专家 e 的 token
            if tok.numel() == 0:
                continue
            out.index_add_(0, tok, w[tok, slot, None] * expert(x[tok]))
        for expert in self.shared:                              # 共享专家：所有 token
            out = out + expert(x)
        return out
```

`(idx == e).nonzero()` 这一步就是"把 token 按专家分组"。在单卡上它是一次 gather；在专家并行下，它变成跨 GPU 的 all-to-all——第四章的主题。

### 4. 两种粒度：Mixtral 与 DeepSeek-V3

两个基线模型代表了 MoE 设计的两种粒度：

```text
                     E（路由专家）  d_ff（专家）  k     共享专家   每专家参数      每 token 激活的 FFN 参数
Mixtral 8x7B         8             14336         2     0          176.16M         352.3M
DeepSeek-V3          256           2048          8     1          44.04M          9 × 44.04M = 396.4M
```

两者每 token 激活的 FFN 参数量相近（352M 与 396M），但 DeepSeek-V3 把它切成了 9 份而不是 2 份。DeepSeek-V2 的论文（DeepSeek-AI 2024）称之为 fine-grained expert segmentation，理由是组合数：

$$\binom{8}{2} = 28, \qquad \binom{256}{8} \approx 4.1 \times 10^{14}$$

在同样的激活参数预算下，Mixtral 每个 token 只有 28 种"专家组合"可选，DeepSeek-V3 有 $$4 \times 10^{14}$$ 种。每个专家更小、更专门化，组合的表达能力更强。这是建模上的收益。

系统上的代价从同一张表就能看出来：DeepSeek-V3 一层有 257 个矩阵组要管理，每个专家的 GEMM 只有 $$7168 \times 2048$$，比 Mixtral 的 $$4096 \times 14336$$ 窄得多；每个 token 要和 8 个专家通信而不是 2 个。后面几章的所有麻烦——访存、all-to-all、GEMM 效率、负载均衡——都在细粒度设计下被放大。


## 二、参数量与激活参数量

### 1. Mixtral 8x7B

Mixtral 的 attention 与 Llama-3-8B 同构（$$d = 4096$$，32 头，8 个 KV 头，$$d_{head} = 128$$）：

$$N_{attn} = 2 \cdot d \cdot n_h d_{head} + 2 \cdot d \cdot n_{kv} d_{head} = 2 \times 4096 \times 4096 + 2 \times 4096 \times 1024 = 41.94\text{M}$$

每层 8 个专家：

$$8 \times 3 \times 4096 \times 14336 = 8 \times 176.16\text{M} = 1.409\text{B}$$

每层合计 1.451B，32 层 46.4B；embedding 与 lm_head 各 $$32000 \times 4096 = 131$$M，合计 0.26B；总参数 **46.7B**。名字里的"8x7B"并不是 $$8 \times 7\text{B} = 56\text{B}$$，因为 attention 与 embedding 在专家之间是共享的。

每 token 激活：每层 attention 41.94M + 2 个专家 352.3M = 394.3M，32 层 12.62B，加 embedding/lm_head 0.26B，**约 12.9B**。激活占总参数的 27.6%。

### 2. DeepSeek-V3 的总参数

DeepSeek-V3 有 61 层，前 3 层是 dense FFN，后 58 层是 MoE。逐项算：

**路由专家。** 每个专家 $$3 \times 7168 \times 2048 = 44.04\text{M}$$；每层 256 个路由专家 + 1 个共享专家 = 257 个：

$$257 \times 44.04\text{M} = 11.32\text{B}$$

58 个 MoE 层：

$$58 \times 11.32\text{B} = 656.5\text{B}$$

其中路由专家 $$58 \times 256 \times 44.04\text{M} = 653.9\text{B}$$，共享专家 $$58 \times 44.04\text{M} = 2.55\text{B}$$。

**dense 层 FFN。** 3 层，每层 $$3 \times 7168 \times 18432 = 396.4\text{M}$$，合计 1.19B。

**MLA attention。** DeepSeek-V3 用 MLA（第三篇有完整推导），每层的权重是六个矩阵：

```text
W_DQ    7168 × 1536     11.01M    q 的下投影
W_UQ    1536 × 24576    37.75M    q 的上投影，24576 = 128 头 × 192
W_DKV   7168 × 576       4.13M    kv 的联合下投影，576 = d_c 512 + d_h^R 64
W_UK     512 × 16384     8.39M    k 的上投影，16384 = 128 头 × 128
W_UV     512 × 16384     8.39M    v 的上投影
W_O    16384 × 7168    117.44M    输出投影
合计                   187.1M
```

61 层 $$\times$$ 187.1M = 11.41B。

**embedding 与 lm_head。** $$129280 \times 7168 = 926.7\text{M}$$，两个不共享，合计 1.85B。

**加总：**

$$656.5 + 1.19 + 11.41 + 1.85 \approx 671\text{B}$$

再加 router（0.11B）与 RMSNorm，与技术报告的 671B 一致。结构非常倾斜：97.4% 的参数是路由专家，attention 只占 1.7%。

### 3. DeepSeek-V3 的激活参数

每个 token 经过一个 MoE 层时用到 8 个路由专家 + 1 个共享专家 = 9 个：

$$58 \times 9 \times 44.04\text{M} = 22.99\text{B}$$

加上永远激活的部分：attention 11.41B、dense FFN 1.19B、embedding/lm_head 1.85B，合计：

$$22.99 + 11.41 + 1.19 + 1.85 \approx 37.4\text{B}$$

技术报告取整写作 37B。注意这个 37B 里有 14.4B（attention + dense + embedding）是"dense 部分"，和 MoE 无关；路由专家只贡献 20.4B。

### 4. 算量按激活参数算，显存按总参数算

每 token 的前向 FLOPs 只和真正参与运算的参数有关：

$$\text{FLOPs/token} \approx 2 \times 37\text{B} = 74\text{ GFLOPs}$$

对照 Llama-3-70B 的 $$2 \times 70.55\text{B} \approx 141$$ GFLOPs，DeepSeek-V3 每 token 的算量只有它的一半。这是 MoE 的全部承诺：用 671B 的参数容量，付 37B 的算量。

但显存按总参数算。FP8 权重（DeepSeek-V3 原生以 FP8 训练与发布）：

$$671\text{B} \times 1\text{ byte} = 671\text{ GB}$$

一台 8 卡 H100 的 HBM 总量是 $$8 \times 80 = 640$$ GB，放不下权重本身，更不用说 KV cache 与激活。至少要两台（1280 GB），而且两台也只是"放得下"。如果用 BF16，1342 GB，至少三台。DeepSeek 报告里的实际部署规模远大于此：prefill 用 4 节点 32 卡（EP32），decode 用 40 节点 320 卡（EP320）。为什么要用这么大的 EP，是第三章要算的东西。

值得一提的是，DeepSeek-V3 的显存压力几乎全部来自权重而不是 KV cache。MLA 让它的 KV cache 每 token 每层只有 $$(512 + 64) \times 2 = 1152$$ 字节，61 层 68.6 KiB，128K 上下文只要 8.6 GiB（第三篇的推导）。Llama-3-70B 的 GQA 每 token 320 KiB，128K 上下文 40 GiB。也就是说，两个模型的显存构成刚好相反：70B 是权重 141 GB、KV 随并发膨胀；V3 是权重 671 GB、KV 很小。这也是 V3 选择 MLA 的原因之一——权重已经占掉这么多，KV cache 必须压到极致，才能在 EP 集群的每张卡上留出足够的并发空间。

Mixtral 也有同样的问题，只是数量级小：BF16 权重 93.4 GB，单张 H100 放不下，至少两卡；算量却只有 $$2 \times 12.9 \approx 26$$ GFLOPs/token，比 Llama-3-8B 的 15 GFLOPs 多不到一倍。它的 KV cache 与 Llama-3-8B 完全相同（每 token 128 KiB），两卡 TP 部署后剩余的显存足够放几十万 token 的 KV。从"算量像 13B、显存像 47B"这一点看，Mixtral 是理解 MoE 部署形态最温和的入门例子；DeepSeek-V3 则把同样的矛盾放大了一个数量级。

到这里，核心问题的前两个数已经有了：参数量 671B，激活参数量 37B。第三个数——每步实际读取的参数量——需要看 decode 时 batch 里的 token 是怎样分布在专家上的。


## 三、decode 的访存形态：稀疏在访存上不成立

### 1. 期望激活专家数的推导

decode 阶段每步为 batch 里的 $$B$$ 个序列各生成一个 token，这 $$B$$ 个 token 各自独立路由到 $$k$$ 个专家。问题是：一层的 $$E$$ 个专家中，有多少个至少被一个 token 选中？被选中的专家权重必须从 HBM 读出来，没被选中的不必。

做两个简化假设：路由均匀（每个专家被某个 token 选中的概率相同），且 token 之间独立。在均匀假设下，一个 token 选 $$k$$ 个专家，某个特定专家在其中的概率是 $$k/E$$，不在其中的概率是 $$1 - k/E$$。$$B$$ 个 token 都不选它的概率是 $$(1 - k/E)^B$$，于是它被至少一个 token 选中的概率是 $$1 - (1 - k/E)^B$$。对 $$E$$ 个专家求期望：

$$\mathbb{E}[\text{激活专家数}] = E \cdot \left[1 - \left(1 - \frac{k}{E}\right)^B\right]$$

$$B = 1$$ 时它等于 $$k$$（一个 token 恰好激活 $$k$$ 个）；$$B \to \infty$$ 时趋于 $$E$$。代入 DeepSeek-V3（$$E = 256$$，$$k = 8$$，$$1 - k/E = 0.96875$$）和 Mixtral（$$E = 8$$，$$k = 2$$，$$1 - k/E = 0.75$$）：

```text
B         DeepSeek-V3 (E=256, k=8)      Mixtral (E=8, k=2)
          期望激活专家   占 E 比例        期望激活专家   占 E 比例
1           8.0           3.1%           2.0           25%
8          57.4          22.4%           7.2           90%
32        163.3          63.8%           8.0          100%
64        222.4          86.9%           8.0          100%
128       251.6          98.3%           8.0          100%
512       256.0         100%             8.0          100%
```

Mixtral 在 $$B = 8$$ 时就几乎读全部专家；DeepSeek-V3 在 $$B = 32$$ 时读 64%，$$B = 128$$ 时读 98%。真实路由并不均匀（热门专家被选中的概率更高），这会让期望激活数比均匀假设略低，但结论不变：**中等 batch 下，几乎所有专家的权重每一步都要读一遍。**

### 2. 每步实际读取的参数量

把每步读取量拆成两部分：一部分是无论 batch 多大都要读的（attention、dense FFN、共享专家、lm_head），另一部分是被激活的路由专家。对 DeepSeek-V3：

$$N_{read}(B) = N_{always} + 58 \times \mathbb{E}[\text{激活专家数}](B) \times 44.04\text{M}$$

其中 $$N_{always} = 671 - 653.9 \approx 17\text{B}$$（其中 embedding 0.93B 严格说只查表读一行，这里不细扣）。代入：

```text
B        期望激活专家    路由专家读取量     每步读取参数量    FP8 字节数
1            8.0           20.4B             37.6B            37.6 GB
8           57.4          146.6B            163.8B           164 GB
32         163.3          417.1B            434.3B           434 GB
128        251.6          642.7B            659.8B           660 GB
512        256.0          653.9B            671.0B           671 GB
```

对照 Llama-3-70B：无论 batch 多大，每步读 70.55B 参数，BF16 141 GB。

也就是说，只有 $$B = 1$$ 时 DeepSeek-V3 的"每步读 37B"才成立。$$B = 32$$ 时它每步读的字节数（FP8 434 GB）已经是 Llama-3-70B（BF16 141 GB）的 3 倍；$$B = 128$$ 时是 4.7 倍。第二篇给出的 decode 时间下界是"权重字节数 / HBM 带宽"，按 H100 的 3.35 TB/s，假设权重能放在一张卡上（当然放不下，这里只为比较）：Llama-3-70B 每步下界 42 ms，DeepSeek-V3 在 $$B = 32$$ 时每步下界 130 ms。稀疏节省了算量，但没有节省访存——而 decode 恰恰是访存瓶颈的阶段。

### 3. 三个数分开算

现在可以完整回答核心问题：

```text
                                   DeepSeek-V3            Llama-3-70B          两者之比
参数量（决定显存）                    671B                   70.55B               9.5x
激活参数量（决定 FLOPs/token）        37B                    70.55B               0.52x
每步读取参数量（决定 decode 带宽）
    B = 1                            37.6B                  70.55B               0.53x
    B = 32                           434B                   70.55B               6.2x
    B = 128                          660B                   70.55B               9.4x
```

"部署 DeepSeek-V3 比 dense 70B 难得多"的原因是第一行和第三行：显存要 9.5 倍，中等 batch 下每步访存要 6–9 倍。只有第二行对 MoE 有利，而 decode 阶段的瓶颈本来就不在第二行。

### 4. 大规模 EP 的动机

出路只有两条。一条是把 batch 压到极小，让每步只读 $$k$$ 个专家——这放弃了吞吐，单卡也放不下 671 GB。另一条是把专家分散到很多张卡上：如果 256 个专家均匀分到 $$N$$ 张卡，每张卡每层只持有 $$256/N$$ 个专家，每步最多只需读自己那 $$256/N$$ 个，而不是全部 256 个。这就是专家并行（Expert Parallelism，EP）。

以 EP32 为例（DeepSeek-V3 prefill 的配置），每卡每层 8 个路由专家，58 层共 $$58 \times 8 \times 44.04\text{M} = 20.4\text{B}$$ 参数，FP8 20.4 GB；加上复制到每卡的 attention、dense、共享专家等约 17 GB，每卡权重约 37 GB。EP320（decode 配置）时每卡每层只有 1 个专家（报告中还为热门专家配置了冗余副本），路由专家部分只有 2.55 GB。

把不同 EP 规模下每卡的权重占用列出来（FP8，路由专家均匀分布，非专家部分约 17 GB 在每卡复制；实际部署中 attention 还会做 TP 或数据并行切分，这里为简单起见按复制算）：

```text
EP 规模    每卡每层路由专家数    每卡路由专家权重    每卡权重合计    剩余显存（80 GB 卡）
1          256                  654 GB             671 GB         放不下
8          32                   81.7 GB            98.7 GB        放不下
16         16                   40.9 GB            57.9 GB        22 GB
32         8                    20.4 GB            37.4 GB        43 GB
64         4                    10.2 GB            27.2 GB        53 GB
320        1（256 + 64 冗余副本） 2.6 GB           19.6 GB        60 GB
```

（EP320 一行按 DeepSeek-V3 报告的 decode 配置：256 个路由专家加 64 个热门专家的冗余副本，共 320 份，每卡 1 份。）

EP16 是让 FP8 权重放得下的最小规模，但每卡只剩 22 GB 给 KV cache 与激活；EP32 之后剩余显存才宽裕起来。EP 让每卡的权重读取量从"随 batch 趋近 671B"回到几十 GB，让 HBM 装得下权重之外还留出 KV cache 的空间。代价是原本在一张卡内部完成的"按专家分组"，变成了跨卡通信。


## 四、专家并行与 all-to-all

### 1. dispatch 与 combine

EP 下一个 MoE 层的执行流程：

1. 每张卡对自己持有的 token 跑 router，得到每个 token 的 $$k$$ 个目标专家；
2. **dispatch**：把每个 token 的 hidden state（$$d$$ 个数）发到它的 $$k$$ 个专家所在的卡——这是一次 all-to-all；
3. 每张卡对收到的 token 跑自己的专家（grouped GEMM）；
4. **combine**：把每个专家的输出发回 token 所在的卡，按门控权重加权求和——第二次 all-to-all。

attention 部分在 EP 下通常是数据并行的（每卡处理 batch 的一个切片，各自持有自己序列的 KV cache），所以 token "所在的卡"是明确的。每个 MoE 层两次 all-to-all，58 层共 116 次，全部在前向的关键路径上。

### 2. 字节数

每个 token 发给每个专家的是一个长度 $$d = 7168$$ 的向量。DeepSeek-V3 的做法是 dispatch 用 FP8、combine 用 BF16（combine 要做加权求和，精度要求更高）：

```text
dispatch  每 token 每专家   7168 × 1 B = 7168 B  = 7 KiB
combine   每 token 每专家   7168 × 2 B = 14336 B = 14 KiB
```

top-8：

```text
dispatch  每 token 每层   8 × 7 KiB  = 56 KiB
combine   每 token 每层   8 × 14 KiB = 112 KiB
合计      每 token 每层   168 KiB
58 层     每 token       9744 KiB ≈ 9.5 MiB
```

严格说要扣掉恰好落在本卡的那部分：$$N$$ 卡均匀分布时约 $$1/N$$ 的专家在本地，EP32 时扣 3%，EP320 时扣 0.3%，可以忽略。

乘上 batch。一张卡持有 $$B_{local}$$ 个 token 时，每层要发出并收回 $$B_{local} \times 168$$ KiB：

```text
B_local     每层 all-to-all 字节     58 层合计      按 50 GB/s（IB）的传输时间下界
32          5.25 MiB                 305 MiB        6.4 ms
128         21 MiB                   1.19 GiB       25 ms
```

prefill 阶段的数字更直观：一个 4096 token 的序列，每层 dispatch + combine 共 $$4096 \times 168\text{ KiB} = 672$$ MiB，58 层 38 GiB。在 EP32 下这 4096 个 token 分摊在 32 张卡上，每卡每层收发 21 MiB、58 层 1.2 GiB；若全部走 50 GB/s 的 IB，每卡传输时间下界约 26 ms。而这 4096 个 token 的算量是 $$4096 \times 74\text{ GFLOPs} \approx 303$$ TFLOP，在 32 张 H100 上按 FP8 60% MFU 大约 8 ms。通信是计算的 3 倍以上——这说明 EP 下的 prefill 如果不把 all-to-all 与计算充分重叠、不把大部分流量留在 NVLink 域内，通信会主导时间。DeepSeek-V3 的节点受限路由（下一节）和 EP32 只跨 4 个节点的配置，都是在压这个比例。

作为参照，第二篇算过 Llama-3-70B 在 8 卡 TP 下每步 decode 的权重读取下界约 5 ms（141 GB / 8 卡 / 3.35 TB/s）。EP 下每卡的权重读取只剩几十 GB（约 10 ms），但 all-to-all 又添上了同一量级的通信时间——而且这个时间与 $$B_{local}$$ 线性增长，权重读取时间则不随 batch 增长。DeepSeek-V3 报告用 DualPipe、把通信 kernel 限制在少量 SM 上与计算重叠、以及自定义 all-to-all kernel，都是在处理这项开销。

### 3. 节点受限路由

跨节点走 InfiniBand，节点内走 NVLink，两者带宽差一个档次。DeepSeek-V3 报告中给出的数字是 IB 约 50 GB/s、NVLink 约 160 GB/s，比值约 1 : 3.2。

如果 8 个专家随机分布在 40 个节点上，一个 token 要跨 IB 发到接近 8 个不同节点。DeepSeek-V3 在路由时加了一条限制：**每个 token 最多发到 4 个节点**（node-limited routing）。做法是先按每个节点上专家的亲和度之和（取节点内 top-3 专家分数相加）选出 4 个节点，再在这 4 个节点的专家里取 top-8。这样每 token 的跨 IB 流量最多是 4 份 dispatch + 4 份 combine，而不是 8 份。

报告还给了一个基于带宽比的说法：一份数据经 IB 到达某节点后，可以经 NVLink 转发给节点内的多张 GPU；由于 NVLink 带宽是 IB 的 3.2 倍，在 IB 传输时间内 NVLink 可以把它转发给约 3.2 个目标而不成为瓶颈——报告的表述是每个 token 可以"等价地"路由到 $$4 \times 3.2 \approx 13$$ 个专家而不增加通信开销，top-8 在这个上限之内。这是报告的近似论证，转述于此供理解设计意图。

节点受限路由是训练时就加进路由规则的，不是部署时的优化——它改变了模型，因此必须在训练时就决定。这是 MoE 设计里"系统约束反过来塑造模型结构"的一个直接例子。

### 4. EP 与 TP 的对比

MoE 的 FFN 有两种切法。

**张量并行（TP）** 切每个专家的矩阵：$$n$$ 卡 TP 下，每个专家的 $$W_{gate}, W_{up}$$ 按列切成 $$n$$ 份、$$W_{down}$$ 按行切成 $$n$$ 份，每张卡持有所有 $$E$$ 个专家的 $$1/n$$。每张卡对所有 token 算所有专家的局部结果，最后做一次 all-reduce。all-reduce 的通信量是标准结论：对 $$B$$ 个 token 的 $$[B, d]$$ 输出，每卡收发

$$2 \cdot \frac{n-1}{n} \cdot B \cdot d \cdot \text{bytes/elem}$$

DeepSeek-V3 的 $$d = 7168$$、BF16、$$n = 8$$：每 token 每层 $$2 \times 7/8 \times 7168 \times 2 = 25$$ KiB。这个数与 dense FFN 的 TP 通信完全相同，**与 $$k$$、$$E$$ 都无关**。

**专家并行（EP）** 按专家切，通信是上一节的 168 KiB 每 token 每层，**与 $$k$$ 成正比，与 $$E$$ 无关**。

按字节数看，DeepSeek-V3 的 EP 通信是 TP-8 的 6.7 倍。那为什么 DeepSeek-V3 不用 TP？三个原因：

第一，TP 的规模上限是一台机器。NVLink 域是 8 卡，跨节点 TP 会让每层两次 all-reduce（attention 与 FFN 各一次）走 IB，延迟不可接受。而 671 GB 权重两台机器都放不下多少 KV cache，TP-8 从一开始就不够。

第二，TP 不减少每卡的专家读取数。TP-8 下每卡持有全部 256 个专家的 1/8，batch 32 时仍然要读 163 个专家的 1/8 切片——每卡读取量是 $$434 / 8 = 54$$ GB，而 EP32 每卡只读自己 8 个专家中被激活的（至多 $$8 \times 58 \times 44.04\text{M} = 20.4$$ GB，加上复制的 17 GB）。

第三，也是最根本的：GEMM 形状。专家的 $$d_{ff} = 2048$$，TP-8 切完每卡只剩 256 列，$$W_{gate}$$ 的切片是 $$[7168, 256]$$。这么瘦的矩阵，GEMM 的 N 维只有 256，Tensor Core 的 tile 利用率与访存效率都差。而 EP 下每卡的专家 GEMM 保持完整的 $$[7168, 2048]$$ 形状。Mixtral 的专家 $$d_{ff} = 14336$$，TP-8 切完还有 1792 列，GEMM 形状仍然健康，所以 Mixtral 在单机 8 卡上用 TP 是完全可行的选择——vLLM 的默认 MoE 部署就是这样。

粗略的判据：**专家少、每个专家宽（Mixtral），TP 划算，通信量小且 GEMM 形状好；专家多、每个专家窄（DeepSeek-V3），TP 切出来的矩阵太瘦，必须用 EP，接受 all-to-all 的代价。** 实际部署常常两者混用：attention 部分 TP 或数据并行，专家部分 EP。


## 五、grouped GEMM 的形态

### 1. 每专家平均行数

dense FFN 对 $$T$$ 个 token 做的 GEMM 是 $$[T, d] \times [d, d_{ff}]$$，M 维等于 $$T$$。prefill 一个 4096 token 的序列，M = 4096，是 Tensor Core 最喜欢的形状。

MoE 层把 $$T$$ 个 token 各复制 $$k$$ 份分到 $$E$$ 个专家，每个专家平均收到

$$\frac{T \cdot k}{E}$$

个 token。代入：

```text
                        T        k     E      每专家平均行数 Tk/E    dense FFN 的 M
DeepSeek-V3 prefill     4096     8     256    128                   4096
DeepSeek-V3 decode      32       8     256    1                     32
DeepSeek-V3 decode      128      8     256    4                     128
Mixtral prefill         4096     2     8      1024                  4096
Mixtral decode          32       2     8      8                     32
```

DeepSeek-V3 prefill 4096 token，每个专家只有 128 行；decode batch 32，每个专家平均 1 行。同样的 token 数，dense FFN 是一个 M = 4096 的大 GEMM，MoE 是 256 个 M = 128 的小 GEMM。

### 2. 为什么 MoE 的 GEMM 天然低效

Tensor Core GEMM kernel 以 tile 为单位计算，典型的 tile 是 128 × 128 或 128 × 256（M × N）。M = 128 恰好填满一个 tile，没有浪费，但一个专家的 GEMM 只有一个 tile 行，无法在 M 方向上做多 tile 的流水与负载分配；M = 1 时 tile 的 128 行中只有 1 行有效，算力利用率是 1/128。

grouped GEMM 是对这个问题的工程回答：把 $$E$$ 个不同形状（行数各异）、共享 K 与 N 维的小 GEMM 打包成一个 kernel launch，让 GPU 的 SM 在专家之间做负载分配，避免 256 次 launch 的开销和 SM 空闲。CUTLASS 的 grouped GEMM、vLLM 的 fused MoE Triton kernel、Megatron 的 grouped GEMM 后端都是这个思路。它解决了 launch 开销与 SM 利用率问题，但没有改变每个专家 M 小这一事实：decode 阶段的 MoE 层，本质上还是在为每个专家读一遍 $$[7168, 2048]$$ 的权重然后只乘 1–4 行——第三章算过的"访存量按激活专家数"，正是这里的直接体现。

第一篇的参数量、第二篇的 FLOPs 在 MoE 上都成立；不成立的是第二篇 Roofline 分析中"batch $$B$$ 时权重 GEMM 算术强度约为 $$B$$ FLOP/byte"这条：MoE 层里每个专家的算术强度是 $$Tk/E$$ 而不是 $$T$$，比 dense 低 $$E/k = 32$$ 倍（DeepSeek-V3）。要让专家 GEMM 越过 H100 的 ridge point（约 295 FLOP/byte，BF16），需要每专家至少 300 行左右，即 $$T \geq 300 \times 32 \approx 9600$$ 个 token 同时在一层——这在 prefill 可以做到，在 decode 只有靠 EP 把大量并发请求的 token 汇聚到同一个专家上。DeepSeek-V3 用 EP320 做 decode，320 卡上的所有请求在每个专家上汇聚，是让专家 GEMM 有足够 M 的另一个理由。


## 六、负载均衡

前面所有推导都假设路由均匀。实际训练中 router 会自发地偏爱少数专家（被选多的专家训练得更好，因此被选得更多），如果不加干预会塌缩到几个专家上。负载不均既是建模问题，也是系统问题。

### 1. 辅助损失

Switch Transformer（Fedus 等 2021）的做法是在训练目标里加一项辅助损失。对一个 batch 的 $$T$$ 个 token，定义

$$f_i = \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}\{\text{token } t \text{ 选中专家 } i\}, \qquad P_i = \frac{1}{T} \sum_{t=1}^{T} s_{t,i}$$

$$f_i$$ 是实际分到专家 $$i$$ 的 token 比例（不可微），$$P_i$$ 是 router 给专家 $$i$$ 的平均概率（可微）。辅助损失：

$$\mathcal{L}_{aux} = \alpha \cdot E \cdot \sum_{i=1}^{E} f_i \cdot P_i$$

在 $$\sum f_i = \sum P_i = 1$$（top-1 时）的约束下，这个内积在 $$f_i = P_i = 1/E$$ 时取最小值 $$\alpha$$；某个专家的 $$f_i$$ 偏高时，梯度会压低它的 $$P_i$$。系数 $$\alpha$$ 取 0.01 量级：太小压不住塌缩，太大会为了均衡牺牲路由质量——这是 aux loss 方法的固有张力。

### 2. 容量因子与 token drop

训练时每个专家的缓冲区大小是固定的，由容量因子（capacity factor）$$C$$ 决定：

$$\text{capacity} = C \cdot \frac{T \cdot k}{E}$$

$$C = 1$$ 时容量恰好等于均匀分配下的平均行数；Switch Transformer 用 1.0–1.25。收到的 token 超过容量的专家丢弃多余 token——这些 token 在这一层不经过 FFN，只沿残差连接直接通过。$$C$$ 越大丢弃越少但填充（padding）越多、计算浪费越大。

固定容量的动机是系统性的：训练框架希望每个专家的输入是一个形状固定的张量 $$[\text{capacity}, d]$$，这样 all-to-all 的缓冲区大小、GEMM 的形状在编译期就能确定，不需要动态分配。代价是两头浪费——欠载的专家要 padding 到 capacity，超载的专家要丢 token。以 DeepSeek-V3 prefill 4096 token、$$C = 1.25$$ 为例，每专家容量 160 行；若某个专家实际收到 200 行，40 行被丢弃（20%）；若只收到 80 行，另外 80 行是零填充，GEMM 的一半算力浪费。

推理时一般不丢弃（dropless），改用动态大小的 grouped GEMM——每个专家有多少行就算多少行，这正是 grouped GEMM 支持不等行数的原因。因此训练与推理在 token drop 上的行为存在差异，训练时被丢弃过的 token 在推理时会正常经过专家；这个差异在实践中通常可以接受，但它是"训练时的均衡策略如何影响推理形态"的一个例子。DeepSeek-V3 训练时就不做 token drop，避免了这个差异。

### 3. aux-loss-free：DeepSeek-V3 的偏置调节

DeepSeek-V3（DeepSeek-AI 2024）采用 auxiliary-loss-free 的均衡策略。给每个专家一个标量偏置 $$b_i$$，top-$$k$$ 选择时用 $$s_i + b_i$$ 排序，但门控权重仍用原始的 $$s_i$$：

$$\text{TopK}(s_1 + b_1, \ldots, s_E + b_E) \quad \text{选专家}; \qquad g_i = \frac{s_i}{\sum_{j \in \text{TopK}} s_j} \quad \text{算权重}$$

每个训练步结束后统计各专家的负载：超载的专家 $$b_i \leftarrow b_i - \gamma$$，欠载的 $$b_i \leftarrow b_i + \gamma$$，$$\gamma$$ 是很小的常数（报告中 0.001）。偏置只影响"选谁"，不影响"选中后的权重"，因此不会像 aux loss 那样把梯度噪声注入主目标；负载均衡变成了一个在训练目标之外运行的控制回路。报告中还保留了一项系数极小的序列级均衡损失（$$\alpha = 10^{-4}$$）作为兜底，防止单个序列内的极端不均。

### 4. 负载不均对 EP 意味着什么

EP 下一层的时间由最慢的那张卡决定——所有卡都要等 combine 的 all-to-all 完成才能进入下一层。设每卡持有 1 个专家（EP256/EP320 的情形），平均每个专家收到 $$Tk/E$$ 个 token。如果某个专家收到 2 倍平均的 token：

- 它所在卡的 grouped GEMM 时间翻倍（M 翻倍；在访存瓶颈的 decode 阶段 M 翻倍时间不一定翻倍，但在 prefill 是接近线性的）；
- 它收到的 dispatch 字节数翻倍，发出的 combine 字节数翻倍——all-to-all 的完成时间由最大的收发方决定；
- 其余 255 张卡在这一层的后半段空闲。

结果是这一层的耗时约为均衡时的 2 倍，全层的算力利用率降到约 50%。58 层里只要几层出现热点，整体吞吐就明显下降。这就是为什么 DeepSeek-V3 的部署方案里有"冗余专家"（把热门专家复制到多张卡上）与周期性根据负载统计重排专家的机制——EP 下负载均衡不再只是训练时的建模问题，而是推理时的调度问题。


## 七、共享专家与 MTP

### 1. 共享专家其实是 dense FFN

DeepSeek-V3 每个 MoE 层有 1 个共享专家，$$d_{ff} = 2048$$，不经过 router，对每个 token 都执行。从系统角度看它就是一个 dense 的窄 FFN：参数 44.04M，每 token 算量 $$2 \times 44.04\text{M}$$，每步无论 batch 多大都要读一遍，不参与 all-to-all（在每张卡上复制）。58 层共 2.55B 参数，算进 37B 激活参数，也算进第三章的 $$N_{always}$$。

建模上的动机（DeepSeek-V2 论文）是让共享专家吸收所有 token 都需要的"通用知识"，使路由专家更专门化、减少专家间的冗余。系统上它是 MoE 层里唯一"行为像 dense"的部分，可以和 attention 一起用 TP 或直接复制，不增加通信。

### 2. MTP：内置的投机草稿

DeepSeek-V3 在主模型之外训练了一个多 token 预测（Multi-Token Prediction，MTP）模块，用第 $$t$$ 个位置的 hidden state 额外预测第 $$t+2$$ 个 token。推理时这个模块可以直接当作投机解码的草稿模型：主模型一步产生一个 token，MTP 头顺带猜出下一个，再由主模型验证。报告中的接受率在 85%–90%，相当于每步 decode 平均产出接近 1.8 个 token。投机解码的期望加速与接受率的关系，第七篇会展开。


## 八、实践：llm_cost.py 的 MoE 支持

在系列脚本 `llm_cost.py` 的 `ModelConfig` 上增加 MoE 字段，并增加四个函数：`moe_param_count`、`active_params`、`expected_active_experts`、`ep_all_to_all_bytes_per_layer`。为了让本篇代码独立可运行，把用到的 attention 参数函数（含 MLA）也一并给出：

```python
from dataclasses import dataclass
from math import comb


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
    # 第三篇加入：MLA（mla_rank=0 表示 GQA/MHA）
    mla_rank: int = 0        # d_c
    rope_dim: int = 0        # d_h^R
    q_lora_rank: int = 0
    # 第五篇加入：MoE（n_experts=0 表示 dense）
    n_experts: int = 0       # 路由专家数 E
    top_k: int = 0           # 每 token 激活的路由专家数 k
    expert_d_ff: int = 0     # 每个专家的 d_ff
    n_shared: int = 0        # 共享专家数（永远激活）
    moe_layers: int = 0      # MoE 层数，其余 layers - moe_layers 层为 dense
    dense_d_ff: int = 0      # dense 层的 d_ff（0 表示与 d_ff 相同）


LLAMA3_70B = ModelConfig("Llama-3-70B", 8192, 80, 64, 8, 128, 28672, 128256)

MIXTRAL_8X7B = ModelConfig(
    "Mixtral-8x7B", 4096, 32, 32, 8, 128, 14336, 32000,
    n_experts=8, top_k=2, expert_d_ff=14336, n_shared=0, moe_layers=32,
)

DEEPSEEK_V3 = ModelConfig(
    "DeepSeek-V3", 7168, 61, 128, 128, 192, 18432, 129280,
    mla_rank=512, rope_dim=64, q_lora_rank=1536,
    n_experts=256, top_k=8, expert_d_ff=2048, n_shared=1,
    moe_layers=58, dense_d_ff=18432,
)


def attn_params(cfg):
    """每层 attention 的权重参数量（GQA/MHA 或 MLA）。"""
    d, h = cfg.hidden, cfg.n_heads
    if cfg.mla_rank:
        nope = cfg.head_dim - cfg.rope_dim                 # 128
        w_dq = d * cfg.q_lora_rank                         # 7168 x 1536
        w_uq = cfg.q_lora_rank * h * cfg.head_dim          # 1536 x 24576
        w_dkv = d * (cfg.mla_rank + cfg.rope_dim)          # 7168 x 576
        w_uk = cfg.mla_rank * h * nope                     # 512 x 16384
        w_uv = cfg.mla_rank * h * nope                     # 512 x 16384
        w_o = h * nope * d                                 # 16384 x 7168
        return w_dq + w_uq + w_dkv + w_uk + w_uv + w_o
    q_o = 2 * d * h * cfg.head_dim
    k_v = 2 * d * cfg.n_kv_heads * cfg.head_dim
    return q_o + k_v


def ffn_params(d, d_ff):
    """SwiGLU FFN：gate、up、down 三个矩阵。"""
    return 3 * d * d_ff


def moe_param_count(cfg):
    """返回 (总参数量, 分项字典)。n_experts=0 时退化为 dense 模型。"""
    d = cfg.hidden
    n_dense = cfg.layers - cfg.moe_layers
    dense_ff = cfg.dense_d_ff or cfg.d_ff
    per_expert = ffn_params(d, cfg.expert_d_ff) if cfg.n_experts else 0
    parts = {
        "attention": cfg.layers * attn_params(cfg),
        "dense_ffn": n_dense * ffn_params(d, dense_ff),
        "router": cfg.moe_layers * d * cfg.n_experts,
        "routed_experts": cfg.moe_layers * cfg.n_experts * per_expert,
        "shared_experts": cfg.moe_layers * cfg.n_shared * per_expert,
        "norms": (2 * cfg.layers + 1) * d,
        "embedding": cfg.vocab * d * (1 if cfg.tie_embeddings else 2),
    }
    return sum(parts.values()), parts


def active_params(cfg):
    """每 token 激活的参数量：总参数减去未被选中的路由专家。"""
    total, parts = moe_param_count(cfg)
    if not cfg.n_experts:
        return total
    per_expert = ffn_params(cfg.hidden, cfg.expert_d_ff)
    return total - parts["routed_experts"] + cfg.moe_layers * cfg.top_k * per_expert


def expected_active_experts(cfg, batch):
    """batch 个 token 独立、均匀路由时，一层里期望被至少一个 token 选中的专家数。"""
    E, k = cfg.n_experts, cfg.top_k
    if not E:
        return 0.0
    return E * (1 - (1 - k / E) ** batch)


def params_read_per_step(cfg, batch):
    """一步 decode 期望从 HBM 读取的参数量（假设全模型在一张卡上，仅用于比较）。"""
    total, parts = moe_param_count(cfg)
    if not cfg.n_experts:
        return total
    per_expert = ffn_params(cfg.hidden, cfg.expert_d_ff)
    read_experts = cfg.moe_layers * expected_active_experts(cfg, batch) * per_expert
    return total - parts["routed_experts"] + read_experts


def ep_all_to_all_bytes_per_layer(cfg, batch, dispatch_bytes=1, combine_bytes=2,
                                  ep_size=None):
    """一个 MoE 层 dispatch + combine 的 all-to-all 字节数（batch 个 token 合计）。
    dispatch_bytes / combine_bytes 是每个元素的字节数（DeepSeek-V3：FP8 / BF16）。
    给出 ep_size 时，按均匀假设扣掉落在本卡的 1/ep_size。"""
    per_token = cfg.top_k * cfg.hidden * (dispatch_bytes + combine_bytes)
    total = batch * per_token
    if ep_size:
        total *= (ep_size - 1) / ep_size
    return total


def routing_combinations(cfg):
    return comb(cfg.n_experts, cfg.top_k) if cfg.n_experts else 1


if __name__ == "__main__":
    KiB = 1024
    for cfg in (MIXTRAL_8X7B, DEEPSEEK_V3, LLAMA3_70B):
        total, parts = moe_param_count(cfg)
        print(f"== {cfg.name}")
        for key, val in parts.items():
            print(f"  {key:16s} {val/1e9:8.3f} B")
        print(f"  total            {total/1e9:8.2f} B")
        print(f"  active           {active_params(cfg)/1e9:8.2f} B")
        print(f"  flops/token      {2*active_params(cfg)/1e9:8.1f} GFLOPs")
        if cfg.n_experts:
            # Mixtral 的 dispatch/combine 都按 BF16 算；DeepSeek-V3 按 FP8/BF16
            db = 1 if cfg.mla_rank else 2
            print(f"  C(E,k)           {routing_combinations(cfg):.3e}")
            for B in (1, 8, 32, 128, 512):
                n_act = expected_active_experts(cfg, B)
                read = params_read_per_step(cfg, B)
                a2a = ep_all_to_all_bytes_per_layer(cfg, B, dispatch_bytes=db) / KiB
                print(f"  B={B:4d}  active experts {n_act:7.1f}  read {read/1e9:7.1f} B"
                      f"  a2a/layer {a2a:9.0f} KiB")
```

运行输出（节选）：

```text
== Mixtral-8x7B
  attention           1.342 B
  routed_experts     45.097 B
  embedding           0.262 B
  total               46.70 B
  active              12.88 B
  flops/token          25.8 GFLOPs
  C(E,k)           2.800e+01
  B=   1  active experts     2.0  read    12.9 B  a2a/layer        32 KiB
  B=   8  active experts     7.2  read    42.2 B  a2a/layer       256 KiB
  B=  32  active experts     8.0  read    46.7 B  a2a/layer      1024 KiB
  B= 128  active experts     8.0  read    46.7 B  a2a/layer      4096 KiB
== DeepSeek-V3
  attention          11.413 B
  dense_ffn           1.189 B
  router              0.106 B
  routed_experts    653.909 B
  shared_experts      2.554 B
  embedding           1.853 B
  total              671.03 B
  active              37.55 B
  flops/token          75.1 GFLOPs
  C(E,k)           4.097e+14
  B=   1  active experts     8.0  read    37.6 B  a2a/layer       168 KiB
  B=   8  active experts    57.4  read   163.8 B  a2a/layer      1344 KiB
  B=  32  active experts   163.3  read   434.3 B  a2a/layer      5376 KiB
  B= 128  active experts   251.6  read   659.8 B  a2a/layer     21504 KiB
  B= 512  active experts   256.0  read   671.0 B  a2a/layer     86016 KiB
== Llama-3-70B
  total               70.55 B
  active              70.55 B
  flops/token        141.1 GFLOPs
```

几点核对：

- Mixtral 总参数 46.70B、激活 12.88B，与第二章手算一致；
- DeepSeek-V3 总参数 671.03B，与技术报告的 671B 一致；激活 37.55B 比报告的 37B 略高，差别来自 router（0.11B）与 embedding 的计法（报告的 37B 是取整数字），正文统一用 37B 与 74 GFLOPs；
- 期望激活专家数 $$B = 32$$ 时 163.3、$$B = 128$$ 时 251.6，与第三章的表一致；
- all-to-all 每 token 每层 168 KiB（DeepSeek-V3）、32 KiB（Mixtral，top-2、$$d = 4096$$、dispatch 与 combine 都按 BF16）。

第六篇会在这个脚本上加 dtype 字节表与训练状态显存，第七篇加量化、投机解码与 LoRA。


## 九、小结

MoE 把 dense 模型里绑在一起的三个数拆开了：

- **参数量**决定显存：DeepSeek-V3 的 671B 在 FP8 下也是 671 GB，一台 8 卡 H100 放不下；
- **激活参数量**决定 FLOPs/token：37B 对应 74 GFLOPs，是 Llama-3-70B 的一半；
- **每步实际读取的参数量**决定 decode 带宽：随 batch 从 37B 趋近 671B，$$B = 32$$ 时已是 434B，因为期望激活专家数 $$E[1 - (1 - k/E)^B]$$ 在中等 batch 下就接近 $$E$$。

第三个数是"37B 的模型比 70B 难部署"的直接原因，也是 DeepSeek-V3 必须用大规模 EP 的原因：只有把专家分散到很多卡上，每卡的读取量才回到可控范围。EP 的代价是每层两次 all-to-all（每 token 每层 dispatch 56 KiB + combine 112 KiB）、每专家 GEMM 只有 $$Tk/E$$ 行（decode 时可能只有 1 行）、以及负载不均时最慢的卡决定全层时间。节点受限路由、aux-loss-free 均衡、冗余专家、共享专家、grouped GEMM，都是围绕这几项代价的应对。

本篇算出的数字汇总：

```text
                                 Mixtral 8x7B      DeepSeek-V3        Llama-3-70B（dense 对照）
专家配置                          8 × 14336, top-2  256 × 2048, top-8  —
                                                    + 1 共享
每专家参数                        176.16M           44.04M             —（FFN 704.6M）
总参数                            46.7B             671B               70.55B
激活参数 / token                  12.9B             37B                70.55B
FLOPs / token（≈ 2 × 激活）       ≈ 26 GFLOPs       74 GFLOPs          141 GFLOPs
权重字节                          BF16 93.4 GB      FP8 671 GB         BF16 141 GB
                                                    BF16 1342 GB
期望激活专家数  B=1 / 32 / 128    2 / 8.0 / 8.0     8 / 163 / 252      —
每步读取参数量  B=1               12.9B             37.6B              70.55B
                B=32              46.7B             434B               70.55B
                B=128             46.7B             660B               70.55B
EP all-to-all / token / 层        32 KiB（BF16）    168 KiB            —
                                                    （FP8 56 + BF16 112）
TP-8 all-reduce / token / 层      —                 25 KiB             28 KiB
grouped GEMM 每专家行数
    prefill T=4096                1024              128                4096（dense M）
    decode B=32                   8                 1                  32（dense M）
```

所有数字都是从超参数推导的理论值。它们回答的是数量级问题：MoE 在哪里省了、在哪里没省、代价转移到了哪里。

下一篇离开结构与算量，进入数值：同样的 GEMM 用 FP16、BF16、FP8 算，结果会差多少，为什么 DeepSeek-V3 的 FP8 训练要每 128 个元素就提升到 FP32 累加一次。

> **一个数用多少位表示，决定了它能算多快、放多少，也决定了它在哪里会悄悄算错。**


## 下一篇

[浮点格式、数值稳定性与混合精度](/floating-point-formats-and-mixed-precision.html)
