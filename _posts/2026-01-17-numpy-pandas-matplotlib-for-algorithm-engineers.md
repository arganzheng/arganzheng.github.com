---
layout: post
series: algorithm-tooling
title: "算法工程师的工具箱（01）：科学计算栈——NumPy 的形状直觉、Pandas 的错误分析、Matplotlib 的曲线"
subtitle: "The Scientific Python Stack: Shapes and Broadcasting in NumPy, Error Analysis in Pandas, Reading Curves in Matplotlib"
tags: [AI, LLM, PyTorch, Python]
catalog: true
updated: 2026-09-14
---

算法工作里的代码，十行里有八行在跟**形状**打交道：这个张量是 `[batch, seq, hidden]` 还是 `[seq, batch, hidden]`、softmax 沿哪一维、mask 怎么加到 score 上、多头 attention 的 reshape 与 transpose 是什么顺序。PyTorch 的 Tensor 语义与 NumPy 的 ndarray 一致，所以形状直觉先在 NumPy 上建立——它没有 GPU、没有自动求导、没有任何干扰，只有形状。本篇用 NumPy 把 L0 讲过的 attention 从公式写成代码并与 PyTorch 对数值；然后讲另外两件天天要做的事：用 Pandas 分析评测结果，用 Matplotlib 看训练曲线。

全篇的核心问题是：

> **看到一个 attention 的公式，能不能写出对应的 `einsum`？[^q0] 拿到评测结果，能不能按类别算出错误率、找出退化的题？[^q1] 看到一条 loss 曲线，知道该看哪里？[^q2]**

## 一、总览

### 1. 三件事

```text
NumPy       ndarray · 轴 · 广播三条规则 · reshape / transpose · einsum      → 手写一个 causal attention，与 PyTorch 对到 1e-7
Pandas      DataFrame · groupby · merge · query                            → 评测结果按类别聚合、找 baseline 对而新模型错的题
Matplotlib  折线 · 多曲线 · 对数坐标 · 阴影带                               → loss 曲线：对数 x 轴看早期，多 seed 画均值与标准差
```

### 2. 本文的章节安排

| 章 | 主题 | 内容 |
|---|---|---|
| 二 | ndarray 与轴 | 形状、dtype、索引；"沿哪个维度"是一个概念 |
| 三 | 广播 | 三条规则；合法、不合法、以及"能跑但错" |
| 四 | reshape、transpose 与 einsum | 多头 attention 的形状变换；把公式翻译成代码 |
| 五 | 手写 causal attention | 30 行 NumPy，与 `F.scaled_dot_product_attention` 对数值 |
| 六 | Pandas：评测的错误分析 | `groupby` 看各类别、`merge` + `query` 找退化的题、顺手算置信区间 |
| 七 | Matplotlib：看曲线 | 对数 x 轴、多 seed 阴影带、双对数 |
| 八 | 本文小结 | |
| 九 | 自测 | 五道题 |

配套脚本：[`01_numpy_pandas_matplotlib.py`](https://github.com/arganzheng/ai-learning-labs/blob/main/algorithm-tooling/01_numpy_pandas_matplotlib.py)，文中的数字都来自它。

## 二、ndarray 与轴

### 1. 形状与 dtype

NumPy 的核心对象是 `ndarray`：一块连续内存加一个**形状**（shape）和一个**元素类型**（dtype）。L0 第一篇的标量 / 向量 / 矩阵 / 张量在这里就是 0 / 1 / 2 / 多维的 ndarray：

```python
import numpy as np
x = np.zeros((32, 128, 4096), dtype=np.float32)   # 一批 32 句、每句 128 个 token、每个 4096 维
x.shape      # (32, 128, 4096)
x.ndim       # 3
x.dtype      # float32：每个元素 4 字节
x.nbytes     # 32 × 128 × 4096 × 4 = 67,108,864 字节 = 64 MiB
```

看到一个数组先看这三样。`nbytes` 是第三篇显存账的起点：同样的形状，`float32` 4 字节、`bfloat16` 2 字节、`int8` 1 字节。

### 2. 索引与切片

```python
x[0]            # 第 0 句：(128, 4096)
x[:, -1]        # 每句的最后一个 token：(32, 4096)
x[:, :64, :]    # 每句的前 64 个 token：(32, 64, 4096)
x[..., :10]     # 只看每个 token 的前 10 维：(32, 128, 10)   ... 表示"前面所有维"
```

一个维度被一个整数索引就**消失**，被一个切片索引就**保留**。`x[:, -1]` 比 `x[:, -1:]` 少一维——前者 `(32, 4096)`，后者 `(32, 1, 4096)`。这个差别在广播时（下一章）经常成为 bug 的来源。

### 3. 轴：所有"沿哪个维度"是一个概念

规约类操作（求和、平均、最大、softmax）都要指定沿哪个**轴**（axis）进行；那个轴在结果里消失：

```python
x.sum(axis=-1)    # 沿最后一维求和：(32, 128)——每个 token 的 4096 个数加起来
x.mean(axis=0)    # 沿 batch 平均：(128, 4096)
x.max(axis=1)     # 沿 seq 取最大：(32, 4096)
x.sum(axis=-1, keepdims=True)   # (32, 128, 1)——保留那一维为 1，方便下一步广播
```

`axis=-1` 是"最后一维"，是最常用的写法，因为按 PyTorch 的约定最后一维是特征维。把这一个概念对到模型里：

```text
softmax 沿词表维       logits [B, T, V]        axis=-1   每个位置的 V 个 logits 变成一个分布
LayerNorm 沿 hidden 维  x [B, T, d]            axis=-1   每个 token 自己归一化
loss 沿 token 维平均    per_token_loss [B, T]   全部平均  每 token 负对数似然的平均
attention 沿 key 维    score [B, h, T_q, T_k]  axis=-1   每个 query 对所有 key 的分数归一化
```

写错轴的 softmax 不报错——沿 batch 维做 softmax 得到的仍是一个形状正确的数组，只是数值全错。这是形状 bug 的第一种：**能跑、形状对、数值错**。

## 三、广播

### 1. 三条规则

形状不同的两个数组做逐元素运算时，NumPy 按**广播**（broadcasting）规则把小的那个"复制"成大的形状。只有三条：

1. **从最后一维往前对齐**（右对齐）；
2. 每一维要么**相等**，要么**其中一个是 1**（或那一维不存在）；
3. 是 1 的那一维被复制到另一个的长度。

```text
  (4, 3) + (3,)      →  右对齐  4,3 / -,3  → 3=3 ✓，缺失的当 1 ✓  →  (4, 3)   bias 加到每一行
  (4, 3) + (4, 1)    →  4=4 ✓，3 vs 1 ✓                            →  (4, 3)   每行乘 / 加自己的一个标量
  (4, 3) + (4,)      →  右对齐  4,3 / -,4  → 3≠4 ✗                 →  ValueError
```

脚本的输出：

```text
(4,3) + (3,)  bias 加到每一行             -> (4, 3)
(4,3) + (4,1) 每行一个标量                 -> (4, 3)
(4,3) + (4,)  行数对不上最后一维            -> ValueError: operands could not be broadcast together with shapes (4,3) (4,)
```

### 2. 模型里的广播

```text
X [B, T, d] + b [d]                          Linear 的偏置加到每个 token
X [B, T, d] * s [B, T, 1]                    RMSNorm：每个 token 乘自己的缩放因子（keepdims=True 留下的那个 1）
S [B, h, T, T] + mask [1, 1, T, T]           causal mask 加到每个 batch、每个 head 的 score 上
logits [B, T, V] - max [B, T, 1]             softmax 减最大值
```

最后一行是第二章 `keepdims=True` 的用处：`x.max(axis=-1, keepdims=True)` 得到 `[B, T, 1]`，才能广播回 `[B, T, V]`；不加 `keepdims` 得到 `[B, T]`，与 `[B, T, V]` 右对齐时 `V` 对 `T`，报错或（更糟）当 $$T = V$$ 时静默算错。

### 3. 能跑但错

广播最危险的地方是它**太宽容**：

```python
a = np.arange(3)          # (3,)
b = np.arange(3)[:, None] # (3, 1)
(a + b).shape             # (3, 3)！
```

想做两个长度 3 的向量逐元素相加，因为一个是 `(3,)` 一个是 `(3, 1)`，广播成了一张 $$3 \times 3$$ 的外积表：

```text
[[0 1 2]
 [1 2 3]
 [2 3 4]]
```

代码不报错，结果的形状看起来也"合理"，只有下游某个地方 loss 突然不对时才发现。防御的办法只有一个习惯：**关键步骤 `assert x.shape == (...)`**，尤其是 loss 前的 logits 与 labels。

## 四、reshape、transpose 与 einsum

### 1. 多头 attention 的形状变换

L0 第一篇说多头 attention 把 $$d$$ 拆成 $$h$$ 个头。代码里是一串 reshape 与 transpose：

```python
Q = X @ W_Q                                  # [B, T, d]
Q = Q.reshape(B, T, h, d // h)               # [B, T, h, d_h]   把最后一维拆成 h 组
Q = Q.transpose(0, 2, 1, 3)                  # [B, h, T, d_h]   把 head 维挪到前面，让每个头独立成一个 [T, d_h]
S = Q @ K.transpose(0, 1, 3, 2)              # [B, h, T, d_h] × [B, h, d_h, T] → [B, h, T, T]
```

两个操作的性质不同：

- **`reshape` 不移动数据**，只改变"怎么读这块内存"——`(B, T, d)` 与 `(B, T, h, d/h)` 是同一块内存的两种解释，零成本；
- **`transpose` 也不移动数据**，但改变了读取顺序，之后的内存**不再连续**。在 NumPy 里这透明；在 PyTorch 里 `transpose` 之后再 `view` 会报错，要先 `.contiguous()`（真的复制一份成连续的）或用 `reshape`（自动判断）。stride 与内存布局的细节属于 Infra 03 系列第二篇，这里知道"transpose 之后不连续"即可。

### 2. einsum：把公式翻译成代码

`einsum`（Einstein summation）用一个下标字符串描述矩阵乘法，是**把论文公式翻译成代码的最短路径**：

```python
S = np.einsum("bhqd,bhkd->bhqk", Q, K)       # 带 batch 与 head 的 Q K^T
```

读法：`Q` 的四维叫 `b h q d`，`K` 的四维叫 `b h k d`；箭头右边是输出的维度 `b h q k`。**只出现在左边、没出现在右边的字母被求和**（这里是 `d`——内积就是沿 $$d$$ 求和）；**两边都出现的字母被保留**；右边的顺序就是输出形状。上面那行等价于 `Q @ K.transpose(0, 1, 3, 2)`，但不用想 transpose 的参数顺序。

几个常见公式的 einsum：

```text
S = Q K^T             "btd,bsd->bts"        每个 query t 对每个 key s 的内积，沿 d 求和
O = P V               "bts,bsd->btd"        attention 权重 P 加权求和 V，沿 s 求和
逐元素乘再求和         "ij,ij->"             Frobenius 内积
矩阵乘法              "ik,kj->ij"           就是 A @ B
转置                  "ij->ji"
批量外积              "bi,bj->bij"
```

读论文里的张量公式时，先在脑子里写出 einsum 的下标，是检验自己是否真的看懂了形状的办法。

## 五、手写 causal attention

### 1. 30 行

把 L0 的公式 $$\text{Attention}(Q, K, V) = \text{softmax}(QK^T / \sqrt{d_k})\,V$$ 加上 causal mask（每个位置只能看自己和前面的）写出来：

```python
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)       # 减最大值：L0 第五篇的数值技巧；keepdims 让它能广播回去
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def causal_attention_np(Q, K, V):                  # Q, K, V: [B, T, d]
    B, T, d = Q.shape
    S = np.einsum("btd,bsd->bts", Q, K) / np.sqrt(d)   # [B, T, T]：query t 对 key s 的分数；除 √d 见 L0 第四篇
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)   # 上三角（不含对角线）为 True：s > t 是"未来"
    S = np.where(mask, -np.inf, S)                     # [T, T] 的 mask 广播到 [B, T, T]；−∞ 过 softmax 变 0
    P = softmax(S, axis=-1)                            # 沿 key 维归一化：每个 query 的权重和为 1
    return np.einsum("bts,bsd->btd", P, V)             # 加权求和 V
```

每一行都在用前面三章的东西：`keepdims` 与广播、`einsum`、`axis=-1`、mask 的广播。

### 2. 与 PyTorch 对数值

```python
out_np = causal_attention_np(Q, K, V)
out_pt = F.scaled_dot_product_attention(torch.tensor(Q), torch.tensor(K), torch.tensor(V), is_causal=True).numpy()
np.abs(out_np - out_pt).max()     # 2.65e-07
```

```text
形状: Q (2, 8, 16) -> S (2, 8, 8) -> 输出 (2, 8, 16)
最大绝对误差: 2.65e-07   (float32 下 1e-6 量级即为一致)
第 0 个 batch 的 attention 权重（每行和为 1，上三角为 0）：
[[1.   0.   0.   0.   0.   0.   0.   0.  ]
 [0.28 0.72 0.   0.   0.   0.   0.   0.  ]
 [0.92 0.05 0.03 0.   0.   0.   0.   0.  ]
 [0.33 0.08 0.5  0.09 0.   0.   0.   0.  ]
 [0.29 0.24 0.   0.3  0.18 0.   0.   0.  ]
 [0.22 0.15 0.12 0.11 0.05 0.35 0.   0.  ]
 [0.04 0.44 0.09 0.3  0.06 0.06 0.01 0.  ]
 [0.01 0.12 0.05 0.12 0.12 0.16 0.23 0.19]]
```

$$2.65 \times 10^{-7}$$ 是 float32 的舍入误差量级——两个实现一致。权重矩阵读起来：第 0 行只能看自己（1.0），第 $$t$$ 行只有前 $$t + 1$$ 个非零，每行和为 1。**"与参考实现对数值"是验证任何手写算子的标准方法**，L3 系列的梯度检查、Infra 系列的 kernel 验证用的都是这一招；标准是误差在浮点精度量级（float32 约 $$10^{-6}$$，float64 约 $$10^{-12}$$），不是"看起来差不多"。

这 30 行是 PyTorch 里 `F.scaled_dot_product_attention` 做的事的数学版；真实实现（FlashAttention）不会显式构造 $$[T, T]$$ 的 $$S$$——那是 Infra 05 系列的内容。

## 六、Pandas：评测的错误分析

### 1. 场景

跑完评测得到一个文件，每题一行：题号、类别、模型答案、标准答案。要回答的问题不是"总分多少"，而是：**哪类题好、哪类题差、相比 baseline 哪些题退化了**。这是表格操作，Pandas 的 `DataFrame` 就是为它设计的。

### 2. 三个操作

```python
import pandas as pd
new = pd.DataFrame(rows)                              # 每行 {"id", "category", "correct"}
new.groupby("category")["correct"].agg(["mean", "count"])          # 各类别正确率与题数
merged = new.merge(base, on=["id", "category"], suffixes=("", "_base"))   # 与 baseline 按题号对齐
merged.query("not correct and correct_base")                       # 退化的题：baseline 对、新模型错
```

- **`groupby`**：按某一列分组，对每组算统计量——"各类别正确率"；
- **`merge`**：两张表按键对齐——把新模型与 baseline 的结果放到同一行；
- **`query`**：用一个字符串表达式筛行——"退化的题"。

脚本用一份合成数据（530 题、四个类别）跑出来：

```text
               acc_new  count  acc_base  delta   ci95
category
algebra          0.770    200     0.695  0.075  0.058
combinatorics    0.525     80     0.475  0.050  0.109
geometry         0.627    150     0.580  0.047  0.077
number_theory    0.610    100     0.640 -0.030  0.096
总体: base 0.619 -> new 0.662
退化的题（base 对、new 错）: 28；改善的题: 51
退化最多的类别: {'algebra': 12, 'geometry': 7, 'combinatorics': 5, 'number_theory': 4}
```

### 3. 顺手算置信区间

最后一列 `ci95` 是 L0 第八篇的 $$1.96\sqrt{\hat p(1 - \hat p)/n}$$，一行向量化就算出来了：

```python
summary["ci95"] = 1.96 * np.sqrt(summary["acc_new"] * (1 - summary["acc_new"]) / summary["count"])
```

带上它再读这张表：总体提升 4.3 个点；`algebra` 的 +7.5 超过它的 ±5.8，可信；`combinatorics` 的 +5.0 在 ±10.9 里面，80 道题分辨不出来；`number_theory` 的 −3.0 也在 ±9.6 里，退化不一定是真的——但 `merge` + `query` 找出的那 4 道"baseline 对、新模型错"的题值得一道道看。**总分 + 各类别 + 置信区间 + 退化题列表**，这四样是 L5 评测系列里"能力分解与错误分析"的全部工具，代码就是上面那几行。

Polars 是 Pandas 的替代品，API 风格接近、在千万行以上快很多；两者选一个熟练即可。

## 七、Matplotlib：看曲线

### 1. 需要的很少

训练是否正常，第一眼看的是曲线：loss、梯度范数、学习率随步数的变化。需要的绘图能力只有折线、多条曲线对比、对数坐标、子图、阴影带。两个习惯值得早建立。

### 2. 对数 x 轴看早期

```python
ax.set_xscale("log")
```

训练的前 1% 步数决定了初始化与 warmup 是否对（L0 第五篇：第一步 loss 该是 $$\ln V$$；L3：warmup 期间的 loss 形状），但在线性 x 轴上这一段被压成左边一条竖线。对数 x 轴把它展开。脚本画的两张子图是同一组曲线的线性与对数版本，右图能看清前 100 步。

### 3. 多 seed 画均值与阴影带

```python
mean, std = curves.mean(0), curves.std(0)                          # curves: [n_seeds, n_steps]
ax.plot(steps, mean)
ax.fill_between(steps, mean - std, mean + std, alpha=0.3)
```

```text
曲线形状 (5, 1000)；step 10 / 100 / 1000 的均值 loss: 6.49 / 4.39 / 1.88，seed 间标准差 0.066
```

单条曲线的抖动分不清是"方法"还是"运气"；5 个 seed 的均值加 ±1 标准差的阴影带，两个方法的曲线是否分得开一眼可见。这是横切"实验方法论"的规则，工具就是 `fill_between` 一行。

### 4. 双对数

scaling law 的图横纵轴都是对数刻度（`ax.set_xscale("log"); ax.set_yscale("log")`），因为幂律在双对数下是直线（L0 第八篇）。看到这种图先读斜率。

## 八、本文小结

- **ndarray** = 内存 + 形状 + dtype；`nbytes` 是显存账的起点。整数索引消灭一维、切片保留一维。
- **轴**：所有"沿哪个维度"是一个概念，那一维在结果里消失，`keepdims=True` 留下一个 1 供广播。softmax 沿词表维、LayerNorm 沿 hidden 维、attention 沿 key 维——写错轴不报错。
- **广播三条规则**：右对齐、相等或为 1、为 1 的被复制。`(3,) + (3, 1)` 静默变成 `(3, 3)`——形状 bug 的第一种是"能跑但错"，防御是 `assert` 形状。
- **reshape** 不移动数据；**transpose** 也不移动但让内存不连续（PyTorch 里 `view` 前要 `contiguous`）。多头 attention 是 reshape 拆 head、transpose 把 head 挪到前面。
- **einsum** 用下标字符串描述张量乘法：只在左边的字母被求和、两边都有的被保留、右边的顺序是输出形状。读公式先写 einsum。
- 30 行 NumPy 的 causal attention 与 PyTorch 对到 $$2.65 \times 10^{-7}$$——"与参考实现对数值到浮点精度"是验证手写算子的标准方法。
- **Pandas** 三个操作：`groupby` 看各类别、`merge` 对齐 baseline、`query` 找退化的题；顺手一行算 `ci95`，读表要带着置信区间。
- **Matplotlib** 两个习惯：对数 x 轴看训练早期；多 seed 画均值与 `fill_between` 阴影带。

## 九、自测

1. `x.shape == (32, 128, 4096)`：`x[0, :, 0]`、`x[:, 0]`、`x[..., :1]` 各是什么形状？

   <details markdown="1"><summary>答案</summary>

   `(128,)`、`(32, 4096)`、`(32, 128, 1)`。

   </details>

2. `(32, 128, 4096) + (128, 1)` 能广播吗？结果是什么形状？`+ (128,)` 呢？

   <details markdown="1"><summary>答案</summary>

   能，右对齐 `4096` 对 `1`、`128` 对 `128`，结果 `(32, 128, 4096)`——每个 token 加自己的一个标量；`(128,)` 对 `4096` 不相等且不是 1，报错。

   </details>

3. 写出 `O = P V`（`P: [B, h, T, T]`，`V: [B, h, T, d_h]`）的 einsum，输出形状是什么？

   <details markdown="1"><summary>答案</summary>

   `"bhts,bhsd->bhtd"`，`[B, h, T, d_h]`。

   </details>

4. 沿错误的轴做 softmax 为什么不报错？怎么防？

   <details markdown="1"><summary>答案</summary>

   结果形状与输入相同，NumPy 无法知道你的意图；关键处 `assert` 形状、对一个小例子手算验证。

   </details>

5. 评测结果里新模型比 baseline 总分高 2 个点，但你想知道"是不是同样的题"：用哪两个 Pandas 操作？

   <details markdown="1"><summary>答案</summary>

   `merge`（按题号对齐）+ `query`（筛 "correct and not correct_base" 与反过来）。

   </details>

下一篇把形状直觉搬到 PyTorch 上：五个核心对象、二十行训练循环、Autograd 要知道的三件事，训一个字符级小 Transformer。

[^q0]: 能。把公式里每个张量的下标写出来：只出现在左边的字母被求和、两边都有的被保留、右边的顺序就是输出形状——$$\text{softmax}(QK^T)V$$ 是 `"bhtd,bhsd->bhts"` 与 `"bhts,bhsd->bhtd"` 两行；30 行 NumPy 与 PyTorch 对到 $$10^{-7}$$。详见[第四章](#四reshapetranspose-与-einsum)、[第五章](#五手写-causal-attention)。
[^q1]: 能。`groupby` 按类别算准确率并带上 `ci95`，`merge` 按题号对齐 baseline，`query` 筛出「baseline 对、新模型错」的题。详见[第六章](#六pandas评测的错误分析)。
[^q2]: x 轴用对数看训练早期；多 seed 画均值与阴影带，一条曲线高出阴影带才算差别；看 train / val 是否分叉判断过拟合。详见[第七章](#七matplotlib看曲线)。
