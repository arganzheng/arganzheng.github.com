---
layout: post
title: "大规模训练工程（01）：训练任务的状态解剖——显存账与 MFU"
subtitle: "Anatomy of Training State: Memory Accounting and MFU"
tags: [Megatron, DeepSpeed, torchtitan, Distributed Training, MFU, AI, AI-Infra]
catalog: true
---

> 本文是[《大规模训练工程：从并行策略到容错恢复》](/large-scale-training-from-parallelism-to-fault-tolerance.html)系列的第 1 篇（共八篇）。上一篇：[大规模训练工程：从并行策略到容错恢复（总纲）](/large-scale-training-from-parallelism-to-fault-tolerance.html)；下一篇：[并行策略全景：每种并行切的是哪种状态](/parallelism-strategies-which-state-to-shard.html)。

一张 H100 有 80 GB 显存、标称 989 TFLOPS 的 bf16 算力。一个 70B 参数的模型，用 bf16 混合精度加 Adam 训练，参数、梯度和优化器状态加在一起是 1.13 TB——是那张卡的 14 倍。序列长 8192 时，它每一层的激活值还要 2.3 GB，80 层就是 180 GB。每个 token 的前向加反向要 4.5×10¹¹ 次浮点运算，一条 8192 token 的序列在一张卡上就算满了也要 3.7 秒。

这几个数字是本系列后面七篇的全部起点。并行策略要解决的是"1.13 TB 放到哪些卡上"；micro-batch 与激活重计算要解决的是"180 GB 的激活怎么塞进剩下的显存"；MFU 调优要解决的是"3.7 秒的理论下限实际跑成了多少"；checkpoint 要解决的是"这 1.13 TB 里哪些必须落盘、多久落一次"；容错要解决的是"一部分卡消失后这些字节怎么重建"。不先把这些数字算出来，后面的每一个决定都只能靠试。

算这些数字不需要 GPU，只需要一套一致的记账方法：训练任务的状态只有四样——**参数、梯度、优化器状态、激活**——每一样有明确的字节数公式、明确的生命周期；算力只有一个主项——每个参数每个 token 6 FLOP——加一个随序列长度增长的注意力项。把它们与硬件的两个数字（显存容量、峰值 FLOPS）放在一起，就得到显存够不够、step 时间下限是多少、MFU 是多少这三个答案。

这一篇不讨论任何并行策略，也不进入任何框架的进程组代码。它只做一件事：把这套记账方法建立起来，给出符号、公式、三档模型（Llama 3 的 8B、70B、405B）在 $$s = 8192$$ 下的完整数字，并与 Megatron Core 0.18.0 里做同一件事的两段源码——`megatron/training/theoretical_memory_usage.py` 与 `megatron/training/training.py` 的 `num_floating_point_operations()`——逐项对上。

本篇的核心问题：

> **一个 70B 参数、序列长 8192 的模型，在一张 80 GB 的 H100 上，参数与优化器状态要多少字节？一层的激活要多少？每个 token 要多少 FLOP？把它们与 80 GB 和 989 TFLOPS 放在一起，就知道后面每一篇要解决的是什么。**

文中所有硬件峰值都是**标称值**（H100 SXM bf16 dense 989 TFLOPS、A100 312 TFLOPS），所有字节数都用十进制单位（1 GB = 10⁹ 字节）除非特别注明 GiB；涉及论文的数字在首次出现处注明出处。


## 一、总览

### 1. 四种状态与一张卡

训练引擎围绕状态组织。一个训练任务在任一时刻持有的全部数据可以归成四类：

```text
                  何时存在              字节数由什么决定                  谁切它（下一篇）
─────────────    ─────────────────    ──────────────────────────────   ────────────────────
参数 θ            常驻                  N × 每参数字节                    TP / PP / ZeRO-3
梯度 ∇θ           反向产生，优化器后清零   N × 每参数字节                    TP / PP / ZeRO-2
优化器状态        常驻                  N × 每参数字节（Adam：12）          TP / PP / ZeRO-1
激活 A            前向产生，反向消费      s · b · h · L × 常数               TP/SP、CP、PP、重计算
```

前三样是**参数量 $$N$$ 的线性函数**，系数由精度与优化器决定，与 batch 大小、序列长度无关；第四样是**每次前向喂进去多少 token 的线性函数**，与参数量无关（只与 $$h$$、$$L$$ 有关）。这两条线在一张卡的显存里相加，超过容量就放不下。整个第二、三、四章就是把这两条线的系数推出来。

### 2. 两本账：字节与 FLOP

训练的性能也只有两本账。**字节的账**决定一个配置能不能跑：四种状态加上显存之外的开销要小于每卡容量。**FLOP 的账**决定它最快能跑多快：每 token 的 FLOP 数乘以每 step 的 token 数，除以卡数乘以峰值算力，就是 step 时间的下限；观测到的 step 时间与这个下限之比就是 MFU。

```text
字节的账     16N（或 18N）+ 激活(s, b, h, L) + 非张量开销   ≤  每卡显存
FLOP 的账    step 时间  ≥  tokens/step × FLOP/token ÷ (卡数 × 峰值 FLOPS)
             MFU  =  下限 ÷ 实测 step 时间
```

两本账的公式都很短，容易算错的是**系数**：每参数到底 16 字节还是 18 字节、激活里那个 34 从哪来、6N 的 N 算不算 embedding、注意力的 $$s^2$$ 项要不要乘 1/2。本篇的大部分篇幅花在把这些系数的来源讲清楚，并与 Megatron 源码里的同一个系数对上。

### 3. 记账符号

全系列使用同一套符号，后续各篇会在总览里复述用到的部分：

```text
N                参数量（个数），不是字节数。bf16 参数是 2N 字节，16 字节/参数的常驻状态是 16N 字节
s  b  h  a  L    序列长、micro-batch 大小（每卡每次前向的序列数）、隐藏维、注意力头数、层数
f  V             FFN 中间维、词表大小
N_d N_t N_p      数据 / 张量 / 流水线并行度（本篇全部为 1）
N_c N_e          上下文 / 专家并行度（本篇全部为 1）
m                每个 DP 副本每 step 的 micro-batch 数（梯度累积步数）
B                global batch，以 token 计：B = s · b · m · N_d
δ                一次 checkpoint 的开销时间（第五篇）
M                集群的平均故障间隔 MTBF（第五、六篇）
```

三档模型的尺寸（Llama 3 论文 Table 3；$$V$$ 取 128256，含特殊 token）：

```text
              L      h       a    kv heads    f        V         N
Llama 3 8B     32    4096    32     8       14336    128256      8.03 B
Llama 3 70B    80    8192    64     8       28672    128256     70.55 B
Llama 3 405B  126   16384   128     8       53248    128256    405.85 B
```

$$N$$ 由第八章的 `ledger/model.py` 按每层 $$2h^2 + 2h\cdot h_{kv} + 3hf + 2h$$、加两份 $$Vh$$ 的 embedding（输入与输出不共享）算出，与官方公布的参数量一致到小数点后一位。

### 4. 本文的章节安排

```text
第二章  四种状态的生命周期     一个 step 的时间线；常驻与瞬态；峰值在哪一刻；三框架里每种状态的存放
第三章  混合精度的字节账       为什么要 fp32 主参数；16 字节的推导；18 字节的来源；Megatron 的 18 与 6 + 12/d；三档模型表
第四章  激活的字节账           sbh(34 + 5as/h) 逐项来源；FlashAttention 去掉 5as/h 的条件；三档模型表；Megatron 的 18 + 4f/h 与 10 + 24/t
第五章  显存之外的开销         CUDA context、库 workspace、NCCL buffer、caching allocator 的 reserved 与碎片；80 GB 的预算表
第六章  算力账                 6N 的来源；注意力的 s² 项与因果 mask；num_floating_point_operations() 对照；三档模型 FLOP/token 与 step 下限
第七章  MFU 与 HFU             PaLM 的定义；重计算为什么抬高 HFU 不抬高 MFU；同一例子算两遍；参考水平；Megatron 的 TFLOP/s/GPU 日志
第八章  小结                   要点、符号与公式速查、源码位置、train-ledger 的第一批文件
```


## 二、四种状态与它们在一个 step 内的生命周期

### 1. 每种状态是什么

**参数**是模型本身。混合精度下它有两份：一份低精度（bf16）用来做前向和反向的 GEMM，一份 fp32 的"主参数"（master weights）由优化器持有、只在优化器步骤被读写。两份都是常驻的——bf16 那份每层前向都要用，fp32 那份每步都要更新。本篇在字节账里把 fp32 主参数归入"优化器状态"一类，因为它的生命周期与 Adam 的两个矩完全相同，而且在 ZeRO-1 / Megatron 分布式优化器里它们是一起被分片的。

**梯度**是反向传播的产物。它在反向过程中逐层产生（从最后一层往前），在优化器步骤被消费，随后清零。所以梯度的显存在反向结束、优化器开始前达到峰值——那一刻全部 $$N$$ 个梯度同时存在。它的 dtype 有两种选择，是 16 字节与 18 字节的分水岭（第三章）。

**优化器状态**是优化器为每个参数维护的历史信息。Adam 需要一阶矩 $$m$$ 和二阶矩 $$v$$，各一个 fp32 数——这是 8 字节；加上 fp32 主参数 4 字节，共 12 字节，占混合精度训练常驻状态的 3/4。它们只在优化器步骤被读写，其余时间静静地占着显存。SGD with momentum 只要 4 字节，Adafactor 之类的分解式优化器接近 0，但大模型预训练几乎无一例外用 AdamW，所以本系列默认 12 字节。

**激活**是前向传播中为反向保留的中间结果：每个线性层的输入（算权重梯度要用）、每个非线性的输入（算它的导数要用）、注意力的 softmax 输出、dropout 的 mask……它们在前向逐层产生、在反向逐层释放，生命周期恰好与梯度**互补**：前向结束时激活最多、梯度为零；反向结束时激活为零、梯度最多。

### 2. 一个 step 的时间线

把四种状态在一个 step 内的占用画出来（单卡、无并行、$$m = 1$$）：

```text
 显存
  ▲
  │            ┌──────── 激活峰值（前向末）
  │           ╱│╲
  │          ╱ │ ╲        ┌──── 梯度峰值（反向末）
  │         ╱  │  ╲       │
  │  ┌─────╱───┼───╲──────┼──────┐   梯度 2N（bf16）或 4N（fp32）：反向中逐层填满，优化器后清零
  │  │    ╱    │    ╲     │      │
  │  ├───╱─────┼─────╲────┼──────┤
  │  │  ╱      │      ╲   │      │   激活：前向逐层堆高，反向逐层释放
  │  ├─╱───────┼───────╲──┼──────┤
  │  │         │          │      │   优化器状态 12N：常驻，只在 optimizer.step() 被读写
  │  ├─────────┼──────────┼──────┤
  │  │         │          │      │   bf16 参数 2N：常驻，每层前向 / 反向读
  │  └─────────┴──────────┴──────┘
  └────────────────────────────────────▶ 时间
     ├─ 前向 ─┤├── 反向 ──┤├ step ┤
```

三点值得注意。第一，**峰值不在同一时刻**：激活的峰值在前向结束处，梯度的峰值在反向结束处，两者不同时达到最大。所以"参数 + 梯度 + 优化器状态 + 激活"简单相加是一个**保守的**上界；精确的峰值是 $$\max(\text{前向末}, \text{反向末})$$，取决于激活总量与梯度总量谁大。对大模型、小 micro-batch，激活往往比梯度小（70B、$$b = 1$$、有 FlashAttention 时 180 GB 激活对 141 GB 梯度——差不多），所以上界通常不算太保守。

第二，**优化器步骤自己有临时显存**：Adam 更新时要读 $$m$$、$$v$$、梯度、主参数，写回 $$m$$、$$v$$、主参数，再把 fp32 主参数 cast 成 bf16 覆盖计算参数。融合的 Adam kernel（`apex` / `torch._fused_adamw_`）几乎不需要额外空间；非融合的 foreach 实现可能为一组参数分配一份临时值。这是 Megatron 的 `compute_activation_memory_without_sp()` 最后乘 1.05 的那 5% 的一部分（第四章）。

第三，**梯度累积不改变峰值**：$$m > 1$$ 时前 $$m - 1$$ 个 micro-batch 的反向把梯度累加进同一块 buffer，激活在每个 micro-batch 的反向后释放，所以每个 micro-batch 的峰值形状与上图相同，只是梯度那一层从第二个 micro-batch 起一直是满的。这就是为什么**激活的账按一个 micro-batch 算**（本篇的 $$b$$ 都是 micro-batch 大小），而 global batch 只影响 step 时间不影响显存——直到流水线并行让多个 micro-batch 同时在途（第二篇）。

### 3. 三框架里每种状态存在哪

这四种状态在三个框架里都存在，但"两份参数"和"梯度的 dtype"的安排不同。这是本系列"框架线"的第一张对照表，只讲结果，实现细节留到第三篇：

```text
                   bf16 计算参数                 梯度                            fp32 主参数 + Adam 矩          每参数字节（常驻）
────────────────   ───────────────────────────   ─────────────────────────────   ────────────────────────────   ─────────────────
Megatron-LM        模型持有，常驻                 默认 fp32 main_grad（连续 buffer，  优化器持有；分布式优化器       18；
                                                 bf16 grad 逐层累加进去后释放）     打开时按 DP 切                 6 + 12/N_d
DeepSpeed ZeRO     模型持有，常驻（Stage 3 下      bf16；Stage 2/3 下 reduce-scatter  优化器持有 fp32 分区           16；
（bf16）            分片，用到时 all-gather）       后只留本卡分片                                                  Stage 1: 4 + 12/N_d
torchtitan         没有常驻的 bf16 副本：FSDP2      reduce-scatter 用 fp32，结果是    分片参数本身就是 fp32，        16 / N_d 常驻，
（FSDP2 默认策略）   前向前 all-gather 出当前层的     fp32 分片梯度                   不另存主参数；Adam 矩 fp32      + 当前层的 bf16 副本
                   bf16 副本，用完即释放
```

三者的每参数字节在 16–18 之间，差别只在**梯度用 bf16 还是 fp32**，以及**fp32 主参数是"另一份"还是"参数本身"**。torchtitan 的做法值得单独说一句：FSDP2 的分片参数就是 fp32 的，`MixedPrecisionPolicy(param_dtype=bfloat16)` 让 all-gather 出来的完整参数是 bf16 临时副本，前向后释放；所以它没有"两份常驻参数"，bf16 那 2 字节只在当前层存在。三种安排的常驻字节数不同，但**全部落在 16N 到 18N 的区间**（再除以各自的分片度），这就是下一章要推导的数。


## 三、混合精度的字节账：16 与 18

### 1. 为什么需要 fp32 主参数

bf16 有 8 位指数、7 位尾数，能表示的相对精度约 $$2^{-8} \approx 0.4\%$$。一个典型的参数更新量是学习率乘梯度，量级在 $$10^{-4}$$ 到 $$10^{-6}$$ 倍参数值——远小于 bf16 能分辨的 0.4%。如果直接在 bf16 参数上做 $$\theta \leftarrow \theta - \eta g$$，绝大多数更新会被舍入成零，训练停滞。所以优化器必须在 fp32 上累加更新，再把结果 cast 成 bf16 给下一步的前向用。这就是"主参数"：它不是冗余，是 bf16 训练能收敛的前提。fp32 训练不需要它（参数本身就是 fp32），但 GEMM 慢一倍以上、显存也多一倍，大模型预训练已经不这样做。

同样的道理适用于 Adam 的两个矩：$$v \leftarrow \beta_2 v + (1 - \beta_2) g^2$$ 每步只把 $$v$$ 改动千分之一（$$\beta_2 = 0.999$$），在 bf16 的 0.4% 分辨率下这个增量大部分会被舍掉。问题不在表示范围（bf16 与 fp32 的指数位一样多），而在**累加精度**，所以两个矩也必须 fp32。

### 2. 16 字节的推导

于是 bf16 混合精度 + Adam 下，每个参数的常驻字节：

```text
bf16 参数（计算用）          2
bf16 梯度                   2
fp32 主参数                 4
fp32 Adam 一阶矩 m           4
fp32 Adam 二阶矩 v           4
──────────────────────────────
                            16 字节 / 参数
```

分成三类就是本篇 `StateBytes` 的三个字段：`params_bytes = 2N`、`grads_bytes = 2N`、`optim_bytes = 12N`。优化器一类占 75%，这是 ZeRO-1 只切优化器状态就能省掉 3/4 显存的原因（第二篇）。

这个 16 与 ZeRO 论文（Rajbhandari et al. 2020）的 $$K = 12$$ 是同一件事：论文把 fp16 参数与梯度记为 $$2N + 2N$$，把 fp32 主参数与两个矩合记为 $$KN = 12N$$，总计 $$(2 + 2 + K)N = 16N$$。

### 3. 18 字节：fp32 梯度

如果梯度不是 bf16 而是 fp32，每参数变成 $$2 + 4 + 4 + 4 + 4 = 18$$ 字节。为什么会有人多花这 2 字节？因为**梯度要累加**：$$m$$ 个 micro-batch 的梯度加在一起，再在 $$N_d$$ 个 DP 副本间求和。每一次加法都在 bf16 上舍入一次，累加 $$m \times N_d$$ 次后误差不可忽略——尤其是 Megatron 的 DP all-reduce 直接在梯度 buffer 上做，归约的中间结果也是 bf16 的。Megatron 因此在 bf16 训练下**默认**用 fp32 累加与归约梯度（`megatron/training/arguments.py` 的 `validate_args()` 在 `args.bf16` 且未指定 `--grad-reduce-in-bf16` 时把 `accumulate_allreduce_grads_in_fp32` 置为 True），bf16 的 `param.grad` 算出后立刻加进 fp32 的 `param.main_grad`（`megatron/core/distributed/distributed_data_parallel.py` 里的反向 hook 做 `param.main_grad.add_(param.grad.data)`），`param.grad` 随即释放。注意这时**没有一份持久的 bf16 梯度**——它只在每层反向的瞬间存在——所以 18 字节的构成是 2 + 4 + 12，而不是 2 + 2 + 4 + 12 = 20。

总纲里"若梯度以 fp32 累加再多 2 字节到 18 字节"说的就是这件事：从 16 到 18 是 bf16 梯度换成 fp32 梯度，多出 2 字节；这份 fp32 梯度同时也是 DP 归约的对象，所以 DP 通信量也随之翻倍（第二篇 $$2N$$ 的通信按 dtype 折算字节时要留意）。

fp16 训练与 bf16 一样是 16 字节，只是多一个 loss scale 标量和溢出检查；fp32 训练是 $$4 + 4 + 8 = 16$$（没有主参数），与 bf16 相同——但 GEMM 的速度差一倍以上。

### 4. Megatron 的 18 与 6 + 12/d

Megatron 把这笔账写在 `megatron/training/theoretical_memory_usage.py` 的 `compute_weight_and_optimizer_memory()` 里。函数先按结构公式算参数量（注意力 $$2h^2(1 + g/a)$$、MLP $$2hf \times 3/2$$（SwiGLU）、norm、embedding），再按 TP/PP/EP 算出"最重的那个模型分片"上的参数数，最后乘每参数字节数：

```python
def num_bytes_per_parameter(data_parallel_size):
    # This estimator assumes bf16 training: bf16 model params, fp32 main gradients,
    # fp32 main params, and fp32 Adam states.
    return 18 if not args.use_distributed_optimizer else 6 + (12 / data_parallel_size)
```

18 就是上一节的 2 + 4 + 12；`6 + 12/d` 是分布式优化器（ZeRO-1）下的每参数字节：bf16 参数 2 与 fp32 梯度 4 不切，fp32 主参数与两个矩 12 按 DP 度 $$d$$ 切。同一仓库 `docs/user-guide/features/dist_optimizer.md` 里的表格列出了三种精度组合：fp16 参数 + fp16 梯度 20 / $$4 + 16/d$$（fp16 下 Megatron 把主参数与梯度都放进优化器分片），bf16 参数 + fp32 梯度 18 / $$6 + 12/d$$，fp32 参数 + fp32 梯度 16 / $$8 + 8/d$$。这三行与本章的推导完全一致，是检验记账方法的一个好锚点。

`report_theoretical_memory()` 在训练日志里把这个数与激活估算一起打印（`training.py` 的 `training_log()` 在第一次报告显存时调用它，随后 `report_memory()` 打印 `torch.cuda.memory_allocated()` 等实测值），理论与实测并排，是第三篇"对账"要用的工具。

### 5. 三档模型的常驻状态

把 $$N$$ 代进去，bf16 + Adam（16 字节）与 Megatron 默认（18 字节）两列：

```text
                 N           bf16 参数    bf16 梯度    fp32 主参数+Adam    合计 16N     合计 18N（fp32 梯度）
Llama 3 8B        8.03 B      16.1 GB      16.1 GB       96.4 GB          128.5 GB      144.5 GB
Llama 3 70B      70.55 B     141.1 GB     141.1 GB      846.6 GB            1.13 TB       1.27 TB
Llama 3 405B    405.85 B     811.7 GB     811.7 GB        4.87 TB           6.49 TB       7.31 TB
```

三个结论。第一，**8B 在一张 80 GB 的卡上也放不下**——128 GB 的常驻状态，还没算激活；"单卡训 8B"只能靠 ZeRO/FSDP 把它切到至少两张卡，或者把优化器状态 offload 到 CPU。第二，70B 的 1.13 TB 是 14 张 H100 的显存总和，即便切得毫无浪费也至少要 15 张卡起步。第三，405B 的 6.49 TB 就是第五篇要写的 checkpoint 的大小量级——checkpoint 里存参数与优化器状态（bf16 参数其实可以从 fp32 主参数恢复，所以严格说是 $$12N$$ 到 $$14N$$），一次要落盘 5–6 TB。

这张表是第二篇的起点：所有并行策略都在回答"这几 TB 放到哪里"。


## 四、激活的字节账：sbh(34 + 5as/h)

### 1. 公式与各项来源

Korthikanti et al. 2022（*Reducing Activation Recomputation in Large Transformer Models*，Megatron 的激活重计算论文）给出一个标准 Transformer 层为反向保留的激活字节数：

$$
A_{\text{layer}} = s\,b\,h\left(34 + 5\,\frac{a\,s}{h}\right)
$$

假设是：激活为 fp16/bf16（2 字节，mask 1 字节），MLP 中间维 $$f = 4h$$，MHA（$$h_{kv} = h$$），有 dropout，不做任何重计算。34 与 5 是逐个张量数出来的（论文 4.1 节；下面的数是每 token、每层的字节，乘回 $$sb$$ 即得总量）：

```text
注意力块                                                MLP 块
  Q/K/V 投影的输入                     2h                  第一个线性层的输入           2h
  QKᵀ 的输入：Q 与 K                   4h                  GELU 的输入（f = 4h 宽）      8h
  softmax 输出          2 a s  ┐                          第二个线性层的输入（4h 宽）    8h
  softmax dropout mask  1 a s  ├ 5 a s                    dropout mask                 1h
  dropout 输出（PV 的输入）2 a s ┘                          小计                        19h
  PV 的输入：V                         2h
  输出投影的输入                       2h                 两个 LayerNorm
  输出 dropout mask                    1h                  各保留一份输入 2h            4h
  小计                          11h + 5as
──────────────────────────────────────────────────────────────────────────────────────
  每层每 token   11h + 19h + 4h = 34h，外加 5as        →  每层  sbh(34 + 5as/h)
```

34 这个系数里，**24 是可以被张量并行切开的**（Q、K、V、PV 输出、GELU 前后的 MLP 中间态——按头或按列分布在 TP 卡上），**10 是切不开的**（两个 LayerNorm 的输入 4h、注意力与 MLP 的输入各 2h、两个 dropout mask 各 1h——每张 TP 卡上都是完整的），这是第二篇讲序列并行时 "10 + 24/t" 的来源。

$$5as$$ 这一项与其他项性质不同：它随 $$s^2$$ 增长（乘回 $$sb$$ 后是 $$5as^2b$$），来自 softmax 输出、dropout mask 和 dropout 输出这三个 $$[b, a, s, s]$$ 的张量——注意力分数矩阵。当 $$s = 8192$$、$$a = 64$$、$$h = 8192$$ 时 $$5as/h = 320$$，是 34 的十倍：**不用 FlashAttention 的话，一层激活 90% 以上是分数矩阵**。

### 2. FlashAttention 之后

FlashAttention（Dao et al. 2022）分块计算注意力，从不把 $$[s, s]$$ 的分数矩阵写进显存；反向时它重算分数矩阵，只需前向保留每行的 softmax 归一化常数（logsumexp，$$a \cdot s \cdot b$$ 个 fp32，相对 $$sbh$$ 可以忽略）。于是 $$5as/h$$ 一项**整个消失**，每层激活退化为 $$34sbh$$，与 $$s$$ 呈线性关系。注意这个"消失"有三个条件：

- 注意力用的是 FlashAttention 或同类融合 kernel（PyTorch 的 `scaled_dot_product_attention` 的 flash / memory-efficient 后端、Transformer Engine 的 fused attention）。用朴素的 `softmax(QKᵀ)V` 实现，这一项就回来了。
- 注意力 dropout 为零或由 kernel 内部用随机数种子重放，不物化 mask。大模型预训练普遍不用注意力 dropout。
- 代价是反向多算一次 $$QK^T$$（每层每 token 约 $$2sh$$ FLOP，因果 mask 下减半），这与第七章讨论的"选择性重计算"在算力账上是同一项——Korthikanti 论文的"选择性重计算"正是只重算这部分，FlashAttention 把它做进了 kernel。

本系列后续的所有激活估算都默认 FlashAttention，即每层 $$34sbh$$；只在讨论"为什么长序列训练必须有 CP"时把 $$s^2$$ 的**计算量**（而不是显存）拿回来。

### 3. 三档模型的激活

$$s = 8192$$、$$b = 1$$、bf16、无重计算：

```text
                 h        a     34sbh 每层     5as²b 每层（无 Flash）   全部层 34sbh·L    + logits（bf16 + fp32）
Llama 3 8B       4096     32      1.14 GB          10.7 GB               36.5 GB             42.8 GB
Llama 3 70B      8192     64      2.28 GB          21.5 GB              182.5 GB            188.8 GB
Llama 3 405B    16384    128      4.56 GB          42.9 GB              574.9 GB            581.3 GB
```

最后一列加上了 logits：输出层的 $$[s, V]$$ 张量在 bf16 是 $$2sV$$，交叉熵通常再在 fp32 上留一份 $$4sV$$，$$V = 128256$$ 时一条序列是 6.3 GB——它不随 $$L$$ 增长，但比一层的激活还大，是流水线并行最后一个 stage 显存偏高的原因之一，也是各框架都有"融合交叉熵 / 分块 logits"的原因。

把这张表与第三章的表放在一起看：**70B 的激活（$$b = 1$$，182 GB）与它的 bf16 梯度（141 GB）是同一量级**，而 8B 的激活（36 GB）已经是它 bf16 参数（16 GB）的两倍。激活不是小数目；它随 micro-batch 线性增长，$$b = 4$$ 时 70B 就是 730 GB。这就是为什么第二篇的 TP/SP/CP 要切它、第四篇要用重计算换它。

### 4. Megatron 源码：18 + 4f/h 与 10 + 24/t

`theoretical_memory_usage.py` 里有两个激活估算函数，对应论文的两种情形。

`compute_activation_memory()` 用于**序列并行 + 选择性重计算**（`report_theoretical_memory()` 在 `args.sequence_parallel and args.recompute_granularity == 'selective'` 时选它）。它的每层公式是：

```python
activation_memory = (args.seq_length * args.micro_batch_size * args.hidden_size) * (
    18 + (4 * (args.ffn_hidden_size / args.hidden_size))
)
```

$$f = 4h$$ 时 $$18 + 4 \times 4 = 34$$，正是论文的系数；Megatron 把它写成 $$18 + 4f/h$$ 以适应 SwiGLU 等 $$f \ne 4h$$ 的结构（Llama 3 70B 的 $$f = 3.5h$$，系数就是 32）。没有 $$5as/h$$ 项——因为函数假设选择性重计算，分数矩阵不保留。随后乘 `num_layers`，加 embedding 层的输入与 dropout（乘在途 micro-batch 数 `pipeline_model_parallel_size`），interleaved 调度再乘一个 $$1 + \frac{p-1}{pv}$$ 的惩罚因子，PP = 1 时加 logits 的 $$4sbh(1 + V/h)$$，最后整体除以 `tensor_model_parallel_size`——因为 SP 打开时整层激活都被 TP 均分（第二篇）。

`compute_activation_memory_without_sp()` 用于**没有 SP**（或非选择性重计算）的情形，每层公式是：

```python
per_layer_memory = args.seq_length * args.micro_batch_size * args.hidden_size * (10 + (24 / args.tensor_model_parallel_size))
```

这就是上一节说的"34 = 10 + 24"：24 被 TP 切开，10 不被切。这一支同样不含 $$5as/h$$，末尾额外乘了 1.05 作为优化器临时 buffer 与杂项的估计（`overhead_factor = 1.05`）。

两支都**假设分数矩阵不落地**——Megatron 认为在它的默认配置下这是常态。如果读者用朴素注意力训练，Megatron 的理论值会显著低估。这是拿 `report_theoretical_memory()` 的输出与 `torch.cuda.max_memory_allocated()` 对账时第一个要核对的假设。

### 5. 重计算的位置

激活是四种状态里唯一**可以用算力换回来**的：不保留，反向时重新算一遍前向。全量重计算把每层激活降到只留层输入（$$2sbh$$），代价是多一次前向、约 33% 的额外 FLOP；选择性重计算只重算注意力分数部分（FlashAttention 已内建）；按层重计算只对若干层做全量重计算（Megatron 的 `recompute_granularity` / `recompute_method` / `recompute_num_layers`，见 `megatron/core/transformer/transformer_config.py` 的 `TransformerConfig`）。它对显存账与算力账的双重影响是第四篇的主题，本篇只需记住：**重计算改变的是激活那一层，也改变第七章 MFU 与 HFU 的差**。


## 五、显存之外：为什么 80 GB 只有 70 多 GB 可用

前两章的字节数都是**张量**。但 `nvidia-smi` 看到的显存占用总是比张量的总和大几 GB 到十几 GB，OOM 时 PyTorch 的报错里 "reserved" 也总比 "allocated" 大。这些差额来自四类东西，它们不出现在任何公式里，但每一次配置都要为它们预留。

### 1. CUDA context 与库的 workspace

每个进程在 GPU 上第一次执行 CUDA 操作时创建 context，它包含加载的 kernel 代码、常量、本地内存栈等。一个只 import 了 PyTorch 的进程约几百 MB；加载 cuBLAS、cuDNN、NCCL、Transformer Engine、FlashAttention 之后，全部 kernel 常驻可以到 1 GB 量级（CUDA 12.2 起 `CUDA_MODULE_LOADING=LAZY` 成为默认，kernel 只在首次调用时加载，减轻了这一项）。这部分不经过 PyTorch 的分配器，在 `torch.cuda.memory_reserved()` 里**看不到**，只能从 `torch.cuda.mem_get_info()`（直接查询驱动的空闲/总量）与 `memory_reserved()` 的差里推出来。

库还要 workspace。cuBLAS 的每个 handle 在每个 stream 上一块 workspace，PyTorch 2.13.0 对 Hopper 与 Blackwell 默认 32 MiB（`aten/src/ATen/cuda/CublasHandlePool.cpp` 的 `getChosenWorkspaceSize()`，由 `CUBLAS_WORKSPACE_CONFIG` 覆盖），从 caching allocator 里分配，所以计入 reserved。cuDNN、Transformer Engine 的 FP8 GEMM 也各有 workspace。这些通常合计几百 MB。

### 2. NCCL 的 buffer

每个 NCCL communicator 为每个 channel 分配收发 buffer（`NCCL_BUFFSIZE` 默认 4 MiB）以及 LL / LL128 协议的专用 buffer，一个 communicator 在几十到几百 MB 之间。一个多维并行的训练进程有多个 communicator：TP 组、DP 组、PP 的相邻 stage、CP 组、embedding 组……PyTorch 为每个 `ProcessGroup` 惰性创建 NCCL communicator，第一次在该组通信时分配。所以**NCCL 的显存在第一个 step 才出现**，几百 MB 到 2–3 GB，取决于组的数量与 channel 数。它也是通过 `cudaMalloc` 直接分配的，不在 `memory_reserved()` 里。本系列把集合通信当黑盒，但这一项在显存预算里不能漏。

### 3. Caching allocator：reserved、allocated 与碎片

PyTorch 不为每个张量调用 `cudaMalloc`——那太慢，而且会与 NCCL 等异步操作产生同步。它用一个 caching allocator（`c10/cuda/CUDACachingAllocator.cpp`）向驱动申请大块 **segment**，再在 segment 内切 **block** 分给张量；张量释放时 block 回到缓存池而不还给驱动。于是有三个不同的数：

```text
allocated   张量实际占用的字节（所有活着的 block 之和）              torch.cuda.memory_allocated()
reserved    allocator 向驱动申请、尚未归还的字节（所有 segment 之和）   torch.cuda.memory_reserved()
device used 驱动看到的本进程占用：reserved + context + NCCL + 其他库    torch.cuda.mem_get_info() 的差 / nvidia-smi
```

`torch.cuda.memory_stats()`（`torch/cuda/memory.py`）返回全部计数器，键的形式是 `"{stat}.{pool}.{current|peak|allocated|freed}"`：`allocated_bytes`、`reserved_bytes`、`active_bytes`、`inactive_split_bytes`（碎片：segment 里被切出、当前空闲但因为相邻 block 在用而无法合并的部分）、`requested_bytes`（用户实际请求的字节，与 `allocated_bytes` 的差是分配器的对齐与舍入开销）、`num_alloc_retries`（`cudaMalloc` 失败后释放缓存重试的次数——它不为零说明已经在 OOM 边缘）、`num_ooms`。Megatron 的 `training_log()` 在 `--log-memory-to-tensorboard` 时记录的就是其中的 `reserved_bytes.all.current`、`allocated_bytes.all.current`、`allocated_bytes.all.peak`。

reserved 与 allocated 的差有两个来源。一是**缓存**：反向释放的激活 block 留在池里等下一个 step 复用，这是设计如此，不是浪费——但它意味着 reserved 在第一个 step 后就基本等于峰值 allocated 加上碎片。二是**碎片**：分配器把请求分成小池（≤ 1 MiB，`kSmallSize`，装在 2 MiB 的 segment 里，`kSmallBuffer`）与大池（大于 1 MiB，segment 至少 20 MiB，`large_segment_size` 默认 20971520，即原先的 `kLargeBuffer`；大于 10 MiB 的请求按 2 MiB 取整单独分配，`kMinLargeAlloc` / `kRoundLarge`；所有请求至少按 512 字节对齐，`kMinBlockSize`——这些常数在 `c10/core/AllocatorConfig.h`）。大池里不同大小的张量交替分配释放，会留下大量无法合并的空洞，典型的训练进程碎片在 reserved 的 5–15%。`PYTORCH_ALLOC_CONF`（2.13.0 推荐的通用名字，旧名 `PYTORCH_CUDA_ALLOC_CONF` 仍接受，见 `c10/core/AllocatorConfig.cpp`）的 `expandable_segments:True` 让 segment 可以通过虚拟地址映射按需扩展，把多个物理块拼成一个连续的虚拟区间，大幅降低碎片，是长序列、变长 batch 训练的常用设置；`garbage_collection_threshold` 与 `max_split_size_mb` 是另两个相关开关。

### 4. 一张 H100 的预算

把以上各项放进一张 80 GB（`mem_get_info()` 报告约 79.6 GiB）的卡：

```text
项目                                   典型量级（GiB）    在 memory_reserved() 里？
─────────────────────────────────────  ───────────────   ─────────────────────────
总容量                                   79.6
CUDA context + 全部库的 kernel            0.5 – 1.0        否
NCCL communicators（多维并行、多个组）     1 – 3            否
cuBLAS / cuDNN / TE workspace             0.2 – 0.5        是
caching allocator 碎片 + 对齐             reserved 的 5–15%  是（inactive_split_bytes）
OOM 安全边际（峰值抖动、临时 buffer）      2 – 4            —
─────────────────────────────────────  ───────────────
可以分给四种状态的张量                      约 68 – 74
```

这就是"80 GB 的卡实际只有 70 多 GB 可用"的来源。它不是一个固定的数——组多、碎片重、workspace 大的配置只剩 65 GiB，精心调过的配置能到 75 GiB。第四篇的配置推导里，每卡显存预算一律按 70 GiB 起算，留出的部分就是这一章。

### 5. 对账的工具

Megatron 的 `report_memory()`（`megatron/training/utils/common_utils.py`）在第一次报告时打印 `memory_allocated` / `max_memory_allocated` / `memory_reserved` / `max_memory_reserved` 四个数，`--log-device-memory-used` 时再加 `torch.cuda.device_memory_used()`——三层数字一次看全：张量、allocator、驱动。与它并排打印的 `report_theoretical_memory()` 给理论值。两者的差按本章的四类逐项归因，就是第三篇"实测与理论对账"的方法。`torch.cuda.memory._record_memory_history()` 加 `memory_snapshot()`（Megatron 的 `--record-memory-history` 打开记录、`--memory-snapshot-path` 指定 pickle 路径，`training_log()` 在日志间隔 dump）能进一步给出每个 block 是哪行代码分配的，是碎片与泄漏排查的工具。


## 六、算力账：6N 与注意力项

### 1. 每参数每 token 6 FLOP

一个权重矩阵 $$W \in \mathbb{R}^{n \times k}$$ 在前向里参与一次 GEMM：一个 token 的输入 $$x \in \mathbb{R}^{n}$$ 乘 $$W$$ 得到 $$k$$ 个输出，每个输出是 $$n$$ 次乘加——$$2nk$$ FLOP，即**每个参数 2 FLOP**。反向要算两个 GEMM：对输入的梯度 $$\partial x = \partial y \, W^T$$（dgrad，$$2nk$$）和对权重的梯度 $$\partial W = x^T \partial y$$（wgrad，$$2nk$$），合计**每个参数 4 FLOP**。前向加反向就是每参数每 token **6 FLOP**：

$$
\text{FLOP/token} \approx 6N
$$

严格地说，这里的参数数应是**参与 GEMM 的参数数** $$N_{\text{GEMM}}$$。输入 embedding 是查表，不算；输出层 $$[h, V]$$ 是一个真正的 GEMM，要算；LayerNorm、bias、RMSNorm 的权重参与的是逐元素运算，量级是 $$h$$ 而不是 $$h^2$$，忽略。对 Llama 3 70B，$$N_{\text{GEMM}} = 68.45\text{B（层）} + 1.05\text{B（输出层）} = 69.5\text{B}$$，与总参数量 $$N = 70.55\text{B}$$ 差 1.5%，所以工程上直接写 $$6N$$，本系列也不再区分。

这个 6 里没有激活函数、softmax、LayerNorm、残差加法、优化器更新——它们都是 $$O(sbh)$$ 而不是 $$O(sbh^2)$$，在 $$h$$ 为几千时占比不到 1%。也没有重计算：重计算是硬件真实执行的 FLOP，但不是模型"需要"的 FLOP，第七章会用这个区别定义 MFU 与 HFU。

### 2. 注意力的 s² 项

注意力有两个不涉及权重的 GEMM：$$QK^T$$（$$[s, d] \times [d, s]$$）和 $$PV$$（$$[s, s] \times [s, d]$$），每个头 $$2s^2d$$ FLOP，$$a$$ 个头合计 $$2s^2h$$，两个 GEMM $$4s^2h$$——这是**每条序列**前向的量。除以 $$s$$ 得每 token 前向 $$4sh$$，乘 3（前向 + 两个反向 GEMM）得每 token 每层 $$12sh$$，$$L$$ 层：

$$
\text{FLOP/token} = 6N + 12\,L\,s\,h \quad (\text{全 mask})
$$

因果 mask 下分数矩阵只有下三角有效，FlashAttention 一类实现会跳过上三角的块，所以有效计算量减半：$$6Lsh$$。Megatron 按因果减半计（下节）。这一项与 $$6N$$ 的比是 $$\frac{6Lsh}{6 \cdot 12Lh^2} \approx \frac{s}{12h}$$（用 $$N \approx 12Lh^2$$），$$s = 8192$$、$$h = 8192$$ 时约 8%，$$s = 128K$$ 时约 130%——**长上下文训练里注意力的 FLOP 超过所有 GEMM 之和**，这是第二篇 CP 一章的算力背景。

### 3. Megatron 的 num_floating_point_operations()

`megatron/training/training.py` 的 `num_floating_point_operations(args, batch_size, ...)` 计算**一个 global batch** 的 FLOP，训练日志里的 `throughput per GPU (TFLOP/s/GPU)` 就是它除以 step 时间、$$10^{12}$$ 与 `world_size`。对标准 Transformer，它的内层函数 `transformer_flops()` 用三个显式的因子：

```python
# - 3x: Each GEMM in the model needs to be performed 3 times (forward pass,
#       backward wgrad [weight gradient], backward dgrad [data gradient]).
forward_backward_expansion_factor = 3
# - 2x: A GEMM of a m*n tensor with a n*k tensor requires 2mnk floating-point operations.
fma_expansion_factor = 2
# - 3x (SwiGLU enabled): h->2*ffn_h GEMM and ffn_h->h GEMM are stacked.
ffn_expansion_factor = 3 if args.swiglu else 2
```

$$3 \times 2 = 6$$ 就是上一节的 6。随后的总式分成两段：**与 token 数线性**的一段——MLP（$$6 \cdot h \cdot f \cdot 3$$ per layer，SwiGLU 三个矩阵）、注意力投影（$$6 \cdot h(h_q + h_k + h_v) + 6 \cdot h_q h$$，GQA 下 $$h_k = h_v = g \cdot d$$）、logits（$$6 \cdot h \cdot V$$）——乘 `total_real_tokens_in_batch`；以及**与 $$\sum_i L_i^2$$ 成正比**的注意力分数一段，系数是：

```python
standard_self_attn_core_term = (
    forward_backward_expansion_factor
    * fma_expansion_factor
    * query_projection_size
    / 2  # causal mask (only half of the mask is non-zero)
    * 2  # QK^T and (QK^T)V
)
```

即每层 $$3 \times 2 \times h \times \tfrac{1}{2} \times 2 = 6h$$，乘 $$\sum_i L_i^2$$（默认 $$b \cdot s^2$$）——正是上一节因果减半后的 $$6Lsh$$ 每 token。0.18.0 把 $$\sum L_i^2$$ 做成参数，是为了 packed sequence（THD 布局）下按真实子序列长度计，而不是按 padding 后的 $$s^2$$ 高估。函数还覆盖 MoE（只算路由到的 top-k 专家）、MLA、MTP、Mamba 混合层，本篇不展开。

注意 Megatron 的 FLOP 里**不含重计算**——`transformer_flops()` 没有任何与 `recompute_granularity` 相关的项。所以 Megatron 日志里的 TFLOP/s/GPU 是**模型 FLOP** 口径，除以峰值就是下一章的 MFU 而不是 HFU。

### 4. 三档模型的 FLOP/token 与 step 时间下限

$$s = 8192$$，因果注意力：

```text
                 N           6N           6N + 6Lsh（精确）   一条 8192 序列的 FLOP    一张 H100 的下限（989 TFLOPS）
Llama 3 8B        8.03 B     48 GFLOP      51 GFLOP           4.2 × 10¹⁴              0.43 s
Llama 3 70B      70.55 B    423 GFLOP     449 GFLOP           3.7 × 10¹⁵              3.72 s
Llama 3 405B    405.85 B   2435 GFLOP    2524 GFLOP           2.1 × 10¹⁶             20.9 s
```

"下限"的意思是：如果这条序列的每一个 FLOP 都以 989 TFLOPS 的速度执行、没有任何通信、气泡、kernel 启动与显存带宽瓶颈，需要这么久。实际训练的 step 时间做到这个下限的 2.5 倍以内——即 MFU 40%——就是好成绩（第七章）。

把它放大到一个真实的 step：70B、global batch 4M token（$$2^{22} = 4{,}194{,}304$$）、1024 张 H100：

$$
T_{\text{step}} \ge \frac{4.19 \times 10^6 \times 4.49 \times 10^{11}}{1024 \times 9.89 \times 10^{14}} = 1.86\ \text{s}
$$

40% MFU 下是 4.65 s。15T token 的预训练要 $$15 \times 10^{12} / 4.19 \times 10^6 = 3.6 \times 10^6$$ 个 step，每个 4.65 s，1024 卡上约 193 天；换成 8192 卡是 24 天。这是第四篇配置推导的目标函数——每一个 MFU 百分点在这个尺度上值几天。


## 七、MFU 与 HFU

### 1. PaLM 的定义

PaLM 论文（Chowdhery et al. 2022，附录 B）定义 **Model FLOPs Utilization（MFU）**为：观测到的吞吐（token/s）与系统在峰值 FLOPS 下能达到的理论最大吞吐之比。理论最大吞吐 = 卡数 × 峰值 FLOPS ÷ 每 token 的**模型 FLOP**（上一章的 $$6N + $$ 注意力项，不含重计算）。等价地：

$$
\text{MFU} = \frac{\text{tokens/s} \times \text{FLOP/token}_{\text{model}}}{N_{\text{GPU}} \times \text{峰值 FLOPS}}
$$

PaLM 同时定义了 **Hardware FLOPs Utilization（HFU）**：硬件实际执行的 FLOP（含重计算）与峰值之比。两者的分子只差重计算那一项。PaLM 选 MFU 作为报告口径的理由是：HFU 依赖实现——开了全量重计算，硬件多执行 1/3 的 FLOP，HFU 就"好看"了 1/3，但训练一个 token 的**时间**没有变快——而 MFU 只依赖模型与硬件，是可以跨系统、跨实现比较的数。

这个定义有两个容易忽略的约定。第一，"峰值 FLOPS"用**dense** 的标称值（H100 bf16 989 TFLOPS，不是带结构化稀疏的 1979）。第二，模型 FLOP 是否包含注意力的 $$s^2$$ 项没有统一惯例：PaLM 附录 B 的公式包含它，很多工程报告直接用 $$6N$$。$$s = 8192$$ 时两者差 6%（70B），报告 MFU 时应注明口径。本系列一律用带因果注意力项的精确值，与 Megatron 的 `num_floating_point_operations()` 一致。

### 2. 重计算：HFU 上升，MFU 不变

全量激活重计算让每层的前向执行两次：正常前向 + 反向前的重算。每参数每 token 的硬件 FLOP 从 $$2 + 4 = 6$$ 变成 $$2 + 2 + 4 = 8$$：

$$
\text{HFU}_{\text{full}} = \text{MFU} \times \frac{8}{6} = \frac{4}{3}\,\text{MFU}
$$

选择性重计算（只重算注意力分数）多的只是前向那部分 $$QK^T$$ 与 $$PV$$：每层每 token $$2sh$$（因果），相对 $$6N + 6Lsh$$ 是很小的一项，HFU 只比 MFU 高一两个百分点。

于是同一个训练任务，开全量重计算之后：step 时间变长（多算了 1/3 的前向）→ tokens/s 下降 → MFU **下降**；而 HFU 的分子也乘了 4/3，抵消了大部分下降，甚至可能上升（重计算的 GEMM 形状规整、kernel 效率高）。**HFU 上升不等于训练变快**。这是为什么读任何"利用率"数字时要先问它是哪一个。

### 3. 同一个例子算两遍

70B、1024 张 H100、global batch 4M token，实测 step 时间 4.5 s：

```text
tokens/s                = 4,194,304 / 4.5                    =  932 K token/s
模型 FLOP/token         = 4.49 × 10¹¹（第六章精确值）
模型 FLOP/s（全集群）    = 932 × 10³ × 4.49 × 10¹¹            =  4.19 × 10¹⁷
峰值 FLOP/s（全集群）    = 1024 × 9.89 × 10¹⁴                  =  1.013 × 10¹⁸
MFU                     = 4.19 × 10¹⁷ / 1.013 × 10¹⁸         =  41.3%
```

假设这个 4.5 s 是**开了全量重计算**才把激活塞进显存的：

```text
硬件 FLOP/token         = 4.49 × 10¹¹ × 4/3                  =  5.99 × 10¹¹
HFU                     = 41.3% × 4/3                         =  55.1%
```

如果只是选择性重计算（或 FlashAttention 内建的重算）：

```text
额外 FLOP/token         = 2 × L × s × h = 2 × 80 × 8192 × 8192 = 1.07 × 10¹⁰    （占模型 FLOP 的 2.4%）
HFU                     = 41.3% × 1.024                       =  42.3%
```

同一个 4.5 s，三个数字：MFU 41.3%、HFU 42.3% 或 55.1%。用第八章的 `cli.py --model 70b --tokens 4194304 --gpus 1024 --step-time 4.5` 可以复现这三个数。对训练团队有意义的只有 41.3%——它直接换算成"这 15T token 要跑多少天"；55.1% 只说明 GPU 忙不忙，不说明忙得有没有用。

### 4. 参考水平

有了定义，可以给出几个公开的锚点（全部为论文报告值）：

```text
来源                                          硬件                   利用率            口径说明
─────────────────────────────────────────    ─────────────────     ──────────────    ──────────────────────────────────────────
Narayanan et al. 2021（Megatron-LM，SC'21）   3072 × A100，1T 模型    52%（163 TFLOPS）  论文的 FLOP 公式 96 s L h²(…) 含全量重计算的
                                                                                       那次前向（8 而不是 6），按 PaLM 术语更接近 HFU
Chowdhery et al. 2022（PaLM）                 6144 × TPU v4，540B    46.2% MFU         PaLM 自己的定义；同表给 MT-NLG 530B 30.2% MFU
Llama 3 论文（Dubey et al. 2024）Table 4      8192 × H100，405B     43%（430 TFLOPS）  bf16 MFU；TP 8 / CP 1 / PP 16 / DP 128，s = 8192
                                             16384 × H100          41%（400 TFLOPS）  同上，DP 翻倍
                                             16384 × H100，长上下文  38%（380 TFLOPS）  s = 131072，CP 16
```

从这几个数得到本系列反复使用的判断标准：**千卡 H100 上 dense 模型做到 40% 以上是好成绩；做不到 30% 说明有明确的问题**（气泡、通信未重叠、重计算过度、数据等待、straggler 之一或几个）。第四篇的任务就是把 41% 与 30% 之间的差逐项归因。

一个附带的观察：Llama 3 的 405B 从 8K 卡到 16K 卡 MFU 只掉了 2 个点（43% → 41%），说明 DP 的通信在 16K 卡上仍然基本被反向藏住了——这是第二篇"DP 通信量与 $$N_d$$ 无关"的一个实证。

### 5. Megatron 的 TFLOP/s/GPU 日志

打开 `--log-throughput` 后，Megatron 每个日志间隔打印 `throughput per GPU (TFLOP/s/GPU)`，计算方式在 `training_log()` 里：

```python
throughput = num_floating_point_operations(args, batch_size, ...) / (
    elapsed_time_per_iteration * 10**12 * args.world_size
)
```

分子是第六章的模型 FLOP（不含重计算），所以这个数除以峰值就是 MFU：430 TFLOP/s/GPU 在 H100 上是 43%。`compute_throughputs_and_append_to_progress_log()` 另外维护整个任务从启动以来的累计吞吐（`Job throughput` 与 `Cumulative throughput`），把 checkpoint、重启、故障的时间都摊进去——这是第六篇"有效训练时间"的一个原始数据源。


## 八、本文小结

### 1. 要点回顾

- 训练的全部状态是四样：参数、梯度、优化器状态、激活。前三样是 $$N$$ 的线性函数，常驻或每 step 满一次；激活是每 micro-batch token 数的线性函数，前向堆高、反向释放。激活峰值在前向末，梯度峰值在反向末，两者不同时。
- bf16 混合精度 + Adam 每参数 16 字节：bf16 参数 2 + bf16 梯度 2 + fp32 主参数 4 + fp32 一阶矩 4 + 二阶矩 4。fp32 主参数不是冗余——bf16 的 0.4% 相对精度存不下 $$10^{-5}$$ 量级的更新。Megatron 默认 fp32 累加梯度，18 字节（2 + 4 + 12），分布式优化器下 $$6 + 12/N_d$$。
- 三档模型的常驻状态：8B 128 GB、70B 1.13 TB、405B 6.49 TB。8B 单卡也放不下；70B 至少 15 张 H100 的显存；405B 的数就是它 checkpoint 的量级。
- 激活每层 $$sbh(34 + 5as/h)$$（Korthikanti et al. 2022）；34 = 10（TP 切不开）+ 24（TP 切得开）；$$5as/h$$ 是分数矩阵，FlashAttention 不物化它，条件是用融合 kernel、注意力 dropout 不落地。$$s = 8192$$、$$b = 1$$：8B 每层 1.14 GB、70B 2.28 GB、405B 4.56 GB；不用 Flash 则各乘 10 倍。logits 一条序列 6.3 GB。
- Megatron 的 `compute_activation_memory()` 用 $$18 + 4f/h$$（SP + 选择性重计算），`compute_activation_memory_without_sp()` 用 $$10 + 24/t$$；两者都不含 $$5as/h$$。
- 80 GB 的卡：CUDA context 0.5–1 GiB、NCCL 1–3 GiB、库 workspace、allocator 碎片 5–15%、安全边际——留给张量的约 68–74 GiB。`memory_allocated` < `memory_reserved` < 驱动看到的占用，三层数字分别对应张量、allocator、进程。
- 每参数每 token 6 FLOP（前向 2、dgrad 2、wgrad 2），严格是参与 GEMM 的参数 $$N_{\text{GEMM}}$$（含输出层，不含输入 embedding），与总参数量 $$N$$ 差约 1.5%，工程上不区分；注意力分数每层每 token $$12sh$$，因果减半为 $$6sh$$。70B、$$s = 8192$$：449 GFLOP/token；一条序列在一张 H100 上下限 3.72 s；4M token/step、1024 卡下限 1.86 s。
- MFU（PaLM）= 观测 token/s × 模型 FLOP/token ÷（卡数 × 峰值），不含重计算；HFU 含重计算。全量重计算 HFU = 4/3 MFU。同一个 4.5 s：MFU 41.3%、HFU 42.3%（选择性）或 55.1%（全量）。Megatron 论文 A100 上 52%（含重计算），PaLM 46.2% MFU，Llama 3 405B 38–43% MFU；千卡 H100 dense 40% 以上是好成绩。
- Megatron 的 `num_floating_point_operations()` 用 3 × 2 的因子、因果减半、按 $$\sum L_i^2$$ 算注意力、不含重计算；日志里的 TFLOP/s/GPU 除以峰值就是 MFU。

### 2. 符号与公式速查

```text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
符号
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
N 参数量      s 序列长   b micro-batch   h 隐藏维   a 头数   L 层数   f FFN 维   V 词表
N_d N_t N_p N_c N_e  数据 / 张量 / 流水 / 上下文 / 专家并行度      m 梯度累积步数      B = s·b·m·N_d
δ 一次 checkpoint 的开销时间      M 集群 MTBF

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
字节
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
常驻状态（bf16 + Adam）        16N = 2N 参数 + 2N 梯度 + 12N（fp32 主参数 + m + v）
常驻状态（fp32 梯度，Megatron） 18N = 2N + 4N + 12N；分布式优化器 (6 + 12/N_d)N
激活 / 层 / micro-batch        sbh(34 + 5as/h)；FlashAttention → 34sbh；TP+SP → 34sbh/N_t；34 = 10 + 24
logits                        s·b·V·(2 + 4)
每卡可用                       ≈ 总容量 − context − NCCL − workspace − 碎片 − 边际 ≈ 70 GiB（80 GB 卡）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FLOP 与时间
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FLOP / token                  6N + 6Lsh（因果；全 mask 为 12Lsh），N = 参与 GEMM 的参数 ≈ N
step 下限                      T ≥ B × FLOP/token ÷ (N_GPU × 峰值)
MFU                           tokens/s × FLOP/token ÷ (N_GPU × 峰值)          不含重计算
HFU                           MFU × 硬件 FLOP / 模型 FLOP；全量重计算 × 4/3；选择性 × (1 + 2Lsh / FLOP/token)
参考                           H100 989 TFLOPS（bf16 dense，标称）；A100 312；千卡 dense 好成绩 ≥ 40% MFU

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
三档模型 @ s = 8192, b = 1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
              N          16N        激活/层     全部激活+logits    FLOP/token    单卡单序列下限
8B            8.03 B     128 GB     1.14 GB      42.8 GB          51 GFLOP      0.43 s
70B          70.55 B    1.13 TB     2.28 GB     188.8 GB         449 GFLOP      3.72 s
405B        405.85 B    6.49 TB     4.56 GB     581.3 GB        2524 GFLOP     20.9 s
```

### 3. 本篇涉及的源码位置

| 路径 | 内容 |
|---|---|
| Megatron Core 0.18.0 `megatron/training/theoretical_memory_usage.py` | `compute_weight_and_optimizer_memory()`：按结构公式算参数量、最重分片，内部 `num_bytes_per_parameter()` 返回 18 或 $$6 + 12/d$$；`compute_activation_memory()`（SP + 选择性重计算，每层 $$18 + 4f/h$$，除以 TP）；`compute_activation_memory_without_sp()`（每层 $$10 + 24/t$$，×1.05）；`report_theoretical_memory()` 按 `sequence_parallel` / `recompute_granularity` 选分支 |
| Megatron Core 0.18.0 `docs/user-guide/features/dist_optimizer.md` | 每参数字节表：fp16 20 / $$4 + 16/d$$，bf16 18 / $$6 + 12/d$$，fp32 16 / $$8 + 8/d$$ |
| Megatron Core 0.18.0 `megatron/training/training.py` | `num_floating_point_operations()` 及内层 `transformer_flops()`（因子 3 × 2、SwiGLU 3、因果 /2、按 $$\sum L_i^2$$ 算注意力、不含重计算）；`training_log()` 计算 `throughput`（TFLOP/s/GPU）、在 `--log-memory-to-tensorboard` 时记录 `torch.cuda.memory_stats()` 的 `reserved_bytes` / `allocated_bytes`、首次报告时调用 `report_theoretical_memory()` 与 `report_memory()`；`compute_throughputs_and_append_to_progress_log()` |
| Megatron Core 0.18.0 `megatron/training/utils/common_utils.py` | `report_memory()`：allocated / max allocated / reserved / max reserved（/ device memory used） |
| Megatron Core 0.18.0 `megatron/training/arguments.py` | `validate_args()`：bf16 下默认 `accumulate_allreduce_grads_in_fp32 = True`（18 字节的来源），`--grad-reduce-in-bf16` 关闭 |
| Megatron Core 0.18.0 `megatron/core/distributed/distributed_data_parallel.py`、`param_and_grad_buffer.py` | 反向 hook `param.main_grad.add_(param.grad.data)`；`_ParamAndGradBuffer` 的连续梯度 buffer，dtype 由 `grad_reduce_in_fp32` 决定 |
| Megatron Core 0.18.0 `megatron/core/transformer/transformer_config.py` | `TransformerConfig.recompute_granularity` / `recompute_method` / `recompute_num_layers` |
| PyTorch 2.13.0 `torch/cuda/memory.py` | `memory_stats()`（键 `allocated_bytes` / `reserved_bytes` / `active_bytes` / `inactive_split_bytes` / `requested_bytes` / `num_alloc_retries` / `num_ooms`）、`memory_allocated()`、`memory_reserved()`、`max_memory_allocated()`、`mem_get_info()`、`memory_summary()`、`_record_memory_history()` |
| PyTorch 2.13.0 `c10/core/AllocatorConfig.h`、`AllocatorConfig.cpp` | `kMinBlockSize` 512、`kSmallSize` 1 MiB、`kSmallBuffer` 2 MiB、`kMinLargeAlloc` 10 MiB、`kRoundLarge` 2 MiB、`large_segment_size()` 默认 20 MiB；环境变量 `PYTORCH_ALLOC_CONF`（旧名 `PYTORCH_CUDA_ALLOC_CONF`）与 `expandable_segments` 等选项 |
| PyTorch 2.13.0 `c10/cuda/CUDACachingAllocator.cpp` | caching allocator 本体：segment / block、小池与大池、expandable segments |
| PyTorch 2.13.0 `aten/src/ATen/cuda/CublasHandlePool.cpp` | `getChosenWorkspaceSize()`：Hopper / Blackwell 默认 32 MiB cuBLAS workspace，`CUBLAS_WORKSPACE_CONFIG` 覆盖 |
| train-ledger `ledger/model.py`、`ledger/memory.py`、`ledger/flops.py`、`cli.py` | 本篇增量，见下 |

### 4. train-ledger 本篇增量

练手项目 `train-ledger/` 的第一批文件是本篇公式的可执行版本：输入模型结构、精度、优化器、micro-batch、每 step token 数、卡数与峰值算力，输出四种状态的字节数、FLOP/token、step 时间下限、MFU 与 HFU。它不依赖 torch，后面七篇会持续给它加文件（第二篇加 `ledger/parallel.py`，它 import 的正是这里的 `ModelSpec`、`StateBytes` 与 `state_bytes()`）。

`ledger/model.py`——模型规格与参数量：

```python
"""train-ledger / ledger/model.py -- model specification and parameter count.

ModelSpec describes a dense decoder-only Transformer (Llama style: RMSNorm,
GQA, SwiGLU MLP, untied input/output embeddings). No torch dependency.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelSpec:
    name: str
    layers: int            # L
    hidden: int            # h
    heads: int             # a
    vocab: int             # V
    seq_len: int           # s (training sequence length)
    ffn_hidden: int        # f (SwiGLU: three h x f matrices)
    kv_heads: int | None = None     # GQA groups; None = MHA
    tie_embeddings: bool = False

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    @property
    def kv_dim(self) -> int:
        return (self.kv_heads or self.heads) * self.head_dim

    @property
    def layer_params(self) -> int:
        h, f = self.hidden, self.ffn_hidden
        attn = h * h + 2 * h * self.kv_dim + h * h      # Wq, Wk, Wv, Wo
        mlp = 3 * h * f                                 # gate, up, down
        norms = 2 * h                                   # two RMSNorms
        return attn + mlp + norms

    @property
    def embedding_params(self) -> int:
        emb = self.vocab * self.hidden
        return emb if self.tie_embeddings else 2 * emb

    @property
    def params(self) -> int:
        """N: total trainable parameters."""
        return self.layers * self.layer_params + self.embedding_params + self.hidden

    @property
    def non_embedding_params(self) -> int:
        return self.params - self.embedding_params


def llama3_8b() -> ModelSpec:
    return ModelSpec("llama3-8b", layers=32, hidden=4096, heads=32, vocab=128256,
                     seq_len=8192, ffn_hidden=14336, kv_heads=8)


def llama3_70b() -> ModelSpec:
    return ModelSpec("llama3-70b", layers=80, hidden=8192, heads=64, vocab=128256,
                     seq_len=8192, ffn_hidden=28672, kv_heads=8)


def llama3_405b() -> ModelSpec:
    return ModelSpec("llama3-405b", layers=126, hidden=16384, heads=128, vocab=128256,
                     seq_len=8192, ffn_hidden=53248, kv_heads=8)
```

`ledger/memory.py`——四种状态的字节数：

```python
"""train-ledger / ledger/memory.py -- bytes of the four training states.

state_bytes(model, precision, optimizer) -> StateBytes    resident: params, grads, optim
activation_bytes_per_layer(model, micro_batch, flash)     transient: one layer, one micro-batch
No torch dependency.
"""
from __future__ import annotations

from dataclasses import dataclass

from ledger.model import ModelSpec

# bytes per parameter: compute-dtype params, grads, and the fp32 master copy
# (only when the compute dtype is not fp32).
PRECISION = {
    "fp32":          dict(param=4, grad=4, master=0),   # 16 B/param with Adam
    "bf16":          dict(param=2, grad=2, master=4),   # 16 B/param with Adam
    "bf16-fp32grad": dict(param=2, grad=4, master=4),   # Megatron default: 18 B/param
    "fp16":          dict(param=2, grad=2, master=4),   # 16 B + loss scale
}
OPTIMIZER = {"adam": 2, "adamw": 2, "sgd-momentum": 1, "sgd": 0}   # fp32 slots / param


@dataclass
class StateBytes:
    params_bytes: int
    grads_bytes: int
    optim_bytes: int     # fp32 master params + optimizer moments

    @property
    def total(self) -> int:
        return self.params_bytes + self.grads_bytes + self.optim_bytes


def bytes_per_param(precision: str, optimizer: str) -> tuple[int, int, int]:
    p = PRECISION[precision]
    return p["param"], p["grad"], p["master"] + 4 * OPTIMIZER[optimizer]


def state_bytes(model: ModelSpec, precision: str = "bf16", optimizer: str = "adam") -> StateBytes:
    n = model.params
    bp, bg, bo = bytes_per_param(precision, optimizer)
    return StateBytes(params_bytes=n * bp, grads_bytes=n * bg, optim_bytes=n * bo)


def activation_bytes_per_layer(model: ModelSpec, micro_batch: int = 1, flash: bool = True) -> float:
    """Korthikanti et al. 2022: s*b*h*(34 + 5*a*s/h) bytes per layer, bf16 activations.
    The 5*a*s/h term is the materialised score matrix (softmax out, dropout mask,
    dropout out); FlashAttention never stores it, so flash=True drops it."""
    s, b, h, a = model.seq_len, micro_batch, model.hidden, model.heads
    total = 34.0 * s * b * h
    if not flash:
        total += 5.0 * a * s * s * b
    return total


def activation_bytes(model: ModelSpec, micro_batch: int = 1, flash: bool = True) -> float:
    """All layers, one micro-batch, no recompute, plus logits (bf16 + fp32 copy for the loss)."""
    logits = model.seq_len * micro_batch * model.vocab * (2 + 4)
    return model.layers * activation_bytes_per_layer(model, micro_batch, flash) + logits


def fmt(b: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(b) < 1000 or unit == "TB":
            return f"{b:7.2f} {unit}"
        b /= 1000
```

`ledger/flops.py`——FLOP/token、MFU/HFU 与 step 时间下限：

```python
"""train-ledger / ledger/flops.py -- FLOPs per token, MFU / HFU, step-time bound.
No torch dependency.
"""
from __future__ import annotations

from ledger.model import ModelSpec

H100_SXM_BF16_DENSE = 989e12        # nominal, no sparsity
A100_SXM_BF16_DENSE = 312e12


def flops_per_token(model: ModelSpec, causal: bool = True) -> float:
    """Forward + backward FLOPs per token.
    6 x (every parameter that sits in a GEMM: non-embedding params + output head h*V):
      forward 2 FLOP/param, backward 4 (dgrad + wgrad).
    + attention scores: QK^T and PV are 2*s*h each per token forward, x3 with backward
      = 12*s*h per layer; a causal mask halves the useful work (Megatron counts it so)."""
    gemm_params = model.non_embedding_params + model.hidden * model.vocab
    attn = 12.0 * model.layers * model.seq_len * model.hidden
    return 6.0 * gemm_params + (attn / 2 if causal else attn)


def flops_per_token_6n(model: ModelSpec) -> float:
    """The textbook 6N, N = all parameters."""
    return 6.0 * model.params


def recompute_factor(model: ModelSpec, policy: str) -> float:
    """hardware FLOPs / model FLOPs under an activation-recompute policy."""
    if policy == "none":
        return 1.0
    if policy == "full":                 # one extra forward: (2 + 4 + 2) / 6
        return 4.0 / 3.0
    if policy == "selective":            # only the two score matmuls are recomputed
        return 1.0 + 2.0 * model.layers * model.seq_len * model.hidden / flops_per_token(model)
    raise ValueError(policy)


def mfu(model: ModelSpec, tokens_per_sec: float, n_gpus: int, peak_flops: float) -> float:
    """PaLM: observed tokens/s over the tokens/s a system at peak FLOPS would reach,
    counting only the FLOPs the model needs (recompute excluded)."""
    return tokens_per_sec * flops_per_token(model) / (n_gpus * peak_flops)


def hfu(model: ModelSpec, tokens_per_sec: float, n_gpus: int, peak_flops: float,
        recompute: str = "none") -> float:
    """Hardware FLOPs utilisation: mfu() with the recomputed FLOPs counted as useful."""
    return mfu(model, tokens_per_sec, n_gpus, peak_flops) * recompute_factor(model, recompute)


def step_time_lower_bound(model: ModelSpec, tokens_per_step: int, n_gpus: int,
                          peak_flops: float = H100_SXM_BF16_DENSE, mfu_target: float = 1.0) -> float:
    """seconds/step if every GPU sustained mfu_target x peak on the model's own FLOPs."""
    return tokens_per_step * flops_per_token(model) / (n_gpus * peak_flops * mfu_target)
```

`cli.py`——把三者串起来：

```python
#!/usr/bin/env python3
"""train-ledger / cli.py -- states, FLOPs and the step-time bound for one model.

  python cli.py                                   # Llama 3 8B / 70B / 405B, s = 8192
  python cli.py --model 70b --tokens 4194304 --gpus 1024 --step-time 4.5
"""
import argparse

from ledger.model import llama3_8b, llama3_70b, llama3_405b
from ledger.memory import state_bytes, activation_bytes_per_layer, activation_bytes, fmt
from ledger.flops import (flops_per_token, flops_per_token_6n, step_time_lower_bound,
                          mfu, hfu, H100_SXM_BF16_DENSE)

MODELS = {"8b": llama3_8b, "70b": llama3_70b, "405b": llama3_405b}


def report(m, a):
    st = state_bytes(m, a.precision, a.optimizer)
    print(f"== {m.name}: L={m.layers} h={m.hidden} a={m.heads} kv={m.kv_heads} "
          f"f={m.ffn_hidden} V={m.vocab} s={m.seq_len}")
    print(f"  N = {m.params / 1e9:.2f} B   ({a.precision} + {a.optimizer}: {st.total / m.params:.0f} B/param)")
    print(f"  params {fmt(st.params_bytes)}  grads {fmt(st.grads_bytes)}  optim {fmt(st.optim_bytes)}"
          f"  resident {fmt(st.total)}")
    print(f"  activation/layer (b={a.micro_batch}): {fmt(activation_bytes_per_layer(m, a.micro_batch))} flash"
          f" | {fmt(activation_bytes_per_layer(m, a.micro_batch, flash=False))} materialised scores"
          f" | all layers + logits {fmt(activation_bytes(m, a.micro_batch))}")
    print(f"  FLOPs/token: 6N = {flops_per_token_6n(m) / 1e9:.0f} GFLOP, with causal attention"
          f" = {flops_per_token(m) / 1e9:.0f} GFLOP;  one {m.seq_len}-token sequence on one H100"
          f" >= {step_time_lower_bound(m, m.seq_len, 1, a.peak):.2f} s")
    if a.tokens and a.gpus:
        t = step_time_lower_bound(m, a.tokens, a.gpus, a.peak)
        print(f"  {a.tokens / 2**20:.0f} Mi tokens/step on {a.gpus} GPUs: >= {t:.2f} s at 100%,"
              f" {t / a.mfu:.2f} s at {a.mfu:.0%} MFU")
        if a.step_time:
            tps = a.tokens / a.step_time
            print(f"  observed {a.step_time} s/step = {tps / 1e3:.0f} K tokens/s -> "
                  f"MFU {mfu(m, tps, a.gpus, a.peak):.1%}  HFU(selective) "
                  f"{hfu(m, tps, a.gpus, a.peak, 'selective'):.1%}  HFU(full) {hfu(m, tps, a.gpus, a.peak, 'full'):.1%}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--optimizer", default="adam")
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=0, help="tokens per step (global batch)")
    ap.add_argument("--gpus", type=int, default=0)
    ap.add_argument("--peak", type=float, default=H100_SXM_BF16_DENSE)
    ap.add_argument("--mfu", type=float, default=0.4)
    ap.add_argument("--step-time", type=float, default=0.0, help="observed seconds per step")
    args = ap.parse_args()
    for name in ([args.model] if args.model else MODELS):
        report(MODELS[name](), args)
```

运行 `python cli.py` 输出三档模型的账（`ledger/` 目录下要有一个空的 `__init__.py`）：

```text
== llama3-8b: L=32 h=4096 a=32 kv=8 f=14336 V=128256 s=8192
  Psi = 8.03 B   (bf16 + adam: 16 B/param)
  params   16.06 GB  grads   16.06 GB  optim   96.36 GB  resident  128.48 GB
  activation/layer (b=1):    1.14 GB flash |   11.88 GB materialised scores | all layers + logits   42.81 GB
  FLOPs/token: 6N = 48 GFLOP, with causal attention = 51 GFLOP;  one 8192-token sequence on one H100 >= 0.43 s

== llama3-70b: L=80 h=8192 a=64 kv=8 f=28672 V=128256 s=8192
  Psi = 70.55 B   (bf16 + adam: 16 B/param)
  params  141.11 GB  grads  141.11 GB  optim  846.64 GB  resident    1.13 TB
  activation/layer (b=1):    2.28 GB flash |   23.76 GB materialised scores | all layers + logits  188.84 GB
  FLOPs/token: 6N = 423 GFLOP, with causal attention = 449 GFLOP;  one 8192-token sequence on one H100 >= 3.72 s

== llama3-405b: L=126 h=16384 a=128 kv=8 f=53248 V=128256 s=8192
  Psi = 405.85 B   (bf16 + adam: 16 B/param)
  params  811.71 GB  grads  811.71 GB  optim    4.87 TB  resident    6.49 TB
  activation/layer (b=1):    4.56 GB flash |   47.51 GB materialised scores | all layers + logits  581.29 GB
  FLOPs/token: 6N = 2435 GFLOP, with causal attention = 2524 GFLOP;  one 8192-token sequence on one H100 >= 20.91 s
```

第七章的例子：

```text
$ python cli.py --model 70b --tokens 4194304 --gpus 1024 --step-time 4.5
== llama3-70b: ...
  4 Mi tokens/step on 1024 GPUs: >= 1.86 s at 100%, 4.65 s at 40% MFU
  observed 4.5 s/step = 932 K tokens/s -> MFU 41.3%  HFU(selective) 42.3%  HFU(full) 55.1%

$ python cli.py --model 70b --precision bf16-fp32grad
  Psi = 70.55 B   (bf16-fp32grad + adam: 18 B/param)
  params  141.11 GB  grads  282.21 GB  optim  846.64 GB  resident    1.27 TB
```

几点说明。`ModelSpec.params` 按 Llama 结构（GQA、SwiGLU、RMSNorm、不共享的输入输出 embedding）计，与 Megatron `compute_weight_and_optimizer_memory()` 的结构公式是同一套算法的简化版（没有 MoE / MLA / MTP 分支）。`state_bytes()` 的 `precision` 字符串对应第三章的四种组合，`"bf16"` 是 16 字节，`"bf16-fp32grad"` 是 Megatron 默认的 18 字节。`activation_bytes_per_layer()` 用的是论文的 34 而不是 Megatron 的 $$18 + 4f/h$$——两者对 $$f = 4h$$ 相同，对 Llama 3 的 $$f = 3.5h$$ 差 6%，本系列统一用 34 以便与论文对照。`flops_per_token()` 的 `causal=True` 与 Megatron 一致；`recompute_factor()` 的 `"selective"` 只算注意力分数的重算，与 FlashAttention 内建的重算是同一项。这个脚本没有并行维度：它算的是"整个模型放在一张无限大的卡上"的账。下一篇给它加上 TP/PP/DP/CP/EP 与 ZeRO 级别，回答这 1.13 TB 怎么切、切了之后每 step 要搬多少字节。

> **每种并行都在"复制"和"切分"之间做交换：复制多占显存，切分多花通信。给定一个模型和一个集群的拓扑，每个维度的通信量是多少、走哪条链路、和计算能不能重叠？**


## 下一篇

[并行策略全景：每种并行切的是哪种状态](/parallelism-strategies-which-state-to-shard.html)
