---
layout: post
series: lora
title: "LoRA 专题（04）：系列总结与通关自测"
subtitle: "LoRA Series: Recap and Final Self-Test"
tags: [AI, LLM, LoRA, Post-Training, peft]
catalog: true
date: 2026-05-29 20:00:00
---

三篇正文回答了一个问题：**为什么给一个几十亿参数的模型加两个瘦矩阵就能把它微调好，每个旋钮该取多少，训完的东西怎么上线**。第一篇讲低秩假设、$$W + \frac{\alpha}{r}BA$$ 的梯度与四本账，并直接检验全量微调的 $$\Delta W$$ 有多低秩；第二篇把十三种配置在同一份数据上各训 80 步，每个旋钮给一组数字；第三篇从 adapter 文件走到 multi-LoRA 服务。本文把三篇压成一张速查表，拎出贯穿三篇的四条线，列常见误区，再给一套自测。

## 一、总览

```text title="三篇正文的骨架与共用的一组实验"
                 为什么                       怎么配                          怎么上线
              ┌──────────────┐           ┌──────────────────┐           ┌──────────────────┐
  01 低秩假设  │ W + (α/r)BA   │  02 选参  │ r · target · α   │  03 工程  │ adapter 文件      │
              │ 梯度手算       │  ───────► │ lr · 初始化 · DoRA │  ───────► │ 合并 · 量化失配    │
              │ 四本账         │           │ QLoRA · 选择表    │           │ 多 adapter · 服务账 │
              │ ΔW 的谱        │           │ 十三种配置对照     │           │ 参考模型 · 新 token │
              └──────────────┘           └──────────────────┘           └──────────────────┘
                     ▲                            ▲                            ▲
                     └──── 同一组实验：Qwen2.5-0.5B · no_robots 800 条 · 80 步 ────┘
                           验证回复 loss（训练前 2.4936）· 普通文本 loss 变化 · s/步
```

一张速查表：

| 问题 | 答案 | 在哪篇 |
|---|---|---|
| LoRA 的公式 | $$y = Wx + \frac{\alpha}{r}BAx$$，$$A \in \mathbb{R}^{r \times d_{in}}$$ 随机、$$B \in \mathbb{R}^{d_{out} \times r}$$ 为零，$$W$$ 冻结 | 01 §2 |
| 两个梯度 | $$\partial \mathcal{L}/\partial B = sG(Ax)^\top$$，$$\partial \mathcal{L}/\partial A = sB^\top Gx^\top$$<br/>没人算 $$\partial \mathcal{L}/\partial W$$<br/>$$\partial \mathcal{L}/\partial x = W^\top G + \cdots$$ 必须穿过 $$W$$ | 01 §2.2 |
| 可训练参数 | $$r(d_{in} + d_{out})$$ 每矩阵；0.5B 全部线性层 $$r=16$$ 是 8.8M（1.78%），8B 是 41.9M（0.52%） | 01 §3.1 |
| 训练状态 | $$2N_{total} + 14N_{trainable}$$ 字节；8B：120 GiB → 15.5 GiB，96% 是冻结权重 | 01 §3.2 |
| 计算量 | 每 token 约 $$4N$$ 对全量 $$6N$$，只省 1/3<br/>kernel 数翻三倍<br/>激活值一分不省 | 01 §3.3–3.5 |
| ΔW 低秩吗 | 全量微调的 $$\Delta W$$ 本身高秩，但截到秩 16 装回去效果几乎不掉 | 01 §4 |
| $$r$$ | 4 → 16 收益大，16 → 64 收益小，遗忘随之增大 | 02 §2 |
| `target_modules` | `all-linear`（不含 lm_head、embedding）比只挂 attention 重要，比加 $$r$$ 重要 | 02 §3 |
| $$\alpha$$ | scaling $$= \alpha/r$$，与 lr 是一个旋钮的两面：越大学得越猛、忘得越多<br/>固定 $$\alpha$$ 加大 $$r$$ 会压掉更新<br/>$$\alpha = 2r$$ 或 rsLoRA（$$\alpha/\sqrt r$$） | 02 §4 |
| lr | $$10^{-4} \sim 5 \times 10^{-4}$$，比全量大 10–20 倍；太大会遗忘 | 02 §5 |
| 初始化 | PiSSA / OLoRA / CorDA 改底座、发布要转换<br/>EVA 不改底座<br/>LoftQ 只对 QLoRA | 02 §6 |
| DoRA | 幅度方向分开训，小 $$r$$ 收益大（$$r = 16$$ 上与 LoRA 相同），每步要构造 $$W + BA$$、慢 27% | 02 §7 |
| QLoRA | NF4（正态分位数 16 格点）+ 双重量化（0.127 bit/参数）+ 分页优化器；只动冻结权重那本账 | 02 §8 |
| adapter 文件 | 只有 $$A$$、$$B$$（BF16 下 0.5B 17 MB、8B 84 MB）；必须与训练时的底座配对，`base_model_name_or_path` 不校验 | 03 §1 |
| 合并 | $$W \mathrel{+}= sBA$$，logits 差在舍入量级<br/>BF16 下要在 FP32 里加<br/>合并后与底座一样大 | 03 §2 |
| 换精度 | FP32 ↔ NF4、先合并再量化：差别与量化误差同量级，要重测 | 03 §3 |
| 多 adapter | `set_adapter` 零拷贝；合成不同任务用 `cat`/`svd`，`linear` 有交叉项 | 03 §4 |
| 参考模型 | `disable_adapter()` 就是训练开始时的策略；trl 里 `ref_model=None` | 03 §4.3 |
| 服务 | 8B 50 个客户：合并 800 GB 对 multi-LoRA 20 GB；显存按 `max_loras × max_lora_rank` 预留 | 03 §5 |
| 新 token | embedding 冻结所以学不会；`trainable_token_indices` 几 KB，`modules_to_save` 136M | 03 §6 |

Table: 三篇正文的速查表

## 二、逐篇回顾

### 01 低秩假设：为什么两个瘦矩阵够用，以及它省了哪几本账

- **一个公式**：$$W' = W + \frac{\alpha}{r} BA$$，$$B = 0$$ 使起点等于底座，$$A$$ 随机使 $$B$$ 有梯度可走；两个都为 0 学不动，两个都随机把 loss 从 2.49 打到 14.8（实测）。
- **梯度只需两次瘦矩阵乘**：$$\partial B = s\, G A^\top$$、$$\partial A = s\, B^\top G$$，但对输入的梯度 $$W^\top G$$ 一步也省不掉——所以 LoRA 的反向不比全量少多少 FLOPs。
- **四本账里省的是两本**：0.5B 上梯度 1885 MB → 34 MB、Adam 状态 3769 MB → 67 MB；前向没有变快（0.91 s 对 0.94 s，算子反而从 5091 涨到 9963），激活值一样。一步 3.46 s → 2.31 s，省的全是优化器那一段。
- **低秩假设的边界**：全量微调 80 步的 $$\Delta W$$ 本身不低秩（秩 16 只占方阵 40–68%、MLP 矩阵 20–28% 的能量），但截到秩 1 就拿到 95% 的收益、秩 16 好于不截——有用的部分低秩，高秩部分只带来遗忘。这就是"内在秩很低"与"LoRA 学得少忘得也少"是同一件事的原因；它在小数据的格式/风格 SFT 上成立，在大数据的继续预训练上失效。

### 02 选参：r、target_modules、alpha、lr 与 QLoRA / DoRA / PiSSA 各让模型变成什么

十三种配置、同一份数据、同样 80 步，一张表回答"怎么选"：

- **$$r$$ 是容量**：$$r = 4 \to 16$$ 验证回复 loss 从 2.4141 到 2.3943（全量 2.3924），$$16 \to 64$$ 不再改善（2.3958），遗忘却从 +0.0107 涨到 +0.0436、超过全量的 +0.0213。小任务 $$r = 16$$ 够；知识注入要大 $$r$$ 或全量。
- **挂哪里比 $$r$$ 重要**：只挂 attention（2.2M）2.4113，挂全部线性层（8.8M）2.3943；同样参数翻 4 倍，$$r$$ 从 16 到 64 却没有收益。默认 `all-linear`。
- **scaling 与 lr 是一个旋钮**：$$r = 64$$ 上 scaling 0.5 / 2 / 4（rsLoRA）的遗忘 +0.0087 / +0.0436 / +0.0727，scaling 4 的验证 loss 反而最差；lr 1e-5 只学到一半、1e-3 遗忘是全量的 4 倍。LoRA 的 lr 比全量大 10–20 倍，但"LoRA 忘得少"的前提是它学得也少。
- **变体在小任务上不比默认好**：LoRA+（$$B$$ 的 lr × 4）2.3998、PiSSA 2.4152 且遗忘更大、DoRA 与 LoRA 相同（2.3942）但慢 27%；它们各有适用场合（长训练、收敛紧、极小 $$r$$），不是默认开关。
- **QLoRA 只动冻结权重那本账**：NF4（16 个正态分位数格点）+ 双重量化 + 分页优化器，0.5B 的底座从 988 MB 到 430 MB；代价是每步的反量化与底座本身的量化误差，adapter 训出来要和它配对的精度一起看（第三篇）。

### 03 工程：adapter 文件、合并、多 LoRA 服务与参考模型

- **adapter 是两个文件**：`adapter_config.json`（`LoraConfig` + 底座名）与 `adapter_model.safetensors`（键名带着挂载位置；0.5B r=16 全部线性层 FP32 存 33.6 MB，BF16 约 17 MB），底座的版本、词表、精度都是它的隐含依赖。
- **合并是精确的**：`merge_and_unload` 后 logits 差 $$3 \times 10^{-4}$$（FP32 舍入），CPU 单条请求快 18%；BF16 上加 $$\Delta W$$ 要升到 FP32 再加。
- **换底座精度 = 量化误差，不是失效**：FP32 训的 adapter 装到 NF4 底座 2.4860、先合并再量化 2.4854，两者与训练前 NF4 底座（2.5780）的差和 FP32 上一样约 0.09；对着 NF4 训的 QLoRA adapter 在 NF4 上更好（2.4663）。上线前在目标精度上重测一次。
- **多 adapter**：`set_adapter` 零拷贝切换（2.3943 / 2.4113 与各自训完一致）；`add_weighted_adapter` 的 `linear` 有交叉项、`cat` 精确但秩相加、`svd` 精确又可控秩，同任务合成拿到的是平均而非叠加；`disable_adapter()` 就是 DPO / GRPO 的免费参考模型（2.4936 = 底座）。
- **服务账**：50 个 r=16 adapter 挂在一个 BF16 底座上多 0.82 GiB，存 50 份合并模型是 46 GiB；vLLM 的 `max_loras × max_lora_rank` 预留的就是这块。
- **新 token 学不会**：LoRA 不碰 `embed_tokens` / `lm_head`，新 token 那一行训一步改动为 0；`trainable_token_indices` 只训那几行（+896 个参数），`modules_to_save` 训整张表（+272M，比 LoRA 本身大 30 倍）。

## 三、贯穿全系列的几条线

### 1. 一个公式，三篇各用它的一部分

$$W' = W + \frac{\alpha}{r}BA$$。第一篇推它的梯度：$$B$$ 与 $$A$$ 的梯度都是瘦的，$$W$$ 没有梯度，但对输入的梯度必须穿过 $$W$$。第二篇调它的每个符号：$$r$$ 是容量，$$\alpha/r$$ 是缩放，$$A$$、$$B$$ 的初始化决定起点，`target_modules` 决定公式套在哪些 $$W$$ 上，QLoRA 把 $$W$$ 换成 $$Q(W)$$，DoRA 在外面套一层幅度归一化。第三篇把它合并回 $$W$$（`merge`）、按请求切换 $$(A_i, B_i)$$（multi-LoRA）、或把 $$BA$$ 项关掉（`disable_adapter` 当参考模型）。

### 2. 四本账，每一篇都回来算一次

可训练参数、训练状态、计算量、激活值。第一篇建账：LoRA 省第一本 50–200 倍、第二本 5–8 倍、第三本 1/3、第四本不省。第二篇里 $$r$$ 改第一本与第二本的一小部分，QLoRA 改第二本里冻结权重那 96%，DoRA 让第三本变贵。第三篇里 adapter 的大小就是第一本乘 2 字节，multi-LoRA 的槽位是第一本乘 `max_loras`，`modules_to_save` 让第一本涨 15 倍。

### 3. 一个假设，从检验到失效

"$$\Delta W$$ 低秩"。第一篇直接检验：全量微调的 $$\Delta W$$ 本身高秩，但有用的部分低秩，截到秩 16 效果几乎不掉。第二篇看它在哪些配置下成立得更好：挂全部线性层（让低秩的改动落在正确的矩阵上）、$$r$$ 够但不必大。第三篇看它的边界之外：新 token 的 embedding 不在任何 $$W$$ 里，低秩假设对它无从谈起。它失效的场景——知识注入、继续预训练——三篇都指向同一个答案：全量。

### 4. 同一组数字

训练前 2.4936 / 2.8748，全量 80 步 2.3924 / +0.0213。第一篇用它们做截秩实验的标尺，第二篇的十三行都对着它们比，第三篇加载、合并、换精度后每一步都回到同一个验证集上重算一次。一个数字能在三篇之间来回对上，是这组实验可复现的全部证据。

## 四、常见误区

| 误区 | 事实 | 在哪篇 |
|---|---|---|
| "LoRA 可训练参数少 100 倍，所以训练快 100 倍" | 只省对权重的梯度；对输入的梯度照算，FLOPs 只省 1/3，kernel 数翻三倍，CPU 上甚至更慢 | 01 §3.3、§3.5 |
| "LoRA 省显存，长序列也不怕" | 激活值一分不省，batch 4 × 512 的激活已经比 LoRA 全部训练状态大；长序列仍要梯度检查点 | 01 §3.4 |
| "$$\Delta W$$ 是低秩的，所以 LoRA 等价于全量" | $$\Delta W$$ 高秩，有用部分低秩；知识注入与继续预训练上 LoRA 明显不如全量 | 01 §4、§6 |
| "$$r$$ 越大越好" | 边际收益递减，遗忘增大；固定 $$\alpha$$ 时大 $$r$$ 还会被 $$\alpha/r$$ 压掉更新 | 02 §2、§4 |
| "只挂 q、v 就够了（论文这么做的）" | 论文是 2021 年 GPT-3 的设定；QLoRA 之后的共识是全部线性层更重要 | 02 §3 |
| "LoRA 用全量的 lr" | 要大 10–20 倍；$$10^{-5}$$ 下 80 步几乎学不到东西 | 02 §5 |
| "LoRA 不会遗忘" | lr 合适时忘得少，因为学得也少；lr $$10^{-3}$$ 下遗忘超过全量 | 02 §5 |
| "PiSSA 的 adapter 可以直接发布" | 它对着改过的 $$W_{res}$$ 学，装到原始 $$W$$ 上会坏；要转换 | 02 §6、03 §1.3 |
| "adapter 写了 `base_model_name_or_path`，装错底座会报错" | 不校验，形状对得上就能加载；换底座效果不可预测 | 03 §1 |
| "BF16 模型直接 `W += ΔW` 合并" | $$\Delta W$$ 小于 $$W$$ 的一个 ulp，大部分丢失；要在 FP32 里加 | 03 §2 |
| "两个 adapter 用 `linear` 加权就是两个任务都会" | 有交叉项 $$B_1A_2 + B_2A_1$$；不同任务用 `cat`/`svd` | 03 §4.2 |
| "LoRA 训 DPO 要再放一份参考模型" | `disable_adapter()` 就是参考模型，8B 省 16 GB | 03 §4.3 |
| "加了新 token，LoRA 训一训就会用了" | embedding 与 lm_head 冻结，新 token 那一行永远不动 | 03 §6 |

Table: 常见误区

## 五、通关自测

### A. 判断与计算

1. 一个 $$d_{in} = d_{out} = 4096$$ 的矩阵，$$r = 8$$ 的 LoRA 可训练参数是原矩阵的几分之一？$$r = 8$$、$$\alpha = 16$$ 与 $$r = 32$$、$$\alpha = 16$$ 的 scaling 各是多少？开 rsLoRA 后呢？

   <details markdown="1"><summary>答案</summary>
   $$8 \times 8192 / 4096^2 = 1/256$$。scaling：$$16/8 = 2$$ 与 $$16/32 = 0.5$$；rsLoRA：$$16/\sqrt 8 = 5.66$$ 与 $$16/\sqrt{32} = 2.83$$。
   </details>

2. 混合精度 AdamW 下，70B 模型全量微调的训练状态多少？全部线性层 $$r = 16$$ 的 LoRA 呢（线性层约占 95% 参数，按 8B 的 0.52% 比例放大到约 0.4%）？QLoRA 呢？

   <details markdown="1"><summary>答案</summary>
   全量 $$16 \times 70 \times 10^9 = 1.12$$ TB。LoRA：冻结 140 GB + 可训练 $$0.28 \times 10^9 \times 14 = 3.9$$ GB ≈ 144 GB。QLoRA：冻结权重 4.127 bit → 36 GB，加 3.9 GB ≈ 40 GB，一张 48 GB 的卡放得下（不含激活值）。
   </details>

3. 用第一篇的 $$2 \times 3$$ 例子（$$W$$、$$x = (2, 1, 1)$$、$$t = (3, 0)$$、$$s = 2$$），若 $$A = (0, 1, 1)$$、$$B = (0, 0)^\top$$，第一步 $$\partial \mathcal{L}/\partial B$$ 是多少？一步 SGD（lr 0.1）后 $$y$$ 是多少？

   <details markdown="1"><summary>答案</summary>
   $$y = Wx = (4, 1)$$，$$G = (1, 1)$$，$$h = Ax = 0 + 1 + 1 = 2$$，$$\partial \mathcal{L}/\partial B = 2 \cdot (1, 1) \cdot 2 = (4, 4)$$。$$B \leftarrow (-0.4, -0.4)$$，$$y = (4, 1) + 2 \cdot (-0.4, -0.4) \cdot 2 = (2.4, -0.6)$$——一步就冲过了目标 $$(3, 0)$$，lr 对这个 $$h$$ 太大。
   </details>

4. NF4 块内绝对值最大 0.2，权重 0.13。量化到哪个格点、反量化是多少、相对误差多大？

   <details markdown="1"><summary>答案</summary>
   $$0.13/0.2 = 0.65$$，最近的格点是 0.7230（比 0.5626 近），反量化 $$0.7230 \times 0.2 = 0.1446$$，误差 0.0146，相对误差 11%——块内最大值决定了缩放，靠近格点稀疏处的权重误差大。
   </details>

5. vLLM 服务 8B 模型，`--max-loras 16 --max-lora-rank 32`，全部线性层。槽位显存多少？若把 16 个客户的 adapter 全部合并成 16 个模型，存储多少？

   <details markdown="1"><summary>答案</summary>
   $$r = 32$$ 全部线性层 83.9M 参数，$$16 \times 83.9\text{M} \times 2 = 2.7$$ GB。合并方案 $$16 \times 16 = 256$$ GB。
   </details>

### B. 跨篇综合

6. 一个团队用 $$r = 8$$、只挂 q v、lr $$10^{-5}$$、$$\alpha = 8$$ 训了 3 个 epoch，说"LoRA 效果比全量差很多"。按三篇的内容，列出至少四个该检查的地方，按可能性排序。

   <details markdown="1"><summary>答案</summary>
   ① lr $$10^{-5}$$ 对 LoRA 太小（02 §5，最常见）；② 只挂 q v，MLP 没动（02 §3）；③ $$r = 8$$、$$\alpha = 8$$ scaling 只有 1（02 §4，与 lr 叠加更小）；④ 若任务是知识注入或领域差得远，低秩假设本身不成立（01 §6）；⑤ 若加了新 token，embedding 没训（03 §6）。先把配置改成默认起点（$$r = 16$$、all-linear、$$\alpha = 32$$、lr $$10^{-4}$$）再比。
   </details>

7. 用 QLoRA（NF4 底座）训好一个 adapter，线上想用 BF16 底座并合并后交给 vLLM。写出步骤，并说明每一步的数值等价程度与要测什么。

   <details markdown="1"><summary>答案</summary>
   ① 加载 BF16 底座 + adapter（adapter 是对 NF4 底座学的，装到 BF16 上差别与底座量化误差同量级，03 §3）；② 在验证集上算一次 loss，与训练日志比（03 §2.1）；③ `merge_and_unload`，FP32 里加再转 BF16（03 §2.2）；④ 合并前后 logits 差应在 $$10^{-5}$$ 以下；⑤ 若线上要 INT4/FP8，对合并后的模型再量化并重测。若差别不可接受，改用 NF4 底座 + adapter 不合并的方案服务。
   </details>

8. 一个底座要服务 30 个客户，其中 25 个是"改口吻"的小 adapter（$$r = 8$$），5 个是"加领域知识"的大需求。给出方案与依据。

   <details markdown="1"><summary>答案</summary>
   25 个口吻 adapter 走 multi-LoRA（低秩假设成立，01 §6；$$r = 8$$ 够，可加 DoRA，02 §7；服务账 03 §5）。5 个知识需求先用大 $$r$$ + all-linear + 数据回放试，验证集与遗忘都要看（02 §2、§5）；不够就全量微调各出一个模型，单独部署。两类不要用 `linear` 合成（03 §4.2）。
   </details>

9. LoRA 训 GRPO，`beta > 0`。参考模型的 logits 怎么来？与 DPO 比，这里省下的显存与算力各是多少？如果 GRPO 中途把 adapter `merge` 进底座会发生什么？

   <details markdown="1"><summary>答案</summary>
   `disable_adapter()` 下再前向一次（03 §4.3）。省一份模型显存（8B 16 GB）与一份权重读取，不省参考前向的计算——DPO 同理。中途 merge 后，"关掉 adapter"得到的不再是初始策略而是当前策略，KL 项恒为 0，参考模型形同虚设。
   </details>

### C. 面试题

10. 用三句话向一个只知道全量微调的人解释 LoRA：它改了什么、省了什么、没省什么。

    <details markdown="1"><summary>答案</summary>
    改了什么：把每个权重矩阵的更新约束成两个瘦矩阵的积 $$BA$$，只训它们，$$W$$ 冻结。省了什么：可训练参数 50–200 倍、优化器状态与主权重 5–8 倍、checkpoint 从 GB 到 MB、推理时可合并零开销。没省什么：激活值、对输入的反向传播、以及知识注入这类高秩改动的能力。
    </details>

11. 为什么 LoRA 的学习率比全量大一个量级？为什么 $$\alpha / r$$ 这个缩放在大 $$r$$ 时有问题？两者有什么联系？

    <details markdown="1"><summary>答案</summary>
    $$B$$ 从零、$$A$$ 是随机方向，要从无到有长出 $$\Delta W$$，且 $$\Delta W$$ 对 $$B$$ 的更新的响应带着 $$A$$ 的小尺度，同样 lr 落到 $$\Delta W$$ 上比全量小。$$\alpha/r$$ 假设 $$r$$ 个方向的贡献线性相加，实际按 $$\sqrt r$$ 增长，除以 $$r$$ 后大 $$r$$ 的有效更新被压小。联系：scaling 与 lr 是一个旋钮的两面，$$BAx$$ 前乘 2 与 lr 乘 2 在 Adam 下几乎等价。
    </details>

12. 说出 LoRA 家族里五个变体各改了公式的哪个符号、解决什么问题、代价是什么。

    <details markdown="1"><summary>答案</summary>
    rsLoRA：scaling $$\alpha/r \to \alpha/\sqrt r$$，修大 $$r$$ 学不动，无代价。LoRA+：$$B$$ 的 lr $$\times \lambda$$，修 $$A$$、$$B$$ 学习速度不对称，无代价。PiSSA：$$A_0$$、$$B_0$$ 取 $$W$$ 的主奇异方向，收敛快，代价是改底座、发布要转换、遗忘风险。DoRA：$$W' = m \odot (W + BA)/\lVert \cdot \rVert_c$$，小 $$r$$ 补差距，代价是每步构造 $$W + BA$$、慢很多。QLoRA：$$W \to Q_{NF4}(W)$$，冻结权重降到 4 bit，代价是每步反量化、慢 30–50%、底座精度失配要重测。
    </details>

## 六、下一步

- 继续后训练：[后训练系列](/post-training-from-sft-to-verifiable-rewards.html)第二篇起的偏好数据、DPO、GRPO，本系列第三篇的 `disable_adapter()` 会在那里再出现。
- 读实现：[HF 源码第四篇](/peft-and-trl-lora-sft-dpo-grpo-in-source.html)的 `inject_adapter`、`lora.Linear.forward`、`merge`，每一行都对应本系列第一、三篇的一个公式。
- 服务侧：[vLLM 系列第十一篇](/request-shapes-multi-lora-and-multimodal.html)的 multi-LoRA kernel 与调度，本系列第三篇只算了账。
- 量化：[高效推理系列](/efficient-inference-and-compression-for-llms.html)的量化篇讲 GPTQ / AWQ / FP8，本系列第二篇的 NF4 是其中面向训练的一种。

## 七、延伸阅读

- Hu 等 2021，*LoRA: Low-Rank Adaptation of Large Language Models*——原始论文；第 7 节的子空间分析是"$$r$$ 边际收益递减"的来源。
- Aghajanyan 等 2020，*Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*——低秩假设的出处。
- Biderman 等 2024，*LoRA Learns Less and Forgets Less*——全量 $$\Delta W$$ 的谱分析、代码/数学上的差距与遗忘的对照。
- Dettmers 等 2023，*QLoRA: Efficient Finetuning of Quantized LLMs*——NF4、双重量化、分页优化器；附录里"全部线性层比 $$r$$ 重要"的实验。
- Kalajdzievski 2023，*A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA*——rsLoRA。
- Hayou 等 2024，*LoRA+: Efficient Low Rank Adaptation of Large Models*——$$B$$ 的 lr 比例。
- Liu 等 2024，*DoRA: Weight-Decomposed Low-Rank Adaptation*。
- Meng 等 2024，*PiSSA*；Paischer 等 2024，*EVA*；Li 等 2023，*LoftQ*——三种初始化。
- Sheng 等 2023，*S-LoRA: Serving Thousands of Concurrent LoRA Adapters*；Chen 等 2023，*Punica*——multi-LoRA 服务的 kernel 与内存管理。
- `peft` 文档的 *LoRA* 与 *Merging adapters* 两页——`LoraConfig` 每个字段的当前语义，比任何博客都准。
