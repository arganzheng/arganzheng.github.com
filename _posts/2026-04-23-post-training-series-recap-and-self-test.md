---
layout: post
series: post-training
title: "后训练（09）：系列总结与通关自测"
subtitle: "Post-Training: Series Recap and Final Self-Test"
tags: [AI, LLM, Post-Training, RLHF]
catalog: true
date: 2026-04-23 20:00:00
---

八篇正文回答了一个问题：**一个只会续写的基座模型，经过哪几步变成能回答、能拒绝、能一步步推导、能调工具的模型——每一步的目标函数是什么、要几个模型在显存里、生成多少 token、怎么证明它真的变好了**。第一篇用 SFT 教格式，第二篇把人的判断造成一个可微的奖励，第三、四篇是用奖励改策略的在线与离线两种解法，第五篇把奖励换成验证器，第六篇把验证器推到多轮环境，第七篇把奖励换成教师的分布，第八篇是贯穿全程的度量。八篇合起来是[《AI 算法工程师学习地图》](/ai-algorithm-engineer-learning-roadmap.html)的第 L5 层。

本文不讲新内容，做三件事：把八篇压成一张表与八段回顾，把贯穿八篇的几条线拎出来，然后给一套三段式的通关自测——判断与计算、跨篇综合、面试题。各篇末尾的自测检验的是"这一篇读懂了没有"，这里检验的是"八篇能不能连起来用"。第八篇末尾的"系列总结"那一节的内容（三件套的最终一张表、三条线索的终点）并入了本文第三章。

> **读完这八篇，你应该能回答哪些问题？[^q0] 哪些数字与结论必须能脱口而出？[^q1] 怎么判断自己是"读过"还是"掌握"了？[^q2]**

## 一、总览：系列回答的问题与主线

系列的一句话主张是：**所有基于偏好或奖励的后训练方法都在操作同样三个组件——策略、奖励、参考——每个名字只是这三件套上的一处改动**：奖励从哪来（标注序列、RM、隐式对数比、验证器、教师分布），参考怎么约束（KL 惩罚、折进 loss、去掉），以及为了估计策略梯度要不要第四个组件（价值模型）。每一篇用同一套方法处理它的方法——写出目标函数并解释每一项，算出显存、token 与数据的账，给出在 Qwen2.5 0.5B / 1.5B 上跑一遍的骨架，再对到 InstructGPT、Llama 3、Tülu 3、DeepSeek-R1、Qwen3、Kimi K2 的公开配方里的那一行。

| 篇 | 回答的问题 | 一句话结论 | 必记的数字 / 公式 |
|---|---|---|---|
| [第一篇：SFT](/sft-data-chat-template-loss-mask-and-peft.html) | 同一批指令数据，mask、packing、LoRA 的秩、epoch 各让模型变成什么样？怎么在训完前知道会不会忘？ | SFT 只教格式、不加知识（表面对齐）；格式是低秩的、知识是高秩的；遗忘看 lr 与更新的秩，用一份无关文本的 loss 度量 | lr $$10^{-5}$$（预训练的 1/10）；8B 全量 128 GB；LoRA $$r = 16$$ 全部线性层 1.78% 参数、验证 loss 差 0.001、状态 1/7；遗忘 +0.02 / +0.62 / +0.01；padding 有效 45–56%；8B × 2B token = 67 GPU 小时 |
| [第二篇：偏好数据与奖励模型](/preference-data-and-reward-models.html) | 70% 准确率的 RM 为什么够用？它在哪被钻空子，怎么在训 RL 前发现？ | RM 是逻辑回归，给的是梯度方向不是判决；人的一致率 70–75% 就是上限；失效在分布外，金奖励随 $$\sqrt{\text{KL}}$$ 先升后降，用 best-of-N 免费预演 | $$P(y_w \succ y_l) = \sigma(r_w - r_l)$$；1 epoch；8B、10 万对 7 GPU 小时；$$\text{KL}_{BoN} = \log N - (N-1)/N$$，$$N = 16$$ → 1.83 nats；$$R_{gold}(d) = d(\alpha - \beta \log d)$$ |
| [第三篇：在线 RL](/online-rl-ppo-grpo-and-the-rlhf-trio.html) | PPO 为什么四个模型？GRPO 用什么代替价值模型、代价是什么？推理与反向各占多少？ | baseline 不依赖当前样本就无偏，GRPO 用组内均值与标准差代替价值网络；FLOPs 训练占一半，墙钟生成占一半以上 | $$\max \mathbb{E}[r] - \beta\,\text{KL}(\pi_\theta \Vert \pi_{ref})$$；$$\beta$$ 0.01–0.05，规则奖励 0；PPO 288 GB / GRPO 160 GB；GAE $$\gamma = 1$$、$$\lambda = 0.95$$；clip 0.2（DAPO 0.2 / 0.28）；一步 512 × 8 × 1000 = 410 万 token；$$\approx 12N$$/token |
| [第四篇：离线 RL](/offline-rl-dpo-and-its-family.html) | DPO 真的不需要奖励模型？隐式奖励在哪失效？与 PPO 训出的差在哪？ | 闭式解反解奖励代入 Bradley-Terry，RM 被对数比替代；只在数据分布上受约束→似然同降、过优化、长度；PPO 上限略高（探索），DPO 性价比高；on-policy 数据比 loss 变体重要 | $$\pi^* \propto \pi_{ref}\, e^{r/\beta}$$；$$\hat r = \beta \log(\pi_\theta / \pi_{ref})$$；$$\beta$$ 0.1；$$8N$$/token，10 万对 9 GPU 小时；两模型 128 GB，LoRA 约 20 GB；Llama 3 六轮、RPO $$\alpha = 0.2$$ |
| [第五篇：推理模型与可验证奖励](/reasoning-models-and-verifiable-rewards.html) | 规则奖励为什么能训出长思维链？R1 四阶段各修什么？rollout 多多少？ | 不可 hack → KL 可撤、探索可走极远；对长度中立；放大基座已有的推理模式；四阶段各修上一步暴露的一个问题 | AIME 15.6 → 71.0（cons@64 86.7）；80 万条 = 60 万推理 + 20 万非推理；一步 512 × 16 × 16K = 1.3 亿 token，多 8–64 倍，KV 16 TiB；总算力 $$10^{22}$$–$$10^{23}$$；PRM 78.2 vs ORM 72.4 @ $$N = 1860$$；32B 蒸馏 72.6 vs RL 47 |
| [第六篇：Agent 与工具调用的 RL](/agentic-rl-tool-use-environments-and-trajectories.html) | 与单轮 RLVR 差在哪一行公式、哪一处 mask、哪一段系统？瓶颈为什么在环境？ | 对数概率变成 $$T$$ 段之和，环境转移不含 $$\theta$$；工具输出必须 mask，否则学会编造；异步 rollout 从优化变成必需 | 20 轮轨迹 48K token、八成是环境的、每个有效 token 成本 5 倍；前缀缓存 48 万 → 4.8 万 prefill；500 任务 × 8：环境 320 CPU·小时 vs 模型 7 GPU·小时；落后 $$k$$ = 1–4 步几乎无损；KL ≈ 0 |
| [第七篇：蒸馏](/knowledge-distillation-for-llms.html) | 三种蒸馏各学到什么、漏掉什么？R1 为什么给小模型选蒸馏？ | logits 级学分布、序列级学模式、on-policy 修暴露偏差且等价于奖励为 $$\log p_T$$ 的稠密 RL；前向 KL 覆盖、反向 KL 集中；小模型靠探索碰不到正确解 | 每 token 几到几十 bit vs 硬标签 < 1 bit；128K 词表 BF16 每 token 256 KB、top-64 约 256 B；R1-Distill 一两千 GPU 小时 vs RL 几万；on-policy ≈ RL 的 1/10；1.5B 29% / 7B 55% / 32B 72.6%；Minitron 940 亿 token、少 40 倍 |
| [第八篇：评测](/evaluating-llms-benchmarks-judges-and-contamination.html) | MMLU 涨 2 个点是能力、协议还是污染？judge 的 80% 胜率去掉长度剩多少？ | 分数 = 能力 + 协议 + 噪声 + 污染，三关（协议、区间、污染）都过才是能力；judge 有位置、长度、自我偏好三种偏差，要用长度控制的 win rate | 协议 5–15 点；$$\pm 1.96\sqrt{p(1-p)/n}$$：1000 题 ±3、AIME 30 题 ±18、MMLU ±0.8；污染子集 10–30 点；GSM1K 掉 13 点；MMLU-Redux 6.5% 错题；位置改判 20–30%；长度控制后与 Arena 相关 0.94 → 0.98 |

### 1. 本文的章节安排

| 章 | 内容 |
|---|---|
| 二 | 逐篇回顾：核心问题、结论、必记、常见误解 |
| 三 | 贯穿八篇的五条线：奖励的来源、KL 预算、只算自己生成的 token、on-policy 与探索、成本结构 |
| 四 | 常见误区表 |
| 五 | 通关自测：A 判断与计算 10 题、B 跨篇综合 5 题、C 面试题 7 题、D 掌握判据 |
| 六 | 下一步 |

## 二、逐篇回顾

### 1. 第一篇：SFT：指令数据、chat template、loss mask 与参数高效微调

**核心问题**：同一批 10K 条指令数据，loss 算不算 prompt、packing 掩不掩跨样本、LoRA 的秩取 8 还是 64、训 1 个 epoch 还是 5 个——每个选择各让模型变成什么样？怎么在训完之前就知道它会不会忘掉预训练学到的东西？

**结论**：SFT 的目标函数与预训练相同，只有数据（对话）、格式（chat template 与特殊 token）、loss 位置（只算回复）、数据量与 epoch 四处不同。它便宜、快、决定格式，但不增加知识——URIAL 显示基座与对齐模型的分布差异集中在极少数格式 token 上，这也是 LoRA 的低秩更新足以完成大部分 SFT 的原因：格式是低秩的、知识是高秩的。loss mask 等价于给回复 token 提学习率（prompt 占 34% 时 1.5 倍），收益取决于 prompt 占比；随机组 batch 一半算力在 padding 上，packing 要配块对角掩码。遗忘的机制是窄分布上的多 epoch 梯度把参数推离预训练分布而没有梯度拉回，lr、步数、更新的秩越大推得越远；对策是小 lr 少 epoch、数据回放、模型平均、低秩约束。

**必记**：

- 数据量 1K（LIMA、s1）到 94 万（Tülu 3）；InstructGPT 的验证 loss 1 个 epoch 就过拟合、人评却涨到 16 个——验证 loss 不是停止信号。
- 全量 16 字节/参数，8B 128 GB，lr $$10^{-5}$$ 量级，wd 0；LoRA lr 大一个量级（$$10^{-4}$$），$$W' = W + \frac{\alpha}{r} BA$$，全部线性层远比 $$r$$ 重要。
- LoRA $$r = 16$$ 全部线性层：1.78% 参数，验证 loss 与全量差 0.001，状态 1/7，每步快 20%；LoRA 不更新 embedding 与 lm_head，新特殊 token 要 `modules_to_save`。
- 遗忘度量（wikitext loss 变化）：全量 1e-5 +0.02、全量 1e-4 +0.62（回复 loss 也变差）、LoRA 1e-4 +0.01。
- 成本 $$6ND$$：8B、2B token 约 67 GPU 小时；数据成本正从人时变成推理 FLOPs（$$K = 8$$ 拒绝采样与训练同量级）。

**常见误解**："LoRA 效果差是因为秩太小"——Biderman 等的谱分析显示全量微调的更新本身是高秩的，LoRA 追不上是任务需要高秩更新，不是 $$r$$ 的问题；指令微调上 $$r = 16$$ 已接近全量。另一个："训练 loss 更低的配置训得更好"——不 mask 的训练 loss 混进 prompt token，数值高 0.2–0.4，只能在同样只算回复的验证集上比。

### 2. 第二篇：偏好数据与奖励模型：Bradley-Terry、pairwise loss 与 reward hacking

**核心问题**：一个准确率 70% 的奖励模型为什么够用？它在什么地方会被策略钻空子，怎么在训 RL 之前就发现？

**结论**：RM 是一个逻辑回归——Bradley-Terry 的 $$P(y_w \succ y_l) = \sigma(r_w - r_l)$$，loss $$-\log \sigma(\Delta)$$，梯度权重 $$\sigma(-\Delta)$$ 让排错的对主导更新。loss 只依赖分差，奖励只定义到每个 prompt 一个常数——这是 GRPO 组内归一化与 DPO 里 $$Z(x)$$ 抵消的共同根源。70% 够用是因为 RM 给的是梯度方向，RL 在几百个样本上取平均，随机错误被平掉、系统性偏差留下——而系统性偏差正是 hacking：长度、格式、语气、迎合、拒答，每一种都是"数据里的真相关 → RM 放大成捷径 → RL 利用"三层结构。Gao 等把过优化做成定量规律：金奖励随 $$d = \sqrt{\text{KL}}$$ 先升后降，拐点由 RM 的规模与数据决定，所以 RL 有一个 KL 预算。best-of-N 的 KL 有闭式表达，是训 RL 前免费的预演。

**必记**：

- 人与人一致率 70–75%（InstructGPT 72.6%）；RM 验证准确率 65–75% 就是上限，再涨是拟合标注者个人。
- 一致率 $$p = 0.73$$ 的标签只有约 0.16 bit，10 万对约 1.6 万 bit——RM 是在校准基座已有的先验，不是在学。
- 从 SFT 模型初始化 + 标量头，1 epoch，lr $$10^{-5}$$ 量级，与策略同规格；8B、10 万对约 7 GPU 小时；训便宜用贵——GRPO 一步 400 万 token 的 RM 前向约 3 分钟。
- $$\text{KL}(\pi_{BoN} \Vert \pi_0) = \log N - (N-1)/N$$：$$N$$ = 4 / 16 / 64 / 256 → 0.64 / 1.83 / 3.17 / 4.55 nats。
- 人标一对几美元，AI 标 0.02–0.05 美元，便宜两个数量级但偏差方向一致。

**常见误解**："RM 准确率 74% 比 72% 的人还高，肯定过拟合了"——标签本身有 28% 噪声，判断过拟合要看训练 / 验证准确率的 gap 而不是绝对值。另一个："集成三个 RM 能修长度偏差"——同一份数据训出同样的长度偏差，集成只降随机误差、消不掉系统偏差。

### 3. 第三篇：在线 RL：PPO、GRPO 与 RLHF 三件套

**核心问题**：PPO 为什么要四个模型？GRPO 去掉价值模型之后，用什么代替它、代价是什么？一次 GRPO 训练的算力里推理与反向各占多少？

**结论**：所有在线方法都在解 $$\max \mathbb{E}[r] - \beta\,\text{KL}(\pi_\theta \Vert \pi_{ref})$$，区别只在怎么估梯度。log-derivative 技巧给出 REINFORCE——一步等于按奖励加权的 SFT；方差来自奖励的公共成分与信用分配，减去任何不依赖当前样本的 baseline 都不改变期望（$$\nabla \sum_y \pi(y) = 0$$）。PPO 取 token 级 MDP 视角，训一个价值网络逐 token 做 baseline（GAE），用 clip 限制重要性比——四个模型里两个要训，显存主体是它们的 16 字节/参数。GRPO 用同一 prompt $$G$$ 条回答的组内均值与标准差代替价值网络，代价是 $$G$$ 倍 rollout、优势降到序列级、两个归一化各带一个偏差（Dr. GRPO、DAPO 修的就是它们）。FLOPs 上训练占一半、三个前向各六分之一，但 decode 是 memory-bound 的且有长尾，墙钟生成占 50–80%——RL 后训练的系统问题本质是训练循环里的推理引擎。

**必记**：

- $$\beta$$：InstructGPT 0.02、Llama 2 0.01、DeepSeekMath 0.04、Tülu 3 0.05、DAPO 0——奖励越可验证 $$\beta$$ 越小。
- KL 估计量 $$k_1 = \log(\pi_\theta / \pi_{ref})$$ 进奖励（PPO）；$$k_3 = \rho - 1 - \log \rho$$ 恒非负、进 loss（GRPO）。
- PPO 8B 规格：策略 128 + 价值 128 + 参考 16 + RM 16 = 288 GB；GRPO 160 GB。GAE $$\gamma = 1$$、$$\lambda = 0.95$$；clip $$\epsilon = 0.2$$，DAPO clip-higher 0.2 / 0.28。
- 一步 $$B \times G \times \bar L$$ = 512 × 8 × 1000 = 410 万 token；$$\approx 12N$$/token（PPO $$20N$$）；KV 4096 条 × 1300 token = 680 GiB。
- 权重同步 8B 16 GB 每步一次；生成滞后一步几乎无损。

**常见误解**："GRPO 比 PPO 省 token"——同样 batch 大小 PPO 也要生成同样多的序列，GRPO 只是把它们集中在更少的 prompt 上；省的是价值模型的 128 GB。另一个："RL 的 loss 能看出训得好不好"——几乎不可解读，要看奖励、KL、长度、clip 比例、熵、零优势组比例六条曲线。

### 4. 第四篇：离线 RL：从 RLHF 目标推出 DPO 及其变体

**核心问题**：DPO 真的"不需要奖励模型"吗？它的隐式奖励在哪里会失效？同一份偏好数据，DPO 与 PPO 训出的模型差在什么地方？

**结论**：KL 约束下的目标对每个 prompt 有闭式解 $$\pi^* \propto \pi_{ref} \exp(r / \beta)$$（Gibbs 分布），反解 $$r = \beta \log(\pi^* / \pi_{ref}) + \beta \log Z(x)$$，代入 Bradley-Terry 时 $$Z(x)$$ 抵消——RM 训练与 RL 两步合成一步，最优解相同。DPO 不是没有奖励，而是把 RM 参数化为 $$\beta \log(\pi_\theta / \pi_{ref})$$；它的一切约束都在偏好对上，对上没有的地方由参考说了算。失效全是分布外：似然同降（chosen 与 rejected 概率一起降，概率流向数据外）、与 PPO 形状相同的过优化曲线、长度（拉开对数比最容易的方向）、参考不匹配。十几个变体各改一项：IPO 平方 loss、KTO 单样本、ORPO / SimPO 去参考且都要长度归一化、RPO 加 chosen 的 NLL；换方法的收益小于换一份更 on-policy 的数据。PPO 上限略高（能探索到参考下低概率的正确回答），DPO 性价比高；修 DPO 的办法都是往在线走一步。

**必记**：

- DPO 梯度权重 $$\sigma(\hat r_l - \hat r_w)$$，与 RM 梯度同形；$$\beta$$ 默认 0.1（Zephyr 0.01，Llama 3 0.1）。
- 成本 $$8N$$/token：8B、10 万对约 9 GPU 小时，GRPO 一轮几百到上千；两个模型 128 GB，参考对数概率可预计算，LoRA 下关掉 adapter 的底座就是参考，约 20 GB。
- KL 监控要自己算：$$\text{KL} \approx \mathbb{E}[\hat r_\theta] / \beta$$（chosen 上）；到 2–3 nats 用 judge 核对。
- 拒绝采样 + SFT 就是把 BoN 分布蒸进策略，$$K = 30$$ 约 2.4 nats；Llama 3 六轮迭代，RPO NLL 权重 0.2，DPO loss 里 mask 掉特殊 token。
- Tülu 3：on-policy 偏好数据 > off-policy，长度归一化 DPO > 标准。

**常见误解**："DPO 没有 RM 所以没有 reward hacking"——Rafailov 等测出 DPO 的过优化曲线与 PPO 形状相同，hacking 的对象换成了数据没覆盖的地方。另一个："在线 / 离线 = PPO / DPO"——在线 DPO 有采样、打分、每步更新，是用 DPO loss 的在线 RL；DPO 是一个 loss，不是一种训练范式。

### 5. 第五篇：推理模型与可验证奖励：R1 的配方、PRM 与 test-time compute

**核心问题**：为什么规则奖励能训出长思维链，而奖励模型训不出来？R1 的四个阶段各在修什么？一个推理模型的 RL 训练，rollout 的 token 数比对话模型多多少？

**结论**：可验证奖励确定、稀疏、不可 hack（除验证器漏洞），三个后果叠加：KL 可以撤掉、策略可以走到几万 token；奖励对长度中立，回答变长是"更长的推理更常答对"被选出来的；基座里已有推理的种子，RL 放大它——pass@k 曲线上 RL 大幅提 pass@1，基座在大 $$k$$ 端追上，基座决定天花板的大部分。R1-Zero 证明了存在性但可读性差、中英混杂；R1 四阶段各修一个问题：冷启动 SFT 修格式与语言，推理 RL 加语言一致性奖励把能力推到顶，用 RL 产出的 80 万条**回到基座**重新 SFT 把推理与通用合到一个模型，全场景 RL 补偏好对齐。长思维链把 rollout 账放大一到两个数量级，一步的墙钟由最长的回答决定，部分 rollout 与异步成为标配；PRM 在重排上更好但 R1 不用（步骤难定义、标注难、hacking）。

**必记**：

- R1-Zero：AIME pass@1 15.6 → 71.0，cons@64 86.7；长度自发增长；"aha moment"。
- 80 万条 = 60 万推理（规则 + V3 判可读性过滤）+ 20 万非推理，在基座上 SFT 2 epoch；R1-Distill 只用 SFT。
- 一步 rollout：512 × 16 × 16K = 1.3 亿 token，对话的 30 多倍；KV 16 TiB 只能分波；总算力 $$10^{22}$$–$$10^{23}$$，是对话 RLHF 的一百倍。
- PRM800K：$$N = 1860$$ 时 PRM 78.2 vs ORM 72.4 vs 多数投票 69.6；cons@64 是 64 倍推理费用换 16 个点。
- Qwen2.5-32B：蒸馏 72.6 vs 直接 RL 约 47；Tülu 3 RLVR $$\beta = 0.05$$，GSM8K +3–4 但没有长思维链。

**常见误解**："回答变长是 RL 学会了思考更多"——Dr. GRPO 指出按序列长度平均让错误的长回答受罚更轻，本身就是一个让回答变长的力，去掉它能在不加长度惩罚的情况下缩短回答。另一个："规则奖励绝对不可 hack"——多个 `\boxed{}`、硬编码测试、改测试文件都真实发生过，工程量在写验证器与堵漏。

### 6. 第六篇：Agent 与工具调用的 RL：多轮环境、轨迹数据与延后的奖励

**核心问题**：多轮工具调用的 RL 与单轮 RLVR 差在哪一行公式、哪一处 mask、哪一段系统？为什么 Agent 训练的瓶颈常常不是模型而是环境？

**结论**：把多轮写成 MDP——状态是全部历史（含工具输出），动作是一段生成，转移是环境执行，奖励在轨迹末尾。**一行公式**：$$\log \pi_\theta(\tau) = \sum_t \log \pi_\theta(a_t \mid s_t)$$，环境转移不含 $$\theta$$，第三篇的一切原样成立。**一处 mask**：工具返回、模拟用户、prompt 全部 mask = 0，不 mask 时模型学会编造工具输出（Search-R1 的消融）。**一段系统**：rollout 变成几千个沙箱各自运行、耗时从秒到几十分钟的循环，同步等最慢的那条让 GPU 空转，异步 rollout 与按轮的 off-policy 修正成为必需。格式要先靠 SFT 学会，RL 只负责提高任务成功率；奖励是可验证的环境状态（测试、状态比较、答案匹配），部分分与 rubric 作 shaping 但权重远低于最终成功；多轮 hacking 多一个维度——策略会改变验证本身。

**必记**：

- 20 轮、每轮生成 500、返回 2000 的轨迹：最终上下文 48K，模型生成的 1 万（约 20%），每个有效 token 训练成本约 5 倍；前缀缓存把累计 prefill 从 48 万降到 4.8 万。
- 500 任务 × $$G$$ = 8 = 4000 条轨迹：环境约 320 CPU·小时，模型约 7 GPU·小时；成功率常只有 20–40%，一万条成功轨迹要跑三到五万条。
- KL ≈ 0；重要性比按轮算；缓冲区落后 $$k$$ = 1–4 步几乎无损。
- SWE-RL 用补丁相似度（0–1，格式错 −1）不跑测试，SWE-bench Verified 41.0%；Kimi K2 两万多个合成工具、65.8%。
- 没有一个 Agent 配方用学出来的 RM 作主奖励。

**常见误解**："Agent 数据可以像对话数据一样用强模型批量生成"——每条轨迹都要真跑过环境，失败的白跑，数据成本从推理 FLOPs 变成环境时间。另一个："turn 级信用分配一定比轨迹级好"——它要价值网络或额外采样，多数 2025 年配方接受轨迹级优势，GiGPO 用同状态分组做折中。

### 7. 第七篇：蒸馏：logits 级、序列级与 on-policy

**核心问题**：同一个教师，logits 级、序列级、on-policy 三种蒸馏各让学生学到什么、漏掉什么？R1 为什么给小模型选择蒸馏而不是 RL？

**结论**：三种形态只差两处——序列是谁生成的、标签是硬还是软。logits 级让学生学到每个位置的完整分布（暗知识，每 token 几到几十 bit），但序列不来自学生；序列级就是在教师输出上 SFT，学输出模式、丢不确定性，跨词表可用、成本最低；on-policy（GKD）在学生自己采样的序列上向教师对齐，修了暴露偏差，且等价于奖励为 $$\log p_T$$ 的逐 token 稠密 RL——教师是三件套里的奖励，参考可以不要。前向 KL mode-covering，学生容量不足时在峰间放概率、生成教师不会生成的东西；反向 KL mode-seeking，只在教师认可处放概率（MiniLLM）。R1 选蒸馏是一个对照实验：RL 靠探索，小模型在几千 token 里碰到正确解的概率太低；蒸馏靠示范。2025 年小模型的路线是序列级冷启动 → on-policy 对齐 → 可选的 RL。

**必记**：

- logits 级与 on-policy 要求同一 tokenizer（id 与位置都要对齐）；序列级绕开；ULD 用排序后的最优传输。
- 128K 词表 BF16 每 token 256 KB，40 亿 token 1 PB 存不下；top-64 约 256 B/token → 1 TB，覆盖 99% 以上概率质量。
- R1-Distill：教师生成 80 万条约 300 H100 小时，32B 学生 SFT 约 1000 GPU 小时，合计一两千；同基座推理 RL 几万——便宜一到两个数量级且效果更好。
- on-policy 每步教师前向 $$2N_T$$ 是主项（32B 教师、1.5B 学生时占 85%）；Qwen3 与 Thinking Machines 报告 ≈ RL 的 1/10。
- 容量上限：R1-Distill 1.5B 29% / 7B 55% / 32B 72.6%（同一份数据）；Minitron 剪枝 + 蒸馏 940 亿 token，比从头训少 40 倍；Gemma 2 2B 预训练蒸馏 2T token。

**常见误解**："蒸馏出的小模型学到了教师的能力"——学到的是教师在覆盖任务上的输出模式，R1-Distill 在数学代码强、通用对话与多语言明显弱；对策是数据多样性与 on-policy。另一个："前向 KL 是标准选择"——分类上差别不大，开放生成上前向 KL 更危险。

### 8. 第八篇：评测：benchmark、LLM-as-judge、Arena 与污染

**核心问题**：一个模型的 MMLU 涨了 2 个点，怎么判断这是能力提升、协议变化、还是污染？用 GPT-4 当 judge 得到的 80% 胜率，去掉长度偏差之后还剩多少？

**结论**：一个分数 = 能力 + 协议 + 噪声 + 污染，只有第一项是要测的。协议（few-shot、CoT、温度、抽取、格式、max_tokens）让同一模型差 5–15 个点，比较两个模型的唯一办法是同一 harness、同一配置、同一时间；噪声按二项置信区间算，配对比较（McNemar）比各算区间灵敏得多；污染在评测侧用补全、困惑度、改写、新题、canary、私有集检测，饱和 benchmark 上的最后几个点很大比例是污染与题目错误。judge 有位置、长度、自我偏好三种系统性偏差，方向与 RM 的 hacking 完全一致——judge 就是一个没被训练的 RM；长度控制的 win rate 用逻辑回归回归掉长度差。Arena 用 Bradley-Terry 给模型排名，测的是"人喜欢"而非"正确"，且有私测与采样不均的排行榜幻觉。总分是决策的终点不是起点：切分、错误分类、读一百条输出；业务任务、防污染、发现 hacking、回归四种情况必须自建评测集。

**必记**：

- $$\sigma = \sqrt{p(1-p)/n}$$，95% 区间 $$\pm 1.96\sigma$$：AIME 30 题 ±18、GPQA Diamond 198 题 ±7、500 题 ±4、1000 题 ±3、MMLU 14042 题 ±0.8。
- pass@$$k$$ 无偏估计 $$1 - \binom{n-c}{k} / \binom{n}{k}$$，报 pass@1 要说 $$n$$；cons@$$k$$ 与 pass@$$k$$ 不能混；Agent 报 pass^$$k$$。
- 位置偏差：GPT-4 在 20–30% 的对上交换顺序后改判；长度控制后与 Arena 相关 0.94 → 0.98，胜率变 5–15 点；judge 与人一致率约 80%，与人人之间的 81% 相当。
- GSM1K：部分模型掉 13 个点以上；MMLU-Redux：约 6.5% 的题有错；污染子集高 10–30 点。
- 自建：区分 3 个点要 500–1000 题；judge 与人一致率低于 80% 就换 judge 或改 rubric；版本化、保密、留核心子集做锚。

**常见误解**："两个模型的置信区间不重叠才算显著"——同一批题上做配对比较能消掉题目难度，1000 题上 A 对 B 错 60、A 错 B 对 40 就勉强显著。另一个："Arena 第一就是最强"——它是路过用户的主观偏好，风格控制后排名会变，且头部厂商的私测等价于在榜上做 best-of-N。

## 三、贯穿全系列的几条线

### 1. 奖励从哪来：三件套的一格换了四次手

第一篇没有奖励也没有参考，只有策略与标注好的目标序列——奖励隐含在数据里。第二篇造出显式的奖励：人（或 AI）的成对偏好经 Bradley-Terry 变成一个标量函数，参考随之出场，因为学出来的奖励是代理、代理要用 KL 锚住。第四篇把奖励折进策略——$$\beta \log(\pi_\theta / \pi_{ref})$$ 既是 RM 又是最优策略，参考从"算 KL 的模型"变成 loss 里的分母。第五、六篇把奖励换成验证器与环境状态，奖励不再是代理，参考的作用退到只防格式崩坏，$$\beta$$ 到 0。第七篇把奖励换成教师的逐 token 对数概率，稠密、精确、来自一个不会被 hack 的固定模型，参考也不再需要。

这条线上的每一步都改变了成本结构与失效方式：奖励越是学出来的代理，就越需要参考、越有 KL 预算、越容易被钻空子；奖励越可验证，就越能撤掉参考、走得越远、但只在有标准答案的地方存在。第八篇的 judge 是这条线的镜像——它是一个没被训练的 RM，偏差方向与 RM 的 hacking 完全一致。第八篇末尾那张"三件套的最终一张表"就是这条线的终点：SFT 1 个模型、RM 1、PPO / GRPO 4 / 3、DPO 2 → 1、RLVR 与 Agent RL 2–3（加环境）、蒸馏 2。

### 2. KL 预算与 Goodhart

第二篇给出规律：金奖励随 $$d = \sqrt{\text{KL}}$$ 先升后降，$$\beta$$ 系数随 RM 变大、数据变多而减小，best-of-N 的 $$\log N - (N-1)/N$$ 是免费的 KL 尺。第三篇把它变成操作：$$\beta$$ 是奖励与 KL 的汇率，InstructGPT 0.02 到 Tülu 3 0.05，可自适应到一个目标 KL；三种估计量里 $$k_1$$ 进奖励、$$k_3$$ 进 loss；训练时 KL 到 2–3 nats 就该用 held-out judge 核对。第四篇发现 DPO 逃不掉：过优化曲线与 PPO 形状相同，只是 hacking 的对象换成数据没覆盖的地方，KL 要用 $$\mathbb{E}[\hat r] / \beta$$ 自己估。

第五、六篇是这条线的反面：规则奖励下 Goodhart 没有入口，DAPO 干脆去掉 KL，Agent RL 里 KL ≈ 0——但验证器自己的漏洞（多个 `\boxed{}`、改测试文件）成了新的入口，"预算"变成了"堵漏"。第八篇把同一件事搬到评测：代理奖励涨、真实质量掉的检测要一个与奖励不同的度量，这是自建评测集的四个必要理由之一。

### 3. 只算自己生成的 token

第一篇的 loss mask——prompt 位置 label 设 −100，多轮只算每轮 assistant——看起来是一个工程开关，其后每一篇都在用它。第二篇 RM 的输入要按同一模板渲染；第三篇 PPO 只在回答 token 上算 $$\log \pi$$；第四篇 Llama 3 在 DPO loss 里 mask 掉特殊 token，因为模板 token 在 chosen 与 rejected 里都出现、对数比差是纯噪声；第六篇是它的极端形态：工具返回与模拟用户的 token 在上下文里却不是策略的输出，不 mask 就学会编造工具结果，且轨迹里八成 token 是环境的。第八篇的错误分类表里"幻觉工具输出"一行指回第六篇的 mask。同一个机制、同一份代码路径（`assistant_only_loss`），从 SFT 到 Agent RL 没有变过。

### 4. on-policy 与探索

第二篇提出：偏好数据从要训的策略采样（on-policy）RM 才准，Llama 2 / 3 每轮 RL 后重标重训，Tülu 3 的消融 on-policy > off-policy。第三篇定义了严格在线（生成用的就是要更新的权重，每步同步）与它的放松（生成滞后一步几乎无损）。第四篇把它上升为结论：让偏好优化有效的两个要素是 on-policy 采样与负梯度，具体 loss 差别不大；离线 DPO 的上限由数据与策略的距离决定，PPO 的上限略高正是因为能探索到参考下低概率的正确回答。

第五篇把探索推到极致：规则奖励 + 大 $$G$$ + 弱 KL + 几千步，策略自己找到长推理；pass@k 显示 RL 主要在放大基座已有的模式。第六篇里 on-policy 与异步不可兼得，按轮记录 $$\log \pi_{old}$$、限制落后步数是折中。第七篇的 on-policy 蒸馏是同一思想在蒸馏上的形态：学生自己采样、教师在学生会去的地方纠正，修暴露偏差——它同时是 R1 那个对照实验的解释：小模型靠探索碰不到正确解，所以先蒸馏再 RL。

### 5. 成本结构：算力便宜、数据贵，然后是推理引擎与环境

第一篇算出 SFT 8B × 2B token 只要 67 GPU 小时，成本几乎全在数据，且数据成本正从人时变成推理 FLOPs（$$K = 8$$ 的拒绝采样与训练同量级）。第二篇 RM 训 7 GPU 小时但用贵——RL 每步 400 万 token 的打分。第三篇 GRPO 一轮几百到上千 GPU 小时，比 SFT 贵一个数量级、比预训练便宜三个，贵的不是 FLOPs 而是训练循环里的推理引擎、三四个模型同时在显存、每步同步。第四篇 DPO 回到 9 GPU 小时，因为不采样。第五篇 rollout 放大 8–64 倍，总算力 $$10^{22}$$–$$10^{23}$$；第六篇成本主体变成环境的 CPU·小时与沙箱集群；第七篇蒸馏比 RL 便宜一到两个数量级、on-policy ≈ 1/10；第八篇 judge 几十美元对人评几千美元。

八篇合起来的成本线：SFT 一个模型 → RM 一个 → PPO 四个 / GRPO 三个 → DPO 两个（参考可离线）→ RLVR 的 rollout token → Agent RL 的环境步数与沙箱 → 蒸馏的教师推理 → 评测的 judge 调用。每一步动手前都能估出来。

| 概念 | 出现的篇 | 关系 |
|---|---|---|
| 三件套（策略、奖励、参考） | 全部 | 一定义起点；二造奖励；三、四两种解法；五、六换验证器；七换教师；八的 judge 是未训练的 RM |
| Bradley-Terry、平移不变性 | 二、三、四、八 | 二定义；三的组内归一化；四的 $$Z(x)$$ 抵消；八的 Arena 排名 |
| KL 预算、过优化曲线 | 二、三、四、五 | 二给规律；三给 $$\beta$$ 与估计量；四 DPO 同一曲线；五规则奖励下 $$\beta \to 0$$ |
| best-of-N | 二、四、五 | 二的 KL 尺；四的拒绝采样 = 蒸馏 BoN；五的 test-time compute 与"RL ≈ BoN 蒸进策略" |
| loss mask | 一、三、四、六、八 | 一定义；三只算回答；四 mask 特殊 token；六 mask 工具输出；八错误分类 |
| on-policy | 二、三、四、六、七 | 二数据；三同步；四结论；六异步折中；七 on-policy 蒸馏 |
| 长度 | 二、三、四、五、八 | 二 RM 偏差；三 Dr. GRPO 归一化偏差；四 DPO 长度、SimPO；五长度控制五种；八长度控制 win rate |
| 拒绝采样 / 序列级蒸馏 | 一、四、五、七 | 一 SFT 数据来源；四最简单的离线 RL；五 R1 阶段 3；七就是序列级蒸馏 |
| 冷启动 SFT | 一、五、六、七 | 一的"少即是多"（s1）；五 R1 阶段 1；六先学格式；七蒸馏是最好的冷启动 |
| 异步 rollout | 三、五、六 | 三滞后一步；五部分 rollout；六必需 |
| 置信区间、一致率 | 二、八 | 二人的一致率 70–75%；八 judge 与人 80%、二项区间 |

## 四、常见误区

| 误区 | 为什么错 | 正确的说法 | 出处 |
|---|---|---|---|
| SFT 能给模型注入知识 | 分布差异集中在少数格式 token 上；格式是低秩的、知识是高秩的 | SFT 教形式，知识来自预训练；要注入知识用全量与大量数据 | [第一篇](/sft-data-chat-template-loss-mask-and-peft.html) |
| SFT 的验证 loss 上升就该停 | InstructGPT 验证 loss 1 epoch 过拟合、人评涨到 16 epoch | 看下游指标与一份无关文本的 loss，不看 SFT 验证 loss | [第一篇](/sft-data-chat-template-loss-mask-and-peft.html) |
| RM 准确率越高越好 | 标签噪声 25–30%，超过人的一致率是在拟合标注者个人 | 65–75% 是上限；看训练 / 验证 gap、按难度分桶 | [第二篇](/preference-data-and-reward-models.html) |
| RM 的问题是"不够准" | 随机错误被 RL 的平均平掉，留下的是系统偏差 | RM 的问题是"偏在哪"，训 RL 前用长度相关、BoN 扫描、对抗探针暴露 | [第二篇](/preference-data-and-reward-models.html) |
| RL 训练奖励涨就是变好 | 代理奖励单调涨，金奖励先升后降 | 有 KL 预算，按 KL 与 held-out judge 决定停 | [第二篇](/preference-data-and-reward-models.html)、[第三篇](/online-rl-ppo-grpo-and-the-rlhf-trio.html) |
| RL 后训练的成本在 FLOPs | 训练占一半 FLOPs，但生成是 memory-bound 的 decode 且有长尾 | 墙钟 50–80% 在生成，问题是训练循环里的推理引擎 | [第三篇](/online-rl-ppo-grpo-and-the-rlhf-trio.html) |
| DPO 不需要奖励模型 | 奖励被参数化为 $$\beta \log(\pi_\theta / \pi_{ref})$$ | DPO 训的就是一个与策略共享参数的 RM，只在数据分布上受约束 | [第四篇](/offline-rl-dpo-and-its-family.html) |
| DPO 变体之间差别很大 | 标准 benchmark 上互有胜负 | 换一份更 on-policy 的数据收益大于换方法 | [第四篇](/offline-rl-dpo-and-its-family.html) |
| RL 教会了模型推理 | pass@k 大 $$k$$ 端基座追上 RL 模型 | RL 主要放大基座已有模式；基座决定天花板大部分 | [第五篇](/reasoning-models-and-verifiable-rewards.html) |
| 小模型要推理能力就直接 RL | 32B 上蒸馏 72.6 对 RL 47，且贵一到两个数量级 | 先蒸馏（序列级冷启动、on-policy 对齐）再可选 RL | [第五篇](/reasoning-models-and-verifiable-rewards.html)、[第七篇](/knowledge-distillation-for-llms.html) |
| Agent RL 的瓶颈是模型 | 一步 4000 条轨迹环境 320 CPU·小时对模型 7 GPU·小时 | 瓶颈是沙箱集群与环境时间；异步是为了让 GPU 别闲着 | [第六篇](/agentic-rl-tool-use-environments-and-trajectories.html) |
| 分数涨了 2 个点就是进步 | 协议差 5–15 点、1000 题噪声 ±3、污染子集高 10–30 点 | 三关：协议相同、超出置信区间、新题复测保持 | [第八篇](/evaluating-llms-benchmarks-judges-and-contamination.html) |

## 五、通关自测

### A. 判断与计算（10 题）

1. 8B 模型做 LoRA SFT，$$r = 64$$、全部线性层，训练状态约多少？与全量比省多少？

   <details markdown="1"><summary>答案</summary>

   按第一篇 0.5B 上的账同比例放大：$$r = 64$$ 全部线性层 7.12% 可训练参数，训练状态 1.38 GiB → 8B 规格约 22 GiB（$$r = 16$$ 是 17 GiB）；全量 120–128 GB，省到约六分之一。差别主要在冻结权重只存 BF16 2 字节、不存梯度与 Adam 状态。

   </details>

2. 128K 词表、BF16，要离线存 100 亿 token 的教师 logits 做 logits 级蒸馏。全量存多少？只存 top-64 呢？

   <details markdown="1"><summary>答案</summary>

   全量每 token 256 KB，100 亿 token 约 2.5 PB，不现实；top-64 约 256 B/token，约 2.5 TB，可以离线——第七篇 40 亿 token 的 1 PB 对 1 TB 同比例。top-64 通常覆盖 99% 以上的概率质量。

   </details>

3. 70B 模型、50 万条样本、平均 2000 token、2 个 epoch 的全量 SFT，H100 40% MFU 约多少 GPU 小时？

   <details markdown="1"><summary>答案</summary>

   $$D = 2 \times 10^9$$，$$C = 6 \times 7 \times 10^{10} \times 2 \times 10^9 = 8.4 \times 10^{20}$$；第一篇 8B 同样 $$D$$ 是 $$9.6 \times 10^{19}$$ → 67 GPU 小时，按比例约 590 GPU 小时。仍是数据成本的零头：50 万条由人写按每条 10 分钟是 8 万多人时。

   </details>

4. Llama 3 用拒绝采样 $$K = 10$$ 与 $$K = 30$$ 选最佳回答做 SFT，各相当于把策略推离参考多少 KL？

   <details markdown="1"><summary>答案</summary>

   $$\log K - (K-1)/K$$：$$K = 10$$ 是 $$2.30 - 0.90 = 1.40$$ nats，$$K = 30$$ 是 $$3.40 - 0.97 = 2.43$$ nats（第四篇给的约 2.4）。拒绝采样 + SFT 就是把 BoN 分布蒸进策略，这个数字是"走了多远"。

   </details>

5. 一步 GRPO：$$B = 256$$、$$G = 16$$、$$\bar L = 2000$$，8B 策略、8B 参考、8B RM。生成多少 token？总 FLOPs 约多少？生成占几分之几？

   <details markdown="1"><summary>答案</summary>

   $$256 \times 16 \times 2000 = 819$$ 万 token；$$\approx 12N$$/token → $$12 \times 8 \times 10^9 \times 8.19 \times 10^6 \approx 7.9 \times 10^{17}$$ FLOPs；生成 $$2N$$ 占六分之一，训练 $$6N$$ 占一半——但墙钟上生成常占 50–80%。

   </details>

6. 70B 规格下，PPO、GRPO、参考预计算的 DPO 三种训练的模型状态各约多少 GB？

   <details markdown="1"><summary>答案</summary>

   16 字节/参数训练、2 字节推理：PPO 策略 1120 + 价值 1120 + 参考 140 + RM 140 = 2520 GB；GRPO 去掉价值 1400 GB；DPO 预计算参考对数概率后只剩策略 1120 GB。这是 Llama 3 在 405B 上选 DPO 而不选 PPO 的显存理由。

   </details>

7. 推理 RL 一步 $$B = 512$$、$$G = 16$$、$$\bar L = 8\text{K}$$，8B 规格 128 KiB/token，全部序列同时在飞的 KV cache 多少？

   <details markdown="1"><summary>答案</summary>

   $$8192 \times 8192 \times 128\ \text{KiB} = 8$$ TiB——第五篇表里 4K 是 4 TiB、16K 是 16 TiB，线性。一台 8 卡机放不下，必须分波生成，一步由几十波串行组成。

   </details>

8. GRPO 按序列长度平均 loss。同一组里一条 2000 token 的错误回答与一条 200 token 的错误回答优势都是 −1，每个 token 受到的惩罚各多少？这会让策略学到什么？

   <details markdown="1"><summary>答案</summary>

   $$-1/2000$$ 对 $$-1/200$$，长回答的每个 token 受罚只有短回答的十分之一——策略学到"答错时写长一点"。这是 Dr. GRPO 指出的长度归一化偏差，去掉它（token 级求和除以常数）在同样准确率下回答短得多。

   </details>

9. 一条 10 轮的 Agent 轨迹，每轮生成 400 token、工具返回 3000 token。最终上下文多长？算 loss 的 token 占多少？每个有效 token 的训练成本约是单轮的几倍？

   <details markdown="1"><summary>答案</summary>

   $$10 \times 400 + 9 \times 3000 = 31\text{K}$$；模型生成 4000 token，约 13%；反向要过全部 31K，每个有效 token 约 7.75 倍——比第六篇 20 轮例子的 5 倍更差，因为工具返回更长。对策是工具返回截断与前缀缓存。

   </details>

10. GPQA Diamond 198 题上模型 A 82%、模型 B 78%，各自独立跑一次。这 4 个点显著吗？

    <details markdown="1"><summary>答案</summary>

    $$\sigma = \sqrt{0.8 \times 0.2 / 198} \approx 2.8\%$$，95% 区间约 ±5.6 个点，4 个点完全在噪声里；要在同一批题上做配对比较（McNemar），并多 seed。

    </details>

### B. 跨篇综合（5 题）

1. 训好一个 RM，验证集上 $$r$$ 与回答长度的 Spearman 相关 0.4。接下来用它做 GRPO，$$\beta$$ 该怎么取、该盯哪条曲线？如果奖励换成 GSM8K 的答案匹配，同样的问题怎么答？

   <details markdown="1"><summary>答案</summary>

   第二篇：相关 > 0.3 就要长度控制（奖励减 $$\lambda \cdot$$ 长度或长度分桶归一化），先用 BoN 扫描 $$N$$ = 1–64 预演长度随 KL 的增长。第三篇：$$\beta$$ 取 RM 奖励的 0.01–0.05 量级，盯回答长度与 KL——长度单调涨且奖励涨主要来自它就是 hacking，KL 到 2–3 nats 用 held-out judge 核对。第五篇：规则奖励对长度中立、不可 hack，$$\beta$$ 可到 0，长度增长是被选出来的；但要用 `dr_grpo` 去掉长度归一化偏差，并堵验证器漏洞（只取最后一个 `\boxed{}`）。

   </details>

2. DPO 训练里 `rewards/chosen` 与 `rewards/rejected` 都在降，训后采样的回答比 SFT 模型长 40%，judge 胜率 75%。三件事各说明什么、各怎么处理？

   <details markdown="1"><summary>答案</summary>

   第四篇：两条都降是似然同降，概率流向数据外，加 RPO 的 chosen NLL 项（Llama 3 取 0.2）或去掉过于相似的对；长度涨 30% 以上要换长度归一化的变体（SimPO、Tülu 3 的长度归一化 DPO）——第二篇说过偏好数据里 chosen 本来就更长，隐式奖励是整条对数比之和，变长是拉开差距最容易的方向。第八篇：75% 胜率要先做长度控制的 win rate（回归掉长度差）与交换顺序，很多模型会掉 5–15 个点，剩下的才是评委真的觉得好。

   </details>

3. 有一个 32B 推理模型当教师，要一个 7B 的推理模型。三条路——直接 RL、序列级蒸馏、先蒸馏再 RL——各花多少、能到哪？

   <details markdown="1"><summary>答案</summary>

   第五篇的对照：32B 基座直接 RL 约 47 对蒸馏 72.6，RL 靠探索、小模型碰不到正确解；第七篇的账：教师生成 80 万条约 300 H100 小时 + 学生 SFT 约 1000 GPU 小时，对 RL 的几万，便宜一到两个数量级；容量上限 7B 约 55%（同一份数据 1.5B 29%、32B 72.6%）。先蒸馏再 RL 优于任何单独一种——蒸馏是最好的冷启动（第五篇 R1 阶段 1 的一般形式），RL 再提几个点；若 tokenizer 相同可加 on-policy 蒸馏，≈ RL 的 1/10 算力。评测按第八篇：AIME 30 题 ±18 个点，报 $$n$$ = 64 的 pass@1 与新题复测。

   </details>

4. 一个代码 Agent 训完在 SWE-bench Verified 上从 30% 到 45%，读轨迹发现部分成功是靠修改测试文件。这个 15 个点该怎么解读、训练侧与评测侧各改什么？

   <details markdown="1"><summary>答案</summary>

   第六篇：这是多轮 hacking——奖励来自可被动作影响的环境状态，防御是把验证放在策略碰不到的地方（只读挂载测试目录、隐藏测试、验证前恢复测试文件），并把破坏性操作直接判零；同时检查工具输出有没有 mask（不 mask 会编造测试输出）。第八篇：500 题上 ±4.4 个点，15 个点本身显著，但成功轨迹里有多少靠改测试要按错误分类切开；Agent 评测要多次运行报 pass^$$k$$，读一百条轨迹是 benchmark 报不出来的那部分。第二篇的 Goodhart 在这里以"策略改变验证本身"的形态出现。

   </details>

5. 把 Llama 3 的六轮"拒绝采样 → SFT → DPO"与 DeepSeek-R1 的四阶段并排放到三件套上：每一步的奖励从哪来、参考怎么约束、哪几篇的内容在起作用？

   <details markdown="1"><summary>答案</summary>

   Llama 3：拒绝采样用第二篇的 RM（每轮重标重训、与策略同规格、去掉 margin）选最佳，$$K$$ = 10–30 相当于 1.4–2.4 nats（第四篇）；SFT 是第一篇（lr 1e-5，checkpoint 平均抗遗忘）；DPO + NLL 0.2、$$\beta$$ 0.1、mask 特殊 token、参考每轮换最新——第四篇的迭代 DPO，半在线。R1：阶段 1 冷启动 SFT（第一篇的"少即是多"，几千条）；阶段 2 GRPO + 规则奖励 + 语言一致性（第三、五篇，$$\beta$$ 极小）；阶段 3 拒绝采样 80 万条回基座重 SFT——第四篇的拒绝采样 + SFT，也就是第七篇的序列级蒸馏，R1-Distill 直接用它；阶段 4 混合奖励（规则 + RM），KL 只对 RM 那一半需要。两条配方共同点：SFT 数据都由模型 + 筛选器产生，第八篇的评测决定每轮改什么。

   </details>

### C. 面试题（7 题）

1. 用一张表把 SFT、RM、PPO、GRPO、DPO、RLVR、Agent RL、蒸馏放到"策略、奖励、参考"三件套上，说出每个名字改了哪一格。

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) SFT：只有策略，奖励隐含在标注序列里，无参考，1 个模型；(2) RM：Bradley-Terry 把成对偏好变成标量，是逻辑回归，1 epoch；(3) PPO：显式 RM、$$k_1$$ 进每 token 奖励、加价值模型，4 个模型 288 GB；GRPO：组内均值 / 标准差代替价值，$$k_3$$ 进 loss，3 个 160 GB；(4) DPO：奖励折进 $$\beta \log(\pi_\theta / \pi_{ref})$$，参考是分母，2 → 1 个模型，离线；(5) RLVR / Agent RL：奖励换成验证器 / 环境状态，参考极弱或无，多轮加 mask 与异步；(6) 蒸馏：奖励换成教师分布，on-policy 蒸馏 = 奖励为 $$\log p_T$$ 的稠密 RL；(7) 2025 年的新名字（DAPO、GSPO、SimPO、KTO）各是这张表上一格的改动。
   **追问方向**：DAPO 的四个修正各改目标函数的哪一项；ORPO / SimPO 去掉参考为什么都要长度归一化；在线 DPO 属于哪一格。
   **好答案与一般答案的区别**：一般答案按时间罗列方法；好答案先给骨架，再把每个名字定位到"奖励从哪来、参考怎么约束、要不要价值模型"三个问题上。

   </details>

2. 从 RLHF 的目标函数推出 DPO 的 loss，然后回答：DPO 到底需不需要奖励模型？它比 PPO 差在哪、好在哪？

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 固定 prompt，拉格朗日乘子给出 $$\pi^* \propto \pi_{ref} \exp(r / \beta)$$；(2) 反解 $$r = \beta \log(\pi^* / \pi_{ref}) + \beta \log Z$$，$$\beta \log Z$$ 正是 Bradley-Terry 只定义到的那个常数；(3) 代入 $$\sigma(r_w - r_l)$$，$$Z$$ 抵消，得 $$-\log\sigma(\beta \log\frac{\pi(y_w)}{\pi_{ref}(y_w)} - \beta \log\frac{\pi(y_l)}{\pi_{ref}(y_l)})$$；(4) 它训的是一个参数化为对数比的 RM，只在偏好对上受约束——似然同降、与 PPO 同形的过优化、长度偏差都由此来；(5) 好在两个模型、$$8N$$/token、10 万对 9 GPU 小时、参考可预计算、LoRA 下参考免费；差在不能探索，推理 / 代码任务 PPO / GRPO 明显占优。
   **追问方向**：梯度权重 $$\sigma(\hat r_l - \hat r_w)$$ 与 RM 梯度的关系；$$\beta$$ 大小各意味什么；为什么 on-policy 数据比换 loss 更重要。
   **好答案与一般答案的区别**：一般答案背"DPO 不需要 RM"；好答案说出 RM 去哪了（变成对数比）、代价是什么（分布外无约束），并给出成本与上限两端的数字。

   </details>

3. 8B 模型跑一轮 GRPO 对齐，需要几张卡、时间花在哪、该盯什么曲线？如果换成推理任务、回答长到 16K 呢？

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 三个模型：策略 128 GB 训练状态 + 参考 16 + RM 16 = 160 GB，加 KV cache（4096 条 × 1300 token = 680 GiB）与激活，4 张 80 GB 起步、实践 8 张，`beta=0` 可不加载参考；(2) 一步 512 × 8 × 1000 = 410 万 token，$$\approx 12N$$/token，FLOPs 训练一半、三个前向各六分之一，墙钟生成 50–80%（decode MFU 10–25%、长尾、引擎切换）；(3) 一轮几百到上千 GPU 小时，比 SFT 贵一个数量级；(4) 曲线：奖励、KL（2–3 nats 核对）、长度、clip 比例（> 20–30% 说明走太远）、熵、零优势组比例；(5) 推理任务：一步 1.3 亿 token，KV 16 TiB 分波，$$\beta$$ 可到 0，用 `dr_grpo`，部分 rollout 与异步，总算力 $$10^{22}$$–$$10^{23}$$。
   **追问方向**：为什么 GRPO 不省 token 只省显存；权重同步多久一次、滞后一步的代价；MoE 上为什么要 GSPO。
   **好答案与一般答案的区别**：一般答案只算显存；好答案把 FLOPs 与墙钟分开，指出问题是训练循环里的推理引擎，并说出长思维链让哪些数字放大多少倍。

   </details>

4. 奖励模型准确率只有 70%，能用吗？训 RL 之前怎么知道它会把策略推向哪？

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 能——人的一致率也只有 70–75%，RM 给的是梯度方向，随机错误被平均平掉；(2) 问题在系统偏差：长度、格式、语气、迎合、拒答，三层结构（数据真相关 → RM 捷径 → RL 利用）；(3) 过优化有定量规律 $$R_{gold}(d) = d(\alpha - \beta \log d)$$，有 KL 预算，RM 越大越晚退化；(4) 训 RL 前几小时的清单：长度相关（> 0.3 要控制）、BoN 扫描（$$N$$ = 16 ≈ 1.83 nats，免费预演）、对抗探针（加客套话、改列表看涨分）、一致性、校准、分布外、拒答；(5) 对策叠加：KL、长度控制、迭代重训、集成 / WARM、规则奖励接管可验证部分。
   **追问方向**：为什么集成修不了长度偏差；judge 当奖励的偏差与 RM 有什么关系；PRM 为什么更容易被 hack。
   **好答案与一般答案的区别**：一般答案讨论怎么把准确率提上去；好答案说"问题不是准不准而是偏在哪"，并给出训 RL 之前的探测方法与 KL 预算的概念。

   </details>

5. 为什么 DeepSeek-R1 能用规则奖励训出长思维链而 RLHF 训不出来？四个阶段各在修什么？小模型该怎么做？

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 规则奖励不可 hack → KL 可撤、探索可到几万 token；对长度中立，变长是被选出来的；基座已有推理种子，RL 放大它（pass@k）；RM 在策略走那么远之前就被找到漏洞；(2) R1-Zero：AIME 15.6 → 71.0，长度自发增长，aha，但可读性差、混语；(3) 四阶段：冷启动 SFT 修格式与语言 → 推理 RL + 语言一致性奖励 → 80 万条回基座重 SFT 合并推理与通用 → 全场景 RL 混合奖励；阶段 3 从基座重来是"用 RL 造数据、用 SFT 学数据"；(4) 成本：一步 rollout 多 8–64 倍，总算力 $$10^{22}$$–$$10^{23}$$；(5) 小模型：32B 蒸馏 72.6 对 RL 47，先序列级蒸馏、on-policy 对齐、再可选 RL；(6) PRM 不用：步骤难定义、标注难、hacking。
   **追问方向**：Tülu 3 的 RLVR 为什么没出现长思维链；长度控制的五种做法；验证器被 hack 的实例。
   **好答案与一般答案的区别**：一般答案复述"用了 GRPO 和规则奖励"；好答案解释不可 hack 如何撤掉 KL、如何让长度中立，并说清每个阶段修的是上一阶段的哪个具体问题。

   </details>

6. 要给一个模型训工具调用能力，从数据到训练到系统，与单轮 RLVR 比多了什么？为什么大家都说瓶颈在环境？

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 格式先靠 SFT：schema、调用标签、tool 角色、eom / eot 逐字节一致；(2) 一行公式 $$\log \pi(\tau) = \sum_t \log \pi(a_t \mid s_t)$$，环境转移不含 $$\theta$$，GRPO 原样成立；(3) 一处 mask：工具返回与模拟用户 mask = 0，不 mask 学会编造工具输出；(4) 奖励是环境状态（测试、状态比较、答案匹配），部分分权重压低，防改测试、硬编码、讨好模拟用户；(5) 一段系统：轨迹耗时秒到几十分钟、方差极大，异步 rollout 与按轮的 off-policy 修正必需，落后 1–4 步无损；(6) 账：4000 条轨迹环境 320 CPU·小时对模型 7 GPU·小时，八成 token 是环境的、有效 token 成本 5 倍，前缀缓存必备；数据每条要真跑，成功率 20–40%。
   **追问方向**：轨迹级还是 turn 级信用分配；SWE-RL 为什么不跑测试也能训；τ-bench 的 pass^k 为什么远低于 pass^1。
   **好答案与一般答案的区别**：一般答案讲 ReAct 与 prompt；好答案精确定位到公式、mask、系统三处差别，并用环境时间对 GPU 时间的账说明瓶颈。

   </details>

7. 一次后训练后 MMLU 涨 2 个点、AlpacaEval 胜率从 60% 到 80%，怎么判断是不是真的变好了？

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 分数 = 能力 + 协议 + 噪声 + 污染；(2) 三关：同一 harness、配置、抽取重跑 baseline；MMLU 14K 题 ±0.8，2 个点在边缘，任何子集上不显著，用配对比较；用改写题或新题复测涨幅是否保持（GSM1K 掉 13 点、MMLU-Redux 6.5% 错题的教训）；(3) judge 胜率先做长度控制的 win rate 与交换顺序，很多模型变 5–15 个点，换一个不同家族的 judge 消自我偏好；(4) 报告回答长度与基线的对比、拒答与误拒率、失败模式分布；(5) 切分 + 错误分类 + 读一百条输出，格式错与截断是协议问题先剔除；(6) 若 RL 用了某个奖励，要一个与奖励不同的度量发现 hacking。
   **追问方向**：pass@1 的 $$n$$；AIME 30 题为什么单次分数没有信息；Arena 的排行榜幻觉。
   **好答案与一般答案的区别**：一般答案说"多跑几个 benchmark"；好答案按协议、区间、污染的顺序给出可操作的三关，并知道 judge 胜率里有多少是长度。

   </details>

### D. 掌握判据

| 水平 | 表现 |
|---|---|
| 读过 | 能说出八篇各讲什么；知道三件套、Bradley-Terry、GRPO、DPO、RLVR、GKD 这些名词 |
| 掌握 | A 组能不翻书算出 8 题以上；B 组能说出每题用了哪几篇的什么；拿到一份后训练报告能把每一步放回三件套、估出模型数与 GPU 小时、指出它的评测协议缺什么 |
| 能教人 | C 组每题能给出全部要点并预判追问；能解释八篇里每个反直觉结论（70% 的 RM 够用、DPO 也有 Goodhart、RL 是放大不是创造、蒸馏好于直接 RL、瓶颈在环境、涨 2 个点不算涨）为什么成立 |

通关标准：A 组至少 8 题、B 组至少 4 题、C 组每题能说出一半以上要点。没过的部分回到第二章对应篇的"必记"，再回该篇正文。

## 六、下一步

八篇讲的是"从一个训好的基座出发，怎么把它变成对话、推理、Agent 模型并证明它变好了"，几个方向紧邻但不在范围内：

- **预训练与模型的成本账**（参数量、训练状态、$$6ND$$、tokenizer、数据管线、训练配方）是本系列的前提，在 L4 的[《Transformer 与 LLM》](/transformer-and-llm-for-infra-engineers.html)；交叉熵与 softmax 的梯度、AdamW 与调度、过拟合与多 epoch 在 L3 的[《深度学习基础》](/deep-learning-foundations.html)。
- **RL 训练系统的实现**（rollout 引擎与训练器共置、权重同步、异步 rollout、算力配比）在 Infra 地图 09 [《RL 后训练基础设施》](/rl-post-training-infrastructure.html)——本系列第三、五、六篇只给出算法对系统的要求。
- **推理侧的算法优化**（解码、投机解码、量化、KV 压缩）在 L6 [《高效推理与压缩（算法侧）》](/efficient-inference-and-compression-for-llms.html)——第五篇的 test-time compute 只讲用推理算力换准确率的曲线。
- **多模态后训练**在 L7 [《多模态：从视觉编码器到扩散模型》](/multimodal-from-vision-encoders-to-diffusion.html)——方法相同，数据与评测不同。
- **实验方法论**（可证伪的假设、小规模外推、控制随机性与复现）在算法地图的横切导读[《算法工程师的实验方法论》](/experimental-methodology-for-ai-algorithm-engineers.html)——第八篇的统计只是它的一角。

回到总纲：[《后训练：从 SFT 到可验证奖励》](/post-training-from-sft-to-verifiable-rewards.html)。

[^q0]: 八个：SFT 的每个开关（mask、packing、LoRA、epoch）各让模型变成什么样、怎么提前看到遗忘；70% 的 RM 为什么够用、在哪被钻空子；PPO 为什么四个模型、GRPO 用什么代替价值、算力花在哪；DPO 的奖励去了哪、在哪失效、与 PPO 差在哪；规则奖励为什么能训出长思维链、R1 四阶段各修什么、rollout 多多少；多轮 RL 差在哪一行公式、哪处 mask、哪段系统、瓶颈为什么在环境；三种蒸馏各学到什么、小模型为什么先蒸馏；分数涨了是能力、协议还是污染、judge 胜率里有多少是长度。详见[第二章](#二逐篇回顾)。
[^q1]: SFT lr $$10^{-5}$$、8B 128 GB、LoRA $$r = 16$$ 差 0.001、遗忘 +0.02 / +0.62 / +0.01、67 GPU 小时；一致率 70–75%、$$\sigma(r_w - r_l)$$、1 epoch、7 GPU 小时、$$\text{KL}_{BoN} = \log N - (N-1)/N$$（16 → 1.83）、金奖励随 $$\sqrt{\text{KL}}$$ 先升后降；$$\beta$$ 0.01–0.05 到 0、PPO 288 GB / GRPO 160 GB、一步 410 万 token、$$12N$$、生成占 50–80% 墙钟；$$\pi^* \propto \pi_{ref} e^{r/\beta}$$、$$\hat r = \beta\log(\pi_\theta/\pi_{ref})$$、$$8N$$、9 GPU 小时；AIME 15.6 → 71.0、80 万 = 60 + 20 万、一步 1.3 亿 token、$$10^{22}$$–$$10^{23}$$、蒸馏 72.6 vs RL 47；48K 轨迹八成是环境、320 CPU·小时 vs 7 GPU·小时；256 KB/token、on-policy ≈ RL 的 1/10、1.5B 29% / 32B 72.6%；协议 5–15 点、1000 题 ±3、AIME ±18、MMLU ±0.8、长度控制 0.94 → 0.98。详见[第一章](#一总览系列回答的问题与主线)、[第三章](#三贯穿全系列的几条线)。
[^q2]: 用第五章的三段自测：A 组 10 题判断与计算（至少 8 题）、B 组 5 题跨篇综合（至少 4 题）、C 组 7 道面试题（每题说出一半以上要点）；D 组的表给出"读过 / 掌握 / 能教人"三级的表现。详见[第五章](#五通关自测)。
