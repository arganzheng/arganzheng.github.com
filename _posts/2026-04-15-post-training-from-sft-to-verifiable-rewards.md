---
layout: post
title: 后训练：从 SFT 到可验证奖励（总纲）
subtitle: "Post-Training: From Supervised Fine-Tuning to Verifiable Rewards"
tags: [AI, LLM, Post-Training, RLHF]
catalog: true
---


## 内容简介

《后训练：从 SFT 到可验证奖励》是一组共八篇的系列文章，对应[《AI 算法工程师学习地图》](/ai-algorithm-engineer-learning-roadmap.html)的第 L5 层。它面向已经理解 Transformer 与预训练（L4，[《Transformer 与 LLM》](/transformer-and-llm-for-infra-engineers.html)十二篇）、准备把一个基座模型变成能对话、会推理、符合偏好的模型的读者，讲的是**后训练的每一种方法在优化什么目标、改了哪个组件、花多少钱、怎么证明它变好了**。

它回答的问题是：

> **一个只会续写的基座模型，经过哪几步变成能回答问题、能拒绝、能一步步推导数学题的模型？每一步的目标函数是什么，为什么这样写？训完之后，怎么知道它真的变好了而不是学会了讨好评委？**

后训练的方法很多——SFT、RM、PPO、GRPO、DPO、KTO、ORPO、RLVR、蒸馏——名字每几个月多几个。本系列不按名字罗列，而按**一个不变的骨架**组织：所有基于偏好或奖励的方法都在操作同样三个组件——**策略**（要训的模型）、**奖励**（判断输出好坏的信号）、**参考**（防止策略跑太远的锚）。每种方法的区别只在于：奖励从哪来（人标、模型打分、规则验证、教师分布）、参考怎么约束（KL 惩罚、闭式折叠进 loss、干脆去掉）、以及为了估计策略梯度还要不要第四个组件（价值模型）。把每种方法放回这三件套上，一族十几个名字就变成一张表。

每一篇用同样的三步处理它的方法——**推导**（目标函数从哪来、每一项在做什么）、**算账**（一次训练要几个模型在显存里、生成多少 token、标多少数据、花多少 GPU 小时）、**动手**（给出用 `trl` 在一个 0.5B–1.5B 的开源小模型上跑通的骨架与该看的曲线；第一篇附完整实验与作者机器上的数字）——最后指出这一步在 Llama 3、DeepSeek-R1、Qwen3、Tülu 3 这些公开配方里的形态。

举一个例子说明这个系列的取法。"DPO 不需要奖励模型"是每篇介绍都有的一句话；本系列要把它变成三件可验证的事：（一）推导：RLHF 的目标 $$\max_\pi \mathbb{E}[r(x, y)] - \beta\, \text{KL}(\pi \| \pi_{ref})$$ 有闭式最优解 $$\pi^*(y \mid x) \propto \pi_{ref}(y \mid x) \exp(r / \beta)$$，反解出 $$r = \beta \log(\pi^* / \pi_{ref}) + \text{const}$$，代入 Bradley-Terry 的偏好概率，奖励模型就被两个策略的对数比替代——**DPO 不是没有奖励，而是把奖励用策略与参考的比值隐式表达了**。（二）算账：PPO 要四个模型在显存里（策略、参考、奖励、价值，7B 规格约 250 GB 训练状态），DPO 只要两个且参考模型的对数概率可以离线预计算，一张 80 GB 的卡加 LoRA 就能训 7B。（三）实验：同一份偏好数据，DPO 与 PPO 各训一遍，比 RM 分数、比 KL、比 GSM8K——看到 DPO 的隐式奖励在训练分布外过拟合得更快。

系列覆盖的范围可以概括为后训练流水线的五段（第四段有一个多轮的延伸）加一个贯穿的度量：

```text
第一段   SFT              指令数据、chat template、loss mask、packing、全量 vs LoRA、遗忘        → 第一篇
第二段   偏好与奖励        偏好数据的三种形态、Bradley-Terry、奖励模型的训练与它的过拟合            → 第二篇
第三段   用奖励优化策略    在线 RL：采样 - 打分 - 更新；PPO、GRPO、RLOO、REINFORCE++                → 第三篇
                          离线 RL：不采样，从固定偏好数据直接推 loss；DPO 一族、拒绝采样          → 第四篇
第四段   可验证奖励        RLVR、R1 的配方、长思维链、PRM 与 ORM、test-time compute               → 第五篇
第四段＋ Agent RL          多轮工具调用作为 MDP、环境与轨迹数据、结果奖励、工具输出的 mask、异步 rollout   → 第六篇
第五段   蒸馏             logits 级、序列级、on-policy；R1 → 小模型                              → 第七篇
度量     评测             benchmark 各测什么、LLM-as-judge 的偏差、Arena、污染、自建评测集         → 第八篇
```


## 为什么写这个系列？

### 后训练是当前算法工程师工作量最集中的一层

预训练一年做一两次，后训练每周都在做：换一批 SFT 数据、加一个奖励、调一个 $$\beta$$、跑一轮评测。2025 年之后模型能力的主要增量——推理、工具调用、长思维链——几乎全部来自后训练而非预训练。但它的方法论比预训练年轻得多：InstructGPT（2022）确立三步法，DPO（2023）去掉奖励模型，DeepSeek-R1（2025）用规则奖励训出推理，每一步都推翻了前一步的一部分默认。一个 2024 年的教程到 2025 年已经缺了最重要的一章。

### 名字太多，骨架只有一个

PPO、GRPO、RLOO、REINFORCE++、DAPO、GSPO；DPO、IPO、KTO、ORPO、SimPO、cDPO；RLHF、RLAIF、RLVR。每个名字对应一篇论文、一组实验、一个"比前者好"的表。不把它们放回同一个骨架，就只能记；放回去之后，每个名字是骨架上一处改动——去掉价值模型、把奖励折进 loss、把成对比较换成单样本、把 KL 项去掉、把 token 级重要性比换成序列级。本系列的第三、四篇各有一张表，把这一族按"改了三件套的哪一件"排开。

### 每一步都能算账，而账很少被算

一轮 GRPO 训练生成多少 token？一个 8B 模型跑 PPO 要几张卡？标 10 万对偏好数据要多少人时，用 AI 反馈替代能省多少、损失多少？蒸馏 800K 条推理样本的教师推理成本是多少？这些数字决定一个团队能不能做某种方法，但论文很少写，教程几乎不写。本系列把 L4 的算账方法带进来：每种方法一张成本表。

### 现有材料的断层

- **论文**每篇一个方法，实验设置各异，很难横向比；
- **框架文档**（`trl`、OpenRLHF、verl）讲配置项，把目标函数当已知；
- **课程与教程**多停在 InstructGPT 三步法或 DPO，缺 2025 年的推理 RL 与蒸馏；
- **技术报告**（Llama 3、DeepSeek-R1、Qwen3、Tülu 3）给出配方但不解释每一步为什么。

本系列取的是中间那段路：每个方法推到目标函数、算到成本、跑到曲线，再对到公开配方里的那一行。


## 适合哪些读者？

### 读完 L4、准备动手做后训练的算法学习者

你理解 Transformer、知道 6ND、读过一两份技术报告，接下来要用 `trl` 训自己的第一个对话模型。本系列是从"能跑通脚本"到"知道每个参数在控制什么、曲线不对时看哪"的那段路。

### 已经在做 SFT / DPO，但对 RL 与推理模型生疏的工程师

你的流水线停在 SFT + DPO，想加 GRPO 或 RLVR，但 PPO 的四个模型、GAE、KL 系数、rollout 引擎这些词让你迟疑。第三、五篇是为此准备的：从策略梯度推到 PPO，再看 GRPO 去掉了什么、RLVR 换掉了什么，以及一次训练的显存与 token 账。

### 需要评测与选型、而不是训练的工程师

你要在几个模型之间选一个，或者判断一次训练有没有变好。第八篇单独讲评测：每个 benchmark 测什么、LLM-as-judge 的三种偏差、Arena 的统计方法、污染怎么查、什么时候必须自建评测集。它可以独立阅读。

### Infra 工程师，想知道 RL 后训练在系统上要什么

rollout 引擎与训练器共置、权重同步、生成与训练的算力配比——这些 Infra 地图里的"选修"负载，其算法需求在第三、五篇：为什么 RL 训练的大头是推理、为什么 GRPO 让 batch 里的序列长度方差极大、为什么 on-policy 与 off-policy 对权重同步的要求不同。


## 系列的整体主线

八篇按后训练流水线的顺序推进，前一篇的产物是后一篇的输入：

```text
第一篇：SFT —— 基座 → 会按格式回答；数据、模板、mask、PEFT、遗忘
        ↓
第二篇：偏好数据与奖励模型 —— 人的判断 → 一个可微的信号；Bradley-Terry、RM 的过拟合与 hacking
        ↓
第三篇：在线 RL —— 用奖励改策略；PPO 的四个模型 → GRPO 的三个；三件套对照表
        ↓
第四篇：离线 RL —— 不采样，把奖励折进 loss；DPO 的推导与它的一族变体；拒绝采样 + SFT
        ↓
第五篇：推理模型与可验证奖励 —— 奖励换成验证器；R1 的配方、长思维链、PRM/ORM、test-time compute
        ↓
第六篇：Agent 与工具调用的 RL —— 奖励延后到多轮之后；环境、轨迹、工具输出的 mask、异步 rollout
        ↓
第七篇：蒸馏 —— 奖励换成教师分布；logits 级、序列级、on-policy；R1 → 小模型
        ↓
第八篇：评测 —— 每一步"变好了"怎么证明；benchmark、judge、Arena、污染、自建
```

第一篇没有奖励也没有参考，只有策略与标注好的目标序列。第二篇造出奖励。第三、四篇是**同一个目标的两种解法**——在线与离线：第三篇在线地采样 - 打分 - 更新，第四篇离线地从固定的偏好对直接推出 loss。第五篇把奖励从"学出来的模型"换成"规则验证器"，这是 2025 年推理模型的转折点。第六篇把 RLVR 从单轮推到多轮：模型的动作里出现真实的工具调用与环境反馈，奖励延后到整条轨迹结束，这是 2025–26 年 Agent 能力的来源。第七篇把奖励换成教师模型的分布，把大模型的能力搬进小模型。第八篇是贯穿全流程的度量——每一篇的实验都用它来判断"变好了没有"。

三条交织的线索：

```text
推导线：交叉熵 → Bradley-Terry 与 pairwise loss → 策略梯度、优势、KL 约束 → 闭式最优解与 DPO → 规则奖励下的 GRPO → 多轮轨迹的信用分配 → 前向 KL 与反向 KL → 评测的统计
成本线：SFT 一个模型 → RM 一个 → PPO 四个 / GRPO 三个 → DPO 两个（参考可离线）→ RLVR 的 rollout token 数 → Agent RL 的环境步数与沙箱 → 蒸馏的教师推理 FLOPs → 评测的 judge 调用
配方线：InstructGPT 三步法 → Llama 3 的六轮 SFT+DPO → Tülu 3 的 RLVR → DeepSeek-R1 的四阶段与蒸馏 → Qwen3 的思考模式融合 → Kimi K2 的大规模 Agent 数据合成与 RL
```

每一篇都用同样的方法：**写出目标函数并解释每一项，算出这一步的显存、token 与数据成本，给出在小模型上跑一遍的骨架与该看的曲线，再对到公开配方里的那一行**。


## 章节结构与分章导读

### 1. SFT：指令数据、chat template、loss mask 与参数高效微调

第一篇建立整个系列的起点：把一个续写模型变成按格式回答的模型。它只有一个组件（策略）和一个目标（交叉熵），但每个工程细节都影响后面所有步骤。

这一篇会覆盖：

- 指令数据的三种来源：人工编写（InstructGPT 的 13K、LIMA 的 1K）、self-instruct 一类的自举、从更强模型蒸馏（ShareGPT、Alpaca、UltraChat）；数量与质量的取舍——LIMA 的"1000 条够了"与 Tülu 3 的 93.9 万条各在什么条件下成立；
- 多轮对话的格式与 chat template：角色标记、特殊 token（第九篇留下的 256 个位）、system prompt；模板不一致是 SFT 最常见的静默错误；
- loss mask：只对回复部分算 loss，prompt 部分不算；为什么算 prompt 会伤害模型；多轮时只算最后一轮还是每一轮；
- packing 与它的 attention 掩码问题（L4 第十二篇的跨文档 attention 在这里再出现一次）；
- 全量微调的账：8B 模型 16 字节/参数的训练状态 128 GB（L4 第六篇），lr $$10^{-5}$$ 量级、2–3 个 epoch、cosine；
- 参数高效微调：LoRA 的 $$W + BA$$（L4 第七篇的参数与状态账）、秩与目标矩阵的选择（r 16–64，全部线性层优于只做 attention）、QLoRA 的 NF4 底座、DoRA 的幅度 - 方向分解；Prefix-Tuning / P-Tuning / Adapter 的位置；LoRA 在什么任务上追不上全量；
- 灾难性遗忘：SFT 后通用能力下降的度量与三种对策——混入预训练数据回放、更小的 lr、模型平均（Llama 3 对每轮 SFT/DPO 的 checkpoint 做平均）；
- 公开配方：InstructGPT 的 SFT、Llama 3 的 SFT（lr 1e-5，8.5K–9K 步，拒绝采样生成的数据）、Tülu 3 的 SFT 数据配比、DeepSeek-R1 的冷启动 SFT（几千条长思维链）。

核心问题是：

> **同一批 10K 条指令数据，loss mask 算不算 prompt、packing 掩不掩跨样本、LoRA 的秩取 8 还是 64、训 1 个 epoch 还是 5 个——每个选择各让模型变成什么样？怎么在训完之前就知道它会不会忘掉预训练学到的东西？**

实验：用 `trl` 的 `SFTTrainer` 在 Qwen2.5-0.5B 上做 SFT；对比有无 loss mask、有无 packing 掩码、全量 vs LoRA r=16 的曲线与输出；用几个通用 benchmark 的子集量化遗忘。

### 2. 偏好数据与奖励模型：Bradley-Terry、pairwise loss 与 reward hacking

第二篇造出后面所有方法都需要的东西：一个把"人觉得这个回答更好"变成可微标量的模型。它是 L2 层"奖励模型本质上是一个分类器"那句话的展开。

这一篇会覆盖：

- 偏好数据的三种形态：成对比较（A 好于 B）、打分（1–10）、排序（多个回答排一列）；为什么成对比较最稠密、最一致——人对绝对分数不可靠，对相对好坏可靠；
- 数据从哪来：人标（InstructGPT 的 33K 对、Anthropic HH）、AI 反馈（RLAIF：用更强的模型按 constitution 打分）、混合（UltraFeedback 用 GPT-4 打分 6.4 万条 prompt × 4 个回答）；标注协议——同一 prompt 的多个回答从哪个策略采样、标注者看不看模型身份；
- Bradley-Terry 模型：$$P(y_w \succ y_l) = \sigma(r(y_w) - r(y_l))$$，它就是逻辑回归；pairwise loss $$-\log \sigma(r_w - r_l)$$ 的梯度与它对"差距"的处理；margin 项；
- 奖励模型的结构：基座模型去掉 lm_head 换一个标量头，从 SFT 模型初始化；用多大的模型做 RM（InstructGPT 用 6B 给 175B 打分；Llama 3 用与策略同规格）；
- RM 的过拟合：偏好数据只有几万对，RM 训 1 个 epoch 就开始过拟合；准确率在 65–75% 就是上限（人与人之间的一致率也只有这么多）；
- reward hacking：策略找到 RM 的漏洞——长度偏好（RM 偏爱长回答，策略越训越长）、格式偏好（列表、加粗）、自信语气；对策：长度惩罚、RM 集成、定期用新样本重训 RM、KL 约束（第三篇）；
- 生成式 RM 与 LLM-as-judge 作为奖励：让模型先写评语再打分，比标量头更可解释也更贵（第八篇的偏差在这里同样存在）；
- 过程奖励模型（PRM）与结果奖励模型（ORM）的区别，留到第五篇展开。

核心问题是：

> **一个准确率 70% 的奖励模型为什么够用？它在什么地方会被策略钻空子，怎么在训 RL 之前就发现？**

动手（建议）：在 UltraFeedback 的子集上训一个 0.5B 的 RM，画训练 / 验证准确率随 epoch 的曲线看过拟合；用它给不同长度的回答打分，量化长度偏好；对比标量 RM 与"让 1.5B 模型当 judge"在同一验证集上的一致率。

### 3. 在线 RL：PPO、GRPO 与 RLHF 三件套

第三篇讲用奖励改策略的在线方法：采样 → 打分 → 更新，循环。它是本系列推导最重的一篇，也是"三件套"这张表第一次完整出现的地方。

这一篇会覆盖：

- RLHF 的目标函数：$$\max_\pi \mathbb{E}_{x, y \sim \pi}[r(x, y)] - \beta\, \text{KL}(\pi \| \pi_{ref})$$；每一项在防什么——没有 KL 项策略会坍缩到 RM 的漏洞上；$$\beta$$ 的量级与它对输出多样性的影响；
- 策略梯度：REINFORCE 的 $$\nabla \log \pi(y) \cdot R$$、为什么方差大、baseline 为什么不改变期望只降方差；把 token 当动作、把序列当 episode——LLM 的 RL 是"只有最后一步有奖励"的稀疏奖励问题；
- PPO 的四个组件：策略、参考（算 KL）、奖励（打分）、价值（估计 baseline）；GAE 的 $$\lambda$$ 与 $$\gamma$$；clip 的 $$\epsilon$$ 在做什么——限制每次更新策略比值 $$\pi / \pi_{old}$$ 的范围；每批 rollout 做几个 epoch 的更新；
- 显存账：7B 规格下策略与价值各 112 GB 训练状态，参考与 RM 各 14 GB 推理权重，合计约 250 GB，加上 rollout 的 KV cache——这是 PPO 需要 8 张卡起步的原因；
- GRPO（DeepSeekMath 2024）：同一 prompt 采样一组 $$G$$ 个回答，用组内的均值与标准差归一化奖励作为优势，去掉价值模型；显存降到三个模型；代价是每个 prompt 要生成 $$G$$ 倍的 token；
- RLOO 与 REINFORCE++：leave-one-out baseline、全局 baseline——同样去掉价值模型的另外两种做法；它们与 GRPO 在方差与偏差上的区别；
- 2025 年对 GRPO 的修正：DAPO 的 clip-higher、动态采样、token 级 loss 与超长惩罚；Dr. GRPO 去掉长度与标准差归一化带来的偏差；GSPO（Qwen3）把 token 级重要性比换成序列级——每一个修正对应 GRPO 目标函数里的一项；
- 成本账：一次 GRPO 训练生成多少 token（prompt 数 × $$G$$ × 平均长度），推理为什么是 RL 训练算力的大头；on-policy 与 off-policy（rollout 与训练用不同版本的权重）对权重同步频率的要求——Infra 地图选修那一节的算法侧；
- 三件套对照表（第一版）：SFT、RM、PPO、GRPO、RLOO、REINFORCE++ 各用了哪几个组件、奖励从哪来、参考怎么约束、是否从当前策略采样（在线 / 离线）。

核心问题是：

> **PPO 为什么要四个模型？GRPO 去掉价值模型之后，用什么代替它、代价是什么？一次 GRPO 训练的算力里推理与反向各占多少？**

动手（建议）：用 `trl` 的 `GRPOTrainer` 在 Qwen2.5-0.5B-Instruct 上用第二篇的 RM 做一轮在线 RL；记录奖励、KL、平均回答长度随步数的曲线；对比 $$\beta$$ 取 0.01 与 0.1、$$G$$ 取 4 与 16 的差别；统计生成 token 数与训练 token 数之比。

### 4. 离线 RL：从 RLHF 目标推出 DPO 及其变体

第四篇讲同一个目标的离线解法：不从当前策略采样、不在线打分、不要奖励模型，直接从一份固定的偏好数据推出一个 loss。它的推导是本系列最漂亮的一段，它的变体是名字最多的一族。

先说清"离线"这个词。这里指**不与当前策略交互、只用一份事先收集好的偏好数据**训练，即文献里的 offline preference optimization / direct alignment；它与 RL 文献里的 offline RL（从固定的轨迹数据学价值函数与策略）共享"不与环境交互"这一层含义，但方法不同——这里没有价值函数，偏好数据也不必来自参考策略。介于两者之间的半在线形态（在线 DPO、迭代 DPO、拒绝采样 + SFT：用当前策略采样、离线地标偏好与更新）也放在本篇。

这一篇会覆盖：

- 闭式最优解：$$\pi^*(y \mid x) \propto \pi_{ref}(y \mid x) \exp(r(x, y) / \beta)$$ 的推导（KL 约束下最大化期望奖励是一个变分问题）；反解 $$r = \beta \log(\pi^* / \pi_{ref}) + \beta \log Z(x)$$，$$Z$$ 在成对比较里抵消；
- DPO 的 loss：$$-\log \sigma\left(\beta \log \frac{\pi(y_w)}{\pi_{ref}(y_w)} - \beta \log \frac{\pi(y_l)}{\pi_{ref}(y_l)}\right)$$；它的梯度权重 $$\sigma(\hat r_l - \hat r_w)$$——隐式奖励排错时权重大；$$\beta$$ 的含义与 0.1 的默认；
- 隐式奖励的问题：DPO 只在训练数据的分布上约束策略，分布外的隐式奖励可以任意；这是它比 PPO 更容易过拟合、"chosen 与 rejected 的概率同时下降"这一现象的来源；
- 一族变体各改了哪一项：IPO（把 sigmoid 换成平方 loss，避免过拟合）、KTO（不要成对数据，单样本的"好 / 坏"标签，前景理论的效用函数）、ORPO（去掉参考模型，SFT loss 加一个 odds ratio 项）、SimPO（去掉参考模型，用长度归一化的对数概率做隐式奖励，加 margin）、cDPO / rDPO（标签噪声）——每一个对应三件套上的一处改动；
- 在线 DPO 与迭代 DPO：用当前策略采样、用 RM 或 judge 标偏好、再做 DPO，循环——把离线方法变成半在线的，Llama 3 的六轮就是这个形态；
- 拒绝采样 + SFT：对每个 prompt 采样 $$K$$ 个回答，用 RM 选最好的做 SFT——最简单的"用奖励改策略"，Llama 2 / 3 的主力手段之一；它与投机解码里的拒绝采样同名不同物；
- DPO 与 PPO 的对照：数据效率、过拟合、能达到的上限、成本（两个模型 vs 四个，参考模型的对数概率可以离线预计算一次）；什么时候选哪个；
- 三件套对照表（第二版）：加上 DPO、IPO、KTO、ORPO、SimPO、拒绝采样、在线 / 迭代 DPO，"在线 / 离线"一列把整族排开。

核心问题是：

> **DPO 真的"不需要奖励模型"吗？它的隐式奖励在哪里会失效？同一份偏好数据，DPO 与 PPO 训出的模型差在什么地方？**

动手（建议）：用第二篇的偏好数据在同一个 SFT 模型上分别跑 DPO 与第三篇的 GRPO；画 chosen / rejected 对数概率随步数的曲线（观察两者同时下降的现象）；用第二篇的 RM 与第八篇的 judge 评两者的输出；对比 $$\beta$$ 0.05 / 0.1 / 0.5。

### 5. 推理模型与可验证奖励：R1 的配方、PRM 与 test-time compute

第五篇讲 2025 年后训练的转折：把奖励从"学出来的模型"换成"规则验证器"，用 RL 训出长思维链。它是三件套里"奖励从哪来"这一格最大的一次改动。

这一篇会覆盖：

- 可验证奖励（RLVR）：数学题比答案、代码跑测试、格式检查——奖励是确定的、不可 hack 的（在验证器正确的前提下）；Tülu 3 首次把它作为独立阶段；它能用的领域边界；
- DeepSeek-R1-Zero：直接在基座上用 GRPO + 规则奖励（准确 + 格式），不做 SFT；训练中回答长度自发增长、出现自我反思（"aha moment"）；它的问题——可读性差、语言混杂；
- DeepSeek-R1 的四阶段：冷启动 SFT（几千条长思维链）→ 推理 RL（加语言一致性奖励）→ 拒绝采样生成 60 万条推理 + 20 万条非推理数据做 SFT → 全场景 RL（推理用规则奖励、通用用 RM）；每一阶段在修前一阶段的什么问题；
- 长思维链的成本：回答从几百 token 到几万 token，rollout 的 token 数与 KV cache 随之增长一到两个数量级；batch 内长度方差极大对推理引擎与训练效率的影响；长度控制——长度惩罚、预算强制、Qwen3 的思考预算；
- 过程奖励与结果奖励：ORM 只看最终答案，PRM 给每一步打分（Lightman 等 2023 的 PRM800K）；PRM 的标注成本与自动化（蒙特卡洛估计每步的正确概率）；R1 为什么没用 PRM（reward hacking 与标注成本）；
- test-time compute：同一模型多采样取多数（self-consistency）、用 PRM 选最优（best-of-N）、搜索（MCTS 一类）；推理算力换准确率的曲线与它的边际；o1 与 R1 把这条曲线内化到模型里——训练让模型自己决定想多久；
- 推理模型的蒸馏预告：R1 用 80 万条样本 SFT 出 1.5B–70B 的推理模型，不做 RL——小模型上直接 RL 不如蒸馏（第七篇展开）；
- 公开配方对照：R1、Qwen3（思考 / 非思考模式融合的四阶段）、Kimi K2 的可验证奖励与自评奖励、gpt-oss 的推理等级。

核心问题是：

> **为什么规则奖励能训出长思维链，而奖励模型训不出来？R1 的四个阶段各在修什么？一个推理模型的 RL 训练，rollout 的 token 数比对话模型多多少？**

动手（建议）：用 `trl` 的 `GRPOTrainer` 在 Qwen2.5-1.5B-Instruct 上以 GSM8K 子集的答案匹配为奖励训 RLVR；画准确率、回答长度、"包含反思词"的比例随步数的曲线；对比有无格式奖励；测 best-of-N 在 N = 1 / 4 / 16 下的准确率与推理 token 数。

### 6. Agent 与工具调用的 RL：多轮环境、轨迹数据与延后的奖励

第六篇把第五篇的可验证奖励从单轮推到多轮：模型不再一次生成到底，而是生成一段 → 调用工具 → 读到环境返回 → 再生成，奖励要等整条轨迹结束才知道。它是 RLVR 的直接延伸，也是 2025 年后 Agent 能力（写代码改仓库、搜索并综合、操作终端与浏览器）的训练来源。

这一篇会覆盖：

- 问题设定：把多轮工具调用写成 MDP——状态是到目前为止的全部对话（含工具输出），动作是模型生成的一段 token（思考 + 工具调用或最终回答），环境是工具与沙箱，奖励通常只在最后一步；与第三篇"单步 bandit"的区别就在这一个"多步"；
- 工具调用的格式：函数 schema（JSON）、chat template 里的 tool 角色与特殊 token、并行调用、ReAct 式的思考 - 行动 - 观察交替；格式本身要先靠 SFT 学会，RL 只在会用工具的模型上才有起点；
- 轨迹数据的三种来源：人工示范（贵且少）、更强模型生成再按结果过滤（拒绝采样在多轮上的形态）、大规模合成环境（Kimi K2 用模拟的工具与用户批量造轨迹）；为什么 Agent SFT 数据比对话 SFT 数据难得多——每条都要真实跑过环境；
- 奖励的设计：结果奖励（测试通过、答案匹配、任务完成的判定）为主；过程奖励与 rubric judge 作为 shaping 的风险（第二篇的 hacking 在多轮上更容易）；对无效调用、超时、超步数的惩罚；
- 训练目标：GRPO / PPO 在轨迹级上的写法——loss 只算模型生成的 token，**工具输出与用户轮次全部 mask 掉**（第一篇 loss mask 的多轮版本）；轨迹级优势与 turn 级信用分配的取舍；KL 与 clip 在长轨迹上的行为；
- 系统与成本：rollout 要与环境交互，每一步有 IO 与沙箱启动的延迟，同一 batch 的轨迹长度与耗时方差极大——**异步 rollout** 与 off-policy 修正成为必需（Infra 地图选修那一节在这里被逼出来）；一次训练的成本要加上环境步数与沙箱数，代码任务里环境比模型贵是常态；
- 公开配方：SWE-RL（用 GitHub PR 的相似度做奖励训代码修复）、Search-R1 与 ReTool（搜索 / 代码解释器作为工具的 RLVR）、Kimi K2 的 Agent 数据合成与 RL、Qwen3 的 Agent 阶段、o3 / gpt-oss 把工具调用放进思维链；
- 评测的预告：τ-bench、BFCL、SWE-bench Verified、GAIA、Terminal-Bench 各测哪一类 Agent 能力，第八篇展开。

核心问题是：

> **多轮工具调用的 RL 与单轮 RLVR 差在哪一行公式、哪一处 mask、哪一段系统？为什么 Agent 训练的瓶颈常常不是模型而是环境？**

动手（建议）：给 Qwen2.5-1.5B-Instruct 一个 Python 计算器工具，在 GSM8K 子集上自写一个最小的多轮 rollout（生成 → 解析工具调用 → 执行 → 拼回对话），以最终答案匹配为奖励用 `trl` 的 `GRPOTrainer` 训练；对比有无工具输出 mask 的曲线；统计每条轨迹的环境步数、耗时与 token 数的分布，看长度方差如何拖慢一个 batch。

### 7. 蒸馏：logits 级、序列级与 on-policy

第七篇讲把大模型的能力搬进小模型。它是三件套里奖励的第三种来源——教师模型的分布——也是"线上回流"到边端的最后一站。

这一篇会覆盖：

- 蒸馏的三种粒度：logits 级（Hinton 等 2015：学生匹配教师的软分布，温度 $$T$$ 的作用）、序列级 / 数据蒸馏（教师生成文本，学生做 SFT——R1 蒸馏用的就是这个）、特征级（匹配中间层，LLM 上少用）；
- 前向 KL 与反向 KL：$$\text{KL}(p_{teacher} \| p_{student})$$ 让学生覆盖教师的所有模式（mode-covering），反向 KL 让学生集中在教师的主模式上（mode-seeking）；MiniLLM 用反向 KL 的理由；
- on-policy 蒸馏（GKD，Agarwal 等 2023）：让学生自己生成序列，教师在学生的序列上给每个 token 打分布，学生向它对齐——修正了"教师序列的分布与学生推理时的分布不一致"这一暴露偏差；它把蒸馏变成了一种 RL（奖励是教师的对数概率）；
- 词表不同怎么办：教师与学生 tokenizer 不同（Llama → Qwen）时 logits 级蒸馏做不了，只能序列级；同系列模型（Qwen3 大 → 小）可以 logits 级；
- 成本账：序列级蒸馏的教师推理 FLOPs（80 万条 × 平均长度 × $$2N_{teacher}$$）；logits 级蒸馏每 token 要传 $$V$$ 个数（128K 词表 × 2 字节 = 256 KB/token，通常只传 top-k）；on-policy 蒸馏每步教师一次前向；
- 蒸馏与量化的组合：先蒸馏再量化，或用量化感知的蒸馏（QAT 的 loss 里加教师项）；Gemma 2 / 3、Qwen3 小模型的蒸馏配方；
- 蒸馏的极限：小模型的容量上限；直接在小模型上 RL 不如蒸馏（R1 报告的对照）；蒸馏出的模型评测分数高但泛化窄的风险。

核心问题是：

> **同一个教师，logits 级、序列级、on-policy 三种蒸馏各让学生学到什么、漏掉什么？R1 为什么给小模型选择蒸馏而不是 RL？**

动手（建议）：以 Qwen2.5-1.5B-Instruct（或第五篇训出的推理模型）为教师、0.5B 为学生，分别做序列级蒸馏（教师生成 → 学生 SFT）与 logits 级蒸馏（top-k 软标签的 KL）；对比 GSM8K 准确率与输出多样性；用 `trl` 的 `GKDTrainer` 跑一轮 on-policy 蒸馏对照。

### 8. 评测：benchmark、LLM-as-judge、Arena 与污染

第八篇讲贯穿全流程的度量。前七篇每一步的"变好了"都靠它判断；它也是"回到数据或配方"那条回边的起点——不会评测就不知道改什么。

这一篇会覆盖：

- 主流 benchmark 各测什么、多大、怎么算分：知识（MMLU 的 57 科 14K 题、MMLU-Pro、GPQA 的研究生级 448 题）、数学（GSM8K 1319 题、MATH 5000 题与 MATH-500、AIME 每年 30 题）、代码（HumanEval 164 题、LiveCodeBench、SWE-bench 的 2294 个真实 issue 与 Verified 500）、指令遵循（IFEval 的 541 条可验证约束）、对话（MT-Bench 80 题两轮、AlpacaEval）、Agent（τ-bench 的多轮客服任务、BFCL 的函数调用、GAIA、Terminal-Bench——它们评的是整条轨迹的结果，方差与成本都比单轮高一个量级）；每个 benchmark 的饱和程度与它还能区分什么；
- 评测协议的细节如何改变分数：few-shot 数、CoT 与否、采样温度、答案抽取的正则、pass@k 的 $$k$$——同一模型在不同协议下相差十几个点是常态；`lm-evaluation-harness` 一类工具的作用是固定协议；
- LLM-as-judge：用强模型给回答打分或成对比较；三种系统性偏差——位置偏差（偏爱先出现的）、长度偏差（偏爱长的）、自我偏好（偏爱同家族模型的文风）——与对策（交换顺序、长度控制的 win rate、多 judge）；judge 的一致率上限；
- 人类偏好 Arena：匿名成对比较 + Bradley-Terry 打分（第二篇的模型在这里用来给模型排名）；它测的是"人喜欢"而非"正确"，风格与长度的影响；style control；
- 污染：benchmark 题目在训练集里（L4 第十一篇的 8-gram 检测是训练侧的对策）；评测侧的检测——用改写过的题、比较训练集出现前后发布的题目、canary 字符串；被污染的分数是无意义的；
- 能力分解与错误分析：总分不告诉你改什么；按题型 / 长度 / 领域切分，看错误的类型（没理解、算错、格式错、拒答）；
- 自建评测集：什么时候必须（业务任务、私有领域、防污染）、怎么建（从真实流量采样、标注协议、大小与置信区间）、怎么维护；
- 评测的统计：多少题才能区分两个模型 1 个点的差异（二项分布的置信区间：1000 题上约 ±3 个点）；多 seed；报告的诚实性。

核心问题是：

> **一个模型的 MMLU 涨了 2 个点，怎么判断这是能力提升、协议变化、还是污染？用 GPT-4 当 judge 得到的 80% 胜率，去掉长度偏差之后还剩多少？**

动手（建议）：用 `lm-evaluation-harness` 在同一模型上跑 GSM8K 的三种协议（0-shot / 8-shot / CoT）看分差；用 1.5B 模型当 judge 对第四篇的 DPO 与 GRPO 输出做成对比较，交换顺序测位置偏差，按长度分桶测长度偏差；对 100 道 GSM8K 题算置信区间。


## 贯穿全系列的实践线

本系列的贯穿物是**一条在单张消费级 GPU 上能跑完的后训练流水线**，用 `trl` 与 Qwen2.5 的 0.5B / 1.5B 模型：

```text
第一篇    SFT           Qwen2.5-0.5B + 指令数据子集 → 对话模型；loss mask、packing、LoRA 的对照
第二篇    RM            UltraFeedback 子集 → 0.5B 奖励模型；过拟合曲线、长度偏好
第三篇    GRPO          第一篇的模型 + 第二篇的 RM → 在线 RL；奖励 / KL / 长度曲线
第四篇    DPO           同一份偏好数据 → DPO；与 GRPO 的输出对照
第五篇    RLVR          Qwen2.5-1.5B-Instruct + GSM8K 答案匹配 → 推理模型；长度与准确率曲线，best-of-N
第六篇    Agent RL      Qwen2.5-1.5B-Instruct + Python 计算器工具 → 多轮 GRPO；工具输出 mask 的对照；轨迹长度分布
第七篇    蒸馏          第五篇的模型为教师 → 0.5B 学生；三种蒸馏对照
第八篇    评测          lm-evaluation-harness 的协议对照；judge 的偏差测量；置信区间
```

第一篇的实验完整可跑，脚本与作者机器上的输出在 [ai-learning-labs/post-training](https://github.com/arganzheng/ai-learning-labs/tree/main/post-training)；第二到八篇给出的是**动手骨架**——`trl` 对应 trainer 的最小配置、自定义的奖励函数 / rollout / 数据格式那几十行，以及训起来之后该看的几条曲线——文中不引用未跑过的数字，公开配方的数字均注明来源。第二、四、八篇在 CPU 或极小配置上可以跑通（分钟到小时级），第三、五、六、七篇的 RL 与蒸馏需要一张 16–24 GB 的 GPU。所有动手的目的是**看曲线的形状与输出的变化**，不是刷分——0.5B 模型的绝对分数没有意义。

与它平行的源码与资料阅读线：

```text
第一篇    Ouyang 等 2022（InstructGPT）· Zhou 等 2023（LIMA）· Hu 等 2021（LoRA）· Dettmers 等 2023（QLoRA）· Llama 3 与 Tülu 3 报告的 SFT 章节 · trl 的 SFTTrainer 源码
第二篇    Bradley & Terry 1952 · Stiennon 等 2020 · Bai 等 2022（HH-RLHF、Constitutional AI / RLAIF）· Cui 等 2023（UltraFeedback）· Gao 等 2022（reward hacking 的 scaling）
第三篇    Schulman 等 2017（PPO）· Schulman 等 2015（GAE）· Shao 等 2024（DeepSeekMath，GRPO）· Ahmadian 等 2024（RLOO）· Yu 等 2025（DAPO）· Liu 等 2025（Dr. GRPO）· Zheng 等 2025（GSPO）· trl 的 GRPOTrainer 与 verl 的 rollout / 训练循环
第四篇    Rafailov 等 2023（DPO）· Azar 等 2023（IPO）· Ethayarajh 等 2024（KTO）· Hong 等 2024（ORPO）· Meng 等 2024（SimPO）· Llama 3 报告的 DPO 章节
第五篇    DeepSeek-AI 2025（R1）· Lambert 等 2024（Tülu 3，RLVR）· Lightman 等 2023（PRM）· Snell 等 2024（test-time compute）· Qwen3 与 Kimi K2 报告的后训练章节
第六篇    Yao 等 2022（ReAct）· Schick 等 2023（Toolformer）· Wei 等 2025（SWE-RL）· Jin 等 2025（Search-R1）· Feng 等 2025（ReTool）· Kimi K2 报告的 Agent 数据合成章节 · verl 的多轮 rollout 与 agent loop
第七篇    Hinton 等 2015 · Kim & Rush 2016（序列级）· Gu 等 2023（MiniLLM）· Agarwal 等 2023（GKD）· R1 报告的蒸馏章节 · Gemma 2 报告
第八篇    Hendrycks 等 2020（MMLU）· Cobbe 等 2021（GSM8K）· Zheng 等 2023（MT-Bench 与 judge 的偏差）· Chiang 等 2024（Chatbot Arena）· Zhou 等 2023（IFEval）· Yao 等 2024（τ-bench）· lm-evaluation-harness 源码
```


## 阅读路径建议

### 完整学习路径

```text
1 → 2 → 3 → 4 → 5 → 6 → 7 → 8
```

### 只做 SFT + DPO 的对话模型

```text
1 → 2 → 4 → 8
```

不碰在线 RL 也需要第二篇——DPO 的数据就是偏好数据，隐式奖励的问题要用 RM 的视角理解。

### 要做推理模型

```text
1 → 3 → 5 → 7 → 8
```

先懂 SFT（冷启动）与 GRPO，再看 RLVR 换掉了什么，蒸馏怎么把它搬进小模型。第二、四篇在需要通用对齐时补读，第六篇在要加工具调用时补读。

### 要做 Agent / 工具调用

```text
1 → 3 → 5 → 6 → 8
```

工具调用的格式先靠 SFT，再走单轮 RLVR 到多轮 Agent RL；第八篇的 Agent benchmark 是终点。

### 只做评测与选型

```text
8 → 2
```

第八篇可独立阅读；第二篇解释 Arena 的 Bradley-Terry 与 judge 作为奖励时的偏差从哪来。

### Infra 工程师，为 RL 后训练搭系统

```text
3 → 5 → 6
```

PPO / GRPO 的组件与显存账、rollout 与训练的算力配比、on-policy 对权重同步的要求、长思维链带来的长度方差、多轮环境逼出的异步 rollout——这三篇给出系统要满足的算法侧需求。


## 本系列的边界

- **预训练**：tokenizer、scaling law、数据工程、训练配方与稳定性，在 L4 的[《Transformer 与 LLM》](/transformer-and-llm-for-infra-engineers.html)第九到十二篇。本系列从一个训好的基座开始。
- **RL 训练系统的实现**：rollout 引擎与训练器的共置、权重同步、显存切换、算力配比的工程（verl、OpenRLHF、slime 的内部）。属于 Infra 地图的选修。本系列只讲算法对系统的要求与成本量级。
- **推理侧的算法优化**：解码策略、投机解码、量化、KV 压缩。属于 L6，见[《高效推理与压缩（算法侧）》](/efficient-inference-and-compression-for-llms.html)。本系列第五篇的 test-time compute 只讲"用推理算力换准确率"的曲线，不讲怎么让推理更快。
- **多模态后训练**：视觉指令微调、多模态偏好数据。属于 L7，见[《多模态：从视觉编码器到扩散模型》](/multimodal-from-vision-encoders-to-diffusion.html)第三篇。方法与本系列相同，数据与评测不同。
- **安全与对齐的规范性问题**：什么算有害、拒答的边界、红队方法论。本系列只讲把任何一种偏好训进模型的技术，不讨论偏好本身应该是什么。
- **Agent 框架与产品层**：编排框架、记忆、多 Agent 协作、prompt 设计。第六篇只讲怎么用 RL 把工具调用训进模型；在模型之上怎么搭 Agent 属于应用地图。
- **强化学习理论**：MDP、贝尔曼方程、收敛性证明。本系列只推到策略梯度与 PPO 的目标函数，把 LLM 当作"只有最后一步有奖励的单步 bandit"处理。


## 前置要求与说明

### 前置要求

- L4 的[《Transformer 与 LLM》](/transformer-and-llm-for-infra-engineers.html)：参数量与训练状态的账（第一、六篇）、LoRA 的形态（第七篇）、SFT 的 lr 与 warmup 从哪来（第十二篇）；
- L3 的[《深度学习基础》](/deep-learning-foundations.html)：交叉熵与 softmax 的梯度（第一篇）、AdamW 与学习率调度（第三篇）、过拟合与多 epoch（第四篇）；
- L2 导读的逻辑回归、分类器的评估（准确率、置信区间）——奖励模型就是逻辑回归；
- 会用 `transformers` 加载模型与 tokenizer、写一个训练循环；用过或看过 `trl` 的任意一个 trainer 更好。

不要求：学过强化学习（第三篇从策略梯度讲起）；有多卡。

### 版本与基线

- 方法与配方以 **InstructGPT**（2022）、**Llama 3**（2024）、**Tülu 3**（2024）、**DeepSeek-R1**（2025）、**Qwen3**（2025）的公开报告为主要对象，Kimi K2 与 gpt-oss 作为 2025 年后半的对照；
- 实验用 **Qwen2.5-0.5B / 1.5B**（base 与 Instruct），偏好数据用 **UltraFeedback** 的子集，可验证奖励用 **GSM8K**；工具用 `trl`、`peft`、`lm-evaluation-harness`，版本以配套仓库的 `requirements.txt` 为准，均为文章发表前的稳定版；
- 显存与算力的账以 7B / 8B dense 模型、H100 80 GB 为基线（与 L4 系列一致），token 与 GPU 小时为理论估算，用于量级判断；
- 论文引用以第一作者与年份标注。


## 章节目录

1. [SFT：指令数据、chat template、loss mask 与参数高效微调](/sft-data-chat-template-loss-mask-and-peft.html)
2. [偏好数据与奖励模型：Bradley-Terry、pairwise loss 与 reward hacking](/preference-data-and-reward-models.html)
3. [在线 RL：PPO、GRPO 与 RLHF 三件套](/online-rl-ppo-grpo-and-the-rlhf-trio.html)
4. [离线 RL：从 RLHF 目标推出 DPO 及其变体](/offline-rl-dpo-and-its-family.html)
5. [推理模型与可验证奖励：R1 的配方、PRM 与 test-time compute](/reasoning-models-and-verifiable-rewards.html)
6. [Agent 与工具调用的 RL：多轮环境、轨迹数据与延后的奖励](/agentic-rl-tool-use-environments-and-trajectories.html)
7. [蒸馏：logits 级、序列级与 on-policy](/knowledge-distillation-for-llms.html)
8. [评测：benchmark、LLM-as-judge、Arena 与污染](/evaluating-llms-benchmarks-judges-and-contamination.html)


## 最终目标

读完这套系列之后，面对一份后训练配方或一个后训练任务，读者应该能够回答：

```text
这一步的目标函数是什么，每一项在防什么？                    → 第一至七篇：每篇的推导
它用了三件套的哪几个，奖励从哪来，参考怎么约束？              → 第三、四篇的对照表
要几个模型在显存里，生成多少 token，标多少数据？              → 每篇的成本表
奖励模型 70% 的准确率够不够，策略会从哪钻空子？               → 第二篇
PPO 与 GRPO、DPO 与 PPO，什么时候选哪个？                    → 第三、四篇
为什么规则奖励能训出推理而 RM 不能？R1 的四阶段各在修什么？    → 第五篇
多轮工具调用的 RL 与单轮差在哪？瓶颈为什么在环境？            → 第六篇
小模型要能力，蒸馏还是 RL？哪种蒸馏？                        → 第七篇
分数涨了 2 个点，是能力、协议还是污染？judge 的胜率可信吗？    → 第八篇
```

最终目标是三种能力：

1. **推导能力**：面对一个新的后训练方法，能把它放回三件套上，说出它改了哪一项、目标函数差在哪、为什么会更好或更差；
2. **算账能力**：面对一个后训练任务，能在动手之前估出模型数、显存、rollout token、标注量与 GPU 小时，判断团队做不做得起；
3. **判断能力**：面对一次训练的结果，能用正确的协议评测、识别 judge 与污染的偏差、从错误分析里找到下一步改数据还是改配方。

这一层是当前算法工程师的日常。方法的名字还会继续增加，但三件套的骨架、成本的算法与评测的纪律不会变。
