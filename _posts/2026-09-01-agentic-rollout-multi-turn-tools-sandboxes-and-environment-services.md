---
layout: post
series: rl-post-training-infra
title: "RL 后训练基础设施（06）：Agentic rollout——多轮、工具、沙箱与环境服务"
subtitle: "Agentic Rollout: Multi-Turn Trajectories, Tools, Sandboxes and Environment Services"
tags: [RL, verl, Agent, vLLM, Kubernetes, AI, AI-Infra]
catalog: true
updated: 2026-09-14
---

前五篇的 rollout 是一件事：推理引擎批量生成。Agent 训练里它变成了一个分布式系统：模型生成一段，解析出工具调用，某个容器里跑一次测试（几十秒），结果追加进上下文，再生成——重复二十轮；同一时刻几千条这样的轨迹在跑，每条占着一个有状态的沙箱；reward 不是一个规则函数，是"测试通过了几个"。500 个代码任务 × $$G = 8$$ × 20 轮是 **8 万次容器执行**，每次 30 秒就是 **670 个 CPU·小时**——与模型侧这一步的十几个 GPU·小时按价格算是同一量级；要在 30 分钟内跑完，需要 **1300 到 1800 个并发沙箱**，取决于沙箱能否在两轮之间释放。

GPU 这一侧在这 30 分钟里做了什么，是这篇更想回答的问题。答案不太好看：decode 大约 500 秒，其余时间在等环境；同一时刻在生成的请求只有几百条、摊到每张卡几十条，decode 的带宽利用率比单轮 RL 还低；而 **prefill 可能比 decode 更贵**——一条轨迹二十轮、上下文涨到 40K，如果每轮都把整个上下文重新 prefill，8B 模型这一步的 prefill 是 26 EFLOP、1600 秒，与整个窗口一样长；有完美的前缀缓存它是 160 秒。两者之间差十倍，决定因素是**几千条轨迹的 KV 有多少能在两轮之间留在显存里**——而按第一篇的账，留不下多少。

这一篇讨论 rollout 变成"模型与环境交替"之后系统的变化：agent loop 的结构与 token 连续性问题；推理引擎侧的前缀缓存、KV 驻留与粘性路由；沙箱集群——容器的启动、状态、并发、隔离，以及它与 GPU 集群的配比；环境与 reward 的服务化（把 Claude Code、mini-SWE-agent 一类现成的 harness 直接接进 RL）；环境耗时方差带来的长尾与调度；最后把环境的账加进前五篇的 GPU 账。它是第五篇异步的最强动机：环境耗时的方差比生成长度的方差大一个量级，同步形态在这里基本不可用。

本篇的核心问题：

> **500 个代码任务 × $$G = 8$$ × 20 轮，每轮跑一次几十秒的测试。一步 rollout 要多少次容器执行、多少 CPU·小时、并发多少个沙箱才能在 30 分钟内完成？[^q0] GPU 这一侧在这 30 分钟里做了什么？[^q1]**

版本：verl v0.9.0（`verl/experimental/agent_loop/`、`verl/trainer/ppo/v1/agent_loop_tq.py`、`docs/advance/agent_loop.rst`、`reward_loop.rst`、uni-agent）、vLLM v0.27.1 的前缀缓存与 OpenAI 兼容接口。算法侧（Agent RL 的 mask、奖励、轨迹级优势）在[《后训练》第六篇](/agentic-rl-tool-use-environments-and-trajectories.html)，本篇只用它的结论：轨迹是样本，环境 token 不进 loss。

## 一、总览

### 1. 先说答案

500 任务 × $$G = 8$$ = 4000 条轨迹，每条 20 轮，每轮：模型生成约 400 token（约 10 秒的 decode 延迟）→ 测试执行 30 秒 → 返回约 1600 token 的输出。8B 模型、32 张 H100 做 rollout。

```text
环境侧
  容器执行次数          4000 × 20 = 80,000 次
  环境 CPU 时间         80,000 × 30 s = 2.4 M CPU·s = 667 CPU·小时（按每次执行 1 核算；测试常吃 2–4 核，则 1300–2700）
  一条轨迹的墙钟        20 × (10 + 30) s ≈ 800 s（不排队）
  30 分钟内完成所需并发   有状态沙箱（整条轨迹占着）：4000 × 800 s / 1800 s ≈ 1800 个
                        两轮之间可释放（快照 / 无状态）：80,000 × 30 s / 1800 s ≈ 1300 个
  机器                  1800 个沙箱 × 2 vCPU × 4 GB ≈ 45 台 80 核 / 320 GB 的 CPU 机器 —— 与 4 台 8 卡 GPU 机器并列
GPU 侧（32 卡）
  生成 token           4000 × 20 × 400 = 32 M
  decode 时间           32 M / (32 卡 × ~2000 token/s) ≈ 500 s —— 占 30 分钟的 28%
  同时在生成的请求      4000 条轨迹 × (10 / 40) ≈ 1000 条 → 每卡 31 条并发 → decode 步只读 16 GB 权重 + 少量 KV，带宽利用低
  prefill              前缀缓存全命中：Σ 新增 token = 4000 × 40 K = 160 M → 2.6 EFLOP → 160 s
                        全部重算：Σ 每轮整个上下文 = 4000 × 20 × 20 K（平均）= 1.6 B → 26 EFLOP → 1600 s
                        实际：KV 驻留决定，两者之间
  训练要过的 token      4000 × 40 K = 160 M（八成是环境的，不进 loss 但过前向 + 反向）
```

三个结论：

- **环境是第三个池**：Agent RL 的集群里 GPU 旁边有一个与之匹配的 CPU 集群，规模按"沙箱并发数"配，它是配比里的新变量；GPU 池空转的主要原因从"等最长的回答"变成"等最慢的测试"。
- **GPU 侧的账从 decode 主导变成 prefill 与 decode 并列**，且 prefill 的量取决于 KV 能否跨轮驻留——32 张卡的 KV 池 1.8 TB，4000 条 40K 的轨迹要 20 TB，大部分轮次的 KV 会在等环境的 30 秒里被逐出，重新 prefill 是常态。
- **异步是必需**：一条轨迹 800 秒，最慢的（测试超时、重试）几千秒，环境耗时的长尾占比 $$f$$ 轻易到 0.8 以上——第二篇的表里共置同步在这一行是 3000 秒对 600 秒。

### 2. 从批量生成到分布式系统

```text
单轮 RL 的 rollout                            Agent RL 的 rollout
推理引擎 generate(prompts) → 回答                agent loop：generate → 解析工具调用 → 环境执行 → 追加 → generate …
一条样本 = 一次生成                            一条样本 = 一条几十轮的轨迹（几万 token，八成来自环境）
时间由回答长度决定                             时间由环境耗时 × 轮数决定，方差大一个量级
KV 用完即弃                                   KV 要跨轮复用（前缀缓存），否则 prefill 二次增长
reward = 规则 / RM                             reward = 测试结果 / 环境状态 / 生成式评判，本身是服务
参与者：推理引擎                               参与者：推理引擎 · agent loop worker · 工具网关 · 沙箱集群 · reward 服务 · 样本缓冲
```

### 3. 本文的章节安排

| 章 | 内容 | 回答的问题 |
|---|---|---|
| 二 | agent loop 的结构 | 一条轨迹怎样生成，mask、token 连续性、轨迹级样本 |
| 三 | 推理引擎侧 | 前缀缓存与 KV 驻留、粘性路由、低并发 decode、接口形态 |
| 四 | 沙箱集群 | 容器的启动 / 状态 / 隔离 / 并发，与 GPU 的配比 |
| 五 | 环境与 reward 的服务化 | 网关、黑盒 agent、三种 reward 形态怎样进同一条通路 |
| 六 | 长尾与调度 | 环境方差、超时 / 重试 / 丢弃、部分 rollout 的轮边界 |
| 七 | 环境账与 GPU 账 | 把 CPU·小时与沙箱并发加进账本 |
| 八 | verl 的实现 | AgentLoop、LLMServerClient、RewardLoop、uni-agent |
| 九 | 小结 | 要点、速查表、下一篇 |

## 二、agent loop 的结构

### 1. 一条轨迹

```text
轮 1   prompt（任务描述 + 仓库信息，~2K）→ 模型生成（思考 + 工具调用，~400）→ 解析：run_tests(...) → 沙箱执行 30 s → 输出 ~1600 token 追加
轮 2   上下文 4K → 生成 ~400 → edit_file(...) → 执行 0.1 s → 追加 ~100
轮 3   上下文 4.5K → 生成 → run_tests → 30 s → 追加 ~1600
…
轮 20  上下文 ~40K → 生成 "任务完成" → 结束
reward：最后一次测试通过的比例 / 是否全部通过 → 0 或 1（或分数）
```

轨迹是**一条样本**：`prompt_ids`（首轮的 prompt）、`response_ids`（之后全部 token，模型生成的与环境返回的交错）、`response_mask`（1 = 模型生成、0 = 环境返回）。训练时 loss 只算 mask = 1 的位置，但前向与反向要过全部位置——第一篇算过，每个有效训练 token 的代价是单轮的约 5 倍。GRPO 的组是同一任务的 $$G$$ 条轨迹，优势按组内归一化，与单轮相同。

### 2. token 连续性

一个不显眼但反复出问题的细节：**环境返回的是文本，模型消费的是 token**。把工具输出用 chat template 包装成一条 `tool` 消息、再 tokenize 追加进上下文，与"模型在生成时看到的 token 序列"必须一致，否则 $$\log \pi_{old}$$ 对不上位置——训练时的 token 序列与采样时的不是同一个序列。三个具体的坑：

- **重新 tokenize 不幂等**：把已生成的 token 解码成文本、拼上工具输出、再整体 tokenize，边界处的 token 可能合并或拆分（`"...}\n"` 与 `"<|im_end|>"` 之间），得到与原来不同的 id 序列。正确做法是**只 tokenize 新增部分、直接拼接 id**，绝不整体重编。
- **chat template 的轮边界**：Qwen、GLM、MiniMax 各家的模板在 assistant 输出与 tool 消息之间插入的特殊 token 不同，有的还要在 assistant 段末尾补 `<|im_end|>\n`；模板写错一个换行，训练分布就带上一个系统偏移。verl 0.9 的 **Continuous Token** 机制（`continuous_token_wiring`）就是把这层做成可复用的 builder，按模型家族处理边界，默认关闭、需要显式打开。
- **思考内容的处理**：推理模型的 `<think>` 段在多轮里是保留在上下文还是丢弃（Qwen3 的模板默认只保留最后一轮的思考），决定了上下文长度与训练时看到的序列——要与采样时一致。

这些问题在单轮 RL 里不存在（一次生成、无需拼接），在 Agent RL 里是第五篇"训推不一致"的另一个来源：不是数值上的不一致，是**序列本身不一致**。

### 3. 每条轨迹一个协程

verl 的实现形状（`docs/advance/agent_loop.rst`）：`AgentLoopManager` 把一个 batch 的 prompt 切成块发给若干 `AgentLoopWorker`（Ray actor，CPU 进程）；每个 worker 为每条轨迹起一个 `AgentLoopBase.run()` **协程**，协程里 `await llm_client.generate(...)`、`await tool.execute(...)`，几千条轨迹是几千个协程在几十个 worker 进程上并发。用户只实现 `run()`——"给一个 prompt，怎样跑到底"——工具、环境、反思、多 agent 全是它内部的事；框架负责 LLM 服务的负载均衡与样本的收集。

这个结构决定了 agent loop worker 是 CPU 侧的**协调者**而不是执行者：它不跑测试，它调沙箱服务；它不做推理，它调 LLM 服务。它的瓶颈是协程数与 Python 的异步调度——几千条轨迹、每条每轮几次 await，单个 worker 进程扛几百条是常见上限，所以 `num_workers` 要按并发轨迹数配。

## 三、推理引擎侧

### 1. 前缀缓存：从二次到线性

第 $$i$$ 轮的上下文 = 前 $$i - 1$$ 轮的全部 + 本轮新增。没有前缀缓存，每轮 prefill 整个上下文：一条轨迹的总 prefill token 是 $$\sum_i C_i \approx$$ 轮数 × 平均上下文 = $$20 \times 20\text{K} = 400\text{K}$$；有前缀缓存（前 $$i - 1$$ 轮的 KV 还在），只 prefill 新增：总量 = 最终上下文 $$40\text{K}$$。**十倍**。

第一篇说过单轮 RL 里前缀缓存"省显存不省 FLOP"，Agent RL 里它是 FLOP 的决定因素：4000 条轨迹 × 400K = 1.6B token 的 prefill = 26 EFLOP（8B），32 卡 MFU 0.5 下 1600 秒；线性时 160 秒。

### 2. KV 驻留：缓存能命中多少

前缀缓存命中的前提是**KV 块还在显存里**。一条 40K 的轨迹 5 GB KV（8B、128 KiB/token），4000 条在飞轨迹 20 TB；32 张卡按共置 0.85 或独占 0.9 的 KV 池 1.6–1.8 TB，**只够 350 条轨迹的完整上下文**。而每条轨迹在两轮之间有 30 秒在等环境，这 30 秒里它的 KV 块占着显存不产生任何 token——vLLM 的前缀缓存是 LRU 的，等待期间新请求进来就把它逐出了。

粗估命中率：一张卡的 KV 池 50 GB，同一时刻挂在它上面的轨迹 125 条（4000 / 32），每条平均上下文 20K = 2.5 GB，要全部驻留需要 310 GB——**六倍于池子**。稳态下大约六分之一的轨迹能命中上一轮的 KV，其余五分之六重新 prefill：prefill 总量约 $$160\text{M} + \frac{5}{6} \times 1.44\text{B} \approx 1.36\text{B}$$ token，约 1400 秒。也就是说**默认配置下 Agent RL 的 prefill 接近"没有前缀缓存"的账**。

四个改善的方向，各有代价：

```text
做法                                  效果                              代价
减少每张卡的在飞轨迹（多给 rollout 卡）  驻留比例 ∝ 卡数 / 轨迹数            GPU 更多，decode 并发更低
KV 卸载到 CPU / 外部存储               等待期间 KV 搬到主机内存，下一轮搬回    每轮 2.5 GB 的 PCIe 往返（0.1 s，相对 30 s 的等待可忽略）；
（vLLM 的 CPU offload、LMCache、Mooncake store connector）                     主机内存 20 TB / 32 卡 = 每台机 5 TB —— 通常放不下全部，但能放下大半
更短的上下文（截断工具输出、摘要历史）    上下文 40K → 15K，KV 与 prefill 都降    改变任务；训练目标里要包含"学会精简"
接受重 prefill，多给 prefill 算力        —                                 prefill 是 compute-bound、MFU 高；1400 s 的 prefill 在 64 卡上 700 s
```

第二条是这两年推理服务侧的主流方向（[《大模型推理系统揭秘》](/deep-dive-into-vllm.html)讨论过 KV 的分层存储），verl 的 `reset_prefix_cache(reset_connector=True)` 里的 `connector` 就是为外挂 KV 存储留的口——权重换了，外挂存储里的 KV 也要清。

### 3. 粘性路由

前缀缓存只在**同一个推理实例**上有效——KV 在那张卡的显存里。所以一条轨迹的所有轮次必须路由到同一个实例：verl 的 `LLMServerClient` 第一轮选最空的实例、之后各轮**粘到**同一个实例（sticky session）。三个后果：

- 负载不均：实例的负载由"落在它上面的轨迹在哪一阶段"决定，等环境的轨迹不占算力但占 KV；实例间的并发差异可达几倍。
- 与部分 rollout 的冲突：权重同步时 abort 后重提，若路由到别的实例，前缀缓存全丢；verl 的动态调度里 `enable_rebalance` 会清掉粘性缓存、让重提的请求重新分配——那是主动放弃缓存换负载均衡。
- 实例故障：一个实例挂了，它上面几百条轨迹的缓存全丢，重提到别处全部重 prefill——异构弹性 rollout（第四篇 NIXL 那条路）的代价之一。

### 4. 低并发的 decode

同一时刻只有约四分之一的轨迹在生成（10 秒生成 / 40 秒一轮），32 张卡上 1000 条并发请求、每卡 31 条。回到第一篇的 decode 模型：每步读 16 GB 权重 + 31 × 20K × 128 KiB ≈ 80 GB 的 KV，共 96 GB / 2 TB/s ≈ 48 毫秒，出 31 个 token → **每卡 650 token/s**，是单轮 RL（2600）的四分之一。KV 读取占了每步的 80%——长上下文让 Agent RL 的 decode 是"KV 带宽受限"而不是"权重带宽受限"。MLA 或 KV 量化在这里的收益比单轮 RL 大得多。

### 5. 接口形态

推理引擎以**服务**形态存在（第二篇的 server 模式），两种接口：

- **token in / token out**（`generate(prompt_ids, sampling_params)`）：agent loop 自己维护 token 序列，拼接、mask、logprob 全在 loop 侧，训练所需的一切都精确——verl 内置的 tool agent loop 走这条路。
- **OpenAI 兼容的 chat completion**：接受 messages、返回文本。它的意义是**接入现成的 agent 框架**——Claude Code、mini-SWE-agent、OpenHands 一类 harness 只会说 OpenAI / Anthropic 的协议，不会给你 token id。代价是 token 连续性（第二章第 2 节）要由服务侧重建：网关记录每次请求实际用的 token 序列与 logprob，事后拼成训练样本。verl 的 uni-agent gateway 做的正是这层。

## 四、沙箱集群

### 1. 一次执行的成本

代码任务的一次工具调用是"在某个容器里跑一条命令"：

```text
环节                    时间                          说明
容器启动（冷）           1–5 s                          拉镜像、创建 namespace、挂载
容器启动（热池）         50–300 ms                      预先起好、任务到了分配；Firecracker / gVisor 的 microVM 更慢一些
命令执行                 0.1 s（cat 一个文件）– 几分钟（跑全部测试）   方差的来源
输出收集与截断           ms                            测试输出几 MB 要截到几 K token
状态保持                 整条轨迹（20 轮 × 40 s = 800 s） 有状态：文件系统改动要跨轮保留
回收                     100 ms – 1 s                   销毁 namespace、释放存储
```

**状态**是沙箱与"无状态函数"的根本差别：轮 2 改了文件、轮 3 跑测试要看到改动，所以一条轨迹的沙箱从第一轮到最后一轮都要活着——等环境的 30 秒里沙箱在跑，等模型生成的 10 秒里沙箱在闲。要释放它得做快照（文件系统层的 overlay 提交 + 进程状态，几百毫秒到几秒），下一轮恢复。第一章那 1800 对 1300 的差就是"整条占着"与"两轮之间释放"的差。

### 2. 隔离与安全

RL 训练里跑的是**模型自己写的代码**，且训练目标是让它拿到高 reward——模型会找到任何能让测试通过的路径，包括改测试文件、mock 掉断言、读环境变量里的答案、访问网络查现成解。沙箱的隔离要求因此高于普通 CI：

- 文件系统：测试文件只读、或 reward 计算用一份模型碰不到的副本；
- 网络：默认断网，需要的依赖预装进镜像；
- 资源：cgroup 限 CPU / 内存 / 进程数（fork 炸弹是常见的"意外"）、超时强杀；
- 内核：容器共享宿主内核，gVisor / Firecracker / Kata 一类用户态内核或 microVM 把它隔开——多 5–20% 的执行开销与更长的启动。

reward hacking 在算法侧是奖励设计问题，在系统侧是**沙箱的边界就是 reward 的边界**：沙箱能被绕过的地方，就是 reward 会被刷的地方。

### 3. 并发与配比

1800 个并发沙箱、每个 2 vCPU + 4 GB，是 3600 核、7 TB 内存——约 45 台 80 核机器。与 4 台 8 卡 GPU 机器（32 卡 rollout）并列，**CPU 机器数是 GPU 机器数的十倍**。按价格（H100 机器约为 80 核 CPU 机器的 10–15 倍），两边同量级。这个比例随任务变：

```text
任务类型                 每轮环境耗时      每轨迹轮数    环境 : 模型（墙钟）    沙箱 : GPU（并发比）
数学（规则验证）          ms              1            ≈ 0               不需要沙箱
简单工具（搜索、计算器）   0.1–1 s          3–5          1 : 5             每卡几十个轻量进程
代码（单元测试）          10–60 s          15–30        3 : 1             每卡 50–60 个容器
SWE-bench 类（完整测试套件） 1–10 min      20–50        10 : 1            每卡 100+ 个容器，单条轨迹小时级
浏览器 / GUI              1–5 s（渲染）    20–100       2 : 1             每个环境 1–2 GB 内存（浏览器）
```

沙箱并发数成了**第三个配比变量**：rollout 卡数、训练卡数、沙箱数三者要让"生成、环境、训练"三段的吞吐匹配。沙箱不够，GPU 上的轨迹卡在"等沙箱分配"；沙箱太多，CPU 集群空转。它比 GPU 配比好调——CPU 机器可以按分钟弹性伸缩，Kubernetes 的 HPA 按排队深度扩容是现成的。

### 4. 镜像与冷启动

每个任务（SWE-bench 的每个 issue）有自己的仓库、依赖、Python 版本——**镜像是按任务的**，几千个任务几千个镜像、每个几 GB。每步 rollout 要在几十台机器上各起几十个容器，镜像分发是启动时间的大头：预热（把本步会用到的镜像提前拉到节点）、分层共享（基础层复用）、镜像仓库的带宽（几十台机器同时拉 5 GB 是几百 GB 的流量）。热池只能对"通用镜像"预起，任务专属的镜像只能预拉。SWE-bench 规模的训练里，**镜像管理是沙箱平台的主要工程量**，不是容器运行时。

## 五、环境与 reward 的服务化

### 1. 为什么是服务

agent loop 需要环境，训练器需要 reward，推理引擎需要都不知道。把环境（沙箱）与 reward 做成独立服务、agent loop 通过 HTTP / gRPC 调用，三个理由：

- **独立扩缩与故障域**：沙箱集群挂一台机器不影响 GPU 任务；GPU 任务重启不用重建沙箱池。
- **复用**：同一个环境服务给 RL 训练、评测、SFT 数据生成用；沙箱平台是公司级基础设施，不是某个训练任务的一部分。
- **接现成的 agent**：服务化的另一面是**模型也是服务**——推理引擎以 OpenAI / Anthropic 兼容接口暴露，任何现成的 harness 都能连上来。

### 2. 黑盒 agent 与网关

verl 0.9 的 **uni-agent** 走到了这一步：一个网关（Uni-Agent Gateway）在推理引擎前面，对外是 OpenAI / Anthropic 兼容的 endpoint；Claude Code、mini-SWE-agent 或任何说这两种协议的 harness 直接把它当模型用；网关记录每个会话里每次请求的 token 序列、采样的 logprob、版本号，会话结束后拼成一条训练轨迹进样本缓冲。release note 的数字是 **1000+ 个并发有状态会话**。

它解决的是"训练用的 agent 与部署用的 agent 是同一个"——训练时的 prompt 组织、工具定义、重试逻辑与线上一致，训练分布不漂移。代价在网关：

- token 连续性由网关重建（第二章第 2 节的全部问题），且 harness 可能在轮与轮之间**改写**历史（压缩上下文、删旧消息），网关看到的是一串不连续的请求，要决定哪些能拼成一条轨迹；
- reward 在 harness 之外算（测试结果），要与会话对上；
- harness 的行为（重试、并行工具调用、子 agent）让"一条轨迹"的定义变模糊——子 agent 的调用是同一条轨迹的分支还是另一条样本？

slime 的 `slime/agent` 模块、AReaL 的 `RolloutWorkflow`（`arun_episode`）是同一方向的不同形态：前者把 agent 逻辑作为 rollout 函数插进 Data Buffer 通路，后者把它作为工作流类。三者共同的结论是：**agent 逻辑不属于训练框架**，框架只提供"给我一条轨迹"的接口。

### 3. 三种 reward，一条通路

```text
形态              在哪算                    时间           方差       与 rollout 的关系
规则验证器        CPU 进程（agent loop worker 或 reward worker）   ms    小        轨迹结束立刻算
代码执行 / 环境状态  沙箱                    s – min        大        就是最后一轮的环境调用；或在轨迹结束后再跑一次隐藏测试
生成式 RM / 评判模型  另一个 LLM 推理服务      s              中        需要 GPU：独立池或与 rollout 共置；本身是一个 serving 作业
```

verl 的 **Reward Loop**（`RewardLoopManager` + 若干 `RewardWorker`）把三种做成同一个接口：轨迹完成 → 分块发给 reward worker 并行算 → 结果写回样本。生成式 RM 的推理实例由 `reward.reward_model.enable_resource_pool` 决定是独立池还是共置——`separate_async` 模式**要求独立池**，因为 standalone rollout 实例从不暂停、没有空闲显存给 RM 用（第二篇 `PPOTrainerSeparateAsync.__init__` 里的断言）。

生成式 RM 是第四个 GPU 负载：它也是 decode（评判要生成理由）或 prefill（打分只要一次前向），也有自己的权重（可能是另一个模型，不需要同步；也可能就是策略自己——self-reward，那就要同步）。规模上它常与 rollout 同量级（每条轨迹评一次、输入是整条轨迹 40K token），在 GPU 配比里要单独算一列。

## 六、长尾与调度

### 1. 环境方差

单轮 RL 的长尾来自回答长度（$$L_{max} / \bar L = 4$$–10）；Agent RL 的长尾来自环境：

```text
来源                          典型                    极端
单次测试的执行时间             30 s                    超时 10 min（死循环、等网络）
轮数                          20                      到上限 50（模型反复试错）
沙箱排队                       0                       几分钟（并发不够、镜像冷）
重试                          0                       3 次（沙箱崩、网络抖）
一条轨迹                       800 s                   1–2 小时
```

$$f$$（长尾占比）在这里轻易到 0.8 以上：一步里 95% 的轨迹 15 分钟完成，最后 5% 卡在超时与重试上再花 30 分钟。**同步形态下 GPU 在这 30 分钟里几乎全空**。第二篇那张表的最后两行——异步 / 共置 2.5 到 4.9 倍——就是这个场景。

### 2. 轨迹级的超时、重试与丢弃

```text
层级          机制                                    落点
单次工具调用   超时（如 120 s）→ 返回 "timeout" 给模型，轨迹继续   沙箱服务
单轮生成       max_tokens；生成被 abort（权重同步）→ 续接        推理引擎 / agent loop
轨迹          最大轮数；总时长上限 → 强制结束、按当前状态算 reward  agent loop
组            G 条里有失败 → failure 组 → 淘汰 / padding / 补发（第五篇）  replay buffer
步            缓冲里终态组够了就训练，不等剩下的                 trainer
```

每一层的选择都是分布上的取舍：超时返回给模型让它"学会处理超时"（但训练数据里多了超时样本）；强制结束的轨迹 reward 通常是 0，会让模型学到"别做太多轮"（可能是好事也可能不是）；failure 组丢弃偏向"不出错的任务"。第五篇的诊断表在这里要加一列：**各层超时 / 重试 / 丢弃的计数与它们的任务分布**。

### 3. 部分 rollout 只能在轮边界

第五篇的部分 rollout 在这里受限：权重同步要 abort 在飞请求，但一条 Agent 轨迹的"在飞"可能是**沙箱正在跑测试**——推理引擎侧没有请求可 abort。可行的边界只有"一轮生成的中间"（abort 生成、保存 token、续接——与单轮相同）和"轮与轮之间"（等本轮工具返回、下一轮用新权重）；沙箱执行本身不能中断续接（状态在容器里）。所以 Agent RL 里一条轨迹跨的版本数 ≈ 轨迹墙钟 / 同步间隔 = 800 / 150 ≈ 5，比单轮 RL 大，staleness 的上限要放宽、decoupled loss 更接近必需。

### 4. 调度的两个目标

agent loop 的调度器同时要满足：**沙箱利用率**（不让容器闲着等模型）与 **GPU 利用率**（不让推理引擎闲着等环境）。两者的自然节律相反（一条轨迹在两侧交替），靠**多路并发**填平——在飞轨迹数是沙箱数的 1.3 倍左右（一条在生成时另 0.3 条在等）、是 GPU 并发能力的 4 倍左右。这个比例随任务的"环境 : 模型"墙钟比（第四章第 3 节的表）变，也随训练变（模型变强、轮数变少、测试更快通过）。

## 七、环境账与 GPU 账

### 1. 把环境加进账本

第一篇的三本账加一本：

```text
环境账（一步）
  执行次数        N_exec = B · G · 轮数
  CPU 时间        Σ 执行时间 × 每次核数                       8 万次 × 30 s × 2 核 = 1300 CPU·h
  沙箱·秒         有状态：B · G · 轨迹墙钟；可释放：Σ 执行时间     3.2 M / 2.4 M 沙箱·s
  所需并发        沙箱·秒 / 目标墙钟                           1800 / 1300
  镜像流量        节点数 × 本步任务镜像的总大小
GPU 账的变化
  decode          B · G · 轮数 · 每轮生成 token —— 与单轮同公式，但并发低、KV 长 → 每卡 token/s 降 3–4 倍
  prefill         线性（全命中）到二次（全重算），由 KV 驻留决定；常接近二次
  训练 token      B · G · 最终上下文，八成 mask = 0
  生成式 RM       + B · G · 最终上下文 的一次前向或一次生成
```

### 2. 核心问题的数字

```text
                         值                         说明
容器执行                  80,000 次
CPU·小时                  667（1 核）/ 1300（2 核）
并发沙箱（30 分钟）        1800（有状态）/ 1300（可释放）
GPU（32 卡）在 30 分钟里    decode ~500 s（28%）· prefill 160–1600 s（视 KV 驻留，常在 1000 以上）· 其余等环境
                         → 同步形态下 GPU 有效利用 < 30%；异步形态下 rollout 卡持续有活，训练卡在另一池
训练侧                    160 M token 的前向 + 反向 + 参考 + 旧策略 ≈ 12N × 160 M = 15 EFLOP（8B）→ 32 卡 MFU 0.4 约 1200 s
```

一个意外的结论：**这一步训练侧的 FLOP（15 EFLOP）比单轮 8B 推理场景（6.8 EFLOP）还多一倍**——因为要过 160M 个 token 的前向 + 反向，虽然只有两成有梯度。Agent RL 不是"生成占 90%"，是"三段都重、外加一个 CPU 集群"。

### 3. 配比

三个池（rollout GPU、训练 GPU、沙箱 CPU）的配比原则仍是"三段吞吐相等"：

$$\frac{\text{rollout 卡数}}{\text{训练卡数}} = \frac{T_{gen+prefill}}{T_{train}}, \qquad \text{沙箱数} = \frac{B \cdot G \cdot \text{轨迹墙钟}}{\text{目标步时间}}$$

前者在 Agent RL 里比单轮 RL 更接近 1 : 1（prefill 与训练 token 都涨了），后者是新增的、且最容易随训练漂移（轮数与测试时长都在变）。

## 八、verl 的实现

### 1. 落点

```text
组件                        路径                                                 说明
AgentLoopBase / run()       verl/experimental/agent_loop/agent_loop.py            用户实现；返回 AgentLoopOutput(prompt_ids, response_ids, response_mask, …)
内置 loop                   single_turn_agent_loop.py · tool_agent_loop.py        单轮；多轮工具调用（tool_parser 解析 <tool_call>）
AgentLoopManager / Worker    agent_loop.py；v1：trainer/ppo/v1/agent_loop_tq.py    manager 切 batch、worker 起协程；v1 版把输出直接写 TransferQueue
LLMServerClient             workers/rollout/llm_server.py                          首轮最空实例、之后粘性；FullyAsyncLLMServerClient 带重试（abort 后重提）
AsyncLLMServer              vllm_rollout/vllm_async_server.py · sglang_rollout/     chat_completion（OpenAI）与 generate（token in/out）两个接口
Continuous Token            utils/tokenizer/continuous_token_wiring.py            轮边界的 token 连续性 builder，按模型家族；默认关
RewardLoopManager / Worker   workers/reward_manager/ · docs/advance/reward_loop.rst  规则 / 沙箱 / 生成式 RM 同一接口；num_workers 并行；RM 独立池或共置
uni-agent                   独立仓库 verl-project/uni-agent                        网关 + 会话记录 + 轨迹重建；1000+ 并发会话
工具定义                    rollout.multi_turn.tool_config_path                    OpenAI function schema；工具实现继承 BaseTool（create / execute / calc_reward / release）
```

`BaseTool` 的四个方法值得看：`create`（为一条轨迹建实例——这里起沙箱）、`execute`（一次调用）、`calc_reward`（工具侧的 reward，如测试通过数）、`release`（轨迹结束回收）。沙箱的生命周期正好对应这四个点。

### 2. 一条轨迹在 v1 里的路

```text
trainer._add_batch_to_generate → AgentLoopManagerTQ 取 prompt → AgentLoopWorker.run(协程)
  → 轮 i：llm_client.generate(prompt_ids) → [粘性] AsyncLLMServer → vLLM engine（前缀缓存命中与否）
        → tool_parser 解析 → tool.execute（沙箱服务 HTTP）→ 追加 token（continuous token builder）
  → 结束：tool.calc_reward / reward loop → AgentLoopOutput（带每段的版本、logprob）
  → 写 TransferQueue：{uid}_{session}_{index}，tag status=finished
→ ReplayBuffer 看到组终态 → trainer 取 batch → 训练
```

与单轮 RL 唯一不同的是中间那几行——推理引擎、缓冲、训练器都不知道这是 Agent 轨迹。这是 agent loop 抽象的价值：**Agent RL 对基础设施的新要求全部落在 rollout 一侧**，形态（第二篇）、显存（第三篇）、同步（第四篇）、异步（第五篇）的机制原样适用。

### 3. 多模态

VLM 的 Agent（GUI 操作、网页浏览）在这条路上多两样：环境返回的是**图像**（截图），进上下文前要过 vision encoder（推理引擎的 `reset_mm_cache` / `reset_encoder_cache` 在权重同步后也要清）；图像 token 多（一张截图几百到上千 token），上下文涨得更快、KV 驻留更差。系统形态不变，账更重。

## 九、本文小结

### 1. 要点回顾

- Agent RL 的 rollout 是一个分布式系统：推理服务 + agent loop 协程 + 工具网关 + 沙箱集群 + reward 服务 + 样本缓冲；一条样本是几十轮、几万 token 的轨迹，八成 token 来自环境（`response_mask = 0`，不进 loss 但过前向 + 反向）。
- **token 连续性**是 Agent RL 特有的训推不一致：只拼接新增 token、不整体重编，chat template 的轮边界按模型家族处理（verl 的 Continuous Token）。
- 推理引擎侧：前缀缓存把 prefill 从二次降到线性（十倍），但**KV 驻留**决定命中率——几千条 40K 轨迹的 KV 是 KV 池的十倍以上，等环境的几十秒里被逐出，默认配置下 prefill 接近"没有缓存"的账；KV 卸载到主机内存 / 外部存储是主要出路；粘性路由是前提也是负载不均的来源；低并发 + 长上下文让 decode 变成 KV 带宽受限，每卡 token/s 降 3–4 倍。
- 沙箱集群：有状态（整条轨迹占着）、隔离要求高于 CI（模型会刷 reward，沙箱边界就是 reward 边界）、镜像按任务（分发是主要工程量）；**沙箱并发数是第三个配比变量**，CPU 机器数常是 GPU 机器数的十倍、成本同量级。
- 服务化：环境与 reward 做成服务，模型也做成服务（OpenAI / Anthropic 兼容），于是现成的 harness（Claude Code、mini-SWE-agent）能直接接进训练——uni-agent 的网关记录会话、重建轨迹，1000+ 并发；三种 reward（规则 / 沙箱 / 生成式 RM）走同一个 Reward Loop，生成式 RM 是第四个 GPU 负载、`separate_async` 下必须独立池。
- 长尾来自环境方差（超时、轮数、排队、重试），$$f > 0.8$$，同步形态不可用；超时 / 重试 / 丢弃每一层都是分布上的取舍；部分 rollout 只能在生成中或轮边界，沙箱执行不可中断，所以轨迹跨的版本数更多、decoupled loss 接近必需。
- 核心问题的数字：8 万次执行、667–1300 CPU·h、1300–1800 并发沙箱；32 张卡 30 分钟里 decode 500 秒、prefill 160–1600 秒（常在 1000 以上）、其余等环境；训练侧 15 EFLOP——**三段都重，外加一个 CPU 集群**。

### 2. 速查表

```text
环境账          执行次数 B·G·轮数 · CPU·h = Σ 执行时间 × 核数 · 沙箱并发 = 沙箱·秒 / 目标墙钟（有状态：B·G·轨迹墙钟）
prefill         全命中：B·G·最终上下文 · 全重算：B·G·轮数·平均上下文（≈ 10×）· 实际由 KV 驻留决定
KV 驻留         需要 = 在飞轨迹 × 平均上下文 × k_kv；池子 = 卡数 × KV 池；比值 > 1 时按比例重 prefill
decode          并发 = 在飞轨迹 × (生成时间 / 一轮时间)；每步读 权重 + 并发 × 上下文 × k_kv → 长上下文下 KV 主导
配比            rollout : train ≈ T_gen+prefill : T_train（接近 1 : 1）· 沙箱数独立配 · 生成式 RM 单独一列
长尾            f = 环境长尾 / 总 → 0.8+；部分 rollout 边界 = 生成中 / 轮间；轨迹跨版本 ≈ 轨迹墙钟 / T_sync
接口            token in/out（精确）· OpenAI/Anthropic 兼容（接现成 harness，网关重建轨迹）
verl            AgentLoopBase.run · AgentLoopManagerTQ · LLMServerClient（粘性）· BaseTool(create/execute/calc_reward/release) · RewardLoop · uni-agent
```

### 3. 下一篇

前六篇把机制讲完了：账、形态、显存切换、权重同步、异步、环境。每一个机制在 verl 里都有实现，但到现在为止只是"落点"——一个文件名、一个函数名。下一篇从 `main_ppo` 的入口开始，按"一个配置文件怎样变成一组 Ray worker、一步训练怎样在它们之间流动"的顺序把这些落点串成一条线：单控制器、worker 组与分发装饰器、资源池与共置、rollout server、checkpoint engine、replay buffer、agent loop；末尾用一节看 slime（少一层抽象）与 AReaL（异步优先）在这条线的哪几段做了不同的选择：

> **一个 bf16 参数从优化器更新完成，到推理引擎用它生成下一个 token，在 verl 里经过哪些函数、哪些进程、哪条链路？把这条链追清楚，前六篇的机制就全部落到了代码上；再问 slime 和 AReaL 在这条链的哪一段做了不同的选择，就知道哪些是必然、哪些是取舍。**

下一篇：verl 源码导读——从一个 GRPO 配置追到每个 worker。

**实践建议**：不需要 SWE-bench 规模。用 verl 的 `tool_agent_loop` 接一个最小的沙箱工具（一个容器池 + 一个 `execute` 跑 `python -c`），在 8 卡上用小模型跑几十步的多轮 RL；记三条曲线：每步的环境时间分布（各轨迹的墙钟直方图）、推理引擎的前缀缓存命中率（vLLM 日志里周期打印的 prefix cache hit rate，或 Prometheus 的 `vllm:prefix_cache_hits` / `vllm:prefix_cache_queries`）、GPU 的空闲比例（`rollouter/idle_ratio` 或 nvidia-smi 的采样）。然后把在飞轨迹数翻倍，看命中率掉多少、prefill 时间涨多少——那就是本篇第三章第 2 节的账在你的配置下的样子。

## 十、自测

1. 500 任务 × $$G = 8$$ × 20 轮、每次执行 30 秒 × 4 核：多少次执行、多少 CPU·小时？30 分钟内完成要多少并发沙箱（无状态与有状态各算）？

   <details markdown="1"><summary>答案</summary>

   8 万次执行；$$80000 \times 30 \times 4 / 3600 \approx 2667$$ 核·小时（文中按核数不同给 667–1300）；无状态：$$80000 \times 30 / 1800 \approx 1333$$ 个；有状态（整条轨迹占一个）：在飞轨迹 4000 条 × 轨迹墙钟 / 30 分钟，常在 1300–1800。

   </details>

2. 一条 20 轮、最终 40K token 的轨迹，前缀缓存全命中与全重算的 prefill 各多少 token？实际由什么决定？

   <details markdown="1"><summary>答案</summary>

   全命中：只 prefill 每轮新增的部分，合计 ≈ 40K；全重算：每轮 prefill 全部历史 $$\sum$$ 平均上下文 ≈ 20 × 20K = 400K，10 倍；实际由 KV 驻留决定——在飞轨迹 × 平均上下文 × $$k_{kv}$$ 与 KV 池的比值 > 1 时按比例重 prefill。

   </details>

3. 为什么 Agent 训练里“低并发 + 长上下文”让 decode 每卡 token/s 降 3–4 倍？

   <details markdown="1"><summary>答案</summary>

   每步读的字节 = 权重 + 并发 × 上下文 × $$k_{kv}$$；上下文 40K 时 KV 项主导，而并发被 KV 池与等待环境的空档压低——每步时间被 KV 读取撑大、同时产出的 token 又少，吞吐降 3–4 倍。

   </details>

4. “沙箱边界就是 reward 边界”是什么意思？对隔离提出了什么高于 CI 的要求？

   <details markdown="1"><summary>答案</summary>

   reward 来自沙箱里的测试结果，模型能碰到的一切都可能被用来刷分——改测试文件、读隐藏答案、伪造输出；所以隐藏测试只读且不可见、网络隔离、文件系统与进程隔离、执行结果由沙箱外的判定器读取，任何一处漏就是 reward hacking 的入口。

   </details>

5. 为什么 Agent RL 里同步形态几乎不可用、decoupled loss 接近必需？

   <details markdown="1"><summary>答案</summary>

   长尾来自环境方差（超时、轮数、排队、重试），$$f > 0.8$$——同步形态下 GPU 大部分时间在等最慢的环境；异步后一条轨迹几十分钟、跨多个权重版本，部分 rollout 只能在轮边界（沙箱执行不可中断），staleness 大且分段，三份 logprob 的 decoupled 修正才能保住训练信号。

   </details>

[^q0]: 执行次数 $$B \times G \times \text{轮数} = 500 \times 8 \times 20 = 8$$ **万次**容器执行；每次几十秒 × 核数，合计 **667–1300 CPU·小时**；要在 30 分钟内完成，沙箱并发 = 沙箱·秒 / 目标墙钟 ≈ **1300–1800 个**——而且沙箱是有状态的（整条轨迹占着一个），并发数由在飞轨迹数决定；CPU 机器数常是 GPU 机器数的十倍、成本同量级——沙箱并发是第三个配比变量。详见[第四章](#四沙箱集群)、[第七章](#七环境账与-gpu-账)。
[^q1]: 32 张卡在这 30 分钟里 decode 只有约 500 秒（每轮生成几百 token、4000 条轨迹分批）；prefill 160–1600 秒（常在 1000 以上）：一条 40K 的轨迹每轮要 prefill 全部历史，前缀缓存能把它从二次降到线性，但 **KV 驻留**决定命中率——几千条 40K 轨迹的 KV 是 KV 池的十倍以上，等环境的几十秒里被逐出，默认配置下 prefill 接近「没有缓存」的账，KV 卸载到主机内存 / 外部存储是主要出路；其余时间在**等环境**。训练侧 15 EFLOP——八成 token 是环境返回的（`response_mask = 0`，不进 loss 但过前向反向）。详见[第三章](#三推理引擎侧)、[第六章](#六长尾与调度)、[第七章](#七环境账与-gpu-账)。
