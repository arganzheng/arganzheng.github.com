---
layout: post
title: 大模型推理系统揭秘（07）：解码的扩展：采样、投机解码与结构化输出
tags: [AI, AI-Infra, 大模型推理]
catalog: true
---

> 本文是[《大模型推理系统揭秘：从 vLLM 看 LLM Serving Infra 核心技术》](/deep-dive-into-vllm.html)系列的第 7 篇（共十四篇）。上一篇：[GPU 执行：如何让每个 Token 算得更快？](/gpu-execution-kernels-and-graphs.html)；下一篇：[Multi-GPU：一张卡不够时如何扩展？](/multi-gpu-scaling-strategies.html)

> **NOTE** 本文基于 vLLM v0.27.1（tag `6e448d0`, 2026-08-11）源码剖析。文中文件路径、类名和函数名均以该版本为准；vLLM 迭代很快，阅读时请以你手上的版本对照。


上一篇把一轮 batch 送进 GPU，跑完 forward，拿到 logits，采样出 token——到这里，"下一个 token"似乎就是一次 `argmax` 或一次 `multinomial`。但在真实的服务里，这一步几乎从来不是那么干净的：一个请求要 `top_p=0.9` 加 `repetition_penalty=1.1`，另一个请求要求输出必须是合法的 JSON，服务本身又开着投机解码想把 300 步 decode 压成 120 步。

这三件事看起来风马牛不相及——一个是采样参数，一个是输出格式，一个是加速手段——但它们落到系统里的位置是同一个：**都在改变"从 logits 到 token"这最后一步的做法**。而且它们都不满足于只改 Sampler：投机解码要调度器多给 token 预算、要 KV Cache 预留槽位、要 model runner 一次算 K+1 个位置；结构化输出要引擎在请求进来时异步编译 grammar、每一步在 GPU 上打一张掩码、还要和投机解码的 draft token 逐个对账；连最"无害"的 penalties，也要求 model runner 每一步把每个请求的全部历史输出搬到 GPU 上数一遍。

所以本篇的核心问题是：

> **同样是"下一个 token"，为什么加上 top-p、加上 draft 模型、加上 JSON schema 之后，调度器、KV 管理和 model runner 都得改？每一种扩展花掉什么、换回什么？**

## 一、总览：三种扩展，同一个落点

### 1. 采样这一步在哪里

先把"最后一步"画清楚。第六篇的 forward 结束在 `hidden_states`；接下来是：

```text
hidden_states [num_tokens, hidden]
      │  只取每个请求最后一个位置（logits_indices）
      ▼
lm_head → logits [num_reqs, vocab]           ← TP 下由 LogitsProcessor 层 gather 到一张卡
      │
      ▼
Sampler(logits, SamplingMetadata) → sampled_token_ids [num_reqs, 1]
      │
      ▼
ModelRunnerOutput → Scheduler.update_from_output()
```

三种扩展分别改的是这条链上的三个不同的东西：

| 扩展 | 改的是什么 | 一句话 |
|---|---|---|
| 采样与 logits processors | **分布本身**：每个请求按自己的参数改写这一行 logits | 同一个 batch 里每一行有不同的温度、不同的禁用词、不同的惩罚 |
| 投机解码 | **每步决定的位置数**：一次 forward 不再只出 1 个 token，而是验证 K 个候选再加 1 个 bonus | logits 从 `[num_reqs, V]` 变成 `[num_reqs + num_drafts, V]` |
| 结构化输出 | **分布的支撑集**：把 grammar 当前状态下不合法的 token 全部置 `-inf` | 在 Sampler 之前，先用一张 bitmask 把 logits 挖空 |

### 2. 三种扩展分别惊动了谁

如果它们只改 Sampler，本篇一章就够了。问题在于它们各自向上游捅了多深：

| | 调度器（第四篇） | KV Cache（第五篇） | Model Runner / 执行（第六篇） | Sampler |
|---|---|---|---|---|
| logits processors | —— | —— | 维护每个请求的 `output_token_ids` 列表并每步上传；batch 增删移动时同步处理器状态 | 主战场 |
| 投机解码 | `num_tokens_with_spec` 进预算；被拒绝的 token 回滚 `num_computed_tokens`；每步收集接受率 | `allocate_slots(num_lookahead_tokens)` 预留槽位；EAGLE 命中 prefix cache 时**少算一块** | 一次 forward 算 `1+K` 个位置；draft 模型有自己的 KV group 和 CUDA graph；"纯 decode batch"的定义从 1 token/req 变成 `1+K` token/req | 换成 `RejectionSampler` |
| 结构化输出 | 请求进入 `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR` 等编译；每步生成 bitmask；每步用 `accept_tokens` 推进 FSM；过滤 draft | —— | 在采样前对 logits 就地打掩码 | 掩码在 Sampler 之前，Sampler 本身不变 |

这张表就是本篇的路线图：每一章讲一种扩展，先说它给三个核心战场加了什么约束，再看 vLLM 的实现，最后算账。

### 3. 回到我们的例子：这一步有多贵？

**回到我们的例子**（Llama-3-70B、8×H100、TP=8、2050 token prompt、生成 300 token）。Llama-3 的词表是 128256，一行 fp32 logits 是 `128256 × 4 B ≈ 513 KB`；batch=64 时整个 logits 张量约 33 MB，按 3.35 TB/s 算读一遍只要 0.01 ms。和第六篇里"一步 decode 约 10 ms"相比，**Sampler 本身的成本是可以忽略的**——这正是我们能在这一步塞进那么多花样的原因。

但有三个东西不便宜，本篇会逐个点名：

- **排序**：top-p 的朴素实现要对每行 128256 个数排序；
- **计数**：penalties 要为每个请求在 GPU 上建一张 `[vocab+1]` 的 int64 直方图，batch=64 就是 66 MB，而且每步重建；
- **多算的位置**：投机解码让 forward 从 `num_reqs` 个位置变成 `num_reqs × (1+K)` 个，在 memory-bound 区间"几乎免费"，一过临界点就不再免费——第三章会算出这个临界点在哪里。

### 4. 本文的章节安排

```text
第二章  采样与 logits processors      Sampler 的流水线、每请求参数的向量化、LogitsProcessor 接口与持久 batch、penalties 的真实代价
第三章  投机解码的工程实现            Proposer 家族、调度器的预算与 KV 预留、model runner 的 K+1 位置、RejectionSampler、收益的量化与何时不该开
第四章  结构化输出                    从 grammar 到 bitmask 的四个位置、后端抽象、异步编译与调度器的等待、与投机解码叠加、reasoning 跳过、代价
第五章  三者叠加                      一步之内的执行顺序；留给多卡（08）与 PD 分离（12）的问题
第六章  本文小结
```

## 二、采样与 logits processors：让同一个 batch 里的每一行按自己的规矩走

### 1. Sampler 的流水线

`Sampler`（`vllm/v1/sample/sampler.py`）的 docstring 把顺序写得很清楚，这里按它的编号照抄骨架：

```python
# vllm/v1/sample/sampler.py（docstring 顺序，代码略去分支）
class Sampler(nn.Module):
    """
    1. 若要 logprobs：按 logprobs_mode 先保存一份"原始" logprobs / logits
    2. logits 转 float32
    3. allowed_token_ids 白名单（不在名单里的置 -inf）
    4. bad_words 排除
    5. 非 argmax-invariant 的 logits processors（会影响 greedy 结果的）
         a) min_tokens   b) logit_bias
    6. penalties：repetition / frequency / presence
    7. sample()：
         a) 若不是 all_random，先算 greedy（argmax）；若 all_greedy 直接返回
         b) 温度
         c) argmax-invariant 的 logits processors（默认只有 min_p）
         d) top_k / top_p
         e) 按分布采样
         f) 逐行 torch.where(temperature < eps, greedy, random)
    8. 取 top-k logprobs 与被采样 token 的 logprob
    """
```

这个顺序不是随意的。**第 5 步与第 7c 步的区分——"会不会改变 argmax"——是整个流水线的组织原则**：`min_tokens` 把 EOS 置 `-inf`、`logit_bias` 直接加偏置，这两个会改变最大值在哪里，所以必须在 greedy 分支之前做；而 `min_p` 只是把低于阈值的 token 砍掉，最大值永远留着，greedy 请求做不做都一样，所以放到温度之后、只对随机采样的行做。这样一个 batch 里 greedy 与 random 的请求可以走同一条流水线，最后一行 `torch.where` 分流，不需要拆 batch。

还有一个容易漏看的点：**logprobs 默认取的是"原始" logits 的 log-softmax**（`logprobs_mode="raw_logprobs"`），也就是第 1 步那份、还没经过 penalties 和温度的。源码注释明说这与 V0 不同——V0 返回的是采样用的 processed 分布。如果你的评测依赖 logprobs，这个差别要知道。

### 2. 每请求不同参数：向量化，而不是循环

`SamplingMetadata`（`vllm/v1/sample/metadata.py`）里，`temperature`、`top_p`、`top_k`、三种 penalty 都是形状 `[num_reqs]` 的张量，`all_greedy` / `all_random` / `no_penalties` 是三个用来短路的标量。整条流水线没有"for req in batch"：

- **温度**：`apply_temperature()` 先把 greedy 请求的温度（`< 1e-5`）用 `torch.where` 换成 1，再对整张 logits 做一次 `div_`；
- **随机采样**：`random_sample()`（`vllm/v1/sample/ops/topk_topp_sampler.py`）不用 `torch.multinomial`——它会触发 CPU-GPU 同步——而是用指数噪声技巧：`q ~ Exp(1)`，`argmax(probs / q)`，等价于按 `probs` 采样。带 `seed` 的请求单独用自己的 `torch.Generator` 覆盖它那一行的噪声（源码注释承认这个逐请求循环"can be slow"）；
- **top-k / top-p**：`apply_top_k_top_p()` 有三条路：CPU 走 Triton 或纯 PyTorch；GPU 上 batch ≥ 8 走 Triton kernel（`topk_topp_triton.py`），小 batch 走 `apply_top_k_top_p_pytorch()`——后者对每行做一次全词表 `sort`，这就是第一章点名的"排序"成本。CUDA 上如果 FlashInfer 可用（`flashinfer_sampler_supported()`，由 `VLLM_USE_FLASHINFER_SAMPLER` 控制），`TopKTopPSampler.forward_cuda()` 会把 top-k/top-p/采样交给 FlashInfer 的融合 kernel；但只要 batch 里有带 seed 的请求，或 `logprobs_mode` 要求 processed 分布，就退回 native 路径。

"每请求不同参数"这件事在 GPU 上是免费的——一个 `[num_reqs]` 的向量广播到 `[num_reqs, V]` 上而已。**真正的成本在于每一步都要把这些参数张量维护好**，这是下一节 `LogitsProcessor` 接口存在的原因。

### 3. LogitsProcessor 接口：为什么它长成"跟着 batch 变化"的样子

`LogitsProcessor`（`vllm/v1/sample/logits_processor/interface.py`）只有四个方法：

```python
# vllm/v1/sample/logits_processor/interface.py
class LogitsProcessor(ABC):
    def __init__(self, vllm_config, device, is_pin_memory): ...
    def apply(self, logits: torch.Tensor) -> torch.Tensor: ...      # 对整张 [num_reqs, V] 就地改
    def is_argmax_invariant(self) -> bool: ...                        # 决定进第 5 步还是第 7c 步
    def update_state(self, batch_update: BatchUpdate | None): ...    # 每次 forward 前调用
```

关键在 `update_state()` 的参数 `BatchUpdate`：

```python
@dataclass(frozen=True)
class BatchUpdate:
    batch_size: int
    removed: Sequence[int]                                  # 被移出 batch 的下标
    added: Sequence[tuple[int, SamplingParams, list[int] | None, list[int]]]
                                                            # (下标, 参数, prompt_ids, output_ids 的引用)
    moved: Sequence[tuple[int, int, MoveDirectionality]]    # 请求在 batch 内的位置变动（单向 / 交换）
```

为什么接口是这个形状？因为 vLLM V1 的 `InputBatch`（`vllm/v1/worker/gpu_input_batch.py`）是一个**持久 batch**：请求不是每步重新排列，而是占着一个下标，完成的请求腾出的坑由后来者填上、或者把队尾的请求挪过来补洞。一个 logits processor 如果维护了任何"按下标索引的状态"（比如 `min_p` 的 `[max_num_reqs]` 张量），就必须跟着这些增删移动同步。`InputBatch` 里的 `BatchUpdateBuilder` 在一步之内收集所有 `removed / added / moved`，`refresh_metadata()` 时打包成一个 `BatchUpdate` 喂给每个处理器——`removed` 按降序排好，处理顺序固定为 removed → added → moved。

`added` 元组里的 `output_tok_ids` 是**对请求输出列表的引用**，不是拷贝（接口注释里特意写了 "Key assumption"）。这样 `MinTokensLogitsProcessor` 在 `update_state()` 时只要看 `len(out_tok_ids) >= min_toks` 就知道哪些请求已经过了最小长度，可以从自己的稀疏字典里删掉——不需要任何人通知它"又生成了一个 token"。

内置的三个处理器（`vllm/v1/sample/logits_processor/builtin.py`）刻意示范了两种状态组织方式：

| 处理器 | 状态形态 | `apply()` 做什么 | argmax-invariant |
|---|---|---|---|
| `MinPLogitsProcessor` | 稠密：`[max_num_reqs]` 的 CPU pinned 张量 + GPU 镜像，`min_p_count` 计数为 0 时整个跳过 | `softmax` → 每行最大概率 × min_p → 低于阈值置 `-inf` | 是 |
| `LogitBiasLogitsProcessor` | 稀疏：`dict[req_idx → dict[token → bias]]`，变化时重建一组 `(req_idx, tok_id, bias)` 的索引张量 | `logits[(reqs, toks)] += biases` 一次 index_put | 否 |
| `MinTokensLogitsProcessor` | 稀疏：`dict[req_idx → (min_toks, output_ids 引用, stop_token_ids)]` | `index_put_(-inf)` 到所有未达最小长度请求的 stop token 上 | 否 |

稀疏路线的通用逻辑抽成了 `process_dict_updates()`，`AdapterLogitsProcessor` 也复用它。

**thinking budget** 走的是同一套协议，但不是 `LogitsProcessor` 子类：`ThinkingBudgetStateHolder`（`vllm/v1/sample/thinking_budget_state.py`）用同样的 `BatchUpdate` 做 `sync_batch()`，但它需要看到本步"已经提交的输出 + 投机 draft"来判断思考段是否超预算，所以 `Sampler.apply_logits_processors()` 把它放在 penalties 之后单独调用（`holder.update_state()` → `holder.apply_to_logits()`），并明确区分 `output_token_ids`（已提交）与 `spec_token_ids`（草稿）。

**自定义处理器**有两条注册路径（`vllm/v1/sample/logits_processor/__init__.py`）：

- 引擎参数 `--logits-processors x.y.z:MyProc`（`ModelConfig.logits_processors`），`_load_logitsprocs_by_fqcns()` 按 `module:Class` 导入；
- Python entry point 组 `vllm.logits_processors`（`LOGITSPROCS_GROUP`），`_load_logitsprocs_plugins()` 自动发现。

两者都要求是 `LogitsProcessor` 子类，并在引擎启动时实例化一次——**V1 不再支持请求级传入 callable**。如果你有一个老式的 `def f(output_ids, logits)`，用 `AdapterLogitsProcessor` 包一层：实现 `new_req_logits_processor(params)` 按请求参数决定要不要给这个请求装一个，`apply()` 会逐行调用。逐行调用意味着一个 Python 循环，这是兼容的代价。

两条硬限制值得记住：pooling 模型拒绝任何自定义处理器；**开启投机解码时拒绝自定义处理器，并且 `min_p` 和 `logit_bias` 也不生效**（`build_logitsprocs()` 只保留 `MinTokensLogitsProcessor`，并打一条 warning）——原因在第三章第 5 节会看到：投机解码下 logits 的行数不再等于请求数，每一行属于哪个请求需要 `SpecDecodeMetadata` 才知道，而现有处理器接口拿不到这个信息。

### 4. penalties 的真实代价

penalties 不是 `LogitsProcessor`，而是流水线里一段硬编码的步骤（`Sampler.apply_penalties()` → `apply_all_penalties()`，`vllm/v1/sample/ops/penalties.py` → `apply_penalties()`，`vllm/model_executor/layers/utils.py`）。它做的事很直白：

```python
# vllm/model_executor/layers/utils.py（简化）
def apply_penalties(logits, prompt_tokens_tensor, output_tokens_tensor,
                    presence_penalties, frequency_penalties, repetition_penalties):
    num_seqs, vocab_size = logits.shape
    _, prompt_mask = get_token_bin_counts_and_mask(prompt_tokens_tensor, vocab_size, num_seqs)
    output_bin_counts, output_mask = get_token_bin_counts_and_mask(output_tokens_tensor, vocab_size, num_seqs)
    apply_repetition_penalties(logits, prompt_mask, output_mask, repetition_penalties)   # 自定义 CUDA op
    logits -= frequency_penalties.unsqueeze(1) * output_bin_counts
    logits -= presence_penalties.unsqueeze(1) * output_mask
```

`get_token_bin_counts_and_mask()` 用 `scatter_add_` 建一张 `[num_seqs, vocab_size + 1]` 的 **int64** 直方图（多出的一列给 padding 用）。代价算一下：

| 项 | 量 | 说明 |
|---|---|---|
| 直方图 | `64 × 128257 × 8 B ≈ 66 MB`，prompt 和 output 各一张 | 每步重新 `zeros` + `scatter_add_` |
| `output_token_ids` 上传 | `_convert_to_tensors()` 把 Python 的 `list[list[int]]` pad 成张量再 H2D | 长度随已生成 token 数增长；300 步时是 `64 × 300` 个 int64 |
| repetition penalty | `apply_repetition_penalties` 自定义 op（`csrc/libtorch_stable/sampler.cu`） | 读两张 mask + 整张 logits |

几十 MB、几百微秒的量级，和 10 ms 的 decode 比仍然是小数——但它是**每一步、只要 batch 里有任何一个请求开了 penalty 就要全 batch 付**的固定成本，而且随 batch 和词表线性增长。源码里 `NOTE(nick)` 直言 "The penalties implementation is currently quite inefficient and will be reworked anyhow"。更隐蔽的成本在 CPU 侧：`no_penalties` 为 False 时 model runner 必须每步维护并上传 `output_token_ids`，异步调度下这个列表甚至可能含有 `-1` 占位（`apply_all_penalties()` 里专门有一行把 `-1` 换成 `vocab_size`）。

**对比一下不同采样参数的成本形态**：

| 参数 | 每步额外工作 | 成本形态 |
|---|---|---|
| `temperature` / `top_k`（Triton 或 FlashInfer） | 一次逐元素 / 一个 kernel | 忽略 |
| `top_p`（小 batch 纯 PyTorch 路径） | 每行全词表 sort | 与 `V log V` 成正比，batch < 8 才会走到 |
| `min_p` | 一次 softmax + amax + mask | 忽略 |
| `logit_bias` / `min_tokens` / `allowed_token_ids` / `bad_words` | 稀疏 index_put | 忽略 |
| 任一 penalty | 两张 `[B, V+1]` int64 直方图 + 历史输出上传 | **O(B·V) 显存流量 + CPU 侧列表维护** |
| 带 `seed` | 逐请求 Python 循环覆盖噪声；FlashInfer 路径失效 | 与带 seed 的请求数线性 |

结论：采样参数里真正值得在容量规划时考虑的只有 penalties 和 seed，其余都淹没在 forward 里。

## 三、投机解码：把"一步一个 token"改成"一步一串候选"

### 1. 第六篇留下的问题

第六篇第五章已经讲完了投机解码的原理——draft 猜 K 个、target 一次验证、拒绝采样保证分布无损——以及 Draft Model / n-gram / EAGLE / Medusa / MTP 在"候选从哪来"上的区别。本章不再重复这些，只回答一个问题：**这套东西在 vLLM 里到底改了哪几个对象**。

答案是三个角色加两处约束：

```text
Proposer（vllm/v1/spec_decode/）        ── 产出 draft_token_ids（可选 draft_probs）
        │
        ▼
Scheduler（vllm/v1/core/sched/scheduler.py） ── 把 draft 算进 token 预算；给 KV 预留 lookahead 槽位；
        │                                        step 后按接受数回滚 num_computed_tokens
        ▼
GPUModelRunner（vllm/v1/worker/gpu_model_runner.py）
        │   _calc_spec_decode_metadata()：一次 forward 取 1+K 个位置的 logits
        ▼
RejectionSampler（vllm/v1/sample/rejection_sampler.py）
            ── 验证 K 个候选 + 采 1 个 bonus，输出 [batch, K+1]，被拒位置填 -1
```

两处约束：**KV**（draft 模型如果有自己的 KV，要独立的 cache group；target 侧要为尚未验证的 token 预留位置）和 **CUDA graph**（"纯 decode batch"的形状从每请求 1 个 token 变成 `1+K` 个）。

### 2. Proposer 接口与家族

`vllm/v1/spec_decode/` 下没有一个统一的抽象基类——不同 proposer 的输入根本不同（n-gram 只要 token 历史，EAGLE 要 target 的 hidden state，Medusa 要最后一层 hidden state），所以 `GPUModelRunner.propose_draft_token_ids()` 里是一段按 `speculative_config.method` 分发的代码，每种 proposer 有自己的 `propose()` 签名。可以按"要不要跑模型、要不要自己的 KV"分成三档：

| 档 | 方法（`SpeculativeConfig.method`） | 类 / 文件 | 跑模型？ | 自己的 KV？ | `draft_probs` |
|---|---|---|---|---|---|
| 抄历史 | `ngram` | `NgramProposer`（`ngram_proposer.py`，numba 实现 `batch_propose_numba()`，`prompt_lookup_min/max` 控制 n-gram 长度） | 否，CPU | 否 | `None` |
| | `ngram_gpu` | `NgramProposerGPU`（`ngram_proposer_gpu.py`） | 否，GPU kernel | 否 | `None` |
| | `suffix` | `SuffixDecodingProposer`（`suffix_decoding.py`，调用 Arctic Inference 的 `SuffixDecodingCache`） | 否 | 否 | `None` |
| 外挂头 | `medusa` | `MedusaProposer`（`medusa.py`）：对 target 的 hidden state 过若干 head，每个 head 取 argmax | 是，但一次 forward | 否 | 无（greedy） |
| | `eagle` / `eagle3` | `EagleProposer`（`eagle.py`，只有二十来行：`SpecDecodeBaseProposer` + `pass_hidden_states_to_model=True`） | 是，串行 K 步 | **是**（一个独立的 KV cache group） | 可选 |
| | `mtp`（含 `deepseek_mtp`、`qwen3_next_mtp` 等十几种 `*_mtp` 类型） | 同 `EagleProposer`，draft 结构是模型的一部分（`vllm/model_executor/models/*_mtp.py`） | 是 | 是 | 可选 |
| | `dflash` / `dspark` / `gemma4_mtp` / `step3p5_mtp` | `DFlashProposer`、`Gemma4Proposer`、`Step3p5MTPProposer` 等变体 | 是 | 是 | —— |
| 独立模型 | `draft_model` | `DraftModelProposer`（`draft_model.py`）：`SpecDecodeBaseProposer` + `pass_hidden_states_to_model=False`；支持异构词表（`VocabMapping`） | 是，一个完整小模型 | 是 | 可选 |
| 自定义 | `custom_class` | `create_custom_proposer()`（`custom_class_proposer.py`） | —— | —— | —— |

几个实现细节决定了后面几节的形状：

- **`SpecDecodeBaseProposer`**（`llm_base_proposer.py`，近 1900 行）是 EAGLE / MTP / draft model 的公共实现：`propose()` 里 `for token_index in range(self.num_speculative_tokens - 1)` 串行跑 draft 模型；`load_model()` 处理与 target 共享 embedding / lm_head（`_maybe_share_embeddings()`、`_maybe_share_lm_head()`，draft model 子类覆写为不共享）；它持有**自己的 `CudagraphDispatcher`**，`initialize_cudagraph_keys()` 只允许 PIECEWISE（draft 侧不做 FULL 图）。
- **draft 有自己的 KV cache group**：`SpecDecodeBaseProposer.kv_cache_gid` 在 `initialize_attn_backend()` 时找到 `is_eagle_group` 为真的那个 group；调度器在 `KVCacheManager` 构造时传入 `use_eagle`，`KVCacheCoordinator` 用它标记 `eagle_group_ids`。对 70B 的 target 来说，EAGLE 头只有一层，每 token 的 draft KV 是 `4 KB`（与 target 的一层相同），2350 token 的请求约 9 MB——相对 734 MB 的 target KV 可以忽略；但一个 16 层、8 KV head、`head_dim=64` 的独立 1B draft 模型，每 token 是 `2 × 8 × 64 × 2 B × 16 = 32 KB`，即 target 的 10%，这就得算进显存预算了。
- **`draft_probs` 有没有**决定了拒绝采样用哪条规则（第 5 节）：`draft_sample_method="probabilistic"` 时 `take_last_draft_probs()` 会把 draft 的分布留给 `RejectionSampler`；n-gram 一族天然没有分布。
- **draft model 的 TP 必须等于 target 的 TP**（`DraftModelProposer._raise_if_draft_tp_mismatch()`），原因是 torch.compile 缓存在不同 TP rank 上会互相覆盖——这条限制留给第八篇。

### 3. 调度器：spec token 进预算，KV 预留槽位

第四篇讲 `schedule()` 时反复出现的 `num_tokens_with_spec`，在这里终于落地：

```text
num_tokens_with_spec = len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids)
```

调度器对一个 running 请求做三件和普通 decode 不同的事（`Scheduler.schedule()`，`vllm/v1/core/sched/scheduler.py`）：

**① 把 draft 算进 token 预算。** `num_new_tokens = num_tokens_with_spec + num_output_placeholders - num_computed_tokens`，普通 decode 是 1，带 K 个 draft 就是 `1+K`。它和 prefill chunk 一样受 `token_budget` 约束，所以当预算紧张时 draft 会被**截断**：

```python
# Scheduler.schedule()（简化）
if request.spec_token_ids:
    num_scheduled_spec_tokens = (num_new_tokens + request.num_computed_tokens
                                 - request.num_tokens - request.num_output_placeholders)
    if num_scheduled_spec_tokens > 0:
        spec_token_ids = request.spec_token_ids[:num_scheduled_spec_tokens]   # 预算不够就只验证前几个
        scheduled_spec_decode_tokens[request.request_id] = spec_token_ids
    request.spec_token_ids = []     # 下一步的 draft 由 update_draft_token_ids() 重新填
```

注意 `num_scheduled_spec_tokens > 0` 这个条件：一个还在 chunked prefill 中的请求，`num_computed_tokens + num_new_tokens` 追不到 `num_tokens`，这个量 ≤ 0，**draft 在 prefill 阶段根本不会被调度**——`update_draft_token_ids()` 里对 `is_prefill_chunk` 的请求也直接丢弃 draft。投机解码只在请求进入纯 decode 后才生效。

另一条上界：`num_new_tokens ≤ max_model_len - num_computed_tokens - num_sampled_tokens_per_step`，源码注释写明 "This is necessary when using spec decoding"——验证 K 个 draft 后还要采一个 bonus，位置不能越界。

**② 给 KV 预留 lookahead 槽位。** `allocate_slots(request, num_new_tokens, num_lookahead_tokens=self.num_lookahead_tokens)`。第五篇 `allocate_slots()` 注释里那个布局图的最右边一段 `lookahead` 就是它。`num_lookahead_tokens` 在 `Scheduler.__init__` 里按方法设定：EAGLE / MTP / draft model 为 `num_spec_tokens`；DFlash 因为要多一个 query 位置是 `num_spec_tokens + 1`；n-gram 一族为 0——它们不跑模型，不需要为 draft 写 KV。为什么 draft 模型需要 lookahead？因为 draft 模型在**这一步的 forward 之后**才跑，它要为下一步的 K 个候选写自己那份 KV，位置必须现在就分好。

**回到我们的例子**：K=3，`num_lookahead_tokens=3`，每张卡每 token 40 KB，预留 120 KB；`block_size=16` 时绝大多数步骤这 3 个槽位落在当前块里，偶尔多占一块（每卡 640 KB）。对一个 734 MB 的请求来说是零头——**lookahead 的代价不在显存，而在它让 `allocate_slots()` 更早触碰 watermark**。

**③ step 之后按接受数回滚。** `update_from_output()` 里：

```python
num_draft_tokens = len(scheduled_spec_token_ids)
num_accepted = max(len(generated_token_ids) - num_sampled, 0)   # 减掉 bonus
num_rejected = num_draft_tokens - num_accepted
request.num_computed_tokens -= num_rejected                    # 被拒的位置"当作没算过"
```

第六篇第五章说 KV 要"回滚"——落到源码上，**回滚就是这一行减法**。被拒 token 占的槽位不需要真的释放，下一步的 token 会直接覆盖它们（`slot_mapping` 按 `num_computed_tokens` 算）。第九篇 MTP 一节说的"暂存状态 vs 提交状态"，在 vLLM 里的实现就是 `num_computed_tokens` 与实际写入 KV 位置之间的这个差。

同一处还调用 `make_spec_decoding_stats()` 累积 `SpecDecodingStats`（`vllm/v1/spec_decode/metrics.py`）：`num_drafts`、`num_draft_tokens`、`num_accepted_tokens`、以及**按位置**的 `num_accepted_tokens_per_pos` / `num_draft_tokens_per_pos`。它随 `SchedulerStats` 回到前端，`SpecDecodingLogging.log()` 定期打印一行：

```text
SpecDecoding metrics: Mean acceptance length: 2.71, Accepted throughput: ... tokens/s,
Drafted throughput: ... tokens/s, Accepted: N tokens, Drafted: M tokens,
Per-position acceptance rate: 0.812, 0.605, 0.412, Avg Draft acceptance rate: 61.0%
```

"Mean acceptance length" 按惯例含 bonus token（`1 + accepted / drafts`）；Prometheus 侧是 `SpecDecodingProm`。**按位置的接受率是调 K 的依据**：第 3 个位置只有 41% 时，第 4 个位置大概率不值得。

**④ draft 从哪里来、何时回到调度器。** 同步调度下，`EngineCore.post_step()` 在一步结束后 `take_draft_token_ids()` → `Scheduler.update_draft_token_ids()` 写回 `request.spec_token_ids`；异步调度下调度器提前一步跑，拿不到 draft，于是 worker 侧在 `update_draft_token_ids_in_output()` 里把 `scheduled_spec_decode_tokens` 的占位换成真实 draft，**长度不够的位置补 `-1`**，并把每个请求补了几个记进 `num_invalid_spec_tokens`——`-1` 在 `RejectionSampler` 的 kernel 里被无条件拒绝。

**⑤ 一个 prefix cache 的特例。** `HybridKVCacheCoordinator.find_longest_cache_hit()` 与 `SingleTypeKVCacheManager.find_longest_cache_hit()` 都有一个 `drop_eagle_block` 参数：EAGLE / MTP 开启时，**prefix cache 命中的最后一个块会被主动丢掉**（`hit_length -= min(alignment_tokens, block_size)`）。原因是 EAGLE 的 draft 需要 target 对"最后一个已计算 token"的 hidden state，而 prefix cache 只保留 KV、不保留 hidden state——全部命中的话就没有任何位置会真正跑 forward，draft 无从下手。所以第五篇算的"125 块全部命中"在开 EAGLE 时会变成 124 块命中、重算 16 个 token。相应地 `cache_blocks()` 会多缓存一块（`num_tokens_to_cache = aligned + block_size`），让下一个请求仍能用到。

### 4. Model runner：一次 forward 算 K+1 个位置

调度器交出的 `SchedulerOutput.scheduled_spec_decode_tokens` 在 model runner 里变成 `SpecDecodeMetadata`（`vllm/v1/spec_decode/metadata.py`）。`GPUModelRunner._calc_spec_decode_metadata()` 的注释自带一个例子，照抄：

```python
# Inputs:
# cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]   ← 5 个请求，本步各调度 4/100/3/100/2 个 token
# num_draft_tokens:         [  3,   0,   2,   0,   1]   ← 请求 0 带 3 个 draft，请求 1 和 3 在 prefill
# Outputs:
# cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
# logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106, 206, 207, 208]
# target_logits_indices:    [  0,   1,   2,   5,   6,   9]
# bonus_logits_indices:     [  3,   4,   7,   8,  10]
```

看请求 0：它本步的 4 个 token 是 `[上一步采样的 token, d₁, d₂, d₃]`，forward 后需要 4 个位置的 logits——前 3 个（`target_logits_indices`）用来验证 `d₁ d₂ d₃`，第 4 个（`bonus_logits_indices`）是全部接受时的 bonus。请求 1 在 prefill，只要最后一个位置（104 → 全是 bonus）。于是 `logits_indices` 总长 `num_tokens + batch_size`，**logits 张量不再是 `[num_reqs, V]` 而是 `[num_reqs + Σ drafts, V]`**——这就是第二章末尾说的"现有 logits processor 接口不知道每一行属于谁"的根源。`draft_token_ids` 直接从 `input_ids[logits_indices][target_logits_indices + 1]` 取出来：draft 已经作为输入 token 喂进了 forward。

**这一步和 CUDA graph 的关系**是本章最容易被忽略的约束。第六篇讲 `FULL_AND_PIECEWISE` 时说"纯 decode batch 形状规则、可以按 batch size 分桶录图"——投机解码把"规则"的定义改了：

```python
# vllm/v1/worker/gpu_model_runner.py
self.uniform_decode_query_len = 1 + self.num_spec_tokens

@staticmethod
def _is_uniform_decode(max_num_scheduled_tokens, uniform_decode_query_len, num_tokens, num_reqs, ...):
    return (max_num_scheduled_tokens == uniform_decode_query_len
            and num_tokens == max_num_scheduled_tokens * num_reqs)
```

只有**每个请求恰好 `1+K` 个 token** 的 batch 才算 uniform decode、才走 FULL 图。`CudagraphDispatcher.initialize_cudagraph_keys()` 也据此只为 `≥ uniform_decode_query_len` 且能被它整除的 capture size 建 FULL 图的 key。这带来两个后果：

- 上一节说的"预算不够就截断 draft"一旦发生，那个请求就只有 `1 + K'` 个 token，整个 batch 不再 uniform，退回 PIECEWISE——**投机解码对 token 预算的敏感度比普通 decode 高**；
- draft 模型自己那 K 步 forward 走 `SpecDecodeBaseProposer` 的独立 dispatcher，只有 PIECEWISE。

采样之后紧接着就是下一轮的 draft（`GPUModelRunner.sample_tokens()` 内部的 `propose_draft_token_ids()`）。默认的 "padded drafter batch" 模式下，EAGLE / draft model 直接消费 GPU 上的 `sampled_token_ids`，不等 D2H 拷贝和 CPU 侧 bookkeeping；`disable_padded_drafter_batch=True` 则退回先同步再 draft。这是投机解码在 V1 里能与异步调度共存的前提——`VllmConfig` 里限制了异步调度只支持 EAGLE 系、`ngram_gpu`、`draft_model` 和 `dspark`。

### 5. RejectionSampler：验证的实现

`RejectionSampler`（`vllm/v1/sample/rejection_sampler.py`）的 docstring 先给了术语表：**accepted tokens**（按 draft/target 概率比接受的）、**recovered tokens**（拒绝后从修正分布采出的）、**bonus tokens**（全部接受后额外的一个，用普通 `Sampler` 采）、**output tokens = accepted + recovered + bonus**。`forward()` 分四步：

```python
# RejectionSampler.forward()（简化）
bonus_logits  = logits[metadata.bonus_logits_indices]            # [batch, V]
bonus_token_ids = self.sampler(bonus_logits, sampling_metadata, predict_bonus_token=True)
                                                                   # ① bonus 走完整的 Sampler：top-p、penalties 全都生效

target_logits = logits[metadata.target_logits_indices].float()   # [num_drafts, V]
target_logits = self.apply_logits_processors(target_logits, sampling_metadata, metadata)
                                                                   # ② 投机版的处理器：penalties / bad_words / min_tokens / thinking budget
target_logits = apply_sampling_constraints(target_logits, metadata.cu_num_draft_tokens, sampling_metadata)
                                                                   # ③ 温度 + top_k/top_p，参数按 draft 数 expand 到每一行

output_token_ids = rejection_sample(metadata.draft_token_ids, metadata.num_draft_tokens,
                                    metadata.max_spec_len, metadata.cu_num_draft_tokens,
                                    draft_probs, target_logits, bonus_token_ids, sampling_metadata, ...)
                                                                   # ④ 两个 Triton kernel
```

第 ② 步解释了为什么投机解码下 logits processor 受限：`apply_logits_processors()` 要用 `repeat_indices`（`arange(num_reqs).repeat_interleave(num_draft_tokens)`）把 `[num_reqs]` 的 penalty 参数展开到 `[num_drafts]` 行；`output_token_ids` 也要用 `_combine_outputs_with_spec_tokens()` 为第 i 个 draft 位置拼出"已输出 + 前 i-1 个 draft"的历史；`MinTokensLogitsProcessor` 有专门的 `apply_with_spec_decode()`。只有这几个内置步骤做了这层适配，通用的 `LogitsProcessor.apply(logits)` 接口没有——所以 `min_p`、`logit_bias` 和自定义处理器只能对 bonus 行生效、对 target 行不生效，vLLM 干脆选择在开投机时禁用它们。

第 ③ 步的 `apply_sampling_constraints()` 用 `expand_batch_to_tokens()`（一个 Triton `expand_kernel`）把温度、top-k、top-p 从每请求展开到每 draft 行，greedy 请求的温度 0 换成 1；然后调用第二章那个 `apply_top_k_top_p()`——源码注释 `NOTE(woosuk)` 提醒它用排序、大词表下慢。

第 ④ 步 `rejection_sample()`：

- 输出缓冲 `output_token_ids = full((batch, max_spec_len + 1), PLACEHOLDER_TOKEN_ID=-1, int32)`，`MAX_SPEC_LEN = 128` 是编译期上限；
- 若 batch 里有 greedy 请求，先跑 `rejection_greedy_sample_kernel`：逐位置比较 `draft_token_id == target_argmax`，第一个不等就写 target 的 argmax 并停止，全等则补 bonus——**greedy 下投机解码严格等价于逐 token greedy**；
- 若有随机请求，`target_probs = softmax(target_logits)`，先用 `sample_recovered_tokens()` 为**每个位置**预采一个 recovered token（用第六篇那个 `max(0, q - p)` 归一化的残差分布；同样用指数噪声技巧），再跑 `rejection_random_sample_kernel`：位置 i 接受当且仅当 `draft_prob > 0 且 target_prob / draft_prob ≥ u_i`（`u_i` 由 `generate_uniform_probs()` 生成，尊重每请求的 seed）；第一个被拒的位置写入预采好的 recovered token 并停止；
- `NO_DRAFT_PROBS` 为真（n-gram 一族）时 `draft_prob = 1`，接受条件退化为 `target_prob ≥ u_i`——即按 target 自己的概率接受这个具体 token。这仍然是一个合法的采样过程（等价于把 draft 视为确定性提议），但**接受率会比有 draft 分布时低**：一个 target 概率 0.6 的 token 只有 60% 机会被接受，即使它就是 target 最想要的那个。
- 两个 kernel 都以 `batch_size` 为 grid，每个 program 顺序扫自己请求的 K 个位置——验证的并行度在请求间，不在位置间，因为位置间有"前一个被拒后面全作废"的依赖。

`SpeculativeConfig.rejection_sample_method` 还有两个非默认值：`"synthetic"` 按给定的每位置接受率随机接受（用于 benchmark 时模拟固定接受率），`"block"` 是块验证。

`parse_output()` 把 `[batch, K+1]` 里 `-1` 之前的部分切成每请求的 token 列表，回到 `ModelRunnerOutput.sampled_token_ids`——这就是调度器 `update_from_output()` 里 `len(generated_token_ids)` 的来源。

### 6. 收益与代价的量化

第六篇给的例子是 batch=1、"验证 6 个位置比 1 个位置只多 3 ms"。这里把它推广到不同 batch，并给出临界点。

**"验证几乎免费"的边界在哪里？** 一步 decode 的时间约等于 `max(权重读取时间, 计算时间)`。每张卡 17.6 GB 权重、3.35 TB/s → 读取 5.3 ms；每 token 每卡的计算量约 `2 × 8.75 GFLOP`，H100 BF16 稠密 989 TFLOPS。两者相等时的 token 数：

```text
T_ridge ≈ (989e12 FLOP/s × 2 B/param) / (2 × 3.35e12 B/s) ≈ 295 token / 步 / 卡
```

也就是说，**一步里 `batch × (1+K)` 超过大约 300 个 token，这一步就从 memory-bound 转入 compute-bound，多算的位置开始真金白银地花时间**（40% MFU 的话临界点还要再低一些）。

| batch | K | 每步 token 数 | 验证是否"免费" | 每步时间（估） |
|---|---|---|---|---|
| 1 | 3 | 4 | 是 | ≈ 10 ms |
| 8 | 3 | 32 | 是 | ≈ 10 ms |
| 32 | 3 | 128 | 是 | ≈ 10–11 ms |
| 64 | 3 | 256 | 接近临界 | ≈ 12 ms |
| 128 | 3 | 512 | **否**，≈ 1.7× 临界 | ≈ 17 ms |
| 128 | 0（不开） | 128 | —— | ≈ 10 ms |

**每步的总时间**再加上 draft：EAGLE 头一层、TP=8 下每卡权重不到 0.3 GB，一次 forward 的权重读取不到 0.1 ms，但 K 步串行、每步几十个 kernel，launch 与同步开销主导，估 0.5–1 ms/步；独立 1B draft model 大约 1–2 ms/步。

**回到我们的例子**（batch=1，300 个输出 token）：

| 配置 | 步数 | 每步 | Decode 总时间 | 相对 |
|---|---|---|---|---|
| 不开投机 | 300 | 10 ms | 3000 ms | 1.0× |
| EAGLE，K=3，平均接受长度 2.5 | 120 | 10 + 3 × 0.7 ≈ 12 ms | ≈ 1450 ms | **≈ 2.1×** |
| EAGLE，K=3，平均接受长度 1.6（难预测的文本） | 188 | 12 ms | ≈ 2250 ms | 1.3× |
| n-gram，K=3，接受长度 1.2（几乎不重复的文本） | 250 | 10 ms（draft 在 CPU） | 2500 ms | 1.2× |
| n-gram，K=3，接受长度 3.5（RAG 抄原文） | 86 | 10 ms | 860 ms | **3.5×** |
| EAGLE，K=3，batch=128 | 每请求 120 步 | 17 + 2 ≈ 19 ms | ≈ 2280 ms | 1.3×（吞吐口径下还要再看：128 个请求同时慢了 1.9 倍） |

最后一行是关键：**batch=128 时，投机解码把每一步拖慢 1.9 倍，只换来 2.5 倍的步数减少**——单请求延迟略好，系统吞吐反而下降（同样的 GPU 时间产出的 token 变少）。这和第六篇结尾"高并发下可能负收益"是同一件事，现在有了数字。

**什么时候不该开**：

- 每步 token 数已接近或超过 `T_ridge`（高并发、吞吐优先）；
- 输出很短（prefill 主导，投机对 prefill 无效——上一节说过 draft 在 prefill 阶段根本不调度）；
- 请求普遍需要 `min_p`、`logit_bias` 或自定义 logits processor（会被禁用 / 静默失效）；
- 采样温度高、文本高熵——接受率随之下降，而 draft 成本不变；
- 需要异步调度但 proposer 不在支持列表里。

`SpeculativeConfig.num_speculative_tokens_per_batch_size` 提供了一个折中：按 batch size 区间动态选 K（`build_dynamic_sd_schedule_lookup()`，`vllm/v1/spec_decode/dynamic/utils.py`），大 batch 时自动降到 0。但 `VllmConfig` 会在 DP > 1 时禁用它——不同 DP rank 若选了不同 K，集合通信会死锁。

## 四、结构化输出：把 grammar 变成每步一张掩码

### 1. 约束在哪里生效：四个位置

结构化输出（`SamplingParams.structured_outputs`，支持 `json` / `json_object` / `regex` / `choice` / `grammar` / `structural_tag`，对应 `StructuredOutputOptions` 枚举）看起来只是"采样时把不合法 token 屏蔽掉"，但它在请求生命周期上分布在四个位置：

```text
① 前端（vllm/sampling_params.py: SamplingParams._validate_structured_outputs）
   校验请求、选后端：backend="auto" 时先试 xgrammar，不支持的 JSON schema 特性落到 guidance，
   非 tekken 的 Mistral tokenizer 落到 outlines
        │
② EngineCore.add_request → StructuredOutputManager.grammar_init()（vllm/v1/structured_output/__init__.py）
   把编译提交到 ThreadPoolExecutor，request.structured_output_request.grammar = Future
   请求状态 = WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
        │
③ 每一步：
   Scheduler.schedule() → 只调度 grammar 已就绪的请求
   Scheduler.get_grammar_bitmask() → GrammarOutput(request_ids, bitmask: np.ndarray)
   GPUModelRunner.sample_tokens(grammar_output) → apply_grammar_bitmask(logits) → Sampler
        │
④ 每一步之后：
   Scheduler.update_from_output() → grammar.accept_tokens(new_token_ids) 推进 FSM
```

第 ② 步的**异步**是关键设计：JSON schema 编译成 grammar 可能要几十到几百毫秒，甚至更久，如果在调度循环里同步做，整个引擎会为一个请求停摆。所以编译进线程池，请求进一个专门的等待状态（第十四篇状态机里的 `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR` 就是它）。调度器每步在 `_try_promote_blocked_waiting_request()` 里看一眼 `structured_output_req.grammar`——这个 property（`StructuredOutputRequest.grammar`，`vllm/v1/structured_output/request.py`）会以 **100 微秒**的超时去 poll Future，好了就把请求放回 `WAITING`，编译抛异常的请求记入 `grammar_compile_error_reqs` 单独失败。代价是这个请求的 TTFT 多了一段编译时间；收益是其他请求完全不受影响。`external_launcher` 模式是例外——各 TP rank 各有调度器，异步编译会让状态转换在不同 rank 上时序不一致，于是退回同步。

第 ③ 步里有一个时序细节值得看 `EngineCore.step()`（`vllm/v1/engine/core.py`）：

```python
scheduler_output = self.scheduler.schedule(...)
future = self.model_executor.execute_model(scheduler_output, non_block=True)   # 先把 forward 发出去
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)          # GPU 在算 forward 时，CPU 填 bitmask
model_output = future.result()
if model_output is None:
    model_output = self.model_executor.sample_tokens(grammar_output)            # 掩码在采样前才送到 worker
```

**bitmask 的生成与 forward 是重叠的**。这是 V1 把 `execute_model` 和 `sample_tokens` 拆成两个 RPC 的原因之一——掩码不需要在 forward 之前就绪，只需要在采样之前就绪。

### 2. 后端抽象

`vllm/v1/structured_output/backend_types.py` 定义了两层：

```python
@dataclass
class StructuredOutputBackend(ABC):                 # 引擎级，一个引擎只有一个
    def compile_grammar(self, request_type: StructuredOutputOptions, grammar_spec: str) -> StructuredOutputGrammar
    def allocate_token_bitmask(self, max_num_seqs: int) -> torch.Tensor
    def destroy(self)

class StructuredOutputGrammar(ABC):                 # 请求级，一个请求一个
    def accept_tokens(self, request_id, tokens: list[int]) -> bool   # 推进 FSM
    def validate_tokens(self, tokens: list[int]) -> list[int]        # 只校验不推进，返回合法前缀
    def rollback(self, num_tokens: int)                              # 回退 FSM
    def fill_bitmask(self, bitmask: torch.Tensor, batch_index: int)  # 填当前状态的掩码
    def is_terminated(self) -> bool
    def reset(self)
```

四个后端：`XgrammarBackend`（`backend_xgrammar.py`，`xgr.GrammarCompiler` 编译、`xgr.GrammarMatcher` 做 FSM）、`GuidanceBackend`（`backend_guidance.py`）、`OutlinesBackend`（`backend_outlines.py`）、`LMFormatEnforcerBackend`（`backend_lm_format_enforcer.py`）。`StructuredOutputManager.grammar_init()` 在第一个请求到来时按 `sampling_params.structured_outputs._backend` 实例化后端，注释明说 "We only support a single backend. We do NOT support different backends on a per-request basis"。`StructuredOutputsConfig.backend` 默认 `"auto"`，前端的选择逻辑（第 1 节 ①）会为每个请求写入 `_backend`，但引擎只认第一个——所以混用不同后端的请求会在第二个不同后端的请求上报错。

`validate_tokens()` / `rollback()` 两个方法不是 grammar 库的常规接口，它们是为投机解码加的（第 4 节）。

### 3. bitmask 的生成与应用

**形状**：xgrammar 的 bitmask 是 int32，每 32 个 token 一个 word，`128256 / 32 = 4008` 个 int32，**每行 16 KB**。`StructuredOutputManager.grammar_bitmask()` 第一次调用时一次性分配 `max_num_seqs × (1 + num_spec_tokens)` 行——每个请求 1 行给普通 token 或 bonus，K 行给 K 个 draft 位置。

**生成**（CPU，调度器进程）：对本步每个使用结构化输出、且**不在 prefill chunk 中**的请求（`get_grammar_bitmask()` 过滤 `is_prefill_chunk`——prefill 阶段不采样，不需要掩码），调用 `grammar.fill_bitmask(bitmask, index)`；不需要约束的行（reasoning 中、grammar 已终止）填全 `-1`（`_full_mask`，即全部允许）。请求数超过 128 且没开投机时，按 16 个一组投进 `executor_for_fillmask` 线程池并行填。返回值转成 `np.ndarray`——注释说明是为了跨进程序列化效率。

**应用**（GPU，worker 进程）：`apply_grammar_bitmask()`（`vllm/v1/structured_output/utils.py`）做两件事：

1. **重排**。调度器给的 bitmask 行序是 `structured_output_request_ids` 的顺序，worker 的 logits 行序是 `input_batch.req_ids` 的顺序，而且投机解码下每个请求占 `1 + num_drafts` 行。函数先按 `input_batch.req_ids` 算出每个结构化请求的 logits 起始行（累加前面请求的 draft 数），再把 bitmask 搬到一张 `[logits.shape[0], 4008]` 的 pinned 张量里对应的行上，其余行全 `-1`；
2. **打掩码**。`xgr.apply_token_bitmask_inplace(logits, bitmask, indices)`——一个 kernel，对 `indices` 指定的行把 bit 为 0 的 token 置 `-inf`。所有行都要打时 `indices=None`。

注意它作用在**进 Sampler 之前的原始 logits** 上。掩码是硬 `-inf`，放在温度、penalties 之前或之后对"哪些 token 可能被选中"没有影响，但会影响 logprobs：`raw_logprobs` 模式下返回的 logprobs 是打掩码之后、其他处理之前的分布。

**回到我们的例子**：batch=64 全部开 JSON 约束、不开投机：每步 CPU 填 64 行、H2D 传 `64 × 16 KB = 1 MB`、GPU 一个掩码 kernel 扫 33 MB logits。三项都在百微秒量级。

### 4. 与投机解码同时开启

两个扩展都想改"每步决定多少 token"，叠加时需要两层配合：

**① draft 先过一遍 grammar。** `Scheduler.update_draft_token_ids()` 拿到 proposer 的 draft 后，对结构化请求调 `grammar.validate_tokens(spec_token_ids)`——**不推进 FSM**，只返回合法的最长前缀。不合法的尾部直接扔掉（同步路径）或补 `-1`（异步路径 `update_draft_token_ids_in_output()`，因为 `scheduled_spec_decode_tokens` 的长度已经定了）。这是为什么 `StructuredOutputGrammar` 需要 `validate_tokens()` 这个"只看不动"的方法。

**② bitmask 每个位置一行。** 第 i 个 draft 位置的合法 token 集合取决于前 i-1 个 draft 都被接受时 FSM 的状态。`grammar_bitmask()` 的循环因此对每个请求：填第 0 行（当前状态）→ `accept_tokens([d₁])` 推进 → 填第 1 行 → `accept_tokens([d₂])` → … → 填 bonus 行 → **`rollback(state_advancements)` 全部回退**。遇到 `-1`（无效 draft）就停止推进、后续行不再约束。于是 FSM 每步被推进 K 次再回退 K 次——结构化输出的 CPU 成本随 K 线性增长，这也是并行填充路径要求 `max_num_spec_tokens == 0` 的原因（推进/回退是有状态的，不好切成独立任务）。

真正采样时，bonus 行走普通 `Sampler`、target 行走 `RejectionSampler`，两者的输入 logits 都已被各自的行掩码挖过，所以**被接受的 draft 与 recovered token 都一定合法**——`update_from_output()` 里的 `accept_tokens()` 若返回 False，源码直接记 error 并把请求置为 `FINISHED_ERROR`，因为这在设计上不应发生。

### 5. reasoning 模式下的跳过

推理模型（DeepSeek-R1 一类）的输出前半段是 `<think>…</think>`，用户要的 JSON 在后面。如果从第一个 token 就打掩码，模型连"思考"都得用 JSON 写。`StructuredOutputsConfig.reasoning_parser` 指定一个 `ReasoningParser`，`StructuredOutputManager` 用它做三件事：

- `should_fill_bitmask(request)`：`reasoning_ended` 为 False 时返回 False，该请求这一步填全 `-1`（不约束）；`enable_in_reasoning=True` 则始终约束；
- `should_advance(request, new_token_ids)`：用 `reasoner.is_reasoning_end_streaming()` 检测本步的新 token 里是否出现了思考结束标记，出现则置 `reasoning_ended=True` 并记下边界 `reasoning_end_token_index`；
- `trim_reasoning_for_advance()`：思考结束的那一步，新 token 可能是"思考尾巴 + 结束标记 + JSON 开头"混在一起，只把结束标记之后的部分喂给 `accept_tokens()`。

和投机解码叠加时更绕：一个 draft 窗口内可能刚好跨过结束标记。`grammar_bitmask()` 的循环里 `detect_reasoning_end` 分支会对每个 draft 位置模拟"到这里为止的序列"判断是否结束，结束后把该窗口剩余位置改为约束、但**容忍**这些 draft 被 grammar 拒绝（它们是在没有掩码时产生的，不保证合法）。

### 6. 代价

| 项 | 量级 | 谁付 | 何时付 |
|---|---|---|---|
| grammar 编译 | 简单 regex / choice：毫秒级；复杂 JSON schema：几十到几百毫秒，极端情况秒级 | 该请求的 TTFT | 请求进入时，异步 |
| bitmask 分配 | `max_num_seqs × (1+K) × 16 KB`，256 × 4 × 16 KB = 16 MB | 引擎进程内存 | 一次 |
| 每步填充 | 每请求每位置一次 `fill_bitmask`（数十微秒量级）+ 投机下 K 次 `accept_tokens` / 1 次 `rollback` | 调度器进程 CPU，与 forward 重叠 | 每步 |
| 每步传输 | `请求数 × (1+K) × 16 KB` 序列化 + H2D | IPC 与 PCIe | 每步 |
| 每步掩码 | 一个 GPU kernel，读写整张 logits | GPU，百微秒以下 | 每步 |
| FSM 推进 | `accept_tokens()`，微秒级 | 调度器进程 | 每步之后 |

结构化输出的"贵"几乎全在 CPU 和延迟上，GPU 侧近乎免费。当调度器进程是瓶颈（大 batch、小模型、每步只有几毫秒）时，每步几毫秒的 CPU 填充就不再能被 forward 遮住，这是 `fill_bitmask_parallel_threshold` 存在的场景。

## 五、三者叠加：执行顺序，以及留给后面两篇的问题

### 1. 一步之内的顺序

把三章的内容按一步（`EngineCore.step()`）的时间轴排开，投机解码 + 结构化输出 + penalties 同时开启时是这样：

```text
Scheduler.schedule()
  ├─ num_new_tokens = 1 + K（draft 进预算，预算不够则截断 K）
  ├─ allocate_slots(..., num_lookahead_tokens=K)
  └─ 只调度 grammar 已就绪、不在 prefill chunk 的结构化请求
        │
        ▼
Executor.execute_model(non_block)                      ── GPU：forward 算 num_reqs × (1+K) 个位置
        │                                               ── CPU（同时）：Scheduler.get_grammar_bitmask()
        │                                                    每个结构化请求填 1+K 行，FSM 推进 K 次再 rollback
        ▼
Executor.sample_tokens(grammar_output)                 ── GPU
  ├─ apply_grammar_bitmask(logits)                      ① 硬掩码：不合法 token → -inf
  ├─ RejectionSampler.forward()
  │    ├─ bonus 行 → Sampler：allowed_token_ids → bad_words → min_tokens/logit_bias →
  │    │             penalties → thinking budget → 温度 → min_p → top_k/top_p → 采样
  │    ├─ target 行 → 投机版处理器（penalties/bad_words/min_tokens/thinking budget）
  │    │             → apply_sampling_constraints（温度/top_k/top_p 按 draft 展开）
  │    └─ rejection_sample：greedy kernel / random kernel → [batch, K+1]，-1 占位
  └─ propose_draft_token_ids()                          ② 下一步的 draft（EAGLE：K 步 draft forward，PIECEWISE 图）
        │
        ▼
Scheduler.update_from_output()
  ├─ num_computed_tokens -= num_rejected                ③ KV "回滚"
  ├─ SpecDecodingStats.observe_draft()
  ├─ grammar.accept_tokens(new_token_ids)               ④ FSM 真正推进（reasoning 已结束的部分）
  └─ update_draft_token_ids()：grammar.validate_tokens(draft) 过滤下一步的 draft
```

三点值得再说一次：掩码在最前（①），所以后面所有处理器和拒绝采样看到的都是已经挖过的分布；draft 在采样之后立刻产生（②），但要等调度器用 grammar 过滤后才成为下一步的输入；`num_computed_tokens` 的回滚（③）是唯一的"KV 回滚"，物理槽位不动。

### 2. 留给第八篇（多卡）的问题

本章只把问题摆出来，答案在下一篇：

- **draft 模型放哪？** `DraftModelProposer` 要求 draft 的 TP 等于 target 的 TP；EAGLE 头只有一层，按 TP=8 切开后每卡的矩阵很小，通信开销（每层两次 all-reduce）相对计算的比例远高于 target——draft 那 K 步的每一步都要付 80 层 target 里一层的通信，却只换来一层的计算。是否值得让 draft 走 TP，还是用 `draft_tensor_parallel_size` 单独设？
- **logits 与 bitmask 在哪张卡上？** `LogitsProcessor._gather_logits()`（`vllm/model_executor/layers/logits_processor.py`）在 CUDA 上用 `tensor_model_parallel_gather` 把词表分片的 logits 收到 rank 0，其他 rank 拿到 `None`。于是掩码、采样、拒绝采样都只在一张卡上发生，但 `GrammarOutput` 仍然要通过 executor 的 RPC 广播到所有 worker。PP > 1 时 logits 在最后一个 stage，采样结果又要回传给第一个 stage 作为下一步输入（`_pp_broadcast_prev_sampled_token_ids()`）。
- **DP 下的一致性。** 动态 K 与 DP 互斥（不同 rank 选不同 K 会让集合通信对不上）；结构化输出的异步编译与 `external_launcher` 互斥。多卡让"每个 rank 做同样的事"成为硬约束，而本篇的三种扩展都在引入 per-request 的差异。

### 3. 留给第十二篇（PD 分离）的问题

- **EAGLE 的第一个 draft 需要什么？** 它需要 target 对 prompt 最后一个 token 的 hidden state——这个东西在 Prefill 实例上产生，KV Transfer 只传 KV 不传 hidden state。Decode 实例是重算最后一个 token，还是 P 侧多传一段？第 3 节的 `drop_eagle_block` 已经是同一问题在 prefix cache 上的影子。
- **grammar 在哪一侧编译？** 编译发生在 `EngineCore.add_request()`，请求先到哪个实例就在哪里编译；但掩码只在 decode 时需要。P 侧编译是浪费，D 侧编译则 TTFT 已经算完、编译时间落到第一个 decode token 上。
- **lookahead 槽位与 KV 传输。** `allocate_slots()` 里 `num_lookahead_tokens` 在 `load_kv_async` 为真时被置 0（源码：`limit_lookahead_tokens`）——等远端 KV 到达时不预留，到达后再补。这是 PD 分离已经渗入投机解码实现的一处。

<details markdown="1">
<summary><b>📂 本章源码导航</b></summary>

**采样与 logits processors**

| 想看什么 | 从哪开始 |
|---|---|
| **采样流水线** | `vllm/v1/sample/sampler.py` → `Sampler.forward()` / `sample()` / `apply_logits_processors()` |
| 每请求参数的容器 | `vllm/v1/sample/metadata.py` → `SamplingMetadata` |
| top-k / top-p / 随机采样 | `vllm/v1/sample/ops/topk_topp_sampler.py` → `TopKTopPSampler`、`apply_top_k_top_p()`、`random_sample()`；Triton 版 `topk_topp_triton.py` |
| penalties | `vllm/v1/sample/ops/penalties.py` → `apply_all_penalties()`；`vllm/model_executor/layers/utils.py` → `apply_penalties()`、`get_token_bin_counts_and_mask()` |
| LogitsProcessor 接口与 BatchUpdate | `vllm/v1/sample/logits_processor/interface.py` |
| 内置处理器 | `vllm/v1/sample/logits_processor/builtin.py` → `MinPLogitsProcessor`、`LogitBiasLogitsProcessor`、`MinTokensLogitsProcessor`、`process_dict_updates()` |
| 加载、注册与 Adapter | `vllm/v1/sample/logits_processor/__init__.py` → `build_logitsprocs()`、`AdapterLogitsProcessor`、`LOGITSPROCS_GROUP` |
| 持久 batch 的增删移动 | `vllm/v1/sample/logits_processor/state.py` → `BatchUpdateBuilder`；`vllm/v1/worker/gpu_input_batch.py` → `InputBatch.refresh_metadata()` |
| thinking budget | `vllm/v1/sample/thinking_budget_state.py` → `ThinkingBudgetStateHolder` |

**投机解码**

| 想看什么 | 从哪开始 |
|---|---|
| 配置与方法枚举 | `vllm/config/speculative.py` → `SpeculativeConfig`、`SpeculativeMethod`、`use_eagle()`、`num_speculative_tokens_per_batch_size` |
| Proposer 家族 | `vllm/v1/spec_decode/`：`llm_base_proposer.py`（`SpecDecodeBaseProposer`）、`eagle.py`、`draft_model.py`、`medusa.py`、`ngram_proposer.py`、`ngram_proposer_gpu.py`、`suffix_decoding.py`、`custom_class_proposer.py` |
| **调度器侧** | `vllm/v1/core/sched/scheduler.py` → `Scheduler.__init__`（`num_lookahead_tokens`）、`schedule()`（`scheduled_spec_decode_tokens`）、`update_from_output()`（回滚与 stats）、`update_draft_token_ids()` / `update_draft_token_ids_in_output()` |
| KV 预留 | `vllm/v1/core/kv_cache_manager.py` → `KVCacheManager.allocate_slots(num_lookahead_tokens=...)` |
| EAGLE 与 prefix cache | `vllm/v1/core/kv_cache_coordinator.py`（`eagle_group_ids`）；`vllm/v1/core/single_type_kv_cache_manager.py` → `find_longest_cache_hit(drop_eagle_block=...)` |
| **model runner 侧** | `vllm/v1/worker/gpu_model_runner.py` → `_calc_spec_decode_metadata()`、`_sample()`、`sample_tokens()`、`propose_draft_token_ids()`、`_is_uniform_decode()`、`uniform_decode_query_len` |
| 元数据 | `vllm/v1/spec_decode/metadata.py` → `SpecDecodeMetadata` |
| 拒绝采样 | `vllm/v1/sample/rejection_sampler.py` → `RejectionSampler.forward()`、`rejection_sample()`、`apply_sampling_constraints()`、`rejection_greedy_sample_kernel` / `rejection_random_sample_kernel`、`sample_recovered_tokens()` |
| 指标 | `vllm/v1/spec_decode/metrics.py` → `SpecDecodingStats`、`SpecDecodingLogging`、`SpecDecodingProm` |
| CUDA graph 的 key | `vllm/v1/cudagraph_dispatcher.py` → `CudagraphDispatcher.initialize_cudagraph_keys(uniform_decode_query_len=...)` |

**结构化输出**

| 想看什么 | 从哪开始 |
|---|---|
| 前端校验与后端选择 | `vllm/sampling_params.py` → `SamplingParams._validate_structured_outputs()`；`vllm/config/structured_outputs.py` → `StructuredOutputsConfig` |
| **管理器** | `vllm/v1/structured_output/__init__.py` → `StructuredOutputManager.grammar_init()` / `grammar_bitmask()` / `should_fill_bitmask()` / `should_advance()` |
| 后端抽象 | `vllm/v1/structured_output/backend_types.py` → `StructuredOutputBackend`、`StructuredOutputGrammar`、`StructuredOutputOptions` |
| 各后端 | `backend_xgrammar.py`（`XgrammarBackend`、`XgrammarGrammar`）、`backend_guidance.py`、`backend_outlines.py`、`backend_lm_format_enforcer.py` |
| 请求级状态与 Future | `vllm/v1/structured_output/request.py` → `StructuredOutputRequest.grammar` |
| 调度器侧 | `vllm/v1/core/sched/scheduler.py` → `get_grammar_bitmask()`、`_try_promote_blocked_waiting_request()`、`update_from_output()` 中的 `accept_tokens` |
| 引擎循环的时序 | `vllm/v1/engine/core.py` → `EngineCore.step()` / `step_with_batch_queue()` |
| GPU 上打掩码 | `vllm/v1/structured_output/utils.py` → `apply_grammar_bitmask()`；调用处 `GPUModelRunner.sample_tokens()` |
| GrammarOutput | `vllm/v1/core/sched/output.py` → `GrammarOutput`、`SchedulerOutput.scheduled_spec_decode_tokens` / `num_invalid_spec_tokens` |

</details>


## 六、本文小结

- 采样、投机解码、结构化输出是对"从 logits 到 token"这最后一步的三种扩展：分别改**分布本身**、**每步决定的位置数**、**分布的支撑集**。它们都不只改 Sampler——投机解码牵动调度预算、KV 预留与 CUDA graph 的形状，结构化输出牵动请求状态机与每步的 CPU 工作，连 penalties 也要求每步维护并上传历史输出。
- `Sampler` 的流水线以"是否改变 argmax"为组织原则，让 greedy 与随机请求走同一条向量化路径；`LogitsProcessor` 接口之所以长成 `update_state(BatchUpdate)` 的样子，是因为 `InputBatch` 是持久 batch，处理器状态必须跟着增删移动同步。V1 只接受引擎级注册的处理器类，开投机时禁用自定义处理器、`min_p` 与 `logit_bias`。
- 采样参数里真正有成本的是 penalties（每步两张 `[B, V+1]` int64 直方图 + 历史输出上传）和 seed（逐请求循环、FlashInfer 路径失效）；其余淹没在 forward 里。
- 投机解码在 vLLM 里是三个角色加两处约束：Proposer 出 draft，调度器把 `1+K` 算进预算、用 `num_lookahead_tokens` 预留 KV、用 `num_computed_tokens -= num_rejected` 回滚；`RejectionSampler` 在 `[num_reqs + Σdrafts, V]` 的 logits 上用两个 Triton kernel 验证。"纯 decode batch"的定义变成每请求 `1+K` 个 token，预算截断 draft 会让 batch 掉出 FULL 图；EAGLE 命中 prefix cache 时主动少算一块。
- 收益的临界点可以算：H100 上一步约 300 个 token 是 memory-bound 与 compute-bound 的分界，`batch × (1+K)` 超过它验证就不再免费。例子里 batch=1、接受长度 2.5 时约 2.1×；batch=128 时每步慢 1.9 倍、吞吐反降。`SpecDecodingStats` 的按位置接受率是调 K 的依据。
- 结构化输出分布在四个位置：前端选后端、`grammar_init()` 异步编译（请求进 `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`）、每步 CPU 填 bitmask 并与 forward 重叠、GPU 上一个 kernel 打 `-inf`、步后 `accept_tokens()` 推进 FSM。与投机叠加时 draft 先经 `validate_tokens()` 过滤，bitmask 每个位置一行，FSM 每步推进 K 次再 `rollback()`。它的成本几乎全在 CPU 与 TTFT 上。
- 三者叠加的顺序：掩码最先、然后是 bonus 行的完整 Sampler 与 target 行的投机版处理器、拒绝采样、紧接着 draft 下一步；调度器在步后回滚与推进 FSM。draft 模型的 TP 绑定、logits 只在 rank 0、EAGLE 需要 P 侧的 hidden state——这些是留给多卡与 PD 分离的问题。


## 下一篇

[Multi-GPU：一张卡不够时如何扩展？](/multi-gpu-scaling-strategies.html)
