---
layout: slides
title: "读 Hugging Face 源码：从 from_pretrained 到 GRPO 的 loss"
subtitle: "六行 LoRA SFT 代码各经过哪些文件的哪些函数，按 ↓ 看图"
permalink: /slides/hf-source-reading.html
series: hf-source-reading
date: 2026-05-25 23:30:00 +0800
author: arganzheng
description: "《读 Hugging Face 源码》系列的分享用幻灯片：from_pretrained 的六步与一次前向、generate 的处理器串与 _sample 循环、tokenizer.json 的五段与 datasets 的 fingerprint、peft 的 lora.Linear 与 trl 的 SFT / DPO / GRPO loss——transformers 5.17.0、tokenizers 0.23.2、datasets 5.0.1、peft 0.21.0、trl 1.13.0。"
theme: white
transition: slide
---

## 这个系列回答一个问题

> 工具箱《Hugging Face 生态》那六行 LoRA SFT 代码——`from_pretrained`、`generate`、`apply_chat_template`、`load_dataset(...).map`、`get_peft_model`、`SFTTrainer`——执行时各经过了**哪些文件的哪些函数**？

**骨架是「注册表 + 字符串键」：找到表、找到 `compute_loss`、找到 `forward` 里那一行调用，往下追**

<aside class="notes" markdown="1">
总纲：/reading-hugging-face-source-code.html。版本基线 transformers 5.17.0、tokenizers 0.23.2、datasets 5.0.1、peft 0.21.0、trl 1.13.0，模型 Qwen2.5-0.5B，数据 HuggingFaceH4/no_robots。引用只给路径与函数名，不给行号。
</aside>

---

## 四篇沿一次 LoRA SFT 的数据流

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 200}}}%%
%% 图：四篇沿一次 LoRA SFT 的数据流
flowchart LR
    D["<b>tokenizers · datasets</b><br/>messages → input_ids / labels<br/>Arrow 文件 → collate_fn"] --> M["<b>transformers 模型侧</b><br/>from_pretrained → nn.Module<br/>forward → logits → ForCausalLMLoss"]
    M --> G["<b>generate</b><br/>LogitsProcessor 一串<br/>_sample 的 while 循环"]
    P["<b>peft · trl</b><br/>lora.Linear 的一行<br/>SFT 的 -100 · dpo_loss · GRPO"] --> M
    D --> P
```

模型两端都是整数 `input_ids`：进的一端是 tokenizer 与 datasets，出的一端是 `generate`

<aside class="notes" markdown="1">
四篇：模型侧 /transformers-from-pretrained-to-forward-and-loss.html，generate /transformers-generate-call-chain.html，数据 /tokenizers-and-datasets-from-messages-to-input-ids.html，peft 与 trl /peft-and-trl-lora-sft-dpo-grpo-in-source.html。
</aside>

<!-- v -->

### 五个库的依赖分层

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 240}}}%%
%% 图：五个库的依赖分层与各自落到的内核
flowchart TB
    subgraph UP["训练与微调"]
        direction LR
        TRL["<b>trl</b><br/>三个 Trainer"] ~~~ PEFT["<b>peft</b><br/>lora.Linear"]
    end
    subgraph MID["模型与数据"]
        direction LR
        TF["<b>transformers</b><br/>from_pretrained · forward · generate"] ~~~ TOK["<b>tokenizers</b><br/>Python 绑定"] ~~~ DS["<b>datasets</b><br/>Dataset · map"]
    end
    subgraph CORE["内核（不是 Python）"]
        direction LR
        TORCH["PyTorch ATen / CUDA"] ~~~ RUST["Rust：tokenizers · safetensors"] ~~~ ARROW["Arrow C++：memory_map"]
    end
    UP --> MID --> CORE
```

上层只 import 下层；读 Python 源码追到最底下一层就停

---

## 01 · 模型侧：先搭骨架，再按名字填

**加载峰值约等于权重本身：meta 骨架 0 字节，mmap 按名字填**

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 230}}}%%
%% 图：from_pretrained 的六步
flowchart TB
    subgraph R1["选类与找文件"]
        direction LR
        A1["<b>AutoConfig</b><br/>config.json：model_type = qwen2"] --> A2["<b>_LazyAutoMapping</b><br/>qwen2 → Qwen2ForCausalLM"] --> A3["<b>找权重文件</b><br/>model.safetensors 或分片 + index"]
    end
    subgraph R2["建模型与填权重"]
        direction LR
        A4["<b>torch.device('meta')</b><br/>cls(config)：只有形状"] --> A5["<b>safe_open + get_slice</b><br/>mmap 按名字填 290 个张量"] --> A6["<b>收尾</b><br/>tie_weights · 初始化缺失键 · eval()"]
    end
    R1 --> R2
```

<aside class="notes" markdown="1">
原文 /transformers-from-pretrained-to-forward-and-loss.html。表存字符串、按需 import，所以「跳转到定义」跳不到 Qwen2ForCausalLM。对不上的键分 missing（随机初始化）/ unexpected（忽略）/ mismatched（默认报错）。返回的模型是 eval 模式。
</aside>

<!-- v -->

### 一次前向的张量形状（Qwen2.5-0.5B）

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 220}}}%%
%% 图：Qwen2.5-0.5B 一次前向的张量形状
flowchart LR
    I["input_ids<br/>#91;B, T#93;"] --> EM["embed_tokens<br/>#91;B, T, 896#93;"]
    EM --> L
    subgraph L["DecoderLayer × 24"]
        direction TB
        Q["<b>q / k / v_proj</b><br/>• q #91;B, 14, T, 64#93;<br/>• k、v #91;B, 2, T, 64#93;"]
        AT["<b>attention → o_proj</b><br/>#91;B, T, 896#93; + 残差"]
        ML["<b>MLP</b><br/>• gate、up #91;B, T, 4864#93;<br/>• down #91;B, T, 896#93; + 残差"]
        Q --> AT --> ML
    end
    L --> H["norm → lm_head<br/>#91;B, T, 151936#93;"]
    H --> LO["ForCausalLMLoss<br/>标量"]
```

<!-- v -->

### 一个键同时选 attention 函数与 mask

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 230}}}%%
%% 图：_attn_implementation 同时决定 attention 函数与 mask 的物化方式
flowchart LR
    K{"config.<br/>_attn_implementation"}
    MF["<b>create_causal_mask</b><br/>kv_idx #60;= q_idx"]
    K -- "eager" --> E1["<b>eager_attention_forward</b><br/>repeat_kv · 物化 T × T"]
    K -- "sdpa（默认）" --> S1["<b>sdpa_attention_forward</b><br/>不物化 T × T"]
    K -- "flash_attention_2" --> F1["<b>flash_attention_forward</b>"]
    MF -- "浮点 #91;B, 1, T, T#93;" --> E1
    MF -- "布尔或 None" --> S1
    MF -- "只给序列长度" --> F1
```

eager 与 sdpa：fp32 下 logits 最大差 $$1.7 \times 10^{-4}$$、argmax 全同；bf16 下差到 1.0

<!-- v -->

### KV cache：每层一个 `torch.cat`

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 260}}}%%
%% 图：prefill 5 个 token 与 decode 1 个 token 时 DynamicCache 和形状的变化
flowchart LR
    subgraph PF["prefill：input_ids #91;1, 5#93;"]
        direction TB
        P1["position_ids #91;0 … 4#93;"] --> P2["keys #91;1, 2, 5, 64#93;"] --> P3["get_seq_length() = 5"]
    end
    subgraph DC["decode：input_ids #91;1, 1#93;"]
        direction TB
        D1["position_ids #91;5#93;"] --> D2["keys #91;1, 2, 6, 64#93;"] --> D3["logits #91;1, 1, 151936#93;"]
    end
    PF -- "past_key_values" --> DC
```

`position_ids` 从 `get_seq_length()` 来：忘了传 cache，每个新 token 都被当成第 0 位

<aside class="notes" markdown="1">
loss 不在建模文件里：self.loss_function 按类名分派到 loss/loss_utils.py 的 ForCausalLMLoss——升 fp32、labels 末尾垫 -100 再左移一位、ignore_index=-100、num_items_in_batch 修梯度累积。配套脚本 labels = input_ids 时 loss 3.9506，与手算 cross_entropy 相等。
</aside>

---

## 02 · generate：一串函数 + 一个循环

**配置 → 处理器串 → 停止条件 → `_sample` 循环；顺序由代码定**

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 220}}}%%
%% 图：_sample 循环的一步
flowchart TB
    subgraph R1["准备输入与前向"]
        direction LR
        PF["<b>_prefill</b><br/>整段 prompt 前向一次<br/>logits_to_keep=1"] --> L1["prepare_inputs<br/>只切最后 1 个 token"] --> L2["forward<br/>带 past_key_values"]
    end
    subgraph R2["选 token 与停止判断"]
        direction LR
        L3["logits#91;:, -1#93;<br/>→ 处理器串"] --> L4["multinomial<br/>或 argmax"] --> L5["cat · unfinished &=<br/>~stopping_criteria"]
    end
    R1 --> R2
    R2 -- "还有没停的：回到 prepare_inputs" --> R1
```

<aside class="notes" markdown="1">
原文 /transformers-generate-call-chain.html。配置三层优先级：kwargs > 仓库 generation_config.json > 全局默认；Qwen2.5 不传长度会生成到 2048。不认识的 kwargs 转给 forward，拼错的由 _validate_model_kwargs 报错。停了的序列填 pad 继续跟跑——静态 batch。DeferredStopCheck 把 GPU→CPU 同步推迟一步。
</aside>

<!-- v -->

### 处理器串：第一步还剩几个有限值

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 220}}}%%
%% 图：配套脚本第一步的分数经过处理器串时剩下的有限值个数
flowchart TB
    subgraph R1["改分数"]
        direction LR
        R["logits<br/>151936 个"] --> P1["repetition penalty 1.1<br/>个数不变"] --> P2["temperature 0.7<br/>第一名 > 0.9"]
    end
    subgraph R2["截断与抽样"]
        direction LR
        P3["top_k 50<br/>剩 50 个"] --> P4["top_p 0.9<br/>剩 1 个"] --> S["multinomial<br/>= argmax"]
    end
    R1 --> R2
```

顺序由代码固定：penalty → temperature → top-k → top-p；`do_sample=False` 时 warper 一个都不装

<aside class="notes" markdown="1">
get_generation_mode：num_beams > 1 → beam search；传 assistant_model → assisted（投机解码）；否则 do_sample 决定 SAMPLE / GREEDY，两者是同一个 _sample，只在选 token 那一行分叉。contrastive 等模式 5.x 移到 Hub 的 custom_generate。
</aside>

---

## 03 · tokenizers 与 datasets：Python 只是壳

**`tokenizer.json` 描述 Rust 流水线；`Dataset` 是 Arrow mmap 视图**

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 170}}}%%
%% 图：map 用 fingerprint 决定重跑还是读缓存
flowchart LR
    IN["<b>哈希输入</b><br/>• 上游 fingerprint<br/>• 变换名 map<br/>• 参数，fn 经 dill"] --> H{"fn 能序列化？"}
    H -- "能" --> FP["cache-‹fingerprint›.arrow"] --> EX{"文件已存在？"}
    H -- "不能" --> RND["随机 fingerprint"] --> RUN
    EX -- "是" --> HIT["Dataset.from_file<br/>0.7 秒"]
    EX -- "否" --> RUN["_map_single 重跑<br/>9500 条 4.3 秒"]
```

改函数里一个 print → 字节码变 → fingerprint 变 → 整个重跑

<aside class="notes" markdown="1">
原文 /tokenizers-and-datasets-from-messages-to-input-ids.html。tokenizer.json 五段：normalizer（NFC）→ pre_tokenizer（GPT-2 正则 + ByteLevel，数字逐位）→ BPE model → post_processor → decoder。chat template 是沙箱 Jinja；推理必传 add_generation_prompt；Qwen2.5 的模板没有 generation 标记，所以拿不到 assistant mask。batched=True 是 map 最大的性能开关。
</aside>

<!-- v -->

### 三个词表大小在 id 轴上的位置

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 260}}}%%
%% 图：Qwen2.5 的三个词表大小在 token id 轴上的位置
flowchart LR
    subgraph EMB["config.vocab_size = 151936"]
        direction LR
        subgraph LEN["len(tok) = 151665"]
            direction LR
            BPE["<b>id 0 – 151642</b><br/>BPE 词表<br/>tok.vocab_size = 151643"]
            ADD["<b>id 151643 – 151664</b><br/>22 个 added_tokens<br/>绕过 BPE"]
            BPE ~~~ ADD
        end
        PAD["<b>id 151665 – 151935</b><br/>271 行无 token<br/>对齐余量"]
        LEN ~~~ PAD
    end
```

加几个特殊 token 只是在 151665 之后续编号，不到 151936 就不用 resize

<!-- v -->

### `shuffle` 只多一张行号表

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 260}}}%%
%% 图：Dataset 的三个字段与 ds[i] 的取数路径
flowchart TB
    subgraph DSO["Dataset 对象"]
        direction LR
        D1["_data：MemoryMappedTable"] ~~~ D2["_indices：行号表，shuffle 之后才有"] ~~~ D3["_format_type"]
    end
    DSO -- "ds#91;i#93;" --> X{"有 _indices？"}
    X -- "否" --> R1["取第 i 行，顺序读"]
    X -- "是" --> R2["查 _indices#91;i#93; = j，取第 j 行，随机读"]
    R1 --> FM["formatter → dict / Tensor"]
    R2 --> FM
```

`shuffle` 只多写一张 `_indices`；之后的 `map` 变慢：先 `map` 再 `shuffle`，或 `flatten_indices()`

---

## 04 · peft 与 trl：换层与填 -100

**`setattr` 换掉 `nn.Linear`；`-100` 在训练前那一次 `map` 里填**

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 260}}}%%
%% 图：get_peft_model 之后从 PeftModel 到一个 q_proj 的对象路径
flowchart TB
    subgraph R1["三层套娃"]
        direction LR
        PM["<b>PeftModel</b>"] -- ".base_model" --> LM["<b>LoraModel</b>"] -- ".model" --> QM["<b>Qwen2ForCausalLM</b><br/>原对象，原地改"]
    end
    subgraph R2["layers#91;0#93;.self_attn.q_proj"]
        direction LR
        AT["Qwen2Attention<br/>forward 不改"] -- ".q_proj" --> LL["<b>lora.Linear</b><br/>• base_layer 冻结，scaling = 2.0<br/>• lora_A kaiming，lora_B 全零"]
    end
    R1 --> R2
```

<aside class="notes" markdown="1">
原文 /peft-and-trl-lora-sft-dpo-grpo-in-source.html。forward 只多一项 lora_B(lora_A(dropout(x))) * scaling，先算瘦的，不物化 BA。24 层 × 7 = 168 处替换，可训练 8.8M（1.75%），state_dict 只存 adapter。all-linear 排除 lm_head；列表 target_modules 按最后一段匹配，字符串走 re.fullmatch。SFT：build_labels 是一次 datasets.map，所有 mask 位为 1 才等于 input_ids；collator 只 pad，labels 用 -100；padding_free 时每段首 token 置 -100。
</aside>

<!-- v -->

### DPO：参考 log p 的三种来源

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 220}}}%%
%% 图：DPOTrainer 一个 batch 的策略 log p 与参考 log p 各从哪来
flowchart LR
    B["batch：prompt<br/>chosen · rejected"] --> P{"precompute？"}
    P -- "是" --> PC["训练前算好的两列"]
    P -- "否" --> M{"peft 且无 ref_model？"}
    M -- "是" --> NR["disable_adapter()<br/>同一模型再前向"]
    M -- "否" --> RM["ref_model 前向<br/>多一份基座"]
    PC --> L["<b>dpo_loss</b><br/>-logsigmoid(β·Δ)"]
    NR --> L
    RM --> L
```

策略侧：`concatenated_forward` 把 chosen、rejected 拼成 2B 条前向一次

<!-- v -->

### GRPO：rewards 按组变成 advantages

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 220}}}%%
%% 图：GRPO 一步里 rewards 按组变成 advantages 的形状变化
flowchart TB
    subgraph R1["采样与打分"]
        direction LR
        GEN["每个 prompt 采 G 条<br/>共 B·G 条"] --> RW["reward_funcs<br/>rewards #91;B·G#93;"] --> V["view(-1, G)<br/>#91;B, G#93;"]
    end
    subgraph R2["组内归一与 loss"]
        direction LR
        MS["行均值、行标准差<br/>repeat_interleave"] --> AD["A = (r − mean)<br/>/ (std + 1e-4)"] --> LS["−min(ratio·A, clip·A)<br/>+ β·KL，按 loss_type 归一"]
    end
    R1 --> R2
```

`loss_type`：grpo 每条按长度平均 · bnpo 全 batch token 平均 · dr_grpo 除以常数

---

## 贯穿全系列的四条线

| 线 | 在哪 |
|---|---|
| **注册表 + 字符串键** | `model_type` → 模型类、`_attn_implementation` → attention、类名 → `LOSS_MAPPING`、`loss_type` → 分支、`target_modules` → 层 |
| **Rust / Arrow 内核 + Python 壳** | `encode_batch` 之下是 Rust，`_data` 之下是 Arrow mmap |
| **`-100` 串起三个库** | assistant mask → `build_labels` 与 collator → `ForCausalLMLoss` 的 `ignore_index` |
| **一个上下文省一份模型** | `disable_adapter()` 就是参考模型，DPO 与 GRPO 都不用再加载基座 |

<!-- v -->

### 一次 LoRA SFT 里各库出场的时刻

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 230}}}%%
%% 图：一次 LoRA SFT 里各库出场的时刻
flowchart LR
    subgraph PRE["训练前，只做一次"]
        direction TB
        A1["load_dataset → .arrow"] --> A2["from_pretrained"] --> A3["get_peft_model<br/>168 处 setattr"] --> A4["_prepare_dataset<br/>map(build_labels)"]
    end
    subgraph STEP["每个训练步"]
        direction TB
        B1["__getitem__ → collator<br/>pad，labels 补 -100"] --> B2["forward：lora.Linear"] --> B3["ForCausalLMLoss"] --> B4["反向，只更新 lora_A / B"]
    end
    subgraph POST["训练后"]
        direction TB
        C1["save_pretrained<br/>只存 adapter"] --> C2["generate 或<br/>merge_and_unload"]
    end
    PRE --> STEP --> POST
```

---

## 常见误区

- 「`architectures` 决定实例化哪个类」——Auto 类只看 `model_type` 查表
- 「加载峰值是权重的两倍」——meta 骨架 + mmap，峰值约等于权重
- 「传了 `temperature` 就在采样」——看 `do_sample`，Qwen2.5 默认 False
- 「`len(tok) == config.vocab_size`」——151665 ≠ 151936，后者是对齐补大
- 「datasets 有缓存，`map` 不会重跑」——缓存键是 fingerprint，函数一改就失效
- 「LoRA 显存约等于 1.8% 参数」——激活与反向的 $$gW$$ 没省，省的是优化器状态
- 「GRPO 一组全对是最好的样本」——优势全零，梯度零贡献
{: .fragments}

---

## 下一步

- **往前**：模型侧对应《Transformer 原理与实现》，LoRA 对应《算法工程师的数学》，三种 loss 对应《后训练》
- **推理侧**：《大模型推理系统揭秘》读 continuous batching 与 PagedAttention 怎么换掉静态 batch 与每层 `cat`
- **训练侧**：verl 源码导读读 rollout 与 actor 分到不同 worker 后的同一条 GRPO loss
- 方法不变：找到注册表、找到 `compute_loss`、找到 `forward` 里那一行调用

<aside class="notes" markdown="1">
系列总结与通关自测：/hf-source-reading-series-recap-and-self-test.html（A 组 10 题、B 组 5 题、C 组 6 道面试题）。总纲：/reading-hugging-face-source-code.html。
</aside>
