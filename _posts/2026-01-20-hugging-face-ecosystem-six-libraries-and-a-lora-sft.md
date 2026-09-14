---
layout: post
series: algorithm-tooling
title: "算法工程师的工具箱（04）：Hugging Face 生态——六个库与一次 LoRA SFT 的组装"
subtitle: "The Hugging Face Ecosystem: Six Libraries, a Six-Line LoRA SFT, and Why Reading the Source Is the Fastest Way to Learn"
tags: [AI, LLM, PyTorch, Python]
catalog: true
updated: 2026-09-14
---

前两篇的训练循环训的是自己写的小模型。真实工作里模型不是自己写的——是从 Hugging Face Hub 下载的 Llama、Qwen、DeepSeek；数据也不是随机切窗口——是 Hub 上的数据集经过 chat template、loss mask、packing；训练器也不是二十行——是 `trl` 的 `SFTTrainer` / `DPOTrainer` / `GRPOTrainer`。Hugging Face 的六个库把这条路铺好了，六行代码能组装一次 LoRA SFT。这一篇讲六个库各管什么、六行背后发生了什么、以及一个比任何教程都重要的习惯：**卡住的时候直接读源码**。

全篇的核心问题是：

> **能不能用 `peft` + `trl` 在一小时内跑起一个 LoRA SFT？卡住的时候能不能直接读源码找到原因？**


## 一、总览

### 1. 六个库

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 220}}}%%
flowchart LR
    HUB["`**Hub**
config.json · safetensors
tokenizer.json · 数据集`"]
    DS["`**datasets**
load · map · filter
streaming · Arrow`"]
    TOK["`**tokenizers**
BPE 训练与编码
chat template 应用`"]
    TF["`**transformers**
AutoModel · AutoTokenizer
generate · Trainer`"]
    PEFT["`**peft**
LoraConfig · get_peft_model
merge_and_unload`"]
    TRL["`**trl**
SFTTrainer · DPOTrainer
GRPOTrainer · RewardTrainer`"]
    ACC["`**accelerate**
launch · 设备放置
DDP / FSDP / DeepSpeed 配置`"]

    HUB --> DS & TF
    DS --> TOK --> TRL
    TF --> PEFT --> TRL
    TRL --> ACC

    classDef lib fill:#eef4fb,stroke:#5b8dc9,stroke-width:1px,color:#222
    classDef hub fill:#fff7e0,stroke:#c98a00,stroke-width:2px,color:#222
    class DS,TOK,TF,PEFT,TRL,ACC lib
    class HUB hub
```

### 2. 本文的章节安排

| 章 | 主题 | 内容 |
|---|---|---|
| 二 | Hub 上的三个文件 | `config.json`、`tokenizer.json`、`*.safetensors`；从 config 算参数量 |
| 三 | 六个库各管什么 | `transformers`、`datasets`、`tokenizers`、`peft`、`trl`、`accelerate` |
| 四 | 六行组装一次 LoRA SFT | 代码；背后的每件事在第二篇二十行里的位置 |
| 五 | 在 0.5B 模型上跑通 | chat template、loss mask 比例、LoRA 参数量、20 步的 loss |
| 六 | 为什么读源码是最快的路 | 六个入口与它们的长度；从 `compute_loss` 往下追 |
| 七 | 本文小结 | |
| 八 | 自测 | 五道题 |

配套脚本：[`04_hf_lora_sft.py`](https://github.com/arganzheng/ai-learning-labs/blob/main/algorithm-tooling/04_hf_lora_sft.py)（需要下载 Qwen2.5-0.5B，约 1 GB）。


## 二、Hub 上的三个文件

拿到一个模型的 Hub 页面，先看三个文件：

### 1. `config.json`：结构超参数

```json
{"hidden_size": 896, "num_hidden_layers": 24, "num_attention_heads": 14, "num_key_value_heads": 2,
 "intermediate_size": 4864, "vocab_size": 151936, "tie_word_embeddings": true, ...}
```

这是 Qwen2.5-0.5B 的。用 L0 第一篇的表就能算出参数量：每层 attention 四个矩阵 + MLP 三个矩阵，乘层数，加词嵌入。`tie_word_embeddings: true` 说明输出层与词嵌入共享一份权重（小模型常这样做，省 $$V \times d = 136$$M 参数）。脚本加载后数出 494M 参数，与名字里的 "0.5B" 对上。L4《Transformer 与 LLM》第一篇专门教从 `config.json` 算参数量。

### 2. `tokenizer.json` 与 `tokenizer_config.json`

词表、合并规则、特殊 token（`<|im_start|>`、`<|im_end|>`、`<|endoftext|>`）、以及 **chat template**——一段 Jinja 模板，规定"一轮对话怎么拼成一个字符串"。第五章会看到它的输出。L4 预训练系列的第一篇讲 tokenizer 本身。

### 3. `*.safetensors`：权重

`state_dict`（第二篇）的磁盘格式：参数名 → 张量，按名字分片成几个文件，带一个 `index.json` 索引。`safetensors` 格式的好处是**不用加载全部就能读某一层**（内存映射），且不像 `pickle` 那样能执行任意代码。

模型卡（README）里的评测数字，读的时候带着 L0 第八篇的置信区间。


## 三、六个库各管什么

| 库 | 负责 | 要会的 |
|---|---|---|
| `transformers` | 模型定义与加载（`modeling_llama.py` 一类）、tokenizer 封装、`generate`、`Trainer` | `AutoModelForCausalLM.from_pretrained(..., dtype=torch.bfloat16)`；`tokenizer.apply_chat_template`；`generate` 的采样参数；读 `modeling_*.py` |
| `datasets` | 数据加载与处理，底层 Apache Arrow（内存映射、零拷贝） | `load_dataset`、`map(batched=True, num_proc=...)`、`filter`、`streaming=True` 处理放不进内存的语料 |
| `tokenizers` | 分词器的训练与快速编码（Rust 实现） | 训练一个 BPE 词表；理解 `tokenizer.json` 里的 normalizer / pre-tokenizer / model / post-processor 四段 |
| `peft` | 参数高效微调 | `LoraConfig(r, lora_alpha, target_modules, dropout)`、`get_peft_model`、训练后 `merge_and_unload` 合回基座 |
| `trl` | 后训练的各个 Trainer | `SFTTrainer`（自动处理 chat template、packing、loss mask）、`DPOTrainer`、`GRPOTrainer`、`RewardTrainer` |
| `accelerate` | 把单卡脚本变多卡，统一 DDP / FSDP / DeepSpeed 的启动 | `accelerate config` 生成配置；`accelerate launch train.py` |

它们的分工对应第二篇的五个对象：`transformers` 给 `nn.Module`（模型）与 tokenizer，`datasets` 给 `Dataset`，`peft` 改 `nn.Module`（在线性层旁边挂 LoRA），`trl` 给训练循环，`accelerate` 给第三篇的多卡启动。

### `generate` 的采样参数

`model.generate(..., do_sample=True, temperature=0.7, top_p=0.9, max_new_tokens=256)` 里的每个参数都是 L0 第五篇第七章的东西：温度、top-p、greedy（`do_sample=False`）。实现上每个都是一个 `LogitsProcessor`——对 logits 做一次变换再采样——第六章给源码入口。


## 四、六行组装一次 LoRA SFT

### 1. 代码

```python
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", lora_dropout=0.05))
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
trainer = SFTTrainer(model=model, train_dataset=ds, processing_class=tok, args=SFTConfig(...))
trainer.train()
```

### 2. 背后发生的事

六行背后每一件事都在第二篇的二十行里有对应位置：

| 发生的事 | 谁做的 | 对应二十行里的 |
|---|---|---|
| 数据被套上 chat template：每一轮用 `im_start` / `im_end` 一类特殊 token 包起来（第五章有实例） | `SFTTrainer` 调 `tok.apply_chat_template` | `Dataset` 的 `__getitem__` |
| 回复之外的 token 的 label 被置成 −100 | `SFTTrainer`（`completion_only_loss`） | `ignore_index=-100`——**SFT 的 loss mask** |
| 多条短样本被 pack 进一个序列（可选） | `SFTConfig(packing=True)` | `collate_fn` |
| LoRA 的 $$A$$、$$B$$ 被挂到每个线性层旁边，基座冻结 | `get_peft_model` | `nn.Module` 的改造；`requires_grad` |
| AdamW 只更新 $$A$$、$$B$$ | `Trainer` 用 `model.parameters()` 里 `requires_grad=True` 的 | `opt = AdamW(model.parameters())` |
| bf16、梯度裁剪、学习率调度、日志、checkpoint | `SFTConfig` 的字段 | `autocast`、`clip_grad_norm_`、`sched`、`log` |

`LoraConfig` 的四个参数：`r` 是秩（L0 第三篇）；`lora_alpha` 是缩放，实际加到输出上的是 $$\frac{\alpha}{r} BA x$$，常取 $$\alpha = 2r$$；`target_modules="all-linear"` 把七个线性层都挂上（也可以只挂 `q_proj, v_proj`）；`lora_dropout` 是 LoRA 分支上的 dropout。


## 五、在 0.5B 模型上跑通

脚本用 Qwen2.5-0.5B 与 12 条写死的问答，CPU 上训 20 步，把六行背后的每件事打印出来。

### 1. 模型与 chat template

```text
config.json: hidden 896, layers 24, heads 14/2 kv, intermediate 4864, vocab 151936, tie_embeddings True
参数量 494 M; tokenizer 词表 151665; chat template 有

一条样本经 chat template：
'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\nParis.<|im_end|>\n'
```

Qwen 的模板自动加了一段默认 system prompt；每一轮用 `<|im_start|>角色\n内容<|im_end|>\n` 包起来。**模型学到的"对话格式"就是这几个特殊 token 的排列**，推理时必须用同一个模板，否则模型不知道该在哪开始回答。

### 2. LoRA 挂到哪、多少参数

```text
可训练 8.80 M / 494 M = 1.78%
训练状态 ≈ 可训练 × 16 B = 141 MB；冻结权重 fp32 1.98 GB（bf16 时减半）
挂了 LoRA 的线性层: ['down_proj', 'gate_proj', 'k_proj', 'o_proj', 'q_proj', 'up_proj', 'v_proj']
```

七个线性层——与 L0 第三篇表里的七个一一对应。0.5B 模型上 $$r = 16$$ 是 1.78%（比 8B 的 0.52% 高，因为小模型 $$d$$ 小、$$r(m + n) / mn$$ 更大）。第三篇的账：训练状态只有 141 MB，冻结权重 2 GB（fp32）——这个模型在 CPU 上都能微调。

### 3. loss mask 的比例

```text
一个 batch: input_ids (4, 36), labels 里被 mask 成 -100 的 token 122/144 (85%，prompt 与 padding 不算 loss)
```

4 条样本 padding 到 36 长，144 个位置里 122 个是 −100：system prompt、user 的问题、padding 都不算 loss，**只有 assistant 的那几个 token（`Paris.<|im_end|>`）进入交叉熵**。这就是 L0 第五篇第四章"SFT 只对回答部分求和"的实物。85% 被 mask 掉意味着有效 token 很少——真实 SFT 数据的回复要长得多，比例会反过来。

### 4. 20 步

```text
loss: 第 1 步 5.254 → 最后 1.658  (15 s, 0.8 s/步)
Q: What is the capital of France?   A: 'Paris.看查看\npositories\nThe capital of the United States'
Q: What is the capital of Italy?    A: 'Rome.看查看\nRowAtIndexPath\n Florence.看查看\nRowAtIndexPath'
```

loss 从 5.3 降到 1.7；生成时**答案学会了**（Paris、Rome——后者不在训练数据里，是基座本来就会的），但**没学会在 `<|im_end|>` 停下**，接着吐出乱码。12 条数据、20 步，模型见到 `<|im_end|>` 这个 token 的次数太少。这是一个真实的 SFT 现象：**结束符必须进 loss、且要见够多次**，否则模型不会停。L5 后训练系列第一篇专门讲 SFT 数据的这些细节；这里它是一个"六行跑通了，但要看懂输出"的例子。

### 5. 合并

```text
merge_and_unload 后参数量 494 M（LoRA 已合回基座，推理零开销）
```

$$W' = W + \frac{\alpha}{r} BA$$，L0 第三篇的"合并"用法。


## 六、为什么读源码是最快的路

Hugging Face 的库是当前算法工作的事实标准，也是**最好的教材**——比论文更准确（论文写的是想法，代码写的是实际做法），比教程更完整。几个值得直接读的入口：

| 想学 | 读 | 大约多长 |
|---|---|---|
| Llama 的结构 | `transformers/models/llama/modeling_llama.py`：`LlamaAttention`、`LlamaMLP`、`LlamaDecoderLayer`、`apply_rotary_pos_emb` | 核心几百行；每个类都是第二篇的 `nn.Module` |
| DPO 的 loss 到底怎么算 | `trl/trainer/dpo_trainer.py` 里 `dpo_loss`：把 L0 第六篇推出的公式变成十几行代码，还能看到 IPO、hinge 等变体各改了哪一行 | 几十行 |
| GRPO 的优势怎么算、KL 怎么加 | `trl/trainer/grpo_trainer.py`：L0 第七篇的 $$(R - \text{mean}) / \text{std}$$ 与裁剪 | 几百行 |
| LoRA 怎么挂上去 | `peft/tuners/lora/layer.py`：`Linear.forward` 里 `result += lora_B(lora_A(dropout(x))) * scaling` | 一行核心——L0 第三篇的 $$BAx$$ |
| SFT 的 loss mask 与 packing | `trl/trainer/sft_trainer.py` 与它的 data collator | 几百行 |
| `generate` 的采样 | `transformers/generation/utils.py` 与 `logits_process.py`：temperature、top-k、top-p 各是一个 `LogitsProcessor` | 每个 processor 十几行——L0 第五篇第七章 |

方法很简单：**遇到一个后训练概念，先读它在 `trl` 里的实现，再读论文。** 库的版本变化快，函数名会变（本文写作时的接口未必与你读到时一致），但找到入口的方法不变——从 Trainer 的 `compute_loss` 往下追，或者在编辑器里对着一个 API 名按"跳转到定义"。读到一个看不懂的公式，回 L0 对应的篇；读到一个看不懂的形状操作，回本系列第一篇。


## 七、本文小结

- **Hub 上的三个文件**：`config.json`（结构超参数，能算出参数量）、`tokenizer.json`（词表、特殊 token、chat template）、`*.safetensors`（`state_dict` 的磁盘格式，可部分加载、不能执行代码）。
- **六个库**各管一段：`transformers` 给模型与 tokenizer、`datasets` 给数据（Arrow）、`tokenizers` 训与编码词表、`peft` 挂 LoRA、`trl` 给后训练的 Trainer、`accelerate` 给多卡启动；对应第二篇的五个对象。
- **六行组装 LoRA SFT**，背后的每件事——chat template、loss mask（−100）、packing、LoRA 挂载、只更新 $$A, B$$、bf16 / 裁剪 / 调度——都在二十行训练循环里有位置。
- 0.5B 上跑通：七个线性层挂 LoRA、可训练 1.78%、训练状态 141 MB；一个 batch 85% 的 token 被 mask；20 步 loss 5.3 → 1.7，答案学会了但没学会停——**结束符要进 loss 且见够多次**。
- **读源码是最快的路**：`modeling_llama.py`、`dpo_loss`、`grpo_trainer.py`、`peft` 的 `Linear.forward`、`LogitsProcessor`；从 `compute_loss` 往下追。库的接口会变，方法不变。

<details markdown="1">
<summary><b>核心问题的答案</b></summary>

**能跑起来**：六个库各管一段——`transformers` 模型与 tokenizer、`datasets` 数据、`peft` 挂 LoRA、`trl` 的 `SFTTrainer`、`accelerate` 多卡、`tokenizers` 词表——六行组装，0.5 B 上 20 步从 loss 5.3 到 1.7（第四、五章）；背后的每件事（chat template、loss mask、packing、只更新 $$A, B$$）都在上一篇的二十行循环里有位置。**卡住能读源码**：从 `Trainer.compute_loss` 往下追——模型结构在 `modeling_llama.py`，LoRA 的前向在 `peft` 的 `Linear.forward`，loss 在 `trl` 各 Trainer 的 `compute_loss` / `dpo_loss`，采样在 `LogitsProcessor`（第六章）。实测的一个教训：答案学会了但没学会停，是因为结束符没进 loss——这类问题只有读源码才能定位。

</details>


## 八、自测

1. 一个模型的 `config.json` 里 `hidden_size: 4096, intermediate_size: 14336, num_hidden_layers: 32, num_attention_heads: 32, num_key_value_heads: 8, vocab_size: 128256, tie_word_embeddings: false`——参数量大约多少？

   <details markdown="1"><summary>答案</summary>

   就是 Llama-3-8B，8.03B（L0 第一篇的表）。

   </details>

2. 为什么推理时必须用与训练相同的 chat template？

   <details markdown="1"><summary>答案</summary>

   模型学到的"该在哪开始回答"编码在特殊 token 的排列里，换模板它不知道边界。

   </details>

3. `LoraConfig(r=16, lora_alpha=32)` 里 `lora_alpha` 在做什么？改成 16 有什么效果？

   <details markdown="1"><summary>答案</summary>

   LoRA 输出的缩放 $$\alpha / r$$：32 / 16 = 2 倍；改成 16 就是 1 倍，等价于把学习率对 LoRA 的作用减半。

   </details>

4. 一个 batch 的 labels 里 85% 是 −100，说明什么？真实 SFT 数据会怎样？

   <details markdown="1"><summary>答案</summary>

   回复很短、prompt 与 padding 占大头，有效训练 token 少；真实数据回复长，比例反过来。

   </details>

5. 想知道 `SFTTrainer` 到底怎么给 prompt 部分打 −100，去读哪个文件的哪一部分？

   <details markdown="1"><summary>答案</summary>

   `trl/trainer/sft_trainer.py` 的数据处理 / collator 部分，搜 `completion_only_loss` 或 `-100`。

   </details>

下一篇是本系列最后一篇：GPU 的两个上限与四块显存（为什么 decode 快不起来、为什么 batch 大才快）、读 profiler、以及让三个月前的实验能复现的最小记录。
