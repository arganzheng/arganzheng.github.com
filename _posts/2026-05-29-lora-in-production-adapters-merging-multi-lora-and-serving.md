---
layout: post
series: lora
title: "LoRA 专题（03）：工程：adapter 文件、合并、多 LoRA 服务与参考模型"
subtitle: "LoRA 03: In Production — Adapter Files, Merging, Multi-LoRA Serving and the Free Reference Model"
tags: [AI, LLM, LoRA, Post-Training, peft, vLLM]
catalog: true
---

> **更新 @2026-09-30**：实验用 peft 0.21.1、transformers 5.17.0、bitsandbytes 0.50.2、PyTorch 2.14（CPU），模型 Qwen2.5-0.5B，adapter 是[第二篇](/lora-hyperparameters-rank-targets-alpha-lr-and-variants.html)训出的 `r16_all`（$$r = 16$$ 全部线性层）与 `r16_attn`（$$r = 16$$ 只挂 attention）。配套脚本 `ai-learning-labs/lora/03_deploy.py`（子实验 `files` / `merge` / `quant` / `multi` / `tokens`）。

前两篇回答了"为什么"与"怎么配"，训出来的东西是一个目录：`adapter_config.json` 加一个几十 MB 的 `adapter_model.safetensors`。本文回答从这个目录到线上服务之间的每一个问题：文件里有什么、装回去与训练结束时是否一致、合并前后差多少、换一个量化过的底座差多少、两个 adapter 能不能同时挂或加起来、一个底座服务几十个客户的显存怎么算、为什么 DPO 用了 LoRA 就不用再放参考模型、为什么新加的特殊 token 学不会。每个问题都有一个实验和一组数字。

## 一、adapter 文件里有什么

### 1.1 两个文件

`model.save_pretrained("r16_all")` 存下的是：

```text title="02 存下来的 adapter 目录"
r16_all/
  README.md                       5 KB     # peft 自动写的模型卡
  adapter_config.json             1 KB     # LoraConfig 的 JSON：r、alpha、target_modules、底座名……
  adapter_model.safetensors    33.6 MB     # 336 个张量 = 24 层 × 7 个矩阵 × (A, B)，FP32
```

`adapter_model.safetensors` 的第一个键是 `base_model.model.model.layers.0.mlp.down_proj.lora_A.weight`，形状 `(16, 4864)`——键名里带着它要挂到底座哪个模块上。对照：底座 494M 参数 FP32 存 1.84 GiB、BF16 存 0.92 GiB；只挂 attention 的 `r16_attn` 是 8.27 MB；8B 底座 BF16 16 GB，r=16 全部线性层的 adapter 42M × 2 B = 84 MB。

`adapter_config.json` 里决定加载行为的字段：

| 字段 | 例子 | 加载时做什么 |
|---|---|---|
| `base_model_name_or_path` | `Qwen/Qwen2.5-0.5B` | 只是记录，**不校验**；`PeftModel.from_pretrained(base, path)` 里的 `base` 是你传的 |
| `peft_type` | `LORA` | 决定用哪个 tuner 类 |
| `r`、`lora_alpha`、`use_rslora` | 16、32、false | 决定 $$A$$、$$B$$ 的形状与 `scaling`（$$\alpha / r = 2$$，rsLoRA 时 $$\alpha / \sqrt r$$） |
| `target_modules` | `["q_proj", ..., "down_proj"]` | 决定往哪些模块里注入；`all-linear` 在保存时已展开成列表 |
| `use_dora`、`init_lora_weights` | false、true | DoRA 多一个幅度向量要加载；PiSSA / OLoRA 的 adapter 要配套改过的底座（见 1.3） |
| `modules_to_save`、`trainable_token_indices` | null | 若非空，safetensors 里还有整份的 embedding / lm_head 或若干行 |
| `peft_version` | `0.21.1` | 记录，跨大版本加载时 `peft` 据此做兼容转换 |

Table: adapter_config.json 里影响加载的字段

`adapter_model.safetensors` 里是 $$24 \times 7 \times 2 = 336$$ 个张量，键名形如 `base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight`，形状 $$16 \times 896$$（$$A$$）或 $$896 \times 16$$（$$B$$）。**没有 $$W$$**。所以：

- adapter 只有底座的几十分之一大：0.5B 上 17 MB 对 0.92 GiB（都按 BF16 算；本实验用 FP32 训，存下来是 33.6 MB），8B 上 84 MB 对 16 GB。这是 multi-LoRA 服务、按任务分发 checkpoint、A/B 测试几十个版本的全部经济基础。
- adapter **必须与训练时的底座配对**。它对着的是那个底座的 $$W$$ 学的 $$\Delta W$$；换一个 $$W$$（另一个版本、另一个精度、另一个模型），$$W + \Delta W$$ 就不再是训练时的模型。`base_model_name_or_path` 不会替你校验，形状对得上就能加载。

### 1.2 什么算"同一个底座"

| 换成 | 形状 | 效果 | 说明 |
|---|---|---|---|
| 同一权重、BF16 ↔ FP32 | 一样 | 差别是精度舍入 | 可以 |
| 同一权重、NF4 量化 | 一样 | 差别是量化误差（第三节实测） | 通常可以，精度掉一点 |
| 同系列不同版本（Qwen2.5-0.5B → 0.5B-Instruct） | 一样 | **不可预测**：adapter 是对 base 的 $$W$$ 学的方向 | 能加载，效果要重测 |
| 同架构不同尺寸（0.5B → 1.5B） | 不一样 | 加载报错 | 不行 |
| 同尺寸不同模型（Qwen → Llama） | 可能一样 | 加载可能不报错但毫无意义 | 不行 |

Table: adapter 装到不同底座上会发生什么

工程上的做法是把底座的 revision（commit hash）写进 adapter 的元数据或训练日志，服务端加载时校验。

### 1.3 PiSSA / OLoRA 的 adapter 是特例

PiSSA、OLoRA 这类初始化会**改底座**：$$W = W_{res} + B_0 A_0$$，训练的是 $$W_{res}$$ 上的 $$B$$、$$A$$。直接存下的 adapter 对着的是 $$W_{res}$$，装到原始 $$W$$ 上就错了。`peft` 的解法是 `save_pretrained(..., path_initial_model_for_weight_conversion=...)`：把训练后的 $$BA$$ 与初始的 $$B_0 A_0$$ 相减，转换成一个对着原始 $$W$$ 的普通 LoRA（秩变成 $$2r$$）。用了这些初始化又要发 adapter，这一步不能省。

## 二、加载与合并：数值上等价到什么程度

### 2.1 加载

```python title="把 adapter 装回底座"
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.float32)
model = PeftModel.from_pretrained(base, "out/ckpt/r16_all")
```

装回去后在 100 条验证样本上的回复 loss 与第二篇训完当场算出的数一致（都是 2.3943）。这是应该的：adapter 存的就是那 336 个张量，前向公式没变。

### 2.2 合并

合并就是把 $$\frac{\alpha}{r} B A$$ 加进 $$W$$：

```python title="merge_and_unload：把 ΔW 加进 W"
# 每个 lora.Linear：base.weight += scaling * B @ A，再换回 nn.Linear
merged = model.merge_and_unload()
```

`peft` 里 `get_delta_weight` 算 `weight_B @ weight_A * scaling`，`merge` 做 `base_layer.weight.data += delta_weight`，`unmerge` 减回去（[HF 源码第四篇](/peft-and-trl-lora-sft-dpo-grpo-in-source.html)第四节有这几个函数）。合并后模型里没有任何 `lora` 模块，参数量回到 494M，结构与原始 `Qwen2ForCausalLM` 完全相同——这就是它可以直接交给 vLLM、llama.cpp、TensorRT-LLM 的原因。

```text title="03_deploy.py merge 的输出"
加载后验证回复 loss 2.3943；与 base 的 logits 最大差 18.816（adapter 确实生效了）
merge_and_unload 后：模型里 lora 模块 0 个，参数量 494.0M（与底座相同）
合并前后 logits 最大差 3.28e-04；一次前向 65 ms → 53 ms
合并后的模型存到 out/ckpt/merged_r16_all（1.85 GiB，与底座一样大）
```

三个数字各说明一件事：

- **合并前后 logits 的最大差 $$3 \times 10^{-4}$$**（logits 本身量级 10，相对 $$10^{-5}$$；adapter 让 logits 变了 18.8），是 FP32 加法舍入穿过 24 层累积的结果，不是行为差异。BF16 下合并要小心：$$W$$ 是 BF16、$$\Delta W$$ 比 $$W$$ 小两个数量级，`W + ΔW` 在 BF16 里会丢掉 $$\Delta W$$ 的大部分有效位。做法是把 $$W$$ 升到 FP32 加完再转回 BF16（`peft` 对 BF16 权重会这样做），或者干脆存 FP32 / BF16 两份。
- **前向时间**：合并后少了 336 个瘦 GEMM 与 168 次加法，单条请求在 CPU 上快 18%（65 → 53 ms）。GPU 上小 batch 时差距类似（kernel launch 主导），大 batch 时差距变小。
- **合并后的模型与底座一样大**：一个 17 MB 的 adapter 换成一个 0.92 GiB 的模型（BF16）。要服务几十个 adapter 时这就是问题，第五节算这笔账。

`merge_adapter()` 与 `merge_and_unload()` 的区别：前者只把 $$\Delta W$$ 加进 $$W$$ 但保留 `lora.Linear` 结构（可以 `unmerge_adapter()` 减回去，用于训练中评估或多 adapter 切换），后者把结构也换回去，不可逆。

## 三、量化底座与 adapter 的失配

adapter 是对着某个精度的 $$W$$ 学的。工程上常见的组合有四种：

| 训练时 | 服务时 | 差别来自 |
|---|---|---|
| FP32 / BF16 底座 + adapter | 同精度底座 + adapter | 无（数值一致） |
| FP32 / BF16 底座 + adapter | 先合并，再 INT4 / NF4 量化 | 一次量化误差落在 $$W + \Delta W$$ 上 |
| FP32 / BF16 底座 + adapter | NF4 底座 + 同一个 adapter | 底座的量化误差；adapter 没变 |
| NF4 底座 + adapter（QLoRA） | BF16 底座 + 同一个 adapter | adapter 学的是补 NF4 底座的 $$\Delta W$$，装到 BF16 上"补过头" |

Table: 训练与服务时底座精度的四种组合

实测（第二篇的 r=16 全部线性层 adapter，FP32 上训的）：

| 组合 | 验证回复 loss | 相对训练前同精度底座 |
|---|---|---|
| FP32 底座（训练前） | 2.4936 | — |
| NF4 底座（训练前） | 2.5780 | 量化误差 +0.0844 |
| FP32 底座 + adapter（训练时的组合） | 2.3943 | −0.0993 |
| NF4 底座 + 同一个 adapter | 2.4860 | −0.0920 |
| 先合并再 NF4 量化 | 2.4854 | −0.0926 |
| 对照：QLoRA 在 NF4 底座上训出的 adapter + NF4 底座（第二篇） | 2.4663 | −0.1117 |

Table: 同一个 adapter 换底座精度、以及先合并再量化的效果（`03_deploy.py quant`；NF4 = bitsandbytes `load_in_4bit`、双重量化，CPU）

看什么：**换底座精度带来的差别与量化误差同量级，adapter 并没有失效**。训练前 FP32 → NF4 差 0.084，训练后两种组合差 0.092；adapter 在 FP32 上带来 −0.099、装到 NF4 上仍带来 −0.092。先合并再量化与量化底座挂 adapter 几乎一样（2.4854 / 2.4860）——0.5B 上 $$\Delta W$$ 比 NF4 的量化步长小得多，加进去再量化基本被四舍五入掉了，于是与不合并同解。最后一行说明配对训练的价值：对着 NF4 底座训的 QLoRA adapter 在 NF4 底座上是 2.4663，比 FP32 上训的搬过来好 0.02——它学会了补一部分量化误差。QLoRA 训的 adapter 装到 BF16 底座上是同一件事反过来——差别不大但确实存在，而且方向不确定，上线前要在验证集上重测一次，不能默认等价。`peft` 对 bitsandbytes 底座的 `merge_and_unload` 会先反量化再加 $$\Delta W$$ 再重新量化，多一次量化误差；要最干净的合并结果，用 BF16 底座加载 adapter 再合并，然后按服务需要量化。

## 四、多个 adapter：切换、合成、参考模型

### 4.1 一个底座挂两个 adapter

```python title="一个底座挂两个 adapter 并切换"
model = PeftModel.from_pretrained(base, "r16_all", adapter_name="all")
model.load_adapter("r16_attn", adapter_name="attn")
model.set_adapter("attn")                 # 前向只走 attn 的 B、A
```

`lora.Linear.forward` 里的循环是 `for active_adapter in self.active_adapters`：每个 `lora.Linear` 是一个 `ModuleDict`，键是 adapter 名，`set_adapter` 只改 `active_adapters` 这个列表。两个 adapter 的参数同时在内存里（0.5B 上 8.8M + 2.2M），切换是零拷贝的。

```text title="set_adapter 切换后的验证 loss"
已加载 adapter：['all', 'attn']，当前激活：all
set_adapter('all' )  验证回复 loss 2.3943     # 第二篇的 r=16 全部线性层
set_adapter('attn')  验证回复 loss 2.4113     # 第二篇的 r=16 attention-only
```

两个数与第二篇训完当场算的一致：切换 adapter 就是切换到那次训练的模型。

### 4.2 加权合成

`add_weighted_adapter` 把几个 adapter 合成一个新的：

| `combination_type` | 做法 | 新 adapter 的秩 | 何时用 |
|---|---|---|---|
| `linear` | $$B = \sum w_i B_i$$、$$A = \sum w_i A_i$$ | $$r$$（要求各 adapter 同 $$r$$） | 同一任务的几个 checkpoint 平均；**不是** $$\sum w_i B_i A_i$$，有交叉项 |
| `cat` | $$B = [B_1, B_2]$$、$$A = [A_1; A_2]$$ 拼接 | $$\sum r_i$$ | 精确等于 $$\sum B_i A_i$$（$$w_i = 1$$ 时），秩变大 |
| `svd` | 先算 $$\sum w_i B_i A_i$$，再 SVD 截到指定秩 | 指定 | 想精确合成又不想秩变大；要算一次全尺寸 $$\Delta W$$ |
| `ties` / `dare_*` / `magnitude_prune` | 先稀疏化或裁剪再合成 | 指定 | 多任务合并时减少冲突（Yadav 等 2023、Yu 等 2024） |

Table: add_weighted_adapter 的合成方式

把上面两个 adapter（`all` r=16 与 `attn` r=16，同一任务的两种配置）各按 0.5 合成：

| 合成方式 | 新 adapter 的秩 | 验证回复 loss |
|---|---|---|
| `all` 单独 | 16 | 2.3943 |
| `attn` 单独 | 16 | 2.4113 |
| `linear` 0.5 / 0.5 | 16 | 2.3992 |
| `cat` | 32 | 2.4065 |
| `svd` 截到秩 16 | 16 | 2.4063 |

Table: 两个同任务 adapter 的三种合成（`03_deploy.py multi`）

三种合成都落在两个原 adapter 之间，没有一种好于最好的那个——同一任务的 adapter 合成拿到的是"平均"，不是"叠加"。`cat` 与 `svd` 几乎一样（2.4065 / 2.4063）：`svd` 截到 16 几乎没丢东西，说明 $$B_1 A_1 + B_2 A_2$$ 这个秩 32 的矩阵有效秩其实不到 16——又一次印证第一篇的低秩假设。

`linear` 与 `cat` 的差别值得记住：两个 adapter 的 $$\Delta W$$ 之和是 $$B_1 A_1 + B_2 A_2$$，`cat` 精确等于它；`linear` 算的是 $$(w_1 B_1 + w_2 B_2)(w_1 A_1 + w_2 A_2)$$，多了 $$B_1 A_2$$、$$B_2 A_1$$ 两个交叉项——两个 adapter 是各自独立训的，$$B_1$$ 与 $$A_2$$ 之间没有任何关系，交叉项是噪声。所以合成不同任务的 adapter 用 `cat` 或 `svd`，`linear` 只适合同一任务不同 checkpoint（它们的 $$A$$、$$B$$ 从同一个初始化出发，方向相近）。

### 4.3 `disable_adapter()`：DPO / GRPO 的免费参考模型

DPO 的 loss 里有 $$\log \pi_\theta / \pi_{ref}$$，GRPO 的 KL 项也要 $$\pi_{ref}$$，$$\pi_{ref}$$ 是训练开始时的策略。全量训练要在显存里再放一份模型；LoRA 下训练开始时的策略就是**底座本身**（$$B = 0$$，$$W' = W$$），所以：

```python title="关闭 adapter 取参考模型的 logits"
with model.disable_adapter():
    ref_logits = model(**batch).logits      # 走的是 W，不是 W + BA
```

```text title="disable_adapter 的输出"
with disable_adapter(): 验证回复 loss 2.4936（base 是 2.4936）
```

关掉 adapter 的模型与从没训过的底座逐位一样——这就是免费的参考模型。

`trl` 的 `DPOTrainer` / `GRPOTrainer` 在模型是 `PeftModel` 且没传 `ref_model` 时就是这么做的（`self.ref_model = None`，参考前向放在关闭 adapter 的上下文里：`use_adapter(model, adapter_name=None)`，旧版叫 `null_ref_context`），省一整份模型的显存与一份前向的权重读取。代价是参考模型的前向没有省——它仍是一次完整前向，只是不走 LoRA 分支。8B 模型上这一招省 16 GB。

## 五、multi-LoRA 服务的账

一个底座服务 $$N$$ 个客户，每个客户一个 adapter。两种方案：

| | $$N$$ 份合并模型 | 一个底座 + $$N$$ 个 adapter |
|---|---|---|
| 0.5B，$$N = 50$$ | $$50 \times 0.92$$ GiB = 46 GiB | 0.92 GiB + $$50 \times 17$$ MB = 1.75 GiB |
| 8B，$$N = 50$$ | $$50 \times 16$$ GB = 800 GB | 16 GB + $$50 \times 84$$ MB = 20 GB |
| 切换一个客户 | 换模型：加载 16 GB，秒到分钟 | 换 adapter 名：零拷贝；冷加载 84 MB，百毫秒 |
| 一个 batch 里混多个客户 | 不行，一个模型一个 batch | 可以：每个请求带 adapter id |
| 每 token 计算 | 一个 GEMM | 一个 GEMM + 按请求分组的瘦 GEMM |

Table: 合并模型与 multi-LoRA 两种服务方案

后一种方案的关键是最后一行：同一个 batch 里的请求各自走不同的 $$B_i A_i$$，不能写成一个普通的 GEMM。vLLM 的做法（Punica 的 SGMV / BGMV kernel）是把 batch 里的 token 按 adapter 分组，一个 kernel 里对每组做各自的瘦矩阵乘——一次 kernel 处理所有 adapter，而不是循环 $$N$$ 次。显存上 vLLM 用 `--max-loras` 与 `--max-lora-rank` 预留槽位：

$$
\text{槽位显存} = \texttt{max\_loras} \times \sum_{\text{挂 LoRA 的矩阵}} \texttt{max\_lora\_rank} \times (d_{in} + d_{out}) \times 2\ \text{字节}
$$

8B、全部线性层、`max_loras=8`、`max_lora_rank=64`：$$8 \times 167.8\text{M} \times 2 = 2.7$$ GB，从 KV cache 的预算里扣。$$N$$ 大于 `max_loras` 的 adapter 在 CPU 内存里排队换入换出。请求形状、kernel 与调度的细节在 [vLLM 系列第十一篇](/request-shapes-multi-lora-and-multimodal.html)，本文只算这笔账。

什么时候仍该合并：只服务一个 adapter；或延迟极敏感、batch 很小（多出的瘦 GEMM 占比最大）；或推理框架不支持 LoRA（大多数边缘部署格式）。

## 六、新加的 token 为什么学不会

一个常见的坑：给模型加一个 `<|tool_call|>` 之类的特殊 token，SFT 数据里用它，挂 LoRA 训完发现模型从不生成它、或生成了也乱。原因在公式里：LoRA 挂在**线性层**上，`embed_tokens` 与 `lm_head` 不是 `target_modules`，是冻结的。新 token 那一行 embedding 从初始化到训完一个字节都没动，输出侧 `lm_head` 对应那一行也没动。

实测：给 Qwen2.5-0.5B 加一个 `<|tool_call|>`（id 151665；Qwen 的 embedding 有 151936 行，预留了空位，不必 `resize_token_embeddings`），各配置训一步，看新 token 那一行 embedding 动了多少：

| 配置 | 可训练参数 | 一步后新 token embedding 的最大改动 |
|---|---|---|
| LoRA r=16 全部线性层 | 8.8M | 0（一个字节没动） |
| + `trainable_token_indices={"embed_tokens": [151665]}` | 8.8M + 896 | $$1.0 \times 10^{-3}$$（= lr，Adam 第一步的步长） |
| + `modules_to_save=["embed_tokens", "lm_head"]` | 281.1M | $$1.0 \times 10^{-3}$$ |

Table: 新 token 的 embedding 在三种配置下训一步的变化（`03_deploy.py tokens`）

三种修法与代价：

| 修法 | 训什么 | 0.5B 上多出的可训练参数 | 说明 |
|---|---|---|---|
| 什么都不做 | — | 0 | 新 token 学不会 |
| `trainable_token_indices={"embed_tokens": [id]}` | 只训那几行 | 每行 896 个 | `peft` 0.15+；`lm_head` 与 `embed_tokens` 绑定时一并处理 |
| `modules_to_save=["embed_tokens", "lm_head"]` | 整张表的副本 | 2 × 136M = 272M（Qwen 绑定权重，`peft` 仍各存一份副本并警告要设 `ensure_weight_tying=True`） | 比 LoRA 本身大 30 倍，adapter 从 17 MB 变成 560 MB |

Table: 让 LoRA 学会新 token 的三种做法

`modules_to_save` 的本质是"这些模块全量训"：`peft` 复制一份、训副本、保存时整份存进 adapter。词表大（Qwen 151936 × 896 = 136M）时这一份比全部 LoRA 参数大得多，能用 `trainable_token_indices` 就不用它。

## 七、上线清单

把本文的每个实验压成一行：

| 检查 | 怎么做 | 依据 |
|---|---|---|
| adapter 与底座配对 | 记录并校验底座 revision；换底座后重测验证集 | 第一节 |
| 加载后与训练结束一致 | 同一验证集算一次 loss，应与训练日志一致 | 2.1 |
| 合并 | FP32 里加、再转目标精度；合并后 logits 与合并前差 $$< 10^{-5}$$ | 2.2 |
| 换精度 | FP32/BF16 训、NF4 服务，或反之，验证集重测；差别应与底座量化误差同量级 | 第三节 |
| 多 adapter 合成 | 不同任务用 `cat` / `svd`，同任务 checkpoint 才用 `linear` | 4.2 |
| DPO / GRPO | `ref_model=None`，用 `disable_adapter()` | 4.3 |
| 服务方案 | 一个 adapter 合并；多个 adapter 走 multi-LoRA，按 `max_loras × max_lora_rank` 预留 | 第五节 |
| 新 token | 用了就必须 `trainable_token_indices` 或 `modules_to_save` | 第六节 |

Table: LoRA 上线前的检查项

## 八、本文小结

- adapter 只有 $$A$$、$$B$$（BF16 下 0.5B 17 MB、8B 84 MB；FP32 翻倍），**必须与训练时的底座配对**，`base_model_name_or_path` 不会替你校验。
- 合并是 $$W \leftarrow W + \frac{\alpha}{r} B A$$，前后 logits 差在浮点舍入量级；BF16 下要在 FP32 里加。合并后结构与原模型相同，可交给任何推理框架，但一个 adapter 变成一整个模型。
- 换底座精度（FP32 ↔ NF4、先合并再量化）带来的差别与量化误差同量级，adapter 不失效但要重测。
- 多 adapter：`set_adapter` 零拷贝切换；合成不同任务用 `cat` / `svd`，`linear` 有交叉项；`disable_adapter()` 就是 DPO / GRPO 的参考模型。
- multi-LoRA 服务：8B 上 50 个客户是 20 GB 对 800 GB；显存按 `max_loras × max_lora_rank` 预留，kernel 按 adapter 分组做瘦 GEMM。
- 新 token 学不会是因为 embedding 冻结；`trainable_token_indices` 几 KB 修好，`modules_to_save` 要 136M。

## 九、自测

1. 一个 8B 模型全部线性层 $$r = 32$$ 的 adapter 多大？BF16 底座多大？50 个客户各一个 adapter，两种服务方案各占多少存储？

   <details markdown="1"><summary>答案</summary>
   $$r = 32$$ 的参数是 $$r = 16$$ 的两倍，83.9M × 2 B = 168 MB；底座 16 GB。合并方案 $$50 \times 16 = 800$$ GB，multi-LoRA 方案 $$16 + 50 \times 0.168 = 24.4$$ GB。
   </details>

2. 用 BF16 底座合并一个 $$\lVert \Delta W \rVert / \lVert W \rVert \approx 10^{-3}$$ 的 adapter，直接 `W += ΔW`。会丢多少？

   <details markdown="1"><summary>答案</summary>
   BF16 有 8 位尾数（相对精度约 $$2^{-8} \approx 4 \times 10^{-3}$$）。$$\Delta W$$ 的元素比 $$W$$ 小三个数量级，小于 $$W$$ 的一个 ulp，加法结果四舍五入后大部分元素的 $$\Delta W$$ 直接丢失——合并后的模型接近没合并。要在 FP32 里加。
   </details>

3. 两个各自独立训练的 $$r = 16$$ adapter，用 `combination_type="linear", weights=[1, 1]` 合成。得到的 $$\Delta W$$ 与 $$B_1 A_1 + B_2 A_2$$ 差在哪？用什么方式能精确得到后者？

   <details markdown="1"><summary>答案</summary>
   `linear` 得到 $$(B_1 + B_2)(A_1 + A_2) = B_1 A_1 + B_2 A_2 + B_1 A_2 + B_2 A_1$$，多两个交叉项，且 $$B_1$$ 与 $$A_2$$ 毫无关系，交叉项是噪声。`cat` 精确等于 $$B_1 A_1 + B_2 A_2$$（秩 32）；`svd` 再截回秩 16 是近似。
   </details>

4. 用 LoRA 训 DPO，`DPOTrainer(ref_model=None, peft_config=...)`。参考模型的 logits 从哪来？省了什么、没省什么？

   <details markdown="1"><summary>答案</summary>
   `with model.disable_adapter()` 下再前向一次：走冻结的 $$W$$，等于训练开始时的策略。省了一整份模型的显存（8B 省 16 GB）与它的权重读取；没省参考模型那次前向的计算。
   </details>

5. 给 Qwen2.5-0.5B 加了 3 个特殊 token，只挂 $$r = 16$$ 全部线性层的 LoRA，训完模型不会用它们。为什么？用 `trainable_token_indices` 修，多训多少参数？用 `modules_to_save=["embed_tokens"]` 呢？

   <details markdown="1"><summary>答案</summary>
   `embed_tokens` 与 `lm_head` 不在 `target_modules` 里，冻结；新 token 的 embedding 行与输出行从没动过。`trainable_token_indices` 训 3 行：$$3 \times 896 = 2688$$ 个（绑定权重时输入输出共用）。`modules_to_save` 训整张 $$151936 \times 896 = 136$$M 的表，是 LoRA 本身 8.8M 的 15 倍。
   </details>

6. 服务端 vLLM 设 `--max-loras 4 --max-lora-rank 16`，有个客户的 adapter 是 $$r = 64$$。会怎样？改 `--max-lora-rank 64` 要多付多少显存（8B，全部线性层）？

   <details markdown="1"><summary>答案</summary>
   加载报错（rank 超过槽位）。改成 64 后槽位是 $$4 \times 167.8\text{M} \times 2 = 1.34$$ GB，原来是 $$4 \times 41.9\text{M} \times 2 = 0.34$$ GB，多 1 GB，从 KV cache 里扣——即使其他三个客户都是 $$r = 16$$，槽位也按最大秩预留。
   </details>

## 下一篇

三篇正文到此结束：低秩假设与四本账、每个旋钮的对照数字、从 adapter 文件到 multi-LoRA 服务。[系列总结](/lora-series-recap-and-self-test.html)把三篇压成一张速查表，拎出贯穿三篇的四条线、列常见误区，再给一套通关自测。
