---
layout: post
series: algorithm-tooling
title: "算法工程师的工具箱（07）：系列总结与通关自测"
subtitle: "Tooling for AI Algorithm Engineers: Series Recap and Final Self-Test"
tags: [AI, LLM, PyTorch, Python]
catalog: true
date: 2026-01-22 20:00:00
---

六篇正文回答了一个问题：**把一个想法变成一次能跑、能复现、能与 baseline 对比的实验，要经过哪些工具、每个工具用到什么程度**。第一篇讲训练代码里那一小撮 Python 语法与流式过语料，第二篇在 NumPy 上建立形状直觉并用 Pandas、Matplotlib 看结果，第三篇写出二十行训练循环，第四篇算这个循环要多少显存，第五篇用 Hugging Face 的六个库组装一次真实的 LoRA SFT，第六篇解释为什么快为什么慢、怎么让实验三个月后还能复现。六篇合起来，是[《算法工程师的工具箱》总纲](/tooling-for-ai-algorithm-engineers.html)里 L1 那一层的全部：会到能做实验为止，越过那条线就进 Infra 地图。

本文不讲新内容，做三件事：把六篇压成一张表与六段回顾，把贯穿六篇的几条线拎出来，然后给一套三段式的通关自测——判断与计算、跨篇综合、面试题。各篇末尾的自测检验的是"这一篇读懂了没有"，这里检验的是"六篇能不能连起来用"。

> **读完这六篇，你应该能回答哪些问题？[^q0] 哪些数字与结论必须能脱口而出？[^q1] 怎么判断自己是"读过"还是"掌握"了？[^q2]**

## 一、总览：系列回答的问题与主线

系列的一句话主张是：**"跑得动跑不动"最终是显存与算力的算术，不是对工具的熟悉程度**。每一篇都从要做的事出发把工具带出来，讲到能做为止，并给出一个能算出来、能跑出来的数字——19 MB 文件读成 `list` 要 103 MB、手写 attention 与 PyTorch 对到 $$2.65 \times 10^{-7}$$、84 万参数一分钟 PPL 128 → 9.6、每参数 16 字节、0.5B 上 85% 的 token 被 mask、decode 每 token 4.8 ms。六篇各算一笔账，用的是同一个模型（Llama-3-8B，8.03B 参数、$$d = 4096$$、32 层）、同一张卡（H100 SXM，80 GB、3.35 TB/s、bf16 稠密约 989 TFLOPS）、同一套脚本（CPU 可跑）。

| 篇 | 回答的问题 | 一句话结论 | 必记的数字 / 公式 |
|---|---|---|---|
| [第一篇：Python 使用层](/python-in-use-for-algorithm-engineers.html) | 别人训练代码里的 `__getitem__`、`yield`、`@torch.no_grad()`、`**kwargs` 在干什么？10 GB 语料怎么在 16 GB 内存里过一遍？多线程为什么没用？ | 它们各是一个 Python 协议，PyTorch 建在上面；生成器让峰值内存与文件大小无关；GIL 让 CPU 密集的线程不并行，进程池才有用 | 19.4 MB → `list` 102.9 MB（5.3 倍）vs 生成器 11.5 MB；串行 / 8 线程 / 8 进程 = 0.46 s / 0.46 s / 0.15 s（1.0× / 3.1×）；六个语法 ↔ 六个 PyTorch API |
| [第二篇：数据科学三剑客](/numpy-pandas-matplotlib-for-algorithm-engineers.html) | 看到 attention 公式能不能写出 `einsum`？拿到评测结果能不能找出退化的题？看 loss 曲线该看哪里？ | 所有"沿哪个维度"是一个概念；广播太宽容所以形状错误常不报错；与参考实现对数值到浮点精度是验证手写算子的标准 | 广播三条规则；`(3,) + (3, 1)` → `(3, 3)`；`"btd,bsd->bts"` / `"bts,bsd->btd"`；误差 $$2.65 \times 10^{-7}$$；`ci95` $$= 1.96\sqrt{\hat p(1 - \hat p)/n}$$；对数 x 轴、多 seed 阴影带 |
| [第三篇：PyTorch 使用层（上）](/pytorch-in-use-five-objects-and-a-training-loop.html) | 不用 `Trainer`，能不能从零写训练循环、训一个小 Transformer、解释每一行？ | 使用层只有五个对象；Autograd 三件事（记图、累加、`no_grad`）；二十行里每一行对应一个概念，`Trainer` 是它加外壳 | 五个对象；`backward()` 累加所以要 `zero_grad`；`.float()` / `ignore_index=-100` / `clip_grad_norm_(1.0)`；84 万参数、第一步 loss 5.07 ≈ $$\ln 128$$、PPL 128 → 9.6；CPU 上 bf16 慢 30 倍 |
| [第四篇：PyTorch 使用层（下）](/pytorch-in-use-mixed-precision-memory-ledger-and-multi-gpu.html) | 8B 全量微调要多少显存？LoRA 为什么放得进一张卡？OOM 看哪一块？ | 每个可训练参数 16 字节；激活是账外的一块、与参数量无关；"全量还是 LoRA"的第一道约束是有几张卡 | 2 + 2 + 4 + 4 + 4 = 16 字节；128.5 / 16.7 / 5.1 GB；激活 $$B = 1$$、$$T = 4096$$ 约 16.5 GiB；checkpointing 多约 30% 计算换 16.5 → 1 GiB；FSDP 8 卡每卡 16 GB |
| [第五篇：Hugging Face 生态](/hugging-face-ecosystem-six-libraries-and-a-lora-sft.html) | 能不能用 `peft` + `trl` 一小时跑起 LoRA SFT？卡住能不能读源码找原因？ | 六个库各管一段，六行组装；背后的每件事都在二十行里有位置；从 `compute_loss` 往下追是学后训练最快的路 | 三个文件；Qwen2.5-0.5B 494M、可训练 8.80M（1.78%）、状态 141 MB；85% 的 token 被 mask；20 步 5.3 → 1.7；$$\frac{\alpha}{r} BAx$$、$$\alpha = 2r$$ |
| [第六篇：GPU 直觉与实验管理](/gpu-intuition-and-experiment-management.html) | 不写 kernel，能不能解释训练为什么慢、decode 为什么快不起来、OOM 从哪来？三个月后能复现吗？ | 两个上限之比是 ridge；decode 强度 1 是 memory-bound 所以 batch 大才快；显存四块；七项记录齐了才谈复现 | ridge ≈ 295 FLOP/字节；16.06 GB / 3.35 TB/s ≈ 4.8 ms → 209 token/s；prefill 4096 token 约 67 ms；MFU 40–50%；KV cache 131 KB/token；反向 ≈ 2 × 前向；seed 差 0.14 |

### 1. 本文的章节安排

| 章 | 内容 |
|---|---|
| 二 | 逐篇回顾：核心问题、结论、必记、常见误解 |
| 三 | 贯穿六篇的五条线：形状、字节的账、惰性与进程、从封装回到二十行、一个数字不算结论 |
| 四 | 常见误区表 |
| 五 | 通关自测：A 判断与计算 10 题、B 跨篇综合 5 题、C 面试题 7 题、D 掌握判据 |
| 六 | 下一步 |

## 二、逐篇回顾

### 1. 第一篇：Python 使用层——读懂训练代码的语法、流式过一遍语料、把实验写成脚本

**核心问题**：别人的训练代码里 `__getitem__`、`yield`、`@torch.no_grad()`、`with autocast(...)`、`**kwargs` 各在干什么？一份 10 GB 的 JSONL 语料怎么在 16 GB 内存的机器上过一遍、去重、统计？预处理开多线程为什么没用、开多进程为什么也到不了 8 倍？

**结论**：这一小撮语法不是 PyTorch 发明的，是 Python 的协议，PyTorch 只是约定"你按这个形状写，我就能用"：`__len__` / `__getitem__` 是 `Dataset` 的全部要求；生成器是 `DataLoader` 的骨架，每次要下一个才算下一个；`model(x)` 走 `__call__` 再到 `forward`，hooks 挂在中间，所以永远写 `model(x)`；装饰器是"函数包函数"；上下文管理器进入时改状态、退出（含异常）保证恢复；`**kwargs` 把一个 dict 原样透传给下一层。过语料用生成器串成"读 → 过滤 → 去重 → 统计"的流水线，同一时刻内存里只有一条记录加去重的哈希集合，这是数据工程的最小形态，`datasets` 的 `streaming=True` + `.filter()` + `.map()` 底层是同一件事。配置全用 `@dataclass`：`replace` 覆盖、`asdict` 存盘、`__post_init__` 校验，可变默认值必须 `field(default_factory=...)`。预处理跑快靠进程不靠线程：GIL 让同一时刻只有一个线程执行字节码，`DataLoader(num_workers)` 与 `datasets.map(num_proc)` 开的都是进程池。

**必记**：

- 19.4 MB 的 JSONL 读成 `list` 峰值 102.9 MB（文件大小的 5.3 倍），生成器 11.5 MB 且与文件大小无关；按 5 倍算，10 GB 读成 `list` 要 50 GB。
- 10 万行 CPU 密集预处理：串行 0.46 s、8 线程 0.46 s（1.0×）、8 进程 0.15 s（3.1×）；差在进程启动与 `pickle`，`chunksize` 是最重要的旋钮。
- 六个语法 ↔ 六个 API：`__len__` / `__getitem__` ↔ `Dataset`；生成器 ↔ `DataLoader`；`__call__` → `forward` ↔ `nn.Module`；装饰器 ↔ `@torch.no_grad()`；上下文管理器 ↔ `with autocast`；`**kwargs` ↔ 配置透传。
- traceback 最后一行是错、往上第一帧是位置、找自己文件的最后一帧；形状 / 设备 / 类型三类错各有第一反应；`import torch` 出问题先怀疑环境。

**常见误解**："CPU 密集的预处理开线程能加速"——GIL 让 8 线程 1.0×，线程只对等待（I/O、等 GPU）有用。另一个："`model.forward(x)` 与 `model(x)` 一样"——输出一样，但前者跳过了 `__call__` 里的 forward hooks，挂了 hook 时行为不同。

### 2. 第二篇：数据科学三剑客——NumPy 的形状直觉、Pandas 的错误分析、Matplotlib 的曲线

**核心问题**：看到一个 attention 的公式，能不能写出对应的 `einsum`？拿到评测结果，能不能按类别算出错误率、找出退化的题？看到一条 loss 曲线，知道该看哪里？

**结论**：形状直觉先在 NumPy 上建立，因为 Tensor 的语义与 ndarray 一致而没有任何干扰。ndarray = 内存 + 形状 + dtype，`nbytes` 是显存账的起点；整数索引消灭一维、切片保留一维。所有"沿哪个维度"是一个概念：softmax 沿词表维、LayerNorm 沿 hidden 维、attention 沿 key 维，那一维在结果里消失，`keepdims=True` 留下一个 1 供广播。广播只有三条规则，危险在于它太宽容：`(3,) + (3, 1)` 静默变成 `(3, 3)` 的外积表，写错轴的 softmax 形状正确数值全错——形状 bug 的第一种是"能跑但错"，防御只有 `assert` 形状。`reshape` 不移动数据，`transpose` 也不移动但让内存不连续（PyTorch 里 `view` 前要 `contiguous`）。`einsum` 把公式翻译成代码：只在左边的字母被求和、两边都有的被保留、右边的顺序是输出形状。30 行 NumPy 的 causal attention 与 `F.scaled_dot_product_attention` 对到 $$2.65 \times 10^{-7}$$——"与参考实现对数值到浮点精度"是验证任何手写算子的标准方法。Pandas 的 `groupby` / `merge` / `query` 回答"哪类题好、哪类差、相比 baseline 哪些退化了"，读表要带置信区间；Matplotlib 两个习惯：对数 x 轴看训练早期，多 seed 画均值与阴影带。

**必记**：

- 广播三条规则：右对齐；每维相等或其中一个是 1；为 1 的被复制。`(4, 3) + (4,)` 报错，`(4, 3) + (4, 1)` 是每行一个标量；`(32, 128, 4096)` 的 float32 是 64 MiB。
- $$QK^T$$ 是 `"btd,bsd->bts"`，$$PV$$ 是 `"bts,bsd->btd"`；带 head 是 `"bhtd,bhsd->bhts"`。
- 对数值的标准：float32 约 $$10^{-6}$$、float64 约 $$10^{-12}$$，不是"看起来差不多"。
- `ci95` $$= 1.96\sqrt{\hat p(1 - \hat p)/n}$$：530 题的评测里 algebra +7.5 超过 ±5.8 可信，combinatorics +5.0 在 ±10.9 里分辨不出，80 道题不够。
- 第一步 loss 该是 $$\ln V$$（图里 7.2 对应 $$V \approx 1300$$）；5 个 seed 的均值差 0.30、约 1.7 个标准差（seed 间标准差 0.179），两条阴影带后期不重叠才算差别。

**常见误解**："形状不对会报错"——广播把 `(3,)` 与 `(3, 1)` 拼成 `(3, 3)`，沿错轴的 softmax 形状照样正确，只有下游 loss 不对时才发现。另一个："两条曲线一条在下面就是更好"——单个 seed 一路抖动、与另一方法的均值交叉，只看一条分不清是方法还是运气。

### 3. 第三篇：PyTorch 使用层（上）——五个对象与二十行训练循环

**核心问题**：不用 `Trainer`，能不能从零写一个训练循环、在小数据集上训一个小 Transformer、并解释每一行为什么在那里？

**结论**：PyTorch 的使用层只有五个对象——`Tensor`（ndarray + `device` + `dtype` + `requires_grad`）、Autograd、`nn.Module`、`Dataset` / `DataLoader`、`Optimizer`——在一步训练里各站一个位置，数据沿一个环流动：取数组 batch → 前向算 logits → `cross_entropy` → `backward()` 累加到 `.grad` → `step()` 更新。Autograd 只需要知道三件事：前向记图；`backward()` 把梯度**累加**到 `.grad`（所以每步要 `zero_grad`，也因此梯度累积不需要额外代码）；`no_grad` 下不建图（推理评测必加，否则中间量堆到 OOM）。`nn.Module` 两个方法：`__init__` 里赋给 `self` 的子模块自动注册（`super().__init__()` 必须调；Python list 不注册要用 `ModuleList`），`forward` 只写前向；`state_dict()` 就是 checkpoint。二十行训练循环比教程的五行多出四样，少一样迟早出事：`autocast`、`.float()` 与 `ignore_index=-100`、`clip_grad_norm_`、warmup + cosine 调度。`Trainer` / `SFTTrainer` 做的是同一件事加日志、评估、checkpoint、分布式——行为不符合预期时回到这二十行想"它在我这张表的哪一行做了不同的事"。

**必记**：

- 手算例子 $$w = 3$$、$$x = 2$$、$$L = (wx - 1)^2$$，`w.grad` $$= 2(wx - 1) \cdot x = 20$$；第二次 `backward` 报错因为图已释放，每次前向建新图。
- 二十行里的对应：`autocast(bfloat16)` 是混合精度、`view(-1, V)` 是 reshape、`.float()` 是 softmax 的数值、`ignore_index=-100` 是 SFT 的 loss mask、`clip_grad_norm_(1.0)` 是梯度裁剪、`zero_grad(set_to_none=True)` 省一次显存写、`.item()` 每 10 步一次因为它会同步 GPU。
- 字符级小 Transformer：词表 128、4 层、$$d = 128$$、4 头、序列长 128，共 840,448 参数；第一步 loss 5.07 ≈ $$\ln 128 = 4.85$$；CPU 一分钟验证 loss 2.261、PPL 128 → 9.6；grad_norm 在 0.3 上下未触发裁剪；warmup 50 步到 $$3 \times 10^{-4}$$、cosine 到 $$3 \times 10^{-5}$$。
- CPU 上开 bf16 `autocast` 一步 1.4 s、关掉 0.05 s——慢 30 倍；混合精度的收益完全来自硬件。

**常见误解**："教程里五行的循环就是训练循环"——它能跑 MNIST，LLM 训练的四样标配（`autocast`、`.float()` + `ignore_index`、`clip_grad_norm_`、调度）少一样迟早出事。另一个："`autocast` 在哪都能加速"——CPU 没有 bf16 硬件路径，PyTorch 用软件模拟，反而慢 30 倍。

### 4. 第四篇：PyTorch 使用层（下）——混合精度、显存的账与多卡启用

**核心问题**：能不能算出一个 8B 模型全量微调要多少显存、为什么 LoRA 能放进一张卡？OOM 的时候知道看哪一块？

**结论**：`autocast` 按一张内置表决定每个算子的精度——矩阵乘在 bf16 上跑，softmax、LayerNorm、loss 一类 reduction 留在 fp32——它**不改变参数的存储精度**：参数本体仍是 fp32 主权重，前向时临时转成 bf16。主权重与优化器状态必须 fp32，因为 bf16 只有 7 位尾数，$$1.0 + 10^{-5}$$ 在 bf16 里还是 $$1.0$$，更新会被吞掉；bf16 与 fp32 同指数位不需要 loss scaling，fp16 最大 65504、要靠 `GradScaler` 兜底。于是训练时每个**可训练**参数在显存里有五份东西，2 + 2 + 4 + 4 + 4 = 16 字节，全随参数量伸缩。代入 Llama-3-8B：全量 128.5 GB，一张 80 GB 的卡放不下；LoRA 冻结基座只留一份 bf16 权重 16.1 GB，41.9M 可训练参数的 16 字节只有 0.67 GB，合计 16.7 GB；QLoRA 把基座量化到 4 bit 是 4.4 + 0.67 ≈ 5.1 GB。"全量还是 LoRA"的第一道约束是有几张卡，先于效果。激活是账外的一块，不按参数算、与 batch × 序列长度 × 层数 × $$d$$ 成正比，可以比参数与状态加起来还大；gradient checkpointing 只存每层输入、反向重算，多约 30% 计算。多卡启用即可：DDP 每卡一份完整模型、all-reduce 梯度，前提是放得进一张卡；FSDP 把参数 / 梯度 / 状态切到各卡、按层 all-gather。

**必记**：

- 16 字节 / 可训练参数 = bf16 权重 2 + bf16 梯度 2 + fp32 主权重 4 + AdamW 两个矩 4 + 4；SGD 无矩是 8，8-bit 优化器约 10；梯度 + 状态那部分是 14 字节。
- Llama-3-8B：全量 $$8.03\text{B} \times 16 = 128.5$$ GB；LoRA $$r = 16$$：$$8.03\text{B} \times 2 + 41.9\text{M} \times 16 = 16.1 + 0.67 = 16.7$$ GB；QLoRA 4.4 + 0.67 ≈ 5.1 GB；推理只有一份 bf16 权重 16 GB，训练是推理的 8 倍。
- 激活：残差流一份 $$4096 \times 4096 \times 2 = 32$$ MiB/层、32 层 1.0 GiB；反向要保存的中间量约 528 MiB/层、32 层 16.5 GiB；$$B = 8$$ 时 132 GiB；checkpointing 换成 1 GiB，多约 30% 计算（前向 1 + 反向 2 + 重算 1，$$4/3$$）。
- OOM 按爆的时机归因：加载就爆是权重；第一步 `backward` 爆是梯度与状态；序列变长爆是激活；加了 LoRA 还爆看激活；生成时爆是忘了 `no_grad` 或 KV cache；reserved 远大于 allocated 是碎片。
- FSDP 8 卡上 128.5 GB → 每卡 16 GB；`numel() × element_size()` 是任何张量的字节数；跑前算、跑后 `max_memory_allocated()`，误差 30% 内算会算。

**常见误解**："开了 `autocast` 参数就是 bf16 了、显存减半"——参数存储精度不变，减半的是矩阵乘的中间结果；主权重与两个矩仍是 fp32。另一个："加了 LoRA 还 OOM 说明 LoRA 没生效"——LoRA 只减 14 字节 / 参数那一块，激活与参数量无关，长序列大 batch 下它才是大头。

### 5. 第五篇：Hugging Face 生态——六个库与一次 LoRA SFT 的组装

**核心问题**：能不能用 `peft` + `trl` 在一小时内跑起一个 LoRA SFT？卡住的时候能不能直接读源码找到原因？

**结论**：拿到一个模型先看 Hub 上的三个文件：`config.json`（结构超参数，能算出参数量）、`tokenizer.json`（词表、特殊 token、chat template）、`*.safetensors`（`state_dict` 的磁盘格式，可部分加载、不能执行代码）。六个库各管一段并对应第三篇的五个对象：`transformers` 给 `nn.Module` 与 tokenizer、`datasets` 给 `Dataset`（Arrow、内存映射）、`tokenizers` 训与编码词表、`peft` 改 `nn.Module`（在线性层旁边挂 LoRA）、`trl` 给训练循环、`accelerate` 给第四篇的多卡启动。六行组装一次 LoRA SFT，背后的六件事——chat template、回复之外的 label 置 −100、packing、LoRA 挂载并冻结基座、AdamW 只更新 $$A, B$$、bf16 / 裁剪 / 调度——都在二十行里有位置。在 0.5B 上跑通：七个线性层挂 LoRA，可训练 1.78%，训练状态 141 MB；一个 batch 85% 的 token 被 mask，只有 assistant 的几个 token 进入交叉熵；20 步 loss 5.3 → 1.7，答案学会了（Paris、Rome）却没学会在 `<|im_end|>` 停下——**结束符必须进 loss、且要见够多次**。读源码是学后训练最快的路：`modeling_llama.py` 的每个类是第三篇的 `nn.Module`，`dpo_loss` 几十行，`peft` 的 `Linear.forward` 一行核心，`generate` 的每个采样参数是一个 `LogitsProcessor`；库的接口会变，从 `compute_loss` 往下追的方法不变。

**必记**：

- Qwen2.5-0.5B 的 `config.json`：hidden 896、24 层、14 / 2 个头、intermediate 4864、vocab 151936、`tie_word_embeddings` true（省 $$V \times d = 136$$M）；494M 参数。
- LoRA $$r = 16$$、$$\alpha = 32$$：输出加 $$\frac{\alpha}{r} BAx$$，常取 $$\alpha = 2r$$；`target_modules="all-linear"` 挂七个线性层；合并 $$W' = W + \frac{\alpha}{r} BA$$，推理零开销。
- 0.5B 可训练 8.80M / 494M = 1.78%（8B 是 0.52%，小模型 $$d$$ 小占比更高）、训练状态 ≈ 8.80M × 16 = 141 MB、fp32 冻结权重 1.98 GB。
- loss mask：batch `(4, 36)` 里 122 / 144 = 85% 是 −100；真实 SFT 数据回复长，比例反过来。推理必须用与训练相同的 chat template。
- 源码入口：`modeling_llama.py`、`dpo_trainer.py` 的 `dpo_loss`、`grpo_trainer.py`、`peft/tuners/lora/layer.py` 的 `result += lora_B(lora_A(dropout(x))) * scaling`、`sft_trainer.py` 与 collator、`logits_process.py`。

**常见误解**："loss 降下来了 SFT 就成了"——20 步 loss 5.3 → 1.7，模型学会了答案却不会停，因为 `<|im_end|>` 见得太少；要看生成输出，不只看 loss。另一个："学后训练先读论文"——论文写的是想法，代码写的是实际做法；先读 `trl` 里的实现再读论文，还能看到 IPO、hinge 等变体各改了哪一行。

### 6. 第六篇：GPU 直觉与实验管理——两个上限、四块显存、能复现

**核心问题**：不写 kernel，能不能解释一次训练为什么慢、一次推理为什么快不起来、一个 OOM 从哪里来？三个月后能不能复现今天这次实验？

**结论**：GPU 直觉只有两个数字和一张表。算力（H100 bf16 稠密约 989 TFLOPS）与显存带宽（3.35 TB/s）之比是 ridge ≈ 295 FLOP/字节：算术强度高于它受算力限制（compute-bound），低于它受带宽限制（memory-bound），这就是 roofline 的全部。decode 每生成一个 token 要读全部权重、每个权重只做 2 次运算，强度 1——batch = 1 时 16.06 GB / 3.35 TB/s ≈ 4.8 ms 是下限，模型多聪明都没用；batch 加到 128 时间几乎不变、吞吐涨 128 倍，到 ridge 之后才受算力限制，这就是"batch 大才快"。prefill 与训练是 compute-bound，效率指标是 MFU，好的训练 40–50%，剩下的花在通信、数据加载、小算子、等待。显存四块：权重、梯度与优化器状态、激活、KV cache——第四块推理特有，Llama-3-8B 每 token 131 KB、并发 64 个 8K 上下文就 64 GB 比权重还大，这是 GQA 与 MLA 的动机。kernel launch 有几微秒固定开销所以小算子多了 GPU 空转；stream 异步所以 `time.time()` 测不出 GPU 时间、`loss.item()` 会隐式同步。读 profiler 表：GEMM 与 attention 应占大头，反向约是前向的两倍；`copy_` / `contiguous` 或几千个小算子占大头就是形状转换或 launch 开销在吃时间。实验管理的最小记录是一行七项，齐了才谈复现；同 seed 两次完全一致，换 seed 差 0.14——20 步训练里 seed 的影响比很多"方法改进"都大。

**必记**：

- ridge $$= 989 \times 10^{12} / 3.35 \times 10^{12} \approx 295$$ FLOP/字节；decode 强度 $$2N / 2N = 1$$，batch $$B$$ 时强度 $$B$$。
- decode 下限 $$16.06\ \text{GB} / 3.35\ \text{TB/s} \approx 4.8$$ ms → 209 token/s；batch 128 仍 4.8 ms、26700 token/s；batch 512 算力时间 8.3 ms 超过带宽时间，compute-bound。
- prefill 4096 token：$$2 \times 8.03 \times 10^9 \times 4096 \approx 66$$ TFLOP，$$66 / 989 \approx 67$$ ms；MFU 40–50% 算好。
- KV cache 每 token 每层 $$2 \times n_{kv} \times d_{head}$$ 个数：Llama-3-8B $$2 \times 8 \times 128 \times 32 \times 2 = 131$$ KB/token，8K 上下文 1 GB，64 并发 64 GB；GQA 8 个 kv 头把它缩到 1/4。
- profiler：`mm` 25% + `addmm` 10% + attention 前后向 26% ≈ 60% 正常；attention 反向 8.4 ms vs 前向 5.2 ms——反向 ≈ 2 × 前向。
- 七项记录：run id · commit · 配置文件 · 数据版本 · seed · 环境 · 指标；同 seed 两次 3.237898 一致，seed 1 差 0.1404。

**常见误解**："decode 慢是模型算得慢"——算力时间 batch 1 时接近 0，全部时间在搬 16 GB 权重；提速只能靠更高带宽、量化（读的字节少）或加大 batch。另一个："`time.time()` 前后一减就是这一步的 GPU 时间"——CPU 把 kernel 扔进 stream 就往下走，要 `torch.cuda.synchronize()` 或用 profiler。

## 三、贯穿全系列的几条线

### 1. 形状：从 `__getitem__` 取一条到 profiler 表里每个算子

总纲把它叫形状线，六篇每篇往前推一步。第一篇的 `__getitem__` 返回一条样本、`collate` 把一批 pad 到最长，第一个 batch 就有了形状 `[4, 25]`；第二篇把形状做成一门语言——轴、`keepdims`、广播三条规则、`reshape` 拆 head 再 `transpose` 挪到前面、`einsum` 的下标——并指出形状 bug 最危险的一种是"能跑但错"，防御是 `assert`。第三篇让同一套规则在 Tensor 上原样成立，二十行里的 `logits.view(-1, V)` 就是它；第一步 loss 该等于 $$\ln V$$，是用词表大小检查初始化。

第四篇把形状换成字节：激活是 $$B \times T \times$$ 层数 $$\times d$$ 个 bf16，序列从 4K 到 32K 激活 8 倍、显式存的 $$[T, T]$$ score 64 倍——形状直接决定 OOM 落在哪一块。第五篇的 loss mask 是形状上的一行数字：`(4, 36)` 的 labels 里 122 个 −100；第六篇的 KV cache 公式与 profiler 里 `mm` 调用 43 次（4 层 × 每层几个线性层 × 前后向）都是在数形状。第一篇说"形状错误经常不报错"、第二篇说"关键处 `assert` 形状"、第四篇说"OOM 先问哪一块"——同一个习惯在三篇里各出现一次。

### 2. 字节的账：从 `nbytes` 到 16 字节 / 参数到 4.8 ms / token

系列主张"跑得动跑不动是算术"，这条线是那道算术本身。第二篇给起点：`nbytes` = 形状 × dtype 字节数；第三篇加上 `device` 与 `dtype`，并用 CPU 上 bf16 `autocast` 慢 30 倍的陷阱说明精度的收益来自硬件。第四篇把账算全：每个可训练参数 16 字节、为什么主权重必须 fp32、代入 Llama-3-8B 得 128.5 / 16.7 / 5.1 GB、再加账外的激活 16.5 GiB，`numel() × element_size()` 是整篇的加法。第五篇在真实模型上验账：Qwen2.5-0.5B 可训练 8.80M × 16 = 141 MB，CPU 上都能微调。

第六篇把同一笔账放到时间轴上：decode 每 token 要搬 16.06 GB 权重，除以 3.35 TB/s 得 4.8 ms；量化到 4 bit 字节降到 1/4 时间也降到 1/4，这是量化对推理有效的根本原因，而训练是 compute-bound 所以量化对训练用处不大；KV cache 131 KB / token 是第四块字节。第四篇的"推理只有一份权重、训练是它的 8 倍"与第六篇的"显存四块"表，是同一张账的训练侧与推理侧。

### 3. 惰性与进程：生成器、`DataLoader`、`num_proc`、每个 rank 一个进程

第一篇建立两个事实：生成器让数据不必全进内存（19 MB → 12 MB 峰值，与文件大小无关），GIL 让 CPU 密集的线程无用、进程池才有用（3.1×）且进程间不共享内存。第三篇的 `DataLoader` 正是这两件事的产品：`for batch in loader` 惰性取数，`num_workers` 个子进程预取。第五篇的 `datasets` 是同一套的工业版：`streaming=True` 所以没有 `len()`、`.map()` 不立即执行、`num_proc=8` 就是那个 `Pool`。

第四篇把"进程"推到多卡：`torchrun --nproc_per_node=8` 起 8 个进程，每张卡一个 rank，进程间只通过集合通信交换梯度，DDP 数学上等价于一个 8 倍的 batch；第五篇的 `accelerate launch` 是它的统一入口。所以第一篇"传给 `Pool.map` 的函数必须模块顶层可导入、全局变量在子进程里是拷贝"这类规则，在 `DataLoader` worker、`datasets.map` 与 DDP 的每个 rank 上都成立。

### 4. 从封装回到二十行：追溯的路径

总纲把第三种能力叫追溯：从高层封装回到二十行，从报错回到四块，从慢回到 profiler 表。第一篇给第一层地图：PyTorch 的每个 API 建在哪个 Python 协议上，traceback 从下往上读、跳过 `_call_impl`、找自己文件的最后一帧。第三篇给第二层：二十行的每一行对应一个概念，`Trainer` 的 `compute_loss` 是第 9–10 行、`training_step` 是第 11–13 行，行为不对时回到这张表想"它在哪一行做了不同的事"。第五篇给第三层：六行 SFT 背后的六件事各落在二十行的哪一行（chat template 在 `__getitem__`、loss mask 在 `ignore_index=-100`、packing 在 `collate_fn`、LoRA 在 `requires_grad`），然后从 `compute_loss` 往下追源码——答案学会了却不会停这种问题只有读源码才能定位。

第四篇与第六篇给另外两条：OOM 按爆的时机归到四块之一；慢按 profiler 表归到 GEMM、attention、`copy_`、launch 开销或 GPU 空转。四条路径的共同点是都有一张"我这一层的表"——协议对照表、二十行、四块、profiler 前几行——问题先落到表上的某一格，再往下一层去。越过表能解释的范围（一个算子太慢、一个并行策略框架不支持、一个 OOM 调参绕不过）就是进 Infra 地图的信号。

### 5. 一个数字不算结论：置信区间、多 seed 与七项记录

第二篇两次说这件事：Pandas 那张表带上 `ci95` 之后，algebra 的 +7.5 超过 ±5.8 可信、combinatorics 的 +5.0 在 ±10.9 里分辨不出；Matplotlib 那张图里单个 seed 一路抖动、与另一方法的均值交叉，5 个 seed 的均值加阴影带到后期不重叠（差 0.30、约 1.7 个标准差）才能说 B 更好。第六篇给出它在工程上的原因与物质基础：同 seed 两次结果完全一致（GPU 某些 kernel 非确定，需要 `use_deterministic_algorithms`），换个 seed 差 0.14——20 步训练里 seed 的影响比很多"方法改进"都大；七项记录少任何一项，结论都只是"当时好像是这样"。

第一篇的 `pip freeze` 存进 run 目录、`asdict(cfg)` 存成 `config.json`，第五篇的模型卡评测数字要带着置信区间读、`datasets` 的 revision 是数据版本，都是这七项的组成部分。总纲的"追溯"能力包含"实验结果三个月后能复现"，这条线是它的另一半：不只是能找到问题，还要能证明结论是真的。

| 概念 | 出现的篇 | 关系 |
|---|---|---|
| 形状与 `assert` | 一、二、三、四、六 | 一说形状错常不报错；二给规则与防御；三搬到 Tensor；四换成激活字节；六换成 KV cache 与算子调用次数 |
| 字节 / 参数、dtype | 二、三、四、五、六 | 二给 `nbytes`；三给 `dtype` 与 CPU 陷阱；四算 16 字节与三种方案；五在 0.5B 上验账；六换成 decode 时间与四块 |
| 生成器、进程、GIL | 一、三、四、五 | 一给机制与数字；三是 `DataLoader`；四是 `torchrun` 每 rank 一进程；五是 `datasets` 的 `streaming` / `num_proc` 与 `accelerate` |
| 二十行训练循环 | 三、四、五 | 三写出并逐行解释；四算它的显存；五把六行 SFT 的每件事对回它的某一行 |
| `ignore_index=-100` / loss mask | 三、五 | 三给它在循环里的位置与含义；五给实物（85% 被 mask）与结束符要进 loss 的教训 |
| 混合精度与硬件 | 三、四、六 | 三给 CPU 慢 30 倍；四给 `autocast` 的表与 bf16 / fp16；六给 Tensor Core 与 ridge 为什么决定收益 |
| OOM 归因 | 四、六 | 四给按时机归因的表；六加上 KV cache 成四块 |
| seed、置信区间、记录 | 二、六 | 二给 `ci95` 与阴影带；六给 seed 差 0.14 与七项记录 |
| 读源码 | 一、三、五 | 一给协议与 traceback；三给"回到二十行想"；五给六个入口与从 `compute_loss` 往下追 |

## 四、常见误区

| 误区 | 为什么错 | 正确的说法 | 出处 |
|---|---|---|---|
| 预处理慢就开多线程 | GIL 让同一时刻只有一个线程执行字节码，CPU 密集的 8 线程 1.0× | 用进程池，调 `chunksize`；线程只对等待有用 | [第一篇](/python-in-use-for-algorithm-engineers.html) |
| 语料多大就要多大内存 | 读成 `list` 才是 5 倍文件大小；生成器一次只放一条 | 生成器流水线，峰值与文件大小无关（19 MB → 12 MB） | [第一篇](/python-in-use-for-algorithm-engineers.html) |
| `model.forward(x)` 与 `model(x)` 等价 | `__call__` 里先跑 forward hooks 再调 `forward` | 永远写 `model(x)` | [第一篇](/python-in-use-for-algorithm-engineers.html) |
| 形状错了会报错 | 广播太宽容：`(3,) + (3, 1)` 成 `(3, 3)`，沿错轴的 softmax 形状照样对 | 关键处 `assert` 形状，尤其 loss 前的 logits 与 labels | [第二篇](/numpy-pandas-matplotlib-for-algorithm-engineers.html) |
| 手写算子"看起来差不多"就对了 | 标准是误差在浮点精度量级 | float32 约 $$10^{-6}$$、float64 约 $$10^{-12}$$；30 行 attention 对到 $$2.65 \times 10^{-7}$$ | [第二篇](/numpy-pandas-matplotlib-for-algorithm-engineers.html) |
| 总分高 2 个点就是更好 | 80 道题的 ±10.9 分辨不出 5 个点；单 seed 与另一方法交叉 | 带 `ci95` 读表；多 seed 均值加阴影带，不重叠才算差别 | [第二篇](/numpy-pandas-matplotlib-for-algorithm-engineers.html) |
| 两次 `backward` 之间不 `zero_grad` 是 bug | Autograd 的语义是累加 | 有意为之就是梯度累积，用小 batch 模拟大 batch | [第三篇](/pytorch-in-use-five-objects-and-a-training-loop.html) |
| 评测时不加 `no_grad` 只是慢一点 | 每步的中间量被保存等待一个不会来的 `backward` | 显存持续增长直到 OOM，推理评测必加 | [第三篇](/pytorch-in-use-five-objects-and-a-training-loop.html) |
| `autocast` 把参数变成 bf16、显存减半 | 参数存储精度不变，主权重与两个矩仍是 fp32 | 减半的是矩阵乘的中间结果；训练状态仍是 16 字节 / 参数 | [第四篇](/pytorch-in-use-mixed-precision-memory-ledger-and-multi-gpu.html) |
| 加了 LoRA 还 OOM 是 LoRA 没生效 | LoRA 只减梯度与状态那 14 字节 / 参数；激活与参数量无关 | 看激活：开 gradient checkpointing、减 batch、缩短序列 | [第四篇](/pytorch-in-use-mixed-precision-memory-ledger-and-multi-gpu.html) |
| loss 降下来 SFT 就成了 | 0.5B 上 20 步 5.3 → 1.7，答案对了却不会停 | 结束符要进 loss 且见够多次；看生成输出 | [第五篇](/hugging-face-ecosystem-six-libraries-and-a-lora-sft.html) |
| decode 慢是 GPU 算不过来 | 强度 1 远低于 ridge 295，算力时间接近 0 | 时间全在搬 16 GB 权重；batch 大、量化、高带宽才有用 | [第六篇](/gpu-intuition-and-experiment-management.html) |

## 五、通关自测

### A. 判断与计算（10 题）

1. 一份 8 GB 的 JSONL，机器内存 32 GB。`json.loads` 后读成 `list` 能不能装下？用生成器峰值大约多少？

   <details markdown="1"><summary>答案</summary>

   按第一篇实测的 5 倍（5.3 倍）算，`list` 要约 40 GB，装不下；生成器峰值与文件大小无关，只有一条记录加去重的哈希集合，几十 MB。

   </details>

2. `x = torch.zeros(16, 512, 1024, dtype=torch.bfloat16)`：`x.nbytes` 多少？`x[:, -1]` 与 `x[:, -1:]` 各是什么形状？

   <details markdown="1"><summary>答案</summary>

   $$16 \times 512 \times 1024 \times 2 = 16$$ MiB（bf16 每元素 2 字节）；`x[:, -1]` 整数索引消灭一维得 `(16, 1024)`，`x[:, -1:]` 切片保留得 `(16, 1, 1024)`。

   </details>

3. `(8, 1, 64) + (16, 64)` 能广播吗？结果什么形状？

   <details markdown="1"><summary>答案</summary>

   能。右对齐：64 对 64 相等；1 对 16，是 1 的被复制成 16；8 对缺失，当 1。结果 `(8, 16, 64)`——注意它悄悄多出一个维度，是"能跑但错"的典型。

   </details>

4. 第三篇的手算例子 $$w = 3$$、$$x = 2$$、$$L = (wx - 1)^2$$，每次重新前向、连续 `backward()` 三次而不 `zero_grad`，`w.grad` 是多少？

   <details markdown="1"><summary>答案</summary>

   单次梯度 $$2(wx - 1) \cdot x = 20$$，累加三次是 60——Autograd 累加语义；这就是梯度累积不需要额外代码的原因。

   </details>

5. 用 Qwen2.5-0.5B（词表 151936）从头初始化训练，第一步 loss 应约多少？看到 15.0 说明什么？

   <details markdown="1"><summary>答案</summary>

   $$\ln 151936 \approx 11.9$$；15.0 明显偏高，初始 logits 太大，检查输出层初始化（第三篇：第一步 loss ≈ $$\ln V$$ 的 sanity check）。

   </details>

6. 一个 3B 模型全量微调（bf16 + AdamW）不算激活要多少显存？一张 24 GB 的卡够不够？改成可训练 1% 的 LoRA 呢？

   <details markdown="1"><summary>答案</summary>

   $$3 \times 16 = 48$$ GB，24 GB 不够；LoRA：冻结 bf16 $$3 \times 2 = 6$$ GB + $$0.03\text{B} \times 16 = 0.48$$ GB ≈ 6.5 GB，够，余下的留给激活。

   </details>

7. Llama-3-8B 微调，$$B = 2$$、$$T = 8192$$，不开 checkpointing 的激活约多少？开了呢？

   <details markdown="1"><summary>答案</summary>

   第四篇 $$B = 1$$、$$T = 4096$$ 约 16.5 GiB，与 $$B \times T$$ 成正比，×4 得约 66 GiB；开 checkpointing 只存每层输入（$$B = 1$$、$$T = 4096$$ 时 1.0 GiB），×4 约 4 GiB，代价多约 30% 计算。

   </details>

8. 第五篇的 0.5B 模型 $$r = 16$$ 时可训练 8.80M、训练状态 141 MB；把 `r` 改成 32，两者各变成多少？

   <details markdown="1"><summary>答案</summary>

   LoRA 参数量 $$r(m + n)$$ 随 $$r$$ 线性：17.6M（约 3.6%），训练状态 $$17.6\text{M} \times 16 \approx 282$$ MB；冻结基座不变。

   </details>

9. Llama-3-8B bf16 推理，128 个并发请求、每个 4K 上下文，KV cache 多大？加上权重能不能放进一张 80 GB 卡？

   <details markdown="1"><summary>答案</summary>

   $$131\ \text{KB} \times 4096 \times 128 \approx 68.7$$ GB；加 16 GB 权重约 85 GB，放不下——要减并发、缩上下文或量化 KV cache。

   </details>

10. 一步训练墙钟 300 ms，profiler 表里 CUDA 时间合计 90 ms、`aten::mm` 与 attention 占其中 60%。瓶颈最可能在哪？

    <details markdown="1"><summary>答案</summary>

    GPU 空转 70%——GPU 时间加起来远小于墙钟时间，是数据加载、同步（如每步 `.item()`）或 launch 开销在等；GEMM 与 attention 占 60% 说明 GPU 上跑的那部分本身正常。

    </details>

### B. 跨篇综合（5 题）

1. 8B LoRA SFT 在一张 80 GB 卡上，账算出 16.7 GB，但 `SFTConfig` 里 `max_length=8192`、每卡 batch 4 时 OOM。哪一块爆了？改什么？

   <details markdown="1"><summary>答案</summary>

   第四篇：16.7 GB 只是参数与状态，激活与参数量无关、与 $$B \times T$$ 成正比——$$B = 1$$、$$T = 4096$$ 约 16.5 GiB，$$B = 4$$、$$T = 8192$$ 是 8 倍约 132 GiB；"加了 LoRA 还 OOM 不是参数的问题，看激活"。改法：`model.gradient_checkpointing_enable()`（第四篇，多约 30% 计算）、减 batch 用梯度累积补（第三篇：累加语义）、或缩短序列 / 开 `packing`（第五篇）。

   </details>

2. 两次"同样配置"的 SFT，评测差 0.5 个点，同事说新配置更好。用哪几篇的什么判断？

   <details markdown="1"><summary>答案</summary>

   第六篇：先查七项里有没有哪一项其实不同（commit、数据版本、环境、seed），并且 20 步训练里换 seed 就差 0.14，单个数字不算结论；第二篇：跑几个 seed 报均值与标准差、画阴影带，两条带不重叠才算差别；对评测表带上 `ci95` $$= 1.96\sqrt{\hat p(1 - \hat p)/n}$$，再用 `merge` + `query` 看退化与改善的是哪些题，而不是只看总分。

   </details>

3. 微调后的模型回答正确但一直不停、吐乱码。从哪一行、哪个文件查？

   <details markdown="1"><summary>答案</summary>

   第五篇：0.5B 实测就是这样——答案学会了但没在 `<|im_end|>` 停，因为结束符进 loss 的次数太少；推理时也要用与训练相同的 chat template。第三篇：loss 在哪一行决定——`cross_entropy(..., ignore_index=-100)`，label 为 −100 的位置不算 loss，所以要确认结束符没有被 mask 成 −100。去读 `trl/trainer/sft_trainer.py` 的数据处理 / collator，搜 `completion_only_loss` 或 `-100`（第五篇的入口）。

   </details>

4. 第五篇那个 Qwen2.5-0.5B `merge_and_unload` 后部署在 H100 上，batch 1 decode 每 token 的下限约多少？合并前后这个数字变不变？

   <details markdown="1"><summary>答案</summary>

   第五篇：494M 参数、合并后推理零开销；第四篇：推理只有一份 bf16 权重，$$494\text{M} \times 2 \approx 0.99$$ GB；第六篇：decode 是 memory-bound，$$0.99\ \text{GB} / 3.35\ \text{TB/s} \approx 0.3$$ ms。合并前 LoRA 分支多读 $$A, B$$（8.80M × 2 字节，可忽略）并多几个小算子（第六篇的 launch 开销），合并后才是"零开销"。

   </details>

5. 用 `accelerate launch` 在 8 卡上跑 8B LoRA SFT，DDP 模式。每张卡显存约多少？为什么每张卡是一个进程而不是一个线程？有效 batch 是多少？

   <details markdown="1"><summary>答案</summary>

   第四篇：DDP 每卡一份完整模型与状态，16.7 GB + 激活，8 卡 all-reduce 梯度，数学上等价于 8 倍 batch；第一篇：GIL 让线程不能并行执行 Python 字节码，所以每个 rank 是独立进程、进程间不共享内存，只通过集合通信交换梯度（第四篇）；第五篇：`accelerate` 统一 DDP / FSDP 的启动，就是 `torchrun --nproc_per_node=8`。

   </details>

### C. 面试题（7 题）

1. 让你微调一个 70B 模型，全量还是 LoRA、几张 80 GB 卡？把账算给我看。

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 第一道约束是显存不是效果——每个可训练参数 16 字节（2 + 2 + 4 + 4 + 4）；(2) 全量 $$70 \times 16 = 1120$$ GB，8 卡 640 GB 不够，要 16 卡以上 + FSDP 切状态；(3) LoRA 冻结 bf16 140 GB + 可训练部分的 16 字节（1.3% 时约 14.6 GB）≈ 155 GB，两张卡起；(4) 激活账外：与 $$B \times T \times$$ 层数 $$\times d$$ 成正比，长序列开 gradient checkpointing 多约 30% 计算；(5) DDP 的前提是放得进一张卡，放不进就 FSDP；跑后 `max_memory_allocated()` 对账。
   **追问方向**：QLoRA 能省到多少（4 bit 约 0.5 字节 / 参数）；推理为什么只有 1/8；张量 / 流水并行什么时候才需要。
   **好答案与一般答案的区别**：一般答案说"70B 肯定得 LoRA"；好答案把 16 字节拆成五份、代入数字、再指出激活那块不在这笔账里。

   </details>

2. 训练 loss 不降或第一步就异常，你排查的顺序是什么？

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 第一步 loss 该 ≈ $$\ln V$$（词表 128 是 4.85、32000 是 10.4），明显偏高是输出层初始化；(2) 形状——沿错轴的 softmax、`view(-1, V)` 与 `labels.view(-1)` 对不上都不报错，loss 前 `assert` 形状；(3) 二十行里少了哪一样：`zero_grad`（累加语义）、`.float()`（bf16 下 softmax 丢精度）、`ignore_index=-100`（prompt 被当目标）、`clip_grad_norm_`、warmup、`sched.step()`；(4) 用高层封装时回到二十行想它在哪一行做了不同的事，再从 `compute_loss` 往下读源码；(5) 用对数 x 轴看前 1% 步、多 seed 排除运气。
   **追问方向**：grad_norm 曲线该长什么样（脚本里 0.3 上下、第一步 2.0）；`autocast` 在 CPU 上为什么反而慢。
   **好答案与一般答案的区别**：一般答案说"调学习率试试"；好答案先用 $$\ln V$$ 与形状排除最便宜的两类错，再逐行对二十行。

   </details>

3. 为什么 LLM 推理 batch 大才快？量化为什么对推理有效、对训练用处不大？

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 两个上限：算力 989 TFLOPS、带宽 3.35 TB/s，ridge ≈ 295 FLOP/字节；(2) decode 每 token 读全部权重、每个权重 2 次运算，强度 1，memory-bound，batch 1 下限 16.06 GB / 3.35 TB/s ≈ 4.8 ms；(3) batch $$B$$ 时权重读一次被 $$B$$ 个 token 共享，强度变 $$B$$，搬的时间不变、算的时间线性涨，到 ridge 前吞吐几乎线性涨（128 时 26700 token/s）；(4) 量化把读的字节降到 1/4，memory-bound 的时间也降 1/4；(5) 训练与 prefill 强度数千，compute-bound，读权重不是瓶颈，且反向要高精度梯度；KV cache 131 KB/token 随并发涨，是 batch 做不大的原因之一。
   **追问方向**：MoE 为什么推理特别难（FLOPs 降但全部专家权重都要读）；MFU 40–50% 剩下的去哪了；prefill 67 ms 怎么算的。
   **好答案与一般答案的区别**：一般答案说"GPU 并行度高"；好答案用算术强度把 decode 放到 roofline 的斜线上，并说出 ridge 之后为什么不再涨。

   </details>

4. `CUDA out of memory`，你怎么定位落在哪一块？

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 四块：权重、梯度与优化器状态（14 字节 / 可训练参数）、激活、KV cache；(2) 按爆的时机归因：加载就爆是权重，第一步 `backward` 爆是状态，序列变长爆是激活（训练）或 KV cache（推理），生成时爆是忘了 `no_grad` 或并发太高；(3) 加了 LoRA 还爆不是参数的问题，看激活；reserved 远大于 allocated 是碎片；(4) 对症：换 bf16 / 量化 / 更多卡、LoRA / FSDP / 8-bit 优化器、checkpointing / 减 batch / 缩序列、减并发；(5) 跑前按账算、跑后 `max_memory_allocated()` 对。
   **追问方向**：激活 16.5 GiB 怎么估的（每层约 6 份 $$d$$ 维 + 3 份 $$d_{ff}$$ 维 bf16）；FSDP 与 DDP 每卡各多少；KV cache 公式与 GQA。
   **好答案与一般答案的区别**：一般答案说"减 batch"；好答案先归到四块之一，再给对应那块的手段，并能用数字预测减 batch 够不够。

   </details>

5. 不用 `Trainer` 写一个 LLM 的训练循环，哪几行是 MNIST 教程里没有的、各在防什么？

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 骨架五行：前向、loss、`backward`、`step`、`zero_grad`；(2) `autocast(bfloat16)`：矩阵乘在 bf16、reduction 留 fp32，不开显存与速度差一倍多；(3) `.float()` + `ignore_index=-100`：softmax 在 fp32 上算、prompt 与 padding 不算 loss——SFT 的 loss mask；(4) `clip_grad_norm_(1.0)`：某一步坏梯度让 loss 冲上去回不来；(5) warmup + cosine 调度：不 warmup 前几步可能发散；(6) `SFTTrainer` 的六件事各落在这几行的哪一行（chat template 在 `__getitem__`、mask 在 −100、packing 在 `collate_fn`、LoRA 在 `requires_grad`）。
   **追问方向**：梯度累积怎么写（累加语义，每 $$k$$ 个 batch 才 `step`）；`Trainer` 的 `compute_loss` / `training_step` 对应哪几行；为什么 `model(x)` 不是 `model.forward(x)`。
   **好答案与一般答案的区别**：一般答案背五行；好答案说出多出的四样各防什么事故，并能把 `SFTTrainer` 映射回来。

   </details>

6. 你的实验结论要让人信，从设计到记录你会做哪些事？

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 多 seed 报均值与标准差，画均值加阴影带，不重叠才算差别（脚本里差 0.30 约 1.7 个标准差；20 步里换 seed 差 0.14）；(2) 评测表带 `ci95`，80 道题的 ±10.9 分辨不出 5 个点；(3) `merge` + `query` 列出退化与改善的题逐道看，不只看总分；(4) 七项记录 run id、commit、配置、数据版本、seed、环境、指标一行齐全，配置用 `dataclass` + `asdict` 存进 run 目录，`pip freeze` 锁环境；(5) 同 seed 两次一致是前提（GPU 上某些 kernel 非确定，需要 `use_deterministic_algorithms`）；前 200 步两条带重叠时不说"收敛更快"。
   **追问方向**：W&B / Hydra / git 各管七项里哪一项；数据版本怎么定义（文件 hash 或 `datasets` revision）；"当时好像是这样"的实验怎么补救。
   **好答案与一般答案的区别**：一般答案说"用 W&B 记一下"；好答案先说方差与置信区间决定结论能不能下，再说七项缺一项结论就不成立。

   </details>

7. 一份 10 GB 的 JSONL 要清洗、去重、tokenize，机器 16 GB 内存 8 核，怎么做到又快又不爆内存？

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 不能读成 `list`——5 倍文件大小约 50 GB；生成器逐行读，读 → 过滤 → 去重 → 统计每步一个生成器串成流水线，内存里只有一条记录加去重的哈希集合；(2) CPU 密集的部分用 `multiprocessing.Pool` 不用线程（GIL：8 线程 1.0×，8 进程 3.1×），`chunksize` 让每次传一批以摊薄 `pickle` 开销，函数要模块顶层可导入；(3) 工业版直接用 `datasets`：`streaming=True`（所以没有 `len()`）、`.filter()` / `.map(batched=True, num_proc=8)`，底层是同一套生成器与进程池；(4) 精确去重靠内容哈希集合，近似去重（MinHash）不在本系列；(5) 输出与数据版本（hash）一起记录，数据变了就是另一个实验。
   **追问方向**：为什么进程到不了 8×；`DataLoader(num_workers)` 与它的关系；`.map()` 为什么不立即执行。
   **好答案与一般答案的区别**：一般答案说"用 pandas 分块读"；好答案说出内存为什么是 5 倍、线程为什么无用、以及 `datasets` 在底层做的就是这两件事。

   </details>

### D. 掌握判据

| 水平 | 表现 |
|---|---|
| 读过 | 能说出六篇各讲什么；知道 16 字节 / 参数、ridge、`ignore_index=-100`、`einsum`、GIL 这些名词 |
| 掌握 | A 组能不翻书算出 8 题以上；B 组能说出每题用了哪几篇的什么；拿到一个 OOM 或一条慢的训练能归到四块 / profiler 表的某一格；跑前能算出显存并与 `max_memory_allocated()` 对上 |
| 能教人 | C 组每题能给出全部要点并预判追问；能解释六篇里每个反直觉结论（线程无用、形状错不报错、`autocast` 不改参数精度、decode 慢与算力无关、loss 降了模型不会停）为什么成立；六个脚本改过参数、看过数字怎么变 |

通关标准：A 组至少 8 题、B 组至少 4 题、C 组每题能说出一半以上要点。没过的部分回到第二章对应篇的"必记"，再回该篇正文；总纲说得对，工具的检验是做不是读——六个脚本跑完、改过，L1 才算够。

## 六、下一步

六篇讲的是"用"，每一篇背后都有一个"改"或"为什么"，它们不在范围内：

- **Python 的机制**——GIL 的来历、生成器怎么暂停恢复、描述符与元类、内存模型、C 扩展、打包交付——在 Infra 地图的[《Python for AI Infra》](/python-for-ai-infra.html)，它是第一篇的深入篇，两张地图共享。
- **PyTorch 内部**——Dispatcher、Autograd 引擎、编译、分布式通信栈的实现——在[《深入 PyTorch》](/deep-dive-into-pytorch.html)；本系列讲"用"，它讲"改"。
- **GPU 编程**——CUDA、kernel、Tensor Core、让 profiler 表里那个算子快起来——在[《GPU Kernel 工程》](/gpu-kernel-engineering.html)；本系列只到"读 profiler 知道慢在哪"。
- **并行策略的选择与实现**——张量 / 流水 / 专家并行、checkpoint、容错、MFU——在[《大规模训练工程：从并行策略到容错恢复》](/large-scale-training-from-parallelism-to-fault-tolerance.html)；本系列只到 DDP / FSDP 启用。
- **本系列的前置**——形状规则与 FLOPs、LoRA 的参数量、交叉熵、置信区间——在 [L0《算法工程师的数学》](/math-for-ai-algorithm-engineers.html)；六篇里每个"L0 第几篇"的引用都指向它。
- 这一层在整张地图上的位置，以及 L2 经典机器学习之后往哪走，见[《AI 算法工程师学习地图》](/ai-algorithm-engineer-learning-roadmap.html)。

回到总纲：[《算法工程师的工具箱：从一个想法到一次能跑的实验》](/tooling-for-ai-algorithm-engineers.html)。

## 七、延伸阅读

本系列有意不展开的内容，以及它们在哪个系列里：

- **Python 的机制**：第一篇只讲训练代码里那一小撮语法的用法；它们在解释器里怎么实现——GIL、生成器的暂停恢复、描述符与元类、内存模型、C 扩展、打包交付——在 Infra 地图的 [01 系列](/python-for-ai-infra.html)——它是两张地图共享的基础、本系列第一篇的深入篇，紧接本系列发布。
- **PyTorch 内部**：Dispatcher、Autograd 引擎、编译、分布式通信栈的实现。在 Infra 地图的 [03 系列](/deep-dive-into-pytorch.html)——同样两张地图共享，本系列讲"用"，它讲"改"。
- **GPU 编程**：CUDA、kernel、Tensor Core。本系列只到"读 profiler 知道慢在哪"；写 kernel 在 Infra 05 系列。
- **并行策略的选择与实现**：张量 / 流水 / 专家并行、checkpoint、容错。在 Infra 07 系列。本系列只到 DDP / FSDP 启用。
- **每个训练概念的原理**：混合精度为什么能工作、梯度裁剪剪的是什么、warmup 为什么必须。分别在 L4《Transformer 与 LLM》第六篇与 L3 深度学习基础系列。本系列只讲怎么用、在训练循环的哪一行。
- **后训练算法本身**：SFT 的数据、DPO / GRPO 的原理。在 L5 后训练系列。本系列只到"用 `trl` 跑起来、知道去哪读源码"。


[^q0]: 六个：别人代码里的 `__getitem__` / `yield` / `@no_grad` / `**kwargs` 在干什么、10 GB 语料怎么在小内存里过一遍、多线程为什么没用（协议、生成器、GIL）；一个公式对应什么 `einsum`、形状对不对、评测结果怎么分析、曲线该看哪里；不用 `Trainer` 怎么写训练循环、每一行为什么在那里；这个模型全量微调要多少显存、LoRA 呢、OOM 落在哪一块；怎么用 `peft` + `trl` 一小时跑起 SFT、卡住了去读哪个文件；这一步的时间花在哪、decode 为什么快不起来、三个月后怎么复现。详见[第二章](#二逐篇回顾)。
[^q1]: 19 MB 读成 `list` 103 MB、生成器 12 MB；串行 / 线程 / 进程 0.46 / 0.46 / 0.15 s；广播三条规则与 `(3,) + (3, 1)` → `(3, 3)`；手写 attention 对到 $$2.65 \times 10^{-7}$$；第一步 loss ≈ $$\ln V$$、84 万参数一分钟 PPL 128 → 9.6；16 字节 / 参数 = 2 + 2 + 4 + 4 + 4，Llama-3-8B 128.5 / 16.7 / 5.1 GB；激活 $$B = 1$$、$$T = 4096$$ 约 16.5 GiB，checkpointing 多 30% 计算；0.5B 可训练 1.78%、85% 的 token 被 mask、20 步 5.3 → 1.7；ridge ≈ 295 FLOP/字节、decode 4.8 ms / token、KV cache 131 KB / token；七项记录、seed 差 0.14。详见[第一章](#一总览系列回答的问题与主线)、[第三章](#三贯穿全系列的几条线)。
[^q2]: 用第五章的三段自测：A 组 10 题判断与计算（至少 8 题）、B 组 5 题跨篇综合（至少 4 题）、C 组 7 道面试题（每题说出一半以上要点）；D 组的表给出"读过 / 掌握 / 能教人"三级的表现。详见[第五章](#五通关自测)。
