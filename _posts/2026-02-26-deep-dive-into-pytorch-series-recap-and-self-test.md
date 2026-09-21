---
layout: post
series: deep-dive-into-pytorch
title: "PyTorch 深度实践（11）：系列总结与通关自测"
subtitle: "Deep Dive into PyTorch: Series Recap and Final Self-Test"
tags: [PyTorch, AI, AI-Infra]
catalog: true
date: 2026-02-26 20:00:00
---

十篇正文回答了一个问题：**PyTorch 如何把 Python 里写下的张量计算，变成可求导、可扩展、可优化、可分布式执行的运行时系统**。第一篇画全局地图，第二到四篇讲编程模型（Tensor、Autograd、`nn.Module` 与训练系统），第五、六篇讲算子运行时（Dispatcher 与自定义算子），第七篇讲编译，第八篇讲性能，第九篇讲多卡，第十篇讲这个系统靠什么一直正确、一直可用。十篇合起来，是从 `loss.backward()` 一路追问到硬件与集群的一张图。

本文不讲新内容，做三件事：把十篇压成一张表与十段回顾，把贯穿全系列的几条线拎出来，然后给一套三段式的通关自测——判断与计算、跨篇综合、面试题。各篇末尾的自测检验的是"这一篇读懂了没有"，这里检验的是"十篇能不能连起来用"。

> **读完这十篇，你应该能回答哪些问题？[^q0] 哪些数字与结论必须能脱口而出？[^q1] 怎么判断自己是"读过"还是"掌握"了？[^q2]**

## 一、总览：系列回答的问题与主线

系列的一句话主张是：**PyTorch 不是一个 Python 库，而是一条分层的运行时链路——Python 表达、C++ 运行时、CUDA 执行——每一层各解决一件事，而所有能力（求导、跨设备、编译、分布式）都建立在同一个算子系统之上**。三条线索贯穿十篇：抽象线（Tensor → Autograd → Module → Operator → Compiler）、执行线（Python → C++ → CUDA → Kernel → Hardware）、工程线（Training → Profiling → Distributed → Testing → Build）。读每一篇时问的都是同一组问题：这一层的职责是什么、边界在哪、代价是多少。

| 篇 | 回答的问题 | 一句话结论 | 必记的数字 / 公式 |
|---|---|---|---|
| [第一篇：PyTorch 整体介绍](/pytorch-overall-introduction.html) | 执行 `torch.add(x, y)` 时内部发生了什么？ | 六个职责层、一条动态调用路径、四层源码；Autograd 只是 Operator Table 上优先级更高的一个 Key | 职责六层；源码四层 `torch/` → `torch/csrc/` → `aten/src/ATen/` → `c10/`，依赖只能向下；四个边界；2.0 于 2023 年 3 月 15 日发布 |
| [第二篇：Tensor 与内存布局](/pytorch-tensor-and-memory-layout.html) | Tensor 到底是什么？ | 数据、形状、布局、类型、设备与生命周期的组合；view 只改元数据，`contiguous` / `.to()` / `clone` 才复制 | `offset = storage_offset + Σ i_k × stride_k`；`[2,3,4]` 连续 strides `(12, 4, 1)`；fp16 最大 65504、bf16 尾数 7 位；`expand` 的 stride 为 0；`allocated` 与 `reserved` 不是一个东西 |
| [第三篇：自动求导与动态计算图](/pytorch-autograd-and-dynamic-computation-graph.html) | PyTorch 如何把链式法则变成一次沿图的反向遍历？ | 前向就地建图，每个 `Node.apply` 算一次 VJP，梯度 `+=` 到叶子的 `.grad`；保存一个输出等于保存整张图 | 每节点算 VJP 不物化 Jacobian（`[B, 4096]` 层每样本 $$4096^2$$ 个数）；`.grad` 是累加；version counter 抓 in-place；`gradcheck` 用 float64 |
| [第四篇：nn.Module 与训练系统](/pytorch-module-and-training-system.html) | 模型结构、参数状态、数据管线和训练循环如何组织成可保存、可迁移的系统？ | `__setattr__` 的注册机制决定框架能否发现对象；一切状态都在 `state_dict` 里 | Adam 训练每参数 16 B（4 + 4 + 4 + 4），7B → 112 GB；GradScaler 初始 65536、溢出 ÷2、连续 2000 步 ×2；`prefetch_factor` 默认 2；worker 是进程 |
| [第五篇：Dispatcher 与算子系统](/pytorch-dispatcher-and-operator-system.html) | 谁决定 `x + y` 调用哪个 Kernel？ | 开发态填表（定义 → 注册 → 实现）、运行态查表（入口 → 分发 → 执行），交汇于 Operator Table；包装 Key 做完事再次分发 | DispatchKeySet = 各输入 OR + TLS include − exclude，取最高优先级；一次 `add` 两次经过 Dispatcher；五种实现模式；AutocastCUDA → AutogradCUDA → CUDA |
| [第六篇：C++ 扩展与自定义算子](/pytorch-cpp-extension-and-custom-operators.html) | 自己写一个算子，怎么接入算子系统？ | 三步、两种接入、四个阶段；实现层必须处理 device、dtype、stride、生命周期四件事 | `blocks = (n + 255) / 256`；warp 32 线程、取数按 32 B 扇区，stride 2 访问带宽利用 50%，最坏多搬 8 倍；`gradcheck` 必须 float64；ABI：`_GLIBCXX_USE_CXX11_ABI` 一致 |
| [第七篇：编译执行与图优化](/pytorch-compilation-and-graph-optimization.html) | `torch.compile` 到底做了什么？ | 前端 Dynamo 捕获、中端 AOTAutograd 变换、后端 Inductor 生成，共享 FX Graph；运行时靠 Guard 决定复用 | `f` 热路径：3 次分发 → 0 次、3 次 launch → 2 次；Guard `size[0] == 128` → `s0 > 64`；缓存条目上限 8；`backend="eager"` / `"aot_eager"` / `"inductor"` |
| [第八篇：性能优化与调试](/pytorch-performance-optimization-and-debugging.html) | 如何判断一个程序慢，以及定位它为什么慢？ | CPU 与 GPU 是两条异步时间线；五类瓶颈 + 显存四类；优化是把瓶颈从一类推到另一类 | CPU 每算子固定成本 10～30 µs；$$T \ge \max(\text{字节}/\text{带宽}, \text{FLOPs}/\text{算力})$$；A100 ridge point FP32 ≈ 10、BF16 ≈ 156 FLOP/Byte；融合 5N → 2N 访存；案例 166 → 2207 samples/s（13 倍） |
| [第九篇：分布式 PyTorch](/pytorch-distributed-training.html) | 一张卡放不下或跑不完时，如何切分状态并让通信与计算重叠？ | 五类状态各做一个决定（复制 / 分片），每个决定对应一种集合通信原语与一个时机 | ring all_reduce 每 rank 收发 $$2(N-1)/N \cdot n$$ → 2n；DDP 2P、FSDP 3P；每 rank 静态显存 16P → 16P/N；TP 每层 4 次 all_reduce、只在节点内；PP 气泡 $$(K-1)/(M+K-1)$$ |
| [第十篇：PyTorch 的工程体系](/pytorch-engineering-system.html) | 它怎么做到一直正确、一直可用？ | 一次改动过七关，守住正确性、性能、兼容性；框架工程是在组合爆炸下求可行 | 2000+ 算子 × 约 15 种 dtype × 设备 × 布局 × 模式 → 几十万测试实例；五种 oracle；`pull` 层约两小时；小版本每三到四个月、cut 距发布约 6 周；弃用保留至少一个小版本（通常两个） |

### 1. 本文的章节安排

| 章 | 内容 |
|---|---|
| 二 | 逐篇回顾：核心问题、结论、必记、常见误解 |
| 三 | 贯穿十篇的五条线：Tensor 元数据向下决定访存、Autograd 是一个 Key、固定成本与"少而大"、16 B/参数这笔账、契约与实现分离 |
| 四 | 常见误区表 |
| 五 | 通关自测：A 判断与计算 10 题、B 跨篇综合 5 题、C 面试题 7 题、D 掌握判据 |
| 六 | 下一步 |

## 二、逐篇回顾

### 1. 第一篇：PyTorch 整体介绍

**核心问题**：PyTorch 如何把 Python 中表达的张量计算，转化为可以自动求导、跨设备执行、编译优化和分布式协作的运行时系统？执行 `torch.add(x, y)` 时内部发生了什么？

**结论**：本篇用三张地图建立全局。静态地图把系统分成六个职责层——用户代码、编程模型、图与编译、算子运行时、设备与通信、Kernel 与硬件；Eager 下每个算子从编程模型直接进入算子运行时，`torch.compile` 先经过图与编译层再落到最底层。动态地图追一次 `torch.add`：Python API → Python Binding → Operator Schema → Dispatcher → ATen Operator → Kernel → Hardware；以两个 `requires_grad=True` 的 CUDA Tensor 为例，分发要经过两轮查表——合成 DispatchKeySet `[Autograd, CUDA]`，先命中 Autograd 记录 `AddBackward0`，去掉自己后再次分发到 CUDA Kernel。代码地图按库分四层：`torch/`（Python，编译栈主体也在这里）→ `torch/csrc/`（绑定、Autograd 引擎、c10d）→ `aten/src/ATen/`（Dispatcher 与算子）→ `c10/`（TensorImpl、Device、Allocator），每层只依赖下面的层；Autograd 引擎住在 `torch/csrc` 却把 Kernel 注册进 ATen 的表，所以调用路径在两层间往返一次。架构演进分五个阶段：0.1.x（2016 年 9 月起）的动态图、1.0（2018 年 12 月）的 API 稳定、1.x 的 C++ 运行时与分布式、1.8—1.13 的图与编译基础设施、2.0（2023 年 3 月 15 日）的 `torch.compile`。

**必记**：

- 职责六层与源码四层不重合：Tensor 元数据是用户最先接触的抽象，源码上却在最底层的 `c10/`。
- 编译后的动态库：`libc10`、`libtorch_cpu` / `libtorch_cuda`、`libtorch_python`；第六篇的扩展链接的正是它们，ABI 问题由此而来。
- 四个边界：Python 与 C++、通用抽象与后端实现、灵活性与可分析性、可移植性与性能特化——后面各篇的取舍都能归到其一。
- 新硬件接入：注册设备类型（PrivateUse1）与 `DeviceGuardImpl`、Allocator、在该 Key 上注册 Kernel、提供 stream / event 运行时 API；算子抽象与 Autograd 不用改。

**常见误解**："PyTorch 是一个 Python 库"——Python 只是表达层和控制层，运行时与抽象在 C++，设备执行在 CUDA。另一个："c10 只是设备抽象和分配器"——它是 Tensor 定义所在，`TensorImpl` / `StorageImpl` 都在这里。

### 2. 第二篇：Tensor 与内存布局

**核心问题**：Tensor 到底是什么？为什么 `transpose()` 不复制、`view()` 有时报错、`reshape()` 有时零拷贝、同 shape 的 Tensor 性能可能完全不同？

**结论**：Tensor 是六样东西的组合：数据（`StorageImpl` 里可共享的字节缓冲）、形状（`sizes`）、布局（`strides` + `storage_offset`）、类型、设备与生命周期（引用计数）。Python 的 `torch.Tensor` 是句柄，指向 C++ `TensorImpl`，后者持有 `Storage`；多个 `TensorImpl` 可指向同一 `StorageImpl`，这是 view 的实现基础。stride 表示"沿某维加 1，存储位置前进多少个元素"，`offset(i, j) = storage_offset + i × stride[0] + j × stride[1]`。`transpose` / `permute` / 切片只改元数据；`view` 要求新形状能在现有内存排列上直接解释，转置后往往做不到；`reshape` 是"能 view 就 view，否则先 `contiguous()` 复制再 view"。dtype 不只是位宽：fp16 指数 5 位、尾数 10 位，最大 65504，易溢出；bf16 指数 8 位与 fp32 相同、尾数只有 7 位，位域是 fp32 的高 16 位，但转换按最近舍入（1.005 → 1.0078125），不是直接砍掉。`device` 是每个 Tensor 自己的属性，`model.to("cuda")` 只搬注册的参数与 buffer。广播靠 `expand` 产生 stride 为 0 的视图，物理上不多占一个字节。显存问题的两个根源：某个小 view 让整块 Storage 活着；`del` 后的块只回到 caching allocator，`memory_reserved` 不降，`empty_cache()` 才归还驱动。

**必记**：

- `[2, 3, 4]` 连续 strides `(12, 4, 1)`；`transpose(0, 2)` 后 `[4, 3, 2]`、`(1, 4, 12)`，不连续。
- `x[0]` 与 `x[:, 0]` 都是 view；前者连续，后者 stride 是原第一维的 stride，不连续；都让整个 Storage 活着。
- `.to()` 到相同 device 与 dtype 直接返回 `self`；跨设备或改 dtype 才复制。
- 1000000 个 float32 ≈ 4 MB，float16 / bfloat16 ≈ 2 MB。
- 分析任何 Tensor 操作先问：是否新 Storage、是否复制、是否换 device / dtype、是否改 stride、是否延长某块内存的生命周期。

**常见误解**："Tensor 是支持 GPU 的 NumPy 数组"——它还带布局、设备与生命周期，同 shape 的连续与非连续 Tensor 走不同的 Kernel 路径。另一个："fp16 与 bf16 都是 16 bit，可以互换"——范围与精度完全不同。

### 3. 第三篇：自动求导与动态计算图

**核心问题**：模型输出变化时，参数应沿什么方向、以多大幅度变化？PyTorch 如何把链式法则变成一次沿计算图执行的反向传播？

**结论**：Autograd 在前向执行每个需要梯度的算子时就地创建 `Node`（`grad_fn`），节点通过 `next_functions` 指向输入的 `grad_fn`，沿链走到头就是整张图；每次前向一张新图，所以 Python 控制流可以直接参与。`backward()` 从结果节点按拓扑序执行，每个 `Node.apply` 算一次 VJP——上游梯度乘局部 Jacobian，从不物化 Jacobian（一个 `[B, 4096] → [B, 4096]` 层每样本的 Jacobian 是 $$4096^2$$ 个数）；公式来自 `derivatives.yaml`。叶子的 `.grad` 用 `+=` 累加，所以每步前必须 `zero_grad()`。节点用 `SavedVariable` 保存反向需要的值并记 version，in-place 修改后 version 不一致就报 "modified by an inplace operation"；哪些算子保存哪些输入由 `derivatives.yaml` 决定，所以同一个 in-place 有时安全有时报错。保存一个 non-leaf 输出（如把 `loss` 存进 list）会通过 `grad_fn` → `SavedVariable` → 上游节点的链让整张图与所有激活活着。`detach()` 作用于一个 Tensor，`no_grad()` 是线程局部开关（产物仍可参与后续求导），`inference_mode()` 更激进——不维护 version counter、不分配 AutogradMeta，产物不能再作为需保存的输入进入 autograd（`x * c` 报错，`x + c` 仍可）。自定义 `autograd.Function` 用 `ctx.save_for_backward` 存值、`backward` 返回对每个输入的 VJP，`gradcheck` 用 float64 有限差分验证。

**必记**：

- `y = x * 2; z = y.sum()`：`z.grad_fn` 是 `SumBackward0`，`y.grad_fn` 是 `MulBackward0`，`x.grad_fn` 是 `None`、`x.is_leaf == True`，链尾是 `AccumulateGrad`。
- 只有 `requires_grad=True` 的 leaf 反向后填 `.grad`；non-leaf 的梯度算完即丢（`retain_grad()` 或 hook 才留）。
- `backward()` 默认 `retain_graph=False`，执行完就释放保存的中间值。
- `exp` 的反向需要它的输出（$$\partial e^x / \partial x = e^x$$），所以 `a = x.exp(); a.add_(1)` 会报错。
- 排查顺序：输入 `requires_grad`？中间结果有 `grad_fn`？loss 连到目标参数？被 `detach` / `no_grad` 截断？in-place？累积与清零？

**常见误解**："`requires_grad` 表示有没有梯度"——它是"要不要追踪"的开关。另一个："`model.eval()` 关闭了梯度"——它只切换 Module 行为，与 `no_grad()` / `inference_mode()` 是两回事。

### 4. 第四篇：nn.Module 与训练系统

**核心问题**：PyTorch 如何把模型结构、参数状态、数据管线和训练循环组织成一个可以保存、迁移、复用和扩展的系统？

**结论**：一切靠 `nn.Module` 的注册机制：`__setattr__` 拦截赋值，把 `Parameter` 登进 `_parameters`、子 Module 登进 `_modules`、`register_buffer` 登进 `_buffers`；只有被登记的对象才参与 `parameters()` 遍历、`state_dict()` 保存、`.to()` 递归迁移与 `train()` / `eval()` 切换——Python list 里的 Module 不会被登记，要用 `ModuleList`。`state_dict` 把状态表示为"限定名 → Tensor"的有序字典，`load_state_dict` 按名字对齐，`strict=False` 会把缺失与多余的键都只记录不报错；优化器有自己的 `state_dict`，两者一起才是 checkpoint；`torch.save` 是 pickle，`weights_only=True` 用受限 unpickler 防任意代码执行。Optimizer 持有 param groups 与 `state`：FP32 + Adam 下每个参数对应 param、grad、`exp_avg`、`exp_avg_sq` 四块各 4 B，合计 16 B，7B 模型静态就是 112 GB——第八、九篇所有显存账的基准。数据管线由 `Dataset`（取一条）、`Sampler`（出索引）、`DataLoader`（worker 进程、`collate`、`pin_memory` 线程、预取）组成；GPU 利用率低不一定是 Kernel 问题，可能是 DataLoader 没及时供给。混合精度里 autocast 在算子层按算子选 dtype，`GradScaler` 处理 fp16 缩放：`scaler.step()` 内部才 unscale，裁剪前要显式 `unscale_`。

**必记**：

- 16 B/参数 = 4（param）+ 4（grad）+ 4（`exp_avg`）+ 4（`exp_avg_sq`）；其中一半属于 Optimizer state。
- GradScaler：初始 scale 65536，溢出 ÷2，连续 2000 步正常 ×2；`update()` 必须在每步 `step()` 之后。
- `DataLoader(num_workers=4)` 的 worker 是进程（绕开 GIL），每个 epoch 重启除非 `persistent_workers`；`prefetch_factor` 默认 2，结果队列深度 = `num_workers × prefetch_factor`。
- `pin_memory=True` 在主进程的专用线程做，之后 `to("cuda", non_blocking=True)` 才是真正异步 DMA。
- 训练问题排查顺序：Module 注册 → Parameter 被发现 → Buffer 迁移 → 同一 device → DataLoader 供给 → loss 连到参数 → grad 清零与更新 → checkpoint 完整。

**常见误解**："`self.running_mean = t` 与 `register_buffer` 一样"——普通 Tensor 属性不进 `state_dict`、不随 `.to()` 迁移。另一个："`load_state_dict(strict=False)` 更安全"——它是"加载了 checkpoint 但效果像没训过"的最常见来源。

### 5. 第五篇：Dispatcher 与算子系统

**核心问题**：写下 `z = x + y` 时，谁决定这个操作对应哪个算子、位于哪个设备、是否需要 Autograd、最终调用哪个 Kernel？

**结论**：算子系统有两个时间轴。开发态三步：定义 Schema（`native_functions.yaml` 或 `torch.library`，含 overload 名、mutability 与 alias 标注）→ 注册实现到 DispatchKey（`dispatch` 字段或 `TORCH_LIBRARY_IMPL`）→ 编写实现；Codegen 横向生成 Binding、`at::` 入口、注册代码与 Autograd 函数。运行态三步：入口（`torch.add` → Binding → `at::add`）→ 分发 → 执行。两者交汇于 Operator Table——每个算子一个 `OperatorEntry`，按 DispatchKey 存实现。分发是位运算：各输入的 KeySet 做 OR，加上 TLS 的 include、减去 exclude，取最高优先级 Key 查表；包装 Key（Autograd、Autocast、Functionalize、Python）优先级高于后端 Key，做完自己的事后把自己排除再次分发，所以一次 `add` 两次经过 Dispatcher；`requires_grad` 与 `no_grad()` 都不改 KeySet——普通 Tensor 的 KeySet 总含 Autograd Key，`no_grad()` 只翻转 GradMode 标志让包装层不记录；`inference_mode()` 才把 Autograd Key 加入 TLS excluded、真正跳过那一跳。实现内部有五种模式：TensorIterator 路径（逐元素 / 归约，Kernel 只写 `a + alpha * b`）、直接 Kernel、厂商库（cuBLAS / cuDNN）、Composite（组合其他 `at::` 算子并重新进入 Dispatcher）、Meta（只推断元数据）。TensorIterator 消费第二篇的 shape / stride / dtype，不是所有算子的必经之路。

**必记**：

- `torch.add` 是 Python 绑定入口，`at::add` 是 Codegen 生成的 C++ 入口，`at::native::add`（`add_cpu` 等）是被注册的实现主体：开发者写第三个，用户调第一个。
- `add.Tensor`、`add.Scalar`、`add.out` 是同名不同 Schema 的 overload，各是独立的 `OperatorEntry`。
- 带 CUDA 与 AutogradCUDA 且开 autocast：AutocastCUDA → AutogradCUDA → CUDA，每层用 `ExcludeDispatchKeyGuard` 去掉自己。
- 反向来自 `derivatives.yaml`：Codegen 据此生成 `VariableType` 包装 Kernel 注册到 Autograd Key；Schema、后端实现、导数公式是三处声明。
- 源码阅读顺序：yaml 找 Schema 与 dispatch → `ATen/native/` 找函数主体 → 判断五种模式 → `derivatives.yaml` 找反向 → Meta 验证 shape → Profiler 看实际 launch。

**常见误解**："Dispatcher 是按 device 的 if/else"——它是多维、可多次的运行时分发系统，Key 由元数据与执行上下文合成。另一个："Schema、Dispatcher、ATen、TensorIterator 是一条直线"——它们分属开发态与运行态两个维度。

### 6. 第六篇：C++ 扩展与自定义算子

**核心问题**：自己写一个算子（`scale_shift(x, alpha, beta) = alpha * x + beta`），把它接入 PyTorch 的算子系统，要做哪几件事、用什么工具、按什么顺序练？

**结论**：三步、两种接入、四个阶段。三步与第五篇相同：定义 Schema → 注册到 DispatchKey（CPU / CUDA / Autograd / Meta 各一份）→ 编写实现；原生算子靠 Codegen 生成粘合代码，自定义算子三步都要自己做。两种接入：Python 的 `torch.library.define / impl / register_autograd / register_fake`，C++ 的 `TORCH_LIBRARY / TORCH_LIBRARY_IMPL`，写进同一张 Operator Table，`torch.ops.myops.scale_shift` 按名字取回；pybind11 暴露的是普通函数，要成为算子必须走 `TORCH_LIBRARY`。四个阶段：纯 Python 建立契约 → C++ CPU（`AT_DISPATCH` 把运行时 dtype 桥接到编译期模板，`cpp_extension.load` 即时编译）→ CUDA（`CUDAGuard` 切到输入所在设备、用当前 stream、launch 后检查）→ Autograd 与 Meta（`register_fake` 供 FakeTensor 与 `torch.compile` 使用）。实现层必须处理四件事：device、dtype、stride（`contiguous()` 用一次拷贝换 Kernel 简单，TensorIterator 用地址计算换零拷贝）、生命周期（`at::Tensor` 是句柄，`data_ptr` 只在 Tensor 存活期间有效）。读懂 Kernel 需要的 CUDA 最小集：`blockIdx.x * blockDim.x + threadIdx.x` 是全局线程号，warp 内 32 个线程锁步执行，访存按 32 B 扇区合并。验证靠 `opcheck`（Schema、Autograd 注册、FakeTensor 元数据、AOT 可追踪——不含数值梯度）与 `gradcheck`（float64），Benchmark 对照原生 `2.0 * x + 1.0` 的两个 Kernel。

**必记**：

- `n = 1024`、每 block 256 线程 → `blocks = (1024 + 255) / 256 = 4`；`threads` 取 128 到 1024 之间、32 的倍数，256 最常见；Kernel 里必须有 `if (i < n)`。
- 一个 warp 32 线程各读 4 B：连续访问 128 B 正好一段、1 次事务；跨步 2 访问跨两段、2 次事务、带宽利用 50%；极端 32 个线程落 32 段，事务数是合并访问的 32 倍。
- `AT_DISPATCH_FLOATING_TYPES_AND_HALF` 不含 bf16，传 bf16 报 "not implemented for BFloat16"，要用 `..._AND2(kHalf, kBFloat16, ...)`。
- 错误定位原则：编译期看头文件与类型；链接期看 ABI 与库版本；加载期看注册；运行期 `NotImplementedError` 看 DispatchKey 槽位。
- 一个算子完成的七条标准：Schema 与 mutability、后端实现、Autograd、Fake / Meta、`opcheck` + `gradcheck`、Benchmark 证明有价值、构建与 ABI 可复现。

**常见误解**："用 pybind11 导出一个函数就是自定义算子"——那只是普通函数，不进 Operator Table，Autograd、Meta、`torch.compile` 都看不见它。另一个："在我机器上能跑就能交付"——PyTorch 版本、CUDA 版本、C++ ABI、编译器、GPU 架构任一不一致都可能加载失败或静默崩溃。

### 7. 第七篇：编译执行与图优化

**核心问题**：`torch.compile` 到底做了什么？它为什么不是"把 Python 翻译成 CUDA"？

**结论**：Eager 无法跨算子优化的原因是结构性的——Dispatcher 每次只看到一个算子；要跨算子优化必须有图。`torch.compile` 的默认路径是经典的三段：前端 TorchDynamo 在 CPython 字节码层做符号求值，用 FakeTensor 推断元数据，把 Tensor 操作记进 FX Graph、把依赖元数据（shape、dtype）的分支特化并记成 Guard、把依赖 Tensor 值或不支持的操作切成 graph break；中端 AOTAutograd 把 torch 级图降到 ATen 级，追踪 autograd 得到反向图，functionalize 去掉 in-place，用 min-cut 切分器决定保存什么、重算什么，包成一个 `autograd.Function`；后端 TorchInductor 把 ATen 图变成循环级 IR，做融合与内存规划，生成 Triton（GPU）或 C++（CPU）源码。三段共享同一种数据结构 `torch.fx.Graph`，只是节点词汇逐段下降。运行时是第二个维度：`torch.compile(f)` 只是包装，流水线在第一次调用或 Guard 失败时按需运行；Graph Break 决定图有多大，Guard 决定何时重编，Dynamic Shape 用符号 `s0` 代替常量，缓存条目上限 8、超限退回 Eager。贯穿全文的 `f`：第一次冷编译，`if` 用 shape 特化掉、Guard 记 `size[0] == 128`；第二次同 shape 命中；第三次 batch 256 触发动态化，Guard 变成 `s0 > 64`；第四次 batch 32 走 tanh，第二份产物。原始的 `if` 最终变成一条 Guard 和两份编译产物。

**必记**：

- 热路径对照 Eager：Python 进入 C++ 3 次 → 1 次；分发 3×2 次 → 1 次（`extern_kernels.mm` 仍是 `torch.mm`，逐元素部分才绕开 Dispatcher）；前向 launch 3 → 2；中间 Tensor 2 → 0；Autograd 节点 3 个 → 1 个 `CompiledFunctionBackward`。
- `backend=` 指"Dynamo 之后的一切"：`"eager"` 只捕获、`"aot_eager"` 加中端、`"inductor"` 全走；`mode="reduce-overhead"` 加 CUDA Graphs，`mode="max-autotune"` 用 Triton 矩阵乘模板。
- `torch.compile` 是带回退的 JIT，`torch.export` 是无回退的 AOT，分歧只在对 graph break 的态度；AOTInductor 在 export 之上编成共享库。
- Inductor 默认不把 pointwise 融进 cuBLAS 的 `mm`（库调用是黑盒），也不消除 launch 本身。
- 排查顺序：`torch._dynamo.explain` → `TORCH_LOGS="graph_breaks"` → `"recompiles"` → `backend="eager"` / `"aot_eager"` → `"aot_graphs"` → `"output_code"` → Profiler。

**常见误解**："`torch.compile` 就是先 `symbolic_trace` 再优化"——`symbolic_trace` 对 Proxy 做 `bool()` 直接报错，Dynamo 在字节码层重新实现了捕获。另一个："Guard 失败与 graph break 是一回事"——前者是调用时的失效、决定是否重编，后者是捕获时的边界、决定图有多大。

### 8. 第八篇：性能优化与调试

**核心问题**：如何判断一个 PyTorch 程序慢，以及如何定位它为什么慢？

**结论**：起点是一个事实——CPU 与 GPU 是两条通过队列（CUDA Stream）连接的异步时间线，step 时间由较长的那条决定。时间维度上慢几乎总是五类之一：CPU 侧的 Python-bound 与 Launch-bound、GPU 侧的 Memory-bound 与 Compute-bound、两侧之间的 Sync-bound。每个算子在 CPU 上有 10～30 µs 的固定成本，单个 Kernel 的 GPU 时间低于它就是 launch-bound；一个 Kernel 的下界是 $$T \ge \max(\text{数据量}/\text{带宽}, \text{运算量}/\text{算力})$$，AI 低于 ridge point 是 memory-bound——逐元素算子 AI 在 0.1 量级，永远 memory-bound，唯一出路是减少访存；大矩阵乘是 compute-bound，出路是 Tensor Core 或减少计算量。空间维度分四类：真的不够、碎片（`reserved` ≫ `allocated`）、泄漏（`allocated` 单调涨，几乎总是持有了带 `grad_fn` 的 Tensor）、峰值（反向开始时、Attention 的 score 矩阵）。测量的纪律：计时必须同步或用 CUDA Event，warmup 后报告中位数，写明口径，正确性测试先于性能改动，一次只改一件事。案例把一个 12 层 Transformer block 从 166 提到 2207 samples/s（13 倍），没有一项是优化某个 Kernel，全部是改变瓶颈类别：先消灭 launch-bound，再对 compute-bound 用 Tensor Core，再对 memory-bound 做融合。

**必记**：

- A100：FP32 19.5 TFLOPS、BF16 Tensor Core 312 TFLOPS、带宽 2 TB/s → ridge point ≈ 10 与 156 FLOP/Byte；bf16 算力是 fp32 的 16 倍。
- `[128, 64]` fp32 的 `add` Kernel 约 3～5 µs，CPU 提交它要几倍时间；launch-bound 阈值约 10～20 µs。
- 融合 `add` + `relu`（两个 N 元输入）：分开 5N 次访存，融合 3N；bias 广播时约 4N → 2N；`4096³` bf16 矩阵乘 AI ≈ 1370 ≫ 156。
- 混合精度静态显存不变：原生 autocast 下参数 leaf 与 `.grad` 仍是 fp32（4 + 4 + 4 + 4）；Megatron 式 bf16 主流程是 bf16 参数 2 + bf16 梯度 2 + fp32 主参数 4 + m 4 + v 4——都是 16 B/参数，收益全在激活值。
- 案例：48.2 ms / 3.1 GB → batch 64：118 ms（吞吐 ×3.3）→ bf16：51 ms → compile：38 ms（47 s 冷编译，Kernel 2100 → 640）→ SDPA：29 ms / 8.4 GB（`att` 每层 268 MB 不再物化）；checkpoint 换 batch 128 反而 1882 < 2207，未采用。
- 低精度对 launch-bound 无效甚至变慢（autocast 插入 cast Kernel）——先解决 CPU 侧问题再上低精度。

**常见误解**："GPU 利用率低就去优化 Kernel"——利用率低说明 GPU 在等 CPU 或同步点，此时优化任何 Kernel 都无效。另一个："`reserved` 高是泄漏"——那是 caching allocator 的缓存；`empty_cache()` 不解决碎片，只减少 `reserved`。

### 9. 第九篇：分布式 PyTorch

**核心问题**：当一张卡放不下模型或跑不完数据时，PyTorch 如何把计算和状态切分到多个设备，并让通信与计算重叠？

**结论**：每种并行策略都是对五类状态——数据、参数、梯度、优化器状态、激活值——各做一个决定：复制（显存 N 份，需 all_reduce 同步）还是分片（显存 1/N，用到时 all_gather 凑齐、用完 reduce_scatter 分发）；每个决定同时决定显存占用、通信原语与通信时机。成本用 α + β 模型：小消息由延迟 α 主导，所以要合并成大消息（DDP 梯度桶、FSDP 分片单元不能太小）；ring all_reduce 每 rank 收发 $$2(N-1)/N \cdot n$$ 字节，N 大时趋近 2n、与进程数无关，而延迟项 $$2(N-1)\alpha$$ 随 N 线性增长。DDP 只分数据、其余全复制，`Reducer` 按反向顺序分桶，桶就绪就在 NCCL stream 上 all_reduce，与更早层的反向重叠——DDP 不省显存。ZeRO 三级逐个分片优化器状态、梯度、参数：ZeRO-1/2 只是 all_reduce = reduce_scatter + all_gather 恒等式的应用，通信仍 2P；ZeRO-3（FSDP）参数也分片，前向 all_gather、反向再 all_gather 加 reduce_scatter，通信按元素数 3P（同 dtype 时比 DDP 多 50%；bf16 参数 + fp32 梯度的常见配置下按字节与 DDP 相同，都是 8 B/参数），换来 16P/N 显存；HSDP 节点内分片、节点间复制，跨 IB 流量降到 2P/N_shard。TP 切层内，通信激活而非参数，每层 4 次 all_reduce 在关键路径上无法重叠，实践上限在 NVLink 域内（常见一机 8 卡）；SP 把边界上复制的激活也切掉。PP 按层切，气泡 $$(K-1)/(M+K-1)$$，1F1B 让每 stage 激活显存 ∝ K 而非 M。CP 切序列、只有 attention 通信；EP 切 expert、all_to_all 路由。加卡不线性的四组原因：通信时间、同步等待、计算效率（per-rank batch 变小回到 launch-bound）、算法效率（超过临界 batch）。

**必记**：

- 每 rank 静态显存：DDP 16P；ZeRO-1 4P + 12P/N；ZeRO-2 2P + 14P/N；ZeRO-3 / FSDP 16P/N（N=8：16P → 5.5P → 3.75P → 2P）；通信 2P、2P、2P、3P。
- 带宽层级：NVLink（H100）约 900 GB/s 双向、all_reduce 总线带宽 300～450 GB/s；PCIe Gen5 x16 约 64 GB/s；IB NDR 每卡约 50 GB/s——节点内外差近一个数量级。
- 7B 模型 8 卡 FSDP：all_gather 2 × 14 GB（bf16）+ reduce_scatter 28 GB（fp32）= 56 GB/rank/step（ring 实发约 49 GB），NVLink 约 190 ms、IB 约 1.1 s——与 DDP 对 fp32 梯度 all_reduce 的 56 GB 相同。
- GPipe 气泡 M=4、K=4 时 43%，M=32 时 9%；决策顺序里 PP 的 micro-batch 数 ≥ 4K。
- 案例：8 卡 DDP 95%；7B 8 卡 FSDP + checkpointing 通信全部隐藏；32 卡 FSDP 掉到 62%（56 GB 不随卡数减少，计算缩到 1/4）；HSDP 回到 94%，代价是显存不随节点数下降。
- `dist.all_reduce` 同步版返回即可读；`async_op=True` 必须 `work.wait()`，且 `wait()` 只让当前 stream 等，不阻塞 CPU。

**常见误解**："FSDP 比 DDP 省显存也省通信"——它不省通信：按元素数 3P 对 2P，按字节在常见 dtype 配置下持平；省的是 16P → 16P/N 的显存。另一个："加卡就该线性加速"——总 batch 不变时 per-rank 计算随 N 缩小，而带宽项通信量不随 N 减少，两条曲线会交叉。

### 10. 第十篇：PyTorch 的工程体系

**核心问题**：前九篇描述的是一个已经正确运行的系统——它是怎么做到一直正确、一直可用的？

**结论**：一次改动从写下到进入用户生产环境要过七关：本地构建能跑、结果正确、没有变慢、审查与 CI 合入、随版本发布、用户升级不坏、使用者跟随演进；前六关是维护者视角，第七关是使用者视角。七关守三件事：正确性（主要由 ②）、性能（③）、兼容性（⑥），构建是验证的前提，合入与发布把验证变成强制的、自动的流程。框架与应用的区别是组合爆炸：2000+ 算子 × 约 15 种 dtype × 设备 × shape × layout × 执行模式，几十万个测试点。这塑造了三件事：数据驱动的 OpInfo（一处声明 `op_db`，几十个模板做笛卡尔积，添加一个算子的测试就是加一个 `OpInfo`）；分层的 CI（`pull` 每 PR 约两小时，`trunk` 合入后，`periodic` 每晚，`inductor` 专项；目标确定按改动文件排序测试；flaky 自动开 disable issue 隔离而不是重跑到过）；按稳定程度分级的接口面（Python API 有弃用周期，算子 Schema 有自动化 BC/FC 检查，C++ API 无 BC 保证但有 `torch/csrc/stable/` 稳定子集与 PrivateUse1，序列化有 `_version` 机制）。"对"由五种 oracle 定义：参考实现、数学恒等式（`gradcheck`）、跨后端一致、跨模式一致、元数据一致——新算子至少要有前两种之一作为绝对标准，否则跨后端一致只能证明两个实现错得一样。性能守门用指令数（`collect_callgrind`，可检测 0.1% 变化）守 CPU 开销，TorchBench 与编译器看板守端到端，回归的定义是曲线上的拐点。第六篇的 `myops` 走完七关表明：复用测试基础设施、契约与实现分离、性能基线入库，树外扩展也能低成本获得同样保障。

**必记**：

- 首次完整 CUDA 构建在 32 核机器上约 1～2 小时；改一个 `.cu` 几分钟，改 `c10/` 头文件可能触发半数文件重编；改 Schema 要重跑 Codegen、几乎整个 `aten` 重编。
- 小版本每三到四个月，cut 出 release 分支距发布约 6 周，之后只接受 cherry-pick；CUDA 通常同时支持两到三个版本。
- wheel 矩阵：Python 版本 × 加速后端（CPU / CUDA / ROCm / XPU）× 平台，每个维度都是 ABI 的一部分；Linux 官方 wheel 2.6 起部分、2.7 起全部切到 cxx11 ABI，扩展读 `torch._C._GLIBCXX_USE_CXX11_ABI` 跟随。
- 弃用：保留至少一个小版本（通常两个）；`torch.symeig` 1.9 弃用、1.13 移除；`weights_only` 默认值 2.4 警告、2.6 切换。
- Schema BC/FC：新增算子或 overload、末尾加带默认值的参数 ✓BC ✗FC；删除 / 重命名 / 改类型 / 改默认值都禁止，要变化就新增 overload；`torch.div` 1.6 的语义变更走 upgrader。
- 使用者节奏：生产 pin 到具体版本连同 CUDA、驱动、扩展一起进镜像；每个小版本评估、每两个小版本升级一次；`-W error::FutureWarning` 让弃用在 CI 里变成错误。

**常见误解**："flaky 测试让作者重跑到过就行"——几十万测试里的 flaky 会让每个 PR 都红、掩盖真回归，必须靠自动隔离流程。另一个："`FutureWarning` 是噪音"——它是倒计时，删除通常在两个小版本后到来。

## 三、贯穿全系列的几条线

### 1. Tensor 元数据一路向下决定访存

第二篇建立的六个字段——尤其是 stride 与 dtype——不只是解释"view 为什么不复制"，它们决定了下面每一层的行为。第五篇的 TensorIterator 消费的正是 shape / stride / storage_offset / dtype：构造迭代空间、检测连续布局、划分并行块，Kernel 只表达对一个元素做什么。第六篇把同一件事放到自定义 Kernel 里：`data_ptr<T>()` 返回的是 `storage_offset` 之后的起始地址，对非连续 Tensor 直接一维遍历会得到错误结果，所以要么 `contiguous()` 用一次拷贝换 Kernel 简单、要么 TensorIterator 用地址计算换零拷贝；而 warp 内 32 个线程访问连续地址才能把 32 B 扇区用满，跨步 2 就只剩 50% 带宽利用。第八篇给出后果的度量：逐元素算子几乎总是 memory-bound，时间由字节数除以带宽决定；单个转置 view 参与运算并不慢（TensorIterator 沿物理布局遍历），两个输入布局正交时才掉带宽；bf16 把数据量减半，所以同一个 Kernel 时间大致减半。第二篇的 dtype 位域（fp16 5/10、bf16 8/7）在第四篇成为 GradScaler 存在的理由，在第八篇成为 Tensor Core 16 倍算力与"主参数必须保留 fp32 副本"的依据。

### 2. Autograd 是一个 DispatchKey

第一篇的一句话"Autograd 是一个 DispatchKey"在后面五篇里反复兑现。第三篇从用户视角看它：每个算子在前向时记录 `grad_fn`、用 `SavedVariable` 保存反向所需的值。第五篇从运行态看它：Autograd 是 Operator Table 上优先级高于后端 Key 的包装 Key，`VariableType` 里的包装 Kernel 由 Codegen 从 `derivatives.yaml` 生成，做完记录后 `AutoDispatchBelowAutograd` 把自己排除、再次分发到 CUDA——一次 `add` 两次经过 Dispatcher；Autocast、Functionalize、Python 子类拦截都是同一条链上的 Key。第六篇要求自定义算子自己往这个槽位填东西：`register_autograd` 或 Autograd Key 上的 `torch::autograd::Function`，`opcheck` 检查它的注册是否合法，`gradcheck` 用有限差分检查数值。第七篇的 AOTAutograd 是把这层"提前做"：用 FakeTensor 追踪 autograd 得到整张反向图，运行时只剩一个 `CompiledFunctionBackward` 节点——编译改变了节点内部的执行方式，没有改变 Autograd 图的拓扑与 Optimizer 看到的接口。第九篇的 DDP `Reducer` 与 FSDP 的反向 hook 都挂在第三篇的 autograd hook 上，梯度就绪的那一刻就是通信开始的时刻。第十篇则用 `gradcheck` 作为五种 oracle 之一守住每个反向节点。

### 3. 固定成本与"少而大"

第五篇拆开的入口 → 分发 → 执行链路，在第八篇被量化为每个算子 10～30 µs 的 CPU 固定成本，与 Tensor 大小无关；单个 Kernel 的 GPU 时间低于它就是 launch-bound，batch=1 的推理几乎必然如此。这个结构在第七篇给了编译器存在的理由——Dispatcher 每次只看到一个算子，无法跨算子优化；Inductor 把 `add` + `relu` 融成一个 Triton Kernel，`f` 的热路径从 3 次分发、3 次 launch 变成 0 次分发、2 次 launch，中间 Tensor 从 2 个变成 0 个。它还不够时，第七篇的 `mode="reduce-overhead"` 与第八篇的 CUDA Graphs 把一串 launch 录成一个图整体重放，优化的是 launch 方式而不是 Kernel 本身；`fused=True` 的优化器把 100 个参数 Tensor × Adam 约 10 个算子的 1000 次 launch 合成一个 Kernel。第九篇把同一个结构搬到通信上：α + β 模型里小消息由延迟 α 主导，"消息越小越浪费"与 Kernel launch 的固定成本是同一个结构——这就是 DDP 梯度桶、FSDP 分片单元不能太小、NCCL 按消息大小切换算法与协议的共同理由。第十篇最后用指令数守住这条线：`collect_callgrind` 完全确定、能检测 0.1% 的变化，PyTorch 用它盯住 Dispatcher 与 Python 绑定层的开销。

### 4. 16 B/参数这笔账

同一个数字在三篇里各算一次。第四篇在 Optimizer 一章第一次写下它：FP32 + Adam 下每个参数对应 param、grad、`exp_avg`、`exp_avg_sq` 四块各 4 B，7B 模型 112 GB，其中一半是 Optimizer state，并预告它是多卡训练最先被切分的状态。第八篇把它放进显存的构成——前三项是静态的、与 batch 无关，激活值随 batch 线性增长；并指出混合精度下这个数字不变（原生 autocast：参数与 `.grad` 都还是 fp32；bf16 主流程：bf16 参数 2 + bf16 梯度 2 + fp32 主参数 4 + m 4 + v 4），收益全在激活值上；案例里 38M 参数的小模型静态部分微不足道，checkpointing 换 batch 反而不值得。第九篇以它为所有显存账的基准：DDP 每卡 16P，ZeRO-1 4P + 12P/N，ZeRO-2 2P + 14P/N，FSDP 16P/N；7B 模型 8 卡 FSDP 静态 14 GB，激活 143 GB 远超显存，第八篇"不值得"的 checkpointing 在这里成为必需（激活降到约 13 GB，代价 +33% 计算）；HSDP 的代价正是静态显存从 3.5 GB 回到 14 GB。第十篇的升级 playbook 第 5 步"旧 checkpoint 在新版本加载并续训"，加载的就是这 16 B 里属于 `state_dict` 与优化器 `state_dict` 的部分。

### 5. 契约与实现分离

第五篇的 Operator Schema 是这条线的起点：抽象算子的定义与它在各后端的实现分开声明，alias 与 mutability 标注是 Autograd 版本检查、编译器安全重排与内存复用的共同基础。第六篇要求自定义算子从"纯 Python 实现建立契约"开始再逐层下沉，`opcheck` 检查 Schema 与实际行为、Fake 与真实实现是否一致。第七篇的编译栈完全依赖这份契约：Dynamo 与 AOTAutograd 用 FakeTensor 推断元数据，缺 `register_fake` 的自定义算子要么 graph break 要么报 "no fake impl"。第十篇把契约变成守门的对象：`op_db` 里每个 `OpInfo` 是一份声明、模板负责验证声明属实，`op_db` 同时是算子能力的权威清单与新后端衡量覆盖度的标尺；CI 把当前 Schema 与最近 nightly 的快照比较，任何 BC 破坏都失败；`torch._refs` 一份代码三种用途——测试 oracle、编译分解来源、新后端兜底。契约稳定、实现自由，是这个系统能同时支持 CPU / CUDA / 树外后端、Eager / compile 的原因。

| 概念 | 出现的篇 | 关系 |
|---|---|---|
| stride / contiguous / dtype | 二、五、六、八 | 二定义；五由 TensorIterator 消费；六决定 `contiguous()` 还是 TensorIterator、决定访存合并；八决定 memory-bound Kernel 的实际带宽与 bf16 的收益 |
| Autograd 作为 Key、`grad_fn`、`derivatives.yaml` | 一、三、五、六、七、九、十 | 一点名；三讲机制；五讲包装 Key 再分发；六自定义算子自己填；七 AOTAutograd 提前做成一个节点；九 DDP / FSDP 的 hook 挂在上面；十 `gradcheck` 守住 |
| 每算子固定成本、launch、α + β | 五、七、八、九、十 | 五给来源；八量化为 10～30 µs；七融合与 CUDA Graphs 是处方；九通信的小消息问题同一结构；十用指令数守门 |
| 16 B/参数 | 四、八、九 | 四第一次算出；八分静态与激活、混合精度不变；九作为所有分布式显存账的基准 |
| Schema / Operator Table / OpInfo / BC-FC | 五、六、七、十 | 五定义契约；六自定义算子补齐并 `opcheck`；七编译器靠 Fake 实现推 shape；十用 `op_db` 与 Schema 快照守门 |
| FakeTensor / Meta | 五、六、七、九、十 | 五 Meta 路径；六 `register_fake`；七 Dynamo 与 AOTAutograd 全程在 FakeTensor 上运行；九 FSDP 在 `meta` device 上构造再 `to_empty`；十元数据一致是五种 oracle 之一 |
| 两条异步时间线、Stream | 四、八、九 | 四 pinned memory + `non_blocking` 才真正异步；八 计时必须同步、五类瓶颈的根源；九 NCCL 通信是独立 stream 上的 Kernel，`Work.wait()` 是 stream 依赖 |
| checkpoint / `state_dict` | 四、九、十 | 四 模型与优化器两份 `state_dict`、`weights_only`；九 分布式 Checkpoint；十 `_version` 升级机制与 playbook 第 5 步 |

## 四、常见误区

| 误区 | 为什么错 | 正确的说法 | 出处 |
|---|---|---|---|
| PyTorch 是一个 Python 库 | Python 只是表达与控制层 | 三层分工：Python 表达和组织，C++ 运行时和抽象，CUDA 设备执行 | [第一篇](/pytorch-overall-introduction.html) |
| `reshape` 总是零拷贝 | 不连续时先 `contiguous()` 复制再 view | 能 view 就 view，否则复制；返回值是否共享内存不确定 | [第二篇](/pytorch-tensor-and-memory-layout.html) |
| `del x` 后显存就释放了 | 小 view 仍持有同一 Storage；释放的块只回缓存 | 用 `.clone()` / `.item()` 切断共享；`reserved` 不降是缓存不是泄漏 | [第二篇](/pytorch-tensor-and-memory-layout.html) |
| 把 `loss` 存进 list 只是保存一个数 | `loss` 通过 `grad_fn` 持有整张图与所有激活 | 用 `loss.item()` 或 `.detach()`；泄漏几乎总是持有了带 `grad_fn` 的 Tensor | [第三篇](/pytorch-autograd-and-dynamic-computation-graph.html) |
| `model.eval()` 关闭了梯度 | 它只切换 Module 行为（dropout、BN） | 评估要 `eval()` + `inference_mode()` 一起用 | [第四篇](/pytorch-module-and-training-system.html) |
| `self.layers = [nn.Linear(...) ...]` 会被 Module 管理 | list 不是 Module，`__setattr__` 不登记 | 用 `nn.ModuleList` / `nn.Sequential`，否则不进优化器、不进 `state_dict`、不迁移 | [第四篇](/pytorch-module-and-training-system.html) |
| Dispatcher 是按 device 的 if/else | Key 由元数据与 TLS 合成，包装 Key 再次分发 | 多维、可多次的运行时分发系统；一次 `add` 两次经过 Dispatcher | [第五篇](/pytorch-dispatcher-and-operator-system.html) |
| 自定义 Kernel 处理连续 float32 就够了 | dtype、device、stride、空 Tensor、生命周期、backward 都要处理 | `AT_DISPATCH` 展开 dtype、`CUDAGuard` 切设备、`contiguous()` 或 TensorIterator、`register_fake` | [第六篇](/pytorch-cpp-extension-and-custom-operators.html) |
| `torch.compile` 把 Python 翻译成 CUDA | 它只捕获可分析的 Tensor 计算，把不能确定的部分变成 Guard | 能确定的固化进代码，不能确定的变成运行时检查；捕获不了就 graph break 退回 Eager | [第七篇](/pytorch-compilation-and-graph-optimization.html) |
| 换 batch size 变慢是编译器 bug | 静态 shape 下每个新 shape 是一次 Guard 失败 + 重编译 | `dynamic=True` / `mark_dynamic` 让 shape 成为符号；缓存条目上限 8 | [第七篇](/pytorch-compilation-and-graph-optimization.html) |
| GPU 利用率低就优化 Kernel | 利用率低说明 GPU 在等 CPU 或同步点 | 先看 GPU 泳道形态，归到 launch / Python / sync-bound，解决后才能谈 Kernel | [第八篇](/pytorch-performance-optimization-and-debugging.html) |
| 低精度总能加速 | launch-bound 下 Kernel 数不变，autocast 还插入 cast Kernel | 先解决 CPU 侧问题，再上 bf16 | [第八篇](/pytorch-performance-optimization-and-debugging.html) |
| FSDP 比 DDP 省显存也省通信 | 参数分片后"用到时凑齐"多出一个 P | 通信 3P 对 2P，多 50%；省的是 16P → 16P/N | [第九篇](/pytorch-distributed-training.html) |
| 加卡就该线性加速 | 带宽项通信量不随 N 减少，per-rank 计算随 N 缩小 | 四组原因：通信、同步等待、计算效率、算法效率；藏不住就换策略 | [第九篇](/pytorch-distributed-training.html) |
| flaky 测试让作者重跑到过 | 几十万测试里的 flaky 会让所有人忽略红色 CI | 自动开 disable issue 隔离、定期重跑、连续通过后恢复 | [第十篇](/pytorch-engineering-system.html) |

## 五、通关自测

### A. 判断与计算（10 题）

1. 形状 `[3, 4, 5]` 的连续 Tensor，strides 是什么？`x.permute(2, 0, 1)` 之后 shape 与 strides 是什么、连续吗？

   <details markdown="1"><summary>答案</summary>

   strides `(20, 5, 1)`；permute 后 shape `[5, 3, 4]`、strides `(1, 20, 5)`，不连续——只重排了元数据，没复制数据；此时 `view(-1)` 会报错，`reshape(-1)` 会复制。

   </details>

2. `x = torch.zeros(4, 6); y = x[1:3, ::2]`：`y` 的 shape、strides、`storage_offset` 各是什么？

   <details markdown="1"><summary>答案</summary>

   `x` 的 strides 是 `(6, 1)`；`y` shape `[2, 3]`、strides `(6, 2)`、`storage_offset = 1 × 6 + 0 = 6`；共享 `x` 的 Storage，不连续。按 `offset = storage_offset + i × 6 + j × 2` 可定位每个元素。

   </details>

3. 一个 13B 参数模型用 Adam 训练，静态显存多少？改成 bf16 混合精度呢？用 ZeRO-1 切到 8 卡，每卡多少？

   <details markdown="1"><summary>答案</summary>

   $$13 \times 10^9 \times 16 = 208$$ GB；混合精度仍是 16 B/参数（2 + 2 + 4 + 4 + 4），208 GB 不变；ZeRO-1 每卡 $$4P + 12P/8 = 5.5P$$，$$13 \times 10^9 \times 5.5 = 71.5$$ GB——一张 80 GB 卡刚放得下静态部分，激活还没算。

   </details>

4. `y = x.detach(); y.add_(1)`——`x` 的值变了吗？`y.requires_grad` 是什么？

   <details markdown="1"><summary>答案</summary>

   变了：`detach()` 返回共享数据、只切断与图连接的新 Tensor，in-place 写 `y` 就是写同一块 Storage；`y.requires_grad == False`、`y.grad_fn is None`。若 `x` 被某个节点 `SavedVariable` 保存，这次修改还会让 `x` 的反向报 version 不一致。

   </details>

5. 两个 CUDA Tensor 相加，都 `requires_grad=True`，但在 `torch.no_grad()` 里，且开了 autocast：Dispatcher 依次命中哪些 Key？如果两者都不需要梯度、也不在 `no_grad` 里呢？

   <details markdown="1"><summary>答案</summary>

   两种情形 KeySet 都是 `{AutogradCUDA, AutocastCUDA, ADInplaceOrView, CUDA}`——`requires_grad` 与 `no_grad` 都不改 KeySet。第一种：AutogradCUDA（读到 GradMode 关闭，不记录 `grad_fn`）→ AutocastCUDA（`add` 属 fallthrough 类，不转精度）→ CUDA。第二种：AutogradCUDA（发现没有输入 `requires_grad`，不记录）→ CUDA。只有 `inference_mode()` 会在合成 KeySet 时把 Autograd 一族剔掉，一次分发直接到后端。

   </details>

6. 对 1 亿个 fp32 元素做 `add` 再 `relu`，分开与融合各访存多少字节？按 2 TB/s 带宽各需多久？

   <details markdown="1"><summary>答案</summary>

   分开 5N 次（add 读 2N 写 N，relu 读 N 写 N）：$$5 \times 10^8 \times 4 = 2$$ GB，约 1 ms；融合 3N 次（读两个输入、写一个输出，中间结果不落显存）：1.2 GB，约 0.6 ms。两个算子 AI 都在 0.1 量级，远低于 ridge point，时间按访存量线性下降——这就是 Inductor 融合省下的东西。

   </details>

7. `[2048, 2048] × [2048, 2048]` 的 bf16 矩阵乘，AI 是多少？在 A100 上是 memory-bound 还是 compute-bound？

   <details markdown="1"><summary>答案</summary>

   运算量 $$2 \times 2048^3 \approx 17.2$$ GFLOPs；数据量 $$3 \times 2048^2 \times 2 \approx 25$$ MB；AI ≈ 680 FLOP/Byte，高于 BF16 ridge point 156 → compute-bound，处方是 Tensor Core 或减少计算量，融合帮不上。

   </details>

8. 16 卡 ring all_reduce 一份 1 GB 的梯度，每卡收发多少字节？要多少步？卡数增到 128 时呢？

   <details markdown="1"><summary>答案</summary>

   每卡收发 $$2 \times 15/16 \times 1 \approx 1.875$$ GB，$$2(N-1) = 30$$ 步；128 卡时 $$2 \times 127/128 \approx 1.98$$ GB、254 步——带宽项趋近 2n 与卡数无关，延迟项随卡数线性涨，这是 NCCL 在大规模时切到 Tree 的原因。

   </details>

9. 流水线并行 K=8 个 stage，M=8 个 micro-batch，气泡占多少？M=64 呢？要把气泡压到 5% 以下，M 至少多少？

   <details markdown="1"><summary>答案</summary>

   $$(K-1)/(M+K-1)$$：M=8 时 $$7/15 \approx 47\%$$；M=64 时 $$7/71 \approx 10\%$$；要 < 5% 需 $$M + 7 > 140$$，M ≥ 134。用 1F1B 才能放心加大 M——GPipe 的激活显存 ∝ M，1F1B ∝ K。

   </details>

10. 第七篇的 `f` 用 `torch.compile` 后依次以 batch 128、200、64 调用，各触发几次编译、Guard 各是什么？

    <details markdown="1"><summary>答案</summary>

    128：冷编译，静态 Guard `size[0] == 128`，relu 分支；200：Guard 失败，重编译为动态 shape，Guard `s0 > 64`，仍走 relu；64：`64 > 64` 为假，两个条目都不命中，第三次编译，tanh 分支，Guard `s0 <= 64`。共 3 次编译、3 个缓存条目（上限 8）。

    </details>

### B. 跨篇综合（5 题）

1. 训练循环里 `losses.append(loss)`，`memory_allocated()` 每步单调增长。用哪几篇解释原因、定位与修复？

   <details markdown="1"><summary>答案</summary>

   第三篇：`loss` 是 non-leaf，通过 `grad_fn` → `SavedVariable` → 上游节点持有整张图与所有激活，每步一张新图全活着；第二篇：Tensor 生命周期比变量名重要，是否延长了某块内存的生命周期；第八篇：`allocated` 单调增长归为泄漏，用 memory snapshot 找持有 `grad_fn` 的引用。修复：`loss.item()` 或 `loss.detach()`。

   </details>

2. 自定义算子只注册了 CPU 与 CUDA 实现，`torch.compile` 后报 "no fake impl"。为什么？缺的是哪个槽位？编译器为什么需要它？

   <details markdown="1"><summary>答案</summary>

   第五篇：Operator Table 按 Key 分槽位，Meta 路径只推断输出元数据；第六篇：自定义算子要自己 `register_fake`，`opcheck` 会检查 Fake 与真实实现的 shape / dtype 一致；第七篇：Dynamo 与 AOTAutograd 全程在 FakeTensor 上运行、真实数据只在最后一步被读取，没有 Fake 实现就无法推断输出形状，只能 graph break 或报错。

   </details>

3. 单卡饱和后用 DDP 扩到 8 卡、总 batch 不变，扩展效率只有 60%，Profiler 显示 GPU 泳道稀疏、CPU 忙。原因与处方？

   <details markdown="1"><summary>答案</summary>

   第九篇："为什么加卡不线性"的第三组原因——总 batch 不变时 per-rank batch 变成 1/8，Kernel 变小；第八篇：单个 Kernel 的 GPU 时间低于 10～30 µs 的 CPU 固定成本就是 launch-bound，处方是增大 per-rank batch、融合 / `torch.compile`、CUDA Graphs、fused 优化器；第四篇：也要排除 DataLoader 供给不足（GPU 大段空白、CPU 停在 DataLoader）。同时检查通信是否暴露（第九篇 DDP 最后一个桶无法重叠）。

   </details>

4. 一个手写 CUDA Kernel 在 `x.transpose(0, 1)` 的输入上结果错误；改成 `x.contiguous()` 后结果对了但变慢。解释两件事，并说另一个选项。

   <details markdown="1"><summary>答案</summary>

   第二篇：转置只改元数据，strides 不再是行主序，`data_ptr` 一维遍历读到错误位置；第六篇：`contiguous()` 用一次拷贝换 Kernel 简单，多出一次完整读写，另一选项是 CUDA 版 TensorIterator（`gpu_kernel`）按 stride 计算地址、零拷贝；第八篇：拷贝是纯 memory-bound 操作，时间 = 字节数 / 带宽，且跨步访问不合并——一个 warp 跨两段只剩 50% 带宽。

   </details>

5. 给 `scale_shift` 的 Schema 末尾加一个带默认值的参数 `gamma`。这个改动经过哪几关、哪几篇的机制？

   <details markdown="1"><summary>答案</summary>

   第六篇：改 `torch.library.define` / `TORCH_LIBRARY` 里的 Schema 与所有实现、Autograd、Fake，重跑 `opcheck` 与 `gradcheck`；第五篇：Schema 是契约，Python 绑定按参数匹配 overload；第十篇：末尾新增带默认值的参数是 ✓BC ✗FC——旧模型用默认值，新模型在旧版本上解析失败——CI 用 Schema 快照检查，`native_functions.yaml` 的改动有额外审批；OpInfo 的 `sample_inputs_func` 要补新参数的样本；性能基线（微基准）要重跑确认没变慢。

   </details>

### C. 面试题（7 题）

1. 从 `z = torch.add(x, y)` 出发，讲一遍 PyTorch 内部发生了什么，以及 Autograd 在哪一步介入。

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) Python API → Codegen 生成的绑定解析参数、匹配 overload（`add.Tensor`）→ `at::add` 进入 Dispatcher；(2) Dispatcher 从 Operator Table 取 Schema 校验，合成 DispatchKeySet（各输入 OR + TLS include − exclude），取最高优先级 Key；(3) Autograd 是包装 Key：记录 `AddBackward0`、保存反向所需值，排除自己再次分发；(4) 后端 Key 命中 `at::native::add`，TensorIterator 处理广播 / dtype 提升 / stride，launch Kernel 后 CPU 立即返回；(5) 结果 `TensorImpl` 与 `StorageImpl` 在 `c10` 构造，挂上 `grad_fn`。
   **追问方向**：`no_grad` 下 KeySet 差在哪；`torch/csrc` 与 ATen 之间为什么往返一次；`torch.compile` 后这条路径怎么变。
   **好答案与一般答案的区别**：一般答案说"查表找到 CUDA Kernel"；好答案讲出两轮查表、包装 Key 再分发、以及 CPU 提交后异步返回。

   </details>

2. `view`、`reshape`、`contiguous`、`clone` 各做什么？为什么同一个 Kernel 在转置过的输入上会慢好几倍？

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) `view` 只改元数据、要求新形状能在现有排列上解释；`reshape` 能 view 就 view 否则复制；`contiguous` 视布局而定；`clone` 一定新 Storage；(2) 转置后 strides 不再行主序；单输入时 TensorIterator 会沿物理布局遍历、不慢，但与另一个连续 Tensor 一起运算时其中一个必然跨 stride——warp 内 32 个线程地址分散，32 B 扇区用不满：跨步 2 只剩 50% 带宽，极端情形多搬 8 倍；(3) 逐元素算子几乎总是 memory-bound，时间由字节数 / 有效带宽决定；(4) 处方：让上游产出连续布局，或一次 `contiguous()` 换后续多个 Kernel 变快。
   **追问方向**：`x[:, 0]` 为什么不连续；`expand` 的 stride 0 与 in-place 的限制；`channels_last` 这类布局。
   **好答案与一般答案的区别**：一般答案背"view 不复制、reshape 可能复制"；好答案把 stride 一路讲到 warp 访存合并与带宽利用率。

   </details>

3. 评估阶段显存持续上涨、速度也慢，代码里已经调了 `model.eval()`。会查什么？`no_grad` 与 `inference_mode` 选哪个？

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) `eval()` 只切换 Module 行为，不关梯度——前向仍建图、保存激活；(2) 加 `inference_mode()`：不建图、不维护 version counter、不分配 AutogradMeta，比 `no_grad` 更快更省，产物不能再进入 autograd，纯推理正确选择；(3) 查是否把输出或 `loss` 存进容器持有了 `grad_fn`（`.item()` / `.detach().cpu()`）；(4) 用 memory snapshot 看 `allocated` 是否单调涨；(5) 区分 `reserved` 高（缓存）与 `allocated` 涨（泄漏）。
   **追问方向**：`detach` 与 `no_grad` 的作用域差异；`inference_mode` 的 Tensor 之后能否参与训练；BN 在 `eval` 下的行为。
   **好答案与一般答案的区别**：一般答案说"加 `torch.no_grad()`"；好答案说清三种上下文各省什么、为什么 `eval()` 不够、以及怎么用工具确认。

   </details>

4. `torch.compile` 做了什么？什么情况下它反而变慢或没效果？

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 三段：Dynamo 字节码层捕获 FX Graph + Guard，AOTAutograd 生成 ATen 级前反向图并 functionalize，Inductor 融合与内存规划生成 Triton / C++；(2) 收益来自省 Python 与分发开销、减少 launch、融合减少访存——对 memory-bound 逐元素链收益最大，对 cuBLAS 黑盒 `mm` 默认不融合；(3) 变慢或无效：频繁 graph break（依赖 Tensor 值的分支、`.item()`）；shape 经常变触发重编译、超过 8 个缓存条目退回 Eager；第一次调用秒级到几十秒编译（案例 47 s）；模型已 compute-bound；(4) 排查顺序：`explain` → `graph_breaks` → `recompiles` → `backend="eager"` / `"aot_eager"` → `output_code`。
   **追问方向**：`dynamic=True` 的代价；`reduce-overhead` 与 CUDA Graphs；`torch.export` 与 `compile` 的区别；缺 `register_fake` 的自定义算子怎么办。
   **好答案与一般答案的区别**：一般答案说"图优化、算子融合"；好答案区分编译器一维与运行时一维，并能说出 Guard 与 graph break 各自的失效模式。

   </details>

5. 一个训练作业 GPU 利用率只有 30%，怎么定位？给出流程与每一步的判据。

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 先建基线与正确性测试，计时必须同步或用 CUDA Event、warmup、中位数、写明口径；(2) `torch.profiler` 看时间线形态：GPU 稀疏 + CPU 忙 + `cudaLaunchKernel` 多而短 → launch-bound（增大 batch、融合 / compile、CUDA Graphs、fused 优化器）；CPU 时间不在算子上 → Python-bound（`with_stack`、向量化）；两侧交替空洞 → sync-bound（`.item()`、`nonzero`，`set_sync_debug_mode("warn")` 抓栈）；GPU 大段空白、CPU 停在 DataLoader → 数据加载（`num_workers`、prefetch、pinned + `non_blocking`）；(3) 解决 CPU 侧后再看 Kernel 分布：`mm` 主导用 bf16 Tensor Core，逐元素主导做融合；(4) 一次只改一件事，每步回到同一个 Benchmark，报告写清节省的是计算 / 访存 / 同步 / launch、显存与精度代价。
   **追问方向**：不 `synchronize` 计时量到了什么；Nsight Systems 与 Nsight Compute 各看什么；`reserved` ≫ `allocated` 报 OOM 怎么办。
   **好答案与一般答案的区别**：一般答案说"用 profiler 找热点 Kernel"；好答案先判断 GPU 是不是在等 CPU，把症状归到五类之一再对症。

   </details>

6. 4 台 8 卡 H100 训一个 70B 模型，怎么选并行策略？说出显存与通信的账。

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 先算显存：16P = 1.1 TB，64 卡纯 FSDP 每卡 17.5 GB 静态状态本可放下，但单层参数 + 激活太大、且跨节点 3P 通信藏不住 → 引入 TP=8 在节点内切每一层，FSDP 在跨节点 dp 维分片，参数是 2D DTensor（TP=8 是一种常见落点，不是唯一方案）；(2) TP 每层 4 次关键路径 all_reduce，通信激活 ∝ B·S·H，实践上限在 NVLink 域内；(3) 跨节点 FSDP 按 TP 分片后每卡通信 3P/8 的元素数，不随 dp 卡数减少，IB 每卡约 50 GB/s，若藏不住换 HSDP 或 PP（micro-batch ≥ 4K）；(4) 激活按 block 做 checkpointing（+33% 计算），序列很长再上 CP；(5) 判断标准只有两条：每卡显存放不放得下、通信时间能否被计算隐藏。
   **追问方向**：ZeRO 三级各分什么、通信为什么是 2P / 2P / 3P；1F1B 为什么显存 ∝ K；`MixedPrecisionPolicy` 里 fp32 分片、bf16 通信的理由；扩展效率不线性的四组原因。
   **好答案与一般答案的区别**：一般答案报一组"TP=8、PP=4"的配置；好答案从五类状态复制还是分片出发，给出每一步的显存与通信数字，并说清节点内外带宽差近一个数量级如何决定策略落点。

   </details>

7. 你写了一个融合算子准备放进公司的训练框架，怎样让它"具备生产质量"，并在 PyTorch 升级后还能用？

   <details markdown="1"><summary>答案</summary>

   **答案要点**：(1) 三步完整：Schema 声明 mutability / alias，CPU / CUDA / Autograd / Meta 四个槽位都注册，`register_fake` 让 `torch.compile` 能捕获；(2) 实现层处理 device（`CUDAGuard`、当前 stream）、dtype（`AT_DISPATCH`，含 bf16）、stride（`contiguous()` 或 TensorIterator）、生命周期；(3) 正确性用五种 oracle 中至少一种绝对标准：对照参考实现 `alpha * x + beta`、`gradcheck`（float64），再加跨设备、非连续、空 Tensor、极端数值；`opcheck` 一站式检查；(4) 性能：`benchmark.Timer` 对照原生两个 Kernel，基线入库、阈值守门，不比原生快就重新评估是否值得维护 C++；(5) 兼容：C++ API 无 BC 保证，扩展 pin 到具体 torch 版本并在 `__init__.py` 做版本检查，`_GLIBCXX_USE_CXX11_ABI` 与 `TORCH_CUDA_ARCH_LIST` 一致，随基础镜像交付；改 Schema 只在末尾加带默认值的参数，或新增 overload 并弃用旧的；(6) 升级 playbook：读发布说明 → 重编译扩展 → `-W error::FutureWarning` 跑测试 → 单卡 loss 对照 → checkpoint 兼容 → 多卡与 compile → 性能基线 → 灰度。
   **追问方向**：`opcheck` 具体查哪几件事；`load` 与 `setup.py` 各适合什么阶段；BC 与 FC 的区别；`torch/csrc/stable/` 稳定子集解决什么。
   **好答案与一般答案的区别**：一般答案停在"写 Kernel、测正确性"；好答案把第六篇的七条完成标准与第十篇的七关连起来，知道每一关守什么、坏在哪一层怎么定位。

   </details>

### D. 掌握判据

| 水平 | 表现 |
|---|---|
| 读过 | 能说出十篇各讲什么；知道 stride、`grad_fn`、DispatchKey、Guard、launch-bound、all_reduce、OpInfo 这些名词 |
| 掌握 | A 组能不翻书算出 8 题以上；B 组能说出每题用了哪几篇的什么；拿到一段慢的训练代码能先判断 GPU 在等谁、拿到一个分布式配置能算出每卡显存与每 step 通信量 |
| 能教人 | C 组每题能给出全部要点并预判追问；能解释十篇里每个反直觉结论（Autograd 只是一个 Key、FSDP 通信比 DDP 多 50%、低精度对 launch-bound 无效、`empty_cache()` 不解决碎片、flaky 不能靠重跑）为什么成立 |

通关标准：A 组至少 8 题、B 组至少 4 题、C 组每题能说出一半以上要点。没过的部分回到第二章对应篇的"必记"，再回该篇正文的相应章节。

## 六、下一步

十篇讲的是 PyTorch 本身：编程模型、算子系统、编译器、性能与分布式运行时、工程体系。几个紧邻的方向不在范围内，总纲的边界一节已经列出——Python 语言本身（CPython 的对象模型、GIL）、C++ 语言本身（模板、RAII、静态注册）、单个 CUDA Kernel 的内部优化（共享内存、Tensor Core 编程）、LLM 推理系统（Continuous Batching、KV Cache、Prefill / Decode 分离）。本系列覆盖它们依赖的基础，不讨论它们自身的设计。

- 本系列是[《AI 算法工程师学习地图》](/ai-algorithm-engineer-learning-roadmap.html) L1 工具箱的深入篇：算法侧的[《算法工程师的工具箱》](/tooling-for-ai-algorithm-engineers.html)讲 PyTorch 的"用"——五个对象、二十行训练循环、显存的账——本系列讲"改"。没读过前者的读者可以回去补"用"的那一半。
- 本系列在 Infra 地图上的位置与它前后的系列，见[《AI-Infra 工程师学习地图》](/ai-infra-learning-roadmap.html)；三张地图如何共享本系列，见[《AI 全栈学习地图》](/ai-fullstack-learning-roadmap.html)。

回到总纲：[《PyTorch 深度实践：从 Tensor 到深度学习运行时》](/deep-dive-into-pytorch.html)。

[^q0]: 总纲那串追问的每一个：Tensor 如何表示输入（元数据 + 共享 Storage，view 不复制）；Module 如何组织模型（`__setattr__` 注册，一切状态在 `state_dict`）；Autograd 如何建图（前向就地建 `grad_fn`，每节点算 VJP，`.grad` 累加）；Dispatcher 如何选算子（DispatchKeySet 查 Operator Table，包装 Key 再分发）；自定义算子如何接入（三步、两种接入、四个阶段，`opcheck`）；Compiler 如何变换（Dynamo → AOTAutograd → Inductor，Guard 决定复用）；Profiler 如何告诉你瓶颈在哪（两条时间线、五类瓶颈、显存四类）；多卡如何协同（五类状态各做复制或分片的决定）；Tests、Build 和 CI 如何保证演进（七关、五种 oracle、OpInfo、BC/FC）。详见[第二章](#二逐篇回顾)。
[^q1]: 源码四层 `torch/` → `torch/csrc/` → `aten/` → `c10/`；`[2,3,4]` 连续 strides `(12, 4, 1)`；fp16 最大 65504、bf16 尾数 7 位；Adam 训练 16 B/参数、7B → 112 GB、混合精度不变；一次 `add` 两次经过 Dispatcher；warp 32 线程、32 B 扇区、跨步 2 带宽 50%；`f` 热路径 6 次分发 → 1、缓存条目上限 8；每算子 CPU 固定成本 10～30 µs；A100 ridge point ≈ 10 / 156 FLOP/Byte、bf16 算力 16 倍；融合 5N → 2N；案例 166 → 2207 samples/s；ring all_reduce 每 rank $$2(N-1)/N \cdot n$$；DDP 2P、FSDP 3P、显存 16P → 16P/N；TP 每层 4 次 all_reduce、度 ≤ 8；PP 气泡 $$(K-1)/(M+K-1)$$；NVLink 与 IB 带宽差近一个数量级；2000+ 算子 × 约 15 种 dtype → 几十万测试；小版本每三到四个月、弃用保留通常两个小版本。详见[第一章](#一总览系列回答的问题与主线)、[第三章](#三贯穿全系列的几条线)。
[^q2]: 用第五章的三段自测：A 组 10 题判断与计算（至少 8 题）、B 组 5 题跨篇综合（至少 4 题）、C 组 7 道面试题（每题说出一半以上要点）；D 组的表给出"读过 / 掌握 / 能教人"三级的表现。详见[第五章](#五通关自测)。
