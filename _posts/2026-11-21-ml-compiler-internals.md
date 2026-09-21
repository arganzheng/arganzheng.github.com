---
layout: post
title: ML 编译器内部：从 SSA、MLIR 到 Triton 编译器（总纲）
subtitle: Inside ML Compilers, from SSA and MLIR to the Triton Compiler
tags: [Compiler, MLIR, LLVM, Triton, TVM, AI-Infra]
catalog: true
---


## 内容简介

《ML 编译器内部：从 SSA、MLIR 到 Triton 编译器》是一组共十三篇正文的系列文章，是 [AI-Infra 学习地图](/ai-infra-learning-roadmap.html)上唯一的**选修**系列。它面向两类人：已经用 Triton 写过 kernel、想知道编译器替自己做的那些决定是**怎么**做出来的工程师；以及准备进入 Triton、MLIR、TVM 这类项目做编译器开发的贡献者。

它回答的问题是：

> **一段块级的张量程序，怎样一步步变成一条条 GPU 指令？中间每一个决定——这个张量每个线程持有哪几个元素、这条 load 为什么是 128 bit、这个 barrier 为什么插在这里、这个循环为什么被拆成了三段——是哪一个 pass 做的，凭什么分析结果做的？**

地图上的两个系列已经把编译器当作黑盒讲过一遍：《PyTorch 深度实践》第七篇讲 `torch.compile` 的流水线（Dynamo 捕获 → AOTAutograd 变换 → Inductor 生成 Triton），《GPU Kernel 工程》第七篇讲 Triton 的编程模型和它的六层编译流水线（Python → TTIR → TTGIR → LLVM IR → PTX → cubin），并教读者去读 TTGIR 里的 `#blocked` 和 PTX 里的 `mma.sync`。那两篇回答了"编译器做了什么"。这个系列回答"**编译器怎么做到的**"：

| 黑盒视角（已在主线覆盖） | 打开黑盒（本系列） |
|---|---|
| Triton 有六层 IR | 每一层 IR 的数据结构是什么，为什么要分这么多层 |
| 编译器为张量选了 `#blocked<{sizePerThread=[8], …}>` | 这个 layout 是 `Coalesce` pass 根据 `AxisInfo` 分析的 contiguity 选的；layout 本身是 $$\mathbb{F}_2$$ 上的一个线性映射 |
| `tl.dot` 用上了 Tensor Core | `AccelerateMatmul` 按架构选 MMA 版本、改写操作数 layout；`DotOpToLLVM` 把它变成 `mma.sync` / `wgmma` 内联汇编 |
| `num_stages=3` 让 K 循环流水化 | 流水化是三个 pass 的合作：给 op 标延迟、算调度、按调度展开循环并把 load 提前 |
| `__syncthreads()` 的位置由编译器决定 | `Membar` 分析在 shared memory 的读写依赖之间插 barrier，`AllocateSharedMemory` 用活跃区间做分配 |
| "Triton 对不规整的代码效果差" | pass 的模式匹配失败时退回了保守路径——能在 IR 上指出是哪个 pass、哪一条模式没匹配上 |

Table: 黑盒视角与打开黑盒的对照

要做到这一点，需要先补上通用编译器的机制：IR、SSA、数据流分析、pattern rewrite、dialect conversion、LLVM 后端。它们不是 Triton 特有的，而是 Triton、TVM、XLA、Inductor 乃至 Java 的 C2 / Graal 共同的语言。系列前四篇讲这套语言，中间七篇用它读 Triton 编译器的源码，第十二篇用它对照 TVM 的另一条路，第十三篇讲编译器开发者的日常工作方式。


## 为什么写这个系列？

### Triton 是 AI-Infra 的"第二种 kernel 语言"，而它的行为由编译器决定

`torch.compile` 生成的所有 GPU 代码是 Triton，vLLM、SGLang、FlashInfer 里大量融合算子是 Triton，Unsloth、Liger-Kernel 这类训练加速库几乎全部是 Triton。这些代码的性能不取决于程序员写了什么，而取决于编译器对它做了什么：同一个 kernel，`tl.load` 的指针能不能被分析出 16 字节对齐，决定它是一条 `ld.global.v4` 还是四条 `ld.global.b32`；一个 layout 转换能不能被消除，决定 epilogue 有没有一次 shared memory 往返。

只会"用"Triton 的工程师，遇到性能问题只能试——换 `BLOCK_SIZE`、加 `tl.multiple_of`、调 `num_stages`——因为不知道编译器在看什么。能读编译器的工程师，会 dump 出 TTGIR，看到 `sizePerThread = [1]`，知道是 `AxisInfo` 没推出 contiguity，再回头看是哪个指针运算把信息丢了。这个系列要把读者从前一种变成后一种。

### MLIR 已经是 ML 编译器的通用基础设施

Triton 编译器是一组 MLIR 方言和 pass；`torch-mlir`、IREE、XLA 的 StableHLO、TVM 的部分组件、以及各家 AI 加速器厂商的编译器，都建立在 MLIR 上。MLIR 定义了一套所有这些项目共享的词汇——Operation、Region、Dialect、Pass、Pattern、Conversion——和一套共享的工具（`mlir-opt`、`--mlir-print-ir-after-all`、lit / FileCheck）。学会这套东西，读任何一个 MLIR 项目的源码都有了入口；不学，读 Triton 的 `lib/` 目录就像读没有类型定义的代码。

### 主线两篇留下的三个缺口

《GPU Kernel 工程》第七篇说清了六层 IR 各做什么，但有三处只能给出结论、给不出机制：

1. **layout 是什么**。那篇教读者读 `sizePerThread` / `threadsPerWarp` / `warpsPerCTA`，但没有说 `#blocked`、`#mma`、`#dot_op`、`#slice` 之间怎样转换、转换的代价怎么算、编译器怎么判断两个 layout 是否等价。答案是 Triton 3.x 引入的 **Linear Layout**：把每种 layout 统一表示为 GF(2) 上的线性映射，layout 之间的转换、组合、求逆都变成矩阵运算。它是理解 Triton 3.x 编译器的钥匙，主线里没有位置放它。
2. **pass 的算法**。那篇列了 Coalesce、Pipeline、RemoveLayoutConversions 的名字和效果，没有讲算法——`RemoveLayoutConversions` 的前向传播与后向重物化怎么决定把一个 `convert_layout` 往哪个方向推、什么时候接受重算换取消除转换。
3. **从 TTGIR 到 PTX 那一跳**。一个块级的 `tt.reduce` 怎么变成"线程内累加 → warp shuffle → shared memory 跨 warp"三级代码；一个 `convert_layout` 什么时候走 shuffle、什么时候走 shared memory；barrier 由谁插、shared memory 由谁分配。这一跳是 `TritonGPUToLLVM` 目录下两万多行 C++，也是 Triton PR 最密集的地方。

### 编译器知识对 Java 工程师是"已有一半"

后端工程师对编译器不陌生：`javac` 生成字节码，HotSpot 的 C1 / C2 做 JIT，Graal 用 Sea of Nodes IR，GraalVM 的 Truffle 做部分求值。SSA、逃逸分析、内联、循环不变量外提，都在 JVM 调优的日常词汇里。ML 编译器用的是同一套骨架，差别在 IR 的抽象层级（张量而不是标量）和优化目标（访存和 layout 而不是分支预测和逃逸）。本系列在每一个通用机制处给出 JVM 的对应物，让读者把已有的一半知识接上另一半。

### 现有材料的断层

编译器教材（龙书、Engineering a Compiler、SSA book）讲标量语言的编译，不涉及张量、layout、GPU 内存层次；MLIR 的官方文档是参考手册，Toy 教程止步于一个玩具语言；Triton 的文档只有用户手册，编译器部分的知识散落在 PR 描述、`docs/` 目录下几篇未完成的设计文档和社区会议的录像里；TVM 的文档偏 API。没有一份材料把"通用编译器机制 → MLIR → 一个真实的生产级 GPU 编译器的每个 pass"连成一条线。这个系列补这条线。


## 适合哪些读者？

### 用 Triton 写 kernel、想知道编译器在看什么的工程师

读完《GPU Kernel 工程》第七篇、写过几个 Triton kernel，遇到过"为什么加了 `tl.multiple_of` 快了一倍"、"为什么这个 kernel 的 TTGIR 里有三个 `convert_layout`"这类问题的人。这个系列让他们能在 IR 上直接看到原因。

### 准备给 Triton / MLIR 项目提 PR 的开发者

Triton 的 issue 里大量是"某个形状下编译出的代码有 spill"、"某个 layout 组合不支持"、"Blackwell 上某个 pattern 没有走 `tcgen05`"。修这类问题需要知道 pass 流水线的顺序、每个 pass 的输入输出、怎么用 `triton-opt` 复现和用 lit 测试固定。第五到第十一篇和第十三篇就是为此准备的。

### 做硬件适配、需要给新后端接 Triton 或 MLIR 的工程师

Triton 的 `third_party/` 目录是它的后端插件机制：NVIDIA 和 AMD 各一套方言和 lowering。给一个新加速器接 Triton，要理解哪些 pass 是通用的（TTIR、TTGIR 的大部分）、哪些是后端的（`TritonGPUToLLVM` 里的 `TargetInfo`、MMA 版本选择、异步拷贝指令）。

### 想在 ML 编译器之间迁移知识的人

学过 TVM 想读 Triton，或者反过来；用过 XLA 想理解 Inductor。第一篇的家谱和第十二篇的设计空间对照给出这些系统在同一张坐标上的位置。


## 系列的整体主线

组织轴是**一个编译器 = IR + pass + lowering**，顺着 Triton 的编译流水线自上而下走；每到一站，先讲这一站需要的通用编译器机制，再打开 Triton 源码看它是怎么实现的。

| 篇 | 主题 | 一句话 |
|---|---|---|
| 第一篇 | 编译器的骨架 | IR、SSA、数据流分析、pass、渐进式下降；ML 编译器家谱 |
| 第二篇 | LLVM | 所有 ML 编译器共同的后端；NVPTX 与 ptxas |
| 第三篇 | MLIR（上） | Operation / Region / Dialect：IR 的数据结构与 ODS |
| 第四篇 | MLIR（下） | Pass、Pattern Rewrite、Dialect Conversion：变换 IR 的三种方式 |
| 第五篇 | Triton 前端 | Python AST 如何变成 TTIR |
| 第六篇 | TTIR 级分析 | AxisInfo：一切向量化决定的地基 |
| 第七篇 | layout 系统 | 分布式 layout 的语义、Linear Layout、初始 layout 与 Coalesce |
| 第八篇 | layout 优化 | RemoveLayoutConversions、AccelerateMatmul、shared layout 与 swizzle |
| 第九篇 | 循环与异步 | 软件流水的三段分解、TMA / wgmma / warp specialization、Blackwell、Gluon |
| 第十篇 | 下降到 LLVM | load / store、reduce、dot、convert_layout 的 lowering；shared memory 分配与 barrier 插入 |
| 第十一篇 | 后端与运行时 | LLVM → PTX → cubin、编译缓存、launcher、AMD 对照 |
| 第十二篇 | TVM | 调度语言与自动调优：另一条路 |
| 第十三篇 | 工作台 | 构建、`triton-opt`、lit 测试、二分定位、读一个真实 PR |

Table: 十三篇的主题与一句话

三条交织的线索：

| 线索 | 从第一篇到第十三篇 |
|---|---|
| 机制线 | SSA 与数据流 → LLVM IR → MLIR 的 IR 结构 → 三种变换框架 → 每种机制在 Triton 里的实例 |
| Triton 线 | 一个 matmul kernel 从 Python 源码开始，逐篇多走一步，到第十一篇变成 cubin 并被 launch |
| 设计线 | 每个决定（分层、显式 layout、用户定 tile、编译器定线程映射）在第一篇提出问题，在 Triton 各篇看到答案，在第十二篇与 TVM / XLA 的另一种答案对照 |

Table: 贯穿系列的三条线索

前四篇的例子用 `clang`、`opt`、`llc`、`mlir-opt` 跑；第五到十一篇的 IR 用 Triton v3.8.0 的 `triton-opt`、`triton-tensor-layout` 和仓库里的 lit 测试（每个 pass 的输入 IR 与期望输出）给出；第十二篇的 TVM 例子用 v0.26.0。所有中间表示都是真实工具的输出，不是示意。


## 章节结构与分章导读

### 1. 编译器的骨架：IR、SSA 与 pass

第一篇不涉及 GPU，也不涉及 MLIR。它建立后面十二篇共用的词汇，并回答一个前置问题：**已经有 LLVM 和 nvcc 了，ML 为什么还要自己的编译器？**

这一篇会讨论：

- 编译器的三段：前端（源码 → IR）、中端（IR → IR）、后端（IR → 机器码），以及"中端"为什么是编译器工作量最大的部分；
- IR 的几种形态：AST、三地址码、控制流图（CFG）与基本块；
- SSA：每个值只赋一次、φ 函数、支配树与支配边界；为什么 SSA 让 use-def 链变成 O(1)、让绝大多数优化变简单；用 `clang -emit-llvm` 和 `opt -passes=mem2reg` 亲眼看一段 C 代码变成 SSA；
- 数据流分析：活跃变量、可达定义、常量传播的格（lattice）与不动点迭代；
- 经典标量优化：常量折叠、公共子表达式消除、死代码消除、循环不变量外提、内联，每个都在 LLVM 上跑一遍；
- pass 与 pass manager：pass 的粒度（模块 / 函数 / 循环）、analysis 与 transform 的区别、analysis 失效；
- "lowering"与"渐进式下降"：为什么不是一步从源码到机器码，每一层 IR 保留什么信息、丢掉什么信息；
- Java 参照：`javac` 只做前端、HotSpot C2 的 Sea of Nodes、Graal IR、JIT 与 AOT 的取舍；
- ML 编译器为什么存在：标量 IR 丢掉了张量语义——融合、layout、内存规划、对 Tensor Core 这种"矩阵指令"的匹配都需要在张量层面做；
- ML 编译器家谱：XLA / HLO、TVM / Halide、Glow、nvcc、Inductor、Triton、MLIR 各自的出身与位置。

核心问题是：

> **`a * b + c` 这一行代码在 LLVM 里是三条 SSA 指令；同样这一行在 Triton 里是三个块级 op，操作 `tensor<128x64xf32>`。两者的"优化"分别指什么？为什么后者的优化空间不能在前者上做出来？**

### 2. LLVM：所有 ML 编译器共同的后端

第二篇讲 LLVM，因为 Triton、XLA、TVM、Inductor 的 CPU 路径最后都落到 LLVM IR，而 GPU 路径落到 LLVM 的 NVPTX 后端。不理解这一层，就不知道 Triton 的 LLVM IR dump 里哪些东西是 Triton 决定的、哪些是 LLVM 决定的。

这一篇会覆盖：

- LLVM IR 的结构：Module、Function、BasicBlock、Instruction；类型系统；`getelementptr`；metadata 与 attribute；
- 新 pass manager 与 `opt -passes=`；`-O2` 流水线里的关键 pass 与顺序；
- 后端：指令选择（SelectionDAG 与 GlobalISel）、指令调度、寄存器分配、机器码发射，以及为什么 GPU 后端的寄存器分配决定 occupancy；
- **NVPTX 后端**：地址空间（global / shared / local / const）在 IR 里怎么表示、`llvm.nvvm.*` intrinsic、内联汇编（`asm sideeffect`）、kernel 的 metadata（`nvvm.annotations`、`maxntid`）；用 `llc -march=nvptx64` 把一段 LLVM IR 编成 PTX；
- PTX 是什么：虚拟 ISA、前向兼容、`.version` / `.target`；
- **ptxas 是第二个编译器**：PTX → SASS 的寄存器分配、指令调度、spill；`-v` 输出怎么读、`n_regs` / `n_spills` 从哪来；为什么 Triton 的 LLVM IR 看起来"寄存器无限"；
- AMD 的对应物：AMDGPU 后端、GCN / CDNA ISA、`llvm.amdgcn.*`；
- LLVM 在 Triton 里的位置：Triton 自己 fork 的 LLVM、`make_llir` / `make_ptx` 两个阶段调用了 LLVM 的哪些部分。

核心问题是：

> **Triton 生成的 LLVM IR 里没有任何一处提到寄存器数量，最终 kernel 却可能因为寄存器不够而 spill 到 local memory。这个决定是谁做的、在哪一步、能不能从 Triton 侧影响它？**

### 3. MLIR（上）：Operation、Region、Dialect

第三篇进入 MLIR。Triton 编译器的每一行 C++ 都在操作 MLIR 的数据结构，这一篇把它们讲清楚。

这一篇会覆盖：

- MLIR 解决的问题：LLVM IR 抽象层级固定，每个领域重新发明 IR 和基础设施；MLIR 把"IR 的基础设施"和"IR 的内容"分开；
- 核心数据结构：Operation 是唯一的单位；Value、Type、Attribute、Block、Region 之间的包含关系；"一切都是 Op"——函数、模块、循环都是带 Region 的 Op；
- 通用形式（generic form）与自定义汇编格式：同一个 IR 的两种打印方式，`--mlir-print-op-generic` 揭示数据结构；
- Dialect：命名空间加一组 Op / Type / Attribute；`builtin`、`func`、`arith`、`scf`、`cf`、`memref`、`tensor`、`linalg`、`vector`、`gpu`、`nvvm`、`llvm` 各方言的分工；
- ODS（Operation Definition Specification）：用 TableGen 声明 Op 的操作数、结果、属性、trait、interface、汇编格式和 verifier，`mlir-tblgen` 生成 C++ 类；读一段 `.td` 文件；
- Trait 与 Interface：`Pure`、`SameOperandsAndResultType`、`MemoryEffects`、`InferTypeOpInterface`——pass 依赖它们而不是具体 Op；
- 类型与属性的定义：参数化类型、属性作为 layout 的载体（这正是 Triton 表示 layout 的方式）；
- 对照读 Triton：`include/triton/Dialect/Triton/IR/TritonOps.td` 里 `tt.load`、`tt.dot`、`tt.reduce` 的定义，`TritonGPUAttrDefs.td` 里 `#blocked` 的参数。

核心问题是：

> **`scf.for` 是一个 Op，它的循环体是一个 Region，循环携带的值（iter_args）是 Block 的参数。在这种表示下，"循环不变量外提"这个优化需要哪些信息、从哪些接口拿？为什么 MLIR 不像 LLVM 那样用 φ 函数？**

### 4. MLIR（下）：Pass、Pattern Rewrite 与 Dialect Conversion

第四篇讲变换 IR 的三种方式。Triton 的每个 pass 都属于其中一种。

这一篇会覆盖：

- Pass 基础设施：`OperationPass<ModuleOp>` 与 `OperationPass<FuncOp>`、PassManager 的嵌套、analysis 的缓存与失效、`--pass-pipeline` 的文本语法；
- Pattern Rewrite：`RewritePattern` 的 `match` 与 `rewrite`、`PatternRewriter` 为什么不能直接修改 IR、greedy driver 的工作方式（worklist、不动点、benefit）；
- Canonicalization 与 folding：`hasCanonicalizer`、`fold` 方法、`arith.constant` 的物化，为什么 canonicalize 是所有流水线里出现最多的 pass；
- 声明式的 pattern：DRR（TableGen）与 PDLL，Triton 的 `Combine.td` 就是 DRR；
- **Dialect Conversion**：`ConversionTarget` 的合法性声明、`TypeConverter`、`ConversionPattern` 与 `OpAdaptor`、`applyPartialConversion` 与 `applyFullConversion`、materialization；为什么它比 greedy rewrite 复杂——一次性替换多个方言时类型不一致的过渡状态怎么处理；
- 用 `mlir-opt` 把一个 `linalg.matmul` 逐步下降：linalg → scf 循环 → memref → llvm，每一步看 IR；
- 数据流分析框架：`dataflow::SparseForwardDataFlowAnalysis`、lattice 的 `join`，Triton 的 `AxisInfo` 就建在它上面；
- 写一个最小的 MLIR pass（C++），编译成 `mlir-opt` 的插件并运行。

核心问题是：

> **`ConvertTritonToTritonGPU` 要把没有 layout 的 `tensor<128x64xf32>` 全部变成带 layout 的 `tensor<128x64xf32, #blocked>`，过程中每一个 Op 的操作数和结果类型都要换。这为什么必须用 Dialect Conversion 而不能用 greedy pattern rewrite？TypeConverter 在其中做什么？**

### 5. Triton 编译器（一）：从 Python AST 到 TTIR

第五篇开始读 Triton 源码。前端是唯一用 Python 写的一段：它不是 Python 解释器，而是一个把 Python 语法树翻译成 MLIR 的**编译器前端**。

这一篇会覆盖：

- `@triton.jit` 做了什么：`JITFunction` 保存源码而不是执行它；调用时怎样从参数值算出**特化 key**（dtype、`constexpr` 值、整数参数的 divisibility 与"是否等于 1"）；为什么同一个函数会被编译多次；
- `ASTSource` 与 `CodeGenerator`：`ast.NodeVisitor` 遍历 Python AST，每个节点变成 MLIR builder 调用；`tl.tensor` 对象是**编译期的句柄**而不是数据；
- 类型系统：`tl.pointer_type`、`tl.block_type`、整数与浮点类型，`tl.constexpr` 的代入与编译期分支删除；
- 控制流：Python 的 `for` / `while` / `if` 如何变成 `scf.for` / `scf.while` / `scf.if`，循环携带值如何被识别为 iter_args，为什么循环里不能改变张量的 shape；
- `tt` 方言的 op 一览：指针运算（`addptr`、`splat`、`broadcast`、`make_range`、`expand_dims`）、访存（`load` / `store` 与 mask、`other`、cache 修饰符）、计算（`dot`、`reduce`、`scan`、`elementwise`）、张量描述符（`make_tensor_descriptor`、`descriptor_load`）；
- `semantic.py`：Python 层的类型检查与广播规则，为什么 Triton 的广播比 NumPy 严格；
- 函数调用与内联：`tt.call` 与 inliner pass；
- TTIR 级 pass：`Combine`（DRR 模式，如 `dot + add → dot(acc)`）、`ReorderBroadcast`、`RewriteTensorDescriptorToPointer`、`LoopUnroll`、CSE、canonicalize——它们的顺序在 `third_party/nvidia/backend/compiler.py` 的 `make_ttir` 里；
- 用 `triton.compile` 拿到一个 matmul 的 TTIR，逐行对照 Python 源码。

核心问题是：

> **Triton 的前端在编译期就知道每个 `constexpr` 的值和每个整数参数是否是 16 的倍数，这些信息是怎么从"调用时的实参"传到"编译出的 IR"里的？为什么 `x_ptr + offsets` 在 TTIR 里是 `tt.addptr` 而不是 `arith.addi`？**

### 6. Triton 编译器（二）：AxisInfo——一切向量化决定的地基

第六篇专讲一个分析：`AxisInfo`。它决定了后面 Coalesce 选什么 layout、load / store 能不能向量化、mask 能不能被证明全真。是 Triton 编译器里"一个分析影响一切"的典型。

这一篇会覆盖：

- 分析的对象：每个整数 / 指针张量沿每一维的三个属性——**divisibility**（元素值的最大 2 幂因子）、**contiguity**（连续递增的最长段）、**constancy**（相等的最长段）——的精确定义；
- 传递规则：`make_range` 的初值、`splat` / `broadcast` / `expand_dims` 如何变换三个属性、`addi` / `muli` / `addptr` 的合并规则（例如两个 divisibility 分别为 16 和 4 的值相加，结果的 divisibility 是 4）、`select` 与 `cmpi` 的处理；每条规则用一个小例子验证；
- 信息从哪来：函数参数的 `tt.divisibility = 16` 属性由前端按实参值标上，`tl.multiple_of` / `tl.max_contiguous` / `tl.max_constancy` 是显式注入（它们只是在 op 上挂一个属性），`constexpr` 常量自带精确值；`tl.assume` 为什么不在这个名单里；
- 建在 MLIR dataflow 框架上：`AxisInfoAnalysis` 是一个 `SparseForwardDataFlowAnalysis`，lattice 的 `join` 取最保守值，循环里为什么要迭代到不动点；
- 信息怎么丢：一次 `remsi`、一个未标注的 stride、一个 `int64` 溢出保护，都能让 contiguity 归 1；用 `triton-opt --test-print-alignment` 把每个值的 AxisInfo 打出来；
- 谁在消费它：Coalesce（第七篇）、`LoadStoreOpToLLVM` 的向量宽度（第十篇）、mask 的全真判断、TMA 描述符的对齐检查；
- 与 LLVM 的 `KnownBits` / `ScalarEvolution` 对照。

核心问题是：

> **`offs = pid * BLOCK + tl.arange(0, BLOCK)`，`BLOCK = 1024`，`pid` 是 `tt.get_program_id`。编译器如何推出 `offs` 沿这一维 contiguity 为 1024、divisibility 为 1024？如果用户写的是 `offs = pid * n + tl.arange(0, BLOCK)`，`n` 是运行时整数，结论变成什么？多了一个 `tl.multiple_of(n, 16)` 又变成什么？**

### 7. Triton 编译器（三）：layout 系统与 Linear Layout

第七篇是整个系列的核心。Triton 编译器区别于其他 GPU 编译器的地方，就是它把"每个线程持有张量的哪些元素"作为类型的一部分显式表示出来，并在 IR 上对它做优化。

这一篇会覆盖：

- 为什么需要 layout：块级程序不指定线程映射，编译器必须补上；把映射放在类型里而不是 pass 的内部状态里，让每个 pass 都能看到并改写它；
- 分布式 layout 的精确语义：`#blocked`（`sizePerThread` / `threadsPerWarp` / `warpsPerCTA` / `order` / `CTAsPerCGA`）、`#slice`（reduce 结果的 layout：从父 layout 去掉一维）、`#nvidia_mma`（累加器在 `mma.sync` / `wgmma` 中的 fragment 布局）、`#dot_op`（`tl.dot` 操作数的布局，`opIdx` 与 `kWidth`）；每种 layout 画出线程 ↔ 元素的格子图；
- 回绕（wrap）与复制：layout 覆盖的元素数与张量元素数不相等时发生什么；
- **Linear Layout**：把 layout 看成从（寄存器编号、lane、warp、block）的二进制位到（元素坐标）的二进制位的 $$\mathbb{F}_2$$ 线性映射；每个 `#blocked` 都能写成一个 0/1 矩阵；layout 的组合（`compose`）、求逆（`invert`）、乘积（`*`）、切片对应矩阵运算；两个 layout 之间的转换是否只需 warp 内 shuffle，看的是转换矩阵的哪些位被触及；用 `triton-tensor-layout` 打印任意 layout 的线程映射并与矩阵对照；
- `lib/Tools/LinearLayout.cpp` 与 `LinearLayoutConversions.cpp`：每种传统 layout 到 Linear Layout 的转换函数，为什么 3.x 之后所有 lowering 只认 Linear Layout；
- `ConvertTritonToTritonGPU`：初始 layout 的选择规则（按 `num_warps` 与元素数生成默认 `#blocked`），这一步是 Dialect Conversion 的实例；
- **Coalesce**：对每个 load / store，按 `AxisInfo` 的 contiguity 与 divisibility 算出每线程最多能连续持有多少元素（不超过 128 bit），沿哪一维排 order，生成新的 `#blocked` 并在前后插 `convert_layout`；
- shared memory 的 layout：`#shared`（swizzled）、`#nvmma_shared`（TMA 与 wgmma 要求的布局）、padded shared；swizzle 参数 `vec` / `perPhase` / `maxPhase` 与 bank conflict 的关系；shared 的 Linear Layout 表示。

核心问题是：

> **一个 `[64, 64]` 的 BF16 张量、`num_warps=4`，写出它默认 `#blocked` layout 的 Linear Layout 矩阵（哪些输入位映射到哪些输出位）。它到 `#nvidia_mma` 累加器 layout 的转换，为什么不能只用 warp 内 shuffle 完成，而必须经过 shared memory？从矩阵的哪一部分能看出来？**

### 8. Triton 编译器（四）：layout 优化与 Tensor Core 路径

第八篇讲在 layout 上做的优化。Coalesce 之后 IR 里到处是 `convert_layout`，每一个都可能是一次 shared memory 往返；这一篇的 pass 负责把它们消掉，并把 `tt.dot` 接到硬件矩阵指令上。

这一篇会覆盖：

- **RemoveLayoutConversions** 的两个阶段：前向传播（把 layout 沿 def-use 链向下推，直到遇到 load / store 这类"锚点"）与后向重物化（把 `convert_layout` 沿操作数向上推、必要时复制一段计算，用重算换取消除转换）；代价模型——什么时候复制值得；循环携带值上的传播为什么特别；`hoistConvertOnTopOfExtOrBroadcast` 这类局部规则；
- **AccelerateMatmul**：按 compute capability 选 MMA 版本（Ampere `mma.sync` v2、Hopper `wgmma` v3、Blackwell `tcgen05` v5），为累加器生成 `#nvidia_mma` layout，为操作数生成 `#dot_op`，决定 `warpsPerCTA` 的形状与 `instrShape`；`kWidth` 是什么、为什么 BF16 与 FP8 不同；Blackwell 路径把累加器放进 Tensor Memory（TMEM）的改写；
- **OptimizeDotOperands**：把转置折进 `ldmatrix.trans` / 让操作数直接从 shared memory 喂给 `wgmma`，消掉一次寄存器往返；
- **OptimizeThreadLocality**：reduce 前重排 layout，让更多 reduce 在线程内完成；
- **ReduceDataDuplication**、**F32DotTC**（TF32 拆三次 BF16 的模拟）、`CombineTensorSelectAndIf` 这些小 pass 的位置；
- 读 lit 测试：`test/TritonGPU/combine.mlir`、`accelerate-matmul.mlir`、`dot-operands.mlir` 里每个 case 的前后 IR；
- 一个 matmul 走完这几个 pass 之后 TTGIR 的样子，标出每个 layout 是哪个 pass 决定的。

核心问题是：

> **Coalesce 之后，一个 matmul kernel 的 TTGIR 里有二十多个 `convert_layout`；`RemoveLayoutConversions` 跑完还剩几个，最终只有一个在 epilogue。为什么那一个消不掉——它的两端各被什么"锚"住了？如果 kernel 的输出也是 `tl.dot` 的输入（例如 attention 的 P·V），它就能消掉，为什么？**

### 9. Triton 编译器（五）：软件流水、异步与 warp specialization

第九篇讲循环变换。GEMM 的 K 循环要把下一块的 load 与当前块的 `mma` 重叠，这在 CUDA 里是手写的多 stage `cp.async` 环形缓冲，在 Triton 里是几个 pass 的合作。

这一篇会覆盖：

- 软件流水的经典理论：modulo scheduling、prologue / kernel / epilogue、initiation interval；
- Triton 的三段分解：**AssignLatencies**（给每个 load 标"要提前几个迭代"）、**ScheduleLoops**（给循环体的每个 op 分配 stage 与 cluster）、**Pipeline**（`PipelineExpander` 按调度展开循环：把 load 变成 `async_copy_global_to_local` 写进 `num_stages` 个 shared memory 缓冲，插入 `async_wait`，生成 prologue 与 epilogue）；`LowerLoops` 把高层的"异步 load"变成具体的缓冲区索引运算；
- 为什么 `num_stages=3` 对应两个 shared memory 缓冲、PTX 里的 `cp.async.wait_group 2`：缓冲区轮转的精确账；
- 间接 load（`loop-pipeline-indirect-load.mlir`）与依赖距离；哪些循环不能流水化；
- **Prefetch**：把 `mma` 操作数 shared → 寄存器的搬运提前一个子迭代；
- **ReorderInstructions**：把 `local_load` 移到离使用者更近的地方，减少寄存器压力；
- Hopper 路径：TMA（`make_tensor_descriptor` → `async_tma_copy_global_to_local`、`mbarrier`）与 `wgmma`（`warp_group_dot`、异步 MMA 与 `warp_group_dot_wait`）；`TMALowering`、`FenceInsertion`、`MMALowering` 这些 TritonNvidiaGPU 方言的 pass；
- **Warp specialization**：把循环体切成 load 分区与 MMA 分区，让不同 warp group 各干一件事；`PartitionLoops`、`PartitionScheduling`、`OptimizePartitionWarps` 的分工，以及自动 warp specialization 的启用条件；
- Blackwell：`tcgen05.mma` 与 TMEM（`tmem_alloc`、`tmem_load` / `tmem_store`、`TensorMemoryAllocation`）、两 CTA 的 MMA、`PlanCTA`；
- **Gluon**：Triton 3.4 之后的显式 layout 子语言——用户在 Python 里直接写 `BlockedLayout`、`NVMMASharedLayout`、`mbarrier`、`tma`、`tcgen05`，跳过 Coalesce 到 Pipeline 的自动决定；它把《GPU Kernel 工程》第七篇说的"编译器的边界"变成了一个正式的出口。

核心问题是：

> **`num_stages=3` 的 matmul 循环，流水化之后 prologue 里有几次 load、循环体里 `async_wait` 等的是哪一批、shared memory 里有几个缓冲？Hopper 上换成 TMA + `wgmma` 之后这些数字怎么变，多出来的 mbarrier 是谁在等谁？warp specialization 把哪些 op 分给了哪些 warp？**

### 10. Triton 编译器（六）：从 TritonGPU 到 LLVM

第十篇讲那一跳：块级的带 layout 的 op 变成每个线程的标量 / 向量代码。这是 `lib/Conversion/TritonGPUToLLVM` 与 `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM` 两个目录，是一次 Dialect Conversion。

这一篇会覆盖：

- 下降前的准备：`scf → cf`（结构化控制流变成基本块与跳转）、`AllocateWarpGroups`、**AllocateSharedMemory**（每个需要 shared memory 的 op 的大小、按活跃区间做偏移分配、`ConvertLayoutOp` 的 scratch 空间）、**Membar**（在 shared memory 的写后读、读后写之间插 `barrier`，为什么它是一个数据流分析）；
- `TritonGPUToLLVMTypeConverter`：`tensor<…, #layout>` 变成 `!llvm.struct<(f32, f32, …)>`——每线程持有的元素数由 Linear Layout 算出；
- 每类 op 的 lowering：
  - `load` / `store`：按 `AxisInfo` 与 layout 算向量宽度，生成带谓词的 `ld.global.v4.b32` 内联 PTX，mask 的处理，`other` 值的填充；
  - `reduce`：线程内累加 → `shfl.sync.bfly` 做 warp 内规约 → 写 shared memory → 跨 warp 再规约，为什么需要 `#slice` layout；
  - `dot`：`mma.sync` 的 fragment 装载顺序与 `ldmatrix`；`wgmma` 的描述符构造；`tcgen05` 的 TMEM 地址；
  - `convert_layout`：三条路——只涉及寄存器位时直接重排、只涉及 lane 位时 `shfl`、涉及 warp 位时走 shared memory；Linear Layout 的"分解"怎么决定走哪条；
  - `make_range`、`splat`、`broadcast`、`expand_dims`、`trans`：视图类 op 为什么几乎不生成代码；
  - `local_alloc` / `local_load` / `async_copy`：shared memory 地址计算与 swizzle 的实现；
- `TargetInfo`：NVIDIA 与 AMD 后端各自实现的接口——shuffle、barrier、原子、打印，通用 lowering 通过它屏蔽差异；
- `PTXAsmFormat`：Triton 自己的内联 PTX 生成器；
- 读一段 matmul 的 LLVM IR dump，标出每段来自哪个 lowering pattern。

核心问题是：

> **TTGIR 里没有一处写 `bar.sync`，生成的 PTX 里却有十几个。每一个是哪个分析、根据什么信息插进去的？一个 `tt.reduce` 沿 axis=1 规约 `[128, 64]` 的张量，下降后哪一部分是寄存器内加法、哪一部分是 `shfl.sync`、哪一部分要经过 shared memory——这由什么决定？**

### 11. Triton 编译器（七）：PTX、cubin 与运行时

第十一篇走完最后一段：LLVM IR 变成可 launch 的 cubin，以及 Python 侧的运行时怎样把它跑起来。

这一篇会覆盖：

- `make_llir` 的收尾：`nvvm` 方言到 LLVM 方言、`mlir::translateModuleToLLVMIR`、Triton 对 LLVM 的 `-O3` 调用、`libdevice` 的链接（数学函数从哪来）；
- `make_ptx`：NVPTX 后端的调用、`ptx_version` 与 compute capability 的对应、生成的 PTX 头部；
- `make_cubin`：`ptxas` 的调用参数（`-O3`、`--gpu-name`、`-lineinfo`、`--fmad`）、Blackwell 用另一版 ptxas 的原因、`n_regs` / `n_spills` / shared memory 用量的回读；
- 编译缓存：cache key 由源码哈希、特化 key、编译选项、后端与 ptxas 版本组成；`~/.triton/cache/<hash>/` 里的 JSON、`.ttir` … `.cubin` 各文件；什么改动会导致重新编译；
- launcher：`make_launcher.py` 生成的 C 代码——参数打包、`cuLaunchKernelEx`、cluster 维度、TMA 描述符的传递；`driver.py` 与 `driver.c` 加载 cubin（`cuModuleLoadData`）与查询设备属性；
- 运行时的几件事：`CompiledKernel.run` 的参数检查、`launch_metadata`、`autotune` 的 benchmark 循环与 `prune_configs_by`、`do_bench` 与 L2 flush、`triton.knobs` 与环境变量（`TRITON_KERNEL_DUMP`、`TRITON_ALWAYS_COMPILE`、`MLIR_ENABLE_DUMP`）；
- **AMD 后端对照**：`third_party/amd` 的 `make_ttgir` 顺序有什么不同（`AMDGPUAccelerateMatmul` 选 MFMA / WMMA、`StreamPipeline` 代替 Pipeline、`ReorderInstructions` 的 AMD 版、`BypassLDSForDotOperand`）、`#amd_mfma` layout、`llvm.amdgcn.*` 与 `hsaco`；哪些代码是共享的、哪些是各自的——这是"给新硬件接 Triton"的地图；
- Interpreter 模式：`TRITON_INTERPRET=1` 用 NumPy 逐 program 执行，它跳过整条流水线，是调正确性的工具。

核心问题是：

> **同一个 `@triton.jit` 函数，用不同的 `BLOCK_SIZE` 调、用 stride 为 1 与不为 1 的张量调、在 A100 与 H100 上调，各产生几个 cache 目录？改了 Triton 的一个 pass 之后，旧的 cache 为什么仍然会被命中——怎样让它失效？**

### 12. TVM：调度语言与自动调优——另一条路

第十二篇换一个系统。TVM 代表 ML 编译器的另一条路线：程序员（或搜索算法）显式地写调度，编译器负责把调度忠实地实现出来。对照它，Triton 的设计取舍才看得清。

这一篇会覆盖：

- Halide 的遗产：**算法与调度分离**——算法说"算什么"，调度说"循环怎么切、怎么排、哪里向量化、哪里并行、放在哪层存储"；
- Tensor Expression（TE）与调度原语：`split` / `reorder` / `tile` / `fuse` / `vectorize` / `unroll` / `bind` / `compute_at` / `cache_read` / `cache_write`，用一个 matmul 把 CPU 与 GPU 的调度写出来、看生成的循环嵌套；
- **TensorIR**（TIR）：block 与 iter_var、`T.grid`、读写区域声明；schedule 原语作为 IR 变换，可 round-trip 打印；TIR 的 lowering 流水线（`LowerTVMBuiltin`、`StorageFlatten`、`ThreadSync`、`InjectDoubleBuffer`、`VectorizeLoop`）——与 Triton 的 pass 一一对应地放在一张表里；
- 自动调优：AutoTVM 的模板搜索、Ansor / MetaSchedule 的 sketch 生成与代价模型、搜索空间与 Triton `autotune` 的枚举空间在规模上差几个量级；`dlight` 的规则式调度；
- 图层：Relax 的 dataflow block、算子融合与 `FuseOps`、与 Triton 只做单 kernel 的边界；MLC-LLM 怎样用 TVM 把 LLM 编到手机和浏览器上；
- TVM 的 FFI 与运行时；TVM 在 v0.20 之后与 MLIR 生态的关系；
- **设计空间对照表**：谁决定 tile（用户 / 搜索 / 编译器）、谁决定线程映射、layout 是显式的还是隐式的、有没有图层、自动调优在哪一层、对新硬件的适配成本——Triton、Gluon、TVM、Halide、XLA / StableHLO、IREE、Inductor、CUTLASS / CuTe、nvcc 放在同一张表里。

核心问题是：

> **同一个 `[M, N, K]` 的 GEMM，Triton 用户决定 `BLOCK_M / BLOCK_N / BLOCK_K` 和 `num_warps`，其余交给编译器；TVM 用户（或 MetaSchedule）决定完整的循环嵌套、每一层的存储位置和线程绑定。两者的搜索空间分别有多大？各自把"编译器可能做错"的风险放在了哪里？**

### 13. 编译器开发者的工作台

第十三篇讲编译器开发者的日常方法，以 Triton 为主，MLIR 通用。

这一篇会覆盖：

- 从源码构建 Triton：LLVM 的 pin（`cmake/llvm-info.json`）与预编译包、`pip install -e .` 背后的 cmake 调用、增量构建、`TRITON_BUILD_WITH_CCACHE`；不需要 GPU 的构建（macOS 也可以）；
- `triton-opt`：对单个 `.mlir` 文件运行任意 pass 序列，`--mlir-print-ir-after-all`、`--mlir-print-ir-before=`、`--mlir-print-op-generic`、`--mlir-print-debuginfo`；`triton-tensor-layout` 与 `triton-llvm-opt`；
- lit 与 FileCheck：`test/` 目录的组织、`RUN:` 行、`CHECK` / `CHECK-NEXT` / `CHECK-DAG` / `CHECK-LABEL`、变量捕获 `[[X:%.*]]`；怎样从一个用户 kernel 抽出一个最小 `.mlir` 用例；
- 从一个错误结果定位 pass：`MLIR_ENABLE_DUMP=1` 拿到每个 pass 后的 IR、二分定位第一个变坏的 pass、`triton-reduce` 缩小用例、interpreter 模式验证语义；
- 调试工具：`TRITON_KERNEL_DUMP` / `TRITON_KERNEL_OVERRIDE`（手改 PTX 或 TTGIR 再喂回流水线）、`TRITON_ENABLE_LLVM_DEBUG` 与 `-debug-only=`、`op->dump()`、断点打在哪个 pattern；
- 读一个真实的 Triton PR：一个 layout / lowering 改动的 diff 结构——`.td` 改动、C++ pattern、lit 测试、Python 端测试各在哪里；
- 给 Triton 加一个 pass 的最小流程：`Passes.td` 注册、`.cpp` 实现、CMake、Python 绑定（`passes.ttgpuir.add_xxx`）、插进 `compiler.py` 的流水线、lit 测试；
- 贡献者的地图：Triton 社区的 RFC、`docs/` 里的设计文档、每周会议、issue 标签；MLIR 上游的 Discourse 与 review 流程。

核心问题是：

> **一个用户报告"某个形状的 kernel 结果错误"。从这个 Python 复现脚本出发，怎样在半小时内定位到是哪个 pass 引入的错误，并把它固定成一个 20 行的 lit 测试？**

### 14. 系列总结与通关自测

最后一篇不讲新内容：把十三篇正文压成一张「问题 → 结论 → 必记」的表并逐篇回顾——上面每篇导读末尾抛出的问题在那里逐条作答——拎出贯穿全系列的几条线与常见误区，然后给一套三段式通关自测——判断与计算、跨篇综合、面试题，答案各自折叠，附「读过 / 掌握 / 能教人」的判据。各篇末尾的自测检验的是一篇读懂了没有，这一篇检验的是十三篇能不能连起来用；读完正文再做。


## 贯穿全系列的源码阅读线

系列没有练手项目——编译器不是靠写一个玩具编译器学会的，而是靠读懂一个真实编译器的每个 pass、能在它的 IR 上动手。贯穿全系列的对象是**同一个 Triton matmul kernel**（BF16、`BLOCK_M=128, BLOCK_N=128, BLOCK_K=32`、`num_warps=4`、`num_stages=3`）：第五篇拿到它的 TTIR，第六篇打出每个值的 AxisInfo，第七篇看 Coalesce 给它的 layout，第八篇看 `AccelerateMatmul` 之后的 `#mma`，第九篇看流水化之后的循环，第十篇看它的 LLVM IR，第十一篇看 PTX 与 cache 目录。每一篇的 IR 都是上一篇的输出。

| 篇 | 项目 | 目录 / 文件 |
|---|---|---|
| 第一篇 | LLVM | `clang -emit-llvm`、`opt -passes=` 的输出 |
| 第二篇 | LLVM | `llvm/lib/Target/NVPTX/`、`llc -march=nvptx64` |
| 第三篇 | MLIR<br/>Triton | `mlir/include/mlir/IR/{Operation,Region,Block}.h`、`mlir/include/mlir/Dialect/SCF/IR/SCFOps.td`<br/>`include/triton/Dialect/Triton/IR/TritonOps.td`、`TritonGPU/IR/TritonGPUAttrDefs.td` |
| 第四篇 | MLIR<br/>Triton | `mlir/include/mlir/IR/PatternMatch.h`、`Transforms/DialectConversion.h`、`Analysis/DataFlowFramework.h`<br/>`lib/Dialect/Triton/Transforms/Combine.td` |
| 第五篇 | Triton | `python/triton/runtime/jit.py`、`compiler/code_generator.py`、`compiler/compiler.py`、`language/semantic.py`、`third_party/nvidia/backend/compiler.py` 的 `make_ttir` |
| 第六篇 | Triton | `include/triton/Analysis/AxisInfo.h`、`lib/Analysis/AxisInfo.cpp`、`test/Analysis/test-alignment.mlir` |
| 第七篇 | Triton | `lib/Tools/LinearLayout.cpp`、`lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp`、`lib/Conversion/TritonToTritonGPU/`、`lib/Dialect/TritonGPU/Transforms/Coalesce.cpp`、`bin/triton-tensor-layout.cpp` |
| 第八篇 | Triton | `Transforms/RemoveLayoutConversions.cpp`、`AccelerateMatmul.cpp`、`OptimizeDotOperands.cpp`、`OptimizeThreadLocality.cpp`；`test/TritonGPU/{combine,accelerate-matmul,dot-operands}.mlir` |
| 第九篇 | Triton | `Transforms/Pipeliner/`、`Prefetch.cpp`、`ReorderInstructions.cpp`、`WarpSpecialization/`；`lib/Dialect/TritonNvidiaGPU/Transforms/`；`python/triton/experimental/gluon/`；`test/TritonGPU/loop-pipeline*.mlir` |
| 第十篇 | Triton | `lib/Conversion/TritonGPUToLLVM/`、`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/`、`lib/Analysis/{Allocation,Membar}.cpp`；`test/Conversion/` |
| 第十一篇 | Triton | `python/triton/compiler/compiler.py`、`runtime/{cache,jit,driver}.py`、`knobs.py`、`third_party/nvidia/backend/{compiler.py,driver.py,driver.c}`；`third_party/amd/backend/compiler.py` |
| 第十二篇 | TVM | `python/tvm/tirx/`、`python/tvm/s_tir/schedule/`、`src/s_tir/schedule/primitive/`、`src/tirx/transform/`、`python/tvm/s_tir/meta_schedule/`、`python/tvm/relax/`、`docs/deep_dive/tensor_ir/` |
| 第十三篇 | Triton | `bin/`、`test/lit.cfg.py`、`unittest/`、`Makefile`、`python/test/unit/test_filecheck.py`、`CONTRIBUTING.md`、`.github/PULL_REQUEST_TEMPLATE.md`、`AGENTS.md` |

Table: 各篇的源码阅读对象

MLIR 的源码用 LLVM 23.1.1；Homebrew 的 `llvm` 包同时装好了 `mlir-opt`、`opt`、`llc` 和全部 `.td` 文件，前四篇的所有例子在一台没有 GPU 的笔记本上就能跑。Triton 的编译器部分在 macOS 上可以从源码构建（只是不能 launch kernel）：第五到十一篇的 `triton-opt` 与 `triton.compile` 产出的各层 IR、直到 PTX，都可以在本地生成；只有 cubin 与 launch 需要一块 NVIDIA 卡。


## 前置要求与说明

### 前置要求

- 读过《GPU Kernel 工程》系列，至少是第一、二、五、六、七篇：知道 warp、shared memory、bank conflict、`mma.sync`、`cp.async` 是什么，用 Triton 写过 kernel，读过 TTGIR 与 PTX 的 dump；
- 读过《PyTorch 深度实践》第七篇：知道 `torch.compile` 的三段与 Inductor 生成 Triton 的位置；
- 能读 C++：模板、继承与虚函数、RAII、lambda；MLIR 与 Triton 的 C++ 大量使用 CRTP 与 TableGen 生成的类，第三篇会解释这些生成代码的形状；
- 能读 Python：`ast` 模块、装饰器、类型注解；
- 线性代数：矩阵乘法、GF(2) 上的矩阵（第七篇会从定义讲起，不要求预先知道）。

不要求：

- 学过编译原理课程或读过龙书；
- 用过 LLVM、MLIR 或 TVM；
- 有 NVIDIA GPU——前十一篇的所有 IR 都能在笔记本上生成。

### 版本基线

- **Triton v3.8.0**（2026-08-28），源码阅读与所有 `triton-opt` / `triton.compile` 的输出以它为准；Gluon 与 Blackwell 相关内容标注它们在 3.4 → 3.8 之间的演进；
- **LLVM / MLIR 23.1.1**（Homebrew `llvm`）用于前四篇的独立例子。Triton 自己 pin 的是 `triton-lang/llvm-project` 上的一个特定 commit（`cmake/llvm-info.json`），比 23.1 略新；两者在本系列涉及的 API 上没有差别，有差别处随文标注；
- **TVM v0.26.0**（2026-08-05）用于第十二篇；
- **PyTorch 2.14.0** 用于 `torch.compile` 生成的 Triton kernel 作对照；
- **CUDA / PTX**：Triton v3.8.0 自带 ptxas 12.9（Blackwell 用 13.3）；架构讨论以 A100（sm_80）为基线，Hopper（sm_90）与 Blackwell（sm_100）随文标注。

### 关于 AMD 与其他后端

主线是 NVIDIA 后端；AMD 后端在第十一篇作为"另一套 `third_party/`"完整对照一次，说明哪些 pass 是共享的、哪些是各自的。其他厂商基于 Triton 的后端（Intel XPU、各家 ASIC）遵循同一套插件机制，不单独展开。


## 章节目录

1. [编译器的骨架：IR、SSA 与 pass](/compiler-skeleton-ir-ssa-and-passes.html)
2. [LLVM：所有 ML 编译器共同的后端](/llvm-the-shared-backend-and-nvptx.html)
3. [MLIR（上）：Operation、Region、Dialect](/mlir-ir-structure-dialects-and-ods.html)
4. [MLIR（下）：Pass、Pattern Rewrite 与 Dialect Conversion](/mlir-passes-pattern-rewriting-and-dialect-conversion.html)
5. [Triton 编译器（一）：从 Python AST 到 TTIR](/triton-compiler-frontend-python-ast-to-ttir.html)
6. [Triton 编译器（二）：AxisInfo——一切向量化决定的地基](/triton-compiler-ttir-analysis-axisinfo.html)
7. [Triton 编译器（三）：layout 系统与 Linear Layout](/triton-compiler-layouts-and-linear-layout.html)
8. [Triton 编译器（四）：layout 优化与 Tensor Core 路径](/triton-compiler-layout-optimization-and-tensor-cores.html)
9. [Triton 编译器（五）：软件流水、异步与 warp specialization](/triton-compiler-software-pipelining-hopper-blackwell-gluon.html)
10. [Triton 编译器（六）：从 TritonGPU 到 LLVM](/triton-compiler-lowering-tritongpu-to-llvm.html)
11. [Triton 编译器（七）：PTX、cubin 与运行时](/triton-compiler-ptx-cubin-cache-runtime-and-amd.html)
12. [TVM：调度语言与自动调优——另一条路](/tvm-schedule-language-and-auto-tuning.html)
13. [编译器开发者的工作台](/ml-compiler-developer-workbench.html)
14. [系列总结与通关自测](/ml-compilers-series-recap-and-self-test.html)


## 最终目标

读完这套系列之后，面对一个 Triton kernel 的任何一层 dump，读者应该能够回答：

| 问题 | 看什么 |
|---|---|
| 这个张量为什么是这个 layout？ | Coalesce 的 AxisInfo 输入、RemoveLayoutConversions 的传播方向 |
| 这个 `convert_layout` 为什么没被消掉？ | 两端的锚点、重物化的代价模型 |
| 这条 load 为什么不是 128 bit？ | AxisInfo 的 contiguity / divisibility 在哪一步丢了 |
| 这个循环为什么没被流水化？ | AssignLatencies 的条件、依赖距离 |
| 这个 `bar.sync` 是谁插的？ | Membar 分析看到的 shared memory 依赖 |
| 这个 kernel 为什么 spill？ | ptxas 的寄存器分配、Triton 侧能影响的 layout 与 `num_warps` |
| `tl.dot` 为什么没走 `wgmma`？ | AccelerateMatmul 的版本选择条件、操作数 layout 的合法性 |
| 换一个后端要改哪些文件？ | `third_party/` 的插件边界、`TargetInfo` |
| 这个优化在 TVM / XLA 里是怎么做的？ | 设计空间表上的位置 |

Table: 读完系列后应能回答的问题

最终目标是三种能力：

1. **阅读能力**：读懂 Triton、MLIR 与 TVM 的编译器源码，知道每个 pass 在流水线里的位置、输入输出和它依赖的分析；
2. **诊断能力**：从一个性能或正确性问题出发，在 IR 上定位到具体的 pass 和 pattern，而不是靠调参数试；
3. **改动能力**：给一个 pass 加一条 pattern、修一个 layout 组合的 lowering、为一个新硬件特性接入一条路径，并用 lit 测试固定它。

这是 AI-Infra 执行平面里最"编译器"的一层。它是选修，因为大多数 Infra 工作用不到；但一旦用到——kernel 性能到了编译器决定的那 10%、新硬件要接 Triton、`torch.compile` 生成的代码不对——它就是唯一的路。
