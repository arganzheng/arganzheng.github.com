---
layout: post
series: ml-compilers
title: "ML 编译器内部（14）：系列总结与通关自测"
subtitle: "ML Compiler Internals: Series Recap and Final Self-Test"
tags: [Compiler, MLIR, LLVM, Triton, TVM, GPU, AI-Infra]
catalog: true
date: 2026-12-04 20:00:00
---

十三篇正文回答了总纲提出的一个问题：**一段块级的张量程序，怎样一步步变成一条条 GPU 指令，中间每一个决定是哪个 pass 凭什么分析做的。** 前四篇讲通用机制（IR / SSA / pass、LLVM、MLIR 的结构与变换），中间七篇顺着 Triton v3.8.0 的编译流水线自上而下逐站打开（前端、AxisInfo、layout、layout 优化、流水与异步、下降到 LLVM、缓存与运行时），第十二篇用 TVM 做对照，第十三篇讲工作台。全部真实 IR 在一台没有 NVIDIA GPU 的 Mac 上跑出来。这一篇不讲新内容：总表、逐篇回顾、贯穿的线、误区、三段式自测，最后是 Infra 地图的收束。

> **读完这十三篇，你应该能回答哪些问题？[^q0] 哪些数字与结论必须能脱口而出？[^q1] 怎么判断自己是"读过"还是"掌握"了？[^q2]**

## 一、总览：系列回答的问题与主线

系列的一句话主张是：**一个编译器 = IR + pass + lowering；Triton 把 layout 放进类型、把决定放进 pass、把正确性放进 verifier 与分析，每一个可观察的编译结果（一条 `ld.global.v4`、一个 `bar.sync`、一个 `convert_layout`、两个 shared memory 缓冲）都能追到某一个 pass 与它依赖的某一个分析。**

| 篇 | 回答的问题 | 一句话结论 | 必记的数字 / 结论 |
|---|---|---|---|
| [01 编译器的骨架](/compiler-skeleton-ir-ssa-and-passes.html) | 编译器由什么组成，"优化"是什么？ | 前端 / IR / pass / 后端；SSA 让 def-use 成为数据结构，φ 在汇合处；数据流分析 = 格 + 传递函数 + 不动点；pass 的"变换 → 清理"节奏；渐进式下降每层丢一类信息；ML 编译器为块级张量语义另起 IR | `mem2reg` 后 `sum` 的两个 φ；`-O2` 是 119 项 pass、126 次 dump；标量 IR 丢了"这是矩阵乘" |
| [02 LLVM](/llvm-the-shared-backend-and-nvptx.html) | 所有 ML 编译器共用的后端做什么、不做什么？ | IR 四层容器、一切是 `Value`；`-O2` 的 cgscc / function / loop 嵌套；后端三件事；**NVPTX 不做寄存器分配**，`ptxas` 是第二个编译器；`align` 决定向量宽度；内联汇编让 LLVM 对 `mma` 无知 | `align 16` → 一条 `ld.global.v4`、`align 4` → 四条 `b32`；地址空间 1 / 3 / 5；`ptxas -v --regAllocOptLevel=2`；`n_regs` 来自 `cuFuncGetAttribute`；AMD 直接打印 `NumVgprs / Occupancy` |
| [03 MLIR（上）](/mlir-ir-structure-dialects-and-ods.html) | MLIR 的 IR 长什么样、怎么声明？ | 基础设施 / 内容分离；Operation 唯一单位，Region / Block / Value；通用形式 = 数据结构打印；ODS 十几行生成五百行；Trait 静态、Interface 动态；Block 参数代替 φ；Type / Attribute 唯一化不可变 → 改 layout 只能建新 op | LICM 用 4 个接口；`scf → cf` 后 Block 参数与 `mem2reg` 的 φ 逐行对应；`RankedTensorType` 的 encoding 槽位 |
| [04 MLIR（下）](/mlir-passes-pattern-rewriting-and-dialect-conversion.html) | 怎样改 IR？ | pass（`builtin.module(func.func(...))` 嵌套、analysis 默认失效）；pattern rewrite（改动全经 rewriter、greedy 到不动点、`fold` 不建 op、canonicalize 弱于 instcombine）；dialect conversion（类型一起变、adaptor、materialization、延迟提交 / 回滚）；数据流框架；`Passes.td → passes.cc → compiler.py` 链 | `ConvertTritonToTritonGPU` 的 TypeConverter 加默认 `#blocked`、target materialization = `convert_layout`；`unrealized_conversion_cast` 是未完成的痕迹；`linalg.matmul` 在笔记本上跑出 16 |
| [05 Triton 前端](/triton-compiler-frontend-python-ast-to-ttir.html) | Python AST 怎样变成 TTIR？ | `JITFunction` 保存源码不执行；特化 key（dtype、constexpr、16 的倍数、等于 1）；`CodeGenerator` 是 `ast.NodeVisitor`，每个节点一次 builder 调用；`constexpr` 折叠 vs 物化；`for` → `scf.for`（空跑找携带值）；`make_ttir` 八个 pass 全是 pattern | `tt.divisibility = 16` 从实参对齐来；`x_ptr + offs` 是 `tt.addptr`（类型化地址、AxisInfo 的对象）；`Combine` 模式认形状不认语义（隔着 broadcast 不合并） |
| [06 AxisInfo](/triton-compiler-ttir-analysis-axisinfo.html) | 编译器凭什么知道一条 load 可以 128 bit？ | 三个量 contiguity / divisibility / constancy 的格与 gcd join；逐 op 传递规则；`splat + arange` 是 contiguity 的唯一来源；参数属性是 divisibility 的来源；`getVectorSize = min(128 / 位宽, contiguity, alignment)`、mask 用 constancy | `pid * 1024 + arange(1024)` → `[1024], [1024], [1]`；换成 `pid * n` → divisibility 1；`multiple_of(n, 16)` → 16；`a_ptrs` 的 `[1, 32], [2, 16]` → alignment 8 |
| [07 layout 系统](/triton-compiler-layouts-and-linear-layout.html) | layout 是什么、怎样统一？ | layout = 硬件位置 → 张量下标的函数，在类型里；`#blocked` 智能构造；`#slice` / `#mma` / `#dot_op`；Linear Layout = GF(2) 线性映射，基向量表；转换代价由 `dst⁻¹∘src` 逐维 `quotient` 判定；默认 layout 每线程 1 元素；Coalesce 按 AxisInfo 定 `order` 与 `sizePerThread` | `[64, 64]` 默认 `#blocked`：lane 管列低 5 位、warp 位 2 管行 1、寄存器管行 2..32；到 `#mma` 需 shared memory（warp 位 → lane 位）；A tile → `[1, 8], [8, 4], [4, 1]` |
| [08 layout 优化](/triton-compiler-layout-optimization-and-tensor-cores.html) | 二十多个 `convert_layout` 怎样变成一个？ | `RemoveLayoutConversions` 两阶段：锚点前向传播 + 冲突消解 + 后向重物化（代价模型）+ 三种 hoist；`AccelerateMatmul` 选 MMA 版本、`warpsPerCTA` 偏向 M、`kWidth = 32 / 位宽`；epilogue 转换被 `dot` 与 `store` 两个锚夹住 | 23 → 3 → 7 → 3 → 1；`[128, 128]` → `warpsPerCTA = [2, 2]`；`#mma → #dot_op` 在 `[4, 1]` 下零 shared memory、`[2, 2]` 下 8 条 `st.shared` + 15 barrier |
| [09 流水与异步](/triton-compiler-software-pipelining-hopper-blackwell-gluon.html) | `num_stages` 变成了什么？ | `AssignLatencies`（提前 `(S−1)/(层级+1)` 个迭代）→ `ScheduleLoops`（最长 latency 路径分 stage）→ `LowerLoops`（缓冲数 = stage 差）→ `PipelineExpander`；Hopper `wgmma` 异步 +1 缓冲、TMA + mbarrier；Blackwell TMEM + `tcgen05`；`warp_specialize` 三区；Gluon 跳过全部自动 pass | `num_stages = 3`：2 缓冲、prologue 2 次 load、`async_wait {num = 2}`、15 个 iter_args；Hopper 3 缓冲、`pendings = 1`、iter_args 4；WS：默认区 4 warp epilogue、1 warp MMA、2 warp TMA |
| [10 下降到 LLVM](/triton-compiler-lowering-tritongpu-to-llvm.html) | tile 级 op 怎样变成每线程指令？ | 张量 → 每线程 struct；`applyLinearLayout` 把基向量 XOR 成地址算术；`load` 按 `vec` 发谓词 `ld.global.vN`；`reduce` 三级由规约维落在 register / lane / warp 位决定；`mma.sync` 内联汇编与 `ldmatrix`；`convert_layout` 三条路；`AllocateSharedMemory` 着色复用；`Membar` 区间相交插 barrier | rowsum：7 加 + 5 shuffle + 16 字节；matmul 64 条 `mma.sync`、26 条 `ldmatrix`、24 条 `cp.async`、10 个 barrier；流水线缓冲与 epilogue scratch 同为 32 KB、`ttg.shared = 32768` |
| [11 缓存与运行时](/triton-compiler-ptx-cubin-cache-runtime-and-amd.html) | 编译流水线怎样组织、产物去哪、AMD 有什么不同？ | `compile()` 按阶段表逐级、`metadata` 累积、缓存组原子；key 五成分；dump / override；惰性加载、运行时生成的 C launcher、`cuLaunchKernelEx`；AMD：layout pass 共用、`#amd_mfma`、自己的流水器、LLVM 直出 ISA、无第二编译器 | `triton_key` 含 `libtriton.so` 哈希；34 个元数据字段；block = `32 × num_warps`；gfx942：wave 64、MFMA 32×32×8、144 VGPR、0 spill、`Occupancy 3` |
| [12 TVM](/tvm-schedule-language-and-auto-tuning.html) | 另一条路怎么走？ | 算法 / 调度分离；block 是算法、循环是调度；原语带形式化前置条件；DLight 规则 = Triton 自动决定的显式版；MetaSchedule 搜带采样点的 trace；设计空间是一条轴 | CPU 1089 → 52 μs；Metal 700 GFLOP/s；DLight `storage_align(…, 16, 8)` 代替 swizzle；MetaSchedule 空间 10⁴–10⁶ vs Triton autotune 几十个 |
| [13 工作台](/ml-compiler-developer-workbench.html) | 怎样改它、测它、定位它？ | macOS 可构建，假 `ptxas` 编到 PTX；lit 277 文件 9.5 s、gtest 毫秒、pytest 需 GPU；`MLIR_ENABLE_DUMP` 74 份；二分八步；lit 两种断言；加 pass 七步；读 PR 倒序 | `lit < 20`；`--run-reproducer`；`TRITON_INTERPRET=1` 是编译器 / kernel 的分界线 |

### 1. 本文的章节安排

| 章 | 内容 |
|---|---|
| 二 | 逐篇回顾：每篇的结论与常见误解，并回答总纲的分章问题 |
| 三 | 贯穿十三篇的几条线与概念表 |
| 四 | 常见误区表 |
| 五 | 通关自测：A 判断与计算 10 题、B 跨篇综合 5 题、C 面试题 7 题、D 掌握判据 |
| 六 | Infra 地图的收束与下一步 |
| 七 | 延伸阅读 |

## 二、逐篇回顾

### 1. 第一篇：编译器的骨架

**结论**：编译器是前端 / IR / 一串 pass / 后端；IR 是让分析与变换可写的数据结构，SSA 把"值从哪来"变成指针（`mem2reg` 后 `sum` 的局部变量消失、两个 φ 出现在循环头）；数据流分析是格上的不动点，join 保守；pass 的节奏是"变换 → 清理"；渐进式下降每一层丢掉一类信息，所以优化要在信息还在的那一层做。**总纲的问题**：`a * b + c` 在 LLVM 里是三条标量指令，优化是 CSE / 强度削减 / 融成 FMA；在 Triton 里是三个块级 op 操作 `tensor<128x64xf32>`，优化是选 layout、合并访存、喂 Tensor Core、流水——后者在前者上做不出来，因为标量 IR 已经丢了"这是同一个张量的 8192 个独立元素"这条信息，重新发现它（多面体分析）代价大且不可靠。

**常见误解**：SSA 是一种优化（它是表示）；φ 是运行时指令；"下降"就是翻译（每一步都在丢信息与做决定）。

### 2. 第二篇：LLVM

**结论**：LLVM IR 四层容器、`Value` 统一 use-def；attribute 是承诺、metadata 是提示；`-O2` 是 119 项的树，内联在 cgscc 里自底向上，向量化最后；后端三件事里 NVPTX 不做寄存器分配；`align` 而不是向量类型决定访存宽度；Triton 的 `-O3` 没有 TargetMachine，加了拆 struct φ 的 pass。**总纲的问题**：寄存器数由 `ptxas` 决定，在 `make_cubin` 那一步（PTX 用无限虚拟寄存器），Triton 只能间接影响——`num_warps` 减半每线程元素、`num_stages` 减在飞 load、tile 减累加器、`maxnreg` 强制上限。

**常见误解**：PTX 是机器码；`.reg .b32 %r<9>` 是寄存器用量；`n_spills` 是 `ptxas` 报的（它是加载后 `cuFuncGetAttribute` 查的 local memory 字节 / 4）。

### 3. 第三篇：MLIR（上）

**结论**：MLIR 把基础设施与内容分开；Operation 是唯一单位，通用形式就是数据结构；ODS 从 `arguments / results / traits / assemblyFormat` 生成类、访问器、builder、verifier、parser / printer；Trait 与 Interface 是 pass 与 op 的契约；Block 参数代替 φ；Type / Attribute 唯一化。**总纲的问题**：LICM 需要"哪里是循环、哪些值是循环自己的"（`LoopLikeOpInterface`）、"操作数是否循环外定义"（`isDefinedOutsideOfLoop`）、"有无内存效应"（`MemoryEffectsOpInterface`——通用 LICM 不做别名分析，地址不变的 load 也不提）、"能否推测执行"（`ConditionallySpeculatable`）；不用 φ 是因为 φ 是需要特例的假指令，Block 参数与函数参数、Region 参数是同一机制，结构化控制流下汇合由 `RegionBranchOpInterface` 描述。

**常见误解**：方言是"语言"（它是命名空间 + 一组定义）；`func.func` 是特殊容器（它是带 Region 的 Op）；改一个值的 layout 可以改类型（类型不可变，只能建新 op）。

### 4. 第四篇：MLIR（下）

**结论**：三种变换机制由简到繁；pattern 的纪律是改动全经 rewriter、`fold` 不建 op；greedy 到不动点、有 `max-iterations` 保险；dialect conversion 允许中间状态类型不一致，adaptor 传已转换操作数、materialization 桥接、最后提交或回滚。**总纲的问题**：加 layout 是全局类型替换，`tt.load` 一变 `tt.dot` 就非法、`scf.for` 四处要同步——不存在每步合法的顺序，greedy 做不了；TypeConverter 定义"无 encoding 的张量 → 默认 `#blocked`"、其他类型不变，并给出 target materialization = `ttg.convert_layout`。

**常见误解**：canonicalize 等于 instcombine（`(x*8+7)-x*8` 它不化）；`unrealized_conversion_cast` 是 bug（它是未完成转换的正常痕迹，`reconcile` 消掉）；pass 顺序无关紧要。

### 5. 第五篇：Triton 前端

**结论**：`@triton.jit` 保存源码；调用时算特化 key（dtype、constexpr 值、整数是否 16 的倍数、是否等于 1）；`CodeGenerator` 逐 AST 节点调 builder，`semantic` 层做类型提升与广播；`constexpr` 在 Python 层折叠、进 IR 前物化；`for` 空跑一遍找携带值再建 `scf.for`；`make_ttir` 八个 pass 全是 pattern。**总纲的问题**：实参信息经 `native_specialize_impl` 变成 `attrs`（`tt.divisibility = 16` 挂在参数上）与 `constants`（constexpr 值），进 `ASTSource`、进 IR 的函数参数属性与常量；`x_ptr + offsets` 是 `tt.addptr` 而不是 `arith.addi`，因为它是类型化的地址运算——AxisInfo 对指针与整数的规则不同、`getelementptr` 的缩放由后端做。

**常见误解**：Triton 执行 Python 函数（它遍历 AST）；`constexpr` 只是"常量"（它决定特化与缓存 key）；`Combine` 会合并所有能合并的地址运算（它匹配形状，隔着 `broadcast` 的两个 `addptr` 不合）。

### 6. 第六篇：AxisInfo

**结论**：每个整数 / 指针值三个量，格上 gcd 做 join；`make_range` 是 contiguity 的源、参数属性是 divisibility 的源、`splat` 是 constancy 的源；加法取 gcd、乘法相乘、`expand_dims / broadcast` 按维处理；循环用不动点。**总纲的问题**：`pid * 1024 + arange(1024)`——`pid` divisibility 1 × 1024 = 1024，`arange` contiguity 1024 且 divisibility 2³⁰，`splat + arange` 的 contiguity = gcd(constancy 1024, contiguity 1024) = 1024、divisibility = gcd(1024, 2³⁰, 1024) = 1024；换成 `pid * n`（`n` 无属性）→ divisibility 1、contiguity 仍 1024 但 alignment 1，load 退化为标量；加 `tl.multiple_of(n, 16)` → divisibility 16，128 bit 恢复。

**常见误解**：contiguity 大就能向量化（还要 divisibility 对齐）；mask 不影响向量化（constancy 限制谓词粒度）；AxisInfo 是 TTGIR 才有的（它在 TTIR 上就能算，Coalesce 与 lowering 各重算一次）。

### 7. 第七篇：layout 系统

**结论**：layout 在类型里，verifier 保一致，只有 `convert_layout` 能改；`#blocked` 智能构造让最快维被最少线程铺满；`#slice` 是父 layout 压掉一维；`#mma` 是硬件 fragment；`#dot_op` 随累加器定；Linear Layout 用基向量表统一一切，转换代价由 `quotient` 判定；默认 layout 每线程 1 元素；Coalesce 按 AxisInfo 定 `order` 与 `sizePerThread`、前后插转换。**总纲的问题**：`[64, 64]` 默认 `#blocked` 的基向量——lane 5 位 → 列 1..16、warp 位 1 → 列 32、warp 位 2 → 行 1、寄存器 5 位 → 行 2..32；到 `#mma` 必须过 shared memory，因为源的 warp 位 2 与寄存器位（行 1、2、4）在目标里是 lane 位 4、8、16——`quotient(warp)` 失败。

**常见误解**：layout 是运行时的东西（它全在编译期）；`#slice` 是一种独立布局（它是父 layout 的函数）；swizzle 是 layout 之外的一步（在 LL 里它就是线性映射）。

### 8. 第八篇：layout 优化

**结论**：`RemoveLayoutConversions` 从锚点（昂贵访存、`dot`、atomic、TMA、TMEM）前向传播，冲突时访存保 `#blocked`、其他偏 `#mma`，后向重物化按代价，三种 hoist；`AccelerateMatmul` 选版本、平衡 `warpsPerCTA`、定 `kWidth`；Hopper 操作数进 shared memory。**总纲的问题**：epilogue 的 `#mma → #blocked` 一端是 `dot`（硬件 fragment、不可重物化）、另一端是 `store`（Coalesce 的合并访存 layout、昂贵访存是锚点），中间只有 `truncf`，冲突消解让它留在 `#mma`；attention 的 P 从 `#mma` 到 `#dot_op<{opIdx = 0}>` 在 `warpsPerCTA = [4, 1]` 下是纯寄存器重排（C 与 A fragment 的 lane 位相同），所以 `warpsPerTileV2` 对链式 dot 把 warp 全放 M 上。

**常见误解**：`convert_layout` 都是昂贵的（三条路里一条零成本）；`RemoveLayoutConversions` 是 pattern 集（它是 1600 行两阶段算法）；`AccelerateMatmul` 直接改上游 load 的 layout（它只改 `dot` 自己，留给下一次 `RemoveLayoutConversions`）。

### 9. 第九篇：流水与异步

**结论**：软件流水四步各做一个决定；`num_stages` 数 stage、缓冲数是 stage 差；`Prefetch` 拆 K 子块；Hopper `wgmma` 操作数在 `#nvmma_shared`、异步、多一个缓冲，TMA 用 mbarrier 事务计数、iter_args 从 15 降到 4；Blackwell 累加器进 TMEM、`tcgen05` 单线程发起；`warp_specialize` 默认区 + 分区；Gluon 显式一切、跳过自动 pass。**总纲的问题**：`num_stages = 3`——prologue 2 次 load、`async_wait {num = 2}` 等最老的一批、2 个缓冲；Hopper 3 个缓冲、mbarrier 是 MMA warp 等 TMA 引擎（`barrier_expect` 声明字节数，phase 区分轮次）；warp specialization 把 TMA 给 2 个 warp、MMA 给 1 个 warp、epilogue 留给默认区 4 个 warp。

**常见误解**：`num_stages` 等于缓冲数；`wait_group N` 等待 N 组（它允许 N 组在飞）；warp specialization 是流水的替代（它是流水在空间上的版本，两者叠加）。

### 10. 第十篇：下降到 LLVM

**结论**：张量变每线程 struct；LL 展开成 XOR 地址算术；`load` 按 `vec` 与谓词发内联 PTX；`reduce` 三级；`mma.sync` 的约束串与 fragment 打包；`ldmatrix` 地址由两个 LL 复合；`convert_layout` 三条路；`AllocateSharedMemory` first-fit 着色复用；`Membar` 沿虚拟块维护读写区间、RAW / WAR 相交插 barrier、`MemWaitOpTrait` 后必插。**总纲的问题**：所有 `bar.sync` 由 `Membar` 依据 `AllocateSharedMemory` 的区间与 `MemoryEffectsOpInterface` 的读写插入；`reduce` 的三级由规约维落在 register / lane / warp 位决定——rowsum 7 加、5 shuffle、一次 shared memory 交换。

**常见误解**：barrier 是 lowering pattern 自己插的（是一个独立分析）；shared memory 总量是各缓冲之和（生命期不重叠的复用地址）；`cp.async.wait_group` 之后不用 barrier（它只保证本线程的拷贝）。

### 11. 第十一篇：缓存与运行时

**结论**：`compile()` 后端无关、`metadata` 累积、缓存组原子；key 五成分；dump / override 以 `src.hash` 为目录；惰性加载、按签名生成的 C launcher、`cuLaunchKernelEx` 属性；AMD 共用 layout pass、自己的流水器、LLVM 直出 ISA。**总纲的问题**：不同 `BLOCK_SIZE`（constexpr → `src.hash`）、stride 为 1 / 16 的倍数 / 其他（特化 → `src.hash`）、A100 / H100（`backend.hash` 的 arch）各一个目录，相乘最多 12 个；改 pass 旧缓存命中是因为 `triton_key` 哈希的是 Python 代码与 `libtriton.so`——没重新构建（或装错 venv）就不变；失效手段是删缓存、`TRITON_ALWAYS_COMPILE`、改一个登记的环境变量。

**常见误解**：所有环境变量都影响缓存 key（只有名单上的）；`.json` 里有 `n_regs`（加载时才查）；AMD 也有 `ptxas` 这一级（没有）。

### 12. 第十二篇：TVM

**结论**：算法 / 调度分离，block 是算法、循环是调度；`S / R` 轴标注是合法性检查的信息源；原语带形式化前置条件（`vectorize` 在 `decompose_reduction` 后被拒）；DLight 规则把 Triton 的自动决定写成显式原语；MetaSchedule 搜带 `sample_*` 的 trace；lowering 无启发式，`ThreadSync` 插 barrier、`vectorize` 显式给 `float4`。**总纲的问题**：Triton 的空间是 autotune 的几十个 config，TVM MetaSchedule 是 10⁴–10⁶ 条 trace；Triton 把风险放在编译器启发式（用户无旋钮），TVM 放在调度作者 / 规则覆盖 / 搜索预算（编译器不会错只会慢）；正确性放在原语里才敢搜大空间。

**常见误解**：TVM 是"自动"的（自动化是调度层上面的 DLight / MetaSchedule，编译器本身忠实执行）；调度写错会生成错误代码（原语拒绝）；TVM 生成 PTX（它生成 CUDA C，`nvcc` 是后端）。

### 13. 第十三篇：工作台

**结论**：Triton 编译器在 macOS 上可构建，假 `ptxas` 让 `compile()` 跑到 PTX，AMD 路径无需工具；三层测试的成本差三个量级；`MLIR_ENABLE_DUMP` 给出完整 pass 表；二分定位八步、大半不需要 GPU；lit 的两种断言与最小化原则；加 pass 七步；读 PR 倒序。**总纲的问题**：固定复现 → 排除 LLVM / `ptxas` → dump 每级 IR → 解释器判界 → 用 `triton.compile("X.ttgir")` 二分 74 份 IR → 锁定 pass、diff → 最小化到 10–30 行 → 写 lit 并确认先失败。

**常见误解**：改编译器必须有 GPU；lit 测试要从 Python 生成（模板明说那不是最小）；pytest 是唯一的测试层。

## 三、贯穿全系列的几条线

### 1. 一个 kernel 的完整路径

matmul kernel（`[128, 128, 32]`、bf16、`num_warps = 4`、`num_stages = 3`、`sm_80`）逐站的可观察量：

| 站 | 篇 | 输入 → 输出 | 关键数字 |
|---|---|---|---|
| 前端 | 05 | Python AST → TTIR | 15 个参数剩 9 个；两个 `addptr` 没合并；`inputPrecision = tf32` |
| AxisInfo | 06 | TTIR → 每个值的三元组 | `a_ptrs`：`[1, 32], [2, 16], [1, 1]` → alignment 8；mask `[16, 16]` |
| 初始 layout | 07 | TTIR → TTGIR（默认 `#blocked`） | 16 个 `convert_layout` |
| Coalesce | 07 | load / store 换 layout | A `[1, 8], [8, 4], [4, 1]`，B / C `[1, 8], [2, 16], [4, 1]`；23 个转换 |
| RemoveLayoutConversions ① | 08 | 传播 + 重物化 | 3 个 |
| AccelerateMatmul | 08 | `dot` 进 `#mma<{[2, 2], [16, 8]}>`、`#dot_op<{kWidth = 2}>` | 7 个 |
| RemoveLayoutConversions ② | 08 | | 3 个（两个 `dot` 操作数、一个 epilogue） |
| Pipeline + Prefetch | 09 | `scf.for` 展开 | 2 缓冲、prologue 2 次 load、`async_wait {num = 2}`、K 拆两半、15 个 iter_args；1 个 `convert_layout` |
| AllocateSharedMemory | 10 | 偏移 | `ttg.shared = 32768`（流水线缓冲与 epilogue scratch 复用） |
| TritonGPUToLLVM | 10 | TTGIR → LLVM 方言 | 64 `mma.sync`、26 `ldmatrix`、24 `cp.async`、10 barrier、16 `st.global.v4`、0 `ld.global` |
| LLVM `-O3` → PTX | 02 / 11 | `translate_to_asm` | `.version` / `.target` 正则；`cp.async.wait_group 2` |
| `ptxas` → cubin | 02 / 11 | 子进程 | `-lineinfo -v --regAllocOptLevel=2 --gpu-name=sm_80`；`n_regs` 加载时查 |
| Hopper | 09 | 同源码 `sm_90` | `#mma v3 [4, 1] [16, 128, 16]`、`#nvmma_shared` 64 / 128 字节、3 缓冲、`warp_group_dot {isAsync}`、`shared = 49152` |
| AMD | 11 | 同源码 `gfx942` | wave 64、`#amd_mfma<{version = 3, [2, 2], [32, 32, 8]}>`、48 `v_mfma`、144 VGPR、0 spill、`Occupancy 3`、16 KB LDS |

### 2. 四条线

- **IR + pass + lowering**：每一站都是"读 IR 的某种信息 → 做一个决定 → 写回 IR"；信息在哪一层还在，决定就要在哪一层做（第一篇 §八 → 第七篇把 layout 放进类型 → 第九篇把 `scf` 保留到最后）。
- **局部改写 + 前后隔离 + 全局清理**：Coalesce、AccelerateMatmul、`ConvertTritonToTritonGPU` 都只改自己的 op、用 `convert_layout` 隔离，`RemoveLayoutConversions` 统一收拾——每个 pass 保持简单，正确性靠 verifier 与不变量。
- **接口让分析与 op 解耦**：LICM 不认识 `scf.for`（第三篇），`Membar` 不认识 `cp.async`（第十篇），AMD 后端复用全部 layout pass（第十一篇）——靠 `LoopLikeOpInterface`、`MemoryEffectsOpInterface`、`DistributedEncodingTrait`。
- **谁决定**：从 XLA / Inductor（编译器全决定）到 Triton（用户定 tile）到 TVM 调度 / CUTLASS / Gluon（用户全决定）是一条轴；风险随位置移动；每个系统都在补另一端（第十二篇 §九）。

### 3. 概念表

| 概念 | 一句话 | 篇 |
|---|---|---|
| SSA / φ / Block 参数 | 每个值定义一次；汇合处的选择；MLIR 用跳转实参代替 φ | 01 / 03 |
| 数据流分析 / 格 / join | 有限高度格上的不动点；join 保守 | 01 / 04 / 06 |
| pass / pattern / conversion | 变换单元；局部形状改写；全局类型替换 | 04 |
| Trait / Interface | 静态性质 / 可查询方法集；pass 与 op 的契约 | 03 |
| AxisInfo | contiguity / divisibility / constancy | 06 |
| layout / encoding | 硬件位置 → 张量下标，在类型里 | 07 |
| Linear Layout | GF(2) 上的线性映射，基向量表 | 07 |
| 锚点 / 重物化 | layout 不能改的 op；把源计算在目标 layout 下重算 | 08 |
| stage / cluster / latency | 迭代提前量、op 的阶段与顺序 | 09 |
| mbarrier / phase | 硬件计数同步对象；复用时的轮次位 | 09 |
| scratch / 着色 | op 需要的临时 shared memory；不重叠的复用地址 | 10 |
| RAW / WAR | 区间相交的两种依赖；barrier 的来源 | 10 |
| 缓存 key 五成分 | `triton_key`、源、后端、选项、环境变量 | 11 |
| SBlock / S·R 轴 / 原语 | TVM 的算法单元、轴类型、保语义变换 | 12 |
| trace / `sample_*` | MetaSchedule 的搜索空间表示 | 12 |

## 四、常见误区

| 误区 | 事实 | 篇 |
|---|---|---|
| Triton 编译器决定寄存器数 | `ptxas` 决定；Triton 只能经 `num_warps` / `num_stages` / tile / `maxnreg` 间接影响 | 02 |
| `.llir` dump 是 lowering 刚出来的 IR | 是 `-O3` 之后的；优化前要看 MLIR LLVM 方言那一步 | 02 / 13 |
| canonicalize 会做所有代数简化 | 只做规范化；`(x*8+7)-x*8` 留给 LLVM | 04 |
| `Combine` 合并所有地址运算 | 匹配 IR 精确形状；隔着 `broadcast` 不合 | 05 |
| contiguity 大就 128 bit | 还需 divisibility 对齐与 mask constancy | 06 |
| `sizePerThread` 是用户选的 | Coalesce 由 AxisInfo 推出；用户只能通过 `multiple_of` 等提示影响 | 07 |
| 所有 `convert_layout` 走 shared memory | 三条路；`#mma → #dot_op` 在 `[4, 1]` 下零成本 | 07 / 08 |
| epilogue 的转换是优化没做好 | 两端被硬件决定锚住，消不掉；Gluon 或 TMA store 是绕法 | 08 / 09 |
| `num_stages = 3` → 3 个缓冲 | 缓冲数 = stage 差 = 2；Hopper 因 `wgmma` 异步 +1 | 09 |
| TTGIR 里有 barrier | 一个没有；全由 `Membar` 推出 | 10 |
| shared memory 用量 = 各缓冲之和 | 生命期不重叠的复用；matmul 32 KB 不是 64 KB | 10 |
| 改 `.cpp` 就生效 | 要重新构建且装进 import 到的那份 Triton；否则 `triton_key` 不变、旧缓存命中 | 11 / 13 |
| TVM 调度写错会算错 | 原语拒绝不合法变换；只会慢 | 12 |
| 改编译器必须有 GPU | lit、gtest、`triton-opt`、编到 PTX 与 AMD ISA 都不需要 | 13 |

## 五、通关自测

### A. 判断与计算（10 题）

1. 判断：`mem2reg` 之后 IR 里不再有内存操作。

   <details markdown="1"><summary>答案</summary>
   错。它只提升可提升的 `alloca`（地址未逃逸、只被 load / store）；数组参数 `a[i]` 的 `load` 仍是内存操作。`sum` 的例子里 `%0 = load i32, ptr %arrayidx` 留下了。
   </details>

2. 计算：`tensor<256x64xbf16>`、`num_warps = 8`。默认 `#blocked` 的 `threadsPerWarp` / `warpsPerCTA` 是什么？每线程几个寄存器（元素）？

   <details markdown="1"><summary>答案</summary>
   `sizePerThread = [1, 1]`、`order = [1, 0]`；第 1 维 64 个元素需要 64 个线程：`threadsPerWarp[1] = 32`、`warpsPerCTA[1] = 2`；剩 1 lane、4 warp 给第 0 维：`threadsPerWarp = [1, 32]`、`warpsPerCTA = [4, 2]`。tile `[4, 64]`，铺 `[256, 64]` 需 64 趟 → 每线程 64 个元素。
   </details>

3. 判断：`tt.load` 的 `other` 值通过在 PTX 里加一条 `selp` 实现。

   <details markdown="1"><summary>答案</summary>
   错。先 `mov` 默认值进目标寄存器，再发**带谓词**的 `@p ld.global.vN`——谓词为假时 load 不执行，寄存器保留默认值。
   </details>

4. 计算：`offs = pid * BLOCK + tl.arange(0, BLOCK)`，`BLOCK = 512`，`x_ptr` 是对齐的 f16 指针。`tl.load(x_ptr + offs)` 每线程几条 load、什么宽度？`num_warps = 4`。若 `BLOCK = 512` 但 `pid` 换成 `tl.program_id(0) * 3`？

   <details markdown="1"><summary>答案</summary>
   `offs` contiguity 512、divisibility gcd(512, 2³⁰) = 512；指针 divisibility gcd(16, 512 × 2) = 16 字节 → alignment min(16 / 2 = 8, 512) = 8 个 f16 = 128 bit；每线程 512 / 128 = 4 个元素 < 8，`vec = min(8, 4) = 4` → 一条 `ld.global.v2.b32`（64 bit）。换成 `pid * 3`：divisibility 3（乘法相乘 1 × 3），`splat + arange` 的 divisibility = gcd(3, 2³⁰) = 1 → 指针 divisibility 2 字节 → alignment 1 → 4 条 `ld.global.b16`。
   </details>

5. 判断：`RemoveLayoutConversions` 的锚点包括所有 `tt.load`。

   <details markdown="1"><summary>答案</summary>
   错。只有 `isExpensiveLoadOrStore`——元素数 ≥ 线程数——的 load / store 是锚点；小张量的 load 层次无所谓、可以跟着别人走，且可以被重物化。
   </details>

6. 计算：`num_stages = 4`、无间接寻址、Ampere。缓冲几个？prologue 几次 load？`async_wait` 的 `num`（A、B 两个 load）？

   <details markdown="1"><summary>答案</summary>
   latency = (4 − 1) / 1 = 3 → load 在 stage 0、dot 在 stage 3 → 3 个缓冲；prologue 3 次（迭代 0、1、2）；允许在飞 (3 − 1) 个迭代 × 2 组 = `num = 4`。
   </details>

7. 判断：`#mma` 累加器沿 axis = 0（按列）规约不需要任何 shuffle。

   <details markdown="1"><summary>答案</summary>
   错。`#mma<{[2, 2], [16, 8]}>` 的行号由 lane 位 4、8、16（行 1、2、4）、寄存器位 2（行 8）、warp 位 2（行 16）、寄存器位 16（行 32）管——沿列规约要 3 次 shuffle（lane 位）、2 次寄存器内加、一次跨 warp。
   </details>

8. 计算：一个 kernel 有 `local_alloc` A（16 KB，循环内活跃）、B（16 KB，循环内）、一个 `convert_layout` scratch C（24 KB，循环后）、一个 `reduce` scratch D（1 KB，循环后且与 C 重叠活跃）。`ttg.shared` 是多少？

   <details markdown="1"><summary>答案</summary>
   干涉：A–B、C–D 各相交，A / B 与 C / D 不相交。着色：A 色 0、B 色 1、C 色 0（与 A 同色，不干涉）、D 色 1。颜色 0 的最大尺寸 24 KB、颜色 1 的 16 KB → 总量 40 KB（对齐后）。不是 57 KB。
   </details>

9. 判断：改了 `Coalesce.cpp` 并 `ninja` 重新构建后，旧的 `~/.triton/cache` 目录全部失效。

   <details markdown="1"><summary>答案</summary>
   对（前提是重建产物就是 Python import 到的那份 `libtriton.so`）：`triton_key()` 含该 `.so` 的 SHA-256，字节变了 key 就变，所有 kernel 落到新目录；旧目录仍在磁盘上但不再命中。
   </details>

10. 判断：TVM 的 `sch.vectorize` 在任何循环上都能调用成功。

    <details markdown="1"><summary>答案</summary>
    错。它要求该循环下的 block 是"局部完整 block"或"局部规约 block"且循环是空间轴等条件；第十二篇 §四.2 中 `decompose_reduction` 之后对 `Y_update` 内层调用被 `ScheduleError` 拒绝。
    </details>

### B. 跨篇综合（5 题）

1. 用户把 `tl.load(x_ptr + offs, mask=offs < n)` 改成 `tl.load(x_ptr + offs, mask=offs < n, other=0.0)` 后 kernel 变慢了。沿着 05 → 06 → 09 → 10 解释可能的原因链。

   <details markdown="1"><summary>答案</summary>
   `other` 让 load 的 lowering 多出 `mov` 默认值（第十篇），本身代价小；但若这个 load 在循环里被流水化，`LowerLoops` 对带非零 `other` 的 load 不能直接用 `cp.async`（它无法写默认值，第九篇 §二.4 的注释）——要么退回"寄存器里流水"（发 remark "severe performance degradation"）、要么在 `local_load` 之后加 `select`（多一次 shared memory 读后修补）。`other=0.0` 恰好是零——`cp.async` 的 `src_size = 0` 语义能写零，Triton 对零值有特殊处理；若写成非零常量或张量就走慢路。排查：`MLIR_ENABLE_DIAGNOSTICS=remarks` 看有没有那条 remark，对比两版 TTGIR 里 `async_copy_global_to_local` 是否存在。
   </details>

2. 一个 attention kernel 在 Ampere 上第二个 `tl.dot` 之前出现了一次经 shared memory 的 `convert_layout`（P 从 `#mma` 到 `#dot_op`）。用 07 与 08 说明它出现的条件，并给出两种消除办法。

   <details markdown="1"><summary>答案</summary>
   条件：第一个 `dot` 的 `#mma` 的 `warpsPerCTA` 在 N 上不为 1（如 `[2, 2]`），P 的 N（第二个 `dot` 的 K）被分到多个 warp，而 `#dot_op<{opIdx = 0}>` 要求每个 warp 持有自己 M 块的全部 K → `quotient(warp)` 失败。`warpsPerTileV2` 本应检测链式 dot 并给 `[numWarps, 1]`，若两个 `dot` 不在同一 Region（例如中间有 `scf.if`）或 shape rank 不同就检测不到。办法：(1) 让第一个 `dot` 的形状 / 位置满足链式检测，或用 `num_warps` 与 `BLOCK_N` 使 `getMmaV2WarpsPerCTA` 自然给出 `[4, 1]`（M 的 reps ≥ N 的）；(2) Gluon 显式指定两个 `dot` 的 `NVMMADistributedLayout(warps_per_cta=[4, 1])`。
   </details>

3. 同一个 kernel 编到 `sm_80` 与 `sm_90`，`metadata["shared"]` 从 32768 变 49152，而 `num_stages` 都是 3。用 09 与 10 解释这 16 KB 从哪来，并说 Hopper 上 epilogue 的 `convert_layout` scratch 为什么没有让它更大。

   <details markdown="1"><summary>答案</summary>
   Hopper 的 `wgmma` 异步、`warp_group_dot_wait {pendings = 1}` 允许一条 MMA 在飞，它读的缓冲不能被覆盖，`loadRequiresAdditionalBuffer` 让缓冲数从 2 变 3：3 × (8 + 8) KB = 48 KB。epilogue 的 `#mma → #blocked` scratch 是 32 KB，在循环后活跃，与流水线缓冲不干涉、同色、复用同一段地址，总量取最大值 48 KB 而不是相加。（用 TMA store 时它变成 `local_alloc` 到 `#nvmma_shared` 再 TMA 写回，同理复用。）
   </details>

4. 你要给 Triton 加一个新的 GPU 后端（假设一个新厂商，warp 大小 32，有自己的 MMA 指令与 fragment 布局）。按 03 / 04 / 07 / 10 / 11，列出必须新写与可以复用的部分。

   <details markdown="1"><summary>答案</summary>
   复用：前端与 TTIR（05）、`make_ttir` 全部 pass、AxisInfo、`ConvertTritonToTritonGPU`、Coalesce、`RemoveLayoutConversions`、`OptimizeThreadLocality`、`ReduceDataDuplication`、流水器的通用部分（若访存模型接近）、Linear Layout 与 `TritonGPUToLLVM` 的通用 pattern（`reduce`、`convert_layout`、`MemoryOp`、`AllocateSharedMemory`、`Membar`）、`compile()` / 缓存 / launcher 骨架。新写：`third_party/<vendor>/backend/compiler.py`（阶段表、选项、`make_ttgir` 顺序）与 `driver.py`（加载、launcher 的 C 模板）；一个 MMA layout 的 `AttrDef` 与 `toLinearLayout`（07 §八.3）；`AccelerateMatmul` 的厂商版（选 MMA 形状、`warpsPerCTA` 规则）；`TargetInfo` 实现（shuffle、barrier、`programId`、shared memory 访问）；`LoadStoreOpToLLVM` 与 `DotOpToLLVM`（指令级 lowering）；LLVM 的目标后端（若不在上游 LLVM 里，这是最大的一块）；`GetEnv.h` 名单里的新环境变量。AMD 后端是这份清单的现成范例。
   </details>

5. 用 12 的设计空间轴与 09 的 Gluon 讨论：Triton 团队为什么不选择"把 layout 暴露成 `tl` 的可选参数"（例如 `tl.load(..., layout=...)`）而是另起一个 Gluon 子语言？

   <details markdown="1"><summary>答案</summary>
   可选参数意味着**同一个 IR 里混合**用户指定与编译器推断的 layout：`RemoveLayoutConversions` 的锚点规则、冲突消解、重物化代价模型都要区分"用户钉死的"与"可改的"，每个自动 pass 都要学会绕开用户的决定，且用户只钉一处时其余处的自动决定可能与之冲突（插出更多 `convert_layout`）——系统处在轴的中间、两端的保证都没有。Gluon 把边界画清：TTGIR 以上全由用户决定、自动 pass 一个不跑，复用的是类型系统、verifier 与 lowering；代价是两套 API、用户代码不可移植。这与 TVM 用 DLight（规则）和手写调度并存而不是让 MetaSchedule 接受"部分固定的 trace"是同一种取舍——只是 MetaSchedule 的 trace 天然支持固定部分决策点，TVM 在这一点上更灵活。
   </details>

### C. 面试题（7 题）

1. 讲一遍 Triton 编译器从 Python 到 cubin 的流水线，每一站说一个它做的决定与依据的分析。

   <details markdown="1"><summary>答案</summary>
   要点：前端（特化 key、AST → TTIR、`make_ttir` 的 pattern）；AxisInfo（三元组、gcd）；`ConvertTritonToTritonGPU`（默认 `#blocked`、Dialect Conversion）；Coalesce（AxisInfo → `order` / `sizePerThread`）；`RemoveLayoutConversions`（锚点、传播、重物化）；`AccelerateMatmul`（MMA 版本、`warpsPerCTA`、`kWidth`）；流水（latency → stage → 缓冲 → 展开）；`AllocateSharedMemory` / `Membar`；`TritonGPUToLLVM`（LL 展开、内联 PTX）；LLVM `-O3` → NVPTX → `ptxas`；缓存与 launcher。追问：哪一站不需要 GPU（全部，除了跑 cubin）。
   </details>

2. 为什么 Triton 把 layout 放在类型里？这个决定的好处与代价。

   <details markdown="1"><summary>答案</summary>
   好处：每个 op 都看得见、verifier 检查一致性、改 layout 只有 `convert_layout` 一种方式、pass 之间不需要旁路的表、lit 测试可以直接断言。代价：类型不可变 → 改 layout 要重建 op 与 RAUW；大量 `convert_layout` 需要专门的全局 pass 清理；类型系统被硬件细节污染（`#nvidia_mma` 这种属性）——Linear Layout 是对后一点的补救（统一表示）。追问：TVM 怎么表示同一信息（`scope` 为 `wmma.matrix_a` 的缓冲，布局对 IR 不透明）。
   </details>

3. 解释 Linear Layout 为什么能统一所有 layout，以及 `convert_layout` 走哪条路是怎么判定的。

   <details markdown="1"><summary>答案</summary>
   硬件位置的每一位独立地贡献下标的某些位、2 的幂对齐的偏移相加是 XOR → 线性；基向量表完全确定映射；swizzle、转置、广播、复制都是线性映射的实例；`#blocked → LL` 是三个 `identityStandardND` 的乘积加 shape 修正。判定：`dst.invertAndCompose(src)` 得源位置 → 目标位置的映射，从 block / warp / lane 起 `quotient` 恒等的维；剩 `register` → 寄存器重排，剩 `{register, lane}` → shuffle，否则 shared memory。追问：为什么 LL 不能表示 padding（非 2 的幂）。
   </details>

4. `num_stages` 在 Ampere 与 Hopper 上分别变成什么？为什么 Hopper 多一个缓冲？

   <details markdown="1"><summary>答案</summary>
   要点见 B.3 与第九篇 §三 / §四：latency、stage 差、`cp.async` + `commit_group` + `wait_group`；`wgmma` 异步与 `pendings = 1`；TMA + mbarrier 把 iter_args 从 15 降到 4。追问：`async_wait {num}` 的语义。
   </details>

5. Membar 分析是什么、为什么必要、它保守在哪。

   <details markdown="1"><summary>答案</summary>
   沿程序顺序维护自上一个 barrier 以来的 shared memory 读写区间，新访问 RAW / WAR / WAW 相交就在前面插 barrier，`MemWaitOpTrait` 后必插；虚拟块上做不动点处理循环与分支。必要：TTGIR 不写 barrier，`cp.async.wait_group` 只保证本线程；复用地址的正确性靠它。保守：不做 warp 级所有权分析（不同 warp 各读写自己那片也插）、区间粒度是整个缓冲（有子切片信息时例外）。追问：`canSkipBarSync` 与 warp specialization 用 mbarrier 的原因。
   </details>

6. Triton 的编译缓存 key 由什么组成？团队分发缓存时会遇到什么问题？

   <details markdown="1"><summary>答案</summary>
   五成分（第十一篇 §三）；分发问题见第十一篇自测 2：架构、`ptxas` 版本、wheel 字节、登记环境变量、`extern_libs` 路径、组文件的绝对路径；解法是远端缓存后端。追问：改 pass 不生效的第一件事查什么。
   </details>

7. 比较 Triton、TVM、CUTLASS 三种"写高性能 GEMM"的方式：谁决定什么、搜索空间、风险、可移植性。

   <details markdown="1"><summary>答案</summary>
   第十二篇 §九 的表：Triton 用户定 tile、编译器定其余、autotune 几十个、风险在启发式、NVIDIA / AMD；TVM 调度作者（或 DLight / MetaSchedule）定一切、原语保正确、空间 10⁶、风险在作者 / 规则 / 搜索、多后端源码生成；CUTLASS 用户定一切、模板、profiler 枚举、风险在用户与编译时间、仅 NVIDIA。追问：Gluon 在哪个位置、为什么出现。
   </details>

### D. 掌握判据

| 层次 | 判据 |
|---|---|
| 读过 | 能说出 IR / SSA / pass / lowering、MLIR 的 Operation / Region / Dialect、Triton 流水线的站名、AxisInfo 三元组、layout 四种、软件流水四步、`Membar`、缓存五成分、TVM 的算法 / 调度分离 |
| 掌握 | 能对一个给定 kernel 手算 AxisInfo 与默认 layout、预测 Coalesce 的 `sizePerThread`、写出 `#blocked` 与 `#mma` 的基向量表并判定一次转换走哪条路、算出 `num_stages` 对应的缓冲与 `async_wait`、解释 PTX 里每个 `bar.sync` 的来源、算缓存目录数、用 `triton-opt` 复现任一篇的 IR、写一个 20 行的 lit 测试、用 TVM 原语写出同一 GEMM 的调度 |
| 能教人 | 能解释为什么 layout 在类型里而不是表里、为什么 Block 参数优于 φ、为什么 Dialect Conversion 而非 greedy、为什么 NVPTX 不分配寄存器、为什么 epilogue 的转换消不掉而 attention 的 P 不需要、为什么 Hopper 多一个缓冲、为什么 TVM 能搜 10⁶ 而 Triton 只列几十个、为什么 Gluon 是另起一门而不是加参数 |

通关标准：A 组 8 题以上正确（计算题精确），B 组 4 题以上能写出完整推理链并指出依据的 pass 与分析，C 组每题能说出至少三个要点并回答一个追问。

## 六、Infra 地图的收束与下一步

本系列是[《AI Infra 学习地图》](/ai-infra-learning-roadmap.html)的**选修**——十二个主线系列之外唯一的一门，也是地图上最后写完的一门。它的位置在 L2 之下：[《GPU Kernel 工程》](/gpu-kernel-engineering.html)第七篇画的六层图是它的入口，[《PyTorch 深度实践》](/deep-dive-into-pytorch.html)第七篇（Dynamo → AOTAutograd → Inductor）是它的另一个入口——Inductor 生成的 Triton kernel 从这里进入编译器。

读完它之后，面对一个 Triton kernel 的性能或正确性问题，追问可以一直到底：

| 追问 | 答案来自 |
|---|---|
| 这条 load 为什么不是 128 bit？ | 06（AxisInfo）→ 07（Coalesce）→ 10（`getVectorSize`） |
| 这个 `convert_layout` 从哪来、能不能消？ | 07（初始 layout / Coalesce）→ 08（锚点与重物化） |
| 为什么 shared memory 用了这么多？ | 09（缓冲数）→ 10（`AllocateSharedMemory` 的复用） |
| 这个 `bar.sync` 是谁插的？ | 10（`Membar`） |
| 为什么 Hopper 上没有 `ldmatrix`？ | 08（v3 路径）→ 09（`wgmma` 操作数在 shared memory） |
| 为什么改了编译器没生效？ | 11（`triton_key`）→ 13（构建） |
| 为什么 TVM 能自动调而 Triton 要手列 config？ | 12（正确性放在原语里） |

下一步：

- 想给 Triton 提 PR：第十三篇的工作台 + [《AI-Infra 开源贡献指南》](/contributing-to-ai-infra-open-source.html)的流程；从 `test/` 里一个 lit 用例开始读、从一个 `emitRemark` 的改进开始写。
- 想给新硬件接后端：B.4 的清单，`third_party/amd` 是范例；先让 lit 全绿再谈性能。
- 想在 MLIR 生态里横向迁移：IREE、torch-mlir、StableHLO 用的是同一套基础设施（第三、四篇），差别只在方言与 pass。
- 想做 GPU 上的极致性能而不想等编译器：Gluon（第九篇 §七）或 CUTLASS，两者在设计空间轴上是邻居。

Infra 地图的十三个系列到此完成。

## 七、延伸阅读

- LLVM：*LLVM Language Reference*；Chris Lattner & Vikram Adve, *LLVM: A Compilation Framework for Lifelong Program Analysis & Transformation*（CGO 2004）；NVPTX 后端文档 *User Guide for NVPTX Back-end*。
- MLIR：Lattner et al., *MLIR: Scaling Compiler Infrastructure for Domain Specific Computation*（CGO 2021）；MLIR 官方文档的 *Language Reference*、*Dialect Conversion*、*Pattern Rewriting*、*Testing Guide*；*Toy* 教程。
- Triton：Tillet, Kung & Cox, *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations*（MAPL 2019）；Triton 仓库 `docs/`、`include/triton/Tools/LinearLayout.h` 的头部文档、`python/tutorials/gluon/`；Adam Goucher 关于 Linear Layout 的原始讨论；NVIDIA *PTX ISA* 手册的 `mma` / `wgmma` / `cp.async.bulk` / `mbarrier` / `tcgen05` 章。
- TVM：Chen et al., *TVM: An Automated End-to-End Optimizing Compiler for Deep Learning*（OSDI 2018）；Feng et al., *TensorIR: An Abstraction for Automatic Tensorized Program Optimization*（ASPLOS 2023）；Shao et al., *Tensor Program Optimization with Probabilistic Programs*（MetaSchedule，NeurIPS 2022）；Ragan-Kelley et al., *Halide*（PLDI 2013）；TVM 仓库 `docs/deep_dive/tensor_ir/`。
- 对照：XLA 的 *Operation Semantics* 与 *GPU backend* 文档；IREE 的 *Transform Dialect* 文档；PyTorch Inductor 的 `torch/_inductor/` 源码；CUTLASS 3 的 *CuTe* 文档。
- 教材：Cooper & Torczon, *Engineering a Compiler*（SSA、数据流、寄存器分配的标准参考）；Muchnick, *Advanced Compiler Design and Implementation*。

[^q0]: 面对一段 Triton kernel 与它的 TTGIR / PTX：每个参数为什么有或没有 `tt.divisibility`、每个 `constexpr` 去了哪（5）；每条 load 的 AxisInfo 三元组与由此得到的向量宽度和谓词粒度（6）；每个张量的 layout 是默认的、Coalesce 给的还是 `dot` 决定的，写出它的基向量表，任一 `convert_layout` 走哪条路（7）；剩下的 `convert_layout` 两端被什么锚住、能不能重物化（8）；`num_stages` 变成了几个缓冲、几次 prologue load、`async_wait` 等谁，Hopper 上哪些变了、warp specialization 把什么给了谁（9）；每个 `bar.sync` 由哪次区间相交推出、`ttg.shared` 怎么算出来、`reduce` 的三级各多大（10）；这次调用落在哪个缓存目录、改了什么会重编、launcher 传了哪些参数、AMD 上同一 kernel 的 layout 与 ISA 长什么样（11）；同一 GEMM 用 TVM 调度怎么写、DLight 与 MetaSchedule 各自怎么得到它、两条路的风险各在哪（12）；出错时怎样二分到 pass 并固定成 lit（13）；以及这一切建在什么通用机制上——SSA 与数据流（1）、LLVM 的后端分工（2）、MLIR 的 Operation / Interface / Conversion（3、4）。详见[第一章](#一总览系列回答的问题与主线)与[第二章](#二逐篇回顾)。

[^q1]: 数字：`-O2` 119 项 pass、126 次 dump；`align 16` 一条 `v4` vs `align 4` 四条 `b32`；地址空间 1 / 3 / 5；`ptxas -v --regAllocOptLevel=2`；`pid * 1024 + arange(1024)` → `[1024], [1024], [1]`；`a_ptrs` → `[1, 32], [2, 16]`、alignment 8；A tile `[1, 8], [8, 4], [4, 1]`；`[128, 128]` → `warpsPerCTA = [2, 2]`；`convert_layout` 23 → 3 → 7 → 3 → 1；`num_stages = 3` → 2 缓冲、prologue 2 次、`async_wait {num = 2}`、15 个 iter_args；Hopper 3 缓冲、`pendings = 1`、4 个 iter_args、`shared` 32768 → 49152；WS 4 / 1 / 2 warp；rowsum 7 加 + 5 shuffle + 16 字节；matmul 64 `mma.sync`、26 `ldmatrix`、24 `cp.async`、10 barrier、`ttg.shared = 32768`；缓存 key 五成分、元数据 34 字段、block = `32 × num_warps`；gfx942 wave 64、MFMA 32×32×8、144 VGPR、`Occupancy 3`；TVM CPU 1089 → 52 μs、Metal 700 GFLOP/s、MetaSchedule 10⁴–10⁶ vs Triton 几十个；lit 277 文件 9.5 秒、`MLIR_ENABLE_DUMP` 74 份。结论：layout 在类型里；Block 参数代替 φ；Dialect Conversion 而非 greedy；NVPTX 不分配寄存器；`sizePerThread` 是 AxisInfo 推的；转换代价由 `quotient` 判定；epilogue 转换消不掉、attention 的 P 在 `[4, 1]` 下零成本；缓冲数 = stage 差；barrier 全由 `Membar` 推出；shared memory 按生命期复用；改 `.cpp` 要重建才进 key；TVM 原语保正确所以能搜大空间。详见[第一章](#一总览系列回答的问题与主线)、[第三章](#三贯穿全系列的几条线)。

[^q2]: 用第五章 D 节：读过——能说出各站名词；掌握——能对给定 kernel 手算 AxisInfo 与默认 layout、预测 Coalesce 结果、写基向量表判定转换路径、算缓冲与 `async_wait`、解释每个 barrier、算缓存目录数、用 `triton-opt` 复现 IR、写 lit、用 TVM 原语写同一 GEMM；能教人——能解释 layout 在类型里、Block 参数优于 φ、Conversion 而非 greedy、NVPTX 不分配寄存器、epilogue 消不掉而 P 零成本、Hopper 多一个缓冲、TVM 搜 10⁶ 而 Triton 几十个、Gluon 另起一门的理由。通关：A 组 8 题以上（计算精确）、B 组 4 题以上完整推理链并指出 pass 与分析、C 组每题三个要点加一个追问。详见[第五章](#五通关自测)。
