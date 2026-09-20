---
layout: post
series: ml-compilers
title: "ML 编译器内部（06）：Triton 编译器（二）——AxisInfo：一切向量化决定的地基"
subtitle: "The Triton Compiler II: AxisInfo, the Analysis Behind Every Vectorization Decision"
tags: [Compiler, MLIR, Triton, GPU, AI-Infra]
catalog: true
---

上一篇结束时，matmul kernel 的 TTIR 已经就位：函数参数上挂着 `tt.divisibility = 16`，`stride_ak = 1` 折叠进了地址运算，`addptr` 链合并成了一个。这些信息接下来要被**一个分析**消费。它决定第七篇的 Coalesce 给每个 load / store 选什么 layout，决定第十篇的 lowering 把一条 `tt.load` 变成一条 `ld.global.v4.b32` 还是四条 `ld.global.b32`，决定一个带 mask 的 store 能不能用一个谓词覆盖八个元素。它叫 `AxisInfo`。

Triton 用户接触过它的两个末端：一头是 `tl.multiple_of(x, 16)` 这种提示，一头是 dump 出来的 TTGIR 里 `sizePerThread = [8]` 还是 `[1]`。中间那段——编译器怎样从"`M` 是 16 的倍数"推到"`c_ptrs` 沿第 1 维每 8 个元素对齐到 16 字节"——是这一篇的内容。它也是第一篇讲的数据流分析在 Triton 里最重要的实例。

总纲对这一篇提出的核心问题是：

> **`offs = pid * BLOCK + tl.arange(0, BLOCK)`，`BLOCK = 1024`，`pid` 是 `tt.get_program_id`。编译器如何推出 `offs` 沿这一维 contiguity 为 1024、divisibility 为 1024？[^q0] 如果用户写的是 `offs = pid * n + tl.arange(0, BLOCK)`，`n` 是运行时整数，结论变成什么？[^q1] 多了一个 `tl.multiple_of(n, 16)` 又变成什么？[^q2]**

## 一、总览

本文按**一个数据流分析的四个组成部分**组织：格（第二章：三个属性的精确定义）→ 起点（第三章：信息从哪里进入）→ 传递函数（第四章：每种 op 的规则，每条用一个小例子验证）→ 汇合与不动点（第五章：`join`、循环、分支，以及它建在 MLIR 数据流框架上的方式）。然后是分析的另一端：谁在消费它、消费的公式是什么（第六章），信息在哪些常见写法里丢掉（第七章），怎样用 `triton-opt` 把每个值的 AxisInfo 打出来（第八章）。最后与 LLVM 的同类分析对照（第九章）。

| 章 | 主题 | 内容 |
|---|---|---|
| 二 | 格 | contiguity / divisibility / constancy / constantValue 的定义；为什么都是 2 的幂；global divisibility |
| 三 | 起点 | 函数参数属性、`tl.multiple_of` 一族、常量、`make_range`、`get_program_id`、poison |
| 四 | 传递函数 | `splat` / `expand_dims` / `broadcast`、`add` / `sub` / `addptr`、`mul`、`div` / `rem`、`cmp`、`select` / 逻辑、`load`，各带例子与 lit 测试证据 |
| 五 | 汇合与循环 | `join` 取 gcd；`scf.for` 的归纳变量与 iter_args；`scf.if`；`SparseForwardDataFlowAnalysis`；跨函数 |
| 六 | 消费者 | `getAlignment` / `getContiguity` / `getMaskAlignment` 的公式；Coalesce、load / store lowering、elementwise 去重 |
| 七 | 信息怎么丢 | 核心问题的三个变体；`%`、运行时 stride、两个连续量相加、`where` |
| 八 | 工具 | `triton-opt -test-print-alignment`；matmul kernel 每个值的 AxisInfo |
| 九 | 对照 | LLVM 的 `KnownBits`、`ScalarEvolution`、`Alignment` |
| 十 | 本文小结 | |
| 十一 | 自测 | 5 道题 |

源码：`include/triton/Analysis/AxisInfo.h`、`lib/Analysis/AxisInfo.cpp`（约 1500 行，本文覆盖其中所有传递规则）、`test/Analysis/test-alignment.mlir`（1290 行 lit 测试，本文引用的每个"预期结果"都来自它）、消费者在 `lib/Dialect/TritonGPU/Transforms/CoalesceUtils.cpp`、`Utility.cpp` 与 `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp`。

## 二、格：三个属性的精确定义

`AxisInfo` 是附在**每个整数或指针类型的值**上的一组信息（浮点值只有平凡信息）。对一个 rank 为 r 的张量，它沿每一维 d 记三个整数，外加一个可选的常量值：

```text
class AxisInfo {
  SmallVector<int64_t> contiguity;     // 每一维一个
  SmallVector<int64_t> divisibility;   // 每一维一个
  SmallVector<int64_t> constancy;      // 每一维一个
  std::optional<int64_t> constantValue;  // 整个张量是同一个常量时，它的值
};
```

### 1. contiguity

**contiguity[d]**：沿维度 d，把元素切成等长的段，使每段内的值是连续递增的整数（`v, v+1, v+2, …`），能保证成立的**最短段长**。头文件里的例子：

```text
[[10, 11, 12, 13, 18, 19, 20, 21],
 [20, 21, 22, 23, 28, 29, 30, 31]]      contiguity = [1, 4]
```

沿第 1 维每 4 个一段连续（`10 11 12 13`、`18 19 20 21`），沿第 0 维没有连续（`10` 下面是 `20`），所以是 `[1, 4]`。

```text
[[12, 16, 20, 24],
 [13, 17, 21, 25],
 [14, 18, 22, 26],
 [15, 19, 23, 27],
 [18, 22, 26, 30],
 [19, 23, 27, 31]]                      contiguity = [2, 1]
```

沿第 0 维 `12 13`、`14 15`、`18 19` 每两个一段连续（第 0 维长 6，不是 2 的幂，所以能保证的段长是 2 而不是 4——`14 15` 后面是 `18`），沿第 1 维步长 4，不连续。

一个 `tl.arange(0, 128)` 的 contiguity 是 128；一个 `tl.splat(x)` 是 1。contiguity 是**保守下界**：分析说 4，实际可能是 8，但绝不会少于 4。

### 2. divisibility

**divisibility[d]**：沿维度 d，把元素按 contiguity[d] 切成段之后，**所有段的第一个元素**的最大公共 2 幂因子。

```text
[[10, 11, 12, 13, 18, 19, 20, 21],
 [20, 21, 22, 23, 28, 29, 30, 31]]      contiguity = [1, 4], divisibility = [1, 2]
```

第 1 维的段首是 `10, 18, 20, 28`，都能被 2 整除、不都能被 4 整除，所以是 2；第 0 维 contiguity 为 1，每个元素都是段首，`10 … 31` 里有奇数，所以是 1。

为什么定义在**段首**而不是每个元素上？因为 divisibility 的用途是判断**地址对齐**：一段 contiguity 为 8 的 bf16 指针，只要段首对齐到 16 字节，这 8 个元素就可以用一条 128 bit 的 load 取回来——段内后面 7 个元素的地址是否对齐无关紧要，它们本来就不对齐。所以 divisibility 只关心每个向量访存的起点。

推论：**contiguity 为 1 的维度上，divisibility 对每个元素都成立**。`getGlobalDivisibility()` 取所有 contiguity 为 1 的维度的 divisibility 的最大值，构造函数把它 max 进每一维——因为"所有元素都是 8 的倍数"这一事实与维度无关。

### 3. constancy

**constancy[d]**：沿维度 d，把元素切成等长的段，使每段内的值全部相等，能保证成立的最短段长。

```text
[[8, 8, 8, 8, 12, 12, 12, 12],
 [16, 16, 16, 16, 20, 20, 20, 20]]      constancy = [1, 4]
```

`tl.splat(x)` 到 `[128, 32]` 的 constancy 是 `[128, 32]`；`tl.arange` 的是 1。constancy 有两个用途：一是作为 contiguity 传递的配角——`splat(pid * BLOCK) + arange` 之所以连续，是因为左边**常量段长 ≥ 右边连续段长**（§四.3）；二是 mask 的向量化——mask 沿最快维 constancy 为 8，意味着 8 个相邻元素的谓词相同，一条向量 store 可以用一个谓词（§六.3）。

### 4. constantValue

整个张量是同一个常量时记下它的值。它让传递函数能做精确的常量折叠（`x * 1 → x` 保留 contiguity、`x % 1 → 0`），也让比较运算能直接判定真假。

### 5. 为什么都是 2 的幂

张量的每一维长度是 2 的幂（Triton 要求 `tl.arange` 与 block shape 都是 2 的幂），三个属性都是"把维度切成等长段"的段长，所以只能是 2 的幂的约数——也是 2 的幂。这让所有的汇合运算都可以用 **gcd**（最大公约数）表达：两个 2 的幂的 gcd 就是较小的那个，两个信息的"更保守"就是取 gcd。`AxisInfo::join` 全程只用 gcd，这是设计上的简洁。

一个由此而来的特殊值：`highestPowOf2Divisor(0)`——0 能被任何 2 的幂整除，所以 `make_range` 从 0 开始时 divisibility 是 $$2^{30}$$（int32 的上界）、`arith.constant 0` 的是 $$2^{62}$$。lit 测试里到处可见的 `1073741824` 和 `4611686018427387904` 就是它们。

## 三、起点：信息从哪里进入

数据流分析需要初值。`AxisInfo::getPessimisticValueState(value)` 给每个值一个**悲观初值**——三个属性全 1，再看这个值有没有额外的信息来源：

| 值的来源 | 初值 | 信息从哪来 |
|---|---|---|
| 函数参数（entry block 的 block argument） | 读 `tt.func` 参数上的属性 `tt.divisibility` / `tt.contiguity` / `tt.constancy` | 前端按实参的 `D` 标记挂上的（第五篇 §三.2）；`ModuleAxisInfoAnalysis` 也会为 `noinline` 函数从调用点推出来挂上（§五.5） |
| 有定义 op 的值 | 读 op 上的**可丢弃属性**（discardable attribute）`tt.divisibility` / `tt.contiguity` / `tt.constancy` | `tl.multiple_of(x, v)`、`tl.max_contiguous(x, v)`、`tl.max_constancy(x, v)`：`semantic.py` 里它们只做一件事——`x.handle.set_attr("tt.divisibility", …)`，把属性挂在**定义 `x` 的那个 op** 上，返回 `x` 本身 |
| 其他 block 的参数（`scf.for` 的 iter_args、`scf.if` 的结果） | 全 1，随后由汇合决定（§五） | — |

然后 `visitOperation` 对每个 op 算出结果的 AxisInfo 后，**再用 op 上的属性覆盖一次**（`initDimVectorFromHint`）：

```cpp
AxisInfo curr = visitors.apply(op, operands);
...
AxisInfo::initDimVectorFromHint(op->getDiscardableAttr("tt.contiguity"), &newContiguity);
AxisInfo::initDimVectorFromHint(op->getDiscardableAttr("tt.divisibility"), &newDivisibility);
AxisInfo::initDimVectorFromHint(op->getDiscardableAttr("tt.constancy"), &newConstancy);
```

这就是 `tl.multiple_of` 的全部机制：**它不是断言、不是运行时检查，只是把分析结果沿这一维替换成用户给的数**。写错了（说是 16 的倍数其实不是）编译器不会发现，生成的向量 load 会在运行时读到错位的数据或越界。它也解释了为什么 `tl.multiple_of` 必须**直接**作用在要用的那个值上：属性挂在定义 op 上，经过任何一次运算之后是新的 op、新的值，提示不再跟着走（信息会按传递规则继续传，但那是计算出来的，不是提示）。

`tl.assume(cond)` 不在这个名单里。它生成 `llvm.intr.assume`，供 LLVM 的优化器用，AxisInfo 不读它——所以 `tl.assume(n % 16 == 0)` 对 Triton 层的向量化决定没有影响，`tl.multiple_of(n, 16)` 才有。

几类叶子 op 的初值（它们没有操作数，信息来自 op 本身）：

| op | contiguity | divisibility | constancy | constantValue | 说明 |
|---|---|---|---|---|---|
| `tt.make_range {start, end}` | `end - start` | `highestPowOf2Divisor(start)` | 1 | — | `arange(0, 128)`：`[128]`、`[2^30]`、`[1]`；`arange(1, 129)`：`[128]`、`[1]`、`[1]` |
| `arith.constant c`（标量） | 1 | `highestPowOf2Divisor(c)` | 1 | c | `constant 128`：div 128 |
| `arith.constant dense<c>`（张量） | 每维 1 | 每维 `highestPowOf2Divisor(c)` | 每维 = shape | c | `tl.zeros` / `tl.full`；`dense<1>`：div 1、const = shape |
| `tt.get_program_id` | 1 | 1 | 1 | — | 没有专门的 visitor，落到悲观值：编译器对 `pid` 一无所知 |
| `ub.poison` | 全 $$2^{62}$$ | 全 $$2^{62}$$ | 全 $$2^{62}$$ | — | 前端给归纳变量的占位符（第五篇 §四.6）；poison 永远不会被读到，给最乐观的值让它不拖累汇合 |

`get_program_id` 是最值得注意的一行：**编译器不知道 `pid` 的任何性质**（它是运行时的 `blockIdx`）。所有从 `pid` 出发的地址运算，对齐信息只能来自乘上去的常量——这就是核心问题里 `pid * BLOCK` 与 `pid * n` 的差别。

## 四、传递函数：每种 op 的规则

`AxisInfoAnalysis` 用一张 visitor 表分派：每种 op 一个 `AxisInfoVisitorImpl<OpTy>`，`getAxisInfo(op, operands)` 从操作数的 AxisInfo 算结果的。下面按 op 类别给出规则，每条规则配 `test-alignment.mlir` 里的一个实例（`expected-remark` 就是 `triton-opt -test-print-alignment` 的真实输出）。

### 1. 形状类：`splat`、`expand_dims`、`broadcast`

| op | 规则 |
|---|---|
| `tt.splat x → tensor<s…>` | 每维 contiguity 1、divisibility = x 的 divisibility、constancy = 该维长度；constantValue 继承 |
| `tt.expand_dims x, axis` | 在 axis 位置插入一维：contiguity 1、constancy 1、divisibility = 各维中"contiguity 为 1 的维的 divisibility"的 gcd（contiguity > 1 的维按 1 计）；x 是常量时用常量的 2 幂因子 |
| `tt.broadcast x` | 被广播的维（原长 1）：contiguity 1、constancy = 新长度；其他维不变；divisibility 全部不变 |

```text
%10 = tt.make_range {end = 128, start = 0}          → contiguity=[128], divisibility=[2^30], constancy=[1]
%11 = tt.expand_dims %10 {axis = 0} : → tensor<1x128xi32>
                                                   → contiguity=[1,128], divisibility=[1,2^30], constancy=[1,1]
%12 = tt.broadcast %11 : tensor<1x128xi32> -> tensor<128x128xi32>
                                                   → contiguity=[1,128], divisibility=[1,2^30], constancy=[128,1]
%13 = tt.splat %arg0 : !tt.ptr<i1> {tt.divisibility=16} -> tensor<128x128x!tt.ptr<i1>>
                                                   → contiguity=[1,1], divisibility=[16,16], constancy=[128,128]
```

`expand_dims` 新维的 divisibility 为什么是 1 而不是 $$2^{30}$$？新维长度为 1，这一维上每个元素自成一段（contiguity 1），divisibility 是"所有元素"的公共因子；`0 1 2 … 127` 的公共 2 幂因子是 1。规则里"contiguity > 1 的维按 1 计"就是这个意思：一段 `[2^n, 2^n+1, …]` 的元素整体没有公共因子，虽然段首有。

### 2. 加减与指针加：`arith.addi`、`arith.subi`、`tt.addptr`

三者共用 `AddSubOpAxisInfoVisitor`。`addptr` 多一步：偏移的单位是**元素**，divisibility 的单位是**字节**，所以先把偏移的 divisibility 乘上元素大小。

**contiguity**（加法）：

$$\text{contig}(a+b) = \max\bigl(\gcd(\text{const}(a), \text{contig}(b)),\ \gcd(\text{contig}(a), \text{const}(b))\bigr)$$

用循环写出来：

```python
# 一段里 t 从 0 到 L-1：a 常量、b 连续 → a + (b0 + t) 连续，段长 = min(常量段, 连续段) = gcd
# 两者都连续 → (a0 + t) + (b0 + t) = a0 + b0 + 2t，步长 2，不连续 → 1
```

**divisibility**（加法）：

- 两边都连续（contig > 1）：段内元素是 `a0 + b0 + 2t`，不再连续，结果 contiguity 变 1，此时 divisibility 对每个元素都要成立——只能说"两个都是偶数则结果是偶数"：两边 divisibility 都是偶数 → 2，否则 → 1；
- 至少一边不连续：`gcd(div(a), div(b))`，但如果有一边 contiguity > 1，还要与结果的 contiguity（乘元素大小）再取一次 gcd——因为结果的段首可能落在那一边的段的**中间**，那里的对齐只有段长保证。

减法的 contiguity 只有 `gcd(contig(a), const(b))` 一项（`a - (b0 + t)` 是递减的，不算连续）。

```text
%0 = tt.make_range {end = 128, start = 0}                → [128], [2^30], [1]
%1 = arith.constant dense<1> : tensor<128xi32>           → [1], [1], [128], value 1
%2 = arith.addi %0, %1     ; 连续 + 常量                  → [128], [1], [1]
%5 = arith.addi %0, %0     ; 连续 + 连续 → 0,2,4,…         → [1], [2], [1]
%9 = arith.addi %0, %arg0  ; %arg0: contig 1, div 4, const 2 → [2], [2], [1]
```

`%9` 是最能说明规则的一例：`[0,1,2,3,…] + [4,4,8,8,12,12,…]`，常量段长 2、连续段长 128，结果连续段长 gcd(2, 128) = 2（`4 5 | 10 11 | 16 17 …`）；divisibility gcd(2^30, 4, 结果 contig 2) = 2。

`addptr` 的例子（指针参数 divisibility 16 字节）：

```text
%cst4 = arith.constant 4 : i32
%8  = tt.addptr %arg3, %cst4 : !tt.ptr<i32>, i32        → div 16   (16 与 4×4=16 的 gcd)
%7  = tt.addptr %arg2, %cst4 : !tt.ptr<i16>, i32        → div 8    (16 与 4×2=8 的 gcd)
%21 = tt.addptr %16, %12 : tensor<128x128x!tt.ptr<i32>>, tensor<128x128xi32>
      ; %16 = splat(ptr i32, div 16), %12 = broadcast(expand_dims(arange 128)) 
                                                        → contiguity=[1,128], divisibility=[4,16], constancy=[128,1]
```

最后一行：第 1 维连续 128（常量指针 + 连续偏移），divisibility 16（指针的 16 字节，与 $$2^{30} \times 4$$ 和 $$128 \times 4$$ 取 gcd 仍是 16）；第 0 维不连续，divisibility 是 gcd(16, 1 × 4) = 4——沿第 0 维相邻的两个指针相差一行，行首的对齐只有 4 字节保证（因为 `%12` 第 0 维 divisibility 是 1）。

### 3. 乘法：`arith.muli`

| 属性 | 规则 |
|---|---|
| contiguity | 只有乘 1 保留：`rhs == 1 ? contig(lhs) : 1`，反之亦然；否则 1 |
| divisibility | `div(lhs) × div(rhs)`（溢出则饱和到上界）；但**连续的一边按 1 计**（`[4,5,6,7] × 2 = [8,10,12,14]`，gcd 是 2 不是 8） |
| constancy | `gcd(const(lhs), const(rhs))` |
| constantValue | 两边都是常量则相乘；任一边是 0 则 0 |

```text
%pid = tt.get_program_id x                              → [1], [1], [1]
%c128 = arith.constant 128 : i32                        → [1], [128], [1], value 128
%1 = arith.muli %pid, %c128                             → [1], [128], [1]
```

**`pid * 128` 的 divisibility 是 128**——尽管 `pid` 一无所知。这是所有 Triton kernel 里 block 起始地址对齐的来源。

### 4. 除法与取模：`arith.divsi/divui`、`arith.remsi/remui`

| op | contiguity | divisibility | constancy |
|---|---|---|---|
| `a / b` | `b == 1 ? contig(a) : 1` | `a == 0` → 不变；`b == 1` → 不变；a 不连续且 b 是 2 的幂常量 → `div(a) / b`；否则 1 | a 连续、b 常量：`max(默认, gcd(contig(a), div(a), div(b)))`——`[0..127] / 64` 是 `0,…,0,1,…,1`，每 64 个相同 |
| `a % b` | a 连续、b 常量：`gcd(contig(a), div(a), div(b))`；否则 1 | b 的 constancy > 1：`gcd(div(a), div(b))`；否则 1 | `b == 1` → 全长（结果全 0）；否则默认 |

```text
%0 = tt.make_range {end = 128, start = 0}
%4 = arith.constant dense<64>
%5 = arith.remsi %0, %4        ; 0..127 % 64 = 0..63, 0..63      → [64], [64], [1]
%12 = tt.make_range {end = 160, start = 32}                        → [128], [32], [1]
%15 = arith.remsi %12, %4      ; 32..159 % 64                      → [32], [32], [1]
%13 = arith.remsi %0, %12      ; 连续 % 连续                        → [1], [1], [1]
```

`%15` 说明了 divisibility 参与 contiguity 计算的原因：`32..159 % 64` 是 `32..63, 0..63, 0..31`——段首 32 不是 64 的倍数，连续段只能保证 32 长。

取模是 Triton kernel 里最常见的信息杀手：`offs % n`（`n` 是运行时值，constancy 是全长但 divisibility 是 1）→ contiguity `gcd(contig, div(offs), 1) = 1`。§七再谈。

### 5. 比较：`arith.cmpi`

比较的结果是 `i1` 张量，contiguity 与 divisibility 恒为 1（布尔值没有对齐可言），有意义的只有 **constancy**：

- 两边都是常量：整个张量同一个结果，constancy = shape，constantValue = 比较结果；
- 默认：`gcd(const(lhs), const(rhs))`；
- **一边全连续、一边全常量**，且谓词是 `<` / `>=`（连续在左）或 `>` / `<=`（常量在左）：`max(默认, gcd(contig(连续侧), div(连续侧), div(常量侧)))`。

最后一条是 mask 向量化的关键。`offs < n`，`offs = pid*128 + arange(0,128)`（contiguity 128、divisibility 128），`n` divisibility 16：结果 constancy = gcd(128, 128, 16) = **16**。含义：`offs` 每 16 个一段，段首是 16 的倍数，`n` 也是 16 的倍数，所以 `n` 不可能落在段的中间——一段 16 个元素的比较结果要么全真要么全假。这让 16 个相邻元素共用一个谓词。如果 `n` 没有 `tt.divisibility`（用户传了 1000 这样的值），constancy 是 gcd(128, 128, 1) = 1——每个元素单独判断。

```text
;; @store_constant_align：%n 有 tt.divisibility = 16
%4 = arith.addi %3, %2          ; pid*128 + arange       → [128], [128], [1]
%9 = tt.splat %n                                         → [1], [16], [128]
%mask = arith.cmpi slt, %4, %9                           → contiguity=[1], divisibility=[1], constancy=[16]
```

### 6. 逻辑、选择、移位、极值、类型转换

| op | 规则 |
|---|---|
| `arith.andi` / `ori` / `xori` | constancy 取 gcd；两边都是常量则算出常量值；contiguity、divisibility 1 |
| `arith.select cond, a, b` | `a`、`b` 的 AxisInfo 取 `join`（gcd），再与 `cond` 的 constancy 取 gcd——`cond` 每 8 个相同，结果至多每 8 个来自同一边 |
| `arith.shli a, c` | divisibility = `div(a) × 2^c`；contiguity 1（除 c = 0） |
| `arith.shrui/shrsi a, c` | divisibility = `div(a) / 2^c`（不小于 1） |
| `arith.maxsi/minsi` | 两边都是常量则算；否则 divisibility 取 gcd、constancy 取 gcd |
| `arith.extsi/extui/trunci`、`tt.bitcast`、`ttg.convert_layout` | **原样传递**——类型转换不改变值的规律；这是为什么 `offs.to(tl.int64)` 不丢信息 |
| `tt.load ptr, mask` | contiguity、divisibility 全 1（加载出来的**数据**没有规律可言）；constancy = `gcd(const(ptr), const(mask))`——从同一个地址加载的值当然相同 |
| `tt.trans` | 按置换重排三个向量 |
| `tt.reshape` | 有专门的合并 / 拆分规则（`ReshapeOpAxisInfoVisitor`，约 120 行），保证相邻维合并时连续性能传递 |

`load` 的规则是理解"AxisInfo 只关于地址、不关于数据"的一个好例子：`tl.load(ptr)` 的结果是内存里的数，编译器不可能知道它们是否连续；但如果 `ptr` 沿某一维 constancy 为 32（32 个线程读同一个地址），结果沿那一维也是 32 个相同的值——第十篇的 elementwise lowering 会用这个 constancy 做去重，让 32 个相同元素的运算只做一次。

## 五、汇合与循环：分析怎样收敛

### 1. `join`：一切取 gcd

两条路径在同一个值上汇合（`scf.if` 的两个分支各给结果一个值，`scf.for` 的 iter_arg 一个来自初值、一个来自 `yield`），用 `AxisInfo::join`：

```cpp
contiguity[d]   = gcd(lhs.contiguity[d], rhs.contiguity[d]);
divisibility[d] = getDivisibilityFromContiguity(lhs, rhs, d);
constancy[d]    = gcd(lhs.constancy[d], rhs.constancy[d]);
constantValue   = 两边相等则保留，否则丢弃;
```

`getDivisibilityFromContiguity` 是 gcd 加一个修正：如果两边 contiguity 不同（例如一边是 `[0,1 | 4,5]` contiguity 2，一边是 `[16,17,18,19]` contiguity 4），结果 contiguity 变小了，原来在段中间的元素变成了段首，它们的对齐没有保证——所以 divisibility 还要与两边的 contiguity 取 gcd。

gcd 是格上的 meet：结果比两边都保守，且是最不保守的那个。`join(x, x) = x` 保证不动点存在；一旦一条路径给出 1，结果就是 1——**任何一条控制流路径丢了信息，汇合点就丢**。

### 2. 建在 MLIR 数据流框架上

`AxisInfoAnalysis` 继承 `mlir::dataflow::SparseForwardDataFlowAnalysis<Lattice<AxisInfo>>`（第四篇讲过这个框架）。它只需要实现三件事：

| 方法 | 做什么 |
|---|---|
| `setToEntryState(lattice)` | 给一个值悲观初值（§三） |
| `visitOperation(op, operands, results)` | 查 visitor 表算结果；用 op 上的提示属性覆盖；`propagateIfChanged(result, result->join(curr))` |
| `visitNonControlFlowArguments(...)` | `scf.for` 的归纳变量：divisibility = `gcd(div(lb), div(step))`——`for k in range(0, K, 32)` 的 `k` 是 32 的倍数；其他 block 参数：悲观初值 |

框架负责其余一切：维护每个值的 lattice、按 use-def 边把变化传播给使用者、把 `scf.for` 的 `yield` 值汇合到下一轮的 block 参数、迭代到没有变化为止。**稀疏**（sparse）指传播沿 SSA 的 use-def 边而不是沿 CFG 逐块——只有操作数变了的 op 才重新算。

`visitOperation` 开头有一个细节：

```cpp
for (auto op : operands)
  if (op->getValue().getRank() == 0)   // 操作数还没有信息（rank 0 = 未初始化）
    return success();                  // 跳过，等操作数就绪后框架会再来
```

未初始化的 lattice 用 rank 0 表示，与"悲观值"（全 1）区分开——这就是第一篇说的 ⊥ 与 ⊤ 的区别：⊥ 是"还不知道"，join 时被另一边吸收；全 1 是"知道了，什么规律都没有"，join 时把另一边拉下来。

### 3. 循环：iter_args 的不动点

对 matmul 的 K 循环：

```text
scf.for %k = %c0 to %K step %c32 iter_args(%acc = %acc0, %a_ptrs = %a_ptrs0, %b_ptrs = %b_ptrs0) {
  ...
  %a_ptrs_next = tt.addptr %a_ptrs, %cst32_splat
  scf.yield %acc_new, %a_ptrs_next, %b_ptrs_next
}
```

`%a_ptrs`（block 参数）的 lattice = `join(初值 %a_ptrs0, yield 值 %a_ptrs_next)`。第一轮：只有初值到达，`%a_ptrs = info(%a_ptrs0)`；据此算出 `%a_ptrs_next = addptr(%a_ptrs, 32 元素 = 64 字节)`——加一个 divisibility 64、constancy 全长的常量，contiguity 不变、divisibility 取 gcd 仍是 16；第二轮 join 结果与第一轮相同，不动点到达。所以**循环里的指针保持了循环外的对齐信息**，前提是每轮的增量本身对齐（`BLOCK_K * stride` 是 16 字节的倍数）。如果增量是 `n * stride`、`n` 是运行时值，`%a_ptrs_next` 的 divisibility 掉到 1（或元素大小），join 之后 `%a_ptrs` 也掉到 1——循环体里所有的 load 都失去向量化。

归纳变量 `%k` 的 divisibility 是 gcd(div(0), div(32)) = 32；如果用户用 `k` 做地址运算（`a_ptr + k`），这个信息就有用。

### 4. 分支

`scf.if` 的结果 = `join(then 分支 yield, else 分支 yield)`。lit 测试 `@if`：一支给 divisibility 8 的值、一支给 16，结果是 8。没有路径敏感——分析不会记住"在 then 里 cond 为真"。

### 5. 跨函数

`make_ttir` 把一切内联，所以 AxisInfo 通常在一个函数内就完成。`noinline=True` 的函数例外：`ModuleAxisInfoAnalysis` 沿调用图（`CallGraph<AxisInfoMapT>`）后序遍历——先分析调用者，把每个调用点实参的 AxisInfo（只处理标量）以 `gcd` 合并后写成被调函数参数的 `tt.divisibility` / `tt.contiguity` / `tt.constancy` 属性，再分析被调函数。一个函数被多处调用时拿到的是所有调用点的 gcd——第一篇 §五.4 说的"上下文不敏感"。

## 六、消费者：三个公式

分析算完之后存在 `ModuleAxisInfoAnalysis` 里，消费者通过三个方法读：

### 1. `getAlignment(ptr)`：每个线程能连续持有几个元素

```cpp
divisibility = axisInfo->getDivisibility(order[0]);          // 沿最快变化的维
maxMultiple  = isPointer ? max(divisibility / elemBytes, 1) : divisibility;   // 字节 → 元素
maxContig    = axisInfo->getContiguity(order[0]);
alignment    = min(maxMultiple, maxContig);
```

`order[0]` 是 layout 里最快变化的维（第七篇），TTIR 阶段还没有 layout 时取最后一维。**alignment = min(对齐能撑几个元素, 连续能撑几个元素)**。对 matmul 的 `a_ptrs`（bf16，第 1 维 contiguity 32、divisibility 16 字节）：maxMultiple = 16 / 2 = 8，maxContig = 32，alignment = **8**——每个线程最多连续拿 8 个 bf16 = 16 字节 = 一条 128 bit load。

### 2. `getContiguity(ptr)` 与向量宽度

Coalesce（`getNumElementsPerThread`，`Transforms/Utility.cpp`）在 alignment 之上再与该 op 允许的每线程最大元素数取 min，得到 layout 的 `sizePerThread`。load / store lowering（`LoadStoreOpToLLVM.cpp` 的 `getVectorSize`）：

```cpp
return std::min<unsigned>(128 / pointeeBitWidth, contiguity);   // NVIDIA 最宽 128 bit
```

这里的 `contiguity` 是 `ModuleAxisInfoAnalysis::getContiguity`：alignment 再与 layout 实际给每线程的连续元素数（`getContigPerThread`）取 min。所以**向量宽度 = min(128 bit 能装几个, 对齐撑几个, 连续撑几个, layout 给了几个)**——四个上界，任何一个掉到 1 就是标量访存。

### 3. `getMaskAlignment(mask)`：一个谓词管几个元素

```cpp
alignment = max(axisInfo->getConstancy(maskOrder[0]), 1);
```

load / store lowering 把向量宽度再与它取 min：`vec = min(vec, getMaskAlignment(mask))`。mask 沿最快维 constancy 16、数据向量宽度 8 → 向量 8，一个谓词；mask constancy 1 → 向量退到 1，八个元素八个谓词八条指令。**这就是 `M`、`N` 是不是 16 的倍数会影响 store 性能的机制**（§四.5 的比较规则 + 这里的取 min）。

### 4. 谁在读

| 消费者 | 篇 | 读什么 |
|---|---|---|
| `Coalesce`、`CoalesceAsyncCopy` | 七 | `getNumElementsPerThread` → 选 load / store 的 `#blocked` layout 的 `sizePerThread` 与 `order` |
| `LoadStoreOpToLLVM`（load / store / atomic） | 十 | `getVectorSize`、`getMaskAlignment` → 生成几条、多宽的 `ld.global` / `st.global`，几个谓词 |
| `ElementwiseOpToLLVM` | 十 | 操作数的 constancy → 相同元素只算一次（`dedup-by-constancy`） |
| Pipeliner 的 `AssignLatencies` / `LowerLoops` | 九 | 判断 load 是否适合走 `cp.async`（需要 16 字节对齐与连续） |
| Gluon 的 `InferCoalescedEncodings` | 九 | 为 `AutoLayout` 推 coalesced layout |
| AMD 后端多处 | 十一 | buffer op 转换、LDS 旁路等 |

## 七、信息怎么丢

回到核心问题，用第四章的规则把三个变体走一遍（`BLOCK = 1024`，`pid = tt.get_program_id`，`n: i32`）。

| 步骤 | `pid * 1024 + arange` | `pid * n + arange`（`n` 无提示） | `pid * n + arange`，`n = tl.multiple_of(n, 16)` |
|---|---|---|---|
| `pid` | [1] [1] [1] | 同 | 同 |
| 乘数 | `constant 1024`：[1] [1024] [1] | `%n`：[1] [1] [1] | `%n` 定义处挂 `tt.divisibility = 16`：[1] [16] [1] |
| `pid * …` | contig 1；div 1 × 1024 = **1024** | div 1 × 1 = **1** | div 1 × 16 = **16** |
| `splat` | [1] [1024] [1024] | [1] [1] [1024] | [1] [16] [1024] |
| `arange(0, 1024)` | [1024] [2^30] [1] | 同 | 同 |
| `addi` | contig = gcd(1024, 1024) = **1024**；div = gcd(1024, 2^30, 1024) = **1024** | contig **1024**；div = gcd(1, 2^30, 1024) = **1** | contig **1024**；div = gcd(16, 2^30, 1024) = **16** |
| `addptr(splat(ptr, div 16), offs)`，f32 | contig 1024；div gcd(16, 1024 × 4) = 16 → alignment min(16/4, 1024) = **4**（128 bit） | contig 1024；div gcd(16, 1 × 4) = 4 → alignment min(4/4, 1024) = **1**（标量） | div gcd(16, 16 × 4) = 16 → alignment **4** |

三个结论：

1. **contiguity 在三种情况下都是 1024**——`arange` 提供的连续性不会被一个不对齐的偏移破坏。丢的是 divisibility。
2. 没有提示时，`pid * n` 的 divisibility 是 1，整条链的对齐信息只剩指针参数自己的 16 字节，与偏移 gcd 之后是 4 字节（一个 f32），**每线程只能拿 1 个元素**，load 退化成 `ld.global.b32`。
3. `tl.multiple_of(n, 16)` 只是在 `%n` 那个值上把 divisibility 从 1 改成 16，后面的传递规则不变，结果完全恢复。它必须写在 `n` 被乘之前——写成 `tl.multiple_of(pid * n, 16)` 也行（挂在 `muli` 上），写成 `tl.multiple_of(offs, 16)`（挂在 `addi` 上）同样有效但已经绕过了推理。

其他常见的信息杀手，各给一行规则：

| 写法 | 丢什么 | 原因（对应规则） |
|---|---|---|
| `offs % n`，`n` 运行时值 | contiguity → 1 | `rem`：需要 `n` 的 divisibility 参与 gcd，`n` 只有 1 |
| `offs % 64`，常量 | contiguity 保留到 gcd(contig, div(offs), 64) | 常量有 divisibility |
| `offs * stride`，`stride` 运行时且非 1 | contiguity → 1；divisibility → div(stride)（乘上 offs 按 1 计） | `mul`：只有乘 1 保留连续；连续侧的 divisibility 按 1 计 |
| `stride` 传 1 但放进了 `do_not_specialize` | 同上（编译器不知道它是 1） | 特化被关掉，`stride` 是普通 `i32` |
| `offs_a + offs_b`，两者都是 `arange` | contiguity → 1，divisibility ≤ 2 | 两个连续量相加步长为 2 |
| `tl.where(cond, offs, other)` | 三个属性都与 `cond` 的 constancy 取 gcd | `select` |
| `offs // 2 * 2` | contiguity → 1（`// 2` 已经不连续）；divisibility 保留 2 | `div` 后再 `mul` |
| `ptr + offs` 其中 `ptr` 的 `data_ptr` 不对齐（切片视图 `x[1:]`） | 指针参数没有 `D`，divisibility 按元素大小算 | binder 的 `(data_ptr & 15) != 0` |
| 循环里 `ptrs += n * stride`，`n` 运行时 | 整条循环携带链的 divisibility → 1 | iter_arg 的 join 把 yield 的 1 传回来 |

## 八、工具：把每个值的 AxisInfo 打出来

`triton-opt -test-print-alignment` 对输入的每个 op 以 remark 形式打印结果值的 AxisInfo，lit 测试用 `expected-remark @below` 核对。对上一篇拿到的 matmul TTIR 直接跑：

```bash
triton-opt matmul_kernel.ttir -test-print-alignment -o /dev/null 2>&1 | head -60
```

`triton-opt` 从 Triton 的构建目录里拿（`build/cmake.*/bin/triton-opt`，在 macOS 上也能构建）。每条 remark 带源码位置——`test-print-alignment` 用 op 的 `loc` 报告，所以能看到每个值来自 Python 的哪一行。把位置去掉、只留 op 与结果，matmul kernel 的地址链是（对照第五篇 §五.3 的 TTIR，SSA 名字这里是编号形式）：

```text
%0 = tt.get_program_id x : i32                               => contiguity = [1], divisibility = [1], constancy = [1]
%2 = arith.muli %0, %c128_i32 : i32                          => contiguity = [1], divisibility = [128], constancy = [1]
%3 = tt.make_range {end = 128, start = 0} : tensor<128xi32>  => contiguity = [128], divisibility = [1073741824], constancy = [1]
%4 = tt.splat %2 : i32 -> tensor<128xi32>                    => contiguity = [1], divisibility = [128], constancy = [128]
%5 = arith.addi %4, %3 : tensor<128xi32>                     => contiguity = [128], divisibility = [128], constancy = [1]      ; offs_m
%9 = tt.make_range {end = 32, start = 0} : tensor<32xi32>    => contiguity = [32], divisibility = [1073741824], constancy = [1] ; offs_k
%10 = tt.expand_dims %5 {axis = 1} : -> tensor<128x1xi32>    => contiguity = [128, 1], divisibility = [128, 1], constancy = [1, 1]
%11 = tt.splat %arg6 : i32 -> tensor<128x1xi32>              => contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 1]   ; stride_am
%12 = arith.muli %10, %11 : tensor<128x1xi32>                => contiguity = [1, 1], divisibility = [16, 16], constancy = [1, 1]
%13 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
                                                             => contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 1]   ; a_ptr
%14 = tt.addptr %13, %12                                     => contiguity = [1, 1], divisibility = [16, 16], constancy = [1, 1]
%15 = tt.expand_dims %9 {axis = 0} : -> tensor<1x32xi32>     => contiguity = [1, 32], divisibility = [1, 1073741824], constancy = [1, 1]
%16 = tt.broadcast %14 : -> tensor<128x32x!tt.ptr<bf16>>     => contiguity = [1, 1], divisibility = [16, 16], constancy = [1, 32]
%17 = tt.broadcast %15 : -> tensor<128x32xi32>               => contiguity = [1, 32], divisibility = [1, 1073741824], constancy = [128, 1]
%18 = tt.addptr %16, %17                                     => contiguity = [1, 32], divisibility = [2, 16], constancy = [1, 1]     ; a_ptrs
```

逐行核对第四章的规则：

- `%2`：`pid * 128`，乘法规则——`pid` 不连续，divisibility 1 × 128 = 128；
- `%5`：`splat + arange`，contiguity = gcd(const 128, contig 128) = 128，divisibility = gcd(128, $$2^{30}$$, 128) = 128——核心问题第一问在 128 上的实例；
- `%10`：`expand_dims` 插入第 1 维，新维 contiguity 1、divisibility 1（第 0 维 contiguity > 1 按 1 计）；
- `%12`：`offs_m[:, None] * stride_am`——第 0 维连续侧按 1 计，1 × 16 = 16；contiguity 掉到 1（乘数不是 1）；
- `%14`：`addptr(splat(a_ptr), %12)`，两边都不连续，divisibility = gcd(16, 16 × 2) = 16；
- `%16` / `%17`：两个 `broadcast`——被广播的维得到 constancy 32 / 128；
- `%18`：**`a_ptrs`**。第 1 维：contiguity = gcd(const 32, contig 32) = 32；divisibility = gcd(16, $$2^{30}$$ × 2, 32 × 2) = 16。第 0 维：contiguity 1；divisibility = gcd(16, 1 × 2) = **2**——沿 M 方向相邻两行的指针相差 `stride_am` 个元素，行首的对齐只能保证到一个 bf16。

消费者的读数：`getAlignment(%18)` 沿 `order[0] = 1`：maxMultiple = 16 / 2 = 8，maxContig = 32，alignment = **8**——每线程 8 个 bf16 = 一条 128 bit load。这正是下一篇 Coalesce 给 `a_ptrs` 的 `#blocked<{sizePerThread = [1, 8], …}>` 里那个 8 的来源。

循环体与写回部分：

```text
%47 = tt.addptr %arg10, %cst_0                              => contiguity = [1, 32], divisibility = [2, 16], constancy = [1, 1]     ; a_ptrs += 32
%48 = arith.muli %arg7, %c32_i32 : i32                       => contiguity = [1], divisibility = [512], constancy = [1]             ; stride_bk * 32
%50 = tt.addptr %arg11, %49                                  => contiguity = [1, 128], divisibility = [2, 16], constancy = [1, 1]    ; b_ptrs += ...
%28:3 = scf.for ... (result #0)                              => contiguity = [1, 32], divisibility = [2, 16], constancy = [1, 1]
%44 = tt.load %arg10                                         => contiguity = [1, 1], divisibility = [1, 1], constancy = [1, 1]      ; 加载的数据：无信息
%38 = arith.cmpi slt, %10, %37 : tensor<128x1xi32>           => contiguity = [1, 1], divisibility = [1, 1], constancy = [16, 1]     ; offs_m[:, None] < M
%40 = arith.cmpi slt, %24, %39 : tensor<1x128xi32>           => contiguity = [1, 1], divisibility = [1, 1], constancy = [1, 16]     ; offs_n[None, :] < N
%43 = arith.andi %41, %42 : tensor<128x128xi1>               => contiguity = [1, 1], divisibility = [1, 1], constancy = [16, 16]    ; mask
```

- `%47`：循环里 `a_ptrs += 32`（64 字节，divisibility 64）之后仍是 `[1, 32]`、`[2, 16]`——与初值相同，所以 iter_arg `%arg10` 的 join 不动点就是这个值（`scf.for` 结果 #0 的那一行），循环内的 load 保住了 128 bit；
- `%48`：`stride_bk * 32` 的 divisibility 是 16 × 32 = 512——两个都不连续的量相乘，divisibility 相乘；
- `%44`：`tt.load` 的结果全 1——数据没有规律；
- `%38` / `%40`：两个比较各得到 constancy 16（`M`、`N` 的 `tt.divisibility = 16` 与 `offs` 的 contiguity 128、divisibility 128 取 gcd），`andi` 后 mask 是 `[16, 16]`。`getMaskAlignment` 沿 `order[0] = 1` 读到 16，与数据向量宽度 8 取 min 后是 8：**写回的 store 每 8 个 bf16 一条指令、一个谓词**。若 `M`、`N` 传的不是 16 的倍数，这里是 `[1, 1]`，store 退化为 8 条带独立谓词的标量指令。

另一条路是 `TRITON_ENABLE_LLVM_DEBUG=1 TRITON_LLVM_DEBUG_ONLY=axis-info`（源码里的 `#define DEBUG_TYPE "axis-info"`）：编译时打印每次 `getAlignment` / `getContiguity` 的输入输出（`LDBG` 宏），能直接看到 Coalesce 与 lowering 拿到了什么数——第十三篇讲这套调试开关。

## 九、对照：LLVM 里的同类分析

| 分析 | 对象 | 格 | 与 AxisInfo 的关系 |
|---|---|---|---|
| `KnownBits` / `computeKnownBits` | 标量整数 | 每一位是 0 / 1 / 未知 | divisibility 是它的子集：低 k 位已知为 0 ⟺ 是 $$2^k$$ 的倍数；LLVM 用它做同样的对齐推理（`getelementptr` 的对齐、`load` 的 `align` 属性） |
| `ScalarEvolution` | 循环里的整数 | 递推表达式 `{start, +, step}` | contiguity 是它在"步长为 1"这个特例上的张量版；SCEV 能表达任意仿射递推，AxisInfo 只记段长 |
| `Alignment` / `getOrEnforceKnownAlignment` | 指针 | 2 的幂 | 与 divisibility 同义 |
| `ConstantRange` / `IntegerRangeAnalysis`（MLIR 也有） | 整数 | 区间 [lo, hi] | 正交：AxisInfo 不记范围，所以它不知道 `offs < n` 是否恒真——那需要区间分析 |

AxisInfo 的独特之处在于它是**按张量维度**的：LLVM 的分析作用在标量上，一个 128 元素的数组的连续性在 LLVM 里要靠 SCEV 分析循环归纳变量才能得到；Triton 把张量当作一个值，一次分析就给出每一维的规律。这正是第一篇说的"在信息还在的那一层做分析"：连续性在张量层是 `make_range` 一个 op 的属性，到了标量层就变成了要重新发现的东西。

## 十、本文小结

1. AxisInfo 给每个整数 / 指针值沿每一维三个 2 的幂：contiguity（连续段长）、divisibility（段首的 2 幂因子，指针以字节计）、constancy（相等段长），加一个可选的常量值。divisibility 定义在段首上，因为它的用途是向量访存起点的对齐。
2. 信息的入口：函数参数的 `tt.divisibility = 16`（来自 binder 的 `D`）、`tl.multiple_of` 一族（只是在定义 op 上挂属性覆盖结果，不是运行时检查）、常量与 `make_range`。`get_program_id` 是悲观的——`pid` 一无所知，对齐只来自乘上去的常量。`tl.assume` 走 LLVM，AxisInfo 不读。
3. 传递规则的核心几条：`splat` 给 constancy；`add` 的 contiguity 是 `max(gcd(const(a), contig(b)), gcd(contig(a), const(b)))`；两个连续量相加变步长 2；`mul` 只有乘 1 保留连续，连续侧的 divisibility 按 1 计；`rem` 需要除数的 divisibility；`cmp` 只产生 constancy，连续 < 常量时为 `gcd(contig, div, div)`；`load` 的结果只有 constancy；类型转换原样传。
4. 汇合取 gcd（contiguity 不同时 divisibility 再与 contiguity 取 gcd）；`scf.for` 的 iter_arg 是初值与 yield 的 join，增量对齐则循环内保持对齐；归纳变量的 divisibility 是 `gcd(lb, step)`。框架是 MLIR 的 `SparseForwardDataFlowAnalysis`，未初始化（⊥）与全 1（⊤）区分。
5. 消费公式：alignment = min(divisibility / 元素字节数, contiguity)；向量宽度 = min(128 bit 能装几个, alignment, layout 给的连续数)；mask 的 constancy 再取一次 min。`M % 16 == 0` 通过 `cmp` 的 constancy 规则影响 store 的谓词数。
6. 丢信息的常见写法：运行时值做乘数或模数、两个 `arange` 相加、不对齐的切片指针、循环增量不对齐。`tl.multiple_of` 恢复的只是被它直接标注的那个值。

## 十一、自测

1. `offs = tl.arange(0, 64) * 2`，`ptrs = x_ptr + offs`（f32，`x_ptr` 对齐 16）。`offs` 与 `ptrs` 的 contiguity / divisibility 各是多少？每线程能连续拿几个元素？

   <details markdown="1"><summary>答案</summary>
   `arange(0,64)`：[64] [2^30] [1]。`× 2`（常量 2，div 2）：contiguity 1（乘数不是 1）；divisibility：lhs 连续按 1 计，1 × 2 = 2；→ [1] [2] [1]。`addptr(splat(x_ptr, div 16), offs)`：偏移 div 2 × 4 字节 = 8；contiguity max(gcd(64, 1), gcd(1, 1)) = 1；divisibility gcd(16, 8) = 8；→ [1] [8] [64→1]。alignment = min(8 / 4, 1) = 1：每线程 1 个元素，标量访存——步长 2 的访问本来就不能向量化，分析结果与事实一致。
   </details>

2. `mask = offs < n`，`offs` 是 `pid * 256 + arange(0, 256)`，`n` 的实参是 1000。mask 沿这一维的 constancy 是多少？换成 1024 呢？`n` 放进 `do_not_specialize` 后传 1024 呢？

   <details markdown="1"><summary>答案</summary>
   1000 不是 16 的倍数，binder 不给 `D`，`n` divisibility 1：constancy = max(gcd(1, 256), gcd(256, 256, 1)) = 1。1024：`D` → divisibility 16 → gcd(256, 256, 16) = 16。`do_not_specialize` 关掉 `D` 标记，即使传 1024 也是 divisibility 1 → constancy 1。（1024 不会像 1 那样被提升为 constexpr——只有等于 1 的整数才提升。）
   </details>

3. 下面两种写法，哪一种能让 `b_ptrs` 沿 K 维向量化？为什么？

   ```python
   # A
   b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :]
   # B
   b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :]) % (K * N)
   ```

   <details markdown="1"><summary>答案</summary>
   A 沿第 1 维（`offs_n`）连续；B 不能——`% (K * N)` 的除数是运行时值，`rem` 的 contiguity 规则需要 `gcd(contig, div(lhs), div(rhs))`，`K * N` 的 divisibility 是 `div(K) × div(N)`（K、N 有 `D` 时是 256，否则 1）；即使是 256，`gcd(128, div(lhs), 256)` 里 `lhs` 沿第 1 维的 divisibility 是 `stride_bk` 项贡献的 16 与 `offs_n` 的 gcd，通常得到 16 而不是 128，向量化仍可能保留一部分；但若 K、N 任一没有 `D`，contiguity 直接掉到 1。取模在这里没有语义必要（写回时用 mask 更合适），是典型的"为了安全多写一步、把信息丢光"的写法。
   </details>

4. `for` 循环 `for k in range(0, K, BLOCK_K)`，循环体里 `a_ptrs += BLOCK_K * stride_ak`。若 `stride_ak` 传的是 3（所以是普通 `i32`），`a_ptrs` 在循环内的 divisibility 是多少？与 `stride_ak = 1` 相比 load 的向量宽度差多少？

   <details markdown="1"><summary>答案</summary>
   `stride_ak = 3`：循环外 `offs_k * 3` 的 contiguity 已经是 1（乘数不是 1），divisibility 1 × 1 = 1（`3` 的 2 幂因子是 1）；`a_ptrs` 初值沿 K 维 contiguity 1、divisibility gcd(16, 1 × 2) = 2。循环增量 `32 × 3 = 96` 元素 = 192 字节，divisibility 64；join 后仍是 2。alignment = min(2 / 2, 1) = 1 → 标量 load（每线程 1 个 bf16）。`stride_ak = 1`：alignment 8 → 128 bit。差 8 倍的访存指令数——而且步长 3 的访问在硬件上本来就不合并，分析没有冤枉它。
   </details>

5. `tl.multiple_of(x, 16)` 与 `tl.assume(x % 16 == 0)` 对 Triton 生成的 PTX 各有什么影响？哪一个可能让错误的提示产生错误的结果？

   <details markdown="1"><summary>答案</summary>
   `multiple_of` 把 `x` 定义 op 上的 `tt.divisibility` 设为 16，AxisInfo 直接采用，影响 Coalesce 的 layout 选择和 load / store 的向量宽度——提示错误时编译器会生成对未对齐地址的 128 bit 访存，运行时结果错误或非法地址错误。`assume` 生成 `llvm.intr.assume`，只被 LLVM 后端的优化用到（可能改善标量地址运算的指令选择），对 Triton 层的 layout 与向量化没有影响；提示错误在 LLVM 里同样是未定义行为，但通常影响更小。两者都不是运行时检查。
   </details>

## 下一篇

AxisInfo 给出了每个访存指针"每线程最多能连续拿几个元素"的上界。但 TTIR 里的张量还没有任何线程的概念——`tensor<128x32xbf16>` 只是 4096 个元素。下一篇进入 Triton 编译器最核心的一层：**layout**。每个张量类型带上一个 encoding 属性，精确规定 4096 个元素在 4 个 warp、128 个线程、每线程若干寄存器之间怎样分布；`#blocked`、`#slice`、`#nvidia_mma`、`#dot_op`、`#shared` 各是什么；Triton 3.x 用 $$\mathbb{F}_2$$ 上的线性映射（Linear Layout）把它们统一成一种数学对象；`ConvertTritonToTritonGPU` 怎样给每个张量选初始 layout；Coalesce 怎样拿本篇的 alignment 为 load / store 重选 layout。

[^q0]: 沿使用者的推理链：`tt.get_program_id` 没有 visitor，落到悲观值 [1] [1] [1]；`arith.constant 1024` 的 divisibility 是 1024；`arith.muli %pid, %c1024` 按乘法规则 divisibility = 1 × 1024 = 1024（`pid` 不连续，不按 1 计），contiguity 1；`tt.splat` 把它变成 tensor，constancy = 1024、divisibility 1024、contiguity 1；`tt.make_range {0, 1024}` 是 contiguity 1024、divisibility $$2^{30}$$、constancy 1；`arith.addi` 的 contiguity = max(gcd(const(lhs)=1024, contig(rhs)=1024), gcd(contig(lhs)=1, const(rhs)=1)) = 1024，divisibility 走"至少一边不连续、且有一边连续"的分支 = gcd(1024, $$2^{30}$$, 结果 contiguity 1024 × 1) = 1024。所以 `offs`：contiguity 1024、divisibility 1024。`test-alignment.mlir` 的 `@store_constant_align` 是同一条链在 128 上的实例，结果 [128] [128] [1]。详见[第四章 §2、§3](#四传递函数每种-op-的规则)与[第七章](#七信息怎么丢)。

[^q1]: contiguity 不变，仍是 1024——`arange` 的连续性与偏移是否对齐无关。divisibility 掉到 1：`%n` 是普通 `i32` 参数，没有 `D` 标记（用户没传 16 的倍数，或它是从别处算出来的），divisibility 1；`pid * n` = 1 × 1 = 1；`splat` 后 divisibility 1；`addi` 的 divisibility = gcd(1, $$2^{30}$$, 1024) = 1。随后 `addptr(splat(ptr, 16 字节), offs)` 的 divisibility 是 gcd(16, 1 × 元素字节数) = 元素字节数（f32 为 4），alignment = min(4 / 4, 1024) = 1：每线程一个元素，load / store 退化为标量指令，尽管数据在内存里其实是连续的——编译器证明不了每个线程那一段的起点对齐。详见[第七章](#七信息怎么丢)。

[^q2]: `tl.multiple_of(n, 16)` 在 `semantic.multiple_of` 里只做一件事：`n.handle.set_attr("tt.divisibility", [16])`，把属性挂在定义 `%n` 的 op（或函数参数）上。`visitOperation` 算完 `%n` 的 AxisInfo 后用这个属性覆盖 divisibility 为 16；之后的链按原规则走：`pid * n` divisibility 1 × 16 = 16，`splat` 16，`addi` = gcd(16, $$2^{30}$$, 1024) = 16，`addptr` = gcd(16, 16 × 4) = 16，alignment = min(16 / 4, 1024) = 4——恢复到 128 bit 向量访存。它不是运行时检查：若 `n` 实际不是 16 的倍数，生成的向量访存会读写错位的地址。`tl.assume(n % 16 == 0)` 不能替代它，因为 `assume` 只进 LLVM，AxisInfo 不读。详见[第三章](#三起点信息从哪里进入)与[第七章](#七信息怎么丢)。
