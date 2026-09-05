---
layout: post
title: "PyTorch 深度实践（10）：PyTorch 的工程体系——一次改动如何安全地到达用户"
subtitle: "The Engineering System of PyTorch: How a Change Travels Safely from Commit to Production"
tags: [PyTorch, AI, AI-Infra]
catalog: true
---

> 本文是[《PyTorch 深度实践：从 Tensor 到深度学习运行时》](/deep-dive-into-pytorch.html)系列的第十篇（共十篇）。上一篇：[分布式 PyTorch](/pytorch-distributed-training.html)

前九篇讲的是 PyTorch **是什么、怎么运行**：Tensor 怎么存、Autograd 怎么记、算子怎么分发、Kernel 怎么写、编译器怎么融合、性能怎么测、多卡怎么通信。每一篇都在描述一个已经存在、并且正确运行的系统。

这一篇换一个问题：**它是怎么做到一直正确、一直可用的？**

PyTorch 有两千多个算子、每个算子有十几种 dtype、两个以上后端、无数种 shape 和 stride 组合，还要在 eager、`torch.compile`、Meta、Autograd、分布式等模式下行为一致。每天有几十个 PR 合入主干，每个 PR 都可能碰到其中任何一层；每三四个月发一个版本，几百万用户的代码、几十个硬件后端、无数已保存的 checkpoint 都要在新版本上继续工作。

> **一个框架的工程体系，就是一次改动从写下到进入用户生产环境所经过的全部关卡，以及每个关卡守住什么。**

本文沿着这条路走一遍：改动先要在开发者机器上**构建**起来，然后证明自己**正确**、**没有变慢**，通过**审查与 CI 合入**主干，随一个版本**发布**成二进制制品，最后到达用户手里时**不能破坏**他们的代码、扩展和 checkpoint。走完这条路，再站到使用者一侧，看 AI-Infra 工程师应当怎样跟随这条演进线；最后把整套方法收回到第六篇的自定义算子上，作为全系列的实践终点。

与前几篇的关系：第六篇讲过单个自定义算子怎么用 `opcheck`、`gradcheck`、Benchmark 验证以及 ABI 为什么会坏，本文不重复那些 API 的用法，而是回答它们背后的问题——PyTorch 自己是怎么系统性地做这些事的。


## 一、总览：一次改动的七个关卡

### 1. 生命周期

以一个典型改动为例——给某个 CUDA Kernel 换一种更快的实现。它从写下到进入用户的训练作业，要经过：

```text
写下改动
   ↓
① 本地构建能跑        改一个 .cu 文件后多久能重新编译、加载、跑通一个测试？          → 第二章
   ↓
② 结果正确            新 Kernel 与旧实现、与 CPU、与 fp64 参考、与 compile 后一致吗？ → 第三章
   ↓
③ 没有变慢            它快了，有没有让别的 shape 慢？CPU 侧开销有没有变化？          → 第四章
   ↓
④ 审查与 CI 合入       谁有权批准？几十万个测试怎么在两小时内跑完？红了怎么办？      → 第五章
   ↓
⑤ 随版本发布          进 nightly 还是等下个小版本？编成多少个 wheel？                → 第六章
   ↓
⑥ 用户升级不坏        用户的代码、C++ 扩展、保存的 checkpoint 在新版本上还能用吗？   → 第七章
   ↓
⑦ 使用者跟随演进       用户侧：什么时候升级、怎么升级、升级坏了怎么回退？            → 第八章
```

前六关是框架维护者的视角，第七关是使用者的视角。两者对 AI-Infra 工程师都重要：读源码、给 PyTorch 提 PR、维护树外后端时是前者；升级集群、维护自己的 C++ 扩展、保证 checkpoint 可加载时是后者。

### 2. 关卡守住的三件事

七个关卡并非各守一样东西。归纳起来，整条链路要给出三个保证：

```text
正确性    改了之后结果还对            主要由 ② 守；④ 把它自动化
性能      改了之后没有变慢            主要由 ③ 守；④⑤ 用看板持续监测
兼容性    改了之后用户的东西还能用     主要由 ⑥ 守；⑤ 的制品矩阵和 ⑦ 的升级实践是它的两端
```

构建（①）不守任何一项，但三项都要在它的产物上验证；合入（④）与发布（⑤）是流程，把前面的验证变成强制的、自动的。

### 3. 为什么框架的关卡与应用不同：组合爆炸

应用代码的测试是**枚举**：一个函数有几个分支，写几个用例覆盖。框架代码面对的是**组合爆炸**：

```text
测试空间 = 算子（2000+）
         × dtype（float16 / bfloat16 / float32 / float64 / complex / int8 ... 约 15 种）
         × 设备（CPU / CUDA / MPS / XPU ...）
         × shape（标量 / 空 Tensor / 一维 / 高维 / 广播）
         × layout（contiguous / non-contiguous / channels_last / 转置视图 / 有 offset 的切片）
         × 执行模式（eager / compile / Meta / Autograd 一阶 / 二阶 / forward AD / 分布式）
```

哪怕每个维度只取几个值，乘起来也是几十万个测试点。手写不可能，每个 PR 全跑一遍也不可能。这个事实塑造了第三章的数据驱动测试、第五章的 CI 分层，以及第七章"接口面按稳定程度分级"的策略——框架工程的大多数设计，都是在组合爆炸下求可行。

### 4. 本文的章节安排

```text
二    第一关：本地构建能跑            仓库地图 · 构建流程与 Codegen · 环境变量与构建类型 · 调试到 C++
三    第二关：结果正确                五种 oracle · OpInfo · 设备与 dtype 泛化 · gradcheck 内部 · 确定性 · 编译器与分布式测试 · test/ 导读
四    第三关：没有变慢                微基准与指令数 · TorchBench 与看板 · 噪声、阈值、归因
五    第四关：审查、CI 与合入          CI 分层与目标确定 · flaky 的流程化处理 · 审批规则、MergeBot、回滚、lintrunner
六    第五关：发布                    节奏与分支 · wheel 矩阵与 ABI · 平台支持窗口 · 治理
七    第六关：用户升级不坏             接口面的稳定分级 · Python API 弃用 · Schema BC/FC · C++ 与稳定 ABI、PrivateUse1 · 序列化版本
八    第七关：使用者如何跟随演进        版本策略 · 升级 playbook · 兼容矩阵 · 回退
九    实践终点：把第六篇的 myops 走完这七关
十    Java 对照
十一  系列总结：从 loss.backward() 一路追问到底
```


## 二、第一关：本地构建能跑

改动的第一步是在自己机器上把它编出来、加载、跑通一个测试。这一步多快，决定了后面所有关卡的反馈周期。理解构建体系还有第二个价值：遇到"在我机器上能跑"的问题时，知道二进制到底绑定了什么。

### 1. 仓库地图

```text
pytorch/
├── c10/                     核心基础设施（第二篇）：TensorImpl、Storage、Device、ScalarType、DispatchKey、Allocator、intrusive_ptr
│   ├── core/  cuda/  util/
├── aten/src/ATen/           Tensor 库（第五篇）
│   ├── native/              算子实现：native_functions.yaml 与 CPU 实现
│   │   ├── cpu/             CPU Kernel（向量化、TensorIterator 循环）
│   │   ├── cuda/            CUDA Kernel
│   │   └── ...              sparse/ quantized/ mkldnn/ 等后端
│   ├── core/                Tensor、TensorBase、Dispatcher、boxing
│   └── TensorIterator.*     第六篇用过的 stride 处理机制
├── torch/                   Python 包
│   ├── csrc/                C++ 侧的 Python 绑定与运行时
│   │   ├── autograd/        Autograd 引擎（第三篇）、generated/（Codegen 产物）
│   │   ├── jit/             TorchScript、序列化的 upgraders（第七章 §3）
│   │   ├── distributed/     c10d：ProcessGroup、NCCL 后端、Reducer（第九篇）
│   │   ├── dynamo/          Dynamo 的 C 扩展（帧求值 hook，第七篇）
│   │   ├── inductor/        AOTInductor 运行时、C shim（第七章 §4）
│   │   └── api/             C++ 前端（libtorch 的 torch::nn 等）
│   ├── nn/  optim/  fx/  distributed/  testing/      Python 实现
│   ├── _dynamo/  _inductor/  _functorch/             编译栈（第七篇）
│   ├── _refs/  _prims/  _decomp/                     参考实现与分解（第三章 §1）
│   └── _C/                  编译出的扩展模块的类型桩
├── torchgen/                Codegen（第五篇第五章）
├── tools/                   构建脚本、autograd/derivatives.yaml、linter、测试基础设施
├── test/                    测试（第三章 §7）
├── benchmarks/              基准（第四章）
├── third_party/             子模块：pybind11、fmt、cutlass、cudnn-frontend、gloo、oneDNN、XNNPACK、kineto、tensorpipe、protobuf...
├── cmake/  CMakeLists.txt   构建定义
├── setup.py                 Python 包入口，驱动 CMake
├── .github/                 CI workflows、merge_rules.yaml、pytorchbot 脚本（第五章）
├── .ci/                     Docker 镜像定义、CI 内的构建与测试脚本
└── docs/                    Sphinx 文档源
```

前九篇的每一个概念都能在这张图上找到位置。读源码时的路径通常是：Python API（`torch/`）→ 绑定（`torch/csrc/`）→ 算子声明（`native_functions.yaml`）→ 实现（`aten/src/ATen/native/`）→ 基础类型（`c10/`）。

### 2. 构建流程与 Codegen 的位置

```text
python setup.py develop（或 pip install -e . --no-build-isolation）
    ↓
tools/build_pytorch_libs.py：整理环境变量 → 调用 CMake configure
    ↓
CMake 生成 build.ninja；其中包含一条 custom command：运行 torchgen
    ↓
torchgen 读 native_functions.yaml、derivatives.yaml、tags.yaml → 生成
    build/aten/src/ATen/*.cpp（算子注册、at:: API、RegisterCPU.cpp / RegisterCUDA.cpp）
    torch/csrc/autograd/generated/*（VariableType：Autograd 包装、反向 Function 类）
    torch/csrc/autograd/generated/python_*_functions.cpp（Python 绑定）
    torch/_C/_VariableFunctions.pyi（类型桩）
    ↓
Ninja 编译上万个 .cpp / .cu → 链接成共享库
    libc10.so          libc10_cuda.so
    libtorch_cpu.so    libtorch_cuda.so      libtorch.so（伞形库）
    libtorch_python.so → 安装为 torch/_C.cpython-*.so
    ↓
develop 模式下 Python 源码原地使用；只有 C++ 改动需要重新构建
```

第五篇讲了 Codegen **生成什么**；这里的重点是它**何时运行**——它是构建的一步，`native_functions.yaml` 改一行，会触发大量生成文件重新编译。这就是为什么改一个算子的 Schema 比改它的实现要慢得多，也是第七章 §3 中"Schema 是契约、轻易不动"的工程侧理由。

### 3. 从源码构建

```bash
git clone --recursive https://github.com/pytorch/pytorch && cd pytorch
pip install -r requirements.txt && pip install cmake ninja
export CMAKE_PREFIX_PATH=$(python -c "import sysconfig; print(sysconfig.get_paths()['data'])")

# 常用开关
export USE_CUDA=1  TORCH_CUDA_ARCH_LIST="8.0;9.0"      # 只为需要的 GPU 架构编 Kernel，构建时间与二进制大小成倍下降
export USE_DISTRIBUTED=1  USE_NCCL=1
export BUILD_TEST=0                                    # 不编 C++ gtest，省时间
export MAX_JOBS=32                                     # 并行度；内存不足时降低（CUDA 文件每个编译进程可能占几 GB）
export CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache CMAKE_CUDA_COMPILER_LAUNCHER=ccache

python setup.py develop
```

三种构建类型：

```text
默认（Release）           -O3，无调试符号；最快
REL_WITH_DEB_INFO=1       -O2 -g；有符号可以用 gdb 看栈，优化后变量可能看不到；日常调试的选择
DEBUG=1                   -O0 -g；变量全部可见，断言全开；慢 5～10 倍，二进制大数倍，某些测试会超时
```

首次完整 CUDA 构建在 32 核机器上约 1～2 小时；之后增量构建只编译改动的文件及依赖它的文件——改一个 `.cu` 几分钟，改 `c10/` 里的头文件可能触发半数文件重编。`ccache` 让切分支后的重建命中缓存。只改 Python 代码不需要构建（develop 模式）。

`TORCH_CUDA_ARCH_LIST` 值得单独强调：不设置时会为所有支持的架构生成代码，构建时间和 `libtorch_cuda.so` 的体积都翻几倍。这也是第六篇 ABI 一节"no kernel image is available"错误的另一面。

### 4. 调试到 C++

Python 栈在 `torch._C` 处断掉。接上 C++ 栈的方法：

```bash
TORCH_SHOW_CPP_STACKTRACES=1 python test.py          # C++ 异常附带 C++ 栈（需要有符号的构建）
gdb --args python test.py                            # 在 gdb 里 run；C++ 崩溃时 bt 看栈
(gdb) break at::native::add_out                      # 在 C++ 函数上下断点，Python 调过来时停下
CUDA_LAUNCH_BLOCKING=1                               # CUDA 错误在 launch 处同步报出，而不是在之后某个同步点（第八篇）
compute-sanitizer python test.py                     # CUDA 的越界 / 竞态检测
```

第五篇的分发路径、第三篇的 Autograd 引擎，用 `REL_WITH_DEB_INFO` 构建加 gdb 单步走一遍，是理解它们最直接的方式。

到这里改动能在本机跑起来了。下一关：证明它是对的。


## 三、第二关：结果正确

### 1. "对"的定义：五种 oracle

测试要有一个"标准答案"（oracle）。应用测试的 oracle 是需求：输入 X 应该输出 Y。框架算子的 oracle 没那么直接——`torch.matmul` 的正确输出是什么？PyTorch 用五种 oracle，每种覆盖一类错误：

| oracle | 检查什么 | 抓什么 bug | 例子 |
|---|---|---|---|
| **参考实现** | 与一个更慢但更可信的实现比对 | 算法错误 | NumPy / SciPy；`torch._refs` 中用基本算子写的 Python 参考实现；fp64 下的同一算子 |
| **数学恒等式** | 结果满足某个必然成立的关系 | 反向公式错误 | `gradcheck`：解析梯度 vs 有限差分；`x.conj().conj() == x` |
| **跨后端一致** | CPU 与 CUDA 结果一致 | 某个 Kernel 的实现错误 | 同一 OpInfo 在两个设备上跑同一模板 |
| **跨模式一致** | eager 与 compile / out= 变体 / inplace 变体 / 视图 结果一致 | 编译器变换错误、变体实现不同步 | `test_variant_consistency_eager`、Inductor 的 OpInfo 测试 |
| **元数据一致** | Meta / Fake 实现的 shape、dtype、stride 与真实实现一致 | 编译器和 FSDP 依赖的"不算数只推 shape"路径出错 | `test_fake`、`opcheck` |

前两种回答"算得对不对"，后三种回答"不同路径是否一致"。**一个新算子至少要有第一种或第二种作为绝对标准**，否则跨后端一致只能证明两个实现错得一样。

参考实现值得多说一句。`torch._refs` 是一组用 PyTorch 基本算子（`torch._prims`）写成的 Python 参考实现，覆盖数百个算子。它同时是三样东西：测试 oracle、`torch.compile` 分解的来源（第七篇 AOTAutograd 的 decomposition 很多来自这里）、以及新后端的兜底实现。一份代码三种用途，是框架工程里"参考实现"的典型价值。

### 2. OpInfo：一处声明，万处生成

第一章 §3 的组合爆炸怎么解？把"算子是什么"声明一次，把"要检查什么"写成模板，让框架做笛卡尔积。

`torch/testing/_internal/common_methods_invocations.py` 里有一个几千行的列表 `op_db`，每个元素是一个 `OpInfo`，描述一个算子的**测试相关元数据**：

```python
OpInfo(
    "add",
    dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),   # CPU 支持哪些 dtype
    dtypesIfCUDA=...,                                                            # CUDA 支持哪些
    sample_inputs_func=sample_inputs_add_sub,        # 生成一组有代表性的输入：标量、广播、空 Tensor、alpha 参数...
    reference_inputs_func=...,                       # 更大更全的输入集，慢测试用
    error_inputs_func=...,                           # 应当报错的输入及期望的错误信息
    ref=np.add,                                      # NumPy 参考实现
    supports_forward_ad=True,                        # 支持 forward-mode AD
    supports_fwgrad_bwgrad=True,                     # 支持 forward-over-backward（二阶）
    supports_out=True,                               # 有 out= 变体
    skips=(
        DecorateInfo(unittest.skip("Skipped!"), "TestBinaryUfuncs", "test_reference_numerics_extremal",
                     dtypes=(torch.complex64,)),     # 精确到 测试类 × 测试函数 × dtype × 设备 的跳过
    ),
)
```

另一边，`test/test_ops.py` 等文件里的测试是**模板**，用 `@ops(op_db)` 装饰，表示"对 `op_db` 里每一个算子都生成一个实例"：

```python
class TestCommon(TestCase):
    @ops(op_db)
    def test_noncontiguous_samples(self, device, dtype, op):
        for sample in op.sample_inputs(device, dtype):
            expected = op(sample.input, *sample.args, **sample.kwargs)
            nc_sample = sample.noncontiguous()                       # 把所有 Tensor 变成非连续视图
            actual = op(nc_sample.input, *nc_sample.args, **nc_sample.kwargs)
            self.assertEqual(actual, expected)

    @ops(op_db)
    def test_numpy_ref(self, device, dtype, op): ...              # 与 op.ref 比对
    @ops(op_db)
    def test_variant_consistency_eager(self, device, dtype, op): ... # 函数 / 方法 / inplace / out= 变体一致
    @ops(op_db)
    def test_dtypes(self, device, dtype, op): ...                  # 声明支持的 dtype 真的支持，声明不支持的真的报错
    @ops(op_db)
    def test_out(self, device, dtype, op): ...                     # out= 变体的 shape / dtype / 别名语义
```

`@ops(op_db)` × `instantiate_device_type_tests`（下一节）把每个模板展开成 `test_noncontiguous_samples_add_cpu_float32`、`test_noncontiguous_samples_add_cuda_bfloat16`……几十个模板 × 两千个算子 × 十几个 dtype × 两个设备，生成几十万个测试实例。**添加一个算子的测试，就是往 `op_db` 加一个 `OpInfo`**——所有模板自动覆盖它。

OpInfo 的每个字段都是一个"声明"，而模板负责验证声明属实：声明 `supports_forward_ad=True`，就会有模板去检查 forward AD；声明某个 dtype，就会有模板去跑那个 dtype。声明与实际不符时测试失败——所以 `op_db` 同时是**算子能力的权威清单**，新后端（第七章 §4）用它衡量自己覆盖了多少。

`skips` 与 `xfail` 的区别是工程纪律：`skip` 是"不测"，`xfail`（`DecorateInfo(unittest.expectedFailure, ...)`）是"预期失败，一旦意外通过就报错"——后者防止 bug 被修好后 skip 条目永远留在那里。

### 3. 设备与 dtype 泛化

组合中设备和 dtype 两维靠 `torch/testing/_internal/common_device_type.py` 处理。一个测试类写一次，按设备实例化多份：

```python
from torch.testing._internal.common_device_type import instantiate_device_type_tests, dtypes, onlyCUDA, skipCUDAIfNoMagma

class TestFoo(TestCase):
    @dtypes(torch.float32, torch.float64)                # 每个 dtype 生成一个测试
    def test_basic(self, device, dtype):                 # device 是字符串："cpu" / "cuda:0"
        x = torch.randn(3, device=device, dtype=dtype)
        ...

    @onlyCUDA
    @dtypes(torch.half)
    def test_cuda_only(self, device, dtype): ...

instantiate_device_type_tests(TestFoo, globals())        # 生成 TestFooCPU、TestFooCUDA（有 GPU 时）
del TestFoo                                              # 泛型类本身不应被运行
```

`instantiate_device_type_tests` 为每种可用设备生成一个子类，测试名加上 `_cpu` / `_cuda` 后缀和 dtype 后缀。可用的装饰器包括 `@dtypes`、`@dtypesIfCUDA`（CUDA 上不同的 dtype 集）、`@onlyCPU` / `@onlyCUDA` / `@onlyNativeDeviceTypes`、`@skipCUDAIf(cond)`、`@precisionOverride({torch.half: 1e-2})`、`@largeTensorTest("20GB")`（显存不够时跳过）、`@deviceCountAtLeast(2)`。

这套机制对外部后端也开放：新设备通过 `common_device_type` 的注册接口挂进来，同一份测试就会在它上面实例化——这就是"OpInfo 是后端一致性套件"的含义。

**比较与容差**。`torch.testing.assert_close(actual, expected, rtol=..., atol=...)` 是标准比较函数，默认容差按 dtype：

```text
float64     rtol 1e-7     atol 1e-7
float32     rtol 1.3e-6   atol 1e-5
float16     rtol 1e-3     atol 1e-5
bfloat16    rtol 1.6e-2   atol 1e-5
```

它还检查 dtype、device、shape 是否相同（可关闭），处理 NaN（`equal_nan`）、复数、稀疏与量化 Tensor。第八篇讨论过混合精度的容差需要"有依据"；这张表就是依据的起点——bf16 只有 8 位尾数，1.6e-2 的相对误差是它本身的精度。

**构造输入**。`torch.testing.make_tensor(shape, device=, dtype=, low=, high=, noncontiguous=True, requires_grad=True)` 是 OpInfo 的 `sample_inputs_func` 大量使用的工具：对整数 dtype 给整数、对 bool 给 bool、可以要求非连续、可以排除零（测除法）、可以在指定区间取值。

### 4. gradcheck 的内部

第六篇说"用 `gradcheck`，必须 float64"。这一节说它做了什么，从而知道它能抓什么、抓不到什么。

`torch.autograd.gradcheck(fn, inputs)` 的核心是比较两个 Jacobian：

```text
数值 Jacobian    对每个输入元素 xᵢ 加减 eps，重新算前向，(f(x+eps) - f(x-eps)) / 2eps      → 与实现无关的真值
解析 Jacobian    对每个输出元素 yⱼ，用 one-hot 的 grad_output 调用 backward，得到 ∂yⱼ/∂x    → 被测的反向实现
```

两者逐元素在 `atol=1e-5, rtol=1e-3` 内一致则通过。为什么必须 float64：eps 默认 1e-6，float32 只有 7 位有效数字，`f(x+eps) - f(x-eps)` 的差本身就在舍入误差里；float64 有 16 位，差分才有意义。

**慢模式**的代价是 O(输入元素数) 次前向 + O(输出元素数) 次反向，大 Tensor 不可行。**快模式**（`fast_mode=True`）用随机向量投影：取随机 u、v，比较 `uᵀ J v` 的数值与解析结果——只需 O(1) 次前向和反向。理论上可能漏掉恰好与 u、v 正交的错误，实践中概率可忽略。PyTorch 自己的测试默认用快模式。

`gradcheck` 还检查几件容易忽略的事：

```text
check_undefined_grad    某些输出的 grad_output 为 None（未使用）时反向不崩
check_batched_grad      反向对 vmap 友好（批量 grad_output）
check_forward_ad        forward-mode AD（dual number）的结果与数值 Jacobian 一致
check_grad_dtypes       梯度 dtype 与输入一致
非确定性检测            同一输入算两次反向结果不同 → 报 nondeterministic 错误，而不是模糊的数值不匹配
```

`gradgradcheck` 对**反向函数本身**做 gradcheck：把 backward 当成一个从 (input, grad_output) 到 grad_input 的函数，检查它的导数。这抓的是"反向公式里某个中间量没有接入计算图"这类错误——一阶 gradcheck 发现不了，二阶优化（如 Hessian-vector product、梯度惩罚）时才会炸。对线性算子（第六篇的 `scale_shift`）二阶导恒为零，`gradgradcheck` 平凡通过；对 gelu、softmax、attention 这类非线性算子，它是必要的。

### 5. 确定性

框架测试有一个应用测试很少遇到的敌人：**同一份代码同一输入，两次结果不同**。来源：

```text
GPU 归约顺序      浮点加法不满足结合律，atomicAdd 的顺序随线程调度变化 → index_add、scatter_add、embedding 反向
cuBLAS / cuDNN    算法自动选择（不同 workspace、不同 split-K）→ 结果在容差内不同
随机数            Dropout、randn；多进程时各 rank 的种子
异步与竞态        Stream 依赖没建对，偶发读到旧数据
```

工具：`torch.use_deterministic_algorithms(True)` 让非确定性算子改用确定性实现或直接报错（配合 `CUBLAS_WORKSPACE_CONFIG=:4096:8`）；`torch.manual_seed` 与 `torch.testing._internal.common_utils.TestCase` 每个测试前重置种子；`freeze_rng_state()` 上下文。

即使如此，几十万个测试实例中总有一些偶发失败（flaky）。它不是靠更仔细写测试能消灭的，只能靠流程管理——放在第五章 §2 讲。

### 6. 编译器与分布式的测试

**编译器**的 oracle 是 eager：同一函数 `torch.compile` 后结果应与不编译一致。`test/inductor/test_torchinductor_opinfo.py` 把整个 `op_db` 在 Inductor 下跑一遍，与 eager 比对，容差放宽（融合改变了浮点运算顺序）；`inductor_expected_failures` 列出已知不支持的算子。Dynamo 测试关心捕获行为：`torch._dynamo.testing.CompileCounter` 统计触发了几次编译、几个 graph break，`fullgraph=True` 断言没有 break。第七篇讨论的 Guard、动态 shape、重编译都有对应的测试类。

**分布式**的困难是每个测试需要多个进程。`torch.testing._internal.common_distributed.MultiProcessTestCase` 为每个测试方法 spawn `world_size` 个进程，每个进程跑同一个测试函数，rank 从 `self.rank` 取；`@skip_if_lt_x_gpu(2)` 在 GPU 不够时跳过。它慢（每个测试重新初始化进程组），所以有两个轻量替代：`MultiThreadedTestCase` 用线程模拟 rank（共享一个进程，快，但只能测逻辑不能测 NCCL）；`FakeProcessGroup`（`FakeStore` 后端）让集合通信直接返回而不真正通信，用于单进程验证 DTensor 的 shape 推导和 FSDP 的分片逻辑。第九篇讨论的 DTensor 有专门的 `DTensorTestBase` 和 `@with_comms` 装饰器。

### 7. `test/` 目录导读

```text
test/
├── test_torch.py                 Tensor 基础操作、大杂烩（历史原因最大的文件）
├── test_ops.py                   OpInfo 模板：一致性、dtype、out=、非连续、参考实现
├── test_ops_gradients.py         OpInfo 模板：gradcheck / gradgradcheck
├── test_ops_fwd_gradients.py     OpInfo 模板：forward AD
├── test_autograd.py              Autograd 引擎本身：图构建、hook、异常、checkpoint
├── test_nn.py                    nn.Module 与各层
├── test_binary_ufuncs.py / test_unary_ufuncs.py / test_reductions.py    按算子类别的专项模板
├── test_tensor_creation_ops.py / test_view_ops.py / test_indexing.py
├── test_cuda.py                  CUDA 运行时：Stream、Event、Caching Allocator、Graphs
├── test_meta.py                  Meta 实现与真实实现一致
├── test_decomp.py                _refs / decomposition 与原算子一致
├── test_fake_tensor.py
├── dynamo/                       Dynamo：捕获、Guard、graph break、重编译
├── inductor/                     Inductor：codegen、融合、OpInfo 全量、CUDA Graphs
├── distributed/                  进程组、DDP、FSDP、TP、PP、DTensor、checkpoint
├── cpp_extensions/               C++ 扩展的构建与加载
├── cpp/                          C++ 单元测试（gtest）：c10、ATen、api、jit
├── forward_backward_compatibility/    算子 Schema 的 BC/FC 检查（第七章 §3）
└── run_test.py                   统一入口：分片、超时、重跑、禁用列表
```

跑单个文件：`python test/test_ops.py -k test_noncontiguous_samples_add`；跑一类：`python test/run_test.py -i test_autograd`。C++ 测试在构建时 `BUILD_TEST=1` 生成到 `build/bin/`。

改动在本机通过了相关测试。但"对"不等于"快"——下一关。


## 四、第三关：没有变慢

正确性有明确的 oracle；性能没有——"变慢了"要先回答"比什么慢、慢多少算问题、噪声有多大"。第八篇建立了单次测量的纪律（同步、warmup、中位数、口径），这一章讨论把它变成**持续的、自动的**检查。

### 1. 微基准与指令数

`torch.utils.benchmark` 是第八篇用过的工具，`Compare` 把多组结果排成表：

```python
from torch.utils.benchmark import Timer, Compare

results = []
for n in [1 << 10, 1 << 16, 1 << 22]:
    for label, stmt in [("native", "2.0 * x + 1.0"), ("fused", "myops.scale_shift(x, 2.0, 1.0)")]:
        x = torch.randn(n, device="cuda")
        results.append(Timer(stmt, globals={"x": x, "myops": myops}, label="scale_shift", sub_label=f"n={n}",
                             description=label).blocked_autorange(min_run_time=1))
Compare(results).print()
```

```text
[----------- scale_shift -----------]
                 |  native  |  fused
1 threads: --------------------------
      n=1024     |   14.2   |   8.1
      n=65536    |   15.0   |   8.3
      n=4194304  |   61.7   |   33.9
Times are in microseconds (us).
```

GPU 时间有噪声（时钟、温度、其他进程），CPU 侧的框架开销更是如此。对 CPU 开销，`Timer.collect_callgrind()` 用 Valgrind 统计**指令数**而不是时间——完全确定，两次运行结果逐位相同，可以检测 0.1% 的变化。PyTorch 用它守护 Dispatcher 和 Python 绑定层的开销（`benchmarks/instruction_counts/`）：第五篇讨论的"每个算子在 CPU 上的固定成本"就是这样被盯住的。

`benchmarks/operator_benchmark/` 是算子级的基准集合：每个算子按几种典型 shape 和 dtype 定义配置，输出可比较的 JSON。

### 2. 端到端：TorchBench 与编译器看板

微基准看不到的问题（算子组合、显存分配模式、编译器的融合决策）要靠整模型。**TorchBench**（`pytorch/benchmark` 仓库）收集了上百个真实模型的可运行版本，统一接口；`benchmarks/dynamo/` 下的脚本在它之上加上 HuggingFace 和 TIMM 两个套件，构成 `torch.compile` 的三大基准集：

```bash
python benchmarks/dynamo/torchbench.py --performance --training --amp --backend=inductor --only=resnet50
python benchmarks/dynamo/huggingface.py --accuracy --inference --bfloat16 --backend=inductor
```

`--performance` 报告相对 eager 的加速比、编译时间、显存压缩比；`--accuracy` 用放宽容差的 `torch._dynamo.utils.same` 比对结果——这是编译器的正确性测试，与第三章 §6 互补：OpInfo 测单算子，这里测整模型。

结果每晚在固定的 A100 / H100 机器上跑，汇总到 HUD（hud.pytorch.org）的 TorchInductor Performance Dashboard：几何平均加速比、通过率、编译时间的逐日曲线。**回归的定义是曲线上的拐点**：某天加速比从 1.8× 掉到 1.6×，在那天合入的 PR 里 bisect。

### 3. 性能守门的三个难题

**噪声**。GPU 基准的运行间方差常有 2–5%，比很多真实回归还大。手段：锁定 GPU 时钟（`nvidia-smi -lgc`）、固定机器、多次运行取中位数、A/B 交替运行而不是先跑完 A 再跑 B；对 CPU 侧用指令数替代时间。

**阈值**。每个 PR 都跑全套基准太贵（几小时 GPU 时间），所以分层：PR 上只跑少量 smoke 模型、宽阈值；主干每晚跑全套；回归确认后人工 bisect。阈值定得太严，噪声触发误报，大家开始忽略；定得太松，回归积累到看板上才发现，此时已经有几十个 PR 需要排查。

**归因**。性能回归常常不是某个 PR 直接导致的：一个改动让某个融合决策变了，另一个改动让某个 Kernel 的 occupancy 掉了。nightly 之间 bisect 到 PR 后，还要用第八篇的 Profiler 定位到具体 Kernel。这是"性能测试比正确性测试更依赖人"的原因，也是为什么性能守门大部分放在合入**之后**（第五章 §1 的分层）。


## 五、第四关：审查、CI 与合入

前两关是开发者自己做的验证。合入主干意味着改动开始影响所有人，所以验证必须变成**强制的、自动的、有人负责的**。

### 1. CI 分层与目标确定

第一章 §3 说过，每个 PR 跑全部几十万测试不可能。PyTorch 的 CI 分成几层：

```text
pull        每个 PR 触发。几个 Linux 构建 + 分片后的测试子集，约两小时。目标：拦住大多数错误
trunk       合入后（或 PR 上加 ciflow/trunk 标签）触发。更多平台（macOS、Windows）、多 GPU、更慢的测试
periodic    每晚。慢测试、ROCm、多机分布式、Debug 构建
inductor    编译器的正确性与性能基准（第四章 §2），部分每 PR、部分每晚
```

在 `pull` 层内部，**目标确定**（Target Determination）按改动的文件排序测试：改了 `aten/native/cuda/Reduce.cu` 就先跑 reduction 相关的测试文件，历史上被这个文件的改动弄红过的测试排在前面。测试文件再被**分片**到多台机器并行。这是在"不能全跑"的前提下把漏网概率压到可接受的折中。

### 2. flaky 的流程化处理

第三章 §5 说过，几十万个测试实例中总有偶发失败。处理它的机制必须是自动的，否则**flaky 测试的成本不是那一个测试，而是它让所有人开始忽略红色的 CI**：

```text
CI 检测到某测试在主干上偶发失败
   → 机器人自动开一个 "DISABLED test_xxx (__main__.TestFoo)" issue
   → 测试运行器启动时拉取这个列表，跳过其中的测试；PR 不再因它变红
   → 定期任务 rerun-disabled-tests 重跑被禁用的测试
   → 连续通过一段时间后自动关 issue，测试恢复
```

被禁用的测试列表是公开的，一个模块下禁用测试堆积，是该模块维护者的待办。

### 3. 审批、合入与回滚

**审批规则**在 `.github/merge_rules.yaml` 中按路径定义：改 `aten/src/ATen/native/cuda/` 需要哪些人批准，改 `torch/distributed/` 需要哪些人，改 `native_functions.yaml`（第七章 §3 的契约）有额外的审批要求。这是把"谁对哪块代码负责"写成机器可执行的规则。

**合入**由机器人执行：审批通过后在 PR 下评论 `@pytorchbot merge`，机器人等待所需的 CI 通过后合入；`-f "原因"` 可以在紧急情况下跳过部分 CI，但会被记录。人不直接点合并按钮，是为了让"CI 必须绿"成为不可绕过的规则。

**回滚**同样机器化：主干红了，任何维护者评论 `@pytorchbot revert -m "原因" -c nosignal`，先恢复主干再讨论——主干保持绿色的优先级高于任何单个改动。

**lint** 统一由 `lintrunner` 运行：ruff、clang-format、clang-tidy、mypy 以及几十个自定义规则（比如禁止在某些目录 `#include` 某些头文件），本地 `lintrunner -a` 自动修复，CI 用同一配置——本地过了 CI 就不会因格式变红。

改动合入了。它现在在主干上，每晚随 nightly 构建。但到用户手里还要等下一个版本。


## 六、第五关：发布

### 1. 节奏与分支

小版本大约每三到四个月一个（2.0 于 2023 年 3 月，之后 2.1、2.2 …），每个版本有若干补丁版本。流程：

```text
主干持续合入 → 每天构建 nightly（pip install --pre torch --index-url .../nightly/cu124）
   ↓ 距发布约 6 周
cut 出 release/2.x 分支 → 发布候选 RC1、RC2 …
   ↓ 分支只接受 cherry-pick：在 "release tracker" issue 里申请，限于回归修复、关键 bug、文档
正式发布 2.x.0
   ↓
补丁 2.x.1、2.x.2：同样走 cherry-pick 流程
```

一个改动合入主干后，落在哪个版本取决于它相对 cut 日期的位置。想让用户尽早用到、或想让用户尽早暴露问题，就是 nightly 存在的理由——第八章会讲使用者怎样利用它。

### 2. 制品：wheel 矩阵与 ABI

用户装的是预编译 wheel。每次发布构建的矩阵：

```text
Python 版本    × 3.9 ~ 3.13
加速后端       × CPU / CUDA 11.8 / CUDA 12.x（通常同时支持两三个）/ ROCm / XPU
平台           × Linux x86_64 / Linux aarch64 / Windows / macOS arm64
```

每个组合一个 wheel，`torch==2.x.y+cu124` 的 `+cu124` 是本地版本标识（Python 系列讨论过）。CUDA wheel 不再打包整个 CUDA Toolkit，而是依赖 `nvidia-*` 的 PyPI 包（cuBLAS、cuDNN、NCCL 各自是一个 wheel），`libtorch_cuda.so` 在加载时通过 rpath 找到它们。`libtorch` 压缩包提供纯 C++ 使用（CMake 的 `find_package(Torch)`）。

第六篇 ABI 一节列出的五个因素在这里有了全貌：wheel 的每个维度都是 ABI 的一部分，C++ 扩展必须与其中一个具体组合匹配。从 2.6 起 Linux wheel 统一使用 cxx11 ABI（此前长期是旧 ABI），这是一次典型的"制品层面的破坏性变更"，扩展作者需要重新编译——第七章 §4 的话题。

### 3. 平台支持窗口

每个版本支持的 CUDA、Python、操作系统和 GPU 架构是一个滑动窗口，在 cut 分支时决定：

```text
CUDA          通常同时支持两到三个版本（例如 11.8 + 12.4 + 12.6），新版本加入时最老的退出
Python        新 Python 发布后几个月加入；到 EOL 前后移除
GPU 架构      Maxwell、Pascal、Volta 等老架构陆续从默认 wheel 的 TORCH_CUDA_ARCH_LIST 中移除，需要时自行从源码构建（第二章 §3）
操作系统      macOS x86_64 wheel 在 2.3 后停止
```

具体窗口以每个版本的发布说明为准。对 AI-Infra 工程师的含义是：**集群的驱动版本、容器镜像的 CUDA 版本和 PyTorch 版本是一个需要一起规划的矩阵**，第八篇优化报告里"版本"一栏、第八章的兼容矩阵管理都是这个原因。

### 4. 治理

PyTorch 于 2022 年进入 Linux Foundation 下的 PyTorch Foundation。技术决策由核心维护者与各模块维护者（`docs/source/community/persons_of_interest.rst`）负责，第五章 §3 的 `merge_rules.yaml` 是这份名单的机器可读版本；大的设计变更先在 `pytorch/rfcs` 仓库提 RFC 讨论。对使用者的意义是：一个功能的走向可以在 RFC 和 issue 里提前看到，而不是等发布说明。

改动随 2.x.0 发布了。最后一关，也是最难的一关：用户升级之后，他们的东西还能用吗？


## 七、第六关：用户升级不坏

正确性和性能是"这一版对不对"，兼容性是"下一版还能不能用"。PyTorch 有多个需要兼容的接口面，稳定程度差别很大，**分级**是在组合爆炸下唯一可行的策略：

```text
接口面              稳定程度                              机制
Python API          stable 特性保证 BC；有明确的弃用流程     §2
算子 Schema         自动化 BC/FC 检查；序列化模型依赖它       §3
C++ API             无 BC 保证；每个小版本都可能破坏扩展      §4（稳定 ABI 子集、PrivateUse1 是缓解）
序列化格式          torch.save / state_dict 有版本机制        §5
平台                滑动窗口，定期移除                        第六章 §3
```

### 1. 特性阶段

新特性按成熟度分三级，发布说明中标注：

```text
prototype    可能随时改变或删除，通常不在默认构建中或需要显式 opt-in
beta         API 基本稳定，性能和覆盖度仍在完善；可能有不兼容改动但会尽量避免
stable       保证向后兼容；改动走弃用流程
```

`torch.compile` 在 2.0 是 beta，DTensor 和 FSDP2 经历了 prototype → beta，`torch.distributed.pipelining` 在 2.4 以 prototype 进入。使用者读发布说明时，这个标签决定了"能不能在生产里用"。

### 2. Python API：弃用周期

stable API 的弃用流程：先在一个版本中发出警告（`FutureWarning` 或 `DeprecationWarning`，说明替代方案和删除时间），保留至少一个小版本（通常两个），然后删除。例如 `torch.symeig` 在 1.9 弃用、1.13 移除；`torch.load` 的 `weights_only` 默认值从 False 改为 True，在 2.4 开始警告、2.6 切换。**用户看到 `FutureWarning` 就应当改代码——它不是噪音，是倒计时**（第八章 §2 会把它变成 CI 规则）。

### 3. 算子 Schema：BC 与 FC

第五篇说 Schema 是算子的契约。它被序列化进 TorchScript 模型和导出的图，所以改 Schema 会影响已保存的模型。两个方向：

```text
BC（向后兼容）   新版本 PyTorch 能加载旧版本保存的模型      → 旧 Schema 的调用在新版本上仍能解析
FC（向前兼容）   旧版本 PyTorch 能加载新版本保存的模型      → 新 Schema 的调用在旧版本上仍能解析
```

允许与不允许的改动：

| 改动 | BC | FC | 说明 |
|---|---|---|---|
| 新增算子 / 新增 overload | ✓ | ✗ | 旧版本不认识新算子 |
| 在末尾新增带默认值的参数 | ✓ | ✗ | 旧模型不传它，用默认值；新模型传了它，旧版本解析失败 |
| 删除参数、重命名、改类型、改默认值 | ✗ | ✗ | 直接禁止；要变化就新增 overload 并弃用旧的 |
| 修改实现（不改 Schema） | ✓ | ✓ | 但可能改变数值结果 |

CI 里的 `test/forward_backward_compatibility/check_forward_backward_compatibility.py` 把当前构建的所有 Schema 与最近一次 nightly 的 Schema 快照比较，任何 BC 破坏都会失败；确有必要的改动加进 `ALLOW_LIST`，附带过期日期——防止列表无限增长。这就是第五章 §3 说 `native_functions.yaml` 有额外审批要求的原因。

语义变更走 **upgrader**：1.6 把 `torch.div` 从整数截断改为真除法，旧模型里的 `div` 节点在加载时被 `torch/csrc/jit/operator_upgraders/` 中的规则替换成 `div(rounding_mode="trunc")`，行为不变。模型文件里记录了保存时的算子版本号，加载时按版本号决定要应用哪些 upgrader。

### 4. C++ API：无保证、稳定子集与后端扩展点

`at::Tensor`、`c10::TensorImpl`、`TORCH_LIBRARY` 这些 C++ 接口**没有 BC 承诺**。每个小版本都可能改内部结构，第六篇的扩展要针对具体 PyTorch 版本重新编译。这是有意为之的：C++ 层是实现，锁定它会让框架无法演进。

两个缓解措施：

**稳定 ABI 子集**。近期版本开始在 `torch/csrc/stable/` 提供一小组有 ABI 稳定承诺的接口（`STABLE_TORCH_LIBRARY`、`torch::stable::Tensor`），底层通过 C 接口（`torch/csrc/inductor/aoti_torch/c/shim.h`，AOTInductor 也用它）与 libtorch 交互，隔开了 C++ 类布局。用它写的扩展一次编译可以跨多个 PyTorch 版本加载。覆盖的功能还有限，但方向明确。

**PrivateUse1**。第五篇的 DispatchKey 列表里预留了 `PrivateUse1` 等几个 Key 给树外后端。一个新硬件厂商不需要把代码合进 PyTorch 主干，而是在自己的包里向 `PrivateUse1` 注册全部算子实现、用 `torch.utils.rename_privateuse1_backend("npu")` 起名、注册设备模块和 Allocator，用户就能写 `x.to("npu")`。Ascend NPU（`torch_npu`）等就是这样接入的。它把硬件后端的发布周期与 PyTorch 解耦，代价是后端要自己跟上每个版本的 C++ API 变化——而 OpInfo（第三章 §2）给了它们衡量算子覆盖度的标尺。

### 5. 序列化：`torch.save` 与 `state_dict` 的版本

`torch.save` 的文件是一个 zip 包：

```text
model.pt（zip）
├── data.pkl        pickle 序列化的对象图；Tensor 被替换为 (storage 类型, key, device, numel) 的引用
├── data/0          storage 0 的原始字节
├── data/1
├── version         格式版本号
└── byteorder
```

pickle 带来的问题是安全：反序列化可以执行任意代码。`torch.load(weights_only=True)` 使用受限的 unpickler，只允许 Tensor、基本容器和显式加入白名单（`torch.serialization.add_safe_globals`）的类型；2.6 起它是默认值。这是 §2 弃用流程的一个实例，也是为什么很多旧代码在 2.6 上报 `UnpicklingError`。`mmap=True` 让大 checkpoint 按需读取而不是一次载入内存。

**`state_dict` 的版本**。`nn.Module` 子类有一个类属性 `_version`，保存进 `state_dict` 的 metadata；加载时 `_load_from_state_dict` 拿到旧版本号，可以做升级：

```python
class _BatchNorm(nn.Module):
    _version = 2
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get("version", None)
        if (version is None or version < 2) and self.track_running_stats:
            # 版本 2 新增了 num_batches_tracked；旧 checkpoint 没有，补一个默认值
            state_dict[prefix + "num_batches_tracked"] = torch.tensor(0, dtype=torch.long)
        super()._load_from_state_dict(...)
```

这就是 BatchNorm 在 1.0 之前加字段而旧 checkpoint 仍能加载的机制。自己写的 Module 加了 buffer 或改了参数名，应该走同样的路，而不是让用户手动改 checkpoint。第九篇的 DCP 在此之上加了分片与 reshard；生态中的 `safetensors` 用纯 Tensor 字典 + JSON 头绕开了 pickle。

到这里，改动走完了维护者一侧的六关。换到另一侧。


## 八、第七关：使用者如何跟随演进

前六关是 PyTorch 团队的事。但兼容性保证只是"框架不主动破坏你"，不等于"你什么都不用做"。AI-Infra 工程师维护的集群、镜像、C++ 扩展和 checkpoint，需要自己的一套跟随策略。

### 1. 版本策略：pin 什么、跟多紧

```text
生产训练 / 推理      pin 到具体版本（torch==2.6.0+cu124），连同 CUDA、驱动、扩展一起进镜像；只在计划内升级
开发与实验           跟最近的稳定小版本；每个小版本发布后一到两周内评估
关注中的新特性       用 nightly 或 RC 单独起环境试跑，不进主开发环境
```

跟得太紧，每次小版本都在处理 FutureWarning 和扩展重编译；跟得太松，一次跨两三个版本的升级会同时撞上所有弃用，且性能改进（第四章看板上那些曲线）拿不到。多数团队的节奏是**每个小版本评估、每两个小版本升级一次**。

### 2. 把弃用警告变成 CI 错误

第七章 §2 说 `FutureWarning` 是倒计时。让它在自己的 CI 里变成错误，就不会在升级那天集中爆发：

```bash
python -W error::FutureWarning -W error::DeprecationWarning -m pytest tests/
```

配合 pytest 的 `filterwarnings` 可以精确到具体消息，对暂时无法处理的加白名单并注明版本。这样每个小版本的弃用都在它发出警告的那个版本被处理掉，而不是在删除的版本才发现。

### 3. 升级 playbook

一次升级按风险从低到高分层验证，每层都有明确的通过标准：

```text
1. 读发布说明          先看 "Backwards Incompatible Changes" 和 "Deprecations"，再看与自己相关的模块；核对第六章 §3 的平台窗口
2. 重编译扩展          所有 C++ / CUDA 扩展针对新版本重编译（第七章 §4：C++ 无 BC 保证）；扩展自己的测试全过
3. CPU 单元测试        自己项目的测试套件在新版本上跑一遍；FutureWarning 当错误
4. 单卡功能测试        小模型跑几十步，loss 曲线与旧版本在容差内一致（第八篇的正确性测试）
5. checkpoint 兼容     用旧版本保存的 checkpoint 在新版本加载并继续训练；注意 weights_only 之类的默认值变化（第七章 §5）
6. 多卡与 compile      DDP / FSDP 与 torch.compile 各跑一轮；编译时间与 graph break 数量对比
7. 性能基线            与旧版本在同一硬件上对比 Benchmark（第八篇的口径）；变慢的项归因后再决定是否接受
8. 灰度                先升级一部分作业或一个集群分区，观察一段时间再全量
```

每一层失败都有回退路径：镜像是版本化的，回退就是换回旧 tag。所以**镜像必须包含完整的兼容矩阵**（下一节），而不是在启动时 `pip install`。

### 4. 兼容矩阵管理

一个训练环境至少有五个相互约束的版本：

```text
NVIDIA 驱动         ≥ CUDA 运行时要求的最低版本；集群级，升级最慢
CUDA 运行时          由 torch wheel 的 +cuXXX 决定（第六章 §2）
PyTorch              小版本
C++ 扩展             针对具体 torch × CUDA 编译（FlashAttention、自定义算子、Apex ...）
上层框架             Lightning、DeepSpeed、Megatron、vLLM 等各自声明的 torch 版本范围
```

管理方式是把这个矩阵**写下来并版本化**：一个镜像对应一组确定的版本，镜像 tag 就是矩阵的名字。任何一项的升级都产生新的镜像 tag，并走 §3 的 playbook。驱动是最慢的一项，往往决定了 CUDA 版本的上限，从而决定了能用的 PyTorch 版本——规划升级时从它开始倒推。

### 5. 提前暴露问题：nightly 与 RC

第六章 §1 说 nightly 存在的理由之一是让用户提前暴露问题。反过来，使用者也应利用它：对依赖重的项目，在 RC 阶段用 RC 跑一遍自己的测试和一个短训练，发现问题在 release tracker issue 里报告——此时修复还能进正式版本；等正式发布后再发现，就要等补丁版本或自己 patch。这是使用者参与前六关的方式，成本很低，收益是升级日不再有惊喜。


## 九、实践终点：把第六篇的 `myops` 走完这七关

第六篇做出了 `myops::scale_shift`：Schema、CPU、CUDA、Autograd、Meta 都有，`opcheck` 和 `gradcheck` 通过。那是"一个算子能用"。这一章让它也走一遍前面七关，变成"一个能持续维护的工程"。给骨架和关键片段，重点是**为什么这样组织**。

### 1. 目录结构

```text
myops/
├── pyproject.toml / setup.py       第一关：cpp_extension.CUDAExtension；把构建时的 torch 版本写进包元数据
├── myops/
│   ├── __init__.py                 import _C（触发 TORCH_LIBRARY）；注册 autograd / fake；运行时版本检查（第六篇十.4）
│   ├── _autograd.py                backward、setup_context、fake
│   └── ops.py                      面向用户的 Python 函数；弃用垫片放这里（第六关）
├── csrc/
│   ├── scale_shift.cpp             TORCH_LIBRARY 定义 + CPU 实现
│   └── scale_shift_cuda.cu
├── test/                           第二关
│   ├── test_ops.py                 设备泛化 × dtype 矩阵 × layout × 参考实现 × 错误输入
│   ├── test_autograd.py            gradcheck / gradgradcheck / forward AD
│   ├── test_compile.py             eager vs compile；Fake 与真实 shape 一致
│   ├── test_distributed.py         DDP 下与单进程大 batch 一致
│   └── test_schema.py              Schema 快照，防止无意的契约变更（第六关）
├── benchmarks/                     第三关
│   ├── bench_scale_shift.py        Compare 表 + JSON 输出
│   └── baseline.json               基线；CI 中比较
└── .github/workflows/ci.yml        第四、五关：lint → build → test → bench 的矩阵；tag 触发构建 wheel
```

三个分离：**实现（csrc/）、契约（Schema + Python 垫片）、验证（test/ + benchmarks/）**各自独立演进；测试按 oracle 分文件（第三章 §1 的五种 oracle 各有归属）；性能基线入库，让"变慢了"有可比对象。

### 2. 第二关：复用 PyTorch 的测试基础设施

`torch.testing._internal` 虽然名字带 internal，却是扩展项目最值得复用的部分——设备泛化、dtype 参数化、容差比较都不用自己写：

```python
# test/test_ops.py
import torch, myops
from torch.testing import assert_close, make_tensor
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_device_type import instantiate_device_type_tests, dtypes

SHAPES = [(), (0,), (1,), (7,), (3, 5), (2, 3, 4, 5)]

def ref_scale_shift(x, alpha, beta):                              # oracle 1：参考实现，在 fp64 上算再 cast 回来
    return (x.double() * alpha + beta).to(x.dtype)

class TestScaleShift(TestCase):
    @dtypes(torch.float32, torch.float64, torch.float16, torch.bfloat16)
    def test_reference(self, device, dtype):
        for shape in SHAPES:
            for noncontig in (False, True):
                x = make_tensor(shape, device=device, dtype=dtype, noncontiguous=noncontig)
                assert_close(myops.scale_shift(x, 2.0, 1.0), ref_scale_shift(x, 2.0, 1.0))     # 默认按 dtype 容差

    @dtypes(torch.float32)
    def test_cpu_cuda_consistency(self, device, dtype):           # oracle 3：跨后端一致
        if device == "cpu": self.skipTest("需要 CUDA 对照")
        x = make_tensor((3, 5), device="cpu", dtype=dtype)
        assert_close(myops.scale_shift(x.to(device), 2.0, 1.0).cpu(), myops.scale_shift(x, 2.0, 1.0))

    @dtypes(torch.float32)
    def test_error_inputs(self, device, dtype):                   # 非法输入必须是清晰的错误，不是崩溃
        with self.assertRaisesRegex(RuntimeError, "floating point"):
            myops.scale_shift(torch.ones(3, device=device, dtype=torch.int32), 2.0, 1.0)

    @dtypes(torch.float32, torch.float64)
    def test_opcheck(self, device, dtype):                        # oracle 5：Schema / Fake / Autograd 注册一致
        for shape in SHAPES:
            x = make_tensor(shape, device=device, dtype=dtype, requires_grad=True)
            torch.library.opcheck(torch.ops.myops.scale_shift, (x, 2.0, 1.0))

instantiate_device_type_tests(TestScaleShift, globals())
if __name__ == "__main__":
    run_tests()
```

`SHAPES` 覆盖了大纲检查清单里的标量、空 Tensor、一维、高维；`noncontiguous=True` 覆盖 layout；`@dtypes` 覆盖 dtype；`instantiate_device_type_tests` 覆盖设备。极端数值（inf / nan / 极大 alpha）加一个 `test_extremal` 用 `torch.tensor([float("inf"), float("nan"), 1e38])` 与参考实现比对（`equal_nan=True`）。

Autograd 测试独立一个文件，因为它的输入要求（fp64、`requires_grad`）不同：

```python
# test/test_autograd.py
from torch.autograd import gradcheck, gradgradcheck

class TestScaleShiftGrad(TestCase):
    @dtypes(torch.float64)
    def test_grad(self, device, dtype):
        x = make_tensor((3, 5), device=device, dtype=dtype, requires_grad=True)
        fn = lambda t: torch.ops.myops.scale_shift(t, 2.0, 1.0)
        self.assertTrue(gradcheck(fn, (x,), check_forward_ad=False, fast_mode=True))
        self.assertTrue(gradgradcheck(fn, (x,)))                  # 线性算子上平凡通过；换成非线性算子时它才是真正的守卫
```

编译器一致性（oracle 4）：

```python
# test/test_compile.py
def test_compile_matches_eager(self, device, dtype):
    x = make_tensor((64, 128), device=device, dtype=dtype, requires_grad=True)
    fn = lambda t: torch.ops.myops.scale_shift(t, 2.0, 1.0).sin().sum()      # 让它处在一个会被融合的上下文里
    compiled = torch.compile(fn, fullgraph=True)                              # fullgraph：没有 Fake 实现会在这里直接失败
    assert_close(compiled(x), fn(x))
    (g_c,) = torch.autograd.grad(compiled(x), x); (g_e,) = torch.autograd.grad(fn(x), x)
    assert_close(g_c, g_e)
```

`fullgraph=True` 是关键：第六篇注册的 Fake 实现如果缺失或 shape 推错，Dynamo 会 graph break 或报错，测试立刻失败，而不是静默退化成 eager。

分布式测试用第三章 §6 的 `MultiProcessTestCase`，oracle 是"DDP 两卡各 batch/2 的梯度 == 单进程整 batch 的梯度"（第九篇 §三.1.1 的数学等价）：

```python
# test/test_distributed.py
from torch.testing._internal.common_distributed import MultiProcessTestCase, skip_if_lt_x_gpu

class TestScaleShiftDDP(MultiProcessTestCase):
    @property
    def world_size(self): return 2
    def setUp(self): super().setUp(); self._spawn_processes()

    @skip_if_lt_x_gpu(2)
    def test_ddp_grad_matches_single(self):
        dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size, init_method=f"file://{self.file_name}")
        torch.manual_seed(0); x = torch.randn(8, 16, device=f"cuda:{self.rank}")   # 所有 rank 同一份全量数据
        model = Model().cuda(self.rank)                                              # 内部调用 myops.scale_shift
        ddp = DDP(copy.deepcopy(model), device_ids=[self.rank])
        ddp(x.chunk(2)[self.rank]).sum().backward()                                 # 每个 rank 一半
        model(x).sum().backward()                                                   # 单进程全量
        for p_d, p_s in zip(ddp.parameters(), model.parameters()):
            assert_close(p_d.grad, p_s.grad / 2)                                    # DDP 求的是平均
        dist.destroy_process_group()
```

### 3. 第三关：Benchmark 与回归阈值

```python
# benchmarks/bench_scale_shift.py
results, record = [], {}
for n in [1 << 10, 1 << 16, 1 << 22]:
    x = torch.randn(n, device="cuda")
    for label, stmt in [("native", "2.0 * x + 1.0"), ("fused", "myops.scale_shift(x, 2.0, 1.0)")]:
        m = Timer(stmt, globals={"x": x, "myops": myops}, label="scale_shift", sub_label=f"n={n}", description=label).blocked_autorange(min_run_time=1)
        results.append(m); record[f"{label}/n={n}"] = m.median
Compare(results).print()
json.dump({"torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "median_s": record}, open(sys.argv[1], "w"))
```

CI 中与 `baseline.json` 比较：同一 GPU 型号下任一项中位数慢 15% 以上则失败。15% 是给 GPU 噪声留的余量（第四章 §3）；基线随硬件与 PyTorch 版本入库，更新基线是一次显式的、需要审查的提交。报告按第八篇的五个问题写：这个融合算子节省的是访存和一次 launch，不增加显存，不改变精度，对 shape 无依赖，无编译成本。

### 4. 第四、五关：CI 矩阵与发布

```yaml
# .github/workflows/ci.yml（节选）
jobs:
  lint:  { runs-on: ubuntu-latest, steps: [ruff, clang-format --dry-run] }
  test:
    needs: lint
    strategy:
      matrix:
        torch: ["2.5.*", "2.6.*"]           # 支持窗口内的版本；与第六章 §3 的平台窗口对齐
        cuda:  ["12.4"]
        python: ["3.10", "3.12"]
    runs-on: [self-hosted, gpu]
    container: pytorch/pytorch:${{ matrix.torch }}-cuda${{ matrix.cuda }}-cudnn9-devel
    steps:
      - run: pip install -e . --no-build-isolation
      - run: python -W error::FutureWarning -m pytest test/ -x -q        # 第八章 §2：弃用警告即错误
      - run: python benchmarks/bench_scale_shift.py out.json && python benchmarks/compare.py baseline.json out.json --tol 0.15
  wheel:
    if: startsWith(github.ref, 'refs/tags/v')                              # 打 tag 才构建制品
    strategy: { matrix: { torch: ["2.5.*", "2.6.*"], python: ["3.10", "3.12"] } }
    steps: [build wheel, 上传到内部 index；文件名带 +torch2.6cu124 本地版本标识]
```

矩阵的每个格子对应第六章 §2 wheel 矩阵的一个格子——扩展与 PyTorch 的 ABI 绑定在这里变成了显式的构建配置。矩阵不能无限大：选支持窗口内的两个 PyTorch 版本、一个 CUDA、两个 Python，其余组合靠 ABI 规则推断。发布用 tag 触发，制品名里带上它针对的 torch 版本，这是第八章 §4 兼容矩阵的"可写下来"的前提。

### 5. 第六关：契约的守卫与一次向后兼容的演进

**Schema 快照**：

```python
# test/test_schema.py
EXPECTED = "myops::scale_shift(Tensor x, float alpha, float beta) -> Tensor"

def test_schema_unchanged():
    assert str(torch.ops.myops.scale_shift.default._schema) == EXPECTED
```

这是第七章 §3 那个 BC/FC 检查的最小版本。改 Schema 时这个测试必然失败，迫使作者更新 `EXPECTED`——在代码审查里，这一行 diff 就是"我改了契约"的显式信号。

**一次演进**：需求是给 `scale_shift` 加一个可选的 `mask`。按第七章 §3 的表，**末尾新增带默认值的参数**是 BC 的：

```text
myops::scale_shift(Tensor x, float alpha, float beta, Tensor? mask=None) -> Tensor
```

旧调用 `scale_shift(x, 2.0, 1.0)` 不受影响；CPU、CUDA、Autograd（mask 不需要梯度，但反向要乘 mask）、Fake 四处实现同步更新；`test_schema.py` 的 `EXPECTED` 更新；测试加 `mask` 的样例（None、全 True、随机、广播形状）。FC 被破坏——用新版本导出的图在旧版本上加载会失败——如果有人依赖导出，就要在发布说明里写明。

如果需求是改语义（比如 `beta` 改成先加再乘），不能改原 Schema：新增 `scale_shift.v2` 或新算子，旧的在 `ops.py` 里加 `FutureWarning` 指向新的，两个版本后删除——第七章 §2 的流程原样照搬。

### 6. 第七关：作为使用者

`myops` 依赖 PyTorch，所以它也要走第八章：`setup.py` 里声明 `torch>=2.5,<2.7`，每个 PyTorch 小版本发布后把 CI 矩阵加一列、跑一遍、必要时重编译并发新 tag；`__init__.py` 里的运行时版本检查（第六篇十.4）把 ABI 不匹配变成明确的 `ImportError`。

### 7. 对照大纲的检查清单

```text
正确性                                          工程质量
├── 多种 shape          SHAPES × make_tensor      ├── 单元测试        test/ 四个文件，设备泛化
├── 多种 dtype          @dtypes                   ├── gradcheck       test_autograd.py，fp64，含 gradgradcheck
├── CPU/CUDA            instantiate_device_type   ├── Benchmark       Compare + baseline.json
├── contiguous/non-     noncontiguous=True        ├── 文档            Schema 即文档；ops.py 的 docstring
├── 空 Tensor / 标量    SHAPES 里的 (0,) 和 ()    ├── CI              矩阵 × lint → build → test → bench → wheel
├── 广播                mask 的广播样例           └── 性能回归        15% 阈值，基线入库
├── requires_grad       opcheck + gradcheck
├── 极端数值 / NaN      test_extremal
└── 非法输入            test_error_inputs
```

大纲要求的"支持 CPU、CUDA、Autograd、Meta，并具有完整测试和 Benchmark 的自定义算子"到这里完成。它把前九篇串成了一条线：Schema（第五篇）→ stride 与 dtype 处理（第二、六篇）→ 反向（第三篇）→ Fake 与 compile（第七篇）→ Benchmark（第八篇）→ DDP 一致性（第九篇）→ 七个关卡（本篇）。


## 十、Java 工程师如何理解 PyTorch 的工程体系

### 1. 生命周期：PR → CI → release train

七个关卡对 Java 工程师并不陌生：本地 `mvn verify` → PR check → merge queue → release train → 用户升级依赖版本。差别在每一关的**规模和绑定**：测试是几十万个而不是几千个；制品是绑定平台 × Python × CUDA 的 wheel 矩阵而不是一个平台无关的 jar；用户升级要重编译 C++ 扩展而不是只改 `pom.xml` 里的版本号。

### 2. OpInfo vs 参数化测试 vs TCK

JUnit 5 的 `@ParameterizedTest` + `@MethodSource` 与 OpInfo 的 `@ops` + `sample_inputs_func` 结构相同：数据源与测试模板分离。差别在规模和角色——OpInfo 的数据源是**整个算子集合的能力声明**，模板有几十个，两者的笛卡尔积是框架的一致性套件。这更像 Java 的 **TCK**（Technology Compatibility Kit）：JDBC、JPA、Servlet 的实现要通过 TCK 才能自称兼容；新 PyTorch 后端要在 OpInfo 上跑出覆盖率才算可用。

`gradcheck` 是基于属性的测试（property-based testing，Java 的 jqwik / QuickTheories）：不检查具体值，检查"解析导数 = 数值导数"这个必然成立的性质。

### 3. Benchmark：JMH 与指令数

`torch.utils.benchmark.Timer` 对应 JMH：warmup、多轮、统计分布、防止死代码消除（JMH 的 Blackhole ≈ 这里必须使用输出）。`collect_callgrind` 的指令数没有 JMH 对应物——Java 的 JIT 让指令数不稳定；PyTorch 的 C++ 分发路径是确定的，所以能用它做零噪声的回归检测。

### 4. 构建：Maven / Gradle vs CMake + setup.py

```text
pom.xml / build.gradle           CMakeLists.txt + setup.py
Maven Central 上的 jar           PyPI 上的 wheel；但 wheel 绑定平台 × Python × CUDA，jar 不绑定
一次编译到处运行                  一个 wheel 只在一个格子里运行；ABI 是显式的兼容性维度
annotation processor 生成代码     torchgen 从 YAML 生成 C++ 和 Python 绑定
增量编译（Gradle daemon）          Ninja 增量 + ccache；改头文件的代价远大于改 .java
```

最大的心智差异是 **ABI**：Java 工程师习惯了字节码的平台无关性，C++ 扩展世界里编译器版本、标准库 ABI、CUDA 版本、GPU 架构都是二进制兼容性的一部分。

### 5. 兼容性：`@Deprecated`、`serialVersionUID` 与 class file version

```text
@Deprecated(since, forRemoval)          FutureWarning + 两个小版本的窗口
-Werror 对 deprecation                  -W error::FutureWarning（第八章 §2）
serialVersionUID + readObject 迁移       nn.Module._version + _load_from_state_dict
class file major version + 向后兼容      TorchScript 算子版本号 + upgrader
Java 序列化的安全问题 → 过滤器          pickle 的安全问题 → weights_only=True
JEP                                     RFC
六个月一个 JDK，LTS 每两年               三到四个月一个小版本，无 LTS
JCP / OpenJDK 治理                       PyTorch Foundation + 模块维护者
```

Java 的 `readObject` 里按 `serialVersionUID` 迁移旧字段，与 `_load_from_state_dict` 里按 `version` 补 `num_batches_tracked` 是同一个模式。"无 LTS"是一个实际差别：Java 团队可以停在 LTS 上几年，PyTorch 使用者没有这个选项，第八章的跟随策略因此是必需的而不是可选的。

### 6. CI：分层与 flaky

`pull` / `trunk` / `periodic` 的分层对应 Java 项目的 PR check / merge queue / nightly；目标确定对应 Gradle 的按改动选择测试。flaky 测试的机器人自动禁用 + 定期重跑 + 自动恢复，是 Java 团队通常靠人做的事情被流程化——规模逼出来的。


## 十一、系列总结：从 `loss.backward()` 一路追问到底

总览篇给了一段代码和一串追问，说读完系列应该能回答。现在逐一回答，每个答案标出它来自哪一篇。

```python
output = model(inputs)
loss = criterion(output, targets)
loss.backward()
```

**Tensor 如何表示输入？**（第二篇）
`inputs` 是一个 `Tensor`，本质是 TensorImpl 元数据（shape、stride、offset、dtype、device）指向一块 Storage。视图操作只改元数据不拷贝数据；stride 决定内存访问模式，进而决定 Kernel 的访存效率（第八篇 memory-bound 的根源之一）。

**Module 如何组织模型？**（第四篇）
`model` 是 `nn.Module` 树，参数和 buffer 按名字注册，`state_dict` 是它的可序列化投影（本篇 §七.5 的 `_version` 就挂在上面）。`model(inputs)` 经过 `__call__` 的 hook 链进入 `forward`。混合精度、优化器、数据加载围绕这个树组织。

**Autograd 如何构建计算图？**（第三篇）
前向中每个算子在 Autograd Key 上被包装：记录一个 `grad_fn` 节点，保存反向需要的 Tensor，把节点接到输入的 `grad_fn` 上。`loss.backward()` 从 `loss.grad_fn` 出发按拓扑序执行反向节点，把梯度累积到叶子的 `.grad`——DDP 的 Reducer（第九篇）就挂在这个累积点上；`gradcheck`（本篇）用有限差分验证每个反向节点。

**Dispatcher 如何选择算子？**（第五篇）
`torch.add(a, b)` 经 Python 绑定进入 Operator Table 中 `aten::add` 的条目，由输入 Tensor 的 DispatchKeySet 决定先走 Autograd 包装再走后端实现。原生算子的注册代码由 Codegen 从 `native_functions.yaml` 生成（本篇 §二.2 讲了它何时运行）；自定义算子手写同样的三步（第六篇）。

**Kernel 在 CPU 或 GPU 上如何执行？**（第六篇）
后端实现用 TensorIterator 处理 stride、`AT_DISPATCH` 处理 dtype、CUDAGuard 处理设备，把 Kernel 提交到当前 Stream 后立即返回。CPU 和 GPU 是两条异步时间线（第八篇）；扩展与 PyTorch 之间是 ABI 边界（本篇 §六.2）。

**Compiler 如何对计算进行变换？**（第七篇）
`torch.compile` 用 Dynamo 从字节码捕获 FX Graph 并生成 Guard，AOTAutograd 把它变成 ATen 级的前向和反向图并做分解，Inductor 融合算子生成 Triton / C++。它的正确性 oracle 是 eager（本篇 §三.6），性能由三大基准套件每晚守护（本篇 §四.2）。

**Profiler 如何告诉我们瓶颈在哪里？**（第八篇）
时间维度上五类瓶颈（Python / Launch / Memory / Compute / Sync-bound），空间维度上显存的构成、碎片、峰值与泄漏；工具地图从 `benchmark.Timer` 到 Nsight Compute。所有优化先测再改，一次只改一件事——本篇第四章把这个纪律变成持续的守门。

**Distributed Runtime 如何让多卡协同？**（第九篇）
对五类状态各做复制或分片的决定，每个决定对应一种集合通信和一个时机。DDP 复制并 all_reduce 梯度，FSDP 分片并 all_gather / reduce_scatter，TP 切层内、PP 切层间、CP 切序列、EP 切 expert。通信是异步 Kernel，重叠要求重排依赖。

**Tests、Build 和 CI 如何保证系统可持续演进？**（本篇）
一次改动要过七关：本地构建、正确（五种 oracle、OpInfo 化解组合爆炸、gradcheck 守住反向）、不慢（微基准与整模型看板）、合入（CI 分层、机器化的审批与回滚）、发布（release 分支与 wheel 矩阵）、不坏用户（弃用周期、Schema BC/FC、`state_dict` 版本、PrivateUse1）、以及使用者自己的跟随策略。

---

总览篇说系列的目标是三种能力。现在可以具体地说它们指什么：

**阅读能力**——拿到 PyTorch 或任何训练 / 推理框架的源码，知道从 `torch/` 到 `torch/csrc/` 到 `native_functions.yaml` 到 `aten/native/` 到 `c10/` 的路径，知道一个算子的 Schema、注册、实现、反向、Meta 分别在哪里，知道 `torch.compile` 的三段在哪个目录。

**诊断能力**——面对一个错误、一次 OOM、一条平坦的 GPU 利用率曲线、一个 hang 住的分布式作业，知道先测什么、用哪个工具、在两条时间线或五类状态的哪个位置找原因。

**扩展能力**——需要一个新算子、一个新的融合、一种新的并行策略时，知道要写哪三步、要注册哪几个 Key、要过哪几种 oracle、要放进哪个 CI 矩阵，以及怎样让它在下一个 PyTorch 版本上还能用。

这三种能力的共同基础是一张地图：**从 Python 用户代码，经过 Autograd、Dispatcher、Kernel、编译器、运行时，到硬件和集群，每一层的职责、边界和代价**。十篇文章画的就是这张图。图画完了，剩下的是在真实系统里反复走它。


## 系列目录

- [总纲：从 Tensor 到深度学习运行时](/deep-dive-into-pytorch.html)

1. [PyTorch 整体介绍](/pytorch-overall-introduction.html)
2. [Tensor 与内存布局](/pytorch-tensor-and-memory-layout.html)
3. [自动求导与动态计算图](/pytorch-autograd-and-dynamic-computation-graph.html)
4. [`nn.Module` 与训练系统](/pytorch-module-and-training-system.html)
5. [Dispatcher 与算子系统](/pytorch-dispatcher-and-operator-system.html)
6. [C++ 扩展与自定义算子](/pytorch-cpp-extension-and-custom-operators.html)
7. [编译执行与图优化](/pytorch-compilation-and-graph-optimization.html)
8. [性能优化与调试](/pytorch-performance-optimization-and-debugging.html)
9. [分布式 PyTorch](/pytorch-distributed-training.html)
10. PyTorch 的工程体系：一次改动如何安全地到达用户（本文）
