---
layout: post
title: "AI-Infra 开源贡献指南（01）：读懂一个百万行的代码库"
subtitle: "Reading a Million-Line Codebase: Maps, Entry Points, Symbols, Tests and History"
tags: [Open Source, PyTorch, vLLM, AI, AI-Infra]
catalog: true
---

> 本文是《AI-Infra 开源贡献指南》系列的第 1 篇。上一篇：[总纲](/contributing-to-ai-infra-open-source.html)；下一篇：[找到切入点：从 issue、RFC 到性能回归](/finding-your-entry-point-in-open-source.html)。

一位工程师在 GitHub 上看到 PyTorch 的一个 issue：`torch.logaddexp` 在 complex128 上 CPU 与 CUDA 的结果不一致。他会写 CUDA，觉得这是个好的第一个 PR。克隆仓库，`rg logaddexp` 一下，两百多个匹配；打开 `torch/__init__.py` 想找 `def logaddexp`，没有；点 IDE 的"跳转到定义"，跳到一个 `.pyi` 文件里的类型签名就断了；再往下找 C++ 实现，发现 `aten/src/ATen/native/BinaryOps.cpp` 里只有一行宏 `CREATE_BINARY_TORCH_IMPL_FUNC(logaddexp_out, logaddexp_stub)`，而 clangd 对整个 `aten/` 目录报"找不到头文件 `ATen/ops/logaddexp_native.h`"——这个文件在仓库里确实不存在。两个小时过去，他还没有找到那个要改的 kernel 在哪个文件。

这不是能力问题。这位工程师读得懂 CUDA kernel，也修得了数值 bug，他缺的是一套**面对百万行代码库的阅读方法**，以及对这个项目"自己提供了哪些辅助"的了解：PyTorch 的 `CONTRIBUTING.md` 有一节 "Codebase structure" 把目录职责写清楚了；`native_functions.yaml` 是所有算子的登记表；报"找不到头文件"是因为那些头文件由 `torchgen` 在构建时生成，构建一次就都有了；而那个 kernel 就在 `aten/src/ATen/native/cuda/LogAddExpKernel.cu`，从 yaml 里的 `dispatch:` 字段两步就能追到。

同样的故事换到 vLLM 也成立：`vllm serve` 这条命令从哪里开始执行？`torch.ops._C.rms_norm` 的定义在哪——`rg 'def rms_norm'` 只能找到 Python 包装，真正的 kernel 在 `csrc/` 下一个 `.cu` 文件里，中间隔着一层 `STABLE_TORCH_LIBRARY_FRAGMENT` 注册。不知道这层机制，就会在 Python 侧兜圈子。

本文讲的就是这套方法：先画地图，再找入口，遇到生成代码知道去哪里找，构建一次让工具链能用，用测试当规格说明，用 git 历史当注释。方法与项目无关，辅助设施因项目而异——所以每一步都同时给出 PyTorch 与 vLLM 的真实路径。全文以 **PyTorch v2.14.0** 与 **vLLM v0.28.0** 的源码树为准，所有路径、章节名与命令都在这两个检出上核对过。

本篇的核心问题：

> **给你一个从未见过的百万行仓库和一个报错信息，两小时之内你能把它定位到一个文件的一个函数吗？靠什么？**


## 一、总览

### 1. 问题

面对一个大型代码库，贡献者典型的失败方式有四种，它们都源于"读法不对"而非"读不懂"：

```text
失败方式                 表现                                                    根因
───────────────────────  ──────────────────────────────────────────────────────  ──────────────────────────────
从头读                   打开仓库根目录，按字母序点开子目录，一天后只看完 c10/ 的一半    没有目标；百万行不可能通读
随便点开一个文件读        从 torch/__init__.py 往下追，被 import 链带进 20 个文件      没有先画地图，不知道哪层做什么
被生成代码卡住            "找不到定义"、"找不到头文件"，以为仓库不完整                  不知道哪些文件是构建时生成的
只读代码不读测试与历史     猜一个函数"应该"做什么；不知道为什么它是这样写的               忽略了测试是规格、commit 是注释
```

四种失败的共同点是把"读代码"当成一种线性活动。在一个人能通读的规模（几万行）里这没问题；超过这个规模，阅读必须是**有目标的检索**：从一个符号、一个报错、一个 issue 出发，用工具追到底，只读路径上的东西。

### 2. 方法

六个步骤，与项目无关：

```text
步骤          做什么                                          为什么
────────────  ──────────────────────────────────────────────  ────────────────────────────────────────────
1 画地图      列出顶层目录与一句话职责；找到项目自带的"地图"文件   知道每一层做什么，才知道一个符号该在哪一层找
2 找入口      从用户可见的 API 或命令出发，追到第一个"真正干活"的文件   入口点是所有追踪的起点；每个项目的入口结构固定
3 识别生成代码 知道哪些目录/文件是构建时生成的、由什么模板生成      避免在仓库里找不存在的文件；知道去改模板而不是产物
4 构建一次    得到 compile_commands.json、生成的头文件与 .pyi      clangd 与 IDE 跳转需要它；没有它 C++ 侧无法追踪
5 读测试      找到与目标符号相关的测试文件与测试框架的约定         测试是最准确的行为规格；也是改动后要跑的东西
6 读历史      git log -S / blame -w -C，顺着 PR 链接看当时的讨论   "为什么这样写"的答案在引入它的那次改动里
```

每一步的产出都写进贡献日志的"项目地图"页（第九章），下一次遇到同一项目的问题时直接从地图开始。

### 3. 两个项目

同一个步骤，两个项目提供的辅助不同：

```text
环节              PyTorch v2.14.0                                        vLLM v0.28.0
────────────────  ─────────────────────────────────────────────────────  ──────────────────────────────────────────────
自带地图          CONTRIBUTING.md 的 "Codebase structure" 一节；各子目录 README   docs/contributing/ 目录（README.md 等 16 个文件）；无目录职责表
语言分层          c10/ aten/ torch/csrc/（C++）→ torch/（Python）           csrc/（CUDA/C++）→ vllm/（Python）；Python 占绝大多数
算子登记表        aten/src/ATen/native/native_functions.yaml               csrc/libtorch_stable/torch_bindings.cpp 的 STABLE_TORCH_LIBRARY_FRAGMENT
Python 入口       torch/__init__.py 从 torch._C._VariableFunctions 导入      pyproject.toml 的 [project.scripts]：vllm = vllm.entrypoints.cli.main:main
生成代码          torchgen 生成 ATen/ops/*.h、python_torch_functions_N.cpp、.pyi   编译产物 _C_stable_libtorch.abi3.so 等；无源码级生成
构建命令          pip install -e . -v --no-build-isolation；tools/nightly.py    VLLM_USE_PRECOMPILED=1 uv pip install -e .；incremental_build.md
测试框架          unittest + torch/testing/_internal/（TestCase、OpInfo）     pytest；tests/ 按子系统分目录
commit 与 PR 的链接  正文含 Pull Request resolved: 与 Approved by:              标题带 (#PR 号)；正文多为 Signed-off-by 与 Co-authored-by
发布节奏          RELEASE.md：按日程切 release 分支，约每两个月一个 minor 版本    RELEASE.md：约每两周一版，v0.12.0 起常规发布递增 minor
```

### 4. 本文的章节安排

```text
第二章  先画地图            两张目录表（含在检出上统计的文件数与行数）；两个项目自带的地图文件
第三章  找到入口点          torch.logaddexp 从 Python 到 CUDA kernel 的追踪；vllm serve 从命令到引擎的追踪
第四章  生成代码            torchgen 生成什么、放在哪、为什么"找不到定义"；.pyi.in；vLLM 的 _C 扩展
第五章  构建一次            compile_commands.json 与 clangd；两个项目文档里的构建命令
第六章  用测试当文档        PyTorch test/ 的组织与 TestCase/OpInfo；vLLM tests/ 的组织
第七章  读历史              git log -S、blame -w -C；两个项目的 commit message 形态；RELEASE.md
第八章  核心问题            两小时定位流程清单
第九章  贡献日志            "项目地图"页的模板与两份填好的样例
第十章  本文小结            要点、对照表、文件位置表
```


## 二、先画地图

### 1. PyTorch 的目录

在 v2.14.0 检出上统计（`git ls-files` 与 `rg --files | wc -l`，行数为 `git ls-files <dir> | xargs cat | wc -l` 的粗略值，含数据文件，不含 `third_party/` 子模块）：仓库根目录 52 个条目，`git ls-files` 列出 21663 个文件、约 475 万行。八个与贡献者直接相关的目录：

| 目录 | 文件数 | 行数（约） | 一句话职责 |
|---|---|---|---|
| `c10/` | 433 | 8.9 万 | 最底层的核心库：`Device`、`ScalarType`、`TensorImpl`、`DispatchKey`、分配器；server 与 mobile 都用 |
| `aten/` | 2718 | 80.7 万 | C++ 张量库（无 autograd）。`aten/src/ATen/native/` 是算子实现；`aten/src/ATen/native/native_functions.yaml` 是全部算子的登记表；`cpu/`、`cuda/`、`mps/` 等子目录是各后端 kernel |
| `torch/csrc/` | 1924 | 49.6 万 | PyTorch 库的 C++ 部分：Python 绑定（文件名惯例以 `python_` 开头）、`autograd/`、`jit/`、`distributed/`、`inductor/` 的 C++ 侧 |
| `torch/` | 4607 | — | Python 包本体（除 `csrc/` 外）：`nn/`、`optim/`、`distributed/`、`_dynamo/`、`_inductor/`、`testing/` 等；`.py` 文件约 126.6 万行 |
| `torchgen/` | 94 | 2.7 万 | 代码生成器：读 `native_functions.yaml`，生成 C++ 与 Python 绑定；`gen.py` 是入口，`model.py` 的 `NativeFunction` 是 yaml 一条记录的内存表示 |
| `test/` | 11357 | 146 万 | Python 单元测试，176 个 `test_*.py` 加 46 个子目录（`distributed/`、`inductor/`、`dynamo/`、`cpp/` 等） |
| `tools/` | 388 | — | 构建与开发脚本：`nightly.py`、`autograd/`（Python 绑定的模板与生成器）、`pyi/gen_pyi.py`、`testing/` |
| `benchmarks/` | 549 | — | 按子系统组织的 benchmark：`operator_benchmark/`、`dynamo/`、`inductor_backends/`、`distributed/`、`transformer/` 等 |

这张表里有一条需要立刻记住的分层关系：**`c10/` → `aten/` → `torch/csrc/` → `torch/`**，从底到顶，下层不依赖上层。一个符号如果是 Tensor 的基本属性（dtype、device、stride），在 `c10/`；如果是一个算子，在 `aten/`；如果是 Python 能看到的东西的 C++ 那一半，在 `torch/csrc/`；纯 Python 的模块，在 `torch/` 其他目录。

### 2. vLLM 的目录

在 v0.28.0 检出上统计：仓库根目录 31 个条目，`git ls-files` 列出 6596 个文件、约 202 万行（`.py` 文件约 131 万行：`vllm/` 下 83 万、`tests/` 下 48 万；其余是 CUDA/C++、JSON 配置、文档与数据）。

| 目录 | 文件数 | 行数（约） | 一句话职责 |
|---|---|---|---|
| `vllm/` | 2841 | 99.8 万 | Python 包本体，58 个顶层条目：`config/`、`engine/`、`v1/`、`model_executor/`、`entrypoints/`、`distributed/`、`compilation/`、`lora/`、`multimodal/`、`platforms/`、`_custom_ops.py` 等 |
| `vllm/v1/` | 350 | 14.7 万 | 引擎核心：`engine/`（`AsyncLLM`、`EngineCore`、`EngineCoreClient`）、`core/`（KV cache 管理与 `sched/` 调度器）、`worker/`、`executor/`、`attention/`、`sample/`、`spec_decode/`、`metrics/` |
| `vllm/model_executor/` | 1337 | 42.1 万 | 模型与层：`models/`（293 个条目，每个模型一个文件）、`layers/`（`linear.py`、`layernorm.py`、`fused_moe/`、`quantization/`、`rotary_embedding/`）、`model_loader/`；其中 570 个 `.json` 是 fused MoE 的 kernel 调优配置，占约 8.2 万行 |
| `vllm/entrypoints/` | 197 | 4.0 万 | 用户入口：`cli/`（`vllm` 命令的子命令）、`openai/`（OpenAI 兼容 HTTP 服务，`api_server.py`）、`llm.py`（离线 `LLM` 类）、`anthropic/`、`grpc_server.py` |
| `csrc/` | 302 | 12.4 万 | CUDA/C++ kernel：`attention/`、`moe/`、`quantization/`、`cpu/`、`rocm/`、`cutlass_extensions/`；`libtorch_stable/` 下是以 libtorch 稳定 ABI 注册的 kernel 与 `torch_bindings.cpp` |
| `tests/` | 1926 | 50.1 万 | pytest 测试，40 个子目录按子系统组织：`kernels/`、`v1/`、`entrypoints/`、`models/`、`distributed/`、`quantization/`、`lora/`、`evals/`、`benchmarks/` |
| `benchmarks/` | 129 | — | `benchmark_serving.py`、`benchmark_throughput.py`、`benchmark_latency.py` 等端到端脚本；`kernels/`、`cutlass_benchmarks/`、`fused_kernels/`、`attention_benchmarks/` 是 kernel 级 |
| `docs/contributing/` | 16 | — | 贡献文档：`README.md`、`incremental_build.md`、`deprecation_policy.md`、`profiling.md`、`vulnerability_management.md`、`editing-agent-instructions.md`；子目录 `ci/`（`failures.md`、`nightly_builds.md`、`update_pytorch_version.md`）、`model/`（`basic.md`、`registration.md`、`tests.md`、`multimodal.md`、`transcription.md`）、`dockerfile/` |

vLLM 的分层比 PyTorch 简单：**`csrc/`（kernel）→ `vllm/`（一切其他）**，Python 占绝大多数。读 vLLM 时真正的难点不是语言边界而是 `vllm/` 内部的分工：请求怎么从 `entrypoints/` 进入 `v1/engine/`，再由 `v1/core/sched/` 调度、`v1/worker/` 执行、`model_executor/` 算出来。这条链第三章会走一遍。

### 3. 两个项目自带的地图

两个项目都提供了自己的地图，只是形态不同。

**PyTorch**：`pytorch CONTRIBUTING.md` 的 "Codebase structure" 一节是一份带链接的目录树，逐条解释 `c10`、`aten`（到 `native/` 下 `cpu/`、`cuda/`、`mps/`、`sparse/`、`quantized/` 各子目录）、`torch`（`csrc/` 下 `jit/`、`autograd/`、`api/`、`distributed/`）、`tools`、`torchgen`、`test`（列出 `test_torch.py`、`test_autograd.py`、`test_nn.py`、`test_jit.py`、`cpp/`、`expect/`、`onnx/`）、`caffe2`。它对 `aten/src/ATen/native/` 的注释值得原文记住："If you want to write a new operator, here is where it should go"。这一节还指向多个子目录自己的 README：`aten/src/README.md`、`aten/src/ATen/native/README.md`（讲 `native_functions.yaml` 每个字段的含义，"Registering a function in `native_functions.yaml`" 一节）、`torch/csrc/README.md`、`torch/csrc/autograd/README.md`、`torch/csrc/jit/README.md`、`tools/README.md`。

`docs/source/community/` 下的 `contribution_guide.md` 在 v2.14.0 中开头就标注 "This page has been deprecated"，指向 GitHub wiki 上的 "The Ultimate Guide to PyTorch Contributions"；同目录的 `governance.md` 与 `persons_of_interest.md` 仍然有效，后者列出每个模块的维护者，是找 reviewer 时的参考。

**vLLM**：没有一份目录职责表。`vllm docs/contributing/README.md` 讲的是流程（Job Board、Developing、Linting、Testing、Issues、Pull Requests & Code Reviews），不讲目录。仓库根目录的 `CONTRIBUTING.md` 只有一句话，指向 docs.vllm.ai 上的 contributing 页面——也就是 `docs/contributing/README.md` 渲染后的版本。目录级的知识分散在 `docs/design/` 与 `docs/contributing/model/`（加模型时该改哪些文件）里。所以读 vLLM 的第一步是自己补一张目录表——上面那张就是。


## 三、找到入口点

入口点是"用户可见的 API 或命令"与"真正干活的代码"之间的第一个连接点。两个项目的入口结构都是固定的：PyTorch 的 `torch.<op>` 全部经过同一条绑定链路，vLLM 的 `vllm <subcommand>` 全部经过同一个 CLI 分发器。掌握这两条链路，任何一个具体符号的追踪都是套模板。

### 1. PyTorch：`torch.logaddexp` 到 CUDA kernel

以引言里那个 issue 为例——`torch.logaddexp` 在 complex128 上 CPU 与 CUDA 不一致。目标是找到 CUDA kernel 所在的文件。

**第一步：确认它是一个 ATen 算子。** 在 `torch/__init__.py` 里找不到 `def logaddexp`，因为它不是 Python 定义的。`torch/__init__.py` 中有这样一段：`from torch._C._VariableFunctions import *`，以及紧随其后的一个循环 `for __name in dir(_C._VariableFunctions)`，把 C++ 模块 `torch._C._VariableFunctions` 上的所有函数挂到 `torch` 命名空间。所以 `torch.logaddexp` 是 `torch._C._VariableFunctions.logaddexp`，一个 C++ 绑定。

**第二步：查登记表。** 所有 ATen 算子都登记在 `aten/src/ATen/native/native_functions.yaml`：

```bash
# 在 pytorch/ 根目录执行
rg -n -A6 '^- func: logaddexp' aten/src/ATen/native/native_functions.yaml
```

输出（v2.14.0）：

```yaml
- func: logaddexp.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA, MPS, XPU: logaddexp_out
  tags: pointwise

- func: logaddexp(Tensor self, Tensor other) -> Tensor
  variants: method, function
  structured_delegate: logaddexp.out
  tags: pointwise
```

这十行告诉我们四件事：`logaddexp` 是一个 structured kernel（`structured: True`），它的实现委托给 `logaddexp.out`（`structured_delegate`）；CPU、CUDA、MPS、XPU 四个后端共用一个 C++ 函数名 `logaddexp_out`（`dispatch:`）；它同时是方法与函数（`variants`），所以 `x.logaddexp(y)` 与 `torch.logaddexp(x, y)` 都存在。每个字段的含义在 `aten/src/ATen/native/README.md` 里有解释。

**第三步：找 `dispatch:` 指向的 C++ 函数。**

```bash
rg -n 'logaddexp' aten/src/ATen/native/BinaryOps.cpp aten/src/ATen/native/BinaryOps.h
```

输出：

```text
aten/src/ATen/native/BinaryOps.cpp:75:#include <ATen/ops/logaddexp2_native.h>
aten/src/ATen/native/BinaryOps.cpp:76:#include <ATen/ops/logaddexp_native.h>
aten/src/ATen/native/BinaryOps.cpp:324:CREATE_BINARY_META_FUNC(logaddexp)
aten/src/ATen/native/BinaryOps.cpp:407:DEFINE_DISPATCH(logaddexp_stub);
aten/src/ATen/native/BinaryOps.cpp:544:CREATE_BINARY_TORCH_IMPL_FUNC(logaddexp_out, logaddexp_stub)
aten/src/ATen/native/BinaryOps.h:95:DECLARE_DISPATCH(structured_binary_fn, logaddexp_stub)
```

这里有两个让"跳转到定义"失效的东西。一是 `#include <ATen/ops/logaddexp_native.h>`——仓库里没有这个文件，它由 `torchgen` 生成（第四章）。二是 `CREATE_BINARY_TORCH_IMPL_FUNC(logaddexp_out, logaddexp_stub)` 是一个宏，展开后定义 `TORCH_IMPL_FUNC(logaddexp_out)`，函数体只做一件事：调用 `logaddexp_stub`。`DEFINE_DISPATCH(logaddexp_stub)` 与 `DECLARE_DISPATCH(structured_binary_fn, logaddexp_stub)` 是 PyTorch 的"按设备分发桩"机制——每个后端各自向这个 stub 注册自己的 kernel。所以下一步是找谁向 `logaddexp_stub` 注册了。

**第四步：找 kernel。**

```bash
rg -l 'logaddexp' aten/src/ATen/native/cpu aten/src/ATen/native/cuda
```

输出：

```text
aten/src/ATen/native/cpu/BinaryOpsKernel.cpp
aten/src/ATen/native/cuda/LogAddExpKernel.cu
```

CPU kernel 是 `aten/src/ATen/native/cpu/BinaryOpsKernel.cpp` 里的 `logaddexp_kernel(TensorIteratorBase& iter)`，文件末尾 `ALSO_REGISTER_AVX512_DISPATCH(logaddexp_stub, &logaddexp_kernel)` 把它注册到 stub；CUDA kernel 是 `aten/src/ATen/native/cuda/LogAddExpKernel.cu`。**这就是要改的文件。**整个过程四条命令，不需要构建，不需要 IDE。

把这条链写成一行，它对任何 `torch.<op>` 都适用：

```text
torch.<op>  →  torch/__init__.py 从 torch._C._VariableFunctions 导入
            →  native_functions.yaml 的 "- func: <op>" 条目（看 dispatch: 与 structured_delegate:）
            →  aten/src/ATen/native/<Something>.cpp 的 TORCH_IMPL_FUNC / 普通函数（常经过宏与 DEFINE_DISPATCH）
            →  aten/src/ATen/native/{cpu,cuda,mps,...}/<Kernel>.{cpp,cu,mm} 里的 kernel，用 REGISTER_DISPATCH 注册
```

中间那一层——`torch._C._VariableFunctions` 的 C++ 实现——不在仓库的静态文件里。`tools/autograd/templates/python_torch_functions.cpp` 是模板，`tools/autograd/gen_python_functions.py` 按 yaml 为每个算子生成一个 `THPVariable_<op>` 函数，分三个 shard 输出到 `torch/csrc/autograd/generated/python_torch_functions_0.cpp` 等（`num_shards=3`）；`torch/csrc/autograd/python_torch_functions_manual.cpp` 的 `gatherTorchFunctions()` 把三个 shard 收集起来，`initTorchFunctions()` 把它们注册为 `torch._C._VariableFunctions` 模块。`torch/csrc/autograd/generated/` 整个目录在 `.gitignore` 里，构建前不存在。

### 2. vLLM：`vllm serve` 到引擎

vLLM 的入口是一条 CLI 命令。目标：找到 `vllm serve <model>` 执行时，请求是被哪个类处理、引擎在哪里被创建。

**第一步：命令从哪里进来。** Python 包的命令行入口在 `pyproject.toml`：

```bash
# 在 vllm/ 根目录执行
rg -n -A1 'project.scripts' pyproject.toml
```

输出：

```toml
[project.scripts]
vllm = "vllm.entrypoints.cli.main:main"
```

**第二步：CLI 分发。** `vllm/entrypoints/cli/main.py` 的 `main()` 导入 `vllm.entrypoints.cli.serve`、`vllm.entrypoints.cli.openai`、`vllm.entrypoints.cli.benchmark.main`、`vllm.entrypoints.cli.run_batch`、`vllm.entrypoints.cli.collect_env`、`vllm.entrypoints.cli.launch` 六个子命令模块，用 `FlexibleArgumentParser` 建子解析器。`vllm/entrypoints/cli/` 目录本身就是子命令清单：

```text
vllm/entrypoints/cli/
├── main.py          # main()：注册子命令、解析参数、分发
├── serve.py         # ServeSubcommand：vllm serve
├── openai.py        # vllm chat / vllm complete
├── benchmark/       # vllm bench ...
├── run_batch.py     # vllm run-batch
├── collect_env.py   # vllm collect-env
├── launch.py
└── types.py         # CLISubcommand 基类
```

**第三步：`serve` 子命令做什么。** `vllm/entrypoints/cli/serve.py` 的 `ServeSubcommand.cmd()`：如果指定了 `--grpc` 走 `vllm.entrypoints.grpc_server.serve_grpc`；如果 `api_server_count > 1` 或启用了 Rust 前端，走 `run_multi_api_server(args)`；否则——最常见的单进程情况——`uvloop.run(run_server(args))`。`run_server` 从 `vllm.entrypoints.openai.api_server` 导入。

**第四步：HTTP 服务与引擎的创建。** `vllm/entrypoints/openai/api_server.py` 是 OpenAI 兼容服务的主文件：`run_server()` → `run_server_worker()` 启动 uvicorn；`build_async_engine_client()` → `build_async_engine_client_from_engine_args()` 创建引擎客户端，后者的关键一行是：

```python
from vllm.v1.engine.async_llm import AsyncLLM
...
async_llm = AsyncLLM.from_vllm_config(...)
```

`init_app_state()` 初始化 app state；HTTP 路由本身不在这个文件里，而是按接口拆在 `vllm/entrypoints/openai/chat_completion/api_router.py`、`completion/api_router.py`、`models/api_router.py` 等文件中（`@router.post(...)` 装饰器），请求处理类如 `OpenAIServingChat` 在 `vllm/entrypoints/openai/chat_completion/serving.py`。

**第五步：引擎。** `vllm/v1/engine/async_llm.py` 的 `AsyncLLM(EngineClient)` 是前端进程里的引擎对象；它的 `__init__` 里 `self.engine_core = EngineCoreClient.make_async_mp_client(...)` 创建一个到引擎核心进程的客户端。引擎核心本身是 `vllm/v1/engine/core.py` 的 `EngineCore`（单进程版）与 `EngineCoreProc`（独立进程版，多进程时用）；调度器在 `vllm/v1/core/sched/`，KV cache 管理在 `vllm/v1/core/kv_cache_manager.py`。

写成一行：

```text
vllm serve  →  pyproject.toml [project.scripts]
            →  vllm/entrypoints/cli/main.py  main()
            →  vllm/entrypoints/cli/serve.py  ServeSubcommand.cmd()
            →  vllm/entrypoints/openai/api_server.py  run_server() / build_async_engine_client_from_engine_args()
            →  vllm/v1/engine/async_llm.py  AsyncLLM  →  vllm/v1/engine/core_client.py  EngineCoreClient
            →  vllm/v1/engine/core.py  EngineCore / EngineCoreProc  →  vllm/v1/core/sched/  调度器
```

从命令到调度器六跳，全在 Python 里，`rg` 加"跳转到定义"就能走完。vLLM 的语言边界在更下面：当追到 `vllm/model_executor/layers/layernorm.py` 里调用 `ops.rms_norm(...)`（`from vllm import _custom_ops as ops`）时，`vllm/_custom_ops.py` 的 `rms_norm()` 只有一行 `torch.ops._C.rms_norm(out, input, weight, epsilon)`，这里 Python 侧到头了——`torch.ops._C` 是编译好的扩展模块注册进来的算子，下一章说怎么追。

### 3. 两条追踪的共同点

两个项目的入口结构差别很大，但追踪方法一样：**找到那个"登记表"**。PyTorch 的登记表是 `native_functions.yaml`，vLLM 的登记表是 `pyproject.toml` 的 `[project.scripts]`（命令）与 `csrc/libtorch_stable/torch_bindings.cpp`（kernel）。每个大项目都有这样的登记表——它们是把"名字"映射到"实现"的地方，找到它就找到了所有入口。


## 四、生成代码与"找不到定义"

### 1. torchgen 生成什么

PyTorch 有大量代码不在仓库里，而是构建时由 `torchgen/` 与 `tools/autograd/` 从 `native_functions.yaml` 和模板生成。贡献者看到的现象是：仓库里 `#include` 了不存在的头文件；`rg` 找不到一个明明能 `import` 的函数；IDE 的"跳转到定义"落到 `.pyi` 里的类型签名后断掉。

生成物的位置（v2.14.0 的 `.gitignore` 与 `cmake/Codegen.cmake` 为准）：

```text
生成物                                        位置                                          来源
────────────────────────────────────────────  ──────────────────────────────────────────  ────────────────────────────────────────────
ATen/ops/<op>.h、<op>_native.h、<op>_ops.h 等   build/aten/src/ATen/ops/（安装后 torch/include/ATen/ops/）   torchgen/gen.py 读 native_functions.yaml
RegisterCPU.cpp、RegisterCUDA.cpp 等分发注册   build/aten/src/ATen/                          torchgen/gen.py
python_torch_functions_{0,1,2}.cpp 等 Python 绑定  torch/csrc/autograd/generated/（.gitignore 整目录）  tools/autograd/gen_python_functions.py + tools/autograd/templates/
VariableType_N.cpp 等 autograd 包装             torch/csrc/autograd/generated/                 tools/autograd/ 的模板
torch/_C/_VariableFunctions.pyi、__init__.pyi、_nn.pyi   torch/_C/（.gitignore 逐文件列出）      tools/pyi/gen_pyi.py 从同名 .pyi.in 生成
torch/_VF.pyi、torch/return_types.pyi、torch/nn/functional.pyi   同上                       同上
```

`cmake/Codegen.cmake` 里能看到这些生成物是怎么进入构建的：它先以 `--dry-run` 跑一次 `torchgen`，得到 `${CMAKE_BINARY_DIR}/aten/src/ATen/ops_generated_*.cmake` 等清单，再 `include` 进来。这也解释了为什么 `aten/src/ATen/native/BinaryOps.cpp` 能 `#include <ATen/ops/logaddexp_native.h>`：这个头文件在 `build/aten/src/ATen/ops/` 下，构建后 `torch/include/ATen/ops/` 里也有一份。

### 2. 怎么处理

三条规则：

**规则一：看到 `ATen/ops/` 头文件，回到 yaml。** `ATen/ops/logaddexp_native.h` 里是 `at::native::logaddexp_out` 等声明，内容完全由 yaml 里 `logaddexp` 条目的 `dispatch:` 决定。要知道它声明了什么，读 yaml 比读生成的头文件快。

**规则二：改 `.pyi.in`，不改 `.pyi`。** `torch/_C/_VariableFunctions.pyi.in`、`torch/_C/__init__.pyi.in`、`torch/_C/_nn.pyi.in`、`torch/_C/return_types.pyi.in` 是模板，`tools/pyi/gen_pyi.py` 的 `gen_pyi()` 把 yaml 里的签名填进去生成 `.pyi`。`.pyi` 在 `.gitignore` 里，改了也提交不上去。PyTorch 的 `AGENTS.md` 把这条写成了给 AI 助手的硬规则："Many `.pyi` files are generated from corresponding `.pyi.in` templates. Always edit the `.pyi.in` file, not the generated `.pyi`."

**规则三：Python 侧的 `torch.<op>` 找不到定义是正常的，用 yaml 代替。** IDE 跳转到 `_VariableFunctions.pyi` 就停了，因为往下是 C++。这时候不要试图从 `.pyi` 继续跳，直接 `rg '^- func: <op>' aten/src/ATen/native/native_functions.yaml`。

如果一定要在 C++ 侧用 clangd 跳转（比如想跟进 `TORCH_IMPL_FUNC` 宏展开后的函数、或者看 `RegisterCUDA.cpp` 里生成的分发代码），就必须构建一次——第五章。

### 3. vLLM 的 `_C` 扩展

vLLM 没有源码级的代码生成，但有另一种"找不到定义"：**编译进 `.so` 的算子**。`vllm/_custom_ops.py` 里几乎每个函数都是 `torch.ops._C.<name>(...)` 一行；`torch.ops._C` 这个命名空间是在扩展模块加载时填充的。追法：

```bash
# 在 vllm/ 根目录执行
rg -n 'rms_norm\(' csrc/libtorch_stable/torch_bindings.cpp
rg -ln 'void rms_norm\(' csrc
```

第一条命中 `csrc/libtorch_stable/torch_bindings.cpp` 里 `STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops)` 块中的 `ops.def("rms_norm(Tensor! result, Tensor input, Tensor? weight, float epsilon) ...")`，同文件末尾 `STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops)` 块把它绑到 C++ 函数；第二条命中 `csrc/libtorch_stable/layernorm_kernels.cu`（CUDA 实现）与 `csrc/cpu/layernorm.cpp`（CPU 实现）。声明在 `csrc/libtorch_stable/ops.h`。

扩展模块的名字在 `vllm/platforms/cuda.py` 里：`import vllm._C_stable_libtorch  # noqa`，注释写着 "import custom ops, trigger op registration"；`CudaPlatform.import_kernels()` 还会尝试导入 `vllm._moe_C_stable_libtorch` 等。这些 `.so` 由 `CMakeLists.txt` 定义（v0.28.0 的注释说明 CUDA 算子已从旧的 `_C` 迁到 `_C_stable_libtorch`，`_C` 仅 ROCm 保留），`*.so` 在 `.gitignore` 里。

vLLM 版的一行链：

```text
vllm/model_executor/layers/<layer>.py  →  vllm/_custom_ops.py  def <name>()：torch.ops._C.<name>(...)
                                       →  csrc/libtorch_stable/torch_bindings.cpp  ops.def("<name>(...)") / ops.impl(...)
                                       →  csrc/libtorch_stable/<kernel>.cu  或  csrc/cpu/<kernel>.cpp
```


## 五、构建一次

### 1. 为什么必须构建

不构建能做的事：用 `rg` 追符号、读 yaml、读 Python、读测试、读 git 历史——本文第三章的两条追踪都不需要构建。不构建做不了的事：

- **clangd 跨文件跳转。** clangd 需要 `compile_commands.json` 才知道每个 `.cpp` 的 include 路径与宏定义；没有它，`aten/` 下每个文件都报一堆找不到头文件的错，"跳转到定义"对宏和模板完全失效。`pytorch CONTRIBUTING.md` 的 "Code completion and IDE support" 一节写明：`python -m pip install -e . -v --no-build-isolation` 会生成 `compile_commands.json`，并提示 `pip install ninja` 以得到 `torch/csrc` 部分的准确信息。`compile_commands.json` 本身在 `.gitignore` 里。
- **看到生成代码。** `torch/include/ATen/ops/`、`torch/csrc/autograd/generated/`、`torch/_C/*.pyi` 只在构建后存在。
- **跑测试。** 改了 C++ 或 CUDA 之后要验证，必须有一个能 `import` 的本地构建。

所以"构建一次"的目标是把工具链打通，不是为了马上改代码。它在贡献流程里的位置是"读懂"与"改动"之间——第一次追踪不需要，第一个 PR 之前必需。

### 2. PyTorch 的构建命令

`pytorch CONTRIBUTING.md` 与 `AGENTS.md` 给出的命令是同一条：

```bash
# 在 pytorch/ 根目录执行；C++/CUDA 全量构建，首次数十分钟到数小时
python -m pip install -e . -v --no-build-isolation
```

`AGENTS.md` 的说法更绝对："All build (both codegen, C++ and python) is done via `pip install -e . -v --no-build-isolation`. You should NEVER run any other command to build PyTorch."——codegen、C++、Python 三者都由这一条命令完成。`CONTRIBUTING.md` 的 "Tips and Debugging" 一节解释了 `-e` 的含义：Python 文件改了不用重装，但 `.pyi`、`.pyi.in`、`.cpp`、`.cu`、`.h` 改了要重装（或者按同一节的说法，把 `build/lib/libtorch_cpu.*` 软链到 `torch/lib/` 之后用 `ninja torch_cpu` 增量编译）。

首次构建可以关掉不需要的部分。"Build only what you need" 一节列出了 `USE_CUDA=0`、`USE_DISTRIBUTED=0`、`BUILD_TEST=0`、`USE_MKLDNN=0`、`USE_FLASH_ATTENTION=0` 等环境变量，并给了一个最小配置的 alias：

```bash
alias BUILD_CONFIG='CMAKE_GENERATOR=Ninja USE_DISTRIBUTED=0 USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 USE_MKLDNN=0 USE_CUDA=0 BUILD_TEST=0 USE_FBGEMM=0 USE_NNPACK=0 USE_XNNPACK=0 BUILD_LAZY_TS_BACKEND=0 USE_PYTORCH_QNNPACK=0 USE_CPU_VECTORIZATION=0 USE_COLORIZE_OUTPUT=1'
BUILD_CONFIG pip install --no-build-isolation -v -e .              # 最小 CPU 构建
BUILD_CONFIG USE_CUDA=1 pip install --no-build-isolation -v -e .   # 加上 CUDA
```

对第三章那个 CUDA kernel 的修复，需要 `USE_CUDA=1`；对一个纯 Python 的改动（比如 `torch/nn/` 或 `torch/_dynamo/`），根本不需要编译 C++——这是 `tools/nightly.py` 的用途。`CONTRIBUTING.md` 的 "Nightly Checkout & Pull" 一节：

```bash
./tools/nightly.py checkout -b my-nightly-branch          # 新建分支并安装 nightly 预编译二进制到仓库目录
./tools/nightly.py checkout -b my-nightly-branch --cuda   # CUDA 版
./tools/nightly.py pull                                   # 把 nightly 提交拉进当前分支并重装
source venv/bin/activate
```

原文对它的定位是 "to ease pure Python development of PyTorch ... This is like a development or editable install, but without needing the ability to compile any C++ code"。代价是它不生成 `compile_commands.json`，C++ 侧仍然没法用 clangd。

v2.14.0 的 `CONTRIBUTING.md` 有一节 "Spin"，把 `spin` 作为开发命令的统一入口：`spin develop` 等价于 editable install，`spin clean` 清理构建产物，`spin lint` / `spin fixlint` 跑 lint，`spin regenerate-type-stubs` 单独重新生成 `.pyi`。第三篇会用到 lint 部分。

### 3. vLLM 的构建命令

`vllm docs/contributing/README.md` 的 "Developing" 一节把两种情况分开：

```bash
# 在 vllm/ 根目录执行
# 只改 Python：用预编译的 kernel wheel，几分钟
VLLM_USE_PRECOMPILED=1 uv pip install -e .

# 也改 CUDA/C++：先装 torch，再装构建依赖，再全量编译（数十分钟到数小时）
uv pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129
grep -v '^torch==' requirements/build/cuda.txt | uv pip install -r -
uv pip install -e . --no-build-isolation
```

`VLLM_USE_PRECOMPILED=1` 是 vLLM 相对 PyTorch 的一个明显优势：绝大多数 vLLM 贡献只改 Python，这条命令让贡献者完全跳过 CUDA 编译。`AGENTS.md` 的 "Installing dependencies" 给的是同一条命令加 `--torch-backend=auto`。

改 kernel 时，每次 `uv pip install -e .` 全量重编太慢。`vllm docs/contributing/incremental_build.md`（标题 "Incremental Compilation Workflow"）给出的做法是绕过 pip、直接用 CMake 增量编译：

```bash
# 在 vllm/ 根目录执行
python tools/generate_cmake_presets.py       # 生成 CMakeUserPresets.json（自动探测 nvcc、Python、核数）
cmake --preset release                       # 配置，构建目录 cmake-build-release/
cmake --build --preset release --target install   # 编译并把 _C_stable_libtorch.abi3.so 等装回 vllm/ 源码树；改完再跑一次即增量
```

这份文档的 "Verifying the Build" 一节列出了 `cmake-build-release/` 下的产物，并说明 `--target install` 把编译出的 `.so` 拷回 `vllm` 包目录，让 editable install 直接用上新 kernel。CMake 配置里 `CMAKE_C_COMPILER_LAUNCHER` 等设为 `ccache`，第二次起编译只重编改过的文件。要让 clangd 能在 `csrc/` 下跳转，在 `CMakeUserPresets.json` 的 `cacheVariables` 里加一项 `"CMAKE_EXPORT_COMPILE_COMMANDS": "ON"`（CPU 后端的 `cmake/cpu_extension.cmake` 已默认打开，CUDA 构建需要自己加），`cmake --preset release` 之后 `cmake-build-release/compile_commands.json` 就有了。

### 4. 两个项目的对照

```text
                   PyTorch v2.14.0                                  vLLM v0.28.0
─────────────────  ───────────────────────────────────────────────  ────────────────────────────────────────────
只改 Python        tools/nightly.py checkout（nightly 预编译二进制）    VLLM_USE_PRECOMPILED=1 uv pip install -e .
改 C++/CUDA        pip install -e . -v --no-build-isolation（可用环境变量裁剪）   uv pip install -e . --no-build-isolation
增量编译           build/ 下 ninja <target>；或 SKBUILD_EDITABLE_REBUILD  cmake --build --preset release --target install
compile_commands   pip install -e 生成（需 ninja）                    cmake --preset 生成于 cmake-build-release/（preset 需打开导出）
统一入口           spin（develop / clean / lint / regenerate-*）         无；uv + pre-commit
```


## 六、用测试当文档

一个函数最准确的规格说明不是它的 docstring，是它的测试：测试写了哪些输入、期望哪些输出、跳过了哪些设备与 dtype，就是这个函数被承诺的行为边界。对贡献者，测试还有第二个用途——它是改动之后必须跑、必须补的东西，第三篇讲规范，本章只讲怎么找到与读懂。

### 1. PyTorch：`test/` 的组织

`pytorch CONTRIBUTING.md` 的 "Python Unit Testing" 一节："All PyTorch test suites are located in the `test` folder and start with `test_`"，整套跑 `python test/run_test.py`，单个文件 `python test/test_jit.py`，单个测试 `python test/test_jit.py TestJit.test_Sequential`；"Better local unit tests with `pytest`" 一节说 pytest 不是官方支持但能用，`pytest test/test_nn.py -k Loss -v`。

在 v2.14.0 检出上，`test/` 顶层有 176 个 `test_*.py` 与 46 个子目录。顶层文件按主题命名：`test_torch.py`、`test_autograd.py`、`test_nn.py`、`test_ops.py`、`test_binary_ufuncs.py`、`test_unary_ufuncs.py`、`test_linalg.py`、`test_cuda.py`、`test_mps.py`……子目录按子系统：`distributed/`、`dynamo/`、`inductor/`、`export/`、`functorch/`、`fx/`、`jit/`、`onnx/`、`quantization/`、`nn/`、`optim/`、`profiler/`、`cpp/`（C++ gtest，构建后在 `build/bin/` 下，"C++ Unit Testing" 一节：`./build/bin/test_jit --gtest_filter=ContainerAliasingTest.MayContainAlias`）、`expect/`（expecttest 的期望文件）。

找一个算子的测试：

```bash
# 在 pytorch/ 根目录执行
rg -l 'logaddexp' test/*.py
```

输出：

```text
test/test_binary_ufuncs.py
test/test_decomp.py
test/test_ops_gradients.py
test/test_linalg.py
test/test_meta.py
test/test_mps.py
```

`test/test_binary_ufuncs.py` 里有 `_test_logaddexp(self, device, dtype, base2)`，用 numpy 的 `np.logaddexp` 做参考实现——这就是"CPU 与 CUDA 结果应当一致，并且与 numpy 一致"这条规格的所在。第七章会看到，修复引言那个 issue 的 commit 改的正是这个文件加 `test/test_linalg.py`。

### 2. PyTorch：测试框架的三样东西

PyTorch 的测试建立在 `torch/testing/_internal/` 上，有三个必须认识的名字。

**`common_utils.py` 的 `TestCase` 与 `run_tests`。** `class TestCase(expecttest.TestCase)` 是所有测试类的基类，提供 `assertEqual`（对 Tensor 做带容差的比较）等断言；`run_tests()` 是每个测试文件末尾 `if __name__ == "__main__": run_tests()` 调用的入口，处理命令行参数、并行、重试等。`AGENTS.md` 的 "Testing" 一节给的骨架就是这四行：

```python
from torch.testing._internal.common_utils import run_tests, TestCase

class TestFeature(TestCase):
    ...

if __name__ == "__main__":
    run_tests()
```

**`common_device_type.py` 的 `instantiate_device_type_tests`。** 一个测试类写一次，按设备实例化成多个类。`test/test_binary_ufuncs.py` 末尾：

```python
instantiate_device_type_tests(TestBinaryUfuncsDevice, globals(), allow_xpu=True)
instantiate_device_type_tests(TestBinaryUfuncsCUDA, globals(), only_for="cuda")
```

这一行把 `TestBinaryUfuncsDevice` 变成 `TestBinaryUfuncsDeviceCPU`、`TestBinaryUfuncsDeviceCUDA` 等，测试方法的 `device` 参数由框架填入。所以在 CI 日志里看到 `TestBinaryUfuncsDeviceCUDA.test_logaddexp_cuda_complex128` 这样的名字时，要去掉后缀找源码：类名 `TestBinaryUfuncsDevice`，方法 `test_logaddexp`。`AGENTS.md` 的要求是 "For any test that checks numerics of the on-device implementation, use `instantiate_device_type_tests` to write device-generic tests"。

**`common_methods_invocations.py` 的 OpInfo。** 这是 PyTorch 对每个算子的元数据登记：支持的 dtype、是否支持 forward AD、样例输入生成器、已知的跳过项。`logaddexp` 的条目：

```python
BinaryUfuncInfo('logaddexp',
                dtypes=floating_and_complex_types_and(torch.bfloat16, torch.float16),
                dtypesIfCUDA=floating_and_complex_types_and(torch.bfloat16, torch.float16, torch.complex32),
                ...
                supports_forward_ad=True,
                supports_fwgrad_bwgrad=True,
                supports_rhs_python_scalar=False,
                skips=(...)),
```

`test/test_ops.py` 的 `TestCommon` 用 `@ops(op_db, ...)` 装饰器对 `op_db` 里每个 OpInfo 跑一组通用测试（正确性、梯度、meta、decomposition 等）。这意味着：**给一个算子加了新 dtype 支持，通常只需要改它的 OpInfo，通用测试会自动覆盖。** 反过来，读一个算子的 OpInfo 是最快知道"它在哪些 dtype 和设备上被承诺可用"的方式——`dtypesIfCUDA` 比 `dtypes` 多了 `torch.complex32`，说明 CUDA 上支持 chalf 而 CPU 不支持。

### 3. vLLM：`tests/` 的组织

`vllm docs/contributing/README.md` 的 "Testing" 一节：vLLM 用 pytest，`pytest tests/` 跑全部，`pytest -s -v tests/test_logger.py` 跑单个文件；`AGENTS.md` 给的是 `.venv/bin/python -m pytest tests/path/to/test_file.py -v`。README 同一节的警告值得注意："Currently, not all unit tests pass when run on CPU platforms"——没有 GPU 的贡献者要依赖 CI。

在 v0.28.0 检出上，`tests/` 有 40 个子目录、顶层 26 个 `test_*.py`。子目录与 `vllm/` 的子系统大致一一对应：

```text
tests/kernels/        对应 csrc/ 与 vllm/model_executor/layers/：attention/、moe/、quantization/、mamba/、core/；test_cache_kernels.py 等
tests/v1/             对应 vllm/v1/：engine/、core/、worker/、executor/、attention/、sample/、spec_decode/、kv_connector/、structured_output/、e2e/
tests/entrypoints/    对应 vllm/entrypoints/：openai/、llm/、serve/、anthropic/、tool_parsers/
tests/models/         模型正确性：与 HF transformers 对比
tests/distributed/    多卡与多节点
tests/quantization/   量化方案
tests/lora/           LoRA
tests/evals/          模型评测（AGENTS.md：模型相关改动要跑）
tests/benchmarks/     benchmark 脚本自身的测试
tests/basic_correctness/、tests/compile/、tests/multimodal/、tests/tool_use/、tests/weight_loading/ ……
```

找一个改动对应的测试目录的规则很简单：`vllm/v1/engine/` 的改动看 `tests/v1/engine/`，`csrc/` 的 kernel 看 `tests/kernels/`，`vllm/entrypoints/openai/` 看 `tests/entrypoints/openai/`。第七章那个 vLLM commit 改了 `vllm/entrypoints/cli/serve.py` 与 `vllm/v1/engine/utils.py`，配套测试加在 `tests/v1/engine/test_startup_watch_processes.py`。

vLLM 没有 PyTorch 那样厚的测试框架层，`tests/conftest.py` 与各子目录的 `conftest.py` 提供 fixture（模型加载、`VllmRunner`、`HfRunner` 等）。`AGENTS.md` 的 "Tests" 一节对新测试提了四个问题（模块做什么、I/O 契约、防的是什么失败、最便宜的层级）和几条原则（复用已有文件、一个测试一个行为、不在 `tests/` 里放一次性 kernel benchmark），第三篇展开。


## 七、读历史

### 1. 三条命令

代码只回答"它是什么"，历史回答"它为什么是这样"。三条 git 命令覆盖大部分需要：

```bash
# 这一行/这一段是谁在哪次改动里写的（-w 忽略空白改动，-C 跟踪跨文件移动）
git blame -w -C -L <start>,<end> --date=short <file>

# 哪些 commit 增删过这个字符串（-S 找出现次数变化的提交，比 --grep 准）
git log -S'<symbol>' --format='%h %cs %s' -- <path>

# 一个 commit 的完整信息与改动统计
git show -s --format='%H%n%an%n%cs%n%B' <sha>
git show --stat --format= <sha>
```

`-w -C` 很重要：大型项目频繁做格式化与文件搬迁，不带这两个参数，`blame` 会把一半的行归到某次 "fix typos" 或 "move file" 上。

### 2. PyTorch：`Pull Request resolved:`

以第三章那个 CUDA kernel 为例：

```bash
# 在 pytorch/ 根目录执行
git log -3 --format='%h %cs %s' -- aten/src/ATen/native/cuda/LogAddExpKernel.cu
```

输出（v2.14.0）：

```text
1ccb743b7b5 2025-11-29 [BE][4/5] fix typos in aten/ (aten/src/ATen/native/) (#157553)
2ddcf53e1a9 2025-11-17 Logaddexp complex inconsistent bw cpu and cuda (#163509)
57a49018b10 2024-11-03 [5/N] Fix Wextra-semi warning  (#139465)
```

第二条就是修引言那个 issue 的提交。看它的完整信息：

```bash
git show -s --format='%H%n%an%n%cs%n%B' 2ddcf53e1a9
```

```text
2ddcf53e1a98d4453a2d2ff2422af19bc04bd26e
Chris Leonard
2025-11-17
Logaddexp complex inconsistent bw cpu and cuda (#163509)

Fixes #158429

Updated LogAddExpKernel.cu to allow for complex numbers. Also, updated unittest to run test_logaddexp on CUDA with complex data types and added a unit test in test_linalg.py to compare results between CUDA and cpu.

@drisspg
Pull Request resolved: https://github.com/pytorch/pytorch/pull/163509
Approved by: https://github.com/isuruf
```

PyTorch 的 commit message 是由合入机器人生成的，格式固定：标题末尾的 `(#163509)` 是 PR 号；正文是 PR 描述；`Fixes #158429` 关联 issue；末尾两个 trailer `Pull Request resolved:` 与 `Approved by:` 分别是 PR 链接与批准者。顺着 `Pull Request resolved:` 的链接（或 `gh pr view 163509 --repo pytorch/pytorch --comments`）能看到当时的 review 对话；顺着 `Fixes #158429`（`gh issue view 158429 --repo pytorch/pytorch`）能看到 issue 原文——截至 2026-09 查询，这个 issue 的标题是 "Inconsistent torch.logaddexp results on complex128 between CPU and CUDA"，标签 `triaged`、`module: linear algebra`、`topic: fuzzer`。

`git show --stat --format= 2ddcf53e1a9` 显示这个 commit 改了四个文件：`aten/src/ATen/native/cuda/LogAddExpKernel.cu`（+235）、`test/test_binary_ufuncs.py`、`test/test_linalg.py`、`torch/testing/_internal/common_methods_invocations.py`。**一个 kernel 修复 = kernel 文件 + 直接测试 + 对比测试 + OpInfo 的 dtype 更新**，这是第六章讲的结构在一个真实 commit 里的样子。`git tag --contains 2ddcf53e1a9` 显示它最早进入 `v2.11.0`。

`git blame` 的一个例子——看 CPU kernel `logaddexp_kernel` 的前三行是谁写的：

```bash
git blame -w -C -L 1062,1064 --date=short aten/src/ATen/native/cpu/BinaryOpsKernel.cpp
```

```text
0503105bc287 (Freey0 2021-05-07 1062) void logaddexp_kernel(TensorIteratorBase& iter) {
455241bbd362 (CaoE   2023-11-06 1063)   if (at::isReducedFloatingType(iter.dtype())) {
455241bbd362 (CaoE   2023-11-06 1064)     AT_DISPATCH_REDUCED_FLOATING_TYPES(iter.dtype(), "logaddexp_cpu", [&
```

函数签名是 2021 年的，reduced-float 分支是 2023 年加的；`git log -S'logaddexp_kernel' -- aten/src/ATen/native/cpu/BinaryOpsKernel.cpp` 再往前追到 `05f097b5bbf 2020-05-27 Implement logaddexp (#38384)`——这个算子的诞生。

### 3. vLLM：`Signed-off-by` 与标题里的 PR 号

vLLM 的 commit 是 GitHub squash merge 产生的，形态不同：

```bash
# 在 vllm/ 根目录执行
git log -5 --format='%h %cs %s' -- vllm/entrypoints/cli/serve.py
```

```text
fa722b9f01 2026-08-10 [Rust Frontend][gRPC] Add explicit data-parallel rank routing (#51178)
7b9f2dad89 2026-08-06 [Frontend] Watch frontend processes during engine startup (#43417)
726ef437a1 2026-07-31 [chore] delete useless code (#49424)
01661cc57f 2026-07-28 [Rust][Benchmark] Make `vllm bench serve` Rust delegation opt-in (#50081)
ab3b6d97aa 2026-07-04 [Frontend] Limit `SO_REUSEPORT` to multi-worker serving (#47529)
```

标题带 `[Frontend]`、`[chore]` 这类前缀（第三篇讲 PR 标题规范），末尾 `(#43417)` 是 PR 号。看正文：

```bash
git show -s --format='%H%n%an%n%cs%n%B' 7b9f2dad89
```

```text
7b9f2dad8920f115c1caea36e096e43c04c3da68
Bugen Zhao
2026-08-06
[Frontend] Watch frontend processes during engine startup (#43417)

Signed-off-by: Bugen Zhao <i@bugenzhao.com>
Signed-off-by: Nick Hill <nickhill123@gmail.com>
Co-authored-by: OpenAI Codex <codex@openai.com>
Co-authored-by: Nick Hill <nickhill123@gmail.com>
```

正文里只有 trailer：`Signed-off-by`（DCO 签名，vLLM 每个 commit 必须有）、`Co-authored-by`（这里包括一个 AI 助手——`AGENTS.md` 要求 AI 辅助的提交用这个 trailer 声明）。**"为什么改"不在 commit 里，在 PR 页面上。**所以读 vLLM 历史的第二步永远是 `gh pr view <PR 号> --repo vllm-project/vllm`。截至 2026-09 查询，#43417 的描述以 "## Purpose" 开头："Fail promptly when a frontend process exits while engine cores are still initializing"，接着解释了 `launch_core_engines` 的启动屏障原来只监视本地引擎核心与 DP coordinator，API server 进程在这个窗口内退出会让父进程一直等；标签是 `frontend`、`ready`、`v1`、`rust`。改动涉及 `vllm/entrypoints/cli/serve.py`、`vllm/v1/engine/core_client.py`、`vllm/v1/engine/utils.py`，测试在 `tests/v1/engine/test_startup_watch_processes.py`。`git tag --contains 7b9f2dad89` 显示它进入了 `v0.28.0`（以及 `v0.27.2rc0`）。

两个项目的对比：

```text
                      PyTorch                                              vLLM
────────────────────  ───────────────────────────────────────────────────  ───────────────────────────────────────────────
commit 正文            PR 描述全文 + Fixes # + Pull Request resolved: + Approved by:   通常只有 Signed-off-by / Co-authored-by trailer
PR 号                  标题末尾 (#N)，正文 Pull Request resolved: 链接             标题末尾 (#N)
"为什么"在哪            commit 里基本能读到；review 细节在 PR 页                    只在 PR 页；必须 gh pr view
关联 issue             正文 Fixes #N                                          PR 描述里（commit 不带）
```

### 4. `RELEASE.md`：改动什么时候到用户手里

两个项目根目录都有 `RELEASE.md`，回答"我的改动合入之后什么时候进到一个版本"。

**PyTorch** `RELEASE.md` 的 "Release Cadence" 一节是一张表，列出每个 minor 版本的 release branch cut 日期与发布日期。v2.14.0 检出里的近几行：

```text
| Minor Version | Release branch cut | Release date | First patch release date | Second patch release date|
| 2.12 | 13 Apr 2026 | 13 May 2026 | Jun 2026 | Not planned |
| 2.13 | 8 Jun 2026 | 8 Jul 2026 | (Aug 2026) | Not planned |
| 2.14 | 10 Aug 2026 | 2 Sept 2026 | (Oct 2026) | Not planned |
| 2.15 | 28 Sept 2026 | 28 Oct 2026 | (Nov 2026) | Not planned |
| 2.16 | 23 Nov 2026 | 22 Dec 2026 | (Jan 2027) | Not planned |
```

表格上方注明 "All future dates below are tentative" 与 "Patch Releases are optional"。从表里读出的节奏：branch cut 到发布约三到四周，相邻 minor 版本相隔约两个月。"General Overview" 一节列出发布的步骤（cut release branch → drafting RCs 与 cherry-pick → final RC → promote to stable）；"Frequently Asked Questions" 解释 branch cut 之后 "new features *are not* added to the release branch"，cherry-pick 需要在 release tracker issue 里提名，可以用 `@pytorchbot cherry-pick -c [reason]` 自动化；"Cherry Picking Fixes" 一节强调 "The cherry pick process is not an invitation to add new features, it is mainly there to fix regressions"，`-c` 的取值是 `regression`、`critical`、`fixnewfeature`、`docs`、`release` 五种。

对贡献者的含义：一个在 8 月 10 日之后合入 main 的 feature，不会进 2.14，要等 10 月 28 日的 2.15；如果它是 regression fix，可以走 cherry-pick 进 2.14 的 patch 版本。

**vLLM** `RELEASE.md` 的 "Release Cadence and Versioning" 一节原文："We aim to have a regular release every 2 weeks. Since v0.12.0, regular releases increment the minor version rather than patch version." 三种版本的定义：Major 留给 "architectural milestones involving sweeping API changes, similar to PyTorch 2.0"；Minor 是常规发布；Patch 是 "special releases for new models, as well as emergency patches for critical performance, functionality and security issues"。"Release Branch" 一节：major/minor 的 release branch cut 在发布前 1–2 天；构建由推 `vX.Y.Z-rc1` 这样的 RC tag 触发，最终 tag `vX.Y.Z` 不触发构建；"Cherry-Pick Criteria" 允许 regression fix、critical fix、对最近一版新功能的修复、文档、release 分支特定改动，并加粗 "No feature work allowed for cherry picks"。

对贡献者的含义：vLLM 合入 main 的改动最多等两周就在下一个 minor 版本里；branch cut 到发布只有一两天，几乎没有 cherry-pick 的窗口——想进某个版本，就在那个版本 cut 之前合入。

两个节奏放在一起：

```text
                    PyTorch                                     vLLM
──────────────────  ──────────────────────────────────────────  ──────────────────────────────────────────
常规版本间隔         约 2 个月（RELEASE.md 日程表）                  约 2 周（"every 2 weeks"）
branch cut → 发布    约 3–4 周                                    1–2 天
版本号递增           minor（2.13 → 2.14）                         minor（0.27 → 0.28，自 v0.12.0 起）
cherry-pick          有明确流程与 tracker issue，@pytorchbot cherry-pick  有准则；窗口极短
patch 版本           optional，日程表列出                          new models 与紧急修复
```


## 八、核心问题：两小时定位流程

回到本篇的核心问题：**给你一个从未见过的百万行仓库和一个报错信息，两小时之内你能把它定位到一个文件的一个函数吗？靠什么？**答案是靠一套固定的流程，而不是靠对这个项目的熟悉。下面是这套流程，每步给出时间预算与两个项目的具体做法。

### 1. 流程清单

```text
时间     步骤                          做法                                                                产出
───────  ────────────────────────────  ──────────────────────────────────────────────────────────────────  ────────────────────
0:00     读报错，提取符号               从 traceback / 错误信息里挑出 2–3 个最具体的标识符：函数名、类名、错误字符串     一个 grep 词表
0:05     画顶层地图                    ls 根目录；读项目自带的地图（PyTorch CONTRIBUTING.md "Codebase structure"；      目录职责表（第二章）
                                        vLLM 自己列 vllm/ 的顶层目录）；判断符号大概在哪一层
0:15     rg 第一轮                     rg -n '<symbol>' 限定到最可能的目录；错误字符串直接 rg -F               候选文件 ≤ 5 个
0:25     找登记表                      PyTorch：rg '^- func: <op>' native_functions.yaml；                    dispatch / 注册位置
                                        vLLM：pyproject.toml [project.scripts]、csrc/libtorch_stable/torch_bindings.cpp
0:35     沿链追到实现                  按第三章的链条走：yaml dispatch → native/*.cpp → cpu|cuda/*.cu；             目标文件 + 函数名
                                        cli/main.py → cli/<sub>.py → openai/api_server.py → v1/engine/
0:50     识别生成代码                  找不到的 #include <ATen/ops/...> 回 yaml；.pyi 回 .pyi.in；               不再被"找不到定义"卡住
                                        torch.ops._C.<x> 回 torch_bindings.cpp
1:00     读测试                        rg -l '<symbol>' test/*.py 或 tests/<subsystem>/；读断言与跳过项；            行为规格；要跑的测试文件
                                        PyTorch 再看 OpInfo 条目
1:20     读历史                        git log -S'<symbol>' -- <file>；git show -s 最近一次相关 commit；            为什么是这样；有没有人已经在改
                                        PyTorch 顺 Pull Request resolved:；vLLM gh pr view <N>
1:40     写进项目地图                  把追过的这条路径、用过的 rg 模式、找到的测试文件记入贡献日志                    第九章的一页
2:00     （可选）开始构建               如果要改 C++/CUDA：按第五章的命令启动构建，在等待时继续读                   compile_commands.json
```

两个小时里没有一步依赖"已经构建好"或"已经熟悉这个项目"。构建放在最后、可选，因为定位阶段用不到它；它是改动阶段的前置条件。

### 2. 三个常见的岔路

**符号太常见。** `rg 'forward'` 在两个仓库都是上万条。对策：加上下文（`rg 'def forward' vllm/model_executor/layers/layernorm.py`）、限定目录、或用错误字符串代替符号——错误字符串几乎总是唯一的。

**追到宏或装饰器就停了。** PyTorch 的 `TORCH_IMPL_FUNC`、`REGISTER_DISPATCH`、`AT_DISPATCH_*`，vLLM 的 `@CustomOp.register`、`@support_torch_compile`。对策：把宏名当符号再 rg 一次，找它的定义看展开成什么；PyTorch 的分发宏在 `aten/src/ATen/native/DispatchStub.h`，看一次就够。

**多个实现，不知道运行时走哪个。** `dispatch:` 列了四个后端，`vllm/model_executor/layers/` 一个层有多个 backend 实现。对策：在 Python 侧加一个断点或 `print(torch.ops.aten.logaddexp)`、`inspect.getsourcefile`；或者读测试——测试里的参数化告诉你哪些组合是被承诺的。

### 3. 用两个真实案例验证

引言里的 PyTorch 案例：报错信息是 "Inconsistent torch.logaddexp results on complex128 between CPU and CUDA"。符号 `logaddexp`，层次判断是算子 → `aten/`；yaml 条目 → `dispatch: CPU, CUDA, MPS, XPU: logaddexp_out` → `BinaryOps.cpp` 的宏 → `rg -l logaddexp aten/src/ATen/native/cuda` → `LogAddExpKernel.cu`。测试在 `test/test_binary_ufuncs.py` 的 `_test_logaddexp`，OpInfo 在 `common_methods_invocations.py`。历史：`git log -- aten/src/ATen/native/cuda/LogAddExpKernel.cu` 显示 2025-11-17 的 #163509 已经修了这个问题（`Fixes #158429`）。**结论：这个 issue 已经被修，在 v2.11.0 里。**四十分钟内得到"不用做"的结论，这本身就是流程的价值——它避免了引言那位工程师白做一周。

vLLM 案例：想知道 `vllm serve` 启动时"engine core 还在初始化、API server 进程已退出"会发生什么。符号 `api_server_count`、`launch_core_engines`；`rg -n 'launch_core_engines' vllm/` 命中 `vllm/v1/engine/utils.py` 与 `vllm/entrypoints/cli/serve.py`；`git log -- vllm/entrypoints/cli/serve.py` 第二条是 2026-08-06 的 #43417 "Watch frontend processes during engine startup"，`gh pr view 43417` 的 Purpose 精确描述了这个场景，测试在 `tests/v1/engine/test_startup_watch_processes.py`。**结论：v0.28.0 已包含这个行为，测试文件就是它的规格。**


## 九、贡献日志：项目地图

本系列的贯穿实践是一份贡献日志 `contrib-log.md`。本篇新增它的第一页——**项目地图**：目录职责、构建命令、测试入口、常用 grep 模式、追过的三条符号路径。每一项都要有出处（一条命令或一个路径），不写感觉。下面是模板与两份填好的样例；读者选定自己的项目后照着填一份。

### 1. 模板

```markdown
项目地图：<项目> <版本 tag>（<填写日期>）

[目录职责]
| 目录 | 职责 | 出处 |
|---|---|---|

[自带的地图]
- <文件路径 + 章节名>

[构建]
- 只改 Python：<命令>
- 改 C++/CUDA：<命令>
- 增量：<命令>
- compile_commands.json 在：<路径>

[测试入口]
- 跑单个文件：<命令>
- 测试目录与源码目录的对应：<规则>
- 框架约定：<基类 / fixture / 元数据>

[常用 grep 模式]
| 想找什么 | 命令 |
|---|---|

[追过的符号路径（≥3 条）]
1. <起点> → ... → <终点文件:函数>（日期）
```

### 2. 样例：PyTorch v2.14.0

```markdown
项目地图：PyTorch v2.14.0（2026-09-12）

[目录职责]
| 目录 | 职责 | 出处 |
|---|---|---|
| c10/ | 核心类型与分发键，最底层 | CONTRIBUTING.md "Codebase structure" |
| aten/src/ATen/native/ | 算子实现；native_functions.yaml 是登记表；cpu/ cuda/ mps/ 是后端 kernel | 同上；aten/src/ATen/native/README.md |
| torch/csrc/ | C++ 绑定与 autograd/jit/distributed/inductor 的 C++ 侧 | 同上；torch/csrc/README.md |
| torch/ | Python 包；_dynamo/ _inductor/ nn/ distributed/ testing/ | 同上 |
| torchgen/ + tools/autograd/ | 代码生成器与模板 | torchgen/gen.py；tools/autograd/templates/ |
| test/ | 176 个 test_*.py + 46 个子目录 | ls test |
| benchmarks/ | operator_benchmark/ dynamo/ inductor_backends/ 等 | ls benchmarks |

[自带的地图]
- CONTRIBUTING.md "Codebase structure"
- aten/src/ATen/native/README.md（yaml 字段）
- docs/source/community/persons_of_interest.md（模块维护者）
- AGENTS.md（构建/测试/.pyi.in 三条硬规则）

[构建]
- 只改 Python：./tools/nightly.py checkout -b <branch> [--cuda]；source venv/bin/activate
- 改 C++/CUDA：python -m pip install -e . -v --no-build-isolation（可加 USE_CUDA=0 等裁剪，见 "Build only what you need"）
- 增量：cd build && ninja <target>；或 SKBUILD_EDITABLE_REBUILD=true spin develop
- compile_commands.json 在：仓库根目录（pip install -e 生成，需 ninja）
- 重新生成 .pyi：spin regenerate-type-stubs

[测试入口]
- 跑单个文件：python test/test_binary_ufuncs.py TestBinaryUfuncsDeviceCPU.test_logaddexp_cpu_float32；或 pytest test/test_nn.py -k Loss -v
- 对应规则：算子 → test/test_<family>_ufuncs.py / test_ops.py + OpInfo；子系统 → test/<subsystem>/
- 框架约定：torch/testing/_internal/common_utils.py TestCase/run_tests；common_device_type.py instantiate_device_type_tests（CI 里的类名去掉 CPU/CUDA 后缀）；common_methods_invocations.py OpInfo

[常用 grep 模式]
| 想找什么 | 命令 |
|---|---|
| 算子登记 | rg -n -A8 '^- func: <op>' aten/src/ATen/native/native_functions.yaml |
| dispatch 指向的 C++ 函数 | rg -n '<name>' aten/src/ATen/native/*.cpp aten/src/ATen/native/*.h |
| 后端 kernel | rg -l '<op>' aten/src/ATen/native/{cpu,cuda,mps} |
| Python 绑定的模板 | rg -n '<op>' tools/autograd/templates/ tools/autograd/*.yaml |
| 算子的测试 | rg -l '<op>' test/*.py；rg -n "'<op>'" torch/testing/_internal/common_methods_invocations.py |
| 错误字符串 | rg -F -n '<exact message>' aten torch/csrc torch |

[追过的符号路径]
1. torch.logaddexp → torch/__init__.py（from torch._C._VariableFunctions import *）→ native_functions.yaml logaddexp / logaddexp.out（dispatch: CPU, CUDA, MPS, XPU: logaddexp_out）→ aten/src/ATen/native/BinaryOps.cpp CREATE_BINARY_TORCH_IMPL_FUNC(logaddexp_out, logaddexp_stub) → aten/src/ATen/native/cuda/LogAddExpKernel.cu；CPU 在 cpu/BinaryOpsKernel.cpp logaddexp_kernel（2026-09-12）
2. torch._C._VariableFunctions 本身 → tools/autograd/templates/python_torch_functions.cpp + gen_python_functions.py（num_shards=3）→ 生成 torch/csrc/autograd/generated/python_torch_functions_{0,1,2}.cpp → torch/csrc/autograd/python_torch_functions_manual.cpp gatherTorchFunctions() / initTorchFunctions()（2026-09-12）
3. #include <ATen/ops/logaddexp_native.h> → 仓库中不存在 → cmake/Codegen.cmake 调 torchgen 生成到 build/aten/src/ATen/ops/，安装后 torch/include/ATen/ops/（2026-09-12）

[历史]
- LogAddExpKernel.cu 最近的实质改动：2ddcf53e1a9 2025-11-17 #163509 "Logaddexp complex inconsistent bw cpu and cuda"，Fixes #158429，进入 v2.11.0
- RELEASE.md：2.15 branch cut 28 Sept 2026，发布 28 Oct 2026
```

### 3. 样例：vLLM v0.28.0

```markdown
项目地图：vLLM v0.28.0（2026-09-12）

[目录职责]
| 目录 | 职责 | 出处 |
|---|---|---|
| vllm/entrypoints/ | cli/（vllm 子命令）、openai/（api_server.py）、llm.py（离线 LLM） | ls vllm/entrypoints |
| vllm/v1/ | engine/（AsyncLLM、EngineCore、EngineCoreClient）、core/sched/（调度）、worker/、executor/、attention/ | ls vllm/v1 |
| vllm/model_executor/ | models/（293 个条目）、layers/、model_loader/；570 个 fused_moe JSON 配置 | ls；git ls-files 'vllm/model_executor/**/*.json' |
| vllm/_custom_ops.py | Python → torch.ops._C.<op> 的包装 | rg 'torch.ops._C' vllm/_custom_ops.py |
| csrc/ | CUDA/C++ kernel；libtorch_stable/torch_bindings.cpp 是登记表 | ls csrc |
| tests/ | 40 个子目录，与 vllm/ 子系统对应 | ls tests |
| benchmarks/ | benchmark_serving/throughput/latency.py；kernels/ | ls benchmarks |
| docs/contributing/ | README.md、incremental_build.md、deprecation_policy.md、ci/、model/ | ls docs/contributing |

[自带的地图]
- docs/contributing/README.md（流程，不含目录）
- docs/contributing/model/（加模型要改的文件）
- AGENTS.md（查重命令、uv 环境、测试四问）
- 无目录职责表——用上表

[构建]
- 只改 Python：VLLM_USE_PRECOMPILED=1 uv pip install -e .（AGENTS.md 加 --torch-backend=auto）
- 改 C++/CUDA：uv pip install torch ... --extra-index-url .../cu129；grep -v '^torch==' requirements/build/cuda.txt | uv pip install -r -；uv pip install -e . --no-build-isolation
- 增量：python tools/generate_cmake_presets.py；cmake --preset release；cmake --build --preset release --target install（docs/contributing/incremental_build.md）
- compile_commands.json 在：cmake-build-release/（preset 的 cacheVariables 需加 CMAKE_EXPORT_COMPILE_COMMANDS: ON）

[测试入口]
- 跑单个文件：.venv/bin/python -m pytest tests/v1/engine/test_startup_watch_processes.py -v
- 对应规则：vllm/v1/<x>/ → tests/v1/<x>/；csrc/ → tests/kernels/；vllm/entrypoints/openai/ → tests/entrypoints/openai/；模型 → tests/models/
- 框架约定：pytest；tests/conftest.py 的 fixture；模型相关改动要跑 tests/evals/ 或 vllm bench（AGENTS.md）
- 注意：CPU 上不是所有测试都能过（docs/contributing/README.md "Testing"）

[常用 grep 模式]
| 想找什么 | 命令 |
|---|---|
| CLI 子命令 | ls vllm/entrypoints/cli/；rg -n 'class .*Subcommand' vllm/entrypoints/cli/ |
| HTTP 路由 | rg -n '@router\.(get\|post)' vllm/entrypoints/openai/ （命中各接口的 api_router.py） |
| 一个 custom op 的 kernel | rg -n '<op>' csrc/libtorch_stable/torch_bindings.cpp；rg -ln 'void <op>\(' csrc |
| 一个层的实现 | rg -n 'class <Layer>' vllm/model_executor/layers/ |
| 一个模型 | ls vllm/model_executor/models/ \| rg -i <name> |
| 配置项 | rg -n '<field>' vllm/config/ vllm/engine/arg_utils.py |
| 错误字符串 | rg -F -n '<exact message>' vllm |

[追过的符号路径]
1. vllm serve → pyproject.toml [project.scripts] → vllm/entrypoints/cli/main.py main() → vllm/entrypoints/cli/serve.py ServeSubcommand.cmd() → vllm/entrypoints/openai/api_server.py run_server() / build_async_engine_client_from_engine_args() → vllm/v1/engine/async_llm.py AsyncLLM.from_vllm_config → vllm/v1/engine/core_client.py EngineCoreClient.make_async_mp_client → vllm/v1/engine/core.py EngineCoreProc（2026-09-12）
2. RMSNorm.forward → vllm/model_executor/layers/layernorm.py ops.rms_norm → vllm/_custom_ops.py rms_norm(): torch.ops._C.rms_norm → csrc/libtorch_stable/torch_bindings.cpp STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) ops.def("rms_norm(...)") → csrc/libtorch_stable/layernorm_kernels.cu rms_norm；CPU 在 csrc/cpu/layernorm.cpp（2026-09-12）
3. torch.ops._C 从哪来 → vllm/platforms/cuda.py import vllm._C_stable_libtorch（"trigger op registration"）→ CMakeLists.txt 的 _C_stable_libtorch 目标 → *.so 在 .gitignore（2026-09-12）

[历史]
- serve.py 最近的实质改动：7b9f2dad89 2026-08-06 #43417 "[Frontend] Watch frontend processes during engine startup"，commit 只有 trailer，Purpose 在 gh pr view；进入 v0.28.0
- RELEASE.md："regular release every 2 weeks"；branch cut 在发布前 1–2 天
```

两份样例里所有路径与命令都是本文前面各章核对过的。读者填自己的那份时，第一条符号路径应该是自己遇到的那个问题——不必是本文的例子。


## 十、本文小结

### 1. 要点回顾

- 百万行代码库不能通读，只能**有目标地检索**：从符号、报错、issue 出发，用工具追到底，只读路径上的东西。四种典型失败——从头读、随便读、被生成代码卡住、只读代码不读测试与历史——都是把阅读当成了线性活动。
- **先画地图。** PyTorch 的分层是 `c10/` → `aten/` → `torch/csrc/` → `torch/`，`CONTRIBUTING.md` 的 "Codebase structure" 一节与各子目录 README 是自带的地图；vLLM 是 `csrc/` → `vllm/`，Python 占绝大多数，没有目录职责表，`docs/contributing/` 讲的是流程，目录表要自己补。
- **找登记表。** PyTorch 的 `aten/src/ATen/native/native_functions.yaml` 把每个算子的 `dispatch:` 指向 C++ 函数，再经 `DEFINE_DISPATCH` 桩到 `cpu/`、`cuda/` 下的 kernel；`torch.<op>` 经 `torch/__init__.py` 的 `from torch._C._VariableFunctions import *` 进入 C++。vLLM 的 `pyproject.toml` `[project.scripts]` 把 `vllm` 命令指向 `vllm/entrypoints/cli/main.py`，`serve` 子命令经 `api_server.py` 到 `vllm/v1/engine/`；kernel 的登记表是 `csrc/libtorch_stable/torch_bindings.cpp`。
- **生成代码是"找不到定义"的主因。** `torchgen` 从 yaml 生成 `ATen/ops/*.h`（`build/` 与 `torch/include/`）、`torch/csrc/autograd/generated/`、`torch/_C/*.pyi`（从 `.pyi.in`）；处理办法是回到 yaml、改 `.pyi.in`、C++ 侧构建后用 clangd。vLLM 的 `torch.ops._C.<op>` 由 `_C_stable_libtorch.abi3.so` 注册，回 `torch_bindings.cpp` 找。
- **构建一次是为了工具链，不是为了改代码。** PyTorch：`python -m pip install -e . -v --no-build-isolation`（生成 `compile_commands.json`，可用 `USE_CUDA=0` 等裁剪），纯 Python 开发用 `tools/nightly.py`；vLLM：`VLLM_USE_PRECOMPILED=1 uv pip install -e .` 跳过 CUDA 编译，改 kernel 用 `incremental_build.md` 的 `cmake --preset release` 流程。
- **测试是规格。** PyTorch 的 `test/` 按主题（176 个 `test_*.py`）与子系统（46 个子目录）组织，框架三件套是 `common_utils.py` 的 `TestCase`/`run_tests`、`common_device_type.py` 的 `instantiate_device_type_tests`、`common_methods_invocations.py` 的 OpInfo；vLLM 的 `tests/` 40 个子目录与 `vllm/` 子系统对应，pytest 加 `conftest.py` fixture。
- **历史是注释。** `git log -S`、`git blame -w -C`、`git show`。PyTorch 的 commit 正文含 PR 描述、`Fixes #`、`Pull Request resolved:`、`Approved by:`；vLLM 的 commit 只有 `Signed-off-by`/`Co-authored-by` trailer，"为什么"要 `gh pr view`。`RELEASE.md`：PyTorch 约两个月一个 minor、cut 到发布三四周、有 cherry-pick 流程；vLLM 约两周一版、cut 到发布一两天。
- **核心问题的答案**是一张两小时流程清单：提取符号 → 画地图 → rg → 找登记表 → 沿链追 → 识别生成代码 → 读测试 → 读历史 → 记入地图；构建可选、放最后。用它走引言的案例，四十分钟得出"已在 v2.11.0 修复"的结论。

### 2. PyTorch 与 vLLM 对照

| 环节 | PyTorch v2.14.0 | vLLM v0.28.0 |
|---|---|---|
| 规模（检出统计） | `git ls-files` 21663 个文件、约 475 万行（不含 `third_party/`）；`test/` 146 万行 | `git ls-files` 6596 个文件、约 202 万行；`.py` 约 131 万行 |
| 分层 | `c10/` → `aten/` → `torch/csrc/` → `torch/` | `csrc/` → `vllm/` |
| 自带地图 | `CONTRIBUTING.md` "Codebase structure"；子目录 README；`persons_of_interest.md` | `docs/contributing/`（流程）；`docs/contributing/model/`；无目录表 |
| 算子/命令登记表 | `native_functions.yaml`；`tools/autograd/` 模板 | `pyproject.toml` `[project.scripts]`；`csrc/libtorch_stable/torch_bindings.cpp` |
| 生成代码 | `torchgen` → `ATen/ops/*.h`、`autograd/generated/`、`.pyi`（从 `.pyi.in`） | 无源码生成；编译产物 `_C_stable_libtorch.abi3.so` |
| 只改 Python 的构建 | `tools/nightly.py checkout` | `VLLM_USE_PRECOMPILED=1 uv pip install -e .` |
| 改 C++/CUDA 的构建 | `pip install -e . -v --no-build-isolation` | `uv pip install -e . --no-build-isolation`；增量用 `cmake --preset release` |
| 测试框架 | unittest；`TestCase`/`run_tests`/`instantiate_device_type_tests`/OpInfo | pytest；`conftest.py` fixture |
| 测试目录 | `test/test_*.py` 按主题 + 46 个子目录 | `tests/` 40 个子目录对应 `vllm/` 子系统 |
| commit 形态 | 正文 = PR 描述 + `Fixes #` + `Pull Request resolved:` + `Approved by:` | 标题 `[Tag] ... (#N)`；正文 `Signed-off-by`、`Co-authored-by` |
| 发布节奏 | 约 2 个月一个 minor；cut → 发布 3–4 周；`@pytorchbot cherry-pick` | 约 2 周一版；cut → 发布 1–2 天；minor 递增 |

### 3. 本篇涉及的文件位置

| 路径 | 内容 |
|---|---|
| pytorch `CONTRIBUTING.md` | "Codebase structure"、"Tips and Debugging"、"Nightly Checkout & Pull"、"Spin"、"Python Unit Testing"、"Better local unit tests with `pytest`"、"C++ Unit Testing"、"Build only what you need"、"Code completion and IDE support" 各节 |
| pytorch `AGENTS.md` | 构建命令唯一性、`TestCase`/`run_tests` 骨架、`instantiate_device_type_tests`、`.pyi.in` 规则 |
| pytorch `RELEASE.md` | "Release Cadence" 日程表、"General Overview"、"Frequently Asked Questions"、"Cherry Picking Fixes" |
| pytorch `aten/src/ATen/native/native_functions.yaml` | `logaddexp` / `logaddexp.out` 条目（`structured`、`structured_delegate`、`dispatch`、`variants`） |
| pytorch `aten/src/ATen/native/README.md` | "Registering a function in `native_functions.yaml`" 及各字段 |
| pytorch `aten/src/ATen/native/BinaryOps.cpp`、`BinaryOps.h` | `CREATE_BINARY_TORCH_IMPL_FUNC(logaddexp_out, logaddexp_stub)`、`DEFINE_DISPATCH`、`DECLARE_DISPATCH` |
| pytorch `aten/src/ATen/native/cpu/BinaryOpsKernel.cpp`、`cuda/LogAddExpKernel.cu` | `logaddexp_kernel`、`ALSO_REGISTER_AVX512_DISPATCH`；CUDA kernel |
| pytorch `torch/__init__.py` | `from torch._C._VariableFunctions import *` 与 `dir(_C._VariableFunctions)` 循环 |
| pytorch `torch/csrc/autograd/python_torch_functions_manual.cpp` | `gatherTorchFunctions()`、`initTorchFunctions()` |
| pytorch `tools/autograd/gen_python_functions.py`、`tools/autograd/templates/python_torch_functions.cpp` | Python 绑定生成器（`num_shards=3`）与模板 |
| pytorch `torchgen/gen.py`、`torchgen/model.py`、`cmake/Codegen.cmake` | 代码生成入口 `main()`、`NativeFunction`、生成物清单的 CMake 接入 |
| pytorch `tools/pyi/gen_pyi.py`、`torch/_C/*.pyi.in` | `.pyi` 生成 |
| pytorch `tools/nightly.py` | 纯 Python 开发的预编译安装 |
| pytorch `torch/testing/_internal/common_utils.py`、`common_device_type.py`、`common_methods_invocations.py` | `TestCase`、`run_tests`；`instantiate_device_type_tests`；`BinaryUfuncInfo('logaddexp', ...)` |
| pytorch `test/test_binary_ufuncs.py`、`test/test_ops.py` | `_test_logaddexp`、末尾的 `instantiate_device_type_tests`；`TestCommon` 与 `@ops(op_db)` |
| pytorch `docs/source/community/` | `contribution_guide.md`（已标 deprecated）、`governance.md`、`persons_of_interest.md` |
| pytorch `.gitignore` | `torch/_C/_VariableFunctions.pyi`、`torch/csrc/autograd/generated/*`、`torch/include/`、`compile_commands.json` |
| vllm `CONTRIBUTING.md`、`docs/contributing/README.md` | 根文件一句话指向 docs；"Job Board"、"Developing"、"Linting"、"Testing" 各节 |
| vllm `docs/contributing/incremental_build.md` | "Incremental Compilation Workflow"：`tools/generate_cmake_presets.py`、`cmake --preset release`、`--target install`、"Verifying the Build" |
| vllm `AGENTS.md` | uv 环境、`VLLM_USE_PRECOMPILED=1`、测试四问、`Co-authored-by` trailer |
| vllm `RELEASE.md` | "Release Cadence and Versioning"、"Release Branch"、"Cherry-Pick Criteria" |
| vllm `pyproject.toml` | `[project.scripts]` `vllm = "vllm.entrypoints.cli.main:main"` |
| vllm `vllm/entrypoints/cli/main.py`、`serve.py` | `main()`；`ServeSubcommand.cmd()`、`run_multi_api_server` |
| vllm `vllm/entrypoints/openai/api_server.py` | `run_server()`、`build_async_engine_client_from_engine_args()`、`init_app_state()` |
| vllm `vllm/v1/engine/async_llm.py`、`core_client.py`、`core.py` | `AsyncLLM`、`EngineCoreClient.make_async_mp_client`、`EngineCore`/`EngineCoreProc` |
| vllm `vllm/_custom_ops.py`、`vllm/platforms/cuda.py` | `torch.ops._C.rms_norm` 包装；`import vllm._C_stable_libtorch` |
| vllm `csrc/libtorch_stable/torch_bindings.cpp`、`ops.h`、`layernorm_kernels.cu`；`csrc/cpu/layernorm.cpp` | `STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops)`、`ops.def("rms_norm(...)")`；`rms_norm` 声明与 CUDA/CPU 实现 |
| vllm `tests/v1/engine/test_startup_watch_processes.py` | #43417 的配套测试 |


## 下一篇

[找到切入点：从 issue、RFC 到性能回归](/finding-your-entry-point-in-open-source.html)。本篇解决的是"给一个问题，能不能找到代码"；下一篇解决的是"该找哪个问题"——大多数失败的贡献不是做错了，而是选错了。它会读两个项目的标签体系（PyTorch `.github/labeler.yml` 与 `actionable`；vLLM `good first issue`、`new-model`、`rfc-required` 与 Job Board）、RFC 模板、CI 失败看板与性能回归模板，用 `gh` 实时抓一组真实 issue 做切入点清单，并回答：一个项目每天新增几十个 issue、几十个 PR，maintainer 最希望有人来做的是哪一类工作？你怎么判断自己选的题不会在一周后被关闭？
