---
layout: post
series: contributing-to-ai-infra-open-source
title: "AI-Infra 开源贡献指南（04）：两个真实 PR 的完整走读——PyTorch 与 vLLM"
subtitle: "Two Real Pull Requests, End to End: One in PyTorch, One in vLLM"
tags: [Open Source, PyTorch, vLLM, AI, AI-Infra]
catalog: true
---

前三篇讲的都是规则：目录怎么读、标签怎么看、PR 怎么写、CI 怎么跑。规则读完之后最常见的一种失败是——它们在脑子里是分开的。一个贡献者知道 vLLM 要在描述里写 Purpose / Test Plan / Test Result，知道 PyTorch 要用 `@pytorchbot merge`，知道 CI 红了要先看是不是 main 本来就红；但真正开一个 PR 的时候，他不知道这些规则在**一个真实的时间线上**是怎么排列的：哪一步会卡多久、reviewer 的第一条评论通常是关于什么、"改了再提"到底要往返几次、合入之后什么时候才算进了版本。规则是静态的，一次贡献是动态的，中间缺一段实录。

本篇补这一段。它选取两个已经合入的真实 PR——PyTorch 一个、vLLM 一个——从 issue 开始，到讨论、实现、测试、CI、review、合入、进入哪个版本，逐步走读。每一步都用 `gh` CLI 从 GitHub 取真实材料：PR 描述、diff、每一条 review 评论及其时间、每一次 CI 触发与失败、每一个标签是谁在什么时候打的。正文引用的所有评论都标注 handle 与日期，所有数字都能用 `gh pr view` / `gh api` 复核；本文不虚构任何一句对话。

两个 PR 都不大：PyTorch 的那个改了一个文件、加了 104 行；vLLM 的那个改了六个文件、加 109 行删 16 行。但一个从 issue 到合入用了一个月、PR 本身开了五天；另一个 PR 开了五十天，其中四十七天没有一条 review。这就是本篇要回答的核心问题：

> **两个都是"小"PR，却各花了作者一到几周。这些时间花在哪里了？哪些是可以省的，哪些是这个项目的正常成本？**

答案要靠时间线算出来，而不是靠感觉。本篇会为两个 PR 各画一张由真实时间戳（`createdAt`、每条评论的 `created_at`、`mergedAt`）构成的时间线表，把时间分解到"等待"、"往返"、"CI"、"作者自己的工作"四类里，再逐项判断可省不可省。最后一章是全系列的总结：把两个走读里出现的每一个环节映射回前三篇的对应小节，并给出系列目录。

版本与数据基线：源码路径以 PyTorch v2.14.0、vLLM v0.28.0 检出为准；两个 PR 的 GitHub 数据（描述、评论、标签、检查状态、时间戳）均为 **截至 2026-09-07 用 `gh` 查询**的结果，时间统一为 UTC。本文不引用其他系列的文章，不展开两个改动背后的技术原理——LU 分解的后端选择与 KV cache 的块分配只解释到"读懂这个 diff"所需的程度。


## 一、总览

### 1. 问题

读完规则之后直接动手，典型的失败方式有四种，两个走读里都能找到对应的现场：

- **不知道 review 在等什么**。PyTorch 的那个 PR 开出来四小时之内就收到了 reviewer 的三条行内评论，没有一条是关于算法的：一条关于代码注释里的数字与 PR 描述不一致，一条关于注释里"Blackwell"指的是哪块卡，一条要求删掉一句 "Authored with Claude"。贡献者如果把 review 想成"检查算法对不对"，会对这些意见感到意外，甚至觉得是吹毛求疵；理解了 reviewer 在守什么（可维护性、数据来源可追溯、项目政策），就知道这些恰恰是必经的第一轮。
- **不知道沉默意味着什么**。vLLM 的那个 PR 在开出后 47 天没有任何 maintainer 评论，期间 mergify 打了一次 `needs-rebase`，原 issue 被 stale bot 自动关闭。作者在这 47 天里没有 ping 过任何人。`docs/contributing/README.md` 写着"7 天没 review 可以 ping"，但知道这条规则和在第 8 天真的去 ping 是两回事。
- **不知道 CI 失败的责任归属**。PyTorch 的 PR 合入时 `pull` 里两个 docs 任务失败，作者用 `@pytorchbot merge -i` 忽略——因为它改的是一个 `.cpp` 文件，Dr. CI 给出的失败原因是脚本退出码 127。vLLM 的 PR 第一次跑 Buildkite 有六个任务失败，全部是作者自己引入的：修好了一个边界之后，三个已有测试恰好卡在那个边界上。两种失败的处理方式完全不同，区分它们的依据在 CI 日志里而不在直觉里。
- **不知道合入之后还有一段**。两个 PR 都在合入后被问"进了哪个版本"。PyTorch 的进了 v2.13.0；vLLM 的合入于 8 月 20 日，v0.28.0 在 8 月 26 日发布，但它**不在** v0.28.0 里——release 分支已经在此之前从 main 分出去了。截至 9 月 7 日它只存在于 v0.28.1rc0 与 v0.29.0 的几个 rc 标签中。不查 tag 就回答"在最新版里"，是错的。

### 2. 方法

走读一个 PR 的通用方法，是把它拆成七个阶段，每个阶段只问一个问题，并且每个问题都有一条能重复执行的命令作为答案的来源：

```text
阶段        问题                                    数据来源（gh / git）
──────────────────────────────────────────────────────────────────────────────────────────────────
起点        它回应的是哪个 issue / RFC？之前谁讨论过？   gh issue view N --comments；PR body 里的 Fixes #
阅读        作者必须读懂哪些文件？怎么找到它们？          gh pr view N --json files；rg 在检出里追符号
diff        改了什么、为什么这样改、避开了什么更大的改法？ gh pr diff N
测试与数据  加了什么测试、放在哪、数字怎么呈现？          diff 里的 test/ 与 tests/ 文件；PR body 里的表格
CI          跑了哪些任务、失败了什么、怎么处理的？        gh pr checks N；gh api .../actions/runs?head_sha=
review      reviewer 问了什么、改了什么、什么维持原样？   gh api .../pulls/N/reviews 与 /comments；/commits
合入之后    进了哪个版本？有没有 follow-up 或 revert？    git tag --contains <sha>；gh pr list --search "N"
```

七个阶段走完之后，把每条记录的时间戳排成一列，就得到时间线；时间线上每一段空白都能归到"谁在等谁"。这个方法与项目无关，它背后的理由也和前三篇一样：**reviewer 的时间是最稀缺的资源**——PR 的每一处不清楚、每一次不必要的往返、每一个作者本可以自己发现的 CI 失败，消耗的都是这份资源；走读的目的就是把这些消耗一项项看清楚。

### 3. 两个项目

两个 PR 的基本事实（截至 2026-09-07 查询）：

```text
                  PyTorch #185344                                          vLLM #47272
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
标题              torch.linalg.lu: improve heuristics for the             [Bugfix][Core] Reserve the KV null block when
                  cuSOLVER vs cuBLAS switch                                validating max_model_len
链接              github.com/pytorch/pytorch/pull/185344                   github.com/vllm-project/vllm/pull/47272
作者              @nikitaved                                               @92hyungjun
类型              算子级：CUDA LU 分解的后端选择启发式（性能）               系统级：引擎启动时 KV cache 容量校验（正确性）
前置 issue        #181999（@IvanYashchuk，2026-04-30）                     #35541（@kvcache670，2026-02-27）；前置部分修复 PR #41069
PR 创建           2026-05-27 08:33                                         2026-07-01 08:51
合入              2026-06-01 14:23（pytorchmergebot 关闭 PR，commit 进 main） 2026-08-20 02:49（@njhill 合入）
历时              5 天 6 小时（issue 起算 32 天）                           49 天 18 小时（issue 起算 174 天）
规模              1 个文件，+104 −0                                        6 个文件，+109 −16
测试              未新增测试（既有 test/test_linalg.py 覆盖两个后端）        tests/v1/core/ 新增 3 个用例；修正 3 个 e2e 测试的配置
数据              4 种 GPU、3792 个采样点的启发式命中率表                    复现脚本 + 前后行为（挂死 → 启动时 ValueError）
标签              module: cuda · module: cublas · module: linear algebra · bug · ready · v1 · kv-cache-manager
                  release notes: linalg_frontend · ciflow/trunk · open source · Merged
堆叠 / 签名       ghstack 0.15.0（8 个 "Update" commit）；CLA               4 个非 merge commit 均带 Signed-off-by；作者的 3 个带 Co-authored-by: Claude
CI                pull · trunk（ciflow/trunk）· Lint · TSan · BC Lint ·     Buildkite #84702（6 失败）→ #84725（88 项全绿）；
                  ghstack 可合入性检查；219 项检查，2 失败                    DCO · pre-commit · mergify
review            @IvanYashchuk 3 条行内 + 1 条总评；@johannesz-codes 1 条    @njhill 1 条行内（含建议代码）+ 1 个 commit；
                                                                           社区 @malaiwah 2 条长评论并开配套 PR #52530
合入方式          @pytorchbot merge → 失败 → @pytorchbot merge -i           ready 标签 → /ci run → 修 CI → /ci run → 合入
进入版本          v2.13.0（tag 2026-07-03，release 2026-07-08）              不在 v0.28.0；v0.28.1rc0 · v0.29.0rc1–rc4；无正式版本
```

两个 PR 分别体现了总纲列出的两套流程的全部要素：PyTorch 的 ghstack、模块标签、`@pytorchbot merge`、多 workflow CI、`merge_rules.yaml` 的权限；vLLM 的标题前缀、Purpose / Test Plan / Test Result、`Signed-off-by`、Buildkite 按领域触发、`ready` 标签、`/ci run`。类型上一个是需要 benchmark 数据的算子级改动，一个是需要端到端测试的系统级改动。有两条选取标准是放宽的，第二章会说明。

### 4. 本文的章节安排

| 章 | 主题 | 内容 |
|---|---|---|
| 二 | 选择两个 PR | 选取标准 · gh 查询过程 · 候选与淘汰原因 · 两条放宽的标准 |
| 三 | 走读一：PyTorch #185344 | 起点 · 阅读 · diff · 测试与数据 · CI · review · 合入之后 |
| 四 | 走读二：vLLM #47272 | 起点 · 阅读 · diff · 测试与数据 · CI · review · 合入之后 |
| 五 | 两个走读的对照表 | 同一环节两种做法 |
| 六 | 核心问题：时间去了哪里 | 两张时间线 · 四类时间的分解 · 可省与不可省 |
| 七 | 映射回前三篇 | 每个环节对应的小节 |
| 八 | 贡献日志：复盘模板 | 按走读格式重述自己的 PR |
| 九 | 本文小结 | 要点 · 对照表 · 文件位置 |
| 十 | 系列总结 | 三种能力 · 系列目录 |


## 二、选择两个 PR

### 1. 选取标准

总纲给了四条：规模适中、流程完整、有代表性、类型互补。落成可查询的条件：

```text
条件                          PyTorch                                       vLLM
──────────────────────────────────────────────────────────────────────────────────────────────────────────
合入时间                      2026-03-01 至 2026-09-07                       同
diff 规模                     约 50–500 行                                   同
关联 issue / RFC              body 含 Fixes #；commit 含 Pull Request resolved  body 含 Fixes # / Closes #，或 RFC 链接
至少一轮导致代码改动的 review  /reviews 与 /comments 有 maintainer 意见，且其后有新 commit  同
项目流程要素                  ghstack（commit 含 ghstack-poisoned / ghstack-source-id）或 @pytorchbot merge；   标题前缀；Purpose / Test Plan / Test Result；
                              module: 标签；gh pr checks 有多个 workflow      Signed-off-by；Buildkite；ready 标签
类型                          一个算子 / kernel 级、PR 里有 benchmark 数字    一个系统 / 接口级、有端到端测试
```

### 2. 查询过程

PyTorch 有一个容易踩的坑：`@pytorchbot merge` 把 PR 的 commit 直接推到 main，然后**关闭** PR 而不是 GitHub 意义上的 merge。所以 `gh pr list --state merged` 在 PyTorch 上只能查到 release 分支上少数几个用 GitHub 按钮合入的 PR。要按 `Merged` 标签查：

```bash
# 任意目录执行
gh pr list --repo pytorch/pytorch --state closed \
  --search 'label:Merged label:"module: cuda" closed:2026-03-01..2026-09-07 -author:app/pytorch-bot' \
  --limit 200 --json number,title,additions,deletions,labels,closedAt,author
```

在结果里按 `additions + deletions` 过滤到 40–500 行，得到十几个候选；再对候选逐个看 body 里有没有 `Fixes`、看 `/reviews` 与 `/commits`：

```bash
gh pr view 185344 --repo pytorch/pytorch --json body,createdAt,closedAt,files,commits,reviews,comments
gh api repos/pytorch/pytorch/pulls/185344/comments --jq '.[] | "\(.created_at) @\(.user.login) \(.body)"'
gh api repos/pytorch/pytorch/pulls/185344/commits --jq '.[] | "\(.commit.author.date) \(.commit.message)"'
```

vLLM 的 PR 是 GitHub 意义上的 merge，`--state merged` 有效，加上 `label:ready` 与标题前缀过滤：

```bash
gh pr list --repo vllm-project/vllm --state merged \
  --search 'is:merged merged:2026-03-01..2026-09-07 label:ready "[Core]" in:title fixes in:body' \
  --limit 100 --json number,title,additions,deletions,labels,mergedAt,author
```

### 3. 候选与淘汰

PyTorch 侧看了六个候选。#188110（cudaMallocAsync 在 OOM 前 trim 池并重试，+92，有 `test/test_cuda.py` 测试和真实训练负载的吞吐图，@ngimel 的 review 导致两处修改）各方面都好，但 body 里没有关联任何 issue，淘汰。#175374（allocator 的 `is_graph_capture` 标志，有 `Fixes`、有测试、有 `CHANGES_REQUESTED`）只有一个 squash 后的 commit，review 前后的代码变化无法从 commit 列表还原；且它从 2 月开到 5 月，与 vLLM 那个的形态重复。#175898（`cholesky_solve` 批量性能，有 benchmark 表和测试）关联的是另一个 PR 而不是 issue，且创建于 2 月。#180090（autograd 在 CUDA graph 捕获期间的 stream 引用，+333）规模在上限、review 三十多轮，走读会超出一小时。最终选 **#185344**：ghstack、`Fixes #181999`、作者自己打的三个 `module:` 标签、`ciflow/trunk`、benchmark 数据、三条 review 意见都导致了代码改动、`@pytorchbot merge` 失败后 `merge -i` 的完整过程。

vLLM 侧看了四个候选。#47165（把不可处理的图片 URL 从 500 改成 422，+288，`verified` 标签）流程完整但 body 用的是作者自己的章节结构而不是 Purpose / Test Plan / Test Result。#49206（priority 调度静默跳过请求，+159）与 #54962（PP 的 tensor 发送等待，+85）都符合条件，但 review 只有一轮且很短。最终选 **#47272**：`[Bugfix][Core]` 前缀、`Fixes #35541`、Purpose / Test Plan / Test Result 三段外还有 "Not a duplicate" 一段（`AGENTS.md` 对 AI 辅助 PR 的要求）、`Signed-off-by` 加 `Co-authored-by: Claude`、`ready` 标签、`/ci run` 两次、六个 Buildkite 任务失败并修复、maintainer 一条带代码的 review 意见导致新 commit 与新测试、社区成员开了配套 PR 又主动关闭——所有流程要素齐全，而且它的时间线极不均匀，正好用来回答核心问题。

### 4. 两条放宽的标准

- **PyTorch #185344 没有新增测试**。它改的是性能启发式（在两个都正确的后端之间选择），正确性由既有的 `test/test_linalg.py` 中 `test_linalg_lu_family` 等用例覆盖，PR 用 3792 个采样点的命中率表代替了测试。在 2026 年 3 月到 9 月合入的、满足其余全部条件的 PyTorch CUDA PR 里，没有一个同时具备 `Fixes #`、ghstack、benchmark 数字、review 导致修改、新增测试——第二章第 3 节里的 #188110 有测试但没有 issue。这里放宽"有测试"。
- **两个 PR 都没有 RFC**。两个都是 issue 驱动的修复；本系列讨论的 RFC 流程（`pytorch/rfcs`、vLLM 的 `750-RFC.yml`）适用于大改动，几十到几百行的 PR 本来就不该走 RFC，这一点与标准不冲突，但要说明。


## 三、走读一：PyTorch #185344

### 1. 起点

issue #181999 由 @IvanYashchuk 于 2026-04-30 08:21 开出，标题 "CUDA linalg: replace hard LU M >= 512 cuSOLVER cutoff with batch-aware heuristic"。它是一份标准的性能问题报告：指出 `aten/src/ATen/native/cuda/linalg/BatchLinearAlgebra.cpp` 里 LU 分解在 cuBLAS batched 与 cuSOLVER looped 两个后端之间的切换条件是一个硬编码的 `m >= 512`，附一段可运行的复现脚本（对 batch 8–128、n 511 与 512 计时 `torch.linalg.solve_ex`），附一张在 RTX 6000 Ada 上的实测表：batch=128 时 n=511 用 9.5 ms，n=512 用 100.1 ms——跨过 512 就掉到十倍慢。issue 的判断是："The right cutoff is a performance heuristic, and it depends strongly on batch size."

标签由 bot 与 triage 打上：`module: performance`、`module: cuda`、`module: cublas`、`module: linear algebra`、`triaged`、`enhancement`、`bot-triaged`。注意**没有 `actionable`**——`CONTRIBUTING.md` "AI-Assisted Development" 一节要求新贡献者只做带 `actionable` 的 issue，但这个 issue 的报告者是 linalg 模块的 maintainer（`.github/merge_rules.yaml` 的 "Linear Algebra" 规则 `approved_by` 列表里有 IvanYashchuk），PR 作者 @nikitaved 是长期贡献者，两人在同一个领域工作，这条规则不适用于他们。对新贡献者的教训是反过来的：这种 issue 是"maintainer 想做、并且知道该怎么做"的题，通常会由熟人接走，不是好的切入点。

issue 下的讨论只有两条。2026-05-27 @christopher-hesse 问："Shouldn't cuBLAS pick the best approach on the current hardware/dtypes/shapes so no heuristic is needed in PyTorch?"，@IvanYashchuk 同日回答 cuBLAS 与 cuSOLVER 没有统一的 LU 接口，batched 与非 batched 分属两个库，所以选择只能在 PyTorch 侧做。这条问答是理解 PR 为什么存在的最短路径——如果 issue 里已经有人问过"为什么不让库自己选"，PR 描述就不必再解释。

### 2. 阅读

要做这个改动，作者需要确认三件事：切换点在哪个函数里；两个后端各自是哪个函数、在哪个文件；有没有用户可控的开关会绕过这个选择。用第一篇的方法，从 Python API 追到 CUDA 实现（在 PyTorch v2.14.0 检出根目录执行）：

```bash
# 1. API 名 → native_functions.yaml 里的算子声明
rg -n "^- func: linalg_lu_factor_ex" aten/src/ATen/native/native_functions.yaml
# 2. 算子实现 → 设备分发 stub
rg -n "lu_factor_stub" aten/src/ATen/native/BatchLinearAlgebra.cpp
# 3. CUDA 侧注册的实现
rg -n "REGISTER_CUDA_DISPATCH\(lu_factor_stub" aten/src/ATen/native/cuda/linalg/
# 4. 两个后端的实现分别在哪
rg -n "^void lu_factor_batched_cublas|^void lu_factor_looped_cusolver" aten/src/ATen/native/cuda/linalg/
# 5. 用户开关
rg -n "linalgPreferredBackend" aten/src/ATen/Context.h
rg -n "preferred_linalg_library" torch/backends/cuda/__init__.py
```

五条命令给出完整的路径：`native_functions.yaml` 的 `linalg_lu_factor_ex` → `aten/src/ATen/native/BatchLinearAlgebra.cpp` 的 `TORCH_IMPL_FUNC(linalg_lu_factor_ex_out)` 调 `lu_factor_stub` → `aten/src/ATen/native/cuda/linalg/BatchLinearAlgebra.cpp` 用 `REGISTER_CUDA_DISPATCH(lu_factor_stub, &lu_factor)` 注册 → `lu_factor` 内部按 `at::globalContext().linalgPreferredBackend()` 与形状决定调 `BatchLinearAlgebraLibBlas.cpp` 的 `lu_factor_batched_cublas` 还是 `BatchLinearAlgebraLib.cpp` 的 `lu_factor_looped_cusolver`。改动只需要落在 `lu_factor` 这一个函数附近，这也是为什么最终 diff 只有一个文件。

测试在哪：`rg -n "def test_linalg_lu" test/test_linalg.py` 找到 `test_linalg_lu_family`、`test_linalg_lu_solve`。它们不按后端区分；新启发式只是在两个都正确的实现之间换了切换点，所以 PR 没有新增测试，靠这些既有用例保证正确性——这是第二章说明的那条放宽标准。

### 3. diff

`gh pr diff 185344 --repo pytorch/pytorch` 一共 127 行，一个文件，两个 hunk。第一个 hunk 在 `lu_factor_batched_magma` 之后新增一段（`#ifdef USE_LINALG_SOLVER` 内、`#ifndef USE_ROCM` 内），先是一个枚举：

```cpp
enum class SolverBackend : char {
  CUSOLVER,
  CUBLAS
};
```

然后是大约三十行注释，把启发式的来源和规则写清楚——这段注释是 review 的主战场，第 6 节会回来看它：

```cpp
  // Based on benchmarks across H100, A100, L40, RTX5090 with about 3800 points:
  // - with batch dims in the range 2^i, with i in 0-8;
  // - square matrices of dim 2^i and (2^{i+1} + 2^i)/2, with 2^k <= 8192;
  // - square matrices of dim 2^i-/+1;
  // Rule: use cuSOLVER when n*n > threshold, where threshold depends on
  // batch size and dtype.
```

规则本身是一个纯函数 `get_lu_factor_solver_backend(batch, m, n, dtype)`：非方阵与 batch=1 走 cuSOLVER（前者因为 cuBLAS 不支持矩形输入，后者因为 cuBLAS 的优势只在批量场景）；batch=2 用一组小阈值（float32/complex64 为 `8400 * batch`，double 类为 `2200 * batch`）；batch>2 时 float32/complex64 用 `18600 * batch`，float64 用 `16600 * batch * isqrt(batch)`，complex128 用 `5200 * batch * isqrt(batch)`；最后 `n * n > threshold` 则 cuSOLVER，否则 cuBLAS。注释解释了为什么 float64 需要 `batch^1.5`——"cuBLAS's cost grows more slowly with N for float64 than for other dtypes"——这是从数据里看出来的经验规律，不是推导。

第二个 hunk 在 `lu_factor` 里把原来的硬编码分支包进 `#ifdef USE_ROCM`，非 ROCm 路径改为调用新函数：

```cpp
+#ifdef USE_ROCM
+    // FIXME: this heuristic is likely incorrect for ROCM.
     if (m != n || (batch_size == 1 || m >= 512)) {
       lu_factor_looped_cusolver(input, pivots, infos, compute_pivots);
     } else {
       lu_factor_batched_cublas(input, pivots, infos, compute_pivots);
     }
+#else
+    const auto solver_backend = get_lu_factor_solver_backend(batch_size, m, n, input.scalar_type());
```

这就是"+104 −0"的来历：旧逻辑一行没删，只是被限制到 ROCm；ROCm 上没有测数据，就不改，留一条 `FIXME`。这是有意避开的更大改法之一。另一个被避开的更大改法在 review 里出现：reviewer 建议给用户一个显式选择 cuBLAS 的开关（第 6 节）。

diff 里没有任何与 LU 无关的改动，没有格式化，没有顺手重构。一个 reviewer 打开它，第一屏是注释、第二屏是函数、第三屏是接线，十分钟够了。

### 4. 测试与数据

没有新增测试文件。PR 的"数据"部分是 body 里的一张验证表，标题 "LU Backend Heuristic Validation"，摘要是：

```text
Overall accuracy: 3696/3792 (97.5%)
--- Accuracy by GPU ---      a100 98.8% · h100 96.9% · l40 96.5% · rtx5090 97.6%
--- Accuracy by dtype ---    float32 98.3% · float64 96.4% · complex64 98.6% · complex128 96.5%
--- Accuracy by batch size --- batch=2 95.8% … batch=64 99.5%
--- Wrong predictions: 96 cases ---（逐条列出 GPU / dtype / batch / N / 预测 / 实际 / 慢多少倍）
Severity breakdown of wrong predictions:
  < 1.5x slowdown (mild):     93
  1.5-3x slowdown (moderate):  3
  > 3x slowdown (severe):      0
```

这张表的呈现方式值得学：它没有说"快了多少"，因为一个启发式不能用单一加速比描述；它说的是**在多大的采样空间里、选对的比例是多少、选错的时候最坏损失多大**。96 个误判逐条列出，最差 1.81x，没有超过 3x 的。reviewer 据此能判断风险上限，而不必自己跑 benchmark。body 还主动指出弱点："batch=2 is the noisy area -- L40/RTX5090 and H100 tend do disagree there"。

对新贡献者来说，这张表也暴露了这个 PR 真正的成本：4 种 GPU、3792 个点。GitHub 上看不到这些数据是怎么采的，但 issue 开出（4 月 30 日）到 PR 开出（5 月 27 日）之间有 27 天，而 PR 本身只开了 5 天——合理的推断是，大部分时间花在数据上，而不是那 104 行代码上。

### 5. CI

作者在开 PR 后 8 分钟内自己打了 `module: cuda`、`module: cublas`、`module: linear algebra`、`release notes: linalg_frontend` 和 `ciflow/trunk` 五个标签（事件时间 08:40–08:41）；`open source` 标签由 pytorchbot 在 08:46 打上。`ciflow/trunk` 是 `.github/pytorch-probot.yml` 的 `ciflow_push_tags` 之一，效果是除了每个 PR 都跑的 `pull.yml`，再以 push 事件拉起 `trunk.yml`。用最后一个 commit 的 sha 查 workflow 运行：

```bash
gh api "repos/pytorch/pytorch/actions/runs?head_sha=f642e3f2fbe25291ebbd71f8e6ed5b0000374652&per_page=100" \
  --jq '.workflow_runs[] | "\(.name)\t\(.event)\t\(.conclusion)\t\(.path)"'
```

```text
pull                              pull_request  failure   .github/workflows/pull.yml
trunk                             push          success   .github/workflows/trunk.yml
Lint                              pull_request  success   .github/workflows/lint.yml
Lint                              push          success   .github/workflows/lint.yml
TSan                              push          success   .github/workflows/tsan.yml
BC Lint                           pull_request  success   .github/workflows/lint-bc.yml
Check mergeability of ghstack PR  pull_request  success   .github/workflows/check_mergeability_ghstack.yml
```

`gh pr checks 185344` 列出 219 项：178 通过、39 跳过、2 失败。两个失败都是 `pull` 里的 `linux-docs / build-docs-cpp-false` 与 `build-docs-python-false`。Dr. CI 在 PR 顶部的评论把它们归为 "2 New Failures"，附的原因是 `[OSDC] Step script exited with code 127. This is a script/workflow error, not an infrastructure issue.`——退出码 127 是"命令找不到"，与一个只改 `.cpp` 的 PR 没有关系。（写作时这两个任务的日志已过期，无法进一步核对；在 v2.14.0 检出里 docs 构建已从 `pull.yml` 移到 `.github/workflows/docs-build.yml`，`pull.yml` 里只留了一段注释说明。）

这两个失败直接影响了合入。`.github/merge_rules.yaml` 的 "Linear Algebra" 规则要求 `mandatory_checks_name` 为 `EasyCLA`、`Lint`、`pull`——`pull` 里任何一项失败，`@pytorchbot merge` 就会拒绝。第 6 节的时间线里能看到这一步。

### 6. review

review 全部发生在 PR 开出后四小时内的一个窗口里，然后是四天后的一次 ping 与一次批准。

**2026-05-27 12:15–12:20，@IvanYashchuk** 提交一次 review，三条行内评论加一段总评。三条行内都针对新增的注释块：

- 第 869 行附近："This says the rule is based on about 3200 points, while the PR description says about 3800 points and the validation table totals 3792. Can we make these numbers agree…This heuristic is benchmark-derived, so the provenance needs to be easy to trust later."
- 第 900 行附近："What Blackwell? The PR description says 5090 was used, it's also Blackwell."
- 第 902 行附近："The benchmark summary and the rule are useful future-reader context, but `Authored with Claude` does not help explain or maintain the heuristic. It's your code still."

总评："The direction is right. The old `m >= 512` cutoff is too coarse for batched square LU…" 然后提了一个更大的问题："Are you planning to add user-controlled overwrite of this choice? It could probably be done through…preferred_linalg_library adding a cublas option there."

**12:23–12:57，@nikitaved** 逐条回复："New data came in, need to update."（数字）；"Will remove it. All the dirty work is done by me :)"（Claude 那行）；关于 Blackwell 的注释解释了 5090 是面向推理的卡、fp64 性能要看 CUDA 13.2 的模拟。**12:50，@johannesz-codes** 插入一条纠正："It's available from 13.0. Also cuBLAS enables it by default if I understand the docs right."，作者 12:54 接受："Thank you, @johannesz-codes, I will update the comment. Indeed, fp64 is there since 13."，12:57 补一句 "I will double-check the heuristic for such cases then. This is a TODO."

**12:36**，作者在 PR 评论区回应总评里的"用户开关"建议："I am not sure that having cublas as a separate backend is justifiable just yet -- LU, QR, LU_solve, LSTSQ -- the benchmarks are old there, they need a re-eval…" 这是**讨论后维持原样**的一项：reviewer 提出了一个合理的扩展，作者给出不做的理由（要把 cuBLAS 做成正式后端，得先重测所有相关算子），reviewer 没有坚持。

**12:30 与 13:03** 两个 "Update" commit 落地第一轮修改。**2026-05-31 11:32**，作者在 Blackwell 那条讨论下补了验证结果："OK, no need to modify things when testing cusolver/cublas on CUDA 13.2 each with/without fp64 emulation enabled. The fp64 advantage really kicks in at quite large sizes, very much away from the captured decision boundary."，11:40–12:01 三个 "Update" commit，11:41 ping："@IvanYashchuk, could you give it another review pass?"

用 `gh api repos/pytorch/pytorch/compare/ccaaba92...f642e3f2` 取第一个与最后一个 commit 之间该文件的差异，剔除 ghstack rebase 带进来的 main 变化之后，review 导致的代码改动正好三处：`about 3200 points` → `about 3800 points`；删掉 `// Authored with Claude.`；把原来关于 Blackwell 的注释换成 "NOTE: additionally validated on Blackwell CUDA 13.2 with FP64 emulation on/off for cuSOLVER (on by default for cuBLAS). No severe mispredictions observed."——同时把一个注释里的全角破折号换成了 `-`。阈值常数一个没变。

**2026-06-01 14:05，@IvanYashchuk** 批准并评论 `@pytorchbot merge`。14:08 pytorchmergebot："Merge started…ETA 0-4 Hours"。14:13："Merge failed. Reason: 1 mandatory check(s) failed. The first few are: - pull / linux-docs / build-docs-cpp-false"。14:20 作者 `@pytorchbot merge -i`。14:22 bot："Your change will be merged while ignoring the following 2 checks…"。14:23 打上 `Merged` 标签、关闭 PR。

关于 "Authored with Claude" 这一条要多说一句。PyTorch 的 `AI_POLICY.md` 与 `CONTRIBUTING.md` 都不禁止 AI 辅助，要求的是"你对你发出的每一行负责"。reviewer 的意见不是"不能用 Claude"，而是**这句话放在源码注释里没有维护价值**——它不解释启发式、不帮助后人修改；PR 描述末尾的 "Authored with Claude." 则保留了下来，进了 commit message。政策的边界在这个例子里非常具体：声明放描述，代码里只放对维护者有用的东西。

### 7. 合入之后

合入 commit 是 main 上的 `230db5d50ab7`，commit message 由 pytorchmergebot 组装：PR 描述全文 + `Pull Request resolved: https://github.com/pytorch/pytorch/pull/185344` + `Approved by: https://github.com/IvanYashchuk`。第一篇讲的"顺着 `Pull Request resolved:` 找到当时的讨论"，就是指这一行。

进了哪个版本：

```bash
# 在 pytorch 完整克隆里执行（本地 v2.13.0 标签若是浅克隆，merge-base 会失败，改用 GitHub compare API 核对）
git tag --contains 230db5d50ab7181876abd9a5ac5c4aca70c7d79b | grep -E '^v2\.'
gh api repos/pytorch/pytorch/compare/v2.13.0...230db5d50ab7181876abd9a5ac5c4aca70c7d79b --jq '{status, behind_by}'
```

前者列出 `v2.13.0-rc1`（tag 日期 2026-06-10）到 `v2.13.0-rc14`、`v2.14.0-rc1` 到 `v2.14.0`；后者返回 `status: behind, behind_by: 567`，即该 commit 是 v2.13.0 的祖先。v2.13.0 的 tag 打于 2026-07-03，GitHub release 发布于 2026-07-08。从 issue 到用户能 `pip install` 到，69 天。

follow-up 与 revert：`gh pr list --repo pytorch/pytorch --state all --search "185344 in:body,title"` 只返回它自己，没有 revert、没有引用它的后续 PR。`git log 230db5d50ab7..v2.14.0 -- aten/src/ATen/native/cuda/linalg/BatchLinearAlgebra.cpp` 有两个 ROCm 相关 commit（#185557、#188720）碰过这个文件，但 `get_lu_factor_solver_backend` 在 v2.14.0 检出里与合入时逐行一致。作者自己留下的 `FIXME: this heuristic is likely incorrect for ROCM.` 也还在。


## 四、走读二：vLLM #47272

### 1. 起点

这个 PR 的起点比它自己早四个月。issue #35541 "[Bug]: vLLM hangs indefinitely with low `num_gpu_blocks_override`" 由 @kvcache670 于 2026-02-27 开出，同日报告者自己开了一个修复 PR #35542（未合入）。2026-03-31 @vnnm404 补充复现条件（"Increasing the prompt length makes it consistent"）。2026-04-28 maintainer @njhill 的 PR #41069 "[Core] Account for `num_gpu_blocks_override` in `max_model_len` checks" 合入，让启动检查考虑 `num_gpu_blocks_override`——但按总块数检查。issue 此后没有人再评论，2026-06-29 被 stale bot 标记，**2026-07-30 自动关闭**——那时 #47272 已经开了一个月。

PR 的 Purpose 第一句就把这段历史交代清楚："Fixes #35541 (still not fully resolved). That issue was previously addressed by #41069, which made the startup capacity check account for `num_gpu_blocks_override`. However, #41069 validates against the **total** block count…" 然后解释缺口：`BlockPool` 永久保留一个 null block，所以可用块只有 `num_gpu_blocks - 1`；启动检查却拿 `max_model_len` 与总块数比。当块数恰好等于 `ceil(max_model_len / block_size)` 时，检查通过（日志打出 "Maximum concurrency for … tokens per request: 1.00x"），运行时一个请求生成到最后一块时分配失败、被抢占、重新入队、再次失败，引擎永远 0 tok/s。

body 的 "Not a duplicate" 一段列出了 #41069（override-only、按总块数）与 #35542（早期未合入的尝试、只加了 override 路径的检查），说明本 PR 与两者的差别。这一段对应 `AGENTS.md` "Duplicate-work checks" 与 "Accountability" 里"PR descriptions for AI-assisted work must include: Why this is not duplicating an existing PR"的要求；body 末尾一句 "AI assistance (Claude Code) was used to investigate and draft this change; all changed lines were reviewed and the tests were run locally as described above." 满足 "Clear statement that AI assistance was used"，六个 commit 里作者自己的三个（其余是 maintainer 的一个 commit 和两个 merge main）都带 `Co-authored-by: Claude <noreply@anthropic.com>` trailer，满足 `docs/contributing/README.md` "AI Assisted Contributions" 的 "Mark commits" 一条。

### 2. 阅读

从现象到代码，作者需要找到：那条 "Maximum concurrency" 日志在哪打的；启动检查是哪个函数、它拿的块数从哪来；null block 是在哪保留的；`num_gpu_blocks_override` 在哪定义。在 vLLM v0.28.0 检出根目录执行：

```bash
rg -n "Maximum concurrency for" vllm/                       # → vllm/v1/core/kv_cache_utils.py
rg -n "def get_kv_cache_configs|def _check_enough_kv_cache_memory|def check_enough_kv_cache_memory|def _auto_fit_max_model_len|def estimate_max_model_len" vllm/v1/core/kv_cache_utils.py
rg -n "null_block" vllm/v1/core/block_pool.py               # → BlockPool.__init__ 里 popleft 一块并标 is_null
rg -n "def get_usage" vllm/v1/core/block_pool.py            # → 已经在减 1，说明"可用 = 总数 − 1"是既定事实
rg -n "num_gpu_blocks_override" vllm/config/cache.py
rg -ln "num_gpu_blocks_override=3[23]" tests/               # → 哪些测试把池子开得刚好够一个请求
```

路径很短：所有块数来源（override、`kv_cache_memory_bytes`、profiling 得到的显存）都汇入 `get_kv_cache_configs` 的 `available_memory`，检查与 auto-fit 都从它出发。这正是作者在 body 里的判断："Since all sources of the block count funnel into `available_memory`, this covers…with one change"。最后一条 `rg` 在 v0.28.0 检出里能找到 `tests/v1/e2e/general/test_async_scheduling.py` 与 `tests/v1/sample/test_logprobs.py` 用 `num_gpu_blocks_override=32` 配 `max_model_len=512`——v0.28.0 不含本 PR，所以看到的是修改前的样子；这两个文件后来成了 CI 失败的来源（第 5 节）。

### 3. diff

`gh pr diff 47272 --repo vllm-project/vllm` 共 227 行，六个文件，十一个 hunk。核心只在 `vllm/v1/core/kv_cache_utils.py` 的两处。

第一处在 `get_kv_cache_configs`：在把 override 折算成 `available_memory` 之后、auto-fit 与检查之前，插入一段：

```python
+    # Reserve the null block BlockPool permanently holds back, so auto-fit and
+    # the capacity check both plan against usable blocks. Allocation below
+    # still uses the full memory.
+    check_memory = [
+        avail_mem - _pool_bytes_per_block(vllm_config, groups) if groups else avail_mem
+        for groups, avail_mem in zip(projected_groups_per_worker, available_memory)
+    ]
```

然后把 `_auto_fit_max_model_len(...)` 与下面的 `for groups, avail_mem in zip(...)` 检查循环的实参从 `available_memory` 换成 `check_memory`。要点是注释最后一句：**分配仍然用全部内存**，只有"够不够"的判断按可用块算。这是最小改法——不改 `BlockPool`、不改调度器、不改错误信息（复用既有的 "estimated maximum model length is …" ValueError）。

第二处在公开函数 `check_enough_kv_cache_memory`（被 `tests/v1/engine/test_init_error_messaging.py` 使用），做同样的减法：

```python
+        groups = get_kv_cache_groups(vllm_config, dict(kv_cache_spec))
+        check_memory = (
+            available_memory - _pool_bytes_per_block(vllm_config, groups)
+            if groups
+            else available_memory
+        )
         _check_enough_kv_cache_memory(
-            available_memory,
+            check_memory,
```

`dict(kv_cache_spec)` 那个拷贝有注释解释："grouping may unify them in-place"——不拷贝会改到调用方的 spec。这一处是 @njhill 在批准前自己推的 commit（第 6 节）。

其余四个文件全是测试。`tests/v1/core/test_kv_cache_utils.py` 加三个用例、改一个既有用例的内存量；`tests/v1/e2e/general/test_async_scheduling.py` 与 `tests/v1/sample/test_logprobs.py` 把 `num_gpu_blocks_override=32` 改成 33；`tests/v1/e2e/general/test_context_length.py` 把 `kv_cache_bytes` 从 1 MB 改成 2 MB；`tests/v1/engine/test_init_error_messaging.py` 把 `dtype="float16"` 改成 `torch.float16`（njhill 顺手修的类型）。

有意避开的更大改法在 PR 讨论里被明确写出来了。社区成员 @malaiwah 建议把上界收敛到 spec 上的一个方法里、同时处理 lookahead slots 与 mamba align 的额外块，还提议把他自己 PR 里的"请求侧"逻辑并进来；作者 08-18 回复："I'd like to keep this PR's scope small — the startup capacity fix only."

### 4. 测试与数据

新增的三个测试都在 `tests/v1/core/test_kv_cache_utils.py`，每个 docstring 都写了"为什么"：

- `test_kv_cache_reserves_null_block_for_max_model_len`，`@pytest.mark.parametrize("use_override", [True, False])`：32 块池对 512 token 的 `max_model_len`（恰好 32 块），override 路径与内存路径都必须 `pytest.raises(ValueError, match="max seq len")`；
- `test_auto_fit_max_model_len_reserves_null_block`：`original_max_model_len = -1`（auto-fit），内存正好 64 块，断言 auto-fit 定在 `63 * block_size` 而不是 64；
- `test_check_enough_kv_cache_memory_reserves_null_block`：公开函数 32 块拒绝、33 块接受。

这些是不需要 GPU 的单元测试（作者的回复里说明是在 CPU 构建上跑的）；它们属于 `.buildkite/test_areas/misc.yaml` 里 "V1 Core + KV + Metrics" 任务的 `pytest -v -s -m 'not cpu_test' v1/core`。端到端的验证放在 body 的 Test Result 里，而不是新写一个 e2e 测试：复现脚本（`facebook/opt-125m`、`load_format="dummy"`、`block_size=16`、`num_gpu_blocks_override=50`、`max_model_len=800`、770 个 prompt token 生成 40 个）——"before: hangs at 0 tok/s; after: raises at startup with `ValueError: ... max seq len (800) ... estimated maximum model length is 784`"。784 = 49 × 16，数字对得上。

Test Plan 只有一条命令：`pytest tests/v1/core/test_kv_cache_utils.py -k reserves_null_block`——这正是 `.github/PULL_REQUEST_TEMPLATE.md` checklist 里 "The test plan, such as providing test command" 要的形态。没有 benchmark，因为这是正确性修复，不涉及性能路径。

### 5. CI

PR 开出时 github-actions bot 的第一条评论就说明了 vLLM 的 CI 策略："PRs do not trigger a full CI run by default. Once the PR is approved and ready to go, your PR reviewer(s) can run CI…" 从 GitHub 上的记录看，7 月 1 日到 8 月 19 日之间这个 PR 只跑过 `pre-commit`、DCO 与 mergify 的检查；08-19 bot 的 "CI is now available for this PR" 也印证了此前没有 Buildkite 权限。

2026-08-19 22:44 @njhill 打 `ready` 标签、22:45 评论 `/ci run`，bot 回复 "✅ Triggered Buildkite CI #84702 for commit `f9b5892cdad2`"。这一轮**六个任务失败**。作者 08-20 01:11 的评论把失败分析得很完整——三个测试套件各在两种硬件（H200 与 MI250/MI300）上失败，同一根因：

> "three existing tests configure the KV cache pool at exactly the boundary the startup check now rejects. Their requests stay far below `max_model_len`, which is why the pre-existing hang never surfaced in them."

对应三处修改：`test_async_scheduling.py`（e2e-scheduling 任务）与 `test_logprobs.py`（v1-sample-plus-logits 任务）的 `num_gpu_blocks_override` 32 → 33，意图（"pool holds at most a single max-length request"）不变；`test_context_length.py`（e2e-core 任务）的 1 MB 池子只有一块——减去 null block 后**零块可用**，auto-fit 以前能算出 16 个 token，现在被正确地拒绝，改成 2 MB（三块，两块可用）。作者还说："I audited the rest of `tests/` for the same exact-boundary pattern; these three files are the only ones."，并在 CPU 构建上逐个复现了 CI 报的错。

修复 commit `e4fa9878` 之后合并 main，08-20 01:38 作者与 njhill 几乎同时 `/ci run`（bot 对第二个回复 "CI is already running for this commit"），Buildkite #84725 用 1 小时 9 分跑完，`gh pr checks 47272` 列出 88 项全部通过。任务名可以直接对回 `.buildkite/test_areas/`：

```text
Buildkite 任务名（gh pr checks）                         test_areas 文件与 label                          source_file_dependencies 命中
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
nvidia-h200-v1-core-plus-kv-plus-metrics                misc.yaml "V1 Core + KV + Metrics"               vllm/v1/ · tests/v1/core
nvidia-h200-e2e-core · amd-mi250-e2e-core               engine.yaml "e2e Core (1 GPU)"，mirror: amd       vllm/v1/ · tests/v1/e2e/general/
nvidia-h200-e2e-scheduling · amd-mi250-e2e-scheduling   engine.yaml "e2e Scheduling (1 GPU)"，mirror: amd  同上
nvidia-h200-v1-sample-plus-logits · amd-mi300-…         misc.yaml "V1 Sample + Logits"                   vllm/v1/
```

`mirror: amd:` 这一段解释了为什么每个失败都成对出现。此外还有 `DCO`、`pre-commit`、`pre-run-check`、mergify 的 `Summary`、`Meta Internal-Only Changes Check` 与 `add-label-on-auto-merge`。

### 6. review

review 的记录要分两部分看：maintainer 的一条，社区成员的两条。

**2026-08-16 12:02 与 14:17，@malaiwah**（不是 maintainer）在 PR 下留了两条很长的评论。第一条给出在一台 RTX 5090 上、用一个混合 Mamba 模型、只改 `--gpu-memory-utilization` 得到的六行实测表：0.9555 时"starts, then 0 tok/s forever"——正是这个 PR 的边界；0.9585 时"re-prefills forever"——这个 PR 修不了的另一行。他明确指出这些数字来自一个 fork 构建。第二条评论宣布自己开了 #52530 处理"请求侧"，并写道："**This PR owns the capacity side.**…**Please do not merge #52530 instead of this.**" 这是查重规则被正确执行的一个实例：发现相邻问题的人主动划清边界，而不是开一个覆盖两者的竞争 PR。

**2026-08-17 21:34，@njhill** 提交唯一一条 maintainer 行内评论，在 `get_kv_cache_configs` 的检查循环上：

> "The auto-fit path skips the reservation, so the boundary case survives when `max_model_len=-1`: `_auto_fit_max_model_len` binary-searches against the full block count and can settle on a length that needs every block including the null one."

他直接给出了替代代码——把减法提到 auto-fit 分支之前、去掉原来的 `check_memory` 分裂——并要求"Worth a third case in the new test: `max_model_len=-1` with memory landing exactly on the boundary"。这条评论决定了 diff 的最终形态：第 3 节引用的那段 `check_memory = [...]` 就是 reviewer 建议的原文。

**2026-08-18 04:18–05:15，@92hyungjun** 推 commit `f3a4aec1` "Reserve the null block in auto-fit max_model_len too"，回复："Applied as suggested — reservation hoisted above the auto-fit branch, `check_memory` split gone — and added the third test…`test_auto_fit_max_model_len_with_hybrid` needed one extra block of memory, which is the reservation doing its job. Verified end-to-end on a CPU build…79/79 tests pass. Ready for CI." 同时回复 @malaiwah：auto-fit 的情况现在已覆盖，"the cases left for #52530 are the runtime-only ones"，并谢绝了把请求侧并进来的提议——这是**讨论后维持原样**的一项。

**2026-08-19 22:43–22:45，@njhill** 自己推了 commit `c0e82683` "also update check_enough_kv_cache_memory method"（+10 −1 主代码、+25 测试、+3 −2 类型修正），合并 main，打 `ready`，批准并留言："Thanks @92hyungjun! I pushed another small update to a method which is currently only used in the tests."，然后 `/ci run`。maintainer 直接往贡献者的分支推 commit，前提是作者开 PR 时保留了"允许 maintainer 编辑"的选项；两个 merge main 的 commit 也是 njhill 推的。

整个 review 只有一轮实质往返，但它改变了 PR 的范围（覆盖 auto-fit）、增加了一个测试、并由 reviewer 补齐了第二个入口。此外，PR 开出时 `claude[bot]` 留了一条自动 review："This pull request is from a fork — automated review is disabled. A repository maintainer can comment `@claude review` to run a one-time review."——vLLM 在 2026 年中已经把 AI review 接进流程，但对 fork 默认关闭。

### 7. 合入之后

合入 commit `76fb6d210a57`，由 @njhill 于 2026-08-20 02:49 合入（GitHub merge，squash，commit 标题带 `(#47272)`）。

进了哪个版本：

```bash
# 在 vllm 完整克隆里执行
git fetch --tags origin
git tag --contains 76fb6d210a57c7efbb61fa6cb029d1fa1911bfb5
git merge-base --is-ancestor 76fb6d210a57c7efbb61fa6cb029d1fa1911bfb5 v0.28.0; echo $?
gh api repos/vllm-project/vllm/releases --jq '.[] | "\(.tag_name) \(.published_at[:10]) prerelease=\(.prerelease)"' | head -3
```

结果：包含它的 tag 是 `v0.28.1rc0`（2026-08-27）、`v0.29.0rc1`（09-01）到 `v0.29.0rc4`（09-04）；`merge-base --is-ancestor … v0.28.0` 返回 1——**不在 v0.28.0**。v0.28.0 的 tag 打于 08-24、GitHub release 发布于 08-26，但 `git merge-base v0.28.0 origin/main` 落在 08-17 的一个 commit（#52570）上，也就是 v0.28.0 的 release 分支在 8 月 17 日从 main 分出，之后只收 cherry-pick；08-20 合入 main 的东西自然不在里面。`RELEASE.md` 说 "We aim to have a regular release every 2 weeks"，且 "After branch cut … cherry picks are allowed in" 只限于明确的标准。截至 2026-09-07，这个修复没有进任何正式版本；一个 8 月 20 日合入的 bugfix，用户最早在 v0.29.0 正式发布后才能不打补丁地用上。

follow-up 与 revert：

```bash
gh pr list --repo vllm-project/vllm --state all --search "47272 in:body,title" \
  --json number,title,state,createdAt
```

返回三个：#52530（@malaiwah，请求侧，08-20 关闭，关闭留言 "Closing in favor of the maintainer-requested path. #47272 has merged as `76fb6d21…`"）；#48724（@ricky-chaoju，07-15 开出，"[Bugfix][Core] Reserve the null block in auto-fit max_model_len"，body 里写 "This is not a duplicate of #47272: that PR fixes the explicit `max_model_len` validation and its diff keeps `check_memory = available_memory`…"——这在 7 月 15 日是对的，8 月 18 日之后就不成立了；截至 09-07 仍 open，mergify 在 08-20 给它打了冲突提示）；#51156（spec decode 相关，只是提及）。没有 revert。#48724 是一个值得记住的反例：它在开出时确实不重复，但原 PR 在 review 中吸收了它的范围，而它没有跟进——查重不是一次性的动作。


## 五、两个走读的对照表

```text
环节            PyTorch #185344                                           vLLM #47272
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
issue 来源      maintainer 自己开的性能 issue（附复现与实测表）；            用户 bug 报告；被部分修复过一次；PR 开出后 issue 被 stale bot 关闭
                无 actionable 标签，由熟悉该模块的长期贡献者接手
关联方式        body 写 Fixes https://…/issues/181999；                     body 写 Fixes #35541 (still not fully resolved)；
                合入 commit 带 Pull Request resolved: 与 Approved by:       合入 commit 标题带 (#47272)
标题            无前缀，一句话                                              [Bugfix][Core] 前缀（docs/contributing/README.md "PR Title and Classification"）
描述结构        ghstack 头 + 数据表 + cc 列表；无固定模板                    Purpose / Test Plan / Test Result（PULL_REQUEST_TEMPLATE.md）+ Not a duplicate（AGENTS.md）
标签由谁打      作者自己打 module: × 3、release notes:、ciflow/trunk；        mergify 按标题 [Bugfix] 打 bug（.github/mergify.yml label-bug）、v1；
                pytorchbot 打 open source                                  maintainer 打 ready；kv-cache-manager 由 mergify 在合入前一天打上（规则不在 v0.28.0 检出中）
commit 形态     8 个 "Update"，commit 体只有 [ghstack-poisoned]；           4 个有标题的 commit + 2 个 merge main；非 merge commit 都带 Signed-off-by；作者的 3 个带 Co-authored-by: Claude
                CLA 由 EasyCLA 检查                                        DCO 由 probot dco 检查
CI 触发         开 PR 即跑 pull + Lint；ciflow/trunk 标签额外拉起 trunk      开 PR 只跑 pre-commit / DCO；ready 之后 maintainer /ci run
CI 规模         219 项检查、7 个 workflow                                  88 项检查、1 个 Buildkite build，按 source_file_dependencies 选任务，AMD 镜像成对
CI 失败         2 个 docs 任务，脚本错误，与改动无关 → merge -i 忽略          6 个测试任务，改动引入 → 修 3 个测试的配置 → 再跑
review 形态     4 小时内 3 条行内 + 1 条总评，全部关于注释与来源；            47 天沉默 → 社区成员 2 条长评论 → maintainer 1 条带代码的行内意见 + 自己推 1 个 commit
                作者逐条回复并推 commit；4 天后 ping
维持原样        "加 cublas 用户开关"——作者说明其他算子 benchmark 已旧，暂不做   "把请求侧一起修"——作者明确只修启动容量
AI 政策         AI_POLICY.md；描述保留 "Authored with Claude."，            AGENTS.md + docs/contributing/README.md；描述末尾声明 + commit trailer；
                代码注释里的同一句被 reviewer 要求删除                        claude[bot] 对 fork PR 默认不 review
合入命令        @pytorchbot merge → merge_rules.yaml 校验 approved_by 与     maintainer 打 ready → 绿 CI → GitHub merge
                mandatory_checks_name（Linear Algebra：EasyCLA · Lint · pull）
进入版本        v2.13.0（合入 32 天后打 tag，37 天后 release）              不在 v0.28.0；只在 rc 标签中；正式版本待 v0.29.0
```


## 六、核心问题：时间去了哪里

### 1. PyTorch #185344 的时间线

全部时间戳来自 `gh issue view`、`gh api .../issues/N/events`、`/pulls/N/comments`、`/pulls/N/commits` 与 `/issues/N/comments`（UTC）：

```text
时间              事件                                                    间隔          归类
─────────────────────────────────────────────────────────────────────────────────────────────────────────
04-30 08:21       @IvanYashchuk 开 issue #181999，附复现与 5090/6000 Ada 数据   —             起点
                  ↓ 采集 4 种 GPU × 3792 个点的 benchmark、拟合阈值（推断）      27 天         作者工作（GitHub 不可见）
05-27 08:33       PR 开出（ghstack）；请求 eqy / syed-ahmed / Aidyn-A / IvanYashchuk  —      —
05-27 08:40–08:46 作者打 5 个标签；pytorchbot 打 open source；pull/Lint/trunk 起跑  13 分钟    作者工作
05-27 08:50、08:56 两个 Update commit（开出后 23 分钟内）                    —             作者工作
05-27 12:15–12:20 @IvanYashchuk 3 条行内 + 总评                              3 小时 42 分  等 review
05-27 12:23–12:57 作者逐条回复；@johannesz-codes 纠正 CUDA 版本                42 分钟       往返
05-27 12:30、13:03 两个 Update commit：3200→3800、删 Claude 句、改 Blackwell 注释  —          作者工作
05-27 13:03 → 05-31 11:32  作者在 CUDA 13.2 上补做 fp64 模拟开/关的验证        3 天 22 小时  作者工作（reviewer 提出的 TODO）
05-31 11:32–12:01 补验证结论评论；3 个 Update commit；ping reviewer            29 分钟       往返
05-31 12:01 → 06-01 14:05  等 reviewer 二次 review                          1 天 2 小时   等 review
06-01 14:05       @IvanYashchuk 批准，@pytorchbot merge                        —             —
06-01 14:13       merge 失败：pull / linux-docs 两项                          8 分钟        CI
06-01 14:20–14:23 作者 @pytorchbot merge -i；bot 合入                          10 分钟       CI / 往返
```

PR 阶段总计 5 天 5 小时 50 分。分解：**等 review** 约 1 天 6 小时（两段，都在 PyTorch "4 个工作日"的承诺之内，实际远快于承诺）；**往返** 约 1.5 小时；**CI** 约 20 分钟（不算 workflow 本身的运行时间，那与作者无关）；**作者工作** 约 4 天——其中 3 天 22 小时是 reviewer 一条评论引出的补充验证。加上 PR 之前的 27 天（推断为 benchmark 采集），按上表归类，从 issue 到合入的 32 天里约 31 天是作者自己在工作，等待与往返合计不到一天半。

**哪些可以省**：几乎没有。三条行内意见都是作者提交前自己核对一遍就能避免的（注释与描述的数字对齐、注释里不放与维护无关的话、说清 "Blackwell" 指哪块卡），但它们合计只耗了 42 分钟往返。CI 的两个 docs 失败不是作者的，8 分钟识别、10 分钟处理。**哪些是正常成本**：27 天的 benchmark 是这类"性能启发式"PR 的本体——没有 3792 个点就没有那张表，没有那张表 reviewer 就无法在不自己跑数据的情况下批准；4 天的 fp64 模拟验证是 reviewer 用一条评论买到的额外置信度，换来的是阈值常数一个都不用改。

### 2. vLLM #47272 的时间线

```text
时间              事件                                                    间隔          归类
─────────────────────────────────────────────────────────────────────────────────────────────────────────
02-27 19:45       @kvcache670 开 issue #35541；同日开 PR #35542（未合入）        —             起点（他人）
04-28 22:44       @njhill 的 #41069 合入，部分修复                            60 天         他人
07-01 08:51       PR 开出；mergify 打 v1、bug；github-actions 提示 CI 不自动跑     63 天         作者工作（不可见）
07-01 09:01、10:02 两次 force-push；10:08 请求 9 位 reviewer                   77 分钟       作者工作
07-12 07:02       mergify：冲突，打 needs-rebase                              11 天         等 review
07-21 10:04–11:00 作者 force-push 解决冲突；needs-rebase 移除                  9 天          等 review（期间作者只做了 rebase）
07-30 02:16       issue #35541 被 stale bot 自动关闭                          —             —
08-16 12:02、14:17 @malaiwah 两条评论：5090 实测表、开 #52530 并划定边界          26 天         等 review
08-17 21:34       @njhill 行内评论：auto-fit 未覆盖，给出代码，要求第三个测试      1 天 7 小时   等 review 结束（距开 PR 47 天 13 小时）
08-18 04:18–05:15 作者推 f3a4aec1、force-push、回复 njhill 与 malaiwah          7 小时        作者工作 + 往返
08-19 21:55–22:45 mergify 打 kv-cache-manager；njhill 推 c0e82683、合并 main、   1 天 17 小时  等 review
                  打 ready、批准、/ci run → Buildkite #84702
08-19 22:45 → 23:46  Buildkite #84702 报 6 项失败；作者定位到 3 个测试，推 e4fa9878     1 小时      CI + 作者工作
                  "Fix CI tests that sized the KV cache pool exactly to max_model_len"
08-20 01:11       作者评论：逐个解释 6 个失败的根因与修法，并说明已审计整个 tests/    1 小时 25 分  作者工作
08-20 01:38       合并 main；作者与 njhill 同时 /ci run → Buildkite #84725       27 分钟       往返
08-20 02:49       88 项全绿（build 用时 1 小时 10 分）；njhill 合入                1 小时 11 分  CI
```

PR 阶段总计 49 天 18 小时。分解：**等 review** 约 47 天 13 小时（07-01 → 08-17，其中 9 天在等一次 rebase）加合入前的 1 天 17 小时，合计约 49 天；**往返** 约 8 小时（08-18 的回复与 08-20 的 `/ci run`）；**CI** 约 4 小时（两个 Buildkite build 加定位与解释失败）；**作者工作** 在 PR 开出后不到 1 天（首日 77 分钟、08-18 数小时、08-19 夜里修测试）。

**哪些可以省**：47 天的沉默里，作者一次也没有 ping。`docs/contributing/README.md` 写着 "If the PR is not reviewed within 7 days, please feel free to ping the reviewer or the vLLM team."——第 8 天、第 15 天各留一句话，不保证有效，但把主动权留在自己手里。请求 9 位 reviewer 的做法反而稀释了责任：没有一个人觉得"这是我的"。另一处可省的是 6 个 CI 失败——`rg -ln "num_gpu_blocks_override=3[23]" tests/` 在提交前就能找到那两个把池子开到边界上的测试；第三个（1 MB 池子）不容易预见，但前两个可以。这两项合起来，PR 的历时可能从 50 天压到两三周。**哪些是正常成本**：CI 不自动跑，是 vLLM 用 GPU 预算换来的规则，必须等 `ready`；maintainer 的一条评论把范围扩大到 auto-fit，是正确的技术判断，7 小时的响应是合理的；两轮 Buildkite 各一个多小时是硬件矩阵的固定开销；合入后错过 v0.28.0，是两周一版 + 提前切分支的节奏决定的，与作者无关。

还有一段"不可见"的时间值得单独说：PR 开出前的 63 天（从 #41069 合入到 #47272 开出）与 PyTorch 那边的 27 天性质不同。PyTorch 作者在采数据；vLLM 作者在这 63 天里的工作无法从 GitHub 看到——可能是遇到 bug、定位、写复现，也可能只是在 7 月 1 日才碰到它。走读能算出来的只有 GitHub 上留下痕迹的时间，这也是贡献日志存在的理由之一：只有作者自己记，才知道时间真正去了哪里。

### 3. 两张时间线并排

```text
                      PyTorch #185344            vLLM #47272
─────────────────────────────────────────────────────────────────────────────
issue → PR 开出        27 天（作者采 benchmark）    124 天（他人部分修复 60 天 + 空白 63 天）
PR 历时                5 天 6 小时                 49 天 18 小时
其中：等 review        ≈ 1 天 6 小时（2 段）        ≈ 49 天（1 段 47 天 + 1 段 2 天）
      往返             ≈ 1.5 小时                  ≈ 8 小时
      CI 处理          ≈ 20 分钟                   ≈ 4 小时
      作者工作         ≈ 4 天（含 reviewer 引出的验证） < 1 天
review 往返轮数        1 轮（3 条意见 + 1 条 TODO）   1 轮（1 条意见 + reviewer 自己补 1 个 commit）
可省的部分（估计）      ≈ 42 分钟（提交前自查注释）    可能 30–40 天（按时 ping）+ 1 轮 CI（提交前 rg 测试）
合入 → 正式版本        32 天打 tag、37 天发布（v2.13.0）  > 18 天，截至 09-07 未发布
```

结论很直接：**小 PR 的时间不在写代码上**。PyTorch 那个的时间在数据上，这是这类改动的本体成本；vLLM 那个的时间在等待上，其中大半是可以用一句 ping 缩短的。两个项目的 review 承诺（PyTorch "4 个工作日"、vLLM "2–3 天给状态、7 天可 ping"）在这两个样本里一个远快于承诺（3 小时 42 分）、一个远慢于承诺（47 天）——承诺是平均值，个案要靠自己推进。


## 七、映射回前三篇

两个走读里出现的每一个环节，都能在前三篇找到对应的方法与文件：

```text
走读中的环节                                     前三篇的对应小节                                        本文出处
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
从 torch.linalg.lu_factor_ex 追到 lu_factor        第一篇 · 入口点与符号追踪（native_functions.yaml → stub → REGISTER_CUDA_DISPATCH）  三.2
从 "Maximum concurrency" 日志追到 get_kv_cache_configs  第一篇 · 从一个符号追到底（rg 从日志字符串反查）        四.2
test_linalg_lu_family / tests/v1/core 当规格说明     第一篇 · 用测试当文档                                    三.2 · 四.4
Pull Request resolved: 与 Approved by:              第一篇 · git log 即注释                                  三.7
git tag --contains · RELEASE.md 的分支切点          第一篇 · changelog 与 release note                       三.7 · 四.7
issue #181999 的标签与 actionable 的缺席            第二篇 · 标签体系（labeler.yml · actionable 的含义）        三.1
issue #35541 的 stale 关闭与部分修复的历史           第二篇 · 查重（gh issue view --comments · gh pr list --search）  四.1 · 四.7
#52530 与 #48724：相邻问题的边界与过期的"不重复"声明   第二篇 · 先讨论再动手 · 查重不是一次性的                    四.6 · 四.7
+104 −0，旧逻辑只限到 ROCm；只修启动容量不碰调度器     第三篇 · 最小 diff                                       三.3 · 四.3
3792 点命中率表 · 复现脚本前后对比                    第三篇 · benchmark 数据 · 测试                           三.4 · 四.4
ghstack 的 8 个 Update commit                      第三篇 · ghstack                                        三.5
Signed-off-by · Co-authored-by: Claude              第三篇 · DCO 与签名 · AI 辅助政策                          四.1
无前缀标题 vs [Bugfix][Core]；无模板 vs 三段式 + Not a duplicate  第三篇 · PR 描述规范                          五
ciflow/trunk · pull.yml / trunk.yml / lint.yml       第三篇 · CI 矩阵（pytorch-probot.yml 的 ciflow_push_tags）  三.5
source_file_dependencies · mirror: amd · /ci run     第三篇 · CI 矩阵（.buildkite/test_areas/*.yaml）          四.5
Dr. CI 的 "New Failures" 与退出码 127                 第三篇 · 读 CI 日志（CONTRIBUTING.md "CI failure tips"）   三.5
6 个失败全是自己引入的边界测试                        第三篇 · 读 CI 日志 · 测试                                四.5
"Authored with Claude" 该放哪                       第三篇 · AI 辅助政策（AI_POLICY.md · CONTRIBUTING.md）      三.6
47 天没 ping                                       第三篇 · review 往返的时间承诺（docs/contributing/README.md）  六.2
拒绝"加 cublas 开关"与"并入请求侧"                    第三篇 · 什么时候争辩、什么时候照做                         三.6 · 四.6
@pytorchbot merge → merge -i；merge_rules.yaml       第三篇 · merge 机制                                      三.6
ready 标签 → /ci run → 合入；mergify needs-rebase    第三篇 · merge 机制                                      四.5 · 四.6
```


## 八、贡献日志：复盘模板

本篇给贡献日志加最后一页：按走读的格式复盘自己在第三篇提交的那个 PR。无论它合入了、还在等、还是关掉了，都按同样的表填。模板里的命令以自己的仓库与编号代入；时间戳统一从 `gh` 取，不要凭记忆。

```markdown
# 复盘：<OWNER/REPO> #<N> — <标题>

## 0. 事实
| 项 | 值 | 来源 |
|---|---|---|
| 链接 / 作者 / 状态 | | gh pr view N --json url,author,state |
| createdAt / mergedAt（或 closedAt） | | gh pr view N --json createdAt,mergedAt,closedAt |
| 规模（文件数 / +/-） | | gh pr view N --json files,additions,deletions |
| 关联 issue / RFC | | body 中的 Fixes # |
| 标签（谁打的、何时） | | gh api repos/O/R/issues/N/events |
| 进入版本 | | git tag --contains <sha>；gh api repos/O/R/releases |

## 1. 起点
- issue 开出时间、报告者、标签；我动手前 issue 下已有的讨论；有没有被部分修复过
- 我在 issue 里留过什么、maintainer 回过什么（引用原文，注日期）

## 2. 阅读
- 我必须读懂的文件（≤ 5 个）与找到它们的 rg 命令（写出在哪个目录执行）
- 我读错或漏读了什么（事后从 review 意见反推）

## 3. diff
- 每个 hunk 一句话：改了什么、为什么
- 有意避开的更大改法，以及是谁提出、我为什么不做

## 4. 测试与数据
- 新增 / 修改的测试文件与用例名；本地运行命令与结果
- benchmark 或前后对比：硬件、shape、方法、数字；呈现在描述的哪一段

## 5. CI
- 触发了哪些 workflow / Buildkite 任务（gh pr checks N）
- 每次失败：任务名、原因、是我的还是 main 的、怎么处理、花了多久

## 6. review
| 时间 | 谁 | 意见（≤ 2 行原文） | 我的回应 | 结果（采纳 / 讨论后维持 / 待办） |
|---|---|---|---|---|

## 7. 合入之后
- 进了哪个版本；follow-up / revert（gh pr list --search "N"）

## 8. 时间去了哪里
| 阶段 | 起止 | 时长 | 归类（等 review / 往返 / CI / 我的工作） | 可省？怎么省 |
|---|---|---|---|---|
（PR 之前的时间也算：从我决定做到开 PR）

## 9. 下次省什么
- 三条以内，每条对应上表一行
```

两个走读填出来的"下次省什么"作为示例：PyTorch #185344——提交前用 PR 描述里的数字校对一遍代码注释；把与维护无关的署名句只留在描述里。vLLM #47272——第 8 天 ping 一次，只请求一两位与文件路径匹配的 reviewer；提交前 `rg` 一遍 `tests/` 里踩在新边界上的配置；PR 开出后每隔两周重跑一次查重命令，看有没有相邻 PR 出现。


## 九、本文小结

### 1. 要点回顾

```text
选 PR         PyTorch 的合入是 pytorchbot 关闭 PR + 推 commit，要查 label:Merged 而不是 is:merged；
              候选按 Fixes # · /reviews 后有新 commit · 流程要素 · 类型互补逐条核对；放宽的标准要写明
七阶段        起点 · 阅读 · diff · 测试与数据 · CI · review · 合入之后，每阶段一条 gh / git 命令作为出处
PyTorch       issue 由 maintainer 开、长期贡献者接；27 天采 3792 个点；PR 5 天：4 小时内 3 条注释类意见、
#185344       1 条"更大改法"被婉拒、reviewer 引出 4 天补验证；merge 被 pull 里两个无关 docs 失败挡住 → merge -i；进 v2.13.0
vLLM          issue 部分修复过一次并被 stale 关闭；PR 47 天无 review、作者未 ping；社区成员划边界开配套 PR；
#47272        maintainer 1 条带代码的意见扩大范围 + 自己补 1 个 commit；ready → /ci run → 6 个自己引入的失败 → 修测试 → 合入；不在 v0.28.0
时间          小 PR 的时间不在写代码：PyTorch 在数据（正常成本），vLLM 在等待（大半可省）
两条经验      注释与描述的数字对齐、署名放描述不放代码；第 8 天 ping、提交前 rg 边界测试、查重要重复做
```

### 2. PyTorch vs vLLM：同一环节两种做法

```text
环节            PyTorch                                         vLLM
──────────────────────────────────────────────────────────────────────────────────────────────────────────
查已合入 PR     gh pr list --state closed --search "label:Merged"   gh pr list --state merged
合入痕迹        commit 含 Pull Request resolved: / Approved by:      commit 标题含 (#N)；GitHub 显示 merged
标签            作者可自己打 module: 与 ciflow/；bot 打 open source   mergify 按标题打；ready 只有 maintainer 打
签名            EasyCLA                                            DCO probot；Signed-off-by 每个 commit
CI 何时跑       开 PR 即跑 pull / Lint；ciflow/ 标签加跑             pre-commit / DCO 即跑；测试任务等 ready + /ci run
CI 失败归属     Dr. CI 分 New Failures / flaky / broken trunk       Buildkite 页面；作者自己解释每个失败
合入            @pytorchbot merge；merge_rules.yaml；-i 可忽略        ready + 绿 CI + maintainer 点合入；mergify 管 rebase
进版本          按固定日程切分支，rc 十几个；合入到 release 约一个月   两周一版，切分支后 main 上的合入要等下一版
```

### 3. 本篇涉及的文件位置

```text
项目 / 版本         路径                                                    与走读的关系
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
PyTorch v2.14.0     aten/src/ATen/native/cuda/linalg/BatchLinearAlgebra.cpp  get_lu_factor_solver_backend · lu_factor · REGISTER_CUDA_DISPATCH(lu_factor_stub)
                    aten/src/ATen/native/BatchLinearAlgebra.cpp              TORCH_IMPL_FUNC(linalg_lu_factor_ex_out) · lu_factor_stub
                    aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp · BatchLinearAlgebraLibBlas.cpp  lu_factor_looped_cusolver · lu_factor_batched_cublas
                    aten/src/ATen/native/native_functions.yaml               linalg_lu_factor_ex
                    aten/src/ATen/Context.h · torch/backends/cuda/__init__.py  linalgPreferredBackend · preferred_linalg_library
                    test/test_linalg.py                                     test_linalg_lu_family · test_linalg_lu_solve
                    .github/merge_rules.yaml                                "Linear Algebra"：patterns · approved_by · mandatory_checks_name
                    .github/pytorch-probot.yml                              ciflow_push_tags 含 ciflow/trunk
                    .github/workflows/{pull,trunk,lint,lint-bc,tsan,check_mergeability_ghstack,docs-build}.yml  本 PR 触发的 workflow
                    .github/PULL_REQUEST_TEMPLATE/fix_issue.md              Fixes # · "Included benchmark results (for PRs impacting perf)"
                    CONTRIBUTING.md                                         AI-Assisted Development · Merging your Change · CI failure tips
                    AI_POLICY.md                                            "We do not accept contributions created by fully autonomous agents"
                    RELEASE.md                                              Release Cadence · Cherry Picking Fixes
vLLM v0.28.0        vllm/v1/core/kv_cache_utils.py                          get_kv_cache_configs · check_enough_kv_cache_memory · _auto_fit_max_model_len · _pool_bytes_per_block
                    vllm/v1/core/block_pool.py                              BlockPool.__init__ 的 null_block · get_usage
                    vllm/config/cache.py                                    num_gpu_blocks_override
                    tests/v1/core/test_kv_cache_utils.py                    本 PR 新增用例所在文件
                    tests/v1/e2e/general/{test_async_scheduling,test_context_length}.py · tests/v1/sample/test_logprobs.py  被修正的三个测试（v0.28.0 中仍为修改前）
                    .buildkite/test_areas/misc.yaml · engine.yaml           "V1 Core + KV + Metrics" · "e2e Core (1 GPU)" · "e2e Scheduling (1 GPU)" · "V1 Sample + Logits"；source_file_dependencies · mirror: amd
                    .github/PULL_REQUEST_TEMPLATE.md                        Purpose / Test Plan / Test Result 与 checklist
                    .github/mergify.yml                                     label-bug · needs-rebase 两条规则
                    AGENTS.md                                               Duplicate-work checks · Accountability
                    docs/contributing/README.md                             DCO and Signed-off-by · AI Assisted Contributions · PR Title and Classification · 2–3 天 / 7 天 · /ci run
                    RELEASE.md                                              "every 2 weeks" · Release Branch · Cherry-Pick Criteria
GitHub（09-07 查）  pytorch/pytorch#185344 · #181999                        PR · issue
                    vllm-project/vllm#47272 · #35541 · #41069 · #52530 · #48724  PR · issue · 前置修复 · 两个相邻 PR
```


## 十、系列总结

四篇文章从"面对一个百万行的开源项目，如何找到切入点、做出一个能被合入的改动"这个问题出发，到两个真实 PR 的时间线结束。回头看，读者手上应当有三样东西。

**一份贡献日志**。它有四页：项目地图（目录职责、构建命令、测试入口、grep 模式、追过的符号路径）、切入点清单（六个候选与查重结果）、PR 草稿与往返记录（diff 摘要、测试、数据、描述、每轮 review）、复盘（本篇的模板）。它不追求好看，只要求每一项都有出处——一条命令、一个链接、一段日志。这四页填完，就是一次完整贡献的全部证据链。

**两套流程的对照**。每篇一张 PyTorch vs vLLM 的"同一环节两种做法"表，四张合起来覆盖了从读代码到进版本的每一步。两个项目是两种典型：治理成熟、流程厚重、CI 全跑、bot 合入的框架；两周一版、按需跑 CI、maintainer 手动合入、规则还在变的引擎。读任何一个新项目的 `CONTRIBUTING.md`，都可以先问"它更像哪一个、差在哪几格"。

**一种看 PR 的眼光**。本篇用七个阶段拆开了两个 PR，每个阶段都有一条命令能取到事实。这套方法不只用于复盘自己的 PR——打开任何一个已合入的 PR，用同样的七条命令走一遍，就能在一小时内学到这个项目的 reviewer 在守什么、CI 在拦什么、时间通常花在哪。它是学习"maintainer 怎么想"的最短路径。

三条贯穿全系列的线，各自的终点：

```text
方法线   有目标的阅读 → 有依据的选题 → 可验证的改动 → 完整流程的复盘
         第一篇的符号追踪在本篇变成 rg 五连（lu_factor_ex → lu_factor；日志字符串 → get_kv_cache_configs）
         第二篇的查重在本篇变成 #52530 主动划界、#48724 的"不重复"声明过期
         第三篇的最小 diff / 数据 / CI / review 在本篇变成 +104 −0 与 3792 个点、6 个自己引入的失败、"It's your code still."
工具线   ripgrep / clangd / git log → gh CLI / 标签 → lintrunner / pre-commit / CI 配置 → PR 页面与 CI 日志
         本篇全部材料来自 gh pr view / diff / checks、gh api …/reviews /comments /commits /events /actions/runs、
         git tag --contains 与 compare API；没有一条引用来自记忆
项目线   PyTorch 与 vLLM 逐篇对照
         本篇的对照落到两个具体编号：#185344 的 ghstack · module: · ciflow/trunk · merge_rules.yaml · @pytorchbot merge -i；
         #47272 的 [Bugfix][Core] · Purpose / Test Plan / Test Result · Signed-off-by · test_areas · ready · /ci run
```

总纲提出的三种能力，现在可以逐条对照：

1. **阅读能力**：进入一个陌生的百万行项目，在有限时间内定位到与问题相关的代码、测试和历史——第一篇的方法，本篇两个 PR 的"阅读"一节各用五条 `rg` 命令把它做了一遍。
2. **判断能力**：从大量 issue、RFC 和 CI 信号里识别出上游真正需要、自己能做、不与他人重复的工作——第二篇的方法，本篇的两个起点分别展示了"maintainer 自己想做的题会被熟人接走"和"被部分修复又被 stale 关闭的 issue 仍然有缺口"，以及查重必须重复做。
3. **交付能力**：以目标项目的规范完成一个改动——diff、测试、数据、描述、CI、review——并把它合入上游——第三篇的规则，本篇用两张时间线量出了每条规则的成本：数据是正常成本，等待大半可省，review 意见的价值在于它引出的验证而不是改动的行数。

这套能力不属于任何一层，却决定了每一层的技术能力最终能否转化为对项目的实际贡献。两个 PR 的作者都不是在写最难的代码；他们做对的是在正确的地方放正确的东西，然后在正确的时间推进。

**系列目录**

- 总纲：[AI-Infra 开源贡献指南](/contributing-to-ai-infra-open-source.html)

1. [读懂一个百万行的代码库](/reading-a-million-line-codebase.html)
2. [找到切入点：从 issue、RFC 到性能回归](/finding-your-entry-point-in-open-source.html)
3. [做出一个能被合入的改动](/landing-a-mergeable-change.html)
4. [两个真实 PR 的完整走读：PyTorch 与 vLLM](/two-real-prs-pytorch-and-vllm.html)
