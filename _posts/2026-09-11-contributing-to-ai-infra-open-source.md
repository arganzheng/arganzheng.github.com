---
layout: post
title: AI-Infra 开源贡献指南（总纲）
subtitle: A Guide to Contributing to AI-Infra Open Source Projects
tags: [Open Source, PyTorch, vLLM, AI, AI-Infra]
catalog: true
---


## 内容简介

《AI-Infra 开源贡献指南》是一组共四篇的系列文章，面向已经能读懂一部分源码、想把自己的第一个改动合入 PyTorch、vLLM 这类大型 AI-Infra 项目的工程师。它不讲任何一层的技术原理，只讲一件事：**如何进入并参与建设一个百万行的开源项目**。

它回答的问题是：

> **面对一个百万行的开源项目，如何找到切入点、做出一个能被合入的改动？**

这个问题在技术上并不深，却拦住了很多有能力的人。一个能写出高质量 kernel 的工程师，可能在 PyTorch 的 `aten/` 目录前不知从何读起；一个能修 bug 的人，可能不知道 vLLM 的 CI 为什么没有跑、PR 为什么三周没人看。这些障碍与 CUDA 或 Python 无关，与**项目的运作方式**有关：

```text
代码怎么组织     → 目录地图、入口点、符号追踪、测试即文档、历史即注释
工作从哪里来     → issue 标签、RFC、roadmap、CI 失败、性能回归、文档与类型缺口
改动怎么被接受   → 最小 diff、测试、benchmark、PR 描述、CI 矩阵、review 往返
人怎么打交道     → maintainer 的时间预算、review 文化、被拒后怎么办、AI 辅助的项目政策
```

系列以 PyTorch 和 vLLM 为主要样本，因为它们是当前 AI-Infra 最活跃的两个开源项目，也是两种典型风格的代表：PyTorch 是一个有十年历史、治理成熟、流程厚重的框架；vLLM 是一个两周发一版、两三天回复一次 review、规则还在快速演化的引擎。同一个环节，两个项目的做法往往不同：

```text
环节            PyTorch                                     vLLM
贡献文档        CONTRIBUTING.md（技术）+ GitHub wiki（流程）   docs/contributing/README.md + AGENTS.md
大改动的入口    pytorch/rfcs 仓库                            [RFC] issue 模板；>500 行无 RFC 会被标 rfc-required
本地检查        lintrunner（.lintrunner.toml），spin lint      pre-commit（.pre-commit-config.yaml）
PR 描述         三种模板；修 issue 必须写 Fixes #              Purpose / Test Plan / Test Result；标题带 [Kernel] 等前缀
签名            CLA                                          DCO，每个 commit 带 Signed-off-by
CI              GitHub Actions，一百多个 workflow，ciflow/ 标签  Buildkite，.buildkite/test_areas/ 按领域触发，/ci run
合入            @pytorchbot merge，merge_rules.yaml 定权限     maintainer 打 ready 标签，mergify 辅助 rebase
堆叠 PR         ghstack                                      无专门工具
```

学会在这两个项目里做贡献，迁移到 NCCL、Megatron-LM、FlashAttention、Triton、SGLang 的成本很低——它们的流程都是这两种风格的变体。

第四篇会选取两个已经合入的真实 PR，各自从 issue 到合入完整走一遍，把前三篇讲的方法落到具体的 diff、CI 日志和 review 对话上。


## 为什么写这个系列？

### 技术能力和贡献能力是两回事

AI-Infra 的学习路径最终指向一个目标：能够阅读、修改和贡献核心项目。前两步在每一层的技术材料里都有覆盖，第三步却几乎没有人系统地讲。结果是很多工程师停在"能在本地改出来"和"能合进上游"之间：

- 花一周写了一个 kernel，提交后发现三个月前已经有人开了 PR；
- 改动是对的，但 PR 一次改了 40 个文件，reviewer 不知道从哪看起，搁置；
- 测试过了，但没有 benchmark 数据，reviewer 问"快了多少"，答不上来；
- CI 红了，不知道是自己的问题还是 main 本来就是红的；
- 提了 issue 描述一个新功能，附上一大段方案，被 maintainer 直接关闭。

这些失败与技术水平无关，是不知道项目的规则。规则大部分写在仓库里——`CONTRIBUTING.md`、PR 模板、CI 配置、merge 规则——但散落各处，没有人告诉你先看哪个。

### 大型代码库需要一套专门的阅读方法

PyTorch 的源码树有几百万行，`test/` 目录下有两百多个条目；vLLM 小一些，也远超一个人能通读的规模。面对这种体量，"从头读"和"随便点开一个文件读"都是错的。有效的读法是有目标的：从一个符号、一个报错、一个 issue 出发，用工具追到底，只读路径上的东西。这套方法需要练习，也需要知道每个项目自己提供了哪些辅助：PyTorch `CONTRIBUTING.md` 里的 "Codebase structure" 一节、`torchgen` 生成的代码在哪里、vLLM 的 `docs/contributing/` 目录、两个项目的测试如何按目录组织。

### 每个项目的规则都在变，但变化有规律

近两年这些项目的贡献规则变化很快，最显著的是对 AI 辅助贡献的态度。PyTorch 的 `CONTRIBUTING.md` 新增了 "AI-Assisted Development" 一节，要求新贡献者的 PR 必须对应一个带 `actionable` 标签的 issue；vLLM 在仓库根目录放了 `AGENTS.md`，明确"纯 agent PR 不允许"、"违反可能被自动封禁"。同时 CI 也在变：vLLM 的 `.buildkite/test-pipeline.yaml` 在 2026 年初被拆分成 `.buildkite/test_areas/` 下按领域组织的多个文件；PyTorch 在 `lintrunner` 之上加了 `spin lint` / `spin fixlint` 这层包装，PR 模板的 checklist 里写的已经是后者。

细节会继续变，但背后的逻辑稳定：maintainer 的 review 时间是项目最稀缺的资源，所有规则都是为了保护它。理解了这一点，就能在规则变化后自己推导出新的做法。本系列讲规则，更讲规则背后的逻辑。

### 现有材料的断层

- **项目自己的贡献文档**是权威来源，但它们假设读者已经知道开源怎么运作，只讲本项目的特殊之处，而且分散在 `CONTRIBUTING.md`、wiki、issue 模板、CI 配置里；
- **通用的开源贡献指南**讲 fork、branch、PR 的基本操作，不涉及百万行项目的阅读方法，也不涉及 benchmark 数据、CI 硬件矩阵这些 AI-Infra 特有的要求；
- **"我的第一个 PyTorch PR"类博客**记录了一次经历，缺少可迁移的方法。

本系列想填补的是从"会用 git 和 GitHub"到"能在 PyTorch 或 vLLM 里独立完成一次从 issue 到合入的全流程"之间的那段路。


## 适合哪些读者？

### 已经掌握某一层技术、准备做第一次贡献的工程师

你已经能读懂 PyTorch 或 vLLM 的某一部分源码——也许是一类算子、调度器、某个通信路径——并且在本地改出过能跑的东西。你想知道的是：**怎么把它变成一个上游愿意合入的 PR**。这是本系列的主要读者。

### 已经提过 PR 但没有合入、或者合入过程很痛苦的开发者

你的 PR 被搁置、被要求拆分、被要求补 benchmark、或者在 review 往返中耗尽了耐心。本系列的第三篇专门讨论这些情况：哪些是可以避免的，哪些是正常的成本，被拒之后怎么办。

### 需要在团队内推动"向上游贡献"的技术负责人

你的团队维护着一份 PyTorch 或 vLLM 的内部 fork，每次升级都要重新打补丁。把改动推回上游是唯一的长期解法，但团队成员不知道从哪开始。本系列可以作为团队的入门材料，第四篇的两个走读可以直接用作范例。


## 系列的整体主线

四篇文章按一次贡献的自然顺序推进：先读懂、再找到、再做出、最后看两个完整的实例：

```text
第一篇：读懂一个百万行的代码库 —— 目录地图、入口点、符号追踪、测试与历史
        ↓
第二篇：找到切入点 —— issue、RFC、roadmap、CI 失败、性能回归、缺口
        ↓
第三篇：做出一个能被合入的改动 —— diff、测试、benchmark、PR、CI、review
        ↓
第四篇：两个真实 PR 的完整走读 —— PyTorch 一个，vLLM 一个
```

三条交织的线索：

```text
方法线：有目标的阅读 → 有依据的选题 → 可验证的改动 → 完整流程的复盘
工具线：ripgrep/clangd/git log → gh CLI/标签/项目看板 → lintrunner/pre-commit/CI 配置 → PR 页面与 CI 日志
项目线：PyTorch 与 vLLM 各自的目录、文档、标签、CI、review 规则，逐篇对照
```

前三篇每篇都同时用 PyTorch 和 vLLM 做例子，把两个项目在同一环节的做法并排放在一起；第四篇把三篇讲的东西在两个真实 PR 上串起来。每一篇都有同样的四段结构：

```text
问题        一个贡献者在这个环节典型的失败方式
方法        与项目无关的通用做法，以及它背后的理由（几乎总是"节省 reviewer 的时间"）
两个项目    PyTorch 和 vLLM 各自的文件、规则与工具，给出真实路径
贡献日志    读者在自己的日志里应该新增哪一页
```


## 章节结构与分章导读

### 1. 读懂一个百万行的代码库

第一篇讲阅读方法。它不假设读者要读哪一层的代码，只讲面对任何一个大型项目时应该做的几件事，以及 PyTorch 和 vLLM 各自提供了什么辅助。

这一篇会覆盖：

- 先画目录地图再读代码：PyTorch 的 `c10/` → `aten/` → `torch/csrc/` → `torch/` 分层，`torchgen/` 与 `native_functions.yaml` 生成的代码在哪里；vLLM 的 `vllm/` 与 `csrc/` 的分工，`vllm/v1/` 引擎与 `vllm/model_executor/` 的位置；`CONTRIBUTING.md` 的 "Codebase structure" 一节和 `docs/contributing/` 目录是两个项目自带的地图；
- 找到入口点：一个 Python API 调用最终落到哪个文件，一个 CLI 命令从哪里开始；PyTorch 的 `torch/_C/` 绑定与 vLLM 的 `vllm/entrypoints/`；
- 构建一次：不构建就没有 `compile_commands.json`，clangd 就无法跳转；PyTorch 的 `pip install -e . --no-build-isolation` 与只做 Python 开发时的 `tools/nightly.py`；vLLM 的 `VLLM_USE_PRECOMPILED=1 uv pip install -e .` 与 `docs/contributing/incremental_build.md` 描述的 CMake 增量编译；
- 从一个符号追到底：ripgrep 的用法、clangd 的跨文件跳转、Python 侧的 `inspect` 与断点；宏和代码生成造成的"找不到定义"怎么处理；
- 用测试当文档：PyTorch 的 `test/` 按主题组织（`test_*.py`、`distributed/`、`inductor/`、`cpp/`），`torch/testing/_internal/` 下的 `TestCase`、`instantiate_device_type_tests`、OpInfo；vLLM 的 `tests/` 按子系统组织（`kernels/`、`distributed/`、`models/`、`entrypoints/`、`evals/`）；一个函数的测试往往是它最准确的规格说明；
- 用 `git blame` 与 `git log` 读历史：一行代码为什么是这样，答案通常在引入它的那个 commit 的 PR 描述里；PyTorch 的 commit message 带 `Pull Request resolved:` 链接，顺着它能找到当时的讨论；
- 读 changelog 与 release note：`RELEASE.md` 说明两个项目的发布节奏（PyTorch 按固定日程切 release 分支，近期约两个月一个 minor 版本，并有 cherry-pick 规则；vLLM 约两周一版，minor 版本号递增），决定了改动什么时候能进到用户手里。

核心问题是：

> **给你一个从未见过的百万行仓库和一个报错信息，两小时之内你能把它定位到一个文件的一个函数吗？靠什么？**

实践：为自己选定的项目建立一份"项目地图"——目录职责、构建命令、测试入口、常用 grep 模式——这是全系列贡献日志的第一页。

### 2. 找到切入点：从 issue、RFC 到性能回归

第二篇讲选题。大多数失败的贡献不是做错了，而是选错了：选了没人要的、选了别人已经在做的、选了需要先讨论却直接动手的。

这一篇会覆盖：

- issue 标签体系：vLLM 的 `good first issue` 与 `new-model` 标签、`docs/contributing/README.md` 里的 "Job Board" 与 onboarding 任务看板；PyTorch 以 `module:` 前缀的模块标签（`.github/labeler.yml`、`label_to_label.yml` 定义了自动打标规则）、`actionable` 标签的含义；标签是 maintainer 表达"我们想要什么"的主要渠道；
- RFC 与 roadmap：PyTorch 的大改动走 `pytorch/rfcs` 仓库，vLLM 用 `.github/ISSUE_TEMPLATE/750-RFC.yml` 模板（Motivation、Proposed Change、Feedback Period 通常至少一周、CC List），超过 500 行的架构改动没有 RFC 会被打上 `rfc-required`；roadmap issue 是找"maintainer 想做但没人手"的工作的地方；
- CI 失败：vLLM 的 `docs/contributing/ci/failures.md` 描述了 CI 失败看板和 `[CI Failure]` issue 模板；PyTorch 的 HUD 显示 main 上哪些任务在红；修一个已知的 flaky test 是低风险、高感谢度的切入点；
- 性能回归：vLLM 的 `700-performance-discussion.yml` issue 模板与 `benchmarks/` 目录；PyTorch 的 `benchmarks/` 下按子系统组织的 benchmark 套件；一个带复现脚本和数字的回归报告本身就是贡献；
- 文档与类型缺口：过时的 docstring、缺失的类型标注、和实际行为不一致的说明；这类改动门槛低，但两个项目现在都明确不欢迎"单个 typo"式的一次性 PR，要成规模、成体系地做；
- "别人不愿做但有价值"的工作：补测试覆盖、给旧代码加 deprecation warning（vLLM 有 `docs/contributing/deprecation_policy.md`）、把 issue 里的复现整理成测试用例、对硬件适配路径做验证；
- 先讨论再动手：在 issue 里留言说明意图和方案概要，等 maintainer 回应；PyTorch 明确"没有办法认领 issue"、"新功能的门槛很高"，vLLM 要求大改动先有 RFC；
- 查重：动手前用 `gh issue view --comments` 和 `gh pr list --search` 检查有没有人在做同一件事，vLLM 的 `AGENTS.md` 把这一步列为强制项。

核心问题是：

> **一个项目每天新增几十个 issue、几十个 PR。maintainer 最希望有人来做的是哪一类工作？你怎么判断自己选的题不会在一周后被关闭？**

实践：从 PyTorch 或 vLLM 各挑三个候选切入点，用统一的表格记录：来源（标签/RFC/CI/回归）、是否已有人在做、预计规模、需要先讨论还是可以直接做；最后选定一个。

### 3. 做出一个能被合入的改动

第三篇讲从改动到合入的全部工程。技术上正确只是必要条件，一个能被合入的改动还要让 reviewer 用最少的时间确认它正确、有价值、不会带来维护负担。

这一篇会覆盖：

- 最小 diff：一个 PR 只做一件事，不顺手重构、不顺手格式化无关文件；PyTorch 的贡献文档把"PR 太长"和"改了无关代码"列为最常见的错误；大改动怎么拆成可独立合入的小块，PyTorch 的 `ghstack` 工作流如何支持一叠相互依赖的 PR；
- 测试：改动必须带测试，或者说明为什么不能测；PyTorch 用 `torch/testing/_internal/common_utils.py` 的 `TestCase` 与 `run_tests`，数值测试用 `instantiate_device_type_tests` 做成设备无关；vLLM 用 pytest，`AGENTS.md` 给出的原则是"先设计再写、复用已有测试文件、一个测试一个行为、不在 `tests/` 里放一次性 kernel benchmark"；
- benchmark 数据：性能改动必须附数字——前后对比、测量方法、硬件、shape；vLLM 的 kernel benchmark 放 `benchmarks/kernels/`，端到端用 `benchmark_serving.py`、`benchmark_throughput.py`、`benchmark_latency.py`；PyTorch 的 PR 模板把"性能相关 PR 附 benchmark 结果"列入 checklist；
- 本地 lint：PyTorch 的 `.lintrunner.toml` 定义了数十个 linter（FLAKE8、CLANGFORMAT、CLANGTIDY、RUFF、CODESPELL 等），用 `spin lint` / `spin fixlint` 运行；vLLM 用 `pre-commit`（ruff、clang-format、typos、markdownlint、mypy 多版本、signoff 检查等），`pre-commit install` 后每次 commit 自动跑；lint 不过 CI 根本不会往下走；
- PR 描述规范：vLLM 的 PR 模板要求 Purpose、Test Plan、Test Result 三段，标题必须带 `[Bugfix]`、`[Kernel]`、`[Core]`、`[Model]`、`[Doc]` 等前缀；PyTorch 的 `.github/PULL_REQUEST_TEMPLATE/` 下有三个模板（修 issue、文档/typo、预先批准），修 issue 的 PR 必须写 `Fixes #`，"没有关联 issue 的 PR 可能被自动关闭"，描述"过于冗长会被视为 spam"；
- DCO 与签名：vLLM 要求每个 commit 带 `Signed-off-by`（`git commit -s`），`pre-commit` 的 `signoff-commit` 钩子和 `mergify` 都会检查；PyTorch 走 CLA；
- CI 矩阵：PyTorch 的 `.github/workflows/` 下有一百多个 workflow，`pull.yml` 在每个 PR 上跑，`trunk.yml`、`periodic.yml`、`slow.yml`、`inductor.yml` 在 main 上、按周期或按标签触发，`ciflow/` 标签（`.github/pytorch-probot.yml` 列出）用来手动拉起某一组任务；vLLM 用 Buildkite，`.buildkite/test_areas/*.yaml` 按领域定义任务，每个任务声明 `source_file_dependencies` 决定改了哪些文件才触发、在什么 GPU 上跑；vLLM 的 CI 不会自动为每个 commit 全跑，需要 reviewer 或 `ready` 标签之后用 `/ci run`；
- 读 CI 日志：怎么区分自己引入的失败和 main 上已有的失败；PyTorch 的 `CONTRIBUTING.md` 有 "CI failure tips" 一节；vLLM 的 CI 失败看板；
- review 往返：vLLM 承诺 reviewer 每 2–3 天给状态、7 天没动可以 ping，改动要求用 `action-required` 标签表示；PyTorch 由 triage 团队打模块标签并分配 reviewer，4 个工作日没回应可以留言催；怎么回复 review 意见、什么时候争辩、什么时候照做；
- merge 机制：PyTorch 由 `@pytorchbot merge` 触发，`.github/merge_rules.yaml` 按文件路径规定谁有权批准、哪些 check 必须过；vLLM 由有权限的 maintainer 打 `ready` 标签并合入，`.github/mergify.yml` 处理自动 rebase 和 `needs-rebase`；
- 被拒后怎么办：区分"方向不对"、"时机不对"、"做法不对"三种拒绝；哪些可以改了再提，哪些应该放弃；
- AI 辅助贡献的项目政策：PyTorch 要求"你对你发出的每一行负责"、新贡献者的 PR 必须有 `actionable` issue、新功能 issue 里不要放 AI 生成的方案；vLLM 要求人类提交者审过每一行、PR 描述里声明使用了 AI、commit 加 `Co-authored-by` trailer、说明为什么不与已有 PR 重复、附测试命令与结果，并明确"纯 agent PR 不允许"。

核心问题是：

> **reviewer 打开你的 PR，只有十分钟。这十分钟里他要确认什么？你的 diff、描述、测试、CI 状态分别替他回答了哪个问题？**

实践：为第二篇选定的切入点写出完整的 PR 草稿——diff、测试、benchmark（如适用）、按目标项目模板写的描述——并在本地跑过 lint 和相关测试；记录每一次 review 往返。

### 4. 两个真实 PR 的完整走读：PyTorch 与 vLLM

第四篇是前三篇的实例化。它选取两个已经合入的真实 PR——PyTorch 一个、vLLM 一个——从 issue 或动机开始，到讨论、实现、测试、CI、review、合入、进入哪个版本，逐步走读。

选取标准：

- 规模适中：diff 在几十到几百行之间，一个人能在一小时内读完；
- 流程完整：有前置讨论、有测试、有 review 往返、有至少一次修改；
- 有代表性：PyTorch 的 PR 体现 `ghstack`、模块标签、`@pytorchbot merge`、多 workflow CI 的流程；vLLM 的 PR 体现标题前缀、Purpose/Test Plan/Test Result 描述、`Signed-off-by`、Buildkite 按领域触发的 CI、`ready` 标签的流程；
- 类型互补：一个偏底层（算子或 kernel 层面的修复或优化，需要 benchmark 数据），一个偏系统（调度、引擎或接口层面的改动，需要端到端测试）。

每个走读会覆盖：

- 起点：PR 对应的 issue 或 RFC 是什么，作者在动手之前做了哪些讨论；
- 阅读：作者需要读懂哪些文件才能做这个改动，用第一篇的方法怎么找到它们；
- diff：逐段解释改了什么、为什么这样改、有没有更大的改法被有意避开；
- 测试与数据：加了哪些测试、放在哪个目录、benchmark 怎么做、数字怎么呈现；
- CI：触发了哪些任务、有没有失败、失败是怎么处理的；
- review：reviewer 提了什么、作者怎么回应、哪些意见被采纳、哪些被讨论后维持原样；
- 合入之后：进了哪个版本、有没有后续的 follow-up 或 revert。

核心问题是：

> **两个都是"小"PR，却各花了作者一到几周。这些时间花在哪里了？哪些是可以省的，哪些是这个项目的正常成本？**

本篇最后给出全系列总结：把两个走读里出现的每一个环节，映射回前三篇的对应小节。

实践：按同样的走读格式，复盘自己在第三篇提交的那个 PR——无论它最终合入了还是没有。


## 贯穿全系列的实践线

本系列的练手项目不是代码，而是一份**贡献日志**：一个由读者自己维护的 Markdown 文件，从选项目开始，记录到第一个 PR 合入（或明确放弃）为止。它的价值在于强迫每一步都留下可回看的依据，而不是靠感觉推进：

```text
第一篇    项目地图            目录职责 · 构建命令 · 测试入口 · 常用 grep 模式 · 追过的三条符号路径
第二篇    切入点清单          六个候选 · 来源 · 查重结果 · 预计规模 · 是否需先讨论 · 最终选定一个
第三篇    PR 草稿与往返记录   diff 摘要 · 测试清单 · benchmark 数据 · 描述初稿 · 本地 lint/测试结果 · 每轮 review 意见与回应
第四篇    复盘                按走读格式重述自己的 PR · 时间花在哪 · 下次可以省掉什么
```

到第三篇结束，读者应该有一个真实提交到 PyTorch 或 vLLM 的 PR；到第四篇结束，应该有一份能给团队里下一个人看的复盘。日志本身不追求好看，只要求每一项都有出处：一个链接、一条命令、一段日志。

与它平行的是文档阅读线。每篇会带读者读两个项目里对应环节的真实文件，路径以本系列写作时的源码树为准：

```text
第一篇    PyTorch  CONTRIBUTING.md（Codebase structure 一节）· docs/source/community/（contribution_guide.md 已标注 deprecated、指向 wiki；governance.md · persons_of_interest.md）· test/ 与 torch/testing/_internal/ 的组织 · RELEASE.md
          vLLM     docs/contributing/README.md · docs/contributing/incremental_build.md · tests/ 的组织 · RELEASE.md
第二篇    PyTorch  .github/labeler.yml · .github/label_to_label.yml · .github/ISSUE_TEMPLATE/ · pytorch/rfcs 仓库 · benchmarks/
          vLLM     .github/ISSUE_TEMPLATE/（750-RFC.yml · 450-ci-failure.yml · 700-performance-discussion.yml）· docs/contributing/ci/failures.md · docs/contributing/deprecation_policy.md
第三篇    PyTorch  CONTRIBUTING.md（AI-Assisted Development · Unit testing · Merging your Change · CI failure tips）· .lintrunner.toml · .github/PULL_REQUEST_TEMPLATE/ · .github/workflows/{pull,trunk,lint,periodic}.yml · .github/pytorch-probot.yml · .github/merge_rules.yaml
          vLLM     docs/contributing/README.md（DCO · AI Assisted Contributions · PR Title · Reviews）· AGENTS.md · .pre-commit-config.yaml · .github/PULL_REQUEST_TEMPLATE.md · .buildkite/test_areas/ · .buildkite/ci_config.yaml · .github/mergify.yml
第四篇    两个 PR 各自触及的源码、测试、benchmark 文件，以及 PR 页面上的 CI 与 review 记录
```


## 阅读路径建议

### 完整学习路径

```text
1 → 2 → 3 → 4
```

四篇本来就短，建议按顺序读完，并且边读边推进自己的贡献日志。

### 已经选好了要做什么，想尽快提 PR

```text
3 → 4 → 1
```

第三篇是规则的全集，第四篇看两个实例，第一篇在读源码卡住时回头补。

### 已经提过 PR 但过程不顺

```text
3 → 2
```

先对照第三篇检查 PR 本身，再回到第二篇检查选题——很多"review 没人理"的问题根源在选题而不在 PR。

### 只想读懂项目，暂时不打算贡献

```text
1 → 4
```

第一篇的阅读方法对任何目的都有用；第四篇的两个走读是学习"maintainer 怎么想"的最短路径。


## 本系列的边界

本系列只讨论**如何参与**一个 AI-Infra 开源项目：阅读方法、选题方法、提交规范、协作规则。以下内容与它紧邻，但不在范围内：

- **任何一层的技术原理**：算子怎么分发、kernel 怎么写、调度器怎么工作、通信怎么走。第四篇走读 PR 时会解释那两个改动本身在做什么，但只到读懂这个 diff 所需的程度，不展开背后的机制。
- **git 与 GitHub 的基本操作**：fork、branch、rebase、解决冲突、开 PR。假设读者作为工程师已经熟练；本系列只讲这些操作在两个项目里的特殊约定（如 `ghstack`、`Signed-off-by`）。
- **开源许可、CLA/DCO 的法律含义**：只说明两个项目各自要求什么，不讨论为什么。
- **成为 maintainer 之后的工作**：triage、release management、governance。PyTorch 的 `docs/source/community/governance.md` 描述了 module maintainer 与 core maintainer 的机制，本系列只把它作为"谁有权批准我的 PR"的背景来读。
- **PyTorch 与 vLLM 之外项目的具体流程**：NCCL、Megatron-LM、FlashAttention、Triton、SGLang 在正文中只做定性提及；它们的贡献规则请以各自仓库的文档为准。


## 前置要求与说明

### 前置要求

- 能读懂目标项目至少一个子系统的源码，在本地改出过能运行的东西——本系列讲怎么把它合进去，不讲怎么写出来；
- 熟练使用 git 与 GitHub：branch、rebase、interactive rebase、解决冲突、fork 工作流；
- 会用命令行工具阅读代码：ripgrep 或等价工具、clangd 或等价的 C++ 语言服务器、Python 调试器；
- 能读英文技术文档并用英文在 issue 和 PR 里写清楚一件事；
- 一个可用的开发环境：能从源码构建 PyTorch 或 vLLM（vLLM 的 Python 层开发可以借助预编译 wheel 免去 CUDA 编译）；性能相关的贡献需要一块 NVIDIA GPU。

不要求：

- 提过任何开源 PR；
- 了解 PyTorch 或 vLLM 的治理结构；
- 用过 `ghstack`、`lintrunner`、`pre-commit`、Buildkite。

### 项目与版本基线

- PyTorch 以 **2.x 主线**为准，正文引用的文件路径与政策表述取自 v2.13.0 附近的源码树；
- vLLM 以 **0.x 主线**为准，正文引用取自 v0.27 附近的源码树；
- 贡献规则比源码变化更快。以下几处在写作时正在演化，正文会随文标注：PyTorch 对 AI 辅助贡献与 `actionable` issue 的要求、`spin` 对 `lintrunner` 的包装；vLLM 的 `AGENTS.md`、Buildkite 配置从单文件迁移到 `.buildkite/test_areas/`、CI 触发方式（`/ci run`）与 PR 数量上限。读者动手前应以目标项目当时的 `CONTRIBUTING.md`、`docs/contributing/`、PR 模板为准，本系列教的是如何读懂它们，而不是替代它们；
- 第四篇选取的两个 PR 在正文中会给出编号与链接；总纲不预先指定，以便在写作时选择当时最有代表性的样本。

### 关于其他项目

NCCL、Megatron-LM、FlashAttention、Triton、SGLang 等项目在正文中只做定性对照：它们中有的由单一公司主导、开发主要在公司内部完成后再同步到公开仓库，外部贡献的空间与 PyTorch、vLLM 不同；有的社区规模小、review 依赖少数几个人；有的没有独立的 RFC 流程，设计讨论直接发生在 issue 和 PR 里。这些差异影响选题策略（第二篇）和对 review 周期的预期（第三篇），但具体规则不在本系列展开。读者把 PyTorch 和 vLLM 两套流程学透之后，读任何一个项目的 `CONTRIBUTING.md` 都应该能在半小时内定位到它与这两者的差别。


## 章节目录

1. [读懂一个百万行的代码库](/reading-a-million-line-codebase.html)
2. [找到切入点：从 issue、RFC 到性能回归](/finding-your-entry-point-in-open-source.html)
3. [做出一个能被合入的改动](/landing-a-mergeable-change.html)
4. [两个真实 PR 的完整走读：PyTorch 与 vLLM](/two-real-prs-pytorch-and-vllm.html)


## 最终目标

读完这套系列之后，面对任何一个 AI-Infra 开源项目——PyTorch、vLLM，或者一个从未接触过的新项目——读者应该能够回答：

```text
这个仓库怎么组织、从哪里开始读？              → 第一篇：目录地图与入口点
这段代码为什么是这样？                        → 第一篇：测试即文档、git log 即注释
maintainer 现在最想要什么？                   → 第二篇：标签、RFC、roadmap、CI 看板
有人在做同样的事吗？                          → 第二篇：查重
这个改动要不要先讨论？                        → 第二篇：规模与风险的判断
reviewer 十分钟内能确认它正确吗？             → 第三篇：最小 diff、测试、benchmark、描述
CI 红了是我的问题吗？                         → 第三篇：CI 矩阵与日志
review 三周没动，怎么办？                     → 第三篇：review 文化与沟通
用了 AI 辅助，该怎么声明？                    → 第三篇：项目政策
一个真实的 PR 从头到尾长什么样？              → 第四篇：两个走读
```

最终目标是三种能力：

1. **阅读能力**：进入一个陌生的百万行项目，在有限时间内定位到与问题相关的代码、测试和历史；
2. **判断能力**：从大量 issue、RFC 和 CI 信号里识别出上游真正需要、自己能做、不与他人重复的工作；
3. **交付能力**：以目标项目的规范完成一个改动——diff、测试、数据、描述、CI、review——并把它合入上游。

这套能力不属于任何一层，却决定了每一层的技术能力最终能否转化为对项目的实际贡献。
