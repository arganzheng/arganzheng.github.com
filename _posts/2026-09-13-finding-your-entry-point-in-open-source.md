---
layout: post
title: "AI-Infra 开源贡献指南（02）：找到切入点——从 issue、RFC 到性能回归"
subtitle: "Finding Your Entry Point: Issues, RFCs, Roadmaps, CI Failures and Regressions"
tags: [Open Source, PyTorch, vLLM, AI, AI-Infra]
catalog: true
---

> 本文是[《AI-Infra 开源贡献指南》](/contributing-to-ai-infra-open-source.html)系列的第 2 篇（共四篇）。上一篇：[读懂一个百万行的代码库](/reading-a-million-line-codebase.html)；下一篇：[做出一个能被合入的改动](/landing-a-mergeable-change.html)。

一个很典型的第一次贡献是这样开始的：打开 PyTorch 的 issue 列表，按 `good first issue` 过滤，看到 #191394 "[Elastic] FileStore rendezvous leaks the mkstemp file descriptor"——`_create_file_store()` 调了 `tempfile.mkstemp()` 却没有关掉返回的文件描述符。问题描述清楚、附了代码链接、改动显然只有几行。花一个晚上搭好环境、复现、修掉、补一个回归测试，第二天开 PR。然后发现，截至 2026-09-07 查询，这个 issue 下已经挂着 **5 个 open 的 PR**（#194259、#194623、#195137、#195711、#196096），最早的一个 8 月 20 日就开了；issue 评论区里还有同一个账号在同一天用同一段模板留了三次"I'd like to take this one"。你的是第六个。

第二种开始方式是反过来的：不看 issue，直接做自己觉得有用的东西。一个工程师给 vLLM 写了一个新的 KV cache 传输路径，一千多行，测试齐全，性能数据漂亮。PR 开出来两天后收到一句话：这个规模的架构改动需要先有一个 `[RFC]` issue 讨论设计。`docs/contributing/README.md` 的 "Notes for Large Changes" 一节写得很明白，只是他没读到。

第三种更隐蔽。有人在 PyTorch 上看到一处 docstring 里的拼写错误，顺手改了提 PR。PR 模板里有一个专门的 "Documentation or Typo Fix" 模板，看起来是被欢迎的。但 vLLM 的 `AGENTS.md` 把"single typo"列为明确不接受的 "low-value busywork"，PyTorch 的 `CONTRIBUTING.md` 说"comments, issues or PRs ... low quality or consistently overly verbose"会让你以后的贡献"not be accepted anymore"。两个项目都在 2026 年针对 AI 生成的琐碎 PR 收紧了规则：vLLM 甚至有一个叫 `closed-as-slop` 的标签，截至查询已经打在 97 个被关闭的 PR 上。

这三种失败的共同点是：**改动本身可能都是对的，错在选题**。选了别人已经在做的、选了需要先讨论却直接动手的、选了项目已经明确说不要的。选题这一步没有任何技术难度，却决定了后面所有工作的命运。本篇的核心问题是总纲给出的那一个：

> **一个项目每天新增几十个 issue、几十个 PR。maintainer 最希望有人来做的是哪一类工作？你怎么判断自己选的题不会在一周后被关闭？**

版本锚点：PyTorch v2.14.0（2026-09-02 发布）与 vLLM v0.28.0（2026-08-26 发布）的源码树；GitHub 上的标签、issue、PR 状态用 `gh` CLI 于 **2026-09-07** 查询，文中一律标注。issue 和 PR 是活的，到本文发布时其中一些很可能已经关闭或合入——这恰恰是本篇要教的东西：切入点必须在动手前的那一刻重新核对。


## 一、总览

### 1. 问题

选题环节的失败方式可以归为四类，每一类都能在两个项目的 issue 区找到本周的实例：

| 失败方式 | 表现 | 根源 |
|---|---|---|
| 撞车 | 同一个 issue 下 5 个 open PR；同一个 refactor 有 6 个人各写一版 | 没查重；把"issue 还 open"当成"没人在做" |
| 越级 | 千行 PR 没有 RFC；新功能 issue 里塞了一整套 AI 生成的方案 | 不知道项目对"多大的改动要先讨论"有明文规定 |
| 选了项目不要的 | 单个 typo PR；孤立的 style cleanup；给一个 `needs research` 的 issue 直接发实现 | 没读 `AGENTS.md` / `CONTRIBUTING.md` 里"不欢迎什么"的段落 |
| 选了做不完的 | 认领一个需要 B200 才能验证的性能优化；认领一个 tracker issue 里的整条线 | 没估规模，没看硬件要求 |

这四类失败在 2026 年比以前更常见，原因是 AI 辅助让"写出一个看起来能用的 PR"的成本降到了几乎为零，于是 reviewer 的时间成了唯一的瓶颈。两个项目的应对是一致的：把"什么值得做"更明确地写下来（标签、模板、政策文件），把"没按规则来"的 PR 更快地关掉（自动关闭、`closed-as-slop`、`Stale`）。这对认真的贡献者其实是好事——规则越明确，选题越有依据。

### 2. 方法

与项目无关的通用做法只有一条主线：**maintainer 已经用标签、模板、看板、政策文件把"我们想要什么"写出来了，选题就是去读它们，然后在动手前核对一次没有人在做**。展开为五步：

```text
1  读"想要什么"     标签体系（哪些标签表示"欢迎外部 PR"）、Job Board / 看板、roadmap 与 tracker issue
2  读"不要什么"     CONTRIBUTING / AI_POLICY / AGENTS.md 里的否定句：不要 typo PR、不要没 RFC 的大改、不要没 actionable 的 PR
3  找信号源        除 issue 以外的四个来源：RFC、CI 失败看板、性能回归报告、文档/类型/测试缺口
4  查重            gh issue view --comments；gh pr list --search；看 assignee；看最近一条评论的日期
5  估规模并决定是否先讨论   几行 / 几十行 / 几百行；要不要硬件；maintainer 有没有说过"我会 review 什么样的 PR"
```

背后的理由和整个系列一样：reviewer 的时间是项目最稀缺的资源。一个已经有 5 个 PR 的 issue，第 6 个 PR 消耗 reviewer 的时间却几乎不增加价值；一个没有 RFC 的千行 PR，reviewer 要先在 PR 里补上本该在 issue 里发生的设计讨论；一个 typo PR 要走完 CI、review、merge 的全部开销，收益是一个字母。反过来，maintainer 最希望有人来做的，是**他们已经决定要做、写清楚了要什么、但自己没时间做**的事——在 PyTorch 是 `actionable` 加上 maintainer 一句"I'd review a PR that ..."，在 vLLM 是 `help wanted` 加上 issue 正文里的分步骤说明。

### 3. 两个项目

| 环节 | PyTorch v2.14.0 | vLLM v0.28.0 |
|---|---|---|
| 标签规模（2026-09-07） | 682 个；`module:` 257、`release notes:` 65、`ciflow/` 68、`oncall:` 19、`topic:` 14 | 63 个；按领域/模型/硬件平铺 |
| "欢迎外部 PR"的标签 | `actionable`（396 open）、`good first issue`（52）、`OSS contribution wanted`、`better-engineering`（308） | `good first issue`（21 open）、`help wanted`（32）、`new-model` |
| issue 状态标签 | `needs reproduction` / `needs research` / `needs design` / `actionable` / `triaged` / `high priority` / `small` / `large` | 无状态机；靠 `stale`（90 天）/ `unstale` / `keep-open` |
| 自动打标 | `.github/labeler.yml`（按文件路径）、`.github/label_to_label.yml`（标签推导标签）、`bot-triaged` | `.github/mergify.yml`（PR 按路径/标题）、`.github/workflows/issue_autolabel.yml`（issue 按关键词） |
| issue 模板 | `bug-report.yml`、`pt2-bug-report.yml`、`feature-request.yml`、`documentation.yml`、`release-feature-request.yml`、`ci-sev.md`、`disable-ci-jobs.md`、`disable-autorevert.md`、`blank.md` | `100-documentation` → `750-RFC` 共 9 个，标题前缀 `[Bug]:` / `[RFC]:` / `[CI Failure]:` / `[Performance]:` 等 |
| RFC | 独立仓库 `pytorch/rfcs`：fork → 复制 `RFC-0000-template.md` → PR 打 `commenting` 标签 → 在主仓开 issue | `.github/ISSUE_TEMPLATE/750-RFC.yml`：Motivation / Proposed Change / Feedback Period / CC List；>500 LOC 无 RFC 标 `rfc-required` |
| CI 失败入口 | HUD（`hud.pytorch.org`）；bot 自动开 `DISABLED test_xxx` issue，标 `skipped` + `module: flaky-tests` | CI Failures Dashboard（GitHub Project 20）；`450-ci-failure.yml` 模板，标 `ci-failure`；`docs/contributing/ci/failures.md` |
| 性能回归入口 | `module: regression`（156 open）、`module: performance`；`benchmarks/` 下 20 余个子目录；`RELEASE.md` 的 cherry-pick 分类含 `regression` | `700-performance-discussion.yml`（三个可选段落之一是 "Report of performance regression"）；`benchmarks/`、`benchmarks/kernels/` |
| "不要 typo PR"的原文 | `CONTRIBUTING.md` "AI-Assisted Development"：low quality / overly verbose → 不再接受；PR 模板 "Overly verbose descriptions will be considered spam" | `AGENTS.md` "No low-value busywork PRs"；`docs/contributing/README.md` "Ensure significance" |
| 先讨论再动手 | 新贡献者的 PR 必须对应 `actionable` issue；新功能 issue 里 "NEVER include AI-generated explanation of how to solve" | 大改动先 `[RFC]`；`AGENTS.md` 三条查重命令；"Fail-closed behavior" |
| 查重工具 | 无明文；实际用 `gh pr list --search "<n> in:body"` | `AGENTS.md` 明文：`gh issue view --comments`、`gh pr list --search "<issue_number> in:body"`、`--search "<short area keywords>"` |

### 4. 本文的章节安排

```text
二、标签           两个项目的标签体系（真实标签名与含义）、自动打标规则、actionable 的状态机、Job Board
三、RFC 与 roadmap  pytorch/rfcs 的三步流程与模板章节；vLLM 的 [RFC] 模板字段与 rfc-required；tracker issue
四、CI 失败        vLLM 的 failures.md 与 [CI Failure] 模板；PyTorch 的 HUD 与 DISABLED issue；从 flaky test 到 PR
五、性能回归       700-performance-discussion.yml 与 benchmarks/ 布局；PyTorch 的 regression 标签与 cherry-pick 分类；回归报告的最小要素
六、文档与类型缺口  三处"不欢迎单个 typo"的原文；怎么把小修改做成体系；一个真实例子
七、不起眼但有价值  补测试、deprecation、把 issue 复现变成测试、测量本身就是贡献
八、先讨论与查重    两个项目的明文规则；三条命令；认领留言的写法与反例；"一周后会不会被关"的预测器
九、贡献日志       切入点清单模板；PyTorch 三个候选、vLLM 三个候选（2026-09-07 实查）；选定一个
十、本文小结       要点 · 对照表 · 文件位置
```


## 二、标签：maintainer 表达"我们想要什么"的主渠道

### 1. PyTorch：682 个标签的分层

`gh label list --repo pytorch/pytorch --limit 1000` 在 2026-09-07 返回 682 个标签。数量吓人，但结构清楚，按前缀分五层：

| 前缀 / 类别 | 数量 | 作用 | 例子 |
|---|---|---|---|
| `module:` | 257 | 归属到哪个子系统，决定谁来 triage | `module: dynamo`、`module: inductor`、`module: cuda`、`module: nn`、`module: mps`、`module: elastic`、`module: docs`、`module: typing`、`module: tests` |
| `oncall:` | 19 | 进哪个 oncall 队列 | `oncall: pt2`、`oncall: distributed`、`oncall: export`、`oncall: distributed infra` |
| `release notes:` | 65 | PR 进 release note 的哪一节 | `release notes: nn`、`release notes: distributed (dtensor)`、`release notes: inductor (aoti)` |
| `ciflow/` | 68 | 打在 PR 上触发某组 CI | `ciflow/trunk`、`ciflow/inductor`、`ciflow/h100`、`ciflow/mps`、`ciflow/b200` |
| `topic:` | 14 | PR 的变更类型 | `topic: bug fixes`、`topic: performance`、`topic: docs`、`topic: not user facing`、`topic: bc breaking`、`topic: deprecation` |

对选题真正重要的是第六层——**issue 状态与"欢迎程度"标签**。它们没有统一前缀，要单独记住（描述引自 `gh label list` 的 description 字段）：

| 标签 | 描述（原文） | 对贡献者的含义 |
|---|---|---|
| `triaged` | This issue has been looked at a team member, and triaged and prioritized into an appropriate module | 有人看过了，但不代表欢迎 PR |
| `needs reproduction` | Ensure you have actionable steps to reproduce the issue. Someone else needs to confirm the repro. | 可以做的事是**复现并确认**，不是修 |
| `needs research` | We need to decide whether or not this merits inclusion, based on research world | 还没决定要不要做；发 PR 会被关 |
| `needs design` | We want to add this feature but we need to figure out how first | 要做，但方案未定；可以参与设计讨论 |
| `actionable` | （无描述） | **可以发 PR 了**——新贡献者唯一应该盯的标签 |
| `good first issue` | （无描述） | RFC-0058 的定义："An 'actionable' issue that is especially simple and well suited for new contributors" |
| `OSS contribution wanted` | PR from open source contributors welcome to solve this issue. | 明确希望外部人来做 |
| `better-engineering` | Relatively self-contained tasks for better engineering contributors | 工程清理类任务，自包含 |
| `internal ramp-up task` | Tasks that are suitable for new folks w/ high-touch guidance from senior PyTorch folks | 面向内部新人，外部人做需先问 |
| `high priority` | （无描述） | 优先级高，但通常已有人在做 |
| `small` / `large` | We think this is a small issue to fix... / We think that this is a pretty chunky piece of work | 规模估计（`medium` 的描述是 docathon 专用，不通用） |
| `low priority` | We're unlikely to get around to doing this in the near future | maintainer 不会做，外部人做也可能等很久 review |
| `module: flaky-tests` / `skipped` | Problem is a flaky test in CI / Denotes a (flaky) test currently skipped in CI. | 第四章的入口 |
| `module: regression` | It used to work, and now it doesn't | 第五章的入口 |
| `has workaround` | （无描述） | 紧迫性低 |
| `bot-triaged` | This is a label only to be used by the auto triage bot | 说明模块标签是 bot 打的，不是人 |
| `Stale` | （无描述） | 长期无更新的 PR 会被打上，随后关闭 |

各状态标签的 open 数量（2026-09-07）：`needs reproduction` 535、`needs research` 211、`needs design` 167、`actionable` 396、`good first issue` 52、`high priority` 309、`better-engineering` 308、`small` 26、`module: docs` 702、`module: typing` 89。全仓 open issue 13,985 个——`actionable` 只占不到 3%，这就是 maintainer 已经筛过一遍的候选池。

### 2. PyTorch 的自动打标：labeler.yml 与 label_to_label.yml

PR 上的很多标签不是人打的。`.github/labeler.yml` 按**改动的文件路径**打标，每个键是一个标签，值是 glob 列表。v2.14.0 里共 213 行，节选：

```yaml
"module: dynamo":
- torch/_dynamo/**
- torch/csrc/dynamo/**
- benchmarks/dynamo/**
- test/dynamo/**

"module: inductor":
- torch/_inductor/**
- test/inductor/**

"module: cpu":
- aten/src/ATen/cpu/**
- aten/src/ATen/native/cpu/**
- aten/src/ATen/native/quantized/cpu/**
- aten/src/ATen/native/Convolution*.cpp
- aten/src/ATen/native/mkldnn/**
- torch/cpu/**

"ciflow/mps":
- aten/src/ATen/mps/**
- aten/src/ATen/native/mps/**
- torch/_inductor/codegen/mps.py
- test/test_mps.py
- test/inductor/test_mps_basic.py

"release notes: distributed (checkpoint)":
- torch/distributed/checkpoint/**
- test/distributed/checkpoint/**
```

三类标签混在一个文件里：`module:`（归属）、`ciflow/`（触发 CI，例如改了 `aten/src/ATen/native/mps/**` 自动跑 MPS 任务）、`release notes:`（归类）。对选题的用处是反向的：**你想做某个子系统，看 `labeler.yml` 里那个 `module:` 对应哪些目录，就知道改动会落在哪、会自动通知谁、会跑哪组 CI**。

`.github/label_to_label.yml` 做的是**标签推导标签**，对 issue 和 PR 都生效，60 行，全文结构如下：

```yaml
# Use this to auto apply labels based on other labels.  Applies to both PRs and
# issues. Currently only supports any and all
- any:
  - "module: opcheck"
  then:
  - "module: custom-operators"
- any:
  - "module: custom-operators"
  - "module: functionalization"
  - "module: aotdispatch"
  - "module: higher order operators"
  - "module: fakeTensor"
  - "module: ProxyTensor"
  - "module: library"
  - "module: reinplacing"
  then:
  - "module: pt2-dispatcher"
- any:
  - "module: dynamo"
  - "module: pt2-dispatcher"
  - "module: inductor"
  - "module: aotinductor"
  - "module: cudagraphs"
  - "oncall: export"
  - "module: compile-time"
  - "module: compiled autograd"
  - "module: flex attention"
  - "module: dynamic shapes"
  then:
  - "oncall: pt2"
- any:
  - "release notes: distributed (c10d)"
  - "release notes: distributed (symm_mem)"
  - "release notes: distributed (pipeline)"
  - "release notes: distributed (fsdp)"
  - "release notes: distributed (dtensor)"
  - "oncall: distributed"
  then:
  - "ciflow/h100-distributed"
```

读法：一个 issue 被打上 `module: dynamo`，就自动进 `oncall: pt2` 的队列；一个 PR 被打上 `release notes: distributed (fsdp)`，就自动触发 `ciflow/h100-distributed`。这张图告诉你**你的 issue 会被哪个 oncall 看到**——如果三天没人理，看看它有没有落进任何一个 `oncall:`，没有的话很可能是 `module:` 标签打错或缺失，可以在评论里请求 triage。

### 3. actionable 与 issue 的状态机

`actionable` 是 PyTorch 在 2026 年最重要的一个标签。`CONTRIBUTING.md` 的 "AI-Assisted Development" 一节原文：

> **PRs must have an associated "actionable" Issue**: Generally, as a new contributor, you should never send a PR that doesn't have a corresponding issue with the "actionable" label. If you just opened the issue, you must wait for a maintainer to review it and mark it actionable before preparing and sending a PR for it.

wiki 的 "The Ultimate Guide to PyTorch Contributions"（`CONTRIBUTING.md` 与三个 PR 模板都链到它）说得更硬：

> Only PRs that address issues labeled actionable will be considered for review. PRs for non-actionable issues will be closed without. If the issue you want to work on is not yet marked actionable, please engage on the issue first to help move it to that state before submitting a PR.

这套状态机正在被正式化。`pytorch/rfcs` 仓库里 2026-09-02 由 @albanD 开出的 PR #106（RFC-0058 "2026 Issue and PR workflow update"，截至查询仍 open）把它写成了定义："issues is our gating mechanism ... until the issue is closed or it is marked 'actionable' (so a PR can be sent for it)"，并给每个状态一个含义：

```text
needs reproduction   等任何人复现；等 maintainer 确认复现
needs research       等任何人提供"这个 bug 是真的 / 这个 feature 有价值"的证据；等 maintainer 决定要不要做
needs design         等任何人提出设计；等 maintainer 认可设计
actionable           issue 里的信息足够任何人写出一个好 PR；打这个标签的 maintainer 愿意 review 对应的改动
not planned          有效但 ROI 太低
```

对贡献者最有用的一句是 `actionable` 的第二个条件：**打标的 maintainer 已经承诺会 review**。这就是为什么 `actionable` 是"预测一周后不会被关"的最强信号——它不只是"可以做"，而是"有人等着收"。

但 `actionable` 池子是被激烈争抢的。第九章会给出实查：最新 15 个 `actionable` issue 里，几乎每一个都在 24 小时内有了 PR，多的有 5 个。所以光看标签不够，还要看第八章的查重。

### 4. vLLM：63 个标签与 Job Board

vLLM 的标签只有 63 个，没有前缀体系，也没有状态机。`gh label list --repo vllm-project/vllm --limit 200`（2026-09-07）的全集里，与选题相关的：

| 标签 | 描述（原文） | open issue 数 | 含义 |
|---|---|---|---|
| `good first issue` | Good for newcomers | 21 | 新人任务；`docs/contributing/README.md` "Job Board" 的第一个链接 |
| `help wanted` | Extra attention is needed | 32 | maintainer 希望外部人来做，通常是**规模较大、有分步骤说明**的任务 |
| `new-model` | Requests to new models | 2 | 新模型请求；Job Board 的第二个链接 |
| `feature request` | New feature or request | 256 | 由 `500-feature-request.yml` 自动打 |
| `bug` | Something isn't working | 818 | 由 `400-bug-report.yml` 自动打；mergify 对标题含 bug/bugfix 的 PR 也打 |
| `RFC` | （无描述） | 207 | 由 `750-RFC.yml` 自动打 |
| `performance` | Performance-related issues | 59 | 由 `700-performance-discussion.yml` 自动打；PR 改 `benchmarks/` 也打 |
| `ci-failure` | Issue about an unexpected test failure in CI | 14 | 由 `450-ci-failure.yml` 自动打 |
| `documentation` / `installation` / `usage` | Improvements or additions to documentation / Installation problems / How to use vllm | — | 前三个模板自动打 |
| `stale` / `unstale` / `keep-open` | Over 90 days of inactivity / Recieved activity after being labelled stale / Prevents stale label being applied | — | `.github/workflows/stale.yml`：90 天无活动标 `stale`，再 30 天关闭 |
| `ready` | ONLY add when PR is ready to merge/full CI is needed | — | PR 侧：maintainer 认可后打，触发全量 CI |
| `ready-run-all-tests` | Trigger CI with all tests for wide-ranging PRs | — | PR 侧 |
| `needs-rebase` | （无描述） | — | mergify 检测到冲突时自动打 |
| `closed-as-slop` | Pull request determined to be low effort and agent generated | 97 个已关闭 PR | PR 侧：低质量 agent 生成 |
| `verified` | Run pre-commit for new contributors without triggering other tests | — | PR 侧 |
| 领域标签 | `frontend`、`v1`、`torch.compile`、`speculative-decoding`、`structured-output`、`kv-connector`、`quantization`、`multi-modality`、`tool-calling`、`scheduler`、`kv-cache-manager`、`mrv2`、`mrv1-only`、`vllm-ir` | — | 由 mergify 按路径打（PR）或 `issue_autolabel.yml` 按关键词打（issue） |
| 硬件标签 | `rocm`、`cpu`、`tpu`、`intel-gpu`、`nvidia` | — | 同上 |
| 模型标签 | `llama`、`qwen`、`deepseek`、`mistral`、`gpt-oss`、`kimi`、`k3`、`glm`、`minimax`、`cohere` | — | 同上 |

`docs/contributing/README.md` 的 "Job Board" 一节全文只有四个链接：

> Unsure on where to start? Check out the following links for tasks to work on:
> - Good first issues
>     - Selected onboarding tasks
> - New model requests
>     - Models with multi-modal capabilities

"Selected onboarding tasks" 和 "Models with multi-modal capabilities" 指向两个 GitHub Project 看板（org 级 project 6 与 10），是 `good first issue` 与 `new-model` 之外由 maintainer 手工挑选过的子集。看板上的任务比标签更"新鲜"，因为标签是自动的，看板是人维护的。

一个实查发现值得写下来：`600-new-model.yml` 模板写的是 `labels: ["new model"]`（带空格），而仓库里实际存在的标签是 `new-model`（连字符）。结果是通过模板提交的新模型请求**不会**自动得到 `new-model` 标签——截至查询只有 2 个 open issue 带 `new-model`，而按标题 `"[New Model]:" in:title` 搜索能找到一长串无标签的。所以找新模型任务时按标题前缀搜，不要按标签。这类不一致在快速演化的项目里常见，是"以当前仓库为准"的一个具体例子。

### 5. vLLM 的自动打标：mergify 与 issue_autolabel

vLLM 的 PR 标签由 `.github/mergify.yml` 打，规则名一律 `label-<领域>`，条件是文件路径正则或标题正则。v0.28.0 里的规则列表（`grep "^- name:" .github/mergify.yml`）：

```text
label-documentation  label-ci-build  label-cohere  label-deepseek  label-frontend  label-rust
label-llama  label-multi-modality  label-mistral  label-new-model  label-performance
label-quantization  label-qwen  label-gpt-oss  label-kimi  label-k3  label-nvidia  label-rocm
label-xpu  label-cpu  label-structured-output  label-speculative-decoding  label-mrv2
label-tpu  label-tpu-remove  label-tool-calling  label-bug  label-kv-connector
```

以及几条非打标规则：`auto-rebase to keep merge candidate within 1 day behind main`、`ping author on conflicts and add 'needs-rebase' label`、`remove 'needs-rebase' label when conflict is resolved`、`assign reviewer for tensorizer changes`、`assign reviewer for modelopt changes`、`comment-pre-commit-failure`、`comment-dco-failure`。两条与选题直接相关的规则原文：

```yaml
- name: label-new-model
  description: Automatically apply new-model label
  conditions:
    - label != stale
    - and:
      - files~=^vllm/model_executor/models/
      - files=vllm/model_executor/models/registry.py
  actions:
    label:
      add:
        - new-model

- name: label-performance
  description: Automatically apply performance label
  conditions:
    - label != stale
    - or:
      - files~=^benchmarks/
      - files~=^vllm/benchmarks/
      - files~=^tests/benchmarks/
      - files~=^\.buildkite/performance-benchmarks/
  actions:
    label:
      add:
        - performance
```

`label-new-model` 的条件是 `and`：既改了 `vllm/model_executor/models/` 下的文件，又改了 `registry.py`——只有真正注册了新模型的 PR 才会被打标，改现有模型的不会。`assign reviewer for tensorizer changes` 和 `assign reviewer for modelopt changes` 两条则告诉你：改这两块代码，reviewer 是谁已经写在配置里了。

issue 侧的自动打标在 `.github/workflows/issue_autolabel.yml`，按 issue 标题和正文里的关键词打 `rocm`、`cpu`、`kimi`、`k3`、`quantization`、`intel-gpu`。`rocm` 的关键词包括 "composable kernel"、"rccl"、"migraphx"、"hipgraph"，子串包括 "VLLM_ROCM_"、"aiter"、"hip-"、"gfx"、"cdna"；`cpu` 的关键词是 "CPU Backend"、"x86"、"ARM"、"Apple Silicon"、"IBM Z"。这意味着：**如果你想做 ROCm 方向，`label:rocm` 过滤出来的 issue 是关键词匹配的结果，会有误报，但不会漏掉标题里写了 gfx 型号的**。

### 6. 用标签读一个 issue

把上面的表用起来。PyTorch #194344（2026-08-21，`actionable`）的标签是 `triaged, actionable, module: correctness (silent), module: testing, module: accelerator, bot-triaged`。逐个翻译：

```text
triaged                        有人看过
actionable                     可以发 PR，有 maintainer 愿意 review
module: correctness (silent)   静默错误结果——高价值类别（RFC-0058 把 Silent Correctness 列为 high priority 的三类之一）
module: testing                与 torch.testing 模块有关
module: accelerator            与共享 accelerator API 有关——暗示 maintainer 想要设备无关的解法
bot-triaged                    模块标签是 bot 打的，准确度打折
```

再读评论：@malfet 2026-08-24 留了两条，"It would be good to have a generic test for it ... rather than write an MPS-specific test"，"Will accept a PR that adds a non-MPS specific test to validate for it (or enable the existing test for MPS platform)"。这就是 maintainer 把"我要什么"写得最清楚的形态——不仅说了要做，还说了怎么做才收。当天有人开了 PR #194396（MPS 专用测试）被关闭，改开 #194631（设备通用测试，`allow_mps=True, allow_xpu=True`）保持 open。**读标签 + 读 maintainer 的最后一条评论**，五分钟内就能判断这个 issue 还有没有位置、位置在哪。


## 三、RFC 与 roadmap：大改动从哪里开始

### 1. PyTorch：pytorch/rfcs 仓库与三步流程

PyTorch 的 RFC 不在主仓库，在 `github.com/pytorch/rfcs`。`gh api repos/pytorch/rfcs/contents`（2026-09-07）列出的根目录：`README.md`、`CONTRIBUTING.md`、`RFC-0000-template.md`，以及 `RFC-0001-torch-function-for-methods.md` 到 `RFC-0054-HUD-Integration-for-Out-of-Tree-CI-Results.md` 共 15 份已合入的 RFC 和若干 `RFC-00xx-assets/` 目录。其中 `RFC-0024-rfc-process.md` 是 RFC 流程本身的 RFC。

`README.md` 先划边界：

> Smaller changes, including bug fixes and documentation improvements can be implemented and reviewed via the normal GitHub pull request workflow on the main PyTorch repo.
>
> RFCs are more suitable for design proposals that are too large to discuss on a feature-request issue, like adding a new abstraction, or if a discussion about the tradeoffs involved in a new addition are non-trivial.
>
> If you are unsure whether something should be an RFC or a feature-request issue, you can ask by opening an issue in the main PyTorch/PyTorch repository.

然后是三步：

```text
Step 1  Create an RFC        fork pytorch/rfcs；复制 RFC-0000-template.md 为 RFC-00xx-your-feature.md；可以只放一个公开 Google Doc 的链接
Step 2  Get Feedback         PR 标题 RFC-00xx-your-feature.md；先打 draft 标签，准备好后换 commenting 标签；
                             在 pytorch/pytorch 开一个 issue 链到 RFC PR，由 triage 路由给相关 core contributors；
                             在 dev-discuss.pytorch.org 的 rfc-chatter 版和 Slack 上扩散；按 CODEOWNERS 找利害相关人
Step 3  Implement            RFC PR 被接受后合入 pytorch/rfcs；实现 PR 要链回 RFC
```

`RFC-0000-template.md` 的章节：Summary、Motivation、Proposed Implementation、Metrics、Drawbacks、Alternatives、Prior Art、How we teach this、Unresolved questions，最后是 Resolution（Level of Support、Additional Context、Next Steps、Tracking issue、Exceptions）。README 还有两段对选题有用：

> The author of an RFC is not obligated to implement it. ... If you are interested in working on the implementation for an accepted RFC, but cannot determine if someone else is already working on it, feel free to ask (e.g. by leaving a comment on the associated issue).

> Some RFC pull requests are tagged with the "shelved" label when they are closed (as part of the rejection process). An RFC closed with "shelved" is marked as such because we want neither to think about evaluating the proposal nor about implementing the described feature until some time in the future.

第一段是一个被忽视的切入点：**已接受但没人实现的 RFC**。每个接受的 RFC 都有一个主仓的 tracking issue，可以直接问。第二段是一个反向信号：`shelved` 的 RFC 不要去做。

`gh pr list --repo pytorch/rfcs --state open`（2026-09-07）有 8 个 open PR，从 #93（2026-04-14）到 #106（2026-09-02）；最近合入的有 #104 "CUDA update policy"（2026-08-20）、#102 "Refine L3 promotion criteria for downstream repositories"（2026-08-25）。可以看到 2026 年的 RFC 主题偏向流程与外部后端（CRCR、out-of-tree CI），而不是新算子——新算子级别的改动走 feature-request issue 就够了。

### 2. vLLM：[RFC] issue 模板与 rfc-required

vLLM 的 RFC 就是一个 issue。`.github/ISSUE_TEMPLATE/750-RFC.yml` 的头部与字段：

```yaml
name: 💬 Request for comments (RFC).
description: Ask for feedback on major architectural changes or design choices.
title: "[RFC]: "
labels: ["RFC"]
```

正文字段依次是（`label:` 字段原文，均带句号）：

| 字段 | description 原文 | required |
|---|---|---|
| `Motivation.` | The motivation of the RFC. | true |
| `Proposed Change.` | The proposed change of the RFC. | true |
| `Feedback Period.` | The feedback period of the RFC. Usually at least one week. | false |
| `CC List.` | The list of people you want to CC. | false |
| `Any Other Things.` | Any other things you would like to mention. | false |
| `Before submitting a new issue...` | 复选框：Make sure you already searched for relevant issues, and asked the chatbot ... | true |

模板顶部还有一行提示："Please take a look at previous RFCs for reference"，链到 `label:RFC sort:updated-desc`。截至查询有 207 个 open 的 `RFC` issue，2026-09-01 到 09-06 这一周新开了至少 8 个（#54477 到 #55584），主题从 "GDS kv offloading" 到 "Custom all-reduce for XPU"。

什么时候必须先开 RFC，`docs/contributing/README.md` 的 "Notes for Large Changes" 一节写死了数字：

> Please keep the changes as concise as possible. For major architectural changes (>500 LOC excluding kernel/data/config/test), we would expect a GitHub issue (RFC) discussing the technical design and justification. Otherwise, we will tag it with `rfc-required` and might not go through the PR.

两个细节：500 行**不算** kernel、数据、配置、测试；"might not go through the PR" 是委婉说法，实际就是不 review。还有一个实查发现：`rfc-required` 这个标签**不在** 2026-09-07 的 `gh label list` 结果里——它要么已被删除、要么从未作为标签存在，实际执行是 reviewer 在评论里说一句然后不再看。这不改变规则本身：超过 500 行的架构改动，先开 `[RFC]:` issue。

### 3. roadmap 与 tracker issue

两个项目都用 issue 当 roadmap，只是叫法不同。vLLM 有标题带 `[Roadmap]:` 的 issue，例如 #33702 "[Roadmap]: PD Disaggregation with `NixlConnector` Roadmap"（@NickLucche，`help wanted`）——正文是一张带复选框的清单，"Currently Supported Features" 下每一项链到对应的 PR，未勾选的项就是待做的。2026-08-31 有人评论说正在做其中 "More efficient h2d copy_blocks operations for HMA groups" 一项并给了 PR #54483 的链接。这是 roadmap issue 的标准用法：**挑一个未勾选的项，评论认领，链 PR**。

PyTorch 的对应物是标题带 `Tracking:` 或 `[Tracker]` 的 issue，以及 `slowroll` 标签（描述："BC breaking changes that are being slowroll. The tag helps us avoid losing track of these"）。例如 #191236 "Tracking: staged BC rollout — SAC-saved tensors respecting user saved_tensors_hooks"（`actionable`、`slowroll`、0 评论）。tracker issue 里的子任务通常规模适中、目标明确，但要注意它们往往由 maintainer 自己按节奏推进——0 评论且 `slowroll` 的 tracker，先问再动。

另一个 PyTorch 特有的入口是 `release-feature-request.yml` 模板（标签 `release-feature-request`，描述 "This tag is to mark Feature Tracked for PyTorch OSS Releases"）。它的字段包括 "Point(s) of contact"、"Release Mode (pytorch/pytorch features only)"、"Plan for documentations / tutorials"、"Testing Support (CI, test cases, etc..)"——这是**每个 release 要 highlight 的功能**的登记表，读它能知道下一版 maintainer 在忙什么，你的切入点最好不要和这些主线正面相撞，但可以做它们的周边（测试、文档、benchmark）。

### 4. 什么时候需要 RFC

把两个项目的规则合成一张判断表：

| 改动 | PyTorch | vLLM |
|---|---|---|
| bug fix，几行到几十行 | `actionable` issue → PR | issue（可选）→ PR |
| 新算子 / 新 kernel 变体 | feature-request issue，等 `actionable` | `[Feature]:` issue；若是模型层 `[New Model]:` |
| 新抽象、新公开 API、跨模块重构 | `pytorch/rfcs` PR + 主仓 issue | `[RFC]:` issue，Feedback Period 至少一周 |
| >500 行架构改动 | RFC | RFC，否则 `rfc-required` |
| BC-breaking | RFC；PR 模板有 "BC-breaking?" 一栏必填 | RFC；走 `deprecation_policy.md` 的三阶段 |

一个常见误判：把"我写了很多代码"当成"需要 RFC"的唯一标准。判断标准其实是**有没有需要 maintainer 拍板的设计决策**——一个 800 行的纯 kernel 优化（不算入 500 行）可以不走 RFC，一个 200 行但引入了新配置项和新公开接口的改动应该走。


## 四、CI 失败：低风险、高感谢度的切入点

### 1. vLLM：Dashboard、failures.md 与 [CI Failure] 模板

vLLM 把 CI 失败的处理流程完整写在 `docs/contributing/ci/failures.md`。它的第一段回答的是 PR 作者的问题——"我的 PR 上 CI 红了，但我觉得不是我弄的"：

> - Check the dashboard of current CI test failures: CI Failures Dashboard
> - If your failure **is already listed**, it's likely unrelated to your PR. Help fixing it is always welcome!
>     - Leave comments with links to additional instances of the failure.
>     - React with a 👍 to signal how many are affected.
> - If your failure **is not listed**, you should **file an issue**.

"Help fixing it is always welcome" 是这份文档里对贡献者最直接的邀请。Dashboard 是 GitHub Project 20，列出 main 上当前已知的失败；"Daily Triage" 一节说 maintainer 每天用 Buildkite analytics 的 2-day view 对比它。文档后面几节是可以直接照做的操作手册：

```text
Filing a CI Test Failure Issue   用 450-ci-failure.yml；标题格式 [CI Failure]: failing-test-job - regex/matching/failing:test；
                                 环境字段写 "Still failing on main as of commit abcdef123"；描述里逐条 FAILED failing/test.py:failing_test1
Logs Wrangling                   .buildkite/scripts/ci-fetch-log.sh --pr <PR>   拉一个 PR 最新 build 的全部失败 job 日志
                                 .buildkite/scripts/ci-clean-log.sh ci.log      去掉时间戳与 ANSI
Investigating                    去 Buildkite main 分支 → 二分找到第一个出问题的 build → 写进 issue → 找到可疑 PR 就 ping 作者
Reproducing                      .buildkite/scripts/rerun-test.sh tests/v1/engine/test_engine_core_client.py::test_kv_cache_events[True-tcp]
Submitting a PR                  描述里写 Closes #12345；给 PR 加 ci-failure 标签
```

`450-ci-failure.yml` 模板本身（标题前缀 `[CI Failure]: `，自动标签 `ci-failure`）的字段：`Name of failing test`（input，required，placeholder 是 `path/to/test_file.py::test_name[params]`）、`Basic information`（三个复选框：Flaky test / Can reproduce locally / Caused by external libraries (e.g. bug in `transformers`)）、`🧪 Describe the failing test`（required）、`📝 History of failing test`（required，描述里给了三种定位方法：Buildkite Test Suites、`git bisect`、手动 unblock 可疑 PR 的 Buildkite step）、`CC List.`（描述："Usually, this includes those who worked on the PR that failed the test"）。

截至查询有 14 个 open 的 `ci-failure` issue。它们的一个特点是**大多需要特定硬件**——"LM Eval PCP (4xB200)"、"Kimi-Linear-48B-A3B Disaggregated DP EP"、"test_deepep_moe.py SIGSEGV on ROCm"。这是 vLLM CI 失败入口的现实：能在一块消费级 GPU 上复现的失败很少留到 issue 阶段。但也有例外，例如 #43350 "Multiple tests failing with assert output_size is not None" 这类纯逻辑断言——它的评论区 2026-08-22 已有人指出在 main 上不再复现并给出了修复它的 commit，这类 issue 可做的事是确认后请求关闭。选 CI 失败当切入点时，第一件事是看 "Name of failing test" 里的路径判断硬件要求，第二件事是看最后一条评论判断它是否还活着。

### 2. PyTorch：HUD、DISABLED issue 与 skipped 标签

PyTorch 的 CI 状态看板是 HUD（`hud.pytorch.org`）。`CONTRIBUTING.md` 的 "CI failure tips" 一节：

> Fairly often, a CI failure might be unrelated to your changes. You can confirm by going to our HUD and seeing if the CI job is failing upstream already. In this case, you can usually ignore the failure.

与 vLLM 不同，PyTorch 的 flaky test 处理是**全自动**的：`pytorch-bot` 检测到一个测试在 trunk 上反复失败，就自动开一个标题为 `DISABLED test_xxx (__main__.TestClass)` 的 issue，打上 `skipped` 和 `module: flaky-tests`，再由 `label_to_label.yml` 推导出 `oncall:`。测试运行器读取 `.pytorch-disabled-tests.json`（`torch/testing/_internal/common_utils.py` 的 `DEFAULT_DISABLED_TESTS_FILE`，`--import-disabled-tests` / `--rerun-disabled-tests` 参数）跳过它们。2026-09-04 一天就有多个这样的 issue，例如 #195960 "DISABLED test_collective_hang (__main__.ProcessGroupGlooWrapperTest)"（`oncall: distributed`、`module: flaky-tests`、`skipped`）。issue 正文是模板化的：

> Platforms: linux, slow
>
> This test was disabled because it is failing in CI. See recent examples and the most recent trunk workflow logs.
>
> Over the past 6 hours, it has been determined flaky in 4 workflow(s) with 4 failures and 4 successes.
>
> **Debugging instructions (after clicking on the recent samples link):** DO NOT ASSUME THINGS ARE OKAY IF THE CI IS GREEN. We now shield flaky tests from developers so CI will thus be green but it will be harder to parse the logs.

截至查询，`skipped` 有 200 个 open，`module: flaky-tests` 有 173 个。这 200 个被跳过的测试就是 200 个**已经被 maintainer 承认是问题、且暂时没人修**的切入点。它们的优点是范围极窄（一个测试函数）、有完整的失败日志链接、修好后关闭 issue 会自动重新启用测试；缺点是根因可能很深（竞态、数值精度、硬件差异），并且 `skipped` issue 不带 `actionable`——按规则，动手前要在 issue 里留言并等 maintainer 回应。

与 CI 相关的另外三个 PyTorch 模板是给 Dev Infra 用的，贡献者只需认得：`ci-sev.md`（标签 `ci: sev`，"critical failure affecting PyTorch CI"）、`disable-ci-jobs.md`（标题 `DISABLED [WORKFLOW_NAME] / [PLATFORM_NAME] / [JOB_NAME]`，"PyTorch Dev Infra only"）、`disable-autorevert.md`（需要 maintainer 手工加 `ci: disable-autorevert` 标签）。看到一个 `ci: sev` open 着，说明 main 上的红不是你的问题。

### 3. 从一个 flaky test 到一个 PR

两个项目修 flaky test 的路径是相同的四步，工具不同：

| 步骤 | PyTorch | vLLM |
|---|---|---|
| 找 | `gh issue list --repo pytorch/pytorch --label skipped --state open`；HUD 的 flakytest 页面 | Dashboard（Project 20）；`gh issue list --repo vllm-project/vllm --label ci-failure --state open`；Buildkite "Test Reliability on main" 按 reliability 升序 |
| 复现 | issue 里的 workflow logs 链接 → 找到 job → 看 Test step；本地 `python test/xxx.py -k test_name` 多跑几次 | `.buildkite/scripts/ci-fetch-log.sh`；`.buildkite/scripts/rerun-test.sh <test id>` 循环跑 |
| 定位 | `git bisect`；HUD 上按时间看首次失败的 commit | Buildkite Test Suites 的历史；`git bisect` |
| 提 PR | 描述写 `Fixes #<DISABLED issue>`，合入后 bot 重新启用测试 | 描述写 `Closes #12345`；加 `ci-failure` 标签 |

修 flaky test 之所以"高感谢度"，是因为它直接减少 maintainer 每天 triage 的噪音，而且不需要任何设计讨论。它之所以"低风险"，是因为改动范围被测试函数本身框死了。它唯一的门槛是耐心——复现一个 4 次成功 4 次失败的测试可能要跑几十遍。


## 五、性能回归：带数字的报告本身就是贡献

### 1. vLLM：700-performance-discussion.yml 与 benchmarks/

vLLM 的性能 issue 模板 `700-performance-discussion.yml`（标题前缀 `[Performance]: `，自动标签 `performance`）的正文是三个**都不必填**的文本框，让一个模板同时服务三种用途：

| 字段（`label:` 原文） | description 原文 |
|---|---|
| `Proposal to improve performance` | How do you plan to improve vllm's performance? |
| `Report of performance regression` | Please provide detailed description of performance comparison to confirm the regression. You may want to run the benchmark script at https://github.com/vllm-project/vllm/tree/main/benchmarks . |
| `Misc discussion on performance` | Anything about the performance. |
| `Your current environment (if you think it is necessary)` | （粘贴 `collect_env.py` 输出） |

第二个字段的 description 直接指向 `benchmarks/` 目录。v0.28.0 的 `benchmarks/README.md` 第一句是 "This directory used to contain vLLM's benchmark scripts"——注意 "used to"：端到端 benchmark 的主入口已经迁到 `vllm bench` 子命令（README 链到 `cli/bench/latency`、`serve`、`throughput` 三份文档），目录里仍保留着 `benchmark_serving.py`、`benchmark_throughput.py`、`benchmark_latency.py` 等脚本，以及一批专项脚本（`benchmark_prefix_caching.py`、`benchmark_prioritization.py`、`benchmark_long_document_qa_throughput.py`、`benchmark_serving_structured_output.py`、`benchmark_ngram_proposer.py`、`benchmark_block_pool.py` 等）。kernel 级 benchmark 在 `benchmarks/kernels/`（`benchmark_fp8_gemm.py`、`benchmark_layernorm.py`、`benchmark_lora.py`、`benchmark_cutlass_moe_fp8.py` 等几十个），另有 `benchmarks/cutlass_benchmarks/`、`benchmarks/fused_kernels/`、`benchmarks/attention_benchmarks/`、`benchmarks/multi_turn/`、`benchmarks/auto_tune/`。CI 的性能基线在 `.buildkite/performance-benchmarks/`（含 `performance-benchmarks-descriptions.md` 与 `tests/`）。

`AGENTS.md` 对 benchmark 的位置有一条硬规定：

> **No one-off kernel benchmarks in `tests/`.** Put kernel perf work in `benchmarks/kernels/`; prove correctness in existing pytest suites.

截至查询 `performance` 标签有 59 个 open issue，2026-09 第一周新开的几个很能说明"什么样的性能 issue 是好切入点"：#55577 "[Performance]: 100x speedup in v1 prompt_logprobs: Triton _topk_log_softmax_kernel vs native PyTorch topk/logsumexp on long contexts" 是带数字的提案；#55462 "[Performance]: AWQ CUDA GEMM kernel is heavily L1/Memory bound (Profiled on RTX 3070 Ti)" 是带 profile 的分析，而且是在消费级卡上做的。**一个带复现脚本、硬件说明、前后数字的性能 issue，即使你自己不修，也已经是一个 maintainer 会感谢的贡献**——它把"有没有问题"这个最贵的问题回答了。

### 2. PyTorch：benchmarks/ 子目录与 regression 的两条路

PyTorch 没有专门的性能 issue 模板，性能问题用 `bug-report.yml`（或 torch.compile 相关的 `pt2-bug-report.yml`，自动标签 `oncall: pt2`）提，靠标签区分：`module: performance`（"Issues related to performance, either of kernel code or framework glue"）、`module: regression`（"It used to work, and now it doesn't"，156 open）、`topic: performance`（PR 侧）。

`benchmarks/` 目录按子系统组织，v2.14.0 有二十多个子目录：`dynamo/`（TorchBench / HF / TIMM 三套模型的 compile 性能，也是 nightly 性能 dashboard 的来源）、`operator_benchmark/`、`instruction_counts/`、`fastrnns/`、`functional_autograd_benchmark/`、`distributed/`、`inductor_backends/`、`transformer/`、`gpt_fast/`、`sparse/`、`nested/`、`profiler_benchmark/`、`serialization/`、`static_runtime/`、`tensorexpr/`、`fuser/`、`framework_overhead_benchmark/`、`overrides_benchmark/`、`record_function_benchmark/`、`strict_numerics/`、`inference/`、`diffusion/`。`benchmarks/README.md` 的 "Benchmark List" 只给了其中五个的链接（Fast RNNs、Dynamo、Functional autograd、Instruction counts、Operator），其余要进子目录自己看。`labeler.yml` 把 `benchmarks/dynamo/**` 归到 `module: dynamo` 和 `ciflow/inductor`——改这个目录会触发 inductor CI。

PyTorch 的回归还有第二条路：**release 周期内的 cherry-pick**。`RELEASE.md` 的 "Cherry Picking Fixes" 一节：

> Typically, within a release cycle fixes are necessary for regressions, test fixes, etc.
> ...
> **NOTE**: The cherry pick process is not an invitation to add new features, it is mainly there to fix regressions

`@pytorchbot cherry-pick` 命令的 `-c` 参数枚举了五种理由：`{regression,critical,fixnewfeature,docs,release}`。"Patch Release Criteria" 一节说 patch release 只考虑"regression break core functionality (stable / beta features)"且"not a viable workaround"的情况。这意味着：**在 release 分支切出之后（`RELEASE.md` 的 "Release Cadence" 一节给了日程），一个能被归类为 `regression` 的修复是最容易被快速合入并进入下一个 patch 版本的改动**——它有明确的紧迫性和明确的流程。

### 3. 回归报告的最小要素

无论哪个项目，一份能被当作切入点的回归报告要有五样东西，缺一样 maintainer 就得回来问：

```text
版本对        好的版本 tag / commit 与坏的版本 tag / commit（不是"最近变慢了"）
复现脚本      能独立运行；vLLM 优先用 vllm bench 或 benchmarks/ 下的脚本，PyTorch 优先用 benchmarks/ 下已有的套件
硬件          GPU 型号、驱动、CUDA/ROCm 版本；vLLM 模板的环境字段就是 collect_env.py 的输出
数字          前后对比，同一台机器同一脚本；至少三次取中位数
范围          只在某个 shape / dtype / 并发下出现，还是普遍
```

有了这五样，报告就是可以直接 `git bisect` 的。如果你还有时间，bisect 出引入回归的 PR 并在报告里 ping 作者——这时候修复往往由原作者一两天内完成，你的贡献是报告本身。


## 六、文档、类型与"不欢迎单个 typo"

### 1. 三处原文

先把两个项目"不要什么"的原话放在一起。vLLM `AGENTS.md` 的 "No low-value busywork PRs" 一节：

> Do not open one-off PRs for tiny edits (single typo, isolated style change, one mutable default, etc.). Mechanical cleanups are acceptable only when bundled with substantive work.

`docs/contributing/README.md` 的 "AI Assisted Contributions" 一节第 2 条：

> **Ensure significance**: Avoid one-off "busywork" PRs (single typo, isolated style cleanup, one mutable default fix, etc.). Bundle mechanical cleanups into a clear, systematic scope.

PyTorch `CONTRIBUTING.md` 的 "AI-Assisted Development" 一节第 1 条：

> **You are personally responsible for what you send**: If the comments, issues or PRs you send are low quality or consistently overly verbose compared to what is expected, your contributions will not be accepted anymore.

以及 `.github/PULL_REQUEST_TEMPLATE/fix_issue.md` 里 Summary 一栏的注释："Overly verbose descriptions will be considered spam."。PyTorch 确实还保留着 `docs_typo.md` 模板（"This template is only for documentation or typo changes. Do not use it for bug fixes or changes to code behavior, as those require an open issue first."），所以 typo PR 在 PyTorch **形式上**仍被允许——但 `AI_POLICY.md` 的最后一段划了底线：

> We do not accept contributions created by fully autonomous agents, and we may close pull requests that appear to have been generated without meaningful human involvement.

一个只改一个字母的 PR，很难证明"meaningful human involvement"。

### 2. 成体系地做

两份文档给出的替代方案是同一个词：**systematic**。文档与类型缺口仍然是门槛最低的切入点之一，条件是把它做成一个有范围、有依据的批次。几种可行的形态：

| 形态 | 依据 | PyTorch 例 | vLLM 例 |
|---|---|---|---|
| 一个模块的 docstring 全面对齐实际行为 | `module: docs`（702 open）里同一模块的多个 issue | `torch/optim/` 下所有 optimizer 的示例代码统一 | `docs/` 下某一功能页与 CLI `--help` 的一致性 |
| 一个子包的类型标注补全 | `module: typing`（89 open）；`CONTRIBUTING.md` "Running `pyrefly`" 一节 | `torch/distributed/elastic/` 下某个子模块 | `pre-commit run mypy-3.12 --all-files --hook-stage manual` 报出的一类错误 |
| 一类失效链接 / 过时引用 | 文档构建的 warning | `docs/source/` 里指向已删除模块的条目 | `docs/` 里指向已改名脚本的链接 |
| 一批 `deprecated` 的清理 | 第七章的 deprecation policy | — | 到期的 `@typing_extensions.deprecated` API |

关键是 PR 描述里能说出"**这是一个什么范围、为什么这个范围、我怎么找全的**"。找全的方法就是第一篇讲的 `rg`：

```bash
# 在 pytorch 检出根目录：找所有 optimizer 文档里用了 SGD 做示例的地方
rg -n "torch.optim.SGD\(" torch/optim/*.py

# 在 vllm 检出根目录：找所有仍在引用已迁移脚本名的文档
rg -n "benchmark_serving.py" docs/
```

### 3. 一个真实例子：Adadelta 的示例

PyTorch #183036 "Adadelta uses SGD in its examples"（2026-05-09，`module: docs`、`module: optimizer`、`actionable`）是一个典型的"看起来是 typo、其实不是"的 issue。报告者发现 `torch.optim.Adadelta` 文档页的示例代码写的是 `torch.optim.SGD(...)`，问"Should i submit a PR to fix this?"。评论区的走向（截至 2026-09-07）：

```text
2026-05-11  两位非 maintainer 确认问题存在、讨论要不要自己提
2026-08-10  @janeyx99（optimizer 维护者）：I'd review a proper fix for this that doesn't cause the same issue for all other optimizers
2026-08-11  @janeyx99：i meant to open this to the community
2026-09-02  有人指出 PR #185401 已修（实查：#185401 "docs: make optimizer load_state_dict example generic" 状态为 CLOSED、未合入）
```

maintainer 那句话就是"成体系"的具体含义：不是把 Adadelta 页面上的 `SGD` 换成 `Adadelta`（那是 typo 级），而是修示例的**生成方式**，让所有 optimizer 页面都不再出现这个问题。这个 issue 到查询时仍 open、仍 `actionable`、没有 open PR——是本篇第九章 PyTorch 候选清单里的一个。


## 七、不起眼但有价值的工作

### 1. 补测试

PyTorch 有一类标题以 `[test]` 开头的 `actionable` issue，例如 #174183 "[test] Add error_inputs for nn.Conv2d module"、#174177 "[test] Add error_inputs for nn.Linear module"（2026-02-03）。它们要求的是给现有模块补 OpInfo 风格的错误输入测试——没有新功能、没有性能数字，但每一个都缩小了"静默接受非法输入"的面。`module: tests` 标签（"Issues related to tests (not the torch.testing module)"）和 `better-engineering`（308 open）下有大量这类任务。#175211 "CUDA/ROCm/Accelerator testing should replace get_device_capability() with feature queries"（`feature`、`module: tests`、`actionable`）则是一个可以拆成很多小 PR 的测试基建任务：把 `torch/testing/_internal/common_cuda.py` 里 `SM53OrLater` 这类按 compute capability 判断的门控换成按功能查询，2026-04 已有人提了五个小 PR 建立模式（截至查询均已关闭，其中 #179827 是被 `Stale` 关掉的）。

vLLM 侧，`AGENTS.md` 的测试原则本身就是选题指南——"what failure am I guarding against, and what is the cheapest level that catches it"——而 #39428 "[torch.compile] E2E correctness testing for fusions"（@ProExpertProg，`help wanted`）是一个 maintainer 明确开出来的测试任务：现有 `tests/compile/fusions_e2e` 只防 fusion 被破坏，不验证 fusion 后的数值正确性。截至查询已有 4 个相关 open PR 各覆盖一个 fusion 切片，评论里有人做了"哪些切片还没人覆盖"的盘点。

### 2. deprecation：vLLM 的三阶段

vLLM 的 `docs/contributing/deprecation_policy.md` 把废弃一个功能写成了三阶段流水线，每个阶段跨一个 minor 版本（`Y`）：

```text
1  Deprecated (Still On By Default)   标记废弃；警告里写明移除版本（如 "This will be removed in v0.10.0"）；
                                      在 help string、日志、API 响应、/metrics、文档、release notes、RFC issue 里同步；
                                      Python API 用 @typing_extensions.deprecated 装饰器
2  Deprecated (Off By Default)        默认关闭；可用 CLI flag 或环境变量重新打开；不打开就报错
3  Removed                            彻底删除；只有走完前两阶段的功能才能删
```

"Important Guidelines" 三条：No Removals in Patch Releases；Grace Period for Existing Deprecations（政策之前废弃的功能从政策生效时开始计）；Documentation is Critical。

这条流水线的每一步都是一个小而明确的 PR：给一个即将废弃的参数加带版本号的警告；到了下一个版本把它改成默认关闭并加逃生开关；再下一个版本删掉。第一步和第三步几乎是机械的，第二步需要判断哪些用户会受影响。`rg "deprecated" vllm/ --type py` 能找到所有在途的废弃项，对照 `RELEASE.md` 的版本号就能知道哪些到期了。

PyTorch 侧对应的是 `topic: deprecation` 标签和 PR 模板的 "BC-breaking?" 一栏——废弃在 PyTorch 是 BC 问题，走 RFC 或至少在 issue 里由 maintainer 确认。

### 3. 把 issue 里的复现整理成测试

`needs reproduction` 在 PyTorch 有 535 个 open——是所有状态标签里最多的。标签描述说得很直接："Someone else needs to confirm the repro."。**复现并把复现写成一个最小测试用例**，是不需要任何权限、不会撞车、maintainer 一定感谢的工作。它推动 issue 从 `needs reproduction` 走向 `actionable`，而按 RFC-0058 的定义，你就成了那个"提供了足够信息让任何人写出好 PR"的人。

具体做法：按 `bug-report.yml` 模板要求的"fully self-contained"标准把复现精简到十行以内；在当前 nightly 上确认仍能复现（注明 commit）；如果能定位到文件和函数，写出来；然后用 `torch/testing/_internal/common_utils.py` 的 `TestCase` 风格把它写成测试草稿贴在评论里。vLLM 没有 `needs reproduction` 标签，但 `bug`（818 open）里大量 issue 停在"报告者环境特殊、无人复现"的状态，同样的工作同样有价值。

### 4. 测量本身就是贡献

vLLM #50128 "[Performance] Measure Transformers backend startup time vs native"（@hmellor，2026-07-28，`help wanted`）是一个把"不起眼但有价值"写成 issue 的范本。正文：

> This comes from an old comment and was never backed by measurements, so the first task is to establish whether the effect is real at all. It may not be.
> ...
> ### Step 1: is there a difference?
> Time engine init for a model that has both implementations, e.g.: `vllm serve Qwen/Qwen3-0.6B --model-impl transformers` / `vllm serve Qwen/Qwen3-0.6B --model-impl vllm`
> Compare time-to-ready across a few runs and a couple of model sizes. Warm vs cold HF cache matters, so control for it.
> If there is no meaningful difference, say so and close this issue.

它明确说"测出来没差别也算完成"。评论区的走向也很典型：2026-08-18 两位贡献者分别在 RTX 3050（4 GB）上做了测量并 profile，定位到 `RMSNormFuser.fuse()` 和 AOT 缓存两处；同日 maintainer 说"Thank you both for the investigation"，自己开 PR #52766 处理棘手的 `RMSNormFuser`，把 AOT 缓存留给贡献者（PR #53295，2026-08-21 open）。**一块 4 GB 的笔记本显卡完成了一个 `help wanted` issue 的第一步**——测量、profile、写清楚。

这一类工作的共同点是：产出不是 diff，而是**信息**——一个确认的复现、一组数字、一个 bisect 结果、一份"哪些切片还没人覆盖"的盘点。它们不会出现在 release note 里，但它们是 maintainer 最缺的东西，而且几乎不可能撞车。


## 八、先讨论再动手，以及查重

### 1. PyTorch 的规则

PyTorch 对"什么时候动手"有三条明文，都在 `CONTRIBUTING.md` 的 "AI-Assisted Development" 一节，上文引过第 1、2 条，第 3 条：

> **New features, utility functions, or core extensions**: Create a short and to the point issue about the problem you're encountering. You should NEVER include AI-generated explanation of how to solve the problem (this will be discussed later once it's decided the feature should be implemented).

三条合起来的流程：**先开一个短 issue 只说问题 → 等 maintainer 把它标成 `actionable` → 再动手**。三个 PR 模板（`fix_issue.md`、`docs_typo.md`、`preapproved.md`）对应三种合法的起点：一个 `actionable` issue（`Fixes #`，"PRs without a linked issue may be automatically closed"）、一个纯文档改动、一个 maintainer 事先同意的改动（`preapproved.md` 要填 "Approved by @<maintainer handle>" 和 "Point to where the pre-approval discussion happened"）。没有第四种。

PyTorch 没有"认领"机制——issue 的 assignee 由 maintainer 设置，RFC-0058 的定义里 "Assign the issue: To no-one indicating we are looking for community help"。所以看到一个 `actionable` 且无 assignee 的 issue，不需要也不可能"申请分配"，能做的只有两件事：查有没有 PR，以及留一条**有内容**的评论。

### 2. vLLM 的规则：三条命令

vLLM 把查重写成了可以直接复制的命令。`AGENTS.md` 的 "Duplicate-work checks" 一节全文：

> Before proposing a PR, run these checks:
>
> ```bash
> gh issue view <issue_number> --repo vllm-project/vllm --comments
> gh pr list --repo vllm-project/vllm --state open --search "<issue_number> in:body"
> gh pr list --repo vllm-project/vllm --state open --search "<short area keywords>"
> ```
>
> - If an open PR already addresses the same fix, do not open another.
> - If your approach is materially different, explain the difference in the issue.

以及 "Fail-closed behavior"：

> If work is duplicate/trivial busywork, **do not proceed**. Return a short explanation of what is missing.

三条命令分别查三样东西：issue 评论里有没有人说"我在做 / 我开了 #xxx"；有没有 PR 在正文里引用了这个 issue 号；有没有 PR 在标题或正文里提到了同一块代码（关键词要短，例如 `flashinfer_utils`、`mkstemp`、`_create_file_store`）。第三条最容易被省掉，但正是它能抓到那些**没有链 issue 的重复 PR**——PyTorch #191394 的 5 个 PR 里，用关键词 `mkstemp` 搜能全部找到，用 issue 号搜会漏掉标题里没写编号的。

同样的命令换个 `--repo` 就能用在 PyTorch 上。本篇第九章的清单就是这样查出来的。

### 3. 认领留言：怎么写、怎么不写

两个项目都没有正式的认领机制，但一条评论仍然有用——前提是它**携带信息**。对比 2026-09-01 出现在多个 vLLM 和 PyTorch issue 下的同一段模板（同一账号，同一天，至少五个 issue，每个 issue 下重复两到三次）：

```text
I'd like to take this one (`<issue title 截断>`). I'll dig into the root cause and follow up with a PR shortly. (claiming via @xxx)
```

与 #31414 下另一位贡献者同日的留言：

```text
I took a look through the current usages of both modules. `vllm.utils.flashinfer` appears to serve as the
general FlashInfer compatibility/wrapper layer, while `vllm.model_executor.layers.quantization.utils.flashinfer_utils` ...
```

第二种才是有用的认领：它证明你读了代码、给出了你理解的边界、隐含了你的方案。第一种不但没用，还给自己贴了标签——第二天那个账号自己在 #31414 下留言 "Stepping back on this one — @lvnpz got here first with the more detailed analysis, and there are already several PRs in flight"。

一条好的认领留言的要素：

```text
1  我核对过：已有的 PR 是 #a、#b（或：没有找到相关 PR）
2  我理解的问题在 <文件>::<函数>，原因是 ...
3  我打算 <方案概要，两三句>；与 #a 的区别是 ...（如有）
4  预计 <规模>；需要 <硬件>（如有）
5  一个明确的问题：这个方向可以吗？/ 需要先开 RFC 吗？
```

然后**等**。PyTorch 的规则是等到 `actionable`；vLLM 没有明文的等待期，但 maintainer 一句 "Sure" 或一个 assignee 就是信号。等待期间可以做第七章的事——复现、测量、写测试草稿——这些不会撞车。

### 4. "一周后会不会被关"的预测器

回到核心问题的后半句。把本篇的信号合成一张打分表，动手前对候选逐项核对：

| 信号 | 加分 | 减分 |
|---|---|---|
| 标签 | PyTorch `actionable` / `good first issue` / `OSS contribution wanted`；vLLM `help wanted` / `good first issue` | PyTorch `needs research` / `needs design` / `low priority` / `internal ramp-up task`；vLLM `stale` |
| maintainer 最后一条评论 | "I'd review a PR that ..." / "Sure" / 给出了具体要求 | "we should discuss" / 指向一个 RFC / 三个月前 |
| open PR 数（三条命令） | 0 | ≥1 且最近有更新；≥3 无论状态 |
| 规模 | 几行到一两百行；改动落在一两个文件 | >500 行无 RFC；跨多个 `module:` |
| 硬件 | 你手上的卡能复现 | 需要 B200 / 多机 / 特定厂商 |
| 与主线的关系 | tracker / roadmap 里未勾选的一项；release 分支上的 regression | 与 `release-feature-request` 正在推进的主线正面冲突 |
| 项目政策 | 成体系的文档/类型批次；补测试；deprecation 流水线的一步 | 单个 typo；孤立 style cleanup；纯 AI 生成 |

减分项里任何一个"硬"的（`needs research`、≥3 个 PR、>500 行无 RFC、单个 typo）都足以让 PR 一周内被关。加分项里最强的两个是 **`actionable` + maintainer 写明了要什么**——这时候你的 PR 不是在申请 review，而是在交付一个已经被下单的东西。


## 九、贡献日志：切入点清单

### 1. 模板

本篇给贡献日志新增的一页是"切入点清单"，格式沿用总纲的四栏并加上依据列：

```markdown
## 切入点清单（查询日期：YYYY-MM-DD）

| # | 项目 | issue | 来源 | 是否已有人在做 | 预计规模 | 需先讨论? | 结论 |
|---|---|---|---|---|---|---|---|
| 1 | | #     标题 | 标签 / RFC / CI / 回归 / 缺口 | open PR 数与编号；最后一条评论日期与作者 | 行数 / 文件数 / 硬件 | 是（为什么）/ 否 | 选定 / 备选 / 放弃（原因） |

查重命令记录：
    gh issue view <n> --repo <owner/repo> --comments
    gh pr list --repo <owner/repo> --state open --search "<n> in:body"
    gh pr list --repo <owner/repo> --state open --search "<keywords>"

选定：#___。理由：___。下一步：___（留言 / 复现 / 开 RFC / 直接 PR）。
```

"是否已有人在做"一栏必须写**具体的 PR 编号和最后活动日期**，不能写"好像没人"。"结论"一栏允许写"放弃"——一份清单里六个候选放弃四个是正常的，这正是这一页存在的意义。

### 2. PyTorch：三个候选（2026-09-07 实查）

用 `gh issue list --repo pytorch/pytorch --label actionable --state open --limit 45` 和 `--label "good first issue"` 取候选，逐个跑三条查重命令。下表中的状态到本文发布时很可能已变化，请重新核对。

| # | issue | 来源 | 是否已有人在做 | 预计规模 | 需先讨论? | 结论 |
|---|---|---|---|---|---|---|
| P1 | #183036 Adadelta uses SGD in its examples | `actionable` + `module: docs` + `module: optimizer` | **无 open PR**（#185401 已关闭未合入）；最后评论 2026-09-02 称"已修"但实查未合入；maintainer @janeyx99 2026-08-10："I'd review a proper fix for this that doesn't cause the same issue for all other optimizers" | 几十行，`torch/optim/` 下示例的生成方式；无硬件要求 | 否——maintainer 已写明要什么；留言确认理解即可 | **选定**：范围清楚、有 maintainer 承诺 review、无竞争者、符合"成体系而非单点"的要求 |
| P2 | #194344 Testing: Add/Extend dtype-converting copy to CPU test | `actionable` + `module: correctness (silent)` + `module: accelerator` | PR #194631 open（2026-08-24，设备通用测试，最后更新 2026-08-26，无 review decision）；#194396 已关闭；@malfet 2026-08-24 写明"non-MPS specific test" | 一个测试函数；需要 MPS 或任一加速器验证 | 否 | **放弃**：#194631 已按 maintainer 要求做了；可做的事是去 review 它 |
| P3 | #191394 [Elastic] FileStore rendezvous leaks the mkstemp file descriptor | `good first issue` + `module: elastic` | **5 个 open PR**：#194259（08-20）、#194623（08-24）、#195137（08-28）、#195711（09-02）、#196096（09-05）；评论区有模板化认领 | 几行；无硬件要求 | 否 | **放弃**：第六个 PR 没有价值；这个 issue 的问题不是缺人修，是缺一个 maintainer 从 5 个里挑一个 |

备选（同样实查）：#175211 "CUDA/ROCm/Accelerator testing should replace get_device_capability() with feature queries"（`actionable`、`module: tests`；无 open PR，2026-04 的五个小 PR 已被 `Stale` 关闭）——规模大、可拆、需要 CUDA 与 ROCm 至少一种，适合先在 issue 里问"从哪个文件开始、上次的模式是否仍被接受"；#189666 "[CUDA] illegal memory atomic on kernelHistogram1D"（`actionable`）——PR #189685 自 2026-07-13 open、已 APPROVED、作者 08-07 在催合入，**不要重复**，但可以学习它的 diff。

### 3. vLLM：三个候选（2026-09-07 实查）

用 `gh issue list --repo vllm-project/vllm --label "good first issue" --state open --limit 21`、`--label "help wanted" --limit 32`、`--label ci-failure` 取候选。

| # | issue | 来源 | 是否已有人在做 | 预计规模 | 需先讨论? | 结论 |
|---|---|---|---|---|---|---|
| V1 | #50128 [Performance] Measure Transformers backend startup time vs native | `help wanted`（@hmellor，2026-07-28） | 测量部分已由两位贡献者完成（2026-08-18）；maintainer 自己开了 #52766 处理 `RMSNormFuser`；AOT 缓存部分 assignee Taimys，PR #53295 open（2026-08-21） | issue 本身几乎完成；剩余是 review #53295 或做 maintainer 评论里提到的"warm start"后续 | 否 | **备选**：直接可做的部分已被认领；关注 #53295 合入后 maintainer 是否开新的 follow-up |
| V2 | #40544 [Feature]: Integrate fused `kMoEFinalizeARResidualRMSNorm` from FlashInfer | `help wanted` + `feature request`（@benchislett） | **无 open PR**；2026-06-26 有人留了详细分析但未跟进；2026-09-01 有模板化认领；issue 作者 2026-05 评论提到 TRTLLM 可能用 MNNVL AR 后端而非此路径 | 几百行：新的 torch.compile custom pass，可能要把 `moe_finalize` 从 fused_moe op 里拆出来；需要支持 FlashInfer 的 NVIDIA 多卡 | **是**——方向本身被作者质疑过；先在 issue 里问"这条路径现在还是想要的吗" | **备选**：有价值、无竞争，但硬件门槛高且需先确认方向 |
| V3 | #31414 [Feature][Cleanup]: Unify `vllm.utils.flashinfer` and `vllm.model_executor.layers.quantization.utils.flashinfer_utils` | `good first issue` + `help wanted` | **6 个 open PR**：#35440（02-26）、#42378（05-12）、#45618（06-14）、#49867（07-26）、#51523（08-08）、#54538（08-31）；评论区 2026-08-27 有人做了"五个实现的对比" | 几十到一百行的重命名与 import 整理 | 否 | **放弃**：与 P3 同病；六个 PR 里没有一个被合入说明 maintainer 尚未决定要哪种切分，再加一个无济于事 |

备选：#39428 "[torch.compile] E2E correctness testing for fusions"（`help wanted`）——4 个 open PR 各覆盖一个 fusion 切片，评论区有盘点；可做的是找出盘点里仍未覆盖的切片，或 review 现有 PR。`ci-failure` 下的 14 个 issue 多数需要 B200 / ROCm / 多机，在单卡上可做的很少，本次未列入。

### 4. 选定与下一步

清单上六个候选，两个"放弃"是因为撞车（P3、V3），一个"放弃"是因为已被按要求做完（P2），两个"备选"分别因为已基本完成（V1）和门槛高需先确认方向（V2）。选定 **P1 #183036**。下一步不是写代码，而是一条留言：说明已核对无 open PR、理解 @janeyx99 要的是"不让其他 optimizer 出同样问题"的修法、给出打算怎么改示例的生成方式、问一句这个方向是否可以。得到回应（或 4 个工作日无回应后按 `CONTRIBUTING.md` "Merging your Change" 一节的建议 ping 一次）再进入下一篇。

这张清单还说明了一件事：在 2026 年的 PyTorch 和 vLLM，**`good first issue` 是竞争最激烈的池子**，因为它是每个新人和每个 agent 的第一个过滤条件。真正空着的位置在 `actionable` + `module: docs`、`needs reproduction`、`skipped`、`help wanted` 里那些需要读代码或需要测量的任务——它们要多花一两个小时理解，回报是没人和你抢。


## 十、本文小结

### 1. 要点回顾

```text
核心问题      maintainer 最想要的：他们已决定要做、写清了要什么、自己没时间做的事
              —— PyTorch 的 actionable + "I'd review a PR that ..."；vLLM 的 help wanted + 分步骤正文
              一周内不被关的预测器：标签状态 · maintainer 最后一条评论 · open PR 数 · 规模与 RFC 门槛 · 硬件 · 项目政策
标签          PyTorch 682 个分五层前缀 + 一组状态标签（needs reproduction → needs research → needs design → actionable）；
              labeler.yml 按路径、label_to_label.yml 按标签推导（module: dynamo → oncall: pt2）
              vLLM 63 个平铺；mergify.yml 按路径/标题打 PR 标签，issue_autolabel.yml 按关键词打 issue 标签；
              Job Board 四个链接；600-new-model.yml 的 "new model" 与实际标签 new-model 不一致——按标题前缀搜
RFC           PyTorch：pytorch/rfcs 仓库，模板九章 + Resolution，draft → commenting → 主仓 issue；已接受未实现的 RFC 是入口
              vLLM：750-RFC.yml 的 Motivation / Proposed Change / Feedback Period（至少一周）/ CC List；>500 LOC 无 RFC 不 review
CI 失败       vLLM：Project 20 看板 · failures.md 的六节操作手册 · 450-ci-failure.yml · ci-fetch-log.sh / rerun-test.sh
              PyTorch：HUD · bot 自动开 DISABLED issue（skipped + module: flaky-tests，200 open）· 修好即自动重新启用
性能回归      vLLM 700-performance-discussion.yml 三段可选；benchmarks/ 主入口已迁 vllm bench，kernel 在 benchmarks/kernels/
              PyTorch module: regression（156）· benchmarks/ 二十余子目录 · RELEASE.md cherry-pick 的 -c regression
              报告五要素：版本对 · 复现脚本 · 硬件 · 数字 · 范围
不要什么      AGENTS.md "No low-value busywork PRs"；README "Ensure significance"；CONTRIBUTING "personally responsible"；
              AI_POLICY "do not accept contributions created by fully autonomous agents"；vLLM closed-as-slop 97 个
不起眼但值    补 error_inputs 测试 · deprecation 三阶段 · needs reproduction（535）复现成测试 · 测量（#50128 用 4 GB 卡完成第一步）
先讨论        PyTorch：短 issue 只说问题 → 等 actionable → PR；三个 PR 模板 = 三种合法起点
              vLLM：三条 gh 命令 → Fail-closed；认领留言要带核对结果、定位、方案、规模、一个问题
清单          六个候选放弃三个、备选两个、选定一个（#183036）；good first issue 是最挤的池子
```

### 2. PyTorch 与 vLLM 对照

| 环节 | PyTorch | vLLM |
|---|---|---|
| "欢迎 PR"的信号 | `actionable`（maintainer 承诺 review） | `help wanted` + 正文分步骤 |
| 状态机 | 四态 + `not planned`（RFC-0058 正式化中） | 无；靠 `stale` 90+30 天 |
| 自动打标 | 路径 → 标签；标签 → 标签 | 路径/标题 → 标签（PR）；关键词 → 标签（issue） |
| RFC 载体 | 独立仓库的 Markdown 文件 + PR | issue 模板 |
| RFC 门槛 | "too large to discuss on a feature-request issue" | >500 LOC（不含 kernel/data/config/test） |
| CI 失败入口 | bot 自动开 issue，人修 | 人开 issue（模板），看板跟踪 |
| 性能 issue | 通用 bug 模板 + 标签 | 专用模板三段 |
| 回归的快速通道 | release 分支 cherry-pick `-c regression` | 无专门通道，两周一版本身就快 |
| typo PR | 形式上有 `docs_typo.md`，实质受 AI 政策约束 | 明文不接受 |
| 查重 | 无明文，同样的 `gh` 命令可用 | `AGENTS.md` 三条命令，Fail-closed |
| 认领 | 无机制；assignee 由 maintainer 设 | 无机制；maintainer 可能设 assignee |

### 3. 本篇涉及的文件位置

| 主题 | PyTorch v2.14.0 | vLLM v0.28.0 |
|---|---|---|
| 标签自动化 | `.github/labeler.yml`；`.github/label_to_label.yml` | `.github/mergify.yml`（`label-*` 规则、`needs-rebase`）；`.github/workflows/issue_autolabel.yml`；`.github/workflows/stale.yml` |
| issue 模板 | `.github/ISSUE_TEMPLATE/{bug-report,pt2-bug-report,feature-request,documentation,release-feature-request}.yml`、`{ci-sev,disable-ci-jobs,disable-autorevert,blank}.md`、`config.yml` | `.github/ISSUE_TEMPLATE/{100-documentation,200-installation,300-usage,400-bug-report,450-ci-failure,500-feature-request,600-new-model,700-performance-discussion,750-RFC}.yml`、`config.yml` |
| PR 模板 | `.github/PULL_REQUEST_TEMPLATE/{fix_issue,docs_typo,preapproved}.md` | `.github/PULL_REQUEST_TEMPLATE.md`；`.github/workflows/new_pr_bot.yml` |
| 贡献规则 | `CONTRIBUTING.md`（"AI-Assisted Development"、"Merging your Change"、"CI failure tips"）；`AI_POLICY.md` | `docs/contributing/README.md`（"Job Board"、"Issues"、"AI Assisted Contributions"、"Notes for Large Changes"、"What to Expect for the Reviews"）；`AGENTS.md`（"Duplicate-work checks"、"No low-value busywork PRs"、"Fail-closed behavior"） |
| RFC | `pytorch/rfcs` 仓库：`README.md`、`RFC-0000-template.md`、`RFC-0024-rfc-process.md` | `.github/ISSUE_TEMPLATE/750-RFC.yml` |
| CI 失败 | `CONTRIBUTING.md` "CI failure tips"；`torch/testing/_internal/common_utils.py`（`DEFAULT_DISABLED_TESTS_FILE`、`--rerun-disabled-tests`）；`.github/scripts/filter_test_configs.py`（`DISABLED_JOBS_URL`、`UNSTABLE_JOBS_URL`） | `docs/contributing/ci/failures.md`；`.buildkite/scripts/{ci-fetch-log,ci-clean-log,rerun-test}.sh` |
| 性能 | `benchmarks/README.md` 与子目录；`RELEASE.md`（"Cherry Picking Fixes"、"Patch Release Criteria"） | `benchmarks/README.md`、`benchmarks/kernels/`、`.buildkite/performance-benchmarks/` |
| deprecation | PR 模板 "BC-breaking?" 一栏；`topic: deprecation` 标签 | `docs/contributing/deprecation_policy.md` |

下一篇进入"做出一个能被合入的改动"：选定的切入点如何变成最小 diff、带什么测试、性能改动附什么数字、按两个项目的模板写 PR 描述、本地 lint 与 CI 矩阵、review 往返与 merge 机制。它的核心问题：

> **reviewer 打开你的 PR，只有十分钟。这十分钟里他要确认什么？你的 diff、描述、测试、CI 状态分别替他回答了哪个问题？**


## 下一篇

[做出一个能被合入的改动](/landing-a-mergeable-change.html)
