---
layout: post
title: "AI-Infra 开源贡献指南（03）：做出一个能被合入的改动"
subtitle: "Landing a Mergeable Change: Diff, Tests, Benchmarks, PR, CI and Review"
tags: [Open Source, PyTorch, vLLM, CI, AI, AI-Infra]
catalog: true
---

> 本文是[《AI-Infra 开源贡献指南》](/contributing-to-ai-infra-open-source.html)系列的第 3 篇（共四篇）。上一篇：[找到切入点：从 issue、RFC 到性能回归](/finding-your-entry-point-in-open-source.html)　下一篇：[两个真实 PR 的完整走读：PyTorch 与 vLLM](/two-real-prs-pytorch-and-vllm.html)

一个 PR 在 vLLM 里开了十天。作者修了一个真实的 bug，本地测试全绿，描述写了两屏，还顺手把同一目录下三个文件的 import 排了序。十天里发生的事是：DCO check 红了（有一个 commit 忘了 `-s`）；`pre-commit` 没跑（新贡献者的 PR 默认不跑，需要 `verified` 或 `ready` 标签）；Buildkite 一个任务也没起（`/ci run` 要有写权限的 reviewer 来敲）；mergify 打上了 `needs-rebase`（main 已经往前走了两百个 commit）；标题没有 `[Bugfix]` 前缀，没人被分派。reviewer 最终打开它时，看到的是一个 diff 里混着无关的 import 重排、描述里找不到"怎么测的"、CI 一片灰色的 PR。他留了一句"could you split the unrelated changes out and add a test plan?"，然后去看下一个。

同一周在 PyTorch 那边，另一个 PR 被 `pr-sanity-checks` 直接拦下：2 300 行，超过了 `.github/scripts/pr-sanity-check.sh` 里写死的 2 000 行上限。作者把它拆成三个 PR 重新提交，但没有用 `ghstack`，三个 PR 互相依赖却各自独立，第一个合入前后两个就一直红着。合入之后 `@pytorchbot merge` 又拒绝了：缺 `release notes:` 标签。

这两个故事里没有一行代码是错的。它们失败在**改动之外的一切**——diff 的形状、描述的结构、测试怎么证明、CI 怎么触发、签名、标签、机器人。这些东西在两个项目的仓库里都有明文规定，散落在 `CONTRIBUTING.md`、`AI_POLICY.md`、PR 模板、`.lintrunner.toml`、`.pre-commit-config.yaml`、`.github/workflows/`、`.buildkite/`、`merge_rules.yaml`、`mergify.yml` 十几个文件里。本篇把它们按一次贡献的自然顺序串起来：从 diff 到测试、数据、lint、描述、CI、review、合入，再到被拒之后怎么办和 AI 辅助的政策。

贯穿全篇的只有一个视角。总纲提出的核心问题是：

> **reviewer 打开你的 PR，只有十分钟。这十分钟里他要确认什么？你的 diff、描述、测试、CI 状态分别替他回答了哪个问题？**

每一条规则都可以还原成这个问题的一个侧面：最小 diff 是让他十分钟读得完；测试是替他回答"这个改动对不对"；benchmark 是替他回答"值不值"；描述是替他回答"为什么这样改、有没有别的改法"；绿色的 CI 是替他回答"会不会弄坏别的东西"。规则会变，这个视角不变。

版本锚点：PyTorch **v2.14.0**、vLLM **v0.28.0**，引用的文件路径、章节标题、字段名、命令均以这两个检出为准，格式为"仓库 路径 的 字段/章节"。GitHub 上的动态信息（标签是否存在）用 `gh` 查询，注明"截至 2026-09 查询"。本篇不展开任何一层的技术原理，也不重复 git 与 GitHub 的基本操作。


## 一、总览

### 1. 问题：这个环节典型的失败方式

从改动到合入之间有一串关卡，每一关都有人倒下。按出现频率排：

```text
关卡        典型失败                                                    后果
diff        一个 PR 改了 40 个文件；顺手重构；顺手格式化无关文件            reviewer 不知从哪看起；PyTorch 超 2000 行直接被 CI 拦下
测试        "本地跑过了"但 diff 里没有测试；或者测试只是把 bug 复现一遍       "how did you verify this?"；搁置
数据        性能 PR 没有前后对比、没有硬件和 shape                          "how much faster?"；搁置
lint        没跑本地 lint；CI 的 Lint / pre-commit 红了                     其他任务不会往下走；vLLM 的 mergify 机器人会留言催
描述        描述两屏长、没有 Fixes #、没有 Test Plan、标题没有前缀            PyTorch 视为 spam；vLLM 没人分派
签名        vLLM 某个 commit 没带 Signed-off-by                             DCO check 红；机器人留言
CI          不知道 vLLM 的 CI 要人来触发；不知道 PyTorch 的 trunk 不在 PR 上跑   以为 CI 是绿的，合入后才发现问题
日志        CI 红了，看不出是自己的问题还是 main 本来就红                    要么乱改，要么反复 rerun
review      三周没人看；或者被要求改而没有回应；或者对每条意见都争辩          PR 变 stale
合入        不知道 @pytorchbot merge 的规则；不知道 vLLM 要等 ready 标签       approved 之后又卡两周
被拒        分不清"方向不对"和"做法不对"，把该放弃的一直改                   耗尽双方耐心
AI          用了 AI 没声明；或者把 AI 输出直接贴进 review 回复               两个项目都写明了可能直接关闭或封禁
```

### 2. 方法：通用做法与背后的理由

这些关卡有一个共同的解释：**maintainer 的 review 时间是项目最稀缺的资源**。PyTorch 每天合入上百个 PR，vLLM 每两周发一版；reviewer 分给一个外部 PR 的第一次注意力通常只有几分钟到十几分钟。所有规则都在做同一件事——让这几分钟的效率最大化，同时把不值得花这几分钟的 PR 提前挡掉。由此推出与项目无关的六条通用做法：

```text
做法                    理由
一个 PR 只做一件事        reviewer 一次只能在脑子里装一个"为什么"；无关改动让他无法用 diff 回答"这一行为什么变了"
改动必须带测试           测试是 reviewer 不用自己跑代码就能确认正确性的唯一办法；也是防止将来被别人弄坏的唯一办法
性能改动必须带数字        没有数字的"更快"要 reviewer 自己去测；他不会
先过本地 lint 再推        CI 的 lint 挂了，其他任务根本不跑；每一次红色的 push 都在消耗信任
按模板写描述             模板的每一栏都是 reviewer 要问的一个问题；空着等于让他问
把 CI 变绿是作者的责任    reviewer 不会替你看日志；分清"我的失败"和"main 的失败"是基本功
```

两个项目在这六条上的具体规定不同，但没有一条相互矛盾。学会读一个项目的规则文件（本篇会列出所有位置），下一个项目的规则半小时就能读完。

### 3. 两个项目：同一环节两种做法

```text
环节            PyTorch v2.14.0                                        vLLM v0.28.0
规则文件        CONTRIBUTING.md（技术）+ AI_POLICY.md + GitHub wiki      docs/contributing/README.md + AGENTS.md
PR 体积上限     .github/scripts/pr-sanity-check.sh：>2000 行 CI 失败     docs/contributing/README.md：>500 行架构改动无 RFC 打 rfc-required
拆分工具        ghstack（CONTRIBUTING.md "Run Specific CI Jobs" 提到）    无专门工具；顺序开 PR；每人最多 6 个 open PR
测试框架        unittest：TestCase / run_tests / instantiate_device_type_tests   pytest；AGENTS.md 的四个问题
benchmark       PR 模板 checklist："Included benchmark results"          benchmarks/kernels/；vllm bench serve|throughput|latency
本地 lint       lintrunner（.lintrunner.toml，61 个 linter）；spin lint / spin fixlint   pre-commit（.pre-commit-config.yaml）；pre-commit run
PR 模板         三个：fix_issue / docs_typo / preapproved                 一个：Purpose / Test Plan / Test Result
标题            无前缀要求；合入前需要 release notes: 或 topic: not user facing 标签   必须带 [Bugfix] / [Kernel] / [Core] … 前缀
签名            CLA（merge_rules.yaml 的 mandatory_checks_name 含 EasyCLA） DCO：git commit -s；signoff-commit 钩子；mergify 检查
CI 系统         GitHub Actions；.github/workflows/ 148 个文件            Buildkite；.buildkite/test_areas/ 35 个文件 + ci_config.yaml
PR 上自动跑     pull.yml + lint.yml；其余靠 ciflow/* 标签                 只有 pre-commit（且需 verified/ready 或 4 个已合入 PR）；测试要 /ci run
读日志          CONTRIBUTING.md "CI failure tips"；HUD                   docs/contributing/ci/failures.md；ci-fetch-log.sh；CI Failures Dashboard
review 承诺     triage 几个工作日内打标签分派；4 个工作日无回应可催       2–3 天一次状态；7 天可 ping；改动要求打 action-required
合入            @pytorchbot merge（-f / -i）；merge_rules.yaml 定权限     maintainer 打 ready 并合入；mergify 管 needs-rebase
AI 政策         AI_POLICY.md：不接受全自主 agent 的贡献；标注 AI 内容     AGENTS.md："Pure code-agent PRs are not allowed"；Co-authored-by
```

### 4. 本文的章节安排

```text
二、最小 diff        一个 PR 只做一件事；PyTorch 的 2000 行硬上限与 ghstack；vLLM 的 500 行 RFC 线与顺序 PR
三、测试             PyTorch 的 TestCase / run_tests / instantiate_device_type_tests；vLLM 的 pytest 与 AGENTS.md 四个问题
四、benchmark        什么算"有数字"；PyTorch 的 benchmarks/ 与模板 checklist；vLLM 的 benchmarks/kernels/ 与 vllm bench
五、本地 lint        .lintrunner.toml 的 61 个 linter 与 spin；.pre-commit-config.yaml 的 hook 清单与 pre-commit run
六、描述与签名       三个 PyTorch 模板逐字段；vLLM 模板与标题前缀表；CLA vs DCO
七、CI 矩阵          PyTorch 的 workflow 家族与 ciflow/*；vLLM 的 test_areas / source_file_dependencies / ci_config / /ci run / ready
八、读 CI 日志       区分自己的失败与 main 的失败；HUD、gh pr checks；failures.md 与 ci-fetch-log.sh
九、review 往返      两个项目的时间承诺原文；回应规则表
十、合入             @pytorchbot merge 的变体与 merge_rules.yaml；release notes 标签；vLLM 的 ready 与 mergify
十一、被拒           方向 / 时机 / 做法三类，各自怎么办
十二、AI 辅助政策    AI_POLICY.md 与 AGENTS.md 逐条对照
十三、回答核心问题   reviewer 的十分钟：diff / 描述 / 测试 / CI 各回答什么
十四、贡献日志       PR 草稿模板；两份按项目模板填好的描述样例；review 往返记录表
十五、小结           要点 · 对照表 · 文件位置表
```


## 二、最小 diff：一个 PR 只做一件事

### 1. 什么叫"一件事"

一个可以用一句话说清目的的改动。判断标准不是行数，而是 reviewer 能否对 diff 里的**每一行**都回答"这一行为什么变了"，并且答案都指向同一句话。以下三种情况都违反这条：

- **顺手重构**：修 bug 时把旁边的函数抽了出来。reviewer 现在要同时验证 bug 修对了、重构没改语义，工作量翻倍，而且一旦要 revert，两件事一起没了；
- **顺手格式化**：IDE 保存时把整个文件重排了 import 或调整了空行。diff 里 80% 是噪音，真正的改动被淹没；
- **顺手扩大范围**："既然改了这个 kernel，就把另外两个 dtype 也支持了吧"。每个新 dtype 都需要新测试、新 benchmark，每一项都是 reviewer 要多问的问题。

反过来，"一件事"可以很大——一个新的 attention backend 也是一件事——但那时应该拆成一叠相互依赖、各自可合入的小 PR。

### 2. PyTorch：2 000 行硬上限与 ghstack

PyTorch 把 PR 体积写进了 CI。`.github/workflows/lint.yml` 的 `pr-sanity-checks` 任务只在 `pull_request` 事件上跑，调用 `.github/scripts/pr-sanity-check.sh`；脚本用 `git diff --stat` 统计非生成文件（`.gitattributes` 里 `linguist-generated=true` 的文件被排除）的增删行数之和：

```bash
# pytorch .github/scripts/pr-sanity-check.sh（节选）
if ((pr_size > 2000)); then
    echo 'Your PR is '"$pr_size"' LOC which is more than the 2000 maximum'
    echo 'allowed within PyTorch infra. PLease make sure to split up'
    echo 'your PR into smaller pieces that can be reviewed.'
    exit 1
fi
```

绕过它需要 `skip-pr-sanity-checks` 标签（截至 2026-09 查询该标签存在），只有 maintainer 会打。这是一个明确的信号：超过 2 000 行的 PR 项目根本不打算 review。

拆分的工具是 `ghstack`。它把本地一串 commit 变成一叠 PR，每个 PR 的 base 是上一个 PR 的 head（分支形如 `gh/<user>/<N>/base`、`gh/<user>/<N>/head`、`gh/<user>/<N>/orig`），可以从下往上逐个 review、逐个合入，修改中间一个会自动更新上面所有。`CONTRIBUTING.md` 在 "Run Specific CI Jobs" 一节直接用 `ghstack submit` 作为提交命令，并说明"It is not recommended to use this workflow unless you are also using `ghstack`"；`.github/workflows/check_mergeability_ghstack.yml` 专门检查 `gh/**/base` 分支的可合并性；`.github/scripts/trymerge.py` 里的 `get_ghstack_prs` 负责在合入时按顺序处理整叠。检出根目录的 `AGENTS.md` 有一节 "ghstack Workflow"，写明了识别 ghstack commit 的三个信号（detached HEAD、`ghstack-source-id` trailer、`origin/gh/USERNAME/N` 分支）和"Never push directly"的规则。

一叠 PR 怎么切，原则是**每一层单独看都是完整的、可合入的、有测试的**：

```text
第 1 层   重构：为新功能腾出接口，不改行为 —— 测试：现有测试全过
第 2 层   新功能的核心实现 + 单元测试 —— 默认关闭或不暴露
第 3 层   接线：把新功能接到公开 API，更新文档 —— 端到端测试
第 4 层   （如有）性能优化 + benchmark
```

### 3. vLLM：500 行 RFC 线与顺序 PR

vLLM 没有 CI 层面的行数上限，但 `docs/contributing/README.md` 的 "Notes for Large Changes" 一节划了一条线：

> Please keep the changes as concise as possible. For major architectural changes (>500 LOC excluding kernel/data/config/test), we would expect a GitHub issue (RFC) discussing the technical design and justification. Otherwise, we will tag it with `rfc-required` and might not go through the PR.

注意排除项：kernel、数据、配置、测试不计入 500 行。一个 800 行的 CUDA kernel 加 300 行测试不需要 RFC；一个 600 行的调度器重构需要。（顺带一提：截至 2026-09 用 `gh api repos/vllm-project/vllm/labels/rfc-required` 查询，这个标签在仓库的标签列表里并不存在，`action-required` 也不存在——文档里写的标签名与实际标签可能已经漂移，以当前仓库为准。）

vLLM 没有 ghstack。拆分大改动的做法是**顺序开 PR**：先开第 1 层，等它合入，再基于新的 main 开第 2 层。这意味着一叠 PR 的总周期等于各层周期之和，所以每层要尽量独立、尽量小。另一个约束来自 "Pull Request Limits and Escalation" 一节：没有写权限的贡献者最多同时开 6 个 PR（"The current cap is 6 open PRs"）。一叠 8 层的改动得分两批。

如果第 2 层等不及第 1 层合入，可以把第 2 层的 PR 基于第 1 层的分支开，在描述里写明"Depends on #NNNN"，并在标题上加 `[WIP]` 或开成 draft；第 1 层合入后 rebase 到 main。但这个做法要求 reviewer 记住依赖关系，只在确有必要时用。

### 4. 拆分之外的最小化手段

- **不动无关文件**：提交前用 `git diff --stat` 看一眼文件清单，每个文件都要能用 PR 的那句话解释；
- **不改公共接口除非必要**：vLLM 有 `docs/contributing/deprecation_policy.md`，PyTorch 的 PR 模板有 "BC-breaking?" 一栏，任何接口变化都会引来额外的审查；
- **把"顺手发现的问题"变成 issue 或另一个 PR**：而不是塞进当前 PR。这本身也是一个低成本的贡献。


## 三、测试：改动必须带测试

### 1. 测试替 reviewer 回答什么

一个带测试的 PR 让 reviewer 不必在脑子里模拟代码执行。他只需要看两件事：**测试是否覆盖了改动声称要解决的问题**（修 bug 的 PR，测试在改动前应该失败、改动后应该通过），以及**测试是否会在将来别人弄坏它时报警**。所以测试的形状比数量重要：一个精确复现 issue 场景的断言，胜过十个泛泛的"跑一遍不报错"。

两个项目都有"改动必须带测试"的规定。PyTorch 的三个 PR 模板 checklist 里都有 `- [ ] Added/updated tests`；vLLM 的 "Code Quality" 一节写着 "Include sufficient tests to ensure the project stays correct and robust. This includes both unit tests and integration tests."。如果确实无法测（比如只能在特定硬件上复现），要在描述里说明原因和手工验证方法。

### 2. PyTorch：TestCase、run_tests、instantiate_device_type_tests

PyTorch 用 `unittest`，但不是裸的 `unittest`。检出根目录 `AGENTS.md` 的 "Testing" 一节给出了最小骨架：

```python
from torch.testing._internal.common_utils import run_tests, TestCase

class TestFeature(TestCase):
    ...

if __name__ == "__main__":
    run_tests()
```

三个名字都在 `torch/testing/_internal/common_utils.py` 里：`TestCase`（继承 `expecttest.TestCase`，提供 `assertEqual` 对 Tensor 的容差比较）、`run_tests`（接管命令行参数、与 CI 的 sharding 和 flaky 检测配合）、`parametrize`（多组输入）。同一节的三条规则：

- "To test Tensor equality, use assertEqual."
- "For tests over multiple inputs, use the `@parametrize` decorator."
- "For any test that checks numerics of the on-device implementation, use `instantiate_device_type_tests` to write device-generic tests."

`instantiate_device_type_tests` 在 `torch/testing/_internal/common_device_type.py`，同文件还有 `dtypes`、`dtypesIfCUDA`、`onlyCUDA`、`ops`（配合 `torch/testing/_internal/opinfo/core.py` 的 `OpInfo`）。写成设备无关的测试意味着同一个测试会在 CPU、CUDA、MPS、XPU 等所有 CI 覆盖的设备上生成一份，这是 PyTorch 保证"一个算子在所有后端行为一致"的机制。

`.lintrunner.toml` 里有一个 linter 叫 `TEST_HAS_MAIN`，检查 `test/` 下的文件是否有 `if __name__ == "__main__": run_tests()` 的入口——没有的话 lint 就会红。这是"测试规范"被写进工具链的一个例子。

跑法（`CONTRIBUTING.md` "Python Unit Testing" 与 "Better local unit tests with `pytest`"，在仓库根目录执行）：

```bash
python test/run_test.py                         # 全部（很慢，本地一般不跑）
python test/test_nn.py TestNN.test_BCELoss      # 单个测试类.方法
pytest test/test_nn.py -k Loss -v               # pytest 非官方支持但可用
```

C++ 测试在 `test/cpp/`，用 Google Test，构建后二进制在 `build/bin/`：`./build/bin/test_jit --gtest_filter=ContainerAliasingTest.MayContainAlias`（"C++ Unit Testing" 一节的原例）。

### 3. vLLM：pytest 与 AGENTS.md 的四个问题

vLLM 用 pytest，`tests/` 按子系统组织（`kernels/`、`distributed/`、`models/`、`entrypoints/`、`evals/`、`v1/` 等，检出的 `tests/` 顶层有七十个左右的目录和文件）。`docs/contributing/README.md` "Testing" 一节的命令：

```bash
# vllm 仓库根目录
uv pip install -r requirements/common.txt -r requirements/dev.txt --torch-backend=auto
pytest tests/                       # 全部
pytest -s -v tests/test_logger.py   # 单个文件
```

真正规定"怎么写测试"的是根目录 `AGENTS.md` 的 "Tests" 一节。它先要求回答四个问题：

> **Design before you write.** Answer four questions first: what is the module for, what is its I/O contract, what failure am I guarding against, and what is the cheapest level that catches it (unit over integration over e2e)?

然后是五条规则（原文标题）："Reuse before create."（扩展已有测试文件与 `conftest.py` fixture，不轻易新建文件）；"Test behavior with intent."（通过公开 API 断言可观察的结果，在测试名或 docstring 里写明为什么；"flaky tests are worse than no tests"）；"Keep it minimal."（一个测试一个行为；"if the test diff dwarfs the code change, cut scope"）；"No one-off kernel benchmarks in `tests/`."（kernel 性能工作放 `benchmarks/kernels/`）；"Run model evals for model-affecting changes."（搜 `tests/evals/` 或用 `vllm bench`，把结果放进 PR，"do not wait for reviewers to ask"）。

kernel 改动还有一条专门的要求。"Adding or Changing Kernels" 一节：自定义 op 要按 PyTorch 的规范注册，返回 Tensor 的 op 需要 meta-function，并且 "Use torch.library.opcheck() to test the function registration and meta-function for any registered ops. See `tests/kernels` for examples."——`tests/kernels/attention/test_attention.py` 等文件里能看到 `opcheck` 的用法。模型相关的测试要求单独写在 `docs/contributing/model/tests.md`（"Required Tests" 的 "Model loading"、"Optional Tests" 的 "Model correctness" 等小节）。

### 4. 没有 GPU 怎么办

vLLM 的 `docs/contributing/README.md` "Testing" 一节直接承认没有 GPU 时很多测试跑不起来："not all unit tests pass when run on CPU platforms … rely on the continuous integration system to run the tests for now"——但下面第七章会讲，vLLM 的 CI 不会自动为你跑。这两条加在一起意味着：没有 GPU 的 vLLM 贡献者，测试验证依赖 reviewer 替你触发 CI，PR 周期会更长，选题时就要考虑。PyTorch 这边 CPU 上能跑的测试多得多，`instantiate_device_type_tests` 生成的 `_cpu` 版本至少能在本地验证逻辑；CUDA 版本交给 `pull` 里的 CUDA job。


## 四、benchmark：性能改动必须带数字

### 1. 什么算"有数字"

"快了很多"不是数字。一份 reviewer 能用的 benchmark 至少包含：

```text
项            要求
基线          改动前的数字，同一台机器、同一份代码除了这个 diff
对比          改动后的数字；最好多跑几次给出方差或中位数
硬件          GPU 型号、驱动/CUDA 版本；CPU 型号（如相关）
输入          shape、dtype、batch size、序列长度——性能对 shape 极敏感
方法          用的什么工具、什么命令，能让 reviewer 复现
覆盖          不只是最有利的 case；至少包含一个可能变慢的 case 并说明
```

没有基线的数字没有意义；没有 shape 的数字不能比较；没有命令的数字不能复现。三者缺一，reviewer 就得自己跑——他不会。

### 2. PyTorch：模板 checklist 与 benchmarks/

PyTorch 的要求写在 PR 模板里。`.github/PULL_REQUEST_TEMPLATE/fix_issue.md` 与 `preapproved.md` 的 Checklist 第四项：

```text
- [ ] Included benchmark results (for PRs impacting perf)
```

工具在 `benchmarks/` 目录，按子系统组织：`operator_benchmark/`（算子级，`benchmark_all_other_test.py` 等入口）、`dynamo/`（编译栈，同时被 `merge_rules.yaml` 的 "ONNX exporter" 规则匹配）、`inductor_backends/`、`distributed/`、`transformer/`、`sparse/`、`instruction_counts/` 等。算子级的小改动通常用 `torch.utils.benchmark.Timer` 写一段几十行的脚本贴进 PR 描述即可；涉及 Inductor 的改动，`.github/pytorch-probot.yml` 的 `ciflow_push_tags` 里有 `ciflow/inductor-perf-compare`、`ciflow/inductor-micro-benchmark`、`ciflow/op-benchmark` 等标签，打上后 CI 会跑对应的性能任务——但这些标签只有有权限的人能打，外部贡献者需要 reviewer 帮忙。

### 3. vLLM：benchmarks/kernels/ 与 vllm bench

vLLM 把 benchmark 分成两层。kernel 层在 `benchmarks/kernels/`，检出里有 66 个文件，命名就是 `benchmark_<kernel>.py`：`benchmark_paged_attention.py`、`benchmark_rmsnorm.py`、`benchmark_moe.py`、`benchmark_fp8_gemm.py`、`benchmark_reshape_and_cache_flash.py` 等。改一个 kernel，先看这个目录有没有对应的脚本；有就直接跑，没有就照着写一个放进去——`AGENTS.md` 明确说 kernel 性能脚本应该放在这里而不是 `tests/`。

端到端层是 `vllm bench` 子命令。这里有一个版本变化要说清楚：`benchmarks/benchmark_serving.py`、`benchmark_throughput.py`、`benchmark_latency.py` 三个文件在 v0.28.0 检出里**仍然存在，但只是弃用桩**——运行它会打印 "DEPRECATED: This script has been moved to the vLLM CLI. Please use the following command instead: vllm bench serve" 然后 `sys.exit(1)`。实现已经搬到 `vllm/benchmarks/`（`serve.py`、`throughput.py`、`latency.py`、`startup.py`、`sweep/`、`mm_processor.py`），CLI 入口在 `vllm/entrypoints/cli/benchmark/`（同名文件）。所以 v0.28.0 的正确命令是：

```bash
# vllm 已安装的环境
vllm bench latency    --model <model> --input-len 512 --output-len 128 --batch-size 8
vllm bench throughput --model <model> --dataset-name random --num-prompts 500
vllm bench serve      --model <model> --dataset-name sharegpt --request-rate 4   # 需先 vllm serve
vllm bench startup    --model <model>
vllm bench sweep      # 参数扫描，见 vllm/benchmarks/sweep/
```

`benchmarks/README.md` 也已改写为 "This directory used to contain vLLM's benchmark scripts"，指向文档站的 Benchmark CLI 页面。如果读到的旧博客或旧 PR 描述里还在用 `python benchmarks/benchmark_serving.py`，那是 v0.28.0 之前的用法。

`docs/contributing/profiling.md` 说明了 `vllm bench serve --profile` 与 `vllm bench latency` 配合 profiler 的用法，性能 PR 里附一段 profile 结论（哪个 kernel 占比从多少降到多少）比单纯的吞吐数字更有说服力。

### 4. 数字怎么呈现

放在 PR 描述的 Test Result（vLLM）或 Summary（PyTorch）里，用表格：

```text
| Case                          | Before (us) | After (us) | Speedup |
|-------------------------------|------------:|-----------:|--------:|
| H100, bf16, [4096, 4096]      |       123.4 |       98.7 |   1.25x |
| H100, bf16, [128, 4096]       |        12.1 |       12.0 |   1.01x |
| A100, fp16, [4096, 4096]      |       201.3 |      165.2 |   1.22x |

Command: python benchmarks/kernels/benchmark_rmsnorm.py --dtype bfloat16
```

第二行那种"几乎没变"的 case 一定要保留——它告诉 reviewer 你测过小 shape，而且没有变慢。


## 五、本地 lint：CI 的第一道门

### 1. 为什么 lint 排在 CI 最前面

两个项目的 CI 都把 lint 放在最前面并且独立成一个任务：PyTorch 是 `.github/workflows/lint.yml`（workflow 名 `Lint`，也是 `merge_rules.yaml` 里每条规则 `mandatory_checks_name` 的一项），vLLM 是 `.github/workflows/pre-commit.yml`。lint 红了，reviewer 看到的第一个信号就是红色，很多人会直接跳过。本地跑一遍只要几十秒到几分钟，是这条流程里性价比最高的一步。

### 2. PyTorch：lintrunner 与 spin

PyTorch 的 lint 由 `lintrunner` 驱动，配置在根目录 `.lintrunner.toml`。检出里有 61 个 `[[linter]]` 段，`code` 字段就是 CI 报错时显示的名字：

```text
格式与风格   FLAKE8 RUFF PYFMT CLANGFORMAT CODESPELL NEWLINE SPACES TABS COPYRIGHT
类型         PYREFLY TYPEIGNORE TYPENOSKIP NOQA
C++          CLANGTIDY CLANGTIDY_EXECUTORCH_COMPATIBILITY INCLUDE PYBIND11_INCLUDE PYBIND11_SPECIALIZATION
             C10_UNUSED C10_NODISCARD RAWTHROW CUBINCLUDE RAWCUDA RAWCUDADEVICE CALL_ONCE ONCE_FLAG
             SCOPED_LIBRARY HEADER_ONLY_LINTER ATEN_CPU_GPU_AGNOSTIC
Python 语义  ERROR_PRONE_ISINSTANCE ISINSTANCE_FAKE_TENSOR EXEC ROOT_LOGGING DEPLOY_DETECTION
             CONTEXT_DECORATOR SET_LINTER DOCSTRING_LINTER IMPORT_LINTER META_NO_CREATE_UNBACKED SYMPY_MINMAX
仓库结构     NATIVEFUNCTIONS GHA WORKFLOWSYNC NO_WORKFLOWS_ON_FORK ACTIONLINT SHELLCHECK CMAKE CMAKE_MINIMUM_REQUIRED
             PYPROJECT PYPIDEP LINTRUNNER_VERSION MERGE_CONFLICTLESS_CSV TESTOWNERS CODEOWNERS_TAXONOMY
测试         TEST_HAS_MAIN TEST_DEVICE_BIAS
稳定 ABI     STABLE_SHIM_VERSION STABLE_SHIM_USAGE GENERATED_SHIMS_VERSION UNSPECIFIED_BACKEND GB_REGISTRY
```

其中 12 个标了 `is_formatter = true`，可以自动修复。很多 linter 是 PyTorch 自己写的（`tools/linter/adapters/` 下），检查的是项目特有的约定：`NATIVEFUNCTIONS` 检查 `native_functions.yaml` 的格式，`WORKFLOWSYNC` 检查 `.github/workflows/` 是否与模板同步，`RAWCUDA` 禁止裸的 `cudaXxx` 调用。这解释了为什么"我本地 flake8 过了"不够——CI 跑的是这 61 个。

v2.14.0 推荐的入口是 `spin`（`CONTRIBUTING.md` 的 "Spin" 一节与 "Linting before committing" 一节）。`pyproject.toml` 的 `[tool.spin.commands]` 注册了这些子命令：`develop`、`editable`、`install`、`clean`、`lint`、`fixlint`、`quicklint`、`quickfix`、`regenerate-version`、`regenerate-type-stubs`、`regenerate-clangtidy-files`、`regenerate-github-workflows`、`docs`、`pyrefly`，实现都在 `.spin/cmds.py`。CONTRIBUTING 的 "Linting" 表格列出的是 `lint`、`quicklint`、`quickfix` 三个；`fixlint` 在表格里没有，但 `pyproject.toml` 里注册了，三个 PR 模板的 checklist 也都写着 `Passes lint (spin fixlint)`，根目录 `AGENTS.md` 的 "Linting" 一节说 "use `spin lint` as to run the lint and `spin fixlint` to apply automatic fixes"。以 `pyproject.toml` 为准，四个都能用：

```bash
# pytorch 仓库根目录
pip install spin            # 或 uv tool install spin --with=packaging,pyyaml,typing_extensions
spin lint                   # 默认 lint：快的 linter 跑全部文件，慢的只跑改动文件
spin fixlint                # 同上并自动修复
spin quicklint              # 只看最近一个 commit + 工作区改动
spin quickfix               # 同上并自动修复
spin quicklint -- --take CLANGTIDY   # 双横线后的参数直传 lintrunner
lintrunner                  # 直接用 lintrunner：相对 merge-base 的改动文件
lintrunner -a               # 并应用修复
```

"default lint" 小节解释了 `spin lint` 的策略："we categorize all linters as either fast or slow. In the default lint, only the fast linters are run on all files; the slow linters are run on the changed files only."——所以本地 `spin lint` 绿了、CI 的 `lintrunner-clang` 还可能红（clang-tidy 是慢 linter，CI 跑得更全），这时按 CI 日志里的 linter 名字用 `--take` 单跑即可。

### 3. vLLM：pre-commit

vLLM 用 `pre-commit`，配置在根目录 `.pre-commit-config.yaml`。`default_install_hook_types` 是 `pre-commit` 和 `commit-msg` 两种，`default_stages` 是 `pre-commit`（本地）和 `manual`（CI）。检出里的 hook id：

```text
外部仓库   ruff-check（带 --fix）ruff-format typos clang-format markdownlint-cli2 actionlint
           pip-compile（cuda / rocm / xpu / cpu / docs 五个变体）check-json
本地脚本   format-torch-nightly-test
           mypy-3.10（本地）mypy-3.11 mypy-3.12 mypy-3.13（stages: [manual]，只在 CI）
           shellcheck png-lint signoff-commit（stages: [commit-msg]）check-spdx-header
           check-root-lazy-imports check-filenames update-dockerfile-graph test-nonroot-entrypoint
           check-forbidden-imports check-torch-cuda-call validate-config validate-docker-versions
           check-boolean-context-manager rust-cargo-autoinherit rust-cargo-sort rust-cargo-fmt
末尾       suggestion（只打印一条提示：--no-verify 跳过全部，SKIP=<hook-id> 跳过某个）
```

`signoff-commit` 值得单独看：它挂在 `commit-msg` 阶段，用 `bash -c` 检查 `COMMIT_EDITMSG` 里有没有 `Signed-off-by: <git user.name> <git user.email>`，没有就追加一行。也就是说装好 pre-commit 之后忘了 `-s` 也不会漏签——但前提是 `pre-commit install` 装了 `commit-msg` 类型的钩子（默认会）。`check-spdx-header` 只在本地跑（"Only run locally as Buildkite will cover this"）；四个 `mypy-3.x` 中只有 3.10 本地跑，其余三个是 `manual` 阶段。

命令（`docs/contributing/README.md` "Linting" 与 `AGENTS.md` "Running linters"，在仓库根目录执行）：

```bash
uv pip install -r requirements/lint.txt      # 或 uv pip install pre-commit>=4.5.1
pre-commit install                           # 装钩子，之后每次 commit 自动跑
pre-commit run                               # 只跑 staged 文件
pre-commit run --all-files                   # 全部文件（-a）
pre-commit run ruff-check --all-files        # 单个 hook
pre-commit run mypy-3.12 --all-files --hook-stage manual   # 按 CI 的方式跑 mypy
```

CI 侧 `.github/workflows/pre-commit.yml` 用 `pre-commit/action` 跑 `--all-files --hook-stage manual`，所以本地想完全对齐 CI 就加 `--hook-stage manual`。`AGENTS.md` 还写了两条风格约定：Python 行宽 88；docstring 用 Google 风格（`Args:` / `Returns:` / `Raises:`），不用 reStructuredText 的 `:param:`。

### 4. 两套工具链对照

```text
                PyTorch                                     vLLM
驱动            lintrunner（.lintrunner.toml）               pre-commit（.pre-commit-config.yaml）
入口            spin lint / spin fixlint / lintrunner -a      pre-commit run [-a] [<hook>]
自动触发        无（手动跑；AGENTS.md 要求 commit 前 lintrunner -a）  pre-commit install 后每次 commit
检查数量        61 个 linter，多数项目自研                    ~35 个 hook，多数是外部工具 + tools/pre_commit/ 脚本
类型检查        PYREFLY（pyrefly check，配置 pyrefly.toml）    mypy-3.10 本地；3.11–3.13 只在 CI
签名检查        无（CLA 在 GitHub check）                      signoff-commit 钩子自动补 Signed-off-by
CI 任务         .github/workflows/lint.yml：lintrunner-clang / lintrunner-pyrefly / lintrunner-noclang / quick-checks / pr-sanity-checks / workflow-checks …   .github/workflows/pre-commit.yml：pre-run-check + pre-commit
CI 何时跑       每个 PR 自动                                   需要 verified / ready / ready-run-all-tests 标签，或作者已有 ≥4 个合入 PR
```

最后一行是 vLLM 新贡献者最容易被绊倒的地方，下面第七章展开。


## 六、PR 描述与签名

### 1. 描述是 reviewer 的第一屏

reviewer 打开 PR 先看描述，再看 diff。描述的任务是让他在读 diff 之前就知道：改了什么、为什么、怎么验证的、有没有风险。两个项目都用模板把这四个问题变成了固定的栏目；填模板不是形式主义，而是保证四个问题都有答案。

### 2. PyTorch：三个模板

`.github/PULL_REQUEST_TEMPLATE/` 下有三个文件，开 PR 时通过 URL 参数 `?template=fix_issue.md` 选择。每个模板顶部都有同样的两行提醒：读 wiki 的 "The Ultimate Guide to PyTorch Contributions" 和 `AI_POLICY.md`。

`fix_issue.md`（修 issue 的 PR）的字段：

```text
## Issue
Fixes #        ← 注释：Issue number. PRs without a linked issue may be automatically closed.
## Summary
               ← 注释：Point to the issue for relevant design discussion. If not discussed there (rare),
                  add up to one paragraph. Overly verbose descriptions will be considered spam.
## Checklist
- [ ] Passes lint (`spin fixlint`)
- [ ] Added/updated tests
- [ ] Updated documentation (if applicable)
- [ ] Included benchmark results (for PRs impacting perf)
## BC-breaking?
               ← 注释：If this change breaks backward compatibility, describe the impact and migration path. Otherwise, write "No".
```

两句注释是这个模板的灵魂："PRs without a linked issue may be automatically closed"——没有 issue 的 PR 可能被自动关闭；"Overly verbose descriptions will be considered spam"——过长的描述被视为 spam。设计讨论应该在 issue 里，PR 描述只是指向它。

`docs_typo.md` 只有一个字段 `## What changed?`（"One sentence describing the change."），顶部加粗强调 "This template is only for documentation or typo changes. Do not use it for bug fixes or changes to code behavior, as those require an open issue first."

`preapproved.md` 用于已经和 maintainer 私下或在别处谈好的改动：`## Approved by` 填 maintainer 的 handle，`## Summary` 写 "Point to where the pre-approval discussion happened"，其余与 `fix_issue.md` 相同。

三个模板都没有 "Test Plan" 栏——PyTorch 假设测试在 diff 里，reviewer 直接看 `test/` 下的改动。但根目录 `AGENTS.md` 的 "Commit messages" 一节要求 commit message "have a Test Plan section that describes how you tested the change"，并且 "include the literal commands that were run in fenced Markdown code blocks"；用 `ghstack` 时 commit message 就是 PR 描述，所以 Test Plan 实际上还是要写。

PyTorch 对标题没有前缀要求，但合入时要求 PR 有一个 `release notes:` 开头的标签或 `topic: not user facing` 标签（第十章讲）。标题本身会成为 release notes 的一行，所以要写成"做了什么"而不是"修了 #12345"。

### 3. vLLM：一个模板加标题前缀

`.github/PULL_REQUEST_TEMPLATE.md` 只有三个栏目：

```text
## Purpose
## Test Plan
## Test Result
```

折叠的 checklist 解释了每栏要写什么：Purpose 是 "Fix some issue (link existing issues this PR will resolve)"；Test Plan 是 "providing test command"；Test Result 是 "pasting the results comparison before and after, or e2e results"；可选一项是文档更新（"such as updating `supported_models.md` and `examples` for a new model"）。模板末尾那行 "BEFORE SUBMITTING, PLEASE READ …"以及所有 HTML 注释会被 `.github/workflows/new_pr_bot.yml` 在 PR 打开时自动删掉，所以不必手工清理；同一个 workflow 还会给第一次贡献的作者留一条欢迎评论，里面写明了 `/ci run` 的规则。

标题前缀是 vLLM 的分派机制。`docs/contributing/README.md` "PR Title and Classification" 一节开头就说 "Only specific types of PRs will be reviewed."，然后列出：

```text
[Bugfix]            bug 修复
[CI/Build]          构建或 CI
[Doc]               文档
[Model]             新模型或改进已有模型；模型名要出现在标题里
[Frontend]          OpenAI API server、LLM 类等前端
[Kernel]            CUDA kernel 或其他计算 kernel
[Core]              核心逻辑（LLMEngine、AsyncLLMEngine、Scheduler 等）
[Hardware][Vendor]  硬件相关，厂商名进前缀，如 [Hardware][AMD]
[Misc]              其他；"Please use this sparingly"
```

跨多类就并列多个前缀。前缀不只是分类：`.github/mergify.yml` 里有 35 条规则，多数是按文件路径或标题正则自动打标签（`label-frontend` 匹配 `vllm/entrypoints/`，`label-bug` 匹配标题里的 `bug`/`bugfix`，`label-documentation` 匹配 `docs/`、`examples/` 与根目录 `.md` 并留言文档预览地址，`label-ci-build` 匹配 `.github/`、`.buildkite/`、`cmake/`、`setup.py` 等），部分规则直接指派 reviewer（如 tensorizer 相关文件指派给一位固定 maintainer）。标题和路径决定了谁会看到你的 PR。

### 4. CLA 与 DCO

两个项目的法律签名机制不同，操作上的差别很大：

```text
            PyTorch：CLA                                   vLLM：DCO
做什么      第一次开 PR 时 EasyCLA 机器人引导签一次协议       每个 commit 的 message 末尾带 Signed-off-by: Name <email>
怎么做      按机器人链接在线签署；公司员工需公司先签         git commit -s；或 pre-commit 的 signoff-commit 钩子自动补
在哪检查    GitHub check "EasyCLA"；merge_rules.yaml 每条规则的 mandatory_checks_name 都含 EasyCLA   GitHub check "dco"；mergify 的 comment-dco-failure 规则在失败时留言
忘了怎么办  签一次即可，历史 commit 不用改                   要 rebase 重写每个没签的 commit：git rebase --exec 'git commit --amend --no-edit -s' main
文件        （wiki）                                         根目录 DCO 文件；docs/contributing/README.md "DCO and Signed-off-by"
```

vLLM 的 "DCO and Signed-off-by" 一节原文："Commits must include a `Signed-off-by:` header which certifies agreement with the terms of the DCO. Using `-s` with `git commit` will automatically add this header."，并给了 PyCharm 与 VSCode（`git.alwaysSignOff`）的自动签名设置。`Signed-off-by` 的名字和邮箱必须与 commit 的作者一致，否则 DCO check 仍会失败——这是用公司邮箱配置 git 但用个人账号推送的人常踩的坑。


## 七、CI 矩阵：什么会跑、什么不会

### 1. 两种哲学

PyTorch 的 CI 是**推送即跑、分层触发**：每个 PR 自动跑一组（`pull`），更重的组（`trunk`、`periodic`、`slow`、`inductor`）在 main 上、按周期或按标签跑。vLLM 的 CI 是**默认不跑、按需触发**：PR 上只有 pre-commit（还有前置条件），测试任务要有人敲 `/ci run`，且只跑与改动文件相关的任务。前者用机器换时间，后者用人的判断换机器。理解这个差别，才能读懂两边"CI 是绿的"分别意味着什么。

### 2. PyTorch：workflow 家族与 ciflow/*

`.github/workflows/` 在 v2.14.0 检出里有 148 个文件。以下划线开头的是可复用的子 workflow（`_linux-build.yml`、`_linux-test.yml` 等），其余是顶层 workflow。与贡献者直接相关的六个：

```text
文件            触发（on: 节，节选）                                                  含义
pull.yml        pull_request；push main / release/* / landchecks/*；tags ciflow/pull/*   每个 PR 必跑；merge_rules 里叫 "pull"
lint.yml        pull_request；push main / release/*；tags ciflow/pull/* ciflow/trunk/*     每个 PR 必跑；merge_rules 里叫 "Lint"
trunk.yml       push main / release/* / landchecks/*；tags ciflow/trunk/*；schedule       main 上跑；PR 上要打 ciflow/trunk 标签
periodic.yml    schedule（工作日每 8 小时等）；tags ciflow/periodic/*；push release/*     周期跑；PR 上要打 ciflow/periodic
slow.yml        push main / release/*；tags ciflow/slow/*；schedule                       慢测试；PR 上要打 ciflow/slow
inductor.yml    push main / release/*；tags ciflow/inductor/*                             Inductor 全量；PR 上要打 ciflow/inductor
```

`pull.yml` 自身有十个左右顶层 job 定义，但通过 `uses: ./.github/workflows/_linux-build.yml` / `_linux-test.yml` 等复用二十多次，实际展开为几十个 build/test 组合（不同 Python、CUDA、编译器、平台）。一个 PR 的 checks 页面因此有几百个条目。检出根目录 `AGENTS.md` 专门提醒："A PR has hundreds of check-runs, so a single `check-runs?per_page=100` call silently truncates and makes red look green. Use `gh pr checks <PR> --json name,state,workflow,link,bucket,completedAt`"。

`ciflow/*` 是 PyTorch 让 PR 跑"非默认"任务的机制。给 PR 打上 `ciflow/trunk` 标签，机器人会给 PR 的 head commit 打一个 `ciflow/trunk/<PR号>` 的 git tag，`trunk.yml` 的 `on.push.tags: ciflow/trunk/*` 就被触发。哪些标签有效，写在 `.github/pytorch-probot.yml` 的 `ciflow_push_tags` 列表里，v2.14.0 有 49 个，可以分成几组：

```text
基础          ciflow/pull ciflow/trunk ciflow/periodic ciflow/slow ciflow/nightly ciflow/unstable
硬件          ciflow/h100 ciflow/h100-distributed ciflow/h100-symm-mem ciflow/h100-cutlass-backend
              ciflow/b200 ciflow/b200-distributed ciflow/b200-symm-mem ciflow/mps ciflow/xpu ciflow/s390 ciflow/riscv64 ciflow/win-arm64
ROCm          ciflow/rocm-mi200 ciflow/rocm-mi300 ciflow/rocm-navi31 ciflow/rocm-nightly ciflow/rocm-preview
              ciflow/periodic-rocm-mi200 ciflow/periodic-rocm-mi300 ciflow/slow-rocm-mi200
Inductor      ciflow/inductor ciflow/inductor-periodic ciflow/inductor-cu126 ciflow/inductor-pallas
              ciflow/inductor-micro-benchmark ciflow/inductor-micro-benchmark-cpu-x86 ciflow/inductor-perf-compare
              ciflow/inductor-perf-test-nightly-{rocm-mi300,rocm-mi350,x86-zen,xpu} ciflow/inductor-rocm-mi200 ciflow/inductor-rocm-mi300
其他          ciflow/binaries ciflow/binaries_libtorch ciflow/binaries_wheel ciflow/triton_binaries ciflow/docker
              ciflow/dtensor ciflow/torchtitan ciflow/tsan ciflow/op-benchmark
```

打标签需要写权限，外部贡献者要在 PR 里请 reviewer 帮忙："Could you add `ciflow/trunk` so the ROCm jobs run?"。同一个文件的 `retryable_workflows`（`pull`、`trunk`、`linux-binary`、`windows-binary`、`inductor-A100-perf-nightly`）说明哪些 workflow 失败后可以让机器人重试；`mergebot: true` 表示合入机器人启用。

一个改动应该请求跑哪些标签，按改动的文件判断：改了 `aten/src/ATen/native/cuda/` 下的 kernel，至少要 `ciflow/trunk`（覆盖更多 CUDA 配置），涉及 bf16 或新架构再加 `ciflow/h100`；改了 `torch/_inductor/`，要 `ciflow/inductor`；改了分布式，`ciflow/h100-distributed`。`CONTRIBUTING.md` "Run Specific CI Jobs" 一节还提供了另一条路：`python tools/testing/explicit_ci_jobs.py --filter-gha '*pull*' --make-commit` 生成一个只保留匹配 workflow 的 commit，配合 `ghstack` 用；但它明说 "It creates a large commit that is of very low signal to reviewers"，只适合调试 CI 本身。

### 3. vLLM：Buildkite、test_areas、source_file_dependencies

vLLM 的 CI 跑在 Buildkite 上（日志公开，不需登录）。v0.28.0 检出里 `.buildkite/test-pipeline.yaml` 只剩一段注释："This file has been deprecated as of Feb 18, 2026. The content has already been migrated to: .buildkite/test_areas for test jobs, .buildkite/image_build for image building jobs, .buildkite/hardware_tests for jobs running on other hardwares (Intel, Ascend NPU, Arm, etc..), .buildkite/ci_config.yaml for configuration of CI pipeline"。

`.buildkite/test_areas/` 有 35 个文件，一个文件一个领域：

```text
attention basic_correctness benchmarks compile cuda disaggregated disaggregated_mooncake distributed docker
e2e_integration engine entrypoints expert_parallelism fault_tolerance jit_monitor kernels lm_eval lora misc
model_executor model_runner_v2 models_basic models_distributed models_language models_multimodal plugins pytorch
quantization ray_compat rust_frontend rust_frontend_cargo samplers spec_decode torch_abi weight_loading
```

每个文件是一个 `group` 加一组 `steps`。以 `kernels.yaml` 的一个 step 为例（原文节选）：

```yaml
# vllm .buildkite/test_areas/kernels.yaml（节选）
group: Kernels
depends_on:
  - image-build
steps:
- label: Kernels Core Operation Test
  device: h200_35gb
  key: kernels-core-operation-test
  timeout_in_minutes: 120
  source_file_dependencies:
  - csrc/
  - tests/kernels/core
  - tests/kernels/test_concat_mla_q.py
  - tests/kernels/test_fused_qk_norm_rope_gate.py
  commands:
    - pytest -v -s kernels/core --ignore=kernels/core/test_minimax_reduce_rms.py kernels/test_concat_mla_q.py kernels/test_fused_qk_norm_rope_gate.py --shard-id=$$BUILDKITE_PARALLEL_JOB --num-shards=$$BUILDKITE_PARALLEL_JOB_COUNT
  parallelism: 3
```

字段含义（统计全部 35 个文件出现频次：`commands` 209 次、`key` 202、`timeout_in_minutes` 196、`source_file_dependencies` 190、`device` 169、`num_devices` 90、`working_dir` 80、`optional` 63、`mirror` 58、`parallelism` 17、`soft_fail` 5）：

```text
字段                        含义
label / key                 显示名与唯一键；gh pr checks 里看到的名字来自 label
device                      跑在什么 GPU 上：h100 / h200_18gb / h200_35gb / b200-k8s 等（带显存后缀的是共享切分的实例）
num_devices                 需要几张卡（多卡分布式测试）
source_file_dependencies    改了这些路径下的文件才触发这个 step；PR 的 CI 只跑与 diff 相交的 step
commands                    在 /vllm-workspace/tests 下执行的命令；$$ 是 Buildkite 的转义
parallelism                 分片数，配合 --shard-id / --num-shards
optional                    默认不跑，需要 ready-run-all-tests 或 /ci run all
soft_fail                   失败不阻塞
mirror                      同时在另一硬件上镜像运行
timeout_in_minutes          超时
```

`source_file_dependencies` 是理解 vLLM CI 的关键：一个只改了 `vllm/entrypoints/openai/` 的 PR 不会触发 `kernels.yaml` 里的任何 step。反过来，`.buildkite/ci_config.yaml` 的 `run_all_patterns` 列出了"改了就全跑"的文件——`docker/Dockerfile`、`CMakeLists.txt`、`requirements/common.txt`、`requirements/cuda.txt`、`setup.py`、`csrc/`、`cmake/` 等（`run_all_exclude_patterns` 再排除 `csrc/cpu/`、`csrc/rocm/` 等）。改一行 `csrc/` 下的 kernel 就会触发全量 CI，这是 kernel PR 周期长的一个原因。同一文件的 `job_dirs` 指向 `.buildkite/image_build`、`.buildkite/test_areas`、`.buildkite/hardware_tests` 三个目录，`repositories` 区分 `premerge`（PR）与 `main`（合入后）的镜像仓库。

### 4. vLLM：谁能让 CI 跑

这是 vLLM 与 PyTorch 差别最大的一点。`docs/contributing/README.md` "What to Expect for the Reviews" 的最后一条：

> Note that not all CI checks will be executed due to limited computational resources. Reviewers with write access and configured trusted contributors can comment `/ci run` when CI signals are needed before a PR is ready. After the PR is approved or has the `ready` label, the PR author can use `/ci run` or `/ci retry`. New commits do not start CI automatically.

实现是 `.github/workflows/run-ci-command.yml`：监听 `issue_comment`，评论内容严格等于 `/ci run`、`/ci run all`、`/ci run nightly`、`/ci retry`、`/ci cancel` 之一才触发（`.github/workflows/scripts/test_run_ci_command.py` 里有测试断言 `/ci run please` 和带前导空格的 ` /ci run` 都不算）。授权逻辑在 `.github/workflows/scripts/run_ci_command.py` 的 `authorize` 函数，按顺序判断：

```text
1  评论者有 admin / maintain / write 权限（TRUSTED_PERMISSIONS）        → 允许
2  评论者在 CI_TRUSTED_USERS 变量列出的受信贡献者名单里                 → 允许
3  评论者不是 PR 作者                                                  → 拒绝："Only reviewers with write access can use CI commands before CI is delegated to the PR author."
4  PR 是 draft                                                        → 拒绝："PR authors cannot run CI while the PR is a draft."
5  PR 有 ready 或 ready-run-all-tests 标签（READY_LABELS）              → 允许："ready label"
6  PR 有受信 reviewer 的 approval                                      → 允许："approval from a trusted reviewer"
7  否则                                                                → 拒绝："A reviewer with write access must run `/ci run`, approve the PR, or add the `ready` label first."
```

所以新贡献者的 PR 流程是：开 PR → 自己什么都触发不了 → reviewer 看过一眼后敲 `/ci run` 或打 `ready` → 之后作者可以自己 `/ci run` / `/ci retry`。**每次 push 新 commit 之后都要重新敲**——"New commits do not start CI automatically"。`ready` 标签的描述（截至 2026-09 查询）是 "ONLY add when PR is ready to merge/full CI is needed"；`ready-run-all-tests` 是 "Trigger CI with all tests for wide-ranging PRs"，会把 `optional: true` 的 step 也跑起来。

pre-commit 也有门槛。`.github/workflows/pre-commit.yml` 的 `pre-run-check` job 要求 PR 有 `verified`、`ready` 或 `ready-run-all-tests` 标签之一，或作者已有至少 4 个合入的 PR（"PR must have the 'verified', 'ready', or 'ready-run-all-tests' label to run pre-commit, or the author must have at least 4 merged PRs"）。`verified` 标签的描述是 "Run pre-commit for new contributors without triggering other tests"。这意味着第一次贡献者连 lint 都要等人打标签——所以本地 `pre-commit run --all-files` 不是可选项。

### 5. 对照表

```text
                    PyTorch                                                 vLLM
系统                GitHub Actions                                          Buildkite（+ GitHub Actions 跑 pre-commit 与机器人）
配置位置            .github/workflows/*.yml（148 个）                        .buildkite/test_areas/*.yaml（35 个）+ ci_config.yaml
按领域组织          按 workflow（pull / trunk / periodic / slow / inductor …）  按 test_area 文件（kernels / entrypoints / distributed …）
按改动文件选任务    无（pull 全跑；filter_test_configs.py 有少量按标签过滤）   source_file_dependencies 精确到路径；run_all_patterns 例外
PR 上默认跑什么     pull + Lint（几十个 job）                                  只有 pre-commit（且需 verified/ready 或 ≥4 合入 PR）
额外任务怎么要      reviewer 打 ciflow/<xxx> 标签                              reviewer 敲 /ci run 或打 ready；之后作者可自己敲
push 后自动重跑     是                                                       否，需再敲 /ci run
硬件声明            在 workflow 的 runner 标签里（间接）                        step 的 device / num_devices 字段（直接）
重试                GitHub Actions 的 re-run；pytorch-probot.yml 的 retryable_workflows 列出可自动重试的 workflow   /ci retry（只重跑失败 job）
日志                GitHub Actions 日志 + HUD                                  Buildkite 公开日志；.buildkite/scripts/ci-fetch-log.sh
```


## 八、读 CI 日志：这是我的问题吗

### 1. 先回答一个问题

CI 红了，第一件事不是看日志，而是问：**main 上这个任务现在是绿的吗？**如果 main 也红，大概率不是你的问题；如果 main 绿而你红，才值得读日志。两个项目都提供了看 main 状态的地方，并且都在文档里把这一步写在最前面。

### 2. PyTorch：HUD 与 "Which commit is used in CI?"

`CONTRIBUTING.md` "CI failure tips" 一节的第二段：

> Fairly often, a CI failure might be unrelated to your changes. You can confirm by going to our HUD and seeing if the CI job is failing upstream already. In this case, you can usually ignore the failure.

HUD（hud.pytorch.org）按 commit 显示 main 上每个 job 的状态，也能看某个 PR 的所有 job；红色的 job 如果在 main 的最近几个 commit 上也红，就是 main 的问题。命令行上用 `gh pr checks <PR> --json name,state,workflow,link,bucket,completedAt` 拉全量状态（根目录 `AGENTS.md` 推荐的写法），再用 `gh run view <run-id> --log-failed` 只看失败步骤的日志。

同节的子节 "Which commit is used in CI?" 解释了一个容易困惑的细节：PR 的 CI 通常跑在 PR 的 head commit（B）上，但 **workflow 文件本身**取自 PR 与 main 的合并结果（C）。所以如果 main 上改了 workflow，你的 PR 会用新的 workflow 跑旧的代码；文中也注明 `ghstack` 的 PR 不受此影响（"they would not automatically ingest the updates from default branch"）。当日志里出现你没见过的 job 名或步骤，先想到这一点。

PyTorch 的失败日志有固定的结构：每个 test job 末尾会打印失败测试的列表和 `To execute this test, run the following from the base repo dir: python test/test_xxx.py TestClass.test_name` 的复现命令。把这行复制到本地跑，是定位问题最快的路。如果是 flaky（重跑就过），PyTorch 有一套 disable/flaky 机制（`test/slow_tests.json` 由 `pytorchbot` 自动更新，`merge_rules.yaml` 里有专门的 "OSS CI / pytorchbot / slow tests" 规则），不要在自己的 PR 里手动 skip 别人的测试。

### 3. vLLM：failures.md、Dashboard 与 ci-fetch-log.sh

`docs/contributing/ci/failures.md` 开头就是这个问题："What should I do when a CI job fails on my PR, but I don't think my PR caused the failure?"，答案分三步：

1. 看 CI Failures Dashboard（GitHub Projects 20）；
2. "If your failure **is already listed**, it's likely unrelated to your PR."——在已有 issue 下留言附上你的实例链接，点 👍；
3. "If your failure **is not listed**, you should **file an issue**."——用 `450-ci-failure.yml` 模板，标题格式 `[CI Failure]: failing-test-job - regex/matching/failing:test`，环境栏写 `Still failing on main as of commit abcdef123`。

拉日志用 "Logs Wrangling" 一节的脚本 `.buildkite/scripts/ci-fetch-log.sh`（日志公开，无需 Buildkite 登录）：

```bash
# vllm 仓库根目录
.buildkite/scripts/ci-fetch-log.sh --pr <PR>                                   # 当前 PR 最新 build 的所有失败 job
.buildkite/scripts/ci-fetch-log.sh "https://buildkite.com/vllm/ci/builds/<N>"  # 某个 build（--soft 含 soft-fail，--all 全部）
.buildkite/scripts/ci-fetch-log.sh "https://buildkite.com/vllm/ci/builds/<N>#<job_uuid>" -   # 单个 job 流到 stdout
.buildkite/scripts/ci-clean-log.sh ci.log                                      # 去掉时间戳和 ANSI 码
.buildkite/scripts/rerun-test.sh tests/v1/engine/test_engine_core_client.py::test_kv_cache_events[True-tcp]   # 循环重跑判断 flaky
```

"Investigating a CI Test Failure" 一节的方法就是二分：到 Buildkite 的 main 分支构建列表里找第一个出现该失败的 build，把发现写进 issue。如果你顺手修了这个 CI 失败，"Submitting a PR" 一节要求描述里写 `Closes #12345` 并打 `ci-failure` 标签（截至 2026-09 查询该标签存在，描述 "Issue about an unexpected test failure in CI"）。

### 4. 分辨"我的"与"main 的"：一张判断表

```text
现象                                          判断                              动作
main 上同一 job 最近几次也红                    main 的问题                       PyTorch：留言 "unrelated, failing on main (HUD link)"；vLLM：Dashboard 找到 issue 留言 +1
main 绿，我红，重跑一次就绿                     flaky，可能是我触发的也可能不是      重跑（PyTorch 请 reviewer re-run 该 job / vLLM /ci retry）；连续两次红就当自己的问题
main 绿，我红，失败测试在我改的文件附近          我的问题                          本地按日志里的复现命令跑；修；push；vLLM 再敲 /ci run
main 绿，我红，失败在完全无关的模块              可能是我的改动有非局部影响          先读 traceback 找到调用链是否经过我的 diff；确实无关再当 flaky 处理
job 根本没跑（灰色 / skipped）                  不是失败                          PyTorch：该 workflow 需要 ciflow 标签；vLLM：source_file_dependencies 不相交，或没人敲 /ci run
lint / pre-commit 红                           一定是我的问题                     本地 spin fixlint / pre-commit run -a，修完再推
DCO / EasyCLA 红                               一定是我的问题                     vLLM：补签名并 force-push；PyTorch：按机器人链接签 CLA
```


## 九、review 往返

### 1. 两个项目的时间承诺

两个项目都在文档里写了明确的时间预期，这些数字决定了你什么时候可以催、什么时候还该等。

PyTorch `CONTRIBUTING.md` "Merging your Change" 一节：

> If not, leave the Reviewers section empty. Our triage squad will review your PR, add a module label, and assign it to the appropriate reviewer in a couple business days. The reviewer will then look at your PR and respond.
>
> Occasionally, things might fall through the cracks (sorry!). In case your PR either doesn't get assigned to a reviewer or doesn't get any response from the reviewer for 4 business days, please leave comment on the PR (mentioning the reviewer if one has been assigned). That'll get it nudged back onto people's radar.
>
> If that still doesn't help, come see us during our office hours

三层升级：几个工作日内 triage 打模块标签并分派 → 4 个工作日无回应留言（@ 被分派的 reviewer）→ 还不行去 Dev Infra Office Hours（"hosted every Friday"）。

vLLM `docs/contributing/README.md` "What to Expect for the Reviews" 一节，自称目标是 "a *transparent reviewing machine*"：

> - After the PR is submitted, the PR will be assigned to a reviewer. Every reviewer will pick up the PRs based on their expertise and availability.
> - After the PR is assigned, the reviewer will provide status updates every 2-3 days. If the PR is not reviewed within 7 days, please feel free to ping the reviewer or the vLLM team.
> - After the review, the reviewer will put an `action-required` label on the PR if there are changes required. The contributor should address the comments and ping the reviewer to re-review the PR.
> - Please respond to all comments within a reasonable time frame. If a comment isn't clear or you disagree with a suggestion, feel free to ask for clarification or discuss the suggestion.

也是三层：分派 → 每 2–3 天一次状态 → 7 天无 review 可以 ping。"Pull Request Limits and Escalation" 一节还给了一条加急通道：用可验证的公司或大学邮箱写信到 `pr-review-request@vllm.ai`，说明生产或研究用例、遇到的问题、改动怎么解决它。

```text
                  PyTorch                                   vLLM
分派              triage squad 打 module 标签、分派，"a couple business days"   自动分派 reviewer（mergify 规则 + 人工）
中途状态          无承诺                                    每 2–3 天一次
可以催的时点      4 个工作日无回应                           7 天无 review
怎么催            PR 里留言并 @ reviewer                     ping reviewer 或 vLLM team；Slack #pr-reviews（new_pr_bot 欢迎语里给了地址）
再升级            Dev Infra Office Hours（每周五）           pr-review-request@vllm.ai（需机构邮箱）
"需要你改"的信号   review 状态 Changes requested              action-required 标签（文档所写；截至 2026-09 查询仓库标签列表中未找到该标签，以当前仓库为准）
```

### 2. 怎么回应 review 意见

review 意见分几类，每类的正确回应不同。原则只有一条：**让 reviewer 下一次打开 PR 时，用最少的时间确认"我提的每一条都被处理了"**。

```text
意见类型                            正确回应                                                        错误回应
明确的修改要求（"rename X to Y"）    照做；在该 comment 下回 "Done"；不要解释为什么原来那样            争辩命名偏好；默默改了不回
指出 bug                            确认 → 修 → 加一个测试覆盖它 → 回复指向新测试                     只修不加测试；"good catch" 之后没有下文
要求拆分                            拆；在原 PR 留言指向新 PR 编号                                    解释"其实它们是相关的"
要求补 benchmark / 测试              补；数字放描述里；回复引用                                        "本地测过了没问题"
设计层面的异议                      先确认自己理解了对方担心的是什么，复述一遍；给出两种方案的取舍；如果坚持原方案，给出可验证的理由（数字、已有 issue、约束）  逐条反驳；或者立刻放弃改成对方说的而不问为什么
"nit:" 开头的小意见                  照做（成本极低，反而争辩成本高）                                  一条一条解释为什么不改
不清楚的意见                        问："do you mean A or B?"                                        猜一个改了
过时的意见（代码已改）              回复 "addressed in <commit>" 并标 resolved                        不理，让 reviewer 自己发现
```

几条操作细节：

- **一轮意见一次性处理完再 push**，不要每改一条 push 一次——PyTorch 每次 push 触发几十个 job，vLLM 每次 push 之后 CI 要重新敲；
- **不要 force-push 覆盖 review 过的历史**（PyTorch 的 `ghstack` 是例外，它本来就是 amend 模型）；vLLM 用新 commit 追加，合入时会 squash；
- **每一条 comment 都要有回复**，哪怕只是 "Done"；GitHub 的 "Resolve conversation" 按钮由作者点（vLLM）或 reviewer 点（PyTorch 习惯不一，跟随 reviewer）；
- **处理完后主动 ping**："@reviewer addressed all comments, PTAL"——vLLM 的文档明确写了 "ping the reviewer to re-review the PR"；
- **争辩要有依据**：数字、已有 issue、上游约束。没有依据就照做。


## 十、合入

### 1. PyTorch：@pytorchbot merge 与 merge_rules.yaml

PyTorch 的 PR 不由人点 "Merge" 按钮，而是由机器人合入。`CONTRIBUTING.md` "Merging your Change" 一节的最后一句："Once your PR is approved, you can merge it in by entering a comment with the content `@pytorchmergebot merge`"；`SECURITY.md` 与 `.github/workflows/nightly.yml` 里写的是 `@pytorchbot merge`——两个 handle 都能触发，实现是 `.github/workflows/trymerge.yml` 调用 `.github/scripts/trymerge.py`。

`trymerge.py` 的 `parse_args` 定义了参数：`--force`、`--ignore-current`、`--revert`、`--dry-run`、`--check-mergeability`、`--comment-id`、`--reason`。对应到评论里的写法：

```text
@pytorchbot merge                      默认：等所有 mandatory checks 通过后合入。explainer 的提示语："Your change will be merged once all checks pass (ETA 0-4 Hours)."
@pytorchbot merge -i                   --ignore-current：忽略当前已失败的 check，等 pending 的跑完再合。提示语："Your change will be merged while ignoring the following N checks: …"
@pytorchbot merge -f "<reason>"        --force：立即合入，绕过 CI。提示语："Your change will be merged immediately since you used the force (-f) flag, bypassing any CI checks (ETA: 1-5 minutes). Please use -f as last resort and instead consider -i/--ignore-current …"
@pytorchbot revert …                   --revert + --reason：回滚已合入的 PR（.github/workflows/revert.yml；需要权限与理由）
@pytorchbot rebase                     .github/workflows/tryrebase.yml → tryrebase.py，rebase 到 main（--branch 可指定分支）
@pytorchbot label "topic: not user facing"    打标签（label_utils.py 的错误提示里给出的例子）
```

`-f` 不是外部贡献者能用的——它需要权限且必须给理由。`trymerge.py` 里 `check_docker_builds_ready` 的注释说明了为什么 `-f` 被越来越多地限制："This gate is enforced even for force merges, since -f is exactly what bypassed it before."

合入前 `trymerge.py` 做四类检查，任何一类不过就在 PR 里留言拒绝：

**权限检查**——`.github/merge_rules.yaml`。v2.14.0 有 33 条规则，每条四个字段：

```yaml
# pytorch .github/merge_rules.yaml（节选）
- name: OSS CI
  patterns:
  - .github/**
  - .ci/**
  - scripts/**
  - tools/**
  approved_by:
  - alband
  - pytorch/pytorch-dev-infra
  mandatory_checks_name:
  - EasyCLA
  - Lint
  - pull
```

`patterns` 是文件 glob，`approved_by` 是有权批准这些文件改动的人或团队，`mandatory_checks_name` 是必须通过的 check（几乎所有规则都是 `EasyCLA`、`Lint`、`pull` 三项，个别加 `inductor` 或 `slow`）。`find_matching_merge_rule` 遍历所有规则，找一条**PR 改动的每个文件都匹配其 `patterns`、且至少一个 approver 在 `approved_by` 里**的规则。最后一条 `Core Maintainers` 的 `patterns` 是 `'*'`，`approved_by` 是核心维护者名单，作为兜底。这解释了一个常见困惑："我的 PR 被 approve 了为什么合不进"——approve 你的人不在你改动文件对应规则的 `approved_by` 里。拒绝信息会告诉你差哪条规则（`reject_reason` 按匹配文件数排序，只报最相关的一条）。

**标签检查**——`ensure_mergeable_labels` 要求 PR 有 `release notes:` 开头的标签或 `topic: not user facing`。`.github/scripts/label_utils.py` 的 `LABEL_ERR_MSG`：

> This PR needs a `release notes:` label. If your changes are user facing and intended to be a part of release notes, please use a label starting with `release notes:`. If not, please add the `topic: not user facing` label. To add a label, you can comment to pytorchbot, for example `@pytorchbot label "topic: not user facing"`

`release notes:` 标签由 `.github/labeler.yml` 按路径自动打（如 `release notes: quantization`、`release notes: distributed (dtensor)`），没自动打上的要自己判断后用 `@pytorchbot label` 加。

**CI 检查**——`mandatory_checks_name` 列出的 check 必须成功；其余 check 失败会阻塞默认合入但可以用 `-i` 忽略。

**全局检查**——`check_for_sev`：如果仓库有 open 的、同时带 `ci: sev` 和 `merge blocking` 标签的 issue，所有合入暂停（"Not merging any PRs at the moment because there is a merge blocking … issue open"）。遇到这个不是你的问题，等。

### 2. vLLM：ready 标签与 mergify

vLLM 的合入是人做的：有写权限的 maintainer approve 之后打 `ready` 标签、（通常）敲 `/ci run` 跑全量 CI、CI 绿了之后点 squash merge。外部贡献者在这一步没有任何操作，只需要保证：

- **没有冲突**：`.github/mergify.yml` 的规则 "ping author on conflicts and add 'needs-rebase' label" 在 PR 与 main 冲突时自动打 `needs-rebase` 并留言：

{% raw %}
```yaml
# vllm .github/mergify.yml（节选）
- name: ping author on conflicts and add 'needs-rebase' label
  conditions:
    - label != stale
    - conflict
    - -closed
  actions:
    label:
      add:
        - needs-rebase
    comment:
      message: |
       This pull request has merge conflicts that must be resolved before it can be
       merged. Please rebase the PR, @{{author}}.
```
{% endraw %}

  对应的 "remove 'needs-rebase' label when conflict is resolved" 规则在冲突解决后自动摘掉标签。vLLM 的 main 每天几十个 commit，一个 PR 挂一周基本必然要 rebase 一次；

- **pre-commit 与 DCO 绿**：mergify 的 `comment-pre-commit-failure` 规则在 PR 有 `ready` 或 `verified` 标签且 pre-commit 失败时留言，附上 `uv pip install pre-commit>=4.5.1` / `pre-commit install` / `pre-commit run --all-files` 三行命令；`comment-dco-failure` 在 DCO 失败时留言；

- **回应最后一轮意见**：`ready` 之后 CI 如果红了，作者可以自己 `/ci retry`；如果需要改代码，push 之后再 `/ci run`。

还有一条自动化：`.github/workflows/add_label_automerge.yml` 在 maintainer 对 PR 启用 GitHub 的 auto-merge 时自动打上 `ready` 标签——所以看到 `ready` 出现，通常意味着 maintainer 已经决定合入，只等 CI。

### 3. 对照

```text
              PyTorch                                                   vLLM
谁触发        作者或任何有权限的人评论 @pytorchbot merge                  有写权限的 maintainer 点 merge
权限来源      merge_rules.yaml 按文件路径 → approved_by                   GitHub 仓库写权限
必过 check    mandatory_checks_name（EasyCLA / Lint / pull …）            pre-commit / DCO / Buildkite（由 maintainer 判断）
必要标签      release notes: 或 topic: not user facing                     ready（maintainer 打）
绕过 CI       -f（需理由与权限）；-i 忽略已失败项                          maintainer 自行判断
冲突处理      @pytorchbot rebase；ghstack 自动                            mergify 打 needs-rebase 催作者
合入方式      机器人 push 到 main（保留 Pull Request resolved: 链接）       squash merge
全局暂停      ci: sev + merge blocking issue                              无自动机制
```


## 十一、被拒之后

### 1. 三类拒绝

不是所有拒绝都一样。按拒绝的对象分三类，各自的正确反应完全不同：

```text
类型        reviewer 在拒绝什么                典型措辞                                            正确反应
方向不对    这个问题本身不该这样解，或不该解     "we don't want to support this" / "this belongs in a plugin" / "closing as won't fix"   放弃这个 PR；如果确信有价值，回到 issue 或 RFC 层面重新讨论，不要改 PR 再提
时机不对    问题对、方向对，但现在不是时候       "let's wait for the refactor in #NNNN to land" / "this area is being rewritten" / "we're in release freeze"   保留分支；订阅被引用的 issue/PR；等条件满足后 rebase 重提，并在描述里引用当时的讨论
做法不对    问题对、方向对，实现有问题           "could you split this" / "needs a test" / "this breaks BC" / "use X instead of Y"   改；这是唯一应该"改了再提"的一类
```

前两类的信号是拒绝里**没有任何关于代码的具体意见**。如果 reviewer 谈的是"我们要不要做这件事"，无论你把代码写得多好都没用；继续 push 只会消耗双方的耐心。第三类的信号是意见都指向 diff 的具体位置，这时每一条都是可以完成的任务。

### 2. 几种特殊情况

- **被关闭但没有解释**：两个项目都有 stale 机制（vLLM `.github/workflows/stale.yml`；mergify 的多数规则带 `label != stale` 条件）。如果是 stale 关闭而不是人关闭，留言说明你仍在跟进并请求 reopen，通常会被接受；
- **被要求先开 issue**：PyTorch 的 `fix_issue.md` 模板注释 "PRs without a linked issue may be automatically closed"，`CONTRIBUTING.md` "AI-Assisted Development" 一节 "you should never send a PR that doesn't have a corresponding issue with the 'actionable' label"。这不是拒绝，是流程：开 issue，等 `actionable`（截至 2026-09 查询该标签存在），再重开 PR 并 `Fixes #`；
- **被打 `rfc-required`**（vLLM 文档所述）：改动超过 500 行架构代码。写 RFC issue（`.github/ISSUE_TEMPLATE/750-RFC.yml`），把 PR 转 draft 等讨论；
- **同一件事别人先合了**：关闭自己的，在对方 PR 下留言补充你发现的边界情况（如果有）。这是第二篇"查重"没做好的代价，不要在第三篇里补救。

### 3. 放弃也是一种结果

一个被明确拒绝方向的 PR 继续改下去，结果几乎总是更糟。正确的做法是在 PR 里留一句 "Understood, closing this. Filed #NNNN to track the underlying issue."，然后把学到的东西记进贡献日志：这个模块的 maintainer 是谁、他们在意什么、哪类改动他们不接。这份信息对下一个 PR 的价值，往往比这个 PR 本身高。


## 十二、AI 辅助贡献的项目政策

### 1. 为什么两个项目都在 2026 年把它写成了硬规则

AI 辅助生成的 PR 让"低质量 PR"的边际成本降到零，而 review 的成本没有变。两个项目的回应是在仓库根目录放一份面向人和 agent 的政策文件——PyTorch 是 `AI_POLICY.md`（`AGENTS.md`、`CLAUDE.md` 开头就是 "AI Policy — MANDATORY. Read `AI_POLICY.md`"），vLLM 是 `AGENTS.md`（`CLAUDE.md` 也在）。两份文件都不是禁止用 AI，而是把"谁负责"说清楚。

### 2. PyTorch：AI_POLICY.md 与 CONTRIBUTING.md

`AI_POLICY.md` 全文不长，五条规则（斜体是原文的强调）：

1. *AI-generated content in comments, issues, or PRs must be clearly disclosed and contained*（用代码块或引用块），并且 "must be accompanied by human commentary explaining its relevance"。给的例子是："Codex produced the following analysis: `<insert codex output here>`, so I believe that <insert human analysis here>"。唯一例外是 pytorchbot 自动化；
2. *Please do not respond to comments+questions by pasting raw or lightly reviewed AI-generated text.* 理由："Other users' comments+questions are requests for your understanding, reasoning, and judgment. Unreviewed AI output shifts the burden of verification onto the other party"；
3. *For pull requests, please carefully read the code before submitting it for review*——"Make sure the implementation is something you understand"，"AI-generated code can sometimes be overly complex, indirect, inconsistent, or include artifacts that obscure the main idea. Before submitting, simplify where possible"；
4. "If your PR is not ready for review, please use the GitHub draft PR feature."；
5. *We do not accept contributions created by fully autonomous agents*，"we may close pull requests that appear to have been generated without meaningful human involvement"。

`CONTRIBUTING.md` "AI-Assisted Development" 一节指向 `AI_POLICY.md`，另加三条"reminders"（加粗是原文）：

- **You are personally responsible for what you send**："If the comments, issues or PRs you send are low quality or consistently overly verbose compared to what is expected, your contributions will not be accepted anymore."
- **PRs must have an associated "actionable" Issue**："as a new contributor, you should never send a PR that doesn't have a corresponding issue with the 'actionable' label. If you just opened the issue, you must wait for a maintainer to review it and mark it actionable before preparing and sending a PR for it."
- **New features, utility functions, or core extensions**："Create a short and to the point issue about the problem you're encountering. You should NEVER include AI-generated explanation of how to solve the problem"。

三个 PR 模板顶部都链接了 `AI_POLICY.md`。根目录 `AGENTS.md` 面向 agent 本身，把同样的规则翻译成对 agent 的指令："You may never act autonomously on GitHub"、"Mark all AI-generated content"、"Never emit only raw AI text as a reply"、"Do not submit code the user hasn't read"，并在 "Commit messages" 一节要求 "Disclose that the PR was authored with an AI assistant."。

### 3. vLLM：AGENTS.md 与 docs/contributing/README.md

`AGENTS.md` 开头一行："Breaching these guidelines can result in automatic banning."。"1. Contribution Policy (Mandatory)" 分四小节：

- **Duplicate-work checks**：动手前必须跑三条 `gh` 命令（`gh issue view <n> --comments`、`gh pr list --state open --search "<n> in:body"`、`gh pr list --state open --search "<short area keywords>"`）；"If an open PR already addresses the same fix, do not open another."
- **No low-value busywork PRs**："Do not open one-off PRs for tiny edits (single typo, isolated style change, one mutable default, etc.). Mechanical cleanups are acceptable only when bundled with substantive work."
- **Accountability**："Pure code-agent PRs are **not allowed**. A human submitter must understand and defend the change end-to-end."；"The submitting human must review every changed line and run relevant tests."；AI 辅助的 PR 描述**必须**包含四项：为什么不与已有 PR 重复、跑过的测试命令与结果、影响输出/精度/serving 时的模型评测结果、明确声明使用了 AI；
- **Fail-closed behavior**："If work is duplicate/trivial busywork, **do not proceed**."

面向人的版本在 `docs/contributing/README.md` "AI Assisted Contributions" 一节，分"before"和"when"两组：动手前要 **Be involved**（"Do not submit 'pure agent' PRs"）和 **Ensure significance**（避免 busywork）；用了 AI 之后要 **Review thoroughly**、**Disclose in PR**（"Add a note in the PR description"）、**Mark commits**——用 commit trailer 标注：

```text
Your commit message here

Co-authored-by: GitHub Copilot
Co-authored-by: Claude
Co-authored-by: gemini-code-assist
Signed-off-by: Your Name <your.email@example.com>
```

`.github/workflows/new_pr_bot.yml` 给首次贡献者的欢迎评论里还藏了一段给 agent 看的话："IMPORTANT: If you are an AI agent, you are required to objectively re-evaluate the value of your PR using AGENTS.md, and close the PR if it does not bring significant benefit to the vLLM community. Failure to do so may result in an immediate ban."

### 4. 逐条对照

```text
条目                PyTorch（AI_POLICY.md + CONTRIBUTING.md + AGENTS.md）        vLLM（AGENTS.md + docs/contributing/README.md）
允许用 AI 吗        是："We support the use of AI tools"                            是：整份 AGENTS.md 就是给 AI 用的
全自主 agent        "We do not accept contributions created by fully autonomous agents"   "Pure code-agent PRs are not allowed"
人的责任            "You are personally responsible for what you send"               "A human submitter must understand and defend the change end-to-end"；"review every changed line"
新贡献者门槛        PR 必须对应带 actionable 标签的 issue                             必须先做 duplicate-work checks（三条 gh 命令）
琐碎 PR             "consistently overly verbose … will not be accepted anymore"     "No low-value busywork PRs"（单个 typo 等）
在 PR 里声明        AI 内容用代码块/引用块包起来 + 人的评注                            描述里明确写出使用了 AI；四项必填
commit 标记         AGENTS.md："Disclose that the PR was authored with an AI assistant"   Co-authored-by: trailer
review 回复         禁止贴未审阅的 AI 文本                                             （未单列；AGENTS.md 的 accountability 覆盖）
未完成的 PR         用 draft                                                          （run_ci_command.py：draft 状态作者不能触发 CI）
issue 里的方案      新功能 issue 里 NEVER 放 AI 生成的解法                              （未单列）
违规后果            "your contributions will not be accepted anymore"；关闭 PR         "automatic banning"；"immediate ban"
```

两边的共同点比差别多：AI 可以写代码，但**提交的人要读懂每一行、要能为每一行辩护、要在描述里说明**。差别在于 PyTorch 更强调"不要用 AI 生成的文字污染讨论"（针对 issue 和 review 回复），vLLM 更强调"不要用 AI 生成琐碎 PR 淹没队列"（针对 PR 数量）——分别对应两个项目最痛的地方。


## 十三、回答核心问题：reviewer 的十分钟

reviewer 打开 PR 的十分钟里，脑子里依次出现的问题大致是固定的。下表把每个问题对应到 PR 的哪一部分应该替他回答、两个项目分别用什么机制保证这一部分存在：

```text
分钟   reviewer 的问题                        谁来回答            PyTorch 的机制                                 vLLM 的机制
0–1    这是什么？该我看吗？                   标题 + 标签          release notes: / module: 标签（labeler.yml 自动） 标题前缀 [Kernel] 等 + mergify 自动标签
1–2    为什么要改？有人讨论过吗？              描述的 Issue/Purpose  Fixes #（无 issue 可能被自动关）；Summary 指向 issue  Purpose 栏链接 issue；>500 行要 RFC
2–3    这改动会不会弄坏别的东西？              CI 状态              pull + Lint 绿；需要时 ciflow/trunk               pre-commit 绿；相关 test_area 经 /ci run 绿
3–6    改动本身对不对？                       diff                 ≤2000 行；一叠 ghstack 每层可单独读             一个 PR 一件事；无关 import 重排为零
6–8    怎么证明它对？                         测试                 test/ 下的 TestCase；instantiate_device_type_tests   tests/ 下的 pytest；AGENTS.md 四个问题；opcheck
8–9    值不值？（性能 PR）                    benchmark            checklist "Included benchmark results"           Test Result 栏的前后对比表；vllm bench / benchmarks/kernels
9–10   有没有我该担心的？                     BC-breaking / 风险   模板的 BC-breaking? 栏                           deprecation_policy.md；Test Plan 写清没测的部分
—      这个人可信吗？                         签名 + AI 声明       EasyCLA；AI_POLICY.md 的声明方式                  DCO；Co-authored-by + 描述里的 AI 声明
```

反过来读这张表，就是一份提交前的自检：每一行都有对应的东西吗？如果"怎么证明它对"这一行是空的，reviewer 会在第 6 分钟停下来写 "how did you test this?"，然后你的 PR 回到队列末尾——下一次被打开可能是几天后。十分钟的预算里，任何一个空格都会让整个 PR 等一轮。

两个项目对这十分钟的分配略有不同。PyTorch 把更多信息放在**结构化元数据**里（标签、`Fixes #`、merge_rules 匹配的 approver），reviewer 打开前很多问题已经被机器回答了；vLLM 把更多信息放在**描述正文**里（Purpose / Test Plan / Test Result），reviewer 要读文字，所以描述写得好坏影响更大。


## 十四、贡献日志：PR 草稿与往返记录

本篇在贡献日志里新增的一页是"PR 草稿与往返记录"。它在 PR 开出之前就开始写，在合入（或放弃）之后才结束。它的价值有两个：开 PR 前逼自己把第十三章那张表填满；PR 过程中把每一轮 review 意见和回应留下痕迹，第四篇复盘时用。

### 1. 模板

```markdown
## PR 草稿：<一句话目的>

### 元信息
- 目标项目 / 版本基线：
- 对应 issue / RFC：
- 拟用标题：                           ← vLLM 带前缀；PyTorch 写成 release notes 的一行
- 拟用模板：                           ← PyTorch: fix_issue / docs_typo / preapproved；vLLM: 唯一模板
- 预计 diff 规模：      文件数 / 行数    ← PyTorch <2000；vLLM 架构改动 <500 或先 RFC
- 是否需要拆分：        是 → 几层、每层一句话
- 用到 AI 吗：          是 → 用在哪、怎么声明

### diff 摘要
| 文件 | 改了什么 | 为什么（对应目的的哪一部分） |
|---|---|---|

### 测试清单
| 测试 | 位置 | 覆盖什么 | 改动前结果 | 改动后结果 |
|---|---|---|---|---|

### benchmark（性能 PR）
| Case（硬件 / dtype / shape） | Before | After | 倍数 |
|---|---|---|---|
命令：

### 本地检查记录
| 步骤 | 命令 | 结果 | 日期 |
|---|---|---|---|
| lint | spin fixlint / pre-commit run --all-files | | |
| 相关测试 | | | |
| 签名 | git log --format=%B -1 \| grep Signed-off-by（vLLM） | | |

### 描述初稿
（按目标项目模板填写，见下）

### CI 记录
| push | 触发方式 | 跑了什么 | 结果 | 红的是不是我的（依据） |
|---|---|---|---|---|

### review 往返
| 轮次 | 日期 | reviewer | 意见（原文短引） | 类型（改/争/问） | 回应 | 状态 |
|---|---|---|---|---|---|---|

### 结局
- 合入 / 关闭 / 放弃：
- 进入版本：
- 拒绝类型（如有）：方向 / 时机 / 做法
- 学到的：
```

### 2. 样例一：PyTorch，按 fix_issue.md 填写

假设改动是修一个 CUDA 算子在空 tensor 输入下的越界读（issue 已标 `actionable`）。描述初稿：

```markdown
## Issue

Fixes #NNNNNN

## Summary

`foo_cuda` indexes `input.data_ptr()[0]` before checking `numel() == 0`; see the
issue for the repro and the discussion of why an early return (rather than a
TORCH_CHECK) is the right fix. This PR adds the early return and a
device-generic test that exercises the empty-input path on CPU and CUDA.

## Checklist

- [x] Passes lint (`spin fixlint`)
- [x] Added/updated tests
- [ ] Updated documentation (if applicable)
- [ ] Included benchmark results (for PRs impacting perf)

## BC-breaking?

No.
```

配套的日志条目：

```text
标题        Fix out-of-bounds read in foo_cuda for empty inputs
标签        需要 release notes: cuda（labeler 按路径自动打；没打就 @pytorchbot label）
diff        aten/src/ATen/native/cuda/Foo.cu（+3）；test/test_foo.py（+14）；共 2 文件 17 行
测试        test/test_foo.py::TestFoo::test_foo_empty_input，用 instantiate_device_type_tests 生成 _cpu / _cuda 两份；改动前 CUDA 版 illegal memory access，改动后通过
lint        spin fixlint → 0 errors（2026-09-xx）
CI          push 1：pull + Lint 自动；请 reviewer 打 ciflow/trunk 补 ROCm
合入        approve 后评论 @pytorchbot merge；若某无关 job 红且 HUD 显示 main 也红 → @pytorchbot merge -i 由 reviewer 决定
```

### 3. 样例二：vLLM，按 PULL_REQUEST_TEMPLATE.md 填写

假设改动是修一个 OpenAI 兼容接口在某个参数组合下返回 500 的 bug。描述初稿：

```markdown
## Purpose

Fix #NNNNN: `/v1/chat/completions` returns HTTP 500 instead of 400 when
`logprobs=true` is combined with `top_logprobs=0`, because the validator in
`vllm/entrypoints/openai/chat_completion/protocol.py` raises `ValueError` outside the request
validation path. Checked open PRs with `gh pr list --search "top_logprobs 500"`;
no existing PR addresses this.

## Test Plan

    pytest -s -v tests/entrypoints/openai/chat_completion/test_chat.py -k top_logprobs_zero

Added `test_chat_top_logprobs_zero_returns_400` next to the existing
`top_logprobs` tests; no new test file.

## Test Result

Before: `FAILED ... assert 500 == 400`
After:  `1 passed`

Full `tests/entrypoints/openai/chat_completion/test_chat.py` also passes locally on one H100
(`pytest -s -v tests/entrypoints/openai/chat_completion/test_chat.py`, 47 passed).

Drafted with an AI assistant; every line reviewed and the test run by me.
```

配套的日志条目：

```text
标题        [Bugfix] Return 400 instead of 500 for logprobs=true with top_logprobs=0
签名        每个 commit git commit -s；pre-commit 的 signoff-commit 钩子已装
lint        pre-commit run --all-files → Passed（2026-09-xx）；mypy-3.12 --hook-stage manual → Passed
diff        vllm/entrypoints/openai/chat_completion/protocol.py（+4 −2）；tests/entrypoints/openai/chat_completion/test_chat.py（+18）
CI 预期     source_file_dependencies：entrypoints.yaml 里匹配 vllm/entrypoints/ 的 step；pre-commit 需 verified/ready 标签
触发        开 PR 后等 reviewer /ci run 或 ready；每次 push 后自己 /ci run（ready 之后）
commit      末尾 Co-authored-by: <agent> 与 Signed-off-by: 两行 trailer
```

### 4. review 往返记录表（填写示例）

```text
轮次  日期        reviewer   意见（原文短引）                                    类型   回应                                              状态
1     09-16       @a         "can you add a test for the CUDA path too?"           改     加 instantiate_device_type_tests；push；回复 "Done, see test_foo_empty_input_cuda"   resolved
1     09-16       @a         "nit: prefer `numel() == 0` over `!numel()`"          改     照改                                               resolved
2     09-18       @b         "why early return instead of TORCH_CHECK?"            争     引用 issue 里 maintainer 的结论 + 与 CPU 实现一致的理由；对方接受   resolved
2     09-18       @b         "this might be worth a note in the docs"              问     问是指 docstring 还是 docs/source；对方说不必了       resolved
—     09-19       @b         approved                                              —      评论 @pytorchbot merge                             merged
```

"类型"一栏只有三种：改（照做）、争（有依据地维持）、问（澄清）。如果一个 PR 的记录里"争"占了多数，通常说明选题或方向阶段就有问题，而不是实现问题。


## 十五、本文小结

### 1. 要点回顾

```text
最小 diff     一个 PR 一件事；PyTorch pr-sanity-check.sh 2000 行硬上限 + ghstack 叠 PR；vLLM >500 行架构改动需 RFC、顺序开 PR、6 个 open PR 上限
测试          PyTorch：TestCase / run_tests / @parametrize / instantiate_device_type_tests（common_utils.py、common_device_type.py），TEST_HAS_MAIN linter 强制入口
              vLLM：pytest；AGENTS.md 四个问题（模块为何 / I/O 契约 / 防什么失败 / 最便宜的层级）+ 五条规则；kernel 用 torch.library.opcheck；模型改动跑 tests/evals 或 vllm bench
benchmark     基线 / 对比 / 硬件 / shape / 命令 / 不利 case；PyTorch 模板 checklist "Included benchmark results"、benchmarks/ 目录
              vLLM benchmarks/kernels/（66 个脚本）；端到端 vllm bench latency|throughput|serve|startup|sweep——benchmarks/benchmark_*.py 在 v0.28.0 只是弃用桩
lint          PyTorch .lintrunner.toml 61 个 linter；spin lint / fixlint / quicklint / quickfix（pyproject.toml [tool.spin.commands]）；lintrunner -a
              vLLM .pre-commit-config.yaml：ruff-check / ruff-format / typos / clang-format / markdownlint-cli2 / mypy-3.10（本地）/ mypy-3.11–13（CI）/ signoff-commit / …；pre-commit run [-a] [--hook-stage manual]
描述          PyTorch 三模板：Fixes #（无 issue 可能自动关）/ Summary（过长视为 spam）/ Checklist / BC-breaking?；docs_typo 只有 What changed?；preapproved 有 Approved by
              vLLM Purpose / Test Plan / Test Result；标题前缀 [Bugfix] [CI/Build] [Doc] [Model] [Frontend] [Kernel] [Core] [Hardware][Vendor] [Misc]
签名          PyTorch CLA（EasyCLA check，在 merge_rules 的 mandatory_checks_name 里）；vLLM DCO（git commit -s；signoff-commit 钩子；mergify comment-dco-failure）
CI            PyTorch 148 个 workflow；PR 自动跑 pull + Lint；trunk / periodic / slow / inductor 靠 ciflow/* 标签（pytorch-probot.yml 列 49 个）
              vLLM 35 个 test_area；source_file_dependencies 决定触发；ci_config.yaml run_all_patterns 例外；PR 默认只跑 pre-commit（需 verified/ready 或 ≥4 合入）；/ci run 授权链：写权限 → 受信名单 → 作者且非 draft 且有 ready/approval
读日志        先看 main 是否也红：PyTorch HUD + "CI failure tips"、"Which commit is used in CI?"；gh pr checks；vLLM CI Failures Dashboard + failures.md + ci-fetch-log.sh + rerun-test.sh
review        PyTorch：triage 几个工作日分派；4 个工作日无回应留言；再去 Office Hours。vLLM：2–3 天一次状态；7 天可 ping；action-required；pr-review-request@vllm.ai
              回应：改 / 争（有依据）/ 问；一轮一次 push；每条都回；处理完主动 ping
合入          PyTorch @pytorchbot merge [-i | -f "<reason>"]；merge_rules.yaml 33 条（patterns / approved_by / mandatory_checks_name）；release notes: 或 topic: not user facing 标签；ci: sev 全局暂停
              vLLM maintainer 打 ready → /ci run → squash merge；mergify needs-rebase；auto-merge 自动加 ready
被拒          方向（放弃，回 issue/RFC）/ 时机（保留分支，等）/ 做法（改）；只有第三类值得"改了再提"
AI            PyTorch AI_POLICY.md：标注并包裹 AI 内容 + 人的评注；不贴未审阅 AI 文本；读懂再提；draft；不接受全自主 agent。CONTRIBUTING：对每一行负责；新贡献者需 actionable issue；新功能 issue 不放 AI 方案
              vLLM AGENTS.md：查重三命令；无 busywork；Pure code-agent PRs not allowed；描述四项必填；Co-authored-by；违规封禁
核心问题      十分钟 = 标题标签 → 目的 → CI → diff → 测试 → 数据 → 风险 → 签名；每一格都要有东西
```

### 2. 两个项目对照

```text
环节        PyTorch v2.14.0                                          vLLM v0.28.0
体积        2000 行 CI 硬上限；ghstack                                 500 行 RFC 线；顺序 PR；6 个 open PR 上限
测试        unittest 体系 + 设备泛化装饰器                              pytest + AGENTS.md 设计原则
benchmark   模板 checklist；benchmarks/ 按子系统                        benchmarks/kernels/ + vllm bench 子命令
lint        lintrunner 61 项，spin 包装，手动跑                          pre-commit ~35 项，commit 时自动跑
描述        三模板，元数据化（Fixes #、标签）                            一模板，正文化（Purpose / Test Plan / Test Result）+ 标题前缀
签名        CLA 一次                                                  DCO 每 commit
CI          推送即跑，分层；ciflow 标签加跑                             默认不跑；/ci run 按需；source_file_dependencies 按文件
日志        HUD                                                       Buildkite 公开日志 + Dashboard + 脚本
review      4 个工作日可催                                            2–3 天状态，7 天可催
合入        机器人 + merge_rules 路径权限 + release notes 标签           人 + ready 标签 + mergify
AI          针对讨论质量（不贴 AI 文本）                                针对 PR 数量（不做 busywork）
```

### 3. 文件位置表

| 项目（版本） | 路径 | 章节 / 字段 / 名称 |
|---|---|---|
| PyTorch v2.14.0 | `CONTRIBUTING.md` | "AI-Assisted Development"、"Spin"（"Building" / "Linting" / "default lint" / "Regenerating"）、"Unit testing"（"Python Unit Testing" / "Better local unit tests with `pytest`" / "Local linting" / "C++ Unit Testing" / "Run Specific CI Jobs"）、"Merging your Change"、"Linting before committing"、"CI failure tips"（"Which commit is used in CI?"）、"Dev Infra Office Hours" |
| | `AI_POLICY.md` | 五条规则全文 |
| | `AGENTS.md` / `CLAUDE.md` | "AI Policy — MANDATORY"、"Testing"、"Linting"、"Commit messages"、"ghstack Workflow"、`gh pr checks … --json` 提示 |
| | `.github/PULL_REQUEST_TEMPLATE/fix_issue.md` / `docs_typo.md` / `preapproved.md` | `Issue`（`Fixes #`）/ `Summary` / `Checklist` / `BC-breaking?`；`What changed?`；`Approved by` |
| | `.lintrunner.toml` | 61 个 `[[linter]]` 的 `code`（FLAKE8、RUFF、PYFMT、CLANGFORMAT、CLANGTIDY、PYREFLY、CODESPELL、TEST_HAS_MAIN、NATIVEFUNCTIONS、WORKFLOWSYNC …）、`is_formatter` |
| | `pyproject.toml`；`.spin/cmds.py` | `[tool.spin.commands]`：`lint` / `fixlint` / `quicklint` / `quickfix` / `develop` / `regenerate-*` / `docs` / `pyrefly` |
| | `.github/workflows/pull.yml` / `trunk.yml` / `periodic.yml` / `slow.yml` / `inductor.yml` / `lint.yml` | `on:` 触发条件；`lint.yml` 的 `pr-sanity-checks` / `lintrunner-clang` / `lintrunner-pyrefly` / `lintrunner-noclang` / `quick-checks` / `workflow-checks` |
| | `.github/scripts/pr-sanity-check.sh` | 2000 行上限；`skip-pr-sanity-checks` 标签 |
| | `.github/pytorch-probot.yml` | `ciflow_push_tags`（49 个）、`retryable_workflows`、`mergebot: true` |
| | `.github/merge_rules.yaml` | 33 条：`name` / `patterns` / `approved_by` / `mandatory_checks_name`（`EasyCLA` / `Lint` / `pull` …）；`Core Maintainers` 兜底 |
| | `.github/scripts/trymerge.py`；`trymerge_explainer.py`；`label_utils.py` | `parse_args`（`--force` / `--ignore-current` / `--revert`）；`find_matching_merge_rule`；`ensure_mergeable_labels`；`check_for_sev`；`get_ghstack_prs`；`NOT_USER_FACING_LABEL`；`LABEL_ERR_MSG` |
| | `.github/workflows/trymerge.yml` / `tryrebase.yml` / `revert.yml` / `check_mergeability_ghstack.yml` | 机器人 workflow |
| | `.github/labeler.yml` | `release notes: *` 自动标签 |
| | `torch/testing/_internal/common_utils.py`；`common_device_type.py`；`opinfo/core.py` | `TestCase` / `run_tests` / `parametrize`；`instantiate_device_type_tests` / `dtypes` / `onlyCUDA` / `ops`；`OpInfo` |
| | `test/run_test.py`；`test/cpp/`；`benchmarks/` | 测试入口；C++ gtest；`operator_benchmark/` / `dynamo/` / `inductor_backends/` 等 |
| | `tools/testing/explicit_ci_jobs.py` | `--filter-gha` / `--make-commit` |
| vLLM v0.28.0 | `docs/contributing/README.md` | "Linting"、"Testing"、"DCO and Signed-off-by"、"AI Assisted Contributions"、"PR Title and Classification"、"Code Quality"、"Adding or Changing Kernels"、"Notes for Large Changes"、"What to Expect for the Reviews"、"Pull Request Limits and Escalation" |
| | `AGENTS.md` / `CLAUDE.md` | "Duplicate-work checks" / "No low-value busywork PRs" / "Accountability" / "Fail-closed behavior"；"Tests"（四个问题、五条规则）；"Running linters"；"Commit messages" |
| | `.github/PULL_REQUEST_TEMPLATE.md` | `Purpose` / `Test Plan` / `Test Result`；折叠 checklist |
| | `DCO` | 协议文本 |
| | `.pre-commit-config.yaml` | `default_install_hook_types`、`default_stages`；hook：`ruff-check` / `ruff-format` / `typos` / `clang-format` / `markdownlint-cli2` / `actionlint` / `pip-compile` / `mypy-3.10`–`3.13` / `shellcheck` / `signoff-commit` / `check-spdx-header` / `validate-config` / … / `suggestion` |
| | `.buildkite/test_areas/*.yaml`（35 个） | `group` / `steps[]`：`label` / `key` / `device` / `num_devices` / `source_file_dependencies` / `commands` / `parallelism` / `optional` / `soft_fail` / `mirror` / `timeout_in_minutes` |
| | `.buildkite/ci_config.yaml` | `job_dirs` / `run_all_patterns` / `run_all_exclude_patterns` / `repositories`（`premerge` / `main`） |
| | `.buildkite/test-pipeline.yaml` | 弃用说明（2026-02-18 迁移） |
| | `.buildkite/scripts/ci-fetch-log.sh` / `ci-clean-log.sh` / `rerun-test.sh` | 日志抓取 / 清洗 / flaky 复现 |
| | `.github/workflows/run-ci-command.yml`；`scripts/run_ci_command.py` | `/ci run` / `/ci run all` / `/ci run nightly` / `/ci retry` / `/ci cancel`；`authorize`、`READY_LABELS`、`TRUSTED_PERMISSIONS`、`CI_TRUSTED_USERS` |
| | `.github/workflows/pre-commit.yml` | `pre-run-check`（`verified` / `ready` / `ready-run-all-tests` 或 ≥4 合入 PR）；`--all-files --hook-stage manual` |
| | `.github/workflows/new_pr_bot.yml`；`add_label_automerge.yml`；`stale.yml` | 描述清理与欢迎评论；auto-merge 自动加 `ready`；stale |
| | `.github/mergify.yml` | 35 条规则：`label-*`、`comment-pre-commit-failure`、`comment-dco-failure`、"ping author on conflicts and add 'needs-rebase' label"、"remove 'needs-rebase' label when conflict is resolved"、`assign` 规则 |
| | `docs/contributing/ci/failures.md` | "Filing a CI Test Failure Issue" / "Logs Wrangling" / "Investigating a CI Test Failure" / "Reproducing a Failure" / "Submitting a PR" / "Daily Triage" |
| | `docs/contributing/model/tests.md`；`deprecation_policy.md`；`profiling.md` | 模型测试要求；弃用政策；`vllm bench serve --profile` |
| | `benchmarks/kernels/`（66 个）；`benchmarks/benchmark_serving.py` 等（弃用桩）；`benchmarks/README.md` | kernel benchmark；指向 `vllm bench` |
| | `vllm/benchmarks/`；`vllm/entrypoints/cli/benchmark/` | `latency` / `throughput` / `serve` / `startup` / `sweep` / `mm_processor` |
| | `tests/`（`kernels/` / `entrypoints/` / `evals/` / `v1/` …） | 测试组织；`tests/kernels/attention/test_attention.py` 的 `opcheck` 用法 |

GitHub 标签（截至 2026-09 查询）：PyTorch 存在 `actionable`、`skip-pr-sanity-checks`、`ciflow/trunk`、`topic: not user facing` 与一组 `release notes: *`；vLLM 存在 `ready`（"ONLY add when PR is ready to merge/full CI is needed"）、`ready-run-all-tests`、`verified`、`needs-rebase`、`ci-failure`，而文档提到的 `action-required` 与 `rfc-required` 在标签列表中未找到。

### 4. 贡献日志本篇增量

一页"PR 草稿与往返记录"：元信息（项目 / issue / 标题 / 模板 / 规模 / 拆分 / AI）· diff 摘要表 · 测试清单表 · benchmark 表与命令 · 本地检查记录（lint / 测试 / 签名，带日期）· 按目标项目模板写好的描述初稿 · CI 记录表（每次 push 的触发方式与红绿归因）· review 往返表（轮次 / 意见短引 / 改-争-问 / 回应 / 状态）· 结局与拒绝类型。到本篇结束，这一页对应的 PR 应该已经真实提交。

下一篇把前三篇讲的所有环节放到两个已经合入的真实 PR 上：PyTorch 一个、vLLM 一个，从 issue 到进入哪个版本，逐段读 diff、逐条读 review，看这些规则在真实的往返里长什么样。

> **两个都是"小"PR，却各花了作者一到几周。这些时间花在哪里了？哪些是可以省的，哪些是这个项目的正常成本？**


## 下一篇

[两个真实 PR 的完整走读：PyTorch 与 vLLM](/two-real-prs-pytorch-and-vllm.html)
