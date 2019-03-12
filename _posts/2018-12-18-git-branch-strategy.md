---
title: Git分支管理策略
layout: post
tags: [git,agile]
catalog: true
---

在使用 Git 时通常会遇到的一个问题是采用何种分支管理实践，即如何管理仓库中作用不同的各类分支。


GitFlow 演进
-----------


### 洪荒时代 —— 单主干开发 (Trunk-based development, TBD)

单主干的分支实践（Trunk-based development, TBD）在 SVN 中比较流行。trunk 是 SVN 中主干分支的名称，对应到 Git 中则是 master 分支。

在项目初期，一切都是从0开始，不存在多分支并行开发一说，所以基本上都是基于主干开发和发布。

**Pros**

* 简单，只需要维护一个分支
* 冲突第一时间发现和解决

**Crons**

* 主干一直在变化，没有一个稳定的版本来进行测试和发布
* 因为主干分支是所有开发人员公用的，一个开发人员引入的 bug 可能对其他很多人造成影响。
* 没有发布分支，主干是一直在变化的，无法知道线上的代码版本，排查线上问题对不上。
* 没有一个稳定的版本，新同学拉下来的代码可能是没法运行的。

### 演进1 —— 引入发布分支 

还是采用主干开发方式，团队所有人都在 trunk 开发新功能，但是在 『迭代功能完成』的时候需要拉一个测试分支，用于修复测试中发现的缺陷，以及最后的发布。而主干则继续开发新功能。

根据发布分支和主干的合并机制，又有如下三种策略：

1. 发布分支发布之后合并回主干
2. 发布分支每修复一个缺陷就合并回主干
3. 发布分支上做修改后，就要根据实际情况进行分析，是否要合并回主干。如果需要合并，应该立即进行。

**release 分支**

* release 为预发布分支，它是指发布正式版本之前（即合并到Master分支之前），我们需要有一个预发布的版本进行测试和发布
* 临时分支，发布后以 tag 形式合并 master 分支（gitflow 模式下还要合并到 develop 分支）后删除
* 命名规范: 可以采用 `release-*` 的形式命名

**Pros**

* 保留了单主干开发的优点
* 解决了单主干开发的缺点

**Crons**

* 如果有多个项目迭代并行，单主干作为开发分支，无法定义『迭代功能完成』
* 发布分支太多，不方便维护
* 主干仍然没有一个稳定的版本可以作为基准


### 演进2 —— 引入功能分支或者个人仓库

单个开发分支并行度还是有限，不能很好的支持多个项目并行开发。一种解决方案就是引入多个开发分支，根据开发分支的粒度，可以分为以下几种策略:

* feature branch : 按照功能维度隔离切分；临时分支。
* personal repository/branch : 按照人员维度隔离切分；长期分支。这种模式其实就是 GithubFlow/GitlabFlow 模式。

**feature 分支**

* 为了开发某种特定功能，从 master/develop 分支上面分出来的
* 临时分支，合并到发布分支后（gitflow 模式下还要合并到 develop 分支）删除
* 命名规范: 可以以 `feature-*` 形式命名

**Pros**

* 提高开发的并行度和隔离性

**Crons**

* 功能分支/个人仓库 之间互相不可见，需要确定合并的机制。


现在功能开发都在各自的分支或者仓库上开发了，那么怎么进行功能合并和发布呢？

根据其合并机制可以衍生出两个分支管理模式:

1. 开发的时候合并 : 显然不能直接合并到 master 去，但是需要一个类似于 master 功能的『长期分支』 —— develop 分支，这样，所有的功能分支都可以合并到这个主干去，发布分支就可以直接从这个分支拉出来进行测试发布了。
2. 发布的时候合并 
    1. 直接合并到 master : GithubFlow 模式
    2. 合并到 Release 分支 : 这个分支和操作流程前面已经介绍过了，这里就不赘述。

**说明**

1、develop 分支

* develop 为开发分支，始终保持最新完成以及bug修复后的代码
* 在 gitlab 模式下，开发新功能时，feature分支都是基于develop分支下创建的，并且会不断的合并会 develop 分支。

2、GithubFlow

GithubFlow 模式，其实就是在 TrunkBased 的基础上，增加了个人仓库和 Pull Request 合并代码的操作，与在同一个仓库里增加个人分支的做法类似，从实用的意义来说，它更合适分布式团队。GithubFlow 也有演进版本，例如强调了多环境部署和将仓库或分支与环境关联的 GitlabFlow 模式。

![github-flow.png](/img/in-post/github-flow.png)

### 演进3 —— 引入 hotfix 分支

如果线上发现有紧急 bug 这时候应该怎么修复呢？前面讨论过的 master 分支、develop 分支、release 分支和 feature 分支显然都适合做这个事情。于是我们引入另一种分支 —— hotfix 分支。

**hotfix 分支**

* 线上出现紧急问题时，需要及时修复，以 master 分支为基线，创建 hotfix 分支进行故障修复。
* 临时分支，跟 release 分支非常类似，测试发布完成后，需要以 tag 形式合并到 master 分支（gitflow 模式下还要合并到 develop 分支）后删除
* 命名规范 : 可以以 `hotfix-*` 命名


GitFlow 简化
-----------

前面我们介绍了从最简单的单主干开发逐渐演进到复杂的 GitFlow 分支管理模式，是一个加法的过程。

GitFlow 模式是若干模式的集大成者，包含一个主干分支、一个开发分支、许多的特性分支、许多的发布分支和 Hotfix 分支，以及许多繁琐的合并规则。由于对每个阶段的每项操作定义十分明确，它曾经是很多重视流程的企业眼里的香馍馍。但它流程复杂，使用起来并不是很容易。

![gitflow.png](/img/in-post/gitflow.png)

但是与软件开发中的其他实践一样，Git 分支管理并没有普遍适用的最佳做法，而只有对每个团队和项目而言最适合的做法。不同的团队应该根据自己团队人员组成和意愿、项目的发布周期等因素选择最适合的策略，找到最适合团队的管理方式。现在我们来看一下怎么简化 GitFlow。


### 简化1 —— 取消 develop 分支

develop 分支是为了让功能分支能够尽快的进行合并以及作为发布分支的基准。但是也正因为功能分支要随时合并回 develop 分支，这就导致 develop 分支不稳定。这就带来一下几个问题:

1. 因为 develop 分支是所有开发人员公用的，一个开发人员引入的 bug 可能对其他很多人造成影响。
2. 开发分支没有一个稳定的版本，新拉的 feature 分支可能是没法运行的
3. 基于不稳定的 develop 分支拉 release 分支，会把未完成的 feature 分支也一起发布上线，弥补的措施是 FeatureToggle 以及频繁的集成和足够的测试覆盖，这对开发团队的能力提出了比较高的要求。
4. 很多合并都需要同时合并 master 和 develop，增加操作的复杂度。


综合对比之下，我们觉得：

1. 从不稳定的 develop 分支拉取 feature 分支，还不如从 稳定的 master 分支拉取；虽然代码及时性相对 develop 分支落后一些。
2. 从不稳定的 develop 分支拉取 release 分支，还不如从 稳定的 master 分支拉取，并且需要上线的 feature 分支自己选择合并过去，确保只发布完善的 feature 分支。带来的问题就是发布前合并，冲突解决偏后。

综上，我们决定废除 develop 分支，引入搭车发布的分支开发模式:

![gitflow_simplify.png](/img/in-post/gitflow_simplify.png)

**使用流程**

相比 git flow，最大的改动在于去掉了 develop 分支。

1、功能迭代：

开发新功能时，先从 master 分支拉取 feature 功能分支，是否推送到远端，看是否需要多人协作，功能开发测试完成后，合并到 release 发布分支，删除分支；

2、版本发布：

功能开发到里程碑之后，或者接近版本迭代周期时(固定发版时间前一天)，从最新的 master 分支拉取 release 发布分支，merge 本次需要发布的所有 feature 分支，发布测试基于 release 分支，测试的 bug 直接在 release 分支修复，release 分支一旦开始测试，后续 checkout 的 feature 功能分支不能再合并到该分支，需要放到下一次 release 分支。 版本发布成功后，合并 release 分支到 master 分支，master 分支需要打上 tag 号。合并后删除 release 分支。

3、Bug 修复

线上 bug 需要修复时，从master分支(生产运行的最新代码)拉取 hotfix 修复分支，是否推送到远端，看是否需要多人协作，修复完成后，合并分支到 master 分支，同时 master 分支需要打上 tag 号。

**原则**

* Master 分支不做直接修改，只 merge 其它分支，push 远端，每一次合并都需要打 tag，包括 release 和 hotfix。
* Feature 分支从 Master 分支 checkout，功能完成 merge 到 release 分支。
* Release 分支从 Master 分支 checkout，一旦提交测试，不在 merge 后续的 feature 分支，确定版本稳定发布后，merge 到 master 分支。
* Hotfix 分支从 maser 分支 checkout，确定修复发版后，merge 回 master 分支，同时必须打上 tag。
* 简化版策略清晰可控，适合“版本发布”的开发模式，同时也没有 gitflow 策略复杂。

**存在问题**

* 因为没有 develop 分支汇总 feature 改动，多个 feature 分支之间的代码改动（包括 hotfix），在 merge 到 release 分支之前，代码需要相互合并才能可见。


### 简化2 —— 弱化 release 分支

简化后的分支开发模式比原来要简单很多，但是实际运行过程中我们发现搭车发布的方式比较适合于有严格迭代计划和发布周期的开发模式，不适合敏捷开发持续发布的开发模式。于是我们把整个流程又做了一次精简，直接让每个功能分支直接作为 release 分支发布，发布之前周知一下，如果有其他特性分支要一起发布的话就合并在一起发，否则就自己单独发布。

![git-branch-cd.png](/img/in-post/git-branch-cd.png)

**说明**

* 每个人拉自己的功能或者修复分支
* 需要发布时先在群里同步有版本需要发布
* 如果有别人的功能和修复也需要发布，等待搭车的分支合并到自己分支
* 发布前合并最新的 master 分支
* 发布后群里通知，相关的同学自己验证功能
* 合并到 master 分支，打上 tag 号
* 删除当前功能分支
* 在 wiki 上记录发布信息


推荐阅读
------

1. [持续集成之“分支策略”​](https://infoq.cn/article/2011/03/ci-branch-strategy)
2. [Git 分支管理最佳实践](https://www.ibm.com/developerworks/cn/java/j-lo-git-mange/index.html)
3. [在阿里，我们如何管理代码分支？](https://yq.aliyun.com/articles/573549)
4. [Git分支管理策略](http://www.ruanyifeng.com/blog/2012/07/git.html)





