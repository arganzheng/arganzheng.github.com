---
title: git分支与maven版本之间的联动
layout: post
tags: [maven, agile]
catalog: true
---


对于 java 开发，项目联合开发的时候往往是通过maven仓库而不是分支源码依赖进行，而且为了让变更立即可见，需要采用 maven 的 snapshot 版本机制；而对于稳定版本（release版本），为了能够拉取更新以及记录变更，也是需要升级版本号的。也就是说二方库有自己的（maven）版本管理机制， 需要跟 git 的分支机制联动起来。

为了方便说明，我们举一个具体的二方库作为例子。假设有这么一个二方库，它的 pom 文件定义如下：

```xml
<groupId>life.arganzheng.internet.ai.platform</groupId>
<artifactId>large-scale-algorithms</artifactId>
<version>1.0.0</version>
```

目前的稳定版本是 `1.0.0`。现在假设 argan 要对这个二方库进行功能开发，按照我们前面讨论的 Git 分支管理策略，他首先拉了一个 feature 分支 —— `feature-test-for-maven-version`。然后他把代码 checkout 下来，完成相应的功能开发。

这时候他要把这个新功能开放给同一个项目中的 magi，进行联调。那么 magi 怎么才能拿到 argan 的修改呢？

因为所有的 java jar 包依赖都是通过 maven 管理的，所以 argan 需要提供一个独一为二的 snapshot 版本才能保证 magi 一定能够从 maven 仓库中拉取到他的 jar 包。


简单的将 pom 文件的 version 增加 `-SNAPSHOT`，即 `1.0.0-SNAPSHOT` 很可能会发生冲突，因为其他人也可能同时拉开发分支进行开发。同样，版本号先递增，再增加 `-SNAPSHOT`，即 `1.0.1-SNAPSHOT` 也不能完全排除冲突。

但是回到开发联调这个问题，其实 argan 只需要提供一个独一为二的 snapshot 版本就可以了，所以简单的方式就是在版本号加上一些版本之外的东东，如 他的名字 + Git分支，也就是，argan 可以在开发节点将 feature 分支的 pom 版本修改为 `${开发者名字}-${Git分支名}-${原有分支版本}-SNAPSHOT`：

```xml
<groupId>life.arganzheng.internet.ai.platform</groupId>
<artifactId>large-scale-algorithms</artifactId>
<version>argan-feature-test-for-maven-version-1.0.0-SNAPSHOT</version>
```

这样这个 pom 就是唯一的，magi 只需要将他的 pom 文件对 argan 的这个二方库的依赖的版本也写成 `argan-feature-test-for-maven-version-1.0.0-SNAPSHOT` 就可以了。

然后双方完成开发和联调之后，再走前文讨论的 [Git 分支管理策略](http://arganzheng.life/git-branch-strategy.html)，进行分布。但是发布的时候需要把版本号修正成正确的版本号，因为 `argan-feature-test-for-maven-version-1.0.0-SNAPSHOT` 只是一个开发阶段的临时唯一版本号。发布的时候需要改成唯一的 release 版本号。

这时候可以这么操作，分支代码发布时，根据当前主干的版本号，修改 pom 文件 version，采用如下策略进行升级:

1. 新算法开发和算法升级等升级中位版本号。eg: 1.0.0 → 1.1.0
2. 小功能和bug fix升级末尾版本号。 eg: 1.0.0  → 1.0.1
3. 大的框架层面的优化和重构升级首位版本号。eg: 1.0.0 → 2.0.0

假设到发布的那天主干的版本已经升级为 `1.1.2` 了，argan 的这个修改是一个比较小的功能修改，那么最终要 release 的版本号就是 `1.1.3`。

注: 代码发布后需要使用 `mvn deploy` 发布到maven仓库。

