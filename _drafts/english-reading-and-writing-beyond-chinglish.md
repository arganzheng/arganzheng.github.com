---
layout: post
category: life
series: english-for-going-abroad
title: "一年英语计划（05）：阅读与写作——从技术文档到成段英文"
subtitle: "Reading and Writing beyond Chinglish"
tags: [英语, 学习, 海外]
catalog: true
date: 2026-12-11
description: "能读 RFC 不等于能读新闻，能写 commit message 不等于能写邮件。阅读从技术文档迁到长文的分级路线；写作三种体裁——工作邮件、design doc 段落、IELTS Task 2——共用一套先结构后句子的写法；LLM 批改的 prompt：只改错、标中式、不重写。附一张中式表达对照表。"
---

第 01 篇量出来的读写数字：非专业长文 120 词/分钟，250 字作文里 9 处中式表达。阅读是我最强的一项，但强得很窄——读 PyTorch 的 RFC 不查词典，读 Guardian 一篇关于住房政策的报道每段都要停。写作则是另一种问题：语法错不多，但每句话都能看出是中文翻过来的——`With the development of technology`、`It is widely believed that`、`we should pay attention to`。

这是 Q3 的主攻项之一（和词汇一起）。阅读大部分靠碎片时间和已有的英文技术阅读习惯，晚上 30 分钟主要给写作。


## 一、总览

按「阅读的迁移 → 写作的三种体裁 → 一套写法 → LLM 批改 → 中式表达对照 → 每日方案」组织。阅读和写作放在一篇是因为它们共享同一个问题——**从技术语域到通用语域的迁移**——而且写作的进步严重依赖阅读输入的语域：只读技术文档的人，写邮件也会写成技术文档。

| 章 | 内容 |
|---|---|
| 二 | 阅读：为什么能读文档读不动新闻，分级路线 |
| 三 | 写作的三种体裁 |
| 四 | 一套写法：先结构后句子 |
| 五 | LLM 批改：prompt 与规则 |
| 六 | 中式表达对照表 |
| 七 | 每日 / 每周方案 |


## 二、阅读：从技术文档到长文

### 1. 为什么能读文档读不动新闻

技术文档对我容易，不是因为英语好，是因为**背景知识和词汇都是现成的**，而且技术写作有固定的结构（问题 → 方案 → 例子），句式简单直接。新闻和长文的难点正好相反：

| 维度 | 技术文档 | 新闻 / 长文 |
|---|---|---|
| 词汇 | 专业词我都会，通用词少 | 通用高阶词多（`scrutiny`, `stark`, `bolster`, `wary`），正是 Oxford 5000 里我缺的那部分 |
| 句式 | 短、直接、少从句 | 长、多插入语、引语嵌套 |
| 背景 | 现成 | 需要目的国的社会常识（制度、人物、事件） |
| 结构 | 固定 | 倒金字塔（新闻）或叙事（长文），信息分布不均 |

所以阅读的目标不是「读更多」，是**换语域**——这一年阅读的量不必增加（我本来每天就读很多英文），但要把一部分从技术换成通用。

### 2. 分级路线

| 级 | 材料 | 毕业判据 |
|---|---|---|
| L1 | 学习者新闻（BBC Learning English 的 News Review、News in Levels） | 不查词读完；六级可跳过 |
| L2 | 短新闻（BBC、Guardian 的 500 词以内报道；AP 新闻） | 150 词/分钟以上，能三句概括 |
| L3 | 长篇报道和评论（Guardian Long Read、The Atlantic、NYT 的专题） | 一篇 2,000 词 15 分钟内读完，能说出作者的立场和两个论据 |
| L4 | 非虚构书（一本关于目的国社会的书，或者任何我感兴趣的非技术非虚构） | 一周一章不累 |
| L5 | 小说 | 可选；对口语的贡献最大（对白），对考试的贡献最小 |

我的起点是 L2 到 L3 之间。方法上只有一条和读技术文档不同：**查词要有节制**。一篇长文每段都查会读不下去，规则是一段最多查一个、其余靠猜；查过两次的词才进 Anki。

### 3. 阅读的时间

阅读不占晚上 30 分钟，它在碎片里：早上刷手机的时间换成读一篇短新闻；周末的一小段时间读一篇长文。Q3 每周至少一篇 L3 长文 + 三句英文概括——这三句是写作练习的一部分。


## 三、写作的三种体裁

我要写的英文只有三种，全年都围绕它们：

| 体裁 | 场景 | 长度 | 特点 | 频率（Q3） |
|---|---|---|---|---|
| **工作邮件 / 消息** | 请求、汇报、拒绝、跟进、道歉 | 50–150 词 | 短、结构固定、语气是关键 | 每周 1–2 封 |
| **Design doc 段落** | 背景、方案对比、决策理由 | 150–300 词 | 逻辑清楚、术语准确、不绕 | 每两周 1 段 |
| **IELTS Task 2 议论文** | 考试 | 250–300 词 | 四段结构、论点 + 论据 + 例子、40 分钟 | 每周 1 篇 |

三种看起来不同，但**中式英语的问题是一样的**：过度铺垫（一封邮件先讲两句背景再说事）、名词化（`the implementation of the feature` 而不是 `implementing the feature`）、万能词（`good`, `important`, `problem`, `thing`）、连接词滥用（`Firstly, Secondly, Moreover, In conclusion` 每段都有）。所以三种体裁可以用同一套方法练，也可以用同一个 prompt 批改。


## 四、一套写法：先结构后句子

中式英语的根源是**边想边翻**：脑子里有中文句子，逐词找英文。避开它的办法不是「用英语思考」（做不到），而是把过程拆成两步——先用任何语言把**结构**定下来，再一句一句写英文，每句只管一个信息。

### 1. 结构

三种体裁各一个骨架，写之前先填骨架（中文填也行，几个词就够）：

**邮件**：

```
一句话说事（我要什么 / 我做了什么）
背景（只有对方需要才写，最多两句）
具体请求或下一步（带日期）
一句收尾
```

**Design doc 段落**（方案对比）：

```
我们要解决的问题（一句）
候选 A：做法 + 优点 + 代价
候选 B：同上
我们选 X，因为（一到两个理由）
不选 Y 的代价我们接受，因为
```

**IELTS Task 2**：

```
开头：改写题目 + 我的立场（两句）
主体 1：论点 → 解释 → 例子
主体 2：论点 → 解释 → 例子（或反方观点 → 反驳）
结尾：重申立场 + 一句延伸
```

### 2. 句子

结构定了之后写句子，几条规则：

- **一句一个信息**。中文习惯一句话带三个从句，英文里拆成三句更清楚。
- **动词优先**。看到自己写 `the implementation of`、`the improvement of`，改成动词 `implement`、`improve`。
- **主语是人或具体的东西**，不是 `it is … that`。`It is important to test the code` → `We need to test the code` 或 `Test the code before merging`。
- **删掉第一句**。写完回头看，第一句常常是铺垫，删掉之后更好。
- **万能词换掉**：`good` → 具体好在哪（`fast`, `reliable`, `clear`）；`problem` → `bug` / `outage` / `regression` / `bottleneck`。

### 3. 邮件的语气

邮件是三种里最容易出错的，因为错的不是语法而是**语气**——中国人写英文邮件常常要么太生硬（`Please send me the report.`），要么太绕（`I was wondering if it might be possible for you to…`）。语气的梯度：

| 场景 | 太硬 | 合适 | 太软 |
|---|---|---|---|
| 请求 | `Send me the data.` | `Could you send me the data by Thursday?` | `I was wondering if you might possibly be able to…` |
| 催 | `You haven't replied.` | `Just following up on this — any update?` | `Sorry to bother you again, I know you're busy, but…` |
| 拒绝 | `No.` / `This is impossible.` | `I don't think that'll work because X. What if we Y instead?` | `I'm so sorry, I really wish I could, but unfortunately…` |
| 不同意 | `You're wrong.` | `I see it differently — my concern is X.` | `I might be wrong, and this is just my opinion, but maybe…` |

「合适」那一列基本是：直接说事 + 一个理由 + 一个替代方案。不需要道歉，不需要三层缓冲。第 06 篇的工作场景会展开。


## 五、LLM 批改

LLM 是写作练习里最大的变化——十年前写完一篇作文没人改，现在 10 秒就有反馈。但用错了会变成 AI 代写：让它「帮我润色」，它给你一篇漂亮的英文，你学到的是零。

### 1. Prompt

原则：**只指出，不重写**。我用的：

> You are an English writing tutor for a Chinese speaker at CEFR B2. I will give you a piece of my writing. Do NOT rewrite it. Instead, list: (1) grammar errors, quoting the exact words; (2) word choices that are wrong or imprecise, with a better word; (3) sentences that are grammatically fine but that a native speaker would not write — "Chinglish" — for each, explain briefly why and give one native way to say it; (4) structural issues: padding, sentences that should be split or merged, an opening that could be cut. Then give an overall count: N grammar, N word choice, N Chinglish. Do not comment on what is good. Text type: [email / design doc / IELTS Task 2].

对 IELTS Task 2 再加一句：

> Finally, estimate the band for each of the four IELTS Task 2 criteria (Task Response, Coherence and Cohesion, Lexical Resource, Grammatical Range and Accuracy) with one sentence of justification each. Be strict; do not inflate.

### 2. 规则

- 批改后**自己改**，不复制它的句子。改完可以再让它看一遍。
- 「中式表达」的条数记进基线表——这是写作进步的主要指标，比 LLM 的估分可靠得多。
- LLM 的 IELTS 估分普遍偏高 0.5–1 分，只看趋势不看绝对值。
- 每次批改里出现的「母语者会这么说」的句式进 Anki，标 `#write`。
- 同一类中式表达连续三次被指出，写进下一节的对照表里，写作前过一眼。

### 3. LLM 的边界

它对语法和用词的判断很准；对「中式表达」的判断大体准但偏保守——有些它标为 Chinglish 的其实是正式书面语，没问题；对结构和逻辑的意见要打折，它倾向于喜欢模板化的结构。design doc 的批改要给它足够的上下文，否则它会挑一些术语的毛病。


## 六、中式表达对照表

从我自己前几次批改里攒的，会持续加。写之前过一眼：

| 我写的 | 问题 | 母语者写法 |
|---|---|---|
| `With the development of technology, …` | 空洞的铺垫开头 | 删掉，直接说事 |
| `It is widely believed / acknowledged that…` | 考试腔，没有信息 | `Most people think…` 或直接说观点 |
| `We should pay attention to X.` | 动作不明 | `We need to watch X.` / `X matters because…` |
| `This will bring great convenience to users.` | `bring convenience` 是直译 | `This makes it much easier for users to…` |
| `The reason is because…` | 冗余 | `This is because…` / `The reason is that…` |
| `According to my experience, …` | 搭配错 | `In my experience, …` |
| `I very like it.` / `very` 修饰动词 | 语法 | `I really like it.` |
| `Please kindly…` | 过度礼貌，印式 / 中式邮件腔 | `Please…` / `Could you…` |
| `Thanks for your kindly help.` | `kindly` 不是形容词 | `Thanks for your help.` |
| `As we all know, …` | 预设读者同意 | 删掉 |
| `In a word, …` | 直译「一句话」 | `In short, …` |
| `On the one hand… on the other hand…` 用于两个同向的点 | 用法错，这是对比结构 | `First… second…` 或 `also` |
| `improve the efficiency of the process` | 名词化 | `make the process faster` / `speed up the process` |
| `Due to the fact that…` | 冗余 | `Because…` |
| `There are many people who think…` | 冗余 | `Many people think…` |
| `I think it is a good idea to…` 开头每句 | 弱化 + 万能词 | `We should…` / 直接说 |
| `some problems` / `many things` | 万能词 | 具体：`two bugs`, `three open questions` |
| `We have a lot of work to do to solve it.` | 空 | 说具体要做什么 |
| `Looking forward to your reply.` | 不错，但每封都用就是模板 | `Let me know what you think.` / `Talk soon.` |
| `Hope this email finds you well.` | 模板开头，母语者越来越少用 | 直接说事，或 `Hi X, quick question:` |


## 七、每日 / 每周方案

Q3 的读写日程（与第 03 篇的词汇共用这个季度；其他季度作为维持项：每两周一封邮件练习 + 每周一篇长文）：

| 时段 | 做什么 | 时长 |
|---|---|---|
| 上班路上 | 泛听（维持） | 20–30 min |
| 午饭 | Anki | 10 min |
| 早上刷手机的时间 | 读一篇 L2 短新闻，一段最多查一词 | 10 min |
| 下班路上 | 自言自语（维持）或复述今天读的新闻 | 10 min |
| 晚上 一三五 | 加卡 5 min → 写作：一 = 邮件 + 批改，三 = Task 2（限时 40 分钟只能写完主体，开头结尾隔天）或 design doc 段落，五 = 改周三的稿 + 批改 | 30 min |
| 晚上 二四 | 加卡 5 min → 读一篇 L3 长文的一半（15 min）→ 三句英文概括（10 min） | 30 min |
| 周六 | 真人口语（维持）；请老师看一眼本周的邮件，问「你会这么写吗」 | 30–45 min |

Q3 结束的目标：250 字作文中式表达从 9 条到 ≤ 4 条，语法错 ≤ 3；非专业长文阅读 ≥ 170 词/分钟；一封工作邮件 10 分钟内写完、不需要先想中文。


## 八、本文小结

- 能读文档读不动新闻是语域问题不是英语问题：通用高阶词、长句式、社会背景。阅读的目标是换语域，不是加量；查词一段最多一个。
- 写作只练三种体裁——邮件、design doc 段落、Task 2——它们的中式问题一样：铺垫、名词化、万能词、连接词滥用。
- 写法是先结构后句子：骨架先填，然后一句一个信息、动词优先、人做主语、删第一句、换万能词。邮件的语气是直接说事 + 理由 + 替代方案。
- LLM 批改只指出不重写，中式表达条数是主指标，估分只看趋势；被指出三次的写进对照表。


## 九、下周就可以做的事

- [ ] 把第五章的 prompt 存好；今晚用第 01 篇那篇 250 字作文跑一遍，数中式表达条数和第 01 篇比
- [ ] 用邮件骨架写一封「请同事周四前给数据」的邮件，批改，改，再批改
- [ ] 订阅一个 L2 新闻源，早上读一篇，一段最多查一词，查过两次的进 Anki
- [ ] 周末读一篇 Guardian Long Read，掐表，写三句概括
- [ ] 把批改里被指出的中式表达加进第六章的表，写作前过一眼
