---
layout: post
category: life
series: english-for-going-abroad
title: "一年英语计划（06）：工作场景——standup、code review、1:1 与面试"
subtitle: "English at Work: Standup, Code Review, 1:1 and Interviews"
tags: [英语, 学习, 海外]
catalog: true
date: 2026-12-12
description: "开会插不上话，多数时候不是词汇问题，是没有打断、澄清、不同意、推迟结论的固定句式。这一篇按场景给句式：standup、会议、code review 评论的语气梯度、design doc 段落骨架、1:1，以及 behavioral 面试的 STAR 模板和把手头项目预写成三个英文故事。"
---

跨国会议上我有过很多次这样的时刻：对方讲完，我有一个明确的不同意见，脑子里中文已经组织好了，然后在「怎么开口」上停了两秒，话题已经到下一个了。事后想，我缺的不是那个意见的英文——是**开口的那半句**：`Can I push back on that?` 五个词，没有它后面的一切都说不出来。

这一篇是工作场景的句式和套路。它不占季度，是第 04 篇口语和第 05 篇写作的**语料**——每周挑一个场景作为话题。写在出国前，因为工作场景是我最确定会遇到、而且第一天就要用的。


## 一、总览

按场景组织，从最高频到最低频：standup → 会议 → code review → 1:1 → design doc → 面试。每个场景给三样东西：这个场景的**规则**（母语者默认的、不会说出来的），**固定句式**（按功能分），和一个**练法**。

| 章 | 场景 | 频率 |
|---|---|---|
| 二 | Standup | 每天 |
| 三 | 会议：打断、澄清、不同意、推迟结论 | 每天 |
| 四 | Code review：评论的语气梯度 | 每天 |
| 五 | 1:1 | 每周 / 每两周 |
| 六 | Design doc 与技术讨论 | 每月 |
| 七 | 面试：behavioral 与项目故事 | 找工作时 |
| 八 | 怎么练 | — |


## 二、Standup

规则：**短**。每人 30–60 秒，三件事——昨天、今天、卡点。母语者的 standup 不是汇报，是同步；讲细节的人会被打断（`let's take that offline`）。

> Yesterday I wrapped up the migration script and got it reviewed.
> Today I'm going to start on the retry logic for the worker.
> One thing I'm blocked on — I need access to the staging DB. [Name], could you help with that after this?

| 功能 | 句式 |
|---|---|
| 昨天做完 | `I wrapped up / finished / landed / shipped X.` |
| 昨天没做完 | `I'm still working through X — turned out to be trickier than expected.` / `X took longer than I thought; should be done today.` |
| 今天 | `Today I'll be on X.` / `I'm picking up X next.` / `Continuing with X.` |
| 卡点 | `I'm blocked on X.` / `I need a review on PR 123.` / `Quick question for [name] after standup.` |
| 没什么可说 | `Nothing blocking. Same as yesterday, still on X.` |
| 把细节推走 | `Happy to go into detail after, don't want to hold everyone up.` |

练法：第 04 篇的「预演」——每天上班路上用英语把今天的 standup 说一遍，30 秒。一个月后这三句话会是自动的，不用想。


## 三、会议：打断、澄清、不同意、推迟结论

会议是最难的场景：多人、有重叠、节奏快、话题跳。规则有几条母语者不会说出来的：

- **打断是正常的**，只要打断的方式对。等一个「完全的停顿」再说话，在英文会议里等不到——母语者在对方句尾语调下降的时候就接进去了。
- **不同意不需要铺垫**。中文里的「我觉得您说的很有道理，不过……」在英文里显得绕；`I see it differently` 就够了。
- **沉默会被理解为同意**。会后说「我其实不同意」在英文文化里很不好。
- **不懂就问**，当场问。母语者也经常 `sorry, what was that?`。

### 1. 打断和进入

| 功能 | 句式 |
|---|---|
| 想说话 | `Can I jump in here?` / `Sorry to interrupt, but—` / `Quick thing on that—` / `Before we move on—` |
| 接过话头 | `To add to that, …` / `Building on what [name] said, …` / `Yeah, and—` |
| 被打断后拿回来 | `Sorry, let me just finish this thought.` / `One more thing and then I'll hand over.` |
| 把话给别人 | `[name], you've looked at this more than I have — thoughts?` |

### 2. 澄清

| 功能 | 句式 |
|---|---|
| 没听清 | `Sorry, could you say that again?` / `I missed that last part.` |
| 没听懂 | `Just to make sure I'm following — you're saying X?` / `What do you mean by Y?` / `Can you give an example?` |
| 确认理解 | `So if I understand correctly, the plan is X. Is that right?` |
| 术语不认识 | `Sorry, what's Z? I'm not familiar with that term.`（完全正常，不丢人） |

### 3. 不同意

语气从软到硬：

| 程度 | 句式 |
|---|---|
| 提出顾虑 | `One concern I have is…` / `I'm a bit worried about X.` |
| 温和反对 | `I see it a bit differently.` / `I'm not sure that's the case — my understanding is…` |
| 明确反对 | `I'd push back on that.` / `I don't think that'll work, because X.` |
| 坚决反对 | `I strongly disagree — here's why.`（少用，用了要有理由） |
| 部分同意 | `I agree with the first part, but on X I think…` / `That's fair, though…` |

规则：**反对之后必须跟理由或替代方案**。`I don't think that'll work.` 然后停下，是最糟的——它把对方放在了辩护的位置。`I don't think that'll work because of X. What if we Y?` 是一次贡献。

### 4. 推迟结论、争取时间

| 功能 | 句式 |
|---|---|
| 想一下 | `Let me think about that for a second.` / `Good question — I don't have a strong opinion yet.` |
| 不确定 | `I'd need to check, but my guess is X.` / `Don't quote me on this, but…` |
| 推到会后 | `Can we take this offline?` / `Let me look into it and get back to you by tomorrow.` / `I'll follow up on Slack.` |
| 结束话题 | `I think we're aligned. Moving on?` / `Let's park this for now.` |

### 5. 开会前和会后

会前：把要说的两三点用英语写在纸上——不是全文，是要点和开头那半句。会后：如果有没说出来的，**立刻**发消息补上：`One thing I didn't get to say in the meeting — I think X because Y.` 这在英文职场里完全正常，比会后什么都不说好得多。


## 四、Code review：评论的语气梯度

Code review 是纯文字，没有表情和语调，所以语气全靠用词。中文程序员的评论常常要么太硬（`This is wrong.`），要么过软到看不出是不是要改。英文 review 有一套约定俗成的梯度，很多团队还会用前缀标出来：

| 级别 | 前缀 / 开头 | 意思 | 例子 |
|---|---|---|---|
| 可忽略 | `nit:` / `Nitpick:` | 小问题，改不改都行 | `nit: trailing whitespace` / `nit: I'd name this fetchUser rather than getUser, but no strong feelings.` |
| 建议 | `Suggestion:` / `Consider…` / `Optional:` | 我觉得更好，但你决定 | `Consider extracting this into a helper — it's used in three places.` |
| 疑问 | `Question:` / `Curious —` | 我没看懂，或者想知道为什么 | `Question: why do we retry here but not in the sibling function?` / `Is this intentional?` |
| 要改 | 没有前缀，直接说 + 理由 | 合并前应该改 | `This will throw when the list is empty — we should guard for that.` |
| 阻塞 | `Blocking:` / `This needs to change before we merge` | 不改不合 | `Blocking: this leaks the connection on the error path.` |
| 赞 | `Nice!` / `Love this.` / `TIL` | 好的地方也要说 | `Nice, much cleaner than what we had.` |

规则：

- **说代码，不说人**。`This function does X` 而不是 `You did X`。
- **陈述问题 + 影响**，不是指令。`This will fail when N = 0` 比 `Fix this` 有用，也不刺人。
- **问句是最好的软化**：`Should this be a set instead of a list?` 大家都知道你是在说应该用 set，但给了对方解释的空间。
- **不确定就说不确定**：`I might be missing something, but doesn't this double-count?`
- 收到评论时的回复：`Good catch, fixed.` / `Fair point — done.` / `I kept it as is because X, let me know if you feel strongly.` / `Addressed in the latest commit.`

练法：第 05 篇 Q3 的写作练习里，每两周用一个真实 PR（自己的或开源的），用英文写五条评论，覆盖五个级别，LLM 批改语气。


## 五、1:1

1:1 和会议不一样：私密、慢、可以讲长的想法。规则：**这是你的会**——大多数英文团队里 1:1 的议程由下属定，manager 是来听的。不带议程去 1:1，等 manager 问 `so how's it going`，然后回答 `good`——是最浪费的用法。

| 话题 | 句式 |
|---|---|
| 开场 | `A few things I wanted to bring up today: X, Y, and if we have time, Z.` |
| 汇报进展 | `X is on track.` / `X is slipping — I think we'll land a week late, here's why.` |
| 求助 | `I could use your help with X.` / `I'm not sure how to approach X — how would you think about it?` |
| 提困惑 | `Something's been bugging me about how we do X.` / `I want to get your read on the situation with Y.` |
| 谈成长 | `Longer term, I'd like to move toward X. What would I need to show for that?` / `What's one thing you think I should work on?` |
| 要反馈 | `Any feedback on how I handled X?` / `Is there anything you'd want me to do differently?` |
| 收反馈 | `That's helpful, thanks.` / `Can you give me an example of when that happened?`（不要辩解） |
| 提意见 | `One thing that would make my life easier is X.` |

练法：每次 1:1 前用英语写三个要点；第 04 篇的口语话题第 6 周就是它。


## 六、Design doc 与技术讨论

第 05 篇给了 design doc 段落的骨架，这里补技术讨论——白板前或者线程里争论方案的场景。它和普通会议的区别是**要讲清一个技术判断的推理**：

| 功能 | 句式 |
|---|---|
| 提出方案 | `One option is to X. The upside is A, the downside is B.` / `What I'd propose is…` |
| 权衡 | `It's a trade-off between X and Y.` / `We're optimizing for X here, so I'd lean toward…` |
| 指出问题 | `The thing that worries me about that is…` / `That works until X happens — then we'd…` |
| 边界情况 | `What happens when N is zero / the network partitions / the cache is cold?` |
| 承认不确定 | `I haven't thought this all the way through, but…` / `Rough idea:` |
| 达成 | `OK, so we go with X, and revisit if Y turns out to be a problem.` / `Let's write that down.` |
| 记录决策 | `Decision: X. Rationale: Y. Rejected: Z because W.` |

技术讨论对我们是最容易的场景——背景知识和词汇都是现成的，缺的只是这些连接用的句式。练法：第 04 篇第 3 周的话题「讲一个技术决策」，加上让 AI 扮演反对者。


## 七、面试：behavioral 与项目故事

技术面试的算法和系统设计部分对语言的要求不高——边写边说、说不清可以画。真正考英语的是 **behavioral 面试**（`Tell me about a time when…`）和**项目介绍**。这两个都可以、也应该提前准备到「熟到不用想」。

### 1. STAR

每个 behavioral 问题的回答结构：**S**ituation（一两句背景）→ **T**ask（我要做什么）→ **A**ction（我具体做了什么——这是主体，60%）→ **R**esult（结果，最好有数字）。两分钟以内。

高频问题（每个准备一个故事，故事可以复用）：

| 问题 | 故事类型 |
|---|---|
| Tell me about a challenging project. | 技术难题 |
| Tell me about a time you disagreed with a teammate / your manager. | 冲突 |
| Tell me about a failure / a mistake you made. | 失败 + 学到的 |
| Tell me about a time you had to learn something quickly. | 学习 |
| Tell me about a time you led without authority / influenced a decision. | 领导 |
| Why do you want to leave / join / move abroad? | 动机 |

### 2. 三个项目故事

从手头做过的项目里选三个，每个写成一页英文：一句话概括 → 背景 → 我的角色 → 两三个关键决策及理由 → 遇到的最大困难 → 结果和数字 → 回头看会怎么做。三个故事覆盖不同类型（一个技术深、一个跨团队、一个有失败）。

写完做三件事：LLM 批改（第 05 篇的 prompt）；讲给 AI 听，让它当面试官追问（`Why did you choose X over Y? What would you do differently?`）；讲给真人老师听，让他说哪里没听懂。**追问**是重点——第一层的故事容易背，面试官问的是第二层和第三层。

### 3. 面试句式

| 功能 | 句式 |
|---|---|
| 争取时间 | `That's a good one, let me think of the best example.` |
| 开始故事 | `So, a couple of years ago I was working on…` / `The situation was…` |
| 转到自己 | `My role was…` / `What I did was…` / `I decided to…` |
| 结果 | `As a result, …` / `We ended up…` / `It cut latency by about 40%.` |
| 反思 | `Looking back, I'd…` / `What I took away from that was…` |
| 没经历过 | `I haven't been in exactly that situation, but the closest is…` |
| 反问 | `Can I ask — how does the team handle X here?` |


## 八、怎么练

这一篇的内容不需要单独的时间，它是第 04、05 篇的话题库：

| 场景 | 嵌入哪里 | 频率 |
|---|---|---|
| Standup | 上班路上预演 30 秒 | 每天 |
| 会议句式 | 第 04 篇 AI 对话，让 AI 扮演一个会打断、会反对的同事 | Q2 每周一次 |
| Code review | 第 05 篇写作练习，真实 PR 五条评论 | Q3 每两周 |
| 1:1 | 每次真实 1:1 前写三个要点（哪怕现在的 1:1 是中文的） | 每周 |
| 技术讨论 | 第 04 篇话题第 3 周，AI 当反对者 | Q2 |
| 面试 | 三个项目故事，Q2 末写、Q3 批改、Q4 讲给真人 | 一次准备，长期复用 |

固定句式的掌握判据很简单：**不用想就出来**。每周从本篇挑 5 个在 AI 对话里刻意用，用到自动为止；这一篇的句式总数大约 80 个，一年绰绰有余。


## 九、本文小结

- 插不上话是缺「开口的半句」，不是缺意见的英文。每个场景有二十来个固定句式，练到自动。
- Standup 短、三件事、细节推走；会议里打断正常、不同意不铺垫、沉默等于同意、反对必须跟理由或替代方案。
- Code review 有语气梯度（nit / suggestion / question / 直接说 / blocking），说代码不说人，问句是最好的软化。
- 1:1 是你的会，带议程去。技术讨论是最容易的场景，缺的只是连接句式。
- 面试的 behavioral 和项目介绍要提前准备到不用想：STAR 结构、三个覆盖不同类型的项目故事、被追问到第三层。


## 十、下周就可以做的事

- [ ] 明天上班路上用英语说一遍今天的 standup，30 秒
- [ ] 从第三章挑 5 个会议句式，本周 AI 对话里让它扮演会打断的同事，每个用一次
- [ ] 找一个自己最近的 PR，用英文写五条不同级别的评论，LLM 批改语气
- [ ] 下次 1:1 前用英语写三个要点（1:1 是中文的也写）
- [ ] 选出三个项目，各写一句话概括和「最大困难」那一段，先不求完整
