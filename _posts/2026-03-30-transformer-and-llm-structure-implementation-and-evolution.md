---
layout: post
title: Transformer 原理与实现：从论文到手搓 GPT-2（总纲）
subtitle: "Transformer Principles and Implementation: From the Paper to GPT-2"
tags: [Transformer, LLM, AI, AI-Infra]
catalog: true
updated: 2026-10-11
redirect_from:
  - /transformer-and-llm-for-infra-engineers.html
comments_path: /transformer-and-llm-for-infra-engineers.html
---

> **系列总览。** 从 Transformer 的静态结构开始，沿着一个 token 的旅程，再亲手写出并训练 GPT-2 风格的小模型。本系列由四篇正文与一篇系列总结组成。

## 内容简介

Transformer 的结构图里每个部件为什么在那里？训练与推理的一次前向分别做什么？一个最小 GPT 如何从公式落到代码，再从随机权重训练成会续写的模型？四篇文章沿着同一条代码线回答这些问题。

## 章节安排

1. [Transformer 长什么样——从一句话到下一个 token](/transformer-architecture-from-a-sentence-to-the-next-token.html)
2. [一个 token 的旅程——训练侧与推理侧](/transformer-token-journey-training-and-inference.html)
3. [手搓 GPT（上）——nanoGPT model.py 逐行解析](/nanogpt-model-py-line-by-line.html)
4. [手搓 GPT（下）——nanoGPT train.py 与训一个会续写的模型](/nanogpt-train-py-and-training-a-model-that-writes.html)

[系列总结与通关自测](/transformer-principles-series-recap-and-self-test.html)

## 读完之后

接下来可以读[《现代 LLM 结构：从 GPT-2 到今天的演进》](/llm-architecture-evolution-roadmap-from-gpt2.html)：从熟悉的 GPT-2 出发，继续看今天的模型改了哪些结构，以及这些改动怎样改变成本。
