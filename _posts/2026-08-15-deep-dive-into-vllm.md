---
layout: post
title: 大模型推理系统揭秘：从 vLLM 看 LLM Serving Infra 核心技术（目录）
tags: [AI, AI-infra, 大模型推理]
catalog: true
---

原文内容较长，现已按章节拆分为独立文章，便于分段阅读与分享。

> **NOTE** 本系列基于 vLLM v0.27.1（tag `6e448d0`, 2026-08-11）源码深度剖析。文中所有文件路径、类名和行号均以该版本为准；vLLM 迭代很快，阅读时请以你手上的版本对照。

## 目录

1. [为什么 LLM Serving 比传统 DL 推理难？](/deep-dive-into-vllm-01-why-llm-serving-is-hard.html)
2. [如何衡量一个 LLM Serving 系统？](/deep-dive-into-vllm-02-how-to-measure-llm-serving.html)
3. [鸟瞰 vLLM：一个请求如何穿过整个推理系统？](/deep-dive-into-vllm-03-vllm-request-lifecycle-overview.html)
4. [Scheduler：GPU 这一轮到底给谁用？](/deep-dive-into-vllm-04-scheduler-batch-and-fairness.html)
5. [KV Cache：LLM Serving 的第一号内存问题](/deep-dive-into-vllm-05-kv-cache-memory-core.html)
6. [GPU 执行：如何让每个 Token 算得更快？](/deep-dive-into-vllm-06-gpu-execution-kernels-and-graphs.html)
7. [Multi-GPU：一张卡不够时如何扩展？](/deep-dive-into-vllm-07-multi-gpu-scaling-strategies.html)
8. [模型适配：如何跟上变化极快的模型世界？](/deep-dive-into-vllm-08-model-adaptation-architecture.html)
9. [硬件解耦：如何不让芯片差异污染 Serving 核心？](/deep-dive-into-vllm-09-hardware-abstraction-and-portability.html)
10. [PD 分离：从单机 Serving 走向集群 Serving](/deep-dive-into-vllm-10-prefill-decode-disaggregation.html)
11. [Serving Infra 的下一站：从模型执行器到分布式智能操作系统](/deep-dive-into-vllm-11-future-of-serving-infra.html)
12. [回到源码：一次请求在 vLLM 内部的真实旅程](/deep-dive-into-vllm-12-source-code-request-walkthrough.html)

---

如果你希望，我也可以继续把每一篇再补上“上一章 / 下一章”的文末导航。
