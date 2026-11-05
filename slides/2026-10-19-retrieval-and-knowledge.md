---
layout: slides
title: "检索与知识接入：私域知识怎么进入模型"
subtitle: "系列精华 · 七篇正文每篇一页：事实进上下文、行为进权重"
permalink: /slides/retrieval-and-knowledge.html
series: retrieval-and-knowledge
date: 2026-10-19 23:30:00 +0800
author: arganzheng
description: "《检索与知识接入》系列的分享用幻灯片：进上下文还是进权重、词法 / 向量 / 结构化三类检索按前提选、解析与分块是失败的上游、索引 / 混合检索 / rerank、从流水线到 agentic retrieval、SQL / 本体 / GraphRAG、检索与生成分开评。"
theme: white
transition: slide
---

## 这个系列的一句话主张

> **事实进上下文、行为进权重**；检索有**三类按前提选**、失败大多在**检索之前**；**检索与生成分开评**。

<aside class="notes" markdown="1">
总纲：/retrieval-and-knowledge-access.html。
</aside>

---

## 几篇怎么连起来

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 170}}}%%
flowchart TB
    K1["01 进上下文还是进权重<br/>事实进上下文、行为进权重"] --> K2["02 三类检索<br/>词法 / 向量 / 结构化，按前提选"]
    K2 --> K3["03 解析与分块<br/>检索失败的上游"] --> K4["04 索引、混合与 rerank"]
    K4 --> K5["05 流水线 → agentic retrieval"]
    K2 --> K6["06 结构化知识<br/>SQL + 语义层、本体、GraphRAG"]
    K3 & K4 & K5 & K6 --> K7["07 检索评测与运营<br/>没找到 vs 找到了没说对"]
```

---

## 01 · 进上下文还是进权重

**事实进上下文、行为进权重——微调学分布不学事实，默认先 prompt 再检索**

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 230}}}%%
flowchart LR
    Q["模型不知道的这件事，是什么类型？"] --> F{"是<b>事实</b>（知道什么）<br/>还是<b>行为</b>（怎么做）？"}
    F -- "事实" --> C["<b>进上下文</b><br/>每秒变的（库存、工单）→ 工具直接查<br/>每天变的（政策、文档）→ 检索<br/>有 schema 的（客户、订单、关系）→ SQL / 本体"]
    F -- "行为" --> B["<b>先用便宜的，再考虑权重</b><br/>术语、格式、风格 → few-shot + 术语表<br/>流程性技能 → SKILL.md<br/>稳定且仍不够 → 微调（SFT）"]
    C -. "回答有可核对的来源，<br/>更新当天生效" .-> R["共同点"]
    B -. "微调学的是分布，<br/>不是事实" .-> R
```

<aside class="notes" markdown="1">
原文 /knowledge-in-context-or-in-weights.html。知识类型表六行。
</aside>

<!-- v -->

### 要点

| 四问 | 任一为「是」→ 不进权重 |
|---|---|
| 会变？ | 价格、政策、库存 |
| 有 schema？ | 表、字段、关系 |
| 要引用？ | 合规、客服 |
| 按用户不同？ | 权限、个性化 |

- *Fine-Tuning or Retrieval?*：检索显著优于微调；RAFT 微调的是「怎么用检索」不是知识本身
- 「把手册 SFT 进模型就学会了」——SFT 学分布不学事实

---

## 02 · 三类检索：coding agent 为什么只用 grep

**词法、向量、结构化各有前提——coding agent 只用 grep 是前提全中**

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 300}}}%%
flowchart TB
    Q["一条查询"] --> L & V & S
    subgraph L["词法 / 精确：grep、BM25"]
        direction TB
        L1["前提：有精确标识符"] --> L2["买到：零过期、可解释"] --> L3["付出：词汇不匹配"]
    end
    subgraph V["向量 / 语义：embedding + ANN"]
        direction TB
        V1["前提：问法 ≠ 写法"] --> V2["买到：语义召回"] --> V3["付出：索引要维护"]
    end
    subgraph S["结构化：SQL、API、图谱"]
        direction TB
        S1["前提：有 schema、问关系"] --> S2["买到：精确关系查询"] --> S3["付出：要 schema"]
    end
    L & V & S --> H["生产系统几乎总是混用"]

```

<aside class="notes" markdown="1">
原文 /three-kinds-of-retrieval-lexical-vector-structured.html。
</aside>

<!-- v -->

### 要点

| 产品 | 选择 |
|---|---|
| Claude Code | 放弃向量库，agentic grep；Anthropic「从 agentic 搜索开始」 |
| Cursor | Merkle 树 + 块哈希缓存 + turbopuffer——写路径付过期税 |
| Cody 5.3 | 移除 embedding |

- **embedding 是点不是边**：「谁调用了这个函数」用 LSP / 图，不用向量库
- 「Claude Code 不用向量库所以向量检索过时了」——代码满足词法三前提，企业文档不满足
- 「向量索引比 grep 省 token」——检索结果作为上下文进每轮，噪声多时更费

---

## 03 · 文档解析与分块：检索失败的上游

**解析规则→布局模型→VLM 按需升级；三个静默失败在检索之前**

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 190}}}%%
flowchart TB
    F["文件<br/>PDF · Office · 扫描件"] --> P{"梯级"}
    P -->|"数字原生"| R1["① 规则抽取"]
    P -->|"复杂表格"| R2["② 布局模型"]
    P -->|"扫描"| R3["③ 视觉 OCR"]
    R1 --> S["结构化文本"]
    R2 --> S
    R3 --> S
    S --> C["分块 + contextual 前缀"]
    C --> M["元数据"]
    M --> I["索引（第四篇）"]

    classDef step fill:#fff7e0,stroke:#c98a00,stroke-width:2px,color:#222
    classDef dec fill:#eef6ff,stroke:#5b8fd6,color:#222
    class F,R1,R2,R3,S,C,M,I step
    class P dec
```

<aside class="notes" markdown="1">
原文 /document-parsing-and-chunking-for-retrieval.html。
</aside>

<!-- v -->

### 要点

| 量 | 数 |
|---|---|
| OmniDocBench | PaddleOCR-VL-1.6（0.9B）96.34 > Gemini 3 Pro 92.91 > GPT-5.2 86.59 |
| Mistral OCR 4 | 每千页 \$4 |
| contextual retrieval（Anthropic） | 失败率 −49%，加 rerank −67% |
| 块大小 | 200–800 token |

- 「直接调 OCR API」——开放文档 VLM 更准更便宜；先规则抽取按需升级
- 「固定长度切一切」——切断函数、条款、表格

---

## 04 · 索引、混合检索与 rerank

**BM25 + 向量 RRF 是默认形态；rerank 只精排 top-20 是性价比最高一步**

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 200}}}%%
flowchart LR
    Q["查询"] --> B["BM25 top-50"] & V["向量 top-50"]
    B & V --> R["RRF（k = 60）→ top-20"] --> RR["rerank → top-5"] --> G["进上下文"]
    P["权限过滤"] -. "在召回时" .-> B & V
```

- Qwen3-Embedding-8B MTEB 70.58；「rerank 太慢不要」——只精排 top-20，几十到几百毫秒，且降 k 省钱

<aside class="notes" markdown="1">
原文 /indexing-hybrid-search-and-reranking.html。
</aside>

---

## 05 · 从流水线到 agentic retrieval

**流水线可控无多跳、agentic 多跳但成本不定——最常见是级联兜底**

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 460}}}%%
flowchart TB
    subgraph P["流水线 RAG（2023 起的标准形态）"]
        direction TB
        P1["query"] --> P2["混合召回 + rerank"] --> P3["top-k 进 prompt"] --> P4["生成"]
    end
    subgraph AG["agentic retrieval（检索是 agent 的一个工具）"]
        direction TB
        A1["决定先查什么"] --> A2["调 search_docs / sql"] --> A3["够不够？不够再查"] --> A1
        A3 -- "够了" --> A4["生成（带引用）"]
    end
    P -. "确定、成本固定" .-> X["取舍"]
    AG -. "能多跳、不可预测" .-> X

```

<aside class="notes" markdown="1">
原文 /from-rag-pipelines-to-agentic-retrieval.html。
</aside>

<!-- v -->

### 要点

| 规则 | 为什么 |
|---|---|
| 工具返回摘要 + 引用，不返全文 | 上下文爆炸；全文另取 |
| 步数上限必配 | 成本不定 |
| 内置 file search 是服务端工具 | 失去权限与分块控制 |

- 「全部改成 agentic」——成本延迟不定

---

## 06 · 结构化知识：SQL、本体与 GraphRAG

**text-to-SQL 裸 schema 只有两成要语义层；GraphRAG 只对全局/多跳**

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 440}}}%%
flowchart TB
    Q["答案在「关系」里，不在任何一个块里"] --> W{"数据长什么样？"}
    W -- "有 schema<br/>「Q3 营收」" --> S["<b>text-to-SQL + 语义层</b>"]
    W -- "对象关系 + 动作" --> O["<b>本体</b><br/>对象导航替代 join"]
    W -- "文档抽出的实体关系" --> G["<b>GraphRAG</b><br/>实体关系建图"]
    S & O & G --> C["都不替代向量检索：<br/>先检索线索，再结构导航"]

```

<aside class="notes" markdown="1">
原文 /structured-knowledge-sql-ontology-and-graphrag.html。
</aside>

<!-- v -->

### 要点

| 判据 | 数 |
|---|---|
| SQL 四道门 | 语法、只读、成本、超时 |
| 建图门限 | 全局 / 多跳问题比例低于一成不建图 |

- 「上 GraphRAG 提升 RAG 效果」——只对全局 / 多跳有用

---

## 07 · 检索评测与运营：没找到 vs 找到了没说对

**检索与生成分开评——没找到 vs 找到了没说对，药完全不同**

![检索侧与生成侧的两段评测](/img/in-post/retrieval-eval-two-stage-evaluation.svg){: style="max-height: 400px"}

<aside class="notes" markdown="1">
原文 /retrieval-evaluation-and-operations.html。
</aside>

<!-- v -->

### 要点

| 项 | 数 / 案例 |
|---|---|
| judge 一致率 | ≥ 85% 才能用 |
| 权限泄漏 | Slack AI 跨权限注入；Copilot 暴露 SharePoint 权限债 |
| 「不该看到」查询集 | 进 CI |

- 「合成评测集就够了」——措辞贴合原文高估 recall
- 「只看端到端准确率」——分不清检索还是生成；三个数字

---

## 常见误区（一）

- 「把手册 SFT 进模型就学会了」——SFT 学分布不学事实
- 「RAG 就是向量库」——三类之一
- 「向量库找『谁调用了这个函数』」——embedding 是点不是边
- 「固定长度切一切」——按结构切
- 「生成后过滤权限」——模型已看到
- 「MTEB 榜首直接用」——查询集决定；换模型全量重建
{: .fragments}

---

## 常见误区（二）

- 「只上向量就够」——BM25 + RRF 是默认
- 「全部改成 agentic」——级联
- 「对裸 schema 做 text-to-SQL」——两成；语义层
- 「本体是一种检索」——是 harness
- 「上 GraphRAG 提升效果」——只对全局 / 多跳
- 「权限泄漏是安全团队的事」——检索放大权限错误
{: .fragments}

---

## 下一步

- **同一路线**：《Prompt 与上下文工程》第 6 篇——上下文 vs 检索；《工具、Agent 与 harness》——agentic retrieval 的那个循环；《评测与可观测》——judge 一致率从哪来
- **算法侧**：《经典机器学习》第 4、9 篇——KNN 就是检索、MinHash 去重
- 原文总纲：`/retrieval-and-knowledge-access.html`；通关自测在系列总结
