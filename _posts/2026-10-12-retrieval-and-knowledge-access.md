---
layout: post
title: 检索与知识接入：私域知识怎么进入模型（总纲）
subtitle: "Retrieval and Knowledge Access: How Private Knowledge Gets into the Model"
tags: [AI, LLM, AI-Application, RAG, Retrieval, Embedding]
catalog: true
---


## 内容简介

《检索与知识接入：私域知识怎么进入模型》是一组共七篇的系列文章，是[《AI 应用工程师学习地图》](/ai-application-engineer-learning-roadmap.html)第三层（L3）的正文。前两层讲清了模型这个组件怎么失效、花多少钱（L1），以及它每一步看到的上下文怎么组织（L2）；L2 的最后一篇把一个决策留给了这一层——**这份资料该全放进上下文、检索、还是让 agent 自己查**。本系列从这个决策的上游开始：通用模型不知道你的业务，私域知识有两条路进入它——**放进上下文**（检索、工具查询）或**放进权重**（微调、继续预训练）——先决定走哪条路，再讲检索本身。

"RAG = 向量数据库"是 2023 年的印象。到 2026 年，检索有三类各有适用条件、生产系统通常混用：**词法 / 精确**（grep、BM25）——Claude Code 团队公开说他们从向量库起步后放弃了它，因为对代码这类有精确标识符、单仓库、能多轮迭代的语料，agent 拿着 grep 自己搜"通常效果更好"，还避开了索引的过期、安全与维护问题；Cursor 则相反，用 Merkle 树同步、按块哈希缓存 embedding、把向量存在远端——两个最好的 coding agent 在架构层面的分歧，分歧点不是语义与词法谁准，是**谁付过期税**。**向量 / 语义**——企业文档、用户问法与文档措辞不一致、千万级语料，仍然是它的场地；embedding 模型的选择（Qwen3-Embedding、Voyage、Gemini Embedding、Cohere Embed v4 各有长处）、混合检索与 rerank 是这一类的工程。**结构化**——SQL、API、知识图谱、本体：解决的是"agent 该对什么对象执行什么动作"与关系推理，Palantir Ontology、GraphRAG 一类属于这里，它们不替代前两类。

它回答的问题是：

> **私域知识该进上下文还是进权重？三类检索各适合什么，coding agent 为什么只用 grep？文档怎么解析与切块？embedding 与 rerank 怎么选？检索什么时候该从流水线变成 agent 的工具？结构化知识怎么接？怎么知道检索出了问题还是生成出了问题？**

七篇沿这条线走。**第一篇**先决定路：按知识的类型（会变的事实、结构化业务数据、非结构化文档、领域术语、流程性技能、海量领域语料）选接入方式，说明为什么微调不是灌知识的工具、默认顺序为什么是 prompt → few-shot → 检索 / 工具 → 微调。**第二篇**讲三类检索，用 coding agent 的架构分歧做主线实例：Claude Code、Codex、Cline 的 agentic grep 与 Cursor、Windsurf 的向量索引各自换了什么、为什么把 grep 的经验搬到企业文档上会失败、为什么 embedding 存的是相似不是关系（问"谁调用了 chargeCard"向量索引给你长得像的，不是调用者）。**第三篇**讲文档处理：解析的三个梯级（规则抽取 → 布局模型 Docling / Marker / MinerU → 视觉语言模型 Mistral OCR 4 / PaddleOCR-VL——0.9B 参数的开放模型在 OmniDocBench 上 96.34 分超过 Gemini 3 Pro 的 92.91 与 GPT-5.2 的 86.59）、三个静默失败（阅读顺序、表格、页眉页脚）、分块策略、Anthropic 的 contextual retrieval（给每块加上下文让检索失败率降 49%、加 rerank 降 67%）、元数据里的权限。**第四篇**讲索引与检索：embedding 模型的选法（MTEB 的局限——版本不可比、二元相关性、榜单领先者未必是生产默认）、向量索引（HNSW / IVF、量化、过滤）、向量库的选择、BM25 + 向量的混合与 RRF、cross-encoder rerank 为什么是性价比最高的一步、增量更新与删除、**权限过滤必须在检索时做**。**第五篇**讲从流水线到 agentic retrieval：固定的 query → top-k → 生成 与"检索是 agent 的一个工具、模型自己决定查什么查几轮"各适合什么，query 改写、多跳、供应商内置的 file search，以及 agentic 检索对工具描述与上下文预算（L2）的要求。**第六篇**讲结构化知识：text-to-SQL 的真实水平与语义层的必要、本体作为业务世界的 harness（对象、属性、关系、动作、权限）、GraphRAG 与 LazyGraphRAG 解决什么（全局性问题、关系）、图谱构建的成本。**第七篇**讲评测与运营：检索与生成分开评（recall@k、MRR、nDCG vs faithfulness、answer relevance、context precision）、评测集从真实查询采样、在线信号、索引新鲜度、权限泄漏作为一个评测项（Slack AI 与 Copilot 的公开事件）。

系列的组织原则：**先决定要不要检索、检索哪一类，再讲怎么做**。第一、二篇是决策，第三到六篇是实现，第七篇是验证。每篇都把手段接回 L1 的失效模式（检索对付的是幻觉与知识截止）与 L2 的上下文预算（检索结果是上下文的一层，取多少、放哪由 L2 决定）。


## 为什么写这个系列？

### RAG 的问题不在向量库

2023–2024 年的 RAG 教程把检索等同于"切块、embedding、向量库、top-k"。这套流水线能在 demo 上跑通，上线后的问题几乎都不在向量库：**解析**把两栏 PDF 的两列逐行交错成流畅的胡话、把财务表格压成一段丢了行列；**分块**把一个函数切成两半、把"第 3 条"与它所属的章节分开；**查询**与文档的词汇不匹配（"报销标准" vs "差旅费用规定"）而向量相似度没有救回来；**权限**在生成时才过滤，模型已经在检索结果里看到了它不该看的；**评测**只看最终答案，分不清是没找到还是找到了没说对。一个公开的开源解析器基准的作者写道："文档解析是 RAG 流水线里最大的静默失败来源——一个打乱阅读顺序、丢掉表格、抹平标题的解析器产出干净的 markdown，同时悄悄毁掉检索质量。"本系列按这些真实的失败点组织，向量库只是第四篇的一节。

### 三类检索的分工在 2025–2026 年变得清楚

coding agent 的架构分歧是最好的教材。Anthropic 在 2025 年 9 月的 Agent SDK 文章里写得直白："语义搜索通常比 agentic 搜索快，但更不准确、更难维护、更不透明"，并建议"从 agentic 搜索开始，只在需要更快或更多变化时加语义搜索"；Sourcegraph 在 Cody 5.3 里移除了 embedding，改用自家的代码搜索；Cline 从设计上不索引。同时 Cursor 与 Windsurf 继续索引，并公开了为保持索引新鲜要付的代价：Merkle 树、按块哈希的 embedding 缓存、内容证明。这不是谁对谁错——代码满足词法检索的全部前提（精确标识符、可枚举范围、agent 能迭代），企业文档不满足。理解这个分歧，比记住"RAG 已死"或"RAG 永生"任何一句口号都有用。

### 解析与 embedding 的格局在一年内换了一遍

2025 年一年之内 Mistral OCR、OlmoOCR、Docling 1.0、Marker 1.0、MinerU、PaddleOCR-VL、DeepSeek-OCR 2 相继发布，一个 0.9B 的开放模型在文档理解基准上超过前沿通用模型；embedding 侧 Qwen3-Embedding-8B 以 70.58 登顶 MMTEB，Gemini Embedding、Voyage-3-large、Cohere Embed v4（128K 上下文、多模态）各占一角，MTEB 自身被批评用二元相关性抹平了差异。这一层的"选型"与 L1 第五篇讲的模型选型同构：榜单缩范围，自己的查询集做决定，且**换 embedding 模型意味着全量重建索引**——选型的隐藏成本。

### 现有材料的断层

- **框架教程**（LangChain / LlamaIndex 的 RAG 教程）讲怎么把组件接起来，不讲每个组件在什么条件下会失败、什么时候不该用；
- **向量库文档**讲索引参数，不讲解析、分块、混合、权限——它们不在向量库的责任范围内，但决定了效果；
- **论文**（Contextual Retrieval、GraphRAG、ColBERT、late chunking）各讲一个改进的收益，不讲它们怎么叠加、什么时候互斥；
- **coding agent 的博客与分析**讲了 grep vs 向量的分歧，但很少有人把它推广成"三类检索的适用条件"。

本系列想填补的是从"能搭一条 RAG 流水线"到"能为一份语料决定用哪类检索、把解析与分块的失败点找出来、选对 embedding 与 rerank、在检索时做权限、分清检索与生成谁出了错"之间的那段路。


## 适合哪些读者？

### 做知识库、客服、企业搜索的应用工程师

RAG 是你的主战场。第三、四、七篇是核心：解析、分块、混合检索、rerank、评测。

### 做 coding agent 或 agent 平台的工程师

第二、五篇：为什么代码用 grep、什么时候要索引、agentic retrieval 对工具与上下文的要求。

### 要接业务系统数据的工程师

第一、六篇：结构化数据不走向量库；SQL、API、本体、图谱各在什么条件下用。

### 技术负责人

第一、二、七篇：决定要不要微调、要不要建向量库、怎么衡量效果。


## 系列的整体主线

| 篇 | 主题 | 建立的东西 | 对付 L1 的哪条失效 | 与 L2 的关系 |
|---|---|---|---|---|
| 1 | 进上下文还是进权重 | 知识类型 → 接入方式的表；默认顺序 | 幻觉、知识截止 | 决定检索结果这一层存在与否 |
| 2 | 三类检索 | 词法 / 向量 / 结构化的适用条件；coding agent 的分歧 | 幻觉（找到依据） | 检索类型决定结果的形态与预算 |
| 3 | 解析与分块 | 三个梯级；三个静默失败；分块策略；contextual retrieval；元数据 | 幻觉（依据的质量） | 块是上下文的单位 |
| 4 | 索引、混合与 rerank | embedding 选法；索引结构；混合 + RRF；rerank；权限过滤 | 幻觉 | top-k 与排序决定放多少、放哪 |
| 5 | 流水线到 agentic | 两种形态的适用；query 改写；多跳；内置 file search | 知识截止（会变的事实随时查） | agentic 检索受 L2 预算与卸载约束 |
| 6 | 结构化知识 | text-to-SQL 与语义层；本体；GraphRAG | 幻觉（关系推理） | 结构化结果的呈现 |
| 7 | 评测与运营 | 检索 / 生成分开评；评测集；在线信号；权限泄漏 | 供应商变更（embedding 换代） | 与 L2 第六篇的门禁同构 |

三条贯穿的线：

- **先问知识是什么类型，再问怎么接。** 会变的事实永远不进权重；有 schema 的数据走 SQL / API / 本体；非结构化文档走检索；风格与格式才考虑微调。第一篇的表是全系列的入口。
- **检索的失败大多在检索之前。** 解析、分块、元数据决定了向量库里有什么；向量库的参数只决定怎么找。第三篇比第四篇重要。
- **检索与生成分开评。** 没找到与找到了没说对是两种病、两种药；混在一起就不知道该改哪一半。第七篇是全系列的验收。


## 章节结构与分章导读

### 1. 进上下文还是进权重：知识类型决定接入方式

> 核心问题：**为什么微调不是灌知识的工具？哪些知识进上下文、哪些进权重？默认顺序为什么是 prompt → few-shot → 检索 / 工具 → 微调？**

从一个常见的错误决策开始：团队想让模型"学会"公司的产品手册，于是拿手册做 SFT——结果模型记住了一部分、编造了另一部分、手册一更新全部作废、而且答不出"这句话出自哪一条"。微调擅长的是**行为**（风格、格式、术语用法、工具调用习惯），不是**事实**；事实要能引用、能更新、能按权限过滤，这三条微调都做不到。然后是知识类型表：会变的事实（价格、库存、政策、工单状态）→ 检索或直接查系统；结构化业务数据与关系（客户—订单—工单）→ SQL / API / 本体；非结构化文档 → 检索；领域术语与表达习惯 → few-shot，量大且稳定时微调；流程性技能 → 示例 + 工具，不够再 SFT；海量领域语料的整体适应 → 继续预训练（贵，且仍要配检索给事实与引用）。默认顺序与每一步的代价（越靠后越贵、越不灵活、越难追溯）；RAFT 一类"微调让模型更会用检索结果"的方法作为两条路的结合。微调的方法属于算法地图 L5，应用工程师要会的是这张表。

### 2. 三类检索：词法、向量、结构化——coding agent 为什么只用 grep

> 核心问题：**三类检索各适合什么？Claude Code 为什么放弃了向量库而 Cursor 没有？为什么 embedding 存的是相似不是关系？**

三类的表：词法 / 精确（grep、glob、BM25、全文索引）适合有精确标识符、可枚举范围、agent 能多轮迭代的语料，不适合词汇不匹配与千万级文档；向量 / 语义（embedding + ANN）适合非结构化文档、用户问法与文档措辞不一致、大规模语料，不适合精确匹配与关系推理；结构化（SQL、API、图谱、本体）适合有 schema 的数据与"对什么对象做什么动作"，不适合自由文本。然后是主线实例：Claude Code 的工具只有 Glob、Grep（基于 ripgrep）、LSP，团队公开说从 RAG 加本地向量库起步后放弃；Anthropic 的建议"从 agentic 搜索开始"；Cline、Sourcegraph Cody 5.3 同一方向；Cursor 的 Merkle 树同步、按块哈希缓存 embedding、远端向量库（turbopuffer）——为了保持索引与代码同步要付的全部机器；分歧的真正轴是**过期税**：embedding 在你重命名一个符号的瞬间过期，索引买到的是大仓库上的快速冷搜索与词汇不匹配的召回，agentic grep 买到的是零过期与不出机器。再讲一个两者都没有的东西：**关系**——embedding 是向量空间里的点，"近"是相似不是"调用"、"实现"、"依赖"，问"谁调用了这个函数"向量索引返回长得像的，结构化（LSP、图）才能回答。最后把经验推广：企业文档为什么不能照搬 grep（词汇不匹配、语料大、用户问法不同），什么时候三类要混用。

### 3. 文档解析与分块：检索失败的上游

> 核心问题：**解析有哪三个梯级、各在什么文档上够用？三个静默失败是什么？分块怎么选？contextual retrieval 为什么有效？元数据里为什么必须有权限？**

解析的三个梯级：**规则抽取**（PyMuPDF4LLM 一类读 PDF 自带的文本层，毫秒级、免费、对数字原生 PDF 够用、对扫描件无效）→ **布局模型**（Docling——IBM 2025 年捐给 LF AI & Data、MIT 许可，Marker，MinerU：布局检测 + 表格结构模型，本地运行，复杂表格与多栏的最强开源选项）→ **视觉语言模型**（Mistral OCR 4：每千页 4 美元、170 种语言、边界框与置信度、可自托管单容器；PaddleOCR-VL-1.6：0.9B 参数、Apache 2.0、OmniDocBench 96.34 分高于 Gemini 3 Pro 92.91 与 GPT-5.2 86.59；LlamaParse；DeepSeek-OCR 2）——唯一能可靠处理扫描、手写、密集财务表格与公式的梯级。三个静默失败：阅读顺序（两栏逐行交错）、表格（压成段落后数字失去行列）、页眉页脚（每块都被注入同一段噪声）。分块：固定、递归、语义、按结构（标题 / 函数 / 条款），块大小与重叠的权衡，代码与法律条款为什么要按结构切；late chunking（先整篇 embedding 再切）与 Anthropic 的 **contextual retrieval**（用模型给每块生成一段"这块在文档里的位置与含义"前置到块上再 embedding + BM25，检索失败率降 49%，加 rerank 降 67%——prompt caching 让这一步的成本可接受，L1 第四篇）。元数据：来源、时间、章节路径、**权限**——权限不在元数据里就无法在检索时过滤（第四篇）。

### 4. 索引、混合检索与 rerank

> 核心问题：**embedding 模型怎么选、MTEB 为什么不能直接用？向量索引与向量库怎么选？混合检索与 RRF 是什么？rerank 为什么是性价比最高的一步？权限为什么必须在检索时过滤？**

embedding 选型与 L1 第五篇同构：MTEB / MMTEB 缩范围（Qwen3-Embedding-8B 70.58、Gemini Embedding 001 68.32、Voyage-3-large、Cohere Embed v4、OpenAI text-embedding-3-large），但版本不可比、二元相关性抹平差异（ZeroEntropy 用分级相关性重标 28 个数据集后排名明显变化）、榜单领先者未必是生产默认；判据是语言与领域、上下文长度（Gemini 2K vs Cohere 128K）、维度与 Matryoshka（256 维的 3-large 仍胜 1,536 维的 ada-002）、许可、以及**换模型 = 全量重建索引**。向量索引：HNSW（内存、高召回）与 IVF（大规模、可磁盘）、标量与二值量化、带过滤的搜索（先过滤还是后过滤）。向量库：pgvector（小规模先用现有基础设施）、Qdrant、Milvus、Elasticsearch / OpenSearch（已有全文索引时的混合首选）、turbopuffer 一类对象存储后端——点名不展开。混合检索：BM25 抓精确匹配、向量抓语义，用 RRF（倒数排名融合）合并，不需要调权重；rerank：cross-encoder 对（查询，候选）对打分比双塔精确，只用于 top-k 后的精排，候选有 Cohere Rerank、Voyage rerank、BGE-reranker、Qwen3-Reranker、ColBERT 一类，增加几十到几百毫秒换明显的精度——性价比最高。增量更新与删除的工程（软删除、重建窗口）。**权限过滤在检索时做**：元数据过滤进查询，不能等生成后再删——模型已经看到了。

### 5. 从流水线到 agentic retrieval

> 核心问题：**固定流水线与 agentic retrieval 各适合什么？query 改写、多跳、多查询各解决什么？供应商内置的 file search 在协议上是什么？agentic 检索对工具描述与上下文预算有什么要求？**

经典流水线 query → embedding → top-k → 塞进 prompt → 生成，优点是延迟可控、成本固定、适合高并发问答；缺点是一次没找到就没找到，查询与文档不匹配时靠改写。改进：query 改写与扩展（HyDE 让模型先写一个假想答案再检索）、多查询、按元数据路由。agentic retrieval 把检索变成 agent 的工具：模型决定查什么、用哪类（grep 还是向量还是 SQL）、查几轮、结果够不够——召回压力小了（一次没找到换词再查），对工具描述（L1 第二篇）、上下文预算与卸载（L2 第四篇）、循环控制（L4）的要求高了。深度研究一类产品是它的极端形态。供应商内置的 file search（OpenAI Responses 的 file search、Gemini 的 file search）在协议上是服务端工具（L1 第二篇第三章）：省事，但你失去了分块、embedding、权限、排序的控制权。决策表：高并发定位型问答 → 流水线；复杂多跳 → agentic；小而稳定 → 全放上下文（L2 第六篇）。

### 6. 结构化知识：SQL、本体与 GraphRAG

> 核心问题：**text-to-SQL 的真实水平如何、为什么需要语义层？本体解决的是什么问题？GraphRAG 什么时候值得、成本多少？**

有 schema 的数据不走向量库。text-to-SQL 在企业真实 schema 上的准确率远低于 demo（Spider 2.0 一类企业级基准上前沿模型只有两成左右），原因是列名不可读、隐含的业务规则、多表关系——解法是**语义层**：把表映射成业务概念（"活跃客户"的定义写一次），模型对语义层而不是对裸表生成查询。**本体**更进一步：对象（客户、订单、设备）、属性、关系、每类对象允许的**动作**与权限——Palantir Foundry 的 Ontology 与其上的 AIP 是代表；它不是检索技术，是业务世界的 harness（L4 会从 agent 的角度再讲一次）：agent 看到的是"这个客户的这几笔订单"，能做的是有类型、有校验、有权限的动作。**GraphRAG**（Microsoft 2024）从文档抽取实体与关系建图、对社区做摘要，回答"这批文档整体说了什么"这类全局性问题——向量检索答不了；成本是建图时对全部文档跑一遍模型，LazyGraphRAG 把大部分工作推到查询时以降低索引成本；LightRAG 一类轻量变体。判断：需要关系推理或全局综合时才建图，多数定位型问答不需要。

### 7. 检索评测与运营

> 核心问题：**怎么分清是没找到还是找到了没说对？检索与生成各评什么？评测集从哪来？上线后看什么？权限泄漏怎么评？**

检索侧：recall@k（正确文档在前 k 个里的比例）、MRR、nDCG（分级相关性），评测集是（查询，相关文档）对——从真实查询采样、人工标相关文档，几十条就能开始；生成侧：faithfulness（答案的每个断言被检索到的材料支持）、answer relevance、context precision / recall（RAGAS 一类框架的定义）——用 LLM-as-judge 并与人工校准（L5）。分开评的意义：recall 低改检索（解析、分块、embedding、混合），faithfulness 低改生成（prompt、引用约束、guardrails）。在线信号：点击 / 采纳引用、追问率、"没找到"的比例、人工转接。运营：索引新鲜度（文档更新到可检索的延迟）、增量重建、embedding 模型换代的全量重建计划、成本（embedding 调用、向量库、rerank）。**权限泄漏作为评测项**：2024 年 Slack AI 被演示通过注入从私有频道抽取数据，Microsoft 365 Copilot 的 SharePoint 过度共享问题让企业发现"能搜到"暴露了本就配置错误的权限——用一组"不该看到"的查询定期测。

### 8. 系列总结与通关自测

一张总表回顾七篇各自回答的问题与必记的数字，逐篇的核心结论与常见误解，贯穿全系列的几条线，然后是三段自测——判断与计算、跨篇综合、面试题——加一份"读过 / 掌握 / 能教人"的判据。


## 贯穿全系列的实践线

本系列不设配套实验。每篇末尾有一节**实践建议**：

| 篇 | 实践建议的内容 |
|---|---|
| 1 | 给你的知识源列表按类型分类，标出各自的接入方式；找出被错误地"微调进去"的事实 |
| 2 | 对每个语料判断三类检索的前提是否满足；coding 场景先试 agentic grep |
| 3 | 抽 20 页最难的文档（表格、多栏、扫描）跑三个梯级的解析器人工比对；检查分块是否切断了结构 |
| 4 | 用 50 条真实查询建检索评测集，比较纯向量 / 混合 / 混合 + rerank 的 recall@5；在查询里加权限过滤 |
| 5 | 把一条固定流水线改成 agent 的检索工具，比较成本、延迟与多跳问题的成功率 |
| 6 | 对一个业务库写语义层的前十个概念；判断是否需要图 |
| 7 | 检索与生成分开建评测，上线后接在线信号；写十条"不该看到"的查询定期跑 |


## 阅读路径建议

### 第一遍怎么读

第一篇全文；第二篇的三类表与 coding agent 分歧；第三篇的三个静默失败与 contextual retrieval；第四篇的混合 + rerank；第七篇的分开评。约两小时。

### 完整学习路径

1 → 2 → 3 → 4 → 5 → 6 → 7，做每篇自测，最后做第八篇。

### 按问题读

- 答案经常"编"：第七篇分开评 → 第三、四篇；
- 用户问法与文档措辞不匹配：第三篇 contextual retrieval → 第四篇混合；
- 要接数据库 / 业务系统：第六篇；
- coding agent 要不要建索引：第二篇；
- 复杂问题一次检索不够：第五篇。


## 本系列的边界

- **不讲微调的方法**：SFT / LoRA / DPO / 继续预训练属于算法地图 L5；本系列只讲何时该微调与它不能做什么。
- **不讲 embedding 模型怎么训**：对比学习属于算法地图；本系列只讲怎么选、怎么评。
- **不讲向量检索的算力与部署**：属于 Infra 地图；本系列只讲索引结构的选择依据。
- **不讲 agent 循环本身**：agentic retrieval 里的循环控制、工具运行时、权限模型属于 L4；本系列只讲检索作为工具的设计要求。
- **不讲评测方法论本身**：LLM-as-judge 的校准、评测集的运营属于 L5；本系列只讲检索专属的指标与分开评的原则。


## 前置要求与说明

### 前置要求

- L1《模型作为组件》与 L2《Prompt 与上下文工程》——本系列直接引用失效模式表、成本公式、上下文分层与缓存规则；
- 后端基础：数据库、全文索引、API 的概念不需要解释。

### 版本与事实基线

以 **2026 年 9 月**的公开材料为准：

| 来源 | 内容 |
|---|---|
| Anthropic | Contextual Retrieval（2024-09）；Agent SDK 文章中关于 agentic 搜索 vs 语义搜索的建议（2025-09）；Claude Code 工具列表 |
| Cursor | 公开文档中的索引机制（Merkle 树、按块哈希缓存、远端向量库） |
| Sourcegraph / Cline | Cody 5.3 移除 embedding；Cline 的"不索引"声明（2025-05） |
| 解析 | Docling（LF AI & Data，MIT）、Marker、MinerU、PyMuPDF4LLM；Mistral OCR 4；PaddleOCR-VL-1.6、DeepSeek-OCR 2；OmniDocBench v1.6 |
| embedding / rerank | Qwen3-Embedding / Reranker（2025-06）、Gemini Embedding 001、Voyage-3-large、Cohere Embed v4、OpenAI text-embedding-3；MTEB / MMTEB 与 ZeroEntropy 的分级相关性重评 |
| 结构化 | Microsoft GraphRAG（2024-07）与 LazyGraphRAG（2024-11）、LightRAG；Spider 2.0；Palantir Ontology / AIP 公开文档 |
| 评测 | RAGAS 的指标定义；BEIR |
| 事件 | Slack AI 数据抽取演示（PromptArmor，2024-08）；Microsoft 365 Copilot 的 SharePoint 过度共享讨论（2024–2025） |

模型与工具的版本会变，文中引用时标注来源与日期。价格沿用 L1 第四篇的 2026 年 9 月价目。


## 章节目录

1. [进上下文还是进权重：知识类型决定接入方式](/knowledge-in-context-or-in-weights.html)
2. [三类检索：词法、向量、结构化——coding agent 为什么只用 grep](/three-kinds-of-retrieval-lexical-vector-structured.html)
3. [文档解析与分块：检索失败的上游](/document-parsing-and-chunking-for-retrieval.html)
4. [索引、混合检索与 rerank](/indexing-hybrid-search-and-reranking.html)
5. [从流水线到 agentic retrieval](/from-rag-pipelines-to-agentic-retrieval.html)
6. [结构化知识：SQL、本体与 GraphRAG](/structured-knowledge-sql-ontology-and-graphrag.html)
7. [检索评测与运营](/retrieval-evaluation-and-operations.html)
8. [系列总结与通关自测](/retrieval-and-knowledge-series-recap-and-self-test.html)


## 最终目标

读完本系列，面对一份要让模型使用的私域知识，读者应该能够：

| 追问 | 答案来自 |
|---|---|
| 这类知识该进上下文还是进权重？微调能解决吗？ | 第一篇 |
| 该用 grep、向量还是 SQL？coding 场景要不要建索引？ | 第二篇 |
| 这批 PDF 该用哪个梯级的解析器？块该按什么切？ | 第三篇 |
| 该用哪个 embedding？要不要混合与 rerank？权限怎么过滤？ | 第四篇 |
| 这个场景该固定流水线还是让 agent 自己查？ | 第五篇 |
| 这个业务库该建语义层、本体还是图？ | 第六篇 |
| 答错了——是没找到还是找到了没说对？ | 第七篇 |

这就是 L3 要建立的能力：**把"让模型知道我们的业务"从一个含糊的愿望变成一组可选择、可实现、可评测的接入方式**。
