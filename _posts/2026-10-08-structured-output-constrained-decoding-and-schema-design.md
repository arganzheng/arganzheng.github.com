---
layout: post
series: context-engineering
title: "Prompt 与上下文工程（03）：结构化输出——约束解码、schema 设计与失败修复"
subtitle: "Structured Output: Constrained Decoding, Schema Design and Failure Repair"
tags: [AI, LLM, AI-Application, Structured Output, JSON Schema]
catalog: true
---

L1 第二篇把结构化输出分成三个层次——提示、JSON 模式、schema 约束——并给了一句结论：schema 约束保证语法不保证语义。这一篇把那一句展开成一篇：约束解码在推理引擎里是怎样实现的、为什么它能给出"不可能生成不合法 JSON"这样的硬保证、为什么 strict 模式对 schema 有限制；然后是应用侧的两件事——**schema 怎么设计**才既可靠又不损害模型的推理（字段顺序就是生成顺序，拒答要有出口，枚举替代自由字符串），**解析层怎么写**才能处理 schema 保证不了的那一半（业务校验、修复重试、流式部分 JSON、截断）。

结构化输出是应用工程里少见的"硬"手段：它把一个概率行为（模型会不会按格式输出）变成一个确定性保证。理解它的机制能让你知道这个保证的边界在哪——边界之内可以放心去掉重试逻辑，边界之外一行校验都不能省。

本篇要回答的核心问题是：

> **约束解码是怎样实现的，为什么能给出硬保证，为什么 strict 模式对 schema 有限制？[^q0] schema 该怎么设计才不损害推理、不把拒答伪装成答案？[^q1] 解析层要处理哪些 schema 保证不了的失败？[^q2]**

## 一、总览

### 1. 保证的边界

| 层次 | 机制 | 保证 | 不保证 |
|---|---|---|---|
| 提示 | prompt 里描述格式 | 无 | 一切 |
| JSON 模式 | 引擎在解码时只允许合法 JSON 的 token | 语法合法的 JSON | 字段、类型、枚举 |
| schema 约束（strict） | 把 schema 编译成自动机，每步屏蔽不合法 token | 输出**必然**符合 schema：字段齐、类型对、枚举在范围、无多余字段 | 值的真实性、拒答不被伪装、推理质量、业务约束（日期合理、id 存在）、不被截断 |

本文的结构就是这张表的第三行：第二章讲"机制"列，第三章讲怎样设计 schema 让"保证"列覆盖更多、让"不保证"列的伤害更小，第四章讲解析层怎样处理"不保证"列。

### 2. 本文的章节安排

第二章约束解码的机制与 strict 模式的限制；第三章 schema 设计的七个模式；第四章解析层：校验、修复、流式、截断；第五章格式遵循率的评测；第六章实践建议。

## 二、约束解码

### 1. 从 schema 到自动机

约束解码的思路：在每一步采样之前，先算出"哪些 token 接在当前已生成的文本后面仍然可能构成一个合法输出"，把其余 token 的概率置零（logits 设为 $$-\infty$$），再从剩下的里采样。画成一步是这样：

![约束解码的一步：schema 允许的候选保留，其余 logit 置为负无穷，剩下的重新归一化后采样](/img/in-post/structured-output-constrained-decoding-mask.svg)

模型每生成一个 token 就要过一遍这三格，所以约束解码不是"生成完再检查"，而是**在生成的每一步把不合法的路堵死**——这也是它能给出硬保证、而"提示 + 事后校验"不能的原因。合法性由一个**自动机**判定：

- **正则 → 有限状态机**：Outlines（2023 年的论文《Efficient Guided Generation for LLMs》）把 JSON schema 先转成正则表达式，再编译成 FSM，预先为每个状态算出允许的 token 集合——采样时只是查表，几乎零开销。限制：正则无法表达递归结构（嵌套任意深的对象、数组里的对象里的数组），只能展开到有限深度。
- **上下文无关文法 → 下推自动机**：XGrammar（2024）与 llguidance（Guidance 项目的引擎）用 CFG 处理递归结构，用下推自动机加栈判定；XGrammar 的贡献是把 token 集合的计算分成"与上下文无关的部分"（预计算）与"依赖栈的部分"（运行时），把每步开销压到微秒级。
- **字节级与 tokenizer 的对齐**：一个 token 可能是半个 JSON 字符串、可能跨越引号，自动机要在字节层面工作并映射回 token 表——这是实现里最繁琐的部分，也是"同一个 schema 在不同 tokenizer 上要重新编译"的原因。

vLLM 与 SGLang 都把这几种后端做成可选（vLLM 的 structured outputs 配置可选 `xgrammar`、`guidance`、`outlines`），四家 API 的 strict 模式是同一原理的私有实现。

### 2. 为什么是硬保证

因为不合法的 token 在采样前就被置零了，模型**没有机会**生成它——不是"训练得很好所以很少错"，是数学上不可能。这与 JSON 模式的差别在于约束的粒度：JSON 模式的自动机只判"是不是合法 JSON"，schema 约束的自动机判"是不是符合这个 schema 的 JSON"。

硬保证的一个后果是**对模型有影响**：被屏蔽的 token 里可能有模型原本最想生成的那个。如果模型想写 `"status": "已完成"` 而 schema 要求 `enum["pending", "shipped", "completed"]`，它被迫在三者里选一个——选得对不对是语义问题，约束不管。这是第三章"schema 设计影响质量"的根源。

### 3. strict 模式为什么有限制

四家的 strict 模式都对 schema 有限制，OpenAI 的最明确：所有字段必须在 `required` 里（可选字段用 `type: ["string", "null"]` 表达）、`additionalProperties: false` 必须显式声明、嵌套深度与属性总数有上限、部分关键字（`minLength`、`pattern` 的某些形式、`oneOf` 的某些用法）不支持或有条件支持。原因都在自动机：

- 可选字段让状态数指数增长（每个字段在与不在两种分支）——要求全部 required 把它压成线性；
- `additionalProperties: true` 意味着任意键名，自动机要接受任意字符串作为键，状态空间失控；
- `minLength: 100` 一类的计数约束要在自动机里数字符，`pattern` 里的复杂正则要与 JSON 字符串的转义规则复合，实现成本高。

限制的另一面是**编译成本**：一个 schema 第一次使用要编译成自动机（OpenAI 文档提到首次请求的额外延迟，之后缓存）。这意味着 schema 应该是**稳定的**——每次请求动态生成一个不同的 schema（比如把候选 id 列表塞进 enum）会让每次都重新编译，且缓存失效。

### 4. schema 也占 token

Anthropic 的实现把 schema 注入为一段额外的 system 文本，实测约 50–200 token 的固定开销加上 schema 本身；它落在缓存前缀里（L1 第四篇）。OpenAI 与 Gemini 也要把 schema 以某种形式让模型"看到"——模型需要知道字段的含义才能填对，自动机只管形状。所以 schema 的字段名与描述（`description`）本身是 prompt 的一部分，写法影响质量。

## 三、schema 设计的七个模式

### 1. 字段顺序即生成顺序

约束解码按 schema 里属性的顺序生成（OpenAI 的 strict 模式明确保证按 schema 顺序输出）。这让字段顺序成为一个设计工具：**把需要推理的字段放在结论之前**。

```json
{
  "type": "object",
  "properties": {
    "evidence": {
      "type": "array", "items": { "type": "string" },
      "description": "支持判断的原文片段"
    },
    "reasoning": {
      "type": "string",
      "description": "从证据到结论的推理，两到三句"
    },
    "category": {
      "type": "string",
      "enum": ["refund", "exchange", "inquiry", "cannot_classify"]
    },
    "confidence": { "type": "number", "description": "0 到 1" }
  },
  "required": ["evidence", "reasoning", "category", "confidence"],
  "additionalProperties": false
}
```

模型先写证据、再写推理、最后给分类——结论以前面的内容为条件。反过来（先 `category` 再 `reasoning`）会让"推理"变成对已给出结论的事后辩护。2024 年的研究 *Let Me Speak Freely?* 发现强制严格格式会降低某些推理任务的表现，原因之一正是模型没有"先想"的空间；推理模型（L1 第三篇）在 thinking 阶段有了这个空间，问题缓解了很多，但字段顺序仍然有效——它让可见输出里也有一条从证据到结论的链，可以被审计。

### 2. 显式的拒答与不确定出口

strict 模式下模型**必须**产出合法 JSON。没有出口，不该回答的输入也会得到一个看似合理的答案——L1 第二篇的例子：乱码输入被分到 A / B / C 之一。每个 schema 都要有出口：分类加 `cannot_classify`，抽取字段允许 `null` 并配 `missing_reason`，判断题加 `insufficient_information`。OpenAI 的结构化输出响应里还有一个独立的 `refusal` 字段——模型因安全原因拒绝时不会硬凑 JSON，解析层要先检查它。

### 3. 枚举替代自由字符串

凡是取值可数的字段都用 `enum`：状态、类别、货币、语言、优先级。自由字符串会收到同义词、大小写变体、多余空格；枚举在 strict 模式下只会收到列表里的值。枚举的值用**机器友好的英文标识符**，中文标签放在 `description` 或应用侧映射——tokenizer 对英文标识符更省、更稳定。

### 4. 描述是 prompt

每个字段的 `description` 是模型填值时唯一的说明。"客户的意图"不如"客户这条消息想让我们做的一件事，取自 enum；无法判断时用 cannot_classify"。字段名也是：`amt` 不如 `refund_amount_cny`。

### 5. 限制深度与数量

嵌套三层以内；数组给 `maxItems`；可选字段用 `null` 联合而不是省略。深层可选嵌套既让自动机变大，也让模型在"要不要填这一层"上犯错。一个抽取任务需要几十个字段时，拆成几次调用（每次一组相关字段）通常比一个巨型 schema 更准、更便宜——因为每次调用的注意力预算集中。

### 6. 稳定的 schema

不要每次请求动态改 schema（把当日可选的商品 id 放进 enum）——编译缓存失效、缓存前缀失效。动态候选放在 prompt 的上下文里，schema 里用 `string` 加应用侧校验。

### 7. 用工具 schema 还是输出 schema

结构化输出成熟前的替代做法是定义一个 `report_result` 工具并强制调用；现在两者语义不同：**输出 schema**用于"这一步的最终结果"，**工具 schema**用于"我要调用某个动作"。混用的代价 L1 第二篇讲过（Fable 5.1 起强制工具调用报错）；原则是结果用 `output_config.format` / `text.format`，动作用 tools。

## 四、解析层

### 1. 校验两次

供应商的 strict 模式保证 schema 形状，解析层仍然用 Pydantic / zod 一类再校一次——两个原因：你的类型系统与供应商支持的 schema 子集可能有细微差别（日期格式、数值范围）；以及**业务校验**：`refund_amount` 不能超过订单金额、`order_id` 必须存在、`date` 不能在未来。业务校验是 schema 保证不了的那一半，一行都不能省。

### 2. 修复与重试

校验失败的处理分两级：

- **机械修复**：JSON 模式或提示层次下常见的 markdown 代码块包裹（去掉 ```` ```json ````）、尾随文字、单引号——解析前清理。strict 模式下不需要。
- **带错误信息的重试**：业务校验失败时，把错误信息作为一条新消息放回上下文（"refund_amount 12000 超过订单金额 899，请修正"），重试**一次**。这是 instructor 一类库的核心循环（Pydantic 校验 → 错误 → 重试），每次重试是一次付费调用（L1 第六篇），限一次；第二次仍失败走降级（人工、默认值、拒绝）。

### 3. 流式部分 JSON

结构化输出与流式可以同时用：输出按增量到达，中途是一个**不完整的 JSON 前缀**。两种消费方式：等 `done` 再解析（简单，失去流式的意义）；**增量解析**——用宽容的解析器（能解析不完整前缀的 partial JSON parser）把已到达的部分解析成部分对象，UI 逐字段渐进展示。OpenAI 与 Anthropic 的 SDK 都提供了流式结构化输出的辅助（返回按增量更新的部分解析对象）。注意：部分对象里的字段值可能是截断的字符串（"北京市朝" 还没到 "阳区"），展示可以、业务逻辑要等完整。

### 4. 截断

`max_tokens` 不够时 JSON 在中间被切断，`stop_reason` / `finish_reason` 是 `max_tokens` / `length`——这在 strict 模式下**仍会发生**（自动机保证每一步合法，不保证能走到终态）。解析层先查 `stop_reason`；推理模型上 `max_tokens` 含思考 token（L1 第三篇），要留够。截断的处理是提高 `max_tokens` 重试一次或缩小 schema。

### 5. 拒答与空值

先查 OpenAI 的 `refusal` 字段；再查 schema 里的出口字段（`cannot_classify`、`null`）；把它们路由到与"正常答案"不同的分支。一个常见 bug 是下游把 `cannot_classify` 当作一个普通类别统计，拒答率被藏在分类分布里。

## 五、格式遵循率的评测

结构化输出让"格式遵循率"从需要评测的指标变成了保证——但只在 strict 模式下、只对 schema 覆盖的部分。仍要评的：

| 指标 | 含义 | 怎么测 |
|---|---|---|
| schema 通过率 | strict 模式下应为 100%；JSON 模式 / 提示层次下是主要指标 | 解析 + schema 校验 |
| 业务校验通过率 | 值在业务允许范围内 | 应用侧校验 |
| 出口使用率 | `cannot_classify` / `null` 的比例 | 与人工标注的"应拒答"对比：过低是伪装成答案，过高是过度保守 |
| 截断率 | `stop_reason == max_tokens` 的比例 | 从 usage / stop_reason 统计 |
| 语义准确率 | 值对不对 | 评测集（L1 第五篇、L5） |
| 顺序效应 | 推理字段前置 vs 后置的准确率差 | 同一评测集两种 schema 对比 |

最后一行值得做一次：多数团队会发现"推理在前"提高 2–5 个百分点，代价是每次多几十个输出 token。

## 六、实践建议

1. **所有可数字段改 enum，所有 schema 加出口**，一小时内能做完，消除一类静默错误。
2. **推理字段前置**，在评测集上对比两种顺序。
3. **schema 稳定化**：把动态候选从 enum 挪到 prompt 上下文，看首 token 延迟与缓存命中率的变化。
4. **解析层清单**：`refusal` → `stop_reason` → schema 校验 → 业务校验 → 出口路由 → 带错误重试一次 → 降级。
5. **把出口使用率与截断率上仪表盘**，它们是格式层面仅剩的两个会静默变坏的指标。
6. **自托管时选后端**：vLLM / SGLang 下 XGrammar 对递归 schema 与开销的平衡最好；Outlines 对简单正则约束（固定格式的 id、日期）足够。

## 七、本文小结

- 约束解码把 schema 编译成自动机（正则 → FSM 用于 Outlines，CFG → 下推自动机用于 XGrammar / llguidance），每步把不合法 token 置零——保证是数学的不是统计的；strict 模式的限制（全部 required、`additionalProperties: false`、深度与关键字限制）来自状态空间与编译成本；schema 首次编译有延迟、应稳定；schema 本身占 token 且是 prompt 的一部分。
- schema 设计七个模式：字段顺序即生成顺序（推理字段前置）、显式拒答出口、枚举替代自由字符串、描述是 prompt、限制深度与数量、稳定的 schema、结果用输出 schema 动作用工具 schema。
- 解析层处理保证之外的一半：二次校验（类型 + 业务）、机械修复与带错误信息的一次重试、流式部分 JSON 的增量解析、`stop_reason` 截断、`refusal` 与出口路由。
- 评测格式层面剩下的指标：业务校验通过率、出口使用率、截断率、顺序效应。

## 八、自测

1. 一个 strict 模式的 schema 里 `category` 是 `enum[A, B, C]`。模型对某个输入"最想"输出 D。会发生什么？这说明硬保证的哪个后果？

   <details markdown="1">
   <summary>答案</summary>
   D 对应的 token 在采样前被置零，模型被迫从 A / B / C 里选概率次高的一个——输出合法但可能错。硬保证只约束形状，被屏蔽的可能正是模型的最优选择，语义质量因此受 schema 设计影响；解法是加出口（`other` / `cannot_classify`）让"都不是"有合法的表达。详见[第二章](#二约束解码)、[第三章](#三schema-设计的七个模式)。
   </details>

2. 为什么 OpenAI 的 strict 模式要求所有字段都在 `required` 里？可选字段该怎么表达？

   <details markdown="1">
   <summary>答案</summary>
   可选字段让自动机的状态数随字段数指数增长（每个字段在 / 不在两个分支），全部 required 把状态空间压成线性，编译与运行都可控。可选用类型联合表达：`"type": ["string", "null"]`，字段总在、值可为 null。详见[第二章](#二约束解码)。
   </details>

3. 一个团队每次请求把当天有效的 200 个商品 id 放进 schema 的 `enum`，发现首 token 延迟高且缓存命中率低。原因与改法？

   <details markdown="1">
   <summary>答案</summary>
   schema 每次不同 → 每次重新编译自动机（首次编译延迟）且 schema 注入的文本每次不同 → 缓存前缀失效。改法：schema 里 `product_id` 用 `string`，候选列表放在 prompt 的上下文里（动态段，末尾），应用侧校验 id 是否在当日列表中，失败带错误重试一次。详见[第三章](#三schema-设计的七个模式)。
   </details>

4. strict 模式下还会出现不完整的 JSON 吗？什么情况？解析层怎么发现？

   <details markdown="1">
   <summary>答案</summary>
   会。自动机保证每一步合法，不保证生成能走到终态；`max_tokens` 耗尽时输出在中间被切断，`stop_reason` / `finish_reason` 为 `max_tokens` / `length`。推理模型上 `max_tokens` 含思考 token，更容易发生。解析前先查 `stop_reason`，截断则提高上限重试一次或缩小 schema。详见[第四章](#四解析层)。
   </details>

5. 对比两个 schema：A 是 `{category, reasoning}`，B 是 `{evidence, reasoning, category}`。预期哪个准确率高、为什么、代价是什么？怎么验证？

   <details markdown="1">
   <summary>答案</summary>
   B。字段顺序即生成顺序，B 让结论以证据与推理为条件生成，A 的 reasoning 是对已给结论的事后辩护；*Let Me Speak Freely?* 指出严格格式压缩推理空间会损害表现，前置推理字段是缓解。代价是每次多几十到几百个输出 token。验证：同一评测集两种 schema 各跑 k 次比准确率，多数团队看到 2–5 个百分点的差距。详见[第三章](#三schema-设计的七个模式)、[第五章](#五格式遵循率的评测)。
   </details>

## 下一篇

[上下文预算与压缩：给每一部分定配额，超了怎么办](/context-budgeting-offloading-and-compaction.html)

[^q0]: 约束解码在每步采样前算出哪些 token 能让已生成文本仍构成合法输出，把其余 token 的 logits 置为 $$-\infty$$。合法性由自动机判定：Outlines 把 schema 转正则再编 FSM（预计算每状态允许的 token 集合，零开销但不能表达递归）；XGrammar / llguidance 用 CFG 加下推自动机处理递归结构，XGrammar 把与上下文无关的部分预计算、把每步开销压到微秒级；实现要在字节级与 tokenizer 对齐。因为不合法 token 在采样前就被排除，保证是数学的——不可能生成不合法 JSON。strict 的限制（全部 required、`additionalProperties: false`、深度 / 数量 / 关键字限制）来自状态空间（可选字段指数增长、任意键名不可枚举）与编译成本；schema 首次使用要编译（有延迟、后缓存），所以 schema 应稳定；schema 还被注入为文本占 token（Anthropic 约 50–200 token 开销）。详见[第二章](#二约束解码)。

[^q1]: 七个模式：（1）字段顺序即生成顺序——推理 / 证据字段放在结论前，避免"事后辩护"，缓解严格格式对推理的损害（*Let Me Speak Freely?*）；（2）显式出口——`cannot_classify`、`null` 加原因、`insufficient_information`，否则 strict 模式会把拒答伪装成答案，另查 OpenAI 的 `refusal` 字段；（3）可数字段用 enum、值用英文标识符；（4）`description` 与字段名是 prompt，写清含义与出口条件；（5）嵌套 ≤ 3 层、数组 `maxItems`、大抽取拆成多次调用；（6）schema 稳定，动态候选放 prompt 不放 enum；（7）结果用输出 schema、动作用工具 schema，不再用强制工具调用做结构化输出。详见[第三章](#三schema-设计的七个模式)。

[^q2]: schema 保证形状不保证：值的真实性与业务约束、拒答、截断、推理质量。解析层清单：先查 `refusal`（OpenAI）与 `stop_reason`（`max_tokens` 表示截断，strict 下仍会发生——提高上限重试一次或缩小 schema）；用 Pydantic / zod 二次校验类型并做业务校验（金额上限、id 存在、日期合理）；业务校验失败把错误信息放回上下文带错误重试**一次**（instructor 的循环），再失败走降级；JSON 模式 / 提示层次下先做机械修复（去代码块包裹、尾随文字）；流式下用宽容的 partial JSON 解析器增量展示，但业务逻辑等完整；把出口字段路由到与正常答案不同的分支，不让 `cannot_classify` 混进分类分布。监控业务校验通过率、出口使用率、截断率。详见[第四章](#四解析层)、[第五章](#五格式遵循率的评测)。
