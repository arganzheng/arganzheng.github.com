---
title: 搜索引擎中的相关性和排序截断
layout: post
---

一般来说，检索需要解决两个问题：

1. 召回：matching
2. 排序和截断: ranking

召回就是使用经典的倒排索引，从query词检索出相关的文档，文档是否匹配query词。但是这些文档哪些在前哪些在后呢？这就涉及到排序和截断的问题了，这是一个ranking问题，这往往是搜索引擎最具有差异性，也是最核心的部分（比如Google的PageRank算法成就了伟大的谷歌公司）。

常见的ranking scheme有：

* [tf-idf](http://en.wikipedia.org/wiki/Tf-idf): 最经典的ranking算法
* [Okapi BM25](http://en.wikipedia.org/wiki/Okapi_BM25): tf-idf的变种
* [PageRank](http://en.wikipedia.org/wiki/PageRank): 大名鼎鼎的Google Ranking算法

这里我们只介绍最经典的TF-IDF模式。


TF-IDF
------

TF: Term Frequency, term在某个document中出现次数。一个term在某个document出现越多次，表示该文档与该term相关性越大。这个算法够简单，大部分情况下也是比较符合逻辑的。

单纯比较词数(term count)的话，长文本明显比短文本有优势，但是并不意味着长文本比短文本更重要（更相关）。为了避免这种问题，这里引入了文档向量([unit vector](https://en.wikipedia.org/wiki/Unit_vector))的概念，对term count进行归一化([normalize](http://en.wikipedia.org/wiki/Vector_normalization))。实现很简单，就是把term count除以该文档的总词数，其实就是求百分比。

这样我们就得到TF的计算公式：

	TF(t) = t在文档中的出现次数 / 该文档的总词数

**TIPS**

在Lucene(ElasticSearch)中引入了一个叫做[Field-length norm](https://www.elastic.co/guide/en/elasticsearch/guide/current/scoring-theory.html#field-norm)的因子，来表示文档长度对权重的影响。计算公式如下：

	norm(d) = 1 / sqrt(numTerms)  

The field-length norm (norm) is the inverse square root of the number of terms in the field.


但是这样之后还是有一个问题，就是常用词往往出现频率很高，但是并不意味着他们很重要。不常见的词在一个文档中多次出现才真正意味着这个词对这篇文档很重要(很相关)。举个例子，”计算机视觉“，视觉明显比计算机要更具有区分度。如果只是简单的依据TF，那么很容易把包含计算机的文档排在包含视觉的文档前面。

所以，我们需要一个重要性调整系数，衡量一个词是不是常见词。如果某个词比较少见，但是它在这篇文章中多次出现，那么它很可能就反映了这篇文章的特性，正是我们所需要的关键词。

用统计学语言表达，就是在词频的基础上，要对每个词分配一个"重要性"权重。最常见的词（停用词，如"的"、"是"、"在"）给予最小的权重，较常见的词（"中国"，"计算机"）给予较小的权重，较少见的词（"蜜蜂"、"养殖"）给予较大的权重。

这个权重就叫做"逆文档频率"（Inverse Document Frequency，缩写为IDF），它的大小与一个词的常见程度成反比。

IDF: Inverse Document Frequency，逆文档频率。如果包含词条t的文档越少，也就是n越小，IDF越大，则说明词条t具有很好的类别区分能力。

IDF的计算公式也很简单：

	IDF(t) = log(文档总数 / 包含t的文档数)

其中 (文档总数 / 包含该词的文档数) 其实就是 文档频率 的倒数，这也是为什么称之为逆文档频率的原因。

现在我们已经知道TF和IDF的含义以及他们的计算公式，那么我们将他们结合在一起计算出一个综合的权重(score/weight)，其实很简单，就是简单的将其相乘：

	TF-IDF(t) = TF(t) * IDF(t)

需要注意的是TF是针对单个文档的统计，而IDF则是针对整个文档库的统计。可以看到，TF-IDF与一个词在文档中的出现次数成正比，与该词在整个文档库中的出现次数成反比。


综合考虑了词频和逆文档频率的IF-IDF算法已经是比较符合实际情况了，不过这个算法无法体现词的位置信息，不同位置的词，其重要性可能会有所不同，比如出现在title中词显然要比出现在body中的重要一些。针对这种情况，有一个简单的方法，就是对不同的字段(field)赋予不同的权重([field weight]())。算法可以跟TF类似——在越短的段落出现的词频权重越大。


Putting it together
-------------------

前面讨论的三个因子：term frequency, inverse document frequency, 和 field-length norm，会在构建索引期间分别计算和保存，然后Together, they are used to calculate the weight of a single term in a particular document。

Full-text relevance formulae, or similarity algorithms, combine several factors to produce a single relevance _score for each document. 

**TIPS**

在ES中每个field都有各自的反向倒排，因此TF/IDF中的document实际上指的是每个field。

在ES中可以清楚的看到权重的计算过程，只需要把explain设置为true:

	PUT /my_index/doc/1
	{ "text" : "quick brown fox" }

	GET /my_index/doc/_search?explain
	{
	  "query": {
	    "term": {
	      "text": "fox"
	    }
	  }
	}

返回的结果清楚的包含每个因子的值：

	weight(text:fox in 0) [PerFieldSimilarity]:  0.15342641	// 1
	result of:
	    fieldWeight in 0                         0.15342641		
	    product of:
	        tf(freq=1.0), with freq of 1:        1.0 // 2
	        idf(docFreq=1, maxDocs=1):           0.30685282 // 3
	        fieldNorm(doc=0):                    0.5 // 4

说明：

1. The final score for term fox in field text in the document with internal Lucene doc ID 0.
2. The term fox appears once in the text field in this document.
3. The inverse document frequency of fox in the text field in all documents in this index.
4. The field-length normalization factor for this field.


使用TF-IDF，我们可以对得到每个单词对应的权重，但是查询往往包含多个单词(term)，怎么把不同的term得到的权重加权起来呢？这就需要引入 [向量空间模型]() 和 [余弦相似性](http://en.wikipedia.org/wiki/Cosine_similarity) 了。


### [Vector Space Model](https://www.elastic.co/guide/en/elasticsearch/guide/current/scoring-theory.html#vector-space-model)

The vector space model provides a way of comparing a multiterm query against a document. The output is a single score that represents how well the document matches the query.

首先，需要把文档和query都表示为向量(vector)，一个向量(vector)其实就是一个一维数组，如 [1.2, 2.6, 5.3, 22.1] ,数组中的元素就是根据前面的TF-IDF计算出来的每个term的权重值。

举个简单的例子说明一下向量空间模型的实际应用。假设有个`query ＝ happy hippopotamus`, 召回阶段会对query进行分词然后检索求并集，假设最后召回下面三篇文档：

1. I am happy in summer.
2. After Christmas I’m a hippopotamus.
3. The happy hippopotamus helped Harry.

根据TF-IDF模式，我们可以和容易的计算出happy和hippopotamus在这三篇文档中的分值(其实每个term在每个文档的分值在建立索引的时候就已经计算好并保存起来了)，分别表示为:

1. Document 1: (happy,____________) ==> [2,0]
2. Document 2: ( ___ ,hippopotamus) ==> [0,5]
3. Document 3: (happy,hippopotamus) ==> [2,5]

然后再把query也向量化 query: (happy,hippopotamus) ==> [2,5]

现在 文档和query都以向量的形式表示在同一个向量空间，同一向量空间中两个向量的相识度其实就是他们之间的角度(angle between them)，角度越小越相似。用[余弦相似性](http://en.wikipedia.org/wiki/Cosine_similarity)就可以简单的解决这个问题，视觉效果如下图所示：

![elas_17in02.png]((/img/in-post/elas_17in02.png)


**TIPS** 

实际上的ranking过程会更复杂一些，包括 [Index-Time Boosting](https://www.elastic.co/guide/en/elasticsearch/guide/current/practical-scoring-function.html#index-boost) 和 [Query-Time Boosting](https://www.elastic.co/guide/en/elasticsearch/guide/current/query-time-boosting.html)，这篇文章有比较详细的说明：[Lucene’s Practical Scoring Functionedit](https://www.elastic.co/guide/en/elasticsearch/guide/current/practical-scoring-function.html)。


Sorting
-------

一般都是以相关性对检索结果进行排序，但是正如数据库查询一样，我们也可以针对某个或者多个field进行排序，比如根据时间排序，为了性能考虑，这时候往往会用到一个称之为 [Doc Values](https://www.elastic.co/guide/en/elasticsearch/guide/current/docvalues-intro.html#docvalues-intro) 的东东，主要用在如下场景：

* Sorting on a field
* Aggregations on a field
* Certain filters (for example, geolocation filters)
* Scripts that refer to fields

简单来说就是排序、过滤和聚合。具体可以参见如下文档：

1. [Doc Values](https://www.elastic.co/guide/en/elasticsearch/guide/2.x/docvalues.html)
2. [Deep Dive on Doc Values](https://www.elastic.co/guide/en/elasticsearch/guide/2.x/_deep_dive_on_doc_values.html)


总结
---

TF/IDF算法，包含以下三个因子：

1、Term frequency:

	How often does the term appear in the field? The more often, the more relevant. A field containing five mentions of the same term is more likely to be relevant than a field containing just one mention.

2、Inverse document frequency:

	How often does each term appear in the index? The more often, the less relevant. Terms that appear in many documents have a lower weight than more-uncommon terms.

3、Field-length norm:

	How long is the field? The longer it is, the less likely it is that words in the field will be relevant. A term appearing in a short title field carries more weight than the same term appearing in a long content field.

1. Individual queries may combine the TF/IDF score with other factors such as the term proximity in phrase queries, or term similarity in fuzzy queries.
2. When multiple query clauses are combined using a compound query like the bool query, the _score from each of these query clauses is combined to calculate the overall _score for the document (使用向量空间模型）。


参考文档
-------

1. [Controlling Relevance](https://www.elastic.co/guide/en/elasticsearch/guide/current/controlling-relevance.html)
2. [Sorting and Relevance](https://www.elastic.co/guide/en/elasticsearch/guide/current/sorting.html)
3. [相关性简介](http://es.xiaoleilu.com/056_Sorting/90_What_is_relevance.html) ElasticSearch权威指南(中文版)
4. [How scoring works in Elasticsearch](https://www.compose.com/articles/how-scoring-works-in-elasticsearch/)
5. [Optimizing Search Results in Elasticsearch with Scoring and Boosting](https://qbox.io/blog/optimizing-search-results-in-elasticsearch-with-scoring-and-boosting)
6. [Advanced Scoring in elasticsearch](https://jontai.me/blog/2013/01/advanced-scoring-in-elasticsearch/)
7. [How to Implement a Search Engine Part 1: Create Index](http://www.ardendertat.com/2011/05/30/how-to-implement-a-search-engine-part-1-create-index/)
8. [http://www.ruanyifeng.com/blog/2013/03/tf-idf.html](TF-IDF与余弦相似性的应用（一）：自动提取关键词)
9. [TF-IDF与余弦相似性的应用（二）：找出相似文章](http://www.ruanyifeng.com/blog/2013/03/cosine_similarity.html)
10. [BM25 The Next Generation of Lucene Relevance](http://opensourceconnections.com/blog/2015/10/16/bm25-the-next-generation-of-lucene-relevation/)


