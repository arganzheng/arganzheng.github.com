---
title: ElasticSearch学习
layout: post
---

### 业界流行的开源搜索框架

1. Lucene: Java. Just a library
2. Sphinx: C++
3. Solr/SolrCloud: Java, uses Lucene internally for all its indexing and search.
	* Netfix
	* Digg
4. ElasticSearch: Java, uses Lucene internally for all its indexing and search.
	* Wikipedia
	* StackOverflow
	* GitHub


ElasticSearch的核心概念
-----------------------

1. near realtime(nrt)
2. cluster
3. node
4. index
5. type
6. document
7. shards & replica


### cluster 

代表一个集群，集群中有多个节点，其中有一个为主节点，这个主节点是可以通过选举产生的，主从节点是对于集群内部来说的。es的一个概念就是去中心化，字面上理解就是无中心节点，这是对于集群外部来说的，因为从外部来看es集群，在逻辑上是个整体，你与任何一个节点的通信和与整个es集群通信是等价的。 

### shards 

代表索引分片，es可以把一个完整的索引分成多个分片，这样的好处是可以把一个大的索引拆分成多个，分布到不同的节点上。构成分布式搜索。分片的数量只能在索引创建前指定，并且索引创建后不能更改。 

### replicas 

代表索引副本，es可以设置多个索引的副本，副本的作用一是提高系统的容错性，当个某个节点某个分片损坏或丢失时可以从副本中恢复。二是提高es的查询效率，es会自动对搜索请求进行负载均衡。 

### recovery 

代表数据恢复或叫数据重新分布，es在有节点加入或退出时会根据机器的负载对索引分片进行重新分配，挂掉的节点重新启动时也会进行数据恢复。 
### river 

代表ES的一个数据源，也是其它存储方式（如：数据库）同步数据到es的一个方法。它是以插件方式存在的一个es服务，通过读取river中的数据并把它索引到es中，官方的river有couchDB的，RabbitMQ的，Twitter的，Wikipedia的，river这个功能将会在后面的文件中重点说到。 

### gateway 

代表ES索引的持久化存储方式，es默认是先把索引存放到内存中，当内存满了时再持久化到硬盘。当这个es集群关闭再重新启动时就会从gateway中读取索引数据。es支持多种类型的gateway，有本地文件系统（默认），分布式文件系统，Hadoop的HDFS和amazon的s3云存储服务。 

### discovery.zen 

es内建的自动节点发现机制，也是默认的节点发现模块。ES是一个基于p2p的系统，它先通过广播寻找存在的节点，再通过多播协议来进行节点之间的通信，同时也支持点对点的交互。 

### Transport 

代表ES内部节点或集群与客户端的交互方式，默认内部是使用tcp协议进行交互，同时它支持http协议（json格式）、thrift、servlet、memcached、zeroMQ等的传输协议（通过插件方式集成）。

DB VS ES
--------

Relational DB  => Databases => Tables => Rows      => Columns
Elasticsearch  => Indices   => Types  => Documents => Fields

B-tree VS Inverted-index

[Inverted-index](http://www.elasticsearch.org/guide/en/elasticsearch/guide/current/inverted-index.html)


搜索体验优化（高级功能）
-----------------------

* pagination
* highlight
* realtime search
* range search: 价格，年龄、日期
* suggestions
* geolocation
* percolation
* fuzzy
* partial matching
* synonyms
* etc.

### ES的一些默认行为

1. In Elasticsearch, all data in every field is indexed by default.
2. 同一个索引下，fields from different types but with the same field name are indexed into the same inverted index. This means that the term frequencies from each type (and thus each language) are mixed together. [types and mappings-avoiding type gotchasedit](http://www.elasticsearch.org/guide/en/elasticsearch/guide/current/mapping.html#_avoiding_type_gotchas)
3. 排序的字段不能被分析（因为分析之后字段被打散成多个token了），但是不分析的字段又不能被搜索，如果一个字段既需要被搜索，同时又可以根据它排序，可以同时保存该字段的分析结果和原始字符串。In String sorting and multi-fields we explained that Elasticsearch cannot sort on an analyzed string field, and demonstrated how to use multi-fields to index the same field once as an analyzed field for search, and once as a not_analyzed field for sorting.
4. In Elasticsearch, all data in every field is indexed by default. That is, every field has a dedicated inverted index for fast retrieval. 


