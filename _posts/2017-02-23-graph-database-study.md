---
title: 图存储引擎学习笔记
layout: post
catalog: true
tags: [图数据库]
category: [技术]
---


存储和索引
---------

有两种存储方式，一种是Index Free Adjacency，不依赖于索引，直接存储关联的物理位置（内存指针，磁盘文件偏移量），这类的产品有neo4j, Titian, OrientDB，Graph Engine(@Microsoft), etc。第二种是Index-Based Adjacency，使用各种索引实现图的检索功能，这类的产品有 FlockDB(@Twitter), Unicore & TAO(@Facebook)，3. ArangoDB, etc.。

关于这两类存储方式的PK，可以参见Index Free Adjacency or Hybrid Indexes for Graph Databases。

简单来说，Index Free Adjacency如果能够保证大部分的节点都是在一个机器上（依赖于比较精准的图分割算法），那么对于多度关系的图遍历和图运算时有比较大的优势的。如果做不到，则意义不大。


业界主流Graph Databases调研
-------------------------

### 1. Index Free Adjacency

1、[neo4j](https://neo4j.com/) 

Cluster需要企业版，不考虑

2、Titan 

* distributed graph database，号称可以storing and querying graphs containing hundreds of billions of vertices and edges distributed across a multi-machine cluster.
* java(要求java8)
* Support for various storage backends: Cassandra, HBase, BerkeleyDB (还有基于Amazon DynamoDB的)
* Support for geo, numeric range, and full-text search via: ElasticSearch, Solr, Lucene
* 支持schame验证（edge label, property key, or vertex label）
* 支持Gremlin图查询语言
* 基于Titan的企业版：DataStax Enterprise Graph

3、OrientDB 

* Multi-Model: K-V, Document, Graph
* Java
* 支持类SQL图查询语言
* 高级功能(Query Profiler, Distributed Clustering configuration, Metrics recording, and Live Monitor with configurable Alerts)收费

4、[Microsoft Graph Engine](https://github.com/Microsoft/GraphEngine)

* 分布式内存KV
* C++ & C#
* run only on Windows Server!
* LIKQ图查询语言
* TSL，类似于protobuf或者thrift这样的IDL

### 2. Index-Based Adjacency

1、FlockDB（@Twitter） 

本质上来说不是一个图存储，而是针对粉丝的follow设计的，底层就是一个基于MySQL的分库分表封装。

2、ArangoDB 

* HashIndex: unique hash index，unique, sparse hash index，non-unique hash index, non-unique, sparse hash index
* Skiplist Index: unique skiplist index, unique, sparse skiplist index, non-unique skiplist index, non-unique, sparse skiplist index
* Persistent Index
* Geo Index
* Fulltext Index
* 跟OrientDB表象上很像，但是更新，更友好，性能也更好一些。
* C++
* AQL图查询语言
* Multi-Model: K-V, Document, Graph
* Hybrid Indexes: 使用索引支持属性搜索，支持多种索引类型： 

3、cayley 

* go
* multiple backend stores: LevelDB, Bolt, PostgreSQL, MongoDB(集群方式必须采用MongoDB)

4、[Facebook Unicorn](https://cs.stanford.edu/~matei/courses/2015/6.S897/readings/unicorn.pdf)

* 分布式内存KV
* 自然语言查询接口



