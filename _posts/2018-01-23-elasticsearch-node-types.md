---
title: ElasticSearch的节点类型
layout: post
tags: [elasticsearch]
catalog: true
---


集群中的每一个ElasticSearch实例都可作为一个节点存在，相同集群名字的节点构成同一个ElasticSearch集群。

默认情况下，集群中的每个节点都可以处理 [HTTP](https://www.elastic.co/guide/en/elasticsearch/reference/current/modules-http.html) 或者 [Transport](https://www.elastic.co/guide/en/elasticsearch/reference/current/modules-transport.html) 请求。 其中 transport 层专门用于节点和[Java TransportClient](https://www.elastic.co/guide/en/elasticsearch/client/java-api/6.1/transport-client.html)之间的通讯；而HTTP层则只供REST客户端使用。

集群中所有的节点都知道集群中其他所有的节点，可以将客户端请求转发到适当的节点。

逻辑上，从职责上划分，ElasticSearch节点有以下类型：

* Master node: 主节点。当一个节点配置`node.master: true`（默认）的时候，它有资格被选作为主节点，控制整个集群。
* Data node: 数据节点。`node.data: true`（默认）。该节点保存数据和执行数据相关的操作，如增删改查，搜索，和聚合。
* Ingest node: 提取节点。`node.ingest: true`（默认）。Ingest node可以通过[ingest pipeline](https://www.elastic.co/guide/en/elasticsearch/reference/current/pipeline.html)对文档执行预处理操作，以便在索引文档之前对文档进行转换或者增强。
* Tribe node: 部落节点。当一个节点配置`tribe.*`的时候，它是一个特殊协调节点，它可以连接多个集群，在所有连接的集群上执行搜索和其他操作。

一个节点默认情况下可以同时是master、data和ingest节点。当集群增大时，最好将其分开配置。

**说明** 

1、Client node

当主节点和数据节点配置都设置为false的时候，该节点只能处理路由请求，处理搜索，分发索引操作等，从本质上来说该客户节点表现为智能负载平衡器。独立的客户端节点在一个比较大的集群中是非常有用的，他协调主节点和数据节点，客户端节点加入集群可以得到集群的状态，根据集群的状态可以直接路由请求。 

警告：添加太多的客户端节点对集群是一种负担，因为主节点必须等待每一个节点集群状态的更新确认！客户节点的作用不应被夸大，数据节点也可以起到类似的作用。配置如下：

```
node.master: false 
node.data: false
```

2、Coordinate node(协调节点)

当一个请求会跨越多个节点的时候（如search或者bulk-indexing），这时候会引入一个协调节点的概念。

从流程上来讲，ES的一次请求非常类似于Map-Reduce操作。在ES中对应的也是两个阶段，称之为scatter-gather。客户端发出一个请求到集群的任意一个节点，这个节点就是所谓的协调节点，它会把请求转发给含有相关数据的节点(scatter阶段)，这些数据节点会在本地执行请求然后把结果返回给协调节点。协调节点将这些结果汇总(reduce)成一个单一的全局结果集(gather阶段)。


参考文档
-------

1. [Elasticsearch Reference [6.1] » Modules » Node](https://www.elastic.co/guide/en/elasticsearch/reference/current/modules-node.html)