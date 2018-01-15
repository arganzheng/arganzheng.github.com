---
layout: post
title: ElasticSearch如何支持深度分页
catalog: true
tags: [elasticsearch]
---


分布式环境下的分页
----------------

ES本身是支持分页查询的，使用方式跟MySQL非常类似：

* from: Indicates the number of initial results that should be skipped, defaults to 0
* size: Indicates the number of results that should be returned, defaults to 10

如：

```
GET /_search?size=5&from=10
```

但是跟MySQL不同，ES是分布式存储的，查询结果一般都是跨多个分片的(spans multiple shards)，每个shard产生自己的排序结果，最后协调节点(coordinating node)对所有分片返回结果进行汇总排序截断返回给客户端。所以深度分页（分页深度）或者一次请求很多结果（分页大小）都会极大降低ES的查询性能。

之前在使用Aerospike实现图数据库的时候就遇到这个问题（[Aerospike UDF学习笔记](http://arganzheng.life/aerospike-udf.html)） ，Aerospike本身不支持排序分页（包括简单的截断也不支持。。），但是它有一个Stream UDFs机制，可以实现这个功能：简单来说就是先在所有的node节点进行排序和截断(map)，最后在client端汇总做最后的排序和截断(reduce)，其实就是一个分布式的归并排序Merge Sort算法。

这个实现方式跟ES分页的实现方式是一样的。只不过ES是内建的，不需要你自己实现。但是两者都有一个巨大的性能问题，就是深度分页。假设某个index有 5 个primary shard。我们要获取第1000页的内容，页面大小是10，即使 10001~10010。那么每个分片需要本地排序产生前10010条记录，然后协调节点(coordinating node)需要对5个主分片返回的`10010*5=50050`条记录进行排序，最后只返回10条数据给客户端。其他的50040条全部都扔掉了。

但是现实中确实有需要深度遍历(scan)某个index的场景，那么怎么解决呢？


解决方案1：服务端缓存——Scroll & Scan
--------------------------------

从上面的分析我们可以看出，为了返回某一页记录，其实我们抛弃了其他的大部分已经排好序的结果。那么简单点就是把这个结果缓存起来，下次就可以用上了。根据这个思路，ES提供了[Scroll](https://www.elastic.co/guide/en/elasticsearch/reference/current/search-request-scroll.html) API。它概念上有点像传统数据库的游标(cursor)。

scroll调用本质上是实时创建了一个快照(snapshot)，然后保持这个快照一个指定的时间，这样，下次请求的时候就不需要重新排序了。从这个方面上来说，scroll就是一个服务端的缓存。既然是缓存，就会有下面两个问题：

1. 一致性问题。ES的快照就是产生时刻的样子了，在过期之前的所有修改它都视而不见。
2. 服务端开销。ES这里会为每一个scroll操作保留一个查询上下文(Search context)。ES默认会合并多个小的索引段(segment)成大的索引段来提供索引速度，在这个时候小的索引段就会被删除。但是在scroll的时候，如果ES发现有索引段正处于使用中，那么就不会对它们进行合并。这意味着需要更多的文件描述符以及比较慢的索引速度。

可以使用 [nodes stats API](https://www.elastic.co/guide/en/elasticsearch/reference/current/cluster-nodes-stats.html) 查看有多少个查询上下文:

```
GET /_nodes/stats/indices/search
```

scroll接口使用方式如下：

```
GET /old_index/_search?scroll=1m  // => 1. Keep the scroll window open for 1 minute.
{
    "query": { "match_all": {}},
    "sort" : ["_doc"],  // => 2. _doc is the most efficient sort order.
    "size":  1000 // => 3. Although we specified a size of 1,000, we get back many more documents. 
}
```

**说明**

1. `scroll=1m`表示保持这个快照的存活时间，这里是1分钟。
2. 深度分页其实最大的开销还是排序，`_doc`字段排序其实就是自然顺序。
3. 虽然这里我们指定了`size=1000`，但是实际上这个请求是分发给每一个分片的，所以我们每次获取 `n <= size * number_of_primary_shards`个文档。

服务端会返回一个`_scroll_id`字段，这是一个Base-64编码的字符串。下一次请求就可以把这个 `_scroll_id` 带上请求 `_search/scroll`来获取下一个批次的数据：

```
GET /_search/scroll // => 1. should not include the index or type name — these are specified in the original search request instead.
{
    "scroll": "1m", // => 2. The scroll parameter tells Elasticsearch to keep the search context open for another 1m.
    "scroll_id" : "cXVlcnlUaGVuRmV0Y2g7NTsxMDk5NDpkUmpiR2FjOFNhNnlCM1ZDMWpWYnRROzEwOTk1OmRSamJHYWM4U2E2eUIzVkMxalZidFE7MTA5OTM6ZFJqYkdhYzhTYTZ5QjNWQzFqVmJ0UTsxMTE5MDpBVUtwN2lxc1FLZV8yRGVjWlI2QUVBOzEwOTk2OmRSamJHYWM4U2E2eUIzVkMxalZidFE7MDs="
}
```

**说明**

1. 服务端将返回下一个批次数据，并且会返回一个新的`_score_id`，下次请求把新的`_score_id`带上就可以请求下一个批次了。
2. `scroll=1m` 只要确保 >= 每一个批次的消费时间即可，因为每一次scroll请求都会设置下一个过期时间。


当scroll过期后，查询上下文(Search Context)会自动被删除。因为保持查询上下文是很昂贵的，所以如果已经处理完某个scroll返回的数据，可以使用 clear-scroll API显式删除它:

```
DELETE /_search/scroll
{
    "scroll_id" : "DXF1ZXJ5QW5kRmV0Y2gBAAAAAAAAAD4WYm9laVYtZndUQlNsdDcwakFMNjU1QQ=="
}
```

可以支持批量删除:

```
DELETE /_search/scroll
{
    "scroll_id" : [
      "DXF1ZXJ5QW5kRmV0Y2gBAAAAAAAAAD4WYm9laVYtZndUQlNsdDcwakFMNjU1QQ==",
      "DnF1ZXJ5VGhlbkZldGNoBQAAAAAAAAABFmtSWWRRWUJrU2o2ZExpSGJCVmQxYUEAAAAAAAAAAxZrUllkUVlCa1NqNmRMaUhiQlZkMWFBAAAAAAAAAAIWa1JZZFFZQmtTajZkTGlIYkJWZDFhQQAAAAAAAAAFFmtSWWRRWUJrU2o2ZExpSGJCVmQxYUEAAAAAAAAABBZrUllkUVlCa1NqNmRMaUhiQlZkMWFB"
    ]
}
```

也可以一次性删除所有:

```
DELETE /_search/scroll/_all
```


**TIPS** 

scroll特别适合做全表扫描，ES的reindex接口内部就是使用scroll机制实现的。



参考文档
-------

1. [Elasticsearch Reference [6.1] » Search APIs » Request Body Search » Scroll](https://www.elastic.co/guide/en/elasticsearch/reference/current/search-request-scroll.html)
2. [Elasticsearch: The Definitive Guide [2.x] » Getting Started » Distributed Search Execution » Scroll](https://www.elastic.co/guide/en/elasticsearch/guide/current/scroll.html)
3. [Efficiently Handling Deep Pagination In A Distributed Search Engine](https://tech.shutterstock.com/2017/05/09/efficiently-handling-deep-pagination-in-a-distributed-search-engine/)


1. [Elasticsearch Reference [6.1] » Search APIs » Request Body Search » Scroll](https://www.elastic.co/guide/en/elasticsearch/reference/current/search-request-scroll.html)
2. []()