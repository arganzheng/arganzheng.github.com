---
title: ArangoDB的索引学习
layout: post
---


ArangoDB内建了很多索引结构，用于解决不同的应用场景（个人感觉有点过多了。。）

* Primary Index:
	* 主键索引: _id, _key
	* unsorted hash index
	* 内存索引
* Edge Index
	* 边索引：_from, _to
	* hash index
	* 内存索引
* Hash Index: 精确查询
	* unique hash index
	* unique, sparse hash index
	* non unique hash index
	* non unique, sparse hash index
	* 内存索引
* Skiplist Index: 范围查询
	* unique skiplist index
	* unique, sparse skiplist index
	* non-unique skiplist index
	* non-unique, sparse skiplist index
	* 内存索引
* Persistent Index
	* a sorted index with persistence
	* implementation is using the RocksDB engine
	* is used as a secondary indexes like MySQL
	* O(log n) to get the primary key + O(1) to fetch the actural document
	* sorted index, can be used for point lookups, range queries and sorting operations
* Geo Index
	* uses Hilbert curves to implement geo-spatial indexes
	* [Using Hilbert curves and Polyhedrons for Geo-Indexing](https://www.arangodb.com/2012/03/31/using-hilbert-curves-and-polyhedrons-for-geo-indexing)
* Fulltext Index
	* word tokenization is done using the word boundary analysis provided by libicu
	* words are indexed in their lower-cased form
	* supports complete match queries (full words) and prefix queries, plus basic logical operations such as and, or and not for combining partial results.

**TIPS & NOTES**

1、ArangoDB的索引除了Persistent Index，其他的都是纯内存索引，因此内存占用率会比较高；另外，重启的时候所有内存索引需要重新构建（或者打开某个Collection，该Collection相关的索引也会构建），导致启动非常耗时而且耗CPU。这也是后来引入Persitent Index的原因。[Suggestion: Have indexes also on disk besides "only memory indexes" #209](https://github.com/arangodb/arangodb/issues/209#issuecomment-193838232)。

另外，即使是内存索引，如果 size of indexes > size of ram，那么也会被操作系统换出。[The kernel will swap only to avoid an out of memory condition](https://en.wikipedia.org/wiki/Swappiness)。另一方面，ArangoDB的数据是以内存映射文件(Memory-Mapped Files)的方式加载的，数据量稍微一大，必然会发生换页。


2、相对于其他的图存储，ArangoDB很友好的支持组合索引、数组索引和嵌套属性索引。

* Combined index over multiple fields
	* db.posts.ensureIndex({ type: "hash", fields: [ "name", "age" ] })
* Indexing array values
	* db.posts.ensureIndex({ type: "hash", fields: [ "tags[*]" ] });
* Indexing attributes and sub-attributes
	* db.posts.ensureIndex({ type: "hash", fields: [ "name" ] })
	* db.posts.ensureIndex({ type: "hash", fields: [ "name.last" ] })	 

3、ArangoDB还引入[Vertex-Centric Indexes](https://docs.arangodb.com/3.1/Manual/Indexing/VertexCentric.html)解决图遍历过程中的supernode问题

4、其实从功能上出发，可以更好的理解为什么需要这些索引类型：

* 精确匹配 => Hash(Unsorted)
* 范围查询 => Sorted => B+ Tree or Skiplist
* 模糊匹配 => Fulltext 
* 地理位置查询 => Geo

这些索引理论上来说应该是可以内存，也可以是持久化的。是否持久化应该是一个选项(flag)，而不是一种索引类型。


参考文章
-------

1. [Handling Indexes](https://docs.arangodb.com/3.1/Manual/Indexing/)
2. [Suggestion: Have indexes also on disk besides "only memory indexes" #209](https://github.com/arangodb/arangodb/issues/209#issuecomment-193838232)
3. [Frequently asked questions](https://www.arangodb.com/documentation/faq/#what-are-the-server-requirements-for-arangodb)
