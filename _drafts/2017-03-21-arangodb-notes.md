---
title: ArangoDB学习笔记
layout: post
catalog: true
---


### 一、数据模型

Multi-Model Database
* K-V
* Document
* Graph: Property Graph Model
	* Vertices: schema-free objects
	* Edges: directed relations between vertices between objects
	* vertices and edges are documents 


### 二、API

HTTP/REST and AQL

* Document Queries
* Graph Queries
* Joins
* All can be combined in the same statement

Foxx Microservice Framework 

* Allows you to extend the HTTP/REST API by JavaScript code

底层API ：TODO


### 三、图遍历——hybrid-indexes

另一种方式是index-free adjacency，对应的开源图数据库有Neo4j、OrientDB等。关于两者的优劣有一个讨论：[Index Free Adjacency or Hybrid Indexes for Graph Databases](https://www.arangodb.com/2016/04/index-free-adjacency-hybrid-indexes-graph-databases/)。

具体实现。

ArangoDB默认会对 _id, _key, _from, _to 构建索引：

* Primary Index
	* _id, _key
	* unsorted hash index
* Edge Index
	* _from, _to
	* hash index


### 四、图存储



### 五、图索引

* 入口节点/边索引：
* 图遍历索引：使用各种索引

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

1、ArangoDB的索引除了Persistent Index，其他的都是内存索引，因此内存占用率会比较高；另外，重启的时候所有内存索引需要重新构建，导致启动非常耗时而且耗CPU。这也是后来引入Persitent Index的原因。[Suggestion: Have indexes also on disk besides "only memory indexes" #209](https://github.com/arangodb/arangodb/issues/209#issuecomment-193838232)。
2、相对于其他的图存储，ArangoDB很友好的支持组合索引、数组索引和嵌套属性索引。

* Combined index over multiple fields
	* db.posts.ensureIndex({ type: "hash", fields: [ "name", "age" ] })
* Indexing array values
	* db.posts.ensureIndex({ type: "hash", fields: [ "tags[*]" ] });
* Indexing attributes and sub-attributes
	* db.posts.ensureIndex({ type: "hash", fields: [ "name" ] })
	* db.posts.ensureIndex({ type: "hash", fields: [ "name.last" ] })	 

3、ArangoDB还引入[Vertex-Centric Indexes](https://docs.arangodb.com/3.1/Manual/Indexing/VertexCentric.html)解决图遍历过程中的supernode问题

### 六、缓存

* four different types of caches:
	* NoCache: This is degenerate and stores nothing
	* LruCache: This utilizes LinkedHashMap of the java.util package along with a custom method to remove the oldest entry when deallocating space—removeEldestEntry(); this makes it an adaptable LRU cache
	* SoftLruCache: An LruCache using soft values to store references and queues for garbage-collected reference instances
	* WeakLruCache: An LruCache using hard values to store references and queues for garbage-collected reference instances
* LRU-K page cache for nodes and relationships
* makes extensive use of the java.nio package(大量使用堆外内存)
* makes use of the SoftReference and WeakReference objects


### 七、事务

* Write-Ahead Log (WAL): ensures atomicity and durability in transactions, also used for failure recovery.
	* LogEntry: Commands + information for phases
	* Changes made in the transaction are collected as commands, 可幂操作
	* Recovery simply replays all transactions since the last safe point
	* LogEntry State:
		* Start: This indicates that the transaction is now active
		* Prepare: This indicates that the transaction is being prepped for a commit
		* OnePhaseCommit: This initiates a commit in EmbeddedGraphDatabase
		* TwoPhaseCommit: This initiates commits in a distributed scenario
		* Done: This indicates that a transaction is complete
		* Command (not an actual state): Encapsulation for the commands in the transaction
* Locking
	* RWLock(Read Write Lock): ReentrantReadWriteLock
	* LockManager: synchronization of RWLock accesses, or creation of the locks and, whenever required, passing an instance of the RAGManager and appropriate removal at times.
	* RAGManager(Resource Allocation Graph Manager): detect whether waiting on a resource can lead to possible future deadlocks
	* Wait-For Graph (WFG): stores a graph of the resources and threads waiting on a RWLock.


### 八、高可用(HA)

* master-slave cluster with master election
* atomic broadcast
	* used to send messages to all instances in a reliable manner
		* no loss of message
		* preserved message ordering 
		* no messages are corrupt or duplicated
	* Apache Zookeeper is used to implement the Atomic Broadcast protocol (around Version 1.8) 
* Replication but not sharding!
* replication out from master to slaves at frequent intervals
* All transaction are committed through the master
	* Then (eventually) applied to the slaves
	* Eventuality defined by the update interval or when synchronization is mandated by interaction
* write-master with read-slaves but also supports writing through slaves
* When writing to a slave:
	* Locks coordinated through the master
	* Thansaction data buffered on the slave, applied first on the master to get a txid, then applied with the same txid on the slave

### 九、Scale

因为不支持sharding，所以其实图的大小和吞吐率（主要是写）是比较受限的。当然，引入sharding，对图分割的合理性是一个挑战，处理不好很容易引起剧烈的性能下降。
当然，对于neo4j，之所以不支持sharding，我觉得客观原因有如下两个：

1. 因为它的index-free adjacency实现依赖于存储指针（ID），而这个指针能够快速转换成偏移量很大程度上依赖于本地定长记录，可以很方便快速的计算出记录的偏移量
2. 入口节点/边的索引依赖于lucene，而分布式索引（类似于ES）的构建并不是一个简单的问题

不过即使不sharding，neo4j单机的性能还是可以的：

1、Capacity (graph size)

can support single graphs having tens of billions of nodes, relationships, and properties.

2、Latency (response time)

With a graph database, most queries follow a pattern whereby an index is used simply to find a starting node (or nodes). The remainder of the traversal then uses a combination of pointer chasing and pattern matching to search the data store.

3、Read and write throughput


参考文章
-------




