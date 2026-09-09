---
title: Titan学习笔记
layout: post
catalog: true
---


### 一、图模型——Property Graph

* Vertex
	* Are often used to represent entities
	* Can be tagged with zero or more labels
	* Can contain properties
* Edge:
	* Connect nodes and structure the graph
	* Directed
	* Always have a single name, a start node and an end node ⇒ no dangling relationships
	* Can have properties
	* must be tagged with one label（类似于关系类型，如friend, wife等）
* Properties：key-value pairs (named values):
	* Property Key
		* Property Key Name: name of the property, and Always a string
		* Property Key Data Type: 
		* Property Key Cardinality:
	* Values can be:
		* Numeric values
		* String values
		* Boolean values
		* Lists of any other type of value
* Labels: 
	* Used to assign types to edges
	* Each vertex can has any number of labels (including none)




### 二、查询语言——AQL

Query Language AQL

* Document Queries
* Graph Queries
* Joins
* All can be combined in the same statement

**TIPS** 其他的图查询语言

* SPARQL: 主要针对RDF图模型
* Gremlin: 类似于DSL
* AQL: ArangoDB的查询语言

底层API ：TODO


### 三、图遍历——index-free adjacency

另一种方式是index-based adjacency，对应的开源图数据库有ArangoDB。关于两者的优劣有一个讨论：[Index Free Adjacency or Hybrid Indexes for Graph Databases](https://www.arangodb.com/2016/04/index-free-adjacency-hybrid-indexes-graph-databases/)。

个人认为如果分布式环境下，index-free adjacency要保证邻近节点能够尽量保存在同台机器，减少网络的交互才有意义。另外，index-free adjacency的拥护者在对比性能的时候一直说index-based adjacency需要O(log n)，而它们只需要O(1)，而实际上如果采用Hash索引，完全是可以达到O(1)性能的。事实上，ArangoDB在所有的性能测试中都是靠前的。


### 四、图存储



### 五、图索引

* 入口节点/边索引：集成Lucene用于倒排检索，支持精确、模糊、范围和地理位置查询
* 图遍历索引：依赖于index-free adjacency进行快速的迭代(iterating)和条件过滤(filtering)

With a graph database, most queries follow a pattern whereby an index is used simply to find a starting node (or nodes). The remainder of the traversal then uses a combination of pointer chasing and pattern matching to search the data store.

**TIPS** 

* lecene等倒排索引虽然可以支持精确、模糊、范围和地理位置多种条件查询，但是性能上并不是最优
* 很多开源图存储，如Titan、ArangoDB等会引入[Vertex-Centric Indexes](https://www.datastax.com/dev/blog/a-solution-to-the-supernode-problem)解决图遍历过程中的supernode问题

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




