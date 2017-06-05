---
title: Aerospike学习笔记
layout: post
catalog: true
tags: [aerospike]
category: 技术
---


### 架构

分为三层：

* Client Layer
* Distribution Layer
	* Cluster Management Module
	* Data Migration Module
	* Transaction Processing Module
		* Sync/Async Replication
		* Proxy
		* Duplicate Resolution
* Data Storage Layer
	* enhanced key-value store with a schemaless data model
	* namespaces => 类似于databases，
	* sets => tables
	* records => rows
	* bins => columns


### Data Model

#### Namespaces

Namespaces

* 类似于database，但是并不一定是一对一的关系。
* The way you collect data in namespaces relates to how the data is stored and managed.
* 包含如下内容：
	* records
	* indexes
	* policies（[Configuration](http://www.aerospike.com/docs/operations/configure/namespace)）
		* How data is stored: on DRAM or disk.
		* How many replicas exist for a record.
		* When records expire

#### Sets

类似于RDBMS的table。

记录可以不归属于某个Set，当时一定会归属于某个namespace。Set specification is optional. Some records in the namespace may not be within a set.

Sets inherit the policies defined by their namespace, and can define additional policies or operations specific to the set. For example, secondary indexes can be specified only on data for a particular set, or a scan operation can be done on a specific set.


#### Records

* Records can belong to a namespace or to a set within the namespace.
* 记录包含如下三个部分：
	* key
		* Unique identifier. Records are addressable using a hash of its key, called the digest.
		* key的类型：Integers, String, and bytes
		* 内部会hash成一个 160-bit(20-bytes) 的digest
	* metadata
		* generation
		* time-to-live (TTL)
		* last-update-time (LUT)
	* bins
		* Bins are equivalent to fields in RDBMS
		* Data type is defined by the value contained in the bin：
			* Basic
				* integer: 8 bytes
				* string: 128 KB; 
				* bytes
				* double: 8 bytes
			* CDTs (Complex Data Types)
				* list
				* map
				* GeoJSON (3.7.0+)
			* native-languages serialized(blobs)

**NOTES & TIPS**

* namespaces需要提前定义，但是sets和bins是可以动态创建的。
* 不像RDBMS，不同记录的同一个bin里面可以有不同类型的value。比如有个price的bin，里面可以同时存储“20000.00”，也可以存储“2万”。但是最好不要这样子，会影响索引。
* 为了性能，索引(primary Keys和Secondary keys)是存储在内存中的。数据(values)可以存储在内存，也可以存储在SSD。
* 不管是存储在哪里，Aerospike 通过 Smart Defragmenter 和 Intelligent Evictor 这两个机制保证数据(values)不会丢失。
* 因为是内存索引，系统启动的时候需要根据数据重新构建，所以启动会比较耗时。
* 不同的namespaces的配置是独立和隔离的。
* Aerospike虽然不是使用LSM Tree，但是为了解决随机写的问题，同样引入了日志结构文件系统（Aerospike log structured file system）。
* Maps and lists 支持任意层次的嵌套，但是索引只能指定第一层。
* list和map底层存储是以 [MessagePack](http://msgpack.org/index.html) 方式序列化的
* There is a limit of 32K unique bin names in use within a namespace
* the record size cannot exceed the write-block-size (usually 128KB for SSDs and 1MB for rotational) => 之前版本支持LDTs(Large Data Types)，不过后面的版本移除了。


### 查询和索引

#### Query

Aerospike通过secondary index，可以支持下面三种条件查询：

* Equal query against string or numeric indexes
* Range query against numeric indexes
* [Point-In-Region or Region-Contain-Point](http://www.aerospike.com/docs/guide/geospatial.html#geospatial-index) query against geo indexes

**TIPS** 

查询到的数据可以通过 Aaerospike Predicate Filtering (3.12+) 或者 Aerospike UDFs (user-defined functions) 进行post-processed。

#### Primary Index

 In Aerospike, the primary key index is a blend of distributed hash table technology with a distributed tree structure in each server.

 The entire keyspace in the namespace (database) is partitioned using a robust hash function into partitions. 

 A total of 4096 partitions are equally distributed across cluster nodes. See [data-distribution](http://www.aerospike.com/docs/architecture/data-distribution.html) for details on hashing and partitioning.

Primary Index是纯内存索引，也没有定时持久化会磁盘。系统重启时候会扫描数据重建索引。企业版通过linux的共享内存区域(Linux shared memory segment)支持快速启动(Fast Restart Feature)。


#### Secondary Index

* are stored in RAM for fast look-up.
* are built on every node in cluster and co-located with the primary index. Each secondary index entry only contains references to records local to the node.
* contain pointers to both master records and replicated records in a cluster.

除了主键之外的字段(bin)都是二级索引，二级索引的key的数据类型只能是如下三种：

* Integer
* String
* Geospatial

但是key所在的字段(bin)数据类型可以是：

* Basic
* List
* MapKeys
* MapValues


**Limitations**

* supports up to 256 secondary indexes per namespace => :(
* There is a limit of 32K unique bin names in use within a namespace => 其实还好
* Fast restart is not supported. On daemon restart, secondary indexes are rebuilt based on record data => :(
* Aerospike is tuned for queries using high selectivity secondary indexes 
* For string data-type, only string size <= 2k can be indexed => 精确匹配问题不大
* RANGE result sets are inclusive (that is, both specified values are included in the results) => 这个应用层过滤一下
* If no set-name is specified during index creation, then the index will only include records without a set name, but not all sets in the namespace.


### Distribution（分布式）

**Features**

* Automatic data location detection
* Automatic cluster balancing
* No single point of failure

**机制**

* Data Distribution: Robust partitioning ensures uniform data distribution, which avoids hot spots and automatically balances data without manual intervention.
* Clustering: The Aerospike clustered database automatically detects failures and heals.
* Replication: This Aerospike feature includes the following replication abilities to avoid a single point of failure:
	* Intra Cluster Replication
	* Rack Aware Replication
	* Cross-Datacenter Replication

### Transaction（事务）

* Single row ACID

### UDF(User-Defined Functions)

* code written by a user that runs inside the Aerospike database server
* currently only supports Lua as the UDF language
* Record UDFs: execute on a single database record. They can create, update, or delete a record.
* Stream UDFs: perform read-only operations on a collection of records. 


### Client(API & SDK)

Aerospike的服务端分布式架构是完全对等的 Shared-Nothing 架构，没有master，也没有metadata server。这就把一些功能下推到客户端了。

client，也就是drivers，主要处理这些事情：

* cluster-status sensing
* efficient transaction routing 
* network connection pooling
* failover protection

API

* put()
* get()
* delete()
* CAS (safe read-modify-write) operations.
* In-database counters.
* Batch get() operations（不支持Batch write()）
* Scan operations.
* List and Map element operations:
	* List
		* append(), insert(), insert_items()
		* get(), get_range(), get_range_from()
		* set()
		* pop(), pop_range()
		* remove(), remove_range()
		* trim()
		* clear()
		* size()
	* Map 
		* set_type()
		* add(), add_items(), increment(), decrement(), clear()
		* remove_by_key(), remove_by_index(), remove_by_rank()
		* remove_by_key_interval(), remove_by_index_range()
		* remove_by_value_interval(), remove_by_rank_range(), remove_all_by_value()
		* size()
		* get_by_key(), get_by_index(), get_by_rank()
		* get_by_key_interval(), get_by_index_range()
		* get_by_value_interval(), get_by_rank_range(), get_all_by_value()
* Queries: Bin values are indexed and the database searched by equality or range.
* UDFs extend database processing by executing application code in Aerospike.
* Aggregation: Use UDFs on a collection of records to return aggregate values.

各个语言封装的API使用方式略有不同，具体参见文档 DEVELOPMENT-Client Libraries，如 [Java Client](http://www.aerospike.com/docs/client/java)。


Prons & Cons
------------

**Prons**

* Fast
* 分布式
* 事务
* 有比较丰富的数据类型（Intege, String, Double, List, Map, GeoJSON..）和相应的操作（Increment, Append..，操作没有redis丰富）
* 有一定的索引支持（一级索引，二级索引，Equality and Range filters)
* 有命名空间
* 支持UDF
* 支持对某个namespace或者set的全量Scan，结合 UDF 
* 支持对某个namespace或者set的Truncation
* 有一定的权限管理
* 有比较丰富的客户端SDK和比较完善的文档
* 有命令行 [aql](http://www.aerospike.com/docs/tools/aql) 和管理工具 [asinfo](http://www.aerospike.com/docs/tools/asinfo)
* 比较活跃，有专门的团队支持


**Cons**

* Index纯内存，成本比较大，重启需要根据数据重新构建索引，启动时间比较长（企业版支持Fast Restart）=> :(
* map索引只支持第一层级属性，而且索引粒度是key或者vallue（而不是一般的某个key对应的value）=> :(
* list索引只支持第一层级属性 => :(
* 采用的是随机sharding，不利于图切割 
* 采用B+ Tree，基于Index-based adjacency 方式遍历需要 klog(n)
* namespace limitations:
	* 1024 sets
	* 256 secondary indexes => :(
	* 32K unique bin names
	* 4 billion of objects per namespace per server（3.12 扩大为 32 billion, 但是仅限于企业版） 
* 不支持动态创建namespace，只能通过修改配置文件、重启服务器（Aerospike计划在下一个release中支持） => :(
* 记录大小有限制: <= 1M => 有点小，不过对于我们的场景基本没问题 
* bin name长度: <= 14 Chars => 一般来说单字段不会超过，嵌套属性如果拼接就很容易超长 :(
* 基于Secondary Index的Query不支持逻辑操作（AND，OR，NOT），只支持单属性查询 => :(
* 3.12引入了[Predicate Filter](http://www.aerospike.com/docs/guide/predicate.html)，可以对 Scan得到的记录或者索引查询结果(scan and secondary index query)进行多条件过滤。不能用于聚合运算(Aggregations)
* 范围查询只支持BETWEEN语句，没有小于，大于查询，并且RANGE结果只支持inclusive 
* Query不支持分页(no cursor or pagination..) => :(
* Query不支持排序(no order by..) => :(
* 没有内建的聚合函数(Aggregations: count, max, min, sum, group by, etc.)，通过UDFs可以支持（queryAggregate），但是使用方式不友好，效率也不高。
* 只支持batch read，不支持batch writes.. => :(
* 如果where条件没有相应的索引就会报错，而不是走全表扫描
* 如果没有指定set name，不是对整个namespace进行检索，而是对没有指定set name的数据进行检索。 => :(
