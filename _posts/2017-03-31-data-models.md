---
title: 数据模型和存储系统
layout: post
catalog: true
---


**数据模型维度**

* Key-Value Stores
    * Redis
    * Riak(Bitcask)
    * LevelDB
    * RocksDB
    * LMDB
    * Aerospike（个人感觉这个更像是RDB）
    * HyperLevelDB
    * HyperDex: (Sorted)
* Column-family Stores
	* HBase
	* Cassandra
* Document-oriented Databases
    * MongoDB
    * ElasticSearch
* Tabular databases(RDB)
    * MySQL
    * Aerospike
    * DynamoDB
* Graph Databases
	* Neo4j
	* Graph Engine (Microsoft)
	* [Titan](titan.thinkaurelius.com) => [JanusGraph](http://janusgraph.org/)
	* Cayley
	* ArangoDB
	* OrientDB
* Time Series Database(TSDB)
	* [OpenTSDB](http://opentsdb.net/)
	* [KairosDB](http://kairosdb.github.io/)
	* [InfluxDB](https://github.com/influxdata/influxdb)
* Object-Oriented Databases
	* ObjectivityDB
* Multi-Model Databases
	* ArangoDB
	* OrientDB

事实上，KV是所有模型的基础，其他的数据模型基本都是基于KV实现。

**其他维度**

* 底层数据结构
	* LSM Tree
	* B+ Tree
	* Hash Tree
* 存储介质
	* 内存
	* 磁盘
* 部署方式
	* 嵌入式
	* 单机
	* 分布式
* 事务
	* CAP
	* ACID
* 分布式方式
	* master-slave
	* peer-to-peer


### Why NoSQL?

1. 丰富的数据模型
	* K-V
	* Document
	* Table
	* Graph
	* Object
	* Time Series
2. schema-less
	* 不需要提前定义
	* 不需要严格一致（宽表/稀疏）
3. 分布式
	* auto sharding and replication
	* Scale out
	* HA
4. 高性能
	* 索引结构
		* LSM 
		* Hash
	* 放弃强一致性



