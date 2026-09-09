---
title: 基于Aerospike实现图数据库
layout: simple-post
catalog: true
category: 技术
tags: [aerospike, Graph Database , 存储]
---


Aerospike介绍
------------

具体参见: [Aerospike学习笔记](http://arganzheng.life/aerospike-notes.html)


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


TODO
----

### 1、封装图模型——Property Graph


#### Property Graph Model

* Vertex
	* 用于表示实体(Entity)
	* 有一个稳定的唯一的id
	* 可以并且一般有一个label，表示实体类型，如Person，Product, etc.
	* Vertex: id -> label*, property*
	* 其中lable决定该实体的schema，如Person, Product, etc. Schema对于在线系统只作为接口文档，存储是schema-less。
* Edge
	* 用于表示关系(Relationship)
	* SPO三元组，附加可选的k-v属性对
	* Edge: (from_id, label, to_id) -> property*
	* 其中label是边的类型，起到labeled edge的作用
	* 关系具有双向性，可以根据这个同时建立反向关系，如：妻子 <=> 丈夫，兄弟 <=> 兄弟
	* 关系可以由 (from_id, label, to_id) 唯一确定，关系id由这个生成
* Property
	* 属性，k-v键值对
	* Property: key -> value
	* key永远是string类型，value可以是：
		* Numeric
		* String
		* Boolean
		* Lists of any other type of value
		* 自定义类型(Json中的Map)
* Label
	* 用于对节点和边进行归类


**说明** 

* vertex label: neo4j的node可以有0个或者多个lable；titan则是0个或者1个，如果没有指定，会使用默认的label。这里采用的是titan的方式。
* 属性理论上来说是应该有一些附加元信息的，如置信度，作者，来源等；Titan的Property of Property能够很好的支持这个功能；但是考虑到复杂性，这里先不考虑。
* 这里不强制要求weighted edge，如果要表示关系的强弱程度(Weight of Edge)，可以在边的properties中增加weight属性，可以用于关系检索时候的排序和截断。
* 关系也可能有别名，如: 妻子==老婆==夫人，由上层进行归一或者业务存储一个alias的property实现

#### 对应到Aerospike的关系

* Graph => namespace
* Vertex => record
* Edge => record
* Property => bin
* Label => set_name

**说明**


* 出于平台化的考虑，每个Graph的顶点和边互不影响，这可以通过Aerospike的namespace实现。
* ~~如果所有垂类和关系都放在一个namespace下，跟Aerospike的namespace对应关系是最直接，管理和跨垂类查询方便，但是Aerospike的namespace对二级索引的限制256个，对于比较大的Graph就不合适了。~~
* ~~综合考虑可以做个简单映射关系，顶点和边各自一个namespace，名字为 {GraphName}_vertex/edge：GraphName => VertextNamespace & EdgeNamespace。~~ 经过讨论，决定后面修改Aerospike源码，让一个namepsace可以支持更多的二级索引。所以现在是一个namesapce对应一个graph。
* Aerospike没有提供listNamespaces的接口，需要一个地方存储所有已经创建的Graph和创建他们的用户和Graph的访问权限（存储在Aerospike就可以）。
* 顶点和边的label作为他们的setName


### 2、存储和索引

存储和索引看起来是两个问题，但是悲剧的是他们是互相影响的。比如说我们是可以把整个json串作为一个value保存起来，但是无法对里面的某些属性进行检索。要考虑检索，需要对需要建立索引的字段单独存储。而且，存储和索引还往往带来另一个副产品——Schema，因为不同的存储方式可能需要schema信息（比如拆字段存储），另外索引也需要字段和类型信息。

**方案**

+ Graph
	+ Vertex
		- data
		- idx_name
		- idx_tags
		- ...
	+ Edge
		- data
		- idx_from
		- idx_to
		- idx_type
		- ...

**说明**

* 实体/关系数据作为一个document(JSON，或者直接使用Aerospike的Map数据类型)存储在"data"字段
	* schema-less
* 索引字段根据需要动态创建
	* 有类型信息
	* bin名称以 "_idx" 开头，如 "_idx.name", "_idx.address.city" 
	* 先扫描数据创建索引字段，再通过对这些索引字段创建二级索引
	* 新插入数据需要知道有哪些索引字段，以便对该字段赋值(setProperty) // 这里需要一个meta表（或者Aerospike有这个信息）
* 索引可以直接通过AQL创建，默认可以支持：
	* 精确查询
	* 范围查询
	* Geo查询
* 后面再根据需要增加全文索引（fulltext index）
* 因为Aerospike不支持逻辑操作（AND，OR，NOT），需要扩展支持
* 图的遍历方式采用的是index-based adjacent方式，而不是index-free adjacency，只要索引方式使用得当，性能并不比后者差 [Index Free Adjacency or Hybrid Indexes for Graph Databases](https://www.arangodb.com/2016/04/index-free-adjacency-hybrid-indexes-graph-databases/)


**思考** 如何存储和索引JSON数据

我们离线抓取得到的数据是json格式，如果仅仅是为了存储数据，那么把整个json作为一个value存储就可以了（类似于document-oriented stores的做法）。这样的问题是除了key之外，其他的属性都没法进行检索。所以综合考虑，有如下三种为了能够对属性进行检索，有下面两种方式：

1. Store as serialized string in a single bin - easy to store, can't query by anything other than key
2. Convert JSON doc into series of bins with nested maps in the Aerospike record - complicated saving logic that you have to maintain, can use secondary indexes to query
3. Hybrid - Store as serialized string in single bin and store any important values as separate bins with secondary indexes to get query ability over certain fields while still keeping overall logic simple.

说白了，分字段存储更多是为了建索引以便检索，而不是为了数据存储或者数据模型。基本上所有的非简单KV的存储引擎都有这个问题。

在Aerospike论坛上有相关的讨论：[The best way to handle JSON object in Aerospike](https://discuss.aerospike.com/t/the-best-way-to-handle-json-object-in-aerospike/1156/2)。

这里我们采用的是第三种方式——Hybrid。


### 3、接口

#### 1、Graph接口

1、创建Graph（因为Aerospike现在不支持动态创建namespace，先不支持 [TODO]）

	Graph g = new Graph(graphName: String) throws GraphExistException;

2、加载已经存在的Graph

	Graph g = Graph.load(graphName: String) throws GraphNotExistException;

需要根据graphName加载graph相关配置，如该graph有哪些索引，需要根据这个在保存数据的时候设置相应的索引字段。

3、删除Graph（因为Aerospike现在不支持动态删除namespace，先不支持 [TODO]）

	Graph.drop(graphName: String) throws GraphNotExistException;

4、清空数据

	graph.dropData();

调用Aerospike的truncate接口：

	public final void truncate(InfoPolicy policy,
	            String ns,
	            String set,
	            Calendar beforeLastUpdate)
	                    throws AerospikeException	

**说明**

1. Graph暂时只有一个GraphName，后面应该是有一个配置信息的。比如用户名、token、persistence、replication-factor等。
2. 因为Aerospike暂时(<=3.12版本)不支持动态创建和删除namespace，所以我们这里也先不支持动态创建和删除Graph [TODO]


#### 2、索引接口

1、创建索引
	
	graph.createIndex(	indexName: String,
						label: String, // setName
						property: String, // binName
						propertyType: String // binType, NUMERIC or STRING or GEO2DSPHERE
					); 

调用Aerospike的createIndex接口：

	public final IndexTask createIndex(Policy policy,
	                    String namespace,
	                    String setName,
	                    String indexName,
	                    String binName,
	                    IndexType indexType)
	                            throws AerospikeException

需要更新graph中的indexInfo属性。

2、删除索引: 
	
	graph.dropIndex(indexName: String,
					label: String, 
					indexName: String
				);

调用Aerospike的dropIndex接口：

	public final void dropIndex(Policy policy,
	             String namespace,
	             String setName,
	             String indexName)
	                     throws AerospikeException

需要更新graph中的indexInfo属性。

3、重建索引（会对数据进行一次扫描以建立索引）:

	graph.reIndex(  indexName: String,
					label: String, // setName
					property: String, // binName
					propertyType: String // binType
				); 

1. 先创建索引
2. 调用Aerospike的scanAll接口对原来数据进行扫描，对索引字段进行赋值

#### 3、实体接口

1、添加实体:

	graph.addVertex(vertex: Vertex);

	graph.addVertex(vertexId: String, 
					label: String,
					properties: List<Property>);

	graph.addVertex(data: Document); // json document

调用Aerospike的put接口：

	public final void put(WritePolicy policy,
	       Key key,
	       Bin... bins)
	               throws AerospikeException

2、更新实体（允许局部更新）：

	graph.updateVertex(vertexId: String,
						label: String, // 如果每个label一个set的话，那么必须提供setName [TODO]
						properties: List<Property>);

	graph.updateVertex(vertexId: String,
						label: String, // 如果每个label一个set的话，那么必须提供setName [TODO]
						data: Document);

调用Aerospike的put方法：

	public final void put(WritePolicy policy,
	       Key key,
	       Bin... bins)
	               throws AerospikeException

3、删除实体：

	graph.deleteVertex(vertexId: String,
						label: String, // 如果每个label一个set的话，那么必须提供setName [TODO]
					);

调用Aerospike的delete接口：

	public final boolean delete(WritePolicy policy,
	             Key key)
	                     throws AerospikeException

#### 4、关系接口

1、添加关系：

	graph.addEdge(edge: Edge);

	graph.addEdge(edgeId: String,
					from: String, 
					to: String,
					label: String,
					bidirection? // 是否双向关系
					properties: List<Property>);

	graph.addEdge(edgeId: String, 
					from: String, 
					to: String,
					label: String,
					bidirection? // 是否双向关系
					data: Document);

调用Aerospike的put接口：

	public final void put(WritePolicy policy,
	       Key key,
	       Bin... bins)
	               throws AerospikeException

2、更新关系：

	graph.updateEdge(edgeId: String,
						label: String, // 如果每个label一个set的话，那么必须提供setName [TODO]
						properties: List<Property>);


	graph.updateEdge(edgeId: String,
						label: String, // 如果每个label一个set的话，那么必须提供setName [TODO]
						data: Document);

调用Aerospike的put接口：

	public final void put(WritePolicy policy,
	       Key key,
	       Bin... bins)
	               throws AerospikeException

3、删除关系: 

	graph.deleteEdge(edgeId: String,
						label: String, // 如果每个label一个set的话，那么必须提供setName [TODO]
					);

调用Aerospike的delete接口：

	public final boolean delete(WritePolicy policy,
	             Key key)
	                     throws AerospikeException

### 类结构定义

* Graph
	- name
	- indexInfo: // 索引信息，初始信息来自于show indexes，需要根据这个设置索引字段，后面根据操作变更。
		- label: setName
		- indexName
		- property
		- type
		- state
	- stat: // 统计信息，初始信息来自于show sets，后面根据操作变更。
		- labelNum
		- vertexNum
		- edgeNum
	+ create(graphName)
	+ load(graphName)
	+ drop(graphName)
	+ createIndex()
	+ dropIndex()
	+ reIndex()
	+ getIndexes
	+ addVertex(vertex)
	+ addVertex(vid, label, properties)
	+ updateVertex()
	+ getVertex(vid)
	+ deleteVertex(vid)
	+ queryVertex(query)
	+ addEdge(edge)
	+ addEdge(eid, from, to, label, bidirection?, properties)
	+ updateEdge()
	+ deleteEdge()
	+ getEdge(eid): Edge
	+ queryEdge(query)
* Vertex
	+ setId(vid)
	+ setLabel(label)
	+ addProperty(key, value)
	+ getProperty(key): Object
* Edge
	+ Edge(eid, from, to, label)
	+ Edge(from, to, label)
	+ addProperty(key, value)
	+ getProperty(key): Object
* Document: JSON & Map 的封装；图检索统一返回的数据结构（顶点或者边）
	+ fromJson(Json: String)
	+ toJson(): String
	+ addProperty(key, value)
	+ getProperty(key): Object


### 4、图遍历

#### 查询语言选型：提供Gremlin图检索查询语言。

#### 实现

方案一、基于tinkerpop-gremlin实现

**Prons**

* 基于语言自身特性实现的DSL，相对于parser方式实现简单一些
* 直接与host language结合，其实就是本地调用，而不是发送一个语句到服务端解析
* 语法解析工作放在客户端，减轻服务端压力

**Cons**

* 性能相对于parser方式差一些
* 各种语言的语法差异导致实现起来有些语法差异（比如java和groovy）
* 对语言有要求，不是所有的语言都可以实现DSL [JVM Language Implementations](https://github.com/tinkerpop/gremlin/wiki/JVM-Language-Implementations)

官方提供的实现基本都是JVM实现：

* Java
* Groovy

方案二、基于Parser实现

**Prons**

* 性能比较好
* 有开源的parser，关键定义好语法和AST

**Cons**

* 编译原理的东东，实现有一定的难度

业界实现方案：

1. Flex + Bison: ArangoDB
2. ANTLR4: Aerospike
3. Boost::Spirit: WD-KG


