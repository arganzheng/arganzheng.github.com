---
title: DynamoDB学习笔记
layout: post
---


### 数据模型


不同于Cassandra的Column family数据模型，DynamoDB的数据模型非常类似于传统的关系型数据：

* Tables: 类似于数据库中的table
* Items: 记录，类似于数据库的row
* Attributes: 属性，类似于数据库的column

**说明**

1、跟传统的RDB不同，DynamoDB是schema-less的，不需要预定义表结构，同一个表中的记录结构也不需要完全一致（但是必须包含主键属性）。

> Except for the required primary key, a DynamoDB table is schemaless, which means that neither the attributes nor their data types need to be defined beforehand. Each item can have its own distinct attributes.

2、另一个跟传统RDB不同的是，Attribute类型可以支持数组(list)和嵌套属性(map)。

3、必须指定Primary keys，主键值只能是简单类型。DynamoDB支持两种类型的主键：

1. Partition Key: 简单主键，包含一个属性，也称为 "Hash attribute"
2. Partition Key and Sort Key: 组合主键，包含两个属性，一个是partition key, 另一个是sort key。这种方式下允许一个partition key对应多条记录。也叫做 "Range attribute"

> The partition key of an item is also known as its hash attribute. The term hash attribute derives from DynamoDB's usage of an internal hash function to evenly distribute data items across partitions, based on their partition key values.
The sort key of an item is also known as its range attribute. The term range attribute derives from the way DynamoDB stores items with the same partition key physically close together, in sorted order by the sort key value.


### Query and Scan

#### 1、通过主键检索(Query against Primary Key)：

You can use the query method to retrieve data from a table. You must specify a partition key value; the sort key is optional.


#### 2、通过二级索引检索(Query or Scan against Secondary Index)：

DynamoDB的二级索引是通过将base table的索引字段copy到另一张新表作为其主键（base table的主键也copy过去作为其value）来实现的，这个表的名称就是索引的名称。这正是我们在HBase上经常做的事情，只是这个过程现在交给底层存储做了。

> When you create an index, you specify which attributes will be copied, or projected, from the base table to the index. At a minimum, DynamoDB will project the key attributes from the base table into the index. 

同样，二级索引分为如下两种类型：

1. Global secondary index: an index with a partition key and sort key that can be different from those on the table.
2. Local secondary index: an index that has the same partition key as the table, but a different sort key.

具体参见: [Improving Data Access with Secondary Indexes](http://docs.aws.amazon.com/amazondynamodb/latest/developerguide/SecondaryIndexes.html)

最后DynamoDB支持对table进行Scan操作，对全表数据进行扫描和过滤。因为Secondary Indexes本质上就是一个特殊的table，所以也能对其进行Scan。

因为二级索引是另一张表的主键，所以只能支持普通类型的属性（不支持对嵌套属性进行索引）。


### 接口


1. putItem：插入或更新一条记录，支持条件更新，支持在更新时返回属性旧值
2. getItem：获取一条完整的记录或某些属性，允许指定用最终一致性读还是严格一致性读
3. batchGetItem：获取一个或多个表中的多条记录或某些属性，只能用最终一致性读。一次最多返回100个属性及小于1MB数据，如果没有返回所有记录，会返回还没有处理的键值以便应用再次去获取
4. updateItem：插入/删除/更新一条记录中的某些属性，支持条件更新，支持更新时返回所有属性旧/新值、被更新属性旧/新值
5. deleteItem：删除一条记录，支持条件删除，支持删除时返回被删除记录
6. query：使用组合主键时查询同一Hash Key的多条记录或某些属性，可指定Range Key范围条件及读一致性要求，可指定返回条数限制。操作保证按主键顺序返回记录，因此可通过在下一条查询时指定上次返回的最大主键作为起始点来实现分页
7. scan：表扫描，可指定多个过滤条件，可指定返回条数限制。实现分页的方法同query

可以看到DynamoDB不但提供了单记录的CRUD操作，还提供了条件更新、多记录读、范围扫描、全表扫描等功能，还算比较灵活。

此外，还可以用MapReduce来分析DynamoDB中的数据。特别的，因为DynamoDB已经是表结构，可以很方便的用Hive来分析。


limitation
----------

1. there is a limit of 400 KB on the item size.
2. You can define up to 5 global secondary indexes and 5 local secondary indexes per table.


参考文章
-------

1. [DynamoDB Core Components](http://docs.aws.amazon.com/amazondynamodb/latest/developerguide/HowItWorks.CoreComponents.html)
2. [Step 4: Query and Scan the Data](http://docs.aws.amazon.com/amazondynamodb/latest/gettingstartedguide/GettingStarted.Js.04.html)
3. [Working with Queries](http://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Query.html)

