---
title: Aerospike UDF学习笔记
layout: post
catalog: true
tags: [aerospike, UDF]
category: 技术
---

### 背景

很多数据存储系统都支持用户自定义函数(UDF, User-Defined Functions)，Aerospike也不例外；UDF language一般是解释型的脚本语言，比如Lua(Redis，Postgres, etc.)，JavaScript(ArangoDB, Cayley, etc.)，Aerospike这里采用了Lua。



### UDF(User-Defined Functions)

* code written by a user(or developer) that runs inside the Aerospike database server
* currently only supports Lua as the UDF language


#### 支持两种类型的UDF

1、Record UDFs

* operates on a single record. 
* They can create, update, or delete a record.

2、Stream UDFs

* perform read-only operations on a stream of records, and it can comprise multiple stream operators to perform very complex queries.（类似于shell的管道）
* Use Aerospike Stream UDFs to `filter`, `transform`, and `aggregate` query results in a distributed fashion.
* [Aerospike Stream UDFs support these operators](http://www.aerospike.com/docs/udf/api/stream.html)。提供了stream api接口，构建类似于shell的管道操作:
	* filter: Filter data in the stream that satisfies a predicate.
		* `filter(p: (a: Value) -> Boolean) -> Stream`
		* `function<A>(a: A): Boolean`
	* map: Transform a piece of data.
		* `map(f: (a: Value) -> Value) -> Stream`
		* `function<A,B>(a: A): B`
		* The type of the return value must be one of those supported by the database: integer, string, list, and map.
	* aggregate: Reduce partitions of data to a single value.
		* `aggregate(x: Value, op: (a: Value, b: Value) -> Value) -> Stream`
		* `function<B,A>(b: B, a: A): B`
		* The aggregate value and the return value should be of same type. 
		* Should return a single value whose type must be one of those supported by the database: integer, string, list, and map.
	* reduce: Allow parallel processing of each group of output data.
		* `reduce(op: (a: Value, b: Value) -> Value) -> Stream`
		* `function<A>(a1: A, a2: A): A`
		* 参数和返回值必须都是一个类型（The two arguments of the reduce function and the return value should be of the same type.）
		* The type of the return value must be one of those supported by the database: integer, string, list, and map.
		* Executes both on the server nodes as well as the client side (in application instance)
* 类似于Map-Reduce
	* Map(filter, map, aggregate): run on all nodes
	* Reduce(final aggregation): run on the requesting client


**NOTES & TIPS**

1、细心的读者可能注意到，aggregate和reduce其实非常相似，就是把两个值合并成一个。所以最后一个动作也称之为`reduce or final aggregation`。

> Instead of coordinated job control, Stream UDF queries are sent to all cluster nodes by a requesting client. They are managed and prioritized by each server. Results return to the requesting client, which performs final operations (such as reduce or final aggregation) before returning the results to the client.


2、Stream UDF的执行步骤如下：

1、Run a Query on a Secondary Index:

	statement.setFilters(filter);

2、如果有PredExp，也一并执行:

    statement.setPredExp(predexps);

3、Apply stream UDF on results of a secondary index query:

	statement.setAggregateFunction(ClassLoader resourceLoader, String resourcePath, String packageName, String functionName, Value... functionArgs);

3、Stream UDFs可以而且经常串联起来作为一个pipeline（其实更类似于java8的Streaming API）：

```lua
function my_stream_udf(s)
    local m = map()
    return s : filter(my_filter_fn) : map(my_map_fn): aggregate(m, my_aggregate_fn): reduce(my_reduce_fn)
end

function my_complex_stream_udf(s)
    return s : filter(my_filter1) : map(my_map1) : filter(my_filter2) : map(my_map2) : map(my_map3) : reduce(my_reduce)
end
```

![query_stream_filter.png](/img/in-post/query_stream_filter.png)

4、Stream UDFs执行分为两个周期：

* Cluster-side : First the stream UDF is executed on all the nodes of the cluster. The result from each node (after applying first reduce) is sent to the application which triggered the UDF.
* Client-side: When the nodes send their result to the application, the client layer will do the final aggregation. Execution at client-side will start from first reduce function and send the result to application.

所以：The UDF file should be present both on the cluster nodes as well as the client nodes as the final phase of reduction happens on the client side.

> One main characteristic of reduce function is that it executes both on the server nodes as well as the client side (in application instance). Each node first runs the data stream through the functions defined in the stream definition. The end result of this is sent to the application instance. The application gets results from all the nodes in the cluster. The client layer in it does the final reduce using the reduce function specified in the stream. So, the reduce function should be able to aggregate the intermediate aggregated values (coming form the cluster nodes). If there is no reduce function, the client layer simply passes all the data coming from the nodes to the application.


5、例子

1. [Lua UDF – Word Count](http://www.aerospike.com/docs/udf/examples/stream_udf_word_count.html)
2. [Query with Multiple Filters](http://www.aerospike.com/launchpad/query_multiple_filters.html)

#### 如何调用UDF

To invoke UDFs in your application:

1. Specify the UDF file name.
2. Specify the function name within the file
3. (optional) Specify one or more parameters.


### UDF最佳实践

#### 1、Disable Caching in dev enviroment:


	mod-lua {
	  cache-enabled false
	}


#### 2、Check for Unintended Variable Definition


	luac -p -l yourModule.lua  | grep [gs]etGlobal

#### 3、Logging

You can set the levels in the logging block of Aerospike's configuration file:

	logging {
	  file PATH {
	    context any warning
	    context ldt info
	    context udf debug
	    context query detail
	  }
	}

In the Lua, you can utilize the following log functions:

	trace() – log a DETAIL message
	debug() – log a DEBUG message
	info() – log an INFO message
	warn() – log a WARNING message

**TIPS**

1、When the format string arguments are not strings or integers, then you need to call tostring() on the variable:

```lua
local l = list{1,2,3,4}
info("We have %s", tostring(l))
```

2、If your Lua file has an error, it will also be emitted to the log.


#### 4、Use Lua Table for Temporary Variables

If variables do not need to be read or stored from Aerospike, which would require data type such as map() or list(), it is best to have a Lua Table object instead.


### Lua UDF – API Reference

API Reference for Aerospike extensions to Lua, including functions, modules and types.


#### Types

Aerospike provides a variety of Lua types which coincide with the types supported by the database.

* [bytes](http://www.aerospike.com/docs/udf/api/bytes.html) - The bytes type provides the ability to build a byte array using bytes and integers. This type coincides with BLOB type in the database.
* [list](http://www.aerospike.com/docs/udf/api/list.html) - A list is data structure that represents a sequence of values.
* [map](http://www.aerospike.com/docs/udf/api/map.html) — A collection of (key, value) pairs, in which a key can only appear once in the collection.
* [record](http://www.aerospike.com/docs/udf/api/record.html) — Represents database records, including bins – (name, value) pairs – and metadata.
* [stream](http://www.aerospike.com/docs/udf/api/stream.html) - Represents streams of records.

#### Modules

The following are modules which provide added functionality.

* [aerospike](http://www.aerospike.com/docs/udf/api/aerospike.html) — The aerospike object is a global object that exposes database operations.
	* `function aerospike:create(r: Record): Integer`
	* `function aerospike:update(r: Record): Integer`
	* `function aerospike:exists(r: Record): Boolean`
	* `function aerospike:remove(r: Record): Integer`
* [logging](http://www.aerospike.com/docs/udf/api/logging.html) — Logging functions that send log messages to the Aerospike Server's logs.
	* `function trace(msg: String , …): nil`
	* `function debug(msg: String, …): nil`
	* `function info(msg: String, …): nil`
	* `function warn(msg: String, …): nil`



### 参考文章

1. [Architecture - User-Defined Functions](http://www.aerospike.com/docs/architecture/udf.html)
2. [Feature Guides - User-Defined Functions](http://www.aerospike.com/docs/guide/udf.html)
3. [User-Defined Functions (UDF) Development Guide](http://www.aerospike.com/docs/udf/udf_guide.html)
4. [Lua UDF – Best Practices](http://www.aerospike.com/docs/udf/best_practices.html)
5. [Developing Stream UDFs](http://www.aerospike.com/docs/udf/developing_stream_udfs.html)
