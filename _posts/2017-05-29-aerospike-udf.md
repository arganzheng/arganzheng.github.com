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

**TIPS & NOTES**

1、When the format string arguments are not strings or integers, then you need to call tostring() on the variable:

```lua
local l = list{1,2,3,4}
info("We have %s", tostring(l))
```

2、If your Lua file has an error, it will also be emitted to the log.


#### 4、Use Lua Table for Temporary Variables

If variables do not need to be read or stored from Aerospike, which would require data type such as map() or list(), it is best to have a Lua Table object instead.

#### 4、Enable Client Logging

我们前面说过，reduce函数会同时在客户端和服务端执行，而事实上，经常是只在客户端执行：[How do I debug the ‘reduce()’ operation of a stream UDF?](https://discuss.aerospike.com/t/how-do-i-debug-the-reduce-operation-of-a-stream-udf/2250/3)。如果想调试一下redece过程，那么需要配置一下客户端日志，让其能够打印出日志。

具体实现是通过callback的方式实现的，以java client为例：[Java Client - Logging](http://www.aerospike.com/docs/client/java/usage/logging.html)。

具体代码如下所示：

首先创建一个Log.Callback实现类：

```java
package life.arganzheng.study.aerospike;

import java.util.Date;

import com.aerospike.client.Log;

public class MyLogCallback implements Log.Callback {

    public MyLogCallback() {
        Log.setLevel(Log.Level.INFO);
        Log.setCallback(this);
    }

    @Override
    public void log(Log.Level level, String message) {
        Date date = new Date();
        System.out.println(date.toString() + ' ' + level + ' ' + message);
    }
}
```

这里简单的将日志打印到终端。
然后在某个地方注册一下这个callback:

```java
public class MyAerospikeClient implements Closeable {
 	
 	...

    private void initAerospikeClient() {
        // init Log Callback for Aerospike client logging
        // @see http://www.aerospike.com/docs/client/java/usage/logging.html
        Log.Callback mycallback = new MyLogCallback();
        Log.setCallback(mycallback);
        Log.setLevel(Log.Level.DEBUG);

   
        ClientPolicy policy = new ClientPolicy();
        policy.timeout = Constants.DEFAULT_TIMEOUT_MS;

        List<Host> hosts = new ArrayList<Host>();
        hosts.add(new Host("xxxxx", Constants.AEROSPIKE_DEFAULT_PORT));
        client = new AerospikeClient(policy, hosts.toArray(new Host[0]));
    }
}
```

只要注册函数被调用了，就可以看到Java Client打印的日志了。Python Client也是类似的做法：

```python
def as_logger(level, func, myfile, line, msg):
    print("**", myfile, line, func, '::', msg, "**")

aerospike.set_log_level(aerospike.LOG_LEVEL_DEBUG)
aerospike.set_log_handler(as_logger)
```

#### 5、anything else

还有一些最佳实现，跟客户端实现有关。具体可以参考: [Java Client Best Practices](http://www.aerospike.com/docs/client/java/usage/best_practices.html)。


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


### 实战例子——排序和截断

Aerospike并不支持排序和分页（包括简单的截断。。），所以功能上其实是有缺陷的（当然，Aerospike团队并不这么认为。。[No cursor no pagination](https://github.com/aerospike/aerospike-server/issues/178)）。如果我们要这些功能，那么目前看起来比较靠谱的做法就是使用Aerospike的Stream UDFs了。

Aerospike的开发人员在github上写了一个简单的 orderBy 示例代码：[aerospike/orderby-example](https://github.com/aerospike/orderby-example)。

> **Problem** You want to query a set of data in Aerospike and organise the result set using the familiar ORDERBY and GROUPBY qualifiers found in SQL. But queries on secondary indexes do not directly provide a an orderby or grooup by capability.
>
> **Solution** Use an Aggregation query and process the query stream to order and group the results. This works great for small result sets as the whole result set will be returned to the client heapspace.

但是仔细看代码发现其实是一个 groupBy 的功能。。[醉了]。不过话说 orderBy 其实也并不难实现。而且特别适合Stream UDF的map-reduce执行方式，即先在所有的node节点进行排序，最后在client端reduce做最后的排序，其实就是一个分布式的归并排序[Merge Sort](http://bubkoo.com/2014/01/15/sort-algorithm/merge-sort/)算法。

具体代码如下：

```lua
------------------------------------------------------------------------------------------
--  Order By and Limit
------------------------------------------------------------------------------------------
function order_by(stream, arguments)

  local function map_record(rec, fields)
	-- Could add other record bins here as well.
	-- This code shows different data access to record bins
	local result = map()

	if fields ~= nil then -- selected fields
	for v in list.iterator(fields) do
	  result[v] = rec[v]
	end
	end

	if fields == nil then -- all fields
	local names = record.bin_names(rec)
	for i, v in ipairs(names) do
	  result[v] = rec[v]
	end
	end
	result["meta_data"] = map()
	result["meta_data"]["digest"] = record.digest(rec)
	result["meta_data"]["generation"] = record.gen(rec)
	result["meta_data"]["set_name"] = record.setname(rec)
	result["meta_data"]["expiry"] = record.ttl(rec)
	return result
  end

  local function compare(x, y)
    return (x < y and -1 ) or (x == y and 0 or 1)
  end

  local function compare(x, y, order)
    if order == "ASC" then
      return x < y
    else -- DESC
      return y < x
    end
  end

  local function list_truncate(l, limit)
    if list.size(l) > limit then
      info("list.size=%d > limit=%d. Trucate it.", (list.size(l)+1), limit)
      list.trim(l, limit + 1)
    end
  end

  -- insert a rec into a sorted list, return the insertion index for merge sort
  local function insert_sort(sorted_list, rec_map, sort_key, order, start_index)

    if sort_key == nil then -- just append it
      list.append(sorted_list, rec_map)
      return 0
    end

    local v = rec_map[sort_key]
    if v == nil then -- sort value not found in rec_map, just append it
      warn("Can not find value for sort key: %s", sort_key)
      list.append(sorted_list, rec_map)
      return 0
    end

    debug("sort_key: %s, order: %s, value: %s", sort_key, order, v)

    len = list.size(sorted_list)
    for i = start_index or 1, len do
      v2 = sorted_list[i][sort_key]
      if compare(v, v2, order) then
        list.insert(sorted_list, i, rec_map)
        return i
      end
    end

    list.append(sorted_list, rec_map)
    return len
  end

  local function sort_aggregator(sort_key, order, limit)
    -- insert a rec into a sorted list is quite easy
    return function(sorted_list, rec)
      -- convert rec to map
      local rec_map = map_record(rec)

      -- apply orderBy
      insert_sort(sorted_list, rec_map, sort_key, order)

      -- apply limit
      list_truncate(sorted_list, limit)

      return sorted_list
    end
  end

  local function sort_reducer(sort_key, order, limit)
    return function(sorted_list1, sorted_list2)
      -- apply merge sort
      local start_index;
      for i = 1, list.size(sorted_list2) do
        local rec_map = sorted_list2[i]
        start_index = insert_sort(sorted_list1, rec_map, sort_key, order, start_index)
      end

      -- apply limit
      list_truncate(sorted_list1, limit)
      return sorted_list1
    end
  end

  -- default order by id ASC, limit 100
  local sort_key;
  local order;
  local limit = 100

  if arguments ~= nil then
    if arguments["sorters"] ~= nil and list.size(arguments["sorters"]) > 0 then
      local sorter = arguments["sorters"][1] -- only support one sort key right now
      if sorter ~= nil then
        sort_key = sorter["sort_key"] or "id"
        order = sorter["order"] or "ASC"
      end
    end
    limit = arguments["limit"] or 100
  end

  local aggregator = sort_aggregator(sort_key, order, limit)
  local reducer = sort_reducer(sort_key, order, limit)
  return stream : aggregate(list(), aggregator) : reduce(reducer)
end
```

调用代码非常简单，需要注意的是Reduce()函数返回的ResultSet里面只有一个元素(这里是返回一个sorted list)：

```java
private KeyRecordIterator queryAggregateByLua(Statement stmt, Qualifier[] qualifiers, //
            OrderList orderList, int limit) {
    Map<String, Object> argument = new HashMap<>();
    List<Value.MapValue> argumentSorters = new ArrayList<>();
    for (OrderEntry order : orderList) {
        Map<String, Object> s = new HashMap<>();
        s.put("sort_key", order.getProperty());
        s.put("order", order.getOrder().name());
        argumentSorters.add(new Value.MapValue(s));
    }
    argument.put("sorters", new Value.ListValue(argumentSorters));

    if (limit > 0) {
        argument.put("limit", limit);
    }

    stmt.setAggregateFunction(this.getClass().getClassLoader(), AS_UTILITY_PATH, QUERY_MODULE, "order_by",
            Value.get(argument));
    ResultSet resultSet = client.queryAggregate(DEFAULT_QUERY_POLICY, stmt);

    if (resultSet == null) {
        return new KeyRecordIterator(stmt.getNamespace(), Collections.emptyList());
    } else { // aggregate 这里返回的是一个list
        List list = (List) resultSet.iterator().next();
        return new KeyRecordIterator(stmt.getNamespace(), list);
    }
}	
```

**说明**

1. 上面代码只支持单个key排序，多个key比较麻烦，但是并不难实现，读者可以自己扩展。
2. 上面代码只实现了简单的截断，并不支持分页(limit with offset)。这是因为Aerospike Stream UDFs的执行机制导致的。分库环境下是没有办法实现分页的。


### 参考文章

1. [Architecture - User-Defined Functions](http://www.aerospike.com/docs/architecture/udf.html)
2. [Feature Guides - User-Defined Functions](http://www.aerospike.com/docs/guide/udf.html)
3. [User-Defined Functions (UDF) Development Guide](http://www.aerospike.com/docs/udf/udf_guide.html)
4. [Lua UDF – Best Practices](http://www.aerospike.com/docs/udf/best_practices.html)
5. [Java Client Best Practices](http://www.aerospike.com/docs/client/java/usage/best_practices.html)
6. [Developing Stream UDFs](http://www.aerospike.com/docs/udf/developing_stream_udfs.html)
