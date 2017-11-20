---
title: neo4j如何批量导入JSON数据
layout: post
catalog: true
tags: [neo4j, graph database]
---

neo4j提供了如下批量导入工具：

1. 通过`neo4j-admin import`命令可以将数据从 CSV 文件批量导入到未使用的数据库。load data into a previously unused database.
2. 通过`[LOAD CSV](https://neo4j.com/docs/developer-manual/3.2/cypher/clauses/load-csv/)`语句 import small to medium-sized CSV files into an existing database.

CSV数据格式参见: [10.2.2. CSV file header format](https://neo4j.com/docs/operations-manual/current/tools/import/file-header-format/)。

neo4j会根据把机器的CPU和磁盘用到极限，在这篇文章中 [Import 10M Stack Overflow Questions into Neo4j In Just 3 Minutes](https://neo4j.com/blog/import-10m-stack-overflow-questions/)，有一些数据可以参考，总体来所还是可以的：

>The actual import only takes 3 minutes, creating a graph store of 18 GB.
>
> IMPORT DONE in 3m 48s 579ms. Imported:
>  31138559 nodes
>  77930024 relationships
>  260665346 properties

大概 3千万个节点，7千八百万条边和 2亿6千万个属性，导入花费了3分48秒。

整个详细过程在这篇文章有描述，感兴趣的同学可以看看：[Effective Bulk Data Import into Neo4j](https://neo4j.com/blog/bulk-data-import-neo4j-3-0/)。

整个工具是现成的，而且性能还可以的，但是有个问题，就是他们的数据格式都是针对CSV格式的。但是实际上大部分情况下，原始数据跟多的是以JSON的形式存在。上面的例子也是如此，所以他们利用 [jq](https://stedolan.github.io/jq/) 把JSON转成了CSV格式了。转换时间比导入还慢。

因为这个需求比较常见，所以 [neo4j APOC](https://neo4j-contrib.github.io/neo4j-apoc-procedures/#_load_json) 就封装了一个 `apoc.load.json` 的 procedure，其实它内部的操作也是把json先转成csv，只不是用的是java代码而不是jq：

	WITH "https://api.stackexchange.com/2.2/questions?pagesize=100&order=desc&sort=creation&tagged=neo4j&site=stackoverflow&filter=!5-i6Zw8Y)4W7vpy91PMYsKM-k9yzEsSC1_Uxlf" AS url
	CALL apoc.load.json(url) YIELD value
	UNWIND value.items AS item
	RETURN item.title, item.owner, item.creation_date, keys(item)

虽然举的例子都是从 web-api 获取的JSON数据，实际上也可以从文件中读取JSON数据：

To get the first 1000 JSON objects from the array in the file, try this:

	WITH "file:///path_to_file.json" as url  
	CALL apoc.load.json(url, '[0:1000]') YIELD value AS article
	RETURN article;


事实上，Cypher 本身就是支持 json 对象作为map的。[jexp/load_conference.groovy](https://gist.github.com/jexp/7c6997242c001abfb2cd) 或者 [Cypher: LOAD JSON from URL AS Data](https://neo4j.com/blog/cypher-load-json-from-url/) ：

	WITH {json} as data
	UNWIND data.items as q
	MERGE (question:Question {id:q.question_id}) ON CREATE
	  SET question.title = q.title, question.share_link = q.share_link, question.favorite_count = q.favorite_count

	MERGE (owner:User {id:q.owner.user_id}) ON CREATE SET owner.display_name = q.owner.display_name
	MERGE (owner)-[:ASKED]->(question)

	FOREACH (tagName IN q.tags | MERGE (tag:Tag {name:tagName}) MERGE (question)-[:TAGGED]->(tag))
	FOREACH (a IN q.answers |
	   MERGE (question)<-[:ANSWERS]-(answer:Answer {id:a.answer_id})
	   MERGE (answerer:User {id:a.owner.user_id}) ON CREATE SET answerer.display_name = a.owner.display_name
	   MERGE (answer)<-[:PROVIDED]-(answerer)
	)

java代码：

	import org.apache.http.*;
	import org.codehaus.jackson.map.ObjectMapper;
	import org.neo4j.graphdb.*;

	// somewhere in your application-scoped setup code
	ObjectMapper mapper = new ObjectMapper();
	HttpClient http = HttpClients.createMinimal();
	GraphDatabaseService db = new GraphDatabaseFactory().newEmbeddedGraphDatabase(PATH);

	// execute API request and parse response as JSON
	HttpResponse response = http.execute(new HttpGet( apiUrl ));
	Map json = mapper.readValue(response.getEntity().getContent(), Map.class)

	// execute Cypher
	String query = "UNWIND {json} AS data ....";
	db.execute(query, singletonMap("json",json));

	// application scoped shutdown, or JVM-shutdown-hook
	db.shutdown();


### 一些思考和谈论

#### 1、如何为两个存在的顶点创建关系?

通常思路是这样子的：

先创建索引：

	CREATE INDEX ON :User(username)
	CREATE INDEX ON :Role(name)

再创建关系：

	MATCH (u:User {username:'admin'}), (r:Role {name:'ROLE_WEB_USER'})
	CREATE (u)-[:HAS_ROLE]->(r)

但是Stack Overflow上有人提出这种做法插入QPS非常低(40左右) (https://stackoverflow.com/questions/33474497/how-to-add-edgesrelationship-neo4j-in-a-very-big-graph)[How to add edges(relationship) Neo4j in a very big graph]，同时也有人指出这会导致 noe4j对所有的节点进行笛卡尔积运算。具体还是要自己测试一下，用[`EXPLAIN`或者`Profile`进行相应的调优](http://neo4j.com/docs/stable/how-do-i-profile-a-query.html)。

这篇文章 [5 Tips & Tricks for Fast Batched Updates of Graph Structures with Neo4j and Cypher](http://jexp.de/blog/2017/03/5-tips-tricks-for-fast-batched-updates-of-graph-structures-with-neo4j-and-cypher/#_inefficient_solutions) 给出了非常好的建议和tips。

**低效的方式**

* 将值直接写入到语句中，而不是通过参数的方式 
* 每一个更新都通过一个Transaction发送一个请求 
* 通过一个Transaction发送大量的单个请求 
* 生成一个巨大复杂的语句（几百行），然后通过一个Transaction进行提交 
* 在一个Transaction中，发送一个巨大的请求，会导致OOM错误

**正确的方式**

你需要构造尽可能小的请求，并且语句格式固定（这样可以利用缓存），然后通过参数方式进行使用。

每一个请求可以只修改一个属性，或者修改整个子图（上百个节点），但是它的语句结构必须是一致的，否则就不能使用缓存。


文章中建议使用 `UNWIND` 进行批量操作：

> To achieve that you just prefix your regular “single-update-query” with an UNWIND that turns a batch of data (up to 10k or 50k entries) into individual rows, which contain the information for each of the (more or less complex) updates.
> 
> You send in a {batch} parameter (up to 10k-50k) of data (hopefully a delta) as a list of maps, which are then applied in a compact query, which is also properly compiled and cached, as it has a fixed structure.

语法结构如下：

```
{batch: [{row1},{row2},{row3},...10k]}

UNWIND {batch} as row

// now perform updates with the data in each "row" map
```

下面是一些例子：

1、创建节点（Create node with properties）

```cypher
{batch: [{name:"Alice",age:32},{name:"Bob",age:42}]}

UNWIND {batch} as row
CREATE (n:Label)
SET n += row
```

2、更新节点信息（MERGE node with properties）

```cypher
{batch: [{id:"alice@example.com",properties:{name:"Alice",age:32}},{id:"bob@example.com",properties:{name:"Bob",age:42}}]}

UNWIND {batch} as row
MERGE (n:Label {row.id})
(ON CREATE) SET n += row.properties
```

3、节点间创建关系（Node lookup and MERGE/CREATE relationship between with properties）

```cypher
{batch: [{from:"alice@example.com",to:"bob@example.com",properties:{since:2012}},{from:"alice@example.com",to:"charlie@example.com",properties:{since:2016}}]}

UNWIND {batch} as row
MATCH (from:Label {row.from})
MATCH (to:Label {row.to})
CREATE/MERGE (from)-[rel:KNOWS]->(to)
(ON CREATE) SET rel += row.properties
```

4、根据id或者id列表查询（Lookup by id, or even list of ids）

good for parent-child trees.

```cypher
{batch: [{from:123,to:[44,12,128],created:"2016-01-13"}, {from:34,to:[23,35,2983],created:"2016-01-15"},...]

UNWIND {batch} as row
MATCH (from) WHERE id(from) = row.from
MATCH (to) WHERE id(from) IN row.to // list of ids
CREATE/MERGE (from)-[rel:FOO]->(to)
SET rel.created = row.created
```

注意：文章例子里有不少根据noe4j内部id进行查询更新的操作，这当然更快，因为不需要走索引，但是这其实是不符合neo4j的推荐的，因为这些id会回收可能会出错。

5、有条件的创建数据（Conditional Data Creation）

有些时候，你希望根据输入动态的创建数据。但是Cypher目前没有诸如WHEN或者IF的条件语句，CASE WHEN也只是一个表达式。作者这里想出了一个trick：Cypher提供FOREACH语句，用来遍历列表中的每一个元素并分别执行更新操作。于是，一个包含0个元素或者1个元素的列表则可以看成一个条件表达式。因为当0个元素的时候，就不会执行遍历，而当1个元素的时候，就只执行一次遍历。

大致思路如下：

```
...
FOREACH (_ IN CASE WHEN predicate THEN [true] ELSE [] END |
... update operations ....
)
```

其中，列表中的true值可以是其他任何值，42，""，null等等。只要它是一个值，那么我们就可以得到一个非空的列表。

相似的，你也可以使用RANGE(1, CASE WHEN predicate THEN 1 ELSE 0 END)。当predicate的值为false的时候，就会范围一个空列表。或者，如果你喜欢使用filter，那么也可以通过filter(_ IN [1] WHERE predicate)来构造。


下面是一个完整的示例：

```
LOAD CSV FROM {url} AS row
MATCH (o:Organization {name:row.org})
FOREACH (_ IN case when row.type = 'Person' then [1] else [] end|
   MERGE (p:Person {name:row.name})
   CREATE (p)-[:WORKS_FOR]->(o)
)
FOREACH (_ IN case when row.type = 'Agency' then [1] else [] end|
   MERGE (a:Agency {name:row.name})
   CREATE (a)-[:WORKS_FOR]->(o)
)
```

需要注意的是，在FOREACH内部创建的变量无法在外部访问。你需要再重新查询一次，或者你需要再FOREACH内完成全部更新操作。

#### 使用APOC库

另一种方式是使用APOC库。

APOC库提供了很多有用的方法供你使用。在这里，我推荐下面3个方法：

* 创建节点和关系，并且可以动态设定标签和属性
* 批量提交和更新
* 动态创建或者操作Map，并赋给属性

1、动态创建节点和关系

通过`apoc.create.node`和`apoc.create.relationship`你可以动态的计算节点标签，关系类型和任意的属性。

* 标签是一个String数组
* 属性就是一个Map

```
UWNIND {batch} as row
CALL apoc.create.node(row.labels, row.properties) yield node
RETURN count(*)
```

创建关系：

```
UWNIND {batch} as row
MATCH (from) WHERE id(n) = row.from
MATCH (to:Label) where to.key = row.to
CALL apoc.create.relationship(from, row.type, row.properties, to) yield rel
RETURN count(*)
```

说明：在`apoc.create.*`方法中，也提供了设置／更新／删除属性和标签的功能。

2、批量提交

大量的提交Transaction是有问题的。你可以用2G-4G的heap来更新百万条记录，但当量级更大了之后就会很困难了。在使用32G的heap下，我最大的Transaction可以达到10M的节点。

这时，apoc.periodic.iterate可以提供很大的帮助。

它的原理很简单：你有两个Cypher语句，第一条语句能够提供可操纵的数据并产生巨大的数据流，第二条语句执行真正的更新操作，它对每一个数据都进行一次更新操作，但是它只在处理一定数量的数据后才创建一个新的Transaction。

打个比方，假如你第一条语句返回了五百万个需要更新的节点，如果使用内部语句的话，那么每一个节点都会进行一次更新操作。但是如果你设置批处理大小为10k的话，那么每一个Transaction会批量更新10k的节点。

如果你的更新操作是相互独立的话（创建节点，更新属性或者更新独立的子图），那么你可以添加parallel:true来充分利用cpu。

比方说，你想计算多个物品的评分，并通过批处理的方式来更新属性，你应该按下面这样操作

```
call apoc.periodic.iterate('
MATCH (n:User)-[r1:LIKES]->(thing)<-[r2:RATED]-(m:User) WHERE id(n)<id(m) RETURN thing, avg( r1.rating + r2.rating ) as score
','
WITH {thing} as t SET t.score = {score}
', {batchSize:10000, parallel:true})
```

3、动态创建／更新Map

Cypher为列表提供了相当便利的操作，如range, collect, unwind, reduce, extract, filter, size等，反而对Map的创建和更新操作支持比较弱。还好，apoc.map.*提供了一系列的方法来简化这个过程。


通过其他数据创建Map：

```
RETURN apoc.map.fromPairs([["alice",38],["bob",42],...​])
// {alice:38, bob: 42, ...}

RETURN apoc.map.fromLists(["alice","bob",...],[38,42])
// {alice:38, bob: 42, ...}

// groups nodes, relationships, maps by key, good for quick lookups by that key
RETURN apoc.map.groupBy([{name:"alice",gender:"female"},{name:"bob",gender:"male"}],"gender")
// {female:{name:"alice",gender:"female"}, male:{name:"bob",gender:"male"}}

RETURN apoc.map.groupByMulti([{name:"alice",gender:"female"},{name:"bob",gender:"male"},{name:"Jane",gender:"female"}],"gender")
// {female:[{name:"alice",gender:"female"},{name:"jane",gender:"female"}], male:[{name:"bob",gender:"male"}]}
```

更新Map:

```
RETURN apoc.map.merge({alice: 38},{bob:42})
// {alice:38, bob: 42}

RETURN apoc.map.setKey({alice:38},"bob",42)
// {alice:38, bob: 42}

RETURN apoc.map.removeKey({alice:38, bob: 42},"alice")
// {bob: 42}

RETURN apoc.map.removeKey({alice:38, bob: 42},["alice","bob","charlie"])
// {}

// remove the given keys and values, good for data from load-csv/json/jdbc/xml
RETURN apoc.map.clean({name: "Alice", ssn:2324434, age:"n/a", location:""},["ssn"],["n/a",""])
// {name:"Alice"}
```


参考文章
-------

1. [5 Tips & Tricks for Fast Batched Updates of Graph Structures with Neo4j and Cypher](http://jexp.de/blog/2017/03/5-tips-tricks-for-fast-batched-updates-of-graph-structures-with-neo4j-and-cypher/#_inefficient_solutions) 

