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

```
➜  Data head company.csv
companyId:ID,name,:LABEL
00253534f3631a5cbdd804f60fefd60a,西藏溥天建设有限责任公司,Company
003e2c44e505953c886520543c89dc6d,上海丽昌超声波工程有限公司,Company
005b2bb9a38f959614120e8f130bddd4,深圳特力得流体系统有限公司销售部,Company
005c01f875938ef2f57a6f7300cc9e30,宁波市江北兴业电力物资经营部,Company
007ab66ccc3010c725d4e4abca46954b,浙江省手工业合作社联合社,Company
00918713ad25f883618e5689f81ad4ed,共青城哈特莱东安投资管理合伙企业（有限合伙）,Company
00c4e2a94545e7e35f80085845496e71,MinghuaCheng,Company
00cf35aa177feaeaee68db5d00129171,深圳飞尔卡思电子有限公司,Company
00fec954458c476a92fc7a9fffa5bc5d,上海蓝科石油化工有限公司,Company
➜  Data head person.csv
personId:ID,name,:LABEL
0001ec01ced8b28fa7a6a4973a26d433,陈美琴,Person
001b3230198e1e73bf1e407503f84b1d,宋又波,Person
001bfb23b2de3865fc61d654a1cf86ce,陈果,Person
002ca864a65643a9c881ec7faca9cdaa,高翔,Person
003d3feb627090b7c6904f2112d29b76,陈松熙,Person
0053cb2d19b27f1e913feb20fc311907,乐胜利,Person
008703badcaf87088717b43fdb10bc97,李晔,Person
00a57f18f4b74fd140b0b48feb5004b2,王晓东,Person
00cc67b543b35bfa60e94010f2f6f82e,覃祖燕,Person
➜  Data head relation.csv
:START_ID,:TYPE,:END_ID,from,to
ea55b396202514e1a93348469eada78d,2,8b6f879c8aa40aed56aac6a01ee339fd,ea55b396202514e1a93348469eada78d,8b6f879c8aa40aed56aac6a01ee339fd
d4395b36984926a1934a0f9b916b32d2,7,8b6f879c8aa40aed56aac6a01ee339fd,d4395b36984926a1934a0f9b916b32d2,8b6f879c8aa40aed56aac6a01ee339fd
d4395b36984926a1934a0f9b916b32d2,9,8b6f879c8aa40aed56aac6a01ee339fd,d4395b36984926a1934a0f9b916b32d2,8b6f879c8aa40aed56aac6a01ee339fd
fbe8227309655b6bee67d28c380c5776,9,8b6f879c8aa40aed56aac6a01ee339fd,fbe8227309655b6bee67d28c380c5776,8b6f879c8aa40aed56aac6a01ee339fd
d4395b36984926a1934a0f9b916b32d2,2,8b6f879c8aa40aed56aac6a01ee339fd,d4395b36984926a1934a0f9b916b32d2,8b6f879c8aa40aed56aac6a01ee339fd
d4395b36984926a1934a0f9b916b32d2,5,8b6f879c8aa40aed56aac6a01ee339fd,d4395b36984926a1934a0f9b916b32d2,8b6f879c8aa40aed56aac6a01ee339fd
ea55b396202514e1a93348469eada78d,0,8b6f879c8aa40aed56aac6a01ee339fd,ea55b396202514e1a93348469eada78d,8b6f879c8aa40aed56aac6a01ee339fd
fbe8227309655b6bee67d28c380c5776,5,8b6f879c8aa40aed56aac6a01ee339fd,fbe8227309655b6bee67d28c380c5776,8b6f879c8aa40aed56aac6a01ee339fd
ea55b396202514e1a93348469eada78d,2,8b6f879c8aa40aed56aac6a01ee339fd,ea55b396202514e1a93348469eada78d,8b6f879c8aa40aed56aac6a01ee339fd
```

**TIPS** mac下执行neo4j的`neo4j-admin import`命令会报如下错误：

```
➜  bin ./neo4j-import --nodes /Users/argan/Data/company.csv --nodes /Users/argan/Data/person.csv --relationships /Users/argan/Data/relation.csv
Error: Could not find or load main class org.neo4j.tooling.ImportTool
➜  bin ./neo4j-admin import --nodes /Users/argan/Data/company.csv --nodes /Users/argan/Data/person.csv --relationships /Users/argan/Data/relation.csv
Error: Could not find or load main class org.neo4j.commandline.admin.AdminTool
➜  app pwd
/Applications/Neo4j Community Edition 3.2.6.app/Contents/Resources/app/bin
```

看起来就是类找不到，谷歌了一下，这是一个常见问题，不知道最新版本fixed了没有。。[MacOS - Neo4j 3.1.0-M13-beta3 neo4j-import - Could not find or load main class #8347](https://github.com/neo4j/neo4j/issues/8347)。只要把bin下面的`neo4j-desktop-3.2.6.jar`copy到一个同级的新建的lib目录就可以了：

```
➜  app mkdir lib
➜  app cp bin/neo4j-desktop-3.2.6.jar lib
total 384
-rw-r--r--@  1 argan  admin    35K Sep 29 21:37 LICENSE.txt
-rw-r--r--@  1 argan  admin   147K Sep 29 21:37 LICENSES.txt
-rw-r--r--@  1 argan  admin   5.6K Sep 29 21:37 NOTICE.txt
drwxr-xr-x@ 13 argan  admin   442B Oct 23 11:23 bin
drwxr-xr-x   3 argan  admin   102B Nov 27 14:51 lib
drwxr-xr-x@  4 argan  admin   136B Nov 20 15:54 plugins
➜  app ll lib
total 171648
-rw-r--r--@ 1 argan  admin    84M Nov 27 14:51 neo4j-desktop-3.2.6.jar
```

再次执行就没有问题了。

```
➜  app bin/neo4j-admin import --nodes /Users/argan/Data/company.csv --nodes /Users/argan/Data/person.csv --relationships /Users/argan/Data/relation.csv
Neo4j version: 3.2.6
Importing the contents of these files into /Applications/Neo4j Community Edition 3.2.6.app/Contents/Resources/app/data/databases/graph.db:
Nodes:
  /Users/argan/Data/company.csv

  /Users/argan/Data/person.csv
Relationships:
  /Users/argan/Data/relation.csv

Available resources:
  Total machine memory: 8.00 GB
  Free machine memory: 27.98 MB
  Max heap memory : 1.78 GB
  Processors: 4
  Configured max memory: -1691220787.00 B

Nodes, started 2017-11-27 06:52:34.636+0000
[*>:??----------------------------------------------------------------------------------------]    0 ∆    0
Done in 194ms
Error in input data
Caused by:Extra column not present in header on line 1130 in /Users/argan/Data/company.csv with value Company

WARNING Import failed. The store files in /Applications/Neo4j Community Edition 3.2.6.app/Contents/Resources/app/data/databases/graph.db are left as they are, although they are likely in an unusable state. Starting a database on these store files will likely fail or observe inconsistent records so start at your own risk or delete the store manually
unexpected error: Extra column not present in header on line 1130 in /Users/argan/Data/company.csv with value Company
```

这是因为我们有一些数据有问题：

```
1130 a9b240d1137fcea2714a685ac0483ed9,北京趣活科技有限公司,Company
1131 b49b7f60fd11ce7e1728cf87c41aacb9,ClearVueYummyExpressHoldings,Ltd,Company
1132 a9b240d1137fcea2714a685ac0483ed9,北京趣活科技有限公司,Company
```
把1131的公司名加上双引号就好了，记得把之前生成的database删除，否则会报错。另外，需要注意的是id不能重复，否则也会报错（也可以通过`--ignore-duplicate-nodes=true`忽略）。

但是执行的时候还有一个蛋疼的问题会引起导入失败，就是关系引用的节点不存在，neo4j默认会报错退出，而不是忽略继续执行：

```
➜  app bin/neo4j-admin import --nodes /Users/argan/Data/company.csv --nodes /Users/argan/Data/person.csv --relationships /Users/argan/Data/relation.csv

...

Nodes, started 2017-11-27 07:32:43.242+0000
[>:??--------------------------||*PROPERTIES---------------------------------||v:??-----------]10.9K ∆10.9K
Done in 327ms
Prepare node index, started 2017-11-27 07:32:43.630+0000
[*DETECT:0.00 B-------------------------------------------------------------------------------]    0 ∆    0
Done in 104ms
Relationships, started 2017-11-27 07:32:43.755+0000
[>:??-----------------------|T|*PREPARE-------------------------------------------------------]    0 ∆    0
Done in 281ms
Error in input data
Caused by:InputRelationship:
   source: /Users/argan/Data/relation.csv:2
   properties: [from, ea55b396202514e1a93348469eada78d, to, 8b6f879c8aa40aed56aac6a01ee339fd]
   startNode: ea55b396202514e1a93348469eada78d (global id space)
   endNode: 8b6f879c8aa40aed56aac6a01ee339fd (global id space)
   type: 2
 referring to missing node ea55b396202514e1a93348469eada78d
```

这可以通过`--ignore-missing-nodes=true`选项忽略：

```
➜  app bin/neo4j-admin import --ignore-missing-nodes=true  --nodes /Users/argan/Data/company.csv --nodes /Users/argan/Data/person.csv --relationships /Users/argan/Data/relation.csv
Neo4j version: 3.2.6
Importing the contents of these files into /Applications/Neo4j Community Edition 3.2.6.app/Contents/Resources/app/data/databases/graph.db:
Nodes:
  /Users/argan/Data/company.csv

  /Users/argan/Data/person.csv
Relationships:
  /Users/argan/Data/relation.csv

Available resources:
  Total machine memory: 8.00 GB
  Free machine memory: 18.06 MB
  Max heap memory : 1.78 GB
  Processors: 4
  Configured max memory: -1700993434.00 B

Nodes, started 2017-11-27 07:41:55.878+0000
[>:??---------------------------|N|*PROPERTIES-------------------------------------|v:??------]10.9K ∆10.9K
Done in 463ms
Prepare node index, started 2017-11-27 07:41:56.400+0000
[*DETECT:0.00 B-------------------------------------------------------------------------------]    0 ∆    0
Done in 34ms
Relationships, started 2017-11-27 07:41:56.450+0000
[>:??--||PREP|*RECORDS(4)============================================================|PROPE|v:]32.7K ∆16.3K
Done in 2s 468ms
Node Degrees, started 2017-11-27 07:41:58.996+0000
[>:??---|*>-----------------------------------------------------------------------------------]32.7K ∆32.7K
Done in 43ms
Relationship --> Relationship  1-9/9, started 2017-11-27 07:41:59.068+0000
[>|*>----------------------------------------------------------------------------------------|]31.2K ∆31.2K
Done in 54ms
RelationshipGroup 1-9/9, started 2017-11-27 07:41:59.150+0000
[*>:??----------------------------------------------------------------------------------------]    0 ∆    0
Done in 12ms
Node --> Relationship, started 2017-11-27 07:41:59.189+0000
[*>:??----------------------------------------------------------------------------------------]    0 ∆    0
Done in 12ms
Relationship <-- Relationship 1-9/9, started 2017-11-27 07:41:59.257+0000
[>|*>-----------------------------------------------------------------------------------------]31.2K ∆31.2K
Done in 56ms
Count groups, started 2017-11-27 07:41:59.351+0000
[*>:??----------------------------------------------------------------------------------------]    0 ∆    0
Done in
Gather, started 2017-11-27 07:41:59.390+0000
[*>:??----------------------------------------------------------------------------------------]    0 ∆    0
Done in
Write, started 2017-11-27 07:41:59.417+0000
[*>:??----------------------------------------------------------------------------------------]    0 ∆    0
Done in
Node --> Group, started 2017-11-27 07:41:59.449+0000
[*>:??----------------------------------------------------------------------------------------]    0 ∆    0
Done in
Node counts, started 2017-11-27 07:41:59.506+0000
[*>--------------------------------------------------|COUNT:0.00 B----------------------------]10.0K ∆10.0K
Done in 13ms
Relationship counts, started 2017-11-27 07:41:59.542+0000
[*>---------------------------------------------------------|COUNT----------------------------]30.0K ∆30.0K
Done in 25ms

IMPORT DONE in 4s 696ms.
Imported:
  6444 nodes
  29700 relationships
  78732 properties
Peak memory usage: 480.06 MB
There were bad entries which were skipped and logged into /Applications/Neo4j Community Edition 3.2.6.app/Contents/Resources/app/import.report

➜  app grep 'referring to missing node' import.report | wc -l
   59400
➜  app grep 'referring to missing node' import.report| sort | uniq | wc -l
    6444
```

**TIPS**

1、导入关系的时候最好先把节点的id索引构建了。这样在做关系节点关联查询的时候会比较快。
2、默认导出的数据会在当前路径的data目录下，导出后把这个文件夹copy到相应目录或者让neo4j加载它就可以了。


--END TIPS--

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


#### 2、如何高效更新更新图数据？

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

**说明** 在`apoc.create.*`方法中，也提供了设置／更新／删除属性和标签的功能。

apoc 3.1之后，还可以动态的更新节点和关系，通过`apoc.merge.node`和`apoc.create.relationship`你可以动态的更新节点标签，关系类型和任意的属性，使用方式跟`apoc.create.node`和`apoc.create.relationship`基本一样：

```
CALL apoc.merge.node(['Label'], {id:uniqueValue}, {prop:value,...}) YIELD node;
CALL apoc.merge.relationship(startNode, 'RELTYPE', {[id:uniqueValue]}, {prop:value}, endNode) YIELD rel;
```

更新节点：

```
UWNIND {batch} as row
CALL apoc.merge.node(row.labels, {id: row.id} , row.properties) yield node
RETURN count(*)
```

更新关系：

```
UWNIND {batch} as row
MATCH (from) WHERE id(n) = row.from
MATCH (to:Label) where to.key = row.to
CALL apoc.merge.relationship(from, row.type, {id: row.id}, row.properties, to) yield rel
RETURN count(*)
```

但是这里有一个比较严重的"bug"，就是apoc的merge操作，本质上就是一个防止重复的CREATE，并不是更新。这点是非常不符合逻辑的。从源码可以看出这是因为它的MERGE只处理ON CREATE场景，这样子其实跟只是避免了重复创建，但是并没有更新功能。[neo4j-apoc-procedures/src/main/java/apoc/merge/Merge.java](https://github.com/neo4j-contrib/neo4j-apoc-procedures/blob/23065dd721c1742dd02b417948b827dbb7cf12a6/src/main/java/apoc/merge/Merge.java)：

```
public class Merge {

    @Context
    public GraphDatabaseService db;

    @Procedure(mode = Mode.WRITE)
    @Description("apoc.merge.node(['Label'], {key:value, ...}, {key:value,...}) - merge node with dynamic labels")
    public Stream<NodeResult> node(@Name("label") List<String> labelNames, @Name("identProps") Map<String, Object> identProps, @Name("props") Map<String, Object> props) {
        
        ...

        final String cypher = "MERGE (n:" + labels + "{" + identPropsString + "}) ON CREATE SET n += $props RETURN n";

        ...
    }

   
    @Procedure(mode = Mode.WRITE)
    @Description("apoc.merge.relationship(startNode, relType,  {key:value, ...}, {key:value, ...}, endNode) - merge relationship with dynamic type")
    public Stream<RelationshipResult> relationship(@Name("startNode") Node startNode, @Name("relationshipType") String relType,
                                                   @Name("identProps") Map<String, Object> identProps, @Name("props") Map<String, Object> props, @Name("endNode") Node endNode) {
        ...

        final String cypher = "WITH $startNode as startNode, $endNode as endNode MERGE (startNode)-[r:"+ wrapInBacktics(relType) +"{"+identPropsString+"}]->(endNode) ON CREATE SET r+= $props RETURN r";

        ...
    }
}
```

从它的单元测试也可以看出来：[neo4j-apoc-procedures/src/test/java/apoc/merge/MergeTest.java](https://github.com/neo4j-contrib/neo4j-apoc-procedures/blob/3.2/src/test/java/apoc/merge/MergeTest.java):

```java
@Test
public void testMergeNode() throws Exception {
    testCall(db, "CALL apoc.merge.node(['Person','Bastard'],{ssid:'123'}, {name:'John'}) YIELD node RETURN node",
            (row) -> {
                Node node = (Node) row.get("node");
                assertEquals(true, node.hasLabel(Label.label("Person")));
                assertEquals(true, node.hasLabel(Label.label("Bastard")));
                assertEquals("John", node.getProperty("name"));
                assertEquals("123", node.getProperty("ssid"));
            });
}

@Test
public void testMergeNodeWithPreExisting() throws Exception {
    db.execute("CREATE (p:Person{ssid:'123', name:'Jim'})");
    testCall(db, "CALL apoc.merge.node(['Person'],{ssid:'123'}, {name:'John'}) YIELD node RETURN node",
            (row) -> {
                Node node = (Node) row.get("node");
                assertEquals(true, node.hasLabel(Label.label("Person")));
                assertEquals("Jim", node.getProperty("name"));
                assertEquals("123", node.getProperty("ssid"));
            });

    testResult(db, "match (p:Person) return count(*) as c", result ->
            assertEquals(1, (long)(Iterators.single(result.columnAs("c"))))
    );
}

@Test
public void testMergeRelationships() throws Exception {
    db.execute("create (:Person{name:'Foo'}), (:Person{name:'Bar'})");

    testCall(db, "MERGE (s:Person{name:'Foo'}) MERGE (e:Person{name:'Bar'}) WITH s,e CALL apoc.merge.relationship(s, 'KNOWS', {rid:123}, {since:'Thu'}, e) YIELD rel RETURN rel",
            (row) -> {
                Relationship rel = (Relationship) row.get("rel");
                assertEquals("KNOWS", rel.getType().name());
                assertEquals(123l, rel.getProperty("rid"));
                assertEquals("Thu", rel.getProperty("since"));
            });

    testCall(db, "MERGE (s:Person{name:'Foo'}) MERGE (e:Person{name:'Bar'}) WITH s,e CALL apoc.merge.relationship(s, 'KNOWS', {rid:123}, {since:'Fri'}, e) YIELD rel RETURN rel",
            (row) -> {
                Relationship rel = (Relationship) row.get("rel");
                assertEquals("KNOWS", rel.getType().name());
                assertEquals(123l, rel.getProperty("rid"));
                assertEquals("Thu", rel.getProperty("since"));
            });
    testCall(db, "MERGE (s:Person{name:'Foo'}) MERGE (e:Person{name:'Bar'}) WITH s,e CALL apoc.merge.relationship(s, 'OTHER', null, null, e) YIELD rel RETURN rel",
            (row) -> {
                Relationship rel = (Relationship) row.get("rel");
                assertEquals("OTHER", rel.getType().name());
                assertTrue(rel.getAllProperties().isEmpty());
            });
}
```

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

Cypher为列表提供了相当便利的操作，如range, collect, unwind, reduce, extract, filter, size等，反而对Map的创建和更新操作支持比较弱。还好，`apoc.map.*`提供了一系列的方法来简化这个过程。


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

#### 3、neo4j如何实现存在就更新，否则插入？

这个需求其实很普遍，比如我有一个节点: `(n:Person {id: 'argan', name: 'argan', age: 32})`。

然后用户又传递了一个person数据过来： `{id: 'argan', age: 30, sex: 'male', email: 'arganzheng@gmail.com'}`

可以看到更新了一个属性：age，新增了两个属性：sex和email。

我们希望最后的结果是 `(n:Person {id: 'argan', name: 'argan', age: 30, sex: 'male', email: 'arganzheng@gmail.com'})`。

需要注意的是name没有传递，所以还是保留着的。如果要删除一个属性，需要把它的value显式的设置为null。

在neo4j要怎么做到呢？

neo4j提供了[merge语句](http://neo4j.com/docs/developer-manual/current/cypher/clauses/merge/)来实现这个功能。

[Merge with ON CREATE and ON MATCH](http://neo4j.com/docs/developer-manual/current/cypher/clauses/merge/#merge-merge-with-on-create-and-on-match)

> Merge a node and set properties if the node needs to be created.

```cypher
MERGE (n:Person { id: 'argan' })
ON CREATE SET n.created = timestamp()
ON MATCH SET n.lastAccessed = timestamp()
RETURN n.name, n.created, n.lastAccessed
```

上面的例子可以这么写：

```
MERGE (n:Node {id: 'argan'})
SET n += {id: 'argan', age: 30, sex: 'male', email: 'arganzheng@gmail.com'}
RETURN n 
```

这里因为采用了 `+=` 本身就是合并属性，所以不需要区分是`ON CREATE`还是`ON MATCH`。 

同样关系也可以用merge保证只创建一次：

```
MATCH (n), (m)
WHERE n.id = "argan" AND m.id = "magi"
CREATE (n)-[:KNOWS]->(m)
```

写成这样子就可以保证唯一了：

```
MATCH (n:User {name: "argan"}), (m:User {name: "magi"})
MERGE (n)-[:KNOWS]->(m)
```

一个好消息是 apoc 3.1之后终于开始支持这个特性了：[Summer 2017 Release of the APOC Procedures Librar](https://dzone.com/articles/summer-2017-release-of-the-apoc-procedures-library)。

> There are now procedures to merge nodes and relationships with dynamic labels, relationship-types, and properties (apoc.merge.node/relationship).
>
>```
> CALL apoc.merge.node(['Label'], {id:uniqueValue}, {prop:value,...}) YIELD node;
> CALL apoc.merge.relationship(startNode, 'RELTYPE', {[id:uniqueValue]}, {prop:value}, endNode) YIELD rel;
>```

具体实现和用法我们上面已经分析过了。这里就不赘述了。源码地址：[neo4j-apoc-procedures/src/main/java/apoc/merge/Merge.java](https://github.com/neo4j-contrib/neo4j-apoc-procedures/blob/23065dd721c1742dd02b417948b827dbb7cf12a6/src/main/java/apoc/merge/Merge.java) ，单侧地址：[neo4j-apoc-procedures/src/test/java/apoc/merge/MergeTest.java](https://github.com/neo4j-contrib/neo4j-apoc-procedures/blob/3.2/src/test/java/apoc/merge/MergeTest.java) 。

但是奇怪的是 [apoc官方文档](https://neo4j-contrib.github.io/neo4j-apoc-procedures/) 并没有相关的说明 :-( 。

**TIPS**

虽然apoc和cypher比较方便，但是很容易语法错误，特别是变量动态拼接的时候。有时候还不如参考一下 apoc 的例子：[neo4j-apoc-procedures/src/main/java/apoc/merge/Merge.java](https://github.com/neo4j-contrib/neo4j-apoc-procedures/blob/23065dd721c1742dd02b417948b827dbb7cf12a6/src/main/java/apoc/merge/Merge.java) 自己实现。


#### 4、neo4j如何支持动态node label？

我们希望构建的节点数据完全是运行时根据用户提供的数据构造的，包括labels。比如用户提供如下数据：

```
{"batch":[{"id":"1","properties":{"name":"argan","label":"Person","id":"1","age":31,"tags":["smart","handson","rich"]}}, {"id":"1","properties":{"name":"magi","label":"Person","id":"2","age":28,"tags":["tall","handson","rich"]}}]}
```

我们希望能够直接这样子批量插入数据：

```cypher
UNWIND {batch} as row 
MERGE (n:row.properties.label {id: row.id})
SET n += row.properties
```

但是遗憾的是这个语句会报错，因为neo4j不支持动态的node labels。把`row.properties.label`去掉或者改成一个固定的字符串就没有问题。

改成这样子也不行：

```
UNWIND {batch} as row  
MERGE (n {id: row.id} )  
SET n:row.properties.label, 
n += row.properties
```

绑定变量也不行：

```
UNWIND {batch} as row  
MERGE (n {id: row.id} )  
SET n:{label}, 
n += row.properties
```

直接指定label就可以了：

```
UNWIND {batch} as row  
MERGE (n {id: row.id} )  
SET n:Test, 
n += row.properties
```

也就是说 [3.3.13.9. Set a label on a node](http://neo4j.com/docs/developer-manual/current/cypher/clauses/set/#set-set-a-label-on-a-node) 也并不支持动态label。。

**NOTES**

neo4j的Set label还有一个问题，就是它其实是新增label，不是修改label。要到更新的效果，你需要先remove掉，再新增。。

```
MATCH (n)
WHERE ID(n) = 14 
REMOVE n:oldLabel
SET n:newLabel
```

---

如果是单条数据更新，那其实很简单，我们只需要做字符串拼接就可以了：

```
String label = vertex.getLabel();
"MERGE (n:" + label + " {id: {id}} " + "SET n += {properties}"
```


但是关键是我们这里是在neo4j内部用unwind展开的服务端变量，如果它不允许动态变量，根本搞不定。难道真的要一条条的插入，那会非常慢的！neo4j的插入性能是众所周知的差。一种做法就是先批量插入数据，设置一个临时的label，然后再批量的更新label。不过需要两次操作，性能肯定至少慢两倍。

有没有什么方式呢？谷歌了很久，发现了也有人遇到这样的问题：[Feature request : apoc support for MERGE for nodes and rels #271](https://github.com/neo4j-contrib/neo4j-apoc-procedures/issues/271) 和 [Is it possible to merge using data-driven node or relationship labels?](https://stackoverflow.com/questions/43740000/is-it-possible-to-merge-using-data-driven-node-or-relationship-labels)。

原理跟单条数据插入一样，只是由于unwind是在服务端(neo4j)进行的，所以拼接也只能在服务端进行，怎么拼接的？使用的就是cypher的[with语句](http://neo4j.com/docs/developer-manual/current/cypher/clauses/with/)：

> The WITH clause allows query parts to be chained together, piping the results from one to be used as starting points or criteria in the next.

但是官方文档说的很简略，其实with的功能很强大。在这里，我们用它来做服务端的字符串拼接：

```
UNWIND {batch} as row 
WITH 'MERGE (n:' + row.properties.label + ' { id: row.id }) SET n += row.properties return n' AS cypher" 
CALL apoc.cypher.doIt(cypher, {}) YIELD value return value.n
```

但是可惜，会报这样的异常：

```
org.neo4j.driver.v1.exceptions.ClientException: Failed to invoke procedure `apoc.cypher.doIt`: Caused by: org.neo4j.graphdb.QueryExecutionException: Variable `row` not defined (line 1, column 23 (offset: 22))
"MERGE (n:Person { id: row.id }) SET n += row.properties return n"
```

所以还是要分两步进行，不过可以合并在一起 [SET label : pass label name as parameter](https://stackoverflow.com/questions/27189237/set-label-pass-label-name-as-parameter)：

```
UNWIND {batch} as row 
MERGE (n { id: row.id }) SET n += row.properties 
WITH n 
CALL apoc.create.addLabels(id(n), [n.label]) YIELD node
RETURN node
```

这样就可以了，测试了一下，性能并没有造成什么影响。但是说句实在话，neo4j的插入性能真心弱。一次普通插入在12ms左右，这还是执行计划缓存的情况下，要不要100多毫秒。。

**NOTES**

本来我们是可以直接使用APOC库的`apoc.merge.node`和`apoc.create.relationship`动态的更新节点标签，关系和节点的。但是正如前面分析的，`apoc.merge.node`和`apoc.create.relationship`现在的实现其实是一个防重复CREATE而已，不能达到更新的目的。否则我们的实现将非常简单明了：


更新节点：

```
UWNIND {batch} as row
CALL apoc.merge.node(row.labels, {id: row.id} , row.properties) yield node
RETURN count(*)
```

更新关系：

```
UWNIND {batch} as row
MATCH (from) WHERE id(from) = row.from
MATCH (to:Label) where to.key = row.to
CALL apoc.merge.relationship(from, row.type, {id: row.id}, row.properties, to) yield rel
RETURN count(*)
```

一种做法就是fork一个分支出来，修改源码，deploy自己的jar包。


更新关系也是类似的做法：

```
UNWIND {batch} as row 
MATCH (from { id: row.from })
MATCH (to { id: row.to }) 
WITH from, to, row
MERGE (from)-[r:type {id: row.id}]->(to)
SET r += row.properties 
WITH r, row.properties.label AS type
CALL apoc.refactor.setType(r, type) YIELD input, output
RETURN 1
```

但是还有一个问题，就是边的方向，这个就没有办法变量直接替换了，只能做条件判断，好在边就三个方向，直接枚举判断就可以了。neo4j中没有条件判断语句，但是可以用 FOREACH trick实现。这个在前面 [有条件的创建数据（Conditional Data Creation）]() 有介绍过。不过非常麻烦，我还是建议直接创建两条单向关系简单直接。


参考文章
-------

1. [5 Tips & Tricks for Fast Batched Updates of Graph Structures with Neo4j and Cypher](http://jexp.de/blog/2017/03/5-tips-tricks-for-fast-batched-updates-of-graph-structures-with-neo4j-and-cypher/#_inefficient_solutions) 
2. [Create Dynamic Relationships With APOC](https://dzone.com/articles/neo4j-create-dynamic-relationship-type)
3. [Neo4j: Dynamically Add Property/Set Dynamic Property](https://dzone.com/articles/neo4j-dynamically-add-propertyset-dynamic-property)
4. [APOC: Database Integration, Import and Export with Awesome Procedures on Cypher](https://dzone.com/articles/apoc-database-integration-import-and-export-with-a)
5. [DaniSancas/neo4j_cypher_cheatsheet.md](https://gist.github.com/DaniSancas/1d5265fc159a95ff457b940fc5046887)
6. [Neo4j的查询速度为何这么慢？这能商用吗？](https://www.zhihu.com/question/45401120) 知乎上有人回答了这个问题，有案例有数据，强烈推荐。
7. [如何将大规模数据导入Neo4j](http://paradoxlife.me/how-to-insert-bulk-data-into-neo4j) 几种方式不错的对比
8. [Neo4j的存储结构](http://m.blog.csdn.net/huaishu/article/details/11713753) 为什么neo4j有时候快，有时候慢
