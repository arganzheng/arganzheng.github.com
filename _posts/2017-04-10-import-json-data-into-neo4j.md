---
title: neo4j如何批量导入JSON数据
layout: post
catalog: true
tags: [neo4j, 图数据库]
---


neo4j提供了如下批量导入工具：

1. 通过`neo4j-admin import`命令可以将数据从 CSV 文件批量导入到未使用的数据库。load data into a previously unused database.
2. 通过`[LOAD CSV](https://neo4j.com/docs/developer-manual/3.2/cypher/clauses/load-csv/)`语句 import small to medium-sized CSV files into an existing database.

CSV数据格式参见: [10.2.2. CSV file header format](https://neo4j.com/docs/operations-manual/current/tools/import/file-header-format/)。

```
➜  Data head company.csv
id:ID,name,:LABEL
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
id:ID,name,:LABEL
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

**TIPS** 


1、格式方面，也可以把header单独放另外的文件，这样可以保证数据文件的一致性（比如csv是从hadoop跑出来的），这会方便很多。具体参见: [https://neo4j.com/docs/operations-manual/current/tutorial/import-tool/#import-tool-separate-headers-example](https://neo4j.com/docs/operations-manual/current/tutorial/import-tool/)。

> #### B.4.3. Using separate header files
> 
> When dealing with very large CSV files it is more convenient to have the header in a separate file. This makes it easier to edit the header as you avoid having to open a huge data file just to change it.
> 
> 	neo4j_home$ bin/neo4j-admin import --nodes "import/movies3-header.csv,import/movies3.csv" --nodes "import/actors3-header.csv,import/actors3.csv" --relationships "import/roles3-header.csv,import/roles3.csv"
> 

2、可以把同一个label的csv文件分成多个，比如：person-part1.csv, person-part2.csv，只要把它们写在一个`--nodes`参数就可以了：

```
neo4j_home$ bin/neo4j-admin import --nodes "import/movies4-header.csv,import/movies4-part1.csv,import/movies4-part2.csv" --nodes "import/actors4-header.csv,import/actors4-part1.csv,import/actors4-part2.csv" --relationships "import/roles4-header.csv,import/roles4-part1.csv,import/roles4-part2.csv"
```


3、mac下执行neo4j的`neo4j-admin import`命令会报如下错误：

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
2、mac下默认导出的数据会在当前路径的data目录下，导出后把这个文件夹copy到相应目录或者让neo4j加载它就可以了。


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


**TIPS** 测试环境导入1.2亿的节点和2亿的关系，耗时大概在16分钟左右。

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


参考文章
-------

1. [APOC: Database Integration, Import and Export with Awesome Procedures on Cypher](https://dzone.com/articles/apoc-database-integration-import-and-export-with-a)
2. [DaniSancas/neo4j_cypher_cheatsheet.md](https://gist.github.com/DaniSancas/1d5265fc159a95ff457b940fc5046887)
3. [Neo4j的查询速度为何这么慢？这能商用吗？](https://www.zhihu.com/question/45401120) 知乎上有人回答了这个问题，有案例有数据，强烈推荐。
4. [如何将大规模数据导入Neo4j](http://paradoxlife.me/how-to-insert-bulk-data-into-neo4j) 几种方式不错的对比
5. [Neo4j的存储结构](http://m.blog.csdn.net/huaishu/article/details/11713753) 为什么neo4j有时候快，有时候慢
6. [https://neo4j.com/docs/operations-manual/current/tutorial/import-tool/#import-tool-separate-headers-example](https://neo4j.com/docs/operations-manual/current/tutorial/import-tool/) 非常详细的官方说明，使用admin-import工具必看

