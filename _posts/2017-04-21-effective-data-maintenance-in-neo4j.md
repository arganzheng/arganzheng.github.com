---
title: neo4j高效数据维护
layout: post
catalog: true
tags: [neo4j, 图数据库]
---


### 1、如何为两个存在的顶点创建关系?

通常思路是这样子的：

先创建索引：

	CREATE INDEX ON :User(username)
	CREATE INDEX ON :Role(name)

再创建关系：

	MATCH (u:User {username:'admin'}), (r:Role {name:'ROLE_WEB_USER'})
	CREATE (u)-[:HAS_ROLE]->(r)

**注意**

1、节点MATCH部分一定要走索引，否则数据量大的情况下会导致noe4j对所有的节点进行扫描过滤，直接卡死。更重要的是这个操作是有加锁的，会影响到其他读写操作。

2、关于关系如何实现存在就更新，否则插入的逻辑，可以参考笔者写的另一篇文章 [neo4j如何实现存在就更新，否则插入？](http://arganzheng.life/neo4j-create-if-not-exist-otherwise-update.html)，这里就不赘述了。


### 2、如何高效更新图数据？

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


### 3、使用APOC库

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
