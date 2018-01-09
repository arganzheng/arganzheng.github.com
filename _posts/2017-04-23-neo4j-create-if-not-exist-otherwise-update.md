---
title: neo4j如何实现存在就更新，否则插入？
layout: post
catalog: true
tags: [neo4j, 图数据库]
---


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

一个好消息是 apoc 3.1之后终于开始支持这个特性了：[Summer 2017 Release of the APOC Procedures Librar](https://dzone.com/articles/summer-2017-release-of-the-apoc-procedures-library)。通过`apoc.merge.node`和`apoc.create.relationship`你可以动态的更新节点标签，关系类型和任意的属性，使用方式跟`apoc.create.node`和`apoc.create.relationship`基本一样：

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


> There are now procedures to merge nodes and relationships with dynamic labels, relationship-types, and properties (apoc.merge.node/relationship).
>
>```
> CALL apoc.merge.node(['Label'], {id:uniqueValue}, {prop:value,...}) YIELD node;
> CALL apoc.merge.relationship(startNode, 'RELTYPE', {[id:uniqueValue]}, {prop:value}, endNode) YIELD rel;
>```

但是奇怪的是 [apoc官方文档](https://neo4j-contrib.github.io/neo4j-apoc-procedures/) 并没有相关的说明 :-( 。

**TIPS**

虽然apoc和cypher比较方便，但是很容易语法错误，特别是变量动态拼接的时候。有时候还不如参考一下 apoc 的例子：[neo4j-apoc-procedures/src/main/java/apoc/merge/Merge.java](https://github.com/neo4j-contrib/neo4j-apoc-procedures/blob/23065dd721c1742dd02b417948b827dbb7cf12a6/src/main/java/apoc/merge/Merge.java) 自己实现。


### neo4j如何支持动态node label和relationship type？

上面的merge语句能够实现“存在更新，否则创建”的逻辑，但是还有一个问题没有解决，就是没有设置节点的label。我们希望构建的节点数据完全是运行时根据用户提供的数据构造的，包括label。比如用户提供如下数据：


```cypher
:param batch: [{properties: {name: "argan", label: "Person", id: "1", age: 31}}, {properties: {name: "magi", label: "Person", id: "2", age: 28}}]
```

下面的cypher语句并没有设置节点label，虽然节点有一个叫做label的属性:

```cypher
UNWIND {batch} as row 
MERGE (n {id: row.id})
SET n += row.properties
```

那么我们能不能简单的指定label呢？

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

如果是单条数据更新，那其实很简单，我们只需要做字符串拼接就可以了：

```
String label = vertex.getLabel();
"MERGE (n:" + label + " {id: {id}} " + "SET n += {properties}"
```

但是关键是我们这里是在neo4j内部用unwind展开的服务端变量，如果它不允许动态变量，根本搞不定。难道真的要一条条的插入，那会非常慢的！neo4j的插入性能是众所周知的差。一种做法就是先批量插入数据，设置一个临时的label，然后再批量的更新label。不过需要两次操作，性能肯定至少慢两倍。

有没有什么方式呢？谷歌了很久，发现了也有人遇到这样的问题：[Feature request : apoc support for MERGE for nodes and rels #271](https://github.com/neo4j-contrib/neo4j-apoc-procedures/issues/271) 和 [Is it possible to merge using data-driven node or relationship labels?](https://stackoverflow.com/questions/43740000/is-it-possible-to-merge-using-data-driven-node-or-relationship-labels)。

原理跟单条数据插入一样，只是由于unwind是在服务端(neo4j)进行的，所以拼接也只能在服务端进行，怎么拼接的？就是用`apoc.cypher.doIt`拼接后让它在服务端执行：

```
UNWIND {batch} as row 
WITH 'MERGE (n:' + row.properties.label + ' { id: row.id }) SET n += row.properties return n' AS cypher
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

这样就可以正确的save数据并且动态设置label了。

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


### merge语气如何支持索引？

上面的cypher语句在功能上已经能够正常的work了。但是有一个非常严重的问题。就是merge本质上是先查询的过程，而neo4j的索引是跟label挂钩的。也就是说如果你在MERGE的匹配条件中如果没有指定label，那么即使id字段有索引，也不会走到的：

```
UNWIND {batch} as row 
MERGE (n { id: row.id }) SET n += row.properties 
WITH n 
CALL apoc.create.addLabels(id(n), [n.label]) YIELD node
RETURN node
```

上面这个cypher语句并不会走id索引。


这就又回到了前面一直纠结的问题，就是如何动态的设置label，就是一直拼接不了，所以才分成两步了：先设置属性，再设置label。但是不走索引在数据量大的时候基本不可接受，所以没办法，还是得推翻前面的方案重新探索一下。

经过不断的探索，最后终于搞定了，累觉不爱，肉瘤满面。。。


```
neo4j> :param batch: [{properties: {name: "argan", label: "Person", id: "1", age: 31}}, {properties: {name: "magi", label: "Person", id: "2", age: 28}}]
neo4j> :params
batch: [{properties={name=argan, label=Person, id=1, age=31}}, {properties={name=magi, label=Person, id=2, age=28}}]
neo4j> UNWIND {batch} as row
       CALL apoc.cypher.doIt('MERGE (n:`' + row.properties.label + '` {id: {id}}) SET n += {properties}', {properties: row.properties, id: row.properties.id}) YIELD value
       RETURN value.n
       ;
0 rows available after 3 ms, consumed after another 0 ms
neo4j> match (n) return n;
+--------------------------------------------------------------+
| n                                                            |
+--------------------------------------------------------------+
| (:Person {name: "argan", id: "1", label: "Person", age: 31}) |
| (:Person {name: "magi", id: "2", label: "Person", age: 28})  |
+--------------------------------------------------------------+

2 rows available after 1 ms, consumed after another 1 ms
```

其实就是下面的语句：

```
UNWIND {batch} as row 
WITH row, 'MERGE (n:`' + row.properties.label + '` {id: {id}}) SET n += {properties}' AS cypher
CALL apoc.cypher.doIt(cypher, {properties: row.properties, id: row.properties.id}) YIELD value 
RETURN value.n
;
```

或者简单点：

```
UNWIND {batch} as row
CALL apoc.cypher.doIt('MERGE (n:`' + row.properties.label + '` {id: {id}}) SET n += {properties}', {properties: row.properties, id: row.properties.id}) YIELD value
RETURN value.n
;
```  

java代码如下：

```java
/**
 * neo4j的match和merge不支持动态label，真是蛋疼。。本来一个简单的语句变成一个复杂很多。。
 */
@Override
public void saveVertice(List<Vertex> vertice) {
    StringBuilder sb = new StringBuilder();
    sb.append(" UNWIND {batch} as row ") //
            .append(" CALL apoc.cypher.doIt(" //
                    + "'MERGE (n:`' + row.properties.label + '` {id: {id}}) SET n += {properties}', " //
                    + "{ properties: row.properties, id: row.properties.id }" //
                    + ") YIELD value") //
            .append(" RETURN 1 ");

    String statement = sb.toString();

    Map<String, Object> params = new HashMap<>();
    List<Map<String, Object>> batches = new ArrayList<>();
    for (Vertex v : vertice) {
        Map<String, Object> map = new HashMap<>();
        map.put("id", v.getId());
        map.put("properties", v.getProperties());
        batches.add(map);
    }
    params.put("batch", batches);

    cypher.query(statement, params, null);
}
```

简单总结一下，就是必须使用apoc.cypher.doIt进行服务端拼接。然后拼接的Cypher语句中使用到的变量必须通过变量绑定的方式传递进去，不能直接使用WITH的变量，因为是使用不到的。


### 关系的save实现

前面我们主要讨论了节点的save操作实现，那么关系呢？

与节点一样，关系在绝大多数情况下可以用merge保证只创建一次（[Understanding how MERGE works](https://neo4j.com/developer/kb/understanding-how-merge-works/)）：

```
MATCH (n), (m)
WHERE n.id = "argan" AND m.id = "magi"
CREATE (n)-[:KNOWS]->(m)
```

然后我们一样会遇到节点一样的问题——如何支持动态 label 以走索引？而且关系还多了一个relationship type。

假设我们现在有如下两个节点：

```
╒══════════════════════════════════════════════════════════════════════╕
│"n"                                                                   │
╞══════════════════════════════════════════════════════════════════════╡
│{"name":"argan","id":"Person/1","label":"Person","key":"1","age":30,"t│
│ags":["smart","handson","rich"]}                                      │
├──────────────────────────────────────────────────────────────────────┤
│{"name":"magi","id":"Person/2","label":"Person","key":"2","age":32,"ta│
│gs":["tall","rich","handson"]}                                        │
└──────────────────────────────────────────────────────────────────────┘
```
然后我们要给这两个节点之间增加一个Friend关系：

```
:param batch: [{properties: {from: "Person/1", to: "Person/2", label: "Friend", id: "1", since: "2009"}}]
```

那么根据上面的讨论，我们只能这么做：

```
UNWIND {batch} as row
WITH split(row.properties.from, '/')  AS fromInfo, split(row.properties.to, '/')  AS toInfo, row
CALL apoc.cypher.doIt('MATCH (from:`' + fromInfo[0] + '` {id: {fromId}}) MATCH (to:`' + toInfo[0] + '` {id: {toId}}) MERGE (from)-[r:`' +  row.properties.label + '` {id: {id}}]->(to) SET r += {properties}', {fromId: row.properties.from, toId: row.properties.to, properties: row.properties, id: row.properties.id}) YIELD value
return value
```

**TIPS**

1、在调试这种字符串拼接，变量替换的语句，最好的做法是先完成动态拼接，然后再一个个的变量替换。

java代码如下：

```
public void saveEdges(List<Edge> edges) {
    StringBuilder sb = new StringBuilder();

    sb.append("UNWIND {batch} as row ") //
            .append(" WITH split(row.properties.from, '/')  AS fromInfo, " //
                    + "split(row.properties.to, '/')  AS toInfo, row ") //
            .append(" CALL apoc.cypher.doIt(" //
                    + "'MATCH (from:`' + fromInfo[0] + '` {id: {fromId}})" //
                    + " MATCH (to:`' + toInfo[0] + '` {id: {toId}}) " //
                    + " MERGE (from)-[r:`' +  row.properties.label + '` {id: {id}}]->(to) " //
                    + " SET n += {properties}', " //
                    + "{ fromId: row.properties.from, toId: row.properties.to, " //
                    + " properties: row.properties, id: row.properties.id }" //
                    + ") YIELD value") //
            .append(" RETURN 1 ");

    String statement = sb.toString();

    Map<String, Object> params = new HashMap<>();
    List<Map<String, Object>> batches = new ArrayList<>();
    for (Edge e : edges) {
        Map<String, Object> map = new HashMap<>();
        map.put("id", e.getId());
        map.put("from", e.getFrom());
        map.put("to", e.getTo());
        map.put("properties", e.getProperties());
        batches.add(map);
    }
    params.put("batch", batches);

    cypher.query(statement, params, null);
}
```

还有一个问题，就是边的方向，这个也没办法进行变量替换了，只能做条件判断，好在边就三个方向，直接枚举判断就可以了。neo4j中没有条件判断语句，但是可以用 FOREACH trick实现。这个在前面 [有条件的创建数据（Conditional Data Creation）]() 有介绍过。不过非常麻烦，我还是建议直接创建两条单向关系简单直接。





