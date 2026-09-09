---
title: tinkerpop-gremlin学习笔记
layout: post
catalog: true
---


* Graph
* TraversalSource


添加节点和边：

	gremlin> graph = TinkerGraph.open()
	==>tinkergraph[vertices:0 edges:0]
	gremlin> v1 = graph.addVertex(id, 1, label, "person", "name", "marko", "age", 29)
	==>v[1]
	gremlin> v2 = graph.addVertex(id, 3, label, "software", "name", "lop", "lang", "java")
	==>v[3]
	gremlin> v1.addEdge("created", v2, id, 9, "weight", 0.4)
	==>e[9][1-created->3]

遍历：

	gremlin> graph = TinkerFactory.createModern()
	==>tinkergraph[vertices:6 edges:6]
	gremlin> g = graph.traversal()
	==>graphtraversalsource[tinkergraph[vertices:6 edges:6], standard]
	gremlin> g.V() //(1)
	==>v[1]
	==>v[2]
	==>v[3]
	==>v[4]
	==>v[5]
	==>v[6]
	gremlin> g.V(1) //(2)
	==>v[1]
	gremlin> g.V(1).values('name') //(3)
	==>marko
	gremlin> g.V(1).outE('knows') //(4)
	==>e[7][1-knows->2]
	==>e[8][1-knows->4]
	gremlin> g.V(1).outE('knows').inV().values('name') //(5)
	==>vadas
	==>josh
	gremlin> g.V(1).out('knows').values('name') //(6)
	==>vadas
	==>josh
	gremlin> g.V(1).out('knows').has('age', gt(30)).values('name') //(7)
	==>josh


**说明**

* id和label是TinkerPop的保留词，一般来说程序中要这么引用：T.id 和 T.label，T = org.apache.tinkerpop.gremlin.structure.T


参考资料
-------

1. [Getting Started with TinkerPop and Gremlin](https://academy.datastax.com/resources/getting-started-tinkerpop-and-gremlin)

