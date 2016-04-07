---
title: 大数据平台学习笔记
layout: post
---


* 数据接入
	* 主要负责数据的收集，分发，预处理以及管理工作
	* Flume, Kafka
* 离线计算平台
	* Hadoop
	* 批处理
	* 三种任务(Job)
		* Map-Reduce Job
		* Hive Job
		* Pig Job
* 实时计算平台
	* Storm(JStorm), Spark, Flink
	* 在线流式计算
* 资源管理和调度
	* YARN, Mesos
* 任务调度系统
	* Ooize
	* azkaban
* 数据存储引擎
	* HDFS，HBase
	* Redis
	* MySQL, PG
	* Tachyon
* 查询语言
	* Hive, Pig, Shark
* 计算模型
	* Map-Reduce
	* Spout-Bolt
	* Streaming

**TIPS**

1. Storm和Flink都是是实时流式数据处理，面向行处理，单条延时比较低。Spark是近实时流式处理，面向RDD处理，吞吐量比较高。如果应用对实时性要求比较高建议试用Storm或者Flink, 否则大家可以考虑利用Spark的丰富的数据操作能力。 


参考文章
------

1. [与 Hadoop 对比，如何看待 Spark 技术？](https://www.zhihu.com/question/26568496)
