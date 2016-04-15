---
title: 大数据平台学习笔记
layout: post
---


* 数据接入
	* 主要负责数据的收集，分发，预处理以及管理工作
	* Scribe/Flume, Kafka
* 离线计算平台
	* Hadoop
	* 批处理
	* 三种任务(Job)
		* Map-Reduce Job
		* Hive Job
		* Pig Job
* 实时计算平台
	* Storm(JStorm), Samza, Spark, Flink
	* 在线流式计算
* 资源管理和调度
	* YARN, Mesos
* 任务调度系统
	* Ooize
	* azkaban
	* YARN
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
* 数据管理可视化平台
	* [Zeppelin](http://zeppelin.incubator.apache.org/) 孵化项目，美团使用了。

**TIPS && NOTES**

1. MapReduce计算模型对多轮迭代的DAG作业支持不给力，每轮迭代都需要将数据落盘，极大地影响了作业执行效率，另外只提供Map和Reduce这两种计算因子，使得用户在实现迭代式计算（比如：机器学习算法）时成本高且效率低。
2. Storm和Flink都是是实时流式数据处理，面向行处理，单条延时比较低。Spark是近实时流式处理，面向RDD处理，吞吐量比较高。如果应用对实时性要求比较高建议试用Storm或者Flink, 否则大家可以考虑利用Spark的丰富的数据操作能力。 


参考文章
------

1. [与 Hadoop 对比，如何看待 Spark 技术？](https://www.zhihu.com/question/26568496)
2. [Spark在美团的实践](http://tech.meituan.com/spark-in-meituan.html?from=singlemessage&isappinstalled=0)
