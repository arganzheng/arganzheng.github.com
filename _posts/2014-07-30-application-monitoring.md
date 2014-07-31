---
title: 服务端监控方案
layout: post
---


需求
----

1. 功能性监控(availability)：我们希望能够及时的发现线上的bug，异常。
2. 性能监控与调优(performance)：我们希望能够识别可能的性能瓶颈，并且有数据支撑方便对性能进行调优。比如慢速SQL，缓存命中率，链接池等。
3. 预警报警：我们希望能够第一时间收到线上异常，进行处理。
4. 辅助线上故障定位、故障分析
5. 运维支持：我们系统能够有数据支撑我们进行容量规划、报警和扩容，等。
6. 自动化运维：比如自动扩容；或者根据监控的数据和配置的SLA进行服务降级。等等。


监控方案
--------

### 监控对象

* 操作系统：OP
* 网络监控：OP
* 应用监控：RD
	* URI
	* Spring
	* JDBC
	* JVM
* 安全监控：OP
* JS监控：OP
* 基调监控：OP


其中应用监控是归属我们RD的职责范畴的。也是我们这次讨论的重点。

具体的监控信息包括：

**Spring监控**

* 类 
* 方法
* 调用次数
* 总耗时
* 平均耗时
* 最大并发
* 最慢
* 错误次数

**URI调用统计监控**

* URI
* 调用总数
* 最大并发
* 总耗时
* 平均耗时
* 最慢
* 错误数
* URIProfile（每个URL访问的具体调用信息）
	* URI请求方法
	* 类型：Service/URL/DAO/IBATIS
	* 总数
	* 总耗时
	* 平均耗时
	* 耗时
	* 错误数

**数据源监控**

* 链接池中链接数
* 链接池链接数峰值
* 池中连接数峰值时间
* 活跃连接数
* 活跃连接数峰值

**JDBC访问统计监控**

* SQL语句
* 执行数
* 执行时间
* 错误数
* 读取行数
* 更行行数

**Exception监控**

* 异常类型：java.io.IOException, etc.
* 异常方法：
* URI
* 产生时间
* 异常数量
* 堆栈信息

**JVM监控**

* Overview
* Threading
* GC
* CPU
* Heap

**其他信息**

* 业务自定义信息


### 监控方案决策

1、每个应用自监控或者统一上报监控？

应用自监控，就是每个应用实例的监控数据存放在应用本身，比如一个Map。然后通过JMX或者其他方式暴露出去。然后开发人员可以通过JConsole或者API（一般是Web界面）得到这些监控数据。比如Druid就是这种做法。访问: hk01-hao123-mob03.hk01.baidu.com:8090/druid/index.html 得到hk01-hao123-mob03.hk01.baidu.com:8090这个应用的监控数据。


而统一上报监控方式，就是所有的应用监控数据都上报到监控中心，由监控中心负责接收、分析、合并、存储、可视化查询、报警等逻辑。这种方式是瘦客户端模型，客户端的职责就是埋点上报监控数据。所有的监控逻辑都在中心处理。


2、监控中心与客户端应用之间要不要通过本地Agent？

很多监控平台，比如阿里的Dragoon、新浪微博的Watchman，都是有个本地agent的概念。Agent是OPS安装系统的时候预先安装好，每台机器一个Agent，负责该机器的所有监控数据上报。相当于应用与监控中心之间的一个通讯网关。应用通过JMX获取采集的数据，然后将数据上报给Agent，Agent再统一上报给Monitor。

这样的好处就是Client上报速度非常的快，而且基本不会失败。另外，同一机器上的多个client可以共用一个Agent通讯。缺点是需要预先按照Agent。所有也有很多监控平台是不走Agent的，直接client上报监控中心的方式。比如腾讯的ITIL和模块调用监控、Etsy的StatsD、Google的Cloud Monitor。大部分处于性能的考虑都是走UDP协议的，Google估计是因为是开发平台，走的是HTTP协议(Thus TCP协议)。这种方式简化了对客户端的预设要求和监控逻辑，实现起来比较简单。


### 应用监控平台概要设计

初步决定采用基于metrics上报的中心监控(无Agent)模式。

#### 业务监控流程

1. 业务对需要监控的地方埋点监控逻辑
2. 监控统计数据通过某种方式上报到监控中心（或者监控中心通过某种方式采集业务监控数据）
3. 监控中心对监控数据提供可视化查询界面，方便查看监控结果
4. 如果监控结果满足配置的报警条件，会自动通知相关的负责人进行处理


#### 监控系统模块

1、Client

主要职责是提供便利的方式让用户添加监控项。包括如下几个模块：

1. Metrics 监控项：counter, timer, etc.
2. AOP拦截配置或者注解方便业务埋点(提供缺省的采集实现，业务通过配置开启相应的监控项)
3. 监控数据上报客户端(Reporter)
4. 当监控中心挂掉的时候，将消息先存储在本地(BDB?)

2、监控中心(MonitorCenter)

监控中心应该提供接收客户端监控统计数据的上报接口。接收数据包，并且对这些数据进行存储，分析和可视化。
可抽象为一个事件状态机，接收客户端发送的事件，对事件进行响应。主要包含如下模块：

1. 上报API接口服务(事件接收器，receive packets, UDP is prefered)
2. 事件处理器
	* EventHandlers, Pipeline模式
	* 内建的EventHandler: metrics(increment counters, timer, etc.)、Storage(periodically save the metrics to disk)、Analizer、Notifier
3. 缓存和存储：对事件进行存储(需要考虑性能和容量)	
4. 定时任务处理器：Triggers, Actions, Scheduler
5. 可视化界面(dashboards)：Visualizer
6. 配置管理界面，配置事件相应的负责人 && 事件处理工作流程。


**上报API接口服务**

1. Counter接口: A counter is a value that never decreases.

		void increment(String key);
		void increment(String key, Integer delta);

2. Gauges接口：A gauge is a value that has a discrete value at any given moment, like "heap_used" or "current_temperature".

		void addGauge(String key, Double value);

3. Metrics接口：A metric is tracked via distribution, and is usually used for timings. Metrics are collected by tracking the count, min, max, mean (average), and a simple bucket-based histogram of the distribution. This distribution can be used to determine median, 90th percentile, etc.

		void addMetric(String key, T value);

	其中针对时间的监控可以提供一个便利函数：

		void addTimeMetric(String key, long timeInMillis);


4. 日志上报接口: A label is just a key/value pair of strings, usually used to report a subsystem's state, like "boiler=offline". 

		void log(LoggerLevel level, String key, String message);

They have no real statistical value, but can be used to raise flags in logging and monitoring. 增加一个日志级别，可以根据日志级别来做相应的action。

说明：由于监控数据并不是很重要，为了保证性能，采用UDP协议, fire and forgot。


### 关键问题

1、如何避免key冲突？如何区分各个应用实例？

为了监控简单，我们希望监控项是不需要预定义的，监控项是一个 `key => value` 的形式。其中key是监控项的唯一ID，而value可以为数值类型（比如counter, timeInterval），文本类型（如exceptionMessage）。
如果不预定义监控项，那么就是由客户端按需创建key，然后上报监控项，服务器检测如果该监控项不存在就创建，否则根据监控项类型进行相应的操作（叠加 for counter，计算平均值 for timer等）。
这个特性很方便客户端监控自动化，但是这样也带来两个可能的问题：

1. 不同的应用，有可能上报一样的key，这样会导致监控项冲突。解决方案可以强制每个应用的key使用packagename或者应用名称作为前缀。
2. 想要查看相同的应用的不同实例的上报情况。这种情形主要发生在查找集群短板的时候。Dragoon的监控上也有实例筛选项。解决方案是在上报接口增加上报来源作为namespace。可以让应用传递参数，也可以自动根据ip来。不过这样的话，每个应用实例就单独统计了。

但是最理想的情况是我们既希望能够合并统计，又希望能够在需要的时候区分查看。比如我们希望统计NanTianMen这个应用的所有实例的监控数据，同时又希望能够单独查看每个实例的监控数据。Google和[OpenTSDB](http://opentsdb.net/docs/build/html/user_guide/writing.html)提供了一种解决方案——对metrics打tags。这样相同key的metrics会合并统计，又可以根据tags进行区分。对于上面的例子，假如上报的metric含有一个host=xxx的tag和一个port=xxx的tag就可以区分出来了。但是这种情况会导致key对应的数据特别多。根据tag过来会影响查询速度。所以需要trade off。


2、Metrics的数据格式

[Google Cloud Monitor](https://developers.google.com/cloud-monitoring/metrics)和[OpenTSDB](http://opentsdb.net/docs/build/html/user_guide/writing.html)对metrics的定义比较相似——支持tags/labels标签的key-value格式：

* key 
* timestamp
* value - 这个OpenTSDB支持数值型的：integer和floating point。而Google Cloud Monitor支持的类型要丰富一些，见下面描述。
* tag(s) - A key/value pair consisting of a tagk (the key) and a tagv (the value). OpenTSDB要求至少要有一个tag。


Google Cloud Mnoitor对Metric进行分类，支持的metricType有 [metricDescriptors](https://developers.google.com/cloud-monitoring/v2beta1/metricDescriptors)：

* cumulative 不知道啥玩意
* delta
* gauge

metric的valueType有：

* bool
* distribution 不知道啥玩意
* double
* int64
* string


3、如果服务器挂掉了，统计数据怎么处理？缓存本地，等服务器起来再发送？还是丢弃？

前期可以先丢弃，后续可以先本地缓存起来，等中心起来再补充上报。受影响比较大的是counter接口。

4、数据存储

因为Events或者Metrics的特殊性。一般都会采用一种专门的存储结构——Distributed time series database。比较有名的开源产品有如下这些：

1. [RRD(round-robin-database)](http://oss.oetiker.ch/rrdtool/)。[RRDtool](http://oss.oetiker.ch/rrdtool/)使用的底层存储。C语言编写的。性能比较高。
2. [whisper](http://graphite.wikidot.com/whisper)。[Graphite](http://graphite.wikidot.com)底层的存储，Python写的。
3. [InfluxDB](http://influxdb.com/)。Go语言编写，不依赖于其他三方库。
4. [OpenTSDB](http://opentsdb.net/)。基于[HBase](http://hbase.org/)编写的Time Series Database。


5、网络通信

如何高性能的接收大量客户端的上报请求。

6、可视化

各种维度的查询和图表展示。


### 补充说明


1. 针对URI发起测试请求的监控方式，在公司的argus中有支持 [业务监控使用说明-http监控方式](http://devops.baidu.com/new/argus/patrol.md)。但是这种方式其实是属于自动化测试的范畴，而且虽然是自动化测试，其实是建立在前期的人肉配置和录制的基础上。一般属于上线之前的测试验证过程。所以这里不建议把这个放在我们这个监控平台中，建议是让测试同学去做这个事情。
2. 完善单元测试，可以减少功能性bug。建议后续逐步完善单元测试，并且搭建[Jenkins](http://www.jenkins-ci.org/)持续集成平台和[Sonar](http://www.sonarqube.org/)代码质量检查。
3. 鉴于线下的压力测试其实并不能很准确的评估线上的性能指标，上次跟小芳也讨论过这个问题，她说公司有一个分流测试工具，可以将线上的请求透明的分流到指定的压力测试机（我的理解是类似于linux下的tea命令）。可以预言一下。