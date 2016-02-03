---
title: java服务端监控平台设计
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


性能监控平台

1. 提供一站式的性能数据收集、计算、存储和展示服务
2. 支持自定义的数据指标名称和数据纬度
3. 提供任意指标任意纬度的实时数据查询

监控对象
-------

1、模块调用监控（URI监控）

* URI
* 调用总数
* 最大并发
* 总耗时
* 平均耗时
* 最快
* 最慢
* 错误数
* URIProfile（每个URL访问的具体调用信息，即使Tracer功能）
	* URI请求方法
	* 类型：Service/URL/DAO/IBATIS
	* 总数
	* 总耗时
	* 平均耗时
	* 耗时
	* 错误数

2、Spring监控

* 类 
* 方法
* 调用次数
* 总耗时
* 平均耗时
* 最大并发
* 最慢
* 错误次数


3、数据源监控

* 链接池中链接数
* 链接池链接数峰值
* 池中连接数峰值时间
* 活跃连接数
* 活跃连接数峰值

4、JDBC访问统计监控

* SQL语句
* 执行数
* 执行时间
* 错误数
* 读取行数
* 更行行数

5、Exception监控

* 异常类型：java.io.IOException, etc.
* 异常方法：
* URI
* 产生时间
* 异常数量
* 堆栈信息

6、JVM监控

* Overview
* Threading
* GC
* CPU
* Heap

7、其他信息

* 业务自定义信息，比如订单数量，支持成功数，点击次数，下载次数，等。
* Cache命中率
* 队列大小
* ...


监控方案设计
----------

### 监控方案决策

#### 1、每个应用自监控或者统一上报监控？

应用自监控，就是每个应用实例的监控数据存放在应用本身，比如一个Map。然后通过JMX或者其他方式暴露出去。然后开发人员可以通过JConsole或者API（一般是Web界面）得到这些监控数据。比如Druid就是这种做法。访问: hk01-xxxx-mob03.hk01:8090/druid/index.html 得到hk01-xxxx-mob03.hk01:8090这个应用的监控数据。

而统一上报监控方式，就是所有的应用监控数据都上报到监控中心，由监控中心负责接收、分析、合并、存储、可视化查询、报警等逻辑。这种方式是瘦客户端模型，客户端的职责就是埋点上报监控数据。所有的监控逻辑都在中心处理。

**结论**

自监控的话实现起来简单，并且没有与监控中心的网络交互，性能也会好很多。但是缺点就是缺乏全局的统计和监控。从实用角度来说还是集中式监控好一些。


#### 2、如何避免key冲突？如何区分各个应用实例？

为了监控简单，我们希望监控项是不需要预定义的，监控项是一个 `key => value` 的形式。其中key是监控项的唯一ID，而value可以为数值类型（比如counter, timeInterval），文本类型（如exceptionMessage）。
如果不预定义监控项，那么就是由客户端按需创建key，然后上报 监控项， 服务器检测如果改监控项不存在就创建，否则根据监控项类型进行相应的操作（叠加 for counter，计算平均值 for timer等）。
这个特性很方便客户端监控自动化，但是这样也带来两个可能的问题：

1. 不同的应用，有可能上报一样的key，这样会导致监控项冲突。
2. 想要查看相同的应用的不同实例的上报情况。这种情形主要发生在查找集群短板的时候。Dragoon的监控上也有实例筛选项。


但是最理想的情况是我们既希望能够合并统计,又希望能够在需要的时候区分查看。比如我们希望统计NanTianMen这个应用的所有实例的监控数据,同时又希望能够单独查看每个实例的监控数据。Google和OpenTSDB提供了一种解决方案——对metrics打tags。这样相同key的 metrics会合并统计,又可以根据tags进行区分。对于上面的例子,假如上报的metric含有一 个host=xxx的tag和一个port=xxx的tag就可以区分出来了。但是这种情况会导致key对应的 数据特别多。根据tag过来会影响查询速度。所以需要trade off。

**结论** 

对于key冲突，可以强制每个应用的客户端必须分配一个独立的appName/projectName	`作为前缀。这个是合理的要求，这个appName也有利于区分应用各自的监控。如果出于安全考虑，不同应用还应该有appKey。
对于同一个应用不同实例的区分，可以在上报接口增加上报来源作为tag。可以让应用传递参数，也可以自动根据ip来。比如Google和OpenTSDB就是通过对metrics打tags来解决这个问题。这样相同key的 metrics会合并统计,又可以根据tags进行区分。对于上面的例子,假如上报的metric含有一个host=xxx的tag。但是这种情况会导致key对应的数据特别多。根据tag过来会影响查询速度。所以需要tradeoff。比如OpenTSDB就是支持并且要求必须有一个tag，比如host=webserver01。

#### 3、监控中心与客户端应用之间要不要通过本地Agent上报？

采用集中式监控中心，意味着客户端与监控中心有交互。很多监控平台，比如阿里的Dragoon、新浪微博的Watchman，[Stackify](http://docs.stackify.com/m/7787/l/242597-getting-started-with-app-monitoring)都是有个本地agent的概念。Agent是OPS安装系统的时候预先安装好，每台机器一个Agent，负责该机器的所有监控数据上报。相当于应用与监控中心之间的一个通讯网关。应用通过JMX获取采集的数据，然后将数据上报给Agent，Agent再统一上报给Monitor。

这样的好处就是Client上报速度非常的快，而且基本不会失败。另外，同一机器上的多个client可以共用一个Agent通讯。而且Agent往往还承当了一个角色，就是主动收集机器监控信息（拉的方式）。缺点是需要预先按照Agent。所有也有很多监控平台是不走Agent的，直接client上报监控中心的方式。比如腾讯的ITIL和模块调用监控、Etsy的StatsD、Google的Cloud Monitor。大部分处于性能的考虑都是走UDP协议的，Google估计是因为是开发平台，走的是HTTP协议(Thus TCP协议)。这种方式简化了对客户端的预设要求和监控逻辑，实现起来比较简单。

**结论**

如果客户端与监控中心网络顺畅的情况下，绕开agent会简单很多。如果跨机房上报，那么异步化可能是很有必要的。采用agent是一个不错的方案。


#### 4、存储最终状态还是事件序列

比如监控一个URL的请求数，每次+1，最终我们能够得到请求总数。这样的好处是节省存储空间和计算时间。但是由于只有一个最终状态，我们没有办法得到在什么时间段请求数最多。于是有另一种记录方式：对于每次请求都记录一次，而不是简单的+1。然后我们根据所有的签到记录，就可以统计出总请求数，和分布状况。但是缺点也很显然，就是浪费存储，并且每次都需要执行统计计算。

**结论**

最终状态还是弱了一些，事件序列会好一些，存储可以采用HBase这样的分布式存储系统，性能问题可以采用预聚合等方式解决。[Google Cloud Monitor](Introduction to the Cloud Monitoring API)就是采用这个这种方式的：

> The Google Cloud Monitoring API lets you access monitoring data for Google Cloud services. The data is organized as metrics and stored as data points that represent information at a specific time or over a specific time period. Examples include the current CPU utilization of your virtual machine, the number of requests received by you web server, or custom metrics you define yourself. A list of data points measured at successive times is called a time series.

不过对于Counter类型的统计，确实可以考虑只是存储最终状态的。因为这种类型的metric，一般要的就是快速得到最终的状态，并且可能会有相应的报警策略。如果每次都要汇总，性能上往往不可接受。

#### 5、数据模型

数据模型非常重要，它决定了监控系统的能力。比如我们为什么不使用NOAH，其中一个原因就是NOAH的监控项只是简单的key-value形式。当然，它会自动记录请求源IP。但是其他的参数，比如应用等，就没有办法上报存储了。

根据上面的描述，其实我们的metrics基本就是抽象为带tags/labels标签的key-value格式。这个也是[Google Cloud Monitor](https://cloud.google.com/monitoring/api/metrics)和[OpenTSDB](http://opentsdb.net/docs/build/html/user_guide/writing.html)对metrics的定义:

* key 
* timestamp
* value - 这个OpenTSDB支持数值型的:integer和floating point。而Google Cloud Monitor支持的类型要丰富一些,见下面描述。
* tag(s) - A key/value pair consisting of a tagk (the key) and a tagv (the value). OpenTSDB要求至少要有一个tag。

Google Cloud Mnoitor对Metric进行分类,支持的metricType有(@see [metric-types](https://cloud.google.com/monitoring/api/metrics#metric-types)):

* cumulative: The value is a total, accumulated since a given start time. For example, the total number of errors detected since a process started.
* delta: The value is a change over a specified time period. For example, the number of errors detected in a minute.
* gauge: The value is an instantaneous sample of a continuously-varying metric at a specific time. For example, a CPU's current temperature.

而metric的valueType有:

* bool: A Boolean value, either "true" or "false".
* distribution: A distribution, consisting of a list of buckets and optionally an underflow bucket and an overflow bucket. Each bucket has an upper bound, a lower bound, and a count. The distribution can be used to create a histogram. 
* double: A double-precision floating-point value.
* int64: An integer value in the range [-263..263-1].
* string: A Unicode string with backslash escaping.


#### 6、 数据存储

因为Events或者Metrics的特殊性，一般都会采用一种专门的存储结构——Distributed time series database。比较有名的开源产品有如下这些:

1. RRD(round-robin-database): RRDtool使用的底层存储。C语言编写的。性能比较高
2. whisper: Graphite底层的存储,Python写的
3. [prometheus](http://prometheus.io/): An open-source service monitoring system and time series database. 目前只有单机版本。
4. [InfluxDB](http://influxdb.com/): 开源distributed time series, metrics, and events database。Go语言编写, 不依赖于其他外部服务。底层支持多种存储引擎，目前是LevelDB, RocksDB, HyberLevelDB和LMDB(0.9之后只支持Bolt，最新版本采用了自己写的存储引擎)。
5. [OpenTSDB](http://opentsdb.net/index.html): 基于HBase编写的Time Series Database
6. [kairosdb](http://kairosdb.github.io/): OpenTSDB的改善版，底层存储引擎是Cassandra。
7. [Heroic](http://spotify.github.io/heroic/#!/index): Kairosdb的改善版，Spotify公司开源的时序数据库（[Spotify的监控框架](http://www.infoq.com/cn/articles/spotify-monitoring-framework-part-02)），引入了ElasticSearch作为元数据索引。目前还处于不稳定状态。

具体可以参考这篇论文: [tsdb: A Compressed Database for Time Series](http://luca.ntop.org/tsdb.pdf)。

**结论**

如果要存储事件序列，那么InfluexDB和OpenTSDB是个非常不错的选择。都是可扩展，分布式存储，文档很详细，还是开源的。
[influexDB 0.9.0](http://influxdb.com/blog/2014/12/08/clustering_tags_and_enhancements_in_0_9_0.html)之后支持tag，使用风格跟Google Cloud Monitor很相似，而且支持String类型。并且最重要的是不需要额外搭建HBase(Thus Hadoop & Zookeeper)，看起来非常值得期待，不过笔者曾经试过0.9.6版本的InfluxDB用来存储我们的接口响应时间，结果根本撑不住（[InfluxDB becomes unavailable after heavy insert load](https://github.com/influxdata/influxdb/issues/5149)）个人觉得这个产品还是太年轻，还没有到产品级别。OpenTSDBvalue不支持String类型，这意味着日志不能上报到OpenTSDB，需要另外处理。

由于这个比较复杂而且非常重要，我们在后面再单独详细讨论。


#### 7、如果服务器挂掉了，统计数据怎么处理？缓存本地，等服务器起来再发送？还是丢弃？

前期可以先丢弃，后续要缓存起来。受影响比较大的是counter接口。

存储的话，可以考虑使用本地存储在RRD文件或者BDB中，或者消息队列中(RabbitMQ, ie.)，最后再异步批量上报给中心的TSDB。

	timestamp	metrics 	value 	tags..
	1366399993 mysql.Binlog_cache_disk_use 0 host=mydb.example.com
	1366399993 mysql.Bytes_received 19453687 host=mydb.example.com
	1366399993 mysql.Bytes_sent 1238166682 host=mydb.example.com


#### 8、网络通信和协议

如何高性能的接收大量客户端的上报请求。以及使用什么通讯协议。

有几种选择：

* HTTP
* TCP
* UDP: fire and forget, 主要需要注意MTU问题。

同时要考虑同步和异步接口。

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


**上报API接口**

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


大概是这样子的使用方式：

	import me.arganzheng.study.monitor.*;

	Agent agent = new Agent("yourAppName");

	agent.increment("myapp.login");
	agent.gauge("heap_free", 8675309);
	agent.time("some.longProcess", new Runnable() {
		public void run() {
	        // Do something....
	    });
	agent.addMetric("Maintenance Now.", 600);

可以考虑使用注解简化客户端上报逻辑[newrelic](https://docs.newrelic.com/docs/agents/java-agent/custom-instrumentation/java-instrumentation-annotation)：

	@Trace(metricName=”YouMetricName”)

By default, the metric name will include the class name followed by the method name


时序数据库讨论
------------

这里我们以两大开源的时序数据库：influxDB和OpenTSDB做对比讨论。


就文档看起来，influexDB使用起来更像传统的RDB。需要创建DB，但是不需要schema，columns是动态创建的。感觉columns就是OpenTSDB的tags键值对。

InfluxDB的抽象更类似于传统的关系型数据库，只是schemeless：Database, shard space, series(table), column。

写入格式：

* OpenTSDB: <metric> <timestamp> <value> <tagk1=tagv1[ tagk2=tagv2 ...tagkN=tagvN]>
* influxDB: name [columns] [points]。其中timestamp由服务端生成。columns和points类似于SQL的insert columns values(..)语法。


例如：统计mysql.Bytes_received，

OpenTSDB是这样子：

	1385327470774 mysql.Bytes_received 19453687 host=mydb.example.com app=mysql

HTTP格式是：

	{
        "metric": "mysql.Bytes_received",
        "timestamp": 1385327470774,
        "value": 19453687,
        "tags": {
           "hostName": "mydb.example.com",
           "app": "mysql"
        }
    },

influxDB则是：

	[
	  {
	    "name" : "mysql.Bytes_received",
	    "columns" : ["app", "value", "host"],
	    "points" : [
	      ["mysql",19453687, "mydb.example.com"]
	    ]
	  }
	]

0.9之后支持tags:

	{
	    "database": "mydb",
	    "retentionPolicy": "default",
	    "points": [
	        {
	            "name": "Bytes_received",
	            "tags": {
	                "host": "mydb.example.com",
	                "app": "mysql",
	                "region": "us-west"
	            },
	            "time": "2009-11-10T23:00:00Z",
	            "fields": {
	                "value": 19453687
	            }
	        }
	    ]
	}

需要注意的是influxDB的tags是默认索引的，但是fields(columns)则是没有索引的。也就是说我们无法高效的执行：响应时间(value) > 1000ms的记录。

另外值得注意的是influxDB的value值可以是String类型，这个OpenTSDB目前是不支持的。这意味着我们可以将错误日志也放在influxDB中。


可视化组件
--------

如果我们采用了一站式的监控平台，像 [Relic](http://newrelic.com/)，[moskito](http://www.moskito.org/)，[prometheus](http://prometheus.io/)，或者[graphite](http://graphite.wikidot.com/)（严格来说，Graphite其实只是包含存储和可视化展示，并没有包含收集），那么你就不太需要关心可视化的事情（那可是相当烦人的，特别是对于一个后端开发工程师来说）。但是如果采用了OpenTSDB或者influxDB，那么其实它们只是解决了数据存储而已。数据收集和数据展示这块还是需要另外的组件来解决。有需求就有产品。在监控可视化这块，[Grafana](http://grafana.org/)貌似是唯一的选择。而且默认支持Graphite, InfluxDB & OpenTSDB。节目风格看起来非常像kibana，试试上就是在kibana的基础上二次开发的，原来是为了Graphite创建的。


这里有一篇文章介绍influxDB和grafana整合的，非常详细，可以参考一下：[OBIEE Monitoring and Diagnostics with InfluxDB and Grafana](http://www.rittmanmead.com/2015/02/obiee-monitoring-and-diagnostics-with-influxdb-and-grafana/)。


日志事件收集组件
--------------

* [fluentd](http://www.fluentd.org/)
* [logstash](https://www.elastic.co/products/logstash)
* [collectd](https://collectd.org/) C写的一个系统参数收集

参考文章
-------

1. [Application Monitoring](http://newrelic.com/application-monitoring/features) 收费产品，功能很强大。
2. [moskito](http://www.moskito.org/) 开源监控产品，思路跟我的挺match的。
3. [OpenTSDB2.0](http://www.slideshare.net/HBaseCon/ecosystem-session-6?related=1) 非常好的PPT
4. [influxDB](http://influxdb.com/)
5. [Grafana](http://grafana.org/)
6. [Relic](http://newrelic.com/)
7. [kairosdb](http://kairosdb.github.io/): OpenTSDB的改善版，底层存储引擎是Cassandra。
8. [Heroic](http://spotify.github.io/heroic/#!/index): Kairosdb的改善版，Spotify公司开源的时序数据库
9. [Spotify的监控框架](http://www.infoq.com/cn/articles/spotify-monitoring-framework-part-02)
10. [Advanced Time Series with Cassandra](http://www.datastax.com/dev/blog/advanced-time-series-with-cassandra)
