---
title: 如何监控线上应用的运行状态
layout: post
---


需求
----

通常我们对线上的运行状态有监控的需求。比如我们需要知道队列的大小和使用情况，缓存的命中率，链接池的大小和使用情况，每个请求的QPS、响应时间，等等。


解决方案
--------


一般的做法如下：

1. 对需要监控的点进行埋点做统计。可以使用AOP尽量做到对应用透明。
2. 统计结果通过某种方式导出。比如最常见的是通过log打印出来，再去查看log。


其实上面的步骤明显的将这个过程进行了抽象：

1. Measurement 
2. Exportor/Reportor

进一步具体化，Measurement主要有如下几个维度：

1. Gauges: A gauge is an instantaneous measurement of a value. 
2. Counters: A counter is just a gauge for an AtomicLong instance. You can increment or decrement its value.
3. Timers: A timer measures both the rate that a particular piece of code is called and the distribution of its duration.
4. Meters: A meter measures the rate of events over time (e.g., “requests per second”). 
5. Histograms: A histogram measures the statistical distribution of values in a stream of data.

谷歌的云监控预定义了一些metrics [Supported metrics](https://developers.google.com/cloud-monitoring/metrics):

* Google Compute Engine - 13 metrics
* Google Cloud SQL - 12 metrics
* Google Cloud Pub/Sub - 14 metrics

并且提供了Read API让开发者查询当前到30天内的metric data。并且可以根据labels进行查询过滤。

支持的metricType有 [metricDescriptors](https://developers.google.com/cloud-monitoring/v2beta1/metricDescriptors)：

* cumulative
* delta
* gauge

metric的valueType有：

* bool
* distribution
* double
* int64
* string


而Exportor或者Reportor则主要有如下几个方式：

1. Reporting Via JMX：实时报告
2. Reporting Via HTTP：实时报告
3. Reporing Via STDOUT：定期报告
4. Reporting Via CSV files：定期报告
5. Reporting Via SLF4J loggers：定期报告
6. Reporting Via Ganglia：定期报告
7. Reporting Via Graphite：定期报告
...


这个抽象基本体现在所有的开源监视框架中：

1. [Yammer Codehale Metrics](http://metrics.codahale.com/) Cassandra和Spring Boot等使用。
2. [Servo - Netflix Application Monitoring Library](https://github.com/Netflix/servo) Netfilx内部使用，不是很成熟，功能也比较简单。
3. [StatsD](https://github.com/etsy/statsd/) Node.js的Metrics库。


差异仅仅在于怎样把这两个步骤做到方便、透明、稳定和高性能而已。


**TIPS** 使用BTrace监控线上运行情况

可以使用使用BTrace脚本监控线上运行情况，埋点逻辑完全脱离应用，应用完全无感知。缺点是BTrace有很多限制，而且写起来很痛苦。只适合于一些临时性的监视任务。不过他的完全透明性可以参考实现。


参考文档
--------

1. [Metrics library](https://github.com/springside/springside4/wiki/Metrics-Library)
2. [Metrics](http://metrics.codahale.com/)
3. [metrics-spring](https://github.com/ryantenney/metrics-spring) 支持Spring XML里定义Reporter，以及用注解埋点，用AOP生成相应的Metric。
4. [metrics-reporter-config](https://github.com/addthis/metrics-reporter-config) Cassandra的一个辅助项目，支持用Yaml定义Reporter。
5. [StatsD](https://github.com/etsy/statsd/) Node.js的Metrics库。
6. [Servo - Netflix Application Monitoring Library](https://github.com/Netflix/servo) Netflix自己的一个metrics库。
7. [Jmx](https://github.com/springside/springside4/wiki/JMX) 对JMX的一个简单介绍
8. [Measure Anything, Measure Everything](http://codeascraft.etsy.com/2011/02/15/measure-anything-measure-everything/)
9. [Counting & Timing](http://code.flickr.net/2008/10/27/counting-timing/)
10. [The State of Open Source Monitoring: The Good, The Bad, The Fucking Terrible and a Glimpse Into Our Future](https://speakerd.s3.amazonaws.com/presentations/506ed72c4f4e12000207cc4c/devopsdays-italy.pdf) 关于监控非常全面的介绍，强烈推荐！
11. [Google Cloud Monitoring API](https://developers.google.com/cloud-monitoring/what-is-cloud-monitoring) 谷歌的云监控API介绍。核心概念也是基于Metrics的Time series采样记录。
12. [The Twitter stack](http://blog.oskarsson.nu/post/40196324612/the-twitter-stack) Twitter的技术栈介绍，其中有比较详细的监控系统介绍。
13. [ostrich](https://github.com/twitter/ostrich) Twitter的监控库