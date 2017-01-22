---
title: Kafka实战
layout: post
---


背景
----

最近要把原来在BB做的那套集中式日志监控系统迁移到Mobojoy这边，原来的实现方案是: Log Agent => Log Server => ElasticSearch => Kibana，其中Log Agent和Log Server之间走的是Thrift RPC，自己实现了一个简单的负载均衡(WRB)。

原来的方案其实运行的挺好的，异步化Agent对应用性能基本没有影响。支持我们这个每天几千万PV的应用一点压力都没有。不够有个缺点就是如果错误日志暴增，Log Server这块处理不过来，会导致消息丢失。当然我们量级没有达到这个程度，而且也是可以通过引入队列缓冲一下处理。不过现在综合考虑，其实直接使用消息队列会更简单。PRC，负载均衡，负载缓冲都内建实现了。另一种方式是直接读取日志，类似于logstash或者flume的方式。不过考虑到灵活性还是决定使用消息队列的方式，反正我们已经部署了Zookeeper。调研了一下，Kafka是最适合做这个数据中转和缓冲的。于是，打算把方案改成: Log Agent => Kafa => ElasticSearch => Kibana。


Kafka介绍
---------

### Kafka基本概念

Broker：Kafka集群包含一个或多个服务器，这种服务器被称为broker。
Topic：每条发布到Kafka集群的消息都有一个类别，这个类别被称为Topic。
Partition：Partition是物理上的概念，每个Topic包含一个或多个Partition。
Producer：负责发布消息到Kafka broker。
Consumer：消息消费者，向Kafka broker读取消息的客户端。
Consumer Group：每个Consumer属于一个特定的Consumer Group（可为每个Consumer指定group name，若不指定group name则属于默认的group）。
Zookeeper: 无论是kafka集群，还是Producer和Consumer，都依赖于Zookeeper来保证系统可用性以及保存一些meta信息。

### Kafka的特点

1. 消息顺序：保证每个partition内部的顺序，但是不保证跨partition的全局顺序。如果需要全局消息有序，topic只能有一个partition。
2. consumer group：consumer group中的consumer并发获取消息，但是为了保证partition消息的顺序性，每个partition只会由一个consumer消费。因此consumer group中的consumer数量需要小于等于topic的partition个数。（如需全局消息有序，只能有一个partition，一个consumer）
3. 同一Topic的一条消息只能被同一个Consumer Group内的一个Consumer消费，但多个Consumer Group可同时消费这一消息。这是Kafka用来实现一个Topic消息的广播（发给所有的Consumer）和单播（发给某一个Consumer）的手段。一个Topic可以对应多个Consumer Group。如果需要实现广播，只要每个Consumer有一个独立的Group就可以了。要实现单播只要所有的Consumer在同一个Group里。
4. Producer Push消息，Client Pull消息模式： 一些logging-centric system，比如Facebook的Scribe和Cloudera的Flume，采用push模式。事实上，push模式和pull模式各有优劣。push模式很难适应消费速率不同的消费者，因为消息发送速率是由broker决定的。push模式的目标是尽可能以最快速度传递消息，但是这样很容易造成Consumer来不及处理消息，典型的表现就是拒绝服务以及网络拥塞。而pull模式则可以根据Consumer的消费能力以适当的速率消费消息。pull模式可简化broker的设计，Consumer可自主控制消费消息的速率，同时Consumer可以自己控制消费方式——即可批量消费也可逐条消费，同时还能选择不同的提交方式从而实现不同的传输语义。

实际上，Kafka的设计理念之一就是同时提供离线处理和实时处理。根据这一特性，可以使用Storm或Spark Streaming这种实时流处理系统对消息进行实时在线处理，同时使用Hadoop这种批处理系统进行离线处理，还可以同时将数据实时备份到另一个数据中心，只需要保证这三个操作所使用的Consumer属于不同的Consumer Group即可。


### kafka的HA

Kafka在0.8以前的版本中，并不提供High Availablity机制，一旦一个或多个Broker宕机，则宕机期间其上所有Partition都无法继续提供服务。若该Broker永远不能再恢复，亦或磁盘故障，则其上数据将丢失。而Kafka的设计目标之一即是提供数据持久化，同时对于分布式系统来说，尤其当集群规模上升到一定程度后，一台或者多台机器宕机的可能性大大提高，对Failover要求非常高。因此，Kafka从0.8开始提供High Availability机制。主要表现在Data Replication和Leader Election两方面。

#### Data Replication

Kafka从0.8开始提供partition级别的replication，replication的数量可在$KAFKA_HOME/config/server.properties 中配置:

	default.replication.factor = 1

该 Replication与leader election配合提供了自动的failover机制。replication对Kafka的吞吐率是有一定影响的，但极大的增强了可用性。默认情况下，Kafka的replication数量为1。 每个partition都有一个唯一的leader，所有的读写操作都在leader上完成，follower批量从leader上pull数据。一般情况下partition的数量大于等于broker的数量，并且所有partition的leader均匀分布在broker上。follower上的日志和其leader上的完全一样。

需要注意的是，replication factor并不会影响consumer的吞吐率测试，因为consumer只会从每个partition的leader读数据，而与replicaiton factor无关。同样，consumer吞吐率也与同步复制还是异步复制无关。

#### Leader Election

引入Replication之后，同一个Partition可能会有多个Replica，而这时需要在这些Replication之间选出一个Leader，Producer和Consumer只与这个Leader交互，其它Replica作为Follower从Leader中复制数据。注意，只有Leader负责数据读写，Follower只向Leader顺序Fetch数据（N条通路），并不提供任何读写服务，系统更加简单且高效。

**Propagate消息**

Producer在发布消息到某个Partition时，先通过ZooKeeper找到该Partition的Leader，然后无论该Topic的Replication Factor为多少（也即该Partition有多少个Replica），Producer只将该消息发送到该Partition的Leader。Leader会将该消息写入其本地Log。每个Follower都从Leader pull数据。这种方式上，Follower存储的数据顺序与Leader保持一致。Follower在收到该消息并写入其Log后，向Leader发送ACK。一旦Leader收到了ISR中的所有Replica的ACK，该消息就被认为已经commit了，Leader将增加HW并且向Producer发送ACK。

为了提高性能，每个Follower在接收到数据后就立马向Leader发送ACK，而非等到数据写入Log中。因此，对于已经commit的消息，Kafka只能保证它被存于多个Replica的内存中，而不能保证它们被持久化到磁盘中，也就不能完全保证异常发生后该条消息一定能被Consumer消费。但考虑到这种场景非常少见，可以认为这种方式在性能和数据持久化上做了一个比较好的平衡。在将来的版本中，Kafka会考虑提供更高的持久性。

Consumer读消息也是从Leader读取，只有被commit过的消息（offset低于HW的消息）才会暴露给Consumer。

Kafka Replication的数据流如下图所示：

![kafka-replication-dataflow](/media/images/kafka-replication-dataflow.png)

关于这方面的内容比较多而且复杂，这里就不展开了，这篇文章写的很好，有兴趣的同学可以学习 [Kafka设计解析（二）：Kafka High Availability （上）](http://www.infoq.com/cn/articles/kafka-analysis-part-2)。


### Kafka客户端

Kafka支持JVM语言(java、scala)，同是也提供了高性能的[C/C++客户端](https://github.com/edenhill/librdkafka)，和基于librdkafka封装的各种语言客户端。如，Python客户端: [confluent-kafka-python](https://github.com/confluentinc/confluent-kafka-python) 。Python客户端还有纯python实现的：[kafka-python](http://github.com/dpkp/kafka-python)。

下面是Python例子（以confluent-kafka-python为例）：

Producer:

	from confluent_kafka import Producer

	p = Producer({'bootstrap.servers': 'mybroker,mybroker2'})
	for data in some_data_source:
	    p.produce('mytopic', data.encode('utf-8'))
	p.flush()

Consumer:

	from confluent_kafka import Consumer, KafkaError

	c = Consumer({'bootstrap.servers': 'mybroker', 'group.id': 'mygroup',
	              'default.topic.config': {'auto.offset.reset': 'smallest'}})
	c.subscribe(['mytopic'])
	running = True
	while running:
	    msg = c.poll()
	    if not msg.error():
	        print('Received message: %s' % msg.value().decode('utf-8'))
	    elif msg.error().code() != KafkaError._PARTITION_EOF:
	        print(msg.error())
	        running = False
	c.close()

跟普通的消息队列使用基本是一样的。


### Kafka的offset管理

kafka读取消息其实是基于offset来进行的，如果offset出错，就可能出现重复读取消息或者跳过未读消息。在0.8.2之前，kafka是将offset保存在ZooKeeper中，但是我们知道zk的写操作是很昂贵的，而且不能线性拓展，频繁的写入zk会导致性能瓶颈。所以在0.8.2引入了[Offset Management](http://www.infoq.com/cn/articles/kafkaesque-days-at-linkedin-part01)，将这个offset保存在一个 compacted kafka topic(_consumer_offsets)，Consumer通过发送OffsetCommitRequest请求到指定broker（偏移量管理者）提交偏移量。这个请求中包含一系列分区以及在这些分区中的消费位置（偏移量）。偏移量管理者会追加键值（key－value）形式的消息到一个指定的topic（__consumer_offsets）。key是由consumerGroup-topic-partition组成的，而value是偏移量。同时为了提供性能，内存中也会维护一份最近的记录，这样在指定key的情况下能快速的给出OffsetFetchRequests而不用扫描全部偏移量topic日志。如果偏移量管理者因某种原因失败，新的broker将会成为偏移量管理者并且通过扫描偏移量topic来重新生成偏移量缓存。



部署和配置
--------

Kafka是用Scala写的，所以只要安装了JRE环境，运行非常简单。直接下载官方编译好的包，解压配置一下就可以直接运行了。

作为Boker必须配置如下几个配置项：

* broker.id
* log.dirs: 默认在tmp目录，很容易撑爆，建议修改。
* zookeeper.connect： Zookeeper所在地址，不需要配置多个。

然后启动：`$ bin/kafka-server-start.sh config/server.properties &`，非常简单。

不过不像RabbitMQ，或者ActiveMQ，Kafka默认并没有web管理界面，只有命令行语句，不是很方便，不过可以安装一个，比如，Yahoo的
[Kafka Manager](https://github.com/yahoo/kafka-manager): A tool for managing Apache Kafka。它支持很多功能：

* Manage multiple clusters
* Easy inspection of cluster state (topics, consumers, offsets, brokers, replica distribution, partition distribution)
* Run preferred replica election
* Generate partition assignments with option to select brokers to use
* Run reassignment of partition (based on generated assignments)
* Create a topic with optional topic configs (0.8.1.1 has different configs than 0.8.2+)
* Delete topic (only supported on 0.8.2+ and remember set delete.topic.enable=true in broker config)
* Topic list now indicates topics marked for deletion (only supported on 0.8.2+)
* Batch generate partition assignments for multiple topics with option to select brokers to use
* Batch run reassignment of partition for multiple topics
* Add partitions to existing topic
* Update config for existing topic
* Optionally enable JMX polling for broker level and topic level metrics.
* Optionally filter out consumers that do not have ids/ owners/ & offsets/ directories in zookeeper.

安装过程蛮简单的，就是要下载很多东东，会很久。具体参见: [kafka manager安装](https://hengyunabc.github.io/kafka-manager-install/)。不过这些管理平台都没有权限管理功能。

需要注意的是，Kafka Manager的`conf/application.conf`配置文件里面配置的`kafka-manager.zkhosts`是为了它自身的高可用，而不是指向要管理的Kafka集群指向的zkhosts。所以不要忘记了手动配置要管理的Kafka集群信息（主要是配置名称，和zk地址）。[Install and Evaluation of Yahoo's Kafka Manager](http://edbaker.weebly.com/blog/install-and-evaluation-of-yahoos-kafka-manager)。

Kafka Manager主要是提供管理界面，监控的话还要依赖于其他的应用，比如：

1. [Burrow](https://github.com/linkedin/Burrow): Kafka Consumer Lag Checking. Linkedin开源的cusumer log监控，go语言编写，貌似没有界面，只有HTTP API，可以配置邮件报警。
2. [Kafka Offset Monitor](https://quantifind.com/KafkaOffsetMonitor/): A little app to monitor the progress of kafka consumers and their lag wrt the queue.

这两个应用的目的都是监控Kafka的offset。

**TIPS**

生产环境最好配置一下下面两个选项：

* delete.topic.enable=true，默认是false，0.8.2之后支持
* auto.create.topics.enable=false，默认是true


[2016-04-13 20:51:39,700] INFO Registered broker 1 at path /brokers/ids/1 with addresses: PLAINTEXT -> EndPoint(0.0.0.0,8092,PLAINTEXT) (kafka.utils.ZkUtils)


### 过期数据自动清除

对于传统的message queue而言，一般会删除已经被消费的消息，而Kafka集群会保留所有的消息，无论其被消费与否。当然，因为磁盘限制，不可能永久保留所有数据（实际上也没必要），因此Kafka提供两种策略去删除旧数据。一是基于时间，二是基于partition文件大小。可以通过配置$KAFKA_HOME/config/server.properties ，让Kafka删除一周前的数据，也可通过配置让Kafka在partition文件超过1GB时删除旧数据:

	############################# Log Retention Policy #############################

	# The following configurations control the disposal of log segments. The policy can
	# be set to delete segments after a period of time, or after a given size has accumulated.
	# A segment will be deleted whenever *either* of these criteria are met. Deletion always happens
	# from the end of the log.

	# The minimum age of a log file to be eligible for deletion
	log.retention.hours=168

	# A size-based retention policy for logs. Segments are pruned from the log as long as the remaining
	# segments don't drop below log.retention.bytes.
	#log.retention.bytes=1073741824

	# The maximum size of a log segment file. When this size is reached a new log segment will be created.
	log.segment.bytes=1073741824

	# The interval at which log segments are checked to see if they can be deleted according
	# to the retention policies
	log.retention.check.interval.ms=300000

	# By default the log cleaner is disabled and the log retention policy will default to 
	# just delete segments after their retention expires.
	# If log.cleaner.enable=true is set the cleaner will be enabled and individual logs 
	# can then be marked for log compaction.
	log.cleaner.enable=false

这里要注意，因为Kafka读取特定消息的时间复杂度为O(1)，即与文件大小无关，所以这里删除文件与Kafka性能无关，选择怎样的删除策略只与磁盘以及具体的需求有关。




推荐阅读
-------

1. [Centralized Logging Solutions Overview](http://elekslabs.com/2014/05/centralized-logging-solutions-overview.html)
2. [Logging and Aggregation at Quora](https://engineering.quora.com/Logging-and-Aggregation-at-Quora)
3. [ELK在广告系统监控中的应用 及 Elasticsearch简介](http://tech.youmi.net/2016/02/137134732.html)
4. [Centralized Logging](http://jasonwilder.com/blog/2012/01/03/centralized-logging/)
5. [Centralized Logging Architecture](http://jasonwilder.com/blog/2013/07/16/centralized-logging-architecture/) 强烈推荐！