---
title: 日志监控系统
layout: post
---


What is a Log?

Timestramp + Data = Log

Life of a Log

record => transmit => analyze => store => delete

Related Tools(open source)

* transport: flume, fluentd, scribe, rsyslog, syslog-ng
* search+analytics: hadoop, graylog2, elsa
* storage: hdfs, cassandra, elasticsearch
* data visualize: kibana, graphite

logstash

* logstash is a unix pile on steroids: pipeline of event processing.
* Inputs, Outputs, Codecs and Filters: 23 inputs | 18 filters | 40 outputs
* transport and process logs to and from anywhere. 
* provide search and analytics

The logstash agent is a processing pipeline with 3 stages: inputs -> filters -> outputs. Inputs generate events, filters modify them, outputs ship them elsewhere.


基于log4j实现分布式日志收集监控系统
===================================


背景
----

线上环境一般都是可线性扩展的分布式系统，提供一个统一查看、监控线上日志（主要是异常），对于第一时间发现和快速定位线上问题，具有非常重要的意义。


设计
----

1. client发送日志到日志中心
2. 日志中心接收日志，存储、报警、以及提供搜索、展示等功能。

### 1. client发送日志到日志中心


可以在业务代码中显示的调用日志中心提供的接口，把日志发送给日志中心。不过对业务透明性，考虑到我们所有的应用程序都是使用的log4j来记录日志。我们可以考虑通过扩展log4j将日志发送到日志中心。

对于记录分布式系统的日志，log4j本身是有提供一个SocketAppender，可以将日志发送到远端的服务器上，但SocketAppender有以下的缺点：

1. 存储和发送日志是一个同步过程，有可能会出现打日志的动作堵死应用程序的场景，毕竟现在是网络调用。
2. 需用户自己开发服务端程序，且客户端和服务端均使用同步socket通信，吞吐率较低。在处理大日志量时力不从心。

log4j其实还提供了一个AsyncAppender，可以异步打日志。不过只支持XML配置，不能使用properties配置文件。

从上面的需求可以看出，新的日志搜集系统要有以下特点：

1. 日志搜集系统要与log4j兼容，最好是能够直接在log4j.properties配置。这样可以对应用完全透明，只需要配置一下log4j.properites。
2. 必须是完全异步的，且吞吐率要求较高。不能影响到应用程序本身。
3. 日志中心要提供查看、搜索、展示、报警的功能。

log4j提供了AppenderSkeleton接口，只需要实现该虚类的方法并置于classpath中，就可以在log4j.properties中配置，log4j在写日志时就会往这个appender中发一份日志。

为了提高日志搜集系统的吞吐率，在通信框架选型时，选中了开源社区使用广泛且有较好口碑的Netty通信框架。同时使用了PB（protobuf）对日志进行编码和压缩，能够节省很大的网络流量。

日志文件的存储使用了ElasticSearch，主要在于它能很好的提供实时搜索功能，同时和Kibana结合的很好。而且由于我们前面上了搜索，已经按照了ES，可以直接拿来用。

展示方面打算使用Kibana了。

大概日志收集工程如下：log4j ==> ES ==> Kibana。


实现
----

### 消息格式定义

Message HEAD

* version：协议版本，SDK版本。
* timestamp: 如果没有上传，则为log server收集到该日志的时间。
* host: 日志从哪台机器打过来的。可以动态获取到。如果client没有指定，那么log server以client IP地址为准。
* "type": 日志类型：主要有log4j、syslog、sqllog、nginx-log, etc. 会影响后面的body格式。
* application: 哪个应用产生的日志：比如Nantianmen, Guanxing, Spider等。这个需要应用配置一下，也需要client发送，log server获取不到。当然，其实从后面的locationInfo也可以判断出来。

Message BODY for Log4j

* threadName：哪个线程上报的错误
* locationInfo: fully.qualified.classname.of.caller.methodName(Filename.java:line) 
* level: String
* message: String
* stackTrace: String


其中stackTrace是从LoggingEvent的ThrowableInfo字段抽取成字符串的：

	protected void subAppend(LoggingEvent event) {
	    this.qw.write(this.layout.format(event));

	    if(layout.ignoresThrowable()) {
	      String[] s = event.getThrowableStrRep();
	      if (s != null) {
		int len = s.length;
		for(int i = 0; i < len; i++) {
		  this.qw.write(s[i]);
		  this.qw.write(Layout.LINE_SEP);
		}
	      }
	    }

	    if(shouldFlush(event)) {
	      this.qw.flush();
	    }
	}

**思考** 

要不要像opentsdb一样，提供tags这样灵活的key=value参数结构？

### 日志上报接口

应用可以显示调用接口上报日志，也可以通过log4j透明上报。不管是使用哪一种，都是调用日志中心提供的接口。

### log4j Appender

	public void append(LoggingEvent event) {

 	}

### ES存储
 
具体参见笔者的另一篇文章：[]()。这里就不赘述了。

部署
----

需要配置的信息

1. log server list(IP:端口，客户端负载均衡?)
2. appId
3. appender信息

这个三个可以合并在log4j配置文件里面。

### log4j.properties

	log4j.rootLogger=DEBUG, server
	log4j.appender.server=org.apache.log4j.net.SocketAppender
	log4j.appender.server.Port=4712
	log4j.appender.server.RemoteHost=loghost
	log4j.appender.server.ReconnectionDelay=10000

**NOTE** 约定

由于分词的特殊性，所有的日志都使用英文。



使用Netty+Protobuf作为Transport
-------------------------------

虽然日志的偶然丢失是可以接受的，但是主要由于链路层的MTU限制，如果UDP超过MTU导致分包，就有可能出现传输错误。而我们的错误日志一般包含堆栈信息，所以一般会超过MTU（1472或者548字节），还是打算使用TCP。由于TCP是面向字节流的协议，所以需要业务自己划分记录的边界。常见的方法有如下几种方式：

1. 特殊的开始(可选)和结束符号。得到这个字符意味着记录的结束，需要保证边界字符的特殊性（可能需要转义）。
2. 固定的header，然后在head中增加body的size信息。读到header之后，根据header中的size信息，继续读取size字节。


**TIPS** 

#### 1. Header本身也可以是变长的，但是每个变长的部分，必须有一个固定的（嵌套）header。最简单的做法就是把Header用PB序列化为byte数组存储。
#### 2. 为了安全起见，很多协议设计的时候会结合这两者。比如Elasticsearch，它的包头就是以ES开头的。然后就是记录剩下的size：

	package org.elasticsearch.transport.netty;

	public class NettyHeader {

	    public static final int HEADER_SIZE = 2 + 4 + 8 + 1 + 4;

	    public static void writeHeader(ChannelBuffer buffer, long requestId, byte status, Version version) {
	        int index = buffer.readerIndex();
	        buffer.setByte(index, 'E');
	        index += 1;
	        buffer.setByte(index, 'S');
	        index += 1;
	        // write the size, the size indicates the remaining message size, not including the size int
	        buffer.setInt(index, buffer.readableBytes() - 6);
	        index += 4;
	        buffer.setLong(index, requestId);
	        index += 8;
	        buffer.setByte(index, status);
	        index += 1;
	        buffer.setInt(index, version.id);
	    }
	}

然后Pipeline中会做这个检查：

	package org.elasticsearch.transport.netty;

	public class SizeHeaderFrameDecoder extends FrameDecoder {

	    private static final long NINETY_PER_HEAP_SIZE = (long) (JvmInfo.jvmInfo().mem().heapMax().bytes() * 0.9);

	    @Override
	    protected Object decode(ChannelHandlerContext ctx, Channel channel, ChannelBuffer buffer) throws Exception {
	        if (buffer.readableBytes() < 6) {
	            return null;
	        }

	        int readerIndex = buffer.readerIndex();
	        if (buffer.getByte(readerIndex) != 'E' || buffer.getByte(readerIndex + 1) != 'S') {
	            throw new StreamCorruptedException("invalid internal transport message format");
	        }

	        int dataLen = buffer.getInt(buffer.readerIndex() + 2);
	        if (dataLen <= 0) {
	            throw new StreamCorruptedException("invalid data length: " + dataLen);
	        }
	        // safety against too large frames being sent
	        if (dataLen > NINETY_PER_HEAP_SIZE) {
	            throw new TooLongFrameException(
	                    "transport content length received [" + new ByteSizeValue(dataLen) + "] exceeded [" + new ByteSizeValue(NINETY_PER_HEAP_SIZE) + "]");
	        }

	        if (buffer.readableBytes() < dataLen + 6) {
	            return null;
	        }
	        buffer.skipBytes(6);
	        return buffer;
	    }
	}


再比如TX电商的IDL协议包格式，也是如此：Head以2字节的55aa Magic Number开始，随后是4个字节的Data Length = 1 + 105 + N + 1。
Data域的格式为0x02起始字，105字节的Body Head，还有N字节的Body，最后是0x03结束字。

还有阿里原来的Dubbo RPC框架，它的包头包含的信息包括：Request/Response, Two way, Heartbeat, Serialization "type", Request ID, Data length。



我们这里也参考这种做法

* 起始标志：1B, 魔数, 0x02
* RPC Version: 2B
* Header Length: UInt16
* Header Body: (PB)
	* Serialization "type"
		* 1: Protocal Buffer
		* 2: Json
		* 3: XML
	* oneway：
	* clientIP: UInt32
	* clientPort: UInt16
	* cmdId: 命令字（要请求的方法：fully.qualified.classname.of.caller.methodName + version ?）
* Data length: 4B, TBD
* Data Body: PB
* 结束标志：1B 魔数，0x03


* decoder 抽取每一个消息出来
* handler 根据消息的header中的cmdId找到业务处理对象，然后根据handler方法的request参数类型，序列化data body，再调用handler方法。

MessageHandler的方法：

	void handlerMessage(RpcController controller, Object request, RpcCallback<FooResponse> done);

每个Service的方法：



	public class NettyHeader {

	    public static final int HEADER_SIZE = 2 + 4 + 8 + 1 + 4;

	    public static void writeHeader(ChannelBuffer buffer, integer version) {
	        int index = buffer.readerIndex();
	        buffer.setInt16(index, x55AA);
	        index += 2;
	        // write the size, the size indicates the remaining message size, not including the size int
	        buffer.setInt(index, buffer.readableBytes());
	        index += 4;

	        buffer.setInt(index, version);
	    }
	}


Message HEAD

* version：协议版本，SDK版本。
* timestamp: 如果没有上传，则为log server收集到该日志的时间。
* host: 日志从哪台机器打过来的。可以动态获取到。如果client没有指定，那么log server以client IP地址为准。
* "type": 日志类型：主要有log4j、syslog、sqllog、nginx-log, etc. 会影响后面的body格式。
* application: 哪个应用产生的日志：比如Nantianmen, Guanxing, Spider等。这个需要应用配置一下，也需要client发送，log server获取不到。当然，其实从后面的locationInfo也可以判断出来。

Message BODY for Log4j

* threadName：哪个线程上报的错误

* locationInfo: fully.qualified.classname.of.caller.methodName(Filename.java:line) 
* level: String
* message: String
* stackTrace: String	


### Thrift IDL

namespace java com.baidu.global.mobile.server.logging

struct LoggingMessage
{
	// 协议版本，SDK版本。
	1: i32 version, 

	// 如果没有上传，则为log server收集到该日志的时间。
	2: i64 timestamp,

	// 日志从哪台机器打过来的。可以动态获取到。如果client没有指定，那么log server以client IP地址为准。
	3: string host,

	// 日志类型：主要有log4j、syslog、sqllog、nginx-log, etc. 会影响后面的body格式。
	4: string "type",

	// 哪个应用产生的日志：比如Nantianmen, Guanxing,
	// Spider等。这个需要应用配置一下log4j支持这个。需要client发送，log
	// server获取不到。当然，从后面的locationInfo也大概可以判断出来。
	5: string application,

	/** body **/
	// 哪个线程上报的错误
	11: string threadName,

	// 格式为：fully.qualified.classname.of.caller.methodName(Filename.java:line)
	12: string locationInfo,

	// 日志等级：ERROR, WARN, INFO, DEBUG, TRACE。理论上来说应该只记录ERROR，最多是WARN。
	13: string level,

	// 日志内容
	14: string message,

	// 堆栈信息
	15: string stackTrace
}


service MyLogger {
    // The 'oneway' modifier indicates that the client only makes a request and
    // does not wait for any response at all. Oneway methods MUST be void.
    oneway void log(1:LoggingMessage loggingMessage);                                                        // 4
}