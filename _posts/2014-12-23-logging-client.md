---
title: 应用如何记录集中日志
layout: post
---


首先需要配置pom.xml，引入logging-client包：

	<dependency>
		<groupId>com.baidu.global.mobile.server.logging.client</groupId>
		<artifactId>logging-client</artifactId>
		<version>1.0-SNAPSHOT</version>
	</dependency>


### 方式一：通过log4j

	# Remote Log Server Appender
	log4j.appender.server=com.baidu.global.mobile.server.logging.client.AsyncSocketAppender
	log4j.appender.server.Port=4560
	log4j.appender.server.RemoteHost=hk01-hao123-mob07.hk01
	log4j.appender.server.Application=guanxing

然后在对于想要集中日志的logger配置该appender，比如

	# Global logging configuration
	log4j.rootLogger=WARN,console, system, server


	# My logging(Application) logging configuration
	log4j.logger.com.baidu.global.mobile.server.guanxing=info, guanxing, server
	log4j.additivity.com.baidu.global.mobile.server.guanxing=false


**NOTE**

1. 注意配置好你的Application，不要混淆了。
2. 如果rootLogger和应用Logger都配置了AsyncSocketAppender，那么注意配置应用的Logger的additivity配置为false，否则会记录两次。


### 方式二：应用直接调用接口发送

通过log4j可以在一定程度上对应用透明，但是其实log4j并不是很灵活。所以你也可以选择直接发送日志给logging-server。接口很简单：

	LogClient logClient = LogClient.getInstance(remoteHost, port);
	logClient.logAsync(loggingMessage);


其中LoggingMessage定义如下：

	public String env; // optional
	public long timestamp; // optional
	public String host;
	
	public String application; 

	public String threadName; // optional
	public String locationInfo; // optional
	public String level; // required
	public String message; // optional
	public String stackTrace; // optional	


-- That's all. --

