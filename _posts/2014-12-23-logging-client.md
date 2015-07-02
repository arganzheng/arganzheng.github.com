---
title: 应用如何记录集中日志
layout: post
---


对于分布式系统，提供一个集中式的日志监控平台是非常有必要的，具体参见笔者前面的一片文章：[日志监控系统](http://blog.arganzheng.me/posts/log-monitoring.html)。这里介绍一下客户端要怎么使用。

首先需要配置pom.xml，引入logging-client包：

	<dependency>
		<groupId>com.baidu.global.mobile.server.logging.client</groupId>
		<artifactId>logging-client</artifactId>
		<version>1.0.0</version>
	</dependency>


### 方式一：通过log4j

	# Remote Log Server
	log4j.appender.remote=com.baidu.global.mobile.server.logging.client.AsyncSocketAppender
	log4j.appender.remote.Port=4560
	log4j.appender.remote.RemoteHost=10.242.112.38,10.242.116.27,10.242.117.36
	log4j.appender.remote.Application=guanxing
	#log4j.appender.remote.threadPoolSize=2
	#log4j.appender.remote.loadBalancerName=WRR
	log4j.appender.remote.Threshold=ERROR

然后在对于想要集中日志的logger配置该appender，比如

	# Global logging configuration
	log4j.rootLogger=WARN,console, system, remote

	# My logging(Application) logging configuration
	log4j.logger.com.baidu.global.mobile.server.guanxing=info, guanxing, remote
	log4j.additivity.com.baidu.global.mobile.server.guanxing=false


**NOTE**

1. 注意配置好你的Application，不要混淆了。
2. 如果rootLogger和应用Logger都配置了AsyncSocketAppender，那么注意配置应用的Logger的additivity配置为false，否则会记录两次。


### 方式二：应用直接调用接口发送

通过log4j可以在一定程度上对应用透明，但是其实log4j并不是很灵活。所以你也可以选择直接发送日志给logging-server。接口很简单：

	package com.baidu.global.mobile.server.logging.client;
	
	import com.baidu.global.mobile.server.logging.common.LoggingMessage;
	
	public interface Logger {
		public void log(LoggingMessage loggingMessage);
	
		public void destroy();
	}

你可以通过LoggerFactory获取：

	package com.baidu.global.mobile.server.logging.client;
	
	/**
	 * <pre>
	 * 因为Agent的本身是有线程池和网络连接池，开销比较大，不适合于每次都New一个，最好是一个应用共用一个，所以提供这么一个单例工厂方法。
	 * 
	 * 建议如果使用Spring的话，直接在Spring配置就可以了。默认就是单例，而且可以很方便的注入。
	 * 
	 * <bean id="logger"
	 * 		class="com.baidu.global.mobile.server.logging.client.LogClient">
	 * 		<constructor-arg value="classpath:${env}_logger.xml" />
	 * </bean>
	 * 
	 * </pre>
	 * 
	 * @author argan
	 */
	public class LoggerFactory {
		private static volatile Logger instance = null;
	
		public static Logger getInstance(String configLocation) {
			if (instance == null) {
				synchronized (LoggerFactory.class) {
					if (instance == null) {
						instance = new LogClient(configLocation);
					}
				}
			}
			return instance;
		}
	
		public static Logger getInstance(LoggerConfig config) {
			if (instance == null) {
				synchronized (LoggerFactory.class) {
					if (instance == null) {
						instance = new LogClient(config);
					}
				}
			}
			return instance;
		}
	
	}

获取方式很简单：

	Logger logClient = LoggerFactory.getInstance("classpath:logger.xml");

或者Spring配置：

	<bean id="logger"
  		class="com.baidu.global.mobile.server.logging.client.LogClient">
 		<constructor-arg value="classpath:${env}_logger.xml" />
 	</bean>

其中logger.xml配置如下：

	<logger>
		<region>HK</region>
		<application>monitor</application>
		<threadPoolSize>2</threadPoolSize>
		<loadBalancerName>WRR</loadBalancerName>
		<servers>10.242.112.38,10.242.116.27,10.242.117.36</servers>
		<endpoints>
			<endpoint>
				<host>10.242.112.38</host>
				<port>4567</port>
				<weight>5</weight>
				<maxFails>1</maxFails>
				<failTimeout>10</failTimeout>
				<timeout>2</timeout>
			</endpoint>
			<endpoint>
				<host>10.242.116.27</host>
				<port>4567</port>
			</endpoint>
			<endpoint>
				<host>10.242.117.36</host>
				<port>4567</port>
			</endpoint>
		</endpoints>
	</logger>

LoggingMessage定义如下：

	public String env; // optional
	public long timestamp; // optional
	public String host;
	
	public String application; 

	public String threadName; // optional
	public String locationInfo; // optional
	public String level; // required
	public String message; // optional
	public String stackTrace; // optional	

可以看到这里其实就是log4j的LogEvent定义，所以其实还是有点语言相关的，要做成通用的可以把stackTrace之类的字段改成detail之类的通用名称。

-- That's all. --

