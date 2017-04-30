---
title: logback学习笔记
layout: post
catalog: true
---


一直使用log4j，虽然知道log4j有性能上的问题，logback和log4j2都已经流行很多年了，但是总是懒得升级。最近新项目使用了spring boot，默认是配置logback，刚好学习一下。

logback的配置很像log4j，而且起核心概念也是跟log4j一样。不外乎就是logger，appender和layout的关系，只不过appender多了一个filter组件。具体可以参考我前面写过的文章: [log4j详细介绍](http://arganzheng.life/log4j-in-detail.html)。

下面看具体实际问题的解决方案。

### 1. 多套环境（dev, test, idc）logback配置问题

logback的加载顺序如下 [Chapter 3: Logback configuration](http://logback.qos.ch/manual/configuration.html) :

> Let us begin by discussing the initialization steps that logback follows to try to configure itself:
>
> Logback tries to find a file called logback.groovy in the classpath.
> 
> If no such file is found, logback tries to find a file called logback-test.xml in the classpath.
>
> If no such file is found, it checks for the file logback.xml in the classpath.
>
> If no such file is found, service-provider loading facility (introduced in JDK 1.6) is used to resolve the implementation of com.qos.logback.classic.spi.Configurator interface by looking up the file META-INF\services\ch.qos.logback.classic.spi.Configurator in the class path. Its contents should specify the fully qualified class name of the desired Configurator implementation.
>
> If none of the above succeeds, logback configures itself automatically using the BasicConfigurator which will cause logging output to be directed to the console.
>
> The last step is meant as last-ditch effort to provide a default (but very basic) logging functionality in the absence of a configuration file.

上面的介绍没有介绍指定路径的方式。但是实际上logback跟log4j一样，是支持指定配置文件路径的。[Specifying the location of the default configuration file as a system property](http://logback.qos.ch/manual/configuration.html#configFileProperty)

> #### Specifying the location of the default configuration file as a system property
>
> You may specify the location of the default configuration file with a system property named "logback.configurationFile". The value of this property can be a URL, a resource on the class path or a path to a file external to the application.
> 
> java -Dlogback.configurationFile=/path/to/config.xml chapters.configuration.MyApp1
> 
> Note that the file extension must be ".xml" or ".groovy". Other extensions are ignored. Explicitly registering a status listener may help debugging issues locating the configuration file.

所以我们可以简单的通过一个环境变量来支持多套环境，就像spring boot的application-{profile}.properties配置文件一样。
创建logback-{spring.profiles.active}.xml配置文件，然后启动的时候指定 -Dspring.profiles.active=production，达到动态切换的目的。

除了这种方式，我们还可以利用logback的条件配置达到不同环境不同配置的效果。

	 <if condition='"prd".equals(property("environment")) || "stg".equals(property("environment"))'>
	        <then>
            <property name="APPLINK_LOG_LEVEL" value="warn"/>
        </then>
        <else>
            <property name="APPLINK_LOG_LEVEL" value="debug"/>
        </else>
    </if>

还可以利用文件导入，做一个间接转发 [Logging with logback and external configuration](http://blog.patouchas.net/technology/logging-with-logback-and-external-configuration/):

	<?xml version="1.0" encoding="UTF-8"?>
	<configuration scan="true" scanPeriod="300 seconds">
	    <property resource="logback.properties" />
	 
	    <if condition='property("env").equals("prod")'>
	        <then>
	            <include file="logback-prod.xml" />
	        </then>
	    </if>
	    <if condition='property("env").equals("dev")'>
	        <then>
	            <include resource="logback-dev.xml" />
	        </then>
	    </if>
	</configuration>


### Logback对log4j的功能增强

logback多log4j最大的贡献是解决了性能问题。不过在功能性方面，也做了很多改进：

1. 自动扫描变化
2. appender filter
3. 支持基本的条件配置
4. 支持文件导入(file inclusion)
5. 支持格式化日志，logger.error("one two three: {} {} {}", "a", "b", "c", new Exception("something went wrong"));


**TIPS** logback配置模板参考

	<?xml version="1.0" encoding="UTF-8"?>
	<configuration>

		<appender name="CONSOLE" class="ch.qos.logback.core.ConsoleAppender">
			<layout class="ch.qos.logback.classic.PatternLayout">
	            <pattern>%date [%thread] %highlight(%-5level) %logger{26} - %msg%n</pattern>
			</layout>
		</appender>

		<appender name="SYSTEM"
			class="ch.qos.logback.core.rolling.RollingFileAppender">
			<file>${LOG_DIR}/system.log</file>
			<encoder class="ch.qos.logback.classic.encoder.PatternLayoutEncoder">
	            <pattern>%date [%thread] %highlight(%-5level) %logger{26} - %msg%n</pattern>
			</encoder>

			<rollingPolicy class="ch.qos.logback.core.rolling.TimeBasedRollingPolicy">
				<!-- rollover daily -->
				<fileNamePattern>${LOG_DIR}/archived/system.%d{yyyy-MM-dd}.log
				</fileNamePattern>
				<!-- keep 30 days' worth of history -->
				<maxHistory>30</maxHistory>
				<timeBasedFileNamingAndTriggeringPolicy
						class="ch.qos.logback.core.rolling.SizeAndTimeBasedFNATP">
					<maxFileSize>100MB</maxFileSize>
				</timeBasedFileNamingAndTriggeringPolicy>
			</rollingPolicy>
		</appender>

		<appender name="APP"
			class="ch.qos.logback.core.rolling.RollingFileAppender">
			<file>${LOG_DIR}/app.log</file>
			<encoder class="ch.qos.logback.classic.encoder.PatternLayoutEncoder">
	            <pattern>%date [%thread] %highlight(%-5level) %logger{26} - %msg%n</pattern>
			</encoder>

			<rollingPolicy class="ch.qos.logback.core.rolling.TimeBasedRollingPolicy">
				<!-- rollover daily -->
				<fileNamePattern>${LOG_DIR}/archived/app.%d{yyyy-MM-dd}.log
				</fileNamePattern>
				<!-- keep 30 days' worth of history -->
				<maxHistory>30</maxHistory>
				<timeBasedFileNamingAndTriggeringPolicy
						class="ch.qos.logback.core.rolling.SizeAndTimeBasedFNATP">
					<maxFileSize>100MB</maxFileSize>
				</timeBasedFileNamingAndTriggeringPolicy>
			</rollingPolicy>
		</appender>

		<appender name="REMOTE"
	        class="me.arganzheng.study.toolkit.logging.client.appender.logback.AsyncSocketAppender">
	        <remoteHost>10.242.77.38</remoteHost>
	        <port>4560</port>
	        <application>api-gateway-web</application>
	        <!-- <threadPoolSize>2</threadPoolSize> <loadBalancerName>WRR</loadBalancerName> -->
	        <filter class="ch.qos.logback.classic.filter.ThresholdFilter">
	            <level>ERROR</level>
	        </filter>
	    </appender>
	    
		<logger name="me.arganzheng.study.mobopay.applications.api.gateway" level="debug"
			additivity="false">
			<appender-ref ref="APP" />
	        <appender-ref ref="REMOTE" />
		</logger>

		<root level="info">
			<appender-ref ref="CONSOLE" />
			<appender-ref ref="SYSTEM" />
	        <appender-ref ref="REMOTE" />
		</root>

	</configuration>






