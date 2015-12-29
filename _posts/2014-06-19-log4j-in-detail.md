---
layout: post
title: log4j详细介绍
---


log4j核心概念
-------------

基本上所有的日志系统（库）对于日志记录的抽象都差不多，就是 日志记录者（Loggers）将message以某种格式（Layouts）写到某些日志文件（Appenders）中。

这里有三个关键的概念——Loggers，Layouts，Appenders，正是Log4J的三个核心概念。下面我们一一介绍一下。

![Log4j Core Components](/media/images/log4j-core-components.png)

### 一、Loggers——The ability to selectively enable or disable logging requests based on their logger.


#### 1. Loggers是树形继承关系的，这个关系将影响到下面的Appenders，因此掌握这个概念很重要。

有且仅有一个root logger，应用程序通过Logger.getRootLogger()获取而不是通过名字；其他的Loggers都是root logger的子孙Loggers，应用程序通过Logger.getLogger(loggerName)获取。他们之间的继承关系通过名字确定:

> Loggers are named entities. Logger names are case-sensitive and they follow the hierarchical naming rule:
>
> **Named Hierarchy**
> 	A logger is said to be an ancestor of another logger if its name followed by a dot is a prefix of the descendant logger name. 
>	A logger is said to be a parent of a child logger if there are no ancestors between itself and the descendant logger.
>
> For example, the logger named "com.foo" is a parent of the logger named "com.foo.Bar". Similarly, "java" is a parent of "java.util" and an ancestor of "java.util.Vector". This naming scheme should be familiar to most developers.

下面是Loggers的重要接口：

	package org.apache.log4j;

	  public class Logger {

	    // Creation & retrieval methods:
	    public static Logger getRootLogger();
	    public static Logger getLogger(String name);

	    // printing methods:
	    public void trace(Object message);
	    public void debug(Object message);
	    public void info(Object message);
	    public void warn(Object message);
	    public void error(Object message);
	    public void fatal(Object message);

	    // generic printing method:
	    public void log(Level l, Object message);
	}

#### 2. 每个Loggers都必须有记录等级，Loggers之间的等级也存在一种继承关系，但是这种继承关系是一种重载的关系，即如果子Logger有定义了自己的LEVEL，那么它就不需要继承父Logger的LEVEL。

> Loggers may be assigned levels. The set of possible levels, that is: TRACE, DEBUG, INFO, WARN, ERROR and FATAL，are defined in the org.apache.log4j.Level class. 
> If a given logger is not assigned a level, then it inherits one from its closest ancestor with an assigned level. More formally:
> **Level Inheritance**
> 	The inherited level for a given logger C, is equal to the first non-null level in the logger hierarchy, starting at C and proceeding upwards in the hierarchy towards the root logger.
> To ensure that all loggers can eventually inherit a level, the root logger always has an assigned level.

每个LEVEL都有一个相应的方法供应用程序调用：分别是 debug, info, warn, error, fatal and log. 如logger.info("..")。但是这里要特别注意一下，Logger只会打印日志级别大于等于它本事level的日志请求。比如一个level配置为warn的logger，发送给这个logger的info、debug级别的日志就不会打印出来。

> A logging request is said to be enabled if its level is higher than or equal to the level of its logger. Otherwise, the request is said to be disabled. A logger without an assigned level will inherit one from the hierarchy. This rule is summarized below.
> 
> **Basic Selection Rule**
> 	A log request of level p in a logger with (either assigned or inherited, whichever is appropriate) level q, is enabled if p >= q.

这个规则是log4j的核心. level之间是有顺序的，对于内建的level，顺序如下：DEBUG < INFO < WARN < ERROR < FATAL. 下面是这个规则的一个例子：

	//建立Logger的一个实例，命名为“com.foo”
	Logger　logger = Logger.getLogger("com.foo"); //"com.foo"是实例进行命名，也可以任意
	//设置logger的级别。通常不在程序中设置logger的级别。一般在配置文件中设置。
	logger.setLevel(Level.INFO);
	Logger barlogger = Logger.getLogger("com.foo.Bar");
	//下面这个请求可用，因为WARN >= INFO
	logger.warn("Low fuel level.");
	//下面这个请求不可用，因为DEBUG < INFO
	logger.debug("Starting search for nearest gas station.");
	//命名为“com.foo.bar”的实例barlogger会继承实例“com.foo”的级别。因此，下面这个请求可用，因为INFO >= INFO
	barlogger.info("Located nearest gas station.");
	//下面这个请求不可用，因为DEBUG < INFO
	barlogger.debug("Exiting gas station search");

说明：

1. 这里“是否可用”的意思是能否输出Logger信息。
2. 通过logger的级别验证，还要再通过apperder的threshold验证（详见下面Appender介绍）。

#### 3. Loggers的获取方式采用了工厂方法，返回的是Singleton。

通过getLogger(name)方式，传递同样的name会返回一样的logger对象。

例如：

	Logger x = Logger.getLogger("wombat");

	Logger y = Logger.getLogger("wombat");

x and y refer to exactly the same logger object. 


### 二、Appenders——Allow logging requests to print to multiple destinations

禁用与使用日志请求只是Log4j其中的一个小小的地方，Log4j日志系统允许把日志输出到不同的地方，如控制台（Console）、文件（Files），根据天数或者文件大小产生新的文件，以流的形式发送到其它地方，异步记录日志等等。

1、定义：appender其实就是输出源(output destination, target, endpoint)。可以是终端、文件、图形界面、远程socket服务、JMS、邮件等。还支持日志的异步写入。

> In log4j speak, an output destination is called an appender. Currently, appenders exist for the console, files, GUI components, remote socket servers, JMS, NT Event Loggers, and remote UNIX Syslog daemons. It is also possible to log [asynchronously](http://logging.apache.org/log4j/1.2/apidocs/org/apache/log4j/AsyncAppender.html).

2、一个logger可以有一个或多个appender，即日志可以同时记录到多个地方。通过logger.addAppender(Appender newAppender)方法将appender添加到指定的logger。

3、每一个发送给logger的合法（即没有被logger等级过滤掉）日志请求都会发送到该logger关联的每一个appenders和该logger的每一个父logger的appender。换一种说法就是logger会继承父logger的appender。可以通过设置`additivity=false`取消。

> Each enabled logging request for a given logger will be forwarded to all the appenders in that logger as well as the appenders higher in the hierarchy. In other words, appenders are inherited additively from the logger hierarchy. For example, if a console appender is added to the root logger, then all enabled logging requests will at least print on the console. If in addition a file appender is added to a logger, say C, then enabled logging requests for C and C's children will print on a file and on the console. It is possible to override this default behavior so that appender accumulation is no longer additive by [setting the additivity flag](http://logging.apache.org/log4j/1.2/apidocs/org/apache/log4j/Category.html#setAdditivityboolean) to false.

这个规则称之为Appender Additivity，也算是log4j最复杂的规则吧。

> **Appender Additivity** 
> 	The output of a log statement of logger C will go to all the appenders in C and its ancestors. This is the meaning of the term "appender additivity".
> 
> 	However, if an ancestor of logger C, say P, has the additivity flag set to false, then C's output will be directed to all the appenders in C and its ancestors upto and including P but not the appenders in any of the ancestors of P.
> 
> 	Loggers have their additivity flag set to true by default.

看一下下面的例子就知道了：

![Log4j Appender Additivity Example](/media/images/log4j-appender-additivity-example.jpg)

**TIPS** 一般来说一条日志，在同一种类型的appender（比如文件）只需要记录一次就可以了。所以，建议对同一种类型的appender设置`additivity=false`。

4、Appender也有level等级。在Logger中称之为level，在appender中称之为Threshold，决定Appender记录什么等级的日志。可以对通过logger级别过滤的日志进行最后一次的过滤。


Log4j内建的Apperders：


- org.apache.log4j.AppenderSkeleton -- Abstract superclass of the other appenders in the package.
	- org.apache.log4j.WriterAppender -- Appends log events to a java.io.Writer or an java.io.OutputStream depending on the user's choice.
		- org.apache.log4j.FileAppender -- 文件
			- org.apache.log4j.DailyRollingFileAppender -- 每天产生一个日志文件
			- org.apache.log4j.RollingFileAppender -- 文件大小到达指定尺寸的时候产生一个新的文件
				- org.apache.log4j.ExternallyRolledFileAppender
		- org.apache.log4j.ConsoleAppender -- 控制台
	- org.apache.log4j.JDBCAppender -- 数据库
	- org.apache.log4j.JMSAppender -- 消息队列
	- org.apache.log4j.SMTPAppender -- 邮件
	- org.apache.log4j.SocketAppender -- Sends LoggingEvent objects to a remote a log server, usually a SocketNode. 
	- org.apache.log4j.SocketHubAppender -- Sends LoggingEvent objects to a set of remote log servers, usually a SocketNodes. 
	- org.apache.log4j.SyslogAppender -- send log messages to a remote syslog daemon
	- org.apache.log4j.AsyncAppender -- log events asynchronously. Can only be script configured using the org.apache.log4j.xml.DOMConfigurator. 
	- org.apache.log4j.LF5Appender -- logs events to a swing based logging console
	- org.apache.log4j.NTEventLogAppender -- Append to the NT event log system. Can only be installed and used on a Windows system.
	- org.apache.log4j.TelnetAppender
	- org.apache.log4j.NullAppender

5、Appender可以有一个或者多个Filter，对日志信息进行过滤。

6、log4j提供了AysnAppender，支持日志异步写入。但是只支持XML配置文件。


### 三、Layouts——customize the output format

如果说appender是控制日志输出到哪里，那么layout就是控制日志以何种格式打印。每一个appender一般有一个layout相关联。远程Appender只是发送日志信息，不需要Layout。

一般都是是配置[PatternLayout](http://logging.apache.org/log4j/1.2/apidocs/org/apache/log4j/PatternLayout.html)，配置起来很类似于C的printf。

Log4j内建的Layout:

- org.apache.log4j.Layout
	- org.apache.log4j.SimpleLayout -- level - log message
	- org.apache.log4j.HTMLLayout -- 以HTML表格形式布局
	- org.apache.log4j.XMLLayout -- 以XML形式布局
	- org.apache.log4j.PatternLayout -- 可以灵活地指定布局模式
	- org.apache.log4j.EnhancedPatternLayout -- PatternLayout的升级版
	- org.apache.log4j.helpers.DateLayout -- This abstract layout takes care of all the date related options and formatting work.
		- org.apache.log4j.TTCCLayout -- 包含日志产生的时间、线程、类别等等信息


### 配置

Log4j支持两种配置文件格式，一种是XML格式的文件，一种是Java Properties文件(key=value)。log4j在启动时，检查用户是否通过环境变量`log4j.configuration`指定了配置文件的路径，如果指定了，则加载之。否则，优先加载log4j.xml，如果找不到，再查找log4j.properties。

Java Properties相对于XML比较简洁，但是一些高级功能只支持XML配置，比如Filters、自定义ErrorHandlers和一些特定类型的Appender，比如Nested appenders such as the AsyncAppender。

log4j的配置其实就是配置上面说的三个核心组件。

1、配置logger

语法为

	log4j.logger.${logerName}=[${logLevel}], [${appenderName1}], [${appenderName2}], ...

方括号表示可选。logLevel默认是ALL级别（最低级别）。

其中根Logger没有名称，直接用rootLogger指定：

	log4j.rootLogger=[logLevel], ${appenderName1}, ${appenderName2}, ...

其他的logger名称一般采用类的全路径名，例如log4j.logger.org.springframework=ERROR。


appenderName:就是指定日志信息输出到哪个地方。您可以同时指定多个输出目的地。例如：log4j.rootLogger＝info,A1,B2,C3。


2、配置日志信息输出目的地(Appender)

语法为：　
	
	log4j.appender.${appenderName}=fully.qualified.name.of.appender.class

其中fully.qualified.name.of.appender.class可以指定下面内建Appender中的一个，也可以是自定义的Appender。

* org.apache.log4j.ConsoleAppender 
* org.apache.log4j.FileAppender
* org.apache.log4j.RollingFileAppender
* org.apache.log4j.DailyRollingFileAppender
* org.apache.log4j.ExternallyRolledFileAppender
* org.apache.log4j.JDBCAppender
* org.apache.log4j.JMSAppender
* org.apache.log4j.WriterAppender
* org.apache.log4j.SMTPAppender
* org.apache.log4j.SocketAppender
* org.apache.log4j.SocketHubAppender
* org.apache.log4j.SyslogAppender
* org.apache.log4j.AsyncAppender
* org.apache.log4j.LF5Appender
* org.apache.log4j.NTEventLogAppender
* org.apache.log4j.NullAppender
* org.apache.log4j.TelnetAppender

3、配置日志信息的格式(Layout)

语法为：

	log4j.appender.${appenderName}.layout=fully.qualified.name.of.layout.class

其中fully.qualified.name.of.layout.class可以指定下面内建Layout中的一个。也可以是自定义的Layout。

* org.apache.log4j.SimpleLayout
* org.apache.log4j.HTMLLayout
* org.apache.log4j.XMLLayout
* org.apache.log4j.PatternLayout
* org.apache.log4j.EnhancedPatternLayout
* org.apache.log4j.TTCCLayout


log4j最佳实践
-------------

1. 多套环境（dev, test, idc）log4j配置问题
2. FileAppender文件路径问题
3. 服务器日志与应用日志分离 && 应用模块之间日志分离
4. 使用 NDC && MDC
5. 日志统一上报
6. 日志异步上报

### 1. 多套环境（dev, test, idc）log4j配置问题

**需求**

不同的运行环境，对于日志有不同的要求。比如在线下环境，我们希望log级别是DEBUG甚至是TRACE级别，这样可以清楚的知道程序的执行逻辑，方便快速的定位问题。也会开放mybatis SQL语句。关闭远程appender功能。等等。而在线上环境，需要对性能做一定的考虑。所以一般会把日志级别设置为INFO，甚至是WARN级别。而且框架（比如spring, mybatis等）的日志级别一般会设置成WARN甚至ERROR。同时会开启远程appender，方便统一查询和监控。

**解决方案**

#### 方案一、通过环境变量`log4j.configuration`指定log4j的配置文件路径。

**步骤**

1、根据环境配置三套log4j.properties：

* log4j.properties -- 这个配置文件是为了方便没有指定时候使用。
* log4j_dev.properties
* log4j_test.properties
* log4j_idc.properties

2、修改JVM启动参数，配置`log4j.configuration`

	startApp_idc.sh

	#!/bin/bash

	# preserve current working directory
	cd `dirname $0`/..
	BASE=`pwd`
	LIBCLASSPATH=`echo $BASE/lib/*.jar | tr ' ' ':'`
	export CLASSPATH=$LIBCLASSPATH:$BASE/conf:$BASE/data
	echo $CLASSPATH
	echo $BASE

	JAVA_OPTS=" -Xmx1024m -Dlog4j.configuration=$BASE/conf/log4j_idc.properties"

	echo $JAVA_OPTS
	 
	$JAVA_HOME/bin/java $JAVA_OPTS me.arganzheng.study.standalone.dataSync.HelloWorld

优点：简单方便。
缺点: 需要修改每套环境的JVM启动变量。

#### 方案二、使用Spring的Log4jConfigListener动态加载

Spring的Log4jConfigListener支持如下功能：

1. 定时刷新配置文件，即修改log4j.properties,不需要重启Web应用。这需要在web.xml中设置一下log4jRefreshInterval参数。
2. 可以把log4j.properties和其他properties一起放在/WEB-INF/，而不是Class-Path。也支持环境变量。
3. 把log文件定在 /WEB-INF/logs/ 而不需要写绝对路径。

在web.xml配置：

	<context-param>
		<param-name>webAppRootKey</param-name>
		<param-value>helloworld</param-value>
	</context-param>
	<context-param>
		<param-name>log4jConfigLocation</param-name>
		<param-value>classpath:conf-log4j/${env}_log4j.properties</param-value>
	</context-param>
	<listener>
		<listener-class>org.springframework.web.util.Log4jConfigListener</listener-class>
	</listener>

就可以了。

具体参见：

1. [environment specific log4j configuration by spring](http://stackoverflow.com/questions/15348550/environment-specific-log4j-configuration-by-spring/15747568#15747568)
2. [Log4jConfigListener动态日志配置切换](http://www.cnblogs.com/Irving/archive/2013/02/28/2936810.html)

#### 方案三、使用auto-config在编译期间为每个环境生成相应的配置文件

原来Alibaba B2B配置方式，现在已经开源了，具体参见：[第 13 章 AutoConfig工具使用指南](http://openwebx.org/docs/autoconfig.html)。

优点：build时刻确定/修改配置；相对于运行时运行时根据环境变量替换配置文件方式，启动脚本不需要任何修改。

缺点：复杂。如果没有配置文件，需要将要求的配置项通过交互界面一一配置。而配置文件一般又不跟代码走。以前是放在home目录下，不知道现在是怎样。

maven的profile机制支持根据不同的环境进行build，不过有比较大的缺陷，不建议使用 [Maven Profile Best Practices](http://java.dzone.com/articles/maven-profile-best-practices)。

**TIPS** 

1. 可以参考这个思路自己简单实现
2. 如果使用auto-config，那么下面的FileAppender文件路径问题也就顺便解决了。


### 2. FileAppender文件路径问题

具体参见笔者前面写的一篇文章: [log4j日志路径问题](http://blog.arganzheng.me/posts/log4j-log-path-problem.html)

缺点是需要修改tomcat启动脚本。

### 3. 服务器日志与应用日志分离 && 应用模块之间日志分离

这个关键在于意识，不在于实现。因为log4j已经内建支持了。但是很多开发同学都没有意识到将不同模块的日志分离的好处。比如将tomcat的日志单独分离出去，把gcm的消息推送日志单独成一个日志文件，将统计日志单独一个文件等等。

实现很简单，只要多配置几个Logger和Appender就可以了。例如：

	# Rules reminder:
	# TRACE < DEBUG < INFO < WARN < ERROR < FATAL

	# Global logging configuration
	log4j.rootLogger=info,console, system 

	## Console output...
	log4j.appender.console=org.apache.log4j.ConsoleAppender
	log4j.appender.console.layout=org.apache.log4j.PatternLayout
	log4j.appender.console.layout.ConversionPattern=[%p]\t%d\t[%t] %c (%F\:%L) - %m%n

	## File output...
	log4j.appender.system=org.apache.log4j.DailyRollingFileAppender
	log4j.appender.system.File=${LOG_DIR}/system.log
	log4j.appender.system.Append=true 
	#log4j.appender.system.Threshold=INFO
	log4j.appender.defaultLogger.DatePattern='.'yyyy-MM-dd
	log4j.appender.system.layout=org.apache.log4j.PatternLayout
	log4j.appender.system.layout.ConversionPattern=%d [%p] [%t] %c (%F\:%L): %m%n

	## File output...
	log4j.appender.mbrowser=org.apache.log4j.DailyRollingFileAppender
	log4j.appender.mbrowser.File=${LOG_DIR}/mbrowser.log
	log4j.appender.mbrowser.Append=true
	#log4j.appender.mbrowser.Threshold=INFO
	log4j.appender.defaultLogger.DatePattern='.'yyyy-MM-dd
	log4j.appender.mbrowser.layout=org.apache.log4j.PatternLayout
	log4j.appender.mbrowser.layout.ConversionPattern=%d [%p] [%t] %c (%F\:%L): %m%n

	# 3rdparty logging configuration
	log4j.logger.org.springframework=WARN

	# My logging(Application) logging configuration
	log4j.logger.me.arganzheng.study.mbrowser=INFO, mbrowser
	log4j.additivity.me.arganzheng.study.mbrowser=false

	# GCM
	log4j.logger.com.google.android.gcm.server=INFO, mbrowser
	log4j.additivity.com.google.android.gcm.server=false


有多个logger和appender的时候要注意一下additivity 配置项。

### 4. 使用 NDC && MDC

这个可以让log4j打印一些通用的信息。不用在构造日志消息的时候拼接进去。事实上，这些公共消息是放在ConversionPattern中的，通过`%X{key}`。获取应用通过MDC
.put(key, value)放进去的值。这个功能可以非常有用，比如我们想要打印每个请求的IP地址，那么可以在servlet filter或者controller interceptor中将请求的IP地址存放在MDC中：

	public class MDCFilter implements Filter {

	    @Override
	    public void doFilter(ServletRequest request, ServletResponse response,
	            FilterChain chain) throws IOException, ServletException {

	        try {
				String ipAddress = request.getHeader("X-FORWARDED-FOR");  
				if (ipAddress == null) {  
					ipAddress = request.getRemoteAddr();  
				}
	            MDC.put("clientIp", ipAddress);

	            chain.doFilter(request, response);

	        } finally {
	            MDC.remove("clientIp");
	        }
	    }
	}


然后配置log4j.properties：

	log4j.appender.consoleAppender.layout.ConversionPattern = %-4r [%t] %5p %c %x - %m - %X{clientIp}%n

**实际例子** Alibaba B2B国际站的MA工程就使用了MDC，记录了每个请求的URL和查询字符串。对于定位问题特别方便。

    <appender name="PROJECT" class="com.alibaba.common.logging.spi.log4j.DailyRollingFileAppender">
	    <param name="file" value="/home/forrest/work2/intl-myalibaba/deploy/logs/sys/webx.log" />
	    <param name="append" value="true" />
	    <param name="encoding" value="GBK" />
	    <param name="threshold" value="info" />
	    <param name="datePattern" value="'.'yyyy-MM-dd" />
	    <layout class="org.apache.log4j.PatternLayout">
	        <param name="ConversionPattern" value="%d [%X{requestURIWithQueryString}] %-5p %c{2} - %m%n" />
	    </layout>
    </appender>

### 5. 日志统一上报

线上应用为了高可用性，往往部署在多台服务器。如果遇到比较诡异的线上问题，需要登录特定设置多台机器查看相应的log文件。比较费时间。如果能够把日志收集在一个地方，并且提供查询界面。可以很方便的查看日志定位问题。

**解决思路** 可以将日志统一存放在DB或者HBase中。然后提供查询界面（可以使用搜索引擎提供全文搜索）。

首先想到的是JDBCAppender，直接将日志记录到DB中。不过这样所有的应用对于DB就有依赖，而且未必所有的机器都有DB访问权限。而且底层如果后续不采用DB，所有的应用都需要修改。另外，JDBCAppender是同步写，对应用性能可能有影响。当然，可以采用AsyncAppender异步化。

那么考虑SocketAppender，将日志发送到远端的服务器上。但SocketAppender有以下的缺点：

* 存储和发送日志是一个同步过程，有可能会出现打日志的动作堵死应用程序的场景。
* 需用户自己开发服务端程序，且客户端和服务端均使用同步socket通信，吞吐率较低。在处理大日志量时力不从心。

同步影响性能问题，我们可以采用AsyncAppender异步化，唯一的成本就是需要将配置文件改成XML形式（当然如果你一开始就使用了XML配置格式，就不需要迁移了）。不过SocketAppender确实采用最原始的Socket通讯，IO性能比较低。对于日志量比较大的应用，可以考虑自己实现一个异步IO的Appender。这样，配置文件也不需要修改成XML了。实现很简单，只需要继承抽象类AppenderSkeleton，然后放在classpath中，就可以在log4j.properties中配置，log4j在写日志时就会往这个appender中发一份日志。为了提高日志搜集系统的吞吐率，在通信框架选型时，可以采用Netty通信框架。还可以使用ProtocolBuffer对数据进行压缩，进一步提高效率。整个实现估计一周左右可以完成。

**NOTES && TIPS**

1. 定期清理过期日志，避免日志撑爆。
2. 远程上报日志比本地日志要耗性能，可以采用异步上报方式。
3. 可以提供邮件报警，增加监控能力。


**实战例子** Alibaba B2B国际站的Syslog系统，就是自己实现了一个SyslogRmiAppender。还提供了配置方式，可以脱离log4j在代码中显示使用。使用Syslog上报的日志包括：

1. SQL
2. 事务
3. 应用日志（一般是错误日志）
4. 性能监控
5. Monitor

**NOTES**

笔者已经实现了一个简单的集中式日志监控平台，具体可以参见:

1. [日志监控系统](http://blog.arganzheng.me/posts/log-monitoring.html)
2. [应用如何记录集中日志](http://blog.arganzheng.me/posts/logging-client.html)

### 6. 日志异步上报

log4j默认是同步打印日志的。本地appender（终端、文件）还好，如果是远程apperder（邮件、DB、socket等）那么可能会对应用性能产生影响。Log4j提供了[AsyncAppender](https://logging.apache.org/log4j/1.2/apidocs/org/apache/log4j/AsyncAppender.html)，可以异步的记录日志。他可以连接多个实际的appenders，然后将logger发给他的日志，异步的转发给这些关联的appenders。这对于远程appender来说还是很有必要的。

**注意** AsyncAppender只支持XML配置格式。


其他的log框架
-------------

1. [Apache Log4j 2](http://logging.apache.org/log4j/2.x/)
2. [Logback ](http://logback.qos.ch/)


建议日志规范
------------


1、直接使用log4j，而不是commons-logging+log4j。即使用Logger.getLogger(".."")获取logger，而不是LogFactory.getLog("..")。

commons-logging的思想是提供了一组通用的日志接口，用户可以自由地选择实现日志接口的第三方软件。目前支持以下日志实现：

* log4J日志器（http://jakarta.apache.org/log4j）
* JDK1.4 Logging日志器(JDK1.4自带)
* SimpleLog日志器(把日志消息输出到标准系统错误流System.err)
* NoOpLog(不输出任何日志信息)

但是commons-logging已经非常老了，据说有bug。另外支持的第三方日志库也不多，就上面四种实现，其实必然是log4j。而且已经被slf4j取代。建议是直接使用一种日志框架，或者使用slf4j。

2、注意log和appender的级别，不要打太多没用的东西。

3、日志内容应该方便查询(grep)，并且带上足够的上下文方便定位问题。

4、注意不要吞了异常堆栈

	try{
		doSomething();
	}catch(Exception e){
		logger.error(e.getMessage()); // make sure you want to DO THIS!
	}

	logger.error(e) => logger.error("", e);

5、没有必要打印异常堆栈就不要打：

	Assert.notNull(language);
	Assert.notNull(product);
	Assert.notNull(uuid);

=>

	try{
		Assert.notNull(language, "language should not be null!");
		Assert.notNull(product, "product should not be null!");
		Assert.notNull(uuid, "uuid should not be null!");
	}catch(IllegalArgumentException e){
		logger.error(e.getMessage());
	}

6、上面的最佳实践根据需要采用实施


参考文档以及推荐阅读
--------------------

1. [Short introduction to log4j](http://logging.apache.org/log4j/1.2/manual.html) 官方文档。
2. [Log4J徹底解説](http://www.nurs.or.jp/~sug/soft/log4j/index.htm) 日本文档，非常详尽！
3. [log4j](http://www.blogjava.net/rendong/archive/2006/09/04/67586.html) 中文文档，应该是目前国内最好的log4j文档了。
4. [Java日志管理：Logger.getLogger()和LogFactory.getLog()的区别（详解Log4j）](http://javacrazyer.iteye.com/blog/1135493) 总结的不错。
5. [你的Java日志，有没有用这些改进办法？](http://mp.weixin.qq.com/s?__biz=MjM5MDE0Mjc4MA==&mid=207451012&idx=1&sn=de9fba4eda0f221363b6d5ae54243416#rd) 国内少有的干货比较多的技术文章。
