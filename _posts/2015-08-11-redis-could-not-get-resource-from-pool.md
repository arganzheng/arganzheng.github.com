---
title: 记一次Redis错误排查经历
layout: post
---


日志集中式监控平台上线已经有一段时间，但是大部分情况下只是作为发布或者出问题时查看日志的便利工具使用。平时大家都不怎么主动上去看看。于是前几天把应用的错误日志也加上邮件、Hi和短信报警，马上就收到很多错误报警，引起了大家的重视。其中有一个Redis报错：

	Push notify error: => redis.clients.jedis.exceptions.JedisConnectionException: Could not get a resource from the pool at redis.clients.util.Pool.getResource(Pool.java:53) at 
	redis.clients.jedis.JedisPool.getResource(JedisPool.java:99) at 
	com.baidu.global.mobile.server.dataaccess.cache.CacheClient.execute(CacheClient.java:285) at 
	com.baidu.global.mobile.server.dataaccess.cache.CacheClient.getAsRaw(CacheClient.java:793) at 
	com.baidu.global.mobile.server.dataaccess.cache.CacheClient.get(CacheClient.java:432) at 
	com.baidu.global.mobile.server.guanxing.facade.MessageTaskFacade.checkPushValidationID(MessageTaskFacade.java:952) at 
	com.baidu.global.mobile.server.guanxing.facade.MessageTaskFacade.pushNotify(MessageTaskFacade.java:983) at 
	com.baidu.global.mobile.server.guanxing.facade.MessageTaskFacade.buildMessage(MessageTaskFacade.java:586) at 
	com.baidu.global.mobile.server.guanxing.facade.MessageTaskFacade.pushRightNow(MessageTaskFacade.java:372) at 
	com.baidu.global.mobile.server.guanxing.facade.MessageTaskFacade$AsynMessagePush.run(MessageTaskFacade.java:350) at 
	java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1142) at 
	java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:617) at 
	java.lang.Thread.run(Thread.java:745) Caused by: java.util.NoSuchElementException: Unable to validate object at org.apache.commons.pool2.impl.GenericObjectPool.borrowObject(GenericObjectPool.java:502) at 
	org.apache.commons.pool2.impl.GenericObjectPool.borrowObject(GenericObjectPool.java:361) at 
	redis.clients.util.Pool.getResource(Pool.java:51) ... 12 more

看起来挺严重的，拿不到Redis连接，而且是在	validate的时候报的错：

	java.util.NoSuchElementException: Unable to validate object at org.apache.commons.pool2.impl.GenericObjectPool.borrowObject(GenericObjectPool.java:502) at 
	org.apache.commons.pool2.impl.GenericObjectPool.borrowObject(GenericObjectPool.java:361) at 

我们使用的是Jedis，在validate的时候默认是发送一个`PING`命令到Redis，然后期待返回的是`PONG`字符串。难道连接池真的不够了？

于是登上redis所在服务器，手动ping了一下，果然发现错误：

	[work@id01-xxx-mob28.id01 redis-2.8.19]$ src/redis-cli -p 6379
	127.0.0.1:6379> ping
	(error) MISCONF Redis is configured to save RDB snapshots, but is currently not able to persist on disk. Commands that may modify the data set are disabled. Please check Redis logs for details about the error.
	127.0.0.1:6379> set hello world
	(error) MISCONF Redis is configured to save RDB snapshots, but is currently not able to persist on disk. Commands that may modify the data set are disabled. Please check Redis logs for details about the error.

读命令都没有问题，但是所有的写操作都不行。

看一下redis的log日志，发现一直在报错：

	[26378] 03 May 17:58:51.215 * 50000 changes in 60 seconds. Saving...
	[26378] 03 May 17:58:51.254 * Background saving started by pid 2854
	[2854] 03 May 17:58:58.949 * DB saved on disk
	[2854] 03 May 17:58:58.967 * RDB: 4 MB of memory used by copy-on-write
	[26378] 03 May 17:58:59.063 * Background saving terminated with success

		...
	
	[55231] 11 Aug 10:43:12.034 # Can't save in background: fork: Cannot allocate memory
	[55231] 11 Aug 10:43:18.043 * 100 changes in 3600 seconds. Saving...
	[55231] 11 Aug 10:43:18.044 # Can't save in background: fork: Cannot allocate memory
	[55231] 11 Aug 10:43:24.052 * 100 changes in 3600 seconds. Saving...
	[55231] 11 Aug 10:43:24.052 # Can't save in background: fork: Cannot allocate memory

原来是内存不足，导致无法folk进程做RDB持久化。看了一下内存使用率：

	[work@id01-xxx-mob28.id01 ~]$ free -g
	             total       used       free     shared    buffers     cached
	Mem:            62         61          1          0          0         25
	-/+ buffers/cache:         35         27
	Swap:            0          0          0	

其实free的内存还是蛮多的，有27G。然后info一下，发现其实连接的Client数并不多，但是使用的内存确实还是蛮多的，有27.55G，超过了free的内存。

	# Clients
	connected_clients:41
	client_longest_output_list:0
	client_biggest_input_buf:0
	blocked_clients:0

	# Memory
	used_memory:29281598848
	used_memory_human:27.27G
	used_memory_rss:30031474688
	used_memory_peak:29581256416
	used_memory_peak_human:27.55G
	used_memory_lua:35840
	mem_fragmentation_ratio:1.03
	mem_allocator:jemalloc-3.6.0

Redis的RDB持久化实现是folk一个子进程，然后让子进程将内存镜像dump到RDB文件中。理论上来说是需要跟父进程一样的内存空间，也就是27.55G，但是由于linux很早就支持的copy-on-write技术，所以实际上并不需要这么多的物理内存的，这个可以从log中分析出来。我们这个Redis最多只有150M左右的COW内存。
官方文档上有说明：[Background saving is failing with a fork() error under Linux even if I've a lot of free RAM!](http://redis.io/topics/faq#background-saving-is-failing-with-a-fork-error-under-linux-even-if-i39ve-a-lot-of-free-ram)

> Redis background saving schema relies on the copy-on-write semantic of fork in modern operating systems: Redis forks (creates a child process) that is an exact copy of the parent. The child process dumps the DB on disk and finally exits. In theory the child should use as much memory as the parent being a copy, but actually thanks to the copy-on-write semantic implemented by most modern operating systems the parent and child process will share the common memory pages. A page will be duplicated only when it changes in the child or in the parent. Since in theory all the pages may change while the child process is saving, Linux can't tell in advance how much memory the child will take, so if the overcommit_memory setting is set to zero fork will fail unless there is as much free RAM as required to really duplicate all the parent memory pages, with the result that if you have a Redis dataset of 3 GB and just 2 GB of free memory it will fail.
>
> Setting overcommit_memory to 1 says Linux to relax and perform the fork in a more optimistic allocation fashion, and this is indeed what you want for Redis.

可以简单的通过修改overcommit_memory系统参数为1来改变这种简单粗暴的检查行为:
	
	sysctl vm.overcommit_memory=1

马上就生效了:

	[55231] 11 Aug 10:59:31.036 * Background saving started by pid 11820
	[11820] 11 Aug 11:04:27.567 * DB saved on disk

为了避免重启失效，修改一下：/etc/sysctl.conf:

	vm.overcommit_memory=1

**TIPS** 看到拿不到连接，validate出错，第一反应总是以为是连接池不够，或者连接太多之类的。但是实际上根据validate的实现机制，只要不是返回`PONG`字符串就是错误，所以上去手动`PING`一下，就知道什么问题了:)。


补记：记一次Redis错误排查经历-下集
==============================

前天（2015年08月18日）早上突然又收到类似的错误：

	get jedis connection has error => redis.clients.jedis.exceptions.JedisConnectionException: Could not get a resource from the pool
		at redis.clients.util.Pool.getResource(Pool.java:53)
		at redis.clients.jedis.JedisPool.getResource(JedisPool.java:99)
		at com.baidu.global.mobile.hao123.dataaccess.utilscache.jedis.RedisUtils.doTask(RedisUtils.java:82)
		at com.baidu.global.mobile.hao123.nantianmen.utils.CacheHelper.getLatestEntriesByCategory(CacheHelper.java:123)
		at com.baidu.global.mobile.hao123.nantianmen.service.NewsServiceImpl.getLatestEntriesByCategory(NewsServiceImpl.java:591)
		at com.baidu.global.mobile.hao123.nantianmen.facade.NanTianMenFacadeImpl.getLatestEntriesByCategory(NanTianMenFacadeImpl.java:192)
		at com.baidu.global.mobile.hao123.nantianmen.web.MbrowserController.getLatestEntriesByCategory(MbrowserController.java:248)
		at com.baidu.global.mobile.hao123.nantianmen.web.MbrowserController.mbrowser(MbrowserController.java:138)
		at sun.reflect.GeneratedMethodAccessor143.invoke(Unknown Source)
		at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
		at java.lang.reflect.Method.invoke(Method.java:483)
		at org.springframework.web.method.support.InvocableHandlerMethod.invoke(InvocableHandlerMethod.java:215)
		at org.springframework.web.method.support.InvocableHandlerMethod.invokeForRequest(InvocableHandlerMethod.java:132)
		at org.springframework.web.servlet.mvc.method.annotation.ServletInvocableHandlerMethod.invokeAndHandle(ServletInvocableHandlerMethod.java:104)
		at org.springframework.web.servlet.mvc.method.annotation.RequestMappingHandlerAdapter.invokeHandleMethod(RequestMappingHandlerAdapter.java:745)
		at org.springframework.web.servlet.mvc.method.annotation.RequestMappingHandlerAdapter.handleInternal(RequestMappingHandlerAdapter.java:685)
		at org.springframework.web.servlet.mvc.method.AbstractHandlerMethodAdapter.handle(AbstractHandlerMethodAdapter.java:80)
		at org.springframework.web.servlet.DispatcherServlet.doDispatch(DispatcherServlet.java:919)
		at org.springframework.web.servlet.DispatcherServlet.doService(DispatcherServlet.java:851)
		at org.springframework.web.servlet.FrameworkServlet.processRequest(FrameworkServlet.java:953)
		at org.springframework.web.servlet.FrameworkServlet.doGet(FrameworkServlet.java:844)
		at javax.servlet.http.HttpServlet.service(HttpServlet.java:618)
		at org.springframework.web.servlet.FrameworkServlet.service(FrameworkServlet.java:829)
		at javax.servlet.http.HttpServlet.service(HttpServlet.java:725)
		at org.apache.catalina.core.ApplicationFilterChain.internalDoFilter(ApplicationFilterChain.java:301)
		at org.apache.catalina.core.ApplicationFilterChain.doFilter(ApplicationFilterChain.java:206)
		at org.apache.catalina.core.ApplicationDispatcher.invoke(ApplicationDispatcher.java:721)
		at org.apache.catalina.core.ApplicationDispatcher.processRequest(ApplicationDispatcher.java:468)
		at org.apache.catalina.core.ApplicationDispatcher.doForward(ApplicationDispatcher.java:391)
		at org.apache.catalina.core.ApplicationDispatcher.forward(ApplicationDispatcher.java:318)
		at org.apache.catalina.core.StandardHostValve.custom(StandardHostValve.java:423)
		at org.apache.catalina.core.StandardHostValve.status(StandardHostValve.java:291)
		at org.apache.catalina.core.StandardHostValve.throwable(StandardHostValve.java:383)
		at org.apache.catalina.core.StandardHostValve.invoke(StandardHostValve.java:166)
		at org.apache.catalina.valves.ErrorReportValve.invoke(ErrorReportValve.java:78)
		at org.apache.catalina.valves.AbstractAccessLogValve.invoke(AbstractAccessLogValve.java:610)
		at org.apache.catalina.core.StandardEngineValve.invoke(StandardEngineValve.java:88)
		at org.apache.catalina.connector.CoyoteAdapter.service(CoyoteAdapter.java:526)
		at org.apache.coyote.http11.AbstractHttp11Processor.process(AbstractHttp11Processor.java:1033)
		at org.apache.coyote.AbstractProtocol$AbstractConnectionHandler.process(AbstractProtocol.java:652)
		at org.apache.coyote.http11.Http11NioProtocol$Http11ConnectionHandler.process(Http11NioProtocol.java:222)
		at org.apache.tomcat.util.net.NioEndpoint$SocketProcessor.doRun(NioEndpoint.java:1566)
		at org.apache.tomcat.util.net.NioEndpoint$SocketProcessor.run(NioEndpoint.java:1523)
		at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1142)
		at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:617)
		at org.apache.tomcat.util.threads.TaskThread$WrappingRunnable.run(TaskThread.java:61)
		at java.lang.Thread.run(Thread.java:745)
	Caused by: java.util.NoSuchElementException: Unable to validate object
		at org.apache.commons.pool2.impl.GenericObjectPool.borrowObject(GenericObjectPool.java:497)
		at org.apache.commons.pool2.impl.GenericObjectPool.borrowObject(GenericObjectPool.java:360)
		at redis.clients.util.Pool.getResource(Pool.java:51)

从日志看起来跟上次的错误基本是一样的，就是在borrowObject的时候进行validate，然后发现validate failed。也就是说PING没有返回正常的PONG！
首先第一反应仍然是上去手动PING了一下发现并没有问题。INFO也一切正常。直接请求报错的接口也没有发现问题。

在集中式日志平台搜索了一下，发现错误时间是从前天早上11点25分开始，到昨天下午4点为止大概有2w左右的错误量。

结合我们做的事情，刚好我们在前天早上11点半的时候对这个Redis进行了迁移。因为要把这台机器（c32-07）下线，所有李贝在新机器m32-06 Copy了一个Redis，然后把c32-6由主库变成从库了。然后就发现不断有这样的错误报出来了。

那其实在上次出现拿不到连接validate出错之后，我就在线上增加了一个监控，会每隔5分钟PING一下我们线上的所有Redis主库，判断返回结果是不是PONG，如果不是就会短信和邮件报警。不过之前漏掉了这个Redis，因为这个是新闻推荐的Redis，之前一直是伟哥在维护，被我遗漏了。于是立即做了两件事情：

1. 将接口层的请求大部分切换到m32-06主库，只留下一台仍然访问c32-06从库。
2. 将c32-06加入到定时PING监控脚本中

时间大概是19号下午5点的样子。

然后今天果然有效果了：

1. 切换到主库之后，错误日志由2w变成2k。基本可以说明从库是有问题的。
2. 监控脚本检测到了PING c32-06从库返回错误信息：LOADING Redis is loading the dataset in memory。

也就是说从库确实在某些时间点是不可用的。

从错误信息`LOADING Redis is loading the dataset in memory`看来，是redis在加载RDB数据到内存中的过程是阻塞的，没法提供服务的，阻塞的时间取决于RDB的大小。理论上来说这个过程只会发生在

1. 启动的时候
2. slave全同步的时候。2.8之后理论上来说只是第一次同步，后面应该都是部分同步。

难道slave一直在全量同步？看了一下Redis的日志，很快就发现问题了。

这个是从库c32-7的redis日志：

	17859:S 18 Aug 11:26:16.363 # Server started, Redis version 3.0.0
	17859:S 18 Aug 11:26:16.363 # WARNING overcommit_memory is set to 0! Background save may fail under low memory condition. To fix this issue add 'vm.overcommit_memory = 1' to /etc/sysctl.conf and then reboot or run the command 'sysctl vm.overcommit_memory=1' for this to take effect.
	17859:S 18 Aug 11:26:52.802 * DB loaded from disk: 36.438 seconds
	17859:S 18 Aug 11:26:52.802 * The server is now ready to accept connections on port 6666
	17859:S 18 Aug 11:26:53.364 * Connecting to MASTER 10.242.73.46:6666
	17859:S 18 Aug 11:26:53.364 * MASTER <-> SLAVE sync started
	17859:S 18 Aug 11:26:53.364 * Non blocking connect for SYNC fired the event.
	17859:S 18 Aug 11:26:53.365 * Master replied to PING, replication can continue...
	17859:S 18 Aug 11:26:53.365 * Partial resynchronization not possible (no cached master)
	17859:S 18 Aug 11:26:53.365 * Full resync from master: 91fa2b65639c2ce2caf88d279aba53e8c4ad3a22:1
	17859:S 18 Aug 11:27:38.686 * MASTER <-> SLAVE sync: receiving 2550593979 bytes from master
	17859:S 18 Aug 11:27:59.262 * MASTER <-> SLAVE sync: Flushing old data
	17859:S 18 Aug 11:28:05.517 * MASTER <-> SLAVE sync: Loading DB in memory
	17859:S 18 Aug 11:28:27.393 * MASTER <-> SLAVE sync: Finished with success
	17859:S 18 Aug 12:51:54.051 # Connection with master lost.
	17859:S 18 Aug 12:51:54.052 * Caching the disconnected master state.
	17859:S 18 Aug 12:51:54.657 * Connecting to MASTER 10.242.73.46:6666
	17859:S 18 Aug 12:51:54.657 * MASTER <-> SLAVE sync started
	17859:S 18 Aug 12:51:54.657 * Non blocking connect for SYNC fired the event.
	17859:S 18 Aug 12:51:54.657 * Master replied to PING, replication can continue...
	17859:S 18 Aug 12:51:54.658 * Trying a partial resynchronization (request 91fa2b65639c2ce2caf88d279aba53e8c4ad3a22:1688184130).
	17859:S 18 Aug 12:51:54.658 * Full resync from master: 91fa2b65639c2ce2caf88d279aba53e8c4ad3a22:1934263988
	17859:S 18 Aug 12:51:54.658 * Discarding previously cached master state.
	17859:S 18 Aug 12:52:39.353 * MASTER <-> SLAVE sync: receiving 2550573652 bytes from master
	17859:S 18 Aug 12:52:56.391 # I/O error trying to sync with MASTER: connection lost
	17859:S 18 Aug 12:52:57.584 * Connecting to MASTER 10.242.73.46:6666
	17859:S 18 Aug 12:52:57.584 * MASTER <-> SLAVE sync started
	17859:S 18 Aug 12:52:57.584 * Non blocking connect for SYNC fired the event.
	17859:S 18 Aug 12:52:57.584 * Master replied to PING, replication can continue...
	17859:S 18 Aug 12:52:57.584 * Partial resynchronization not possible (no cached master)
	17859:S 18 Aug 12:52:57.585 * Full resync from master: 91fa2b65639c2ce2caf88d279aba53e8c4ad3a22:2022862913
	17859:S 18 Aug 12:53:42.309 * MASTER <-> SLAVE sync: receiving 2550573652 bytes from master
	17859:S 18 Aug 12:54:02.551 * MASTER <-> SLAVE sync: Flushing old data
	17859:S 18 Aug 12:54:09.241 * MASTER <-> SLAVE sync: Loading DB in memory
	17859:S 18 Aug 12:54:31.375 * MASTER <-> SLAVE sync: Finished with success
	17859:S 18 Aug 13:22:01.266 # Connection with master lost.
	17859:S 18 Aug 13:22:01.267 * Caching the disconnected master state.
	17859:S 18 Aug 13:22:02.340 * Connecting to MASTER 10.242.73.46:6666
	17859:S 18 Aug 13:22:02.341 * MASTER <-> SLAVE sync started
	17859:S 18 Aug 13:22:02.380 * Non blocking connect for SYNC fired the event.
	17859:S 18 Aug 13:22:02.404 * Master replied to PING, replication can continue...
	17859:S 18 Aug 13:22:02.404 * Trying a partial resynchronization (request 91fa2b65639c2ce2caf88d279aba53e8c4ad3a22:2548455709).
	17859:S 18 Aug 13:22:02.404 * Full resync from master: 91fa2b65639c2ce2caf88d279aba53e8c4ad3a22:2964290063
	17859:S 18 Aug 13:22:02.405 * Discarding previously cached master state.
	17859:S 18 Aug 13:22:47.186 * MASTER <-> SLAVE sync: receiving 2550563247 bytes from master
	17859:S 18 Aug 13:23:10.751 # I/O error trying to sync with MASTER: connection lost
	17859:S 18 Aug 13:23:11.097 * Connecting to MASTER 10.242.73.46:6666
	17859:S 18 Aug 13:23:11.097 * MASTER <-> SLAVE sync started
	17859:S 18 Aug 13:23:11.097 * Non blocking connect for SYNC fired the event.
	17859:S 18 Aug 13:23:11.097 * Master replied to PING, replication can continue...
	17859:S 18 Aug 13:23:11.097 * Partial resynchronization not possible (no cached master)
	17859:S 18 Aug 13:23:11.098 * Full resync from master: 91fa2b65639c2ce2caf88d279aba53e8c4ad3a22:3034293813
	17859:S 18 Aug 13:23:55.925 * MASTER <-> SLAVE sync: receiving 2550563310 bytes from master
	17859:S 18 Aug 13:24:16.412 * MASTER <-> SLAVE sync: Flushing old data
	17859:S 18 Aug 13:24:24.694 * MASTER <-> SLAVE sync: Loading DB in memory
	17859:S 18 Aug 13:24:47.453 * MASTER <-> SLAVE sync: Finished with success
	17859:S 18 Aug 13:52:27.452 # Connection with master lost.
	17859:S 18 Aug 13:52:27.452 * Caching the disconnected master state.
	17859:S 18 Aug 13:52:28.512 * Connecting to MASTER 10.242.73.46:6666
	17859:S 18 Aug 13:52:28.512 * MASTER <-> SLAVE sync started
	17859:S 18 Aug 13:52:28.574 * Non blocking connect for SYNC fired the event.
	17859:S 18 Aug 13:52:28.574 * Master replied to PING, replication can continue...
	17859:S 18 Aug 13:52:28.574 * Trying a partial resynchronization (request 91fa2b65639c2ce2caf88d279aba53e8c4ad3a22:3560083735).
	17859:S 18 Aug 13:52:28.575 * Full resync from master: 91fa2b65639c2ce2caf88d279aba53e8c4ad3a22:3994311298
	17859:S 18 Aug 13:52:28.575 * Discarding previously cached master state.
	17859:S 18 Aug 13:53:13.042 * MASTER <-> SLAVE sync: receiving 2550551123 bytes from master
	17859:S 18 Aug 13:53:33.314 * MASTER <-> SLAVE sync: Flushing old data
	17859:S 18 Aug 13:53:41.168 * MASTER <-> SLAVE sync: Loading DB in memory
	17859:S 18 Aug 13:54:03.683 * MASTER <-> SLAVE sync: Finished with success
	17859:S 18 Aug 13:54:03.688 # Connection with master lost.
	17859:S 18 Aug 13:54:03.688 * Caching the disconnected master state.
	17859:S 18 Aug 13:54:03.783 * Connecting to MASTER 10.242.73.46:6666
	17859:S 18 Aug 13:54:03.783 * MASTER <-> SLAVE sync started
	17859:S 18 Aug 13:54:03.783 * Non blocking connect for SYNC fired the event.
	17859:S 18 Aug 13:54:03.783 * Master replied to PING, replication can continue...
	17859:S 18 Aug 13:54:03.783 * Trying a partial resynchronization (request 91fa2b65639c2ce2caf88d279aba53e8c4ad3a22:3996662976).
	17859:S 18 Aug 13:54:03.783 * Full resync from master: 91fa2b65639c2ce2caf88d279aba53e8c4ad3a22:4045725312
	17859:S 18 Aug 13:54:03.783 * Discarding previously cached master state.
	17859:S 18 Aug 13:54:48.517 * MASTER <-> SLAVE sync: receiving 2550550079 bytes from master
	17859:S 18 Aug 13:55:08.913 * MASTER <-> SLAVE sync: Flushing old data
	17859:S 18 Aug 13:55:15.929 * MASTER <-> SLAVE sync: Loading DB in memory
	17859:S 18 Aug 13:55:38.210 * MASTER <-> SLAVE sync: Finished with success
	。。。


看到有很多的`Connection with master lost`的错误，然后slave尝试要求master进行部分同步：`Trying a partial resynchronization (request 91fa2b65639c2ce2caf88d279aba53e8c4ad3a22:3996662976)`。但是master还是要求全量同步：`Full resync from master: 91fa2b65639c2ce2caf88d279aba53e8c4ad3a22:4045725312`。然后就是我们熟悉的全量同步的过程了。上面的过程大概每个一分多钟就发生一次，一直重复着。

然后再看看主库m32-06的日志：

	12345:M 18 Aug 11:24:38.122 # Server started, Redis version 3.0.0
	12345:M 18 Aug 11:24:38.122 # WARNING overcommit_memory is set to 0! Background save may fail under low memory condition. To fix this issue add 'vm.overcommit_memory = 1' to /etc/sysctl.conf and then reboot or run the command 'sysctl vm.overcommit_memory=1' for this to take effect.
	12345:M 18 Aug 11:25:14.322 * DB loaded from disk: 36.200 seconds
	12345:M 18 Aug 11:25:14.322 * The server is now ready to accept connections on port 6666
	12345:M 18 Aug 11:26:53.366 * Slave 10.242.117.36:6666 asks for synchronization
	12345:M 18 Aug 11:26:53.366 * Full resync requested by slave 10.242.117.36:6666
	12345:M 18 Aug 11:26:53.366 * Starting BGSAVE for SYNC with target: disk
	12345:M 18 Aug 11:26:53.523 * Background saving started by pid 17147
	17147:C 18 Aug 11:27:38.393 * DB saved on disk
	17147:C 18 Aug 11:27:38.482 * RDB: 0 MB of memory used by copy-on-write
	12345:M 18 Aug 11:27:38.686 * Background saving terminated with success
	12345:M 18 Aug 11:27:58.746 * Synchronization with slave 10.242.117.36:6666 succeeded
	12345:M 18 Aug 12:51:54.071 # Client id=2 addr=10.242.117.36:21949 fd=5 name= age=5100 idle=0 flags=S db=0 sub=0 psub=0 multi=-1 qbuf=0 qbuf-free=0 obl=0 oll=12899 omem=268443524 events=rw cmd=replconf scheduled to be closed ASAP for overcoming of output buffer limits.
	12345:M 18 Aug 12:51:54.081 # Connection with slave 10.242.117.36:6666 lost.
	12345:M 18 Aug 12:51:54.700 * Slave 10.242.117.36:6666 asks for synchronization
	12345:M 18 Aug 12:51:54.700 * Unable to partial resync with slave 10.242.117.36:6666 for lack of backlog (Slave request was: 1688184130).
	12345:M 18 Aug 12:51:54.700 * Starting BGSAVE for SYNC with target: disk
	12345:M 18 Aug 12:51:54.865 * Background saving started by pid 15986
	15986:C 18 Aug 12:52:39.111 * DB saved on disk
	15986:C 18 Aug 12:52:39.201 * RDB: 474 MB of memory used by copy-on-write
	12345:M 18 Aug 12:52:39.396 * Background saving terminated with success
	12345:M 18 Aug 12:52:56.304 # Client id=17 addr=10.242.117.36:46501 fd=5 name= age=62 idle=62 flags=S db=0 sub=0 psub=0 multi=-1 qbuf=0 qbuf-free=0 obl=15882 oll=7597 omem=131532334 events=rw cmd=psync scheduled to be closed ASAP for overcoming of output buffer limits.
	12345:M 18 Aug 12:52:56.405 # Connection with slave 10.242.117.36:6666 lost.
	12345:M 18 Aug 12:52:57.627 * Slave 10.242.117.36:6666 asks for synchronization
	12345:M 18 Aug 12:52:57.627 * Full resync requested by slave 10.242.117.36:6666
	12345:M 18 Aug 12:52:57.627 * Starting BGSAVE for SYNC with target: disk
	12345:M 18 Aug 12:52:57.789 * Background saving started by pid 18402
	18402:C 18 Aug 12:53:42.084 * DB saved on disk
	18402:C 18 Aug 12:53:42.175 * RDB: 2 MB of memory used by copy-on-write
	12345:M 18 Aug 12:53:42.353 * Background saving terminated with success
	12345:M 18 Aug 12:54:02.103 * Synchronization with slave 10.242.117.36:6666 succeeded
	12345:M 18 Aug 13:21:59.734 # Client id=19 addr=10.242.117.36:46801 fd=5 name= age=1742 idle=0 flags=S db=0 sub=0 psub=0 multi=-1 qbuf=0 qbuf-free=0 obl=0 oll=12899 omem=268443524 events=rw cmd=replconf scheduled to be closed ASAP for overcoming of output buffer limits.
	12345:M 18 Aug 13:21:59.783 # Connection with slave 10.242.117.36:6666 lost.
	12345:M 18 Aug 13:22:02.430 * Slave 10.242.117.36:6666 asks for synchronization
	12345:M 18 Aug 13:22:02.430 * Unable to partial resync with slave 10.242.117.36:6666 for lack of backlog (Slave request was: 2548455709).
	12345:M 18 Aug 13:22:02.430 * Starting BGSAVE for SYNC with target: disk
	12345:M 18 Aug 13:22:02.602 * Background saving started by pid 31064
	31064:C 18 Aug 13:22:46.863 * DB saved on disk
	31064:C 18 Aug 13:22:46.954 * RDB: 444 MB of memory used by copy-on-write
	12345:M 18 Aug 13:22:47.138 * Background saving terminated with success
	12345:M 18 Aug 13:23:10.557 # Client id=26 addr=10.242.117.36:55208 fd=5 name= age=68 idle=68 flags=S db=0 sub=0 psub=0 multi=-1 qbuf=0 qbuf-free=0 obl=15411 oll=6422 omem=107089538 events=rw cmd=psync scheduled to be closed ASAP for overcoming of output buffer limits.
	12345:M 18 Aug 13:23:10.657 # Connection with slave 10.242.117.36:6666 lost.
	12345:M 18 Aug 13:23:11.121 * Slave 10.242.117.36:6666 asks for synchronization
	12345:M 18 Aug 13:23:11.121 * Full resync requested by slave 10.242.117.36:6666
	12345:M 18 Aug 13:23:11.121 * Starting BGSAVE for SYNC with target: disk
	12345:M 18 Aug 13:23:11.288 * Background saving started by pid 1363
	1363:C 18 Aug 13:23:55.725 * DB saved on disk
	1363:C 18 Aug 13:23:55.818 * RDB: 1 MB of memory used by copy-on-write
	12345:M 18 Aug 13:23:55.948 * Background saving terminated with success
	12345:M 18 Aug 13:24:15.940 * Synchronization with slave 10.242.117.36:6666 succeeded
	12345:M 18 Aug 13:52:25.584 # Client id=28 addr=10.242.117.36:55550 fd=5 name= age=1754 idle=0 flags=S db=0 sub=0 psub=0 multi=-1 qbuf=0 qbuf-free=0 obl=0 oll=12899 omem=268443524 events=rw cmd=replconf scheduled to be closed ASAP for overcoming of output buffer limits.
	12345:M 18 Aug 13:52:25.593 # Connection with slave 10.242.117.36:6666 lost.
	12345:M 18 Aug 13:52:28.577 * Slave 10.242.117.36:6666 asks for synchronization
	12345:M 18 Aug 13:52:28.577 * Unable to partial resync with slave 10.242.117.36:6666 for lack of backlog (Slave request was: 3560083735).
	12345:M 18 Aug 13:52:28.577 * Starting BGSAVE for SYNC with target: disk
	12345:M 18 Aug 13:52:28.750 * Background saving started by pid 14284
	14284:C 18 Aug 13:53:12.715 * DB saved on disk
	14284:C 18 Aug 13:53:12.807 * RDB: 344 MB of memory used by copy-on-write
	12345:M 18 Aug 13:53:13.044 * Background saving terminated with success
	12345:M 18 Aug 13:53:32.805 * Synchronization with slave 10.242.117.36:6666 succeeded
	12345:M 18 Aug 13:53:34.763 # Client id=35 addr=10.242.117.36:12979 fd=5 name= age=66 idle=1 flags=S db=0 sub=0 psub=0 multi=-1 qbuf=0 qbuf-free=0 obl=0 oll=4991 omem=77418232 events=rw cmd=psync scheduled to be closed ASAP for overcoming of output buffer limits.
	12345:M 18 Aug 13:53:34.864 # Connection with slave 10.242.117.36:6666 lost.
	12345:M 18 Aug 13:54:03.785 * Slave 10.242.117.36:6666 asks for synchronization
	12345:M 18 Aug 13:54:03.785 * Unable to partial resync with slave 10.242.117.36:6666 for lack of backlog (Slave request was: 3996662976).
	12345:M 18 Aug 13:54:03.785 * Starting BGSAVE for SYNC with target: disk
	12345:M 18 Aug 13:54:03.952 * Background saving started by pid 18056
	18056:C 18 Aug 13:54:48.271 * DB saved on disk
	18056:C 18 Aug 13:54:48.363 * RDB: 2 MB of memory used by copy-on-write
	12345:M 18 Aug 13:54:48.518 * Background saving terminated with success
	12345:M 18 Aug 13:55:08.460 * Synchronization with slave 10.242.117.36:6666 succeeded


可以看到跟slave的日志非常对应：

	12345:M 18 Aug 12:51:54.081 # Connection with slave 10.242.117.36:6666 lost.
	12345:M 18 Aug 12:51:54.700 * Slave 10.242.117.36:6666 asks for synchronization
	12345:M 18 Aug 12:51:54.700 * Unable to partial resync with slave 10.242.117.36:6666 for lack of backlog (Slave request was: 1688184130).
	12345:M 18 Aug 12:51:54.700 * Starting BGSAVE for SYNC with target: disk
	12345:M 18 Aug 12:51:54.865 * Background saving started by pid 15986
	15986:C 18 Aug 12:52:39.111 * DB saved on disk
	15986:C 18 Aug 12:52:39.201 * RDB: 474 MB of memory used by copy-on-write
	12345:M 18 Aug 12:52:39.396 * Background saving terminated with success
	12345:M 18 Aug 12:52:56.304 # Client id=17 addr=10.242.117.36:46501 fd=5 name= age=62 idle=62 flags=S db=0 sub=0 psub=0 multi=-1 qbuf=0 qbuf-free=0 obl=15882 oll=7597 omem=131532334 events=rw cmd=psync scheduled to be closed ASAP for overcoming of output buffer limits.

而且从日志我们可以知道为什么主库不能对从库进行部分同步：`Unable to partial resync with slave 10.242.117.36:6666 for lack of backlog (Slave request was: 1688184130)`。这个涉及到Redis的部分同步的实现。在2.8版本，redis使用了新的复制方式，引入了replication backlog以支持部分同步。

> The master then starts background saving, and starts to buffer all new commands received that will modify the dataset. When the background saving is complete, the master transfers the database file to the slave, which saves it on disk, and then loads it into memory. The master will then send to the slave all buffered commands. This is done as a stream of commands and is in the same format of the Redis protocol itself.

当主服务器进行命令传播的时候，maser不仅将所有的数据更新命令发送到所有slave的replication buffer，还会写入replication backlog。当断开的slave重新连接上master的时候，slave将会发送psync命令（包含复制的偏移量offset），请求partial resync。如果请求的offset不存在，那么执行全量的sync操作，相当于重新建立主从复制。

replication backlog是一个环形缓冲区，整个master进程中只会存在一个，所有的slave公用。backlog的大小通过repl-backlog-size参数设置，默认大小是1M，其大小可以根据每秒产生的命令、（master执行rdb bgsave） +（master发送rdb到slave） + （slave load rdb文件）时间之和来估算积压缓冲区的大小，repl-backlog-size值不小于这两者的乘积。

所以在主从同步的时候，slave会落后master的时间 ＝（master执行rdb bgsave）+ (master发送rdb到slave) + (slave load rdb文件) 的时间之和。
然后如果在这个时间master的数据变更非常巨大，超过了replication backlog，那么老的数据变更命令就会被丢弃，导致需要全量同步。

那么我们这个Redis的数据变更量有多大呢？这个可以大概从redis的日志猜测出来：`RDB: 474 MB of memory used by copy-on-write`。发现这个COW的内存大小变化挺大的，有时候是2M
，有时候能够达到1414 MB。而整个Redis数据内存大小也才10G左右。这么大的变更量，默认的1M repl-backlog-size根本不够用。

那现在明白了为什么每次都全量同步的原因了，剩下最根本的问题，就是主从链接为什么会每隔1分钟左右就断开一次，导致需要重新同步？

从日志还是能够看出一些端倪：

	Client id=17 addr=10.242.117.36:46501 fd=5 name= age=62 idle=62 flags=S db=0 sub=0 psub=0 multi=-1 qbuf=0 qbuf-free=0 obl=15882 oll=7597 omem=131532334 events=rw cmd=psync scheduled to be closed ASAP for overcoming of output buffer limits.

注意到这么一句话：`psync scheduled to be closed ASAP for overcoming of output buffer limits`。看起来是psync因为超过output buffer limits将被close。

谷歌一下，发现有挺多人遇到一样的问题：[redis 2.8 psync #1400](https://github.com/antirez/redis/issues/1400)。然后设置

	set "client-output-buffer-limit slave 0 0 0 

就没有问题了。

于是查看了一下`client-output-buffer-limit`。发现这是Redis的一个保护机制。配置格式是：
	
	client-output-buffer-limit <class> <hard limit> <soft limit> <soft seconds>

具体参数含义如下：

* class: 客户端种类，包括Normal，Slaves和Pub/Sub
	* Normal: 普通的客户端。默认limit 是0，也就是不限制。
	* Pub/Sub: 发布与订阅的客户端的。默认hard limit 32M，soft limit 8M/60s。
	* Slaves: 从库的复制客户端。默认hard limit 256M，soft limit 64M/60s。
* hard limit: 缓冲区大小的硬性限制。
* soft limit: 缓冲去大小的软性限制。
* soft seconds: 缓冲区大小达到了（超过）soft limit值的持续时间。

client-output-buffer-limit参数限制分配的缓冲区的大小，防止内存无节制的分配，Redis将会做如下自我保护：

1. client buffer的大小达到了soft limit并持续了soft seconds时间，将立即断开和客户端的连接
2. client buffer的大小达到了hard limit，server也会立即断开和客户端的连接

再看看我们从库的这个配置，其实就是默认配置：

	# 客户端的输出缓冲区的限制，因为某种原因客户端从服务器读取数据的速度不够快，
	# 可用于强制断开连接（一个常见的原因是一个发布 / 订阅客户端消费消息的速度无法赶上生产它们的速度）。
	#  可以三种不同客户端的方式进行设置：
	# normal ->  正常客户端
	# slave  -> slave 和 MONITOR 客户端
	# pubsub ->  至少订阅了一个 pubsub channel 或 pattern 的客户端
	#  每个 client-output-buffer-limit 语法 :
	# client-output-buffer-limit <class><hard limit> <soft limit> <soft seconds>
	#  一旦达到硬限制客户端会立即断开，或者达到软限制并保持达成的指定秒数（连续）。
	#  例如，如果硬限制为 32 兆字节和软限制为 16 兆字节 /10 秒，客户端将会立即断开
	#  如果输出缓冲区的大小达到 32 兆字节，客户端达到 16 兆字节和连续超过了限制 10 秒，也将断开连接。
	#  默认 normal 客户端不做限制，因为他们在一个请求后未要求时（以推的方式）不接收数据，
	#  只有异步客户端可能会出现请求数据的速度比它可以读取的速度快的场景。
	#  把硬限制和软限制都设置为 0 来禁用该特性
	client-output-buffer-limit normal 0 0 0
	client-output-buffer-limit slave 256mb 64mb 60
	client-output-buffer-limit pubsub 32mb 8mb 60


redis的replication buffer其实就是client buffer的一种。里面存放的数据是下面三个时间内所有的master数据更新操作：

1. master执行rdb bgsave产生snapshot的时间
2. master发送rdb到slave网络传输时间
3. slave load rdb文件把数据恢复到内存的时间

可以看到跟replication backlog是一模一样的！

replication buffer由client-output-buffer-limit slave设置，当这个值太小会导致主从复制链接断开:

1. 当master-slave复制连接断开，server端会释放连接相关的数据结构。replication buffer中的数据也就丢失了，此时主从之间重新开始复制过程。
2. 还有个更严重的问题，主从复制连接断开，导致主从上出现rdb bgsave和rdb重传操作无限循环。

看起来确实server(这里就是master)会因为缓冲区的大小问题主动关闭客户端(slave)链接。因为我们的数据变更量太大，超过了client-output-buffer-limit。导致主从同步连接被断开，然后slave要求psync，但是由于repl-backlog-size太小，导致psync失败，需要full sync，而full sync需要Discarding previously cached master state，重新load RDB文件到内存，而这个加载数据过程是阻塞式的。所以导致slave出现间歇式的不可用。而切换到master之后，master的整个同步操作都是fork一个子进程进行的，所以不影响父进程继续服务。所有的现象都能清清楚楚的解释上。

从这个问题我们可以发现其实Redis的主从同步非常依赖于两个参数的合理配置：

1. client-output-buffer-limit
2. repl-backlog-size

真的要遇到问题，才能够深入的了解底层的实现机制啊。

--EOF--

参考和推荐文章
------------

1. [Redis 3.0 中文版 - 复制](http://wiki.jikexueyuan.com/project/redis-guide/replication.html)
2. [Redis Clients Handling-Output buffers limits](http://redis.io/topics/clients#output-buffers-limits)
3. [redis主从复制（2）— replication buffer与replication backlog](http://mdba.cn/?p=804) 非常简洁明了的文章，强烈推荐！
4. [redis主从复制（4）— client buffer](http://mdba.cn/?p=833)
5. [Top Redis Headaches for Devops – Replication Buffer - See more at: https://redislabs.com/blog/top-redis-headaches-for-devops-replication-buffer#sthash.VO8bl2Ul.dpuf](https://redislabs.com/blog/top-redis-headaches-for-devops-replication-buffer)
6. [Top Redis Headaches for Devops – Client Buffers](https://redislabs.com/blog/top-redis-headaches-for-devops-client-buffers)
