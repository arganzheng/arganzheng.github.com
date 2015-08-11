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

--EOF--



