---
title: Redis的事务
layout: post
---


Redis提供了一个乐观锁的事务机制，WATCH-MULTI-EXEC。其作用非常类似于DB的check-and-set(CAS)乐观锁实现：
	
	boolean success = false;
	do{
		y = exec_sql("elect a from xxx");
		x = y + n;
		success = exec_sql("udpate a = x where a = y");
	} while(!success)



同样的，在Redis中，我们可以这样子实现：

	boolean success = false;
	do{
		redis.watch('a');
		y = redis.get('a');
		redis.multi();
		redis.set('a', y + n); // QUEUED
		success = redis.exec(); // failure if a has changed and was watched.
	}while(!success)

With a WATCH clause, the commands between MULTI and EXEC are executed only if the value has not changed.


值得注意的是Redis并不支持回滚，虽然它有一个看起来像是回滚的命令：DISCARD。但是这个命令的作用是丢弃那些还没有执行(queued commands)的命令，而不是已经执行的命令。


**TIPS** 另一种事务方式是使用脚本。


### 参考文章

1. [Transactions](http://redis.io/topics/transactions)