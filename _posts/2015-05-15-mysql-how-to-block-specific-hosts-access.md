---
title: 如何限制某个IP对MySQL的访问
layout: post
---

背景
----

昨天下午发布完不久突然收到很多报警邮件，线上所有接口都报处理超时，基本访问不了。查看tomcat日志，发现大量的获取DB链接timeout错误：

	2015-05-14 21:28:21,578 [ERROR] [http-nio-8092-exec-155] me.aranzheng.study.nantianmen.web.ManagementController (ManagementController.java:567): has error
	org.mybatis.spring.MyBatisSystemException: nested exception is org.apache.ibatis.exceptions.PersistenceException:
	### Error querying database.  Cause: org.springframework.jdbc.CannotGetJdbcConnectionException: Could not get JDBC Connection; nested exception is java.sql.SQLException: An attempt by a client to checkout a Connection has timed out.
	### The error may exist in com/me/arganzheng/study/dataaccess/rule/t5/t5-rule.xml
	### The error may involve me.aranzheng.study.dataaccess.rule.t5.dao.T5RuleMapper.queryRuleByDimens
	        at org.apache.ibatis.executor.SimpleExecutor.doQuery(SimpleExecutor.java:56)
	        at org.apache.ibatis.executor.BaseExecutor.queryFromDatabase(BaseExecutor.java:259)
	        at org.apache.ibatis.executor.BaseExecutor.query(BaseExecutor.java:132)
	        at org.apache.ibatis.executor.CachingExecutor.query(CachingExecutor.java:105)
	        at org.apache.ibatis.executor.CachingExecutor.query(CachingExecutor.java:81)
	        at org.apache.ibatis.session.defaults.DefaultSqlSession.selectList(DefaultSqlSession.java:104)
	        ... 57 more
	Caused by: java.sql.SQLException: An attempt by a client to checkout a Connection has timed out.
	        at com.mchange.v2.sql.SqlUtils.toSQLException(SqlUtils.java:106)
	        at com.mchange.v2.sql.SqlUtils.toSQLException(SqlUtils.java:65)
	        at com.mchange.v2.c3p0.impl.C3P0PooledConnectionPool.checkoutPooledConnection(C3P0PooledConnectionPool.java:527)
	        at com.mchange.v2.c3p0.impl.AbstractPoolBackedDataSource.getConnection(AbstractPoolBackedDataSource.java:128)
	        at org.springframework.jdbc.datasource.DataSourceUtils.doGetConnection(DataSourceUtils.java:111)
	        at org.springframework.jdbc.datasource.DataSourceUtils.getConnection(DataSourceUtils.java:77)
	        ... 67 more
	Caused by: com.mchange.v2.resourcepool.TimeoutException: A client timed out while waiting to acquire a resource from com.mchange.v2.resourcepool.BasicResourcePool@4e85a959 -- time
	out at awaitAvailable()
	        at com.mchange.v2.resourcepool.BasicResourcePool.awaitAvailable(BasicResourcePool.java:1317)
	        at com.mchange.v2.resourcepool.BasicResourcePool.prelimCheckoutResource(BasicResourcePool.java:557)
	        at com.mchange.v2.resourcepool.BasicResourcePool.checkoutResource(BasicResourcePool.java:477)
	        at com.mchange.v2.c3p0.impl.C3P0PooledConnectionPool.checkoutPooledConnection(C3P0PooledConnectionPool.java:525)
	        ... 70 more

然后 jstack 查看发现tomcat的所有线程都处于等待状态，处于卡死状态。

日志显示是数据库链接问题，所以查看tomcat连接的数据库连接，是在m32-08机器，登陆上去，top了一下，发现mysqld的CPU占用率是1400%！明显mysql有问题。查看mysql-slow.log发现滚动的很厉害。怀疑是发布的新代码导致，但是通过slow log看不出是哪个日志，因为太多了。于是使用show processlist;查看当前的执行情况，一下子就发现问题了：

一共有423个process在执行中，其中有几次进程执行时间非常明显，在180s~380s之间，一看发现是主机地址不是我们团队的机器，是外部团队的。大群里询问了一下，但是没有人回复。于是打算先把这个IP地址的访问限制掉。想到两种方式：一是通过MySQL的权限控制，而是在机器层面直接iptable限制。但是汗颜的是不熟悉命令，不知道怎么操作。。。在那里谷歌半天。。就在这个时候群里有人回复了，机器是来自于北京团队的。我们的推荐新闻数据来自于北京数据挖掘部门的分析，然后采用的是直接将数据库暴露给他们，让他们查询。本来他们的压力并不大，3s查询一次增量，每次执行时间大概2s左右。后来发现是他们调整了拉取的SQL语句，导致索引失效，从而走全表扫描了。

立即让北京团队同学停止了拉取程序，服务瞬间恢复了。然后重新搭建了一个从库给他们单独使用，避免影响我们自己的服务。

这事情其实是结束了，但是确给我留下了两个教训。

1. 不要和外部系统共用一个数据库。因为如果他出问题，就会影响到你的服务，而且这是不可控的。至少也不用共用一个账户密码，要不直接改密码就可以了。可惜我们都没有想到。。
2. 书到用时方恨少。想想如果北京团队一直没有人回复，然后我们这边一直没有找到屏蔽IP访问的操作方法，故障时间就加长了。因为共用账号了，所以改账户密码又需要修改和发布所有线上服务。

所以今天抽时间学习怎样限制某个主机对MySQL的访问。下次就不会这么手忙脚乱了。


如何限制某个主机对MySQL的访问
--------------------------

### 法一：临时方案，直接kill

	mysql> show processlist;

	+----+-------------+--------------------+----------+---------+------+-------+------------------+
	| Id | User        | Host               | db       | Command | Time | State | Info             |
	+----+-------------+--------------------+----------+---------+------+-------+------------------+
	| 49 | application | 192.168.44.1:51718 | XXXXXXXX | Sleep   |  183 |       | NULL             ||
	| 55 | application | 192.168.44.1:51769 | XXXXXXXX | Sleep   |  148 |       | NULL             |
	| 56 | application | 192.168.44.1:51770 | XXXXXXXX | Sleep   |  148 |       | NULL             |
	| 57 | application | 192.168.44.1:51771 | XXXXXXXX | Sleep   |  148 |       | NULL             |
	| 58 | application | 192.168.44.1:51968 | XXXXXXXX | Sleep   |   11 |       | NULL             |
	| 59 | root        | localhost          | NULL     | Query   |    0 | NULL  | show processlist |
	+----+-------------+--------------------+----------+---------+------+-------+--------------

	mysql> kill 52;
	Query OK, 0 rows affected (0.00 sec)

但是很快就会上来的。所以这个其实没有什么用。

### 法二：使用MySQL GRANT

	mysql> GRANT USAGE ON *.* TO 'user'@'<blockIP>';

### 法三：使用iptable

这个比较复杂，而且因为是机器层面的，所以要特别小心。

有两种方式，一种是只允许某些IP，其他的都block，即白名单的方式：

> For this example, I’m going to allow access from 3 servers, and deny every other address.  Execute the following commands in order, and replace IP address 1, 2, and 3 with the addresses of your servers.
> 
> 	  iptables -A INPUT -p tcp -s IPaddress1/32 --dport 3306 -j ACCEPT
>	  iptables -A INPUT -p tcp -s IPaddress2/32 --dport 3306 -j ACCEPT
>	  iptables -A INPUT -p tcp -s IPaddress3/32 --dport 3306 -j ACCEPT
>	  iptables -A INPUT -p tcp --dport 3306 -j DROP
>
> When configuring iptables, order is important.  The effects of each command are immediate, but you must save your iptables settings or you’ll lose them on your next restart.  Do this by issuing the following command:
>
>     service iptables save
>
> You iptable settings are now saved, and will be used next time iptables is restarted.  By default in Red Hat, and centOS, the settings are stored here: /etc/sysconfig/iptables

另一种反过来，不允许某些IP，其他的都允许，即设置黑名单：

	iptables -A INPUT -p tcp -s IPaddress1/32 --dport 3306 -j DROP
	iptables -A INPUT -p tcp -s IPaddress2/32 --dport 3306 -j DROP
	iptables -A INPUT -p tcp -s IPaddress3/32 --dport 3306 -j DROP
	iptables -A INPUT -p tcp --dport 3306 -j ACCEPT


补充
----

### 如何查看MySQL状态


#### 1. [SHOW VARIABLES](http://dev.mysql.com/doc/refman/5.1/en/show-variables.html)

	SHOW [GLOBAL | SESSION] VARIABLES [LIKE 'pattern' | WHERE expr]


SHOW VARIABLES显示MySQL的系统变量，一般是配置信息。具体参见: [Server System Variables](http://dev.mysql.com/doc/refman/5.1/en/server-system-variables.html)。

### 2. [SHOW STATUS](http://dev.mysql.com/doc/refman/5.1/en/show-status.html)

	SHOW [GLOBAL | SESSION] STATUS [LIKE 'pattern' | WHERE expr]

SHOW STATUS显示MySQL的状态信息，一般是配置项对应的当前值。比如针对线程池的配置项thread_cache_size（通过show variables可以得到），我们可以通过Threads_cached，Threads_created变量知道当前线程池的状态。具体参见: [Server Status Variables](http://dev.mysql.com/doc/refman/5.1/en/server-status-variables.html)。

**NOTES**

如果没有指定GLOBAL或者SESSION限制符，那么默认是使用SESSION。GLOBAL变量和SESSION状态是有差别的，GLOBAL是所有SESSION(Connections)的汇集统计，而SESSION只显示当前connection的状态。具体可以参考: [Difference between show status and show global status in mysql](http://dba.stackexchange.com/questions/54337/difference-between-show-status-and-show-global-status-in-mysql)。


#### 3. [SHOW ENGINE Syntax](http://dev.mysql.com/doc/refman/5.1/en/show-engine.html)

	SHOW ENGINE engine_name {STATUS | MUTEX}

可以知道具体的存储引擎的状态参数。比如我们最常用的INNODB:

	mysql> show engine innodb status\G


#### 4. [SHOW PROCESSLIST](http://dev.mysql.com/doc/refman/5.1/en/show-processlist.html)

	SHOW [FULL] PROCESSLIST

这个命令非常有用。它可以让我们知道当前有多少个线程，在处理什么连接（MySQL是典型的[One thread per connection](https://dev.mysql.com/doc/refman/5.1/en/connection-threads.html)的线程处理模型）。

**TIPS**

1. 如果没有指定FULL限定词，那么INFO列只会展示100个字符。
2. 可以通过kill <tid> kill其中的线程。


### MySQL的线程模型

具体参见：[MySQL Threads](http://dev.mysql.com/doc/refman/5.1/en/mysql-threads.html)


参考文章
-------

1. [How do I kill all the processes in Mysql “show processlist”?](http://stackoverflow.com/questions/1903838/how-do-i-kill-all-the-processes-in-mysql-show-processlist)
2. [How do I manually block and then unblock a specific IP/Hostname in MYSQL](http://stackoverflow.com/questions/17621710/how-do-i-manually-block-and-then-unblock-a-specific-ip-hostname-in-mysql)
3. [Restricting mySQL access to specific IP addresses](http://www.creativeintensity.com/index.php/allowing-mysql-access-from-specific-ip-addresses-only-in-linux/)
4. [Block MySQL port for everyone except localhost and a single IP](https://snipt.net/johan_adriaans/block-mysql-port-for-everyone-except-localhost-and-a-single-ip/)

