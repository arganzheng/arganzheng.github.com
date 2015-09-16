---
title: 配置MySQL Slave
layout: post
---


需求
----

已经有一主一备，需要再增加一个只读备机。


步骤
----

### 1. 安装mysql

最好找个跟Master一样的MySQL版本（特别是主版本号）。基本不需要源码安装，到官网找到相应平台编译好的二进制包就可以了。

具体参考: [2.2 Installing MySQL on Unix/Linux Using Generic Binaries](http://dev.mysql.com/doc/refman/5.1/en/binary-installation.html)。可以不建立mysql组和用户。

	shell> tar zxvf /path/to/mysql-VERSION-OS.tar.gz
	shell> cd /path/to/mysql-VERSION-OS
	shell> scripts/mysql_install_db --basedir=/path/to/mysql-VERSION-OS --datadir=/path/to/mysql-VERSION-OS/data
	# Next command is optional
	shell> cp support-files/my-medium.cnf /etc/my.cnf 
	# 根据需要编辑 /etc/my.cnf 
	shell> bin/mysqld_safe &
	# Next command is optional
	shell> cp support-files/mysql.server /etc/init.d/mysql.server	

### 2. 用户权限配置

将原有的slave的权限copy过来就可以了。可以使用下面这个脚本将备机的权限导出：

	MYSQL_CONN="-uroot -pXXXXX"
	mysql ${MYSQL_CONN} --skip-column-names -A -e"SELECT CONCAT('SHOW GRANTS FOR ''',user,'''@''',host,''';') FROM mysql.user WHERE user<>''" | mysql ${MYSQL_CONN} --skip-column-names -A | sed 's/$/;/g' > MySQLUserGrants.sql

然后将其导入到新的slave:

	mysql -uroot -p -A < MySQLUserGrants.sql	


### 3. 数据迁移和主从配置

现在要把Master的数据迁移到Slave中。google一下有很多介绍的文章，基本都是一个思路：先用mysqldump把master某个时刻的数据给全量dump下来，然后导入到slave中。再配置主从关系，让slave从dump的那个时刻开始追赶。推荐这篇文章：[Setting up MySQL replication without the downtime](http://plusbryan.com/mysql-replication-without-downtime) 。可以不用读锁，基本对线上没有任何影响。mysqldump的时候可能会报mysql.event表dump失败，无关紧要。

具体过程如下：

下面语句创建一个带有binlog位置的备份文件，dump的过程中会影响DB性能，但是不会锁表(--single-transaction只对InnoDB有效)：

	mysqldump --skip-lock-tables --single-transaction --flush-logs --hex-blob --master-data=2 -A  > ~/dump.sql

由于带了`--master-data`选项，所以dump.sql文件前面含有binlog位置信息，可以用如下命令找到：

	head dump.sql -n80 | grep "MASTER_LOG_POS"

然后我们需要將dump.sql文件scp到备库所在机器，由于这个文件一般很大，而且是文本文件，最好先压缩一下：

	gzip ~/dump.sql

再scp过去:

	scp ~/dump.sql.gz mysql-user@<<slave-server-ip>>:~/

然后配置一下slave:

	#*******************************  Replication related settings **********************
	server-id 	= 30
	relay-log-index = copyer1-relay-bin.index
	relay-log	= copyer1-relay-bin
	log-bin=mysql-bin
	binlog_format=mixed
	binlog_cache_size = 1M
	read-only  = 1 

重启，导入备份文件：

	gunzip ~/dump.sql.gz
	mysql -u root -p < ~/dump.sql

然后设置主从关系：

	mysql> CHANGE MASTER TO MASTER_HOST='xxxx',MASTER_USER='xxx', MASTER_PASSWORD='xxx', MASTER_LOG_FILE='mysql-bin.000005', MASTER_LOG_POS=3119;

其中 MASTER_LOG_FILE='mysql-bin.000005', MASTER_LOG_POS=3119 在前面dump.sql文件中可以得到。
然后启动slave同步：

	mysql> start slave;


Relay Server
------------

在前面我们讨论过怎样搭建一个只读从库，现在我们来讨论一种特殊的从库——relay server。它原来的作用只是用来做日志中继，减轻master推送日志给太大slave的压力。

经典的配置是将relay server的storage_engine设置为BLACKHOLE。这个存储引擎只会记录binlog，但是不会真正执行。所以replicate的速度会比普通的slave快一些。具体参考《MySQL High Availability》P159的《Hierarchal Replication》。

有一点要注意的是就是正常的slave是不会记录master的binlog到自己的binlog的，这样在master上的操作就无法传播（中继）给下面的slave。如果要中继，那么就需要配置一下`log_slave_updates`：

	#*******************************  Replication related settings **********************
	server-id 	= 2
	relay-log	= copyer1-relay-bin
	relay-log-index = copyer1-relay-bin.index
	log-bin=master-bin
	log-bin-index=master-bin.index
	binlog_format=mixed
	log_slave_updates
	binlog_cache_size = 1M
	read-only  = 0

**TIPS**

relay server对于机房搬迁，需要切换master，非常有用。可以直接将应用切换到relay server，再讲relay server和老master同步关系断掉。这样relay server就是唯一的master了。

	mysql> stop slave;
	mysql> change master to MASTER_HOST='';
	mysql> reset slave;


但是要注意的是，如果你不是所有的slave都是链接relay，而是有一些slave也连接到master，那么在切换之前，需要先讲这些slave的主从关系也切换到relay。这是一个比较蛋疼的问题，因为master和relay的binlog日志内容是一样的，但是位置却是不一样的。所以要找到slave最后一次从master同步到的日志在relay的binlog中的位置。然后从这个位置开始追过relay。具体步骤如下：

1、停止slave和master的主从关系：slave> stop slave;

2、查看slave最后一条同步日志（可以在第一步停止主从之前先show slave status大概知道在哪个文件）：

	$ bin/mysqlbinlog data/copyer1-relay-bin.000026 | tail -n20

	# at 503974180
	#150424 11:11:46 server id 6  end_log_pos 825520049 	Xid = 5020740988
	COMMIT/*!*/;
	# at 503974207
	#150424 11:11:47 server id 6  end_log_pos 825520120 	Query	thread_id=52352340	exec_time=0	error_code=0
	SET TIMESTAMP=1429845107/*!*/;
	BEGIN
	/*!*/;
	# at 503974278
	#150424 11:11:47 server id 6  end_log_pos 825520368 	Query	thread_id=52352340	exec_time=0	error_code=0
	SET TIMESTAMP=1429845107/*!*/;
	UPDATE QRTZ_SCHEDULER_STATE SET LAST_CHECKIN_TIME = 1429845107658 WHERE SCHED_NAME = 'org.springframework.scheduling.quartz.SchedulerFactoryBean#0' AND INSTANCE_NAME = '10.240.35.46'
	/*!*/;
	# at 503974526
	#150424 11:11:47 server id 6  end_log_pos 825520395 	Xid = 5020740994
	COMMIT/*!*/;
	DELIMITER ;
	# End of log file
	ROLLBACK /* added by mysqlbinlog */;
	/*!50003 SET COMPLETION_TYPE=@OLD_COMPLETION_TYPE*/;


3、根据日志的内容去relay中查找相应的位置，正常来说slave和relay应该都基本同步的，所以找最后一个binlog文件一般就能找到：

	$ bin/mysqlbinlog data/master-bin.000014 | grep "UPDATE QRTZ_SCHEDULER_STATE SET LAST_CHECKIN_TIME = 1429845107658 WHERE SCHED_NAME = 'org.springframework.scheduling.quartz.SchedulerFactoryBean#0' AND INSTANCE_NAME = '10.240.35.46'" --context 10
	#150424 11:11:46 server id 6  end_log_pos 353896351 	Xid = 13198964
	COMMIT/*!*/;
	# at 353896351
	#150424 11:11:47 server id 6  end_log_pos 353896410 	Query	thread_id=52352340	exec_time=0	error_code=0
	SET TIMESTAMP=1429845107/*!*/;
	BEGIN
	/*!*/;
	# at 353896410
	#150424 11:11:47 server id 6  end_log_pos 353896658 	Query	thread_id=52352340	exec_time=0	error_code=0
	SET TIMESTAMP=1429845107/*!*/;
	UPDATE QRTZ_SCHEDULER_STATE SET LAST_CHECKIN_TIME = 1429845107658 WHERE SCHED_NAME = 'org.springframework.scheduling.quartz.SchedulerFactoryBean#0' AND INSTANCE_NAME = '10.240.35.46'
	/*!*/;
	# at 353896658
	#150424 11:11:47 server id 6  end_log_pos 353896685 	Xid = 13198966
	COMMIT/*!*/;
	# at 353896685
	#150424 11:11:49 server id 6  end_log_pos 353896744 	Query	thread_id=52348283	exec_time=0	error_code=0
	SET TIMESTAMP=1429845109/*!*/;
	BEGIN
	/*!*/;
	# at 353896744

4、设置slave指向relay作为master：

	slave> change master to MASTER_HOST='';
	slave> reset slave;

	slave> CHANGE MASTER TO MASTER_HOST='relayIP',MASTER_USER='xxx', MASTER_PASSWORD='xxx', MASTER_LOG_FILE='master-bin.000014', MASTER_LOG_POS=353896744;
	slave> START slave;
	slave> SHOW SLAVE　STATUS;


**NOTES** 关于MySQL的BlackHold存储引擎

MySQL的BlackHold存储引擎，看起来就是做relay server的最好选择，因为它不存储数据，只是记录日志。这样子相当于SQL线程压力大大减轻。但是可惜的是[黑洞存储引擎有一定的限制](https://dev.mysql.com/doc/refman/5.6/en/replication-features-blackhole.html)，它只适用于statement-based-replication的主从同步方式，不适合Row或者Mixed方式的主从同步。因为这种情况下updates和deletes会被简单的忽略。

参考文章
--------

1. [2.2 Installing MySQL on Unix/Linux Using Generic Binaries](http://dev.mysql.com/doc/refman/5.1/en/binary-installation.html)
2. [How can I export the privileges from MySQL and then import to a new server?](http://serverfault.com/questions/8860/how-can-i-export-the-privileges-from-mysql-and-then-import-to-a-new-server)
3. [How To Set Up Master Slave Replication in MySQL](https://www.digitalocean.com/community/tutorials/how-to-set-up-master-slave-replication-in-mysql)
4. [Setting up MySQL replication without the downtime](http://plusbryan.com/mysql-replication-without-downtime) 
5. [Promoting a mysql slave to master](https://onemoretech.wordpress.com/2013/09/19/promoting-a-mysql-slave-to-master/)
6. [Promote a Slave to Replace a Failed Master
Background](http://mysql.wingtiplabs.com/documentation/pro95ras/promote-a-slave-to-replace-a-failed-master)
