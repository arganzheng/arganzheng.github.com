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

下面语句创建一个带有binlog位置的备份文件，dump的过程中会影响DB性能，但是不会锁表：

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

其中 MASTER_LOG_FILE='mysql-bin.000005', MASTER_LOG_POS=3119 根据 master中执行：show master status得到。
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


参考文章
--------

1. [2.2 Installing MySQL on Unix/Linux Using Generic Binaries](http://dev.mysql.com/doc/refman/5.1/en/binary-installation.html)
2. [How can I export the privileges from MySQL and then import to a new server?](http://serverfault.com/questions/8860/how-can-i-export-the-privileges-from-mysql-and-then-import-to-a-new-server)
3. [How To Set Up Master Slave Replication in MySQL](https://www.digitalocean.com/community/tutorials/how-to-set-up-master-slave-replication-in-mysql)
4. [Setting up MySQL replication without the downtime](http://plusbryan.com/mysql-replication-without-downtime) 
5. [Promoting a mysql slave to master](https://onemoretech.wordpress.com/2013/09/19/promoting-a-mysql-slave-to-master/)
