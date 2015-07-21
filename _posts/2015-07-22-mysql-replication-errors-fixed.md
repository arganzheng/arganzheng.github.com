---
title: 记一次MySQL主从同步错误处理
layout: post
---

其实MySQL主从同步错误已经出现过好几次了，每次处理完都没有整理，今天恰好又出现了一次，花点时间记录一下，方便下次操作。

问题
----

今天早上收到报警短信，说印尼的一台机器home目录快爆满了，登上去看了一下，发现是一个应用的日志文件超级大，有好几百G。查看了一下日志内容，发现都是"Too many open file"错误。用`ulimit -a`查看，发现`open files`只有默认的65536，太小了。于是参考这篇文章——[Increase “Open Files Limit”](https://rtcamp.com/tutorials/linux/increase-open-files-limit/)，提高文件句柄数：

1、修改 `/etc/security/limits.conf`:

	*         hard    nofile      5000000
	*         soft    nofile      5000000

2、修改`/etc/sysctl.conf`:

	fs.file-max = 5000000

然后登出，然后悲剧了——无法ssh登陆了：

	ssh xxx@id01-xxx-mob21.id01
	Connection to id01-xxx-mob21.id01 closed.

上面有个MySQL Relay Server，从其他slave上看这个服务是正常的，也就是说机器并没有挂，只是无法ssh登陆上去了。找OP同学问了一下，他们说以前也遇到这个问题，就是修改了文件句柄数超过100w，就会出现这个问题。然后要重启机器，切换到单用户模式进入系统将最大句柄数修改回去，再次重启就可以了。

于是提单让sys同学操作了一下，果然可以了。但是由于机器被重启了，上面的服务就停掉了，需要重启。其他应用都还好，但是MySQL就有问题了。重启之后主从同步就不正常了，使用show slave status报如下错误：

	Last_Errno: 1062
	Last_Error: Error 'Duplicate entry '218ea6fadaf4410d840df6c270416f1e' for key 'PRIMARY'' on query. Default database: 'browser'. Query: 'insert into ...

使用最常见的解决方案，skip:

	mysql>STOP SLAVE; SET GLOBAL SQL_SLAVE_SKIP_COUNTER=1; START SLAVE;

但是发现skip了几次还是报同样的错误。。。谷歌了一下，可以配置全局忽略1062错误：[MySQL Skip Duplicate Replication Errors](http://www.ducea.com/2008/02/13/mysql-skip-duplicate-replication-errors/)	。于是配置了一下/etc/my.cnf:

	#*******************************  Replication related settings **********************
	server-id 	= 121
	relay-log-index = copyer1-relay-bin.index
	relay-log	= copyer1-relay-bin
	log-bin=mysql-bin
	log-bin-index=master-bin.index
	binlog_format=mixed
	log_slave_updates
	binlog_cache_size = 1M
	read-only  = 1
	slave-skip-errors = 1062

使用：`bin/mysqladmin -uroot -p shutdown`命令或者`kill -3`停止mysql，然后使用`bin/mysqld_safe`重启。果然不再报1062错误了，但是报了另一个错误：

	mysql> show slave status \G
	*************************** 1. row ***************************
	               Slave_IO_State: Waiting for master to send event
	                  Master_Host: 10.242.73.34
	                  Master_User: copyer
	                  Master_Port: 3306
	                Connect_Retry: 60
	              Master_Log_File: master-bin.002371
	          Read_Master_Log_Pos: 735103168
	               Relay_Log_File: copyer1-relay-bin.023159
	                Relay_Log_Pos: 946556300
	        Relay_Master_Log_File: master-bin.002367
	             Slave_IO_Running: Yes
	            Slave_SQL_Running: No
	              Replicate_Do_DB:
	          Replicate_Ignore_DB:
	           Replicate_Do_Table:
	       Replicate_Ignore_Table:
	      Replicate_Wild_Do_Table:
	  Replicate_Wild_Ignore_Table:
	                   Last_Errno: 1594
	                   Last_Error: Relay log read failure: Could not parse relay log event entry. The possible reasons are: the master's binary log is corrupted (you can check this by running 'mysqlbinlog' on the binary log), the slave's relay log is corrupted (you can check this by running 'mysqlbinlog' on the relay log), a network problem, or a bug in the master's or slave's MySQL code. If you want to check the master's binary log or slave's relay log, you will be able to know their names by issuing 'SHOW SLAVE STATUS' on this slave.
	                 Skip_Counter: 0
	          Exec_Master_Log_Pos: 946556154
	              Relay_Log_Space: 5047257477
	              Until_Condition: None
	               Until_Log_File:
	                Until_Log_Pos: 0
	           Master_SSL_Allowed: No
	           Master_SSL_CA_File:
	           Master_SSL_CA_Path:
	              Master_SSL_Cert:
	            Master_SSL_Cipher:
	               Master_SSL_Key:
	        Seconds_Behind_Master: NULL
	Master_SSL_Verify_Server_Cert: No
	                Last_IO_Errno: 0
	                Last_IO_Error:
	               Last_SQL_Errno: 1594
	               Last_SQL_Error: Relay log read failure: Could not parse relay log event entry. The possible reasons are: the master's binary log is corrupted (you can check this by running 'mysqlbinlog' on the binary log), the slave's relay log is corrupted (you can check this by running 'mysqlbinlog' on the relay log), a network problem, or a bug in the master's or slave's MySQL code. If you want to check the master's binary log or slave's relay log, you will be able to know their names by issuing 'SHOW SLAVE STATUS' on this slave.


看日志也有相应的错误：

	work@id01-xxx-mob21.id01 mysql-5.1.73-linux-x86_64-glibc23]$ 
	150721 17:04:19  InnoDB: Started; log sequence number 464 4033168739
	150721 17:04:20 [Note] Slave SQL thread initialized, starting replication in log 'master-bin.002367' at position 936478209, relay log './copyer1-relay-bin.023159' position: 936478355
	150721 17:04:20 [Note] Event Scheduler: Loaded 0 events
	150721 17:04:20 [Note] /home/work/Mysql/mysql-5.1.73-linux-x86_64-glibc23/bin/mysqld: ready for connections.
	Version: '5.1.73-log'  socket: '/home/work/Mysql/mysql-5.1.73-linux-x86_64-glibc23/mysql.sock'  port: 3306  MySQL Community Server (GPL)
	150721 17:04:20 [Note] Slave I/O thread: connected to master 'copyer@10.242.73.34:3306',replication started in log 'master-bin.002371' at position 595269823
	150721 17:04:24 [ERROR] Error in Log_event::read_log_event(): 'Event too big', data_len: 1852402531, event_type: 111
	150721 17:04:24 [ERROR] Error reading relay log event: slave SQL thread aborted because of I/O error
	150721 17:04:24 [ERROR] Slave SQL: Relay log read failure: Could not parse relay log event entry. The possible reasons are: the master's binary log is corrupted (you can check this by running 'mysqlbinlog' on the binary log), the slave's relay log is corrupted (you can check this by running 'mysqlbinlog' on the relay log), a network problem, or a bug in the master's or slave's MySQL code. If you want to check the master's binary log or slave's relay log, you will be able to know their names by issuing 'SHOW SLAVE STATUS' on this slave. Error_code: 1594
	150721 17:04:24 [ERROR] Error running query, slave SQL thread aborted. Fix the problem, and restart the slave SQL thread with "SLAVE START". We stopped at log 'master-bin.002367' position 946556154
	150721 17:05:01 [Note] Slave I/O thread killed while reading event
	150721 17:05:01 [Note] Slave I/O thread exiting, read up to log 'master-bin.002371', position 688891426
	150721 17:05:01 [Note] Slave SQL thread initialized, starting replication in log 'master-bin.002367' at position 946556154, relay log './copyer1-relay-bin.023159' position: 946556300
	150721 17:05:01 [ERROR] Error in Log_event::read_log_event(): 'Event too big', data_len: 1852402531, event_type: 111
	150721 17:05:01 [ERROR] Error reading relay log event: slave SQL thread aborted because of I/O error
	150721 17:05:01 [ERROR] Slave SQL: Relay log read failure: Could not parse relay log event entry. The possible reasons are: the master's binary log is corrupted (you can check this by running 'mysqlbinlog' on the binary log), the slave's relay log is corrupted (you can check this by running 'mysqlbinlog' on the relay log), a network problem, or a bug in the master's or slave's MySQL code. If you want to check the master's binary log or slave's relay log, you will be able to know their names by issuing 'SHOW SLAVE STATUS' on this slave. Error_code: 1594
	150721 17:05:01 [ERROR] Error running query, slave SQL thread aborted. Fix the problem, and restart the slave SQL thread with "SLAVE START". We stopped at log 'master-bin.002367' position 946556154
	150721 17:05:01 [Note] Slave I/O thread: connected to master 'copyer@10.242.73.34:3306',replication started in log 'master-bin.002371' at position 688891426


谷歌了一下，其实就是slave的binlog损坏了，需要从损坏的位置重新进行主从同步。[Fixing MySQL replication after slaves's relay log was corrupted](http://www.redips.net/mysql/replication-slave-relay-log-corrupted/)。损坏的位置在日志其实有很明显的说明：

> 150721 17:04:24 [ERROR] Error running query, slave SQL thread aborted. Fix the problem, and restart the slave SQL thread with "SLAVE START". We stopped at log 'master-bin.002367' position 946556154

当然，你也可以通过`show slave status`的Relay_Master_Log_File和Exec_Master_Log_Pos得到。

接下来就很简单了：

	# stop slave
	mysql> stop slave;
	 
	# make slave forget its replication position in the master's binary log
	mysql> reset slave;
	 
	# change slave to start reading from stopped position
	mysql> CHANGE MASTER TO MASTER_HOST='10.242.73.34',MASTER_USER='copyer', MASTER_PASSWORD='xxx', MASTER_LOG_FILE='master-bin.002367', MASTER_LOG_POS=946556154;
	 
	# start slave
	mysql> start slave;

再`show slave status`一下发现已经OK了。


但是还没有完，这个只是解决了relay server与master的主从同步问题，slave与relay之间的主从同步也因为relay server异常重启而挂掉了。

	mysql> show slave status \G
	*************************** 1. row ***************************
	               Slave_IO_State:
	                  Master_Host: 10.244.17.77
	                  Master_User: copyer
	                  Master_Port: 3306
	                Connect_Retry: 60
	              Master_Log_File: mysql-bin.001205
	          Read_Master_Log_Pos: 813554836
	               Relay_Log_File: copyer1-relay-bin.006906
	                Relay_Log_Pos: 813554981
	        Relay_Master_Log_File: mysql-bin.001205
	             Slave_IO_Running: No
	            Slave_SQL_Running: Yes
	              Replicate_Do_DB:
	          Replicate_Ignore_DB:
	           Replicate_Do_Table:
	       Replicate_Ignore_Table:
	      Replicate_Wild_Do_Table:
	  Replicate_Wild_Ignore_Table:
	                   Last_Errno: 0
	                   Last_Error:
	                 Skip_Counter: 0
	          Exec_Master_Log_Pos: 813554836
	              Relay_Log_Space: 1887297445
	              Until_Condition: None
	               Until_Log_File:
	                Until_Log_Pos: 0
	           Master_SSL_Allowed: No
	           Master_SSL_CA_File:
	           Master_SSL_CA_Path:
	              Master_SSL_Cert:
	            Master_SSL_Cipher:
	               Master_SSL_Key:
	        Seconds_Behind_Master: NULL
	Master_SSL_Verify_Server_Cert: No
	                Last_IO_Errno: 1236
	                Last_IO_Error: Got fatal error 1236 from master when reading data from binary log: 'Client requested master to start replication from impossible position'
	               Last_SQL_Errno: 0
	               Last_SQL_Error:
	1 row in set (0.00 sec)

在relay上执行show master status：

	mysql> show master status\G
	*************************** 1. row ***************************
	            File: mysql-bin.001213
	        Position: 19007623
	    Binlog_Do_DB:
	Binlog_Ignore_DB:

[work@id01-xxx-mob21.id01 mysql-5.1.73-linux-x86_64-glibc23]$ ll data/

	....
	-rw-rw----  1 work work 1073742068 Jul 21 12:41 mysql-bin.001200
	-rw-rw----  1 work work 1073742153 Jul 21 13:02 mysql-bin.001201
	-rw-rw----  1 work work 1073741984 Jul 21 13:23 mysql-bin.001202
	-rw-rw----  1 work work 1073742112 Jul 21 13:43 mysql-bin.001203
	-rw-rw----  1 work work 1073742182 Jul 21 14:04 mysql-bin.001204
	-rw-rw----  1 work work  813125632 Jul 21 14:19 mysql-bin.001205
	-rw-rw----  1 work work    6997678 Jul 21 17:03 mysql-bin.001206
	-rw-rw----  1 work work 1073742112 Jul 21 17:23 mysql-bin.001207
	-rw-rw----  1 work work 1073742139 Jul 21 17:27 mysql-bin.001208
	-rw-rw----  1 work work 1073742097 Jul 21 17:32 mysql-bin.001209
	-rw-rw----  1 work work 1073742059 Jul 21 17:42 mysql-bin.001210
	-rw-rw----  1 work work 1073741999 Jul 21 17:46 mysql-bin.001211
	-rw-rw----  1 work work 1073742034 Jul 21 17:54 mysql-bin.001212
	-rw-rw----  1 work work  225594935 Jul 21 17:58 mysql-bin.001213	

果然有问题：mysql-bin.001205只有813125632字节大小，但是请求的位置却是813554836。

这个错误原因很简单，原本的slave正在pulling的master.bin因為各种原因不正常关闭，当下次master正常重启时，master會另开信的binlog写日志，此時slave重新连接上并不知道有这件事，继续向master要求宕掉的binlog的下一笔日志记录，就会出现什么的错误。

显然解决方案很简单，就是给slave重新指定一个合理的同步位置就可以了。关键是哪个位置呢？根据上面分析，这里其实指定到下一个binlog的起始位置就可以了：

	mysql> stop slave;
	 
	# 指到下一个 binlog 档, pos 指 0 或 4, 都是一样意思.
	mysql> CHANGE MASTER TO MASTER_HOST='10.244.17.77',MASTER_USER='copyer', MASTER_PASSWORD='xxxx', MASTER_LOG_FILE='mysql-bin.001206', MASTER_LOG_POS=0; 

	# start slave
	mysql> start slave;
	 
再用`SHOW SLAVE STATUS`查看果然没有问题了。

参考文章：[MySQL Replication 遇到 Got fatal error 1236 from master 修复](http://www.hksilicon.com/kb/cn/articles/248076/MySQL-ReplicationGot-fatal-error-1236-from-master)

