---
title: MySQL主从同步失败
layout: post
---

最近MySQL经常报主从错误，如果是网络问题，应该是间断性的，但是今天早上一直在报错。上去看了一下log，发现一直在报这个错误：

	150527 11:30:01 [Note] Slave I/O thread killed while connecting to master
	150527 11:30:01 [Note] Slave I/O thread exiting, read up to log 'master-bin.000052', position 308678027
	150527 11:30:01 [Note] Slave SQL thread initialized, starting replication in log 'master-bin.000052' at position 308678027, relay log './copyer1-relay-bin.002681' position: 5382766
	150527 11:30:02 [ERROR] Slave I/O: error connecting to master 'copyer@10.242.73.34:3306' - retry-time: 60  retries: 86400, Error_code: 1129

在命令行手动执行远程连接：

	[xxx@xxx-xxx-xxx.xxx mysql-5.1.73-linux-x86_64-glibc23]$ bin/mysql -h10.242.73.34 -ucoyper -p
	Enter password:
	ERROR 1129 (HY000): Host 'xxxxx' is blocked because of many connection errors; unblock with 'mysqladmin flush-hosts'

这下子对1129错误码有了更明确的错误信息和指示。谷歌一下，根据官方文档[B.5.2.6 Host 'host_name' is blocked](https://dev.mysql.com/doc/refman/5.0/en/blocked-host.html) 进行相应的操作：

登陆到master，执行：

	mysql> flush hosts;
	Query OK, 0 rows affected (0.00 sec)

slave立马就可以连接成功了。

再把允许的错误连接调大：

	mysql> SET GLOBAL max_connect_errors=10000;
	Query OK, 0 rows affected (0.00 sec)

固化在配置文件中：
	[root@xxxx ~]# vim /etc/my.cnf
	max_connect_errors = 10000


但是正如官方文档所说的，这个错误一般是因为网络原因，但是如果是网络原因，调整这个参数其实对于主从同步没有什么帮助。。

> If you get the Host 'host_name' is blocked error message for a given host, you should first verify that there is nothing wrong with TCP/IP connections from that host. If you are having network problems, it does you no good to increase the value of the max_connect_errors variable.


但是同步问题还是一直存在，统计了一下，发现主要是如下错误码

	[xxx@xxx log]$ grep 'Error_code:' mysqld.log| awk '{print $NF}'|sort | uniq -c | sort -n
	      1 2003
	      2 1159
	     17 1129
	     83 2013

对照官方错误码文档[Appendix B Errors, Error Codes, and Common Problems](http://dev.mysql.com/doc/refman/5.0/en/error-handling.html)，主要是如下错误：

> Error: 1129 SQLSTATE: HY000 (ER_HOST_IS_BLOCKED)
> Message: Host '%s' is blocked because of many connection errors; unblock with 'mysqladmin flush-hosts'
> 
> Error: 1159 SQLSTATE: 08S01 (ER_NET_READ_INTERRUPTED)
> Message: Got timeout reading communication packets
>
> Error: 2003 (CR_CONN_HOST_ERROR)
> Message: Can't connect to MySQL server on '%s' (%d)
>
> Error: 2013 (CR_SERVER_LOST)
> Message: Lost connection to MySQL server during query

一看很明显，都是网络问题，1129前面已经说过了。现在看报的最多的2013。同样在官方文档上也有相应的说明：

1. [B.5.2.3 Lost connection to MySQL server](https://dev.mysql.com/doc/refman/5.0/en/error-lost-connection.html) 
2. [B.5.2.9 MySQL server has gone away](http://dev.mysql.com/doc/refman/5.0/en/gone-away.html)

建议是调整timeout相关的参数看看[5.1.4 Server System Variables](http://dev.mysql.com/doc/refman/5.0/en/server-system-variables.html)：

* connect_timeout
* net_read_timeout 
* net_write_timeout
* slave_net_timeout
* wait_timeout

使用`show variables;`可以看到现在的值：

* connect_timeout = 10    
* interactive_timeout = 28800 
* net_read_timeout = 30                                                                                        |
* net_retry_count = 10 
* net_write_timeout = 60   
* slave_net_timeout = 3600 
* wait_timeout = 28800     


可以用[SET语句](https://dev.mysql.com/doc/refman/5.0/en/set-statement.html)进行动态修改：

	mysql> SET GLOBAL net_read_timeout=60;
	mysql> SET GLOBAL net_write_timeout=120;

改了一下，貌似好一些，但是还是会不时报错。。网络问题，没有太好的办法:(






