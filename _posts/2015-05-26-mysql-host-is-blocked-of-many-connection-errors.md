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