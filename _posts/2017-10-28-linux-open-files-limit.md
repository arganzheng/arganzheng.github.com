---
title: 如何查看和设置文件句柄数
layout: post
category: 技术
tags: [linux]
---


### 1. 系统级别限制(System-Wide Limit)

Set this higher than user-limit set above.

配置文件：`/etc/sysctl.conf`:

	fs.file-max = 2097152

然后运行: `sysctl -p`让其生效。

这个命令会增加系统级别（所有用户）可以打开的文件句柄数。

可以通过`/proc/sys/fs/file-max`或者`sysctl fs.file-max` 验证:

	$ cat /proc/sys/fs/file-max
	2097152

还可以通过`/proc/sys/fs/file-nr`文件查看整个系统目前使用的文件句柄数量：

	# cat /proc/sys/fs/file-nr
	3391    969     52427
	|	 	 |       |
	|	 	 |       |
	|        |       maximum open file descriptors
	|        total free allocated file descriptors
	total allocated file descriptors
	(the number of file descriptors allocated since boot)

打开的文件描述符是=column 1 – column 2; 也就是 3391 - 969 = 2325 在上面的例子中。 

**TIPS** 也可以通过修改`/proc/sys/fs/file-max`修改系统最大文件句柄数：

	echo "2097152" > /proc/sys/fs/file-max 


### 2. 用户级别限制(Per-User Limit)

你也可以单独为某个用户配置最大文件句柄数，如 mysql, httpd，等。

使用`ulimit`命令可以查看或者动态设置每个用户的资源限制：

> ulimit refers to the per-user limitations for various resources.

	$ ulimit -a
	core file size          (blocks, -c) unlimited
	data seg size           (kbytes, -d) unlimited
	file size               (blocks, -f) unlimited
	pending signals                 (-i) 1549357
	max locked memory       (kbytes, -l) 64
	max memory size         (kbytes, -m) unlimited
	open files                      (-n) 65536
	pipe size            (512 bytes, -p) 8
	POSIX message queues     (bytes, -q) 819200
	stack size              (kbytes, -s) 10240
	cpu time               (seconds, -t) unlimited
	max user processes              (-u) 30720
	virtual memory          (kbytes, -v) unlimited
	file locks                      (-x) unlimited

对应的配置文件在 `/etc/security/limits.conf`中:

	*         hard    nofile      500000
	*         soft    nofile      500000
	root      hard    nofile      500000
	root      soft    nofile      500000
	httpd     hard    nofile      500000
	httpd 	  soft    nofile      50000
	elasticsearch  -  nofile      65536

This change will only take effect the next time the elasticsearch user opens a new session.

可以切换用户验证：

	# su - httpd
	$ ulimit -Hn
	$ ulimit -Sn

如果要修改当前session的limit值，可以直接使用ulimit -n newValue：

	ulimit -n 65536  

The new limit is only applied during the current session.


#### 3. 进程级别限制(Per-Process Limit)

一般来说，不需要单独为某个进程设置最大的可用句柄数，因为进程会继承父进程的最大可用句柄数，而这个数一般来说就是上面的`ulimit -n`得到的值。当然，如果想要修改这个值，可用通过`ulimit -n xxx`把它设置成新的值。但是需要注意的是，不同于系统和用户级别，进程级别的文件句柄数限制必须单个进程单独设置，一般来说是在启动的时候使用ulimit命令设置:

	...
	LIMIT="/bin/limit -n 30960"
	SUPERVISE="/home/work/psop_v2/supervise/bin/supervise"

	START_COMMAND="${SUPERVISE} -p ${STA_DIR} -f \"${LIMIT} ${BIN_DIR}/run.sh\""
	...

说明：这里的`/bin/limit`应该是百度自己搞的命令，本质上相对于ulimit，不知道为什么要自己搞一个。。

可以通过`/proc/<pid>/limits`检查某个进程的资源限制情况：

	$ cat /proc/8411/limits
	Limit                     Soft Limit           Hard Limit           Units
	Max cpu time              unlimited            unlimited            seconds
	Max file size             unlimited            unlimited            bytes
	Max data size             unlimited            unlimited            bytes
	Max stack size            10485760             unlimited            bytes
	Max core file size        unlimited            unlimited            bytes
	Max resident set          unlimited            unlimited            bytes
	Max processes             30720                30720                processes
	Max open files            30960                30960                files
	Max locked memory         65536                65536                bytes
	Max address space         unlimited            unlimited            bytes
	Max file locks            unlimited            unlimited            locks
	Max pending signals       1549357              1549357              signals
	Max msgqueue size         819200               819200               bytes
	Max nice priority         0                    0
	Max realtime priority     0                    0
	Max realtime timeout      unlimited            unlimited            us


参考文章
-------

1. [Increase “Open Files Limit”](https://easyengine.io/tutorials/linux/increase-open-files-limit/)
2. [Configuring system settings](https://www.elastic.co/guide/en/elasticsearch/reference/master/setting-system-settings.html)
