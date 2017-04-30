---
title: 安装RabbitMQ
layout: post
catalog: true
---


### 1. 安装erlang

erlang的安装比想象中要麻烦的多，试验了好几种，总是出现依赖错误。。最后还是通过kerl脚本安装搞定了。。具体参见：[Installing Erlang](https://docs.basho.com/riak/1.1.0/tutorials/installation/Installing-Erlang/)。

	[work@study.arganzheng.me downloads]$ ./kerl list releases
	Getting the available releases from erlang.org...
	R10B-0 R10B-10 R10B-1a R10B-2 R10B-3 R10B-4 R10B-5 R10B-6 R10B-7 R10B-8 R10B-9 R11B-0 R11B-1 R11B-2 R11B-3 R11B-4 R11B-5 R12B-0 R12B-1 R12B-2 R12B-3 R12B-4 R12B-5 R13A R13B01 R13B02-1 R13B02 R13B03 R13B04 R13B R14A R14B01 R14B02 R14B03 R14B04 R14B_erts-5.8.1.1 R14B R15B01 R15B02 R15B02_with_MSVCR100_installer_fix R15B03-1 R15B03 R15B R16A_RELEASE_CANDIDATE R16B01 R16B02 R16B03-1 R16B03 R16B 17.0-rc1 17.0-rc2 17.0 17.1 17.3 17.4 17.5 18.0 18.1 18.2.1 18.2 18.3
	Run 'kerl update releases' to update this list from erlang.org
	[work@study.arganzheng.me downloads]$ ./kerl update releases
	Getting the available releases from erlang.org...
	The available releases are:
	R10B-0 R10B-10 R10B-1a R10B-2 R10B-3 R10B-4 R10B-5 R10B-6 R10B-7 R10B-8 R10B-9 R11B-0 R11B-1 R11B-2 R11B-3 R11B-4 R11B-5 R12B-0 R12B-1 R12B-2 R12B-3 R12B-4 R12B-5 R13A R13B01 R13B02-1 R13B02 R13B03 R13B04 R13B R14A R14B01 R14B02 R14B03 R14B04 R14B_erts-5.8.1.1 R14B R15B01 R15B02 R15B02_with_MSVCR100_installer_fix R15B03-1 R15B03 R15B R16A_RELEASE_CANDIDATE R16B01 R16B02 R16B03-1 R16B03 R16B 17.0-rc1 17.0-rc2 17.0 17.1 17.3 17.4 17.5 18.0 18.1 18.2.1 18.2 18.3
	[work@study.arganzheng.me downloads]$ ./kerl build 18.3 erlang-18.3
	Downloading otp_src_18.3.tar.gz to /home/work/.kerl/archives
	  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
	                                 Dload  Upload   Total   Spent    Left  Speed
	100 65.1M  100 65.1M    0     0   259k      0  0:04:17  0:04:17 --:--:--  255k^@
	Getting the checksum file from erlang.org...
	  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
	                                 Dload  Upload   Total   Spent    Left  Speed
	100 27754  100 27754    0     0  13640      0  0:00:02  0:00:02 --:--:-- 87188
	Verifying archive checksum...
	Checksum verified (7e4ff32f97c36fb3dab736f8d481830b)
	Extracting source code
	Building Erlang/OTP 18.3 (erlang-18.3), please wait...
	Erlang/OTP 18.3 (erlang-18.3) has been successfully built
	[work@study.arganzheng.me downloads]$ ./kerl list builds
	18.3,erlang-18.3
	[work@study.arganzheng.me downloads]$ ./kerl install erlang-18.3 /home/work/erlang/18.3
	Installing Erlang/OTP 18.3 (erlang-18.3) in /home/work/erlang/18.3...
	You can activate this installation running the following command:
	. /home/work/erlang/18.3/activate
	Later on, you can leave the installation typing:
	kerl_deactivate
	[work@study.arganzheng.me downloads]$ ./kerl list installations
	erlang-18.3 /home/work/erlang/18.3
	[work@study.arganzheng.me downloads]$ . /home/work/erlang/18.3/activate
	[work@study.arganzheng.me downloads]$ ./kerl active
	The current active installation is:
	/home/work/erlang/18.3


### 2. 安装rabbitmq

一开始图省事，想直接使用yum或者rpm包安装，结果发现问题一大堆。。。：

	[root@study.arganzheng.me downloads]# rpm -Uvh rabbitmq-server-3.6.1-1.noarch.rpm
	error: Failed dependencies:
		erlang >= R16B-03 is needed by rabbitmq-server-3.6.1-1.noarch

会报没有找到erlang依赖，但是其实我们已经通过kerl脚本安装过了，所以可以直接忽略（[Install RabbitMQ on CentOS 6](http://www.stephenrhoades.com/?p=484)）:

	[root@study.arganzheng.me downloads]# rpm --nodeps -Uvh rabbitmq-server-3.6.1-1.noarch.rpm
	Preparing...                ########################################### [100%]
	   1:rabbitmq-server        ########################################### [100%]

果然很简单，但是发现用root安装会导致各种奇怪的问题：

	[root@study.arganzheng.me ~]# /etc/init.d/rabbitmq-server start
	Starting rabbitmq-server: FAILED - check /var/log/rabbitmq/startup_{log, _err}
	rabbitmq-server.
	[root@study.arganzheng.me rabbitmq]# tailf startup_err
	mkdir: cannot create directory `/var/run/rabbitmq': Permission denied


最后还是决定使用源码安装。。首先卸载掉原来的安装：

	[root@study.arganzheng.me ~]# yum remove rabbitmq-server
	Loaded plugins: aliases, downloadonly, fastestmirror, priorities, security
	Setting up Remove Process
	Resolving Dependencies
	--> Running transaction check
	---> Package rabbitmq-server.noarch 0:3.6.1-1 will be erased
	--> Finished Dependency Resolution

	Dependencies Resolved

	================================================================================================================================================
	 Package                                 Arch                           Version                         Repository                         Size
	================================================================================================================================================
	Removing:
	 rabbitmq-server                         noarch                         3.6.1-1                         installed                         5.5 M

	Transaction Summary
	================================================================================================================================================
	Remove        1 Package(s)

	Installed size: 5.5 M
	Is this ok [y/N]: y
	Downloading Packages:
	Running rpm_check_debug
	Running Transaction Test
	Transaction Test Succeeded
	Running Transaction
	Warning: RPMDB altered outside of yum.
	  Erasing    : rabbitmq-server-3.6.1-1.noarch                                                                                               1/1
	Stopping rabbitmq-server: RabbitMQ is not running
	rabbitmq-server.
	  Verifying  : rabbitmq-server-3.6.1-1.noarch                                                                                               1/1

	Removed:
	  rabbitmq-server.noarch 0:3.6.1-1

	Complete!

再源码安装：[building RabbitMQ server from source](http://www.rabbitmq.com/build-server.html)。

但是默认会安装到/usr/local目录，然后又是root启动。。所以我们要修改他的默认安装地址，然而文档并没有说明怎么指定，查看Makefile发现如下变量：

	DESTDIR ?=

	PREFIX ?= /usr/local
	WINDOWS_PREFIX ?= rabbitmq-server-windows-$(VERSION)

	MANDIR ?= $(PREFIX)/share/man
	RMQ_ROOTDIR ?= $(PREFIX)/lib/erlang
	RMQ_BINDIR ?= $(RMQ_ROOTDIR)/bin
	RMQ_LIBDIR ?= $(RMQ_ROOTDIR)/lib
	RMQ_ERLAPP_DIR ?= $(RMQ_LIBDIR)/rabbitmq_server-$(VERSION)


所以我们可以这样子编译：

	[work@study.arganzheng.me downloads]$ cd rabbitmq-server-3.6.1
	[work@study.arganzheng.me rabbitmq-server-3.6.1]$ make
	[work@study.arganzheng.me rabbitmq-server-3.6.1]$ make install PREFIX=/home/work/rabbitmq RMQ_ROOTDIR=/home/work/rabbitmq RMQ_ERLAPP_DIR=
	/usr/bin/make64 MAC=64 install PREFIX=/home/work/rabbitmq RMQ_ROOTDIR=/home/work/rabbitmq

	[work@study.arganzheng.me ~]$ vim .bashrc

	# .bashrc

	# User specific aliases and functions

	# Source global definitions
	if [ -f /etc/bashrc ]; then
		. /etc/bashrc
	fi

	ERL_HOME=/home/work/erlang/18.3
	export PATH=$PATH:$ERL_HOME/bin

	RABBITMQ_HOME=/home/work/rabbitmq/lib/rabbitmq_server-3.6.1
	export PATH=$PATH:$RABBITMQ_HOME/sbin

	[work@study.arganzheng.me ~]$ source .bashrc

### 3. 配置和启动

	[work@study.arganzheng.me ~]$ rabbitmq-server start
	/home/work/rabbitmq/lib/rabbitmq_server-3.6.1/sbin/rabbitmq-server: line 49: /var/lib/rabbitmq/mnesia/rabbit@hkg02-mj-data07.pid: Permission denied
	Failed to write pid file: /var/lib/rabbitmq/mnesia/rabbit@hkg02-mj-data07.pid
	
修改一下用户组：

	[root@study.arganzheng.me work]# chown -R work:work /var/lib/rabbitmq/

再次启动：

	[work@study.arganzheng.me ~]$ rabbitmq-server start

	              RabbitMQ 3.6.1. Copyright (C) 2007-2016 Pivotal Software, Inc.
	  ##  ##      Licensed under the MPL.  See http://www.rabbitmq.com/
	  ##  ##
	  ##########  Logs: /var/log/rabbitmq/rabbit@hkg02-mj-data07.log
	  ######  ##        /var/log/rabbitmq/rabbit@hkg02-mj-data07-sasl.log
	  ##########
	              Starting broker... completed with 6 plugins.

	[work@study.arganzheng.me ~]$ tailf /var/log/rabbitmq/rabbit@hkg02-mj-data07.log
	Statistics database started.

	=INFO REPORT==== 8-Apr-2016::16:54:51 ===
	Server startup complete; 6 plugins started.
	 * rabbitmq_management
	 * rabbitmq_management_agent
	 * rabbitmq_web_dispatch
	 * webmachine
	 * mochiweb
	 * amqp_client

成功了，肉留满面啊。。

**TIPS**

1. 注意到源码安装的RabbitMQ默认安装了6个插件，其实就是下面要介绍的监控和管理插件。
2. 如果要后台运行: rabbitmq-server -detached
3. 正在生产环境使用需要配置一下，比如内存，文件句柄限制，等。

### 4. 管理和监控

[Management Plugin](https://www.rabbitmq.com/management.html)

如果是源码安装，这一步可以忽略：

	[work@study.arganzheng.me ~]$ rabbitmq-plugins enable rabbitmq_management
	The following plugins have been enabled:
	  mochiweb
	  webmachine
	  rabbitmq_web_dispatch
	  amqp_client
	  rabbitmq_management_agent
	  rabbitmq_management

	Applying plugin configuration to rabbit@hkg02-mj-data07... started 6 plugins.

然后重启服务。


管理插件默认是监听15672端口，我们可以修改这个配置项：

	[work@study.arganzheng.me ~]$ vim /etc/rabbitmq/rabbitmq.config
   	{rabbitmq_management,
      [ {http_log_dir,  "/tmp/rabbit-mgmt"},
        {rates_mode,    basic},
        {listener, [{port,     8345}]}
    ]},

现在我们可以在浏览器直接访问了：http://10.242.111.26:8345/。但是发现登陆不了，要用户名密码。默认的guest/guest并不能登陆成功，日志显示：

	=WARNING REPORT==== 8-Apr-2016::17:05:33 ===
	HTTP access denied: user '' - invalid credentials

	=ERROR REPORT==== 8-Apr-2016::17:05:33 ===
	webmachine error: path="/api/whoami"
	"Unauthorized"

	=INFO REPORT==== 8-Apr-2016::17:05:33 ===
	webmachine_log_handler: closing log file: "/tmp/rabbit-mgmt/access.log"

	=INFO REPORT==== 8-Apr-2016::17:05:33 ===
	opening log file: "/tmp/rabbit-mgmt/access.log.2016_04_08_09"
	^@
	=WARNING REPORT==== 8-Apr-2016::17:06:03 ===
	HTTP access denied: user 'guest' - User can only log in via localhost

guest只能通过本机访问。查看文档需要使用`rabbitmqctl add_user`来创建一个non-administrator user 和 `rabbitmqctl set_user_tags`来提升一个普通用户为管理员:


	[work@study.arganzheng.me ~]$ rabbitmqctl add_user argan xxxx
	Creating user "argan" ...
	[work@study.arganzheng.me ~]$ rabbitmqctl list_users
	Listing users ...
	argan	[]
	guest	[administrator]
	[work@study.arganzheng.me ~]$ rabbitmqctl set_user_tags argan administrator
	Setting tags for user "argan" to [administrator] ...
	[work@study.arganzheng.me ~]$ rabbitmqctl list_users
	Listing users ...
	argan	[administrator]
	guest	[administrator]

果然可以登陆成功了，管理节目还是很清新的～。

还可以通过管理插件查看API：http://10.242.111.26:8345/api/。

但是发现用管理节目创建一个队列一直没有反应，Console显示为 PUT http://10.242.111.26:8345/api/queues/%2F/91-order-wq 401 (Unauthorized)。日志显示：

	=ERROR REPORT==== 8-Apr-2016::17:43:45 ===
	webmachine error: path="/api/queues/%2F/91-order-wq"
	"Unauthorized"
	^@^@^@^@
	=WARNING REPORT==== 8-Apr-2016::17:48:10 ===
	HTTP access denied: user 'argan' - User not authorised to access virtual host

	=ERROR REPORT==== 8-Apr-2016::17:48:10 ===
	webmachine error: path="/api/queues/%2F/91-order-wq"
	"Unauthorized"

是没有权限，原来 administrator 还不是万能的。。好吧。。

	[work@study.arganzheng.me ~]$ rabbitmqctl add_vhost /mobopay
	Creating vhost "/mobopay" ...
	[work@study.arganzheng.me ~]$ rabbitmqctl set_user_tags argan  administrator management
	Setting tags for user "argan" to [administrator,management] ...
	[work@study.arganzheng.me ~]$ rabbitmqctl set_permissions -p /mobopay argan '.*' '.*' '.*'
	Setting permissions for user "argan" in vhost "/mobopay" ... 

果然可以了。	 


### 4. 使用

可以参考这个系列，[RabbitMQ系列 第三篇：工作队列Work Queue](http://www.tuicool.com/articles/jEBVJb)


