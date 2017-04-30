---
title: 使用supervisor进行进程监管
layout: post
catalog: true
---

[supervisor](http://supervisord.org/)是一个python写的进程管理工具，可以监管进程的运行。它提供了进程挂掉自动重启功能，并且还有一个管理页面，可以看到监管的程序的运行状态，也可以直接在整个页面上对进程进行重启、关闭、查看日志等操作。


### 安装

安装比较简单，直接看官网，这里不赘述了。

### 使用方法

supervisor的主要配置入口在supervisord.conf：

	[unix_http_server]
	file=/home/work/rdbin/kg/supervisor/run/supervisor.sock   ; (the path to the socket file)
	;chmod=0700                 ; socked file mode (default 0700)
	;chown=nobody:nogroup       ; socket file uid:gid owner
	;username=user              ; (default is no username (open server))
	;password=123               ; (default is no password (open server))

	[inet_http_server]         ; inet (TCP) server disabled by default
	port=nj03-bdg-kg-offline-03.nj03.baidu.com:8011      ; (ip_address:port specifier, *:port for all iface)
	username=xxx              ; (default is no username (open server))
	password=xxx               ; (default is no password (open server))

	[supervisord]
	logfile=./supervisord.log ; (main log file;default $CWD/supervisord.log)
	;logfile_maxbytes=50MB       ; (max main logfile bytes b4 rotation;default 50MB)
	;logfile_backups=10          ; (num of main logfile rotation backups;default 10)
	loglevel=info			; (log level;default info; others: debug,warn,trace)
	pidfile=./supervisord.pid ; (supervisord pidfile;default supervisord.pid)
	nodaemon=false               ; (start in foreground if true;default false)
	;minfds=1024                 ; (min. avail startup file descriptors;default 1024)
	;minprocs=200                ; (min. avail process descriptors;default 200)
	;umask=022                   ; (process file creation umask;default 022)
	;user=chrism                 ; (default is current user, required if root)
	;identifier=supervisor       ; (supervisord identifier, default is 'supervisor')
	;directory=/tmp              ; (default is not to cd during start)
	;nocleanup=true              ; (don't clean up tempfiles at start;default false)
	childlogdir=./log/supervisor ; ('AUTO' child log dir, default $TEMP)
	;environment=KEY=value       ; (key value pairs to add to environment)
	;strip_ansi=false            ; (strip ansi escape codes in logs; def. false)

	; the below section must remain in the config file for RPC
	; (supervisorctl/web interface) to work, additional interfaces may be
	; added by defining them in separate rpcinterface: sections
	[rpcinterface:supervisor]
	supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface

	[supervisorctl]
	;serverurl=unix:///home/work/rdbin/kg/supervisor/run/supervisor.sock ; use a unix:// URL  for a unix socket
	serverurl=http://127.0.0.1:8011 ; use an http:// url to specify an inet socket
	;username=chris              ; should be same as http_username if set
	;password=123                ; should be same as http_password if set
	;prompt=mysupervisor         ; cmd line prompt (default "supervisor")
	;history_file=~/.sc_history  ; use readline history if available


	[include]
	files = supervisor.d/*.ini program/*.program

采用include的方式可以更好的管理配置。

	cd $ /home/work/rdbin/kg/supervisor
	$ mkdir program

在program目录新建一个以.program为后缀的文件，然后填上必要的配置项（完整配置项参考http://supervisord.org/configuration.html#program-x-section-settings）。

比如我们要监管ES，那么可以这么建立一个配置文件：	

	$ vim elasticsearch.program

	[program:elasticsearch]
	command=/home/work/elasticsearch-5.1.1/bin/elasticsearch
	directory=/home/work/elasticsearch-5.1.1
	stopsignal=TERM
	stopwaitsecs=60
	user=work

再比如kafka的监控：

	$ vim program/kafka_broker.program
	directory=/home/work/kafka
	stopsignal=TERM
	stopwaitsecs=60
	user=work
	environment=JMX_PORT=8999

然后启动supervisor就可以了:

	$ supervisord -c supervisord.conf

如果启动之后新增或者更新配置，那么可以执行这个命令加载：

    $ ./supervisorctl.sh update

如果只需要重启某个程序，只需要运行 ./supervisorctl.sh restart program_name 或者在界面上点击就可以了。


