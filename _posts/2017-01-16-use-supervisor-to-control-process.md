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


---


补记：使用supervisor进行进程监管(2)
===============================


[supervisor](http://supervisord.org/)是一个python写的进程管理工具，可以监管进程的运行。它提供了进程挂掉自动重启功能，并且还有一个管理页面，可以看到监管的程序的运行状态，也可以直接在整个页面上对进程进行重启、关闭、查看日志等操作。


### 安装

如果有网络并且已经安装了easy_install，那么就是简单一条命令：

```shell
#easy_install supervisor
```

其他方式可以参考官网，这里不赘述了。

### 使用方法

supervisor的主要配置入口在supervisord.conf，这里可以统一放在 `/root/script/supervisor/`下面：

`#vim /root/script/supervisor/supervisord.conf`

配置如下内容：

```shell
[unix_http_server]
file=/root/script/supervisor/run/supervisor.sock   ; (the path to the socket file)
;chmod=0700                 ; socked file mode (default 0700)
;chown=nobody:nogroup       ; socket file uid:gid owner
;username=user              ; (default is no username (open server))
;password=123               ; (default is no password (open server))

[inet_http_server]         ; inet (TCP) server disabled by default
port=10.21.7.1:8001      ; (ip_address:port specifier, *:port for all iface)
username=user              ; (default is no username (open server))
password=123               ; (default is no password (open server))

[supervisord]
logfile=./supervisord.log ; (main log file;default $CWD/supervisord.log)
;logfile_maxbytes=50MB       ; (max main logfile bytes b4 rotation;default 50MB)
;logfile_backups=10          ; (num of main logfile rotation backups;default 10)
loglevel=info            ; (log level;default info; others: debug,warn,trace)
pidfile=./supervisord.pid ; (supervisord pidfile;default supervisord.pid)
nodaemon=false               ; (start in foreground if true;default false)
;minfds=1024                 ; (min. avail startup file descriptors;default 1024)
;minprocs=200                ; (min. avail process descriptors;default 200)
;umask=022                   ; (process file creation umask;default 022)
;user=chrism                 ; (default is current user, required if root)
;identifier=supervisor       ; (supervisord identifier, default is 'supervisor')
;directory=/tmp              ; (default is not to cd during start)
;nocleanup=true              ; (don't clean up tempfiles at start;default false)
childlogdir=/root/script/supervisor/log/supervisor ; ('AUTO' child log dir, default $TEMP)
;environment=KEY=value       ; (key value pairs to add to environment)
;strip_ansi=false            ; (strip ansi escape codes in logs; def. false)

; the below section must remain in the config file for RPC
; (supervisorctl/web interface) to work, additional interfaces may be
; added by defining them in separate rpcinterface: sections
[rpcinterface:supervisor]
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface

[supervisorctl]
;serverurl=unix:///root/script/supervisor/run/supervisor.sock ; use a unix:// URL  for a unix socket
serverurl=http://10.21.7.1:8001 ; use an http:// url to specify an inet socket
;username=chris              ; should be same as http_username if set
;password=123                ; should be same as http_password if set
;prompt=mysupervisor         ; cmd line prompt (default "supervisor")
;history_file=~/.sc_history  ; use readline history if available


[include]
files = supervisor.d/*.ini program/*.program
```	

这里采用include的方式，可以更好的管理配置。

```
#cd /root/script/supervisor
#mkdir program
```

在program目录新建一个以.program为后缀的文件，然后填上必要的配置项（完整配置项参考http://supervisord.org/configuration.html#program-x-section-settings）。

比如这里我们要监管某个TensorFlow-Serving，那么可以这么建立一个配置文件：	


`#vim /root/script/supervisor/program/tensorflow_model_server_new.program`

配置如下内容：

```shell
[program:tensorflow_model_server_new]
command=/home/jhli/wide-and-deep/tensorflow_model_server_new --port=8500 --model_name=wide_and_deep --model_base_path=hdfs://footstone/data/project/dataming/browser_search_dev/jhli/wide_n_deep_test
directory=/home/jhli/wide-and-deep/
stopsignal=TERM
stopwaitsecs=60
user=jhli
stderr_logfile=/home/jhli/wide-and-deep/tensorflow.log
stdout_logfile=/home/jhli/wide-and-deep/tensorflow.log	
```

然后启动supervisor就可以了:

```shell
#supervisord -c /root/script/supervisor/supervisord.conf
```

当然启动前，记得确保如下目录的存在：

```shell
#mkdir -p /root/script/supervisor/program
#mkdir -p /root/script/supervisor/log/supervisor
#mkdir -p /root/script/supervisor/run
```

supervisor还提供了一个可视化管理界面，比如上面的配置，可以在 `http://10.21.7.1:8001/` 进行访问查看，界面效果大致如下所示：

![Supervisor管理界面](/img/in-post/supervisor-web.png)

如果启动之后新增或者更新配置，那么可以执行这个命令加载：

```
# 根据最新的配置文件，启动新配置或有改动的进程，配置没有改动的进程不会受影响而重启
#supervisorctl update
```

如果只需要重启某个程序，只需要运行 `#supervisorctl restart program_name` 或者在界面上点击就可以了。




