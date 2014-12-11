---
title: nginx日志自动按天分隔
layout: post
---

log4j可以的DailyRollingFileAppender可以实现日志按天归档，避免日志过大。同时也方便按天查询统计日志。但是nginx没有实现这个功能，所有的日志都是一个文件死磕到底。像error_log还好，但是access_log一般都比较大，按天归档是很有必要的，怎样实现呢？

可以这样简单处理：

写一个脚本: nginx_log_split.sh

	#!/bin/bash

	cd /home/work/nginx
	log_dir="/home/work/nginx/logs"
	time=`date +%Y%m%d`
	nginx_dir="/home/work/nginx"

	for logfile in `ls -l $log_dir |  grep -v "^d" | awk '{print $9}' | grep ".log$" | grep -v "[_-]\{1\}[0-9]\{8\}.log"` ;do
		mv $log_dir/$logfile $log_dir/${logfile%.*}_$time.log
	done;

	$nginx_dir/sbin/nginx -s reload

**NOTE**

注意，这里使用的是ls -l而不是ll，虽然ll一般都是ls -l的alias，但是alias是跟登陆的shell绑定的，new出来的shell是不会继承的。所以不能用别名。否则会报错。


这个文件会找出 /home/work/nginx/logs 下的所有日志文件，重新命名为 xxx_{YYYYMMDD}.log文件。然后让nginx重新加载配置文件以写到新的文件。

然后配置一个crontab，让它在接近夜晚0点的时候执行：

	$ crontab -l

	59 23 * * * sleep 56;sh /home/work/userbin/nginx_log_split.sh >/dev/null 2>&1

由于crontab的最小调度粒度是分，所以这里先sleep了56s，再执行这个nginx_log_split脚本。
