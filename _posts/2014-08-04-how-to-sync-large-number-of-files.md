---
title: 如何实时同步大量小文件
layout: post
---

需求
----

静态资源服务器存放新闻、视频、图片，每隔一段时间定时拉取更新。为了反正静态资源服务器(现在是用MFS)挂掉，需要将静态资源同步到普通的linux文件系统做备份。这样故障的时候可以切换回去。

大概有900w+的小文件，使用rsync，在没有任何文件变更的情况下，都需要执行20多分钟（扫描对比差异）。如何实时的同步呢？


方案
----

### 1、cron + rsync

	[work@xxx-xxxx-mob01.hk01.baidu.com zhengzhibin]$ cat ~/userbin/sync_static_mbrowser.sh
	#!/bin/bash

	source /home/work/.bashrc

	BASH_PATH="/home/work/.bashrc"
	LOG_PATH="/home/work/userbin/log"
	RSYNC_RESULT=`ps auxf | grep rsync | grep -v grep`
	RSYNC_PID=`ps auxf | grep rsync | grep -v grep | awk '{print $2}'`  # may be more one 

	count=`ps auxf | grep rsync | grep -v grep | awk '{print $2}' | wc -l` 

	if [ $count -ge 16 ];then
	        echo "[FATAL]"`date`" rsync running time out, killing!" >> $LOG_PATH
	        kill -9 $RSYNC_PID
	        exit 0
	fi


	# check if rsync is still running
	if [ "$RSYNC_RESULT"x == ""x ];then
	        echo "[NORMAL]"`date`" rsync will be running normally." >> $LOG_PATH
	        rsync -auv --delete-after argan@xxx-xxx-mob05.hk01:/home/work/STATIC/mbrowser  /home/work/mnt/mfs/
	else
	        echo "[WARNING]"`date`" last rsync is still running, new one will NOT run!" >> $LOG_PATH
	fi

优点: 简单
缺点：定时执行，实时性比较差；另外，rsync同步数据时，需要扫描所有文件后进行比对，进行差量传输。如果文件数量达到了百万甚至千万量级，扫描所有文件将是非常耗时的。而且正在发生变化的往往是其中很少的一部分，这是非常低效的方式。

### 2. inotify + rsync

为了避免每次都扫描对比差异，可以使用inotify得到关注目录的变更情况，然后使用rsync同步变更文件。例如下面的脚本：

	#!/bin/bash

	DEST="$1"

	if [ -z "$DEST" ]; then exit 1; fi

	inotifywait -r -m -e close_write --format '%w%f' . |\
	while read file
	do
	        echo $file
		rsync -azvq $file ${DEST}/$file
		echo -n 'Completed at '
		date
	done

这个脚本会监控本地文件夹并且当有文件变更的时候执行rsync同步到远程备份机器。可以这么执行：
	
	sync.sh argan@myhost.domain.com:STATIC/

但是这个脚本有个问题，就是他对每个变更文件都使用了rsync。而每一次rsync都要建立SSH链接，这个其实是比较费时间的。特别是当你的文件很小的时候，这个建立SSH链接时间相比真正的同步时间（传输变更文件时间）就非常浪费。针对这种情况，OpenSSH提供了一个非常实用的特性可以解决这个问题——[SSH Connection Multiplexing](http://www.revsys.com/writings/quicktips/ssh-faster-connections.html)。它可以复用同一个SSH链接。因此只有第一次建立SSH链接比较耗时。只需要在~/.ssh/config文件配置一下：

	Host myhost.domain.com
	    ControlMaster auto
	    ControlPath /tmp/%h%p%r

如果是OpenSSH 5.6+的，还可以使用[ControlPersist](http://lwn.net/Articles/401422/)，保持SSH链接时间。

也可以先采用NFS协议将dest目录mount到source目录所在机器，然后直接使用cp进行同步。

**TIPS** Trailing Slashes Do Matter...Sometimes

rsync对SourceDir和DestDir最后是否带斜杠有不同的处理。例如：

	rsync -auv /home/work/STATIC/mbrowser/guanxing  /home/work/mnt/mfs/mbrowser 

将产生备份到 /home/work/mnt/mfs/mbrowser/guanxing 目录

而

	rsync -auv /home/work/STATIC/mbrowser/guanxing/  /home/work/mnt/mfs/mbrowser 

将产生备份到 /home/work/mnt/mfs/mbrowser 目录

DestDir最后有没有斜杠也会对rsync产生影响。例如：

	rsync -auv --progress --delete-after /home/work/arganzheng/test.txt work@xxx-xxx-mob05.hk01:/home/work/arganzheng/

将产生备份到hk01-hao123-mob05.hk01:/home/work/arganzheng/test.txt。而且如果arganzheng目录不存在，会自动创建。

而
	rsync -auv --progress --delete-after /home/work/arganzheng/test.txt work@hxxx-xxx-mob05.hk01:/home/work/arganzheng 

将产生备份到xxx-xxx-mob05.hk01:/home/work/arganzheng。也就是说将test.txt重命名为arganzheng了。当然，如果xxx-xxx-mob05.hk01:/home/work/arganzheng已经存在，并且是目录，那么这种情况DestDir有没有最后斜杆就没有影响。


### 3. lsyncd

有很多工具封装了inotify监听和rsync的同步机制，提供了更便利的使用方式。比如[Lsyncd](https://github.com/axkibe/lsyncd)。lsyncd的配置文件是用lua语言写的，可读性很不错。Lua作为配置语言从这里可以看出来。[inosync](https://github.com/hollow/inosync)，是python版本的lsyncd。


### 4. 使用watchdog


参考文档
--------

1. [Inotify: 高效、实时的Linux文件系统事件监控框架](http://www.infoq.com/cn/articles/inotify-linux-file-system-event-monitoring)
2. [inotify-tools](https://github.com/rvoicilas/inotify-tools/wiki)
3. [Auto rsync local changes to remote server](http://goodcode.io/blog/linux-auto-rsync/)
4. [Recursive Inotify Scripting With Lsyncd](http://backdrift.org/recursive-inotify-scripting-with-lsyncd)
5. [Lsyncd](https://github.com/axkibe/lsyncd)
6. [How To Mirror Local and Remote Directories on a VPS with lsyncd](https://www.digitalocean.com/community/tutorials/how-to-mirror-local-and-remote-directories-on-a-vps-with-lsyncd)
7. [Compile lsyncd with Lua at special location](http://blog.a2o.si/2013/10/11/compile-lsyncd-with-lua-at-special-location/)
8. [Replication using lsyncd](http://www.lucasrolff.com/ha/replication-using-lsyncd/)
8. [watchdog](https://github.com/gorakhargosh/watchdog)