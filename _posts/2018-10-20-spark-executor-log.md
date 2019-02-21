---
layout: post
title: Spark如何查看某个applicationId的executor日志
tags: [spark]
catalog: true
---


### 方法一：直接在界面上查看

如果你运行在YARN模式，你可以在ResourceManager节点的WEB UI页面根据 任务状态、用户名 或者 applicationId Search 到你的应用。然后 

* 点击表格中 `Tracking UI` 列的History 链接；或者
* 点击相关的ApplicationId 链接，进入到详情页面点击上面的 `Tracking URL: History` 链接

就进入到Spark作业监控的 WEB UI 界面，这个页面就是你 Spark 应用程序历史执行界面：例如`http://bj01-prd-hadoop001.arganzheng.lan:18088/history/application_1537448956025_2516133/jobs/`。到这个界面之后，可以点击 Executors 菜单，这时候你可以进入到Spark程序的 Executors 界面，里面列出所有Executor信息，以表格的形式展示，在表格中有Logs这列，里面就是你Spark应用程序运行的日志。


### 方法二: 直接查看 HDFS 上的日志文件

例如用 xfhuang 账户运行的日志，可以用下面命令这里看到所有 application 的日志：

```shell
$ hadoop fs -ls /tmp/logs/xfhuang/logs/
Found 33 items
drwxrwx---+  - xfhuang hadoop          0 2018-10-18 17:06 /tmp/logs/xfhuang/logs/application_1537448956025_2344152
drwxrwx---+  - xfhuang hadoop          0 2018-10-18 17:42 /tmp/logs/xfhuang/logs/application_1537448956025_2346489
...
drwxrwx---+  - xfhuang hadoop          0 2018-10-19 21:16 /tmp/logs/xfhuang/logs/application_1537448956025_2448690
drwxrwx---+  - xfhuang hadoop          0 2018-10-20 15:47 /tmp/logs/xfhuang/logs/application_1537448956025_2514112
drwxrwx---+  - xfhuang hadoop          0 2018-10-20 16:39 /tmp/logs/xfhuang/logs/application_1537448956025_2516133
```

如果要看具体的某个 applicationId 产生的日志，则直接指定：

```shell
$ hadoop fs -ls /tmp/logs/xfhuang/logs/application_1537448956025_2409589
Found 237 items
-rw-r-----+ 1 xfhuang hadoop 11301 2018-10-19 10:11 /tmp/logs/xfhuang/logs/application_1537448956025_2409589/bigdata-offline-hadoop0100.bjthq.arganzheng.lan_8041
-rw-r-----+ 1 xfhuang hadoop 205262 2018-10-19 10:11 /tmp/logs/xfhuang/logs/application_1537448956025_2409589/bigdata-offline-hadoop0116.bjthq.arganzheng.lan_8041
-rw-r-----+ 1 xfhuang hadoop 11301 2018-10-19 10:11 /tmp/logs/xfhuang/logs/application_1537448956025_2409589/bigdata-offline-hadoop0118.bjthq.arganzheng.lan_8041
...
-rw-r-----+ 1 xfhuang hadoop 11301 2018-10-19 10:11 /tmp/logs/xfhuang/logs/application_1537448956025_2409589/bigdata-offline-hadoop0485.bjthq.arganzheng.lan_8041
-rw-r-----+ 1 xfhuang hadoop 623665 2018-10-19 10:11 /tmp/logs/xfhuang/logs/application_1537448956025_2409589/bigdata-offline-hadoop0491.bjthq.arganzheng.lan_8041
-rw-r-----+ 1 xfhuang hadoop 11287 2018-10-19 10:11 /tmp/logs/xfhuang/logs/application_1537448956025_2409589/bj01-prd-hadoop008.arganzheng.lan_8041
...
-rw-r-----+ 1 xfhuang hadoop 22240 2018-10-19 10:11 /tmp/logs/xfhuang/logs/application_1537448956025_2409589/bjthq-dm-dn0482.arganzheng.lan_8041
-rw-r-----+ 1 xfhuang hadoop 394399 2018-10-19 10:11 /tmp/logs/xfhuang/logs/application_1537448956025_2409589/bjthq-dm-dn0483.arganzheng.lan_8041
...
-rw-r-----+ 1 xfhuang hadoop 229187 2018-10-19 10:11 /tmp/logs/xfhuang/logs/application_1537448956025_2409589/bp-bigdata-offline-hadoop0616.bj01.arganzheng.lan_8041
```

是分机器存放的，可以具体看某一台机器的日志：

```shell
$ hadoop fs -ls /tmp/logs/xfhuang/logs/application_1537448956025_2409589/bjthq-bi-dn0008.arganzheng.lan_8041
-rw-r-----+  1 xfhuang hadoop     154095 2018-10-19 10:11 /tmp/logs/xfhuang/logs/application_1537448956025_2409589/bjthq-bi-dn0008.arganzheng.lan_8041

$ hadoop fs -cat /tmp/logs/xfhuang/logs/application_1537448956025_2409589/bjthq-bi-dn0008.arganzheng.lan_8041 | head
��h��׶9�A@���P  VERSIONAPPLICATION_ACLVIEW_APPnobody,xfhuang
MODIFY_APPnobody,xfhuangAPPLICATION_OWNER   xfhuang/-container_e93_1537448956025_2409589_01_000527�Xxcontainer-localizer-syslog1682018-10-19 09:49:41,402 WARN [main] org.apache.hadoop.yarn.server.nodemanager.containermanager.localizer.ContainerLocalizer: Localization running as nobody not xfhuang
stderr15347718/10/19 09:49:43 INFO executor.CoarseGrainedExecutorBackend: Started daemon with process name: 7201@bjthq-bi-dn0008.arganzheng.lan
18/10/19 09:49:43 INFO executor.CoarseGrainedExecutorBackend: Registered signal handlers for [TERM, HUP, INT]
18/10/19 09:49:44 INFO spark.SecurityManager: Changing view acls to: nobody,xfhuang
18/10/19 09:49:44 INFO spark.SecurityManager: Changing modify acls to: nobody,xfhuang
18/10/19 09:49:44 INFO spark.SecurityManager: SecurityManager: authentication disabled; ui acls disabled; users with view permissions: Set(nobody, xfhuang); users with modify permissions: Set(nobody, xfhuang)
18/10/19 09:49:44 INFO spark.SecurityManager: Changing view acls to: nobody,xfhuang
18/10/19 09:49:44 INFO spark.SecurityManager: Changing modify acls to: nobody,xfhuang
18/10/19 09:49:44 INFO spark.SecurityManager: SecurityManager: authentication disabled; ui acls disabled; users with view permissions: Set(nobody, xfhuang); users with modify permissions: Set(nobody, xfhuang)
cat: Unable to write to output stream.
```

下载到本地方便查看：

```shell
$ hadoop fs -get /tmp/logs/xfhuang/logs/application_1537448956025_2409589/bjthq-bi-dn0008.arganzheng.lan_8041  /home/xfhuang/temp/
```

里面的日志就是这台机器上这个 applicationId 下 executor 的运行日志。


### 方法三：通过 `yarn logs -applicationId` 命令查看

如果是 YARN 模式，最简单地收集日志的方式是使用 YARN 的日志收集工具（`yarn logs -applicationId`），这个工具可以收集你应用程序相关的运行日志，但是这个工具是有限制的：应用程序必须运行完，因为YARN必须首先聚合这些日志；而且你必须开启日志聚合功能（yarn.log-aggregation-enable，在默认情况下，这个参数是 false）。


```shell
$ yarn logs -applicationId  application_1537448956025_2409589 | head
18/10/20 17:14:06 INFO client.ConfiguredRMFailoverProxyProvider: Failing over to rm130
Unable to get ApplicationState. Attempting to fetch logs directly from the filesystem.


Container: container_e93_1537448956025_2409589_01_000232 on bigdata-offline-hadoop0100.bjthq.arganzheng.lan_8041
============================================================================================================
LogType:container-localizer-syslog
Log Upload Time:Fri Oct 19 10:11:06 +0800 2018
LogLength:168
Log Contents:
2018-10-19 09:46:41,580 WARN [main] org.apache.hadoop.yarn.server.nodemanager.containermanager.localizer.ContainerLocalizer: Localization running as nobody not xfhuang
```



