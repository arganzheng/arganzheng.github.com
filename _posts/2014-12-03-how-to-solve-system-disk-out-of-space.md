---
title: 如何解决系统盘爆满问题
layout: post
---


感谢 @邓译文 的 TIPS

### 1. 关闭休眠功能

在命令行中执行：`@powercfg -h off`，可以多出好几G的磁盘空间。

### 2. 将虚拟内存关闭或者转移到非系统盘

具体参见：[虚拟内存设置](http://jingyan.baidu.com/article/2fb0ba4075567800f2ec5fcb.html)。可以多出好几G的磁盘空间。

### 3. 删除本地邮件，并且转存到非系统盘

一般系统原装的outlook默认是安装在系统盘，邮件一多很容易爆满，可以在C:\Users\zhengzhibin\AppData\Local\Microsoft\Outlook找到最大的文件（有可能有好几G），直接删除。然后配置一下邮件的存储位置：[outlook 2013 如何修改邮件存放位置](http://jingyan.baidu.com/article/295430f1304ee20c7e00501f.html)。

### 4. 使用磁盘分区工具扩大系统盘

比如 [分区助手](http://rj.baidu.com/soft/detail/11603.html?ald)
