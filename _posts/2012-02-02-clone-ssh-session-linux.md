---
layout: post
title: Linux里复制终端Session(像SecureCRT一样)
---

This Tip is from YangQi

在你的登录账户下的.ssh文件夹新建一个文件：config

    cd ~/.ssh

config的文件中，内容为：

    host *
    ControlMaster auto
    ControlPath ~/.ssh/master-%r@%h:%p

 在linux里面，像secureCRT一样，复制session,不需要重复输入密码

---

 举例：

 1. 你在窗口a登录了server2

 2. 新开窗口b，ssh user@server2后，就不需要再输入用户名密码。将经常需要登录的服务器，比如跳板机，写在online.sh里面，每次只要运行online，效果更佳。
