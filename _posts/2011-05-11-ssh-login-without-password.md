---
layout: post
title: shell如何实现ssh免密码登陆
---


在shell中经常有这样的情况，一个可以写成脚本的程序，本来你可以执行它然后去星巴克喝咖啡，但是由于里面有些交互行命令需要你回答，如login某个服务器，或者输入svn密码，等。但是这样都是我们可以预料的，如果能将我们的回答先存放在某个地方，让shell在那个时候自动去读取这个答案，那么他就可以继续执行了。google了一下，发现有个命令还是不错的，叫作expect，man了一下，有如下描述：
>
In general, Expect is useful for running  any  program  which  requires interaction between the program and the user.  All that is necessary is that the interaction can be characterized programmatically.  Expect can also give the user back control (without halting the program being controlled) if desired.  Similarly, the user can  return  control  to  the script at any time.

这里有一篇文章是介绍利用这个工具实现ssh自动登录：[Auto SSH login using ‘expect’ tool](http://www.techpulp.com/blog/2009/02/auto-ssh-login-using-expect-tool/)

这一篇也是类似的: [Using expect to auto login and execute commands(relieve you from tedious login and typing)](http://www.doxer.org/learn-linux/using-expect-to-auto-login-and-execute-commandsrelieve-you-from-tedious-login-and-typing/)

但是另一种更常见的方式是使用ssh public key实现自动登录。
举个例子：我要ssh到后裔我的虚拟机器。正常情况下我应该这么操作：

    forrest@ubuntu:~$ ssh admin@10.249.197.118
    admin@10.249.197.118's password: 
    Last login: Wed Apr 27 19:08:12 2011 from 10.16.42.97

如果要实现ssh key登录，需要作如下操作：

首先在想要免登的机器上用ssh-keygen生成你登录机的ssh public key：

    forrest@ubuntu:~$ ssh-keygen -t rsa
    Generating public/private rsa key pair.
    Enter file in which to save the key (/home/forrest/.ssh/id_rsa): 
    Enter passphrase (empty for no passphrase): 
    Enter same passphrase again: 
    Your identification has been saved in /home/forrest/.ssh/id_rsa.
    Your public key has been saved in /home/forrest/.ssh/id_rsa.pub.
    The key fingerprint is:
    57:48:59:94:a0:91:1a:ac:54:46:53:a2:09:a7:93:e2 forrest@ubuntu
    The key's randomart image is:
    +--[ RSA 2048]----+
    |  . .+*.o.o=o.   |
    |   =.+oo.+...    |
    |. +.o. o. . .    |
    |.. .. .    .     |
    | E      S .      |
    |         .       |
    |                 |
    |                 |
    |                 |
    +-----------------+

然后将这个生成的public key copy到目标机器的~./ssh目录，这里是我的后裔机器：

    forrest@ubuntu:$ cd .ssh
    forrest@ubuntu:~/.ssh$ ll
    total 24
    drwx------   2 forrest forrest 4096 2011-05-17 17:33 ./
    drwxr-xr-x 102 forrest forrest 4096 2011-05-17 17:09 ../
    -rw-------   1 forrest forrest 1671 2011-05-17 17:33 id_rsa
    -rw-r--r--   1 forrest forrest  396 2011-05-17 17:33 id_rsa.pub
    -rw-r--r--   1 forrest forrest 5132 2011-05-05 18:38 known_hosts
    
可以看到生成了两个文件，一个是你的私钥，一个是公钥。将公钥scp到目标机器的~/.ssh目录，并且创建（或者追加文件到）一个authorized_keys文件：
    
    forrest@ubuntu:~/.ssh$ scp id_rsa.pub admin@10.249.197.118:~/.ssh/
    admin@10.249.197.118's password: 
    id_rsa.pub                                                                                                   100%  396     0.4KB/s   00:00    
    
    [admin@localhost .ssh]$ ls
    id_rsa.pub
    [admin@localhost .ssh]$ cat id_rsa.pub >> authorized_keys
    [admin@localhost .ssh]$ ll
    total 8
    -rw-rw-r-- 1 admin admin 396 May 17 19:44 authorized_keys
    -rw-r--r-- 1 admin admin 396 May 17 19:44 id_rsa.pub

OK了，试验一下，不行。换了台机器，又试验了一下，还是不行。但是倒过来，试验从那台机器免秘密登录我本机确实可以的，所以方法应该没有问题。估计线上机器还作了限制。

网上说必须将下面这个三个目录的访问权限设置为writable for ower only：[Getting started with SSH](http://kimmo.suominen.com/docs/ssh/)

但是笔者试了多次还是不行:(
悲剧，不过还好，except还算可以用的:

    forrest@ubuntu:~/study/shell$ cat ssh_login.exp 
    #!/usr/bin/expect
    
    spawn ssh forrest@10.20.157.187
    expect "password:"
    send "hello1234?\n"
    interact

一下子就登上去了。

### 使用ssh-keygen和ssh-copy-id更快捷方便的实现免密码登录

前面使用ssh public key的方式其实shell有更方便的命令可以简化这个过程——使用ssh-copy-id可以自动将你的公钥scp到目标机器，并且更改访问权限。这样，整个过程只需要3个步骤，非常方便。
具体可以参考一下这篇文章：[3 Steps to Perform SSH Login Without Password Using ssh-keygen & ssh-copy-id](http://www.thegeekstuff.com/2008/11/3-steps-to-perform-ssh-login-without-password-using-ssh-keygen-ssh-copy-id/)


