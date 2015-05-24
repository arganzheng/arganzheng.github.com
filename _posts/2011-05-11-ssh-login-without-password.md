---
layout: post
title: shell如何实现ssh免密码登陆
---


在shell中经常有这样的情况，一个可以写成脚本的程序，本来你可以执行它然后去星巴克喝咖啡，但是由于里面有些交互行命令需要你回答，如login某个服务器，或者输入svn密码，等。但是这样都是我们可以预料的，如果能将我们的回答先存放在某个地方，让shell在那个时候自动去读取这个答案，那么他就可以继续执行了。google了一下，发现有个命令还是不错的，叫作expect，man了一下，有如下描述：
> In general, Expect is useful for running  any  program  which  requires interaction between the program and the user.  All that is necessary is that the interaction can be characterized programmatically.  Expect can also give the user back control (without halting the program being controlled) if desired.  Similarly, the user can  return  control  to  the script at any time.

这里有一篇文章是介绍利用这个工具实现ssh自动登录：[Auto SSH login using ‘expect’ tool](http://www.techpulp.com/blog/2009/02/auto-ssh-login-using-expect-tool/)

这一篇也是类似的: [Using expect to auto login and execute commands(relieve you from tedious login and typing)](http://www.doxer.org/learn-linux/using-expect-to-auto-login-and-execute-commandsrelieve-you-from-tedious-login-and-typing/)。

    forrest@ubuntu:~/study/shell$ cat ssh_login.exp 
    #!/usr/bin/expect
    
    spawn ssh forrest@10.20.43.123
    expect "password:"
    send "123456\n"
    interact

一下子就登上去了。

但是使用expect自动登陆有个问题，就是expect脚本跟shell脚本还是有差异的，需要另外学习expect脚本。

---

另一种更常见的方式是使用ssh public key实现自动登录。

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

然后将这个生成的public key copy到目标机器的~/.ssh/authorized_keys文件。这里是我的后裔机器：

    forrest@ubuntu:$ cd .ssh
    forrest@ubuntu:~/.ssh$ ll
    total 24
    drwx------   2 forrest forrest 4096 2011-05-17 17:33 ./
    drwxr-xr-x 102 forrest forrest 4096 2011-05-17 17:09 ../
    -rw-------   1 forrest forrest 1671 2011-05-17 17:33 id_rsa
    -rw-r--r--   1 forrest forrest  396 2011-05-17 17:33 id_rsa.pub
    -rw-r--r--   1 forrest forrest 5132 2011-05-05 18:38 known_hosts
    
可以看到生成了两个文件，一个是你的私钥，一个是公钥。将公钥追加到目标机器的~/.ssh/authorized_keys文件：
    
    forrest@ubuntu:~/.ssh$ ssh admin@10.249.197.118 mkdir -p .ssh
    admin@10.249.197.118's password:
 
    forrest@ubuntu:~/.ssh$ cat id_rsa.pub | ssh admin@10.249.197.118 'cat >> .ssh/authorized_keys'
    admin@10.249.197.118's password:  

登陆到目标机器验证一下：

    [admin@localhost .ssh]$ ll
    -rw-rw-r-- 1 admin admin 396 May 17 19:44 authorized_keys

OK了，试验一下，不行。换了台机器，又试验了一下，还是不行。但是倒过来，试验从那台机器免秘密登录我本机确实可以的，所以方法应该没有问题。估计线上机器还作了限制。

谷歌了一下，网上说必须将下面这个三个目录的访问权限设置为writable for ower only：[Getting started with SSH - Directory and file permissions](https://kimmo.suominen.com/docs/ssh/#chmod)

> #### Directory and file permissions
> 
> If access to the remote system is still denied you should check the permissions of the following files on it:
>
>   * the home directory itself
>   * the ~/.ssh directory
>   * the ~/.ssh/authorized_keys file
>
> To make the remote system allow access you must change the permissions to disallow writing by others than the owner.
> 
>     hrothgar% cd
>     hrothgar% chmod go-w . .ssh .ssh/authorized_keys
>
> Remember to do this on all the systems you want to have access to.

试了一下，果然可以了。

### 使用ssh-keygen和ssh-copy-id更快捷方便的实现免密码登录

前面使用ssh public key的方式其实shell有更方便的命令可以简化这个过程——使用ssh-copy-id可以自动将你的公钥追加到目标机器的~/.ssh/authorized_keys文件，并且更改访问权限。这样，整个过程只需要3个步骤，非常方便。

> **SYNOPSIS**
>       ssh-copy-id [-i [identity_file]] [user@]machine
>
> **DESCRIPTION**
>   ssh-copy-id is a script that uses ssh to log into a remote machine (presumably using a login password, so password authentication should be enabled, unless you've done some clever use of multiple identities) It also changes the permissions of the remote user's home,  ~/.ssh,  and ~/.ssh/authorized_keys  to  remove  group writability (which would otherwise prevent you from logging in, if the remote sshd has StrictModes set in its configuration).  
> If the -i option is given then the identity file (defaults to ~/.ssh/id_rsa.pub) is used, regardless of  whether there are any keys in your ssh-agent.  Otherwise, if this:  ssh-add -L provides any output, it uses that in preference to the identity file.  If the -i option is used, or the ssh-add produced no output, then it uses the contents of the identity file.  Once it has one or more fingerprints (by whatever means) it uses ssh to append them to ~/.ssh/authorized_keys on the remote machine (creating the file, and directory, if necessary)

具体可以参考一下这篇文章：[3 Steps to Perform SSH Login Without Password Using ssh-keygen & ssh-copy-id](http://www.thegeekstuff.com/2008/11/3-steps-to-perform-ssh-login-without-password-using-ssh-keygen-ssh-copy-id/)

**TIPS**

有些系统没有自带ssh-copy-id命令，这时候就只能回退到原始的步骤了。可以用下面这个命令替代ssh-copy-id [Copy your ssh public key to a server from a machine that doesn't have ssh-copy-id](http://www.commandlinefu.com/commands/view/188/copy-your-ssh-public-key-to-a-server-from-a-machine-that-doesnt-have-ssh-copy-id):

    cat ~/.ssh/id_rsa.pub | ssh work@jp01-hao123-mob00.jp01.baidu.com "cat > tmp.pubkey ; mkdir -p .ssh ; touch .ssh/authorized_keys ; sed -i.bak -e '/$(awk '{print $NF}' ~/.ssh/id_rsa.pub)/d' .ssh/authorized_keys; cat tmp.pubkey >> .ssh/authorized_keys; rm tmp.pubkey; chmod go-w . .ssh .ssh/authorized_keys"


可以用shell封装一下：

    [admin@localhost .ssh]$ cat ssh_copy_id.sh
    machines=( \
              "work@jp01" \
              "work@hk01" \
              "work@hk02" \
              )

    # ssh-copy-id 
    ssh_copy_id()
    {
        local identity_file=$1
        local target=$2

        eval "cat $identity_file" | ssh "$target" "cat > tmp.pubkey ; mkdir -p .ssh ; touch .ssh/authorized_keys ; sed -i.bak -e '/$(awk '{print $NF}' ~/.ssh/id_rsa.pub)/d' .ssh/authorized_keys; cat tmp.pubkey >> .ssh/authorized_keys; rm tmp.pubkey; chmod go-w . .ssh .ssh/authorized_keys;"

        if [[ $? -eq 0 ]]; then
            echo "copy $identity_file to $target success!"
        else
            echo "Oops! Something is wrong! Please try it manually:("
        fi
    }

    PS3='Select the machine you want to copy-ssh-id to: '

    select machine in "${machines[@]}" "Other" "Quit"
    do
        case $machine in 
            "${machines[$REPLY-1]}")
                echo "Begin copy-ssh-id to $machine!"
                ssh_copy_id "~/.ssh/id_rsa.pub" "$machine"
                ;;
            "Other")
                echo "Type the machine in format user@host that you want to execute ssh-copy-id,  followed by [ENTER]:"

                read target

                echo $target | grep -q "@"
                if [ $? != 0 ];then
                    target="work@$target"
                fi

                ssh_copy_id "~/.ssh/id_rsa.pub" "$target"
                ;;
            "Quit")
                break
                ;;
            *) 
                echo invalid option
                ;;
        esac
    done


补记：如何在shell代码中正确的实现免登陆操作
-------------------------------------------

其实ssh免登陆操作最大的必要性是用在定时/自动执行的shell脚本中，这时候往往没有用户干预，所以免登陆是非常必要的，否则下面一切免谈。那么如何在shell中正确的实现免登陆呢？

使用except脚本当然是可以的，但是前面说过，他的语法跟shell不一样，写起来非常蛋疼。所以一般是不采用，只是用来做简单的免登陆而已。

ssh-public-key方式是可以的，但是在shell脚本中要注意一下几种情况，避免出现错误。
1. 主机key变化。这会导致提示`The authenticity of host *** can’t be established`，需要输出一个“yes”的交互。
2. public-key变化。这会导致免登失败，需要一个手动输入密码的交互。

这两个情况都会导致public-key方式免登陆失败，而且恶心的是程序就卡在那里了，在等待一个 yes 或者 密码 的输入，而因为在脚本中，不会弹出提示，你根本就不知道发生什么问题了。

所以对于shell最好采用防御式编程，增加如下一些参数：

    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o PasswordAuthentication=no -o ConnectionAttempts=5 work@${ERROR_SLAVES[$CNT]} bash < /home/work/userbin/restart_slave_replication.sh
    
对于rsync，同样有这个问题：

    rsync -auv -e "ssh -o StrictHostKeyChecking=no -o PasswordAuthentication=no" /home/work/mnt/nfs/mbrowser/t5pac/ work@xxx-xxx-xxx.xxx:/home/work/STATIC/mbrowser/t5pac 

这样至少可以保证你的程序不会卡住不动。





