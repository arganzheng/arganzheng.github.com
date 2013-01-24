---
layout: post
title: 如何在远程Linux机器上执行Shell命令
---


通常我们需要写个shell脚本然后放上服务器上执行。一般的步骤是先登陆上服务器，然后将shell脚本上传到服务器的某个目录下，最后是执行这个脚本。然而shell本身是支持远程执行的。

可以使用如下命令：

    ssh user@server bash < /path/to/local/script.sh
    
在远程机器上运行一段脚本。这条命令最大的好处就是不用把脚本拷到远程机器上。

如果你只是想执行on line command，那么可以使用管道：

    echo "date" | ssh user@server bash
    
当然，远程执行的前提其实是要先ssh到服务器的，需要身份验证。可以使用`except`或者`ssh public key`来实现免密码登陆。这样就更完美了。具体可以参见笔者的另一篇文章：[shell如何实现ssh免密码登陆](http://blog.arganzheng.me/posts/ssh-login-without-password.html)


这篇文章 [在多台Linux机器上执行命令](http://blog.csdn.net/gjyalpha/article/details/7264107) 使用了`socat`来实现这个功能。其实不用，不过socat还是挺强大的。

如果服务器有python运行环境，那么可以使用python的`paramkio`库，这样可以直接享受python带来的好处。

例如：下面的代码在远程服务器上执行`ifconfig`命令：
>Example 5-9. Connecting to an SSH server and remotely executing a command
>    
    #!/usr/bin/env python 
>
    import paramiko
>    
    hostname = '192.168.1.15' 
    port = 22
    username = 'jmjones' 
    password = 'xxxYYYxxx'
>    
    if __name__ == "__main__": 
        paramiko.util.log_to_file('paramiko.log')
        s = paramiko.SSHClient()
        s.load_system_host_keys()
        s.connect(hostname, port, username, password) 
        stdin, stdout, stderr = s.exec_command('ifconfig') 
        print stdout.read()
        s.close()

### 常见应用场景

1. 启动服务
2. 查看所有log文件
3. 上传下载文件


