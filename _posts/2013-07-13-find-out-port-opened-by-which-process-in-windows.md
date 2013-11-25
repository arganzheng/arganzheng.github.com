---
layout: post
title: windows下如何查看端口被哪个进程占用
---


比如要找出哪个进程占用了80端口：

Here is netstat command & output to find which process is holding port 80.

    C:\Users\arganzheng>netstat -nao | findstr "0.0.0.0:80"
    TCP    0.0.0.0:80             0.0.0.0:0              LISTENING       2608

可以看到，2608就是占用80端口的进程id。

然后可以使用tasklist 根据pid得到进程信息：

    C:\Users\arganzheng>tasklist /svc /FI "PID eq 2608"
    
    映像名称                       PID 服务
    ========================= ======== ============================================
    nginx.exe                     2608 暂缺


然后我们可以快速的Kill掉它：

打开运行窗口或者直接到cmd终端（Open Run (Window key + R) OR commend prompt (cmd.exe) ），执行命令`taskkill /F /IM nginx.exe`:

    C:\Users\arganzheng>taskkill /F /IM nginx.exe
    成功: 已终止进程 "nginx.exe"，其 PID 为 2608。

## 参考文章

1. [Nginx wont leave! how to remove it](http://stackoverflow.com/questions/12239778/nginx-wont-leave-how-to-remove-it)