---
title: 高效windows环境开发
layout: post
---


windows下如何查看端口被哪个进程占用
-----------------------------------


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


windows下如何查看文件被哪个进程打开
-----------------------------------


有时候经常发现删除不了某个文件，提示文件被进程占用。这时候需要找出赵勇的进程，kill掉它就可以了。

任务管理器==>性能(Tab)==>资源监视器==>CPU(Tab)==>关联的文件句柄，输入文件名就可以找到了。

windows设置远程桌面连接
-----------------------

1. 开放远程连接权限：控制面板==>系统和安全==>系统==>左侧的远程设置，选择 仅运行使用网络级别身份验证的远程桌面的计算机。
2. 由于公司防火墙策略，默认端口3389可能被禁用。通过修改注册表该为其他端口，比如3398。
3. 找一台机器连接测试一下。ipconfig或者用我的电脑查看自己的计算机名称。然后运行`mstsc`，输入计算机名称：`arganzheng-pc0.tencent.com:3398`和用户名：`tencent\arganzheng`测试一下。


windows下mount网络硬盘
----------------------

使用net use命令

    net use 盘符: \\远程机器\共享名\路径 /user:...

例子：

    net use z: \\tencent.com\tfs\部门目录\ECC腾讯电商控股公司\电商研发部\ /user:arganzheng@tencent.com


github on windows
-----------------

### 1. 使用[github for windows](http://windows.github.com/)

### 2. 设置代理

To configure GitHub for Windows to use your  corporate proxy, edit the .gitconfig file typically found at C:\Users\<yourusername>\.gitconfig or C:\Documents & Settings\<yourusername>\.gitconfig

    [user]
        name = arganzheng
    	email = arganzheng@gmail.com
    [http]
    proxy = http://web-proxy.oa.com:8080
    [https]
    proxy = http://web-proxy.oa.com:8080

不错的文章： [git简易指南](http://rogerdudler.github.io/git-guide/index.zh.html)


cygwin的配置和使用
------------------

具体参考：[惊艳的cygwin——Windows下的Linux命令行环境的配置和使用](http://oldratlee.com/post/2012-12-22/stunning-cygwin)。写的非常详细，这里就不赘述了。


sublime text
------------

具体参考笔者的另一篇文章：[tips for sublime text](http://blog.arganzheng.me/posts/tips-for-sublime-text.html)
