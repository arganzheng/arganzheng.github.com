---
title: 卓有成效的程序员——windows篇
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


windows下如何启动和停止windows服务
----------------------------------

比如MySQL，一般是安装成windows服务的，一般设置为开机启动，也可以手动启动：

    D:\Code>net start mysql56
    MySQL56 服务正在启动 .
    MySQL56 服务已经启动成功。

不过这里需要知道服务的名称。所以如果不知道服务名次的话，可以通过 任务管理器=>服务Tab=>服务按钮，进入服务管理界面启动。或者通过控制面板进入也是可以的。停止服务就比较简单了，因为`net start`会列出所有已经启动的服务。这时候就可以知道名称了。

    D:\Code>net stop mysql56
    MySQL56 服务正在停止 .
    MySQL56 服务已成功停止。

当然，通过任务管理器停止服务也是很方便的。


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


windows下如何查看域账户情况
------------------------

    C:\Users\arganzheng>net user /domain arganzheng
    这项请求将在域 tencent.com 的域控制器处理。
    
    用户名                 arganzheng
    全名                   arganzheng(郑志彬)
    注释                   XXX
    用户的注释
    国家/地区代码          000 (系统默认值)
    帐户启用               Yes
    帐户到期               从不
    
    上次设置密码           2013/6/24 9:44:22
    密码到期               2013/9/22 9:44:22
    密码可更改             2013/6/24 9:44:22
    需要密码               Yes
    用户可以更改密码       Yes
    
    允许的工作站           All
    登录脚本
    用户配置文件
    主目录
    上次登录               2013/7/15 15:07:36
    
    可允许的登录小时数     All
    
    本地组成员
    全局组成员             XXX
    命令成功完成。


代理客户端
---------


推荐 Proxifier。简单，易用。


Host绑定管理
-----------


hostsMaker


SSH终端
------


SecureCRT


浏览器插件 
---------


比较恶心的是mac下的快捷键与windows/linux下的略有不同。。。

* firefox+vimperator
    * x: 关闭当期tab
* chrome+vimuim
    * yy: copy the current URL to clipboard（常规做法：Alt-d, C-c）


github on windows
-----------------


#### 1. 使用[github for windows](http://windows.github.com/)

#### 2. 设置代理

To configure GitHub for Windows to use your  corporate proxy, edit the .gitconfig file typically found at C:\Users\<yourusername>\.gitconfig or C:\Documents & Settings\<yourusername>\.gitconfig

    [user]
        name = arganzheng
    	email = arganzheng@gmail.com
    [http]
    proxy = http://web-proxy.oa.com:8080
    [https]
    proxy = http://web-proxy.oa.com:8080

不错的文章： [git简易指南](http://rogerdudler.github.io/git-guide/index.zh.html)


**TIPS** SVN的话推荐使用 TortoiseSVN，非常好用。


cygwin的配置和使用
------------------


windows下虚拟机性能不行，不推荐。建议安装Ggywin。

具体参考：[惊艳的cygwin——Windows下的Linux命令行环境的配置和使用](http://oldratlee.com/post/2012-12-22/stunning-cygwin)。写的非常详细，这里就不赘述了。


sublime text
------------

Sublime Text 2默认不支持GB2312和GBK，可以安装一个插件让它支持 [解决乱码，让Sublime Text 2支持GB2312和GBK](http://www.fuzhaopeng.com/2012/sublime-text-2-with-gb2312-gbk-support/)。在这之前，先安装一个插件管理器方便插件的安装和卸载 [Sublime Package Control](http://wbond.net/sublime_packages/package_control/installation)。内网安装插件要设置代理：[Sublime Text 2 Package Control安装与设置代理方法](http://www.tanktan.com/blog/sublime2-proxy/)。

具体参考笔者的另一篇文章：[tips for sublime text](http://blog.arganzheng.me/posts/tips-for-sublime-text.html)


文章收集和多终端同步
-----------------

推荐Evernote，特别是浏览器剪辑插件，收集文章特别方便。


数据同步
-------

强烈推荐Dropbox。


文本编辑
-------

sublime text或者stackedit直接编辑markdown文件。




