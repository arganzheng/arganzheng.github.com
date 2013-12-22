---
title: 使用EC2和SSH翻墙
layout: post
---


前提
---

拥有一台墙外的机器——这里是EC2。具体申请和配置过程请参考笔者的上一篇文章 [使用亚马逊EC2搭建web工程](http://blog.arganzheng.me/posts/setup-java-web-project-on-amazon-ec2.html)


步骤
---

### 1. SSH配置

参考文章[EC2 + SSH 翻墙 3/3— SSH 翻墙设置](http://www.qishansun.com/?p=12)中有针对windows和linux详细的配置说明。笔者这里只针对mac做简要的说明：

	cd ~/.ssh
	vim config

输入以下内容

	Host ec2-ami
	HostName 54.201.85.167 
	User ec2-user
	IdentityFile ~/.ssh/argan-aws-ec2.pem
	CompressionLevel 6
	DynamicForward localhost:3128

其中

* Host 为设定一个名字给该连接，可任意命名，笔者这里命名为用ec2-ami；
* HostName 这里输入附加到该Instance的Elastic IP， 或者 public DNS 的连接，这些都可从Dashboard instance 信息里面看到；
* User 因为之前EC2上建立的是ubuntu AMI，其默认的登录用户名为ec2-user；
* IdentityFile 需输入你的 pem 文件的路径；
* DynamicForward 默认用localhost:3128，在后面的浏览器设置中要用到。

配置完成之后，从teminal中输入：

	ssh ec2-ami

就可以直接登陆进去了，这意味着一条ssh通道已经建立了。下面只需要配置一下浏览器就可以了。


### 2. 浏览器配置

Chrome和Safari的代理都是使用系统设置中的代理，是全局性的，不方便，当然可以安装插件，像`SwitchySharp`之类的。不过这里笔者直接采用firefox作为走代理的浏览器，其他浏览器访问不需要翻墙的网站。

Firefox配置如下：

Preference -> Advanced -> Network -> Settings, 选择手动代理设置，SOCKS Host中输入localhost，端口号3128，SOCKS v5。No Proxy for: localhost, 127.0.0.1。OK 确认。浏览器配置完毕。
![Firefox-浏览器代理配置](http://qishansun.info/blog/wp-content/uploads/2011/07/broswer1.png)

### 3. 外面的世界

现在可以开眼看看外面的世界了。先用`http://ip138.com`查看一下你的IP地址的地理位置：

	您的IP是：[54.201.85.167] 来自：美国 新泽西州Merck公司

再打开一个被墙的网站试试，比如`http://www.slideshare.net/brendangregg/linux-performance-analysis-and-tools?utm_source=slideshow02&utm_medium=ssemail&utm_campaign=share_slideshow`。果然可以了。

但是发现facebook和twitter还是上不了，网上有人要做额外的配置：

>在Firefox地址栏输入 about:config 
>有安全提示，点击继续；
>找到“network.proxy.socks_remote_dns”，双击改为True（默认False）；
>可能需要重启Firefox。

但是笔者试过，还是不行。不过这两个网站我基本也不上的，所以let it  be:)


**Welcome to the real world!**


参考文章
-------

1. [EC2 + SSH 翻墙 3/3— SSH 翻墙设置](http://www.qishansun.com/?p=12)
2. [使用 amazon EC2 ssh 代理 翻墙](http://yangrongquan.sinaapp.com/archives/25)