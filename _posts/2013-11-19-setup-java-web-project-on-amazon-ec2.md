---
layout: post
title: 使用亚马逊EC2搭建web工程
---


申请Amazon EC2免费主机实例
------------------------

具体参考这篇文章[使用免费的Amazon EC2 服务搭建nginx+wordpress](http://www.hiyangqi.com/linux/use-amazon-ec2-and-nginx-to-launch-wordpress.html)，非常详细，这里就不赘述了。仅仅列出一些信息供参考。


arganzheng-mbp:~ argan$ chmod 400 argan-aws-ec2.pem argan-aws-ec2.pem 
arganzheng-mbp:~ argan$ ls -ahl |grep pem
-r--------@  1 argan  staff   1.7K 11 19 09:54 argan-aws-ec2.pem
arganzheng-mbp:~ argan$ ssh -i argan-aws-ec2.pem ec2-user@54.201.85.167
The authenticity of host '54.201.85.167 (54.201.85.167)' can't be established.
RSA key fingerprint is 18:9b:80:a0:a3:d0:9f:b1:25:5a:d5:08:5a:fe:d0:a3.
Are you sure you want to continue connecting (yes/no)? yes
Warning: Permanently added '54.201.85.167' (RSA) to the list of known hosts.

       __|  __|_  )
       _|  (     /   Amazon Linux AMI
      ___|\___|___|

https://aws.amazon.com/amazon-linux-ami/2013.09-release-notes/
No packages needed for security; 7 packages available
Run "sudo yum update" to apply all updates.
[ec2-user@ip-172-31-36-169 home]$ uname -a
Linux ip-172-31-36-169 3.4.68-59.97.amzn1.x86_64 #1 SMP Tue Nov 5 07:40:09 UTC 2013 x86_64 x86_64 x86_64 GNU/Linux


安装需要的软件
------------


Amazon Linux AMI 本身预安装了一些软件：

> The Amazon Linux AMI is an EBS-backed, PV-GRUB image. It includes Linux 3.4, AWS tools, and repository access to PostgreSQL, Python, Ruby, and Tomcat.


	[ec2-user@ip-172-31-36-169 home]$ echo ${JAVA_HOME}
	/usr/lib/jvm/jre

	[ec2-user@ip-172-31-36-169 home]$ java -version
	java version "1.6.0_24"
	OpenJDK Runtime Environment (IcedTea6 1.11.14) (amazon-65.1.11.14.57.amzn1-x86_64)
	OpenJDK 64-Bit Server VM (build 20.0-b12, mixed mode)


可以看到它预装了JRE环境，使用的是OpenJDK Runtime Environment。但是值得注意的是仅仅是JRE。所以要编译java代码的话还是要安装个JDK。所以没有什么意思。另外，它确实有个yum的repos，里面有一些常用的软件，比如tomcat:

	[ec2-user@ip-172-31-36-169 home]$ yum info tomcat6
	已加载插件：priorities, update-motd, upgrade-helper
	可安装的软件包
	名称    ：tomcat6
	架构    ：noarch
	版本    ：6.0.37
	发布    ：1.2.amzn1
	大小    ：106 k
	源    ：amzn-main/latest
	简介    ： Apache Servlet/JSP Engine, RI for Servlet 2.5/JSP 2.1 API
	网址    ：http://tomcat.apache.org/
	协议    ： ASL 2.0
	...

	[ec2-user@ip-172-31-36-169 home]$ yum info tomcat7
	已加载插件：priorities, update-motd, upgrade-helper
	可安装的软件包
	名称    ：tomcat7
	架构    ：noarch
	版本    ：7.0.42
	发布    ：1.33.amzn1
	大小    ：89 k
	源    ：amzn-main/latest
	简介    ： Apache Servlet/JSP Engine, RI for Servlet 3.0/JSP 2.2 API
	网址    ：http://tomcat.apache.org/
	协议    ： ASL 2.0
	...

但是还是有很多软件找不到，比如maven。所以总体来说还是选择ubuntu比较方便，除非要使用AWS tools。


### 安装git

这个比较简单，仓库中有，直接：

	sudo yum install git

就可以了。

然后就可以把工程代码拉到EC2上：

	$ mkdir code
    $ cd code 
    $ git clone git@github.com:arganzheng/spring-mvc-rest-event-push-sample.git

会报没有权限。需要把你的EC2的SSH Key配置到github账户上。具体参见这篇文档 [Generating SSH Keys](https://help.github.com/articles/generating-ssh-keys)。

测试通过之后，再次运行git clone命令，哗的一下就拉下来了。

### 安装maven

我们的sample工程是一个标准的maven工程，所以需要安装一个maven来install。可惜EC2的这个yum仓库中没有maven的源。需要自己安装一下，不过maven是java代码，我们已经有JRE环境，所以安装很简单

    $ mkdir downloads
    $ cd downloads/
    $ wget http://mirrors.gigenet.com/apache/maven/maven-3/3.0.5/binaries/apache-maven-3.0.5-bin.tar.gz
    $ cd ..
    $ mkdir install
    $ cd downloads/
    $ tar -zxvf apache-maven-3.0.5-bin.tar.gz -C /home/ec2-user/install/
    $ cd ../install/
    $ ls
    $ cd apache-maven-3.0.5/
    $ vim /etc/profile.d/maven.sh
    $ cat /etc/profile.d/maven.sh

		export M2_HOME=/home/ec2-user/install/apache-maven-3.0.5
		export M2=$M2_HOME/bin
		PATH=$M2:$PATH 

    $ source /etc/profile.d/maven.sh

	[ec2-user@ip-172-31-36-169 apache-maven-3.0.5]$ mvn -version
	Apache Maven 3.0.5 (r01de14724cdef164cd33c7c8c2fe155faf9602da; 2013-02-19 13:51:28+0000)
	Maven home: /home/ec2-user/install/apache-maven-3.0.5
	Java version: 1.6.0_24, vendor: Sun Microsystems Inc.
	Java home: /usr/lib/jvm/java-1.6.0-openjdk-1.6.0.0.x86_64/jre
	Default locale: zh_CN, platform encoding: UTF-8
	OS name: "linux", version: "3.4.68-59.97.amzn1.x86_64", arch: "amd64", family: "unix"

但是当我们运行mvn clean install的时候，发现报错了，因为编译java代码需要JDK而不是JRE。所以我们需要安装一个JDK。


### 安装JDK

本来是很简单愉快的事情，但是Oracle太恶心了。居然需要注册才能下载。。不能wget。而且我的账号貌似被锁住了。怒了，直接Google找到一个安装源：[Java.Net](https://jdk6.java.net/download.html)。简单的安装一下就可以了。不过安装完之后要修改一下JAVA_HOME环境变量，让它指向你的JDK。

### 安装tomcat

yum上有tomcat6和tomcat7，不过建议还是自己安装一下，反正是java代码，安装就是简单解压而已（这时候Java的优势就体现出来了）。

具体可以参考这篇：[Setting up Java+Tomcat on EC2](https://sites.google.com/site/amistrongeryet/setting-up-java-tomcat-on-ec2)。

安装完成之后就配置一下Context，将下面这段配置加到server.xml中就可以了：

	<Context docBase="/home/ec2-user/.m2/repository/com/qq/ecc/openapi/spring-mvc-rest-event-push-sample-1.0.0.war" path="sample" />

顺便修改一下server.xml，把监听端口改成80，不过改完之后要配置一下EC2实例的`Secret Groups`的Inbound策略。

而且要以sudo权限启动：

    $ sudo bin/startup.sh

然后用你的公网IP访问看看：

	http://54.201.85.167/sample/v1/users/argan

或者DNS域名:

	http://ec2-54-201-85-167.us-west-2.compute.amazonaws.com/sample/v1/users/argan


It works! :)


----


补记：安装nginx和配置CNAME
=========================


注意到EC2的DNS域名非常的长，基本记不住。而且sample工程霸占了80端口，以后其他的工程就没法用了。最好是用nginx来转发。

一、安装nginx
-------------


源代码安装很简单，具体参见 http://wiki.nginx.org/Install。需要注意的默认会安装到/usr/local/nginx目录下，建议通过--prefix选项更改。
安装完成之后需要配置一下nginx：

	server {
        listen       80;
        server_name  ec2.arganzheng.me ec2-54-201-85-167.us-west-2.compute.amazonaws.com; 

        location / {
            proxy_pass http://localhost:8080;
        }
    }

    
二、配置CNAME，指向你的EC2 DNS域名
----------------------------------


进入Goddy账户，增加一个CNAME(Alias):

    ec2 ec2-54-201-85-167.us-west-2.compute.amazonaws.com
    
保存即可。

--EOF--

