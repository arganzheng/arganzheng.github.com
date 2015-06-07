---
title: tomcat监控
layout: post
---


### 法一、使用JMX（RMI）

配置允许远程JMX访问：

* -Dcom.sun.management.jmxremote 
* -Dcom.sun.management.jmxremote.port={port to access} 
* -Dcom.sun.management.jmxremote.authenticate=false 
* -Dcom.sun.management.jmxremote.ssl=false
* -Djava.rmi.server.hostname={optional, allow what ip to access this Tomcat}

具体配置如下：

$ cd {tomcat-folder}
$ vim bin/setenv.sh

	export CATALINA_OPTS="-Dcom.sun.management.jmxremote -Dcom.sun.management.jmxremote.port=8991 -Dcom.sun.management.jmxremote.authenticate=false -Dcom.sun.management.jmxremote.ssl=false -Djava.rmi.server.hostname=localhost"

由于线上防火墙只是开放了8000~9000范围的端口，所以这里配置了8991。重启tomcat，然后就可以通过8991端口访问JMX了。可以通过jconsole连接监控：

	jconsole 10.242.73.25:8991


但是发现一直连接不了。。谷歌了一下有人说是`java.rmi.server.hostname`必须配置成tomcat所在服务器的IP，并且不能是127.0.0.1。使用`hostname -i`得到机器IP，配置一下，结果还是不行，又有人说要加上`com.sun.management.jmxremote.local.only=false`，但是加上还是不行。最后发现了这篇文章：[The Madness of Tunneling JMX is Over!](http://realjenius.com/2012/11/21/java7-jmx-tunneling-freedom/)。里面提到一个配置项：`com.sun.management.jmxremote.rmi.port`，于是照着配置一下：

	export CATALINA_OPTS="-Dcom.sun.management.jmxremote -Dcom.sun.management.jmxremote.port=8991 -Dcom.sun.management.jmxremote.rmi.port=8991  -Dcom.sun.management.jmxremote.authenticate=false -Dcom.sun.management.jmxremote.ssl=false -Djava.rmi.server.hostname=10.242.73.25 -Dcom.sun.management.jmxremote.local.only=false"

发现一下子就登陆上去了。为什么会这样子呢？这是RMI的机制导致的 [Spring: Cannot connect to a JMX Server using RMI from behind a firewall](http://stackoverflow.com/questions/22306363/spring-cannot-connect-to-a-jmx-server-using-rmi-from-behind-a-firewall) :

> The initial port you define with `com.sun.management.jmxremote.port` is called a registry port and is only used to start negotiation and determine next port(s) to use for "real" communication. Java RMI mechanism uses dynamically allocated ports and in general is not compatible with firewalls.

**题外话**

RMI之所以使用的范围受限制主要有两方面原因，其一：必须要是java，平台的异构性受到限制；其二：穿越防火墙不方便。RMI穿越防火墙不方便主要是因为除了RMI服务注册的端口(默认1099)外，与RMI的通讯还需要另外的端口来传送数据，而另外的端口是随机分配的，这就非常不利于穿越防火墙。


对于JMX，如果使用JDK7+（笔者线上的JDK版本是8），那么可以使用`com.sun.management.jmxremote.rmi.port`指定后续的通讯端口，这样就不会是动态分配的一个端口的了，而且可以指定跟注册端口同一个端口。

再测试了一下前面的两个配置项，发现其实根本不需要（前提是你`hostname -i`返回的不是loopback地址），所以最后的配置如下：

	export CATALINA_OPTS="-Dcom.sun.management.jmxremote -Dcom.sun.management.jmxremote.port=8991 -Dcom.sun.management.jmxremote.rmi.port=8991  -Dcom.sun.management.jmxremote.authenticate=false -Dcom.sun.management.jmxremote.ssl=false”

整个tomcat的setenv.sh配置如下：

	# determine base directory; preserve where you're running from
	#echo "Path to $(basename $0) is $(readlink -f $0)"
	realpath=$(readlink -f "$0")
	filepath=$(dirname "$realpath")
	basedir=${filepath%/*}

	LOG_DIR=${basedir}/logs
	mkdir -p ${LOG_DIR}
	GC_LOG_DIR=${basedir}/logs/gc
	mkdir -p ${GC_LOG_DIR}
	GC_FILE_PATH="${GC_LOG_DIR}/gc-$(date +%s).log"

	JAVA_OPTS="$JAVA_OPTS -server -Xmn512M -Xms2048M -Xmx2048M -Djava.net.preferIPv4Stack=true -Djava.awt.headless=true -Dorg.apache.catalina.connector.RECYCLE_FACADES=false -XX:MetaspaceSize=128M -XX:MaxMetaspaceSize=256M -XX:+UseConcMarkSweepGC -XX:+UseParNewGC -XX:+CMSParallelRemarkEnabled -XX:+UseCMSCompactAtFullCollection -XX:CMSFullGCsBeforeCompaction=1 -XX:+CMSClassUnloadingEnabled -XX:+UseFastAccessorMethods -XX:+UseCMSInitiatingOccupancyOnly  -XX:SurvivorRatio=65536 -XX:MaxTenuringThreshold=0 -XX:CMSInitiatingOccupancyFraction=81 -XX:SoftRefLRUPolicyMSPerMB=0 -XX:+PrintGCDetails -XX:+PrintGCTimeStamps -verbose:gc -Xloggc:${GC_FILE_PATH} -XX:+UseGCLogFileRotation -XX:NumberOfGCLogFiles=10 -XX:GCLogFileSize=100M"
	JAVA_OPTS="$JAVA_OPTS -DLOG_DIR=${LOG_DIR} -Denv=idc"

	export CATALINA_OPTS="-Dcom.sun.management.jmxremote -Dcom.sun.management.jmxremote.port=8991 -Dcom.sun.management.jmxremote.rmi.port=8991 -Dcom.sun.management.jmxremote.authenticate=false -Dcom.sun.management.jmxremote.ssl=false"

server.xml配置如下：

	<?xml version='1.0' encoding='utf-8'?>

	<Server port="8005" shutdown="SHUTDOWN">
	  <!--APR library loader. Documentation at /docs/apr.html -->
	  <Listener className="org.apache.catalina.core.AprLifecycleListener" SSLEngine="on" />
	  <!-- Prevent memory leaks due to use of particular java/javax APIs-->
	  <Listener className="org.apache.catalina.core.JreMemoryLeakPreventionListener" />
	  <Listener className="org.apache.catalina.mbeans.GlobalResourcesLifecycleListener" />
	  <Listener className="org.apache.catalina.core.ThreadLocalLeakPreventionListener" />

	  <GlobalNamingResources>
	    <Resource name="UserDatabase" auth="Container"
	              type="org.apache.catalina.UserDatabase"
	              description="User database that can be updated and saved"
	              factory="org.apache.catalina.users.MemoryUserDatabaseFactory"
	              pathname="conf/tomcat-users.xml" />
	  </GlobalNamingResources>

	  <Service name="Catalina">
	    <!--The connectors can use a shared executor, you can define one or more named thread pools-->
	    <Executor name="tomcatThreadPool" namePrefix="catalina-exec-"
	        maxThreads="3000" minSpareThreads="10" />
	    <Connector executor="tomcatThreadPool"
	               port="8091" protocol="HTTP/1.1"
	               connectionTimeout="20000"
	               redirectPort="8443" />
	    <Engine name="Catalina" defaultHost="localhost">
	      <Realm className="org.apache.catalina.realm.LockOutRealm">
	        <Realm className="org.apache.catalina.realm.UserDatabaseRealm"
	               resourceName="UserDatabase"/>
	      </Realm>
	      <Host name="localhost"  appBase="webapps"
	            unpackWARs="true" autoDeploy="true">
	      </Host>
	    </Engine>
	  </Service>
	</Server>


让JMX的两个RMI端口固定还有另一种方式。是配置Tomcat的JmxRemoteLifecycleListener并且将一个`catalina-jmx-remote.jar`的extras jar包放在Tomcat的lib目录下。
具体参见：[Connecting Visual VM to Tomcat 7](http://blog.markshead.com/1129/connecting-visual-vm-to-tomcat-7/)。这里不赘述了。但是这种方式相对于前面的一个简单的启动项而言要麻烦得多，所以不推荐，除非前面的方式行不通。

最后还有一种方式，就是走SSH隧道。这种方式是在没法指定RMI固定端口的情况下使用。也是比较麻烦的。比如下面的jstatd就没有选项可以指定RMI固定端口了。我们在下面介绍VisualVM的时候介绍这种方式。

**TIPS** 使用VisualVM连接JMX

上面使用的是jconsole连接JMX，但是实际上JDK6以后，就可以使用VisualVM了，而且不需要单独下载，跟JConsole一样，都是JDK的一部分，位于JDK跟目录的bin文件夹下，名称叫做jvisualvm。可以说它是jconsole的超集。JConsole只是单纯的收集和展示JMX数据（即在MBeans上进行浏览和操作），而visualvm则还可以通过 jvmstat、SA（Serviceability Agent）以及 Attach API 等多种方式从程序运行时获得实时数据，从而进行动态的性能分析。同时，它能自动选择更快更轻量级的技术尽量减少性能分析对应用程序造成的影响，提高性能分析的精度。而且，即使是JMX展示，界面也比jconsole要好看的多。还有丰富的插件体系。所以建议如果本地环境是JDK6+的，直接用visualvm就可以了。

VisualVM目前支持两种remote connection方式，分别是jstatd和JMX方式。JMX方式就是我们前面介绍的。下面简单介绍一下jstatd方式。

不同于JMX，jstatd方式需要在远程服务器上启动[jstatd后台进程](http://docs.oracle.com/javase/6/docs/technotes/tools/share/jstatd.html)：

> The jstatd tool is an RMI server application that monitors for the creation and termination of instrumented HotSpot Java virtual machines (JVMs) and provides a interface to allow remote monitoring tools to attach to JVMs running on the local host.
>
> The jstatd server requires the presence of an RMI registry on the local host. The jstatd server will attempt to attach to the RMI registry on the default port, or on the port indicated by the -p port option. If an RMI registry is not found, one will be created within the jstatd application bound to the port indicated by the -p port option or to the default RMI registry port if -p port is omitted. Creation of an internal RMI registry can be inhibited by specifying the -nr option.

直接在远程机器上运行jstatd是会报错的：

	[work@xxx.xxx] apache-tomcat-8.0.8-nantianmen-1]$ jstatd
	Could not create remote object
	access denied ("java.util.PropertyPermission" "java.rmi.server.ignoreSubClasses" "write")
	java.security.AccessControlException: access denied ("java.util.PropertyPermission" "java.rmi.server.ignoreSubClasses" "write")
		at java.security.AccessControlContext.checkPermission(AccessControlContext.java:457)
		at java.security.AccessController.checkPermission(AccessController.java:884)
		at java.lang.SecurityManager.checkPermission(SecurityManager.java:549)
		at java.lang.System.setProperty(System.java:789)
		at sun.tools.jstatd.Jstatd.main(Jstatd.java:139)

需要先需要准备一个java.policy文件，保存到如/home/work/jstatd.all.policy

	grant codebase "file:${java.home}/../lib/tools.jar" {
	   permission java.security.AllPermission;
	};

表示允许jstatd服务具有JVM全部的访问权限。然后就可以正常启动了：	

	[work@xxx.xxx ~]$ vim jstatd.all.policy
	[work@xxx.xxx ~]$ jstatd -J-Djava.security.policy=jstatd.all.policy -J-Djava.rmi.server.logCalls=true
	Jun 03, 2015 7:54:09 PM sun.rmi.server.UnicastServerRef logCall
	FINER: RMI TCP Connection(1)-10.242.73.25: [10.242.73.25: sun.rmi.registry.RegistryImpl[0:0:0, 0]: void rebind(java.lang.String, java.rmi.Remote)]
	Jun 03, 2015 7:54:09 PM sun.rmi.server.UnicastServerRef logCall
	FINER: RMI TCP Connection(2)-10.242.73.25: [10.242.73.25: sun.rmi.transport.DGCImpl[0:0:0, 2]: java.rmi.dgc.Lease dirty(java.rmi.server.ObjID[], long, java.rmi.dgc.Lease)]


但是请注意，jstatd同样是基于RMI的通讯机制，所以跟前面的JMX方式一样，在有防火墙的前提下，需要配置一下RMI的端口才行，否则默认是1099注册端口。通过-p 8999指定。

	jstatd -J-Djava.security.policy=jstatd.all.policy -p 8999 -J-Djava.rmi.server.logCalls=true 

但是很遗憾，-p选项只是指定了RMI的注册端口，并没有选项可以指定RMI的通讯端口。前面的`com.sun.management.jmxremote.rmi.port`配置项也是不行的。

这时候需要用到另外一种翻墙技术——SSH隧道。具体步骤如下：

1、启动jstatd

	jstatd -J-Djava.security.policy=jstatd.all.policy -p 8999 -J-Djava.rmi.server.logCalls=true 

2、找到RMI的随机通信端口

	netstat -nlp | grep jstatd 

可以看到类似如下输出：

	tcp        0      0 0.0.0.0:56704               0.0.0.0:*                   LISTEN      11140/jstatd
	tcp        0      0 0.0.0.0:8999                0.0.0.0:*                   LISTEN      11140/jstatd

56704就是RMI的随机通讯端口。如果你有root权限，你当然可以修改IP Tables为其打开防火墙。但是一般都没有，同时也麻烦，因为端口是动态变化的。使用SSH隧道相对简单一些：

3、为动态端口开启SSH隧道

	ssh -L 56704:localhost:56704 login_name@host_name

4、 使用VisualVM连接

这里需要注意的是使用了SSH隧道之后，选择的是"Local" machine，而不是”Remote"。

具体参见：[Using VisualVM to connect to a remote jstatd instance through a firewall [duplicate]](http://stackoverflow.com/questions/7424519/using-visualvm-to-connect-to-a-remote-jstatd-instance-through-a-firewall)

或者使用SOCKS代理[VisualVM: Monitoring Remote JVM Over SSH (JMX Or Not)](http://java.dzone.com/articles/visualvm-monitoring-remote-jvm)。

但是这两种方式对于需要通过跳板机才能登陆到服务器的仍然不适用。所以说RMI真是蛋疼啊。所以还是学习用命令行吧，主要是jstat。

当然，还有另外一种方式，就是使用Java的Agent模式进行连接。也比较复杂，具体参考：[JMX connectivity through the firewall](https://olegz.wordpress.com/2009/03/23/jmx-connectivity-through-the-firewall/)。总之，都很麻烦。


### 法二、JMXProxyServlet（HTTP）

> Tomcat offers an alternative to using remote (or even local) JMX connections while still giving you access to everything JMX has to offer: Tomcat's JMXProxyServlet.
> 
> The JMXProxyServlet allows a client to issue JMX queries via an HTTP interface. This technique offers the following advantages over using JMX directly from a client program:
> * You don't have to launch a full JVM and make a remote JMX connection just to ask for one small piece of data from a runing server
> * You don't have to know how to work with JMX connections
> * You don't need any of the complex configuration covered in the rest of this page
> * Your client program does not have to be written in Java
> 
> A perfect example of JMX overkill can be seen in the case of popular server-monitoring software such as Nagios or Ichinga: if you want to monitor 10 items via JMX, you will have to launch 10 JVMs, make 10 JMX connections, and then shut them all down every few minutes. With the JMXProxyServlet, you can make 10 HTTP connections and be done with it.


Tomcat有个Tomcat Manager工程就是通过JMXProxyServlet暴露JMX监控项的。其实就是将MXBean以HTTP的方式暴露出去，可以避免RMI带来的各种不便。


参考文章
-------

1. [Monitoring Tomcat](http://wiki.apache.org/tomcat/FAQ/Monitoring)
2. [Monitoring Tomcat with JMX](http://events.linuxfoundation.org/sites/events/files/slides/Monitoring%20Apache%20Tomcat%20with%20JMX.pdf)
3. [Monitoring and Managing Tomcat](https://tomcat.apache.org/tomcat-8.0-doc/monitoring.html)
4. [Monitoring Tomcat 7 with JMX](http://www.i-net-design.com/2011/07/15/monitoring-tomcat-7-with-jmc/)
5. [The Pros and Cons of Using Tomcat JMX](https://www.mulesoft.com/tcat/tomcat-jmx)
6. [Monitoring and Management for the Java Platform](http://docs.oracle.com/javase/7/docs/technotes/guides/management/)
7. [Remote Debugging using JConsole, JMX and SSH Tunnels](http://blog.cantremember.com/debugging-with-jconsole-jmx-ssh-tunnels/)
