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

对于JMX，如果使用JDK7+（笔者线上的JDK版本是8），那么可以使用`com.sun.management.jmxremote.rmi.port`指定后续的通讯端口，这样就不会是动态分配的一个端口的了，而且可以指定跟注册端口同一个端口。

再测试了一下前面的两个配置项，发现其实根本不需要，所以最后的配置如下：

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


Tomcat有个Tomcat Manager工程就是通过JMXProxyServlet暴露JMX监控项的。

参考文章
-------

1. [Monitoring Tomcat](http://wiki.apache.org/tomcat/FAQ/Monitoring)
2. [Monitoring Tomcat with JMX](http://events.linuxfoundation.org/sites/events/files/slides/Monitoring%20Apache%20Tomcat%20with%20JMX.pdf)
3. [Monitoring and Managing Tomcat](https://tomcat.apache.org/tomcat-8.0-doc/monitoring.html)
4. [Monitoring Tomcat 7 with JMX](http://www.i-net-design.com/2011/07/15/monitoring-tomcat-7-with-jmc/)
5. [The Pros and Cons of Using Tomcat JMX](https://www.mulesoft.com/tcat/tomcat-jmx)
