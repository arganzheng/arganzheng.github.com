---
title: Tomcat调优
layout: post
---

Tomcat有线程池的概念，比如下面这个配置：

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

配置了一个share thread pool。默认实现是StandardThreadExecutor，这个Exector底层其实是使用了Jdk的ThreadPoolExecutor，minSpareThreads实际上指的就是corePoolSize，对它了解的人都知道，将请求数小于等于它时会马上创建一个新的线程，如果大于它就会把新的请求放到队列中，如果请求数超过corePoolSize加上队列的长度但又小于maxPoolSize时也会马上创建新的线程。而Tomcat中默认是使用了一个无界队列TaskQueue，它继承自LinkedBlockingQueue<Runnable>, 这种情况下如果没有配置maxQueueSize的话，配置maxThreads其实是没有意义的，因为默认创建的是无界队列根本没有读取配置的参数几乎永远不会满，自然不存在超过需要用maxThreads来判断了。但是实际上我用JMX观察过我们的tomcat，发现什么的配置tomcat的活动线程数可以达到1000,然后queueSize基本都是0。也就是说tomcat的线程池是优先创建线程直到maxThreads，再将请求扔队列的。真的是这样子吗？测试一下不就知道了吗？把tomcat的[ThreadPoolExecutor](http://grepcode.com/file/repo1.maven.org/maven2/org.apache.geronimo.ext.tomcat/catalina/7.0.39.2/org/apache/catalina/core/StandardThreadExecutor.java#StandardThreadExecutor)和[TaskQueue](http://grepcode.com/file/repo1.maven.org/maven2/org.apache.tomcat/tomcat-coyote/7.0.52/org/apache/tomcat/util/threads/TaskQueue.java)源码copy到Ecliipse，然后就可以开始测试了：


	import java.util.concurrent.LinkedBlockingQueue;
	import java.util.concurrent.TimeUnit;

	public class QuickTest {
	    public static void main(String[] args) throws InterruptedException {
	    	//  LinkedBlockingQueue<Runnable> queue = new LinkedBlockingQueue<Runnable>();
	        TaskQueue taskqueue = new TaskQueue();
	        final ThreadPoolExecutor executor = new ThreadPoolExecutor(2, 6, 60000, TimeUnit.MILLISECONDS, taskqueue);
	        taskqueue.setParent((ThreadPoolExecutor) executor);

	        for (int i = 0; i < 100; i++) {
	            System.out.println("Submit " + (i + 1) + " task to executor..");
	            executor.submit(new Runnable() {

	                @Override
	                public void run() {
	                    for (int j = 0; j < 10; j++) {
	                        System.out.println(executor.toString());
	                        try {
	                            Thread.sleep(1000);
	                        } catch (InterruptedException e) {
	                            // TODO Auto-generated catch block
	                            e.printStackTrace();
	                        }
	                    }
	                }
	            });
	        }

	        Thread.currentThread().join();
	    }
	}

测试结果发现，如果使用JDK的LinkedBlockingQueue，那么就是JDK的线程池方式，如果使用TaskQueue，那么就是先创建线程的方式。也就是说tomcat的TaskQueue做了手脚，看一下代码就简单明了了：


	package org.apache.tomcat.util.threads;

	import java.util.Collection;
	import java.util.concurrent.LinkedBlockingQueue;
	import java.util.concurrent.RejectedExecutionException;
	import java.util.concurrent.TimeUnit;

	/**
	 * As task queue specifically designed to run with a thread pool executor.
	 * The task queue is optimised to properly utilize threads within
	 * a thread pool executor. If you use a normal queue, the executor will spawn threads
	 * when there are idle threads and you wont be able to force items unto the queue itself
	 * @author fhanik
	 *
	 */
	public class TaskQueue extends LinkedBlockingQueue<Runnable> {

	    private static final long serialVersionUID = 1L;

	    private ThreadPoolExecutor parent = null;

	    // no need to be volatile, the one times when we change and read it occur in
	    // a single thread (the one that did stop a context and fired listeners)
	    private Integer forcedRemainingCapacity = null;

	    public TaskQueue() {
	        super();
	    }

	    public TaskQueue(int capacity) {
	        super(capacity);
	    }

	    public TaskQueue(Collection<? extends Runnable> c) {
	        super(c);
	    }

	    public void setParent(ThreadPoolExecutor tp) {
	        parent = tp;
	    }

	    ...

	    @Override
	    public boolean offer(Runnable o) {
	      //we can't do any checks
	        if (parent==null) return super.offer(o);
	        //we are maxed out on threads, simply queue the object
	        if (parent.getPoolSize() == parent.getMaximumPoolSize()) return super.offer(o);
	        //we have idle threads, just add it to the queue
	        if (parent.getSubmittedCount()<(parent.getPoolSize())) return super.offer(o);
	        //if we have less threads than maximum force creation of a new thread
	        if (parent.getPoolSize()<parent.getMaximumPoolSize()) return false;
	        //if we reached here, we need to add it to the queue
	        return super.offer(o);
	    }

	    ...

	}

关键就是在offer方法。另外，看到有些文章说tomcat connector默认的处理协议是BIO方式(http://yikebocai.com/2014/11/tomcat-source-code-study-3/)，这样100个线程池最多只能处理100个并发。但是查阅了一下[tomcat的官方文档](https://tomcat.apache.org/tomcat-8.0-doc/config/http.html)，其实并不是这样子的。

> **protocol**
> 
> Sets the protocol to handle incoming traffic. The default value is HTTP/1.1 which uses an auto-switching mechanism to select either a non blocking Java NIO based connector or an APR/native based connector. If the PATH (Windows) or LD_LIBRARY_PATH (on most unix systems) environment variables contain the Tomcat native library, the APR/native connector will be used. If the native library cannot be found, the non blocking Java based connector will be used. Note that the APR/native connector has different settings for HTTPS than the Java connectors.
To use an explicit protocol rather than rely on the auto-switching mechanism described above, the following values may be used: 
>
> org.apache.coyote.http11.Http11Protocol - blocking Java connector
org.apache.coyote.http11.Http11NioProtocol - non blocking Java NIO connector
org.apache.coyote.http11.Http11Nio2Protocol - non blocking Java NIO2 connector
org.apache.coyote.http11.Http11AprProtocol - the APR/native connector.
Custom implementations may also be used.
Take a look at our [Connector Comparison chart](https://tomcat.apache.org/tomcat-8.0-doc/config/http.html#Connector_Comparison). The configuration for both Java connectors is identical, for http and https.
For more information on the APR connector and APR specific SSL settings please visit the APR documentation

也就是说从tomcat6开始就支持NIO方式了，tomcat8开始则支持NIO2，也就是AIO。默认是HTTP/1.1，这是自动切换的方式，一般来说线上都是会默认使用Http11NioProtocol。这个可以通过在tomcat的启动日志中看到这么一行信息：10-Jun-2015 14:29:55.255 INFO [main] org.apache.coyote.AbstractProtocol.init Initializing ProtocolHandler ["http-nio-8094"]。或者通过JMX也可以看到。

另外，这篇文章介绍了tomcat的参数调优支持13K并发连接数，其中还配置了TCP层面的东西，比如buffer大小：

  	<Connector port="8080" 
               connectionTimeout="200000" 
               redirectPort="8443" 
		protocol="org.apache.coyote.http11.Http11NioProtocol" 
		maxThreads="32000" 
		socket.appReadBufSize="1024" 
		socket.appWriteBufSize="1024" 
		bufferSize="1024"/>

可以配置然后用AB压测一下看看。

另外，最近看到几篇文章，感觉真的要遇到问题然后深入代码级别才能学到深层次的东东。比如这篇：[Tomcat7.0.26的连接数控制bug的问题排查](http://ifeve.com/tomcat7-0-26-connect-bug/)。还有就是要善于测试和通过实验验证自己的想法，比如这篇：[Tomcat-connector的微调(1): acceptCount参数](http://ifeve.com/tomcat-connector-tuning-1/)。要多看看博客，多向牛人学习了。

推荐阅读
-------

1. [Cool, Tomcat is able to handle more than 13,000 concurrent connections.](http://blog.krecan.net/2010/05/02/cool-tomcat-is-able-to-handle-more-than-13000-concurrent-connections/)
2. [Tomcat-connector的微调(1): acceptCount参数](http://ifeve.com/tomcat-connector-tuning-1/)
3. [Tomcat-connector的微调(2): maxConnections, maxThreads](http://ifeve.com/tomcat-connector-tuning-2/)
4. [Tomcat-connector的微调(3): processorCache与socket.processorCache](http://ifeve.com/tomcat-connector-tuning-3/)
5. [Tomcat7.0.26的连接数控制bug的问题排查](http://ifeve.com/tomcat7-0-26-connect-bug/)
