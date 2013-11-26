---
layout: post
title: Java Heap OOM问题
---


今天线上API机器出问题了，很多机器告警，tomcat重启不久又挂掉。

看了一下日志，发现是OOM了：

    java.lang.OutOfMemoryError: Java heap space
    Dumping heap to java_pid21673.hprof ...
    Heap dump file created [1934558576 bytes in 8.670 secs]
    [Unloading class sun.reflect.GeneratedConstructorAccessor17]
    2013-7-28 12:03:35 org.apache.tomcat.util.http.Parameters processParameters
    320305317242: Invalid chunk starting at byte [66] and ending at byte [66] with a value of [null] ignored
    2013-7-28 12:03:46 org.apache.coyote.http11.Http11Protocol$Http11ConnectionHandler process
    321317326330: Error reading request, ignored
    java.lang.OutOfMemoryError: GC overhead limit exceeded
    Exception in thread "catalina-exec-795" java.lang.NullPointerException
            at java.util.concurrent.ConcurrentLinkedQueue.offer(ConcurrentLinkedQueue.java:273)
            at org.apache.coyote.http11.Http11Protocol$Http11ConnectionHandler$1.offer(Http11Protocol.java:551)
            at org.apache.coyote.http11.Http11Protocol$Http11ConnectionHandler$1.offer(Http11Protocol.java:568)
            at org.apache.coyote.http11.Http11Protocol$Http11ConnectionHandler.process(Http11Protocol.java:632)
            at org.apache.tomcat.util.net.JIoEndpoint$SocketProcessor.run(JIoEndpoint.java:396)
            at java.util.concurrent.ThreadPoolExecutor$Worker.runTask(ThreadPoolExecutor.java:886)
            at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:908)
            at java.lang.Thread.run(Thread.java:662)
    2013-7-28 12:03:51 org.apache.coyote.http11.Http11Protocol$Http11ConnectionHandler process
    321317326330: Error reading request, ignored
    
好在crash掉之后生成了java_pid21673.hprof文件，gzip再sz到本地用Eclipse MAT插件查看了一下。一下子就非常了然了：

<a href="http://www.flickr.com/photos/arganzheng/9380562455/" title="oom by arganzheng, on Flickr"><img src="http://farm4.staticflickr.com/3726/9380562455_b9320df526_o.jpg" width="893" height="253" alt="oom"></a>

基本heap都被byte[]给占光了。

再看一下dominator视图，一下子就知道是哪个接口出问题了：

<a href="http://www.flickr.com/photos/arganzheng/9380576179/" title="oom-dominator by arganzheng, on Flickr"><img src="http://farm6.staticflickr.com/5325/9380576179_953f31f76a_o.jpg" width="1146" height="320" alt="oom-dominator"></a>


这个CPS接口原来就出过问题。原来的缓冲区是1M，但是因为后台返回的数据特别大，后台人员没有采取分页或者分文件的方式，而是直接将缓冲区大小调整到10M了（刚好跟dominator视图的byte[]大小对应上了），然后超时时间设置为3s，然后就是OOM了。。


### OOM时一些常用的命令和工具

##### 1. jps JVM Process Status Tool

作用：显示指定系统内所有的HotSpot虚拟机进程

##### 2. jstat JVM Statistics Monitoring Tool 虚拟机统计信息监控

作用：用于收集HotSpot虚拟机各方面的运行数据，比如类装载、内存、垃圾收集、JIT编译等运行数据，是运行期间命令行下定位虚拟机性能问题的首选工具。

命令格式：

    jstat [ option vimid [interval[s|ms] [count]] ]
    
其中option代表用户希望查询的虚拟机信息。主要分为3类：类装载、垃圾收集和运行期编译状况。其中查询垃圾收集信息最常见。


    webadmin@ISD-223-106-sles10-64:/home/arganzheng> jstat -gc 27624 3000
     S0C    S1C    S0U    S1U      EC       EU        OC         OU       PC     PU    YGC     YGCT    FGC    FGCT     GCT   
    192.0  192.0   0.0    0.0   43264.0  16467.2   87424.0     6399.5   131072.0 29151.6    495    2.587  241    52.412   55.000
    192.0  192.0   0.0    0.0   43264.0  16467.2   87424.0     6399.5   131072.0 29151.6    495    2.587  241    52.412   55.000
    192.0  192.0   0.0    0.0   43264.0  17333.8   87424.0     6399.5   131072.0 29151.6    495    2.587  241    52.412   55.000
    192.0  192.0   0.0    0.0   43264.0  17333.8   87424.0     6399.5   131072.0 29151.6    495    2.587  241    52.412   55.000


##### 3. jmap Memory Map for Java 堆内存监控

作用：

1. 生成虚拟机的存转储快照（heapdump文件）
2. 查询finalize执行队列
3. java堆和永久代的详细信息，如空间使用率、当前用的是哪种收集器等


    webadmin@app_tencent_238:~> jmap -heap 26276  
    Attaching to process ID 26276, please wait...
    Debugger attached successfully.
    Server compiler detected.
    JVM version is 20.4-b02
    
    using thread-local object allocation.
    Parallel GC with 8 thread(s)
    
    Heap Configuration:
       MinHeapFreeRatio = 40
       MaxHeapFreeRatio = 70
       MaxHeapSize      = 2147483648 (2048.0MB)
       NewSize          = 1073741824 (1024.0MB)
       MaxNewSize       = 1073741824 (1024.0MB)
       OldSize          = 5439488 (5.1875MB)
       NewRatio         = 2
       SurvivorRatio    = 8
       PermSize         = 268435456 (256.0MB)
       MaxPermSize      = 268435456 (256.0MB)
    
    Heap Usage:
    PS Young Generation
    Eden Space:
       capacity = 1031012352 (983.25MB)
       used     = 509640488 (486.03104400634766MB)
       free     = 521371864 (497.21895599365234MB)
       49.431074905298516% used
    From Space:
       capacity = 21037056 (20.0625MB)
       used     = 15845120 (15.111083984375MB)
       free     = 5191936 (4.951416015625MB)
       75.32004478193146% used
    To Space:
       capacity = 21037056 (20.0625MB)
       used     = 0 (0.0MB)
       free     = 21037056 (20.0625MB)
       0.0% used
    PS Old Generation
       capacity = 1073741824 (1024.0MB)
       used     = 905028440 (863.1023788452148MB)
       free     = 168713384 (160.89762115478516MB)
       84.28734168410301% used
    PS Perm Generation
       capacity = 268435456 (256.0MB)
       used     = 54234160 (51.72172546386719MB)
       free     = 214201296 (204.2782745361328MB)
       20.20379900932312% used

##### 4. jstack Stack Trace for Java 线程以及栈内存监控

作用：生成虚拟机当前时刻的线程快照（一般称之为treaddump或者javacore文件）。线程快照就是当前虚拟机内每一条线程正在执行的方法堆栈的集合，生成线程快照的主要目的是定位线程出现长时间停顿的原因，如线程间死锁、死循环、请求外部资源导致的长时间等待等都是导致线程长时间停顿的常见原因。

##### 5. 图形化工具JConsole和VisualVM

##### 6. BTrace动态日志跟踪

作用：再不停止目标程序的前提下，通过HotSpot虚拟机的HotSwap技术动态加入原本并不存在的调试代码（非常类似于JMocket）。

### 参考文章

1. [Java and android performance analysis, Eclipse Memory Analyzer](http://kohlerm.blogspot.com/2009/02/memory-leaks-are-easy-to-find.html) 
2. [Shallow and retained sizes](http://www.yourkit.com/docs/90/help/sizes.jsp)
3. [10 Tips for using the Eclipse Memory Analyzer](http://eclipsesource.com/blogs/2013/01/21/10-tips-for-using-the-eclipse-memory-analyzer/)
4. 深入理解Java虚拟机 by 周志明