---
title: 一个java大堆引发的『血案』
layout: post
tags: [java, jvm, high performance, tunning]
catalog: true
---


### 现象


线上预估服务扩容新增了两台机器，同样的代码，同样的机器配置，但是有一台机器的平均时延就是比另一台机器慢一些。

#### 1、64机器（时延慢一些的机器）

JVM启动参数：

```sh
/home/app/predictor-serving/jdk/jre/bin/java -Djava.util.logging.config.file=/home/app/predictor-serving/tomcat/conf/logging.properties -Djava.util.logging.manager=org.apache.juli.ClassLoaderLogManager -server -Xmx32g -Xms32g -XX:+UseG1GC -XX:MaxGCPauseMillis=200 -XX:InitiatingHeapOccupancyPercent=8 -XX:ParallelGCThreads=5 -XX:ConcGCThreads=5 -XX:+HeapDumpOnOutOfMemoryError -Xss512k -Dapp.name=predictor-serving -Dapp.env=prd -Dconfig.version=54 -Dconfig.host=vivocfg-agent.prd.bj01.vivo.lan:8080/vivocfg -Djdk.tls.ephemeralDHKeySize=2048 -Djava.protocol.handler.pkgs=org.apache.catalina.webresources -classpath /home/app/predictor-serving/tomcat/bin/bootstrap.jar:/home/app/predictor-serving/tomcat/bin/tomcat-juli.jar -Dcatalina.base=/home/app/predictor-serving/tomcat -Dcatalina.home=/home/app/predictor-serving/tomcat -Djava.io.tmpdir=/home/app/predictor-serving/tomcat/temp org.apache.catalina.startup.Bootstrap start
```

GC 情况：

```sh
[root@predictor-serving-prd-10-21-8-64.v-bj-1.vivo.lan:/root]
# jstat -gcutil 9886 1000 20
  S0     S1     E      O      M     CCS    YGC     YGCT    FGC    FGCT     GCT
  0.00 100.00  87.40  23.89  96.36      -     86    3.508     0    0.000    3.508
  0.00 100.00  16.02  23.80  96.36      -     87    3.527     0    0.000    3.527
  0.00 100.00  40.20  23.80  96.36      -     87    3.527     0    0.000    3.527
  0.00 100.00  60.50  23.80  96.36      -     87    3.527     0    0.000    3.527
  0.00 100.00  79.55  23.80  96.36      -     87    3.527     0    0.000    3.527
  0.00 100.00  13.50  23.90  96.37      -     88    3.542     0    0.000    3.542
  0.00 100.00  38.01  23.90  96.37      -     88    3.542     0    0.000    3.542
  0.00 100.00  64.00  23.90  96.37      -     88    3.542     0    0.000    3.542
  0.00 100.00  84.02  23.90  96.37      -     88    3.542     0    0.000    3.542
  0.00 100.00   9.31  23.82  96.37      -     89    3.553     0    0.000    3.553
  0.00 100.00  26.92  23.82  96.37      -     89    3.553     0    0.000    3.553
  0.00 100.00  49.65  23.82  96.37      -     89    3.553     0    0.000    3.553
  0.00 100.00  73.31  23.82  96.37      -     89    3.553     0    0.000    3.553
  0.00 100.00   1.24  23.81  96.37      -     90    3.565     0    0.000    3.565
  0.00 100.00  24.36  23.81  96.37      -     90    3.565     0    0.000    3.565
  0.00 100.00  44.22  23.81  96.37      -     90    3.565     0    0.000    3.565
  0.00 100.00  69.05  23.81  96.37      -     90    3.565     0    0.000    3.565
  0.00 100.00  93.87  23.81  96.37      -     90    3.565     0    0.000    3.565
  0.00 100.00  23.27  23.82  96.37      -     91    3.578     0    0.000    3.578
  0.00 100.00  51.67  23.82  96.37      -     91    3.578     0    0.000    3.578
```

机器负载：

```sh
[root@bjthq-algproject-cpd-appsearch005.vivo.lan:/root]
# top -H -p 32289
top - 20:23:10 up 320 days,  8:00,  4 users,  load average: 7.44, 6.05, 6.21
Tasks: 325 total,   5 running, 320 sleeping,   0 stopped,   0 zombie
Cpu(s): 13.3%us,  0.0%sy,  0.0%ni, 86.7%id,  0.0%wa,  0.0%hi,  0.0%si,  0.0%st
Mem:  65842452k total, 46458808k used, 19383644k free,   474072k buffers
Swap: 16776188k total,    55940k used, 16720248k free, 14153080k cached

  PID USER      PR  NI  VIRT  RES  SHR S %CPU %MEM    TIME+  COMMAND
32289 root      20   0 51.1g  28g  15m R 129.9 44.9  10:38.93 java
32290 root      20   0 51.1g  28g  15m R 129.9 44.9   9:46.26 java
32291 root      20   0 51.1g  28g  15m R 129.9 44.9  10:25.39 java
32292 root      20   0 51.1g  28g  15m R 129.9 44.9  10:21.27 java
32293 root      20   0 51.1g  28g  15m R 129.9 44.9  10:05.72 java
32256 root      20   0 51.1g  28g  15m S  0.0 44.9   0:00.00 java
32263 root      20   0 51.1g  28g  15m S  0.0 44.9   0:00.61 java
32275 root      20   0 51.1g  28g  15m S  0.0 44.9   0:44.43 java
32276 root      20   0 51.1g  28g  15m S  0.0 44.9   0:44.42 java
32277 root      20   0 51.1g  28g  15m S  0.0 44.9   0:44.43 java
32278 root      20   0 51.1g  28g  15m S  0.0 44.9   0:44.41 java
32279 root      20   0 51.1g  28g  15m S  0.0 44.9   0:44.41 java
32280 root      20   0 51.1g  28g  15m S  0.0 44.9   0:07.94 java
32281 root      20   0 51.1g  28g  15m S  0.0 44.9   0:54.43 java
32282 root      20   0 51.1g  28g  15m S  0.0 44.9   0:54.49 java
32283 root      20   0 51.1g  28g  15m S  0.0 44.9   0:56.90 java
32284 root      20   0 51.1g  28g  15m S  0.0 44.9   1:05.33 java
32285 root      20   0 51.1g  28g  15m S  0.0 44.9   1:15.22 java
32288 root      20   0 51.1g  28g  15m S  0.0 44.9   0:10.25 java
32294 root      20   0 51.1g  28g  15m S  0.0 44.9   0:03.38 java
32295 root      20   0 51.1g  28g  15m S  0.0 44.9   0:00.21 java
32296 root      20   0 51.1g  28g  15m S  0.0 44.9   0:00.23 java
32297 root      20   0 51.1g  28g  15m S  0.0 44.9   0:00.00 java
32298 root      20   0 51.1g  28g  15m S  0.0 44.9   0:00.00 java
32299 root      20   0 51.1g  28g  15m S  0.0 44.9   0:10.70 java
32300 root      20   0 51.1g  28g  15m S  0.0 44.9   0:08.77 java
32301 root      20   0 51.1g  28g  15m S  0.0 44.9   0:08.98 java
32302 root      20   0 51.1g  28g  15m S  0.0 44.9   0:12.11 java
32303 root      20   0 51.1g  28g  15m S  0.0 44.9   0:10.80 java
32304 root      20   0 51.1g  28g  15m S  0.0 44.9   0:10.11 java
32305 root      20   0 51.1g  28g  15m S  0.0 44.9   0:08.64 java
32306 root      20   0 51.1g  28g  15m S  0.0 44.9   0:08.14 java
32307 root      20   0 51.1g  28g  15m S  0.0 44.9   0:08.79 java
32308 root      20   0 51.1g  28g  15m S  0.0 44.9   0:10.62 java
32309 root      20   0 51.1g  28g  15m S  0.0 44.9   0:02.45 java
32310 root      20   0 51.1g  28g  15m S  0.0 44.9   0:02.49 java
32311 root      20   0 51.1g  28g  15m S  0.0 44.9   0:02.47 java
32312 root      20   0 51.1g  28g  15m S  0.0 44.9   0:02.50 java
32313 root      20   0 51.1g  28g  15m S  0.0 44.9   0:02.44 java
32314 root      20   0 51.1g  28g  15m S  0.0 44.9   0:00.00 java
32315 root      20   0 51.1g  28g  15m S  0.0 44.9   0:05.02 java
```

#### 2、63机器（时延快一些的机器）

JVM 启动参数：

```sh
/home/app/predictor-serving/jdk/jre/bin/java -Djava.util.logging.config.file=/home/app/predictor-serving/tomcat/conf/logging.properties -Djava.util.logging.manager=org.apache.juli.ClassLoaderLogManager -server -Xmx16g -Xms16g -XX:+UseG1GC -XX:MaxGCPauseMillis=100 -XX:+HeapDumpOnOutOfMemoryError -Xss512k -Dapp.name=predictor-serving -Dapp.env=prd -Dconfig.version=54 -Dconfig.host=vivocfg-agent.prd.bj01.vivo.lan:8080/vivocfg -Djdk.tls.ephemeralDHKeySize=2048 -Djava.protocol.handler.pkgs=org.apache.catalina.webresources -classpath /home/app/predictor-serving/tomcat/bin/bootstrap.jar:/home/app/predictor-serving/tomcat/bin/tomcat-juli.jar -Dcatalina.base=/home/app/predictor-serving/tomcat -Dcatalina.home=/home/app/predictor-serving/tomcat -Djava.io.tmpdir=/home/app/predictor-serving/tomcat/temp org.apache.catalina.startup.Bootstrap start
```

GC 情况：

```
[root@10.21.8.63:/root]
#jstat -gcutil 23355 1000 20
  S0     S1     E      O      M     CCS    YGC     YGCT    FGC    FGCT     GCT
  0.00 100.00   9.39  39.19  96.26  94.69     67    3.686     0    0.000    3.686
  0.00 100.00  53.37  39.19  96.26  94.69     67    3.686     0    0.000    3.686
  0.00 100.00  86.81  39.19  96.26  94.69     67    3.686     0    0.000    3.686
  0.00 100.00  30.64  39.22  96.26  94.69     68    3.698     0    0.000    3.698
  0.00 100.00  64.78  39.22  96.26  94.69     68    3.698     0    0.000    3.698
  0.00 100.00   8.15  39.20  96.26  94.70     69    3.710     0    0.000    3.710
  0.00 100.00  41.89  39.20  96.26  94.70     69    3.710     0    0.000    3.710
  0.00 100.00  74.63  39.20  96.26  94.70     69    3.710     0    0.000    3.710
  0.00 100.00   9.00  39.22  96.26  94.71     70    3.727     0    0.000    3.727
  0.00 100.00  42.98  39.22  96.26  94.71     70    3.727     0    0.000    3.727
  0.00 100.00  75.80  39.22  96.26  94.71     70    3.727     0    0.000    3.727
  0.00 100.00  13.58  39.22  96.28  94.72     71    3.740     0    0.000    3.740
  0.00 100.00  32.89  39.22  96.28  94.72     71    3.740     0    0.000    3.740
  0.00 100.00  69.43  39.22  96.28  94.72     71    3.740     0    0.000    3.740
  0.00 100.00   5.20  39.21  96.28  94.72     72    3.752     0    0.000    3.752
  0.00 100.00  35.14  39.21  96.28  94.72     72    3.752     0    0.000    3.752
  0.00 100.00  61.29  39.21  96.28  94.72     72    3.752     0    0.000    3.752
  0.00 100.00  86.97  39.21  96.28  94.72     72    3.752     0    0.000    3.752
  0.00 100.00  21.18  39.23  96.28  94.72     73    3.768     0    0.000    3.768
  0.00 100.00  50.35  39.23  96.28  94.72     73    3.768     0    0.000    3.768
```

机器负载也差不多。

### 分析

可以看到两者其实唯一的区别就是 JVM 启动参数略有不同，64机器的启动参数多配置了几个参数，并且堆大小是32G:

```
-Xmx32g -Xms32g -XX:+UseG1GC -XX:MaxGCPauseMillis=200 -XX:InitiatingHeapOccupancyPercent=8 -XX:ParallelGCThreads=5 -XX:ConcGCThreads=5
```

推测应该是JVM 配置导致 GC 有差异，但是从 gcutil 看来两者都很正常。把64的 JVM 参数改成跟63一样，两者时延就一样了。突然想起来 JVM 有个压缩指针的概念，32G 是个临界点，会不会是这个原因呢？把64机器的堆大小改成30G，果然也是一样的时延。用jcmd查看 当前进程的 JVM参数就知道两者运行时的 JVM 参数差异了：

32G堆大小的时候 JVM 运行时参数：

```sh
jcmd 21293 VM.flags
21293:
-XX:AutoBoxCacheMax=20000 -XX:CICompilerCount=15 -XX:ConcGCThreads=6 -XX:G1HeapRegionSize=16777216 -XX:+HeapDumpOnOutOfMemoryError -XX:InitialHeapSize=34359738368 -XX:MarkStackSize=4194304 -XX:MaxGCPauseMillis=200 -XX:MaxHeapSize=34359738368 -XX:MaxNewSize=20602421248 -XX:MinHeapDeltaBytes=16777216 -XX:+PrintGC -XX:+PrintGCApplicationStoppedTime -XX:+PrintGCDateStamps -XX:+PrintGCDetails -XX:+PrintGCTimeStamps -XX:ThreadStackSize=512 -XX:+UseFastUnorderedTimeStamps -XX:+UseG1GC
```

改成30G 堆大小之后的 JVM 运行时参数：

```sh
jcmd 17011 VM.flags
17011:
-XX:AutoBoxCacheMax=20000 -XX:CICompilerCount=15 -XX:ConcGCThreads=6 -XX:G1HeapRegionSize=8388608 -XX:+HeapDumpOnOutOfMemoryError -XX:InitialHeapSize=32212254720 -XX:MarkStackSize=4194304 -XX:MaxGCPauseMillis=200 -XX:MaxHeapSize=32212254720 -XX:MaxNewSize=19327352832 -XX:MinHeapDeltaBytes=8388608 -XX:+PrintGC -XX:+PrintGCApplicationStoppedTime -XX:+PrintGCDateStamps -XX:+PrintGCDetails -XX:+PrintGCTimeStamps -XX:ThreadStackSize=512 -XX:+UseCompressedClassPointers -XX:+UseCompressedOops -XX:+UseFastUnorderedTimeStamps -XX:+UseG1GC
```

可以看到 30G 堆大小的 JVM 参数多了这两个：`-XX:+UseCompressedClassPointers -XX:+UseCompressedOops`，这正式启动压缩指针的参数。


### 为什么超级大堆反而慢呢？

这是因为 JVM 存储对象的格式其实是取决于你通过 `-Xmx` 指定的堆的大小。当最大堆内存大小小于 32G 的时候（所谓的窄指针模式`narrow pointer mode`），普通对象实例有一个12字节的头部，数组有一个16字节 的头部，并且每个对象指针只占用4个字节。而当`-Xmx`大于 32G（所谓的宽指针模式`wide pointer mode`），普通对象实例将会有一个16个字节的头部，数组有一个20字节的头部，而且每个对象指针将占用8个字节。这些是因为JVM 需要更大的对象地址空间来支持超级大堆。

但是带来最直接的后果就是对象存储空间相对变大，特别在你的应用程序大量使用了指针密集型的数据结构（如 trees, hash maps, 等），或者有很多小对象，你的内存空间有可能会比小堆增加了1.5倍。具体可以参见 [Why 35GB Heap is Less Than 32GB](https://blog.codecentric.de/en/2014/02/35gb-heap-less-32gb-java-jvm-memory-oddities/)。

另一个不为人知的副作用就是性能相对会变慢（大概20%左右），因为使用较大指针在主内存和缓存之间移动数据，占用较大宽带；同时会降低CPU缓存命中率，因为
64位对象引用增大了，CPU能缓存的oop将会更少，从而降低了CPU缓存的效率。最后64位对象引用需要占用更多的堆空间，留给其他数据的空间将会减少，从而GC也会承受较大压力（不过实际观察这个影响不大，基本可以忽略）。


**TIPS** 32位的指针如何寻址32G的内存空间？

压缩后的指针是32bit，理论上上使用指针压缩的最大堆应该就是4G。但是实际上并非如此，由于对象是8字节对齐的，因此对象起始地址最低三位总是0，因此可以存储时可以右移3bit，高位空出来的3bit可以表示更高的数值，所以实际上可以使用指针压缩的 maxHeapSize 是 `4G * 8 = 32G`。


### 一些有用的工具

#### 1、使用 `-XX:+PrintFlagsFinal` 打印 JVM 默认参数值

当你在网上兴冲冲找到一个可优化的参数时，先用 `-XX: +PrintFlagsFinal` 看看，它可能已经默认打开了。

JDK7与JDK8，甚至JDK7中的不同小版本，有些参数值都不一样，所以不要轻信网上任何文章，一切以生产环境同版本的JDK打出来的为准。

经常以类似下面的语句去查看参数，偷懒不起应用，用 -version 代替。有些参数设置后会影响其他参数，所以也要带上。

查看所有默认的参数：

```sh
bin/java -XX:+UnlockDiagnosticVMOptions -XX:+PrintFlagsFinal -version
```

查看当使用 `UseConcMarkSweepGC` 的时候，ParallelGCThreads 默认值时多少:

```sh
java -Xmx1024m -Xms1024m -XX:+UseConcMarkSweepGC -XX:+PrintFlagsFinal -version | grep ParallelGCThreads
```

对于不同版本里的默认值，建议是顺势而为，JDK在那个版本默认打开不打开总有它的理由。安全第一，没有很好的因由，不要随便因为网上某篇文章的推荐(包括你现在在读的这篇)就去设置。


#### 2、jcmd: 查看 当前进程的 JVM参数

jcmd 是 jdk7 之后新增的工具, 它是 `java flight recorder` 的唯一启动方式, 详细的内容请见 [java flight recorder 的使用](https://zshell.cc/2017/06/25/jvm-tools--jcmd_jvm%E7%AE%A1%E7%90%86%E7%9A%84%E5%8F%A6%E7%B1%BB%E5%B7%A5%E5%85%B7/); 不过, oracle 顺手又为其附带了一些 “便捷” 小工具:

* 列举 jvm 进程 (对标 jps)
* dump 栈信息 (对标 jstack)
* dump 堆信息 (对标 jmap -dump)
* 统计类信息 (对标 jmap -histo)
* 获取系统信息 (对标 jinfo)

这样一下子就有意思了, jcmd 似乎有了想要取代其他命令的野心。 不过 jcmd 并没有完全实现其他命令的功能，而且 jcmd 的选项名字一般都比较长, 不容易记住。所以建议除了 java flight recorder 必须要使用 jcmd 之外，其余的功能暂时还是建议使用传统的工具来解决问题。这里我们完全可以用 `jinfo -flags {vmid}` 来查看当前进程的 JVM参数。


推荐阅读
-------

1. [Java HotSpot™ Virtual Machine Performance Enhancements](https://docs.oracle.com/javase/8/docs/technotes/guides/vm/performance-enhancements-7.html)
2. [JAVA堆大小不要超过32GB!!!](http://view.inews.qq.com/a/20180906B0QBHH00)
3. [Trick behind JVM's compressed Oops](https://stackoverflow.com/questions/25120546/trick-behind-jvms-compressed-oops)
4. [JVM Anatomy Quark #23: Compressed References](https://shipilev.net/jvm/anatomy-quarks/23-compressed-references/)
5. [Why 35GB Heap is Less Than 32GB – Java JVM Memory Oddities](https://blog.codecentric.de/en/2014/02/35gb-heap-less-32gb-java-jvm-memory-oddities/)
6. [jstack 命令使用经验总结](https://zshell.cc/2017/09/24/jvm-tools--jstack%E5%91%BD%E4%BB%A4%E4%BD%BF%E7%94%A8%E7%BB%8F%E9%AA%8C%E6%80%BB%E7%BB%93/)
7. [HotSpot: What Are Those GC Worker Threads in G1 GC?](http://xmlandmore.blogspot.com/2014/06/oracle-has-published-some-tuning-guides.html)
8. [JVM实用参数（三）打印所有XX参数及值](https://yq.aliyun.com/articles/20380)
9. [Inspecting HotSpot JVM Options](https://q-redux.blogspot.com/search/label/PrintFlagsFinal)
10. [jcmd: jvm 管理的另类工具](https://zshell.cc/2017/06/25/jvm-tools--jcmd_jvm%E7%AE%A1%E7%90%86%E7%9A%84%E5%8F%A6%E7%B1%BB%E5%B7%A5%E5%85%B7/)

