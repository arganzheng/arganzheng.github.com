---
title: Java虚拟机学习笔记
layout: post
---

Java内存区域
-----------

![JVM的体系结构](/media/images/jvm.png)


1. 程序计数器
	* 对于一个运行中的Java程序而言，其中的每一个线程都有它自己的PC寄存器，它是该线程启动时创建的。
	* PC寄存器大小是一个字长，既能够持有一个本地指针，也能够持有returnAddress。
	* 当线程执行某个Java方法时，PC寄存器总是指向下一条将被执行指令的“地址”。
	* 如果一个线程正在执行一个本地方法，那么此时PC寄存器的值是“undefined”。
2. Java虚拟机栈：线程私有
	* 每当启动一个新线程时，Java虚拟机都会为它分配一个Java栈。
	* Java栈是以帧为单位保存线程的运行状态。
	* 虚拟机只会直接对Java栈执行两种操作：以帧为单位的压栈或出栈。
	* 当前帧、当前类、当前常量池。
	* Java栈上的所有数据都是此线程私有的。
	* Java栈和帧在内存中也不必是连续的。
	* 栈帧：
		* 局部变量区: 一个以字长为单位，从0开始计数的数组，通过索引来使用其中的数据。
		* 操作数栈: 一个以字长为单位的数组，通过标准的栈操作来访问的。
		* 帧数据区: 支持常量池解析、正常方法返回、异常派发机制。
3. 本地方法栈：Native方法栈
4. Java堆
5. 方法区
	* HotSpot称之为永久区
	* 由于所有线程都共享方法区，因此对方法区数据的访问必须被设计为是线程安全的。
	* 包括 运行时常量池，string.intern()会动态添加常量进去。
	* 方法区可以运行时调整，也可以被垃圾收集。


**说明** 

1. 直接内存，Direct Memory，并不是虚拟机运行时数据区的一部分，也不是Java虚拟机规范中定义的内存区域。 在Java1.4中新加入的NIO(New Input/Output)类，引入了一种基于通道(Channel)与缓冲区(Buffer)的I/O方式，它可以使用Native函数库直接分配堆外内存，然后通过一个存储在Java堆里面的DirectByteBuffer对象作为这块内存的引用进行操作。这样能在一些场景中显著提高性能，因为避免了在Java堆和Native堆中来回复制数据。

2. Java中所有对象的存储空间都是在堆中分配的，但是这个对象的引用却是在堆栈中分配,也就是说在建立一个对象时从两个地方都分配内存，在堆中分配的内存实际建立这个对象，而在堆栈中分配的内存只是一个指向这个堆对象的引用而已。


OOM异常与分析
------------

### 1. Java堆溢出

* -Xms，-Xmx
* MAT分析core dump文件。

### 2. 虚拟机栈和本地方法栈溢出

* -Xss
* StackOverflowError

### 3. 运行时常量池溢出

* -XX:PermSize, -XX:MaxPermSize
* String.intern()

### 4. 方法区溢出

* CGLIB、JSP(JSP需要编译成java类文件)

### 5. 本机直接内存溢出

* -XX:MaxDirectMemorySize，如果不指定，默认与Java堆的最大值（-Xmx）一样。
* 取决于引用NativeMemory的java引用的引用类型(Stone, Soft, Weak)，在清理引用的时候，才会顺便GC本机直接内存的废弃对象。所以经常发生本机直接内存溢出时，堆中还有许多空闲内存。


垃圾收集器与内存分配策略
---------------------

GC需要完成的三件事情：

* 哪些内存需要回收（识别垃圾）
* 什么时候回收
* 如何回收

### 哪些内存需要回收

1) 引用计数算法

优点：实现简单，判定效率高
缺点：无法解决对象之间的相互循环引用问题

2) 根搜索算法

通过一系列的名为“GC Roots”的对象作为起始点，从这些节点开始向下搜索，搜索走过的路径称之为引用链(Reference Chain)，当一个对象到GC Roots没有任何引用链相连（用图论的话来说就是从GC Roots到这个对象不可达）时，则证明这个对象是不可用的。

在Java语言里，可作为GC Roots的对象包括下面几种：

* 虚拟机栈（栈帧中的本地变量表）中应用的对象。
* 方法区中的类静态属性应用的对象
* 方法区中的常量引用的对象
* 本地方法区中JNI（即一般说的Native方法）的引用的对象。


### 引用

* 强引用：只要引用还存在，垃圾收集器永远不会回收被强引用的对象，宁愿抛出OOM异常中止程序运行。
* 软引用（SoftReference）：描述还有用，但并非必需的对象。系统将要OOM之前，将会把这些对象列入回收范围并进行第二次回收。软引用可以用来实现内存敏感的高速缓存。
* 弱引用（WeakReference）：也是描述非必需的对象，但是它的强度与软引用更弱一些，被弱引用的对象只能生存到下一次GC发生之前。当GC工作时，无论内存是否足够，都会回收WeakReference引用的对象。
* 虚引用：也称为幽灵引用或者幻影引用（PhantomReference），它是最弱的一种引用关系。虚引用并不会对对象的生存时间构成影响，也无法通过虚引用来取得一个对象实例。为一个对象设置虚引用关联的唯一目的就是通过能在这个对象被GC回收时收到一个系统通知。

**TIPS** WeakReference是在做Full GC时会回收，但SoftReference只在OOME前回收内存。如果不知道这个很容易出问题，具体参考 [Font.create()引发OOME问题](https://code.google.com/p/hatter-source-code/wiki/Blog_FontCreateCauseOOME)。


### 对象的三种状态

1. 可触及状态：能够从GC Roots到达
2. 可复活状态：不能从GC Roots达到对象，这个状态回收期准备回收，称之为“死缓”阶段。在GC之前会调用对象的finalize()方法，这些对象的finalize()方法可能可以把对象重新扭转到可触及状态。
3. 不可触及状态：当虚拟机执行完对象的finalize()方法后，对象没有转入可触及状态，就进入不可触及状态。这时候就是真正的“死刑”犯了。


### Java启动参数设置

-server -Xmx2g -Xms2g -Xmn256m 
-XX:PermSize=128m -Xss256k
-XX:+DisableExplicitGC
-XX:+CMSParallelRemarkEnabled
-XX:+UseCMSCompactAtFullCollection
-XX:LargePageSizeInBytes=128m
-XX:+UseFastAccessorMethods
-XX:+UseCMSInitiatingOccupancyOnly
-XX:CMSInitiatingOccupancyFraction=70

### 回收方法区

1. 废弃常量
2. 无用的类
	* 该类所有的实例都已经被回收，也就是说Java堆中不存在该类的任何实例
	* 加载该类的ClassLoader已经被回收
	* 该类对应的java.lang.Class对象没有在任何地方被引用，无法在任何地方通过反射访问该类的方法。


### 垃圾收集算法

**基本概念**

* 分代收集（Generational Collecting）:基于对对象生命周期分析后得出的垃圾回收算法。把对象分为年青代、年老代、持久代，对不同生命周期的对象使用不同的算法进行回收。
* 串行收集:串行收集使用单线程处理所有垃圾回收工作，因为无需多线程交互，实现容易，而且效率比较高。但是，其局限性也比较明显，即无法使用多处理器的优势。
* 并行收集:并行收集使用多线程处理垃圾回收工作，因而速度快，效率高。而且理论上CPU数目越多，越能体现出并行收集器的优势。
* 并发收集:可以保证（垃圾回收的）大部分工作都并发进行（应用不停止），垃圾回收只暂停很少的时间。

### JVM的GC

#### JVM的分代内存布局

![JVM的分代内存布局](/media/images/jvm-gc.png)

#### 新生代的GC

![新生代的GC](/media/images/gc-young.png)

#### 老年代的GC

![老年代的GC](/media/images/gc-old.png)

#### 各种GC行为比较

![各种GC行为比较](/media/images/gc-comparison.png)


#### GC算法的选择

![GC算法的选择](/media/images/gc-selection.png)


**More about CMS**

设计目的：获取最短回收停顿时间为目标的垃圾收集器

四个阶段

1. 初始标记(CMS initial mark, stop the world)
2. 并发标记(CMS concurrent mark)
3. 重新标记(CMS reamark, stop the world)
4. 并发清除(CMS concurrent sweep)

优点（Concurrent Low Pause Collector）

* 并发收集
* 低停顿

缺点

* 对CPU资源敏感。与用户线程并发执行，占用了一部分CPU资源，总吞吐量降低。
    * 默认启动的回收线程数是(CPU数量+3)/4
* 无法处理浮动垃圾，可能出现`Concurrent Mode Failure`失败而导致另一次Full GC的发生。
    * Floating Garbage 在并发清理阶段产生的垃圾
    * -XX:CMSInitiatingOccupancyFraction 控制CMS预留空间的大小，默认是68%。
    * Concurrent Mode Failure 是CMS预留的内存空间被浮动垃圾占据而不满足程序需求导致的错误
    * 这时候会临时采用Serial Old收集器来重新进行老年代的GC
* 标记-清除算法实现的GC都会产生大量的空间碎片。
    * -XX:+UseCMSCompactAtFullCollection 控制CMS GC之后再进行一次碎片整理过程。
    * -XX:+CMSFullGCsBeforeCompaction 设置执行了多少次不压缩的Full GC之后，跟着来一次带压缩的。

### JVM性能和故障诊断工具

#### jps

jps用来查看host上运行的所有java进程的pid（jvmid），一般情况下使用这个工具的目的只是为了找出运行的jvm进程ID，即lvmid，然后可以进一步使用其它的工具来监控和分析JVM。

常用的几个参数：

* -l  输出java应用程序的main class的完整包
* -q  仅显示pid，不显示其它任何相关信息
* -m  输出传递给main方法的参数
* -v  输出传递给JVM的参数。在诊断JVM相关问题的时候，这个参数可以查看JVM相关参数的设置

 
#### jstat

Jstat（ “Java Virtual Machine statistics monitoring tool” ）是JDK自带的一个轻量级小工具。主要对Java应用程序的资源和性能进行实时的命令行的监控，包括了对Heap size和垃圾回收状况的监控。

语法结构如下：jstat [Options] vmid [interval] [count]

* Options -- 选项，我们一般使用 -gcutil 查看gc情况
* vmid    -- VM的进程号，即当前运行的java进程号
* interval-- 间隔时间，单位为毫秒
* count   -- 打印次数，如果缺省则打印无数次

#### jmap

jmap 是一个可以输出所有内存中对象的工具，甚至可以将VM 中的heap，以二进制输出成文本。

使用方法

* jmap -histo pid>a.log  可以将其保存到文本中去，在一段时间后，使用文本对比工具，可以对比出GC回收了哪些对象。
* jmap -dump:format=b,file=f1 PID  可以将该PID进程的内存heap输出出来到f1文件里。 

#### JConsole 和 VisualVM

通过RMI通信方式，将目标jvm通过jmx或者jstatd暴露出来的数据进行图形化展示。配置与使用本来很简单，但是由于线上防火墙以及RMI动态通信端口原因，处理起来非常麻烦。具体可以参考笔者的另一篇文章：[tomcat监控](http://blog.arganzheng.me/posts/tomcat-monitor.html)。

#### BTrace

具体参见笔者的另一篇文章：[BTrace实战](http://blog.arganzheng.me/posts/btrace-in-action.html)


类加载器(ClassLoader)
--------------------

虚拟机设计团队把类加载阶段中的“通过一个类的全限定名来获取描述此类的二进制字节流”这个动作放到Java虚拟机外部去实现，以便让应用程序自己决定如何去获取所需要的类。实现这个动作的代码模块被称为“类加载器”。

### 类的相等性

比较两个类是否“相等”，只有在这两个类是由同一个类加载器加载的前提之下才有意义，否者，即使这两个类是来源于同一个Class文件，只要加载它们的类加载器不同，那么这两个类就必定不相等。

这里的“相等”，包括代表类的Class对象的equals()方法，isAssignableFrom()方法，isInstance()方法的返回结果，也包括了使用instanceof关键字做对象所属关系判定等情况。


### 各种ClassLoader介绍


站在JVM的角度讲

1. 启动加载器(`Bootstrap ClassLoader`)：C++实现，是JVM的一部分。
2. 所有其他的类加载器：Java实现，独立于JVM外部，并且全部继承自抽象类`java.lang.ClassLoader`。

从Java开发人员的角度来看：

1. 启动加载类(`Bootstrap ClassLoader`)：C++实现，是JVM的一部分。负责加载`$JAVAHOME/lib`目录下的类。或者被`-Xbootclasspath`参数所指定的路径中的，并且是JVM所识别的（仅安装文件名称识别，如rt.jar，名称不符合的类库即使放在lib目录也不会被加载）类库加载到JVM内存中。启动加载类无法被Java程序直接引用。
2. 扩展类加载器(`Extension ClassLoader`)：这个ClassLoader由`sun.misc.Launcher$ExtClassLoader`实现，它负责加载`$JAVAHOME/lib/ext`目录中的，或者被`java.ext.dirs`系统变量所指定的路径中的所有类库，开发者可以直接使用扩展类加载器。
3. 应用程序类加载器(`Application ClassLoader`)：这个类加载器由`sun.misc.Launcher$AppClassLoader`来实现。由于这个类加载器是`ClassLoader.getSystemClassLoader()`方法的返回值，所以一般也称之为系统类加载器(`SystemClassLoader`)。它负责加载用户类路径(通过`$classpath`环境变量指定)上所指定的类库，开发者可以直接使用这个类加载器，如果应用程序中没有自定义过自己的类加载器，一般情况下这个就是程序中默认的类加载器。

![各种类加载器](/media/images/classloader2.png)

我们的应用程序都是由这三种类加载器相互配合进行加载的，如果有必要，还可以加入自己定义的类加载器。这些类加载器之间的关系一般会如下图所示：

![类加载器双亲委派模型](/media/images/classloader.png)

即所谓的双亲委派模型(Parents Delegation Model)。注意，这里虽有看起来像是继承关系，其实是通过组合关系来复用父类加载器的代码。
    
    protected synchronized Class<?> loadClass(String name, boolean resolve) throws ClassNotFoundException{
    
    	// 首先，检查请求的类是否已经被加载过了
    	Class c = findLoadedClass(name);
    	if(c == null){
    		try{
    			if(parent != null){
    				c = parent.loadClass(name, false);
    			}else{ // 顶级类加载器，即BootstrapClassLoader
    				c = findBootstrapClassOrNull(name);
    			}
    		}catch(ClassNotFoundException e){
    			// 如果父类加载器抛出ClassNotFoundException
    			// 则说明父类加载器无法完成加载请求
    		}
    		if(c == null){
    			// 在父类加载器无法加载的时候
    			// 再调用本身的findClass方法来进行类加载
    			c = findClass(name);
    		}
    	}
    	if(resolve){
    		resolveClass(c);
    	}
    	return c;
    }


其实就是一个模版方法，要遵循双亲委派模型，用户自定义类加载器不应该覆盖loadClass方法，而应该把自己的类加载逻辑写到`findClass()`方法中，在`loadClass()`方法的逻辑里如果父类加载失败，则会调用自己的`findClass()`方法来完成加载，这样就可以保证新写出来的类加载器是符合双亲委派规则的。

双亲委派模型的设计目的是为了解决各个类加载器的基础类的统一问题（越基础的类由越上层的加载器进行加载），基础类之所以被称为“基础”，是因为它们总是作为被用户代码调用的API。但是世事往往没有绝对的完美，如果基础类又要调用回用户的代码，那该怎么办呢？一个典型的例子就是JNDI服务。

为了解决这个问题，Java设计团队只好引入了一个不太优雅的设计：线程上下文类加载器(`Thread Context ClassLoader`)。这个类加载器可以通过`java.lang.Thread`类的`setContextClassLoader()`方法进行设置，如果创建线程时还未设置，它将会从父线程中继承一个；如果在应用程序的全局范围内都没有设置过，那么这个类加载器默认就是应用程序类加载器。

有了线程上下文类加载器，就可以实现让父类加载器请求字类加载器去完成类加载的动作，这种行为实际上就是打通了双亲委派模型的层次结构来逆向使用类加载器，已经违背了双亲委派模型的一般性原则，但是这也是无可奈何的事情。Java中所有涉及SPI的加载动作基本上都采用这种方式，例如JNDI、JDBC、JCE、JAXB和JBI等。


### 获取ClassLoader的途径

1. 获取当前类的ClassLoader: clazz.getClassLoader();
2. 获取当前线程上下文的ClassLoader: Thread.currentThread().getContextClassLoader();
3. 获取系统的ClassLoader: ClassLoader.getSytemClassLoader();
4. 获得调用者的ClassLoader: DriverManager.getCallerClassLoader();


参考资料以及推荐阅读
-----------------

1. 深入理解Java虚拟机-周志明著
2. [Introduction to Java's Architecture](http://www.artima.com/insidejvm/ed2/introarchP.html)
3. [The Architecture of the Java Virtual Machine](http://www.artima.com/insidejvm/ed2/jvm2.html)
4. [HotSpot中OutOfMemoryError解析](https://code.google.com/p/hatter-source-code/wiki/Study_Java_HotSpot_OOME)
5. [HotSpot问题诊断](https://code.google.com/p/hatter-source-code/wiki/Blog_HotSpotDiagnosis)
6. [Java Heap OOM问题](http://blog.arganzheng.me/posts/java-head-space-oom.html)
7. [Font.create()引发OOME问题](https://code.google.com/p/hatter-source-code/wiki/Blog_FontCreateCauseOOME)
8. [聊聊我对Java内存模型的理解](https://code.google.com/p/hatter-source-code/wiki/Blog_JavaMemoryModel)



