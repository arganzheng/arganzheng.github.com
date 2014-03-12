---
title: ZooKeeper简介
layout: post
---

### Zookeeper的简介

ZooKeeper是一个构建在Paxos算法上的高可用的分布式数据管理与系统协调框架，提供了一系列原语集，更上层的应用可以用它来实现同步，配置管理，名称服务，Master选举，分布式锁，分布式队列等。

zookeeper提供如下服务保证

* 顺序一致性: client的updates请求都会根据它发出的顺序被顺序的处理
* 原子性: 一个update操作要么成功要么失败，没有其他可能的结果
* 一致性的镜像: client不论连接到哪个server，展示给它都是同一个视图
* 可靠性: 一旦一个update被应用就被持久化了，除非另一个update请求更新了当前值
* 及时性: 客户端所看到的系统在一个时间范围内是最新的

### zookeeper的设计目标

#### 1. 简单

ZooKeeper允许分布式进程之间通过一个共享的层级命名空间来相互协作，这个命名空间以类似于文件系统的方式组织起来。命名空间中的数据单元被称为znode，它相当于文件或者目录。与典型文件系统不同的是，文件系统用于持久化存储，而ZooKeeper的数据是保持在内存中的，这意味着ZooKeeper能达到很高的吞吐量和低延迟。

#### 2. 高性能、高可用性，和严格的顺序访问

高性能使得ZooKeeper能用于大规模分布式系统，高可用性能避免单点故障，严格的顺序化意味着复杂的同步操作可以在客户端实现。

#### 3. 集群化

ZooKeeper本身也是有集群化的。如下图：

![ZooKeeper Cluster](/media/images/zookeeper-cluster.jpg)

客户端与单个服务端相连，它维持一个TCP连接，在其上发送请求，获得响应，获得监控事件和发送心跳检测。如果到服务端的TCP连接断了，客户端会连接另一个服务端。组成ZooKeeper服务的每个服务端都知道其它服务端的存在，它们维护一个服务端状态的内存镜像，连同事务日志和快照保存在持久化存储中，只要大部分服务端可用，ZooKeeper服务就可用。

#### 4. 顺序化

ZooKeeper为每个更新操作都标记一个号码以反映事务的顺序。后来的操作使用这个顺序来实现高度的抽象，比如同步原语。

#### 5. 快速

这个特性在以读操作为主的工作中尤为明显。ZooKeeper应用程序运行于数以千计的机器中，当读操作与写操作的比例为10:1时，ZooKeeper能获得最佳性能。


### Zookeeper的架构


所有的server组成一个从Zookeeper的服务，server和server之间的交互是通过协议进行通信，选取出一个leader，负责向follower广播所有变化消息。其实所有的follower都和leader通信，follower接受来自leader的所有变化信息，保存在自己的内存。follower转发来自客户端的写请求给leader。客户端的读请求会在follower端直接服务，无须转发给leader。

### Zookeeper Atomic Broadcast(ZAB)协议

首先2011年Yahoo公开了Zab协议论文。第二选举了leader，只有leader才能提出决议。这就意味着其实跟消息发送的次序是非常有关系的，所以内部的通信基本上都是通过TPS来做的。第三没有abort的两段式提交。第四是基于状态增量的消息传输。第五是高可用、高性能。

### ZooKeeper数据模型

ZooKeeper的数据模型极像一个标准的文件系统，一个名字是一个以/分隔的路径元素的序列，ZooKeeper命名空间的每个节点通过路径来标识。。

[ZooKeeper Data Model](/media/images/zk-data-model.jpg)


Zookeeper的视图结构类似标准的Unix文件系统，但是没有引入文件系统相关概念：目录和文件，而是使用了自己特有的节点(node)概念，称为znode。Znode是ZooKeeper中数据的最小单元，每个znode上都可以保存数据，同时还可以挂载子节点，也构成了一个层次化的命名空间，我们称之为树。

### Znode节点类型

ZooKeeper的每个节点是有生命周期的，这取决于节点的类型。在ZooKeeper中，节点类型可以分为持久节点（PERSISTENT ）、临时节点（EPHEMERAL），以及时序节点（SEQUENTIAL ），具体在节点创建过程中，一般是组合使用，可以生成以下4种节点类型：

1、持久节点（PERSISTENT）

所谓持久节点，是指在节点创建后，就一直存在，直到有删除操作来主动清除这个节点——不会因为创建该节点的客户端会话失效而消失。

2、持久顺序节点（PERSISTENT_SEQUENTIAL ）

这类节点的基本特性和上面的节点类型是一致的。额外的特性是，在ZK中，每个父节点会为他的第一级子节点维护一份时序，会记录每个子节点创建的先后顺序。基于这个特性，在创建子节点的时候，可以设置这个属性，那么在创建节点过程中，ZK会自动为给定节点名加上一个数字后缀，作为新的节点名。这个数字后缀的上限是整型的最大值。

3、临时节点（EPHEMERAL ）

和持久节点不同的是，临时节点的生命周期和客户端会话绑定。也就是说，如果客户端会话失效，那么这个节点就会自动被清除掉。注意，这里提到的是会话失效，而非连接断开。另外，在临时节点下面不能创建子节点。

4、临时顺序节点（EPHEMERAL_SEQUENTIAL）

### 节点信息

	[zk: localhost:2181(CONNECTED) 4] get /YINSHI.MONITOR.ALIVE.CHECK
	?t 10.232.102.191:21811353595654255
	cZxid = 0x300000002
	ctime = Thu Dec 08 23:29:53 CST 2011
	mZxid = 0xe00008bbf
	mtime = Thu Jul 28 07:17:34 CST 2012
	pZxid = 0x300000002
	cversion = 0
	dataVersion = 2164293
	aclVersion = 0
	ephemeralOwner = 0x0
	dataLength = 39
	numChildren = 0

上面这个信息，是在ZK命令行的一个输出信息，从这个输出内容中可以清楚的看到，ZK的一个节点包含了哪些信息。其中比较重要的信息包括节点的数据内容，节点创建/修改的事务ID，节点/修改创建时间，当前的数据版本号，数据内容长度，子节点个数等。

ZooKeeper被设计用来存储管理服务的数据：状态信息、配置、位置信息等，所以每个节点存储的数据通常很小，几字节到几K字节不等。默认上限是1M。

zonde维持一个stat结构，它包含数据变化的版本号、ACL变化和时间戳，以允许cache校验和协调化的更新。每当znode的数据变化时，版本号将增加。一个客户端收到数据时，它也会收到数据的版本号。

保存在每个znode中的数据都是自动读写的。读操作获取znode的所有数据，写操作替换掉znode的所有数据。每个节点有一个访问控制表（ACL）来限制谁能做哪些操作。

**TIPS** 节点大小

配置文件中的jute.maxbuffer属性表示节点可存的数据大小，default=0xfffff=1MB，如果超出该值则会引起ZooKeeper系统不稳定，甚至崩溃，直接重启无效，所以禁止创建数据超过1MB的节点；鉴于超过10k的节点数据对性能的较大影响，建议数据大小尽可能控制在10k之内。


### ZooKeeper基本API

ZooKeeper的一个设计目标是易于编程。所以，它只支持如下操作：

* create 在命名空间的某个位置创建一个节点
* delete 删除一个节点
* exists 测试某位置的节点是否存在
* get data 从一个节点读取数据
* set data 往一个节点写数据
* get children 获取一个节点的子节点列表
* sync 等待数据被传播以同步数据


### Zookeeper Session

客户端和server间采用长连接。连接建立后，server产生session ID返还给客户端。客户端定期发送pin包来检查和保持和server的连接。一旦session结束或超时，所有ephemeral而节点会被删除。客户端可根据情况设置合适的session超时时间。

### Zookeeper  Watches

Watches是客户端安装在server的事件侦听方法；当侦听的变化发生时，server发消息给客户端进行通知。客户端使用单线程对所有事件按顺序同步回调。每次触发后会被自动删除。如果需要再次侦听事件，必须重新安装Watches。无法保证跟踪到每一个变化，避免安装大量的Watches侦听在同一个节点。Watches的创建和触发规则：所有的API是怎么创建一个Watches，写请求的API，会触发一些事件。举一个例子：当我们调用Watches，我们用一个create，谁创建这个Watches这个节点就会被触发掉。

### ZooKeeper Observer

不参与选举，永远是follower。Observer数量增加，进一步提升集群的服务能力。不会增加重新选举leader的开销。很好地支持跨数据中心、本地读、异地写的功能。

### zookeeper实现

角色

* 领导者(Leader): 领导者不接受client的请求，负责进行投票的发起和决议，最终更新状态。
* 跟随者(Follower): Follower用于接收客户请求并返回客户结果。参与Leader发起的投票。
* 观察者(Observer): Oberserver可以接收客户端连接，将写请求转发给leader节点。但是Observer不参加投票过程，只是同步leader的状态。Observer为系统扩展提供了一种方法。
* 学习者(Learner): 和Leader进行状态同步的server统称Learner，上述Follower和Observer都是Learner。


下图从较高层次说明了ZooKeeper的构成

![zk-service](/media/images/zk-service.jpg)

集群数据库是存在于内存中的数据库，保存命名空间的所有数据。更新操作会被记录到硬盘中以便恢复，写操作先被序列化到硬盘中，再应用到内存数据库中。通常Zookeeper由2n+1台servers组成，只要有n+1台（大多数）server可用，整个系统保持可用。

对于follower收到的client的请求，对于读操作，由follower的本地内存数据库直接给client返回结果；对于会改变系统状态的写操作，则交由Leader进行提议投票，超过半数通过后返回结果给client：

![zk-proposal](/media/images/zk-proposal.jpg)

Zookeeper的核心是原子广播，这个机制保证了各个server之间的同步。实现这个机制的协议叫做Zab协议。Zab协议有两种模式，它们分别是恢复模式和广播模式。当服务启动或者在领导者崩溃后，Zab就进入了恢复模式，当leader被选举出来，且大多数server的完成了和leader的状态同步以后，恢复模式就结束了。状态同步保证了leader和server具有相同的系统状态。

广播模式需要保证proposal被按顺序处理，因此zk采用了递增的事务id号(zxid)来保证。所有的提议(proposal)都在被提出的时候加上了zxid。实现中zxid是一个64为的数字，它高32位是epoch用来标识leader关系是否改变，每次一个leader被选出来，它都会有一个新的epoch。低32位是个递增计数。 当leader崩溃或者leader失去大多数的follower，这时候zk进入恢复模式，恢复模式需要重新选举出一个新的leader，让所有的server都恢复到一个正确的状态。


#### 选举和同步过程

zk的实现中用了基于paxos算法（主要是fastpaxos）的实现。具体如下：

1. 每个Server启动以后都询问其它的Server它要投票给谁。
2. 对于其他server的询问，server每次根据自己的状态都回复自己推荐的leader的id和上一次处理事务的zxid（系统启动时每个server都会推荐自己）
3. 收到所有Server回复以后，就计算出zxid最大的哪个Server，并将这个Server相关信息设置成下一次要投票的Server。
4. 计算这过程中获得票数最多的的sever为获胜者，如果获胜者的票数超过半数，则改server被选为leader。否则，继续这个过程，直到leader被选举出来。

此外恢复模式下，如果是重新刚从崩溃状态恢复的或者刚启动的的server还会从磁盘快照中恢复数据和会话信息（zk会记录事务日志并定期进行快照，方便在恢复时进行状态恢复）。 选完leader以后，zookeeper就进入状态同步过程。

状态同步过程如下：

1. Leader就会开始等待server连接
2. Follower连接Leader，将最大的zxid发送给Leader
3. Leader根据Follower的zxid确定同步点
4. 完成同步后通知Follower已经成为uptodate状态
5. Follower收到uptodate消息后，又可以重新接受client的请求进行服务了


### ZooKeeper性能数据

#### 1. 创建销毁读取性能

创建节点性能测试

![创建节点性能测试](/media/images/zk-znode-create-performance.jpg)


删除节点性能测试

![删除节点性能测试](/media/images/zk-znode-delete-performance.jpg)


读取节点内容性能测试

![读取节点内容性能测试](/media/images/zk-znode-get-performance.jpg)
  

#### 2. 综合读写性能曲线

![综合读写性能曲线](/media/images/zk-performance-test.jpg)


### ZooKeeper的常见应用场景


#### 1. 命名服务(Naming Service)

命名服务也是分布式系统中比较常见的一类场景。在分布式系统中，通过使用命名服务，客户端应用能够根据指定名字来获取资源或服务的地址，提供者等信息。被命名的实体通常可以是集群中的机器，提供的服务地址，远程对象等等——这些我们都可以统称他们为名字（Name）。其中较为常见的就是一些分布式服务框架中的服务地址列表。通过调用ZK提供的创建节点的API，能够很容易创建一个全局唯一的znode，所有应用从中读取，避免写死。

#### 2. 分布式通知/协调

ZooKeeper中特有watcher注册与异步通知机制，能够很好的实现分布式环境下不同系统之间的通知与协调，实现对数据变更的实时处理。使用方法通常是不同系统都对ZK上同一个znode进行注册，监听znode的变化（包括znode本身内容及子节点的），其中一个系统update了znode，那么另一个系统能够收到通知，并作出相应处理

#### 3. 分布式锁

这个主要得益于ZooKeeper为我们保证了数据的强一致性。锁服务可以分为两类，一个是保持独占，另一个是控制时序。

* 所谓保持独占，就是所有试图来获取这个锁的客户端，最终只有一个可以成功获得这把锁。通常的做法是把zk上的一个znode看作是一把锁，通过create znode的方式来实现。所有客户端都去创建 /distribute_lock 节点，最终成功创建的那个客户端也即拥有了这把锁。
* 控制时序，就是所有视图来获取这个锁的客户端，最终都是会被安排执行，只是有个全局时序了。做法和上面基本类似，只是这里 /distribute_lock 已经预先存在，客户端在它下面创建临时有序节点（这个可以通过节点的属性控制：CreateMode.EPHEMERAL_SEQUENTIAL来指定）。Zk的父节点（/distribute_lock）维持一份sequence,保证子节点创建的时序性，从而也形成了每个客户端的全局时序。

#### 4. 分布式队列

简单地讲有两种，一种是常规的先进先出队列，另一种是要等到队列成员聚齐之后的才统一按序执行。对于第一种先进先出队列，和分布式锁服务中的控制时序场景基本原理一致，这里不再赘述。

第二种队列其实是在FIFO队列的基础上作了一个增强。通常可以在 /queue 这个znode下预先建立一个/queue/num 节点，并且赋值为n（或者直接给/queue赋值n），表示队列大小，之后每次有队列成员加入后，就判断下是否已经到达队列大小，决定是否可以开始执行了。这种用法的典型场景是，分布式环境中，一个大任务Task A，需要在很多子任务完成（或条件就绪）情况下才能进行。这个时候，凡是其中一个子任务完成（就绪），那么就去 /taskList 下建立自己的临时时序节点（CreateMode.EPHEMERAL_SEQUENTIAL），当 /taskList 发现自己下面的子节点满足指定个数，就可以进行下一步按序进行处理了。


参考资料以及推荐阅读
-----------------

1. [ZooKeeper数据模型](http://nileader.blog.51cto.com/1381108/946788)
2. [李欣：ZooKeeper在携程的使用及前景](http://v.csdn.hudong.com/open/view/detail/83-SDCC2012-ctrip-ZooKeeper)
3. [ZooKeeper典型应用场景一览](http://jm-blog.aliapp.com/?p=1232)
