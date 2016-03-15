---
title: MySQL高可用性方案
layout: post
---


一些基本概念和背景知识
-------------------


### Switchover, Failover和Failback

当一个系统宕掉之后，手动的切换到冗余或者备份系统，这个过程称之为Switchover。如果是自动切换，不需要人工干预，那么称之为failover。

> When a manual process is used to switch from one system	to a redundant or standby system in case of	a failure, we are talking about switchover.	Automatic switchover, without human	intervention, is called failover.

原系统恢复之后，从备份系统切换回去的过程称之为Failback。


### Asynchronous VS Synchronous

为了构建一个冗余的数据库系统，数据需要移动到另一个服务器/存储。在数据传输方面有两个基本概念：asynchronous和synchronous。

Asynchronous data replication: server A将数据发送给server B，但是并不等待server B的确认回复('acknowledge	message')，就直接提交事务。

优点

* 独立
* 性能

缺点

* 同步丢失带来的数据一致性问题
* 同步失败但是无法回滚带来的一致性问题。

Synchronous	data replication: 与异步同步不同，第二个server在接收到同步数据后，需要发送一个'acknowledge	message'给主节点，主节点收到ACK确认后才提交事务。

带来最大的缺点是性能不仅仅取决于两个节点的执行性能，还极大的依赖于节点之间的网络传输时延。所以不推荐跨机房或者走公网搭建synchronous数据同步。

**TIPS**

MySQL支持三种级别的同步方式：

1. Asynchronous replication: 5.5以前的版本
2. [Semi-synchronous Replication](http://dev.mysql.com/doc/refman/5.7/en/replication-semisync.html) (MySQL 5.6+)
3. Synchronous replication (MySQL Cluster NDB)


MySQL 5.5+对同步复制功能的改进以及对基于Replication方式的HA方案的影响
--------------------------------------------------------------

### Multi-threaded Slave (MTS)

MySQL 5.6允许你基于数据库级别进行并行复制。这个特性称之为“Multi-Threaded Slave” (MTS)。

通过设置 `slave_parallel_workers=N`, where N > 1(推荐配置是16) 就可以了开启这个特性。 

MySQL 5.7取消了5.6只能一个线程一个数据库的并行复制的限制(one thread per schema)，引入了logical-clock概念，一个组提交内事务都可以并行，可以达到接近主库并发效果。


开通并行复制, 需要指定如下几个变量:

	slave-parallel-type=LOGICAL_CLOCK // 基于组提交的并行复制方式，默认是DATABASE，则为基于库的并行复制方式(即mysql 5.6的并行复制机制)
	slave-parallel-workers=16  // 性能测试结果显示16个worker线程性能最好。如果配置为0，则退化为mysql 5.5的单线程复制机制 
	master_info_repository=TABLE  // master info存储在table里以最小化磁盘IO
	relay_log_info_repository=TABLE  // relay info也存储在table里以最小化磁盘IO
	relay_log_recovery=ON  // 与 relay_log_info_repository=TABLE 一起开启`Crash-safe Slave`特性。

需要注意的是新的并行复制机制依赖于GTID，如果变量gtid_mode=OFF, 则会默认启用Anonymous_Gtid模式 (所以，还是乖乖打开GTID为好, 又不会死人)。

这个特性可以极大程度上减少甚至消除master和slave之间的因为单一SQL Thread引起的同步延迟。

### Semi-Synchronous Replication

MySQL 5.5以后引入了半同步复制 [Semi-Synchronous replication](http://dev.mysql.com/doc/refman/5.5/en/replication-semisync.html)，在很大程度上减少了"binlog events exist only on the crashed master"的风险，避免数据丢失。

不过需要注意的是，Semi-Synchronous replication不没有解决所有的一致性问题。Semi-Synchronous replication只是保证了至少一个（不是所有）slave接收到master在commit时发送过来的binlog事件。其他的slaves仍然有可能没有收到所有的binlog事件。 Without applying differential relay log events from the latest slave to non-latest slaves, slaves can not be consistent each other.

MySQL 5.7版本增强半同步复制，确保从库先收到，并且允许设置半同步从库个数。称之为`Lossless Semi-Synchronous Replication`。同时性能方面也有了一定的改善。

**解决问题**

主要是解决了slave的同步延迟问题，使得一致性得到进一步的保障。在这之前，要不是使用复杂的切换脚本从Master将bin-log补齐，要不就是简单的采用Shared storage/DRBD的方式。


### Global Transaction Identifier (GTID)


全局事务ID(GTID)是MySQL5.6引入的一个非常实用的特性。GTID is a unique identifier created and associated with each transaction committed on the server of origin (master). This identifier is unique not only to the server on which it originated, but is unique across all servers in a given replication setup. There is a one-to-one mapping between all transactions and all GTIDs。这个新特性使故障转移和slave提升变得容易很多。

需要注意的是MySQL和MariaDB具有不同的GTID实现。

**解决问题**

主要是解决了slave切换master的复杂性。在GTID之前，我们需要根据slave当前的执行SQL到新的master中找到bin-log的同步位置(`CHANGE MASTER TO ... MASTER_LOG_FILE=xxx, MASTER_LOG_POS=xxx`)。这个是一个麻烦同时又容易出错的过程。使用GTID就不需要了。你只需要将`MASTER_AUTO_POSITION`选项enable。

很多脚本会判断MySQL版本是否支持GTID，根据这个来执行master切换。


### 多源复制 (Multi-Source Replication)

这个也是一个非常大的改进。在5.7之前，一个Slave只能专属于某个master，一个master下面可以有多个slaves，master和slaves是一对多的关系。但是在5.7之后，replication slave 可以同时接收来自多个master的复制。

合久必分，分久必合。有分库分表的场景就有合并的需要。Multi-source replication特别适用于后台报表或者管理查询系统。这些系统往往不是根据shardingId来查询的，把数据聚合一起可以简化查询流程和提供查询性能。当然，这也有一个问题就是同步时延；还有就是数据量要能够存放在单库单表中。

需要注意的是Multi-source replication并不对可能带来的冲突进行检测或者解决，留给应用自己处理。

### Crash-safe Slave

Crash safe means even if a slave mysqld/OS crash, you can recover the slave and continue replication without restoring MySQL databases onto the slave. To make crash safe slave work, you have to use InnoDB storage engine only, and in 5.6 you need to set relay_log_info_repository=TABLE and relay_log_recovery=1.

Durability (sync_binlog = 1 and innodb_flush_log_at_trx_commit = 1) is NOT required.


**总结**

总之，MySQL5.6以后对复制功能做了很大的完善，极大的提高了Master和Slave之间的同步延迟问题。

* MTS(Mutl-threaded Slave)解决了单一SQL Thread apply同步事件慢的问题
* Semi-Synchronous Replication解决了Master和Slave之间的同步延迟问题
* GTID使得故障转移和slave提升变得容易很多
* Crash-safe Slave 让slave恢复变得快速容易很多
* Multi-Source Replication可以在MySQL基本实现数据汇聚，是分库分表的一个反向支持


MySQL高可用性的解决方案
---------------------

现在我们开始进入我们的专题，MySQL的HA方案讨论。

MySQL的HA方案可以在几个层面上做到:

1. 基于MySQL replication的拓扑方式
	* MASTER-SLAVE replication：需要Slave Promotion
	* MASTER-MASTER replication：使用热备(hot standby)让master切换变得容易一些
	* MASTER-RELAY replication
	* ... 
	* 使用一些工具实现failover
		* [MMM](http://mysql-mmm.org/doku.php)。HA: 99%
		* [MHA](https://code.google.com/p/mysql-master-ha/wiki/Overview)。HA: 99.9%
		* [PRM](https://github.com/percona/percona-pacemaker-agents/blob/master/doc/PRM-setup-guide.rst) 貌似只支持到 MySQL 5.6，不支持最新的MySQL 5.7。
		* [MySQL Fabric](http://dev.mysql.com/doc/mysql-utilities/1.5/en/fabric.html)
		* [Orchestrator](https://www.percona.com/blog/2016/03/08/orchestrator-mysql-replication-topology-manager/)
		* [MariaDB Replication Manager (MRM)](https://github.com/mariadb-corporation/replication-manager)
		* 需要配合使用VIP或者数据库中间件（如PaceMaker, Cobar/MyCat, etc.）对应用保持透明
		* 需要注意这些工具对MySQL版本和配置的要求
			* MySQL or MariaDB，比如MRM只支持MariaDB；MHA、PRM等不支持 MariaDB GTID based replication topologies, MySQL Fabric要求MySQL 5.6.x+, etc.
			* Replication methods (statement and row based replication)
			* supporting GTID? 
			* etc.
2. 基于集群的部署方式 (true Multimaster Cluster based on (virtually) synchronous replication)。HA: 99.999%
	* 官方：[Mysql Cluster](http://dev.mysql.com/doc/refman/5.6/en/mysql-cluster.html)
	* 开源：[Galera Cluster for MySQL](http://galeracluster.com/products/)
		* [Percona XtraDB Cluster(PXC, uses Galera cluster)](https://www.percona.com/doc/percona-xtradb-cluster/5.6/index.html)
		* [MariaDB Galera Cluster](https://mariadb.com/kb/en/mariadb/galera-cluster/)
		* Codership
	* 有一些局限性和对业务并不是很透明，另外因为是同步复制，所以性能上会有一些影响。每个节点都需要并发的跟其他节点通讯，对网络要求比较高。
	* 需要配合使用Load Balancer (MaxScale, HAProxy和VIP) 对应用透明和负载均衡。[MySQL Load Balancing with HAProxy - Tutorial](http://severalnines.com/tutorials/mysql-load-balancing-haproxy-tutorial)
3. 存储方式上的同步
	* Synchronous Replication using DRBD(Distributed Replicated Block Device)。HA: 99.9%
	* Shared Storage (SAN, etc.)。HA: 99.5% ~ 99.9%。存在存储单点问题，不推荐。

### 1. 基于MySQL replication的拓扑方式

首先看一下基于MySQL同步的HA方案。这是最老也是最经得起时间考验的方案。


#### MySQL复制拓扑结构

MySQL支持多重拓扑结构，不同的拓扑结构有不同的应用场景。

首先MySQL Replication中有三种服务器角色：

* Master
* Slave: 扩展读操作的Slave
* Relay: 同时扮演master角色的slave称之为中继服务器(Relay)

这些角色常常构成下面几种常见的拓扑结构：

1. Master-Slaves
	* 一个master下面挂多个只读slaves，最常见的拓扑结构
	* 读写分离的典型架构
	* master挂掉之后，需要进行比较复杂的slave选举和切换，使用MHA可以自动化这个过程。
2. Master-Relay-Slaves (Chain Replication)
	* 减少Master同步压力
	* Relay作为master备份，切换之后slaves不需要重新切换到新的master。
	* 使用半同步可以避免同步延迟放大。
3. Master with Active Master (Circular Replication)
	* 所有的Master对等接收读写请求
	* 需要设置合理的auto-increment offset来避免自增主键冲突，或者使用全局的主键分配器
	* 推荐只有master提供服务，其他master作为热备（类似于下面的Master with Backup Master）
4. Master with Backup Master (Multiple Replication)
	* 单master，一个热备master，一个或者多个slave
	* master和backup master之间使用Semi-synchronous replication 减少同步延迟
	* 这个拓扑结构的好处在于不需要slave promotion过程，直接把backup master顶上去就可以了，而且半同步保证backup master基本是up-to-date的。
5. Multiple Masters to Single Slave (Multi-Source Replication)
6. Galera with Replication Slave (Hybrid Replication)
	* Hybrid replication is a combination of MySQL asynchronous replication and virtually synchronous replication provided by Galera. 
	* Galera Cluster提供正常的读写请求，本身通过 近实时同步 保证HA
	* 异步复制的slave可以作为报表/OLAP等类型的查询数据库，或者mysqldump，同时也是一个容灾备份。


#### 基于MySQL复制方式的故障failover方案

分为三种情况：

1. Master故障
2. Slave故障
3. Relay故障

##### 1. Master故障failover

如果master发生故障，必须迅速替换它以保证整个系统的正常运行。这时候要做的第一件事情就是找一个新的可用的master，将所有的客户端连接到新的master上。同时，slaves虽然还可以提供读服务，但是在没有切换到新的master之前，它们只能提供过时的数据，所以还需要把slave指向新的master。

具体切换步骤如下：

1. (中间件或者监控程序)监控到master挂掉了
2. 找出一个可用的master替代
3. 将客户端连接到新的master
4. 将其他的slave挂到新的master下

具体分析一下：

* 步骤一：(中间件或者监控程序)监控到master挂掉了
	* 我们需要有一种方式能够监控到Master挂掉了，这需要一个很好的监控系统，至少针对master。
	* MONyog Ultimate Monitor会是一个不错的选择。
* 步骤二: 找出一个可用的master替代
	* 我们需要选举一个最新的slave出来，同时在promote它为新的master之前，要保证relaylog中的所有事务已经被SQL Thread apply。
	* 如果拓扑结构一开始就已经有backup master，那么会比slave选举简单很多，直接把backup master推举上去就可以了。
	* 严格意义来说，在切换master之前需要检查一下新master是up to date的，如果不是，要补齐未apply的事务先。使用半同步可以避免这个问题。
* 步骤三: 将客户端连接到新的master
	* 我们希望对客户端透明，同时也能够最大的减少不可用时间。
	* 这里一般是使用VIP的方式，或者引入数据库中间件来达到自动切换的目的。
* 步骤四: 将其他的slave挂到新的master下。这里一般有两个步骤需要操作：
	* 把promoted的slave设置为可写。// 如果是dual-master，那么可以省略这个步骤
	* 其它的slaves切换到新的master进行同步。这里需要重新找到在新master的同步位置 // 如果slaves是挂在backup master或者relay server下，则不需要这个步骤；另外，使用GTID可以大大简化这个步骤

前面也提到过，有很多开源的工具可以将这个过程自动化，即由switchover进化为failover。另外，不同的MySQL版本，不同的拓扑结构，不同的工具（主要是master失败检测方式）有不同的成本和效果。需要深入了解和实际测试再做决定(TODO)。一般来说，failover的时间大概在5s～20s。


**TIPS & Recommendations**

* 尽量使用GTID-based Replication，简化部署和failover。
* 尽量使用InnoDB存储引擎，因为它提供了完整的ACID事务支持和快速奔溃恢复（full transaction capability with ACID compliance and better crash recovery）
* 尽量不要使用环形复制拓扑结构，不要有多个对等的master同时提供读写服务。因为容易带来数据冲突。
* 推荐使用Master with Backup Master (Multiple Replication)拓扑结构，同时强烈建议master和backup master之间使用semi-synchronous replication。
* backup master和slaves都配置成read-only模式，这样可以减少数据冲突的风险。
* 同步数据包一般都会比较大， 所以记得讲`max_allowed_packet`设置成一个比较大的数值，避免同步错误。
* 复制链接参数不应该配置在my.cnf中，而是通过命令动态指定。

##### 2. Slave故障failover

由于Slave只是提供读扩展和备份，所以当slave故障之后，要做的事情很简单：检测到slave故障，把它从负载均衡器的池子里删除，恢复后再加回去就可以了。

##### 3. Relay故障failover

对于中继服务器的故障，需要特殊处理。如果它们发生了故障，剩下的slave必须重定向到其他的中继服务器或者master。这里需要考虑一下master的负载问题。


### 2. 基于集群的部署方式 (true Multimaster Cluster based on (virtually) synchronous replication)


#### 1. [Mysql Cluster](http://dev.mysql.com/doc/refman/5.6/en/mysql-cluster.html)

MySQL Cluster本质上是在提交的时候通过两阶段提交协议将数据同步复制到多个节点（至少两个数据副本），来实现HA。这点是跟Replication最大的不同，后者是异步，或者半同步。 

MySQL Cluster包含以下三种类型的节点: 

![MySQL Cluster Components](http://dev.mysql.com/doc/refman/5.7/en/images/cluster-components-1.png)

* SQL Node: SQL节点就是传统的MySQL server，包括NDB/Cluster存储引擎。SQL节点处理所有的SQL请求，解析，优化和查询缓存。应用一般是与SQL节点通讯。	
* Data Node: 数据节点组成data node group，它的作用主要是存储数据，同时还负责：
	* 事务协调
	* 记录锁 (record level locking)
	* Failover
	* Fail-back
	* 实现两阶段提交协议	
* Management Node: 管理节点主要负责MySQL集群配置和管理，同时是split-brain的仲裁者。

优点

* 高可用性：官方数据是99.999%。
* 高可扩展性：自动sharding，透明分片
* 数据一致性：同步复制+两阶段提交保证数据一致性。

[缺点](http://dev.mysql.com/doc/refman/5.7/en/mysql-cluster-limitations.html)

* 只能使用NDB存储引擎。如果使用InnoDB存储引擎，那么无法利用MySQL Cluster的优势。
	* 主要体现在事务处理 [Limits Relating to Transaction Handling in MySQL Cluster](http://dev.mysql.com/doc/refman/5.7/en/mysql-cluster-limitations-transactions.html)
		* 事务隔离级别: NDB只支持READ COMMITTED隔离级别。[Differences Between the NDB and InnoDB Storage Engines](http://dev.mysql.com/doc/refman/5.7/en/mysql-cluster-ndb-innodb-engines.html)
		* 回滚：不支持partial transactions, and no partial rollbacks of transactions. A duplicate key or similar error causes the entire transaction to be rolled back.
		* 不支持大事务: 
		* etc.
* 从MySQL升级到MySQL集群，对业务并非完全透明，可能需要修改schema和Sql queries。
	* [Noncompliance with SQL Syntax in MySQL Cluster](http://dev.mysql.com/doc/refman/5.7/en/mysql-cluster-limitations-syntax.html)。
	* [Limits Associated with Database Objects in MySQL Cluster](http://dev.mysql.com/doc/refman/5.7/en/mysql-cluster-limitations-database-objects.html)
	* 对Join，子查询，范围操作性能很差。
* 不支持分布式锁，可能会引起脑裂问题(split brain issues)。
* 对BLOB和TEXT字段的支持不友好
* [Unsupported or Missing Features in MySQL Cluster](http://dev.mysql.com/doc/refman/5.7/en/mysql-cluster-limitations-unsupported.html)
	* Index prefixes
	* Savepoints and rollbacks
	* Durability of commits
	* Replication
		* 不支持Statement-Based复制方式，支持ROW或者MIXED。
		* 不支持GTID
* 存储限制[Limitations Relating to MySQL Cluster Disk Data Storage](http://dev.mysql.com/doc/refman/5.7/en/mysql-cluster-limitations-disk-data.html)
* 网络通讯严重，分区容忍性差，容易脑裂。
* 短板效应，集群的吞吐率取决于最差的节点。

#### 业界使用情况 TODO
 
#### 2. [Percona XtraDB Cluster(PXC, uses Galera cluster)](https://www.percona.com/doc/percona-xtradb-cluster/5.6/index.html)

![Percona-xtradb-cluster](https://www.percona.com/doc/percona-xtradb-cluster/5.6/_images/cluster-diagram1.png)

与MySQL Cluster实现上的不同 [About Percona XtraDB Cluster](https://www.percona.com/doc/percona-xtradb-cluster/5.6/intro.html)

* 集群中每一个节点都是普通的MySQL/Percona Server，这意味着你可以把现存的MySQL/Percona Server直接接入集群，或者从集群中拎出来作为普通的MySQL服务。
* 每个节点拥有所有的数据，这意味着节点的数据完全对等，任何查询都可以在本地完成。同时也意味着数据的高度冗余，对写扩展(write scalling)并不是很友好。


[优点](https://www.percona.com/doc/percona-xtradb-cluster/5.6/index.html)

* 同步复制，高度一致性
* 多master复制，任何节点都可以读写
* 并发复制 (Parallel applying events on slave. Real “parallel replication”)
* Automatic node provisioning.
* 数据一致性


[缺点](http://www.percona.com/doc/percona-xtradb-cluster/limitation.html)

* 只支持InnoDB存储引擎(包括XtraDB引擎)的复制
* 不支持LOCK/UNLOCK TABLES, GET_LOCK(), RELESE_LOCK()等 锁 语句或者函数
* 短板效应，集群的吞吐率取决于最差的节点。
* ...	

总体来说，Percona XtraDB Cluster比MySQL Cluster NDB要简单易用一些，归根于他对等节点的设计，同时使用了XTraDB作为存储引擎。所以相对来说对应用要透明很多。


#### 3. [MariaDB Galera Cluster](https://mariadb.com/kb/en/mariadb/galera-cluster/)

跟Percona XtranDB Cluster一样，MariaDB Galera Cluster也是一个同步多master MariaDB集群 (synchronous multi-master cluster for MariaDB），同样只支持XtraDB/InnoDB存储引擎。


[优点](https://mariadb.com/kb/en/mariadb/what-is-mariadb-galera-cluster/)

* 同步复制，高度一致性
* 多master复制，任何节点都可以读写(Active-active multi-master topology)
* 并发复制 (True parallel replication, on row level)
* Automatic membership control, failed nodes drop from the cluster; Automatic node joining
* 数据一致性
* Direct client connections, native MariaDB look & fee

[缺点](https://mariadb.com/kb/en/mariadb/mariadb-galera-cluster-known-limitations/)

* 只支持InnoDB存储引擎(包括XtraDB引擎)的复制
* 不支持LOCK/UNLOCK TABLES, GET_LOCK(), RELESE_LOCK()等 锁 语句或者函数
* 短板效应，集群的吞吐率取决于最差的节点。
* ...

对比PXC和MGC，可以发现两者基本都是一样的。


MySQL集群管理平台
----------------

现在已经有一些商业的或者开源的MySQL集群管理平台可以简化整个MySQL 部署，管理，监控，和扩容操作。

1. [ClusterControl](http://severalnines.com/product/clustercontrol)：商业
	* MySQL based cluster：Galera Cluster for MySQL, Percona XtraDB Cluster, MariaDB Galera Cluster, MySQL Cluster and MySQL Replication
	* MongoDB clusters：MongoDB Sharded Cluster, MongoDB Replica Set and TokuMX
2. [Percona Toolkit for MySQL](https://www.percona.com/software/mysql-tools/percona-toolkit)
	* a collection of advanced open source command-line tools
	* supports Percona Server, MySQL and MariaDB and works best with Percona Server and other Percona products.


业界HA方案
---------



参考文章和推荐阅读
----------------

1. [MySQL Replication failover: Maxscale vs MHA (part 1)](http://severalnines.com/blog/mysql-replication-failover-maxscale-vs-mha-part-1)
2. [MySQL Replication failover: Maxscale vs MHA (part 2)](http://severalnines.com/blog/mysql-replication-failover-maxscale-vs-mha-part-2)
3. [MySQL Replication failover: Maxscale vs MHA (part 3)](http://severalnines.com/blog/mysql-replication-failover-maxscale-vs-mha-part-3)
4. [Database High Availability (DB HA)](https://cwiki.apache.org/confluence/pages/viewpage.action?pageId=34838207)
5. [High-availability options for MySQL, October 2013 update](https://www.percona.com/blog/2013/10/23/high-availability-options-for-mysql-october-2013-update/)
6. [mysql-master-ha](https://code.google.com/p/mysql-master-ha/wiki/Other_HA_Solutions)
7. [High Availability Solutions	for	the	MariaDB	and MySQL® Database](https://mariadb.com/sites/default/files/High_Availability_Solutions_for_the_MariaDB_and_MySQL_Database_-_MariaDB_White_Paper.pdf)
8. [MySQL Replication for High Availability - Tutorial](http://severalnines.com/tutorials/mysql-replication-high-availability-tutorial) 关于MySQL同步最新最详细的介绍，强烈推荐。
9. [High Availability Database Tools](https://dev.acquia.com/blog/high-availability-database-tools)
10. [MySQL Server Using InnoDB Compared with MySQL Cluster](http://dev.mysql.com/doc/refman/5.7/en/mysql-cluster-compared.html)

