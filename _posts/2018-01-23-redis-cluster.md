---
title: Redis集群学习
layout: post
catalog: true
tags: [redis, cluster, 高可用, 分布式, 缓存]
category: [技术]
---

Redis3.0版本加入了cluster功能，解决了Redis单点无法横向扩展的问题。

分布式系统要解决的不外乎以下两个问题：

1. sharding/partition 以提高系统的吞吐率
2. replication 以提供系统的高可用

然后因为sharding了，所以就有一个 哈希或者路由 的事情要处理。引入的问题包括：

1. 数据分布方式：取模, 范围映射，一致性哈希, etc.
2. 客户端哈希还是服务端转发路由，如果需要客户端或者任意一个节点都能接受请求进行转发，那么就需要引入集群的概念，让集群中的节点能够互相知道数据的分布情况（相关技术参见下面讨论）
3. 对于没有sharding key的请求(如果系统设计支持)，需要有个scatter和gather的过程
4. 节点的加入和退出导致的数据重分配(rebalance)

这又需要引入一些集群相关的技术：

1. 集群架构：有明确而固定的master-slave架构，还是去中心化的对称架构
2. lease 集群发现
3. Gossip 节点发现和集群各节点之间的信息同步

而引入replication固然解决了每个sharding的HA问题，并且如果是强一致性主从同步的话还可以提供读服务，减轻主副本的读压力。但是同样会引入一些问题：

1. 主从副本如何同步: Quorum W+R>N [with vector clock], Merkle tree [with anti-entropy]：
2. 事务和一致性问题: vector clock, Quorum W+R>N [with vector clock], MVCC
3. 主从选举和切换问题: Paxos, Raft, etc.

当然，由于CAP原理，所以在实现方案上会有所取舍。下面我们就Redis 3.0集群方案分别讨论。


1. sharding
-----------

### Redis 集群的数据分片

Redis 集群没有使用一致性hash，而是引入了 哈希槽(hash slot) 的概念。

Redis 集群有16384个哈希槽，每个key通过 CRC16 校验后对 16384 取模来决定放置哪个槽。集群的每个节点负责一部分hash槽。

数据分布算法为：

```
slotId = crc16(key) % 16384
```

客户端根据slotId决定将请求路由到哪个redis节点。

举个例子,比如当前集群有3个节点，那么：

* 节点 A 包含 0 到 5500号哈希槽
* 节点 B 包含5501 到 11000 号哈希槽
* 节点 C 包含11001 到 16384号哈希槽

这种结构很容易添加或者删除节点。比如如果我想新添加个节点D，我需要从节点 A，B，C中迁移部分哈希槽到D上。 同样如果我想移除节点A，那么我只需要将A中的哈希槽迁移到B和C节点上，然后将没有任何槽的A节点从集群中移除即可。

由于从一个节点将哈希槽移动到另一个节点并不需要停止服务，所以无论添加删除或者改变某个节点的哈希槽的数量都不会造成集群不可用的状态。


redis默认采用key作为hash key，但是为了让用户对数据分布有一定的控制权，Redis引入了HashTag的概念，使得数据分布算法可以根据key的某一部分进行计算，让相关的的多条记录可以落在同一个数据分片。例如：

* 某条商品交易记录的key为：product_trade_{prod123}
* 这个商品的详情记录的key为：product_detail_{prod123}

Redis会根据`{}`之间的字符串作为数据分布算法的输入进行hash。


另外，Redis对于需要跨slot的请求，并不支持。所以这里没有协调节点scatter和gather的过程。

### 客户端的路由(Redirection and Resharding)

在确定了数据如何分布到Redis的不同实例之后，实际数据访问时，根据请求中涉及的key，用对应的数据分布算法得出数据分布在哪个实例，再将请求路由至该实例，这个过程叫做请求路由。实现上，有如下三种实现方式：

1. 客户端路由
2. 代理层路由
3. 协调节点路由

Redis主要是 客户端路由 + 协调节点重定向 两种方式实现数据的路由请求。

#### MOVED Redirection

一个 Redis 客户端可以向集群中的任意节点（包括从节点）发送命令请求。 节点会对命令请求进行分析， 如果该命令是集群可以执行的命令， 那么节点会查找这个命令所要处理的键所在的槽。如果要查找的哈希槽正好就由接收到命令的节点负责处理， 那么节点就直接执行这个命令；如果所查找的槽不是由该节点处理的话，节点将查看自身内部所保存的哈希槽到节点ID的映射记录， 并向客户端回复一个 MOVED 错误，告知其正确的路由信息。client根据moved响应更新其内部的路由缓存信息，以便下一次请求时能够直接路由到正确的节点，降低交互次数。下面是一个例子：

```
1. Client => A: GET foo
2. A => Client: -MOVED 8 192.168.5.21:6391
3. Client => B: GET foo
4. B => Client: "bar"
```

`-MOVED 8 192.168.5.21:6391`类似于HTTP协议中的301永久跳转，表示8号槽位不在我这里，位于`192.168.5.21:6391`。客户端需要根据这个 IP 和端口号，向所属的节点重新发送一次 GET 命令请求；虽然不是必须的，但是客户端应当 记录/更新 `hashslot->node` 映射表，这样下次就可以直接访问到正确的节点了。当集群处于稳定状态时，所有客户端最终都会保存有一个哈希槽至节点的映射记录（map of hash slots to nodes），使得集群非常高效：客户端可以直接向正确的节点发送命令请求，无须重定向。

#### ASK redirection

MOVED 重定向相当于HTTP协议中的301跳转，代表永久性转移(Permanently Moved)。而ASK重定向则是302跳转，代表暂时性转移(Temporarily Moved)。

当我们使用 MOVED 的时候，意味着我们认为哈希槽永久地被另一个不同的节点处理，并且希望接下来的所有查询都尝试发到这个指定的节点上去。而 ASK 意味着我们只要下一个查询发送到指定节点上去。

从客户端看来，ASK 重定向的完整语义如下：

* 如果接受到 ASK 重定向，那么把查询的对象调整为指定的节点。
* 先发送 ASKING 命令，再开始发送查询。
* 现在不要更新本地客户端的映射表把哈希槽 8 映射到节点 B。

一旦完成了哈希槽 8 的转移，节点 A 会发送一个 MOVED 消息，客户端也许会永久地把哈希槽 8 映射到新的 ip:端口号 上。 注意，即使客户端出现bug，过早地执行这个映射更新，也是没有问题的，因为它不会在查询前发送 ASKING 命令，节点 B 会用 MOVED 重定向错误把客户端重定向到节点 A 上。

这个命令主要是当集群处于数据重分布（目前由人工触发）的时候，通过ASK命令来控制客户端的路由。例如，假设我们打算将slot1迁移到新的节点上，迁移过程中，如果客户端访问已经完成迁移的key，这时候节点就会响应ASK告知客户端向目标节点重试。
由于迁移的过程可能持续一段时间，这段时间某个slot的数据同时在新旧两个节点上各分布了一部分，由于move操作会使得客户端的路由缓存变更，如果新旧两个节点对于迁移中的slot上所有不在自己节点的key都回应moved响应，客户端的路由缓存信息可能会频繁变动，因此引入ask临时跳转，将重定向和路由缓存更新分离。


#### 分片的迁移(Resharding)

Redis 集群支持在集群运行过程中添加或移除节点。实际上，添加或移除节点都被抽象为同一个操作，那就是把哈希槽从一个节点移到另一个节点。

在一个稳定的Redis集群下，每一个slot对应的节点是确定的。但是在某些情况下，节点和分片的对应关系需要发生变更：

* 向集群添加一个新节点，就是把一个空节点加入到集群中并把某些哈希槽从已存在的节点移到新节点上。
* 从集群中移除一个节点，就是把该节点上的哈希槽移到其他已存在的节点上。
* 负载不均需要调整slot分布

此时，需要进行分片的迁移。实现这个的核心是能把哈希槽移来移去。从实际角度看，哈希槽就只是一堆键，所以 Redis 集群在reshard 时做的就是把键从一个节点移到另一个节点。

分片迁移的触发和过程控制由外部系统完成，Redis Cluster只提供迁移过程中需要的原语供外部系统调用。这些原语主要包括两种：

1. 节点迁移状态设置：迁移前标记源/目标节点
2. key迁移的原子化命令：迁移的具体步骤

举个例子，假设我们有两个 Redis 节点，称为 A 和 B。我们想要把哈希槽 8 从 节点A 移到 节点B，所以我们发送了这样的命令：

1. 向 节点B 发送状态变更命令：CLUSTER SETSLOT 8 IMPORTING A
2. 向 节点A 发送状态变更命令：CLUSTER SETSLOT 8 MIGRATING B
3. 针对A的slot上的所有key，分别向A发送MIGRATE命令，告知A将对应key的数据迁移到B。

当节点A的状态设置为MIGRATING之后，表示对应的slot 8正在从A迁出，为保证该slot数据的一致性，对于某个迁移中的slot：

* 如果客户端访问的key尚未迁移出，则正常的处理该key。
* 如果key已经迁移出或者根本就不存在该key，则回复客户端ASK信息让其跳转到B执行。

当节点B的状态被设置为IMPORTING后，表示对应的slot正在向B迁入中，此时B对于该迁入的slot的读写请求也会区别对待——只处理该slot上的所有来自于ASK的跳转请求。当来自客户端的正常访问不是从ASK跳转而来的，说明客户端尚不知道迁移正在进行，很有可能操作了一个目前尚未迁移完成的正处在A上的key，如果此时这个key在A上已经被修改了，那么B和A的修改值将在未来发生冲突。所以对于该slot上的所有非ASK跳转而来的操作，B不会进行处理，而是通过MOVED命令让客户端跳转至A执行。这就是前面说过的ASK命令的用处。

这样的状态控制保证了同一个key的迁移过程中总是在源节点执行，迁移后总是在目标节点执行，杜绝了两边同时写导致值冲突的可能性。而且迁移过程中新增的key总是在目标节点执行，源节点不会再有新增的key，使得迁移过程数据(时间)有界，可以在确定的某个时刻结束。

剩下的问题就在于某个key在迁移过程中数据的一致性问题了。单个key的迁移过程被抽象为原子化的`MIGRATE`命令，这个命令完成了 数据传输到B=>等待B接收完成 => 在A上删除该key 的动作。Redis单机对于命令的处理是单线程的，同一个key在执行MIGRATE的过程中不会处理对该key的其他操作，从而保证了迁移的原子性。A和B各自的slave通过主从复制分别同步新老master节点的增删数据。

当slot的所有key从A上迁移到B上后，客户端通过`CLUSTER SETSLOT`命令设置B的分片信息，使之包含迁移的slot。设置过程中会自增一个新的epoch，它大于当前集群的所有epoch值，这个配置信息会通过gossip协议传播到集群中的其他每一个节点，完成分片节点映射关系的更新。


2. Replication
--------------

上面讨论如何将数据划分到没有交集的各个数据节点上，即，不同节点间没有相同的数据。但是，在很多情况下，我们还需要对同一份数据存放在不同的节点上。这样，当某个节点宕机的时候，就可以让备份节点继续提供服务。同时，备份节点还可以提供读服务，缓解主节点压力，提高读性能。

Replication主要要解决的问题是主从副本的数据同步问题，以及主从副本的动态切换问题。

不同的存储系统采用的解决方案不尽相同。有的采用客户端双写、有采用 Quorum W+R>N [with vector clock]，有采用存储层复制的，具体有细分为同步或者异步，拉取或者推送。

Redis采用的是主备复制的方式：master节点对外提供读写服务，slave节点作为master的数据备份，拥有master的全量数据，对外不提供写服务。主备复制由slave主动触发。

具体参见:

1. [Replication](https://redis.io/topics/replication)
2. [记一次Redis错误排查经历](http://arganzheng.life/redis-could-not-get-resource-from-pool.html)

可以了解到非常详细的细节。这里就不赘述了。

对于主从数据同步的数据一致性问题，Redis采用的是命令传播和确定的机制，并不保证主从数据是实时和一致的。

### failover(主从选举和切换问题)

初始阶段，主从节点是通过配置确定的(`SLAVEOF`命令)。但是在运行期间，假设说master挂掉了，这时候为了保证可用性，应该从salves中选举新的master出来提供服务，其他的slave挂载在新的master上。同时要处理好脑裂问题。

这里有两个问题：

1. master故障发现
2. failover决策和执行

Redis 3.0之前，提供了一个sentinel的机制，简单来说过程如下：

1. 通过Redis的PUB/SUB机制互相发现和交换信息
2. 通过向Redis主从节点发送INFO命令获取节点信息
3. 通过定时向Redis主从节点发送PING命令判断实例是否正常
4. 通过类似于Raft协议选举leader sentinel作为failover的发起者
5. leader sentinel确定之后，从master所有的slave中依据一定的规则选取一个作为新的master，告知其他slave连接这个新的master

Redis 3.0引入cluster之后，把failover的功能做进集群里面，不在需要额外部署sentinel了。

具体过程如下：

1. 故障检测：当某个master宕机时，宕机事件如何被集群其他节点感知
2. 故障确认：多个节点就某个master是否宕机如何达成一致
3. slave选举：集群确认了某个master宕机后，如何把它的slave升级成新的master；如果原master有多个slave，选择谁升级。
4. 集群结构变更：选举成功的slave升级为新的master后如何让全集群的其他节点知道以更新它们的集群结构信息。

#### 1. 节点故障检测(Failure detection)

Redis 集群失效检测是用来识别出大多数节点何时无法访问某一个主节点或从节点。当这个事件发生时，就提升一个从节点来做主节点；若如果无法提升从节点来做主节点的话，那么整个集群就置为错误状态并停止接收客户端的查询。

每个节点都有一份跟其他已知节点相关的标识列表。其中有两个标识是用于失效检测，分别是 PFAIL 和 FAIL。PFAIL 表示可能失效（Possible failure），这是一个非公认的（non acknowledged）失效类型。FAIL 表示一个节点已经失效，而且这个情况已经被大多数主节点在某段固定时间内确认过的了。

**PFAIL 标识**

当一个节点在超过 NODE_TIMEOUT 时间后仍无法访问某个节点，那么它会用 PFAIL 来标识这个不可达的节点。无论节点类型是什么，主节点和从节点都能标识其他的节点为 PFAIL。

Redis 集群节点的不可达性（non reachability）是指，发送给某个节点的一个活跃的 ping 包（active ping）(一个我们发送后要等待其回复的 ping 包)已经等待了超过 NODE_TIMEOUT 时间，那么我们认为这个节点具有不可达性。为了让这个机制能正常工作，NODE_TIMEOUT 必须比网络往返时间（network round trip time）大。节点为了在普通操作中增加可达性，当在经过一半 NODE_TIMEOUT 时间还没收到目标节点对于 ping 包的回复的时候，就会马上尝试重连接该节点。这个机制能保证连接都保持有效，所以节点间的失效连接通常都不会导致错误的失效报告。

#### 2. 故障确认(Failure agreement)

正如前面分析的，PFAIL只是一个非公认的（non acknowledged）失效类型。可能只是跟目标节点断开连接而已。

**FAIL 标识**

单独一个 PFAIL 标识只是每个节点对于其他节点的本地信息，但是它还也不足够触发从节点的提升。要让一个节点真正被认为失效了，那需要让 PFAIL 状态上升为 FAIL 状态。每个节点向其他每个节点发送的 gossip 消息中有包含一些随机的已知节点的状态。最终每个节点都能收到一份其他每个节点的节点状态。通过这种方式，每个节点都有机会告诉集群中的其他节点关于它们检测到的失效节点。

当下面的条件满足的时候，会考虑让 PFAIL 状态升级为 FAIL 状态：

* 某个节点，我们称为节点 A，标记另一个节点 B 为 PFAIL。
* 节点 A 通过 gossip 协议收集到集群中大部分主节点标识的节点 B 的状态信息。
* 大部分主节点在 `NODE_TIMEOUT * FAIL_REPORT_VALIDITY_MULT`(目前实现是2) 时间内标记节点 B 为 PFAIL或者FAIL 状态。

如果以上所有条件都满足了，那么节点 A 会：

* 标记节点 B 为 FAIL。
* 向所有可达节点发送一个 FAIL 消息。

FAIL 消息会强制每个接收到这消息的节点把节点 B 标记为 FAIL 状态。

注意，FAIL 标识基本都是单向的，也就是说，一个节点能从 PFAIL 状态升级到 FAIL 状态，但要清除 FAIL 标识只有以下两种可能方法：

* 节点已经恢复可达的，并且它是一个主节点，但是并没有服务任何slot。在这种情况下，FAIL 标识可以清除掉，因为没有任何slot的主节点实际上并没有参与到集群中，只是等着加入集群。
* 节点已经恢复可达的，并且它是一个主节点，但经过了很长时间（N * NODE_TIMEOUT）后也没有检测到任何从节点被提升了。这时候还是让他先回去，因为没有候选人可以替代。

需要注意的是 PFAIL -> FAIL 的转变使用的是一种弱协议（weak agreement）：

1. 节点是在一段时间内收集其他节点的信息，是在不同时间从不同节点收集到关于其他节点的状态信息，来推断得到某个节点的当前状态，这就不能保证一定是稳定准确的结论。 
2. 当每个节点检测到 FAIL 节点的时候会强迫集群里的其他节点把各自对该节点的记录更新为 FAIL，但没有一种方式能保证这个消息能到达所有节点。比如有个节点可能检测到了 FAIL 的节点，但是因为分区问题，这个节点无法到达其他任何一个节点。

本质上来说，FAIL 标识只是用来触发从节点提升（slave promotion）算法的安全部分。

#### 3. slave的选举和提升(Slave election and promotion)

当上面的B节点已经被集群（必须至少包括它的一个slave）公认为FAIL状态时，那么它的从节点就会发起竞选，期望替代B成为新的master。从节点的选举和提升都是由从节点处理的，其他的主节点会投票要提升哪个从节点。

当以下条件满足时，一个从节点可以发起选举：

* 该从节点的主节点处于 FAIL 状态。
* 这个主节点负责的哈希槽数目不为零。
* 从节点和主节点之间的复制连接（replication link）断线不超过一段给定的时间（配置项），这是为了确保从节点的数据不要落后太多。

一个从节点想要被推选出来，那么第一步应该是提高它的 currentEpoch 计数，并且向主节点们请求投票。

1. 从节点通过广播一个`FAILOVER_AUTH_REQUEST`数据包给集群里的每个主节点来请求选票。要求所有收到这条消息并且具有投票权的主节点向这个节点投票。从节点最多等待 `NODE_TIMEOUT * 2` 这么长时间，一般来说至少等2秒钟。
2. 如果一个主节点具有投票权（它正在复制处理槽，并且在最近的时间段内尚未投票给其他的从节点)。那么主节点将会回复一个`FAILOVER_AUTH_ACK`消息，并且在 `NODE_TIMEOUT * 2` 这段时间内不能再给同个主节点的其他从节点投票。这样做的目的主要是为了避免多个slaves被同时选举上。
3. 从节点会忽视所有 epoch 比 发起投票请求时候的 currentEpoch 小的 AUTH_ACK 回应，这样能避免把之前的投票的算为当前的合理投票。
4. 每个参与选举的从节点都会接收到主节点的`FAILOVER_AUTH_ACK`消息，并根据自己收到多少条这种消息来统计自己获得了多少主节点的支持。
5. 如果集群里有N个具有投票权的主节点，那么当一个从节点收到了大多数(N/2+1)主节点的投票支持(ACKs)，那么它就赢得了选举。
6. 如果在一个配置纪元(epoch)里无法在 2 * NODE_TIMEOUT 时间内得到大多数投票，那么当前选举会被中断并在 NODE_TIMEOUT * 4 这段时间后由尝试发起选举，直到选出新的主节点为止。

这种选举主节点的方法与选举领头Sentinel的方法非常类似，因为两者都是基于Raft算法的领头选举(leader election)方法来实现的。

#### 4. 集群结构变更

集群结构变更信息也是通过gossip协议进行扩散。

一旦有从节点赢得选举，它就会以最新的epoch 通过 ping/pong 数据包向其他节点宣布自己已经是主节点，并提供它负责的哈希槽。让集群中的其他节点尽快的更新拓扑信息。

为了加速其他节点的重新配置，该节点会广播一个 pong 包 给集群里的所有节点（那些现在访问不到的节点最终也会收到一个 ping 包或 pong 包，并且进行重新配置）。

其他节点会检测到有一个新的主节点（带着更新的configEpoch）在负责处理之前一个旧的主节点负责的哈希槽，然后就升级自己的配置信息。旧主节点的从节点，或者是经过故障转移后重新加入集群的该旧主节点，不仅会升级配置信息，还会配置成为新主节点的slaves。


3. Cluster
----------

在 Redis 集群中，节点负责存储数据、记录集群的状态（包括键值到正确节点的映射）。集群节点同样能自动发现其他节点，检测出没正常工作的节点， 并且在需要的时候在从节点中推选出主节点。

为了执行这些任务，所有的集群节点都通过一个TCP总线(TCP bus)和一个称之为 Redis Cluster Bus 的二进制协议建立连接。 每一个节点都通过Cluster bus与集群上的其余每个节点连接起来。节点间使用 gossip 协议来传播集群的信息，包括：发现新的节点、 发送ping心跳包（用来确保其他的节点都在正常工作中）、在特定情况发生时发送集群消息。集群连接也用于在集群中发布或订阅消息。

在很多分布式系统，会使用ZK做集群统一配置信息服务、节点发现以及leader选举。但是Redis Cluster为了避免引入ZK依赖，自己实现了这一切。

用到的算法和协议在前面其实已经有提到过，主要是Raft协议和Gossip协议：

* Raft协议：用于leader选举
* Gossip协议：用于节点发现和同步各个节点的信息

由于去中心化的架构下不存在统一的配置中心，各个节点对整个集群状态的认知来自于各个节点间的信息交互。在Redis Cluster中，这个信息交互通过Redis Cluster Bus来完成，后者端口独立。

* 所有的节点两两相互连接
* 集群消息通信通过集群总线(TCP bus)通信，集群总线端口大小为客户端服务端口+10000，这个10000是固定值
* 节点与节点之间通过二进制协议(Redis Cluster Bus)进行通信

### 集群状态

Redis Cluster中的每一个节点内部都保存了集群的状态信息，这些信息存储在clusterState中，它的数据结构如下所示：

```
typedef struct clusterNode {
    mstime_t ctime; /* Node object creation time. */
    char name[CLUSTER_NAMELEN]; /* Node name, hex string, sha1-size */
    int flags;      /* CLUSTER_NODE_... */
    uint64_t configEpoch; /* Last configEpoch observed for this node */
    unsigned char slots[CLUSTER_SLOTS/8]; /* slots handled by this node */
    int numslots;   /* Number of slots handled by this node */
    int numslaves;  /* Number of slave nodes, if this is a master */
    struct clusterNode **slaves; /* pointers to slave nodes */
    struct clusterNode *slaveof; /* pointer to the master node. Note that it
                                    may be NULL even if the node is a slave
                                    if we don't have the master node in our
                                    tables. */
    mstime_t ping_sent;      /* Unix time we sent latest ping */
    mstime_t pong_received;  /* Unix time we received the pong */
    mstime_t fail_time;      /* Unix time when FAIL flag was set */
    mstime_t voted_time;     /* Last time we voted for a slave of this master */
    mstime_t repl_offset_time;  /* Unix time we received offset for this node */
    mstime_t orphaned_time;     /* Starting time of orphaned master condition */
    long long repl_offset;      /* Last known repl offset for this node. */
    char ip[NET_IP_STR_LEN];  /* Latest known IP address of this node */
    int port;                   /* Latest known clients port of this node */
    int cport;                  /* Latest known cluster port of this node. */
    clusterLink *link;          /* TCP/IP link with this node */
    list *fail_reports;         /* List of nodes signaling this as failing */
} clusterNode;

typedef struct clusterState {
    clusterNode *myself;  /* This node */
    uint64_t currentEpoch;
    int state;            /* CLUSTER_OK, CLUSTER_FAIL, ... */
    int size;             /* Num of master nodes with at least one slot */
    dict *nodes;          /* Hash table of name -> clusterNode structures */
    dict *nodes_black_list; /* Nodes we don't re-add for a few seconds. */
    clusterNode *migrating_slots_to[CLUSTER_SLOTS];
    clusterNode *importing_slots_from[CLUSTER_SLOTS];
    clusterNode *slots[CLUSTER_SLOTS];
    uint64_t slots_keys_count[CLUSTER_SLOTS];
    rax *slots_to_keys;
    /* The following fields are used to take the slave state on elections. */
    mstime_t failover_auth_time; /* Time of previous or next election. */
    int failover_auth_count;    /* Number of votes received so far. */
    int failover_auth_sent;     /* True if we already asked for votes. */
    int failover_auth_rank;     /* This slave rank for current auth request. */
    uint64_t failover_auth_epoch; /* Epoch of the current election. */
    int cant_failover_reason;   /* Why a slave is currently not able to
                                   failover. See the CANT_FAILOVER_* macros. */
    /* Manual failover state in common. */
    mstime_t mf_end;            /* Manual failover time limit (ms unixtime).
                                   It is zero if there is no MF in progress. */
    /* Manual failover state of master. */
    clusterNode *mf_slave;      /* Slave performing the manual failover. */
    /* Manual failover state of slave. */
    long long mf_master_offset; /* Master offset the slave needs to start MF
                                   or zero if stil not received. */
    int mf_can_start;           /* If non-zero signal that the manual failover
                                   can start requesting masters vote. */
    /* The followign fields are used by masters to take state on elections. */
    uint64_t lastVoteEpoch;     /* Epoch of the last vote granted. */
    int todo_before_sleep; /* Things to do in clusterBeforeSleep(). */
    /* Messages received and sent by type. */
    long long stats_bus_messages_sent[CLUSTERMSG_TYPE_COUNT];
    long long stats_bus_messages_received[CLUSTERMSG_TYPE_COUNT];
    long long stats_pfail_nodes;    /* Number of nodes in PFAIL status,
                                       excluding nodes without address. */
} clusterState;
```

**说明**

1、clusterState 记录了从集群中的某个节点的视角的集群状态信息，包括数据的分片方式、节点的主从关系，并通过epoch作为版本号实现集群结构（状态）信息的一致性，同时也控制着数据迁移和故障转移的过程。每个节点维护一份:

* myself：指针指向自己的clusterNode
* currentEpoch：表示整个集群中的最大版本号，集群信息每变更一次，该版本号就会自增以保证每个信息的版本号唯一
* nodes：包含了本节点所知道的集群中的所有节点的信息(clusterNode)，其中也包括它自己，为clusterNode指针数组
* slots：slot与clusterNode指针映射关系
* migrating_slots_to, importing_slots_from：记录slots的迁移信息
* failover_auth_time, failover_auth_count, failover_auth_sent, failover_auth_rank, failover_auth_epoch：Failover相关

2、clusterNode，代表集群中的一个节点，其中比较关键的信息包括：

* configEpoch: 当前节点见过的最大epoch
* slots：槽位图，由当前clusterNode负责的slot为1
* salve, slaveof：主从关系信息
* ping_sent, pong_received：心跳包收发时间
* `clusterLink *link`：Node间的联接
* `list *fail_reports`：收到的节点不可达投票
*  ip, port


### 集群消息和通讯

集群间互相发送消息，使用另外的端口，所有的消息在该端口上完成，可以称为集群消息总线(Redis Cluster Bus)，这样可以做到不影响客户端访问redis，可见redis对于性能的追求。

所有消息都由消息头包裹，消息头可以认为是消息的一部分。消息头由 [cluster.h/clusterMsg](http://download.redis.io/redis-stable/src/cluster.h) 结构记录，如下：

```
structclusterMsg{

	uint32_t totlen; //消息总长度，包括消息头长度和正文长度

	uint16_t type; //消息类型

	uint16_t count; //消息正文包含节点信息数量，只有在meet、ping、pong这三种涉及到gossip协议的类型使用

	uint64_t currentEpoch; //发送者的配置纪元

	uint64_t configEpoch; //该节点是主节点时，是发送者的配置纪元；是从节点时，是对应正在复制的主节点的配置纪元

	char sender[REDIS_CLUSTER_NAMELEN]; //发送者名字(ID)

	unsigned char myslots[REDIS_CLUSTER_SLOTS/8]; //发送者目前的槽指派信息

	char slaveof[REDIS_CLUSTER_NAMELEN]; //主节点时记录的是40位长的都是0的字符串，从节点时记录的是复制的主节点的名字

	uint16_t port; //发送者端口号

	uint16_t flag; //发送者标识值

	unsigned char state; //发送者所处的集群状态

	union clusterMsgData data; //消息的正文

} clusterMsg;
```

clusterMsg.data属性也就是消息的正文是一个联合体(union)，共有三种类型结构体，包括ping、fail、publish，其中pong、meet类型都和ping一样。

```
union clusterMsgData {
	// MEET、PING、PONG消息的正文
	struct {
		// 每条MEET、PING、PONG消息都包含两个clusterMsgDataGossip结构
		clusterMsgDataGossip gossip[1];
	} ping;

	// FAIL 消息的正文
	struct {
		clusterMsgDataFail about;
	} fail;

	// PUBLISH 消息的正文
	struct {
		clusterMsgDataPublish msg;
	} publish;

	// UPDATE 消息的正文
    struct {
        clusterMsgDataUpdate nodecfg;
    } update;
};
```

从这里看起来貌似集群只有MEET、PING、PONG、FAIL、PUBLISH和UPDATE六种消息。但是我们上面在讨论failover的时候知道还有其他的消息的，所以其实Redis Cluster一共是有9如下9种消息的：

```
/* Message types.
 *
 * Note that the PING, PONG and MEET messages are actually the same exact
 * kind of packet. PONG is the reply to ping, in the exact format as a PING,
 * while MEET is a special PING that forces the receiver to add the sender
 * as a node (if it is not already in the list). */
#define CLUSTERMSG_TYPE_PING 0          /* Ping */
#define CLUSTERMSG_TYPE_PONG 1          /* Pong (reply to Ping) */
#define CLUSTERMSG_TYPE_MEET 2          /* Meet "let's join" message */
#define CLUSTERMSG_TYPE_FAIL 3          /* Mark node xxx as failing */
#define CLUSTERMSG_TYPE_PUBLISH 4       /* Pub/Sub Publish propagation */
#define CLUSTERMSG_TYPE_FAILOVER_AUTH_REQUEST 5 /* May I failover? */
#define CLUSTERMSG_TYPE_FAILOVER_AUTH_ACK 6     /* Yes, you have my vote */
#define CLUSTERMSG_TYPE_UPDATE 7        /* Another node slots configuration */
#define CLUSTERMSG_TYPE_MFSTART 8       /* Pause clients for manual failover */
#define CLUSTERMSG_TYPE_COUNT 9         /* Total number of message types. */
```

只不过只有6种需要消息正文而已。

**说明**

1、集群节点间相互通信使用了gossip协议的push/pull方式，ping和pong消息，节点会把自己的详细信息和已经跟自己完成握手的3个节点地址发送给对方，详细信息包括消息类型，集群当前的epoch，节点自己的epoch，节点复制偏移量，节点名称，节点数据分布表，节点master的名称，节点地址，节点flag位，节点所处的集群状态。节点根据自己的epoch和对方的epoch来决定哪些数据需要更新，哪些数据需要告诉对方更新。然后根据对方发送的其他地址信息，来发现新节点的加入，从而和新节点完成握手。

2、节点默认每秒在集群中的其他节点选择一个节点，发送ping消息。选择节点步骤是：

* 随机 5 个节点。
* 跳过断开连接和已经在ping还没收到pong响应的节点。
* 从筛选的节点中选择最近一次接收pong回复距离现在最旧的节点。

除了常规的选择节点外，对于那些一直未随机到节点，redis也有所支持。当有节点距离上一次接收到pong消息超过节点超时配置的一半，节点就会给这些节点发送ping消息。

ping消息会带上其他节点的信息，选择其他节点步骤是：

* 最多选择3个节点。
* 最多随机遍历6个节点，如果因为一些条件不能被选出，可能会不满3个。
* 忽略自己。
* 忽略正在握手的节点。
* 忽略带有 NOADDR 标识的节点。
* 忽略连接断开而且没有负责任何slot的节点。

ping消息会把发送节点的ping_sent改成当前时间，直到接收到pong消息，才更新ping_sent为0。当ping消息发送后，超过节点超时配置的一半，就会把发送节点的连接断开。超过节点超时配置，就会认为该节点已经下线。

接收到某个节点发来的ping或者pong消息，节点会更新对接收节点的认识。比如该节点主从角色是否变化，该节点负责的slot是否变化，然后获取消息带上的节点信息，处理新节点的加入。

3、没有PFAIL消息，因为PFAIL包含在PING消息中。


### 一致性的达成

当集群结构不发生变化的时候，集群中的各个节点通过Gossip协议可以在几轮交互之后得知全集群的结构信息，并且达到一致的状态。然而，故障转移、分片迁移等情况的发生会导致集群结构变更，由于无统一的配置服务器，变更的信息只能靠各个节点自行协商，优先得知变更信息的节点利用epoch变量将自己的最新消息扩散到整个集群，达到最终一致。

* clusterNode的configEpoch属性描述的粒度是单个节点，即某个节点的数据分片，主备信息版本。
* clusterState的currentEpoch属性的粒度是整个集群，它的存在是用来辅助epoch自增生成。由于currentEpoch也是各个节点各自保存的，Redis Cluster在结构发生变更时，通过一定的时间窗口控制和更新规则保证每个节点看到的currentEpoch都是最新的。

集群信息的更新遵循一下规则：

* 当某个节点率先知道了信息变更时，这个节点将currentEpoch自增使之成为集群中的最大值，再用自增后的currentEpoch作为新的epoch版本。
* 当某个节点收到了比自己大的currentEpoch时，更新自己的currentEpoch值使之保持最新
* 当收到的Redis Cluster Bus消息中某个节点信息的epoch值大于接收者自己内部存储的epoch值时，意味着自己的信息太旧了，此时将自己的映射信息更新为消息的内容。
* 当收到的Redis Cluster Bus消息中某个节点信息未包括在接收节点的内部配置信息时，意味着接收者尚未意识到消息所指节点的存在，此时接收者直接将消息的信息添加到自己的内部配置信息中。

上述规则保证了消息的更新始终是单向的，始终朝着epoch值更大的信息收敛，同时epoch也随着每次配置变更时currentEpoch的自增而单向增加，确定了各节点更新的方向稳定。


参考文档
-------

1. [Redis cluster tutorial](https://redis.io/topics/cluster-tutorial)
2. [Redis 集群规范](http://www.redis.cn/topics/cluster-spec.html)
3. [redis3.0 cluster功能介绍](http://weizijun.cn/2015/12/30/redis3.0%20cluster%E5%8A%9F%E8%83%BD%E4%BB%8B%E7%BB%8D/) 挺详细的
