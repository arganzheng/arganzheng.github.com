---
title: kafka broker间歇出现CLOSE_WAIT问题
layout: post
catalog: true
---

使用kafka有一段时间了，偶尔会出现一些问题，比如Kafka集群的某个broker会突然卡住，排除发现改broker有大量的CLOSE_WAIT状态，而且Recv-Q有数据没有处理，猜测可能是client发送了请求数据，broker还没处理完，client退出了，broker再也不能处理完那些请求数据，所以不能发送FIN。

![kafka-broker-close-wait](/img/in-post/kafka-broker-close-wait.jpg)


这个情况已经出现过几次，重启就好了，但是为什么会出现这么多的CLOSE_WAIT原因一直未明。比较麻烦的是，该broker与ZK的heart beat并没有失败，所以ZK并不认为其挂掉了，因此也不会触发HA。

Kafka还有一个问题很蛋疼，每一个Consumer或者Broker的增加或者减少都会触发 Consumer Rebalance。因为每个Consumer只负责调整自己所消费的Partition，为了保证整个Consumer Group的一致性，当一个Consumer触发了Rebalance时，该Consumer Group内的其它所有其它Consumer也应该同时触发Rebalance。而在Rebalancing状态下该consumer group都处于不可读写的状态，虽然这个时间很短。
