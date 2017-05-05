---
layout: post
title: redis slave的key过期机制
tags: [redis]
catalog: true
category: [技术]
---


Redis的Scale out一直是大家关注的，因为它的单进程单线程模型，scale out基本是势在必行。对于读多写少的应用，最简单的方式就是利用Redis的Replication机制，master提供写，slave提供读。但是在 3.2[^1] 之前，由于slave所有写操作都来自于master，被动清理的key并不会发送DEL命令到slave，所以会导致一个key在master已经过期了，但是在slave上还能查到。这对于通过read only slave进行redis读扩展来说是一个不可接受的事情。

Redis的作者很早就意识到这个问题了[^2]，而且在3.2之后也对这个问题进行了彻底的修复[^1]。不过随着Redis 3.0集群的出现，redis的scale out有了更多的选择。


#### Footnotes

[^1]: [Improve expire consistency on slaves #1768](https://github.com/antirez/redis/issues/1768)
[^2]: [Redis slaves, while not allowed to expire keys without master input, should reply to clients consistently with the key expire information. #187](https://github.com/antirez/redis/issues/187)