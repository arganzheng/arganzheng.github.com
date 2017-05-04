---
layout: post
title: Bloom filter在分布式环境中的应用
tags: [Bloom filter, 分布式]
catalog: true
category: [技术]
---


概述
----

布隆过滤器是一个应用非常广泛的概率型数据结构，一般用于判断一个元素是否存在一个集合中，比如在字处理软件中，需要检查一个英语单词是否拼写正确（也就是要判断它是否在已知的字典中）；在 缓存系统中，判断一个元素是否在缓存中；在网络爬虫里，一个网址是否被访问过等等。最直接的方法就是将集合中全部的元素存在计算机中，遇到一个新元素时，将它和集合中的元素直接比较即可。一般来讲，计算机中的集合是用哈希表（hash table）来存储的。它的好处是快速准确，缺点是费存储空间。当集合比较小时，这个问题不显著，但是当集合巨大时，哈希表存储效率低的问题就显现出来了。关于这个，只需要根据元素的数量和大小简单的计算一下就知道了。虽然可以适用分布式K-V系统（如Redis）来承载，但是成本仍然高昂。布隆过滤器只需要哈希表 1/8 到 1/4 的大小就能解决同样的问题，以一定的误判率为代价。所需要的内存大小可以通过公式精确的计算出来：[Bloom Filter Calculator](https://krisives.github.io/bloom-calculator/)。

布隆过滤器其实是一个比较简单的数据结构，不难实现，接口简单，代码也不长。已经有很多开源的实现，像谷歌的guava基础库就有实现。
布隆过滤器有许多的变种，像 Counting Bloom Filter, 可以支持删除元素。


分布式环境下的应用
----------------

考虑一个应用场景，比如用内存实现session登录验证，数据量和并发量少的时候，单机的HashMap是可以非常简单的实现。但是这样会导致session服务器成为单点，一种做法就是session赋值，每台session服务器的session(HashMap)互相同步，这样保证请求到任何机器都可以得到一样的结果。但是这种方式有比较严重的缺陷，不考虑赋值带来的同步成本和一致性问题，单是如果数据量一大，单机内存就存不下这么多的session数据了。所以解决方案就是采用对session进行sharding，存放在分布式kv中，比如redis。

那么对于布隆过滤器呢，也有类似的问题。单机情况下非常好理解，一旦落到分布式环境，要怎么解决数据和服务的扩展性问题呢？带着这个问题我看了一下布隆过滤器的一个非常重要的应用——[Squild的CacheDigests](http://wiki.squid-cache.org/SquidFaq/CacheDigests)，它其实就是基于[Summary Cache协议](https://www.cs.umd.edu/class/spring2007/cmsc818s/Lectures/bloom_sc.pdf)，主要为了解决 Web Cache Sharing 问题。

**Web Cache Sharing**

为了减少带宽使用、降低web server的负载，我们常常通过代理服务器（proxy）浏览网页，浏览过的网页会被cache在proxy中，如果有别的请求或者下一次访问相同的网页，proxy就不用将请求转给web server，而是直接将本地cache的网页返回。当然，网页是经常变化的，因此proxy在返回cache的网页之前会做一些检查，只有满足特定的条件之后才会使用cache的网页，以保证client得到的不是过期的网页。
 
一旦网络中存在很多proxy，我们自然希望它们之间能够共享cache，这样可以进一步减少对web server的访问，web cache sharing的概念就由此而来。

**ICP(Internet Cache Protocol)**

在bloom filter被提议应用在web cache sharing之前，Internet Cache Protocol（ICP）作为一种web cache之间相互协作的协议已经被广泛应用。ICP的设计思路是，proxy中只要有cache miss发生，马上发送查询消息给它邻近的proxy，看能不能在它们的cache中找到相关的网页。这种方案随着proxy数量的增加，网络通信量和proxy的CPU占用率都成二次方增长，因此扩展性很差，并有很大的开销。


**Summary Cache**

为了减少这些开销，一篇名为 [Summary Cache: A Scalable Wide-Area Web Cache Sharing Protocol](http://www.cs.wisc.edu/~jussara/papers/00ton.pdf) 的论文提出了一种名为Summary Cache的协议。也就是在这篇文章中，Counting Bloom Filter 被提出。
 
Summary Cache采用了和ICP完全不同的设计思路。为了proxy之间能够协作，它们必须知道彼此cache的内容。ICP采用的是实时获取的方案，需要协作时才发请求广播式地查询。这样当然获取的信息是最准确的，但也导致通信的开销很大。Summary cache利用了这样一个简单的事实：proxy中的cache在短时间内不会有很大幅度的变化。因此它希望将proxy中cache的内容用简洁的形式表示（cache summary），让每个proxy保存所有其它proxy的cache summary，然后定期更新summary。这样一来，每当cache miss发生时，proxy会首先检查自己保存的所有cache summary，然后将请求发给那些检查结果为“有”的proxy。
 
这种设计有两个关键的点需要认真考虑，一个是cache summary更新的频率，另一个是如何简洁地表示cache的内容，即如何构建cache summary。对于第一个问题，论文作者的回答是，当cache更新的内容在1%－10%之间时发送更新信息，或者每隔一段时间发送更新，这段时间的长度也应该保证cache的更新幅度在1%－10%之间。对于第二个问题，就是用bloom filter。每一个cache的网页由URL唯一标识，因此proxy的cache内容可以表示为一个URL列表。进而我们可以将URL列表这个集合用bloom filter表示，从而支持membership query。当false positive发生时，无非是请求的proxy以为另一个proxy有某个网页而实际没有，这样会造成一定的延时。由于cache的网页本身就有可能过期，因此很低的false positive概率很难造成什么影响。在cache更新后通知其它proxy时，有两种选择，一种是发送整个更新后的bloom filter，另一种是只发送bloom filter的更新位置。前者适合在更新频率较低的情况下使用，后者在更新频率高的情况下使用。
 
**cache更新问题**

Proxy在发送cache summary以及保存其它proxy的cache summary时，无疑使用的是占用空间更少的标准bloom filter。但是proxy同时要表示自己本地的cache集合，由于本地的cache不断加入新的网页，也不断移除不经常使用的网页，因此是一个动态变化的集合。标准bloom filter不支持删除操作，因此不能胜任这个工作。为了避免不断重建bloom filter，作者通过把bloom filter中的每一位扩展为一个counter来支持删除操作，同时，位到counter的扩展使得CBF到bloom filter的转化非常方便。从这个应用我们也可以看到，很多情况下我们需要bloom filter支持删除操作，但同时也希望支持删除操作之后还能还原为标准的bloom filter，以减少网络传输的信息量。

但是Counting Bloom Filter比起标准Bloom Filter需要更多的内存空间，所以真实应用中，像Squild的CacheDigests就是采用定时重新构建标准Bloom Filter的方式。


然而Summery cache方式其实还是采用的是session复制的方式，只是采用的是定时拉取(Poll)其他节点的cache概要(Summary, 即Bloom filter)的方式。那么大数据量时候的单机内存问题呢？Squild的解决方案是将Bloom Filter Swapped out 到磁盘。这样也可以解决持久化问题，方便恢复和系统重启。具体实现没有深入源码，应该是Memory Map File。


参考文章
-------

1. [Bloom Filters and Summary Cache](https://www.cs.umd.edu/class/spring2007/cmsc818s/Lectures/bloom_sc.pdf)
2. [Data Structures from the Future: Bloom Filters, Distributed Hash Tables, and More!](https://www.usenix.org/legacy/event/lisa10/tech/slides/limoncelli.pdf)
3. [Bloom Filter应用之Web Cache Sharing](http://blog.csdn.net/jiaomeng/article/details/1531423)

