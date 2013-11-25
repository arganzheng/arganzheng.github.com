---
layout: post
title: 高并发下额度限制问题
---


问题
----

在很多场景下资源是有额度的，跟事务一样，我们要求资源不多不少的分配出去。比如

1. 普通会员只能免费发布10个产品
2. 火车票订票系统
3. 秒杀系统
4. 抽奖系统
5. 流量控制系统

等等，非常常见。

一般的想法就是：假设资源S的总份额是C，当前已经消耗的份额是X（或者已知当前剩余的份额为N）。每个用户要获取资源时，我们做如下判断：

    C = getTotalResourceCount();
    X = getResourceConsume();
    
    if (X>C){
        // allow，用户可以进行 发布产品，下单支付等。
    }else{
        // out of resource!
    }

这段程序在单用户可以运行正确，但是在分布式高并发下就会出错。原因在于每个用户拿到的X或者N都是过去时的状态（譬如我们看到的星光，其实到我们视线的时候已经是很久以前的状态了。）如果要保证严格的正确性，那么必须加锁排队串行操作。

    C = getTotalResourceCount();
    X = getResourceConsume();
    
    if (X<=C){
        synchronized(C){ // 加锁排队操作
            // 重新获取X
            X = getResourceConsume();
            
            if (X<=C)
                // allow，用户可以进行 发布产品，下单支付等。
            }
        }
    }else{
        // out of resource!
    }

这里用了`Double-Checked-Locking` 技术提供并发度。

**说明** synchronized其实是使用了内存锁，只是对同个JVM的有效，如果是多台机器（分布式环境），那么需要使用分布式锁。比如数据库记录锁，或者memcached，或者zookeeper。参见：[A simple distributed lock with memcached](http://bluxte.net/musings/2009/10/28/simple-distributed-lock-memcached)


但是并发度还是不够，因为最后的那一段还是每次只能有一个人执行，即使此时X远远小于C。

如果允许少量的资源溢出（即允许多发放资源）那么可以这么处理。就是当X很小的，直接放过，只有当X和C比较接近的时候（C-X<某个阀值），再做排队。

但还是有个问题，就是判断 C-X < 某个阀值，还是会出现并发问题。所以这种方式不能保证正确性。不过如果阀值比较大的话，出错的概率会比较小。
