---
title: 获取redis集群信息
layout: post
catalog: true
tags: [redis, cluster, monitor]
category: [技术]
---

redis info 命令只能获取单个实例的信息。写了一个 shell ，可以方便的获取 redis 集群的汇总信息:

```shell
#!/bin/sh

ip=10.21.61.43
port=6379
sum_size=0
sum_mem=0
sum_conn=0
for addr in `redis-cli -p $port -h $ip cluster nodes | awk '{print $2}'`
do
    host=`echo $addr | awk -F: '{print $1}'`
    hport=`echo $addr | awk -F: '{print $2}'`
    role=`redis-cli -p $hport -h $host info | grep "role:" | awk -F: '{print $2}' | tr -d '\r\n'`

    if [ $role = "slave" ];then
        continue
    fi

    dbsize=`redis-cli -p $hport -h $host dbsize`
    mem=`redis-cli -p $hport -h $host info | grep "used_memory:" | awk -F: '{print $2}' | tr -d '\r\n' | tr -d 'G'`
    conn=`redis-cli -p $hport -h $host info | grep "connected_clients:" | awk -F: '{print $2}' | tr -d '\r\n' | tr -d 'G'`
    mem=$(awk "BEGIN{print $mem/1024.0/1024.0/1024.0}")
    
    echo "$host $hport mem $mem G , dbsize $dbsize, connected_clients $conn"

    sum_size=`expr $dbsize + $sum_size`
    sum_mem=$(awk "BEGIN{print $sum_mem + $mem}")
    sum_conn=$(awk "BEGIN{print $sum_conn + $conn}")
done

echo "集群dbsize 总数:$sum_size"
echo "集群内存（G） 总数:$sum_mem"
echo "连接数 总数:$sum_conn"
```
