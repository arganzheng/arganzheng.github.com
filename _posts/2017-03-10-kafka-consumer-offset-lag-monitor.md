---
title: Kafka offset lag监控
layout: post
---

利用NOAH的自定义脚本监控功能，写了一个脚本监控kafka的consumer offset lag，如果大于10000就报警。

脚本如下：

	#!/bin/bash

	old_IFS=$IFS
	IFS=$'\n'
	 # get all the consumer group of kafka
	group_list=`/home/work/kafka/bin/kafka-consumer-groups.sh --bootstrap-server localhost:8092 --list`
	for group in $group_list; do
		# get the offset lag of each group
		group_info=`/home/work/kafka/bin/kafka-consumer-groups.sh --bootstrap-server localhost:8092  --group ${group} --describe`

		# init variable
		group_name="GROUP"
		topic_name="TOPIC"
		total_lag=0

		for line in $group_info; do
		    group_name=`echo $line | awk '{print $1}'`
	            if [[ $group_name == GROUP*  ||  $group_name == Consumer* ]] ; then
			# echo "$group_name is ignore.."
			continue
		    fi
		    topic_name=`echo $line | awk '{print $2}'`
	            if [[ $topic_name == __consumer_offsets* ]] ; then
			# echo "$topic_name is ignore.."
			continue
		    fi

		    lag_of_this_partition=`echo $line | awk '{print $6}'`

		    total_lag=$((total_lag + lag_of_this_partition))
		done

	        if [[ $group_name != GROUP*  &&  $group_name != Consumer* && $topic_name != __consumer_offsets* ]] ; then
		    key="kafkaOffsetLag-${group_name}-${topic_name}"
		    value=$total_lag
	            if [[ $total_lag > 10000 ]]; then
	                value="LARGE"
	            else
	                value="SMALL"
	            fi
		    echo $key:$value
		fi
	done

**说明**

1、`if [[ $group_name == GROUP*  ||  $group_name == Consumer* ]]` 判断是为了过滤命令的header以及Kafka rebalancing时输出的提示信息：

	Consumer group `crawled_html_data_HDFS_pusher_group` is rebalancing.


2、正常来说应该输出具体的lag，而不是简单的LARGE或者SMALL，但是因为NOAH的监控表达式不支持数字的判断，只有正则相等性匹配，所以只能这么搞了。。

	regex_equal('kafkaOffsetLag.*', 'LARGE') 


----

### 补充：Kafka的监控命令

	[work@nj03-bdg-kg-offline-01.nj03 kafka]$ bin/kafka-run-class.sh kafka.tools.ConsumerOffsetChecker --zookeeper localhost:2181 --group db_updator_group3  --topic kg_url_expand
	[2017-02-28 15:38:45,221] WARN WARNING: ConsumerOffsetChecker is deprecated and will be dropped in releases following 0.9.0. Use ConsumerGroupCommand instead. (kafka.tools.ConsumerOffsetChecker$)
	Group           Topic                          Pid Offset          logSize         Lag             Owner
	db_updator_group3 kg_url_expand                  0   36951029        36951029        0               none
	db_updator_group3 kg_url_expand                  1   36935997        36935997        0               none
	db_updator_group3 kg_url_expand                  2   36949551        36949551        0               none
	db_updator_group3 kg_url_expand                  3   36945975        36945975        0               none
	db_updator_group3 kg_url_expand                  4   36950408        36950408        0               none
	db_updator_group3 kg_url_expand                  5   36950323        36950323        0               none
	db_updator_group3 kg_url_expand                  6   36937171        36937171        0               none
	db_updator_group3 kg_url_expand                  7   36943758        36943758        0               none
	db_updator_group3 kg_url_expand                  8   36946116        36946116        0               none
	db_updator_group3 kg_url_expand                  9   36942941        36942941        0               none
	db_updator_group3 kg_url_expand                  10  36948368        36948368        0               none
	db_updator_group3 kg_url_expand                  11  36943450        36943450        0               none
	db_updator_group3 kg_url_expand                  12  36950434        36950434        0               none
	db_updator_group3 kg_url_expand                  13  36930054        36930054        0               none
	db_updator_group3 kg_url_expand                  14  36930610        36930610        0               none

	[work@nj03-bdg-kg-offline-01.nj03 kafka]$ bin/kafka-consumer-groups.sh --bootstrap-server localhost:8092 --list
	rb_test_group
	page_entity_consumer_group
	pyspider_consumer_group
	page_entity_HDFS_pusher_group
	entity_consumer_group_2
	original_page_entity_HDFS_pusher_group
	KafkaManagerOffsetCache
	crawled_html_data_HDFS_pusher_group
	db_updator_group3
	rb_test_group2
	publish_event_gi_consumer_group
	qa_kg_url_expand_group
	rb_test_group1
	original_page_entity_HDFS_pusher_group1
	hdfs_publish_event_es_consumer_group2

	[work@nj03-bdg-kg-offline-01.nj03 kafka]$vim  bin/kafka-consumer-groups.sh --bootstrap-server localhost:8092  --group db_updator_group3 --describe
	GROUP                          TOPIC                          PARTITION  CURRENT-OFFSET  LOG-END-OFFSET  LAG             OWNER
	db_updator_group3              kg_url_expand                  12         36950434        36950434        0               kafka-python-1.3.1_/10.194.163.51
	db_updator_group3              kg_url_expand                  7          36943758        36943758        0               kafka-python-1.3.1_/10.194.163.51
	db_updator_group3              kg_url_expand                  6          36937171        36937171        0               kafka-python-1.3.1_/10.194.164.14
	db_updator_group3              kg_url_expand                  9          36942941        36942941        0               kafka-python-1.3.1_/10.194.164.14
	db_updator_group3              kg_url_expand                  4          36950408        36950408        0               kafka-python-1.3.1_/10.194.164.14
	db_updator_group3              kg_url_expand                  3          36945975        36945975        0               kafka-python-1.3.1_/10.194.163.51
	db_updator_group3              kg_url_expand                  0          36951029        36951029        0               kafka-python-1.3.1_/10.194.163.51
	db_updator_group3              kg_url_expand                  10         36948368        36948368        0               kafka-python-1.3.1_/10.194.163.51
	db_updator_group3              kg_url_expand                  5          36950323        36950323        0               kafka-python-1.3.1_/10.194.164.14
	db_updator_group3              kg_url_expand                  13         36930054        36930054        0               kafka-python-1.3.1_/10.194.163.51
	db_updator_group3              kg_url_expand                  2          36949551        36949551        0               kafka-python-1.3.1_/10.194.164.14
	db_updator_group3              kg_url_expand                  8          36946116        36946116        0               kafka-python-1.3.1_/10.194.163.51
	db_updator_group3              kg_url_expand                  14         36930610        36930610        0               kafka-python-1.3.1_/10.194.164.14
	db_updator_group3              kg_url_expand                  1          36935997        36935997        0               kafka-python-1.3.1_/10.194.164.14
	db_updator_group3              kg_url_expand                  11         36943450        36943450        0               kafka-python-1.3.1_/10.194.164.14


