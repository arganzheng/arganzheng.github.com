---
title: 使用logstash收集nginx访问日志
layout: post
---


首先安装logstash，这个非常简单，不赘述。建议把所有插件都安装上，省心。


然后要配置一下logstash for nginx。logstash基本原理：input => filter => output。在我们这里input就是nginx的access日志，output就是ElasticSearch。filter则是用来解析和过滤日志用。一般我们要把message结构化再存储，方便后面的搜索和统计。因此需要对message进行解析。logstash是使用grok过滤器，使用match正则表达式解析。要根据自己的log_format来定制。

比如这里我们的log_format配置是：

	log_format  main      '$http_host '
	                      '$remote_addr - $remote_user [$time_local] '
	                      '"$request" $status $body_bytes_sent "$request_body" '
	                      '"$http_referer" "$http_user_agent" "$http_x_forwarded_for" '
	                      '$request_time '  
	                      '$upstream_response_time';

那么相应的grok就是：

	%{IPORHOST:http_host} %{IPORHOST:remote_addr} - %{USERNAME:remote_user} \[%{HTTPDATE:time_local}\] "%{WORD:method} %{URIPATHPARAM:request} HTTP/%{NUMBER:http_version}" %{INT:status} %{INT:body_bytes_sent} %{QS:request_body} %{QS:http_referer} %{QS:http_user_agent} %{QS:http_x_forwarded_for} %{NUMBER:request_time:float} %{NUMBER:upstream_response_time:float}

可以用`logstash.study.arganzheng.me 95.159.86.100 - - [18/Apr/2016:11:38:51 +0800] "GET /ad.php?p=com.adslib.mobovee&hp=com.pay.m.KOF95&l=d784ee59431b73536284463336b26c03&c=mobojoy HTTP/1.1" 200 10 "-" "-" "Dalvik/2.1.0 (Linux; U; Android 5.0.2; SM-T810 Build/LRX22G)" "-" 0.002 0.002`在Grok Debugger测试一下，或者直接用`input { stdin { } }`测试。

可以把这个Grok抽取表达式作为一个pattern固定下来：
	
	$ cd logstash
	$ mkdir patterns # 可以在grok中通过 patterns_dir 指定，默认是这个位置
	$ vim /home/work/logstash/patterns/nginx

添加如下一行：

	NGINXACCESS %{IPORHOST:http_host} %{IPORHOST:remote_addr} - %{USERNAME:remote_user} \[%{HTTPDATE:time_local}\] "%{WORD:method} %{URIPATHPARAM:request} HTTP/%{NUMBER:http_version}" %{INT:status} %{INT:body_bytes_sent} %{QS:request_body} %{QS:http_referer} %{QS:http_user_agent} %{QS:http_x_forwarded_for} %{NUMBER:request_time:float} %{NUMBER:upstream_response_time:float}

**TIPS** 

1. logstash默认定义了一些 [grok pattern](https://github.com/elastic/logstash/blob/v1.4.2/patterns/grok-patterns)，一般来说对于字符串，有双引号包含的用QS，没有的用DATA类型，如%{DATA:request_body}。
2. grok匹配很容易出错，可以使用[Grok Debugger](https://grokdebug.herokuapp.com/)进行在线调试。	


综上，整个logstash.conf配置文件如下：

	## input { stdin { } }
	input {
		file {
			path => ["/home/work/nginx/logs/*.log"]
	    	exclude => ["*.gz"]
	    	# start_position => "beginning"
		}
	}

	## nginx log format config
	# log_format  main     '$http_host '
	#                      '$remote_addr - $remote_user [$time_local] '
	#                      '"$request" $status $body_bytes_sent "$request_body" '
	#                      '"$http_referer" "$http_user_agent" "$http_x_forwarded_for" '
	#                      '$request_time $upstream_response_time';

	filter {
	  	grok {
	  		# patterns_dir => [ "/home/work/logstash/patterns" ]
	       	match => { "message" => "%{NGINXACCESS}" }
	    }
	  	date {
	    	match => [ "time_local" , "dd/MMM/yyyy:HH:mm:ss Z" ]
	  	}
	  	geoip {
	        source => "remote_addr"
	    }
	}

	output {
	  elasticsearch {
	        hosts => ["10.242.122.23:8200","10.242.99.47:8200","10.242.103.33:8200"]
	        index => "logstash-nginx-%{+YYYY.MM.dd}"
	  }
	  # stdout { codec => rubydebug }
	}


**TIPS**

1、如果想要让URL参数也解析并且成为索引字段，比如一些通用参数，如uid, country, language, etc. 那么可以使用[KV插件](https://www.elastic.co/guide/en/logstash/current/plugins-filters-kv.html)：

	filter {

		grok {
			...
		}
		
		...

		#再单独将取得的URL、request字段取出来进行key-value值匹配，需要kv插件。提供字段分隔符"&?"，值键分隔符"="，则会自动将字段和值采集出来。
	  	kv {
          	source => "request" # 默认是message，我们这里只需要解析上面grok抽取出来的request字段
          	field_split => "&?"
          	value_split => "="
          	include_keys => [ "uid", "country", "language" ]
      	}
	　
	  	#把所有字段进行urldecode（显示中文）
	  	urldecode {
	     	all_fields => true
	  	}

	}



2、过滤掉安全扫描。logstash可以简单的过滤消息，例如：

	if [message] !~ ".*ERROR.*|.*error.*|.*FATAL.*|.*fatal.*|.*WARN.*|.*warn.*" {
    	drop { }
    }


然后我们就可以启动logstash查看一下效果了:

	$ bin/logstash -f conf/logstash-nginx.conf

会看到logstash已经在扫描nginx日志发送到ES集群了。可以配置一下Kibana，对日志进行分析和可视化。具体参见参考文章。



参考文章
-------

1. [Centralized Logging with Logstash and Kibana On CentOS 7](https://www.digitalocean.com/community/tutorial_series/centralized-logging-with-logstash-and-kibana-on-centos-7) 一个序列教程，比较新。
2. [Adding Logstash Filters To Improve Centralized Logging](https://www.digitalocean.com/community/tutorials/adding-logstash-filters-to-improve-centralized-logging/)
3. [GETTING THE BEST OUT OF LOGSTASH FOR NGINX](https://railsadventures.wordpress.com/2014/07/18/getting-the-best-out-of-logstash-for-nginx/)
4. [How To Map User Location with GeoIP and ELK (Elasticsearch, Logstash, and Kibana)](https://www.digitalocean.com/community/tutorials/how-to-map-user-location-with-geoip-and-elk-elasticsearch-logstash-and-kibana)
5. [Setting up Logstash 1.4.2 to forward Nginx logs to Elasticsearch](http://www.bravo-kernel.com/2014/12/setting-up-logstash-1-4-2-to-forward-nginx-logs-to-elasticsearch/)
6. [grok](https://www.elastic.co/guide/en/logstash/current/plugins-filters-grok.html)
7. [kv](https://www.elastic.co/guide/en/logstash/current/plugins-filters-kv.html)
8. [What Logstash Pitfalls You Need to Avoid](http://logz.io/blog/5-logstash-pitfalls-and-how-to-avoid-them/)
9. [How to Deploy the ELK Stack in Production](http://logz.io/blog/deploy-elk-production/)
