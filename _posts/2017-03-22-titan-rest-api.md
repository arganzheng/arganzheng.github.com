---
title: Titan如何提供REST服务
layout: post
---

默认Titan是作为一个jar包依赖，本地调用。但是通过[Titan Server](http://s3.thinkaurelius.com/docs/titan/1.0.0/server.html)可以提供远程服务。

Titan使用了[Gremlin Server](http://tinkerpop.incubator.apache.org/docs/3.0.1-incubating/#gremlin-server)引擎作为服务器来提供服务，默认配置是使用WebSockets，但是我们可以配置成其他协议或者自己扩展实现。

下面是如何配置Titan Server，让其提供HTTP REST API：

1、If you're starting from the titan-1.0.0-hadoop1.zip, I'd suggest making a copy of conf/gremlin-server/gremlin-server.yaml into conf/rest-gremlin-server.yaml. Also, copy conf/titan-cassandra.properties to conf/gremlin-server/titan-cassandra-server.properties

	cp conf/gremlin-server/gremlin-server.yaml conf/rest-gremlin-server.yaml
	cp conf/titan-cassandra.properties conf/gremlin-server/titan-cassandra-server.properties

2、If you're planning to connect to Gremlin Server from something other than localhost, you'll want to update the host with its IP address or 0.0.0.0 in rest-gremlin-server.yaml

	host: 0.0.0.0

3、Configure the Gremlin Server for REST instead of WebSockets (refer to the Gremlin Server documentation). In the rest-gremlin-server.yaml, update the channelizer

	channelizer: org.apache.tinkerpop.gremlin.server.channel.HttpChannelizer

4、Configure the Gremlin Server mapping with the Cassandra properties file with your graph configuration

	graphs: {
	  graph: conf/gremlin-server/titan-cassandra-server.properties}

And also, be sure to start Gremlin Server with the configuration file：

	./bin/gremlin-server.sh ./conf/gremlin-server/rest-gremlin-server.yaml

5、Update the titan-cassandra-server.properties with your specific Cassandra properties. In particular, you'll want to update the keyspace name and possibly storage.hostname if Cassandra isn't running on localhost.

	storage.hostname=127.0.0.1
	storage.cassandra.keyspace=TITAN_DEMO

6、Sending Gremlin to the Gremlin Server configured for REST is the same, regardless of the actual Gremlin query. Do a HTTP POST of a JSON object with a pair { "gremlin": "g.V().count()" } to the Gremlin Server. Here's what it looks like with a curl:

	curl -XPOST -Hcontent-type:application/json -d '{"gremlin":"g.V().count()"}' http://localhost:8182



参考文章
-------

1. [Gremlin Server](http://tinkerpop.incubator.apache.org/docs/3.0.1-incubating/#gremlin-server)
2. [Gremlin Server Rest API](https://groups.google.com/forum/#!msg/aureliusgraphs/v4UcYwE5UVU/vRa_xvD7CAAJ)



