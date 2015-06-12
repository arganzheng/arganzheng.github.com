---
title: 安装OpenTSDB
layout: post
---

安装HBase
---------

OpenTSDB依赖于HBase作为底层存储，所以需要先安装HBase。

因为是Java编写的，安装其实非常简单：
	
	mkdir -p ~/tools/hd
	cd ~/tools/hd
	wget http://www.apache.org/dist/hbase/stable/hbase-1.0.1.1-bin.tar.gz
	tar xfz hbase-1.0.1.1-bin.tar.gz
	ln -s hbase-1.0.1.1 hbase
	cd hbase

然后需要配置一下，主要是下面两个文件：

* hbase-env.sh: 主要是设置环境变量，类似于tomcat的setenv.sh
* hbase-site.xml: HBase的主要配置项

其中hbase-env.sh主要需要修改：

	export JAVA_HOME=/System/Library/Frameworks/JavaVM.framework/Versions/CurrentJDK/Home
	export HBASE_OPTS="-Djava.net.preferIPv4Stack=true -Djava.security.krb5.realm=OX.AC.UK -Djava.security.krb5.kdc=kdc0.ox.ac.uk:kdc1.ox.ac.uk"

hbase-site.xml主要是hbase.rootdir：

	<configuration>
		<property>
			<name>hbase.rootdir</name>
	    	<value>file:///Users/argan/tools/hd/hbasedata</value>
		</property>
	</configuration>

**说明**

1. hbase在完全分布式模式下底层是依赖于hdfs文件系统(hdfs://)，但是在测试环境(standalone和伪分布模式)可以简单使用本地文件系统(file:///)。
2. Hbase要求Zookeper. 默认Hbase自带了一个embedded instance of Zookeeper, 可以避免安装zookeeper。
3. OpenTSDB建议单机安装的时候配置interface为loopback interface，可以简化一些测试问题[Setup HBase](http://opentsdb.net/setup-hbase.html)。
4. OpenTSDB建议Stay on a single node unless you can deploy HBase on at least 5 machines, preferably at least 10.

OK，配置完成之后可以启动HBase了：

	bin/start-hbase.sh


现在让我们测试一下。可以使用HBase shell测试：

	arganzhengs-MacBook-Pro:hbase argan$ bin/hbase shell
	2015-06-10 14:40:38,363 WARN  [main] util.NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
	HBase Shell; enter 'help<RETURN>' for list of supported commands.
	Type "exit<RETURN>" to leave the HBase Shell
	Version 1.0.1.1, re1dbf4df30d214fca14908df71d038081577ea46, Sun May 17 12:34:26 PDT 2015

	hbase(main):001:0> create 'test', 'cf'
	0 row(s) in 0.6900 seconds

	=> Hbase::Table - test
	hbase(main):003:0> list 'test'
	TABLE
	test
	1 row(s) in 0.0280 seconds

	=> ["test"]
	hbase(main):004:0> put 'test', 'row1', 'cf:a', 'value1'
	0 row(s) in 0.1090 seconds

	hbase(main):005:0> put 'test', 'row2', 'cf:b', 'value2'
	0 row(s) in 0.0080 seconds

	hbase(main):006:0> put 'test', 'row3', 'cf:c', 'value3'
	0 row(s) in 0.0050 seconds

	hbase(main):007:0> scan 'test'
	ROW                                    COLUMN+CELL
	 row1                                  column=cf:a, timestamp=1433919775161, value=value1
	 row2                                  column=cf:b, timestamp=1433919785792, value=value2
	 row3                                  column=cf:c, timestamp=1433919793798, value=value3
	3 row(s) in 0.0260 seconds

	hbase(main):008:0> get 'test', 'row1'
	COLUMN                                 CELL
	 cf:a                                  timestamp=1433919775161, value=value1
	1 row(s) in 0.0200 seconds

	hbase(main):009:0> disable
	disable        disable_all    disable_peer
	hbase(main):009:0> disable 'test'
	0 row(s) in 1.2050 seconds

	hbase(main):011:0> drop 'test'
	0 row(s) in 0.1880 seconds

	hbase(main):013:0> exit
	arganzhengs-MacBook-Pro:hbase argan$

停止HBase:

	/bin/stop-hbase.sh

HBase还提供了一个WebUI:

> HBase also puts up a UI listing vital attributes. By default it’s deployed on the Master host at port 16010 (HBase RegionServers listen on port 16020 by default and put up an informational HTTP server at port 16030). If the Master is running on a host named master.example.org on the default port, point your browser at http://master.example.org:16010 to see the web interface.
> 
> Prior to HBase 0.98 the master UI was deployed on port 60010, and the HBase RegionServers UI on port 60030.

对于我们这个是 http://localhost:16010。


安装OpenTSDB
-----------

[Installation](http://opentsdb.net/docs/build/html/installation.html)


首先确定一下Zookeeper是可以访问的：

	arganzhengs-MacBook-Pro:logs argan$ telnet localhost 2181
	Trying ::1...
	telnet: connect to address ::1: Connection refused
	Trying 127.0.0.1...
	Connected to localhost.
	Escape character is '^]'.
	stats
	Zookeeper version: 3.4.6-1569965, built on 02/20/2014 09:09 GMT
	Clients:
	 /127.0.0.1:53406[1](queued=0,recved=787,sent=787)
	 /127.0.0.1:53403[1](queued=0,recved=855,sent=856)
	 /127.0.0.1:53408[1](queued=0,recved=808,sent=808)
	 /127.0.0.1:53410[1](queued=0,recved=786,sent=786)
	 /127.0.0.1:54239[0](queued=0,recved=1,sent=0)
	 /127.0.0.1:53401[1](queued=0,recved=1517,sent=1543)
	 /127.0.0.1:53405[1](queued=0,recved=789,sent=789)
	 /127.0.0.1:54071[1](queued=0,recved=70,sent=70)
	 /127.0.0.1:53407[1](queued=0,recved=787,sent=787)

	Latency min/avg/max: 0/0/15
	Received: 6536
	Sent: 6562
	Connections: 9
	Outstanding: 0
	Zxid: 0x63
	Mode: standalone
	Node count: 36
	Connection closed by foreign host.
	arganzhengs-MacBook-Pro:logs argan$

我们选择源代码安装，需要做好心理准备，非常非常久。。

	git clone git://github.com/OpenTSDB/opentsdb.git
	cd opentsdb
	./build.sh

后来发现其实是因为OpenTSDB在build时候去maven中央仓库拉取第三方jar包，它不是通过标准的pom.xml指定依赖，所以即使你有本地setting.xml也没有生效。解决方案是修改maven地址：/Users/argan/tools/opentsdb/third_party/的所有include.mk文件中。

build成功之后可以将其make install到系统目录，不过这个是可选的。

然后就要建表：

	env COMPRESSION=NONE HBASE_HOME=/Users/argan/tools/hd/hbase ./src/create_table.sh

发现失败了，报错是连接不上Zookeeper和Hbase。telnet一下Zookeeper的端口，发现果然连接不上，hbase也挂了，看一下HBase的日志，是如下错误信息：

	2015-06-10 19:39:28,207 INFO  [RS:0;localhost:53402] util.RetryCounter: Sleeping 4000ms before retry #2...
	2015-06-10 19:39:31,960 WARN  [M:0;localhost:53400] zookeeper.RecoverableZooKeeper: Possibly transient ZooKeeper, quorum=localhost:2181, exception=org.apache.zookeeper.KeeperException$SessionExpiredException: KeeperErrorCode = Session expired for /hbase/rs/localhost,53400,1433918361361
	2015-06-10 19:39:31,960 ERROR [M:0;localhost:53400] zookeeper.RecoverableZooKeeper: ZooKeeper delete failed after 4 attempts
	2015-06-10 19:39:31,960 WARN  [M:0;localhost:53400] regionserver.HRegionServer: Failed deleting my ephemeral node
	org.apache.zookeeper.KeeperException$SessionExpiredException: KeeperErrorCode = Session expired for /hbase/rs/localhost,53400,1433918361361
	        at org.apache.zookeeper.KeeperException.create(KeeperException.java:127)
	        at org.apache.zookeeper.KeeperException.create(KeeperException.java:51)
	        at org.apache.zookeeper.ZooKeeper.delete(ZooKeeper.java:873)
	        at org.apache.hadoop.hbase.zookeeper.RecoverableZooKeeper.delete(RecoverableZooKeeper.java:179)
	        at org.apache.hadoop.hbase.zookeeper.ZKUtil.deleteNode(ZKUtil.java:1288)
	        at org.apache.hadoop.hbase.zookeeper.ZKUtil.deleteNode(ZKUtil.java:1277)
	        at org.apache.hadoop.hbase.regionserver.HRegionServer.deleteMyEphemeralNode(HRegionServer.java:1316)
	        at org.apache.hadoop.hbase.regionserver.HRegionServer.run(HRegionServer.java:1007)
	        at org.apache.hadoop.hbase.master.HMasterCommandLine$LocalHMaster.run(HMasterCommandLine.java:281)
	        at java.lang.Thread.run(Thread.java:745)
	2015-06-10 19:39:31,961 INFO  [M:0;localhost:53400] regionserver.HRegionServer: stopping server localhost,53400,1433918361361; zookeeper connection closed.
	2015-06-10 19:39:31,961 INFO  [M:0;localhost:53400] regionserver.HRegionServer: M:0;localhost:53400 exiting
	2015-06-10 19:39:31,961 INFO  [NIOServerCxn.Factory:0.0.0.0/0.0.0.0:2181] server.NIOServerCnxnFactory: NIOServerCnxn factory exited run method
	2015-06-10 19:39:31,962 INFO  [M:0;localhost:53400] server.ZooKeeperServer: shutting down
	2015-06-10 19:39:31,962 INFO  [M:0;localhost:53400] server.SessionTrackerImpl: Shutting down
	2015-06-10 19:39:31,962 INFO  [M:0;localhost:53400] server.PrepRequestProcessor: Shutting down
	2015-06-10 19:39:31,962 INFO  [ProcessThread(sid:0 cport:-1):] server.PrepRequestProcessor: PrepRequestProcessor exited loop!
	2015-06-10 19:39:31,962 INFO  [M:0;localhost:53400] server.SyncRequestProcessor: Shutting down
	2015-06-10 19:39:31,962 INFO  [SyncThread:0] server.SyncRequestProcessor: SyncRequestProcessor exited!
	2015-06-10 19:39:31,962 INFO  [M:0;localhost:53400] server.FinalRequestProcessor: shutdown of request processor complete
	2015-06-10 19:39:31,964 INFO  [M:0;localhost:53400] zookeeper.MiniZooKeeperCluster: Shutdown MiniZK cluster with all ZK servers
	2015-06-10 19:39:32,000 INFO  [SessionTracker] server.SessionTrackerImpl: SessionTrackerImpl exited loop!
	2015-06-10 19:39:32,210 WARN  [RS:0;localhost:53402] zookeeper.RecoverableZooKeeper: Possibly transient ZooKeeper, quorum=localhost:2181, exception=org.apache.zookeeper.KeeperException$SessionExpiredException: KeeperErrorCode = Session expired for /hbase/rs/localhost,53402,1433918362840
	2015-06-10 19:39:32,210 INFO  [RS:0;localhost:53402] util.RetryCounter: Sleeping 8000ms before retry #3...
	2015-06-10 19:39:40,215 WARN  [RS:0;localhost:53402] zookeeper.RecoverableZooKeeper: Possibly transient ZooKeeper, quorum=localhost:2181, exception=org.apache.zookeeper.KeeperException$SessionExpiredException: KeeperErrorCode = Session expired for /hbase/rs/localhost,53402,1433918362840
	2015-06-10 19:39:40,216 ERROR [RS:0;localhost:53402] zookeeper.RecoverableZooKeeper: ZooKeeper delete failed after 4 attempts
	2015-06-10 19:39:40,216 WARN  [RS:0;localhost:53402] regionserver.HRegionServer: Failed deleting my ephemeral node
	org.apache.zookeeper.KeeperException$SessionExpiredException: KeeperErrorCode = Session expired for /hbase/rs/localhost,53402,1433918362840
	        at org.apache.zookeeper.KeeperException.create(KeeperException.java:127)
	        at org.apache.zookeeper.KeeperException.create(KeeperException.java:51)
	        at org.apache.zookeeper.ZooKeeper.delete(ZooKeeper.java:873)
	        at org.apache.hadoop.hbase.zookeeper.RecoverableZooKeeper.delete(RecoverableZooKeeper.java:179)
	        at org.apache.hadoop.hbase.zookeeper.ZKUtil.deleteNode(ZKUtil.java:1288)
	        at org.apache.hadoop.hbase.zookeeper.ZKUtil.deleteNode(ZKUtil.java:1277)
	        at org.apache.hadoop.hbase.regionserver.HRegionServer.deleteMyEphemeralNode(HRegionServer.java:1316)
	        at org.apache.hadoop.hbase.regionserver.HRegionServer.run(HRegionServer.java:1007)
	        at java.lang.Thread.run(Thread.java:745)
	2015-06-10 19:39:40,216 INFO  [RS:0;localhost:53402] regionserver.HRegionServer: stopping server localhost,53402,1433918362840; zookeeper connection closed.
	2015-06-10 19:39:40,216 INFO  [RS:0;localhost:53402] regionserver.HRegionServer: RS:0;localhost:53402 exiting
	2015-06-10 19:39:40,222 INFO  [Thread-5] regionserver.ShutdownHook: Shutdown hook starting; hbase.shutdown.hook=true; fsShutdownHook=org.apache.hadoop.fs.FileSystem$Cache$ClientFinalizer@1a75e76a
	2015-06-10 19:39:40,222 INFO  [Thread-5] regionserver.ShutdownHook: Starting fs shutdown hook thread.
	2015-06-10 19:39:40,223 INFO  [Thread-5] regionserver.ShutdownHook: Shutdown hook finished.


出错之后自动shutdown了。。重启之后再执行就可以了：

	arganzhengs-MacBook-Pro:opentsdb argan$ env COMPRESSION=NONE HBASE_HOME=/Users/argan/tools/hd/hbase ./src/create_table.sh
	2015-06-10 20:48:26,002 WARN  [main] util.NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
	HBase Shell; enter 'help<RETURN>' for list of supported commands.
	Type "exit<RETURN>" to leave the HBase Shell
	Version 1.0.1.1, re1dbf4df30d214fca14908df71d038081577ea46, Sun May 17 12:34:26 PDT 2015

	create 'tsdb-uid',
	  {NAME => 'id', COMPRESSION => 'NONE', BLOOMFILTER => 'ROW'},
	  {NAME => 'name', COMPRESSION => 'NONE', BLOOMFILTER => 'ROW'}
	0 row(s) in 0.5970 seconds

	Hbase::Table - tsdb-uid

	create 'tsdb',
	  {NAME => 't', VERSIONS => 1, COMPRESSION => 'NONE', BLOOMFILTER => 'ROW'}
	0 row(s) in 0.1520 seconds

	Hbase::Table - tsdb

	create 'tsdb-tree',
	  {NAME => 't', VERSIONS => 1, COMPRESSION => 'NONE', BLOOMFILTER => 'ROW'}
	0 row(s) in 0.1610 seconds

	Hbase::Table - tsdb-tree

	create 'tsdb-meta',
	  {NAME => 'name', COMPRESSION => 'NONE', BLOOMFILTER => 'ROW'}
	0 row(s) in 0.1540 seconds

	Hbase::Table - tsdb-meta


OK，现在我们可以准备启动TSD了。不过在这之前，我们需要配置一下OpenTSDB先[configuration](http://opentsdb.net/docs/build/html/user_guide/configuration.html)：

将 src/opentsdb.conf 拷贝到如下目录之一：

* ./opentsdb.conf
* /etc/opentsdb.conf
* /etc/opentsdb/opentsdb.conf
* /opt/opentsdb/opentsdb.conf

然后配置如下四个必须配置项：

* tsd.network.port=8242
* tsd.http.cachedir=/tmp/tsd - Path to write temporary files to
* tsd.http.staticroot=build/staticroot - Path to the static GUI files found in ./build/staticroot
* tsd.storage.hbase.zk_quorum=localhost - A comma separated list of Zookeeper hosts to connect to, default is "localhost". If HBase and Zookeeper are not running on the same machine, specify the host and port here.
* tsd.core.auto_create_metrics=True - Whether or not to automatically create UIDs for new metric types, default is False. 建议打开。

然后就可以简单启动TSD了：

	./build/tsdb tsd

当然，你也可以通过命令行指定（会覆盖opentsdb.conf的配置）：

	Usage: tsd --port=PORT --staticroot=PATH --cachedir=PATH
	Starts the TSD, the Time Series Daemon
	  --async-io=true|false Use async NIO (default true) or traditional blocking io
	  --auto-metric         Automatically add metrics to tsdb as they are inserted.  Warning: this may cause unexpected metrics to be tracked
	  --backlog=NUM         Size of connection attempt queue (default: 3072 or kernel somaxconn.
	  --bind=ADDR           Address to bind to (default: 0.0.0.0).
	  --cachedir=PATH       Directory under which to cache result of requests.
	  --config=PATH         Path to a configuration file (default: Searches for file see docs).
	  --flush-interval=MSEC Maximum time for which a new data point can be buffered (default: 1000).
	  --port=NUM            TCP port to listen on.
	  --staticroot=PATH     Web root from which to serve static files (/s URLs).
	  --table=TABLE         Name of the HBase table where to store the time series (default: tsdb).
	  --uidtable=TABLE      Name of the HBase table to use for Unique IDs (default: tsdb-uid).
	  --worker-threads=NUM  Number for async io workers (default: cpu * 2).
	  --zkbasedir=PATH      Path under which is the znode for the -ROOT- region (default: /hbase).
	  --zkquorum=SPEC       Specification of the ZooKeeper quorum to use (default: localhost).

	tsdtmp=${TMPDIR-'/tmp'}/tsd    # For best performance, make sure
	mkdir -p "$tsdtmp"             # your temporary directory uses tmpfs
	./build/tsdb tsd --port=8242 --staticroot=build/staticroot --cachedir="$tsdtmp" --zkquorum=myhost:2181

然后就可以通过 http://127.0.0.1:8242 访问TSD的web界面了。不过话说，比起influxDB来说，这个界面还真是挺粗糙的。

OpenTSDB没有像influxDB一样，可以通过WebUI插入数据，不过它支持telnet，也是挺方便的：

	bogon:~ argan$ telnet localhost 8242
	Trying ::1...
	Connected to localhost.
	Escape character is '^]'.
	put sys.cpu.user 1356998400 42.5 host=webserver01 cpu=0
	put: unknown metric: No such name for 'metrics': 'sys.cpu.user'

发现报错了，谷歌了一下是需要开启一个配置项：

	# Whether or not to automatically create UIDs for new metric types, default is False
	tsd.core.auto_create_metrics = True

或者启动项开启：

	./tsdb tsd --auto-metric


重启就可以了。也支持[HTTP接口](http://opentsdb.net/docs/build/html/api_http/put.html)：

	bogon:~ argan$ curl -i  -H "Content-Type: application/json" -X POST -d '{"metric": "sys.cpu.nice", "timestamp": 1433989867597,"value": 18, "tags": { "host": "web01"}}' http://localhost:8242/api/put/?details
	HTTP/1.1 200 OK
	Content-Type: application/json; charset=UTF-8
	Content-Length: 36

**TIPS** 

如果build的时候遇到

	Can't exec "aclocal": No such file or directory at /usr/local/Cellar/autoconf/2.69/share/autoconf/Autom4te/FileUtils.pm line 326.
	autoreconf: failed to run aclocal: No such file or directory

是因为缺少automake，安装一下就可以了：

	$ brew install automake

	sudo make install
	> env COMPRESSION=SNAPPY HBASE_HOME=/opt/cloudera/parcels/CDH/lib/hbase ./src/create_table.sh
	> hbase shell
	> list


**NOTES**

1. Make sure you have LZO installed and make sure it's enabled for the tables used by OpenTSDB.

参考文档
-------

1. [Setup HBase](http://opentsdb.net/setup-hbase.html) OpenTSDB的HBase安装文档，比HBase官方文档要清晰的多。。
2. [Hadoop & Hbase on OSX 10.8 Mountain Lion](http://freddy.cellcore.org/post/52568231952/hadoop-hbase-on-osx-10-8-mountain-lion) 非常详细的文章，mac推荐。
3. [First Time setting up Graphana & OpenTSDB with CDH5](http://www.geovanie.me/graphana-opentsdb-and-cdh5/)
4. [OpenTSDB部署手记](http://debugo.com/opentsdb/)

