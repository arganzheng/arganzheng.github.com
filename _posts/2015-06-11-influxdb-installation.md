---
title: InfluxDB安装和使用
layout: post
---

influxDB v0.9.0 处于 pre-release 阶段，在底层存储、集群和数据模型和接口都与之前的版本有所不同，并且不兼容的。所以安装使用之前需要先确定版本。一般来说当然是0.9.0更好，但是鉴于现在处于不稳定阶段，所以我们还是采用了v0.8.8版本。


influxDB官网有一个[在线体验网站](http://play.influxdb.org/)，可以上去体验一下。

mac下使用homebrew安装本来是一件非常简单的事情：

	brew update
	brew install influxdb

如果想要安装0.9.0版本，可以指定--devel选项：

	brew install influxdb --devel

但是由于有GWF的存在，然后influxDB依赖于golang模块，这个模块是存储在 https://storage.googleapis.com/golang/go1.4.2.src.tar.gz 上，所以悲剧了，一直下不来。
百度了一下，有很多人遇到类似的问题，解决方案是先通过其他工具将所需的软件包下载下来，注意版本一定要对应，然后放到/Library/Caches/Homebrew/目录，再使用brew install xxx的时候，brew就能直接安装了，注意软件包的命名。但是发现还是不行，报SHA1 mismatch错误。谷歌了一下，说可以手动修改一下SHA1在`/usr/local/Library/Formula/go.rb`文件中。

	class Go < Formula
	  desc "Go programming environment"
	  homepage "https://golang.org"
	  # Version 1.5 is going to require version 1.4 present to bootstrap 1.4
	  # Perhaps we can use our previous bottles, ala the discussion around PyPy?
	  # https://docs.google.com/document/d/1OaatvGhEAq7VseQ9kkavxKNAfepWy2yhPUBs96FGV28
	  url "https://storage.googleapis.com/golang/go1.4.2.src.tar.gz"
	  sha1 "460caac03379f746c473814a65223397e9c9a2f6"
	  version "1.4.2"

	  head "https://go.googlesource.com/go", :using => :git

	  bottle do
	    revision 1
	    sha1 "b3ec148a548331c3fd75435b7aa6ae2378ce995e" => :yosemite
	    sha1 "a4ea2ffdd9db813c870b0ce73c011788ac60cb51" => :mavericks
	    sha1 "bc52571c43f59f92ca461ff310693501f2419a04" => :mountain_lion
  end
  ...

改成下载tar.gz包的SHA1值之后，再brew install go，会报`/usr/local/Cellar/go/1.4.2` is not a directory。手动创建这个目录，然后将/usr/local/Cellar/go下的所有文件移动到/usr/local/Cellar/go/1.4.2下就可以了：

	bogon:~ argan$ brew install go
	Warning: go-1.4.2 already installed, it's just not linked
	bogon:~ argan$ brew install influxdb
	Error: You must `brew link go' before influxdb can be installed
	bogon:~ argan$ brew link go
	Linking /usr/local/Cellar/go/1.4.2... 10 symlinks created

各位看官，本来一件一个命令搞定的事情，由于伟大的GFW的存在，废了多少时间和脑细胞才搞定。。做一个中国程序员真心心塞啊。。

后来发现其实go并没有成功安装。所以还是手动安装吧。。Download, compile and install Go:

	$ cd 
	$ wget https://storage.googleapis.com/golang/go1.4.2.src.tar.gz
	$ tar -xvf go1.4.2.src.tar.gz
	$ cd go/src
	$ ./all.bash # install Golang
	...
	ALL TESTS PASSED

	---
	Installed Go for darwin/amd64 in /Users/argan/go
	Installed commands in /Users/argan/go/bin
	*** You need to add /Users/argan/go/bin to your PATH.

	vim ~/.bash_profile
	export GOROOT=$HOME/go
	export GOPATH=$HOME/gopath
	PATH=$PATH:$GOROOT/bin
	$ source ~/.bash_profile

即使成功安装了golang，执行

	brew install influxdb

还是报错。把tar包下载下来遇到跟golang一样的问题。。。难道要手动编译influxDB。。但是0.8.8版本的influxdb是混合的，谷歌了一下，0.9之后是纯go的，可以很简单的用go编译安装。。[How to compile InfluxDB from source](https://anomaly.io/compile-influxdb/)。Download and compile latest InfluxDB。结果还是报错了。。

	bogon:~ argan$ go get github.com/influxdb/influxdb

结果还是出错了。。。

	bogon:~ argan$ go get github.com/influxdb/influxdb
	^@^@^@^@package github.com/influxdb/influxdb
	imports github.com/gogo/protobuf/proto
	imports github.com/hashicorp/raft
	imports github.com/armon/go-metrics
	imports github.com/hashicorp/go-msgpack/codec
	imports github.com/hashicorp/raft-boltdb
	imports github.com/boltdb/bolt
	imports golang.org/x/crypto/bcrypt: unrecognized import path "golang.org/x/crypto/bcrypt"

我想一个人静静。。
然后就可以简单的启动了：

	$GOPATH/bin/influxd -config=/usr/local/etc/influxdb.conf

--- FAIL: TestDialTimeout (0.00s)
	dial_test.go:108: got error "dial tcp 127.0.0.1:34119: connection reset by peer"; not a timeout


**一些注意事项**

1、使用端口

* 8083: UI
* 8086: API, Send and query data
* 8090: Cluster management raft
* 8099: Cluster management protobuf

你可以修改配置文件使用其他端口，配置文件路径一般在：/opt/influxdb/shared/config.toml 或者 /usr/local/etc/influxdb.conf。

2、上报

从0.7.1开始，InfluxDB会默认每24小时上报服务情况到m.influxdb.com，以便他们统计使用情况。你可以通过`reporting-disabled = true` 关闭。 


3、File Limits

InfluxDB可能会打开很多文件，一般来说需要修改文件上限。具体可以参考[Open Files Limit](http://docs.basho.com/riak/latest/ops/tuning/open-files-limit/)。



安装成功之后，让我们测试一下。influxDB有个WebUI可以在上面直接创建DB和插入数据。默认是http://localhost:8083。数据库进程默认是8086。也可以通过HTTP接口写入数据：

	$ curl -X POST -d '[{"name": "hd_used", "columns" : ["value", "host", "mount"], "points" : [23.2, "serverA", "/mnt"]}]' http://localhost:8086/db/<database>/series?u=<user>&p=<pass>

**TIPS** 指定时间和序列号

InfluxDB对写入的每个data point会默认分配系统当前时间和一个序列号。但是如果你发送的是历史数据，那么这个时间就会不对。你可以在写入的时候指定：

	[
	  {
	    "name": "log_lines",
	    "columns": ["time", "sequence_number", "line"],
	    "points": [
	      [1400425947368, 1, "this line is first"],
	      [1400425947368, 2, "and this is second"]
	    ]
	  }
	]

注意：InfluxDB默认的时间精度是microsecond(u)，你可以在POST的时候指定：s for seconds, ms for milliseconds, u for microseconds. 查询的时候则通过`time_precision`查询参数指定。

然后我们可以通过HTTP接口查询：

	curl -G 'http://localhost:8086/db/mydb/series?u=root&p=root&pretty=true' --data-urlencode "q=select * from log_lines"


[sharding_and_storage](http://influxdb.com/docs/v0.8/advanced_topics/sharding_and_storage.html)
----------------------

### Storage Engines

InfluxDB 0.8版本支持多种存储引擎。包括LevelDB, RocksDB, HyperLevelDB和LMDB。前三种其实都是LSM结构，LMDB则是一个memory-mapped Copy on Write B+Tree。默认是RocksDB。0.9之后则只支持BoltDB一种存储引擎。

### Databases和Shard Spaces

Data in InfluxDB is organized into databases which have many shard spaces which have many shards. A shard maps to an underlying storage engine database. That is, each shard will be a separate LevelDB or LMDB. The implications of this are that if you want to keep your underlying storage engine databases small, configure things so your data will be split across many shards.

Shard spaces have the following properties:

	{
	  "name": "high_precision",
	  "database": "pauls_db",
	  "retentionPolicy": "7d",
	  "shardDuration": "1d",
	  "regex": "/^[a-z].*/",
	  "replicationFactor": 1,
	  "split": 1
	}

可以看到一个shard space归属于某个特定的database。

The retentionPolicy is the period of time that data will be kept around. The exact semantics are that data is kept at least that long. The amount of time it is kept after that is determined by the shardDuration.

The replicationFactor setting tells the InfluxDB cluster how many servers should have a copy of each shard in the given shard space. Finally, split tells the cluster how many shards to create for a given interval of time. Data for that interval will be distributed across the shards. This setting is how you achieve write scalability. You may want to have replicationFactor * split == number of servers. That will ensure that every server in the cluster will be hot for writes at any given time.

If you want high availability, you'll need at least 3 nodes running. You can scale up anytime as your data needs grow.
Replication factor gives you more scalability on queries and more robust fault tolerance. For most use cases, 2 or 3 should be enough.


参考文档
-------

1. [Installation](http://influxdb.com/docs/v0.8/introduction/installation.html)  