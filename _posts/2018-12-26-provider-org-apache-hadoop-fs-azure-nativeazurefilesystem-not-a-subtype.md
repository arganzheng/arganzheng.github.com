---
title: 创建Hadoop FileSystem报Provider org.apache.hadoop.fs.azure.NativeAzureFileSystem not a subtype异常
layout: post
tags: [Spark, hadoop]
catalog: true
---


为了验证一个线上问题，写了一个简单的Spark任务，在提交节点上听过`spark-submit`命令提交任务，总是跑不起来。

报如下错误：

```shell
18/12/26 16:37:24 ERROR study.Application: error
java.util.ServiceConfigurationError: org.apache.hadoop.fs.FileSystem: Provider org.apache.hadoop.fs.azure.NativeAzureFileSystem not a subtype
        at java.util.ServiceLoader.fail(ServiceLoader.java:239)
        at java.util.ServiceLoader.access$300(ServiceLoader.java:185)
        at java.util.ServiceLoader$LazyIterator.nextService(ServiceLoader.java:376)
        at java.util.ServiceLoader$LazyIterator.next(ServiceLoader.java:404)
        at java.util.ServiceLoader$1.next(ServiceLoader.java:480)
        at org.apache.hadoop.fs.FileSystem.loadFileSystems(FileSystem.java:2400)
        at org.apache.hadoop.fs.FileSystem.getFileSystemClass(FileSystem.java:2411)
        at org.apache.hadoop.fs.FileSystem.createFileSystem(FileSystem.java:2428)
        at org.apache.hadoop.fs.FileSystem.access$200(FileSystem.java:88)
        at org.apache.hadoop.fs.FileSystem$Cache.getInternal(FileSystem.java:2467)
        at org.apache.hadoop.fs.FileSystem$Cache.get(FileSystem.java:2449)
        at org.apache.hadoop.fs.FileSystem.get(FileSystem.java:367)
        at org.apache.hadoop.fs.FileSystem.get(FileSystem.java:166)
        at life.arganzheng.internet.ai.platform.study.FeatureOnlineService.syncByModelMeta(FeatureOnlineService.java:105)
        at life.arganzheng.internet.ai.platform.study.FeatureOnlineService.syncByModelName(FeatureOnlineService.java:80)
        at life.arganzheng.internet.ai.platform.study.Application.run(Application.java:34)
        at life.arganzheng.internet.ai.platform.study.Application.main(Application.java:18)
        at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
        at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62)
        at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
        at java.lang.reflect.Method.invoke(Method.java:497)
        at org.apache.spark.deploy.SparkSubmit$.org$apache$spark$deploy$SparkSubmit$$runMain(SparkSubmit.scala:744)
        at org.apache.spark.deploy.SparkSubmit$.doRunMain$1(SparkSubmit.scala:187)
        at org.apache.spark.deploy.SparkSubmit$.submit(SparkSubmit.scala:212)
        at org.apache.spark.deploy.SparkSubmit$.main(SparkSubmit.scala:126)
        at org.apache.spark.deploy.SparkSubmit.main(SparkSubmit.scala)
```

错误的代码是这一段：

```java
...
// 避免打jar包时SPI文件出现问题
Configuration cfg = new Configuration();
cfg.set("fs.hdfs.impl", "org.apache.hadoop.hdfs.DistributedFileSystem");

String hiveSuccessPath = null;
try {
    // 一个特征集就是一张hive表，需要先检查特征数据是否写OK
    FileSystem fileSystem = FileSystem.get(URI.create("/hive/warehouse/dm_ads_db.db"), cfg);
    for (String featureSetPath : featureSetPathToKeys.keySet()) {
        // hdfs://footstone/hive/warehouse/dm_ads_db.db/feature_app_cn_base/dt=20180913/_SUCCESS
        hiveSuccessPath = featureSetPath + "/dt=" + dateForFeature + "/_SUCCESS";
        boolean success = fileSystem.exists(new Path(hiveSuccessPath));
        if (!success) {
            LOGGER.warn("hdfs file not exist:{}", hiveSuccessPath);
            throw new IllegalArgumentException("feature data is not ready:" + featureSetPath);
        }
    }
} catch (RuntimeException e) {
    throw e;
} catch (IOException e) {
    throw new RuntimeException("check path error:" + hiveSuccessPath, e);
}
...
```

就是这一句`FileSystem fileSystem = FileSystem.get(URI.create("/hive/warehouse/dm_ads_db.db"), cfg);`

改成：

```java
FileSystem fileSystem = FileSystem.get(URI.create("hdfs://footstone/hive/warehouse/dm_ads_db.db"), cfg);
```

或者

```java
FileSystem fileSystem = FileSystem.get(cfg);
```

还是报一样的错误。

谷歌都没有任何结论。只能跟着源码看下去了。

Hadoop对文件系统进行了抽象，HDFS只是其中的一个实现。Java抽象类`org.apache.hadoop.fs.FileSystem`定义了Hadoop中一个文件系统的客户端接口，并且该抽象类允许自定义实现对应的文件系统，运行时通过Java的SPI机制加载相应的实现类。下面是常见的几个具体实现：


| 文件系统   |    URI schema | Java实现（都在org.apache.hadoop包中）  | 描述 |
| :-------- | :--------: | :-------- | :-- |
| Local | file  | fs.LocalFileSystem | 使用客户端校验和的本地磁盘文件系统。使用RawLocalFileSystem表示无校验和的本地磁盘文件系统。 |
| HDFS | hdfs | hdfs.DistributedFileSystem | Hadoop的分布式文件系统。将HDFS设计成与MapReduce结合使用，可以实现高性能。  |
| WebHDFS | webhdfs | Hdfs.web.WebHdfsFileSystem | 基于HTTP的文件系统，提供对HDFS的认证读/写访问。|
| HAR | har| Hdfs.web.WebHdfsFileSystem | WebHDFS的HTTPS版本 |
| Secure WebHDFS | swebhdfs | fs.HarFileSystem | 一个构建在其他文件系统之上用于文件存档的文件系统。Hadoop存档文件系统通常用于将HDFS中的多个文件打包成一个存档文件，以减少namenode内存的使用。使用hadoop的achive命令来创建HAR文件。 |
| View | viewfs| viewfs.ViewFileSystem | 针对其他Hadoop文件系统的客户端挂载表。通常用于为联邦namenode创建挂载点。 |
| FTP | ftp | fs.ftp.FTPFileSystem | 由FTP服务器支持的文件系统 |
| S3 | S3a | fs.s3a.S3AFileSystem | 由Amazon S3 支持的文件系统。替代老版本的s3n（S3）原生 |
| Azure| wasb | fs.azure.NativeAzureFileSystem | 由Microsoft Azure 支持的文件系统 |
| Swift| swift| fs.swift.snative.SwiftNativeFileSystem | 由OpenStack Swift 支持的文件系统 |

SPI通过`java.util.ServiceLoader`类加载`META-INF/services/`目录下的ServiceProvider。于是解压打包的jar包（是一个大的uberjar），发现里面确实有`META-INF/services/org.apache.hadoop.fs.FileSystem`文件，打开一看，里面写了好几个实现类：

```
 org.apache.hadoop.fs.LocalFileSystem
 org.apache.hadoop.fs.viewfs.ViewFileSystem
 org.apache.hadoop.fs.s3.S3FileSystem
 org.apache.hadoop.fs.s3native.NativeS3FileSystem
 org.apache.hadoop.fs.ftp.FTPFileSystem
 org.apache.hadoop.fs.HarFileSystem
```

但是呢，就没有`NativeAzureFileSystem`的声明，类路径也没有这个依赖。

而且报错的代码也很奇怪：

```java
private static void loadFileSystems() {
  synchronized (FileSystem.class) {
    if (!FILE_SYSTEMS_LOADED) {
      ServiceLoader<FileSystem> serviceLoader = ServiceLoader.load(FileSystem.class);
      for (FileSystem fs : serviceLoader) {
        SERVICE_FILE_SYSTEMS.put(fs.getScheme(), fs.getClass());
      }
      FILE_SYSTEMS_LOADED = true;
    }
  }
}
```

```java
private S nextService() {
    if (!hasNextService())
        throw new NoSuchElementException();
    String cn = nextName;
    nextName = null;
    Class<?> c = null;
    try {
        c = Class.forName(cn, false, loader);
    } catch (ClassNotFoundException x) {
        fail(service,
             "Provider " + cn + " not found");
    }
    if (!service.isAssignableFrom(c)) {
        fail(service,
             "Provider " + cn  + " not a subtype");
    }
    try {
        S p = service.cast(c.newInstance());
        providers.put(cn, p);
        return p;
    } catch (Throwable x) {
        fail(service,
             "Provider " + cn + " could not be instantiated",
             x);
    }
    throw new Error();          // This cannot happen
}
```

出错的这一段：

```java
if (!service.isAssignableFrom(c)) {
      fail(service,
           "Provider " + cn  + " not a subtype");
}
```

service是`org.apache.hadoop.fs.FileSystem`, cn是`org.apache.hadoop.fs.azure.NativeAzureFileSystem`，`org.apache.hadoop.fs.azure.NativeAzureFileSystem extends org.apache.hadoop.fs.FileSystem`，所以正常来说 isAssignableFrom 这个判断应该是pass才对的。

综合判断：

1、打出来的uberjar包`META-INF/services/org.apache.hadoop.fs.FileSystem`没有`org.apache.hadoop.fs.azure.NativeAzureFileSystem`的注册，也没有`org.apache.hadoop.fs.azure.NativeAzureFileSystem`的class文件，但是forName却成功了 => Spark运行时有这个SPI注册和依赖包。
2、isAssignableFrom应该成功却失败了 => NativeAzureFileSystem是由不同的classLoader加载的。


分析那么多，怎么解决呢？对比可以正常跑的uberjar包，发现人家压根就没有`META-INF/services/org.apache.hadoop.fs.FileSystem`这个文件，继续跟踪下去，发现人家压根就没有打包spark的依赖。也就是说确实`META-INF/services/org.apache.hadoop.fs.FileSystem`就是spark-core带进去的。而且集群运行时的spark-core引进来的依赖中`META-INF/services/org.apache.hadoop.fs.FileSystem`比hadoop官方注册的FileSystem Provider要多（我们线上用的是CDH版本）。 最后把spark的依赖改成provider就可以跑通了。

```xml
<dependency>
  <groupId>org.apache.spark</groupId>
  <artifactId>spark-core_${scala.version}</artifactId>
  <version>${spark.version}</version>
  <scope>provided</scope>
</dependency>
```

时间关系，我没有找一下[arthas](https://alibaba.github.io/arthas)查看一下类加载信息。留给感兴趣的同学。

```shell
[zchen@bjthq-dm-submit012.vivo.lan:/home/zchen/tools]
$ java -jar arthas-boot.jar --telnet-port 10000 --http-port 10001
[INFO] arthas-boot version: 3.0.5.2
[INFO] Found existing java process, please choose one and hit RETURN.
* [1]: 23363 org.apache.spark.deploy.SparkSubmit
  [2]: 5491 org.apache.spark.deploy.SparkSubmit
  [3]: 1781 org.apache.spark.deploy.SparkSubmit
  [4]: 33109 org.apache.spark.deploy.SparkSubmit
  [5]: 33382 /home/zchen/lib/java/itheme/recommend-mongo-1.0.0.jar
  [6]: 24150 org.apache.spark.deploy.SparkSubmit
  [7]: 21095 org.apache.spark.deploy.SparkSubmit
  [8]: 9259 /home/zchen/apps/jars/task-submit/predictor-java-spark-0.0.1-SNAPSHOT.jar
  [9]: 29275 org.apache.spark.deploy.SparkSubmit
7
[INFO] arthas home: /home/zchen/.arthas/lib/3.0.5/arthas
[INFO] Try to attach process 21095
[INFO] Attach process 21095 success.
[INFO] arthas-client connect 127.0.0.1 10000
  ,---.  ,------. ,--------.,--.  ,--.  ,---.   ,---.                           
 /  O  \ |  .--. ''--.  .--'|  '--'  | /  O  \ '   .-'                          
|  .-.  ||  '--'.'   |  |   |  .--.  ||  .-.  |`.  `-.                          
|  | |  ||  |\  \    |  |   |  |  |  ||  | |  |.-'    |                         
`--' `--'`--' '--'   `--'   `--'  `--'`--' `--'`-----'                          
                                                                                

wiki: https://alibaba.github.io/arthas
version: 3.0.5
pid: 21095
time: 2018-12-27 11:39:49

```