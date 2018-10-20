---
title: Spark任务读取HDFS文件报Filesystem closed异常
layout: post
tags: [Spark, hadoop]
catalog: true
---


现象
---

今天早上过来发现昨天晚上浏览器搜索样本生成任务出问题了。日志显示：

```
[INFO ] 2018-10-18 22:03:41.443 [launcher-proc-1] spark-sample-generate -       at com.argan.internt.ai.platform.datadriver.sample.plugin.DefaultPluginServiceImpl.loadPlugin(Default
PluginServiceImpl.java:65)
[INFO ] 2018-10-18 22:03:41.444 [launcher-proc-1] spark-sample-generate -       at com.argan.internt.ai.platform.datadriver.sample.function.SampleRddReducerFunction.getFeatureProces
sor(SampleRddReducerFunction.java:72)
[INFO ] 2018-10-18 22:03:41.444 [launcher-proc-1] spark-sample-generate -       at com.argan.internt.ai.platform.datadriver.sample.function.SampleRddReducerFunction.call(SampleRddRe
ducerFunction.java:60)
[INFO ] 2018-10-18 22:03:41.444 [launcher-proc-1] spark-sample-generate -       at com.argan.internt.ai.platform.datadriver.sample.function.SampleRddReducerFunction.call(SampleRddRe
ducerFunction.java:26)
[INFO ] 2018-10-18 22:03:41.444 [launcher-proc-1] spark-sample-generate -       at org.apache.spark.api.java.JavaPairRDD$$anonfun$pairFunToScalaFun$1.apply(JavaPairRDD.scala:1018)
[INFO ] 2018-10-18 22:03:41.444 [launcher-proc-1] spark-sample-generate -       at org.apache.spark.api.java.JavaPairRDD$$anonfun$pairFunToScalaFun$1.apply(JavaPairRDD.scala:1018)
[INFO ] 2018-10-18 22:03:41.444 [launcher-proc-1] spark-sample-generate -       at scala.collection.Iterator$$anon$11.next(Iterator.scala:328)
[INFO ] 2018-10-18 22:03:41.444 [launcher-proc-1] spark-sample-generate -       at org.apache.spark.rdd.PairRDDFunctions$$anonfun$saveAsHadoopDataset$1$$anonfun$13$$anonfun$apply$7
.apply$mcV$sp(PairRDDFunctions.scala:1196)
[INFO ] 2018-10-18 22:03:41.444 [launcher-proc-1] spark-sample-generate -       at org.apache.spark.rdd.PairRDDFunctions$$anonfun$saveAsHadoopDataset$1$$anonfun$13$$anonfun$apply$7
.apply(PairRDDFunctions.scala:1195)
[INFO ] 2018-10-18 22:03:41.444 [launcher-proc-1] spark-sample-generate -       at org.apache.spark.rdd.PairRDDFunctions$$anonfun$saveAsHadoopDataset$1$$anonfun$13$$anonfun$apply$7
.apply(PairRDDFunctions.scala:1195)
[INFO ] 2018-10-18 22:03:41.444 [launcher-proc-1] spark-sample-generate -       at org.apache.spark.util.Utils$.tryWithSafeFinallyAndFailureCallbacks(Utils.scala:1282)
[INFO ] 2018-10-18 22:03:41.444 [launcher-proc-1] spark-sample-generate -       at org.apache.spark.rdd.PairRDDFunctions$$anonfun$saveAsHadoopDataset$1$$anonfun$13.apply(PairRDDFun
ctions.scala:1203)
[INFO ] 2018-10-18 22:03:41.444 [launcher-proc-1] spark-sample-generate -       at org.apache.spark.rdd.PairRDDFunctions$$anonfun$saveAsHadoopDataset$1$$anonfun$13.apply(PairRDDFun
ctions.scala:1183)
[INFO ] 2018-10-18 22:03:41.444 [launcher-proc-1] spark-sample-generate -       at org.apache.spark.scheduler.ResultTask.runTask(ResultTask.scala:66)
[INFO ] 2018-10-18 22:03:41.444 [launcher-proc-1] spark-sample-generate -       at org.apache.spark.scheduler.Task.run(Task.scala:89)
[INFO ] 2018-10-18 22:03:41.444 [launcher-proc-1] spark-sample-generate -       at org.apache.spark.executor.Executor$TaskRunner.run(Executor.scala:242)
[INFO ] 2018-10-18 22:03:41.444 [launcher-proc-1] spark-sample-generate -       at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1142)
[INFO ] 2018-10-18 22:03:41.444 [launcher-proc-1] spark-sample-generate -       at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:617)
[INFO ] 2018-10-18 22:03:41.444 [launcher-proc-1] spark-sample-generate -       at java.lang.Thread.run(Thread.java:745)
[INFO ] 2018-10-18 22:03:41.444 [launcher-proc-1] spark-sample-generate - Caused by: java.io.IOException: Filesystem closed
[INFO ] 2018-10-18 22:03:41.444 [launcher-proc-1] spark-sample-generate -       at org.apache.hadoop.hdfs.DFSClient.checkOpen(DFSClient.java:892)
[INFO ] 2018-10-18 22:03:41.444 [launcher-proc-1] spark-sample-generate -       at org.apache.hadoop.hdfs.DFSInputStream.readWithStrategy(DFSInputStream.java:884)
[INFO ] 2018-10-18 22:03:41.444 [launcher-proc-1] spark-sample-generate -       at org.apache.hadoop.hdfs.DFSInputStream.read(DFSInputStream.java:957)
[INFO ] 2018-10-18 22:03:41.444 [launcher-proc-1] spark-sample-generate -       at java.io.DataInputStream.read(DataInputStream.java:100)
[INFO ] 2018-10-18 22:03:41.444 [launcher-proc-1] spark-sample-generate -       at org.apache.hadoop.io.IOUtils.copyBytes(IOUtils.java:93)
[INFO ] 2018-10-18 22:03:41.444 [launcher-proc-1] spark-sample-generate -       at org.apache.hadoop.io.IOUtils.copyBytes(IOUtils.java:61)
[INFO ] 2018-10-18 22:03:41.444 [launcher-proc-1] spark-sample-generate -       at org.apache.hadoop.io.IOUtils.copyBytes(IOUtils.java:121)
[INFO ] 2018-10-18 22:03:41.444 [launcher-proc-1] spark-sample-generate -       at org.apache.hadoop.fs.FileUtil.copy(FileUtil.java:369)
[INFO ] 2018-10-18 22:03:41.444 [launcher-proc-1] spark-sample-generate -       at org.apache.hadoop.fs.FileUtil.copy(FileUtil.java:341)
[INFO ] 2018-10-18 22:03:41.444 [launcher-proc-1] spark-sample-generate -       at org.apache.hadoop.fs.FileUtil.copy(FileUtil.java:292)
[INFO ] 2018-10-18 22:03:41.444 [launcher-proc-1] spark-sample-generate -       at org.apache.hadoop.fs.FileSystem.copyToLocalFile(FileSystem.java:2123)
[INFO ] 2018-10-18 22:03:41.444 [launcher-proc-1] spark-sample-generate -       at org.apache.hadoop.fs.FileSystem.copyToLocalFile(FileSystem.java:2092)
[INFO ] 2018-10-18 22:03:41.444 [launcher-proc-1] spark-sample-generate -       at org.apache.hadoop.fs.FileSystem.copyToLocalFile(FileSystem.java:2068)
[INFO ] 2018-10-18 22:03:41.444 [launcher-proc-1] spark-sample-generate -       at com.argan.internt.ai.platform.algorithm.commons.util.HdfsClient.copyToLocal(HdfsClient.java:95)
[INFO ] 2018-10-18 22:03:41.444 [launcher-proc-1] spark-sample-generate -       ... 19 more
```

前面遇到很多次 Spark 任务执行异常，这是难得的一次异常堆栈这么清晰明了的。就是 Spark 的 executor 的 task 在从 HDFS 上下载插件到本地目录的时候报错了，错误显示是读取 HDFS 文件流时检查发现文件系统被关闭了。


分析
----

看一下我们加载插件的代码:

```java
public class DefaultPluginServiceImpl implements PluginService {

    ...

    @Override
    public void loadPlugin(String pluginName) {

        ...

        HdfsClient.build( hdfsRoot, hdfsUsername, hdfsWorkDir ).copyToLocal( hdfsLoc, destFile ).close();

        ...
    }

    ...

}


public class HdfsClient implements Closeable {

    ...

    private FileSystem fileSystem;

    public static HdfsClient build(String hdfsRoot, String loginName, String workDir) {
        HdfsClient client = new HdfsClient(hdfsRoot, loginName, workDir);
        client.init();
        return client;
    }

    private void init() throws IllegalArgumentException {

        ...

        System.setProperty("HADOOP_USER_NAME", loginName);
        Configuration cfg = new Configuration();
        cfg.set("fs.hdfs.impl", "org.apache.hadoop.hdfs.DistributedFileSystem"); // 避免打jar包时SPI文件出现问题

        String[] rootPaths = hdfsRoot.split(",");

        this.fileSystem = getFileSystem(rootPaths, workDir, cfg);
    }

    /**
     * 根据路径获取FileSystem，因为ip可能会变，因此要尝试不同的连接地址
     */
    private FileSystem getFileSystem(String[] rootPaths, String uploadPath, Configuration cfg) throws RuntimeException {

        FileSystem fileSystem = null;
        Throwable ex = null;
        for (String path : rootPaths) {
            try {
                fileSystem = FileSystem.get(URI.create(path), cfg);
                fileSystem.exists(new Path(uploadPath)); // 检查hdfs服务是否正常
                return fileSystem;
            } catch (Throwable e) {
                ex = e;
                logger.warn("HDFS init FileSystem error:" + path);
            }
        }
        logger.error("HDFS init error:" + hdfsRoot, ex);
        throw new RuntimeException("HDFS init error");
    }

    public void close() {
        if (null != fileSystem) {
            try {
                fileSystem.close();
            } catch (IOException e) {
                logger.error("hdfs close exception", e);
            }
        }
    }
}
```

可以看到每个HdfsClient实例相当于一个FileSystem实例。让我们分析一下代码：

```java
HdfsClient.build( hdfsRoot, hdfsUsername, hdfsWorkDir ).copyToLocal( hdfsLoc, destFile ).close();
```

这里每次都会 new 一个HdfsClient实例也就是得到一个FileSystem，然后去把插件 copyToLocal，最后 close 掉。这里每次 copyToLocal 都 close 掉 FileSystem 在多线程环境下确实会存在问题。但是这里 HdfsClient 并不是单例，难道它的实例变量 fileSystem 是共享的？fileSystem的获取是通过`fileSystem = FileSystem.get(URI.create(path), cfg);`得到的，跟进去FileSystem源码看一下就知道了：

```java
@InterfaceAudience.Public
@InterfaceStability.Stable
public abstract class FileSystem extends Configured implements Closeable {

  /** FileSystem cache */
  static final Cache CACHE = new Cache();

  /** The key this instance is stored under in the cache. */
  private Cache.Key key;

  /** Returns the FileSystem for this URI's scheme and authority.  The scheme
   * of the URI determines a configuration property name,
   * <tt>fs.<i>scheme</i>.class</tt> whose value names the FileSystem class.
   * The entire URI is passed to the FileSystem instance's initialize method.
   */
  public static FileSystem get(URI uri, Configuration conf) throws IOException {
    String scheme = uri.getScheme();
    String authority = uri.getAuthority();

    if (scheme == null && authority == null) {     // use default FS
      return get(conf);
    }

    if (scheme != null && authority == null) {     // no authority
      URI defaultUri = getDefaultUri(conf);
      if (scheme.equals(defaultUri.getScheme())    // if scheme matches default
          && defaultUri.getAuthority() != null) {  // & default has authority
        return get(defaultUri, conf);              // return default
      }
    }
    
    String disableCacheName = String.format("fs.%s.impl.disable.cache", scheme);
    if (conf.getBoolean(disableCacheName, false)) {
      return createFileSystem(uri, conf);
    }

    return CACHE.get(uri, conf);
  }

  /** Caching FileSystem objects */
  static class Cache {
    private final ClientFinalizer clientFinalizer = new ClientFinalizer();

    private final Map<Key, FileSystem> map = new HashMap<Key, FileSystem>();
    private final Set<Key> toAutoClose = new HashSet<Key>();

    /** A variable that makes all objects in the cache unique */
    private static AtomicLong unique = new AtomicLong(1);

    FileSystem get(URI uri, Configuration conf) throws IOException{
      Key key = new Key(uri, conf);
      return getInternal(uri, conf, key);
    }

    private FileSystem getInternal(URI uri, Configuration conf, Key key) throws IOException{
      FileSystem fs;
      synchronized (this) {
        fs = map.get(key);
      }
      if (fs != null) {
        return fs;
      }

      fs = createFileSystem(uri, conf);
      synchronized (this) { // refetch the lock again
        FileSystem oldfs = map.get(key);
        if (oldfs != null) { // a file system is created while lock is releasing
          fs.close(); // close the new file system
          return oldfs;  // return the old file system
        }
        
        // now insert the new file system into the map
        if (map.isEmpty()
                && !ShutdownHookManager.get().isShutdownInProgress()) {
          ShutdownHookManager.get().addShutdownHook(clientFinalizer, SHUTDOWN_HOOK_PRIORITY);
        }
        fs.key = key;
        map.put(key, fs);
        if (conf.getBoolean("fs.automatic.close", true)) {
          toAutoClose.add(key);
        }
        return fs;
      }
    }
  }

  /** FileSystem.Cache.Key */
    static class Key {
      final String scheme;
      final String authority;
      final UserGroupInformation ugi;
      final long unique;   // an artificial way to make a key unique

      Key(URI uri, Configuration conf) throws IOException {
        this(uri, conf, 0);
      }

      Key(URI uri, Configuration conf, long unique) throws IOException {
        scheme = uri.getScheme()==null?"":uri.getScheme().toLowerCase();
        authority = uri.getAuthority()==null?"":uri.getAuthority().toLowerCase();
        this.unique = unique;
        
        this.ugi = UserGroupInformation.getCurrentUser();
      }

      @Override
      public int hashCode() {
        return (scheme + authority).hashCode() + ugi.hashCode() + (int)unique;
      }

      static boolean isEqual(Object a, Object b) {
        return a == b || (a != null && a.equals(b));        
      }

      @Override
      public boolean equals(Object obj) {
        if (obj == this) {
          return true;
        }
        if (obj != null && obj instanceof Key) {
          Key that = (Key)obj;
          return isEqual(this.scheme, that.scheme)
                 && isEqual(this.authority, that.authority)
                 && isEqual(this.ugi, that.ugi)
                 && (this.unique == that.unique);
        }
        return false;        
      }

      @Override
      public String toString() {
        return "("+ugi.toString() + ")@" + scheme + "://" + authority;        
      }
    }
  }
}
```

果然是源码之前了无秘密，`FileSystem.get`静态方法确实不是每次都返回一个新的 FileSystem 实例，如果打开了缓存的话，就会直接从静态缓存对象中返回已经创建的实例。而缓存默认是打开的。缓存的方式就是常见的懒加载模式（存在就返回，不存在就创建并放在 cache 中）。值得注意的是缓存的 key 其实跟 conf 其实没有什么关系，主要是 uri (rootPath)。

晓峰他们昨天晚上没有搞定是因为之前在跑应用市场搜索广告的样本训练任务的时候是没有问题的。同样的代码，只是这个下载浏览器搜索广告的插件出问题了，所以方向就被带歪了。但是实际上，对比一下两者样本训练任务的参数，发现两者配置的 `spark.executor.cores` 参数值并不一样，前者是配置了 1个而后者是配置了4个，这就是解释了前者为什么一直没有这个问题。

解决方案
-------

知道了原因解决方案就容易了。一种解决方案是关闭缓存：

```
conf.setBoolean("fs.hdfs.impl.disable.cache", true);
```

网上也有人建议说不要 close 掉文件系统，让 Hadoop 自己做资源回收。

1. [Do we need to call fs.close() explicitly while working with HDFS using java API?](https://stackoverflow.com/questions/38846204/do-we-need-to-call-fs-close-explicitly-while-working-with-hdfs-using-java-api)
2. [Don't call filesystem.close in Hadoop](http://ben-tech.blogspot.com/2013/08/dont-call-filesystemclose-in-hadoop.html)

最终我们采取了第二种方案。


参考文章
-------

1. [Hadoop Error: java.io.IOException: Filesystem closed](https://wp.huangshiyang.com/filesystem_closed)
2. [java.io.IOException: Filesystem closed](https://blog.csdn.net/bitcarmanlee/article/details/68488616)