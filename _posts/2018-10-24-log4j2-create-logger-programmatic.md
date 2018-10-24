---
layout: post
title: log4j2如何动态的创建logger和appender
tags: [log4j2]
catalog: true
---


背景
---

平台现在通过页面通过 `SparkLauncher` 提交Spark训练任务，随着而来的一个问题就是任务出错之后不知道哪里出问题，而日志在提交节点上，并且所有的任务混在一起打印日志，很不方便查看。能不能把 Spark 任务的日志分别打印在自己的日志文件呢？


解决方案
-------

Spark2.x 有 `redirectToLog` 方法可以把任务日志重定向，但是 Spark1.x 没有提供这个功能，也不具备扩展性，所以只能自己解决了。

解决方案其实蛮简单的，就是不同的 task 创建属于自己的 Logger（即 loggerName 不一样）， 每个Logger有自己的 FileAppender，FileAppender 指向自己的日志文件（文件名称用 taskId 命名）即可以了。


实现
----

Log4j2 其实有很丰富的 API，而且是 builder 模式，用起来还是蛮方便的。[Programmatic Configuration](https://logging.apache.org/log4j/2.x/manual/customconfig.html)。


```java
package org.apache.spark.launcher;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.concurrent.ThreadFactory;

import org.apache.commons.io.FilenameUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.core.Appender;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.appender.FileAppender;
import org.apache.logging.log4j.core.config.Configuration;
import org.apache.logging.log4j.core.layout.PatternLayout;

/**
 * Redirects lines read from a given input stream to a log4j2 logger.
 * 
 * @author zhengzhibin
 * @date 2018/10/24
 */
public class Log4j2OutputRedirector {

    private final BufferedReader reader;
    private final Logger sink;
    private final Thread thread;

    private volatile boolean active;

    Log4j2OutputRedirector(InputStream in, ThreadFactory tf) {
        this(in, OutputRedirector.class.getName(), tf);
    }

    Log4j2OutputRedirector(InputStream in, String loggerName, ThreadFactory tf) {
        this.active = true;
        this.reader = new BufferedReader(new InputStreamReader(in));
        this.thread = tf.newThread(new Runnable() {
            @Override
            public void run() {
                redirect();
            }
        });
        this.sink = getLogger(loggerName);
        thread.start();
    }

    protected Logger getLogger(String loggerName) {
        // This is the root logger provided by log4j
        Logger logger = LogManager.getLogger(loggerName);

        LoggerContext context = LoggerContext.getContext(false);
        Configuration config = context.getConfiguration();
        PatternLayout layout = PatternLayout.createDefaultLayout(config);

        String logPath = config.getProperties().get("log-path");
        if (StringUtils.isBlank(logPath)) {
            logPath = FilenameUtils.concat(System.getProperty("user.home"), "task-logs");
        }

        String fileName = FilenameUtils.concat(logPath, loggerName + ".log");

        // Define file appender with layout and output log file name
        Appender appender = FileAppender.newBuilder().setConfiguration(config) //
            .withName("programmaticFileAppender") //
            .withLayout(layout) //
            .withFileName(fileName) //
            .build();

        appender.start();

        // cast org.apache.logging.log4j.Logger to org.apache.logging.log4j.core.Logger
        org.apache.logging.log4j.core.Logger loggerImpl = (org.apache.logging.log4j.core.Logger)logger;
        loggerImpl.addAppender(appender);

        return logger;
    }

    protected void redirect() {
        try {
            String line;
            while ((line = reader.readLine()) != null) {
                if (active) {
                    sink.info(line.replaceFirst("\\s*$", ""));
                }
            }
        } catch (IOException e) {
            sink.error("Error reading child process output.", e);
        }
    }

    /**
     * This method just stops the output of the process from showing up in the local logs. The child's output will still
     * be read (and, thus, the redirect thread will still be alive) to avoid the child process hanging because of lack
     * of output buffer.
     */
    void stop() {
        active = false;
    }

}
```

**说明**

1. log4j2的 `org.apache.logging.log4j.Logger` 并没有 addAppender 方法，log4j1.x 是有的，强制转换成实现类`org.apache.logging.log4j.core.Logger` 就可以了。[How to add appender to Logger in Log4j2](https://stackoverflow.com/questions/38241654/how-to-add-appender-to-logger-in-log4j2)
2. 这里会导致日志没有打印到原来的日志文件，如果需要，可以通过设置 rootLogger 和 additive 实现。


补充
---

log4j2有个 RoutingAppender 看起来是可以实现这个功能 [How do I dynamically write to separate log files?](https://logging.apache.org/log4j/2.0/faq.html#separate_log_files)，可以在创建 OutputRedirector 的时候在日志收集线程把 taskId 写入 ThreadContext 中。理论上可行，不过我没有实验这种方案。有兴趣的读者可以试试。

