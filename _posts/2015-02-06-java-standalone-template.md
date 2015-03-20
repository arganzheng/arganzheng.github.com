---
title: java standalone模板
layout: post
---


经常有一些需求需要跑一个standalone程序。不像war有标准的目录结构，standalone需要自己打包，组织好依赖和classpath等。于是搞了一个standalone模板。大概如下:

standalone工程目录如下：

* mystandalone
    * src
        * bin
        * main
        * resources
            * cache
            * conf-mybatis
            * conf-spring
            * xxx_properties
            * ...
    * pom.xml

打包后的工程目录如下：

* mystandalone
	* lib 依赖的jar包
    * conf 所有配置文件
    * bin 启动脚本

其中lib目录可以用copy-dependencies插件copy过去，本身生成的jar包可以通过指定outputDirectory也打包到lib目录下。
conf目录可以用resources插件从`src/main/resources`目录里面copy到conf目录。

本来想尝试一下maven的assembly插件的，但是发现这个插件文档真的是看不懂。。所以，还是ant吧。。maven有ant插件，可以直接用。于是整个pom.xml文件大概如下所示：


    <?xml version="1.0" encoding="UTF-8"?>
    <project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
          xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/maven-v4_0_0.xsd">
          <modelVersion>4.0.0</modelVersion>
          <groupId>me.arganzheng.study.standalone</groupId>
          <artifactId>count-processor</artifactId>
          <packaging>jar</packaging>
          <version>1.0.0</version>
          <name>A Standalone for count processor</name>

          <properties>
               <target.dir>count-processor</target.dir>
               <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
          </properties>

          <dependencies>
               ...
          </dependencies>

          <build>
               <!-- 不把 src/main/resources也打包到jar包中，资源文件后面单独copy到conf目录中 -->
               <resources>
                    <resource>
                         <directory>src/main/resources</directory>
                         <excludes>
                              <exclude>**/*</exclude>
                         </excludes>
                    </resource>
               </resources>
               <plugins>
                    <plugin>
                         <groupId>org.apache.maven.plugins</groupId>
                         <artifactId>maven-compiler-plugin</artifactId>
                         <version>2.3.2</version>
                         <configuration>
                              <encoding>${project.build.sourceEncoding}</encoding>
                              <source>1.6</source>
                              <target>1.6</target>
                         </configuration>
                    </plugin>
                    <!-- 解决资源文件的编码问题 -->
                    <plugin>
                         <groupId>org.apache.maven.plugins</groupId>
                         <artifactId>maven-resources-plugin</artifactId>
                         <version>2.3</version>
                         <configuration>
                              <encoding>${project.build.sourceEncoding}</encoding>
                         </configuration>
                    </plugin>
                    <!-- 默认打包到${project.build.directory}目录，这里直接将其打包到lib目录。这里不指定mainClass和指定classPath，这些统一在startup启动脚本处理 -->
                    <plugin>
                         <groupId>org.apache.maven.plugins</groupId>
                         <artifactId>maven-jar-plugin</artifactId>
                         <configuration>
                              <outputDirectory>
                                   ${project.build.directory}/${target.dir}/lib
                              </outputDirectory>
                         </configuration>
                    </plugin>
                    <!-- 将依赖的jar包也拷贝到lib目录 -->
                    <plugin>
                         <groupId>org.apache.maven.plugins</groupId>
                         <artifactId>maven-dependency-plugin</artifactId>
                         <version>2.5.1</version>
                         <executions>
                              <execution>
                                   <id>copy-dependencies</id>
                                   <phase>package</phase>
                                   <goals>
                                        <goal>copy-dependencies</goal>
                                   </goals>
                                   <configuration>
                                        <outputDirectory>
                                             ${project.build.directory}/${target.dir}/lib/
                                        </outputDirectory>
                                   </configuration>
                              </execution>
                         </executions>
                    </plugin>
                    <!-- 使用ant插件将lib, conf和 bin目录打包成一个zip包，方便发布 -->
                    <plugin>
                         <groupId>org.apache.maven.plugins</groupId>
                         <artifactId>maven-antrun-plugin</artifactId>
                         <executions>
                              <!-- 将resources 打包到 conf 目录下 -->
                              <execution>
                                   <phase>process-resources</phase>
                                   <configuration>
                                        <tasks>
                                             <copy todir="${project.build.directory}/${target.dir}/bin">
                                                  <fileset dir="src/bin" />
                                             </copy>
                                             <copy todir="${project.build.directory}/${target.dir}/conf">
                                                  <fileset dir="src/main/resources" />
                                             </copy>
                                        </tasks>
                                   </configuration>
                                   <goals>
                                        <goal>run</goal>
                                   </goals>
                              </execution>
                              <!-- 打包成zip -->
                              <execution>
                                   <id>makeZipfile</id>
                                   <phase>package</phase>
                                   <configuration>
                                        <tasks>
                                             <zip destfile="${project.build.directory}/${target.dir}.zip"
                                                  duplicate="preserve">
                                                  <zipfileset dir="${project.build.directory}/${target.dir}"
                                                       includes="**/*.*" />
                                             </zip>
                                        </tasks>
                                   </configuration>
                                   <goals>
                                        <goal>run</goal>
                                   </goals>
                              </execution>
                         </executions>
                    </plugin>
               </plugins>
          </build>
    </project>

这样就可以了，但是如前面所说，启动脚本需要处理一下：

	$ cat bin/start.sh

	#!/bin/bash

	# determine base directory; preserve where you're running from
	#echo "Path to $(basename $0) is $(readlink -f $0)"
	realpath=$(readlink -f "$0")
	filepath=$(dirname "$realpath")
	basedir=${filepath%/*}

	LOG_DIR=${basedir}/logs
	mkdir -p ${LOG_DIR}
	GC_LOG_DIR=${basedir}/logs/gc
	mkdir -p ${GC_LOG_DIR}
	GC_FILE_PATH="${GC_LOG_DIR}/gc-$(date +%s).log"

	LIBCLASSPATH=`echo ${basedir}/lib/*.jar | tr ' ' ':'`
	export CLASSPATH=$LIBCLASSPATH:${basedir}/conf
	echo $CLASSPATH
	echo $basedir

	JAVA_OPTS="-server -Xms1024m -Xmx1024m -XX:+DisableExplicitGC -Xloggc:${GC_FILE_PATH} -XX:+PrintGCDetails -XX:+HeapDumpOnOutOfMemoryError -Dlog4j.configuration=file:$basedir/conf/log4j.properties -DLOG_DIR=${LOG_DIR}"
	echo $JAVA_OPTS

	java $JAVA_OPTS me.arganzheng.study.standalone.message.queue.processor.Main


**TIPS** 用maven执行main函数

对于standalone工程一般有个Main入口类，打包后通过startup脚本执行当然是没有问题的，但是每次都要打包，很麻烦，用maven也是可以指定的：

比如这个：java $JAVA_OPTS me.arganzheng.study.standalone.message.queue.processor.Main
用maven就可以这样子：mvn compile exec:java -Dexec.mainClass="me.arganzheng.study.standalone.message.queue.processor.Main" 执行执行了。


**NOTE** 关于log4j.configuration的一个坑

如果lo4j.properties在classpath中，那么其实不需要配置这个环境变量，但是有一种情况，比如有多个log4j.properties，然后你想指定具体的一个。但是千万记得，如果是绝对路径，一定要加上`file:`前缀，否则会读取不到的。如果只是文件名，比如 -Dlog4j.configuration=foobar.xml 那么就不需要。

	log4j:WARN No appenders could be found for logger (org.springframework.core.xx).
	log4j:WARN Please initialize the log4j system properly.

具体参考: [log4j - Default Initialization under Tomcat](http://logging.apache.org/log4j/1.2/manual.html#defaultInit)
