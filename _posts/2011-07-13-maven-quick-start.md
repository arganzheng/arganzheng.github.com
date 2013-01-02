---
layout: post
title: Maven快速入门
---


不管是构建web程序或者standalone程序，甚至是简单的写个main测试，只要你的工程需要引入依赖或者打包构建，那么使用maven是最好也是最简单的做法。

1. 安装maven
这个google一下，应该能搞定。

2. 创建工程

如果你已经能够确定`-DarchetypeArtifactId`，那么可以直接指定，并设置`-DinteractiveMode=false`
    
    mvn archetype:generate -DgroupId=com.mycompany.app -DartifactId=my-app -DarchetypeArtifactId=maven-archetype-quickstart -DinteractiveMode=false
  
例如：
  
    forrest@ubuntu:~/work2/test$ mvn archetype:generate -DgroupId=com.mycompany.app -DartifactId=my-app -DarchetypeArtifactId=maven-archetype-quickstart -DinteractiveMode=false
    [INFO] Scanning for projects...
    [INFO] Searching repository for plugin with prefix: 'archetype'.
    [INFO] ------------------------------------------------------------------------
    [INFO] Building Maven Default Project
    [INFO]    task-segment: [archetype:generate] (aggregator-style)
    [INFO] ------------------------------------------------------------------------
    [INFO] Preparing archetype:generate
    [INFO] No goals needed for project - skipping
    [INFO] [archetype:generate {execution: default-cli}]
    [INFO] Generating project in Batch mode
    Downloading: http://repo1.maven.org/maven2/org/apache/maven/archetypes/maven-archetype-quickstart/1.0/maven-archetype-quickstart-1.0.jar
      
    Downloading: http://repo1.maven.org/maven2/org/apache/maven/archetypes/maven-archetype-quickstart/1.0/maven-archetype-quickstart-1.0.pom
     
    [INFO] ----------------------------------------------------------------------------
    [INFO] Using following parameters for creating project from Old (1.x) Archetype: maven-archetype-quickstart:1.0
    [INFO] ----------------------------------------------------------------------------
    [INFO] Parameter: groupId, Value: com.mycompany.app
    [INFO] Parameter: packageName, Value: com.mycompany.app
    [INFO] Parameter: package, Value: com.mycompany.app
    [INFO] Parameter: artifactId, Value: my-app
    [INFO] Parameter: basedir, Value: /home/forrest/work2/test
    [INFO] Parameter: version, Value: 1.0-SNAPSHOT
    [INFO] ********************* End of debug info from resources from generated POM ***********************
    [INFO] project created from Old (1.x) Archetype in dir: /home/forrest/work2/test/my-app
    [INFO] ------------------------------------------------------------------------
    [INFO] BUILD SUCCESSFUL
    [INFO] ------------------------------------------------------------------------
    [INFO] Total time: 3 seconds
    [INFO] Finished at: Wed Jul 13 11:19:53 CST 2011
    [INFO] Final Memory: 20M/169M
    [INFO] ------------------------------------------------------------------------
    forrest@ubuntu:~/work2/test$ 
    forrest@ubuntu:~/work2/test$ tree .
    .
    └── my-app
        ├── pom.xml
        └── src
            ├── main
            │   └── java
            │       └── com
            │           └── mycompany
            │               └── app
            │                   └── App.java
            └── test
                └── java
                    └── com
                        └── mycompany
                            └── app
                                └── AppTest.java

    12 directories, 3 files
    
否则，`mvn archetype:generate`会使用交互模式，让你动态选择`archetypeArtifactId`，填写groupId之类的（加入你没有填写的话）：`mvn archetype:generate`
会一一提示你或者或者填写相应信息的。

现在工程的骨架已经构建好，你可以根据需要修改pom文件了。Have fun:)

一个例子：

    <project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/maven-v4_0_0.xsd">
        <modelVersion>4.0.0</modelVersion>
        <groupId>com.mycompany.app</groupId>
        <artifactId>my-app</artifactId>
        <packaging>jar</packaging>
        <version>1.0-SNAPSHOT</version>
        <name>my-app</name>
        <url>http://maven.apache.org&lt;/url>
        <dependencies>
                <dependency>
                        <groupId>com.alibaba.intl.sourcing.shared</groupId>
                        <artifactId>biz.srproduct.sketch</artifactId>
                        <version>1.0.4</version>
                </dependency>
                <dependency>
                        <groupId>com.alibaba.intl.sourcing.shared</groupId>
                        <artifactId>test</artifactId>
                        <version>1.0-SNAPSHOT</version>
                </dependency>
        </dependencies>

        <build>
                <plugins>
                        <plugin>
                                <groupId>org.apache.maven.plugins</groupId>
                                <artifactId>maven-compiler-plugin</artifactId>
                                <configuration>
                                        <source>1.6</source>
                                        <target>1.6</target>
                                </configuration>
                        </plugin>
                </plugins>
        </build>    
    </project>

