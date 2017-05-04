---
layout: post
title: maven学习笔记
catalog: true
category: [技术]
tags: [maven]
---


maven的核心概念
--------------

maven的核心概念

1. 坐标
2. 依赖
3. 仓库
4. 生命周期
5. 插件
6. 继承和聚合


## 仓库

Maven中的仓库用来存放生成的构建和各种依赖。

* 本地仓库
* 远程仓库
    * 中央仓库
    * 私服(内部仓库)
    * 其他重要的仓库

### 仓库和镜像

一般来说公司都会有一个私服，然后我们可以在本地的settings.xml中配置使用：

```xml
<?xml version="1.0" encoding="UTF-8"?>

<settings xmlns="http://maven.apache.org/SETTINGS/1.0.0"
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
          xsi:schemaLocation="http://maven.apache.org/SETTINGS/1.0.0 http://maven.apache.org/xsd/settings-1.0.0.xsd">

  <profiles>
      <profile>
         <id>dev</id>
          <activation>
            <activeByDefault>true</activeByDefault>
          </activation>

         <repositories>
              <repository>
                  <id>icafe-nexus-snapshots</id>
                  <url>http://maven.scm.arganzheng.life:8081/nexus/content/groups/public-snapshots</url>
                  <releases>
                      <enabled>true</enabled>
                  </releases>
                  <snapshots>
                      <enabled>true</enabled>
                  </snapshots>
               </repository>

               <repository>
                  <id>icafe-nexus-public</id>
                  <url>http://maven.scm.arganzheng.life:8081/nexus/content/groups/public</url>
                  <releases>
                      <enabled>true</enabled>
                  </releases>
                  <snapshots>
                      <enabled>true</enabled>
                  </snapshots>
               </repository>
          </repositories>
          <pluginRepositories>
                <pluginRepository>
                    <id>nexus</id>
                    <url>http://maven.scm.arganzheng.life:8081/nexus/content/groups/public</url>
                    <releases>
                        <enabled>true</enabled>
                    </releases>
                    <snapshots>
                        <enabled>true</enabled>
                     </snapshots>
                 </pluginRepository>
           </pluginRepositories>
       </profile>
  </profiles>

  <activeProfiles>
         <activeProfile>dev</activeProfile>
  </activeProfiles>

  <mirrors>
    <mirror>
      <id>icafe-nexus-snapshots</id>
      <mirrorOf>snapshots</mirrorOf>
      <name>public snapshots</name>
      <url>http://maven.scm.arganzheng.life:8081/nexus/content/groups/public-snapshots</url>
    </mirror>
    <mirror>
      <id>icafe-nexus-public</id>
      <mirrorOf>*</mirrorOf>
      <name>public</name>
      <url>http://maven.scm.arganzheng.life:8081/nexus/content/groups/public</url>
    </mirror>
  </mirrors>

</settings>
```

### 部署到远程仓库和认证

为了让我们的构建能被其他项目依赖，我们需要使用`mvn deploy`命令把其发布到私服仓库中去。首先我们需要在项目的pom.xml中指定发布的仓库地址，这是通过`distributionManagement`指定：

```xml
<distributionManagement>
    <repository>
		<id>Argan_Local</id>
		<url>http://maven.scm.arganzheng.life:8081/nexus/content/repositories/Argan_Local</url>
    </repository>
    <snapshotRepository>
		<id>Argan_Local_Snapshots</id>
		<url>http://maven.scm.arganzheng.life:8081/nexus/content/repositories/Argan_Local_Snapshots</url>
    </snapshotRepository>
 </distributionManagement>
```

往远程仓库部署构建的时候，往往需要认证。这是通过在settings.xml中配置server元素，其id与仓库的id匹配，并且配置正确的认证信息：

```xml
<servers>
    <server>
      <id>Argan_Local</id>
      <username>argan</username>
      <password>xxx</password>
    </server>
    <server>
      <id>Argan_Local_Snapshots</id>
      <username>argan</username>
      <password>xxx</password>
    </server>
</servers>
```

关于maven仓库的更多介绍可以参考笔者之前的文章 [如何构建maven私有仓库](http://arganzheng.life/setup-maven-repository.html)

## 生命周期

maven对项目的某个构建过程进行了抽象和定义，这个过程被称为生命周期(lifecycle)。每套生命周期(lifecycle)由多个阶段(phase)组成，这些阶段组成有顺序的流水线(Pipeline)，每个阶段(phase)会挂接一到多个目标(goal)。goal是maven里定义任务的最小单元，相当于ant里的target。

Maven中定义的工程周期和阶段只是抽象的概念，不涉及具体的功能。 具体的功能由插件（Plugin）实现。一个插件可以实现多个目标（Goal）。

整个LifeCycle就是一个pipeline的流水线机制。

Maven有三套相互独立的生命周期，这三套生命周期分别是：

* clean Lifecycle: 在进行真正的构建之前进行一些清理工作。
* default Lifecycle: 构建的核心部分，编译，测试，打包，部署等等。
* site Lifecycle: 生成项目报告，站点，发布站点。

需要强调一下，这三个生命周期是相互独立的，你可以仅仅调用clean来清理工作目录，仅仅调用site来生成站点。当然你也可以直接运行 `mvn clean install site` 运行所有这三套生命周期。

### clean生命周期

clean生命周期顾名思义，用于清理项目。它一共包含了三个阶段：

1. pre-clean: 执行一些需要在clean之前完成的工作
2. clean: 移除所有上一次构建生成的文件
3. post-clean: 执行一些需要在clean之后立刻完成的工作

**TIPS**

mvn clean 中的clean就是上面的clean，在一个生命周期中，运行某个阶段的时候，它之前的所有阶段都会被运行，也就是说，mvn clean 等同于 mvn pre-clean clean ，如果我们运行 mvn post-clean ，那么 pre-clean，clean 都会被运行。这是Maven很重要的一个规则，可以大大简化命令行的输入。

### default生命周期

default生命周期是maven最重要也是最复杂的生命周期，它定义了真正构建时所需要执行的所有步骤，包含的阶段如下：

1. validate: validate the project is correct and all necessary information is available.
2. initialize: initialize build state, e.g. set properties or create directories.
3. generate-sources: generate any source code for inclusion in compilation.
4. process-sources: process the source code, for example to filter any values.
5. generate-resources: generate resources for inclusion in the package.
6. process-resources: 复制并处理主资源文件，一般来说，是对 src/main/resources 目录的内容进行变量替换等内容等，复制到项目输出的主classpath目录中。
7. compile: 编译项目的源代码。一般来说，是编译 src/main/java 目录下的Java文件到项目输出的主classpath目录中。
8. process-classes: post-process the generated files from compilation, for example to do bytecode enhancement on Java classes.
9. generate-test-sources: generate any test source code for inclusion in compilation.
10. process-test-sources: 处理项目测试资源文件。一般来说，是对 src/test/resources 目录的内容进行变量替换等内容等，复制到项目输出的测试classpath目录中。
11. generate-test-resources: create resources for testing.
12. process-test-resources: 复制并处理测试资源文件，至目标测试目录。
13. test-compile: 编译项目的测试代码。一般来说，是编译 src/test/java 目录下的Java文件到项目输出的测试classpath目录中。
14. process-test-classes: post-process the generated files from test compilation, for example to do bytecode enhancement on Java classes. For Maven 2.0.5 and above.
15. test: 使用合适的单元测试框架运行测试，测试代码不会被打包或部署。
16. prepare-package: perform any operations necessary to prepare a package before the actual packaging. This often results in an unpacked, processed version of the package. (Maven 2.1 and above)
17. package: 接受编译好的代码，打包成可发布的格式，如JAR、war。
18. pre-integration-test: perform actions required before integration tests are executed. This may involve things such as setting up the required environment.
19. integration-test: process and deploy the package if necessary into an environment where integration tests can be run.
20. post-integration-test: perform actions required after integration tests have been executed. This may including cleaning up the environment.
21. verify: run any checks to verify the package is valid and meets quality criteria.
22. install: 将包安装至本地仓库，供本地其他maven项目依赖。
23. deploy: 将最终的包复制到远程的仓库，以让其它开发人员与maven项目使用。


### site的生命周期

site生命周期的目的是建立和发布项目站点，该生命周期包含如下阶段：

1. pre-site: 执行一些需要在生成站点文档之前完成的工作
2. site: 生成项目的站点文档
3. post-site: 执行一些需要在生成站点文档之后完成的工作，并且为部署做准备
4. site-deploy: 将生成的站点文档部署到特定的服务器上

这里经常用到的是site阶段和site-deploy阶段，用以生成和发布Maven站点，这可是Maven相当强大的功能，Manager比较喜欢，文档及统计数据自动生成，很好看。

### 命令行执行maven命令

1. mvn groupId:artifactId[:version]:goal
2. mvn plugin_prefix:goal
3. mvn phase 

**说明** 

1. 第二种命令格式其实是第一种命令格式的简写，之所以可以这样子是因为该plugin在默认的两个group中有定义。如果该插件没有在这两个group下定义，那么就需要自己在pom文件中定义一下。比如jetty插件：

        <plugin>
        <groupId>org.mortbay.jetty</groupId>
        <artifactId>maven-jetty-plugin</artifactId>
        <version>6.1.16</version>
        <configuration>
           <contextPath>/</contextPath>
           <scanTargets>
              <scanTarget>target/classes/</scanTarget>
           </scanTargets>
          <scanIntervalSeconds>5</scanIntervalSeconds>
        </configuration>
        </plugin>

2. 第三种命令格式`mvn phase`，其实是执行maven默认绑定或者你通过<plugins>配置绑定的一个或者多个plugin的某个goal。这个可以`help:effective-pom`查看。


## 继承与聚合

聚合往往用来作为模块化开发。但是即使是没有模块依赖关系的project，也可以使用聚合pom来管理其之间的依赖关系。比如一个项目药修改4个工程，这四个工程之间有依赖关系。一般做法是修改了A工程，需要同时build依赖到A工程的其他工程，手工处理这种依赖关系是很麻烦且容易出错的事情。使用顶级聚合pom，就可以自动按照依赖关系编译这4个工程了。

