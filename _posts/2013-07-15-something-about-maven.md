---
layout: post
title: maven学习笔记
---

## maven的核心概念

1. 生命周期(Lifecycle)
2. 阶段(Phase)
3. 目标(Goal)


这三者的关系是这样子的：

maven对构建(build)的过程进行了抽象和定义，这个过程被称为构建的生命周期(lifecycle)。生命周期(lifecycle)由多个阶段(phase)组成,每个阶段(phase)会挂接一到多个goal。goal是maven里定义任务的最小单元，相当于ant里的target。

Maven中定义的工程周期和阶段只是抽象的概念，不涉及具体的功能。 具体的功能由插件（Plugin）实现。一个插件可以实现多个目标（Goal）。

整个LifeCycle就是一个pipeline的流水线机制。


## 命令行执行maven命令

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

