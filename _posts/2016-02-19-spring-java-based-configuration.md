---
title: Spring Java-based配置
layout: post
---

Spring目前为止支持如下三种配置方式：

1. XML-based configuration: Spring 1.0
2. Annotation-based configuration (with component scanning): Spring 2.0
3. Java-based configuration Spring 3.0


servlet 3之后，web.xml中的配置也可以通过java类来。

[WebApplicationInitializer](http://docs.spring.io/spring/docs/3.1.x/javadoc-api/org/springframework/web/WebApplicationInitializer.html)


参考文章
-------

1. [Consider Replacing Spring XML Configuration with JavaConfig](https://dzone.com/articles/consider-replacing-spring-xml)
2. [Spring App Migration: From XML to Java-based Config](http://www.robinhowlett.com/blog/2013/02/13/spring-app-migration-from-xml-to-java-based-config/) 非常详细的迁移文章，而且使用了component scanning。
3. [Spring 4 MVC HelloWorld Tutorial – Annotation/JavaConfig Example](http://websystique.com/springmvc/spring-4-mvc-helloworld-tutorial-annotation-javaconfig-full-example/) Spring MVC配置示例
4. [Spring bean management using Java configuration](http://www.ibm.com/developerworks/library/ws-springjava/)
