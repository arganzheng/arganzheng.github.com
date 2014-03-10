---
title: 已知一个java类，如何知道其在哪个jar包
layout: post
---


比如我们想知道HeadApiProtocol.class在哪个jar包，路径在哪里，可以这么处理：

    System.out.println(HeadApiProtocol.class.getProtectionDomain().getCodeSource().getLocation());

打印结果是：

    file:/D:/code/api_metadata_proj/target/api-metadata-1.0/WEB-INF/lib/com.paipai.util-3.1.9.jar

然后再把这个jar包添加到classpath中，就可以实现动态编译了。