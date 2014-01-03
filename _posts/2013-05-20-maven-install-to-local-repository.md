---
title: 如何把一个jar包install到maven本地仓库
layout: post
---


使用mvn install:install-file命令：

    $ mvn install:install-file -Dfile=idl-tools.jar -DgroupId=me.arganzheng.study -DartifactId=idl-tools -Dversion=1.0 -Dpackaging=jar
  
然后就可以在依赖到地方引用了
  
    <dependency>
        <groupId>me.arganzheng.study</groupId>
        <artifactId>idl-tools</artifactId>
        <version>1.0</version>
    </dependency>

### 参考文档
1. [http://www.mkyong.com/maven/how-to-include-library-manully-into-maven-local-repository/](http://www.mkyong.com/maven/how-to-include-library-manully-into-maven-local-repository/)

