---
title: 如何把一个jar包install到maven本地仓库
layout: post
catalog: true
---


使用mvn install:install-file命令：

    $ mvn install:install-file -Dfile=idl-tools.jar -DgroupId=me.arganzheng.study -DartifactId=idl-tools -Dversion=1.0 -Dpackaging=jar
  
然后就可以在依赖到地方引用了
  
    <dependency>
        <groupId>me.arganzheng.study</groupId>
        <artifactId>idl-tools</artifactId>
        <version>1.0</version>
    </dependency>


---

### 补记: 如何把一个jar包deploy到maven私服仓库

跟`install-file`命令非常类似：

    mvn deploy:deploy-file -DgroupId=<group-id> \
      -DartifactId=<artifact-id> \
      -Dversion=<version> \
      -Dpackaging=<type-of-packaging> \
      -Dfile=<path-to-file> \
      -DrepositoryId=<id-to-map-on-server-section-of-settings.xml> \
      -Durl=<url-of-the-repository-to-deploy>

例如：

    mvn deploy:deploy-file -DgroupId=com.argan.internet \
      -DartifactId=privilege-client \
      -Dversion=1.0.0.20180705-RELEASE \
      -Dpackaging=jar \
      -Dfile=privilege-client-1.0.0.20180705-RELEASE.jar \
      -DrepositoryId=ai-releases \
      -Durl=http://10.21.7.1:8081/repository/ai-releases/

如果要把原来的 pom 也 deploy 上去，那么可以用`-DpomFile`选项： 

    mvn deploy:deploy-file -DgroupId=com.argan.internet \
      -DartifactId=privilege-client \
      -Dversion=1.0.0.20180705-RELEASE \
      -Dpackaging=jar \
      -Dfile=privilege-client-1.0.0.20180705-RELEASE.jar \
      -DpomFile=privilege-client-1.0.0.20180705-RELEASE.pom \
      -DgeneratePom=false \
      -DrepositoryId=ai-releases \
      -Durl=http://10.21.7.1:8081/repository/ai-releases/

但是要注意，相关的依赖也要依次 deloy 上去。。这是一个非常费时费力的操作。。

如果要 deploy source jar，那么需要将 packaging 设置为 `java-source`，并且 `generatePom`要设置为false:

    mvn deploy:deploy-file -DgroupId=com.argan.internet \
      -DartifactId=privilege-client \
      -Dversion=1.0.0.20180705-RELEASE \
      -Dpackaging=java-source \
      -DgeneratePom=false \
      -Dfile=privilege-client-1.0.0.20180705-RELEASE-sources.jar \
      -DrepositoryId=ai-releases \
      -Durl=http://10.21.7.1:8081/repository/ai-releases/


### 参考文档

1. [http://www.mkyong.com/maven/how-to-include-library-manully-into-maven-local-repository/](http://www.mkyong.com/maven/how-to-include-library-manully-into-maven-local-repository/)
2. [Guide to deploying 3rd party JARs to remote repository](https://maven.apache.org/guides/mini/guide-3rd-party-jars-remote.html)

