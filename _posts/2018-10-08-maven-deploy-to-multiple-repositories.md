---
title: maven如何deploy到多个repositories
layout: post
tags: [maven]
catalog: true
---


背景
---

公司的私服部署在测试环境，跟线上的环境在网络上是物理隔离的。导致在线上无法使用 maven 编译，因为拉取不到测试环境的私服。只能在线下进行打包，然后通过终端上传到线上机器，问题是打出来的包往往有几十上百 M，上传速度超级慢，还容易断开出错。


解决方案
-------

因为网络是物理隔离的，所以只能在线上部署一个maven私服了。部署过程很简单，这里不赘述。但是部署完成之后有一个问题，就是需要将二方库也同时 deploy 到线上环境的私服。并且为了保证能够给其他团队使用，还需要保证原来的 deploy 到测试环境的私服照常。

经过调研，最后决定采用 maven 的profile 机制来保证不同的环境deploy到不同的repositories。

具体如下：

首先修改 `.m2`的 settings.xml 文件，配置多个 profile：

```xml 
<profiles>
  <profile>
    <id>argan</id>
    
    <properties>
      <releases.id>nexus-releases</releases.id>
      <releases.name>argan Releases Repository</releases.name>
      <releases.url>http://nexus.argan.xyz/nexus/content/repositories/releases</releases.url>

      <snapshots.id>nexus-snapshots</snapshots.id>
      <snapshots.name>argan Snapshots Repository</snapshots.name>
      <snapshots.url>http://nexus.argan.xyz/nexus/content/repositories/snapshots</snapshots.url>
    </properties>

    <repositories>
      <repository>
          <id>nexus</id>
          <name>Nexus</name>
          <url>http://nexus.argan.xyz/nexus/content/groups/public/</url>
          <releases>
              <enabled>true</enabled>
              <updatePolicy>always</updatePolicy>  
              <checksumPolicy>warn</checksumPolicy>  
          </releases>
          <snapshots>
              <enabled>true</enabled>  
              <updatePolicy>always</updatePolicy>  
              <checksumPolicy>warn</checksumPolicy>  
          </snapshots>
      </repository>
    </repositories>
       
    <pluginRepositories>
      <pluginRepository>
          <id>nexus</id>
          <name>Nexus</name>
          <url>http://nexus.argan.xyz/nexus/content/groups/public/</url>
          <releases>
              <enabled>true</enabled>
              <updatePolicy>always</updatePolicy>  
              <checksumPolicy>warn</checksumPolicy>  
          </releases>
          <snapshots>
              <enabled>true</enabled>  
              <updatePolicy>always</updatePolicy>  
              <checksumPolicy>warn</checksumPolicy>  
          </snapshots>
      </pluginRepository>
    </pluginRepositories>
  </profile>

  <profile>
    <id>ai</id>
    
    <properties>
      <releases.id>ai-releases</releases.id>
      <releases.name>AI Releases Repository</releases.name>
      <releases.url>http://10.21.7.1:8081/repository/ai-releases/</releases.url>

      <snapshots.id>ai-snapshots</snapshots.id>
      <snapshots.name>AI Snapshots Repository</snapshots.name>
      <snapshots.url>http://10.21.7.1:8081/repository/ai-snapshots</snapshots.url>
    </properties>

    <repositories>
          <repository>
              <id>ai-nexus</id>
              <name>AI-Nexus</name>
              <url>http://10.21.7.1:8081/repository/ai-public/</url>
              <releases>
                  <enabled>true</enabled>
                  <updatePolicy>always</updatePolicy>  
                  <checksumPolicy>warn</checksumPolicy>  
              </releases>
              <snapshots>
                  <enabled>true</enabled>  
                  <updatePolicy>always</updatePolicy>  
                  <checksumPolicy>warn</checksumPolicy>  
              </snapshots>
          </repository>
      </repositories>
       
      <pluginRepositories>
          <pluginRepository>
              <id>ai-nexus</id>
              <name>AI-Nexus</name>
              <url>http://10.21.7.1:8081/repository/ai-public/</url>
              <releases>
                  <enabled>true</enabled>
                  <updatePolicy>always</updatePolicy>  
                  <checksumPolicy>warn</checksumPolicy>  
              </releases>
              <snapshots>
                  <enabled>true</enabled>  
                  <updatePolicy>always</updatePolicy>  
                  <checksumPolicy>warn</checksumPolicy>  
              </snapshots>
          </pluginRepository>
      </pluginRepositories>
  </profile>
</profiles>
```

增加了一个 ai 的 profile，但是默认active profile还是原来的 argan。

```xml
<activeProfiles>
  <activeProfile>argan</activeProfile>
</activeProfiles>
```

然后在项目的 `pom.xml` 文件中 改成使用配置项：

```xml
<!-- deploy repository -->
<distributionManagement>
  <repository>
    <id>${releases.id}</id>
    <name>${releases.name}</name>
    <url>${releases.url}</url>
  </repository>
  <snapshotRepository>
    <id>${snapshots.id}</id>
    <name>${snapshots.name}</name>
    <url>${snapshots.url}</url>
  </snapshotRepository>
</distributionManagement>
```

这样，当需要 deploy 到 线上环境的私服（即ai-nexus）的时候，只需要指定`-Pai`就可以了：

```shell
mvn clean deploy -Pai
```

不指定的时候还是会 deploy 到测试环境的私服。

最终的 settings.xml 文件配置如下：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<settings xmlns="http://maven.apache.org/SETTINGS/1.0.0"
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
          xsi:schemaLocation="http://maven.apache.org/SETTINGS/1.0.0 http://maven.apache.org/xsd/settings-1.0.0.xsd"> 
  <servers>
      <!-- VIVO Nexus -->
      <server>  
        <id>nexus-releases</id>  
        <username>admin</username>  
        <password>admin123456</password>  
      </server>  
      <server>  
        <id>nexus-snapshots</id>  
        <username>admin</username>  
        <password>admin123456</password>  
      </server>  
      <!-- AI Nexus-->
      <server>  
        <id>ai-releases</id>  
        <username>admin</username>  
        <password>admin123</password>  
      </server>  
      <server>  
        <id>ai-snapshots</id>  
        <username>admin</username>  
        <password>admin123</password>  
      </server>   
  </servers>
   
  <mirrors>
    <mirror>
        <id>ai-nexus</id>
        <name>AI-Nexus</name>
        <url>http://10.21.7.1:8081/repository/ai-public/</url>
        <mirrorOf>*</mirrorOf>
    </mirror> 
    <!-- 为了避免线上环境拉取测试环境私服失败，mirror 统一走线上环境私服。
    <mirror>
          <id>nexus</id>
          <name>Nexus</name>
          <url>http://nexus.vivo.xyz/nexus/content/groups/public/</url>
          <mirrorOf>*</mirrorOf>
    </mirror> 
     -->
  </mirrors>
   
   
  <profiles>
    <profile>
      <id>vivo</id>
      
      <properties>
        <releases.id>nexus-releases</releases.id>
        <releases.name>VIVO Releases Repository</releases.name>
        <releases.url>http://nexus.vivo.xyz/nexus/content/repositories/releases</releases.url>

        <snapshots.id>nexus-snapshots</snapshots.id>
        <snapshots.name>VIVO Snapshots Repository</snapshots.name>
        <snapshots.url>http://nexus.vivo.xyz/nexus/content/repositories/snapshots</snapshots.url>
      </properties>

      <repositories>
        <repository>
            <id>nexus</id>
            <name>Nexus</name>
            <url>http://nexus.vivo.xyz/nexus/content/groups/public/</url>
            <releases>
                <enabled>true</enabled>
                <updatePolicy>always</updatePolicy>  
                <checksumPolicy>warn</checksumPolicy>  
            </releases>
            <snapshots>
                <enabled>true</enabled>  
                <updatePolicy>always</updatePolicy>  
                <checksumPolicy>warn</checksumPolicy>  
            </snapshots>
        </repository>
      </repositories>
         
      <pluginRepositories>
        <pluginRepository>
            <id>nexus</id>
            <name>Nexus</name>
            <url>http://nexus.vivo.xyz/nexus/content/groups/public/</url>
            <releases>
                <enabled>true</enabled>
                <updatePolicy>always</updatePolicy>  
                <checksumPolicy>warn</checksumPolicy>  
            </releases>
            <snapshots>
                <enabled>true</enabled>  
                <updatePolicy>always</updatePolicy>  
                <checksumPolicy>warn</checksumPolicy>  
            </snapshots>
        </pluginRepository>
      </pluginRepositories>
    </profile>

    <profile>
      <id>ai</id>
      
      <properties>
        <releases.id>ai-releases</releases.id>
        <releases.name>AI Releases Repository</releases.name>
        <releases.url>http://10.21.7.1:8081/repository/ai-releases/</releases.url>

        <snapshots.id>ai-snapshots</snapshots.id>
        <snapshots.name>AI Snapshots Repository</snapshots.name>
        <snapshots.url>http://10.21.7.1:8081/repository/ai-snapshots</snapshots.url>
      </properties>

      <repositories>
            <repository>
                <id>ai-nexus</id>
                <name>AI-Nexus</name>
                <url>http://10.21.7.1:8081/repository/ai-public/</url>
                <releases>
                    <enabled>true</enabled>
                    <updatePolicy>always</updatePolicy>  
                    <checksumPolicy>warn</checksumPolicy>  
                </releases>
                <snapshots>
                    <enabled>true</enabled>  
                    <updatePolicy>always</updatePolicy>  
                    <checksumPolicy>warn</checksumPolicy>  
                </snapshots>
            </repository>
        </repositories>
         
        <pluginRepositories>
            <pluginRepository>
                <id>ai-nexus</id>
                <name>AI-Nexus</name>
                <url>http://10.21.7.1:8081/repository/ai-public/</url>
                <releases>
                    <enabled>true</enabled>
                    <updatePolicy>always</updatePolicy>  
                    <checksumPolicy>warn</checksumPolicy>  
                </releases>
                <snapshots>
                    <enabled>true</enabled>  
                    <updatePolicy>always</updatePolicy>  
                    <checksumPolicy>warn</checksumPolicy>  
                </snapshots>
            </pluginRepository>
        </pluginRepositories>
    </profile>

  </profiles>
  
  <activeProfiles>
    <activeProfile>vivo</activeProfile>
  </activeProfiles>
   
</settings>
```

**说明**

1、网上也找过一些资料，可以同时 deploy 到多个仓库（参见参考文档），不过因为两个网络是不通的，所以同时 deploy 基本上有一个会失败。所以还是让用户自己选择在什么情况下 deploy 到哪里去。
2、为了避免线上环境拉取测试环境私服失败，mirror 统一走线上环境私服。


参考文档
-------

1. [Maven Deploy to Nexus](https://www.baeldung.com/maven-deploy-nexus)
2. [deploy artifacts to multiple maven repositories](https://www.experts-exchange.com/questions/27280297/deploy-artifacts-to-multiple-maven-repositories.html)
