---
title: 如何构建maven私有仓库
layout: post
catalog: true
---


私有仓库，也称之为私服，它是一种特殊的远程仓库。它是架设在局域网内的仓库服务，私服代理广域网上的远程仓库，供局域网内的maven用户使用。当Maven需要下载构建的时候，它从私服请求，如果私服不存在该构件，则会从外部的远程仓库下载，缓存在私服上之后，再为maven的下载请求提供服务。此外，一些无法从远程仓库下载到的构件也能从本地上传到私服上供大家使用。

![私服的用途](/img/in-post/maven-private-repository.jpg)


通过建立自己的私服，就可以带来如下几个好处：

1. 加速Maven构建
2. 节省外网带宽
3. 部署第三方构件
4. 节省中央maven仓库的负担
5. 提高稳定性，增强控制

使用Nexus搭建私服
----------------

Sonatype Nexus是当前最流行的Maven仓库管理软件。具体搭建非常简单，网上资料很多，这里不赘述了。


配置Maven使用私服
---------------

### 1. 远程仓库的配置

很多情况下，默认中央仓库无法满足项目的需求，可能项目需要的构件存在另外一个远程仓库中，如JBOSS Maven仓库。这时，我们可以在POM如下配置仓库。

	<project>
	    ...
	    <repositories>
	        <repository>
	            <id>jboss</id>
	            <name>JBoss Repository</name>
	  			<url>http://repository.jboss.com/maven2/</url>
	            <releases>
                    <enabled>true</enabled>
                </releases>
                <snapshots>
                    <enabled>false</enabled>
                </snapshots>
                    <layout>default</layout>
                <repository>
        </repositories>
	</project>

在repositories元素下，可以使用repository子元素 一个或者多个远程仓库。任何一个仓库声明的id必须是唯一的，该配置中的url指向了仓库的地址，一般采用http协议，releases元素的enabled的值为true，snapshots元素的enabled的值为false，则说明，只会下载远程仓库发布版本的构件而不会下载快照版本的构件。对于releases和snapshots，它们还包含了另外两个元素updatePolicy和checkSumPolicy。

	<releases>
        <updatePolicy>daily</ updatePolicy>
        <enabled>true</enabled>
        <checkSumPolicy>ignore</checkSumPolicy>
	</releases>  

updatePolicy为从Jboss仓库检查的频率，值可以为daily（每天）、never（从不）、always（每次构建都检查）、Interval(每隔多少分钟检查一次)。

checkSumPolicy用来配置Maven检查检验和文件的策略，当构件被部署到Maven仓库时，会同时部署对应的检验和文件。在下载构件的时候，Maven会验证检验和文件。如果验证检验失败，当checkSumPolicy为warn时，Maven会在执行构件时输出警告信息，其他可用的值有：fail当验证检验失败，就让Maven构建失败；ignore完全忽略校验和失误。


上面的例子是配置到了JBOSS远程仓库，如果想要配置到我们的私有仓库，那么只需要url换一下就可以了。而且为了避免每个项目的POM都要配置这么一大坨，可以把配置放在settings.xml文件中：

	<settings>
		
		...

		<mirrors>
		    <mirror>
		      <!--This is used to direct the public snapshots repo in the 
		          profile below over to a different nexus group -->
		      <id>nexus-public-snapshots</id>
		      <mirrorOf>public-snapshots</mirrorOf>
		      <url>http://maven.scm.baidu.com:8081/nexus/content/groups/public-snapshots</url>
		    </mirror>
		    <mirror>
		      <!--This sends everything else to /public -->
		      <id>nexus</id>
		      <mirrorOf>*</mirrorOf>
		      <url>http://maven.scm.baidu.com:8081/nexus/content/groups/public</url>
		    </mirror>
		  </mirrors>

	    <profile>
	      <!--this profile will allow snapshots to be searched when activated-->
	      <id>public-snapshots</id>
	      <repositories>
	        <repository>
	          <id>public-snapshots</id>
	          <url>http://public-snapshots</url>
	          <releases><enabled>false</enabled></releases>
	          <snapshots><enabled>true</enabled></snapshots>
	        </repository>
	      </repositories>
	     <pluginRepositories>
	        <pluginRepository>
	          <id>public-snapshots</id>
	          <url>http://public-snapshots</url>
	          <releases><enabled>false</enabled></releases>
	          <snapshots><enabled>true</enabled></snapshots>
	        </pluginRepository>
	      </pluginRepositories>
	    </profile>
	  </profiles>
	  <activeProfiles>
	    <activeProfile>public-snapshots</activeProfile>
	  </activeProfiles>
	</settings>

注意：这里的url使用了ID引用，引用到了前面的mirro定义。要知道私服的一个非常重要的作用就是作为中央仓库的镜像。


### 2. 远程仓库的验证

大部分仓库都无需认证就可以访问，但有时处于安全方面的考虑，我们需要提供认证信息才能访问远程仓库（主要是为了将构件部署到仓库中去）。为了防止非法的仓库访问，管理员往往会提供一组用户名和密码，这时，为了能访问仓库，就需要在setting.xml文件中配置：

	<settings>
	    
	    ...
        
        <servers>
		    <server>
		      <id>Baidu_Local</id>
		      <username>xx</username>
		      <password>xxxx</password>
		    </server>
		    <server>
		      <id>Baidu_Local_Snapshots</id>
		      <username>xx</username>
		      <password>xxxx</password>
		    </server>
  	</servers>

	</settings>

setting文件中的server的id必须与pom文件的需要认证的repository元素的id一致，正是此id将认证信息和仓库配置联系在了一起。


### 3. 将构件部署至远程仓库

编辑项目的pom.xml文件

	<project>

        ...

        <distributionManagement>
	        <repository>
	            <id>Baidu_Local</id>
	            <url>http://maven.scm.baidu.com:8081/nexus/content/repositories/Baidu_Local</url>
	        </repository>
	        <snapshotRepository>
	            <id>Baidu_Local_Snapshots</id>
	            <url>http://maven.scm.baidu.com:8081/nexus/content/repositories/Baidu_Local_Snapshots</url>
	        </snapshotRepository>
	    </distributionManagement>

	</project>

往远程仓库部署构件的时候，往往需要认证。简而言之，就是需要在setting文件中配置server元素，其id与仓库的id匹配，并配置正确的认证信息。


### 4. 镜像

如果仓库x可以提供仓库y存储的所有内容，那我们可以说x是y的镜像，典型的事例是，很多国外的服务在中国都有中国的镜像，这样的目的是能够提供更快的服务。

	<settings>
        
        ...
        
        <mirrors>
            <mirror>
                <id>maven.net.cn</id>
                <name>one of the central mirror in China</name>
                <url>http://maven.net.cn/content/groups/public</url>
                <mirrorOf>central</mirrorOf>
            </mirror>
	    </mirrors>

	</settings>

以上就是配置的就是中央仓库在中国的镜像。

镜像更为常见的是结合私服的使用，由于私服可以代理任何外部的公共仓库，因此，对于组织内部的Maven用户来说，使用一个私服地址，就相当于使用了所有需要的外部仓库，这可以将配置集中到私服，从而简化Maven本身的配置，在这种情况下，任何需要的构件都可以从私服获得，私服就是所有仓库的镜像，这时可以如此配置：

	<settings>
	    ...
	    <mirrors>
            <mirror>
                <id>internal repository</id>
                <name>internal repository manager</name>
                <url>http://192.168.2.11:1521/maven2/</url>
                <mirrorOf>*</mirrorOf>
            </mirror>
        </mirrors>
	</settings>

mirrorOf元素的值为*代表该配置是所有仓库的镜像，任何对于远程仓库的请求都会被转至http://192.168.2.11:1521/maven2/。

Maven还支持其他高级的镜像配置：

* <mirrorOf>*</mirrorOf>
* <mirrorOf>external:*</mirrorOf>匹配所有不在本机上的远程仓库
* <mirrorOf>repo1,repo2</mirrorOf>匹配仓库repo1,repo2
* <mirrorOf>*,!repo2</mirrorOf>匹配所有仓库，repo2除外

需要注意的是，由于镜像仓库完全屏蔽掉被镜像仓库，当镜像仓库不稳定或者停止服务的时候， Maven仍将无法访问被镜像仓库，因而无法下载构件。

参考文章
-------

1. Maven实战，作者：徐晓斌



