---
layout: post
title: 如何使用tomcat高效调试
---


### 1. 使用maven tomcat插件快速部署和启动web工程：

    maven tomcat:run

如果要调试：可以先在命令行设置如下环境变量：

    set MAVEN_OPTS=-Xrunjdwp:transport=dt_socket,server=y,suspend=n,address=9000

然后在eclipse中就可以链接9000端口（可随意设定）进行调试了。


### 2. 使用Eclipse Server插件调试

如果在J2EE版本的Eclipse中，那么也可以使用内置的Servers插件进行。不过有些小技巧。

首先，通过Window==>Show Views==>Servers调出Servers视图。然后添加你的本地web服务器目录（这里选择tomcat为例）。

配置完成之后会生成一个叫做Servers的Project。打开项目，里面其实是tomcat的配置文件，这样修改tomat配置不会影响到你本地tomcat，而tomcat代码是共享的。

然后一切就像部署到本地tomcat一样处理了。打开server.xml文件，在里面配置要添加的工程：

    <?xml version="1.0" encoding="UTF-8"?>
    
    <Server port="8005" shutdown="SHUTDOWN">
      <Service name="Catalina">
        <Connector connectionTimeout="20000" port="8080" protocol="HTTP/1.1" redirectPort="8443"/>
        <!-- Define an AJP 1.3 Connector on port 8009 -->
        <Connector port="8009" protocol="AJP/1.3" redirectPort="8443"/>
        
        <Engine defaultHost="localhost" name="Catalina">
          <Realm className="org.apache.catalina.realm.UserDatabaseRealm" resourceName="UserDatabase"/>
          
          <Host appBase="webapps" autoDeploy="true" name="localhost" unpackWARs="true" xmlNamespaceAware="false" xmlValidation="false">
            <Context docBase="C:\Users\arganzheng\workspace\myWebProj\WebContent" path="/xxxx" />
            
          </Host>
          
        </Engine>
      </Service>
    </Server>
    
这里使用了Context配置项，配置了一个外部webapp，tomcat启动的时候就好加载它。

    <Context docBase="C:\Users\arganzheng\workspace\myWebProj\WebContent" path="/xxxx" />

注意：如果你是通过Eclipse的Web/Dynamic Web Project向导生成的web工程，需要修改一个它的buildPath中的Default output folder为WebContent\WEB-INF\classes。这样他就是一个标准的war包目录了。


### 3. 使用独立外部tomcat调试

如果是自己启动tomcat，那么可以直接使用jpda调试：

    $./catalina.sh jpda run
