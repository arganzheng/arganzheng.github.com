---
title: 使用java web start网络启动应用程序
layout: post
---


昨天刚把swing应用打包成一个可以执行的jar包。liuhe说可以更进一步搞个java web start网络启动。于是谷歌一下，发现挺简单的：

主要两个步骤：

1) 使用keytool生成一个keystore文件。然后使用该keystore用jarsigner对你的jar包进行签名。

2) 写个jnlp文件

     <?xml version="1.0" encoding="UTF-8"?>
     <jnlp codebase="http://10.6.223.106:9200/" name="OpenApiTestTool" href="http://10.6.223.106:9200/swing.jnlp">
          <information>
               <title>Open Api Test</title>
               <vendor>OpenApi</vendor>
               <offline-allowed />
               <description>An Application To Test Open Api!</description>
          </information>
          <application-desc main-class="me.arganzheng.study.api.ui.ApiTestUI" />
          <security>
               <all-permissions />
          </security>
          <resources>
               <j2se version="1.4+" />
               <jar href="me.arganzheng.study.api.util.jar" />
          </resources>
     </jnlp>

放在某个web工程的目录下，直接访问jnlp路径就可以了。

参考文章： 

1. [jnlp（Java网络加载协议）原来很简单](http://blog.csdn.net/zmxj/article/details/297649)
2. [How to deploy SWT Applications using Java Web Start](http://www.eclipse.org/swt/jws/)

