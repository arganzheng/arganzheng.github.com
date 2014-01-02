---
title: 如何让tomcat不解压你的war包
layout: post
---


默认情况下，tomcat会解压你的war包到webapps目录下，这样每次重新发布war包还要记得先把原来解压的war目录删除才会生效，麻烦又容易忘记。总以为发布没有生效。解决方案是增加这个一个配置：

    <Context  docBase="/data/b2b2c/japp/api/meta.war" unpackWAR="false"  privileged="true" antiResourceLocking="false" reloadable="false" antiJARLocking="false" allowLinking="false" >

关键是这个参数：`unpackWAR="false"`，务必配置成false！

TIPS: 事实上，tomcat还是会解压你的war包，在work目录下。但是就没有缓存问题了。

--EOF--
