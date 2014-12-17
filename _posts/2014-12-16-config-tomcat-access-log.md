---
title: 配置tomcat的access_log
layout: post
---

一般应用都是跑在nginx后面，nginx本身就有access_log了，不需要重复配置tomcat的access_log。不过在测试环境，可能是直接访问tomcat。特别是一些别有用心的程序，比如安全扫描程序。

今天就发现一个tomcat应用有很多莫名其妙的请求。看了一下nginx的access_log，没有发现请求日志，但是应用配置的log4j日志确实很多异常的请求日志。于是配置了一下tomcat的access_log，打印一下请求的IP。谷歌了一下，发现配置蛮简单的：

    <!-- Define the default virtual host Note: XML Schema validation will not work with Xerces 2.2.  -->
  	<Host name="localhost"  appBase="webapps" unpackWARs="true" autoDeploy="true" xmlValidation="false" xmlNamespaceAware="false">

	    <!-- Access log processes all example. Documentation at: /docs/config/valve.html -->
	    <Valve className="org.apache.catalina.valves.AccessLogValve" directory="logs"  
	           prefix="localhost_access_log." suffix=".txt" pattern="common" resolveHosts="false"/>
  	</Host>

然后在logs目录下，就有一个localhost_access_log.2014-12-16.txt的access_log文件，里面的日志格式如下：

	10.212.114.42 - - [16/Dec/2014:17:57:23 +0800] "GET /notexist_path.bootstrap HTTP/1.0" 404 -
	10.212.114.42 - - [16/Dec/2014:17:57:23 +0800] "GET /ap-application.bootstrap HTTP/1.0" 404 -
	10.212.114.43 - - [16/Dec/2014:17:57:24 +0800] "GET /admin/notexist_path.bootstrap HTTP/1.0" 404 -
	10.212.114.43 - - [16/Dec/2014:17:57:24 +0800] "GET /admin/ap-application.bootstrap HTTP/1.0" 404 -
	10.212.114.43 - - [16/Dec/2014:17:57:24 +0800] "GET /module/action/param1/notexist_path HTTP/1.0" 404 -
	10.212.114.43 - - [16/Dec/2014:17:57:24 +0800] "GET /module/action/param1/$(@phpinfo()) HTTP/1.0" 404 -
	10.212.114.42 - - [16/Dec/2014:17:57:26 +0800] "GET /resin-doc/resource/tutorial/jndi-appconfig/notexist_path HTTP/1.0" 404 -

	