---
title: 如何让java程序优先使用自定义的DNS nameserver
layout: post
---


背景
----

以上省略1w字。基于这些原因，我们想指定我们信任的DNS name server。比如Google著名的8.8.8.8。通过BGP方式，可以让一个IP对应N台服务器，那么在权威DNS服务器够多够分散的情况下，绕过ISP提供的LocalDNS可以最大化的控制域名解析。


解决方案
--------

JDK默认的DNS查询方式是具体参见 [Understanding host name resolution and DNS behavior in Java](http://www.myhowto.org/java/42-understanding-host-name-resolution-and-dns-behavior-in-java/)。


当然，想要实现自己的DNS查询逻辑并不是一件麻烦的事情，麻烦的在于怎样对客户端透明。特别是在不需要修改原有代码的前提下。这个非常重要。比如原来我们请求一个HTTP接口，不管是使用Apache的HttpClient还是使用JDK的HttpURLConnection。都是不需要涉及到DNS查询的，我们只需要传递请求的URL，而这个URL一般都是域名格式，然后HttpClient和HttpURLConnection会调用InetAddress将域名解析成IP地址，最后发出HTTP请求。

所以如果我们不将逻辑植入InetAddress的域名解析代码中，那么就意味着所有针对域名的请求都需要拆分为两步：

1、将域名转换为IP地址，然后替换原来的URL中的域名
	
	String dottedQuadIpAddress = MyNameServiceProvider.getByName( "blog.arganzheng.me" ).getHostAddress();

	String url = url.replace(...)

2、对IP地址发送请求
	
	URL url = new URL(url);
	HttpURLConnection conn = (HttpURLConnection) url.openConnection();
	...

所有的代码都需要这么改动，可以想象工作量有多大。显然是不可接受的，还是要想办法怎样在不改动代码的情况下实现指定nameserver。

可能的方案有如下几种：

1. 遵循默认的实现机制，修改/etc/resolv.conf下的nameserver配置。但是这个在安卓客户端显示是不可能的，何况这个配置是启动时加载的，并不能动态修改生效。
2. 拦截InetAddress的getHostByName方法，植入我们的逻辑。也就是我们常说的AOP。
3. 覆盖InetAddress，然后通过classLoader机制，让其优先加载你的这个覆盖类。

AOP也好，重载也好，关键在于让JDK加载你的类。

开源的java DNS client

1. [dnsjava](http://www.xbill.org/dnsjava/)
2. [dns](https://github.com/jonahb/dns)


JDK本身提供了一种替换实现的方式，JNDI查询

	//Override system DNS setting with Google free DNS server
	System.setProperty("sun.net.spi.nameservice.nameservers", "8.8.8.8");
	System.setProperty("sun.net.spi.nameservice.provider.1", "dns,sun");


比如安全也是这种方式：[Web Services Security - Part 2: Encryption](http://www.javaranch.com/journal/2008/10/Journal200810.jsp#a5)

这篇文章 [Local Managed DNS (Java)](http://rkuzmik.blogspot.com/2006/08/local-managed-dns-java_11.html) 详细的介绍了如何让JVM优先使用你自定义的nameserver provider。并且，由于这篇文章成文于JDK7之前，所以需要在自定义nameserver provider中实现完整的DNS解析，而不能委托给下一个nameserver provider。所以文章引入了一个开源的 Java DNS client —— [dnsjava](http://www.xbill.org/dnsjava/)。但是在JDK7之后，nameservice provider是链状结构了，如果第一个provider解析失败，后面的providers将依次尝试解析。

> **sun.net.spi.nameservice.provider.<n>=<default|dns,sun|...>**
> Specifies the name service provider that you can use. By default, Java will use the system configured name lookup mechanism, such as file, nis, etc. You can specify your own by setting this option. <n> takes the value of a positive number, it indicates the precedence order with a small number takes higher precendence over a bigger number. Aside from the default provider, the JDK includes a DNS provider named "dns,sun".
> Prior to JDK 7, the first provider that was successfully loaded was used. In JDK 7, providers are chained, which means that if a lookup on a provider fails, the next provider in the list is consulted to resolve the name.

询问了一下安卓客户端的同学，JDK编译版本是1.6，但是运行时的版本呢，是不是跟安卓版本有一个对应关系呢？结果令我大吃一惊——安卓上JDK的关系就如安卓跟Linux的关系一样，它修改了OpenJDK，并没有完全准许JDK的版本机制。比如最新的安卓是支持1.7的大部分功能，但又不是所有的。所以很难说它支持哪个版本的JDK。甚至它的行为跟PC的JDK/JRE行为都不一致。所以要查看源码或者测试才能得出结论。



参考文章
--------

1. [Understanding host name resolution and DNS behavior in Java](http://www.myhowto.org/java/42-understanding-host-name-resolution-and-dns-behavior-in-java/)
2. [dnsjava](http://www.xbill.org/dnsjava/) 看一下它的 [README](http://www.xbill.org/dnsjava/dnsjava-current/README)。源代码 [jdnssec](http://svn.verisignlabs.com/jdnssec/dnsjava/trunk/org/xbill/DNS/
)
3. [Local Managed DNS (Java)](http://rkuzmik.blogspot.com/2006/08/local-managed-dns-java_11.html) 被墙了，但是可以从Google Cache中获取到历史页面。
4. [Ping & DNS](http://www.ulfdittmer.com/android/#pingdns)
5. [Networking Properties](http://docs.oracle.com/javase/1.5.0/docs/guide/net/properties.html)
6. [Clean DNS server in JVM](http://stackoverflow.com/questions/17420620/clean-dns-server-in-jvm)
7. [How to set a custom DNS server with Java System properties](http://slackhacker.com/2010/07/21/how-to-set-a-custom-dns-server-with-java-system-properties/)
8. [全局精确流量调度新思路-HttpDNS服务详解](http://mp.weixin.qq.com/s?__biz=MzA3ODgyNzcwMw==&mid=201837080&idx=1&sn=b2a152b84df1c7dbd294ea66037cf262&scene=2&from=timeline&isappinstalled=0#rd)