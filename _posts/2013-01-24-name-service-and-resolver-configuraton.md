---
layout: post
title: 域名解析过程及其相关配置
---


## /etc/host.conf

[The host.conf File](http://www.tldp.org/LDP/nag/node82.html#SECTION008110000)

>
The central file that controls your resolver setup is host.conf. It resides in /etc and tells the resolver which services to use, and in what order.
Options in host.conf must occur on separate lines. Fields may be separated by white space (spaces or tabs). A hash sign (#) introduces a comment that extends to the next newline.

这个文件是域名解析的入口配置文件， 它决定了应该使用哪些域名解析服务以及他们的顺序，从配置文件上来说就是先走/etc/resolv.conf还是先走/etc/hosts。

一个简单示例：
>
A sample file for vlager is shown below:
>
    # /etc/host.conf
    # We have named running, but no NIS (yet)
    order   bind hosts
    # Allow multiple addrs
    multi   on
    # Guard against spoof attempts
    nospoof on
    # Trim local domain (not really necessary).
    trim    vbrew.com.


## /etc/resolv.conf

网上关于这方面的结算少之又少，这篇文章 [Configuring Name Server Lookups--resolv.conf](http://www.tldp.org/LDP/nag/node84.html) 是唯一一篇让我觉得满意的，下面直接引用了，并且加上注解和说明。

>
When configuring the resolver library to use the BIND name service for host lookups, you also have to tell it which name servers to use. There is a separate file for this, called resolv.conf. If this file does not exist or is empty, the resolver assumes the name server is on your local host.
If you run a name server on your local host, you have to set it up separately, as will be explained in the following section. If your are on a local network and have the opportunity to use an existing nameserver, this should always be preferred.


>
The most important option in resolv.conf is `nameserver`, which gives the IP-address of a name server to use. If you specify several name servers by giving the nameserver option several times, they are tried in the order given. You should therefore put the most reliable server first. Currently, up to three name servers are supported.

注：查询算法是这样子的：[/etc/resolv.conf](http://manpages.ubuntu.com/manpages/natty/man5/resolv.conf.5.html)
> The algorithm used is to try a name server, and if the  query  times out, try the next, until out of name servers, then repeat trying all the name servers until a maximum number of retries are made.

需要注意的是，如果nameserver不是time out（一般就是宕机了或者配置错了），而是针对给定的host返回查询不到的结果，那么算法是不会继续查询下一个nameserver，而是直接返回域名解析错误了。也就是说在任何时候最多只有一台NameServer是起作用的。这个很重要，也是最容易出问题的。

>
If no nameserver option is given, the resolver attempts to connect to the name server on the local host.

>
Two other options, `domain` and `search` deal with default domains that are tacked onto a hostname if BIND fails to resolve it with the first query. The search option specifies a list of domain names to be tried. The list items are separated by spaces or tabs.

>
If no search option is given, a default search list is constructed from the local domain name by using the domain name itself, plus all parent domains up to the root. The local domain name may be given using the domain statement; if none is given, the resolver obtains it through the getdomainname(2) system call.

search和domain配置项有点类似于no host entry found exception handler。就是当nameserver返回找不到域名解析的时候，BIND会尝试使用这两个配置项再做一下尝试。但是它们提供的是主域名（顶级域名），用于方便使用 short name 查询。search的优先级高于domain，如果search没有配置，则会采用domain的配置。如果domain也没有配置，会使用`getdomainname(2)`系统调用获取。

>
If this sounds confusing to you, consider this sample resolv.conf file for the Virtual Brewery:
>
    # /etc/resolv.conf
    # Our domain
    domain         vbrew.com
    #
    # We use vlager as central nameserver:
    nameserver     191.72.1.1
>
When resolving the name vale, the resolver would look up vale, and failing this, vale.vbrew.com, and vale.com.

这个一个例子，不过用了domain没有用search，但是domain和search是互斥的，因为只有一个生效。

## 相关命令

1. ping blog.arganzheng.me
2. nslookup blog.arganzheng.me 8.8.8.8 
3. dig blog.arganzheng.me
